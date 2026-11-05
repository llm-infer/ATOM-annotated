# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import pickle
import queue
import threading
import time
from contextlib import ExitStack

import torch
import zmq

from atom.config import Config, ParallelConfig
from atom.kv_transfer.disaggregation import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
from atom.metrics.scheduler import SchedulerMetrics
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.dynamic_chunking import (
    DYNAMIC_CHUNKING_POLL_STEPS,
    ChunkSizePredictor,
)
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.prefix_hash import attach_shipped_chains
from atom.model_engine.scheduler import DecodeScheduler, PrefillScheduler, Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    get_exit_sequence,
    new_block_table,
)
from atom.model_engine.state_runtime import StateRuntime
from atom.utils import (
    engine_process_name,
    envs,
    init_exit_handler,
    make_zmq_socket,
    set_process_title,
)
from atom.utils.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)
from atom.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
    tune_gc,
    unfreeze_gc_heap,
)

# 本文件：EngineCore——每个DP rank一个的引擎子进程，ATOM的"主循环"在这里
#   它由CoreManager spawn（launch_engine_core -> EngineCore.run_engine），内部持有：
#     Scheduler（第3批）：决定每一步算哪些请求、各算多少token
#     AsyncIOProcManager（async_proc.py）：spawn并驱动本rank的ModelRunner进程（每张GPU一个，第5批）
#   主线程跑busy_loop：收请求 -> schedule -> forward -> postprocess -> 发结果，周而复始
#
# 本文件的类（由run_engine按配置选择）：
#   EngineCore          -dp 1时使用；也是所有变体的基类
#   DPEngineCoreProc    data_parallel_size > 1时使用（--enable-dp-attention会把DP乘以TP，也走这里）：
#                       每一步先和其他DP rank同步状态，没活时跑dummy batch陪跑
#   PPEngineCoreProc    -pp > 1时使用，定义在pp_engine_core.py（【分支】见第9批）
#   PrefillEngineCore / DecodeEngineCore   【分支】PD分离（同GPU的RapidServe），由DisaggCoreManager启动，见第9批
#
# 一个step的数据流（单DP）：
#   CoreManager ──ADD帧──► 输入线程 ──► input_queue ──► busy_loop：scheduler.extend(seqs)
#   busy_loop：scheduler.schedule() ──► ScheduledBatch ──► runner_mgr.call_func("forward") ──► ModelRunner × TP
#   ModelRunner rank 0 ──► ScheduledBatchOutput ──► scheduler.postprocess() ──► 流式增量 / 已结束的序列
#   ──► output_queue ──► 输出线程 ──► CoreManager
#
# 对比nano-vllm：nano-vllm没有这个进程，LLMEngine.step()在主进程里直接调用调度器和model_runner：
#     schedule() 得到 (seqs, is_prefill) -> model_runner.call("run", seqs, is_prefill) -> scheduler.postprocess(...)
#   ATOM把这三步原样搬进了EngineCore进程的busy_loop（_process_engine_step_inner），区别在于：
#   ① 请求和结果经ZMQ进出（输入、输出各一个I/O线程）；② step循环自己转，不由调用方驱动；
#   ③ 多了控制命令、指标推送、KV传输轮询、DP同步等旁路
#
# 读法（本文件约2300行，建议按这个顺序读）：
#   ① 下面的【1】~【4】：读懂本文件需要的推理原理（后文用"见文件开头【N】"指回这里）
#   ② EngineCore类上方的线程/队列图 → __init__上方的启动流程图和显存预算条
#   ③ busy_loop → _process_engine_step_inner：一步引擎的主干（两处各有一张流程图）
#   ④ process_input_sockets / process_output_sockets：两个I/O线程和消息格式
#   ⑤ DPEngineCoreProc：DP>1时的锁步与dummy batch（类上方有原理图和时序图）
#   ⑥ 标【分支】的KV传输、动态分块、RL权重卸载，以及文件末尾的PrefillEngineCore / DecodeEngineCore（同GPU的PD分离）：
#     第一遍可以跳过
# 术语速查（后文第一次用到时不一定展开；"→"后是讲得最细的地方）：
#   TP / PP / PCP / DCP     张量并行 / 流水线并行 / prefill上下文并行 / decode上下文并行 → arg_utils.py的EngineArgs里
#                           tensor_parallel_size等字段上方的【科普】；TP为什么要求SPMD见下面【4】
#   DP                      多个EngineCore各管一份完整模型、各自接请求 → 下面DPEngineCoreProc上方
#   KV block                KV cache按block_size个token一块分页存放，block_table记录每个请求用了哪几块 →
#                           config.py里KVCacheTensor上方的【科普】；CPU侧由BlockManager记账（第4批）
#   CUDA Graph / piecewise  录下一次forward的全部kernel、之后整体回放 / 在注意力处把图切成几段 →
#                           config.py里CUDAGraphMode上方的【科普】
#   PD分离 / KV offload     prefill和decode交给不同实例 / 把KV换出到CPU或存储 → arg_utils.py的kv_transfer_config字段上方、
#                           文件开头_KV_OFFLOAD_MODES上方的【科普】；本文件末尾的RapidServe是"同一组GPU上"的PD分离
#   KV connector            在实例之间、或GPU与CPU/存储之间搬KV的插件（mooncake、LMCache等，第9批）
#   MTP                     投机解码的一种，decode每步要一起验证mtp_k个草稿token → arg_utils.py的method字段上方的【科普】
#   ITL / TPOT / TTFT       相邻两个输出token的间隔 / 平均每个输出token的耗时 / 首token延迟。例（公式同llm_engine.py里的
#                           TTFT/TPOT）：100.00s到达、100.25s出第1个token、102.25s出第21个 → TTFT = 0.25s，
#                           TPOT = (102.25 − 100.25) / (21 − 1) = 0.1s；ITL是每个相邻间隔，平时约0.1s，若其间插进一次0.5s的
#                           prefill（见【2】），那一个间隔就约0.6s——平均TPOT只多0.025s，P99 ITL却由这种尖峰决定
#   H2D / D2H               主机→GPU / GPU→主机的拷贝
#   RCCL / gloo             AMD的GPU集合通信库（相当于NCCL）/ PyTorch在CPU张量上做集合通信的后端
#   CUDA IPC                同机进程间共享显存：一个进程把一块显存导出成句柄，另一个进程打开后看到的就是同一块显存，不拷贝
#   "第N批"                 仓库根目录ANNOTATION_PLAN.md的注释批次：3调度器、4 KV账本与状态缓存、5 ModelRunner、
#                           7 MoE/MLA/投机解码、9 PP/PCP/PD分离/KV offload、10 RL rollout/指标
#
# ======================= 读本文件前需要的推理原理（不对应某一行代码） =======================
#
# 【1】引擎step与连续批处理（continuous batching，又叫iteration-level scheduling）
#   自回归生成：一个请求 = 1次prefill（处理整个prompt，顺带采样出第1个token）+ 若干次decode（每次1个token）。
#   引擎以"一次forward"为调度粒度，每一步都重新决定这一步算谁——busy_loop里 schedule → forward → postprocess 的循环就是它。
#   静态批处理一次凑一批、整批跑完才换下一批；连续批处理每一步都允许新请求进来、结束的请求离开。
#   例：A要生成3个token，B要生成8个，C在第1步之后到达（P=prefill，D=decode，.=占着槽位空等或排队）
#     静态批处理  step: 1  2  3  4  5  6  7  8  9  10
#              A:     P  D  D  .  .  .  .  .            ← A第3步就完了，槽位和KV一直空占到整批结束
#              B:     P  D  D  D  D  D  D  D
#              C:     .  .  .  .  .  .  .  .  P  D ...  ← 必须等整批结束
#     连续批处理（按ATOM的规则：一步要么全是prefill、要么全是decode，见【2】；为简单起见按不开延迟输出画，
#       开了延迟输出时请求结束要晚一步才被发现，见【3】）
#                step: 1  2  3  4  5  6  7  8  9
#              A:     P  .  D  D                        ← 第4步拿到第3个token，postprocess立刻释放它的KV block
#              B:     P  .  D  D  D  D  D  D  D
#              C:        P  D  D  ...                   ← 第2步就被调度（这一步是C的prefill，A、B的decode停一步）
#   代价：每一步都要在CPU上做一次调度和postprocess，这部分CPU时间必须很小或者被GPU时间盖住（见【3】）
#
# 【2】prefill与decode：两种性质完全不同的负载
#   prefill：一次处理T个prompt token（几百~几千），线性层是 [T, hidden] x [hidden, out]，每读一次权重做T份计算 → 计算受限
#   decode：每个请求每步只有1个新token，batch=B时线性层是 [B, hidden] x [hidden, out]，每步都要把全部权重
#     （外加所有请求的KV cache）从HBM（GPU的显存）读一遍，只做B份计算 → 访存带宽受限
#   量级估算（70B稠密模型，FP8权重≈70GB；按MI300X理论峰值：HBM≈5.3TB/s，FP8稠密≈2600 TFLOPS（BF16稠密≈1300））：
#     decode B=64：读权重 70GB / 5.3TB/s ≈ 13ms；计算 2×70e9×64 ≈ 9e12 FLOP ≈ 3.5ms → 由带宽决定，B翻倍耗时几乎不变
#     prefill T=8192：2×70e9×8192 ≈ 1.1e15 FLOP ≈ 0.45s → 由算力决定，耗时近似与token数成正比
#     分界点约 2600e12 / 5.3e12 ≈ 490 FLOP/字节；FP8权重每字节每token 2 FLOP → 一步大约超过245个token才转为计算受限
#     （实际能跑到的峰值明显低于理论值，但"decode带宽受限、prefill算力受限"的结论不变）
#   推论：① decode要尽量把batch做大，让更多请求共摊一次权重读取——这正是连续批处理提高吞吐的来源；
#     ② 一次prefill可能耗时几百ms，这一步里正在decode的请求拿不到新token，表现为ITL/TPOT的尖峰
#   ATOM基类Scheduler的选择（scheduler.py的_schedule："Tries prefill first; if no new prefills are ready, falls back to
#     decoding"，代码里留着"TODO for prefill/decode mixed batch"）：一个ScheduledBatch要么是prefill批、要么是decode批，
#     与nano-vllm相同的prefill优先。长prompt靠chunked prefill（enable_chunked_prefill默认True，每步最多
#     max_num_batched_tokens个token）切成多步；PrefillDelayer（见_init_prefill_delayer）把零散的prefill攒成一批再放行，
#     减少decode被打断的次数。vLLM V1等引擎则可以在同一步里混合prefill chunk和decode
#
# 【3】CPU调度与GPU执行的重叠：延迟输出（deferred output）
#   前置：GPU的异步执行与stream（本文件的RapidServe部分、async_proc.py的多步流水图、utils/__init__.py的H2D拷贝都指回这里）
#     CPU调用一个GPU算子只是"发射（launch）"：把kernel排进一个stream（GPU上的FIFO命令队列）就返回，只花几µs；GPU按队列
#     顺序逐个执行。同一stream里的命令严格串行，不同stream之间可以并发
#     CPU只在要读GPU结果时才必须等：.item()、.tolist()、不带non_blocking的.cpu()、torch.cuda.synchronize()、
#     event.synchronize()（event = 排进stream的一个"执行到这里"的标记）——这些是同步点，CPU停在那里直到排在前面的命令都执行完
#     例：decode一步约1200个kernel（80层×每层约15个），CPU发射完要几ms，GPU跑完要10ms量级；只要中途不碰同步点，
#       CPU发射完就能去做别的事，GPU队列里的活也不会断
#   朴素做法（nano-vllm）：CPU调度 → GPU forward → 同步等采样结果拷回CPU → CPU postprocess → 再调度，GPU在CPU干活时空闲。
#     decode一步GPU只要10ms量级时，几ms的Python调度+postprocess就是两三成的GPU空转
#   ATOM的ModelRunner默认开延迟输出（tokenID_processor.is_deferred_out：非PP、且不是"要交接递归状态的PD producer"时为True）：
#     forward(第N步) 发射第N步的kernel并启动采样结果的异步D2H拷贝后，只等待并返回*第N-1步*的token；
#     第N+1步decode的input_ids直接取GPU上的prev_token_ids，不需要先等CPU拿到token
#     （实现：采样结果的D2H拷贝排在单独的async_copy_stream上，拷完record一个event；下一次forward才对这个event做
#      synchronize取出token，见model_runner.py里tokenIDProcessor的send_to_cpu_async / recv_async_output）
#   时间线：
#     CPU(EngineCore): [sched N][launch N][等N-1的token][post N-1][sched N+1][launch N+1][等N的token][post N]...
#     GPU:             ...forward N-1 ===][========== forward N ==========][========== forward N+1 ======...
#     postprocess(N-1)和schedule(N+1)发生在GPU执行第N步期间，被GPU时间盖住，GPU的队列里始终有活
#   代价：scheduler.postprocess拿到的是上一步的结果，序列里要先放占位token（scheduler.postprocess里的num_placeholder），
#     请求的结束（如采到EOS）也晚一步才被发现，多算的那一步结果被丢弃
#   哪些部署关掉它：PP>1，以及"要交接递归状态的PD producer"。后者的原因（【分支】，第9批）：
#     递归状态：GDN等线性注意力层不像MHA/GQA那样每个token在KV cache里占一格，而是每个请求只存一个定长矩阵S，
#       每来一个token就原地更新一次S，见model_config/qwen3_next.py文件头【科普】②
#     PD分离的producer（prefill一侧）要把"prompt末尾的状态 + 第1个采样token T0"交给consumer（decode一侧）。开了延迟输出，
#       producer要再跑一步decode才能把T0取回CPU，而这一步已经把T0写进了状态：
#         分页KV：   T0的K/V落在T0自己的那一格，consumer再写一遍还是同样的值 → 无害
#         递归状态： S已经吃进过T0，consumer再用T0更新一次 → T0被算了两遍，之后的输出全错
#     （model_runner.py的tokenIDProcessor.__init__：is_deferred_out = 非PP 且 not(PD producer 且 有递归状态)）
#
# 【4】为什么每个TP rank一个进程，以及一步怎样广播给所有TP worker
#   张量并行把每层权重切到TP张卡上，层内要做all-reduce。所有rank必须运行"同一个程序、同样形状的输入"
#     （SPMD，Single Program Multiple Data：程序相同，各rank手里的数据分片不同）：
#     任何一个rank少调一次集合通信或形状不同，其他rank就会永远卡在那次集合通信里
#   从单卡MHA/GQA出发看TP=2（例：64个q头、8个KV头、hidden=8192、80层）：
#     单卡：  x ─► q/k/v投影（64个q头、8个KV头）─► 每个头各自做注意力 ─► 拼接64个头 ─► o_proj ─► y
#     TP=2：  GPU0：q头0~31、KV头0~3的投影权重 + o_proj里对应的一半行 ─► 部分和 y0 ─┐
#             GPU1：q头32~63、KV头4~7                                 ─► 部分和 y1 ─┴► all-reduce(求和) ─► y = y0+y1（两卡都拿到）
#     MLP同理（up/gate按列切、down按行切，再一次all-reduce），所以每层约2次all-reduce，一次forward约160次；
#     每张卡只存一半权重、只存自己那4个KV头的KV cache（列切/行切的直觉见arg_utils.py的tensor_parallel_size字段上方的
#     【科普】，实现见model_ops/linear.py的QKVParallelLinear / RowParallelLinear）
#   为什么"少调一次"就会挂死：all-reduce是"所有成员都到齐才返回"的会合点
#     GPU0：收到第N步 ─► forward ─► 第0层的all-reduce ─► 等GPU1……（永远等下去）
#     GPU1：没收到第N步，停在别处 ──✗
#   所以EngineCore只调度一次，把同一个ScheduledBatch经共享内存MessageQueue广播给每个worker（async_proc.py），
#     每个rank执行完全相同的forward，只有rank 0把结果送回
#   用进程而不是线程：每个rank独占一张GPU、各自发射大量kernel，Python里的kernel发射是CPU密集的，多线程会被GIL（CPython的全局解释器锁：
#     同一时刻只有一个线程在执行Python代码）串行化；
#     RCCL/NCCL的惯例也是一个进程一个rank。EngineCore自己也不兼任rank 0（nano-vllm是主进程兼任rank 0），
#     于是调度/postprocess不会和rank 0的kernel发射争同一个GIL
#   DP>1时还要跨DP rank锁步，见下面的DPEngineCoreProc
logger = logging.getLogger("atom")

# Pace of the idle KV drain. The busy loops never block, so an unpaced drain
# would fire one worker RPC round per spin; 1ms matches the PP head's existing
# idle token-poll timeout and is far below any transfer latency.
# 【译】空闲时KV排空的节奏。busy loop从不阻塞，所以不加节奏的排空会在每次空转时都发起一轮worker RPC；
#   1ms与PP head现有的空闲token轮询超时一致，而且远低于任何传输延迟。
# 【分支】KV传输（PD分离 / KV offload），见第9批
KV_IDLE_DRAIN_INTERVAL_S = 0.001

# Upper bound on the drain that runs after the loop exits. A peer that died
# mid-transfer leaves a completion that never arrives; exiting late beats
# never exiting.
# 【译】循环退出后那次排空的时间上限。一个在传输中途死掉的对端会留下一个永远不会到达的完成信号；
#   晚点退出总比永远不退出好。
KV_SHUTDOWN_DRAIN_TIMEOUT_S = 2.0


# EngineCore：引擎子进程的主体（一个DP rank一个实例）
#
# 进程内的三个线程，以及它们之间的队列：
#
#   CoreManager（主进程）           EngineCore进程
#   ──────────────────────          ─────────────────────────────────────────────────────────────────
#   ROUTER(input)   ══ ADD ═══════►  ┐ 输入线程 process_input_sockets（DEALER×2，一个Poller同时等两个socket）
#   ROUTER(control) ══ UTILITY ═══►  ┘   ADD      ─► input_queue.put(list[Sequence])
#                      SHUTDOWN          UTILITY  ─► utility_queue.put((cmd, args))，置_has_pending_utility
#                                        SHUTDOWN ─► input_queue.put([退出哨兵])，线程结束
#                                             │                         │
#                                             ▼                         ▼
#                                     busy_loop主线程（run_engine里调用）
#                                       pull_and_process_input_queue    utility_handler.process_queue（执行控制命令）
#                                       schedule ─► forward ─► postprocess
#                                       output_queue.put(...)：
#                                         list[Sequence]               已结束的序列（含被拒绝的）
#                                         ("STREAM", [...])            本step的流式增量
#                                         ("READY", payload)           初始化完成（只发一次）
#                                         ("METRICS", dict)            定期指标
#                                         ("UTILITY_RESPONSE", dict)   工具RPC的回复
#                                         [退出哨兵]                    exit()时：通知输出线程收尾
#                                             │
#                                             ▼
#   PULL(output)    ◄═══════════════  输出线程 process_output_sockets（PUSH）：pickle成 (EngineCoreRequestType, data) 发出
#
# 为什么拆成三个线程：busy_loop在forward时会阻塞等ModelRunner（几毫秒到几秒），两个I/O线程保证这期间新请求照样被收进
#   队列、结果照样被发出去。线程在等socket/队列时会释放GIL，所以互不拖累；线程之间只通过线程安全的queue.Queue交换数据
class EngineCore:
    # This process's name, for the title and every GC log line. A class
    # attribute because it is per-process state and each engine is spawned into
    # its own interpreter; `_setup_engine_process` is the only writer.
    # 【译】本进程的名字，用于进程标题和每一行GC日志。是类属性，因为它是每个进程的状态，而每个engine都被spawn进
    #   自己的解释器；_setup_engine_process 是唯一的写入者。
    _process_name = "EngineCore"

    # 构造EngineCore（在子进程里执行）。步骤见下面的"启动流程图"；为什么必须是这个顺序见流程图上方的①~⑤
    # 对比nano-vllm：nano-vllm的ModelRunner.__init__里allocate_kv_cache直接改写config.num_kvcache_blocks，主进程里的
    #   Scheduler随后读同一个对象；ATOM的ModelRunner在另一个进程里，它的改写对EngineCore不可见，所以要显式用RPC
    #   拿回block数，再写进本进程的config（Config.num_kvcache_blocks初始为-1）
    #
    # 前置（下面的显存预算里，两项既不是权重也不是KV的开销）：
    #   激活（activation）：forward中的中间张量——每层的输入/输出hidden、q/k/v、MLP中间结果、注意力的workspace……一层用完就释放，
    #     下一层复用同一块显存，所以占用的不是所有层之和，而是某一时刻同时活着的最大值（峰值），并随一次forward的token数线性增长。
    #     例（下面②里的70B模型：hidden=64×128=8192、MLP中间维28672；一次prefill max_num_batched_tokens = 16384个token，默认值）：
    #     单是MLP的gate/up输出 [16384, 2×28672] BF16 就 ≈ 1.9GB。KV cache只能分配在这个峰值之外，否则运行中一次满额prefill就会OOM
    #   CUDA Graph的私有内存池：录图时forward里的中间张量都从一个专用池分配，地址被写死进graph、回放时原地复用，
    #     所以这块显存录完后一直被占着、不能再分给KV（各个batch尺寸的graph共用一个池，图解见utils/__init__.py里
    #     _ATOM_WEAKREF_OP上方的"覆盖复用"；CUDA Graph本身见config.py里CUDAGraphMode上方的【科普】）
    #
    # 为什么启动顺序必须是 加载权重 → warmup → 测显存算block数 → 分配KV → 捕获CUDA Graph → 建Scheduler：
    #   ① warmup（ModelRunner.__init__末尾的_maybe_warmup → warmup_model）：先reset_peak_memory_stats，再跑一次
    #     max_num_batched_tokens量级的假prefill，让PyTorch分配器记下激活的峰值——相当于一次profile run
    #   ② get_num_blocks：预算 = 总显存×gpu_memory_utilization − (权重+激活峰值 + 分配器之外的RCCL等缓冲
    #     + CUDA Graph池的估计 + 2%余量)，再与实际空闲显存取min；剩下的全给KV cache，除以每个block的字节数得到block数
    #     （逐项拆解见下面的"显存预算条"）
    #     例（量级，数字与预算条一致）：192GB的卡、utilization=0.9 → 预算172.8GB；权重70 + 激活峰值8 + 其他约7.4
    #       （非torch≈2 + graph池≈1.6 + 2%余量≈3.8）→ KV约87GB；
    #       80层、8个KV头（GQA：64个q头每8个共用一组K/V）、head_dim=128、FP8 KV的模型，
    #       每token的KV = 2(K和V)×80×8×128×1B = 160KB，再加FP8的每token每头fp32 scale 2×80×8×4B = 5KB → 约165KB；
    #       block_size=16 → 每个block 2,703,360 B ≈ 2.58MB → 87.36GB / 2.58MB ≈ 34,698个block ≈ 55.5万个token的KV容量
    #       （逐项算法见 model_runner.py 中 get_num_blocks 上方的显存预算条，那里是规范位置）
    #   ③ 为什么不能写死：权重大小、激活峰值、TP切分、KV dtype、同卡上的其他进程都会改变结果，只能在本机实测；
    #     而block数就是调度器的准入上限（BlockManager没有空闲block时请求只能等待或被抢占），所以必须测完才能建Scheduler
    #   ④ CUDA Graph在KV分配之后捕获：graph里记录的是kernel参数中的显存地址，KV cache必须已经在它最终的地址上；
    #     graph私有内存池占的显存已经在②里用_estimate_cudagraph_overhead预留
    #   ⑤ 最后冻结GC（_freeze_after_startup）：启动期要活一辈子的对象此时都已建好（分代GC与gc.freeze见gc_utils.py文件开头的【科普】）
    #
    # 启动流程图（单DP、非PD分离；每一格右边是"在哪个进程里做"）：
    #   ┌───────────────────────────────────────────────┐
    #   │ 启动输出线程、输入线程（握手空帧尽早发出）       │ EngineCore
    #   │ init_exit_handler / _init_data_parallel        │ EngineCore（DP>1时在这里等所有DP rank会合）
    #   └───────────────────────┬───────────────────────┘
    #                           ▼
    #   ┌───────────────────────────────────────────────┐
    #   │ AsyncIOProcManager：spawn tp×pcp 个worker      │ 每个worker进程里：ModelRunner.__init__
    #   │   load_model → _maybe_warmup（假prefill，        │   = 初始化RCCL → 加载本rank的权重分片
    #   │   记下激活峰值）                                 │   → warmup测峰值
    #   └───────────────────────┬───────────────────────┘
    #                           ▼
    #   _post_model_load_hook（基类空；RapidServe在这里交换权重IPC句柄）
    #                           ▼
    #   ┌───────────────────────────────────────────────┐
    #   │ call_func("get_num_blocks")  → num_blocks      │ 只取rank 0的返回值
    #   │ call_func("allocate_kv_cache", num_blocks)     │ 每个rank分配自己的那份KV
    #   │ config.num_kvcache_blocks = num_blocks（回填）  │ EngineCore
    #   └───────────────────────┬───────────────────────┘
    #                           ▼
    #   enable_dynamic_chunking？ ── 是 ──> call_func("profile_dynamic_chunking")（失败则关掉该特性）
    #                           ▼
    #   enforce_eager 或 本进程是PD分离的decode？ ── 否 ──> call_func("capture_cudagraph")
    #                           │ 是：跳过（decode要等KV导入后在DecodeEngineCore.__init__里再捕获）
    #                           ▼
    #   Scheduler(config)（BlockManager按num_kvcache_blocks建block池） → 按需挂PrefillDelayer
    #                           ▼
    #   EngineUtilityHandler → _freeze_after_startup（本进程 + 每个worker都gc.freeze）
    #                           ▼
    #   _send_ready_signal：("READY", {"max_pool_tokens": ...}) ──输出线程──> CoreManager.__init__ 放行
    #   任一步抛异常：finally里good=False → self._finalizer() → exit()，异常继续抛给run_engine
    #
    # 显存预算条（get_num_blocks的公式，model_runner.py；数字沿用上面②的例子：192GB的卡、util=0.9、
    #   非PIECEWISE模式时graph池按"激活峰值×0.2"估计）：
    #
    #   0                                                                                        192GB（total）
    #   ├─────── 权重 70 ───────┼激活峰值 8┼非torch≈2┼graph≈1.6┼余量3.8┼──────── KV cache ≈87 ────────┼不进预算 19.2┤
    #   │◄──────── peak_torch（= max(peak, current)）──────►│         │        │                              │
    #   │◄──────────────────────── budget = total × gpu_memory_utilization = 172.8GB ──────────────►│
    #     非torch：RCCL/custom all-reduce缓冲等不经PyTorch分配器的显存 = (total−free) − memory_reserved()
    #       （all-reduce见文件开头【4】；custom all-reduce是aiter实现的同机all-reduce，用IPC把各卡的一块缓冲互相映射后
    #        直接读写、不走RCCL，术语见async_proc.py文件开头）
    #     graph：_estimate_cudagraph_overhead（enforce_eager时为0；PIECEWISE时按逐token字节数×捕获的token总数估计，上限约预算的15%）
    #     余量：total × 2%；RapidServe再多扣 _kv_budget_extra_reserve = 4×2% = 8%；KV connector缓冲（若有）也先扣掉
    #     最终 available_for_kv = min(上面算出的值, 实际空闲显存 − connector缓冲)，再交给plan_pools按各子池切分成block
    #   读法：gpu_memory_utilization每调高0.01，在这张卡上就多出约1.92GiB的KV（约763个2.58MB的block、约1.22万个token）；
    #     调得太高的代价是分配器碎片、其他进程或之后的临时分配没有余地，运行中OOM
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        # input_queue：输入线程放入、busy_loop取出的新请求；每个元素是一个ADD帧带来的一批Sequence
        #   （类型标注写的是Sequence，实际放的是list[Sequence]，关闭时是只含退出哨兵的[exit_seq]）
        self.input_queue = queue.Queue[Sequence]()
        # output_queue：busy_loop等各处放入、输出线程取出的待发消息（已结束序列的列表，或 ("STREAM"/"READY"/..., 数据) 元组，见类上方的图）
        self.output_queue = queue.Queue[list[Sequence]]()
        # 【译】（行尾注释）用于流式中间输出的队列
        # 解释：stream_output_queue：scheduler.postprocess把每一步的流式增量 [(seq_id, RequestOutput), ...] 放进来，随后转进output_queue
        self.stream_output_queue = (
            queue.Queue()
        )  # Queue for streaming intermediate outputs
        # Queue for utility commands (processed in busy_loop to avoid thread contention)
        # 【译】工具命令的队列（在busy_loop里处理，以避免线程争用）
        # 解释：命令由输入线程放入，但只在busy_loop主线程里执行，这样命令不会和调度/forward并发地修改调度器等状态
        self.utility_queue = queue.Queue()
        # 【译】（行尾注释）一个标志，避免每轮循环都去检查空队列
        self._has_pending_utility = (
            False  # Flag to avoid checking empty queue every loop
        )
        # 【译】（行尾注释）RL训练时权重被卸载出显存时为True
        # 【分支】RL rollout的权重卸载与热更新，见第10批
        self._is_rl_weights_offloaded = (
            False  # True when weights are offloaded for RL training
        )
        # Separate from memory residency: an update in progress, or one that
        # failed after an in-place write, must not serve.
        # 【译】与显存驻留状态分开：一次进行中的更新、或者一次在原地写入之后失败的更新，都不能对外服务。
        self._weight_sync_fenced = False
        self._rl_weights_inconsistent = False
        self.input_address = input_address
        self.output_address = output_address
        # Control traffic arrives on its own socket so CoreManager can keep the
        # request socket single-writer; see CoreManager._send_request.
        # 【译】控制流量走自己的socket，这样CoreManager可以让请求socket保持单写者；见 CoreManager._send_request。
        self.control_address = config.parallel_config.control_address
        assert self.control_address, (
            "parallel_config.control_address is unset -- an EngineCore must be "
            "launched through CoreManager, which allocates the control channel"
        )
        # 先启动输出线程：之后的READY、以及失败时的SHUTDOWN都要靠它发出去
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()

        # Start input thread BEFORE _init_data_parallel so that CoreManager
        # can receive the input socket connection and proceed to start the
        # remaining DP ranks.  Without this, _init_data_parallel blocks on
        # rendezvous waiting for all DP ranks, but they haven't been spawned
        # yet because CoreManager is still waiting for *this* rank's socket.
        # The READY signal (sent at the end of __init__) gates actual request
        # processing, so starting the input thread early is safe.
        # 【译】在 _init_data_parallel 之前启动输入线程，这样CoreManager能收到输入socket的连接，进而去启动其余的DP rank。
        #   否则 _init_data_parallel 会阻塞在集合点上等所有DP rank，而它们还没被spawn，因为CoreManager还在等*这个*rank的
        #   socket。READY信号（在 __init__ 末尾发送）才是真正放行请求处理的闸门，所以提前启动输入线程是安全的。
        # 解释：v0.1.8的CoreManager.__init__是先start全部EngineCore进程、再逐个bind并等握手，所以"其余rank还没被spawn"
        #   在当前顺序下不会发生；但CoreManager仍要逐个等每个rank的握手空帧，输入线程越早启动、握手越早完成。
        #   READY之前不会有请求发来（CoreManager.__init__在等READY）
        self.input_thread = threading.Thread(
            target=self.process_input_sockets,
            args=(self.input_address, self.control_address),
            daemon=True,
        )
        self.input_thread.start()

        # 保存config.mark_trace（本类不再读取它；ModelRunner自己从config读）
        self.mark_trace = getattr(config, "mark_trace", False)
        # 装退出处理：设self.still_running=True、self._finalizer=weakref.finalize(self, self.exit)，收到SIGTERM时调用_finalizer
        #   （随后构造的AsyncIOProcManager会重新装SIGTERM处理器：先关worker、再回调这里的_finalizer，见async_proc.py）
        #   必须在构造AsyncIOProcManager之前：_finalizer要作为"worker意外退出时的回调"传给它
        init_exit_handler(self)
        # DP>1时（DPEngineCoreProc）建立DP rank之间的gloo进程组；基类为空操作
        self._init_data_parallel(config)

        # Initialize model runner processes
        # 【译】初始化model runner进程
        try:
            good = False
            # Number of worker processes for THIS EngineCore = one pipeline
            # stage's slice = tp x pcp. Pipeline parallelism spans *separate*
            # EngineCores (one per stage, spawned by CoreManager), not extra
            # workers inside a single EngineCore — so pp does NOT multiply here.
            # tp_world_size, not tensor_parallel_size: under simulated TP only
            # the first tp_world_size shards get a process.
            # 【译】这个EngineCore的worker进程数 = 一个流水线stage的切片 = tp x pcp。流水线并行跨越的是*彼此独立*的EngineCore
            #   （每个stage一个，由CoreManager spawn），而不是在单个EngineCore里多开worker——所以pp在这里不参与相乘。
            #   用 tp_world_size 而不是 tensor_parallel_size：在模拟TP下只有前 tp_world_size 份分片才有进程。
            # 解释：AsyncIOProcManager构造时spawn这些进程，每个进程里构造config.runner_qualname指定的runner类（默认
            #   atom.model_engine.model_runner.ModelRunner，第5批）。之后EngineCore用 call_func(方法名, 参数) 在所有worker上调用
            #   runner的同名方法，wait_out=True时阻塞等rank 0的返回值。例：-tp 8 -> 8个ModelRunner进程；-tp 4 -pcp 2 -> 8个
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tp_world_size * config.prefill_context_parallel_size,
                config.runner_qualname,
                config,
            )
            # 子类钩子（基类为空）：PD分离（RapidServe）在这里交换权重的CUDA IPC句柄，见第9批
            self._post_model_load_hook()
            # KV cache大小：ModelRunner用 显存×gpu_memory_utilization 减去权重、激活峰值等开销，算出能放多少个block（第5批）
            #   返回dict：num_kvcache_blocks、pool_entries、pool_entries_per_req、state_runtime；call_func只拿rank 0的返回值
            block_info = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            # （此时每个worker的ModelRunner构造已经跑完warmup，激活峰值已记录在分配器统计里，get_num_blocks读的就是它）
            num_blocks = block_info["num_kvcache_blocks"]
            # Sizing happens in the runner subprocess, so nothing it wrote to
            # its own `config` is visible here. Carry the per-class entry table
            # across; BlockManager (built in Scheduler below) and the
            # sliding-window pool each look up the class they declared, so
            # adding an architecture never touches this line.
            # 【译】规模计算发生在runner子进程里，所以它写进自己那份 config 的任何东西在这里都看不到。把按类别的条目表带过来；
            #   BlockManager（在下面的Scheduler里构建）和滑动窗口池各自查找自己声明的类别，所以新增一种架构永远不用改这一行。
            config.pool_entries = block_info.get("pool_entries", {})
            config.pool_entries_per_req = block_info.get("pool_entries_per_req", {})
            # 【分支】线性注意力/Mamba等递归状态的运行时描述（从可pickle的wire格式还原），交给Scheduler，见第4批
            self.state_runtime = StateRuntime.from_wire(block_info["state_runtime"])
            # 让每个ModelRunner按这个block数真正分配KV cache显存，并绑定到各注意力层（第5批）
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            # ★ 回填：Config.num_kvcache_blocks初始为-1，这里写入真实值；下面构造的Scheduler/BlockManager读的就是它
            #   只写进了本EngineCore进程的config副本，主进程里LLMEngine的Config仍是-1（所以READY要另外上报max_pool_tokens）
            config.num_kvcache_blocks = num_blocks
            # Config has already turned the feature off wherever it cannot act,
            # so every stage of a pipeline takes this branch or none does: the
            # sweep's dummy forwards cross stages and must run in lockstep.
            # 【译】Config已经在这个特性无法起作用的地方把它关掉了，所以流水线的每个stage要么都走这个分支，要么都不走：
            #   扫描用的dummy forward会跨stage，必须锁步运行。
            # 【分支】动态分块的PP prefill，见第9批 dynamic_chunking.py
            # 【科普】动态分块（dynamic chunking）是什么——从chunked prefill和MHA的注意力开销出发：
            #   前置①流水线并行PP（完整讲解见arg_utils.py pipeline_parallel_size上方的【科普】）：-pp 2时把80层的模型切成两段，
            #     stage0（一个EngineCore + 它的GPU）算第0~39层，把hidden state交给stage1算第40~79层。同一个长prompt的各个chunk
            #     可以流水起来：stage1算c0时，stage0已经在算c1（c1在stage0上只用到stage0自己存的、c0在第0~39层的KV）
            #   前置②流水线气泡：某个stage手上没活、在等上游送来下一块的空闲时间。每块在各stage上耗时都是t_i时，一条prompt的
            #     完成时间 = Σt_i + (stage数−1) × max(t_i)——最慢的那一块决定了气泡有多大
            #   长prompt按chunk分步prefill（见文件开头【2】）。一块有x个新token、前面已有L个token的KV时，
            #   这一块的注意力工作量 ∝ 2Lx + x²（2Lx：新token对前缀；x²：块内），L越大越慢：
            #     固定x=8k：块0(L=0) 1份   块1(L=8k) 3份   块2(L=16k) 5份   块3(L=24k) 7份     （以x²=8k×8k次点积为1份）
            #   PP下每块依次流过各stage，后一块总比前一块慢，下游stage就要空等（流水线气泡）：
            #     stage0: [c0][ c1 ][  c2  ][   c3   ]
            #     stage1:     [c0]..[ c1 ]..[  c2  ]...[   c3   ]         （..=等上游送来下一块）
            #     只算注意力项、-pp 2：完成时间 = (1+3+5+7) + 1×7 = 23份；同样32k个token若切成4块等耗时的块（注意力总量
            #     不变，仍是(32k)² = 16份，每块4份），完成时间 = 16 + 1×4 = 20份。-pp 4时是 16+3×7 = 37份 对 16+3×4 = 28份
            #   动态分块用延迟模型 t(L,x) = c + γ·L + b·x + a·(2Lx + x²) 给后面的块选更小的x，让每块耗时接近第一块，
            #   再按--dynamic-chunking-smooth-factor（默认0.75）在固定块和等耗时块之间插值，且不小于最小块（默认4096）。
            #   代价：块数变多，每块都要付的c和γ·L也多付几次；所以只在一个请求独占prefill时才缩块（别的请求本身就能填气泡，
            #     dynamic_chunking.has_sole_prefill，见docs/dynamic_chunking_guide.md的Runtime behavior一节）
            #   b由这里的启动扫描测出（同时得到c的初值）；a、γ、c在服务中用真实prefill拟合（dynamic_chunking.py模块docstring：
            #     "Startup dummy forwards fit b; a two-size runtime prefill sweep fits a, gamma and c"；另见下面的英文注释）；
            #   详见docs/dynamic_chunking_guide.md，模型在dynamic_chunking.py，每块大小由Scheduler._dynamic_chunk_limit决定
            if config.enable_dynamic_chunking:
                # Startup profiling only measures the chunk cost that carries no
                # attention; the attention terms are calibrated from real
                # prefills while serving, so no chunk is rebalanced until then.
                # 【译】启动时的profiling只测量不含注意力的那部分chunk开销；注意力项在服务过程中用真实的prefill来校准，
                #   所以在那之前不会对任何chunk做重新平衡。
                profile = self.runner_mgr.call_func(
                    "profile_dynamic_chunking", wait_out=True
                )
                if not profile or "linear_coeff" not in profile:
                    config.enable_dynamic_chunking = False
                    reason = (profile or {}).get("error", "")
                    logger.warning(
                        "%s: disabling dynamic chunking because startup profiling "
                        "did not produce a chunk overhead baseline%s",
                        self.label,
                        f": {reason}" if reason else "",
                    )
                if (
                    config.enable_dynamic_chunking
                    and config.parallel_config.pipeline_parallel_rank != 0
                ):
                    # Downstream stages execute metadata from the PP head and
                    # never make chunking decisions themselves.
                    # 【译】下游stage执行PP head发来的元数据，自己从不做分块决策。
                    config.enable_dynamic_chunking = False
            # 捕获CUDA Graph（--enforce-eager时跳过；PD分离的decode进程要等KV导入后再捕获），第5批
            #   返回 (耗时, 捕获的batch尺寸列表, graph内存池字节数)
            # 为什么decode要用CUDA Graph：decode一步的GPU时间很短（10ms量级），却要发射"层数×每层十几个"小kernel，
            #   例如80层×15≈1200次launch，每次几µs的CPU开销加起来与GPU时间同一量级，GPU反过来等CPU（launch-bound）；
            #   graph把整步录下来，之后一次replay就发射全部kernel。prefill的kernel大、形状多变，launch开销可以忽略
            #   （capture/replay、按batch尺寸分桶和padding的入门图解见config.py里CUDAGraphMode上方的【科普】；默认
            #    --cudagraph-mode FULL录整张图，PIECEWISE类模式按编译切出的片段各录各的、注意力留在图外）
            #   捕获按config.capture_sizes从大到小进行（ModelRunner.capture_cudagraph里sort(reverse=True)）；运行时把实际
            #   batch大小向上取到最近的已捕获尺寸再replay（多出来的行是padding）
            #
            # 什么时候用 --enforce-eager（跳过这里的capture）：
            #   适用场景：① 调试——kernel在graph里replay时出错很难定位，eager下能逐个kernel同步、打断点；
            #     ② 想要更快的启动——捕获要对每个capture size各跑一遍forward（耗时见下面logger.info打出的cap_cost，大模型常是几十秒）；
            #     ③ 显存很紧——graph池（_estimate_cudagraph_overhead估计的那部分）可以全部让给KV cache；
            #     ④ 只做prefill的进程——PrefillEngineCore强制eager，因为prefill的kernel大、launch开销占比小（见上面"为什么"）
            #   代价：decode每一步都要由CPU逐个launch上千个kernel，batch越小越明显（GPU时间短，CPU发射时间占比大），ITL上升、吞吐下降
            #   注意：enforce_eager只关掉CUDA Graph，torch.compile的级别由 --level 单独决定（CLAUDE.md："--level 0=eager"才是不编译）
            #   另：要真正不录图只能用--enforce-eager；原生路径下--cudagraph-mode NONE照样走整图捕获
            #     （ModelRunner不单独判断NONE，见config.py里CUDAGraphMode上方的模式表）
            if not config.enforce_eager and not config.disagg_is_decode:
                cap_cost, bs, pool_bytes = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{bs} cost: {cap_cost:.2f} "
                    f"seconds, pool: {pool_bytes / (1 << 30):.2f}GB"
                )
            good = True
        finally:
            # 任何一步失败（good仍为False）：调用_finalizer -> exit()，关掉已启动的ModelRunner；异常继续向上抛给run_engine
            logger.info(
                f"{self.label}: load model runner {'success' if good else 'failed'}"
            )
            if not good:
                self._finalizer()

        # Decode in disagg mode defers Scheduler creation until after kvcache IPC
        # import sets config.num_kvcache_blocks (BlockManager asserts num_blocks > 0).
        # Keep the attribute defined (None) so EngineUtilityHandler below and other
        # consumers can reference it before DecodeEngineCore creates the real one.
        # 【译】PD分离模式下的decode会推迟创建Scheduler，直到kvcache的IPC导入设置好 config.num_kvcache_blocks
        #   （BlockManager断言 num_blocks > 0）。让这个属性保持已定义（None），这样下面的 EngineUtilityHandler 和其他使用方
        #   在 DecodeEngineCore 创建真正的调度器之前就可以引用它。
        self.scheduler = None
        if not config.disagg_is_decode:
            # 构造调度器（第3批）：内部创建BlockManager（按num_kvcache_blocks建block池）、waiting/running队列等；state_runtime见第4批
            self.scheduler = Scheduler(
                config,
                state_runtime=self.state_runtime,
            )
            # 单DP、单PP且ATOM_PREFILL_DECODE_INTERVAL>0（默认0）时才挂prefill delayer；DP>1时在DPEngineCoreProc.__init__里挂
            if (
                config.parallel_config.data_parallel_size == 1
                and config.pipeline_parallel_size == 1
                and envs.ATOM_PREFILL_DECODE_INTERVAL > 0
            ):
                self._init_prefill_delayer(config)

        # 【分支】动态分块：校准结果的轮询状态（见_poll_dynamic_chunking_calibration），第9批
        self._dynamic_chunking_enabled = config.enable_dynamic_chunking
        self._dynamic_chunking_poll_countdown = DYNAMIC_CHUNKING_POLL_STEPS

        # 【分支】KV传输（跨节点PD分离、KV offload）是否开启：--kv-transfer-config非空，第9批
        self.kv_transfer_enabled = bool(config.kv_transfer_config)
        # 空闲时下一次推进KV传输的时间（按KV_IDLE_DRAIN_INTERVAL_S节流）
        self._next_idle_kv_drain = 0.0
        if self.kv_transfer_enabled:
            # Physical: one output per launched worker, else this waits forever.
            # 【译】物理数量：每个启动的worker一个输出，否则这里会永远等下去。
            self.kv_aggregator = KVOutputAggregator(world_size=config.tp_world_size)

        # 工具RPC处理器（engine_utility.py）：在busy_loop主线程里执行控制命令（abort、profile、权重更新、集合RPC……），
        #   回复放进output_queue；也负责定期push_metrics
        self.utility_handler = EngineUtilityHandler(
            self.runner_mgr,
            self.output_queue,
            label=self.label,
            scheduler=self.scheduler,
        )

        # KV cache allocated, graphs captured, BlockPool built: everything this
        # process holds for its lifetime exists, and the next thing is traffic.
        # 【译】KV cache已分配、graph已捕获、BlockPool已建好：这个进程整个生命周期要持有的东西都已存在，接下来就是流量了。
        self._freeze_after_startup()

        # 往output_queue放 ("READY", {"max_pool_tokens": ...})，由输出线程发给CoreManager（CoreManager.__init__在等它）
        self._send_ready_signal()
        logger.info(f"{self.label}: EngineCore fully initialized and ready")

    # 给调度器挂上PrefillDelayer（第3批prefill_delayer.py）：把prefill攒到一定量再放行，DP>1时还要跨DP rank对齐
    #   cpu_group：DP rank之间的gloo进程组（DP>1时传入），delayer在schedule()里用它做跨DP的all_reduce
    #   调用方：EngineCore.__init__（单DP）、DPEngineCoreProc.__init__（DP>1）
    #   ATOM_ENABLE_PREFILL_DELAYER默认开；RapidServe或没有调度器时不挂
    def _init_prefill_delayer(self, config: Config, cpu_group=None):
        if (
            not envs.ATOM_ENABLE_PREFILL_DELAYER
            or config.enable_rapidserve
            or self.scheduler is None
        ):
            return
        from atom.model_engine.prefill_delayer import PrefillDelayer

        self.scheduler.set_prefill_delayer(
            PrefillDelayer(
                dp_size=config.parallel_config.data_parallel_size,
                cpu_group=cpu_group,
                max_num_batched_tokens=config.max_num_batched_tokens,
                target_fill=envs.ATOM_PREFILL_DELAYER_TARGET_FILL,
                ttft_max_ticks=envs.ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS,
                partial_max_ticks=envs.ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS,
                stall_ticks=envs.ATOM_PREFILL_DELAYER_STALL_TICKS,
                kv_high_watermark=envs.ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK,
                token_usage_low_watermark=envs.ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK,
                max_queue_ms=envs.ATOM_PREFILL_DELAYER_MAX_QUEUE_MS,
                prefill_decode_interval=envs.ATOM_PREFILL_DECODE_INTERVAL,
                exchange_prefill_costs=self.scheduler.cost_balance,
            )
        )

    # 启动完成后冻结GC堆：gc.freeze()把当前所有对象移进永久代，以后的GC不再扫描它们
    #   （启动时创建的大量长寿对象不会拖慢每次GC）；本进程和每个ModelRunner进程都冻结
    #   调用方：__init__末尾、DecodeEngineCore.__init__
    def _freeze_after_startup(self):
        """Freeze this process and its ModelRunner workers.

        The workers are driven from here because only the caller knows warmup
        is over -- weights, compile and capture all land via RPCs it sends, and
        `--enforce-eager` skips the capture step entirely.
        """
        # 【译】冻结本进程及其ModelRunner worker。
        #
        #   worker由这里来驱动，因为只有调用方知道预热已经结束——权重、编译和捕获都是通过它发出的RPC完成的，
        #   而 --enforce-eager 会完全跳过捕获这一步。
        freeze_gc_heap(self._process_name)
        try:
            self.runner_mgr.call_func("freeze_gc_heap", wait_out=True)
        except Exception as e:  # noqa: BLE001 - never fail startup over this
            # 【译】（上一行行尾注释）noqa: BLE001 - 绝不因为这一步让启动失败
            logger.warning(f"{self._process_name}: worker heap freeze skipped: {e}")

    # 每个run_engine开头调用：设置进程名、父进程死亡时由内核回收本进程、调GC阈值、可选的GC调试回调
    @staticmethod
    def _setup_engine_process(name: str):
        """Identity, orphan reaping and GC policy, for every `run_engine`.

        The disaggregated overrides do not call the base one, and each omission
        is silent: an unreaped orphan pins VRAM, an untitled process is
        `python` in `ps`, an unattached callback leaves ATOM_GC_DEBUG inert.
        """
        # 【译】身份、孤儿回收和GC策略，供每个 run_engine 使用。
        #
        #   分离模式重写的run_engine并不调用基类的那个，而每一处遗漏都是无声的：没被回收的孤儿进程会占住显存，
        #   没设标题的进程在 ps 里显示为 python，没挂上的回调会让 ATOM_GC_DEBUG 不起作用。
        # 解释：所以把这些步骤提成这个staticmethod，三个run_engine开头都调用它
        from atom.utils import enable_orphan_reaping

        EngineCore._process_name = name
        set_process_title(name)
        # 【译】（行尾注释）孤儿进程会占住显存和IPC句柄；见它的文档
        # 解释：用prctl(PR_SET_PDEATHSIG)让内核在父进程（CoreManager所在进程）退出时给本进程发SIGKILL
        enable_orphan_reaping()  # orphans pin VRAM + IPC handles; see its docs
        tune_gc()
        maybe_attach_gc_debug_callback(name)

    # 发READY：放进output_queue，由输出线程pickle成 (READY, payload) 发给CoreManager；PD分离的子类重写它来做握手
    def _send_ready_signal(self):
        self.output_queue.put_nowait(("READY", self._ready_payload()))

    # READY附带的数据：{"max_pool_tokens": KV池能容纳的最长prompt}；没有调度器（PD分离的decode初始化期间）时为None
    def _ready_payload(self) -> dict[str, int] | None:
        """Startup facts the frontend cannot read off its own Config.

        `num_kvcache_blocks` is measured in this subprocess, so the API server's
        Config still holds the placeholder. Publishing the derived prompt
        ceiling rather than the raw block count leaves the dcp arithmetic with
        its owner, `BlockManager`, and gives the frontend a single number to
        compare a prompt against.
        """
        # 【译】前端无法从它自己的Config读到的启动事实。
        #
        #   num_kvcache_blocks 是在这个子进程里测出来的，所以API server的Config里仍是占位值。发布推导出的prompt上限而不是
        #   原始的block数，把dcp相关的算术留给它的拥有者 BlockManager，也给前端一个可以直接拿prompt去比的数。
        if self.scheduler is None:
            return None
        return {"max_pool_tokens": self.scheduler.block_manager.max_pool_tokens}

    # 子类钩子：模型加载后、计算KV大小之前调用；基类什么也不做
    def _post_model_load_hook(self):
        """Called after ModelRunner is initialized (model loaded) but before
        get_num_blocks/allocate_kv_cache.  Override in subclasses to inject
        inter-process synchronization at this point in the init sequence."""
        # 【译】在ModelRunner初始化完成（模型已加载）之后、get_num_blocks/allocate_kv_cache之前调用。
        #   在子类中重写它，以便在初始化序列的这个时间点注入进程间同步。

    # 子类钩子：DPEngineCoreProc在这里建立DP进程组；在__init__中、构造ModelRunner之前调用
    def _init_data_parallel(self, config: Config):
        pass

    # 退出EngineCore（由still_running保证最多执行一次）：调用方是run_engine的finally、SIGTERM处理、weakref.finalize，
    #   以及AsyncIOProcManager发现worker意外退出时（它持有的parent_finalizer就是本对象的_finalizer）
    #   步骤：解冻GC堆 -> 通知所有ModelRunner退出（"exit" RPC）-> 在共同的截止时间内join，超时则terminate/kill
    #     -> _send_engine_dead：通知输出线程给CoreManager发SHUTDOWN
    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # Frozen weights and KV cache are unreachable *and* uncollectable, so
        # an engine destroyed in-process would read as a GPU memory leak.
        # 【译】被冻结的权重和KV cache既不可达、*也*不可回收，所以一个在进程内被销毁的engine看起来会像是GPU显存泄漏。
        unfreeze_gc_heap()
        # runner_mgr还没建好就失败了（例如DP初始化出错）：没有worker要关，直接通知CoreManager
        if not hasattr(self, "runner_mgr"):
            self._send_engine_dead()
            return
        # 停掉AsyncIOProcManager的监控线程，免得下面正常关闭worker时被当成"意外退出"
        self.runner_mgr.keep_monitoring = False
        try:
            # 广播"exit"：每个worker执行runner.exit()后跳出自己的busy_loop（不等回复）
            self.runner_mgr.call_func("exit")
        except Exception:
            logger.debug(
                "runner exit call failed; shared memory may already be freed",
                exc_info=True,
            )
        # One deadline for all workers, so a long ATOM_SHUTDOWN_TIMEOUT_S is not
        # paid once per worker.
        # 【译】所有worker共用一个截止时间，这样一个很长的 ATOM_SHUTDOWN_TIMEOUT_S 不会按每个worker各付一次。
        timeout = envs.ATOM_SHUTDOWN_TIMEOUT_S
        deadline = time.monotonic() + timeout
        for proc in self.runner_mgr.procs:
            try:
                alive = proc.is_alive()
            except ValueError:
                # 【译】（行尾注释）进程对象已经被CoreManager关闭了
                continue  # process object already closed by CoreManager
            if alive:
                proc.join(timeout=max(deadline - time.monotonic(), 0))
                # The join above has a timeout; nothing after it did. A worker
                # that outlives it keeps its VRAM slice and its all-reduce IPC
                # handles, and `multiprocessing`'s atexit handler then joins the
                # same process again with NO timeout -- so an engine that is
                # already on its way out hangs there forever. Seen twice on the
                # k3-dev line: the MainThread parks in `_exit_function -> join`,
                # all TP workers stay alive, /metrics keeps answering 200, and
                # only a manual kill recovers the node. `enable_orphan_reaping`
                # cannot help, because PR_SET_PDEATHSIG fires when the parent
                # *dies* and this parent never does.
                # 【译】上面的join有超时；之后的任何东西都没有。一个活过了超时的worker会继续占着它那份显存和all-reduce的IPC句柄，
                #   而 multiprocessing 的atexit处理函数随后会*不带超时地*再join同一个进程——于是一个本来已经在退出途中的engine会
                #   永远卡在那里。在k3-dev分支上见过两次：主线程停在 _exit_function -> join，所有TP worker都活着，/metrics 一直
                #   返回200，只有手动kill才能恢复节点。enable_orphan_reaping 帮不上忙，因为 PR_SET_PDEATHSIG 在父进程*死亡*时
                #   才触发，而这个父进程从不死亡。
                try:
                    if proc.is_alive():
                        logger.warning(
                            "%s: worker pid=%s still alive after %gs "
                            "(ATOM_SHUTDOWN_TIMEOUT_S); terminating",
                            self.label,
                            getattr(proc, "pid", "?"),
                            timeout,
                        )
                        proc.terminate()
                        proc.join(timeout=5)
                    if proc.is_alive():
                        logger.error(
                            "%s: worker pid=%s ignored SIGTERM; killing",
                            self.label,
                            getattr(proc, "pid", "?"),
                        )
                        proc.kill()
                        proc.join(timeout=5)
                except (ValueError, OSError):
                    # 【译】（行尾注释）进程对象已经关闭 / 已经被回收
                    pass  # process object already closed / already reaped
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    # 通知CoreManager"本engine已退出"：往output_queue放一个只含退出哨兵的列表
    #   退出哨兵 = get_exit_sequence()：token为[-1]、status=EXIT_ENGINE的Sequence（sequence.py）
    #   输出线程发现列表里有EXIT_ENGINE的序列 -> 发一个SHUTDOWN帧并结束；这里最多等它0.5秒
    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait([get_exit_sequence()])
        self.output_thread.join(timeout=0.5)

    # EngineCore子进程的入口（launch_engine_core里Process的target）。@staticmethod：不需要实例，子进程里直接调用
    #   按配置选择具体类：PP>1 -> PPEngineCoreProc（第9批）；DP>1 -> DPEngineCoreProc；否则 -> EngineCore
    #   构造完成（READY已发出）后进入busy_loop，直到收到关闭；不论正常退出还是异常，finally里都调用exit()
    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        # engine_process_name：例如EngineCore、EngineCore_DP0、EngineCore_PP1
        EngineCore._setup_engine_process(engine_process_name(config))

        engine: EngineCore = None
        try:
            # 【分支】PP：每个stage一个PPEngineCoreProc，见第9批 pp_engine_core.py
            if config.pipeline_parallel_size > 1:
                from atom.model_engine.pp_engine_core import PPEngineCoreProc

                engine = PPEngineCoreProc(config, input_address, output_address)
            elif config.parallel_config.data_parallel_size > 1:
                # DP>1（包括--enable-dp-attention改写之后的情况）
                engine = DPEngineCoreProc(config, input_address, output_address)
            else:
                engine = EngineCore(config, input_address, output_address)
            # 构造函数返回时READY已经发出；进入主循环，直到收到SHUTDOWN
            engine.busy_loop()
        except Exception:
            logger.exception("run_engine failed")
            raise
        finally:
            # engine为None表示构造失败，这里就不调用exit()
            if engine is not None:
                engine.exit()

    # 【分支】RL rollout（第10批）：权重被卸载或正在做权重同步时，本轮不执行模型，sleep 10ms避免空转占满CPU
    def _is_idle_rl_weights_offloaded(self) -> bool:
        """Check if weights are offloaded for RL training.

        When offloaded, busy-wait with a short delay to avoid CPU spin.
        Returns True if the caller should skip model execution this tick.
        """
        # 【译】检查权重是否因RL训练而被卸载。
        #
        #   被卸载时，用一个短延迟忙等，避免CPU空转。如果调用方本轮应该跳过模型执行，返回True。
        if self._is_rl_weights_offloaded or self._weight_sync_fenced:
            time.sleep(0.01)
            return True
        return False

    # EngineCore的主循环（单DP版本；DP>1见DPEngineCoreProc.busy_loop）。每一轮：
    #
    #   ┌─► ① utility_handler.process_queue     执行积压的控制命令（abort、profile、权重更新、集合RPC……）
    #   │   ② 到时间就push_metrics               往output_queue放 ("METRICS", 快照)，默认每1秒一次
    #   │   ③ scheduler.heartbeat_throughput     空闲时也按时结算吞吐统计窗口
    #   │   ④ pull_and_process_input_queue       新请求 -> scheduler.extend()；见到退出哨兵 -> shutdown -> 跳出循环
    #   │   ⑤ RL权重已卸载 / 正在权重同步         sleep 10ms，跳过本轮
    #   │   ⑥ 调度器里还有请求                   _process_engine_step()：
    #   │        schedule() ──► ScheduledBatch ──► call_func("forward", batch, wait_out=True) ──► ScheduledBatchOutput
    #   │           │                              （阻塞到ModelRunner rank 0返回）
    #   │           └─ take_rejected()：被拒绝的请求直接作为"已结束"输出
    #   │        ──► scheduler.postprocess() ──► 流式增量（"STREAM"）+ 已结束的序列 ──► output_queue ──► 输出线程
    #   │      调度器空了但KV传输还在途         _advance_idle_kv_transfer()（第9批）
    #   └──────────────────────────────────────┘
    #
    # 循环从不阻塞等待：没有请求时也一直空转（每轮只做几次队列和时间检查）
    # 对比nano-vllm：nano-vllm的generate里 while not is_finished(): step()，由调用方驱动，所有请求做完就返回；
    #   ATOM的busy_loop常驻，空闲时也在转，直到收到SHUTDOWN
    #
    # 为什么忙等（spin）而不是阻塞在队列上：decode一步只有10ms量级，阻塞的线程被唤醒、被OS调度回来再拿到GIL会带来
    #   几十µs以上的抖动；空转只多占一个CPU核。可能耗时的ZMQ收发和pickle都放在输入/输出线程里，主线程每轮只做
    #   不阻塞的queue检查。取舍：空转的主线程一直持有GIL，有新消息到达的I/O线程要等解释器的切换间隔
    #   （sys.getswitchinterval()，默认5ms）才能抢到GIL；forward期间主线程阻塞在call_func里，GIL是释放的
    def busy_loop(self):
        shutdown = False
        # 指标推送间隔：ATOM_METRICS_UPDATE_INTERVAL_S（默认1.0秒）
        metrics_interval = envs.ATOM_METRICS_UPDATE_INTERVAL_S
        next_metrics_push = 0.0
        try:
            while True:
                # ① 处理控制命令（_has_pending_utility为False时立即返回）
                self.utility_handler.process_queue(self.utility_queue, self)
                # ② 定期推送指标；time.monotonic()：单调时钟，不受系统时间调整的影响，适合算时间间隔
                now = time.monotonic()
                if now >= next_metrics_push:
                    next_metrics_push = now + metrics_interval
                    self.utility_handler.push_metrics()
                # ③ 吞吐统计窗口的心跳
                self.scheduler.heartbeat_throughput(now)
                # ④ 取新请求；一旦见过退出哨兵，shutdown就一直为True（or短路，不再读输入队列）
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                # ⑤ RL权重卸载/同步中：跳过本轮
                if self._is_idle_rl_weights_offloaded():
                    continue
                # ⑥ 有活就跑一步；没活但KV传输还在途，就推进KV传输
                if not self.scheduler.is_finished():
                    self._process_engine_step()
                elif self.has_pending_kv_work():
                    self._advance_idle_kv_transfer()
        finally:
            # Teardown runs even on exceptions so the sender thread/socket
            # don't leak. Isolate the final publish so a publisher hiccup
            # cannot skip shutdown_kv_events().
            # 【译】即使出现异常也要执行拆除，这样发送线程/socket不会泄漏。把最后一次发布隔离开，这样发布器的一次小故障
            #   不会跳过 shutdown_kv_events()。
            # 解释：先给在途的KV传输最多2秒收尾，再发布最后一批KV事件、关闭KV事件发布器（第4批）
            self._drain_kv_work_at_exit()
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()

    # 执行一步引擎（调度 + forward + 后处理），返回是否真的跑了forward；真正的逻辑在_process_engine_step_inner
    #   这一层只负责：不论成功还是异常，都发布本步产生的KV cache事件（block存入/淘汰，第4批）
    #   PrefillEngineCore / DecodeEngineCore整体重写了这个方法
    def _process_engine_step(self):
        try:
            return self._process_engine_step_inner()
        finally:
            # Swallow publisher errors so they cannot mask an exception from
            # the engine step itself.
            # 【译】吞掉发布器的错误，这样它们不会掩盖engine step本身抛出的异常。
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish in engine-step finally failed")

    # 【分支】多模态：通知worker释放这些请求的多模态缓存（只有cache_seed != -1的请求才有），不等回复
    def _release_multimodal_requests(self, sequences):
        request_ids = [
            seq.id for seq in sequences if getattr(seq, "cache_seed", -1) != -1
        ]
        if request_ids:
            # Ordered after the completed forward on every worker, including
            # final/aborted requests when there will be no subsequent batch.
            # 【译】在每个worker上排在已完成的forward之后，包括没有后续batch时的最后一批/被abort的请求。
            self.runner_mgr.call_func("release_multimodal_requests", request_ids)

    # 一步引擎的主体，返回True表示本步执行了forward和postprocess
    #   ① scheduler.schedule()：返回 (ScheduledBatch, {seq_id: Sequence})，没有任何请求时返回None
    #   ② take_rejected()：调度器拒绝的请求（如prompt超出KV池容量）直接作为已结束序列输出
    #   ③ 【分支】KV传输：把connector元数据交给worker（触发异步KV加载），第9批
    #   ④ forward：ModelRunner执行模型并采样，返回ScheduledBatchOutput（开延迟输出时，拿到的是上一步的token，第5批）
    #   ⑤ 轮询KV传输进度
    #   ⑥ scheduler.postprocess()：把token追加进序列、判断结束、释放block，产出流式增量和已结束序列（第3批）
    #   ⑦ 流式增量 -> output_queue（("STREAM", ...)），已结束的序列 -> output_queue（先STREAM后结束序列）
    # 对比nano-vllm：对应nano-vllm的LLMEngine.step()：schedule -> model_runner.call("run") -> postprocess
    # 决策流程（与下面的代码一一对应）：
    #   result = scheduler.schedule()
    #   take_rejected()非空？ ── 是 ──> 释放其多模态缓存，作为"已结束"序列放进output_queue（不论本步有没有batch）
    #            │
    #   result为None 或 scheduled_batch为None？ ── 是 ──> _advance_idle_kv_transfer()；return False
    #            │ 否
    #   开了KV传输 且 batch带connector元数据？ ── 是 ──> call_func("process_kvconnector_output", meta)（不等待）
    #            │
    #   batch.req_ids非空？ ── 否 ───────────────────────────┐
    #            │ 是                                         │
    #   记指标 → call_func("forward", batch, wait_out=True)   │
    #   有PrefillDelayer且本批含prefill？ 是 -> notify_prefill_executed()
    #            │                                            │
    #            ▼◄───────────────────────────────────────────┘
    #   _poll_kv_transfer_progress()（未开KV传输时立即返回）
    #            │
    #   batch为空？ ── 是 ──> return False（不做postprocess）
    #            │ 否
    #   postprocess(seqs, fwd_out) ──► stream_output_queue里的每份增量 ──► ("STREAM", ...) ──► output_queue
    #   finished_seqs非空？ 是 -> 释放多模态缓存 ──► output_queue.put(finished_seqs)（一定排在STREAM之后）
    #   return True
    def _process_engine_step_inner(self):
        result = self.scheduler.schedule()

        # Surface admit-rejected seqs (those `_unschedulable_reason` flags in
        # the scheduler) through the same finished-seq path as normal seqs.
        # Without this, `llm.generate()` blocks forever waiting for an output
        # the rejected seq will never produce.
        # 【译】把入场时被拒绝的seq（调度器里 _unschedulable_reason 标记的那些）走和正常seq一样的"已结束"路径交出去。
        #   否则 llm.generate() 会永远阻塞，等一个被拒绝的seq永远不会产生的输出。
        rejected = self.scheduler.take_rejected()
        if rejected:
            self._release_multimodal_requests(rejected)
            self.output_queue.put_nowait(rejected)

        # 调度器里没有任何请求：顺便推进一下KV传输，本步不跑forward
        if result is None:
            self._advance_idle_kv_transfer()
            return False
        # scheduled_batch：打包好交给ModelRunner的批（第3批ScheduledBatch）；seqs：{seq_id: Sequence}，留在本进程做postprocess
        scheduled_batch, seqs = result

        # 防御性分支：基类Scheduler没活时直接返回None，不会返回 (None, ...)
        if scheduled_batch is None:
            logger.debug("%s: No sequences to schedule, skipping forward", self.label)
            self._advance_idle_kv_transfer()
            return False

        # Dispatch KV connector metadata to workers (triggers async KV load)
        # 【译】把KV connector元数据分发给worker（触发异步KV加载）
        # 【分支】PD分离 / KV offload，见第9批
        if (
            self.kv_transfer_enabled
            and scheduled_batch.connector_meta_output is not None
        ):
            self.runner_mgr.call_func(
                "process_kvconnector_output", scheduled_batch.connector_meta_output
            )

        # Run the model forward pass if there are actual sequences
        # 【译】如果确实有序列，就运行模型的forward
        has_seqs = len(scheduled_batch.req_ids) > 0
        if has_seqs:
            # 指标：计算注意力FLOP估算需要的聚合量，并记录本次forward（第10批）
            self.scheduler.compute_detailed_aggregates(scheduled_batch, seqs)
            self.scheduler.metrics.record_forward(scheduled_batch, seqs)
            # ★ forward：ScheduledBatch经共享内存广播给所有TP worker，每个worker都执行ModelRunner.forward；
            #   wait_out=True：阻塞到rank 0把ScheduledBatchOutput（每个请求采样出的token等）发回来
            #   开延迟输出时（默认），它在GPU还在执行本步时就会返回：rank 0只等上一步的token，见文件开头【3】
            fwd_out = self.runner_mgr.call_func(
                "forward", scheduled_batch, wait_out=True
            )
            if (
                self.scheduler.prefill_delayer is not None
                and scheduled_batch.total_seqs_num_prefill > 0
            ):
                # Arm post-prefill decode protection only after the prefill
                # forward really completed. A delayer FIRE merely grants
                # admission and can still result in a decode/empty batch.
                # 【译】只在prefill的forward真正完成之后，才启用prefill之后的decode保护。delayer的一次FIRE只是准许入场，
                #   最后仍可能得到一个decode批或空批。
                self.scheduler.prefill_delayer.notify_prefill_executed()

        # Aggregate KV transfer status from all workers (only when PD disaggregation is active)
        # 【译】汇总所有worker的KV传输状态（只在PD分离开启时）
        # 解释：实际条件是kv_transfer_enabled（PD分离或KV offload任一开启），否则立即返回
        self._poll_kv_transfer_progress()

        if not has_seqs:
            logger.debug("%s: Empty scheduled batch, skipping postprocess", self.label)
            return False

        # dict.values()：只取Sequence对象
        seqs = seqs.values()
        # Pass stream_output_queue to postprocess for streaming callbacks
        # 【译】把 stream_output_queue 传给 postprocess，用于流式回调
        # 解释：postprocess把本步的流式增量放进stream_output_queue，返回本步结束的序列列表（第3批）
        #   开延迟输出时fwd_out.is_deferred_out为True，其中的token属于上一步的batch，postprocess用fwd_out.get_idx(seq.id)
        #   按请求id对号入座；本步刚做完prefill的序列还没有真实token，只先占位（见文件开头【3】）
        finished_seqs = self.scheduler.postprocess(
            seqs,
            fwd_out,
            stream_output_queue=self.stream_output_queue,
            batch=scheduled_batch,
        )

        # Send stream outputs to main process via output_queue
        # 【译】通过 output_queue 把流式输出发给主进程
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                # Send stream outputs as intermediate results
                # 【译】把流式输出作为中间结果发送
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass

        # 已结束的序列排在STREAM之后入队：CoreManager的输出线程依赖这个顺序（先交付最后一个token，再处理结束）
        if finished_seqs:
            self._release_multimodal_requests(finished_seqs)
            self.output_queue.put_nowait(finished_seqs)

        return True

    # 【分支】KV传输（第9批）：调度器已空但KV传输还没收尾时返回True，让busy_loop继续轮询
    def has_pending_kv_work(self) -> bool:
        """True while KV transfer work outlives the scheduler queues.

        ``postprocess`` parks a finished request in ``deferred_free_blocks``
        and drops it from ``running`` in the same pass, so
        ``Scheduler.is_finished()`` reads "idle" while that request's RDMA
        send or offload save is still in flight. Every busy loop ORs this
        predicate in next to ``is_finished()``; without it the last request's
        completion signals are never polled, its deferred blocks are never
        freed, and its save is never reported.

        Every liveness condition lives here. The loops call this and nothing
        else, so a new kind of pending work only has to be added once.
        """
        # 【译】当KV传输工作比调度器队列活得更久时为True。
        #
        #   postprocess 把一个已结束的请求停放在 deferred_free_blocks 里，并在同一遍里把它从 running 中移除，于是
        #   Scheduler.is_finished() 显示"空闲"，而这个请求的RDMA发送或offload保存仍在进行中。每个busy loop都把这个谓词和
        #   is_finished() 一起做"或"运算；没有它，最后一个请求的完成信号永远不会被轮询，它延迟释放的block永远不会被释放，
        #   它的保存也永远不会被报告。
        #
        #   每一个存活条件都放在这里。循环只调用这个函数、不调用别的，所以新增一种待处理工作只需要在这里加一次。
        # 解释：v0.1.8的Scheduler.is_finished()已经把deferred_free_blocks算进去了，所以这一项在基类循环里通常已被
        #   is_finished覆盖；state store的pin和connector.has_pending_work()仍只由这里判断
        if not self.kv_transfer_enabled:
            return False
        if getattr(self.scheduler, "deferred_free_blocks", None):
            return True
        # Unit-store twin of `deferred_free_blocks`: a state store handed to the
        # worker keeps its PAGE units pinned out of the KV pool until its report
        # settles them or the reclaim ages them out. Both exits run only from
        # `_poll_kv_transfer_progress`, so the loop must keep polling while a
        # pin is outstanding or the units leak until the abandon window. Guarded
        # for the scheduler doubles that implement only the connector surface.
        # 【译】unit-store版的 deferred_free_blocks：交给worker的一次状态存储会把它的PAGE单元钉在KV池之外，直到它的报告
        #   结清它们，或回收机制让它们超期作废。两条出路都只从 _poll_kv_transfer_progress 走，所以只要还有pin未结清，
        #   循环就必须继续轮询，否则这些单元会一直泄漏到放弃窗口。对只实现了connector接口的调度器替身用getattr做保护。
        bm = getattr(self.scheduler, "block_manager", None)
        pending_pins = getattr(bm, "has_pending_state_store_pins", None)
        if pending_pins is not None and pending_pins():
            return True
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not hasattr(connector, "has_pending_work"):
            return False
        return bool(connector.has_pending_work())

    # 【分支】KV传输：本轮不跑forward时推进offload的加载/保存并轮询完成情况；按KV_IDLE_DRAIN_INTERVAL_S（1ms）节流
    def _advance_idle_kv_transfer(self, dispatch_new: bool = True) -> None:
        # No forward batch will run this tick, but offload load/save work may
        # still need to be dispatched or reported back to the scheduler.
        # `dispatch_new=False` on the shutdown drain: report in-flight work
        # back, but start no new transfers (see `_dispatch_idle_offload_work`).
        # 【译】这一拍不会运行forward batch，但offload的加载/保存工作可能仍需要被分发出去，或把结果报告回调度器。
        #   关闭时的排空用 dispatch_new=False：把在途的工作报告回来，但不开始新的传输（见 _dispatch_idle_offload_work）。
        now = time.monotonic()
        if now < self._next_idle_kv_drain:
            return
        self._next_idle_kv_drain = now + KV_IDLE_DRAIN_INTERVAL_S
        self._dispatch_idle_offload_work(dispatch_new=dispatch_new)
        self._poll_kv_transfer_progress()

    # 【分支】KV传输：busy_loop退出后，给在途的KV传输最多KV_SHUTDOWN_DRAIN_TIMEOUT_S（2秒）来报告完成
    def _drain_kv_work_at_exit(self) -> None:
        """Give in-flight KV transfers a bounded window to report back.

        The loop exits as soon as its queues are empty, so a save dispatched
        by the final batch would otherwise be abandoned with its completion
        unrecorded and its blocks still deferred.

        Drains with `dispatch_new=False`: it must let already-dispatched
        transfers finish and report, but must not publish new state loads/stores.
        A fresh store dispatched here spills bytes nothing will read back, and --
        worse -- keeps `has_pending_kv_work()` True, so this very loop could
        manufacture its own work and never converge before the deadline.
        """
        # 【译】给在途的KV传输一个有界的窗口来报告结果。
        #
        #   循环一旦队列为空就会退出，所以最后一个batch分发出去的保存，否则会在完成情况没有被记录、block仍处于延迟释放的
        #   状态下被丢弃。
        #
        #   用 dispatch_new=False 排空：它必须让已经分发的传输完成并报告，但不能发布新的状态加载/存储。在这里分发一次新的
        #   存储，会把没人会读回来的字节溢出出去，而且——更糟的是——会让 has_pending_kv_work() 保持为True，于是这个循环
        #   可能自己制造工作，在截止时间之前永远收敛不了。
        if not self.kv_transfer_enabled:
            return
        deadline = time.monotonic() + KV_SHUTDOWN_DRAIN_TIMEOUT_S
        try:
            while self.has_pending_kv_work():
                if time.monotonic() >= deadline:
                    logger.warning(
                        "%s: KV transfer still pending after %.1fs, exiting anyway",
                        self.label,
                        KV_SHUTDOWN_DRAIN_TIMEOUT_S,
                    )
                    break
                self._advance_idle_kv_transfer(dispatch_new=False)
                time.sleep(KV_IDLE_DRAIN_INTERVAL_S)
        except Exception:
            logger.exception("KV transfer drain during shutdown failed")

    # 【分支】KV传输：向所有worker收集KV传输完成情况（call_func_with_aggregation，非阻塞，有rank没回复时返回None），
    #   交给调度器更新（哪些请求的KV已收到/已发完），第9批
    def _poll_kv_transfer_progress(self) -> None:
        if not self.kv_transfer_enabled:
            return
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        self.scheduler._update_from_kv_xfer_finished(kvoutput)
        # Reclaim any offload save whose completion report never came (LMCache
        # force-unpinned it upstream). Self-throttled, so calling it on every
        # poll is cheap; without it a stalled save hangs the engine forever.
        # 【译】回收那些完成报告一直没来的offload保存（LMCache在上游强制解除了它的pin）。它自己会节流，所以每次轮询都调用
        #   也很便宜；没有它，一次卡住的保存会让engine永远挂起。
        reconcile = getattr(self.scheduler, "_reconcile_stalled_deferred_saves", None)
        if callable(reconcile):
            reconcile()

    # 【分支】动态分块（PP prefill，第9批）：每DYNAMIC_CHUNKING_POLL_STEPS步问一次worker是否已拟合出chunk延迟模型，拟合好就装进调度器
    def _poll_dynamic_chunking_calibration(self) -> None:
        """Install a freshly calibrated chunk latency model, if one is ready.

        The workers time their own prefills, so this is one RPC round trip every
        `DYNAMIC_CHUNKING_POLL_STEPS` steps and nothing on the forward path.

        Called from the PP head's step loop rather than from here: dynamic
        chunking needs more than one stage, and every such deployment runs
        `PPEngineCoreProc`, whose head loop replaces `_process_engine_step`.
        Every other stage had the feature turned off during startup because it
        makes no chunking decisions.
        """
        # 【译】如果一个新校准好的chunk延迟模型已经就绪，就安装它。
        #
        #   worker自己给自己的prefill计时，所以这里只是每 DYNAMIC_CHUNKING_POLL_STEPS 步一次RPC往返，不占用forward路径。
        #
        #   从PP head的step循环里调用，而不是从这里：动态分块需要不止一个stage，而每个这样的部署都运行 PPEngineCoreProc，
        #   它的head循环取代了 _process_engine_step。其他所有stage都在启动时关掉了这个特性，因为它们不做分块决策。
        if not self._dynamic_chunking_enabled:
            return
        self._dynamic_chunking_poll_countdown -= 1
        if self._dynamic_chunking_poll_countdown > 0:
            return
        self._dynamic_chunking_poll_countdown = DYNAMIC_CHUNKING_POLL_STEPS
        fit = self.runner_mgr.call_func("take_dynamic_chunking_fit", wait_out=True)
        coefficients = fit.get("coefficients") if fit else None
        if coefficients is None:
            if fit and fit.get("gave_up"):
                # The workers have stopped timing, so end the sweep too rather
                # than size every other request for a model that never lands.
                # 【译】worker已经停止计时了，所以也结束这次扫描，而不是为一个永远不会到来的模型去确定其他每个请求的大小。
                self._dynamic_chunking_enabled = False
                self.scheduler.abandon_chunk_latency_calibration()
            return
        # The workers stop timing prefills once they have answered with a fit, so
        # there is never a second one to collect - whether the scheduler takes
        # this model or rejects it as not worth acting on.
        # 【译】worker一旦用一个拟合结果作答就停止给prefill计时，所以永远不会有第二个可收——无论调度器是采用这个模型，
        #   还是认为它不值得采用而拒绝。
        self._dynamic_chunking_enabled = False
        try:
            predictor = ChunkSizePredictor.from_coefficients(coefficients)
        except (TypeError, ValueError):
            logger.exception(
                "%s: rejecting dynamic chunking coefficients %r",
                self.label,
                coefficients,
            )
            self.scheduler.abandon_chunk_latency_calibration()
            return
        self.scheduler.install_chunk_latency_model(predictor)

    # 【分支】KV offload（第9批）：空闲时也要把调度器积攒的状态加载/存储发布出去，并把connector元数据交给worker执行
    def _dispatch_idle_offload_work(self, dispatch_new: bool = True) -> None:
        if not self.kv_transfer_enabled:
            return
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not getattr(connector, "is_offload", False):
            return
        # getattr for the same reason as `kv_connector` above: this path is
        # reached with scheduler doubles that implement only the connector
        # surface.
        #
        # `dispatch_new=False` on the shutdown drain: publishing new state
        # loads/stores there hands the connector work whose bytes nothing will
        # read back, and -- worse -- each fresh store keeps `has_pending_kv_work`
        # True, so the drain loop that waits on it manufactures its own exit
        # condition and never converges. Still build/process the meta so
        # already-dispatched transfers finish and report.
        # 【译】用getattr的原因与上面的 kv_connector 相同：到达这条路径的调度器替身只实现了connector接口。
        #
        #   关闭时的排空用 dispatch_new=False：在那里发布新的状态加载/存储，会交给connector一些没人会读回其字节的工作，
        #   而且——更糟的是——每一次新的存储都会让 has_pending_kv_work 保持为True，于是等待它的排空循环会自己制造退出条件、
        #   永远收敛不了。仍然构建/处理元数据，让已经分发的传输完成并报告。
        if dispatch_new:
            for name in ("_publish_state_loads", "_publish_state_stores"):
                publish = getattr(self.scheduler, name, None)
                if publish is not None:
                    publish()
        meta = connector.build_connector_meta()
        if not connector_metadata_has_work(meta):
            return
        self.runner_mgr.call_func("process_kvconnector_output", meta)

    # 把输入线程放进input_queue的新请求一次性全部取出，交给调度器（scheduler.extend -> 进入waiting队列）
    #   返回True表示见到了退出哨兵（status == EXIT_ENGINE），busy_loop据此退出；同一次取出的其他请求不再交给调度器
    #   Queue.get_nowait()：不阻塞地取，队列空时抛queue.Empty（这里先用empty()判断，只有本线程在取，所以不会抛）
    # 对比nano-vllm：nano-vllm的add_request在调用方线程里直接scheduler.add(seq)；ATOM的请求先经socket到输入线程，
    #   再由busy_loop在每轮开头批量取出，所以（基类中）调度器只被busy_loop这一个线程访问，不需要加锁
    def pull_and_process_input_queue(self):
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    logger.debug(f"{self.label}: input_queue get exit engine")
                    return True
                recv_reqs.append(seq)
        if len(recv_reqs) > 0:
            logger.debug(f"{self.label}: put {len(recv_reqs)} reqs to scheduler")
            self.scheduler.extend(recv_reqs)
        return False

    # 输入线程的主体（在__init__里启动的daemon线程）：connect两个DEALER socket，循环poll，按消息类型分发
    #   ADD      -> 装回随帧发来的prompt hash链，记下入队时间，整批放进input_queue
    #   UTILITY  -> (cmd, args)放进utility_queue，置_has_pending_utility，等busy_loop执行
    #   SHUTDOWN -> input_queue放入[退出哨兵]，本线程结束
    # with ExitStack() as stack, zmq.Context() as ctx：ExitStack可以动态登记多个上下文管理器，
    #   with块结束时按相反顺序全部关闭（这里是两个socket），然后再关闭ctx
    def process_input_sockets(self, input_address: str, control_address: str):
        """Input IO thread, serving both the request and control sockets.

        Two sockets, one thread: requests arrive on ``input_address`` and
        control traffic (utility commands, abort, shutdown) on
        ``control_address``. The split exists on the sending side -- it lets
        CoreManager keep the request socket single-writer and therefore
        lock-free -- and both feed the same dispatch below.
        """
        # 【译】输入IO线程，同时服务请求socket和控制socket。
        #
        #   两个socket，一个线程：请求从 input_address 到达，控制流量（工具命令、abort、关闭）从 control_address 到达。
        #   拆分存在于发送端——它让CoreManager可以让请求socket保持单写者，从而不用加锁——两者都汇入下面同一个分发逻辑。
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, bind=False)
            )
            control_socket = stack.enter_context(
                make_zmq_socket(ctx, control_address, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message on each socket - this is required
            # before the front-end ROUTER sockets can send messages
            # back to us.
            # 【译】在每个socket上先发一条初始消息——这是必须的，前端的ROUTER socket之后才能往回给我们发消息。
            # 解释：ROUTER要先收到对端的一条消息才知道它的identity；CoreManager.__init__里的recv_multipart()等的就是这个空帧
            input_socket.send(b"")
            control_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)
            poller.register(control_socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input and control sockets connected")
            alive = True

            while alive:
                for sock, _ in poller.poll():
                    # (RequestType, RequestData)
                    # 【译】（RequestType, RequestData）
                    # 解释：每条消息是pickle后的元组；ADD帧可能还带第3个元素（prompt hash链）
                    obj = sock.recv(copy=False)
                    # 记下收到的时刻，作为请求的入队时间（排队时延等指标，第10批）
                    received_at = time.perf_counter()
                    try:
                        # ADD may carry the seqs' shipped prompt chains third.
                        # 【译】ADD可能把seqs随行的prompt链作为第三个元素带着。
                        # 解释：request_type, reqs, *chains = ...：星号解包，多出来的元素收进列表chains（没有第3个元素时chains=[]）
                        request_type, reqs, *chains = pickle.loads(obj)
                    except Exception:
                        # This thread is the only way requests reach the engine,
                        # so letting it die strands every later request: the
                        # busy loop keeps polling an empty input queue, the
                        # workers idle, and clients wait forever with no error
                        # on any log but this thread's own traceback. Drop the
                        # frame loudly and keep serving.
                        # 【译】这个线程是请求到达engine的唯一途径，所以让它死掉会把之后的每个请求都困住：busy loop一直轮询一个空的输入队列，
                        #   worker闲着，客户端永远等下去，除了这个线程自己的traceback之外哪个日志里都没有错误。大声地丢掉这一帧，继续服务。
                        logger.exception(
                            f"{self.label}: dropping undecodable input frame "
                            f"({len(obj.bytes)} bytes)"
                        )
                        continue
                    if request_type == EngineCoreRequestType.ADD:
                        # 把前端算好的prompt前缀hash链装回各个seq（见engine_core_mgr._add_frame）；
                        #   hash链与前缀缓存是什么见arg_utils.py的enable_prefix_caching字段上方的【科普】
                        attach_shipped_chains(reqs, *chains)
                        for req in reqs:
                            SchedulerMetrics.enqueue(req, received_at=received_at)
                        req_ids = [req.id for req in reqs]
                        logger.debug(
                            f"{self.label}: input get {request_type} {req_ids}"
                        )
                        # 整批放进input_queue，busy_loop下一轮开头取走
                        self.input_queue.put_nowait(reqs)
                    elif request_type == EngineCoreRequestType.UTILITY:
                        cmd = reqs.get("cmd") if isinstance(reqs, dict) else None
                        logger.debug(f"{self.label}: input get UTILITY command: {cmd}")
                        self.utility_queue.put_nowait((cmd, reqs))
                        self._has_pending_utility = True
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(f"{self.label}: input get {request_type}")
                        # 用"退出哨兵"通知busy_loop退出（与正常请求走同一个FIFO队列）
                        self.input_queue.put_nowait([get_exit_sequence()])
                        alive = False
                        reason = request_type
            logger.debug(f"{self.label}: input thread exit due to {reason}")

    # 输出线程的主体：从output_queue取消息，按类型pickle成 (EngineCoreRequestType, data)，经PUSH socket发给CoreManager
    #   ("STREAM", x) / ("READY", x) / ("METRICS", x) / ("UTILITY_RESPONSE", x) -> 换成对应的类型码转发
    #   list[Sequence] -> 去掉退出哨兵后作为ADD（已结束的序列）发出；列表里有退出哨兵时，再发一个SHUTDOWN并结束线程
    #   linger=4000：关闭socket时最多再等4秒把没发完的消息发出去（保证最后的SHUTDOWN能送达）
    #   注意：同一个ADD类型码在两个方向含义不同——CoreManager发来的是新请求，这里发回去的是已结束的序列
    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        # 【译】输出socket的IO线程。
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                # 阻塞等待下一条待发消息
                item = self.output_queue.get()
                if isinstance(item, tuple) and item[0] == "STREAM":
                    # Send stream outputs
                    # 【译】发送流式输出
                    stream_outputs = item[1]
                    obj = pickle.dumps((EngineCoreRequestType.STREAM, stream_outputs))
                    socket.send(obj)
                    continue

                if isinstance(item, tuple) and item[0] == "READY":
                    # Send READY signal to indicate EngineCore is fully initialized
                    # 【译】发送READY信号，表示EngineCore已经完全初始化
                    obj = pickle.dumps((EngineCoreRequestType.READY, item[1]))
                    socket.send(obj)
                    logger.debug(f"{self.label}: sent READY signal")
                    continue

                if isinstance(item, tuple) and item[0] == "METRICS":
                    obj = pickle.dumps((EngineCoreRequestType.METRICS, item[1]))
                    socket.send(obj)
                    continue

                if isinstance(item, tuple) and item[0] == "UTILITY_RESPONSE":
                    # Send utility command response back to CoreManager
                    # 【译】把工具命令的回复发回CoreManager
                    response_data = item[1]
                    serialized_obj = pickle.dumps(
                        (EngineCoreRequestType.UTILITY_RESPONSE, response_data)
                    )
                    socket.send(serialized_obj)
                    continue

                # Regular finished sequences
                # 【译】普通的已结束序列
                seqs = item
                valid_seqs = [
                    seq for seq in seqs if seq.status != SequenceStatus.EXIT_ENGINE
                ]
                num_valid = len(valid_seqs)
                if num_valid > 0:
                    obj = pickle.dumps((EngineCoreRequestType.ADD, valid_seqs))
                    socket.send(obj)
                    logger.info(f"{self.label}: output send {num_valid} reqs")
                # 列表里有退出哨兵（来自_send_engine_dead）：告诉CoreManager本engine要退出了，然后结束本线程
                if len(valid_seqs) != len(seqs):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.debug(
                        f"{self.label}: output send {EngineCoreRequestType.SHUTDOWN}"
                    )
                    break


# DPEngineCoreProc：数据并行（data_parallel_size > 1）时每个DP rank的EngineCore
#   每个DP rank有自己的调度器和请求，但forward内部可能有跨DP rank的集合通信，例如：
#     - 每步一次CPU（gloo）all_gather，交换各rank的token数、bs、是否在prefill等（ForwardMode.decide里的sync_dp_metadata，
#       forward_context.py；DPMetadata.num_tokens_across_dp的那次all_reduce只用在录CUDA Graph等拿不到这张表的地方）
#     - MoE层在 dp×tp 张卡上的all_gather / all-to-all（第7批）
#     - 调度器里的PrefillDelayer在schedule()中做跨DP的all_reduce（第3批）
#   集合通信要求组里每个成员都以同样的次数、同样的顺序调用，否则先到的rank会永远等下去。
#   所以：只要任何一个rank还有活，所有rank都必须一起"走一步"——没活的rank跑一个dummy batch陪跑
#   从单卡的一层forward出发看（例：DP=2、--enable-dp-attention的MoE模型，每个DP rank简化成1张卡、80层）：
#     rank0：本rank的64个decode token ─► 注意力（只读自己请求的KV）─┐                          ┌► 专家FFN ─► 送回 ─► 下一层
#                                                                  ├► MoE：按router结果把token ─┤
#     rank1：本rank的 0个token        ─► 注意力                    ─┘  发给持有对应专家的rank   └► 专家FFN ─► 送回 ─► 下一层
#     （前置：MoE层把FFN换成很多个"专家"FFN，router给每个token只选其中top-k个，如256选8；专家并行把专家分到各卡上，
#      256个专家、DP=2时每个rank放128个。每层先把token发到它选中的专家所在的rank（dispatch），算完再收回（combine），
#      这两次是all-to-all，组里每个rank都必须参加。完整讲解见arg_utils.py的enable_expert_parallel字段上方的【科普】）
#     专家分散在各rank上，rank0的token会有一部分路由到rank1上的专家：rank1即使自己没有请求，也必须参加每一层的这次交换。
#     rank1什么都不做 → rank0停在第0层的交换里等它 → 整个DP组卡死。dummy batch = 让rank1带1个假token
#     （id=-1、token全为0、block_table=[0]，见下面_execute_dummy_batch）也跑完同样的80层、参加同样次数的集合通信，结果丢弃
#
# 时序（DP rank 0有一个请求A，DP rank 1没有请求）：
#
#   DP rank 0（有请求）                                  DP rank 1（空闲）
#   pull_input：A进waiting                              pull_input：无
#   local_unfinished=1                                  local_unfinished=0
#   ══ _sync_dp_state：gloo all_reduce(MAX) [1,0,0] 与 [0,0,0] → 两边都得到 global_has_unfinished=1 ══
#   _process_engine_step()（两边都调用）                  _process_engine_step()
#     schedule() → A的prefill batch                       schedule() → None → 返回False
#     forward(batch) ──┐                                _execute_dummy_batch()：一个假的decode请求
#                      └──── forward内部的跨DP集合通信，两边在这里会合 ────┘
#     postprocess → A的第一个token（流式输出）
#   ……A在decode期间，rank 1每一步都跑dummy……
#   A结束：local_unfinished=0                           local_unfinished=0
#   ══ all_reduce → global_has_unfinished=0；但engines_running还是上一轮的True ══
#   _process_engine_step()→False → dummy              _process_engine_step()→False → dummy    （多走的一步）
#   engines_running=False                               engines_running=False
#   ══ all_reduce → 0，且engines_running=False → 两边都只检查KV收尾，然后continue空转 ══
#
# 关闭也要一致：只有"全局有rank要关闭"且"全局没有未完成请求"时，所有rank才一起退出循环
# 对比nano-vllm：nano-vllm只有单机TP，没有DP，也就没有这层同步
#
# 为什么要DP：TP把一个请求切到更多卡上时，每张卡上的矩阵更窄、每层的all-reduce更多，扩展效率下降；DP让每组卡独立服务
#   不同的请求，吞吐近似线性增长。MoE模型常用"注意力DP + 专家并行"：注意力部分各rank只算自己的请求，MoE层再把所有rank
#   的token汇到一起做专家计算——这就是forward内部跨DP集合通信的来源
# dummy batch的代价：它只有1个token（开MTP时mtp_k+1个），但decode是带宽受限的（见文件开头【2】），1个token的forward
#   照样要把整份权重读一遍，耗时接近一次小batch的真实decode。空闲rank付出这份代价，换来其他rank不被卡死
# 锁步的另一面：每一步的耗时由最慢的rank决定。某个rank在做8k token的prefill时，只做decode的rank会在集合通信处等它
#   几百ms。PrefillDelayer在DP>1时跨rank协调prefill的放行（cost_balance：各rank的prefill代价随同一次all_reduce交换），
#   就是为了减少这种互相等待
class DPEngineCoreProc(EngineCore):
    def __init__(self, config: Config, input_address: str, output_address: str):
        # （下面两行被注释掉的是旧代码：dp_group现在在_init_data_parallel里创建）
        # self.dp_group = config.parallel_config.dp_group
        self.dp_rank = config.parallel_config.data_parallel_rank
        # self.dp_group = config.parallel_config.stateless_init_dp_group()
        # 基类__init__里会调用下面重写的_init_data_parallel建好self.dp_group，然后加载模型、发READY
        super().__init__(config, input_address, output_address)
        # Initialize to True so first iteration reaches all_reduce
        # 【译】初始化为True，让第一轮迭代能走到all_reduce
        # 解释：在当前代码里_sync_dp_state的all_reduce每轮都会执行，与这个标志无关。engines_running的实际作用是：
        #   上一轮全局还有活（或刚启动）时，本轮即使全局没活，所有rank也再一起走一步（dummy batch），见busy_loop
        self.engines_running = True
        # DP同步失败（通常是有rank已经退出）后置True，之后不再做all_reduce，直接按"要关闭"处理
        self._shutting_down = False

        # DP>1时在这里挂PrefillDelayer（此时self.scheduler才存在），用DP的gloo组做跨rank协调
        self._init_prefill_delayer(config, self.dp_group)

    # 重写基类钩子：校验DP rank，并建立DP rank之间的gloo（CPU）进程组self.dp_group
    #   在EngineCore.__init__里、构造ModelRunner之前调用；stateless_init_dp_group会阻塞到所有DP rank都连上master（集合点）
    #   这个组只连接各EngineCore进程（在CPU上同步小张量），与ModelRunner里做GPU通信的DP组是两套
    def _init_data_parallel(self, config: Config):
        dp_rank = config.parallel_config.data_parallel_rank
        dp_size = config.parallel_config.data_parallel_size
        local_dp_rank = config.parallel_config.data_parallel_rank_local
        local_dp_size = config.parallel_config.data_parallel_size_local

        assert dp_size > 1
        assert local_dp_rank is not None
        # Two independent bounds. The old chained form (local <= global) held
        # only by coincidence on one node: the second node of a 2x4 run has
        # local ranks 0..3 against global ranks 4..7.
        # 【译】两个相互独立的范围检查。旧的链式写法（local <= global）只在单节点上碰巧成立：一个2x4运行的第二个节点上，
        #   本地rank是0..3，而全局rank是4..7。
        assert 0 <= dp_rank < dp_size, f"dp_rank={dp_rank} outside [0,{dp_size})"
        assert (
            0 <= local_dp_rank < local_dp_size
        ), f"local_dp_rank={local_dp_rank} outside [0,{local_dp_size})"

        self.dp_rank = dp_rank
        self.dp_group = config.parallel_config.stateless_init_dp_group()
        # NOTE: PrefillDelayer attachment lives in __init__ (after
        # super().__init__ creates self.scheduler) — not here. This
        # function runs during super().__init__, before self.scheduler
        # exists.
        # 【译】注意：PrefillDelayer的挂接在 __init__ 里（在 super().__init__ 创建 self.scheduler 之后）——不在这里。
        #   这个函数在 super().__init__ 期间运行，那时 self.scheduler 还不存在。

    # 先走基类的退出流程（关ModelRunner、通知CoreManager），再销毁DP的gloo进程组
    #   dp_group := ...：海象运算符，赋值的同时返回这个值，供if判断
    def exit(self):
        super().exit()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    # DP版主循环：比基类多了"每轮先跨DP同步状态"和"没活就跑dummy batch"，保证所有DP rank步调一致（见类上方时序图）
    #   每一轮：
    #     ①~④ 与基类相同：控制命令、指标、吞吐心跳、取新请求
    #     ⑤ local_unfinished：本rank有未完成请求，且没被RL卸载/权重同步挡住
    #     ⑥ _sync_dp_state：一次all_reduce(MAX)得到三个全局"或"：有未完成 / 要关闭 / 已卸载
    #     ⑦ 全局要关闭且全局没活 -> 所有rank一起退出
    #     ⑧ 任一rank在RL卸载/同步 -> 所有rank一起sleep跳过（否则有的rank进forward、有的不进，集合通信会卡住）
    #     ⑨ 全局没活且上一轮也没活 -> 只推进本地KV传输，continue
    #     ⑩ 否则所有rank都走一步：有batch就forward，没有就dummy batch
    def busy_loop(self):
        shutdown = False
        metrics_interval = envs.ATOM_METRICS_UPDATE_INTERVAL_S
        next_metrics_push = 0.0
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                now = time.monotonic()
                if now >= next_metrics_push:
                    next_metrics_push = now + metrics_interval
                    self.utility_handler.push_metrics()
                self.scheduler.heartbeat_throughput(now)
                shutdown = shutdown or self.pull_and_process_input_queue()
                # ⑤ 本rank是否需要执行
                local_unfinished = not self.scheduler.is_finished() and not (
                    self._is_rl_weights_offloaded or self._weight_sync_fenced
                )

                # ⑥ 跨DP同步（每轮一次，所有rank都必须调用）
                global_has_unfinished, global_shutdown, global_offloaded = (
                    self._sync_dp_state(
                        local_unfinished,
                        shutdown,
                        self._is_rl_weights_offloaded or self._weight_sync_fenced,
                    )
                )

                # ⑦ 某个rank还有请求时，要关闭的rank也得继续陪跑
                if global_shutdown and not global_has_unfinished:
                    logger.info(
                        f"{self.label}: All DP ranks agreed to shutdown, exiting busy_loop"
                    )
                    break

                # ⑧ 任一rank在RL卸载/权重同步：所有rank一起sleep并跳过本轮
                if global_offloaded:
                    time.sleep(0.01)
                    continue

                # ⑨ 全局空闲（本轮没活，上一轮也没活）
                if not global_has_unfinished and not self.engines_running:
                    # （进入这个分支时engines_running已经是False，这一行不改变状态）
                    self.engines_running = False
                    if self.has_pending_kv_work():
                        # Local RPCs only. Anything that reaches schedule()
                        # would run the delayer's cross-DP all_reduce off
                        # lockstep, so the idle drain must stay off that path.
                        # 【译】只做本地RPC。任何会走到 schedule() 的东西都会让delayer的跨DP all_reduce脱离锁步运行，
                        #   所以空闲时的排空必须远离那条路径。
                        self._advance_idle_kv_transfer()
                    continue

                # ⑩ 全局有活：本rank有batch就正常执行一步；没有（返回False）就跑dummy batch参与集合通信
                executed = self._process_engine_step()
                if not executed:
                    self._execute_dummy_batch()

                # 记下本轮全局是否有活，决定下一轮全局空闲时是否还要再走一步
                self.engines_running = global_has_unfinished
        finally:
            # Isolate the final publish so a publisher hiccup cannot skip
            # shutdown_kv_events() (which closes the sender thread/socket).
            # 【译】把最后一次发布隔离开，这样发布器的一次小故障不会跳过 shutdown_kv_events()（它会关闭发送线程/socket）。
            self._drain_kv_work_at_exit()
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during DP shutdown failed")
            self.scheduler.shutdown_kv_events()

    # 在所有TP worker上跑一个假的decode batch（ModelRunner.dummy_execution，第5批）并等它完成：
    #   一个id=-1的假序列，token全为0，block_table=[0]；只为了让本rank参与forward里的跨DP集合通信，结果丢弃
    #   开MTP（投机解码）时假序列有mtp_k+1个token，与真实decode的形状一致
    def _execute_dummy_batch(self):
        return self.runner_mgr.call_func("dummy_execution", wait_out=True)

    # 在DP的gloo组上做一次all_reduce(MAX)，把三个本地布尔值变成全局"或"：
    #   [本rank有未完成请求, 本rank要关闭, 本rank已卸载权重] -> 任一rank为1，结果就是1
    #   例：rank0=[1,0,0]、rank1=[0,1,0] -> 两边都得到[1,1,0]：还有活，所以rank1虽然要关闭，也继续陪跑
    #   all_reduce失败（通常是其他rank已经退出）：置_shutting_down，之后直接返回"要关闭"，不再通信
    #   device="cpu"：gloo后端在CPU张量上通信，不占GPU
    def _sync_dp_state(
        self,
        local_has_unfinished: bool,
        local_shutdown: bool = False,
        local_offloaded: bool = False,
    ) -> tuple[bool, bool, bool]:
        if self._shutting_down:
            return local_has_unfinished, True, local_offloaded

        try:
            state_tensor = torch.tensor(
                [
                    1 if local_has_unfinished else 0,
                    1 if local_shutdown else 0,
                    1 if local_offloaded else 0,
                ],
                dtype=torch.int64,
                device="cpu",
            )
            torch.distributed.all_reduce(
                state_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_has_unfinished = state_tensor[0].item() == 1
            global_shutdown = state_tensor[1].item() == 1
            global_offloaded = state_tensor[2].item() == 1
            return global_has_unfinished, global_shutdown, global_offloaded
        except RuntimeError as e:
            logger.warning(f"{self.label}: _sync_dp_state failed: {e}")
            self._shutting_down = True
            return local_has_unfinished, True, local_offloaded

    # 只同步"是否要关闭"的旧辅助函数；v0.1.8里没有调用方（busy_loop改用_sync_dp_state一次同步三个值）
    def _sync_shutdown_state(self, local_should_shutdown: bool) -> bool:
        try:
            tensor = torch.tensor(
                [local_should_shutdown], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_should_shutdown = bool(tensor.item())
            return global_should_shutdown
        except RuntimeError as e:
            # If all_reduce fails, it means other ranks are shutting down
            # 【译】如果all_reduce失败，说明其他rank正在关闭
            logger.warning(
                f"{self.label}: Shutdown sync failed, assuming shutdown: {e}"
            )
            return True

    # 只同步"是否有未完成请求"的旧辅助函数（调用ParallelConfig.has_unfinished_dp）；v0.1.8里同样没有调用方
    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        if self._shutting_down:
            logger.info(f"{self.label}: Skipping DP sync during shutdown")
            return local_unfinished
        try:
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        except RuntimeError as e:
            # Handle case where other ranks have already shut down
            # 【译】处理其他rank已经关闭的情况
            logger.warning(f"{self.label}: DP sync failed during shutdown: {e}")
            return local_unfinished


# ---------------------------------------------------------------------------
# Disaggregated prefill/decode engine cores
# ---------------------------------------------------------------------------
# 【译】分离式的prefill/decode engine core
# 【分支】PD分离（同GPU的RapidServe，由DisaggCoreManager启动），见第9批
# 原理：prefill计算受限、decode带宽受限（见文件开头【2】），放在同一个step循环里会互相干扰——一次长prefill让所有decode停顿。
#   RapidServe把两者拆成两个EngineCore进程，在同一组GPU上并发运行：各自在自己的stream（见文件开头【3】的前置）上发kernel，默认两边都能用整卡的
#   计算单元（CU），disagg_constrained=True时才按CU屏蔽（CU mask）划分（见下面"代价①"）；权重和KV cache通过CUDA IPC
#   共用同一份显存（prefill直接写decode要读的KV缓冲区），不用拷贝KV
#
# 进程与通道（同一组GPU上的两个EngineCore；地址由DisaggCoreManager.__init__预先生成，写进两份deepcopy的config）：
#
#   DisaggCoreManager ──ADD（同一批seq，扇出给两边）──┬──────────────────────────────┐
#                                                     ▼                              ▼
#   ┌───────────── PrefillEngineCore（engine 0）─────────────┐      ┌───────────── DecodeEngineCore（engine 1）─────────────┐
#   │ PrefillScheduler：没有BlockManager                      │      │ DecodeScheduler：拥有BlockManager                      │
#   │ RapidServeModelRunner.prefill_forward（强制eager）       │      │ RapidServeModelRunner.forward（CUDA Graph）            │
#   │ 拥有真正的权重和KV cache显存                            │      │ 权重和KV cache都是从prefill导入的IPC视图                │
#   └──────────────────────────────────────────────────────────┘      └──────────────────────────────────────────────────────────┘
#        PULL(bind d2p)  ◄══════ BlockAssignment / ABORT ═══════════  PUSH(connect d2p)
#        PUSH(connect p2d) ═════ PrefillDone(第一个token) ══════════► PULL(bind p2d)
#                                                                      输出线程 ──已结束序列/STREAM──► CoreManager（只有decode产出结果）
#
# 启动握手（两轮，都在各自READY之前）：
#   Prefill                                                   Decode
#   load_model（真实权重）                                     _build_and_load_model：参数建在meta设备上（不占显存）
#   _post_model_load_hook：                                    _post_model_load_hook：
#     export_model_weight_ipc_handles                            PULL connect weight_ipc，阻塞recv
#     PUSH bind weight_ipc ──── 权重句柄文件路径 ────────────►   import_model_weight_ipc_handles（参数换成prefill显存的视图）
#     PULL connect weight_ack，阻塞recv                          sleep(2) 等分配器回收
#                         ◄──────────────── "ok" ──────────────  PUSH bind weight_ack
#   get_num_blocks / allocate_kv_cache（显存只有一份权重）       get_num_blocks→0、allocate_kv_cache跳过（RapidServeModelRunner）
#   _send_ready_signal：                                       （_ready_deferred：此时不发READY）
#     export_kv_cache_ipc_handle                                 PULL connect kvcache_ipc，阻塞recv
#     PUSH bind kvcache_ipc ── {句柄, num_kvcache_blocks} ───►   import_kv_cache_ipc_handle → 回填num_kvcache_blocks
#     READY ──► CoreManager                                      capture_cudagraph → DecodeScheduler → gc.freeze → READY ──► CoreManager
#   （下面PrefillEngineCore / DecodeEngineCore的_post_model_load_hook的docstring都说decode"释放自己的那份权重"：v0.1.8的decode其实从未分配权重，
#     参数一开始就在meta上，见model_runner.py的RapidServeModelRunner._build_and_load_model）
#
# 一个请求的时间线：
#   decode：ADD到达 → extend → allocate_waiting分配block → BlockAssignment ──►
#   prefill：ADD到达（进waiting）；收到BlockAssignment → _apply_pending_assignments填好block_table → schedule
#            → prefill_forward（KV直接写进共享的KV池）→ PrefillDone(第一个token) ──►
#   decode：on_prefill_done → 序列进running → 之后每步正常decode，流式输出和结束都由decode发给CoreManager
#
# 适用场景：同一台机器、同一组GPU，既要高吞吐又要稳定的ITL——长prompt的prefill不再插进decode的step里，
#   decode的TPOT不会被几百ms的prefill打出尖峰（见文件开头【2】）
# 好处：不需要跨节点KV传输（对比connector式PD分离，第9批）：KV只有一份、权重只有一份，零拷贝
# 代价：① 两个进程争同一组GPU的算力：默认（disagg_constrained=False）只是各用独立的stream，互相抢占计算单元；
#     disagg_constrained=True时才用按CU屏蔽的stream：decode经一块4字节的共享内存报出本步的decode token数，
#     decode token≥64且有prefill在排队时两边各分一半CU（scheduler.py的_optimal_cu_fraction，目前只有0.5一档），否则仍用整卡stream：
#       CU编号  0 ──────────── total/2 ──────────── total
#               └── prefill的stream ──┘└── decode的stream ──┘   （一张卡有几百个CU；kernel的线程块只会被派到mask允许的CU上）
#     ② prefill的KV预算要多留 4×2%=8% 余量（_kv_budget_extra_reserve）；
#   ③ prefill一侧不用CUDA Graph，两侧都不挂PrefillDelayer（_init_prefill_delayer遇到enable_rapidserve直接返回）；
#   ④ 启动多两轮握手（含decode侧固定的2秒sleep）


# PrefillEngineCore：只做prefill的engine（DisaggCoreManager的engine 0）
#   不分配KV block（PrefillScheduler没有BlockManager），block由decode分配后通过BlockAssignment告诉它；
#   每次prefill forward之后给decode发PrefillDone（带上第一个采样token）
#   构造顺序：记下PD分离用的地址（基类__init__中途会调用下面重写的钩子）-> 基类__init__ -> 把调度器换成PrefillScheduler
class PrefillEngineCore(EngineCore):
    """Disaggregated prefill instance.

    Responsibilities:
    - Runs only prefill forward passes (enforce_eager=True, no CUDA graphs).
    - Receives BlockAssignment messages from DecodeEngineCore over a direct
      ZMQ PULL socket (disagg_d2p_addr) and populates seq.block_table before
      calling schedule().
    - Runs each forward pass on a dedicated CUDA stream (_prefill_stream).
      Records a CUDA event immediately after each forward; a separate watcher
      thread polls the events and sends PrefillDone only after the GPU has
      committed all KV writes — ensuring decode never reads a partial cache.
    - Imports the KV cache tensor from decode via CUDA IPC at startup so that
      its forward pass writes directly into decode's GPU buffer.
    - Does NOT sample tokens; completed sequences are discarded here.
    """

    # 【译】分离式的prefill实例。
    #
    #   职责：
    #   - 只运行prefill的forward（enforce_eager=True，不用CUDA graph）。
    #   - 通过一个直接的ZMQ PULL socket（disagg_d2p_addr）接收DecodeEngineCore发来的BlockAssignment消息，
    #     在调用 schedule() 之前填好 seq.block_table。
    #   - 每次forward都在一个专用的CUDA stream（_prefill_stream）上运行。每次forward之后立刻记录一个CUDA event；一个
    #     单独的观察线程轮询这些event，只在GPU提交了所有KV写入之后才发送PrefillDone——确保decode永远不会读到不完整的cache。
    #   - 启动时通过CUDA IPC从decode导入KV cache张量，这样它的forward直接写进decode的GPU缓冲区。
    #   - 不采样token；完成的序列在这里被丢弃。
    # 解释：与v0.1.8代码不符的几处：KV cache是prefill导出、decode导入（见_send_ready_signal和DecodeEngineCore.__init__）；
    #   没有观察线程，ModelRunner.prefill_forward在返回前对stream做synchronize，PrefillDone随后在_process_engine_step里发送；
    #   prefill_forward会返回每个seq采样出的第一个token，随PrefillDone交给decode
    def __init__(self, config: Config, input_address: str, output_address: str):

        # Force eager mode — no CUDA graph capture on the prefill side.
        # 【译】强制eager模式——prefill一侧不捕获CUDA graph。
        config.enforce_eager = True

        # 【译】（行尾注释）PULL：接收BlockAssignment
        self._disagg_d2p_addr = config.disagg_d2p_addr  # PULL: receive BlockAssignment
        # 【译】（行尾注释）PUSH：发送PrefillDone
        self._disagg_p2d_addr = config.disagg_p2d_addr  # PUSH: send PrefillDone
        self._disagg_weight_ipc_addr = config.disagg_weight_ipc_addr
        self._disagg_weight_ack_addr = config.disagg_weight_ack_addr
        self._disagg_kvcache_ipc_addr = config.disagg_kvcache_ipc_addr

        # Maps seq_id → BlockAssignment, populated by the receiver thread.
        # 【译】seq_id → BlockAssignment，由接收线程填充。
        self._pending_assignments: dict = {}

        # ZMQ context for disagg sockets (separate from the main engine sockets).
        # 【译】分离模式socket用的ZMQ上下文（与engine主socket的上下文分开）。
        self._disagg_ctx = zmq.Context()

        # Store config reference so _send_ready_signal can read num_kvcache_blocks
        # after EngineCore.__init__ sets it.
        # 【译】保存config的引用，这样 _send_ready_signal 可以在 EngineCore.__init__ 设好 num_kvcache_blocks 之后读取它。
        self._config = config

        super().__init__(config, input_address, output_address)
        # Replace the base Scheduler created by EngineCore.__init__ with
        # PrefillScheduler, which has no BlockManager and only schedules
        # sequences that already have a block_table from decode. The base
        # Scheduler may have started a KV event publisher with bound sockets;
        # prefill never publishes, so release it instead of leaking a thread
        # and an endpoint for the life of the process.
        # 【译】把 EngineCore.__init__ 创建的基础Scheduler替换成PrefillScheduler：它没有BlockManager，只调度已经从decode拿到
        #   block_table的序列。基础Scheduler可能已经启动了一个绑定了socket的KV事件发布器；prefill从不发布，所以释放它，
        #   而不是让一个线程和一个端点在进程的整个生命周期里泄漏。
        if self.scheduler is not None:
            self.scheduler.shutdown_kv_events()
        self.scheduler = PrefillScheduler(
            config, disagg_cu_shm_name=config.disagg_cu_shm_name
        )
        self.utility_handler.scheduler = self.scheduler

    # 启动握手第1轮（prefill侧）：导出权重的CUDA IPC句柄 -> PUSH给decode -> 等decode的ACK
    def _post_model_load_hook(self):
        """Round 1 bootstrap: export weights → send to decode → wait for ACK.

        Called after model is loaded but before get_num_blocks/allocate_kv_cache.
        Decode receives the handles, imports them, frees its own copy, and sends
        an ACK.  Only after the ACK does prefill proceed to measure free GPU memory
        and allocate KV cache — ensuring decode's weight tensors are already freed
        so prefill sees the full available memory budget.
        """
        # 【译】启动握手第1轮：导出权重 → 发给decode → 等待ACK。
        #
        #   在模型加载之后、get_num_blocks/allocate_kv_cache之前调用。decode收到句柄、导入它们、释放自己的那份拷贝，
        #   然后发送ACK。只有在ACK之后，prefill才去测量空闲显存并分配KV cache——确保decode的权重张量已经被释放，
        #   这样prefill看到的是全部可用的显存预算。
        logger.info("PrefillEngineCore: exporting weight IPC handles...")
        weight_handles = self.runner_mgr.call_func(
            "export_model_weight_ipc_handles", wait_out=True
        )
        weight_payload = pickle.dumps(weight_handles)
        logger.info(
            f"PrefillEngineCore: sending weight handles ({len(weight_payload)} bytes)..."
        )
        with self._disagg_ctx.socket(zmq.PUSH) as w_sock:
            w_sock.bind(self._disagg_weight_ipc_addr)
            w_sock.send(weight_payload)
        logger.info("PrefillEngineCore: weight handles sent, waiting for ACK...")
        with self._disagg_ctx.socket(zmq.PULL) as ack_sock:
            ack_sock.connect(self._disagg_weight_ack_addr)
            ack_sock.recv()
        logger.info("PrefillEngineCore: weight ACK received — decode weights freed")

    # 启动握手第2轮（prefill侧）：导出KV cache的IPC句柄和block数 -> PUSH给decode -> 再发正常的READY
    def _send_ready_signal(self):
        """Round 2 bootstrap: export kvcache handle → send to decode → emit READY."""
        # 【译】启动握手第2轮：导出kvcache句柄 → 发给decode → 发出READY。
        logger.info("PrefillEngineCore: exporting kvcache IPC handle...")
        kvcache_args = self.runner_mgr.call_func(
            "export_kv_cache_ipc_handle", wait_out=True
        )
        kvcache_bundle = pickle.dumps(
            {
                "kvcache_args": kvcache_args,
                "num_kvcache_blocks": self._config.num_kvcache_blocks,
            }
        )
        logger.info(
            f"PrefillEngineCore: sending kvcache bundle ({len(kvcache_bundle)} bytes)..."
        )
        # Keep socket alive on self until exit() so linger doesn't block shutdown.
        # 【译】把socket保留在self上直到 exit()，这样linger不会阻塞关闭。
        self._bootstrap_push_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._bootstrap_push_sock.bind(self._disagg_kvcache_ipc_addr)
        self._bootstrap_push_sock.send(kvcache_bundle)
        logger.info("PrefillEngineCore: kvcache bundle sent")
        super()._send_ready_signal()

    # run_engine在构造完成后调用：建prefill stream池，建与decode之间的两条运行期通道（发PrefillDone、收BlockAssignment）
    def _init_disagg(self):
        """Start runtime coordination sockets and threads.

        Called from run_engine() after super().__init__().  The kvcache+weight
        bootstrap was already sent inside _send_ready_signal(), so _init_disagg
        only needs to set up the per-request BlockAssignment/PrefillDone channel.
        """
        # 【译】启动运行期的协调socket和线程。
        #
        #   在 run_engine() 里、super().__init__() 之后调用。kvcache+权重的启动握手已经在 _send_ready_signal() 里发出了，
        #   所以 _init_disagg 只需要建立每个请求的BlockAssignment/PrefillDone通道。
        # # --- Create pool of CU-masked CUDA streams for prefill ---
        # 【译】--- 创建按CU屏蔽的CUDA stream池，用于prefill ---
        # 解释：只有disagg_constrained=True时池里才有按CU屏蔽的stream（目前只有0.5一档）；默认只建一条整卡stream，
        #   见PrefillEngineCore类上方RapidServe说明里的"代价①"
        logger.info("PrefillEngineCore: creating prefill stream pool...")
        self.runner_mgr.call_func("create_prefill_stream_pool", wait_out=True)
        logger.info("PrefillEngineCore: prefill stream pool created")

        # --- Open the PUSH socket to send PrefillDone messages ---
        # Prefill connects (not binds) so decode's PULL bind is ready first,
        # preventing messages from being dropped before decode connects.
        # 【译】--- 打开PUSH socket，用于发送PrefillDone消息 ---
        #   prefill connect（而不是bind），这样decode的PULL bind先就绪，防止消息在decode连上之前被丢掉。
        self._p2d_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._p2d_sock.connect(self._disagg_p2d_addr)

        # --- Start thread to receive BlockAssignment from decode ---
        # Prefill binds so decode's PUSH connect finds a ready socket.
        # 【译】--- 启动线程接收decode发来的BlockAssignment ---
        #   prefill bind，这样decode的PUSH connect能找到一个已就绪的socket。
        self._assignment_sock = self._disagg_ctx.socket(zmq.PULL)
        self._assignment_sock.bind(self._disagg_d2p_addr)
        self._assignment_thread = threading.Thread(
            target=self._recv_block_assignments,
            daemon=True,
            name="PrefillEngineCore-AssignmentRecv",
        )
        self._assignment_thread.start()

    # 后台线程：从decode收BlockAssignment存进_pending_assignments（收到ABORT则删除，并从waiting里移除）
    def _recv_block_assignments(self):
        """Background thread: pulls BlockAssignment messages from decode."""
        # 【译】后台线程：从decode拉取BlockAssignment消息。
        from atom.model_engine.disagg_types import BlockAssignment, DisaggMsgType

        sock = self._assignment_sock
        while True:
            try:
                raw = sock.recv()
            except zmq.error.ContextTerminated:
                break
            msg_type, payload = pickle.loads(raw)
            if msg_type == DisaggMsgType.BLOCK_ASSIGNMENT:
                assignment: BlockAssignment = payload
                with self.scheduler._pending_lock:
                    self._pending_assignments[assignment.seq_id] = assignment
            elif msg_type == DisaggMsgType.ABORT:
                seq_id = payload
                with self.scheduler._pending_lock:
                    self._pending_assignments.pop(seq_id, None)
                # Also remove from waiting queue if present.
                # 【译】如果它在等待队列里，也一并移除。
                self.scheduler.waiting = type(self.scheduler.waiting)(
                    s for s in self.scheduler.waiting if s.id != seq_id
                )

    # 每步schedule()之前：把收到的block分配写进waiting里对应seq的block_table和num_cached_tokens
    def _apply_pending_assignments(self):
        """Copy block assignments received from decode into waiting sequences.

        Must be called at the top of each engine step before schedule().
        """
        # 【译】把从decode收到的block分配复制进等待中的序列。
        #
        #   必须在每个engine step的开头、schedule()之前调用。
        with self.scheduler._pending_lock:
            if not self._pending_assignments:
                return
            for seq in self.scheduler.waiting:
                if seq.id in self._pending_assignments:
                    assignment = self._pending_assignments.pop(seq.id)
                    seq.block_table = new_block_table(assignment.block_table)
                    seq.num_cached_tokens = assignment.num_cached_tokens

    # prefill版的一步：应用block分配 -> schedule -> prefill_forward（返回每个seq采样出的第一个token）
    #   -> 对每个seq给decode发PrefillDone -> 把这些seq标为结束并移出running（prefill这边不产生输出）
    def _process_engine_step(self):
        from atom.model_engine.disagg_types import DisaggMsgType, PrefillDone

        self._apply_pending_assignments()
        if not self.scheduler.has_requests():
            return False
        result = self.scheduler.schedule()
        if result is None:
            return False
        scheduled_batch, seqs = result
        if scheduled_batch is None:
            return False

        # Run on the dedicated prefill stream; returns sampled token IDs (one per seq).
        # 【译】在专用的prefill stream上运行；返回采样出的token ID（每个seq一个）。
        self.scheduler.metrics.record_forward(scheduled_batch, seqs)
        t0 = time.perf_counter()
        sampled_token_ids = self.runner_mgr.call_func(
            "prefill_forward", scheduled_batch, wait_out=True
        )
        iter_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"prefill iter {iter_ms:.2f}ms | "
            f"reqs={scheduled_batch.total_seqs_num} | "
            f"tokens={scheduled_batch.total_tokens_num}"
        )

        # Notify decode that prefill is done, including the first generated token
        # so decode can append it before its first decode step.
        # 【译】通知decode这次prefill已完成，并带上第一个生成的token，这样decode可以在它的第一个decode step之前把它追加上。
        for seq_id, num_tokens, token_id in zip(
            scheduled_batch.req_ids,
            scheduled_batch.num_scheduled_tokens,
            sampled_token_ids,
        ):
            done = PrefillDone(
                seq_id=seq_id,
                num_tokens_computed=int(num_tokens),
                sampled_token_id=int(token_id),
            )
            self._p2d_sock.send(pickle.dumps((DisaggMsgType.PREFILL_DONE, done)))

        # Remove completed sequences — prefill produces no output tokens.
        # 【译】移除已完成的序列——prefill不产生输出token。
        for seq in list(seqs.values()):
            seq.status = SequenceStatus.FINISHED
            try:
                self.scheduler.running.remove(seq)
            except ValueError:
                pass
        return True

    # PrefillEngineCore进程的入口（DisaggCoreManager里Process的target）：构造 -> _init_disagg -> busy_loop（基类的）-> exit
    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        EngineCore._setup_engine_process("PrefillEngineCore")
        engine = None
        try:
            engine = PrefillEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception:
            logger.exception("PrefillEngineCore.run_engine failed")
            raise
        finally:
            if engine is not None:
                engine.exit()

    # 基类退出之后，关闭握手用的PUSH socket并销毁PD分离用的ZMQ上下文
    def exit(self):
        super().exit()
        try:
            if hasattr(self, "_bootstrap_push_sock"):
                self._bootstrap_push_sock.close(linger=0)
        except Exception:
            logger.debug("bootstrap socket close failed", exc_info=True)
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            logger.debug("disagg context destroy failed", exc_info=True)


# DecodeEngineCore：负责decode的engine（DisaggCoreManager的engine 1），拥有BlockManager
#   新请求到达时分配block并告诉prefill；收到PrefillDone后把序列放进running做decode；所有结果都由它产出
#   构造顺序：记下地址 -> 基类__init__（READY被推迟；期间_post_model_load_hook做权重握手）-> 收prefill的KV bundle并导入
#     -> 捕获CUDA graph -> 建DecodeScheduler -> 发READY
class DecodeEngineCore(EngineCore):
    """Disaggregated decode instance.

    Responsibilities:
    - Owns the BlockManager and the KV cache tensor.
    - On new request arrival: allocates KV blocks, sends BlockAssignment to
      prefill, and holds the sequence in prefill_pending until PrefillDone
      arrives.
    - PrefillDone signals move the sequence into the decode scheduler's
      running queue.
    - Exports the kv_cache IPC handle to prefill at startup.
    - Runs normal CUDA-graph-accelerated decode.
    """

    # 【译】分离式的decode实例。
    #
    #   职责：
    #   - 拥有BlockManager和KV cache张量。
    #   - 新请求到达时：分配KV block，给prefill发送BlockAssignment，并把序列放在 prefill_pending 里，直到PrefillDone到达。
    #   - PrefillDone信号把序列移入decode调度器的running队列。
    #   - 启动时把kv_cache的IPC句柄导出给prefill。
    #   - 运行正常的、用CUDA graph加速的decode。
    # 解释：v0.1.8里KV cache的IPC句柄是prefill导出、decode导入（见下面__init__里的"Round 2"）
    def __init__(self, config: Config, input_address: str, output_address: str):
        # 【译】（行尾注释）PUSH：发送BlockAssignment
        self._disagg_d2p_addr = config.disagg_d2p_addr  # PUSH: send BlockAssignment
        # 【译】（行尾注释）PULL：接收PrefillDone
        self._disagg_p2d_addr = config.disagg_p2d_addr  # PULL: receive PrefillDone
        self._disagg_weight_ipc_addr = config.disagg_weight_ipc_addr
        self._disagg_weight_ack_addr = config.disagg_weight_ack_addr
        self._disagg_kvcache_ipc_addr = config.disagg_kvcache_ipc_addr

        self._disagg_ctx = zmq.Context()

        # Store config so we can read/update num_kvcache_blocks after import.
        # 【译】保存config，这样导入之后可以读/更新 num_kvcache_blocks。
        self._config = config

        # Suppress _send_ready_signal during super().__init__() — we send the
        # real READY only after kvcache IPC import and cudagraph capture.
        # 【译】在 super().__init__() 期间抑制 _send_ready_signal——只有在kvcache IPC导入和cudagraph捕获之后才发送真正的READY。
        self._ready_deferred = True
        super().__init__(config, input_address, output_address)
        self._ready_deferred = False

        # --- Round 2: receive kvcache bundle from prefill ---
        # 【译】--- 第2轮：从prefill接收kvcache bundle ---
        logger.info("DecodeEngineCore: waiting for kvcache bundle from prefill...")
        with self._disagg_ctx.socket(zmq.PULL) as sock:
            sock.connect(self._disagg_kvcache_ipc_addr)
            raw = sock.recv()
        bundle = pickle.loads(raw)
        num_kvcache_blocks = bundle["num_kvcache_blocks"]
        logger.info(
            f"DecodeEngineCore: received kvcache bundle ({num_kvcache_blocks} blocks)"
        )

        # --- Import kvcache — sets self.kv_cache + binds to attention modules ---
        # 【译】--- 导入kvcache——设置 self.kv_cache 并绑定到各注意力模块 ---
        self.runner_mgr.call_func(
            "import_kv_cache_ipc_handle",
            bundle["kvcache_args"],
            num_kvcache_blocks,
            wait_out=True,
        )
        logger.info("DecodeEngineCore: kvcache IPC import complete")

        # --- Capture CUDA graphs now that kvcache is real ---
        # 【译】--- 既然kvcache已经是真的了，现在捕获CUDA graph ---
        # 解释：这里同样把收到的block数回填进config.num_kvcache_blocks（对应基类__init__里的回填）
        config.num_kvcache_blocks = num_kvcache_blocks

        if not config.enforce_eager:
            cap_cost, bs, _ = self.runner_mgr.call_func(
                "capture_cudagraph", wait_out=True
            )
            logger.info(
                f"DecodeEngineCore: cudagraph capture{bs} cost: {cap_cost:.2f}s"
            )

        # --- Create DecodeScheduler now that num_kvcache_blocks is set ---
        # 【译】--- 既然 num_kvcache_blocks 已经设好，现在创建DecodeScheduler ---
        self.scheduler = DecodeScheduler(
            config,
            disagg_cu_shm_name=config.disagg_cu_shm_name,
            state_runtime=self.state_runtime,
        )
        # The block pool exists only now, so the freeze inside
        # `super().__init__()` ran before there was one to take. Safe to repeat:
        # `_ready_deferred` held READY back, so nothing has been admitted yet
        # and `gc.freeze()` is additive.
        # 【译】block池直到现在才存在，所以 super().__init__() 里那次冻结运行时还没有它可冻。重复执行是安全的：
        #   _ready_deferred 压住了READY，所以还没有任何请求被接纳，而且 gc.freeze() 是累加的。
        self._freeze_after_startup()

        # EngineUtilityHandler was built in super().__init__() with scheduler=None
        # (decode defers scheduler creation); wire the real one in for MTP stats.
        # 【译】EngineUtilityHandler 是在 super().__init__() 里用 scheduler=None 构建的（decode推迟创建调度器）；
        #   把真正的调度器接进去，供MTP统计使用。
        self.utility_handler.scheduler = self.scheduler

        # --- Now truly ready ---
        # 【译】--- 现在真正就绪了 ---
        super()._send_ready_signal()
        logger.info("DecodeEngineCore: fully initialized and ready")

    # 启动握手第1轮（decode侧）：收prefill的权重IPC句柄 -> 导入并释放自己的权重 -> 等2秒 -> 发ACK
    def _post_model_load_hook(self):
        """Round 1 bootstrap: receive weight IPC handles from prefill, import them,
        free decode's own weight copy, then send ACK so prefill can proceed to
        allocate KV cache with the full GPU memory budget.
        """
        # 【译】启动握手第1轮：从prefill接收权重IPC句柄，导入它们，释放decode自己的那份权重拷贝，然后发送ACK，
        #   让prefill可以用完整的显存预算去分配KV cache。
        logger.info("DecodeEngineCore: waiting for weight IPC handles from prefill...")
        with self._disagg_ctx.socket(zmq.PULL) as w_sock:
            w_sock.connect(self._disagg_weight_ipc_addr)
            raw = w_sock.recv()
        weight_handles = pickle.loads(raw)
        logger.info(
            f"DecodeEngineCore: received weight handles ({len(weight_handles)} params), importing..."
        )
        self.runner_mgr.call_func(
            "import_model_weight_ipc_handles", weight_handles, wait_out=True
        )
        # Give the GPU allocator a moment to reclaim the freed weight memory
        # before prefill measures free VRAM for KV cache sizing.  Without this,
        # the freed pages may not yet be visible and prefill can OOM on alloc.
        # 【译】给GPU分配器一点时间回收被释放的权重显存，然后prefill再去测量空闲显存来确定KV cache大小。没有这一步，
        #   被释放的页可能还不可见，prefill分配时就可能OOM。

        time.sleep(2)
        logger.info(
            "DecodeEngineCore: weight import complete, sending ACK to prefill..."
        )
        with self._disagg_ctx.socket(zmq.PUSH) as ack_sock:
            ack_sock.bind(self._disagg_weight_ack_addr)
            ack_sock.send(b"ok")
        logger.info("DecodeEngineCore: weight ACK sent")

    # __init__期间（_ready_deferred为True）不发READY；之后委托给基类
    def _send_ready_signal(self):
        """No-op during super().__init__(): READY is deferred until after IPC import."""
        # 【译】在 super().__init__() 期间什么也不做：READY被推迟到IPC导入之后。
        if getattr(self, "_ready_deferred", False):
            return
        super()._send_ready_signal()

    # run_engine在构造完成后调用：建decode stream池、BlockAssignment发送通道、PrefillDone接收线程
    def _init_disagg(self):
        """Start runtime coordination sockets and threads.

        Called from run_engine() after __init__().  The kvcache+weight bootstrap
        was already handled in __init__(), so _init_disagg only sets up the
        per-request BlockAssignment/PrefillDone channel.
        """
        # 【译】启动运行期的协调socket和线程。
        #
        #   在 run_engine() 里、__init__() 之后调用。kvcache+权重的启动握手已经在 __init__() 里处理过了，
        #   所以 _init_disagg 只需要建立每个请求的BlockAssignment/PrefillDone通道。
        # Create pool of CU-masked CUDA streams for decode forward passes.
        # 【译】为decode的forward创建按CU屏蔽的CUDA stream池。
        # 解释：同prefill侧，默认池里只有一条整卡stream（见PrefillEngineCore._init_disagg）
        self.runner_mgr.call_func("create_decode_stream_pool", wait_out=True)

        # Decode PUSH connects to prefill's bound PULL (d2p channel).
        # 【译】decode的PUSH connect到prefill已bind的PULL（d2p通道）。
        self._d2p_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._d2p_sock.connect(self._disagg_d2p_addr)

        # Decode PULL binds so prefill's connecting PUSH finds a ready socket (p2d channel).
        # 【译】decode的PULL bind，这样prefill来connect的PUSH能找到一个已就绪的socket（p2d通道）。
        self._p2d_recv_sock = self._disagg_ctx.socket(zmq.PULL)
        self._p2d_recv_sock.bind(self._disagg_p2d_addr)

        # Start thread to receive PrefillDone from prefill.
        # 【译】启动线程，接收prefill发来的PrefillDone。
        self._prefill_done_thread = threading.Thread(
            target=self._recv_prefill_done,
            daemon=True,
            name="DecodeEngineCore-PrefillDoneRecv",
        )
        self._prefill_done_thread.start()

    # 后台线程：收PrefillDone，调用scheduler.on_prefill_done把序列移进running（带上prefill采样出的第一个token）
    def _recv_prefill_done(self):
        """Background thread: receives PrefillDone messages from prefill."""
        # 【译】后台线程：接收prefill发来的PrefillDone消息。
        from atom.model_engine.disagg_types import DisaggMsgType, PrefillDone

        sock = self._p2d_recv_sock
        while True:
            try:
                raw = sock.recv()
            except zmq.error.ContextTerminated:
                break
            msg_type, payload = pickle.loads(raw)
            if msg_type != DisaggMsgType.PREFILL_DONE:
                continue
            done: PrefillDone = payload
            self.scheduler.on_prefill_done(
                done.seq_id, done.num_tokens_computed, done.sampled_token_id
            )
            logger.info(
                f"DecodeEngineCore: seq {done.seq_id} prefill done "
                f"({done.num_tokens_computed} tokens cached), moved to running queue"
            )

    # 给prefill发一个BlockAssignment：seq的block_table、已缓存token数、上下文长度
    def _send_block_assignment(self, seq: Sequence):
        """Send BlockAssignment to prefill for a newly allocated sequence."""
        # 【译】为一个新分配了block的序列，给prefill发送BlockAssignment。
        from atom.model_engine.disagg_types import BlockAssignment, DisaggMsgType

        assignment = BlockAssignment(
            seq_id=seq.id,
            block_table=list(seq.block_table),
            num_cached_tokens=seq.num_cached_tokens,
            context_len=seq.num_tokens,
        )
        self._d2p_sock.send(pickle.dumps((DisaggMsgType.BLOCK_ASSIGNMENT, assignment)))
        logger.info(
            f"DecodeEngineCore: seq {seq.id} blocks assigned "
            f"({len(assignment.block_table)} blocks), waiting for prefill"
        )

    # 重写：新请求进waiting后，立刻尝试分配block（allocate_waiting），给分到block的seq发BlockAssignment
    def pull_and_process_input_queue(self):
        """Override: add new sequences to waiting queue, then allocate blocks
        and send BlockAssignment to prefill for any that fit."""
        # 【译】重写：把新序列加入等待队列，然后为放得下的那些分配block，并给prefill发送BlockAssignment。
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    return True
                recv_reqs.append(seq)
        if recv_reqs:
            self.scheduler.extend(recv_reqs)
        for seq in self.scheduler.allocate_waiting():
            self._send_block_assignment(seq)
        return False

    # decode版的一步：schedule -> forward -> postprocess -> 流式/结束输出（没有KV传输、多模态、拒绝请求等旁路）
    def _process_engine_step(self):
        """Override: handle None from DecodeScheduler when prefill_waiting is
        non-empty but running is empty (sequences are still being prefilled)."""
        # 【译】重写：处理 DecodeScheduler 在 prefill_waiting 非空但running为空时返回的None（序列还在做prefill）。
        if not self.scheduler.has_requests():
            return False
        result = self.scheduler.schedule()
        if result is None:
            # Sequences exist but are still waiting for PrefillDone — spin.
            # 【译】序列存在，但还在等PrefillDone——空转。
            return False
        scheduled_batch, seqs = result
        if scheduled_batch is None:
            return False
        self.scheduler.metrics.record_forward(scheduled_batch, seqs)
        t0 = time.perf_counter()
        fwd_out = self.runner_mgr.call_func("forward", scheduled_batch, wait_out=True)
        iter_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"iter {iter_ms:.2f}ms | "
            f"reqs={scheduled_batch.total_seqs_num} "
            f"(prefill={scheduled_batch.total_seqs_num_prefill} "
            f"decode={scheduled_batch.total_seqs_num_decode}) | "
            f"tokens={scheduled_batch.total_tokens_num} "
            f"(prefill={scheduled_batch.total_tokens_num_prefill} "
            f"decode={scheduled_batch.total_tokens_num_decode})"
        )
        finished_seqs = self.scheduler.postprocess(
            seqs.values(), fwd_out, stream_output_queue=self.stream_output_queue
        )
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass
        if finished_seqs:
            self.output_queue.put_nowait(finished_seqs)
        return True

    # DecodeEngineCore进程的入口：构造（含握手）-> _init_disagg -> busy_loop -> exit
    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        EngineCore._setup_engine_process("DecodeEngineCore")
        engine = None
        try:
            engine = DecodeEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception:
            logger.exception("DecodeEngineCore.run_engine failed")
            raise
        finally:
            if engine is not None:
                engine.exit()

    # 基类退出之后销毁PD分离用的ZMQ上下文
    def exit(self):
        super().exit()
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            logger.debug("disagg context destroy failed", exc_info=True)
