# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import hashlib
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import pickle
import queue
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock, Thread, local

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_CMD,
    DISCARD_WEIGHT_SYNC_CMD,
    FINISH_WEIGHT_SYNC_CMD,
    RpcResponseRouter,
    RpcResult,
    checked_timeout,
    engine_budget,
)
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.engine_utility import (
    FIRE_AND_FORGET_UTILITY_CMDS,
    WEIGHT_UPDATE_UTILITY_CMDS,
    completes_weight_sync,
)
from atom.model_engine.prefix_hash import take_shipped_chains
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.utils import (
    envs,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
)

# 本文件：CoreManager——主进程里"管理EngineCore进程"的一层
#   ① 启动EngineCore子进程（每个DP rank一个；PP时每个stage一个），建立与它们之间的ZMQ socket
#   ② add_request：把LLMEngine / API server交来的Sequence用pickle打包，发给某个DP rank（DP>1时做负载均衡）
#   ③ 每个DP rank一个后台输出线程：收EngineCore推回的结果，调用流式回调，或把已结束的序列放进outputs_queue
#   ④ 控制通道：工具RPC（utility command）、集合RPC（collective RPC）、RL权重同步、abort、关闭
#
# 在整体架构中的位置（第2批的主线）：
#   openai_server.py（在线）/ simple_inference.py（离线）
#        │
#   LLMEngine（llm_engine.py：tokenizer，文本 -> Sequence）
#        │  core_mgr.add_request(seqs) / core_mgr.get_output()
#   CoreManager（本文件，仍在主进程里）
#        ║  ZMQ + pickle
#   EngineCore进程（engine_core.py：Scheduler + busy_loop）
#        ║  共享内存广播队列 + ZMQ
#   ModelRunner进程 × TP（每张GPU一个，由AsyncIOProc承载，async_proc.py；执行forward见第5批）
#
# 进程与ZMQ socket拓扑（单机 -dp 2 -tp 2 为例，共4张GPU；下面画的是DP rank r的那一组）：
#
#   主进程：CoreManager（三个socket都由这一端bind）       子进程：EngineCore DP r（spawn出来，进程名EngineCore_DP{r}）
#
#   input_sockets[r]    ROUTER（bind） ══ ADD ══════════════►  DEALER（connect）─┐
#   control_sockets[r]  ROUTER（bind） ══ UTILITY/SHUTDOWN ═►  DEALER（connect）─┴─ 输入线程（一个线程poll两个socket）
#   output_sockets[r]   PULL（bind）   ◄═ ADD/STREAM/READY ═══  PUSH（connect）──── 输出线程
#                                          METRICS/UTILITY_RESPONSE（类型码见engine_core_protocol.py）
#   每个rank一个输出线程在PULL上收                              busy_loop主线程：Scheduler + AsyncIOProcManager
#                                                                 │ 下发：共享内存广播队列MessageQueue（所有TP worker读同一份）
#                                                                 │ 回复：每个worker一条ZMQ PUSH(connect) ─► PULL(bind)
#                                                                 ├─ ModelRunner进程 TP0（GPU 2r）
#                                                                 └─ ModelRunner进程 TP1（GPU 2r+1）
#
#   为什么请求和控制分两个socket：input_sockets只有add_request一个线程写，可以不加锁；控制消息会从多个线程发出，
#     放在另一个socket上加锁串行化（详见_send_request的docstring）
#   多机DP时ipc://换成tcp://（端口见InternodeDPSocketPlan），bind/connect方向不变
#
# 对比nano-vllm：nano-vllm没有独立的引擎进程，也就没有这一层——LLMEngine.step()在主进程里直接调用
#   scheduler.schedule()、model_runner.call("run")、scheduler.postprocess()，进程间只靠共享内存+Event给其他TP rank发指令。
#   ATOM把调度和执行搬进了EngineCore子进程，主进程只负责HTTP、tokenize和收发消息：
#   好处是主进程里的Python开销不会和调度循环抢同一个解释器（GIL），还能起多个EngineCore做DP；
#   代价是每个请求、每个结果都要pickle后经ZMQ走一趟
logger = logging.getLogger("atom")
# 权重同步最后那条FINISH命令的等待上限（秒），见_broadcast_and_wait末尾；【分支】RL权重热更新，见第10批
_WEIGHT_SYNC_FINISH_TIMEOUT_S = 30.0
# Only the sync that owns them sends these. A FINISH unfences every engine,
# and a DISCARD aborts whichever sync is between its buckets.
# 【译】只有拥有这些命令的那次同步才会发送它们。一个FINISH会解除所有engine的隔离（fence），
#   一个DISCARD会中止正处在两个分桶之间的那次同步。
# 解释：_refuse_unowned_weight_sync_cmd据此拒绝从外部直接发送这些命令
_WEIGHT_SYNC_OWNED_CMDS = WEIGHT_UPDATE_UTILITY_CMDS | {
    FINISH_WEIGHT_SYNC_CMD,
    DISCARD_WEIGHT_SYNC_CMD,
}

# Valid values for Config.dp_load_balance / --dp-load-balance, and the default.
# Single source of truth for argparse (choices + default) so the CLI flag and
# the Config field can never diverge.
# 【译】Config.dp_load_balance / --dp-load-balance 的合法取值及默认值。argparse（choices + default）
#   只从这里取，保证命令行参数和Config字段永远不会不一致。
# 解释：arg_utils.py从本模块导入这两个常量；三种策略的含义见_select_dp_rank_locked
DP_LB_STRATEGIES = ("round_robin", "least_requests", "least_tokens")
DP_LB_DEFAULT = "least_requests"

# Engine sockets for a multi-node run cannot use IPC paths, and the two ends
# must derive identical TCP ports without negotiating. Both compute this plan
# from the shared master port.
# 【译】多节点运行时engine的socket不能用IPC路径，而且两端必须在不协商的情况下算出相同的TCP端口。
#   两端都从共享的master端口计算这个方案。
# 解释：例如data_parallel_master_port=29500 -> 起点29600；每个engine占3个端口：
#   engine 0：input 29600、output 29601、control 29602；engine 1：29603、29604、29605……
INTERNODE_DP_SOCKET_PORT_OFFSET = 100
INTERNODE_DP_PORTS_PER_ENGINE = 3


# 【分支】多机DP：一个engine的三个TCP端口（单机用ipc://地址，不构造它）
# @dataclass(frozen=True)：自动生成__init__等方法，且实例不可修改（给字段赋值会抛FrozenInstanceError）
@dataclass(frozen=True)
class InternodeDPSocketPlan:
    """The three TCP ports one engine's sockets use, in a multi-node run.

    Single-node runs never build one of these: their engines are local, so
    CoreManager hands them IPC paths from ``get_open_zmq_ipc_path()`` instead.
    """

    # 【译】多节点运行时，一个engine的socket使用的三个TCP端口。
    #
    #   单节点运行从不构造它：那时engine都在本地，CoreManager改为用 get_open_zmq_ipc_path() 给它们分配IPC路径。
    # 解释：rank是全局engine编号（即全局DP rank），三个端口对应input / output / control三个socket
    rank: int
    input_port: int
    output_port: int
    control_port: int


# 由DP master端口确定性地推出每个engine的TCP端口（B级：多机DP）
#   输入：engine_count（全局engine数）、master_port；输出：按rank排列的InternodeDPSocketPlan列表
#   调用方：CoreManager.__init__（仅is_multinode_dp时）；协调节点和工作节点各算一遍，结果一致
def build_internode_dp_socket_plan(
    *, engine_count: int, master_port: int
) -> list[InternodeDPSocketPlan]:
    """Derive deterministic per-engine TCP ports from the DP master port.

    Multi-node only -- ``CoreManager.__init__`` calls this when
    ``is_multinode_dp`` and passes ``None`` otherwise, and that ``None`` is what
    selects the IPC path (and, with it, the connect/bind polarity) further down.

    Three ports per engine: requests, outputs, and control. Control is separate
    so the request socket keeps a single writer thread (see _send_request).
    """
    # 【译】由DP master端口推导出确定的、每个engine一组的TCP端口。
    #
    #   仅用于多节点——CoreManager.__init__ 在 is_multinode_dp 时调用它，否则传 None；正是这个 None
    #   在后面选择了IPC路径（以及随之而来的connect/bind方向）。
    #
    #   每个engine三个端口：请求、输出、控制。控制单独一个端口，这样请求socket只有一个写线程（见 _send_request）。
    # 解释：v0.1.8里ipc和tcp的bind/connect方向是一样的：都是CoreManager bind、EngineCore connect（见__init__里
    #   "PULL always binds"那段注释）；多机时的区别只在地址：协调节点bind tcp://0.0.0.0:port，engine连tcp://<master_ip>:port
    base = master_port + INTERNODE_DP_SOCKET_PORT_OFFSET
    return [
        InternodeDPSocketPlan(
            rank=rank,
            input_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE,
            output_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE + 1,
            control_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE + 2,
        )
        for rank in range(engine_count)
    ]


# 列出本节点要启动的每个engine的 (全局DP rank, 本地DP rank)
#   调用方：CoreManager.__init__——在dp-attention改写配置之前调用，所以这里读到的tp_size还是命令行的-tp
# 例1（单机 -dp 2 -tp 4，不开dp-attention）：[(0,0), (1,1)]，即2个EngineCore，每个管4个ModelRunner
# 例2（单机 -dp 2 -tp 4 --enable-dp-attention）：每个TP rank自成一个engine，
#   [(0,0), (1,1), ..., (7,7)] 共8个EngineCore，每个只管1个ModelRunner
# 例3（两台机器，每台 -tp 8 -dp 2 --data-parallel-size-local 1 --enable-dp-attention，第二台 data_parallel_rank=1）：
#   第二台 rank_offset=1、local_dp_size=1 -> [(8,0), (9,1), ..., (15,7)]：全局rank 8~15，用本机GPU 0~7
def iter_dp_rank_assignments(config) -> list[tuple[int, int]]:
    """The (global_dp_rank, local_dp_rank) pairs this node owns.

    The global rank identifies the rank within the whole DP group; the local
    rank indexes this node's GPUs. They coincide on a single node and diverge
    on every node after the first.

    Under DP-attention each TP rank becomes its own engine, so both the count
    and the global offset scale by tp_size.
    """
    # 【译】本节点拥有的 (global_dp_rank, local_dp_rank) 对。
    #
    #   全局rank标识它在整个DP组中的位置；本地rank用来索引本节点的GPU。单节点上两者相同，从第二个节点开始就不同了。
    #
    #   在DP-attention下，每个TP rank都成为一个独立的engine，所以数量和全局偏移都要乘以tp_size。
    pc = config.parallel_config
    tp_size = config.tensor_parallel_size
    local_dp_size = pc.data_parallel_size_local
    # 多机时：本节点第一个全局DP rank（命令行传入的data_parallel_rank）；单机为0
    rank_offset = pc.data_parallel_rank if pc.is_multinode_dp else 0

    # dp-attention：每个 (本机DP组, TP rank) 组合都是一个engine，编号 = DP组号 × tp_size + tp_rank
    if config.enable_dp_attention:
        return [
            (
                (rank_offset + local_base) * tp_size + tp_rank,
                local_base * tp_size + tp_rank,
            )
            for local_base in range(local_dp_size)
            for tp_rank in range(tp_size)
        ]
    # 普通情况：每个DP rank一个engine
    return [
        (rank_offset + local_rank, local_rank) for local_rank in range(local_dp_size)
    ]


# 【分支】--fake-eplb（在少量GPU上模拟大规模部署，只用于测性能）：dp-attention下决定实际启动几个EngineCore；见第7批EPLB
#   正常情况直接返回logical（每张卡一个）；调用方：CoreManager.__init__
def _resolve_dp_engine_count(config: Config, logical: int) -> int:
    """How many DP-attention EngineCores to launch for a `logical`-wide run.

    Normally `logical`, one per device. Under `--fake-eplb` on a box with fewer
    visible devices, launch only that many and record the width to keep sharding
    experts for, so each device owns the slice it would own in the full
    deployment; `FusedMoE` repeats the gathered tokens by the same ratio to
    match its token volume too.

    Gated on fake_eplb like `Config.tp_world_size`, and for the same reason.
    """
    # 【译】对于一个 logical 宽的运行，要启动多少个DP-attention EngineCore。
    #
    #   通常就是 logical，每个设备一个。在可见设备更少的机器上用 --fake-eplb 时，只启动可见数量那么多个，
    #   并记录专家仍按哪个宽度切分，这样每个设备拥有它在完整部署中会拥有的那一份；FusedMoE 也按同样的比例
    #   重复聚合后的token，使token量也匹配。
    #
    #   和 Config.tp_world_size 一样以 fake_eplb 为条件，原因也相同。
    import torch

    if not config.fake_eplb:
        return logical
    visible = torch.cuda.device_count()
    # 可见设备数为0、或者不少于logical：不需要模拟
    if not 0 < visible < logical:
        return logical
    if logical % visible:
        raise ValueError(
            f"Simulated DP-attention: {logical} ranks do not divide evenly over "
            f"{visible} visible device(s). Make the visible device count a "
            f"divisor of tp x dp, e.g. via HIP_VISIBLE_DEVICES."
        )
    # 例：logical=8、visible=4 -> 只启动4个engine，专家仍按8份切，每个rank把聚合到的token重复2遍
    config.dp_logical_size = logical
    logger.warning(
        "Simulated DP-attention: running %d of a %d-rank deployment. Experts "
        "shard %d ways and each rank repeats the gathered tokens %dx, so "
        "per-device MoE shapes match -- but collectives only cover the ranks "
        "that exist, so THE MODEL OUTPUT IS MEANINGLESS. Benchmarking only.",
        visible,
        logical,
        logical,
        logical // visible,
    )
    return visible


# 把一批Sequence打包成一个ADD请求帧（已经pickle好的bytes），由_send_request发出
#   帧格式：(EngineCoreRequestType.ADD, seqs) 或 (ADD, seqs, chains)
#   chains：LLMEngine在前端预先算好的prompt前缀hash链（ATOM_FRONTEND_PREFIX_HASH，默认开）。Sequence被pickle时
#     不带它，所以单独作为第3个元素随帧发送；EngineCore输入线程再用attach_shipped_chains装回去（前缀缓存匹配，第4批）
#   调用方：add_request、_dispatch_to_dp_ranks、DisaggCoreManager.add_request
# pickle：Python标准的对象序列化，dumps把对象变成bytes，loads反过来；ATOM在进程间传的所有消息都是pickle后的元组
def _add_frame(seqs: list[Sequence]) -> bytes:
    """An ADD request for an engine that admits `seqs` with a BlockManager:
    their shipped prompt chains, if any, ride third (`take_shipped_chains`)."""
    # 【译】发给一个"用BlockManager接纳seqs"的engine的ADD请求：如果有随行的prompt链，就作为第三个元素
    #   （take_shipped_chains）。
    chains = take_shipped_chains(seqs)
    # 例：没有链时 frame=(ADD, [seq1, seq2])；有链时 frame=(ADD, [seq1, seq2], [chain1, chain2])
    frame = (EngineCoreRequestType.ADD, seqs) + ((chains,) if chains else ())
    return pickle.dumps(frame)


# CoreManager：主进程里的"EngineCore进程管理器 + 请求路由器"
# 谁创建：LLMEngine.__init__（config.enable_rapidserve时改用下面的子类DisaggCoreManager）
# 谁调用：
#   离线：LLMEngine.add_request -> add_request；LLMEngine.step -> get_output（阻塞等一批已结束的序列）
#   在线：api_server直接调用core_mgr.add_request，结果全部经流式回调（seq.stream_callback）送回；客户端断开时abort_request
#   RL rollout：collective_rpc / weight_sync_session / broadcast_utility_command_sync（第10批）
#
# 启动时序（__init__）：
#   CoreManager（主进程）                                EngineCore进程（每个DP rank一个）
#   launch_engine_core × N（只创建Process对象）
#   process.start() × N ──────────── spawn ─────────►   run_engine -> EngineCore.__init__
#                                                       ├─ 启动输出线程（PUSH connect）
#                                                       ├─ 启动输入线程（两个DEALER connect，各发一个空帧）
#   逐个rank：bind ROUTER(input)，recv空帧 ◄─────────────┤   ROUTER靠这个空帧记下对端的identity
#            bind ROUTER(control)，recv空帧 ◄───────────┤
#            bind PULL(output)                          ├─ DP>1：建DP rank之间的gloo进程组
#                                                       ├─ AsyncIOProcManager：spawn ModelRunner × TP，加载模型
#                                                       ├─ get_num_blocks -> allocate_kv_cache -> capture_cudagraph
#                                                       ├─ 构造Scheduler（用回填的num_kvcache_blocks）
#   _wait_for_all_ready_signals ◄──── READY ────────────┴─ _send_ready_signal（携带max_pool_tokens）
#   为每个rank启动输出线程                                    进入busy_loop，开始处理请求
class CoreManager:
    # 初始化"所有继承方法都会用到"的成员变量，在spawn任何engine之前调用
    #   调用方：CoreManager.__init__、DisaggCoreManager.__init__（子类不走父类__init__，但共用这些字段）
    def _init_shared_state(
        self,
        config: Config,
        *,
        label: str,
        local_engine_count: int,
        global_engine_count: int | None = None,
    ) -> None:
        """Every field the inherited methods touch, before any engine is spawned.

        Subclasses spawn their engines differently and so cannot run this
        class's ``__init__`` -- but they inherit its output threads, ``close()``
        and DP-load bookkeeping, all of which read the fields set here. This is
        the one place to add another such field.

        It exists because the alternative was tried: ``DisaggCoreManager`` used
        to hand-copy this block, and the copy drifted. ``_flush_stream_batch_fn``
        was added to the copy and to the API server that assigns it, but not to
        this class -- so the offline entrypoint, which is the one path that
        neither initialises nor assigns it, had its output thread die on the
        first streamed token and hung until CI timed out an hour later.

        ``local_engine_count`` is the engines spawned on THIS node;
        ``global_engine_count`` is the engines this manager routes to, which on
        a coordinator spans other nodes too. They are equal on a single node.
        Routing state sizes to the global count -- a short array would
        IndexError the moment the balancer picked a remote rank.
        """
        # 【译】继承下来的方法会访问的每一个字段，在spawn任何engine之前设置好。
        #
        #   子类spawn engine的方式不同，因此不能运行本类的 __init__——但它们继承了本类的输出线程、close()和DP负载记账，
        #   这些都会读这里设置的字段。要再加一个这样的字段，就加在这里。
        #
        #   之所以有这个方法，是因为另一种做法已经试过了：DisaggCoreManager 以前手抄了这一段，结果抄本跑偏了。
        #   _flush_stream_batch_fn 被加进了抄本和给它赋值的API server，却没有加进本类——于是离线入口（唯一一条既不初始化
        #   也不赋值它的路径）的输出线程在第一个流式token时就挂掉，一直卡到一小时后CI超时。
        #
        #   local_engine_count 是在本节点上spawn的engine数；global_engine_count 是本manager路由到的engine数，在协调节点上
        #   它还包括其他节点的。单节点上两者相等。路由状态按全局数量确定大小——数组短了的话，负载均衡器一选到远端rank
        #   就会IndexError。
        # 解释：label是日志前缀（"Engine Core Mgr"或"DisaggCoreManager"）
        self.label = label
        # 【译】（行尾注释）记录是否已经关闭过
        # 解释：close()可能被调用多次（显式调用、weakref.finalize、启动失败时的清理），靠它保证只执行一次
        self._closed = False  # Track whether already closed
        self.local_engine_count = local_engine_count
        self.global_engine_count = (
            local_engine_count if global_engine_count is None else global_engine_count
        )
        # ZMQ上下文：一个进程里所有socket的容器；io_threads=2表示ZMQ在后台用2个I/O线程收发数据（与Python线程无关）
        self.ctx = zmq.Context(io_threads=2)
        # 已结束序列的队列：输出线程收到ADD（一批已结束的序列）后put，get_output()从这里取
        # queue.Queue：线程安全的FIFO队列，put/get内部有锁，用来在线程之间传数据；get()在队列空时阻塞
        #   queue.Queue[list[Sequence]]() 只是带类型参数的构造，运行时与 queue.Queue() 相同
        # 读这个队列的只有get_output（离线generate、rollout）和asyncio模式的_async_output_handler；在线服务不调用它，结果走流式回调
        self.outputs_queue = queue.Queue[list[Sequence]]()
        # Utility replies are matched to their caller by request id, not by
        # position on a shared queue, where a late or unasked-for reply became
        # the next caller's answer.
        # 【译】工具命令的回复按请求id匹配给调用方，而不是按在共享队列里的位置——按位置时，一个迟到的或没人要的回复
        #   会变成下一个调用方的答案。
        # 解释：RpcResponseRouter（collective_rpc.py）给每个request_id一个独立的队列，输出线程收到UTILITY_RESPONSE后按id投递
        self._rpc_router = RpcResponseRouter()
        # Held across every bucket plus FINISH/abort. RLock lets the sender's
        # outer session call the same synchronous command method per bucket.
        # 【译】在所有分桶以及FINISH/abort期间一直持有。用RLock，让发送方外层的会话能对每个分桶调用同一个同步命令方法。
        # 解释：RLock（可重入锁）允许同一个线程重复获取而不死锁；普通Lock在同一线程里第二次acquire会卡死
        self._weight_sync_lock = RLock()
        # threading.local()：每个线程看到自己独立的一套属性；这里记录"当前线程所在的权重同步会话"（sync_id、failure、active）
        self._weight_sync_context = local()
        # Identity-only marker: unlike request_id, a caller cannot forge it by
        # choosing a string. Never put it on the wire.
        # 【译】只看身份的标记：不同于request_id，调用方无法通过挑一个字符串来伪造它。永远不要把它放到网络上传输。
        # 解释：object()每次都创建一个独一无二的对象，用`is`比较；只有内部的_broadcast_and_wait会传它，见_refuse_unowned_weight_sync_cmd
        self._weight_sync_dispatch_token = object()
        # Runner processes behind each engine -- EngineCore's own sizing -- so
        # an engine that never answers still fails one result per rank. Read
        # with defaults: managers are also built from bare stand-in configs.
        # 【译】每个engine背后的runner进程数——即EngineCore自己的规模计算——这样一个从不回复的engine仍会为每个rank
        #   各产生一个失败结果。用带默认值的方式读取：manager也会用简陋的替身config来构造。
        # 解释：= tp_world_size × pcp，与EngineCore.__init__里传给AsyncIOProcManager的进程数相同；_flatten_dp_replies用它补齐缺失的回复
        self._rpc_ranks_per_engine = getattr(config, "tp_world_size", 1) * getattr(
            config, "prefill_context_parallel_size", 1
        )
        # seq_id -> 流式回调。add_request时从Sequence上摘下来存在这里（回调是闭包，不能pickle到别的进程），
        #   输出线程收到STREAM时按seq_id调用
        self._seq_id_to_callback = {}
        # Batched stream-flush hook, resolved lazily by the API server (avoids
        # an api_server <-> engine_core_mgr import cycle). Stays None on every
        # path that never streams, which the output thread checks for.
        # 【译】按批刷新流式输出的钩子，由API server延迟设置（避免 api_server <-> engine_core_mgr 循环导入）。
        #   在所有不做流式输出的路径上它都保持None，输出线程会检查这一点。
        self._flush_stream_batch_fn = None
        # Longest prompt the KV pool can hold, reported by each rank's READY.
        # None until the first one arrives, and thereafter the smallest across
        # ranks: a DP rank sizes its pool from its own free memory, so a prompt
        # is only safe to admit if it fits wherever the router sends it.
        # 【译】KV池能容纳的最长prompt，由每个rank的READY报告。第一个READY到来之前为None，之后取所有rank中的最小值：
        #   每个DP rank按自己的空闲显存确定池的大小，所以只有当一个prompt无论被路由到哪里都放得下时，接纳它才是安全的。
        # 解释：api_server的_validate_pool_capacity用它提前拒绝过长的prompt
        self.max_pool_tokens: int | None = None
        # 下面几个列表按engine下标对齐（下标即DP rank；PP时是engine序号）：
        #   engine_core_processes：本节点spawn的EngineCore进程对象
        #   _distributed_stores：给ModelRunner初始化torch.distributed用的TCPStore（见_start_distributed_store）
        #   input_sockets / engine_core_identities：请求用的ROUTER socket，以及对端DEALER的identity（发消息时要带上它）
        #   output_sockets：接收结果的PULL socket
        #   shutdown_paths：每个输出线程的inproc地址，close()往这里发一个空消息让线程退出
        #   output_threads：每个rank一个输出线程
        self.engine_core_processes = []
        self._distributed_stores = []
        self.input_sockets = []
        self.output_sockets = []
        self.engine_core_identities = []
        self.shutdown_paths = []
        self.output_threads = []
        # Fair-rotation cursor, advanced once per selection. round_robin picks the
        # rank directly (cursor % n); the load-aware strategies use it only to seed
        # the argmin start offset so fully-tied ranks rotate instead of always
        # resolving to rank 0.
        # 【译】公平轮转的游标，每次选择后前进一格。round_robin直接用它选rank（cursor % n）；负载感知的策略只用它作为
        #   argmin扫描的起点偏移，这样完全打平的几个rank会轮流被选中，而不是总落到rank 0。
        self._rank_rotation_cursor = 0

        # --- DP request load balancing (see _select_dp_rank_locked) ---
        # A subclass may fan out through its own add_request() and never charge
        # load at all, but the inherited output thread still calls
        # _release_seq_load() on every finished sequence, so these MUST exist.
        # Strategy: "round_robin" | "least_requests" | "least_tokens" (validated
        # at the CLI by argparse choices=DP_LB_STRATEGIES).
        # 【译】--- DP请求负载均衡（见 _select_dp_rank_locked）---
        #   子类可能通过自己的 add_request() 扇出、根本不记负载，但继承来的输出线程仍会对每个结束的序列调用
        #   _release_seq_load()，所以这些字段必须存在。
        #   策略："round_robin" | "least_requests" | "least_tokens"（由argparse的 choices=DP_LB_STRATEGIES 在命令行校验）。
        self._dp_lb_strategy = config.dp_load_balance
        # Token-equivalent weight of one in-flight request for "least_tokens".
        # Read once here: this is a construction-time config value (CoreManager
        # is built after env/args are finalized), not a runtime-tunable knob.
        # 【译】"least_tokens"策略中一个在途请求折合多少个token。在这里只读一次：这是构造时的配置值
        #   （CoreManager在环境变量和参数都确定之后才构造），不是运行时可调的旋钮。
        # 解释：ATOM_DP_LB_REQ_EQUIV默认512
        self._dp_lb_req_equiv = envs.ATOM_DP_LB_REQ_EQUIV
        # ATOM_DP_SESSION_AFFINITY（默认关）：同一个会话的请求固定发往同一个DP rank，复用那里的前缀缓存
        self._dp_session_affinity_enabled = envs.ATOM_DP_SESSION_AFFINITY
        # Session id -> rank whose local prefix cache owns the session. Owners
        # are immutable for the lifetime of the process: moving one turn to a
        # light rank discards the dominant optimization in agentic workloads,
        # namely reuse of the accumulated conversation prefix.
        # 【译】会话id -> 其本地前缀缓存拥有该会话的rank。拥有者在进程生命周期内不可改变：把某一轮挪到一个轻载的rank，
        #   会丢掉agent类负载里最主要的优化，即复用不断累积的对话前缀。
        self._dp_session_owners: dict[str, int] = {}
        # Last prompt length observed for each sticky session.  A later turn on
        # the same owner normally reuses the old prompt, so only its positive
        # growth is new prefill debt.  Keeping this small scalar per session
        # avoids charging the complete cached conversation on every turn.
        # 【译】每个粘性会话上次观察到的prompt长度。同一拥有者上的后续一轮通常复用旧prompt，所以只有它正向增长的部分
        #   才是新的prefill负担。每个会话只保存这一个小标量，避免每一轮都把完整的已缓存对话算进负载。
        self._dp_session_prompt_tokens: dict[str, int] = {}
        # 各类路由决策的计数器（由get_dp_router_statistics导出，用于观测）
        self._dp_route_counters = {
            "affinity_new_total": 0,
            "affinity_owner_hit_total": 0,
            # Kept as an explicit invariant/metric: strict affinity must leave
            # this at zero. It makes an accidental reintroduction of spill
            # visible in benchmark artifacts.
            # 【译】作为显式的不变量/指标保留：严格亲和必须让它保持为0。这样如果意外重新引入了溢出（spill），
            #   在benchmark产物里就能看出来。
            "affinity_spill_total": 0,
            "affinity_parent_ignored_total": 0,
            "explicit_total": 0,
            "load_balanced_total": 0,
        }
        # 每个rank累计被路由到的请求数（只增不减，统计用）
        self._rank_routed_total = [0] * self.global_engine_count
        # Authoritative local load per rank. Request count is in-flight until
        # finish/abort; token count is queued/in-flight PREFILL work and is
        # released as soon as the first model output proves prefill completed.
        # Guarded by _lb_lock because dispatch runs on the request thread while
        # completion/release runs on the per-rank output threads.
        # 【译】每个rank权威的本地负载。请求数在完成/abort之前一直算在途；token数是排队中/进行中的PREFILL工作量，
        #   一旦第一个模型输出证明prefill已完成就释放。由 _lb_lock 保护，因为分发在请求线程上运行，
        #   而完成/释放在每个rank的输出线程上运行。
        # 解释：例如rank0收到两个prompt长度为1000和200的请求 -> _rank_reqs[0]=2、_rank_tokens[0]=1200；
        #   第一个请求吐出第一个token后 -> _rank_tokens[0]=200（请求数仍为2，直到请求结束）
        self._rank_reqs = [0] * self.global_engine_count
        self._rank_tokens = [0] * self.global_engine_count
        # seq_id -> (dp_rank, req_cost, tok_cost) so release subtracts exactly
        # what dispatch added, and only for ranks that were actually charged.
        # 【译】seq_id -> (dp_rank, req_cost, tok_cost)，这样释放时减去的正好是分发时加上的，而且只针对真正被记过账的rank。
        self._seq_load = {}
        # threading.Lock：互斥锁，with self._lb_lock: 块内同一时刻只有一个线程在执行
        self._lb_lock = Lock()
        # Control traffic (utility commands, abort, shutdown) travels on its own
        # sockets so that input_sockets keeps a single writer -- see
        # _send_request. These have several writer threads and so do need
        # serializing, but none of them runs per request.
        # 【译】控制流量（工具命令、abort、关闭）走自己的socket，让 input_sockets 保持单写者——见 _send_request。
        #   这些控制socket有多个写线程，所以确实需要串行化，但它们都不是每个请求都会跑的。
        self.control_sockets = []
        self.control_identities = []
        self._control_send_lock = Lock()
        # dp_rank -> newest metrics snapshot, refreshed by the output threads
        # from EngineCore's own periodic push. Read directly by the exporter, so
        # scraping costs no round trip and cannot time out.
        # 【译】dp_rank -> 最新的指标快照，由输出线程根据EngineCore自己定期推送的数据刷新。exporter直接读它，
        #   所以抓取指标不需要一次往返，也不会超时。
        self.latest_metrics: dict[int, dict] = {}

    # 在CoreManager进程里起一个TCPStore（torch.distributed的键值集合点服务），供ModelRunner初始化分布式时连接
    #   端口为0时由OS分配，再把实际端口写回pc.data_parallel_base_port——这一步在为各rank deepcopy配置之前，
    #   所以每个EngineCore（及其ModelRunner）拿到的都是真实端口
    #   多机：只有全局DP rank 0所在节点起store，其他节点只设_managed_distributed_store=True，作为客户端连过去
    #   调用方：CoreManager.__init__、DisaggCoreManager.__init__
    def _start_distributed_store(
        self, config: Config, *, multinode: bool = False
    ) -> None:
        """Bind the actual rendezvous server before publishing its port to workers."""
        # 【译】先绑定真正的集合点服务器，再把它的端口公布给worker。
        from torch.distributed import TCPStore

        pc = config.parallel_config
        if multinode and pc.data_parallel_base_port == 0:
            raise ValueError(
                "Multi-node DP requires the same nonzero --data-parallel-base-port "
                "(or ATOM_DP_BASE_PORT) on every node."
            )
        if not multinode or pc.data_parallel_rank == 0:
            # is_master=True：本进程就是服务端；wait_for_workers=False：不等客户端连上就返回
            store = TCPStore(
                pc.data_parallel_master_ip,
                pc.data_parallel_base_port,
                is_master=True,
                wait_for_workers=False,
            )
            # 保存引用：TCPStore对象被回收时服务就停了，close()的最后才清空这个列表
            self._distributed_stores.append(store)
            pc.data_parallel_base_port = store.port
            logger.info(
                "%s: model-runner TCPStore listening on %s:%d",
                self.label,
                pc.data_parallel_master_ip,
                store.port,
            )
        # 告诉ModelRunner：store由CoreManager持有，自己作为客户端连接（ModelRunner._setup_device_and_distributed，第5批）
        pc._managed_distributed_store = True

    # 构造CoreManager：改写DP配置 -> 规划socket地址 -> 起TCPStore -> spawn所有EngineCore -> bind socket并握手
    #   -> 等所有rank的READY -> 启动输出线程。启动时序图见类上方
    #   config：LLMEngine构造的Config；dp-attention的改写直接写进这个对象（主进程里LLMEngine持有的也是它）
    def __init__(self, config: Config):
        # ---------- 1. 并行拓扑检查：多机DP + PP暂不支持 ----------
        pp_size = config.pipeline_parallel_size
        self.pp_size = pp_size
        pc = config.parallel_config
        multinode = pc.is_multinode_dp
        if multinode and pp_size > 1:
            raise ValueError(
                "Multi-node data parallelism combined with pipeline "
                "parallelism (pipeline_parallel_size > 1) is not supported: "
                "the engine index space folds PP stages into DP ranks, and the "
                "two mappings have not been reconciled across nodes."
            )

        # ---------- 2. 确定本节点要起哪些engine（必须在下面的dp-attention改写之前算，用的是原始tp） ----------
        rank_assignments = iter_dp_rank_assignments(config)
        # --enable-dp-attention：注意力按DP切（每张卡各算各的请求，注意力部分不做TP），MoE层仍按 dp×tp 张卡切分（第7批）
        #   所以每张卡都是一个独立的EngineCore（各自有调度器），配置随之改写：
        #   例：单机 -tp 4 -dp 2 -> 8个engine；data_parallel_size 2 -> 8，data_parallel_size_local -> 8，tensor_parallel_size 4 -> 1
        if config.enable_dp_attention:
            assert pp_size == 1, "Pipeline parallel + DP-attention is not supported yet"
            # One engine per assignment this node owns; on a single node that is
            # the whole tp x dp grid. --fake-eplb may shrink it to the visible
            # device count, taking the first N assignments.
            # 【译】本节点拥有的每个分配一个engine；单节点上就是整个 tp x dp 网格。--fake-eplb 可能把它缩小到可见设备数，
            #   取前N个分配。
            local_engine_count = _resolve_dp_engine_count(config, len(rank_assignments))
            logger.info(
                f"Enable dp attention, using {local_engine_count} data parallel ranks"
            )
            # Under DP-attention every TP rank becomes its own DP rank.
            # 【译】在DP-attention下，每个TP rank都成为自己的DP rank。
            config.parallel_config.data_parallel_size *= config.tensor_parallel_size
            config.parallel_config.data_parallel_size_local = local_engine_count
            # 多机：本节点的起始全局rank也换算成新编号（例：原rank 1、tp 8 -> 8，与iter_dp_rank_assignments一致）
            if multinode:
                config.parallel_config.data_parallel_rank *= config.tensor_parallel_size
            # 从此主进程和所有EngineCore看到的都是TP=1（Config.tp_world_size是属性，也随之变成1）
            config.tensor_parallel_size = 1
        else:
            # 普通情况：engine数 = 本节点DP rank数 × PP stage数；PP与DP暂不能同时大于1
            dp_size = config.parallel_config.data_parallel_size
            assert not (
                pp_size > 1 and dp_size > 1
            ), "Pipeline parallel combined with data parallel is not supported yet."
            local_engine_count = len(rank_assignments) * pp_size

        # 全局engine数（路由用）：普通情况 = dp × pp；dp-attention下data_parallel_size已经乘过tp
        global_engine_count = (
            config.parallel_config.data_parallel_size * pp_size
            if not config.enable_dp_attention
            else config.parallel_config.data_parallel_size
        )
        # Global DP rank 0's node owns the router; the others only host engines.
        # 【译】全局DP rank 0所在的节点持有路由器；其他节点只托管engine。
        self.is_coordinator = not multinode or pc.data_parallel_rank == 0
        # 多机：按master端口推出每个engine的TCP端口；单机为None，后面走IPC地址
        socket_plan = (
            build_internode_dp_socket_plan(
                engine_count=global_engine_count,
                master_port=pc.data_parallel_master_port,
            )
            if multinode
            else None
        )

        # Inter-stage ZMQ channels (head<->downstream metadata, last->head
        # tokens), shared across the single dp group. PP+DP would need per-group
        # sets — deferred with the assertion above. Not shared state: only this
        # class's spawn loop reads them.
        # 【译】stage之间的ZMQ通道（head<->下游的元数据、最后一个stage->head的token），在唯一的dp组内共享。
        #   PP+DP需要每组一套——和上面的断言一起推迟了。不是共享状态：只有本类的spawn循环读它们。
        # 【分支】PP的stage间通信，见第9批 pp_engine_core.py / distributed/pp_transport.py
        self.pp_meta_addrs = []
        self.pp_token_addr = ""
        self.pp_kv_status_addr = ""
        if pp_size > 1:
            self.pp_meta_addrs = [get_open_zmq_ipc_path() for _ in range(pp_size)]
            self.pp_token_addr = get_open_zmq_ipc_path()
            self.pp_kv_status_addr = get_open_zmq_ipc_path()

        # ---------- 3. 初始化共享字段（见_init_shared_state） ----------
        self._init_shared_state(
            config,
            label="Engine Core Mgr",
            local_engine_count=local_engine_count,
            global_engine_count=global_engine_count,
        )

        # ---------- 4. 强制spawn ----------
        # multiprocessing的启动方式：fork复制父进程的整个内存（包括已初始化的GPU运行时，子进程里再用CUDA/HIP会崩溃）；
        #   spawn启动一个全新的Python解释器、重新import模块，安全但慢。CLAUDE.md规定必须用spawn
        # get_start_method(allow_none=True)返回None表示还没人设置过；force=False：已经设置过就不覆盖
        # torch.multiprocessing重新导出了标准multiprocessing的函数，这里设置的是全局默认方式，后面的multiprocessing.Process也用它
        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        # 每个engine的Process对象、地址、dp_rank、config：先全部创建，再统一start
        processes_info = []
        local_dp_ranks = []

        try:
            # ---------- 5. 起TCPStore（必须在下面deepcopy配置之前：要把实际端口写进config） ----------
            self._start_distributed_store(config, multinode=multinode)
            # ---------- 6. 为每个engine创建Process对象（此时还没start） ----------
            # engine_index -> (分配, pp_rank)：例 dp=1、pp=2 时 engine 0 = (dp0, pp0)，engine 1 = (dp0, pp1)
            for engine_index in range(self.local_engine_count):
                assignment_index = engine_index // self.pp_size
                dp_rank, local_dp_rank = rank_assignments[assignment_index]
                pp_rank = engine_index % self.pp_size
                logger.info(
                    f"{self.label}: Creating EngineCore engine {engine_index}"
                    f" (global dp={dp_rank}, local dp={local_dp_rank}, "
                    f"pp={pp_rank}) of {self.local_engine_count}"
                )

                # Create config for this (dp, pp) stage
                # 【译】为这个(dp, pp) stage创建配置
                import copy

                # deepcopy：每个engine一份独立的Config副本，改rank字段不影响别的engine；它会随Process参数被pickle传给子进程
                rank_config = copy.deepcopy(config)
                rank_config.parallel_config.data_parallel_rank = dp_rank
                rank_config.parallel_config.data_parallel_rank_local = local_dp_rank
                rank_config.parallel_config.pipeline_parallel_rank = pp_rank
                if self.pp_size > 1:
                    rank_config.parallel_config.pp_meta_addrs = self.pp_meta_addrs
                    rank_config.parallel_config.pp_token_addr = self.pp_token_addr
                    rank_config.parallel_config.pp_kv_status_addr = (
                        self.pp_kv_status_addr
                    )

                # 多机：engine端连接 tcp://<master_ip>:port（协调节点在0.0.0.0上bind）；单机为None，由launch_engine_core分配ipc地址
                if socket_plan is not None:
                    plan = socket_plan[dp_rank]
                    ip = config.parallel_config.data_parallel_master_ip
                    engine_addresses = {
                        "input_address": f"tcp://{ip}:{plan.input_port}",
                        "output_address": f"tcp://{ip}:{plan.output_port}",
                        "control_address": f"tcp://{ip}:{plan.control_port}",
                    }
                else:
                    engine_addresses = None

                # 只创建Process对象并分配地址，不start
                engine_core_process, addresses, local_dp_rank = launch_engine_core(
                    rank_config, dp_rank, local_dp_rank, addresses=engine_addresses
                )

                processes_info.append(
                    {
                        "process": engine_core_process,
                        "addresses": addresses,
                        "dp_rank": dp_rank,
                        "config": rank_config,
                    }
                )
                local_dp_ranks.append(local_dp_rank)

            try:
                # No visible-device mask is published here. Device placement is
                # owned by ModelRunner._setup_device_and_distributed, which
                # selects an ABSOLUTE cuda:{local_dp_rank*tp_size+rank}. Masking
                # the child as well would renumber its devices and compound the
                # two offsets -- see set_device_control_env_var's docstring.
                # 【译】这里不发布可见设备掩码。设备放置由 ModelRunner._setup_device_and_distributed 负责，它选择一个绝对编号的
                #   cuda:{local_dp_rank*tp_size+rank}。如果再给子进程设掩码，就会给它的设备重新编号，让两个偏移叠加——
                #   见 set_device_control_env_var 的docstring。
                # 解释：例 -dp 2 -tp 4：DP rank 1的4个ModelRunner分别用cuda:4~7，所有进程都能看到全部8张卡
                # ---------- 7. 统一start所有EngineCore进程 ----------
                for info, local_dp_rank in zip(processes_info, local_dp_ranks):
                    dp_rank = info["dp_rank"]
                    logger.info(
                        f"{self.label}: Starting EngineCore for DP rank "
                        f"{dp_rank}/{self.global_engine_count}"
                    )
                    info["process"].start()
                    self.engine_core_processes.append(info["process"])

                # 【分支】多机DP的工作节点：只托管engine，不建socket（全部由协调节点bind），在这里join子进程直到它们退出
                if not self.is_coordinator:
                    # A worker node hosts engines but owns no router: the
                    # coordinator binds every socket. Wait for the children and
                    # return -- raising here would be caught by the enclosing
                    # handler and reported as a startup failure.
                    # 【译】工作节点托管engine但不持有路由器：协调节点bind所有socket。等待子进程并返回——在这里raise会被外层的
                    #   处理器捕获，并被报告为启动失败。
                    logger.info(
                        f"{self.label}: worker node for global DP ranks "
                        f"{[i['dp_rank'] for i in processes_info]}; "
                        f"coordinator at "
                        f"{config.parallel_config.data_parallel_master_ip}"
                    )
                    self._finalizer = weakref.finalize(self, self.close)
                    self.async_output_queue = None
                    self._output_handler_task = None
                    self._asyncio_mode = False
                    for proc in self.engine_core_processes:
                        proc.join()
                    return

                # ---------- 8. bind三个socket并与每个engine握手 ----------
                # 多机：协调节点在0.0.0.0（所有网卡）上bind全部全局engine的端口；单机：用launch_engine_core分配的ipc地址
                if socket_plan is not None:
                    bind_addresses = [
                        {
                            "input_address": f"tcp://0.0.0.0:{p.input_port}",
                            "output_address": f"tcp://0.0.0.0:{p.output_port}",
                            "control_address": f"tcp://0.0.0.0:{p.control_port}",
                        }
                        for p in socket_plan
                    ]
                else:
                    bind_addresses = [info["addresses"] for info in processes_info]

                # ZMQ socket类型速览（本批反复出现）：
                #   ROUTER / DEALER：异步的请求-应答对。ROUTER收到的每条消息前面会多出一帧"发送方identity"；ROUTER发送时也要把
                #     identity放在第一帧，ZMQ据此决定发给哪个对端。DEALER收发的就是消息本身
                #   PUSH / PULL：单向流水线，PUSH只能发，PULL只能收
                #   PAIR：一对一的双向socket，这里配合inproc://（同一进程内线程之间的通信）当"关闭信号"用
                #   bind / connect：与socket类型无关，谁bind谁就是地址固定的一端；ZMQ允许先connect后bind（会自动重连）
                for addresses in bind_addresses:
                    input_socket = make_zmq_socket(
                        self.ctx, addresses["input_address"], zmq.ROUTER, bind=True
                    )
                    # 握手：阻塞等待EngineCore输入线程的DEALER发来的那个空帧，recv_multipart返回 [identity, b""]
                    #   记下identity，之后_send_request用它把消息路由到这个engine
                    identity, _ = input_socket.recv_multipart()
                    self.input_sockets.append(input_socket)
                    self.engine_core_identities.append(identity)

                    # 控制socket同样握手一次
                    control_socket = make_zmq_socket(
                        self.ctx, addresses["control_address"], zmq.ROUTER, bind=True
                    )
                    control_identity, _ = control_socket.recv_multipart()
                    self.control_sockets.append(control_socket)
                    self.control_identities.append(control_identity)

                    # PULL always binds; the engine's PUSH always connects.
                    # True for ipc:// and tcp:// alike -- the transport
                    # difference is carried by the address (the engine gets
                    # tcp://<master_ip>:port, we bind tcp://0.0.0.0:port).
                    # 【译】PULL总是bind；engine的PUSH总是connect。ipc:// 和 tcp:// 都是如此——传输方式的区别体现在地址里
                    #   （engine拿到的是 tcp://<master_ip>:port，我们bind的是 tcp://0.0.0.0:port）。
                    output_socket = make_zmq_socket(
                        self.ctx,
                        addresses["output_address"],
                        zmq.PULL,
                        bind=True,
                    )
                    self.output_sockets.append(output_socket)
                    # 每个输出线程一个inproc地址，close()时用来通知它退出
                    self.shutdown_paths.append(get_open_zmq_inproc_path())

                # ---------- 9. 等所有engine完成初始化（模型加载、KV分配、CUDA Graph捕获都在EngineCore里完成） ----------
                self._wait_for_all_ready_signals()
                logger.info(
                    f"{self.label}: All EngineCores are fully initialized and ready"
                )

                # ---------- 10. 为每个rank启动输出线程（READY已在上面被同步读掉，之后的消息都由输出线程处理） ----------
                for dp_rank in range(len(self.output_sockets)):
                    output_thread = self._create_output_thread(
                        dp_rank,
                        self.output_sockets[dp_rank],
                        self.shutdown_paths[dp_rank],
                    )
                    output_thread.start()
                    self.output_threads.append(output_thread)

            finally:
                # A worker node reaches this `finally` via its normal early
                # return above, with its children already joined and dead.
                # The liveness check does not apply to it -- dead children are
                # expected, not a startup failure.
                # 【译】工作节点是通过上面正常的提前return到达这个finally的，那时它的子进程已经join完、都已退出。
                #   存活检查不适用于它——子进程退出是预期的，不是启动失败。
                # 解释：协调节点上只要有任何一个EngineCore进程已经退出（finished_procs），就视为启动失败并清理
                if self.is_coordinator and self.finished_procs():
                    logger.error(
                        f"{self.label}: Some processes failed to start, shutting down all"
                    )
                    self.close()
                    raise RuntimeError("Failed to start all EngineCore processes")

        except Exception as e:
            # 任何一步失败：close()关掉已启动的进程和socket，再把异常抛给LLMEngine
            logger.error(
                f"{self.label}: Failed to initialize all EngineCores, cleaning up: {e}"
            )
            self.close()
            raise

        logger.info(
            f"{self.label}: All {len(self.output_sockets)} EngineCores initialized and ready"
        )
        # weakref.finalize(obj, func)：obj被垃圾回收或解释器退出时自动调用func（最多调用一次），
        #   这样即使用户忘了调用close()，EngineCore进程也会被关掉
        self._finalizer = weakref.finalize(self, self.close)
        # asyncio模式：输出改由asyncio.Queue交付（get_output_async）。Config.asyncio_mode默认False，
        #   仓库里没有把它设为True的调用方
        self.async_output_queue = asyncio.Queue() if config.asyncio_mode else None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    # 处理一个rank的READY附带的数据：max_pool_tokens取所有rank中的最小值；调用方：两种READY等待函数
    def _record_ready_payload(self, data) -> None:
        """Fold one rank's READY facts into the manager's view of capacity."""
        # 【译】把一个rank的READY事实合并进manager对容量的认识。
        if not data:
            return
        reported = data.get("max_pool_tokens")
        if reported is None:
            return
        if self.max_pool_tokens is None:
            self.max_pool_tokens = reported
        else:
            self.max_pool_tokens = min(self.max_pool_tokens, reported)

    # 在所有rank的输出socket上并行等待READY（谁先好先收谁），全部收到才返回
    #   READY由EngineCore.__init__最后的_send_ready_signal经输出线程发出；初始化期间收到SHUTDOWN或其他消息都算启动失败
    #   zmq.Poller：同时监视多个socket，poll()阻塞到其中任意一个可读，返回 [(socket, 事件), ...]
    def _wait_for_all_ready_signals(self):
        """Wait for READY signals from all DP ranks in parallel (no timeout)."""
        # 【译】并行等待所有DP rank的READY信号（没有超时）。
        poller = zmq.Poller()
        for dp_rank, output_socket in enumerate(self.output_sockets):
            poller.register(output_socket, zmq.POLLIN)

        engine_count = len(self.output_sockets)
        ready_received = [False] * engine_count
        remaining = engine_count

        while remaining > 0:
            # 【译】（行尾注释）无限期等待
            socks = poller.poll()  # Wait indefinitely
            if not socks:
                continue

            for socket, _ in socks:
                # Find which DP rank this socket belongs to
                # 【译】找出这个socket属于哪个DP rank
                dp_rank = self.output_sockets.index(socket)
                if ready_received[dp_rank]:
                    continue

                # copy=False：返回zmq.Frame，零拷贝地引用ZMQ的缓冲区；pickle.loads可以直接读它
                obj = socket.recv(copy=False)
                request_type, data = pickle.loads(obj)

                if request_type == EngineCoreRequestType.READY:
                    logger.info(
                        f"{self.label}: DP rank {dp_rank} is fully initialized and ready"
                    )
                    self._record_ready_payload(data)
                    ready_received[dp_rank] = True
                    remaining -= 1
                elif request_type == EngineCoreRequestType.SHUTDOWN:
                    raise RuntimeError(
                        f"{self.label}: Received unexpected SHUTDOWN signal from DP rank {dp_rank} during initialization"
                    )
                else:
                    raise RuntimeError(
                        f"{self.label}: Expected READY signal from DP rank {dp_rank}, but got {request_type}"
                    )

    # 为一个DP rank创建输出线程（返回Thread对象，由调用方start）：在PULL socket上收EngineCore发来的消息并分发
    #   消息类型（EngineCoreRequestType）与处理：
    #     STREAM            [(seq_id, RequestOutput), ...]：本step的流式增量 -> 调用每个seq的流式回调；
    #                       第一次输出说明prefill已完成，释放它的prefill token负载；seq结束时摘掉回调、释放全部负载
    #     ADD               list[Sequence]：已结束的序列（也包括被调度器拒绝的）-> 补发终止输出、释放负载、放进outputs_queue
    #     METRICS           指标快照 -> latest_metrics[dp_rank]
    #     UTILITY_RESPONSE  工具RPC的回复 -> 按request_id交给等待它的调用方
    #     SHUTDOWN          EngineCore要退出了 -> 线程结束
    #   另有一个inproc的PAIR socket：close()往里发一个空消息，让线程退出
    #   注意：同一个ADD类型码在两个方向含义不同——发给EngineCore是新请求，从EngineCore发回来是已结束的序列
    # threading.Thread(target=f, daemon=True)：在新线程里运行f；daemon线程不会阻止进程退出（主线程结束时被直接终止）
    # 对比nano-vllm：nano-vllm的step()同步拿到token，不需要后台线程；ATOM的结果是从子进程异步推回来的，
    #   所以每个DP rank要有一个线程专门收
    def _create_output_thread(
        self, dp_rank: int, output_socket: zmq.Socket, shutdown_path: str
    ) -> Thread:
        def process_outputs_socket():
            assert isinstance(output_socket, zmq.Socket)
            # 关闭信号socket：本线程bind，close()里connect并发一个空消息（inproc要求两端用同一个Context）
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(output_socket, zmq.POLLIN)
                logger.debug(f"{self.label} (DP {dp_rank}): output thread started")
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    # 关闭信号优先：两个socket同时可读时也直接退出
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        # 【译】关闭信号，退出线程。
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive shutdown signal"
                        )
                        break

                    obj = output_socket.recv(copy=False)
                    try:
                        request_type, data = pickle.loads(obj)
                    except Exception:
                        # Every request on this DP rank waits on this thread,
                        # so one message it cannot decode must not end it.
                        # 【译】这个DP rank上的每个请求都在等这个线程，所以一条它解码不了的消息不能让它结束。
                        logger.exception(
                            f"{self.label} (DP {dp_rank}): dropping an engine "
                            f"message that cannot be decoded"
                        )
                        continue
                    # EngineCore退出前（_send_engine_dead）会发来SHUTDOWN；这里再向该rank的控制socket回送一个SHUTDOWN
                    #   （本地进程还活着、或是远端rank时，见_shutdown_engine_core_rank），然后结束本线程
                    if request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive SHUTDOWN request"
                        )
                        self._shutdown_engine_core_rank(dp_rank)
                        break
                    elif request_type == EngineCoreRequestType.STREAM:
                        # 【译】（行尾注释）(seq_id, RequestOutput) 元组的列表
                        stream_outputs = data  # List of (seq_id, RequestOutput) tuples
                        logger.debug(
                            f"{self.label}: Received STREAM message with {len(stream_outputs)} outputs"
                        )
                        # Delivered only through the per-seq callbacks below.
                        # These also used to go onto stream_outputs_queue,
                        # which nothing ever read, so every RequestOutput
                        # stayed reachable for the life of the process and
                        # made each gen-2 GC pass progressively slower.
                        #
                        # The f-strings below are built by the caller before
                        # logger.debug() can drop them, so check the level
                        # once per step rather than twice per chunk.
                        # 【译】只通过下面的逐seq回调交付。它们以前还会放进 stream_outputs_queue，可是从来没有任何地方读它，
                        #   于是每个RequestOutput在进程的整个生命周期内都保持可达，让每一次第2代GC扫描越来越慢。
                        #
                        #   下面的f-string在 logger.debug() 有机会丢弃它们之前就已经由调用方构造好了，所以每个step检查一次日志级别，
                        #   而不是每个chunk检查两次。
                        dbg = logger.isEnabledFor(logging.DEBUG)
                        for seq_id, request_output in stream_outputs:
                            # The first emitted model output means this
                            # sequence's prompt prefill has completed. Keep the
                            # request count charged for decode pressure, but
                            # stop advertising its prompt as queued prefill.
                            # 【译】第一个发出的模型输出意味着这个序列的prompt prefill已经完成。保留请求数的记账（反映decode压力），
                            #   但不再把它的prompt当作排队中的prefill来报告。
                            self._mark_seq_prefill_complete(seq_id)
                            callback = self._seq_id_to_callback.get(seq_id)
                            if dbg:
                                logger.debug(
                                    f"{self.label}: seq_id={seq_id}, callback={'found' if callback is not None else 'NOT FOUND'}, tokens={request_output.output_tokens}"
                                )
                            # 回调由api_server注册：它只把RequestOutput缓冲进一个线程局部的批里，真正投递在下面的_flush_stream_batch_fn
                            if callback is not None:
                                try:
                                    callback(request_output)
                                    if dbg:
                                        logger.debug(
                                            f"{self.label}: Successfully called callback for seq_id={seq_id}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"Error calling stream_callback for sequence {seq_id}: {e}",
                                        exc_info=True,
                                    )
                            # 序列结束：摘掉回调，释放它在DP负载里的全部记账
                            if request_output.finished:
                                self._seq_id_to_callback.pop(seq_id, None)
                                self._release_seq_load(seq_id)
                                if dbg:
                                    logger.debug(
                                        f"{self.label}: Cleaned up callback for finished sequence {seq_id}"
                                    )
                        # Batched stream dispatch: the per-seq callbacks only buffer
                        # their chunks into a thread-local; flush the whole step's
                        # buffer into the per-request stream collectors now (one
                        # call_soon_threadsafe per loop). Resolved lazily by the API
                        # server to avoid the api_server <-> engine_core_mgr import
                        # cycle. No-op when no streaming request is in flight.
                        # 【译】按批分发流式输出：逐seq的回调只是把chunk缓冲进一个线程局部变量；现在把整个step的缓冲一次性刷到每个请求的
                        #   流收集器里（每个事件循环一次 call_soon_threadsafe）。由API server延迟设置，以避免 api_server <-> engine_core_mgr
                        #   的循环导入。没有流式请求在途时什么也不做。
                        if self._flush_stream_batch_fn is not None:
                            try:
                                self._flush_stream_batch_fn()
                            except Exception as e:
                                logger.warning(
                                    f"{self.label}: flush_stream_batch failed: {e}",
                                    exc_info=True,
                                )
                    elif request_type == EngineCoreRequestType.METRICS:
                        # EngineCore按ATOM_METRICS_UPDATE_INTERVAL_S（默认1秒）主动推送的指标快照
                        self.latest_metrics[dp_rank] = data
                    elif request_type == EngineCoreRequestType.UTILITY_RESPONSE:
                        # 工具RPC的回复：按request_id投递给正在等待的调用方（见_route_utility_response）
                        self._route_utility_response(dp_rank, data)
                    elif request_type == EngineCoreRequestType.ADD:
                        # （下一行是被注释掉的旧调试代码）
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        seqs = data
                        # Offline (non-streaming) completions arrive here as
                        # finished sequences; release their in-flight DP load.
                        #
                        # So do sequences the scheduler rejected before they
                        # ever ran (`_unschedulable_reason`, abort-while-waiting)
                        # — those never reach `postprocess`, so no STREAM chunk
                        # is ever built for them. An online client is still
                        # holding a callback and would wait forever, so the
                        # terminal output it is owed has to be raised here.
                        # Anything already delivered through STREAM has had its
                        # callback popped by then (STREAM is enqueued ahead of
                        # the finished-seq list and this socket is FIFO), so a
                        # normal completion finds nothing to do below.
                        # 【译】离线（非流式）的完成结果以"已结束序列"的形式到达这里；释放它们在途的DP负载。
                        #
                        #   调度器在序列运行之前就拒绝掉的那些（_unschedulable_reason、等待中被abort）也会到这里——它们永远到不了
                        #   postprocess，所以从来没有为它们构造过STREAM chunk。在线客户端仍然持有回调，会一直等下去，所以欠它的终止输出
                        #   必须在这里补发。凡是已经通过STREAM交付过的，回调那时已经被摘掉了（STREAM先于已结束序列列表入队，而且这个socket
                        #   是FIFO的），所以正常完成的序列在下面什么也不用做。
                        delivered = False
                        for seq in seqs:
                            delivered |= self._deliver_terminal_output(seq)
                            self._release_seq_load(seq.id)
                        if delivered and self._flush_stream_batch_fn is not None:
                            # The callbacks above only buffer into a thread-local;
                            # without this flush the chunk never reaches the
                            # request's collector and the client still hangs.
                            # 【译】上面的回调只是缓冲进线程局部变量；没有这次刷新，chunk永远到不了请求的收集器，客户端仍会卡住。
                            try:
                                self._flush_stream_batch_fn()
                            except Exception as e:
                                logger.warning(
                                    f"{self.label}: flush_stream_batch failed: {e}",
                                    exc_info=True,
                                )
                        # 交给get_output()（离线generate的主循环在那里阻塞等待）；put_nowait：不阻塞地放入（Queue没有上限，不会满）
                        self.outputs_queue.put_nowait(seqs)
            finally:
                # Close sockets.
                # 【译】关闭socket。
                # 解释：linger=0：关闭时丢弃还没发出的消息，不等待
                shutdown_socket.close(linger=0)
                output_socket.close(linger=0)

        # 线程名形如EngineCoreOutputThread-DP0
        return Thread(
            target=process_outputs_socket,
            name=f"EngineCoreOutputThread-DP{dp_rank}",
            daemon=True,
        )

    # asyncio模式下，确保后台任务_async_output_handler已在当前事件循环里启动（只启动一次）；调用方：get_output_async
    #   asyncio.get_running_loop()：没有正在运行的事件循环时抛RuntimeError
    #   loop.create_task(coro)：把协程包装成Task放进事件循环并发执行
    def _ensure_output_handler_task(self):
        if self._asyncio_mode and self._output_handler_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._output_handler_task = loop.create_task(
                    self._async_output_handler()
                )
            except RuntimeError:
                # If no running event loop, try to get/create one
                # 【译】如果没有正在运行的事件循环，尝试获取/创建一个
                # 解释：不管取到的还是新建的，只要这个循环没在运行就报错——实际上要求调用方处在一个运行中的事件循环里
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._output_handler_task = loop.create_task(
                        self._async_output_handler()
                    )
                else:
                    raise RuntimeError(
                        "CoreManager with asyncio_mode requires a running event loop"
                    )

    # 后台协程：把线程安全的outputs_queue里的结果搬到asyncio.Queue，供get_output_async在协程里await
    #   loop.run_in_executor(None, f)：在默认线程池里执行阻塞函数f（这里是Queue.get），await它不会卡住事件循环
    async def _async_output_handler(self):
        loop = asyncio.get_event_loop()
        while True:
            # Use run_in_executor to avoid blocking event loop
            # 【译】用 run_in_executor，避免阻塞事件循环
            seqs = await loop.run_in_executor(None, self.outputs_queue.get)
            # 收到异常对象就转交并结束（仓库里目前没有往outputs_queue放异常的地方，属于防御性代码）
            if isinstance(seqs, BaseException):
                await self.async_output_queue.put(seqs)
                break
            await self.async_output_queue.put(seqs)

    # get_output的asyncio版本：await一批已结束的序列；没开asyncio_mode时报错
    async def get_output_async(self) -> list[Sequence]:
        if not self.async_output_queue:
            raise RuntimeError("Engine async mode not enabled")

        # Ensure output handler task is started
        # 【译】确保输出处理任务已经启动
        self._ensure_output_handler_task()

        seqs = await self.async_output_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    # 关闭所有EngineCore：发SHUTDOWN -> 关socket -> 停输出线程 -> 等进程退出（超时则terminate/kill）-> 释放TCPStore
    #   调用方：LLMEngine.close()、weakref.finalize（对象被回收/解释器退出）、__init__失败时的清理；靠_closed保证只执行一次
    # 一个DP rank的关闭链路：
    #   CoreManager.close ── 控制socket发SHUTDOWN ──► EngineCore输入线程：input_queue放入[退出哨兵]，线程结束
    #   ──► busy_loop：pull_and_process_input_queue见到哨兵 -> 跳出循环 -> run_engine的finally调用engine.exit()
    #   ──► exit()：通知ModelRunner退出并join -> _send_engine_dead：output_queue放入[退出哨兵]
    #   ──► EngineCore输出线程：发出SHUTDOWN帧后结束（CoreManager的输出线程此时多半已被下面的inproc信号停掉）
    def close(self):
        if self._closed:
            return
        self._closed = True

        logger.info(
            f"{self.label}: Shutting down {len(self.input_sockets)} EngineCores"
        )

        # 1. 给每个rank发SHUTDOWN（走控制socket）
        for dp_rank in range(len(self.input_sockets)):
            self._shutdown_engine_core_rank(dp_rank)

        # 2. 关闭请求socket和控制socket
        for input_socket in self.input_sockets:
            if not input_socket.closed:
                input_socket.close()

        for control_socket in self.control_sockets:
            if not control_socket.closed:
                control_socket.close()

        # 3. 经inproc PAIR通知每个输出线程退出；with块结束时自动关闭这个临时socket
        for shutdown_path in self.shutdown_paths:
            if shutdown_path:
                try:
                    with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                        shutdown_sender.connect(shutdown_path)
                        shutdown_sender.send(b"")
                except Exception as e:
                    logger.debug(f"{self.label}: Error sending shutdown signal: {e}")

        # 4. 等输出线程结束（每个最多0.5秒）
        for thread in self.output_threads:
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

        # Wait for EngineCore processes to exit gracefully.
        # Use a single deadline so all processes share the grace period
        # instead of sequential per-process timeouts.  This prevents early
        # process exits from destroying the NCCL TCPStore while later
        # processes' HeartbeatMonitor threads still depend on it.
        # 【译】等待EngineCore进程优雅退出。所有进程共用一个截止时间、共享这段宽限期，而不是每个进程依次各等一次超时。
        #   这样可以避免先退出的进程销毁NCCL的TCPStore，而后面进程的HeartbeatMonitor线程仍依赖它。
        # 解释：ATOM_SHUTDOWN_TIMEOUT_S默认5秒
        import time

        deadline = time.monotonic() + envs.ATOM_SHUTDOWN_TIMEOUT_S
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                remaining = max(deadline - time.monotonic(), 0)
                proc.join(timeout=remaining)

        # Terminate any that are still alive.
        # 【译】终止所有仍然存活的进程。
        # 解释：terminate()发SIGTERM；EngineCore里init_exit_handler装的处理函数会调用exit()
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.terminate()
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.join(timeout=1)

        # Final join + close to release sentinel semaphores
        # 【译】最后再join并close，释放哨兵（sentinel）信号量
        # 解释：kill()发SIGKILL（无法被捕获）；Process.close()释放进程对象占用的资源，之后不能再使用这个对象
        for proc in self.engine_core_processes:
            if proc is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=1)
                try:
                    proc.close()
                except (ValueError, OSError):
                    pass

        # Release the rendezvous server after stopping the local engine processes.
        # 【译】在停掉本地engine进程之后再释放集合点服务器。
        self._distributed_stores.clear()
        logger.info(f"{self.label}: All EngineCores shut down")

    # 把一个已pickle的请求发给指定engine（热路径：每次add_request都走这里）
    #   ROUTER发送时第一帧必须是目标对端的identity，ZMQ据此路由；对端DEALER只收到payload这一帧
    #   copy=False：零拷贝发送，ZMQ直接引用payload的内存
    def _send_request(self, dp_rank: int, payload: bytes) -> None:
        """Send one already-pickled request to an engine core. Hot path.

        Deliberately unsynchronized: ``input_sockets`` carries nothing but
        ``add_request``, which runs on a single thread (the API server's event
        loop online, the caller's thread offline). Everything that can be sent
        from another thread -- utility commands, abort, shutdown -- goes to
        :meth:`_send_control` on a separate socket instead.

        That separation is load-bearing, not stylistic. A ZMQ socket is not
        thread-safe: two unserialized ``send_multipart`` calls interleave their
        frames, the DEALER on the other end then reads a routing identity where
        a payload should be, and its input thread dies on ``UnpicklingError``.
        Nothing recovers from that -- the engine spins on a forever-empty input
        queue, the workers idle, and every client hangs with no error logged
        anywhere but that thread's own traceback. So: never send to
        ``input_sockets`` from anywhere but here.
        """
        # 【译】把一个已经pickle好的请求发给一个engine core。热路径。
        #
        #   故意不加同步：input_sockets 上只走 add_request，而它只在一个线程上运行（在线时是API server的事件循环，
        #   离线时是调用方的线程）。其他所有可能从别的线程发出的东西——工具命令、abort、关闭——都改走 :meth:`_send_control`，
        #   用另一个socket。
        #
        #   这种分离是承重的，不是风格问题。ZMQ socket不是线程安全的：两个没有串行化的 send_multipart 调用会让各自的帧交错，
        #   另一端的DEALER于是在本该是payload的位置读到一个路由identity，它的输入线程就死于 UnpicklingError。没有任何东西
        #   能从中恢复——engine在一个永远为空的输入队列上空转，worker闲着，每个客户端都卡住，除了那个线程自己的traceback
        #   之外，哪里都没有错误日志。所以：除了这里，永远不要往 input_sockets 发送。
        # 解释：v0.1.8的EngineCore输入线程已经会捕获解码失败并丢弃那一帧（见process_input_sockets），不再整个线程死掉；
        #   但交错的帧仍会让请求丢失
        self.input_sockets[dp_rank].send_multipart(
            [self.engine_core_identities[dp_rank], payload], copy=False
        )

    # 发送一条已pickle的控制消息（工具RPC、abort、SHUTDOWN）：加锁串行化，因为可能有多个线程同时发
    def _send_control(self, dp_rank: int, payload: bytes, copy: bool = False) -> None:
        """Send one already-pickled control message. Serialized, never hot.

        Writers here are the event loop (abort, the /debug/* endpoints) and the
        per-rank output threads (shutdown), so this does need a lock -- but none
        of them runs per request, so its cost never lands on admission.
        """
        # 【译】发送一条已经pickle好的控制消息。串行化，从不在热路径上。
        #
        #   这里的写者是事件循环（abort、/debug/* 接口）和每个rank的输出线程（关闭），所以确实需要一把锁——
        #   但它们都不是每个请求都会跑的，所以锁的开销永远落不到请求接纳上。
        # 解释：copy参数：广播时同一个payload要发给多个rank，调用方传copy=True让ZMQ各自拷贝一份
        with self._control_send_lock:
            self.control_sockets[dp_rank].send_multipart(
                [self.control_identities[dp_rank], payload], copy=copy
            )

    # 接收一批Sequence并发往EngineCore（离线：LLMEngine.add_request；在线：api_server直接调用）
    #   1. 把流式回调从Sequence上摘下来存进_seq_id_to_callback（回调是闭包，无法pickle，也只有主进程用得上）
    #   2. 选目标engine：PP -> 只发给stage 0（PP head）；只有1个可路由的engine -> 发给它；否则 -> DP负载均衡
    # 对比nano-vllm：nano-vllm的add_request直接scheduler.add(seq)；ATOM要跨进程发送，还要在多个DP rank之间分配
    def add_request(self, seqs: list[Sequence]):
        logger.debug(
            f"{self.label}: Add request, sequence ids: {[seq.id for seq in seqs]}"
        )
        # Register callbacks before sending to engine core
        # 【译】在发给engine core之前先注册回调
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None
        if self.pp_size > 1:
            # Pipeline parallel (dp=1): requests enter only at stage 0, which
            # drives the pipeline downstream.
            # 【译】流水线并行（dp=1）：请求只从stage 0进入，由它驱动下游的流水线。
            logger.debug(f"{self.label}: Add {len(seqs)} requests to PP head 0")
            self._send_request(0, _add_frame(seqs))
        elif self._routable_engine_count == 1:
            # Single routable engine, send all requests
            # 【译】只有一个可路由的engine，把所有请求都发给它
            logger.debug(f"{self.label}: Add {len(seqs)} requests to DP rank 0")
            self._send_request(0, _add_frame(seqs))
        else:
            self._dispatch_to_dp_ranks(seqs)

    # 可以路由到的engine数（= global_engine_count；测试里手工构造、没有这个字段的对象退回local_engine_count）
    @property
    def _routable_engine_count(self) -> int:
        """How many engine ranks this manager may route to.

        Equals ``global_engine_count``, which on a coordinator spans other
        nodes. Falls back to ``local_engine_count`` for callers that build the
        routing state by hand instead of through ``_init_shared_state`` --
        ``tests/test_dp_load_balance.py`` does exactly that, and single-node
        callers predate the global/local split, where the two are equal anyway.
        """
        # 【译】这个manager可以路由到多少个engine rank。
        #
        #   等于 global_engine_count，在协调节点上它包括其他节点的。对于不通过 _init_shared_state、而是手工构造路由状态的
        #   调用方，退回到 local_engine_count——tests/test_dp_load_balance.py 正是这样做的；单节点的调用方早于全局/本地的区分，
        #   那时两者本来就相等。
        return getattr(self, "global_engine_count", self.local_engine_count)

    # 取出每个seq显式指定的data_parallel_rank（客户端可以指定发往哪个DP rank），一次性校验整批；调用方：_dispatch_to_dp_ranks
    def _resolve_and_validate_hints(self, seqs: list[Sequence]) -> list[int | None]:
        """Resolve every seq's explicit ``data_parallel_rank`` hint and validate
        the whole batch, once.

        Returns the per-seq resolved hint (an int rank, or None for a
        load-balanced seq) so the dispatch loop can reuse it instead of calling
        getattr/int a second time per seq.

        Validation runs BEFORE any load is charged so a bad hint in the middle
        of a batch cannot leave earlier siblings charged-but-undispatched (a
        permanent in-flight-load leak).
        """
        # 【译】解析每个seq显式的 data_parallel_rank 提示，并对整批做一次校验。
        #
        #   返回每个seq解析后的提示（int类型的rank，负载均衡的seq则为None），这样分发循环可以复用它，而不必对每个seq
        #   再调用一次 getattr/int。
        #
        #   校验在记任何负载之前进行，这样批中间的一个错误提示不会让前面的兄弟请求"记了账却没发出去"（一个永久性的在途负载泄漏）。
        hints: list[int | None] = []
        engine_count = self._routable_engine_count
        for seq in seqs:
            raw = getattr(seq, "data_parallel_rank", None)
            hint = None if raw is None else int(raw)
            if hint is not None and not 0 <= hint < engine_count:
                raise ValueError(
                    f"Invalid data_parallel_rank={hint}; "
                    f"global_engine_count={engine_count}"
                )
            hints.append(hint)
        return hints

    # DP>1时把一批seq分到各个DP rank，并给每个rank发一个ADD帧
    #   步骤：① 校验显式rank提示 ② 持锁逐个选rank并记负载 ③ 逐rank发送 ④ 发送失败时回滚没发出去的负载 ⑤ 打一行汇总日志
    # 例（默认least_requests，2个rank，初始空闲，游标为0）：一次add 4个seq，prompt长度 [1000, 200, 300, 50]
    #   seq0 -> rank0（两个rank都是(0,0)，从游标位置rank0开始扫）   reqs=[1,0] tokens=[1000,0]
    #   seq1 -> rank1（(0,0) < (1,1000)）                          reqs=[1,1] tokens=[1000,200]
    #   seq2 -> rank1（请求数打平，比token：200 < 1000）             reqs=[1,2] tokens=[1000,500]
    #   seq3 -> rank0（请求数 1 < 2）                               reqs=[2,2] tokens=[1050,500]
    #   结果：rank0收到[seq0, seq3]，rank1收到[seq1, seq2]，各发一个ADD帧
    def _dispatch_to_dp_ranks(self, seqs: list[Sequence]) -> None:
        """Route a batch across DP ranks and send each rank its sub-batch.

        Honors an explicit ``data_parallel_rank`` hint; otherwise picks a rank
        via ``_select_dp_rank_locked`` (load-aware by default). Selection and the
        in-flight-load charge happen atomically under ``_lb_lock`` so a burst of
        requests spreads across ranks instead of all landing on the current
        minimum.
        """
        # 【译】把一批请求路由到各个DP rank，并给每个rank发送它的子批。
        #
        #   尊重显式的 data_parallel_rank 提示；否则通过 _select_dp_rank_locked 选一个rank（默认是负载感知的）。
        #   选择和在途负载的记账在 _lb_lock 下原子地完成，这样一阵突发请求会分散到各个rank，而不是全都落到当前的最小值上。
        # Resolve + validate all hints in one pass first — no charging until the
        # whole batch is known good, so a rejected batch never leaks partial
        # load. The resolved hints are reused in the loop below to avoid a second
        # getattr/int pass per seq.
        # 【译】先一次性解析并校验所有提示——在确认整批没问题之前不记账，这样被拒绝的批永远不会泄漏部分负载。
        #   解析好的提示在下面的循环里复用，避免对每个seq再做一遍 getattr/int。
        hints = self._resolve_and_validate_hints(seqs)

        # round_robin normally skips load bookkeeping. Session affinity still
        # needs queued-prefill counters even if the fallback strategy is RR.
        # 【译】round_robin通常跳过负载记账。会话亲和即使在回退策略是RR时也需要排队prefill计数器。
        track_load = (
            self._dp_lb_strategy != "round_robin" or self._dp_session_affinity_enabled
        )
        engine_count = self._routable_engine_count
        # dp_seqs[r]：分给rank r的seq列表
        dp_seqs = [[] for _ in range(engine_count)]
        reqs_snapshot = tokens_snapshot = None
        # 持锁：选rank和记账必须原子地完成（输出线程会并发地释放负载）
        with self._lb_lock:
            for seq, hint in zip(seqs, hints):
                dp_rank = self._select_dp_rank_for_seq_locked(seq, hint)
                if track_load:
                    self._charge_seq_load_locked(seq, dp_rank)
                dp_seqs[dp_rank].append(seq)
            # Copy the counters under the lock so the snapshot log below is a
            # consistent instant, not a torn read racing _release_seq_load.
            # 【译】在锁内复制计数器，这样下面的快照日志是一个一致的瞬间，而不是与 _release_seq_load 竞争的撕裂读取。
            if track_load:
                reqs_snapshot = list(self._rank_reqs)
                tokens_snapshot = list(self._rank_tokens)

        # Track which ranks were actually handed off, plus a compact per-rank
        # delta ("rankR:Nreq/Ttok") for the single summary log after the loop. If
        # a send fails partway, the seqs on the not-yet-dispatched ranks were
        # charged above but will never produce a finished output to release them,
        # so we roll back their in-flight load before propagating — otherwise
        # routing skews forever.
        # 【译】记录哪些rank真正发出去了，外加每个rank一段紧凑的增量（"rankR:Nreq/Ttok"），用于循环之后的那一行汇总日志。
        #   如果中途发送失败，还没发出的那些rank上的seq在上面已经记了账，却永远不会产生一个"已结束"输出来释放它们，
        #   所以我们先回滚它们的在途负载再把异常抛出去——否则路由会永久性地偏斜。
        dispatched = [False] * engine_count
        added = []
        try:
            for dp_rank, rank_seqs in enumerate(dp_seqs):
                if not rank_seqs:
                    continue
                self._send_request(dp_rank, _add_frame(rank_seqs))
                dispatched[dp_rank] = True
                batch_prefill_tokens = sum(
                    int(getattr(seq, "num_prompt_tokens", 0) or 0) for seq in rank_seqs
                )
                added.append(
                    f"rank{dp_rank}: {len(rank_seqs)} req / {batch_prefill_tokens} tok"
                )
        except Exception:
            # Roll back only ranks we never handed off. _release_seq_load is
            # idempotent (pops from _seq_load), so even if a failing send had
            # already delivered its frames and the engine finished + released
            # those seqs on an output thread, this rollback cannot double-count:
            # whichever release runs first wins, the other is a no-op.
            # 【译】只回滚我们从未发出的rank。_release_seq_load 是幂等的（从 _seq_load 里pop），所以即使某个失败的发送其实已经
            #   把帧送达，engine也已经完成并在输出线程上释放了那些seq，这次回滚也不会重复扣减：哪个释放先跑就算哪个，
            #   另一个什么也不做。
            if track_load:
                for dp_rank, rank_seqs in enumerate(dp_seqs):
                    if rank_seqs and not dispatched[dp_rank]:
                        for seq in rank_seqs:
                            self._release_seq_load(seq.id)
            raise

        # One line per add: the per-rank delta this add placed, plus (for the
        # load-aware strategies) the resulting in-flight distribution across all
        # ranks, so a single grep shows both what changed and how balanced it is.
        # 【译】每次add打一行日志：这次add在每个rank上放了多少，外加（对负载感知的策略）由此得到的所有rank的在途分布，
        #   这样一次grep就能同时看到改变了什么、以及有多均衡。
        # 解释：上面的例子会打出 "add rank0: 2 req / 1050 tok, rank1: 2 req / 500 tok | in-flight reqs=[2, 2] queued_prefill_tokens=[1050, 500]"
        if reqs_snapshot is not None:
            logger.info(
                "%s: add %s | in-flight reqs=%s queued_prefill_tokens=%s",
                self.label,
                ", ".join(added),
                reqs_snapshot,
                tokens_snapshot,
            )
        else:
            logger.info("%s: add %s", self.label, ", ".join(added))

    # 按策略为一个新请求选DP rank（不考虑会话亲和）；调用方必须持有_lb_lock；调用方：_select_dp_rank_for_seq_locked
    # 例：3个rank，_rank_reqs=[3,2,2]，_rank_tokens=[100,900,300]，req_equiv=512（默认）
    #   least_requests：比较 (请求数, token数) -> (3,100)、(2,900)、(2,300) -> 选rank2
    #   least_tokens：  比较 token + 512×请求数 -> 1636、1924、1324 -> 选rank2
    #   round_robin：   不看负载，按游标轮流 0、1、2、0……
    def _select_dp_rank_locked(self) -> int:
        """Pick a DP engine rank for a new request. Caller must hold _lb_lock.

        - "round_robin": load-agnostic rotation.
        - "least_requests" (default): fewest in-flight requests, ties broken by
          the lighter in-flight prompt-token load. Request count keeps the
          lockstep DP ranks in phase; the token tie-break packs pending prefill
          work evenly across the equal-request ranks.
        - "least_tokens": lowest combined load ``tokens + req_equiv * reqs``
          (prefill pressure + decode-slot pressure).

        Fully-tied ranks are resolved by a rotating cursor so selection does not
        always fall on rank 0. See docs/distributed_guide.md for the rationale.
        """
        # 【译】为一个新请求选一个DP engine rank。调用方必须持有 _lb_lock。
        #
        #   - "round_robin"：与负载无关的轮转。
        #   - "least_requests"（默认）：在途请求最少者优先，平局时看在途prompt token负载更轻者。请求数让锁步运行的DP rank
        #     保持同相；token平局裁决则把待处理的prefill工作均匀地分摊到请求数相同的rank上。
        #   - "least_tokens"：综合负载 tokens + req_equiv * reqs 最低者（prefill压力 + decode槽位压力）。
        #
        #   完全打平的rank由一个轮转游标裁决，这样选择不会总落在rank 0。原理见 docs/distributed_guide.md。
        n = self._routable_engine_count
        if self._dp_lb_strategy == "round_robin":
            dp_rank = self._rank_rotation_cursor % n
            self._rank_rotation_cursor += 1
            return dp_rank

        # argmin over per-rank load, scanned from a rotating start offset so a run
        # of fully-equal ranks spreads evenly. Scores are computed inline (no
        # intermediate list) — the loop reads the counters directly.
        # 【译】对每个rank的负载求argmin，从一个轮转的起点偏移开始扫描，这样一串完全相等的rank会被均匀地分散。
        #   分数直接内联计算（不建中间列表）——循环直接读计数器。
        least_requests = self._dp_lb_strategy == "least_requests"
        best_rank = 0
        best_score = None
        offset = self._rank_rotation_cursor % n
        for i in range(n):
            r = (offset + i) % n
            if least_requests:
                # Lexicographic (request count, prompt-token load): tuples compare
                # element-wise, so tokens only decide among request-count ties.
                # 【译】按字典序比较 (请求数, prompt-token负载)：元组逐元素比较，所以token只在请求数打平时才起决定作用。
                score = (self._rank_reqs[r], self._rank_tokens[r])
            else:  # "least_tokens"
                # 【译】（上一行行尾注释）"least_tokens"
                score = (
                    self._rank_tokens[r] + self._dp_lb_req_equiv * self._rank_reqs[r]
                )
            # 严格小于：分数相同时保留先扫到的rank（离游标起点最近的那个）
            if best_score is None or score < best_score:
                best_score = score
                best_rank = r
        self._rank_rotation_cursor += 1
        return best_rank

    # 用rendezvous hashing（最高随机权重哈希）把会话映射到固定rank：对每个rank算 hash(会话id + rank)，取最大者
    #   v0.1.8里仓库内没有调用方（新会话改用下面的_select_new_session_rank_locked按负载放置）
    def _stable_session_rank(self, session_id: str) -> int:
        """Map a session to a stable rank with rendezvous hashing.

        Python's built-in ``hash`` is process-randomized, so it cannot define
        cache ownership. Rendezvous hashing is deterministic and, unlike a
        simple modulo, remaps only sessions owned by a rank that is added or
        removed. The DP width is tiny, and this runs only once per new session.
        """
        # 【译】用rendezvous hashing把一个会话映射到一个稳定的rank。
        #
        #   Python内置的 hash 在每个进程里是随机化的，所以不能用它来定义缓存归属。rendezvous hashing是确定性的，而且不同于
        #   简单的取模，增删一个rank时只会重新映射原本属于那个rank的会话。DP宽度很小，而且每个新会话只运行一次。
        session_key = str(session_id).encode("utf-8")
        best_rank = 0
        best_score = -1
        for rank in range(self._routable_engine_count):
            # blake2b：标准库的哈希函数，digest_size=8取64位；rank.to_bytes(4, "little")把rank编码成4字节小端整数
            digest = hashlib.blake2b(
                session_key + rank.to_bytes(4, "little"), digest_size=8
            ).digest()
            score = int.from_bytes(digest, "little")
            if score > best_score:
                best_rank = rank
                best_score = score
        return best_rank

    # 为一个新的粘性会话选rank：取综合负载（token + req_equiv×请求数）最轻的rank，负载相同时用rendezvous hash打破平局
    def _select_new_session_rank_locked(self, session_id: str) -> int:
        """Place a new sticky session on the lightest DP rank.

        Existing sessions never call this function: locality wins once an
        owner has cache state.  Before that first placement there is no cache
        to preserve, so use estimated outstanding prefill debt plus the
        configured token-equivalent decode pressure.  Rendezvous hashing is
        only the deterministic tie-breaker; it must not override real load.
        """
        # 【译】把一个新的粘性会话放到最轻的DP rank上。
        #
        #   已有的会话从不调用这个函数：一旦拥有者有了缓存状态，局部性优先。在第一次放置之前没有缓存需要保留，所以用
        #   估计的未完成prefill负担加上配置的、折合成token的decode压力。rendezvous hashing只是确定性的平局裁决，
        #   不能凌驾于真实负载之上。
        session_key = str(session_id).encode("utf-8")
        best_rank = 0
        best_load = None
        best_hash = -1
        for rank in range(self._routable_engine_count):
            load = (
                self._rank_tokens[rank] + self._dp_lb_req_equiv * self._rank_reqs[rank]
            )
            tie_hash = int.from_bytes(
                hashlib.blake2b(
                    session_key + rank.to_bytes(4, "little"), digest_size=8
                ).digest(),
                "little",
            )
            if (
                best_load is None
                or load < best_load
                or (load == best_load and tie_hash > best_hash)
            ):
                best_rank = rank
                best_load = load
                best_hash = tie_hash
        return best_rank

    # 记一次路由决策（decision是_dp_route_counters的键），并原样返回rank
    def _record_dp_route_locked(self, decision: str, rank: int) -> int:
        """Account for one routing decision and return its target rank."""
        # 【译】为一次路由决策记账，并返回它的目标rank。
        self._dp_route_counters[decision] += 1
        self._rank_routed_total[rank] += 1
        return rank

    # 为一个seq选rank，优先级：显式提示 > 会话已有的拥有者 > 负载均衡
    #   seq.dp_session_id / dp_parent_session_id：api_server从请求里取出的会话id（agent类负载）
    #   调用方：_dispatch_to_dp_ranks（已持有_lb_lock）
    def _select_dp_rank_for_seq_locked(
        self, seq: Sequence, explicit_rank: int | None
    ) -> int:
        """Route one sequence using explicit hint, strict owner, then load.

        This intentionally matches SGLang Model Gateway's agentic routing
        semantics: a stable correlation/session id selects one DP rank, and
        every later turn stays there. Load cannot move an existing session;
        doing so turns a cheap cache hit into a potentially huge prefill.

        Parent lineage does not affect placement. Each child correlation id is
        its own sticky session, preventing sibling subagents from dogpiling the
        root's rank. Requests without a session retain normal load balancing.
        """
        # 【译】按照 显式提示、严格的拥有者、然后负载 的顺序路由一个序列。
        #
        #   这有意与 SGLang Model Gateway 的agent路由语义保持一致：一个稳定的关联/会话id选定一个DP rank，之后的每一轮都
        #   留在那里。负载不能移动一个已有的会话；那样做会把一次廉价的缓存命中变成一次可能非常大的prefill。
        #
        #   父级血缘不影响放置。每个子关联id都是自己独立的粘性会话，防止兄弟子agent全都挤到根会话所在的rank上。
        #   没有会话的请求保留正常的负载均衡。
        if explicit_rank is not None:
            if self._dp_session_affinity_enabled:
                # An explicit placement is authoritative and becomes the
                # session's owner for subsequent unhinted requests.
                # 【译】显式的放置是权威的，并会成为该会话后续未带提示的请求的拥有者。
                session_id = getattr(seq, "dp_session_id", None)
                if session_id:
                    old_owner = self._dp_session_owners.get(session_id)
                    if old_owner is not None and old_owner != explicit_rank:
                        # The new rank cannot be assumed to own the old prefix.
                        # 【译】不能假设新rank拥有旧的前缀。
                        self._dp_session_prompt_tokens.pop(session_id, None)
                    self._dp_session_owners[session_id] = explicit_rank
            return self._record_dp_route_locked("explicit_total", explicit_rank)

        session_id = getattr(seq, "dp_session_id", None)
        parent_id = getattr(seq, "dp_parent_session_id", None)
        # 没开会话亲和，或请求不带会话id：普通负载均衡
        if not self._dp_session_affinity_enabled or not session_id:
            rank = self._select_dp_rank_locked()
            return self._record_dp_route_locked("load_balanced_total", rank)

        # 新会话（或记录的拥有者越界）：按负载选一个拥有者并记下，此后这个会话的请求都发往它
        owner = self._dp_session_owners.get(session_id)
        if owner is None or not 0 <= owner < self._routable_engine_count:
            owner = self._select_new_session_rank_locked(session_id)
            self._dp_session_owners[session_id] = owner
            if parent_id:
                self._dp_route_counters["affinity_parent_ignored_total"] += 1
            logger.debug(
                "%s: DPA load-aware affinity new session=%s parent=%s owner=rank%d "
                "load=%d tokens=%d reqs=%d",
                self.label,
                session_id,
                parent_id,
                owner,
                self._rank_tokens[owner]
                + self._dp_lb_req_equiv * self._rank_reqs[owner],
                self._rank_tokens[owner],
                self._rank_reqs[owner],
            )
            return self._record_dp_route_locked("affinity_new_total", owner)

        return self._record_dp_route_locked("affinity_owner_hit_total", owner)

    # 返回DP路由状态的一致快照（持锁复制）；调用方：LLMEngine的统计接口
    def get_dp_router_statistics(self) -> dict:
        """Return a consistent, non-mutating snapshot of DP routing state."""
        # 【译】返回DP路由状态的一个一致的、不修改状态的快照。
        with self._lb_lock:
            sessions_per_rank = [0] * self._routable_engine_count
            for rank in self._dp_session_owners.values():
                if 0 <= rank < self._routable_engine_count:
                    sessions_per_rank[rank] += 1
            return {
                "enabled": self._routable_engine_count > 1,
                **self._dp_route_counters,
                "requests_per_rank": list(self._rank_routed_total),
                "inflight_requests_per_rank": list(self._rank_reqs),
                "queued_prefill_tokens_per_rank": list(self._rank_tokens),
                "session_count_per_rank": sessions_per_rank,
            }

    # 把一个seq的在途负载记到dp_rank上：请求数+1，token数+prompt长度（会话亲和时只记相对上一轮的增长）
    #   (dp_rank, req_cost, tok_cost)记进_seq_load，释放时按原值扣回；调用方必须持有_lb_lock
    # 例（会话亲和）：会话S第1轮prompt 5000 -> tok_cost=5000；第2轮prompt 5600 -> tok_cost=600（前5000个token预计命中前缀缓存）
    def _charge_seq_load_locked(self, seq: Sequence, dp_rank: int) -> None:
        """Record a seq's in-flight load on dp_rank. Caller must hold _lb_lock."""
        # 【译】在 dp_rank 上记录一个seq的在途负载。调用方必须持有 _lb_lock。
        req_cost = 1
        prompt_tokens = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        tok_cost = prompt_tokens
        session_id = getattr(seq, "dp_session_id", None)
        if self._dp_session_affinity_enabled and session_id:
            previous_prompt_tokens = self._dp_session_prompt_tokens.get(session_id)
            if previous_prompt_tokens is not None:
                # Agentic turns normally extend their previous prompt.  Charge
                # only that extension; request-equivalent load still accounts
                # for lookup/decode pressure when the delta is zero.
                # 【译】agent的各轮通常是在上一轮prompt的基础上延长。只对延长的部分记账；当增量为0时，按请求折合的负载
                #   仍然反映了查找/decode压力。
                tok_cost = max(0, prompt_tokens - previous_prompt_tokens)
            self._dp_session_prompt_tokens[session_id] = prompt_tokens
        self._rank_reqs[dp_rank] += req_cost
        self._rank_tokens[dp_rank] += tok_cost
        self._seq_load[seq.id] = (dp_rank, req_cost, tok_cost)

    # 给"从未运行过"的seq（被调度器拒绝，或在等待中被abort）补发一个终止输出，否则在线客户端会一直等
    #   返回是否调用了回调；调用方：输出线程的ADD分支
    def _deliver_terminal_output(self, seq) -> bool:
        """Give an online client the terminal output for a seq that never streamed.

        A sequence the scheduler rejected before it ran produced no tokens and
        no STREAM chunk, so nothing has answered the client yet. Synthesises the
        finished `RequestOutput` the normal path would have built in
        `postprocess`, carrying `leave_reason` as the finish reason so the
        response says why it ended rather than closing empty.

        Returns whether a callback was invoked, so the caller knows to flush the
        batch the callback buffered into. Offline sequences register no callback
        and normal completions have had theirs popped by the STREAM that
        delivered them, so both answer False and keep their existing path.
        """
        # 【译】为一个从未产生流式输出的seq，给在线客户端交付终止输出。
        #
        #   一个在运行之前就被调度器拒绝的序列不产生token，也没有STREAM chunk，所以还没有任何东西回应过客户端。这里合成
        #   正常路径本会在 postprocess 里构造的那个"已结束"RequestOutput，把 leave_reason 作为结束原因，让响应说明它为什么
        #   结束，而不是空着关闭。
        #
        #   返回是否调用了回调，这样调用方知道要刷新回调缓冲进去的那一批。离线序列不注册回调，正常完成的序列的回调已经被
        #   交付它们的STREAM摘掉了，所以两者都返回False，保持各自原来的路径。
        callback = self._seq_id_to_callback.pop(seq.id, None)
        if callback is None:
            return False
        # Never empty: an empty reason reaches the client as `finish_reason:
        # null`, which the OpenAI schema reserves for a choice still being
        # generated -- so a client would keep waiting on a finished stream.
        # 【译】永远不为空：空原因到了客户端就是 finish_reason: null，而OpenAI的schema把它留给仍在生成中的choice——
        #   于是客户端会在一个已经结束的流上一直等。
        reason = seq.leave_reason or "rejected"
        try:
            callback(
                RequestOutput(
                    request_id=seq.id,
                    output_tokens=[],
                    finished=True,
                    finish_reason=reason,
                )
            )
        except Exception as e:
            logger.warning(
                f"Error delivering terminal output for sequence {seq.id}: {e}",
                exc_info=True,
            )
            return False
        logger.info(
            "%s: seq %s returned without running: %s", self.label, seq.id, reason
        )
        return True

    # seq结束或被abort时撤销它的在途负载；调用方：输出线程、abort_request、发送失败时的回滚
    def _release_seq_load(self, seq_id) -> None:
        """Undo a seq's in-flight load when it finishes or is aborted.

        Idempotent: a seq is only charged once and released once, so a repeated
        call (e.g. finish followed by abort) is a no-op.
        """
        # 【译】当一个seq结束或被abort时，撤销它的在途负载。
        #
        #   幂等：一个seq只会被记账一次、释放一次，所以重复调用（例如先完成再abort）什么也不做。
        with self._lb_lock:
            entry = self._seq_load.pop(seq_id, None)
            if entry is None:
                return
            dp_rank, req_cost, tok_cost = entry
            self._rank_reqs[dp_rank] -= req_cost
            self._rank_tokens[dp_rank] -= tok_cost

    # seq第一次产生输出时（即prefill完成）只释放它的prefill token负载，请求数保留到结束；调用方：输出线程的STREAM分支
    def _mark_seq_prefill_complete(self, seq_id) -> None:
        """Release only a sequence's prefill-token charge, once."""
        # 【译】只释放一个序列的prefill-token负载，且只释放一次。
        # Every decode token reaches this path. The prefill charge may be zero
        # at admission, or already released by the first output. Avoid taking
        # the router lock for a no-op; re-read under the lock before changing
        # counters because abort/finish may release the charge concurrently.
        # 【译】每个decode token都会走到这条路径。prefill负载可能在接纳时就是0，或者已经被第一个输出释放了。对无事可做的
        #   情况避免去拿路由锁；但在修改计数器之前要在锁内重新读一次，因为abort/完成可能同时在释放负载。
        # 解释：这是"双重检查"——锁外的快速检查过滤掉绝大多数decode token，锁内的检查保证正确性
        entry = self._seq_load.get(seq_id)
        if entry is None:
            return
        _, _, tok_cost = entry
        if tok_cost == 0:
            return
        with self._lb_lock:
            entry = self._seq_load.get(seq_id)
            if entry is None:
                return
            dp_rank, req_cost, tok_cost = entry
            if tok_cost == 0:
                return
            self._rank_tokens[dp_rank] -= tok_cost
            self._seq_load[seq_id] = (dp_rank, req_cost, 0)

    # 清空所有DP路由状态（游标、在途负载、会话拥有者）；调用方：LLMEngine.generate等离线批处理的开头、rollout
    def reset_dp_router(self) -> None:
        """Reset all DP routing state (rotation cursor + in-flight load).

        Called at the start of a fresh offline ``generate()`` batch so counts do
        not leak across independent batches and DP assignment is deterministic.

        Precondition: the previous batch has fully drained. If any request is
        still charged when this runs (e.g. this CoreManager is being shared with
        a concurrent streaming path), that request's later release becomes a
        no-op and the per-rank counters would drift — so we warn loudly instead
        of corrupting accounting silently.
        """
        # 【译】重置所有DP路由状态（轮转游标 + 在途负载）。
        #
        #   在一次新的离线 generate() 批开始时调用，这样计数不会在相互独立的批之间泄漏，DP分配也是确定性的。
        #
        #   前提：上一批已经完全排空。如果运行它时还有请求被记着账（例如这个CoreManager同时被一个并发的流式路径共享），
        #   那个请求后来的释放就会变成空操作，每个rank的计数器就会漂移——所以我们大声警告，而不是悄悄地破坏记账。
        with self._lb_lock:
            if self._seq_load:
                logger.warning(
                    "%s: reset_dp_router() called with %d request(s) still "
                    "charged in-flight; dropping their load. Expected only "
                    "between fully-drained offline batches — a shared/concurrent "
                    "CoreManager will see counters drift.",
                    self.label,
                    len(self._seq_load),
                )
            self._rank_rotation_cursor = 0
            self._rank_reqs = [0] * self._routable_engine_count
            self._rank_tokens = [0] * self._routable_engine_count
            self._seq_load.clear()
            self._dp_session_owners.clear()
            self._dp_session_prompt_tokens.clear()

    # 发送一个只带命令名的工具RPC（不等回复）：dp_rank为None时发给所有rank；走控制socket
    #   权重同步专属的命令不允许从这里发；调用方：LLMEngine（如get_mtp_stats）
    def send_utility_command(self, cmd: str, dp_rank: int | None = None):
        self._refuse_unowned_weight_sync_cmd(cmd, None)
        if dp_rank is None:
            # Send to all DP ranks
            # 【译】发给所有DP rank
            for rank in range(len(self.control_sockets)):
                logger.debug(
                    f"{self.label}: Send utility command '{cmd}' to DP rank {rank}"
                )
                self._send_control(
                    rank, pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd}))
                )
        else:
            logger.debug(
                f"{self.label}: Send utility command '{cmd}' to DP rank {dp_rank}"
            )
            self._send_control(
                dp_rank, pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd}))
            )

    # 客户端断开时取消一个请求：先释放本地的DP负载记账，再向所有DP rank广播"abort_request"工具命令
    #   调用方：api_server（检测到客户端断开时）、atomesh服务
    #   EngineCore侧：输入线程放进utility_queue -> busy_loop里utility_handler调用scheduler.abort_request（第3批）
    def abort_request(self, req_id):
        """Tell the engine core(s) to drop a request (client disconnected).

        Broadcast to every DP rank (only the one holding ``req_id`` acts). The
        scheduler finishes the seq at its next step via the normal stop path,
        freeing its KV blocks. Fire-and-forget; safe if the seq already finished.
        """
        # 【译】告诉engine core丢弃一个请求（客户端已断开）。
        #
        #   广播给每个DP rank（只有持有 req_id 的那个会处理）。调度器在下一个step通过正常的停止路径结束这个seq，
        #   释放它的KV block。发出即不管；即使seq已经结束也是安全的。
        # Release DP load bookkeeping now: an aborted seq may never emit a
        # finished STREAM output, so relying on the finish path alone would leak
        # its in-flight count. _release_seq_load is idempotent.
        # 【译】现在就释放DP负载记账：一个被abort的seq可能永远不会发出"已结束"的STREAM输出，所以只依赖完成路径会泄漏
        #   它的在途计数。_release_seq_load 是幂等的。
        self._release_seq_load(req_id)
        try:
            self.broadcast_utility_command("abort_request", req_id=req_id)
        except Exception as e:  # noqa: BLE001 - disconnect cleanup must not raise
            # 【译】（上一行行尾注释）noqa: BLE001 - 断连清理不能抛异常
            logger.warning(f"{self.label}: abort_request({req_id}) failed: {e}")

    # 输出线程收到UTILITY_RESPONSE后调用：按回复里的request_id投递给登记了这个id的调用方
    def _route_utility_response(self, dp_rank: int, data):
        """Hand a utility reply to the caller waiting on its request id.

        A reply nobody waits for is dropped: one with no usable id was asked
        for by a sender that never waits, and one with an id nobody holds is
        late, its caller having given up.
        """
        # 【译】把一个工具回复交给正在等它请求id的调用方。
        #
        #   没人等的回复会被丢弃：没有可用id的回复，是一个从不等待的发送方要的；有id但没人持有的回复是迟到的，
        #   它的调用方已经放弃了。
        request_id = data.get("request_id") if isinstance(data, dict) else None
        if not isinstance(request_id, str) or not request_id:
            # Routing an unhashable id would raise on this thread and end it.
            # 【译】路由一个不可哈希的id会在这个线程上抛异常，让线程结束。
            logger.debug(
                f"{self.label}: dropping a utility reply no caller can be "
                f"waiting for, from DP rank {dp_rank}"
            )
            return
        if not self._rpc_router.route(request_id, (dp_rank, data)):
            logger.warning(
                f"{self.label}: dropping late utility reply {request_id} "
                f"from DP rank {dp_rank}"
            )

    # 集合RPC：在每个DP engine的每个TP runner上执行method，按 DP为主、TP为次 的顺序返回RpcResult列表
    #   流程：生成request_id -> 登记回复队列 -> 广播COLLECTIVE_RPC_CMD工具命令 -> 等每个DP rank一个回复（有截止时间）
    #     -> _flatten_dp_replies展开成每个TP rank一条
    #   调用方：rollout/async_engine.py（RL，第10批）；PP和DisaggCoreManager下不支持
    # 例：-dp 2 -tp 4 -> 返回8条，顺序为 (dp0, tp0..3)、(dp1, tp0..3)
    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 300.0,
    ) -> list[RpcResult]:
        """Run *method* on every DP engine's every TP runner.

        Returns replies in DP-major then TP-rank order. Every rank is
        represented: a DP engine that did not answer in time contributes one
        failed :class:`RpcResult` per TP rank rather than silently shortening
        the list, because a caller checking coverage needs to know which ranks
        are missing.

        Never raises for a worker-side failure -- the failure travels in
        ``RpcResult.error`` so the ranks that did succeed are still reported.
        Raises up front for a timeout that is not a finite number of seconds,
        and for a manager whose engines are not one DP rank each, which the
        replies would otherwise be reported as.
        """
        # 【译】在每个DP engine的每个TP runner上运行 *method*。
        #
        #   按DP为主、TP rank为次的顺序返回回复。每个rank都有代表：在规定时间内没有回答的DP engine会为每个TP rank贡献一个
        #   失败的 :class:`RpcResult`，而不是悄悄缩短列表，因为检查覆盖率的调用方需要知道缺了哪些rank。
        #
        #   对worker侧的失败从不抛异常——失败放在 RpcResult.error 里传回，这样成功的rank仍会被报告。对于不是有限秒数的
        #   timeout，以及engine不是"每个一个DP rank"的manager，会提前抛异常，否则回复会被当成DP rank来报告。
        timeout = checked_timeout(timeout)
        unsupported = self._collective_rpc_unsupported()
        if unsupported is not None:
            raise NotImplementedError(
                f"{self.label}: collective_rpc needs one engine per DP rank, "
                f"but {unsupported}"
            )
        request_id = uuid.uuid4().hex
        engine_count = len(self.control_sockets)
        deadline = time.monotonic() + timeout
        # engine侧的预算比调用方的等待略短（engine_budget），让engine报告的失败原因能在调用方放弃之前到达；跨进程所以用墙钟time.time()
        worker_deadline = time.time() + engine_budget(timeout)

        with self._rpc_router.register(request_id) as replies:
            self.broadcast_utility_command(
                COLLECTIVE_RPC_CMD,
                request_id=request_id,
                method=method,
                args=tuple(args),
                kwargs=dict(kwargs or {}),
                barrier=bool(barrier),
                timeout=engine_budget(timeout),
                deadline=worker_deadline,
            )
            by_dp_rank: dict[int, dict] = {}
            while len(by_dp_rank) < engine_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    dp_rank, body = replies.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                if dp_rank in by_dp_rank:
                    logger.warning(
                        f"{self.label}: duplicate collective_rpc reply from "
                        f"DP rank {dp_rank}, ignoring"
                    )
                    continue
                by_dp_rank[dp_rank] = body

        return self._flatten_dp_replies(request_id, method, engine_count, by_dp_rank)

    # 返回不支持集合RPC的原因（PP：每个engine只是一个stage）；支持时返回None。DisaggCoreManager重写了它
    def _collective_rpc_unsupported(self) -> str | None:
        """Why this manager's engines are not one DP rank each, if they are not."""
        # 【译】如果这个manager的engine不是每个一个DP rank，说明原因。
        pp_size = getattr(self, "pp_size", 1)
        if pp_size > 1:
            # One engine per stage, each holding a slice of the layers.
            # 【译】每个stage一个engine，各自只持有一部分层。
            return f"its {pp_size} engines are pipeline stages"
        return None

    # 把"每个DP rank一个回复"展开成"每个TP rank一个RpcResult"；没回复或engine整体失败的DP rank，为它的每个TP rank补一条错误
    def _flatten_dp_replies(
        self,
        request_id: str,
        method: str,
        engine_count: int,
        by_dp_rank: dict[int, dict],
    ) -> list[RpcResult]:
        """DP-major, TP-minor, with a placeholder for every absent rank."""
        # 【译】DP为主、TP为次，每个缺席的rank都有一个占位。
        ranks = self._rpc_ranks_per_engine
        flat: list[RpcResult] = []
        for dp_rank in range(engine_count):
            body = by_dp_rank.get(dp_rank)
            if body is None:
                error = (
                    f"DP rank {dp_rank} did not answer {method!r} before the deadline"
                )
                flat.extend(
                    RpcResult(request_id, tp, error=error) for tp in range(ranks)
                )
                continue
            if body.get("error"):
                # The engine failed before reaching its runners, so every one of
                # them shares the reason.
                # 【译】engine在到达它的runner之前就失败了，所以每个runner共享同一个原因。
                flat.extend(
                    RpcResult(request_id, tp, error=body["error"])
                    for tp in range(ranks)
                )
                continue
            for entry in body.get("results", []):
                flat.append(
                    RpcResult(
                        request_id,
                        entry.get("tp_rank", -1),
                        value=entry.get("value"),
                        error=entry.get("error"),
                    )
                )
        return flat

    # 【分支】RL权重热更新（weight sync），见第10批。下面这组方法实现"分桶更新权重"的协议：
    #   一次同步 = 若干个update_weights*分桶 + 最后一个分桶之后的FINISH；任何一个engine失败 -> 向所有engine发DISCARD
    # 本方法：权重同步专属的命令（含FINISH/DISCARD）只能由内部持有_weight_sync_dispatch_token的路径发送
    def _refuse_unowned_weight_sync_cmd(self, cmd: str, dispatch_token) -> None:
        if (
            cmd in _WEIGHT_SYNC_OWNED_CMDS
            and dispatch_token is not self._weight_sync_dispatch_token
        ):
            raise ValueError(
                f"{self.label}: {cmd!r} cannot be sent directly; weight syncs "
                f"go through broadcast_utility_command_sync, which alone "
                f"finishes or aborts them"
            )

    # 把一个工具命令（带任意kwargs）广播给所有engine，不等回复；只pickle一次，所有rank复用
    #   调用方：abort_request、collective_rpc、_broadcast_and_wait
    def broadcast_utility_command(self, cmd: str, *, _dispatch_token=None, **kwargs):
        self._refuse_unowned_weight_sync_cmd(cmd, _dispatch_token)
        payload = {"cmd": cmd, **kwargs}
        # Serialize once and reuse for all ranks (optimization: avoid repeated pickle.dumps)
        # 【译】只序列化一次，所有rank复用（优化：避免重复 pickle.dumps）
        serialized_payload = pickle.dumps((EngineCoreRequestType.UTILITY, payload))
        for rank in range(len(self.control_sockets)):
            logger.debug(
                f"{self.label}: Broadcast utility command '{cmd}' to DP rank {rank}"
            )
            # copy=True: the same buffer is reused for every rank.
            # 【译】copy=True：同一个缓冲区被每个rank复用。
            self._send_control(rank, serialized_payload, copy=True)

    # 一次权重同步会话（with core_mgr.weight_sync_session(): ...）：持有_weight_sync_lock，会话内的所有分桶共享一个sync_id
    #   @contextmanager：把一个带yield的生成器函数变成上下文管理器，yield之前相当于进入with，之后（finally）相当于退出with
    #   调用方：rollout/weight_sync.py
    @contextmanager
    def weight_sync_session(self):
        """Serialize one direct/SHM/IPC synchronization session.

        Its updates share one sync id until that sync finishes, and none can
        be sent once it has failed: the engines would otherwise take the rest
        of a failed sync's buckets for a new sync and commit them. A nested
        session joins the outer one's sync.
        """
        # 【译】串行化一次直接/共享内存/IPC同步会话。
        #
        #   它的各次更新共享一个sync id，直到那次同步结束；一旦失败就不能再发送：否则engine会把一次失败同步的剩余分桶
        #   当作一次新同步接收并提交。嵌套的会话加入外层会话的同步。
        with self._weight_sync_lock:
            ctx = self._weight_sync_context
            outer = getattr(ctx, "active", False)
            if not outer:
                ctx.sync_id = uuid.uuid4().hex
                ctx.failure = None
            ctx.active = True
            try:
                yield
            finally:
                ctx.active = outer

    # 同步失败时向所有engine发DISCARD，让它们保持隔离并丢弃这次同步的临时数据；对每个rank都尝试，从不抛异常
    def _discard_failed_weight_sync(self, failed_cmd: str) -> None:
        """Fence every engine this reaches and drop the failed sync's scratch.

        Tried on every rank and never raises: the caller is already raising
        the failure that matters, and a rank whose socket fails here must not
        keep its peers, which may hold the failed update, from hearing it.
        """
        # 【译】隔离（fence）这里能触达的每个engine，并丢弃失败同步的临时数据。
        #
        #   对每个rank都尝试，且从不抛异常：调用方已经在抛那个真正重要的失败了，而且一个在这里socket出错的rank，
        #   不能妨碍它的同伴（它们可能持有失败的更新）收到这个消息。
        payload = pickle.dumps(
            (
                EngineCoreRequestType.UTILITY,
                {"cmd": DISCARD_WEIGHT_SYNC_CMD, "failed_cmd": failed_cmd},
            )
        )
        for rank in range(len(self.control_sockets)):
            try:
                self._send_control(rank, payload, copy=True)
            except Exception:
                logger.exception(
                    f"{self.label}: could not tell DP rank {rank} to discard "
                    f"the failed {failed_cmd!r}"
                )

    # 同步版广播：发给所有engine并按DP顺序返回回复；任何engine报错或超时就抛异常
    #   先做合法性检查（FINISH不能直接发、DISCARD只能在会话内发、非最后分桶必须在会话内、fire-and-forget命令不能等）
    #   权重更新命令走_sync_weight_update，其他命令直接_broadcast_and_wait
    #   调用方：LLMEngine（start_profile等）、rollout
    def broadcast_utility_command_sync(
        self,
        cmd: str,
        timeout: float = 300.0,
        **kwargs,
    ):
        """Run *cmd* on every engine and return their replies in DP order.

        Raises if an engine reports an error or does not answer in time.
        """
        # 【译】在每个engine上运行 *cmd*，按DP顺序返回它们的回复。
        #
        #   如果某个engine报告错误或者没有及时回答，就抛异常。
        owns_sync = getattr(self._weight_sync_context, "active", False)
        if cmd == FINISH_WEIGHT_SYNC_CMD:
            raise ValueError(
                f"{self.label}: {cmd!r} commits a weight sync; only the sync "
                f"whose complete update succeeded on every DP engine sends it"
            )
        if cmd == DISCARD_WEIGHT_SYNC_CMD:
            if not owns_sync:
                raise ValueError(
                    f"{self.label}: {cmd!r} aborts the weight sync in flight; "
                    f"send it from inside the `with weight_sync_session()` that "
                    f"owns it"
                )
            ctx = self._weight_sync_context
            ctx.failure = ctx.failure or "aborted by its sender"
        is_nonfinal_bucket = cmd in WEIGHT_UPDATE_UTILITY_CMDS and not (
            completes_weight_sync(cmd, kwargs)
        )
        if is_nonfinal_bucket and not owns_sync:
            raise ValueError(
                f"{self.label}: non-final {cmd!r} needs "
                f"`with weight_sync_session()` so another sync cannot interleave"
            )
        if cmd in FIRE_AND_FORGET_UTILITY_CMDS:
            raise ValueError(
                f"{self.label}: {cmd!r} is fire-and-forget; its handler never "
                f"answers, so waiting on it could only time out. Send it with "
                f"broadcast_utility_command instead."
            )
        if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
            # Re-entrant, so the weight_sync_session() a bucket belongs to
            # takes it again rather than waiting on itself.
            # 【译】可重入，这样分桶所属的 weight_sync_session() 会再次获取它，而不是等待自己。
            with self._weight_sync_lock:
                return self._sync_weight_update(cmd, timeout, owns_sync, kwargs)
        return self._broadcast_and_wait(cmd, timeout, **kwargs)

    # 发送一次权重更新并带上它所属同步的sync_id：会话外则自成一次同步；会话内失败会关闭会话的同步，完成则开启下一次
    def _sync_weight_update(
        self, cmd: str, timeout: float, in_session: bool, kwargs: dict
    ):
        """Send one weight update tagged with the sync it belongs to.

        Outside a session the update is a whole sync on its own. Inside one, a
        failure closes the session's sync and a finish starts its next one.
        """
        # 【译】发送一次权重更新，并标上它所属的同步。
        #
        #   在会话之外，这次更新本身就是一次完整的同步。在会话之内，失败会关闭会话的同步，完成会开启它的下一次同步。
        if not in_session:
            return self._broadcast_and_wait(
                cmd, timeout, sync_id=uuid.uuid4().hex, **kwargs
            )
        ctx = self._weight_sync_context
        if ctx.failure is not None:
            raise RuntimeError(
                f"{self.label}: this session's weight sync already failed "
                f"({ctx.failure}), and a failed sync cannot be completed; "
                f"retry it from a new weight_sync_session()"
            )
        try:
            responses = self._broadcast_and_wait(
                cmd, timeout, sync_id=ctx.sync_id, **kwargs
            )
        except BaseException as exc:
            ctx.failure = f"{cmd}: {type(exc).__name__}: {exc}"
            raise
        if completes_weight_sync(cmd, kwargs):
            ctx.sync_id = uuid.uuid4().hex
        return responses

    # 广播一个工具命令并等所有engine回复（用request_id匹配回复，有截止时间）
    #   缺回复 -> TimeoutError；有engine报错 -> RuntimeError；权重更新失败时还要先DISCARD
    #   如果这次命令完成了一次权重同步，再发一轮FINISH（递归调用自己）
    def _broadcast_and_wait(self, cmd: str, timeout: float, **kwargs):
        """The round trip behind broadcast_utility_command_sync, unchecked.

        That method has already refused what its caller may not send, and for
        an update has taken the session lock and named the update's sync. The
        only other caller is the FINISH below, which runs under that lock.
        """
        # 【译】broadcast_utility_command_sync 背后的那次往返，不做检查。
        #
        #   那个方法已经拒绝了调用方不允许发送的内容，并且对一次更新已经拿到会话锁、命名了这次更新的同步。
        #   唯一的另一个调用方是下面的FINISH，它在那把锁下运行。
        timeout = checked_timeout(timeout)
        request_id = uuid.uuid4().hex
        # The global engine count on a coordinator, as the broadcast reaches.
        # 【译】协调节点上就是全局engine数，与广播能触达的范围一致。
        engine_count = len(self.control_sockets)
        deadline = time.monotonic() + timeout
        worker_deadline = time.time() + engine_budget(timeout)
        by_dp_rank: dict[int, object] = {}
        with self._rpc_router.register(request_id) as replies:
            try:
                self.broadcast_utility_command(
                    cmd,
                    _dispatch_token=self._weight_sync_dispatch_token,
                    request_id=request_id,
                    timeout=engine_budget(timeout),
                    deadline=worker_deadline,
                    **kwargs,
                )
            except Exception:
                # The engines before the failed send already hold this update.
                # 【译】在失败的那次发送之前的engine已经持有了这次更新。
                if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                    self._discard_failed_weight_sync(cmd)
                raise
            while len(by_dp_rank) < engine_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    dp_rank, body = replies.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                by_dp_rank.setdefault(dp_rank, body)

        responses = [by_dp_rank[dp_rank] for dp_rank in sorted(by_dp_rank)]
        missing = sorted(set(range(engine_count)) - set(by_dp_rank))
        if missing:
            if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                self._discard_failed_weight_sync(cmd)
            raise TimeoutError(
                f"{self.label}: no reply to utility command {cmd!r} from DP "
                f"rank(s) {missing} within {timeout}s"
            )
        failed = [r for r in responses if isinstance(r, dict) and r.get("error")]
        if failed:
            if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                # An engine whose local TP ranks all succeeded has no local
                # reason to discard its packed/expert/IPC scratch. Once any DP
                # engine failed, every engine belongs to the abandoned sync.
                # 【译】一个本地TP rank全部成功的engine，自己没有理由丢弃它的打包/专家/IPC临时数据。可一旦任何一个DP engine失败，
                #   每个engine都属于那次被放弃的同步。
                self._discard_failed_weight_sync(cmd)
            # Callers read r["result"]; an error reply has none, and handing it
            # back would turn a named failure into a KeyError somewhere else.
            # 【译】调用方读的是 r["result"]；错误回复里没有它，把它交回去会把一个有名字的失败变成别处的一个KeyError。
            raise RuntimeError(
                f"{self.label}: utility command {cmd!r} failed on {len(failed)} "
                f"of {len(responses)} engine(s): "
                + "; ".join(str(r["error"]) for r in failed)
            )
        if completes_weight_sync(cmd, kwargs):
            try:
                self._broadcast_and_wait(
                    FINISH_WEIGHT_SYNC_CMD,
                    _WEIGHT_SYNC_FINISH_TIMEOUT_S,
                    sync_id=kwargs.get("sync_id"),
                    completed_cmd=cmd,
                    completed_request_id=request_id,
                )
            except Exception:
                self._discard_failed_weight_sync(cmd)
                raise
        return responses

    # 给一个rank发SHUTDOWN（走控制socket）：本地进程还活着，或者是没有本地进程的远端rank（多机协调节点上）
    #   调用方：close()、输出线程收到SHUTDOWN时
    def _shutdown_engine_core_rank(self, dp_rank: int):
        # Determine whether this rank has a local process and/or a control socket.
        # On a coordinator with remote engines, dp_rank may exceed the local
        # process list -- but we still hold a live control socket for it, and
        # the remote engine must receive SHUTDOWN so its worker node unblocks.
        # 【译】确定这个rank是否有本地进程和/或控制socket。在有远端engine的协调节点上，dp_rank可能超出本地进程列表的范围——
        #   但我们仍持有它的一个活的控制socket，而远端engine必须收到SHUTDOWN，它所在的工作节点才能解除阻塞。
        has_local_process = dp_rank < len(self.engine_core_processes)
        process = self.engine_core_processes[dp_rank] if has_local_process else None
        has_local_alive = process is not None and process.is_alive()

        has_control_socket = (
            dp_rank < len(self.control_sockets)
            and not self.control_sockets[dp_rank].closed
        )

        # Send SHUTDOWN if:
        #   - local rank with a live process (original behavior), OR
        #   - remote rank that has an open control socket (new: coordinator → worker).
        # 【译】在以下情况发送SHUTDOWN：
        #     - 本地rank且进程还活着（原有行为），或者
        #     - 远端rank且有打开的控制socket（新增：协调节点 → 工作节点）。
        should_send = has_local_alive or (not has_local_process and has_control_socket)
        if should_send:
            try:
                if has_control_socket:
                    self._send_control(
                        dp_rank, pickle.dumps((EngineCoreRequestType.SHUTDOWN, None))
                    )
                    logger.debug(f"{self.label}: Sent shutdown to DP rank {dp_rank}")
                else:
                    logger.warning(
                        f"{self.label}: no usable control socket for DP rank "
                        f"{dp_rank}; shutdown not delivered"
                    )
            except Exception as e:  # noqa: BLE001 - teardown must not raise
                # 【译】（上一行行尾注释）noqa: BLE001 - 拆除过程不能抛异常
                logger.debug(
                    f"{self.label}: Error sending shutdown to DP rank {dp_rank}: {e}"
                )

    # 阻塞取一批已结束的序列（list[Sequence]）；调用方：LLMEngine.step（离线generate的主循环）、rollout
    #   Queue.get()在队列空时阻塞，直到某个输出线程put进来
    #   取到异常对象就抛出（仓库里目前没有往outputs_queue放异常的地方，属于防御性代码）
    def get_output(self) -> list[Sequence]:
        seqs = self.outputs_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    # outputs_queue里是否还有没取走的结果；generate的循环条件之一：engine还活着，或者还有剩余结果
    def is_rest(self):
        return not self.outputs_queue.empty()

    # 是否还有EngineCore进程存活
    def is_alive(self):
        return any(
            proc is not None and proc.is_alive() for proc in self.engine_core_processes
        )

    # 是否有EngineCore进程已经退出（启动阶段用它判断启动失败）
    def finished_procs(self):
        return any(
            proc is not None and not proc.is_alive()
            for proc in self.engine_core_processes
        )


# 为一个(DP rank, PP stage)创建EngineCore的进程对象（不start）并分配三个ZMQ地址
#   输入：config（这个engine专属的Config副本）、dp_rank（全局）、local_dp_rank（本机）、addresses（多机时的TCP地址）
#   输出：(Process对象, {"input_address", "output_address", "control_address"}, local_dp_rank)
#   子进程入口：EngineCore.run_engine(config, input_address, output_address)；control_address经config传过去
#   调用方：CoreManager.__init__
def launch_engine_core(
    config: Config,
    dp_rank: int = 0,
    local_dp_rank: int | None = None,
    addresses: dict | None = None,
):
    if addresses is None:
        # 单机：每个socket一个新的 ipc://<临时目录>/<uuid> 地址（Unix域socket）
        input_address = get_open_zmq_ipc_path()
        output_address = get_open_zmq_ipc_path()
        control_address = get_open_zmq_ipc_path()
    else:
        # Multi-node: TCP endpoints derived from the shared port plan. The
        # engine connects; the coordinator binds.
        # 【译】多节点：由共享端口方案推出的TCP端点。engine connect；协调节点bind。
        input_address = addresses["input_address"]
        output_address = addresses["output_address"]
        control_address = addresses["control_address"]
    import torch

    # Imported here, not at module scope: EngineCore pulls the heavy
    # engine_core -> async_proc -> aiter chain. Spawning a worker is inherently a
    # GPU-side operation, so the cost belongs here and keeps CoreManager (routing
    # only) importable on a CPU-only runner.
    # 【译】在这里而不是模块顶层导入：EngineCore会拉进很重的 engine_core -> async_proc -> aiter 链。spawn一个worker
    #   本来就是GPU侧的操作，所以这个开销属于这里，同时让CoreManager（只做路由）可以在只有CPU的机器上导入。
    from atom.model_engine.engine_core import EngineCore

    # 同CoreManager.__init__：还没设置过启动方式时设为spawn
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method("spawn", force=False)

    if local_dp_rank is None:
        local_dp_rank = dp_rank
    config.parallel_config.data_parallel_rank = dp_rank
    config.parallel_config.data_parallel_rank_local = local_dp_rank
    # Rides on the config rather than run_engine's signature, which every
    # EngineCore subclass would otherwise have to thread through.
    # 【译】放在config上，而不是run_engine的签名里，否则每个EngineCore子类都得把它一路传下去。
    config.parallel_config.control_address = control_address

    # tp_world_size: the GPUs this DP rank really occupies.
    # 【译】tp_world_size：这个DP rank真正占用的GPU数。
    # 解释：例 -dp 2 -tp 4：local DP rank 1 -> 日志显示 GPUs 4 to 7（这里只按DP算；PP>1时实际编号见ModelRunner）
    logger.info(
        f"Creating EngineCore process: global DP rank {dp_rank} "
        f"(local {local_dp_rank}), GPUs "
        f"{local_dp_rank * config.tp_world_size} to "
        f"{(local_dp_rank + 1) * config.tp_world_size - 1}"
    )

    # multiprocessing.Process(target=f, kwargs=...)：start()时在新进程里执行 f(**kwargs)
    #   spawn方式下kwargs（包括整个Config）会被pickle后传给子进程
    #   EngineCore.run_engine是staticmethod，在子进程里由它决定构造EngineCore / DPEngineCoreProc / PPEngineCoreProc
    process = multiprocessing.Process(
        target=EngineCore.run_engine,
        name=f"EngineCore-DP{dp_rank}",
        kwargs={
            "config": config,
            "input_address": input_address,
            "output_address": output_address,
        },
    )

    return (
        process,
        {
            "input_address": input_address,
            "output_address": output_address,
            "control_address": control_address,
        },
        local_dp_rank,
    )


# 【分支】PD分离：同一组GPU上的prefill/decode分离（RapidServe，--enable-rapidserve），见第9批
#   与跨节点PD分离（--kv-transfer-config）不同，这里是同一台机器、同一组GPU上的两个EngineCore进程，通过CUDA IPC
#   共享权重和KV cache。LLMEngine在config.enable_rapidserve时构造它而不是CoreManager
#   engine下标：0 = PrefillEngineCore，1 = DecodeEngineCore
class DisaggCoreManager(CoreManager):
    """CoreManager for intra-GPU prefill/decode disaggregation.

    Spawns two separate EngineCore processes on the same GPU(s):
      - PrefillEngineCore: runs prefill forward passes, writes KV cache.
      - DecodeEngineCore: owns BlockManager and KV cache, runs decode.

    add_request() fans out every new sequence to BOTH processes.
    Only DecodeEngineCore produces finished sequences back to LLMEngine.

    The two processes coordinate via direct ZMQ PUSH/PULL sockets whose
    addresses are established here before spawning and passed through config.
    """

    # 【译】用于GPU内prefill/decode分离的CoreManager。
    #
    #   在同一组GPU上spawn两个独立的EngineCore进程：
    #     - PrefillEngineCore：运行prefill forward，写KV cache。
    #     - DecodeEngineCore：拥有BlockManager和KV cache，运行decode。
    #
    #   add_request() 把每个新序列扇出给两个进程。只有DecodeEngineCore会把结束的序列返回给LLMEngine。
    #
    #   两个进程通过直接的ZMQ PUSH/PULL socket协调，这些socket的地址在spawn之前就在这里确定，并通过config传过去。
    def __init__(self, config: Config):
        import copy

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        # Generate the inter-process ZMQ addresses before spawning.
        # 【译】在spawn之前生成进程间的ZMQ地址。
        # 【译】（行尾注释）decode → prefill（BlockAssignment）
        d2p_addr = get_open_zmq_ipc_path()  # decode → prefill (BlockAssignment)
        # 【译】（行尾注释）prefill → decode（PrefillDone）
        p2d_addr = get_open_zmq_ipc_path()  # prefill → decode (PrefillDone)
        # Bootstrap round 1: weight IPC handles (prefill → decode) + ACK (decode → prefill)
        # 【译】启动握手第1轮：权重IPC句柄（prefill → decode）+ ACK（decode → prefill）
        weight_ipc_addr = get_open_zmq_ipc_path()
        weight_ack_addr = get_open_zmq_ipc_path()
        # Bootstrap round 2: kvcache handle + num_blocks (prefill → decode)
        # 【译】启动握手第2轮：kvcache句柄 + num_blocks（prefill → decode）
        kvcache_ipc_addr = get_open_zmq_ipc_path()

        # Shared memory for dynamic CU partitioning: 4 bytes (float32).
        # DecodeScheduler writes the chosen CU fraction; PrefillScheduler reads it.
        # 0.0 means no mask (None).
        # Only created in constrained mode; unconstrained mode runs prefill
        # and decode on plain separate streams with no CU coordination.
        # 【译】用于动态CU划分的共享内存：4字节（float32）。DecodeScheduler写入选定的CU比例；PrefillScheduler读取它。
        #   0.0表示不加掩码（None）。只在受约束模式下创建；非受约束模式下prefill和decode跑在普通的独立stream上，不做CU协调。
        if config.disagg_constrained:
            cu_shm_name = f"atom_cu_split_{os.getpid()}"
            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=cu_shm_name, create=True, size=4
            )
            self._cu_shm.buf[:4] = b"\x00" * 4
        else:
            cu_shm_name = ""
            self._cu_shm = None

        # Build per-process configs.
        # 【译】为每个进程构建各自的config。
        prefill_config = copy.deepcopy(config)
        if config.disagg_prefill_max_num_seqs is not None:
            prefill_config.max_num_seqs = config.disagg_prefill_max_num_seqs
        prefill_config.enforce_eager = True
        prefill_config.disagg_d2p_addr = d2p_addr
        prefill_config.disagg_p2d_addr = p2d_addr
        prefill_config.disagg_weight_ipc_addr = weight_ipc_addr
        prefill_config.disagg_weight_ack_addr = weight_ack_addr
        prefill_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        prefill_config.disagg_cu_shm_name = cu_shm_name
        # Prefill gets its own server; decode honors the configured port.
        # 【译】prefill用自己的服务器；decode沿用配置的端口。
        # 解释：端口为0 -> 下面的_start_distributed_store(prefill_config)会为prefill另起一个由OS分配端口的TCPStore
        prefill_config.parallel_config.data_parallel_base_port = 0

        decode_config = copy.deepcopy(config)
        decode_config.disagg_d2p_addr = d2p_addr
        decode_config.disagg_p2d_addr = p2d_addr
        decode_config.disagg_weight_ipc_addr = weight_ipc_addr
        decode_config.disagg_weight_ack_addr = weight_ack_addr
        decode_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        decode_config.disagg_cu_shm_name = cu_shm_name
        # Decode allocates no GPU memory — kvcache and weights are imported from
        # prefill via CUDA IPC after prefill's READY signal.
        # 【译】decode不分配GPU显存——kvcache和权重在prefill发出READY之后经CUDA IPC从prefill导入。
        decode_config.disagg_is_decode = True

        if config.torch_profiler_dir:
            prefill_config.torch_profiler_dir = os.path.join(
                config.torch_profiler_dir, "prefill"
            )
            decode_config.torch_profiler_dir = os.path.join(
                config.torch_profiler_dir, "decode"
            )
            os.makedirs(prefill_config.torch_profiler_dir, exist_ok=True)
            os.makedirs(decode_config.torch_profiler_dir, exist_ok=True)

        # Addresses for the standard CoreManager input/output/control sockets.
        # 【译】标准的CoreManager输入/输出/控制socket的地址。
        prefill_input_addr = get_open_zmq_ipc_path()
        prefill_output_addr = get_open_zmq_ipc_path()
        decode_input_addr = get_open_zmq_ipc_path()
        decode_output_addr = get_open_zmq_ipc_path()
        prefill_config.parallel_config.control_address = get_open_zmq_ipc_path()
        decode_config.parallel_config.control_address = get_open_zmq_ipc_path()

        from atom.model_engine.engine_core import DecodeEngineCore, PrefillEngineCore

        prefill_proc = multiprocessing.Process(
            target=PrefillEngineCore.run_engine,
            name="PrefillEngineCore",
            kwargs={
                "config": prefill_config,
                "input_address": prefill_input_addr,
                "output_address": prefill_output_addr,
            },
        )
        decode_proc = multiprocessing.Process(
            target=DecodeEngineCore.run_engine,
            name="DecodeEngineCore",
            kwargs={
                "config": decode_config,
                "input_address": decode_input_addr,
                "output_address": decode_output_addr,
            },
        )

        # Set up the inherited state without running CoreManager.__init__,
        # which would spawn its own engines the base way. This manager fans out
        # through its own add_request() and never charges DP load, but the
        # inherited output thread still releases it on every finished sequence.
        # 【译】设置继承来的状态，但不运行 CoreManager.__init__——那会按基类的方式spawn它自己的engine。这个manager通过
        #   自己的 add_request() 扇出、从不记DP负载，但继承来的输出线程仍会在每个结束的序列上释放负载。
        self._init_shared_state(
            config,
            label="DisaggCoreManager",
            # 【译】（行尾注释）prefill + decode
            local_engine_count=2,  # prefill + decode
        )

        import weakref

        # 启动一个进程，并按基类的方式bind三个socket、完成握手（这里的PULL没传bind，但make_zmq_socket对PULL默认就是bind）
        def _connect_proc(proc, in_addr, out_addr, ctrl_addr, name):
            proc.start()
            self.engine_core_processes.append(proc)
            in_sock = make_zmq_socket(self.ctx, in_addr, zmq.ROUTER, bind=True)
            identity, _ = in_sock.recv_multipart()
            self.input_sockets.append(in_sock)
            self.engine_core_identities.append(identity)
            ctrl_sock = make_zmq_socket(self.ctx, ctrl_addr, zmq.ROUTER, bind=True)
            ctrl_identity, _ = ctrl_sock.recv_multipart()
            self.control_sockets.append(ctrl_sock)
            self.control_identities.append(ctrl_identity)
            out_sock = make_zmq_socket(self.ctx, out_addr, zmq.PULL)
            self.output_sockets.append(out_sock)
            self.shutdown_paths.append(get_open_zmq_inproc_path())
            logger.info(f"{self.label}: {name} process started and connected")

        try:
            self._start_distributed_store(decode_config)
            self._start_distributed_store(prefill_config)
            # Start both processes simultaneously.  Prefill binds the bootstrap
            # PUSH socket and blocks on send() until decode connects and calls
            # recv() — they rendezvous naturally without any sequential ordering.
            # 【译】同时启动两个进程。prefill bind启动用的PUSH socket，并阻塞在send()上，直到decode connect并调用recv()——
            #   它们自然地会合，不需要任何先后顺序。
            _connect_proc(
                prefill_proc,
                prefill_input_addr,
                prefill_output_addr,
                prefill_config.parallel_config.control_address,
                "prefill",
            )
            _connect_proc(
                decode_proc,
                decode_input_addr,
                decode_output_addr,
                decode_config.parallel_config.control_address,
                "decode",
            )
            self._wait_for_single_ready(idx=0)
            self._wait_for_single_ready(idx=1)
            logger.info(f"{self.label}: both EngineCores ready")

            # Start output thread for decode only (index 1).
            # Prefill has a separate output thread just for READY/error monitoring.
            # 【译】只为decode启动输出线程（下标1）。prefill有一个单独的输出线程，只用于READY/错误监控。
            # 解释：实际代码对两个下标都用同一个_create_output_thread创建输出线程；prefill那边不产生结束序列
            for idx, name in [(0, "prefill"), (1, "decode")]:
                t = self._create_output_thread(
                    idx, self.output_sockets[idx], self.shutdown_paths[idx]
                )
                t.start()
                self.output_threads.append(t)

            if self.finished_procs():
                raise RuntimeError("DisaggCoreManager: a process failed to start")

        except Exception:
            self.close()
            raise

        self._finalizer = weakref.finalize(self, self.close)
        self.async_output_queue = None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    # 阻塞读output_sockets[idx]直到收到READY（初始化期间收到SHUTDOWN则报错，其他消息忽略）
    def _wait_for_single_ready(self, idx: int):
        """Block until output_sockets[idx] sends a READY signal."""
        # 【译】阻塞直到 output_sockets[idx] 发来READY信号。
        sock = self.output_sockets[idx]
        while True:
            obj = sock.recv(copy=False)
            request_type, data = pickle.loads(obj)
            if request_type == EngineCoreRequestType.READY:
                self._record_ready_payload(data)
                return
            if request_type == EngineCoreRequestType.SHUTDOWN:
                raise RuntimeError(
                    f"{self.label}: process {idx} sent SHUTDOWN during initialization"
                )

    # 扇出：每个新序列同时发给decode（下标1，负责分配block并产出结果）和prefill（下标0，只算prefill）
    def add_request(self, seqs: list[Sequence]):
        """Fan-out: send every new sequence to BOTH prefill and decode."""
        # 【译】扇出：把每个新序列同时发给prefill和decode。
        logger.debug(f"{self.label}: fan-out {len(seqs)} seqs to prefill and decode")
        # Register stream callbacks before sending (decode will produce output).
        # 【译】在发送之前注册流式回调（输出由decode产生）。
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None

        # Send decode payload as-is. Decode admits and allocates the blocks
        # (`DecodeScheduler.allocate_waiting`), so it alone gets the chains.
        # 【译】decode的payload原样发送。decode负责接纳并分配block（DecodeScheduler.allocate_waiting），所以只有它拿到链。
        decode_payload = _add_frame(seqs)
        self._send_request(1, decode_payload)

        # For prefill: limit each sequence to 1 output token.  Prefill discards
        # all sampled tokens (postprocess is a no-op), but setting max_tokens=1
        # ensures the forward pass terminates after a single generate step and
        # that num_scheduled_tokens correctly reflects only the prompt tokens.
        # 【译】对prefill：把每个序列限制为1个输出token。prefill丢弃所有采样出的token（postprocess是空操作），但设置
        #   max_tokens=1 可以保证forward在一次生成step后结束，并且 num_scheduled_tokens 只正确反映prompt token。
        import copy as _copy

        prefill_seqs = []
        for seq in seqs:
            # copy.copy：浅拷贝，只改副本的max_tokens，不影响发给decode的原对象（它已经pickle发出了）
            ps = _copy.copy(seq)
            ps.max_tokens = 1
            prefill_seqs.append(ps)
        prefill_payload = pickle.dumps((EngineCoreRequestType.ADD, prefill_seqs))
        self._send_request(0, prefill_payload)

    # 两个engine是同一副本的prefill/decode两半，不是DP rank，所以不支持集合RPC
    def _collective_rpc_unsupported(self) -> str | None:
        return "its two engines are the prefill and decode halves of one replica"

    # 先走基类的关闭流程，再释放CU划分用的共享内存
    def close(self):
        super().close()
        # Clean up dynamic CU partitioning shared memory (if created).
        # 【译】清理动态CU划分的共享内存（如果创建过）。
        if getattr(self, "_cu_shm", None) is not None:
            try:
                self._cu_shm.close()
                self._cu_shm.unlink()
            except Exception:
                pass
