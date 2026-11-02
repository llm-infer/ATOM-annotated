# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Asynchronous I/O process management for model runner workers.

This module provides:

- :class:`AsyncIOProc`: A single worker process that runs a model runner
  and communicates via ZMQ sockets and shared-memory broadcast queues.
- :class:`AsyncIOProcManager`: Manages multiple ``AsyncIOProc`` workers,
  routes function calls via broadcast, and aggregates KV transfer outputs
  from all workers.
"""

# 【译】模型执行器（model runner）worker的异步I/O进程管理。
#
#   本模块提供：
#
#   - AsyncIOProc：单个worker进程，运行一个model runner，通过ZMQ socket和共享内存广播队列通信。
#   - AsyncIOProcManager：管理多个 AsyncIOProc worker，通过广播路由函数调用，并汇总所有worker的KV传输输出。

import logging
import multiprocessing
import pickle
import queue
import threading
import time
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Thread
from typing import ClassVar

# zmq（pyzmq）：ZeroMQ的Python绑定。本文件用到的socket类型：
#   PUSH / PULL   单向管道：PUSH端只发，PULL端只收；多个PUSH可以连到同一个PULL
#   DEALER        异步的请求端，可与ROUTER配对（这里只在遗留的输入线程里出现）
#   make_zmq_socket的默认规则：PUSH做connect，PULL/DEALER/ROUTER做bind（见atom/utils/__init__.py）
import zmq
import zmq.asyncio

# MessageQueue：aiter里的共享内存广播队列（移植自vLLM的shm_broadcast，源码不在本仓库）
#   一个写者、多个读者；消息先pickle，再写进共享内存环形缓冲区的一块，每个读者各自读到同一条消息
from aiter.dist.shm_broadcast import MessageQueue

from atom.kv_transfer.disaggregation import KVConnectorOutput, KVOutputAggregator
from atom.model_engine.block_table_codec import (
    FORWARD_RPC,
    BlockTableDeltaDecoder,
    BlockTableDeltaEncoder,
)
from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_METHODS,
    UTILITY_MANAGED_RUNNER_METHODS,
    RpcPayload,
    RpcResult,
    checked_timeout,
    is_plain_rpc_value,
)
from atom.utils import (
    envs,
    get_mp_context,
    get_open_zmq_ipc_path,
    init_exit_handler,
    make_zmq_socket,
    resolve_obj_by_qualname,
    set_process_title,
    shutdown_all_processes,
    worker_process_name,
)
from atom.utils.gc_utils import maybe_attach_gc_debug_callback, tune_gc
from atom.utils.numa_utils import numa_bind_to_node

logger = logging.getLogger("atom")


# =====================================================================================
# 本文件：EngineCore进程与每张GPU上的ModelRunner之间的"RPC层"
# =====================================================================================
#   AsyncIOProc         每个TP rank一个子进程（ModelRunner进程）：构造ModelRunner，
#                       然后在busy_loop里一直"取指令 → 调用runner的同名方法 → 把结果送回"
#   AsyncIOProcManager  运行在EngineCore进程里：spawn这些子进程、广播指令（call_func）、收集结果
#   调用方：EngineCore.__init__ 创建 AsyncIOProcManager(finalizer, tp_world_size x PCP大小, runner类名, config)
#     （PP的每个stage是一个独立的EngineCore，各自有自己的manager，所以进程数不乘PP）
#
# 进程与通道（TP=2为例）：
#
#   EngineCore进程（AsyncIOProcManager）                        ModelRunner进程 × TP（每个都是一个AsyncIOProc）
#
#   call_func / collective_rpc                                  ┌──────── AsyncIOProc(rank 0) ────────┐
#     │ rpc_broadcast_mq.enqueue((方法名, *参数))                │ busy_loop（主线程）                  │
#     ▼                                                         │   get_func()：从共享内存取一条指令     │
#   ┌───────────────────────────────┐   每个rank读到同一条     │   runner.<方法名>(*参数)              │
#   │ MessageQueue                  │ ═══════════════════════> │   按调用类型把结果放进不同的queue      │
#   │ （共享内存环形缓冲区，一写多读）│ ═════════╗               │ 发送线程：queue → pickle → PUSH       │
#   └───────────────────────────────┘          ║               └───┬────────────┬──────────────┬───────┘
#                                              ║                   │主通道       │KV通道         │集合RPC通道
#   输出线程       PULL(bind) ◄─────────────────────────────────────┘（仅rank 0）│              │
#     → outputs_queue（call_func(wait_out=True)从这里取）          ║             │              │
#   KV输出线程[r]  PULL(bind) ◄──────────────────────────────────────────────────┘              │
#     → kv_outputs_queues[r]（call_func_with_aggregation汇总）     ║                            │
#   RPC输出线程[r] PULL(bind) ◄─────────────────────────────────────────────────────────────────┘
#     → rpc_outputs_queues[r]（collective_rpc逐rank收齐）          ║
#   监控线程：等任何一个子进程退出 → exit()                         ╚══> AsyncIOProc(rank 1)：结构相同但没有主通道，
#                                                                       普通调用的返回值直接丢弃（KV/RPC通道照常）
#   各rank之间的张量通信（每层的all-reduce等）发生在模型forward内部，走RCCL，不经过这里
#
# 一次forward调用的时序（TP=2）：
#
#   EngineCore主线程                     MessageQueue        TP0（AsyncIOProc）                 TP1（AsyncIOProc）
#     │ call_func("forward", batch, wait_out=True)               │ 阻塞在dequeue                    │ 阻塞在dequeue
#     │ encode_rpc：block table压成增量
#     │ enqueue(("forward", 压缩后的batch)) ──> [一块] ──dequeue──>│                                  │
#     │ outputs_queue.get() 阻塞                       └──dequeue──┼─────────────────────────────────>│
#     │                                                          │ decode_rpc：还原block table        │ decode_rpc
#     │                                                          │ ModelRunner.forward(batch)        │ ModelRunner.forward(batch)
#     │                                                          │   每层all-reduce <═══ RCCL ═══>    │   每层all-reduce
#     │                                                          │ out = ScheduledBatchOutput        │ out = ScheduledBatchOutput
#     │                                                          │ io_queues[1].put(out)             │ 没有主通道：丢弃
#     │                                                          │ 发送线程：pickle → PUSH            │ 回到dequeue
#     │ <── manager输出线程：PULL收到 → pickle.loads → outputs_queue.put
#     ▼ 返回fwd_out，交给scheduler.postprocess
#
# 对比nano-vllm（见nano-vllm-annotated/nanovllm/engine/model_runner.py开头的时序图）：
#   nano-vllm：rank 0的ModelRunner就在主进程里（和调度器同一个进程），由它把[方法名, 参数]pickle进一块1MB的
#     SharedMemory，再对每个子进程event.set()；rank 0自己直接调用方法拿到返回值，其他rank的返回值丢弃
#   ATOM：调度器所在的EngineCore进程不碰GPU，所有TP rank（包括rank 0）都是独立的子进程，因此：
#     - 指令广播：思路相同（共享内存 + pickle），换成了aiter的MessageQueue：缓冲区分成多块、可以有多条指令在途
#       （call_func不等结果时会连续发几条），读者轮询共享内存里的标志位而不是等Event（按vLLM原版的实现）
#     - 结果回传：rank 0也在别的进程里，结果只能跨进程送回，用ZMQ PUSH/PULL + pickle（只有rank 0回传）
#     - 多了两类逐rank通道：KV传输状态聚合（PD分离/KV offload，见第9批）和集合RPC（要求每个rank都回复）
#     - 多了一个监控线程：任何rank意外退出就关闭整个管理器，避免EngineCore永远卡在等结果上
#     - forward指令里最大的block table只发增量（block_table_codec.py），nano-vllm每步重发整张表


# 【分支】KV聚合的"在途记录"，只在PD分离/KV offload时使用，见第9批
#   call_func_with_aggregation一次只允许一个聚合在途，这里存它已经收到的各rank结果
@dataclass
class _PendingKvAggregation:
    """Worker outputs retained for one outstanding KV aggregation RPC.

    Worker replies drain once and carry no generation identifier, so at most
    one aggregation may be outstanding. ``worker_outputs[rank]`` stays
    ``None`` until that rank's queue is consumed.
    """

    # 【译】为一次未完成的KV聚合RPC保留的worker输出。
    #
    #   worker的回复只能取走一次，而且不带任何"第几轮"的标识，所以最多只能有一个聚合在途。
    #   在该rank的queue被消费之前，worker_outputs[rank] 一直是 None。

    # 每个rank一个槽位，收到前为None
    worker_outputs: list[KVConnectorOutput | None]
    # 发起时间（monotonic），用于超时告警
    started_at: float
    # 是否已经打过"还在等"的告警（只打一次）
    wait_warning_logged: bool = False

    # 还没回复的rank列表
    def missing_worker_ranks(self) -> list[int]:
        return [
            rank for rank, output in enumerate(self.worker_outputs) if output is None
        ]


# =====================================================================================
# AsyncIOProc：一个ModelRunner进程的全部内容
# =====================================================================================
# 用法很特别：AsyncIOProcManager用 ctx.Process(target=AsyncIOProc, args=...) 启动子进程，
#   子进程执行的就是"构造一个AsyncIOProc"——而__init__的最后一行是self.busy_loop()，
#   直到收到exit才返回。所以这个构造函数的执行时间就是整个进程的生命周期（同nano-vllm的target=ModelRunner）
#
# 构造过程（__init__）：
#   1. 进程级准备：父进程死亡时自动退出、进程标题、GC阈值、NUMA绑核
#   2. 连接共享内存广播队列（读者端），注册退出处理
#   3. 启动发送线程：主通道（仅rank 0）、KV通道、集合RPC通道
#   4. 构造ModelRunner(rank, config)：初始化分布式、加载权重、预热……（第5批）
#   5. 进入busy_loop
#
# 成员变量：
#   label               日志标签，如"AsyncIOProc(ModelRunner0/2)"
#   rank                本进程在本EngineCore内的worker编号（0 ~ proc_num-1，通常就是TP rank）
#   io_addrs            (输入地址, 主输出地址)。manager传入的输入地址恒为None；主输出地址只有rank 0有
#   io_queues           (输入queue, 主输出queue)，各配一个I/O线程；输入那一个实际上不会被用到
#   io_threads          本进程启动的所有I/O线程
#   kv_output_addr / kv_queue      KV聚合结果的专用通道（每个rank都有）
#   rpc_output_addr / rpc_queue    集合RPC回复的专用通道（每个rank都有）
#   rpc_broadcast_mq    共享内存广播队列的读者端，所有指令都从这里来
#   all_ranks_barrier   所有rank共享的multiprocessing.Barrier，集合RPC要求时在调用后集合
#   _block_table_decoder  把forward指令里的增量block table还原成完整的表
#   runners             只有一个元素的列表：[ModelRunner实例]（保留列表形式，busy_loop按列表遍历）
#   still_running / _finalizer  由init_exit_handler设置，保证exit()只执行一次
class AsyncIOProc:
    """A single worker process that runs a model runner with ZMQ I/O.

    Each worker receives function calls via shared-memory broadcast,
    executes them on the runner, and sends results back via ZMQ.
    KV aggregation outputs are sent on a dedicated channel to avoid
    mixing with regular forward outputs.

    Args:
        label: Human-readable label for logging.
        io_addrs: ``(input_addr, output_addr)`` ZMQ endpoints.
        input_shm_handle: Shared memory handle for the broadcast queue.
        runner_qualname: Fully qualified class name of the runner to instantiate.
        rank: TP rank of this worker.
        kv_output_addr: Optional ZMQ endpoint for KV aggregation output.
        rpc_output_addr: Optional ZMQ endpoint for generic collective-RPC
            replies. Every rank owns one, unlike the primary channel which
            only rank 0 has, so the caller can prove all ranks answered.
    """

    # 【译】一个运行model runner、带ZMQ I/O的worker进程。
    #
    #   每个worker通过共享内存广播接收函数调用，在runner上执行，再通过ZMQ把结果送回。
    #   KV聚合的输出走一条专用通道，避免与常规的forward输出混在一起。
    #
    #   参数：
    #     label：用于日志的可读标签。
    #     io_addrs：(input_addr, output_addr) 两个ZMQ端点。
    #     input_shm_handle：广播队列的共享内存句柄。
    #     runner_qualname：要实例化的runner类的完整限定名。
    #     rank：该worker的TP rank。
    #     kv_output_addr：可选，KV聚合输出的ZMQ端点。
    #     rpc_output_addr：可选，通用集合RPC回复的ZMQ端点。与只有rank 0才有的主通道不同，每个rank都有一个，
    #       所以调用方能证明所有rank都回复了。
    # 解释：参数表里没有列出的 all_ranks_barrier 是第7个位置参数；*args 原样转给runner的构造函数
    #   （实际只有一个：config），**kwargs 也转给runner

    # Function names whose output goes to the KV channel instead of primary
    # 【译】这些函数名的输出走KV通道，而不是主通道
    # frozenset：不可变集合，作为类常量；async_proc_aggregation是ModelRunner上"收集KV传输完成状态"的方法
    _KV_FUNC_NAMES = frozenset(["async_proc_aggregation"])

    def __init__(
        self,
        label: str,
        io_addrs: tuple[str, str],
        input_shm_handle: int,
        runner_qualname: str,
        rank: int,
        kv_output_addr: str | None = None,
        all_ranks_barrier=None,
        *args,
        rpc_output_addr: str | None = None,
        **kwargs,
    ):
        # Bind this worker's lifetime to its parent EngineCore: if the parent
        # exits for any reason, have the kernel reap this process immediately
        # instead of leaving it orphaned. A ModelRunner worker holds a large GPU
        # allocation and the custom all-reduce IPC resources; an orphan blocks
        # forever in busy_loop() on the shm dequeue while keeping those pinned,
        # causing the stale-IPC all-reduce crash on the next restart. Must be
        # armed here, before any GPU / IPC state is created.
        # 【译】把这个worker的生命周期绑定到它的父进程EngineCore上：父进程无论因为什么原因退出，都让内核立刻回收本进程，
        #   而不是把它留成孤儿。ModelRunner worker持有一大块GPU显存和custom all-reduce的IPC资源；孤儿进程会在
        #   busy_loop() 里永远阻塞在共享内存的dequeue上，同时一直占着这些资源，导致下次重启时出现"过期IPC"的all-reduce崩溃。
        #   必须在这里、在创建任何GPU / IPC状态之前装好。
        # 解释：内部用prctl(PR_SET_PDEATHSIG)，父进程一退出内核就给本进程发SIGKILL（见atom/utils/__init__.py）
        from atom.utils import enable_orphan_reaping

        enable_orphan_reaping()

        # Named once, before anything logs: `ps`, the GC lines here and the
        # freeze line the runner emits over RPC all have to be the same string,
        # or a dp deployment's workers are indistinguishable from each other.
        # 【译】在任何日志输出之前只命名一次：`ps`、这里的GC日志行、以及runner通过RPC打出的freeze日志行都必须是同一个字符串，
        #   否则dp部署里的各个worker彼此无法区分。
        # 解释：名字形如"TP0"，DP>1时为"DP1TP0"；args[0]就是config
        name = worker_process_name(args[0] if args else None, rank)
        set_process_title(name)

        # Second-order next to the EngineCore's call but not zero: without it
        # a freeze still lands at the wave boundary, where the prefill burst
        # churns enough objects to trigger gen-2 here. Runs before the model
        # is built so the thresholds cover the startup heap. Freezing that heap
        # happens later, via the EngineCore's `freeze_gc_heap` RPC.
        # 【译】与EngineCore那边的调用相比是次要的，但不是零：没有它的话，冻结仍会发生在一波请求的边界上，那时prefill的突发
        #   会产生足够多的对象，在这里触发第2代回收。它在模型构建之前运行，这样阈值能覆盖启动阶段的堆。冻结那个堆则发生在
        #   之后，通过EngineCore的 freeze_gc_heap RPC 完成。
        # 解释：tune_gc按ATOM_GC_THRESHOLD调高本进程的GC阈值（未设置时什么也不做），见gc_utils.py
        tune_gc()
        maybe_attach_gc_debug_callback(name)

        # NUMA-local CPU/memory pinning (see atom.utils.numa_utils).
        # Auto-detects the GPU's local node by default; gated by
        # ATOM_NUMA_BIND. Must run before any large allocation / native
        # (mooncake) thread spawn so the mask is inherited by child threads and
        # first-touch lands memory locally. The node-local GPU index is
        # dp_local_rank*tp_size+tp_rank (engine_core_mgr GPU assignment).
        # Best-effort: any failure here is logged and skipped rather than
        # taking the worker down, hence the blanket except.
        # 【译】按NUMA就近绑定CPU/内存（见 atom.utils.numa_utils）。默认自动探测GPU所在的本地节点；受ATOM_NUMA_BIND控制。
        #   必须在任何大块分配 / 原生（mooncake）线程创建之前运行，这样子线程会继承CPU掩码，首次访问（first-touch）
        #   的内存也落在本地节点。节点内的GPU编号是 dp_local_rank*tp_size+tp_rank（engine_core_mgr中的GPU分配方式）。
        #   尽力而为：这里的任何失败都只记日志并跳过，而不会让worker挂掉，所以用了捕获一切的except。
        try:
            cfg = args[0]
            # Node-local: this indexes a GPU on THIS machine. The global rank
            # would run off the end of a later node's device list.
            # 【译】本节点内的编号：这里索引的是"这台机器上"的GPU。用全局rank的话，在后面的节点上会越过设备列表的末尾。
            dp_local_rank = cfg.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = cfg.parallel_config.data_parallel_rank
            # 例：DP=2（单机）、TP=4时，DP rank 1的rank 2 → 本机第 1*4+2 = 6 号GPU
            gpu = dp_local_rank * cfg.tp_world_size + rank
            numa_bind_to_node(gpu, label)
        # （行尾的noqa说明：绑定只是一项优化，所以故意捕获所有异常）
        except Exception as e:  # noqa: BLE001 - binding is an optimization
            # NUMA binding only affects locality, never correctness, so any
            # failure (missing libnuma, restricted cpuset, odd topology) must
            # degrade to an unbound worker rather than kill it.
            # 【译】NUMA绑定只影响访存局部性，从不影响正确性，所以任何失败（缺少libnuma、受限的cpuset、奇怪的拓扑）
            #   都必须降级为一个不绑定的worker，而不是杀掉它。
            logger.warning(f"AsyncIOProc({label}): NUMA bind skipped: {e}")
        # ---------- 成员变量（含义见类上方的说明） ----------
        self.label = f"AsyncIOProc({label})"
        self.rank = rank
        self.io_addrs = io_addrs
        # queue.Queue：线程安全的FIFO队列，put_nowait不阻塞，get()在为空时阻塞；用来在主线程和I/O线程之间传对象
        self.io_queues = queue.Queue(), queue.Queue()
        self.io_threads: list[threading.Thread] = []

        # KV aggregation output channel
        # 【译】KV聚合输出通道
        self.kv_output_addr = kv_output_addr
        self.kv_queue: queue.Queue | None = None

        # Generic collective-RPC reply channel. Every rank has one, so a caller
        # can tell "all ranks finished" from "rank 0 finished".
        # 【译】通用集合RPC回复通道。每个rank都有一个，所以调用方能区分"所有rank都完成了"和"rank 0完成了"。
        self.rpc_output_addr = rpc_output_addr
        self.rpc_queue: queue.Queue | None = None

        # 读者端：用manager导出的句柄连接到同一块共享内存，rank决定自己在读者中的编号
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, rank)
        # atexit.register(f)：解释器正常退出时调用f，作为exit()没走完时的兜底
        import atexit

        atexit.register(self._cleanup_shared_memory)
        # 装SIGTERM处理器、忽略SIGINT，并用weakref.finalize保证self.exit()最多执行一次（见atom/utils/__init__.py）
        init_exit_handler(self)

        # Start I/O threads for primary input/output
        # 【译】为主输入/输出启动I/O线程
        # 输入线程（recv_input_from_socket）和主输出线程（send_output_to_socket）各配一个queue；
        #   地址为None就不启动。manager传入的输入地址恒为None，所以实际只有rank 0会启动一个主输出线程
        # threading.Thread(target, args, daemon=True)：daemon线程不会阻止解释器退出
        for addr, q, func in zip(
            self.io_addrs,
            self.io_queues,
            [self.recv_input_from_socket, self.send_output_to_socket],
        ):
            if addr is None:
                continue
            t = threading.Thread(target=func, args=(addr, q), daemon=True)
            t.start()
            self.io_threads.append(t)

        # Dedicated KV aggregation output thread
        # 【译】专用的KV聚合输出线程
        if self.kv_output_addr is not None:
            self.kv_queue = queue.Queue()
            t = threading.Thread(
                target=self.send_output_to_socket,
                args=(self.kv_output_addr, self.kv_queue),
                daemon=True,
            )
            t.start()
            self.io_threads.append(t)

        # Dedicated collective-RPC reply thread
        # 【译】专用的集合RPC回复线程
        if self.rpc_output_addr is not None:
            self.rpc_queue = queue.Queue()
            t = threading.Thread(
                target=self.send_rpc_output_to_socket,
                args=(self.rpc_output_addr, self.rpc_queue),
                daemon=True,
            )
            t.start()
            self.io_threads.append(t)

        # 集合RPC带barrier=True时，每个rank执行完后在这里集合（见busy_loop）
        self.all_ranks_barrier = all_ranks_barrier
        self._block_table_decoder = BlockTableDeltaDecoder()

        # 按"模块.类名"字符串找到runner类（通常是atom.model_engine.model_runner.ModelRunner，
        #   RapidServe时是RapidServeModelRunner），由config.runner_qualname决定
        runner_class = resolve_obj_by_qualname(runner_qualname)
        self.runners: list[object] = []
        # 构造ModelRunner(rank, config)：初始化分布式、加载模型、预热等都在这里完成，耗时最长（第5批）
        self.runners = [runner_class(rank, *args, **kwargs)]
        # 进入主循环，直到收到"exit"才返回；返回后__init__结束，子进程随之退出
        self.busy_loop()

    # 关闭本worker：通知runner退出、关闭共享内存读者句柄、等I/O线程
    # 调用方：weakref.finalize（SIGTERM、解释器退出时）；正常关闭时是manager广播"exit"，busy_loop把它当普通方法
    #   调用在runner上，然后跳出循环，进程退出时finalizer再调用这里（still_running保证只执行一次）
    def exit(self):
        if not getattr(self, "still_running", True):
            return
        self.still_running = False
        logger.debug(f"{self.label}: Shutting down runner...")
        # runner.exit()：销毁分布式环境、释放CUDA Graph和显存（第5批）
        for el in self.runners:
            el.exit()
        # Close shared memory reader handle to prevent resource_tracker leak
        # 【译】关闭共享内存的读者句柄，防止resource_tracker泄漏
        self._cleanup_shared_memory()
        for t in self.io_threads:
            t.join(timeout=0.5)

    def _cleanup_shared_memory(self):
        """Close shared memory handles owned by this process."""
        # 【译】关闭本进程持有的共享内存句柄。
        # 读者端只close不unlink：共享内存段由manager（创建者）负责unlink
        if hasattr(self, "rpc_broadcast_mq"):
            mq = self.rpc_broadcast_mq
            if hasattr(mq, "buffer") and hasattr(mq.buffer, "shared_memory"):
                try:
                    mq.buffer.shared_memory.close()
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    # Already-closed handles and a racing peer's unlink both
                    # land here; neither is actionable during shutdown, but a
                    # silent pass hides genuine leaks from anyone debugging one.
                    # 【译】已经关闭的句柄和对端抢先的unlink都会走到这里；关闭过程中这两种情况都不需要处理，
                    #   但静默pass会把真正的泄漏藏起来，让排查的人看不到。
                    logger.debug(f"{self.label}: shm close failed: {e}")

    # 【遗留】输入线程：用DEALER连到某个ROUTER，收到的对象放进input_queue
    #   当前代码里manager传入的输入地址恒为None，这个线程不会启动；所有指令都走共享内存广播队列
    def recv_input_from_socket(self, addr: str, input_queue: queue.Queue):
        # ExitStack：把多个上下文管理器叠在一起，with结束时按相反顺序关闭（这里是socket和zmq.Context）
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # 先发一个空帧，让ROUTER端记住这个DEALER的identity
            socket.send(b"")
            poller.register(socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")

            # poll的超时是1秒：即使没有消息，也能定期检查still_running，从而能退出
            while getattr(self, "still_running", True):
                for socket, _ in poller.poll(timeout=1000):
                    # copy=False：零拷贝接收，得到zmq.Frame，pickle.loads可以直接读它的缓冲区
                    serialized_obj = socket.recv(copy=False)
                    input_obj = pickle.loads(serialized_obj)
                    input_queue.put_nowait(input_obj)

    # 主输出 / KV输出的发送线程：从queue里取结果 → pickle → 经PUSH socket发给manager
    # 输入：addr（manager那边PULL绑定的地址）、output_queue（busy_loop往里放结果）
    def send_output_to_socket(self, addr: str, output_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            # PUSH按make_zmq_socket的默认规则做connect（PULL端在manager里bind）；
            #   linger=4000：关闭socket时，最多再等4秒把还没发出去的消息发完
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            # 永久循环：daemon线程，进程退出时随之结束；queue.get()在没有结果时阻塞
            while True:
                result = output_queue.get()
                serialized_obj = pickle.dumps(result)
                socket.send(serialized_obj)

    def send_rpc_output_to_socket(self, addr: str, output_queue: queue.Queue):
        """Send request id separately, so a corrupt body stays attributable."""
        # 【译】把request id单独发送，这样即使消息体损坏，也还能知道它属于哪个请求。
        # 集合RPC回复的发送线程：每条消息是两帧 [request_id, pickle后的RpcResult]
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: RPC output socket connected")
            while True:
                result = output_queue.get()
                # send_multipart：把多帧作为一条消息原子地发出，接收端用recv_multipart一次收齐
                socket.send_multipart(self._serialize_rpc_result(result))

    @staticmethod
    def _serialize_rpc_result(result: RpcResult) -> list[bytes]:
        """Serialize once; turn an unpicklable value into a picklable failure."""
        # 【译】只序列化一次；把无法pickle的值变成一个可以pickle的失败结果。
        try:
            body = pickle.dumps(result)
        # （行尾的noqa说明：失败通过同一条通道报告给调用方，所以故意捕获所有异常）
        except Exception as exc:  # noqa: BLE001 - report over the same channel
            result = RpcResult(
                result.request_id,
                result.tp_rank,
                error=f"unpicklable RPC result: {type(exc).__name__}: {exc}",
            )
            body = pickle.dumps(result)
        return [result.request_id.encode("utf-8"), body]

    # ---------- 主循环：取指令 → 分派 → 送回结果 ----------
    # 每条指令的格式是 (方法名, *参数)，有两种：
    #   普通调用    call_func发出：参数原样传给runner.<方法名>(*参数)；只有rank 0的非None返回值送回主通道
    #   集合RPC     collective_rpc / utility_rpc发出：参数只有一个RpcPayload（带kwargs、request_id、是否要barrier），
    #               每个rank都把RpcResult送回自己的RPC通道（见collective_rpc.py）
    # 例：call_func("forward", batch, wait_out=True)
    #   → 每个rank：get_func()得到("forward", [batch]) → runner.forward(batch) → ScheduledBatchOutput
    #   → rank 0放进io_queues[1]由发送线程发回；rank 1没有主通道，结果丢弃
    # 注意：普通调用的返回值为None时不回送任何东西，所以用wait_out=True调用一个返回None的方法会让调用方永远等下去
    #   （ModelRunner.freeze_gc_heap特意返回一个计数就是为此）
    def busy_loop(self):
        """Main event loop: dequeue RPCs and dispatch to runners."""
        # 【译】主事件循环：从队列取出RPC并分派给runner。
        while True:
            # 阻塞直到有新指令（共享内存广播队列的dequeue）
            func_name, args = self.get_func()
            # 判断是不是集合RPC：恰好一个参数且它是RpcPayload
            payload = (
                args[0] if len(args) == 1 and isinstance(args[0], RpcPayload) else None
            )
            if payload is None:
                # 普通调用：位置参数原样传递，没有kwargs，不做barrier
                call_args, call_kwargs = args, {}
                need_barrier = False
            else:
                # 集合RPC：从RpcPayload里取出位置参数、kwargs和是否需要barrier
                call_args = payload.args
                call_kwargs = payload.call_kwargs()
                need_barrier = payload.barrier

            for runner in self.runners:
                # 集合RPC：_run_generic_rpc把任何结果（包括异常、找不到方法）都包装成RpcResult，保证每个rank都会回复
                if payload is not None:
                    out = self._run_generic_rpc(
                        runner, func_name, call_args, call_kwargs, payload
                    )
                else:
                    # 普通调用：runner上没有这个方法就静默跳过（不回复）
                    func = getattr(runner, func_name, None)
                    if func is None:
                        continue
                    # 真正执行，例如runner.forward(batch)
                    out = func(*call_args)
                # barrier：例如权重更新时，每个rank都要读同一块共享缓冲区，必须等所有rank都读完才能各自继续
                if need_barrier and self.all_ranks_barrier is not None:
                    try:
                        self.all_ranks_barrier.wait()
                    # BrokenBarrierError：barrier被abort()（manager发现有rank死了）或等待出错时抛出
                    except threading.BrokenBarrierError:
                        if payload is None:
                            raise
                        # Broken by the manager once a rank is found dead: that
                        # rank can never arrive, and this one would otherwise
                        # wait for it forever instead of answering.
                        # 【译】被manager在发现某个rank已死后打破：那个rank永远到不了，而本rank否则会一直等它，而不是回复。
                        out = self._failure(
                            payload,
                            f"{func_name!r} ran, but the barrier broke "
                            f"before every TP rank reached it",
                        )
                # ---------- 把结果送回 ----------
                if payload is not None:
                    # Generic replies go to this rank's own channel. Routing
                    # them to the primary would drop every rank but 0, which is
                    # the whole limitation the channel exists to remove.
                    # 【译】通用回复走本rank自己的通道。如果送到主通道，除rank 0以外的每个rank的回复都会被丢弃，
                    #   而这正是这条通道存在所要消除的限制。
                    if self.rpc_queue is not None:
                        self.rpc_queue.put_nowait(out)
                    elif self.io_addrs[1] is not None:
                        # No dedicated channel configured (older manager, or a
                        # test): fall back rather than silently discard.
                        # 【译】没有配置专用通道（旧的manager，或者测试）：退回主通道，而不是悄悄丢弃。
                        self.io_queues[1].put_nowait(out)
                # 普通调用且有返回值：
                elif out is not None:
                    # 非KV方法的结果走主通道（只有rank 0的io_addrs[1]不为None，所以其他rank的结果在这里被丢弃）
                    if (
                        self.io_addrs[1] is not None
                        and func_name not in self._KV_FUNC_NAMES
                    ):
                        self.io_queues[1].put_nowait(out)
                    # KV方法（async_proc_aggregation）的结果走KV通道，每个rank都发
                    if self.kv_queue is not None and func_name in self._KV_FUNC_NAMES:
                        self.kv_queue.put_nowait(out)
            # 收到exit：上面已经调用过runner.exit()，跳出循环，__init__随之结束
            if func_name == "exit":
                break
        logger.debug(f"{self.label}: exit busy_loop...")

    # 【分支】集合RPC在worker端的执行，见第10批（RL rollout通过它调用worker方法）
    # 输入：runner、方法名、位置参数、kwargs、RpcPayload；输出：本rank的RpcResult（成功带value，失败带error）
    def _run_generic_rpc(
        self,
        runner: object,
        func_name: str,
        call_args: tuple,
        call_kwargs: dict,
        payload: RpcPayload,
    ) -> RpcResult:
        """Invoke one generic RPC, converting every outcome into a reply.

        Never raises and never returns ``None``: a missing or malformed method
        name, a raising target and a return that is not plain data become an
        ``RpcResult`` carrying ``error``; the RPC sender still converts an
        unpicklable value before it writes the socket. Anything else would
        leave the caller blocked in an untimed queue get, which is how a typo
        in a method name currently costs five minutes and reports a timeout
        instead of the typo.
        """
        # 【译】执行一次通用RPC，把每一种结果都转换成一个回复。
        #
        #   从不抛异常，也从不返回 None：缺失或格式错误的方法名、抛异常的目标方法、以及返回值不是纯数据的情况，
        #   都会变成一个带 error 的 RpcResult；在写socket之前，RPC发送线程还会把无法pickle的值再转换一次。
        #   其他任何做法都会让调用方阻塞在一个没有超时的queue get上——这正是目前一个方法名拼写错误要花五分钟、
        #   最后报告的是超时而不是拼写错误的原因。
        try:
            func = getattr(runner, func_name, None)
        # （行尾的noqa说明：异常改为报告给调用方，所以故意捕获所有异常）
        except Exception as exc:  # noqa: BLE001 - reported to the caller instead
            # getattr's default covers AttributeError only. A non-string name
            # raises TypeError, which would otherwise end this worker's loop.
            # 【译】getattr的默认值只覆盖AttributeError。不是字符串的名字会抛TypeError，否则就会结束这个worker的循环。
            return self._failure(
                payload,
                f"cannot look up {func_name!r} on {type(runner).__name__}: "
                f"{type(exc).__name__}: {exc}",
            )
        if func is None:
            return self._failure(
                payload, f"{type(runner).__name__} has no method {func_name!r}"
            )
        try:
            value = func(*call_args, **call_kwargs)
        except Exception as exc:  # noqa: BLE001 - reported to the caller instead
            return self._failure(payload, f"{type(exc).__name__}: {exc}")
        # 返回值只允许None/bool/int/float/str/bytes及其list/tuple/dict嵌套：它要在EngineCore和API server里各unpickle一次，
        #   张量或只有worker才能import的类都会出问题（见collective_rpc.is_plain_rpc_value）
        if not is_plain_rpc_value(value):
            return self._failure(
                payload,
                f"{func_name!r} returned a {type(value).__name__} that is not "
                f"plain data: an RPC reply may hold only None, bool, int, "
                f"float, str and bytes, in lists, tuples and dicts",
            )
        return RpcResult(payload.request_id, self.rank, value=value)

    def _failure(self, payload: RpcPayload, error: str) -> RpcResult:
        """This rank's reply to *payload* when the call could not succeed."""
        # 【译】当调用无法成功时，本rank对 *payload* 的回复。
        return RpcResult(payload.request_id, self.rank, error=error)

    # 从共享内存广播队列取一条指令：(方法名, *参数)；如果是forward，把其中的增量block table还原成完整的表
    #   dequeue()在没有新消息时阻塞（读者轮询共享内存）；返回的args是list，decode_rpc会原地替换args[0]
    def get_func(self):
        method_name, *args = self.rpc_broadcast_mq.dequeue()
        return method_name, self._block_table_decoder.decode_rpc(method_name, args)


# 进程是否还活着；进程对象已被close()时is_alive()会抛ValueError，此时视为已死
def _alive(proc) -> bool:
    """``is_alive()``, except that a process teardown has closed is not."""
    # 【译】is_alive()，只是被teardown关闭过的进程不算活着。
    try:
        return proc.is_alive()
    # 【译】（行尾注释）被shutdown_all_processes关闭了
    except ValueError:  # closed by shutdown_all_processes
        return False


# 【分支】RPC输出线程收到无法unpickle的回复时，用它代替回复放进对应rank的queue，带上能读出的request_id
# @dataclass(frozen=True)：生成的实例不可修改（赋值会抛异常），可哈希
@dataclass(frozen=True)
class _RpcReceiveError:
    request_id: str
    error: str


# =====================================================================================
# AsyncIOProcManager：在EngineCore进程里管理一组ModelRunner进程
# =====================================================================================
# 职责：
#   1. 创建共享内存广播队列，spawn proc_num个AsyncIOProc子进程
#   2. 为每类输出起接收线程：主通道（rank 0）、每个rank的KV通道、每个rank的集合RPC通道
#   3. 对外提供三种调用：
#        call_func                  普通调用，广播给所有rank，可选等待rank 0的结果
#        call_func_with_aggregation KV传输状态聚合，收齐所有rank的结果再合并（第9批）
#        collective_rpc / utility_rpc  集合RPC，每个rank一个RpcResult（第10批RL rollout使用）
#   4. 监控子进程，任何一个意外退出就整体关闭
#
# 成员变量（按__init__中的顺序）：
#   parent_finalizer    EngineCore的finalizer；本管理器退出时调用它，让EngineCore也一起退出
#   proc_num            worker进程数（= tp_world_size x PCP大小）
#   procs               multiprocessing.Process列表，下标就是rank
#   runner_label / label  日志标签，如"AsyncIOProcManager(ModelRunner)"
#   rpc_broadcast_mq    共享内存广播队列的写者端（本进程创建，退出时负责unlink）
#   _block_table_encoder  forward指令的block table增量编码器（必须看到每一条forward）
#   still_running       False之后各接收线程会在1秒内退出循环
#   all_ranks_barrier   multiprocessing.Barrier(proc_num)，传给每个worker，集合RPC按需使用
#   kv_output_*         KV聚合：每个rank的地址、queue、接收线程，聚合器，以及唯一在途的那次聚合
#   rpc_output_*        集合RPC：每个rank的地址、queue、接收线程
#   _rpc_lock           保证同一时刻只有一个集合RPC在途
#   _dead_ranks         rank -> 退出码，由监控线程在关闭前写入，用于报告"哪个rank死了"
#   zmq_ctx             接收线程共用的ZMQ Context（2个I/O线程）
#   outputs_queue / output_thread  主通道的接收queue和线程
#   keep_monitoring     正常关闭前由EngineCore置为False，让监控线程忽略随后的子进程退出
class AsyncIOProcManager:
    """Manages a pool of :class:`AsyncIOProc` workers.

    Handles process lifecycle, function dispatch via shared-memory broadcast,
    and KV output aggregation across all workers.

    The manager maintains two output channels:
    - **Primary channel** (rank 0 only): Regular forward outputs.
    - **KV channels** (all ranks): Per-worker KV transfer status, aggregated
      by :class:`KVOutputAggregator` before returning to the caller.

    Args:
        finalizer: Callback invoked when the manager shuts down.
        proc_num: Number of worker processes (= TP world size).
        runner: Fully qualified class name of the model runner.
        *args: Additional arguments forwarded to the runner constructor.
    """

    # 【译】管理一组 AsyncIOProc worker。
    #
    #   负责进程生命周期、通过共享内存广播分派函数调用，以及跨所有worker汇总KV输出。
    #
    #   管理器维护两类输出通道：
    #   - 主通道（只有rank 0）：常规的forward输出。
    #   - KV通道（所有rank）：每个worker的KV传输状态，由 KVOutputAggregator 汇总后再返回给调用方。
    #
    #   参数：
    #     finalizer：管理器关闭时调用的回调。
    #     proc_num：worker进程数（= TP world size）。
    #     runner：model runner的完整限定类名。
    #     *args：转发给runner构造函数的其他参数。
    # 解释：后来又加了第三类通道——每个rank一条的集合RPC回复通道（见下面的rpc_output_*）；
    #   proc_num在PCP>1时是tp_world_size x PCP大小，不只是TP

    # The public collective_rpc runs only these.
    # 【译】公开的collective_rpc只运行这些方法。
    # ClassVar：标注这是类变量而不是实例字段
    _PUBLIC_RPC_METHODS: ClassVar[frozenset[str]] = COLLECTIVE_RPC_METHODS
    # Names the engine's wire protocol owns, which even the utility handler's
    # transport may not send. With the utility-managed ones, which need
    # EngineUtilityHandler to keep engine state around their worker call,
    # they are what must never be opened to the public path.
    # 【译】由引擎自己的线路协议占用的名字，连utility handler的传输通道也不能发送。再加上"由utility管理"的那些
    #   （它们需要EngineUtilityHandler在worker调用前后维护引擎状态），就是绝不能对公开路径开放的全部名字。
    # 解释：FORWARD_RPC即"forward"；"exit"；以及KV聚合方法。_RESERVED_RPC_NAMES = 协议名 ∪ utility管理的方法名
    _PROTOCOL_RPC_NAMES: ClassVar[frozenset[str]] = frozenset(
        {FORWARD_RPC, "exit", *AsyncIOProc._KV_FUNC_NAMES}
    )
    _RESERVED_RPC_NAMES: ClassVar[frozenset[str]] = frozenset(
        {*_PROTOCOL_RPC_NAMES, *UTILITY_MANAGED_RUNNER_METHODS}
    )

    # 输入：finalizer（EngineCore的finalizer）、proc_num、runner（类的完整限定名字符串）、*args（转给runner，即config）
    def __init__(self, finalizer, proc_num: int, runner: str, *args):
        self.parent_finalizer = finalizer
        self.proc_num = proc_num

        # 两个IPC地址：[0]输入（遗留，实际不用），[1]主输出（rank 0的PUSH → 本进程的PULL）
        # get_open_zmq_ipc_path：生成形如 ipc:///tmp/<uuid> 的唯一地址（Unix域socket），此时还没有任何socket绑定它
        io_addrs = [get_open_zmq_ipc_path(), get_open_zmq_ipc_path()]
        self.procs: list[multiprocessing.Process] = []
        # spawn方式的multiprocessing上下文：子进程是全新的Python解释器，参数靠pickle传过去；
        #   不能用fork，因为fork会复制父进程里已经初始化的GPU运行时状态，子进程里再用会崩溃（见CLAUDE.md）
        ctx = get_mp_context()
        self.runner_label = runner.split(".")[-1]
        self.label = f"AsyncIOProcManager({self.runner_label})"

        # 写者端：MessageQueue(n_reader=proc_num, n_local_reader=proc_num, ...)，所有读者都在本机；
        #   max_chunk_bytes=16MiB：每块的大小，一条forward指令（pickle后的ScheduledBatch）要放得进去
        #   （超过的话vLLM原版会改走ZMQ，aiter的行为以其源码为准）
        self.rpc_broadcast_mq = MessageQueue(
            proc_num, proc_num, max_chunk_bytes=16 * 1024 * 1024
        )
        self._block_table_encoder = BlockTableDeltaEncoder()
        # 导出一个可以pickle的句柄，作为参数传给每个子进程，子进程用create_from_handle连接到同一块共享内存
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        self.still_running = True
        # Register atexit to clean up shared memory even if exit() doesn't complete
        # 【译】注册atexit，即使exit()没有执行完也能清理共享内存
        import atexit

        atexit.register(self._cleanup_shared_memory)
        # 跨进程的Barrier：spawn时作为参数传给子进程（multiprocessing的同步原语只能在创建进程时传递）
        self.all_ranks_barrier = ctx.Barrier(proc_num)
        # 注意：signal.signal会覆盖EngineCore之前装的SIGTERM处理器，之后EngineCore进程收到SIGTERM时，
        #   先走本管理器的exit()，再由它调用parent_finalizer（即EngineCore.exit）
        init_exit_handler(self)

        # KV output aggregation infrastructure
        # 【译】KV输出聚合的基础设施
        # 【分支】PD分离/KV offload，见第9批；聚合器在第一次call_func_with_aggregation时才创建
        self.kv_output_aggregator: KVOutputAggregator | None = None
        self.kv_output_addrs = [get_open_zmq_ipc_path() for _ in range(proc_num)]
        self.kv_outputs_queues: list[queue.Queue] = [
            queue.Queue() for _ in range(proc_num)
        ]
        self._pending_kv_aggregation: _PendingKvAggregation | None = None
        self.kv_output_threads: list[threading.Thread] = []

        # Generic collective-RPC reply channels, one per rank
        # 【译】通用集合RPC的回复通道，每个rank一条
        self.rpc_output_addrs = [get_open_zmq_ipc_path() for _ in range(proc_num)]
        self.rpc_outputs_queues: list[queue.Queue] = [
            queue.Queue() for _ in range(proc_num)
        ]
        self.rpc_output_threads: list[threading.Thread] = []
        # Every call reads these same per-rank queues; see collective_rpc.
        # 【译】每次调用都读同样这些逐rank的queue；见 collective_rpc。
        self._rpc_lock = threading.Lock()
        # rank -> exit code, written by the monitor before it tears down.
        # 【译】rank -> 退出码，由监控线程在执行关闭之前写入。
        self._dead_ranks: dict[int, int | None] = {}

        # ---------- spawn每个worker进程 ----------
        for i in range(proc_num):
            # 例：proc_num=2 → "ModelRunner0/2"、"ModelRunner1/2"
            label = f"ModelRunner{i}/{proc_num}"
            # Only rank 0 gets the primary output address
            # 【译】只有rank 0拿到主输出地址
            # 输入地址恒为None（不启动遗留的输入线程）
            addrs = [None, io_addrs[1]] if i == 0 else [None, None]

            # target=AsyncIOProc：子进程里执行AsyncIOProc(*args, **kwargs)，即上面说的"构造函数就是整个进程"
            process = ctx.Process(
                target=AsyncIOProc,
                name=label,
                args=(
                    label,
                    addrs,
                    scheduler_output_handle,
                    runner,
                    i,
                    self.kv_output_addrs[i],
                    self.all_ranks_barrier,
                    *args,
                ),
                # Keyword-only on AsyncIOProc, so it cannot be mistaken for one
                # of the *args forwarded to the runner's constructor.
                # 【译】在AsyncIOProc上它是仅限关键字的参数，所以不会被误当成转发给runner构造函数的*args之一。
                kwargs={"rpc_output_addr": self.rpc_output_addrs[i]},
            )
            # start()立即返回；子进程在后台加载模型。下面的接收socket稍后才bind，ZMQ的connect会自动重试，先后顺序无所谓
            process.start()
            self.procs.append(process)

        # 本进程所有接收socket共用一个ZMQ Context，io_threads=2：ZMQ内部用2个后台I/O线程
        self.zmq_ctx = zmq.Context(io_threads=2)

        # Primary output queue (rank 0 only)
        # 【译】主输出queue（只有rank 0）
        # call_func(wait_out=True)从outputs_queue阻塞地取结果
        self.outputs_queue: queue.Queue = queue.Queue()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            name=f"{self.label}_output_thread",
            args=(io_addrs[1],),
            daemon=True,
        )
        self.output_thread.start()

        # Per-worker KV output channels
        # 【译】每个worker各自的KV输出通道
        for i, output_addr in enumerate(self.kv_output_addrs):
            t = threading.Thread(
                target=self.process_kv_output_sockets,
                name=f"{self.label}_kv_output_thread_{i}",
                args=(output_addr, i),
                daemon=True,
            )
            t.start()
            self.kv_output_threads.append(t)

        # Per-worker collective-RPC reply channels
        # 【译】每个worker各自的集合RPC回复通道
        for i, output_addr in enumerate(self.rpc_output_addrs):
            t = threading.Thread(
                target=self.process_rpc_output_sockets,
                name=f"{self.label}_rpc_output_thread_{i}",
                args=(output_addr, i),
                daemon=True,
            )
            t.start()
            self.rpc_output_threads.append(t)

        # 启动监控线程，然后__init__返回（此时worker多半还在加载模型；EngineCore随后call_func("get_num_blocks", wait_out=True)
        #   会一直等到rank 0构造完ModelRunner、进入busy_loop并回复）
        self.monitor_procs()

    # 关闭整个管理器
    # 调用方：监控线程（某个worker意外退出时）、weakref.finalize（SIGTERM或解释器退出时）
    # 正常关闭时EngineCore.exit先置keep_monitoring=False、call_func("exit")让worker自己退出、再join它们，
    #   本方法最终在EngineCore进程退出时由finalizer执行
    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # 第1步：unlink并关闭共享内存（本进程是创建者）
        self._cleanup_shared_memory()
        logger.info(f"{self.label}: shutdown all runners...")
        # 第2步：在ATOM_SHUTDOWN_TIMEOUT_S的总时限内等worker自己退出（所有进程共用一个截止时间）
        deadline = time.monotonic() + envs.ATOM_SHUTDOWN_TIMEOUT_S
        for proc in self.procs:
            if proc.is_alive():
                proc.join(timeout=max(deadline - time.monotonic(), 0))
        # 第3步：还没退出的worker：terminate → 等待 → kill（见atom/utils/__init__.py的shutdown_all_processes）
        shutdown_all_processes(self.procs, allowed_seconds=1)
        # 注意：这里是重新绑定为新列表，监控线程和进行中的collective_rpc手里持有的是旧列表的引用
        self.procs = []
        # 第4步：等接收线程退出（still_running已为False，它们会在poll超时后退出循环）
        self.output_thread.join(timeout=1)
        for thread in self.kv_output_threads:
            thread.join(timeout=0.5)
        for thread in self.rpc_output_threads:
            thread.join(timeout=0.5)
        logger.info(f"{self.label}: All runners are shutdown.")
        # 第5步：往主输出queue放一个SystemExit：如果EngineCore正阻塞在call_func(wait_out=True)上，
        #   它会取到这个对象并把它抛出，而不是永远等下去
        self.outputs_queue.put_nowait(SystemExit())

        # 第6步：让EngineCore也退出
        self.parent_finalizer()

    def _cleanup_shared_memory(self):
        """Clean up shared memory (creator side: close + unlink)."""
        # 【译】清理共享内存（创建者一侧：close + unlink）。
        if hasattr(self, "rpc_broadcast_mq"):
            mq = self.rpc_broadcast_mq
            if hasattr(mq, "buffer") and hasattr(mq.buffer, "shared_memory"):
                try:
                    shm = mq.buffer.shared_memory
                    # 只有创建者才unlink（删除共享内存段的名字）；读者只close
                    if mq.buffer.is_creator:
                        shm.unlink()
                        mq.buffer.is_creator = False
                    shm.close()
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    # Creator side: an unlink of an already-removed segment is
                    # the common case and is harmless. Log rather than pass so
                    # a real leak is visible to whoever goes looking.
                    # 【译】创建者一侧：unlink一个已经被删除的段是常见情况，且无害。这里记日志而不是pass，
                    #   这样真正的泄漏对于去排查的人是可见的。
                    logger.debug(f"{self.label}: shm unlink/close failed: {e}")

    # 主通道接收线程：rank 0的结果 → outputs_queue
    # 输入：主输出地址（本进程PULL bind）
    def process_output_sockets(self, output_address: str):
        """Receive results from rank 0's primary output channel."""
        # 【译】从rank 0的主输出通道接收结果。
        # PULL按默认规则bind；rank 0的PUSH会connect过来
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            # zmq.Poller：同时等待一个或多个socket可读；poll(timeout=1000)最多等1秒，超时返回空列表，
            #   用于定期检查still_running
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                # 零拷贝接收后反序列化，放进outputs_queue，交给阻塞在call_func(wait_out=True)里的调用方
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)
                self.outputs_queue.put_nowait(obj)
        finally:
            # finally：无论怎么退出都关闭socket；linger=0表示丢弃没发完的消息，立即关闭
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: output thread exit")

    # 【分支】KV通道接收线程，每个rank一个：结果放进kv_outputs_queues[worker_id]，见第9批
    def process_kv_output_sockets(self, output_address: str, worker_id: int):
        """Receive KV output from each worker's dedicated channel."""
        # 【译】从每个worker的专用通道接收KV输出。
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)
                self.kv_outputs_queues[worker_id].put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: kv output thread {worker_id} exit")

    # 【分支】集合RPC接收线程，每个rank一个：结果放进rpc_outputs_queues[worker_id]
    def process_rpc_output_sockets(self, output_address: str, worker_id: int):
        """Receive generic collective-RPC replies from one worker."""
        # 【译】从一个worker接收通用集合RPC回复。
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                # 两帧：[request_id, 消息体]；先单独解出request_id，这样即使消息体unpickle失败也知道它属于哪次调用
                request_id_frame, body = output_socket.recv_multipart(copy=False)
                request_id = bytes(request_id_frame).decode("utf-8", errors="replace")
                try:
                    obj = pickle.loads(body)
                except Exception as exc:
                    obj = _RpcReceiveError(
                        request_id,
                        f"could not decode RPC reply from TP rank {worker_id}: "
                        f"{type(exc).__name__}: {exc}",
                    )
                    logger.exception(obj.error)
                self.rpc_outputs_queues[worker_id].put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: rpc output thread {worker_id} exit")

    # ---------- 集合RPC（collective RPC）----------
    # 【分支】给RL rollout等外部调用方用的"在每个TP rank上调用一个worker方法"，见第10批
    # 调用链：AsyncLLMEngine.collective_rpc → CoreManager.collective_rpc → （工具RPC，经控制socket）
    #   → EngineUtilityHandler._handle_collective_rpc → 这里 → 每个AsyncIOProc._run_generic_rpc
    # 公开入口：只允许COLLECTIVE_RPC_METHODS白名单里的方法
    def collective_rpc(
        self,
        func_name: str,
        payload: RpcPayload,
        timeout: float = 300.0,
    ) -> list[RpcResult]:
        """Public worker RPC: runs only the methods in COLLECTIVE_RPC_METHODS."""
        # 【译】公开的worker RPC：只运行 COLLECTIVE_RPC_METHODS 里的方法。
        return self._collective_rpc(
            func_name, payload, timeout, allow_utility_managed=False
        )

    # EngineUtilityHandler专用入口：允许"由utility管理"的方法（权重更新、显存释放/恢复等），但仍拒绝协议保留名
    def utility_rpc(
        self,
        func_name: str,
        payload: RpcPayload,
        timeout: float = 300.0,
    ) -> list[RpcResult]:
        """All-rank transport for EngineUtilityHandler-owned methods."""
        # 【译】为EngineUtilityHandler所拥有的方法提供的全rank传输通道。
        return self._collective_rpc(
            func_name, payload, timeout, allow_utility_managed=True
        )

    # 两个入口共用的实现
    # 输入：方法名、RpcPayload、超时（秒）、是否允许utility管理的方法
    # 输出：长度恰为proc_num、按rank排序的RpcResult列表（失败的rank带error，不抛异常）
    def _collective_rpc(
        self,
        func_name: str,
        payload: RpcPayload,
        timeout: float,
        *,
        allow_utility_managed: bool,
    ) -> list[RpcResult]:
        """Run *func_name* on every TP runner and return one reply per rank.

        Unlike :meth:`call_func`, which surfaces only rank 0's return, this
        collects from every rank's own channel, so "all ranks finished" is
        observable.

        Calls are serialized. Every caller reads the same per-rank channels, so
        two outstanding at once would each take the other's replies, drop them
        as stale, and time out. What the ``request_id`` buys here is that a late
        reply from a call that already gave up is recognised and dropped,
        rather than taken as the next call's answer.

        Always returns ``proc_num`` results in rank order. A rank that died, or
        that did not answer within *timeout*, yields a failed ``RpcResult``
        rather than an exception or a short list -- the caller needs to know
        *which* rank is missing, and a raise here would lose the ranks that did
        answer.
        """
        # 【译】在每个TP runner上运行 *func_name*，每个rank返回一个回复。
        #
        #   与只给出rank 0返回值的 call_func 不同，这里从每个rank自己的通道收集，所以"所有rank都完成了"是可以观察到的。
        #
        #   调用是串行的。每个调用方读的都是同样这些逐rank通道，所以如果同时有两个在途，它们会各自拿走对方的回复、
        #   把它们当作过期回复丢掉，然后超时。request_id 在这里换来的是：一个已经放弃的调用迟到的回复会被识别出来并丢弃，
        #   而不是被当作下一个调用的答案。
        #
        #   总是按rank顺序返回 proc_num 个结果。死掉的rank、或在 *timeout* 内没有回复的rank，得到的是一个失败的
        #   RpcResult，而不是异常或更短的列表——调用方需要知道*哪个*rank缺失了，而在这里抛异常会丢掉那些已经回复的rank。
        if not isinstance(payload, RpcPayload):
            raise TypeError(
                f"collective_rpc needs an RpcPayload, got {type(payload).__name__}"
            )
        # Reached from the utility handler too, with whatever timeout the
        # utility command carried.
        # 【译】也会从utility handler到达这里，带着utility命令携带的任意超时值。
        # checked_timeout：拒绝NaN/无穷大，这两种值永远到不了截止时间
        timeout = checked_timeout(timeout)
        # 名字检查：公开入口只放行白名单；utility入口放行除协议保留名以外的方法
        if not allow_utility_managed:
            if func_name not in self._PUBLIC_RPC_METHODS:
                raise ValueError(
                    f"{self.label}: {func_name!r} is not open to collective_rpc, "
                    f"which runs only {sorted(self._PUBLIC_RPC_METHODS)}"
                )
        elif func_name in self._PROTOCOL_RPC_NAMES:
            # Each would break the protocol this path goes around. The workers'
            # decoder resets its cached rows on a forward the encoder never saw,
            # failing the next scheduled one. Exit tears down every worker's
            # runner and ends its loop, behind a reply that reads as success.
            # And a KV aggregation drains transfer completions the scheduler
            # would then never see, stalling the requests waiting on them.
            # 【译】它们中的每一个都会破坏本路径所绕开的协议。worker的解码器在遇到编码器从未见过的forward时会重置缓存的行，
            #   使下一次被调度的forward失败。exit会拆掉每个worker的runner并结束其循环，而回复看起来却像成功。
            #   KV聚合则会取走调度器之后再也看不到的传输完成信息，让等待它们的请求卡住。
            raise ValueError(
                f"{self.label}: {func_name!r} is reserved for the engine's own "
                f"protocol and cannot go through collective_rpc"
            )

        # 截止时间已过：一个都不发
        if timeout <= 0:
            # Sent anyway, every rank ran it while this call gave up at once and
            # dropped their replies as late: a failure reported for work that
            # happened -- a resume that took effect while the engine still
            # recorded the weights offloaded -- which a retry then repeated.
            # 【译】如果照样发出去，每个rank都会执行它，而这次调用却立刻放弃并把它们的回复当作迟到的丢掉：
            #   对已经发生的工作报告了失败——例如一次resume已经生效，而引擎仍记录权重处于offload状态——随后的重试又把它重复执行一遍。
            error = (
                f"the deadline for {func_name!r} had passed before it was sent, "
                f"so no TP rank ran it"
            )
            return [
                RpcResult(payload.request_id, rank, error=error)
                for rank in range(self.proc_num)
            ]

        logger.debug(
            f"{self.label}: collective_rpc {func_name} id={payload.request_id}"
        )
        # 串行化：同一时刻只有一个集合RPC在读这些逐rank的queue
        with self._rpc_lock:
            barrier = getattr(self, "all_ranks_barrier", None)
            # barrier已经被打破过（有rank死过）：它会一直保持打破状态，需要barrier的调用不可能成功，直接报错
            if payload.barrier and barrier is not None and barrier.broken:
                error = (
                    f"cannot run barrier RPC {func_name!r}: the TP barrier is "
                    f"broken; restart the runner manager"
                )
                return [
                    RpcResult(payload.request_id, rank, error=error)
                    for rank in range(self.proc_num)
                ]
            # 广播：与普通调用走同一个共享内存队列，参数只有一个RpcPayload
            self.rpc_broadcast_mq.enqueue((func_name, payload))

            # This call's own copy: a worker's death makes the monitor run
            # exit(), which empties self.procs while replies are still awaited.
            # 【译】这次调用自己的副本：worker死亡会使监控线程执行 exit()，而 exit() 会在仍有回复待收时清空 self.procs。
            procs = list(self.procs)
            deadline = time.monotonic() + timeout
            results: list[RpcResult] = []
            # 按rank顺序逐个等；所有rank共用同一个截止时间
            for rank, output_queue in enumerate(self.rpc_outputs_queues):
                results.append(
                    self._await_rank_reply(
                        rank, output_queue, func_name, payload, deadline, procs
                    )
                )
        return results

    # 等一个rank的回复：收到正确的RpcResult就返回；否则构造一个带原因的失败RpcResult
    def _await_rank_reply(
        self,
        rank: int,
        output_queue: queue.Queue,
        func_name: str,
        payload: RpcPayload,
        deadline: float,
        procs: list,
    ) -> RpcResult:
        """Wait for one rank's reply, or synthesise the reason there is none."""
        # 【译】等待一个rank的回复，或者合成"为什么没有回复"的原因。
        while True:
            remaining = deadline - time.monotonic()
            try:
                # Poll rather than block for the whole budget, so a worker that
                # dies mid-call is reported promptly instead of at the deadline.
                # 【译】轮询而不是在整个预算内阻塞，这样一个在调用中途死掉的worker能被及时报告，而不是等到截止时间。
                # 每次最多等1秒（剩余时间不足1秒时等剩余时间）
                reply = output_queue.get(timeout=max(0.0, min(1.0, remaining)))
            except queue.Empty:
                # Before this call can give up: once it has, nothing else would
                # release ranks still waiting on a dead one.
                # 【译】在本次调用可能放弃之前做：一旦它放弃了，就再没有别的东西会释放那些还在等一个死rank的rank。
                self._break_barrier_for_dead_ranks(procs)
                # Death before the deadline, even once it has passed: waiting on
                # another rank can use up the budget, and a rank that died is
                # still the reason this one never answered.
                # 【译】即使截止时间已过，死亡也优先于超时：等待其他rank可能用光了预算，而一个已经死掉的rank仍然是它从未回复的原因。
                dead = dict(self._dead_ranks)
                if rank in dead:
                    error = (
                        f"TP rank {rank} died before answering {func_name!r} "
                        f"(exitcode={dead[rank]})"
                    )
                elif not self.still_running:
                    # Teardown stops every rank and their reply threads with
                    # them, so no answer can come any more.
                    # 【译】teardown会停止每个rank以及它们的回复线程，所以不可能再有回复了。
                    cause = f", after TP rank(s) {sorted(dead)} died" if dead else ""
                    error = (
                        f"TP rank {rank} was shut down before answering "
                        f"{func_name!r}{cause}"
                    )
                elif rank < len(procs) and not _alive(procs[rank]):
                    error = f"TP rank {rank} died before answering {func_name!r}"
                elif remaining <= 0:
                    error = f"timed out waiting for {func_name!r} on TP rank {rank}"
                else:
                    # 还活着、也没超时：继续等
                    continue
            else:
                # 收到了东西：依次检查是不是解码失败、类型对不对、是不是本次调用的、rank对不对
                if isinstance(reply, _RpcReceiveError):
                    if reply.request_id != payload.request_id:
                        logger.warning(
                            "%s: dropping stale unreadable reply %s from rank %d "
                            "while awaiting %s",
                            self.label,
                            reply.request_id,
                            rank,
                            payload.request_id,
                        )
                        continue
                    error = reply.error
                elif not isinstance(reply, RpcResult):
                    error = (
                        f"unexpected reply type {type(reply).__name__} from rank {rank}"
                    )
                elif reply.request_id != payload.request_id:
                    # A late reply from an earlier call. Dropping it is correct:
                    # that caller has already been answered or has given up.
                    # 【译】一个来自更早调用的迟到回复。丢弃它是正确的：那个调用方已经得到回答或者已经放弃了。
                    logger.warning(
                        "%s: dropping stale reply %s from rank %d while awaiting %s",
                        self.label,
                        reply.request_id,
                        rank,
                        payload.request_id,
                    )
                    continue
                elif reply.tp_rank != rank:
                    error = (
                        f"reply rank mismatch: channel {rank} carried {reply.tp_rank}"
                    )
                else:
                    return reply
            return RpcResult(payload.request_id, rank, error=error)

    # 发现有rank死了就abort()共享的barrier，让正在barrier上等待的其他rank得到BrokenBarrierError并回复
    def _break_barrier_for_dead_ranks(self, procs: list) -> None:
        """Release the ranks waiting at the barrier once one can never arrive.

        A dead rank never reaches it, so the rest would wait there forever and
        never answer. Breaking it hands each of them a BrokenBarrierError to
        answer with instead. A rank that is only slow is left to the deadline:
        it can still arrive, and a broken barrier stays broken for every later
        call that needs one.

        Checked while waiting on any call, not only barrier ones: a barrier
        call that timed out on a slow rank leaves the rest waiting, and if that
        rank dies afterwards, only a later call is there to notice. Not during
        teardown, which stops every rank anyway.
        """
        # 【译】一旦某个rank永远到不了，就释放在barrier处等待的那些rank。
        #
        #   一个死掉的rank永远到不了barrier，所以其余rank会在那里一直等下去，永远不回复。打破它会给它们每个一个
        #   BrokenBarrierError，让它们拿去回复。一个只是慢的rank则交给截止时间处理：它仍然可能到达，而一个被打破的barrier
        #   对之后每一个需要它的调用都会保持打破状态。
        #
        #   在等待任何调用时都做这个检查，而不只是barrier调用：一个在慢rank上超时的barrier调用会让其余rank继续等，
        #   如果那个rank之后才死掉，只有之后的某个调用会注意到。teardown期间不做，因为它反正会停掉每个rank。
        barrier = getattr(self, "all_ranks_barrier", None)
        if barrier is None or barrier.broken or not self.still_running:
            return
        dead = [rank for rank, proc in enumerate(procs) if not _alive(proc)]
        if dead:
            logger.error(
                "%s: TP rank(s) %s died during a barrier call; releasing the rest",
                self.label,
                dead,
            )
            barrier.abort()

    # ---------- 普通调用：EngineCore与ModelRunner之间的主干路径 ----------
    # 输入：方法名、位置参数、wait_out（是否等待结果）
    # 输出：wait_out=True时返回rank 0的返回值；否则None（不等待）
    # 调用方：EngineCore的几乎所有worker操作，例如
    #   call_func("get_num_blocks", wait_out=True) / call_func("allocate_kv_cache", n, wait_out=True)  启动时
    #   call_func("forward", scheduled_batch, wait_out=True)       每个step一次
    #   call_func("process_kvconnector_output", meta)               不等待
    #   call_func("exit")                                           关闭时
    # 注意：outputs_queue是所有调用共用的一个FIFO，结果按顺序配对。所以不等待结果的调用，目标方法必须返回None，
    #   否则rank 0送回的结果会被下一个wait_out=True的调用当成自己的
    def call_func(self, func_name: str, *args, wait_out: bool = False):
        """Standard RPC call for non-KV operations."""
        # 【译】用于非KV操作的标准RPC调用。
        logger.debug(f"{self.label}: call_func {func_name} {args}")
        # 只对forward生效：把batch里的block_tables换成"相对上次发送的增量"（block_table_codec.py）
        args = self._block_table_encoder.encode_rpc(func_name, args)
        # 指令格式：(方法名, *参数)，例如("forward", batch)
        msg = (func_name, *args)
        # 广播给所有rank（写一次，所有读者都能读到）
        self.rpc_broadcast_mq.enqueue(msg)
        if wait_out:
            # 阻塞等待rank 0的结果（没有超时）
            ret = self.outputs_queue.get()
            # exit()放进来的SystemExit：说明管理器已关闭，抛给调用方
            if isinstance(ret, SystemExit):
                raise ret
            return ret

    # 【分支】以下两个方法是KV传输状态聚合（PD分离/KV offload），见第9批
    #   调用方：EngineCore._poll_kv_transfer_progress，每个step一次
    def _start_kv_aggregation(
        self, func_name: str, args: tuple[object, ...]
    ) -> _PendingKvAggregation:
        """Broadcast one KV aggregation RPC and record its empty result slots."""
        # 【译】广播一次KV聚合RPC，并记录它的空结果槽位。
        self.rpc_broadcast_mq.enqueue((func_name, *args))
        pending = _PendingKvAggregation(
            worker_outputs=[None] * self.proc_num,
            started_at=time.monotonic(),
        )
        self._pending_kv_aggregation = pending
        return pending

    # 输入：方法名（目前只有"async_proc_aggregation"）、超时告警阈值
    # 输出：所有rank都到齐时返回合并后的KVConnectorOutput；还有rank没回复时返回None（不阻塞）
    def call_func_with_aggregation(self, func_name: str, *args, timeout: float = 10.0):
        """RPC call with KV output aggregation across all workers.

        At most one aggregation is outstanding. The first call starts it;
        subsequent calls non-blockingly drain its per-rank replies. Once a batch
        is complete, the next batch is started before returning the completed
        result, keeping one poll in flight without delaying every other engine
        step. A missing rank leaves the current aggregation open and returns
        ``None``. ``timeout`` is only the age after which it is logged.

        Args:
            func_name: Method name to invoke on each worker's runner.
            timeout: Seconds an incomplete aggregation may age before logging.

        Returns:
            Aggregated :class:`KVConnectorOutput`, or ``None`` while the
            current aggregation is still missing ranks.
        """
        # 【译】跨所有worker做KV输出聚合的RPC调用。
        #
        #   最多只有一个聚合在途。第一次调用启动它；之后的调用以非阻塞方式取走它的逐rank回复。一批收齐后，
        #   在返回这批已完成的结果之前先启动下一批，从而始终保持一个轮询在途，又不会拖慢其他每个引擎step。
        #   缺少某个rank时，当前聚合保持打开并返回 None。timeout 只是聚合"多老之后打日志"的阈值。
        #
        #   参数：
        #     func_name：在每个worker的runner上调用的方法名。
        #     timeout：一个未完成的聚合在打日志之前可以存在的秒数。
        #
        #   返回：
        #     聚合后的 KVConnectorOutput；当前聚合仍缺少某些rank时返回 None。
        if self.kv_output_aggregator is None:
            self.kv_output_aggregator = KVOutputAggregator(world_size=self.proc_num)

        logger.debug(f"{self.label}: call_func_with_aggregation {func_name} {args}")
        pending = self._pending_kv_aggregation
        if pending is None:
            pending = self._start_kv_aggregation(func_name, args)

        # Each rank has its own queue, so later ranks can be consumed while
        # an earlier rank is still missing. Replies are drain-once and carry
        # no generation identifier, so they belong to this outstanding
        # aggregation until it closes.
        # 【译】每个rank有自己的queue，所以在较早的rank还缺失时，可以先消费较晚的rank。回复只能取走一次且不带
        #   "第几轮"的标识，所以在当前在途的聚合关闭之前，它们都属于这个聚合。
        for rank, output_queue in enumerate(self.kv_outputs_queues):
            if pending.worker_outputs[rank] is not None:
                continue
            try:
                pending.worker_outputs[rank] = output_queue.get_nowait()
            except queue.Empty:
                continue

        # 还有rank没到：超过timeout只告警一次，返回None，下个step再来取
        missing = pending.missing_worker_ranks()
        if missing:
            waited = time.monotonic() - pending.started_at
            if timeout > 0 and waited >= timeout and not pending.wait_warning_logged:
                logger.error(
                    "%s: KV aggregation still waiting for workers %s after %.1fs",
                    self.label,
                    missing,
                    waited,
                )
                pending.wait_warning_logged = True
            return None

        # 全部到齐：合并成一个KVConnectorOutput（一个请求只有在所有rank都报告传输完成时才算完成）
        kv_output = self.kv_output_aggregator.aggregate(
            worker_outputs=[
                output for output in pending.worker_outputs if output is not None
            ]
        )
        self._pending_kv_aggregation = None
        # Keep the completion query pipelined: workers can produce this next
        # snapshot while EngineCore processes the completed one and executes
        # the following forward. This still leaves exactly one outstanding RPC.
        # 【译】让完成状态查询保持流水线化：在EngineCore处理已完成的这一份并执行下一次forward的同时，
        #   worker可以生成下一份快照。这样仍然恰好只有一个在途的RPC。
        self._start_kv_aggregation(func_name, args)
        logger.debug(f"Aggregated KV output: {kv_output}")
        return kv_output

    # ---------- 监控线程：任何一个worker意外退出，就关闭整个管理器 ----------
    # weakref.ref(self)：弱引用，不增加引用计数；用_self = self_ref()取回对象，对象已被回收时得到None
    def monitor_procs(self):
        self_ref = weakref.ref(self)
        # 局部变量持有原列表：之后exit()把self.procs重新绑定为[]，这里的procs不受影响
        procs = self.procs
        # 正常关闭前EngineCore会把它置为False，监控线程看到后直接返回，不做任何事
        self.keep_monitoring = True

        def monitor_engine_cores():
            # sentinel：每个Process的"退出信号"句柄，进程结束时变为就绪
            sentinels = [proc.sentinel for proc in procs]
            # multiprocessing.connection.wait：阻塞直到列表中至少一个句柄就绪，即至少一个worker退出
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or not _self.keep_monitoring:
                return
            # 找出是哪个进程退出了，先join回收它（最多5秒）
            dead_proc = next(proc for proc in procs if proc.sentinel == died[0])
            dead_proc.join(timeout=5)
            # Before exit() empties self.procs and stops every other rank, so
            # a collective_rpc in flight can still name the one that died.
            # 【译】在 exit() 清空 self.procs、停掉其他每个rank之前做，这样一个在途的collective_rpc仍然能指出死掉的是哪一个。
            _self._dead_ranks[procs.index(dead_proc)] = dead_proc.exitcode
            # 注意：这里的f-string引用的是外层的self而不是_self，所以闭包会一直强引用self，上面的弱引用实际上没起到作用
            logger.error(
                f"{self.label}: [{dead_proc.name}] proc died unexpectedly "
                f"(exitcode={dead_proc.exitcode}), shutting down.",
            )
            # 关闭整个管理器：会往outputs_queue放SystemExit，唤醒阻塞在call_func上的EngineCore，并调用EngineCore的finalizer
            _self.exit()

        # daemon线程：不阻止进程退出
        Thread(
            target=monitor_engine_cores, daemon=True, name=f"{self.runner_label}Monitor"
        ).start()
