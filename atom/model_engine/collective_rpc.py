# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wire types for the generic collective RPC.

Deliberately free of heavy imports. ``async_proc`` cannot be imported without a
real AITER build, and ``engine_utility`` needs these same types to build a
request, so keeping them here is what lets the dispatch layer stay importable on
a machine with no GPU -- which is also how the non-GPU test runner sees them.
"""

# 【译】通用集合RPC的线路类型。
#
#   故意不带重量级的导入。async_proc 没有真实的AITER构建就无法导入，而 engine_utility 需要同样这些类型来构造请求，
#   所以把它们放在这里，分发层才能在没有GPU的机器上保持可导入——这也是非GPU测试运行器看到它们的方式。

import math
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# 【导读】集合RPC（collective RPC）的线路类型与常量
# 调用方：async_proc.py（worker端执行与回复）、engine_utility.py（EngineCore端构造请求）、
#   engine_core_mgr.py（CoreManager端按request_id路由回复、计算引擎侧时间预算）
# 与主干的关系：主干的forward走call_func（只回rank 0、不带request_id）；集合RPC是旁路，
#   给RL rollout等外部调用方"在每个TP rank上调一个方法并拿回每个rank的结果"，见第10批

# The utility-command name the dispatch layer registers. Shared so the manager
# that sends it and the handler that receives it cannot drift apart.
# 【译】分发层注册的工具命令名。共享出来，使发送它的manager和接收它的handler不会各说各话。
COLLECTIVE_RPC_CMD = "collective_rpc"
# 权重同步的两个控制命令：放弃 / 提交（见engine_utility.py的围栏协议）
DISCARD_WEIGHT_SYNC_CMD = "discard_failed_weight_sync"
FINISH_WEIGHT_SYNC_CMD = "finish_weight_sync"

# The runner methods the public collective_rpc runs, and so the ones a
# capability report advertises. An allowlist: the rest of a runner -- its
# profiler, KV allocation, graph capture, dummy forwards -- has engine state
# around it that a direct call bypasses, and under DP a dummy forward on one
# engine waits on peers that never join it. A method is opened by adding it.
# 【译】公开的collective_rpc会运行的runner方法，因此也就是能力报告所宣称的那些。这是白名单：runner的其余部分——
#   它的profiler、KV分配、graph捕获、dummy forward——外面都包着引擎状态，直接调用会绕过它；而且在DP下，
#   一个引擎上的dummy forward会等待永远不会加入的对端。要开放一个方法，就把它加进来。
COLLECTIVE_RPC_METHODS = frozenset(
    {
        # hidden-state extraction
        # 【译】隐藏状态提取
        "configure_hidden_states",
        # capability discovery itself
        # 【译】能力发现本身
        "get_worker_capabilities",
    }
)

# These methods have an EngineUtilityHandler command that owns engine state
# around the worker call. Driving the worker method directly would bypass that
# state -- sleeping/offload flags and all-rank update checks included.
# 【译】这些方法有一个EngineUtilityHandler命令在worker调用前后维护引擎状态。直接驱动worker方法会绕过那些状态——
#   包括睡眠/offload标志和全rank的更新检查。
UTILITY_MANAGED_RUNNER_METHODS = frozenset(
    {
        "update_weights",
        "update_weights_from_shm",
        "update_weights_from_ipc",
        DISCARD_WEIGHT_SYNC_CMD,
        "release_memory",
        "resume_memory",
        "clear_kv_cache",
    }
)


# 把超时转换成有限的秒数；NaN、无穷大或无法转换时抛ValueError
def checked_timeout(timeout) -> float:
    """*timeout* in seconds, refusing one no deadline can be built from.

    A deadline computed from NaN is never reached -- every comparison with it
    is false -- and one from infinity never is by definition, so a call with a
    single silent rank would wait forever. Zero or less is fine: it asks for
    whatever has already arrived.
    """
    # 【译】把 *timeout* 转成秒，拒绝无法据以构造截止时间的值。
    #
    #   由NaN算出的截止时间永远到不了——与它的每次比较都是假——而由无穷大算出的按定义也到不了，所以只要有一个rank沉默，
    #   调用就会永远等下去。零或负数没问题：它表示"已经到了的就要，其余不等"。
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        seconds = math.nan
    if not math.isfinite(seconds):
        raise ValueError(
            f"collective_rpc timeout must be a finite number of seconds, "
            f"got {timeout!r}"
        )
    return seconds


# 引擎侧的时间预算：比调用方自己的等待时间短一点（最多短5秒或10%），让引擎的错误说明能在调用方放弃前送达
def engine_budget(timeout: float) -> float:
    """How long an engine may spend on a call its caller waits *timeout* for.

    Short of the caller's own wait, so the engine's account of what went wrong
    -- which rank timed out, which one died -- reaches a caller still waiting,
    instead of arriving after it has given up as a reply nobody takes.
    """
    # 【译】对于调用方等待 *timeout* 的一次调用，引擎可以花多长时间。
    #
    #   比调用方自己的等待时间短，这样引擎对出错原因的说明——哪个rank超时、哪个rank死了——能送达仍在等待的调用方，
    #   而不是在它放弃之后才到达、成为没人收的回复。
    # 例：timeout=30 → 30 - min(5, 3) = 27秒；timeout=300 → 295秒
    return max(0.0, timeout - min(5.0, 0.1 * timeout))


_PLAIN_SCALARS = (type(None), bool, int, float, str, bytes)
_PLAIN_DEPTH_LIMIT = 64


# 判断一个返回值是否只由内置纯数据组成（None/bool/int/float/str/bytes 及其list/tuple/dict嵌套，深度不超过64）
def is_plain_rpc_value(value) -> bool:
    """Whether *value* is builtin data that any process can unpickle.

    A generic RPC's return is unpickled in EngineCore and again in the API
    server. A class only the worker imports ended the API server's output
    thread for that DP rank, and a CUDA tensor was rebuilt on its device in
    each process, with a HIP context of its own. Exact types: a subclass of a
    builtin needs its own import as well.
    """
    # 【译】*value* 是否是任何进程都能unpickle的内置数据。
    #
    #   一次通用RPC的返回值要在EngineCore里unpickle一次，在API server里再unpickle一次。一个只有worker才导入的类
    #   曾结束了API server中那个DP rank的输出线程，而一个CUDA张量会在每个进程里都在它的设备上被重建，各带一个自己的HIP上下文。
    #   要求精确类型：内置类型的子类也需要它自己的import。

    def plain(item, depth: int) -> bool:
        if depth > _PLAIN_DEPTH_LIMIT:
            return False
        kind = type(item)
        if kind in _PLAIN_SCALARS:
            return True
        if kind in (list, tuple):
            return all(plain(element, depth + 1) for element in item)
        if kind is dict:
            return all(
                plain(key, depth + 1) and plain(element, depth + 1)
                for key, element in item.items()
            )
        return False

    return plain(value, 0)


# 一次集合RPC调用的参数包：request_id、位置参数、关键字参数、是否在调用后做全rank barrier
@dataclass(frozen=True)
class RpcPayload:
    """One generic collective-RPC call: kwargs plus an explicit barrier request.

    The plain worker wire format is ``(func_name, *args)``, which cannot carry
    kwargs. Rather than change that format and every existing ``call_func``
    caller, a generic call sends exactly one positional argument -- this -- and
    the worker recognises it by type.

    ``request_id`` lets the manager match replies to the call that produced
    them. The older KV channel matches by count instead, which is why it can
    only ever have one aggregation outstanding.
    """

    # 【译】一次通用集合RPC调用：kwargs，再加一个显式的barrier请求。
    #
    #   普通的worker线路格式是 (func_name, *args)，无法携带kwargs。与其改变这个格式和每一个现有的 call_func 调用方，
    #   不如让通用调用恰好只发送一个位置参数——就是这个对象——由worker按类型识别它。
    #
    #   request_id 让manager能把回复与产生它的调用配对。较旧的KV通道则是按数量配对，这就是为什么它永远只能有一个聚合在途。

    request_id: str
    args: tuple = ()
    kwargs: dict | None = None
    barrier: bool = False

    def call_kwargs(self) -> dict:
        return self.kwargs or {}


# 一个rank的回复：request_id、tp_rank，以及value或error二者之一；ok表示成功
@dataclass(frozen=True)
class RpcResult:
    """A generic-RPC reply, so a ``None`` return still reaches the caller.

    ``busy_loop`` only forwards non-``None`` worker returns, and
    ``call_func(wait_out=True)`` blocks on an untimed queue get, so a method
    that legitimately returns ``None`` deadlocks the caller. Wrapping every
    outcome -- including failures -- means the generic path always answers.
    """

    # 【译】一个通用RPC回复，这样一个返回 None 的调用也能到达调用方。
    #
    #   busy_loop 只转发非 None 的worker返回值，而 call_func(wait_out=True) 阻塞在一个没有超时的queue get上，
    #   所以一个合法地返回 None 的方法会让调用方死锁。把每一种结果——包括失败——都包装起来，意味着通用路径总会回复。

    request_id: str
    tp_rank: int
    value: object = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# CoreManager侧：按request_id把各DP引擎的回复交给正在等待的那个调用方（每个调用一个私有queue）
class RpcResponseRouter:
    """Deliver DP-engine replies to whichever caller is waiting for them.

    ``broadcast_utility_command_sync`` reads a fixed number of replies off one
    shared queue, so it matches by count: two overlapping callers take each
    other's replies, and a late reply from an abandoned call is handed to the
    next caller as its own. Upstream already recorded that biting the old
    pull-based metrics.

    Registering a request id gives that call its own queue, so concurrent calls
    cannot collide and a reply nobody is waiting for is dropped rather than
    mis-delivered.
    """

    # 【译】把DP引擎的回复交给正在等它们的那个调用方。
    #
    #   broadcast_utility_command_sync 从一个共享queue里读固定数量的回复，所以它是按数量配对的：两个重叠的调用方会拿走
    #   彼此的回复，而一个被放弃的调用的迟到回复会被当成下一个调用方自己的回复交给它。上游已经记录过这在旧的拉取式指标上
    #   造成的问题。
    #
    #   登记一个request id会给那次调用一个自己的queue，这样并发的调用不会冲突，没人等待的回复会被丢弃，而不是被错误投递。
    # 解释：v0.1.8里broadcast_utility_command_sync也已经改为经由这个路由器按request_id取回复（见engine_core_mgr.py）

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    # 上下文管理器：登记request_id并返回它的私有queue；with块结束（包括超时）时自动注销
    @contextmanager
    def register(self, request_id: str) -> Iterator[queue.Queue]:
        own: queue.Queue = queue.Queue()
        with self._lock:
            if request_id in self._queues:
                raise RuntimeError(f"request id already in flight: {request_id}")
            self._queues[request_id] = own
        try:
            yield own
        finally:
            # Unregister even on the timeout path, so a late reply is dropped by
            # ``route`` rather than kept for a caller that has given up.
            # 【译】即使在超时路径上也要注销，这样迟到的回复会被 route 丢弃，而不是为一个已经放弃的调用方保留着。
            with self._lock:
                self._queues.pop(request_id, None)

    # 输出线程调用：把一条回复放进对应调用方的queue；没人等就返回False（由调用方决定丢弃）
    def route(self, request_id: str, item: object) -> bool:
        """Hand *item* to the waiter for *request_id*; False if nobody waits."""
        # 【译】把 *item* 交给 *request_id* 的等待者；没人在等则返回False。
        with self._lock:
            own = self._queues.get(request_id)
        if own is None:
            return False
        own.put_nowait(item)
        return True

    # 当前在途（已登记）的调用数
    def in_flight(self) -> int:
        with self._lock:
            return len(self._queues)
