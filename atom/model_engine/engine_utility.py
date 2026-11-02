# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import queue
import time
import uuid
from typing import ClassVar

from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_CMD,
    DISCARD_WEIGHT_SYNC_CMD,
    FINISH_WEIGHT_SYNC_CMD,
    RpcPayload,
    checked_timeout,
)
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

logger = logging.getLogger("atom")

# 本文件：EngineCore里的工具RPC（utility command）处理器 EngineUtilityHandler
#   什么是工具RPC：除"加请求"以外，外部对引擎的控制操作——统计查询、profile、中止请求、RL的权重更新/
#     显存释放与恢复、集合RPC——都打包成 {"cmd": 命令名, ...参数}，经CoreManager的控制socket发给每个EngineCore
#   在EngineCore进程里的流转：
#     输入线程收到UTILITY消息 → utility_queue.put((cmd, args))，并置 engine._has_pending_utility = True
#     → busy_loop每轮开头调用 process_queue → 按 _UTILITY_HANDLERS 找到 _handle_xxx 执行
#     → 需要回复的命令往output_queue放 ("UTILITY_RESPONSE", {...})，经输出socket回到CoreManager，
#       CoreManager按request_id把回复交给正在等它的调用方（collective_rpc.RpcResponseRouter）
#   为什么在busy_loop主线程里执行：这样对runner_mgr的调用（call_func等）都是串行的，不会和forward交错；
#     代价是一条慢命令执行期间，这个DP rank停止调度
#   另外：push_metrics / collect_metrics 由busy_loop定时调用，把指标快照主动推给CoreManager（METRICS消息）
#
# 命令一览：
#   get_mtp_stats / get_mtp_statistics      投机解码统计（前者只打日志、不回复）
#   get_cache_statistics                    前缀缓存统计
#   start_profile / stop_profile            torch profiler（性能剖析）
#   abort_request                           中止请求（不回复）
#   configure_hidden_states                 TorchSpec的隐藏状态提取
#   collective_rpc                          集合RPC（collective RPC）：在每个TP rank上调用一个白名单方法
#   【分支】RL rollout，见第10批：
#   release_memory / resume_memory / clear_kv_cache           显存释放、恢复，清空KV cache
#   update_weights / update_weights_shm / update_weights_ipc  权重热更新（直接传张量 / POSIX共享内存 / CUDA IPC）
#   discard_failed_weight_sync / finish_weight_sync / get_weight_sync_status  权重同步的放弃、提交、状态查询
#
# 权重同步的"围栏（fence）"协议：
#   收到任何 update_weights*：置fenced → busy_loop不再调度（EngineCore._is_idle_rl_weights_offloaded）
#   一次同步可以分成多个bucket（同一个sync_id，最后一个is_last=True）；任一步失败就记下原因，之后的bucket和FINISH都被拒绝
#   CoreManager确认所有DP engine都成功后发 finish_weight_sync → 解除围栏、恢复服务；失败时发 discard 清理各rank的暂存
# 对比nano-vllm：nano-vllm没有这一层（没有统计接口、profile、权重热更新）


# Commands whose senders never wait, so their handlers answer nothing and a
# synchronous caller could only wait out its timeout on them.
# 【译】这些命令的发送方从不等待，所以它们的处理函数什么也不回复，同步调用方在它们上面只能等到超时。
FIRE_AND_FORGET_UTILITY_CMDS = frozenset({"abort_request", "get_mtp_stats"})
# 权重更新类命令：触发上面说的围栏协议
WEIGHT_UPDATE_UTILITY_CMDS = frozenset(
    {"update_weights", "update_weights_shm", "update_weights_ipc"}
)


# 这条命令是否是一次权重同步的最后一步：直接的update_weights，或者is_last的bucket（缺省视为最后一个）
def completes_weight_sync(cmd: str, args: dict) -> bool:
    """Whether *cmd* is the update a sync ends on: direct, or a last bucket."""
    # 【译】*cmd* 是否是一次同步结束时的那个更新：直接更新，或者最后一个bucket。
    return cmd == "update_weights" or (
        cmd in WEIGHT_UPDATE_UTILITY_CMDS and bool(args.get("is_last", True))
    )


# For a direct weight update whose sender gives no deadline of its own;
# broadcast_utility_command_sync always does.
# 【译】用于发送方没有给出自己截止时间的直接权重更新；broadcast_utility_command_sync 总会给出。
_DIRECT_UPDATE_TIMEOUT_S = 300.0
# 失败后清理各rank暂存的时间上限
_FAILED_SYNC_CLEANUP_TIMEOUT_S = 1.0


# 算出本条命令还剩多少秒：优先用调用方给的绝对截止时间（墙钟），否则用timeout，都没有就用default
def _remaining_timeout(args: dict, default: float) -> float:
    """Command budget, preferring the caller's absolute wall-clock deadline."""
    # 【译】命令的时间预算，优先使用调用方的绝对墙钟截止时间。
    if args.get("deadline") is not None:
        deadline = checked_timeout(args["deadline"])
        return max(0.0, deadline - time.time())
    return checked_timeout(args.get("timeout", default))


# 包装EngineCore的output_queue：工具命令执行期间，给每条UTILITY_RESPONSE自动盖上当前命令的request_id
class _ReplyQueue:
    """The EngineCore output queue, as the utility handlers see it.

    Stamps every ``UTILITY_RESPONSE`` with the request id of the command it
    answers, so CoreManager matches replies to callers by id rather than by
    position, where a late or unasked-for reply became the next caller's.
    """

    # 【译】EngineCore的输出queue，按工具处理函数看到的样子。
    #
    #   给每条 UTILITY_RESPONSE 盖上它所回复的命令的request id，这样CoreManager按id而不是按位置把回复与调用方配对——
    #   按位置时，一条迟到的或没人要的回复会变成下一个调用方的回复。

    def __init__(self, output_queue):
        self._queue = output_queue
        # 当前正在执行的命令的request_id；由_execute_utility_command在执行前后设置/清空
        self.request_id = None

    # 与queue.Queue.put_nowait同名，处理函数可以把它当普通queue用
    def put_nowait(self, item) -> None:
        if self.request_id is not None and item[0] == "UTILITY_RESPONSE":
            item = ("UTILITY_RESPONSE", {**item[1], "request_id": self.request_id})
        self._queue.put_nowait(item)


# =====================================================================================
# EngineUtilityHandler：所有工具命令的集中处理器（每个EngineCore一个）
# =====================================================================================
# 创建：EngineCore.__init__（在Scheduler之后）；PD分离的两种EngineCore会把它的scheduler换成自己真正运行的那个
# 调用方：EngineCore / DPEngineCoreProc / PP各stage（pp_engine_core.py）的busy_loop，每轮调用process_queue，定时调用push_metrics
class EngineUtilityHandler:
    """Centralised handler for all utility commands dispatched by EngineCore.

    Covers weight management, memory lifecycle, profiling, MTP statistics,
    and TorchSpec hidden-state extraction.  Every command is registered in
    ``_UTILITY_HANDLERS`` and executed in the main busy-loop thread so that
    ``runner_mgr.call_func`` calls are serialized.

    Parameters
    ----------
    runner_mgr : AsyncIOProcManager
        The model-runner process manager used to execute ``call_func``.
    output_queue : queue.Queue
        The EngineCore output queue for pushing ``UTILITY_RESPONSE`` messages
        back to ``CoreManager``.
    label : str, optional
        Label used in log messages (default ``"Engine Core"``).
    scheduler : Scheduler, optional
        The scheduler instance, needed by MTP statistics handlers.
    """

    # 【译】由EngineCore分派的所有工具命令的集中处理器。
    #
    #   涵盖权重管理、显存生命周期、profiling、MTP统计以及TorchSpec隐藏状态提取。每个命令都登记在 _UTILITY_HANDLERS 里，
    #   并在busy loop主线程中执行，这样 runner_mgr.call_func 的调用是串行的。
    #
    #   参数
    #   ----------
    #   runner_mgr：AsyncIOProcManager类型
    #       用来执行 call_func 的model runner进程管理器。
    #   output_queue：queue.Queue类型
    #       EngineCore的输出queue，用于把 UTILITY_RESPONSE 消息推回给 CoreManager。
    #   label：str类型，可选
    #       日志中使用的标签（默认 "Engine Core"）。
    #   scheduler：Scheduler类型，可选
    #       调度器实例，MTP统计的处理函数需要它。

    # Utility command name  ->  handler method name
    # 【译】工具命令名  ->  处理方法名
    # 查表分派：命令名字符串 → 方法名字符串，执行时用getattr(self, 方法名)取到方法
    _UTILITY_HANDLERS: ClassVar[dict[str, str]] = {
        "update_weights": "_handle_update_weights",
        "update_weights_shm": "_handle_update_weights_shm",
        "update_weights_ipc": "_handle_update_weights_ipc",
        DISCARD_WEIGHT_SYNC_CMD: "_handle_discard_failed_weight_sync",
        FINISH_WEIGHT_SYNC_CMD: "_handle_finish_weight_sync",
        "get_weight_sync_status": "_handle_get_weight_sync_status",
        "release_memory": "_handle_release_memory",
        "resume_memory": "_handle_resume_memory",
        "clear_kv_cache": "_handle_clear_kv_cache",
        "configure_hidden_states": "_handle_configure_hidden_states",
        "start_profile": "_handle_start_profile",
        "stop_profile": "_handle_stop_profile",
        "get_mtp_stats": "_handle_get_mtp_stats",
        "get_mtp_statistics": "_handle_get_mtp_statistics",
        "get_cache_statistics": "_handle_get_cache_statistics",
        "abort_request": "_handle_abort_request",
        COLLECTIVE_RPC_CMD: "_handle_collective_rpc",
    }

    # ---------- 成员变量 ----------
    #   runner_mgr        AsyncIOProcManager，下发worker操作
    #   output_queue      包装后的EngineCore输出queue（自动盖request_id）
    #   scheduler         统计类命令读它的engine_stats和block_manager；PD分离decode端起初为None
    #   _weight_sync_*    权重同步状态：是否围栏中、是否进行中、失败原因、当前sync_id、等待提交的请求id
    def __init__(
        self, runner_mgr, output_queue, label: str = "Engine Core", scheduler=None
    ):
        self.runner_mgr = runner_mgr
        self.output_queue = _ReplyQueue(output_queue)
        self.label = label
        self.scheduler = scheduler
        self._weight_sync_fenced = False
        self._weight_sync_in_progress = False
        # Why the current sync can no longer complete. Its later updates and
        # its FINISH are refused while this is set.
        # 【译】当前这次同步为什么已经无法完成。设置期间，它之后的更新和它的FINISH都会被拒绝。
        self._weight_sync_failure: str | None = None
        # The sync the last weight update belonged to; a new id starts a new one.
        # 【译】上一次权重更新所属的同步；一个新的id开始一次新的同步。
        self._sync_id: str | None = None
        # Request id of the complete update this engine applied, the only one
        # a FINISH may commit.
        # 【译】本引擎已经应用的那次完整更新的请求id，FINISH只能提交它。
        self._awaiting_finish: str | None = None

    # 处理所有待执行的工具命令，并维护引擎的"睡眠/围栏"标志
    # 输入：utility_queue（EngineCore输入线程放入的(cmd, args)）、engine（EngineCore本身，读写它的标志位）
    # 调用方：EngineCore.busy_loop每轮开头
    def process_queue(self, utility_queue, engine):
        """Drain *utility_queue* and execute each command.

        ``engine._has_pending_utility`` is cleared before the drain, not after
        it: the input thread enqueues and then sets it, so a command landing
        between the last empty get and a later clear lost its flag and sat
        unseen until another arrived. For a FINISH that cost a whole sync.

        Memory residency stays on ``engine._is_rl_weights_offloaded``; weight
        synchronization safety uses ``engine._weight_sync_fenced``. The busy
        loop skips model execution while either is true.
        """
        # 【译】取空 *utility_queue* 并执行其中每一条命令。
        #
        #   engine._has_pending_utility 在取之前清掉，而不是之后：输入线程是先入队再置位，所以如果在最后一次空的get和之后的
        #   清零之间来了一条命令，它的标志就丢了，这条命令会一直没人看见，直到又来一条。对一个FINISH来说，这会耗掉整整一次同步。
        #
        #   显存驻留状态仍记在 engine._is_rl_weights_offloaded 上；权重同步的安全性用 engine._weight_sync_fenced。
        #   只要其中任一为真，busy loop就跳过模型执行。
        # 快速路径：没有待处理命令时直接返回，避免每轮都去查queue
        if not engine._has_pending_utility:
            return
        engine._has_pending_utility = False

        while True:
            try:
                cmd, args = utility_queue.get_nowait()
                if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                    # Local success is not global success: another DP engine
                    # can still fail this bucket. Stay fenced until CoreManager
                    # explicitly finishes a complete sync on every engine.
                    # 【译】本地成功不等于全局成功：另一个DP引擎仍可能让这个bucket失败。在CoreManager对每个引擎明确地完成一次完整同步之前，
                    #   保持围栏。
                    self._weight_sync_fenced = True
                    self._awaiting_finish = None
                    engine._weight_sync_fenced = True
                    # sync_id变了：开始一次新的同步（若上一次停在半路，先清理各rank的暂存）
                    if args.get("sync_id") != self._sync_id:
                        self._begin_weight_sync(args.get("sync_id"))
                    self._weight_sync_in_progress = True
                # 真正执行命令（查表分派）
                reply = self._execute_utility_command(cmd, args)
                # Track sleep/wake transitions
                # 【译】跟踪睡眠/唤醒的状态切换
                if cmd == "release_memory":
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if "weights" in tags:
                        # Some ranks may already have released their weights
                        # before a peer reports failure. Conservatively stop
                        # serving until an all-rank resume succeeds.
                        # 【译】在某个对端报告失败之前，有些rank可能已经释放了权重。保守起见停止服务，直到一次全rank的resume成功。
                        engine._is_rl_weights_offloaded = True
                        if isinstance(reply, dict) and reply.get("error"):
                            logger.error(
                                f"{self.label}: partial weight release; engine "
                                f"stays idle until resume succeeds: {reply['error']}"
                            )
                        else:
                            logger.info(f"{self.label}: engine entered sleep mode")
                # resume成功且包含权重：退出睡眠（但若仍在围栏中，要等完整同步成功才恢复服务）
                elif cmd == "resume_memory":
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if "weights" in tags and not (
                        isinstance(reply, dict) and reply.get("error")
                    ):
                        engine._is_rl_weights_offloaded = False
                        if self._weight_sync_fenced:
                            logger.warning(
                                f"{self.label}: weights resumed, but serving "
                                f"stays fenced until a complete sync succeeds"
                            )
                        else:
                            logger.info(f"{self.label}: engine exited sleep mode")
                # 权重更新：失败就记下原因并标记"权重不一致"；成功且是最后一步就记下等待提交的请求id
                elif cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                    failed = isinstance(reply, dict) and bool(reply.get("error"))
                    if failed:
                        self._weight_sync_in_progress = False
                        self._weight_sync_failure = self._weight_sync_failure or str(
                            reply["error"]
                        )
                        engine._rl_weights_inconsistent = True
                        logger.error(
                            f"{self.label}: weight update failed; serving stays "
                            f"fenced until a complete sync succeeds: {reply['error']}"
                        )
                    elif completes_weight_sync(cmd, args):
                        self._awaiting_finish = args.get("request_id")
                # 放弃本次同步：保持围栏、标记权重不一致
                elif cmd == DISCARD_WEIGHT_SYNC_CMD:
                    self._weight_sync_fenced = True
                    self._weight_sync_in_progress = False
                    self._awaiting_finish = None
                    self._weight_sync_failure = self._weight_sync_failure or str(
                        args.get("error") or args.get("failed_cmd") or "aborted"
                    )
                    engine._rl_weights_inconsistent = True
                    engine._weight_sync_fenced = True
                    logger.error(
                        f"{self.label}: global weight sync aborted; serving stays "
                        f"fenced until a complete sync succeeds"
                    )
                # 提交：成功则清除所有围栏/睡眠标志，恢复服务
                elif cmd == FINISH_WEIGHT_SYNC_CMD:
                    if isinstance(reply, dict) and reply.get("error"):
                        logger.error(
                            f"{self.label}: weight sync not committed; serving "
                            f"stays fenced: {reply['error']}"
                        )
                    else:
                        self._weight_sync_fenced = False
                        self._weight_sync_in_progress = False
                        self._weight_sync_failure = None
                        self._awaiting_finish = None
                        engine._rl_weights_inconsistent = False
                        engine._weight_sync_fenced = False
                        engine._is_rl_weights_offloaded = False
                        logger.info(
                            f"{self.label}: globally complete weight sync "
                            f"committed; engine exited sleep mode"
                        )
            except queue.Empty:
                break

    # 执行一条命令：查表找处理函数，执行期间给回复盖上request_id；未知命令也回复一个错误
    def _execute_utility_command(self, cmd: str, args: dict) -> dict | None:
        """Run *cmd*'s handler and return what it returned: the weight
        updates hand back the reply they sent."""
        # 【译】运行 *cmd* 的处理函数并返回它的返回值：权重更新类返回它们发出的那条回复。
        import time as _time

        log = logger.info
        log(f"{self.label}: executing utility command: {cmd}")
        t0 = _time.monotonic()

        reply = None
        handler_name = self._UTILITY_HANDLERS.get(cmd)
        # 让这期间放进output_queue的UTILITY_RESPONSE都带上本命令的request_id
        self.output_queue.request_id = (
            args.get("request_id") if isinstance(args, dict) else None
        )
        try:
            if handler_name:
                handler = getattr(self, handler_name)
                try:
                    reply = handler(args)
                except Exception as exc:
                    # Still fatal to the engine, as before; but a synchronous
                    # caller now hears why instead of waiting out its timeout.
                    # 【译】仍然对引擎是致命的，和以前一样；但同步调用方现在能听到原因，而不是一直等到超时。
                    if cmd not in FIRE_AND_FORGET_UTILITY_CMDS:
                        self.output_queue.put_nowait(
                            ("UTILITY_RESPONSE", self._error_reply(cmd, exc))
                        )
                    raise
            else:
                # Answer, do not just log: a synchronous caller would otherwise
                # wait out its timeout and learn only that, not the misspelling.
                # 【译】要回复，而不只是记日志：否则同步调用方会一直等到超时，而且只知道超时，不知道是拼错了。
                logger.warning(f"{self.label}: Unknown utility command: {cmd}")
                self.output_queue.put_nowait(
                    (
                        "UTILITY_RESPONSE",
                        {"cmd": cmd, "error": f"unknown utility command {cmd!r}"},
                    )
                )
        finally:
            self.output_queue.request_id = None

        elapsed = _time.monotonic() - t0
        log(f"{self.label}: utility command '{cmd}' finished in {elapsed:.2f}s")
        return reply

    @staticmethod
    def _error_reply(cmd: str, exc: Exception) -> dict:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}

    # 【分支】集合RPC在EngineCore一侧的处理：校验参数 → runner_mgr.collective_rpc → 把各rank结果汇成一条回复
    #   调用链见async_proc.py的collective_rpc；外部使用者是RL rollout（第10批）
    def _handle_collective_rpc(self, args: dict):
        """Invoke an arbitrary ModelRunner method on every TP rank.

        Runs in the EngineCore busy loop, so this DP rank stops scheduling for
        the call's duration. That is wanted for a weight swap, but it does mean
        a caller passing a long timeout is deliberately stalling generation.
        """
        # 【译】在每个TP rank上调用任意一个ModelRunner方法。
        #
        #   在EngineCore的busy loop里运行，所以在调用期间这个DP rank停止调度。这对权重切换是想要的效果，但也意味着
        #   一个传入很长超时的调用方是在有意地让生成停顿。
        # 解释：所谓"任意"受白名单限制：manager的公开collective_rpc只放行COLLECTIVE_RPC_METHODS
        method = args.get("method")
        request_id = args.get("request_id")
        # Checked here, before the broadcast: every TP worker resolves the name
        # with getattr, which raises TypeError on anything but a string, and the
        # manager's output thread routes replies with the id as a dict key.
        # 【译】在广播之前在这里检查：每个TP worker都用getattr解析方法名，对非字符串会抛TypeError；
        #   而manager的输出线程用这个id作为dict的键来路由回复。
        if (
            not isinstance(method, str)
            or not method
            or not isinstance(request_id, str)
            or not request_id
        ):
            self.output_queue.put_nowait(
                (
                    "UTILITY_RESPONSE",
                    {
                        "cmd": COLLECTIVE_RPC_CMD,
                        "request_id": request_id,
                        "error": "collective_rpc needs a method name and a request "
                        "id, both non-empty strings",
                    },
                )
            )
            return

        try:
            payload = RpcPayload(
                request_id=request_id,
                args=tuple(args.get("args", ())),
                kwargs=dict(args.get("kwargs") or {}),
                barrier=bool(args.get("barrier", False)),
            )
            replies = self.runner_mgr.collective_rpc(
                method, payload, timeout=_remaining_timeout(args, 300.0)
            )
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the loop
            # Raising here would kill the EngineCore busy loop and take the
            # engine down with it; the caller gets the reason instead.
            # 【译】在这里抛异常会杀死EngineCore的busy loop，连带整个引擎；改为把原因告诉调用方。
            self.output_queue.put_nowait(
                (
                    "UTILITY_RESPONSE",
                    {
                        "cmd": COLLECTIVE_RPC_CMD,
                        "request_id": request_id,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return

        failures = [r for r in replies if not r.ok]
        logger.info(
            f"{self.label}: collective_rpc {method} ranks={len(replies)} "
            f"failed={len(failures)}"
        )
        self.output_queue.put_nowait(
            (
                "UTILITY_RESPONSE",
                {
                    "cmd": COLLECTIVE_RPC_CMD,
                    "request_id": request_id,
                    "method": method,
                    # Includes PCP: this is the number of worker processes
                    # behind this EngineCore, not logical tensor parallelism.
                    # 【译】包含PCP：这是这个EngineCore背后的worker进程数，而不是逻辑上的张量并行度。
                    "worker_world_size": self.runner_mgr.proc_num,
                    # Kept for wire compatibility; prefer worker_world_size.
                    # 【译】为了线路兼容而保留；请优先使用 worker_world_size。
                    "tp_world_size": self.runner_mgr.proc_num,
                    "results": [
                        {
                            "tp_rank": r.tp_rank,
                            "value": r.value,
                            "error": r.error,
                        }
                        for r in replies
                    ],
                },
            )
        )

    # ---------- 以下是RL权重同步与显存管理（【分支】第10批） ----------
    def _begin_weight_sync(self, sync_id: str | None) -> None:
        """Start *sync_id* from a clean slate.

        Its id is all that marks a sync's first update. Without one, the rest
        of a failed sync's buckets passed for a new sync and committed.
        """
        # 【译】从干净状态开始 *sync_id*。
        #
        #   一次同步的第一个更新只靠它的id来标记。没有id的话，一次失败同步剩下的bucket会被当成一次新同步而被提交。
        abandoned = self._weight_sync_in_progress
        self._sync_id = sync_id
        self._weight_sync_failure = None
        if abandoned:
            # The previous sync stopped between buckets and was never aborted;
            # its packed shards and IPC mapping must not carry into this one.
            # 【译】上一次同步停在两个bucket之间，从未被放弃；它打包的分片和IPC映射不能带进这一次。
            reply = self._discard_failed_update_on_every_rank(
                "abandoned weight sync", _FAILED_SYNC_CLEANUP_TIMEOUT_S
            )
            if reply.get("error"):
                self._weight_sync_failure = (
                    f"the abandoned sync's scratch was not cleared: {reply['error']}"
                )

    # 在每个TP rank上执行一次直接权重更新，汇总成一条回复（任一rank失败、或各rank更新的参数个数不同，都算失败）
    def _update_on_every_rank(
        self, cmd: str, method: str, args: dict, *call_args, barrier: bool = False
    ) -> dict:
        """Run a direct weight update on every TP rank; the reply for all.

        ``call_func`` returns rank 0's result alone, so a rank that rejected a
        tensor still reported success and the caller went on with a partly
        updated model -- if that rank's raise had not ended its worker. Through
        the generic path every rank answers on its own channel, a failure is
        caught where it happens, and success means every rank succeeded.

        Every rank is sent the same tensors, so every rank should update the
        same number of parameters: one that updated fewer skipped what its
        peers wrote, and the shards no longer belong to one model.

        *barrier* holds each rank at the worker barrier until every rank has
        finished, so none moves on while another still reads the caller's
        shared buffer.
        """
        # 【译】在每个TP rank上执行一次直接权重更新；返回代表所有rank的那条回复。
        #
        #   call_func 只返回rank 0的结果，所以一个拒绝了某个张量的rank仍然报告成功，调用方就带着一个部分更新的模型继续下去——
        #   前提是那个rank的异常没有结束它的worker。走通用路径时，每个rank都在自己的通道上回复，失败在发生处就被捕获，
        #   而成功意味着每个rank都成功了。
        #
        #   每个rank收到的是同样的张量，所以每个rank应当更新同样数量的参数：更新得更少的那个跳过了对端写入的东西，
        #   这些分片就不再属于同一个模型。
        #
        #   *barrier* 让每个rank在worker barrier处等到所有rank都完成，这样在另一个rank还在读调用方的共享缓冲区时，
        #   没有rank会继续往下走。
        if self._weight_sync_failure is not None:
            return {
                "cmd": cmd,
                "error": f"weight sync {self._sync_id!r} already failed, so "
                f"this update cannot complete it: {self._weight_sync_failure}",
            }
        payload = RpcPayload(
            request_id=f"{cmd}-{uuid.uuid4().hex}", args=call_args, barrier=barrier
        )
        timeout = _remaining_timeout(args, _DIRECT_UPDATE_TIMEOUT_S)
        try:
            replies = self.runner_mgr.utility_rpc(method, payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - answered, never raised at the loop
            self._discard_failed_update_on_every_rank(cmd, timeout)
            return self._error_reply(cmd, exc)
        failed = [r for r in replies if not r.ok]
        if failed:
            error = "; ".join(f"TP rank {r.tp_rank}: {r.error}" for r in failed)
        elif any(r.value != replies[0].value for r in replies[1:]):
            error = "TP ranks updated different numbers of parameters: " + ", ".join(
                f"rank {r.tp_rank} updated {r.value}" for r in replies
            )
        else:
            result = replies[0].value
            logger.info(
                f"{self.label}: {cmd} completed on every TP rank, updated={result}"
            )
            return {"cmd": cmd, "result": result}
        self._discard_failed_update_on_every_rank(cmd, timeout)
        logger.error(f"{self.label}: {cmd} failed: {error}")
        return {"cmd": cmd, "error": error}

    # 失败后让每个rank丢弃本次更新的暂存（超时不超过_FAILED_SYNC_CLEANUP_TIMEOUT_S）
    def _discard_failed_update_on_every_rank(self, cmd: str, timeout: float) -> dict:
        """Keep no rank's scratch after this engine's update failed."""
        # 【译】本引擎的更新失败后，不保留任何rank的暂存。
        timeout = min(
            _FAILED_SYNC_CLEANUP_TIMEOUT_S,
            max(0.0, checked_timeout(timeout)),
        )
        payload = RpcPayload(request_id=f"{cmd}-discard-{uuid.uuid4().hex}")
        try:
            replies = self.runner_mgr.utility_rpc(
                DISCARD_WEIGHT_SYNC_CMD, payload, timeout=timeout
            )
            failed = [r for r in replies if not r.ok]
            if failed:
                error = (
                    f"{cmd} cleanup failed on {len(failed)}/{len(replies)} "
                    f"TP rank(s)"
                )
                logger.error(f"{self.label}: {error}")
                return {"cmd": DISCARD_WEIGHT_SYNC_CMD, "error": error}
        except Exception as exc:
            logger.exception(f"{self.label}: {cmd} cleanup could not be broadcast")
            return self._error_reply(DISCARD_WEIGHT_SYNC_CMD, exc)
        return {"cmd": DISCARD_WEIGHT_SYNC_CMD, "result": True}

    def _handle_discard_failed_weight_sync(self, args: dict) -> dict:
        """Clear abandoned update scratch on every TP rank of this engine."""
        # 【译】清除本引擎每个TP rank上被放弃的更新暂存。
        reply = self._discard_failed_update_on_every_rank(
            args.get("failed_cmd", "weight update"),
            _FAILED_SYNC_CLEANUP_TIMEOUT_S,
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_finish_weight_sync(self, args: dict) -> dict:
        """Commit the complete update this engine applied, once CoreManager
        has heard every DP engine apply it."""
        # 【译】在CoreManager听到每个DP引擎都应用了它之后，提交本引擎已应用的那次完整更新。
        completed = args.get("completed_request_id")
        sync_id = args.get("sync_id")
        if sync_id is None or sync_id != self._sync_id:
            error = (
                f"weight sync {sync_id!r} is not the sync this engine is in "
                f"({self._sync_id!r})"
            )
        elif self._weight_sync_failure is not None:
            error = f"weight sync {sync_id!r} failed: {self._weight_sync_failure}"
        elif completed is None or completed != self._awaiting_finish:
            error = (
                f"no complete weight update {completed!r} succeeded on this "
                f"engine since its last sync ended"
            )
        else:
            error = None
        if error is None:
            reply = {"cmd": FINISH_WEIGHT_SYNC_CMD, "result": True}
        else:
            reply = {"cmd": FINISH_WEIGHT_SYNC_CMD, "error": error}
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_get_weight_sync_status(self, args: dict) -> dict:
        """Expose a stuck fence instead of making generation merely hang."""
        # 【译】把卡住的围栏暴露出来，而不是让生成只是挂着。
        reply = {
            "cmd": "get_weight_sync_status",
            "result": {
                "fenced": self._weight_sync_fenced,
                "in_progress": self._weight_sync_in_progress,
                "failure": self._weight_sync_failure,
            },
        }
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights(self, args: dict) -> dict:
        """Handle direct weight update command."""
        # 【译】处理直接权重更新命令。
        reply = self._update_on_every_rank(
            "update_weights",
            "update_weights",
            args,
            args.get("named_tensors", []),
            args.get("flush_cache", True),
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights_shm(self, args: dict) -> dict:
        """Handle shared-memory weight update command.

        Only lightweight metadata (shm_name, bucket_meta) travels through the
        control path.  The actual tensor data resides in POSIX shared memory and
        is read directly by each ModelRunner process.

        After all ModelRunners finish, a UTILITY_RESPONSE is pushed onto the
        output_queue so that the caller (LLMEngine) can synchronise.

        After completion, a ``UTILITY_RESPONSE`` is pushed so the caller
        (LLMEngine) can synchronise.
        """
        # 【译】处理共享内存权重更新命令。
        #
        #   只有轻量的元数据（shm_name、bucket_meta）走控制路径。真正的张量数据放在POSIX共享内存里，由每个ModelRunner进程
        #   直接读取。
        #
        #   所有ModelRunner完成后，往output_queue推一条 UTILITY_RESPONSE，让调用方（LLMEngine）可以同步。
        #
        #   完成后推一条 UTILITY_RESPONSE，让调用方（LLMEngine）可以同步。
        # 解释：最后两段是重复的；回复经CoreManager送给发起命令的那一方（RL场景下是rollout引擎，见第10批）
        reply = self._update_on_every_rank(
            "update_weights_shm",
            "update_weights_from_shm",
            args,
            args.get("shm_name", ""),
            args.get("bucket_meta", {}),
            args.get("is_last", True),
            barrier=True,
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights_ipc(self, args: dict) -> dict:
        """Handle CUDA IPC weight update command.

        The caller (LLMEngine) sends a CUDA IPC handle pointing to a GPU
        buffer that already contains the weight data. Each ModelRunner
        sub-process uses ``rebuild_ipc_handle()`` to map the same GPU memory
        and reads weights directly — no CPU round-trip.

        When ``ipc_handles`` (per-GPU dict) is present, each ModelRunner
        opens only its own GPU's handle — always same-GPU IPC, safe on ROCm.
        """
        # 【译】处理CUDA IPC权重更新命令。
        #
        #   调用方（LLMEngine）发来一个CUDA IPC句柄，指向一块已经装好权重数据的GPU缓冲区。每个ModelRunner子进程用
        #   rebuild_ipc_handle() 映射同一块GPU显存，直接读取权重——没有CPU往返。
        #
        #   当有 ipc_handles（按GPU的dict）时，每个ModelRunner只打开它自己那块GPU的句柄——始终是同一GPU上的IPC，在ROCm上是安全的。
        reply = self._update_on_every_rank(
            "update_weights_ipc",
            "update_weights_from_ipc",
            args,
            args.get("ipc_handle"),
            args.get("bucket_meta", {}),
            args.get("is_last", True),
            args.get("ipc_handles"),
            barrier=True,
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    # 在每个TP rank上执行一个EngineCore管理的worker操作（经utility_rpc），任一rank失败就汇总成错误
    def _utility_on_every_rank(
        self, cmd: str, method: str, *call_args, timeout: float = 300.0
    ) -> dict:
        """Run an EngineCore-owned worker operation and hear every TP rank."""
        # 【译】执行一个由EngineCore拥有的worker操作，并听取每个TP rank的结果。
        payload = RpcPayload(
            request_id=f"{cmd}-{uuid.uuid4().hex}",
            args=call_args,
        )
        try:
            replies = self.runner_mgr.utility_rpc(method, payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - returned as the utility error
            return self._error_reply(cmd, exc)
        failed = [reply for reply in replies if not reply.ok]
        if failed:
            return {
                "cmd": cmd,
                "error": "; ".join(
                    f"TP rank {reply.tp_rank}: {reply.error}" for reply in failed
                ),
            }
        return {"cmd": cmd, "result": replies[0].value}

    def _handle_release_memory(self, args: dict) -> dict:
        """Handle memory release command (sleep mode)."""
        # 【译】处理显存释放命令（睡眠模式）。
        tags = args.get("tags", ["weights", "kv_cache"])
        # tags缺省时同时释放权重和KV cache
        reply = self._utility_on_every_rank(
            "release_memory",
            "release_memory",
            tags,
            timeout=_remaining_timeout(args, 300.0),
        )
        logger.info(f"{self.label}: release_memory completed, tags={tags}")
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_resume_memory(self, args: dict) -> dict:
        """Handle memory resume command (wake up mode).

        Allowed while the weight sync is fenced: a released parameter gets its
        storage back only here, so refusing left a failed sync nothing to write
        its retry into. The fence, not residency, is what keeps serving off.
        """
        # 【译】处理显存恢复命令（唤醒模式）。
        #
        #   在权重同步处于围栏中时也允许：被释放的参数只有在这里才能拿回它的存储，所以拒绝它会让一次失败的同步没有地方
        #   写入它的重试。保持服务关闭的是围栏，而不是驻留状态。
        tags = args.get("tags", ["weights", "kv_cache"])
        reply = self._utility_on_every_rank(
            "resume_memory",
            "resume_memory",
            tags,
            timeout=_remaining_timeout(args, 300.0),
        )
        logger.info(f"{self.label}: resume_memory completed, tags={tags}")
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_clear_kv_cache(self, args: dict) -> dict:
        """Handle KV cache clear command."""
        # 【译】处理KV cache清空命令。
        reply = self._utility_on_every_rank(
            "clear_kv_cache",
            "clear_kv_cache",
            timeout=_remaining_timeout(args, 300.0),
        )
        logger.info(f"{self.label}: KV cache cleared")
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_abort_request(self, args: dict):
        """Cancel queued work promptly; running forwards finish via postprocess."""
        # 【译】及时取消排队中的工作；正在运行的forward会经由postprocess结束。
        # 调用方：CoreManager.abort_request（客户端断开连接时，广播到每个DP rank，只有持有该请求的那个起作用）；不回复
        req_id = args.get("req_id") if isinstance(args, dict) else None
        if req_id is None or self.scheduler is None:
            return
        abort = getattr(self.scheduler, "abort_request", None)
        # 优先用调度器自己的abort_request；没有的话退回手动把序列状态设为ABORTED
        if callable(abort):
            found = abort(req_id)
        else:
            found = False
            for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
                if seq.id == req_id:
                    seq.status = SequenceStatus.ABORTED
                    found = True
        logger.info(f"{self.label}: abort_request req_id={req_id} found={found}")

    def _handle_configure_hidden_states(self, args: dict):
        """Configure hidden states extraction on all model runners (TorchSpec)."""
        # 【译】在所有model runner上配置隐藏状态提取（TorchSpec）。
        # 注意这里用的是call_func(wait_out=True)，只拿到rank 0的结果
        aux_layer_ids = args.get("aux_layer_ids", [])
        mooncake_config = args.get("mooncake_config", {})
        result = self.runner_mgr.call_func(
            "configure_hidden_states", aux_layer_ids, mooncake_config, wait_out=True
        )
        logger.info(
            f"{self.label}: configure_hidden_states completed, "
            f"aux_layers={aux_layer_ids}"
        )
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "configure_hidden_states", "result": result})
        )

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------
    # 【译】Profiler

    # 开始profile：让ModelRunner启动torch profiler（只等rank 0的回复），并让调度器开始输出详细的逐步统计
    def _handle_start_profile(self, args: dict):
        result = self.runner_mgr.call_func("start_profiler", wait_out=True)
        # Flip the scheduler flag so per-iteration detailed aggregates
        # (compute_detailed_aggregates) are emitted while profiling is active.
        # 【译】翻转调度器的标志，使profiling进行期间输出逐迭代的详细统计（compute_detailed_aggregates）。
        if self.scheduler is not None:
            self.scheduler.profile_active = True
        logger.info(f"{self.label}: profiler started")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "start_profile", "result": result})
        )

    # 停止profile：ModelRunner停止profiler并写出trace，结果随回复返回（LLMEngine.stop_profile收集）
    def _handle_stop_profile(self, args: dict):
        logger.info(f"{self.label}: stopping profiler...")
        result = self.runner_mgr.call_func("stop_profiler", wait_out=True)
        if self.scheduler is not None:
            self.scheduler.profile_active = False
        logger.info(f"{self.label}: profiler stopped, result={result}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "stop_profile", "result": result})
        )

    # ------------------------------------------------------------------
    # MTP statistics
    # ------------------------------------------------------------------
    # 【译】MTP统计

    def _handle_get_mtp_stats(self, args: dict):
        """Print MTP statistics to log (fire-and-forget)."""
        # 【译】把MTP统计打印到日志（fire-and-forget）。
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is not None and stats.spec_enabled:
            stats.log_spec()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available "
                "(MTP not enabled or no tokens processed)\n"
            )

    def _handle_get_mtp_statistics(self, args: dict):
        """Return structured MTP statistics via UTILITY_RESPONSE."""
        # 【译】通过 UTILITY_RESPONSE 返回结构化的MTP统计。
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.spec_enabled:
            # 未开启投机解码时只回{"enabled": False}；LLMEngine.get_mtp_statistics据此过滤
            result = {"enabled": False}
        else:
            result = stats.spec_statistics()
            result["enabled"] = True
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_mtp_statistics", "result": result})
        )

    # ------------------------------------------------------------------
    # Prefix cache statistics
    # ------------------------------------------------------------------
    # 【译】前缀缓存统计

    def _handle_get_cache_statistics(self, args: dict):
        """Return structured prefix-cache statistics via UTILITY_RESPONSE.

        Same counters the periodic `[Cache Stats]` log line reports, on demand
        instead of every hundredth request — a client measuring reuse over a
        handful of requests cannot wait for that interval, and reading it out
        of a log is not something a client can do at all.
        """
        # 【译】通过 UTILITY_RESPONSE 返回结构化的前缀缓存统计。
        #
        #   与周期性的 [Cache Stats] 日志行报告的是同样的计数器，只是按需而不是每一百个请求一次——一个在少量请求上测量复用的
        #   客户端等不了那个间隔，而且客户端根本不可能从日志里读出它。
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.cache_enabled:
            result = {"enabled": False}
        else:
            result = stats.cache_statistics()
            result["enabled"] = True
            # The cache section counts the reuse a request wanted and did not
            # get; the funnel is where it was lost.
            # 【译】缓存部分统计的是一个请求想要却没有得到的复用；漏斗（funnel）说明它丢在了哪里。
            result |= self.scheduler.block_manager.checkpoint_funnel()
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_cache_statistics", "result": result})
        )

    # 推送本rank的指标快照：由busy_loop每ATOM_METRICS_UPDATE_INTERVAL_S调用一次；CoreManager存进latest_metrics
    #   LLMEngine.get_metrics_statistics（/metrics）读的就是这些快照
    def push_metrics(self, *, scheduler_metrics: bool = True) -> None:
        """Publish this rank's metrics snapshot on the output socket.

        Pushed on the engine's own clock rather than answered on demand. The
        pull version was a synchronous round trip with a 5s deadline fired every
        5s from the API server; whenever the engine was busy -- a long prefill,
        a GEMM autotune, a large batch -- it could not answer in time, so under
        load it failed on essentially every attempt, buried the server log in
        tracebacks, and left late replies in the response queue for the *next*
        caller to mistake for its own. Pushing removes the deadline, and with it
        the last off-loop writer on the control socket.
        """
        # 【译】把本rank的指标快照发布到输出socket上。
        #
        #   按引擎自己的节奏推送，而不是按需回答。拉取版本是API server每5秒发起一次、截止时间5秒的同步往返；只要引擎在忙——
        #   一次长prefill、一次GEMM自动调优、一个大batch——它就来不及回答，所以在负载下几乎每次尝试都失败，把server日志埋在
        #   traceback里，还把迟到的回复留在响应queue里，让*下一个*调用方误以为是自己的。推送去掉了截止时间，也就去掉了
        #   控制socket上最后一个事件循环之外的写者。
        # Poll ready device events even after the final forward. Workers write
        # native metrics directly; this RPC has no response payload.
        # 【译】即使在最后一次forward之后也要轮询已就绪的设备事件。worker直接写入原生指标；这个RPC没有响应负载。
        if envs.ATOM_ENABLE_METRICS_DEVICE_TIMER and self.runner_mgr is not None:
            self.runner_mgr.call_func("poll_forward_metrics")
        if scheduler_metrics:
            self.output_queue.put_nowait(("METRICS", self.collect_metrics()))

    # 收集本rank的调度器、KV、MTP、缓存、offload指标，组成一个dict（字段由LLMEngine.get_metrics_statistics聚合）
    def collect_metrics(self) -> dict:
        """One rank's scheduler, KV, MTP, and cache metrics."""
        # 【译】一个rank的调度器、KV、MTP和缓存指标。
        if self.scheduler is None:
            result = {"enabled": False}
        else:
            running, waiting = self.scheduler.get_request_counts()
            # None on the P/D prefill side, which owns no blocks — the decode
            # process does. Its snapshot then carries no kv_blocks_* keys at
            # all rather than a fabricated empty pool; the aggregator sums with
            # `.get(key, 0)`, so the decode rank's real figures come through
            # unchanged.
            # 【译】P/D的prefill一侧为None，它不拥有block——decode进程拥有。于是它的快照里根本不带 kv_blocks_* 键，
            #   而不是编造一个空池；聚合方用 .get(key, 0) 求和，所以decode rank的真实数字原样通过。
            block_manager = getattr(self.scheduler, "block_manager", None)
            kv_pool = None if block_manager is None else block_manager.kv
            kv_connector = getattr(self.scheduler, "kv_connector", None)

            engine_stats = self.scheduler.engine_stats
            if not engine_stats.spec_enabled:
                mtp = {"enabled": False}
            else:
                mtp = {"enabled": True, **engine_stats.spec_statistics()}

            if not engine_stats.cache_enabled:
                cache = {"enabled": False}
            else:
                cache = {
                    "enabled": True,
                    **engine_stats.cache_statistics(),
                    **self.scheduler.block_manager.checkpoint_funnel(),
                }

            offload = (
                kv_connector.get_statistics()
                if kv_connector is not None and hasattr(kv_connector, "get_statistics")
                else {}
            )
            result = {
                "enabled": True,
                # "prefill" / "decode" / "" — lets the aggregator recognise a
                # P/D pair, where one request is held by both ranks at once.
                # 【译】"prefill" / "decode" / ""——让聚合方能识别一对P/D，在那里一个请求同时被两个rank持有。
                "role": getattr(self.scheduler, "_METRICS_ROLE", ""),
                "requests_running": running,
                "requests_waiting": waiting,
                "requests_parked_kv_load": int(
                    getattr(self.scheduler, "_num_parked_remote_kv", 0)
                ),
                "requests_partial_prefill": int(
                    getattr(self.scheduler, "_partial_prefill_count", 0)
                ),
                "requests_finished": int(
                    getattr(self.scheduler, "total_finished_requests", 0)
                ),
                "prompt_tokens": int(getattr(self.scheduler, "total_prompt_tokens", 0)),
                "generation_tokens": int(
                    getattr(self.scheduler, "total_generation_tokens", 0)
                ),
                "preemptions": int(getattr(self.scheduler, "total_preemptions", 0)),
                "mtp": mtp,
                "cache": cache,
                "offload": offload,
            }
            if kv_pool is not None:
                # KV池的占用情况：evictable = 空闲但还挂着前缀缓存内容、可被淘汰复用的block；vacant = 真正空着的block
                reusable = kv_pool.num_reusable_free
                result |= {
                    "kv_blocks_used": kv_pool.num_used,
                    "kv_blocks_free": kv_pool.num_free,
                    "kv_blocks_total": kv_pool.num_blocks,
                    "kv_blocks_indexed": kv_pool.num_indexed,
                    "kv_blocks_evictable": reusable,
                    "kv_blocks_vacant": kv_pool.num_free - reusable,
                }

            metrics = getattr(self.scheduler, "metrics", None)
            if metrics is not None:
                parked = sum(
                    seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
                    for seq in self.scheduler.waiting
                )
                # Shared-cache disaggregation has a separate prefill queue,
                # whereas connector-based PD parks requests in `waiting`.
                # 【译】共享cache的PD分离有一个单独的prefill队列，而基于connector的PD把请求停放在 waiting 里。
                external = parked + len(getattr(self.scheduler, "prefill_waiting", ()))
                result["scheduler_metrics"] = {
                    "running": running,
                    "waiting": max(0, waiting - external),
                    "waiting_kv": external,
                }

        return result
