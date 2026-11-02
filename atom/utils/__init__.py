# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 本文件是atom.utils包本身：一组被全仓库共用的零散工具，按用途大致分成：
#   1. 进程管理：get_mp_context（spawn）、set_process_title / engine_process_name / worker_process_name（进程名）、
#      enable_orphan_reaping（父进程死了子进程自动退出）、shutdown_all_processes / kill_process_tree、init_exit_handler
#   2. 网络与ZMQ：端口和地址工具（get_open_port、get_open_zmq_ipc_path、make_zmq_path...）、
#      make_zmq_socket（建socket并设缓冲区/HWM）、zmq_socket_ctx
#   3. CPU->GPU数据搬运：upload_numpy、pack_rows、CpuGpuBuffer（pinned CPU缓冲 + GPU缓冲成对出现）
#   4. torch.compile / CUDA Graph辅助：mark_spliting_op（标记piecewise编译的切分点）、weak_ref_tensor(s)、
#      CompilationCounter、is_torch_equal_or_newer
#   5. 日志：getLogger，模块导入时就配置好名为"atom"的logger
#   以及杂项：set_ulimit、get_hf_text_config、resolve_obj_by_qualname
# 调用方遍布全仓库：EngineCore / AsyncIOProc / ModelRunner（第2批、第5批）用进程与ZMQ工具，
#   ModelRunner和各attention backend用CpuGpuBuffer、upload_numpy，utils/backends.py等编译代码用编译辅助
# 注意：任何`import atom.utils.xxx`（包括`from atom.utils import envs`）都会先执行本文件，连带导入torch、zmq、transformers
import contextlib
import copy
import dataclasses
import importlib
import ipaddress
import json
import logging
import multiprocessing
import os
import signal
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing.context import ForkContext, SpawnContext
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import psutil
import torch
import zmq
import zmq.asyncio
from packaging import version
from packaging.version import Version
from transformers import PretrainedConfig

from atom.utils.custom_register import direct_register_custom_op

if TYPE_CHECKING:
    from atom.config import Config

# patch（unittest.mock）：下面set_device_control_env_var用patch.dict临时修改os.environ
from unittest.mock import patch

# 本文件各函数共用的"atom" logger；文件末尾的getLogger()会给它挂上控制台handler
logger = logging.getLogger("atom")


# 把进程的"可打开文件数"软上限提高到target_soft_limit（不超过硬上限）
# 调用方：entrypoints/openai_server.py、benchmarks/benchmark_serving.py启动时调用
def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file soft limit toward ``target_soft_limit`` (capped at
    the hard limit).

    High streaming concurrency needs roughly one file descriptor per in-flight
    connection plus the engine's ZMQ/shared-memory fds. The default soft
    ``RLIMIT_NOFILE`` (~1024) is exhausted under large concurrency (e.g. the
    conc=1000 accuracy job), surfacing as EMFILE on ``accept()`` — which drops
    incoming connections. vLLM and SGLang raise this at process startup for the
    same reason; ATOM must too (the mesh launch scripts already pass
    ``--ulimit nofile`` to docker, but plain server launches do not).
    """
    # 【译】把打开文件数的软上限提高到``target_soft_limit``（以硬上限为顶）。
    #   高流式并发下，每个进行中的连接大约要占一个文件描述符，再加上引擎的ZMQ/共享内存fd。
    #   默认的软``RLIMIT_NOFILE``（约1024）在大并发下会耗尽（例如conc=1000的精度测试任务），
    #   表现为``accept()``返回EMFILE——进来的连接被丢弃。vLLM和SGLang出于同样原因在进程启动时提高这个上限；
    #   ATOM也必须这样做（mesh的启动脚本已经给docker传了``--ulimit nofile``，但普通的服务启动没有）。
    try:
        import resource
    except ImportError:  # POSIX-only; Windows has no RLIMIT_NOFILE.
        # 【译】（行尾）仅POSIX系统有；Windows没有RLIMIT_NOFILE。
        logger.warning("resource module unavailable (non-POSIX); skipping ulimit bump.")
        return

    resource_type = resource.RLIMIT_NOFILE
    soft, hard = resource.getrlimit(resource_type)
    # 硬上限为无穷大时直接取目标值，否则取min(目标值, 硬上限)；当前软上限已经够大就什么都不做
    desired = (
        target_soft_limit
        if hard == resource.RLIM_INFINITY
        else min(target_soft_limit, hard)
    )
    if soft >= desired:
        return
    try:
        resource.setrlimit(resource_type, (desired, hard))
        logger.info(
            "Raised RLIMIT_NOFILE soft limit %d -> %d (hard=%d)", soft, desired, hard
        )
    except (ValueError, OSError) as e:
        logger.warning(
            "Found RLIMIT_NOFILE soft=%d hard=%d and failed to automatically "
            "raise the soft limit to %d (error: %s). This can cause fd-limit "
            "errors like `OSError: [Errno 24] Too many open files` under high "
            "connection concurrency. The hard limit is the ceiling and cannot "
            "be raised from inside the process — raise it where the server is "
            "launched: docker `--ulimit nofile=65536:524288`, systemd "
            "`LimitNOFILE=`, or /etc/security/limits.conf.",
            soft,
            hard,
            desired,
            e,
        )


# 上下文管理器：在with块内临时设置一个"占位名"环境变量，值是本DP rank负责的GPU编号（如"2,3"）
# 按docstring所说，它是刻意无效的（inert），只为rollout适配器保留一个协议常量
# 注意：当前engine_core_mgr.py已经不再调用它（那里只在注释里引用这段docstring），
#   docstring里"CoreManager在data_parallel_size > 1时用它包住进程启动"的描述与现在的代码不符
@contextlib.contextmanager
def set_device_control_env_var(config: "Config", local_dp_rank: int):
    """Publish this engine's GPU slice under the RLHF device-map variable.

    Inert for the DP path that calls it, and must stay that way. It sets a
    variable named by the literal placeholder string; the only reader is
    ``RLHFModelRunner.DP_DEVICE_MAP_ENV`` (atom/rollout/model_runner_ext.py),
    which consults it solely when ``data_parallel_size <= 1``. ``CoreManager``
    only wraps a spawn in this when ``data_parallel_size > 1``, so the two
    never meet. Do not delete the placeholder string as dead -- it is a live
    protocol constant for the rollout adapter.

    Making it set a real ``HIP_VISIBLE_DEVICES`` was tried and reverted. The
    mask renumbers devices in the child (``cuda:0`` becomes the first *visible*
    GPU), but ``ModelRunner._setup_device_and_distributed`` independently
    computes an ABSOLUTE index -- ``local_dp_rank * tp_size + rank`` -- and
    selects ``cuda:{that}``. The two offsets compound: with ``-dp 4 -tp 2``,
    DP rank 1 would get a mask of "2,3" (two visible devices) and then ask for
    ``cuda:2``, which no longer exists. Startup dies on the
    ``local_device_rank >= torch.cuda.device_count()`` check -- every
    multi-GPU DP run, including ones that ship today.

    Either the mask or the absolute index can own device placement, not both.
    ATOM uses the absolute index, so this stays inert.
    """
    # 【译】把本引擎的GPU切片发布到RLHF的device-map变量下。
    #   对调用它的DP路径来说它不起作用，而且必须保持这样。它设置的变量名就是那个字面占位字符串；
    #   唯一的读取方是``RLHFModelRunner.DP_DEVICE_MAP_ENV``（atom/rollout/model_runner_ext.py），
    #   而它只在``data_parallel_size <= 1``时才读取。``CoreManager``只在``data_parallel_size > 1``时
    #   才用它包住进程启动，所以两者永远不会相遇。不要把这个占位字符串当成死代码删掉——它是rollout适配器的有效协议常量。
    #
    #   曾尝试让它设置真正的``HIP_VISIBLE_DEVICES``，后来回滚了。这个掩码会在子进程里给设备重新编号
    #   （``cuda:0``变成第一块*可见*GPU），但``ModelRunner._setup_device_and_distributed``会独立计算一个
    #   绝对下标——``local_dp_rank * tp_size + rank``——并选择``cuda:{该下标}``。两个偏移叠加在一起：
    #   在``-dp 4 -tp 2``下，DP rank 1会拿到掩码"2,3"（两块可见设备），然后去要``cuda:2``，而它已经不存在了。
    #   启动会在``local_device_rank >= torch.cuda.device_count()``检查处失败——
    #   每一次多GPU的DP运行都会这样，包括今天已经在发布的那些。
    #
    #   设备放置要么归掩码管，要么归绝对下标管，不能两者都管。ATOM使用绝对下标，所以这个函数保持无效。
    world_size = config.tensor_parallel_size
    evar = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"

    value = get_device_indices(evar, local_dp_rank, world_size)
    logger.debug(
        "%s=%s for local DP rank %d (inert placeholder)", evar, value, local_dp_rank
    )
    # patch.dict：只在with块内修改os.environ，退出with时自动恢复原值
    with patch.dict(os.environ, values=((evar, value),)):
        yield


# 计算DP rank负责的连续GPU编号[local_dp_rank*world_size, (local_dp_rank+1)*world_size)，拼成逗号分隔字符串
# 调用方：仅上面的set_device_control_env_var
def get_device_indices(
    device_control_env_var: str, local_dp_rank: int, world_size: int
):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.
    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    # 【译】返回指定数据并行rank对应的设备下标，形式是逗号分隔的字符串。
    #   例如world_size=2、local_dp_rank=1、共有4块设备时，local_dp_rank=1会选中设备2和3。
    # 注：range和join不会抛IndexError，下面的except分支实际上不会触发
    try:
        value = ",".join(
            str(i)
            for i in range(local_dp_rank * world_size, (local_dp_rank + 1) * world_size)
        )
    except IndexError as e:
        raise ValueError(
            f"Error setting {device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e
    return value


# 装饰器：把一个函数标记为piecewise编译的"切分点"（给它加spliting_op=True属性）
#   torch.compile捕获整张图后，utils/backends.py按节点target的spliting_op属性在这些位置把图切开：
#   切分点（通常是attention）在子图之外执行，其余子图分别编译，并可被CUDA Graph捕获
# is_custom=True时还会用direct_register_custom_op把函数注册成torch.ops.aiter.<函数名>自定义算子，
#   让Dynamo把它当成一个不透明节点：gen_fake提供fake实现（只推导输出形状），mutates_args声明会被原地修改的参数
#   装饰器返回的仍是原函数；调用方通过torch.ops.aiter.<函数名>(...)调用才走注册的算子（如paged_attention.py）
# 调用方：model_ops/base_attention.py的unified_attention_with_output_base等attention入口
# 【分支】piecewise编译与CUDA Graph详见第5批
def mark_spliting_op(
    is_custom: bool,
    gen_fake: Callable[..., Any] | None = None,
    mutates_args: list[str] | None = None,
):
    def decorator(func):
        if not is_custom:
            func.spliting_op = True
            return func

        direct_register_custom_op(
            op_name=func.__name__,
            op_func=func,
            mutates_args=mutates_args if mutates_args is not None else [],
            fake_impl=gen_fake,
        )
        registered_op = getattr(torch.ops.aiter, func.__name__)
        registered_op.spliting_op = True
        return func

    return decorator


# 多模态模型的HF config把文本部分放在text_config下；本函数取出它，纯文本模型原样返回
# 调用方：model_runner.py、gdn_attn.py、PD分离/KV offload的配置读取等需要文本模型字段（层数、头数）的地方
def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    # 【译】对多模态模型，取出与LLM相关的"子"配置。纯文本模型上什么都不做。
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        # 【译】代码假设text_config里应当有`num_attention_heads`（以及其他字段）。
        #   在这里断言，以便transformers的config与这一假设不符时尽早失败。
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config


# 返回spawn方式的multiprocessing上下文（返回类型标注写了ForkContext，但实际永远是spawn）
#   CLAUDE.md规定多进程必须用spawn：fork会把父进程已初始化的GPU运行时状态复制给子进程，子进程再初始化CUDA/HIP时会崩溃
# 调用方：model_engine/async_proc.py创建ModelRunner子进程
# 对比nano-vllm：nano-vllm在LLMEngine.__init__里直接mp.get_context("spawn")
def get_mp_context() -> ForkContext | SpawnContext:
    """Get a multiprocessing context with 'spawn' start method."""
    # 【译】获取start method为'spawn'的multiprocessing上下文。
    return multiprocessing.get_context("spawn")


# 设置当前进程标题，形如"ATOM::EngineCore_DP0"、"ATOM::TP3"，方便在ps/top/rocm-smi里区分各rank
#   前缀默认来自ATOM_PROCESS_NAME_PREFIX（默认"ATOM"）；调用方：engine_core.py、async_proc.py
def set_process_title(name: str, suffix: str = "", prefix: str | None = None) -> None:
    """Set the current process title (comm/cmdline) for ps/top/rocm-smi.

    rocm-smi --showpids reads the process ``comm`` field, which defaults to the
    interpreter name (``python``). Setting a title makes GPU-holding worker
    processes distinguishable by rank. Soft dependency: no-op if setproctitle
    is not installed.
    """
    # 【译】为ps/top/rocm-smi设置当前进程的标题（comm/cmdline）。
    #   rocm-smi --showpids读取进程的``comm``字段，默认是解释器名（``python``）。设置标题后，
    #   占用GPU的各个worker进程就能按rank区分开。这是软依赖：没装setproctitle时什么也不做。
    try:
        import setproctitle
    except ImportError:
        return
    from atom.utils import envs

    if prefix is None:
        prefix = envs.ATOM_PROCESS_NAME_PREFIX
    if suffix:
        name = f"{name}_{suffix}"
    setproctitle.setproctitle(f"{prefix}::{name}")


# EngineCore进程的名字：PP>1时为EngineCore_PP<rank>，DP>1时为EngineCore_DP<rank>，否则就是EngineCore
# 调用方：engine_core.py（用作进程标题和日志里的名字）
def engine_process_name(config: "Config") -> str:
    """What an EngineCore process is called, in `ps` and in its own logs."""
    # 【译】EngineCore进程在`ps`里和在它自己日志里叫什么。
    pc = config.parallel_config
    if config.pipeline_parallel_size > 1:
        return f"EngineCore_PP{pc.pipeline_parallel_rank}"
    if pc.data_parallel_size > 1:
        return f"EngineCore_DP{pc.data_parallel_rank}"
    return "EngineCore"


# ModelRunner worker进程的名字：DP>1时为DP<dp_rank>TP<rank>，否则为TP<rank>
# 调用方：async_proc.py、model_runner.py
def worker_process_name(config: "Config | None", rank: int) -> str:
    """What a ModelRunner worker is called. `config` may be absent on paths
    that build a worker without one."""
    # 【译】ModelRunner worker叫什么。在不带config构建worker的路径上，`config`可能为空。
    pc = getattr(config, "parallel_config", None)
    if pc is not None and pc.data_parallel_size > 1:
        return f"DP{pc.data_parallel_rank}TP{rank}"
    return f"TP{rank}"


# 分几步关闭一组子进程：回收已退出的 -> terminate(SIGTERM)其余的 -> 在allowed_seconds总时限内等待
#   -> 还活着的连同子孙进程一起SIGKILL -> 最后close()释放进程对象的资源
# 调用方：model_engine/async_proc.py关闭ModelRunner进程
def shutdown_all_processes(procs: list[BaseProcess], allowed_seconds: int = 2):
    # First join any already-exited processes (instant, no wait).
    # 【译】先join已经退出的进程（立即返回，不等待）。
    for proc in procs:
        if not proc.is_alive():
            proc.join(timeout=0)

    # Terminate remaining alive processes.
    # 【译】终止其余仍存活的进程。
    alive = [p for p in procs if p.is_alive()]
    for proc in alive:
        proc.terminate()

    # Wait for remaining procs to terminate.
    # 【译】等待剩余进程终止。
    # deadline是所有进程共享的总时限，而不是每个进程各等allowed_seconds秒
    deadline = time.monotonic() + allowed_seconds
    for proc in alive:
        remaining = max(deadline - time.monotonic(), 0)
        proc.join(remaining)

    # Force kill anything still alive.
    # 【译】强制杀掉仍然存活的进程。
    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)
            proc.join(timeout=1)  # wait for kill to take effect
            # 【译】（行尾）等待kill生效

    # Release internal process resources (sentinel semaphores, etc.)
    # 【译】释放进程内部资源（sentinel信号量等）
    for proc in procs:
        try:
            proc.close()
        except (ValueError, OSError):
            pass


# 让内核在"父进程退出"时给本进程发sig（默认SIGKILL），防止孤儿进程一直占着显存和IPC资源
# 调用方：async_proc.py（ModelRunner worker）、engine_core.py（EngineCore）在各自进程入口尽早调用
def enable_orphan_reaping(sig: int = signal.SIGKILL) -> bool:
    """Arm the kernel to reap *this* process if its parent ever exits.

    ATOM runs a tree of processes: the server (main) -> one ``EngineCore``
    process per DP rank -> ``tensor_parallel_size`` ``ModelRunner`` worker
    processes.  Each worker holds a large slice of GPU VRAM plus the custom
    all-reduce IPC resources (``hipIpcGetMemHandle`` handles + the rendezvous
    ``TCPStore`` bound to ``MASTER_PORT``).

    If a parent exits abnormally (OOM-kill, segfault, ``SIGKILL``,
    ``docker stop``) the kernel does not clean up its children: the worker's
    ``busy_loop`` blocks forever on the shared-memory RPC dequeue and the
    EngineCore blocks on its input queue.  These orphans keep the VRAM and the
    IPC/TCP resources pinned, so a subsequent restart either fails to bind the
    rendezvous port or, worse, opens a *stale* IPC mem handle exported by the
    dead run -> the ``hipIpcGetMemHandle`` all-reduce crash operators work
    around with ``docker rm -f`` + a lowered ``--gpu-memory-utilization``.

    This helper wires up ``prctl(PR_SET_PDEATHSIG)`` so the kernel delivers
    ``sig`` to the caller the instant its parent exits, for *any* reason --
    turning a silent orphan into an immediate, self-reaping exit that releases
    every GPU and IPC resource it held.  The setting is per-thread and is
    cleared across ``execve``, so it cannot be inherited: each process must arm
    itself early in its entrypoint, before any GPU / IPC state is created.

    Linux-only (no-op elsewhere).  Returns ``True`` when reaping is armed and
    ``False`` if it could not be set.  If the parent is found to be already gone
    at arm time, it does not return: it calls ``os._exit(1)`` rather than let the
    process linger as the orphan this is meant to prevent.

    Caveat: ``PR_SET_PDEATHSIG`` fires when the *thread that created this
    process* exits, not when the parent process as a whole exits.  Arm it only
    from a process whose creating thread lives for the process's lifetime --
    ATOM spawns workers from the main thread, so this holds; a short-lived
    creator thread could otherwise trigger a premature kill.
    """
    # 【译】让内核在*本*进程的父进程退出时回收它。
    #   ATOM运行的是一棵进程树：server（主进程）-> 每个DP rank一个``EngineCore``进程 ->
    #   ``tensor_parallel_size``个``ModelRunner`` worker进程。每个worker持有一大块GPU显存，
    #   以及custom all-reduce的IPC资源（``hipIpcGetMemHandle``句柄 + 绑定在``MASTER_PORT``上的rendezvous ``TCPStore``）。
    #
    #   如果父进程异常退出（OOM-kill、段错误、``SIGKILL``、``docker stop``），内核不会清理它的子进程：
    #   worker的``busy_loop``永远阻塞在共享内存RPC的出队上，EngineCore阻塞在它的输入队列上。
    #   这些孤儿进程一直占着显存和IPC/TCP资源，于是下一次重启要么绑定不了rendezvous端口，
    #   要么更糟——打开上一次已死进程导出的*过期*IPC内存句柄——这就是``hipIpcGetMemHandle`` all-reduce崩溃，
    #   运维人员只能靠``docker rm -f``再调低``--gpu-memory-utilization``来绕过。
    #
    #   这个辅助函数接上``prctl(PR_SET_PDEATHSIG)``，让内核在父进程退出的那一刻——无论什么原因——
    #   向调用者发送``sig``，把无声的孤儿变成立即的、自我回收的退出，释放它持有的所有GPU和IPC资源。
    #   这个设置是按线程的，并且在``execve``时被清除，所以无法继承：每个进程都必须在自己的入口处尽早设置，
    #   要早于创建任何GPU / IPC状态。
    #
    #   仅限Linux（其他平台什么也不做）。设置成功返回``True``，无法设置返回``False``。
    #   如果设置时发现父进程已经不在了，它不会返回：而是调用``os._exit(1)``，不让进程变成它本要防止的那种孤儿。
    #
    #   注意：``PR_SET_PDEATHSIG``在*创建本进程的那个线程*退出时触发，而不是在父进程整体退出时。
    #   只能在创建者线程与进程同生命周期的进程里设置它——ATOM从主线程spawn worker，所以满足这一点；
    #   如果创建者是一个短命线程，就可能触发过早的kill。
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        # Resolve libc from the already-loaded image (``CDLL(None)``) rather than
        # hard-coding ``libc.so.6``; this works on glibc and musl (e.g. Alpine)
        # alike, where the soname differs.
        # 【译】从已加载的进程映像（``CDLL(None)``）解析libc，而不是写死``libc.so.6``；
        #   这样在glibc和musl（例如Alpine）上都能工作，二者的soname不同。
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            logger.warning("prctl(PR_SET_PDEATHSIG) failed: errno=%d", err)
            return False
    except Exception as e:  # noqa: BLE001 - best-effort; platform-dependent
        # ctypes/prctl is Linux-specific and can fail on a restricted or
        # non-glibc image. Orphan reaping is a safety net, so losing it must
        # not stop the process that was trying to arm it.
        # 【译】ctypes/prctl是Linux特有的，在受限或非glibc的镜像上可能失败。孤儿回收只是一道安全网，
        #   失去它不能阻止正在设置它的那个进程继续运行。
        #   （except行尾）尽力而为；依赖平台。
        logger.warning("Could not arm orphan reaping: %s", e)
        return False

    # The parent can die between spawning us and this call -- before or after the
    # prctl above -- so PR_SET_PDEATHSIG may never fire.  Detect it unambiguously
    # via multiprocessing's parent handle (a sentinel pipe): unlike
    # ``getppid() == 1``, this is not fooled by a parent that legitimately runs
    # as PID 1 (ATOM's server is PID 1 in the container).  If the parent is
    # already gone, exit now rather than linger as the orphan this prevents.
    # 【译】父进程可能在spawn出我们和这次调用之间死掉——在上面的prctl之前或之后都有可能——所以PR_SET_PDEATHSIG可能永远不会触发。
    #   通过multiprocessing的parent句柄（一个sentinel管道）无歧义地检测这种情况：与``getppid() == 1``不同，
    #   它不会被一个合法地以PID 1运行的父进程骗过（ATOM的server在容器里就是PID 1）。
    #   如果父进程已经不在，立即退出，而不是作为本函数要防止的孤儿继续逗留。
    parent = multiprocessing.parent_process()
    if parent is not None and not parent.is_alive():
        logger.warning("Parent already exited while arming orphan reaping; exiting.")
        os._exit(1)
    return True


# 用psutil找到pid的所有子孙进程，先SIGKILL子孙、再SIGKILL它自己；进程已经不存在时静默忽略
# 调用方：上面的shutdown_all_processes
def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    # 【译】向给定pid的所有后代进程发送SIGKILL，杀掉它们。
    #   参数：
    #     pid (int)：父进程的进程ID
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    # 【译】递归获取所有子进程
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    # 【译】先向所有子进程发送SIGKILL
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    # 【译】最后杀掉父进程本身
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# ---- 网络地址工具：IPv6地址拼端口时要用方括号包起来（[::1]:8000），下面几个函数统一处理这件事 ----
# 判断字符串是否是合法的IPv6地址
def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


# "host:port"或"[ipv6]:port" -> (host, port)；仓库内目前没有调用方
def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    # 【译】ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


# split_host_port的逆操作；仓库内目前没有调用方
def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


# 生成torch.distributed初始化用的tcp://地址
# 注意：model_runner.py、plugin/register.py等用的是aiter.dist.utils里的同名函数，这个版本目前没人调用
def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


# 拼出"tcp://ip:port"，IPv6地址加方括号
def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


# 生成一个唯一的ZMQ进程内（inproc://）地址，只能在同一进程的线程之间通信；调用方：engine_core_mgr.py
def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


# 让操作系统分配一个当前空闲的TCP端口
# 调用方：PD分离的mooncake / moriio connector（握手端口）、diffusion/engine/core_manager.py（MASTER_PORT）
def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    # 【译】为vLLM进程获取一个可以监听的空闲端口。
    #   需要处理的一个边界情况是数据并行：要避开可能被数据并行主进程使用的端口。
    #   目前给数据并行主进程预留了10个端口，当前用了2个。
    # 注：docstring沿用自vLLM；这里的实现直接调用_get_open_port，并没有做预留端口的处理
    return _get_open_port()


# 取count个互不相同的空闲端口；仓库内目前没有调用方
def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of open ports."""
    # 【译】获取一组空闲端口。
    ports = set()
    while len(ports) < count:
        ports.add(get_open_port())
    return list(ports)


# bind到端口0让内核分配空闲端口，读出端口号后立刻关闭socket
#   固有竞态：关闭之后、真正使用之前，端口可能被别的进程抢走（config.py的stateless_init_dp_group因此会重试）
def _get_open_port() -> int:
    # try ipv4
    # 【译】先尝试ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        # 【译】再尝试ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


# IPC socket文件放在系统临时目录下；@lru_cache：只计算一次，之后直接返回缓存结果
@lru_cache
def get_zmq_base_path() -> str:
    return tempfile.gettempdir()


# 生成唯一的ZMQ IPC地址（ipc://<临时目录>/<uuid>），走Unix域socket，用于同一台机器上的进程间通信
# 调用方：engine_core_mgr.py、async_proc.py（引擎各进程之间的socket地址），diffusion
def get_open_zmq_ipc_path() -> str:
    base_rpc_path = get_zmq_base_path()
    return f"ipc://{base_rpc_path}/{uuid4()}"


# local_only时用IPC地址，否则用tcp地址（port为0时自动分配）；仓库内目前没有调用方
def get_engine_client_zmq_addr(local_only: bool, host: str, port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""
    # 【译】分配一个新的ZMQ socket地址。
    #   如果local_only为True，参与方都在同一台机器上，返回一个唯一的IPC地址。
    #   否则用给定的host和port构造TCP地址（port == 0表示分配一个可用端口）。

    return (
        get_open_zmq_ipc_path()
        if local_only
        else (get_tcp_uri(host, port or get_open_port()))
    )


# 立即关闭一组socket（linger=0：不等待尚未发出的消息）；仓库内目前没有调用方
def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


# 把"tcp://host:port" / "ipc://path"拆成(scheme, host, port)并校验：tcp必须有host和port，非tcp不能带port
# 调用方：make_zmq_socket（判断是否需要启用IPv6）
def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    # 【译】把zmq路径拆分成各个部分。
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    # Port 0 is a request for the kernel to assign one, not a missing port, so
    # it has to survive the emptiness check below.
    # 【译】端口0是请求内核分配一个端口，而不是缺少端口，所以它必须能通过下面的非空检查。
    port = "" if parsed.port is None else str(parsed.port)

    if scheme == "tcp" and not (host and port):
        # The host and port fields are required for tcp
        # 【译】tcp必须有host和port字段
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        # 【译】port只对tcp有意义
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


# 由(scheme, host, port)拼出ZMQ地址，IPv6加方括号；是split_zmq_path的逆操作
# 调用方：PD分离的mooncake / moriio connector
def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    # 【译】由各个部分组成ZMQ路径。
    #   参数：
    #     scheme：ZMQ传输方式（例如tcp、ipc、inproc）。
    #     host：主机——可以是IPv4地址、IPv6地址或主机名。
    #     port：可选的端口号，只用于TCP socket。
    #   返回：
    #     格式正确的ZMQ路径字符串。
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# 创建一个ZMQ socket并设好选项，再按bind参数决定bind还是connect
#   ctx：zmq.Context（同步）或zmq.asyncio.Context（异步）；socket_type：zmq.PUSH/PULL/DEALER/ROUTER等
#   bind=None时自动判断：PUSH/SUB/XSUB默认connect，其他类型默认bind
# 调用方：async_proc.py、engine_core.py、engine_core_mgr.py（引擎三层进程之间）、moriio connector、diffusion
# 对比nano-vllm：nano-vllm的进程间通信用共享内存+Event；ATOM的多进程架构建立在ZMQ socket上，见第2批
# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783
# 【译】改编自：https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""
    # 【译】创建一个具有正确bind/connect语义的ZMQ socket。

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    # 【译】根据系统内存计算缓冲区大小
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    # 【译】对于内存充裕的系统（总内存>32GB、可用>16GB）：
    #   - 设置0.5GB的大缓冲区以提高吞吐
    #   对于内存较少的系统：
    #   - 使用系统默认值（-1），避免占用过多内存
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
        # 【译】（行尾）0.5GB，单位为字节
    else:
        buf_size = -1  # Use system default buffer size
        # 【译】（行尾）使用系统默认的缓冲区大小

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    # HWM（高水位线）设为0表示不限制排队的消息数：不会因为队列满而丢消息或阻塞发送方（代价是可能占用较多内存）
    #   接收侧类型（PULL/DEALER/ROUTER）设RCVHWM/RCVBUF，发送侧类型（PUSH/DEALER/ROUTER）设SNDHWM/SNDBUF
    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    # identity：ROUTER端靠它识别消息来自哪个DEALER；linger：close时最多等待未发出消息的毫秒数
    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    # XPUB_VERBOSE：让XPUB把每条订阅消息都交给应用（包括重复订阅）
    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    # 【译】判断路径是否是带IPv6地址的TCP socket；如果是，就在zmq socket上启用IPv6。
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


# 给对象self装上退出处理：weakref.finalize保证self.exit()最多执行一次（收到SIGTERM、对象被回收或解释器退出时）
#   SIGTERM -> 记日志并调用finalizer；SIGINT（Ctrl+C）被忽略，由主进程统一用SIGTERM有序关闭
# 调用方：EngineCore、AsyncIOProc、ModelRunner等的__init__；要求self有exit()方法和label属性
def init_exit_handler(self: Any):
    import weakref

    self.still_running = True
    self._finalizer = weakref.finalize(self, self.exit)

    def signal_handler(signum, frame):
        sig_str = signal.Signals(signum).name
        msg = f"{self.label}: received signal {signum} ({sig_str}), exiting..."
        logger.info(msg)
        self._finalizer()

    # Ignore SIGINT in subprocesses — let the main process handle Ctrl+C
    # and orchestrate orderly shutdown via SIGTERM. This prevents C++ NCCL
    # HeartbeatMonitor TCPStore errors caused by ranks exiting independently.
    # 【译】在子进程里忽略SIGINT——让主进程处理Ctrl+C，并通过SIGTERM编排有序关闭。
    #   这样可以避免各rank各自退出引起的C++ NCCL HeartbeatMonitor TCPStore错误。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal_handler)


# 上下文管理器：新建一个zmq.Context和socket，with块结束时destroy整个Context（连带关闭socket，linger用于destroy）
# 调用方：PD分离的mooncake / moriio connector
@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""
    # 【译】ZMQ socket的上下文管理器

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)


# A pageable host-to-device copy larger than the driver's staging buffer cannot
# be handed off and returned from: the host waits for it, and because it goes
# out on the current stream it waits behind everything already queued there.
# Under it, the driver stages the bytes and returns immediately.
#
# Measured on MI355X with 4 ms of work in flight, host time charged to the copy:
#
#     bytes    pageable   pin_memory()   persistent pinned
#     64 KB    0.018 ms      0.021 ms         0.022 ms
#    400 KB    0.037 ms      0.167 ms         0.014 ms
#    512 KB    0.032 ms          --           0.014 ms
#    560 KB    3.047 ms          --           0.015 ms   <-- cliff
#    800 KB    3.058 ms      0.168 ms         0.031 ms
#
# Three things follow. Below the cliff every route costs about the same, so the
# one that allocates nothing wins. Above it pageable is ~100x worse and the
# choice is forced. And neither route here is as good as a persistent pinned
# buffer: charged against queue depth at 1.5 MB, pageable grows +84 us per
# queued matmul and a `CpuGpuBuffer` is flat, while a per-call `pin_memory()`
# lands anywhere from +3 to +36 depending on how fragmented the caching host
# allocator already is -- better than pageable, and not something to rely on.
# That is why `CpuGpuBuffer` exists and why a per-forward buffer should be one.
# 【译】超过驱动staging缓冲区大小的可分页（pageable）主机到设备拷贝，无法"交出去就返回"：
#   主机要等它完成，而且因为它走当前stream，还要排在该stream上已入队的所有工作之后。
#   低于这个大小时，驱动会先把这些字节暂存起来，然后立即返回。
#
#   在MI355X上测得（有4 ms的工作在途，统计计入这次拷贝的主机耗时）：
#     表头依次是：字节数、可分页内存、每次调用pin_memory()、持久的锁页缓冲；
#     560 KB那一行就是"悬崖"：可分页从0.032 ms跳到3.047 ms
#
#   由此得出三点。悬崖以下，各种方式开销差不多，所以什么都不分配的那种胜出。悬崖以上，
#   可分页差了约100倍，只能换方式。而且这里两种方式都不如持久的锁页缓冲：在1.5 MB时按队列深度计，
#   可分页每多排一个matmul就增加84 us，`CpuGpuBuffer`保持平稳；而每次调用`pin_memory()`
#   则在+3到+36之间不等，取决于缓存式主机分配器已经碎片化到什么程度——比可分页好，但不能依赖。
#   这就是`CpuGpuBuffer`存在的原因，也是为什么每次forward都要上传的缓冲区应当是一个CpuGpuBuffer。
# 512KB：不超过它就用可分页拷贝，超过它就临时pin_memory再拷（见下面的upload_numpy）
_PAGEABLE_H2D_LIMIT = 512 * 1024


# 对超限的上传打警告；@lru_cache(maxsize=8)让同一个字节数只警告一次，避免刷屏
@lru_cache(maxsize=8)
def _warn_oversize_upload(nbytes: int) -> None:
    logger.warning(
        "upload_numpy: %d B is past the %d B pageable limit, so it is staged "
        "through a fresh pinned block every call. A per-forward upload this "
        "size should own a CpuGpuBuffer instead.",
        nbytes,
        _PAGEABLE_H2D_LIMIT,
    )


# 把numpy数组拷到GPU：<=512KB直接可分页拷贝（驱动暂存后立即返回）；更大的先pin_memory再拷，并打警告
# 输入：任意形状的numpy数组；输出：device上的torch.Tensor（non_blocking默认True）
# 调用方：aiter_attention.py、aiter_mla.py、deepseek_v4_attn.py等attention backend准备元数据时
def upload_numpy(arr: np.ndarray, device, *, non_blocking: bool = True) -> torch.Tensor:
    """Copy a numpy array to `device` by whichever route its size calls for.

    For one-off and size-varying uploads, where the size can cross the cliff as
    a config knob moves and nobody notices until a step takes milliseconds
    longer. A buffer uploaded every forward should be a `CpuGpuBuffer`.

    A shared reusable staging block is not a third option, and was tried: its
    copy goes out on the current stream, so the event that makes reuse safe
    sits behind whatever is queued, and waiting on it costs what pageable
    costs. Only a block nobody else touches escapes that.
    """
    # 【译】按数组大小选择合适的方式，把numpy数组拷到`device`。
    #   用于一次性的、大小会变化的上传：这种上传的大小可能随某个配置项变化而越过悬崖，
    #   没人注意到，直到某一步多花了几毫秒。每次forward都上传的缓冲区应当用`CpuGpuBuffer`。
    #
    #   共享的可复用暂存块并不是第三种选择，而且已经试过：它的拷贝走当前stream，所以保证复用安全的event
    #   排在已入队的工作之后，等它的开销和可分页一样。只有一块没有别人碰的内存才能避开这个问题。
    if arr.nbytes <= _PAGEABLE_H2D_LIMIT:
        return torch.from_numpy(arr).to(device, non_blocking=non_blocking)
    _warn_oversize_upload(arr.nbytes)
    return torch.from_numpy(arr).pin_memory().to(device, non_blocking=non_blocking)


# 把一组长短不一的行（ragged rows）左对齐写进二维int32数组dst，每行剩余部分补0
#   例：rows=[[5,6],[7]]，dst形状[4,3] -> dst[0]=[5,6,0]、dst[1]=[7,0,0]，dst[2:]不动（只清零用到的行）
# 调用方：aiter_attention.py、deepseek_v41/cache.py、token_layout/slots.py等（拼block table、slot等元数据）
def pack_rows(dst: np.ndarray, rows: Sequence) -> None:
    """Left-align ragged `rows` into a 2-D int32 `dst`, zero-filling the rest.

    A numpy row assign costs ~245ns on MI355X and does not vary with the row's
    length until the bytes start to matter (~500 ints), so a marshal of short
    rows is dispatch and nothing else -- 4.0 ms at 16k rows. Through one
    `memoryview` of the destination the same write is ~46ns.

    `.cast` is what makes the flattening safe: it refuses a non-contiguous
    buffer, where `reshape(-1)` would hand back a copy and silently drop every
    write. Rows must export a buffer of int32 -- an `array("i")` or an int32
    ndarray -- since that is what makes each write a memcpy.
    """
    # 【译】把参差不齐的`rows`左对齐写入二维int32的`dst`，其余位置补零。
    #   在MI355X上，一次numpy行赋值约245ns，而且在字节数开始起作用之前（约500个int）不随行长变化，
    #   所以对短行的整理只剩分派开销——16k行要4.0 ms。通过目标数组的一个`memoryview`，同样的写入只要约46ns。
    #
    #   `.cast`是让展平变安全的关键：它拒绝非连续的缓冲区，而`reshape(-1)`会返回一个副本并悄悄丢掉所有写入。
    #   每一行都必须导出int32缓冲区——`array("i")`或int32 ndarray——这样每次写入才是一次memcpy。
    if dst.dtype != np.int32:
        # `.cast("i")` would reinterpret the bits of any other dtype rather
        # than convert them, so this must be checked and not assumed.
        # 【译】`.cast("i")`会重新解释其他dtype的比特，而不是转换它们，所以必须检查，不能假设。
        raise TypeError(f"pack_rows needs an int32 destination, got {dst.dtype}")
    # Both dimensions are checked here because a flat write has no edge in
    # either: an over-wide row would land in the next row, and an over-long
    # batch would run off the end with only a memoryview structure error to
    # say so. The numpy form this replaces raised on both.
    # 【译】这里两个维度都要检查，因为扁平写入在哪个维度上都没有边界：过宽的行会写进下一行，
    #   过长的batch会写出末尾，只得到一个memoryview结构错误来提示。被替换掉的numpy写法对两者都会报错。
    n_rows, (max_rows, cols) = len(rows), dst.shape
    if n_rows > max_rows:
        raise ValueError(f"{n_rows} rows exceed the {max_rows}-row destination")
    dst[:n_rows] = 0  # one memset over the rows in play, not one per row
    # 【译】（行尾）对用到的这些行做一次memset，而不是每行一次
    # memoryview(dst).cast("B").cast("i")：先看成字节、再看成int32的一维视图，写flat就是直接写dst的内存
    flat = memoryview(dst).cast("B").cast("i")
    base = 0
    for row in rows:
        n = len(row)
        if n > cols:
            raise ValueError(f"row of {n} exceeds the {cols}-column destination")
        flat[base : base + n] = row
        base += cols  # the destination's stride, not the row's length
        # 【译】（行尾）按目标数组的步长（列数）前进，而不是按这一行的长度


# 一对同形状的缓冲区：cpu（pinned锁页内存）+ gpu，外加与cpu共享内存的numpy视图np
# 典型用法（ModelRunner准备每步输入）：CPU侧写self.np / self.cpu -> copy_to_gpu(n)异步拷前n行 -> kernel读self.gpu
#   锁页内存才能真正异步地做H2D拷贝；缓冲区预先分配、每步复用，避免反复分配（原因见上面_PAGEABLE_H2D_LIMIT的说明）
# 调用方：model_runner.py（input_ids、decode_src、forward_vars里的positions/temperatures等）、各attention backend的元数据缓冲区，共二十多个文件
# 对比nano-vllm：nano-vllm每步用torch.tensor(..., pin_memory=True).cuda(non_blocking=True)新建张量；ATOM改为预分配复用
# 【分支】publication（utils/h2d.py）：绑定后由发布者统一执行拷贝并检查写入时机（ATOM_H2D_BACKEND），见第5批
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    # 【译】在CPU和GPU之间方便地拷贝张量的缓冲区。
    # __setattr__：一旦被h2d发布者绑定（_publication不为None），就禁止替换cpu/gpu/np这三个存储属性——发布者记录的是原来那块内存
    def __setattr__(self, name, value):
        if (
            name in ("cpu", "gpu", "np")
            and self.__dict__.get("_publication") is not None
        ):
            from atom.utils.h2d import PublicationError

            raise PublicationError(
                "bound metadata storage is fixed; recreate the buffer and owner"
            )
        object.__setattr__(self, name, value)

    # 输入：*size形状、dtype、device；pin_memory=True时CPU侧用锁页内存；with_numpy=True时额外建共享内存的numpy视图
    #   publication_group / publication_unit：供h2d发布者分组、计量（按行等）
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = True,
        with_numpy: bool = True,
        publication_group: str | None = None,
        publication_unit: str = "rows",
    ) -> None:
        self._publication = None
        self.publication_group = publication_group
        self.publication_unit = publication_unit
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        # 【译】为了让类型标注保持简单（避免泛型和子类），只在需要时才创建numpy数组属性。
        #   因此在`with_numpy=False`时访问`self.np`会抛AttributeError。
        if with_numpy:
            # numpy没有bfloat16类型，bf16张量无法转成numpy视图
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    # 把cpu的前n行（第0维，n=None时全部）异步拷到gpu，返回gpu上对应的切片
    #   已绑定发布者时改由发布者执行拷贝（republish_reason说明为什么要重新发布）
    def copy_to_gpu(
        self, n: int | None = None, *, republish_reason: str | None = None
    ) -> torch.Tensor:
        if self._publication is not None:
            return self._publication.copy_to_gpu(n, republish_reason=republish_reason)
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    # 反方向：gpu的前n行异步拷回cpu
    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        # 【译】注意：这个方法是非阻塞的，需要显式同步才能确保数据已经拷到CPU。
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)

    # 复制出一个独立的缓冲区：同形状/dtype/device，并带上当前内容
    def clone(self) -> "CpuGpuBuffer":
        """Allocate an independent buffer with the same shape/dtype/device and
        copy over the current cpu+gpu contents (preserves one-time inits like
        arange/zeros). Used to build per-in-flight-slot metadata rings so
        concurrent pipeline microbatches never share a staging buffer."""
        # 【译】分配一个形状/dtype/device都相同的独立缓冲区，并拷贝当前cpu+gpu的内容（保留arange/zeros这类一次性初始化）。
        #   用来构建按在途槽位划分的元数据环，使并发的流水线微批次永远不会共用同一个暂存缓冲区。
        new = CpuGpuBuffer(
            *self.cpu.shape,
            dtype=self.cpu.dtype,
            device=self.gpu.device,
            pin_memory=self.cpu.is_pinned(),
            with_numpy=hasattr(self, "np"),
            publication_group=self.publication_group,
            publication_unit=self.publication_unit,
        )
        new.cpu.copy_(self.cpu)
        new.gpu.copy_(self.gpu)
        return new


# 这两个模块级变量在本文件里没有被使用：utils/decorators.py定义并使用了它自己的同名全局变量
context_manager = None
torch_compile_start_time: float = 0.0


# 判断已安装的torch版本是否>=target；调用方：utils/backends.py、compiler_inferface.py、fused_moe/utils.py等按torch版本选API
def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    # 【译】检查已安装的torch版本是否>=目标版本。
    #   参数：
    #     target：版本字符串，例如"2.6.0"。
    #   返回：
    #     条件是否满足。
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:  # noqa: BLE001 - any parse failure falls back to PKG-INFO
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        # 【译】退回用PKG-INFO读取包信息，文档生成时需要这样。
        #   （except行尾）任何解析失败都退回PKG-INFO
        return Version(importlib.metadata.version("torch")) >= Version(target)


# Helper function used in testing.
# 【译】测试中使用的辅助函数。
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)


# Lazily-compiled fallback weak-ref op. vLLM ships a `torch.ops._C.weak_ref_tensor`
# (csrc/libtorch_stable/ops.h) that returns a tensor SHARING the input's memory
# but NOT owning its storage — so once the original tensor is freed the
# CUDACachingAllocator block is reclaimable and a cudagraph memory pool can
# OVERLAY it across shapes. That op is absent in some ROCm builds (no vLLM _C),
# and a pure-Python weak ref is impossible: dlpack / set_(untyped_storage) both
# share the Storage object and thus KEEP it alive -> the pool cannot reuse the
# memory -> every captured piece's output accumulates (35GB+ on DSV4 TP8
# PIECEWISE). We therefore JIT-compile a tiny `at::from_blob(..., no-op deleter)`
# equivalent once (cached under ~/.cache/torch_extensions) so PIECEWISE cudagraph
# pools overlay exactly like upstream vLLM.
# 【译】惰性编译的后备weak-ref算子。vLLM提供了一个`torch.ops._C.weak_ref_tensor`
#   （csrc/libtorch_stable/ops.h），它返回一个与输入*共享*内存、但*不拥有*其storage的张量——
#   所以原张量一旦释放，CUDACachingAllocator的block就可以回收，cudagraph内存池就能在不同形状之间*覆盖复用*它。
#   某些ROCm构建里没有这个算子（没有vLLM的_C），而纯Python的弱引用做不到：dlpack / set_(untyped_storage)
#   都共享Storage对象，从而*保持*它存活 -> 内存池无法复用这块内存 -> 每个捕获片段的输出不断累积
#   （DSV4 TP8 PIECEWISE下超过35GB）。因此我们一次性JIT编译一个很小的、等价于`at::from_blob(..., 空操作deleter)`的算子
#   （缓存在~/.cache/torch_extensions下），让PIECEWISE cudagraph的内存池像上游vLLM一样覆盖复用。
_ATOM_WEAKREF_OP = None  # None = not attempted, False = unavailable, else callable
# 【译】（行尾）None = 尚未尝试，False = 不可用，否则是可调用对象


# 取得weak-ref算子：优先用已注册的原生算子，否则JIT编译一个；结果缓存在_ATOM_WEAKREF_OP，不可用时返回None
# 调用方：下面的weak_ref_tensor
def _get_weak_ref_op():
    global _ATOM_WEAKREF_OP
    # 已经尝试过：可用就返回算子；False（不可用）经`or None`变成None
    if _ATOM_WEAKREF_OP is not None:
        return _ATOM_WEAKREF_OP or None

    # 1) Prefer a native op if this build already registered one (e.g. vLLM _C).
    # 【译】1) 优先使用本构建已经注册的原生算子（例如vLLM的_C）。
    try:
        op = torch.ops._C.weak_ref_tensor
        # Probe that it is actually callable / registered.
        # 【译】探测它确实可调用/已注册。
        op  # noqa: B018
        _ATOM_WEAKREF_OP = op
        return op
    except (AttributeError, RuntimeError):
        pass

    # 2) JIT-compile a minimal from_blob weak ref (no-op deleter => shares memory
    #    without owning the allocator block). One-time compile, then cached.
    # 【译】2) JIT编译一个最小的from_blob弱引用（空操作deleter => 共享内存但不拥有分配器的block）。
    #   只编译一次，之后使用缓存。
    try:
        import os as _os

        # ATOM_DISABLE_JIT_WEAKREF=1可以跳过JIT编译（这个变量直接读os.environ，没有登记在envs.py里）
        if _os.environ.get("ATOM_DISABLE_JIT_WEAKREF", "0") == "1":
            _ATOM_WEAKREF_OP = False
            return None
        from torch.utils.cpp_extension import load_inline

        _cpp = r"""
#include <torch/extension.h>
at::Tensor atom_weak_ref_tensor(at::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "weak_ref_tensor: input must be CUDA");
  void* data_ptr = input.data_ptr();
  auto options = at::TensorOptions()
                     .dtype(input.scalar_type())
                     .device(input.device());
  // Empty deleter: the returned tensor shares `input`'s memory but does NOT own
  // the CUDACachingAllocator block, so freeing `input` lets the pool reclaim it.
  return at::from_blob(data_ptr, input.sizes(), input.strides(),
                       [](void*) {}, options);
}
"""
        # 【译】（_cpp字符串里的C++注释）空deleter：返回的张量共享`input`的内存，但不拥有CUDACachingAllocator的block，
        #   所以释放`input`后内存池可以回收它。
        # load_inline：把上面的C++源码现场编译成扩展模块并加载，functions列出要导出给Python的函数
        _mod = load_inline(
            name="atom_weakref",
            cpp_sources=[_cpp],
            functions=["atom_weak_ref_tensor"],
            with_cuda=True,
            verbose=False,
        )
        _ATOM_WEAKREF_OP = _mod.atom_weak_ref_tensor
        return _ATOM_WEAKREF_OP
    except Exception as _e:  # noqa: BLE001
        # Any build/compile failure -> disable (return-as-is is functionally
        # correct, just uses more memory). Logged once.
        # 【译】任何构建/编译失败 -> 禁用（原样返回张量在功能上是正确的，只是更占内存）。只记一次日志。
        try:
            from aiter import logger as _logger

            _logger.warning(
                "atom weak_ref_tensor JIT compile failed (%s); PIECEWISE "
                "cudagraph pools will NOT overlay -> higher memory. Set "
                "ATOM_DISABLE_JIT_WEAKREF=1 to silence.",
                _e,
            )
        except Exception:  # noqa: BLE001, S110 - the logger itself is what failed
            # Deliberately silent: this handler wraps the *warning* above, so
            # the only thing left to report with is the thing that just broke.
            # The caller still gets the correct (unoptimized) fallback below.
            # 【译】这里故意保持沉默：这个处理器包住的是上面的*警告*，所以唯一能用来报告的，正是刚刚出错的那个东西。
            #   调用方仍会得到下面正确的（未优化的）后备结果。
            #   （except行尾）出错的正是logger本身
            pass
        _ATOM_WEAKREF_OP = False
        return None


# 返回与tensor共享显存、但不让其storage保持存活的张量；没有可用算子时原样返回（功能正确，只是更占显存）
# 调用方：utils/cuda_graph.py的CUDAGraphWrapper（录制CUDA Graph后只保留输出的弱引用，让graph内存池能在不同形状间复用）
def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor shares the same data as the original tensor but does NOT keep
    the original tensor's storage alive — essential for cudagraph memory pools to
    overlay captured outputs across shapes. Falls back to returning the tensor
    unchanged (functionally correct, more memory) if no weak-ref op is available.
    """
    # 【译】为张量创建弱引用。
    #   新张量与原张量共享同一份数据，但*不会*让原张量的storage保持存活——这对cudagraph内存池在不同形状之间
    #   覆盖复用捕获的输出至关重要。没有可用的weak-ref算子时，退回直接返回原张量（功能正确，更占内存）。
    # 空张量（numel()==0）和非张量直接原样返回
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        op = _get_weak_ref_op()
        if op is not None:
            try:
                return op(tensor)
            except (AttributeError, RuntimeError):
                return tensor
        return tensor
    else:
        return tensor


# weak_ref_tensor的批量版：支持单个张量、张量list、张量tuple
def weak_ref_tensors(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    # 【译】便捷函数：为单个张量、张量列表或张量元组创建弱引用。
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


# torch.compile过程的计数器（看到几张图、编译了几次、缓存更新了几次…），主要给测试断言编译行为用（沿用vLLM的设计）
# 全局唯一实例compilation_counter由utils/backends.py、compiler_inferface.py在编译过程中累加
@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    # including the splitting ops
    # 【译】包含切分算子
    num_piecewise_graphs_seen: int = 0
    # not including the splitting ops
    # 【译】不包含切分算子
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    # 【译】gpu_model_runner尝试触发CUDAGraph捕获的次数
    num_gpu_runner_capture_triggers: int = 0
    # Number of CUDAGraphs captured
    # 【译】已捕获的CUDAGraph数量
    num_cudagraph_captured: int = 0
    # InductorAdapter.compile calls
    # 【译】InductorAdapter.compile的调用次数
    num_inductor_compiles: int = 0
    # EagerAdapter.compile calls
    # 【译】EagerAdapter.compile的调用次数
    num_eager_compiles: int = 0
    # The number of time vLLM's compiler cache entry was updated
    # 【译】vLLM编译器缓存条目被更新的次数
    num_cache_entries_updated: int = 0
    # The number of standalone_compile compiled artifacts saved
    # 【译】保存下来的standalone_compile编译产物数量
    num_compiled_artifacts_saved: int = 0
    # Number of times a model was loaded with CompilationLevel.DYNAMO_AS_IS
    # 【译】以CompilationLevel.DYNAMO_AS_IS加载模型的次数
    dynamo_as_is_count: int = 0

    # clone：深拷贝一份当前计数
    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)

    # expect：with块结束后断言各计数的增量，例如 with compilation_counter.expect(num_graphs_seen=1): ...
    @contextmanager
    def expect(self, **kwargs):
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()


# 按"模块路径.对象名"字符串动态导入并返回对象，例如"pkg.module.ClassName"
# 调用方：utils/selector.py（选attention backend类）、async_proc.py、model_runner.py、spec_decode/drafter.py、diffusion/registry.py等
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    # 【译】按完全限定类名解析出对象。
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# Mirrored from Torch 2.11's private
# _get_dynamo_config_for_logging().clean_for_json(). The cleanup function is
# nested, so ATOM cannot reuse it while fixing the serializer below.
# 【译】照抄自Torch 2.11私有的_get_dynamo_config_for_logging().clean_for_json()。
#   那个清理函数是嵌套定义的，ATOM在修正下面的序列化器时无法复用它。
# Torch把Dynamo配置序列化成JSON写进编译指标时要跳过的配置项
_TORCH_DYNAMO_METRICS_CONFIG_BLOCKLIST = {
    "TYPE_CHECKING",
    "log_file_name",
    "verbose",
    "repro_after",
    "repro_level",
    "repro_forward_only",
    "repro_tolerance",
    "repro_ignore_non_fp",
    "same_two_models_use_fp64",
    "base_dir",
    "debug_dir_root",
    "_save_config_ignore",
    "log_compilation_metrics",
    "inject_BUILD_SET_unimplemented_TESTING_ONLY",
    "_autograd_backward_strict_mode_banned_ops",
    "reorderable_logging_functions",
    "ignore_logger_methods",
    "traceable_tensor_subclasses",
    "nontraceable_tensor_subclasses",
    "_custom_ops_profile",
}

# ATOM adds bound logger methods to this Torch 2.11 setting. Those callables are
# the only extra value excluded from the metrics JSON.
# 【译】ATOM会往这个Torch 2.11配置项里加入绑定的logger方法。这些可调用对象是指标JSON中额外排除的唯一一项。
_DYNAMO_METRICS_CONFIG_BLOCKLIST = _TORCH_DYNAMO_METRICS_CONFIG_BLOCKLIST | {
    "ignore_logging_functions"
}


# 背景：getLogger()把logger.info等日志方法放进torch._dynamo.config.ignore_logging_functions，
#   让Dynamo在被追踪的代码里遇到这些日志调用时直接忽略、不打断图；但Torch把配置序列化成JSON（编译指标日志）时，
#   遇到这些函数对象会抛TypeError
# 本函数先探测：原函数正常就不动；原函数抛TypeError时，换成跳过ignore_logging_functions的版本
# 调用方：下面的getLogger()
def _patch_torch_dynamo_metrics_config_logging() -> None:
    """Extend Torch's metrics cleanup for ATOM's callable logging ignores."""
    # 【译】扩展Torch的指标清理逻辑，以兼容ATOM放进去的可调用日志忽略项。
    config = torch._dynamo.config
    if not hasattr(config, "ignore_logging_functions"):
        return

    try:
        from torch._dynamo import utils as dynamo_utils
    except ImportError:
        return

    # 已经打过补丁（带_atom_ignore_logging_functions_patched标记）或者没有这个私有函数时，什么都不做
    original = getattr(dynamo_utils, "_get_dynamo_config_for_logging", None)
    if original is None or getattr(
        original, "_atom_ignore_logging_functions_patched", False
    ):
        return

    # 先调一次原函数：不抛TypeError说明它能正常序列化（走else直接return），不需要打补丁
    try:
        original()
    except TypeError:
        pass
    else:
        return

    def wrapped_get_dynamo_config_for_logging():
        config_dict = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in config.get_config_copy().items()
            if key not in _DYNAMO_METRICS_CONFIG_BLOCKLIST
        }
        return json.dumps(config_dict, sort_keys=True)

    # Verify the copied serializer before replacing Torch's private helper.
    # 【译】替换Torch的私有辅助函数之前，先验证复制过来的序列化器能正常工作。
    try:
        wrapped_get_dynamo_config_for_logging()
    except (TypeError, ValueError):
        return

    wrapped_get_dynamo_config_for_logging._atom_ignore_logging_functions_patched = True
    dynamo_utils._get_dynamo_config_for_logging = wrapped_get_dynamo_config_for_logging


# 初始化并返回名为"atom"的logger：第一次调用时把级别设为INFO、挂上控制台handler、设置格式，并让Dynamo忽略日志调用
#   ATOM_LOG_MORE=1时用详细格式（进程名、pid、文件行号、函数名），否则只有时间和消息
# 调用方：本文件末尾在导入时执行一次；之后各模块用logging.getLogger("atom")拿到的是同一个logger
def getLogger():
    if not logger.handlers:
        # Must match the handler level below: a logger at DEBUG feeding a
        # handler at INFO still builds a LogRecord per logger.debug() call
        # and then discards it -- 10.4% of the API server's CPU at c=2048.
        # 【译】必须与下面handler的级别一致：logger设为DEBUG、handler设为INFO时，每次logger.debug()调用
        #   仍会构造一个LogRecord，然后再丢弃——在c=2048时占了API server 10.4%的CPU。
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        from atom.utils import envs as _envs

        if _envs.ATOM_LOG_MORE:
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)01d - %(processName)s:%(process)d - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="[%(name)s %(asctime)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)
        # 让Dynamo在追踪时直接忽略这些日志方法（不引发graph break）：新版torch叫ignore_logging_functions（set），旧版叫ignore_logger_methods（tuple）
        ignored_logger_methods = {
            logging.Logger.info,
            logging.Logger.warning,
            logging.Logger.debug,
            logger.warning,
            logger.info,
            logger.debug,
        }
        if hasattr(torch._dynamo.config, "ignore_logging_functions"):
            torch._dynamo.config.ignore_logging_functions = set(
                torch._dynamo.config.ignore_logging_functions
            )
            torch._dynamo.config.ignore_logging_functions.update(ignored_logger_methods)
            _patch_torch_dynamo_metrics_config_logging()
        elif hasattr(torch._dynamo.config, "ignore_logger_methods"):
            torch._dynamo.config.ignore_logger_methods = tuple(ignored_logger_methods)

    return logger


# 模块导入时执行一次：从此任何`import atom.utils`之后，"atom" logger都已就绪
logger = getLogger()
