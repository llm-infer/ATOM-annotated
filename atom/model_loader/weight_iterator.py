# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Reading checkpoint tensors off disk.

Split out of `loader.py`, which imports AITER at module level: the unit test
gate has no AITER build, and the shard-skipping logic here is worth covering
against real files.
"""

# 【译】从磁盘读取 checkpoint 张量。
#   从 `loader.py` 拆出来，因为后者在模块级导入了 AITER：单元测试门禁没有 AITER 构建，而这里跳过分片的逻辑值得用真实文件覆盖测试
#
# 【导读】weight_iterator.py：权重加载的第一格——把磁盘上的 *.safetensors 变成逐个的 (名字, CPU 张量)（全流程见 loader.py 文件头）
#   唯一入口 safetensors_weights_iterator：loader.py 把它作为 weights_iterator 传给 loading_core，在主线程上被 for 循环逐个拉取
#
#   前置：safetensors 文件格式
#     ┌──────────────┬────────────────────────────────────────────────┬──────────────────────────────┐
#     │ 8 字节：N     │ header：N 字节的 JSON                             │ 数据区：各张量的原始字节首尾相接 │
#     │（u64 小端）    │ {"model.layers.3.self_attn.k_proj.weight":      │   … [k_proj 的 8 MiB] …       │
#     │              │   {"dtype": "BF16", "shape": [1024, 4096],       │                              │
#     │              │    "data_offsets": [s, e]},  …,                  │                              │
#     │              │  "__metadata__": {…}}                            │                              │
#     └──────────────┴────────────────────────────────────────────────┴──────────────────────────────┘
#     data_offsets 相对数据区起点（文件偏移 8 + N）。所以只读开头一小段就知道一个分片里有哪些张量、各在哪里——
#     下面"按 header 跳过分片"和"只读想要的那一段"都靠这一点。大模型拆成很多分片（DeepSeek-V3 / R1 为 163 个），
#     model.safetensors.index.json 记录"张量名 → 分片文件"
#
#   读路径（都在主线程的生成器里，被 loading_core 的 for 循环拉动）：
#     ① 默认 mmap：safetensors.safe_open(...).get_tensor(name)——张量映射在文件上，真正读盘发生在后面工作线程 copy_ 时的缺页
#     ② ATOM_DISABLE_MMAP=true：_read_shard 用一次 readinto 把本分片里"想要的张量覆盖的那段字节"读进一块 CPU 缓冲，
#        每个张量都是它的视图（给 mmap 表现差的存储用）
#     ③ 叠加在 ①② 之上的后台预读（ATOM_LOADER_PREFETCH，默认开）：另起线程把本节点分到的分片顺序 read 一遍、丢掉数据，
#        只为把文件装进 page cache（内核缓存文件内容的内存）；之后 ① 的缺页、② 的 readinto 都直接命中内存
#     再加"按 header 跳过分片"：wants(name) 对一个分片里所有名字都为 False 时，这个分片既不预读也不读
#   数字（源码注释里的实测，本地 NVMe）：
#     缺页驱动的 mmap 读只能维持 3.2 GB/s，设备能到 6.9 GB/s，单个顺序读者就有 6.06 GB/s → 差距在访问模式，于是有了 ③
#     DeepSeek-R1 MXFP4（350 GiB，TP = 4）冷加载：约 156 s → 约 68 s（envs.py ATOM_LOADER_PREFETCH 的注释）
#     DeepSeek-V4.1-Flash，TP = 4，冷，disable_mmap：不预读 282~315 s → 预读 169~207 s → 再换成 _read_shard 的单次拷贝 84~86 s
# 对比nano-vllm：nano-vllm 只有 ①（glob 出所有 *.safetensors，safe_open 逐个 get_tensor）

import atexit
import concurrent.futures
import itertools
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Generator
from glob import glob

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `ATOM_DISABLE_MMAP=true` reader maps dtypes through `_TYPES` and would
# raise `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
# 【译】safetensors<=0.7.0 的 Python `_TYPES` 字典缺少 `F8_E8M0`（MX 格式的 scale）这一项，尽管 torch 和 safetensors 的
#   Rust 二进制都支持它。走 mmap 的 `safe_open` 路径经过 Rust，没有问题；但 `ATOM_DISABLE_MMAP=true` 的读取器通过 `_TYPES`
#   映射 dtype，会在 DeepSeek-V4-Pro 的分片上抛 `KeyError: 'F8_E8M0'`。这里补登记这个 dtype 字符串，让两条路径行为一致
#   （loader.py 开头有同样的一段，两处都幂等）
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu


_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
# header 长度的合理上限（100 MiB）：读到的长度超出就当作文件损坏，交给真正的读取器去报错


# 只读 header：返回 (header 字典, 数据区起点的文件偏移)。调用方：_shard_tensor_names、_read_shard
def _read_header(st_file: str) -> tuple[dict, int] | None:
    """A safetensors file's header and the file offset its data starts at.

    The header is a little-endian u64 byte count followed by that much JSON, so
    this costs one small read and never touches the tensor data.

    Returns None if the header cannot be read, so the caller falls back to the
    real reader and that produces the real diagnostic -- a truncated or corrupt
    file should not be reported as a JSON error from a fast path.
    """
    # 【译】一个 safetensors 文件的 header，以及它的数据区开始的文件偏移。
    #   header 是一个小端 u64 字节数，后面跟着这么多字节的 JSON，所以这只需要一次小的读取，从不碰张量数据。
    #   header 读不了时返回 None，让调用方退回到真正的读取器、由它给出真正的诊断——截断或损坏的文件不应该被一条快速路径
    #   报告成 JSON 错误
    try:
        with open(st_file, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = int.from_bytes(raw_len, "little")
            if not 0 < header_len <= _MAX_SAFETENSORS_HEADER_BYTES:
                return None
            raw_header = f.read(header_len)
            if len(raw_header) != header_len:
                return None
            header = json.loads(raw_header)
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    # __metadata__ 是文件级的元信息（如 {"format": "pt"}），不是张量
    header.pop("__metadata__", None)
    return header, 8 + header_len


def _shard_tensor_names(st_file: str) -> list[str] | None:
    """Tensor names in a safetensors file, from its header alone."""
    # 【译】一个 safetensors 文件里的张量名，只看 header 得到。调用方：_shards_worth_reading
    parsed = _read_header(st_file)
    return None if parsed is None else list(parsed[0])


# ATOM_DISABLE_MMAP=true 时的读法（调用方：_iter_shards）
#   例：一个分片的数据区里，想要的 3 个张量位于 [1000 MiB, 1200 MiB) → lo 向下对齐到 64 字节，只读这约 200 MiB，
#     一次 readinto 进 buf，三个张量都是 buf 的切片再 view 成各自的 dtype / shape（零拷贝）
#   代价：这些视图共享一块缓冲，只要还有一个张量在排队等写入，整块缓冲就不会释放
def _read_shard(
    st_file: str, wants: Callable[[str], bool] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """The wanted tensors of a shard, read with `read()` rather than mmap.

    One `readinto` of the byte span the wanted tensors cover, into a tensor
    that every result is a view of. `safetensors.torch.load(f.read())` copies
    twice -- into `bytes`, then out of it per tensor. DeepSeek-V4.1-Flash,
    TP=4, cold, prefetching: 169-207s with that, 84-86s with this.
    """
    # 【译】一个分片中想要的那些张量，用 `read()` 而不是 mmap 读出。
    #   对想要的张量所覆盖的字节区间做一次 `readinto`，读进一个张量，每个结果都是它的视图。
    #   `safetensors.torch.load(f.read())` 会拷两次——先拷进 `bytes`，再逐个张量拷出来。DeepSeek-V4.1-Flash，TP=4，冷启动，
    #   开预读：那样 169-207 秒，这样 84-86 秒
    # header 读不了：退回 safetensors 自己的读取器（整个文件读进 bytes），让它报出真正的错误或照常读出
    parsed = _read_header(st_file)
    if parsed is None:
        with open(st_file, "rb") as f:
            for name, tensor in safetensors.torch.load(f.read()).items():
                if wants is None or wants(name):
                    yield name, tensor
        return
    header, data_start = parsed
    entries = [(n, m) for n, m in header.items() if wants is None or wants(n)]
    if not entries:
        return
    # Start the span on an aligned offset, so a tensor's offset in the buffer
    # keeps the alignment it has in the data section.
    # 【译】区间起点取在对齐的偏移上，让张量在缓冲里的偏移保持它在数据区里的对齐
    # & ~63：把低 6 位清零 = 向下取到 64 的倍数；hi 取最后一个想要的张量的结尾
    lo = min(m["data_offsets"][0] for _, m in entries) & ~63
    hi = max(m["data_offsets"][1] for _, m in entries)
    # memoryview(buf.numpy())：让 readinto 直接写进 torch 张量的内存（numpy 数组与 CPU 张量共享存储），不经过 bytes 中转
    # buffering=0：不用 Python 的缓冲层，readinto 直接对应 read 系统调用；一次可能读不满，所以循环到读满为止
    buf = torch.empty(hi - lo, dtype=torch.uint8)
    view = memoryview(buf.numpy())
    with open(st_file, "rb", buffering=0) as f:
        f.seek(data_start + lo)
        done = 0
        while done < len(view):
            n = f.readinto(view[done:])
            if not n:
                raise OSError(f"{st_file}: truncated at byte {data_start + lo + done}")
            done += n
    for name, meta in entries:
        start, end = meta["data_offsets"]
        dtype = safetensors.torch._TYPES[meta["dtype"]]
        raw = buf[start - lo : end - lo]
        # safetensors' writer orders tensors so each is aligned; another
        # writer need not, and `view(dtype)` rejects a misaligned offset.
        # 【译】safetensors 的写入器会排好张量顺序让每个都对齐；别的写入器不一定，而 `view(dtype)` 不接受没对齐的偏移
        #   例：bf16 张量（itemsize 2）若落在奇数偏移，先 clone 出一份对齐的副本再 view
        if (start - lo) % dtype.itemsize:
            raw = raw.clone()
        yield name, raw.view(dtype).reshape(meta["shape"])


# 本进程在本节点内的 (rank, 进程数)，给预读分工用。调用方：_start_prefetch
#   例：单机 TP = 8 → (0..7, 8)，每个 rank 预读 1/8 的分片；两机共 16 个 rank、没有设 LOCAL_RANK 环境变量时，
#     world = 16 > 本机可见 8 张卡，判断不了节点布局 → (0, 1)：每个 rank 都预读全部分片（重复读由 page cache 吸收）
def _node_local_rank() -> tuple[int, int]:
    """This process's rank and world size *within its node*.

    Page cache is per node, so a prefetch split has to be node-local: splitting
    by global rank would hand each node only 1/world of the shards and leave
    the rest of that node's checkpoint to be faulted in on demand.

    Falls back to ``(0, 1)`` -- every rank prefetches everything -- whenever the
    node layout cannot be established. That errs the safe way: duplicate reads
    are absorbed by the shared page cache, whereas over-estimating the node
    count would silently leave part of the checkpoint unprefetched.
    """
    # 【译】本进程*在它所在节点内*的 rank 和进程数。
    #   page cache 是按节点的，所以预读的分工必须按节点内划分：按全局 rank 划分会让每个节点只分到 1/world 的分片，
    #   该节点 checkpoint 的其余部分要靠按需缺页读入。
    #   只要确定不了节点布局，就退回 ``(0, 1)``——每个 rank 预读全部。这是往安全的方向犯错：重复的读取会被共享的
    #   page cache 吸收，而高估节点数会悄悄让 checkpoint 的一部分没被预读
    # 优先用 torchrun 一类启动器设置的 LOCAL_RANK / LOCAL_WORLD_SIZE
    env_rank, env_size = os.environ.get("LOCAL_RANK"), os.environ.get(
        "LOCAL_WORLD_SIZE"
    )
    if env_rank is not None and env_size is not None:
        try:
            return int(env_rank), max(1, int(env_size))
        except ValueError:
            pass
    if not torch.distributed.is_initialized():
        return 0, 1
    world = torch.distributed.get_world_size()
    try:
        visible = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        visible = 0
    if visible and world <= visible:
        # Every rank fits on this node's GPUs, so global rank is node-local.
        # 【译】所有 rank 都能放进本节点的 GPU，所以全局 rank 就是节点内 rank
        return torch.distributed.get_rank(), world
    return 0, 1


# Set when prefetching is no longer useful -- the load finished, or the process
# is exiting. `ThreadPoolExecutor`'s workers are not daemon threads and are
# joined by `concurrent.futures`' own atexit hook, so the `daemon=True` on the
# thread below cannot by itself keep a failed load from blocking process exit
# behind an in-flight multi-GB read. The read loop has to bail out on its own.
# 【译】预读不再有用时置位——加载结束了，或进程正在退出。`ThreadPoolExecutor` 的工作线程不是守护线程，会被
#   `concurrent.futures` 自己的 atexit 钩子 join，所以下面线程上的 `daemon=True` 单靠自己并不能阻止"一次失败的加载让进程退出
#   卡在一个正在进行的几 GB 读取后面"。读取循环必须自己退出
# threading.Event：线程间共享的布尔标志，set() 置位、is_set() 查询、clear() 复位
_prefetch_stop = threading.Event()

# Registered after `concurrent.futures.thread` imported (and registered
# `_python_exit`), so LIFO ordering runs this first and the reads are already
# unwinding by the time the executor is joined.
# 【译】在 `concurrent.futures.thread` 导入（并注册了 `_python_exit`）之后才注册，所以按 atexit 后进先出的顺序，这个先执行，
#   等到 executor 被 join 时读取已经在退出了
atexit.register(_prefetch_stop.set)


def _read_whole_file(path: str, block_size: int) -> None:
    """Read `path` sequentially so the kernel caches it, discarding the data.

    Abandons the file as soon as `_prefetch_stop` is set: the pages read so far
    stay cached, and warming the rest is worthless once nobody is loading.
    """
    # 【译】顺序读一遍 `path`，让内核把它缓存起来，读到的数据直接丢掉。
    #   `_prefetch_stop` 一置位就放弃这个文件：已读的页留在缓存里，而没人加载之后再预热剩下的部分毫无价值
    # 调用方：_start_prefetch 的线程池；每次读 block_size（默认 16 MiB，ATOM_LOADER_PREFETCH_BLOCK_MB）
    with open(path, "rb") as f:
        while not _prefetch_stop.is_set() and f.read(block_size):
            pass


# 启动后台预读（调用方：safetensors_weights_iterator，ATOM_LOADER_PREFETCH 为真时）
#   例：DeepSeek-R1 163 个分片、单机 TP = 8 → rank r 预读 files[r::8]，即第 r、r+8、r+16 … 个，每个 rank 20 或 21 个；
#     每个 rank 同时最多读 4 个文件（ATOM_LOADER_PREFETCH_THREADS），读完一个再补一个（有界窗口）
#   各 rank 的加载器仍然要读全部 163 个分片（每个 rank 都要从每个张量里切出自己那一片），预读只负责让它们读的时候命中 page cache
def _start_prefetch(files: list[str], num_threads: int, block_size: int) -> None:
    """Warm the page cache for this rank's share of `files`, in the background.

    A plain sequential ``read()`` rather than ``posix_fadvise(WILLNEED)``:
    WILLNEED is a hint the kernel may drop, and for a 350 GiB checkpoint it
    drops most of it, leaving the real work to demand faults through the mmap.
    Measured on a local NVMe here, that fault pattern sustains 3.2 GB/s while
    the device does 6.9 GB/s and even a *single* sequential reader reaches
    6.06 GB/s -- so the gap is the access pattern, not queue depth, and a
    sequential reader is exactly what closes it.

    Detached on purpose: loading starts immediately and rides whatever is
    already cached instead of waiting for the whole checkpoint.
    """
    # 【译】在后台为本 rank 分到的 `files` 预热 page cache。
    #   用普通的顺序 ``read()`` 而不是 ``posix_fadvise(WILLNEED)``：WILLNEED 只是内核可以丢弃的提示，对一个 350 GiB 的
    #   checkpoint 它丢掉了大部分，真正的活又落回到经 mmap 的按需缺页。在这里的本地 NVMe 上测得：那种缺页模式只能维持
    #   3.2 GB/s，而设备能到 6.9 GB/s，就连*单个*顺序读者也能到 6.06 GB/s——所以差距在访问模式而不是队列深度，
    #   而顺序读者正好能弥补它。
    #   刻意不等它完成：加载立刻开始，借用已经缓存了的部分，而不是等整个 checkpoint 预热完
    rank, local_world = _node_local_rank()
    mine = files[rank::local_world]
    if not mine:
        return
    # A thread count of zero is the obvious way to try to switch this off, and
    # `ThreadPoolExecutor` rejects it -- from inside the detached thread below,
    # where it would surface only as a bare traceback. `ATOM_LOADER_PREFETCH`
    # is the documented off switch; treat 0 as "as few as possible" instead.
    # 【译】线程数设为 0 是想关掉它的最直观办法，但 `ThreadPoolExecutor` 不接受 0——而且是在下面那个分离的线程里报错，
    #   只会表现为一段光秃秃的 traceback。`ATOM_LOADER_PREFETCH` 才是文档里的关闭开关；这里把 0 当作"尽量少"处理
    num_threads = max(1, num_threads)
    _prefetch_stop.clear()

    def _run() -> None:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            remaining = iter(mine)
            pending = {
                pool.submit(_read_whole_file, p, block_size)
                for p in itertools.islice(remaining, num_threads)
            }
            # Bounded window: the point is to stay ahead of the loader, not to
            # queue every shard at once and compete with it for bandwidth.
            # 【译】有界窗口：目的是走在加载器前面，而不是一次把所有分片都排进队列、和加载器抢带宽
            # concurrent.futures.wait(FIRST_COMPLETED)：等到至少一个完成，返回 (已完成, 未完成) 两个集合
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    try:
                        fut.result()
                    except OSError as e:
                        # A prefetch failure costs speed, never correctness:
                        # the loader still reads the shard itself.
                        # 【译】预读失败只损失速度、不影响正确性：加载器仍会自己读这个分片
                        logger.debug("Prefetch failed: %s", e)
                    nxt = next(remaining, None)
                    if nxt is not None and not _prefetch_stop.is_set():
                        pending.add(pool.submit(_read_whole_file, nxt, block_size))
        logger.info(
            "Checkpoint prefetch finished: %d/%d shards in %.1fs",
            len(mine),
            len(files),
            time.perf_counter() - started,
        )

    logger.info(
        "Prefetching %d/%d shards into page cache in the background "
        "(node-local rank %d of %d, %d threads, %d MiB blocks)",
        len(mine),
        len(files),
        rank,
        local_world,
        num_threads,
        block_size // (1024 * 1024),
    )
    # 守护线程：主线程不等它；进程退出时靠 _prefetch_stop 让里面的读取循环尽快结束（见上面 _prefetch_stop 的注释）
    threading.Thread(target=_run, name="ckpt-prefetch", daemon=True).start()


# 按 header 过滤分片：只保留至少含一个想要的张量的分片（或 header 读不了的分片）。调用方：safetensors_weights_iterator
#   例：MTP 草稿模型从目标模型的 163 个分片里挑 MTP 那一层（V3 的 model.layers.61.*），wants 对其余张量都返回 False →
#     通常只剩少数几个分片要读；假权重时 wants 恒为 False → 一个分片都不剩（但每个分片的 header 仍要读一次）
def _shards_worth_reading(
    files: list[str], wants: Callable[[str], bool] | None
) -> list[str]:
    """Drop shards holding nothing the caller wants, by header alone.

    A drafter load reads the target's checkpoint to pick out the MTP block --
    typically one shard out of dozens -- so this is what keeps both the loader
    and the prefetcher off the other 98%.

    A shard whose header cannot be read is kept: the real reader should produce
    the real diagnostic, not a fast path whose only job is to skip files.
    """
    # 【译】只凭 header 丢掉不含调用方想要的任何东西的分片。
    #   草稿模型加载会读目标模型的 checkpoint 来挑出 MTP 块——通常是几十个分片里的一个——所以这一步让加载器和预读线程
    #   都不用碰另外 98%。header 读不了的分片会被保留：应该由真正的读取器给出真正的诊断，而不是一条只负责跳过文件的快速路径
    # any(map(wants, names))：只要有一个名字想要就保留，遇到第一个 True 就停
    if wants is None:
        return files
    kept = []
    for st_file in files:
        names = _shard_tensor_names(st_file)
        if names is None or any(map(wants, names)):
            kept.append(st_file)
    return kept


# 【主干】逐个产出 (checkpoint 名, CPU 张量)。调用方：loading_core.load_weights_into_model（经 loader.py 注入）
#   wants：loading_core 的 _wanted（改名结果非 None = 想要；假权重时恒为 False）
#   步骤：① 解析路径（hub id 先下载）→ ② 按 index.json 过滤 → ③ 按 header 丢掉无关分片 → ④（默认）排序并启动预读
#        → ⑤ 逐分片读（_iter_shards）→ ⑥ 无论正常结束还是中途异常，都让预读停下
#   生成器：函数体里有 yield，调用时不执行，被 for 循环拉一次走到下一个 yield；for 循环提前退出或抛异常时会执行 finally
def safetensors_weights_iterator(
    model_name_or_path: str,
    disable_mmap: bool = False,
    wants: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    `wants` lets the caller reject a tensor by name before it is materialized.
    Without it every tensor of every shard is built and then thrown away by the
    caller -- which is what a drafter load does, since it reads the target's
    checkpoint to pick out the MTP block and discards the other ~98%.
    """
    # 【译】遍历模型 safetensor 文件里的权重。
    #   `wants` 让调用方在张量实体化之前就按名字拒绝它。没有它的话，每个分片的每个张量都会被构建出来再被调用方扔掉——
    #   草稿模型加载正是这样：它读目标模型的 checkpoint 来挑出 MTP 块，丢掉另外约 98%
    logger.info(f"disable_mmap: {disable_mmap}")
    # ① 本地目录直接用；否则当作 HF hub id 下载 *.safetensors（跳过 original/ 目录，有些仓库在那里另放一份原始格式的权重），
    #   已在 hub 缓存里时只是解析出缓存目录（weight_utils.py）
    path = (
        model_name_or_path
        if os.path.isdir(model_name_or_path)
        else download_weights_from_hf(
            model_name_or_path, None, ["*.safetensors"], ignore_patterns=["original/*"]
        )
    )
    # ② 有 model.safetensors.index.json 时只保留索引里出现的文件——同一个目录里可能还有一份不在索引里的合并版
    #   model.safetensors，两份都读就会重复加载（weight_utils.filter_duplicate_safetensors_files）
    # ③ 按 header 丢掉没有想要张量的分片
    hf_weights_files = filter_duplicate_safetensors_files(
        glob(os.path.join(path, "*.safetensors")), path, SAFE_WEIGHTS_INDEX_NAME
    )
    hf_weights_files = _shards_worth_reading(hf_weights_files, wants)
    # 进度条只在全局 rank 0 显示（每个分片走一格）
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    # The prefetcher and the WILLNEED hint do the same job, and doing both is
    # worse than either: the hint asks the kernel to read-ahead 350 GiB of
    # random-ish ranges while the prefetcher is streaming the same files
    # sequentially, so they compete for the same device.
    #
    # Prefetching serves the whole-file reads of `disable_mmap` as well: they
    # go through the same page cache, and without it every rank streams every
    # shard itself, one shard at a time. DeepSeek-V4.1-Flash, TP=4, cold, on a
    # local NVMe: 282-315s without it, 169-207s with it.
    # 【译】预读线程和 WILLNEED 提示做的是同一件事，两者都做比只做一个更糟：提示要求内核对 350 GiB 有点随机的区间做预读，
    #   而预读线程正在顺序地流式读同样的文件，二者争抢同一个设备。
    #   预读也服务于 `disable_mmap` 的整段读取：它们走的是同一个 page cache；没有预读时，每个 rank 都要自己一个分片接一个分片地
    #   流式读完每个分片。DeepSeek-V4.1-Flash，TP=4，冷启动，本地 NVMe：不预读 282-315 秒，预读 169-207 秒
    # ④ 所以开预读时 _iter_shards 不再发 fadvise（见下面 not prefetching 的条件）
    prefetching = envs.ATOM_LOADER_PREFETCH
    if prefetching:
        # Sort only now, and read in the same order below. `glob` returns
        # directory order, which is stable enough within one run but is not a
        # promise, and the ranks must agree exactly or the stride partition
        # leaves shards unclaimed. Reading in that same order also keeps the
        # loader trailing the prefetchers instead of racing ahead of them.
        # 【译】现在才排序，并且下面按同样的顺序读。`glob` 返回的是目录顺序，在一次运行内足够稳定，但并不是承诺；
        #   而各 rank 必须完全一致，否则按步长的分工会漏掉分片。按同样的顺序读，还能让加载器跟在预读线程后面，而不是跑到它们前面
        # 注：不开预读时不排序，读取顺序就是 glob 的目录顺序（那时没有按步长的分工，各 rank 顺序不一致也无妨）
        hf_weights_files = sorted(hf_weights_files)
        _start_prefetch(
            hf_weights_files,
            envs.ATOM_LOADER_PREFETCH_THREADS,
            envs.ATOM_LOADER_PREFETCH_BLOCK_MB * 1024 * 1024,
        )

    iters = tqdm(
        hf_weights_files,
        desc=f"Loading safetensors shards[{model_name_or_path}]",
        disable=not enable_tqdm,
    )
    # ⑤ yield from：把另一个生成器产出的每一项原样转发出去；⑥ finally 让预读停下
    try:
        yield from _iter_shards(iters, disable_mmap, prefetching, wants)
    finally:
        # Whether the caller drained this or abandoned it, nothing is going to
        # read these files again, so stop warming the cache for them. Without
        # this a load that raises leaves the prefetcher streaming the rest of
        # the checkpoint and the process cannot exit until it finishes.
        # 【译】不论调用方是读完了还是中途放弃了，都不会再有人读这些文件了，所以停止为它们预热缓存。没有这一句，一次抛异常
        #   的加载会让预读线程继续流式读完 checkpoint 的剩余部分，进程要等它读完才能退出
        _prefetch_stop.set()


# 按给定顺序逐个分片读出想要的张量。调用方：safetensors_weights_iterator（iters 是包着分片列表的 tqdm 进度条）
def _iter_shards(
    iters, disable_mmap: bool, prefetching: bool, wants: Callable[[str], bool] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield every wanted tensor of each shard, in the order given."""
    # 【译】按给定的顺序，产出每个分片中每个想要的张量
    for st_file in iters:
        # Advise kernel for sequential read-ahead (mmap optimization)
        # 【译】提示内核做顺序预读（mmap 的优化）
        # 只在"没开预读线程 + ATOM_LOADER_FADVISE=true（默认 false）+ 用 mmap"时发；默认不发，原因见 envs.py 该变量的注释
        #   （实测冷加载没收益、热加载多 15.5 s）
        if (
            not prefetching
            and envs.ATOM_LOADER_FADVISE
            and not disable_mmap
            and hasattr(os, "posix_fadvise")
        ):
            try:
                fd = os.open(st_file, os.O_RDONLY)
                file_size = os.fstat(fd).st_size
                os.posix_fadvise(
                    fd,
                    0,
                    file_size,
                    os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED,
                )
                os.close(fd)
            except OSError:
                pass

        # 两条读法：disable_mmap → 整段 readinto（_read_shard）；否则 mmap：safe_open 打开、按需 get_tensor，
        #   不想要的名字连张量对象都不建
        if disable_mmap:
            yield from _read_shard(st_file, wants)
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                # `.keys()` is not redundant here: `safe_open` is a Rust object
                # with no `__iter__`, so iterating it directly raises TypeError.
                # 【译】这里的 `.keys()` 不是多余的：`safe_open` 是一个没有 `__iter__` 的 Rust 对象，直接迭代它会抛 TypeError
                for name in f.keys():  # noqa: SIM118
                    if wants is None or wants(name):
                        yield name, f.get_tensor(name)
