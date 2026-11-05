# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Batched staging for MoE expert weights.

On a large MoE checkpoint each expert's weights arrive as their own tensor, so
the per-expert `weight_loader` issues one small host-to-device copy per
(expert, shard) — tens of thousands of them, latency-bound rather than
bandwidth-bound. The pool coalesces every arrival for one fused parameter into
a CPU staging buffer and writes the result back with a single large copy.

Ownership rule: the pool writes back only the (expert slot, shard) regions it
actually staged. Any other loader path may write the same parameter as long as
it touches different regions, which is what lets a checkpoint that stores
routed experts one way and shared experts another -- Qwen3.5 BF16 stacks the
routed experts into a single tensor -- load correctly. Before writing such a
parameter, that other path calls `decline` so the two never race for a region.

Deliberately free of AITER and of `atom.config`: the pool talks to a MoE module
only through the protocol below, so it can be unit-tested on a plain CPU
runner.

    stage_expert_weight(param, staging, loaded_weight, local_expert_id,
                        shard_id, weight_name) -> bool
    expected_batched_arrivals(param) -> int | None
    _map_global_expert_id_to_local_expert_id(global_expert_id) -> int
    is_batched_expert_slot(local_expert_id) -> bool
    flush_staged(param, staging, filled) -> None
"""

# 【译】MoE 专家权重的批量暂存。
#   在大型 MoE checkpoint 里，每个专家的权重都作为独立的张量到达，于是逐专家的 `weight_loader` 为每个 (专家, 分片) 发一次小的
#   host-to-device 拷贝——成千上万次，受延迟限制而不是带宽限制。这个池子把同一个融合参数的每一次到达都合并进一个 CPU 暂存缓冲，
#   再用一次大拷贝把结果写回。
#   所有权规则：池子只写回它真正暂存过的那些 (专家槽位, 分片) 区域。任何其他加载路径都可以写同一个参数，只要碰的是不同区域——
#   正是这一点让"路由专家一种存法、共享专家另一种存法"的 checkpoint（Qwen3.5 BF16 把路由专家堆成单个张量）能正确加载。
#   在写这样的参数之前，那条路径要调 `decline`，这样两者永远不会争抢同一个区域。
#   刻意不依赖 AITER 和 `atom.config`：池子只通过上面的协议和 MoE 模块打交道，所以能在普通 CPU runner 上做单元测试
#   （协议：stage_expert_weight 把一份到达写进暂存缓冲；expected_batched_arrivals 一个参数要收几份；
#    _map_global_expert_id_to_local_expert_id 全局专家号 → 本 rank 槽位（不归本 rank 为 -1）；is_batched_expert_slot 这个槽位
#    走不走暂存；flush_staged 把暂存的区域写回参数）
#
# 【导读】expert_staging.py：loader.py 文件头【要点②】的实现。前置：MoE 专家堆成 [E, ...] 见 loader.py 文件头【前置③】
#   前置：pinned（页锁定）内存。普通（pageable）的 CPU 内存可能被操作系统换出，GPU 的 DMA 引擎不能直接读它，所以从 pageable 内存
#     做 H2D 时，驱动要先把数据拷进一块内部的 pinned 中转区再传——每次拷贝都有这份固定开销；pinned 内存则可以直接 DMA。
#     代价是分配 pinned 内存很慢（源码注释实测约 4 GB/s），而且占的是不可换出的物理内存
#   使用者：loading_core 构造一个 ExpertStagingPool（传入"参数全名 → 所属模块"的查找函数）；weight_dispatch 的 per expert 路径
#     对可暂存的参数提交 stage 任务（在工作线程里并发执行），fused / merged 路径写参数之前先 decline；loading_core 收尾时调
#     flush_pending。协议的实现方：FusedMoE（moe.py，第 7 批）。单元测试：tests/test_model_loader_staging.py
#
#   一个参数的一生（DeepSeek-V3，TP = 8，第 3 层的 w13_weight，本 rank [E_local, 512, 7168] FP8）：
#     第 1 份到达（某个工作线程）：is_batchable → 是（expected_batched_arrivals = 256 个路由专家 × w1/w3 = 512）
#       _get_or_create_entry：在池锁下发布 StagingEntry(staging=None, expected=512)
#       _ensure_staging：在条目锁下分配一块和参数同形状的 pinned CPU 缓冲（256 × 512 × 7168 字节 ≈ 0.875 GiB 的路由专家部分）并清零
#       stage_expert_weight：把本 rank 那一片拷进缓冲的 [专家, 行段]；filled 加入 (17, "w1")
#     第 2 ~ 511 份（多个线程并发）：拿到同一个条目，各写各的区域、各自加入 filled
#     第 512 份：complete → 在池锁下把条目从表里摘下（claimed）→ flush_staged：dst[:256].copy_(staging[:256])，一次 H2D
#       （融合的共享专家槽位不暂存，由 _direct_load 直接写；EPLB 冗余槽位由 flush_staged 清零，加载后再填）
#     收尾 flush_pending：还留在表里的条目 = 没收齐（checkpoint 缺专家）→ 写回已到的部分、报告
#   并发设计：池锁 _lock 只保护条目表这类小状态，持有时间极短；慢的 pinned 分配放在条目锁下做，只挡住等同一个参数的线程
# 对比nano-vllm：nano-vllm 没有 MoE，也没有并行加载，每个张量一次同步 copy_

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("atom")

# `zero_` on a packed dtype only reaches `fill_cpu` -- which has no kernel for
# it -- once the tensor is big enough for torch's parallel path (GRAIN_SIZE,
# 32768 elements). Below that it takes a memset fast path and succeeds, so a
# small probe would answer "zeroable" for every dtype and defeat the point.
# 【译】对打包 dtype 做 `zero_`，只有张量大到走 torch 的并行路径（GRAIN_SIZE，32768 个元素）时才会走到 `fill_cpu`——而它没有
#   这个 dtype 的 kernel。比这小的张量走 memset 快速路径并且成功，所以一个小探针会对每种 dtype 都回答"可清零"，失去意义
_ZERO_PROBE_NUMEL = 1 << 15


# 探测某 dtype 的 CPU 张量能否 zero_（打包的 fp4 等不能），能就按参数原 dtype 暂存，不能就按 uint8 暂存原始字节
#   @functools.cache：按参数（dtype）缓存返回值，每种 dtype 只探测一次。调用方：_allocate_staging
@functools.cache
def _cpu_zeroable(dtype: torch.dtype) -> bool:
    """Whether a staging-sized CPU tensor of `dtype` can be `zero_`d.

    Cached per dtype: a checkpoint has a handful of dtypes and tens of
    thousands of tensors, and the probe allocates (briefly) to answer.
    """
    # 【译】一个暂存大小的 `dtype` CPU 张量能不能被 `zero_`。
    #   按 dtype 缓存：一个 checkpoint 只有少数几种 dtype，却有几万个张量，而探针要（短暂地）分配一次内存才能回答
    try:
        torch.empty(_ZERO_PROBE_NUMEL, dtype=dtype, device="cpu").zero_()
    except NotImplementedError:
        return False
    except RuntimeError:
        # Some dtypes cannot even be allocated flat on CPU. Treat that the same
        # way -- raw bytes stage correctly for anything.
        # 【译】有些 dtype 甚至不能在 CPU 上平铺分配。同样处理——用原始字节暂存对任何 dtype 都正确
        return False
    return True


# 一个"正在收集中"的融合参数：缓冲 + 已到的区域集合 + 自己的锁
#   filled 的元素是 (本地专家槽位, shard_id)，例如 (17, "w1")；expected 是要收齐的份数（w13：2 × 路由专家数，w2：路由专家数）
@dataclass
class StagingEntry:
    """One in-flight fused parameter's staging buffer and its filled regions."""

    # 【译】一个在途的融合参数的暂存缓冲，以及它已填好的区域

    param: torch.nn.Parameter
    # None until the first arrival materializes it: the entry is published
    # before its buffer exists so concurrent first-arrivals for one parameter
    # share one allocation. See `ExpertStagingPool._ensure_staging`.
    # 【译】第一次到达把它实体化之前一直是 None：条目先于缓冲发布，这样同一参数的并发首次到达共用一次分配。
    #   见 `ExpertStagingPool._ensure_staging`
    staging: torch.Tensor | None
    moe: Any
    expected: int
    name: str
    filled: set[tuple[int, str]] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set by `decline` when it takes this entry off the table. A thread that
    # was mid-arrival at that moment reads it and writes its own region back,
    # because `decline`'s flush could not have covered a region that had not
    # landed yet. Written and read under the pool lock, never the entry lock.
    # 【译】`decline` 把这个条目从表上拿走时置位。那一刻正在处理到达的线程会读到它，并把自己的区域写回——因为 `decline` 的
    #   刷出不可能覆盖一个还没落下的区域。只在池锁下读写，从不在条目锁下
    disowned: bool = False

    @property
    def complete(self) -> bool:
        return len(self.filled) >= self.expected

    # 报告用，例："model.layers.3.mlp.experts.w13_weight: 510/512 (expert slot, shard) regions staged, slots 0..255"
    def missing_description(self) -> str:
        got = len(self.filled)
        slots = sorted({slot for slot, _ in self.filled})
        span = f"{slots[0]}..{slots[-1]}" if slots else "none"
        return (
            f"{self.name}: {got}/{self.expected} (expert slot, shard) regions "
            f"staged, slots {span}"
        )


@dataclass
class StagingReport:
    """What the pool still held when loading finished."""

    # 【译】加载结束时池子里还留着的东西（incomplete：每个没收齐的参数一条描述）

    incomplete: list[str] = field(default_factory=list)


class ExpertStagingPool:
    """Coalesces per-expert arrivals for a fused MoE parameter.

    `resolve_moe` maps a full parameter name to the module that owns it (or
    None); the loader supplies it because only the loader holds the model.
    """

    # 【译】为一个融合 MoE 参数合并逐专家的到达。
    #   `resolve_moe` 把参数全名映射到拥有它的模块（或 None）；由加载器提供，因为只有加载器持有模型
    # 状态（全部以 id(param) 为键）：
    #   _entries   收集中的条目            _declined  已让给别的路径、不再暂存的参数
    #   _batchable 是否可暂存的缓存判定      _alloc_*   暂存缓冲的分配统计（flush_pending 时打印一次）

    def __init__(self, resolve_moe: Callable[[str], Any]):
        self._resolve_moe = resolve_moe
        self._entries: dict[int, StagingEntry] = {}
        self._declined: set[int] = set()
        self._lock = threading.Lock()
        self._batchable: dict[int, bool] = {}
        # Staging-allocation accounting, reported once at flush time. Pinning
        # host memory runs at ~4 GB/s, so a buffer built and then dropped is
        # not free -- these counts are how that would show up.
        # 【译】暂存分配的统计，在刷出时报告一次。pin 主机内存的速度约 4 GB/s，所以一块建好又丢掉的缓冲并不免费——
        #   这几个计数就是这种浪费会显现出来的地方
        self._alloc_count = 0
        self._alloc_bytes = 0
        self._alloc_seconds = 0.0

    def is_batchable(self, param: torch.nn.Parameter, full_param_name: str) -> bool:
        """Whether arrivals for this parameter should go through the pool."""
        # 【译】这个参数的到达是否应该经过池子
        # 判定：所属模块实现了暂存协议，且 expected_batched_arrivals 返回非零（FusedMoE 对 w13/w2 的权重及部分量化格式的 scale
        #   返回份数，其他参数返回 None）。调用方：weight_dispatch._dispatch_per_expert（主线程，结果按参数缓存）
        pid = id(param)
        if pid not in self._batchable:
            moe = self._resolve_moe(full_param_name)
            expected = (
                moe.expected_batched_arrivals(param)
                if moe is not None and hasattr(moe, "stage_expert_weight")
                else None
            )
            self._batchable[pid] = bool(expected)
        return self._batchable[pid]

    # 收下一份到达（作为任务在工作线程里执行，由 weight_dispatch 提交）。流程：
    #   已 decline → 直接写参数；非本 rank 的专家 → 丢弃；不暂存的槽位（融合的共享专家）→ 直接写；
    #   否则取 / 建条目 → 确保缓冲已分配 → stage_expert_weight 写进缓冲（写不了就 decline 整个参数并直接写）
    #   → 登记 filled → 若这是最后一份（或条目已被 decline 摘走），由本线程负责写回
    def stage(
        self,
        param: torch.nn.Parameter,
        full_param_name: str,
        shard_id: str,
        global_expert_id: int,
        loaded_weight: torch.Tensor,
    ) -> None:
        """Stage one arrival, falling back to the per-expert loader if needed.

        Runs on a worker thread, concurrently with other arrivals for the same
        and for other parameters.
        """
        # 【译】暂存一份到达，必要时退回逐专家的 loader。
        #   在工作线程上运行，与同一参数和其他参数的其他到达并发
        pid = id(param)
        with self._lock:
            declined = pid in self._declined
            entry = None if declined else self._entries.get(pid)
        if declined:
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        # Map to this rank's local expert id BEFORE touching the entry table.
        # Under expert parallelism every rank iterates all global experts, but a
        # non-local expert contributes nothing to this rank's staging. If such a
        # straggler ran after the param already reached `expected` and flushed
        # (which drops its entry), creating an entry here would leave a fresh,
        # never-filled entry that is miscounted as "under-filled" at the end of
        # loading. Return early so non-local shards never create entries.
        # 【译】在碰条目表*之前*先映射成本 rank 的本地专家号。专家并行下每个 rank 都遍历全部全局专家，但非本地的专家对本 rank 的
        #   暂存没有任何贡献。如果这样一个掉队者在参数已经达到 `expected` 并刷出（刷出会删掉它的条目）之后才运行，在这里新建条目
        #   就会留下一个全新的、永远填不满的条目，在加载结束时被误算成"未填满"。提前返回，让非本地的分片永远不会创建条目
        # 例：EP = 8、256 个专家时 rank 0 只放专家 0~31，专家 100 的张量在 rank 0 上映射为 -1
        moe = self._resolve_moe(full_param_name)
        local_eid = moe._map_global_expert_id_to_local_expert_id(global_expert_id)
        if local_eid == -1:
            return

        # This slot is somebody else's (a fused shared expert), but the
        # parameter as a whole is still batchable -- do not poison it the way
        # an unstageable shard below does.
        # 【译】这个槽位属于别人（一个融合的共享专家），但这个参数整体仍是可暂存的——别像下面"无法暂存的分片"那样把它整个毒化
        #   （共享专家每层只有 3 个张量，不值得为它暂存；FusedMoE.is_batched_expert_slot 只对路由基础专家返回 True）
        if not moe.is_batched_expert_slot(local_eid):
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        if entry is None:
            entry = self._get_or_create_entry(param, moe, full_param_name)
            if entry is None:  # declined before this arrival could publish one
                self._direct_load(
                    param, full_param_name, shard_id, global_expert_id, loaded_weight
                )
                return

        # An entry read straight from the table may have been published by
        # another thread that has not finished allocating its buffer yet.
        # 【译】直接从表里读到的条目，可能是另一个线程发布的、而它还没分配完缓冲
        self._ensure_staging(entry)
        staged = moe.stage_expert_weight(
            param=param,
            staging=entry.staging,
            loaded_weight=loaded_weight,
            local_expert_id=local_eid,
            shard_id=shard_id,
            weight_name=full_param_name,
        )
        if not staged:
            # The shard shape or dtype is not one the batched path handles, and
            # that verdict holds for every arrival of this parameter, so hand
            # the whole thing over -- writing back what already landed first.
            # 【译】分片的形状或 dtype 不是批量路径能处理的，而这个结论对这个参数的每一次到达都成立，所以把整个参数交出去——
            #   先把已经落下的写回
            self.decline(param)
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        with entry.lock:
            entry.filled.add((local_eid, shard_id))
            complete = entry.complete
        # Two ways this thread owes the write-back. `claimed` is the ordinary
        # one: this arrival was the last, so it takes the entry and flushes it.
        # `disowned` is the race -- `decline` took the entry while this thread
        # was still working on it, either during the allocation above (it skips
        # the flush entirely while `staging` is None) or between the copy and
        # the `filled.add` (it flushes a `filled` set this region is missing
        # from). Either way the region is lost unless we write it back here,
        # which is safe: the path that declined writes only regions the pool
        # never staged, so the two never touch the same bytes.
        # 【译】有两种情况由本线程负责写回。`claimed` 是普通的那种：这次到达是最后一份，所以它拿走条目并刷出。`disowned` 是竞态：
        #   `decline` 在本线程还在处理时拿走了条目——要么发生在上面的分配期间（`staging` 为 None 时它整个跳过刷出），要么发生在
        #   拷贝与 `filled.add` 之间（它刷出的 `filled` 集合里缺了这个区域）。无论哪种，除非在这里写回，否则这个区域就丢了；
        #   而这样做是安全的：做 decline 的那条路径只写池子从未暂存过的区域，所以两者永远不会碰同一批字节
        # `self._entries.pop(pid, None) is entry`：只有一个线程能成功摘下，保证写回只发生一次
        with self._lock:
            claimed = complete and self._entries.pop(pid, None) is entry
            disowned = entry.disowned
        if claimed or disowned:
            with entry.lock:
                moe.flush_staged(param, entry.staging, entry.filled)

    # 让出一个参数：之后它的到达都直接写参数。调用方：weight_dispatch 的 fused expert / merged 路径（写之前），
    #   以及 stage 里遇到无法暂存的分片时
    def decline(self, param: torch.nn.Parameter) -> None:
        """Hand this parameter over to another loader path.

        Called before any write that does not go through the pool. Whatever has
        already been staged is written back first: the other path is about to
        write different regions of the same parameter, and dropping the buffer
        would silently lose the arrivals that did land.
        """
        # 【译】把这个参数交给另一条加载路径。
        #   在任何不经过池子的写入之前调用。已经暂存的部分先写回：另一条路径即将写同一参数的不同区域，而丢掉缓冲会悄悄丢失
        #   那些确实已经落下的到达
        pid = id(param)
        with self._lock:
            self._declined.add(pid)
            entry = self._entries.pop(pid, None)
            if entry is not None:
                entry.disowned = True
        # `staging is None` means the entry was published but no arrival ever
        # materialized its buffer, so there is nothing staged to write back.
        # An arrival still in flight will notice `disowned` and write its own
        # region back once it lands.
        # 【译】`staging is None` 表示条目已发布、但还没有任何到达把它的缓冲实体化，所以没有已暂存的东西需要写回。
        #   仍在途中的到达会注意到 `disowned`，等它落下后自己把区域写回
        if entry is not None and entry.staging is not None:
            with entry.lock:
                entry.moe.flush_staged(param, entry.staging, entry.filled)

    # 收尾：排空线程池之后由 loading_core 调用。正常情况下每个条目都已在最后一份到达时刷出并摘下，这里应该是空的
    def flush_pending(self) -> StagingReport:
        """Write back and drop every entry still held, after the drain.

        An entry that is still here never reached its arrival count. Its staged
        regions are written back regardless -- partial data beats none, and the
        report tells the caller exactly what is missing.
        """
        # 【译】排空之后，写回并丢弃仍持有的每个条目。
        #   还留在这里的条目从没达到它的到达数。不管怎样都写回它已暂存的区域——部分数据胜过没有，而报告会准确告诉调用方缺了什么
        with self._lock:
            pending = list(self._entries.values())
            self._entries.clear()
        report = StagingReport()
        for entry in pending:
            with entry.lock:
                if entry.staging is not None:
                    entry.moe.flush_staged(entry.param, entry.staging, entry.filled)
                report.incomplete.append(entry.missing_description())
        # One allocation per batched parameter is the floor. A count above it
        # means buffers are being built and dropped again.
        # 【译】每个批量参数一次分配是下限。数字高于它，说明缓冲在被建好又丢掉
        logger.info(
            "Staging buffers: %d allocated (%.1f GiB, %.2fs pinning)",
            self._alloc_count,
            self._alloc_bytes / 1024**3,
            self._alloc_seconds,
        )
        return report

    # ── internals ─────────────────────────────────────────────────────────

    # 先发布条目（池锁下，很快），再在锁外分配缓冲——同一参数的并发首次到达只会分配一次；已 decline 则返回 None
    def _get_or_create_entry(self, param, moe, name: str) -> StagingEntry | None:
        pid = id(param)
        with self._lock:
            if pid in self._declined:
                return None
            entry = self._entries.get(pid)
            if entry is None:
                entry = StagingEntry(
                    param=param,
                    staging=None,
                    moe=moe,
                    expected=moe.expected_batched_arrivals(param),
                    name=name,
                )
                self._entries[pid] = entry
        self._ensure_staging(entry)
        return entry

    def _ensure_staging(self, entry: StagingEntry) -> None:
        """Materialize `entry`'s buffer once, whichever thread gets here first.

        The alternative -- allocate first, publish second -- lets every
        concurrent first-arrival for one parameter build a parameter-sized
        pinned buffer so that one of them can be kept (341 of 573 buffers on
        DeepSeek-R1 MXFP4). Waiting here instead idles the other workers for
        the length of one allocation, which is only the better trade while
        something else is keeping the device fed: with `ATOM_LOADER_PREFETCH`
        off, those idle workers were the ones driving the cold read and the
        load went 154s -> 183s. With prefetch on -- the default -- the reader
        owns that job and this is worth ~5s of warm load.

        Deliberately outside the pool lock: pinning host memory runs at
        ~4 GB/s, so holding the pool lock across it would serialize every layer
        against every other. The entry's own lock only orders the threads that
        want this one parameter.
        """
        # 【译】把 `entry` 的缓冲实体化一次，哪个线程先到就由哪个来做。
        #   另一种做法——先分配、后发布——会让同一参数的每个并发首次到达都建一块参数大小的 pinned 缓冲，只为保留其中一块
        #   （DeepSeek-R1 MXFP4 上 573 块缓冲里有 341 块是这样浪费的）。改在这里等待，会让其他工作线程空闲一次分配的时长，而这只在
        #   有别的东西让设备保持忙碌时才是更好的权衡：关掉 `ATOM_LOADER_PREFETCH` 时，正是这些空闲的工作线程在驱动冷读取，
        #   加载从 154 秒变成 183 秒。开着预读时（默认）读取由预读线程负责，这样做在热加载时值约 5 秒。
        #   刻意放在池锁之外：pin 主机内存的速度约 4 GB/s，在它期间持有池锁会让每一层都和其他所有层串行。条目自己的锁只给
        #   想要这一个参数的线程排序
        # 双重检查（锁外查一次、锁内再查一次）：已分配时不用拿锁；多个线程同时到达时只有第一个真正分配
        if entry.staging is not None:
            return
        with entry.lock:
            if entry.staging is not None:
                return
            t0 = time.perf_counter()
            staging = self._allocate_staging(entry.param)
            elapsed = time.perf_counter() - t0
            entry.staging = staging
        with self._lock:
            self._alloc_count += 1
            self._alloc_bytes += staging.numel() * staging.element_size()
            self._alloc_seconds += elapsed

    # 分配一块与参数同形状、清零的 CPU 缓冲；有 GPU 时用 pinned，失败退回普通内存
    @staticmethod
    def _allocate_staging(param: torch.nn.Parameter) -> torch.Tensor:
        """A zeroed host buffer shaped like `param`.

        Zero-initialised on purpose: a slot that is only partially written (a
        padded MXFP4 shard, say) must read back as zero, matching what the
        parameter itself was initialised to.
        """
        # 【译】一块形状和 `param` 一样、已清零的主机缓冲。
        #   刻意零初始化：一个只写了一部分的槽位（比如一个带 padding 的 MXFP4 分片）必须读回零，与参数本身的初始化值一致
        # Pick the dtype before allocating. Discovering it the other way --
        # allocate, let `zero_` raise, allocate again -- costs a full pinned
        # allocation per parameter, and host pinning runs at ~4.2 GB/s cold.
        # On DeepSeek-R1 MXFP4 every routed-expert parameter is packed, so the
        # discarded first buffer was ~40% of the loader worker pool's time.
        # 【译】先选好 dtype 再分配。反过来发现它——先分配、让 `zero_` 抛异常、再分配一次——每个参数要多付一次完整的 pinned 分配，
        #   而冷启动时主机 pin 内存只有约 4.2 GB/s。在 DeepSeek-R1 MXFP4 上每个路由专家参数都是打包的，被丢掉的第一块缓冲占了
        #   加载工作线程池约 40% 的时间
        dtype = param.data.dtype if _cpu_zeroable(param.data.dtype) else torch.uint8

        def _alloc(pinned: bool) -> torch.Tensor:
            try:
                t = torch.empty(
                    param.data.shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pinned,
                )
                # Kept inside the try: `_cpu_zeroable` probes one size, and a
                # dtype it clears there could still fail here. Falling back
                # costs what this function used to cost every time, which is
                # the right price for being wrong rarely.
                # 【译】放在 try 里面：`_cpu_zeroable` 只探测一种大小，在那里通过的 dtype 在这里仍可能失败。退路的代价就是这个
                #   函数以前每次都要付的代价，对"偶尔判断错"来说是合适的价格
                t.zero_()
            except NotImplementedError:
                # Stage the raw bytes instead and let the flush re-view the
                # parameter as uint8.
                # 【译】改为暂存原始字节，让刷出时把参数重新按 uint8 来看
                t = torch.empty(
                    param.data.shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pinned,
                )
                t.zero_()
            return t

        try:
            return _alloc(torch.cuda.is_available())
        except RuntimeError as e:
            logger.warning("Pinned staging alloc failed (%s); using unpinned.", e)
            return _alloc(False)

    # 不经暂存、直接调 FusedMoE.weight_loader 写 GPU 参数（与 weight_dispatch 不可暂存时的调用相同）
    @staticmethod
    def _direct_load(
        param, full_param_name, shard_id, global_expert_id, loaded_weight
    ) -> None:
        param.weight_loader(
            param, loaded_weight, full_param_name, shard_id, global_expert_id
        )
