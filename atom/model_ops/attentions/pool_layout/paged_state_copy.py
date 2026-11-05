# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Descriptor-driven bitwise copy between segmented GPU byte streams."""

# 【译】由描述符驱动的、在分段的 GPU 字节流之间做逐位拷贝。
#
# =====================================================================================
# 【导读】paged_state_copy.py —— 状态快照 ↔ 页单元的字节搬运（第5批 5b，B级）
#   前置：为什么要把每请求状态拍成快照、存进 KV 池的"页单元"，见第4批 state_pool.py 文件开头【科普】⑤ 与
#     page_unit_checkpoint.py 文件头；锁页内存、stream 与非阻塞 H2D 见 atom/utils/__init__.py 中
#     _PAGEABLE_H2D_LIMIT 上方的"科普·锁页内存与stream"。
#   问题：源和目的都是"若干段不连续的字节"，而且切法不同——
#     源 = 活跃槽位的状态：GDN 的状态张量是 [层, 槽位, …]，一个槽位在每层各是一段（_checkpoint_slot_bases）；
#          V4 是槽位里的几段行范围（v4_pool_geometry.entry_row_runs）
#     目的 = units_per_checkpoint 个页单元，每个页单元又是若干区域：MLA 池每行（层）一段 + 每个索引层一段
#          （page_unit_geometry），MHA 池每个 (字段, 层) 一段（mha_page_unit_geometry）
#   做法（三步，前两步只在启动时做一次）：
#     ① plan_segmented_copy(源各段大小, 目的各段大小, 总字节)：只看"大小"求交，得到一串 span，
#        每个 span 完整落在一个源段和一个目的段里（与地址无关，所以同一个计划服务每一次 store / restore）
#     ② _tiling：把每个 span 切成 4 KiB 的 tile，记下"第 t 个 tile 属于哪个 span、从 span 内哪里开始"
#     ③ 每步：write_descriptor 用本步各拷贝的基址算出 (源地址, 目的地址, 长度) 行 → DescriptorStaging 经锁页内存
#        非阻塞上传 → launch_copy_descriptor 一次 launch 拷完（每个 tile 一个 program）
#   带数字走一遍：源两段 [10000, 6000]，目的两段 [8192, 8192]（如两个页单元各一个区域），共 16000 字节
#     ① span = (源段, 源内偏移, 目的段, 目的内偏移, 长度)：
#          (0, 0, 0, 0, 8192)  (0, 8192, 1, 0, 1808)  (1, 0, 1, 1808, 6000)
#     ② tile：每个 span 的 tile 数 = ceil(长度 / 4096) = [2, 1, 2] → span_of_tile = [0, 0, 1, 2, 2]，
#          tile_start = [0, 4096, 0, 0, 4096]
#     ③ 本步有 2 个 store：src_bases 形状 (2, 2)、dst_bases (2, 2) → 描述符 2 × 3 = 6 行；grid = 2 × 5 = 10 个 program
#        program 7 → 第 7 // 5 = 1 个拷贝的第 7 % 5 = 2 个 tile → span 1 → 读描述符第 1 × 3 + 1 = 4 行，拷它的 [0, 1808)
#   谁用：gdn_attn.py（Qwen3-Next / Qwen3.5 / Kimi-K3 的 execute_paged_state_copies、warmup_per_req_cache）、
#     deepseek_v4_attn.py、deepseek_v41/checkpoints.py；DescriptorStaging 由 backends.py 的
#     AttentionMetadataBuilder._checkpoint_staging 构造（行数 = 每批最多操作数 × num_spans）
#   对比nano-vllm：nano-vllm 没有状态快照，也没有这种"多段到多段"的拷贝。
# =====================================================================================

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

# 没有 triton 也能 import（CPU 上的测试只用到计划和描述符的算术）；此时 _copy_tiles_kernel 为 None，真正 launch 时才报错
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

# 一个 Triton program 拷贝的字节数（一个 tile）
_TILE_BYTES = 4096


@dataclass(frozen=True, eq=False)
class SegmentedCopyPlan:
    """Where two ordered byte streams meet, in offsets rather than addresses.

    Which source segment meets which destination segment, at what offset into
    each and for how many bytes, follows from the two streams' *sizes* alone.
    Addresses enter only when a copy is issued.

    Holding them apart is what makes a finely segmented copy affordable. A
    caller whose geometry outlives its copies — a checkpoint image is the same
    shape for the life of the pool — walks the intersection once here and then
    spends a few vector adds per copy where it used to spend a Python loop per
    span. Measured on a DeepSeek-V4 image: 0.53 us a span against about none,
    which is the difference between an image cut fine enough to save PAGE
    units and one that costs more host time than the units are worth.

    The first five arrays are parallel and one span long. `src` and `dst` name
    the roles the plan was built in, not a direction: an intersection is
    symmetric, so `write_descriptor` can read it either way and a restore
    reuses the plan its store was cut by.

    The last two are the same spans cut into tiles, which is the unit the
    kernel actually runs on. Compared by identity (`eq=False`): the fields are
    arrays, so a generated `__eq__` would raise rather than answer.
    """

    # 【译】两个有序字节流在哪里相交，用偏移而不是地址表示。
    #   哪个源段与哪个目的段相交、各自从多少偏移开始、长多少字节，只由两个流各段的"大小"决定。地址只在真正发起
    #   拷贝时才进来。
    #   把两者分开，才让细粒度分段的拷贝负担得起。一个几何比它的拷贝活得更久的调用方——checkpoint 镜像在池的整个
    #   生命周期内形状不变——在这里只求一次交集，之后每次拷贝只花几次向量加法，而以前是每个 span 一次 Python 循环。
    #   在一份 DeepSeek-V4 镜像上实测：每个 span 0.53 us 对比几乎为零，这就是"镜像能切得足够细以节省 PAGE 单元"和
    #   "它花的 host 时间比省下的单元还值钱"之间的差别。
    #   前五个数组是平行的，长度都等于 span 数。src 和 dst 指的是构建计划时的角色，而不是方向：交集是对称的，所以
    #   write_descriptor 可以按任一方向读它，恢复（restore）复用存储（store）时切出的同一个计划。
    #   最后两个是把同样这些 span 切成 tile 的结果，tile 才是 kernel 真正运行的单位。按身份比较（eq=False）：字段是
    #   数组，自动生成的 __eq__ 会抛错而不是给出答案。
    # 构造方：plan_segmented_copy；持有方：各后端的 _checkpoint_copy_plan 缓存（gdn_attn.py、deepseek_v4_attn.py、
    #   deepseek_v41/checkpoints.py）

    src_seg: np.ndarray
    src_off: np.ndarray
    dst_seg: np.ndarray
    dst_off: np.ndarray
    length: np.ndarray
    # Which span a tile belongs to, and its byte offset inside that span.
    # 【译】一个 tile 属于哪个 span，以及它在该 span 内的字节偏移。
    span_of_tile: np.ndarray
    tile_start: np.ndarray
    # The two above, per device, uploaded on first use. Fixed geometry, so one
    # upload serves every copy for the life of the plan.
    # 【译】上面两个数组按 device 缓存，第一次使用时上传。几何固定，所以一次上传就服务这个计划生命周期内的每次拷贝。
    _resident: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_spans(self) -> int:
        return int(self.length.size)

    @property
    def num_tiles(self) -> int:
        return int(self.span_of_tile.size)

    def tiling_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(span_of_tile, tile_start)` resident on `device`.

        Uploaded pageably, which synchronizes the stream — acceptable only
        because it happens once per plan and the first copy is a warmup, not a
        request. A caller that reaches this from a live batch pays the whole
        outstanding queue for it.
        """
        # 【译】驻留在 device 上的 (span_of_tile, tile_start)。
        #   用可分页内存上传，这会同步 stream——之所以可以接受，只是因为每个计划只发生一次，而且第一次拷贝是 warmup、
        #   不是真实请求。从一个活跃 batch 里走到这里的调用方，要为此付出整个待执行队列的等待。
        # （warmup 指各后端的 warmup_per_req_cache：启动时用槽位 0 往前几个页单元拷一次，把计划、上传和 Triton JIT 都提前付掉）
        tables = self._resident.get(device)
        if tables is None:
            tables = (
                torch.from_numpy(self.span_of_tile).to(device),
                torch.from_numpy(self.tile_start).to(device),
            )
            self._resident[device] = tables
        return tables

    def write_descriptor(
        self,
        out: np.ndarray,
        src_bases: np.ndarray,
        dst_bases: np.ndarray,
        *,
        forward: bool = True,
    ) -> None:
        """Fill `(copies * num_spans, 3)` int64 rows: source, destination, length.

        `src_bases` and `dst_bases` are `(copies, segments)` — one row of
        segment addresses per copy, which is where a caller's geometry enters:
        a slot's base plus a range's offset, a PAGE unit's base plus a
        region's. `out` holds the copies back to back in the same order.

        Every copy in one call goes the same way, so a caller with both
        directions makes two calls. `forward=False` copies the destination
        stream back into the source instead.

        The copies are filled in one pass rather than one at a time. At these
        sizes each of the three fills below is nearly all numpy call overhead
        — a span table is a few hundred entries — so paying it per copy is
        what made describing a batch a quarter of the whole copy path, once
        the kernel stopped being the bottleneck. Batched it is about 7x
        cheaper, and it is the same three lines with one more axis.
        """
        # 【译】填写 (copies * num_spans, 3) 个 int64 行：源、目的、长度。
        #   src_bases 和 dst_bases 的形状是 (copies, segments)——每次拷贝一行段地址，调用方的几何就从这里进来：一个槽位
        #   的基址加一段范围的偏移，一个 PAGE 单元的基址加一个区域的偏移。out 按同样顺序首尾相接地装着这些拷贝。
        #   一次调用里的所有拷贝方向相同，所以两个方向都有的调用方要调两次。forward=False 表示反过来，把目的流拷回源流。
        #   所有拷贝一遍填完，而不是一次一个。在这样的规模下，下面三次填写几乎全是 numpy 的调用开销——一张 span 表只有
        #   几百项——所以按拷贝逐个付这笔开销，曾让"描述一个 batch"占到整个拷贝路径的四分之一（在 kernel 不再是瓶颈
        #   之后）。成批做大约便宜 7 倍，而且只是同样三行多了一个轴。
        # 调用方：gdn_attn.py execute_paged_state_copies（store 用 forward=True、restore 用 forward=False，同一个计划）与
        #   warmup_per_req_cache；deepseek_v4_attn.py、deepseek_v41/checkpoints.py 同理
        # Numpy would broadcast a short `dst_bases` rather than complain, and
        # every copy in the batch would then be aimed at the first image's
        # addresses -- silent cross-request corruption, at raw pointers, with
        # the other images' checkpoint records still claiming them. This is
        # where the pointers are made, so it is where the shapes are checked;
        # `launch_copy_descriptor` only inherits them. Both sides are asked
        # the same way, down to the rank: checking only the leading axis of
        # `dst_bases` would let a 1-D one reach `dst_bases[:, dst_seg]`, which
        # raises an index error about an array the caller did not pass.
        # 【译】numpy 会对一个过短的 dst_bases 做广播而不是报错，于是 batch 里的每次拷贝都会对准第一份镜像的地址——在
        #   裸指针上悄无声息地跨请求破坏数据，而其他镜像的 checkpoint 记录还声称拥有那些地方。指针是在这里造出来的，
        #   所以形状也在这里检查；launch_copy_descriptor 只是继承它们。两边按同样方式检查，连维数都查：只检查 dst_bases
        #   的第一维，会让一个一维的 dst_bases 走到 dst_bases[:, dst_seg]，抛出一个关于调用方根本没传过的数组的下标错误。
        if (
            src_bases.ndim != 2
            or dst_bases.ndim != 2
            or len(dst_bases) != len(src_bases)
        ):
            raise ValueError(
                f"a copy needs one row of bases per copy on both sides, got "
                f"{src_bases.shape} and {dst_bases.shape}"
            )
        copies = len(src_bases)
        if out.shape != (copies * self.num_spans, 3):
            raise ValueError(
                f"a descriptor for {copies} copies of {self.num_spans} spans "
                f"must be {(copies * self.num_spans, 3)}, got {out.shape}"
            )
        # 向量化填写：rows 形状 (copies, num_spans, 3)；src_bases[:, src_seg] 用 span 的段号做"花式索引"，
        #   得到 (copies, num_spans) 的段基址，再加上段内偏移。forward=False 时只是把源 / 目的两列对调
        src_col, dst_col = (0, 1) if forward else (1, 0)
        rows = out.reshape(copies, self.num_spans, 3)
        np.add(src_bases[:, self.src_seg], self.src_off, out=rows[:, :, src_col])
        np.add(dst_bases[:, self.dst_seg], self.dst_off, out=rows[:, :, dst_col])
        rows[:, :, 2] = self.length


def plan_segmented_copy(
    src_sizes: Sequence[int],
    dst_sizes: Sequence[int],
    total_bytes: int,
) -> SegmentedCopyPlan:
    """Intersect two ordered byte streams into the spans a copy is made of."""
    # 【译】把两个有序字节流求交，得到一次拷贝由哪些 span 组成。
    # 输入：src_sizes / dst_sizes = 两个流各段的字节数（按顺序）；total_bytes = 要拷的总字节（可以比两个流短，
    #   例如镜像 image_bytes 不到 units_per_checkpoint 个整页单元，尾部的填充不拷）
    # 输出：SegmentedCopyPlan（例子见文件头）
    # 调用方：gdn_attn.py / deepseek_v4_attn.py / deepseek_v41/checkpoints.py 的 _checkpoint_copy_plan（各只算一次并缓存）
    total_bytes = int(total_bytes)
    if total_bytes < 0:
        raise ValueError("copy length must be non-negative")
    if sum(src_sizes) < total_bytes:
        raise ValueError("source segmented stream is shorter than the copy")
    if sum(dst_sizes) < total_bytes:
        raise ValueError("destination segmented stream is shorter than the copy")
    if any(size <= 0 for size in (*src_sizes, *dst_sizes)):
        raise ValueError("segmented streams cannot contain empty segments")

    # 双指针归并：每次取"源段剩余、目的段剩余、总剩余"三者的最小值作为一个 span，哪边用完就前进到下一段
    src_seg: list[int] = []
    src_off: list[int] = []
    dst_seg: list[int] = []
    dst_off: list[int] = []
    length: list[int] = []
    src_i = dst_i = 0
    src_used = dst_used = 0
    remaining = total_bytes
    while remaining:
        src_left = src_sizes[src_i] - src_used
        dst_left = dst_sizes[dst_i] - dst_used
        nbytes = min(src_left, dst_left, remaining)
        src_seg.append(src_i)
        src_off.append(src_used)
        dst_seg.append(dst_i)
        dst_off.append(dst_used)
        length.append(nbytes)
        remaining -= nbytes
        src_used += nbytes
        dst_used += nbytes
        if src_used == src_sizes[src_i]:
            src_i += 1
            src_used = 0
        if dst_used == dst_sizes[dst_i]:
            dst_i += 1
            dst_used = 0
    i64 = np.int64
    lengths = np.array(length, dtype=i64)
    span_of_tile, tile_start = _tiling(lengths)
    return SegmentedCopyPlan(
        src_seg=np.array(src_seg, dtype=i64),
        src_off=np.array(src_off, dtype=i64),
        dst_seg=np.array(dst_seg, dtype=i64),
        dst_off=np.array(dst_off, dtype=i64),
        length=lengths,
        span_of_tile=span_of_tile,
        tile_start=tile_start,
    )


def _tiling(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut the spans into tiles: which span each is in, and where it starts.

    A copy kernel wants one program per tile *that exists*. Deriving that on
    the device would need a search; deriving it here needs none, because a
    plan's spans do not change. The result is two small arrays that go to the
    device once and serve every copy the plan describes.
    """
    # 【译】把 span 切成 tile：每个 tile 在哪个 span 里、从哪里开始。
    #   拷贝 kernel 希望每个"实际存在"的 tile 一个 program。在 device 上推导这一点需要一次搜索；在这里推导则不需要，
    #   因为计划的 span 不会变。结果是两个小数组，上传 device 一次，服务这个计划描述的每次拷贝。
    # 例：长度 [8192, 1808, 6000] → 每个 span 的 tile 数 [2, 1, 2] → span_of_tile [0, 0, 1, 2, 2]、tile_start [0, 4096, 0, 0, 4096]
    counts = -(-lengths // _TILE_BYTES)
    total = int(counts.sum())
    span_of_tile = np.repeat(np.arange(lengths.size, dtype=np.int32), counts)
    # Exclusive prefix sum. Written as cumsum-minus-self rather than by
    # prepending a zero and dropping the last, which has no answer for a plan
    # with no spans at all: the prepended zero survives, and the subtraction
    # below then fails to broadcast against an empty `counts`.
    # 【译】排他前缀和。写成"cumsum 减去自身"，而不是"前面补一个 0 再去掉最后一个"，后者对一个完全没有 span 的计划
    #   无解：补上的 0 会留下来，下面的减法就无法和空的 counts 广播。
    # 例：counts [2, 1, 2] → cumsum [2, 3, 5] − counts = first [0, 2, 3]（每个 span 的第一个 tile 编号）
    first = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(first, counts)
    return span_of_tile, within * _TILE_BYTES


if triton is not None:

    @triton.jit
    def _copy_tiles_kernel(
        descriptor,
        span_of_tile,
        tile_start,
        num_tiles,
        num_spans,
        TILE: tl.constexpr,
    ):
        # One program per tile that exists. The tiling is the plan's, resident
        # on the device, so which span this tile belongs to and where it
        # starts are two loads rather than a search.
        #
        # The two counts are ordinary arguments, not `tl.constexpr`: only
        # `TILE` has to be one, for `tl.arange`. Specialising on the other two
        # would key the compiled kernel to a pool geometry, so every image
        # shape would miss the on-disk cache and pay the JIT again -- for a
        # divide the copy does not notice.
        #
        # A row is `write_descriptor`'s: source, destination, length. Triton
        # cannot read a plain module constant, so the three stay literal here
        # and the round-trip test is what holds the two ends to the same order.
        # 【译】每个实际存在的 tile 一个 program。tiling 是计划的、驻留在 device 上，所以这个 tile 属于哪个 span、从哪里
        #   开始，只需两次 load，而不是一次搜索。
        #
        #   两个计数是普通参数，不是 tl.constexpr：只有 TILE 必须是（给 tl.arange 用）。对另外两个做特化，会让编译出的
        #   kernel 绑定到某种池几何，于是每种镜像形状都会错过磁盘缓存、重新付一次 JIT——只为了一个拷贝根本察觉不到的除法。
        #
        #   一行是 write_descriptor 写的：源、目的、长度。Triton 读不了普通的模块常量，所以这三个位置在这里写成字面量，
        #   靠往返测试保证两端顺序一致。
        # grid 一维：pid = 拷贝编号 op × num_tiles + tile 编号；每个 program 搬 [start, start + TILE) ∩ [0, length) 这段字节，
        #   地址是 int64 整数，用 .to(tl.pointer_type(tl.uint8)) 转成字节指针后按字节 load / store
        pid = tl.program_id(0)
        op = pid // num_tiles
        tile = pid % num_tiles
        span = tl.load(span_of_tile + tile)
        start = tl.load(tile_start + tile)
        row = descriptor + (op * num_spans + span) * 3
        src_ptr = tl.load(row)
        dst_ptr = tl.load(row + 1)
        length = tl.load(row + 2)
        offsets = start + tl.arange(0, TILE)
        mask = offsets < length
        src = (src_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
        tl.store(dst, tl.load(src, mask=mask), mask=mask)

else:
    _copy_tiles_kernel = None


class DescriptorStaging:
    """Pinned copy-descriptor rows, one buffer per slot, each fenced on its upload.

    A non-blocking H2D reads its pinned source when the stream reaches the
    copy, not when it is enqueued. Refilling a buffer whose last upload is
    still queued would rewrite the descriptor under an earlier copy, so each
    slot waits for its own previous upload before handing its rows out again.

    Slot 0 serves `build()`; every other slot is an independent buffer, so an
    out-of-band restore never shares staging with a step. Buffers are reused
    because allocating pinned memory synchronizes; `reserve` allocates them
    before serving.
    """

    # 【译】锁页内存里的拷贝描述符行，每个"槽"一个缓冲区，每个都用它自己的上传来做栅栏（fence）。
    #   非阻塞的 H2D 是在 stream 执行到这次拷贝时才读它的锁页源数据，而不是在入队时。重新填写一个上次上传还在队列里
    #   的缓冲区，会在更早的那次拷贝底下改写描述符，所以每个槽在再次交出它的行之前，都要等自己上一次上传完成。
    #   槽 0 服务 build()；其他每个槽都是独立的缓冲区，所以带外（out-of-band）的恢复永远不会和某一步共用暂存。缓冲区
    #   被复用，因为分配锁页内存本身会同步；reserve 在开始服务之前把它们分配好。
    # 注意这里的"槽"是描述符暂存的编号（descriptor_slot），不是状态池的槽位。槽 0 = 每步 build() 里的拷贝；
    #   其他槽给 KV offload 等在步外发起的恢复（reserve_checkpoint_descriptors 预先分配）
    # 构造方：backends.py AttentionMetadataBuilder._checkpoint_staging（行数 = checkpoint_copy_ops_per_batch(max_num_seqs)
    #   × num_spans，即每条序列每步最多 1 个 store + 1 个 restore）、deepseek_v41/checkpoints.py

    def __init__(self, rows: int, device: torch.device | str) -> None:
        self._rows = rows
        self._device = torch.device(device)
        # slot -> (pinned host rows, device rows, upload fence)
        # 【译】槽号 -> (锁页的 host 行, device 行, 上传栅栏 Event)；_pending = 上传已入队、还没确认完成的槽
        self._slots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]
        ] = {}
        self._pending: set[int] = set()

    # reserve：启动时把这些槽的缓冲区先分配好（分配锁页内存会同步，不能拖到服务中途）
    #   调用方：backends.py reserve_checkpoint_descriptors、deepseek_v41/backend.py、deepseek_v41/checkpoints.py
    def reserve(self, slots: Sequence[int]) -> None:
        for slot in slots:
            self._slot(slot)

    def rows(self, slot: int = 0) -> np.ndarray:
        """This slot's host rows, safe to overwrite."""
        # 【译】这个槽的 host 侧行，可以安全覆盖。（若这个槽上一次的上传还没执行到，先在它的 Event 上等）
        host, _, uploaded = self._slot(slot)
        if slot in self._pending:
            uploaded.synchronize()
            self._pending.discard(slot)
        return host.numpy()

    def upload(self, slot: int, count: int) -> torch.Tensor:
        """Copy the first ``count`` rows to the device on the current stream."""
        # 【译】在当前 stream 上把前 count 行拷到 device。
        # non_blocking=True：从锁页内存异步拷贝，CPU 立刻返回；随后 record() 一个 Event，下次 rows() 用它判断能不能改写
        host, device, uploaded = self._slot(slot)
        descriptor = device[:count].copy_(host[:count], non_blocking=True)
        if uploaded is not None:
            uploaded.record()
            self._pending.add(slot)
        return descriptor

    # _slot：按需创建一个槽的 (host, device, Event)；CPU 设备（测试）上不用锁页内存，也没有 Event
    def _slot(
        self, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        if slot < 0:
            raise ValueError("checkpoint descriptor slot must be non-negative")
        entry = self._slots.get(slot)
        if entry is None:
            cuda = self._device.type == "cuda"
            host = torch.zeros(self._rows, 3, dtype=torch.int64, pin_memory=cuda)
            entry = self._slots[slot] = (
                host,
                torch.zeros_like(host, device=self._device),
                torch.cuda.Event() if cuda else None,
            )
        return entry


def launch_copy_descriptor(descriptor: torch.Tensor, plan: SegmentedCopyPlan) -> None:
    """Copy every span a resident descriptor names, in one launch.

    One row per span, row-major, holding several copies back to back — every
    op of a step goes out together.

    The upload is the caller's, and `descriptor` arrives on the device
    already. That is not tidiness: a pageable `torch.from_numpy(x).to(dev)`
    synchronizes the current stream, and this runs from `build()` with the
    previous forward still enqueued, so the host waits out the whole queue
    rather than the 800 KB. Measured behind 4 ms of work it cost 2.9 ms
    against 0.1 ms staged through pinned memory — a cost the transfer's own
    size says nothing about. `CpuGpuBuffer` is what the caller stages with,
    and what the rest of this repo already uses to avoid exactly this.

    The grid is one program per tile that exists. It used to be rectangular,
    `(spans, ceil(widest / TILE))`, which gives every span as many programs as
    the *widest* one needs: on a DeepSeek-V4 image, whose spans run 8 KiB to
    1.4 MB, that is 46,364 programs to do 2,631 tiles of work. One op could
    afford the waste; a batch cannot, and this path now always batches. At 256
    ops the rectangular grid measured 5.14 ms against 1.19 ms for this one.

    What makes the dense grid cheap is that the tiling is not a function of
    the copy — the plan's spans are fixed, so `span_of_tile` and `tile_start`
    are computed once and stay resident. No device-side search, and no
    `widest` for a caller to get wrong: passing one too small used to truncate
    every longer span silently, byte-correct on its prefix and stale on its
    tail.
    """
    # 【译】一次 launch 拷贝一个驻留在 device 上的描述符所列出的每个 span。
    #   每个 span 一行，行优先，几次拷贝首尾相接——一步里的所有操作一起发出。
    #   上传是调用方的事，descriptor 到这里时已经在 device 上了。这不是为了整洁：可分页的 torch.from_numpy(x).to(dev)
    #   会同步当前 stream，而这里是在 build() 中运行的、上一次 forward 还在队列里，于是 host 要等完整个队列，而不只是
    #   那 800 KB。在 4 ms 的工作后面实测，它花 2.9 ms，而通过锁页内存暂存只要 0.1 ms——这个代价和传输本身的大小毫无
    #   关系。调用方用 CpuGpuBuffer 暂存，本仓库其余地方也正是用它来避免这个问题。
    #   grid 是"每个实际存在的 tile 一个 program"。以前是矩形的 (spans, ceil(widest / TILE))，让每个 span 都得到"最宽的
    #   那个"所需的 program 数：在一份 DeepSeek-V4 镜像上，span 从 8 KiB 到 1.4 MB 不等，那就是 46,364 个 program 去做
    #   2,631 个 tile 的活。单个操作还承受得起这种浪费，一个 batch 不行，而这条路径现在总是成批的。256 个操作时，矩形
    #   grid 实测 5.14 ms，这个只要 1.19 ms。
    #   让稠密 grid 便宜的是：tiling 不是拷贝的函数——计划的 span 是固定的，所以 span_of_tile 和 tile_start 只算一次并
    #   一直驻留。不需要 device 侧搜索，也不需要调用方给出一个可能给错的 widest：以前传得太小会悄悄截断每个更长的 span，
    #   前缀字节正确、尾部却是旧数据。
    # 解释：docstring 说调用方用 CpuGpuBuffer 暂存；v0.1.8 的调用方实际用的是上面的 DescriptorStaging（同样是锁页内存 +
    #   非阻塞上传，另外按槽加了栅栏）
    # 调用方：gdn_attn.py、deepseek_v4_attn.py、deepseek_v41/checkpoints.py（staging.upload(...) 的返回值直接传进来）
    rows = descriptor.shape[0]
    if rows == 0:
        return
    if _copy_tiles_kernel is None:
        raise RuntimeError("paged state copy requires Triton")
    # Checked because this is where host arithmetic becomes device pointers:
    # the kernel reads `descriptor` as a contiguous int64 (rows, 3) and indexes
    # it by op, so a wrong shape or dtype is a wrong address rather than an
    # error.
    # 【译】在这里检查，是因为这是 host 端算术变成 device 指针的地方：kernel 把 descriptor 当作连续的 int64 (rows, 3)
    #   读取并按操作编号索引，所以形状或 dtype 错了就是地址错了，而不是报错。
    if descriptor.dtype != torch.int64 or descriptor.ndim != 2:
        raise ValueError("a copy descriptor must be a 2-D int64 tensor")
    if descriptor.shape[1] != 3 or not descriptor.is_contiguous():
        raise ValueError("a copy descriptor must be contiguous with 3 columns")
    if rows % plan.num_spans:
        raise ValueError(
            f"descriptor of {rows} rows is not a whole number of "
            f"{plan.num_spans}-span copies"
        )
    # grid = 拷贝次数 × 每次的 tile 数（例：2 次 × 5 个 = 10 个 program）
    span_of_tile, tile_start = plan.tiling_on(descriptor.device)
    _copy_tiles_kernel[(rows // plan.num_spans * plan.num_tiles,)](
        descriptor,
        span_of_tile,
        tile_start,
        plan.num_tiles,
        plan.num_spans,
        TILE=_TILE_BYTES,
        # Four warps, not more: this kernel is nothing but load and store, and
        # its speed turns out to be set by the width of one lane's access,
        # `TILE / (num_warps * 64)`. Sixteen bytes is the fast point -- a
        # 128-bit access -- and three unrelated (TILE, warps) pairs that land
        # on it measured within 0.5% of each other, while eight warps halves
        # the width and costs 12%. Raising this to fill more of the machine
        # makes it slower, so it is not a knob to turn up.
        # 【译】4 个 warp，不要更多：这个 kernel 只有 load 和 store，它的速度实际上由单个 lane 一次访问的宽度决定，即
        #   TILE / (num_warps * 64)。16 字节是最快的点——一次 128 位访问——三组互不相关、都落在这个点上的 (TILE, warps)
        #   实测彼此相差不到 0.5%，而 8 个 warp 会把宽度减半、慢 12%。为了占满更多机器而调大它只会更慢，所以这不是一个
        #   往上调的旋钮。
        # （64 是 AMD GPU 一个 wavefront 的 lane 数：4096 / (4 × 64) = 16 字节）
        num_warps=4,
    )
