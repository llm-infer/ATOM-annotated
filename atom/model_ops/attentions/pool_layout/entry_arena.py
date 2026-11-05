# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What one entry of a cache class holds, and where those bytes are put.

An entry is one unit of an index space — `sub_pool_spec` sizes both pools in
these terms, so it is a block of paged KV in the PAGE pool and one request's
attention state in the STATE pool. Either way it holds several tensor
families: DeepSeek-V4's compressor keeps a `kv_state`/`score_state` pair for
each of its three flavors, GDN keeps a recurrent k and v, an MHA block keeps
k, v and their dequantization scales. Each family is an `EntryField`, whose
`shape` is what ONE (layer, entry) pair of it holds.

There is one align-place-advance walk over a field list — `field_extents` —
and the entry's own size, an arena's field offsets and a checkpoint image's
ranges are three answers to that same walk, so all three come from it and
cannot drift.

An arena is that declaration materialized. `EntryMajorArena` puts the entry
axis outermost, so entry `i` starts at `i * slot_stride` and is a contiguous
slice. That is what a per-request state wants. Its natural declaration — one
tensor per family, layer outermost and the request slot inside, `[layers,
entries, ...]` — spreads one request across as many disjoint allocations as
there are families, which is fine until something needs the state *as a
whole*. Three things do:

  - saving it as a prefix-cache checkpoint, which wants one `copy_` per range;
  - relocating it when the pool boundary moves, which needs an entry to be
    the unit of movement;
  - shipping it over RDMA, which wants one registered range per entry.

So the arena keeps the same per-layer views the kernels already take, backed
by one allocation: a per-layer view is the same shape as before with a larger
slot stride, and `entry(i)` is a contiguous slice.

The stride is the entry's own size when the arena owns its buffer. It is not
when the arena lives at the front of a slot in a shared plane — there the
entries are a slot apart and the space between them belongs to whatever else
the plane holds, so `live_entries` says which part of the index range is
really the caller's and nothing outside it is ever written.

A row space with planes of differing width cannot hold one entry contiguously
at all: a field is one strided tensor, so it lands in one plane or the other.
`plan_field_planes` decides which, `SplitEntryMajorArena` hides the split from
consumers asking for a field by name, and what stays contiguous is a *slot* —
which is the range a PD transfer registers, and the range a checkpoint's own
is carved out of by `checkpoint_ranges_for`, since an image holds only the
fields a resumer reads (`EntryField.in_checkpoint`).

The entry axis is not always the one that goes outermost, which is why the
declaration and the materializer are separate things in one module: a paged
KV pool whose blocks are still laid out layer-major reads the same field list
through a sibling arena. What picks between the two is where the entry axis
sits, not which pool the entries are drawn from. They share a file because
they are one topic — neither arena means anything without the field list it
reads, and the two differ only in that axis. Not because they have to: a
member of this package may import a sibling member, which
`tests/test_layout_packages.py` allows precisely so that a declaration and
the arithmetic over it are placed by topic rather than by import rule.

Backends stay in charge of what the fields are; this module only owns the
arithmetic. The layout is deliberately the one DeepSeek-V4's PD staging path
already builds by hand on every transfer (`_make_gather_slot`) — making it
physical is what lets that gather collapse into a copy.
"""

# 【译】一个缓存类的一个条目里装着什么，以及这些字节放在哪里。
#
#   条目是编号空间里的一个单位——sub_pool_spec 就按这个单位给两个池定大小，所以在 PAGE 池里它是一个分页
#   KV block，在 STATE 池里它是一个请求的注意力状态。不管哪种，它都装着好几族张量：DeepSeek-V4 的压缩器
#   为三种压缩各存一对 kv_state / score_state，GDN 存一份递归的 k 和 v，MHA 的一个 block 存 k、v 和它们的
#   反量化 scale。每一族是一个 EntryField，它的 shape 是"一个 (层, 条目) 对"里装的东西。
#
#   对字段列表只有一个"对齐—放置—前进"的遍历——field_extents——条目自身的大小、arena 里各字段的偏移、
#   checkpoint 镜像的范围是对同一次遍历的三个回答，所以三者都从它来，不会彼此漂移。
#
#   arena 就是把这份声明物化（真正落到一块显存上）。EntryMajorArena 把条目轴放在最外层，所以条目 i 从
#   i * slot_stride 处开始，是一段连续切片——这正是每请求状态想要的。它"自然"的声明方式——每族一个张量、
#   层在最外、请求槽位在里面，即 [layers, entries, ...]——会把一个请求分散到和族数一样多的互不相连的分配里；
#   在有东西需要把状态"作为一个整体"处理之前，这都没问题。有三件事需要：
#
#     - 把它存成前缀缓存的 checkpoint，希望每段范围一次 copy_；
#     - 池边界移动时给它搬家，需要条目是搬动的单位；
#     - 通过 RDMA 发送，希望每个条目注册一段范围。
#
#   所以 arena 保留 kernel 已经在用的那些逐层视图，但背后只有一次分配：逐层视图的形状和以前一样，只是
#   槽位步长更大，而 entry(i) 是一段连续切片。
#
#   arena 拥有自己的缓冲区时，步长就是条目自身的大小。当 arena 住在共享平面里某个槽位的前部时就不是了——
#   那时条目之间相隔一个槽位，中间的空间属于平面里的其他东西，所以由 live_entries 说明编号范围里哪一部分
#   真正属于调用方，范围外的一律不写。
#
#   平面宽度各不相同的行空间根本无法连续地容纳一个条目：一个字段是一个带步长的张量，所以它只能落在某一个
#   平面里。plan_field_planes 决定落在哪个，SplitEntryMajorArena 对按名字要字段的使用方隐藏这个拆分；
#   保持连续的是"槽位"——它是 PD 传输注册的范围，也是 checkpoint_ranges_for 从中切出 checkpoint 自己那几段
#   范围的地方，因为镜像只装续算方会读的字段（EntryField.in_checkpoint）。
#
#   条目轴并不总是放在最外层，这就是为什么声明和物化器是同一模块里分开的两样东西：block 仍按"层在外"
#   排的分页 KV 池通过一个兄弟 arena 读同一份字段列表。决定用哪一个的是条目轴放在哪，而不是条目从哪个池里
#   取。它们放在同一个文件里是因为它们是同一个主题——两个 arena 离开它们读的字段列表都毫无意义，而两者只在
#   那条轴上不同。并不是因为不得不这样：本包成员可以 import 兄弟成员，tests/test_layout_packages.py 恰恰
#   允许这一点，好让声明和作用于它的算术按主题、而不是按 import 规则来放置。
#
#   字段是什么仍由后端决定；本模块只负责算术。这个布局有意和 DeepSeek-V4 的 PD 暂存路径每次传输时手工
#   拼出来的那个一样（_make_gather_slot）——让它在物理上就是这样，那次 gather 就能退化成一次拷贝。
#
# =====================================================================================
# 【导读】entry_arena.py —— "一个条目 = 若干字段"的声明（EntryField），以及把它落到显存上的两种排法（第5批 5b，B级）
#   概念位置：pool_layout/__init__.py【科普】①②。
#   用一个小例子把全部算术走一遍：字段 A（2 层，每层 (100,) fp32 = 400 B）、字段 B（2 层，每层 (8,) bf16 = 16 B）
#     field_extents：A 占 [0, 800)；B 对齐到 256 → 占 [1024, 1056)；entry_bytes_for = 向上取整到 256 = 1280
#     EntryMajorArena（条目在外；STATE 用：一条请求的状态连续），entries = 3：
#       │←──────────── 条目0（1280 B）────────────→│←── 条目1 ──…
#       │A层0 400│A层1 400│填充 224│B层0 16│B层1 16│填充 224│A层0│A层1│…
#       A[层 l, 条目 e] 在字节 e × 1280 + l × 400；entry(1) = buf[1280 : 2560]，一次 copy_ 就是整份状态
#     LayerMajorArena（层在外；PAGE 用：一层的所有 block 连续），entries = 3：
#       │A：层0[e0│e1│e2] 层1[e0│e1│e2]（2400 B）│填充 672│B：层0[e0│e1│e2] 层1[e0│e1│e2]（96 B）│填充 672│
#       字段 f 的区域起点 = field_extents 的偏移 × entries（B：1024 × 3 = 3072）；A[l, e] 在 (l × 3 + e) × 400
#       总字节同样是 1280 × 3 = 3840——两种排法花的钱一样，只是"哪条轴连续"不同
#   谁用：
#     EntryField           所有后端声明自己的条目：mha_kv_pool.py、mla_kv_pool.py、aiter_attention.py（索引 key）、
#                          qwen4_exp_attn.py、deepseek_v4_attn.py、v4_pool_fields.py、v41_pool_geometry.py
#     LayerMajorArena      经 carve_layer_major 用于 MhaKvPool / MlaKvPool.allocate；qwen4_exp_attn.py 直接构造
#     EntryMajorArena      deepseek_v41/cache.py（PAGE 的 pages、STATE 的 state）；deepseek_v4_attn.py 经 SplitEntryMajorArena
#     plan_regions / carve ModelRunner._carve_paged_pool、aiter_attention.py、deepseek_v4_attn.py、v41_pool_geometry.py
#   对比nano-vllm：nano-vllm 只有 torch.empty(2, 层, 块, …) 一个张量；ATOM 一个条目里的字段可多可少、
#     各自 dtype 不同，所以需要"字节偏移 + as_strided 重解释"的这一层。
# =====================================================================================

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby

import torch

# Field offsets inside an entry, and `entry_bytes` itself, are rounded up to
# this. 256 B is the torch caching allocator's own granularity and a multiple
# of every element size in play, so a field pointer is always safe for the
# widest vector load a kernel might use. Real state shapes are already much
# coarser than this, so the rounding is normally free.
# 【译】条目内各字段的偏移、以及 entry_bytes 本身，都向上取整到这个值。256 B 是 torch 缓存分配器自己的粒度，
#   也是所有在用元素大小的倍数，所以字段指针对 kernel 可能用的最宽的向量化读取总是安全的。真实的状态形状
#   本来就比这粗得多，所以取整通常不额外花空间。
_ALIGN = 256


# 向上取整到 to 的倍数：-(-n // to) 就是整数的向上取整除法（Python 的 // 向负无穷取整）。例：_align_up(800) = 1024
def _align_up(n: int, to: int = _ALIGN) -> int:
    return -(-n // to) * to


def plan_regions(sizes: list[int]) -> tuple[list[int], int]:
    """Byte offsets for regions packed back to back in one allocation.

    Returns `(offsets, total)`. Both the offsets and `total` are `_ALIGN`-
    aligned, so `plan_regions(a) + plan_regions(b)` shifted by `a`'s total
    lays out exactly as `plan_regions(a + b)` would — plan groups separately
    and concatenate rather than slicing one flat result positionally. An
    empty list plans to `([], 0)`, so an absent group needs no special case.

    Lives beside the field extents because `_ALIGN` does: whoever carves an
    arena out of a shared allocation has to place every other region on the
    boundary that arena's own fields assume.
    """
    # 【译】在一次分配里首尾相接地摆放若干区域时，各区域的字节偏移。
    #   返回 (offsets, total)。offsets 和 total 都按 _ALIGN 对齐，所以 plan_regions(a) 加上"整体平移 a 的 total
    #   之后的 plan_regions(b)"，和 plan_regions(a + b) 的布局完全一样——可以分组分别规划再拼接，而不必对一个
    #   扁平结果按位置切片。空列表规划为 ([], 0)，所以缺席的组不需要特殊处理。
    #   它放在字段范围旁边是因为 _ALIGN 在这里：谁从共享分配里切出 arena，谁就得把其他每个区域都放在那个
    #   arena 自己的字段所假定的边界上。
    # 例：sizes = [1000, 300] → offsets = [0, 1024]，total = 向上取整(1324) = 1536
    # 调用方：本文件 carve；ModelRunner._carve_paged_pool（各 builder 的分页池区域）、deepseek_v4_attn.py（各平面）、
    #   v41_pool_geometry.paged_extents、plugin/vllm 的 V4 桥接
    offsets: list[int] = []
    offset = 0
    for nbytes in sizes:
        offset = _align_up(offset)
        offsets.append(offset)
        offset += nbytes
    return offsets, _align_up(offset)


def carve(buf: torch.Tensor | None, sizes: list[int]) -> list[torch.Tensor | None]:
    """`buf` cut into one region per size, placed by `plan_regions`.

    The one place a region's start is decided, so every consumer of a shared
    allocation — the runner's paged pool, a pool's field groups, a builder's
    several pools — places them the same way and none has to be told the
    offsets. `None` in, `None`s out: a pool that owns its memory carves
    nothing and lets each arena allocate.
    """
    # 【译】按 plan_regions 的摆放，把 buf 切成每个大小一段。
    #   这是唯一决定区域起点的地方，所以共享一次分配的每个使用方——runner 的分页池、一个池的各字段组、
    #   一个 builder 的多个池——都以同样方式摆放，谁都不需要被告知偏移。传入 None 就返回一串 None：
    #   自己拥有显存的池什么也不切，让每个 arena 自己分配。
    # 输入 buf：一维 uint8 张量（或 None）；输出：每段一个 buf 的切片视图（不拷贝）
    # 调用方：本文件 carve_layer_major；aiter_attention.py（不同 KvGeometry 的几个 MHA 池共用一段分页区域）
    offsets, total = plan_regions(sizes)
    if buf is None:
        return [None] * len(sizes)
    # Slicing past the end truncates rather than raising, so a short buffer
    # comes back as a short last region and surfaces as an arena complaining
    # about bytes the caller never chose.
    # 【译】切片越过末尾时会截断而不是报错，所以一个太短的缓冲区会变成一个太短的最后区域，最终表现为某个
    #   arena 抱怨一个调用方从没选过的字节数。（所以在这里先按 total 检查）
    if buf.numel() < total:
        raise ValueError(
            f"a buffer of {buf.numel()} B cannot hold {len(sizes)} regions "
            f"needing {total} B"
        )
    return [buf[start : start + size] for start, size in zip(offsets, sizes)]


def plan_field_planes(
    fields: list[EntryField], plane_row_bytes: list[int]
) -> tuple[list[list[EntryField]], int]:
    """Split fields across the planes of one row space, in the fewest rows.

    Every plane materializes the same rows at its own width, so a slot that
    reserves `r` rows offers `r * plane_row_bytes[p]` bytes in plane `p` — and
    the same `r` in all of them, because a row index has to mean one thing
    across planes. A field cannot straddle two, since its view is one strided
    tensor, so the question is which plane each one goes in.

    Returns `(per_plane_fields, rows)`. Enumerating every assignment rather
    than packing greedily: `2^len(fields)` is 64 for DeepSeek-V4 and this runs
    once at startup, so there is no reason to settle for a heuristic answer to
    a question with an exact one. Ties go to the first assignment found, which
    keeps a given field list mapping to the same layout every run.

    Field order inside a plane stays the declared order, which is the order a
    checkpoint and a PD transfer see the bytes in.
    """
    # 【译】把字段分到一个行空间的各个平面上，用的行数最少。
    #   每个平面以自己的宽度物化同样的行，所以一个预留 r 行的槽位在平面 p 里提供 r * plane_row_bytes[p] 字节
    #   ——而且所有平面里是同一个 r，因为行号在各平面间必须含义一致。一个字段不能横跨两个平面（它的视图是一个
    #   带步长的张量），所以问题是每个字段放进哪个平面。
    #   返回 (per_plane_fields, rows)。枚举所有分配方案而不是贪心装箱：对 DeepSeek-V4 来说 2^len(fields) 是 64，
    #   而且只在启动时跑一次，没理由对一个有精确解的问题满足于启发式答案。平局取最先找到的方案，保证同一份
    #   字段列表每次运行都映射到同一个布局。
    #   平面内字段的顺序保持声明顺序，也就是 checkpoint 和 PD 传输看到这些字节的顺序。
    # 例：V4 fp8 有两个平面，每行 [512 B（NoPE）, 128 B（RoPE）]；6 个压缩器状态字段 → 2^6 = 64 种方案；
    #   每种方案的行数 = max(各平面 ceil(该平面字段字节和 / 该平面行宽))，取最小者。返回的 rows 就是
    #   UnifiedPoolGeometry 的 arena_rows（槽位最前面给压缩器状态的行数）
    # 调用方：deepseek_v4_attn.py 的 builder __init__（_arena_planes, arena_rows）、plugin/vllm/deepseek_v4_bridge.py
    if not plane_row_bytes:
        raise ValueError("a row space needs at least one plane")
    num_planes = len(plane_row_bytes)
    assignments = num_planes ** len(fields)
    if assignments > 1 << 16:
        raise ValueError(
            f"refusing to enumerate {assignments} assignments of "
            f"{len(fields)} fields over {num_planes} planes"
        )

    # code 的 num_planes 进制展开就是一种分配：第 k 位 = 第 k 个字段放进哪个平面
    best: tuple[list[list[EntryField]], int] | None = None
    for code in range(assignments):
        groups: list[list[EntryField]] = [[] for _ in plane_row_bytes]
        rest = code
        for field in fields:
            groups[rest % num_planes].append(field)
            rest //= num_planes
        rows = max(
            -(-entry_bytes_for(group) // row_bytes)
            for group, row_bytes in zip(groups, plane_row_bytes)
        )
        if best is None or rows < best[1]:
            best = (groups, rows)
    assert best is not None
    return best


@dataclass(frozen=True)
class EntryField:
    """One tensor family inside an entry.

    `shape` is what ONE (layer, entry) pair holds — the same trailing shape
    the backend passes today, without the leading layer and slot dims.
    """

    # 【译】条目里的一族张量。
    #   shape 是"一个 (层, 条目) 对"装的东西——就是后端现在传的那个尾部形状，不含最前面的层维和槽位维。
    # 例：Qwen3-8B、bf16、block_size 16 的 MHA K：EntryField("k", 36, (32768,), uint8)——每层每 block 32 KiB，
    #   按字节声明，块内怎么排由 mha_kv_pool.kv_views 决定；MLA：EntryField("kv", 61, (16, 576), bf16)
    # @dataclass(frozen=True)：自动生成 __init__ / __eq__ 等，且实例不可修改

    name: str
    layers: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    # Value the field is initialized to. Score states start at -inf so an
    # unwritten ring position loses the softmax; kv states start at zero.
    # 【译】字段的初始值。score 状态从 -inf 开始，这样环里没写过的位置在 softmax 中权重为 0；kv 状态从 0 开始。
    fill: float = 0.0
    # Byte alignment the field's offset inside an entry must satisfy, over and
    # above `_ALIGN`. A field read as a plain strided tensor needs nothing more
    # than the retype boundary; one also read as rows of a *retyped plane* — a
    # window whose row is several plane rows wide — needs its offset to land on
    # one of those wider rows, or the row index the kernel computes is off by a
    # fraction of a row and nothing about the view says so.
    # 【译】字段在条目内的偏移必须满足的字节对齐，在 _ALIGN 之外额外要求。只当作普通带步长张量读取的字段，
    #   只需要"重解释类型"的边界；还要被当作"重解释过类型的平面"的行来读的字段——比如一个窗口，它的一行有
    #   好几个平面行那么宽——需要它的偏移落在那种更宽的行的边界上，否则 kernel 算出的行号会差零点几行，而
    #   视图本身看不出任何问题。
    align: int = 0
    # Whether a checkpoint image holds this field at all. A ring whose next
    # reader starts exactly at the boundary the checkpoint was taken on owes
    # one nothing: DeepSeek-V4's HCA compressor pools `[P, P + 128)` and a
    # checkpoint sits on a multiple of 128, so every row its resumer reads is
    # a row that same resumer writes. Declared by the field rather than worked
    # out at copy time, so the copy path only reads what a field says about
    # itself and the geometry stays in one place.
    #
    # All-or-nothing on purpose. A field carried in *some* of its rows is a
    # ring, and which rows those are depends on the position the checkpoint
    # was taken at — a phase this module is not given and must not guess.
    # 【译】checkpoint 镜像到底包不包含这个字段。一个环形缓冲，如果它的下一个读者恰好从 checkpoint 所在的边界
    #   开始，那么 checkpoint 不欠它任何东西：DeepSeek-V4 的 HCA 压缩器池化的是 [P, P + 128)，而 checkpoint
    #   位于 128 的倍数上，所以续算方读的每一行都是它自己会写的行。由字段自己声明，而不是在拷贝时推算，这样
    #   拷贝路径只读字段对自己的描述，几何信息集中在一个地方。
    #
    #   有意做成全有或全无。只携带"部分"行的字段就是环，而带哪几行取决于 checkpoint 拍下时所处的位置——
    #   这个相位本模块拿不到，也不该去猜。
    # 读取方：本文件 checkpoint_ranges_for（V4 的 HCA 压缩器状态、V4.1 的 compress_kv / compress_score 声明为 False）
    in_checkpoint: bool = True

    # align 只能是 _ALIGN 的倍数（更小的对齐每个字段本来就有）
    def __post_init__(self):
        if self.align and self.align % _ALIGN:
            raise ValueError(
                f"{self.name}: alignment {self.align} must be a multiple of "
                f"{_ALIGN}, which every field already has"
            )

    @property
    def per_layer_numel(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes_per_entry(self) -> int:
        """Bytes this field occupies in one entry, across all its layers."""
        # 【译】这个字段在一个条目里占的字节，所有层合计。= layers × prod(shape) × 元素字节；例：上面 MHA K 为 36 × 32768 = 1,179,648 B
        return self.layers * self.per_layer_numel * self.dtype.itemsize


def field_extents(
    fields: list[EntryField],
) -> Iterator[tuple[EntryField, int, int]]:
    """Each field with the `[start, end)` bytes it occupies in an entry.

    The one place the align-place-advance walk is written. An arena's field
    offsets, the entry's own size and a checkpoint's ranges are three answers
    to the same question and have to agree, so all three come from here.
    """
    # 【译】每个字段，连同它在条目里占的 [start, end) 字节。
    #   这是唯一写出"对齐—放置—前进"遍历的地方。arena 的字段偏移、条目自身的大小、checkpoint 的范围是同一个
    #   问题的三个答案，必须一致，所以三者都从这里来。
    # 例（文件头的小例子）：依次产出 (A, 0, 800)、(B, 1024, 1056)
    # 生成器（yield）：调用方逐个取，不预先建列表
    offset = 0
    for field in fields:
        offset = _align_up(offset, max(_ALIGN, field.align))
        yield field, offset, offset + field.bytes_per_entry
        offset += field.bytes_per_entry


def entry_bytes_for(fields: list[EntryField]) -> int:
    """Bytes one entry costs, including inter-field alignment.

    Sizing calls this before any GPU allocation exists, so it is a free
    function rather than a property of a built arena — the byte budget and
    the allocation must come from the same expression or the two drift.
    """
    # 【译】一个条目的成本（字节），含字段之间的对齐。
    #   定大小时还没有任何 GPU 分配，所以它是一个自由函数，而不是已构建 arena 的属性——字节预算和实际分配必须
    #   来自同一个表达式，否则两者会漂移。
    # 调用方：MhaKvPool / MlaKvPool（entry_bytes = 各字段组之和）、qwen4_exp_attn.py、v41_pool_geometry.py、本文件两个 arena
    end = 0
    for _, _, field_end in field_extents(fields):
        end = field_end
    return _align_up(end)


def checkpoint_ranges_for(fields: list[EntryField]) -> list[tuple[int, int]]:
    """`(offset, num_bytes)` of an entry a checkpoint image holds.

    Consecutive carried fields merge into one range, so the ordinary
    all-carried case is a single range, and the alignment padding inside a run
    rides along with it — splitting a range to shave padding costs more
    descriptor than it saves. A field left out breaks the run, which is the
    point: merging across it would put it back in the image.
    """
    # 【译】checkpoint 镜像包含的条目字节范围 (offset, num_bytes)。
    #   相邻的、都要携带的字段合并成一段，所以"全都携带"的普通情况只有一段，段内的对齐填充也一起带上——为了
    #   省掉填充而拆段，多花的描述符比省下的还多。被排除的字段会打断这一段，这正是目的：跨过它合并就会把它
    #   又放回镜像里。
    # 例：V4.1 的 STATE 字段 [window（带）, compress_kv（不带）, compress_score（不带）, cursor（带）] → 两段：window、cursor
    # 调用方：deepseek_v4_attn.py（checkpoint 的源侧范围）、v41_pool_geometry.checkpoint_image_ranges
    # itertools.groupby：把"相邻且 key 相同"的元素分成一组（这里 key = 是否携带）
    ranges: list[tuple[int, int]] = []
    for carried, run in groupby(field_extents(fields), lambda e: e[0].in_checkpoint):
        if not carried:
            continue
        extents = list(run)
        start = extents[0][1]
        nbytes = extents[-1][2] - start
        # A run of zero-byte fields spans nothing, and a zero-length range is
        # not one: `plan_segmented_copy` refuses empty segments, and it is only
        # reached on the first copy, so emitting one here would let a config
        # size, cross-check and start cleanly and then abort mid-serving.
        # 【译】一串零字节字段什么也不覆盖，而零长度的范围不算范围：plan_segmented_copy 拒绝空段，而它要到第一次
        #   拷贝时才会被调用，所以在这里产出一个空段，会让某个配置顺利定大小、通过交叉检查、正常启动，然后在服务
        #   中途中止。
        if nbytes:
            ranges.append((start, nbytes))
    return ranges


class SplitEntryMajorArena:
    """One request's state, spread over the planes of a row space.

    A row space materializes the same rows at several widths, and a field is
    one strided tensor so it cannot straddle two of them — see
    `plan_field_planes`. Consumers still want to ask for a field by name
    without knowing which plane it landed in, which is all this is.

    There is deliberately no whole-entry accessor. When the state shares a slot
    with that request's windows, the range worth copying is the slot, and the
    caller who knows the geometry takes it from the plane directly.
    """

    # 【译】一个请求的状态，分散在一个行空间的几个平面上。
    #   行空间以几种宽度物化同样的行，而一个字段是一个带步长的张量，不能横跨两个平面——见 plan_field_planes。
    #   使用方仍然想按名字要一个字段、而不必知道它落在哪个平面，这个类做的就只是这件事。
    #   有意不提供"整个条目"的访问器。当状态和该请求的窗口共用一个槽位时，值得拷贝的范围是整个槽位，由知道
    #   几何的调用方直接从平面里取。
    # 构造方：deepseek_v4_attn.py 的 allocate_per_req_cache（每个平面一个 EntryMajorArena，slot_stride = 槽位行数 × 行宽、
    #   live_entries = num_slots）、plugin/vllm/deepseek_v4_bridge.py

    def __init__(self, arenas: list[EntryMajorArena]):
        if not arenas:
            raise ValueError("a split arena needs at least one plane")
        self.arenas = list(arenas)
        self._by_field: dict[str, EntryMajorArena] = {}
        for arena in self.arenas:
            for field in arena.fields:
                if field.name in self._by_field:
                    raise ValueError(f"field {field.name!r} is in two planes")
                self._by_field[field.name] = arena

    @property
    def entry_bytes(self) -> int:
        """Bytes one request's state takes, summed over the planes."""
        # 【译】一个请求的状态占的字节，各平面求和。
        return sum(a.entry_bytes for a in self.arenas)

    def view(self, name: str) -> torch.Tensor:
        return self._by_field[name].view(name)

    def field_offset(self, name: str) -> int:
        """Bytes into the plane's slot where field `name` begins."""
        # 【译】字段 name 在它所在平面的槽位里从第几个字节开始。
        return self._by_field[name].field_offset(name)


class EntryMajorArena:
    """`entries` fixed-size entries, one stride apart.

    Exposes the per-layer views kernels expect (`view(name)` →
    `[layers, entries, *shape]`) and the whole-entry byte range that
    checkpointing, relocation and RDMA need (`entry(i)`).

    The stride between entries is `entry_bytes` when the arena owns a buffer
    of its own, and whatever the caller says when it does not — an arena
    living at the front of a slot in a shared plane is strided by the slot,
    not by its own size.
    """

    # 【译】entries 个固定大小的条目，彼此相隔一个步长。
    #   对外提供 kernel 期望的逐层视图（view(name) → [layers, entries, *shape]），以及 checkpoint、搬迁和 RDMA
    #   需要的整条目字节范围（entry(i)）。
    #   arena 拥有自己的缓冲区时，条目之间的步长就是 entry_bytes；不拥有时由调用方指定——住在共享平面里槽位
    #   前部的 arena，步长是槽位大小，而不是它自己的大小。
    # 构造方：deepseek_v41/cache.py（pages：slot_stride = page_bytes；state：slot_stride = state_bytes）、
    #   本文件 SplitEntryMajorArena 的各平面（deepseek_v4_attn.py）
    # 参数：fields = 字段列表；entries = 条目数；buf = 可选的外部 uint8 缓冲（必须已清零、连续、起点对齐）；
    #   slot_stride = 条目间距（默认 entry_bytes）；live_entries = 从末尾数、真正归调用方的条目数（默认全部）

    def __init__(
        self,
        fields: list[EntryField],
        entries: int,
        device,
        buf: torch.Tensor | None = None,
        slot_stride: int | None = None,
        live_entries: int | None = None,
    ):
        # No fields is legal here and not in `LayerMajorArena`, because "empty"
        # differs: a layer-major group is a region a model may not want, and
        # `carve_layer_major` drops it to None; a plane is addressed by index,
        # still costs its rows, and `plan_field_planes` empties one whenever
        # the fields fit in fewer.
        # 【译】这里允许没有字段，而 LayerMajorArena 不允许，因为"空"的含义不同：layer-major 的一个组是模型可能
        #   不需要的区域，carve_layer_major 会把它变成 None；而平面是按下标寻址的，仍然要占它的行，并且只要字段能
        #   装进更少的平面，plan_field_planes 就会让某个平面变空。
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")

        self.fields = list(fields)
        self.entries = entries
        self.entry_bytes = entry_bytes_for(fields)
        self.slot_stride = self.entry_bytes if slot_stride is None else slot_stride
        if self.slot_stride < self.entry_bytes:
            raise ValueError(
                f"slot stride {self.slot_stride} is under the "
                f"{self.entry_bytes} an entry takes"
            )
        # Every entry repeats the first entry's field offsets, so whatever
        # alignment the strictest field asks for has to survive the stride and
        # the buffer's own start as well — otherwise only entry 0 satisfies it.
        # 【译】每个条目都重复第一个条目的字段偏移，所以最严格的字段要求的对齐，也必须在步长和缓冲区自身的起点上
        #   成立——否则只有条目 0 满足。
        self._align = max([_ALIGN] + [f.align for f in self.fields])
        if self.slot_stride % self._align:
            raise ValueError(
                f"slot stride {self.slot_stride} must be a multiple of "
                f"{self._align}, or entries past the first fall off the boundary "
                "their field views retype from"
            )
        # Entries the caller will actually hand out, counted from the END:
        # a pool that grows takes the next index down, so the top of the range
        # is the part that is live. The rest is addressable but belongs to
        # whatever else shares the buffer, and must not be written here.
        # 【译】调用方真正会分发出去的条目数，从"末尾"数起：池增长时取的是往下的下一个编号，所以编号范围的顶部才是
        #   活的部分。其余部分可寻址，但属于共享这个缓冲区的其他东西，这里不能写。
        #   （对应 V4 的 UnifiedPoolGeometry.physical_slot：组 0 在最顶端，新组往下长，见 v4_pool_geometry.py）
        self.live_entries = entries if live_entries is None else live_entries
        if not 0 <= self.live_entries <= entries:
            raise ValueError(f"live_entries {self.live_entries} outside 0..{entries}")

        self._offsets = {f.name: start for f, start, _ in field_extents(self.fields)}

        self._by_name = {f.name: f for f in self.fields}
        # Zeroed, not `empty`: alignment padding falls outside every field
        # view, and an entry is copied whole by checkpointing and RDMA, so
        # uninitialized padding would travel. One memset at startup.
        #
        # `buf` lets a caller carve the arena out of a larger allocation it also
        # carves the paged pools from, so the two are one contiguous region
        # whose internal boundary can move. It must already be zeroed for the
        # same reason. Owning the allocation stays the default — the tests and
        # any single-pool backend construct arenas standalone.
        # 【译】用 zeros 而不是 empty：对齐填充落在所有字段视图之外，而 checkpoint 和 RDMA 会整条目拷贝，未初始化的
        #   填充就会被一起带走。启动时一次 memset 而已。
        #
        #   buf 让调用方可以从一块更大的分配里切出 arena，这块分配同时也切出分页池，于是两者是一段连续区域，内部
        #   边界可以移动。出于同样理由，它必须已经清零。默认仍是自己拥有分配——测试和任何单池后端都单独构造 arena。
        # want：最后一个条目只需要 entry_bytes，不需要一整个步长，所以是 (entries − 1) × slot_stride + entry_bytes
        want = (entries - 1) * self.slot_stride + self.entry_bytes if entries else 0
        if buf is None:
            self.buf = torch.zeros(want, dtype=torch.uint8, device=device)
        else:
            if buf.dtype is not torch.uint8 or buf.numel() < want:
                raise ValueError(
                    f"buf must hold at least {want} uint8 elements, got "
                    f"{buf.numel()} {buf.dtype}"
                )
            if not buf.is_contiguous():
                raise ValueError("buf must be contiguous")
            if buf.storage_offset() % self._align:
                raise ValueError(
                    f"buf must start on a {self._align}B boundary, got storage "
                    f"offset {buf.storage_offset()}: field views retype the "
                    "buffer, which needs the offset to divide every itemsize"
                )
            self.buf = buf
        # 只给"活的"那些条目（编号范围的顶部 live_entries 个）填初值；其余不归本 arena，不碰
        for field in self.fields:
            self.view(field.name)[:, entries - self.live_entries :].fill_(field.fill)

    @property
    def total_bytes(self) -> int:
        """Bytes the entries span, from the first to the end of the last.

        Equal to `entries * entry_bytes` only when the arena owns its buffer;
        with a wider slot stride the gaps between entries belong to whoever
        else shares it, and this counts them.
        """
        # 【译】条目覆盖的字节数，从第一个条目到最后一个条目的末尾。
        #   只有 arena 拥有自己的缓冲区时才等于 entries * entry_bytes；槽位步长更宽时，条目之间的空隙属于共享它的
        #   其他东西，这里也把它们算进去。
        return (self.entries - 1) * self.slot_stride + self.entry_bytes

    def view(self, name: str) -> torch.Tensor:
        """`[layers, entries, *shape]` — a drop-in for the standalone tensor.

        Only the slot stride differs from a standalone allocation: it is the
        whole entry rather than this field alone. Kernels that take the slot
        stride as an argument (both V4 compressor kernels do) are unaffected;
        one that assumes contiguity is not, and has to be checked.
        """
        # 【译】[layers, entries, *shape]——可以直接替换原来那个独立张量。
        #   和独立分配相比只有槽位步长不同：它是整个条目，而不是只有这个字段。把槽位步长作为参数传入的 kernel
        #   （V4 的两个压缩器 kernel 都是）不受影响；假定连续的 kernel 则不然，需要检查。
        # torch.as_strided(size, stride, storage_offset)：按显式给出的各维大小和步长（单位是元素）造一个视图，不拷贝。
        #   文件头例子里 A 的视图：size (2, 3, 100)，stride (100, 1280/4 = 320, 1)，偏移 0
        field = self._by_name[name]
        itemsize = field.dtype.itemsize
        # `as_strided`'s storage_offset is ABSOLUTE, so `typed`'s own offset
        # has to be added: omit it and a carved arena addresses from the front
        # of the host allocation and writes through whatever precedes it. An
        # owned buffer sits at offset 0, which is what hides this.
        #
        # Byte offsets convert to element offsets by plain division: `_ALIGN`
        # is a multiple of every itemsize, which is what makes that exact.
        # 【译】as_strided 的 storage_offset 是"绝对"的（相对底层 storage 的起点），所以必须加上 typed 自己的偏移：
        #   漏掉它，一个切出来的 arena 就会从宿主分配的最前面开始寻址，写穿它前面的任何东西。自己拥有的缓冲区
        #   偏移是 0，正是这一点把问题掩盖了。
        #
        #   字节偏移换成元素偏移只需普通除法：_ALIGN 是所有元素大小的倍数，这保证了除法是精确的。
        # inner：shape 内部各维的行优先步长（最后一维为 1），例 shape (16, 576) → (576, 1)
        typed = self.buf.view(field.dtype)
        inner: tuple[int, ...] = ()
        acc = 1
        for dim in reversed(field.shape):
            inner = (acc,) + inner
            acc *= dim
        return typed.as_strided(
            (field.layers, self.entries) + field.shape,
            (field.per_layer_numel, self.slot_stride // itemsize) + inner,
            typed.storage_offset() + self._offsets[field.name] // itemsize,
        )

    def entry(self, index: int) -> torch.Tensor:
        """One entry's whole state as a contiguous 1-D uint8 slice."""
        # 【译】一个条目的全部状态，作为一段连续的一维 uint8 切片（checkpoint / 搬迁 / RDMA 的单位）。
        start = index * self.slot_stride
        return self.buf[start : start + self.entry_bytes]

    def field_offset(self, name: str) -> int:
        """Byte offset of a field from the start of an entry."""
        # 【译】字段相对条目起点的字节偏移。
        return self._offsets[name]


def carve_layer_major(
    groups: list[list[EntryField]],
    entries: int,
    device,
    buf: torch.Tensor | None = None,
) -> list[LayerMajorArena | None]:
    """One layer-major arena per field group, packed into one allocation.

    A paged pool is several groups that are laid out apart but bought together
    -- an MHA block's cache, its scales and an indexer's keys. `plan_regions`
    places them, so each starts on the boundary its own field views retype
    from, and an empty group gets `None` instead of an arena over nothing.

    Every group's `entry_bytes_for` is aligned already, so packing adds no
    padding and the regions come to exactly what a caller summing the same
    groups was charged. `buf` is None for a pool that owns its memory.
    """
    # 【译】每个字段组一个 layer-major arena，打包进一次分配。
    #   一个分页池是若干个"分开摆放、但一起购买"的组——MHA block 的缓存、它的 scale、索引器的 key。plan_regions
    #   负责摆放，于是每个组都从它自己的字段视图重解释类型所需的边界开始；空组得到 None，而不是一个什么都不装
    #   的 arena。
    #   每个组的 entry_bytes_for 本身已经对齐，所以打包不会再添填充，各区域加起来正好是调用方对同样这些组求和
    #   时被收取的费用。自己拥有显存的池传 buf=None。
    # 例：MhaKvPool 的三组 [cache(k, v), scale(k_scale, v_scale), extra] → bf16 时 scale 组为空 → [arena, None, None 或 arena]
    # 调用方：MhaKvPool.allocate、MlaKvPool.allocate（[cache, index] 两组）
    sizes = [entry_bytes_for(group) * entries for group in groups]
    return [
        LayerMajorArena(group, entries, device, buf=region) if group else None
        for group, region in zip(groups, carve(buf, sizes))
    ]


class LayerMajorArena:
    """The same fields with the layer axis outermost instead of the entry axis.

    A field's whole region comes first with the layer axis inside it, so field
    `f` begins at `field_extents`' offset for it *times* `entries` — the same
    walk, scaled by the axis that moved outside.

    This is the shape a paged KV pool already has: `[layers, blocks, ...]` per
    tensor, one layer's slice contiguous, which several attention kernels
    assume. The cost is that an *entry* is not contiguous — one block's k, v
    and scales sit `entries` apart — so there is no `entry(i)` here and nothing
    can copy or register a block as a unit. That absence is why this is not
    `EntryMajorArena`, and it is the deletion condition: turn the pool
    block-major and that class serves both, `entry(i)` included.

    Unlike `EntryMajorArena` it fills only the allocation it owns; a `buf`
    handed in may already hold KV, and an IPC-imported one does. Handing one in
    also checks the declaration against it: too few bytes, or padding the
    allocation does not have, is refused rather than addressed past.
    """

    # 【译】同样的字段，但最外层是层轴而不是条目轴。
    #   一个字段的整个区域排在一起、层轴在其内部，所以字段 f 起始于 field_extents 给它的偏移"乘以" entries——
    #   同一次遍历，按移到外面的那条轴放大。
    #   这就是分页 KV 池原本的形状：每个张量 [layers, blocks, ...]，一层的切片连续，好几个注意力 kernel 都这样
    #   假定。代价是"条目"不连续——一个 block 的 k、v 和 scale 相隔 entries 那么远——所以这里没有 entry(i)，
    #   也没有东西能把一个 block 当作整体来拷贝或注册。正是这个缺失让它不能是 EntryMajorArena，这也是它的删除
    #   条件：把池改成 block 在外（block-major），那个类就能同时服务两者，包括 entry(i)。
    #   和 EntryMajorArena 不同，它只填充自己拥有的分配；传进来的 buf 可能已经装着 KV（IPC 导入的就是）。传入
    #   buf 时还会拿声明核对它：字节不够、或者分配里没有声明所需的对齐填充，都会被拒绝，而不是越界寻址。
    # 所以 MHA 池里"一个 block 的全部字节"散落在每个字段每一层各一段：页单元 checkpoint 要按这些段逐段拷贝
    #   （mha_page_unit_geometry.py）；PD 传输也按 (字段, 层) 注册区域（MhaKvPool.region_tensors）

    def __init__(
        self,
        fields: list[EntryField],
        entries: int,
        device,
        buf: torch.Tensor | None = None,
    ):
        if not fields:
            raise ValueError("a layer-major arena needs at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")

        self.fields = list(fields)
        self.entries = entries
        self.entry_bytes = entry_bytes_for(fields)
        self._align = max([_ALIGN] + [f.align for f in self.fields])
        self._offsets = {f.name: start for f, start, _ in field_extents(self.fields)}
        self._by_name = {f.name: f for f in self.fields}

        want = self.entry_bytes * entries
        if buf is None:
            self.buf = torch.zeros(want, dtype=torch.uint8, device=device)
            for field in self.fields:
                self.view(field.name).fill_(field.fill)
        else:
            if buf.dtype is not torch.uint8 or buf.numel() < want:
                raise ValueError(
                    f"buf must hold at least {want} uint8 elements, got "
                    f"{buf.numel()} {buf.dtype}"
                )
            if not buf.is_contiguous():
                raise ValueError("buf must be contiguous")
            if buf.storage_offset() % self._align:
                raise ValueError(
                    f"buf must start on a {self._align}B boundary, got storage "
                    f"offset {buf.storage_offset()}: field views retype the "
                    "buffer, which needs the offset to divide every itemsize"
                )
            # Refused and not applied: the buffer may be an imported pool
            # already holding the peer's KV, which the arena cannot tell from a
            # fresh one. Refused and not dropped either -- a `-inf` score plane
            # arriving as 0.0 turns "never selected" into "always".
            # 【译】拒绝而不是照做：缓冲区可能是一个已经装着对端 KV 的导入池，arena 无法把它和一个新池区分开。也不是
            #   悄悄忽略——一个本该是 -inf 的 score 平面以 0.0 到达，会把"永远不被选中"变成"总是被选中"。
            unfillable = [f.name for f in self.fields if f.fill]
            if unfillable:
                raise ValueError(
                    f"fields {unfillable} declare a non-zero fill, which an "
                    "arena over a caller's buffer cannot apply; give the "
                    "caller the fill before declaring one"
                )
            self.buf = buf

    @property
    def total_bytes(self) -> int:
        """Bytes the whole arena spans."""
        # 【译】整个 arena 覆盖的字节数。
        return self.entry_bytes * self.entries

    def view(self, name: str) -> torch.Tensor:
        """`[layers, entries, *shape]` — the same signature `EntryMajorArena`
        gives, and contiguous per layer, which is what the paged path binds."""
        # 【译】[layers, entries, *shape]——和 EntryMajorArena 给出的签名相同，并且每层连续，这正是分页路径绑定的形状。
        # 步长 (entries × 每层元素数, 每层元素数, inner…)：先跨层、再跨 block；起点 = 字段偏移 × entries
        field = self._by_name[name]
        itemsize = field.dtype.itemsize
        # `as_strided`'s storage_offset is ABSOLUTE, so `typed`'s own offset has
        # to be added -- omit it and a carved arena addresses from the front of
        # the host allocation and writes through whatever precedes it.
        # 【译】as_strided 的 storage_offset 是"绝对"的，所以必须加上 typed 自己的偏移——漏掉它，一个切出来的 arena
        #   就会从宿主分配的最前面开始寻址，写穿它前面的任何东西。
        typed = self.buf.view(field.dtype)
        inner: tuple[int, ...] = ()
        acc = 1
        for dim in reversed(field.shape):
            inner = (acc,) + inner
            acc *= dim
        return typed.as_strided(
            (field.layers, self.entries) + field.shape,
            (self.entries * field.per_layer_numel, field.per_layer_numel) + inner,
            typed.storage_offset()
            + self._offsets[field.name] * self.entries // itemsize,
        )
