# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where every DeepSeek-V4 KV row lives, in rows rather than bytes.

V4 keeps two kinds of KV in the same pool: compressed history addressed by
block, and a per-request sliding window addressed by slot. They used to be two
regions of one per-layer tensor, which made the split between them a startup
constant — shrinking the block count moves the start of all 46 layers, so
neither kind can ever give space back to the other.

This module states the layout as a single **row space** that both kinds share,
and lets each pool materialize it at its own row width. A row is one token's KV
at one layer; what a row *costs* differs per plane (512 B packed fp8 NoPE,
128 B bf16 RoPE) but a row *index* means the same thing in both. That is a hard
requirement, not a convenience: `pa_sparse_prefill_fp8_opus` and
`mla_decode_fwd_v4_nm` each take an NoPE pool, a RoPE pool, and ONE shared
index buffer. It is also why NoPE and RoPE cannot be two sub-blocks of one
envelope — an envelope of `E` bytes puts a block's row at `b*E/512 + r` in one
plane and `b*E/128 + r` in the other, and those agree only for `E == 0`.

The row space::

    row 0                                       plane_rows
    ├─────────────────────────┬  gap  ┬────────────────────┤
    │ num_blocks envelopes    │       │ num_slots slots    │
    │ (compressed history)    │       │ (state + windows)  │
    └─────────────────────────┴───────┴────────────────────┘
                                        slot j occupies
                                        [j*S, (j+1)*S)

Slots take the high end so the two regions grow towards each other and neither
moves what the other has placed: a new slot takes the next position *down*, a
new block the next row up. **Slots are numbered by position, not by age** —
slot `j` is at `j * slot_rows` whatever the split is, and `physical_slot` maps
a pool group to one. Numbering from `plane_rows` downwards instead, which is
how a two-ended layout reads most naturally, would make a slot's address
*decrease* with its index; the compressor state is handed to its kernels as a
tensor strided by slot, and that stride would be negative. A per-layer
view anchors at that layer's own base row and runs to the end of the plane, so
both regions are addressed through one base pointer and moving the boundary is
a host-side counter change — no re-carve, no re-capture.

A slot holds a request's compressor state first and its sliding windows after.
The state is bytes rather than rows and does not divide by compress class, so
it takes whole rows off the front of the slot and each plane materializes its
share at that plane's row width. Putting it first is what frees the window row
count from having to be even: only a slot's own start has to land on an
alignment boundary, and `slot_rows` is rounded up to keep it there.

Inside an envelope the layers are grouped by compress class rather than
interleaved, because within a class every layer contributes the same number of
rows. That regularity makes a class's layer stride a constant, which is what
lets one index formula serve every layer of the class — and the index buffers
the attention path already builds (`prefix_swa_indices` / `prefix_hca_indices`
/ `prefix_csa_indices`) are already split exactly that way.

Dense layers keep no compressed KV at all, so they appear only in the entry.
A layer whose window is wider than a plane's row — a DSpark draft layer, whose
block attention wants unquantized KV where the pool is packed — appears in
neither: it is `ABSENT_RATIO` here and its ring is a state field instead, still
addressed by `WindowParams` through `field_window_params`.
"""

# 【译】DeepSeek-V4 的每一行 KV 住在哪里——以"行"而不是字节为单位。
#
#   V4 在同一个池里存两类 KV：按 block 寻址的压缩历史，以及按槽位寻址的每请求滑动窗口。它们以前是每层一个
#   张量里的两个区域，这让两者的分界成了启动时的常量——缩小 block 数会移动全部 46 层的起点，所以哪一类都
#   不可能把空间让给另一类。
#
#   本模块把布局表述为两类共享的单一"行空间"，并让每个池以自己的行宽来物化它。一行是某一层一个 token 的
#   KV；一行的"成本"因平面而异（fp8 打包的 NoPE 512 B，bf16 的 RoPE 128 B），但一个行"号"在两个平面里
#   含义相同。这是硬性要求，不是图方便：pa_sparse_prefill_fp8_opus 和 mla_decode_fwd_v4_nm 各自接收一个
#   NoPE 池、一个 RoPE 池，以及"一个"共享的下标缓冲。这也是为什么 NoPE 和 RoPE 不能作为一个信封里的两个
#   子块——E 字节的信封会把一个 block 的行放在一个平面的 b*E/512 + r 处、另一个平面的 b*E/128 + r 处，
#   两者只在 E == 0 时一致。
#
#   行空间（图见上）：行 0 起是 num_blocks 个信封（压缩历史），顶端 plane_rows 往下是 num_slots 个槽位
#   （状态 + 窗口），中间是空隙；槽位 j 占 [j*S, (j+1)*S)。
#
#   槽位占高端，这样两个区域相向生长，谁都不会移动对方已经放好的东西：新槽位取往下的下一个位置，新 block
#   取往上的下一行。槽位按位置编号，而不是按先后——不论分界在哪，槽位 j 都在 j * slot_rows 处，
#   physical_slot 把池里的组映射到一个位置。若改成从 plane_rows 往下编号（这是两端布局最自然的读法），
#   槽位地址就会随编号"减小"；压缩器状态是以"按槽位跨步"的张量交给 kernel 的，那个步长就会是负的。每层的
#   视图从该层自己的基行开始、一直延伸到平面末尾，所以两个区域都通过一个基指针寻址，移动分界只是 host 侧
#   改一个计数——不用重新切分，也不用重新捕获（CUDA Graph）。
#
#   一个槽位先放请求的压缩器状态，后放它的滑动窗口。状态是以字节而非行计的，也不按压缩类划分，所以它从
#   槽位前部占用整数行，每个平面按自己的行宽物化它的那一份。把它放在前面，窗口的行数就不必是偶数：只有
#   槽位自己的起点必须落在对齐边界上，slot_rows 会向上取整来保证这一点。
#
#   在一个信封里，各层按压缩类分组而不是交错排列，因为同一类里每层贡献的行数相同。这种规整让一个类的层步长
#   是常数，于是一个下标公式能服务该类的每一层——而注意力路径已经在构造的下标缓冲（prefix_swa_indices /
#   prefix_hca_indices / prefix_csa_indices）恰好就是这样分的。
#
#   dense 层根本不存压缩 KV，所以它们只出现在条目（槽位里的窗口部分）里。窗口比平面一行还宽的层——DSpark
#   草稿层，它的块注意力在池是打包格式时仍想要未量化的 KV——两边都不出现：它在这里是 ABSENT_RATIO，它的环
#   改成一个状态字段，仍然通过 field_window_params 由 WindowParams 寻址。
#
# =====================================================================================
# 【导读】v4_pool_geometry.py —— V4 统一行空间的全部地址公式（第5批 5b，B级；device 侧的同一套公式在 v4_kernels/pool_index.py）
#   前置：CSA / HCA、压缩率 4 / 128、block_size 256 见 config.py Config.__post_init__ 第12步 V4 部分的【科普】；
#     概览图与 V4-Pro 的字节数见 pool_layout/__init__.py【科普】④；一行由哪些字节构成见 v4_pool_fields.py。
#   术语：信封（envelope）= 一个 block 在所有压缩层的行（按类分组：先 HCA 后 CSA）；条目（entry）= 一个槽位里
#     全部层的窗口行（按类分组：先 dense、再 HCA、再 CSA）；槽位（slot）= arena_rows（压缩器状态）+ 条目，
#     取整到 slot_align_rows；类（class）= 同一压缩率的所有层（ClassLayout）。
#   一条请求带数字走一遍（V4-Pro 式配置：层 0 dense、层 1~30 CSA、层 31~61 HCA；block_size 256、窗口 128、
#     不开投机、arena_rows = 0；num_blocks = 10、num_slots = 4；数字均由本文件的公式算出）：
#     类布局：HCA 31 层 × 2 行 → 信封偏移 0；CSA 30 层 × 64 行 → 信封偏移 62；envelope_rows = 1982
#             条目：dense 1 × 128 → 偏移 0；HCA 31 × 128 → 偏移 128；CSA 30 × 128 → 偏移 4096；entry_rows = slot_rows = 7936
#     平面：10 个信封 = 19,820 行 → 占 ceil(19820 / 7936) = 3 个槽位高；再加 4 个槽位 → plane_rows = 7 × 7936 = 55,552
#     请求拿到组 0 → physical_slot(0) = 6 → 它的槽位是行 [47616, 55552)
#     CSA 的第 3 个层（类内下标 2，全局层号 3）：基行 = 62 + 2 × 64 = 190
#       压缩行（block 3 的第 10 行）：compress_index = 3 × 1982 + 10 = 5956 → 平面行 190 + 5956 = 6146
#       窗口行（pos = 200 → 环位置 q = 200 % 128 = 72 = 第 1 段第 8 个）：
#         用 WindowParams(ring_start 4034, slot_rows 7936, ring_slots 128, ring_stride 64, run_rows 1920).index(6, 200)：
#         = 6 × 7936 + 4034 + 1 × 1920 + 8 = 53578 → 平面行 190 + 53578 = 53768
#         = 槽位起点 47616 + CSA 条目偏移 4096 + 层 2 × 64 + 1 × 1920 + 8（基行里的层项正好抵消）
#   谁用：deepseek_v4_attn.py（builder 构造几何、按它切平面与视图、算 checkpoint 范围）、v4_kernels/*（pool_index.py 的
#     device 函数、state_writes.py、paged_prefill_indices.py、paged_decode_indices.py、dspark_fp8_indices.py）、
#     models/deepseek_v4.py（visible_csa）、plugin/sglang 与 plugin/vllm 的 V4 桥接
# =====================================================================================

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Compress ratios, as they appear in the model config's per-layer list.
# 【译】压缩率，取值与模型 config（hf_config.compress_ratios）的逐层列表一致：0 = 只有窗口，4 = CSA，128 = HCA。
DENSE_RATIO = 0
CSA_RATIO = 4
HCA_RATIO = 128

# A token sees the compressed groups closed at or before its OWN position. The
# per-sequence `ctx // ratio` is the LAST token's count, and an MTP or DSpark
# sequence hands `1 + k` tokens to one forward, so the earlier ones would read
# groups holding their own future drafts. It is not a second cap either: it
# cannot bind while `pos < ctx`, which `require_step_within_full_q` keeps true
# at ATOM's producers of per-seq step lengths. The plugin bridges inherit the
# same premise from the host engine's `positions` and do not check it.
#
# Every host-side count comes from these two. The Triton kernels take the ratio
# as a `constexpr` and divide inline; keep the spellings identical.
# 【译】一个 token 能看到的，是在它"自己"的位置或之前已经闭合的压缩组。按序列算的 ctx // ratio 是"最后一个"
#   token 的计数，而 MTP 或 DSpark 的序列一次 forward 交出 1 + k 个 token，所以前面的那些 token 会读到包含
#   它们自己未来草稿的组。它也不是第二道上限：只要 pos < ctx 它就不可能起约束作用，而 ATOM 产生"每序列步长"
#   的地方由 require_step_within_full_q 保证这一点。插件桥接从宿主引擎的 positions 继承同样的前提，但并不检查。
#
#   所有 host 侧计数都来自这两个函数。Triton kernel 以 constexpr 接收 ratio 并在内部做除法；两边写法要保持一致。
# 例：CSA 组 c0 = token 0~3；token 3 已写入时它就闭合 → visible_csa(3) = 1，visible_csa(2) = 0；
#   HCA：visible_hca(127) = 1、visible_hca(255) = 2。MTP k=2 时一步送 pos = 6, 7, 8（ctx = 9）：可见 CSA 组分别为
#   1、2、2；若都按序列算的 ctx // 4 = 2，pos 6 的 token 就会看到组 c1（token 4~7），其中 token 7 是它未来的草稿


def visible_csa(pos):
    """CSA groups visible to the token at `pos` — int, numpy array or tensor."""
    # 【译】位置 pos 处的 token 可见的 CSA 组数——pos 可以是 int、numpy 数组或张量（同一个表达式三者通用）。
    return (pos + 1) // CSA_RATIO


def visible_hca(pos):
    """HCA groups visible to the token at `pos` — int, numpy array or tensor."""
    # 【译】位置 pos 处的 token 可见的 HCA 组数——pos 可以是 int、numpy 数组或张量。
    return (pos + 1) // HCA_RATIO


def require_step_within_full_q(longest: int, full_q: int, source: str) -> None:
    """A sequence may not forward more tokens in one step than `full_q`.

    This is what `pos < ctx` reduces to, so ATOM's three producers of per-seq
    forward lengths check it rather than assume it. Two things break together
    when it fails. `positions` are anchored at `ctx - full_q`, so a longer step
    puts its last token at or past `ctx` -- the one case where the per-sequence
    bound the rule above drops would have bound something. And the DSpark
    rectangle's `lead = full_q - len` goes negative, so
    `_v4_decode_indptr_kernel` maps a whole band and reads tokens past the
    sequence's span, stamping one sequence's visibility onto another's rows.

    `ValueError`, not `assert`: the second failure is a device-side
    out-of-bounds read, and `python -O` strips asserts.
    """
    # 【译】一条序列一步 forward 的 token 数不能超过 full_q。
    #   这就是 pos < ctx 归结出来的条件，所以 ATOM 里三个产生"每序列 forward 长度"的地方检查它，而不是假设它。
    #   它不成立时两件事一起坏掉。positions 以 ctx - full_q 为锚点，所以更长的一步会把最后一个 token 放在 ctx
    #   或更后面——这正是上面那条规则去掉的"按序列的上界"本来会起作用的唯一情形。而 DSpark 矩形的
    #   lead = full_q - len 会变成负数，于是 _v4_decode_indptr_kernel 映射整整一条带，读到超出该序列范围的
    #   token，把一条序列的可见性盖到另一条序列的行上。
    #   用 ValueError 而不是 assert：第二种失败是 device 侧的越界读，而 python -O 会去掉 assert。
    # 输入：longest = 本步某条序列最多送几个 token；full_q = decode 矩形的宽度（1 + 投机步数）；source = 报错用的来源名
    # 调用方：deepseek_v4_attn.py 的 prepare_decode 与 build_for_cudagraph_capture
    # 观察：docstring 说"三个产生者"，v0.1.8 的 atom/ 下只有这两处调用本函数（_attach_v4_paged_decode_meta 里
    #   另有一个引用本函数理由的、针对缓冲区容量的 ValueError，检查的不是同一个条件）
    if longest > full_q:
        raise ValueError(
            f"{source} forwards {longest} tokens for some sequence but full_q "
            f"is {full_q}, which would place a token past its own context"
        )


# Not a compress ratio: a layer whose KV this row space does not hold at all.
# A DSpark draft layer wants its window at a width the planes do not offer —
# unquantized where the pool is packed — so the window becomes a state field
# instead, and the layer reserves neither an envelope nor an entry here. It is
# spelled as a ratio because that is the per-layer vocabulary the caller
# already has; it never reaches `classes`, so asking one for its layout raises.
# 【译】不是压缩率：表示这个行空间根本不保存该层的 KV。DSpark 草稿层想要的窗口宽度是平面提供不了的——池是
#   打包格式时它要未量化的——所以窗口改成一个状态字段，这一层在这里既不占信封也不占条目。把它写成一个 ratio，
#   是因为这是调用方已有的逐层词汇；它永远不会进入 classes，所以向某个类询问它的布局会报错。
# 谁传：deepseek_v4_attn.py 的 _geometry_ratios（_field_window_layers 里的层）
ABSENT_RATIO = -1

# Envelope order puts the narrow class first so a future one can be appended
# without moving it. The entry leads with dense, the one class that has no
# envelope part at all, which is what frees the two orders from having to agree.
# 【译】信封顺序把"窄"的类（HCA：每 block 每层只有 2 行）放在最前，这样将来新加一类可以追加在后面而不移动它。
#   条目以 dense 打头——它是唯一完全没有信封部分的类——这让两种顺序不必一致。
_ENVELOPE_ORDER = (HCA_RATIO, CSA_RATIO)
_ENTRY_ORDER = (DENSE_RATIO, HCA_RATIO, CSA_RATIO)
_KNOWN_RATIOS = frozenset(_ENTRY_ORDER) | {ABSENT_RATIO}


def merge_abutting(runs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """`(start, count)` pairs in order, with the ones that touch joined.

    Rows of a row space and bytes of a copy are both described this way, and
    both want as few of them as possible: every range a checkpoint copy is cut
    into costs a span, and a span costs grid.

    Ascending and disjoint is required, not assumed. Each run is compared only
    against the one before it, so an out-of-order input merges nothing and
    reads as legal: `[(0, 400), (256, 64), (320, 64)]` used to return
    `[(0, 400), (256, 128)]`, which double-counts 128 bytes and overlaps the
    first range. Nothing downstream can see that -- `checkpoint_image_bytes`
    would over-count, and both the op validator and the sizing cross-check
    compare against that same wrong number -- so the ordering is checked here
    rather than left as a property of whoever happens to build the list.
    """
    # 【译】按顺序给出的 (start, count) 对，把首尾相接的合并起来。
    #   行空间的行和一次拷贝的字节都用这种方式描述，而且都希望段数越少越好：checkpoint 拷贝被切成的每一段都要花
    #   一个 span，而一个 span 要花 grid。
    #   升序且不相交是"要求"而不是"假设"。每一段只和前一段比较，所以乱序的输入什么也不会合并，看起来却是合法的：
    #   [(0, 400), (256, 64), (320, 64)] 以前会返回 [(0, 400), (256, 128)]，多算了 128 字节并与第一段重叠。
    #   下游谁也看不出来——checkpoint_image_bytes 会多算，而算子校验和定大小时的交叉检查都和同一个错误数字
    #   比较——所以顺序在这里检查，而不是当作"恰好构造这个列表的人"的性质。
    # 例：[(0, 4), (4, 2), (8, 1)] → [(0, 6), (8, 1)]
    # 调用方：本文件 UnifiedPoolGeometry.entry_row_runs；deepseek_v4_attn.py（checkpoint 的字节范围）
    merged: list[tuple[int, int]] = []
    for start, count in runs:
        if count < 0 or start < 0:
            raise ValueError(f"a run must be non-negative, got ({start}, {count})")
        if merged:
            end = merged[-1][0] + merged[-1][1]
            if start < end:
                raise ValueError(
                    f"runs must ascend and not overlap: ({start}, {count}) "
                    f"starts inside the run ending at {end}"
                )
            if start == end:
                merged[-1] = (merged[-1][0], merged[-1][1] + count)
                continue
        merged.append((start, count))
    return merged


def ring_offset_for(num_layers: int, ring_stride: int, ring_pos: int) -> int:
    """The layer-independent part of a window row, `f(q)`.

    Every layer's window row has to be `i * ring_stride + f(q)` for one shared
    `f`: the layer term is forced by the compress side, which cannot pad, and a
    shared `f` is what keeps the index layer-independent. Whenever the window
    is longer than `ring_stride` the layers overlap if laid out contiguously,
    so the window is cut into `ring_stride`-sized runs and the runs interleaved
    across the class.

    Give each run the class's full height. Two positions then collide only if
    they share a run *and* a position inside it: positions in one run differ by
    less than `ring_stride` and so cannot be a whole layer apart, and positions
    in different runs differ by at least the class height and so cannot be
    either.
    """
    # 【译】窗口行里与层无关的那部分 f(q)。
    #   每层的窗口行都必须是 i * ring_stride + f(q)，其中 f 为所有层共用：层那一项是被压缩那一侧强制的（它不能
    #   填充），而共用的 f 让下标与层无关。只要窗口比 ring_stride 长，连续排放时各层就会重叠，所以把窗口切成
    #   ring_stride 大小的若干段，再把这些段在整个类里交错排放。
    #   给每一段整个类的高度。这样两个位置只有在同属一段、且段内位置相同时才会冲突：同一段里的位置相差小于
    #   ring_stride，不可能正好差一整层；不同段里的位置相差至少一个类的高度，也不可能。
    # 为什么层项是 i * ring_stride：压缩类的层 i 的基行是 envelope_offset + i * block_rows（layer_base_row），而
    #   ring_stride = block_rows；窗口行也从这个基行起算，所以条目里层 i 的窗口必须同样带 i * ring_stride，基行里的
    #   层项才能抵消，让 kernel 的下标公式不含层号。图（CSA：30 层、ring_stride 64、窗口 128，行相对 CSA 的条目偏移）：
    #     段 0：[层0 位置0~63 │ 层1 位置0~63 │ … │ 层29 位置0~63]       行 0 ~ 1919
    #     段 1：[层0 位置64~127 │ 层1 位置64~127 │ … │ 层29 …]          行 1920 ~ 3839
    #     层 i、位置 q 的行 = i × 64 + (q // 64) × (30 × 64) + q % 64；例 i = 2、q = 72 → 128 + 1920 + 8 = 2056
    chunk, within = divmod(ring_pos, ring_stride)
    return chunk * (num_layers * ring_stride) + within


def entry_rows_for(num_layers: int, ring_stride: int, ring_slots: int) -> int:
    """Height the interleaved construction reaches — and the least possible.

    `ring_offset_for` peaks at the last window position (a later run always
    outweighs a longer tail inside an earlier one), so the tallest row belongs
    to the last layer at that position.

    No construction does better. Count rows by residue mod `ring_stride`: a
    height of `L` offers each residue at most `ceil(L / ring_stride)` rows, and
    every one of the `num_layers` layers needs a row of its own for each window
    position falling in that residue. Working that inequality back over all
    `ring_stride` residues yields exactly this height.
    """
    # 【译】交错构造达到的高度——也是可能的最小高度。
    #   ring_offset_for 在最后一个窗口位置达到最大（靠后的段总比靠前段里更长的尾巴更重），所以最高的那一行属于
    #   最后一层在那个位置的行。
    #   没有哪种构造能做得更好。按模 ring_stride 的余数数行：高度 L 给每个余数最多 ceil(L / ring_stride) 行，而
    #   num_layers 层中的每一层都需要为落在该余数上的每个窗口位置各占一行。把这个不等式在全部 ring_stride 个
    #   余数上倒推回去，正好得到这个高度。
    # 例：CSA 30 层、stride 64、窗口 128 → 29 × 64 + (1 × 1920 + 63) + 1 = 3840 = 30 × 128（128 能被 64 整除，零填充）；
    #   窗口 128 + 投机 3 = 131 → 5699 行，而 30 × 131 = 3930 个位置真正用到，空着 1769 行（见 ClassLayout.entry_row_runs）
    return (
        (num_layers - 1) * ring_stride
        + ring_offset_for(num_layers, ring_stride, ring_slots - 1)
        + 1
    )


@dataclass(frozen=True)
class WindowParams:
    """A class's window-row formula as five scalars a kernel can carry.

    `index(slot, pos)` is the whole of it::

        q = pos % ring_slots
        slot * slot_rows + ring_start + (q // ring_stride) * run_rows
                                      + (q %  ring_stride)

    All five are constant for the life of an allocation, so a kernel may carry
    them by value even inside a captured graph. `ring_start` is an ordinary
    argument rather than a `constexpr` only to keep a reallocation from forcing
    a Triton recompile; moving the compress/window boundary does not change it.
    See `UnifiedPoolGeometry.window_params` for why.

    `slot` is a *physical* slot — a position in the plane, not a pool group.
    `UnifiedPoolGeometry.physical_slot` is the one place the two meet, and
    everything downstream of it speaks positions: the windows, the compressor
    state's strided view, and the DSpark draft's separate plane.

    `pos` is an absolute token position and the wrap happens here, matching
    `pool_index.window_row` term for term — the two are transcriptions of one
    formula and `tests/test_pool_index.py` is where they meet.
    """

    # 【译】一个类的窗口行公式，压成 kernel 能携带的五个标量。
    #   index(slot, pos) 就是它的全部（公式见上）。
    #   五个值在一次分配的生命周期内都是常量，所以 kernel 即使在捕获的图里也可以按值携带它们。ring_start 是普通
    #   参数而不是 constexpr，只是为了不让重新分配强迫 Triton 重新编译；移动压缩/窗口分界并不改变它。原因见
    #   UnifiedPoolGeometry.window_params。
    #   slot 是"物理"槽位——平面里的一个位置，而不是池里的组。UnifiedPoolGeometry.physical_slot 是两者唯一
    #   交汇的地方，它下游的一切都说"位置"：窗口、压缩器状态的跨步视图、DSpark 草稿的单独平面。
    #   pos 是 token 的绝对位置，取模回绕在这里做，和 pool_index.window_row 逐项对应——两者是同一个公式的两份
    #   抄写，tests/test_pool_index.py 是它们对上的地方。
    # 五个字段：ring_start = 该类窗口相对层基行的起点（已扣掉基行的锚点）；slot_rows = 一个槽位的行数；
    #   ring_slots = 环长（窗口 + 投机）；ring_stride = 每段长度；run_rows = 一段的总高度（层数 × ring_stride）
    # 例：V4-Pro 的 CSA 类：WindowParams(4034, 7936, 128, 64, 1920)（推导见文件头【导读】）
    # 产出方：UnifiedPoolGeometry.window_params / field_window_params、v41_pool_geometry.V41PoolGeometry.window；
    #   读取方：v4_kernels 的写窗口 / 建下标 kernel（state_writes.py、paged_*_indices.py、dspark_fp8_indices.py）

    ring_start: int
    slot_rows: int
    ring_slots: int
    ring_stride: int
    run_rows: int

    def index(self, slot: int, pos: int) -> int:
        chunk, within = divmod(pos % self.ring_slots, self.ring_stride)
        return slot * self.slot_rows + self.ring_start + chunk * self.run_rows + within


@dataclass(frozen=True)
class ClassLayout:
    """One compress class's rows, inside an envelope and inside an entry.

    `layers` are global layer ids in class order; a layer's position in that
    list is the only per-layer input any address formula takes.

    `block_rows` is how many rows one block compresses to in one layer of this
    class (`block_size // ratio`), and doubles as the class's **layer stride**:
    layer `i` starts `i * block_rows` rows into the class's part of an
    envelope. A dense class compresses to nothing, so it is free to pack its
    window rows layer after layer and takes `ring_slots` as its stride instead.
    """

    # 【译】一个压缩类的行，在信封里和在条目里各一份。
    #   layers 是按类内顺序排列的全局层号；一层在这个列表里的位置，是任何地址公式唯一的逐层输入。
    #   block_rows 是这个类的一层里一个 block 压缩成多少行（block_size // ratio），同时也是这个类的"层步长"：
    #   第 i 层从该类在信封里那部分的第 i * block_rows 行开始。dense 类压缩成零行，所以可以把它的窗口行一层接
    #   一层地排，改用 ring_slots 作为步长。
    # 例（V4-Pro 式配置）：HCA：31 层，block_rows 2，ring_stride 2，envelope_offset 0，entry_offset 128；
    #   CSA：30 层，64，64，62，4096；dense：1 层，0，128，0，0

    ratio: int
    layers: tuple[int, ...]
    block_rows: int
    ring_stride: int
    ring_slots: int
    envelope_offset: int
    entry_offset: int

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def envelope_rows(self) -> int:
        """Rows this class takes in one envelope, across all its layers."""
        # 【译】这个类在一个信封里占的行数，所有层合计（CSA：30 × 64 = 1920）。
        return self.num_layers * self.block_rows

    @property
    def entry_rows(self) -> int:
        """Rows this class's sliding windows take in one entry."""
        # 【译】这个类的滑动窗口在一个条目里占的行数（含交错带来的填充，见 entry_rows_for）。
        return entry_rows_for(self.num_layers, self.ring_stride, self.ring_slots)

    def layer_index(self, layer_id: int) -> int:
        return self.layers.index(layer_id)

    def ring_offset(self, ring_pos: int) -> int:
        return ring_offset_for(self.num_layers, self.ring_stride, ring_pos)

    def ring_row(self, layer_index: int, ring_pos: int) -> int:
        """Row of a window position, relative to the class's part of an entry."""
        # 【译】一个窗口位置的行，相对于该类在条目里那部分的起点。
        return layer_index * self.ring_stride + self.ring_offset(ring_pos)

    def entry_row_runs(self) -> list[tuple[int, int]]:
        """`(start, count)` runs of entry rows some `ring_row` reaches.

        The interleave is by ring position, not by layer: run `c` of the ring
        owns the rows `[c*run_rows, (c+1)*run_rows)` and every layer of the
        class has its positions for that run inside them. So all the whole
        runs together are one contiguous range, and only the ring's last,
        partial run is scattered — there each layer reaches
        `ring_slots % ring_stride` rows of its own `ring_stride` slice and
        leaves the rest.

        Those leftovers are `entry_rows` less `num_layers * ring_slots`: what
        a layer-independent index formula costs (`entry_rows_for`). No
        `(layer, position)` pair maps to one, so nothing writes or reads them,
        which is what lets a copy that only has to preserve windows — a
        checkpoint image is gathered back into a slot, never read by an
        attention kernel — leave them out.
        """
        # 【译】某个 ring_row 能到达的条目行，以 (start, count) 段给出。
        #   交错是按环位置、而不是按层进行的：环的第 c 段拥有 [c*run_rows, (c+1)*run_rows) 这些行，类里每一层在这一段
        #   的位置都在其中。所以所有完整的段合起来是一段连续范围，只有环最后那个不完整的段是分散的——在那里每层只用到
        #   自己 ring_stride 宽切片中的 ring_slots % ring_stride 行，其余空着。
        #   这些剩余的行数就是 entry_rows 减去 num_layers * ring_slots：一个与层无关的下标公式的代价（entry_rows_for）。
        #   没有任何 (层, 位置) 对映射到它们，所以没人写也没人读它们，这让一个只需保留窗口的拷贝——checkpoint 镜像
        #   被收集回一个槽位，从不被注意力 kernel 直接读——可以把它们略去。
        # 例：CSA、窗口 128 → [(0, 3840)] 一段；窗口 131 → [(0, 3840), (3840, 3), (3904, 3), …] 共 1 + 30 段
        run_rows = self.num_layers * self.ring_stride
        whole, partial = divmod(self.ring_slots, self.ring_stride)
        runs = [(0, whole * run_rows)] if whole else []
        if partial:
            runs += [
                (whole * run_rows + i * self.ring_stride, partial)
                for i in range(self.num_layers)
            ]
        return runs


class UnifiedPoolGeometry:
    """The row space, and every address formula that reads from it.

    Sole owner of the arithmetic: allocation, view binding, the index kernels
    and their references all take their offsets from here rather than each
    re-deriving them. A carve is only ever as correct as the agreement between
    the party that splits the buffer and the parties that compute offsets into
    it, and that agreement is not something a test of the carve alone can see.
    """

    # 【译】行空间，以及从它读取的每一个地址公式。
    #   算术的唯一拥有者：分配、视图绑定、下标 kernel 及其参考实现都从这里取偏移，而不是各自重新推导。一次切分的
    #   正确性，永远只取决于"切缓冲区的一方"和"往里算偏移的各方"之间的一致；而这种一致，单测切分本身是看不出来的。
    # 生命周期：deepseek_v4_attn.py 的 builder __init__ 以容量 0 构造（定大小要先知道一个 block、一个槽位的行数）
    #   → allocate_per_req_cache 用 with_capacity(num_blocks, num_slots) 换成定好大小的同一布局

    def __init__(
        self,
        ratios: list[int],
        num_blocks: int,
        num_slots: int,
        ring_slots: int,
        block_size: int,
        plane_rows: int | None = None,
        arena_rows: int = 0,
        slot_positions: int | None = None,
        slot_align_rows: int = 2,
    ):
        """`ring_slots` is `win_with_spec` — the sliding window plus the draft
        lookahead,
        i.e. how many of a request's most recent writes stay addressable.

        `arena_rows` is how many rows off the front of a slot the compressor
        state takes; 0 leaves a slot holding nothing but windows, which is what
        the window-only tests want.

        `plane_rows` sizes a plane above what `num_blocks` and `num_slots`
        currently need, leaving the surplus as the gap the two ends grow into.
        It defaults to exactly what they need, which pins the boundary.

        `slot_align_rows` is the row multiple a slot's size is rounded to. Two
        is enough for the compressor state alone (see `slot_rows`); a window
        materialized at a width wider than a plane's row needs the slot to be a
        multiple of that many rows as well, or the retyped view it is addressed
        through does not divide evenly. The caller owns that arithmetic because
        it owns the plane widths.
        """
        # 【译】ring_slots 就是 win_with_spec——滑动窗口加上草稿的前瞻量，即一个请求最近写入的多少个位置要保持可寻址。
        #   arena_rows 是压缩器状态从槽位前部占用的行数；0 表示槽位里只有窗口，这是只测窗口的测试想要的。
        #   plane_rows 让平面比 num_blocks 和 num_slots 当前所需的更大，多出的部分就是两端可以生长进去的空隙。默认值
        #   恰好是它们所需的大小，这会把分界钉死。
        #   slot_align_rows 是槽位大小要取整到的行数倍数。只为压缩器状态的话 2 就够了（见 slot_rows）；如果有窗口以比
        #   平面一行更宽的宽度物化，槽位还必须是那么多行的倍数，否则用来寻址它的重解释视图就除不尽。这份算术归调用方，
        #   因为平面宽度归调用方。
        # 其余参数：ratios = 每层的压缩率（含草稿层，ABSENT_RATIO 表示不在本行空间）；num_blocks / num_slots = 当前分界；
        #   block_size = 每 block 的原始 token 数（V4 为 256）；slot_positions = 槽位按哪个平面的位置数编号（默认本平面的）
        if not ratios:
            raise ValueError("a V4 pool needs at least one layer")
        unknown = set(ratios) - _KNOWN_RATIOS
        if unknown:
            raise ValueError(f"unknown V4 compress ratios {sorted(unknown)}")
        if slot_align_rows < 1:
            raise ValueError(f"slot_align_rows must be positive, got {slot_align_rows}")
        if ring_slots < 1:
            raise ValueError(f"ring_slots must be positive, got {ring_slots}")
        if arena_rows < 0:
            raise ValueError(f"arena_rows must not be negative, got {arena_rows}")

        self.arena_rows = arena_rows
        self.ratios = list(ratios)
        self.num_layers = len(ratios)
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.ring_slots = ring_slots
        self.block_size = block_size

        def layers_of(ratio: int) -> tuple[int, ...]:
            return tuple(i for i, r in enumerate(ratios) if r == ratio)

        def stride_of(ratio: int) -> int:
            return ring_slots if ratio == DENSE_RATIO else block_size // ratio

        # Both orders are walked before any layout is built, so each class
        # learns its offset in the envelope and in the entry at once.
        # 【译】在构建任何布局之前先走完两种顺序，这样每个类一次就得知它在信封里和条目里的偏移。
        envelope_offsets: dict[int, int] = {}
        cursor = 0
        for ratio in _ENVELOPE_ORDER:
            if layers_of(ratio):
                envelope_offsets[ratio] = cursor
                cursor += len(layers_of(ratio)) * (block_size // ratio)
        self.envelope_rows = cursor

        entry_offsets: dict[int, int] = {}
        cursor = 0
        for ratio in _ENTRY_ORDER:
            layers = layers_of(ratio)
            if layers:
                entry_offsets[ratio] = cursor
                cursor += entry_rows_for(len(layers), stride_of(ratio), ring_slots)
        self.entry_rows = cursor

        self.classes: dict[int, ClassLayout] = {
            ratio: ClassLayout(
                ratio=ratio,
                layers=layers_of(ratio),
                block_rows=0 if ratio == DENSE_RATIO else block_size // ratio,
                ring_stride=stride_of(ratio),
                ring_slots=ring_slots,
                envelope_offset=envelope_offsets.get(ratio, 0),
                entry_offset=entry_offsets[ratio],
            )
            for ratio in _ENTRY_ORDER
            if layers_of(ratio)
        }

        # A slot's start is the one address the compressor state's alignment
        # rests on, so it is rounded up: the narrow plane is 128 B wide and
        # `EntryMajorArena` retypes from a 256 B boundary, which alone wants an even
        # row. Rounding the slot rather than the window is what keeps
        # `entry_rows` free of the constraint — the state sits at the front,
        # offset zero.
        # 【译】槽位的起点是压缩器状态的对齐唯一依赖的地址，所以要向上取整：窄平面一行 128 B，而 EntryMajorArena 从
        #   256 B 边界重解释类型，单这一点就要求偶数行。取整的是槽位而不是窗口，这让 entry_rows 不受这个约束——状态在
        #   最前面，偏移为 0。
        self.slot_align_rows = slot_align_rows
        rows = arena_rows + self.entry_rows
        self.slot_rows = -(-rows // slot_align_rows) * slot_align_rows

        # 默认 plane_rows：压缩区向上取整到整槽位高，再加 num_slots 个槽位。例：10 × 1982 = 19820 行 → 3 个槽位高 → 共 7 × 7936
        block_rows = num_blocks * self.envelope_rows
        if plane_rows is None:
            whole = num_slots + -(-block_rows // self.slot_rows)
            self.plane_rows = whole * self.slot_rows
        else:
            self.plane_rows = plane_rows
        # Whole slots the plane holds. Slot `j` sits at `j * slot_rows`, so the
        # pool hands out the top `num_slots` of these positions and the
        # remainder under one slot at the very top is the only waste the layout
        # carries — it does not scale with anything.
        # 【译】平面能容纳的完整槽位数。槽位 j 在 j * slot_rows 处，所以池发出的是这些位置中最顶上的 num_slots 个；
        #   最顶端不足一个槽位的余数是这个布局唯一的浪费——它不随任何东西增长。
        held = self.plane_rows // self.slot_rows
        if num_slots > held or (held - num_slots) * self.slot_rows < block_rows:
            raise ValueError(
                f"plane of {self.plane_rows} rows cannot hold {num_blocks} "
                f"blocks ({self.envelope_rows} rows each) plus {num_slots} "
                f"slots ({self.slot_rows} rows each)"
            )
        self.slot_positions = held if slot_positions is None else slot_positions
        # Positions this plane does not reach, because it took its numbering
        # from one with more room. Subtracted back out of every row it computes.
        # 【译】这个平面到达不了的位置数，因为它的编号取自一个空间更大的平面。它算出的每一行都要再减掉这部分。
        # （slot_positions 由 deepseek_v4_attn.py / vllm 桥接在构造 arena 时传入；with_capacity 不转发它，新对象按自己的 held 编号）
        self.slot_origin = self.slot_positions - held
        if self.slot_origin < 0:
            raise ValueError(
                f"plane of {self.plane_rows} rows holds {held} slots, more than "
                f"the {slot_positions} positions it was told to number by"
            )

    def with_capacity(
        self, num_blocks: int, num_slots: int, plane_rows: int | None = None
    ) -> UnifiedPoolGeometry:
        """The same layout at a different split.

        Sizing has to state a block's and a slot's cost before either count is
        known, and the counts it produces then have to describe the same
        layout. Building one shape at capacity zero and re-sizing it here keeps
        that a single object rather than two constructions that must agree.
        """
        # 【译】同一个布局，换一个分界。
        #   定大小时必须在任何计数已知之前说出一个 block 和一个槽位的成本，而它算出的计数随后又必须描述同一个布局。
        #   先以容量 0 构造一个形状、再在这里改大小，让这件事始终是一个对象，而不是两次必须一致的构造。
        # 调用方：deepseek_v4_attn.py allocate_per_req_cache（with_capacity(num_blocks, num_slots)）
        return UnifiedPoolGeometry(
            self.ratios,
            num_blocks,
            num_slots,
            self.ring_slots,
            self.block_size,
            plane_rows,
            self.arena_rows,
            slot_align_rows=self.slot_align_rows,
        )

    # ---- byte helpers ---------------------------------------------------
    # 【译】---- 字节换算辅助函数 ----

    # 字节换算：都是"行数 × 这个平面的行宽"。例（V4-Pro 式、fp8 两平面 512 B + 128 B）：block_bytes 合计 1982 × 640 B，
    #   slot_bytes 合计 7936 × 640 B；deepseek_v4_attn.py 的 sub_pool_specs 用它们声明 page_pool / state_pool
    def plane_bytes(self, row_bytes: int) -> int:
        return self.plane_rows * row_bytes

    def block_bytes(self, row_bytes: int) -> int:
        """Bytes one block costs in a plane of this row width."""
        # 【译】在这个行宽的平面里，一个 block 的成本（字节）。
        return self.envelope_rows * row_bytes

    def slot_bytes(self, row_bytes: int) -> int:
        """Bytes one request costs in a plane of this row width.

        The whole slot: compressor state and windows both, since the two are
        allocated and relinquished together and no caller can have one without
        the other.
        """
        # 【译】在这个行宽的平面里，一个请求的成本（字节）。
        #   整个槽位：压缩器状态和窗口都算，因为两者一起分配、一起交还，没有调用方能只要其中一个。
        return self.slot_rows * row_bytes

    # ---- addressing -----------------------------------------------------
    # 【译】---- 寻址 ----

    # layer_class：某层所属的压缩类；ABSENT_RATIO 的层（DSpark 草稿层）不在任何类里，直接报错
    def layer_class(self, layer_id: int) -> ClassLayout:
        ratio = self.ratios[layer_id]
        if ratio == ABSENT_RATIO:
            raise ValueError(
                f"layer {layer_id} keeps no KV in this row space; its window is "
                "a state field, so it has no class, no base row and no view"
            )
        return self.classes[ratio]

    def layer_base_row(self, layer_id: int) -> int:
        """Row a layer's view anchors at; it runs from there to the plane end.

        Compressed classes anchor inside envelope 0, so a compress index needs
        no layer term. Dense layers have no envelope part, so they anchor at
        the offset their windows carry inside an entry instead — the same
        cancellation, arranged from the other side.
        """
        # 【译】一层的视图锚定在哪一行；从那里一直到平面末尾。
        #   压缩类锚定在信封 0 内部，所以压缩行的下标不需要层那一项。dense 层没有信封部分，所以改为锚定在它们的窗口
        #   在条目内的偏移处——同样的抵消，从另一侧安排。
        # 例：CSA 类内第 2 层 → 62 + 2 × 64 = 190；HCA 第 0 层 → 0；dense 第 0 层 → 0
        # 调用方：deepseek_v4_attn.py（按它从平面切出每层视图 plane[base:]）
        cls = self.layer_class(layer_id)
        index = cls.layer_index(layer_id)
        if cls.ratio == DENSE_RATIO:
            return index * cls.ring_stride
        return cls.envelope_offset + index * cls.block_rows

    def layer_view_rows(self, layer_id: int) -> int:
        """Rows a layer's view spans — its base to the end of the plane."""
        # 【译】一层视图覆盖的行数——从它的基行到平面末尾。
        return self.plane_rows - self.layer_base_row(layer_id)

    def compress_index(self, layer_id: int, block: int, row: int) -> int:
        """Index of a compressed row, relative to the layer's own base.

        Layer-independent by construction: the layer's contribution already
        sits in the base, leaving the envelope stride times the block.
        """
        # 【译】一个压缩行的下标，相对于该层自己的基行。
        #   按构造与层无关：层的贡献已经在基行里了，剩下的只是信封步长乘以 block 号。
        # 例：block 3 的第 10 行 → 3 × 1982 + 10 = 5956（再加基行 190 得平面行 6146）
        # （这几个标量版的地址函数主要供测试与参考实现用；kernel 侧见 v4_kernels/pool_index.py）
        cls = self.layer_class(layer_id)
        if cls.ratio == DENSE_RATIO:
            raise ValueError(f"layer {layer_id} is dense and keeps no compressed KV")
        if not 0 <= row < cls.block_rows:
            raise ValueError(
                f"row {row} outside 0..{cls.block_rows} for ratio {cls.ratio}"
            )
        return block * self.envelope_rows + row

    def window_index(self, layer_id: int, slot: int, ring_pos: int) -> int:
        """Index of a sliding-window row, relative to the layer's own base."""
        # 【译】一个滑动窗口行的下标，相对于该层自己的基行。slot 是物理位置，ring_pos 已经是环内位置（0 ≤ ring_pos < ring_slots）
        if not 0 <= ring_pos < self.ring_slots:
            raise ValueError(f"ring position {ring_pos} outside 0..{self.ring_slots}")
        return self.window_params(self.ratios[layer_id]).index(slot, ring_pos)

    def window_params(self, ratio: int) -> WindowParams:
        """The same formula, flattened into what a Triton kernel can take.

        Kernels get this rather than the class layout so there is one
        expression for a window row, not one here and a transcription of it in
        each kernel.

        **Moving the boundary changes nothing here.** `ring_start` is an offset
        inside a slot and a compress row counts up from row 0, so neither
        address knows where the other region stops; `num_blocks` and `num_slots`
        only gate which block ids and slots the allocator may hand out. That is
        what makes the layout safe under CUDA-graph capture, where a by-value
        scalar is frozen exactly as hard as a `constexpr` — the argument-vs-
        constexpr choice buys a recompile, not a re-capture.

        The one thing that must not move is `plane_rows`, because
        `physical_slot` counts back from the topmost position and that is how
        many the plane holds. It defaults to a tight fit around the current
        split, which is right while the boundary is pinned; an elastic pool has
        to pass its own fixed capacity instead, or every group lands somewhere
        else the moment the split it was supposed to be independent of changes.
        """
        # 【译】同一个公式，摊平成 Triton kernel 能接收的形式。
        #   kernel 拿到的是这个，而不是类布局，这样窗口行只有一个表达式，而不是这里一个、每个 kernel 里各抄一份。
        #   移动分界在这里什么都不改变。ring_start 是槽位内的偏移，压缩行从第 0 行往上数，所以两个地址都不知道对方
        #   区域在哪里结束；num_blocks 和 num_slots 只限制分配器可以发出哪些 block 号和槽位。这正是这个布局在 CUDA
        #   Graph 捕获下安全的原因：在捕获的图里，按值传入的标量和 constexpr 一样被冻结——参数还是 constexpr 的选择
        #   影响的是要不要重新编译，而不是要不要重新捕获。
        #   唯一不能动的是 plane_rows，因为 physical_slot 是从最顶端的位置倒数的，而那就是平面能放下的位置数。它默认
        #   紧贴当前分界，在分界钉死时是对的；一个弹性的池必须传入自己固定的容量，否则在它本应与之无关的分界一变化时，
        #   每个组都会落到别的地方。
        # ring_start = arena_rows + 该类的条目偏移 − 该类层基行的锚点 − slot_origin × slot_rows：
        #   "窗口在槽位里从哪行开始"减去"视图已经从哪行开始"，于是 基行 + index 正好落到槽位里的窗口行
        cls = self.classes[ratio]
        anchor = 0 if ratio == DENSE_RATIO else cls.envelope_offset
        return WindowParams(
            ring_start=self.arena_rows
            + cls.entry_offset
            - anchor
            - self.slot_origin * self.slot_rows,
            slot_rows=self.slot_rows,
            ring_slots=self.ring_slots,
            ring_stride=cls.ring_stride,
            run_rows=cls.num_layers * cls.ring_stride,
        )

    def field_window_params(
        self, field_offset_rows: int, rows_per_window_row: int
    ) -> WindowParams:
        """A window that lives in a state field rather than in the entry.

        A layer whose KV is wider than a plane's row needs
        `rows_per_window_row` of them per ring position, so it takes bytes off
        the front of the slot like the compressor state and is read through a
        view of the plane retyped to its own width, where those rows are one.
        The same five scalars still describe it, counted in that view's rows:
        the slot is that many times shorter and the ring starts at the field's
        own offset.

        A ratio of one — a layer whose KV happens to match the plane — is not a
        special case: the expression below collapses to the field's offset,
        which is what `window_params` would give a class placed there. Nothing
        here branches on a dtype.
        """
        # 【译】住在状态字段里、而不在条目里的窗口。
        #   KV 比平面一行还宽的层，每个环位置需要 rows_per_window_row 个平面行，所以它像压缩器状态一样从槽位前部占用
        #   字节，并通过一个按它自己的宽度重解释过的平面视图来读——在那个视图里那几行就是一行。同样的五个标量仍然
        #   描述它，只是以那个视图的行为单位：槽位短了那么多倍，环从字段自己的偏移开始。
        #   比例为 1——一层的 KV 恰好和平面一样宽——不是特例：下面的表达式会退化成字段的偏移，也就是 window_params
        #   会给一个放在那里的类的值。这里没有任何按 dtype 的分支。
        # 例：DSpark 草稿层要 bf16 窗口，一个位置 512 × 2B = 1024 B，而 fp8 的 NoPE 平面一行 512 B → rows_per_window_row = 2
        # 调用方：deepseek_v4_attn.py（状态字段里的窗口，_field_window_layers）
        if rows_per_window_row < 1:
            raise ValueError(
                f"rows per window row must be positive, got {rows_per_window_row}"
            )
        for name, rows in (
            ("slot", self.slot_rows),
            ("field offset", field_offset_rows),
        ):
            if rows % rows_per_window_row:
                raise ValueError(
                    f"{name} of {rows} rows does not divide by the "
                    f"{rows_per_window_row} plane rows one window row takes"
                )
        slot_rows = self.slot_rows // rows_per_window_row
        return WindowParams(
            ring_start=field_offset_rows // rows_per_window_row
            - self.slot_origin * slot_rows,
            slot_rows=slot_rows,
            ring_slots=self.ring_slots,
            # One run: the ring is contiguous here, since a field is not
            # interleaved with anything the way a class's layers are.
            # 【译】只有一段：这里的环是连续的，因为字段不像类里的各层那样和别的东西交错。
            ring_stride=self.ring_slots,
            run_rows=self.ring_slots,
        )

    def entry_row_runs(self) -> list[tuple[int, int]]:
        """`(start, count)` runs of entry rows any window reaches, in order.

        Every class's runs, offset into the entry and merged where they abut.
        The complement is interleave padding — see `ClassLayout.entry_row_runs`
        for why nothing reaches it. `classes` is built in entry order, so
        walking it is walking the entry.
        """
        # 【译】任何窗口能到达的条目行，按顺序以 (start, count) 段给出。
        #   每个类的段平移到条目内的位置，相接处合并。剩下的补集是交错带来的填充——为什么没人到达它见
        #   ClassLayout.entry_row_runs。classes 是按条目顺序建的，所以遍历它就是在遍历条目。
        # 例：V4-Pro 式、窗口 128 → [(0, 7936)] 一段（没有填充）；加投机后 CSA / HCA 的尾段才会把它拆开
        # 调用方：deepseek_v4_attn.py（checkpoint 只拷这些行；docs/scheduling_kv_cache_guide.md "State checkpoints" 称
        #   DSpark 配置下可省掉条目的 17.3%）
        return merge_abutting(
            (cls.entry_offset + start, count)
            for cls in self.classes.values()
            for start, count in cls.entry_row_runs()
        )

    def physical_slot(self, group: int) -> int:
        """Where pool group `group` sits in the plane.

        Groups run 0..num_slots-1 with 0 the one furthest from the compressed
        blocks, so growing the pool takes the next position *down* and shrinking
        gives back the one that abuts the gap. Group order and position order
        are therefore opposites, and this is the only place they meet.
        """
        # 【译】池的第 group 组在平面里的位置。
        #   组号从 0 到 num_slots-1，0 是离压缩 block 最远的那个，所以池增长时取往下的下一个位置，缩小时交还紧挨空隙
        #   的那个。组的顺序和位置的顺序因此正好相反，而这里是两者唯一交汇的地方。
        # 组号就是 BlockManager 发给请求的状态槽位下标（seq.state_slots，见第4批 state_pool.py）
        # 例：slot_positions = 7、num_slots = 4 → 组 0, 1, 2, 3 → 位置 6, 5, 4, 3
        # 调用方：deepseek_v4_attn.py 4 处、vllm 桥接；deepseek_v4_attn.py 另有一处把一批请求的组号向量化换算成
        #   slot_positions − 1 − pool_slots（同一公式的内联版）
        if not 0 <= group < self.num_slots:
            raise ValueError(f"group {group} outside 0..{self.num_slots}")
        return self.slot_positions - 1 - group

    def slot_span(self, slot: int) -> tuple[int, int]:
        """`[start, stop)` plane rows a request occupies, state and windows.

        One range rather than one per class or per layer: a slot is the unit a
        checkpoint copies and a PD transfer registers.
        """
        # 【译】一个请求占用的平面行 [start, stop)，状态和窗口都含在内。
        #   一个范围，而不是每类或每层一个：槽位是 checkpoint 拷贝和 PD 传输注册的单位。例：位置 6 → [47616, 55552)
        start = (slot - self.slot_origin) * self.slot_rows
        return start, start + self.slot_rows

    def arena_span(self, slot: int) -> tuple[int, int]:
        """`[start, stop)` plane rows holding a request's compressor state."""
        # 【译】存放一个请求压缩器状态的平面行 [start, stop)（槽位最前面的 arena_rows 行）。
        start = (slot - self.slot_origin) * self.slot_rows
        return start, start + self.arena_rows

    def absolute_row(self, layer_id: int, index: int) -> int:
        """Plane row an index reaches — what the two address spaces share."""
        # 【译】一个下标到达的平面行——两个地址空间（压缩行与窗口行）共享的东西：基行 + 下标。
        return self.layer_base_row(layer_id) + index
