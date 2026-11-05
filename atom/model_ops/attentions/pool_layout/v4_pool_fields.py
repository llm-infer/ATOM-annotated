# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What one DeepSeek-V4 KV row and one indexer block hold.

The sibling of `v4_pool_geometry`, which says where rows go; this says what a
row is. Declaration only -- which of these a build wants is policy and stays
with the backend (the gfx fallback off fp8, `--index_cache_dtype`, whether the
GPU has an FP4 indexer), so nothing here has a branch of its own.

`MAIN_KV_NOPE` and `MAIN_KV_ROPE` are the `semantic_role` a PD consumer matches
its producer's regions on: a wire format between two processes that may be
running different builds, not free to rename.
"""

# 【译】DeepSeek-V4 的一行 KV 和一个索引器 block 里装着什么。
#
#   它是 v4_pool_geometry 的兄弟：那个说行放在哪，这个说一行是什么。只有声明——某次构建要用其中哪些是
#   策略问题，留给后端决定（不支持时从 fp8 回退的 gfx 判断、--index_cache_dtype、GPU 有没有 FP4 索引器），
#   所以这里没有任何自己的分支。
#
#   MAIN_KV_NOPE 和 MAIN_KV_ROPE 是 PD 消费端用来匹配生产端区域的 semantic_role：它们是两个可能运行着不同
#   构建的进程之间的线上格式，不能随意改名。
#
# =====================================================================================
# 【导读】v4_pool_fields.py —— V4 的"一行"和"一个索引器 block"的字节构成（第5批 5b，B级）
#   前置：V4 的 CSA / HCA 与索引器见 config.py Config.__post_init__ 第12步 V4 部分的【科普】；
#     FP8 / scale 见 arg_utils.py kv_cache_dtype 上方的【科普】；V4 的行空间见 pool_layout/__init__.py【科普】④。
#   一行主 KV（某层的一个压缩后 token，或一个窗口位置）：
#     fp8 构建： NoPE 平面 512 B（MAIN_KV_NOPE，(512,) fp8）  +  RoPE 平面 128 B（MAIN_KV_ROPE，(64,) bf16）
#                两个平面用同一个行号寻址（RoPE 不量化，所以单独一个 bf16 平面）
#     bf16 构建：一个平面 1024 B（(512,) bf16，RoPE 内联在这一行里）
#   CSA 索引器一个 block（一层；block_size 256 → 256 / 4 = 64 个压缩行；index_head_dim 128）：
#     fp8         ：[数据 64 × 128 B = 8192 B][scale 64 × 4 B = 256 B]               = 8448 B（132 B / 行）
#     fp4（gfx950）：[数据 (1, 4, 64, 16) = 4096 B][scale (1, 4, 64) = 256 B]         = 4352 B（68 B / 行）
#     fp4（gfx1250）：同样的字节数，按自然行序 [(64, 64)][(64, 4)]
#   谁用：deepseek_v4_attn.py（builder __init__ 声明平面与索引器字段、算 _indexer_block_bytes）、models/deepseek_v4.py、
#     config.py（MQA_LOGITS_PRESHUFFLE_ROWS）、kimi_mla_gdn_attn.py、v41_pool_geometry.py
# =====================================================================================

from __future__ import annotations

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

MAIN_KV_NOPE = "dsv4.main_kv.nope"
MAIN_KV_ROPE = "dsv4.main_kv.rope"

# Rows per block that `deepgemm_fp8_paged_mqa_logits` requires to stay in its
# preshuffled layout, which is the only layout it computes correctly -- with
# `Preshuffle=False` it disagrees with the flat `fp8_mqa_logits` kernel by ~100%
# at every block size, and aiter's assert guards only the preshuffle side. A
# cache that kernel pages over holds a multiple of this many rows per block, or
# exactly 8 -- the one shorter page it assembles a tile from.
#
# Here rather than beside one of its enforcers, because there are several: a
# block size (`atom.config`), a pooled row count (Kimi/GLM) and a CSA2 index
# plane all have to agree with it. Only CSA2 takes the short page.
# 【译】deepgemm_fp8_paged_mqa_logits 要保持其预重排（preshuffled）布局所要求的每 block 行数；预重排是它唯一
#   能算对的布局——Preshuffle=False 时，它在任何 block 大小下都和扁平的 fp8_mqa_logits kernel 相差约 100%，
#   而 aiter 的断言只守着预重排那一侧。这个 kernel 分页读取的缓存，每个 block 的行数要么是它的倍数，要么
#   恰好是 8——那是它能拼出一个 tile 的唯一一种更短的页。
#
#   放在这里而不是放在某个执行者旁边，是因为执行者有好几个：一个 block 大小（atom.config）、一个池化后的
#   行数（Kimi/GLM）和一个 CSA2 索引平面都必须与它一致。只有 CSA2 使用那种短页。
# 三个执行者（已核实）：
#   config.py glm5_kpool_block_size：GLM-5.3 的 block_size = index_kpool × 16 = 64（为什么见它上方的【科普】④——
#     那里说"每 block 索引行数必须是 16 的倍数"，对 GLM-5.3 成立；"恰好 8 行"的短页只给下面的 CSA2）
#   kimi_mla_gdn_attn.py：池化后每 block 的行数 block_size // index_kpool 不是 16 的倍数就报错
#   v41_pool_geometry.V41PoolGeometry.__post_init__：index_block_rows 必须是 16 的倍数或恰好 8
MQA_LOGITS_PRESHUFFLE_ROWS = 16

# Internal: FP8 keeps both regions in one pool and so has fewer PD roles than
# it has regions, which is why those are spelled out where the pools are.
# 【译】内部使用：FP8 把两个区域放在一个池里，因此它的 PD 角色比区域少，这就是为什么那些角色在定义池的
#   地方才写出来。
CSA_INDEXER_DATA = "csa_indexer_data"
CSA_INDEXER_SCALE = "csa_indexer_scale"

# FP4 索引器的两种字节排法（同样的字节数，顺序不同）：gfx950 用 pa_mqa_logits_fp4 的预重排；gfx1250 用自然行序
FP4_GFX950_PRESHUFFLE = "fp4-gfx950-preshuffle"
FP4_GFX1250_NATURAL = "fp4-gfx1250-natural"


def fp4_indexer_layout_for_arch(arch: str) -> str:
    """Return the on-wire FP4 indexer layout for a GPU architecture."""
    # 【译】返回某个 GPU 架构上 FP4 索引器的线上布局。
    # 调用方：deepseek_v4_attn.py（builder 的 indexer_layout）、models/deepseek_v4.py
    if arch == "gfx1250":
        return FP4_GFX1250_NATURAL
    return FP4_GFX950_PRESHUFFLE


def main_kv_plane_fields(
    head_dim: int,
    main_dtype: torch.dtype,
    rope: tuple[int, torch.dtype] | None = None,
) -> list[EntryField]:
    """One row of each plane the main KV pool has, in declared order.

    How many there are is the single answer the carve, the state split, the
    checkpoint copy and the PD registration all take from here. `rope` is the
    second plane's `(width, dtype)`, or None on a build that keeps RoPE inline
    in the NoPE row -- a pair so that half a plane cannot be described.
    """
    # 【译】主 KV 池每个平面的一行，按声明顺序。
    #   平面有几个，这个唯一答案被切分、状态拆分、checkpoint 拷贝和 PD 注册统统从这里拿。rope 是第二个平面的
    #   (宽度, dtype)；对把 RoPE 内联在 NoPE 行里的构建则为 None——用一个二元组，这样就没法只描述半个平面。
    # 例：fp8 构建 main_kv_plane_fields(512, fp8, (64, bf16)) → [NoPE (512,) fp8 = 512 B, RoPE (64,) bf16 = 128 B]；
    #   bf16 构建 main_kv_plane_fields(512, bf16) → [NoPE (512,) bf16 = 1024 B]
    # 调用方：deepseek_v4_attn.py builder __init__（self._plane_fields；_plane_row_widths 由它得出每平面行宽）
    fields = [EntryField(MAIN_KV_NOPE, 1, (head_dim,), main_dtype)]
    if rope is not None:
        rope_head_dim, rope_dtype = rope
        fields.append(EntryField(MAIN_KV_ROPE, 1, (rope_head_dim,), rope_dtype))
    return fields


def fp8_indexer_block_fields(
    rows: int, index_head_dim: int, data_dtype: torch.dtype
) -> list[EntryField]:
    """`[rows*index_head_dim data][rows*4 fp32 scale]`, one layer's block.

    Two regions, NOT interleaved per row, which is how all three consumers
    address it: `fused_compress.py` (write), `cache_kernels.cu:1638/1651`, and
    `pa_mqa_logits.py:493-500`. Both live in one pool, so that pool's row width
    is the block over its rows.
    """
    # 【译】一层的一个 block：[rows*index_head_dim 的数据][rows*4 字节的 fp32 scale]。
    #   两个区域，不按行交错，三个使用方都是这样寻址的：fused_compress.py（写入）、cache_kernels.cu:1638/1651、
    #   pa_mqa_logits.py:493-500。两者住在同一个池里，所以那个池的"行宽"是整个 block 除以行数。
    # （cache_kernels.cu、pa_mqa_logits.py 在 aiter 里，不在本仓库；fused_compress.py 在 model_ops/v4_kernels/）
    # 例：rows = 64、index_head_dim = 128 → 数据 8192 B + scale 256 B = 8448 B，即每行 132 B（recipes/DeepSeek-V4.md
    #   说的"132 B / token"，这里的 token 是压缩后的一行）
    # 调用方：deepseek_v4_attn.py（索引器 fp8 时）、v41_pool_geometry.index_planes（V4.1 的 FP8 索引平面）
    return [
        EntryField(CSA_INDEXER_DATA, 1, (rows, index_head_dim), data_dtype),
        EntryField(CSA_INDEXER_SCALE, 1, (rows,), torch.float32),
    ]


def fp4_indexer_block_fields(
    rows: int,
    index_head_dim: int,
    layout: str = FP4_GFX950_PRESHUFFLE,
) -> list[EntryField]:
    """Packed E2M1 plus one e8m0 byte per group of 32, one layer's block.

    gfx950 keeps the legacy `pa_mqa_logits_fp4` preshuffle. gfx1250 OPUS reads
    natural rows: 64 packed E2M1 bytes and four E8M0 bytes for D=128. One pool
    per region here, so a region's shape is a pool's shape after the layer and
    block axes.
    """
    # 【译】一层的一个 block：打包的 E2M1，外加每 32 个值一个 e8m0 字节。
    #   gfx950 保留旧的 pa_mqa_logits_fp4 预重排布局。gfx1250 的 OPUS 读自然行序：D=128 时每行 64 个打包 E2M1
    #   字节加 4 个 E8M0 字节。这里每个区域一个池，所以区域的形状就是池在层轴和 block 轴之后的形状。
    # E2M1：4 位浮点（1 符号、2 指数、1 尾数），两个值打包进一个字节；e8m0：只有 8 位指数的 scale（即 2 的幂）
    # 例：rows = 64、D = 128 → k_tiles = 1：数据 (1, 4, 64, 16) = 4096 B、scale (1, 4, 64) = 256 B → 每行 68 B
    # 调用方：deepseek_v4_attn.py（索引器 fp4 时）
    if index_head_dim % 128 != 0:
        raise ValueError(
            f"FP4 index_head_dim must be a multiple of 128, got {index_head_dim}"
        )
    # 预重排布局按 128 维一个 K tile 组织，每个 tile 拆成 4 组 × 每行 16 字节（= 32 个 E2M1 值）
    k_tiles = index_head_dim // 128
    if layout == FP4_GFX1250_NATURAL:
        return [
            EntryField(CSA_INDEXER_DATA, 1, (rows, index_head_dim // 2), torch.uint8),
            EntryField(CSA_INDEXER_SCALE, 1, (rows, index_head_dim // 32), torch.uint8),
        ]
    if layout != FP4_GFX950_PRESHUFFLE:
        raise ValueError(f"unknown FP4 indexer layout {layout!r}")
    return [
        EntryField(CSA_INDEXER_DATA, 1, (k_tiles, 4, rows, 16), torch.uint8),
        EntryField(CSA_INDEXER_SCALE, 1, (k_tiles, 4, rows), torch.uint8),
    ]


def indexer_block_regions(fields: list[EntryField]) -> tuple[dict[str, int], int]:
    """Where each region starts inside one layer's block, and the block's size.

    Packed, with nothing between regions -- deliberately not the aligned walk
    `entry_arena.field_extents` does. Every consumer recomputes a region's
    start from `rows` and the head dim rather than reading one, so a gap would
    go unnoticed: a scale view would begin one gap early and read
    valid-looking numbers rather than crash.
    """
    # 【译】每个区域在一层的 block 里从哪里开始，以及 block 的大小。
    #   紧密排列，区域之间没有任何东西——有意不用 entry_arena.field_extents 那种带对齐的遍历。每个使用方都是用
    #   rows 和 head dim 重新算出区域起点，而不是读取它，所以一段空隙不会被发现：scale 视图会提前一个空隙开始，
    #   读到看起来正常的数字，而不是崩溃。
    # 例：fp8、rows 64 → ({"csa_indexer_data": 0, "csa_indexer_scale": 8192}, 8448)
    # 调用方：deepseek_v4_attn.py（_indexer_regions、_indexer_block_bytes）、v41_pool_geometry.index_planes
    offsets: dict[str, int] = {}
    total = 0
    for field in fields:
        offsets[field.name] = total
        total += field.bytes_per_entry
    return offsets, total
