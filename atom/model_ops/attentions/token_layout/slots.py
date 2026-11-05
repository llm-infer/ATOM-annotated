# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which KV slot each of this step's tokens is written to.

One expression for both sides of the forward. The two differ only in how the
token axis is shaped -- `.prefill` builds a ragged one, `.decode` a rectangle
-- and neither difference reaches the gather, which reads positions and a
packed table. Keeping it in one place is not tidiness: a slot computed two ways
is a KV write landing on another request's row, and nothing faults.
"""

# 【译】这一步的每个 token 的 KV 写到哪个槽位。
#
#   forward 的两边用同一个表达式。两者只在 token 轴的形状上不同——.prefill 构建参差的，.decode 构建矩形的——而这些
#   差别都到不了 gather，gather 读的是位置和一张打包好的表。把它放在一处不是为了整洁：一个用两种方式算出来的槽位，
#   就是一次落在另一个请求的行上的 KV 写入，而且什么都不会报错。

from __future__ import annotations

import numpy as np


def slot_mapping(
    positions: np.ndarray,
    seqlens_q: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """`block_table[pos // block_size] * block_size + pos % block_size`.

    `block_tables` is the packed 2-D buffer as `prepare_block_tables` leaves it,
    so a row's base is its index times the buffer's stride and no second marshal
    of the same rows happens here -- which is also why a decode step marshals
    BEFORE it asks for slots, and why a caller that trimmed its own ragged rows
    (rejected drafts) may still pass the untrimmed table: a token's block index
    stays under its own row's trimmed length, so the entries read are the same
    either way. Flattening the table has two ways to go wrong and
    they need separate guards: `.cast` refuses a non-contiguous buffer, where
    `reshape(-1)` would hand back a copy and the gather silently read zeros, but
    it reinterprets a wider dtype rather than rejecting it -- an int64 table
    reads as twice as many int32s and answers without a word.

    `out` and `scratch` are two int64 buffers of the token axis. Taking the
    offset first lets the block index land in `scratch` and be overwritten in
    place by the slot it gathers, so the only allocations left are the repeat
    and the gather itself.

    Widening at the multiply rather than after it: a block id is int32 and
    `id * block_size` overflows that for a large enough pool, which numpy would
    wrap without a word.
    """
    # 【译】block_table[pos // block_size] * block_size + pos % block_size。
    #
    #   block_tables 是 prepare_block_tables 留下的那个打包好的二维缓冲，所以一行的起点就是它的行号乘以缓冲的步长，
    #   这里不会把同样的行再整理一遍——这也是为什么 decode 步要在请求槽位"之前"先整理表，以及为什么一个自己裁剪过参差
    #   行（被拒的草稿）的调用方仍然可以传入未裁剪的表：一个 token 的 block 下标始终小于它自己那一行裁剪后的长度，所以
    #   不管哪种情况读到的项都一样。展平这张表有两种出错方式，需要分别把守：.cast 拒绝非连续的缓冲区（reshape(-1) 会
    #   返回一个副本，gather 就会悄悄读到零），但它会把更宽的 dtype 重新解释而不是拒绝——一张 int64 的表会被读成两倍
    #   多的 int32，并且一声不吭地给出答案。
    #
    #   out 和 scratch 是两个 token 轴长度的 int64 缓冲。先算偏移，让 block 下标落进 scratch，再被它 gather 到的槽位
    #   原地覆盖，于是剩下的分配只有那次 repeat 和 gather 本身。
    #
    #   在乘法时就加宽、而不是乘完再加宽：block id 是 int32，对足够大的池 id * block_size 会溢出 int32，而 numpy 会
    #   一声不吭地回绕。
    # 输入：positions [N]（prefill_positions / decode_positions 的结果）；seqlens_q [B]（每条序列这一步的 token 数，
    #   决定每个 token 属于表的哪一行）；block_tables [max_bs, 列数] int32（打包好的表，空位补 0）；block_size；
    #   out / scratch：可选的 [N] int64 缓冲（backends.py 传 forward_vars["slot_mapping"].np 与 token_axis_scratch）
    # 输出：[N] int64 槽位。例（token_layout/__init__.py【科普】①，表的步长 = 4 列）：B 是第 1 行，pos 32 →
    #   blk = (32 >> 4) + 1 × 4 = 6 → flat[6] = 9 → 9 × 16 + (32 & 15) = 144
    # 调用方：backends.py CommonAttentionBuilder._write_prefill_slots（非 DCP 的 prefill）、aiter_attention.py /
    #   aiter_mla.py 的 prepare_decode（max_seqlen_q > 1，即投机解码的矩形 decode）
    # 对比nano-vllm：nano-vllm 在 prepare_prefill 里按 block 用 range(slot_start, slot_end) 逐段 extend，decode 时
    #   block_table[-1] * block_size + last_block_num_tokens − 1；这里一个向量化公式覆盖两边（单 token decode 仍在 builder 里内联）
    if block_tables.dtype != np.int32:
        # Raised, not asserted, to match `pack_rows` on the other side of this
        # buffer and to survive `python -O` as the contiguity check does.
        # 【译】用 raise 而不是 assert，是为了和这个缓冲区另一侧的 pack_rows 保持一致，也为了像连续性检查那样在
        #   python -O 下依然有效。（pack_rows 在 atom/utils/__init__.py，负责把参差的 block_table 写进这个打包缓冲）
        raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
    n = positions.shape[0]
    stride = block_tables.shape[1]
    # flat：把二维表零拷贝地看成一维 int32；memoryview.cast 只接受 C 连续缓冲（否则抛错），这就是"连续性检查"
    flat = np.asarray(memoryview(block_tables).cast("B").cast("i"))
    out = np.empty(n, dtype=np.int64) if out is None else out
    blk = np.empty(n, dtype=np.int64) if scratch is None else scratch[:n]
    # Splitting the position is the step's largest array op, and `np.divmod` on
    # int64 costs 6x a shift and a mask. Positions are non-negative, so the two
    # agree whenever the block size is a power of two -- which every shipped
    # config is, though nothing rejects one that is not.
    # 【译】拆分位置是这一步最大的数组运算，而 int64 上的 np.divmod 比一次移位加一次按位与贵 6 倍。位置是非负的，所以
    #   只要 block size 是 2 的幂，两者结果一致——所有已发布的配置都是，尽管没有任何东西拒绝不是的配置。
    # 例：block_size 16 → offset = pos & 15、block 下标 = pos >> 4（16 的 bit_length 是 5，减 1 得移位 4）；
    #   block_size & (block_size − 1) 非 0 就说明不是 2 的幂，走通用的取余 / 整除
    if block_size & (block_size - 1):
        np.remainder(positions, block_size, out=out)
        np.floor_divide(positions, block_size, out=blk)
    else:
        np.bitwise_and(positions, block_size - 1, out=out)
        np.right_shift(positions, int(block_size).bit_length() - 1, out=blk)
    # 加上每个 token 所在行的起点（行号 × 步长），blk 变成扁平表里的下标；再 gather 出物理 block 号、乘 block_size、加偏移
    blk += np.repeat(np.arange(len(seqlens_q), dtype=np.int64) * stride, seqlens_q)
    np.multiply(flat[blk], block_size, out=blk, dtype=np.int64)
    return np.add(blk, out, out=out)
