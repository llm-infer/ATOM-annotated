# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The per-token index arrays a decode step has to derive.

The counterpart of `.prefill`, on a rectangular token axis: a decode step runs
the same `max_seqlen_q` rows for every sequence, so sequence `i` forwards
`[context_lens[i] - max_seqlen_q, context_lens[i])` and the flat axis is that
range concatenated in sequence order.

The slot each of these positions maps to is `.slots`, shared with `.prefill`:
the gather reads a position and a packed table, and neither cares which side
shaped the axis. What stays with the builders is the one-token step, one
position per row computed inline from the ragged rows, around which the MLA one
wraps a DCP variant.
"""

# 【译】一个 decode 步需要推导的逐 token 下标数组。
#
#   .prefill 的对应物，在一条矩形的 token 轴上：decode 步对每条序列都跑同样的 max_seqlen_q 行，所以序列 i 送入
#   [context_lens[i] - max_seqlen_q, context_lens[i])，扁平轴就是这些区间按序列顺序拼起来。
#
#   这些位置各自对应的槽位由 .slots 给出，与 .prefill 共用：gather 读的是位置和一张打包好的表，两者都不关心轴是
#   哪一边形成的。留在 builder 里的是单 token 的一步——每行一个位置，从参差的行内联算出——MLA 的 builder 还在它
#   外面包了一个 DCP 变体。

from __future__ import annotations

import numpy as np


def decode_positions(
    context_lens: np.ndarray, max_seqlen_q: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence.

    Broadcast rather than `tile(ramp) + repeat(start)`: the rectangle IS a 2-D
    shape, so writing it as one leaves the token axis touched once instead of
    three times (1.4-2.2x, `/app/logs_claude/tool/decode_positions_perf.py`).
    """
    # 【译】每个 token 在它"自己"那条序列里的绝对下标。
    #   用广播而不是 tile(ramp) + repeat(start)：这个矩形"本身"就是二维形状，按二维写，token 轴只被碰一次而不是三次
    #   （快 1.4–2.2 倍，/app/logs_claude/tool/decode_positions_perf.py——该脚本不在本仓库里）。
    # 输入：context_lens [B]（已含本步要送的 token，投机解码时已减去上一轮被拒的草稿数）；max_seqlen_q = 每条序列这一步
    #   送几个 token（1 + 投机步数）；out = 可选 [B × max_seqlen_q] int32 缓冲
    # 输出：[B × max_seqlen_q] int32。例：context_lens = [23, 43]、max_seqlen_q = 3 → [20, 21, 22, 40, 41, 42]
    # 调用方：aiter_attention.py AiterAttentionMetadataBuilder.prepare_decode、aiter_mla.py 的 decode 路径
    # （np.add(…, out=…reshape(B, max_seqlen_q))：列向量 [B, 1] + 行向量 [max_seqlen_q] 广播成矩形，直接写进 out）
    bs = len(context_lens)
    out = np.empty(bs * max_seqlen_q, dtype=np.int32) if out is None else out
    np.add(
        (context_lens - max_seqlen_q)[:, None],
        np.arange(max_seqlen_q, dtype=np.int32),
        out=out.reshape(bs, max_seqlen_q),
    )
    return out
