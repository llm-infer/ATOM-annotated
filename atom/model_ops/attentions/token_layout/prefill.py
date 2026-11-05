# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a prefill step's tokens sit in their own sequences.

Sequence `i` forwards tokens `[num_cached_tokens[i], context_lens[i])`, so the
axis is that ragged range flattened in sequence order -- the layout attention
reads through `cu_seqlens_q`. Everything per-sequence is an input: this file
derives nothing the step already knows. The slot each of these positions maps
to is `.slots`, shared with `.decode`.
"""

# 【译】一个 prefill 步的 token 在它们各自的序列里处在什么位置。
#
#   序列 i 送入 token [num_cached_tokens[i], context_lens[i])，所以这条轴就是这些参差的区间按序列顺序摊平——
#   也就是注意力通过 cu_seqlens_q 读取的布局。每条序列的量都是输入：本文件不推导任何这一步已经知道的东西。这些
#   位置各自对应的槽位由 .slots 给出，与 .decode 共用。
# 对比nano-vllm：nano-vllm 在 prepare_prefill 里 positions.extend(range(start, end)) 逐条序列拼；这里一次 repeat + 一次 add

from __future__ import annotations

import numpy as np


def prefill_positions(
    token_offsets: np.ndarray,
    cached_lens: np.ndarray,
    cu_seqlens_q: np.ndarray,
    seqlens_q: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence.

    Token `t` of sequence `i` is at `cached_lens[i] + (t - cu_seqlens_q[i])`, so
    a chunked prefill resumes at the cached prefix rather than at zero. The
    whole per-sequence part is one repeat, leaving the token axis read once and
    written once. `token_offsets` is the flat axis, `arange(sum(seqlens_q))`,
    passed in because the caller already owns a resident one.
    """
    # 【译】每个 token 在它"自己"那条序列里的绝对下标。
    #   序列 i 的第 t 个 token 位于 cached_lens[i] + (t - cu_seqlens_q[i])，所以 chunked prefill 从已缓存的前缀处接着往下，
    #   而不是从 0 开始。整个"按序列"的部分只是一次 repeat，token 轴只读一次、写一次。token_offsets 就是这条扁平轴
    #   arange(sum(seqlens_q))，由调用方传入，因为它已经持有一个常驻的。
    # 输入：token_offsets [N]（ModelRunner.arange_np[:N]）；cached_lens [B]；cu_seqlens_q [B+1]；seqlens_q [B]；out [N]（可选）
    # 输出：[N] positions。例（token_layout/__init__.py【科普】①）：cached_lens = [0, 32]、cu_seqlens_q = [0, 20, 28]、
    #   seqlens_q = [20, 8] → starts = repeat([0 − 0, 32 − 20], [20, 8]) = [0 × 20, 12 × 8] → positions = [0 … 19, 32 … 39]
    # 调用方：backends.py CommonAttentionBuilder.prepare_prefill（写进 forward_vars["positions"]）、deepseek_v41/metadata.py
    starts = np.repeat(cached_lens - cu_seqlens_q[:-1], seqlens_q)
    return np.add(token_offsets, starts, out=out)
