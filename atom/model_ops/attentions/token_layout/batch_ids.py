# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The map between a token axis and the sequence axis.

Every other per-token array in this package is indexed by token; almost every
quantity the scheduler hands over is indexed by sequence. This converts between
them, so a kernel can resolve `per_seq[batch_id[t]]`.

There are TWO axes, named after the `cu_seqlens_q` / `cu_seqlens_k` each follows:
query lengths in gives a `batch_id_per_q_token`, context lengths a
`batch_id_per_k_token`. They are not interchangeable, and mixing them does not
fault -- it silently resolves a token to another request's row.

Which builder depends on where the result must live. `build_batch_ids` (host)
covers any width bounded by `max_num_batched_tokens`, which a pinned
`CpuGpuBuffer` can hold; every q-axis caller is one. `build_batch_ids_device`
covers the `total_kv` width, which nothing caps -- no fixed buffer fits it, a
pageable upload that size would sit on the critical path, and its input is
already on the device.
"""

# 【译】token 轴与序列轴之间的映射。
#
#   本包里其他每个逐 token 数组都按 token 索引；调度器交过来的几乎每个量都按序列索引。这里在两者之间转换，让 kernel
#   可以解析 per_seq[batch_id[t]]。
#
#   有"两条"轴，以各自对应的 cu_seqlens_q / cu_seqlens_k 命名：传入 query 长度得到 batch_id_per_q_token，传入
#   context 长度得到 batch_id_per_k_token。它们不可互换，而且混用不会出错——它会悄悄把一个 token 解析到另一个请求的行。
#
#   用哪个构建函数取决于结果必须住在哪里。build_batch_ids（host）覆盖任何以 max_num_batched_tokens 为上界的宽度，
#   一个锁页的 CpuGpuBuffer 装得下；所有 q 轴的调用方都是这种。build_batch_ids_device 覆盖 total_kv 宽度，那没有
#   任何上限——没有固定缓冲区装得下，那么大的可分页上传会卡在关键路径上，而且它的输入本来就在 device 上。
# 例（token_layout/__init__.py【科普】①）：seqlens_q = [20, 8] → q 轴 [0 × 20, 1 × 8]；context_lens = [20, 40] → k 轴 [0 × 20, 1 × 40]

from __future__ import annotations

import numpy as np
import torch


def build_batch_ids(
    seqlens: np.ndarray,
    pad_to: int | None = None,
    pad: int = -1,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Which sequence each token of a flat token axis belongs to.

    Sequence `i` contributes `seqlens[i]` consecutive entries of value `i`, in
    sequence order -- the layout attention reads through the matching
    `cu_seqlens_*`.

    `pad_to` widens the result to a CUDAGraph bucket, filling the tail with
    `pad`. The tail is NOT sequence 0: a captured step runs at the bucket width
    whatever the batch, so a fabricated token naming a real sequence would have
    every consumer resolve it to that request's row and quietly attend on its
    behalf. `-1` is what the kernels bail on.
    """
    # 【译】一条扁平 token 轴上的每个 token 属于哪条序列。
    #   序列 i 贡献 seqlens[i] 个连续的、值为 i 的项，按序列顺序排列——也就是注意力通过对应的 cu_seqlens_* 读取的布局。
    #   pad_to 把结果加宽到一个 CUDA Graph 档位，尾部填 pad。尾部"不是"序列 0：捕获的一步不管 batch 多大都按档位宽度
    #   运行，一个指向真实序列的伪造 token 会让每个使用方都把它解析到那个请求的行，悄悄地替它做注意力。-1 是 kernel
    #   看到就跳过的值。
    # 输入：seqlens [B] = 每条序列的 token 数；pad_to = 档位宽度（None 表示不补）；out = 可选的预分配 int32 缓冲
    # 输出：[pad_to 或 sum(seqlens)] 的 int32 数组。例：seqlens = [20, 8]、pad_to = 32 → [0 × 20, 1 × 8, −1 × 4]
    # 调用方：backends.py CommonAttentionBuilder.publish_batch_ids（写进 forward_vars["batch_id_per_q_token"] 并上传；
    #   aiter_attention.py、aiter_mla.py 的 decode / prefill 路径调它）、deepseek_v4_attn.py（多处，含 TBO 的各 ubatch）、
    #   deepseek_v41/metadata.py、qwen4_exp_attn.py、models/deepseek_v41/bounded_replay.py
    # 读 −1 的 kernel 侧：例如 attention_mla.py、dcp_ops.py 里"`-1` marks a CUDAGraph pad token"处
    total = int(seqlens.sum())
    width = total if pad_to is None else pad_to
    # Asserted, not raised: numpy refuses the write below on its own, so this
    # only trades its message for a better one. The dtype guard in `.prefill`
    # is a `raise` because there numpy would answer, wrongly and in silence.
    # 【译】用 assert 而不是 raise：下面那次写入 numpy 自己就会拒绝，所以这里只是把它的报错信息换成一个更好的。
    #   .prefill 里的 dtype 检查用 raise，是因为在那里 numpy 会给出答案——错误的、悄无声息的答案。
    # 观察：prefill.py 里没有 dtype 检查；用 raise 做 dtype 检查的是 slots.py 的 slot_mapping（block_tables 必须是 int32），
    #   这里的 ".prefill" 应指 ".slots"。
    assert width >= total, f"pad_to={pad_to} is under the {total} tokens scheduled"
    out = np.empty(width, dtype=np.int32) if out is None else out[:width]
    # np.repeat(arange(B), seqlens)：把 0, 1, …, B−1 分别重复 seqlens[i] 次，例 repeat([0, 1], [20, 8])
    out[:total] = np.repeat(np.arange(len(seqlens), dtype=np.int32), seqlens)
    out[total:] = pad
    return out


def build_batch_ids_device(seqlens: torch.Tensor, *, total: int) -> torch.Tensor:
    """`build_batch_ids` for a result too wide to stage through a host buffer.

    No `pad_to`: the gather it feeds is exactly this long, so there is no
    CUDAGraph bucket to pad out to and no sentinel to write.

    `total` is `seqlens.sum()`, and required rather than optional: without it
    `repeat_interleave` synchronizes the device to read its own output length
    back, once per forward between the metadata and the first layer. Every
    caller already holds the sum on the host -- omitting it would only be
    slower, which is the kind of thing a default hides.
    """
    # 【译】给太宽、没法经 host 缓冲暂存的结果用的 build_batch_ids。
    #   没有 pad_to：它喂给的 gather 恰好就这么长，所以没有 CUDA Graph 档位要补，也不需要写哨兵值。
    #   total 就是 seqlens.sum()，而且是必填而不是可选：没有它，repeat_interleave 要同步 device 来读回自己的输出长度，
    #   每次 forward 在元数据和第一层之间同步一次。每个调用方都已经在 host 上持有这个和——省略它只会更慢，而这正是
    #   默认值会掩盖的那种事。
    # 输入：seqlens = device 上的 [B]（context_lens）；total = host 上已知的 total_kv。输出：device 上的 [total] int32
    # 例：seqlens = [20, 40]、total = 60 → [0 × 20, 1 × 40]
    # 调用方：backends.py CommonAttentionBuilder.prepare_prefill（有前缀命中或 TBO 时生成 batch_id_per_k_token）、
    #   以及 plugin/sglang/models/qwen3_5_attention.py
    return torch.repeat_interleave(
        torch.arange(seqlens.numel(), dtype=torch.int32, device=seqlens.device),
        seqlens.to(torch.int64),
        output_size=total,
    )
