# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fill a decode step's `input_ids` for the sequences carried over on GPU.

A decode step feeds each request `[anchor, draft_0 .. draft_{n-1}]`, where the
anchor is the id that request sampled last step. That id lives in one of two
places:

* a request the previous forward also held ("deferred") sampled it on the GPU
  and it was never copied to the host -- that is the whole point of deferred
  output, which trades a D2H sync for reading `prev_token_ids` in place;
* a request the scheduler admitted since then has its id on the host, already
  written into `batch.scheduled_tokens` at that request's own token offset.

The caller stages `scheduled_tokens` over the whole decode region first, which
is correct by construction for the second kind, then calls this to overwrite
the spans belonging to the first.

Addressing: one program per request, writing `[cu[i], cu[i+1])`. The span comes
from the same `cu` the rest of the step uses, so per-request length is whatever
that says -- uniform `q` for a rectangular MTP step, per-request lengths under
DSpark's ragged buckets, no separate copy of "how long is each one" to drift.

Past `cu[bs]` come the graph's padded rows, zeroed by programs of the same
launch. Nobody else writes them and the MoE path does consume them, so they
would otherwise carry the previous forward's ids; zero is a legal vocab id. One
launch and not two, because the two regions are disjoint by construction.

The predecessor of this kernel placed the two kinds as two contiguous BLOCKS
(`[deferred | new]`, or `[new | deferred]` when the previous forward was a
prefill). The scheduler does not order the batch that way -- the two interleave
-- so every request could receive a neighbour's anchor. It went unseen because
the block layout only comes up when a forward lands between two decode steps
and displaces the previous batch, which is what a chunked prefill does. Writing
each request's own span keeps that unexpressible.
"""

# 【译】为在 GPU 上延续下来的序列填写 decode 步的 `input_ids`。
#
#   decode 步给每个请求喂 `[anchor, draft_0 .. draft_{n-1}]`，其中 anchor 是该请求上一步采样得到的 id。这个 id 在两个地方之一：
#
#   * 上一次 forward 也包含的请求（"deferred"，延续下来的）在 GPU 上采样得到它，而且从没拷回 host——这正是延迟输出的
#     全部意义：用原地读取 `prev_token_ids` 换掉一次 D2H 同步；
#   * 调度器在那之后才接纳的请求，它的 id 在 host 上，已经写进 `batch.scheduled_tokens` 中该请求自己的 token 偏移处。
#
#   调用方先把 `scheduled_tokens` 整体写进整个 decode 区域——对第二种请求这天然就是对的——再调用本函数覆盖属于
#   第一种请求的那些段。
#
#   寻址：每个请求一个 program，写 `[cu[i], cu[i+1])`。段来自本步其余部分也在用的同一个 `cu`，所以每个请求的长度就是
#   它说的那样——矩形 MTP 步是统一的 `q`，DSpark 的 ragged 桶下是各请求自己的长度——不存在另一份"每个有多长"的拷贝
#   可以与之偏离。
#
#   `cu[bs]` 之后是图的补齐行，由同一次 launch 里的另一些 program 清零。没有别人写它们，而 MoE 路径确实会读它们，
#   否则它们会带着上一次 forward 的 id；0 是合法的词表 id。用一次 launch 而不是两次，因为两个区域天然不相交。
#
#   这个 kernel 的前身把两种请求放成两个连续的块（`[deferred | new]`，上一次 forward 是 prefill 时则为 `[new | deferred]`）。
#   调度器并不这样排 batch——两者是交错的——所以每个请求都可能拿到邻居的 anchor。这个问题一直没被发现，因为块布局只在
#   "某次 forward 落在两个 decode 步之间、挤掉了上一个 batch"时才出现，而 chunked prefill 恰恰会这样。改成每个请求写
#   自己的段，这种错误就无从表达了。
#
# 【导读】decode_input_ids.py：延迟输出下，decode 步的 input_ids 怎样直接由上一步留在 GPU 上的采样结果拼出来，不回 CPU
#   调用方：model_runner.py tokenIDProcessor.prepare_input_ids 的 decode 路径（is_deferred_out 为 True 且有 prev_batch 时）；
#     测试 tests/test_decode_input_ids.py 用下面的 CPU 参考实现验证性质
#   前置：延迟输出（原理见 engine_core.py 文件头【3】；GPU 侧怎样记下 prev_batch / prev_token_ids、怎样把每个请求分成
#     deferred / new，见 model_runner.py tokenIDProcessor 上方【科普】）。要点：第 N 步 forward 只等待并返回第 N-1 步的 token，
#     第 N 步自己采到的 sampled_token_ids 留在 GPU 上（另由 async_copy_stream 异步拷回），下一步才取。所以 CPU 准备第 N+1 步
#     输入时，还没有第 N 步的 token：调度器写进序列 / scheduled_tokens 的只是占位（eos id，见 scheduler.py 文件头【前置②】）
#   对比nano-vllm：nano-vllm 在 postprocess 里同步把 token 拷回 CPU、append 进序列，prepare_decode 再用 seq.last_token 建
#     input_ids 并 H2D——CPU 必须等 GPU 算完才能准备下一步。ATOM 让 CPU 准备下一步时根本不需要知道 token 的值：
#     input_ids 由这个 kernel 在 GPU 上从 prev_token_ids 拼出来，排在上一步采样 kernel 之后（同一条 stream）
#
#   带数字走一遍（MTP k=1 → 每请求 q=2 行；不开 DP；阶梯为 CLI 默认，止于 512）
#     第 N-1 步 decode [3, 7, 9]；第 N 步只做请求 5 的 prefill（调度器一步只做一种）；第 N+1 步 decode [3, 5, 7, 9]
#     第 N 步结束时：prev_batch = [5]，prev_token_ids = [t5]、draft_token_ids = [[d5]]（都在 GPU 上）；
#       3 / 7 / 9 在第 N-1 步采的 token（和 draft）已在第 N 步的 postprocess 里拷回 CPU、由调度器写进序列 → 第 N+1 步它们是"new"
#     get_token_locations：只有 5 在 prev_batch 里（prev 行 0）→ src = [-1, 0, -1, -1]；cu = cu_seqlens_q = [0, 2, 4, 6, 8]
#     ① CPU 先写 pinned（随 token_inputs 组一起 H2D，见 utils/h2d.py 文件头）：
#          input_ids[0:8] = [a3, e3 | P, P | a7, e7 | a9, e9]     （竖线分隔各请求的段）
#          a = 调度器给的 anchor（真值）；e = 调度器给的 draft（prepare_input_ids 对 new 请求逐个补写）；P = 请求 5 的占位
#     ② 本 kernel，grid = bs + tail = 4 + 1 个 program：
#          program 0、2、3（src < 0）直接返回，保留 CPU 写的值
#          program 1（src = 0）：col 0 写 prev_token_ids[0] = t5，col 1 写 draft_token_ids[0][0] = d5 → input_ids[2:4] = [t5, d5]
#          program 4（tail）：把 [cu[4] = 8, width) 清零
#     结果：input_ids = [a3, e3, t5, d5, a7, e7, a9, e9, 0, 0, …]
#     旧实现按 [new | deferred] 两个连续块放，会把 [t5, d5] 写到第 6~7 行（请求 9 的位置）——上面 docstring 最后一段说的 bug
#
#   width（要清零到哪）：调用方传 fill_to = max(本步 decode token 数, 图宽 × tokens_per_seq)。回放的图读固定的 running 行数，
#     多出来的行不清零就是上一次 forward 留下的 id（PIECEWISE 路径另由 ModelRunner._padded_decode_inputs 清零）
#   注意：调用方实际取的图宽是阶梯的最大档，而不是被回放的那一档（上例：512 × 2 → 清零 [8, 1024)，tail 仍只要 1 个 program）；
#     不影响正确性。这条"观察"和理由（含 DP 下的情形）写在 model_runner.py tokenIDProcessor.prepare_input_ids 里 fill_to 的上方

import torch
import triton
import triton.language as tl

from atom.utils.decorators import mark_trace

NEW_SEQUENCE = -1
"""`src[i]` for a request whose anchor is already staged from the host.

Any negative value means the same thing; the kernel tests `src < 0` because
Triton cannot read a module-level global from inside a `@jit`ed function, and a
`constexpr` copy would be a second place for the sentinel to be defined."""
# 【译】表示"该请求的 anchor 已经从 host 写好"的 `src[i]` 值。
#
#   任何负数含义都相同；kernel 判断的是 `src < 0`，因为 Triton 不能在 `@jit` 函数内部读模块级全局变量，
#   而复制一个 `constexpr` 又会让这个哨兵值有第二处定义。


BLOCK_PAD = 1024
"""Tokens one padding program zeroes. One value, so the tail costs one JIT
specialization rather than one per capture bucket -- those leak into serving."""
# 【译】一个补齐 program 清零的 token 数。只用这一个值，所以尾部只花一次 JIT 特化，而不是每个录制档位一次——
#   那些编译会漏到线上服务时才发生。
# 解释：constexpr 的每个不同取值都要单独编译一份 kernel；若块大小跟着图宽变，每个新图宽第一次出现时都会现场编译


# _fill_deferred_decode_ids_kernel：grid = (bs + tail,)，前 bs 个 program 各管一个请求，后 tail 个清零补齐尾部
#   参数（行尾英文的中文）：
#     out_ptr          [>= width] int32：已由调度器的 id 写好的 input_ids decode 区
#     cu_ptr           [bs+1] int32：各请求 token 数的前缀和（即 cu_seqlens_q 的前 bs+1 项）
#     src_ptr          [bs] int32：该请求在上一步 prev / draft 里的行号，或 NEW_SEQUENCE
#     prev_ptr         [num_prev]：上一步每行采样出的 id
#     draft_ptr        [num_prev, k]：上一步每行的 draft
#     draft_row_stride draft 表一行的步长（元素数）
#     num_requests     编号小于它的 program 填一个请求的段，其余清零尾部
#     width            forward 会读的 token 数
#     HAS_DRAFT        本步有没有 draft（constexpr：为 False 时 draft 那段代码在编译时就被删掉）
#     BLOCK_Q          ≥ 最长的单请求 token 数（tl.arange 的长度必须是 2 的幂，所以由宿主函数取 next_power_of_2）
#     BLOCK_PAD        一个清零 program 覆盖的 token 数（1024）
# Triton 要点：@triton.jit 默认会按整数参数的取值做特化（例如是否等于 1、能否被 16 整除），取值一变就可能重新编译；
#   do_not_specialize 让每步都在变的 num_requests / width 不参与特化，避免线上反复编译
@triton.jit(do_not_specialize=["num_requests", "width"])
def _fill_deferred_decode_ids_kernel(
    out_ptr,  # [>=width] int32 — staged with the scheduler's ids
    cu_ptr,  # [bs+1] int32 — exclusive prefix sum of per-request token counts
    src_ptr,  # [bs] int32 — row in prev/draft, or NEW_SEQUENCE
    prev_ptr,  # [num_prev] — last step's sampled id per row
    draft_ptr,  # [num_prev, k] — last step's drafts per row
    draft_row_stride,
    num_requests,  # programs below this fill a span, the rest zero the tail
    width,  # tokens the forward reads
    HAS_DRAFT: tl.constexpr,
    BLOCK_Q: tl.constexpr,  # >= the longest per-request token count
    BLOCK_PAD: tl.constexpr,
):
    """One program per request, then one per block of the graph's padded tail."""
    # 【译】每个请求一个 program，然后图的补齐尾部每个块一个 program。
    i = tl.program_id(0)
    if i >= num_requests:
        # 尾部 program：第 j 个（j = i - num_requests）覆盖 [j×1024, (j+1)×1024)，只写其中落在 [cu[bs], width) 的位置
        #   （tail 按 width 而不是 width - cu[bs] 算，所以靠前的几个可能整块被 mask 掉、什么都不写）
        col = (i - num_requests) * BLOCK_PAD + tl.arange(0, BLOCK_PAD)
        end = tl.load(cu_ptr + num_requests)  # `cu` says where requests end
        tl.store(out_ptr + col, 0, mask=(col >= end) & (col < width))
        return

    # 请求 program：src < 0 表示新请求，CPU 已写好真值，不动
    src = tl.load(src_ptr + i)
    if src < 0:  # NEW_SEQUENCE — the host already staged this request's span
        return

    # 本请求的段 [start, end)；live 标出 BLOCK_Q 个列里真正属于它的那些（长度 end - start，矩形 MTP 时恒为 q）
    start = tl.load(cu_ptr + i)
    end = tl.load(cu_ptr + i + 1)
    col = tl.arange(0, BLOCK_Q)
    live = col < (end - start)

    # col 0 = anchor = 上一步这一行采样出的 id（上一步若是投机解码，就是"最后一个被接受 / bonus 的 token"，
    #   见 model_runner.py postprocess 里对 prev_token_ids 的改写）
    val = tl.load(prev_ptr + src)
    if HAS_DRAFT:
        # Column `col - 1` of the draft row; anchors (col == 0) would index -1,
        # so mask them out here and pick the anchor back with `tl.where`.
        # 【译】取 draft 行的第 `col - 1` 列；anchor（col == 0）会索引到 -1，所以在这里把它 mask 掉，再用 `tl.where` 把 anchor 选回来。
        # 例：k=2、BLOCK_Q=4、段长 3 → col=[0,1,2,3]，live=[T,T,T,F] → val=[prev, draft[0], draft[1], (不写)]
        drafts = tl.load(
            draft_ptr + src * draft_row_stride + (col - 1),
            mask=live & (col > 0),
            other=0,
        )
        val = tl.where(col == 0, val, drafts)
    else:
        # 没有 draft：每个请求只有 1 行（宿主函数已断言），把标量扩成向量好和下面的 store 对齐
        val = tl.broadcast_to(val, [BLOCK_Q])

    tl.store(out_ptr + start + col, val, mask=live)


@mark_trace(prefix="decode_input_ids")
def fill_deferred_decode_ids(
    out: torch.Tensor,
    cu: torch.Tensor,
    src: torch.Tensor,
    prev_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    *,
    max_tokens_per_seq: int,
    width: int,
) -> None:
    """Overwrite the deferred requests' spans of `out`, and zero the tail.

    Args:
      out:             `[>= width]` int32 — the decode region of `input_ids`,
                       already staged from `batch.scheduled_tokens`.
      cu:              `[bs+1]` int32 — exclusive prefix sum of per-request
                       token counts. The ONLY statement of how long each
                       request's span is.
      src:             `[bs]` int32 — row into `prev_token_ids` /
                       `draft_token_ids`, or `NEW_SEQUENCE` to leave the span
                       as staged.
      prev_token_ids:  `[num_prev]` — previous forward's sampled ids.
      draft_token_ids: `[num_prev, k]`, or None when this step has no drafts
                       (then every span must be one token long).
      max_tokens_per_seq: an upper bound on `cu[i+1] - cu[i]`; sizes the
                       kernel's column vector.
      width:           tokens the forward will read; `[cu[bs], width)` is
                       zeroed. Required, because a caller that forgets it
                       leaves the graph's padded rows on the last step's ids.
    """
    # 【译】覆盖 `out` 中延续请求的那些段，并把尾部清零。
    #
    #   参数：
    #     out：             `[>= width]` int32——`input_ids` 的 decode 区，已经由 `batch.scheduled_tokens` 写好。
    #     cu：              `[bs+1]` int32——各请求 token 数的前缀和。这是"每个请求的段有多长"的**唯一**来源。
    #     src：             `[bs]` int32——指向 `prev_token_ids` / `draft_token_ids` 的行号，或 `NEW_SEQUENCE` 表示保留已写好的段。
    #     prev_token_ids：  `[num_prev]`——上一次 forward 采样出的 id。
    #     draft_token_ids： `[num_prev, k]`；本步没有 draft 时为 None（此时每个段必须只有 1 个 token）。
    #     max_tokens_per_seq：`cu[i+1] - cu[i]` 的上界；决定 kernel 列向量的长度。
    #     width：           forward 会读的 token 数；`[cu[bs], width)` 被清零。必须给，因为忘了它的调用方会让图的补齐行
    #                       留着上一步的 id。
    # 输入都已在 GPU 上：cu 与 src 是本步 token_inputs 组刚发布的（cu_seqlens_q、decode_src），prev / draft 是上一步留下的
    # @mark_trace（utils/decorators.py）：profiler 里给这次调用打上 decode_input_ids 标签，不改变行为
    bs = int(src.shape[0])
    # Python 写法：-(-a // b) 是向上取整的整数除法（ceil(a / b)），例 width=1024 → 1，width=1025 → 2
    tail = -(-int(width) // BLOCK_PAD)
    if bs + tail == 0:
        return
    assert cu.shape[0] == bs + 1, f"cu must be [bs+1]={bs + 1}, got {tuple(cu.shape)}"
    has_draft = draft_token_ids is not None
    assert has_draft or max_tokens_per_seq == 1, (
        "a step with no draft ids must feed exactly one token per request, "
        f"got max_tokens_per_seq={max_tokens_per_seq}"
    )
    # kernel[grid](...)：Triton 的 launch 语法，grid 是 program 数的元组。没有 draft 时 draft_ptr 传 prev_token_ids 只是占位
    #   （HAS_DRAFT=False 时 kernel 里读 draft 的代码已被编译期删掉）
    _fill_deferred_decode_ids_kernel[(bs + tail,)](
        out,
        cu,
        src,
        prev_token_ids,
        draft_token_ids if has_draft else prev_token_ids,
        draft_token_ids.stride(0) if has_draft else 0,
        bs,
        int(width),
        HAS_DRAFT=has_draft,
        BLOCK_Q=triton.next_power_of_2(max(int(max_tokens_per_seq), 1)),
        BLOCK_PAD=BLOCK_PAD,
    )


def fill_deferred_decode_ids_reference(
    out: torch.Tensor,
    cu: torch.Tensor,
    src: torch.Tensor,
    prev_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    *,
    max_tokens_per_seq: int,
    width: int,
) -> None:
    """Pure-torch twin of the kernel, runnable on CPU.

    Written as a plain loop over requests rather than a vectorised rework: the
    property under test is which request each written slot belongs to, and a
    loop states that directly. Tests assert PROPERTIES against this (non-uniform
    spans, interleaved kinds, coverage), not merely that the kernel agrees with
    it -- two implementations of the same wrong addressing agree perfectly.
    """
    # 【译】kernel 的纯 torch 孪生实现，可以在 CPU 上跑。
    #
    #   写成对请求的普通循环，而不是向量化重写：要测的性质是"每个被写的位置属于哪个请求"，循环把这一点直接写了出来。
    #   测试针对它断言的是性质（不均匀的段、交错的两种请求、覆盖完整），而不只是"kernel 与它一致"——同一种错误寻址的
    #   两个实现会完美地一致。
    # 行尾英文：只有 kernel 需要一个静态的列数上界（BLOCK_Q），这里用不到
    del max_tokens_per_seq  # only the kernel needs a static column bound
    cu_l = cu.tolist()
    out[cu_l[-1] : width] = 0
    for i, row in enumerate(src.tolist()):
        # 行尾英文：NEW_SEQUENCE，与 kernel 做的是同一个判断
        if row < 0:  # NEW_SEQUENCE, same test the kernel makes
            continue
        start, end = cu_l[i], cu_l[i + 1]
        if end <= start:
            continue
        out[start] = prev_token_ids[row]
        if end - start > 1:
            assert draft_token_ids is not None, (
                f"request {i} wants {end - start} tokens but this step has no "
                f"draft ids"
            )
            out[start + 1 : end] = draft_token_ids[row, : end - start - 1]
