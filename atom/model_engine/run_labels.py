"""Profiler ``record_function`` label taxonomy for forward passes.

Kept in its own dependency-free module so the label contract is unit-testable
without importing the heavy ``model_runner`` chain (aiter kernels, config, …).

Kinds (label prefix — groups in Perfetto UI, greppable):
  - ``prefill``             real prefill (eager)
  - ``decode``             real decode via CUDAGraph
  - ``eager_decode``       real decode forced eager
  - ``dummy_decode``       DP-sync dummy (CUDAGraph path)
  - ``dummy_eager_decode`` DP-sync dummy forced eager
  - ``dummy_prefill``      warmup dummy prefill

Fields (``key=value`` inside brackets):
  - ``bs``   the scheduled (real) batch size; on the CUDAGraph path shown as
             ``<real>/<graph>`` when the replayed graph is padded above the real
             batch (e.g. ``bs=117/128``). ``parse_trace.py`` reads the leading
             ``\\d+`` so the real batch is still what it extracts.
  - ``tok``  total scheduled tokens
  - ``ctx``  per-seq context lengths (prefill/eager paths; truncated if many)
  - ``p``/``d``  prefill / decode seq counts (cudagraph decode path)
  - ``spec`` speculative steps (when > 0)
  - ``tbo=1`` appended when the step ran Two-Batch-Overlap ubatches

NOTE: ``tools/parse_trace.py`` selects real steps via ``startswith("prefill[")``
/ ``startswith("decode[")``. The ``dummy_`` / ``eager_`` prefixes deliberately
fall outside those, so dummies never pollute the prefill/decode statistics.
"""

# 【译】forward pass的profiler record_function 标签分类法。
#
#   放在单独一个无依赖的模块里，这样标签约定可以在不导入沉重的 model_runner 依赖链（aiter kernel、config……）的情况下做单元测试。
#
#   种类（标签前缀——在Perfetto UI里分组，可grep）：
#     - prefill             真实的prefill（eager）
#     - decode              经CUDAGraph的真实decode
#     - eager_decode        被强制eager的真实decode
#     - dummy_decode        DP同步用的dummy（CUDAGraph路径）
#     - dummy_eager_decode  被强制eager的DP同步dummy
#     - dummy_prefill       预热用的dummy prefill
#
#   字段（方括号里的 key=value）：
#     - bs   调度的（真实）batch大小；在CUDAGraph路径上，当回放的graph被填充到大于真实batch时显示为 <真实>/<graph>
#            （例如 bs=117/128）。parse_trace.py 读取开头的 \d+，所以它提取的仍是真实batch。
#     - tok  调度的总token数
#     - ctx  每条序列的上下文长度（prefill/eager路径；序列多时截断）
#     - p/d  prefill / decode 序列数（cudagraph decode路径）
#     - spec 投机步数（大于0时）
#     - tbo=1 该step运行了Two-Batch-Overlap的ubatch时追加
#
#   注意：tools/parse_trace.py 通过 startswith("prefill[") / startswith("decode[") 挑出真实的step。dummy_ / eager_ 前缀
#   故意落在这些之外，这样dummy永远不会污染prefill/decode的统计。

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atom.model_engine.scheduler import ScheduledBatch


# 【导读】为每次forward生成profiler里的区段名（例如 "decode[bs=117/128 tok=117 d=117]"）
# 调用方：ModelRunner在run_model时用它给torch.profiler.record_function命名（model_runner.py，第5批）
# 与主干的关系：纯字符串函数，不影响计算；每次forward都会构建，但只在开了profiler（start_profile）抓trace时才有意义
# 读标签要用到的两个前置：
#   dummy_decode / dummy_eager_decode：DP>1时没活的rank每步陪跑的1个token假批（开MTP时k+1个；锁步，见engine_core.py
#     DPEngineCoreProc类上方）；dummy_prefill是启动预热。它们都单独打标签、不混进prefill / decode的统计
#   bs=117/128：真实117条，回放的是按128档捕获的CUDA Graph——Graph只按固定的batch档位捕获，不足的行补齐到最近的档位
#     （见config.py CUDAGraphMode上方的【科普】）


# 按上面的分类法拼出一次forward的标签字符串
def build_run_label(
    *,
    is_prefill: bool,
    use_cudagraph: bool,
    is_dummy: bool,
    tbo_on: bool,
    scheduled_bs: int,
    batch: ScheduledBatch | None,
    running_bs: int,
    detailed_suffix: str = "",
) -> str:
    """Build the ``record_function`` label for one forward pass.

    Pure function (no runtime state) — see module docstring for the taxonomy.

    ``detailed_suffix`` is an already-formatted ``" key=value ..."`` fragment
    (empty on the normal path) appended inside the brackets; see
    ``ModelRunner._detailed_label_suffix``.

    Both batch sizes are required: every step has both, and a step that ran
    padded is exactly the one worth seeing in a trace. The cudagraph path
    reports ``running_bs`` -- what actually ran -- and prefixes the real
    ``scheduled_bs`` as ``bs=<real>/<graph>`` when the two differ. The eager
    path reports ``scheduled_bs``, because eager attention runs the scheduled
    rows even where MoE pads past them.
    """
    # 【译】为一次forward pass构建 record_function 标签。
    #
    #   纯函数（没有运行时状态）——分类法见模块docstring。
    #
    #   detailed_suffix 是一个已经格式化好的 " key=value ..." 片段（正常路径上为空），追加在方括号内；
    #   见 ModelRunner._detailed_label_suffix。
    #
    #   两个batch大小都是必需的：每个step都同时有这两个值，而一个跑了填充的step恰恰是trace里最值得看到的。cudagraph路径报告
    #   running_bs——实际跑了多少——并在两者不同时把真实的 scheduled_bs 作为前缀写成 bs=<真实>/<graph>。eager路径报告
    #   scheduled_bs，因为eager注意力跑的就是调度的那些行，即使MoE会填充到超过它们。
    # 例：use_cudagraph、scheduled_bs=117、running_bs=128、batch有117条decode → "decode[bs=117/128 tok=117 d=117]"
    if use_cudagraph:
        kind = "dummy_decode" if is_dummy else "decode"
        if running_bs > scheduled_bs:
            label = f"{kind}[bs={scheduled_bs}/{running_bs}"
        else:
            # `decide` ceils to a captured size, so not-greater means equal.
            # 【译】decide 会向上取整到一个已捕获的大小，所以"不大于"就意味着相等。
            label = f"{kind}[bs={running_bs}"
        if batch is not None:
            label += f" tok={batch.total_tokens_num}"
            if batch.total_seqs_num_prefill > 0:
                label += f" p={batch.total_seqs_num_prefill}"
            label += f" d={batch.total_seqs_num_decode}"
            if batch.num_spec_step > 0:
                label += f" spec={batch.num_spec_step}"
    else:
        if is_prefill:
            kind = "dummy_prefill" if is_dummy else "prefill"
        else:
            kind = "dummy_eager_decode" if is_dummy else "eager_decode"
        label = f"{kind}[bs={scheduled_bs}"
        if batch is not None:
            ctx = batch.context_lens
            if len(ctx) == 1:
                ctx_str = str(ctx[0])
            elif len(ctx) <= 5:
                ctx_str = str(ctx.tolist())
            else:
                ctx_str = f"{ctx[:3].tolist()}...+{len(ctx) - 3}"
            label += f" tok={batch.total_tokens_num} ctx={ctx_str}"
            if is_prefill and not is_dummy:
                # Attention work of the chunk: sum over seqs of new tokens x the
                # mean context they attend (prior context + half the chunk).
                # 【译】这个chunk的注意力工作量：对所有序列求和 新token数 x 它们平均关注的上下文（之前的上下文 + chunk的一半）。
                # 例：一条序列ctx=4096（含本chunk）、本chunk新token n=1024 → 1024 x (4096 - 512) ≈ 3.7M，标签里显示 nctx=3.7M
                n = getattr(batch, "num_scheduled_tokens", None)
                if n is not None and len(n) == len(ctx):
                    n = n.astype("float64")
                    span = (n * (ctx - n / 2)).sum()
                    label += f" nctx={span / 1e6:.1f}M"
    label += detailed_suffix
    if tbo_on:
        label += " tbo=1"
    label += "]"
    return label
