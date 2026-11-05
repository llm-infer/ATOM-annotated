# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture one op's body into its OWN cudagraph, between the dense pieces.

Under ``--cudagraph-mode AF_PIECEWISE`` the attention core gets a cudagraph of
its own, so small-batch decode replays through it instead of running eager
between the graphed dense pieces. A cudagraph bakes the ADDRESS of everything it
reads, so both sides of that boundary need one that does not move:

  * inputs -- the runner clones each one at capture and refreshes the clone at
    replay, so the producer upstream can allocate however it likes;
  * the output -- the dense piece downstream was captured reading one address
    per (layer, rows), so an eagerly computed result is delivered into it.

Both are the runner's business, not the model's. A model opts in by decorating
the method that IS its core -- no separate wrapper, the decorated function is the
core itself::

    @piecewise_core(key=decode_bucket_key)
    def _attn_compress(self, *, x):
        ...the body that needs its own graph...

and writes nothing else -- no staging buffers, no per-input policy, no dummy
forward to measure shapes. The inputs and their order come off the function's
own signature (the leading ``self``/layer is skipped). A caller on the FULL /
eager path just passes ``piecewise=False`` and the body runs directly.

What this deliberately does NOT do is let the producer write the graph's input
buffer directly. That saves the clone's copy, but only by making every producer
in the chain aware of the capture -- which is what the previous design did, and
is what made this feature reach into the model layer. ``copy_per_step`` buys
that back for whichever inputs cannot be captured on -- name one and the runner
clones and refreshes it. V4 currently names none.
"""

# 【译】把一个算子的函数体录进它*自己的* cudagraph，夹在稠密段之间。
#
#   在 ``--cudagraph-mode AF_PIECEWISE`` 下，attention 核心有一张自己的 cudagraph，所以小 batch 的 decode 回放它，
#   而不是在已录成图的稠密段之间 eager 运行。cudagraph 会把它读的一切东西的*地址*固化下来，所以这条边界的两侧
#   都需要一个不会移动的地址：
#
#     * 输入——runner 在录制时克隆每个输入，回放时刷新这份克隆，这样上游的生产者想怎么分配都行；
#     * 输出——下游的稠密段录制时按每个 (layer, rows) 读一个地址，所以 eager 算出的结果要送进那里。
#
#   这两件事都归 runner 管，不归模型管。模型只要给"就是它的核心"的那个方法加上装饰器——没有单独的 wrapper，
#   被装饰的函数本身就是核心::
#
#       @piecewise_core(key=decode_bucket_key)
#       def _attn_compress(self, *, x):
#           ……需要自己一张图的函数体……
#
#   此外什么都不用写——没有中转 buffer、没有逐个输入的策略、也没有用来测量形状的假 forward。输入及其顺序直接取自
#   函数自己的签名（跳过开头的 ``self`` / layer）。走 FULL / eager 路径的调用方只要传 ``piecewise=False``，函数体就直接运行。
#
#   这里刻意*不*让生产者直接写进图的输入 buffer。那样能省掉克隆的那次拷贝，但代价是链条上每个生产者都得知道录制这件事——
#   之前的设计就是这样，也正是它让这个功能伸进了模型层。``copy_per_step`` 把这一点要回来，用于那些不能直接在其上录制的输入——
#   点名一个，runner 就克隆并刷新它。V4 目前一个也没点名。
# 观察：上面"输入——runner 克隆每个输入"一条与实现不符。piecewise_core 的默认是*零拷贝*：直接在生产者的张量上录制，
#   只有 copy_per_step 点名的输入才克隆 + 每步刷新（见 piecewise_core 的 docstring 与 _resolve_zero_copy），
#   docstring 最后一段说的也是这个
#
# 【导读】attn_ffn_piecewise.py：AF_PIECEWISE——在 PIECEWISE 稠密段之外，再给切分算子（"attention 核心"）单独录一张图
#   （第 5 批 5e，B 级）
#   前置：CUDA Graph 要求固定地址 / 固定形状、PIECEWISE 怎样在切分算子处切段、段与段之间怎么传张量——见 cuda_graph.py 文件头
#     （那里是概要，完整科普见 config.py 中 CUDAGraphMode 上方的【科普】②④）
#   为什么要它：plain PIECEWISE 下切分算子每层都 eager 运行，CPU 每层都要发射它的 kernel；decode 小 batch 时这部分 launch 开销
#     又回来了。AF_PIECEWISE 让切分算子也有自己的图，一次 decode 全部变成回放：
#
#       PIECEWISE：     [段0 回放] → 切分算子 eager → [段1 回放] → 切分算子 eager → …
#       AF_PIECEWISE：  [段0 回放] → [核心图 回放]   → [段1 回放] → [核心图 回放]   → …
#
#   v0.1.8 里唯一的使用者是 DeepSeek-V4（deepseek_v4.py，第 8 批）：
#     切分算子 torch.ops.aiter.v4_attn_compress → layer._attn_compress（被 @piecewise_core 装饰）
#     这个"核心"是 V4 注意力里唯一按 batch 形状发射的部分：KV 压缩器（grid = graph_bs × 每序列上限），gfx950 上还有 FP4
#     decode 打分；它不返回值，作用全在分页状态上（"void core"，调用时 outputs=None），按 token 形状的工作都留在两侧的稠密段里
#     runner 是 deepseek_v4.py 的模块级全局 v4_attn_runner = CudagraphCaptureRunner()（cuda_graph.py）
#
#   键：稠密段只按 num_tokens 区分；核心的 grid 取决于序列数，所以键是 (layer_name, num_tokens, bucket_bs, q_eff)
#     例（DSpark 每序列 q=6 个 token，默认 capture_sizes = [1, …, 8, 16, 32, …]）：bs=16 的矩形 batch 是 num_tokens = 96；
#       ModelRunner._capture_attn_ffn_graphs 再为 bs=16 录"参差"桶：num_tokens_pad = b × 6 落在 [16, 96) 的 b = 3…8，
#       即 18、24、30、36、42、48（_piecewise_skip_capture 可能再跳过一些）
#       → 每层在 bs=16 下最多 7 张核心图 (layer, 96, 16, 6)、(layer, 48, 16, 6)、…、(layer, 18, 16, 6)；
#       这些 num_tokens 的稠密段按 num_tokens 去重：已录过就回放，没录过就此录下
#
#   一层在各种 step 下走哪条路（wrapper 的分派，见 piecewise_core 里的 wrapper）：
#     step                                            piecewise  capture  in_hipgraph  结果
#     FULL / eager / 录制轮的预热 forward（runtime NONE）  False      –        –        直接调函数体
#     plain PIECEWISE                                  True       False    –        函数体 eager；有返回值则拷进 StableOutputs 的 slot
#     AF 录制轮，第一次遇到这个键                         True       True     True     预热一次 → runner.capture 录图 → 再 eager 跑一次
#     AF 真实 decode，键已录过                           True       True     False    runner.replay（零拷贝输入不用刷新）
#     AF 但 prefill / dummy run / 键没录过               True       True     任意      函数体 eager（同 plain PIECEWISE）
#   piecewise 由 v4_attn_compress 按 forward_context.cudagraph_runtime_mode == PIECEWISE 判断；
#   capture 是 layer.attn_ffn_piecewise（= cudagraph_mode.is_attn_ffn_piecewise()，只有 AF_PIECEWISE 为真）；
#   in_hipgraph：原生路径下只在 ModelRunner.capture_cudagraph / _capture_attn_ffn_graphs 设的 forward context 里为 True
#     （vLLM 插件的 deepseek_v4_bridge.py 在录制时也会置 True，第 10 批）
#
#   代价：每层、每个桶多一张图（录制时间 + 显存）；可选的 max_tokens 上限可以压显存
#     （piecewise_core 里的实测：按粒度拆分前，这组图的池是 8.37GB，拆分后 1.62GB）
#   对比nano-vllm：nano-vllm 只录 decode 整图，attention 本来就在图里，不存在"段之间的算子要不要单独录"的问题
#
# 本文件的函数与调用方（grep 核实）
#   decode_bucket_key          V4 的 @piecewise_core(key=decode_bucket_key)
#   _is_decode / _annotation_names_tensor / _partition_inputs / _resolve_zero_copy   piecewise_core 内部
#   piecewise_core             deepseek_v4.py 的 _attn_compress（以及 tests/test_dspark.py）

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any, get_args

__all__ = [
    "decode_bucket_key",
    "piecewise_core",
]


# 核心图键的后半段：(bucket_bs, q_eff)，与 (layer_name, num_tokens) 拼成完整的键
#   例：DSpark bs=16、q=6 的 decode → (16, 6)
def decode_bucket_key(forward_context) -> tuple:
    """``(bucket_bs, q_eff)`` for a core captured per decode bucket.

    Nothing here is model-specific -- both come off the forward context -- so
    every decode core keys the same way. ``bucket_bs`` is the step's
    ``running_bs`` (a capture Context has no forward_mode, so batch_size is
    already the bucket).
    """
    # 【译】为按 decode 桶录制的核心给出 ``(bucket_bs, q_eff)``。
    #
    #   这里没有任何模型特有的东西——两个值都来自 forward context——所以每个 decode 核心都用同样的方式取键。
    #   ``bucket_bs`` 是这一步的 ``running_bs``（录制用的 Context 没有 forward_mode，所以 batch_size 本身就是桶）。
    # q_eff = attn_metadata.max_seqlen_q（每序列最多几个 query token，普通 decode 为 1，MTP k=3 为 4）
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    context = getattr(forward_context, "context", None)
    q_eff = int(getattr(attn_metadata, "max_seqlen_q", 1) or 1)
    forward_mode = getattr(context, "forward_mode", None)
    # 真实 step：ForwardMode.running_bs 是补齐后的桶大小；录制轮：没有 forward_mode，退回 context.scheduled_bs（就是桶大小）
    if forward_mode is not None and getattr(forward_mode, "running_bs", 0):
        bucket_bs = int(forward_mode.running_bs)
    else:
        bucket_bs = int(getattr(context, "scheduled_bs", 0) or 0)
    return (bucket_bs, q_eff)


def _is_decode(forward_context) -> bool:
    """Whether this step is a decode.

    Checked by NAME, like ``_annotation_names_tensor`` above and for the same
    reason: this module stays free of a torch import, and ``AttnState`` lives
    behind one. Any ``PREFILL_*`` reads as not-decode, which is the only
    distinction that matters here.
    """
    # 【译】这一步是不是 decode。
    #
    #   按*名字*检查，与上面的 ``_annotation_names_tensor`` 一样，理由也相同：这个模块不 import torch，而 ``AttnState``
    #   在一个 import torch 的模块里。任何 ``PREFILL_*`` 都读作"不是 decode"，这里唯一要紧的就是这个区分。
    # 解释：_annotation_names_tensor 其实定义在下面；只有 decode 能录图——prefill 的形状每次都不同
    state = getattr(getattr(forward_context, "attn_metadata", None), "state", None)
    return getattr(state, "name", "") == "DECODE"


def _annotation_names_tensor(annotation: Any) -> bool:
    """Whether a parameter's annotation names a tensor.

    This is the whole rule that tells a graph INPUT from a pass-through config
    arg: ``torch.Tensor`` (and ``torch.Tensor | None``) is an input; an
    explicitly non-tensor annotation (``bool``, ``int``, an enum, ...) is config.
    Checked by NAME so this module stays free of a torch import, and handles the
    ``from __future__ import annotations`` case where the annotation is a string.
    An UNANNOTATED parameter counts as an input, so a core written before this
    split -- everything a tensor -- is unchanged.
    """
    # 【译】一个参数的类型注解是否指向张量。
    #
    #   区分图*输入*和透传的配置参数，规则就只有这一条：``torch.Tensor``（以及 ``torch.Tensor | None``）是输入；
    #   明确不是张量的注解（``bool``、``int``、枚举……）是配置。按*名字*检查，这样这个模块不必 import torch，
    #   也能处理 ``from __future__ import annotations`` 下注解是字符串的情况。
    #   *没有*注解的参数算作输入，所以在这次拆分之前写的、参数全是张量的核心不受影响。
    # 例：x: torch.Tensor | None → True；compressor_already_launched: bool → False；字符串 "torch.Tensor" → True
    if annotation is inspect.Parameter.empty:
        return True
    if isinstance(annotation, str):
        return "Tensor" in annotation
    # Bare ``torch.Tensor`` has no args; a Union/Optional yields its members.
    # 【译】裸的 ``torch.Tensor`` 没有 args；Union / Optional 会给出它的各个成员。
    parts = get_args(annotation) or (annotation,)
    return any(getattr(p, "__name__", "") == "Tensor" for p in parts)


def _partition_inputs(fn: Callable) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The core's parameters after the leading layer, split into the graph's
    tensor INPUTS (in order) and the pass-through CONFIG args.

    One source rather than two: the split comes off the signature itself, so
    there is no names list beside the function to drift from it. A parameter is a
    graph input unless its annotation names no tensor (see
    ``_annotation_names_tensor``); a config arg is forwarded to the body
    untouched and BAKED into the graph at capture -- never cloned, never captured
    on an address -- so it must be invariant for a given graph key (or fold it
    into ``key``). Order matters (the runner expands inputs by it), so
    ``*args``/``**kwargs`` are rejected here rather than mis-expanded later. The
    leading parameter is the layer that owns the core, not one of its inputs.
    """
    # 【译】核心在开头的 layer 之后的参数，拆成图的张量*输入*（按顺序）和透传的*配置*参数。
    #
    #   只有一个来源而不是两个：拆分直接取自签名本身，所以函数旁边没有一份会与它走样的名字列表。一个参数是图输入，除非它的
    #   注解指向的不是张量（见 ``_annotation_names_tensor``）；配置参数原样转给函数体，在录制时被*固化*进图——从不克隆、
    #   也从不在某个地址上录制——所以对同一个图键它必须不变（或者把它并进 ``key``）。顺序很重要（runner 按它展开输入），
    #   所以 ``*args`` / ``**kwargs`` 在这里就被拒绝，而不是之后被错误展开。开头那个参数是拥有这个核心的 layer，不是它的输入。
    # 例：V4 的 _attn_compress(self, *, x, idx_q_quant, idx_weights, idx_q_scale, compressor_already_launched: bool)
    #   → 输入 (x, idx_q_quant, idx_weights, idx_q_scale)，透传 (compressor_already_launched,)
    #   在装饰时（模块 import 时）只算一次
    inputs: list[str] = []
    passthrough: list[str] = []
    for i, (name, p) in enumerate(inspect.signature(fn).parameters.items()):
        if i == 0:
            continue
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            raise TypeError(
                f"{fn.__name__} must name its inputs explicitly; "
                f"'{'**' if p.kind is p.VAR_KEYWORD else '*'}{name}' has no "
                "declared order for the runner to expand by."
            )
        (inputs if _annotation_names_tensor(p.annotation) else passthrough).append(name)
    return tuple(inputs), tuple(passthrough)


def _resolve_zero_copy(names: tuple[str, ...], copied: frozenset) -> frozenset:
    """Which inputs the graph captures on directly.

    ``ATOM_ATTN_FFN_ZC`` overrides it with a comma-separated whitelist (empty
    string = capture on nothing), which is how a suspected input gets taken off
    the zero-copy path for one run without touching code.
    """
    # 【译】图直接在哪些输入上录制（零拷贝）。
    #
    #   ``ATOM_ATTN_FFN_ZC`` 用一个逗号分隔的白名单覆盖它（空字符串 = 一个都不零拷贝），这样不改代码就能在一次运行里
    #   把一个可疑的输入从零拷贝路径上拿下来。
    # 默认：全部张量输入减去 copy_per_step 点名的；V4 的 copy_per_step=() → 四个输入全部零拷贝
    # 注意：ATOM_ATTN_FFN_ZC 直接读 os.environ（不在 envs.py 里），而且在装饰时（模块 import 时）读，所以必须在启动前设置
    override = os.environ.get("ATOM_ATTN_FFN_ZC")
    if override is not None:
        wanted = {p.strip() for p in override.split(",") if p.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(
                f"ATOM_ATTN_FFN_ZC names {sorted(unknown)} are not inputs; "
                f"known names are {list(names)}"
            )
        return frozenset(wanted)
    return frozenset(names) - copied


def piecewise_core(
    *,
    key: Callable[[Any], tuple] = lambda _fc: (),
    max_tokens: int | None = None,
    copy_per_step: tuple[str, ...] = (),
):
    """Give the decorated function its own cudagraph, keyed per layer and shape.

    ``key`` adds whatever else changes the graph's shape beyond the layer and the
    token count -- for a decode core that is the bucket, `decode_bucket_key`.
    Only DECODE steps are captured; a prefill runs eager, delivering into the
    persistent output slot. ``max_tokens`` is an OPTIONAL extra cap on the rows
    a captured graph covers -- a memory lever, off by default. It used to be 512
    and to be the only gate, which silently excluded every decode bucket above
    ~85 sequences at DSpark q=6.

    Every TENSOR input is captured on directly -- the graph reads the producer's
    own tensor and nothing copies it -- EXCEPT the names in ``copy_per_step``,
    which the runner clones and refreshes each step. Name one when capturing on it
    is wrong, whatever the reason; the entry's own comment should say which.

    A parameter annotated as something other than a tensor (``bool``, ``int``, an
    enum, ...) is NOT a graph input: it is forwarded to the body untouched and
    baked into the graph at capture, so a core can keep its config flags in its
    own signature without the model having to strip them. Being baked, such an arg
    must be invariant for a given graph ``key`` (or folded into it).

    The wrapped function is called as ``fn(layer, **inputs)`` and returns one
    tensor. It routes three ways off two flags the caller passes each step:

      * ``piecewise=False`` -- FULL / eager / a PIECEWISE forward running under a
        FULL-runtime graph. Simply called: nothing is captured, and no downstream
        piece is holding an address to deliver to.
      * ``piecewise=True, capture=False`` -- plain PIECEWISE. The core stays
        eager, but the dense piece downstream was captured reading a fixed
        address, so the result is delivered into the persistent output slot.
      * ``piecewise=True, capture=True`` -- the core gets its own cudagraph
        (AF_PIECEWISE): captured in the capture pass, replayed after.

    So ``capture`` is the mode gate -- with it off this collapses to exactly the
    eager-attention-plus-stable-buffer path plain PIECEWISE always ran, and the
    caller has no branch of its own to keep.
    """

    # 【译】给被装饰的函数一张自己的 cudagraph，按 layer 和形状分键。
    #
    #   ``key`` 补上除 layer 和 token 数之外、还会改变图形状的东西——对 decode 核心来说就是桶，`decode_bucket_key`。
    #   只录 DECODE 步；prefill 以 eager 方式运行，结果送进持久的输出 slot。``max_tokens`` 是一个*可选的*额外上限，
    #   限制录下的图覆盖多少行——一个显存杠杆，默认关闭。它以前是 512，而且是唯一的门槛，结果悄悄排除了
    #   DSpark q=6 时所有超过约 85 条序列的 decode 桶。
    #
    #   每个*张量*输入都直接在其上录制——图读生产者自己的张量，没有任何拷贝——*除了* ``copy_per_step`` 点名的那些，
    #   runner 会克隆它们并每步刷新。只要在其上直接录制是错的，不管什么原因，就点名它；那一项自己的注释应说明是哪种原因。
    #
    #   注解为非张量（``bool``、``int``、枚举……）的参数*不是*图输入：它原样转给函数体，在录制时固化进图，所以核心可以把
    #   配置开关留在自己的签名里，而模型不必把它们剥掉。既然被固化，这样的参数对同一个图 ``key`` 必须不变（或并进 key）。
    #
    #   被包装的函数以 ``fn(layer, **inputs)`` 调用，返回一个张量。它根据调用方每步传的两个标志分三路：
    #
    #     * ``piecewise=False``——FULL / eager / 在 FULL 运行时图下跑的 PIECEWISE forward。直接调用：什么都不录，
    #       下游也没有哪个段握着一个要送达的地址。
    #     * ``piecewise=True, capture=False``——plain PIECEWISE。核心保持 eager，但下游稠密段录制时读的是一个固定地址，
    #       所以结果要送进持久的输出 slot。
    #     * ``piecewise=True, capture=True``——核心有自己的 cudagraph（AF_PIECEWISE）：在录制轮录下，之后回放。
    #
    #   所以 ``capture`` 就是模式开关——关掉它，这里就退化成 plain PIECEWISE 一直在跑的"eager attention + 稳定 buffer"路径，
    #   调用方不必自己保留一个分支。
    # 解释：V4 的核心不返回值（返回 None），所以"送进 slot"那一步对它是空操作
    # 第一步：在装饰时检查参数、算好输入划分
    if isinstance(copy_per_step, str):
        # `("positions")` is a string, not a 1-tuple, and `frozenset` of it is a
        # set of CHARACTERS -- which subtracts nothing from the input names, so
        # the entry silently means "copy nothing" while reading as its opposite.
        # That shipped once. A one-name tuple needs its trailing comma.
        # 【译】`("positions")` 是一个字符串，不是 1 元组，而它的 `frozenset` 是一组*字符*——从输入名里什么也减不掉，
        #   于是这一项悄悄变成"什么都不拷"，读起来却是相反的意思。这个错误发布过一次。只有一个名字的元组要带结尾的逗号。
        raise TypeError(
            f"copy_per_step must be a tuple of names, got the string "
            f"{copy_per_step!r}. A single name needs a trailing comma: "
            f'("{copy_per_step}",). Without it this silently copies nothing.'
        )
    copied = frozenset(copy_per_step)

    def decorate(fn: Callable) -> Callable:
        # 装饰时（import 时）算一次：张量输入名 / 透传名 / 零拷贝名
        input_names, passthrough_names = _partition_inputs(fn)
        zero_copy = _resolve_zero_copy(input_names, copied)

        # 运行时每次调用都进这里。除了函数自己的参数，调用方还传：
        #   piecewise / capture：分派标志（见上面 docstring）；runner：CudagraphCaptureRunner；
        #   outputs：StableOutputs 或 None（void core）；forward_context：当前 forward context
        @functools.wraps(fn)
        def wrapper(
            layer,
            *,
            piecewise,
            capture=False,
            runner=None,
            outputs=None,
            forward_context=None,
            **inputs,
        ):
            # FULL / eager: the body runs directly, so a caller on that path only
            # states `piecewise=False` and none of the capture collaborators.
            # 【译】FULL / eager：函数体直接运行，所以这条路径上的调用方只需写 `piecewise=False`，不用传任何录制相关的参数。
            if not piecewise:
                return fn(layer, **inputs)

            layer_name = getattr(layer, "layer_name", id(layer))
            # Row count off the first input the caller actually passed. A core
            # whose signature covers several call shapes leaves the inputs of the
            # shapes it is not in as None -- declaration order still decides
            # which one anchors, but a None one is skipped rather than crashing.
            # Every tensor input is [num_tokens, ...], so any of them will do.
            # 【译】行数取自调用方实际传入的第一个输入。签名覆盖多种调用形态的核心，会把不属于当前形态的输入留成 None——
            #   声明顺序仍决定哪个当锚点，但为 None 的会被跳过，而不是崩溃。每个张量输入都是 [num_tokens, ...]，任何一个都行。
            anchor = next(
                (inputs[n] for n in input_names if inputs.get(n) is not None), None
            )
            if anchor is None:
                raise ValueError(
                    f"{fn.__name__} was called with every tensor input None; "
                    "the runner has no way to size the graph. Inputs are "
                    f"{list(input_names)}."
                )
            num_tokens = int(anchor.shape[0])
            # One output slot per (layer, flat row count): the address the dense
            # piece downstream was captured reading.
            # 【译】每个 (layer, 扁平行数) 一个输出 slot：下游稠密段录制时读的那个地址。
            out_key = (layer_name, num_tokens)
            # Config args are not graph inputs: bind them into the compute so they
            # are baked at capture, and hand the runner only the tensor inputs.
            # EVERY declared input, None included. A core whose signature spans
            # several call shapes leaves the inputs of the shapes it is not in as
            # None, and those parameters still have to be bound or the call is a
            # missing-argument TypeError. The runner already leaves a None alone
            # (`input_buffers` / `replay` both check), so it takes the same dict.
            # 【译】配置参数不是图输入：把它们绑进计算，让它们在录制时被固化，只把张量输入交给 runner。
            #   *每个*声明过的输入都要给，None 也算。签名覆盖多种调用形态的核心，会把不属于当前形态的输入留成 None，
            #   这些参数仍必须绑定，否则调用时会因缺参数而 TypeError。runner 本来就会跳过 None（`input_buffers` / `replay` 都检查），
            #   所以它接收同一个 dict。
            # functools.partial(fn, layer, **透传参数)：先把 layer 和配置参数固定下来，得到只差张量输入的 core(**tensor_inputs)
            tensor_inputs = {n: inputs.get(n) for n in input_names}
            core = functools.partial(
                fn, layer, **{n: inputs[n] for n in passthrough_names if n in inputs}
            )

            context = getattr(forward_context, "context", None)
            # `capture` off (plain PIECEWISE) short-circuits to the deliver tail
            # below: the core runs eager and only its result is stabilised, with
            # no graph of its own ever recorded.
            # DECODE, not a row count. Prefill is what must not be captured --
            # its shapes are one-off, and the compressor's prefill plan is
            # sliced to an actual count rather than a graph-fixed capacity -- and
            # asking that question directly beats bounding tokens and hoping the
            # two coincide. They did not: at DSpark q=6 a 512-row bound also cut
            # every decode above bs~85, so AF was silently OFF for the three
            # largest buckets. `max_tokens` survives as an optional cap for the
            # reason the bound was introduced (capture memory), now off by
            # default -- the pool measured 8.37GB before the granularity split
            # and 1.62GB after.
            # 【译】`capture` 关闭（plain PIECEWISE）时直接短路到下面的 deliver 尾部：核心 eager 运行，只把结果稳定下来，
            #   从不录它自己的图。
            #   判断的是 DECODE，而不是行数。不能录的是 prefill——它的形状只出现一次，而压缩器的 prefill 计划按实际数量切片，
            #   而不是按图固定的容量——直接问这个问题，胜过限制 token 数再指望两者恰好一致。它们并不一致：DSpark q=6 时，
            #   512 行的上限也砍掉了 bs 约 85 以上的每个 decode，于是最大的三个桶上 AF 悄悄是*关*的。`max_tokens` 作为可选上限
            #   保留下来，理由正是当初引入那个上限的理由（录制显存），现在默认关闭——按粒度拆分前测得池为 8.37GB，拆分后 1.62GB。
            # 能录图的条件：开了 capture、不是 dummy run（预热 / DP 空跑）、有注意力元数据、是 decode、没超过可选的行数上限
            eligible = (
                capture
                and not getattr(context, "is_dummy_run", False)
                and getattr(forward_context, "attn_metadata", None) is not None
                and _is_decode(forward_context)
                and (max_tokens is None or num_tokens <= max_tokens)
            )
            # num_tokens is a KEY DIM, not an incidental one: the graph is
            # captured at exactly this flat row count.
            # 【译】num_tokens 是一个*键的维度*，不是附带的维度：图正是按这个扁平行数录制的。
            # 例：("model.layers.3.attn", 96) + (16, 6) → ("model.layers.3.attn", 96, 16, 6)
            graph_key = (
                (layer_name, num_tokens) + tuple(key(forward_context))
                if eligible
                else None
            )
            # True only inside the capture pass: WHICH KIND of forward this is,
            # building the graphs vs serving a real step.
            # 【译】只在录制轮里为 True：这是*哪一种* forward——在建图，还是在服务真实的一步。
            capturing = getattr(forward_context, "in_hipgraph", False)

            if graph_key is not None:
                if capturing and not runner.has_graph(graph_key):
                    # First time the capture pass reaches this key. Warm up on
                    # the buffers the graph will read, size the output slot from
                    # that result, then record.
                    # 【译】录制轮第一次走到这个键。先在图将要读的 buffer 上预热，用那次结果确定输出 slot 的大小，然后录制。
                    read_from, refresh = runner.input_buffers(
                        tensor_inputs, input_names, zero_copy
                    )
                    # A void core -- one whose whole effect is on paged state --
                    # has no result to stabilise and no slot to size.
                    # 【译】不返回值的核心——它的全部作用都在分页状态上——没有要稳定的结果，也没有要定大小的 slot。
                    warm = core(**read_from)
                    out_slot = None if warm is None else outputs.slot(out_key, warm)
                    runner.capture(graph_key, read_from, refresh, core, out_slot)
                # 真实 decode 步、这个键已录过 → 回放（录制轮重复遇到同一键时不回放，落到下面 eager 跑）
                elif runner.has_graph(graph_key) and not capturing:
                    return runner.replay(graph_key, tensor_inputs)

            # Everything else: not eligible, the capture pass revisiting a key,
            # or a step whose key was never captured. Also the tail of the
            # capture branch above -- it fed on clones, so its result is not this
            # forward's answer.
            # 【译】其余所有情况：不满足录图条件、录制轮再次遇到某个键、或者某一步的键从没录过。也是上面录制分支的尾部——
            #   它喂的是克隆，所以它的结果不是这次 forward 的答案。
            # 有返回值时拷进 (layer, num_tokens) 的固定 slot，下游段才能在录制时的地址读到它
            result = core(**tensor_inputs)
            return None if result is None else outputs.deliver(out_key, result)

        # 挂在 wrapper 上，供测试和调试查看划分结果
        wrapper.input_names = input_names
        wrapper.passthrough_names = passthrough_names
        wrapper.zero_copy_names = zero_copy
        wrapper.max_tokens = max_tokens
        return wrapper

    return decorate
