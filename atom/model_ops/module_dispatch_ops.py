# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Custom ops dispatching to module methods via `static_forward_context`.

Each op below takes a `layer_name: str` and looks up the owning module from
`compilation_config.static_forward_context[layer_name]`. The module's
internal state (sub-modules, persistent buffers, per-fwd metadata fetched
through `get_forward_context()`) stays out of the op's argument list, which:

  - hides dynamic-shape internals from Dynamo / fake-tensor mode (no graph
    break, no per-shape recompile)
  - bypasses functionalization for buffer mutations — the buffers aren't in
    the op's signature, so the pass needn't insert defensive clones
  - leaves CUDAGraph capture transparent — the op still launches the same
    CUDA work on the stream; only the Python dispatch is opaque to compile

Caller contract (per op): the module registered at `layer_name` must expose
the methods listed in each op's docstring.

Currently registered:
  - torch.ops.aiter.maybe_dual_stream_forward  — V2/V3.2/V4 MoE (summed)
  - torch.ops.aiter.maybe_dual_stream_split_forward — K3 MoE (routed/shared unsummed)
  - torch.ops.aiter.indexer_score_topk         — V4 sparse indexer
"""

# 【译】通过 `static_forward_context` 分发到 module 方法的自定义算子。
#
#   下面每个算子都接收一个 `layer_name: str`，并从 `compilation_config.static_forward_context[layer_name]` 查回所属的 module。
#   module 的内部状态（子模块、常驻 buffer、经 `get_forward_context()` 取得的每次 forward 的元数据）不出现在算子的参数列表里，这样：
#
#     - 对 Dynamo / fake 张量模式隐藏了动态形状的内部细节（不会 graph break，也不会按形状重编译）
#     - buffer 的原地修改绕过了 functionalization——buffer 不在算子签名里，这个 pass 就不必插入防御性拷贝
#     - 对 CUDAGraph 捕获是透明的——算子仍在 stream 上发起同样的 CUDA 工作；只是 Python 层的分发对编译器不可见
#
#   调用方约定（每个算子各自的）：在 `layer_name` 下登记的 module 必须提供每个算子 docstring 里列出的方法。
#
#   目前注册的有：
#     - torch.ops.aiter.maybe_dual_stream_forward  —— V2/V3.2/V4 的 MoE（返回相加后的结果）
#     - torch.ops.aiter.maybe_dual_stream_split_forward —— K3 的 MoE（routed / shared 分开返回、不相加）
#     - torch.ops.aiter.indexer_score_topk         —— V4 的稀疏索引器
#
# 【导读】module_dispatch_ops.py：几个"只带 layer_name 字符串"的黑盒自定义算子（B 级）
#   通用原理（为什么用 layer_name 查回 module、为什么必须是黑盒）见 utils/decorators.py 文件头【科普】②⑥：
#     ATOM 的 torch.compile 只抓一次图、之后不检查 guard，被追踪代码里的 Python 分支会被冻结成抓图那一刻的取值；
#     这里的几个算子里都有"每次调用要现场判断"的逻辑（按 token 数、TBO 是否在跑、当前是否在 PIECEWISE 录图），
#     放进黑盒算子里，它们就在每次真正执行时由 Python 现场决定。
#   和 attention 切分点的区别：这些算子没有标 spliting_op（不是用 utils/__init__.py 的 mark_spliting_op 注册的），
#     不会切图，留在 backends.py 切出的稠密子图里当一个节点；录 CUDA Graph 时它们随所在子图一起被录进去，
#     回放时 Python 分发逻辑不再执行（录图那一刻的选择被固定在图里）
#   注册：custom_register.py 的 direct_register_custom_op；本模块在导入时就完成注册，所以 deepseek_v4.py、kimi_k3/model.py、
#     minimax_m3/model.py 用 `from atom.model_ops import module_dispatch_ops  # noqa: F401` 的方式只为触发注册而导入它
#   观察：上面 docstring 的列表漏了本文件最后注册的 torch.ops.aiter.tbo_all_reduce（它不带 layer_name，是 TBO 的 all_reduce 包装）；
#     docstring 说"方法列在每个算子的 docstring 里"，实际写在各算子上方的注释里
#   对比nano-vllm：nano-vllm 只支持 Qwen3 dense 模型、不编译整个模型，没有 MoE，也不需要这类黑盒算子

import torch

from atom.config import CUDAGraphMode, get_current_atom_config
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import get_current_cudagraph_runtime_mode

# ---------------------------------------------------------------------------
# Dual-stream MoE dispatch (V2 / V3.2 / V4 / K3)
# ---------------------------------------------------------------------------
#
# Caller contract (the MoE module looked up by `layer_name`):
#   - `_use_dual_stream: bool`
#   - `single_stream_moe_forward(hidden_states) -> Tensor`
#   - `dual_stream_moe_forward(hidden_states) -> Tensor`
#
# Per-token gating: decode benefits from dual-stream, prefill doesn't —
# threshold from `envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`.
# 【译】双 stream MoE 分发（V2 / V3.2 / V4 / K3）
#
#   调用方约定（按 `layer_name` 查回的 MoE module 必须有）：
#     - `_use_dual_stream: bool`（是否启用双 stream）
#     - `single_stream_moe_forward(hidden_states) -> Tensor`（单 stream 实现）
#     - `dual_stream_moe_forward(hidden_states) -> Tensor`（双 stream 实现）
#
#   按 token 数门控：decode 能从双 stream 中受益，prefill 不能——阈值来自 `envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`。
#
# 【科普】双 stream MoE：DeepSeek 式 MoE 层 = 共享专家（每个 token 都过的普通 MLP）+ 路由专家（gate 选 top-k 个专家）
#   （MoE 入门见 model_engine/arg_utils.py 中 MoE 组字段上方的【科普】MoE与EP）。两条分支读同一份 hidden_states，最后才相加，彼此独立。
#   stream 是 GPU 上的一条命令队列：同一条 stream 里的 kernel 按顺序执行，不同 stream 的 kernel 可以同时跑
#   （arg_utils.py enable_tbo 上方的【科普】也用到它）。于是可以把共享专家放到另一条 stream（alt_stream）上：
#
#     单 stream：  主 stream：[共享专家 MLP][gate → top-k → fused MoE（路由专家）][相加]
#     双 stream：  主 stream：[gate → top-k → fused MoE（路由专家）]──────┐ 等待 alt → [相加]
#                 alt     ：[共享专家 MLP]───────────────────────────────┘
#   为什么只对少 token 有用：decode 时一步只有几个到几百个 token，单个 kernel 填不满 GPU 的计算单元，两条分支并行能互相填空；
#     prefill 时每个 kernel 已经占满 GPU，并行没有收益，还多了 stream 同步的开销。所以按 token 数门控：
#     0 < token 数 <= ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD（默认 1024；设成 0 时模型构造时就不启用双 stream）
#   为什么要放进黑盒算子：门控条件依赖每次调用的 token 数，写在被追踪的 forward 里会被冻结成 warmup 那次（几千 token 的 prefill）
#     的结果——永远单 stream（decorators.py【科普】②）
#   实现在各模型里：deepseek_v2.py DeepseekV2MoE 的 dual_stream_moe_forward / single_stream_moe_forward（V3.2、GLM-5 复用），
#     deepseek_v4.py、kimi_k3/model.py 各有自己的版本；_use_dual_stream 只在共享专家没有被融合进 fused MoE、且未开 TBO 等条件下为 True


def _dual_stream_is_active(self, num_tokens: int) -> bool:
    """Whether this call should fork the shared branch onto ``alt_stream``.

    Shared by the single-tensor op and the split-return op below so the two
    can never drift into disagreeing about when dual-stream engages.
    """
    # 【译】这次调用是否应当把共享分支分叉到 ``alt_stream`` 上。
    #
    #   由下面返回单个张量的算子和分开返回的算子共用，保证两者对"何时启用双 stream"永远不会出现分歧。
    # 输入：self 是查回的 MoE module，num_tokens 是本次的 token 数（hidden_states.shape[0]）
    # 四个条件同时满足才用双 stream：模型启用了双 stream、token 数在 (0, 阈值]、TBO 没在跑、不是被禁止的 PIECEWISE 录图
    # Under TBO the two micro-batches already overlap on separate threads
    # 【译】TBO 下两个微批已经在不同线程上互相重叠了
    # 解释：TBO（two-batch overlap）是另一种重叠手段，见 arg_utils.py enable_tbo 上方的【科普】（第 9 批）；两者不叠加
    from atom.utils.tbo.ubatching import tbo_active

    # Graph ownership belongs to the active frontend.  Eager NONE and
    # whole-model FULL capture both support the fork/join topology; PIECEWISE
    # capture holds it too, but is opt-in (ATOM_DUAL_STREAM_PIECEWISE).
    # 【译】录图归当前的前端（原生 ATOM 或 vLLM 插件）管。eager 的 NONE 和整模型的 FULL 捕获都支持这种分叉 / 汇合拓扑；
    #   PIECEWISE 捕获也能容纳它，但需要手动开启（ATOM_DUAL_STREAM_PIECEWISE）。
    # 解释：get_current_cudagraph_runtime_mode 读 forward_context（插件模式读 vLLM 的）：ModelRunner 只在 PIECEWISE 录图 / 回放
    #   的那次 forward 里把它设成 PIECEWISE，其余时候是 NONE（forward_context.py）。ATOM_DUAL_STREAM_PIECEWISE 默认 "0"
    piecewise_blocked = (
        get_current_cudagraph_runtime_mode() == CUDAGraphMode.PIECEWISE
        and not envs.ATOM_DUAL_STREAM_PIECEWISE
    )
    return (
        self._use_dual_stream
        and 0 < num_tokens <= envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD
        and not tbo_active()
        and not piecewise_blocked
    )


# 算子 torch.ops.aiter.maybe_dual_stream_forward 的真实实现：查回 MoE module，按门控走双 stream 或单 stream
# 输入：hidden_states [T, H]（T = 本次 token 数）；layer_name = MoE module 的 prefix（如 "model.layers.3.mlp"），
#   module 在 __init__ 里以这个 prefix 把自己登记进 static_forward_context
# 输出：[T, H]，路由专家与共享专家已经相加
# 调用方：deepseek_v2.py DeepseekV2MoE.forward（_use_dual_stream 为 True 时）、deepseek_v4.py、kimi_k3/model.py；
#   deepseek_v2.py 的 PCP MoE 合并算子在 dummy / warmup 时也直接调用这个 Python 函数
def maybe_dual_stream_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    if _dual_stream_is_active(self, hidden_states.shape[0]):
        return self.dual_stream_moe_forward(hidden_states)
    return self.single_stream_moe_forward(hidden_states)


# fake 实现：抓图 / 假跑时只需要知道输出和输入同形状同 dtype
def _maybe_dual_stream_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_forward",
    op_func=maybe_dual_stream_forward,
    # Op returns a fresh tensor; never writes into `hidden_states`. Declaring
    # `mutates_args=["hidden_states"]` (the V2 original) misleads the
    # functionalization pass into inserting defensive input clones.
    # 【译】算子返回一个新张量，从不写入 `hidden_states`。若声明 `mutates_args=["hidden_states"]`（V2 最初的写法），
    #   会误导 functionalization pass 插入防御性的输入拷贝。
    mutates_args=(),
    fake_impl=_maybe_dual_stream_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Dual-stream MoE dispatch, routed/shared returned UNSUMMED (K3)
# ---------------------------------------------------------------------------
#
# Caller contract (the MoE module looked up by `layer_name`):
#   - `_use_dual_stream: bool`
#   - `single_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`
#   - `dual_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`
#
# Same gating as `maybe_dual_stream_forward`; the only difference is the return.
# The op above must sum internally because torch's schema inference has no
# representation for an optional tensor in a tuple return -- so a model that can
# legally defer the routed + shared add across the layer boundary (K3, whose
# next attn_res folds both addends into its on-load) cannot use it without
# paying the [T, H] elementwise kernel it is trying to skip.
#
# This op returns both branches instead. Both returns are always real tensors,
# which the schema does allow; the caller decides whether deferring is legal
# from its own static config, NOT from anything about these tensors. A model
# whose branches must be summed before a collective keeps using the op above.
#
# Deliberately a separate op rather than a widened signature on
# `maybe_dual_stream_forward`: that one is shared with DeepSeek V2/V3.2/V4,
# none of which defer the add, and all of which live in @support_torch_compile
# files.
# 【译】双 stream MoE 分发，routed / shared 不相加、分开返回（K3）
#
#   调用方约定（按 `layer_name` 查回的 MoE module 必须有）：
#     - `_use_dual_stream: bool`（是否启用双 stream）
#     - `single_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`（单 stream 实现）
#     - `dual_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`（双 stream 实现）
#
#   门控与 `maybe_dual_stream_forward` 相同，唯一的区别是返回值。上面那个算子必须在内部相加，因为 torch 的 schema 推断
#   无法表示"tuple 返回值里的可选张量"——所以一个可以合法地把 routed + shared 的加法推迟到层边界之后的模型
#   （K3：它下一个 attn_res 在加载输入时就把两个加数一起折算进去）用不了它，否则就要付出它本想省掉的那个 [T, H] 逐元素 kernel。
#
#   这个算子改为把两个分支都返回。两个返回值永远是真实张量，这是 schema 允许的；能否推迟相加由调用方根据自己的静态配置决定，
#   而不是根据这两个张量的任何性质。分支必须在集合通信之前相加的模型，继续用上面那个算子。
#
#   有意做成一个单独的算子，而不是放宽 `maybe_dual_stream_forward` 的签名：那个算子由 DeepSeek V2/V3.2/V4 共用，
#   它们都不推迟相加，而且都位于 @support_torch_compile 文件中。
# 解释：改一个被三个已编译模型共用的算子签名，会改变它们的计算图（也会让它们的编译缓存失效），所以宁可新增一个


# 算子 torch.ops.aiter.maybe_dual_stream_split_forward 的真实实现：与上面相同的门控，返回 (routed, shared) 两个 [T, H]
# 调用方：kimi_k3/model.py 的 MoE（下一层可以推迟相加时）
def maybe_dual_stream_split_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    if _dual_stream_is_active(self, hidden_states.shape[0]):
        return self.dual_stream_split_moe_forward(hidden_states)
    return self.single_stream_split_moe_forward(hidden_states)


def _maybe_dual_stream_split_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden_states), torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_split_forward",
    op_func=maybe_dual_stream_split_forward,
    mutates_args=(),
    fake_impl=_maybe_dual_stream_split_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Sparse indexer score + top-k (V2/V3.2/V4)
# ---------------------------------------------------------------------------
#
# Caller contract (the Indexer module looked up by `layer_name`):
#   - `indexer_score_topk(q_quant, weights, q_scale, topk) -> Tensor`  — real impl,
#     (q_quant is FP8, or packed FP4 uint8 when the FP4 indexer is on; q_scale is
#     the paired e8m0 Q scale on the FP4 path, None for FP8)
#     must return `[total_tokens, topk] int32` indices
#
# `topk` is on the op signature (not derived from the module) so the fake
# impl can size the output without any module lookup.
#
# Other inputs (block_tables, KV cache, per-fwd metadata) are read by the
# module from `self` or `get_forward_context().attn_metadata`.
#
# Why opaque (same rationale for V2 and V4):
#   - prefill paths allocate scratch tensors with shapes that depend on
#     per-fwd `total_committed` / `total_kv` — Dynamo's fake-tensor pass
#     can't size them without a graph break.
#   - decode paths write into module-owned buffers; keeping the buffers
#     out of the op signature avoids functionalization clones.
# 【译】稀疏索引器的打分 + top-k（V2 / V3.2 / V4）
#
#   调用方约定（按 `layer_name` 查回的 Indexer module 必须有）：
#     - `indexer_score_topk(q_quant, weights, q_scale, topk) -> Tensor` —— 真正的实现
#       （q_quant 是 FP8，开了 FP4 索引器时是打包的 FP4 uint8；q_scale 在 FP4 路径上是配套的 e8m0 Q scale，FP8 时为 None）
#       必须返回 `[total_tokens, topk] int32` 的下标
#
#   `topk` 放在算子签名里（而不是从 module 推导），这样 fake 实现不用查 module 就能确定输出大小。
#
#   其他输入（block_tables、KV cache、每次 forward 的元数据）由 module 从 `self` 或 `get_forward_context().attn_metadata` 读取。
#
#   为什么做成黑盒（V2 和 V4 理由相同）：
#     - prefill 路径会分配临时张量，形状取决于每次 forward 的 `total_committed` / `total_kv`——Dynamo 的 fake 张量 pass
#       不 graph break 就无法确定它们的大小。
#     - decode 路径写入 module 自己的 buffer；让 buffer 不出现在算子签名里，可以避免 functionalization 的拷贝。
# 解释：DSA 稀疏注意力与索引器是什么见 model_engine/arg_utils.py 的【科普】DSA（以及 config.py glm5_kpool_block_size 上方【科普】②）
# 观察：标题写 V2/V3.2/V4，但在 v0.1.8 里只有 deepseek_v4.py 的 Indexer 调用 torch.ops.aiter.indexer_score_topk；
#   V3.2（deepseek_v2.py）的索引器走的是同文件里的 sparse_attn_indexer / sparse_attn_indexer_pcp


# 算子 torch.ops.aiter.indexer_score_topk 的真实实现：查回 Indexer module，调用它的同名方法
# 输入：q_quant [T, 索引头数, 索引 head_dim]（FP8，或 FP4 打包），weights 各索引头的权重，q_scale 见上，topk 选多少个
# 输出：[T, topk] int32，每个 token 选中的 KV 位置下标
# 调用方：deepseek_v4.py Indexer.score_topk_from（layer_name 传 self.prefix）。插件（plugin/sglang、plugin/vllm 的 deepseek_v4）
#   替换 / 覆写的是 module 上的 Indexer.indexer_score_topk 方法，这个算子本身不变
def indexer_score_topk(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor | None,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    indexer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return indexer.indexer_score_topk(q_quant, weights, q_scale, topk)


# fake 实现：输出 [T, topk] int32，只用到签名里的 topk，不需要查 module
def _indexer_score_topk_fake(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor | None,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    return torch.empty(
        (q_quant.shape[0], topk),
        dtype=torch.int32,
        device=q_quant.device,
    )


direct_register_custom_op(
    op_name="indexer_score_topk",
    op_func=indexer_score_topk,
    # Output is a fresh tensor (per module contract). Internal buffer
    # mutations on the module are looked up via `layer_name`, not passed
    # in, so functionalization stays unaware and skips defensive clones.
    # 【译】输出是一个新张量（按 module 的约定）。module 内部 buffer 的修改是经 `layer_name` 查到的，而不是作为参数传入，
    #   所以 functionalization 察觉不到它们，也就不会插入防御性拷贝。
    mutates_args=(),
    fake_impl=_indexer_score_topk_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# 【分支】TBO（two-batch overlap，第 9 批）下纯 TP 的 all_reduce：torch.ops.aiter.tbo_all_reduce 的真实实现
#   TBO 是什么见 model_engine/arg_utils.py enable_tbo 上方的【科普】。做成黑盒算子的原因同上：tbo_active()、当前是哪个微批
#   都是运行时状态，写在被追踪的代码里会被冻结
# 输入 / 输出：x 是本 rank 的部分和 [T, H]，返回 TP 组内求和后的结果（不开 TBO 时就是普通的 tensor_model_parallel_all_reduce）
# 调用方：model_ops/communication_op.py 的 tensor_model_parallel_all_reduce（纯 TP + TBO 时改走这个算子）
def tbo_all_reduce(x: torch.Tensor) -> torch.Tensor:
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce

    from atom.utils.tbo.ubatching import tbo_active

    if not tbo_active():
        return tensor_model_parallel_all_reduce(x)

    # Default "inline": correct Plan-A baseline (no overlap, never hangs). Only
    # move the AR onto the comm stream when explicitly opted into "overlap".
    # 【译】默认 "inline"：正确的方案 A 基线（不重叠，永不挂死）。只有显式选择 "overlap" 时才把 AR 移到通信 stream 上。
    # 观察：这条注释与 envs.py 不一致——ATOM_TBO_TP_AR_MODE 的默认值是 "overlap"，所以默认走的是下面的重叠路径，不是 inline
    if envs.ATOM_TBO_TP_AR_MODE != "overlap":
        return tensor_model_parallel_all_reduce(x)

    from atom.utils.tbo.ubatching import (
        tbo_current_ubatch_id,
        tbo_get_comm_stream,
        tbo_get_ubatch_tp_comm,
        tbo_switch_to_compute_sync,
        tbo_yield_and_switch_from_compute_to_comm,
    )

    # 重叠路径：每个微批有自己的通信器（pynccl），在通信 stream 上做 all_reduce，同时把 CPU 交给另一个微批去发它的计算
    ubatch_id = tbo_current_ubatch_id()

    ub_comm = tbo_get_ubatch_tp_comm(ubatch_id)
    if ub_comm is None:
        # world_size == 1: nothing to reduce.
        # 【译】world_size == 1：没有需要归约的。
        return x

    # Hand the CPU baton to the partner ubatch and move onto the comm stream,
    # so this AR overlaps the partner's compute.
    # 【译】把 CPU 的接力棒交给伙伴微批，自己切到通信 stream 上，这样这次 AR 就能与伙伴的计算重叠。
    tbo_yield_and_switch_from_compute_to_comm()
    comm_stream = tbo_get_comm_stream()
    # out-of-place all_reduce on this ubatch's dedicated communicator, launched
    # on the comm stream (current stream after the switch above).
    # 【译】在本微批专用的通信器上做非原地的 all_reduce，在通信 stream 上发起（上面切换之后它就是当前 stream）。
    x = ub_comm.all_reduce(x, stream=comm_stream)

    # 切回计算 stream，并等通信 stream 上的 all_reduce 完成后再继续
    tbo_switch_to_compute_sync()
    return x


def _tbo_all_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="tbo_all_reduce",
    op_func=tbo_all_reduce,
    mutates_args=(),
    fake_impl=_tbo_all_reduce_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
