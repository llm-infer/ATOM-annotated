# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Routing a checkpoint tensor to the parameter (or parameters) it feeds.

By the time a tensor gets here its name has already been rewritten
(`CheckpointNameRewriter`); what remains is deciding *how* it is written, which
depends on how the checkpoint packs its weights:

  packed        one checkpoint tensor covers several parameters, or several
                cover one (`packed_modules_mapping`)
  fused expert  all routed experts of a layer in a single stacked tensor
  per expert    one tensor per (expert, shard), the batched staging path
  merged        `experts.<id>.<name>` feeding a merged parameter
  generic       everything else

The branches are ordered, and falling through one means "not mine, try the
next". Two of them also carry state across tensors: the fused-expert format is
detected from the first tensor that looks like it, and that detection then
picks the mapping used for every tensor after it.
"""

# 【译】把一个 checkpoint 张量路由到它所供给的那个（或那些）参数。
#   张量到这里时名字已经改写过了（`CheckpointNameRewriter`）；剩下要决定的是*怎么*写，这取决于 checkpoint 如何打包权重：
#     packed        一个 checkpoint 张量对应多个参数，或者多个对应一个（`packed_modules_mapping`）
#     fused expert  一层的所有路由专家在一个堆叠张量里
#     per expert    每个 (专家, 分片) 一个张量，走批量暂存路径
#     merged        `experts.<id>.<name>` 供给一个合并参数
#     generic       其他一切
#   这些分支是有序的，落空一个就意味着"不是我的，试下一个"。其中两个还跨张量携带状态：融合专家格式由第一个看起来像它的张量
#   检测出来，然后这次检测决定之后每个张量所用的映射
#
# 【导读】weight_dispatch.py：全流程图（loader.py 文件头）里"分发"那一格。前置：TP 切分、融合参数、MoE 专家堆叠见 loader.py
#   文件头【前置①~③】。唯一使用者：loading_core.load_weights_into_model 构造 WeightDispatcher，主循环对每个张量调
#   dispatch(原 checkpoint 名, 改名后的名字, 张量)；本类只决定"交给谁、带什么参数"，真正的切片和拷贝在各参数的 weight_loader 里
#   判定顺序（每条路径一个例子）：
#
#     dispatch（入口）
#      ├─ _dispatch_packed      命中 packed_modules_mapping 的 key 就认领（专家张量除外），目标参数不存在也不再往下试
#      │     例：…self_attn.k_proj.weight → …qkv_proj.weight，shard "k"（逐步推算见 loader.py 文件头【例】）
#      │         …mlp.gate_proj.weight → …mlp.gate_up_proj.weight，shard 0
#      └─ _dispatch_expert
#           ├─ 模型没有 get_expert_mapping（dense 模型）→ _dispatch_generic
#           ├─ _dispatch_fused_expert   Qwen3.5 BF16：…mlp.experts.gate_up_proj 一个张量装着全部专家
#           │     → …mlp.experts.w13_weight，交给模型的 load_fused_expert_weights 按专家切开再写
#           ├─ _dispatch_per_expert     …mlp.experts.17.gate_proj.weight → …mlp.experts.w13_weight，expert 17，shard "w1"
#           │     → 可暂存时 ExpertStagingPool.stage，否则 FusedMoE.weight_loader(param, t, name, "w1", 17)
#           └─ _dispatch_merged_or_generic
#                ├─ merged：…mlp.experts.100.w2_bias（名字里恰有两个数字段）→ …mlp.experts.w2_bias，expert 100
#                └─ generic：其余，例如 …input_layernorm.weight → 同名参数（RMSNorm 没有专属 loader → default_weight_loader）
#
#   per expert 带数字走一遍（DeepSeek-V3，TP = 8，不开 EP，FP8 checkpoint，I = 2048，H = 7168）：
#     checkpoint 张量 model.layers.3.mlp.experts.17.gate_proj.weight [2048, 7168]
#     → _dispatch_packed：名字含 "mlp.experts." 且不是现成的参数名 → 跳过所有 packed 规则（否则 "gate_proj" 会误命中）
#     → _dispatch_per_expert：在 expert_weight_prefixes 里命中 "experts.17.gate_proj." → ("experts.w13_", 17, "w1")
#       名字变成 model.layers.3.mlp.experts.w13_weight；本 rank 的参数 [E_local, 512, 7168]（E_local = 256 个路由专家
#       + 融合进来的共享专家槽位；512 = 2 × 2048/8，前 256 行放 w1 = gate，后 256 行放 w3 = up）
#     → 开了线程池且该参数可暂存：submit(staging_pool.stage, param, name, "w1", 17, tensor)
#       FusedMoE.stage_expert_weight 取 tensor 的第 [256r, 256r+256) 行，写进 pinned 缓冲的 [17, 0:256, :]
#     → 这一层的 w13 收齐 512 份（256 个专家 × w1 / w3）后一次 copy_ 写回 GPU（expert_staging.py）
# 对比nano-vllm：nano-vllm 只有 packed + generic 两条路径（它不支持 MoE）

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from atom.model_loader.weight_names import extract_expert_target_and_id

logger = logging.getLogger("atom")


# 分发器：每次加载新建一个（目标模型、草稿模型各一次），持有本次加载的全部上下文；
#   状态只有两份结果（loaded_weights_record、dropped_ckpt_keys）和融合专家格式的检测结果
class WeightDispatcher:
    """Writes checkpoint tensors into model parameters.

    Accumulates `loaded_weights_record` (parameter names that were written, for
    the plugin hosts' own load checks) and `dropped_ckpt_keys` (tensors whose
    rewritten name matches no parameter, which is almost always a rewrite-rule
    bug and is reported after loading).
    """

    # 【译】把 checkpoint 张量写进模型参数。
    #   累积 `loaded_weights_record`（写入过的参数名，供插件宿主自己的加载检查）和 `dropped_ckpt_keys`（改名后的名字匹配不到
    #   任何参数的张量——这几乎总是改名规则的 bug，加载完后报告）

    def __init__(
        self,
        *,
        model: nn.Module,
        params_dict: dict[str, nn.Parameter],
        hf_config: Any,
        prefix: str,
        spec_decode: bool,
        submit: Callable[..., None],
        staging_pool: Any,
        batching_enabled: bool,
        batching_excluded: Callable[[nn.Parameter], bool] | None,
        default_weight_loader: Callable,
        packed_modules_mapping: Mapping[str, Any],
        expert_index: Mapping[str, tuple[str, int, str]],
        expert_weight_prefixes: Sequence[str],
        has_expert_mapping: bool,
        detect_fused_expert_fn: Callable[[str], bool] | None,
        get_fused_expert_mapping_fn: Callable[[], Sequence[tuple]] | None,
        load_fused_expert_weights_fn: Callable | None,
        on_fused_param: Callable[[nn.Parameter], None] | None = None,
    ):
        # 成员（都由 loading_core 传入）：
        #   submit：提交一次 weight_loader 调用（同步 / 线程池 / 经流式量化器，见 loading_core 的 _submit）
        #   staging_pool / batching_enabled / batching_excluded：专家暂存池，开线程池时才启用，流式模块的参数除外
        #   packed_modules_mapping：packed 路径的规则（loader.py 文件头【前置②】）
        #   expert_index / expert_weight_prefixes / has_expert_mapping：per expert 路径的预索引（loading_core.py）
        #   detect_fused_expert_fn / get_fused_expert_mapping_fn / load_fused_expert_weights_fn：fused expert 路径，
        #     只有 Qwen3.5 系列提供（qwen3_5.py）
        self.model = model
        self.params_dict = params_dict
        self.hf_config = hf_config
        self.prefix = prefix
        self.spec_decode = spec_decode
        self.submit = submit
        self.staging_pool = staging_pool
        self.batching_enabled = batching_enabled
        self.batching_excluded = batching_excluded
        self.default_weight_loader = default_weight_loader
        self.packed_modules_mapping = packed_modules_mapping
        self.expert_index = expert_index
        self.expert_weight_prefixes = expert_weight_prefixes
        self.has_expert_mapping = has_expert_mapping
        self.detect_fused_expert_fn = detect_fused_expert_fn
        self.get_fused_expert_mapping_fn = get_fused_expert_mapping_fn
        self.load_fused_expert_weights_fn = load_fused_expert_weights_fn
        # Fused writes bypass `submit` and need storage prepared first.
        # 【译】融合写入绕过 `submit`，需要先把存储准备好（流式量化时参数还在 meta 上，见 online_quant_streaming 的
        #   materialize_fused_param）
        self.on_fused_param = on_fused_param

        self.loaded_weights_record: set[str] = set()
        self.dropped_ckpt_keys: list[tuple[str, str]] = []
        # Latched on the first tensor that looks like a fused-expert one, then
        # used for every tensor after it. Per-load state on purpose: the same
        # model class loads a fused-expert target checkpoint and a per-expert
        # drafter block in two separate passes.
        # 【译】在第一个看起来像融合专家的张量上锁定，之后的每个张量都沿用。刻意做成每次加载的状态：同一个模型类会分两次加载——
        #   一次是融合专家格式的目标 checkpoint，一次是逐专家格式的草稿块
        self._is_fused_expert = False
        self._fused_expert_params_mapping: Sequence[tuple] = ()

    # 入口：先试 packed，没被认领再走专家相关的几条路径。调用方：loading_core 的主循环（主线程）
    def dispatch(self, orig_ckpt_name: str, name: str, tensor: torch.Tensor) -> None:
        if not self._dispatch_packed(orig_ckpt_name, name, tensor):
            self._dispatch_expert(orig_ckpt_name, name, tensor)

    # ── packed modules ────────────────────────────────────────────────────

    def _dispatch_packed(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> bool:
        """True when a packed rule claimed this tensor, handled or abandoned.

        Abandoned counts as claimed: a rule that matched but whose target
        parameter is missing stops the search rather than falling through to
        the expert paths, which is what the original `break` did.
        """
        # 【译】有 packed 规则认领了这个张量（处理了或放弃了）就返回 True。
        #   放弃也算认领：规则匹配上了但目标参数不存在时，停止搜索，而不是落到专家路径——原来的 `break` 就是这样做的
        # 返回 False 才会去试专家路径
        for k in self.packed_modules_mapping:
            # We handle the experts below in expert_params_mapping
            # 【译】专家张量在下面的 expert_params_mapping 路径里处理
            # 为什么要跳过：专家张量名里也有 gate_proj / up_proj（…experts.17.gate_proj.weight），不跳过就会被改成
            #   …experts.17.gate_up_proj.weight 而找不到参数；"不是现成的参数名"这个条件给名字恰好就是参数名的情况留了口子
            if (
                "mlp.experts." in name
                or "ffn.experts." in name
                or "block_sparse_moe.experts." in name
            ) and name not in self.params_dict:
                continue
            # 按 dict 的插入顺序找第一个是名字子串的 key（同 nano-vllm 的 `if k in weight_name`）
            # 观察：子串匹配加上插入顺序，使 list 形式的 "gate_up_proj" 这个 key 实际匹配不到——它本身包含更早的 key "up_proj"。
            #   qwen3_moe / qwen3_next / qwen3_5 的映射都把它放在 "up_proj" 之后，于是 …mlp.gate_up_proj.weight 会先被 "up_proj"
            #   认领、改成 …mlp.gate_gate_up_proj.weight、记进 dropped_ckpt_keys（据代码推断）。这些 list 项起作用的地方是
            #   config.py QuantizationConfig.remap_layer_name（量化层名的改写），那里先加 "." 再匹配
            if k not in name:
                continue
            packed_value = self.packed_modules_mapping[k]
            # Handle both tuple (fuse parameter) and list (shard parameter)
            # 【译】同时处理元组（融合参数）和列表（拆分参数）两种值
            if isinstance(packed_value, list):
                # Checkpoint has fused weight, split into separate params
                # 【译】checkpoint 里是融合后的权重，拆进分开的几个参数：每个目标参数都收到同一个完整张量，
                #   由它自己的 weight_loader 按 shard_idx（第几份）切出自己那部分
                for shard_idx, target_name in enumerate(packed_value):
                    param_name = name.replace(k, target_name)
                    if "output_scale" in param_name:
                        continue
                    param = self._parameter(orig_ckpt_name, param_name)
                    if param is None:
                        continue
                    self.submit(param.weight_loader, param, tensor, shard_idx)
                    self._record(param_name)
            else:
                # Checkpoint has separate weights, load into fused param
                # 【译】checkpoint 里是分开的权重，加载进融合参数（最常见的情况：q/k/v → qkv_proj、gate/up → gate_up_proj）
                v, shard_id = packed_value
                param_name = name.replace(k, v)
                # FIXME output_scale has a value, so accuracy is incorrect. this should be loaded and used in llfp4.
                # 【译】FIXME：output_scale 是有值的，所以精度不对。它应该被加载并在 llfp4 中使用
                #   （现状：名字含 output_scale 的张量在这里被直接跳过；_report_coverage 也把 output_scale 列为"无害的丢弃"）
                if "output_scale" not in param_name:
                    param = self._parameter(orig_ckpt_name, param_name)
                    if param is None:
                        return True
                    self.submit(param.weight_loader, param, tensor, shard_id)
                    self._record(param_name)
            return True
        return False

    # ── expert paths ──────────────────────────────────────────────────────

    # 专家相关的三条路径 + 兜底（顺序见文件头的树）。dense 模型（没有 get_expert_mapping）直接走 generic
    def _dispatch_expert(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        self._detect_fused_expert_format(name)
        if not self.has_expert_mapping:
            self._dispatch_generic(orig_ckpt_name, name, tensor)
            return
        if self._dispatch_fused_expert(name, tensor):
            return
        matched, name = self._dispatch_per_expert(name, tensor)
        if not matched:
            self._dispatch_merged_or_generic(orig_ckpt_name, name, tensor)

    # 融合专家格式的检测：一旦某个张量被模型判定为融合格式就锁定为 True，并取一次映射；之后不再检测
    #   例（qwen3_5.detect_fused_expert_format）：名字含 "experts.gate_up_proj" → True；"experts.0.gate_proj" → False
    def _detect_fused_expert_format(self, name: str) -> None:
        if self.detect_fused_expert_fn is None or self._is_fused_expert:
            return
        self._is_fused_expert = self.detect_fused_expert_fn(name)
        if self._is_fused_expert and self.get_fused_expert_mapping_fn is not None:
            self._fused_expert_params_mapping = self.get_fused_expert_mapping_fn()

    def _dispatch_fused_expert(self, name: str, tensor: torch.Tensor) -> bool:
        """All routed experts of a layer arriving as one stacked tensor."""
        # 【译】一层的所有路由专家作为一个堆叠张量到达
        # 例：Qwen3.5 BF16 的 …mlp.experts.gate_up_proj [E, 2I, H]：映射 ("experts.w13_weight", "experts.gate_up_proj", "w1")
        #   → 名字换成 …mlp.experts.w13_weight → 模型的 load_fused_expert_weights 把它沿倒数第 2 维切成 gate / up，
        #   再逐个专家调 FusedMoE.weight_loader（qwen3_5.py）
        if not (
            self._is_fused_expert
            and self.load_fused_expert_weights_fn is not None
            and self._fused_expert_params_mapping
        ):
            return False
        for mapping_entry in self._fused_expert_params_mapping:
            param_name, weight_name, shard_id = mapping_entry[:3]
            if weight_name not in name:
                continue
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in self.params_dict:
                continue

            # Writes the routed experts straight into the fused parameter, so
            # the staging pool must not also own it -- see ExpertStagingPool's
            # ownership rule.
            # 【译】它把路由专家直接写进融合参数，所以暂存池不能同时拥有这个参数——见 ExpertStagingPool 的所有权规则
            #   （decline 会先把暂存池里已经攒下的区域写回，再让出这个参数）
            self.staging_pool.decline(self.params_dict[name_mapped])
            if self.on_fused_param is not None:
                self.on_fused_param(self.params_dict[name_mapped])

            # Generic call - model provides implementation details
            # 【译】通用调用——实现细节由模型提供
            num_experts = getattr(self.hf_config, "n_routed_experts", 0) or getattr(
                self.hf_config, "num_experts", 0
            )
            if self.load_fused_expert_weights_fn(
                name,  # Original checkpoint name
                name_mapped,  # Mapped parameter name
                self.params_dict,
                tensor,
                shard_id,
                num_experts,
            ):
                self._record(name_mapped)
                return True
        return False

    def _dispatch_per_expert(self, name: str, tensor: torch.Tensor) -> tuple[bool, str]:
        """One tensor per (expert, shard); the batched staging path.

        Returns the rewritten name alongside the verdict: the expert prefix is
        substituted in place, and the merged/generic fallbacks below work on
        the substituted name.
        """
        # 【译】每个 (专家, 分片) 一个张量；批量暂存路径。
        #   把判定结果和改写后的名字一起返回：专家前缀是原地替换的，下面的 merged / generic 兜底处理的是替换后的名字
        # 例子和数字见文件头 per expert 带数字走一遍 那一段
        for wm_name in self.expert_weight_prefixes:
            if wm_name not in name:
                continue
            pm_name, expert_id, shard_id = self.expert_index[wm_name]
            name = name.replace(wm_name, pm_name)
            # 以下三种情况认领后直接丢弃：模型没有的 bias；不加载草稿时的 mtp 权重；参数不存在（注释见下）
            if name.endswith((".bias", "_bias")) and name not in self.params_dict:
                return True, name
            if "mtp" in name and not self.spec_decode:
                return True, name
            param = self.params_dict.get(name)
            if param is None:
                # Parameter absent from model (e.g. weight scales for an
                # unquantized drafter MTP block); skip silently.
                # 【译】模型里没有这个参数（例如一个不量化的草稿 MTP 块对应的 weight scale）；静默跳过
                #   （注意不进 dropped_ckpt_keys，所以 _report_coverage 看不到这类丢弃）
                return True, name
            # 暂存还是直接写：开了线程池、不是流式模块的参数、并且所属模块支持暂存协议（is_batchable）→ 进暂存池；
            #   否则直接调 FusedMoE.weight_loader(param, tensor, name, shard_id, expert_id)
            batching_excluded = (
                self.batching_excluded is not None and self.batching_excluded(param)
            )
            if (
                self.batching_enabled
                and not batching_excluded
                and self.staging_pool.is_batchable(param, name)
            ):
                self.submit(
                    self.staging_pool.stage, param, name, shard_id, expert_id, tensor
                )
            else:
                self.submit(
                    param.weight_loader, param, tensor, name, shard_id, expert_id
                )
            self._record(name)
            return True, name
        return False, name

    # 专家映射没命中的张量：merged（名字形如 …experts.<id>.<后缀>）或 generic
    #   merged 例：model.layers.10.mlp.experts.100.w2_bias → (model.layers.10.mlp.experts.w2_bias, 100)（weight_names.py
    #   extract_expert_target_and_id），然后 weight_loader(param, tensor, "", "", 100)——weight_name 和 shard_id 传空串，
    #   FusedMoE.weight_loader 看到空 weight_name 且是 MXFP4 时走 mxf4_merged_weight_loader
    def _dispatch_merged_or_generic(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        if "mtp" in name and not self.spec_decode:
            return
        if merged_target := extract_expert_target_and_id(name):
            fused_name, expert_id = merged_target
            param = self._parameter(orig_ckpt_name, fused_name)
            if param is None:
                return
            # Merged loader writes expert slots directly; same ownership rule
            # as the fused path above.
            # 【译】merged loader 直接写专家槽位；所有权规则同上面的融合路径
            self.staging_pool.decline(param)
            weight_loader = getattr(param, "weight_loader", self.default_weight_loader)
            self.submit(
                weight_loader,
                param,
                tensor,
                "",  # use merged moe loader
                "",
                expert_id,
            )
            self._record(fused_name)
            return
        self._dispatch_generic(orig_ckpt_name, name, tensor)

    # 兜底：名字就是参数名。有专属 weight_loader 的用它（如 RowParallelLinear 的 o_proj、VocabParallelEmbedding），
    #   没有的用 default_weight_loader（loader.py）
    def _dispatch_generic(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        param = self._parameter(orig_ckpt_name, name)
        if param is None:
            return
        weight_loader = getattr(param, "weight_loader", self.default_weight_loader)
        self.submit(weight_loader, param, tensor)
        self._record(name)

    # ── helpers ───────────────────────────────────────────────────────────

    # 按名字找参数；找不到时记进 dropped_ckpt_keys 并返回 None（不抛异常——对比nano-vllm：get_parameter 找不到直接报错）
    #   model.get_parameter("a.b.c")：按带点的全名逐级找子模块再取参数，找不到抛 AttributeError
    def _parameter(self, orig_ckpt_name: str, param_name: str) -> nn.Parameter | None:
        try:
            param = self.model.get_parameter(param_name)
        except AttributeError:
            self.dropped_ckpt_keys.append((orig_ckpt_name, param_name))
            return None
        # Loaders are handed `(param, tensor)` and nothing else, so a failure
        # inside one can only report shapes, leaving the weight to be identified
        # by arithmetic. Carry both names on the parameter instead. Each
        # parameter is dispatched by a single task, so this is safe to write
        # here and read from the worker.
        # 【译】loader 只拿到 `(param, tensor)`，所以 loader 内部出错时只能报形状，权重是哪一个只能靠算术去推。改为把两个名字
        #   都挂在参数上。每个参数只由一个任务分发，所以在这里写、在工作线程里读是安全的
        # 注：packed 参数会收到多个任务（qkv_proj 收到 q、k、v 三个），"只由一个任务分发"对它不成立，后到的张量会覆盖这个属性；
        #   但读这个属性的只有 default_weight_loader 的报错（loader.py _weight_identity），而 packed 路径用的是参数自己的
        #   weight_loader，所以目前不影响
        param._atom_load_names = (param_name, orig_ckpt_name)
        return param

    # 记账：提交写入任务时就记（不等写完），带上插件宿主的 prefix
    def _record(self, param_name: str) -> None:
        self.loaded_weights_record.add(self.prefix + param_name)
