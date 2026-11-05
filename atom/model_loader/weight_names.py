# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint-name rewriting for the model loader.

Turning an on-disk tensor name into the parameter name it belongs to is a
sequence of model-declared rules (prefix/suffix/substring maps, MTP filters,
shared-expert fusion).  Keeping it here — free of AITER and of any GPU state —
means the rules can be unit-tested directly instead of only through a full
`load_model` run.
"""

# 【译】模型加载器的 checkpoint 名改写。
#   把磁盘上的张量名变成它所属的参数名，是一串由模型声明的规则（前缀 / 后缀 / 子串映射、MTP 过滤、共享专家融合）。
#   把它放在这里——不依赖 AITER、不碰任何 GPU 状态——意味着这些规则可以直接做单元测试，而不必只能通过完整的 `load_model` 运行
#
# 【导读】weight_names.py：全流程图（loader.py 文件头）里"改名"那一格。使用者：loading_core.load_weights_into_model 构造
#   CheckpointNameRewriter，对每个 checkpoint 名调一次 rewrite（结果被缓存，迭代器用它判断"要不要读这个张量"）；
#   WeightsMapper 还被各模型、插件、config.py 直接使用（loader.py 重新导出了它）；extract_expert_target_and_id 给 weight_dispatch
#   的 merged 路径用。单元测试：tests/test_weight_name_rewrite.py
#
#   CheckpointNameRewriter.rewrite 的 9 条规则（按执行顺序；任何一步得到 None = 这个模型不要该张量）：
#     ① WeightsMapper（模型的 weights_mapper / hf_to_atom_mapper，或插件显式传入）：子串 / 前缀 / 后缀改名，映射到 None 即丢弃
#          例：DeepSeek-V4.1 后缀 ".gate.bias" → ".gate.e_score_correction_bias"；子串 ".engram.embed." → None（Engram 表不是参数）
#     ② 名字含 "mtp" 而现在不是在加载草稿模型 → 丢弃
#     ③ 以 kv_scale 结尾，或含 inv_freq（RoPE 的频率表，ATOM 启动时自己算）→ 丢弃
#     ④ 以 skip_weight_prefixes 中任一前缀开头 → 丢弃（例：Kimi-K2.5 只跑文本时的 "vision_tower."、"mm_projector."）
#     ⑤ 加载草稿模型且模型定义了 remap_mtp_weight_name → 用它改名或过滤（例：Qwen3.5 MTP 只留 "mtp." 开头和 embed / lm_head）
#     ⑥ weights_mapping 子串替换（每一条都试，命中就整串替换）
#          例：Qwen3.5 MTP 的 {"mtp.": "model."}；DeepSeek-V4.1 的 {".scale": ".weight_scale_inv"}，接着 ⑦ 再变成 ".weight_scale"
#     ⑦ weight_scale_inv → weight_scale（DeepSeek 的 FP8 checkpoint 把块 scale 叫 weight_scale_inv，存的就是反量化时要乘的数；
#          ATOM 的参数统一叫 weight_scale）
#     ⑧ 不加载草稿时，model.layers.{i}. 的 i ≥ num_hidden_layers → 丢弃（例：V3 的 checkpoint 里第 61 层是 MTP 层，目标模型只有 0~60）
#     ⑨ 共享专家融合（量化格式相同等条件满足时）：…mlp.shared_experts.X → …mlp.experts.{n_routed}.X（V3 是 experts.256），
#          之后像普通专家一样分发进 FusedMoE 的堆叠张量（loader.py 文件头【前置③】）
#   为什么顺序固定：② 必须在 ⑤ 之前——草稿模型会把 mtp.* 改成不再含 "mtp" 的名字（Qwen3.5 MTP：⑥ 把 "mtp." 改成 "model."）；
#     ⑦ 要在 ⑥ 之后，V4.1 才能经 ".scale" → ".weight_scale_inv" → ".weight_scale" 两步到位
#   例（DeepSeek-V3 FP8，目标模型）：model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv
#     ①~⑥ 不变 → ⑦ …shared_experts.down_proj.weight_scale → ⑧ 3 < 61 保留 → ⑨ …mlp.experts.256.down_proj.weight_scale
#     → 分发时命中 "experts.256.down_proj." → …mlp.experts.w2_weight_scale 的第 256 个专家槽位（weight_dispatch.py）
# 对比nano-vllm：nano-vllm 没有这一层，checkpoint 名除了 packed_modules_mapping 的替换外必须和参数名一模一样

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

# WeightsMapper is adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py
# 【译】WeightsMapper 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py
WeightsMapping = Mapping[str, str | None]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""
# 【译】如果某个 key 映射到 `None`，对应的权重被忽略
# （这是给上面类型别名写的"属性 docstring"：只是一个字符串表达式，Python 不会把它挂到任何对象上，只给读代码的人看）


# 去掉映射到 None 的条目；调用方：WeightsMapper.get_rename_mapper
def _drop_ignored(mapping: WeightsMapping) -> dict[str, str]:
    """Keep only the genuine renames of `mapping`, dropping the `None` values."""
    # 【译】只保留 `mapping` 中真正的改名，去掉值为 `None` 的条目
    return {orig: new for orig, new in mapping.items() if new is not None}


# 三张改名表：子串 / 前缀 / 后缀，按这个顺序依次应用（每张表内按插入顺序逐条试）；任何一条映射到 None 就丢弃这个名字
#   例：WeightsMapper(orig_to_new_prefix={"model.": "model.language_model."})（插件里常见）
#     改写前后："model.layers.0.mlp.gate_proj.weight" → "model.language_model.layers.0.mlp.gate_proj.weight"
#   使用者：本文件的 CheckpointNameRewriter（规则 ①）、各模型类的 weights_mapper / hf_to_atom_mapper、插件；
#     config.py 的 remap_layer_name 也调插件传入的 weights_mapper._map_name 改写量化排除项
#   @dataclass + field(default_factory=dict)：每个实例都拿到自己的一个新 dict（直接写 = {} 会让所有实例共享同一个可变对象）
@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    # 【译】如果权重名匹配下面的模式，就改写它

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        # 【译】合并两个 `WeightsMapper` 的映射得到一个新的（同名 key 以右边为准）
        # __or__：重载 `|` 运算符，mapper_a | mapper_b
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    # 核心：依次过三张表。注意替换细节：
    #   子串、前缀都只替换第一次出现（replace(..., 1)）；前缀用 startswith 判断
    #   后缀：key.rsplit(suffix, 1) 从右边切一次再用 new_key 连回去 = 只替换最后一次出现
    #   例：后缀 {".gate.bias": ".gate.e_score_correction_bias"}：
    #     "layers.3.ffn.gate.bias" → "layers.3.ffn.gate.e_score_correction_bias"；"layers.3.ffn.gate.bias_vl" 不以它结尾，不变
    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    # 下面三个是批量版本（对 (名字, 张量) 迭代器 / 名字列表 / 字典），映射到 None 的项被过滤掉；vLLM 风格的接口，
    #   ATOM 自己的加载路径只用 _map_name
    #   (out_name := ...)：海象运算符，在表达式里赋值并返回该值
    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }

    def get_rename_mapper(self) -> "WeightsMapper":
        """Return the mapper variant suitable for quantization config names.

        Called by vLLM, not by ATOM: a repo-wide grep finds no caller here
        beyond `tests/test_weight_name_rewrite.py`, because this is one of the
        methods vLLM duck-types on whatever a model exposes as
        `hf_to_vllm_mapper`. ATOM's vLLM plugin models publish their own
        `WeightsMapper` under that name (e.g.
        `atom/plugin/vllm/models/qwen3_5.py`, `hf_to_vllm_mapper =
        hf_to_atom_mapper`), so these three vLLM 0.31 call sites land here:

        * `model_executor/model_loader/utils.py` and
          `model_executor/models/interfaces.py` --
          `quant_config.apply_vllm_mapper(hf_to_vllm_mapper.get_rename_mapper())`
        * `lora/worker_manager.py` -- LoRA weight-name parsing.

        Renamed from `get_unstacked_mapper` to follow vLLM's own rename; the
        name is part of that external contract, so it is not ours to choose.

        Consumers that *name* modules rather than load them (vLLM's
        quantization config layer lists, LoRA name parsing) cannot act on a
        `None` mapping — "do not load this weight" would silently shrink an
        ignore list — so those entries are dropped.  ATOM keeps packed-module
        handling separate from checkpoint-name rewriting, so there are no
        vLLM-style stacked mappings to drop here.
        """
        # 【译】返回适合量化配置层名的映射器变体。
        #   由 vLLM 而不是 ATOM 调用：在整个仓库里 grep，除了 `tests/test_weight_name_rewrite.py` 找不到调用方，因为这是 vLLM 对模型
        #   以 `hf_to_vllm_mapper` 暴露的任何东西做鸭子类型调用的方法之一。ATOM 的 vLLM 插件模型在这个名字下发布自己的
        #   `WeightsMapper`（例如 `atom/plugin/vllm/models/qwen3_5.py` 里 `hf_to_vllm_mapper = hf_to_atom_mapper`），所以 vLLM 0.31 的
        #   这三个调用点会落到这里：
        #   * `model_executor/model_loader/utils.py` 和 `model_executor/models/interfaces.py` ——
        #     `quant_config.apply_vllm_mapper(hf_to_vllm_mapper.get_rename_mapper())`（量化配置改名）
        #   * `lora/worker_manager.py` —— LoRA 权重名解析。
        #   从 `get_unstacked_mapper` 改名而来，以跟随 vLLM 自己的改名；这个名字是那份外部契约的一部分，不由我们来选。
        #   *命名*模块而不是加载模块的使用方（vLLM 量化配置的层列表、LoRA 名字解析）没法处理 `None` 映射——"不加载这个权重"会悄悄
        #   缩小一个忽略列表——所以去掉这些条目。ATOM 把 packed 模块的处理和 checkpoint 改名分开了，所以这里没有 vLLM 风格的
        #   堆叠映射需要去掉
        return WeightsMapper(
            orig_to_new_substr=_drop_ignored(self.orig_to_new_substr),
            orig_to_new_prefix=_drop_ignored(self.orig_to_new_prefix),
            orig_to_new_suffix=_drop_ignored(self.orig_to_new_suffix),
        )


# 规则 ⑨ 的第一步：名字里有没有共享专家的模块段；调用方：CheckpointNameRewriter._maybe_fuse_shared_expert
#   单数形式带着结尾的 "."（"shared_expert."），不会误匹配复数的 "shared_experts."，所以列表顺序不影响结果
def have_shared_expert(name: str) -> str | None:
    """Return the `...shared_expert(s).` substring in `name`, if any.

    Matches both `mlp.` (GLM4, Qwen, ...) and `ffn.` (DeepSeek-V4) module
    naming. The matched substring is replaced by the caller with
    `<prefix>experts.{n_routed}.` so the shared expert lands in the fused
    MoE buffer's extra slot. Returning the full prefix (incl. mlp./ffn.)
    lets the rewrite preserve the module-naming style.
    """
    # 【译】返回 `name` 里 `...shared_expert(s).` 这一段（如果有的话）。
    #   同时匹配 `mlp.`（GLM4、Qwen……）和 `ffn.`（DeepSeek-V4）两种模块命名。调用方把匹配到的子串替换成
    #   `<prefix>experts.{n_routed}.`，让共享专家落进融合 MoE buffer 多出来的那个槽位。返回完整前缀（含 mlp. / ffn.）
    #   能让改写保留原来的模块命名风格
    maybe_matching_list = [
        "block_sparse_moe.shared_experts.",
        "block_sparse_moe.shared_expert.",
        "mlp.shared_experts.",
        "mlp.shared_expert.",
        "ffn.shared_experts.",
        "ffn.shared_expert.",
    ]
    for maybe_matching_name in maybe_matching_list:
        if maybe_matching_name in name:
            return maybe_matching_name
    return None


# 例：name = "model.layers.3.mlp.shared_experts.gate_proj.weight"，matching_name = "mlp.shared_experts."
#   → ("model.layers.3.mlp.shared_experts", "model.layers.3.mlp.experts")，交给 fuse_shared_expert 比较两者的量化规格
#   调用方：CheckpointNameRewriter._maybe_fuse_shared_expert
def shared_expert_prefixes(name: str, matching_name: str) -> tuple[str, str]:
    """Split `name` into the (shared expert, routed expert) module prefixes.

    Both are needed to decide whether the two are quantized identically and can
    therefore share one fused MoE buffer.
    """
    # 【译】把 `name` 拆成 (共享专家, 路由专家) 两个模块前缀。
    #   判断两者是否以相同方式量化、从而能否共用一个融合 MoE buffer 时，两个都要用到
    layer_prefix = name.split(matching_name, 1)[0]
    module_prefix = matching_name.split("shared_expert", 1)[0]
    return (
        layer_prefix + matching_name.rstrip("."),
        layer_prefix + f"{module_prefix}experts",
    )


# 改名器（9 条规则见文件头）。每次加载由 loading_core 构造一个，字段都来自模型类属性和 hf_config
@dataclass
class CheckpointNameRewriter:
    """Turns an on-disk tensor name into the parameter name it belongs to.

    Nine model-declared rules applied in a fixed order; `rewrite` returns None
    for a tensor this model does not want. Order matters and is not obvious --
    the MTP filter has to run before `mtp_remap`, because a drafter checkpoint
    remaps `mtp.*` onto names that no longer contain "mtp" -- so it lives in
    one place rather than being spread through the loading loop.

    `fuse_shared_expert(shared_prefix, routed_prefix) -> bool` is injected: the
    real decision reads the quantization config, which would drag AITER into
    this module and out of reach of the unit test gate.
    """

    # 【译】把磁盘上的张量名变成它所属的参数名。
    #   九条由模型声明的规则按固定顺序应用；对这个模型不要的张量，`rewrite` 返回 None。顺序很重要而且不显然——MTP 过滤必须在
    #   `mtp_remap` 之前执行，因为草稿 checkpoint 会把 `mtp.*` 改到不再含 "mtp" 的名字上——所以顺序放在一个地方，而不是散落在
    #   加载循环里。
    #   `fuse_shared_expert(shared_prefix, routed_prefix) -> bool` 是注入的：真正的判断要读量化配置，那会把 AITER 拖进本模块，
    #   使它脱离单元测试门禁的覆盖范围
    # 字段：weights_mapper ①；spec_decode 决定 ② ⑤ ⑧；skip_weight_prefixes ④；mtp_remap ⑤；weights_mapping ⑥；
    #   num_hidden_layers ⑧；n_routed_experts / fuse_shared_expert / disable_fused_shared_loading ⑨

    weights_mapper: WeightsMapper | None = None
    weights_mapping: Mapping[str, str] = field(default_factory=dict)
    skip_weight_prefixes: Sequence[str] = ()
    mtp_remap: Callable[[str], str | None] | None = None
    spec_decode: bool = False
    num_hidden_layers: int = 0
    n_routed_experts: int | None = None
    fuse_shared_expert: Callable[[str, str], bool] = lambda *_: False
    disable_fused_shared_loading: bool = False

    # 按 ①~⑨ 执行；调用方：loading_core 的 _wanted（每个 checkpoint 名一次，结果缓存）
    def rewrite(self, name: str) -> str | None:
        name = self._apply_mapper(name)
        if name is None:
            return None
        # Draft models remap ckpt-side `mtp.*` entries into params whose names
        # do not themselves contain `mtp` (Qwen3.5 MTP rewrites `mtp.*` ->
        # `model.*`), so gate only on `spec_decode`: dropping every `mtp` name
        # first would discard the whole drafter checkpoint before the
        # model-specific remap ever runs.
        # 【译】草稿模型会把 checkpoint 侧的 `mtp.*` 条目重映射到名字本身不含 `mtp` 的参数上（Qwen3.5 MTP 把 `mtp.*` 改写成
        #   `model.*`），所以这里只按 `spec_decode` 把关：如果先把所有含 `mtp` 的名字都丢掉，模型自己的重映射还没机会运行，
        #   整个草稿 checkpoint 就被扔光了
        # ② 加载目标模型时丢掉 mtp 权重
        if "mtp" in name and not self.spec_decode:
            return None
        # ③ 不加载 checkpoint 里的 KV cache 量化 scale（kv_scale）和 RoPE 频率表（inv_freq，rotary embedding 启动时自己算）
        if name.endswith("kv_scale") or "inv_freq" in name:
            return None
        # Model-declared prefixes to ignore, e.g. the vision encoder of a
        # multimodal checkpoint being served text-only.
        # 【译】模型声明要忽略的前缀，例如一个只按文本服务的多模态 checkpoint 里的视觉编码器
        # ④
        if any(name.startswith(p) for p in self.skip_weight_prefixes):
            return None
        # ⑤ 草稿模型自己的改名 / 过滤
        if self.spec_decode and self.mtp_remap is not None:
            name = self.mtp_remap(name)
            if name is None:
                return None
        # ⑥ 子串替换：str.replace 不带次数，所有出现都替换——这正是文件头说它"粗糙"、要配 WeightsMapper 的原因
        #   （_report_coverage 的注释里举过 `attn_norm.weight` 被误改成 `attn_model.norm.weight` 的例子）
        for part, replacement in self.weights_mapping.items():
            if part in name:
                name = name.replace(part, replacement)
        # ⑦ ⑧ ⑨
        if "weight_scale_inv" in name:
            name = name.replace("weight_scale_inv", "weight_scale")
        if self._is_past_last_layer(name):
            return None
        return self._maybe_fuse_shared_expert(name)

    # ── steps ─────────────────────────────────────────────────────────────

    def _apply_mapper(self, name: str) -> str | None:
        if self.weights_mapper is None:
            return name
        return self.weights_mapper._map_name(name)

    # ⑧：加载草稿时不截断（草稿的层号可能本来就 ≥ num_hidden_layers）；num_hidden_layers 为 0（未知）时也不截断
    #   re.search(r"model\.layers\.(\d+)\.", name)：在名字任意位置找 "model.layers.<数字>."，group(1) 取出数字
    def _is_past_last_layer(self, name: str) -> bool:
        if not self.num_hidden_layers or self.spec_decode:
            return False
        match = re.search(r"model\.layers\.(\d+)\.", name)
        return match is not None and int(match.group(1)) >= self.num_hidden_layers

    def _maybe_fuse_shared_expert(self, name: str) -> str:
        matching_name = have_shared_expert(name)
        if matching_name is None:
            return name
        # Some models keep shared experts unfused (V4-Pro with FP4 routed vs
        # FP8 shared, or DP + mori all2all); their shared weights must land on
        # the standalone Expert module instead of the fused slot.
        # 【译】有些模型让共享专家保持不融合（V4-Pro 的路由专家是 FP4、共享专家是 FP8，或者 DP + mori all2all）；它们的共享
        #   权重必须落到独立的 Expert 模块上，而不是融合槽位
        if self.disable_fused_shared_loading:
            return name
        # 量化规格 / 并行配置不允许融合（loader.py 的 _fuse_shared_expert → topK.py）→ 名字不变，共享专家按独立模块加载
        if not self.fuse_shared_expert(*shared_expert_prefixes(name, matching_name)):
            return name
        if self.n_routed_experts is None:
            raise AttributeError(
                "Cannot remap shared expert weights without n_routed_experts, "
                "num_local_experts, or num_experts on the model config."
            )
        # Keep the module-naming prefix (mlp. / ffn.) so the rewritten name
        # matches this model's routed-expert parameter naming. The id follows
        # the *logical* routed experts, matching
        # `FusedMoE.make_expert_params_mapping`; FusedMoE translates it to a
        # physical slot, which under EPLB also has to clear the replicas.
        # 【译】保留模块命名前缀（mlp. / ffn.），让改写后的名字和这个模型路由专家参数的命名一致。编号跟随*逻辑*路由专家，
        #   与 `FusedMoE.make_expert_params_mapping` 一致；FusedMoE 会把它翻译成物理槽位，在 EPLB 下还要清掉各个副本
        # 例："model.layers.3.mlp.shared_experts.gate_proj.weight"，n_routed = 256 → "model.layers.3.mlp.experts.256.gate_proj.weight"
        module_prefix = matching_name.split("shared_expert", 1)[0]
        return name.replace(
            matching_name, f"{module_prefix}experts.{self.n_routed_experts}."
        )


# merged 路径用（weight_dispatch._dispatch_merged_or_generic）：名字里恰好有两个纯数字段（层号、专家号）时，
#   去掉最后一个数字段（专家号）得到合并参数名
def extract_expert_target_and_id(name: str) -> tuple[str, int] | None:
    """Extract fused parameter name and expert id from expert checkpoint name.
    like 'model.layers.10.mlp.experts.100.w2_bias' -> model.layers.10.mlp.experts.w2_bias and 100
    """
    # 【译】从专家的 checkpoint 名里提取融合参数名和专家编号，
    #   例如 'model.layers.10.mlp.experts.100.w2_bias' -> model.layers.10.mlp.experts.w2_bias 和 100
    # parts[::-1].index(token)：在反转的列表里找第一次出现 = 原列表里最后一次出现，再换算回正向下标后 pop 掉
    #   （层号和专家号相同时，例如 layers.10.experts.10，要删的是后面那个）
    if "experts" not in name:
        return None
    parts = name.split(".")
    ids = [s for s in parts if s.isdigit()]
    if len(ids) != 2:
        return None
    expert_id = int(ids[-1])
    expert_token = str(expert_id)
    if expert_token not in parts:
        return None
    fused_parts = parts.copy()
    fused_parts.pop(len(parts) - 1 - parts[::-1].index(expert_token))
    return ".".join(fused_parts), expert_id
