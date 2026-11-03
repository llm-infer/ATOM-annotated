# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture lookup for existing native input builders and model hooks.

Implementations are imported only when their architecture is selected.
"""

# 【译】按架构查找现有的原生输入构建函数和模型钩子。
#
#   只有选中某个架构时才导入它的实现。
#
# 【导读】多模态的"按模型架构分派"表：hf_config.architectures[0] -> 模块路径字符串，用到时才import_module
#   （用字符串而不是直接import，导入本模块时不会把各个多模态模型都加载进来）
#   _MULTIMODAL_ARCH_TO_MODEL：提供M-RoPE位置计算的模型类（Qwen3.5 / Qwen3.5-MoE / Qwen4-exp）
#     （M-RoPE：图片token的位置用(t, h, w)三个编号而不是一维序号，带例子的讲解见model_config/qwen3_5.py文件头【科普】②）
#   _MULTIMODAL_ARCH_TO_INPUT_BUILDER：自定义输入构建函数（DeepSeek-V4.1、Kimi-K3）；其他架构默认用qwen3_5_vl.build_qwen_vl_inputs
#   _MULTIMODAL_ARCH_TO_PROCESSOR：checkpoint自带processor的架构（DeepSeek-V4.1）；不在表中时调用方回退到AutoProcessor
# 调用方：api_server.py（get_native_multimodal_processor）、processing.py（get_multimodal_input_builder）、
#   llm_engine.py（get_mrope_input_positions，预处理请求时计算M-RoPE位置，本批）
# 【分支】这些模型的视觉部分（Qwen3.5-VL、Kimi-K3-VL、DeepSeek-V4.1、Qwen4-exp）见第8批

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from atom.config import Config

_MULTIMODAL_ARCH_TO_MODEL: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
    "Qwen4ExpForConditionalGeneration": (
        "atom.models.qwen4_exp.Qwen4ExpForConditionalGeneration"
    ),
}

_MULTIMODAL_ARCH_TO_INPUT_BUILDER: dict[str, str] = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.image_processing.build_inputs"
    ),
    "KimiK3ForConditionalGeneration": "atom.models.kimi_k3_vl.build_kimi_k3_inputs",
}

# Architectures whose checkpoint ships its own processor rather than one
# `AutoProcessor` can build. Selected before that fallback, so a model listed
# here never reaches it.
# 【译】checkpoint自带processor（而不是`AutoProcessor`能构建的那种）的架构。它在那个回退之前被选中，
#   所以列在这里的模型永远不会走到回退。
_MULTIMODAL_ARCH_TO_PROCESSOR: dict[str, str] = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.image_processing.DeepseekV41ImageProcessor"
    ),
}


# "包.模块.名字" -> 导入模块并取出对象
def _resolve(qualname: str) -> Any:
    module, _, name = qualname.rpartition(".")
    return getattr(import_module(module), name)


# 调用方：api_server._get_multimodal_processor（首个多模态请求时懒加载一次）
#   有注册时以(atom_config, tokenizer, encoder)构造checkpoint自带的processor；否则返回None
def get_native_multimodal_processor(atom_config, tokenizer, encoder):
    """The checkpoint's own processor, or None to fall back to `AutoProcessor`."""
    # 【译】checkpoint自己的processor；返回None则回退到`AutoProcessor`。
    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    factory = (
        _MULTIMODAL_ARCH_TO_PROCESSOR.get(architectures[0]) if architectures else None
    )
    return (
        None if factory is None else _resolve(factory)(atom_config, tokenizer, encoder)
    )


# 调用方：processing.prepare_multimodal_inputs；返回的构建函数签名为
#   (atom_config, processor, prompt, images, chat_template_kwargs, tools=None)，返回(input_ids, multimodal_data)
def get_multimodal_input_builder(
    atom_config: Config, *, is_text_prompt: bool = False
) -> Callable:
    """Select an existing builder, retaining the Qwen convention as fallback.

    The registered special builders currently require a chat conversation.
    Validate this before loading media.
    """
    # 【译】选择一个现有的构建函数，回退时沿用Qwen的约定。
    #
    #   目前注册的特殊构建函数都要求chat对话形式。要在加载媒体之前校验这一点。
    hf_config = getattr(atom_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    builder = (
        _MULTIMODAL_ARCH_TO_INPUT_BUILDER.get(architectures[0])
        if architectures
        else None
    )
    if builder is not None and is_text_prompt:
        raise ValueError(
            "This model's multimodal processor requires a chat conversation"
        )
    return _resolve(builder or "atom.models.qwen3_5_vl.build_qwen_vl_inputs")


# 调用方：LLMEngine预处理多模态请求时（llm_engine.py，本批）；返回(mrope_positions, mrope_position_delta)，存进Sequence
#   M-RoPE：把位置编码拆成时间/高/宽三个分量，图像token按二维网格取位置（【分支】见第8批Qwen3.5）
#   架构不在表中、或模型类没有get_mrope_input_positions时返回(None, 0)，即使用普通的一维RoPE位置
def get_mrope_input_positions(
    atom_config: Config,
    input_tokens: list[int],
    multimodal_data: dict,
) -> tuple[np.ndarray | None, int]:
    """Return request-level MRoPE positions via the model's MRoPE interface."""
    # 【译】通过模型的MRoPE接口返回请求级别的MRoPE位置。

    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    if not architectures:
        return None, 0

    model_qualname = _MULTIMODAL_ARCH_TO_MODEL.get(architectures[0])
    if model_qualname is None:
        return None, 0

    model_cls = _resolve(model_qualname)
    mrope_getter = getattr(model_cls, "get_mrope_input_positions", None)
    if mrope_getter is None:
        return None, 0

    return mrope_getter(atom_config, input_tokens, multimodal_data)
