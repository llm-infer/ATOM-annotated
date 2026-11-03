# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request adaptation for the checkpoint's standalone V4.1 encoder.

Prompt construction stays in the official encoder. These hooks only translate
ATOM's common controls and attach top-level tool schemas to the system turn.
"""

# 【译】为checkpoint自带的独立V4.1编码器做请求适配。
#
#   prompt的构造仍由官方编码器完成。这些钩子只负责翻译ATOM的通用控制参数，并把顶层的工具schema挂到system轮次上。
#
# 【导读】DeepSeek-V4.1不用Jinja chat template（chat template是什么见chat_encoders.py文件头【科普】），
#   而是随checkpoint附带一个Python编码器（encoding_dsv41）；本文件是这个编码器的两个前处理钩子
# 调用方：chat_encoder_adapters.py的_PREPARERS["encoding_dsv41"]（编码器模块名就是encoding_dsv41时直接选用；
#   模块名是通用的"encoding"时，只有hf config的model_type为deepseek_v41才映射过来，见该文件的_MODEL_ENCODERS），
#   chat_encoders.py渲染prompt前调用prepare_messages，编码器适配器调用prepare_kwargs
# 与主干的关系：服务层prompt渲染的一个模型特化分支；【分支】DeepSeek-V4.1模型见第8批

from typing import Any


# 复制messages；有tools时挂到第一条system消息上（没有system消息就在开头插入一条只带tools的system消息）
def prepare_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    prepared = [dict(message) for message in messages]
    if tools:
        if prepared and prepared[0].get("role") == "system":
            prepared[0]["tools"] = tools
        else:
            prepared.insert(0, {"role": "system", "tools": tools})
    return prepared


# 把通用的thinking_effort改名为V4.1编码器认的reasoning_effort，并校验reasoning_effort / thinking_mode的取值
def prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Explicit API effort is merged last under the common thinking_effort key;
    # it must also override a native reasoning_effort in template defaults.
    # 【译】API显式给出的effort最后以通用的thinking_effort键合并进来；它也必须覆盖模板默认值里原生的reasoning_effort。
    if "thinking_effort" in kwargs:
        kwargs["reasoning_effort"] = kwargs.pop("thinking_effort")
    effort = kwargs.get("reasoning_effort")
    if effort is not None and not (
        (type(effort) is int and 1 <= effort <= 100)
        or (isinstance(effort, str) and effort in ("low", "high", "max"))
    ):
        raise ValueError(
            "DeepSeek-V4.1 reasoning_effort must be an integer in [1, 100] "
            "or 'low', 'high', 'max'"
        )
    if kwargs.get("thinking_mode", "thinking") not in ("thinking", "chat"):
        raise ValueError("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'")
    if kwargs.get("return_multi_modal_data"):
        raise ValueError("return_multi_modal_data is not a text chat-template control")
    return kwargs
