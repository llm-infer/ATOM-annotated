# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal input preparation and shared media placeholder operations."""

# 【译】多模态输入准备，以及共享的媒体占位符操作。
#
# 【导读】多模态请求的输入准备入口：把"对话 + 媒体"变成(input_ids, multimodal_data)
# 调用方：api_server._prepare_multimodal_inputs（在线服务；放进线程池执行，避免下载图片和processor计算阻塞事件循环）、
#   examples/multimodal_inference.py（离线示例，第1批）；expand_media_placeholders由models/kimi_k3_vl.py调用
# 流程：normalize_multimodal_data规范化 -> 只允许image -> registry按架构选输入构建函数 -> MediaLoader加载图片
#   -> 把对话中第n个图片标记绑定到第n张图 -> 构建函数（各模型的processor逻辑）输出token id和多模态数据
# 与主干的关系：产出的multimodal_data随请求进入LLMEngine -> Sequence.multimodal_data -> 调度器 -> ModelRunner
#   （带embedding_spans的数据走model_engine/multimodal_runtime.py的span路径）
# 【分支】各模型的输入构建函数（qwen3_5_vl.build_qwen_vl_inputs、kimi_k3_vl.build_kimi_k3_inputs、
#   deepseek_v41/image_processing.build_inputs）见第8批

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.multimodal import MediaLoader, MultiModalDataDict, normalize_multimodal_data
from atom.multimodal.media import load_multimodal_data
from atom.multimodal.registry import get_multimodal_input_builder

if TYPE_CHECKING:
    from atom.config import Config


# 把每个占位token展开成该媒体项对应个数的token
#   例：input_ids=[1, P, 2]，tokens_per_media=[3]，placeholder_token_id=P -> [1, P, P, P, 2]
#   占位符个数与媒体项数不一致时报错；调用方：models/kimi_k3_vl.py的build_kimi_k3_inputs（【分支】Kimi-K3见第8批）
def expand_media_placeholders(
    input_ids: Sequence[int],
    tokens_per_media: Sequence[int],
    placeholder_token_id: int,
) -> list[int]:
    """Repeat each single placeholder token into its media item's token run.

    Processors that leave the expansion to the model (Kimi-K3) emit exactly one
    placeholder per image, but ATOM needs one token per image embedding: the
    scheduler allocates KV blocks and positions from the token count, and the
    prefill scatter matches embeddings against placeholder positions.
    """
    # 【译】把每个单独的占位token重复成对应媒体项的那一串token。
    #
    #   有些processor把展开留给模型来做（Kimi-K3），每张图只输出恰好一个占位符；但ATOM需要每个图像embedding对应一个token：
    #   调度器按token数分配KV block和位置，prefill时的散射也是按占位符位置去匹配embedding的。
    num_placeholders = sum(1 for token in input_ids if token == placeholder_token_id)
    if num_placeholders != len(tokens_per_media):
        raise ValueError(
            f"prompt has {num_placeholders} media placeholder tokens but "
            f"{len(tokens_per_media)} media items were preprocessed"
        )

    expanded: list[int] = []
    media_index = 0
    for token in input_ids:
        if token == placeholder_token_id:
            expanded.extend([token] * tokens_per_media[media_index])
            media_index += 1
        else:
            expanded.append(token)
    return expanded


# 输入：规范化后的对话（content为text/image part列表）和已加载的图片列表
#   输出：新的消息列表，每个image part多了"image"字段，指向对应的PIL图像；图片标记数与图片数不一致时报错
def _bind_images_to_messages(
    conversation: list[dict[str, Any]], images: list
) -> list[dict[str, Any]]:
    """Bind the nth image marker to the nth image across all chat turns."""
    # 【译】把所有对话轮次中的第n个图片标记绑定到第n张图。
    messages = []
    image_index = 0
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "image":
                    if image_index >= len(images):
                        raise ValueError(
                            "Conversation has more image markers than images"
                        )
                    part = {**part, "image": images[image_index]}
                    image_index += 1
                elif part["type"] != "text":
                    raise ValueError(
                        "Expected a normalized conversation with image/text parts"
                    )
                parts.append(dict(part))
            content = parts
        # Retain the native processors' handling of assistant/tool messages.
        # 【译】保留原生processor对assistant/tool消息的处理方式。
        # 解释：content为None时换成空字符串，其余字段原样保留
        messages.append({**message, "content": "" if content is None else content})
    if image_index != len(images):
        raise ValueError(
            f"Conversation has {image_index} image markers but {len(images)} images"
        )
    return messages


# 输入：atom_config、processor（AutoProcessor或checkpoint自带的processor）、prompt（对话列表或已渲染的文本）、
#   multi_modal_data、chat_template_kwargs、tools；可选media_loader（默认MediaLoader()，http超时30秒）
# 输出：(input_ids, multimodal_data)
def prepare_multimodal_inputs(
    atom_config: Config,
    processor: Any,
    prompt: str | list[dict[str, Any]],
    multi_modal_data: MultiModalDataDict,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    media_loader: MediaLoader | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Prepare a prompt and separate media using the existing native builders.

    Chat callers supply a conversation with image markers, in the same order as
    multi_modal_data['image']. Qwen-style processors also accept a rendered text
    prompt, with one image placeholder per item and no chat template applied.
    Video/audio are rejected before any media is loaded.
    """
    # 【译】用现有的原生构建函数准备prompt并分离出媒体。
    #
    #   chat调用方传入带图片标记的对话，标记顺序与multi_modal_data['image']一致。Qwen风格的processor也接受已渲染的文本prompt，
    #   其中每个项一个图片占位符，且不再套用chat template。视频/音频在加载任何媒体之前就会被拒绝。
    media_items = normalize_multimodal_data(multi_modal_data)
    if not media_items:
        raise ValueError("Multimodal request did not contain any media")
    unsupported = media_items.keys() - {"image"}
    if unsupported:
        raise ValueError(
            f"Native multimodal inference does not support {', '.join(sorted(unsupported))} "
            "inputs yet; supported modalities: image"
        )
    # 先选构建函数再加载媒体：特殊构建函数不接受文本prompt，可以在下载图片之前就报错
    builder = get_multimodal_input_builder(
        atom_config, is_text_prompt=isinstance(prompt, str)
    )
    media = load_multimodal_data(media_items, media_loader or MediaLoader())
    images = media["image"]
    # 对话形式：把图片对象绑定进消息，由构建函数套用chat template并调用processor
    if not isinstance(prompt, str):
        prompt = _bind_images_to_messages(prompt, images)
    return builder(
        atom_config,
        processor,
        prompt,
        images,
        dict(chat_template_kwargs or {}),
        tools=tools,
    )
