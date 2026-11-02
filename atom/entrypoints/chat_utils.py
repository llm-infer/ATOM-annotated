# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Adapt chat content parts to a conversation and separate media data."""

# 【译】把chat消息的内容片段整理成对话，并把媒体数据分离出来。
#
# 本文件（第2批2c，B级）：OpenAI chat消息里的多模态内容片段 → (对话, 媒体) 两部分
#   调用方：api_server._has_multimodal_content / _prepare_multimodal_inputs（只在请求含媒体时）
#   输出交给 multimodal/processing.prepare_multimodal_inputs（【分支】第2批2b）去下载/解码媒体、渲染模板
# 例：
#   输入 messages = [{"role": "user", "content": [
#            {"type": "text", "text": "图里是什么？"},
#            {"type": "image_url", "image_url": {"url": "https://.../cat.png", "detail": "high"}}]}]   ← 图片片段
#   输出 conversation = [{"role": "user", "content": [
#            {"type": "text", "text": "图里是什么？"}, {"detail": "high", "type": "image"}]}]
#        media = {"image": ["https://.../cat.png"]}
#   即：媒体本身被抽到media里，对话中原位置只留一个带模态类型的占位片段
import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Any

from atom.multimodal import EncodedMedia, Modality, MultiModalDataItems

# 内容片段的type → 模态。image和image_url都算image，以此类推；input_audio是OpenAI的base64内联音频
_PART_MODALITIES: dict[str, Modality] = {
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
    "audio": "audio",
    "audio_url": "audio",
    "input_audio": "audio",
}


# 是否有任何消息的content是片段列表、且其中含媒体类型的片段；content是纯字符串的消息直接跳过
def has_multimodal_content(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(part, Mapping)
        and isinstance(part.get("type"), str)
        and part["type"] in _PART_MODALITIES
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


# 解析一个媒体片段，返回(模态, 数据, 选项)：
#   *_url类型：数据取url字符串，url之外的字段（如detail）并入选项
#   input_audio：把base64解码成EncodedMedia(字节, 格式)
#   其他（image / video / audio）：数据就是该类型键下的值
# 只做格式检查和base64解码，不做网络或文件I/O；格式不对时抛ValueError（接口返回400）
def _parse_media_part(part: Mapping[str, Any]) -> tuple[Modality, Any, dict[str, Any]]:
    part_type = part["type"]
    modality = _PART_MODALITIES[part_type]
    data = part.get(part_type)
    options = {k: v for k, v in part.items() if k not in ("type", part_type)}
    if part_type.endswith("_url"):
        if not isinstance(data, Mapping) or not isinstance(data.get("url"), str):
            raise ValueError(f"{part_type} must include {part_type}.url")
        options.update({k: v for k, v in data.items() if k != "url"})
        data = data["url"]
    elif part_type == "input_audio":
        if (
            not isinstance(data, Mapping)
            or not isinstance(data.get("data"), str)
            or not isinstance(data.get("format"), str)
            or not data["format"]
        ):
            raise ValueError("input_audio must include base64 data and format")
        try:
            data = EncodedMedia(
                base64.b64decode(data["data"], validate=True), data["format"]
            )
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 data for input_audio") from exc
    if data is None:
        raise ValueError(f"{part_type} must include media data")
    return modality, data, options


# 遍历所有消息：文本片段原样复制，媒体片段换成{**选项, "type": 模态}的占位，媒体数据按出现顺序追加进media[模态]
def parse_chat_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], MultiModalDataItems]:
    """Extract media in encounter order, leaving typed markers in the chat.

    The nth marker of a modality refers to the nth item of that modality across
    the whole conversation. Metadata stays on messages/content parts. This
    adapter performs no network/file I/O or media decoding.
    """
    # 【译】按出现顺序取出媒体，在对话里留下带类型的占位标记。
    #
    #   某个模态的第n个标记，对应整段对话中该模态的第n个媒体项。元数据保留在消息/内容片段上。
    #   这个适配器不做任何网络/文件I/O，也不解码媒体。
    conversation = []
    media: MultiModalDataItems = {}
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Messages must be mappings")  # noqa: TRY004
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must have a non-empty role")
        content = message.get("content")
        # content是片段列表：逐个检查并转换；content是字符串或None时原样保留，其他类型报错
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, Mapping):
                    raise ValueError(  # noqa: TRY004
                        "Message content parts must be mappings"
                    )
                part_type = part.get("type")
                if part_type == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError("text content part must include text")
                    parts.append(dict(part))
                elif isinstance(part_type, str) and part_type in _PART_MODALITIES:
                    modality, data, options = _parse_media_part(part)
                    media.setdefault(modality, []).append(data)
                    parts.append({**options, "type": modality})
                else:
                    raise ValueError(f"Unsupported content part type: {part_type!r}")
            content = parts
        elif content is not None and not isinstance(content, str):
            raise ValueError("Message content must be text, a list of parts or None")
        conversation.append({**message, "content": content})
    return conversation, media
