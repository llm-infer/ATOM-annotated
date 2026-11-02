# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-independent media inputs, before HF processing or token expansion."""

# 【译】与模型无关的媒体输入类型，处于HF processor处理和token展开之前。
#
# 【导读】多模态输入的数据类型定义
# 一个多模态chat请求在服务层的流程（本批2b）：
#   chat_utils.parse_chat_messages：从OpenAI消息的content parts中拆出媒体，得到MultiModalDataItems（按模态分组的列表）
#   -> processing.prepare_multimodal_inputs：parse.normalize_multimodal_data规范化 -> media.MediaLoader读取并解码（得到PIL图像）
#   -> registry按模型架构选出输入构建函数（Qwen3.5-VL / Kimi-K3 / DeepSeek-V4.1），由它跑processor、展开占位token
#   -> 得到(input_ids, multimodal_data)，交给LLMEngine（预处理时计算M-RoPE位置）构造Sequence
# 本文件只定义类型：媒体可以是URL/路径字符串、原始字节（EncodedMedia）或已解码的对象（PIL图像、AudioData、VideoData）
# v0.1.8的原生引擎只支持image；audio / video的类型已经定义，但prepare_multimodal_inputs会在加载前拒绝它们

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# 支持的模态名
Modality = Literal["image", "video", "audio"]


# 未解码的媒体字节（例如从base64 data URL解出的内容）；format是可选的格式提示；构造时校验必须是非空bytes
@dataclass(frozen=True)
class EncodedMedia:
    data: bytes
    format: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Encoded media must contain non-empty bytes")


# 已解码的音频：waveform为一维实数数组 [采样点数]，sampling_rate为正整数；__post_init__做校验
@dataclass(frozen=True)
class AudioData:
    """Decoded mono waveform and its sampling rate; no implicit resampling."""

    # 【译】已解码的单声道波形及其采样率；不做隐式重采样。

    waveform: np.ndarray
    sampling_rate: int

    def __post_init__(self):
        if (
            not isinstance(self.waveform, np.ndarray)
            or self.waveform.ndim != 1
            or not self.waveform.size
            or not np.issubdtype(self.waveform.dtype, np.number)
            or np.iscomplexobj(self.waveform)
        ):
            raise ValueError(
                "AudioData.waveform must be a non-empty 1D real numeric array"
            )
        if (
            isinstance(self.sampling_rate, bool)
            or not isinstance(self.sampling_rate, int)
            or self.sampling_rate <= 0
        ):
            raise ValueError("AudioData.sampling_rate must be a positive integer")


# 已解码的视频：frames为uint8数组 [T帧, H, W, 3]；fps与timestamps至少给一个；timestamps须逐帧一个、有限非负、单调不减
@dataclass(frozen=True)
class VideoData:
    """Decoded RGB frames in THWC order, with fps or per-frame timestamps."""

    # 【译】已解码的RGB帧，按THWC顺序，附带fps或逐帧时间戳。

    frames: np.ndarray
    fps: float | None = None
    timestamps: tuple[float, ...] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.frames, np.ndarray)
            or self.frames.ndim != 4
            or not self.frames.size
            or self.frames.shape[-1] != 3
            or self.frames.dtype != np.uint8
        ):
            raise ValueError(
                "VideoData.frames must be a non-empty uint8 RGB THWC array"
            )
        if self.fps is None and self.timestamps is None:
            raise ValueError("VideoData requires fps or timestamps")
        if self.fps is not None and (
            isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0
        ):
            raise ValueError("VideoData.fps must be finite and positive")
        if self.timestamps is not None:
            if len(self.timestamps) != len(self.frames) or any(
                not math.isfinite(t) or t < 0 for t in self.timestamps
            ):
                raise ValueError(
                    "VideoData requires one finite non-negative timestamp per frame"
                )
            if any(a > b for a, b in zip(self.timestamps, self.timestamps[1:])):
                raise ValueError("VideoData.timestamps must be non-decreasing")


# 各模态的单个输入项可以是什么：MediaSource = URL/路径字符串、PathLike、原始bytes或EncodedMedia；
#   图像还可以是PIL.Image，音频还可以是AudioData或(waveform, sampling_rate)元组，视频还可以是VideoData
MediaSource: TypeAlias = str | PathLike[str] | bytes | EncodedMedia
ImageItem: TypeAlias = Union[MediaSource, "Image.Image"]
AudioItem: TypeAlias = MediaSource | AudioData | tuple[np.ndarray, int]
VideoItem: TypeAlias = MediaSource | VideoData
MediaData: TypeAlias = ImageItem | AudioItem | VideoItem

# A modality may contain one item or an ordered list of items. Conversation
# structure and protocol-specific content parts belong to the entrypoint.
# 【译】一个模态可以包含单个项，也可以是有序的项列表。对话结构和协议特有的content part属于入口层（entrypoint）的职责。
# 解释：MultiModalDataDict是调用方给的松散形式，如{"image": url}或{"image": [img1, img2]}；
#   MultiModalDataItems是规范化之后的形式，每个模态一定是列表，如{"image": [EncodedMedia(...), "https://..."]}
MultiModalDataDict: TypeAlias = Mapping[str, MediaData | list[MediaData] | None]
MultiModalDataItems: TypeAlias = dict[Modality, list[MediaData]]
