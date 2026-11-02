# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Normalize media grouped by modality, independently of chat protocols."""

# 【译】按模态分组地规范化媒体，与具体的chat协议无关。
#
# 【导读】把松散的MultiModalDataDict规范化为MultiModalDataItems：校验模态名、单项转列表、bytes包成EncodedMedia、
#   音频二元组转AudioData，并检查每一项的类型
# 调用方：processing.prepare_multimodal_inputs（每个多模态请求一次）；这一步不加载媒体、不复制数组

from collections.abc import Mapping
from os import PathLike, fspath
from typing import cast

from .inputs import (
    AudioData,
    EncodedMedia,
    MediaData,
    Modality,
    MultiModalDataDict,
    MultiModalDataItems,
    VideoData,
)


# 规范化单个媒体项：PathLike -> str；bytes -> EncodedMedia；音频的(waveform, sr)元组 -> AudioData；然后校验类型
#   字符串只检查非空（是URL还是本地路径留给MediaConnector判断）；PIL只在图像分支里才导入
def _normalize_item(modality: Modality, data: MediaData) -> MediaData:
    if isinstance(data, PathLike):
        data = fspath(data)
    if isinstance(data, bytes):
        data = EncodedMedia(data)
    if modality == "audio" and isinstance(data, tuple) and len(data) == 2:
        data = AudioData(*data)

    if isinstance(data, str):
        if not data.strip():
            raise ValueError(f"{modality} URL/path must not be empty")
    elif not isinstance(data, EncodedMedia):
        if modality == "image":
            from PIL import Image

            valid = isinstance(data, Image.Image)
        else:
            valid = isinstance(data, AudioData if modality == "audio" else VideoData)
        if not valid:
            raise ValueError(
                f"Invalid {modality} input: expected a URL/path, encoded bytes or decoded {modality} data"
            )
    return data


# 输入：{"image": x 或 [x, ...], "audio": ..., "video": ...}；输出：{模态: [规范化后的项, ...]}，值为None或空列表的模态被去掉
#   未知模态名、类型不对的项都抛ValueError
def normalize_multimodal_data(data: MultiModalDataDict) -> MultiModalDataItems:
    """Normalize single items and lists without loading media or copying arrays.

    Empty modalities are omitted. Audio tuples are single items; lists contain
    separate items. Sampling rates and video timing must be supplied explicitly.
    """
    # 【译】规范化单个项和列表，不加载媒体，也不复制数组。
    #
    #   空的模态被省略。音频元组算作单个项；列表里是各自独立的项。采样率和视频时间信息必须显式提供。
    if not isinstance(data, Mapping):
        raise ValueError(  # noqa: TRY004
            "Multimodal data must be a mapping keyed by modality"
        )
    normalized: MultiModalDataItems = {}
    for name, value in data.items():
        if name not in ("image", "audio", "video"):
            raise ValueError(f"Unsupported modality: {name!r}")
        modality = cast(Modality, name)
        if value is None:
            continue
        # 只有list才视为多个项：所以音频的(waveform, sr)元组算一个项
        items = value if isinstance(value, list) else [value]
        if items:
            normalized[modality] = [_normalize_item(modality, item) for item in items]
    return normalized
