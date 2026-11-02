# SPDX-License-Identifier: MIT
# 【导读】atom.multimodal包：与模型无关的多模态输入层（2b），在HF processor和token展开之前处理媒体
#   inputs.py 类型定义 / parse.py 规范化 / media/ 读取与解码 / processing.py 输入准备入口 / registry.py 按模型架构分派
# 这里只导出类型、MediaLoader和normalize_multimodal_data；服务层的主要入口是processing.prepare_multimodal_inputs

from .inputs import (
    AudioData,
    AudioItem,
    EncodedMedia,
    ImageItem,
    MediaData,
    MediaSource,
    Modality,
    MultiModalDataDict,
    MultiModalDataItems,
    VideoData,
    VideoItem,
)
from .media import MediaLoader
from .parse import normalize_multimodal_data

__all__ = [
    "AudioData",
    "AudioItem",
    "EncodedMedia",
    "ImageItem",
    "MediaData",
    "MediaLoader",
    "MediaSource",
    "Modality",
    "MultiModalDataDict",
    "MultiModalDataItems",
    "VideoData",
    "VideoItem",
    "normalize_multimodal_data",
]
