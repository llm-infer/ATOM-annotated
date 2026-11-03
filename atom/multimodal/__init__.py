# SPDX-License-Identifier: MIT
# 【导读】atom.multimodal包：与模型无关的多模态输入层（2b），在HF processor和token展开之前处理媒体
#   inputs.py 类型定义 / parse.py 规范化 / media/ 读取与解码 / processing.py 输入准备入口 / registry.py 按模型架构分派
# 这里只导出类型、MediaLoader和normalize_multimodal_data；服务层的主要入口是processing.prepare_multimodal_inputs
# 前置（读本包只需记住两点；带图的完整科普见examples/multimodal_inference.py文件头【科普】图片占位token与视觉编码器）：
#   ① 图片没有token id。进语言模型的是视觉编码器（ViT）输出的一串hidden_size维向量：prompt里先为每张图放一段相同的占位token，
#     forward时把这些位置的词嵌入换成视觉向量。例（Qwen3.5）：448×448的图切成16×16的patch得784个，相邻2×2合并后剩196个向量，
#     所以这张图在prompt里占196个占位token（image_token_id=248056），也占196个token的KV cache
#   ② HF processor是模型仓库附带的预处理器：把图片切patch、归一化成pixel_values张量，并把占位token展开成正确个数
#     （有的processor每张图只放1个，由processing.expand_media_placeholders展开）。本包负责processor之前的媒体规范化、
#     下载与解码，以及按模型架构选出调用processor的输入构建函数

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
