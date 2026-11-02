# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Media loading, independent of model processors and serving protocols."""

# 【译】媒体加载，与模型processor和服务协议无关。
#
# 【导读】媒体加载器：把规范化后的媒体项变成解码后的对象（图片 -> RGB模式的PIL.Image）
# 调用方：processing.prepare_multimodal_inputs（load_multimodal_data，默认用MediaLoader()）
# 读取字节由connector.MediaConnector负责（data URL / http(s) / file:// / 本地路径），图片解码由image.load_image负责

from __future__ import annotations

from ..inputs import (
    AudioData,
    AudioItem,
    ImageItem,
    MediaData,
    Modality,
    MultiModalDataItems,
    VideoData,
    VideoItem,
)
from .connector import MediaConnector
from .image import load_image


# 按模态分派的加载器：只有图片真正做解码，音频/视频只接受已解码的对象（可以子类化，自己实现load_audio / load_video）
#   timeout是http(s)下载的超时秒数
class MediaLoader:
    """Decode images and accept decoded audio/video with timing metadata.

    Encoded audio/video require a codec implementation of load_audio/load_video.
    The native engine rejects unsupported modalities before loading media.
    """

    # 【译】解码图片，并接受带时间元数据的已解码音频/视频。
    #
    #   编码的音频/视频需要一个实现了load_audio/load_video的编解码器。原生引擎会在加载媒体之前拒绝不支持的模态。

    def __init__(self, timeout: float = 30):
        self.connector = MediaConnector(timeout)

    # 图片：必要时先读字节，再解码成RGB模式的PIL图像
    def load_image(self, data: ImageItem):
        return load_image(data, self.connector)

    # 音频：只接受AudioData或(waveform, sr)元组，编码的音频直接报错
    def load_audio(self, data: AudioItem) -> AudioData:
        if isinstance(data, tuple):
            data = AudioData(*data)
        if isinstance(data, AudioData):
            return data
        raise ValueError(
            "Audio decoding requires a media loader; pass decoded AudioData "
            "or implement MediaLoader.load_audio"
        )

    # 视频：只接受VideoData，编码的视频直接报错
    def load_video(self, data: VideoItem) -> VideoData:
        if isinstance(data, VideoData):
            return data
        raise ValueError(
            "Video decoding requires a media loader; pass decoded VideoData "
            "or implement MediaLoader.load_video"
        )

    # 按模态名分派到对应的load_*
    def load(self, modality: Modality, data: MediaData):
        loaders = {
            "image": self.load_image,
            "audio": self.load_audio,
            "video": self.load_video,
        }
        return loaders[modality](data)


# 输入/输出都是{模态: [项, ...]}，输出中的项是解码后的对象
def load_multimodal_data(
    data: MultiModalDataItems, loader: MediaLoader
) -> MultiModalDataItems:
    """Decode normalized media, retaining the order within each modality."""
    # 【译】解码规范化后的媒体，保持每个模态内的顺序。
    return {
        modality: [loader.load(modality, item) for item in items]
        for modality, items in data.items()
    }
