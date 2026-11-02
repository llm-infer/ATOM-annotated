# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】图片加载：把一个图片输入项解码成RGB模式的PIL.Image
# 调用方：MediaLoader.load_image（media/__init__.py）
# 输入可以是：PIL图像（直接转RGB）、EncodedMedia或bytes（直接解码）、URL/路径字符串（先经MediaConnector读出字节）
from __future__ import annotations

import io
from typing import TYPE_CHECKING

from ..inputs import EncodedMedia, ImageItem
from .connector import MediaConnector

if TYPE_CHECKING:
    from PIL import Image


# 输入：图片项和连接器；输出：RGB模式的PIL.Image；无法识别的图片数据转成ValueError
# PIL在函数内导入：只有真正加载图片时才需要Pillow
def load_image(data: ImageItem, connector: MediaConnector) -> Image.Image:
    from PIL import Image, UnidentifiedImageError

    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, EncodedMedia):
        image_bytes = data.data
    elif isinstance(data, bytes):
        image_bytes = data
    else:
        image_bytes = connector.read(data)
    try:
        # convert返回一张新图像，所以with块结束、关闭原图后仍可使用返回值
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data") from exc
