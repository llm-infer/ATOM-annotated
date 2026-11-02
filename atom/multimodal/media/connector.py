# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read encoded media without depending on a modality or a model."""

# 【译】读取编码后的媒体字节，不依赖具体模态或模型。
#
# 【导读】媒体来源 -> 原始字节：支持data:...;base64,...形式的data URL、http(s) URL（urllib，带超时）、file:// URL和本地路径
# 调用方：image.load_image（经由MediaLoader）
# 注意http下载是阻塞I/O，所以api_server把整个多模态预处理放到线程池里执行

import base64
import binascii
import urllib.request
from os import PathLike, fspath
from pathlib import Path
from urllib.parse import unquote, urlsplit


# 无状态的读取器，只保存http(s)下载的超时时间（秒）
class MediaConnector:
    def __init__(self, timeout: float = 30):
        self.timeout = timeout

    # 输入：字符串或PathLike；输出：bytes
    #   依次判断：data URL -> http(s) URL -> file:// URL（只允许本机） -> 当作本地文件路径读取
    def read(self, source: str | PathLike[str]) -> bytes:
        source = fspath(source)
        if source.startswith("data:"):
            try:
                header, encoded = source.split(",", 1)
                # 只支持base64编码的data URL；格式不对和base64非法都统一转成ValueError
                if not header.endswith(";base64"):
                    raise ValueError("Expected a base64 data URL")
                return base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid base64 media data URL") from exc
        if source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=self.timeout) as response:
                return response.read()
        if source.startswith("file://"):
            parsed = urlsplit(source)
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("file URLs must refer to a local path")
            source = unquote(parsed.path)
        return Path(source).read_bytes()
