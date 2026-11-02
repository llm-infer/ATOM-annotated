# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4.1's spaced DSML tags, using the shared V4 recovery and streaming engine."""

# 【译】V4.1 带空格的 DSML 标签，复用 V4 共享的恢复逻辑和流式引擎。
#
# 【导读】DeepSeek-V4.1 的 DSML 工具调用格式解析器（NAME="dsml_v41"）
#   与 V4 的区别只在拼写：标记后多一个空格（"<｜DSML｜ invoke"），section 名从 tool_calls 改为 calls，例如
#     <｜DSML｜ calls><｜DSML｜ invoke name="NAME"><｜DSML｜ parameter name="P" string="true">V</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>  （V4.1写法）
#   所以只需覆盖 DsmlParser 的类属性，并用 _compile_patterns 按新拼写重新编译正则；parse_region 等逻辑全部继承
#   DETECT_MARKERS 都带空格，因此不会与 V4 的 DsmlParser（无空格）互相认领模板
# 调用方：stream.py 的 ToolCallStreamParser；registry._DETECT_ORDER 中排在 DsmlParser 之后
# 与主干的关系：服务层协议适配。【分支】DeepSeek-V4.1 模型实现，见第8批

from typing import ClassVar

from .deepseekv4_tool_parser import DsmlParser, _compile_patterns


# V4.1 拼写的 DSML 解析器：只覆盖标记字面量与正则，解析逻辑继承自 DsmlParser
class DsmlV41Parser(DsmlParser):
    NAME: ClassVar[str] = "dsml_v41"
    DSML_PREFIX: ClassVar[str] = "｜DSML｜ "
    SECTION: ClassVar[str] = "calls"
    PARAM_RE, INVOKE_RE = _compile_patterns(DSML_PREFIX, SECTION)
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ calls>",
        "<｜DSML｜ invoke",
        "<invoke name=",
        "<calls>",
    )
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ calls>",
        "<｜DSML｜ invoke",
        "<｜DSML｜ parameter",
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<｜DSML｜ calls>", "<calls>")
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</｜DSML｜ calls>", "</calls>")
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = (
        "</｜DSML｜ invoke>",
        "</invoke>",
    )
    CALL_CONTINUES: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ parameter",
        "<parameter",
        "</｜DSML｜ invoke>",
        "</invoke>",
    )
