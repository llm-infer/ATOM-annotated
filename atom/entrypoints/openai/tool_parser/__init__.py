# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parsing for models that emit tool calls in their text output.

Model-specific wire formats, normalized into the OpenAI ``tool_calls`` structure.
Which one a model uses is resolved once at startup from its chat template
(:func:`~.registry.resolve_tool_call_parser`) or set explicitly with
``--tool-call-parser``; it is not inferred from the output, because inferring
it means deciding from a prefix that may not carry the discriminator yet.

==============================  ===============================================
Module                          Format
==============================  ===============================================
`kimi_tool_parser`              Kimi-K2 ``<|tool_call_begin|>`` special tokens
`kimi_k3_tool_parser`           Kimi-K3 ``<|open|>call tool="`` channel format
`qwen3_tool_parser`             Qwen3 (qwen3_coder / qwen3_xml) ``<function=`` XML
`deepseekv4_tool_parser`        DeepSeek-V4 ``<｜DSML｜invoke>`` markup
`glm_tool_parser`               GLM-4.5/4.6/5.x ``<tool_call>``/``<arg_key>``
`minimax_tool_parser`           MiniMax-M3 ``]<]minimax[>[``-prefixed tags
==============================  ===============================================

The XML-ish formats (Qwen / DSML / GLM / MiniMax) carry no value types on the
wire, so when the request's ``tools`` schema is supplied each parameter is
coerced to its declared JSON-Schema type; otherwise it is left as a string.
Kimi-K2 arguments are already JSON; Kimi-K3 carries a per-argument
``type="..."`` on the wire, so neither needs the request schema.

Two entry points, both format-agnostic:

- :func:`parse_tool_calls` — a complete output -> ``(content, [ToolCall])``
- :class:`ToolCallStreamParser` — chunks -> ``(event_type, data)`` tuples

OpenAI format::

    {"tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}

To add a format: implement :class:`~.tool_parser.ToolCallParser` in its own
``<model>_tool_parser.py``, declaring its ``START_MARKERS`` and a
``parse_region``, then add it to ``_DETECT_ORDER`` in :mod:`.registry`, whose
ordering constraints are documented there. That one entry is enough: startup
resolution, the streaming read-ahead, ``--tool-call-parser``\'s accepted names
and the property tests are all derived from it. A format writes no state
machine and no whole-output parser of its own -- :mod:`.stream` is the only
reader, and it is the only reader for both delivery modes.
"""

# 【译】为在文本输出中写出工具调用的模型解析工具调用。
#
#   各模型特有的线上格式，统一规范成 OpenAI 的 ``tool_calls`` 结构。一个模型使用哪种格式，在启动时从它的chat模板解析一次
#   （:func:`~.registry.resolve_tool_call_parser`），或者用 ``--tool-call-parser`` 显式设置；它不从输出推断，
#   因为推断意味着要根据一个可能还没带上判别符的前缀做决定。
#
#   模块                          格式
#   `kimi_tool_parser`            Kimi-K2 的 ``<|tool_call_begin|>`` 特殊token
#   `kimi_k3_tool_parser`         Kimi-K3 的 ``<|open|>call tool="`` 频道格式
#   `qwen3_tool_parser`           Qwen3（qwen3_coder / qwen3_xml）的 ``<function=`` XML
#   `deepseekv4_tool_parser`      DeepSeek-V4 的 ``<｜DSML｜invoke>`` 标记
#   `glm_tool_parser`             GLM-4.5/4.6/5.x 的 ``<tool_call>``/``<arg_key>``
#   `minimax_tool_parser`         MiniMax-M3 以 ``]<]minimax[>[`` 为前缀的标签
#
#   类XML的格式（Qwen / DSML / GLM / MiniMax）在线上不携带值类型，所以提供了请求的 ``tools`` schema 时，
#   每个参数会被转换成它声明的JSON-Schema类型；否则保留为字符串。Kimi-K2 的参数本来就是JSON；Kimi-K3 在线上为每个参数
#   携带 ``type="..."``，所以两者都不需要请求schema。
#
#   两个入口，都与格式无关：
#   - :func:`parse_tool_calls` —— 完整输出 -> ``(content, [ToolCall])``
#   - :class:`ToolCallStreamParser` —— 逐个chunk -> ``(event_type, data)`` 元组
#
#   OpenAI 格式：{"tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}
#
#   新增一种格式：在它自己的 ``<model>_tool_parser.py`` 里实现 :class:`~.tool_parser.ToolCallParser`，声明它的 ``START_MARKERS``
#   和一个 ``parse_region``，然后把它加入 :mod:`.registry` 的 ``_DETECT_ORDER``，那里记录了顺序上的约束。这一项就够了：
#   启动时的格式解析、流式预读、``--tool-call-parser`` 接受的名字以及属性测试都由它推导而来。一种格式不需要写自己的状态机，
#   也不需要写整段输出的解析器——:mod:`.stream` 是唯一的读取器，而且是两种交付模式共同的唯一读取器。
# 解释：表格漏了 deepseekv41_tool_parser（DeepSeek-V4.1，标记后带空格的 ``<｜DSML｜ invoke``，NAME="dsml_v41"）；
#   示例中的 id "call_0" 是早期写法，现在 id 由 unique_tool_call_id 随机生成（Kimi-K2 用模型自己的 functions.NAME:INDEX）
#
# 【导读】工具调用解析器（tool parser）家族总览：模型输出里的工具调用标记 -> OpenAI 的 tool_calls
# 要解决的问题：模型用各自的文本标记写出工具调用（如 Qwen 的 <tool_call><function=...>），而 OpenAI / Anthropic 协议要的是结构化的
#   tool_calls / tool_use；流式时还要边收token边判断"这段是普通回答还是工具调用"——既不能把半截标记发给客户端，
#   也不能把在谈论语法的普通文字误判成调用
# 文件分工：
#   tool_parser.py    基类 ToolCallParser（每种格式只声明标记字面量 + 实现 parse_region）、ToolCall、RegionParse 和共用判定函数
#   stream.py         唯一的解析引擎 ToolCallStreamParser：预读、缓冲区域、判定闭合、提前播报工具名、按顺序产出事件；
#                     非流式的 read_whole 就是它在单个chunk上运行一次，所以 stream=true/false 的结果不会不一致
#   registry.py       已注册格式列表 _DETECT_ORDER；启动时 resolve_tool_call_parser 从chat模板识别格式；非流式入口 parse_tool_calls
#   schema.py         从请求 tools 取参数类型，把字符串参数值还原成声明的类型
#   *_tool_parser.py  各模型的格式：qwen3 / glm / kimi（K2）/ kimi_k3 / minimax / deepseekv4（deepseekv41 继承它）
# 读法建议：先读 tool_parser.py 的 ToolCallParser 和 stream.py 的 _pump，再挑一个最简单的格式（qwen3_tool_parser.py）看 parse_region；
#   其余格式都是同一套接口：START_MARKERS / CALL_OPENERS / CALL_CLOSERS / CALL_SELF_CLOSERS + parse_region
# 在请求链路中的位置：
#   启动：api_server.main() -> registry.resolve_tool_call_parser -> 得到本模型的解析器类（或None = 不解析）
#   每个请求：serving_chat.py（/v1/chat/completions）、serving_anthropic.py 与 api_server.py（/v1/messages、/v1/responses）、
#     atomesh 的 standalone 服务 -> 先由 reasoning.py 分离出推理内容，回答部分再交给 ToolCallStreamParser.process / flush
#     -> 事件被组装成 SSE 中的 delta.tool_calls（或非流式响应的 message.tool_calls）
# 与主干的关系：纯服务层后处理，作用于detokenize后的文本，不影响EngineCore进程、调度与KV cache

from .deepseekv4_tool_parser import DsmlParser
from .deepseekv41_tool_parser import DsmlV41Parser
from .glm_tool_parser import GlmParser
from .kimi_k3_tool_parser import KimiK3Parser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
from .registry import parse_tool_calls
from .stream import (
    ToolCallStreamParser,
    flatten_tool_events,
    read_whole,
    read_whole_events,
)
from .tool_parser import RegionParse, ToolCall, ToolCallParser

__all__ = [
    "DsmlParser",
    "DsmlV41Parser",
    "GlmParser",
    "KimiK3Parser",
    "KimiParser",
    "MiniMaxParser",
    "QwenXmlParser",
    "RegionParse",
    "ToolCall",
    "ToolCallParser",
    "ToolCallStreamParser",
    "flatten_tool_events",
    "parse_tool_calls",
    "read_whole",
    "read_whole_events",
]
