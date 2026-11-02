# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request-schema lookup and value coercion shared by every wire format.

The XML-ish tool-call formats (Qwen, DSML, GLM, MiniMax) carry no value types on
the wire, so parameter values arrive as strings. When the request supplies a
``tools`` schema each value is coerced to its declared JSON-Schema type;
otherwise it is left alone.
"""

# 【译】各种线上格式共用的请求schema查找与参数值类型转换。
#
#   类XML的工具调用格式（Qwen、DSML、GLM、MiniMax）在线上不携带值的类型，所以参数值以字符串到达。
#   当请求提供了 ``tools`` schema 时，每个值会被转换成它声明的JSON-Schema类型；否则保持原样。
#
# 【导读】工具调用参数的类型还原：从请求的 tools 里取出"函数名 -> {参数名: 类型}"，再把字符串参数值转成对应类型
#   例：tools 声明 get_weather(days: integer)，模型写出 <parameter=days>3</parameter>，
#   coerce_param_value("3", "integer") 得到 3，最终 arguments 是 {"days": 3} 而不是 {"days": "3"}
# 调用方：各格式的 parse_region（qwen3 / deepseekv4 / glm / minimax / kimi / kimi_k3 解析器）；
#   stream.py 的 ToolCallStreamParser 在构造时用 build_param_types 预先建好映射
# 与主干的关系：服务层响应后处理的一部分，不涉及引擎

import ast
import json
from typing import Any

from .tool_parser import qualified_tool_name


# 已构建好的"函数名 -> {参数名: 类型}"映射；用子类本身作标记，表示"已经建过，无需重建"
class ParamTypes(dict):
    """A mapping :func:`build_param_types` has already produced.

    Every ``parse_region`` begins by asking for this mapping, and the streaming
    reader asks once per chunk while it is still deciding whether the region
    reveals a call. Rebuilding it each time walks the whole request catalogue:
    measured at 90 us per call with 200 declared tools, which was the entire
    difference between a 0.12 ms and a 1.28 ms streamed call. Carrying the
    built mapping in place of the list it came from makes the rebuild a type
    check. The subclass is the tag -- a plain dict could be a caller's tools.
    """


# 【译】一个 :func:`build_param_types` 已经产出的映射。
#
#   每个 ``parse_region`` 一开始都要这个映射，而流式读取器在还没判断出区域是否显露出一个调用时，每个chunk都要问一次。
#   每次重建都要遍历请求的整个工具目录：在声明了200个工具时实测每次90us，这就是一次流式调用耗时0.12ms还是1.28ms的全部差别。
#   用建好的映射代替它来源的那个列表随处传递，重建就变成了一次类型检查。子类本身就是标记——普通dict可能是调用方传来的tools。


# 从请求的tools构建 {函数名(含namespace): {参数名: JSON-Schema类型}}；传入已建好的 ParamTypes 时直接原样返回
def build_param_types(tools: list | None) -> dict[str, dict[str, Any]]:
    """Map ``function_name -> {param_name: json_schema_type}`` from request tools.

    Accepts OpenAI (``{"type": "function", "function": {...}}``) and bare
    (``{"name": ..., "parameters"/"input_schema": {...}}``) tool entries, and
    its own output, which it returns unchanged (see :class:`ParamTypes`).
    """
    # 【译】从请求的tools构建 ``function_name -> {param_name: json_schema_type}`` 映射。
    #
    #   接受 OpenAI 形式（``{"type": "function", "function": {...}}``）和裸形式
    #   （``{"name": ..., "parameters"/"input_schema": {...}}``）的工具条目，也接受它自己的输出，
    #   此时原样返回（见 :class:`ParamTypes`）。
    if isinstance(tools, ParamTypes):
        return tools
    out: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        try:
            name = qualified_tool_name(tool)
        except ValueError:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        props = schema.get("properties") if isinstance(schema, dict) else None
        out[name] = {
            k: (v.get("type") if isinstance(v, dict) else None)
            for k, v in (props or {}).items()
        }
    return ParamTypes(out)


# 按声明类型转换一个字符串参数值（int/float/bool/object/array/null…）；类型未知或转换失败时返回原字符串
def coerce_param_value(value: str, ptype: Any) -> Any:
    """Coerce a string parameter value to its declared JSON-Schema type.

    No schema type (string/unknown) -> returned unchanged. Conversion failures
    fall back to the original string rather than raising.
    """
    # 【译】把一个字符串参数值转换成它声明的JSON-Schema类型。
    #
    #   没有schema类型（字符串/未知）-> 原样返回。转换失败时退回原字符串，而不是抛异常。
    # 解释：返回前会先去掉首尾换行（XML格式的值常写成 "\n值\n"）
    v = value.strip("\n")
    if ptype is None:
        return v
    t = str(ptype).lower()
    try:
        if t in ("string", "str", "text", "varchar", "char", "enum"):
            return v
        if t in ("null", "none"):
            return None
        if t.startswith(("int", "uint", "long", "short", "unsigned")):
            return int(v)
        if t.startswith(("num", "float", "double", "decimal")):
            f = float(v)
            return int(f) if f.is_integer() else f
        if t.startswith(("bool", "binary")):
            return v.strip().lower() == "true"
        if t.startswith(("object", "dict", "map", "array", "list", "tuple")):
            try:
                return json.loads(v)
            except (ValueError, TypeError):
                # 【译】（行尾注释）对单引号的Python字面量更安全
                return ast.literal_eval(v)  # safer for single-quoted Python literals
    except Exception:  # noqa: BLE001
        return v
    return v


# 解码一个不带类型的值：有schema类型按类型转换；否则先试JSON，失败则保留原字符串
def coerce_json_or_raw(value: str, ptype: Any) -> Any:
    """Decode one untyped value: schema type wins, else JSON, else raw string.

    Shared by GLM (``<arg_value>``) and MiniMax (``<tag>``), whose templates both
    render non-string values with ``tojson`` and string values raw.
    """
    # 【译】解码一个无类型的值：schema类型优先，其次JSON，最后原始字符串。
    #
    #   GLM（``<arg_value>``）和 MiniMax（``<tag>``）共用，它们的模板都用 ``tojson`` 渲染非字符串值，字符串值则原样渲染。
    v = value.strip("\n")
    if ptype is not None:
        return coerce_param_value(v, ptype)
    s = v.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return v
