# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3 (qwen3_coder / qwen3_xml) XML tool-call format::

    <tool_call>
    <function=NAME>
    <parameter=PNAME>VALUE</parameter>
    ...
    </function>
    </tool_call>

The XML carries no value types, so parameters are coerced against the request's
``tools`` schema when supplied. Mirrors the qwen3_coder/qwen3_xml parsers in
vLLM and SGLang.
"""

# 【译】Qwen3（qwen3_coder / qwen3_xml）的XML工具调用格式：
#       <tool_call>                              （外层包裹）
#       <function=NAME>                          （调用开始，NAME 为工具名）
#       <parameter=PNAME>VALUE</parameter>      （每个参数一行）
#       ...
#       </function>                              （调用结束）
#       </tool_call>                             （包裹结束）
#   XML不携带值的类型，所以在请求提供了 ``tools`` schema 时，参数会按它做类型转换。
#   对应 vLLM 和 SGLang 中的 qwen3_coder/qwen3_xml 解析器。
#
# 【导读】Qwen 系 XML 工具调用格式的解析器（NAME="qwen"，可用 --tool-call-parser qwen 显式指定）
#   用于chat模板里出现 "<function=" 的模型（如 Qwen3-Coder、Qwen3.5 系列），启动时由 registry.py 从模板识别
# 实现方式：parse_region 用 _FUNCTION_RE 找出每个 <function=...>，分"已闭合"（到 </function>）和"未闭合"（被截断）两种；
#   未闭合的只有通过 _is_truncated_call（名字已声明 + 后面紧跟本格式的下一个记号）才算调用，否则当作普通文字
#   每个调用的 span 用 markup_begin/markup_end 扩展到外层的 <tool_call> / </tool_call>
# 调用方：stream.py 的 ToolCallStreamParser（流式与非流式共用）
# 与主干的关系：服务层协议适配；对应的模型实现见第6批（Qwen3）、第8批（Qwen3.5）

import json
import re
from typing import Any, ClassVar

from .kimi_tool_parser import KIMI_SECTION_BEGIN
from .schema import build_param_types, coerce_param_value
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

# Also read by GlmParser.detect: '<function=' is what tells Qwen's <tool_call>
# apart from GLM's identically-named tag.
# 【译】GlmParser.detect 也会读它：'<function=' 是区分 Qwen 的 <tool_call> 与 GLM 同名标签的依据。
QWEN_TOOL_PREFIX = "<function="

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <function=NAME>" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
# 【译】一个调用的主体里不能再包含另一个开始标记——那个字面量正是用来打开调用的。没有这个保护，非贪婪的主体会从普通文字里
#   *引用*的开始标记一直延伸到真实调用的闭合标记，于是一个先解释"你写 <function=NAME>"再真正调用的回答，会产生一个以占位符命名、
#   带着真实调用参数的调用，而那句话被删掉。随后 `finditer` 从真实调用之后继续，所以模型真正发出的那个调用根本没有发出去。
#   GLM 最先加上了这个保护；这里是把它推广到其他格式。
# 解释：_FUNCTION_RE 的两个分支——group(1) 是到 </function> 为止的已闭合调用，group(2) 是一直到末尾的未闭合（被截断）调用
_NOT_NESTED = r"(?:(?!<function=).)"
_FUNCTION_RE = re.compile(
    r"<function=(" + _NOT_NESTED + r"*?)</function>|<function=(" + _NOT_NESTED + r"*)",
    re.DOTALL,
)
_PARAM_OPENER = "<parameter="
_PARAM_RE = re.compile(
    # `\Z`, not `$`: `$` also matches before a trailing newline, so a value
    # ending in one was cut a byte short. `(?=<tool_call>)` is the fifth
    # terminator `parse_region` names -- a value ends where the next call
    # opens, or it swallows that call's wrapper as data.
    # 【译】用 `\Z` 而不是 `$`：`$` 也会匹配末尾换行符之前的位置，所以以换行结尾的值会被少截一个字节。
    #   `(?=<tool_call>)` 是 `parse_region` 说的第五个终止条件——值在下一个调用开始的地方结束，否则它会把那个调用的包裹当作数据吞掉。
    # 解释：参数值的终止条件依次为：</parameter>、下一个 <parameter=、</function>、下一个 <tool_call>、文本结尾
    r"<parameter=(.*?)"
    r"(?:</parameter>|(?=<parameter=)|(?=</function>)|(?=<tool_call>)|\Z)",
    re.DOTALL,
)


# What may follow the name inside a call: another parameter, or the close of
# the very block the name opened. NOT `</tool_call>`, which closes the
# *outer* wrapper -- `<function=get_weather></tool_call>` leaves the function
# block unterminated, so `parse` reads it as prose. The peek used to accept
# it and `parse` did not, which is the whole of the mismatch this shared
# tuple exists to prevent: one spelling, both readers.
# 【译】调用内名字后面可以跟的东西：另一个参数，或者正是这个名字打开的那个块的闭合。不包括 `</tool_call>`，它关闭的是*外层*包裹——
#   `<function=get_weather></tool_call>` 让function块没有终止，所以 `parse` 把它当作普通文字。过去偷看接受它而 `parse` 不接受，
#   这正是这个共享元组要防止的全部不一致：一种拼写，两个读取者共用。
_CALL_CONTINUES = (_PARAM_OPENER, "</function>")


# 把 "NAME>后续内容" 在 '>' 处切开，返回 (名字, 后续内容)；'>' 还没到时返回None
def _name_and_rest(fn_text: str) -> tuple[str, str] | None:
    """Split `NAME>whatever` at the tag close, or ``None`` if it has not come."""
    # 【译】在标签结束处把 `NAME>whatever` 切开，如果 '>' 还没到则返回 ``None``。
    gt = fn_text.find(">")
    if gt == -1:
        return None
    return fn_text[:gt].strip(), fn_text[gt + 1 :].lstrip()


# 判断一个没闭合的 <function=... 是被截断的真实调用，还是在引用标签的普通文字
def _is_truncated_call(fn_text: str, param_types: dict, *, at_end: bool) -> bool:
    """Is this unclosed `<function=...` a cut-off call, or prose quoting a tag?

    The unclosed branch exists for a call the model was cut off mid-way
    through. It cannot tell that from an answer explaining how to call a tool,
    and used to accept both: "the model writes <tool_call><function=get_weather>
    and then the parameters" produced `get_weather({})`, deleted the rest of
    the sentence and reported `finish_reason: tool_calls`, so an agentic
    client ran a tool nobody asked for.

    Two things separate them, and prose has to fail both. The name is one the
    request declared -- prose can name a real tool, so this alone is not
    enough. And what follows the name is this format's own next token: a
    cut-off call stops inside its own syntax, while prose continues in
    English. The early announcement runs this same function over the region
    so far, so the two cannot answer differently.
    """
    # 【译】这个没闭合的 `<function=...` 是一个被截断的调用，还是在引用标签的普通文字？
    #
    #   未闭合分支是为模型写到一半被截断的调用而存在的。它分辨不出这种情况和一个解释如何调用工具的回答，过去两者都接受：
    #   "模型写 <tool_call><function=get_weather> 然后写参数" 会产生 `get_weather({})`，删掉这句话的剩余部分，
    #   并报告 `finish_reason: tool_calls`，于是一个agent客户端执行了一个没人要求的工具。
    #
    #   有两件事能区分它们，普通文字必须两项都不满足。名字是请求声明过的——普通文字也可能说出一个真实的工具名，所以单靠这一点不够。
    #   并且名字后面跟的是本格式自己的下一个记号：被截断的调用停在它自己的语法内部，而普通文字会继续用英文写下去。
    #   提前播报在目前为止的区域上运行的正是这个函数，所以两者不可能给出不同的答案。
    split = _name_and_rest(fn_text)
    if split is None:
        return False
    name, rest = split
    return declared_tools_allow(name, param_types) and (
        (not rest and at_end)
        or continues_a_call(rest, _CALL_CONTINUES, arrived=not at_end)
    )


# 把一个 <function=NAME>... 块的内部解析成 ToolCall：逐个提取 <parameter=名>值，并按声明类型转换；名字不合法时返回None
def _parse_function(
    fn_text: str, param_types: dict[str, dict[str, Any]]
) -> ToolCall | None:
    """Parse the inside of one ``<function=NAME>...`` block into a ToolCall."""
    # 【译】把一个 ``<function=NAME>...`` 块的内部解析成一个ToolCall。
    gt = fn_text.find(">")
    if gt == -1:
        return None
    name = fn_text[:gt].strip()
    # Not `if not name`, which rejects only the empty one:
    # `<function=YOUR FUNCTION NAME>` -- the placeholder a model writes while
    # explaining the format -- went out as a call. GLM refused the same
    # sentence, so one `<tool_call>` family gave two answers.
    # 【译】不是 `if not name`，那只能拒绝空名字：`<function=YOUR FUNCTION NAME>`——模型在解释格式时写的占位符——曾被作为调用发出。
    #   GLM 拒绝了同一句话，于是同一个 `<tool_call>` 家族给出了两种答案。
    if not usable_tool_name(name):
        return None
    body = fn_text[gt + 1 :]
    types = param_types.get(name, {})
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(body):
        seg = pm.group(1)
        if seg is None:
            continue
        pgt = seg.find(">")
        if pgt == -1:
            continue
        pname = seg[:pgt].strip()
        pval = seg[pgt + 1 :]
        if pname:
            args[pname] = coerce_param_value(pval, types.get(pname))
    return ToolCall(
        id=unique_tool_call_id(),
        type="function",
        function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    )


# Qwen XML 格式的解析器类：声明起始标记、包裹符，实现 parse_region
class QwenXmlParser(ToolCallParser):
    NAME: ClassVar[str] = "qwen"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>", QWEN_TOOL_PREFIX)
    # `</tool_call>` closes the wrapper the calls sit in, so it is markup too
    # -- and so is the newline between it and `</function>`. Not a
    # `REGION_END_MARKER`: a model writing *about* tool calls puts the literal
    # inside a `<parameter=` value, where closing the region on it would cut a
    # real call in half.
    # 【译】`</tool_call>` 关闭的是调用所在的包裹，所以它也是标记——它与 `</function>` 之间的换行也是。它不是 `REGION_END_MARKER`：
    #   一个在*谈论*工具调用的模型会把这个字面量放进 `<parameter=` 的值里，在那里用它关闭区域会把一个真实调用切成两半。
    # 解释：REGION_END_MARKERS 为空时，stream.py 仍会把 CALL_CLOSERS 当作"触发去问 region_end"的字面量，
    #   真正判定闭合的是 region_end 中的 CALL_SELF_CLOSERS（"</function>"）
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ("</function>",)

    # 用本格式写出一次调用（只供测试生成语料）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in args.items())
        return f"<tool_call><function={name}>{body}</function></tool_call>"

    # 识别规则：含 "<function=" 且不含 Kimi 的 section 标记
    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the Qwen3 XML format (and not the Kimi token format)."""
        # 【译】识别 Qwen3 XML 格式（而不是 Kimi 的token格式）。
        return QWEN_TOOL_PREFIX in text and KIMI_SECTION_BEGIN not in text

    # 解析一个区域：逐个匹配 <function=...>；已闭合的直接采用，未闭合的需通过 _is_truncated_call 判定
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for fm in _FUNCTION_RE.finditer(region):
            closed = fm.group(1) is not None
            fn_text = fm.group(1) if closed else fm.group(2)
            if not fn_text:
                continue
            tc = _parse_function(fn_text, param_types)
            if tc is None:
                continue
            if not closed and not _is_truncated_call(
                fn_text, param_types, at_end=at_end
            ):
                continue
            calls.append(tc)
            spans.append(
                (
                    cls.markup_begin(region, fm.start()),
                    cls.markup_end(region, fm.end()),
                )
            )
        return RegionParse(tuple(calls), tuple(spans))
