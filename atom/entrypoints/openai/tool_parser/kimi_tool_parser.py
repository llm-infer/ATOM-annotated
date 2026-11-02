# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K2 special-token tool-call format::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>
    <|tool_calls_section_end|>

Arguments are already JSON on the wire, so no schema coercion is applied and
``tools`` is unused for parsing. The call id is the model's own
``functions.NAME:INDEX``.

The section end is a special token that cannot occur inside an argument value,
so this is the one format that can say where a region closes without waiting
for end of stream -- see ``REGION_END_MARKERS``. That is what lets a second
section, and the answer after the last one, be read at all: both used to be
swallowed, differently, by the two readers this format used to have.
"""

# 【译】Kimi-K2 的特殊token工具调用格式：
#       <|tool_calls_section_begin|>                       （section 开始）
#       <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>ARGS_JSON<|tool_call_end|>   （一个条目）
#       <|tool_calls_section_end|>                         （section 结束）
#   参数在线上已经是JSON，所以不做schema类型转换，解析时也不使用 ``tools``。调用id就是模型自己的 ``functions.NAME:INDEX``。
#
#   section结束标记是一个特殊token，不可能出现在参数值内部，所以这是唯一一个不用等到流结束就能说出区域在哪里闭合的格式——
#   见 ``REGION_END_MARKERS``。正是这一点让第二个section以及最后一个section之后的回答能够被读到：
#   这两者过去都会被这个格式原来的两个读取器以不同的方式吞掉。
# 解释：tools 现在仍会传进来，但只用于截断判定中的"名字是否已声明"（_is_truncated_call），不用于类型转换
#
# 【导读】Kimi-K2 特殊token工具调用格式的解析器（NAME="kimi"）
#   不在 registry._DETECT_ORDER 里，但 resolve_from_prompt 会把它排在最后参与识别（模板中出现 <|tool_calls_section_begin|>）
# 实现方式：一个区域 = 一个section；region_end 在看到 section 结束token时直接返回闭合位置（唯一重写了 region_end 的格式）；
#   _ENTRY_RE 逐个匹配条目，整个section作为一个span（而不是每个调用一个）
# 调用方：stream.py 的 ToolCallStreamParser；qwen3_tool_parser.py 用 KIMI_SECTION_BEGIN 排除 Kimi 模板
# 与主干的关系：服务层协议适配；Kimi-K3 是另一套格式，见 kimi_k3_tool_parser.py（模型见第8批）

import json
import re
from typing import ClassVar

from .schema import build_param_types
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    usable_tool_name,
)

# Kimi-K2 的三个特殊token：section开始、section结束、条目结束
KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
KIMI_SECTION_END = "<|tool_calls_section_end|>"
KIMI_ENTRY_END = "<|tool_call_end|>"

_ENTRY_RE = re.compile(
    r"<\|tool_call_begin\|>"
    # `[\w.\-]`, not `\w`: function names are `^[a-zA-Z0-9_-]{1,64}$` and MCP
    # servers namespace theirs `server.tool`. With `\w` the entry did not
    # match, so the whole section -- special tokens included -- went out as
    # `content` with `finish_reason: stop`.
    # 【译】用 `[\w.\-]` 而不是 `\w`：函数名是 `^[a-zA-Z0-9_-]{1,64}$`，MCP 服务器会把它们的工具命名为 `server.tool`。
    #   用 `\w` 时条目匹配不上，整个section——包括特殊token——都作为 `content` 发了出去，`finish_reason: stop`。
    r"functions\.([\w.\-]+):(\d+)" r"<\|tool_call_argument_begin\|>"
    # Ends at `<|tool_call_end|>`, the next entry, the section close, the next
    # *section*, or end of input. Without end-of-input a `max_tokens`
    # truncation matched nothing and went out as raw special tokens -- and
    # beside a complete entry, was dropped entirely. The section opener is the
    # fifth terminator `parse_region` names; here the next call opens a new
    # section rather than a sibling entry.
    # 【译】结束于 `<|tool_call_end|>`、下一个条目、section闭合、下一个*section*，或输入结尾。没有"输入结尾"时，被 `max_tokens`
    #   截断的调用什么也匹配不上，作为原始特殊token发出——而当它旁边有一个完整条目时，则被整个丢掉。section开始标记是
    #   `parse_region` 说的第五个终止条件；在这里，下一个调用会打开一个新的section，而不是一个兄弟条目。
    # 解释：三个捕获组——group(1) 工具名、group(2) 条目序号、group(3) 参数JSON；命名组 closed 表示条目以 <|tool_call_end|> 正常结束
    r"(.*?)"
    r"(?:(?P<closed><\|tool_call_end\|>)"
    r"|(?=<\|tool_call_begin\|>)"
    r"|(?=<\|tool_calls_section_end\|>)"
    r"|(?=<\|tool_calls_section_begin\|>)"
    r"|\Z)",
    re.DOTALL,
)


# 判断一个未闭合的条目是被截断的真实调用（名字已声明，且参数以 '{' 开头或为空），还是在谈论格式的普通文字
def _is_truncated_call(
    name: str, args: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed entry a cut-off call, or prose about the format?

    The gate that travels with an unclosed alternative. Without it a sentence
    quoting `<|tool_call_begin|>functions.x:0<|tool_call_argument_begin|>`
    parses as a call -- the same pair-error K3 had, and it was reintroduced
    here in the very commit that fixed it there.

    The follower is `{`: this format's arguments are JSON on the wire, so a
    real call continues with an object or, at end of region, with nothing.
    """
    # 【译】这个未闭合的条目是一个被截断的调用，还是关于这个格式的普通文字？
    #
    #   这是伴随未闭合分支的那道闸门。没有它，一句引用了 `<|tool_call_begin|>functions.x:0<|tool_call_argument_begin|>`
    #   的话会被解析成一个调用——与 K3 曾有的那个配对错误相同，而它恰恰是在修好 K3 的那次提交里又在这里被引入的。
    #
    #   后续记号是 `{`：这个格式的参数在线上是JSON，所以真实的调用会接着一个对象，或者在区域结尾处什么也不接。
    if not declared_tools_allow(name, param_types):
        return False
    rest = args.lstrip()
    return (not rest and at_end) or continues_a_call(rest, ("{",), arrived=not at_end)


# 解析一个section中的全部条目，返回 (被接受的ToolCall列表, 第一个被接受条目的起始偏移)
def _parse_entries(
    section_text: str, param_types: dict, *, at_end: bool
) -> tuple[list[ToolCall], int]:
    """The entries this section really made, and where the first one starts.

    The offset is returned rather than recomputed by the caller, because the
    caller cannot: entries that fail the truncation gate are skipped here, and
    asking `_ENTRY_RE.search` for the first *match* answers a different
    question than "the first entry we accepted". They differ exactly when an
    answer quotes the wire format before making a real call, which is the case
    the section anchor exists for -- so the recomputation was wrong precisely
    where it mattered.
    """
    # 【译】这个section真正做出的条目，以及第一个条目从哪里开始。
    #
    #   偏移量由这里返回，而不是让调用方重新计算，因为调用方算不出来：没通过截断闸门的条目在这里被跳过了，
    #   而向 `_ENTRY_RE.search` 要第一个*匹配*，回答的是和"我们接受的第一个条目"不同的问题。两者恰好在回答先引用线上格式
    #   再真正调用时不同，而这正是section锚点存在的原因——所以重新计算恰恰在最要紧的地方出错。
    tool_calls = []
    first_start = -1
    for match in _ENTRY_RE.finditer(section_text):
        name = match.group(1)
        index = match.group(2)
        # `[\w.\-]+` is not the same set: it admits a leading `.` or `-`, so
        # `functions..hidden:0` shipped a call named `.hidden`. "Safe by
        # construction" is why this parser was left out of the sweep.
        # 【译】`[\w.\-]+` 并不是同一个集合：它允许以 `.` 或 `-` 开头，所以 `functions..hidden:0` 发出了一个名为 `.hidden` 的调用。
        #   "构造上就安全"正是这个解析器被排除在那次统一修复之外的原因。
        if not usable_tool_name(name):
            continue
        if match.group("closed") is None and not _is_truncated_call(
            name, match.group(3), param_types, at_end=at_end
        ):
            continue
        # `"{}"` and not `""` for a zero-argument call. This format passes
        # the wire bytes through where the other five build a JSON object, so
        # it was the one format whose no-argument call reached the client as
        # an empty string -- `json.loads("")` raises, and on `/v1/messages` it
        # becomes an `input_json_delta` the Anthropic SDK cannot accumulate.
        # 【译】零参数调用用 `"{}"` 而不是 `""`。这个格式把线上的字节直接透传，而另外五种格式会构建一个JSON对象，
        #   所以它曾是唯一一个零参数调用以空字符串到达客户端的格式——`json.loads("")` 会抛异常，而在 `/v1/messages` 上它会变成
        #   一个 Anthropic SDK 无法累积的 `input_json_delta`。
        arguments = match.group(3).strip() or "{}"
        tool_id = f"functions.{name}:{index}"
        if first_start < 0:
            first_start = match.start()
        tool_calls.append(
            ToolCall(
                id=tool_id,
                type="function",
                function={"name": name, "arguments": arguments},
            )
        )
    return tool_calls, first_start


# Kimi-K2 格式的解析器类：一个区域就是一个section，section结束token即区域结束
class KimiParser(ToolCallParser):
    NAME: ClassVar[str] = "kimi"
    # The section opener, and the only literal detection keys on. The entry
    # markers inside it are `parse_region`'s business, never a reader's.
    # 【译】section开始标记，也是识别所依据的唯一字面量。section内部的条目标记归 `parse_region` 管，从来不归读取器管。
    START_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_BEGIN,)
    # An entry's own end. `CALL_CLOSERS` stays empty: the section end is a
    # `REGION_END_MARKER`, which is a stronger statement than "markup at a
    # call's right edge" and is what `markup_end` must not walk over here.
    # 【译】一个条目自己的结束标记。`CALL_CLOSERS` 保持为空：section结束是一个 `REGION_END_MARKER`，这比"位于调用右边界的标记"
    #   是更强的表述，而且正是 `markup_end` 在这里不能越过的东西。
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = (KIMI_ENTRY_END,)
    # A special token, so it cannot appear inside a JSON argument value. That
    # is the whole licence for closing a region on it: the XML formats' own
    # closers fail this test, because a model writing about tool calls puts
    # one inside a parameter.
    # 【译】它是一个特殊token，所以不可能出现在JSON参数值内部。这就是可以凭它关闭区域的全部依据：
    #   XML格式自己的闭合标记通不过这个检验，因为一个在谈论工具调用的模型会把它写进某个参数里。
    REGION_END_MARKERS: ClassVar[tuple[str, ...]] = (KIMI_SECTION_END,)

    # 用本格式写出一次调用（只供测试生成语料）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        # One call, so the entry index is 0. Do not read a multi-call shape
        # off two of these concatenated: a real section increments the index
        # across its entries, and taking the doubled `:0` for model output
        # produced a duplicate-id "defect" that does not exist.
        # 【译】只有一个调用，所以条目序号是0。不要把两个这样的结果拼接起来当作多调用的形状：真实的section会在它的条目之间递增序号，
        #   把两个 `:0` 当作模型输出，会产生一个并不存在的"重复id缺陷"。
        return (
            f"{KIMI_SECTION_BEGIN}<|tool_call_begin|>functions.{name}:0"
            f"<|tool_call_argument_begin|>{json.dumps(args)}"
            f"{KIMI_ENTRY_END}{KIMI_SECTION_END}"
        )

    @classmethod
    def detect(cls, text: str) -> bool:
        return KIMI_SECTION_BEGIN in text

    # 重写基类的 region_end：找到第一个 section 结束token，返回它之后的位置；没找到返回0
    @classmethod
    def region_end(cls, region: str) -> int:
        at = region.find(KIMI_SECTION_END)
        return at + len(KIMI_SECTION_END) if at != -1 else 0

    # 解析一个section：取出全部条目；span 从第一个被接受条目之前最近的 section 开始标记，一直到区域末尾
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """One section: everything between its two markers, or what arrived.

        `region_end` hands this exactly one section at a time, so a response
        with two of them is two regions and the text between and after them
        reaches the client. Reading the *first* section out of the whole
        output -- which is what `_SECTION_RE.search` did -- lost the second
        call entirely and delivered the raw wire tokens of both as content.
        """
        # 【译】一个section：它的两个标记之间的一切，或者已经到达的部分。
        #
        #   `region_end` 每次只交给它恰好一个section，所以有两个section的响应就是两个区域，它们之间和之后的文字都能到达客户端。
        #   从整个输出中读出*第一个*section——这正是 `_SECTION_RE.search` 过去做的——会把第二个调用整个丢掉，
        #   并把两者的原始线上token都作为content交付出去。
        entries, first_start = _parse_entries(
            region, build_param_types(tools), at_end=at_end
        )
        if not entries:
            return RegionParse()
        # The region opens at the *first* section marker in the text, which an
        # answer that quotes one before making a real call puts in the wrong
        # place. The section that matters is the last one opened before the
        # first entry; everything before it is the answer.
        #
        # The first entry *accepted*, not the first thing the entry pattern
        # matched. `_ENTRY_RE.search` was the latter, so a quotation the
        # truncation gate had already rejected still moved the anchor back to
        # its section -- and the sentence the model wrote between the
        # quotation and its real call was deleted as markup. The other five
        # formats keep it; this was the only one that did not.
        # 【译】区域在文本中*第一个*section标记处打开，而一个先引用标记再真正调用的回答会让这个位置不对。真正要紧的section是
        #   第一个条目之前最后打开的那个；它之前的一切都是回答。
        #
        #   用第一个*被接受*的条目，而不是条目模式匹配到的第一个东西。`_ENTRY_RE.search` 是后者，所以一段已经被截断闸门拒绝的引用
        #   仍然会把锚点拉回到它所在的section——于是模型在引用和真实调用之间写的那句话被当作标记删掉了。
        #   另外五种格式都保留它；这是唯一一个没保留的。
        opened = region.rfind(KIMI_SECTION_BEGIN, 0, first_start)
        # One span for the whole section, not one per entry: `region_end`
        # already hands this exactly one section, and between two entries
        # there is only `<|tool_call_end|><|tool_call_begin|>` -- special
        # tokens a model cannot write prose between.
        # 【译】整个section一个span，而不是每个条目一个：`region_end` 已经每次只交给它恰好一个section，而两个条目之间只有
        #   `<|tool_call_end|><|tool_call_begin|>`——模型不可能在这些特殊token之间写普通文字。
        return RegionParse(tuple(entries), ((max(opened, 0), len(region)),))
