# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 tool-call format.

Every tag is prefixed by the ns_token ``]<]minimax[>[``::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">
    ]<]minimax[>[<pname>value]<]minimax[>[</pname>
    ...
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
not a ``name="..."`` attribute. Values: schema type wins, else JSON, else raw
string.

The ns_token is *not* stripped first, though this said so long after the code
stopped doing it: parsing a copy with it deleted made every offset meaningless
against the bytes that arrived.
"""

# 【译】MiniMax-M3 的工具调用格式。
#
#   每个标签前面都有 ns_token ``]<]minimax[>[`` 前缀：
#       ]<]minimax[>[<tool_call>                                （外层包裹）
#       ]<]minimax[>[<invoke name="NAME">                       （调用开始）
#       ]<]minimax[>[<pname>value]<]minimax[>[</pname>          （参数：标签名就是参数名）
#       ...
#       ]<]minimax[>[</invoke>                                  （调用结束）
#       ]<]minimax[>[</tool_call>                               （包裹结束）
#
#   与 DSML 不同，参数是用标签本身命名的（``<city>Paris</city>``），而不是用 ``name="..."`` 属性。
#   值：schema类型优先，其次JSON，最后原始字符串。
#
#   ns_token *不会*先被剥掉，尽管代码不再这样做之后，这里的文档还这么写了很久：在删掉它的副本上解析，
#   会让每个偏移量相对于实际到达的字节都失去意义。
# 解释：DSML是DeepSeek-V4的工具调用格式，参数写成 <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>，
#   参数名在name属性里；MiniMax则直接用参数名当标签名。DSML的完整格式见 deepseekv4_tool_parser.py 文件头
#
# 【导读】MiniMax-M3 工具调用格式的解析器（NAME="minimax"），启动时 registry.py 按模板中出现 ns_token 识别
# 实现方式：ns_token 在正则里作为可选前缀匹配（_NS），并声明为 CALL_FILLERS，让 markup_begin/markup_end 越过它；
#   _INVOKE_RE 匹配每个 <invoke name="...">（已闭合 / 未闭合），_PARAM_RE 逐个取 <参数名>值</参数名>，值用 coerce_json_or_raw 还原类型；
#   未闭合调用需通过 _is_truncated_call（名字已声明 + 后面紧跟一个已声明参数名的标签）
# 调用方：stream.py 的 ToolCallStreamParser
# 与主干的关系：服务层协议适配。【分支】MiniMax-M3 模型实现，见第8批

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

# MiniMax 的命名空间token，出现在每个标签之前
MINIMAX_NS = "]<]minimax[>["

# The ns_token is matched optionally wherever a tag can appear, as DSML does
# with its own marker, rather than deleted from a copy of the text first.
# Deleting it made every offset in the parsed copy meaningless against the
# bytes that actually arrived, which is why `parse` could only ever report
# "content precedes the call" and never "and this follows it".
# 【译】ns_token 在每个可能出现标签的地方都作为可选项匹配，就像 DSML 对它自己的标记所做的那样，而不是先从文本副本里删掉。
#   删掉它会让解析副本中的每个偏移量相对于实际到达的字节都失去意义，这就是为什么 `parse` 只能报告"内容在调用之前"，
#   却永远报告不了"而这段在它之后"。
_NS = r"(?:" + re.escape(MINIMAX_NS) + r")?"

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <invoke name="NAME">" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
# 【译】一个调用的主体里不能再包含另一个开始标记——那个字面量正是用来打开调用的。没有这个保护，非贪婪的主体会从普通文字里
#   *引用*的开始标记一直延伸到真实调用的闭合标记，于是一个先解释"你写 <invoke name="NAME">"再真正调用的回答，会产生一个以占位符命名、
#   带着真实调用参数的调用，而那句话被删掉。随后 `finditer` 从真实调用之后继续，所以模型真正发出的那个调用根本没有发出去。
#   GLM 最先加上了这个保护；这里是把它推广到其他格式。
_NOT_NESTED = r"(?:(?!" + _NS + r"<invoke\s).)"
# No leading ns_token: `CALL_FILLERS` declares it and `markup_begin` walks
# back over it, so matching it here too is a second reader of one token -- and
# the expensive kind. A pattern opening with an optional group has no fixed
# first byte, so the scan tries every position rather than skipping to the
# next `<`: on 18 KB, 524 us against 3.8, and 262 -> 3.8 for `_PARAM_RE`. The
# other five formats open on a literal and never paid it.
# 【译】开头不带 ns_token：`CALL_FILLERS` 声明了它，`markup_begin` 会往回越过它，所以在这里也匹配它，就是同一个token的第二个读取者——
#   而且是代价高的那种。以一个可选组开头的模式没有固定的首字节，所以扫描会尝试每个位置，而不是直接跳到下一个 `<`：
#   在18KB上为524us对3.8us，`_PARAM_RE` 则是262 -> 3.8。另外五种格式都以字面量开头，从来没有付出这个代价。
# 解释：两个分支——group(1)/group(2) 是到 </invoke> 为止的已闭合调用（名字/主体），group(3)/group(4) 是未闭合调用
_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*?)"
    + _NS
    + r"</invoke>|"
    + r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
# Ends at the matching close tag, at the next parameter, at the close of the
# invoke, or at end of input. That last alternative is what keeps a value the
# model was cut off inside; without it a `max_tokens` truncation arrived as a
# call with `{}` arguments while Qwen and GLM returned the partial value.
# 【译】结束于匹配的闭合标签、下一个参数、invoke的闭合，或输入结尾。最后这个分支保住了模型被截断在其中的那个值；
#   没有它，被 `max_tokens` 截断时会得到一个参数为 `{}` 的调用，而 Qwen 和 GLM 返回的是部分值。
_PARAM_RE = re.compile(
    r"<([\w-]+)>(.*?)"
    r"(?:" + _NS + r"</\1>"
    r"|(?=" + _NS + r"<[\w-]+>)"
    r"|(?=" + _NS + r"</invoke>)"
    r"|\Z)",
    re.DOTALL,
)
# 匹配主体开头的第一个参数标签（可带ns_token），截断判定用它取参数名
_FIRST_TAG_RE = re.compile(r"^" + _NS + r"<([\w-]+)>")


# 判断一个未闭合的 <invoke name=...> 是被截断的真实调用还是普通文字：名字已声明，且主体以（已声明参数名的）标签开头
def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`. The sweep that added that
    test to Qwen and GLM missed this format and DSML both, and "you emit
    `<invoke name="get_weather">` and then a `<city>Paris</city>` line" was
    still dispatched as a real call with the arguments filled in.

    The follower is a tag, because this format names parameters by the tag
    itself and so has no keyword to look for. Which tag is checked too when
    the request declared a schema for this tool -- otherwise `<br>` reads as
    the start of a call body -- but a tool declared without one still has to
    be callable, so an empty schema falls back to "any tag".
    """
    # 【译】这个未闭合的 `<invoke name=...>` 是一个被截断的调用，还是普通文字？
    #
    #   见 :func:`QwenXmlParser._is_truncated_call`。给 Qwen 和 GLM 加上这个检验的那次统一修复漏掉了这个格式和 DSML，
    #   于是"你输出 `<invoke name="get_weather">` 然后写一行 `<city>Paris</city>`"仍会被当作真实调用分发，参数还被填上了。
    #
    #   后续记号是一个标签，因为这个格式用标签本身命名参数，没有可查找的关键字。当请求为这个工具声明了schema时，
    #   还会检查是哪个标签——否则 `<br>` 会被读成调用主体的开始——但一个没带schema声明的工具仍然必须可以被调用，
    #   所以空schema退回到"任意标签"。
    # 解释：_is_truncated_call 实际是 qwen3_tool_parser.py 的模块级函数，不是 QwenXmlParser 的方法
    if not declared_tools_allow(name, param_types):
        return False
    # An undeclared schema falls back to "any tag", the same way an
    # empty one does below.
    # 【译】没有声明schema时退回到"任意标签"，与下面空schema的处理方式相同。
    types = param_types.get(name) or {}
    rest = body.lstrip()
    if not rest:
        return at_end
    tag = _FIRST_TAG_RE.match(rest)
    if tag is not None:
        return not types or tag.group(1) in types
    # A tag that has not finished arriving. At end of region that is all
    # there will ever be, so a prefix counts -- the rule `continues_a_call`
    # states and the other three formats have applied since it was written.
    # The followers are the declared tags, built from the request, because
    # this format names a parameter by the tag itself.
    #
    # An empty schema falls back to "any tag" here exactly as it does for a
    # complete tag four lines up. Gating on `bool(openers)` instead made a
    # zero-parameter tool the one tool whose truncated call could not be
    # recovered -- the same tool declared with one property recovered it --
    # and a truncation is precisely what this function exists to catch.
    # 【译】一个还没完整到达的标签。在区域结尾处，那就是它的全部了，所以前缀也算数——这是 `continues_a_call` 陈述的规则，
    #   另外三种格式从它被写出来起就一直在用。后续记号是声明过的标签，由请求构建，因为这个格式用标签本身命名参数。
    #
    #   空schema在这里退回到"任意标签"，与它在上面四行处理完整标签时完全一样。改为以 `bool(openers)` 为条件，
    #   会让零参数工具成为唯一一个被截断的调用无法恢复的工具——同一个工具声明一个属性时就能恢复——而截断正是这个函数要捕获的情况。
    followers = tuple(f"{MINIMAX_NS}<{t}>" for t in types) + tuple(
        f"<{t}>" for t in types
    ) or (f"{MINIMAX_NS}<", "<")
    return continues_a_call(rest, followers, arrived=not at_end)


# MiniMax-M3 格式的解析器类
class MiniMaxParser(ToolCallParser):
    NAME: ClassVar[str] = "minimax"
    # `<invoke name="` too: `_INVOKE_RE` matches it anywhere, so an invoke the
    # model wrote without the ns_token was a call when parsed whole and plain
    # text when streamed -- the read-ahead never opened a region for it. DSML
    # lists the same marker-less malform for the same reason.
    # 【译】也包括 `<invoke name="`：`_INVOKE_RE` 在任何地方都会匹配它，所以模型没写 ns_token 的 invoke 在整体解析时是调用，
    #   流式时却是普通文本——预读从来没为它打开区域。DSML 出于同样的原因也列出了同样的无标记畸形写法。
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        MINIMAX_NS,
        "<tool_call>",
        '<invoke name="',
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)
    CALL_FILLERS: ClassVar[tuple[str, ...]] = (MINIMAX_NS,)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ("</invoke>",)

    # The ns_token opens a region like the other two, rather than being
    # framing the reader drops. It prefixes every tag including ones inside a
    # call, so calling it framing would delete it from an answer that merely
    # *mentions* it while an answer that mentions the tag it prefixes keeps
    # both -- the same token surviving or not depending on what followed it.
    # `parse` had exactly that split (every occurrence deleted from a copy of
    # the text when a call was found, all of them left when one was not); a
    # region that parses to nothing is released whole, so this way both
    # answers keep their bytes.
    # 【译】ns_token 像另外两个标记一样打开区域，而不是作为读取器丢弃的框架标记。它出现在每个标签前面，包括调用内部的标签，
    #   所以把它当作框架标记，会把它从一个仅仅*提到*它的回答里删掉，而一个提到它所修饰的标签的回答却两者都保留——
    #   同一个token是否保留取决于它后面跟着什么。`parse` 过去恰好有这种分裂（找到调用时从文本副本中删掉每一处，没找到时全部保留）；
    #   一个解析不出东西的区域会被整体释放，所以这样做两种回答都能保住自己的字节。

    # 用本格式写出一次调用（只供测试生成语料）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f"{MINIMAX_NS}<{k}>{v}{MINIMAX_NS}</{k}>" for k, v in args.items()
        )
        return (
            f'{MINIMAX_NS}<tool_call>{MINIMAX_NS}<invoke name="{name}">'
            f"{body}{MINIMAX_NS}</invoke>{MINIMAX_NS}</tool_call>"
        )

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        # 【译】识别 MiniMax-M3 的 ns_token 工具调用格式。
        return MINIMAX_NS in text

    # 解析一个区域：逐个匹配 <invoke name="...">，取名字和参数；span 用 markup_begin/markup_end 扩展到 ns_token 与 <tool_call> 包裹
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _INVOKE_RE.finditer(region):
            closed = m.group(1) is not None
            name = m.group(1) if closed else m.group(3)
            body = m.group(2) if closed else (m.group(4) or "")
            # Strip first, then judge. The other order -- `if not name` before
            # `name.strip()` -- let `name="   "` through the guard and out as
            # the empty string.
            # 【译】先strip，再判断。反过来的顺序——先 `if not name` 再 `name.strip()`——让 `name="   "` 通过了检查，并以空字符串发了出去。
            name = (name or "").strip()
            if not usable_tool_name(name):
                continue
            if not closed and not _is_truncated_call(
                name, body, param_types, at_end=at_end
            ):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
            spans.append(
                (cls.markup_begin(region, m.start()), cls.markup_end(region, m.end()))
            )
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        return RegionParse(tuple(tool_calls), tuple(spans))
