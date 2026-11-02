# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 channel-format tool-call format::

    <|open|>call tool="NAME" index="i"<|sep|>
      <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|> ...
    <|close|>call ...

Argument VALUEs are raw-by-type (string unquoted, number/bool/object as literals)
and are coerced then assembled into one JSON object per call. Unlike the XML-ish
formats the type travels on the wire (``type="..."``) so no request ``tools``
schema is needed; ``tools`` is unused.

K3 interleaves think/response/tools sections in one channel-framed stream, so
partial-chunk parsing is unreliable: this parser buffers the whole output and
parses once at flush. Plain (non-tool) answers also carry channel framing
tokens, which are stripped from ``content``.
"""

# 【译】Kimi-K3 的频道格式（channel format）工具调用：
#       <|open|>call tool="NAME" index="i"<|sep|>                                         （调用开始）
#         <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|> ...         （每个参数，带类型）
#       <|close|>call ...                                                                   （调用结束）
#   参数的 VALUE 按类型原样书写（字符串不加引号，数字/布尔/对象写成字面量），先做类型转换，再为每个调用组装成一个JSON对象。
#   与类XML格式不同，类型随线上数据一起传输（``type="..."``），所以不需要请求的 ``tools`` schema；``tools`` 不被使用。
#
#   K3 在一个带频道框架的流里交错输出 think/response/tools 几个段，所以按部分chunk解析不可靠：这个解析器缓冲整个输出，
#   在flush时一次性解析。普通（非工具）回答也带着频道框架token，它们会从 ``content`` 中被剥掉。
# 解释：这段docstring已部分过时——
#   1. tools 现在会用于截断判定中的"名字是否已声明"（_is_truncated_call），只是不用于类型转换；
#   2. 现在并不缓冲整个输出：response 频道边到边流式输出，只有 call / tools 段作为区域缓冲，
#      最后一个 <|close|>call 之后的 <|close|>tools 一到，基类 region_end 就判定区域闭合（见下方类docstring）；
#   3. 框架token由 stream.py 的扫描器在 opens_region 返回False时丢弃
#
# 【导读】Kimi-K3 频道格式工具调用的解析器（NAME="kimi_k3"），在 registry._DETECT_ORDER 中排第一
# 特点：K3 的每个回答都包着频道框架（<|open|>response<|sep|> ... <|close|>response<|sep|><|end_of_msg|>），
#   所以 START_MARKERS 包含全部框架token（不能被chunk切开），但 opens_region 只对 call 前缀和 tools 段开始返回True，
#   其余框架token由引擎直接丢弃——这就是 opens_region 与 START_MARKERS 分开的原因
# token 字面量统一来自 kimi_k3_tokens.py（推理内容分离 reasoning_dialects.py 也读它）
# 调用方：stream.py 的 ToolCallStreamParser
# 与主干的关系：服务层协议适配。【分支】Kimi-K3 模型实现，见第8批

import json
import re
from typing import ClassVar

from .. import kimi_k3_tokens as k3
from .schema import build_param_types
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

# K3 channel tokens this parser matches on. Kept local so the parser is
# self-contained; the reasoning splitter declares its own copies.
# 【译】本解析器匹配用到的 K3 频道token。放在本地，让解析器自成一体；推理拆分器声明了它自己的副本。
# 解释：已过时——两边现在都从 kimi_k3_tokens.py 取token；这几个本地常量只剩 is_kimi_k3、START_MARKERS、_REGION_MARKERS 在用
KIMI_K3_CALL_PREFIX = '<|open|>call tool="'
KIMI_K3_TOOLS_START = "<|open|>tools<|sep|>"
KIMI_K3_RESPONSE_START = "<|open|>response<|sep|>"
KIMI_K3_RESPONSE_END = "<|close|>response<|sep|>"
KIMI_K3_END_OF_MSG = "<|end_of_msg|>"

# A call's body may not contain another call opener, nor an argument's value
# another argument opener -- those literals are what open one. Without the
# guard the non-greedy body ran from an opener *quoted in prose* to the real
# call's closer, so an answer explaining `<|open|>call tool="NAME"<|sep|>`
# before making a real call produced one call named `NAME` carrying the real
# call's arguments. Every registered format carries this guard now.
# 【译】一个调用的主体里不能再包含另一个调用开始标记，一个参数的值里也不能再包含另一个参数开始标记——那些字面量正是用来打开它们的。
#   没有这个保护，非贪婪的主体会从*普通文字里引用*的开始标记一直延伸到真实调用的闭合标记，于是一个先解释
#   `<|open|>call tool="NAME"<|sep|>` 再真正调用的回答，会产生一个名为 `NAME`、带着真实调用参数的调用。现在每种已注册格式都带着这个保护。
_NOT_NESTED_CALL = r'(?:(?!<\|open\|>call tool=").)'
_NOT_NESTED_ARG = r'(?:(?!<\|open\|>argument key=").)'
# What may follow a call's opener in a real call: an argument, or the close of
# the call the name opened. One tuple, read by `_is_truncated_call`.
# 【译】在真实调用中，调用开始标记后面可以跟的东西：一个参数，或者这个名字打开的那个调用的闭合。只用一个元组，由 `_is_truncated_call` 读取。
_CALL_CONTINUES = ('<|open|>argument key="', "<|close|>call")
# Closed, cut off where the next call or section opens, or cut off at end of
# input. Without end-of-input a call truncated by `max_tokens` parsed to
# nothing and its raw tokens went out as the answer. The opener lookaheads are
# the fifth terminator `parse_region` names, and this format needed them most:
# the tempered dot already stops the body at a sibling, so with no terminator
# accepting that position the whole entry matched nothing and was dropped --
# after the announcement had named it.
# 【译】已闭合、在下一个调用或段开始处被截断、或在输入结尾被截断。没有"输入结尾"时，被 `max_tokens` 截断的调用什么也解析不出，
#   它的原始token作为回答发了出去。开始标记的前瞻是 `parse_region` 说的第五个终止条件，而这个格式最需要它们：
#   受限的点号（tempered dot）已经让主体停在兄弟调用处，所以如果没有终止条件接受那个位置，整个条目什么也匹配不上而被丢掉——
#   而那时提前播报已经说出了它的名字。
# 解释：命名组 name / index / body，closed 非空表示以 <|close|>call 正常结束
_K3_CALL_RE = re.compile(
    r'<\|open\|>call tool="(?P<name>[^"]*)"(?:\s+index="(?P<index>\d+)")?<\|sep\|>'
    r"(?P<body>" + _NOT_NESTED_CALL + r"*?)"
    # The separator belongs to the closer, as it does in the model's encoder
    # and in vLLM's and SGLang's readers. Optional: a truncation can stop
    # between the two.
    # 【译】分隔符属于闭合标记，这与模型的编码器以及 vLLM、SGLang 的读取器一致。它是可选的：截断可能停在两者之间。
    r"(?:(?P<closed><\|close\|>call(?:<\|sep\|>)?)"
    r'|(?=<\|open\|>call tool=")'
    r"|(?=<\|open\|>tools<\|sep\|>)"
    r"|(?=<\|close\|>tools)"
    r"|\Z)",
    re.DOTALL,
)
_K3_ARG_RE = re.compile(
    r'<\|open\|>argument key="(?P<key>[^"]*)"(?:\s+type="(?P<type>[^"]*)")?<\|sep\|>'
    # Same alternation one level down: a value the model was cut off inside
    # survives, and an argument missing its closer before a sibling is not
    # deleted -- that dropped a parameter and dispatched the call without it.
    # 【译】下一层用同样的分支选择：模型被截断在其中的值得以保留，而一个在兄弟参数之前缺少闭合标记的参数不会被删掉——
    #   那曾丢掉一个参数，并在没有它的情况下分发了调用。
    r"(?P<val>" + _NOT_NESTED_ARG + r"*?)"
    r"(?:<\|close\|>argument"
    r'|(?=<\|open\|>argument key=")'
    r"|(?=<\|open\|>tools<\|sep\|>)"
    r"|(?=<\|close\|>call)"
    r"|\Z)",
    re.DOTALL,
)
# The channel framing this format wraps every answer in, tool call or not.
# Declared once and consumed once: `START_MARKERS` lists these so the
# read-ahead cannot split one, `opens_region` says no to them so the reader
# drops them, and that is the only place they are removed. There used to be a
# second removal inside `parse` -- a regex built from a hand-kept copy of this
# list -- and the two disagreed about four tokens, which reached the client
# verbatim when streamed and were deleted when not.
#
# `call` and `argument` are deliberately absent: they carry data
# (`tool="..."`, `key="..."`) so they cannot be declared as literals, and they
# only ever occur inside a tools section, where `_K3_CALL_RE` and `_K3_ARG_RE`
# account for them.
# 【译】这种格式包裹在每个回答外面的频道框架，无论有没有工具调用。只声明一次，只消费一次：`START_MARKERS` 列出它们，
#   使预读不会把其中某个切开；`opens_region` 对它们说"否"，使读取器丢弃它们；这是唯一删除它们的地方。过去在 `parse` 里还有
#   第二次删除——用一份手工维护的列表副本构建的正则——而两者对四个token的看法不一致，这些token流式时原样到达客户端，非流式时被删掉。
#
#   `call` 和 `argument` 被刻意排除在外：它们携带数据（`tool="..."`、`key="..."`），所以不能被声明为字面量，
#   而且它们只出现在tools段内部，由 `_K3_CALL_RE` 和 `_K3_ARG_RE` 负责处理。
# 解释：这段注释没有紧挨着代码，它说明的是下面 _K3_CONTENT_FRAMING 的作用；这里说"call / argument 不在列表中"指的是
#   带数据的开始标记，它们的闭合标记 <|close|>call、<|close|>argument 属于 TOOL_REGION_FRAMING，仍在 START_MARKERS 中

# This format's own framing: what wraps every answer, plus what brackets a
# call. Both halves come from `kimi_k3_tokens`, which the reasoning dialect
# reads too -- do not re-spell these literals here, that is how the two
# copies came to disagree.
# 【译】本格式自己的框架：包裹每个回答的那些，加上括住一个调用的那些。两部分都来自 `kimi_k3_tokens`，推理方言也读它——
#   不要在这里重新拼写这些字面量，两份副本正是这样变得不一致的。
_K3_CONTENT_FRAMING = (
    *k3.CHANNEL_FRAMING,
    k3.THINK_START,
    k3.THINK_END,
    *k3.TOOL_REGION_FRAMING,
    *k3.UNPAIRED_FRAMING,
)


# 文本里是否出现 K3 特有的五个token之一（call前缀、tools段开始、response开始/结束、end_of_msg）；detect 用它在chat模板上识别
def is_kimi_k3(text: str) -> bool:
    return (
        KIMI_K3_CALL_PREFIX in text
        or KIMI_K3_TOOLS_START in text
        or KIMI_K3_RESPONSE_START in text
        or KIMI_K3_RESPONSE_END in text
        or KIMI_K3_END_OF_MSG in text
    )


# 按线上携带的 type 把参数值转换成对应类型：字符串原样返回，其他类型先strip再转换，失败返回strip后的原值
def _k3_coerce(val: str, ptype: str | None):
    t = (ptype or "").lower()
    # Strings are returned verbatim: leading/trailing whitespace can be
    # semantically significant (e.g. an enum of whitespace values), so stripping
    # would corrupt valid values. Non-string types strip first, since surrounding
    # whitespace there is only formatting noise around the literal to coerce.
    # 【译】字符串原样返回：首尾空白可能有语义（例如一个由空白值组成的枚举），所以strip会破坏合法的值。
    #   非字符串类型先strip，因为那里周围的空白只是包在要转换的字面量外面的格式噪声。
    if t.startswith("str"):
        return val
    v = val.strip()
    try:
        if t.startswith("int"):
            return int(v)
        if t.startswith(("num", "float", "double", "decimal")):
            f = float(v)
            return int(f) if f.is_integer() else f
        if t.startswith("bool"):
            return v.lower() == "true"
        # 【译】（行尾注释）对象 / 数组 / 未知类型
        return json.loads(v)  # object / array / unknown
    # 【译】（行尾注释）noqa 说明：尽力而为的类型转换，失败就返回原值
    except Exception:  # noqa: BLE001 - best-effort coercion, return raw
        return v


# 判断一个未闭合的 <|open|>call tool=... 是被截断的真实调用（名字已声明，后面紧跟参数开始或调用闭合），还是普通文字
def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<|open|>call tool=...` a cut-off call, or prose?

    The gate travels with the unclosed alternative or a sentence that merely
    *quotes* the opener parses as a call.
    """
    # 【译】这个未闭合的 `<|open|>call tool=...` 是一个被截断的调用，还是普通文字？
    #
    #   这道闸门伴随未闭合分支，否则一句仅仅*引用*了开始标记的话会被解析成一个调用。
    if not declared_tools_allow(name, param_types):
        return False
    rest = body.lstrip()
    return (not rest and at_end) or continues_a_call(
        rest, _CALL_CONTINUES, arrived=not at_end
    )


# Kimi-K3 格式的解析器类：16个 START_MARKERS 中只有 call 前缀和 tools 段开始会打开区域，其余都是被丢弃的框架token
class KimiK3Parser(ToolCallParser):
    """Kimi-K3 channel format: buffer the tools section, parse + emit at flush.

    A tools section is parsed whole because K3's arguments interleave and a
    partial one emits garbage. The *response* channel is not a tools section
    and streams as it arrives: it is plain text wrapped in framing this format
    also removes when it is not streaming.

    Buffering everything was simpler and was justified by "the outputs are
    short". Every K3 answer opens with `<|open|>response<|sep|>`, which is one
    of the markers below, so that read as a tool region and the whole body
    arrived in one frame at EOS -- measured on a 324-character answer, 324 of
    them. It is the common path for this model, not an edge case.
    """

    # 【译】Kimi-K3 频道格式：缓冲tools段，在flush时解析并发出。
    #
    #   tools段被整体解析，因为 K3 的参数是交错的，解析一部分会发出垃圾。*response* 频道不是tools段，它随到随流式输出：
    #   它是被框架包着的普通文本，这些框架在非流式时也由本格式删除。
    #
    #   缓冲一切更简单，理由是"输出都很短"。每个 K3 回答都以 `<|open|>response<|sep|>` 开头，而它是下面的标记之一，
    #   于是它被读成了一个工具区域，整个正文在EOS时一帧到达——在一个324字符的回答上实测，324个字符全是这样。
    #   对这个模型来说这是常见路径，不是边缘情况。
    # 解释：第一句"在flush时解析"已不准确——最后一个 <|close|>call 之后的 <|close|>tools 一到，基类 region_end 就判定区域闭合并发出，
    #   不必等到flush（只有没有 tools 段包裹的裸调用才会一直缓冲到流结束）

    NAME: ClassVar[str] = "kimi_k3"
    # Every token `parse` strips from content, plus the call prefix that opens
    # a region. Derived from `_K3_CONTENT_FRAMING` rather than written out
    # again: the read-ahead must not split a token the stripper removes, and a
    # hand-kept second copy of the list is how four of them came to be missing
    # -- they reached the client verbatim when streamed and were deleted when
    # not. `is_kimi_k3` keys on five of these; this is not that list.
    # 【译】`parse` 从content中剥掉的每个token，加上打开区域的调用前缀。由 `_K3_CONTENT_FRAMING` 推导，而不是再写一遍：
    #   预读不能切开剥离器要删除的token，而一份手工维护的第二份列表副本正是其中四个缺失的原因——它们流式时原样到达客户端，
    #   非流式时被删掉。`is_kimi_k3` 依据的是其中五个；这里不是那份列表。
    # 解释："parse 剥掉的token"是历史说法，现在 parse() 已不存在，框架token由 stream.py 在 opens_region 返回False时丢弃
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        KIMI_K3_CALL_PREFIX,
        *_K3_CONTENT_FRAMING,
    )
    # The two that mean a tool call is coming. Every other marker above is
    # channel framing that wraps every answer this model gives, so they are
    # literals the read-ahead must not split and nothing more.
    # 【译】表示有工具调用要来的那两个。上面其他每个标记都是包裹这个模型每个回答的频道框架，所以它们只是预读不能切开的字面量，仅此而已。
    _REGION_MARKERS: ClassVar[frozenset[str]] = frozenset(
        {KIMI_K3_CALL_PREFIX, KIMI_K3_TOOLS_START}
    )
    # The tools channel closing after the last call. Framing would drop it
    # anyway once it is handed back, but naming it here keeps the newline
    # between the two tokens out of the answer.
    # 【译】最后一个调用之后关闭的tools频道。一旦被交还，框架处理反正也会丢掉它，但在这里把它列出来，可以让两个token之间的换行不进入回答。
    # 解释：这段注释说的是下一行的 CALL_CLOSERS（TOOLS_END）；CALL_OPENERS 是 tools 段的开始标记
    CALL_OPENERS: ClassVar[tuple[str, ...]] = (k3.TOOLS_START,)
    # With the separator, because that is where the model puts it. Spelled
    # without it, `markup_end` stopped on the separator and
    # `<|sep|><|close|>tools<|sep|>` reached the client as the answer between
    # two adjacent calls. Not `CALL_FILLERS = (k3.BARE_SEPARATOR,)`, which
    # would fix the same leak: that tuple feeds the *call*-level walkers too
    # and moves every K3 call's span.
    # 【译】带着分隔符，因为模型就把它放在那里。不带分隔符拼写时，`markup_end` 停在分隔符上，
    #   `<|sep|><|close|>tools<|sep|>` 作为两个相邻调用之间的回答到达了客户端。不用 `CALL_FILLERS = (k3.BARE_SEPARATOR,)`，
    #   虽然它也能修好同样的泄漏：那个元组也供*调用*级别的扫描函数使用，会移动每个 K3 调用的span。
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = k3.both_spellings(k3.TOOLS_END)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = k3.both_spellings(k3.CALL_END)

    # 用本格式写出一次调用（只供测试生成语料，与 K3 官方编码器逐字节一致）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        """Byte for byte what `/data/Kimi-K3/encoding_k3.py` emits.

        It used to emit a reduction: no `index`, no `type`, no separator after
        any closer, no `<|close|>tools` at all. This is the sole source of the
        K3 corpus, so every K3 property was asked about a shape the model
        never writes -- `_k3_coerce`'s `type=` branch was unreachable from it.

        `type="string"` because the signature takes `dict[str, str]`; the
        model chooses per value in `_xtml_type`.
        """
        # 【译】与 `/data/Kimi-K3/encoding_k3.py` 输出的内容逐字节相同。
        #
        #   它过去输出的是一个简化版：没有 `index`、没有 `type`、任何闭合标记后都没有分隔符，也完全没有 `<|close|>tools`。
        #   这是 K3 测试语料的唯一来源，所以每个 K3 性质测试问的都是模型从来不会写出的形状——`_k3_coerce` 的 `type=` 分支从它这里根本走不到。
        #
        #   用 `type="string"` 是因为函数签名接收的是 `dict[str, str]`；模型在 `_xtml_type` 中为每个值选择类型。
        sep = k3.BARE_SEPARATOR
        body = "".join(
            f'<|open|>argument key="{k}" type="string"{sep}{v}{k3.ARGUMENT_END}{sep}'
            for k, v in args.items()
        )
        return (
            f'{k3.TOOLS_START}<|open|>call tool="{name}" index="0"{sep}'
            f"{body}{k3.CALL_END}{sep}{k3.TOOLS_END}{sep}"
        )

    # 只有 call 前缀和 tools 段开始会打开工具调用区域，其余 START_MARKERS 都是框架token
    @classmethod
    def opens_region(cls, marker: str) -> bool:
        return marker in cls._REGION_MARKERS

    # 识别规则见 is_kimi_k3
    @classmethod
    def detect(cls, text: str) -> bool:
        return is_kimi_k3(text)

    # 解析一个区域：逐个匹配 <|open|>call ...，取名字和带类型的参数，按 type 转换后组装成JSON参数
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """Every complete call in the section, and where the section ends.

        A call cut off by `max_tokens` has no `<|close|>call` and so parses to
        nothing, which means the region was not a call after all and its bytes
        are released unchanged -- the same answer this format now gives to an
        answer that merely quotes the opener. Both used to be decided here, by
        a second opener regex that accepted shapes `_K3_CALL_RE` rejects, and
        the two ways of getting it wrong were opposite: a quotation lost 62
        characters, and a truncated call kept its half-written payload with the
        dangling `<|close|>argument` still in it.
        """
        # 【译】tools段中每个完整的调用，以及这个段在哪里结束。
        #
        #   被 `max_tokens` 截断的调用没有 `<|close|>call`，所以什么也解析不出，这意味着这个区域终究不是调用，它的字节被原样释放——
        #   这与本格式现在对一个仅仅引用了开始标记的回答给出的答案相同。两者过去都在这里判定，靠的是第二个开始标记正则，
        #   它接受 `_K3_CALL_RE` 拒绝的形状，而两种出错方式正好相反：一段引用丢了62个字符，一个被截断的调用则保留了写到一半的载荷，
        #   里面还带着悬空的 `<|close|>argument`。
        # 解释：与代码不符——现在 _K3_CALL_RE 有未闭合分支，一个被截断的调用只要通过 _is_truncated_call
        #   （名字已声明，且后面紧跟参数开始标记或调用闭合）仍会被当作调用；只有通不过检验的才原样释放
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _K3_CALL_RE.finditer(region):
            # Before the truncation gate, which only runs for an unclosed
            # call: `tool="(?P<name>[^"]*)"` matches an empty and an all-space
            # name, and a *closed* one had nothing to stop it.
            # 【译】放在截断闸门之前，因为闸门只对未闭合的调用运行：`tool="(?P<name>[^"]*)"` 能匹配空名字和全空格名字，而一个*已闭合*的调用没有任何东西拦住它。
            if not usable_tool_name(m.group("name")):
                continue
            if m.group("closed") is None and not _is_truncated_call(
                m.group("name"), m.group("body"), param_types, at_end=at_end
            ):
                continue
            args: dict = {}
            for a in _K3_ARG_RE.finditer(m.group("body")):
                args[a.group("key")] = _k3_coerce(a.group("val"), a.group("type"))
            spans.append(
                (cls.markup_begin(region, m.start()), cls.markup_end(region, m.end()))
            )
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": m.group("name"),
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        return RegionParse(tuple(tool_calls), tuple(spans))
