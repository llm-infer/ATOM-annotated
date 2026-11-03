# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One reading of a model's tool-call syntax, driven by one engine.

A format used to be read twice: ``parse`` took a complete output and the
``process``/``flush`` state machine took it a chunk at a time. Both had to
decide where content ends, whether an unclosed tag is a call or a quotation,
what to do with a region that parses to nothing, and which bytes are framing --
so every one of those rules existed six times over, once per format, in two
copies. Three rounds of review found the copies disagreeing about all four,
and every disagreement is a response a client gets one way with
``stream=false`` and another with ``stream=true``.

So a format now declares only what is particular to it:

``START_MARKERS``     literals that must not be split across a chunk boundary
``opens_region``      which of those hand the stream over (the rest are framing)
``parse_region``      what one region's bytes mean -- the calls, and the two
                      offsets that bracket the format's own markup

and :class:`ToolCallStreamParser` -- the engine -- owns everything else:
reading ahead of the region, releasing content, dropping framing, the rule
that a start marker is not a promise, stamping call indices, and handing back
whatever followed the markup. ``stream=false`` is the engine run over a single
chunk, so the two modes cannot disagree: there is nothing left for them to
disagree with.
"""

# 【译】对模型工具调用语法的唯一一种读法，由唯一一个引擎驱动。
#
#   过去一种格式要被读两遍：``parse`` 接收完整输出，``process``/``flush`` 状态机则一次接收一个chunk。
#   两者都得判断：普通内容在哪里结束、一个没闭合的标签是调用还是引用、一个解析不出任何东西的区域该怎么办、
#   哪些字节是框架标记——于是每条规则都存在六份（每种格式一份），而且每份还有两个副本。
#   三轮review发现这些副本在这四点上全都不一致，而每一处不一致都意味着客户端在 ``stream=false``
#   和 ``stream=true`` 下拿到不同的响应。
#
#   所以现在一种格式只声明它特有的东西：
#     ``START_MARKERS``  不能被chunk边界切开的字面量
#     ``opens_region``   其中哪些会把流移交给工具调用解析（其余是框架标记）
#     ``parse_region``   一个区域的字节是什么意思——解析出的调用，以及框住本格式自身标记的两个偏移量
#   而 :class:`ToolCallStreamParser`（即"引擎"）负责其余一切：在区域之前预读、释放普通内容、丢弃框架标记、
#   "起始标记不等于承诺"这条规则、给调用编号，以及交还标记之后的内容。``stream=false`` 就是引擎在单个chunk上
#   跑一遍，所以两种模式不可能不一致：已经没有什么东西能让它们产生分歧了。
#
# 【导读】工具调用解析器（tool parser）家族的基类与公共判定函数（家族总览见本目录 __init__.py）
# 本文件定义：
#   ToolCall        OpenAI格式的一次工具调用（id / type="function" / function={name, arguments(JSON字符串)}）
#   RegionParse     一种格式对一个"工具调用区域"的解析结果：calls + 属于标记的spans
#   ToolCallParser  每种线上格式的抽象基类：只有类属性和类方法，从不实例化，没有逐请求状态
#   以及各格式共用的小函数：工具名是否合法（usable_tool_name）、是否属于请求声明的工具（declared_tools_allow）、
#   名字后面是否紧跟本格式的下一个记号（continues_a_call）、把调用的匹配范围扩展到外层包裹符（begin/end_of_markup）
# 调用方：各模型的 *_tool_parser.py 继承 ToolCallParser 并调用这些函数；stream.py 的 ToolCallStreamParser
#   调用 opens_region / region_end / parse_region；registry.py 调用 detect；serving_chat.py 和 schema.py 调用 qualified_tool_name
# 与主干的关系：属于第2批服务层的协议适配，只处理detokenize之后的文本，不涉及EngineCore进程、调度与KV cache

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

# A tool name is an identifier, which is what the model was given.
#
# `\w` and not `[A-Za-z_]`: OpenAI's own grammar is `^[a-zA-Z0-9_-]{1,64}$`, so
# a leading digit is legal (`7z_extract`), and `\w` is Unicode-aware, so a CJK
# name is too -- which matters rather a lot on a Chinese model family.
# Rejecting one is silent. `.` and `-` in the tail because MCP servers
# namespace theirs `server.tool`; prose is still rejected because a space
# cannot appear anywhere. `\Z` and not `$`, which also matches before a
# trailing newline and would admit a name with one.
# 【译】工具名是一个标识符，因为模型拿到的就是标识符。
#
#   用 `\w` 而不是 `[A-Za-z_]`：OpenAI 自己的语法是 `^[a-zA-Z0-9_-]{1,64}$`，所以数字开头是合法的（`7z_extract`）；
#   而且 `\w` 支持 Unicode，所以中日韩文字的名字也合法——这对一个中文模型家族相当重要。
#   拒绝一个名字是静默发生的。尾部允许 `.` 和 `-`，因为 MCP 服务器会把自己的工具命名为 `server.tool`；
#   普通文字仍会被拒绝，因为任何位置都不能出现空格。用 `\Z` 而不是 `$`：`$` 还能匹配末尾换行符之前的位置，
#   会放过一个带换行的名字。
# 解释：可选的 `::xxx` 后缀对应 qualified_tool_name 拼出的 "namespace::name" 形式
_TOOL_NAME_RE = re.compile(r"^\w[\w.\-]*(?:::\w[\w.\-]*)?\Z")


# 判断调用名之后紧跟的文本是不是本格式的下一个记号（记号由各格式提供，判定逻辑只此一份）
def continues_a_call(rest: str, tokens: tuple[str, ...], *, arrived: bool) -> bool:
    """Is what follows a call's name this format's own next token?

    One implementation, because four formats had a copy and the copies were
    the point of failure: each format supplies the tokens, this decides.

    `arrived` is `parse_region`'s `at_end`, inverted, and it is the whole of
    the difference between the two moments that function is asked about. At
    end of region a token cut off part-way through is all there will ever be,
    so a *prefix* counts -- that is what a call truncated by `max_tokens`
    looks like. Mid-region, where the early name is read, a prefix means "not
    yet" and more is coming; accepting one there let a chunk boundary landing
    one character into `<br>` announce a tool for prose, since `<` is a prefix
    of `<parameter=`. Same text, same parser: announced at one chunk size,
    silent at another.
    """
    # 【译】调用名之后跟着的，是不是这种格式自己的下一个记号？
    #
    #   只有一份实现，因为原来有四种格式各有一份副本，而副本正是出错之处：每种格式提供记号，由这里来判定。
    #
    #   `arrived` 就是 `parse_region` 的 `at_end` 取反，它是该函数被询问的两个时刻之间的全部差别。
    #   在区域结束时，一个只写了一半的记号就是它的全部了，所以*前缀*也算数——被 `max_tokens` 截断的调用就长这样。
    #   而在区域中途（读取提前播报的工具名的时候），前缀意味着"还没到"、后面还有更多；在那里接受前缀，
    #   曾让一个恰好落在 `<br>` 第一个字符之后的chunk边界为一段普通文字宣布出一个工具，因为 `<` 是 `<parameter=` 的前缀。
    #   同样的文本、同样的解析器：在一种chunk大小下宣布了调用，在另一种下却没有。
    # 解释：arrived=True（区域中途）要求 rest 以某个完整记号开头；arrived=False（区域已结束）时 rest 只要是某个记号的前缀即可
    if arrived:
        return any(rest.startswith(tok) for tok in tokens)
    return any(tok.startswith(rest[: len(tok)]) for tok in tokens)


# 请求声明的工具列表能否排除这个名字：声明了列表就要求名字在其中，没声明（param_types为空）则一律放行
def declared_tools_allow(name: str, param_types: dict) -> bool:
    """Can the request's declared tools rule this name out?

    Only when it declared any. Every format's truncation gate asks whether the
    name is one the request listed, because prose quoting an opener usually
    invents one -- but a request with no ``tools`` field declares nothing, and
    six formats read that as "no name is real". A call cut off by
    `max_tokens` then went out as raw special tokens in ``content`` while the
    *same* generation, one byte longer, was read as a call: the answer turned
    on whether the client had listed its tools rather than on what the model
    wrote. Agent harnesses that describe their tools in the system prompt hit
    it on every truncation.

    So: a declared list refutes a name it does not contain, and an absent list
    refutes nothing. Whatever follower test the format applies still runs --
    that is the evidence which does not come from the request.
    """
    # 【译】请求声明的工具能否排除这个名字？
    #
    #   只有在它声明了工具时才能。每种格式的截断判定都会问"这个名字是不是请求列出的"，因为引用起始标记的普通文字
    #   通常会编造一个名字——但没有 ``tools`` 字段的请求什么也没声明，而六种格式都把这理解成"没有一个名字是真的"。
    #   于是一个被 `max_tokens` 截断的调用会以原始特殊token的形式出现在 ``content`` 里，而*同一个*生成结果只要
    #   多一个字节，就会被读成调用：结果取决于客户端有没有列出工具，而不是模型写了什么。
    #   在系统提示词里描述工具的agent框架，每次截断都会碰到这个问题。
    #
    #   所以：声明的列表可以否定不在其中的名字，缺省的列表什么也不否定。格式自己施加的后续记号测试仍然会运行——
    #   那是不来自请求的证据。
    return not param_types or name in param_types


# 名字本身是否可以分发给客户端：非空且匹配 _TOOL_NAME_RE
def usable_tool_name(name: str | None) -> bool:
    """Is this something a client could dispatch?

    One predicate, because six formats answered it six ways and three of them
    did not ask at all. `<invoke name="">` and `<invoke name="   ">` -- both of
    which `name="([^"]*)"` matches -- reached the client as a call whose name
    was the empty string: not dispatchable, and matching nothing the request
    declared. On `/v1/messages` it becomes a `tool_use` block with an empty
    `name`, which the Anthropic SDK surfaces as a tool the caller never
    registered.

    The three spelled their non-answers differently, which is the tell that
    the axis and not any one format is the thing to fix: DSML and K3 asked
    nothing, and MiniMax asked `if not name` *before* `name.strip()`, so a
    name of pure whitespace passed the guard and then became empty.

    Separate from `declared_tools_allow`, which asks whether the *request*
    knows this name. This asks whether it is a name at all, and holds for a
    request that declared no tools -- the case where that one deliberately
    refutes nothing.
    """
    # 【译】这是不是客户端可以分发（执行）的东西？
    #
    #   只用一个判定，因为六种格式有六种回答，其中三种根本没问。`<invoke name="">` 和 `<invoke name="   ">`
    #   （`name="([^"]*)"` 都能匹配它们）会作为名字为空字符串的调用到达客户端：无法分发，也不匹配请求声明的任何工具。
    #   在 `/v1/messages` 上它会变成一个 `name` 为空的 `tool_use` 块，Anthropic SDK 会把它呈现为一个调用方从未注册过的工具。
    #
    #   这三种格式"不回答"的写法各不相同，这说明该修的是这个维度本身，而不是某一种格式：DSML 和 K3 什么都不问，
    #   MiniMax 在 `name.strip()` *之前*问 `if not name`，所以纯空白的名字通过了检查，然后才变成空串。
    #
    #   它与 `declared_tools_allow` 是分开的，后者问的是*请求*是否认识这个名字。这里问的是它到底算不算一个名字，
    #   并且对没有声明任何工具的请求同样适用——那正是后者刻意什么也不否定的情形。
    return bool(name) and _TOOL_NAME_RE.match(name) is not None


# 把工具定义中可选的namespace与函数名合成 "namespace::name" 形式的完整工具名并校验，非法时抛ValueError
#   调用方：serving_chat.py（校验请求的tools / tool_choice）、schema.py（按工具名索引参数类型）
def qualified_tool_name(tool: dict) -> str:
    """Resolve an optional namespace to the OpenAI function-name identity.

    Accept one namespace, either beside or inside function, and reject a
    conflicting qualified name. This identity is shared by request validation
    and schema lookup; the prompt encoder still receives the original schema.
    """
    # 【译】把可选的命名空间解析成 OpenAI 函数名意义下的身份标识。
    #
    #   接受一个命名空间，它可以写在 function 旁边或里面；与之冲突的已限定名字会被拒绝。
    #   请求校验和schema查找共用这个身份标识；prompt编码器收到的仍是原始schema。
    fn = tool.get("function", tool)
    name = fn.get("name")
    namespace = tool.get("namespace")
    if namespace is None:
        namespace = fn.get("namespace")
    if isinstance(namespace, dict):
        if not isinstance(namespace.get("name"), str):
            raise ValueError("tool namespace must have a string name")  # noqa: TRY004
        namespace = namespace["name"]
    if namespace is not None:
        if (
            not isinstance(namespace, str)
            or not usable_tool_name(namespace)
            or "::" in namespace
        ):
            raise ValueError(
                "tool namespace must be a nonempty identifier without '::'"
            )
        if isinstance(name, str) and "::" in name:
            prefix, _, _ = name.partition("::")
            if prefix != namespace:
                raise ValueError("conflicting tool namespaces")
        else:
            name = f"{namespace}::{name}" if isinstance(name, str) else None
    if not isinstance(name, str) or not usable_tool_name(name):
        raise ValueError(
            "function.name must be a dispatchable name: word characters, dots "
            "or dashes, optionally qualified by one namespace with '::'"
        )
    return name


# 生成一次工具调用的id：call_<uuid4十六进制>，各格式的解析器构造ToolCall时调用
def unique_tool_call_id() -> str:
    # OpenAI tool_call ids must be unique across the whole conversation, not just
    # within one response. A per-response index (call_0, call_1, ...) collides
    # across turns -> clients (e.g. qwen-code) dedupe by id and silently ignore
    # every repeat, causing an infinite tool-call retry loop. Use a random id.
    # 【译】OpenAI 的 tool_call id 必须在整个对话中唯一，而不只是在一个响应内唯一。按响应内序号编号（call_0、call_1……）
    #   在多轮之间会冲突 -> 客户端（例如 qwen-code）按id去重，会静默忽略每一个重复的调用，导致无限的工具调用重试循环。
    #   所以用随机id。
    return f"call_{uuid.uuid4().hex}"


# 从一次调用自身匹配的起点 at 向左越过本格式的开头包裹符（openers）、填充符（fillers）及其间的空白，返回标记真正的起点
def begin_of_markup(
    region: str,
    at: int,
    openers: tuple[str, ...] = (),
    fillers: tuple[str, ...] = (),
) -> int:
    """Pull `at` back over the wrapper a format opens its calls with.

    The mirror of :func:`end_of_markup`, and needed for the same reason: a
    call's own match starts at `<function=`, and the `<tool_call>` before it is
    markup too. Without it every real call left its wrapper in the answer.

    It also draws the line an answer that *quotes* a marker before making a
    real call depends on. A region opens at the first marker in the text, so
    the sentence between a quotation and the call that follows it is inside
    the region -- and it stops here, because prose is not one of the literals
    this walks over.

    Offsets, not slices. `region[:j].rstrip()` copies everything to the left
    of the scan to read its last few bytes, once per step, so a region holding
    several calls after a large argument paid O(calls x region) -- measured
    7.3x on 128 KB with eight calls, and growing with the payload where this
    is flat. `end_of_markup` walks the other direction with `startswith(s, j)`
    and never copied; the same fix is already written out on
    `_region_owes_a_closer`, one layer up, and was not carried into the
    function that shape came from.
    """
    # 【译】把 `at` 往回拉，越过格式在调用前面写的包裹符。
    #
    #   与 :func:`end_of_markup` 对称，需要它的原因也相同：一次调用自身的匹配从 `<function=` 开始，
    #   而它前面的 `<tool_call>` 也是标记。没有这一步，每个真实的调用都会把包裹符留在答案里。
    #
    #   它还划出了一条界线，"先*引用*一个标记、再做真实调用"的答案要靠这条界线。区域从文本中第一个标记处开始，
    #   所以引用与其后的调用之间的那句话位于区域之内——而扫描在这里停下，因为普通文字不是它会越过的字面量。
    #
    #   用偏移量，不用切片。`region[:j].rstrip()` 为了读最后几个字节，每一步都把扫描点左边的全部内容复制一遍，
    #   所以一个在大参数之后含有多个调用的区域要付出 O(调用数 x 区域长度)——实测128KB、8个调用时慢7.3倍，
    #   并且随负载增长，而这里的开销是平的。`end_of_markup` 用 `startswith(s, j)` 往另一个方向走，从不复制；
    #   同样的修法已经写在上一层的 `_region_owes_a_closer` 里，却没有带回到那个写法所源自的这个函数中。
    if not openers:
        return at
    begin = j = at
    while j > 0:
        end = j
        while end > 0 and region[end - 1].isspace():
            end -= 1
        if end < j:
            # Whitespace moves the scan but not the answer, exactly as in
            # `end_of_markup`: a newline between two markers is markup, a
            # newline between prose and a marker belongs to the prose.
            # 【译】空白移动扫描点，但不移动答案，与 `end_of_markup` 完全一样：两个标记之间的换行属于标记，
            #   普通文字与标记之间的换行属于普通文字。
            j = end
            continue
        token = next(
            (t for t in (*fillers, *openers) if region.endswith(t, 0, j)),
            None,
        )
        if token is None:
            break
        j -= len(token)
        begin = j
    return begin


# 从一次调用自身匹配的终点 at 向右越过本格式的结尾包裹符（closers），返回标记真正的终点
def end_of_markup(
    region: str,
    at: int,
    closers: tuple[str, ...] = (),
    fillers: tuple[str, ...] = (),
) -> int:
    """Extend `at` over the wrapper a format closes its calls with.

    A call's own match ends at `</function>`; the `</tool_call>` after it is
    markup too, and so is the newline between them. Without this the wrapper
    was either left in the answer or swallowed along with everything after it,
    depending on which format's ``parse`` you read.

    Whitespace and `fillers` (MiniMax repeats its namespace token before every
    tag) advance the *scan* but never the answer: only a closer moves `end`,
    so the newline a model writes before resuming prose survives.
    """
    # 【译】把 `at` 向后延伸，越过格式在调用后面写的包裹符。
    #
    #   一次调用自身的匹配在 `</function>` 结束；其后的 `</tool_call>` 也是标记，两者之间的换行也是。
    #   没有这一步，包裹符要么被留在答案里，要么连同它后面的一切一起被吞掉，取决于你读的是哪个格式的 ``parse``。
    #
    #   空白和 `fillers`（MiniMax 在每个标签前都重复它的命名空间token）只推进*扫描*，从不推进答案：
    #   只有结尾包裹符会移动 `end`，所以模型在恢复写普通文字之前写下的换行会被保留。
    if not closers:
        return at
    end = j = at
    n = len(region)
    while j < n:
        if region[j].isspace():
            j += 1
            continue
        skipped = next((s for s in fillers if region.startswith(s, j)), None)
        if skipped is not None:
            j += len(skipped)
            continue
        closer = next((c for c in closers if region.startswith(c, j)), None)
        if closer is None:
            break
        j += len(closer)
        end = j
    return end


# OpenAI格式的一次工具调用；arguments 是JSON字符串（各格式用 json.dumps 生成）
@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    # 【译】已解析的、OpenAI格式的工具调用。

    id: str
    type: str
    function: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


# 一种格式对一个区域的解析结果：calls 是解析出的调用；spans 是区域内属于标记、要从答案中删掉的若干 [begin, end) 区间
@dataclass(frozen=True)
class RegionParse:
    """What one format made of a region's bytes.

    ``spans`` is every stretch of this region that is the format's own markup,
    in ascending order and not overlapping: everything inside a span is
    markup, everything outside every span is answer. The region is accounted
    for byte by byte and no format has to remember to return the leftovers.

    Several spans and not one outer pair, because a region can hold more than
    one call and a model writes between them -- "let me also check Rome"
    between two `<tool_call>` blocks is the ordinary shape of a parallel-call
    answer. A single outer pair calls that sentence markup and deletes it. Kimi-K2 is
    the exception, and only because it is the one format whose
    `REGION_END_MARKERS` close a region per section, so the question never
    reached here -- which is also why it still returns a single span covering
    its whole section, where the others return one per call.

    Each span is a call's own match widened by :meth:`ToolCallParser.
    markup_begin` / :meth:`~ToolCallParser.markup_end`, which walk over that
    format's wrappers and fillers and stop at prose -- so the wrapper opening
    the first call and the one closing the last belong to those calls' spans,
    and whitespace between two markers is markup while whitespace between
    prose and a marker is not.

    ``calls`` empty means the region was a quotation rather than a call, and
    the engine releases its bytes unchanged; ``spans`` is then ignored.
    """

    # 【译】一种格式对一个区域字节的解读。
    #
    #   ``spans`` 是这个区域中属于本格式自身标记的每一段，按升序排列且互不重叠：span 内的一切都是标记，
    #   所有 span 之外的一切都是答案。区域按字节逐个记账，没有哪个格式需要记得返回剩余部分。
    #
    #   用多个span而不是一对最外层边界，是因为一个区域可以包含多个调用，而模型会在它们之间写字——
    #   在两个 `<tool_call>` 块之间写"让我也查一下罗马"，是并行调用回答的常见形状。单个最外层边界会把这句话当作标记删掉。
    #   Kimi-K2 是例外，而且只是因为它是唯一一个 `REGION_END_MARKERS` 按section关闭区域的格式，所以这个问题从来不会到达这里——
    #   这也是为什么它仍然返回覆盖整个section的单个span，而其他格式每个调用返回一个。
    #
    #   每个span是一次调用自身的匹配，再经 :meth:`ToolCallParser.markup_begin` / :meth:`~ToolCallParser.markup_end` 扩宽，
    #   二者会越过本格式的包裹符和填充符并在普通文字处停下——所以打开第一个调用的包裹符和关闭最后一个调用的包裹符
    #   都属于这些调用的span；两个标记之间的空白是标记，而普通文字与标记之间的空白不是。
    #
    #   ``calls`` 为空表示这个区域是一段引用而不是调用，引擎会原样释放它的字节；此时 ``spans`` 被忽略。

    calls: tuple[ToolCall, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()

    @property
    def begins(self) -> int:
        """Where the first call's markup starts. Derived, so that the answer
        cannot drift from ``spans``."""
        # 【译】第一个调用的标记从哪里开始。它是推导出来的，这样它就不会和 ``spans`` 不一致。
        return self.spans[0][0] if self.spans else 0

    @property
    def consumed(self) -> int:
        """Where the last call's markup ends."""
        # 【译】最后一个调用的标记在哪里结束。
        return self.spans[-1][1] if self.spans else 0


# 空解析结果（没有调用）的共享实例；atom/ 内目前没有引用它
NO_CALLS = RegionParse()


# 每种线上（on-the-wire）工具调用格式的抽象基类：子类声明下面这些标记字面量并实现 parse_region
#   全部是类方法、从不实例化；逐请求的状态都在 stream.py 的 ToolCallStreamParser（引擎）里
class ToolCallParser(ABC):
    """One on-the-wire tool-call format.

    Entirely class-side. A format holds no per-request state -- the engine
    does -- so there is nothing here for two requests to share or for one
    region to leak into the next.
    """

    # 【译】一种线上的工具调用格式。
    #
    #   完全在类层面实现。格式不持有任何逐请求的状态——状态在引擎里——所以这里没有任何东西会被两个请求共享，
    #   也不会从一个区域泄漏到下一个。

    NAME: ClassVar[str]
    # Every literal that opens this format's tool-call region, plus every
    # literal it treats as framing. Declared once, here, rather than spelled
    # out again in each parser's own logic: the read-ahead needs them to know
    # how much of its buffer could still be the start of one, and the property
    # tests enumerate them so a newly registered format is covered the moment
    # it exists rather than when someone writes a case.
    # 【译】打开本格式工具调用区域的每一个字面量，加上它当作框架标记的每一个字面量。在这里统一声明一次，
    #   而不是在每个解析器自己的逻辑里再拼写一遍：预读需要它们来判断缓冲区末尾还有多少可能是某个标记的开头；
    #   属性测试会枚举它们，这样新注册的格式一出现就被覆盖，而不必等有人去写测试用例。
    START_MARKERS: ClassVar[tuple[str, ...]] = ()
    # Literals that can close a region mid-stream, so the engine knows from
    # the chunk alone whether to ask `region_end`. Empty means "the closers
    # below", which is right for every format but Kimi, whose region is a
    # section of several entries and whose entry end is not a section end.
    # 【译】能在流中途关闭区域的字面量，让引擎仅凭当前chunk就知道要不要去问 `region_end`。为空表示"用下面的closers"，
    #   这对除 Kimi 以外的所有格式都正确：Kimi 的区域是包含多个条目的一个section，条目结束不等于section结束。
    # 解释：为空时 stream.py 用 CALL_SELF_CLOSERS + CALL_CLOSERS 作为触发字面量
    REGION_END_MARKERS: ClassVar[tuple[str, ...]] = ()
    # The wrapper literals this format writes around its calls -- before the
    # first and after the last -- and any token it repeats between tags. All
    # markup; see `begin_of_markup` and `end_of_markup`.
    # 【译】本格式写在调用周围（第一个调用之前、最后一个调用之后）的包裹字面量，以及在标签之间重复出现的token。
    #   全部属于标记；见 `begin_of_markup` 和 `end_of_markup`。
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ()
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ()
    CALL_FILLERS: ClassVar[tuple[str, ...]] = ()
    # What closes a *call*, where `CALL_CLOSERS` closes the wrapper around
    # one. Telling them apart is the difference between prose and a dispatch:
    # `<tool_call><function=NAME></tool_call>` has the wrapper closed and the
    # call still open, so it is a model describing the syntax. Declared so
    # the suite can build that shape for every format rather than for the one
    # somebody wrote out. Empty where a call *is* its wrapper.
    # 【译】关闭一次*调用*的字面量，而 `CALL_CLOSERS` 关闭的是调用外面的包裹。区分二者就是区分普通文字与一次真正的分发：
    #   `<tool_call><function=NAME></tool_call>` 包裹已关闭而调用仍未关闭，所以这是模型在描述语法。
    #   声明它，是为了让测试套件能为每种格式构造这种形状，而不只是为某个人写出来的那一种。
    #   如果一次调用本身就是它的包裹，则为空。
    # 解释：与代码不完全相符——GLM 的调用就是它的包裹，但它声明了 CALL_SELF_CLOSERS=("</tool_call>",)，
    #   为空的是 CALL_CLOSERS；而且 region_end 在 CALL_SELF_CLOSERS 为空时永远返回0
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ()
    # The literals that *identify* this format, where that is narrower than
    # "what must not be split". Empty means the two coincide, as they do for
    # five of the six.
    #
    # 【译】*标识*本格式的字面量，当它比"不能被切开的字面量"更窄时才需要声明。为空表示两者一致，六种格式里有五种是这样。
    # They came apart because DSML matches its `｜DSML｜` marker optionally --
    # the V4-Flash malform drops it -- so `<invoke name=` is a start marker,
    # and that is also how MiniMax opens a call. DSML claimed MiniMax's
    # templates and only `_DETECT_ORDER` kept it from winning them.
    #
    # Leniency about what to *parse* stays: once identified, a format should
    # read the malforms its own model emits. Only the claim must be single.
    # 【译】二者之所以分开，是因为 DSML 把 `｜DSML｜` 标记当作可选来匹配——V4-Flash 的畸形输出会丢掉它——
    #   所以 `<invoke name=` 也是一个起始标记，而这恰好也是 MiniMax 开始一次调用的方式。
    #   DSML 曾经认领了 MiniMax 的模板，只是靠 `_DETECT_ORDER` 才没让它得逞。
    #
    #   对*解析*的宽容保持不变：一旦确定了格式，它就应该能读懂自己模型产生的畸形输出。只有"认领"必须是唯一的。
    # 解释：现在共有七种格式（多了 dsml_v41），DsmlParser 和 DsmlV41Parser 都声明了 DETECT_MARKERS
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = ()

    # 某个起始标记是否打开工具调用区域（把后续的流移交给本格式）；返回False的标记是框架标记，由引擎直接丢弃
    @classmethod
    def opens_region(cls, marker: str) -> bool:
        """Does this marker hand the rest of the stream to this format?

        `START_MARKERS` answers "which literals must not be split across a
        chunk boundary", and for most formats every one of them also opens a
        tool-call region, so the two questions have the same answer and this
        is that default. Kimi-K3 is where they come apart: three of its five
        are channel framing that wraps *every* answer, tool call or not, and
        treating those as a handover stopped it streaming at all.

        A marker this says no to is framing: the reader drops it and carries
        on. That is the only place framing is removed, on either delivery
        path -- Kimi-K3 used to strip its own list a second time inside
        ``parse``, and the two lists disagreed about four tokens.
        """
        # 【译】这个标记会不会把流的剩余部分移交给本格式？
        #
        #   `START_MARKERS` 回答的是"哪些字面量不能被chunk边界切开"，而对大多数格式来说，其中每一个也都会打开一个
        #   工具调用区域，所以两个问题的答案相同，这里的默认实现就是这个答案。Kimi-K3 是两者分开的地方：
        #   它的五个标记中有三个是通道框架，包裹着*每一个*回答，无论有没有工具调用；把它们当作移交标记曾让它完全无法流式输出。
        #
        #   这里回答"否"的标记就是框架标记：读取器丢弃它然后继续。在两条交付路径上，这都是唯一删除框架标记的地方——
        #   Kimi-K3 过去在 ``parse`` 里会再剥离一遍自己的列表，而两份列表对四个token的看法不一致。
        # 解释："五个中有三个"是旧数字——现在 KimiK3Parser.START_MARKERS 有16个（全部通道框架token都列入），
        #   其中只有 call 前缀和 tools 段开始两个会打开区域
        return True

    @classmethod
    def markup_begin(cls, region: str, at: int) -> int:
        """Where this format's markup really starts, given its first call
        started at ``at``. See :func:`begin_of_markup`."""
        # 【译】假定本格式的第一个调用从 ``at`` 开始，求本格式的标记真正从哪里开始。见 :func:`begin_of_markup`。
        return begin_of_markup(region, at, cls.CALL_OPENERS, cls.CALL_FILLERS)

    @classmethod
    def markup_end(cls, region: str, at: int) -> int:
        """Where this format's markup really ends, given its last call ended
        at ``at``. See :func:`end_of_markup`."""
        # 【译】假定本格式的最后一个调用在 ``at`` 结束，求本格式的标记真正在哪里结束。见 :func:`end_of_markup`。
        return end_of_markup(region, at, cls.CALL_CLOSERS, cls.CALL_FILLERS)

    @classmethod
    def find_start(cls, text: str) -> int:
        """Index of the earliest start marker, or -1."""
        # 【译】最早出现的起始标记的下标，没有则为 -1。
        # 解释：atom/ 内目前没有调用方
        positions = [i for i in (text.find(m) for m in cls.START_MARKERS) if i != -1]
        return min(positions) if positions else -1

    # registry.py 在启动时用它在渲染出的chat模板里识别模型使用哪种格式
    @classmethod
    def detect(cls, text: str) -> bool:
        """Is this text this format's, as opposed to merely readable by it?

        Keyed on `DETECT_MARKERS` where a format declares them: the literals
        that must not be split are not always the literals that identify.
        """
        # 【译】这段文本是不是本格式的——而不只是本格式能读懂？
        #
        #   声明了 `DETECT_MARKERS` 的格式按它判断：不能被切开的字面量，并不总是能标识格式的字面量。
        return any(m in text for m in (cls.DETECT_MARKERS or cls.START_MARKERS))

    # 流式时判断区域是否已闭合：返回 region 中构成闭合单元的字节数，0 表示"还没有"（stream.py 在收到区域结束字面量时调用）
    @classmethod
    def region_end(cls, region: str) -> int:
        """How many bytes of ``region`` form a closed unit, or 0 for "not yet".

        Asked only once one of ``REGION_END_MARKERS`` has arrived, so the
        per-chunk cost is a look at the chunk and not at the buffer. A region
        that never answers waits for end of stream and takes everything the
        model wrote after its call with it -- 0 of 397 characters streamed on
        five of six formats, which is the ordinary agentic shape.

        The default answers for all of them, on the one literal that cannot
        appear inside an argument value: the call's own closer, which every
        grammar here lists among a value's terminators. The *wrapper* closer
        can hide in a value, and reading the old note about that as "no safe
        literal exists" is what left four formats buffering to EOS.

        Once the call has closed no value is open, so `end_of_markup` can walk
        the wrapper: the region ends where the markup after the last call
        ends, and stays open while the wrapper does.
        """
        # 【译】``region`` 中有多少字节构成了一个闭合单元，0 表示"还没有"。
        #
        #   只在某个 ``REGION_END_MARKERS`` 已经到达时才会被询问，所以每个chunk的开销只是看一眼chunk，而不是看整个缓冲区。
        #   一个永远不回答的区域会一直等到流结束，并把模型在其调用之后写的一切都带走——六种格式中有五种在397个字符里
        #   流式输出了0个，而这正是普通的agent回答形状。
        #
        #   默认实现能回答所有格式，依据的是唯一一个不可能出现在参数值内部的字面量：调用自己的闭合标记，
        #   这里的每种语法都把它列为参数值的终止符之一。*包裹*闭合标记可以藏在参数值里，而把原来那条关于此事的注释
        #   理解成"不存在安全的字面量"，正是让四种格式一直缓冲到EOS的原因。
        #
        #   一旦调用闭合，就没有打开着的参数值了，所以 `end_of_markup` 可以越过包裹：区域在最后一个调用之后的标记
        #   结束处结束，只要包裹还开着，区域就保持打开。
        # 解释：例如 Qwen 的 region="<tool_call>\n<function=f>...</function>\n</tool_call>"：先找到最后一个"</function>"，
        #   再越过"\n</tool_call>"，返回包裹结束处；若"</tool_call>"还没到，end==at，返回0继续等待
        at = max(
            (i + len(c) for c in cls.CALL_SELF_CLOSERS if (i := region.rfind(c)) >= 0),
            default=0,
        )
        if not at:
            return 0
        end = end_of_markup(region, at, cls.CALL_CLOSERS, cls.CALL_FILLERS)
        # A format with a wrapper owes its closer; one whose call *is* its
        # wrapper (GLM) is done at the call's own.
        # 【译】有包裹的格式还欠一个包裹闭合标记；调用本身就是包裹的格式（GLM）在调用自己的闭合标记处就结束了。
        return 0 if cls.CALL_CLOSERS and end == at else end

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        """One call, in this format's own syntax. The inverse of `parse_region`.

        Nothing in the engine renders; this exists so the test corpus can be
        generated from the registry rather than hand-written. A hand-written
        sample can be wrong in a way that makes every assertion about it
        vacuous -- MiniMax's was written in DSML's spelling and parsed to
        `get_weather({})`, exercising none of the parameter path.

        Not `@abstractmethod`: nothing here is ever instantiated, so ABC would
        not enforce it. `test_every_format_can_write_its_own_syntax` does.
        """
        # 【译】用本格式自己的语法写出一次调用。是 `parse_region` 的逆操作。
        #
        #   引擎里没有任何地方做渲染；它存在的目的是让测试语料可以从注册表生成，而不是手写。手写的样例可能错得
        #   让关于它的每个断言都形同虚设——MiniMax 的样例曾用 DSML 的拼写来写，被解析成 `get_weather({})`，参数路径完全没被测到。
        #
        #   没有加 `@abstractmethod`：这里的类从不实例化，所以 ABC 不会强制它。由 `test_every_format_can_write_its_own_syntax` 来强制。
        raise NotImplementedError(f"{cls.NAME} cannot render its own syntax")

    # 每种格式必须实现的核心：解析一个区域的字节，返回 RegionParse（calls + 标记spans）
    #   stream.py 在区域闭合或流结束时以 at_end=True 调用；在提前播报工具名时以 at_end=False 调用
    @classmethod
    @abstractmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """What this region's bytes mean.

        It sees only the region -- never the content before it, which the
        engine has already released, and never the answer after it, which it
        reports the end of rather than consuming.

        Returning no calls says the region was a quotation: an answer
        explaining that a model "writes ``<tool_call>`` to call something"
        opens one and never closes it. The engine then releases the bytes
        unchanged, so a start marker is not a promise -- a rule that used to
        be written out in each format, and was missing from two of them.

        Where a call's own pattern *ends* is part of that meaning, and it has
        five terminators, not four: its closing tag, the next parameter, the
        end of the call, end of input, and **the next call's opener**. Four of
        the six formats were missing the last, so a call cut off mid-value ran
        on into the sibling behind it and the tool was invoked with markup as
        data -- ``city="Par<tool_call>"``. The call count stayed right either
        way, which is what hid it.

        ``at_end`` is whether more bytes can still arrive for this region, and
        it is the *only* difference between the two questions this used to be
        asked twice. With ``at_end`` a token cut off part-way through is all
        there will ever be, so a prefix of one counts -- that is what a call
        truncated by ``max_tokens`` looks like. Without it a prefix means "not
        yet", and accepting one let a chunk boundary landing one character
        into ``<br>`` name a tool for prose.

        The engine calls this with ``at_end=False`` to find the name to
        announce early. That is what makes the early name and the parsed call
        agree by construction: they are the same enumeration of the same
        function over a prefix and then the whole. There used to be a separate
        ``peek_name`` per format, and its disagreements with this were
        the single most repeated defect in this module -- most recently a
        DeepSeek-V4 response whose peek skipped a self-closing
        ``<invoke name="x"/>`` that the parse returned first, putting three
        tool calls on the wire where the model had written two.
        """
        # 【译】这个区域的字节是什么意思。
        #
        #   它只看到区域本身——从不看到它之前的内容（引擎已经释放了），也不看到它之后的答案（它只报告标记在哪里结束，而不消费后面的内容）。
        #
        #   不返回任何调用表示这个区域是一段引用：一个解释模型"写 ``<tool_call>`` 来调用东西"的回答打开了一个区域却从不关闭它。
        #   引擎随后原样释放这些字节，所以起始标记不等于承诺——这条规则过去在每种格式里各写一遍，其中两种还漏掉了。
        #
        #   一次调用自身的模式在哪里*结束*也是这个含义的一部分，它有五个终止条件，而不是四个：它的闭合标签、下一个参数、
        #   调用结束、输入结束，以及**下一个调用的开始标记**。六种格式中有四种漏掉了最后一个，于是一个在参数值中途被截断的
        #   调用会一直延续到它后面的兄弟调用里，工具被以标记作为数据来调用——``city="Par<tool_call>"``。
        #   无论哪种情况调用数量都是对的，正是这一点把问题藏了起来。
        #
        #   ``at_end`` 说的是这个区域还会不会有更多字节到来，它是这个函数过去被问的两个问题之间*唯一*的差别。
        #   在 ``at_end`` 时，一个只写了一半的token就是它的全部了，所以它的前缀也算数——被 ``max_tokens`` 截断的调用就长这样。
        #   不在末尾时，前缀意味着"还没到"，而接受前缀曾让一个落在 ``<br>`` 第一个字符之后的chunk边界为普通文字命名了一个工具。
        #
        #   引擎以 ``at_end=False`` 调用它来找出要提前播报的工具名。正是这一点使提前播报的名字与最终解析出的调用在构造上就一致：
        #   它们是同一个函数先在前缀上、再在整体上的同一种枚举。过去每种格式各有一个单独的 ``peek_name``，它与本函数的不一致
        #   是本模块里重复次数最多的缺陷——最近一次是一个 DeepSeek-V4 响应：peek 跳过了一个自闭合的 ``<invoke name="x"/>``，
        #   而 parse 把它作为第一个返回，结果线上出现了三个工具调用，而模型只写了两个。
        # 解释：at_end=True 表示不会再有字节到来（区域已闭合或流已结束）
