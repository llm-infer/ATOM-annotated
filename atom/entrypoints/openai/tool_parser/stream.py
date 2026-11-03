# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The engine: one reader of a model's output, for both delivery modes.

A format (:class:`~.tool_parser.ToolCallParser`) says what its own bytes mean.
Everything a reader has to decide that is *not* particular to a format is
decided here, once: how far ahead of a region it is safe to release, which
markers are framing and get dropped, what a region that parses to no call
means, which index a call is stamped with, and what happens to the answer that
follows the markup.

`stream=false` is :func:`read_whole`, which is this engine over a single chunk.
That is the whole of why the two delivery modes agree -- not a test that
compares them, but no second implementation to compare against.
"""

# 【译】引擎：模型输出的唯一读取器，两种交付模式共用。
#
#   一种格式（:class:`~.tool_parser.ToolCallParser`）只说明它自己的字节是什么意思。读取器要决定的、
#   *不*属于某种格式特有的一切，都在这里统一决定一次：在区域之前可以安全释放多少内容、哪些标记是框架标记要丢弃、
#   一个解析不出调用的区域意味着什么、一个调用被编上哪个index，以及标记之后的回答怎么处理。
#
#   `stream=false` 就是 :func:`read_whole`，也就是这个引擎在单个chunk上运行。这就是两种交付模式一致的全部原因——
#   不是靠一个比较二者的测试，而是根本不存在可以拿来比较的第二份实现。
#
# 【导读】工具调用的流式解析引擎 ToolCallStreamParser：把模型输出（逐chunk或一次全部）变成有序事件
#   ("content", 文本) / ("tool_call_start", {...}) / ("tool_call_args", {...}) / ("tool_call_end", None)
# 工作方式（_pump 一个循环，两种状态）：
#   1. 区域外：MarkerScanner 预读——能确定不是某个起始标记开头的文本立即作为 content 释放，
#      末尾"可能是标记前半截"的几个字符先扣住，等下一个chunk再判断；
#      命中完整标记时：opens_region 为 False 的是框架标记，直接丢掉；为 True 的打开一个 Region 开始缓冲
#   2. 区域内：字节追加进 Region；只有当新chunk（连同上一chunk的尾巴）里出现区域结束字面量时，才去问 region_end
#      区域是否闭合（避免每个chunk都拼接整个缓冲区）；区域未闭合前可通过 _announce 提前播报工具名
#   3. 区域闭合（或流结束）：_close_region 用 parse_region(at_end=True) 解析，按 spans 交错输出调用之间的文字和调用事件，
#      标记之后剩余的文本交还给第1步继续扫描
# 例（Qwen格式，tools 中声明了 get_weather）：
#   chunk1 "好的。<tool"           -> ("content", "好的。")，"<tool" 被扣住
#   chunk2 "_call>\n<function=get_weather>\n<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>"
#                                  -> 先提前播报 ("tool_call_start", {index:0, name:"get_weather"})，
#                                     区域闭合后 ("tool_call_args", {index:0, arguments:'{"city": "北京"}'})、("tool_call_end", None)
#   chunk3 "查到了。"             -> ("content", "查到了。")
# 调用方：流式——serving_chat.py（chat completions）、serving_anthropic.py / api_server.py（/v1/messages、/v1/responses）、
#   atomesh/atom_standalone_service.py 各自为每个请求（每个choice）创建一个 ToolCallStreamParser，逐个delta调用 process，结束时调用 flush；
#   非流式——registry.parse_tool_calls -> read_whole
# 与主干的关系：服务层的后处理，输入是detokenize后的文本增量，输出交给SSE组帧（见本批 sse.py / serving_chat.py）
# 阅读路线（约1050行）：Region（区域缓冲）-> ToolCallStreamParser.process / flush -> _pump（主循环，对应上面1~3步）
#   -> _announce（提前播报工具名）/ _closed_len（区域是否已闭合）-> _close_region（解析并按spans交错输出）-> _emit_call / _start_event；
#   模块级的 _markup_spans / _region_tail / _region_owes_a_closer 是 _close_region 用到的判定，可以后读；
#   文件末尾的 read_whole / read_whole_events / flatten_tool_events 是非流式入口

import logging

from ..marker_scanner import MarkerScanner
from .schema import ParamTypes, build_param_types
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    begin_of_markup,
)

logger = logging.getLogger("atom")

# How far into a region a name may be before the peek gives up. Every
# format on this box puts it in the first 30-70 characters; the margin is
# for a long name or leading whitespace, and the bound is what keeps the
# peek from re-scanning a growing buffer once per token.
# 【译】工具名最多可以在区域里多深的位置，超过就不再偷看。这台机器上的每种格式都把名字放在前30~70个字符内；
#   余量是留给长名字或开头空白的，而这个上限保证了偷看（peek）不会每个token都重新扫描一遍不断增长的缓冲区。
_PEEK_WINDOW = 256

# Do not add a "give up on a region that has produced nothing after N bytes"
# probe. It was tried and reverted: it rests on acceptance being monotone in
# how many bytes have arrived, and that is false -- MiniMax gates its in-progress test on the
# first tag being in the declared schema, and DSML's wrapper-less and
# direct-JSON branches cannot match any prefix at all -- so real calls over
# N bytes were delivered as text, differently from `stream=false`, on three of
# the six formats. It was also quadratic, because giving up re-feeds bytes that
# immediately reopen a region with a fresh budget: 1.19 ms -> 18.2 s on a
# 250 KB answer, in the request coroutine, stalling the event loop for
# everything else.
#
# Fixing the latency needs the *format* to say "this can no longer become a
# call", which is a different question from "does not parse yet" and one no
# format answers today.
# 【译】不要加"区域在N个字节后仍未产生任何东西就放弃"的探测。试过，又撤回了：它依赖于"接受与否随到达的字节数单调"，
#   而这是错的——MiniMax 的进行中测试要求第一个标签在声明的schema里，DSML 的无包裹分支和直接JSON分支则对任何前缀都无法匹配——
#   于是超过N字节的真实调用被当作文本交付，与 `stream=false` 不同，六种格式中有三种如此。它还是平方复杂度的，
#   因为放弃时会把字节重新喂回去，而这些字节立刻又以新的预算重新打开一个区域：一个250KB的回答从1.19ms变成18.2s，
#   而且是在请求协程里，阻塞了事件循环上的其他一切。
#
#   要修这个延迟，需要*格式*能说出"这已经不可能变成一个调用了"，这与"还解析不出来"是不同的问题，目前没有任何格式回答它。


# 一个工具调用区域打开以来缓冲的字节：用list存分片、需要时才join；另外单独保存前 _PEEK_WINDOW 个字节（head）供提前播报工具名
class Region:
    """The bytes buffered since a tool-call region opened.

    A list and a join, not `self.buf += text`. Appending to a *string
    attribute* is quadratic in CPython: the instance dict holds a reference,
    so the in-place fast path never applies and every chunk copies the whole
    buffer. Measured on a 128 KB tool call, streamed four characters at a
    time: 23 ms of event-loop CPU in `process` alone, growing 17x for an 8x
    payload while `flush` stayed linear. The same loop over a *local* string
    is linear, which is why a microbenchmark of `s += x` finds nothing.

    `head` is the first `_PEEK_WINDOW` bytes, kept separately so the early
    name announcement never needs the region materialised -- it only ever
    looks that far in, and handing it the whole buffer is how the quadratic
    scan it already fixed once could come back.
    """

    # 【译】一个工具调用区域打开以来缓冲的字节。
    #
    #   用 list 加 join，而不是 `self.buf += text`。在CPython里往*字符串属性*上追加是平方复杂度的：实例字典持有一个引用，
    #   所以原地追加的快速路径永远用不上，每个chunk都会复制整个缓冲区。在一个128KB的工具调用上实测，每次流式输出4个字符：
    #   仅 `process` 就花了23ms的事件循环CPU，负载变成8倍时它增长了17倍，而 `flush` 保持线性。
    #   同样的循环作用在一个*局部*字符串上是线性的，所以对 `s += x` 做微基准测试什么也发现不了。
    #
    #   `head` 是前 `_PEEK_WINDOW` 个字节，单独保存，这样提前播报工具名时永远不需要把区域拼接出来——它只看这么深，
    #   而把整个缓冲区交给它，正是它曾经修过一次的平方扫描可能卷土重来的方式。

    __slots__ = ("_len", "_parts", "head")

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0
        self.head = ""

    def append(self, text: str) -> None:
        if not text:
            return
        self._parts.append(text)
        self._len += len(text)
        if len(self.head) < _PEEK_WINDOW:
            self.head = (self.head + text)[:_PEEK_WINDOW]

    def text(self) -> str:
        """Everything buffered, without giving it up.

        Collapses the parts as a side effect, so asking twice costs one join
        rather than two. Only the region-close probe asks, and only once a
        literal that could close the region has actually arrived.
        """
        # 【译】缓冲的全部内容，但不交出去。
        #
        #   顺带把分片合并成一个，所以问两次只花一次join。只有区域闭合探测会问，而且只在一个可能关闭区域的字面量真正到达之后才问。
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    def take(self) -> str:
        """Everything buffered, and start again."""
        # 【译】缓冲的全部内容，并清空重新开始。
        out = self.text()
        self._parts.clear()
        self._len = 0
        self.head = ""
        return out

    def __len__(self) -> int:
        return self._len

    def __bool__(self) -> bool:
        return self._len > 0


# 请求声明的全部工具名集合，提前播报时用来判断名字是否已声明
def _peek_declared(tools: list | None) -> frozenset[str]:
    return frozenset(build_param_types(tools))


# 区域的最后一段标记是否还欠一个包裹闭合标记（例如解析出了调用，但 </tool_calls> 还在后面的文字之后）
def _region_owes_a_closer(body: str, parser_cls, spans: list[tuple[int, int]]) -> bool:
    """Did this region's own markup leave its wrapper unclosed?

    At the end of the region the two cases are byte-identical::

        [call] prose </tool_calls>    the wrapper -- markup, and must go
        [call] prose </tool_call>     the model naming it -- answer, and stays

    What separates them is the right edge of the last markup span:
    `markup_end` steps over a closer and stops at prose, so a span ending *on*
    one closed its wrapper and a span ending on anything else still owes it.

    One `endswith` at one position, not a count. Counting openers minus
    closers broke twice over: literals inside argument *values* invented a
    debt, discharged against the earliest closer in the document, so DSML
    deleted twelve bytes out of a sentence and left the real wrapper in. A
    value can never *be* a span's right edge.

    Bounded `endswith` rather than `body[start:stop].rstrip()`, which copies
    the span to read its last few bytes -- O(1) against O(span), measured 10x
    on a 128 KB argument and flat where the copy is linear.

    The merged partition is safe here, where a balance would not be: merging
    leaves the last span's right edge alone, but would read two unclosed calls
    as one balanced pair.
    """
    # 【译】这个区域自己的标记有没有留下一个没闭合的包裹？
    #
    #   在区域末尾，下面两种情况逐字节相同：
    #       [调用] 文字 </tool_calls>    包裹——是标记，必须去掉
    #       [调用] 文字 </tool_call>     模型在说出它的名字——是回答，要保留
    #
    #   区分它们的是最后一个标记span的右边界：`markup_end` 会越过闭合标记、在普通文字处停下，所以恰好结束*在*一个闭合标记上的span
    #   已经关闭了它的包裹，而结束在其他任何东西上的span仍然欠着一个。
    #
    #   在一个位置上做一次 `endswith`，而不是计数。用开始标记数减闭合标记数的做法坏过两次：参数*值*里的字面量会凭空制造一笔"欠账"，
    #   并用文档里最早出现的闭合标记来抵消，于是 DSML 从一句话里删掉了12个字节，却把真正的包裹留在了原处。
    #   一个参数值永远不可能*是*span的右边界。
    #
    #   用有界的 `endswith` 而不是 `body[start:stop].rstrip()`，后者为了读最后几个字节要复制整个span——
    #   O(1) 对 O(span)，在128KB的参数上实测快10倍，而且是平的，复制则是线性的。
    #
    #   在这里用合并后的划分是安全的，而用"配平"则不安全：合并不会改变最后一个span的右边界，但会把两个没闭合的调用读成一对配平的。
    closers = parser_cls.CALL_CLOSERS
    if not closers or not spans:
        return False
    start, end = spans[-1]
    while end > start and body[end - 1].isspace():
        end -= 1
    return not any(body.endswith(c, start, end) for c in closers)


# 返回关闭本区域的包裹标记的起始位置（只在区域确实欠一个闭合标记时往回找），否则返回 len(body)；该位置之后的字节交还给扫描器
def _region_tail(body: str, parser_cls, spans: list[tuple[int, int]]) -> int:
    """Where the wrapper closing this region starts, or ``len(body)``.

    `markup_begin`/`markup_end` walk a *call's* edges and stop at prose, so a
    sentence between the last call and `</tool_calls>` left that tag belonging
    to nobody and it reached the client verbatim.

    Only when the region owes one. Walking unconditionally could not tell the
    wrapper from an answer that *ends by naming* it: of 53 shapes whose
    trailing literal was the model's own text it kept 26, against 47 here.
    Deleting the walk instead keeps all 47 and leaves the wrapper in `content`
    wherever a region really does owe one -- the worse trade, and three red
    tests.

    Takes the whole trailing run once entered, so two quoted closers against
    one owed wrapper lose both. Spending the count as a budget fixes that and
    needs the unmerged spans plus a per-span balance; measured identical on
    all 5220 cross-format corpus pairs, so it waits for a real output.

    An offset, not a span. Splicing the wrapper in as one made it compete for
    a call: two adjacent calls merge into a single span, so the wrapper span
    became the last, took the leftover call, and emitted the prose before it
    -- the same output ordered differently depending on whether the closing
    tag had arrived.

    Never before the markup it follows, or a region that is nothing but a call
    walks back over the call's own closer, in front of the caller's cursor.

    Trailing edge only. A wrapper opener followed by prose is byte-identical
    to an answer quoting the opener before calling for real, and
    `RegionParse.begins` already resolves that in favour of keeping the text;
    a leading scan here deleted exactly such a quotation. So an opening
    wrapper with prose after it still leaks, which is the smaller harm.
    """
    # 【译】关闭这个区域的包裹从哪里开始，或者 ``len(body)``。
    #
    #   `markup_begin`/`markup_end` 走的是*调用*的边缘，并在普通文字处停下，所以最后一个调用与 `</tool_calls>` 之间的一句话
    #   会让这个标签不属于任何人，它就原样到达了客户端。
    #
    #   只在区域欠一个闭合标记时才这样做。无条件地往回走，就分不清包裹和一个*以说出它的名字结尾*的回答：
    #   在53种末尾字面量是模型自己文字的形状里，那样只保留了26种，这里保留47种。改为删掉这次回走，则47种全部保留，
    #   但在区域确实欠一个包裹的地方会把包裹留在 `content` 里——这是更差的取舍，还有三个失败的测试。
    #
    #   一旦进入，就拿走整段末尾连续的闭合标记，所以两个被引用的闭合标记对上一个欠着的包裹时，两个都会丢失。
    #   把欠账数当作预算来花可以修这个问题，但需要未合并的spans加上逐span的配平；在全部5220对跨格式语料上实测结果完全相同，
    #   所以等真实输出出现再说。
    #
    #   是一个偏移量，不是一个span。把包裹作为一个span拼进去，会让它和调用争抢：两个相邻的调用会合并成一个span，
    #   于是包裹span成了最后一个，拿走了剩下的调用，并把它前面的文字先发了出去——同一份输出，排序取决于闭合标签是否已经到达。
    #
    #   绝不早于它所跟随的标记，否则一个只包含一个调用的区域会往回越过调用自己的闭合标记，跑到调用方游标的前面。
    #
    #   只处理尾部边缘。一个后面跟着普通文字的包裹开始标记，与一个先引用开始标记再真正调用的回答逐字节相同，
    #   而 `RegionParse.begins` 已经按"保留文字"解决了这个问题；在这里做开头扫描恰恰删掉了这样一段引用。
    #   所以后面跟着文字的开始包裹仍会泄漏，这是较小的伤害。
    if not _region_owes_a_closer(body, parser_cls, spans):
        return len(body)
    return max(
        spans[-1][1],
        begin_of_markup(
            body, len(body), parser_cls.CALL_CLOSERS, parser_cls.CALL_FILLERS
        ),
    )


# 把格式报告的标记spans裁剪到 body 范围内、去掉空span、合并真正重叠的span，得到升序且不重叠的划分
def _markup_spans(
    spans: tuple[tuple[int, int], ...], body: str
) -> list[tuple[int, int]]:
    """A format's markup intervals, clamped and merged into a partition.

    The engine walks these in order and treats the gaps as answer, so they
    have to be ascending and non-overlapping whatever a format reports.

    Only genuinely overlapping spans are joined. Two spans separated by
    whitespace used to be joined here as well, because that whitespace is the
    template's separator and became a `content: "\n"` delta between two
    `tool_calls` deltas -- but joining them broke the pairing downstream,
    which is one call per span with the last span taking the remainder. Two
    adjacent calls merged into one span, so that span took a single call and
    the *last* one took two, putting a call behind a sentence the model wrote
    after it. The separator is dropped where it is read instead, in
    `_close_region`, which needs no span to disappear to do it.

    Every span kept here is non-empty, so the engine always consumes at least
    one byte and cannot hand a region back, find the same marker and parse it
    forever. A format that reports only degenerate spans would break that, so
    it is refused rather than papered over -- the caller falls back to
    releasing the region unchanged.
    """
    # 【译】一种格式的标记区间，裁剪并合并成一个划分。
    #
    #   引擎按顺序遍历它们，并把间隙当作回答，所以无论格式报告了什么，它们都必须是升序且不重叠的。
    #
    #   只合并真正重叠的span。两个被空白隔开的span过去也会在这里合并，因为那个空白是模板的分隔符，
    #   它会在两个 `tool_calls` delta 之间变成一个 `content: "\n"` delta——但合并它们破坏了下游的配对规则，
    #   即每个span一个调用、最后一个span拿走剩余的。两个相邻的调用合并成一个span，于是这个span只拿到一个调用，
    #   而*最后*一个span拿了两个，把一个调用放到了模型在它之后才写的一句话后面。分隔符改为在读到它的地方丢弃，
    #   即 `_close_region` 里，那里不需要让任何span消失。
    #
    #   这里保留的每个span都非空，所以引擎每次至少消费一个字节，不可能把区域交还回去、又找到同一个标记、永远解析下去。
    #   只报告退化span的格式会破坏这一点，所以它会被拒绝而不是被掩盖——调用方会退回到原样释放这个区域。
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(spans):
        start, stop = max(0, min(start, len(body))), max(0, min(stop, len(body)))
        if stop <= start:
            continue
        # `<`, not `<=`: two spans that merely touch are two calls the model
        # wrote with no separator between them, and joining those is the same
        # reordering as joining whitespace-separated ones. The existing
        # property used a newline and so never reached this.
        # 【译】用 `<` 而不是 `<=`：两个仅仅相接的span是模型没有写分隔符的两个调用，合并它们与合并空白分隔的span是同样的重排。
        #   已有的属性测试用的是换行，所以从来没有走到这里。
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


# 把请求的tools预先转换成 ParamTypes（每个请求只建一次）；建不出任何东西时保留原值
def _resolved_tools(tools: list | None):
    """The request's tools as the parsers want them: built once, not per chunk.

    Falls back to whatever was passed when nothing usable comes out, so a
    caller's truthiness test sees the same answer either way.
    """
    # 【译】解析器需要的那种形式的请求tools：只构建一次，而不是每个chunk一次。
    #
    #   当构建不出任何可用内容时，退回到传入的原值，这样调用方的真值测试无论哪种情况看到的答案都一样。
    if tools is None or isinstance(tools, ParamTypes):
        return tools
    return build_param_types(tools) or tools


# 工具调用解析引擎：每个请求（每个choice）一个实例，持有该请求全部的解析状态
class ToolCallStreamParser:
    """Reads one request's output, in chunks or all at once.

    Emits ``(event_type, data)`` tuples:

    - ``("content", text)`` -- answer text, in the order the model wrote it
    - ``("tool_call_start", {"index", "id", "type", "function": {"name", ...}})``
    - ``("tool_call_args", {"index", "function": {"arguments": ...}})``
    - ``("tool_call_end", None)`` -- a region's calls are complete

    ``parser_cls`` is the format this model was resolved to at startup, from
    its chat template (`registry.resolve_from_prompt`). ``None`` means no
    registered format recognised it: everything is emitted as content and
    nothing is parsed, which the server announced at startup. It is
    deliberately not a fallback to guessing -- the guess this replaces
    mis-read a Hermes `<tool_call>{...}` as GLM and delivered the whole JSON
    blob as the tool's *name*.

    ``suppress_calls`` is ``tool_choice: "none"``. The format is still read --
    that is the difference from ``parser_cls=None``, and it matters, because a
    format whose framing wraps *every* answer would otherwise leak that
    framing to the client the moment a request forbade tool calls. What is
    suppressed is dispatch: a region's bytes are released as the prose the
    request said they must be.

    ``tools`` enables JSON-Schema type coercion of parameter values. It may be
    assigned after construction (several call sites do) and is read when a
    region closes, so it takes effect as long as it is set before then. It is
    resolved to :class:`~.schema.ParamTypes` on the way in -- the catalogue is
    walked once per request rather than once per chunk. A request whose tools
    declare no usable name keeps the list it was given, so that ``not
    self.tools`` still asks what it always asked.
    """

    # 【译】读取一个请求的输出，可以分chunk，也可以一次全部读。
    #
    #   产出 ``(event_type, data)`` 元组：
    #   - ``("content", text)`` —— 回答文本，按模型写出的顺序
    #   - ``("tool_call_start", {"index", "id", "type", "function": {"name", ...}})`` —— 一个调用开始（带工具名）
    #   - ``("tool_call_args", {"index", "function": {"arguments": ...}})`` —— 该调用的完整参数
    #   - ``("tool_call_end", None)`` —— 一个区域的调用已经完整
    #
    #   ``parser_cls`` 是启动时从这个模型的chat模板（`registry.resolve_from_prompt`）解析出的格式。``None`` 表示没有已注册格式认出它：
    #   一切都作为content输出，什么也不解析，服务启动时已经宣告了这一点。它刻意不退回到猜测——被它取代的那个猜测曾把
    #   Hermes 的 `<tool_call>{...}` 误读成 GLM，把整段JSON当作工具的*名字*交付了出去。
    #
    #   ``suppress_calls`` 就是 ``tool_choice: "none"``。格式仍然会被读取——这是它与 ``parser_cls=None`` 的区别，而且很重要，
    #   因为框架标记包裹*每一个*回答的格式，否则一旦请求禁止工具调用就会把框架泄漏给客户端。被抑制的是分发：
    #   区域的字节按请求的要求作为普通文字处理。
    #
    #   ``tools`` 启用参数值的JSON-Schema类型转换。它可以在构造之后再赋值（有好几个调用点这么做），并在区域闭合时被读取，
    #   所以只要在那之前设置就能生效。它在传入时被解析成 :class:`~.schema.ParamTypes`——每个请求只遍历一次工具目录，
    #   而不是每个chunk一次。如果请求的tools没有声明任何可用的名字，就保留传入的列表，这样 ``not self.tools`` 问的仍是它一直在问的问题。
    # 解释：最后一句与代码稍有出入——被抑制时，区域的标记照样被删除，只保留标记周围的回答（见 _close_region），并不是把整个区域当文字释放

    def __init__(
        self,
        tools: list | None = None,
        parser_cls: type[ToolCallParser] | None = None,
        *,
        suppress_calls: bool = False,
    ) -> None:
        self.tools = tools  # type: ignore[assignment]
        self.parser_cls = parser_cls
        self.suppress_calls = suppress_calls
        # Reads ahead of the region: releases everything that cannot begin one
        # of `parser_cls`'s own markers, and drops the ones that are framing.
        # One scanner, not one here and another on the format -- two readers
        # of the same marker set is how framing came to be dropped before a
        # region and kept inside one.
        # 【译】在区域之前预读：释放一切不可能是 `parser_cls` 自己某个标记开头的内容，并丢弃其中属于框架标记的那些。
        #   只用一个扫描器，而不是这里一个、格式里另一个——同一组标记有两个读取器，正是框架标记会在区域之前被丢弃、在区域之内却被保留的原因。
        self._scanner = (
            MarkerScanner(parser_cls.START_MARKERS) if parser_cls is not None else None
        )
        self._region: Region | None = None

        # Both closers, where the format declared no region end of its own.
        # The wrapper's is a *trigger* only -- it arrives on the chunk that
        # completes the region, and without it the call's own closer asks
        # first, hears "not yet", and nothing asks again. `region_end` decides
        # from the self closer; see there for why the wrapper's cannot.
        # 【译】格式没有声明自己的区域结束标记时，使用两种闭合标记。包裹的闭合标记只是一个*触发器*——它出现在完成区域的那个chunk里，
        #   没有它的话，调用自己的闭合标记先来询问，得到"还没有"，之后就再也没人问了。`region_end` 根据调用自身的闭合标记做决定；
        #   包裹的闭合标记为什么不行，见那里的说明。
        self._end_markers: tuple[str, ...] = ()
        if parser_cls is not None:
            self._end_markers = parser_cls.REGION_END_MARKERS or (
                parser_cls.CALL_SELF_CLOSERS + parser_cls.CALL_CLOSERS
            )
        # The tail a region-closing literal split across a chunk boundary
        # would need, and nothing more.
        # 【译】一个被chunk边界切开的区域结束字面量所需要的那段尾巴，不多不少。
        # 解释：即最长结束字面量长度减1；下一个chunk会拼上这段尾巴再查找，保证逐字符到达的结束标记也能被发现
        self._end_tail_len = max((len(m) for m in self._end_markers), default=1) - 1
        self._end_tail = ""
        # Stamped by the engine, so no format has to. Kimi took its index off
        # the wire, where every section restarts at 0, and a client
        # accumulating arguments by index merged two calls into one.
        # 【译】由引擎编号，这样没有哪个格式需要做。Kimi 过去直接从线上取index，而每个section都从0重新开始，
        #   于是一个按index累积参数的客户端把两个调用合并成了一个。
        self._index = 0
        self._emitted = 0
        # The name already sent for the region being buffered, if any.
        # 【译】为正在缓冲的区域已经发出的工具名（如果有）。
        self._announced: str | None = None
        # Set once no name can still turn up, so the peek stops running per
        # token over a region that will never yield one.
        # 【译】一旦不可能再出现名字就设置它，这样偷看就不会在一个永远产生不了名字的区域上每个token都运行一次。
        self._peek_exhausted = False
        # The request's declared names, built once rather than per token.
        # 【译】请求声明的名字，只构建一次，而不是每个token一次。
        self._declared: frozenset[str] | None = None

    @property
    def tools(self):
        """The request's tools, resolved once (see :func:`_resolved_tools`).

        A property and not a line in ``__init__`` because several call sites
        assign it afterwards, and the resolution has to happen for those too
        or they are the slow path again.
        """
        # 【译】请求的tools，只解析一次（见 :func:`_resolved_tools`）。
        #
        #   用property而不是 ``__init__`` 里的一行，是因为好几个调用点会在之后再赋值，这些赋值也必须经过解析，否则它们又走回了慢路径。
        return self._tools

    @tools.setter
    def tools(self, value: list | None) -> None:
        self._tools = _resolved_tools(value)
        self._declared = None

    @property
    def fmt(self) -> str | None:
        """The format this stream is being read as, or None."""
        # 【译】这个流按哪种格式读取，或者None。
        return self.parser_cls.NAME if self.parser_cls is not None else None

    # -- public ------------------------------------------------------------
    # 【译】-- 公开接口 --
    def process(self, text: str) -> list:
        """Consume one chunk; return the events it completed."""
        # 【译】消费一个chunk；返回它完成的事件。
        if self.parser_cls is None:
            return [("content", text)] if text else []
        return self._pump(text, final=False)

    def flush(self) -> list:
        """End of stream: nothing more is coming, so nothing is held back."""
        # 【译】流结束：不会再有更多内容，所以不再扣住任何东西。
        if self.parser_cls is None:
            return []
        return self._pump("", final=True)

    # -- the loop ----------------------------------------------------------
    # 【译】-- 主循环 --
    def _pump(self, text: str, *, final: bool) -> list:
        """Read `text`, and whatever a closing region hands back, to a stop.

        The one loop. Content before a region, framing, the region itself and
        the answer after it are all reached from here, so "what happens to
        these bytes" has a single answer per byte rather than one per format
        per delivery mode.
        """
        # 【译】读取 `text`，以及一个闭合的区域交还回来的内容，直到停下。
        #
        #   唯一的循环。区域之前的内容、框架标记、区域本身以及它之后的回答都从这里到达，
        #   所以"这些字节怎么处理"每个字节只有一个答案，而不是每种格式、每种交付模式各一个。
        out: list = []
        # 解释：循环每轮处于两种状态之一——self._region 为 None 时在区域外（交给扫描器），否则在区域内（缓冲并探测闭合）；
        #   区域闭合后剩下的文本放回 text，下一轮回到区域外继续扫描，直到没有可处理的文本为止
        while True:
            if self._region is None:
                if text:
                    scan = self._scanner.feed(text)
                    text = ""
                    if scan.released:
                        out.append(("content", scan.released))
                    if scan.hit is not None:
                        if not self.parser_cls.opens_region(scan.hit):
                            # Framing this format wraps every answer in.
                            # Dropping it and carrying on is what lets such a
                            # format stream at all: treating it as a handover
                            # meant a Kimi-K3 answer, which opens with
                            # `<|open|>response<|sep|>`, buffered its entire
                            # body to EOS -- 324 of 324 characters in one
                            # frame at the end.
                            # 【译】这种格式包裹在每个回答外面的框架标记。丢弃它然后继续，正是这样的格式能够流式输出的原因：
                            #   把它当作移交，意味着一个以 `<|open|>response<|sep|>` 开头的 Kimi-K3 回答会把整个正文缓冲到EOS——
                            #   324个字符中的324个都在最后一帧里。
                            text = scan.rest
                        else:
                            # The region has opened. Opened, but not read
                            # here: the marker goes back through the region
                            # branch below, so a chunk carrying a whole call
                            # is closed on this pass rather than one late.
                            #
                            # Not skipped for `suppress_calls`: releasing the
                            # marker as content there let the rest of the
                            # region stream out as content too, putting raw
                            # wire tokens in the answer. A region is buffered
                            # the same way whatever the request said about
                            # dispatch; only `_close_region` differs.
                            # 【译】区域已经打开。打开了，但不在这里读：标记会回到下面的区域分支再走一遍，所以一个携带完整调用的chunk会在这一轮就被关闭，
                            #   而不是晚一轮。
                            #
                            #   对 `suppress_calls` 也不跳过：在那种情况下把标记作为content释放，会让区域的其余部分也作为content流出去，
                            #   把原始的线上token放进回答里。无论请求对分发说了什么，区域都以同样方式缓冲；只有 `_close_region` 不同。
                            self._region = Region()
                            self._end_tail = ""
                            text = scan.hit + scan.rest
                        continue
                if not final:
                    return out
                # 解释：只有流结束（final=True）时才把扫描器扣住的"可能是标记前半截"的尾巴作为content放出
                held = self._scanner.flush()
                if held:
                    out.append(("content", held))
                return out

            # 解释：区域内——probe = 上一chunk的尾巴 + 本chunk，只在probe里出现结束字面量时才去问 region_end；
            #   final=True 时整个缓冲区都算作区域（end=len）
            probe = ""
            if text:
                if self._end_markers:
                    probe = self._end_tail + text
                    self._end_tail = (
                        probe[-self._end_tail_len :] if self._end_tail_len else ""
                    )
                self._region.append(text)
                text = ""
                out.extend(self._announce())
            end = len(self._region) if final else self._closed_len(probe)
            if end <= 0:
                return out
            body = self._region.take()
            self._region = None
            self._end_tail = ""
            events, rest = self._close_region(body[:end])
            out.extend(events)
            text = rest + body[end:]

    def _closed_len(self, probe: str) -> int:
        """Whether the open region has closed, without materialising it.

        The probe is the literal itself: nothing in a chunk can close a region
        unless one of `REGION_END_MARKERS` lands in it, so until one does
        there is nothing to look at. Asking the format instead would mean
        joining the buffer once per chunk, which is quadratic in the region --
        measured at 55 ms of event-loop CPU on a 50 KB argument against 4 ms
        for the same payload in a format that did not do it.

        `probe` is the incoming chunk with the tail a marker split across the
        boundary would need, so a section end arriving one character at a time
        is still seen.
        """
        # 【译】打开的区域是否已经闭合，而不把它拼接出来。
        #
        #   探测的就是字面量本身：除非某个 `REGION_END_MARKERS` 落在chunk里，否则chunk里没有任何东西能关闭区域，所以在那之前没有什么可看的。
        #   改为询问格式意味着每个chunk都要把缓冲区拼接一次，这对区域来说是平方复杂度的——在一个50KB的参数上实测55ms的事件循环CPU，
        #   而同样的负载在不这样做的格式上只要4ms。
        #
        #   `probe` 是传入的chunk加上一个被边界切开的标记所需要的尾巴，所以逐字符到达的section结束标记也能被看到。
        if not probe or not any(m in probe for m in self._end_markers):
            return 0
        return self.parser_cls.region_end(self._region.text())

    # 区域闭合时调用：parse_region(at_end=True) 解析，产出 [调用间的文字, 调用事件..., tool_call_end]，并返回标记之后要交还扫描器的文本
    def _close_region(self, body: str) -> tuple[list, str]:
        """One region's events, and the bytes that were not its markup.

        The announcement is per region and is settled here either way: bound
        to the region's first call, or dropped. Carrying it past the region it
        was made in is how one region's peek came to be matched against the
        next region's parse, and reported as a mismatch.
        """
        # 【译】一个区域的事件，以及不属于它的标记的那些字节。
        #
        #   提前播报是按区域进行的，无论如何都在这里了结：要么绑定到区域的第一个调用，要么丢弃。
        #   把它带过它所在的区域，正是一个区域的偷看结果被拿去和下一个区域的解析结果比对、并被报告为不匹配的原因。
        parsed = self.parser_cls.parse_region(body, self.tools, at_end=True)
        if self.suppress_calls and parsed.calls:
            # `tool_choice: "none"`. The format is read either way -- that is
            # what separates this from `parser_cls=None` -- and what is
            # suppressed is dispatch, not reading. Skipping the parse instead
            # put the model's raw wire tokens in `content`, on every format
            # and both delivery paths. The markup is now located exactly as it
            # is for a dispatched call and only the answer around it goes out,
            # so a response that was nothing but a call has empty content.
            # 【译】`tool_choice: "none"`。无论如何都会读取格式——这正是它与 `parser_cls=None` 的区别——被抑制的是分发，而不是读取。
            #   改为跳过解析，会在所有格式、两条交付路径上把模型的原始线上token放进 `content`。现在标记的定位与要分发的调用完全一样，
            #   只有它周围的回答被发出，所以一个只有调用的响应，content为空。
            logger.debug(
                "tool_choice=none: dropped %d %s call(s) from the response",
                len(parsed.calls),
                self.parser_cls.NAME,
            )
            parsed = RegionParse((), parsed.spans)
        events: list = []
        rest = ""
        # Everything outside the format's markup is answer, including whatever
        # the model wrote *between* two calls. Reading only the outer pair
        # deleted that -- "let me also check Rome" between two `<tool_call>`
        # blocks, on five of the six formats, silently.
        #
        # `spans` empty with calls present would mean a format reported
        # nothing this engine can consume, and consuming nothing loops
        # forever. No registered format can do it -- 1.2M fuzzed regions
        # produced zero -- so this is the boundary check for the seventh, and
        # it fails towards releasing the region rather than towards eating a
        # byte of it.
        # 【译】格式标记之外的一切都是回答，包括模型写在两个调用*之间*的内容。只读最外层的一对会删掉它——在两个 `<tool_call>` 块之间的
        #   "让我也查一下罗马"，在六种格式中的五种上被静默删除。
        #
        #   有调用却 `spans` 为空，意味着格式报告了引擎无法消费的东西，而什么也不消费会无限循环。没有任何已注册格式会这样——
        #   120万个模糊测试区域一次也没有产生——所以这是为第七种格式准备的边界检查，它失败时倾向于释放区域，而不是吃掉其中的某个字节。
        spans = _markup_spans(parsed.spans, body)
        if parsed.calls and not spans:
            logger.warning(
                "%s reported %d call(s) with no usable markup span in %d bytes; "
                "releasing the region as text",
                self.parser_cls.NAME,
                len(parsed.calls),
                len(body),
            )
        if spans and (parsed.calls or self.suppress_calls):
            pending = list(parsed.calls)
            at = 0
            for i, (start, stop) in enumerate(spans):
                gap = body[at:start]
                # Whitespace *between* two calls is the template's separator,
                # not the answer: every one of these chat templates renders
                # one, and `end_of_markup` stops at it because it moves only
                # on a closer. Before the first call it is the model's, since
                # that edge is where prose ended. Dropped here rather than by
                # joining the two spans, which cost the one-call-per-span
                # pairing and reordered the calls.
                # 【译】两个调用*之间*的空白是模板的分隔符，不是回答：这些chat模板每一个都会渲染一个，而 `end_of_markup` 会在它前面停下，
                #   因为它只在遇到闭合标记时才移动。在第一个调用之前的空白是模型写的，因为那个边缘就是文字结束的地方。
                #   在这里丢弃它，而不是靠合并两个span，后者会破坏"每个span一个调用"的配对并打乱调用顺序。
                if gap and (i == 0 or gap.strip()):
                    events.append(("content", gap))
                at = stop
                # In the order the model wrote them, so a client sees the
                # sentence between two calls between them and not hoisted in
                # front of both. One call per span, and the last span takes
                # whatever is left -- which is how Kimi-K2's single
                # section-wide span carries all of its entries.
                # 【译】按模型写出它们的顺序，这样客户端看到的两个调用之间的句子就在它们之间，而不是被提到两个调用前面。
                #   每个span一个调用，最后一个span拿走剩下的全部——Kimi-K2 覆盖整个section的单个span就是这样携带它所有条目的。
                take = len(pending) if i == len(spans) - 1 else 1
                for call in pending[:take]:
                    events.extend(self._emit_call(call))
                del pending[:take]
            for call in pending:
                events.extend(self._emit_call(call))
            if parsed.calls:
                # Not for a suppressed region: there is no call to end, and a
                # bare `tool_call_end` is read downstream as "a call just
                # completed" -- `completes_a_tool_call` says so, and the
                # Anthropic path closes a block on it.
                # 【译】被抑制的区域不发：没有要结束的调用，而一个孤立的 `tool_call_end` 在下游会被读成"刚完成了一个调用"——
                #   `completes_a_tool_call` 是这么判断的，Anthropic 路径也会据此关闭一个块。
                events.append(("tool_call_end", None))
            # Up to the region's own closing wrapper, which belongs to no
            # call. What is left goes back through the scanner as ordinary
            # answer text.
            # 【译】直到区域自己的闭合包裹为止，它不属于任何调用。剩下的部分作为普通回答文本回到扫描器。
            rest = body[at : _region_tail(body, self.parser_cls, spans)]
        elif body:
            # A start marker is not a promise. An answer explaining that a
            # model "writes <tool_call> to call something" opens the region
            # and never closes it, and this used to drop everything from the
            # marker on -- no event, no error, `finish_reason` still `stop`.
            # Fifty characters of eighty-two, on the shapes measured.
            #
            # Released verbatim, and released here rather than handed back:
            # handing it back would find the same marker again.
            #
            # A name may already have gone out, and there is no retracting it.
            # What there is: no arguments follow, so nothing downstream counts
            # this as a usable call and `finish_reason` stays `stop`. The text
            # is still delivered -- the promise costs a dangling name, not the
            # answer.
            # 【译】起始标记不等于承诺。一个解释模型"写 <tool_call> 来调用东西"的回答会打开区域却从不关闭，而过去这会丢掉从标记开始的一切——
            #   没有事件、没有错误，`finish_reason` 仍然是 `stop`。在测量的那些形状上，82个字符丢了50个。
            #
            #   原样释放，并且在这里释放而不是交还回去：交还回去会再次找到同一个标记。
            #
            #   可能已经有一个名字发出去了，而且无法撤回。能做到的是：后面不会有参数，所以下游没有任何东西会把它当作可用的调用，
            #   `finish_reason` 仍是 `stop`。文本照样交付——这个"承诺"的代价是一个悬空的名字，而不是回答本身。
            events.append(("content", body))
        if self._announced is not None and not parsed.calls:
            # The name went out and nothing will follow it. Step past the index
            # it was sent at, exactly as `_start_event` does when it drops one:
            # a later call landing on an index the client has already bound to
            # a name is merged into it by any accumulator that keys on index,
            # which is the OpenAI streaming contract.
            #
            # Not reachable while every format's acceptance is monotone in how
            # many bytes have arrived -- an announcement is the first call of
            # `parse_region` over a prefix of these same bytes. That property
            # is fuzzed rather than assumed
            # (`TestTheEarlyNameCannotDisagreeWithTheParse`), and this is what
            # the wire looks like if it ever stops holding.
            # 【译】名字发出去了，后面不会有任何东西跟上。跳过它被发出时用的那个index，与 `_start_event` 丢弃一个名字时完全一样：
            #   一个后来的调用如果落在客户端已经绑定了某个名字的index上，任何按index累积的客户端都会把它合并进去，这就是OpenAI的流式约定。
            #
            #   只要每种格式的接受与否随到达的字节数单调，这里就不可达——一次播报就是 `parse_region` 在这些相同字节的一个前缀上的第一个调用。
            #   这个性质是通过模糊测试验证的而不是假设的（`TestTheEarlyNameCannotDisagreeWithTheParse`），这里是万一它不再成立时线上的样子。
            self._index += 1
        self._announced = None
        self._peek_exhausted = False
        return events, rest

    # -- events ------------------------------------------------------------
    # 【译】-- 事件 --
    def _announce(self) -> list:
        """Send the tool's name as soon as the region reveals it, once.

        The name is read out of ``parse_region`` itself, over the region so
        far and with ``at_end=False``. That is what makes it agree with the
        call that eventually goes out: same function, same enumeration, the
        second run seeing a superset of the first's bytes. Every format used
        to answer this with a regex of its own, and four of the five had it
        disagree with their parse -- most recently over a self-closing
        ``<invoke name="x"/>`` the peek skipped and the parse returned first,
        which put three tool calls on the wire for a response containing two.

        Only for a name the request actually declared. That check is what
        makes an early name safe to send: it cannot be taken back, and an
        answer merely quoting `<tool_call><function=NAME>` opens a region too.
        A name the client never offered is overwhelmingly likelier to be prose
        than a call, so it waits for the region to close like everything else.
        SGLang's cursor parsers announce with no such check and will emit a
        call named after whatever follows the tag.

        Asked of `Region.head` -- the first `_PEEK_WINDOW` bytes and no more
        -- and asked at most once more after it fills. Running a format's
        regex over the whole region on every chunk is quadratic in the
        response and measured 3.0 -> 9.8 -> 36 -> 137 ms across 2k/4k/8k/16k
        tokens: the shape `marker_scanner` exists to retire, put back one
        layer up.
        """
        # 【译】一旦区域显露出工具名就立即发送，只发一次。
        #
        #   名字是从 ``parse_region`` 本身读出来的，作用于目前为止的区域，并使用 ``at_end=False``。正是这一点让它与最终发出的调用一致：
        #   同一个函数、同一种枚举，第二次运行看到的字节是第一次的超集。过去每种格式都用自己的正则来回答这个问题，五种里有四种
        #   与自己的解析不一致——最近一次是一个自闭合的 ``<invoke name="x"/>``，偷看跳过了它而解析把它作为第一个返回，
        #   使得一个包含两个调用的响应在线上出现了三个工具调用。
        #
        #   只针对请求确实声明了的名字。这个检查是提前发送名字安全的原因：它无法撤回，而一个仅仅引用了
        #   `<tool_call><function=NAME>` 的回答同样会打开一个区域。客户端从未提供过的名字，是普通文字的可能性远大于是调用，
        #   所以它和其他一切一样等待区域闭合。SGLang 的游标式解析器播报时没有这样的检查，会发出一个以标签后面任意内容命名的调用。
        #
        #   只问 `Region.head`——前 `_PEEK_WINDOW` 个字节，不再多——并且在它填满后最多再问一次。每个chunk都在整个区域上跑格式的正则，
        #   对响应长度是平方复杂度的，实测在2k/4k/8k/16k token上分别为 3.0 -> 9.8 -> 36 -> 137 ms：这正是 `marker_scanner` 要消除的形状，
        #   却在上一层又放了回来。
        if (
            self._announced is not None
            or self._peek_exhausted
            or self.suppress_calls
            or not self.tools
        ):
            return []
        head = self._region.head
        parsed = self.parser_cls.parse_region(head, self.tools, at_end=False)
        if not parsed.calls:
            self._peek_exhausted = len(head) >= _PEEK_WINDOW
            return []
        if parsed.begins > 0:
            # There is answer text ahead of the call *inside* this region --
            # an answer that quotes a marker and then calls for real. That
            # text is only released when the region closes, so announcing now
            # would put the call in front of prose that `stream=false` puts
            # behind it, and a client that closes its text pane on a tool call
            # renders the explanation after the call. The name still goes out,
            # with the arguments, in the right order.
            # 【译】在这个区域*内部*、调用之前还有回答文字——一个先引用标记再真正调用的回答。那段文字只会在区域闭合时释放，
            #   所以现在播报会把调用放到 `stream=false` 放在它后面的文字前面，而一个遇到工具调用就关闭文本面板的客户端，
            #   会把解释渲染在调用之后。名字仍然会和参数一起、按正确的顺序发出。
            self._peek_exhausted = True
            return []
        name = parsed.calls[0].function["name"]
        if self._declared is None:
            self._declared = _peek_declared(self.tools)
        if name not in self._declared:
            self._peek_exhausted = True
            return []
        self._announced = name
        return [
            (
                "tool_call_start",
                {
                    "index": self._index,
                    # The parsed call's own id, not a fresh one. Announcing
                    # skips the start event the call would otherwise send, so
                    # whatever goes out here is the only id the client sees --
                    # and Kimi's is the model's `functions.NAME:INDEX`, which
                    # survived only for requests that declared no tools.
                    # 【译】用解析出的调用自己的id，而不是新生成一个。播报会跳过该调用本来要发送的start事件，所以这里发出的就是客户端看到的唯一id——
                    #   而 Kimi 的id是模型自己的 `functions.NAME:INDEX`，过去只有在请求没有声明tools时才能保留下来。
                    "id": parsed.calls[0].id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                },
            )
        ]

    # 生成一个调用的 tool_call_start 事件；若该调用的名字已经提前播报过，则不重复发送
    def _start_event(self, call: ToolCall) -> list:
        """The `tool_call_start` for a call, unless its name already went out.

        A mismatch is not reachable: the announcement is the first call of
        ``parse_region`` over a prefix of these same bytes, and acceptance is
        monotone in what has arrived. It is checked rather than asserted
        because the caller is on a live SSE stream that has already sent its
        200 -- an exception here reached the client as a connection cut
        mid-frame with no `[DONE]` and, on the `n>1` path, took the other
        choices with it.

        So it is logged and recovered from. The announced name is already on
        the wire and cannot be retracted, but it can be left with no arguments
        -- the shape every other unfulfilled announcement takes, and one that
        `completes_a_tool_call` and `finish_reason` both read as "not a call".
        The real call goes out at the next index, whole.
        """
        # 【译】一个调用的 `tool_call_start`，除非它的名字已经发出去了。
        #
        #   不匹配是不可达的：播报是 ``parse_region`` 在这些相同字节的一个前缀上的第一个调用，而接受与否随到达的内容单调。
        #   之所以检查而不是断言，是因为调用方处在一个已经发出200的实时SSE流上——这里抛异常，会以一次帧中途的连接中断到达客户端，
        #   没有 `[DONE]`，而且在 `n>1` 路径上还会把其他choice一起带走。
        #
        #   所以记录日志并恢复。已播报的名字已经在线上，无法撤回，但可以让它没有参数——这是其他所有未兑现的播报的形状，
        #   `completes_a_tool_call` 和 `finish_reason` 都会把它读成"不是一个调用"。真正的调用在下一个index上完整发出。
        name = call.function["name"]
        if self._announced is not None and self._announced != name:
            logger.warning(
                "%s announced %r and then parsed %r from the same region; "
                "sending %r as a new call and leaving %r without arguments",
                self.parser_cls.__name__,
                self._announced,
                name,
                name,
                self._announced,
            )
            self._announced = None
            # Past the dangling announcement, so the real call does not land
            # on an index the client has already bound to the wrong name.
            # 【译】越过这个悬空的播报，这样真正的调用就不会落在客户端已经绑定到错误名字的index上。
            self._index += 1
            return self._start_event(call)
        if self._announced is not None:
            # 【译】（行尾注释）只绑定到区域的第一个调用
            self._announced = None  # binds to the first call of the region only
            return []
        return [
            (
                "tool_call_start",
                {
                    "index": self._index,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                },
            )
        ]

    # 把一个解析出的调用变成 start + args 两个事件，使用引擎自己的index，然后index加1
    def _emit_call(self, call: ToolCall) -> list:
        """One parsed call as start+args events, at the engine's own index.

        The arguments go out unconditionally, empty ones included: a
        zero-parameter tool is still a call the client should run, and
        `finish_reason` keys on this event. Gating it reported `stop` for a
        response that had already sent a `tool_calls` delta.
        """
        # 【译】把一个解析出的调用变成start+args事件，使用引擎自己的index。
        #
        #   参数总是无条件发出，包括空参数：一个零参数的工具仍是客户端应该执行的调用，而 `finish_reason` 以这个事件为依据。
        #   给它加条件，曾让一个已经发出 `tool_calls` delta 的响应报告 `stop`。
        events = self._start_event(call)
        events.append(
            (
                "tool_call_args",
                {
                    "index": self._index,
                    "function": {"arguments": call.function["arguments"]},
                },
            )
        )
        self._index += 1
        self._emitted += 1
        return events


# 非流式：完整输出 -> (content, [ToolCall])，即引擎在单个chunk上 process + flush，再展平事件
#   调用方：registry.parse_tool_calls
def read_whole(
    parser_cls: type[ToolCallParser] | None,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> tuple[str, list[ToolCall]]:
    """A complete output as ``(content, calls)`` -- the engine, in one chunk.

    Not a second implementation of anything. `stream=false` and `stream=true`
    read the model's words with the same code, so the four rules they used to
    answer separately -- where content ends, whether an unclosed tag is a call,
    what a region that parses to nothing means, which bytes are framing -- have
    one answer each and no way to drift.

    Content comes back byte-for-byte except for markers the format declares:
    the region a call occupied, and the framing `opens_region` says no to.
    Nothing is trimmed. A trailing `.strip()` here cost a code-block answer its
    final newline, which streaming had no way to reproduce.
    """
    # 【译】把一份完整输出变成 ``(content, calls)``——引擎在一个chunk上运行。
    #
    #   不是任何东西的第二份实现。`stream=false` 和 `stream=true` 用同一份代码读模型的话，所以它们过去分别回答的四条规则——
    #   内容在哪里结束、一个未闭合的标签是否是调用、一个解析不出东西的区域意味着什么、哪些字节是框架标记——每条只有一个答案，不会漂移。
    #
    #   返回的内容除了格式声明的标记之外逐字节保留：调用占据的区域，以及 `opens_region` 说"否"的框架标记。
    #   什么都不裁剪。这里曾有一个末尾的 `.strip()`，让一个代码块回答丢了最后的换行，而流式路径无法复现这一点。
    return flatten_tool_events(
        read_whole_events(parser_cls, text, tools, suppress_calls=suppress_calls)
    )


# 非流式：完整输出 -> 引擎的有序事件列表（保留文字与调用的相对顺序）
def read_whole_events(
    parser_cls: type[ToolCallParser] | None,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> list:
    """The same single chunk, as the engine's own ordered events.

    `read_whole` flattens these into `(content, calls)`, which loses where the
    content sat relative to the calls. A caller that renders blocks in order
    -- the Anthropic non-streaming path -- needs that order, and building it
    from a joined string put a response's whole answer in one block ahead of
    every `tool_use`, so `stream=false` disagreed with `stream=true` about the
    order of the blocks in the same generation.
    """
    # 【译】同样的单个chunk，以引擎自己的有序事件形式返回。
    #
    #   `read_whole` 把这些事件展平成 `(content, calls)`，这会丢掉内容相对于调用的位置。按顺序渲染块的调用方——Anthropic 非流式路径——
    #   需要这个顺序，而从拼接后的字符串构建它，会把响应的整个回答放进一个块、排在每个 `tool_use` 前面，
    #   于是对同一次生成，`stream=false` 与 `stream=true` 给出的块顺序不一致。
    # 解释：目前 Anthropic 非流式路径用的是 serving_anthropic.py 的 read_whole_blocks（它还要交错处理推理内容），
    #   read_whole_events 在 atom/ 内没有调用方，只由 __init__.py 导出
    engine = ToolCallStreamParser(
        tools=tools, parser_cls=parser_cls, suppress_calls=suppress_calls
    )
    events = engine.process(text)
    events.extend(engine.flush())
    return events


# 把有序事件展平成 (拼接后的content, [ToolCall])；api_server.py 的 /v1/messages 非流式路径也用它从事件中取出调用
def flatten_tool_events(events: list) -> tuple[str, list[ToolCall]]:
    """Ordered events as `(joined content, calls)`."""
    # 【译】把有序事件变成 `(拼接后的content, calls)`。
    content: list[str] = []
    calls: list[ToolCall] = []
    pending: dict | None = None
    for etype, data in events:
        if etype == "content":
            content.append(data)
        elif etype == "tool_call_start":
            fn = data["function"]
            pending = {"id": data["id"], "name": fn["name"]}
        elif etype == "tool_call_args":
            if pending is None:
                # An announcement is what normally supplies the name, and it
                # cannot go missing between the two events of one call -- but
                # a format whose peek and parse disagree leaves a start event
                # behind without one. Dropping the arguments is what the
                # streaming clients do with the same pair.
                # 【译】名字通常由播报提供，它不可能在一个调用的两个事件之间丢失——但偷看与解析不一致的格式会留下一个没有名字的start事件。
                #   丢弃这些参数，正是流式客户端对同样一对事件的处理方式。
                continue
            calls.append(
                ToolCall(
                    id=pending["id"],
                    type="function",
                    function={
                        "name": pending["name"],
                        "arguments": data["function"]["arguments"],
                    },
                )
            )
            pending = None
    return "".join(content), calls
