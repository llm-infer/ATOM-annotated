# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How much of a stream can be released without splitting a marker.

One question, one answer. Everything on the streaming path that has to notice
a literal in the model's output -- the reasoning channel's delimiters, the
tool-call formats' opening tags -- asks it, and asking it in more than one
place is how the four incompatible answers this replaces came about.

The rule: release everything except the longest *suffix* of the buffer that is
a prefix of some marker. Not "hold everything once a marker's first character
appears anywhere", which is the shape being retired: it withholds on a '<' in
the middle of an answer that will never become a tag, and the buffer it holds
grows without bound, so the scan is O(n) per chunk and O(n^2) over a response.
Measured at 64 KB of answer that cost 515 ms of pure host CPU against 6 ms
here, and the first byte of a '<'-bearing answer never reached the client
until the stream ended.

The `'<'` test is still the fast path -- it was the right idea at the wrong
scope. It rejects the common chunk, which carries no marker character at all,
before any per-marker work happens.

Both halves run once per token per stream, so how they are spelled shows up in
event-loop time rather than in a profile of the model. Two spellings that read
as equivalent are not: `frozenset.intersection(buf)` hashes every character of
`buf` (2.9 us on 900 chars) where `any(c in buf ...)` is a C substring search
per marker character (0.12 us, 25x), and the buffer is not bounded by a marker
on the way in -- callers concatenate a backlog into `text`.

Judge a change here on a whole response through the real pipeline -- reasoning
filter into tool parser, four-character tokens -- and not on one function.
Alternating arms, three rounds each: 1579 -> 1383 ns/token on plain prose and
1652 -> 1537 on an answer ending in a tool call. A single cross-process pair
said the opposite before the arms were alternated.
"""

# 【译】一个流能放出多少而不切断某个标记。
#
#   一个问题，一个答案。流式路径上凡是需要在模型输出中发现某个字面量的地方——推理通道的分隔符、
#   各工具调用格式的开标签——都来问它；在不止一个地方回答这个问题，正是它所取代的四种互不兼容的答案产生的原因。
#
#   规则：除了缓冲区中"是某个标记前缀"的最长*后缀*之外，其余全部放出。而不是"一旦某个标记的首字符出现在任何地方
#   就全部暂扣"——那是要淘汰的做法：它会因为回答中间一个永远不会变成标签的'<'而暂扣，而且暂扣的缓冲区无限增长，
#   于是每个chunk的扫描是O(n)，整个响应是O(n^2)。在64KB的回答上实测，那种做法耗费515ms纯主机CPU，这里只要6ms；
#   而且一个含'<'的回答，第一个字节要等到流结束才能到达客户端。
#
#   `'<'`测试仍然是快速路径——想法是对的，只是用错了范围。它在任何逐标记的工作发生之前，
#   就把常见的chunk（根本不含任何标记字符）排除掉。
#
#   两部分在每个流的每个token上都要跑一次，所以写法上的差别体现在事件循环的耗时里，而不是模型的profile里。
#   两种看起来等价的写法其实不等价：`frozenset.intersection(buf)`要对`buf`的每个字符做哈希（900个字符2.9us），
#   而`any(c in buf ...)`对每个标记字符做一次C实现的子串搜索（0.12us，快25倍）；而且传入的缓冲区长度并不受标记长度限制——
#   调用方会把积压的内容拼进`text`。
#
#   评估这里的改动，要看完整响应在真实流水线上的表现——推理过滤器接工具调用解析器、每个token四个字符——而不是只看一个函数。
#   两组交替运行、各三轮：普通文本从1579降到1383 ns/token，以工具调用结尾的回答从1652降到1537。
#   在交替运行之前，单次的跨进程对比给出的结论恰好相反。
#
# 【导读】流式标记扫描器：决定每个chunk里哪些字节可以立即发给客户端、哪些要暂扣（可能是某个标记的开头）
# 调用方：reasoning.ReasoningFilter（推理通道的开/结束标记、外框）和tool_parser/stream.py的ToolCallStreamParser
#   （各工具调用格式的START_MARKERS）；两者共用这一个实现，保证"何时放出"的规则只有一份
# 与主干的关系：服务层SSE输出路径上的纯CPU工具，不涉及引擎；每个流的每个token都要调用，所以非常在意常数开销
# 算法示例（标记集合{"<think>", "</think>"}）：
#   feed("答案是 a<b")  -> 尾部"b"、"<b"都不是任何标记的前缀，全部放出（中间的'<'不会造成暂扣）
#   feed("嗯<th")       -> 尾部"<th"是"<think>"的前缀：放出"嗯"，暂扣"<th"（暂扣长度永远小于最长标记长度）
#   feed("ink>推理")    -> 缓冲区"<think>推理"中出现完整标记：返回Scan(released="", hit="<think>", rest="推理")，
#                          rest交还调用方（由它根据新状态决定怎么处理）

from __future__ import annotations

import functools
from dataclasses import dataclass


# 单个标记版本的"可暂扣后缀长度"
def partial_suffix_len(text: str, marker: str) -> int:
    """Length of the longest proper prefix of `marker` that ends `text`.

    `("if (a < b", "<tool_call>")` is 0 -- the '<' is not at the end, so
    nothing here can still grow into the marker. `("... <tool_", ...)` is 6.
    Bounded by `len(marker) - 1`: a whole marker is not a partial one, and its
    caller has already looked for complete ones.

    Kept for a single marker asked about in isolation. :class:`MarkerScanner`
    does not call it: over a set of markers it re-slices the same suffixes once
    per marker, which is the work `_prefixes_by_len` precomputes away.
    """
    # 【译】`text`结尾处、同时是`marker`真前缀的最长长度。
    #
    #   `("if (a < b", "<tool_call>")`为0——'<'不在结尾，所以这里没有任何东西还能长成这个标记。`("... <tool_", ...)`为6。
    #   上界是`len(marker) - 1`：完整的标记不算部分标记，而且调用方已经找过完整的了。
    #
    #   保留它是为了单独询问某一个标记的场景。:class:`MarkerScanner`不调用它：对一组标记，它会为每个标记重复切出同样的后缀，
    #   而这正是`_prefixes_by_len`预先算好、从而省掉的工作。
    # 解释：v0.1.8中atom/内没有调用方，只有tests/entrypoints/test_marker_scanner.py使用
    for k in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


# 第一层缓存：以调用方传入的原样元组为键（lru_cache要求参数可哈希），内部再调用以规范化集合为键的_plan
@functools.lru_cache(maxsize=64)
def _plan_for(markers) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """:func:`_plan`, keyed on the marker *set* rather than how it was spelled.

    Two caches, and both earn their place. `_plan` is keyed on the normalised
    set, so two spellings of one set share a single plan *object*. This one is
    keyed on the spelling, so the callers that ask once per token -- always
    with the same module-level constant -- skip the normalisation as well:
    `sorted(set(...))` on every call was 148 ns of the 193 it took to answer.
    """
    # 【译】:func:`_plan`，按标记*集合*而不是它的写法来取。
    #
    #   两层缓存，各有其用。`_plan`以规范化后的集合为键，所以同一集合的两种写法共享同一个计划*对象*。
    #   这一层以写法为键，所以每个token都来问一次的调用方——总是传同一个模块级常量——连规范化也省掉了：
    #   每次调用都做`sorted(set(...))`，占了总共193ns中的148ns。
    return _plan(tuple(sorted(set(markers))))


# 核心循环：返回可暂扣的尾部长度；ordered/firsts/by_len来自_plan
def _suffix_len(text: str, ordered: tuple, firsts: tuple, by_len: dict) -> int:
    """Longest suffix of `text` that is a proper prefix of one of the markers.

    The one loop. It was written twice -- once here for callers holding their
    own buffer and once as a method on the scanner -- in a module whose whole
    thesis is that asking this question in more than one place is how the
    answers drift apart.

    Nothing can be held unless the tail *starts* a marker, so a marker's first
    character has to appear within the last `longest - 1` bytes. One substring
    search over that window answers that, and it is a bet rather than a free
    win: 2.4x when it rejects (737 ns -> 307 ns) against 1.3x slower when it
    does not (847 ns -> 1108 ns), because the loop then repeats the search it
    just paid for. A chunk whose last twenty bytes contain a marker's opening
    character is the rare one, and this runs once per token on every stream
    including models whose markers never appear at all.
    """
    # 【译】`text`中同时是某个标记真前缀的最长后缀。
    #
    #   唯一的那个循环。它曾被写过两遍——一遍在这里给自己持有缓冲区的调用方，一遍作为扫描器的方法——
    #   而本模块的核心论点恰恰是：在不止一个地方回答这个问题，就是各处答案逐渐分歧的原因。
    #
    #   只有尾部*开头*是某个标记时才可能需要暂扣，所以标记的首字符必须出现在最后`longest - 1`个字节之内。
    #   对这个窗口做一次子串搜索就能回答，而这是个赌注而非白捡的好处：排除时快2.4倍（737ns -> 307ns），
    #   没排除时慢1.3倍（847ns -> 1108ns），因为接下来的循环会重复刚刚付过代价的搜索。最后二十个字节里含有标记首字符的
    #   chunk是少数，而这段代码在每个流的每个token上都要跑一次，包括那些标记从不出现的模型。
    longest = len(ordered[0])
    if longest == 1:
        # 【译】（行尾注释）单字符标记没有真前缀
        return 0  # a one-character marker has no proper prefix
    if not any(c in text[-(longest - 1) :] for c in firsts):
        return 0
    # 从长到短试：先用首字符快速过滤，再查该长度的前缀集合
    for k in range(min(longest - 1, len(text)), 0, -1):
        if text[-k] in firsts and text[-k:] in by_len[k]:
            return k
    return 0


# 第二层缓存：对一个标记集合预先计算（从长到短的标记, 首字符集合, {长度k: 所有长度为k的真前缀}）
@functools.lru_cache(maxsize=64)
def _plan(markers: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """Everything about a marker set that does not depend on the stream.

    Cached on the set, not computed per scanner: a scanner is built per
    request and the sets are class constants -- one per tool-call format plus
    the reasoning dialects' -- so this runs a handful of times per process.
    Building it per scanner cost 8.7 us of request setup to save 4.5 us per
    chunk, which is the right trade only for streams longer than two chunks.

    Returns the markers longest-first, their distinct first characters, and
    every proper prefix grouped by length.

    Longest-first settles a tie only between markers that are *both already
    complete in the buffer*. A chunk ending exactly at the shorter of a
    prefix pair reports the shorter one, because the longer one has not
    arrived to be preferred -- so which of the two fires is a function of
    where the chunk boundary landed, not of the text. `_earliest_complete`
    says the same thing from the other side.

    Nothing here withholds a complete match on the chance it could still
    grow; that would be the stronger rule, and it would cost every marker
    that is a prefix of another a wait of the length difference. It is not
    needed while no format has a prefix pair whose halves disagree about
    handing the stream over -- which is what
    `TestAPrefixPairCannotChangeTheHandover` holds them to. The moment one
    does, this becomes "the region opened, or did not, depending on
    chunking", and the stronger rule has to be written.
    """
    # 【译】标记集合中与流无关的全部信息。
    #
    #   按集合缓存，而不是每个扫描器各算一遍：扫描器每个请求建一个，而集合是类常量——每种工具调用格式一个，
    #   再加上推理方言的——所以每个进程只会运行寥寥几次。每个扫描器各建一次的话，请求准备要多花8.7us，
    #   换来每个chunk省4.5us，只有对超过两个chunk的流才划算。
    #
    #   返回：按长度从长到短排列的标记、它们互不相同的首字符，以及按长度分组的所有真前缀。
    #
    #   "最长优先"只在*两个标记都已在缓冲区中完整出现*时才用来打破平局。一个chunk恰好结束在前缀对中较短的那个标记处时，
    #   报告的是较短的那个，因为较长的那个还没到，无从优先——所以两者中哪个触发，取决于chunk边界落在哪里，而不取决于文本。
    #   `_earliest_complete`从另一面说的是同一件事。
    #
    #   这里不会因为一个完整匹配还有可能继续变长而暂扣它；那是更强的规则，代价是每个作为其他标记前缀的标记
    #   都要多等两者长度之差。只要没有哪种格式存在"前缀对的两半对是否交出流意见不一"的情况，就不需要它——
    #   `TestAPrefixPairCannotChangeTheHandover`就是约束这一点的测试。一旦出现这样的格式，这里就会变成
    #   "区域是否打开取决于分块方式"，那时就必须写出更强的规则。
    ordered = tuple(sorted(set(markers), key=len, reverse=True))
    firsts = tuple(sorted({m[0] for m in ordered}))
    by_len: dict[int, set[str]] = {}
    for m in ordered:
        for k in range(1, len(m)):
            by_len.setdefault(k, set()).add(m[:k])
    return ordered, firsts, {k: frozenset(v) for k, v in by_len.items()}


# 无状态版本：给自己维护缓冲区的调用方
def held_suffix_len(text: str, markers: tuple[str, ...]) -> int:
    """Longest suffix of `text` that is a proper prefix of any of `markers`.

    The stateless form of what :class:`MarkerScanner` withholds, for callers
    that own their own buffer. Same cached plan and the same loop, so they get
    the same answer rather than a second implementation of the rule.
    """
    # 【译】`text`中同时是`markers`任一真前缀的最长后缀。
    #
    #   这是:class:`MarkerScanner`所暂扣内容的无状态形式，给自己持有缓冲区的调用方用。缓存的计划和循环都相同，
    #   所以它们得到同样的答案，而不是规则的第二种实现。
    # 解释：v0.1.8中atom/内没有调用方，只有tests/entrypoints/test_marker_scanner.py使用
    if not markers:
        return 0
    return _suffix_len(text, *_plan_for(markers))


# MarkerScanner.feed()的返回值：released可立即发送；hit/rest为命中的标记和它之后的剩余文本
@dataclass(frozen=True)
class Scan:
    """What one chunk produced.

    `released` is safe to send now. `hit` is the marker that completed, if one
    did, and `rest` is everything after it -- handed back rather than kept,
    because who owns the text after a marker is the caller's decision and not
    this class's.
    """

    # 【译】一个chunk产出的结果。
    #
    #   `released`可以立即发送。`hit`是完整出现的那个标记（如果有），`rest`是它之后的全部内容——交还给调用方而不是自己留着，
    #   因为标记之后的文本归谁处理，是调用方的决定，不是这个类的。

    released: str
    hit: str | None = None
    rest: str = ""


# 有状态扫描器：每个流一个实例（ReasoningFilter按当前状态创建、tool_parser/stream.py按工具调用格式创建）
class MarkerScanner:
    """Incremental reader over a stream that must not split a marker.

    Stateful across chunks and cheap to hold: what it *withholds* is bounded by
    the longest marker, which is what makes the withhold bounded and the cost
    per chunk independent of how long the response runs. What it *scans* is
    that tail plus the incoming chunk, which the caller sizes.
    """

    # 【译】在一个不能切断标记的流上做增量读取。
    #
    #   跨chunk有状态，但持有成本很低：它*暂扣*的内容以最长标记为上界，这使得暂扣有界，每个chunk的开销与响应多长无关。
    #   它*扫描*的是那段尾部加上新到的chunk，chunk大小由调用方决定。

    def __init__(self, markers: tuple[str, ...]):
        if not markers or any(not m for m in markers):
            raise ValueError("a scanner needs at least one non-empty marker")
        self._plan = _plan_for(markers)
        self._markers, self._firsts, self._prefixes_by_len = self._plan
        self._longest = len(self._markers[0])
        self._buf = ""

    @property
    def held(self) -> str:
        """What is being withheld right now. Bounded by the longest marker."""
        # 【译】当前正被暂扣的内容。以最长标记为上界。
        return self._buf

    # 喂入一个chunk：快速路径（不含任何首字符）-> 找最早的完整标记 -> 否则只暂扣"可能是标记前缀"的尾部
    def feed(self, text: str) -> Scan:
        buf = self._buf + text
        if not any(c in buf for c in self._firsts):
            # Nothing here can begin a marker, so nothing needs holding: the
            # fast path, not a shortcut past correctness. `in` on a str is a
            # C substring search; a set intersection would hash every
            # character of `buf` instead, which is 25x more for the same
            # answer and is paid once per token per stream.
            # 【译】这里没有任何东西能作为标记的开头，所以什么都不用暂扣：这是快速路径，不是绕过正确性的捷径。
            #   对str用`in`是C实现的子串搜索；集合求交则要对`buf`的每个字符做哈希，同样的答案要多花25倍，而且每个流的每个token都要付一次。
            self._buf = ""
            return Scan(buf)

        at, hit = self._earliest_complete(buf)
        if hit is not None:
            self._buf = ""
            return Scan(buf[:at], hit, buf[at + len(hit) :])

        cut = len(buf) - _suffix_len(buf, *self._plan)
        self._buf = buf[cut:]
        # The invariant the whole class exists for: a stall is not something
        # to test for here, it is something that cannot be represented.
        # 【译】整个类存在的意义就是这条不变式：卡顿在这里不是需要测试的情况，而是根本无法表示的状态。
        assert len(self._buf) < self._longest, "withheld more than a marker could be"
        return Scan(buf[:cut])

    def flush(self) -> str:
        """Release the held tail at end of stream; it never became a marker."""
        # 【译】流结束时放出暂扣的尾部；它最终没有成为标记。
        out, self._buf = self._buf, ""
        return out

    def _earliest_complete(self, buf: str) -> tuple[int, str | None]:
        """Where the first complete marker starts, and which one it is.

        Earliest wins, and at the same position the longest does -- `<think>`
        must not be reported where `<thinking>` was meant when both are
        registered. Only among those already complete in `buf`, though: see
        `_plan` for why that is weaker than it sounds, and what holds the
        registered formats inside the gap.
        """
        # 【译】第一个完整标记从哪里开始，以及是哪一个。
        #
        #   最早的胜出，同一位置则最长的胜出——两个都注册时，本该是`<thinking>`就不能报告成`<think>`。
        #   不过只在`buf`中已经完整出现的标记之间比较：为什么这比听起来要弱、以及是什么把已注册的格式约束在这个缺口之内，见`_plan`。
        best_at, best = len(buf), None
        # 【译】（行尾注释）已经按从长到短排好序
        # 解释：因为按从长到短遍历且用严格小于比较，同一位置上先遇到的（更长的）标记会保留下来
        for m in self._markers:  # already longest-first
            at = buf.find(m)
            if 0 <= at < best_at:
                best_at, best = at, m
        return best_at, best
