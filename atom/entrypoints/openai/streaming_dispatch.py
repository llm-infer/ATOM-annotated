# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-thread dispatch and per-request delivery for streaming model output.

Two halves of one hand-off. :class:`StreamBatchDispatcher` runs on the engine
output threads and schedules a whole engine step per event loop.
:class:`StreamOutputCollector` merges pending token deltas before decoding them
when the SSE consumer reads, so a slow consumer does not accumulate decode work.
"""

# 【译】流式模型输出的跨线程分发与逐请求投递。
#
#   一次交接的两半。StreamBatchDispatcher 运行在引擎输出线程上，按事件循环调度一整个引擎step。
#   StreamOutputCollector 在SSE消费者读取时，先合并待处理的token增量再解码，这样慢消费者不会积压解码工作。
#
# 本文件（第2批2c，B级；它是流式路径的核心机制）：引擎输出线程 → 事件循环 的交接，以及增量反分词
#   [输出线程] 流式回调 → StreamBatchDispatcher.enqueue（缓冲到threading.local）
#             → flush（每个目标事件循环一次 loop.call_soon_threadsafe(_deliver, items)）
#   [事件循环] _deliver → StreamOutputCollector.put_nowait（还没被读走的chunk就合并）→ Event.set()
#             → SSE生成器 await collector.get() → IncrementalStreamDetokenizer.update（token id → 新增文本）
#   示意图见api_server.py文件头的图2；调用方：api_server.setup_streaming_request*、serving_chat / serving_completion
# 前置（事件循环与线程，只说本文件用到的；完整讲解见api_server.py文件头的asyncio条目和图2）：
#   API server进程里与本文件有关的线程有两类：uvicorn的事件循环线程（一个；所有请求的协程、asyncio.Event都属于它），和CoreManager的
#   输出线程（每个DP rank一个，从ZMQ收EngineCore的结果、调用各序列的回调）。asyncio的对象不是线程安全的，输出线程不能直接
#   Event.set()，只能用loop.call_soon_threadsafe(fn, ...)把fn排进事件循环、由事件循环线程去执行；每调用一次就唤醒一次
#   事件循环，所以这里按step攒批：并发1000个流时，一个step是1次调用而不是1000次。另外CPython有GIL（同一时刻只有一个线程
#   执行Python代码），几个输出线程各自反分词并不会并行；放到事件循环的get()里做，还能先把落后的chunk合并再只解码一次
# 阅读路线（本文件约890行；定义顺序与阅读顺序不同）：
#   第一遍只读主路径：上面的"前置" → IncrementalStreamDetokenizer上方的【科普】（含BPE与UTF-8的前置）与update / _decode
#     → merge_chunk与StreamOutputCollector（合并即背压）→ 文件末尾的StreamBatchDispatcher（enqueue → flush → _deliver）
#   其余部分是优化与观测，第一遍可跳过：
#   _DeltaReuse / enable_delta_reuse / _probe*：反分词的"增量复用"优化，启动时用探测文本验证安全后才开启
#   FrameWait / longest_silence_seconds：流的静默看门狗（api_server._client_stream使用，/metrics导出）
#   StreamDeliveryTiming：token间延迟（ITL）计时
# 对比nano-vllm：nano-vllm离线结束后一次性decode全部输出，不需要增量反分词，也没有跨线程投递
import array
import logging
import random
import threading
import time
from asyncio import AbstractEventLoop, Event
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from atom.model_engine.sequence import new_token_ids

logger = logging.getLogger("atom")

# How long one stream may go without a chunk before it is worth a line. Long
# enough that a slow prefill or a busy engine step never trips it; short
# enough that a wedged response is named while the client is still waiting.
# 【译】一个流多久没有chunk就值得打一行日志。足够长，慢的prefill或繁忙的引擎step不会触发它；
#   足够短，卡死的响应能在客户端还在等的时候被点名。
SILENCE_LOG_SECONDS = 30.0

# Every stream currently waiting to hand its client a frame, and since when.
# Keyed by `id()`: entries are added and removed around a single await in one
# place, so an id cannot outlive the watch that owns it.
# 【译】当前每个正在等着交给客户端一帧的流，以及从何时开始等。以 id() 为key：条目在同一处、围绕同一个await加入和删除，
#   所以一个id不会比拥有它的计时器活得更久。
_WAITING_SINCE: dict[int, float] = {}

# Fields a later chunk overrides on the one it merges into, when it has a value
# of its own. The SSE consumers keep the newest non-empty value they see, so
# merging this way hands them what reading each chunk separately would have.
# 【译】后来的chunk合并进前一个时，若自己有值就覆盖的字段。SSE消费者保留它们看到的最新非空值，
#   所以这样合并交给它们的结果，与逐个读取每个chunk时一样。
_LATEST_WINS = ("finish_reason", "kv_transfer_params", "num_cached_tokens")

# Stands in for the decoded text of one token when the run has declared its own
# output meaningless -- forced speculative acceptance, today. It says what it is,
# so a dump of such a run cannot be mistaken for something the model wrote, and
# it carries no marker any dialect or tool-call format intercepts, so it reaches
# the client whatever the model is.
# 【译】当一次运行声明自己的输出没有意义时（目前是强制投机接受），用它代替一个token的解码文本。它直接说明自己是什么，
#   所以这种运行的输出转储不会被误认为是模型写的；它也不含任何方言或工具调用格式会拦截的标记，所以无论什么模型，它都能到达客户端。
SYNTHETIC_TOKEN_TEXT = "synthetic "

# One audit per thousand reuses, which is what the shipped 1.7x was measured at.
# 【译】每复用一千次审计一次，交付的1.7倍加速就是在这个设置下测得的。
DEFAULT_AUDIT_EVERY = 1000


# 进程级的"增量复用"开关与统计（全局唯一实例_DELTA_REUSE）：
#   开启时，IncrementalStreamDetokenizer._decode用上次输出的增量代替第一次decode，每audit_every次仍真解码一次做核对，
#   一旦不一致就在全进程关闭复用
@dataclass
class _DeltaReuse:
    """Whether `_decode` may skip its first `tokenizer.decode`, process-wide.

    Off until `enable_delta_reuse` has watched this tokenizer agree with it.
    Consulted per call rather than captured per stream, so switching it off
    reaches the four thousand streams already in flight, not only the next one.
    """

    # 【译】进程范围内：_decode 是否可以跳过它的第一次 tokenizer.decode。
    #
    #   在 enable_delta_reuse 观察到这个tokenizer与它一致之前一直关闭。每次调用都去查询，而不是每个流各自捕获一份，
    #   所以关掉它时会影响到已经在途的四千个流，而不只是下一个流。

    enabled: bool = False
    audit_every: int = DEFAULT_AUDIT_EVERY
    calls: int = 0
    # 【译】（行尾注释）真正有东西可比的比较次数
    audits: int = 0  # comparisons that had something to compare
    mismatches: int = 0
    # The probe's negative control spoils a delta on purpose, and an ERROR
    # saying reuse is off, followed by the line saying it is on, is a log
    # nobody can read.
    # 【译】探测中的阴性对照会故意弄坏一个增量；先出一条说复用已关闭的ERROR、紧接着又一行说它已开启，这样的日志没人看得懂。
    probing: bool = False

    def should_audit(self) -> bool:
        # `calls` starts at zero, so a stream's first update is always checked
        # and a tokenizer the probe let through still meets a comparison at once.
        # 【译】calls 从0开始，所以一个流的第一次更新总会被检查，被探测放行的tokenizer也会立即遇到一次比较。
        # 解释：calls是进程级计数，严格说必定被检查的是进程内的第一次调用，之后每audit_every次检查一次
        due = self.calls % self.audit_every == 0
        self.calls += 1
        return due

    # 发现复用结果与真实解码不一致：在全进程关闭复用（探测期间不打ERROR日志）；当次调用已经用了真实解码值，输出仍正确
    def report_mismatch(self, reused: str, decoded: str) -> None:
        self.mismatches += 1
        if self.enabled:
            self.enabled = False
            if self.probing:
                return
            logger.error(
                "[detokenizer] the reused delta disagreed with the tokenizer "
                "(reused %r, decoded %r); delta reuse is now off for this "
                "process. Output stays correct -- this call used the decoded "
                "value -- but the tokenizer decodes a token span differently "
                "depending on where the window starts, which the startup probe "
                "did not reach.",
                reused,
                decoded,
            )


_DELTA_REUSE = _DeltaReuse()


# 增量反分词器：每个流一个，每次拿到新token，只返回"新增的文本"
# 【科普】反分词（detokenize）与两个偏移量 —— 从"模型每步吐出一个token id"讲起
#   模型每个decode step输出一个（投机解码时可能是几个）token id。tokenizer的词表把每个id对应到一段字节，
#   decode就是"按id查出字节、按顺序拼起来、再按UTF-8变成字符串"。读懂下面的图需要两点前置：
#   · UTF-8是变长编码，一个字符占1~4个字节：ASCII（如"!"）1个，常用汉字（如"你"）3个，emoji（如😀）4个；
#     只拿到一个字符的前几个字节时解不出这个字符，decode会用替换字符U+FFFD（显示为�）顶上
#   · 字节级BPE（Qwen、DeepSeek等模型的tokenizer都属于这一类）：从256个单字节出发，训练时把语料里最常一起出现的相邻片段
#     反复合并成新token。常见的字、词合并成了一个token；不常见的字符（很多emoji、生僻字）可能只合并到一半，被拆成几个token
#   所以字节级BPE的词表里，一个token不一定是完整的字符：
#       token：  T0          T1              T2              T3
#       字节：   E4 BD A0    F0 9F           98 80           21
#       文本：   "你"        （😀的前2个字节） （😀的后2个字节） "!"        ← 😀（U+1F600）的UTF-8是F0 9F 98 80
#   单独decode([T1])只能得到U+FFFD（�），所以"每来一个token就把它单独decode出来发给客户端"行不通，要用下面的窗口法
# 原理：为什么流式必须"增量反分词"：客户端要的是文本增量（delta），而引擎每步给的是token id。
#   最朴素的做法是每步decode全部输出再和上次比，代价随输出长度线性增长（输出1万token时每步都要解码1万个）；
#   只解码新token又会出错（上面的T1）。这里的窗口法每步只解码最近几个token，开销与输出长度无关
# 为什么不能直接decode(新token)：除了上面"半个字符"的问题，有的tokenizer的解码结果还和前文有关（如前导空格）
# 做法：维护窗口 tokens[prefix_offset:]：
#   prefix_text = decode(tokens[prefix_offset:read_offset])   （上次已输出部分对应的文本）
#   new_text    = decode(tokens[prefix_offset:])              （加上新token之后的文本）
#   delta = new_text[len(prefix_text):]；若new_text以�结尾（字符还不完整）就先不输出，等下一个token
# 例：某个字被切成两个字节级token a、b
#   update([a])：prefix_text=""，new_text="�" → 不完整，返回""，窗口不动
#   update([b])：prefix_text=""，new_text="字" → 返回"字"；窗口前移：prefix_offset=0（旧read_offset），read_offset=2
# 完整的状态演示：沿用上面【科普】里的T0~T3，输出"你😀!"；每次update一个token（复用关闭时的两次decode）
#   调用            tokens          prefix_text=decode(              new_text=decode(          返回     之后的(prefix, read)
#                                   tokens[prefix:read])             tokens[prefix:])
#   ─────────────   ─────────────   ──────────────────────────────   ───────────────────────   ──────   ───────────────────
#   初始            []                                                                                   (0, 0)
#   update([T0])    [T0]            decode([])        = ""           decode([T0])     = "你"   "你"     (0, 1)
#   update([T1])    [T0,T1]         decode([T0])      = "你"         decode([T0,T1])  = "你�"  ""       (0, 1) 以�结尾，扣住
#   update([T2])    [T0,T1,T2]      decode([T0])      = "你"         decode([T0..T2]) = "你😀" "😀"     (1, 3)
#   update([T3],    [T0..T3]        decode([T1,T2])   = "😀"         decode([T1..T3]) = "😀!"  "!"      (3, 4)
#     finished=True)
#   要点：
#   ① 窗口[prefix_offset:]最多只覆盖"上一段已输出 + 本次新增"，所以每步的decode开销与总输出长度无关
#   ② delta按Python字符串长度切（码点数）：len("😀")=1，与它占几个字节、几个token无关
#   ③ 第2行若消费者落后，T1、T2会在collector里先合并（merge_chunk），一次update([T1, T2])直接得到"😀"，中间那次扣住根本不会发生
#   ④ 增量复用开启时，prefix_text直接取last_delta（update([T2])、update([T3])两行分别是"你"、"😀"，正好等于真实decode的结果），
#     省掉第一次decode
@dataclass
class IncrementalStreamDetokenizer:
    """Decode token deltas without emitting incomplete UTF-8 characters."""

    # 【译】解码token增量，不输出不完整的UTF-8字符。

    tokenizer: Any
    # Grows for the whole life of a stream, one entry per token, and only ever
    # sliced into `tokenizer.decode` -- which takes an array. Nothing here is
    # serialized, so this one has no boundary to convert back at.
    # 【译】在流的整个生命周期里不断增长，每个token一项，并且只会被切片后传给 tokenizer.decode——它接受array。
    #   这里没有任何东西会被序列化，所以不存在需要转换回list的边界。
    tokens: array.array = field(default_factory=new_token_ids)
    # prefix_offset / read_offset：窗口起点 / 已输出部分的终点（都是tokens的下标）
    prefix_offset: int = 0
    read_offset: int = 0
    # Emitted once per token in place of the decoded text, for runs whose text
    # is a byproduct rather than an answer. See `SYNTHETIC_TOKEN_TEXT`.
    # 【译】对文本只是副产品而非答案的运行，每个token输出一次它来代替解码文本。见 SYNTHETIC_TOKEN_TEXT。
    synthetic_text: str | None = None
    # The last delta emitted, which is what the next call's first decode would
    # produce: emitting advances the window to exactly the span that produced
    # it. Maintained whether or not reuse is on, so the two paths never diverge
    # in state -- only in whether they pay for the decode.
    # 【译】上一次输出的增量，也就是下一次调用的第一次decode会得到的结果：输出时窗口恰好前移到产生它的那段span。
    #   无论复用是否开启都会维护它，所以两条路径的状态从不分叉——区别只在于是否为那次decode付出代价。
    last_delta: str = ""

    def update(self, token_ids: list[int], finished: bool) -> str:
        """The text this stream has not been given yet, from its next tokens.

        One instance per stream, and calls on it must be in token order, one at
        a time. Handing the instance across threads is fine -- `flush` does,
        and `call_soon_threadsafe` supplies the happens-before edge -- but two
        overlapping calls are not.

        That was always true, since `tokens` and the two offsets are mutated
        here. What changed is the failure: `last_delta` now carries a value
        from the previous call, so an interleaved one does not merely reorder
        text, it leaves that value naming a span it no longer describes, and
        every later delta is cut at the wrong length. The audit finds it
        eventually; nothing raises.

        Empty `token_ids` is legal and yields "" without disturbing the stream.
        After `finished=True` there is nothing left to ask for.
        """
        # 【译】根据这个流接下来的token，返回还没交给它的文本。
        #
        #   每个流一个实例，对它的调用必须按token顺序、一次一个。把实例交给别的线程没问题——flush 就这么做，
        #   call_soon_threadsafe 提供了happens-before关系——但两个重叠的调用不行。
        #
        #   这一点一直成立，因为 tokens 和两个偏移量都在这里被修改。变化的是失败的样子：last_delta 现在带着上一次调用的值，
        #   所以交错的调用不只是打乱文本顺序，还会让这个值指向一段它已不再描述的span，之后的每个增量都按错误的长度切。
        #   审计最终会发现它；不会抛任何异常。
        #
        #   token_ids 为空是合法的，返回""且不扰动流。finished=True 之后就没有什么可再要的了。
        decoded = self._decode(token_ids, finished)
        if self.synthetic_text is None:
            return decoded
        # Decoded and thrown away: the run is measuring throughput, and skipping
        # the work would make the server look faster than the one being measured.
        # 【译】解码后扔掉：这次运行在测吞吐，跳过这项工作会让服务器看起来比被测的那个更快。
        return self.synthetic_text * len(token_ids)

    def _decode(self, token_ids: list[int], finished: bool) -> str:
        """Both decodes share a window start, so subtracting one from the other
        is what isolates the new text. A character can span several tokens --
        an emoji is two, each decoding alone to U+FFFD -- so the new tokens
        cannot be decoded by themselves, and the window start cannot be the
        stream's or every token would re-decode the whole output.

        The first decode only ever yields a length, and its span is the one the
        previous call already emitted, so a stream that has emitted before
        knows the answer. `_DELTA_REUSE` is whether that shortcut is trusted
        for this tokenizer: measured 1.74x at one token per update and 1.40x
        at sixty-four, on DeepSeek-V4-Pro.
        """
        # 【译】两次decode共用同一个窗口起点，所以用一个减去另一个就能分离出新文本。一个字符可能跨多个token——一个emoji是两个，
        #   各自单独解码都得到U+FFFD——所以新token不能单独解码；而窗口起点也不能是流的开头，否则每个token都要重新解码整个输出。
        #
        #   第一次decode只用来得到一个长度，而它的span正是上一次调用已经输出过的那段，所以输出过的流已经知道答案。
        #   _DELTA_REUSE 表示这个tokenizer是否可信任这条捷径：在DeepSeek-V4-Pro上实测，每次更新1个token时快1.74倍，64个时快1.40倍。
        self.tokens.extend(token_ids)
        # 第一次decode：复用开启且本次不审计 → 直接用last_delta；否则真解码，复用开启时顺便核对last_delta是否一致
        reuse = _DELTA_REUSE
        if reuse.enabled and not reuse.should_audit():
            prefix_text = self.last_delta
        else:
            prefix_text = self.tokenizer.decode(
                self.tokens[self.prefix_offset : self.read_offset],
                skip_special_tokens=True,
            )
            if reuse.enabled:
                if prefix_text:
                    reuse.audits += 1
                if prefix_text != self.last_delta:
                    reuse.report_mismatch(self.last_delta, prefix_text)
        # 第二次decode：窗口起点到末尾（包含新token）
        new_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset :],
            skip_special_tokens=True,
        )

        # 文本变长且最后一个字符完整 → 输出增量，窗口前移
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            self.last_delta = delta
            return delta
        # Withheld or final: the window did not move, so the last delta is
        # still the text of the span it names.
        # 【译】扣住或结束：窗口没有移动，所以上一个增量仍然是它所指那段span的文本。
        # 结束时即使末尾不完整也全部输出；未结束则本次返回""
        if finished:
            return new_text[len(prefix_text) :]
        return ""


# Boundaries, not coverage: characters that span several tokens, runs of
# whitespace, and a leading space -- the shapes that make a span's text depend
# on where its window started. Replayed at several merge depths because the
# premise is about where the window lands.
# 【译】要的是边界情况，而不是覆盖面：跨多个token的字符、连续空白、前导空格——这些形态会让一段span的文本取决于窗口从哪里开始。
#   在多种合并深度下重放，因为这个前提关乎窗口落在哪里。
# 以下到enable_delta_reuse是启动期的探测：用这些文本和随机token流验证增量复用对当前tokenizer是否安全
_PROBE_TEXTS = (
    "The quick brown fox jumps over the lazy dog. " * 3,
    "深度学习模型的推理性能取决于内存带宽和计算密度。" * 2,
    "party 🎉 family 👨‍👩‍👧‍👦 done. " * 2,
    "𝓗𝓮𝓵𝓵𝓸 ∑x²  ≈ ∫f(t)dt " * 2,
    " leading  spaces   and\t\ttabs\n\n ",
    "def merge(a, b):\n    return {**a, **b}\n",
)
_PROBE_CHUNKS = (1, 2, 3, 5, 8, 17, 64)
_PROBE_RANDOM_TOKENS = 128


def _probe_streams(tokenizer) -> list[list[int]]:
    """Token streams to replay, from both sides of the model.

    Encoded text covers what a tokenizer produces; sampled vocabulary covers
    what a *model* produces, which is any id at all -- including the lone byte
    tokens that never appear when encoding prose and are exactly where decoding
    a span depends on its neighbours. Seeded, so a failure is reproducible.
    """
    # 【译】要重放的token流，来自模型的两侧。
    #
    #   编码出的文本覆盖tokenizer会产生的东西；随机采样的词表覆盖"模型"会产生的东西，也就是任何id——
    #   包括编码普通文本时从不出现的孤立字节token，而它们正是解码一段span会依赖邻居的地方。用了固定种子，所以失败可复现。
    streams = [tokenizer.encode(t, add_special_tokens=False) for t in _PROBE_TEXTS]
    size = getattr(tokenizer, "vocab_size", 0) or 0
    if size:
        rng = random.Random(0)
        streams += [
            [rng.randrange(size) for _ in range(_PROBE_RANDOM_TOKENS)] for _ in range(2)
        ]
    return [s for s in streams if s]


def _replay(tokenizer, ids: list[int], chunk: int) -> str:
    """One stream through the real detokenizer, `chunk` tokens per update."""
    # 【译】让一个流通过真实的反分词器，每次更新 chunk 个token。
    state = IncrementalStreamDetokenizer(tokenizer)
    return "".join(
        state.update(ids[i : i + chunk], i + chunk >= len(ids))
        for i in range(0, len(ids), chunk)
    )


def _a_wrong_delta_is_caught(tokenizer, ids: list[int]) -> bool:
    """Negative control. Without it, "no mismatches" is also what a comparison
    that never ran would report, and the probe would pass every tokenizer."""
    # 【译】阴性对照。没有它的话，一个从未运行过的比较也会报告"没有不一致"，探测就会放行所有tokenizer。
    state = IncrementalStreamDetokenizer(tokenizer)
    for i, tid in enumerate(ids):
        state.update([tid], False)
        if state.last_delta and i + 1 < len(ids):
            state.last_delta += "\0"
            before = _DELTA_REUSE.mismatches
            state.update([ids[i + 1]], False)
            return _DELTA_REUSE.mismatches > before
    return False


def _probe(tokenizer) -> tuple[str | None, int]:
    """Why this tokenizer cannot be trusted with reuse, or None, and how many
    comparisons said so. Runs with reuse on and the audit at every call, so the
    check is the one `_decode` already carries rather than a second copy of it.
    """
    # 【译】返回这个tokenizer为什么不能信任复用（或None），以及有多少次比较支持这个结论。
    #   运行时开启复用、并且每次调用都审计，所以用的是 _decode 自带的检查，而不是它的第二份拷贝。
    streams = _probe_streams(tokenizer)
    for ids in streams:
        want = tokenizer.decode(ids, skip_special_tokens=True)
        for chunk in _PROBE_CHUNKS:
            reused = _replay(tokenizer, ids, chunk)
            _DELTA_REUSE.enabled = False
            plain = _replay(tokenizer, ids, chunk)
            _DELTA_REUSE.enabled = True
            # Two failures worth telling apart: the first is the one this
            # shortcut causes, the second says the incremental scheme itself
            # does not suit this tokenizer and reuse would only amplify it.
            # 【译】有两种值得区分的失败：第一种是这条捷径造成的；第二种说明增量方案本身就不适合这个tokenizer，复用只会放大问题。
            if reused != plain:
                return "reusing the delta changes the text this tokenizer emits", 0
            if reused != want:
                return (
                    (
                        "this tokenizer disagrees with a whole decode even "
                        "without reuse"
                    ),
                    0,
                )
    if _DELTA_REUSE.mismatches:
        return "the tokenizer disagreed with a reused delta", 0
    if not _DELTA_REUSE.audits:
        return "nothing was ever compared", 0
    if not _a_wrong_delta_is_caught(tokenizer, streams[0]):
        return "a deliberately wrong delta went unnoticed", 0
    return None, _DELTA_REUSE.audits


def _audit_interval(value: int | str) -> int:
    """How often to check a reused delta, from `ATOM_DETOKENIZER_AUDIT_EVERY`.

    Empty means the default. Anything else unusable warns and takes the default
    as well -- including 0, which reads like "never audit" but would divide by
    zero, and is not how reuse is turned off.
    """
    # 【译】多久检查一次复用的增量，取自 ATOM_DETOKENIZER_AUDIT_EVERY。
    #
    #   空值表示默认值。其他无法使用的值会给出警告并同样取默认值——包括0，它读起来像"从不审计"，但会导致除以零，
    #   而且关闭复用也不是这么关的。
    if value is None or value == "":
        return DEFAULT_AUDIT_EVERY
    try:
        interval = int(value)
    except (TypeError, ValueError):
        interval = 0
    if interval < 1:
        logger.warning(
            "[detokenizer] unusable ATOM_DETOKENIZER_AUDIT_EVERY=%r, using %d; "
            "reuse is turned off with ATOM_DETOKENIZER_DELTA_REUSE=off",
            value,
            DEFAULT_AUDIT_EVERY,
        )
        return DEFAULT_AUDIT_EVERY
    return interval


# 启动时调用一次（api_server.main）：mode为ATOM_DETOKENIZER_DELTA_REUSE（auto / on / off）
#   on / off：直接强制；auto：开启复用、每次都审计地跑一遍_probe，全部通过才保持开启
def enable_delta_reuse(
    tokenizer, mode: str = "auto", audit_every: int | str = ""
) -> bool:
    """Let `_decode` skip its first decode, if this tokenizer allows it.

    The shortcut holds where decoding a token span does not depend on where the
    window started -- true for the byte-level BPE tokenizers measured (DeepSeek,
    Qwen3.5, GLM-5.2), and not something to assume from a class name: a
    SentencePiece-style decoder adds or strips a leading space by position.

    So it is verified rather than declared, and any exception is a failure --
    an unverified shortcut is worth less than the decode it saves.

    Unrelated to the KV prefix cache, which is what "cache" means everywhere
    else in this repository.

    Neither setting may end the process or turn reuse on by accident: an
    unreadable mode leaves reuse off, which is what someone spelling this
    setting is reaching for, and an unreadable audit interval falls back to the
    default, since it is read at a callsite that has already loaded the weights.
    """
    # 【译】如果这个tokenizer允许，就让 _decode 跳过它的第一次decode。
    #
    #   这条捷径成立的前提是：解码一段token span的结果与窗口从哪里开始无关——对实测过的字节级BPE tokenizer
    #   （DeepSeek、Qwen3.5、GLM-5.2）成立，但不能凭类名假定：SentencePiece风格的解码器会按位置添加或去掉前导空格。
    #
    #   所以它是经过验证的，而不是声明出来的，并且任何异常都算失败——一条未经验证的捷径，价值还不如它省下的那次decode。
    #
    #   与KV前缀缓存无关，后者才是本仓库其他地方说"cache"时的意思。
    #
    #   两个设置都不能让进程退出，也不能意外打开复用：无法识别的模式让复用保持关闭，这正是写这个设置的人想要的；
    #   无法识别的审计间隔退回默认值，因为读取它的调用点已经加载完权重了。
    setting = str(mode).strip().lower()
    if setting not in ("auto", "on", "off"):
        logger.warning(
            "[detokenizer] unknown delta reuse mode %r, leaving reuse off", mode
        )
        setting = "off"
    _DELTA_REUSE.audit_every = _audit_interval(audit_every)
    if setting != "auto":
        _DELTA_REUSE.enabled = setting == "on"
        logger.info("[detokenizer] delta reuse forced %s", setting)
        return _DELTA_REUSE.enabled

    before = replace(_DELTA_REUSE)
    _DELTA_REUSE.enabled, _DELTA_REUSE.audit_every, _DELTA_REUSE.probing = True, 1, True
    _DELTA_REUSE.audits = _DELTA_REUSE.mismatches = 0
    try:
        why, armed = _probe(tokenizer)
    except Exception:
        logger.exception("[detokenizer] delta reuse probe raised; leaving it off")
        why, armed = "the probe raised", 0
    finally:
        _DELTA_REUSE.audit_every = before.audit_every
        _DELTA_REUSE.calls = before.calls
        _DELTA_REUSE.probing = False
        _DELTA_REUSE.audits = _DELTA_REUSE.mismatches = 0

    _DELTA_REUSE.enabled = why is None
    if why:
        logger.warning("[detokenizer] delta reuse off: %s", why)
    else:
        logger.info(
            "[detokenizer] delta reuse on: %d comparisons agreed across %d merge "
            "depths, and a spoiled delta was caught",
            armed,
            len(_PROBE_CHUNKS),
        )
    return _DELTA_REUSE.enabled


# 合并两个未读chunk（消费者落后时）。例：
#   into = {"token_ids": [5], "finished": False}，new = {"token_ids": [7], "finished": True, "finish_reason": "eos"}
#   → 合并后 into = {"token_ids": [5, 7], "text": "", "finished": True, "finish_reason": "eos"}
def merge_chunk(into: dict, new: dict) -> None:
    """Fold ``new`` into the chunk already waiting. ``into`` is modified.

    Both deltas extend in place: ``into`` holds the copy ``put_nowait`` took,
    and rebuilding them walked the whole accumulation on every merge.
    """
    # 【译】把 new 折叠进已经在等待的chunk。into 会被修改。
    #
    #   两种增量都原地扩展：into 持有的是 put_nowait 拿到的那份拷贝，而重建它们会在每次合并时遍历全部累积内容。
    into["token_ids"].extend(new.get("token_ids") or ())
    # Popped so the string has one reference and CPython grows it in place;
    # assigning `into["text"] + ...` back reallocates and copies per merge.
    # 【译】先pop出来，让这个字符串只有一个引用，CPython就能原地扩展它；写成 into["text"] + ... 再赋回去，每次合并都会重新分配并拷贝。
    text = into.pop("text", "")
    try:
        text += new.get("text", "")
    finally:
        into["text"] = text
    into["finished"] = bool(into.get("finished") or new.get("finished"))
    for key in _LATEST_WINS:
        if new.get(key):
            into[key] = new[key]


# 每个请求一个的"收件箱"（文件头所说的collector），只在事件循环线程上被访问
#   put_nowait（_deliver调用）：放入chunk；同一个tag已有未读chunk时就合并进去
#   get（SSE生成器await）：取出一个chunk，并在这里完成反分词，补上"text"
# 消费者落后的例子：第1步的chunk还没被读走，第2步的chunk到了 → 合并成一个；get()只解码一次，得到两步的全部新增文本
# 原理：这里的"合并"是一种天然的背压（backpressure）：每帧的固定开销（反分词、JSON编码、一次socket写）
#   按消费者的节奏发生，而不是按引擎产出token的节奏。例：引擎每10ms一个step，事件循环忙了50ms才轮到这个流 →
#   这期间5个step的chunk已合并成一个，get()只解码一次、只发一帧（含5个step的全部文本）。
#   代价是该流这一帧的ITL变成50ms；好处是事件循环越忙，每token的平均开销越低，不会因为积压越来越多帧而雪崩
# 时间线（引擎每10ms一个step；"put"是_deliver里的put_nowait，"get"是SSE生成器的await collector.get()）：
#   引擎step      输出线程                     事件循环上的collector._pending          SSE生成器
#   ──────────    ──────────────────────────   ─────────────────────────────────────   ──────────────────────────────
#   t=10 step1    enqueue → flush → _deliver   put {ids:[a]}           → {None: [a]}   get取走[a] → decode → 1帧
#   t=20 step2    同上                         put {ids:[b]}           → {None: [b]}   （事件循环在忙别的流，没轮到）
#   t=30 step3    同上                         put {ids:[c]} 合并      → {None: [b,c]} （仍在忙）
#   t=40 step4    同上                         put {ids:[d]} 合并      → {None: [b,c,d]}
#   t=50/60 5,6   同上                         合并e、f                → {None: [b..f]}
#   t=62                                                                               get取走[b..f] → 一次update → 1帧
#   对比不合并的做法：t=62时队列里有5个chunk，要做5次反分词、5次JSON编码、5次socket写；
#     而这5次工作本身又会推迟别的流，越积越多。合并后无论落后多少步，一个流每次被调度只付一次固定开销
#   finished也会被合并进去（merge_chunk：finished取或，finish_reason等"后来者覆盖"），所以结束不会丢
# asyncio.Event（第一次出现）：协程间的信号量，set()唤醒所有wait()的协程，clear()复位
class StreamOutputCollector:
    """Per-request delivery point that merges chunks when the consumer lags.

    A stream holds at most one pending chunk per tag. Raw token deltas from the
    dispatcher merge here before detokenization, coroutine wakeup, JSON encoding
    and socket writing. Detokenizer state is advanced only by ``get()``, on the
    event loop; the engine output threads never decode collector-bound chunks.

    There is no timer or minimum batch size: get() can consume any pending
    delta immediately. Merging covers only unread chunks, but decoding runs
    synchronously on the event loop, so this is not a delivery-latency bound.
    Large backlogs can delay other consumers while their text is decoded.

    ``tag`` is the fan-out sibling index (``SamplingParams.n>1``) or ``None`` for
    a plain single-sequence stream. Chunks merge per tag, so siblings never mix.
    """

    # 【译】每个请求的投递点，消费者落后时合并chunk。
    #
    #   一个流对每个tag最多持有一个待处理chunk。来自投递器的原始token增量在这里合并，然后才进行反分词、协程唤醒、
    #   JSON编码和socket写入。反分词器的状态只由事件循环上的 get() 推进；引擎输出线程从不解码发往collector的chunk。
    #
    #   没有定时器，也没有最小批量：get() 可以立即消费任何待处理的增量。合并只覆盖未读的chunk，但解码是在事件循环上同步执行的，
    #   所以这不是投递延迟的上界。大量积压在解码时可能拖慢其他消费者。
    #
    #   tag 是扇出兄弟序号（SamplingParams.n>1），普通单序列流为 None。chunk按tag分别合并，所以兄弟之间从不混淆。

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id
        # _pending：tag → 未读chunk（dict保持插入顺序，get()总取最早的那个tag）；_ready：有未读chunk时为set状态
        self._pending: dict[Any, dict] = {}
        self._ready = Event()

    def put_nowait(self, payload: dict | tuple[int, dict]) -> None:
        """Accept one prepared chunk. Called on the event loop, never off it."""
        # 【译】接收一个准备好的chunk。在事件循环上调用，绝不在别处调用。
        # payload是chunk（单序列流）或(tag, chunk)（扇出）
        if type(payload) is tuple:
            tag, chunk = payload
        else:
            tag, chunk = None, payload
        waiting = self._pending.get(tag)
        if waiting is None:
            # A merge extends this list, so it has to be ours. The scheduler
            # already copies per step, but that is too far away to rely on.
            # 【译】合并会扩展这个列表，所以它必须归我们所有。调度器已经每步拷贝过一次，但那离这里太远，不能依赖。
            chunk["token_ids"] = list(chunk.get("token_ids") or ())
            self._pending[tag] = chunk
        else:
            merge_chunk(waiting, chunk)
        self._ready.set()

    async def get(self) -> dict | tuple[int, dict]:
        """Await the next chunk, carrying whatever merged into it."""
        # 【译】等待下一个chunk，它携带着所有合并进来的内容。
        # 没有未读chunk就挂起在Event上；取出最早的一个后若已空就clear，下次get会再等待
        while not self._pending:
            await self._ready.wait()
        tag, chunk = next(iter(self._pending.items()))
        del self._pending[tag]
        if not self._pending:
            self._ready.clear()
        # 投递器挂在chunk上的反分词器：在这里（事件循环上）把合并后的全部token一次解码成新增文本
        state = chunk.pop("_detokenizer", None)
        if state is not None:
            try:
                chunk["text"] = state.update(
                    chunk["token_ids"], bool(chunk.get("finished"))
                )
            except Exception:
                logger.exception(
                    "Error detokenizing stream %s (tag=%s)", self.request_id, tag
                )
                # Keep token accounting and terminal metadata, and let fan-out
                # siblings continue. Failed tokens remain in the detokenizer;
                # a later update can decode them without appending them twice.
                # 【译】保留token计数和终止元数据，并让扇出的兄弟继续。失败的token留在反分词器里；之后的更新可以解码它们，而不会重复追加。
                chunk["text"] = ""
        return chunk if tag is None else (tag, chunk)


# 静默看门狗：用with包住"等下一帧"的那次await（api_server._client_stream），进入时登记、退出时注销；
#   等待超过SILENCE_LOG_SECONDS就在事后打一行warning
class FrameWait:
    """Times one gap between frames the client actually receives.

    Measured here and not at :meth:`StreamOutputCollector.get`, which is where
    it started. That is the one place a stream waits for the *engine*, but two
    stages sit between it and the socket -- the reasoning channel's read-ahead
    and the tool-call format's -- and while either withholds, `get` keeps
    returning on schedule. The gauge read zero while the client received
    nothing, which is the exact symptom it was built for.

    The first frame still has to be excluded, and the docstring here once
    claimed otherwise. Every response generator awaits the collector before
    yielding anything, so the wait for frame one is admission, queueing and
    prefill -- measured, a request 200 ms into a queue with no token yet
    produced put 0.2 s on the gauge, which is `atom:requests_waiting` wearing
    a different name, and at a deep queue would log a line per admitted
    request blaming the read-ahead. `armed` is off for that one wait.

    A timestamp and a dict entry rather than `asyncio.wait_for`: this runs
    once per frame per stream, and arming a timer costs 1.38 us against
    0.07 us for this. No timer also means no background task to own.
    """

    # 【译】给客户端实际收到的两帧之间的一段间隔计时。
    #
    #   在这里测量，而不是在它最初所在的 StreamOutputCollector.get。那是流等待"引擎"的唯一地方，但在它和socket之间还隔着两个阶段——
    #   推理通道的预读和工具调用格式的预读——只要其中任何一个扣着不发，get 仍会按时返回。于是客户端什么都没收到时仪表读数为零，
    #   而这正是它被造出来要发现的症状。
    #
    #   第一帧仍然必须排除在外，这里的docstring曾经声称相反。每个响应生成器在yield任何东西之前都要先await collector，
    #   所以等第一帧的时间是准入、排队和prefill——实测，一个在队列里待了200毫秒、还没产生任何token的请求会给仪表贡献0.2秒，
    #   这其实是换了个名字的 atom:requests_waiting；在深队列下，还会为每个准入的请求打一行日志去怪罪预读。
    #   armed 在那一次等待时是关闭的。
    #
    #   用时间戳和一个dict条目，而不是 asyncio.wait_for：这里每个流每帧执行一次，设置一个定时器要1.38微秒，而这样只要0.07微秒。
    #   没有定时器也就不需要管理后台任务。

    # __slots__：实例只有这三个属性、没有__dict__，创建更快更省内存（每帧都要创建一个）
    __slots__ = ("_started", "armed", "request_id")

    def __init__(self, request_id: str = "", *, armed: bool = True) -> None:
        self.request_id = request_id
        self.armed = armed
        self._started = 0.0

    def __enter__(self) -> None:
        # No `as`: nothing needs the watch itself, only its lifetime.
        # 【译】不用 as：没有谁需要这个计时器对象本身，只需要它的生命周期。
        self._started = time.monotonic()
        if self.armed:
            _WAITING_SINCE[id(self)] = self._started

    def __exit__(self, *exc) -> None:
        _WAITING_SINCE.pop(id(self), None)
        silence = time.monotonic() - self._started
        if self.armed and silence >= SILENCE_LOG_SECONDS:
            # After the fact, and free: one comparison on a frame that was
            # going to arrive anyway. Catches a stall that recovered, which
            # the gauge cannot -- by scrape time it is over.
            # 【译】事后检查，而且没有开销：只是在一个反正要到达的帧上做一次比较。它能抓到已经恢复的卡顿，这是仪表做不到的——
            #   到抓取指标时卡顿已经结束了。
            logger.warning(
                f"request {self.request_id or '<unnamed>'} sent the client "
                f"nothing for {silence:.1f}s before recovering; if this "
                f"repeats, the engine or one of the marker read-aheads is "
                f"holding output back"
            )


# /metrics里的"流静默时长"仪表（metrics_setup.py把它交给导出器）
def longest_silence_seconds() -> float:
    """How long the most starved in-flight stream has been waiting.

    Zero when nothing is waiting. Exported as a gauge so a stalled response
    shows up while it is stalled, rather than as a support ticket.
    """
    # 【译】在途流中最"饥饿"的那个已经等了多久。
    #
    #   没有流在等待时为0。作为仪表（gauge）导出，让卡住的响应在卡住期间就能被看到，而不是变成一张客服工单。
    if not _WAITING_SINCE:
        return 0.0
    now = time.monotonic()
    return now - min(_WAITING_SINCE.values())


# 注意测量点：这是"投递到collector"之间的间隔（事件循环上），不含之后的反分词、JSON编码和网络传输；
#   客户端实测的ITL还会再加上这些，以及前面说的合并效应
# token间延迟（ITL）计时：_deliver每投递一个带新token的chunk就记录一次与上次投递的间隔
@dataclass
class StreamDeliveryTiming:
    """Track frontend delivery intervals independently of token decoding."""

    # 【译】独立于token解码，跟踪前端投递的时间间隔。

    last_output_at: float | None = None

    def record(
        self, num_new_tokens: int, observe: Callable[[float, int], None]
    ) -> None:
        now = time.perf_counter()
        if self.last_output_at is None:
            self.last_output_at = now
        else:
            observe(now - self.last_output_at, num_new_tokens)
            # Keep instrumentation work out of the next interval.
            # 【译】不把计量工作本身算进下一个间隔。
            self.last_output_at = time.perf_counter()


# 一个流的全部状态：反分词器 + 投递计时。由new_state()创建，被流式回调的闭包持有
@dataclass
class StreamState:
    detokenizer: IncrementalStreamDetokenizer
    timing: StreamDeliveryTiming = field(default_factory=StreamDeliveryTiming)


# NamedTuple：带字段名的不可变元组；记录"一个chunk要投给哪个事件循环上的哪个collector"
class _BufferedChunk(NamedTuple):
    """One stream's chunk, waiting for the end of the current engine step."""

    # 【译】一个流的chunk，等待当前引擎step结束。

    loop: AbstractEventLoop
    collector: Any
    state: StreamState
    chunk: dict
    tag: int | None


# 全局唯一的批量投递器（api_server.main创建）。enqueue / flush在引擎输出线程上调用，_deliver在事件循环上执行
class StreamBatchDispatcher:
    """Collect one engine step per output thread and dispatch it by event loop.

    The dispatcher has no persistent per-stream registry. Each engine callback
    creates one composed stream state and must reuse it for every chunk of its
    (collector, tag); every collector-bound chunk carries that same state.
    Merging keeps the first pending chunk's state, so it must not be replaced
    partway through a stream.

    For StreamOutputCollector, references cross threads but only get() on the
    event loop mutates the detokenizer. Frontend delivery updates the separate
    timing state on that same loop, before merging. Output threads pass the state
    through. An unread chunk or queued delivery keeps the state (including its
    token history) alive after the engine drops the finished callback; consuming
    or discarding those references allows it to be reclaimed. Queue consumers use
    eager decoding on their output thread instead.
    """

    # 【译】每个输出线程收集一个引擎step，再按事件循环分发出去。
    #
    #   投递器没有持久的逐流注册表。每个引擎回调创建一个组合好的流状态，并且必须对它的 (collector, tag) 的每个chunk复用它；
    #   每个发往collector的chunk都携带同一个状态。合并时保留第一个待处理chunk的状态，所以不能在流的中途替换它。
    #
    #   对 StreamOutputCollector 而言，引用会跨线程，但只有事件循环上的 get() 会修改反分词器。前端投递在同一个事件循环上、
    #   在合并之前更新独立的计时状态。输出线程只是把状态传过去。引擎丢掉已结束的回调之后，未读的chunk或排队中的投递
    #   仍会让这个状态（包括它的token历史）保持存活；这些引用被消费或丢弃后，它就可以被回收。队列类消费者则在它们的输出线程上直接解码。

    def __init__(
        self,
        tokenizer: Any,
        synthetic_text: str | None = None,
        observe_inter_token_latency: Callable[[float, int], None] | None = None,
    ):
        self.tokenizer = tokenizer
        self.synthetic_text = synthetic_text
        self._observe_inter_token_latency = observe_inter_token_latency
        # threading.local()：每个线程看到各自独立的属性。DP>1时有多个输出线程，各自的缓冲互不干扰，也不需要加锁
        self._thread_local = threading.local()

    def new_state(self) -> StreamState:
        """Make decoding and delivery timing state for one stream's callback."""
        # 【译】为一个流的回调创建解码状态和投递计时状态。
        return StreamState(
            IncrementalStreamDetokenizer(
                self.tokenizer, synthetic_text=self.synthetic_text
            )
        )

    def enqueue(
        self,
        *,
        loop: AbstractEventLoop,
        collector: Any,
        state: StreamState,
        chunk: dict,
        tag: int | None = None,
    ) -> None:
        """Buffer a raw chunk until the current engine step is flushed."""
        # 【译】缓冲一个原始chunk，直到当前引擎step被冲刷。
        # 只是往本线程的列表里追加，不碰任何asyncio对象
        buf = getattr(self._thread_local, "buf", None)
        if buf is None:
            buf = self._thread_local.buf = []
        buf.append(_BufferedChunk(loop, collector, state, chunk, tag))

    def flush(self) -> None:
        """Schedule raw chunks, letting each consumer coalesce before decoding."""
        # 【译】调度原始chunk，让每个消费者在解码之前先合并。
        # 取走本线程的缓冲并换成新列表，然后按目标事件循环分组
        tl = self._thread_local
        buf = getattr(tl, "buf", None)
        if not buf:
            return
        tl.buf = []

        by_loop: dict[AbstractEventLoop, list[_BufferedChunk]] = {}
        for item in buf:
            if isinstance(item.collector, StreamOutputCollector):
                # Keep this state with the pending chunk until get(). Moving
                # only the JSON/socket work downstream still made four output
                # threads decode every token while contending for the GIL.
                # 【译】让这个状态跟着待处理chunk，直到 get()。只把JSON/socket工作挪到下游时，四个输出线程仍要在争抢GIL的同时解码每个token。
                item.chunk["_detokenizer"] = item.state.detokenizer
            else:
                # Queue consumers cannot decode on read and still receive a
                # prepared chunk, as before.
                # 【译】队列类消费者无法在读取时解码，所以仍像以前一样收到准备好的chunk。
                # （本仓库中这类消费者只出现在测试里）
                item.chunk["text"] = item.state.detokenizer.update(
                    item.chunk.get("token_ids") or [],
                    bool(item.chunk.get("finished")),
                )
            by_loop.setdefault(item.loop, []).append(item)

        # 每个事件循环一次call_soon_threadsafe：把这一step发往该循环的全部chunk一起交给_deliver
        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._deliver, items)

    def _deliver(self, items: list[_BufferedChunk]) -> None:
        """Run on the target event loop and hand a whole step to its collectors.

        A step is delivered in one callback, never split across loop iterations.
        Splitting was tried as a fairness measure -- deliver 128, re-arm the rest
        with call_soon -- and it silently corrupts streams: the output thread can
        schedule the next step's delivery in between, so a collector receives
        step N+1's chunk before step N's leftovers. Deltas then merge in the
        wrong order, and an end-of-stream that lands before a straggler is
        overwritten by it, hanging that client for good.
        """
        # 【译】在目标事件循环上运行，把一整个step交给它的各个collector。
        #
        #   一个step在一次回调里投递完，从不拆到多轮事件循环迭代中。曾经为了公平尝试过拆分——先投递128个，剩下的用 call_soon 重新排队——
        #   结果会悄悄破坏流：输出线程可能在两者之间排入下一个step的投递，于是某个collector先收到第N+1步的chunk、后收到第N步剩下的。
        #   增量就会按错误的顺序合并，而一个先于掉队chunk到达的流结束标记会被它覆盖，让那个客户端永远挂起。
        # 先记录ITL（带新token的chunk才记），再put_nowait进collector
        for item in items:
            num_new_tokens = len(item.chunk.get("token_ids") or ())
            if num_new_tokens and self._observe_inter_token_latency is not None:
                item.state.timing.record(
                    num_new_tokens, self._observe_inter_token_latency
                )
            payload = item.chunk if item.tag is None else (item.tag, item.chunk)
            item.collector.put_nowait(payload)
