# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning-channel dialects (model-specific data for the general engine).

The engine in ``reasoning.py`` is dialect-agnostic: it iterates ``DIALECTS`` to
detect/split the reasoning channel. All model-specific marker knowledge lives
here. Adding a model = add one ``ReasoningDialect`` entry (and its ``split`` for
whole-response separation).

Two dialects today, named by format rather than model:
  - inline ``<think>...</think>`` (DeepSeek-R1, Qwen3, Kimi-K2, MiniMax, ...).
    The opening tag may be emitted in the output or injected by the template.
  - structured channel format: one stream split into named channels (think /
    response / tools), each wrapped in framing tokens. The same concept as
    OpenAI Harmony's analysis/final/commentary channels (gpt-oss). The opening
    tag is template-injected, so the output begins *inside* the reasoning
    channel. Different channel-format models use different framing tokens; the
    entry below carries Kimi-K3's (``<|open|>think<|sep|>`` ...), and another
    such model would add its own entry with its own tokens.
"""

# 【译】推理通道的方言（供通用引擎使用的、与模型相关的数据）。
#
#   ``reasoning.py``中的引擎与方言无关：它遍历``DIALECTS``来检测/切分推理通道。所有与模型相关的标记知识都在这里。
#   增加一个模型 = 增加一项``ReasoningDialect``（以及它用于整段响应切分的``split``）。
#
#   目前有两种方言，按格式而不是按模型命名：
#     - 内联``<think>...</think>``（DeepSeek-R1、Qwen3、Kimi-K2、MiniMax……）。开标签可能由模型输出，也可能由模板注入。
#     - 结构化通道格式：一个流被分成几个命名通道（think / response / tools），每个通道包在外框token里。
#       与OpenAI Harmony的analysis/final/commentary通道（gpt-oss）是同一个概念。开标签由模板注入，
#       所以输出从推理通道*内部*开始。不同的通道格式模型使用不同的外框token；下面那项带的是Kimi-K3的
#       （``<|open|>think<|sep|>`` ...），另一个这类模型会加一项带自己token的条目。
# 解释：两处与现在的代码不完全一致——
#   1. 引擎已不再"遍历DIALECTS来切分"：启动时resolve_dialect只选出一个方言，之后每个响应只用它；
#      DIALECTS只在resolve_dialect、template_opens_reasoning_implicitly和收集prompt开标记时被遍历
#   2. "两种"指两种格式，DIALECTS实际有三项：MiniMax-M3的<mm:think>是内联形态的另一种拼写，单独占一项
#
# 【导读】推理通道的"方言表"：每种模型家族用什么标记包住推理内容（reasoning content）
#   推理内容是什么、为什么要拆出来：api_server.py 中 chat_completions 里的【科普】推理内容；引擎一侧的分工见 reasoning.py 文件头
# 调用方：reasoning.py（ReasoningChannel / ReasoningFilter / separate_reasoning读取方言的标记和split）；
#   resolve_dialect由api_server.py和atomesh/atom_standalone_service.py在启动时各调用一次
# 与主干的关系：纯数据 + 非流式切分规则，没有状态；流式那一半在reasoning.ReasoningFilter里，两边读同一份标记表
# 【分支】Kimi-K3、MiniMax-M3等新模型的输出格式，见第8批

import re
from collections.abc import Callable
from dataclasses import dataclass

from . import kimi_k3_tokens as k3

# Structured-channel format tokens (``<|open|>SECTION<|sep|>`` ... framing).
# Named by the format concept, not the model: channel formats are a cross-model
# pattern (e.g. gpt-oss/Harmony uses the same idea with different framing tokens).
# The token *values* below are Kimi-K3's — a different channel-format model would
# declare its own values. Declared locally so this module is self-contained
# (each parser owns the token strings it uses); the tool-call parser keeps its
# own copies of the subset it needs.
# 【译】结构化通道格式的token（``<|open|>SECTION<|sep|>`` ... 外框）。
#   按格式概念命名，而不是按模型：通道格式是跨模型的模式（例如gpt-oss/Harmony用同样的思路、不同的外框token）。
#   下面的token*取值*是Kimi-K3的——另一个通道格式模型会声明自己的取值。在本地声明，使本模块自成一体
#   （每个解析器拥有自己用到的token字符串）；工具调用解析器保留它所需子集的自有副本。
# 解释：后来新增了kimi_k3_tokens.py作为K3 token的统一出处，本文件的_CHANNEL_END_MARKERS和外框都改从那里读；
#   下面8个常量里，本文件只用到THINK_START / THINK_END / RESPONSE_START，其余几个在v0.1.8中已无人引用
CHANNEL_THINK_START = "<|open|>think<|sep|>"
CHANNEL_THINK_END = "<|close|>think<|sep|>"
CHANNEL_RESPONSE_START = "<|open|>response<|sep|>"
CHANNEL_RESPONSE_END = "<|close|>response<|sep|>"
CHANNEL_MESSAGE_END = "<|close|>message<|sep|>"
CHANNEL_END_OF_MSG = "<|end_of_msg|>"
CHANNEL_TOOLS_START = "<|open|>tools<|sep|>"
CHANNEL_CALL_PREFIX = '<|open|>call tool="'

# Result of splitting a full response: (reasoning_content or None, content).
#
# **Both halves come back byte-for-byte.** What may be removed is a marker a
# dialect declares; everything else, and whitespace in particular, survives.
# This is the rule `ToolCallParser.parse` already states for the stage after
# this one, and it exists for the same reason: the streaming filter releases
# bytes as they arrive and owns nothing to tidy them with, so any trimming
# done only here is a divergence a client sees.
#
# It was not applied here, and the symptom was the one that rule cites
# verbatim -- a trailing `.strip()` cost a code-block answer its final
# newline. Measured across 12544 (dialect, shape, chunking) comparisons,
# stripping put `stream=false` and `stream=true` at 50% byte-agreement on
# content; without it they agree exactly.
# 【译】切分完整响应的结果：(reasoning_content或None, content)。
#
#   **两半都按原字节返回。**可以去掉的只有方言声明过的标记；其余一切，尤其是空白字符，都要保留。
#   这是`ToolCallParser.parse`对下一阶段已经声明过的规则，理由也相同：流式过滤器在字节到达时就放出它们，
#   手里没有任何东西可用来整理，所以任何只在这里做的修剪，都是客户端能看到的不一致。
#
#   这里以前没有遵守它，症状正是那条规则原文引用的那个——末尾的`.strip()`让一个代码块回答丢了最后的换行。
#   在12544组（方言, 形态, 分块方式）对比中实测，做strip时`stream=false`与`stream=true`的content只有50%逐字节一致；
#   不做strip则完全一致。
SplitResult = tuple[str | None, str]


# 一种方言 = 一组标记 + 一个非流式split函数；frozen dataclass，作为常量放进DIALECTS
@dataclass(frozen=True)
class ReasoningDialect:
    """How one model family delimits its reasoning channel.

    - ``prompt_open_marker``: what a rendered prompt ends with when the template
      has already opened the reasoning channel (output then begins in reasoning).
    - ``output_open_marker``: the marker the model *emits* to open reasoning
      mid-stream (``<think>``); ``None`` when the template injects it instead.
    - ``think_end_marker``: the marker that ends the reasoning channel.
    - ``split``: whole-response separator returning ``SplitResult`` or ``None``
      if this dialect does not apply to the text.
    - ``template_efforts``: reasoning-effort levels this model's chat template
      accepts (e.g. K3's ``low``/``high``/``max``); empty when the model has no
      effort control.
    """

    # 【译】一个模型家族如何界定它的推理通道。
    #
    #   - ``prompt_open_marker``：模板已经打开推理通道时，渲染后的prompt结尾的标记（此时输出从推理内容开始）。
    #   - ``output_open_marker``：模型在流中途*输出*、用来打开推理的标记（``<think>``）；由模板注入时为``None``。
    #   - ``think_end_marker``：结束推理通道的标记。
    #   - ``split``：整段响应的切分函数，返回``SplitResult``；这个方言不适用于该文本时返回``None``。
    #   - ``template_efforts``：这个模型的chat template接受的推理强度档位（例如K3的``low``/``high``/``max``）；
    #     模型没有强度控制时为空。

    prompt_open_marker: str
    output_open_marker: str | None
    think_end_marker: str
    split: Callable[[str, bool], SplitResult | None]
    template_efforts: frozenset[str] = frozenset()
    # The literals whose presence in a *chat template* says the model speaks
    # this dialect. Read once at startup, exactly as the tool-call format is:
    # a template states what it taught the model, while an output only
    # exhibits whatever it happens to contain.
    # 【译】这些字面量出现在*chat template*里，就说明模型说这种方言。启动时读一次，与工具调用格式的做法完全相同：
    #   模板陈述的是它教给模型的东西，而输出只展示它碰巧包含的内容。
    template_markers: tuple[str, ...] = ()
    # Literals this format wraps every answer in, which are neither the
    # channel's own delimiters nor any tool format's. Removed from both halves
    # of the split and dropped by the streaming filter, so the two agree
    # without either of them depending on which tool-call parser happened to
    # be resolved -- the tool parser removes the same tokens, and a K3
    # deployment whose template refused the tools probe has none.
    # 【译】这种格式包在每个回答外面的字面量，它们既不是通道自己的分隔符，也不属于任何工具调用格式。
    #   在切分结果的两半中都会被去掉，流式过滤器也丢弃它们，这样两条路径一致，且都不依赖碰巧解析出的是哪个工具调用解析器——
    #   工具调用解析器会去掉同样的token，而一个模板拒绝了tools探测的K3部署根本没有工具调用解析器。
    content_framing: tuple[str, ...] = ()
    # Every marker that ends the reasoning channel, `think_end_marker`
    # included. A channel format can leave the think channel by *opening*
    # another one, and the streaming filter knew only the explicit close: a
    # K3 answer that goes straight to `<|open|>response<|sep|>` -- which its
    # own docs call the common path -- was streamed entirely as
    # `reasoning_content` with an empty `content`, while the non-streaming
    # split read it correctly.
    # 【译】所有会结束推理通道的标记，包括`think_end_marker`。通道格式可以通过*打开*另一个通道来离开think通道，
    #   而流式过滤器以前只认显式的关闭标记：一个直接跳到`<|open|>response<|sep|>`的K3回答——K3自己的文档称之为常见路径——
    #   流式时整个被当作`reasoning_content`、`content`为空，而非流式切分却读对了。
    # 解释：这个字段本身只放think_end_marker之外的那些；"包括think_end_marker"的完整集合是下面的end_markers属性
    extra_end_markers: tuple[str, ...] = ()

    # 完整的结束标记 = think_end_marker + extra_end_markers（ReasoningFilter在状态0/1监视它们）
    @property
    def end_markers(self) -> tuple[str, ...]:
        return (self.think_end_marker, *self.extra_end_markers)

    # 去掉所有外框字面量（非流式切分的两半都调用；流式由ReasoningFilter在命中时丢弃）
    def strip_framing(self, text: str) -> str:
        for marker in self.content_framing:
            text = text.replace(marker, "")
        return text

    # resolve_dialect用：文本中出现任一template_markers即算
    def taught_by(self, template_source: str) -> bool:
        """Does this template teach the model this dialect?"""
        # 【译】这个模板是否教过模型这种方言？
        return any(m in template_source for m in self.template_markers)


# --- Structured-channel dialect ---
# 【译】--- 结构化通道方言 ---


# Every literal that ends this format's reasoning channel, and the only place
# they are listed. `_split_channel` reads this and `_CHANNEL_DIALECT` derives
# `extra_end_markers` from it, so the non-streaming split and the streaming
# filter cannot be told different things; adding a marker in
# one place alone used to change only the streamed path.
# 【译】结束这种格式推理通道的所有字面量，而且这是唯一列出它们的地方。`_split_channel`读取它，
#   `_CHANNEL_DIALECT`从它推导出`extra_end_markers`，因此非流式切分与流式过滤器不可能被告知不同的东西；
#   过去只在一处加标记，只会改变流式路径。
_CHANNEL_END_MARKERS = (CHANNEL_THINK_END, CHANNEL_RESPONSE_START, k3.TOOLS_START)


# K3通道格式的非流式切分：推理通道在最先出现的结束标记处结束（结束标记见上面的_CHANNEL_END_MARKERS）
def _split_channel(text: str, starts_thinking: bool = False) -> SplitResult | None:
    """One rule: the reasoning channel ends at whichever closer comes first.

    This was three branches, and the middle one -- `<|open|>response<|sep|>`
    with no `<|close|>think<|sep|>` immediately before it -- returned
    `reasoning=None` and threw away everything ahead of the marker. A single
    byte between the two markers was enough to reach it, and the chain of
    thought then appeared in neither field. Silent data loss, not a
    divergence.

    The content comes back with its channel framing still on it, and that is
    deliberate: removing a tool-format's literals is the tool parser's job and
    it does it on both delivery paths, from one declared list. Doing it here
    too meant the reasoning stage removed them when the response was not
    streamed and left them when it was -- the streaming filter has no such
    step -- so the two paths only agreed by way of a *third* stage, and only
    when a K3 parser had been resolved. This one also truncated at
    `<|end_of_msg|>` rather than removing it, so anything after that token was
    deleted on one path alone.

    Gated on `starts_thinking`, which the two other branches were not: these
    markers only mean anything if a reasoning channel was actually opened.
    Ungated, any model's answer that *quotes* one had the text before it
    deleted -- an answer about K3's wire format lost 19 characters on
    `stream=false` and kept them on `stream=true`. That is the inference
    `parse_tool_calls` was changed to stop making, in the half that was left
    still making it; a prompt that opens some *other* dialect's channel can
    still reach this one, and the structural answer is to resolve the dialect
    at startup as the tool-call format now is.
    """
    # 【译】一条规则：推理通道在最先出现的那个关闭标记处结束。
    #
    #   这里以前有三个分支，中间那个——有`<|open|>response<|sep|>`但它紧前面没有`<|close|>think<|sep|>`——
    #   返回`reasoning=None`，并丢掉了标记之前的全部内容。两个标记之间只要有一个字节就会走到这个分支，
    #   思维链于是两个字段里都没有。这是静默的数据丢失，而不只是不一致。
    #
    #   返回的content仍带着通道外框，这是有意的：去掉工具调用格式的字面量是工具调用解析器的工作，它在两条交付路径上
    #   都按同一份声明的列表来做。在这里也做的话，推理阶段在非流式时去掉它们、流式时却保留——流式过滤器没有这一步——
    #   于是两条路径只能借助*第三个*阶段才一致，而且只有解析出K3解析器时才一致。这个函数过去还在`<|end_of_msg|>`处截断
    #   而不是去掉它，所以该token之后的内容只在一条路径上被删掉。
    #
    #   以`starts_thinking`为前提，而另外两个分支没有：这些标记只有在推理通道确实被打开时才有意义。
    #   不加这个前提的话，任何*引用*了其中一个标记的模型回答，标记之前的文本都会被删掉——一个讲K3线上格式的回答
    #   在`stream=false`时丢了19个字符，`stream=true`时却保留着。这正是`parse_tool_calls`已被改为不再做的推断，
    #   而剩下的这一半还在做；一个打开了*其他*方言通道的prompt仍然可能走到这里，结构性的解决办法是像现在的工具调用格式那样，
    #   在启动时就确定方言。
    # 解释：starts_thinking=False时也不会直接放弃，而是先在输出里找模型自己写的<|open|>think<|sep|>（见下面的注释），
    #   找到之后才去找结束标记；所以"前提"可以理解为：通道必须被打开过（prompt预置或模型自己打开），结束标记才算数
    before = ""
    if not starts_thinking:
        # A prompt that left thinking off does not stop the model opening a
        # channel of its own. The two inline dialects have always read that;
        # returning `None` here let the tool parser strip the framing and glue
        # the trace onto the answer, on both paths, so parity could not see it.
        # 【译】prompt没有打开思考，并不能阻止模型自己打开一个通道。两个内联方言一直能读出这种情况；
        #   这里如果返回`None`，工具调用解析器就会去掉外框、把推理过程粘到回答上，两条路径都如此，所以一致性检查发现不了。
        at = text.find(CHANNEL_THINK_START)
        if at != -1:
            # Searched, and what precedes it is content -- the two rulings
            # `_split_inline_tag` makes. Anchoring at 0 instead put this out
            # of step with the streaming filter on `\n\n<|open|>think<|sep|>`.
            # 【译】用搜索，并且它之前的内容算作content——与`_split_inline_tag`的两条裁定相同。
            #   如果改为锚定在位置0，遇到`\n\n<|open|>think<|sep|>`时就会与流式过滤器不一致。
            before, text = text[:at], text[at + len(CHANNEL_THINK_START) :]
        elif text.startswith(CHANNEL_THINK_END):
            # Declined to think; offset 0 only, see `_split_inline_tag`.
            # 【译】拒绝思考；仅限偏移0，见`_split_inline_tag`。
            return (None, text[len(CHANNEL_THINK_END) :])
        else:
            return None
    best_at, best = len(text), None
    for marker in _CHANNEL_END_MARKERS:
        at = text.find(marker)
        if 0 <= at < best_at:
            best_at, best = at, marker
    if best is None:
        # Never closed. Seeded, that is a trace cut off at `max_tokens` and
        # `separate_reasoning`'s fallback says so; self-opened, the channel's
        # start is already known.
        # 【译】一直没有关闭。如果是prompt预置打开的，这是一段在`max_tokens`处被截断的推理，`separate_reasoning`的兜底分支会这样处理；
        #   如果是模型自己打开的，通道的起点已经知道了。
        # 解释：模型自己打开的情况返回(开标记之后的全部, 开标记之前的部分)
        if starts_thinking:
            return None
        return (text or None, before)
    reasoning = text[:best_at]
    content = before + text[best_at + len(best) :]
    return (reasoning or None, content)


# --- Generic <think>...</think> dialect (K2/DeepSeek/Qwen3/MiniMax/...) ---
# 【译】--- 通用 <think>...</think> 方言（K2/DeepSeek/Qwen3/MiniMax/...）---

# No `\s*` after `</think>`. The newline a model puts before its answer is
# not a marker this dialect declares, so it survives -- see `SplitResult`.
# 【译】`</think>`后面不带`\s*`。模型在回答前放的换行不是这个方言声明的标记，所以保留下来——见`SplitResult`。
THINK_OPEN_MARKER = "<think>"
THINK_END_MARKER = "</think>"

# MiniMax-M3 spells the same shape with its own tags. Its template sets
# `think_begin_token = '<mm:think>'` and ends the generation prompt with it
# when `thinking_mode == "enabled"`, leaves no prefix under "adaptive" (so the
# model writes the opener itself), and emits `</mm:think>` alone under
# "disabled". Neither literal was declared anywhere, so the whole chain of
# thought was delivered as `content` with `reasoning_content: null`, on both
# delivery paths -- and on `/v1/messages`, inside the client's text block.
# 【译】MiniMax-M3用自己的标签拼写同一种形态。它的模板设置了`think_begin_token = '<mm:think>'`，
#   在`thinking_mode == "enabled"`时用它结束生成提示；在"adaptive"下不加前缀（由模型自己写开标记）；
#   在"disabled"下只输出一个`</mm:think>`。这两个字面量以前在哪里都没有声明，于是整段思维链都作为`content`交付、
#   `reasoning_content: null`，两条交付路径都如此——在`/v1/messages`上则出现在客户端的text块里。
MM_THINK_OPEN_MARKER = "<mm:think>"
MM_THINK_END_MARKER = "</mm:think>"


# 工厂函数：预编译"闭合块"和"未闭合块"两个正则，返回闭包split；DIALECTS中两个内联方言各用它生成一个
def _inline_tag_split(open_marker: str, end_marker: str) -> Callable:
    """A split for one `OPEN ... END` inline-tag dialect.

    Built per dialect from that dialect's own markers rather than reading
    module constants, because a second inline-tag format arrived and the
    single hardcoded pair would have silently split its output with the wrong
    tags -- the dialect declaring one thing and its split doing another is the
    two-sources shape this module exists to retire.
    """
    # 【译】为一种`OPEN ... END`内联标签方言生成split函数。
    #
    #   按方言用它自己的标记来构造，而不是读模块常量，因为后来出现了第二种内联标签格式，单个写死的标记对
    #   会悄悄用错误的标签切分它的输出——方言声明一套、它的split做另一套，正是本模块要消除的"两个来源"问题。
    closed_re = re.compile(
        re.escape(open_marker) + r"(.*?)" + re.escape(end_marker) + r"(.*)",
        flags=re.DOTALL,
    )
    open_re = re.compile(re.escape(open_marker) + r"(.*)", flags=re.DOTALL)

    def split(text: str, starts_thinking: bool = False) -> SplitResult | None:
        return _split_inline_tag(text, starts_thinking, end_marker, closed_re, open_re)

    return split


# 内联标签方言的非流式切分：按 预置打开 -> 偏移0拒绝思考 -> 闭合块 -> 未闭合块 的顺序判断，与ReasoningFilter的状态机一致
def _split_inline_tag(
    text: str,
    starts_thinking: bool,
    end_marker: str,
    closed_re,
    open_re,
) -> SplitResult | None:
    # Ordered as the streaming filter's state machine is, because that is what
    # this has to agree with. `starts_thinking` means the prompt already
    # opened the channel, so the output *begins* in it: the first `</think>`
    # closes it and any `<think>` before that is literal text inside the
    # reasoning, not an opener. Letting the searches below run first read one
    # as an opener and dropped everything ahead of it.
    # 【译】判断顺序与流式过滤器的状态机相同，因为这里必须与它一致。`starts_thinking`表示prompt已经打开了通道，
    #   所以输出*从通道内*开始：第一个`</think>`关闭它，在那之前出现的任何`<think>`都是推理内部的普通文本，而不是开标记。
    #   如果让下面的搜索先执行，就会把它当成开标记，并丢掉它之前的所有内容。
    if starts_thinking:
        # `</think>` with no `<think>`. Reasoning only because the prompt says
        # the channel is open -- ungated, this guessed, and disagreed with the
        # streaming path, which cannot honour an end marker it has no opener
        # for without waiting for one, and waiting is the stall. vLLM's
        # non-streaming path still guesses here and its streaming path does
        # not; the two do not agree, and this is the half worth copying.
        # 【译】有`</think>`而没有`<think>`。之所以算推理，只是因为prompt表明通道已打开——不加这个前提就是在猜，
        #   而且与流式路径不一致：流式路径要认一个没有开标记的结束标记，就只能等它出现，而等待就是卡顿。
        #   vLLM的非流式路径在这里仍然在猜，它的流式路径不猜；两者并不一致，值得照搬的是后者这一半。
        if end_marker in text:
            reasoning, _, content = text.partition(end_marker)
            return (reasoning or None, content)
        # Never closed: all reasoning, no answer. That is what a reasoning
        # model stopped at `max_tokens` looks like, and `separate_reasoning`'s
        # own fallback says it -- returning `None` defers to it rather than
        # writing the same answer twice.
        # 【译】一直没关闭：全是推理、没有回答。这就是推理模型在`max_tokens`处被截停的样子，
        #   `separate_reasoning`自己的兜底分支会给出这个结论——返回`None`就是交给它处理，而不是把同样的答案写两遍。
        return None

    # A closer with nothing before it: the model declined to think this turn.
    # MiniMax-M3's template renders every earlier no-thinking assistant turn
    # as a bare `</mm:think>`, so that is the trained form -- measured live,
    # 0 of 10 single-turn replies and 7 of 30 multi-turn. vLLM and SGLang both
    # carry code for it.
    #
    # Offset 0 only. Honouring the marker anywhere is the guess `0858a50d4`
    # removed for cause: it made an ordinary answer wait for a marker that
    # never came, and fed pre-marker text to the tool-call sniffer. Neither
    # reaches here -- nothing precedes offset 0, and the decision is made
    # within one marker's length, which the scanner already buffers.
    # 【译】前面什么都没有的关闭标记：模型本轮拒绝思考。MiniMax-M3的模板把之前每个不思考的assistant轮次都渲染成
    #   一个单独的`</mm:think>`，所以这是训练出来的形式——线上实测，单轮回复10次中0次，多轮30次中7次。
    #   vLLM和SGLang都有处理它的代码。
    #
    #   仅限偏移0。在任何位置都认这个标记，正是提交`0858a50d4`有理由删掉的那种猜测：它让普通回答去等一个永远不来的标记，
    #   还把标记之前的文本喂给了工具调用嗅探器。这两个问题在这里都不会出现——偏移0之前没有任何内容，
    #   而且判断在一个标记的长度之内就能做出，扫描器本来就会缓冲这么多。
    if text.startswith(end_marker):
        return (None, text[len(end_marker) :])

    # Closed block: <think>...</think> answer.
    #
    # Searched, not anchored at position 0: a block does not have to open the
    # output. Anchoring meant a model that answers, opens a `<think>` block
    # and answers again matched nothing, so the client was handed the literal
    # tags with the chain of thought inside `content`.
    #
    # What precedes the block is content because it *is* content -- text
    # outside the reasoning channel. Nothing about the split needs the
    # streaming filter to justify it; this function has the whole output.
    #
    # The *first* block only, and that one IS a parity choice rather than a
    # reading of the format: the streaming filter closes the channel on the
    # first `</think>` and never reopens it, so splitting every block here
    # would make `stream=false` disagree with `stream=true` on any output
    # with two -- swapping one divergence for another. Whether *both* should
    # reopen is a separate question, and answering it means changing the
    # filter, not this.
    # 【译】闭合块：<think>...</think> 回答。
    #
    #   用搜索，不锚定在位置0：思考块不一定在输出开头。锚定的话，一个先回答、再打开`<think>`块、然后又回答的模型
    #   什么都匹配不上，客户端拿到的是带着字面标签、思维链夹在`content`里的结果。
    #
    #   块之前的内容算作content，因为它*就是*content——推理通道之外的文本。这一点不需要拿流式过滤器来论证；
    #   本函数拿到的是整个输出。
    #
    #   只处理*第一个*块，这一点确实是为了一致性而不是对格式的解读：流式过滤器在第一个`</think>`处关闭通道、
    #   之后再也不打开，所以这里如果切分每一个块，对任何含两个块的输出`stream=false`都会与`stream=true`不一致——
    #   只是用一种不一致换了另一种。两边是否*都*应该重新打开是另一个问题，要回答它得改过滤器，而不是改这里。
    match = closed_re.search(text)
    if match:
        return (match.group(1) or None, text[: match.start()] + match.group(2))
    # Unclosed block (truncated response). Searched, and split, for the same
    # reasons as the closed one above.
    # 【译】未闭合的块（被截断的响应）。同样用搜索并切分，理由与上面的闭合块相同。
    match = open_re.search(text)
    if match:
        return (match.group(1) or None, text[: match.start()])
    return None


# "Channel" here follows the established meaning from OpenAI's Harmony format:
# one output stream carrying several named sections (think / response / tools),
# each wrapped in framing tokens, that we de-multiplex into separate fields.
# Harmony's analysis/final/commentary channels map onto K3's think/response/tools.
# We name the tokens by this cross-model concept (CHANNEL_*) rather than by the
# model. Channel-format models differ in their framing tokens, so each gets its
# own DIALECTS entry; the entry below carries Kimi-K3's token values.
#
# Detection/priority order: structured-channel dialects before inline-tag ones,
# so a specific channel marker is tried before the generic <think> tag.
# separate_reasoning() returns the first dialect whose split() matches. A dialect
# is identified by its markers/split behavior, not a label.
# 【译】这里的"通道"沿用OpenAI Harmony格式中已有的含义：一个输出流承载几个命名段落（think / response / tools），
#   每段都包在外框token里，我们把它们解复用到不同的字段。Harmony的analysis/final/commentary通道对应K3的think/response/tools。
#   我们按这个跨模型的概念（CHANNEL_*）给token命名，而不是按模型。各个通道格式模型的外框token不同，
#   所以每个模型在DIALECTS中各占一项；下面这项带的是Kimi-K3的token取值。
#
#   检测/优先级顺序：结构化通道方言排在内联标签方言之前，这样具体的通道标记会先于通用的<think>标签被尝试。
#   separate_reasoning()返回第一个split()匹配上的方言。方言由它的标记/split行为来识别，而不是靠一个标签名。
# 解释：第二段描述的是旧行为。现在separate_reasoning只用启动时resolve_dialect选出的那一个方言（找不到用FALLBACK_DIALECT），
#   不再逐个尝试；DIALECTS的顺序只决定resolve_dialect在多个方言都匹配时选哪个

# Read from the module both consumers read, not restated here -- a second
# hand-kept copy is how the two lists drifted apart. The tool-region brackets
# stay out: the reasoning stage cannot remove what is between them (see
# `kimi_k3_tokens`).
# 【译】从两个使用方都读取的那个模块里读，而不是在这里重抄一遍——第二份手工维护的副本正是两份列表逐渐分歧的原因。
#   工具区域的括号不放进来：推理阶段无法去掉它们之间的内容（见`kimi_k3_tokens`）。
_CHANNEL_CONTENT_FRAMING = k3.CHANNEL_FRAMING

# 【分支】Kimi-K3的结构化通道方言（K3模型本身见第8批）
_CHANNEL_DIALECT = ReasoningDialect(
    prompt_open_marker=CHANNEL_THINK_START,
    # The same literal, not `None`. "The template injects it, so the model
    # never writes one" is false when the prompt leaves thinking off and the
    # model opens a channel anyway -- and `ReasoningFilter` builds
    # `_open_markers` from this field, so state 0 had nothing to leave itself
    # on and delivered the whole trace as the answer.
    # 【译】用同一个字面量，而不是`None`。"模板会注入它，所以模型从不自己写"——当prompt没有打开思考而模型仍然打开通道时，
    #   这个说法就不成立了；而`ReasoningFilter`用这个字段构造`_open_markers`，于是状态0没有可以离开的依据，把整段推理当作回答交付了。
    output_open_marker=CHANNEL_THINK_START,
    think_end_marker=CHANNEL_THINK_END,
    # `response` is what the trained format always opens next -- Kimi-K3's
    # encoder emits a complete response section before any tools section, so
    # `tools` is reachable only from a malformed generation. Listed anyway,
    # and cheaply: without it a think block that jumps straight to
    # `<|open|>tools<|sep|>` puts the entire tool call in `reasoning_content`
    # with empty `content`, on both delivery paths, and on `/v1/messages`
    # where reasoning is dropped by default the client gets nothing at all.
    # 【译】训练出的格式接下来总是打开`response`——Kimi-K3的编码器在任何tools段之前都会输出一个完整的response段，
    #   所以只有生成出错时才会走到`tools`。仍然把它列上，代价很小：没有它的话，一个直接跳到`<|open|>tools<|sep|>`的think块
    #   会把整个工具调用放进`reasoning_content`、`content`为空，两条交付路径都如此；而在默认丢弃推理内容的`/v1/messages`上，
    #   客户端什么也拿不到。
    extra_end_markers=tuple(m for m in _CHANNEL_END_MARKERS if m != CHANNEL_THINK_END),
    split=_split_channel,
    # 【译】（行尾注释）Kimi-K3
    template_efforts=frozenset({"low", "high", "max"}),  # Kimi-K3
    template_markers=(CHANNEL_THINK_START, CHANNEL_THINK_END),
    content_framing=_CHANNEL_CONTENT_FRAMING,
)

# 方言表：顺序 = resolve_dialect的匹配优先级（通道格式 -> MiniMax-M3 -> 通用<think>），最后一项兼作FALLBACK_DIALECT
DIALECTS: tuple[ReasoningDialect, ...] = (
    # Structured channel format — Kimi-K3 token values (see CHANNEL_* above)
    # 【译】结构化通道格式——Kimi-K3的token取值（见上面的CHANNEL_*）
    _CHANNEL_DIALECT,
    # MiniMax-M3: the inline-tag shape in its own spelling. Before the generic
    # entry, so its longer markers are matched first -- though the two sets do
    # not overlap as substrings, which is checked below.
    # 【译】MiniMax-M3：内联标签形态的自有拼写。放在通用项之前，这样它更长的标记先被匹配——不过两组标记并不互为子串，下面会检查这一点。
    # 解释：本文件下方并没有这项检查（tests/entrypoints/test_reasoning.py里也没有找到对应断言）
    # 【分支】MiniMax-M3模型本身见第8批
    ReasoningDialect(
        prompt_open_marker=MM_THINK_OPEN_MARKER,
        output_open_marker=MM_THINK_OPEN_MARKER,
        think_end_marker=MM_THINK_END_MARKER,
        split=_inline_tag_split(MM_THINK_OPEN_MARKER, MM_THINK_END_MARKER),
        template_markers=(MM_THINK_OPEN_MARKER, MM_THINK_END_MARKER),
    ),
    # Generic <think>...</think> (K2/DeepSeek/Qwen3/...)
    # 【译】通用 <think>...</think>（K2/DeepSeek/Qwen3/...）
    ReasoningDialect(
        prompt_open_marker=THINK_OPEN_MARKER,
        output_open_marker=THINK_OPEN_MARKER,
        think_end_marker=THINK_END_MARKER,
        split=_inline_tag_split(THINK_OPEN_MARKER, THINK_END_MARKER),
        template_markers=(THINK_OPEN_MARKER, THINK_END_MARKER),
    ),
)

# The dialect a model speaks when its template names none of them. The
# inline-`<think>` one, and not "no dialect at all": a template that renders
# the tag conditionally may not carry it in the source we can read, and the
# cost of guessing wrong here is nil -- a model that never writes `<think>`
# never matches its markers, so the split is a no-op. Guessing the other way
# would ship a chain of thought to the client as the answer.
# 【译】模板一个方言都没提到时模型所说的方言。选inline-`<think>`那个，而不是"没有方言"：一个有条件地渲染该标签的模板，
#   在我们能读到的源码里不一定带着它；而在这里猜错的代价为零——从不写`<think>`的模型永远匹配不上它的标记，切分就是空操作。
#   反过来猜的话，会把思维链当作回答发给客户端。
FALLBACK_DIALECT = DIALECTS[-1]


# 启动时调用一次（api_server.py、atomesh）：先在渲染出的探测prompt、再在模板源码中找各方言的template_markers
def resolve_dialect(
    template_source: str, rendered_prompt: str = ""
) -> tuple[ReasoningDialect, bool]:
    """Which reasoning dialect this model speaks, and whether it was stated.

    Asked once, at startup -- the same question, in the same place and by the
    same means, as the tool-call format. It used to be asked twice per response
    and differently each time: the non-streaming split tried every dialect in
    order and took the first that matched, while the streaming filter used the
    *union* of every dialect's end markers and closed on whichever came first.
    So a `<think>` model answering a question about Kimi's wire format ended
    its chain of thought at the quoted `<|open|>response<|sep|>` when streamed
    and at the real `</think>` when not.

    Both kinds of evidence, and the *render* first, because the source is not
    always readable. Kimi-K3 ships neither a Jinja template nor an encoder
    under the path the loader looks in, so `chat_template_source` returns ""
    for it -- and falling back to the inline-`<think>` dialect on that made a
    K3 answer come back as `reasoning_content` with `content` empty, on both
    delivery modes, while the tool-call format resolved correctly off the
    rendered probe. Same model, two answers to "what does this model speak",
    from two different pieces of evidence. Now it is one function reading
    both.

    The second return value is whether a dialect was actually named, so the
    caller can say at startup which one it is and whether it was a fallback.
    """
    # 【译】这个模型说哪种推理方言，以及它是否被明确说明了。
    #
    #   只在启动时问一次——与工具调用格式是同一个问题，在同一个地方、用同样的方法问。它过去每个响应问两次，
    #   而且每次问法不同：非流式切分按顺序尝试每个方言、取第一个匹配的；流式过滤器则用所有方言结束标记的*并集*，
    #   哪个先来就在哪里关闭。于是一个`<think>`模型在回答关于Kimi线上格式的问题时，流式下思维链在被引用的
    #   `<|open|>response<|sep|>`处结束，非流式下却在真正的`</think>`处结束。
    #
    #   两种证据都看，而且先看*渲染结果*，因为源码并不总能读到。Kimi-K3在加载器查找的路径下既没有Jinja模板也没有编码器，
    #   所以`chat_template_source`对它返回""——在这种情况下退回inline-`<think>`方言，曾让K3的回答以`reasoning_content`返回、
    #   `content`为空，两种交付模式都如此，而工具调用格式却根据渲染出的探测prompt正确解析了。同一个模型，
    #   对"这个模型说什么"给出两个答案，来自两份不同的证据。现在由一个函数同时读两者。
    #
    #   第二个返回值表示是否真的命中了某个方言，这样调用方可以在启动时说明是哪一个、是否为兜底。
    for evidence in (rendered_prompt, template_source):
        if not evidence:
            continue
        for dialect in DIALECTS:
            if dialect.taught_by(evidence):
                return dialect, True
    return FALLBACK_DIALECT, False
