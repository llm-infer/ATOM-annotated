# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models.

A dialect-agnostic engine: it separates the reasoning channel from the final
answer, both for a complete response (``separate_reasoning``) and for a token
stream (``ReasoningFilter``), emitting the standard ``reasoning_content`` field.
All model-specific marker knowledge lives in ``reasoning_dialects.DIALECTS``;
this module contains no per-model conditions. Add a model there, not here.
"""

# 【译】思考模型的推理/思考内容分离。
#
#   一个与方言无关的引擎：它把推理通道与最终回答分开，既用于完整响应（``separate_reasoning``），
#   也用于token流（``ReasoningFilter``），输出标准的``reasoning_content``字段。
#   所有与模型相关的标记知识都放在``reasoning_dialects.DIALECTS``里；本模块不含任何按模型的判断。
#   要支持新模型，加在那里，不要加在这里。
#
# 【导读】推理内容（reasoning content，即thinking部分）与最终回答的分离
# 要解决的问题：思考模型（DeepSeek-R1、Qwen3、Kimi-K3、MiniMax-M3等）把思考过程和回答写在同一个输出流里，
#   靠标记分隔（<think>...</think>、<mm:think>...</mm:think>、K3的<|open|>think<|sep|>...<|close|>think<|sep|>）；
#   OpenAI接口要求思考放进reasoning_content字段、回答放进content字段（Anthropic接口则是thinking块和text块）
#   下文的"推理通道"就是被这对标记包住的那段思考；一族模型的这套标记写法叫一种"方言"（dialect，定义在reasoning_dialects.py）
#   入门科普（推理模型为什么先写思考、模板怎样把开标记预填在prompt末尾，带图）：api_server.py 中 chat_completions 里的【科普】推理内容
# 这一组文件的分工：
#   reasoning.py（本文件）：方言无关的引擎
#     ReasoningChannel：一次响应"怎么读推理通道" = (方言, 输出是否一开始就在通道内) 两个事实绑在一起；
#       .split(text) 走非流式（separate_reasoning），.stream() 返回流式过滤器（ReasoningFilter）
#     ReasoningFilter：流式状态机（0=通道打开前、1=通道内、2=通道关闭后），逐chunk输出(field, text)
#     template_opens_reasoning_implicitly / prompt_starts_in_reasoning / thinking_switched_off：决定starts_open的输入
#   reasoning_dialects.py：每种方言的标记和非流式split规则，以及启动时选方言的resolve_dialect
#   marker_scanner.py：流式路径上"能放出多少字节而不切断一个标记"的唯一实现，与工具调用解析器共用
#     （标记可能被chunk边界切开：上一个chunk以"<th"结尾、下一个以"ink>"开头，所以"<th"要先扣住；例子见该文件头）
#   kimi_k3_tokens.py：Kimi-K3通道格式的token字面量，推理方言和K3工具调用解析器共用
# 调用方：
#   启动时：api_server.py（以及atomesh/atom_standalone_service.py）调resolve_dialect从chat template确定方言，
#     调template_opens_reasoning_implicitly判断模型是否"不写开标记就直接开始思考"（如DeepSeek-R1）
#   每个请求：api_server.reasoning_channel()（atomesh里是对应的方法）构造ReasoningChannel，用于
#     /v1/chat/completions（serving_chat.py：非流式split、流式stream）、/v1/messages（serving_anthropic.py）
#     和/v1/responses（api_server.responses_create）；/v1/completions没有chat template，不做推理分离
# 与主干的关系：位于服务层的输出后处理，在detokenize之后、工具调用解析器之前——
#   先由本模块切出reasoning_content，剩下的content再交给tool_parser解析工具调用（见serving_chat.py）
# 贯穿全文的约束：同一个输出在stream=true和stream=false下必须切出逐字节相同的结果；
#   下面大段英文注释记录的多是两条路径曾经不一致的bug，读的时候抓住这一条主线即可

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The same reader the tool-call side uses, not a second one: this module's
# whole thesis is that asking "how much can be released without splitting a
# marker" in more than one place is how the answers drift apart, and it used
# to answer it itself, twice, with a different tie-break.
# 【译】与工具调用那边用的是同一个读取器，而不是另写一个：本模块的核心论点是，在不止一个地方回答
#   "能放出多少字节而不切断一个标记"，正是各处答案逐渐分歧的原因；而它过去自己回答过这个问题，
#   还回答了两次，平局规则也不一样。
from .marker_scanner import MarkerScanner
from .reasoning_dialects import DIALECTS, FALLBACK_DIALECT, ReasoningDialect

# Markers a rendered prompt ends with when the template already opened reasoning
# (the output then begins inside the reasoning channel with no opening tag).
# Every dialect's, because this is asked of a *prompt* before the response's
# dialect matters, and a prompt carries at most one of them.
# 【译】当模板已经打开推理通道时，渲染后的prompt会以这些标记结尾（此时输出一开始就在推理通道内，没有开标签）。
#   收集所有方言的标记，因为这个问题针对的是*prompt*，此时还轮不到响应的方言起作用，而且一个prompt最多只带其中一个。
_REASONING_OPEN_MARKERS = tuple(d.prompt_open_marker for d in DIALECTS)
# Reasoning-effort levels accepted across all loaded dialects' chat templates.
# resolve_thinking() clamps a request's effort to this set before forwarding it,
# so an effort no template understands is never passed through.
# 【译】所有已加载方言的chat template接受的推理强度（reasoning effort）档位的并集。
#   resolve_thinking()在转发请求的effort之前先把它限制在这个集合内，因此任何模板都不认识的effort永远不会被透传。
# 解释：resolve_thinking在serving_chat.py里；字符串effort不在集合内时被置为None（整数1~100另有校验）
VALID_TEMPLATE_EFFORTS = frozenset().union(*(d.template_efforts for d in DIALECTS))


# 启动时调用一次（api_server.py、atomesh）：从模板源码判断模型是否"隐式进入思考"，结果参与每个请求的starts_open
def template_opens_reasoning_implicitly(template_source: str) -> bool:
    """Does this model begin inside the reasoning channel with no marker at all?

    Some families close a reasoning block they never open: DeepSeek-R1 emits
    `</think>` but neither its prompt nor its output carries `<think>`. Nothing
    in a single response says so -- the first token is already reasoning and
    looks like an answer -- so it has to be known before the response starts.

    The chat template says it. A template that mentions an end marker and not
    the matching opener is describing exactly that shape; one that mentions
    both (Qwen3, MiniMax-M3) describes a model that opens its own, and one
    that mentions neither (gpt-oss) has no reasoning channel to speak of.

    ``template_source`` is the template's own text, which is the only place
    this shows: an end marker is what the template does with a *reply*, so it
    never reaches a fresh prompt. Measured -- Qwen3.5's source carries both
    markers while its rendered prompt carries only the opener, and Qwen3-8B's
    rendered prompt carries neither, so asking a render would answer False for
    every model alive. Get it from `chat_encoders.chat_template_source`, which
    also handles the two shapes that answer False by accident: a ``dict`` of
    named templates, and the ``None`` of a model that ships a Python encoder.

    This is what vLLM expresses by registering `DeepSeekR1ReasoningParser` for
    R1 -- an override whose only job is to treat a stream with no start token
    as reasoning until `</think>`. Same fact, derived instead of listed.
    """
    # 【译】这个模型是否一个标记都不写、直接从推理通道内部开始输出？
    #
    #   有些模型家族会关闭一个从没打开过的推理块：DeepSeek-R1会输出`</think>`，但它的prompt和输出里都没有`<think>`。
    #   单个响应里没有任何东西能说明这一点——第一个token就已经是推理内容，看起来却像回答——所以必须在响应开始之前就知道。
    #
    #   chat template会说明这一点。一个模板提到了结束标记却没提对应的开标记，描述的正是这种形态；
    #   两个都提到的（Qwen3、MiniMax-M3）描述的是会自己打开推理块的模型；两个都没提的（gpt-oss）根本谈不上有推理通道。
    #
    #   ``template_source``是模板自身的源码文本，这是唯一能看出这一点的地方：结束标记是模板处理*回复*时用的，
    #   所以永远不会出现在新渲染的prompt里。实测——Qwen3.5的源码两个标记都有，而渲染出的prompt只有开标记；
    #   Qwen3-8B渲染出的prompt两个都没有，所以如果去问渲染结果，对现存的所有模型都会答False。
    #   请从`chat_encoders.chat_template_source`获取它，那个函数还处理了两种会意外答False的形态：
    #   由多个命名模板组成的``dict``，以及自带Python编码器的模型对应的``None``。
    #
    #   vLLM通过为R1注册`DeepSeekR1ReasoningParser`来表达同一件事——那个覆盖实现唯一的作用，
    #   就是把没有开始token的流一直当作推理，直到`</think>`。同一个事实，这里是推导出来的，而不是逐个列出来的。
    for dialect in DIALECTS:
        if not dialect.think_end_marker:
            continue
        opener = dialect.output_open_marker or dialect.prompt_open_marker
        if dialect.think_end_marker in template_source and (
            not opener or opener not in template_source
        ):
            return True
    return False


# 每个请求调用：合并后的模板kwargs里是否带着"关闭思考"的取值；toggle=(参数名, 关闭值, 开启值)，启动时从模板解析得到
def thinking_switched_off(
    template_kwargs: dict | None, toggle: tuple[str, Any, Any] | None
) -> bool:
    """Did the render actually carry this template's off-switch?

    The merged kwargs are the whole answer -- server defaults, the client's
    `chat_template_kwargs`, and the request's `thinking` written in by the
    toggle's name on top. Reading the request field alone missed the first
    two, so an operator's `--default-chat-template-kwargs` never reached the
    decision.

    Only the resolved toggle's own name and value count: a kwarg the template
    does not read changes nothing about what the model does, and believing it
    would stop the channel being separated while the model went on producing
    one.
    """
    # 【译】这次渲染是否真的带上了该模板的"关闭思考"开关？
    #
    #   合并后的kwargs就是全部答案——服务端默认值、客户端的`chat_template_kwargs`，以及按开关名写在最上层的请求`thinking`。
    #   只读请求字段会漏掉前两者，于是运维设置的`--default-chat-template-kwargs`从来到不了这个判断。
    #
    #   只有解析出的开关自己的名字和取值才算数：模板不读取的kwarg对模型的行为毫无影响，
    #   如果相信它，就会在模型继续产出推理通道的同时停止分离这个通道。
    if not template_kwargs or toggle is None:
        return False
    name, off_value, _on_value = toggle
    return name in template_kwargs and template_kwargs[name] == off_value


# 每个请求调用：渲染后的prompt是否以某个方言的"开思考"标记结尾（即模板把开标记注入到了prompt末尾）
def prompt_starts_in_reasoning(prompt: str) -> bool:
    """True if the rendered ``prompt`` ends by opening a reasoning channel.

    Model-agnostic: callers pass the rendered prompt and don't need to know which
    dialect's marker applies. Used to seed the streaming filter
    (:attr:`ReasoningFilter.starts_thinking`).

    Offsets, not `prompt.rstrip()`, which copies the whole rendered prompt to
    read its last twenty bytes. That prompt is the largest string the server
    handles -- system prompt, tool schemas and the entire history -- and this
    runs once per request on the event loop: 4.8 us at 512 KB against 0.43 us
    here. Same shape as `begin_of_markup`.
    """
    # 【译】如果渲染后的``prompt``以打开推理通道结尾，返回True。
    #
    #   与模型无关：调用方传入渲染后的prompt，不需要知道该用哪个方言的标记。
    #   用来给流式过滤器设初始状态（:attr:`ReasoningFilter.starts_thinking`）。
    #
    #   用偏移量，而不是`prompt.rstrip()`——后者为了读最后二十个字节要复制整个渲染后的prompt。
    #   这个prompt是服务端处理的最大字符串——系统提示、工具schema和全部历史——而这里每个请求都要在事件循环上跑一次：
    #   512KB时rstrip要4.8us，这里只要0.43us。与`begin_of_markup`的写法相同。
    end = len(prompt)
    while end > 0 and prompt[end - 1].isspace():
        end -= 1
    return any(prompt.endswith(m, 0, end) for m in _REASONING_OPEN_MARKERS)


# 已分词prompt（多模态请求）的版本：只decode末尾几个token再判断
def prompt_tokens_start_in_reasoning(
    token_ids: Sequence[int], decode: Callable[[Sequence[int]], str]
) -> bool:
    """:func:`prompt_starts_in_reasoning` for an already-tokenized prompt.

    Multimodal requests reach the engine as token ids, and decoding all of them
    would be wasteful — an image prompt runs to thousands of tokens while only
    the end is inspected. Decoding as many trailing tokens as the longest marker
    has *characters* is always enough, because a token never renders to fewer
    than one character.

    ``decode`` is injected so this module stays free of tokenizer knowledge.
    """
    # 【译】:func:`prompt_starts_in_reasoning`的已分词版本。
    #
    #   多模态请求以token id的形式到达引擎，把它们全部decode会很浪费——一个图像prompt有几千个token，
    #   而只需要检查结尾。decode的尾部token数取最长标记的*字符*数就总是足够，因为一个token渲染出来至少有一个字符。
    #
    #   ``decode``通过参数注入，这样本模块不需要了解tokenizer。
    if not _REASONING_OPEN_MARKERS or not len(token_ids):
        return False
    tail = max(len(m) for m in _REASONING_OPEN_MARKERS)
    return prompt_starts_in_reasoning(decode(token_ids[-tail:]))


# 非流式：完整输出 -> (reasoning_content, content)；由ReasoningChannel.split调用
def separate_reasoning(
    text: str,
    starts_thinking: bool = False,
    dialect: ReasoningDialect | None = None,
) -> tuple[str | None, str]:
    """Separate reasoning content from the final answer.

    ``dialect`` is the one this model speaks, resolved from its chat template
    at startup (:func:`~.reasoning_dialects.resolve_dialect`). One, not each in
    turn: trying them in order let any model's answer be split by a dialect it
    does not speak, so an answer *about* Kimi's wire format lost the text
    before a quoted channel token -- and the streaming filter, which used a
    different rule again, kept it. Callers should go through
    :class:`ReasoningChannel`, which carries this and ``starts_thinking``
    together so the two paths cannot be given different ones.

    ``starts_thinking`` is the same answer :class:`ReasoningFilter` takes, and
    for the same reason: an output that begins inside the reasoning channel
    carries no opening marker, so nothing in the text says so. Both paths have
    to be told, or the same response is split one way when streamed and
    another when not -- measured, a reasoning model truncated at ``max_tokens``
    returned its whole trace as ``content`` here and as ``reasoning_content``
    when streamed.

    Returns:
        Tuple of (reasoning_content, content). reasoning_content is None if no
        thinking block was found.
    """
    # 【译】把推理内容与最终回答分开。
    #
    #   ``dialect``是这个模型所说的方言，启动时从它的chat template解析得到（:func:`~.reasoning_dialects.resolve_dialect`）。
    #   只用一个，而不是逐个尝试：按顺序尝试会让任何模型的回答被一个它并不说的方言切分，于是一个*谈论*Kimi线上格式的回答
    #   丢掉了被引用的通道token之前的文本——而流式过滤器用的又是另一套规则，保留了这段文本。
    #   调用方应通过:class:`ReasoningChannel`调用，它把这个参数和``starts_thinking``绑在一起，两条路径就不会拿到不同的值。
    #
    #   ``starts_thinking``与:class:`ReasoningFilter`接收的是同一个答案，原因也相同：一开始就处于推理通道内的输出
    #   不带开标记，文本本身说明不了这一点。两条路径都必须被告知，否则同一个响应在流式和非流式下会被切成两种样子——
    #   实测，一个在``max_tokens``处被截断的推理模型，在这里把整段推理作为``content``返回，流式时却作为``reasoning_content``返回。
    #
    #   返回：
    #       (reasoning_content, content)元组。没找到思考块时reasoning_content为None。
    speaking = dialect or FALLBACK_DIALECT
    result = speaking.split(text, starts_thinking)
    if result is not None:
        # Framing removed here and not inside the dialect's split, because the
        # two fallbacks below are splits too and they were missing it -- so a
        # channel-format answer with no closer, or one on a stream that never
        # opened the channel, kept its tokens on this path while the streaming
        # filter dropped them. One place, applied to whatever comes back.
        # 【译】外框（framing）在这里去掉，而不是在方言的split里去掉，因为下面两个兜底分支也是切分，它们之前漏了这一步——
        #   于是一个没有关闭标记的通道格式回答，或者一个从未打开通道的流上的回答，在这条路径上保留了这些token，
        #   而流式过滤器却把它们丢掉了。统一在一个地方处理，对返回的任何结果都执行。
        reasoning, content = result
        return (
            speaking.strip_framing(reasoning or "") or None,
            speaking.strip_framing(content),
        )
    if starts_thinking:
        # The prompt opened the channel and the model never closed it, which
        # is what a reasoning model stopped at `max_tokens` looks like. It
        # produced reasoning and no answer; the streaming filter says exactly
        # that from state 1, and this has to agree.
        #
        # `or None` because it has to agree on the empty case too: the filter
        # emits no segment at all for an empty output, so a bare `text` here
        # put `reasoning_content: ""` in the non-streaming body and nothing in
        # the streamed one. Every other return in this function spells absence
        # as `None`.
        # 【译】prompt打开了通道而模型一直没有关闭它，这正是推理模型在`max_tokens`处被截停的样子。它只产出了推理、没有回答；
        #   流式过滤器从状态1出发给出的正是这个结论，这里必须与之一致。
        #
        #   用`or None`是因为空输出的情况也必须一致：对空输出，过滤器根本不产出任何片段，所以这里如果直接返回`text`，
        #   非流式响应体里会出现`reasoning_content: ""`，而流式响应里什么都没有。本函数的其他返回都用`None`表示"没有"。
        return (speaking.strip_framing(text) or None, "")
    # No reasoning markers — return content as-is (tool calls parsed separately).
    # 【译】没有推理标记——原样返回内容（工具调用另行解析）。
    return (None, speaking.strip_framing(text))


# 流式过滤器 = 三状态机 + MarkerScanner（由ReasoningChannel.stream()创建，每个响应/每个n分支一个）
#   状态0（通道打开前）：监视开标记、结束标记和外框；结束标记出现在偏移0处表示模型声明"本轮不思考"，直接进入状态2
#   状态1（通道内）：监视所有结束标记（think_end_marker + extra_end_markers）；放出的文本记为reasoning_content
#   状态2（通道后）：只监视外框；<think>方言没有外框，此时_scanner为None，chunk直接透传为content
#   MarkerScanner只暂扣"可能是某个标记前缀"的尾部（长度小于最长标记），所以不会把流卡住
#   例（<think>方言，starts_thinking=False），chunk依次为"<thi"、"nk>想一想</th"、"ink>答案"：
#     第1块整块暂扣；第2块命中<think>进入状态1，放出"想一想"作为reasoning_content并暂扣"</th"；
#     第3块命中</think>进入状态2，"答案"作为content放出
class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Chunks in, ``(field, text)`` tuples out, where field is
    ``"reasoning_content"`` or ``"content"``. Three states: before the channel
    opens, inside it, after it.

    ``starts_thinking`` handles templates that inject the opening marker into
    the prompt itself (Kimi-K3 ends the prompt with its think opener): the
    output then begins *inside* the channel with no opening tag, so nothing in
    the text says so and the filter must be told.

    ``dialect`` is the one this model speaks, and the same object
    :func:`separate_reasoning` is given. Go through :class:`ReasoningChannel`
    rather than passing it by hand; that is what keeps the two paths on one
    answer.

    Every marker question goes to :class:`MarkerScanner`, the same reader the
    tool-call side uses. It used to have its own: an `_earliest_marker` whose
    tie-break was the opposite of the scanner's (first-declared rather than
    longest, so `<think>` would be reported where `<thinking>` was meant), the
    hold-and-cut arithmetic open-coded twice, and `self.buf += text` on an
    attribute -- the quadratic append `Region` exists to avoid -- on the
    reasoning half of every long chain of thought. All three in a module whose
    first paragraph says asking this question in more than one place is the
    bug being retired.
    """

    # 【译】有状态的流式过滤器，把推理内容与正文分开。
    #
    #   输入chunk，输出``(field, text)``元组，field是``"reasoning_content"``或``"content"``。
    #   三个状态：通道打开之前、通道之内、通道之后。
    #
    #   ``starts_thinking``用于那些把开标记注入到prompt本身的模板（Kimi-K3在prompt末尾放了它的think开标记）：
    #   此时输出从通道*内部*开始，没有开标签，文本本身说明不了这一点，必须告诉过滤器。
    #
    #   ``dialect``是这个模型说的方言，与:func:`separate_reasoning`拿到的是同一个对象。请通过:class:`ReasoningChannel`获取，
    #   不要手工传参；正是这样才保证两条路径用同一个答案。
    #
    #   所有关于标记的问题都交给:class:`MarkerScanner`，即工具调用那边用的同一个读取器。它过去有自己的一套：
    #   一个`_earliest_marker`，其平局规则与扫描器相反（取最先声明的而不是最长的，于是本该是`<thinking>`时会报告`<think>`）；
    #   "暂扣再切分"的算术手写了两遍；还在属性上做`self.buf += text`——正是`Region`要避免的二次方拼接——
    #   而且发生在每条长思维链的推理那一半上。三个问题全都出现在一个第一段就声明"在多处问这个问题正是要消除的bug"的模块里。

    def __init__(
        self,
        starts_thinking: bool = False,
        dialect: ReasoningDialect | None = None,
    ) -> None:
        self.starts_thinking = starts_thinking
        self.dialect = dialect
        speaking = dialect or FALLBACK_DIALECT
        self._end_markers = frozenset(speaking.end_markers)
        self._framing = frozenset(speaking.content_framing)
        self._open_markers = frozenset(
            (speaking.output_open_marker,) if speaking.output_open_marker else ()
        )
        self.state = 1 if starts_thinking else 0
        # Nothing has been released yet, so an end marker arriving now sits at
        # offset 0 -- which is the only place it means "I did not think this
        # turn". See `_split_inline_tag`, whose answer this has to match.
        # 【译】此时还什么都没放出，所以现在到来的结束标记位于偏移0——只有在这个位置它才表示"本轮我没有思考"。
        #   见`_split_inline_tag`，这里的结论必须与它一致。
        self._at_offset_zero = not starts_thinking
        self._scanner = self._scanner_for(self.state)

    def _scanner_for(self, state: int) -> MarkerScanner | None:
        """What this state has to watch for. ``None`` when that is nothing,
        which is the common case for the inline-`<think>` dialect after the
        channel has closed."""
        # 【译】这个状态需要监视什么。没有要监视的就返回``None``，这是inline-`<think>`方言在通道关闭后的常见情况。
        watched = set(self._framing)
        if state == 0:
            # End markers too, though the channel is not open: at offset 0 one
            # means the model declined to think, further in it is the model's
            # own words, and watching is what tells them apart. Free --
            # measured 1.00x on all four dialects, since `MarkerScanner` costs
            # by chunk and not by marker count.
            # 【译】通道虽未打开，也要监视结束标记：在偏移0处出现表示模型拒绝思考，出现在更后面则是模型自己说的话，
            #   监视它才能区分这两种情况。这没有代价——在全部四种方言上实测都是1.00倍，因为`MarkerScanner`的开销按chunk算，不按标记数算。
            # 解释：当前DIALECTS只有三项（K3通道格式、MiniMax-M3的<mm:think>、通用<think>），"四种"无法与现有代码一一对应
            watched |= self._open_markers | self._end_markers
        elif state == 1:
            watched |= self._end_markers
        return MarkerScanner(tuple(sorted(watched))) if watched else None

    @property
    def _field(self) -> str:
        return "reasoning_content" if self.state == 1 else "content"

    # 返回[(field, text), ...]；一个chunk可能连续跨过多个标记，所以用while循环反复处理scan.rest
    def process(self, text: str) -> list:
        """Consume one chunk; return what it completed."""
        # 【译】消费一个chunk；返回它所完成的片段。
        out: list = []
        while True:
            if self._scanner is None:
                if text:
                    out.append((self._field, text))
                return out
            scan = self._scanner.feed(text)
            text = ""
            if scan.released:
                out.append((self._field, scan.released))
                if self.state == 0:
                    self._at_offset_zero = False
            if scan.hit is None:
                return out
            if scan.hit in self._framing and not (
                self.state == 1 and scan.hit in self._end_markers
            ):
                # Framing this format wraps every answer in. Removed here and
                # not only by the tool parser: doing it there alone made the
                # answer depend on which tool-call format was resolved, so a
                # K3 deployment whose template refused the tools probe handed
                # its channel tokens to the client on both delivery paths.
                # 【译】这种格式包裹在每个回答外面的外框。在这里去掉，而不是只由工具调用解析器去掉：只在那里做的话，
                #   结果会取决于解析出的是哪种工具调用格式，于是一个模板拒绝了tools探测的K3部署，在两条交付路径上都把通道token交给了客户端。
                text = scan.rest
                continue
            if self.state == 0 and scan.hit in self._end_markers:
                # Below the framing branch on purpose: K3 declares
                # `<|open|>response<|sep|>` as both, and framing wins -- it
                # says the same thing this rule would, at any offset.
                # 【译】故意放在外框分支下面：K3把`<|open|>response<|sep|>`同时声明为外框和结束标记，外框优先——
                #   在任何偏移处，它表达的都与这条规则相同。
                if self._at_offset_zero:
                    # 【译】（行尾注释）拒绝思考；剩下的都是回答
                    self.state = 2  # declined to think; the rest is the answer
                    self._scanner = self._scanner_for(self.state)
                else:
                    # 【译】（行尾注释）模型自己说的话
                    out.append((self._field, scan.hit))  # the model's own words
                text = scan.rest
                continue
            # A channel boundary: state 0 -> 1 on the opener, 1 -> 2 on any
            # end marker. A dialect that opens the channel from the prompt has
            # no opener to see, so it starts in state 1 and only ever takes
            # the second of these.
            # 【译】通道边界：遇到开标记时状态0 -> 1，遇到任一结束标记时1 -> 2。由prompt打开通道的方言看不到开标记，
            #   所以从状态1开始，只会走第二种转移。
            self.state = 1 if self.state == 0 else 2
            self._scanner = self._scanner_for(self.state)
            text = scan.rest

    # 流结束时调用：放出扫描器仍暂扣的尾部
    def flush(self) -> list:
        """End of stream: whatever is held never became a marker."""
        # 【译】流结束：仍被暂扣的内容最终都没有成为标记。
        rest = self._scanner.flush() if self._scanner is not None else ""
        self._scanner = None
        return [(self._field, rest)] if rest else []


# 每个请求一个：由api_server.reasoning_channel()（atomesh里是对应的方法）构造，交给chat / messages / responses三个接口的处理代码
@dataclass(frozen=True)
class ReasoningChannel:
    """How to read one response's reasoning channel, decided before it starts.

    Two facts, and they have to travel together: which dialect the model
    speaks, and whether its output begins inside the channel already. Passed
    separately, they were passed differently -- the streaming filter got the
    second and never the first, and answered with the union of every dialect's
    markers instead. This is one object with one accessor per delivery mode,
    so the two cannot be handed different answers.

    ``starts_open`` is not simply "this template opens the channel": a request
    that turns thinking off renders a prompt that does not, and OR-ing the
    model-level fact in regardless made an ordinary answer come back as
    ``reasoning_content`` with empty ``content``. The caller resolves it.
    """

    # 【译】如何读取一个响应的推理通道，在响应开始前就决定好。
    #
    #   两个事实，必须一起传递：模型说哪种方言，以及它的输出是否一开始就已在通道内。分开传时它们被传得不一致——
    #   流式过滤器拿到了第二个、从没拿到第一个，于是改用所有方言标记的并集来判断。现在它是一个对象，
    #   每种交付模式一个访问方法，两者就不可能拿到不同的答案。
    #
    #   ``starts_open``并不简单等于"这个模板会打开通道"：一个关闭了思考的请求渲染出的prompt不会打开通道，
    #   而不管怎样都把模型级的事实OR进去，曾让普通回答以``reasoning_content``返回、``content``为空。由调用方决定它。

    dialect: ReasoningDialect | None = None
    starts_open: bool = False

    def split(self, text: str) -> tuple[str | None, str]:
        """A complete output as ``(reasoning, content)``."""
        # 【译】把一个完整输出切成``(reasoning, content)``。
        return separate_reasoning(text, self.starts_open, self.dialect)

    def stream(self) -> ReasoningFilter:
        """A filter over the same output arriving in chunks."""
        # 【译】对同一个输出按chunk到达时使用的过滤器。
        return ReasoningFilter(starts_thinking=self.starts_open, dialect=self.dialect)


# For a caller with no model behind it -- tests, and the completions endpoint,
# which has no chat template and therefore no reasoning channel to read.
# 【译】给背后没有模型的调用方用——测试，以及completions接口：它没有chat template，因此也没有推理通道可读。
# 解释：serving_completion.py并不引用它；实际是serving_chat.py各函数和atomesh的reasoning参数以它作默认值
NO_REASONING = ReasoningChannel()
