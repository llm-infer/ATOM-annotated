# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Format detection.

Several formats share tags and are only told apart by a discriminator that a
later entry would also match, so detection order is load-bearing. `_DETECT_ORDER`
is the single place that ordering is expressed — do not reorder without
re-reading the notes on each entry.

It is consulted on a *rendered chat template*, at startup, and nowhere else.
Running it on a model's output as well was the same question asked of worse
evidence: a template states the format it taught the model, while an output
only exhibits whatever it happens to contain, so an ordinary answer quoting
`<|tool_calls_section_begin|>` was read as a Kimi section and everything after
it deleted — on the non-streaming path only, because streaming had already
been given the startup answer. One question, one place to ask it.
"""

# 【译】格式识别。
#
#   好几种格式共用标签，只能靠一个判别符区分，而这个判别符排在后面的条目也会匹配，所以识别顺序是承重的。
#   `_DETECT_ORDER` 是表达这个顺序的唯一位置——没有重读每个条目的说明之前，不要调整顺序。
#
#   它只在启动时、在*渲染出的chat模板*上被查询，别处都不用。在模型输出上也跑一遍，等于拿更差的证据问同一个问题：
#   模板陈述的是它教给模型的格式，而输出只展示它碰巧包含的东西，于是一个引用了 `<|tool_calls_section_begin|>`
#   的普通回答被读成了一个 Kimi section，其后的一切都被删掉——而且只发生在非流式路径上，因为流式路径早已拿到了
#   启动时的答案。一个问题，只在一个地方问。
#
# 【导读】工具调用格式的注册表：决定"这个模型的工具调用用哪种格式"，并提供非流式的整段解析入口
# 本文件提供：
#   _DETECT_ORDER / PARSERS_BY_NAME     全部已注册格式（新增格式只需在 _DETECT_ORDER 加一项）
#   resolve_tool_call_parser            启动时只调用一次：--tool-call-parser 显式指定，或用 "auto" 从渲染出的chat模板识别
#   validate_tool_call_parser           只校验 --tool-call-parser 的名字（不需要tokenizer，可在加载权重之前报错）
#   parse_tool_calls                    非流式：完整输出 -> (content, [ToolCall])，内部就是 stream.read_whole
#   forbids_tool_calls                  请求的 tool_choice 是否禁止调用工具（OpenAI 的 "none" / Anthropic 的 {"type": "none"}）
# 调用方：api_server.py 的 main() 调用 resolve_tool_call_parser，结果作为全局的解析器类传给各接口；
#   serving_chat.py 调用 parse_tool_calls / forbids_tool_calls；atomesh/server.py 调用 validate_tool_call_parser，
#   atomesh/atom_standalone_service.py 调用 resolve_tool_call_parser
# 与主干的关系：只在服务层启动时和响应后处理时使用，与EngineCore进程无关

import logging
from typing import Any

from ..chat_encoders import render_probe_prompt
from .deepseekv4_tool_parser import DsmlParser
from .deepseekv41_tool_parser import DsmlV41Parser
from .glm_tool_parser import GlmParser
from .kimi_k3_tool_parser import KimiK3Parser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
from .stream import read_whole
from .tool_parser import ToolCall, ToolCallParser

# Checked in order on a COMPLETE output. Kimi (K2) is not listed: it is the
# terminal fallback, because its parse() also defines the "no tool calls at all"
# result.
#
#   K3 first             — its `<|open|>...<|sep|>` channel tokens are disjoint
#                          from every other format's tags, so it never collides;
#                          it also strips its own channel framing from plain
#                          answers (which the terminal K2 fallback would not).
#   MiniMax before DSML — both use `<invoke name=..>`; MiniMax additionally
#                         prefixes every tag with the ns_token.
#   GLM before Qwen     — both use `<tool_call>`; GLM never emits `<function=`,
#                         which GlmParser.detect checks for explicitly.
# 【译】在一份完整输出上按顺序检查。Kimi（K2）不在列表里：它是最终的兜底，因为它的 parse() 也定义了"完全没有工具调用"的结果。
#
#   K3 最先          —— 它的 `<|open|>...<|sep|>` 频道token与其他任何格式的标签都不相交，所以从不冲突；
#                        它还会从普通回答中剥掉自己的频道框架（最后兜底的 K2 不会这样做）。
#   MiniMax 在 DSML 之前 —— 两者都用 `<invoke name=..>`；MiniMax 还会在每个标签前加 ns_token 前缀。
#   GLM 在 Qwen 之前 —— 两者都用 `<tool_call>`；GLM 从不输出 `<function=`，GlmParser.detect 会显式检查这一点。
# 解释：这段注释已有部分过时——按模块docstring，现在这个顺序只用于启动时在渲染出的chat模板上识别格式
#   （resolve_from_prompt，K2 也被追加在最后参与识别），不再在完整输出上检查；列表中的 DsmlV41Parser 注释没有提到，
#   它与 DsmlParser 靠各自的 DETECT_MARKERS 区分（V4.1 的标记是 "｜DSML｜ " 后带空格）
#   GlmParser.detect 识别的依据是 "<arg_key>"，见 glm_tool_parser.py
_DETECT_ORDER: tuple[type[ToolCallParser], ...] = (
    KimiK3Parser,
    MiniMaxParser,
    DsmlParser,
    DsmlV41Parser,
    GlmParser,
    QwenXmlParser,
)


# 非流式入口：解析一份完整输出，返回 (去掉工具调用区域和框架标记后的content, [ToolCall])
#   调用方：serving_chat.py 的非流式chat路径
def parse_tool_calls(
    text: str,
    tools: list | None = None,
    parser_cls: "type[ToolCallParser] | None" = None,
    *,
    suppress_calls: bool = False,
) -> tuple[str, list[ToolCall]]:
    """Parse tool calls from a complete model output.

    Args:
        text: Raw model output that may contain tool calls.
        tools: Optional request tool definitions; used to type-coerce parameter
            values to their declared JSON-Schema types.
        parser_cls: The format resolved for this model at startup, or ``None``
            for "do not parse". ``None`` is not "work it out from the text":
            the streaming path is handed the same ``None`` and emits
            everything as content, so guessing here would answer the same
            request two ways — measured, an answer describing
            `<|tool_calls_section_begin|>` lost 30 characters when
            `stream=false` and arrived whole when `stream=true`.
        suppress_calls: ``tool_choice: "none"``. See the note above
            :func:`forbids_tool_calls` for why this is a flag and not
            "use no parser".

    Returns:
        Tuple of (content_text, list_of_tool_calls). ``content_text`` has the
        tool-call regions and this format's declared framing removed, and is
        ``text`` byte-for-byte when there is neither.

    This is :func:`~.stream.read_whole` — the streaming engine over one chunk —
    and deliberately not a second implementation of it.
    """
    # 【译】从一份完整的模型输出中解析工具调用。
    #
    #   参数：
    #       text：可能含有工具调用的原始模型输出。
    #       tools：可选的请求工具定义；用于把参数值强制转换成它们声明的JSON-Schema类型。
    #       parser_cls：启动时为这个模型解析出的格式，``None`` 表示"不解析"。``None`` 不等于"从文本里推断"：
    #           流式路径拿到的也是同一个 ``None``，并把一切都作为content输出，所以在这里猜测会让同一个请求有两种答案——
    #           实测：一个描述 `<|tool_calls_section_begin|>` 的回答在 `stream=false` 时丢了30个字符，在 `stream=true` 时却完整到达。
    #       suppress_calls：即 ``tool_choice: "none"``。为什么这是一个标志而不是"不用解析器"，见 :func:`forbids_tool_calls` 上方的说明。
    #
    #   返回：
    #       (content_text, list_of_tool_calls) 二元组。``content_text`` 已去掉工具调用区域和本格式声明的框架标记；
    #       两者都没有时，它与 ``text`` 逐字节相同。
    #
    #   它就是 :func:`~.stream.read_whole`——流式引擎在单个chunk上运行——刻意不做第二份实现。
    return read_whole(parser_cls, text, tools, suppress_calls=suppress_calls)


# -- format resolution -----------------------------------------------------
# 【译】-- 格式解析（启动时决定用哪个解析器类）--

# Every format by the name `--tool-call-parser` takes. Derived from the same
# order, so a newly registered format is selectable without a second edit.
# 【译】`--tool-call-parser` 接受的名字到每种格式的映射。由同一个顺序推导，所以新注册的格式不需要第二处修改就能被选择。
# 解释：名字即各类的 NAME：kimi_k3 / minimax / dsml / dsml_v41 / glm / qwen / kimi
PARSERS_BY_NAME: dict[str, type[ToolCallParser]] = {
    p.NAME: p for p in (*_DETECT_ORDER, KimiParser)
}


# 在渲染出的chat模板（带tools）上按 _DETECT_ORDER + KimiParser 的顺序，返回第一个 detect 命中的格式，都不命中返回None
def resolve_from_prompt(rendered_prompt: str) -> type[ToolCallParser] | None:
    """Which format this model will emit, decided before it emits anything.

    A chat template rendered with a tools payload *is* the model's instructions
    for how to call one, so the same cascade `parse_tool_calls` runs on a
    complete output answers the question on the prompt instead -- earlier, and
    without depending on what the model happens to produce first.

    Asked once at startup. `None` means no registered format recognised the
    prompt, which is the honest answer for a model with no tool syntax ATOM
    knows; the caller says so out loud rather than falling back to guessing,
    because a guess here is a tool call fabricated out of ordinary text.

    This replaces `sniff_stream`, which decided from a *prefix* of the output.
    That was strictly harder -- a format's discriminator may not have arrived
    yet, so the answer needed a "cannot tell yet" state, and that state was
    read as "and therefore send nothing", which is how one '<' in an answer
    withheld the rest of the stream.
    """
    # 【译】这个模型会输出哪种格式——在它输出任何东西之前就决定。
    #
    #   一个用tools载荷渲染出来的chat模板*本身*就是教模型如何调用工具的说明，所以 `parse_tool_calls` 在完整输出上跑的
    #   那串级联检查，可以改在prompt上回答同一个问题——更早，并且不依赖模型碰巧先输出了什么。
    #
    #   启动时只问一次。`None` 表示没有任何已注册格式认出这个prompt，对于一个工具语法不为ATOM所知的模型，这是诚实的回答；
    #   调用方会明确地把这一点说出来，而不是退回去猜，因为在这里猜，就是从普通文本里捏造出一个工具调用。
    #
    #   它取代了 `sniff_stream`，后者根据输出的*前缀*来判断。那严格来说更难——格式的判别符可能还没到达，
    #   所以答案需要一个"暂时还分辨不出"的状态，而这个状态又被理解成"因此什么都不发"，
    #   于是回答中的一个 '<' 就扣住了整个流的剩余部分。
    # Kimi is included here though `_DETECT_ORDER` omits it. Its `parse` also
    # defines "no tool calls at all", which is why it is not an entry in an
    # ordering meant to tell formats apart -- but a K2 prompt does carry K2's
    # section tokens, so on a prompt it is a real answer. Omitting it meant a
    # K2 deployment resolved to nothing and delivered its tool calls as raw
    # section tokens in `delta.content`.
    # 【译】Kimi 被包含在这里，尽管 `_DETECT_ORDER` 没有它。它的 `parse` 也定义了"完全没有工具调用"，所以它不是这个
    #   用来区分格式的顺序中的一项——但 K2 的prompt确实带着 K2 的section token，所以在prompt上它是一个真实的答案。
    #   漏掉它会让 K2 部署解析不到任何格式，并把它的工具调用作为原始section token放在 `delta.content` 里发出去。
    # 解释：现在各格式都已没有 parse() 方法（只有 parse_region），"它的 parse 定义了'没有工具调用'"是历史说法；
    #   KimiParser 排在最后参与识别
    for parser in (*_DETECT_ORDER, KimiParser):
        if parser.detect(rendered_prompt):
            return parser
    return None


logger = logging.getLogger("atom")

AUTO = "auto"

# 解释：--tool-call-parser 的help文本常量，api_server.py 和 atomesh/server.py 共用
# Written once and used by both entrypoints, which is also the shape of the
# bug it replaces: the flag existed on the OpenAI server and not on atomesh,
# so the documented escape hatch did not exist there.
# 【译】只写一次、两个入口共用，这也正是它所取代的bug的形状：这个参数原来只在OpenAI server上有、atomesh上没有，
#   所以文档里写的那个应急开关在那边并不存在。
# 解释：help文本译文：工具调用的线上格式。'auto'（默认）在启动时从模型的chat模板读出；显式给出名字则覆盖它。
#   两者都解析不出时，工具调用作为普通文本交付，启动日志会说明这一点——绝不会从输出中猜测格式。
TOOL_CALL_PARSER_HELP = (
    "Tool-call wire format. 'auto' (default) reads it off the model's chat "
    "template at startup; an explicit name overrides that. When neither "
    "resolves, tool calls are delivered as plain text and the startup log "
    "says so \u2014 the format is never guessed from output."
)


# 请求是否禁止模型调用工具（tool_choice 为 "none"）；为真时引擎仍解析区域，但丢弃解析出的调用（suppress_calls）
#   调用方：serving_chat.py、api_server.py（/v1/messages、/v1/responses）、atomesh 的 standalone 服务
def forbids_tool_calls(tool_choice: Any) -> bool:
    """Did this request say the model may not call a tool?

    Both protocols' spellings, because both endpoints ask. OpenAI sends the
    bare string ``"none"``; Anthropic sends ``{"type": "none"}``. Anything
    else -- absent, ``auto``, ``required``, ``any``, a named tool -- is not a
    prohibition, and in particular is not answered here.
    """
    # 【译】这个请求是否说了模型不能调用工具？
    #
    #   两种协议的写法都认，因为两个接口都会问。OpenAI 发送裸字符串 ``"none"``；Anthropic 发送 ``{"type": "none"}``。
    #   其他任何值——缺省、``auto``、``required``、``any``、指定某个工具——都不是禁止，尤其不在这里回答。
    if isinstance(tool_choice, str):
        return tool_choice == "none"
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "none"
    return False


"""Why ``tool_choice: "none"`` is a flag and not "use no parser".

`tool_choice: "none"` used to be enforced where the events were *sent*, twelve
places across two endpoints, while the parser went on consuming the region. So
the model's own words were deleted rather than its call suppressed: a
95-character answer reached the client as its first six, with no event and
`finish_reason: stop`.

The fix for that dropped the parser instead -- and dropping the parser also
drops everything else a parser does. A format whose framing wraps *every*
answer then leaks that framing: Kimi-K3's `Hello there.` arrived as
`<|open|>response<|sep|>Hello there.<|close|>response<|sep|><|end_of_msg|>`
the moment a request said `none`. The format still has to be read; what is
suppressed is dispatch. `ToolCallStreamParser(suppress_calls=True)` and
:func:`parse_tool_calls(..., suppress_calls=True)` read the region exactly as
a permitted one and drop only the calls, so the answer around a forbidden call
survives and the call's own markup does not. Releasing those bytes instead was
tried first and is what leaked the wire format: they are not prose, whatever
the request said about dispatch.
"""
# 【译】为什么 ``tool_choice: "none"`` 是一个标志，而不是"不用解析器"。
#
#   `tool_choice: "none"` 过去是在事件*发送*的地方强制执行的，分布在两个接口的十二处，而解析器照样消费区域。
#   于是被删掉的是模型自己说的话，而不是被抑制的调用：一个95个字符的回答到达客户端时只剩前六个字符，
#   没有任何事件，`finish_reason: stop`。
#
#   针对这个问题的修复改成了直接不用解析器——而不用解析器也就丢掉了解析器做的其他一切。框架标记包裹*每一个*
#   回答的格式就会泄漏这些框架：一旦请求说了 `none`，Kimi-K3 的 `Hello there.` 到达时就变成了
#   `<|open|>response<|sep|>Hello there.<|close|>response<|sep|><|end_of_msg|>`。格式仍然必须被读取；
#   要抑制的是分发。`ToolCallStreamParser(suppress_calls=True)` 和 :func:`parse_tool_calls(..., suppress_calls=True)`
#   像读一个被允许的区域那样原样读取它，只丢弃调用，所以被禁止的调用周围的回答得以保留，而调用自己的标记不会保留。
#   改为原样释放这些字节是最先尝试的做法，正是它泄漏了线上格式：无论请求对分发说了什么，这些字节都不是普通文字。


# 校验 --tool-call-parser 的名字：返回对应的解析器类；"auto"/空返回None（表示去读模板）；未知名字抛ValueError
def validate_tool_call_parser(override: str | None) -> type[ToolCallParser] | None:
    """The format `override` names, or ``None`` for "read the template".

    Separate from :func:`resolve_tool_call_parser` because the name can be
    checked before a tokenizer exists, and therefore before the weights load:
    atomesh resolved inside the service constructor, so a typo was reported
    only after a full model load.
    """
    # 【译】`override` 指定的格式，``None`` 表示"去读模板"。
    #
    #   与 :func:`resolve_tool_call_parser` 分开，是因为名字可以在tokenizer存在之前、也就是加载权重之前就检查：
    #   atomesh 过去在服务的构造函数里才解析，所以一个拼写错误要等完整加载完模型之后才会报告。
    if not override or override == AUTO:
        return None
    parser = PARSERS_BY_NAME.get(override)
    if parser is None:
        raise ValueError(
            f"--tool-call-parser={override!r} is not a known format; "
            f"choose one of {sorted(PARSERS_BY_NAME)} or {AUTO!r}"
        )
    return parser


# 启动时决定本模型的工具调用格式：显式名字优先；否则用 render_probe_prompt（chat_encoders.py）渲染一轮带tools的对话，
#   再交给 resolve_from_prompt 识别；返回None表示不解析工具调用（全部作为普通文本输出）
def resolve_tool_call_parser(
    override: str | None,
    tokenizer: Any,
    custom_encoder: Any = None,
    *,
    model: str = "",
) -> type[ToolCallParser] | None:
    """The format this model's tool calls will arrive in, or ``None``.

    ``override`` is ``--tool-call-parser``: a format name, or ``"auto"``/``None``
    to read it off the chat template. An unknown name raises rather than
    falling back, because a typo that silently disables tool parsing is the
    failure this whole path exists to stop.

    ``None`` is a real answer, not an error: gpt-oss and DeepSeek-R1 render no
    tool syntax ATOM knows, and for them parsing nothing is correct. It is
    logged either way.

    Rendering is best-effort — a template can raise on a tools payload it does
    not accept — and a failure to render is reported and treated as
    unrecognised, never as a reason to fall back to reading the output.
    """
    # 【译】这个模型的工具调用会以哪种格式到达，或者 ``None``。
    #
    #   ``override`` 就是 ``--tool-call-parser``：一个格式名，或 ``"auto"``/``None`` 表示从chat模板读出。
    #   未知名字会抛异常而不是退回默认，因为一个悄悄关掉工具调用解析的拼写错误，正是这整条路径要阻止的故障。
    #
    #   ``None`` 是一个真实的答案，不是错误：gpt-oss 和 DeepSeek-R1 渲染出的工具语法都不为ATOM所知，
    #   对它们来说什么也不解析才是正确的。无论哪种情况都会记日志。
    #
    #   渲染是尽力而为的——模板可能因为不接受某个tools载荷而抛异常——渲染失败会被报告并当作"未识别"，
    #   绝不会成为退回去读输出的理由。
    parser = validate_tool_call_parser(override)
    if parser is not None:
        logger.info(f"Tool-call format: {parser.NAME} (from --tool-call-parser)")
        return parser

    rendered = render_probe_prompt(tokenizer, custom_encoder, tools=True)
    if rendered is None:
        logger.warning(
            f"Could not render {model or 'the model'}'s chat template with a tools "
            f"payload, so its tool-call format is unknown and tool calls will be "
            f"delivered as plain text. Pass --tool-call-parser to set it."
        )
        return None

    parser = resolve_from_prompt(rendered)
    if parser is None:
        logger.info(
            f"No known tool-call format in {model or 'the model'}'s chat template; "
            f"tool calls will be delivered as plain text. Pass --tool-call-parser "
            f"if this model does emit one."
        )
    else:
        logger.info(f"Tool-call format: {parser.NAME} (from the chat template)")
    return parser
