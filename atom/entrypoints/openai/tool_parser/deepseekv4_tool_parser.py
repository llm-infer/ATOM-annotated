# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 DSML tool-call format::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="NAME">
    <｜DSML｜parameter name="PNAME" string="true|false">VALUE</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

``string="true"`` -> value is a raw string; ``string="false"`` -> value is JSON.
DeepSeek-V4-Flash occasionally malforms this (singular ``tool_call``, a missing
``invoke`` wrapper, or params without ``string=``); the parser recovers those
best-effort: it infers a dropped tool name from the parameter signature vs the
request's ``tools`` and infers a missing value type from the schema / JSON.

Inferring a name has two conditions, both load-bearing: only at end of region,
since an ``<invoke>`` still arriving names the tool outright, and only when
some declared tool shares a parameter.
"""

# 【译】DeepSeek-V4 的 DSML 工具调用格式：
#       <｜DSML｜tool_calls>                                                       （外层包裹）
#       <｜DSML｜invoke name="NAME">                                               （调用开始）
#       <｜DSML｜parameter name="PNAME" string="true|false">VALUE</｜DSML｜parameter>   （参数）
#       ...
#       </｜DSML｜invoke>                                                          （调用结束）
#       </｜DSML｜tool_calls>                                                      （包裹结束）
#   ``string="true"`` -> 值是原始字符串；``string="false"`` -> 值是JSON。
#   DeepSeek-V4-Flash 偶尔会写出畸形的格式（单数的 ``tool_call``、缺少 ``invoke`` 包裹、或参数没有 ``string=``）；
#   解析器尽力恢复这些情况：根据参数签名与请求 ``tools`` 的对比推断被丢掉的工具名，并根据schema / JSON 推断缺失的值类型。
#
#   推断名字有两个条件，两个都是承重的：只在区域结尾处推断，因为一个仍在到达中的 ``<invoke>`` 会直接说出工具名；
#   并且只在某个声明的工具与之共享至少一个参数时推断。
#
# 【导读】DeepSeek-V4 DSML 工具调用格式的解析器（NAME="dsml"）；V4.1 的 DsmlV41Parser 继承它（见 deepseekv41_tool_parser.py）
# 实现方式：
#   _compile_patterns 按给定的标记拼写生成参数正则和 invoke 正则，｜DSML｜ 标记在每处都作为可选项匹配（模型常常漏写它）
#   parse_region 两条路径：
#     1. 区域内有 <invoke ...>（已闭合 / 自闭合 / 未闭合）：逐个解析，参数按 string 属性或 schema 还原类型，
#        没有 <parameter> 时尝试把主体当作直接JSON，再剥掉多余的 {"arguments": ...} 外壳
#     2. 没有任何 <invoke>、且只在 at_end 时：V4-Flash 漏写包裹的畸形输出，用 _infer_name 按参数签名推断工具名
# 调用方：stream.py 的 ToolCallStreamParser
# 与主干的关系：服务层协议适配。【分支】DeepSeek-V4 / V4.1 模型实现，见第8批

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_param_value
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

# DSML 标记（注意其中的竖线是全角字符"｜"）
_DSML = "｜DSML｜"


# The model often DROPS the ``｜DSML｜`` marker and emits bare
# ``<invoke name=...>``/``<parameter ...>``/``<tool_calls>`` tags, so the marker
# is matched OPTIONALLY everywhere.
# 【译】模型经常*丢掉* ``｜DSML｜`` 标记，输出裸的 ``<invoke name=...>``/``<parameter ...>``/``<tool_calls>`` 标签，
#   所以这个标记在每处都作为*可选项*匹配。
# 按一种标签拼写（marker + section）编译参数正则和 invoke 正则；DsmlParser 与 DsmlV41Parser 各调用一次
def _compile_patterns(marker: str, section: str) -> tuple[re.Pattern, re.Pattern]:
    """Compile the shared DSML recovery grammar for one tag spelling."""
    # 【译】为一种标签拼写编译共享的 DSML 恢复语法。
    optional = r"(?:" + re.escape(marker) + r")?"
    param_re = re.compile(
        r"<" + optional + r'parameter\s+name="(.*?)"(?:\s+string="(true|false)")?\s*>'
        # The tool_calls? lookahead is the fifth terminator `parse_region` names:
        # a value ends where the next call opens, or it swallows that call's
        # wrapper as data.
        # 【译】tool_calls? 前瞻是 `parse_region` 说的第五个终止条件：值在下一个调用开始的地方结束，否则它会把那个调用的包裹当作数据吞掉。
        # 解释：param_re 的三个捕获组——group(1) 参数名、group(2) string 属性（可能为None）、group(3) 值
        r"(.*?)(?:</" + optional + r"parameter>"
        r"|(?=<" + optional + r"parameter\s)"
        r"|(?=</" + optional + r"invoke>)"
        r"|(?=<" + optional + section + r">)"
        r"|\Z)",
        re.DOTALL,
    )
    # Long-form `<invoke name="x">...</invoke>` OR self-closing `<invoke name="x"/>`
    # (the zero-arg shape; group(2) is None for self-closing). Matches SGLang's V4
    # detector, which accepts both.
    # A call's body may not contain another opener -- that literal is what opens
    # one. Without the guard the non-greedy body ran from a *quoted* opener in
    # prose all the way to the real call's closer, so an answer explaining
    # "you write <invoke name="NAME">" before making a real call produced one call named after the
    # placeholder, carrying the real call's arguments, with the sentence deleted.
    # `finditer` then resumed past the real call, so the call the model actually
    # made never went out. GLM was given this guard first; this is the sweep.
    # 【译】长格式 `<invoke name="x">...</invoke>` 或自闭合的 `<invoke name="x"/>`（零参数的形状；自闭合时 group(2) 为 None）。
    #   与 SGLang 的 V4 检测器一致，它两种都接受。
    #   一个调用的主体里不能再包含另一个开始标记——那个字面量正是用来打开调用的。没有这个保护，非贪婪的主体会从普通文字里
    #   *引用*的开始标记一直延伸到真实调用的闭合标记，于是一个先解释"你写 <invoke name="NAME">"再真正调用的回答，会产生一个
    #   以占位符命名、带着真实调用参数的调用，而那句话被删掉。随后 `finditer` 从真实调用之后继续，所以模型真正发出的那个调用
    #   根本没有发出去。GLM 最先加上了这个保护；这里是把它推广到其他格式。
    not_nested = r"(?:(?!<" + optional + r"invoke\s).)"
    # `closed | self-closing | unclosed`, in one pattern. Recovering truncation
    # in an `else:` under `if invokes:` instead meant no cut-off call could be
    # recovered once any complete invoke existed in the region -- the ordinary
    # `max_tokens` shape -- and it was not monotone in arrived bytes, which the
    # early announcement (`parse_region` over a prefix) requires.
    # 【译】在一个模式里同时写 `已闭合 | 自闭合 | 未闭合`。过去在 `if invokes:` 下面的 `else:` 里恢复截断，意味着只要区域里存在任何
    #   一个完整的invoke（这正是普通的 `max_tokens` 形状），被截断的调用就无法恢复——而且它对到达的字节数不单调，
    #   而提前播报（在前缀上运行 `parse_region`）要求单调。
    # 解释：invoke_re 的捕获组——group(1)/group(2) 为已闭合或自闭合调用的名字/主体，group(3)/group(4) 为未闭合调用的名字/主体
    invoke_re = re.compile(
        r"<"
        + optional
        + r'invoke\s+name="([^"]*)"\s*(?:/>|>('
        + not_nested
        + r"*?)</"
        + optional
        + r"invoke>)"
        + r"|<"
        + optional
        + r'invoke\s+name="([^"]*)"\s*>('
        + not_nested
        + r"*)",
        re.DOTALL,
    )
    return param_re, invoke_re


# 剥掉模型多包的 {"arguments": {...}} / {"input": {...}} 外壳（最多4层），除非该工具真的声明了名为 arguments/input 的参数
def _unwrap_wrapper_args(args: Any, allowed: set) -> Any:
    """Strip spurious ``{"arguments": {...}}`` / ``{"input": {...}}`` envelopes.

    Non-tuned models (DeepSeek-V4-Pro) frequently wrap the real args in an extra
    ``arguments``/``input`` object — sometimes nested 2-3 deep, or stringified —
    so a call meant as ``{"cmd": "ls"}`` arrives as ``{"arguments": {"cmd":
    "ls"}}``. Recursively unwrap while the sole key is a wrapper that is NOT
    itself a declared param of the tool. Mirrors vLLM's ``_unwrap_wrapper_args``
    (deepseek_v4.py)."""
    # 【译】剥掉多余的 ``{"arguments": {...}}`` / ``{"input": {...}}`` 外壳。
    #
    #   未经调优的模型（DeepSeek-V4-Pro）经常把真正的参数再包进一个额外的 ``arguments``/``input`` 对象——有时嵌套2~3层，
    #   或者被转成字符串——所以一个本意是 ``{"cmd": "ls"}`` 的调用到达时变成 ``{"arguments": {"cmd": "ls"}}``。
    #   只要唯一的键是一个外壳键、且它本身不是该工具声明的参数，就递归地剥开。对应 vLLM 的 ``_unwrap_wrapper_args``（deepseek_v4.py）。
    # 【译】（行尾注释）设上限，防止病态的深层嵌套
    for _ in range(4):  # bounded against pathological nesting
        if not (isinstance(args, dict) and len(args) == 1):
            break
        ((k, v),) = args.items()
        if k not in ("arguments", "input"):
            break
        if allowed and k in allowed:
            # 【译】（行尾注释）这个工具真的有一个名为 arguments/input 的参数
            break  # this tool really has a param named arguments/input
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (ValueError, TypeError):
                break
        if not isinstance(v, dict):
            break
        args = v
    return args


# 解码一个 <parameter> 的值：string="true" 原样返回，"false" 按JSON解析；没有该属性时用schema类型，否则尝试JSON
def _coerce(value: str, string_attr: str | None, ptype: Any) -> Any:
    """Decode one ``<parameter>`` body.

    Deliberately not :func:`~.schema.coerce_json_or_raw`: on a JSON-decode miss
    this falls back to ``value.strip()`` where that one falls back to
    ``value.strip("\\n")``, which differs for values with surrounding spaces.
    """
    # 【译】解码一个 ``<parameter>`` 的主体。
    #
    #   刻意不用 :func:`~.schema.coerce_json_or_raw`：JSON解码失败时，这里退回到 ``value.strip()``，而那个函数退回到
    #   ``value.strip("\\n")``，对首尾带空格的值两者结果不同。
    if string_attr == "true":
        return value
    if string_attr == "false":
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    # attr absent -> use declared schema type if known, else infer via JSON.
    # 【译】没有该属性 -> 已知声明的schema类型就用它，否则通过JSON推断。
    if ptype is not None:
        return coerce_param_value(value, ptype)
    v = value.strip()
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


# 在请求声明的工具中，找参数集合与 arg_names 最匹配的那个（至少共享一个参数），用于 V4-Flash 漏写 invoke 时推断工具名
def _infer_name(arg_names: set, param_types: dict[str, dict[str, Any]]) -> str | None:
    """The request tool whose parameters best match ``arg_names``, if one does.

    At least one shared parameter. Every score is negative once the sets are
    disjoint and the running best started below all of them, so the first
    declared tool with any properties won on no evidence: a region of bare
    ``<parameter name="zzz">`` dispatched ``get_weather``. This branch has no
    name on the wire, so nothing shared means it was prose.
    """
    # 【译】参数与 ``arg_names`` 最匹配的请求工具，如果有的话。
    #
    #   至少共享一个参数。一旦两个集合不相交，每个得分都是负数，而当前最佳值的初值低于所有得分，所以第一个有任何属性的声明工具
    #   会在毫无证据的情况下胜出：一个只有裸 ``<parameter name="zzz">`` 的区域分发了 ``get_weather``。这个分支在线上没有名字，
    #   所以没有任何共享参数就意味着它是普通文字。
    best, best_score = None, float("-inf")
    for name, props in param_types.items():
        shared = len(arg_names.intersection(props))
        if not shared:
            continue
        # |p ^ a| == |p| + |a| - 2|p & a|, so it never has to be built.
        # 【译】|p ^ a| == |p| + |a| - 2|p & a|，所以永远不需要真的构造对称差集合。
        # 解释：得分 = 共享参数数 - 0.1 x 不共享的参数数，共享越多、多余越少得分越高
        score = shared - 0.1 * (len(props) + len(arg_names) - 2 * shared)
        if score > best_score:
            best_score, best = score, name
    return best


# What may follow an unclosed opener in a real call: another parameter, or the
# close of the invoke the name opened. Either spelling of the marker, which
# the model drops about as often as it writes. One tuple, read by both
# `_is_truncated_call` at both ends of a region's life -- the peek used to
# encode this separately and accepted any tag with a slash in it, `<br/>`
# included.
# 【译】在真实调用中，一个未闭合的开始标记后面可以跟的东西：另一个参数，或者这个名字打开的那个invoke的闭合。标记的两种拼写都算，
#   模型漏写它和写它的频率差不多。只用一个元组，在区域生命周期的两端都由 `_is_truncated_call` 读取——偷看过去单独编码了这一点，
#   接受任何带斜杠的标签，包括 `<br/>`。
_CALL_CONTINUES = (
    "<" + _DSML + "parameter",
    "<parameter",
    "</" + _DSML + "invoke>",
    "</invoke>",
)


# 判断一个未闭合的 <invoke name=...> 是被截断的真实调用（名字已声明，后面紧跟参数或 </invoke>），还是普通文字
def _is_truncated_call(
    name: str, body: str, param_types: dict, tokens: tuple[str, ...], *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`, which this is the DSML
    spelling of; the sweep that added it there and to GLM missed this format,
    and the same sentence -- "you emit `<invoke name="get_weather">` and
    inside it a `<parameter>` line" -- was still being dispatched as a call.
    """
    # 【译】这个未闭合的 `<invoke name=...>` 是一个被截断的调用，还是普通文字？
    #
    #   见 :func:`QwenXmlParser._is_truncated_call`，这是它的 DSML 写法；那次把它加到 Qwen 和 GLM 的统一修复漏掉了这个格式，
    #   于是同一句话——"你输出 `<invoke name="get_weather">` 并在里面写一行 `<parameter>`"——仍被当作调用分发。
    # 解释：_is_truncated_call 实际是 qwen3_tool_parser.py 的模块级函数，不是 QwenXmlParser 的方法
    if not declared_tools_allow(name, param_types):
        return False
    rest = body.lstrip()
    return (not rest and at_end) or continues_a_call(rest, tokens, arrived=not at_end)


# DeepSeek-V4 DSML 格式的解析器类；DSML_PREFIX / SECTION / PARAM_RE / INVOKE_RE / CALL_CONTINUES 是子类（V4.1）可以覆盖的参数
class DsmlParser(ToolCallParser):
    NAME: ClassVar[str] = "dsml"
    DSML_PREFIX: ClassVar[str] = _DSML
    SECTION: ClassVar[str] = "tool_calls"
    PARAM_RE, INVOKE_RE = _compile_patterns(_DSML, r"tool_calls?")
    CALL_CONTINUES: ClassVar[tuple[str, ...]] = _CALL_CONTINUES
    # Region-start markers, both marked and marker-less variants.
    # 【译】区域起始标记，带标记和不带标记的写法都有。
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        # 【译】（行尾注释）带标记（覆盖 tool_call / tool_calls）
        "<" + _DSML + "tool_call",  # marked (covers tool_call / tool_calls)
        # 【译】（行尾注释）带标记的 invoke
        "<" + _DSML + "invoke",  # marked invoke
        # 【译】（行尾注释）不带标记的 invoke（常见的畸形写法）
        "<invoke name=",  # marker-less invoke (common malform)
        # 【译】（行尾注释）不带标记的 section 开始
        "<tool_calls>",  # marker-less section open
        # ...and its singular spelling, which `CALL_OPENERS` already claims as
        # markup. Declaring it here was impossible while this tuple was also
        # the fingerprint -- `<tool_call>` is the Hermes and Qwen opener too,
        # so DSML would have claimed their templates. `DETECT_MARKERS` is that
        # separation, and this is the first thing it buys: the wrapper stops
        # being released as content before the region opens.
        # 【译】……以及它的单数拼写，`CALL_OPENERS` 已经把它认作标记。当这个元组同时也是识别指纹时，不可能在这里声明它——
        #   `<tool_call>` 也是 Hermes 和 Qwen 的开始标记，所以 DSML 会认领它们的模板。`DETECT_MARKERS` 就是那次分离，
        #   而这是它带来的第一个好处：包裹不会在区域打开之前被当作content释放出去。
        "<tool_call>",
    )
    # What makes a text DSML's, which is narrower than what must not be split:
    # the marker-bearing spellings only. The marker-less ones above are
    # malform recovery, and `<invoke name=` is also how MiniMax opens a call
    # -- keying identification on them had DSML claiming MiniMax's templates
    # with nothing but `_DETECT_ORDER` in the way.
    # 【译】让一段文本属于 DSML 的东西，比"不能被切开的东西"更窄：只有带标记的拼写。上面不带标记的那些是畸形恢复，
    #   而 `<invoke name=` 也是 MiniMax 开始一次调用的方式——把识别依据放在它们上面，会让 DSML 认领 MiniMax 的模板，
    #   中间只隔着一个 `_DETECT_ORDER`。
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_call",
        "<" + _DSML + "invoke",
        "<" + _DSML + "parameter",
    )
    # The section wrapper closing after the last invoke -- markup, not answer.
    # Both spellings, since the model drops the marker about as often as it
    # writes it, and both arities, since it writes the singular too.
    # 【译】最后一个invoke之后关闭的section包裹——是标记，不是回答。两种拼写都有，因为模型漏写标记和写它的频率差不多；
    #   单复数也都有，因为它也会写单数。
    # 解释：这段说明同时适用于下面的 CALL_OPENERS（开始包裹）和 CALL_CLOSERS（结束包裹）
    CALL_OPENERS: ClassVar[tuple[str, ...]] = (
        "<" + _DSML + "tool_calls>",
        "<" + _DSML + "tool_call>",
        "<tool_calls>",
        "<tool_call>",
    )
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = (
        "</" + _DSML + "tool_calls>",
        "</" + _DSML + "tool_call>",
        "</tool_calls>",
        "</tool_call>",
    )
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = (
        "</" + _DSML + "invoke>",
        "</invoke>",
    )

    # detect() is inherited: any start marker present means DSML.
    # 【译】detect() 是继承来的：出现任何起始标记就意味着是 DSML。
    # 解释：已过时——基类 detect 在声明了 DETECT_MARKERS 时只看 DETECT_MARKERS（上面三个带标记的拼写），而不是任何起始标记

    # 用本格式写出一次调用（只供测试生成语料；用 DSML_PREFIX / SECTION，子类 V4.1 也适用）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f'<{cls.DSML_PREFIX}parameter name="{k}" string="true">{v}</{cls.DSML_PREFIX}parameter>'
            for k, v in args.items()
        )
        return (
            f'<{cls.DSML_PREFIX}{cls.SECTION}><{cls.DSML_PREFIX}invoke name="{name}">'
            f"{body}</{cls.DSML_PREFIX}invoke></{cls.DSML_PREFIX}{cls.SECTION}>"
        )

    # 解析一个区域：有 <invoke> 时逐个解析；没有 <invoke> 且在区域结尾时，按参数签名推断工具名（V4-Flash 畸形输出）
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        calls: list[tuple[str, dict[str, Any]]] = []
        # A truncated or wrapper-less call runs to the end of what arrived;
        # complete invokes start and end where their own matches do, one span
        # each.
        # 【译】被截断或没有包裹的调用一直延伸到已到达内容的末尾；完整的invoke在它们自己的匹配处开始和结束，每个一个span。
        spans: list[tuple[int, int]] = []
        invokes = list(cls.INVOKE_RE.finditer(region))
        if invokes:
            for m in invokes:
                closed = m.group(1) is not None
                name = m.group(1) if closed else m.group(3)
                # `None` for a self-closing `<invoke .../>`, which is closed
                # and carries no body.
                # 【译】自闭合的 `<invoke .../>` 为 `None`，它是闭合的，且没有主体。
                body = (m.group(2) if closed else m.group(4)) or ""
                # Before the truncation gate, which only runs for an unclosed
                # invoke: `name="([^"]*)"` matches an empty and an all-space
                # name, and a *closed* one had nothing to stop it.
                # 【译】放在截断闸门之前，因为闸门只对未闭合的invoke运行：`name="([^"]*)"` 能匹配空名字和全空格名字，而一个*已闭合*的invoke没有任何东西拦住它。
                if not usable_tool_name(name):
                    continue
                if not closed and not _is_truncated_call(
                    name, body, param_types, cls.CALL_CONTINUES, at_end=at_end
                ):
                    continue
                spans.append(
                    (
                        cls.markup_begin(region, m.start()),
                        cls.markup_end(region, m.end()),
                    )
                )
                types = param_types.get(name, {})
                args: dict[str, Any] = {
                    pm.group(1): _coerce(
                        pm.group(3), pm.group(2), types.get(pm.group(1))
                    )
                    for pm in cls.PARAM_RE.finditer(body)
                }
                # Direct-JSON parameter body (DSML "Format 2", also accepted by
                # vLLM/SGLang): `<invoke name="x"> { "k": "v" } </invoke>` with no
                # <parameter> tags. Falls through here with empty args; recover them.
                # 【译】直接JSON的参数主体（DSML 的"格式2"，vLLM/SGLang 也接受）：`<invoke name="x"> { "k": "v" } </invoke>`，没有 <parameter> 标签。
                #   这种情况走到这里时参数为空；把它们恢复出来。
                if not args:
                    stripped = body.strip()
                    if stripped.startswith("{"):
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                args = parsed
                        except (ValueError, TypeError):
                            pass
                args = _unwrap_wrapper_args(args, set(types))
                calls.append((name, args))
        elif at_end:
            # `elif at_end`, not `else`: an `<invoke>` still on its way moves
            # the region to the loop above and with it which bytes are markup,
            # so this branch's acceptance is not monotone in arrived bytes --
            # which the early announcement assumes. A name inferred half-way
            # through went out ahead of prose the whole parse puts in front of
            # it, so `stream=true` ordered the answer differently. A signature
            # is only complete when the region is. Free: `invokes` is already
            # computed, and the peek now skips this scan.
            #
            # No `<invoke>` at all: the documented V4-Flash malform that drops
            # the wrapper and writes bare `<parameter>` lines. The name has to
            # be inferred from the parameter signature, which is why this is
            # its own branch -- a *truncated* call has its name in the opener
            # and is read by the loop above. Inferring for both scored
            # `get_time` for a cut-off `<invoke name="get_weather">` because
            # it happened to share more parameters, so the parse named a
            # different tool than the announcement had.
            #
            # Only when the parameters *are* the region. Every other branch
            # gates on an opener; this one has none to gate on, since the name
            # is inferred from the signature rather than read, so where the
            # markup starts is the whole of the evidence. Prose in front of it
            # means the model was writing about the syntax -- and no shape
            # built from the front of a call can reach here to say so, which
            # is why it went two rounds unseen while deleting 152 characters
            # of such an answer down to 18 and dispatching a name it had
            # invented.
            # 【译】用 `elif at_end` 而不是 `else`：一个仍在到达途中的 `<invoke>` 会把区域转到上面的循环，连带改变哪些字节是标记，
            #   所以这个分支的接受与否对到达的字节数不单调——而提前播报假设它单调。一个在中途推断出的名字，会排在完整解析放在它前面的文字之前发出，
            #   于是 `stream=true` 给回答排了不同的顺序。签名只有在区域完整时才完整。没有额外开销：`invokes` 已经算过了，而偷看现在会跳过这次扫描。
            #
            #   完全没有 `<invoke>`：这是文档记载的 V4-Flash 畸形写法，丢掉包裹、直接写裸的 `<parameter>` 行。名字只能从参数签名推断，
            #   这就是它自成一个分支的原因——一个*被截断*的调用在开始标记里就有名字，由上面的循环读取。对两者都做推断，曾因为 `get_time`
            #   恰好共享了更多参数，而为一个被截断的 `<invoke name="get_weather">` 选了 `get_time`，于是解析给出的工具与播报的不同。
            #
            #   只在参数*就是*整个区域时才这样做。其他每个分支都以一个开始标记为闸门；这个分支没有可以作闸门的东西，因为名字是从签名推断而不是读出来的，
            #   所以标记从哪里开始就是全部证据。参数前面有普通文字，意味着模型在写关于语法的东西——而任何从调用前部构造出来的形状都到不了这里来说明这一点，
            #   这就是它连续两轮没被发现的原因，期间它把这样一个回答的152个字符删到只剩18个，还分发了一个它编造出来的名字。
            matches = list(cls.PARAM_RE.finditer(region))
            if matches and cls.markup_begin(region, matches[0].start()) == 0:
                raw = {pm.group(1): (pm.group(3), pm.group(2)) for pm in matches}
                # No `or "unknown"`: a signature matching nothing declared is
                # not a call the client could dispatch, and naming it shipped
                # one anyway, with `finish_reason: tool_calls`, for an answer
                # that had merely written the tags.
                # 【译】不加 `or "unknown"`：一个与任何声明都不匹配的签名不是客户端能够分发的调用，而给它起名字，曾为一个仅仅写了这些标签的回答
                #   照样发出了一个调用，还带着 `finish_reason: tool_calls`。
                name = _infer_name(set(raw), param_types)
                if name is not None:
                    # Nothing to anchor to, so the parameters run from wherever
                    # the region began to the end of what arrived.
                    # 【译】没有可以锚定的东西，所以参数从区域开始的地方一直延伸到已到达内容的末尾。
                    spans.append((0, len(region)))
                    types = param_types.get(name, {})
                    args = {k: _coerce(v, s, types.get(k)) for k, (v, s) in raw.items()}
                    args = _unwrap_wrapper_args(args, set(types))
                    calls.append((name, args))

        tool_calls = tuple(
            ToolCall(
                id=unique_tool_call_id(),
                type="function",
                function={
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            )
            for name, args in calls
        )
        return RegionParse(tool_calls, tuple(spans))
