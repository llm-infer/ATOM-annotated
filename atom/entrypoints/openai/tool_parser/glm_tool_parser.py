# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-4.5 / 4.6 / 5.x tool-call format::

    <tool_call>NAME
    <arg_key>K1</arg_key><arg_value>V1</arg_value>
    <arg_key>K2</arg_key><arg_value>V2</arg_value>
    ...</tool_call>

The function name follows the opening tag directly (no ``<function=`` wrapper,
which is how this is told apart from the Qwen3 XML format). GLM's chat template
renders non-string argument values with ``tojson`` and string values raw, so a
value is JSON-decoded when the request schema declares a non-string type (or
when it parses as JSON) and otherwise kept as a raw string.
"""

# 【译】GLM-4.5 / 4.6 / 5.x 的工具调用格式：
#       <tool_call>NAME                                    （开始标签后直接跟工具名）
#       <arg_key>K1</arg_key><arg_value>V1</arg_value>     （每个参数一对键/值标签）
#       <arg_key>K2</arg_key><arg_value>V2</arg_value>     （第二个参数）
#       ...</tool_call>                                    （调用结束）
#   函数名直接跟在开始标签后面（没有 ``<function=`` 包裹，这正是它与 Qwen3 XML 格式的区别）。GLM 的chat模板用 ``tojson``
#   渲染非字符串的参数值，字符串值则原样渲染，所以当请求schema声明了非字符串类型（或者值能按JSON解析）时，值会被JSON解码，
#   否则保留为原始字符串。
#
# 【导读】GLM 系工具调用格式的解析器（NAME="glm"）
#   启动时 registry.py 用 detect 识别：模板中有 "<arg_key>" 且没有 "<function="
# 实现方式：_TOOLCALL_RE 匹配每个 <tool_call>...（已闭合 / 未闭合两个分支），名字取到第一个 <arg_key>（或第一个 '<'）为止，
#   未闭合的调用必须通过"名字已声明 + 后面紧跟 <arg_key>"的检验，否则当作普通文字；参数值用 coerce_json_or_raw 还原类型
#   与其他格式不同，GLM 没有外层包裹：调用本身就是 <tool_call>...</tool_call>，所以只声明 CALL_SELF_CLOSERS
# 调用方：stream.py 的 ToolCallStreamParser
# 与主干的关系：服务层协议适配；GLM-4 MoE 模型见第7批，GLM-5.x 见第8批

import json
import re
from typing import Any, ClassVar

from .qwen3_tool_parser import QWEN_TOOL_PREFIX
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

# 解释：本格式的调用开始标签
TOOL_CALL_OPEN = "<tool_call>"
# A call's body may not contain another `<tool_call>`. Without that the
# non-greedy match ran from the *first* opener in the region to the first
# close, so an answer that quotes the tag and then makes a real call produced
# a "name" of the whole sentence -- rejected by `usable_tool_name`, after which
# `finditer` resumed past the real call and found nothing at all. The tag is
# what this format opens a call with, so it cannot appear inside one.
# 【译】一个调用的主体里不能再包含另一个 `<tool_call>`。没有这一点，非贪婪匹配会从区域中*第一个*开始标记一直延伸到第一个闭合标记，
#   于是一个先引用标签再真正调用的回答，会产生一个以整句话为"名字"的调用——被 `usable_tool_name` 拒绝，之后 `finditer`
#   从真实调用之后继续，什么也没找到。这个标签是本格式打开调用用的，所以它不可能出现在调用内部。
_NOT_NESTED = r"(?:(?!<tool_call>).)"
_TOOLCALL_RE = re.compile(
    r"<tool_call>("
    + _NOT_NESTED
    + r"*?)</tool_call>|<tool_call>("
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
# No `$` on the unclosed alternative. An unclosed call ends where the next one
# opens, not only at end of stream: with the anchor, a call the model forgot
# to close followed by a second call matched neither alternative at the first
# opener, so `finditer` skipped the first call entirely and returned the
# second -- one call where the model made two, and a different one from the
# one the early name had already announced.
# This format needs the name test more than the others do: without it the
# unterminated branch above turns any prose after a `<tool_call>` into a call
# -- an answer explaining that "the model writes <tool_call> followed by the
# name" produced a call named " followed by the name. Hope that helps!", and a
# Hermes-style `<tool_call>{"name": ...}` produced one named after the whole
# JSON object. Both reached clients as `tool_calls`. It is `usable_tool_name`
# now, shared: the same question was being answered six ways, and three
# formats were not asking it at all.
# 【译】未闭合分支上没有 `$`。一个未闭合的调用在下一个调用开始的地方结束，而不只在流结束处：有这个锚点时，一个模型忘了闭合、
#   后面跟着第二个调用的调用，在第一个开始标记处两个分支都匹配不上，于是 `finditer` 完全跳过第一个调用、返回第二个——
#   模型做了两个调用却只有一个，而且和提前播报出去的名字不是同一个。
#   这个格式比其他格式更需要名字检验：没有它，上面的未终止分支会把 `<tool_call>` 后面的任何普通文字都变成一个调用——
#   一个解释"模型写 <tool_call> 然后跟名字"的回答，产生了一个名为 " followed by the name. Hope that helps!" 的调用，
#   而 Hermes 风格的 `<tool_call>{"name": ...}` 则产生了一个以整个JSON对象命名的调用。两者都以 `tool_calls` 到达了客户端。
#   现在它是共享的 `usable_tool_name`：同一个问题曾有六种回答方式，其中三种格式根本没问。
# 解释：这两段注释说的是上面的 _TOOLCALL_RE 和 parse_region 里的 usable_tool_name 检查，不是紧接着的 _ARG_OPENER；
#   _ARG_RE 匹配 <arg_key>键</arg_key><arg_value>值，值在 </arg_value>、下一个 <arg_key>、</tool_call> 或文本结尾处结束
_ARG_OPENER = "<arg_key>"
_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>"
    r"(.*?)(?:</arg_value>|(?=<arg_key>)|(?=</tool_call>)|\Z)",
    re.DOTALL,
)


# GLM 格式的解析器类
class GlmParser(ToolCallParser):
    NAME: ClassVar[str] = "glm"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    # The call's own closer, and there is no wrapper around it, so
    # `CALL_CLOSERS` stays empty. Blank here because the two coincide answered
    # "is there an inner block" with the tuple that says "what ends a call",
    # and left this the one format whose region could never be seen to close.
    # 【译】调用自己的闭合标记，它外面没有包裹，所以 `CALL_CLOSERS` 保持为空。这里曾经是空的，因为把两者当作一回事时，
    #   用表示"什么结束一个调用"的元组回答了"有没有内层块"的问题，结果让它成为唯一一个区域永远无法被看到闭合的格式。
    # 解释：tool_parser.py 的 region_end 只看 CALL_SELF_CLOSERS；它为空时永远返回0，区域只能等到流结束
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)

    # 用本格式写出一次调用（只供测试生成语料）
    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        body = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args.items()
        )
        return f"<tool_call>{name}{body}</tool_call>"

    # 识别规则：模板中有 "<arg_key>" 且没有 "<function="（后者属于 Qwen）
    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the GLM ``<tool_call>...<arg_key>`` format (never Qwen/DSML)."""
        # 【译】识别 GLM 的 ``<tool_call>...<arg_key>`` 格式（绝不是 Qwen/DSML）。
        # 【译】（行尾注释）'<function=' -> 是 Qwen，不是 GLM
        if QWEN_TOOL_PREFIX in text:  # '<function=' -> Qwen, not GLM
            return False
        # `<arg_key>` and not a bare `<tool_call>`: this runs on a rendered
        # chat template, where a Hermes-JSON model shows the same tag and has
        # no `<arg_key>` anywhere. Accepting the tag alone bound every such
        # model to this parser for the process lifetime and logged it as a
        # success -- /mnt/Qwen3-8B resolved to `glm` while /data/Qwen3.5-27B
        # resolved to `qwen`, two members of one family disagreeing.
        # 【译】用 `<arg_key>` 而不是单独的 `<tool_call>`：这里运行在渲染出的chat模板上，Hermes-JSON 模型的模板里有同样的标签，
        #   却没有任何 `<arg_key>`。只凭标签就接受，会把每个这样的模型在整个进程生命周期内绑定到这个解析器，并把它当作成功记进日志——
        #   /mnt/Qwen3-8B 被解析成了 `glm`，而 /data/Qwen3.5-27B 被解析成 `qwen`，同一家族的两个成员结论不一致。
        return "<arg_key>" in text

    # 解析一个区域：逐个匹配 <tool_call>...，取名字和参数；每个调用的 span 从 <tool_call> 开始，用 markup_end 向后扩展
    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _TOOLCALL_RE.finditer(region):
            closed = m.group(1) is not None
            body = m.group(1) if closed else m.group(2)
            if not body:
                continue
            ak = body.find("<arg_key>")
            # The argument opener having arrived *is* the follower evidence,
            # and it is consumed into the name/body split rather than left in
            # `rest` -- so an empty `rest` here means "the follower came and
            # went", not "nothing came". Reading the two the same way made a
            # call whose arguments were still arriving unnameable, which is
            # the one shape announcing early exists for.
            # 【译】参数开始标记已经到达，这件事*本身*就是"后续记号"的证据，而且它被归入名字/主体的切分，而不是留在 `rest` 里——
            #   所以这里 `rest` 为空意味着"后续记号来过又被拿走了"，而不是"什么也没来"。把两者等同看待，会让一个参数仍在到达中的调用
            #   无法被命名，而那正是提前播报所要服务的形状。
            arg_key_arrived = ak != -1
            if closed or arg_key_arrived:
                name, rest = body[:ak].strip() if ak != -1 else body.strip(), ""
            else:
                # Unclosed and no complete `<arg_key>` yet. The name runs to
                # the first `<`, and whatever follows is the test below.
                #
                # Cut on the character and not on `len(name)`: the name is
                # stripped, so a single newline after `<tool_call>` shifted
                # that slice by one and left the name's own last character in
                # `rest`. A `read_file` call truncated after a leading newline
                # failed the test and was shipped to the user as raw markup.
                # 【译】没闭合，也还没有完整的 `<arg_key>`。名字一直到第一个 `<` 为止，后面的内容交给下面的检验。
                #
                #   按字符切，而不是按 `len(name)` 切：名字被strip过，所以 `<tool_call>` 后面的一个换行会让那次切片偏移一位，
                #   把名字自己的最后一个字符留在 `rest` 里。一个在开头换行之后被截断的 `read_file` 调用因此没通过检验，作为原始标记交给了用户。
                lt = body.find("<")
                name = (body if lt == -1 else body[:lt]).strip()
                rest = "" if lt == -1 else body[lt:]
            if not usable_tool_name(name):
                continue
            # See `QwenXmlParser._is_truncated_call`: an unclosed region is a
            # cut-off call or prose quoting the tag, and prose has to fail
            # both tests -- a declared name, and nothing after it but this
            # format's own next token. `<tool_call>get_weather<br>` fails the
            # second; a call cut off inside its own `<arg_key>` passes it, and
            # used to be rejected along with the prose.
            # 【译】见 `QwenXmlParser._is_truncated_call`：一个未闭合的区域要么是被截断的调用，要么是在引用标签的普通文字，
            #   普通文字必须两项检验都不通过——名字已声明，以及名字之后只有本格式自己的下一个记号。`<tool_call>get_weather<br>`
            #   不通过第二项；一个在自己的 `<arg_key>` 内被截断的调用能通过它，而过去它曾和普通文字一起被拒绝。
            # 解释：QwenXmlParser 并没有 _is_truncated_call 方法，它是 qwen3_tool_parser.py 中的模块级函数
            if not closed and not (
                declared_tools_allow(name, param_types)
                and (
                    arg_key_arrived
                    or (not rest and at_end)
                    or continues_a_call(rest, (_ARG_OPENER,), arrived=not at_end)
                )
            ):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _ARG_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
            spans.append((m.start(), cls.markup_end(region, m.end())))
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
