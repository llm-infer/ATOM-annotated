# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's channel tokens, owned in one place.

Two readers strip these: the reasoning stage, which separates the think
channel from the answer, and the tool-call parser, which reads the tools
section. Each used to carry its own list and they drifted apart: tokens only
the parser listed were removed only when a K3 tool parser happened to be
resolved, so a deployment whose template refused the tools probe leaked them
to `content` on both delivery paths. Which literals are framing is a property
of the model's wire format, not of whether tool calling was configured.

The split below is by *owner*, and it is the one thing this module asserts:

- :data:`CHANNEL_FRAMING` wraps every answer whether or not a tool was called,
  so the reasoning stage removes it and the tool parser does too.
- :data:`TOOL_REGION_FRAMING` brackets a call. Those are the tool parser's
  alone: the reasoning stage cannot remove the payload between them -- the
  tool name, the argument keys and their values -- so stripping only the
  brackets would hand the client mangled half-text instead of either a call
  or an honest quotation.
"""

# 【译】Kimi-K3的通道token，集中在一处维护。
#
#   有两个读取方会去掉这些token：推理阶段（把think通道与回答分开）和工具调用解析器（读取tools段）。
#   它们过去各自带一份列表，结果逐渐分歧：只有解析器列出的token，只在碰巧解析出K3工具调用解析器时才会被去掉，
#   于是一个模板拒绝了tools探测的部署，在两条交付路径上都把它们漏进了`content`。哪些字面量属于外框，
#   是模型线上格式的属性，与是否配置了工具调用无关。
#
#   下面按*归属方*划分，这也是本模块唯一断言的事情：
#
#   - :data:`CHANNEL_FRAMING`包在每个回答外面，无论是否调用了工具，所以推理阶段会去掉它，工具调用解析器也会。
#   - :data:`TOOL_REGION_FRAMING`括住一次调用。它们只归工具调用解析器：推理阶段无法去掉它们之间的载荷——
#     工具名、参数键和参数值——所以只去掉括号的话，交给客户端的会是残缺的半截文本，既不是一次调用，也不是如实的引用。
#
# 【导读】Kimi-K3输出格式（通道格式）的token字面量表
# K3把一次输出分成think / response / tools等通道，每段用<|open|>名字<|sep|> ... <|close|>名字<|sep|>包住，大致形如：
#   （prompt末尾已带<|open|>think<|sep|>）推理……<|close|>think<|sep|><|open|>response<|sep|>回答……<|close|>response<|sep|>
#   调用工具时再跟<|open|>tools<|sep|><|open|>call tool="NAME"<|sep|>……<|close|>call<|sep|>……<|close|>tools<|sep|>
# 调用方：reasoning_dialects.py（K3推理方言的结束标记和外框）、tool_parser/kimi_k3_tool_parser.py（tools段解析和外框剥离）
# 与主干的关系：服务层输出后处理用到的纯常量模块；【分支】Kimi-K3模型本身见第8批

THINK_START = "<|open|>think<|sep|>"
THINK_END = "<|close|>think<|sep|>"
RESPONSE_START = "<|open|>response<|sep|>"
RESPONSE_END = "<|close|>response<|sep|>"
MESSAGE_START = "<|open|>message<|sep|>"
MESSAGE_END = "<|close|>message<|sep|>"
END_OF_MSG = "<|end_of_msg|>"
TOOLS_START = "<|open|>tools<|sep|>"


# Wrappers around the answer itself. Paired spellings only; the bare closers
# are in `UNPAIRED_FRAMING` below, for the reason given there.
# 【译】包在回答本身外面的标记。只放成对的拼写；单独的关闭标记在下面的`UNPAIRED_FRAMING`里，原因见那里。
CHANNEL_FRAMING: tuple[str, ...] = (
    RESPONSE_START,
    RESPONSE_END,
    MESSAGE_START,
    MESSAGE_END,
    END_OF_MSG,
)

# The unpaired closers, and the separator on its own. These are framing too,
# but they cannot move into `CHANNEL_FRAMING`, for two independent reasons,
# even though the move looks obviously right.
#
# `<|sep|>` occurs *inside* a tool region (`<|open|>call tool="x"<|sep|>`),
# and the reasoning stage runs first. Removing it there turns a region the
# reasoning stage is only passing through into mangled half-text before the
# parser sees it.
#
# The bare closers are proper prefixes of the paired ones, and `MarkerScanner`
# reports a complete short marker rather than waiting to see whether the long
# one follows (see `_plan`, which documents the gap and the test that holds
# formats inside it). In this parser's list that is harmless -- strip
# `<|close|>think` and the `<|sep|>` it leaves is also framing, so both
# chunkings converge. Add the bare ones to the reasoning dialect *without*
# `<|sep|>` and they stop converging: the shorter one fires, the separator
# survives, and `<|close|>think<|sep|>` -- the token that ends the reasoning
# channel -- never matches, so at four bytes per chunk the whole answer stays
# in `reasoning_content`. They are a set or they are nothing.
# Named on its own as well, because it is also what sits between a call's
# closer and the section's -- the wrapper walk has to step over it there.
# 【译】不成对的关闭标记，以及单独的分隔符。它们也是外框，但不能移到`CHANNEL_FRAMING`里，原因有两个且彼此独立，
#   尽管这样移动看起来显然是对的。
#
#   `<|sep|>`会出现在工具区域*内部*（`<|open|>call tool="x"<|sep|>`），而推理阶段先运行。在那里去掉它，
#   会让推理阶段只是路过的区域在解析器看到之前就变成残缺的半截文本。
#
#   单独的关闭标记是成对标记的真前缀，而`MarkerScanner`会报告完整的短标记，而不是等着看后面是否跟着长标记
#   （见`_plan`，那里记录了这个缺口以及把各格式约束在缺口之内的测试）。在这个解析器的列表里这无害——
#   去掉`<|close|>think`后剩下的`<|sep|>`也是外框，所以两种分块方式殊途同归。如果把单独的关闭标记加进推理方言
#   而*不加*`<|sep|>`，就不再收敛了：短标记先触发，分隔符留了下来，而`<|close|>think<|sep|>`——结束推理通道的那个token——
#   永远匹配不上，于是在每个chunk四个字节时，整个回答都留在`reasoning_content`里。它们要么作为一组整体出现，要么一个都不放。
#   单独命名它，还因为它也位于一次调用的关闭标记与整段的关闭标记之间——包装结构的遍历要在那里跨过它。
# 解释："这个解析器的列表"指kimi_k3_tool_parser里的外框列表，它同时包含UNPAIRED_FRAMING；推理方言只用CHANNEL_FRAMING
BARE_SEPARATOR = "<|sep|>"

UNPAIRED_FRAMING: tuple[str, ...] = (
    "<|close|>response",
    "<|close|>think",
    "<|close|>message",
    BARE_SEPARATOR,
)

# The model's own `_close_tag` is `<|close|>` + tag + `<|sep|>`, so each of
# these has two spellings on the wire: whole, and cut short by `max_tokens`
# between the tag and the separator. Longest first -- the markup walkers take
# the first that matches, so the bare form listed first would leave the
# separator behind.
# 【译】模型自己的`_close_tag`是`<|close|>` + 标签名 + `<|sep|>`，所以下面每一个在线上都有两种拼写：完整的，
#   以及被`max_tokens`截断在标签名和分隔符之间的。长的在前——标记遍历器取第一个匹配的，如果先列短的形式，分隔符就会被留下。
# 解释：两种拼写由下面的both_spellings生成，这里只定义不带分隔符的部分
TOOLS_END = "<|close|>tools"
CALL_END = "<|close|>call"
ARGUMENT_END = "<|close|>argument"


# 给K3工具调用解析器用：生成(带<|sep|>, 不带<|sep|>)两种关闭标记
def both_spellings(end: str) -> tuple[str, str]:
    """`end` with its separator and without, in the order a walker wants."""
    # 【译】`end`带分隔符和不带分隔符两种形式，按遍历器需要的顺序排列。
    return (end + BARE_SEPARATOR, end)


# Brackets around a tool call. The tool parser's, for the reason in the module
# docstring.
# 【译】调用两侧的括号。归工具调用解析器所有，原因见模块docstring。
TOOL_REGION_FRAMING: tuple[str, ...] = (
    TOOLS_START,
    TOOLS_END,
    CALL_END,
    ARGUMENT_END,
)
