# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Anthropic Messages API adapter for ATOM.

Translates Anthropic /v1/messages requests to ATOM's internal format and
converts responses back to Anthropic format. Enables Claude Code and other
Anthropic-compatible tools to use ATOM as a backend.
"""

# 【译】ATOM的Anthropic Messages API适配器。
#
#   把Anthropic /v1/messages 请求翻译成ATOM的内部格式，再把响应转换回Anthropic格式。
#   这样Claude Code和其他兼容Anthropic的工具就能把ATOM当作后端使用。
#
# 【导读】/v1/messages（Anthropic Messages API）的协议适配层：本文件只做格式转换，不接触引擎
# 调用方：api_server.py 的 anthropic_messages 路由（@app.post("/v1/messages")）；
#   api_server.py 的 /v1/responses 路由（responses_create）也复用了这里的 read_whole_blocks 和 completes_a_tool_call
# 与主干的关系：
#   请求方向：anthropic_to_openai_messages / anthropic_to_openai_tools 把Anthropic的消息和工具定义转成OpenAI chat格式，
#     之后与 /v1/chat/completions 走同一条路：chat template渲染 -> generate_async -> LLMEngine -> CoreManager（本批）
#   响应方向：引擎吐出的增量文本先经推理内容分离（reasoning.py 的 ReasoningChannel），再把content片段交给
#     工具调用解析器（tool_parser/ 的 ToolCallStreamParser），得到有序事件 (类型, 数据)，
#     类型为 "reasoning" / "content" / "tool_call_start" / "tool_call_args" / "tool_call_end"
#     非流式：read_whole_blocks -> build_anthropic_response，一次性拼成content块列表
#     流式：AnthropicBlocks + tool_event_frames 把事件变成SSE事件序列：message_start -> 若干组
#       （content_block_start, content_block_delta..., content_block_stop） -> message_delta -> message_stop
# Anthropic与OpenAI chat的关键差别：Anthropic把响应组织成有序的、带index的content块（text / thinking / tool_use），
#   块的种类一变就要先stop旧块再start新块；OpenAI chat的字段是平铺的（content、reasoning_content、tool_calls）。
#   所以本文件的docstring大多在讲"如何保住事件的先后顺序"

import base64
import hashlib
import json
import logging
import os
from typing import Any

from pydantic import BaseModel

from .reasoning import ReasoningChannel
from .sse import event_frame
from .tool_parser import ToolCallStreamParser

logger = logging.getLogger("atom")


# ── Anthropic Request Schema ───────────────────────────────────────────
# 【译】── Anthropic 请求的数据模型 ───────────────────────────────────────


# 一个content块：按type区分 text / tool_use / tool_result，各自只用其中几个字段（pydantic自动校验JSON）
class AnthropicContentBlock(BaseModel):
    type: str
    text: str | None = None
    # tool_use fields
    # 【译】tool_use 块的字段
    id: str | None = None
    name: str | None = None
    input: Any | None = None
    # tool_result fields
    # 【译】tool_result 块的字段
    tool_use_id: str | None = None
    content: Any | None = None


# 一条消息：role 为 user / assistant；content 可以是纯字符串，也可以是content块列表
class AnthropicMessage(BaseModel):
    role: str
    # 【译】（行尾注释）str 或 list[AnthropicContentBlock]
    content: Any  # str or list[AnthropicContentBlock]


# /v1/messages 的请求体；FastAPI按这个模型解析并校验JSON
class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    # 【译】（行尾注释）str 或 list
    system: Any | None = None  # str or list
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    metadata: dict | None = None
    # 【译】（行尾注释）形如 {"type":"enabled","budget_tokens":N}
    thinking: dict | None = None  # {"type":"enabled","budget_tokens":N}
    # 【译】（行尾注释）PD交接的元数据，与chat接口相同
    # 【分支】PD分离：prefill节点与decode节点之间交接KV时携带的参数，见第9批
    kv_transfer_params: dict | None = None  # PD handoff metadata, as on chat


# ── Format Conversion ──────────────────────────────────────────────────
# 【译】── 格式转换 ───────────────────────────────────────────────────────


# 请求方向：Anthropic的消息列表（加顶层system） -> OpenAI chat的messages列表，之后交给chat template渲染
def anthropic_to_openai_messages(
    messages: list[AnthropicMessage],
    system: Any | None = None,
) -> list[dict]:
    """Convert Anthropic messages to OpenAI format."""
    # 【译】把Anthropic消息转换成OpenAI格式。
    result = []

    # System message
    # 【译】system消息
    # 解释：Anthropic的system是顶层字段（字符串或文本块列表），OpenAI是一条role=system的消息；
    #   以 x-anthropic-billing-header 开头的文本块被跳过，不进入prompt
    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for b in system:
                if b.get("type") == "text":
                    text = b["text"]
                    if text.startswith("x-anthropic-billing-header"):
                        continue
                    text_parts.append(text)
            if text_parts:
                result.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in messages:
        role = msg.role
        content = msg.content

        # assistant：text块拼成content，thinking块拼成reasoning_content，tool_use块变成tool_calls（input用json.dumps转成字符串）
        if role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                reasoning_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "thinking":
                            thinking = block.get("thinking")
                            if isinstance(thinking, str) and thinking:
                                reasoning_parts.append(thinking)
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )
                entry = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if reasoning_parts:
                    entry["reasoning_content"] = "\n".join(reasoning_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

        # user：text块拼成一条user消息；每个tool_result块变成一条role=tool的消息，排在这条user消息之后
        elif role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_results = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    b.get("text", "")
                                    for b in tool_content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": str(tool_content),
                                }
                            )
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                result.extend(tool_results)
        else:
            result.append({"role": role, "content": str(content) if content else ""})

    return result


# 工具定义：Anthropic的 {name, description, input_schema} -> OpenAI的 {"type": "function", "function": {..., "parameters"}}
def anthropic_to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tool definitions to OpenAI format."""
    # 【译】把Anthropic的工具定义转换成OpenAI格式。
    if not tools:
        return None
    result = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return result


# ── Response Construction ──────────────────────────────────────────────
# 【译】── 响应构建 ───────────────────────────────────────────────────────


# 流式生成中途抛异常时的收尾帧：[补发message_start] -> 关闭所有未关闭的块 -> error -> message_delta -> message_stop
def stream_failure_frames(exc, blocks, output_tokens: int, *, opening: str | None):
    """Everything a stream that raised still owes the client.

    ``opening`` is the `message_start` frame when the stream never sent one,
    else ``None``. A failure before the first chunk otherwise produced a
    stream whose first frame is `message_delta`, and the Anthropic SDK builds
    its accumulator from `message_start` and raises on anything before it --
    so the client saw an SDK error rather than the `error` frame this delivers.
    """
    # 【译】一个抛了异常的流仍然欠客户端的全部帧。
    #
    #   如果流还没发过 message_start，opening 就是那一帧，否则为 None。不补发的话，在第一个chunk之前
    #   发生的失败会产生一个第一帧是 message_delta 的流，而Anthropic SDK是从 message_start 开始建立累加器的，
    #   在它之前收到任何东西都会抛错——于是客户端看到的是SDK错误，而不是这里发送的 error 帧。
    if opening is not None:
        yield opening
    # Every block this stream opened has to be closed before the terminator.
    # 【译】这个流打开过的每个块都必须在终止帧之前关闭。
    yield from blocks.close()
    yield stream_error(str(exc))
    yield stream_message_delta("end_turn", output_tokens)
    yield stream_message_stop()


def _sig() -> str:
    """The opaque signature Anthropic puts on a thinking block."""
    # 【译】Anthropic放在thinking块上的不透明签名。
    # 解释：这里并不做真正的签名，只是对32个随机字节取sha256再base64编码，让签名字段有一个看起来合法的值
    return base64.b64encode(hashlib.sha256(os.urandom(32)).digest()).decode()


# 非流式路径的核心：把一个完整输出依次过推理内容分离和工具调用解析，得到与流式完全相同的有序事件列表
def read_whole_blocks(
    channel: ReasoningChannel,
    parser_cls,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> list:
    """One complete output as ordered events, reasoning included.

    The streaming branch's own loop over a single chunk: the reasoning filter
    first, the tool parser on the content segments it yields. `split()`
    returns `(reasoning, content)` and loses the interleaving at that line, so
    blocks rebuilt from the pair put a whole answer in one block ahead of a
    `thinking` that belonged in the middle of it -- two blocks where streaming
    sent three, for the same generation.

    Not Anthropic-specific except that Anthropic is the only endpoint whose
    wire format has an order to lose; the chat path's fields are flat.
    """
    # 【译】把一个完整输出变成有序事件，包含推理内容。
    #
    #   这就是流式分支自己的那套循环，只是只跑一个chunk：先过推理过滤器，再把它产出的content片段交给工具调用解析器。
    #   split() 返回 (reasoning, content)，在那一行就丢失了两者的交错顺序，所以由这一对重建的块会把整段回答
    #   放进一个块，排在本应位于回答中间的 thinking 之前——同一次生成，流式发了三个块，这里只有两个。
    #
    #   它并不是Anthropic专用的，只不过Anthropic是唯一一个线上格式有"顺序"可丢的接口；chat路径的字段是平铺的。
    # 解释：除 /v1/messages 外，api_server.py 的 /v1/responses 非流式分支也调用本函数构造响应
    reader = channel.stream()
    engine = ToolCallStreamParser(
        tools=tools, parser_cls=parser_cls, suppress_calls=suppress_calls
    )
    out: list = []
    for field, segment in reader.process(text) + reader.flush():
        if not segment:
            continue
        if field == "reasoning_content":
            out.append(("reasoning", segment))
        else:
            out.extend(engine.process(segment))
    out.extend(engine.flush())
    return out


# 有序事件 -> Anthropic content块列表（thinking / text / tool_use），相邻的content事件合并成一个text块
def _blocks_in_order(events: list) -> list[dict]:
    """The engine's events as Anthropic content blocks, in arrival order.

    The streaming path already does this, through `AnthropicBlocks` and
    `tool_event_frames`. The non-streaming path was given `(content_text,
    tool_calls)` instead -- the same events with the order thrown away -- and
    rebuilt one text block ahead of every `tool_use`. So the same generation
    came back as `['text', 'tool_use', 'tool_use']` unstreamed and
    `['tool_use', 'text', 'tool_use']` streamed, and a client rendering blocks
    in order showed the sentence introducing the second call before the first.

    Consecutive content events are joined: they are one run of answer that the
    engine happened to emit in pieces, and Anthropic has no way to say "two
    adjacent text blocks" that means anything different.
    """
    # 【译】把引擎的事件按到达顺序转成Anthropic的content块。
    #
    #   流式路径早已通过 AnthropicBlocks 和 tool_event_frames 这样做了。非流式路径以前拿到的却是
    #   (content_text, tool_calls)——同样的事件，但顺序被扔掉了——然后在每个 tool_use 前面重建一个text块。
    #   结果同一次生成，非流式返回 ['text', 'tool_use', 'tool_use']，流式返回 ['tool_use', 'text', 'tool_use']，
    #   按顺序渲染块的客户端会把介绍第二个调用的那句话显示在第一个调用之前。
    #
    #   相邻的content事件会被拼接：它们是引擎恰好分几段吐出的同一段回答，而Anthropic也没法用
    #   "两个相邻的text块"表达任何不同的含义。
    blocks: list[dict] = []
    pending: dict | None = None
    for etype, data in events:
        if not data and etype in ("reasoning", "content"):
            continue
        if etype == "reasoning":
            blocks.append({"type": "thinking", "thinking": data, "signature": _sig()})
        elif etype == "content":
            if blocks and blocks[-1]["type"] == "text":
                blocks[-1]["text"] += data
            else:
                blocks.append({"type": "text", "text": data})
        elif etype == "tool_call_start":
            pending = {"id": data["id"], "name": data["function"]["name"]}
        elif etype == "tool_call_args" and pending is not None:
            raw = data["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # A call cut off mid-arguments. Logged rather than silent, as
                # SGLang does: the client sees a call with no arguments and
                # nothing else says why.
                # 【译】一个在参数中途被截断的调用。像SGLang那样记日志而不是静默处理：客户端看到的是一个没有参数的调用，
                #   除此之外没有任何地方说明原因。
                logger.warning(
                    "tool call %s: arguments are not JSON, sending {}: %r",
                    pending["name"],
                    raw[:120],
                )
                args = {}
            # Whatever decoded, verbatim. Coercing a non-dict to `{}` dropped
            # a Kimi-K2 call's arguments on `stream=false` alone -- that
            # format passes the wire bytes through and the streaming path
            # forwards them for the SDK to accumulate.
            # 【译】解码出什么就原样放什么。以前把非dict强制变成 {}，导致仅在 stream=false 时丢掉了Kimi-K2调用的参数——
            #   该格式把线上的原始字节直接透传，流式路径则把它们转发给SDK去累加。
            blocks.append(
                {
                    "type": "tool_use",
                    "id": pending["id"],
                    "name": pending["name"],
                    "input": args,
                }
            )
            pending = None
    return blocks


# 非流式响应：有序事件 -> 完整的Anthropic message JSON（content块 + stop_reason + usage）
def build_anthropic_response(
    request_id: str,
    model: str,
    events: list,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    *,
    stop_reason: str,
) -> dict:
    """One response, built from the ordered events and nothing else.

    It also took `(content_text, reasoning_content, tool_calls)` and rebuilt
    the blocks from them, so this file held two tool_use builders and two
    orderings -- the two-readers shape the rest of this branch exists to
    delete. It cost what that always costs: reasoning was prepended ahead of
    everything, so a model that answers, thinks and answers again came back as
    `[thinking, text]` with the two answers glued together where streaming
    sent `[text, thinking, text]`.

    `stop_reason` has no default on purpose -- it used to be overwritten here
    whenever `tool_calls` was non-empty, discarding what
    `anthropic_stop_reason_with_calls` had already decided.
    """
    # 【译】一个响应，只由有序事件构建，不依赖其他任何东西。
    #
    #   它以前还接收 (content_text, reasoning_content, tool_calls) 并据此重建块，于是本文件里有两个 tool_use 构造器、
    #   两种排序方式——正是这条改动分支其余部分要消除的"两个读取者"结构。代价也和往常一样：推理内容被放到最前面，
    #   一个"先回答、再思考、再回答"的模型返回成 [thinking, text]，两段回答被粘在一起，而流式发送的是 [text, thinking, text]。
    #
    #   stop_reason 故意没有默认值——以前只要 tool_calls 非空就会在这里被覆盖，
    #   丢弃了 anthropic_stop_reason_with_calls（在 api_server.py 中）已经做出的决定。
    # A response is never empty: Anthropic has no representation for one, and
    # the streaming path forces a final text block for the same reason.
    # 【译】响应永远不为空：Anthropic没有表示空响应的方式，流式路径出于同样的原因也会强制补一个最后的text块。
    content = _blocks_in_order(events) or [{"type": "text", "text": ""}]

    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            # Anthropic convention: input_tokens counts only the
            # non-cached (freshly processed) prompt tokens; cached tokens
            # are reported separately in cache_read_input_tokens.
            # 【译】Anthropic的约定：input_tokens只统计未命中缓存（新处理）的prompt token；
            #   命中缓存的token单独报告在cache_read_input_tokens里。
            # 解释：cache_read_input_tokens由调用方传入，api_server.py 取的是引擎输出的 num_cached_tokens（前缀缓存命中的token数）
            "input_tokens": max(input_tokens - cache_read_input_tokens, 0),
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }


# ── Streaming ──────────────────────────────────────────────────────────
# 【译】── 流式输出 ───────────────────────────────────────────────────────


def format_sse(event: str, data: Any) -> str:
    """Format a server-sent event."""
    # 【译】格式化一个SSE事件。
    # 解释：只是 sse.py 中 event_frame 的薄包装，产出 "event: <名字>\ndata: <JSON>\n\n"
    return event_frame(event, data)


# 流式的第一帧 message_start：一个content为空的message骨架，带输入侧的usage
def stream_message_start(
    request_id: str,
    model: str,
    input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> str:
    return format_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{request_id}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": max(input_tokens - cache_read_input_tokens, 0),
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cache_read_input_tokens,
                },
            },
        },
    )


# 流式输出的块状态机：index是当前块的序号（每关闭一个块+1），kind是当前打开块的种类（None表示没有打开的块）
class AnthropicBlocks:
    """One open content block at a time, closed before the next one opens.

    Anthropic frames a response as indexed blocks of a kind -- text, thinking,
    tool_use -- and a change of kind is a close and an open. Those transitions
    used to be written out at each of the four places a segment could arrive,
    each covering the subset its author needed. The one nobody needed,
    text -> thinking, was missing: a reasoning segment arriving after content
    had started matched no branch and was dropped, with no error and no log.
    Measured on a model that answers, opens a `<think>` block and answers
    again, 29 characters of reasoning went nowhere.

    So the transition is asked for rather than written out: `delta` says which
    kind this text belongs to and the switching is this class's problem. It
    cannot silently do nothing, because there is no branch left to fall off.
    """

    # 【译】同一时刻只有一个打开的content块，下一个打开之前先关闭它。
    #
    #   Anthropic把响应组织成带index、有种类（text、thinking、tool_use）的块，种类一变就是一次关闭加一次打开。
    #   这些切换以前在片段可能到达的四个地方各写了一遍，每处只覆盖作者当时需要的那部分。没人需要的那个切换
    #   text -> thinking 就漏掉了：content开始之后才到达的推理片段匹配不到任何分支，被丢弃，没有报错也没有日志。
    #   在一个"先回答、打开<think>块、再回答"的模型上实测，有29个字符的推理内容就这样不见了。
    #
    #   所以切换是"请求"出来的，而不是逐处写出来的：delta 说明这段文本属于哪个种类，切换由本类负责。
    #   它不可能静默地什么都不做，因为已经没有可以漏掉的分支了。

    def __init__(self) -> None:
        self.index = 0
        self.kind: str | None = None
        # Every call the parser has named, by the index it named it at, and
        # kept for the life of the response. A call spans two parser batches
        # -- the name when the region reveals it, the arguments when the
        # region closes -- and anything at all can arrive in between: a
        # reasoning segment lands as a `thinking` block, which closes whatever
        # was open. Held on the open block instead, the id and name were gone
        # by the time the arguments came and the call reached the client as
        # `tool_use` with `input: {}`.
        # 【译】解析器报出过名字的每个调用，以解析器给出的index为键，在整个响应生命周期内保留。一个调用横跨
        #   解析器的两批输出——区域露出名字时给出名字，区域闭合时给出参数——中间可能到达任何东西：
        #   比如一个推理片段会变成thinking块，从而关闭当时打开的块。如果把id和name挂在打开的块上，
        #   等参数到来时它们已经没了，调用到达客户端时就成了 input: {} 的 tool_use。
        self.calls: dict[int, dict] = {}
        # Which call's block is open, so two calls in a row are two blocks.
        # 【译】当前打开的是哪个调用的块，这样连续两个调用就是两个块。
        self.open_call_index: int | None = None

    def close(self):
        """End the open block, if any. A thinking block signs off first."""
        # 【译】结束当前打开的块（如果有）。thinking块要先签名（发出signature_delta）再结束。
        if self.kind is None:
            return
        if self.kind == "thinking":
            yield stream_signature_delta(self.index)
        yield stream_content_block_stop(self.index)
        self.index += 1
        self.kind = None
        self.open_call_index = None

    def open(self, kind: str, **start_kwargs):
        """Start a block of `kind`, closing whatever was open."""
        # 【译】开始一个 kind 种类的块，先关闭当前打开的块。
        yield from self.close()
        yield stream_content_block_start(self.index, kind, **start_kwargs)
        self.kind = kind

    def delta(self, kind: str, text: str, **start_kwargs):
        """Emit `text` as `kind`, switching blocks if that is not the open one."""
        # 【译】把 text 作为 kind 种类发出；如果当前打开的块不是这个种类，就切换块。
        if self.kind != kind:
            yield from self.open(kind, **start_kwargs)
        yield stream_content_block_delta(self.index, text, kind)

    def tool_delta(self, index: int, arguments: str):
        """Arguments for the call named at ``index``, opening its block.

        The block is opened *here*, when the arguments arrive, and not when
        the name did. On this protocol a `content_block_start` of type
        `tool_use` already carries `"input": {}` and is, on its own, a
        complete zero-argument call -- there is no frame for "the name is
        known, the arguments are still coming". So a name sent before its
        arguments cannot be represented, and sending one anyway put a
        syntactically perfect call the model never made in front of the
        client: measured on a DeepSeek-V4 response containing two calls,
        which reached `/v1/messages` as three.

        OpenAI's wire has the same shape -- a `tool_calls` delta with
        `arguments: ""` -- and is safe with it, because a client there
        accumulates by index and waits for `finish_reason`. That is why the
        name still goes out early on the other endpoint and not on this one.
        """
        # 【译】发送在 index 处报出名字的那个调用的参数，并打开它的块。
        #
        #   块是在这里——参数到达时——打开的，而不是在名字到达时。在这个协议里，一个类型为 tool_use 的
        #   content_block_start 本身就带着 "input": {}，单独看就是一个完整的零参数调用——没有一种帧能表示
        #   "名字已知、参数还在路上"。所以先于参数发出的名字无法表示；硬要发的话，客户端面前就会出现一个
        #   语法上完美、但模型从没发起过的调用：在一个包含两个调用的DeepSeek-V4响应上实测，到达 /v1/messages 时变成了三个。
        #
        #   OpenAI的线上格式也有同样的形状——一个 arguments: "" 的 tool_calls 增量——但在那里是安全的，因为那边的
        #   客户端按index累加，并等待 finish_reason。这就是为什么在另一个接口上名字仍然提前发出，而在这个接口上不会。
        call = self.calls.get(index)
        if call is None:
            # Arguments for a call whose name never came. There is nothing to
            # open a block with, and opening one with empty fields would be a
            # `tool_use` the client can neither dispatch nor return a result
            # for. A parser bug; dropping the frame is the only honest option.
            # 【译】参数属于一个从未报出名字的调用。没有东西可以用来打开块，用空字段打开的话就是一个客户端
            #   既无法分派、也无法返回结果的 tool_use。这是解析器的bug；丢掉这一帧是唯一诚实的选择。
            return
        if self.kind != "tool_use" or self.open_call_index != index:
            yield from self.open("tool_use", **call)
            self.open_call_index = index
        yield stream_content_block_delta(self.index, arguments, "tool_use")


# 流式路径：一批工具调用解析器事件 -> Anthropic SSE帧（content -> text块；tool_call_start只记录；tool_call_args打开tool_use块）
def tool_event_frames(events, blocks: AnthropicBlocks):
    """One batch of tool-parser events as Anthropic frames.

    Written out twice in the streaming endpoint, once for `process` and once
    for `flush`, twenty-two lines each. That is the same hazard
    :class:`AnthropicBlocks` exists to remove one level up -- two copies of a
    dispatch means a fix that lands in one of them, and nothing says so.

    A plain generator and not `yield from` at the call site, because the
    endpoint is an *async* generator and `yield from` is a syntax error inside
    one. Whether a call started is left to the caller to read off `events`;
    returning it from a generator would need the `yield from` that cannot be
    written there.

    Which call is open lives on `blocks` and not here, because this runs once
    per parser batch and one call spans two of them. See `AnthropicBlocks`.
    """
    # 【译】把一批工具调用解析器事件转成Anthropic帧。
    #
    #   以前在流式接口里写了两遍，process 一遍、flush 一遍，每遍22行。这与 AnthropicBlocks 在上一层要消除的是同一个隐患——
    #   分派逻辑有两份拷贝，意味着修复可能只落在其中一份上，而且没有任何东西会提醒你。
    #
    #   它是一个普通生成器，调用处不用 yield from，因为接口本身是一个 async 生成器，在其中写 yield from 是语法错误。
    #   是否开始了一个调用，留给调用方自己从 events 里读；要从生成器里返回这个值就需要 yield from，而那里写不了。
    #
    #   当前打开的是哪个调用记在 blocks 上而不是这里，因为本函数每批解析器输出运行一次，而一个调用横跨两批。见 AnthropicBlocks。
    # 解释：api_server.py 的调用处写成 for _frame in tool_event_frames(events, blocks): yield _frame
    for etype, edata in events:
        if etype == "content":
            yield from blocks.delta("text", edata)
        elif etype == "tool_call_start":
            # Recorded, not sent. See `AnthropicBlocks.tool_delta` for why a
            # name on its own cannot be put on this wire.
            # 【译】只记录，不发送。为什么单独一个名字不能放到这个线上格式里，见 AnthropicBlocks.tool_delta。
            fn = edata.get("function", {})
            blocks.calls[edata.get("index", 0)] = {
                "tool_use_id": edata.get("id", ""),
                "tool_name": fn.get("name", ""),
            }
        elif etype == "tool_call_args":
            yield from blocks.tool_delta(
                edata.get("index", 0), edata.get("function", {}).get("arguments", "")
            )
        elif etype == "tool_call_end":
            yield from blocks.close()


# 调用方：api_server.py 的 /v1/messages 和 /v1/responses 流式分支，用它累计 has_tool_calls，进而决定结束原因
def completes_a_tool_call(events) -> bool:
    """Whether this batch produced a *usable* tool call.

    Keyed on the arguments and not the name, which is what makes announcing a
    name early safe. A name can be sent before the call is known to close --
    the point of announcing it -- so a response truncated at `max_tokens`
    mid-call has sent a name and nothing else. Reporting `tool_use` there
    would tell the client to run a tool whose arguments never arrived.

    Every parser emits name and arguments together unless it announced early,
    so this reads the same as the name for every format that does not.
    """
    # 【译】这一批事件是否产出了一个"可用的"工具调用。
    #
    #   以参数而不是名字为准，这正是提前公布名字之所以安全的原因。名字可以在确认调用会闭合之前发出——这就是提前公布的意义——
    #   所以一个在调用中途被 max_tokens 截断的响应只发出了名字，别的什么都没有。此时报告 tool_use，
    #   就等于让客户端去执行一个参数从未到达的工具。
    #
    #   除非提前公布，否则每个解析器都会把名字和参数一起发出，所以对于不提前公布的格式，按参数判断和按名字判断结果相同。
    return any(etype == "tool_call_args" for etype, _ in events)


# content_block_start帧：按块种类给出空的初始块（tool_use块带id、name和空的input）
def stream_content_block_start(
    index: int,
    block_type: str = "text",
    tool_use_id: str = "",
    tool_name: str = "",
) -> str:
    if block_type == "thinking":
        block = {"type": "thinking", "thinking": "", "signature": ""}
    elif block_type == "tool_use":
        block = {
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": {},
        }
    else:
        block = {"type": "text", "text": ""}
    return format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        },
    )


# content_block_delta帧：text -> text_delta，thinking -> thinking_delta，tool_use -> input_json_delta（参数JSON的片段）
def stream_content_block_delta(index: int, text: str, block_type: str = "text") -> str:
    if block_type == "thinking":
        delta = {"type": "thinking_delta", "thinking": text}
    elif block_type == "tool_use":
        delta = {"type": "input_json_delta", "partial_json": text}
    else:
        delta = {"type": "text_delta", "text": text}
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )


# thinking块结束前的signature_delta帧（由 AnthropicBlocks.close 发出）
def stream_signature_delta(index: int) -> str:
    """Emit a signature_delta for thinking blocks (required by Claude Code)."""
    # 【译】为thinking块发送一个signature_delta（Claude Code要求有它）。
    dummy_sig = _sig()
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": dummy_sig},
        },
    )


# content_block_stop帧：关闭第index个块
def stream_content_block_stop(index: int) -> str:
    return format_sse(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


# message_delta帧：报告stop_reason和输出token数，紧接在message_stop之前
def stream_message_delta(stop_reason: str = "end_turn", output_tokens: int = 0) -> str:
    return format_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


# error帧：流无法正常结束时发出，见 stream_failure_frames
def stream_error(message: str, error_type: str = "api_error") -> str:
    """Anthropic's `error` event, for a stream that cannot finish.

    Without one, an exception raised inside the SSE generator ended the
    response mid-frame: an open content block with no `content_block_stop`, no
    `message_delta`, no `message_stop` and nothing saying why. An Anthropic
    SDK client blocks on the unterminated block until its own read timeout
    rather than surfacing the failure.
    """
    # 【译】Anthropic的 error 事件，用于无法正常结束的流。
    #
    #   没有它时，SSE生成器内部抛出的异常会让响应在帧中途结束：一个打开的content块没有 content_block_stop，
    #   没有 message_delta，没有 message_stop，也没有任何说明原因的东西。Anthropic SDK客户端会一直卡在这个
    #   未终止的块上，直到它自己的读超时，而不是把失败报告出来。
    return format_sse(
        "error", {"type": "error", "error": {"type": error_type, "message": message}}
    )


# message_stop帧：流的最后一帧
def stream_message_stop() -> str:
    return format_sse("message_stop", {"type": "message_stop"})
