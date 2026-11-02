# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat completion handler for the OpenAI-compatible API."""

# 【译】OpenAI兼容API的chat completion处理器。
#
# 本文件（第2批2c，B级）：/v1/chat/completions 的请求校验与响应构造（接口函数本身在api_server.py）
#   校验：normalize_chat_tools、validate_tool_list、validate_chat_request、resolve_thinking
#   流式：stream_chat_response（n=1）、stream_chat_response_fanout（n>1）——从StreamOutputCollector取chunk，产出SSE帧
#   非流式：build_chat_response / build_chat_response_multi——把generate_async*的结果dict组装成ChatCompletionResponse
# 每段生成文本都要经过两道处理（流式时逐chunk进行，非流式时对全文一次完成）：
#   原始文本 ──推理过滤器（reasoning.py）──> reasoning_content片段 / content片段
#            content片段 ──工具调用解析器（tool_parser/）──> 普通文本 / tool_calls
# 一个流式响应的帧序列（每行是一帧 "data: {...}"，这里省略了id、model等公共字段）：
#   {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}  ← 第一个chunk到达时
#   {"choices": [{"index": 0, "delta": {"reasoning_content": "先想一想"}, ...}]}                      ← 思考段
#   {"choices": [{"index": 0, "delta": {"content": "答案是42"}, ...}]}                                ← 正文
#   {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}  ┐
#   {"choices": [], "usage": {...}}                                    ├ 最后三帧合并成一次发送
#   [DONE]                                                             ┘
# 对比nano-vllm：nano-vllm没有这一层
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from .protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    STREAM_DONE_MESSAGE,
    TOOL_CHOICE_VALUES,
    ChatCompletionRequest,
    ChatCompletionResponse,
    openai_stop_reason_with_calls,
)
from .reasoning import (
    NO_REASONING,
    VALID_TEMPLATE_EFFORTS,
    ReasoningChannel,
)
from .sse import data_frame
from .streaming_dispatch import StreamOutputCollector
from .tool_parser import ToolCallStreamParser, parse_tool_calls
from .tool_parser.registry import forbids_tool_calls
from .tool_parser.tool_parser import qualified_tool_name

logger = logging.getLogger("atom")


# ============================================================================
# Request validation & thinking control
# ============================================================================
# 【译】请求校验与思考控制


# 把Anthropic风格的工具定义转成OpenAI的function工具，其他形状原样返回。调用方：api_server.chat_completions
# 例：{"name": "get_weather", "description": "查天气", "input_schema": {...}}
#   → {"type": "function", "function": {"name": "get_weather", "description": "查天气", "parameters": {...}}}
def normalize_chat_tools(tools: Any) -> Any:
    """Accept Anthropic-style tools on the OpenAI-compatible endpoint.

    Well-formed OpenAI tools and malformed values are left unchanged so the
    existing validator remains authoritative. Only the unambiguous Anthropic
    shape (name + input_schema, without type/function) is converted.
    """
    # 【译】在OpenAI兼容接口上接受Anthropic风格的工具定义。
    #
    #   格式正确的OpenAI工具和格式错误的值都原样保留，让现有的校验器仍然说了算。
    #   只有毫无歧义的Anthropic形状（有name + input_schema，没有type/function）才会被转换。
    if not isinstance(tools, list):
        return tools

    normalized = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and "type" not in tool
            and "function" not in tool
            and isinstance(tool.get("name"), str)
            and isinstance(tool.get("input_schema"), dict)
        ):
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            )
        else:
            normalized.append(tool)
    return normalized


# 请求没带tools时，决定还要不要启用工具调用解析器（parser_cls为None表示不解析）
def _tool_parser_for_request(parser_cls, tools):
    """Return the parser this request actually needs.

    A no-tools request cannot produce a dispatchable tool call. Disable parsers
    whose markers only open tool-call regions so an unclosed marker remains
    ordinary content instead of withholding the stream until EOS. Parsers that
    consume model-wide channel framing must remain active without tools.
    """
    # 【译】返回这个请求实际需要的解析器。
    #
    #   没有工具的请求不可能产生可派发的工具调用。对那些标记只用于打开工具调用区域的解析器，直接禁用，
    #   这样一个未闭合的标记仍是普通内容，而不会把流一直扣到EOS。消费全模型通道框架的解析器，在没有工具时也必须保持启用。
    if parser_cls is None or tools:
        return parser_cls
    consumes_channel_framing = any(
        not parser_cls.opens_region(marker) for marker in parser_cls.START_MARKERS
    )
    return parser_cls if consumes_channel_framing else None


# 从请求解析(是否开启思考, 思考力度)，api_server.chat_completions据此写chat template参数。例：
#   请求带 {"thinking": {"type": "disabled"}}  → (False, None)
#   请求带 {"reasoning_effort": "none"}        → (False, None)
#   请求带 {"reasoning_effort": 32}            → (None, 32)    （整数预算，必须在[1, 100]内，否则ValueError）
#   什么都没给                                 → (None, None)
def resolve_thinking(
    request: ChatCompletionRequest,
) -> tuple[bool | None, str | int | None]:
    """Resolve (enabled, effort) from the request's thinking / reasoning_effort.

    ``thinking`` (extra_body) takes precedence over ``reasoning_effort``.
    Thinking is disabled when ``thinking.type == "disabled"`` or
    ``reasoning_effort == "none"``. Effort is only returned when it is one of
    the values the template understands.

    ``enabled`` is ``None`` when the request stated no preference -- setting
    an effort is not asking for reasoning to be switched *on*. Collapsing
    that to ``True`` was harmless while the caller wrote a key no template
    read; once it wrote the template's real switch, a request carrying only
    `reasoning_effort` re-enabled reasoning over an operator's
    `--default-chat-template-kwargs '{"enable_thinking": false}'`, and over a
    client's own `chat_template_kwargs`, because the toggle is merged last.
    """
    # 【译】从请求的 thinking / reasoning_effort 解析出 (enabled, effort)。
    #
    #   thinking（extra_body）优先于 reasoning_effort。thinking.type == "disabled" 或 reasoning_effort == "none" 时关闭思考。
    #   只有当effort是模板能理解的取值之一时才返回它。
    #
    #   请求没有表态时 enabled 为 None——设置effort并不是要求把推理"打开"。在调用方写的还是一个没有模板读取的key时，
    #   把它折叠成 True 是无害的；一旦调用方写入模板真正的开关，一个只带 reasoning_effort 的请求就会越过运维人员的
    #   --default-chat-template-kwargs '{"enable_thinking": false}'、以及客户端自己的 chat_template_kwargs，把推理重新打开，
    #   因为这个开关是最后合并的。
    # 解释："thinking优先"指effort的取值；对enabled而言，reasoning_effort == "none"会覆盖thinking.type给出的结果
    #   字符串effort不在VALID_TEMPLATE_EFFORTS（各方言模板认识的取值）里时被静默丢弃
    thinking = request.thinking or {}
    enabled: bool | None = None
    if isinstance(thinking, dict) and thinking.get("type") is not None:
        enabled = thinking.get("type") != "disabled"
    if request.reasoning_effort == "none":
        enabled = False

    effort = None
    if isinstance(thinking, dict) and thinking.get("effort") is not None:
        effort = thinking.get("effort")
    elif request.reasoning_effort is not None:
        effort = request.reasoning_effort
    if isinstance(effort, (int, float)):
        if type(effort) is not int or not 1 <= effort <= 100:
            raise ValueError("reasoning effort must be an integer in [1, 100]")
    elif not isinstance(effort, str) or effort not in VALID_TEMPLATE_EFFORTS:
        effort = None
    return enabled, effort


# 校验单个工具：必须是{"type": "function", "function": {...}}，返回带命名空间的工具名（tool_parser的qualified_tool_name）
def _validate_one_tool(tool: Any, index: int) -> str:
    if not isinstance(tool, dict):
        # ValueError (not TypeError) so the handler maps it to HTTP 400.
        # 【译】用ValueError（而不是TypeError），这样handler会把它映射成HTTP 400。
        raise ValueError(f"tools[{index}] must be an object")  # noqa: TRY004
    if tool.get("type") != "function":
        raise ValueError(f"tools[{index}].type must be 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict):
        raise ValueError(f"tools[{index}].function must be an object")  # noqa: TRY004
    try:
        return qualified_tool_name(tool)
    except ValueError as exc:
        raise ValueError(f"tools[{index}].{exc}") from exc


# 校验工具列表：必须是数组，每个工具合法，且名字不重复。调用方：validate_chat_request、api_server的messages / responses接口
def validate_tool_list(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")  # noqa: TRY004
    seen: set[str] = set()
    for i, tool in enumerate(tools):
        name = _validate_one_tool(tool, i)
        if name in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)


# chat请求的整体形状校验（api_server.chat_completions在渲染模板之前调用）
def validate_chat_request(request: ChatCompletionRequest) -> None:
    """Validate tool / tool_choice / response_format shape before dispatch.

    Raises ``ValueError`` (surfaced as HTTP 400) on malformed input so the
    engine is never handed a request the chat template cannot render.
    """
    # 【译】在派发之前校验 tool / tool_choice / response_format 的形状。
    #
    #   输入格式错误时抛出 ValueError（表现为HTTP 400），这样引擎永远不会拿到一个chat template无法渲染的请求。
    validate_tool_list(request.tools)

    tool_choice = request.tool_choice
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(
                    f"tool_choice string must be one of {sorted(TOOL_CHOICE_VALUES)}"
                )
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                raise ValueError("tool_choice object must have type 'function'")
            fn = tool_choice.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError(  # noqa: TRY004
                    "tool_choice.function.name must be a string"
                )
            # A named tool_choice must reference a declared tool.
            # 【译】指名的 tool_choice 必须引用一个已声明的工具。
            names = {
                qualified_tool_name(t)
                for t in (request.tools or [])
                if isinstance(t, dict) and isinstance(t.get("function"), dict)
            }
            if qualified_tool_name(tool_choice) not in names:
                raise ValueError(f"tool_choice names unknown tool: {fn['name']}")
        else:
            raise ValueError("tool_choice must be a string or an object")

    rf = request.response_format
    if rf is not None:
        if not isinstance(rf, dict):
            raise ValueError("response_format must be an object")
        rf_type = rf.get("type")
        if rf_type not in ("text", "json_object", "json_schema"):
            raise ValueError(
                "response_format.type must be 'text', 'json_object', or 'json_schema'"
            )
        if rf_type == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or not isinstance(js.get("schema"), dict):
                raise ValueError("response_format.json_schema.schema must be an object")


# 构造一帧chat.completion.chunk并编码成SSE：delta是本帧的增量（role / content / reasoning_content / tool_calls），
#   finish_reason只在结束帧上给出
def create_chat_chunk(
    request_id: str,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    index: int = 0,
) -> str:
    """Create a chat completion chunk in SSE format.

    ``index`` selects the ``choices[0].index`` field so fan-out siblings
    (SamplingParams.n>1) can be multiplexed over a single stream.
    """
    # 【译】创建一个SSE格式的chat completion chunk。
    #
    #   index 决定 choices[0].index 字段，这样扇出的兄弟序列（SamplingParams.n>1）可以复用同一个流。
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta if delta else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return data_frame(chunk)


# =====================================================================================
# 流式chat响应（n=1）：文件头图1中的⑬⑭（api_server.py）
# =====================================================================================
# 调用方：api_server.chat_completions创建这个异步生成器，交给 StreamingResponse(_client_stream(gen))；
#   uvicorn每取一帧就写一次socket
# 循环：await collector.get()（已反分词的chunk）→ 推理过滤器 → 工具解析器 → yield SSE帧；finished后发结束帧
# finally：无论正常结束还是客户端断开都会执行，调用api_server传进来的cleanup_stream / cleanup_request
#   （以参数传入而不是import，避免serving_chat与api_server循环导入）
async def stream_chat_response(
    request_id: str,
    model: str,
    stream_collector: StreamOutputCollector,
    seq_id: int,
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with reasoning and tool calls.

    Yields SSE chunks with:
    - reasoning_content deltas during thinking phase
    - content deltas for the answer
    - tool_calls deltas when model invokes tools

    ``num_prompt_tokens`` is the engine-computed prompt length (``Sequence.
    num_prompt_tokens``); reusing it avoids re-tokenizing the prompt on the
    event loop at stream start.
    """
    # 【译】生成带推理内容和工具调用的流式chat completion响应。
    #
    #   产出的SSE chunk包括：
    #   - 思考阶段的 reasoning_content 增量
    #   - 回答的 content 增量
    #   - 模型调用工具时的 tool_calls 增量
    #
    #   num_prompt_tokens 是引擎算出的prompt长度（Sequence.num_prompt_tokens）；复用它可以避免在流开始时
    #   在事件循环上重新tokenize prompt。
    num_tokens_input = num_prompt_tokens
    num_tokens_output = 0
    num_cached_tokens = 0
    # The engine's own reason, kept so the final chunk can report it. The
    # streaming paths hardcoded `stop`, so a response the engine cut off at
    # `max_tokens` was reported as complete while `stream=false` reported
    # `length` for the same generation -- and a `stop_<token_id>` stop, which
    # the Anthropic path maps to `stop_sequence`, collapsed the same way.
    # 【译】引擎自己给出的原因，保留下来让最后一个chunk报告它。以前流式路径硬编码 stop，所以一个被引擎在 max_tokens 处截断的响应
    #   被报告为完整，而 stream=false 对同一次生成报告的是 length——而 stop_<token_id> 这种停止（Anthropic路径把它映射成
    #   stop_sequence）也以同样的方式被压扁了。
    # 解释：如今api_server.anthropic_stop_reason已把stop_<token_id>映射为end_turn，不再映射成stop_sequence
    engine_finish_reason: str | None = None
    # 每个流一个推理过滤器：它是跨chunk的状态机（思考标记可能被切在两个chunk之间）
    # 工具解析器同理；tool_choice禁止工具调用时suppress_calls=True，解析出的调用不会作为tool_calls发出
    reasoning_filter = reasoning.stream()
    tool_parser = ToolCallStreamParser(
        tools=tools,
        parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
        suppress_calls=forbids_tool_calls(tool_choice),
    )
    has_tool_calls = False

    # 【分支】PD分离：prefill节点的KV位置，放进最后的usage帧带回给路由器（第9批）
    kv_transfer_params_value = None

    # Assume abort until the engine's finished chunk arrives. A client
    # disconnect closes the generator before then, leaving this True so the
    # finally aborts the still-running seq; normal completion flips it to False.
    # 【译】在引擎的finished chunk到达之前都假定是abort。客户端断开会在那之前关闭生成器，这个标志保持True，
    #   finally就会abort仍在运行的序列；正常结束时把它改成False。
    aborted = True
    try:
        role_sent = False
        while True:
            # 等下一个chunk：{"text": 已反分词的新增文本, "token_ids": [...], "finished": bool, "finish_reason": ...,
            #   "num_cached_tokens": ..., 可能还有"kv_transfer_params"}；落后时多个chunk已被合并成一个
            chunk_data = await stream_collector.get()

            # 第一个chunk到达时先发一帧role=assistant（OpenAI流式协议的惯例）
            if not role_sent:
                yield create_chat_chunk(
                    request_id, model, delta={"role": "assistant", "content": ""}
                )
                role_sent = True
            new_text = chunk_data["text"]
            if chunk_data.get("finish_reason"):
                engine_finish_reason = chunk_data["finish_reason"]
            num_tokens_output += len(chunk_data.get("token_ids", []))
            _ct = chunk_data.get("num_cached_tokens", 0)
            if _ct:
                num_cached_tokens = _ct

            if "kv_transfer_params" in chunk_data:
                kv_transfer_params_value = chunk_data["kv_transfer_params"]

            # Phase 1: Process through reasoning filter
            # 【译】阶段1：经过推理过滤器
            # segments是[(字段, 文本), ...]，字段为"reasoning_content"或"content"；结束时flush出过滤器里还扣着的文本
            segments = reasoning_filter.process(new_text)
            if chunk_data.get("finished", False):
                segments.extend(reasoning_filter.flush())

            # Phase 2: For content segments, check for tool calls
            # 【译】阶段2：对正文片段检查工具调用
            for field, text in segments:
                if field == "reasoning_content":
                    if text:
                        yield create_chat_chunk(
                            request_id, model, delta={"reasoning_content": text}
                        )
                elif field == "content":
                    # Run through tool parser
                    # 【译】交给工具解析器
                    # events：("content", 文本) / ("tool_call_start", {...}) / ("tool_call_args", {...})，后两者作为delta.tool_calls发出
                    events = tool_parser.process(text)
                    for event_type, data in events:
                        if event_type == "content":
                            yield create_chat_chunk(
                                request_id, model, delta={"content": data}
                            )
                        elif event_type == "tool_call_start":
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                            )
                        elif event_type == "tool_call_args":
                            has_tool_calls = True
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                            )

            if chunk_data.get("finished", False):
                # Flush tool parser
                # 【译】冲刷工具解析器
                for event_type, data in tool_parser.flush():
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}
                        )
                    elif event_type == "tool_call_start":
                        yield create_chat_chunk(
                            request_id, model, delta={"tool_calls": [data]}
                        )
                    elif event_type == "tool_call_args":
                        has_tool_calls = True
                        yield create_chat_chunk(
                            request_id, model, delta={"tool_calls": [data]}
                        )
                break

        # 正常收到finished：不需要abort（引擎已经结束了这条序列）
        aborted = False

        # Final chunks
        # 【译】最后的几个chunk
        # finish_reason：length > tool_calls > 引擎原因映射（protocol.openai_stop_reason_with_calls）
        finish_reason = openai_stop_reason_with_calls(
            engine_finish_reason, has_tool_calls
        )
        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": num_tokens_output,
            "total_tokens": num_tokens_input + num_tokens_output,
            "prompt_tokens_details": {"cached_tokens": num_cached_tokens},
        }
        usage_chunk = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if kv_transfer_params_value is not None:
            usage_chunk["kv_transfer_params"] = kv_transfer_params_value
        # Coalesce finish + usage + [DONE] into one send: at a wave boundary many
        # requests finalize at once, so collapsing 3 socket writes/req to 1 cuts
        # the syscalls that saturate the API event loop.
        # 【译】把finish + usage + [DONE]合并成一次发送：在一波请求的边界上，很多请求同时结束，
        #   把每个请求3次socket写合成1次，能减少那些把API事件循环打满的系统调用。
        yield (
            create_chat_chunk(request_id, model, finish_reason=finish_reason)
            + data_frame(usage_chunk)
            + STREAM_DONE_MESSAGE
        )
    finally:
        # 清理：aborted为True时cleanup_stream会向引擎发abort；然后删掉请求级的登记
        cleanup_stream(seq_id, aborted=aborted)
        cleanup_request(request_id)


# 非流式：对完整文本一次性做"推理内容分离 + 工具调用解析"，得到一个choice
def _build_chat_choice(
    raw_text: str,
    finish_reason: str | None,
    index: int = 0,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
) -> dict[str, Any]:
    """Build one entry of ``choices[...]`` from a raw output string.

    Factored out of :func:`build_chat_response` so multi-sample responses
    (SamplingParams.n>1) can reuse the reasoning + tool-call separation
    without duplicating the logic.
    """
    # 【译】从一段原始输出字符串构造 choices[...] 中的一项。
    #
    #   从 build_chat_response 中抽出来，让多样本响应（SamplingParams.n>1）能复用推理内容 + 工具调用的分离逻辑，而不用复制一份。
    reasoning_content, content_with_tools = reasoning.split(raw_text)
    content, tool_calls = parse_tool_calls(
        content_with_tools,
        tools,
        parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
        suppress_calls=forbids_tool_calls(tool_choice),
    )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

    # `is not None`, not truthiness: `""` is `Sequence.leave_reason`'s initial
    # value and what the engine forwards for a response with no recorded
    # reason. A truthiness test reported `finish_reason: null` for it, which an
    # OpenAI client reads as "still in progress" and some SDKs raise on.
    # 【译】用 is not None 而不是真值判断："" 是 Sequence.leave_reason 的初始值，也是引擎对没有记录原因的响应转发的值。
    #   用真值判断会为它报告 finish_reason: null，OpenAI客户端会把它读成"仍在进行中"，有些SDK还会因此报错。
    effective_finish_reason = (
        openai_stop_reason_with_calls(finish_reason, bool(tool_calls))
        if (tool_calls or finish_reason is not None)
        else None
    )
    return {
        "index": index,
        "message": message,
        "finish_reason": effective_finish_reason,
    }


# 非流式、n=1：把generate_async的结果dict组装成ChatCompletionResponse。调用方：api_server.chat_completions
# usage除了OpenAI的标准字段，还附带ATOM自己的ttft_s / tpot_s / latency_s（秒，保留4位小数）
def build_chat_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: dict[str, Any],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
) -> ChatCompletionResponse:
    """Build a non-streaming chat completion response (single choice)."""
    # 【译】构造一个非流式的chat completion响应（单个choice）。
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            _build_chat_choice(
                raw_text,
                final_output["finish_reason"],
                index=0,
                tools=tools,
                tool_choice=tool_choice,
                reasoning=reasoning,
                tool_parser_cls=tool_parser_cls,
            )
        ],
        usage={
            "prompt_tokens": final_output["num_tokens_input"],
            "completion_tokens": final_output["num_tokens_output"],
            "total_tokens": final_output["num_tokens_input"]
            + final_output["num_tokens_output"],
            "prompt_tokens_details": {
                "cached_tokens": final_output.get("num_cached_tokens", 0)
            },
            "ttft_s": round(final_output.get("ttft", 0.0), 4),
            "tpot_s": round(final_output.get("tpot", 0.0), 4),
            "latency_s": round(final_output.get("latency", 0.0), 4),
        },
    )
    # 可选字段（PD分离的KV位置、回显的prompt token id）用model_copy(update=...)补上：返回一个更新了这些字段的副本
    update: dict[str, Any] = {}
    if "kv_transfer_output_meta_info" in final_output:
        update["kv_transfer_params"] = final_output["kv_transfer_output_meta_info"]
    if "prompt_token_ids" in final_output:
        update["prompt_token_ids"] = final_output["prompt_token_ids"]
    if update:
        response = response.model_copy(update=update)
    return response


# 非流式、n>1：每个兄弟一个choice（index=兄弟序号）
def build_chat_response_multi(
    request_id: str,
    model: str,
    final_outputs: list[dict[str, Any]],
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
) -> ChatCompletionResponse:
    """Build a non-streaming response with one choice per fan-out sibling.

    Assumes all ``final_outputs`` share the same prompt and therefore the
    same ``num_tokens_input``. Completion-token counts are summed across
    siblings for usage; ttft/tpot/latency are reported as the max observed
    across siblings, which approximates wall-clock time to return the full
    multi-sample response to the client.
    """
    # 【译】构造一个非流式响应，每个扇出的兄弟序列一个choice。
    #
    #   假定所有 final_outputs 共用同一个prompt，因而 num_tokens_input 相同。usage里的completion token数是各兄弟之和；
    #   ttft/tpot/latency 报告各兄弟中观察到的最大值，近似于把完整的多样本响应返回给客户端所需的墙钟时间。
    assert final_outputs, "build_chat_response_multi requires at least one output"
    choices = [
        _build_chat_choice(
            out["text"],
            out["finish_reason"],
            index=i,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            tool_parser_cls=tool_parser_cls,
        )
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": final_outputs[0].get("num_cached_tokens", 0)
            },
            "ttft_s": round(
                max((out.get("ttft", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "tpot_s": round(
                max((out.get("tpot", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "latency_s": round(
                max((out.get("latency", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "num_choices": len(final_outputs),
        },
    )
    # Sibling outputs share the first output's prompt IDs.
    # 【译】兄弟输出共用第一个输出的prompt ID。
    if "prompt_token_ids" in final_outputs[0]:
        response = response.model_copy(
            update={"prompt_token_ids": final_outputs[0]["prompt_token_ids"]}
        )
    return response


# 流式、n>1：n个兄弟共用一个collector，get()返回(兄弟序号, chunk)；每个兄弟各有自己的推理过滤器和工具解析器，
#   帧里用choices[0].index区分；所有兄弟都finished后，一次性发出n个结束帧 + usage + [DONE]
async def stream_chat_response_fanout(
    request_id: str,
    model: str,
    shared_collector: StreamOutputCollector,
    seq_ids: list[int],
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
    tools=None,
    tool_choice=None,
    reasoning: ReasoningChannel = NO_REASONING,
    tool_parser_cls=None,
) -> AsyncGenerator[str, None]:
    """Streaming variant that multiplexes ``len(seq_ids)`` fan-out siblings
    into a single SSE stream, tagging every chunk with ``choices[0].index``.

    The shared queue receives ``(sibling_index, chunk_data)`` tuples from
    the engine callbacks registered in :func:`setup_streaming_request_fanout`.
    Reasoning + tool-call state is kept independently per sibling.

    ``num_prompt_tokens`` is the engine-computed prompt length shared by all
    siblings (they tokenize the same prompt once); reusing it avoids
    re-tokenizing on the event loop at stream start.
    """
    # 【译】流式变体：把 len(seq_ids) 个扇出的兄弟序列复用到一个SSE流里，每个chunk都标上 choices[0].index。
    #
    #   共享队列从 setup_streaming_request_fanout 中注册的引擎回调接收 (sibling_index, chunk_data) 元组。
    #   推理内容和工具调用的状态按兄弟分别维护。
    #
    #   num_prompt_tokens 是引擎算出的、所有兄弟共用的prompt长度（它们只对同一个prompt tokenize一次）；
    #   复用它可以避免在流开始时在事件循环上重新tokenize。
    n = len(seq_ids)
    num_tokens_input = num_prompt_tokens
    num_tokens_output = [0] * n
    # Every sibling answers the same prompt, so they start in the same state.
    # 【译】每个兄弟回答的是同一个prompt，所以它们从相同的状态开始。
    reasoning_filters = [reasoning.stream() for _ in range(n)]
    tool_parsers = [
        ToolCallStreamParser(
            tools=tools,
            parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
            suppress_calls=forbids_tool_calls(tool_choice),
        )
        for _ in range(n)
    ]
    has_tool_calls = [False] * n
    # See the single-choice path: the engine's reason, per sibling.
    # 【译】见单choice路径：引擎给出的原因，每个兄弟一份。
    engine_finish_reasons: list[str | None] = [None] * n
    finished = [False] * n
    kv_transfer_params_value = None
    num_cached_tokens = 0

    # Assume abort until every sibling reports finished; a client disconnect
    # closes the generator first, leaving this True so the finally aborts
    # whichever siblings are still running.
    # 【译】在每个兄弟都报告finished之前都假定是abort；客户端断开会先关闭生成器，这个标志保持True，
    #   finally就会abort所有仍在运行的兄弟。
    aborted = True
    try:
        role_sent = [False] * n
        while not all(finished):
            idx, chunk_data = await shared_collector.get()

            if not role_sent[idx]:
                yield create_chat_chunk(
                    request_id,
                    model,
                    delta={"role": "assistant", "content": ""},
                    index=idx,
                )
                role_sent[idx] = True

            if finished[idx]:
                # Defensive: should not happen, engine emits finished once per seq.
                # 【译】防御性检查：不应发生，引擎对每条序列只发一次finished。
                continue
            new_text = chunk_data["text"]
            if chunk_data.get("finish_reason"):
                engine_finish_reasons[idx] = chunk_data["finish_reason"]
            num_tokens_output[idx] += len(chunk_data.get("token_ids", []))
            _ct = chunk_data.get("num_cached_tokens", 0)
            if _ct:
                num_cached_tokens = _ct

            if "kv_transfer_params" in chunk_data:
                kv_transfer_params_value = chunk_data["kv_transfer_params"]

            segments = reasoning_filters[idx].process(new_text)
            if chunk_data.get("finished", False):
                segments.extend(reasoning_filters[idx].flush())

            for field, text in segments:
                if field == "reasoning_content":
                    if text:
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"reasoning_content": text},
                            index=idx,
                        )
                elif field == "content":
                    for event_type, data in tool_parsers[idx].process(text):
                        if event_type == "content":
                            yield create_chat_chunk(
                                request_id, model, delta={"content": data}, index=idx
                            )
                        elif event_type == "tool_call_start":
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                                index=idx,
                            )
                        elif event_type == "tool_call_args":
                            has_tool_calls[idx] = True
                            yield create_chat_chunk(
                                request_id,
                                model,
                                delta={"tool_calls": [data]},
                                index=idx,
                            )

            if chunk_data.get("finished", False):
                for event_type, data in tool_parsers[idx].flush():
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}, index=idx
                        )
                    elif event_type == "tool_call_start":
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )
                    elif event_type == "tool_call_args":
                        has_tool_calls[idx] = True
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )
                finished[idx] = True

        aborted = False

        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": sum(num_tokens_output),
            "total_tokens": num_tokens_input + sum(num_tokens_output),
            "num_choices": n,
            "prompt_tokens_details": {"cached_tokens": num_cached_tokens},
        }
        usage_chunk = {
            "id": request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        if kv_transfer_params_value is not None:
            usage_chunk["kv_transfer_params"] = kv_transfer_params_value
        # Coalesce the per-sibling finish chunks + usage + [DONE] into one send.
        # 【译】把每个兄弟的finish chunk + usage + [DONE] 合并成一次发送。
        yield (
            "".join(
                create_chat_chunk(
                    request_id,
                    model,
                    finish_reason=openai_stop_reason_with_calls(
                        engine_finish_reasons[i], has_tool_calls[i]
                    ),
                    index=i,
                )
                for i in range(n)
            )
            + data_frame(usage_chunk)
            + STREAM_DONE_MESSAGE
        )
    finally:
        # 对所有兄弟用同一个aborted标志清理（已结束的兄弟收到abort也只是空操作）
        for sid in seq_ids:
            cleanup_stream(sid, aborted=aborted)
        cleanup_request(request_id)
