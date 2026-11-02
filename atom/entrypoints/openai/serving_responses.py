# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI Responses API adapter for ATOM, including Codex tool shapes.

Translates ``POST /v1/responses`` (and Codex's custom / namespace /
tool_search tools) to ATOM's internal chat format, then restores the
Responses wire shape on the way out. Same role as ``serving_anthropic.py``
for ``/v1/messages``.
"""

# 【译】ATOM的OpenAI Responses API适配器，包括Codex的工具形态。
#
#   把 POST /v1/responses（以及Codex的 custom / namespace / tool_search 工具）翻译成ATOM内部的chat格式，
#   返回时再还原成Responses的线上格式。它对 /v1/responses 的作用，与 serving_anthropic.py 对 /v1/messages 的作用相同。
#
# 【导读】/v1/responses（OpenAI Responses API，Codex CLI使用的协议）的协议适配层，只做格式转换，不接触引擎
# 调用方：api_server.py 的 responses_create 路由（@app.post("/v1/responses")）
# 与主干的关系：与 serving_anthropic.py 同构
#   请求方向：unsupported_responses_parameter 拒绝本服务不支持的参数（background / previous_response_id / conversation：
#     服务端不保存历史，对话历史必须放在input里）-> normalize_codex_request 把Codex特有的工具形态改写成普通function工具，
#     改写记录存进 CodexToolContext -> responses_to_openai_messages / responses_tools_to_openai 转成chat的messages和tools，
#     之后与 /v1/chat/completions 走同一条 chat template渲染 -> generate_async -> LLMEngine 的路径（本批）
#   响应方向：引擎输出同样先经推理内容分离（reasoning.py）和工具调用解析器（tool_parser/），得到有序事件
#     ("reasoning" / "content" / "tool_call_start" / "tool_call_args", 数据)
#     非流式：serving_anthropic.read_whole_blocks 产出事件 -> build_responses_response 拼成完整的response对象
#     流式：ResponseStreamEmitter 把事件变成 response.created -> response.output_item.added -> ...delta -> ...done ->
#       response.completed 这一串SSE事件，最后发 data: [DONE]
#   两个方向的出口都经过 CodexToolContext.rewrite_*，把function_call还原成Codex原来的 custom_tool_call / tool_search_call

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from .protocol import STREAM_DONE_MESSAGE
from .sse import event_frame

logger = logging.getLogger("atom")

DEFAULT_MAX_OUTPUT_TOKENS = 4096
_BACKGROUND_MESSAGE = (
    "Background mode is not supported. Please set 'background' to false or omit it."
)
_PREVIOUS_RESPONSE_MESSAGE = (
    "previous_response_id is not supported on this server. "
    "Include conversation history in 'input'."
)
_CONVERSATION_MESSAGE = (
    "conversation is not supported on this server. "
    "Include conversation history in 'input'."
)


# /v1/responses 的请求体模型；extra="allow" 让未声明的字段也能通过校验
class ResponsesRequest(BaseModel):
    """Loose Responses body. Tools/input stay untyped so Codex shapes parse."""

    # 【译】宽松的Responses请求体。tools / input 不做类型约束，这样Codex的各种形态都能解析通过。
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any = ""
    instructions: str | None = None
    stream: bool = False
    background: bool | None = None
    previous_response_id: str | None = None
    conversation: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Any = None
    stop: list[str] | None = None
    store: bool | None = None


# ── Codex tool-shape normalize / restore ────────────────────────────────
# 【译】── Codex工具形态的归一化 / 还原 ─────────────────────────────────────


# normalize_codex_request 留下的改写记录，响应时用它把function调用还原成Codex的形态
#   kinds：工具名 -> "custom" / "tool_search"（不在表里的就是普通function）
#   namespaces：工具名 -> 它原本所属的namespace分组名
#   original_tools：请求里的原始tools，响应的tools字段原样还原
@dataclass
class CodexToolContext:
    kinds: dict[str, str] = field(default_factory=dict)
    namespaces: dict[str, str] = field(default_factory=dict)
    original_tools: Any = None

    def is_empty(self) -> bool:
        return not self.kinds and self.original_tools is None

    def kind(self, name: str) -> str:
        return self.kinds.get(name, "function")

    def namespace(self, name: str) -> str | None:
        return self.namespaces.get(name)

    # 还原整个响应：tools字段换回原始tools，output里每个function_call逐个还原
    def rewrite_response(self, response: dict) -> None:
        if self.original_tools is not None:
            response["tools"] = copy.deepcopy(self.original_tools)
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    self.rewrite_output_item(item)

    # 还原一个输出项：custom工具 -> custom_tool_call（参数JSON里的input取回原始文本）；
    #   tool_search -> tool_search_call（execution固定为client）；普通function只补上namespace
    def rewrite_output_item(self, item: dict) -> None:
        if item.get("type") != "function_call":
            return
        name = item.get("name") or ""
        kind = self.kind(name)
        if kind == "custom":
            item["type"] = "custom_tool_call"
            item["input"] = custom_tool_input(item.get("arguments") or "")
            item.pop("arguments", None)
            namespace = self.namespace(name)
            if namespace:
                item["namespace"] = namespace
        elif kind == "tool_search":
            item["type"] = "tool_search_call"
            item["execution"] = "client"
            item["arguments"] = parse_tool_search_arguments(item.get("arguments") or "")
            item.pop("name", None)
        else:
            namespace = self.namespace(name)
            if namespace:
                item["namespace"] = namespace


# 请求校验：返回不支持的原因（作为400错误信息），支持则返回None
def unsupported_responses_parameter(body: dict) -> str | None:
    """Why this request cannot run on the Python Responses path, or None."""
    # 【译】说明这个请求为什么不能在Python的Responses路径上运行；能运行则返回None。
    if body.get("background") is True:
        return _BACKGROUND_MESSAGE
    if body.get("previous_response_id"):
        return _PREVIOUS_RESPONSE_MESSAGE
    if body.get("conversation"):
        return _CONVERSATION_MESSAGE
    return None


# 请求方向的第一步：先改写input里的历史项，再改写tools；顺序重要——
#   tool_search_output / additional_tools 携带的工具会被追加到all_tools里，随后一起参与工具归一化
def normalize_codex_request(body: dict) -> tuple[dict, CodexToolContext]:
    """Rewrite Codex tool/history shapes into function calls.

    Mutates and returns ``body``. Caller keeps ``CodexToolContext`` for the
    response rewrite; it is not stuffed into metadata the way the mesh path
    has to, because nothing here goes through a typed protocol crate.
    """
    # 【译】把Codex的工具 / 历史形态改写成function调用。
    #
    #   原地修改并返回 body。调用方保留 CodexToolContext 用于改写响应；它不像mesh路径那样必须塞进metadata，
    #   因为这里没有任何东西要经过带类型的协议crate。
    # 解释：mesh路径指 atomesh 入口（见 atomesh/ 与 atom/mesh 的Rust代码），那里的请求要经过Rust的强类型协议结构
    if not isinstance(body, dict):
        raise ValueError("Responses request must be a JSON object")  # noqa: TRY004

    original_tools = copy.deepcopy(body["tools"]) if "tools" in body else None
    context = CodexToolContext(original_tools=original_tools)
    all_tools = list(body["tools"]) if isinstance(body.get("tools"), list) else []

    if isinstance(body.get("input"), list):
        body["input"] = _normalize_input_items(body["input"], all_tools)

    normalized_tools = _normalize_tools(all_tools, context)
    if normalized_tools or "tools" in body:
        body["tools"] = normalized_tools

    return body, context


# 改写历史输入项：custom_tool_call / tool_search_call -> function_call，它们的输出 -> function_call_output；
#   tool_search_output 和 additional_tools 里带的工具追加到all_tools（additional_tools项本身不进历史）
def _normalize_input_items(items: list, all_tools: list) -> list:
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        item_type = item.get("type") or ""
        if item_type == "custom_tool_call":
            call_id = _string_field(item, "call_id")
            name = _string_field(item, "name")
            raw_input = item.get("input", "")
            input_text = raw_input if isinstance(raw_input, str) else ""
            normalized.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps({"input": input_text}),
                }
            )
        elif item_type == "custom_tool_call_output":
            normalized.append(
                {
                    "type": "function_call_output",
                    "id": item.get("id"),
                    "call_id": _string_field(item, "call_id"),
                    "output": _value_as_text(item.get("output")),
                }
            )
        elif item_type == "tool_search_call":
            call_id = _string_field(item, "call_id")
            normalized.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or call_id,
                    "call_id": call_id,
                    "name": "tool_search",
                    "arguments": _value_as_text(item.get("arguments")),
                }
            )
        elif item_type == "tool_search_output":
            tools = item.get("tools")
            if isinstance(tools, list):
                all_tools.extend(tools)
            normalized.append(
                {
                    "type": "function_call_output",
                    "id": item.get("id"),
                    "call_id": _string_field(item, "call_id"),
                    "output": json.dumps({"tools": tools if tools is not None else []}),
                }
            )
        elif item_type == "additional_tools":
            tools = item.get("tools")
            if isinstance(tools, list):
                all_tools.extend(tools)
        else:
            normalized.append(item)
    return normalized


# 改写工具列表：function原样保留；custom -> 只有一个字符串参数input的function；namespace -> 展开成子工具，
#   描述前加"[namespace]"；客户端执行的tool_search -> 名为tool_search的function；
#   web_search_preview / code_interpreter / mcp 原样保留；其余类型丢弃
def _normalize_tools(tools: list, context: CodexToolContext) -> list[dict]:
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            normalized.append(copy.deepcopy(tool))
        elif tool_type == "custom":
            name = tool.get("name")
            if isinstance(name, str):
                context.kinds[name] = "custom"
                normalized.append(_custom_as_function(tool, None))
        elif tool_type == "namespace":
            namespace = tool.get("name") or "namespace"
            for child in tool.get("tools") or []:
                if not isinstance(child, dict) or not isinstance(
                    child.get("name"), str
                ):
                    continue
                name = child["name"]
                context.namespaces[name] = namespace
                if child.get("type") == "function":
                    child = copy.deepcopy(child)
                    description = child.get("description") or ""
                    child["description"] = f"[{namespace}] {description}".rstrip()
                    normalized.append(child)
                elif child.get("type") == "custom":
                    context.kinds[name] = "custom"
                    normalized.append(_custom_as_function(child, namespace))
        elif tool_type == "tool_search" and tool.get("execution") == "client":
            context.kinds["tool_search"] = "tool_search"
            normalized.append(
                {
                    "type": "function",
                    "name": "tool_search",
                    "description": tool.get("description")
                    or "Search for and load deferred client tools.",
                    "parameters": tool.get("parameters") or {"type": "object"},
                }
            )
        elif tool_type in ("web_search_preview", "code_interpreter", "mcp"):
            normalized.append(copy.deepcopy(tool))
    return normalized


# custom工具 -> function工具：参数固定为 {"input": 字符串}；如果custom工具带grammar格式，把语法定义附加到描述里
def _custom_as_function(tool: dict, namespace: str | None) -> dict:
    description = tool.get("description") or ""
    if namespace:
        description = f"[{namespace}] {description}".rstrip()
    fmt = tool.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "grammar":
        syntax = fmt.get("syntax") or "grammar"
        definition = fmt.get("definition") or ""
        description += (
            f"\nReturn raw input matching this {syntax} grammar:\n{definition}"
        )
    return {
        "type": "function",
        "name": tool.get("name") or "",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw freeform input for this tool.",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        },
    }


# 与 _custom_as_function 相反：从模型给出的参数JSON里取出 input 字段，还原成custom工具的原始文本
def custom_tool_input(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    if isinstance(parsed, dict):
        value = parsed.get("input")
        return value if isinstance(value, str) else arguments
    if isinstance(parsed, str):
        return parsed
    return arguments


# tool_search的参数：能解析成JSON就用JSON，否则把原文当作 {"query": 原文}
def parse_tool_search_arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"query": arguments}


# 取字典里的字符串字段，不是字符串就返回空串
def _string_field(value: dict, key: str) -> str:
    field = value.get(key)
    return field if isinstance(field, str) else ""


# 任意值 -> 文本：None -> ""，字符串原样，其他尝试json.dumps
def _value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


# ── Responses → internal chat ───────────────────────────────────────────
# 【译】── Responses -> 内部chat格式 ───────────────────────────────────────


# Responses的工具是扁平的 {type, name, parameters}，chat template要的是嵌套的 {type, function: {...}}；
#   只保留type为function的工具，web_search_preview等内置工具不进chat template
def responses_tools_to_openai(tools: list | None) -> list[dict] | None:
    """Function tools in the nested shape ``apply_chat_template`` expects."""
    # 【译】把function工具转成 apply_chat_template 期望的嵌套形态。
    if not tools:
        return None
    result = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            result.append({"type": "function", "function": fn})
        elif isinstance(tool.get("name"), str):
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters") or {},
                    },
                }
            )
    return result or None


# instructions -> system消息；input是字符串时作为一条user消息；是列表时逐项转换：
#   message -> 对应role的消息，function_call -> 带tool_calls的assistant消息，
#   function_call_output -> role=tool的消息，reasoning -> 带reasoning_content的assistant消息
def responses_to_openai_messages(request: ResponsesRequest) -> list[dict]:
    """Convert a (already-normalized) Responses request to OpenAI messages."""
    # 【译】把一个（已经归一化过的）Responses请求转换成OpenAI消息。
    messages: list[dict] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    raw_input = request.input
    if isinstance(raw_input, str) or raw_input is None:
        messages.append({"role": "user", "content": raw_input or ""})
        return messages
    if not isinstance(raw_input, list):
        raise ValueError("input must be a string or an array")  # noqa: TRY004

    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in (None, "message") and "role" in item:
            messages.append(
                _role_message(
                    item.get("role") or "user", _content_text(item.get("content"))
                )
            )
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "",
                            },
                        }
                    ],
                }
            )
            if item.get("output") is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _value_as_text(item.get("output")),
                    }
                )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": _value_as_text(item.get("output")),
                }
            )
        elif item_type == "reasoning":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": _reasoning_text(item.get("content")),
                }
            )
        elif "role" in item:
            messages.append(
                _role_message(
                    item.get("role") or "user", _content_text(item.get("content"))
                )
            )
    if not messages:
        raise ValueError("Request must contain at least one message")
    return messages


# 只认 user / assistant / system 三种role，其他（如developer）一律当作user
def _role_message(role: str, text: str) -> dict:
    if role in ("user", "assistant", "system"):
        return {"role": role, "content": text}
    return {"role": "user", "content": text}


# 把content部件列表里的文本类部件（input_text / output_text / text）拼成一个字符串
def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (
                "input_text",
                "output_text",
                "text",
                None,
            ):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


# 把reasoning项的内容部件（reasoning_text / summary_text）用换行拼起来
def _reasoning_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("reasoning_text", "summary_text", None)
        )
    return ""


# ── Non-streaming response ──────────────────────────────────────────────
# 【译】── 非流式响应 ─────────────────────────────────────────────────────


# 非流式出口：有序事件 -> 完整的response对象（output、usage，以及回显请求里的参数），最后做Codex还原
def build_responses_response(
    request_id: str,
    model: str,
    events: list,
    *,
    context: CodexToolContext,
    request: ResponsesRequest,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    created_at: int | None = None,
) -> dict:
    """One Responses object from ordered engine events, then Codex rewrite."""
    # 【译】由有序的引擎事件构建一个Responses对象，然后做Codex改写。
    output = _output_items_from_events(events, request_id, context)
    response = {
        "id": request_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cache_read_input_tokens},
        },
        "parallel_tool_calls": (
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        "store": False,
        "tools": request.tools or [],
        "metadata": request.metadata or {},
        "tool_choice": (
            request.tool_choice if request.tool_choice is not None else "auto"
        ),
    }
    if request.instructions is not None:
        response["instructions"] = request.instructions
    if request.max_output_tokens is not None:
        response["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        response["temperature"] = request.temperature
    if request.top_p is not None:
        response["top_p"] = request.top_p
    context.rewrite_response(response)
    return response


# 事件 -> output项列表：reasoning事件各成一个reasoning项；相邻的content合并进同一个message项；
#   tool_call_start只记下id和name，等tool_call_args到来时才生成function_call项
def _output_items_from_events(
    events: list, request_id: str, context: CodexToolContext
) -> list[dict]:
    output: list[dict] = []
    pending: dict | None = None
    for etype, data in events:
        if not data and etype in ("reasoning", "content"):
            continue
        if etype == "reasoning":
            output.append(
                {
                    "id": f"reasoning_{request_id}",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": data}],
                    "status": "completed",
                }
            )
        elif etype == "content":
            if output and output[-1].get("type") == "message":
                parts = output[-1].setdefault("content", [])
                if parts and parts[-1].get("type") in ("output_text", "text"):
                    parts[-1]["text"] += data
                else:
                    parts.append({"type": "output_text", "text": data})
            else:
                output.append(
                    {
                        "id": f"msg_{request_id}",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": data}],
                        "status": "completed",
                    }
                )
        elif etype == "tool_call_start":
            fn = data.get("function") or {}
            pending = {
                "id": data.get("id") or "",
                "name": fn.get("name") or "",
            }
        elif etype == "tool_call_args" and pending is not None:
            item = {
                "id": pending["id"],
                "type": "function_call",
                "call_id": pending["id"],
                "name": pending["name"],
                "arguments": (data.get("function") or {}).get("arguments") or "",
                "status": "completed",
            }
            context.rewrite_output_item(item)
            output.append(item)
            pending = None
    return output


# ── Streaming ───────────────────────────────────────────────────────────
# 【译】── 流式输出 ───────────────────────────────────────────────────────


# 流式中一个工具调用的状态：output_index / item_id 在输出项被"added"时分配，参数逐段累加到accumulated_args
@dataclass
class _ToolCallState:
    output_index: int = -1
    item_id: str = ""
    call_id: str = ""
    name: str = ""
    accumulated_args: str = ""
    added: bool = False


# 流式出口：把有序事件翻译成Responses的SSE事件序列；每个事件带一个递增的sequence_number
# 关键行为：
#   推理内容先累积在reasoning_buffer里，等content或工具调用开始、或流结束时，才作为一个完整的reasoning项一次发出（added + done）
#   文本：第一次出现时发 output_item.added（message）和 content_part.added，之后每段发 output_text.delta
#   工具调用：call_id和name都已知时才发 output_item.added；普通function的参数逐段发 function_call_arguments.delta，
#     custom / tool_search 的输入要等结束时才能还原，所以只在 finish_frames 里一次性发出
class ResponseStreamEmitter:
    """Responses SSE sequence, including Codex custom / tool_search shapes."""

    # 【译】Responses的SSE事件序列，包括Codex的 custom / tool_search 形态。
    def __init__(
        self,
        response_id: str,
        model: str,
        *,
        context: CodexToolContext | None = None,
        request: ResponsesRequest | None = None,
        created_at: int | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.created_at = created_at if created_at is not None else int(time.time())
        self.context = context or CodexToolContext()
        self.request = request
        self.sequence_number = 0
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.accumulated_text = ""
        self.reasoning_buffer = ""
        self.has_emitted_output_item_added = False
        self.has_emitted_content_part_added = False
        self.current_message_output_index: int | None = None
        self.current_item_id: str | None = None
        self.next_output_index = 0
        self.completed_items: list[dict] = []
        self.tool_call_items: list[_ToolCallState] = []

    # 开场两帧：response.created 和 response.in_progress
    def opening_frames(self) -> list[str]:
        return [
            self._event(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": self._next_sequence(),
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "in_progress",
                        "model": self.model,
                        "output": [],
                    },
                },
            ),
            self._event(
                "response.in_progress",
                {
                    "type": "response.in_progress",
                    "sequence_number": self._next_sequence(),
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "status": "in_progress",
                    },
                },
            ),
        ]

    # 一批引擎事件 -> 若干SSE帧（调用方：api_server.py 的流式循环，process和flush的结果都走这里）
    def frames_for_events(self, events: list) -> list[str]:
        frames: list[str] = []
        for etype, data in events:
            frames.extend(self._on_event(etype, data))
        return frames

    # 结束：发出缓存的推理内容，收尾工具调用或文本消息，然后发 response.completed 和 data: [DONE]
    def finish_frames(
        self,
        *,
        usage: dict | None = None,
        has_tool_calls: bool = False,
    ) -> list[str]:
        frames: list[str] = []
        frames.extend(self._flush_reasoning())
        if has_tool_calls or any(state.added for state in self.tool_call_items):
            frames.extend(self._finish_tool_calls())
        else:
            frames.extend(self._finish_message())
        frames.append(self._completed_frame(usage))
        frames.append(STREAM_DONE_MESSAGE)
        return frames

    # error事件：流中途失败时使用，见文件末尾的 stream_failure_frames
    def error_frame(self, message: str, error_type: str = "api_error") -> str:
        return self._event(
            "error",
            {
                "type": "error",
                "code": error_type,
                "message": message,
                "param": None,
                "sequence_number": self._next_sequence(),
            },
        )

    # 按事件类型分派；content或工具调用开始前，先把缓存的推理内容作为一个reasoning项发出去
    def _on_event(self, etype: str, data: Any) -> list[str]:
        if etype == "reasoning":
            if not data:
                return []
            self.reasoning_buffer += data
            return []
        if etype == "content":
            if not data:
                return []
            frames = self._flush_reasoning()
            frames.extend(self._on_text(data))
            return frames
        if etype == "tool_call_start":
            frames = self._flush_reasoning()
            frames.extend(self._close_message_if_open())
            frames.extend(self._on_tool_start(data or {}))
            return frames
        if etype == "tool_call_args":
            return self._on_tool_args(data or {})
        return []

    # 文本事件：必要时先打开message项和content part，再发 output_text.delta
    def _on_text(self, text: str) -> list[str]:
        frames: list[str] = []
        if self.current_item_id is None:
            output_index, item_id = self._allocate("msg")
            item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [],
            }
            frames.append(self._output_item_added(output_index, item))
            self.has_emitted_output_item_added = True
            self.current_item_id = item_id
            self.current_message_output_index = output_index
        output_index = self.current_message_output_index
        item_id = self.current_item_id
        assert output_index is not None and item_id is not None
        if not self.has_emitted_content_part_added:
            frames.append(
                self._event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {"type": "text", "text": ""},
                    },
                )
            )
            self.has_emitted_content_part_added = True
        self.accumulated_text += text
        frames.append(
            self._event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_sequence(),
                    "output_index": output_index,
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        return frames

    # 工具调用开始：按解析器给的index找到（或补齐）状态槽，记下id和name
    def _on_tool_start(self, data: dict) -> list[str]:
        index = int(data.get("index") or 0)
        while len(self.tool_call_items) <= index:
            self.tool_call_items.append(_ToolCallState())
        state = self.tool_call_items[index]
        fn = data.get("function") or {}
        if data.get("id"):
            state.call_id = data["id"]
        if fn.get("name"):
            state.name = fn["name"]
        return self._maybe_add_tool_item(state)

    # 工具参数：累加；普通function的参数在项已added后立即作为delta发出
    def _on_tool_args(self, data: dict) -> list[str]:
        index = int(data.get("index") or 0)
        while len(self.tool_call_items) <= index:
            self.tool_call_items.append(_ToolCallState())
        state = self.tool_call_items[index]
        args = (data.get("function") or {}).get("arguments") or ""
        if args:
            state.accumulated_args += args
        frames = self._maybe_add_tool_item(state)
        if state.added and self.context.kind(state.name) == "function" and args:
            frames.append(
                self._event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": self._next_sequence(),
                        "output_index": state.output_index,
                        "item_id": state.item_id,
                        "delta": args,
                    },
                )
            )
        return frames

    # 第一次同时拿到call_id和name时，分配output_index并发 output_item.added；
    #   id前缀按种类区分：ctc（custom）/ tsc（tool_search）/ fc（function）；已经累积的参数一并补发
    def _maybe_add_tool_item(self, state: _ToolCallState) -> list[str]:
        if state.added or not state.call_id or not state.name:
            return []
        kind = self.context.kind(state.name)
        prefix = {"custom": "ctc", "tool_search": "tsc"}.get(kind, "fc")
        output_index, item_id = self._allocate(prefix)
        state.output_index = output_index
        state.item_id = item_id
        state.added = True
        if kind == "custom":
            item = {
                "id": item_id,
                "type": "custom_tool_call",
                "call_id": state.call_id,
                "name": state.name,
                "input": "",
                "status": "in_progress",
            }
        elif kind == "tool_search":
            item = {
                "id": item_id,
                "type": "tool_search_call",
                "call_id": state.call_id,
                "execution": "client",
                "arguments": {},
                "status": "in_progress",
            }
        else:
            item = {
                "id": item_id,
                "type": "function_call",
                "call_id": state.call_id,
                "name": state.name,
                "arguments": "",
                "status": "in_progress",
            }
        namespace = self.context.namespace(state.name)
        if namespace:
            item["namespace"] = namespace
        frames = [self._output_item_added(output_index, item)]
        if kind == "function" and state.accumulated_args:
            frames.append(
                self._event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": state.accumulated_args,
                    },
                )
            )
        return frames

    # 结束所有工具调用：先关闭打开的message项；custom还原原始输入并发 custom_tool_call_input.delta/done，
    #   function发 function_call_arguments.done；每个调用最后发 output_item.done；缺id或name的调用被跳过
    def _finish_tool_calls(self) -> list[str]:
        frames: list[str] = []
        frames.extend(self._close_message_if_open())
        for state in self.tool_call_items:
            if not state.added:
                logger.warning(
                    "Skipping incomplete streamed tool call without id or name"
                )
                continue
            kind = self.context.kind(state.name)
            namespace = self.context.namespace(state.name)
            if kind == "custom":
                input_text = custom_tool_input(state.accumulated_args)
                if input_text:
                    frames.append(
                        self._event(
                            "response.custom_tool_call_input.delta",
                            {
                                "type": "response.custom_tool_call_input.delta",
                                "sequence_number": self._next_sequence(),
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "call_id": state.call_id,
                                "delta": input_text,
                            },
                        )
                    )
                frames.append(
                    self._event(
                        "response.custom_tool_call_input.done",
                        {
                            "type": "response.custom_tool_call_input.done",
                            "sequence_number": self._next_sequence(),
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "call_id": state.call_id,
                            "input": input_text,
                        },
                    )
                )
                item = {
                    "id": state.item_id,
                    "type": "custom_tool_call",
                    "call_id": state.call_id,
                    "name": state.name,
                    "input": input_text,
                    "status": "completed",
                }
            elif kind == "tool_search":
                item = {
                    "id": state.item_id,
                    "type": "tool_search_call",
                    "call_id": state.call_id,
                    "execution": "client",
                    "arguments": parse_tool_search_arguments(state.accumulated_args),
                    "status": "completed",
                }
            else:
                frames.append(
                    self._event(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "sequence_number": self._next_sequence(),
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "arguments": state.accumulated_args,
                        },
                    )
                )
                item = {
                    "id": state.item_id,
                    "type": "function_call",
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": state.accumulated_args,
                    "status": "completed",
                }
            if namespace:
                item["namespace"] = namespace
            frames.append(self._output_item_done(state.output_index, item))
        return frames

    # 结束文本消息：output_text.done -> content_part.done -> output_item.done
    def _finish_message(self) -> list[str]:
        frames: list[str] = []
        if self.current_item_id is None:
            return frames
        output_index = self.current_message_output_index
        item_id = self.current_item_id
        assert output_index is not None and item_id is not None
        if self.has_emitted_content_part_added:
            frames.append(
                self._event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "text": self.accumulated_text,
                    },
                )
            )
            frames.append(
                self._event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "sequence_number": self._next_sequence(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": 0,
                        "part": {"type": "text", "text": self.accumulated_text},
                    },
                )
            )
        if self.has_emitted_output_item_added:
            item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.accumulated_text}],
                "status": "completed",
            }
            frames.append(self._output_item_done(output_index, item))
        self.current_item_id = None
        return frames

    # 工具调用开始前，如果有打开的message项，先把它结束掉
    def _close_message_if_open(self) -> list[str]:
        if self.current_item_id is None:
            return []
        return self._finish_message()

    # 把缓存的推理内容作为一个完整的reasoning项发出：一次 output_item.added 加一次 output_item.done
    def _flush_reasoning(self) -> list[str]:
        if not self.reasoning_buffer:
            return []
        text = self.reasoning_buffer
        self.reasoning_buffer = ""
        output_index, item_id = self._allocate("rs")
        item = {
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
            "status": "completed",
        }
        return [
            self._output_item_added(output_index, item),
            self._output_item_done(output_index, item),
        ]

    # response.completed：output取所有已done的项（一个都没有时用累积的文本兜底），并回显请求参数，最后做Codex还原
    def _completed_frame(self, usage: dict | None) -> str:
        output = list(self.completed_items)
        if not output:
            output = [
                {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.accumulated_text}],
                }
            ]
        response_obj: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output,
            "store": False,
            "parallel_tool_calls": True,
        }
        if usage is not None:
            response_obj["usage"] = usage
        req = self.request
        if req is not None:
            if req.instructions is not None:
                response_obj["instructions"] = req.instructions
            if req.max_output_tokens is not None:
                response_obj["max_output_tokens"] = req.max_output_tokens
            if req.temperature is not None:
                response_obj["temperature"] = req.temperature
            if req.top_p is not None:
                response_obj["top_p"] = req.top_p
            response_obj["parallel_tool_calls"] = (
                True if req.parallel_tool_calls is None else req.parallel_tool_calls
            )
            response_obj["tools"] = req.tools or []
            response_obj["metadata"] = req.metadata or {}
            response_obj["tool_choice"] = (
                req.tool_choice if req.tool_choice is not None else "auto"
            )
        self.context.rewrite_response(response_obj)
        return self._event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": self._next_sequence(),
                "response": response_obj,
            },
        )

    # 分配下一个output_index，并生成 "<前缀>_<uuid>" 形式的项id
    def _allocate(self, prefix: str) -> tuple[int, str]:
        index = self.next_output_index
        self.next_output_index += 1
        return index, f"{prefix}_{uuid.uuid4().hex}"

    # response.output_item.added 帧
    def _output_item_added(self, output_index: int, item: dict) -> str:
        return self._event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": self._next_sequence(),
                "output_index": output_index,
                "item": item,
            },
        )

    # response.output_item.done 帧；同时把这一项存进completed_items，供 response.completed 使用
    def _output_item_done(self, output_index: int, item: dict) -> str:
        self.completed_items.append(copy.deepcopy(item))
        return self._event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": self._next_sequence(),
                "output_index": output_index,
                "item": item,
            },
        )

    def _event(self, event: str, payload: dict) -> str:
        return event_frame(event, payload)

    def _next_sequence(self) -> int:
        seq = self.sequence_number
        self.sequence_number += 1
        return seq


# 流式中途抛异常时的收尾帧：[还没发开场帧就补发] -> error -> data: [DONE]（调用方：api_server.py）
def stream_failure_frames(
    exc: BaseException, emitter: ResponseStreamEmitter, *, opened: bool
):
    if not opened:
        yield from emitter.opening_frames()
    yield emitter.error_frame(str(exc))
    yield STREAM_DONE_MESSAGE


# 这一批事件是否产出了一个可用的工具调用（以参数而不是名字为准，含义同 serving_anthropic.completes_a_tool_call）
# 注意：api_server.py 导入并使用的是 serving_anthropic 里的同名函数，本文件这个版本在 atom/ 内没有调用方
def completes_a_tool_call(events) -> bool:
    return any(etype == "tool_call_args" for etype, _ in events)
