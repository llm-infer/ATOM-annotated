"""Request-local first-output timing for the HTTP API."""

# 【译】HTTP API中按请求本地记录的首个输出耗时。
#
# 【导读】TTFT（time to first token，首个输出时间）的测量：从API进程收到请求算起，到第一个真正的生成内容发出为止
# 机制：RequestTimingMiddleware是一个ASGI中间件（ASGI是什么见api_server.py文件头"几个第一次出现的概念"中的uvicorn一条），
#   只拦截四个生成接口的POST请求，把RequestTiming放进ContextVar（每个请求独立）；
#   流式：api_server的流包装函数用get_stream_timing()取出它，在第一个has_generated_output()为真的SSE事件处记一次
#   非流式：api_server在收到第一个内部token时调用record_nonstream_first_token()
# 调用方：api_server.py（add_middleware注册中间件，observe_ttft=RequestMetrics.observe_time_to_first_token）
# 与主干的关系：服务层的观测代码，不影响请求处理；【分支】指标见第10批

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


# fields是dict，且keys中任一字段是非空字符串
def _has_text(fields: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(fields, dict) and any(
        isinstance(fields.get(key), str) and bool(fields[key]) for key in keys
    )


# 判断一个SSE事件payload是否带有生成内容（OpenAI chat/completions、Anthropic messages、Responses三种事件格式）；
#   只有role的首帧、空delta、错误事件都不算
def has_generated_output(payload: Any) -> bool:
    """Recognize text, reasoning or tool output in an API generation event."""
    # 【译】识别API生成事件中的文本、推理或工具调用输出。
    if not isinstance(payload, dict) or "error" in payload:
        return False
    choices = payload.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if not isinstance(choice, dict):
            continue
        if _has_text(choice, ("text",)):
            return True
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        if _has_text(delta, ("content", "reasoning_content", "reasoning")):
            return True
        calls = delta.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        functions = [
            call.get("function", {}) for call in calls if isinstance(call, dict)
        ]
        functions.append(delta.get("function_call") or {})
        if any(_has_text(f, ("name", "arguments")) for f in functions):
            return True
    # Anthropic messages emit generation in content blocks, not choices.
    # 【译】Anthropic messages把生成内容放在content块里，而不是choices里。
    if payload.get("type") == "content_block_delta":
        delta = payload.get("delta") or {}
        return _has_text(delta, ("text", "thinking", "partial_json"))
    if payload.get("type") == "content_block_start":
        block = payload.get("content_block") or {}
        return _has_text(block, ("text", "thinking", "name"))
    # OpenAI Responses / Codex events carry generation in `delta`.
    # 【译】OpenAI Responses / Codex的事件把生成内容放在`delta`里。
    if payload.get("type") in (
        "response.output_text.delta",
        "response.custom_tool_call_input.delta",
        "response.function_call_arguments.delta",
    ):
        delta = payload.get("delta")
        return isinstance(delta, str) and bool(delta)
    return False


# 一个请求的计时状态：started_at是perf_counter起点；recorded保证只记一次；streaming_response由中间件根据响应头设置
@dataclass
class RequestTiming:
    started_at: float
    observe: Callable[[float, bool], None]
    recorded: bool = False
    streaming_response: bool = False

    # 只在第一次调用时把(耗时, 是否流式)交给observe
    def first_output(self, *, streaming: bool) -> None:
        if not self.recorded:
            self.recorded = True
            self.observe(time.perf_counter() - self.started_at, streaming)


# 当前请求的RequestTiming：放在ContextVar里，中间件设置、同一请求的处理代码读取，无需层层传参
_request_timing: ContextVar[RequestTiming | None] = ContextVar(
    "request_timing", default=None
)


# 流包装函数调用：只有本请求是成功的SSE响应且还没记录过时才返回计时对象，否则返回None
def get_stream_timing() -> RequestTiming | None:
    """Return timing for a successful SSE response, after response headers."""
    # 【译】对一个成功的SSE响应，在响应头发出之后返回它的计时对象。
    timing = _request_timing.get()
    if timing is not None and timing.streaming_response and not timing.recorded:
        return timing
    return None


# 非流式路径调用：第一个内部token到达时记一次（若已记录过则什么也不做）
def record_nonstream_first_token() -> None:
    """Record an internal token arrival; a buffered response has no SSE event."""
    # 【译】记录一次内部token到达；缓冲式响应没有SSE事件可供观察。
    timing = _request_timing.get()
    if timing is not None:
        timing.first_output(streaming=False)


# 纯ASGI中间件（直接实现__call__(scope, receive, send)），只对四个生成接口的POST请求计时，其余请求原样放行
class RequestTimingMiddleware:
    """Start before preprocessing and mark successful SSE responses.

    The client stream wrapper observes generated output; this middleware only
    owns request lifetime and response eligibility, without inspecting bodies.
    """

    # 【译】在预处理之前开始计时，并标记成功的SSE响应。
    #
    #   客户端流包装函数负责观察生成的输出；这个中间件只负责请求的生命周期和响应是否符合条件，不检查响应体。

    def __init__(self, app, observe_ttft: Callable[[float, bool], None]):
        self.app = app
        self.observe_ttft = observe_ttft

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/messages",
                "/v1/responses",
            }
        ):
            await self.app(scope, receive, send)
            return

        # 计时起点在应用处理请求之前，因此TTFT包含prompt预处理（模板渲染、分词、多模态加载）的时间
        timing = RequestTiming(time.perf_counter(), self.observe_ttft)
        context_token = _request_timing.set(timing)

        # 包一层send：在响应头（http.response.start）经过时，2xx且content-type为text/event-stream才标记为可记录的流式响应
        async def timed_send(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                is_sse = headers.get(b"content-type", b"").startswith(
                    b"text/event-stream"
                )
                timing.streaming_response = 200 <= message["status"] < 300 and is_sse
            await send(message)

        # finally里reset：请求结束后恢复ContextVar，计时对象不会泄漏到其他请求
        try:
            await self.app(scope, receive, timed_send)
        finally:
            _request_timing.reset(context_token)
