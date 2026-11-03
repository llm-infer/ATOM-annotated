# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Text completion handler for the OpenAI-compatible API."""

# 【译】OpenAI兼容API的文本补全处理器。
#
# 本文件（第2批2c，B级）：/v1/completions 的响应构造（接口函数在api_server.completions）
#   结构与serving_chat.py相同，但更简单：没有chat template、没有推理内容分离、没有工具调用解析，
#   生成的文本原样放进choices[i].text
#   流式：stream_completion_response（n=1）、stream_completion_response_fanout（n>1）
#   非流式：build_completion_response / build_completion_response_multi
# 原理：completions接口不套chat template，prompt原样交给模型续写——适合base模型，或客户端自己已经拼好模板的场景
#   （如PD路由器、评测框架）。对chat模型直接发裸文本，等于让它在没有角色标记的格式上生成，质量通常明显变差
#   （角色标记、特殊token与chat template的科普见chat_encoders.py文件头）
# 流式帧序列：若干 {"choices": [{"text": "...", "finish_reason": null}]}
#   → 最后一次发送里：最后一段文本帧 + 带finish_reason的空文本帧 + usage帧 + [DONE]
#
# ---- 适用场景：prompt的三种给法（CompletionRequest.get_prompt_or_tokens的优先级从上到下）----
#   给法                                 服务端做什么                 适用场景
#   ───────────────────────────────────  ───────────────────────────  ─────────────────────────────────────────────
#   prompt_token_ids（或kv_transfer_      直接用，不tokenize           PD路由器、RL rollout：上游已有token id，避免两边
#     params["prompt_token_ids"]）                                   tokenizer不一致；[]或两处不一致 → 400
#   "prompt": [151644, 872, …]（数组）    同上，走预tokenize路径        OpenAI规范里的token数组形式；不支持二维批量
#   "prompt": "Question: …\nAnswer:"     preprocess里tokenize          base模型续写、lm_eval（local-completions）的few-shot
# ---- 与/v1/chat/completions的取舍 ----
#   ┌───────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐
#   │                       │ /v1/completions（本文件）           │ /v1/chat/completions                │
#   ├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
#   │ chat template         │ 不套，prompt原样续写                │ 服务端渲染                          │
#   │ 推理内容 / 工具调用    │ 不拆分，全部在choices[i].text里     │ 拆成reasoning_content / tool_calls  │
#   │ 多模态                │ 不支持                              │ 支持（content片段列表）             │
#   │ request_id前缀        │ cmpl-                               │ chatcmpl-                           │
#   │ 适用场景              │ base模型、评测、压测、自带模板的上游 │ 对话 / 指令模型的常规调用            │
#   └───────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘
#   好处：路径最短、结果可预测（服务端不改动prompt和输出）；代价：对chat模型要由调用方自己保证格式正确
# 例：{"prompt": "1+1=", "max_tokens": 4, "temperature": 0, "stream": true}
#   → 假设逐step产出" 2"、"\n"、"1"、"+"，第4个token时到达max_tokens（leave_reason="max_tokens"）
#   → 帧：{"text": " 2"} {"text": "\n"} {"text": "1"}，最后一次发送：{"text": "+"} + {"text": "", "finish_reason": "length"}
#     + usage帧 + [DONE]——"length"告诉客户端输出被截断，而不是自然结束
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from .protocol import (
    STREAM_DONE_MESSAGE,
    TEXT_COMPLETION_OBJECT,
    CompletionResponse,
    openai_stop_reason,
)
from .sse import data_frame
from .streaming_dispatch import StreamOutputCollector

logger = logging.getLogger("atom")


# 构造一帧text_completion并编码成SSE；extra_fields会加到帧的顶层（例如PD分离的kv_transfer_params）
def create_completion_chunk(
    request_id: str,
    model: str,
    text: str,
    finish_reason: str | None = None,
    usage: dict | None = None,
    index: int = 0,
    **extra_fields: Any,
) -> str:
    """Create a text completion chunk in SSE format.

    ``index`` selects ``choices[0].index``; fan-out siblings share one SSE
    stream and are distinguished by this field.

    ``finish_reason`` belongs on the terminal chunk and nowhere else, already
    translated by `openai_stop_reason`. Both streaming paths here used to pass
    the engine's own word on every *content* chunk while hardcoding `"stop"`
    on the terminal one, so a single response reported `max_tokens` in a place
    clients do not read and `stop` in the place they do.
    """
    # 【译】创建一个SSE格式的文本补全chunk。
    #
    #   index 决定 choices[0].index；扇出的兄弟序列共用一个SSE流，靠这个字段区分。
    #
    #   finish_reason 只属于结束chunk，不出现在别处，并且已经由 openai_stop_reason 翻译过。这里的两条流式路径以前在每个
    #   "内容"chunk上传引擎自己的词，却在结束chunk上硬编码 "stop"，于是同一个响应在客户端不读的地方报告 max_tokens，
    #   在客户端读的地方报告 stop。
    chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "text": text,
                "finish_reason": finish_reason,
                # 恒为None：HTTP接口不支持logprobs（见api_server._build_sampling_params前的说明）
                "logprobs": None,
            }
        ],
    }
    chunk.update(extra_fields)
    if usage is not None:
        chunk["usage"] = usage
    return data_frame(chunk)


# 流式文本补全（n=1）。调用方：api_server.completions，交给StreamingResponse(_client_stream(gen))
# 每个chunk产出一帧文本；finished时把最后一帧文本、结束帧、usage帧和[DONE]拼在一起发出；finally里清理
async def stream_completion_response(
    request_id: str,
    model: str,
    stream_collector: StreamOutputCollector,
    seq_id: int,
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
) -> AsyncGenerator[str, None]:
    """Generate streaming text completion response.

    ``num_prompt_tokens`` is the engine-computed prompt length (``Sequence.
    num_prompt_tokens``); reusing it avoids re-tokenizing the prompt on the
    event loop at stream start.
    """
    # 【译】生成流式的文本补全响应。
    #
    #   num_prompt_tokens 是引擎算出的prompt长度（Sequence.num_prompt_tokens）；复用它可以避免在流开始时
    #   在事件循环上重新tokenize prompt。
    num_tokens_input = num_prompt_tokens
    num_tokens_output = 0

    # Assume abort until the engine's finished chunk arrives. On client
    # disconnect the generator is closed (GeneratorExit) before we get there,
    # so the finally below aborts the still-running seq; on normal completion
    # we flip this to False and skip the (no-op) abort.
    # 【译】在引擎的finished chunk到达之前都假定是abort。客户端断开时，生成器会在那之前被关闭（GeneratorExit），
    #   于是下面的finally会abort仍在运行的序列；正常结束时我们把它改成False，跳过（本来就是空操作的）abort。
    aborted = True
    # The engine's own word, kept for the terminal chunk. It arrives on the
    # chunk that reports `finished`, which is not the frame it belongs on.
    # 【译】引擎自己的词，留给结束chunk用。它随报告 finished 的那个chunk到达，而那一帧并不是它该出现的地方。
    engine_reason: str | None = None
    try:
        while True:
            # chunk_data：{"text": 已反分词的新增文本, "token_ids": [...], "finished": ..., "finish_reason": ..., ...}
            chunk_data = await stream_collector.get()
            new_text = chunk_data["text"]
            num_tokens_output += len(chunk_data.get("token_ids", []))
            engine_reason = chunk_data.get("finish_reason") or engine_reason

            extra_fields: dict[str, Any] = {}
            if "kv_transfer_params" in chunk_data:
                extra_fields["kv_transfer_params"] = chunk_data["kv_transfer_params"]

            content_chunk = create_completion_chunk(
                request_id,
                model,
                new_text,
                # 【译】（行尾注释）由结束chunk携带它
                finish_reason=None,  # the terminal chunk carries it
                **extra_fields,
            )

            if chunk_data.get("finished", False):
                # 正常结束：引擎已经结束了这条序列，不需要abort
                aborted = False
                # Coalesce the finalization SSE messages (content + stop + usage
                # + [DONE]) into a single send. At a wave boundary many requests
                # finish simultaneously; collapsing 4 sends/req to 1 cuts the
                # per-request socket-write syscalls that saturate the API loop.
                # 【译】把收尾的几条SSE消息（content + stop + usage + [DONE]）合并成一次发送。在一波请求的边界上很多请求同时结束；
                #   把每个请求4次发送合成1次，能减少那些把API事件循环打满的逐请求socket写系统调用。
                # 注意：这里的usage没有prompt_tokens_details（前缀缓存命中数），chat路径有
                usage_chunk = {
                    "id": request_id,
                    "object": TEXT_COMPLETION_OBJECT,
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": num_tokens_input,
                        "completion_tokens": num_tokens_output,
                        "total_tokens": num_tokens_input + num_tokens_output,
                    },
                }
                yield (
                    content_chunk
                    + create_completion_chunk(
                        request_id,
                        model,
                        "",
                        openai_stop_reason(engine_reason) or "stop",
                    )
                    + data_frame(usage_chunk)
                    + STREAM_DONE_MESSAGE
                )
                return

            yield content_chunk
    finally:
        # 正常结束时aborted为False，只做登记清理；客户端断开时为True，cleanup_stream会向引擎发abort
        cleanup_stream(seq_id, aborted=aborted)
        cleanup_request(request_id)


# 非流式、n=1：把generate_async的结果dict组装成CompletionResponse；finish_reason经openai_stop_reason翻译
def build_completion_response(
    request_id: str,
    model: str,
    final_output: dict[str, Any],
) -> CompletionResponse:
    """Build a non-streaming text completion response (single choice)."""
    # 【译】构造一个非流式的文本补全响应（单个choice）。
    response = CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "text": final_output["text"],
                "finish_reason": openai_stop_reason(final_output["finish_reason"]),
            }
        ],
        usage={
            "prompt_tokens": final_output["num_tokens_input"],
            "completion_tokens": final_output["num_tokens_output"],
            "total_tokens": final_output["num_tokens_input"]
            + final_output["num_tokens_output"],
            "ttft_s": round(final_output.get("ttft", 0.0), 4),
            "tpot_s": round(final_output.get("tpot", 0.0), 4),
            "latency_s": round(final_output.get("latency", 0.0), 4),
        },
    )
    update: dict[str, Any] = {}
    if "kv_transfer_output_meta_info" in final_output:
        update["kv_transfer_params"] = final_output["kv_transfer_output_meta_info"]
    if "prompt_token_ids" in final_output:
        update["prompt_token_ids"] = final_output["prompt_token_ids"]
    if update:
        response = response.model_copy(update=update)
    return response


# 非流式、n>1：每个兄弟一个choice；usage里completion token数求和，ttft/tpot/latency取各兄弟的最大值
def build_completion_response_multi(
    request_id: str,
    model: str,
    final_outputs: list[dict[str, Any]],
) -> CompletionResponse:
    """Build a non-streaming response with one choice per fan-out sibling."""
    # 【译】构造一个非流式响应，每个扇出的兄弟序列一个choice。
    assert final_outputs, "build_completion_response_multi requires at least one output"
    choices = [
        {
            "index": i,
            "text": out["text"],
            "finish_reason": openai_stop_reason(out["finish_reason"]),
        }
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    response = CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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


# 流式、n>1：从共享collector取(兄弟序号, chunk)，以choices[0].index=兄弟序号逐帧发出；全部finished后一次性发结束帧
async def stream_completion_response_fanout(
    request_id: str,
    model: str,
    shared_collector: StreamOutputCollector,
    seq_ids: list[int],
    num_prompt_tokens: int,
    cleanup_stream,
    cleanup_request,
) -> AsyncGenerator[str, None]:
    """Streaming variant multiplexing ``len(seq_ids)`` siblings into one SSE.

    Each chunk pulled from ``shared_collector`` is a ``(sibling_index, chunk_data)``
    tuple; we re-emit with ``choices[0].index = sibling_index``. Finishes
    only when every sibling has reported ``finished=True``.

    ``num_prompt_tokens`` is the engine-computed prompt length shared by all
    siblings; reusing it avoids re-tokenizing on the event loop at stream
    start.
    """
    # 【译】流式变体：把 len(seq_ids) 个兄弟序列复用到一个SSE流里。
    #
    #   从 shared_collector 取出的每个chunk都是 (sibling_index, chunk_data) 元组；我们以 choices[0].index = sibling_index 重新发出。
    #   只有当每个兄弟都报告了 finished=True 时才结束。
    #
    #   num_prompt_tokens 是引擎算出的、所有兄弟共用的prompt长度；复用它可以避免在流开始时在事件循环上重新tokenize。
    n = len(seq_ids)
    num_tokens_input = num_prompt_tokens
    num_tokens_output = [0] * n
    finished = [False] * n
    engine_reasons: list[str | None] = [None] * n

    # Assume abort until every sibling has reported finished; a client
    # disconnect closes the generator first, leaving this True so the finally
    # aborts whichever siblings are still running.
    # 【译】在每个兄弟都报告finished之前都假定是abort；客户端断开会先关闭生成器，这个标志保持True，
    #   finally就会abort所有仍在运行的兄弟。
    aborted = True
    try:
        while not all(finished):
            idx, chunk_data = await shared_collector.get()
            if finished[idx]:
                continue
            new_text = chunk_data["text"]
            num_tokens_output[idx] += len(chunk_data.get("token_ids", []))
            engine_reasons[idx] = chunk_data.get("finish_reason") or engine_reasons[idx]

            extra_fields: dict[str, Any] = {}
            if "kv_transfer_params" in chunk_data:
                extra_fields["kv_transfer_params"] = chunk_data["kv_transfer_params"]

            yield create_completion_chunk(
                request_id,
                model,
                new_text,
                # 【译】（行尾注释）由结束chunk携带它
                finish_reason=None,  # the terminal chunk carries it
                index=idx,
                **extra_fields,
            )

            if chunk_data.get("finished", False):
                finished[idx] = True

        aborted = False

        usage = {
            "prompt_tokens": num_tokens_input,
            "completion_tokens": sum(num_tokens_output),
            "total_tokens": num_tokens_input + sum(num_tokens_output),
            "num_choices": n,
        }
        usage_chunk = {
            "id": request_id,
            "object": TEXT_COMPLETION_OBJECT,
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }
        # Coalesce the per-sibling stop chunks + usage + [DONE] into one send.
        # 【译】把每个兄弟的停止chunk + usage + [DONE] 合并成一次发送。
        yield (
            "".join(
                create_completion_chunk(
                    request_id,
                    model,
                    "",
                    openai_stop_reason(engine_reasons[i]) or "stop",
                    index=i,
                )
                for i in range(n)
            )
            + data_frame(usage_chunk)
            + STREAM_DONE_MESSAGE
        )
    finally:
        for sid in seq_ids:
            cleanup_stream(sid, aborted=aborted)
        cleanup_request(request_id)
