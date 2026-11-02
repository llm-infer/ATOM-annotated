# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM OpenAI-compatible API server package."""

# 【译】ATOM的OpenAI兼容API服务器包。
#
# 本包（atom/entrypoints/openai/）的文件分工（第2批2c）：
#   api_server.py                 FastAPI应用：所有HTTP接口、与引擎的交互、启动入口main()
#   protocol.py                   请求/响应的pydantic模型
#   serving_chat.py               /v1/chat/completions 的校验与流式/非流式响应构造
#   serving_completion.py         /v1/completions 的流式/非流式响应构造
#   serving_anthropic.py          /v1/messages（Anthropic Messages API）的格式转换
#   serving_responses.py          /v1/responses（OpenAI Responses API，含Codex的工具形态）的格式转换
#   streaming_dispatch.py         引擎输出线程 → 事件循环的跨线程投递，以及增量反分词
#   sse.py                        SSE帧的编码与解析
#   chat_encoders.py、chat_encoder_adapters.py、deepseek_v41_encoder.py
#                                 chat template渲染：Jinja模板，或模型自带的Python encoder
#   reasoning.py、reasoning_dialects.py、marker_scanner.py、kimi_k3_tokens.py、tool_parser/
#                                 推理内容（思考段）分离、工具调用解析
#   request_timing.py、metrics_setup.py   TTFT计时与Prometheus指标


# 包级别的入口：延迟到调用时才导入api_server（它会拉起torch、transformers等重依赖）
# 仓库内的命令行启动走的是 atom/entrypoints/openai_server.py，没有找到调用这个函数的地方
def main():
    """Lazy entry point to avoid importing heavy deps at package level."""
    # 【译】延迟导入的入口，避免在包级别导入重依赖。
    # 和openai_server.py一样：先设置Prometheus多进程目录，再导入api_server
    from atom.metrics.prometheus import initialize_metrics

    initialize_metrics()
    from .api_server import main as _main

    _main()


__all__ = ["main"]
