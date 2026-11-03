# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SSE frame encoding and inspection for the streaming endpoints.

One frame is encoded per streamed chunk per request, so at high concurrency
this is one of the fixed per-token costs that caps the API server before the
GPU does. ``msgspec`` with a reusable encoder measures 5.8x faster than
``json.dumps`` on a chat delta (0.25 vs 1.47 us/chunk; see
``/app/logs_claude/bench_sse_encode.py``).

Encoding straight to ``bytes`` and letting the ASGI layer write them is another
10% on top, but the frame builders are shared with the atomesh service, which
assembles ``list[str]`` batches -- not worth splitting them for 1.6% of the
original cost.

Unlike ``json.dumps``, msgspec rejects NaN and Infinity rather than emitting
the non-standard literals. Any float that can reach a frame (logprobs, for
instance) has to be sanitised before it gets here.
"""

# 【译】流式接口的SSE帧编码与检查。
#
#   每个请求的每个流式chunk都要编码一帧，所以高并发时，这是在GPU之前就先把API server压到上限的固定逐token开销之一。
#   使用可复用编码器的 msgspec 在一个chat增量上实测比 json.dumps 快5.8倍（每chunk 0.25 vs 1.47微秒；
#   见 /app/logs_claude/bench_sse_encode.py）。
#
#   直接编码成 bytes、让ASGI层去写，还能再快10%，但帧构造函数与atomesh服务共用，而后者拼装的是 list[str] 批次——
#   为原始开销的1.6%把它们拆开不值得。
#
#   与 json.dumps 不同，msgspec遇到NaN和Infinity会拒绝，而不是输出非标准的字面量。任何可能进入帧的浮点数
#   （例如logprobs）都必须在到达这里之前清洗过。
#
# 本文件（第2批2c，B级）：SSE（服务端推送事件）帧的编码与解析，三个小函数
# SSE帧格式（文本协议，每帧以一个空行结束）：
#   匿名帧：data: {"id": "chatcmpl-...", "choices": [...]}\n\n                ← OpenAI的流式响应用这种
#   具名帧：event: message_start\ndata: {"type": "message_start", ...}\n\n     ← Anthropic / Responses的流式响应用这种
# 调用方：serving_chat / serving_completion / atomesh的atom_standalone_service（data_frame）、
#   serving_anthropic / serving_responses（event_frame）、api_server._client_stream（iter_sse_data，用于日志和TTFT）

# 原理：为什么用SSE而不是WebSocket。生成是单向的（服务端 → 客户端），SSE就是一个普通的HTTP响应、只是不一次性写完，
#   能穿过一般的HTTP代理和负载均衡，客户端用任何HTTP库逐行读即可。代价是协议本身没有"正常结束"的标志——
#   连接断开既可能是生成完了，也可能是中途出错，所以OpenAI约定以 "data: [DONE]" 作为结尾哨兵，没收到它就说明响应不完整
#   中间的代理若缓冲响应（例如nginx默认开启proxy_buffering），客户端会看到所有帧在最后一下子到达，流式就失去意义
import re
from collections.abc import Iterator
from typing import Any

import msgspec

# 模块级的msgspec JSON编码器，所有帧复用；encode()返回bytes，下面再.decode()成str
_encoder = msgspec.json.Encoder()


# 编码一个匿名帧："data: <json>\n\n"
def data_frame(payload: Any) -> str:
    """Encode one anonymous ``data:`` SSE frame."""
    # 【译】编码一个匿名的 data: SSE帧。
    return f"data: {_encoder.encode(payload).decode()}\n\n"


# 编码一个具名事件帧："event: <名字>\ndata: <json>\n\n"
def event_frame(event: str, payload: Any) -> str:
    """Encode one named-event SSE frame."""
    # 【译】编码一个具名事件的SSE帧。
    return f"event: {event}\ndata: {_encoder.encode(payload).decode()}\n\n"


# 解析：从一个（可能拼了多帧的）字符串里逐个取出每帧的data内容
# 例：'data: {"a": 1}\n\ndata: [DONE]\n\n' → 依次产出 '{"a": 1}'、'[DONE]'
def iter_sse_data(chunk: str) -> Iterator[str]:
    """Read data from complete local frames, including coalesced sends.

    The endpoint generators yield whole frames before ASGI encodes them. No
    cross-chunk byte buffer is needed here. Named events, comments, multiline
    data and both LF/CRLF delimiters are supported.
    """
    # 【译】从完整的本地帧中读出data部分，包括被合并在一起发送的多帧。
    #
    #   各接口的生成器在ASGI编码之前yield的都是完整的帧，所以这里不需要跨chunk的字节缓冲。
    #   支持具名事件、注释行、多行data，以及LF和CRLF两种分隔符。
    # 按空行切分帧；每帧取所有以"data:"开头的行，去掉前缀、行尾的\r和前导空格；一帧有多行data时用换行拼接
    for frame in re.split(r"\r?\n\r?\n", chunk):
        data = [
            line[5:].removesuffix("\r").lstrip(" ")
            for line in frame.split("\n")
            if line.startswith("data:")
        ]
        if data:
            yield "\n".join(data)
