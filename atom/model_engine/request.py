# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# 流式输出的一个增量：每生成一批token，引擎就调用一次Sequence.stream_callback(RequestOutput)
# 主要给在线服务（api_server）用：拿到增量token后detokenize，作为SSE chunk推给客户端
# 离线generate()不走流式回调，而是等序列结束后统一返回
@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    # 【译】传给流式回调的输出结构。

    # 引擎内部的序列id（Sequence.id）
    request_id: int
    # 本次回调新增的token（增量，不是全部已生成的token）
    output_tokens: List[int]
    # 序列是否已结束
    finished: bool
    # 结束原因，取自Scheduler.postprocess写入的seq.leave_reason："eos"、"max_tokens"、
    #   "stop_sequence"（命中stop字符串）、"stop_<token_id>"（命中stop token id）、"aborted"；未结束时为None
    finish_reason: Optional[str] = None
    # 【分支】PD分离：prefill节点返回给路由层的KV传输信息，decode节点据此拉取KV，见第9批
    kv_transfer_params_output: Optional[Dict[str, Any]] = None
    # 前缀缓存命中的token数，用于在usage里统计cached tokens
    num_cached_tokens: int = 0
