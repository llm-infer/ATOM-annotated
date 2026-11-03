# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# 流式输出的一个增量：每生成一批token，引擎就调用一次Sequence.stream_callback(RequestOutput)
# 主要给在线服务（api_server）用：拿到增量token后detokenize，作为SSE chunk推给客户端
# 离线generate()不走流式回调，而是等序列结束后统一返回
# 谁创建、谁调用（已核对代码）：
#   1. EngineCore进程里的Scheduler.postprocess（scheduler.py）每步为有新token或刚结束的序列构造RequestOutput，
#      放进stream_output_queue，经ZMQ送回前端进程
#   2. 回调函数本身不能跨进程pickle：CoreManager.add_request在发出序列前把seq.stream_callback
#      挪进自己的_seq_id_to_callback字典并把seq上的置为None（engine_core_mgr.py），收到RequestOutput后按request_id查表调用
#   3. 被调度器直接拒绝、从没运行的序列，由CoreManager自己构造一个output_tokens=[]、finished=True的终止RequestOutput
#      （engine_core_mgr.py里"returned without running"那段），保证客户端不会一直等
# 例：prompt后依次生成"Hello"" world""!"三个token（非投机、非延迟输出），回调大致收到三次：
#   ([id_Hello], finished=False) -> ([id_world], False) -> ([id_!], True, finish_reason="eos"或"max_tokens")
#   投机解码一步可能接受多个token，这时一次回调的output_tokens会有多个元素
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
    #   另外两种：准入时被判定无法调度（如prompt超长）时为"unschedulable: <原因>"（Scheduler在调度循环里写入）；
    #   CoreManager为未运行就返回的序列兜底填"rejected"（leave_reason为空时），因为空原因在OpenAI协议里表示"仍在生成"
    finish_reason: Optional[str] = None
    # 【分支】PD分离：prefill节点返回给路由层的KV传输信息，decode节点据此拉取KV，见第9批
    kv_transfer_params_output: Optional[Dict[str, Any]] = None
    # 前缀缓存命中的token数，用于在usage里统计cached tokens
    # 注意：它取自seq.prefix_cache_hit_tokens（Scheduler.postprocess构造时传入），而不是seq.num_cached_tokens；
    #   后者是"已算好KV的token数"，prefill结束时等于整个prompt长度，不能当作命中数
    #   例：prompt 40个token，前缀缓存命中16个 -> 这里是16，而prefill完成后seq.num_cached_tokens是40
    #   PD分离的decode端则沿用prefill节点通过kv_transfer_params带来的命中数
    num_cached_tokens: int = 0
