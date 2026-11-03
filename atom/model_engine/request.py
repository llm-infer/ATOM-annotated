# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# 流式输出的一个增量：每生成一批token，前端进程就用一个RequestOutput调用一次该序列的流式回调
#   （回调创建时挂在Sequence.stream_callback上，发往EngineCore之前被摘下，见下方第2条）
# 主要给在线服务（api_server）用：拿到增量token后detokenize，作为SSE chunk推给客户端
# 离线generate()不走流式回调，而是等序列结束后统一返回
# 谁创建、谁调用（已核对代码）：
#   1. EngineCore进程里的Scheduler.postprocess（scheduler.py）每步为有新token或刚结束的序列构造RequestOutput，
#      放进stream_output_queue，经ZMQ送回前端进程
#   2. 回调函数本身不能跨进程pickle：CoreManager.add_request在发出序列前把seq.stream_callback
#      挪进自己的_seq_id_to_callback字典并把seq上的置为None（engine_core_mgr.py），收到RequestOutput后按request_id查表调用
#   3. 不经postprocess就结束的序列，由CoreManager._deliver_terminal_output补一个终止RequestOutput（见下方"要点"第4条）
# 例：prompt后依次生成"Hello"" world""!"三个token（非投机、非延迟输出），回调大致收到三次：
#   ([id_Hello], finished=False) -> ([id_world], False) -> ([id_!], True, finish_reason="eos"或"max_tokens")
#   投机解码一步可能接受多个token，这时一次回调的output_tokens会有多个元素
#
# 时序图：一个在线流式请求的RequestOutput怎样回到客户端（已核对scheduler.py / engine_core.py /
#   engine_core_mgr.py / api_server.py / streaming_dispatch.py）
#
#     客户端            api_server事件循环          CoreManager输出线程               EngineCore进程（调度器）
#       │                     │                             │                               │
#       │── POST(stream) ────>│                             │                               │
#       │                     │ setup_streaming_request：   │                               │
#       │                     │ preprocess(callback)        │                               │
#       │                     │── add_request ─────────────>│                               │
#       │                     │                             │ 回调存进_seq_id_to_           │
#       │                     │                             │ callback，seq上置None         │
#       │                     │                             │── (ADD,[seq]) ZMQ ───────────>│ scheduler.extend -> waiting
#       │                     │                             │                               │ step k：prefill最后一个chunk
#       │                     │                             │                               │ postprocess构造RequestOutput
#       │                     │                             │<── (STREAM,[(id,out)]) ───────│ output_tokens=[t0]
#       │                     │                             │ 查表调回调->enqueue           │
#       │                     │<── call_soon_threadsafe ────│ 本批处理完才flush             │
#       │                     │ collector合并未读chunk      │                               │
#       │                     │ SSE读取时增量反分词         │                               │
#       │<── SSE chunk ───────│                             │                               │
#       │                     │                             │<── STREAM([t1]) ──────────────│ step k+1：decode（投机时可多个id）
#       │<── SSE chunk ───────│<────────────────────────────│                               │
#      ...                    │                             │                               │
#       │                     │                             │<── STREAM(finished=True) ─────│ 命中eos/stop/max_tokens：
#       │                     │                             │ 回调后删除该回调              │ _mark_finished、释放block
#       │<── 最后一个chunk ───│<────────────────────────────│                               │
#       │<── [DONE] ──────────│                             │<── (ADD,[结束的seq]) ─────────│ 同一步里排在STREAM之后
#       │                     │                             │ 在线只做收尾；                │
#       │                     │                             │ 离线交给generate              │
#
#   要点：
#   - 同一步里STREAM先于"已结束的seq"入队（engine_core.py），所以客户端总是先拿到最后一个token再看到结束
#   - RequestOutput只是"增量"：output_tokens是本步新增的token id，文本由前端事件循环增量反分词得到
#   - 中间chunk的prefill步没有RequestOutput（postprocess对is_partial_prefill的序列直接continue）；
#     投机解码一步接受3个token时output_tokens就是3个id
#   - 不经postprocess就结束的请求（unschedulable、在waiting里被abort、PD接收KV失败，见sequence.py状态表⑧⑨⑩）
#     不会有终止的STREAM，它们经take_rejected作为结束的seq回到前端，CoreManager在收到时
#     由_deliver_terminal_output补一个output_tokens=[]、finished=True的RequestOutput，保证客户端不会一直等
#   - n>1：每个兄弟序列各一个回调（_send_stream_chunk_tagged），按sibling_index打标签后汇入同一个collector
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
    #   （PD分离是什么见sequence.py的SequenceStatus；"第N批"指ANNOTATION_PLAN.md里的阅读批次）
    kv_transfer_params_output: Optional[Dict[str, Any]] = None
    # 前缀缓存命中的token数，用于在usage里统计cached tokens
    # 注意：它取自seq.prefix_cache_hit_tokens（Scheduler.postprocess构造时传入），而不是seq.num_cached_tokens；
    #   后者是"已算好KV的token数"，prefill结束时等于整个prompt长度，不能当作命中数
    #   例：prompt 40个token，前缀缓存命中16个 -> 这里是16，而prefill完成后seq.num_cached_tokens是40
    #   PD分离的decode端则沿用prefill节点通过kv_transfer_params带来的命中数
    num_cached_tokens: int = 0
