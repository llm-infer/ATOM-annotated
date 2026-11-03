# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight protocol types shared between CoreManager and EngineCore.

Deliberately free of torch/aiter/zmq imports so the CPU-only CoreManager
(request thread) and unit tests can import ``EngineCoreRequestType`` without
dragging in the heavy ``engine_core`` -> ``async_proc`` -> ``aiter`` chain.
"""

# 【译】CoreManager与EngineCore之间共享的轻量协议类型。
#
#   故意不导入torch/aiter/zmq，这样只用CPU的CoreManager（请求线程）和单元测试可以导入 EngineCoreRequestType，
#   而不会拖进沉重的 engine_core -> async_proc -> aiter 依赖链。

import enum


# 本文件：CoreManager（用户进程）与EngineCore进程之间每条ZMQ消息的"类型标签"
#   消息格式：pickle.dumps((EngineCoreRequestType.X, 数据))，收方先pickle.loads再按类型分派
#   使用方：engine_core_mgr.py（发请求、收结果）、engine_core.py（收请求、发结果）
#
#   方向                         类型               数据
#   CoreManager → EngineCore     ADD                list[Sequence]（可选第三项：前端算好的prompt hash链）
#                                UTILITY            {"cmd": 命令名, ...参数}，工具RPC（utility command）
#                                SHUTDOWN           None，让EngineCore退出
#   EngineCore → CoreManager     READY              初始化完成（带max_pool_tokens等启动信息）
#                                ADD                list[Sequence]：已结束（或被拒绝）的序列
#                                STREAM             [(seq_id, RequestOutput), ...]：流式增量，交给流式回调
#                                UTILITY_RESPONSE   工具RPC的回复（带request_id）
#                                METRICS            EngineCore定时推送的指标快照
#                                SHUTDOWN           EngineCore已退出
#   ABORT、START_DP_WAVE、EXECUTOR_FAILED 在仓库里没有使用（中止请求走的是UTILITY命令"abort_request"）
#
# 一个请求在这条通道上的消息时间线（DP=1，时间只是示意；→ 为CoreManager发往EngineCore，← 反之）：
#   ← READY                      EngineCore加载完模型、分好KV池、抓完CUDA Graph后只发一次
#   → ADD [A]           t=0      请求A到达；前端已经tokenize好，发过去的是token id而不是文本
#   ← STREAM [(A,..)]   t=80ms   A的prefill那一步结束，带回第1个token（这就是TTFT的来源）
#   → ADD [B]           t=95ms   B到达：不必等A结束，B在随后的step里就会被调度（continuous batching）
#   ← STREAM [(A,..),(B,..)]     此后每个step一条STREAM，装着"本步有新token的所有序列"
#   ← ADD [A]                    A结束（eos / stop / max_tokens）：EngineCore保证先发它最后一步的STREAM，再发这条
#   → UTILITY abort_request(B)   客户端断开：B在EngineCore的下一个postprocess里以"aborted"结束
# 要点：一条STREAM对应EngineCore busy_loop的一次迭代（一次schedule + forward + postprocess），而不是一个请求；
#   从前端看，EngineCore是一条持续运转的流水线：任何时刻都能塞进新请求，输出按step一批批流回来
#
# 为什么"pickle整个Python对象 + ZMQ"就够了，而不用共享内存：
#   这条通道上只走控制面数据——token id、采样参数和少量元数据。例：4096个token的prompt ≈ 4096 × 4B = 16KB；
#   decode阶段每条序列每步只多一两个token。真正的大块数据（权重、KV cache，GB级）始终留在GPU上，从不经过这里。
#   这个量级的序列化开销远小于一步forward（毫秒级），换来的是实现简单、进程之间互不共享内存（一方崩溃不会破坏另一方）。
#   代价：pickle.loads会执行任意对象的反序列化逻辑，所以这些socket只能在同一信任域内的进程之间使用
#
# enum.Enum：枚举类，每个成员是单例，可以用 == 或 is 比较；反序列化后仍是同一个成员
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    # 【译】请求类型定义为十六进制字节串，这样无需单独的编码步骤就可以通过socket发送。
    # 解释：ATOM实际上把(类型, 数据)整体pickle后发送，并没有直接把这些字节串写进socket

    # 两个方向都用：发请求（新序列）和回结果（已结束的序列）
    #   同一个类型码在两个方向上含义不同，不会混淆：每个方向是各自独立的socket，收方只会收到一个方向的消息
    ADD = b"\x00"
    # 以下两个未使用
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    # 工具RPC：统计、profile、权重更新、中止请求、集合RPC等，由EngineUtilityHandler执行
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    # 【译】在EngineCoreProc内部使用的哨兵。
    # 解释：ATOM里没有EngineCoreProc这个类（这几项沿用自vLLM），EXECUTOR_FAILED在仓库里未被使用
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used within EngineCore.
    # 【译】在EngineCore内部使用的哨兵。
    # 解释：实际在两个方向上都用：CoreManager发SHUTDOWN让EngineCore退出；EngineCore退出时也发一条SHUTDOWN通知CoreManager
    SHUTDOWN = b"\x05"
    # Stream output for callbacks
    # 【译】给流式回调用的流式输出
    STREAM = b"\x06"
    # Signal that EngineCore is fully initialized and ready
    # 【译】表示EngineCore已经完全初始化、可以接收请求的信号
    READY = b"\x07"
    # Response to a synchronous utility command
    # 【译】对同步工具命令的回复
    UTILITY_RESPONSE = b"\x08"
    # Unsolicited metrics snapshot, pushed by EngineCore on its own clock.
    # 【译】主动推送的指标快照，由EngineCore按自己的节奏推送。
    METRICS = b"\x09"
