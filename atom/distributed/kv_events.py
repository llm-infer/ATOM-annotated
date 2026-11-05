# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV cache events: wire-compatible with vLLM's `vllm.distributed.kv_events`,
plus ATOM extensions (`BlockTransferred`, CPU/DISK/REMOTE medium constants)."""

# 【译】KV cache事件：线上格式与vLLM的 vllm.distributed.kv_events 兼容，外加ATOM的扩展（BlockTransferred、
#   CPU/DISK/REMOTE 这几个存储介质常量）。
#
# ======================================================================================================
# 【导读】【分支】KV cache事件：把"本实例的前缀缓存里新存入了哪些block、淘汰了哪些block"实时广播给引擎之外的订阅者
#   挂在主干的位置：BlockManager 记录事件（block_manager.py），Scheduler.publish_kv_events 每步发出（scheduler.py），
#   EngineCore 每个engine step结束时调用（engine_core.py _process_engine_step 的 finally）。默认关闭。
#
# ① 前置：前缀缓存的block hash（完整讲解：arg_utils.py enable_prefix_caching上方的【科普】，实现在prefix_hash.py、block_manager.py）
#   prompt按block切开，每个填满的block算一个"连同前面所有token"的链式hash：h0 = H(block0)，h1 = H(h0, block1)，……
#   所以一个hash就代表"从开头到这个block为止的整段前缀"。开DCP时一个hash block = block_size × DCP 个token
#   （prefix_hash.py hash_block_size_for），事件里的 block_size 也按这个算。
#
# ② 为什么要对外发事件：KV感知的路由
#   多实例部署时，前面有一个路由器把请求分给N个ATOM实例。共享同一段系统提示词的请求，发到已经缓存了这段前缀的实例，
#   只需prefill后面几十个token；发到别的实例就要从头prefill几千个。路由器在别的进程、甚至别的机器上，读不到各实例的
#   BlockManager，于是每个实例把自己前缀索引的变化当作事件流发出去，路由器据此维护一份"实例 → 它缓存着的hash集合"：
#
#     ATOM实例0 ──PUB──►┐                          ┌ 实例0：{h0, h1, h2, …}
#     ATOM实例1 ──PUB──►├──► KV感知路由器（SUB）───┤ 实例1：{h0, h7, …}        新请求先算出自己的hash链，
#     ATOM实例2 ──PUB──►┘                          └ 实例2：{…}                发给前缀命中最长的实例
#
# ③ 事件类型（何时产生都在 block_manager.py）：
#   BlockStored       一串连续的block刚进入前缀索引：hash_blocks（forward算完后，本步填满的prompt block，以及
#                     hash_decode_blocks 交来的decode生成填满的block）、publish_loaded_prefix（从offload加载回来的前缀）、
#                     record_remote_store（PD分离decode侧收到的远端block，medium=REMOTE；PD分离见第9批）
#   BlockRemoved      block池惰性淘汰时一个hash被挤出索引（_record_evicted，每个hash一条）；前缀命中复用空闲block不算淘汰
#   AllBlocksCleared  clear_cache 清空整个前缀索引（/reset_prefix_cache一类管理接口）
#   BlockTransferred  ATOM扩展：block在GPU/CPU/DISK/REMOTE之间搬动；目前仓库里没有任何地方产生它（grep只有定义）
#   例：block_size=16，一个40 token的prompt做完prefill → 2个满block进索引（第3个block只有8个token，不算hash）→
#     BlockStored(block_hashes=[h0, h1], parent_block_hash=None, token_ids=[前32个token], block_size=16, token_offset=0)
#     请求结束后这两个block回到空闲链表，hash仍在索引里（还能被命中）；直到它们被拿去写新内容时，
#     才各发一条 BlockRemoved(block_hashes=[h0]) / BlockRemoved(block_hashes=[h1])
#
# ④ 数据流与线程（每步至多一批，绝不阻塞推理）：
#
#   busy_loop线程（调度 + forward）                                   发送线程 atom-kv-event-sender
#   ──────────────────────────────                                   ──────────────────────────────
#   BlockManager._event_log.append(事件)（分配 / 淘汰block时）
#   EngineCore._process_engine_step 的 finally：
#     Scheduler.publish_kv_events()（scheduler.py）
#       events = block_manager.take_events()   取空日志
#       ZmqEventPublisher.publish(events)（本文件）
#         EventBatch(ts, events, dp_rank) → 分配seq → msgpack编码
#         queue.put_nowait((seq, payload)) ──有界队列──►              _run()：queue.get()
#           满了：扔掉最旧的一批，dropped+1                              PUB.send_multipart([topic, seq(8字节大端), payload])
#                                                                       开了重放时，同时追加进重放环形缓冲区
#                                                                    ROUTER（可选）：收到 [client_id, start_seq] →
#                                                                      每轮最多重放64批 seq ≥ start_seq 的缓存，
#                                                                      最后发 [client_id, REPLAY_DONE, msgpack([最旧seq, 最新seq])]
#
#   为什么宁可丢事件也不阻塞：publish 在busy_loop线程上、每步都调用；如果订阅者慢、队列满时去阻塞等待，decode就会被
#     一个外部组件拖住，所有请求的延迟都变大。而事件只是路由的参考信息：漏掉一条，路由器最多做出一次不够好的路由
#     （例如把请求发给一个其实已经淘汰了该前缀的实例，只是少命中一些），不会让任何请求算错。
#   丢失如何被发现：seq在入队时就分配，丢掉的批也占一个seq → 订阅者看到seq跳号。传输层丢的（订阅者连得晚 / 处理慢）
#     还在重放缓冲区里，可以按seq补回；队列溢出或编码失败丢的从没发出去，补不回来（stats['dropped'] / ['encode_errors']）。
#   队列容量：buffer_steps 默认10000批（每步至多一批）。PUB的hwm默认0 = 不限：ZMQ的PUB从不阻塞发送方，会替每个慢订阅者
#     无限缓存（内存随之增长）；设了hwm时，超出的消息由ZMQ对该订阅者直接丢弃。
#
# ⑤ 配置与部署（各字段见 config.py KVEventsConfig 上方的说明；只由 ATOM_KV_EVENTS_* 环境变量配置，见 envs.py）：
#   ATOM_KV_EVENTS_ENABLE=1 开启；PUB默认绑在 tcp://127.0.0.1:5557，重放端点默认不开（ATOM_KV_EVENTS_REPLAY_ENDPOINT为空）。
#   DP：每个DP rank有自己的Scheduler和发布器，端口按rank偏移（offset_endpoint：rank0 5557、rank1 5558、……）；
#   PP：只有head stage发布（scheduler.py __init__ 里的说明）；RapidServe的PrefillScheduler没有BlockManager，publish是空操作。
#
# 对比nano-vllm：nano-vllm是单实例离线引擎，没有对外的KV事件；本文件沿用vLLM的消息格式，现成的vLLM事件订阅者可以直接用。
# ======================================================================================================

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Any, Final

import msgspec

logger = logging.getLogger("atom")

# Where a block lives.
# 【译】block所在的位置（存储介质）。
# GPU：HBM里的KV池（默认）；REMOTE：PD分离时从远端收到的（BlockManager.record_remote_store，第9批）；
#   CPU / DISK：给KV offload分层预留，目前仓库里没有事件用到它们（grep）
MEDIUM_GPU: Final[str] = "GPU"
MEDIUM_CPU: Final[str] = "CPU"
MEDIUM_DISK: Final[str] = "DISK"
MEDIUM_REMOTE: Final[str] = "REMOTE"

# Reserved seq frame that terminates a replay response. Its payload is a
# msgpack `[oldest_available_seq, latest_seq]` window so a consumer knows when
# the replay is complete and whether earlier events were already evicted
# (start_seq < oldest). 2**64-1 is reserved for this and never used as a data
# seq. (Replay requires a DEALER/ROUTER-style client that can read the
# multi-message reply; a REQ socket cannot.)
# 【译】保留的seq帧，用来结束一次重放回复。它的载荷是msgpack编码的 [oldest_available_seq, latest_seq] 窗口，这样消费者
#   知道重放什么时候结束，以及更早的事件是否已经被淘汰（start_seq < oldest）。2**64-1 保留给它，从不用作数据seq。
#   （重放需要一个能读取多条回复消息的 DEALER/ROUTER 式客户端；REQ socket做不到。）
# REQ socket严格"一问一答"，收到第一条回复后就不能再收；重放的回复是很多条消息，所以客户端要用DEALER
REPLAY_DONE: Final[bytes] = b"\xff" * 8


# 所有事件的基类。msgspec.Struct：类似dataclass，但编解码由C实现、很快；类定义里的关键字参数是msgspec的选项：
#   array_like=True     编码成msgpack数组（按字段声明顺序，不带字段名）：体积小，但字段顺序就是线上格式，与vLLM兼容
#                       的前提就是保持顺序——新字段只能加在末尾
#   omit_defaults=True  按字典编码时省略等于默认值的字段；但这里同时是array_like，实测（msgspec 0.22）默认值照样逐位编码
#   gc=False            实例不被Python的循环垃圾回收器跟踪（事件里没有循环引用），每步都新建很多事件时省GC开销
#   tag=True            带类型标签（默认是类名），子类继承，组成"带标签的联合类型"：array_like时标签是数组的第一个元素，
#                       解码 EventType 时按标签分派到具体的子类
#   例（msgspec 0.22实测）：BlockRemoved(block_hashes=[123]) 编码成 ["BlockRemoved", [123], "GPU", null]，
#     EventBatch 编码成 [ts, [事件1, 事件2, …], dp_rank]。vLLM的消费者按同样的位置解码；位置在它已知字段之后的
#     多余元素（如BlockStored末尾的token_offset）会被忽略
class KVCacheEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    """Tagged-union base so subscribers can dispatch on event type."""

    # 【译】带标签的联合类型的基类，让订阅者能按事件类型分派。


class BlockStored(KVCacheEvent):
    """A run of contiguous prefix-cacheable blocks just became resident."""

    # 【译】一串连续的、可做前缀缓存的block刚刚驻留下来。
    # 字段（ATOM填的，见 block_manager.py _make_block_stored）：
    #   block_hashes       这串block的链式hash，顺序即前缀顺序
    #   parent_block_hash  第一个block之前那个block的hash（从序列开头开始时为None）：订阅者据此把这串接到已有的链上
    #   token_ids          这些block的全部token（长度 = len(block_hashes) × block_size）
    #   block_size         一个hash block的token数（开DCP时 = block_size × DCP）
    #   medium             GPU，或 REMOTE（record_remote_store）
    #   token_offset       见下
    #   lora_id / lora_name / extra_keys / group_idx：vLLM的字段，ATOM不填（保持默认None）

    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = MEDIUM_GPU
    lora_name: str | None = None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    # Reserved wire slots; emitted as None until hybrid-cache wiring lands.
    # 【译】保留的线上位置；在混合缓存的接线完成之前一律发None。
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    # ATOM extension (trailing, so strict vLLM array_like consumers ignore it):
    # sequence position of the first token of the first block in this run.
    # With block_size, block i covers [token_offset + i*block_size, +block_size).
    # 【译】ATOM扩展（放在末尾，所以严格的vLLM array_like消费者会忽略它）：本串第一个block的第一个token在序列中的位置。
    #   配合 block_size，第i个block覆盖 [token_offset + i*block_size, 再加block_size)。
    # 例：前缀命中了前32个token（2个block），本步新填满第2、3个block → token_offset=32，两个block分别覆盖 [32, 48)、[48, 64)
    token_offset: int | None = None


class BlockRemoved(KVCacheEvent):
    """One or more blocks were evicted from the given medium."""

    # 【译】一个或多个block从给定的介质中被淘汰。
    # ATOM每淘汰一个hash发一条（block_manager.py _record_evicted），即 block_hashes 总是只有一个元素

    block_hashes: list[int]
    medium: str | None = MEDIUM_GPU
    group_idx: int | None = None


class AllBlocksCleared(KVCacheEvent):
    """Entire cache (or one medium, when set) was cleared."""

    # 【译】整个缓存（设置了medium时只是那一种介质）被清空了。
    # ATOM只在 BlockManager.clear_cache 发它，medium保持None（整个前缀索引）；正被请求使用的block不受影响，只是不能再被命中

    medium: str | None = None


class BlockTransferred(KVCacheEvent):
    """A block moved between tiers without changing identity.

    Emitted on GPU↔CPU/DISK swap, REMOTE→GPU receive, and GPU→REMOTE send.
    ATOM-only — strict vLLM consumers should narrow the union to exclude it.
    """

    # 【译】一个block在存储层级之间移动，身份（hash）不变。
    #
    #   在 GPU↔CPU/DISK 换入换出、REMOTE→GPU 接收、GPU→REMOTE 发送时发出。
    #   仅ATOM有——严格的vLLM消费者应把联合类型收窄、排除它。
    # 观察：grep整个仓库，没有任何地方构造BlockTransferred（KV offload / PD分离的路径都没发它），目前只是线上格式里预留的类型

    block_hashes: list[int]
    from_medium: str
    to_medium: str
    group_idx: int | None = None


# Union of all events. Subscribers that only know the vLLM-compatible subset
# should narrow this to `BlockStored | BlockRemoved | AllBlocksCleared`.
# 【译】所有事件的联合类型。只认识vLLM兼容子集的订阅者，应把它收窄为 BlockStored | BlockRemoved | AllBlocksCleared。
# A | B：Python 3.10起的联合类型写法（等价于 Union[A, B]）；msgspec解码EventType时按标签选具体类型
EventType = BlockStored | BlockRemoved | AllBlocksCleared | BlockTransferred


class EventBatch(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """A batch of events emitted at one publish tick. Field layout matches
    vLLM's EventBatch (ts as float seconds, events list, optional dp_rank)."""

    # 【译】一次发布时刻发出的一批事件。字段布局与vLLM的EventBatch一致（ts为浮点秒、events列表、可选的dp_rank）。
    # 一个engine step产生的全部事件 = 一个EventBatch = 线上的一条消息（占一个seq）；dp_rank让路由器分清是哪个DP rank的缓存

    ts: float
    events: list[EventType]
    data_parallel_rank: int | None = None


# ----- Publisher ---------------------------------------------------------- #
# 【译】----- 发布器 -----


class EventPublisher(ABC):
    """Strategy for delivering EventBatch off the hot path.

    Implementations must be safe to call from the scheduler thread; the actual
    network/IO must run in a background thread or be lock-free so the
    `publish()` call returns quickly.
    """

    # 【译】把EventBatch送出热路径的策略。
    #
    #   实现必须能在调度器线程上安全调用；真正的网络/IO必须在后台线程里进行或是无锁的，这样 publish() 调用能很快返回。
    # ABC + @abstractmethod：抽象基类，子类必须实现这两个方法才能实例化。两个实现：NullEventPublisher（关闭时）、ZmqEventPublisher

    @abstractmethod
    def publish(self, events: Iterable[KVCacheEvent]) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class NullEventPublisher(EventPublisher):
    """No-op. Default when KV events are disabled."""

    # 【译】空操作。KV事件关闭时的默认实现。
    # 让Scheduler不管开没开都能直接调 publish() / shutdown()

    def publish(self, events: Iterable[KVCacheEvent]) -> None:
        return

    def shutdown(self) -> None:
        return


# Batches a single sender-loop iteration replays before it goes back to the
# live queue. Bounds how long a large replay (default retention 10k batches)
# can keep the live PUB stream waiting; it does not bound the replay itself.
# 【译】发送循环每一轮在回到实时队列之前最多重放的批数。它限制一次大重放（默认保留1万批）能让实时PUB流等待多久；
#   它不限制重放本身的总量。
REPLAY_CHUNK: Final = 64


def _bind(sock, endpoint: str, role: str, zmq_error_cls) -> None:
    """Bind, turning EADDRINUSE and friends into an error that names the
    endpoint rules, since the collision is usually a config mistake."""
    # 【译】绑定，并把 EADDRINUSE 之类的错误变成一个讲明端点规则的错误，因为冲突通常是配置错误。
    # 例：DP=2时 rank0 的PUB绑5557、rank1绑5558；若把重放端点配成5558，就和rank1的PUB冲突——两个端口至少要相差DP数
    try:
        sock.bind(endpoint)
    except zmq_error_cls as e:
        raise RuntimeError(
            f"KV events: cannot bind {role} socket to {endpoint!r}: {e}. "
            "Each publisher binds its configured PUB and replay endpoints "
            "offset by its DP rank, so (a) the two configured tcp ports must "
            "be at least data_parallel_size apart, and (b) every other engine "
            "process on this host (a P/D peer, another deployment) needs its "
            "own ATOM_KV_EVENTS_ENDPOINT / ATOM_KV_EVENTS_REPLAY_ENDPOINT."
        ) from e


def offset_endpoint(endpoint: str, data_parallel_rank: int | None) -> str:
    """Make a configured ZMQ bind endpoint unique per data-parallel rank.

    Every DP rank runs its own Scheduler and therefore its own publisher, but
    they all read the same `KVEventsConfig`, so binding the configured address
    verbatim collides from rank 1 onward. `tcp://host:port` gets the rank added
    to the port; `ipc://` and `inproc://` get a `_dp{rank}` suffix. Rank 0 (or
    no DP) keeps the configured endpoint, so single-engine deployments and
    existing consumers see no change.
    """
    # 【译】让配置的ZMQ绑定端点在每个数据并行rank上唯一。
    #
    #   每个DP rank运行自己的Scheduler，因此有自己的发布器，但它们读的是同一份 KVEventsConfig，所以原样绑定配置的地址，
    #   从rank 1起就会冲突。tcp://host:port 把rank加到端口上；ipc:// 和 inproc:// 加 _dp{rank} 后缀。rank 0（或没有DP）
    #   保持配置的端点，所以单引擎部署和已有的消费者看不到任何变化。
    # 例：("tcp://127.0.0.1:5557", 2) → "tcp://127.0.0.1:5559"；("ipc:///tmp/kv", 1) → "ipc:///tmp/kv_dp1"；rank为0或None → 原样
    # 调用方：ZmqEventPublisher.__init__（PUB与重放端点各一次）
    if not data_parallel_rank:
        return endpoint
    if endpoint.startswith("tcp://"):
        host, sep, port = endpoint.rpartition(":")
        if sep and port.isdigit():
            return f"{host}:{int(port) + data_parallel_rank}"
        raise ValueError(
            f"cannot offset KV event endpoint {endpoint!r} for dp_rank "
            f"{data_parallel_rank}: tcp endpoints need an explicit numeric port"
        )
    return f"{endpoint}_dp{data_parallel_rank}"


class ZmqEventPublisher(EventPublisher):
    """ZMQ PUB-socket publisher.

    Uses a background sender thread + bounded queue so the scheduler never
    blocks on network IO. If the queue fills (slow subscriber), the oldest
    batch is dropped — KV events are advisory and a missed eviction is
    cheaper than stalling inference.

    Every message is a three-frame multipart `[topic, seq, payload]`, where
    `topic` is the (possibly empty) subscription key, `seq` is a monotonic
    8-byte big-endian batch counter (wrapping at 2**64-1; the value 2**64-1 is
    reserved for the REPLAY_DONE terminal frame and never used as a data seq),
    and `payload` is the msgpack-encoded EventBatch. Consumers must use
    `recv_multipart()`.

    `seq` is assigned at enqueue time, so a batch that never reaches the wire
    still consumes a sequence number: the loss surfaces to subscribers as a
    gap in the seq stream rather than vanishing silently. Three loss cases:
      * transport drop (slow/late SUB) — detectable as a gap AND recoverable
        from the replay buffer (the batch was sent, so it is buffered);
      * queue-overflow drop (slow encoder/sender) — detectable as a gap but
        NOT recoverable (never sent, never buffered); also counted in
        `stats['dropped']`;
      * encode failure — same as overflow, counted in `stats['encode_errors']`.

    Sequence numbers start at 0 per publisher process and advance by one per
    batch, so the 2**64-1 modulus is a wire-format bound, not an operational
    one: at one batch per scheduler step the counter cannot wrap within the
    lifetime of a process. Replay ordering (`seq >= start_seq`) therefore
    assumes no wrap inside the retained window; see `_advance_replay`.
    """

    # 【译】基于ZMQ PUB socket的发布器。
    #
    #   用后台发送线程 + 有界队列，让调度器永远不会阻塞在网络IO上。队列满了（订阅者慢）时丢弃最旧的一批——KV事件只是
    #   参考信息，漏掉一次淘汰通知的代价比拖住推理小。
    #
    #   每条消息是三帧的multipart [topic, seq, payload]：topic 是（可能为空的）订阅键；seq 是单调递增的8字节大端批次计数器
    #   （在2**64-1处回绕；2**64-1这个值保留给REPLAY_DONE终止帧，从不用作数据seq）；payload 是msgpack编码的EventBatch。
    #   消费方必须用 recv_multipart()。
    #
    #   seq 在入队时分配，所以一个没能上线的批次仍然消耗一个序号：丢失在订阅者那里表现为seq流里的一个缺口，而不是悄无
    #   声息地消失。三种丢失情况：
    #     * 传输层丢失（订阅者慢 / 连得晚）——能看到缺口，也能从重放缓冲区补回（这批发出去过，所以缓冲了）；
    #     * 队列溢出丢失（编码 / 发送慢）——能看到缺口，但补不回来（从没发出、也没缓冲）；也计入 stats['dropped']；
    #     * 编码失败——同溢出，计入 stats['encode_errors']。
    #
    #   序号在每个发布器进程里从0开始、每批加一，所以2**64-1这个模数是线上格式的界限，不是实际运行中的界限：每个调度步
    #   一批的话，计数器在进程的生命周期内不可能回绕。因此重放的排序（seq >= start_seq）假设保留的窗口内没有回绕；
    #   见 _advance_replay。
    # 估算：每步10ms、每步一批，2**64批要约58亿年
    # 订阅端用法：SUB连endpoint、订阅topic前缀（b""收全部），recv_multipart() 得到 [topic, seq, payload]，
    #   msgspec.msgpack.Decoder(EventBatch).decode(payload)；发现seq跳号时用DEALER连replay_endpoint补

    # 参数：endpoint / replay_endpoint  配置的绑定地址（按DP rank偏移后才绑定，结果存在self.endpoint / self.replay_endpoint）
    #       topic                每条消息的第一帧（SUB按前缀过滤）
    #       hwm                  PUB socket的高水位线（0 = 不限）
    #       buffer_steps         进程内发送队列的容量（批）
    #       replay_buffer_steps  重放环形缓冲区的容量（批）
    #       data_parallel_rank   写进每个EventBatch，并决定端口偏移
    #       encoder              msgspec编码器（测试可以注入）
    # 调用方：make_publisher（Scheduler.__init__ 经它构造，参数来自 KVEventsConfig）
    def __init__(
        self,
        endpoint: str,
        *,
        topic: str = "",
        hwm: int = 0,
        buffer_steps: int = 10_000,
        replay_endpoint: str = "",
        replay_buffer_steps: int = 10_000,
        data_parallel_rank: int | None = None,
        encoder: msgspec.msgpack.Encoder | None = None,
    ) -> None:
        # 为什么必须 ≥ 1：queue.Queue(maxsize=0) 表示"无界"，那样"满了丢最旧"的背压就失效了；
        #   同理 deque(maxlen=0) 什么也存不下，重放形同关闭
        if buffer_steps < 1:
            raise ValueError(
                f"buffer_steps must be >= 1 to keep the drop-on-overflow "
                f"backpressure intact; got {buffer_steps}"
            )
        if replay_endpoint and replay_buffer_steps < 1:
            raise ValueError(
                f"replay_buffer_steps must be >= 1 when replay is enabled "
                f"(maxlen=0 would silently disable replay retention); "
                f"got {replay_buffer_steps}"
            )
        # Local import: keep pyzmq an optional runtime dep. BlockManager imports
        # this module unconditionally, but only the zmq publisher path needs pyzmq.
        # 【译】在函数内导入：让pyzmq保持为可选的运行时依赖。BlockManager无条件导入本模块，但只有zmq发布器这条路径需要pyzmq。
        import zmq

        self._dp_rank = data_parallel_rank
        self._topic_bytes = topic.encode("utf-8")
        self._encoder = encoder or msgspec.msgpack.Encoder()
        # Queue items are (seq, payload) tuples; None is the shutdown sentinel.
        # 【译】队列元素是 (seq, payload) 元组；None 是关闭哨兵。
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(
            maxsize=buffer_steps
        )

        # Effective bind addresses, after the per-DP-rank offset.
        # 【译】实际的绑定地址，经过按DP rank的偏移之后。
        self.endpoint = offset_endpoint(endpoint, data_parallel_rank)
        self.replay_endpoint = (
            offset_endpoint(replay_endpoint, data_parallel_rank)
            if replay_endpoint
            else ""
        )

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.set_hwm(hwm)
        _bind(self._socket, self.endpoint, "PUB", zmq.ZMQError)
        # Captured so the sender thread never re-imports zmq.
        # 【译】先把这些对象抓住，这样发送线程永远不需要重新导入zmq。
        self._zmq_error_cls = zmq.ZMQError
        self._zmq_again_cls = zmq.Again
        self._zmq_noblock = zmq.NOBLOCK

        # Optional replay: a ROUTER socket + ring buffer of recently-sent
        # batches. A subscriber that detects a seq gap can request everything
        # from a start sequence number and get the buffered batches back. The
        # ROUTER is created here but used only by the sender thread.
        # `replay_buffer_steps` is a distinct knob from `buffer_steps` (the
        # in-flight send queue): it bounds the long-lived retention of encoded
        # payloads (which can include large token_id lists) for the publisher's
        # lifetime, and only when replay is enabled.
        # 【译】可选的重放：一个ROUTER socket + 最近发出批次的环形缓冲区。检测到seq缺口的订阅者可以请求"从某个起始序号起"
        #   的全部批次，拿回缓冲着的批次。ROUTER在这里创建，但只由发送线程使用。
        #   replay_buffer_steps 是一个与 buffer_steps（在途发送队列）不同的开关：它限制编码后载荷（可能包含很大的token_id列表）
        #   在发布器整个生命周期内的长期保留量，而且只在开启重放时才有。
        # 为什么用ROUTER：PUB是单向广播，收不了请求；ROUTER给每条收到的消息加上发送方的身份帧，回复时带上它，
        #   就能只回给提出请求的那个客户端
        # 内存估算：一步prefill 8192个token时，BlockStored的token_ids按msgpack每个约3~5字节，一批约几十KB；
        #   10000批全是这样的话约几百MB，按事件速率和内存预算调小
        self._replay = None
        self._replay_buffer: deque[tuple[int, bytes, bytes]] | None = None
        if self.replay_endpoint:
            self._replay = ctx.socket(zmq.ROUTER)
            # Replay is serviced on the sender thread, so a replay client that
            # stops reading must never stall live publication. ROUTER_MANDATORY
            # turns a full per-peer pipe into an error instead of a silent drop,
            # and every replay send is NOBLOCK, so that error is EAGAIN: the
            # request is abandoned (see _service_replay) and live sends resume.
            # 【译】重放在发送线程上处理，所以一个不再读取的重放客户端绝不能拖住实时发布。ROUTER_MANDATORY 把"某个对端的
            #   管道满了"从静默丢弃变成一个错误；而每次重放发送都是NOBLOCK，所以这个错误就是EAGAIN：放弃这次请求
            #   （见 _service_replay），实时发送继续。
            # 观察：本文件里没有 _service_replay（publish里的注释也提到它）；对应的逻辑在 _advance_replay / _replay_send，
            #   疑似改名后注释未同步
            self._replay.setsockopt(zmq.ROUTER_MANDATORY, 1)
            _bind(self._replay, self.replay_endpoint, "replay ROUTER", zmq.ZMQError)
            self._replay_buffer = deque(maxlen=replay_buffer_steps)
        # In-progress replay, if any: (routing prefix, next seq to send). Set
        # by _accept_replay_request, advanced a chunk at a time by
        # _advance_replay so live sends interleave with a long replay.
        # 【译】正在进行中的重放（如果有）：(路由前缀, 下一个要发的seq)。由 _accept_replay_request 设置，由 _advance_replay
        #   每次推进一块，这样实时发送可以和一次很长的重放交替进行。
        self._pending_replay: tuple[list[bytes], int] | None = None
        self._replay_chunk = REPLAY_CHUNK

        # _seq_gen：itertools.count() 每次next()返回0、1、2……（seq的来源）；
        # 下面这些计数器由busy_loop线程（publish）和发送线程共同修改，读写都在 self._lock 下（stats属性读快照）
        # _closing：shutdown后为True，publish直接返回
        # 发送线程 daemon=True：进程退出时不等它（正常关闭走 shutdown() 里的 join）
        self._seq_gen = itertools.count()
        self._drops = 0
        self._sent = 0
        self._replayed = 0
        self._replay_aborted = 0
        self._encode_errors = 0
        self._closing = False
        self._lock = threading.Lock()
        self._sender = threading.Thread(
            target=self._run, name="atom-kv-event-sender", daemon=True
        )
        self._sender.start()

    # 调度器侧（busy_loop线程）每步调用一次：打包成一个EventBatch、编码、非阻塞入队；任何情况下都立刻返回
    # 输入：events（BlockManager.take_events() 取出的本步事件）；输出：无
    # 调用方：Scheduler.publish_kv_events（EngineCore每个engine step结束时的finally里，以及关闭前最后一次）
    def publish(self, events: Iterable[KVCacheEvent]) -> None:
        if self._closing:
            return
        evt_list = list(events)
        if not evt_list:
            return
        batch = EventBatch(
            ts=time.time(),
            events=evt_list,
            data_parallel_rank=self._dp_rank,
        )
        # Assign the sequence number here (at enqueue), before encoding and
        # not at send: a batch lost to an encode failure or dropped on overflow
        # below still consumes a seq, so every loss is visible to subscribers
        # as a gap instead of vanishing silently.
        # Keep seq in [0, 2**64-2] so the wire frame, the replay-buffer key, and
        # the start_seq comparison in _service_replay all use the same value,
        # AND never collide with the reserved all-0xFF REPLAY_DONE terminal
        # (modulo, not mask).
        # 【译】在这里（入队时）分配序号，在编码之前、而不是在发送时：因为编码失败或在下面溢出时被丢掉的批次也会消耗一个seq，
        #   所以每一次丢失对订阅者都可见，表现为一个缺口，而不会悄无声息地消失。
        #   把seq保持在 [0, 2**64-2]，这样线上帧、重放缓冲区的key、以及 _service_replay 里与 start_seq 的比较用的都是同一个值，
        #   而且永远不会与保留的全0xFF的 REPLAY_DONE 终止帧冲突（用取模，不用掩码）。
        # 为什么取模而不是掩码：& (2**64-1) 会留下 2**64-1 这个值（正是REPLAY_DONE）；% (2**64-1) 的结果最大是 2**64-2
        seq = next(self._seq_gen) % 0xFFFFFFFFFFFFFFFF

        try:
            payload = self._encoder.encode(batch)
        except Exception:
            # Surface via stats.encode_errors. Log the first occurrence with
            # traceback so the root cause is discoverable; further failures are
            # tracked via the counter only to avoid log spam.
            # 【译】通过 stats.encode_errors 暴露出来。第一次失败时连同traceback记日志，以便找到根因；之后的失败只计数，避免刷屏。
            # 常见原因：事件里混进了msgspec不会编码的类型（例如array("i")而不是list，见block_manager.py _make_block_stored）
            with self._lock:
                first_failure = self._encode_errors == 0
                self._encode_errors += 1
            if first_failure:
                logger.exception(
                    "KV event encode failed; subsequent failures will be "
                    "tracked via stats['encode_errors']"
                )
            return

        # Non-blocking enqueue; drop oldest on overflow.
        # 【译】非阻塞入队；溢出时丢弃最旧的。
        # put_nowait满了 → get_nowait扔掉队头（最旧的一批）再试；发送线程恰好同时取走了队里的东西时，get_nowait可能扑空
        #   （下面pragma说的"race window"），再循环一次即可
        while True:
            try:
                self._queue.put_nowait((seq, payload))
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._drops += 1
                except queue.Empty:  # pragma: no cover - race window
                    pass

    # 引擎关闭时由 Scheduler.shutdown_kv_events 调用：先让publish不再入队，再往队列里放None哨兵（满了就挤掉最旧的一批），
    #   等发送线程最多2秒
    def shutdown(self) -> None:
        # 【译】（行尾）从这里开始 publish() 会直接返回
        self._closing = True  # publish() will return early from here on
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._drops += 1
                except queue.Empty:
                    pass
        self._sender.join(timeout=2.0)
        # linger：close后ZMQ还会花多久把没发完的消息送出去。发送线程已退出 → 给1000ms把队里剩下的发完；
        #   线程还卡着（2秒没退出）→ 0，直接丢掉，不让关闭再拖下去
        linger = 0 if self._sender.is_alive() else 1000
        # Best-effort close on shutdown: a socket already torn down by the
        # sender thread must not turn shutdown into an error.
        # 【译】关闭时尽力而为：一个已经被发送线程拆掉的socket，不能让shutdown变成一个错误。
        with contextlib.suppress(Exception):  # pragma: no cover
            self._socket.close(linger=linger)
        if self._replay is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._replay.close(linger=0)

    # --- internal ---
    # 【译】--- 内部实现 ---
    def _run(self) -> None:
        # Poll the replay socket between sends. When replay is disabled the
        # queue.get() blocks (timeout=None); when enabled it wakes periodically
        # so replay requests are serviced even while no events are flowing.
        # While a replay is in progress the loop alternates: one chunk of
        # replay, then whatever is on the live queue (without waiting), so a
        # 10k-batch replay to a fast client cannot starve live publication.
        # 【译】在两次发送之间轮询重放socket。不开重放时 queue.get() 一直阻塞（timeout=None）；开了重放时它会定期醒来，
        #   这样即使没有事件在流动，也能处理重放请求。重放进行中时循环交替进行：一块重放，然后是实时队列里已有的东西
        #   （不等待），这样给一个很快的客户端做1万批的重放，也不会饿死实时发布。
        # 发送线程主循环；每轮：①（开了重放）没有进行中的重放且ROUTER上有请求 → 接下它；有进行中的重放 → 推进一块
        #   ② 从实时队列取一批（重放进行中不等待，否则最多等50ms）→ PUB发出 → 进重放缓冲区；取到None（关闭哨兵）就退出
        idle_timeout = 0.05 if self._replay is not None else None
        while True:
            if self._replay is not None:
                try:
                    if self._pending_replay is None and self._replay.poll(0):
                        self._accept_replay_request()
                    if self._pending_replay is not None:
                        self._advance_replay()
                except self._zmq_error_cls:  # pragma: no cover - closed on shutdown
                    return
                except Exception:  # pragma: no cover - replay is non-critical
                    logger.exception("KV event replay request failed")
                    self._pending_replay = None
            get_timeout = 0 if self._pending_replay is not None else idle_timeout
            try:
                item = self._queue.get(timeout=get_timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            seq, payload = item
            try:
                # seq is already masked to uint64 at enqueue.
                # 【译】seq在入队时已经被限制在uint64范围内（实际是取模，见publish）。
                seq_bytes = seq.to_bytes(8, "big")
                # PUB的发送不会阻塞：没有订阅者时消息直接丢掉；有订阅者时进入ZMQ为它维护的队列（受hwm限制）。
                #   只有真正交给了socket的批才进重放缓冲区——所以队列溢出丢掉的批补不回来
                self._socket.send_multipart([self._topic_bytes, seq_bytes, payload])
                if self._replay_buffer is not None:
                    self._replay_buffer.append((seq, seq_bytes, payload))
                with self._lock:
                    self._sent += 1
            except self._zmq_error_cls:  # pragma: no cover - socket closed
                return

    def _accept_replay_request(self) -> None:
        """Take one replay request off the ROUTER and make it the pending
        replay. Request frame is `[client_id, (delim,) start_seq]`; the routing
        prefix is echoed back on every reply frame."""
        # 【译】从ROUTER上取一个重放请求，设为待处理的重放。请求帧是 [client_id, (分隔帧,) start_seq]；
        #   路由前缀在每个回复帧上原样带回。
        # 例：DEALER客户端 send(b"\x00" * 8)（start_seq=0）→ ROUTER收到 [client_id, b"\x00"*8] → 前缀 = [client_id]
        frames = self._replay.recv_multipart()
        # The wire contract is exactly one 8-byte big-endian start_seq after
        # the routing prefix. int.from_bytes would happily read b"" as 0 and
        # a longer frame as a huge value, so a garbled request could trigger a
        # full replay; reject anything but the exact width.
        # 【译】线上约定是：路由前缀之后恰好一个8字节大端的start_seq。int.from_bytes 会乐于把 b"" 读成0、把更长的帧读成一个
        #   巨大的值，所以一个乱码请求可能触发一次完整重放；只接受精确的宽度。
        if len(frames) < 2 or len(frames[-1]) != 8:
            logger.warning("KV event replay: malformed request %r", frames)
            return
        start_seq = int.from_bytes(frames[-1], "big")
        self._pending_replay = (frames[:-1], start_seq)

    def _advance_replay(self) -> None:
        """Send the next chunk of the pending replay: up to `_replay_chunk`
        buffered batches with seq >= the request's start sequence, in order.
        When the buffer is exhausted, send a terminal frame so the consumer
        knows the reply is complete, and clear the pending replay.

        The terminal frame is `[*prefix, REPLAY_DONE, [oldest, latest]]`:
        REPLAY_DONE distinguishes it from data frames, and the msgpack window
        lets the consumer see whether events before `start_seq` were already
        evicted (start_seq < oldest) and terminate without a timeout even on a
        zero-match request.

        Ordering is plain integer comparison. A window straddling the 2**64-1
        wrap would misorder, but the counter starts at 0 per process and steps
        by one per batch, so reaching the wrap is not achievable in practice
        (~1.8e19 batches). Not handled by design; see the class docstring."""
        # 【译】发送待处理重放的下一块：最多 _replay_chunk 个 seq >= 请求起始序号 的缓冲批次，按顺序。缓冲区耗尽时，发一个
        #   终止帧让消费者知道回复已经完整，并清除待处理的重放。
        #
        #   终止帧是 [*prefix, REPLAY_DONE, [oldest, latest]]：REPLAY_DONE 把它与数据帧区分开；msgpack编码的窗口让消费者
        #   看出 start_seq 之前的事件是否已被淘汰（start_seq < oldest），而且即使请求一条也没匹配上，也能不靠超时就结束。
        #
        #   排序就是普通的整数比较。跨过2**64-1回绕点的窗口会排错序，但计数器在每个进程里从0开始、每批加一，实际上到不了
        #   回绕点（约1.8e19批）。按设计不处理；见类的docstring。
        # 例：缓冲区里是seq 100~199。客户端请求start_seq=150 → 本轮发150~199（不到64批）+ 终止帧，窗口[100, 199]；
        #   请求start_seq=50 → 第一轮发100~163，下一轮发164~199 + 终止帧[100, 199]，客户端据 50 < 100 知道50~99已丢失
        assert self._pending_replay is not None
        prefix, next_seq = self._pending_replay
        # Safe to iterate the deque directly: the sender thread is the only
        # mutator and it is the same thread running this method. Batches
        # appended by live sends between chunks simply extend the replay,
        # which is what a consumer catching up wants.
        # 【译】直接遍历deque是安全的：发送线程是唯一的修改者，而运行本方法的就是同一个线程。两块之间由实时发送追加的批次
        #   只是延长了这次重放，这正是一个正在追赶的消费者想要的。
        buf = self._replay_buffer or ()
        sent = 0
        for seq, seq_bytes, payload in buf:
            if seq < next_seq:
                continue
            if not self._replay_send([*prefix, seq_bytes, payload]):
                self._pending_replay = None
                return
            with self._lock:
                self._replayed += 1
            next_seq = seq + 1
            sent += 1
            if sent >= self._replay_chunk:
                self._pending_replay = (prefix, next_seq)
                return
        # Terminal frame with the available window. Encode with the module
        # helper (fresh encoder) rather than self._encoder, which the scheduler
        # thread uses concurrently in publish().
        # 【译】带上可用窗口的终止帧。用模块级的辅助函数（每次新建编码器）编码，而不是 self._encoder，因为调度器线程在
        #   publish() 里会同时使用它。
        oldest = buf[0][0] if buf else None
        latest = buf[-1][0] if buf else None
        window = msgspec.msgpack.encode([oldest, latest])
        self._replay_send([*prefix, REPLAY_DONE, window])
        self._pending_replay = None

    def _replay_send(self, frames: list[bytes]) -> bool:
        """Non-blocking send on the replay ROUTER. Returns False when the
        client's pipe is full (or the client is gone), in which case the caller
        abandons the rest of this replay: the consumer is not draining, and
        blocking here would stall the live PUB stream that shares this thread.
        The consumer sees a missing REPLAY_DONE and can re-request."""
        # 【译】在重放ROUTER上非阻塞发送。客户端的管道满了（或客户端已经走了）时返回False，此时调用方放弃这次重放的剩余
        #   部分：消费者没在读取，而在这里阻塞会拖住共用这个线程的实时PUB流。消费者看到缺少REPLAY_DONE，可以重新请求。
        try:
            self._replay.send_multipart(frames, flags=self._zmq_noblock)
            return True
        except self._zmq_again_cls:
            reason = "client not draining"
        # 【译】（下一行行尾）EHOSTUNREACH：对端已断开（ROUTER_MANDATORY下发往不存在的对端会报这个错）
        except self._zmq_error_cls as e:  # EHOSTUNREACH: peer disconnected
            reason = f"{e.__class__.__name__}: {e}"
        with self._lock:
            self._replay_aborted += 1
            first = self._replay_aborted == 1
        if first:
            logger.warning(
                "KV event replay abandoned (%s); further aborts are counted in "
                "stats['replay_aborted']",
                reason,
            )
        return False

    # Test/diagnostic hooks.
    # 【译】测试/诊断用的钩子。
    # 返回五个计数器的快照（加锁读取）：sent已发、dropped队列溢出丢弃、replayed已重放、replay_aborted放弃的重放、encode_errors编码失败
    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sent": self._sent,
                "dropped": self._drops,
                "replayed": self._replayed,
                "replay_aborted": self._replay_aborted,
                "encode_errors": self._encode_errors,
            }


def make_publisher(
    enabled: bool,
    publisher_kind: str,
    endpoint: str,
    *,
    topic: str = "",
    hwm: int = 0,
    buffer_steps: int = 10_000,
    replay_endpoint: str = "",
    replay_buffer_steps: int = 10_000,
    data_parallel_rank: int | None = None,
) -> EventPublisher:
    """Construct a publisher from plain-config args. Returns `NullEventPublisher`
    when disabled, so callers can always call `publish()` without checking."""
    # 【译】用普通的配置参数构造一个发布器。关闭时返回 NullEventPublisher，这样调用方总可以直接调 publish() 而不用检查。
    # 调用方：Scheduler.__init__（scheduler.py）：开启且是PP head stage时传 KVEventsConfig 的各字段；否则 enabled=False、kind="null"
    if not enabled or publisher_kind == "null":
        return NullEventPublisher()
    if publisher_kind == "zmq":
        return ZmqEventPublisher(
            endpoint=endpoint,
            topic=topic,
            hwm=hwm,
            buffer_steps=buffer_steps,
            replay_endpoint=replay_endpoint,
            replay_buffer_steps=replay_buffer_steps,
            data_parallel_rank=data_parallel_rank,
        )
    raise ValueError(f"unknown KV event publisher: {publisher_kind!r}")
