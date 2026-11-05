# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State checkpoints backed by arbitrary PAGE-sized physical units."""

# 【译】由任意的、PAGE 大小的物理单元承载的状态 checkpoint。
#
# =====================================================================================
# 【导读】page_unit_checkpoint.py —— copy 方式的状态快照：把活跃槽位的字节切成"页单元"存进 KV 池（第4批 4b，B级）
#   前置：为什么要状态快照、命中怎么退到最近的快照、页单元是什么、为什么不用 fork——见 state_pool.py 文件开头【科普】②③④⑤
#     （一句话：页单元 = KV 池里一个 block 的全部字节；一份快照 = units_per_checkpoint 个不必连续的 block，例如 Kimi-K3 是 127 个）
#   谁用：v0.1.8 中 Qwen3-Next / Qwen3.5（GDN）、Kimi-K3 / GLM-5.3（KDA）在 PP=1 且非 RapidServe 时，以及 DeepSeek-V4
#     （各后端 state_transfer() 返回 StateTransfer.copy(layout_id)）；BlockManager 此时构造 PagedStateCheckpointCoordinator，
#     它就是 BlockManager.state_caches 的唯一成员，StateSlotPool 只负责分配活跃槽位
#   本文件结构：
#     PagedStateCheckpointSpec / checkpoint_copy_ops_per_batch / CheckpointStoreOp / CheckpointRestoreOp    几何与给 runner 的操作描述
#     CheckpointRecord / _OffloadPin / SuspendedCheckpointRestore                                         内部记录
#     PageUnitCheckpointStore             "仓库"：hash → 快照记录、页单元归属、LRU 淘汰、CPU 层的 pin
#     PagedStateCheckpointCoordinator     "调度"：实现 state_cache.py 的协议，决定每个 batch 存哪些、取哪些
#
#   一份快照记录的状态机（PageUnitCheckpointStore）：
#
#     begin_store ──> COPYING ──complete_inflight（下一轮开头：它的 scatter 已随上一批发出）──> READY ──┐
#      （借好页单元，                                                        （进 hash_to_checkpoint 与 _lru， │ _evict（LRU 淘汰，需未钉住）
#        还不可见）                                                            可被命中、可被提名到 CPU 层）  │ unindex（边界 KV block 被淘汰）
#          │ unindex / clear 时拷贝仍在途                                                                    ▼
#          └────────────────────────────────────────────────────────────> EVICTING ──pin 归零、拷贝结束──> _release_record（页单元还回 KV 池）
#
#   一条请求带数字走一遍（一格 = 一轮 Scheduler.schedule；ATOM 每轮先结算上一批、再准入、最后构造本批）：
#     存：请求A 第 N 批 forward 恰好结束在 8192（阶梯）
#       第 N 轮 postprocess  BlockManager.hash_blocks → checkpoint(A, 512, h)：只记意图 _pending[(id(A), h)]
#       第 N+1 轮构造 batch   take_checkpoint_ops → begin_store(h, A.state_slot)：向 KV 池借 units_per_checkpoint 个 block，记录为 COPYING；
#                            CheckpointStoreOp 随第 N+1 批在 forward 之前执行（scatter），所以拷到的正是"第 N 批结束时"的状态
#       第 N+2 轮开头         complete_previous_batch → complete_inflight：变 READY，从此可被命中；开了 CPU 层时同时提名给 offload
#     取：请求B 在第 M 轮准入，命中 h
#       第 M 轮准入          BlockManager._attach_state_slots：pop_many 拿新活跃槽位 s' → begin_restore(h, s')：pin_count+1、刷新 LRU、排队
#       第 M 轮构造 batch     take_restore_ops：CheckpointRestoreOp 进本批 StateMaintenanceOps，B 的第一次 forward 之前 gather 回 s'
#       第 M+1 轮开头         complete_inflight 释放这个 restore pin（快照仍在索引里，别的请求还能命中它）
#
#   和 KV 抢空间：页单元就是 KV 池（BlockManager.kv，BlockPool）的 block。begin_store 只要 units_per_checkpoint 个；
#     KV 池空闲表（含只是缓存着旧前缀的 block）不够时，才按快照 LRU 整份淘汰（ensure_free_units）；
#     反过来 KV 分配（BlockManager._fresh_block）也经 ensure_free_units 拿 block，所以 READY 且未钉住的快照对活的 KV 而言等同空闲
#   对比nano-vllm：nano-vllm 的 KV 池里只有 KV block；这里同一个池里还住着状态快照，并且 BlockManager 的准入门
#     （_has_page_units / _checkpoint_has_room）要把它们一起算进去。
# =====================================================================================

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass
from time import monotonic

from atom.kv_transfer.disaggregation.types import StateStoreOperationId
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence

_CHANGE_LOG_LIMIT = 4096
# generation 变更日志最多记多少条（见 PageUnitCheckpointStore._changes / changed_since）

logger = logging.getLogger("atom")

#: How many reclaimed store operations to remember. Only has to outlive a
#: report already on the wire, which is one step, so this is generous.
# 【译】要记住多少个被回收的存储操作。它只需要比一份已经在路上的报告活得久，而那只是一步，所以这个值很宽裕。
_RECLAIMED_MEMORY = 4096

COPYING = "COPYING"
READY = "READY"
EVICTING = "EVICTING"
# 快照记录的三个状态（状态机见文件开头）：COPYING = 页单元已借、scatter 还没随 batch 跑完，不可见；
#   READY = 可被命中；EVICTING = 已不可见，等最后一个读者 / 写者结束后释放页单元


@dataclass(frozen=True)
class PagedStateCheckpointSpec:
    """Runtime geometry for PAGE-backed state checkpoints."""

    # 【译】PAGE 承载的状态 checkpoint 的运行时几何。
    # 构造：ModelRunner.get_num_blocks（model_runner.py）：
    #   page_unit_bytes = PAGE 子池一个 entry 的字节数（后端的 checkpoint_page_unit_bytes 可以把它收窄为"传输实际发布的字节"）
    #   slot_bytes      = 一个活跃槽位（Pool.STATE 的 entry）的字节数
    #   image_bytes     = 快照实际要存的字节数（后端 checkpoint_image_bytes 收窄；None 则等于 slot_bytes）
    #   layout_id       = 后端的版本化布局 id（必须与 StateTransfer.paged_layout_id 一致，StateRuntime 校验）
    # 经 StateRuntime.to_wire / from_wire 传到 EngineCore，再交给 BlockManager

    page_unit_bytes: int
    slot_bytes: int
    layout_id: str
    # Bytes of a slot a checkpoint image actually holds, which is less than
    # all of them: a resumer reads only part of the slot it resumes into, and
    # a compressor whose next pool starts exactly at the boundary reads none
    # of its own. `slot_bytes` stays for three things that still want the
    # whole slot — the `image_bytes <= slot_bytes` sanity check below, the
    # geometry cross-check in `allocate_per_req_cache`, and the startup log
    # line that reports an image as a fraction of one.
    # 【译】一份 checkpoint 镜像实际持有的槽位字节数，它比全部字节少：恢复者只读它恢复进去的槽位的一部分，
    #   而一个下一个池恰好从边界开始的压缩器，它自己的那部分一字节都不读。slot_bytes 仍留着做三件仍然需要整个槽位的事——
    #   下面 image_bytes <= slot_bytes 的健全性检查、allocate_per_req_cache 里的几何交叉检查，以及把镜像报告为槽位一部分的那行启动日志。
    # 例：DeepSeek-V4 的 HCA 状态占槽位的 52%、恢复时完全用不到，不计入 image（docs/scheduling_kv_cache_guide.md "State checkpoints"）
    image_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("page_unit_bytes", self.page_unit_bytes),
            ("slot_bytes", self.slot_bytes),
            ("image_bytes", self.image_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_bytes > self.slot_bytes:
            raise ValueError(
                f"image_bytes {self.image_bytes} exceeds the {self.slot_bytes} "
                "a slot holds"
            )
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("paged state checkpoints need a non-empty layout id")

    # 一份快照占几个页单元 = ceil(image_bytes / page_unit_bytes)；例：Kimi-K3 是 127（kimi_mla_gdn_attn.py state_transfer 的 docstring）
    @property
    def units_per_checkpoint(self) -> int:
        return (self.image_bytes + self.page_unit_bytes - 1) // self.page_unit_bytes

    def to_wire(self) -> dict[str, int | str]:
        return {
            "page_unit_bytes": self.page_unit_bytes,
            "slot_bytes": self.slot_bytes,
            "image_bytes": self.image_bytes,
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_wire(cls, wire: object) -> PagedStateCheckpointSpec:
        if not isinstance(wire, Mapping):
            raise TypeError("paged state checkpoint spec must be a mapping")
        expected = {"page_unit_bytes", "slot_bytes", "image_bytes", "layout_id"}
        if set(wire) != expected:
            raise ValueError(
                "invalid paged state checkpoint spec fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            page_unit_bytes=wire["page_unit_bytes"],  # type: ignore[arg-type]
            slot_bytes=wire["slot_bytes"],  # type: ignore[arg-type]
            image_bytes=wire["image_bytes"],  # type: ignore[arg-type]
            layout_id=wire["layout_id"],  # type: ignore[arg-type]
        )


def checkpoint_copy_ops_per_batch(max_num_seqs: int) -> int:
    """PAGE checkpoint copies one batch's descriptor holds, stores and restores.

    One store and one restore per sequence. The runner sizes its pinned
    descriptor staging with this (`AttentionMetadataBuilder.
    _checkpoint_staging`), and the coordinator bounds how many queued midstep
    stores one batch begins with it (`attach_midstep_scratch`). Both read it
    from here: a coordinator that began more than the staging holds would make
    `build()` raise mid-serving.
    """
    # 【译】一个 batch 的描述符能装下的 PAGE checkpoint 拷贝数，存与取合计。
    #   每条序列一次存、一次取。runner 用它来定它的 pinned 描述符暂存区大小（AttentionMetadataBuilder._checkpoint_staging），
    #   协调器用它来限制一个 batch 开始多少个排队的 midstep 存储（attach_midstep_scratch）。两边都从这里读：
    #   一个开始了超过暂存区容量的协调器，会让 build() 在服务中途抛异常。
    # 例：max_num_seqs=256 → 512。调用方：model_ops/attentions/backends.py（暂存区）、BlockManager.__init__（attach_midstep_scratch）
    return 2 * int(max_num_seqs or 0)


@dataclass(frozen=True)
class CheckpointStoreOp:
    """Scatter the checkpointed part of an Active Slot into PAGE units."""

    # 【译】把一个活跃槽位里要 checkpoint 的那部分 scatter（分散写）进若干 PAGE 单元。
    # src_slot = 源活跃槽位；unit_ids = 按顺序的页单元（KV block id）；total_bytes = image_bytes；layout_id 用于 runner 校验
    # 执行：各后端的 execute_paged_state_copies（gdn_attn.py、kimi_mla_gdn_attn.py、deepseek_v4_attn.py），由 build 在 forward 前调用

    src_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    layout_id: str


@dataclass(frozen=True)
class CheckpointRestoreOp:
    """Gather one ordered PAGE-unit image back into an Active Slot."""

    # 【译】把一份按顺序排列的 PAGE 单元镜像 gather（收集）回一个活跃槽位。

    dst_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    layout_id: str


# 一份快照的记录（PageUnitCheckpointStore.records 的值）：
#   prefix_hash = 它登记的链式 hash（边界前最后一个 block 的 hash）；unit_ids = 它占的页单元；
#   state = COPYING / READY / EVICTING；pin_count = 正在读它的 restore 数 + 正在把它拷到 CPU 的 offload 数（>0 时不能淘汰）
@dataclass
class CheckpointRecord:
    prefix_hash: int
    unit_ids: tuple[int, ...]
    state: str = COPYING
    pin_count: int = 0


@dataclass(frozen=True)
class SuspendedCheckpointRestore:
    """A queued restore removed from execution while retaining its source pin."""

    # 【译】一个被移出执行队列、但仍保留着对源快照的 pin 的排队恢复。
    # 【分支】只被 kv_transfer/offload/mp/native_state_scheduler.py（LMCache MP 的整份快照外部恢复）使用，见第9批

    checkpoint_id: int
    operation: CheckpointRestoreOp


@dataclass
class _OffloadPin:
    """A store that has been dispatched to the worker but not yet indexed.

    Two-phase, because a store's life has two independent ends. The PAGE units
    it gathers are the KV pool's and are free the moment the D2H gather drains
    (`source_released`); whether the CPU `batched_put` then succeeded is decided
    strictly afterwards and cannot touch them. The pin therefore holds the units
    only until the source is released, but the *record itself* stays in
    `_offload_pins` until the index report lands -- so `has_offload_pins` and
    `_hash_in_flight`, which mean "dispatched but unreported", keep answering
    from dispatch all the way to the report, across the gap the source release
    opens in the middle.
    """

    # 【译】一个已经派发给 worker、但还没被登记进索引的存储。
    #   分两阶段，因为一个存储的生命有两个相互独立的终点。它 gather 的 PAGE 单元属于 KV 池，D2H gather 一结束就空出来了
    #   （source_released）；而 CPU 上的 batched_put 是否成功，严格地在那之后才确定，而且碰不到这些单元。所以 pin 只持有单元
    #   直到源被释放，但"记录本身"要一直留在 _offload_pins 里直到索引报告到达——这样 has_offload_pins 和 _hash_in_flight
    #   （意思是"已派发、未报告"）从派发一直到报告都能正确回答，跨过源释放在中间打开的那个空档。
    # 【分支】CPU 层（state_offload.py），见第9批

    checkpoint_id: int
    pinned_at: float
    source_released: bool = False
    # Explicit connector leases may still have DMA reading their PAGE units
    # after a timeout. Only a release or terminal completion can end them.
    # 【译】显式的 connector 租约在超时之后仍可能有 DMA 在读它们的 PAGE 单元。只有释放或终结性的完成才能结束它们。
    timeout_reclaimable: bool = True


class PageUnitCheckpointStore:
    """Content index and ownership table for split state images."""

    # 【译】被切分的状态镜像的内容索引和归属表。
    # 由 PagedStateCheckpointCoordinator 创建并独占（self.store）；pool = BlockManager.kv（同一个 KV BlockPool）
    # 成员一览：
    #   hash_to_checkpoint  {hash: checkpoint_id}   只放 READY 的；lookup 还会再确认 state == READY
    #   records             {checkpoint_id: CheckpointRecord}   所有活着的记录（含 COPYING / EVICTING）
    #   _pending_by_hash    {hash: checkpoint_id}   COPYING 中的，防止同一 hash 被重复存
    #   _lru                OrderedDict[checkpoint_id]   READY 的淘汰顺序，左 = 最久没用（_next_victim 取第一个可淘汰的）
    #   _inflight_stores / _queued_restores / _inflight_restores   本批 / 上批的拷贝，complete_inflight 结算
    #   _offload_ready / _offload_pins / _offload_reclaimed        【分支】CPU 层的提名、在途 pin 与被回收的操作
    #   generation / _changes                                      READY 集合的变更计数与日志（供外部缓存 contains 的答案）

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
        offload_sink: bool = False,
    ):
        self.pool = pool
        self.spec = spec
        # Whether anything downstream can carry a store. False leaves the
        # offload queue permanently empty and takes no pins, which is what a
        # deployment with no CPU tier must cost: a pin nobody releases would
        # hold 127 blocks per checkpoint out of the pool forever.
        # 【译】下游有没有东西能承接一次存储。False 让 offload 队列永远为空、也不取任何 pin——这正是没有 CPU 层的部署该付的代价：
        #   一个没人释放的 pin 会让每个 checkpoint 永远从池里扣住 127 个 block。
        self._offload_sink = offload_sink
        self.hash_to_checkpoint: dict[int, int] = {}
        # Bumped whenever the READY set can change (publish, unindex, evict,
        # release, clear), so callers can cache `contains` answers. The log
        # names the hash behind each recent bump (None: every hash), so a
        # caller can invalidate only the answers that bump can affect.
        # 【译】每当 READY 集合可能变化时（publish、unindex、evict、release、clear）就加一，这样调用方可以缓存 contains 的答案。
        #   日志记下每次最近加一背后的 hash（None：所有 hash），所以调用方只需让那次加一可能影响到的答案失效。
        # 调用方：kv_transfer/offload/mp/native_state_scheduler.py（用 generation + changed_since 做 memo）
        # deque(maxlen=N)：满了以后再 append 会自动从另一端挤掉最旧的
        self.generation = 0
        self._changes: deque[tuple[int, int | None]] = deque(maxlen=_CHANGE_LOG_LIMIT)
        self.records: dict[int, CheckpointRecord] = {}
        self._pending_by_hash: dict[int, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._inflight_stores: list[int] = []
        self._queued_restores: list[tuple[int, CheckpointRestoreOp]] = []
        self._inflight_restores: list[int] = []
        self._next_checkpoint_id = 0
        self.evictions = 0
        # Checkpoints that reached READY and have not been offered to the CPU
        # tier yet. Candidates only: NOT pinned, so one still counts as free
        # space and `_next_victim` may spend it. The pin is taken later, in
        # `take_offload_stores`, and only for the few actually in flight.
        # 【译】已经 READY、还没被提交给 CPU 层的 checkpoint。只是候选：没有被钉住，所以仍算作空闲空间，_next_victim 可以花掉它。
        #   pin 稍后在 take_offload_stores 里才取，而且只对真正在途的那少数几个取。
        self._offload_ready: deque[int] = deque()
        # Ceiling on that backlog. Under sustained tier backpressure the drain
        # loop in `take_offload_stores` exits at its first iteration -- pins
        # already at `max_inflight` -- and so never reaches the stale-drop that
        # otherwise clears spent nominations. An unbounded nomination stream
        # would then grow `_offload_ready` without limit with ids whose
        # checkpoints have long since been evicted. A nomination is unpinned and
        # the pool may spend its checkpoint anyway, so dropping the oldest
        # (coldest) on overflow costs a missed offload, never a lost image: the
        # checkpoint stays HBM-resident until it is evicted. Sized well above
        # any real in-flight-plus-waiting count so it only bites a runaway.
        # 【译】这个积压的上限。在持续的层背压下，take_offload_stores 里的排空循环会在第一次迭代就退出——pin 已经到了 max_inflight——
        #   于是永远走不到平时会清掉已失效提名的那段丢弃逻辑。一个无界的提名流会让 _offload_ready 无限增长，里面的 id 对应的
        #   checkpoint 早就被淘汰了。提名没有被钉住，池子无论如何都可能花掉它的 checkpoint，所以溢出时丢掉最老（最冷）的，
        #   代价是错过一次 offload，绝不会丢掉一份镜像：checkpoint 会一直留在 HBM 里直到被淘汰。
        #   上限设得远高于任何真实的"在途 + 等待"数量，所以它只会咬住失控的情况。
        self._offload_backlog_cap = 8192
        self.offload_nominations_dropped = 0
        # StateStoreOperationId -> _OffloadPin (checkpoint, when pinned, whether
        # the source units are already back).
        # A store's report carries no request -- by the time one lands its owner
        # is long gone -- so the hash is the only thing left to name it by. The
        # hash ALONE is not an identity though: the same prefix is stored again
        # after an eviction or a load miss, and `KVOutputAggregator` tombstones
        # every `(channel, operation_id)` it has taken quorum on, so the second
        # store under a bare hash was dropped as a duplicate -- its pin held
        # until stale reclamation and its bytes never re-indexed. The
        # generation separates the attempts, and keying the pins by the whole
        # operation is also what stops a late report from an earlier attempt
        # from settling a newer pin for the same hash.
        # 【译】StateStoreOperationId -> _OffloadPin（哪个 checkpoint、何时钉住、源单元是否已经还回）。
        #   一份存储的报告不带请求——等它到达时，它的主人早走了——所以只剩 hash 能用来指称它。但光有 hash 并不是身份：
        #   同一个前缀在一次淘汰或一次加载 miss 之后会被再存一次，而 KVOutputAggregator 会给它取得过法定数的每个
        #   (channel, operation_id) 打上墓碑，于是以裸 hash 为键的第二次存储被当成重复丢掉了——它的 pin 一直扣到陈旧回收，
        #   它的字节也再没被登记。generation 把各次尝试分开；以整个操作为键，也是阻止"较早一次尝试的迟到报告去了结同一 hash
        #   的较新 pin"的办法。
        self._offload_pins: dict[StateStoreOperationId, _OffloadPin] = {}
        self._offload_generation = 0
        self.offload_pins_reclaimed = 0
        # Operations whose pin the stale reclaimer took back. Bounded, because
        # this only has to outlive a report already on the wire. A store whose
        # source was reclaimed underneath it may not be indexed -- see
        # `was_reclaimed`.
        # 【译】被陈旧回收器收回了 pin 的操作。有上限，因为它只需比一份已经在路上的报告活得久。一个源在它底下被回收的存储
        #   不能被登记——见 was_reclaimed。
        self._offload_reclaimed: OrderedDict = OrderedDict()

    @property
    def units_per_checkpoint(self) -> int:
        return self.spec.units_per_checkpoint

    # lookup：READY 的快照 id，否则 -1；contains：是否有 READY 的；contains_or_pending：READY 或正在 COPYING
    # 调用方：本类 begin_store / begin_restore / acquire_checkpoint_source；协调器 _reachable / contains / reserve_midstep / take_checkpoint_ops；
    #   native_state_scheduler.py（contains）
    def lookup(self, prefix_hash: int) -> int:
        checkpoint_id = self.hash_to_checkpoint.get(prefix_hash, -1)
        record = self.records.get(checkpoint_id)
        if record is None or record.state != READY:
            return -1
        return checkpoint_id

    def contains(self, prefix_hash: int) -> bool:
        return self.lookup(prefix_hash) >= 0

    def contains_or_pending(self, prefix_hash: int) -> bool:
        return self.contains(prefix_hash) or prefix_hash in self._pending_by_hash

    def _new_identity(self) -> int:
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        return checkpoint_id

    def _is_evictable(self, checkpoint_id: int, protected: int = -1) -> bool:
        """Whether this checkpoint may be spent. Eligibility, not policy.

        The one statement of what is evictable. Everything that asks about
        free units goes through it, so a new state, a grace period or a
        second kind of pin cannot leave two answers behind. Do not order
        here -- which eligible checkpoint to spend first is `_next_victim`.
        """
        # 【译】这个 checkpoint 能否被花掉。说的是资格，不是策略。
        #   这是"什么可淘汰"的唯一表述。所有问空闲单元的地方都经过它，所以新增一个状态、一个宽限期或第二种 pin，
        #   都不会留下两种答案。不要在这里排序——先花哪一个有资格的，是 _next_victim 的事。
        # 条件：不是调用方要保护的那个（protected，即本次准入马上要钉住的快照）、READY、pin_count == 0
        record = self.records[checkpoint_id]
        return (
            checkpoint_id != protected
            and record.state == READY
            and record.pin_count == 0
        )

    def _evictable(self, protected: int = -1) -> Iterator[int]:
        """Every checkpoint that may be spent. Yield order carries no promise."""
        # 【译】所有可以被花掉的 checkpoint。产出的顺序不作任何承诺。
        return (cid for cid in self._lru if self._is_evictable(cid, protected))

    def _next_victim(self, protected: int = -1) -> int:
        """Which eligible checkpoint to spend when the free list is short.

        This is the eviction policy, and the only place it lives: least
        recently used, which `_lru` already orders. A different policy
        replaces this method and nothing else -- in particular it must not
        touch `_is_evictable`, which is the eligibility rule three callers
        share.
        """
        # 【译】空闲表不够时花掉哪一个有资格的 checkpoint。
        #   这就是淘汰策略，而且只住在这里：最近最少使用，_lru 已经排好了。换一种策略只需替换这个方法，别的都不用动——
        #   尤其不能碰 _is_evictable，那是三个调用方共享的资格规则。
        return next(self._evictable(protected), -1)

    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        """Whether `count` units could be had, evicting if it came to that.

        Asked once per waiting sequence in `can_allocate` and once per running
        one in `can_append`, so it is per-sequence per-pass and the walk has to
        be paid for. Two things keep it cheap. The free list is checked first,
        which is the whole answer whenever the pool is not tight. And the walk
        below stops at the shortfall rather than totalling the cache: the
        question is whether the eligible set reaches `count`, not how large it
        is, and a warm pool holds `num_kvcache_blocks / units_per_checkpoint`
        checkpoints -- thousands, walked for an answer a couple of them settle.

        Which checkpoints those are does not change the answer, only how soon
        the loop reaches it, so a future `_next_victim` cannot move this gate.
        """
        # 【译】在必要时淘汰的前提下，能否凑出 count 个单元。
        #   can_allocate 里每个等待中的序列问一次，can_append 里每个运行中的序列问一次，所以是"每序列每轮"，遍历必须付得起。
        #   两点让它便宜。先查空闲表，只要池子不紧，那就是全部答案。而下面的遍历在补足缺口时就停，而不是把缓存总计一遍：
        #   问题是有资格的集合能否达到 count，而不是它有多大；一个热的池子装着 num_kvcache_blocks / units_per_checkpoint 个
        #   checkpoint——成千上万个——而答案通常两三个就定了。
        #
        #   具体是哪几个只影响循环多快到达答案，不影响答案本身，所以将来换掉 _next_victim 也挪不动这道门。
        # 调用方：BlockManager._has_page_units（准入 / 追加时 KV + 草稿 lookahead 的块数）、_checkpoint_has_room（按需快照前：本次要的块 + 一份镜像）
        # 例：pool.num_free=50，count=200，快照每份 127 个单元 → 缺 150：数到第 2 份（254 ≥ 150）就返回 True
        if count <= self.pool.num_free:
            return True
        protected = self.lookup(protected_hash) if protected_hash is not None else -1
        shortfall = count - self.pool.num_free
        for checkpoint_id in self._evictable(protected):
            shortfall -= len(self.records[checkpoint_id].unit_ids)
            if shortfall <= 0:
                return True
        return False

    def ensure_free_units(self, count: int) -> bool:
        """Raise the free list to `count`, spending checkpoints for the shortfall.

        Free units are taken first -- a caller asking for what is already
        there evicts nothing -- and `pop` hands out never-used blocks before
        cached ones, so a store reaches for the cache only once the pool has
        nothing spare. Each eviction returns a whole image's units, so the
        loop overshoots by at most one checkpoint.

        Unreachable counts are refused before anything is spent. The loop
        alone gives up only once it has evicted everything it can, so a count
        the cache cannot reach would destroy the cache on the way to saying
        no. The test lives here rather than in the one caller that used to
        carry it, because every caller needs it and only the argument being
        1 keeps `_fresh_block` from needing it today.
        """
        # 【译】把空闲表提高到 count，为缺口花掉 checkpoint。
        #   先用空闲单元——问的就是已有的量时什么都不淘汰——而 pop 会先发从没用过的 block、再发缓存着的，所以存储只在池子
        #   一点富余都没有时才伸手去动缓存。每次淘汰还回一整份镜像的单元，所以循环最多多出一个 checkpoint。
        #
        #   够不到的数量在花掉任何东西之前就被拒绝。光靠循环，只有在淘汰完能淘汰的一切之后才会放弃，所以一个缓存够不到的数量
        #   会在说"不"的路上把缓存毁掉。这个检查放在这里而不是放在以前带着它的那一个调用方里，因为每个调用方都需要它，
        #   只是参数为 1 才让 _fresh_block 今天不需要它。
        # 调用方：本类 begin_store；协调器 reserve_transfer_units；BlockManager._ensure_page_units（_fresh_block、disown_claimed_prefix）
        if not self.has_available_units(count):
            return False
        while self.pool.num_free < count:
            victim = self._next_victim()
            if victim < 0:
                return False
            self._evict(victim)
        return True

    # begin_store：为 hash 开始一次存储——借 units_per_checkpoint 个页单元、建 COPYING 记录、返回给 runner 的 CheckpointStoreOp。
    #   已有 READY 或 COPYING 的同 hash、或凑不出单元时返回 None（调用方计 checkpoints_dropped）
    # 调用方：协调器 take_checkpoint_ops / _take_scratch_stores（构造 batch 时）
    # 页单元的归属标记为 ("state-checkpoint", checkpoint_id)，BlockPool.release_units 时会校验，防止张冠李戴
    def begin_store(self, prefix_hash: int, src_slot: int) -> CheckpointStoreOp | None:
        if self.lookup(prefix_hash) >= 0 or prefix_hash in self._pending_by_hash:
            return None
        needed = self.units_per_checkpoint
        # A store takes what its own image needs and nothing more. It used to
        # take a floor for live KV on top, which meant one accepted store
        # spent tens of checkpoints to build a cushion -- and the cushion
        # bought nothing: the pool cannot starve live KV. A READY unpinned
        # checkpoint is already counted as available by `has_available_units`,
        # so holding one costs live KV nothing; the unevictable set (COPYING,
        # or pinned by a restore) is created after every allocation in a pass
        # and resolved before the next one allocates; and every `_fresh_block`
        # sits behind a pin-aware check in its own pass, so the reachable
        # outcome is a refused admission, never the raise.
        #
        # A store that will be dropped has to cost nothing, which is what
        # `ensure_free_units` refusing before it evicts buys. Its answer is
        # read rather than assumed: `_next_victim` is meant to be replaced,
        # and a policy that passes over an eligible checkpoint would leave the
        # loop short after spending some -- taking an identity and a record
        # for a store that cannot happen would then be the second cost.
        # 【译】一次存储只拿它自己镜像需要的，一点不多。它以前还要额外为活的 KV 拿一个保底量，这意味着一次被接受的存储要花掉
        #   几十个 checkpoint 来垫一个缓冲——而这个缓冲什么也买不到：池子不会饿死活的 KV。一个 READY 且未钉住的 checkpoint
        #   已经被 has_available_units 算作可用，所以持有它不让活的 KV 付出任何代价；不可淘汰的集合（COPYING，或被某次恢复钉住）
        #   是在一轮里所有分配之后才产生、并在下一轮分配之前解决的；而每个 _fresh_block 都在它自己那一轮里有一道感知 pin 的检查，
        #   所以可达的结果是一次被拒绝的准入，绝不是那个异常。
        #
        #   一次将被丢弃的存储必须什么都不花，这就是"ensure_free_units 在淘汰之前先拒绝"换来的。它的答案是读出来的而不是假定的：
        #   _next_victim 是准备被替换的，一个跳过某个有资格 checkpoint 的策略会在花掉一些之后让循环仍然不够——那时再为一次不可能
        #   发生的存储拿一个身份和一份记录，就是第二重代价了。
        if not self.ensure_free_units(needed):
            return None

        checkpoint_id = self._new_identity()
        owner = ("state-checkpoint", checkpoint_id)
        unit_ids = self.pool.reserve_units(needed, owner)
        if unit_ids is None:
            return None
        record = CheckpointRecord(
            prefix_hash=prefix_hash,
            unit_ids=tuple(unit_ids),
        )
        self.records[checkpoint_id] = record
        self._pending_by_hash[prefix_hash] = checkpoint_id
        self._inflight_stores.append(checkpoint_id)
        return CheckpointStoreOp(
            src_slot=src_slot,
            unit_ids=record.unit_ids,
            total_bytes=self.spec.image_bytes,
            layout_id=self.spec.layout_id,
        )

    def restore_queued_for(self, dst_slot: int) -> bool:
        """Whether a queued restore will write `dst_slot` on the next batch."""
        # 【译】是否有一个排队中的恢复会在下一个 batch 写 dst_slot。
        return any(op.dst_slot == dst_slot for _cid, op in self._queued_restores)

    # begin_restore：为命中的 hash 排一次恢复到 dst_slot：pin_count+1（防止被淘汰）、挪到 LRU 尾部（被用 = 刷新）、进 _queued_restores
    #   没有 READY 的快照时返回 None。调用方：协调器 begin_restore ← BlockManager._attach_state_slots
    # take_restore_ops：构造 batch 时取走排队的恢复，移进 _inflight_restores（下一轮 complete_inflight 释放 pin）
    # cancel_queued_restore：取消写某个槽位的排队恢复并释放 pin（协调器 forget_pending ← BlockManager.deallocate）
    def begin_restore(
        self, prefix_hash: int, dst_slot: int
    ) -> CheckpointRestoreOp | None:
        checkpoint_id = self.lookup(prefix_hash)
        if checkpoint_id < 0:
            return None
        record = self.records[checkpoint_id]
        record.pin_count += 1
        self._lru.move_to_end(checkpoint_id)
        op = CheckpointRestoreOp(
            dst_slot=dst_slot,
            unit_ids=record.unit_ids,
            total_bytes=self.spec.image_bytes,
            layout_id=self.spec.layout_id,
        )
        self._queued_restores.append((checkpoint_id, op))
        return op

    def take_restore_ops(self) -> tuple[CheckpointRestoreOp, ...]:
        queued, self._queued_restores = self._queued_restores, []
        self._inflight_restores.extend(checkpoint_id for checkpoint_id, _ in queued)
        return tuple(op for _, op in queued)

    def cancel_queued_restore(self, dst_slot: int) -> None:
        kept: list[tuple[int, CheckpointRestoreOp]] = []
        for checkpoint_id, op in self._queued_restores:
            if op.dst_slot == dst_slot:
                self._release_restore_pin(checkpoint_id)
            else:
                kept.append((checkpoint_id, op))
        self._queued_restores = kept

    def suspend_queued_restore(
        self, dst_slot: int
    ) -> SuspendedCheckpointRestore | None:
        """Remove one queued restore without releasing its checkpoint pin.

        An external full-snapshot restore uses this to prevent an older local
        checkpoint copy from overwriting the destination SLOT later.  The
        caller must eventually either resume or release the returned lease.
        """
        # 【译】移除一个排队中的恢复，但不释放它对 checkpoint 的 pin。
        #   外部的整份快照恢复用它来防止一份更老的本地 checkpoint 拷贝稍后覆盖目标槽位。调用方最终必须恢复（resume）或释放返回的这份租约。
        found: SuspendedCheckpointRestore | None = None
        kept: list[tuple[int, CheckpointRestoreOp]] = []
        for checkpoint_id, op in self._queued_restores:
            if op.dst_slot == dst_slot:
                if found is not None:
                    raise AssertionError(
                        f"multiple queued checkpoint restores target slot {dst_slot}"
                    )
                found = SuspendedCheckpointRestore(checkpoint_id, op)
            else:
                kept.append((checkpoint_id, op))
        self._queued_restores = kept
        return found

    def resume_suspended_restore(self, restore: SuspendedCheckpointRestore) -> None:
        """Put a suspended restore back on the next maintenance batch."""
        # 【译】把一个被挂起的恢复放回下一个维护批次。
        record = self.records.get(restore.checkpoint_id)
        if record is None or record.pin_count <= 0:
            raise AssertionError("suspended checkpoint restore lost its source pin")
        self._queued_restores.append((restore.checkpoint_id, restore.operation))

    def release_suspended_restore(self, restore: SuspendedCheckpointRestore) -> None:
        """Release a suspended restore whose destination was filled elsewhere."""
        # 【译】释放一个被挂起的恢复——它的目标已经由别处填好了。
        self._release_restore_pin(restore.checkpoint_id)

    def _note_change(self, prefix_hash: int | None) -> None:
        self.generation += 1
        self._changes.append((self.generation, prefix_hash))

    def changed_since(self, generation: int) -> set[int] | None:
        """Hashes whose READY state may have changed after ``generation``.

        None when that cannot be told: the log no longer reaches back that
        far, or a `clear` touched every hash.
        """
        # 【译】在 generation 之后 READY 状态可能变了的 hash。
        #   无法判断时返回 None：日志已经回溯不到那么远，或者一次 clear 动了所有 hash。
        if generation >= self.generation:
            return set()
        if not self._changes or self._changes[0][0] > generation + 1:
            return None
        changed: set[int] = set()
        for bumped_at, prefix_hash in reversed(self._changes):
            if bumped_at <= generation:
                break
            if prefix_hash is None:
                return None
            changed.add(prefix_hash)
        return changed

    def adopt_units(
        self, unit_ids: tuple[int, ...], owner: Hashable, prefix_hash: int
    ) -> bool:
        """Publish reserved units as a READY checkpoint without a copy.

        The units change owner in place, so no other allocation can observe
        them free in between. When `prefix_hash` is already READY or pending,
        the units are released instead and False is returned. The adopted
        checkpoint is not nominated for offload: it just arrived from that tier.
        """
        # 【译】不经拷贝，把已预留的单元发布为一个 READY checkpoint。
        #   单元原地换主人，所以中间不会有别的分配看到它们是空闲的。prefix_hash 已经是 READY 或在途时，改为释放这些单元并返回 False。
        #   被收养的 checkpoint 不会被提名去 offload：它就是刚从那一层来的。
        # 【分支】调用方：协调器 adopt_transfer_units ← native_state_scheduler.py（外部加载的镜像直接变成本地快照），见第9批
        if self.contains_or_pending(prefix_hash):
            self.pool.release_units(unit_ids, owner)
            return False
        checkpoint_id = self._new_identity()
        self.pool.rekey_units(unit_ids, owner, ("state-checkpoint", checkpoint_id))
        record = CheckpointRecord(
            prefix_hash=int(prefix_hash),
            unit_ids=tuple(unit_ids),
            state=READY,
        )
        self.records[checkpoint_id] = record
        self.hash_to_checkpoint[record.prefix_hash] = checkpoint_id
        self._lru[checkpoint_id] = None
        self._note_change(record.prefix_hash)
        return True

    # complete_inflight：上一批的拷贝都已随 batch 发出——
    #   存：COPYING → READY（进索引和 LRU，并提名给 CPU 层）；期间被 unindex 的（EVICTING）直接释放；同 hash 已有 READY 的则丢掉这份
    #   取：释放 restore pin
    # 调用方：协调器 complete_previous_batch ← BlockManager.complete_previous_state_batch（Scheduler.schedule 每轮开头）
    def complete_inflight(self) -> None:
        stores, self._inflight_stores = self._inflight_stores, []
        for checkpoint_id in stores:
            record = self.records.get(checkpoint_id)
            if record is None:
                continue
            if self._pending_by_hash.get(record.prefix_hash) == checkpoint_id:
                del self._pending_by_hash[record.prefix_hash]
            if record.state == EVICTING:
                self._release_record(checkpoint_id)
                continue
            if record.state != COPYING:
                continue
            # Publish only after the scatter has ridden a batch.
            # 【译】只在 scatter 已经随一个 batch 跑过之后才发布。
            if self.lookup(record.prefix_hash) >= 0:
                self._release_record(checkpoint_id)
                continue
            record.state = READY
            self.hash_to_checkpoint[record.prefix_hash] = checkpoint_id
            self._note_change(record.prefix_hash)
            self._lru[checkpoint_id] = None
            self._queue_offload_store(checkpoint_id, record)

        restores, self._inflight_restores = self._inflight_restores, []
        for checkpoint_id in restores:
            self._release_restore_pin(checkpoint_id)

    # ------------------------------- offload ------------------------------- #
    # 【译】offload
    # 【分支】下面到 was_reclaimed 是 CPU 层（state_offload.py）的存储一侧，见第9批 9c。流程：
    #   READY 时提名（_queue_offload_store，不钉住）→ take_offload_stores 在 max_inflight 以内取出并钉住（pin_count+1）
    #   → worker 做 D2H gather → release_offload_store_source（第一阶段：单元还给 KV 池，但操作仍算"在途"）
    #   → 报告到达 settle_offload_store（第二阶段：退役操作；成功与否由 BlockManager.settle_state_store 决定是否 note_stored）
    #   报告丢失时由 reclaim_stale_offload_pins 超时兜底
    def _queue_offload_store(self, checkpoint_id: int, record) -> None:
        """Nominate this checkpoint for the CPU tier. Takes no pin.

        READY and nowhere else: `begin_store` is too early (the scatter has not
        ridden a batch, so the units hold no image yet) and `_evict`/`unindex`
        are too late (they fire when the pool wants those units *now*). READY is
        when the bytes first exist and when the record is least wanted -- it has
        just entered the LRU at the cold end.

        Nomination, not reservation: a queued candidate is unpinned, so
        `_next_victim` may still spend it. The pool never waits on the CPU tier.
        Dedup is free -- `begin_store` refuses a hash it already holds.
        """
        # 【译】把这个 checkpoint 提名给 CPU 层。不取 pin。
        #   只在 READY 时、而不在别处：begin_store 太早（scatter 还没随 batch 跑过，单元里还没有镜像），_evict / unindex 太晚
        #   （它们在池子"现在"就要这些单元时触发）。READY 是字节第一次存在的时刻，也是这份记录最不被需要的时刻——它刚刚从冷的一端进入 LRU。
        #
        #   是提名而不是预留：排队的候选没有被钉住，所以 _next_victim 仍然可以花掉它。池子从不等待 CPU 层。
        #   去重是免费的——begin_store 会拒绝它已经持有的 hash。
        if not self._offload_sink:
            return
        self._offload_ready.append(checkpoint_id)
        # Bound the backlog (see `_offload_backlog_cap`). popleft-on-overflow
        # rather than a `deque(maxlen=)`: `take_offload_stores` re-queues
        # deferred nominations with `extendleft`, and a maxlen deque would evict
        # from whichever end an append pushed against -- dropping the freshly
        # re-queued oldest entries we take pains to preserve there.
        # 【译】给积压设上限（见 _offload_backlog_cap）。溢出时 popleft，而不是用 deque(maxlen=)：take_offload_stores 用 extendleft
        #   把推迟的提名重新排回去，而一个 maxlen 的 deque 会从 append 推挤的那一端淘汰——丢掉的正是我们在那里费心保留的、
        #   刚重新排回来的最老条目。
        if len(self._offload_ready) > self._offload_backlog_cap:
            if self.offload_nominations_dropped == 0:
                logger.warning(
                    "state offload nomination backlog exceeded %d; dropping "
                    "oldest nominations. The CPU tier is draining checkpoints "
                    "slower than they reach READY -- offload is falling behind.",
                    self._offload_backlog_cap,
                )
            while len(self._offload_ready) > self._offload_backlog_cap:
                self._offload_ready.popleft()
                self.offload_nominations_dropped += 1

    @property
    def store_backlog(self) -> deque:
        """Nominations waiting for an in-flight slot. A gauge for the log."""
        # 【译】等待一个在途名额的提名。日志用的一个仪表值。
        return self._offload_ready

    def take_offload_stores(
        self, max_inflight: int
    ) -> list[tuple[StateStoreOperationId, tuple[int, ...]]]:
        """`(operation, unit_ids)` to hand the tier now, pinning each.

        The pin is taken HERE, not at READY, and that is what bounds it: a pin
        lives in this process while the D2H runs in the worker, so it spans
        several scheduler passes. Pinning at READY would make every checkpoint
        un-evictable for that window and break the admission invariant that a
        READY unpinned checkpoint counts as available to live KV.

        Over the cap, candidates wait rather than being dropped. Drained, not
        read: a second submission would store the same image twice and the
        second report would unpin a record the first released.

        Each hand-out gets its own generation, so a prefix stored again after
        an eviction is a different operation from the one that stored it
        before. Two attempts at the same hash may not be in flight at once --
        `_hash_in_flight` refuses that -- but they may follow one another
        closely enough that the earlier one's report is still on the wire.
        """
        # 【译】现在要交给这一层的 (operation, unit_ids)，每个都钉住。
        #   pin 在这里取，而不是在 READY 时取，这正是给它划界的方式：pin 住在本进程里，而 D2H 在 worker 里跑，所以它横跨好几轮调度。
        #   在 READY 时就钉住，会让每个 checkpoint 在那个窗口里都不可淘汰，破坏"READY 且未钉住的 checkpoint 对活的 KV 而言算可用"
        #   这一准入不变量。
        #
        #   超过上限的候选等待而不是被丢弃。是排空而不是读取：第二次提交会把同一份镜像存两次，第二份报告会去释放一份第一份已经释放过的记录。
        #
        #   每次交出都有它自己的 generation，所以一个在淘汰之后被再次存储的前缀，和之前存它的那次是不同的操作。同一个 hash 的两次尝试
        #   不能同时在途——_hash_in_flight 会拒绝——但它们可能紧挨着发生，以至于较早那次的报告还在路上。
        # 调用方：协调器 take_offload_stores ← BlockManager.take_state_stores ← Scheduler（max_inflight = _state_store_pending_cap，
        #   即 connector 的 max_pending_saves，由 OFFLOAD_MAX_PENDING_SAVES / OFFLOAD_COPY_WORKERS 算出）
        out: list[tuple[StateStoreOperationId, tuple[int, ...]]] = []
        # Nominations whose hash is transiently in flight: held aside and
        # re-queued after the pass, NOT dropped (see below).
        # 【译】hash 暂时在途的提名：先放一边，这一轮结束后重新排队，而不是丢掉（见下文）。
        deferred: list = []
        while self._offload_ready and len(self._offload_pins) < max_inflight:
            checkpoint_id = self._offload_ready.popleft()
            record = self.records.get(checkpoint_id)
            # Spent while it waited -- which nomination deliberately allows.
            # Terminal: there is nothing left to store, so drop it.
            # 【译】等待期间被花掉了——提名本来就允许这样。终结性的：已经没有东西可存，所以丢掉它。
            if record is None or record.state != READY:
                continue
            if self._hash_in_flight(record.prefix_hash):
                # NOT terminal. The record is still READY and is a live, valid
                # nomination; its only problem is that an earlier generation of
                # the same hash is still on the wire (an eviction + re-store of
                # one prefix while the first store is mid-flight). Dropping it
                # here -- as a bare `continue` did -- lost this checkpoint image
                # from the store queue for good once the in-flight copy settled.
                # Defer it to a later pass instead. Held aside rather than
                # re-appended inside the loop so a hash that stays in flight
                # cannot spin this call.
                # 【译】不是终结性的。这份记录仍是 READY，是一个活的、有效的提名；它唯一的问题是同一 hash 的一个更早的 generation 还在路上
                #   （同一前缀的一次淘汰 + 再存发生在第一次存储的中途）。在这里丢掉它——像以前一个裸 continue 那样——会让这份 checkpoint 镜像
                #   在在途那份了结之后永远从存储队列里消失。改为推迟到之后的某一轮。先放一边而不是在循环里重新 append，
                #   这样一个一直在途的 hash 不会让这次调用空转。
                deferred.append(checkpoint_id)
                continue
            out.append(
                self._pin_offload_source(checkpoint_id, timeout_reclaimable=True)
            )
        # Re-queue the deferred nominations for the next drain, once the
        # colliding in-flight generation has had a chance to settle. The drain
        # is `popleft()` (oldest at the left), so `extend` would append these
        # older, already-waiting nominations *behind* newer ones and let them
        # starve until they age out at the `state != READY` check. Put them
        # back at the front, preserving their original oldest-first order.
        # 【译】把推迟的提名重新排回去，等下一次排空，那时冲突的在途 generation 已经有机会了结。排空用的是 popleft()（最老的在左边），
        #   所以 extend 会把这些更老、早已在等的提名追加到更新的提名"后面"，让它们一直饿着，直到在 state != READY 检查处老化掉。
        #   把它们放回前面，保持原来最老在前的顺序。
        # extendleft(reversed(x))：extendleft 会逐个插到左端（顺序会反过来），先 reversed 一次正好保持原顺序
        if deferred:
            self._offload_ready.extendleft(reversed(deferred))
        return out

    def acquire_checkpoint_source(
        self, prefix_hash: int, *, max_inflight: int | None = None
    ) -> tuple[StateStoreOperationId, tuple[int, ...]] | None:
        """Lease the existing READY image for an admitted external transfer.

        The caller must finish scheduler admission before acquiring this
        source. A lease pins the image's
        existing PAGE units; it neither copies an Active Slot nor queues a
        checkpoint. Missing, COPYING, or already dispatched hashes return
        None, as does a full optional operation limit. The limit includes
        operations whose source was released but whose result is outstanding.

        Release with `release_offload_store_source` once every reader has
        stopped, then `settle_offload_store` on terminal completion. Settlement
        can also release a source for a transfer known to have stopped or never
        started. These leases are never reclaimed by elapsed time: a missing
        report cannot prove that another process has stopped its DMA.
        """
        # 【译】为一次已准入的外部传输租用现有的 READY 镜像。
        #   调用方必须先完成调度器准入，再获取这个源。租约钉住镜像已有的 PAGE 单元；它既不拷贝活跃槽位，也不排队 checkpoint。
        #   缺失、COPYING 或已经派发出去的 hash 返回 None；可选的操作上限满了也返回 None。这个上限包括那些源已经释放、
        #   但结果仍未到达的操作。
        #
        #   在每个读者都停下之后用 release_offload_store_source 释放，然后在终结性完成时调 settle_offload_store。
        #   了结也可以为一次已知已经停止或从未开始的传输释放源。这些租约永远不会按经过的时间被回收：
        #   报告缺失不能证明另一个进程已经停止了它的 DMA。
        # 【分支】调用方：协调器 acquire_checkpoint_source ← native_state_scheduler.py（LMCache MP），见第9批
        if max_inflight is not None and len(self._offload_pins) >= max_inflight:
            return None
        checkpoint_id = self.lookup(prefix_hash)
        if checkpoint_id < 0 or self._hash_in_flight(prefix_hash):
            return None
        return self._pin_offload_source(checkpoint_id, timeout_reclaimable=False)

    # _pin_offload_source：给这次交出分配新的 generation，pin_count+1，记一个 _OffloadPin（timeout_reclaimable 区分旧提名路径与显式租约）
    def _pin_offload_source(
        self, checkpoint_id: int, *, timeout_reclaimable: bool
    ) -> tuple[StateStoreOperationId, tuple[int, ...]]:
        record = self.records[checkpoint_id]
        self._offload_generation += 1
        op = StateStoreOperationId(int(record.prefix_hash), self._offload_generation)
        record.pin_count += 1
        self._offload_pins[op] = _OffloadPin(
            checkpoint_id, monotonic(), timeout_reclaimable=timeout_reclaimable
        )
        return op, record.unit_ids

    def _hash_in_flight(self, prefix_hash: int) -> bool:
        """Whether some generation of `prefix_hash` is already pinned.

        Kept from when the pins were keyed by hash: two live stores of one
        image would copy the same bytes twice and the second report would unpin
        a record the first already released. The generation exists to tell
        *sequential* attempts apart, not to license concurrent ones.
        """
        # 【译】prefix_hash 的某个 generation 是否已经被钉住。
        #   从 pin 还以 hash 为键时保留下来：同一份镜像的两次活的存储会把同样的字节拷两次，第二份报告会去释放一份第一份已经释放的记录。
        #   generation 的存在是为了区分"先后"的尝试，而不是为了允许"并发"的尝试。
        return any(op.prefix_hash == prefix_hash for op in self._offload_pins)

    def release_offload_store_source(self, op: StateStoreOperationId) -> None:
        """Phase one: the gather drained, so hand the PAGE units back now.

        This is the earlier of a store's two ends. It returns the units to the
        KV pool the instant the D2H copy is done reading them -- holding an
        image out of the pool across the subsequent CPU `batched_put` would cost
        reuse for a step that cannot touch the units. But it does NOT remove the
        operation from `_offload_pins`: the store is still dispatched-but-
        unreported, so `has_offload_pins` and `_hash_in_flight` must keep
        seeing it until the index report lands. It only marks the source gone,
        so the report (or a reclaim) does not release the same units twice.

        Idempotent, and matched on the whole operation: a release for a
        generation no longer here -- already reported, or reclaimed -- does
        nothing.
        """
        # 【译】第一阶段：gather 已经排空，所以现在就把 PAGE 单元还回去。
        #   这是一次存储两个终点中较早的那个。D2H 拷贝一读完单元，就把它们还给 KV 池——在随后的 CPU batched_put 期间把一份镜像
        #   扣在池外，只会为一个碰不到这些单元的步骤损失复用。但它"不"把这个操作从 _offload_pins 里删掉：这次存储仍然是
        #   已派发、未报告的，所以 has_offload_pins 和 _hash_in_flight 必须一直看得到它，直到索引报告到达。它只标记源已经走了，
        #   这样报告（或一次回收）不会把同样的单元释放两次。
        #
        #   幂等，并按整个操作匹配：对一个已经不在这里的 generation 的释放——已报告或已回收——什么都不做。
        # 调用方：协调器 ← BlockManager.release_state_store_source；native_state_scheduler.py
        pin = self._offload_pins.get(op)
        if pin is None or pin.source_released:
            return
        pin.source_released = True
        self._release_offload_pin(pin.checkpoint_id)

    def settle_offload_store(self, op: StateStoreOperationId) -> None:
        """Phase two: the store reported, either way. Retire the operation.

        Removes the pin so `has_offload_pins`/`_hash_in_flight` stop seeing this
        operation -- the report is the true end of "dispatched but unreported".
        The units are normally already back (`release_offload_store_source` ran
        first); this is the backstop for a store that failed or reported before
        its source was released, so the units are freed here only when they were
        not freed there. Whether the hash becomes reachable is
        `StateOffloadIndex`'s business, not this one's.

        Matched on the whole operation: a report from an attempt whose pin was
        already reclaimed names a generation no longer here and settles
        nothing, rather than releasing a newer attempt's pin.
        """
        # 【译】第二阶段：这次存储报告了，不管结果如何。让这个操作退役。
        #   删掉 pin，让 has_offload_pins / _hash_in_flight 不再看到这个操作——报告才是"已派发、未报告"的真正终点。
        #   单元通常已经还回去了（release_offload_store_source 先跑过）；这是给失败的、或在源释放之前就报告了的存储的兜底，
        #   所以只有在那边没释放时才在这里释放单元。这个 hash 是否变得可达，是 StateOffloadIndex 的事，不归这里管。
        #
        #   按整个操作匹配：一份来自"pin 已经被回收"的尝试的报告，指的是一个已不在这里的 generation，什么都不了结，
        #   而不会去释放一次更新尝试的 pin。
        # 调用方：协调器 ← BlockManager.settle_state_store；native_state_scheduler.py
        pin = self._offload_pins.pop(op, None)
        if pin is None:
            return
        if not pin.source_released:
            self._release_offload_pin(pin.checkpoint_id)

    def has_offload_pins(self) -> bool:
        """Whether any dispatched store is still awaiting its index report.

        A pin lives here from the moment `take_offload_stores` hands a store to
        the worker until `settle_offload_store` retires it on the index report,
        or the last-resort `reclaim_stale_offload_pins` times it out. The source
        release in between hands the PAGE units back but leaves the pin, so this
        stays true across it: it is therefore true exactly while a store is
        dispatched-but-unreported, which is the interval the engine must keep
        polling across -- the report that clears the pin, and the reclaim that
        is its only other exit, both run only from `_poll_kv_transfer_progress`,
        which the idle loop skips once liveness reads False. Because reclaim
        shares this same dict, the signal cannot latch the busy loop: a lost
        report clears here when the reclaim fires.
        """
        # 【译】是否还有任何已派发的存储在等它的索引报告。
        #   一个 pin 从 take_offload_stores 把存储交给 worker 的那一刻起住在这里，直到 settle_offload_store 在索引报告时让它退役，
        #   或者最后手段 reclaim_stale_offload_pins 让它超时。中间的源释放把 PAGE 单元还回去但留着 pin，所以这个函数跨过它仍为真：
        #   它恰好在"一次存储已派发、未报告"期间为真，这正是引擎必须持续轮询的区间——清掉 pin 的报告，以及作为它唯一另一个出口的回收，
        #   都只在 _poll_kv_transfer_progress 里运行，而空闲循环一旦读到活性为 False 就跳过它。因为回收共用同一个 dict，
        #   这个信号不会把忙循环锁死：一份丢失的报告会在回收触发时在这里清掉。
        # 调用方：协调器 ← BlockManager.has_pending_state_store_pins（引擎据此决定空闲时是否还要继续轮询 KV 传输）
        return bool(self._offload_pins)

    def reclaim_stale_offload_pins(self, timeout_s: float) -> int:
        """Release pins whose report never came, after `timeout_s`.

        The pin lives here in the engine process and the D2H runs in the
        worker, so a lost report -- a crashed worker, a dropped completion --
        would hold a whole image out of the pool forever. This is the unit-side
        twin of `Scheduler._reconcile_stalled_deferred_saves`.

        **This is a last resort and it does not prove the reader stopped.** It
        used to be justified by LMCache's own pin timeout, on the grounds that
        by then upstream has force-unpinned and stopped reading. That argument
        does not hold for this source: a K3 state store bypasses
        `CacheEngine.store()` entirely and gathers ATOM PAGE units through
        `page_unit_views -> StagedTransfer.pack -> storage_manager.batched_put`,
        and those units are not covered by LMCache's GPU-source pin monitor.
        Nothing here can tell a lost report from a worker still inside the
        gather.

        What follows from that is not that the units may never be reclaimed --
        that would leak an image per dropped completion -- but that a store
        whose source was reclaimed can no longer be trusted: if the reader had
        not stopped, the pool may have handed those units to another request
        whose writes the gather then picked up, and the CPU image is a mix of
        two prefixes under the first one's hash. Resuming onto it is silent
        wrong output. So the operation is remembered here and
        `BlockManager.settle_state_store` refuses to index it. The reclaim
        recovers the memory and forfeits the entry, which is the only pair of
        outcomes this can honestly offer.

        Explicit `acquire_checkpoint_source` leases are excluded. Their
        connector must confirm source release or terminal completion; the
        legacy nomination path's timeout policy does not apply to them.
        """
        # 【译】释放那些报告一直没来的 pin，在 timeout_s 之后。
        #   pin 住在引擎进程里，而 D2H 在 worker 里跑，所以一份丢失的报告——worker 崩溃、完成通知被丢——会把一整份镜像永远扣在池外。
        #   这是 Scheduler._reconcile_stalled_deferred_saves 在单元一侧的双胞胎。
        #
        #   **这是最后手段，而且它不能证明读者已经停了。** 它以前的理由是 LMCache 自己的 pin 超时：到那时上游已经强制解钉、停止读取。
        #   这个论证对这个源不成立：K3 的状态存储完全绕过 CacheEngine.store()，经
        #   page_unit_views -> StagedTransfer.pack -> storage_manager.batched_put 去 gather ATOM 的 PAGE 单元，
        #   而这些单元不在 LMCache 的 GPU 源 pin 监视范围内。这里没有任何东西能区分"报告丢了"和"worker 还在 gather 里"。
        #
        #   由此推出的不是"单元永远不能回收"——那样每丢一次完成通知就泄漏一份镜像——而是"源被回收过的存储再也不可信"：
        #   如果读者其实没停，池子可能已经把这些单元交给了另一个请求，gather 随后又读进了它写的东西，于是 CPU 上的镜像是两个前缀的混合，
        #   却挂在第一个前缀的 hash 名下。从它恢复就是静默的错误输出。所以这个操作被记在这里，BlockManager.settle_state_store
        #   拒绝登记它。回收收回了内存、放弃了这个条目，这是它能诚实给出的唯一一对结果。
        #
        #   显式的 acquire_checkpoint_source 租约被排除在外。它们的 connector 必须确认源释放或终结性完成；旧的提名路径的超时策略不适用于它们。
        # 调用方：协调器 ← BlockManager.reclaim_stale_state_store_pins
        if timeout_s <= 0 or not self._offload_pins:
            return 0
        cutoff = monotonic() - timeout_s
        stale = [
            op
            for op, pin in self._offload_pins.items()
            if pin.timeout_reclaimable and pin.pinned_at <= cutoff
        ]
        for op in stale:
            pin = self._offload_pins.pop(op)
            self.offload_pins_reclaimed += 1
            if pin.source_released:
                # The gather already drained and the units are back; only the
                # index report was lost. Nothing to release again -- doing so
                # would underflow the record -- and nothing to forfeit: the
                # bytes are trustworthy because the reader provably stopped, so
                # a late report may still index the hash. This is a lost report,
                # not a taken-back source.
                # 【译】gather 已经排空、单元已经还回；丢的只是索引报告。没有什么要再释放的——再释放会让记录的计数下溢——也没有什么要放弃的：
                #   读者可以证明已经停了，所以字节可信，一份迟到的报告仍然可以登记这个 hash。这是一份丢失的报告，而不是一个被收回的源。
                logger.warning(
                    "state offload: store %s never reported after %.1fs; its "
                    "units were already returned at source release, so only "
                    "the index report is lost.",
                    op,
                    timeout_s,
                )
                continue
            # The source was never released, so the worker may still be inside
            # the gather. Taking the units back now means they cannot be
            # trusted: the pool may hand them to another request whose writes
            # this gather then picks up. Release and forfeit the entry.
            # 【译】源从没被释放，所以 worker 可能还在 gather 里。现在把单元收回意味着它们不可信：池子可能把它们交给另一个请求，
            #   而这次 gather 随后会读进那个请求写的东西。释放并放弃这个条目。
            self._release_offload_pin(pin.checkpoint_id)
            self._offload_reclaimed[op] = None
            while len(self._offload_reclaimed) > _RECLAIMED_MEMORY:
                self._offload_reclaimed.popitem(last=False)
            logger.warning(
                "state offload: reclaimed the units of store %s after %.1fs "
                "with no report. The worker may still be reading them, so the "
                "image will not be indexed even if a report arrives later.",
                op,
                timeout_s,
            )
        return len(stale)

    def was_reclaimed(self, op) -> bool:
        """Whether this store's source was taken back before it reported."""
        # 【译】这次存储的源是否在它报告之前就被收回了。调用方：协调器 ← BlockManager.settle_state_store
        return op in self._offload_reclaimed

    def _release_offload_pin(self, checkpoint_id: int) -> None:
        record = self.records.get(checkpoint_id)
        if record is None:
            return
        if record.pin_count <= 0:
            raise AssertionError("checkpoint offload pin underflow")
        record.pin_count -= 1
        if record.state == EVICTING and record.pin_count == 0:
            # `unindex` fired during the copy: the boundary's KV block was
            # spent, so this image is unreachable in HBM from now on. The copy
            # was still worth finishing -- the CPU copy outliving the HBM one
            # is the entire point of the tier -- and only now do the units go
            # back.
            # 【译】拷贝期间 unindex 触发了：这个边界的 KV block 被花掉了，所以这份镜像从现在起在 HBM 里不可达。
            #   拷贝仍然值得做完——CPU 拷贝比 HBM 那份活得更久正是这一层的全部意义——只有现在单元才还回去。
            self._release_record(checkpoint_id)

    # _release_offload_pin / _release_restore_pin：pin_count-1；如果记录已是 EVICTING 且没人再钉着，就真正释放页单元
    def _release_restore_pin(self, checkpoint_id: int) -> None:
        record = self.records.get(checkpoint_id)
        if record is None:
            return
        if record.pin_count <= 0:
            raise AssertionError("checkpoint restore pin underflow")
        record.pin_count -= 1
        if record.state == EVICTING and record.pin_count == 0:
            self._release_record(checkpoint_id)

    # unindex：边界 KV block 被淘汰时作废同 hash 的快照（READY 或 COPYING 的都算）——标为 EVICTING、移出 LRU；
    #   没有在途写入、也没人钉着就立刻释放页单元，否则等它们结束（_release_*_pin / complete_inflight 里释放）
    # 调用方：协调器 unindex ← BlockManager._record_evicted
    def unindex(self, prefix_hash: int) -> bool:
        self._note_change(prefix_hash)
        checkpoint_id = self.hash_to_checkpoint.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            checkpoint_id = self._pending_by_hash.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            return False
        record = self.records.get(checkpoint_id)
        if record is None:
            return False
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        # Keep units alive while a queued GPU writer can still access them.
        # 【译】只要还有排队的 GPU 写者可能访问这些单元，就让它们活着。
        if checkpoint_id not in self._inflight_stores and record.pin_count == 0:
            self._release_record(checkpoint_id)
        return True

    # clear：清空前缀缓存时清空索引，所有记录标为 EVICTING，能立刻释放的立刻释放（调用方：协调器 clear_index ← BlockManager.clear_cache，
    #   后者在 v0.1.8 的 atom/ 里只有测试调用）
    def clear(self) -> None:
        self._note_change(None)
        self.hash_to_checkpoint.clear()
        self._pending_by_hash.clear()
        self._lru.clear()
        inflight_stores = set(self._inflight_stores)
        for checkpoint_id in list(self.records):
            record = self.records[checkpoint_id]
            record.state = EVICTING
            if checkpoint_id not in inflight_stores and record.pin_count == 0:
                self._release_record(checkpoint_id)

    # _evict：LRU 淘汰一份 READY 且未钉住的快照（只由 ensure_free_units 调用），页单元立即还回 KV 池，evictions += 1
    # _release_record：删记录、清所有索引项、按归属标记把页单元还给 BlockPool
    def _evict(self, checkpoint_id: int) -> None:
        record = self.records[checkpoint_id]
        if record.state != READY or record.pin_count:
            raise AssertionError("only an unpinned READY checkpoint is evictable")
        self._note_change(record.prefix_hash)
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        self._release_record(checkpoint_id)
        self.evictions += 1

    def _release_record(self, checkpoint_id: int) -> None:
        record = self.records.pop(checkpoint_id)
        self._note_change(record.prefix_hash)
        self._lru.pop(checkpoint_id, None)
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        if self._pending_by_hash.get(record.prefix_hash) == checkpoint_id:
            del self._pending_by_hash[record.prefix_hash]
        self.pool.release_units(record.unit_ids, ("state-checkpoint", checkpoint_id))


class PagedStateCheckpointCoordinator:
    """Schedules PAGE-backed checkpoints for per-request state."""

    # 【译】为每请求状态安排 PAGE 承载的 checkpoint。
    # 它实现 state_cache.py 的 StateCheckpointCache 协议，copy 方式下是 BlockManager.state_caches 的唯一成员
    #   （BlockManager.__init__：PagedStateCheckpointCoordinator(self.kv, checkpoint_spec, enabled=..., readable_midstep=...)，
    #    之后 attach_midstep_scratch(self.state, ...)；有 CPU 层时再 attach_offload(state_offload, sink=can_store_state)）
    # 每轮被调用的顺序（Scheduler.schedule 一轮）：
    #   开头   complete_previous_batch（上一批的存 → READY、取 → 放 pin、midstep scratch 槽位还回）
    #   准入   resumable_hit（门控）、begin_restore（命中时排恢复）、has_available_units（KV 与快照共用池的准入门）
    #   构造   take_checkpoint_ops（把待存的边界变成 CheckpointStoreOp，连同排队的恢复一起交给本批）
    #   之后   postprocess 里 checkpoint（记意图）、publish_midstep（midstep 时）；被淘汰的 KV 触发 unindex
    # 和 StateSlotPool 的分工：本类管"快照"（住在 KV 池的页单元里）；活跃槽位、以及 midstep 用的临时 scratch 槽位仍由 StateSlotPool 分配

    # copy 方式交接时拥有者不动，checkpoint 之后的 forward 不欠任何东西，所以 successor_room = 0（任何位置都能保存，
    #   见 state_cache.py successor_room；对比 fork(1) 的 1）
    successor_room = 0.0

    # ------------------------- midstep checkpointing ------------------------ #
    #
    # A PAGE image is written by a copy the runner issues before a later batch,
    # out of a slot. Unreadable by default: the image then comes from the slot
    # a forward *ended* in, so the engine cuts prefill chunks onto rungs.
    #
    # A backend whose kernel exposes interior states (GDN's `h`) can declare
    # `readable_midstep`, and then a position inside a forward reaches the
    # image through a scratch slot borrowed from the state pool:
    #
    #   `reserve_midstep`  before the forward, borrow a scratch slot per
    #                      position. The runner copies the interior state in.
    #   `publish_midstep`  after it, queue `(hash, scratch)` for storing.
    #   `take_checkpoint_ops`
    #                      while building the next batch, `begin_store` each
    #                      queued scratch, exactly as `checkpoint`'s pending
    #                      boundaries are begun. Not at publish: `begin_store`
    #                      enters the store's inflight list, which the *next*
    #                      `complete_previous_batch` turns READY, so begun at
    #                      publish the image would be READY -- and restorable
    #                      -- one batch before its copy ran.
    #   `complete_previous_batch`
    #                      the batch carrying those copies has been issued, so
    #                      the scratch goes back to the state pool. A later
    #                      writer of the slot runs after it on the same stream.
    #
    # A scratch slot is never a request's runtime slot, so nothing the owner
    # does next can overwrite it before its store is issued.
    # 【译】midstep checkpoint
    #
    #   一份 PAGE 镜像由 runner 在之后某个 batch 之前发起的一次拷贝写成，源是一个槽位。默认不可读：镜像来自某次 forward
    #   "结束"时所在的槽位，所以引擎把 prefill chunk 切在阶梯上。
    #
    #   kernel 暴露内部状态的后端（GDN 的 h）可以声明 readable_midstep，于是 forward 内部的某个位置通过一个从状态池借来的
    #   scratch 槽位到达镜像：
    #     reserve_midstep          forward 之前，为每个位置借一个 scratch 槽位。runner 把内部状态拷进去。
    #     publish_midstep          forward 之后，把 (hash, scratch) 排队等待存储。
    #     take_checkpoint_ops      构造下一个 batch 时，对每个排队的 scratch 调 begin_store，与 checkpoint 的待存边界被开始的方式完全一样。
    #                              不在 publish 时做：begin_store 会进入存储的在途列表，由"下一次" complete_previous_batch 变成 READY，
    #                              所以若在 publish 时开始，镜像会在它的拷贝运行之前一个 batch 就 READY——并且可被恢复。
    #     complete_previous_batch  携带这些拷贝的 batch 已经发出，所以 scratch 回到状态池。这个槽位之后的写者会在同一个流上排在它后面。
    #
    #   scratch 槽位永远不是某个请求的运行时槽位，所以拥有者接下来做的任何事都不可能在它的存储发出之前覆盖它。
    # 【分支】只有 Qwen GDN 在 ATOM_GDN_STATE_MIDSTEP=1 时声明 readable_midstep=True（gdn_attn.py state_transfer）；默认关闭，
    #   这时下面 reserve / publish / cancel_midstep 都是空操作，prefill 照常被 checkpoint_cut 切在阶梯上
    # 时间线（midstep，一格 = 一轮）：第 N 轮 plan_midstep → reserve_midstep 借 scratch → 第 N 批 forward 中 runner 把 h 切片拷进 scratch
    #   → 第 N 轮 postprocess commit_midstep → publish_midstep 排队 → 第 N+1 轮构造 batch 时 _take_scratch_stores → begin_store（COPYING）
    #   → 第 N+2 轮开头 complete_previous_batch：READY，scratch 还回状态池

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
        enabled: bool,
        offload=None,
        readable_midstep: bool = False,
    ) -> None:
        # pool：KV 的 BlockPool（页单元从这里借）；spec：页单元几何；enabled = 开了前缀缓存且状态池有槽位；
        # offload：CPU 层索引（构造时总是 None，由 attach_offload 接上）；readable_midstep：后端是否能 midstep 拍快照
        self.enabled = enabled
        self.readable_midstep = bool(readable_midstep)
        # The state pool scratch slots are borrowed from, and how many stores
        # one batch's copy descriptor has room for. Set by
        # `attach_midstep_scratch`; a readable coordinator without them
        # reserves nothing.
        # 【译】借 scratch 槽位的那个状态池，以及一个 batch 的拷贝描述符能装多少次存储。由 attach_midstep_scratch 设置；
        #   没有它们的可读协调器什么都不预留。
        self._scratch = None
        self._max_ops_per_batch = 0
        # Published `(hash, scratch slot)` not yet begun, oldest first.
        # 【译】已发布、尚未开始的 (hash, scratch 槽位)，最老的在前。
        self._scratch_queued: list[tuple[int, int]] = []
        # Scratch slots whose store rides the batch built last pass.
        # 【译】其存储搭在上一轮构造的 batch 上的 scratch 槽位。
        self._scratch_issued: list[int] = []
        # The CPU tier beneath this one, or None. Not a member of
        # `BlockManager.state_caches`: every one of those is a veto over where a
        # prefix may resume, and this does the opposite -- it makes MORE
        # boundaries reachable. It is consulted in exactly two places,
        # `resumable_hit` (the vote) and `take_offload_stores` (the sink).
        # 【译】这一层下面的 CPU 层，或 None。不是 BlockManager.state_caches 的成员：那里的每一个都是对"前缀可以在哪里恢复"的否决，
        #   而它做的恰恰相反——让"更多"边界变得可达。它只在两个地方被询问：resumable_hit（投票）和 take_offload_stores（接收端）。
        self.offload = offload
        self.store = PageUnitCheckpointStore(
            pool, spec, offload_sink=offload is not None
        )
        # Keyed by `(seq id, prefix hash)` rather than by seq: two boundaries of
        # one prompt are two checkpoints, and keying by seq alone let the later
        # one overwrite the earlier before either was stored. Re-reaching the
        # *same* hash still collapses, which is what the hash in the key is
        # for -- that is one boundary reached twice, not two boundaries.
        # 【译】以 (seq id, prefix hash) 为键而不是以 seq 为键：一个 prompt 的两个边界是两个 checkpoint，只以 seq 为键会让后一个
        #   在两者都还没存之前就覆盖前一个。再次到达"同一个" hash 仍然会合并，这就是键里放 hash 的用处——那是一个边界到达了两次，
        #   而不是两个边界。
        # id(seq)：Python 内置函数，返回对象在其生命周期内唯一的整数标识（CPython 里是内存地址）
        self._pending: dict[tuple[int, int], tuple[Sequence, int]] = {}
        self._store_ops: list[CheckpointStoreOp] = []
        # External loads write temporary PAGE images before a runner gathers
        # them into an Active Slot. Owners include a transfer generation, so a
        # late release can never release a newer reservation.
        # 【译】外部加载会先写临时的 PAGE 镜像，之后再由 runner 把它们 gather 进一个活跃槽位。所有者里包含传输的 generation，
        #   所以一次迟到的释放永远不会释放一个更新的预留。
        # 【分支】reserve / release / adopt_transfer_units 只被 kv_transfer/offload/mp/native_state_scheduler.py 使用，见第9批
        self._transfer_units: dict[Hashable, tuple[int, ...]] = {}
        self.checkpoints_kept = 0
        self.checkpoints_dropped = 0
        self.checkpoints_orphaned = 0

    # applies：本类是否管这条 seq（开启且 seq 有每请求状态）；含义同 StateSlotPool.applies
    def applies(self, seq: Sequence) -> bool:
        return self.enabled and seq.has_per_req_cache

    def attach_midstep_scratch(self, slot_pool, max_ops_per_batch: int) -> None:
        """Where scratch slots come from, and the per-batch copy budget.

        `max_ops_per_batch` is what the runner's descriptor staging holds,
        stores and restores together (`AttentionMetadataBuilder.
        _checkpoint_staging`). A sequence can now publish several positions in
        one forward, so the one-store-per-sequence bound no longer holds on its
        own; `take_checkpoint_ops` begins only what fits and leaves the rest
        queued, still held in their scratch slots, for the next batch.
        """
        # 【译】scratch 槽位从哪里来，以及每个 batch 的拷贝预算。
        #   max_ops_per_batch 是 runner 的描述符暂存区能装的量，存与取合计（AttentionMetadataBuilder._checkpoint_staging）。
        #   一个序列现在可以在一次 forward 里发布好几个位置，所以"每序列一次存储"的上界单靠它自己已经不成立；
        #   take_checkpoint_ops 只开始装得下的那些，其余的留在队列里、仍占着各自的 scratch 槽位，等下一个 batch。
        # 调用方：BlockManager.__init__（slot_pool = self.state，max_ops_per_batch = checkpoint_copy_ops_per_batch(max_num_seqs)）
        self._scratch = slot_pool
        self._max_ops_per_batch = int(max_ops_per_batch)

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        """Borrow a scratch slot for each `(position, hash)` this forward covers.

        Best-effort and order-preserving, like the slot pool's: stops at the
        first position no slot is free for. A hash already stored, or already
        on its way, is skipped -- writing it again would be refused by
        `begin_store` after spending the slot and the copy.
        """
        # 【译】为这次 forward 覆盖到的每个 (position, hash) 借一个 scratch 槽位。
        #   尽力而为、保持顺序，和槽位池的一样：在第一个借不到空闲槽位的位置停下。已经存好、或已经在路上的 hash 被跳过——
        #   再写一次会在花掉槽位和拷贝之后被 begin_store 拒绝。
        # 调用方：BlockManager.plan_midstep；返回的 (slot, pos, h) 存进 seq.midstep_reservations，runner 据此拷贝
        if not (self.readable_midstep and self.applies(seq)):
            return []
        if self._scratch is None:
            return []
        queued = {h for h, _slot in self._scratch_queued}
        out = []
        for pos, h in positions:
            if h in queued or self.store.contains_or_pending(h):
                continue
            if not self._scratch.has_free():
                self.checkpoints_dropped += 1
                break
            out.append((self._scratch.pop(), pos, h))
        return out

    def publish_midstep(self, reservations: list[tuple], seq=None) -> None:
        """Queue each written scratch slot for storing under its hash.

        Nothing is findable yet: the image is begun at the next
        `take_checkpoint_ops` and READY one batch after that, the same two
        steps `checkpoint`'s boundaries take.
        """
        # 【译】把每个已写好的 scratch 槽位排队，等待以它的 hash 存储。
        #   此时还什么都找不到：镜像在下一次 take_checkpoint_ops 时开始，再过一个 batch 才 READY——和 checkpoint 的边界走的是同样两步。
        # 调用方：BlockManager.commit_midstep（hash_blocks 末尾）
        del seq
        for slot, _pos, h in reservations:
            self._scratch_queued.append((h, slot))

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Hand back scratch slots whose forward never ran."""
        # 【译】还回那些 forward 从没跑过的 scratch 槽位。调用方：BlockManager.cancel_midstep（重新规划、deallocate 时）
        for slot, _pos, _h in reservations:
            self._scratch.release(slot)

    def _take_scratch_stores(self, budget: int) -> None:
        """Begin queued scratch stores, at most `budget` of them."""
        # 【译】开始排队中的 scratch 存储，最多 budget 个。超出预算的留在队列里（仍占着 scratch 槽位）等下一个 batch
        queued, self._scratch_queued = self._scratch_queued, []
        for i, (h, slot) in enumerate(queued):
            if budget <= 0:
                self._scratch_queued = queued[i:]
                return
            op = None
            if not self.store.contains_or_pending(h):
                op = self.store.begin_store(h, slot)
            if op is None:
                # Already stored, or no units for it: the scratch holds
                # nothing anybody will read, and the forward that wrote it has
                # completed, so it can go back now.
                # 【译】已经存过了，或者没有单元可用：这个 scratch 里装的东西没人会读，而写它的 forward 已经完成，所以现在就可以还回去。
                self._scratch.release(slot)
                self.checkpoints_dropped += 1
                continue
            self._store_ops.append(op)
            self._scratch_issued.append(slot)
            self.checkpoints_kept += 1
            budget -= 1

    # resumable_hit（协议方法，见 state_cache.py）：从 hit-1 往左扫，返回第一个"镜像能拿到"（_reachable：HBM 里有 READY 的，
    #   或 CPU 层可能有）的边界 + 1；都没有返回 0。与 StateSlotPool 不同，这里没有 min_fork_tokens 测试——copy 不欠后继 token。
    #   assume_checkpointed=True 时直接返回 hit（反事实：每个边界都有快照），供 _record_checkpoint_demand 算按需位置
    # 调用方：BlockManager._gated_hit
    def resumable_hit(
        self,
        seq: Sequence,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        if not self.applies(seq):
            return hit
        for i in range(hit - 1, -1, -1):
            if assume_checkpointed or self._reachable(block_hashes[i]):
                return i + 1
        return 0

    def _reachable(self, h: int) -> bool:
        """Whether `h`'s image can be *produced*, in HBM or from the CPU tier.

        The scan runs right to left and stops at the first boundary it accepts,
        so accepting one nothing can deliver costs the whole walk-back, hiding
        every shorter checkpoint still in HBM. No preference rule is needed:
        both tiers key on the same content hash and `_attach_state_slots` tries
        HBM first regardless.

        The tier's half is deliberately optimistic -- `hashes` means "was stored
        once", never "is still there". A false positive costs one park plus a
        recompute and retracts itself (`fail_load` -> `forget`); being certain
        would cost a synchronous cross-process lookup on the admission path.

        The tier's half goes through `could_serve`, not a bare `h in hashes`, so
        it also honours `can_load`: a store-only role (`kv_producer`) populates
        `hashes` from its own stores but cannot load them back, and voting a hit
        it would then refuse strands the scan at the tier rung -- skipping a
        still-resident HBM rung -- and forfeits the checkpoint to a recompute.
        """
        # 【译】h 的镜像能否被"产生"出来，在 HBM 里或从 CPU 层。
        #   扫描从右往左、停在它接受的第一个边界，所以接受一个谁也交付不了的边界，代价是整段回退——它会把每个仍在 HBM 里的更短 checkpoint
        #   都挡住。不需要优先级规则：两层都以同一个内容 hash 为键，而 _attach_state_slots 无论如何都先试 HBM。
        #
        #   这一层的那一半是故意乐观的——hashes 意味着"曾经存过一次"，绝不是"现在还在"。假阳性的代价是一次停车加一次重算，
        #   并且会自我撤回（fail_load -> forget）；要做到确定，就得在准入路径上做一次同步的跨进程查找。
        #
        #   这一层的那一半走 could_serve 而不是裸的 h in hashes，所以它也遵守 can_load：只存的角色（kv_producer）用自己的存储填充
        #   hashes，却不能把它们加载回来；投出一个它随后会拒绝的命中，会让扫描卡在这一层的那一级——跳过一个仍在 HBM 里的级——
        #   并把这个 checkpoint 丢给一次重算。
        if self.store.contains(h):
            return True
        return self.offload is not None and self.offload.could_serve(h)

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """File a boundary to be stored, keyed by hash rather than by seq.

        Every boundary a seq reaches survives, not just its last: an anchor and
        the prompt-end checkpoint that follows it a chunk later are separate
        entries. Keying by seq alone would have the second overwrite the first
        before either is stored, which costs a shortened prefill chunk on every
        prompt and buys nothing. What makes keeping both affordable is the
        image's price against a whole Active Slot under `fork`.

        The anchor is the placement that pays, and the ladder is not (see
        `BlockManager._record_checkpoint_end`), so
        `--state-checkpoint-interval-tokens -1` drops the grid and leaves the
        anchor and the demand rung as the only two placements.

        One entry per seq per drain, though, and that is not the same as one
        per seq. A pending boundary names a hash and the slot that will be read
        for it, and the slot is read at the drain -- so two boundaries surviving
        into one drain would both be stored from whatever the *last* forward
        left there, filing the earlier hash over the later state. A request
        resuming on it would continue from a point ahead of its own prefix, and
        nothing downstream could tell: `_validate_paged_state_op` checks layout,
        size and unit count, all of which still match.

        A drain normally follows every forward, so the two boundaries of one
        prompt are ordinarily stored from separate slots correctly. The
        exception is a pass that schedules nothing (`scheduler.py:1828` passes
        `state_maintenance_ops=None` on an empty batch), which carries
        `_pending` into the next drain. `_supersede` resolves that the only way
        the bytes allow: the newer boundary is the one the slot actually holds,
        so it wins and the older is dropped rather than mis-stored.
        """
        # 【译】登记一个待存的边界，以 hash 而不是以 seq 为键。
        #   一个 seq 到达的每个边界都会保留下来，而不只是最后一个：一个锚点，和它之后一个 chunk 才到的 prompt 末尾 checkpoint，
        #   是两个独立的条目。只以 seq 为键会让第二个在两者都没存之前就覆盖第一个，这会让每个 prompt 都付出一个被截短的 prefill chunk，
        #   却什么也换不来。让"两个都保留"付得起的，是镜像的价格相对于 fork 下一整个活跃槽位而言很便宜。
        #
        #   锚点是会带来回报的放置，阶梯不是（见 BlockManager._record_checkpoint_end），所以 --state-checkpoint-interval-tokens -1
        #   会去掉网格，只留锚点和按需一级这两种放置。
        #
        #   不过是"每个 seq 每次排空一个条目"，这和"每个 seq 一个"不是一回事。一个待存边界指明了一个 hash 和将被读取的槽位，
        #   而槽位是在排空时才读的——所以如果两个边界都活到同一次排空，它们都会从"最后一次" forward 留在那里的东西存起来，
        #   把较早的 hash 登记在较晚的状态上。从它恢复的请求会从一个"超前于它自己前缀"的点继续，而下游没有任何东西能察觉：
        #   _validate_paged_state_op 检查的布局、大小和单元数都仍然对得上。
        #
        #   通常每次 forward 之后都有一次排空，所以一个 prompt 的两个边界通常从各自不同的槽位状态正确地存下来。例外是一个什么都没调度的轮次
        #   （scheduler.py:1828 在空 batch 上传 state_maintenance_ops=None），它把 _pending 带进下一次排空。_supersede 用字节唯一允许的方式
        #   解决：较新的边界才是槽位里实际装着的，所以它胜出，较老的被丢弃而不是被存错。
        # 观察：docstring 里的 scheduler.py:1828 是旧行号；v0.1.8 中对应的是 Scheduler._schedule 构造 decode batch 处
        #   state_maintenance_ops=(take_state_maintenance_ops() if scheduled_seqs else None)
        # 调用方：BlockManager.hash_blocks（checkpointers_at 认可本类时）；boundary_blocks 不用（del 掉），位置已经编码在 h 里
        del boundary_blocks
        if self.applies(seq) and seq.state_slot >= 0:
            self._supersede(id(seq))
            self._pending[(id(seq), h)] = (seq, h)

    def _supersede(self, seq_id: int) -> None:
        """Drop this seq's earlier pending boundaries; the slot has moved on.

        Counted as dropped, not silently forgotten: this is reuse the placement
        asked for and did not get, and it is the only signal that empty passes
        are costing checkpoints.
        """
        # 【译】丢掉这个 seq 更早的待存边界；槽位已经往前走了。
        #   计为 dropped，而不是悄悄忘掉：这是放置要求了却没得到的复用，也是"空轮次正在让我们损失 checkpoint"的唯一信号。
        stale = [k for k in self._pending if k[0] == seq_id]
        for key in stale:
            del self._pending[key]
        self.checkpoints_dropped += len(stale)

    def forget_pending(self, seq: Sequence) -> None:
        """Drop every boundary this seq had pending, not just its last.

        A seq can now hold several. All of them describe state in the slot
        that is about to go back on the free list, so all of them die with it.
        """
        # 【译】丢掉这个 seq 所有待存的边界，而不只是最后一个。
        #   一个 seq 现在可以有好几个。它们描述的都是那个马上要回到空闲表的槽位里的状态，所以全部随它一起死掉。
        #   （同时取消写这个槽位的排队恢复）调用方：BlockManager.deallocate
        seq_id = id(seq)
        for key in [k for k in self._pending if k[0] == seq_id]:
            del self._pending[key]
        self.store.cancel_queued_restore(seq.state_slot)

    def restore_queued_for(self, dst_slot: int) -> bool:
        """Whether a queued restore will write `dst_slot` on the next batch.

        The evidence that a PAGE checkpoint really is behind a boundary the
        request has claimed; see `BlockManager._state_leg_secured`.
        """
        # 【译】是否有一个排队中的恢复会在下一个 batch 写 dst_slot。
        #   这是"一个 PAGE checkpoint 确实在这个请求认领的边界后面"的证据；见 BlockManager._state_leg_secured。
        return self.store.restore_queued_for(dst_slot)

    # begin_restore：命中时排一次恢复（见 store.begin_restore）；调用方 BlockManager._attach_state_slots，False 时再试 CPU 层加载
    def begin_restore(self, h: int, dst_slot: int) -> bool:
        return self.store.begin_restore(h, dst_slot) is not None

    # take_checkpoint_ops：构造 batch 时调用一次（BlockManager.take_state_maintenance_ops），返回 (本批要执行的存, 本批要执行的取)
    #   1. 把 _pending 里每个边界变成 begin_store(h, seq.state_slot)（槽位此刻装的正是那个边界的状态，见 checkpoint 的 docstring）
    #   2. 取走排队的恢复（store.take_restore_ops）
    #   3. 用剩下的描述符预算（max_ops_per_batch − 存 − 取）开始排队中的 midstep scratch 存储
    def take_checkpoint_ops(
        self,
    ) -> tuple[tuple[CheckpointStoreOp, ...], tuple[CheckpointRestoreOp, ...]]:
        pending, self._pending = self._pending, {}
        for seq, h in pending.values():
            # Safe to read now because `checkpoint` keeps at most one pending
            # boundary per seq: this slot holds the state as of that boundary
            # and no other. See `_supersede`.
            # 【译】现在读是安全的，因为 checkpoint 对每个 seq 最多只保留一个待存边界：这个槽位装的就是那个边界时的状态，没有别的。见 _supersede。
            src_slot = seq.state_slot
            if src_slot < 0 or self.store.contains_or_pending(h):
                continue
            op = self.store.begin_store(h, src_slot)
            if op is None:
                self.checkpoints_dropped += 1
                continue
            self._store_ops.append(op)
            self.checkpoints_kept += 1
        restores = self.store.take_restore_ops()
        if self._scratch_queued:
            self._take_scratch_stores(
                self._max_ops_per_batch - len(self._store_ops) - len(restores)
            )
        stores, self._store_ops = self._store_ops, []
        return tuple(stores), restores

    # complete_previous_batch：每轮开头（BlockManager.complete_previous_state_batch）结算上一批：存 → READY、取 → 放 pin、scratch 还回状态池
    def complete_previous_batch(self) -> None:
        self.store.complete_inflight()
        # The batch that carried these stores has been issued.
        # 【译】携带这些存储的 batch 已经发出。
        issued, self._scratch_issued = self._scratch_issued, []
        for slot in issued:
            self._scratch.release(slot)

    # 下面两个直接转给 store：BlockManager 的 _has_page_units / _checkpoint_has_room / _ensure_page_units 用它们做 KV 池的准入与分配
    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        return self.store.has_available_units(count, protected_hash)

    def ensure_free_units(self, count: int) -> bool:
        return self.store.ensure_free_units(count)

    def reserve_transfer_units(self, owner: Hashable) -> tuple[int, ...] | None:
        """Reserve one raw checkpoint image for an admitted external load.

        `owner` must uniquely identify this transfer attempt, including its
        generation. Duplicate active owners and insufficient available units
        are refused. The caller releases these units only after both the
        transfer and restore stop using them. No checkpoint is published and no Active Slot is touched.
        """
        # 【译】为一次已准入的外部加载预留一份原始 checkpoint 镜像的单元。
        #   owner 必须唯一地标识这次传输尝试，包括它的 generation。重复的活跃所有者、或可用单元不足都会被拒绝。
        #   调用方只有在传输和恢复都不再使用这些单元之后才释放它们。不发布任何 checkpoint，也不碰任何活跃槽位。
        if owner is None:
            raise ValueError("a transfer reservation needs an owner")
        if owner in self._transfer_units:
            return None
        if not self.store.ensure_free_units(self.store.units_per_checkpoint):
            return None
        units = self.store.pool.reserve_units(
            self.store.units_per_checkpoint, ("state-transfer", owner)
        )
        if units is None:
            return None
        self._transfer_units[owner] = tuple(units)
        return self._transfer_units[owner]

    def release_transfer_units(self, owner: Hashable) -> None:
        """Release a completed or safely cancelled load reservation, once."""
        # 【译】释放一个已完成或已安全取消的加载预留，只释放一次。
        units = self._transfer_units.pop(owner, None)
        if units is not None:
            self.store.pool.release_units(units, ("state-transfer", owner))

    def adopt_transfer_units(self, owner: Hashable, prefix_hash: int) -> bool:
        """Publish a completed external image without exposing it to reuse.

        Returns ``True`` when the transfer's units became the canonical READY
        checkpoint. If the same hash was published concurrently, the incoming
        units are released and ``False`` is returned. The adopted checkpoint is
        deliberately not nominated for offload: it just arrived from that tier.
        """
        # 【译】发布一份已完成的外部镜像，同时不让它在中途暴露给复用。
        #   当这次传输的单元成为规范的 READY checkpoint 时返回 True。如果同一个 hash 被并发发布了，进来的单元被释放并返回 False。
        #   被收养的 checkpoint 故意不提名去 offload：它就是刚从那一层来的。
        units = self._transfer_units.get(owner)
        if units is None:
            return False
        adopted = self.store.adopt_units(units, ("state-transfer", owner), prefix_hash)
        # Only now: if adoption raised, the units stay reachable for release.
        # 【译】只在现在才删：如果收养抛了异常，这些单元仍然能被找到并释放。
        del self._transfer_units[owner]
        return adopted

    # 下面三个挂起 / 恢复 / 释放排队恢复的方法直接转给 store（【分支】native_state_scheduler.py 用，见第9批）
    def suspend_queued_restore(
        self, dst_slot: int
    ) -> SuspendedCheckpointRestore | None:
        return self.store.suspend_queued_restore(dst_slot)

    def resume_suspended_restore(self, restore: SuspendedCheckpointRestore) -> None:
        self.store.resume_suspended_restore(restore)

    def release_suspended_restore(self, restore: SuspendedCheckpointRestore) -> None:
        self.store.release_suspended_restore(restore)

    def contains(self, h: int) -> bool:
        """Whether HBM holds a READY image for `h` right now.

        Distinct from `_reachable`, which also counts the CPU tier: this one
        answers "would a resume be free", and is what the `state_hbm` /
        `state_tier` split reports.
        """
        # 【译】HBM 现在是否持有 h 的一份 READY 镜像。
        #   与 _reachable 不同，那个还算上了 CPU 层：这个回答的是"一次恢复是否免费"，也是 state_hbm / state_tier 划分所报告的。
        # 调用方：BlockManager._joint_kv_boundary（state_from_hbm）
        return self.store.contains(h)

    def attach_offload(self, index, *, sink: bool = True) -> None:
        """Wire the CPU tier in, after both objects exist.

        Not a constructor argument because the two are built in the wrong
        order: this class needs `checkpoint_spec`, which comes off
        `state_runtime`, while the index needs the connector config. One call
        sets both halves -- the vote (`_reachable`) and the sink (whether a
        READY checkpoint is nominated at all) -- so they cannot be turned on
        separately, which would either vote for hashes nothing stores or pin
        units nothing releases.
        """
        # 【译】在两个对象都存在之后，把 CPU 层接进来。
        #   不作为构造参数，是因为两者的构造顺序是反的：本类需要 checkpoint_spec，它来自 state_runtime；而索引需要 connector 配置。
        #   一次调用设置两半——投票（_reachable）和接收端（一个 READY checkpoint 到底要不要被提名）——所以它们不能被分开打开，
        #   否则要么为没人存的 hash 投票，要么钉住没人释放的单元。
        # 调用方：BlockManager.__init__（state_offload 建好之后）
        self.offload = index
        # `sink` is the store half specifically: a load-only role still votes
        # off `hashes` but must nominate nothing, since a nomination that is
        # handed over takes a pin nobody will release.
        # 【译】sink 专指存储那一半：只取的角色仍然根据 hashes 投票，但什么都不能提名，因为一次被交出去的提名会取一个没人会释放的 pin。
        self.store._offload_sink = index is not None and bool(sink)

    # 【分支】下面七个方法是 CPU 层存储一侧的薄转发（调用方：BlockManager 的 take_state_stores / settle_state_store /
    #   release_state_store_source / has_pending_state_store_pins / reclaim_stale_state_store_pins，以及 native_state_scheduler.py），
    #   语义见 PageUnitCheckpointStore 同名方法
    def take_offload_stores(
        self, max_inflight: int
    ) -> list[tuple[StateStoreOperationId, tuple[int, ...]]]:
        """`(operation, unit_ids)` to hand the tier now. See the store."""
        # 【译】现在要交给这一层的 (operation, unit_ids)。见 store。
        return self.store.take_offload_stores(max_inflight)

    def acquire_checkpoint_source(
        self, prefix_hash: int, *, max_inflight: int | None = None
    ) -> tuple[StateStoreOperationId, tuple[int, ...]] | None:
        """Lease an admitted transfer's exact READY image. See the store."""
        # 【译】为一次已准入的传输租用它确切的 READY 镜像。见 store。
        return self.store.acquire_checkpoint_source(
            prefix_hash, max_inflight=max_inflight
        )

    def release_offload_store_source(self, op: StateStoreOperationId) -> None:
        """The gather drained; hand the units back but keep the pin. See store."""
        # 【译】gather 已排空；把单元还回去，但保留 pin。见 store。
        self.store.release_offload_store_source(op)

    def settle_offload_store(self, op: StateStoreOperationId) -> None:
        """One store reported, either way; retire the operation. See store."""
        # 【译】一次存储报告了，不管结果如何；让这个操作退役。见 store。
        self.store.settle_offload_store(op)

    def has_offload_pins(self) -> bool:
        """Whether any dispatched store is still pinning its units. See store."""
        # 【译】是否还有任何已派发的存储在钉着它的单元。见 store。
        return self.store.has_offload_pins()

    def reclaim_stale_offload_pins(self, timeout_s: float) -> int:
        """Release offload pins whose report never came. See the store."""
        # 【译】释放那些报告一直没来的 offload pin。见 store。
        return self.store.reclaim_stale_offload_pins(timeout_s)

    def was_reclaimed(self, op) -> bool:
        """Whether this store's source was taken back before it reported."""
        # 【译】这次存储的源是否在它报告之前就被收回了。
        return self.store.was_reclaimed(op)

    def unindex(self, h: int) -> None:
        # `_pending` is keyed by `(seq, hash)`, so one hash can be pending for
        # several sequences at once -- two turns of a conversation reaching the
        # same boundary. All of them lose it together.
        # 【译】_pending 以 (seq, hash) 为键，所以一个 hash 可以同时为好几个序列待存——一段对话的两轮到达了同一个边界。它们一起失去它。
        # 调用方：BlockManager._record_evicted（该 hash 的 KV block 被淘汰）；有东西被删就计一次 checkpoints_orphaned
        stale = [key for key, (_, pending_h) in self._pending.items() if pending_h == h]
        for key in stale:
            del self._pending[key]
        removed = self.store.unindex(h)
        if stale or removed:
            self.checkpoints_orphaned += 1

    def clear_index(self) -> None:
        """Drop everything, for `/reset_prefix_cache`-style admin calls.

        None of the four fates moves, and that is deliberate rather than an
        oversight: each argues for a different fix — `dropped` for a bigger
        pool, `evicted` for a longer-lived one, `orphaned` for a bigger paged
        pool — and an operator emptying the cache on purpose argues for none of
        them. Charging a reset to any of them would send tuning after a number
        the operator created. The reset is visible in the drop in
        `checkpoints_kept`'s growth rate, and in the admin call itself.
        """
        # 【译】丢掉一切，用于 /reset_prefix_cache 一类的管理调用。
        #   四种命运计数一个都不动，这是故意的而不是疏忽：每一种都指向不同的修法——dropped 要更大的池，evicted 要活得更久的池，
        #   orphaned 要更大的分页池——而运维有意清空缓存，指向的是哪一种都不是。把一次重置记到任何一种上，都会让调优去追一个
        #   运维自己制造的数字。重置能从 checkpoints_kept 增长率的下降里看出来，也能从那次管理调用本身看出来。
        # 调用方：BlockManager.clear_cache
        self._pending.clear()
        self.store.clear()

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of this class's checkpoints, plus the CPU tier's own.

        The tier's counters are folded in here rather than reported separately
        because the two only mean anything against each other: `stores_completed`
        is worth reading only as a fraction of `checkpoints_kept`, and a tier
        that stored nothing is indistinguishable from a tier that was never
        asked unless both numbers sit on one line. Folding them in makes the
        load and store legs visible again under PAGE.

        Keys are prefixed `state_offload_` because the aggregation upstream is
        by key across every state class.
        """
        # 【译】这个类的 checkpoint 后来怎样了，外加 CPU 层自己的。
        #   CPU 层的计数在这里合并，而不是单独报告，因为两者只有放在一起才有意义：stores_completed 只有作为 checkpoints_kept 的一部分
        #   才值得读，而一个什么都没存的层，和一个从没被要求过的层，除非两个数字在同一行上，否则无法区分。合并进来让 PAGE 下的
        #   加载和存储两条腿重新可见。
        #
        #   键以 state_offload_ 为前缀，因为上游是按键在所有状态类之间汇总的。
        # 调用方：BlockManager.state_checkpoint_fates（周期性的 "state checkpoints:" 日志）
        fates = {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.store.evictions,
            # Non-zero means store reports are being lost, which would
            # otherwise show only as a pool that has quietly shrunk.
            # 【译】非零表示存储报告在丢失，否则这只会表现为一个悄悄缩小了的池子。
            "offload_pins_reclaimed": self.store.offload_pins_reclaimed,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }
        # `getattr`, not a direct call: `attach_offload` accepts anything that
        # answers `hashes`, and the tests attach a double that does not carry
        # counters. A missing `stats` means "no numbers to fold", not an error.
        # 【译】用 getattr 而不是直接调用：attach_offload 接受任何能回答 hashes 的东西，而测试接上的替身不带计数器。
        #   缺 stats 表示"没有数字可合并"，不是错误。
        stats = getattr(self.offload, "stats", None) if self.offload else None
        if callable(stats):
            fates.update((f"state_offload_{k}", v) for k, v in stats().items())
            # A gauge, not a counter: how many nominations are waiting for a
            # slot under `OFFLOAD_MAX_PENDING_SAVES`. The queue is bounded by
            # `_offload_backlog_cap` and drained oldest-first, so a value that
            # climbs toward the cap means checkpoints are reaching READY faster
            # than the tier drains them and fresh nominations are queueing behind
            # stale ones; at the cap the oldest are dropped, counted next to it.
            # 【译】是仪表值而不是计数器：有多少提名在 OFFLOAD_MAX_PENDING_SAVES 之下等一个名额。队列受 _offload_backlog_cap 限制、
            #   从最老的开始排空，所以一个向上限攀升的值意味着 checkpoint 变 READY 的速度快于这一层排空它们的速度，新提名排在陈旧提名后面；
            #   到了上限，最老的被丢弃，丢弃数紧挨着它记录。
            fates["state_offload_store_backlog"] = len(self.store.store_backlog)
            fates["state_offload_nominations_dropped"] = (
                self.store.offload_nominations_dropped
            )
        return fates
