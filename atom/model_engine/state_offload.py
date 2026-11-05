# SPDX-License-Identifier: MIT
"""Scheduler-side bookkeeping for the state-cache offload tier.

One thing lives here and it touches no device: the set of hashes believed to be
in LMCache, plus the loads offered against them. The bytes and the transfers are
the worker's business (`kv_transfer/offload/hybrid/kimi_k3/state_tier.py`).

The set is in memory and never persisted: `LocalDiskBackend.__init__` starts
from an empty dict and never scans its directory, so an index recovered from
disk would be a pure false-positive generator. Index and bytes share one server
lifetime.
"""

# 【译】状态缓存 offload 层在调度器一侧的账本。
#   这里只放一样东西，而且它不碰任何设备：一组"被认为在 LMCache 里"的 hash，以及针对它们发起的加载。
#   字节和传输是 worker 的事（kv_transfer/offload/hybrid/kimi_k3/state_tier.py）。
#
#   这组 hash 只在内存里、从不持久化：LocalDiskBackend.__init__ 从一个空 dict 开始、从不扫描它的目录，
#   所以从磁盘恢复出来的索引只会不停地制造假阳性。索引和字节的生命周期都是同一次服务进程。
#
# =====================================================================================
# 【导读】【分支】状态快照的 CPU 层（state tier）——引擎进程里的索引与能力判定（第4批 4b 的 B 级；完整的 offload 见第9批 9c）
#   前置①：KV offload 分层——GPU 显存（HBM）里的前缀缓存是第一级，被淘汰的 KV 另存到 CPU 内存 / LMCache（第二级，更大更慢），
#     以后同前缀的请求从第二级读回、只 prefill 剩下的部分（完整讲解见 arg_utils.py 文件开头的【科普】KV offload，实现在 atom/kv_transfer/offload/）
#   前置②：带每请求状态的模型光把 KV 读回来不够，还要那个位置的状态快照（state_pool.py 文件开头【科普】③）；
#     而 HBM 里"快照 ⊆ KV"：边界 block 一被淘汰，快照就 unindex（同处④(d)）。所以 KV 去了 CPU，快照也得跟着去，
#     这就是"状态层"：把页单元快照（state_pool.py【科普】⑤）也存一份到 LMCache
#
#       HBM（GPU）                                         CPU（LMCache；字节由 worker 一侧的 StateOffloadTier 搬运）
#       KV block 前缀缓存   ─────淘汰前另存────────────>    KV chunk（每块 LMCache chunk_size 个 token）
#       页单元状态快照       ─READY 后提名，D2H 拷出───>    状态快照（每份 image_bytes；总量 OFFLOAD_STATE_CPU_SIZE，默认 32 GiB）
#
#       新请求：HBM 前缀命中到 hbm；CPU 里还有更远的边界 B ——
#         KV 腿取覆盖 [hbm, B) 的 chunk，状态腿取 B 处的快照（StateOffloadIndex.request_load），
#         两条腿都能到 B 才一起执行，否则都不执行（joint boundary，BlockManager._joint_kv_boundary）；请求先 park，数据到了再继续
#
#   本文件的三样东西：
#     StateOffloadIndex       "CPU 里大概有哪些快照"（乐观的 hash 集合）+ 加载 / 存储的计数；由 BlockManager 持有（self.state_offload）
#     state_tier_capability   只看配置判断 worker 会不会真的建状态层、能存还是能取（BlockManager.__init__ 调用）
#     state_tier_cpu_bytes / state_tier_chunk_tokens   CPU 预算与 LMCache chunk 大小（引擎、worker 两侧共用同一个读法）
#   生效条件：kv_transfer_config 的 connector 是 lmcache_offload（或 multi 里恰好一个），PP=1，布局是 kimi_k3 / qwen，
#     kv_role 能存或能取；并且模型走 copy（页单元）快照——fork 方式下没有页单元可存（BlockManager.take_state_stores 直接返回空），索引始终为空
#   对比nano-vllm：没有对应物（nano-vllm 没有 KV offload，也没有状态快照）。
# =====================================================================================

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, replace

logger = logging.getLogger("atom")


class StateOffloadIndex:
    """What is believed to be in LMCache, and what is being fetched back.

    `hashes` answers the membership half of the tier's vote. It is deliberately
    optimistic: LMCache's own LRU can drop bytes at any time, so a hash here
    means "was stored once", never "is still there". The false positive costs
    one lookup and a park/unpark, and is handled by the `failed_loading` path
    (`fail_load` -> `forget`).
    """

    # 【译】被认为在 LMCache 里的东西，以及正在取回的东西。
    #   hashes 回答的是这一层投票里"在不在"的那一半。它是故意乐观的：LMCache 自己的 LRU 随时可能丢掉字节，
    #   所以这里的一个 hash 意味着"曾经存过一次"，而绝不是"现在还在"。假阳性的代价是一次查找加一次停车 / 出车（park/unpark），
    #   由 failed_loading 路径处理（fail_load -> forget）。
    # 构造：BlockManager.__init__（state_tier_capability(config).hosts_state_tier 为真时），max_cpu_entries =
    #   state_tier_cpu_bytes() // checkpoint_spec.image_bytes（CPU 预算能放几份快照）；copy 方式下再 attach_offload 给快照协调器
    # 读写方：BlockManager（_tier_can_serve / _request_state_load / settle_state_load / settle_state_store / take_state_stores 等）、
    #   page_unit_checkpoint.py PagedStateCheckpointCoordinator._reachable（投票）与 checkpoint_fates（合并 stats）

    def __init__(
        self,
        *,
        can_store: bool = True,
        can_load: bool = True,
        max_cpu_entries: int | None = None,
    ) -> None:
        # The two legs are separately granted. A `kv_producer` role saves and
        # never loads, a `kv_consumer` loads and never saves, and offering a
        # load the worker will not serve parks the request that took it.
        # 【译】两条腿是分别授予的。kv_producer 角色只存不取，kv_consumer 只取不存；提供一次 worker 不会执行的加载，
        #   会让接受它的请求一直停着。
        self.can_store = bool(can_store)
        self.can_load = bool(can_load)
        # The optimistic membership index (see the class docstring). It only
        # grows on `note_stored` and shrinks on `forget` (a failed load), so a
        # long-lived server that stored many distinct prefixes would grow it
        # without bound. Cap it: `hashes` stays a plain set -- membership stays
        # O(1) and every caller/test keeps `in` / `len` / `== set()` /
        # `discard` -- and `_hash_lru` records insertion order beside it purely
        # so the oldest hash can be dropped on overflow. The two move in
        # lockstep (every add and every forget touches both), so `_hash_lru`
        # never diverges in length from `hashes`. Dropping the oldest hash is
        # safe for the same reason the index is optimistic: a false "not stored"
        # only costs one recompute, never wrong output.
        # 【译】乐观的成员索引（见类 docstring）。它只在 note_stored 时增长、在 forget（加载失败）时缩小，所以一个存过很多不同前缀的
        #   长期运行的服务会让它无限增长。给它设上限：hashes 仍是普通 set——成员判断保持 O(1)，每个调用方 / 测试照旧用
        #   in / len / == set() / discard——_hash_lru 在旁边只记录插入顺序，好在溢出时丢掉最老的 hash。两者同步变化
        #   （每次加入和每次 forget 都同时动两者），所以 _hash_lru 的长度永远不会和 hashes 不一致。丢掉最老的 hash 是安全的，
        #   理由和索引是乐观的一样：错误的"没存过"只会多一次重算，绝不会产生错误输出。
        # OrderedDict：记住插入顺序的 dict，popitem(last=False) 弹出最早插入的，move_to_end 把某项挪到最新端——常用来实现 LRU
        self.hashes: set[int] = set()
        self._hash_lru: OrderedDict[int, None] = OrderedDict()
        # ~1M hashes -> a few tens of MB: the leak bound when no codec bound
        # is known.
        # 【译】约 100 万个 hash -> 几十 MB：不知道 codec 上限时的泄漏上界。
        # 有 max_cpu_entries 时取 min(2^20, max_cpu_entries − 余量)，余量 = max(16, max_cpu_entries // 32)，让索引比 CPU 侧略早丢掉旧项
        # 例：image_bytes = 53.6 MiB（Kimi-K3），32 GiB → max_cpu_entries = 611，余量 = max(16, 19) = 19 → _hash_cap = 592
        self._hash_cap = 1 << 20
        if max_cpu_entries is not None and max_cpu_entries > 0:
            margin = max(16, max_cpu_entries // 32)
            self._hash_cap = max(1, min(self._hash_cap, max_cpu_entries - margin))
        self.hashes_evicted = 0
        # req_id -> hash, for loads offered and not yet settled. Keyed by
        # request because that is what comes back, on the same
        # `finished_loading`/`failed_loading` channel a KV load uses.
        # 【译】req_id -> hash，记录已经发出、尚未了结的加载。以请求为键，因为回来的报告就是按请求来的——
        #   和 KV 加载用的是同一条 finished_loading / failed_loading 通道。
        self.pending_loads: dict = {}
        self.loads_attempted = 0
        self.loads_completed = 0
        self.loads_failed = 0
        # Store-side counters. Until this commit the only store-side signal was
        # `indexed` growing, which is why a shell that refused every store
        # (`enqueue_state_stores` never forwarded) looked identical to a tier
        # with nothing to do -- 94 refusals produced one warning line and no
        # number anywhere. `stores_refused` is the probe that would have said
        # so on the first pass, and it is deliberately apart from
        # `stores_failed`: refused means nobody tried, failed means the worker
        # tried and could not.
        # 【译】存储一侧的计数器。在这次提交之前，存储一侧唯一的信号是 indexed 在增长，这就是为什么一个拒绝了所有存储的外壳
        #   （enqueue_state_stores 从未被转发）看起来和一个无事可做的层一模一样——94 次拒绝只产生了一行警告，哪里都没有数字。
        #   stores_refused 就是能在第一轮就说明这一点的探针，它故意和 stores_failed 分开：refused 表示没人尝试过，
        #   failed 表示 worker 试了但没成功。
        # 写入方：BlockManager.take_state_stores（attempted）、settle_state_store（completed / failed / untrusted）、
        #   Scheduler 发现 connector 不收存储时（refused，scheduler.py）
        self.stores_attempted = 0
        self.stores_completed = 0
        self.stores_failed = 0
        self.stores_refused = 0
        # Stores that reported success after the stale reclaimer had already
        # taken their source units back. Not counted as completed: nothing can
        # say the worker had stopped reading, so the image is forfeited rather
        # than indexed. Any non-zero value means the reclaim window is firing
        # on live transfers and is set too low.
        # 【译】在陈旧回收器已经把源页单元收回之后才报告成功的存储。不计为 completed：没有任何东西能证明 worker 当时已经停止读取，
        #   所以这份镜像被放弃而不是被登记。任何非零值都说明回收窗口在对活着的传输开火，设得太短了。
        self.stores_untrusted = 0

    # ------------------------------- stores -------------------------------- #
    # 【译】存储
    def note_stored(self, h: int) -> None:
        """A store landed in LMCache and the hash is now worth voting for.

        Called from the report the worker sends back, never at submission: a
        hash advertised before its bytes exist parks the next request over that
        prefix against a `get` that must miss.
        """
        # 【译】一次存储已经落进 LMCache，这个 hash 现在值得投票了。
        #   从 worker 发回的报告里调用，绝不在提交时调用：在字节存在之前就宣传一个 hash，会让该前缀上的下一个请求对着一次
        #   必然 miss 的 get 停车。
        # 调用方：BlockManager.settle_state_store（报告成功、且源未被回收时）
        h = int(h)
        self.hashes.add(h)
        # Reinsert at the young end whether or not it was already present, so a
        # re-stored prefix is treated as freshly used, then trim the cold end.
        # 【译】不管原来在不在，都重新插到"年轻"的一端，这样被重新存储的前缀被当作刚用过；然后修剪冷的一端。
        self._hash_lru.pop(h, None)
        self._hash_lru[h] = None
        while len(self._hash_lru) > self._hash_cap:
            old, _ = self._hash_lru.popitem(last=False)
            self.hashes.discard(old)
            self.hashes_evicted += 1

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        # 【译】丢掉一个加载失败的 hash，让下一个请求不再重试。
        self.hashes.discard(h)
        self._hash_lru.pop(h, None)

    def could_serve(self, h: int) -> bool:
        """Whether a load for `h` could be offered at all: this tier can load
        AND believes it holds `h`.

        The membership+capability half of `request_load`'s guard (minus its
        per-request in-flight check), factored out so the admission-path voters
        -- `PageUnitCheckpointCoordinator._reachable` and
        `BlockManager._tier_can_serve` -- test exactly what `request_load` will
        accept and cannot drift from it. Mirroring only `h in hashes` there,
        without `can_load`, made a `kv_producer` (can_load=False, but `hashes`
        populated from its own stores) vote a tier hit it would then refuse: the
        right-to-left resumable scan stopped at the tier rung, skipped a
        still-resident HBM rung, and `request_load` returned False -- the
        boundary disowned and the HBM checkpoint forfeited to a full recompute.
        """
        # 【译】能否为 h 提供一次加载：这一层能取，并且相信自己持有 h。
        #   它是 request_load 守卫里"成员 + 能力"的那一半（不含"每个请求只能有一个在途加载"的检查），单独提出来，
        #   让准入路径上的投票者——PageUnitCheckpointCoordinator._reachable 和 BlockManager._tier_can_serve——检查的
        #   恰好就是 request_load 会接受的条件，不会偏离。那里如果只镜像 h in hashes 而不看 can_load，会让 kv_producer
        #   （can_load=False，但 hashes 由它自己的存储填充）投出一个它随后会拒绝的命中：从右往左的可恢复扫描停在这一层的那一级，
        #   跳过了一个仍在 HBM 里的快照，然后 request_load 返回 False——边界被放弃，HBM 里的快照也白白丢掉、改成完整重算。
        # 观察：docstring 里的 PageUnitCheckpointCoordinator 实际叫 PagedStateCheckpointCoordinator（page_unit_checkpoint.py）
        return self.can_load and h in self.hashes

    # -------------------------------- loads -------------------------------- #
    # 【译】加载
    def request_load(self, req_id, h: int) -> bool:
        """Offer to fetch `h` back for `req_id`. False if this tier cannot.

        The guard between believing and delivering: a load is resolved only by
        a worker report, so offering one for a hash never stored would park the
        request against bytes no `get` can produce.
        """
        # 【译】为 req_id 发起取回 h。这一层做不到时返回 False。
        #   这是"相信"和"交付"之间的守卫：加载只能由 worker 的报告来了结，所以为一个从没存过的 hash 发起加载，
        #   会让请求对着任何 get 都产生不了的字节停车。
        # 调用方：BlockManager._request_state_load（HBM 里没有这个快照时）；返回 True 后请求带着 offload_joint.load_hash 停车，
        #   等 worker 报告后由 settle_state_load 调 complete_load / fail_load
        if not self.could_serve(h):
            # A store-only role (can_load False) or a hash never stored. Voting
            # for either parks the request against a report that never comes;
            # refuse so the boundary is disowned and recomputed.
            # 【译】只存不取的角色（can_load 为 False），或从没存过的 hash。为任何一种投票都会让请求对着一份永远不来的报告停车；
            #   拒绝，让这个边界被放弃并重算。
            return False
        if req_id in self.pending_loads:
            # One request, one outstanding load: reports are keyed by request
            # id, so the first completion would unpark while the second is
            # still writing. No in-tree path reaches this today, and the
            # refusal costs only a disown.
            # 【译】一个请求同时只能有一个在途加载：报告按请求 id 区分，第一份完成报告会让请求出车，而第二份还在写。
            #   今天树内没有路径会走到这里，拒绝的代价只是放弃一个边界。
            logger.warning(
                "state offload: request %s already has a load in flight; "
                "refusing a second one and letting the boundary be disowned.",
                req_id,
            )
            return False
        self.pending_loads[req_id] = int(h)
        self.loads_attempted += 1
        return True

    def complete_load(self, req_id) -> None:
        """The bytes landed in the request's slot.

        The hash stays indexed: a load reads LMCache, it does not consume it,
        and the next request over the same prefix must still find it.
        """
        # 【译】字节已经落进请求的槽位。
        #   hash 仍保留在索引里：加载是读 LMCache，不会消耗它，同一前缀上的下一个请求还得能找到它。
        h = self.pending_loads.pop(req_id, None)
        if h is not None:
            self.loads_completed += 1
            # The codec's `get` made this image its youngest; keep the same
            # order so the cap trims what the codec will trim.
            # 【译】codec 的 get 让这份镜像变成了它最年轻的一份；这里保持同样的顺序，让上限修剪掉的正是 codec 将要修剪的。
            if h in self._hash_lru:
                self._hash_lru.move_to_end(h)

    def fail_load(self, req_id) -> None:
        """No bytes came back. Retract the claim as well as counting it.

        A miss is the only evidence that LMCache's LRU dropped what this index
        advertises, so it has to be what un-advertises it. Leaving the hash makes
        every later request over that prefix park, miss and recompute.
        """
        # 【译】没有字节回来。除了计数，还要撤回声明。
        #   miss 是"LMCache 的 LRU 丢掉了本索引宣传的东西"的唯一证据，所以必须由它来撤销宣传。
        #   留着这个 hash，之后该前缀上的每个请求都会停车、miss、再重算。
        h = self.pending_loads.pop(req_id, None)
        if h is None:
            return
        self.loads_failed += 1
        self.forget(h)

    def abandon_load(self, req_id) -> None:
        """The request went away before its bytes did. Neither outcome.

        Not `fail_load`: an abort says nothing about the bytes, and forgetting
        a loadable hash sends the next request back to a full recompute.
        """
        # 【译】请求在它的字节到达之前就走了。两种结果都不算。
        #   不是 fail_load：中止说明不了字节的任何情况，忘掉一个可加载的 hash 会让下一个请求退回到完整重算。
        # 调用方：BlockManager.cancel_state_load / abandon_state_load / deallocate（请求被释放而加载还在途时）
        self.pending_loads.pop(req_id, None)

    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Read the load counters together: `failed / attempted` is this index's
        false-positive rate, and `attempted - completed - failed` is what is in
        flight or was abandoned by an aborted request. Read the store counters
        the same way, and read `stores_completed` against `checkpoints_kept`:
        the gap is how much of what HBM keeps the CPU tier never received.
        """
        # 【译】周期性 "state checkpoints:" 日志行用的计数器。
        #   加载计数要一起看：failed / attempted 是本索引的假阳性率，attempted - completed - failed 是在途的或被中止请求丢下的。
        #   存储计数也这样看，并且拿 stores_completed 和 checkpoints_kept 比：差值就是 HBM 保存了、而 CPU 层从没收到的那部分。
        # 调用方：PagedStateCheckpointCoordinator.checkpoint_fates（加前缀 state_offload_ 合并进快照命运统计）
        return {
            # Store leg. `attempted - completed - failed` is in flight;
            # `refused` is a wiring fault, not backpressure -- any non-zero
            # value means the connector never took the store at all.
            # 【译】存储腿。attempted - completed - failed 是在途的；refused 是接线故障而不是背压——任何非零值都说明 connector 根本没接收存储。
            "stores_attempted": self.stores_attempted,
            "stores_completed": self.stores_completed,
            "stores_failed": self.stores_failed,
            "stores_refused": self.stores_refused,
            "stores_untrusted": self.stores_untrusted,
            # Load leg.
            # 【译】加载腿。
            "loads_attempted": self.loads_attempted,
            "loads_completed": self.loads_completed,
            "loads_failed": self.loads_failed,
            "indexed": len(self.hashes),
            # Hashes dropped at `_hash_cap`, which tracks the worker codec's
            # LRU state-entry bound (`OFFLOAD_STATE_CPU_SIZE`). Each one is an
            # image the CPU tier has most likely dropped too; a value climbing
            # far past `indexed` means the state budget is smaller than the
            # reusable working set.
            # 【译】在 _hash_cap 处被丢掉的 hash，它跟随 worker codec 的 LRU 状态条目上限（OFFLOAD_STATE_CPU_SIZE）。每一个都是一份
            #   CPU 层多半也已经丢掉的镜像；数值远远超过 indexed 说明状态预算比可复用的工作集小。
            "hashes_evicted": self.hashes_evicted,
        }


#: Offload layouts whose worker half builds a `StateOffloadTier`. Not every
#: `lmcache_offload` does: `dense` has no per-request state at all and `hybrid`
#: (DSV4) keeps its own in the SLOT sidecar. `qwen` subclasses the K3 tier.
# 【译】worker 一侧会构建 StateOffloadTier 的 offload 布局。不是每个 lmcache_offload 都会：dense 根本没有每请求状态，
#   hybrid（DSV4）把自己的状态放在 SLOT 旁路里。qwen 继承自 K3 的状态层。
_STATE_TIER_LAYOUTS = frozenset({"kimi_k3", "qwen"})

#: Roles from which the worker will save / load. Mirrors `_do_save` / `_do_load`
#: in `OffloadWorkerMixin`; the state legs ride those same halves.
# 【译】worker 会从哪些角色存 / 取。镜像 OffloadWorkerMixin 里的 _do_save / _do_load；状态的两条腿就搭在这同样的两半上。
# 例：kv_role="offload"（默认）既存又取；"kv_producer" 只存；"kv_consumer" 只取
_SAVE_ROLES = frozenset({"offload", "kv_both", "kv_producer"})
_LOAD_ROLES = frozenset({"offload", "kv_both", "kv_consumer"})


@dataclass(frozen=True)
class StateTierCapability:
    """What the worker half will actually do for the state tier.

    Derived from configuration, not from the connector's public name. The name
    only says which connector class is constructed; whether that class builds a
    `StateOffloadTier` depends on the layout it resolves, the pipeline depth,
    and the role it was given. The engine used to install a `StateOffloadIndex`
    for any `lmcache_offload`, so a K3 run under PP>1 -- or any `dense`/`hybrid`
    run -- got an index against a worker that would never build the tier. That
    mismatch is what left stores emitted with nowhere to go, and it is the
    common root of the tier-none fallback paths.

    `reason` is filled in whenever the tier is off, so the log names which check
    refused it rather than leaving the operator to guess.
    """

    # 【译】worker 一侧对状态层实际会做什么。
    #   由配置推出，而不是由 connector 的公开名字推出。名字只说明构造哪个 connector 类；那个类会不会建 StateOffloadTier，
    #   还取决于它解析出的布局、流水线深度和它被赋予的角色。引擎过去对任何 lmcache_offload 都装一个 StateOffloadIndex，
    #   于是 PP>1 下的 K3——或任何 dense / hybrid 运行——拿到的索引对面是一个永远不会建层的 worker。正是这种不匹配让存储
    #   发出去却无处可去，它也是各种"无层"回退路径的共同根源。
    #
    #   层关闭时总会填上 reason，让日志点名是哪项检查拒绝了它，而不是让运维去猜。

    can_store_state: bool
    can_load_state: bool
    reason: str = ""

    @property
    def hosts_state_tier(self) -> bool:
        """Whether to build the engine-side index at all.

        Either leg is enough to need one: a store-only role still has to hold
        pins and settle reports, and a load-only role still has to vote.
        """
        # 【译】到底要不要建引擎一侧的索引。
        #   任何一条腿就足以需要它：只存的角色仍要持有 pin、了结报告；只取的角色仍要投票。
        return self.can_store_state or self.can_load_state


def _offload_subconfig(cfg: dict) -> tuple[dict | None, str]:
    """The one offload sub-connector under `multi`, or a reason there is none.

    Exactly one, as asked: the tier's bytes ride a specific connector's worker
    half and its completions ride that connector's `get_finished`, so two
    providers would each hold half an answer and a request parked on the wrong
    one would never be reported.

    Two offload sub-connectors (e.g. `[lmcache_offload(dense),
    lmcache_offload(kimi_k3)]`) is refused *loudly* here -- and, for any two
    offload backends, by `_build_subconnectors` when the composite is built --
    rather than degraded to a no-tier fallback. A silent
    fallback disabled the state tier but left the KV load path live: dense won
    `get_num_new_matched_tokens` and queued its H2D, the tier never armed, and
    the prefill forward ran over the block table dense's worker was still
    writing -- torn KV, silent wrong output, and a `finished_loading` the
    scheduler never accounted (review round 5, finding 0). Making it a startup
    `ValueError` keeps that half-configured composite unreachable and names the
    cause where the operator can act on it. A composite with *no* offload sub is
    still a soft no-tier (`None`), the legal `[producer]`-only shape.
    """
    # 【译】multi 下那一个 offload 子 connector，或者说明为什么没有。
    #   恰好一个，如所要求：状态层的字节搭的是某个具体 connector 的 worker 一半，完成报告搭的是那个 connector 的 get_finished，
    #   所以两个提供者会各持半个答案，停在错的那个上的请求永远收不到报告。
    #
    #   两个 offload 子 connector（例如 [lmcache_offload(dense), lmcache_offload(kimi_k3)]）在这里被"大声"拒绝——
    #   而且对任意两个 offload 后端，_build_subconnectors 在构建组合时也会拒绝——而不是降级成无层回退。静默回退关掉了状态层，
    #   却让 KV 加载路径仍然活着：dense 赢下了 get_num_new_matched_tokens 并排上了它的 H2D，状态层从没就绪，
    #   prefill forward 在 dense 的 worker 还在写的 block table 上跑了——KV 被撕裂、输出静默出错，
    #   还有一个调度器从没记账的 finished_loading（review 第 5 轮，finding 0）。把它变成启动时的 ValueError，
    #   让这种半配置的组合不可达，并在运维能动手的地方点名原因。完全没有 offload 子项的组合仍是软性的无层（None），
    #   即合法的仅 [producer] 形态。
    # 输入：cfg = kv_transfer_config（kv_connector == "multi" 时）；输出：(子 connector 的 dict 或 None, 原因字符串)
    # 调用方：state_tier_capability、state_tier_chunk_tokens
    subs = [s for s in (cfg.get("connectors") or ()) if isinstance(s, dict)]
    offload = [s for s in subs if s.get("kv_connector") in _STATE_TIER_BACKENDS]
    if not offload:
        return None, "multi lists no offload connector"
    if len(offload) > 1:
        raise ValueError(
            f"kv_transfer_config: `multi` lists {len(offload)} offload "
            "connectors, but the state tier's bytes and completions ride one "
            "connector's worker half -- two would each hold half an answer and "
            "a request parked on the wrong one would never be reported. Use at "
            "most one offload sub-connector."
        )
    return offload[0], ""


class _SubConnectorView:
    """`config` with one `multi` sub-connector's transfer config in front.

    Attribute lookups fall through to the real config, so the model fields the
    layout selector reads are unchanged; only `kv_transfer_config` is replaced.
    """

    # 【译】把 multi 的某个子 connector 的传输配置放在最前面的 config 视图。
    #   属性查找会落到真实的 config 上，所以布局选择器读的模型字段不变；只有 kv_transfer_config 被替换。

    def __init__(self, config, sub: dict) -> None:
        self._config = config
        self.kv_transfer_config = sub

    def __getattr__(self, name):
        # __getattr__ only fires for names normal lookup misses, so if `_config`
        # itself is unbound -- a pickle/deepcopy probe, or a raise between
        # `__new__` and the assignment in `__init__` -- reading `self._config`
        # here re-enters __getattr__ for "_config" and recurses to
        # RecursionError. This view is built at `BlockManager.__init__`, so that
        # blast radius is engine startup. Fail as an ordinary missing attribute.
        # 【译】__getattr__ 只在正常查找找不到名字时才触发，所以如果 _config 本身还没绑定——pickle / deepcopy 的探测，
        #   或者在 __new__ 和 __init__ 里赋值之间抛了异常——在这里读 self._config 会为 "_config" 再次进入 __getattr__，
        #   一路递归到 RecursionError。这个视图在 BlockManager.__init__ 时构造，所以影响范围是引擎启动。按普通的属性缺失来失败。
        if name == "_config":
            raise AttributeError(name)
        return getattr(self._config, name)


def state_tier_capability(config) -> StateTierCapability:
    """What the worker will do for the state tier, decided from config alone.

    A capability descriptor rather than a name check, because the two are not
    the same question and the worker refuses on conditions the name cannot see
    (no transfer config, a `multi` composite with no offload sub, `pp_size > 1`,
    an unknown explicit override, a layout that offloads no state, or a `kv_role`
    that neither saves nor loads). Config alone because this runs in the engine
    process at
    `BlockManager.__init__`, before any worker connector exists -- so it must
    agree with the worker by construction, which is why the layout comes from
    the same `select_offload_layout` the worker uses and the roles from the
    same sets `OffloadWorkerMixin` reads.

    A false negative (tier off) is much cheaper than a false positive: off
    costs reuse, on-against-a-worker-that-refuses parks requests against loads
    nobody can report.
    """
    # 【译】worker 会为状态层做什么，只凭配置决定。
    #   用能力描述而不是名字检查，因为两者不是同一个问题，worker 还会在名字看不到的条件下拒绝（没有传输配置、multi 组合里没有
    #   offload 子项、pp_size > 1、未知的显式覆盖、一个不 offload 状态的布局，或者一个既不存也不取的 kv_role）。
    #   只凭配置，是因为它在引擎进程的 BlockManager.__init__ 里运行，那时任何 worker connector 都还不存在——所以它必须在构造上
    #   就与 worker 一致，这就是为什么布局来自 worker 用的同一个 select_offload_layout，角色来自 OffloadWorkerMixin 读的同一组集合。
    #
    #   假阴性（层关闭）比假阳性便宜得多：关闭只损失复用；对着一个会拒绝的 worker 打开，会让请求对着没人能报告的加载停车。
    # 判定顺序（任何一步不过就返回带 reason 的"关闭"）：
    #   有 kv_transfer_config 且为 dict → multi 时取唯一的 offload 子项 → connector 是 lmcache_offload → PP=1
    #   → select_offload_layout 解析出 kimi_k3 / qwen → kv_role 能存或能取 → StateTierCapability(can_store, can_load)
    # 调用方：BlockManager.__init__（结果存为 self.state_tier_capability）
    # dataclasses.replace(obj, reason=...)：复制一个 frozen dataclass 并改掉指定字段
    from atom.kv_transfer.offload.config import select_offload_layout

    none = StateTierCapability(False, False, "")
    cfg = getattr(config, "kv_transfer_config", None) or {}
    if not isinstance(cfg, dict):
        return replace(none, reason="no kv_transfer_config")

    name = cfg.get("kv_connector", "moriio")
    layout_config = config
    if name == "multi":
        sub, why = _offload_subconfig(cfg)
        if sub is None:
            return replace(none, reason=why)
        cfg = sub
        name = cfg.get("kv_connector")
        # The layout selector reads `config.kv_transfer_config`, and under
        # `multi` the interesting one is the sub-connector's -- an
        # `offload_layout` override lives there, not on the composite. Model
        # fields still come from the real config.
        # 【译】布局选择器读的是 config.kv_transfer_config，而在 multi 下有意义的是子 connector 的那份——offload_layout 覆盖写在那里，
        #   不在组合上。模型字段仍然来自真实的 config。
        layout_config = _SubConnectorView(config, cfg)
    if name not in _STATE_TIER_BACKENDS:
        return replace(none, reason=f"connector {name!r} hosts no state tier")

    # The worker refuses PP outright: `CacheEngineKey` has no PP component, so
    # two stages at the same TP rank would overwrite each other's entries.
    # 【译】worker 直接拒绝 PP：CacheEngineKey 没有 PP 分量，所以同一个 TP rank 上的两个 stage 会互相覆盖对方的条目。
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if pp_size > 1:
        return replace(none, reason=f"pipeline_parallel_size={pp_size}")

    try:
        layout = select_offload_layout(layout_config)
    except ValueError as exc:  # an unknown explicit override
        # 【译】（行尾）一个未知的显式覆盖
        return replace(none, reason=str(exc))
    if layout not in _STATE_TIER_LAYOUTS:
        return replace(none, reason=f"layout {layout!r} offloads no state")

    role = cfg.get("kv_role", "offload")
    can_store = role in _SAVE_ROLES
    can_load = role in _LOAD_ROLES
    if not (can_store or can_load):
        return replace(none, reason=f"kv_role {role!r} neither saves nor loads")
    return StateTierCapability(can_store, can_load)


def state_tier_cpu_bytes() -> int:
    """CPU bytes the state tier may keep per worker (`OFFLOAD_STATE_CPU_SIZE`,
    GiB, default 32), carved from the paged-KV LMCache pool it shares.

    One reader for both halves: the worker's `StateByteCodec` evicts against
    it and the engine's `StateOffloadIndex` mirrors the resulting entry bound.
    """
    # 【译】状态层每个 worker 可以占用的 CPU 字节数（OFFLOAD_STATE_CPU_SIZE，单位 GiB，默认 32），从它共用的分页 KV LMCache 池里划出来。
    #   两半共用一个读取方：worker 的 StateByteCodec 按它淘汰，引擎的 StateOffloadIndex 镜像由此得出的条目上限。
    # 调用方：BlockManager.__init__（算 max_cpu_entries）、kv_transfer/offload/hybrid/kimi_k3/connector.py（max_cache_bytes）
    return int(float(os.environ.get("OFFLOAD_STATE_CPU_SIZE", "32")) * (1 << 30))


def state_tier_chunk_tokens(config) -> int:
    """The LMCache chunk size in tokens for the joint KV leg, or 0.

    Decided from config for the same reason `state_tier_capability` is: this
    runs in the engine process at `BlockManager.__init__`, and the scheduler
    holds whatever `get_kvconnector` returned -- a connector object without
    `chunk_size` would zero this and disable the joint KV load silently. The KV
    leg moves whole LMCache chunks, so a joint boundary has to be a multiple of
    this.

    Kept out of `state_tier_capability` on purpose: that check is pure and has a
    second caller (`hosts_state_tier`), and this reaches into LMCache -- an
    optional dependency whose absence must not warn on an engine that hosts no
    tier. Call this only when the capability says a tier is hosted; the WARNING
    below is then worth printing, because a missing chunk size really does
    disable the joint KV load.
    """
    # 【译】联合 KV 腿用的 LMCache chunk 大小（以 token 计），或 0。
    #   和 state_tier_capability 一样由配置决定，理由相同：它在引擎进程的 BlockManager.__init__ 里运行，而调度器持有的是
    #   get_kvconnector 返回的随便什么东西——一个没有 chunk_size 的 connector 对象会把它变成 0，静默关掉联合 KV 加载。
    #   KV 腿搬的是整块 LMCache chunk，所以联合边界必须是它的整数倍。
    #
    #   故意不放进 state_tier_capability：那个检查是纯的，并且有第二个调用方（hosts_state_tier），而这个要伸进 LMCache——
    #   一个可选依赖，在不承载状态层的引擎上它的缺失不应该报警。只在能力判定说承载了状态层时才调用它；
    #   那时下面的 WARNING 才值得打印，因为缺 chunk 大小确实会关掉联合 KV 加载。
    # 调用方：BlockManager.__init__（hosts_state_tier 为真时，结果存为 _joint_chunk_tokens，供 _joint_kv_boundary 对齐 chunk）
    # 例：chunk_size=256，状态快照在 B=8192 → KV 腿要取覆盖 [hbm, 8192) 的整块 chunk；B 不在 chunk 网格上时取覆盖它的那块（多取一块，绝不少取）
    try:
        from atom.kv_transfer.offload.config import build_lmcache_config

        kvcfg = getattr(config, "kv_transfer_config", None) or {}
        # Under `multi` the lmcache.* keys (chunk_size, offload_layout) live on
        # the offload SUB-connector, not the composite. Passing the raw
        # composite here read a zero chunk grid. Unwrap the sub the same way
        # `state_tier_capability` does, so the gate's chunk size and the tier
        # the capability check builds stay consistent.
        # 【译】在 multi 下，lmcache.* 键（chunk_size、offload_layout）在 offload 子 connector 上，不在组合上。直接传入原始组合
        #   读到的是一个为 0 的 chunk 网格。用和 state_tier_capability 一样的方式解开子项，让门控的 chunk 大小和能力检查构建的层保持一致。
        if isinstance(kvcfg, dict) and kvcfg.get("kv_connector") == "multi":
            sub, _why = _offload_subconfig(kvcfg)
            if sub is not None:
                kvcfg = sub
        return int(build_lmcache_config(kvcfg).chunk_size)
    except Exception:
        # Blind on purpose: this runs at model load, the import reaches a
        # third-party package that may be absent entirely, and the only
        # consequence of not knowing the chunk size is that the joint KV load
        # stays off. Refusing to start would be worse.
        # 【译】故意盲捕所有异常：它在模型加载时运行，import 伸向一个可能完全不存在的第三方包，而不知道 chunk 大小的唯一后果是
        #   联合 KV 加载保持关闭。拒绝启动会更糟。
        logger.warning(
            "state offload: could not read the LMCache chunk size; the joint "
            "KV load needs it and stays off",
            exc_info=True,
        )
        return 0


#: The connector backends whose worker half can build a `StateOffloadTier`.
#: `multi` qualifies when it lists exactly one.
# 【译】worker 一侧能构建 StateOffloadTier 的 connector 后端。multi 在恰好列出一个时也算。
# 解释：定义在文件末尾也没关系——它只在上面函数的函数体里被读取，调用时模块早已加载完
_STATE_TIER_BACKENDS = frozenset({"lmcache_offload"})
