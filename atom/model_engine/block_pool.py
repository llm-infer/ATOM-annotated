# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】block_pool.py：分页 KV block 的池子——空闲表、引用计数、按内容寻址的 hash 索引、惰性淘汰
#   位置：Scheduler ──> BlockManager（block_manager.py：命中多少、给谁分配、何时释放，以及状态 checkpoint 等分支）
#                         └─ self.kv = BlockPool(num_kvcache_blocks, on_evict=…, on_remap=…)   ← 本文件，进程里只有这一个实例
#                               └─ blocks[i] = Block(i)（kv_block.py：block_id / ref_count / hash / token_ids）
#   只记账、不碰显存（同 nano-vllm）：GPU 上的 KV 张量由 ModelRunner 按 block 号分页；这里决定"哪个号空闲、被几个人用、装的什么内容"。
#   前置：分页 KV 与 block table → sequence.py BlockTable类上方"科普：KV cache分块"；
#         前缀 hash（hash 了什么、为什么链式、DCP 下块为什么变大）→ prefix_hash.py 文件头【科普】①~④
#
# 阅读路线
#   ① 下面的"账本图"和"几条请求带数字走一遍"——本文件的核心，先看
#   ② BlockPool 类 docstring（空闲 block 为什么分成两种容器）和 __init__ 的成员
#   ③ 主干方法，按 BlockManager 调用的顺序：查（lookup / index_get / count_used）→ 命中复用（claim_many / claim）
#      → 新分配（pop + allocate，淘汰发生在这里）→ 登记 hash（publish）→ 释放（free，不清内容）
#   ④ 【分支】SLRU 淘汰策略、raw unit（状态 checkpoint 借用 KV block，第 4 批 page_unit_checkpoint.py）、
#      扩缩池（extend / retire_top）、clear_index（清空前缀缓存）、eviction_stats（统计）
#
# ---------------------------------------------------------------------------------------------------------
# 账本图（6 个 block；画的是下面例子第 4 步之后的时刻）
#
#   blocks[0..5]：每个物理 block 一条 Block 记录，此刻 ref_count = [1, 1, 0, 1, 0, 0]
#
#   _used = {0, 1, 3}             正在被序列引用（ref_count ≥ 1）——永远不是淘汰候选
#   _free = {2, 4, 5}             ref_count = 0。_free 和 _used 不相交，合起来正好是 0 ~ num_blocks−1；
#    │                            "是否空闲"只看这个集合，下面三个容器只决定"先给出谁"
#    ├─ _vacant    = 堆[4, 5]     空闲、没有 hash：拿走它什么都不损失 → 最先给出，id 小的先（最小堆）
#    ├─ _cached    = [2]          空闲、hash 还在索引里（内容仍可被前缀命中）：按释放先后排队（OrderedDict，左端最早）
#    │                            → _vacant 用完才动它，先给出最早释放的 = 淘汰最冷的内容
#    └─ _protected = []           仅 ATOM_PREFIX_CACHE_POLICY=slru：被命中复用过的空闲块，最后才动（默认 lru 时一直为空）
#
#   _hash_to_block_id = {h0: 0, h1: 1, h2: 2, h2': 3}     内容寻址索引：链式 hash → 物理 block 号
#     2 号已经空闲，h2 却还在——这就是"惰性淘汰"：释放不删索引，等到 2 号被分给别的内容（allocate）的那一刻才删
#
# block 在容器之间怎么走（箭头上是 BlockPool 的方法）：
#
#                  ┌───── claim / claim_many：前缀命中，ref 0→1，内容原样保留 ─────┐
#                  │                                                            ▼
#   ┌──────────────────────────┐  pop + allocate（_vacant 空了才轮到）  ┌──────────────────────────┐
#   │ _cached：空闲、有 hash     │ ──── 删 hash = 淘汰 ──────────────────> │ _used：ref_count ≥ 1      │──┐ claim：已在用的块又被命中
#   └──────────────────────────┘ <──── free：ref 减到 0、有 hash ──────── │ publish：写满后登记 hash   │<─┘ → 只 ref+1（共享）
#   ┌──────────────────────────┐  pop + allocate（优先）               │                          │
#   │ _vacant：空闲、无 hash     │ ────────────────────────────────────> │                          │
#   └──────────────────────────┘ <──── free：ref 减到 0、无 hash ──────── └──────────────────────────┘
#
# ---------------------------------------------------------------------------------------------------------
# 几条请求带数字走一遍（block_size = 4，6 个 block，默认 lru；每一步都对照 BlockManager 实际调用的方法推算过）
#   （这里只看池子内部的容器怎么变；BlockManager 一侧从准入、decode 到释放、淘汰的全程见 block_manager.py 文件头的 100 块例子，
#    准入时命中块正被占用 / 在空闲表里两种情况的对比见 BlockManager.can_allocate 里的 20 块例子）
#   A = 12 个 token [a0 … a11]；B 与 A 共享前 8 个：[a0 … a7, b8 … b11]；C 与谁都不同：[c0 … c11]
#   h0 = H(a0..a3)，h1 = H(h0, a4..a7)，h2 = H(h1, a8..a11)，h2' = H(h1, b8..b11)
#   （H 即 prefix_hash.block_hash；纯文本请求的种子是 -1。为了好读，下面假设请求在 prefill 那一步就结束，不追加 decode 块）
#
#   0 启动：BlockPool(6)
#        ref=[0 0 0 0 0 0]   _vacant={0..5}   _cached=[]          索引={}
#   1 A 准入。can_allocate：查 h0 未命中 → 需要 3 块，空闲 6 块，够；allocate：3 次 _fresh_block（= pop + allocate），
#     pop 从 _vacant 堆顶依次给出 0、1、2 → A.block_table = [0, 1, 2]
#        ref=[1 1 1 0 0 0]   _vacant={3,4,5}  _cached=[]          索引={}
#   2 A 的 prefill 算完（postprocess → hash_blocks(A, 12)）：0~2 块都写满 → publish(0,h0)、publish(1,h1)、publish(2,h2)
#        ref=[1 1 1 0 0 0]   _vacant={3,4,5}  _cached=[]          索引={h0:0, h1:1, h2:2}
#   3 A 结束 → deallocate：逆序 free(2)、free(1)、free(0)，ref 1→0，都有 hash → 依次进 _cached；
#     索引一条不删，显存里的 KV 也不动
#        ref=[0 0 0 0 0 0]   _vacant={3,4,5}  _cached=[2,1,0]     索引={h0:0, h1:1, h2:2}   ← 不变
#   4 B 准入。can_allocate → _hit_walk：只查前 ⌈12/4⌉ − 1 = 2 块（最后一块永远不复用：prefill 至少要算一块才有 logits），
#     h0→0、h1→1 都在索引里，再比对 1 号的 token_ids 与 B 的 a4..a7 相同 → 命中 2 块。
#     需要的空闲块 = 3 − count_used([0,1]) = 3 − 0 = 3：命中的块不用重算，但它们此刻在空闲表里，拿回来也占空闲名额。
#     allocate：claim_many([0,1]) → ref 0→1，移出 _free 和 _cached（OrderedDict.pop，O(1)）；
#     剩下 1 块走 pop → 堆顶 3。B.block_table = [0, 1, 3]，num_cached_tokens = 8，prefill 只算位置 8~11
#        ref=[1 1 0 1 0 0]   _vacant={4,5}    _cached=[2]         索引={h0:0, h1:1, h2:2}
#     B 的 prefill 算完：hash_blocks(B, 4) 只有第 2 块新写满，父 hash 读 1 号的 h1 → publish(3, h2')，索引加上 h2':3
#     （上面"账本图"画的就是这个时刻）
#   5 B 结束 → 逆序 free(3)、free(1)、free(0)，依次进 _cached 的右端
#        ref=[0 0 0 0 0 0]   _vacant={4,5}    _cached=[2,3,1,0]   索引={h0:0, h1:1, h2:2, h2':3}
#   6 C 准入，没有命中 → 3 次 pop：4、5（先把 _vacant 用完）；第 3 次 _vacant 已空 → 取 _cached 最左端 = 2（最早释放的）。
#     allocate(2) 里 _unindex(2)：删掉 h2、index_version + 1、on_evict(h2) → BlockManager._record_evicted
#     （total_evicted_blocks + 1；开了 KV events 时记一条 BlockRemoved；有状态 checkpoint 的模型作废挂在 h2 上的 checkpoint），
#     blocks_evicted + 1；reset → ref=1、hash=-1。C.block_table = [4, 5, 2]
#        ref=[0 0 1 0 1 1]   _vacant={}       _cached=[3,1,0]     索引={h0:0, h1:1, h2':3}
#     h2 在第 3 步就"空闲"了，却直到这一步才真正失效：第 3~5 步之间来一条以 a0..a11 开头的更长请求，还能命中它
#   变体：B 若在第 2 步之后、A 还没结束时到来——0、1 号在 _used 里，claim 只把 ref 1→2（共享）；count_used = 2，
#     B 只需从空闲表拿 3 − 2 = 1 块
#
# 为什么释放的 block 要保留内容（惰性淘汰）
#   KV 已经躺在显存里，"释放"只表示没人在用了。留着它的 hash，后来的请求（同一个 system prompt 的下一个请求、
#   多轮对话的下一轮 = 上一轮 prompt + 回答）还能零成本复用；马上清掉什么也省不下——反正要等有新内容要写时才需要这块地方。
#   所以真正的淘汰发生在 allocate（这个号要装别的东西了），而不是 free。
#   一个 block 的 hash 和内容到底什么时候失效：
#     ① 它被 pop 出来分给新内容（allocate → _unindex），即上例第 6 步；
#     ② clear_index（BlockManager.clear_cache；docstring 说供 /reset_prefix_cache 一类的管理接口用，但 v0.1.8 的 atom/ 里
#        没有调用方，只有测试调用，见那里的"观察"）；
#     ③ 同一个 hash 被登记到了另一个块上（publish 改指，见 publish 里的例子），旧块从此找不到；
#     ④ 【分支】池子缩小、顶部的块被收回（retire_top）。
#   ref_count ≥ 1 的块永远不会被淘汰。
#
# 淘汰顺序（_take_free）：_vacant（id 最小的先）→ _cached（最早释放的先）→ _protected（仅 slru，最早进入的先）
#   _cached 按"最近一次释放"排队：被 claim 拿回去的块会从队列里删掉，下次释放时排到最右端——所以这是按最近使用时间的 LRU，
#     不是按第一次入队的 FIFO（这就是它用 OrderedDict 而不用 deque 的原因，见类 docstring）。
#   再加上 BlockManager.deallocate 是逆序释放：一条请求越靠后的块越早进队、越早被淘汰，开头那几块（最常被共享的 system prompt）
#     活得最久——上例第 6 步淘汰的是 A 的最后一块 2，而不是 0、1（同 nano-vllm 的逆序释放）
#
# 代码断言或维护的不变式
#   · _free ∩ _used = ∅，_free ∪ _used = {0 … num_blocks−1}：没有显式断言，由各方法成对增删来维护
#   · 空闲块 ref_count = 0：claim / claim_many / allocate 拿空闲块时 assert ref_count == 0
#   · 在用块 ref_count ≥ 1：free 只在减到 0 时才把它移出 _used
#   · 索引条目 h → b 一定满足 blocks[b].hash == h：publish 先撤旧再登新，_unindex 只删指向自己的条目，_adopt 连 hash 一起搬
#   · Block.token_ids 必须是 array('i')（Block.update 断言）
#   · raw unit 不能用 free 释放，只能整份按所有者归还（free / release_units / rekey_units 里的检查）
#   · 没有空闲块时 pop 抛 AssertionError（上游的准入检查保证到不了这里）
#
# 对比nano-vllm（BlockManager 约 240 行，账本和策略写在一起；ATOM 把账本拆到本文件，策略留在 3000+ 行的 BlockManager）：
#   nano-vllm 的做法                                  ATOM BlockPool 的做法
#   free_block_ids：一个 deque，左取右还                 _vacant（最小堆）+ _cached（OrderedDict）[+ _protected]
#     → 只按释放先后：队头若是有内容的块就先淘汰它，        → 先用没装过内容的块，空块用完才淘汰最旧的有内容块
#       哪怕后面排着从没装过东西的空块
#   命中空闲块：free_block_ids.remove(id)，O(n)          _take_named：OrderedDict.pop，O(1)
#   used_block_ids / hash_to_block_id                  同（_used / _hash_to_block_id），另有 _free 集合作为"是否空闲"的唯一判据
#   _allocate_block 里删 hash（同样是惰性淘汰）           allocate → _unindex，并回调 on_evict、计数 blocks_evicted
#   —                                                  publish 撤销旧 hash；on_remap / index_version 让 BlockManager 知道它缓存的
#                                                        前缀查找结果（准入时的 walk、can_allocate_memoized）何时失效
#   —                                                  SLRU、raw unit（状态 checkpoint 借 KV block 存放）、extend / retire_top、
#                                                        eviction_stats（区分"池太小"和"划分不对"）
#
# 两种淘汰策略的适用场景与代价（ATOM_PREFIX_CACHE_POLICY，见 envs.py；默认 lru）
#   lru ：只看最近一次释放的时间。简单；但一串一次性的长 prompt（每条都不会再被命中）会把热门前缀挤出去
#   slru：被命中复用过的块释放时进受保护段 _protected，新内容先待在试用段 _cached；受保护段最多占
#         num_blocks × ATOM_PREFIX_CACHE_PROTECTED_RATIO（默认 0.5，6 块时 3 块），超出时最旧的降回 _cached 的右端。
#         适合"少数热门前缀 + 大量一次性请求"混在一起的流量；代价是新前缀只能在剩下的空间里竞争，热点换了要等降级
import array
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import islice

from atom.model_engine.kv_block import Block


# 【分支】扩缩池：retire_top 收回最高号 block 的结果（见文件末尾的 resizing 一节）
#   Python 语法：@dataclass(frozen=True) 自动生成 __init__ 等方法，并且实例创建后字段不可修改
@dataclass(frozen=True)
class BlockRetirement:
    """What taking the highest block out of the pool cost.

    `moved_to` is -1 when the block was free and could simply be dropped.
    Otherwise it is where the block's contents now live, and every holder's
    block table has to follow — which is why this is reported rather than
    handled here: the pool does not know who holds what.
    """

    # 【译】把最高号的 block 从池子里拿走付出了什么。
    #
    #   块原本空闲、可以直接丢掉时，moved_to 为 -1。否则它是这个块的内容现在所在的位置，每个持有者的
    #   block table 都得跟着改——这就是为什么这里只报告而不处理：池子不知道谁持有什么。
    retired: int
    moved_to: int


class BlockPool:
    """Paged blocks with ref counts and a content-addressed index.

    `BlockManager` owns the one instance, over the compressed KV blocks. This
    was split out when a second pool existed — the sliding window was its own
    content-addressed block pool, driven in lockstep with this one — and is
    kept separate because the class it serves is a whole mechanism (free list,
    hash index, lazy eviction) rather than a helper of the manager. The window
    is a per-request ring sharing a slot with the compressor state now; see
    `v4_pool_geometry.py`.

    Eviction is lazy: a freed block keeps its hash and contents until its slot
    is handed out for something else, so a later request can still claim a
    freed-but-not-overwritten block. `allocate` — not `free` — is therefore the
    eviction event, and `on_evict` fires there.

    Free blocks sit in one of two containers, by whether they still hold
    reusable content:

      `_vacant`    no hash. Nothing is lost by taking one, so they go first,
                   lowest id first — which also drains the top of the pool,
                   where a shrinking boundary eats.
      `_cached`    still reachable by hash. Least-recently-freed first, so
                   handing one out evicts the coldest content.

    One queue for both would evict a cached block while a vacant one waited
    behind it, purely on release order.

    `_cached` is an insertion-ordered mapping rather than a queue because
    `claim` takes a *named* block off the free list on every prefix hit: left
    in place, its entry would put the block back at its old position when it is
    freed again, which is the LRU order inverted for exactly the blocks being
    reused most. Removing it costs O(1) here and O(n) from a deque, on a path
    that runs once per hit block.

    Optional ``slru`` puts blocks claimed for reuse into a bounded protected
    queue on release. New content remains probationary until claimed, so a
    stream of one-off prefixes cannot evict the whole hot set. Vacant blocks
    still go first; referenced blocks are never candidates in either policy.
    """

    # 【译】带引用计数和内容寻址索引的分页 block。
    #
    #   BlockManager 持有唯一的实例，管理压缩后的 KV block。这个类是在存在第二个池的时候拆出来的——滑动窗口曾有
    #   自己的内容寻址 block 池，与这个池同步驱动——之所以仍然独立，是因为它服务的是一整套机制（空闲表、hash 索引、
    #   惰性淘汰），而不是 manager 的一个辅助函数。现在滑动窗口是每请求的环形缓冲区，与压缩器状态共用一个槽位；
    #   见 v4_pool_geometry.py。
    #
    #   淘汰是惰性的：释放的 block 保留它的 hash 和内容，直到它的位置被分给别的东西，所以之后的请求仍可以认领一个
    #   已释放但还没被覆盖的 block。因此淘汰事件是 allocate 而不是 free，on_evict 也在那里触发。
    #
    #   空闲 block 按是否还持有可复用的内容，放在两个容器之一：
    #
    #     _vacant    没有 hash。拿走一个什么也不损失，所以它们先给出，id 小的先——这也顺带让池子顶部空出来，
    #                那里是收缩的边界会吃掉的地方。
    #     _cached    仍可通过 hash 找到。最早释放的先给出，所以给出一个就是淘汰最冷的内容。
    #
    #   两者共用一个队列的话，会仅仅因为释放顺序就淘汰一个有缓存的 block，而一个空的 block 还在它后面等着。
    #
    #   _cached 用按插入顺序排列的映射而不是队列，是因为每次前缀命中时 claim 都会把一个"指名的"block 从空闲表拿走：
    #   如果把它的条目留在原处，等它再次被释放时，这个条目会让它回到旧位置——对恰恰被复用得最多的那些 block 来说，
    #   这就把 LRU 顺序弄反了。在这里删除是 O(1)，从 deque 里删是 O(n)，而这条路径每个命中块都要走一次。
    #
    #   可选的 slru 让为复用而被认领过的 block 在释放时进入一个有界的受保护队列。新内容在被认领之前一直处于试用期，
    #   所以一串一次性的前缀不能把整个热点集合淘汰掉。空块仍然最先给出；两种策略下，被引用的 block 都永远不是候选。
    # 解释："压缩后的 KV block"是 DeepSeek-V4 的叫法（它的分页池存压缩后的 KV 行），对普通 MHA / GQA / MLA 模型就是唯一的分页 KV 池。
    #   "收缩的边界"：分页 KV 池和状态池可以共用一块固定显存（"平面"），边界下移时池子从最高号开始交出 block（retire_top）；
    #   小 id 先给，使用中的块就集中在低号，顶部多半空着，收回时不用搬数据。

    # 构造：BlockManager.__init__ 里 BlockPool(num_blocks, on_evict=self._record_evicted,
    #   cache_policy=envs.ATOM_PREFIX_CACHE_POLICY, protected_ratio=envs.ATOM_PREFIX_CACHE_PROTECTED_RATIO,
    #   on_remap=self._note_index_change)，num_blocks = config.num_kvcache_blocks（EngineCore 按显存测出后回填）；不传 max_blocks
    #   on_evict(h)：一个可复用的 hash 被销毁（淘汰，或 publish 撤销旧 hash）时回调
    #   on_remap(h)：一个 hash 的索引条目改指到另一个 block 时回调
    #   【分支】状态 checkpoint 的 PagedStateCheckpointCoordinator（page_unit_checkpoint.py）拿到的也是这同一个实例 self.kv；
    #   只有测试（tests/test_block_pool.py 等）直接构造 BlockPool(n)
    def __init__(
        self,
        num_blocks: int,
        on_evict: Callable[[int], None] | None = None,
        max_blocks: int | None = None,
        cache_policy: str = "lru",
        protected_ratio: float = 0.5,
        on_remap: Callable[[int], None] | None = None,
    ):
        # 规范化策略名：" SLRU " → "slru"；只认 lru / slru，受保护比例必须在 (0, 1) 之间
        cache_policy = cache_policy.strip().lower()
        if cache_policy not in {"lru", "slru"}:
            raise ValueError(f"unknown prefix cache policy: {cache_policy!r}")
        if not 0 < protected_ratio < 1:
            raise ValueError("protected_ratio must be between 0 and 1")
        self.cache_policy = cache_policy
        self.protected_ratio = protected_ratio
        # 【分支】slru 专用：_protected = 受保护段（被复用过的空闲块，按进入先后排）；
        #   _reused = 当前持有期间被前缀命中认领过的块 id，free 时据此决定进 _protected 还是 _cached
        self._protected: OrderedDict[int, None] = OrderedDict()
        self._reused: set[int] = set()
        # `max_blocks` is how far `extend` may go, and so how many Block
        # objects exist. It is the pool's share of a fixed plane rather than
        # its current size; a pool with a pinned boundary passes neither and
        # gets a maximum equal to its size.
        # 【译】max_blocks 是 extend 最多能扩到多大，因此也是 Block 对象的个数。它是这个池在一个固定平面里的份额，
        #   而不是当前大小；边界固定的池两者都不传，最大值就等于它的大小。
        # 观察：v0.1.8 的 atom/ 里 BlockManager 不传 max_blocks，也没有代码调用 extend / retire_top（只有
        #   tests/test_block_pool.py 等测试用到），所以生产中池大小固定为 num_kvcache_blocks
        self.max_blocks: int = num_blocks if max_blocks is None else max_blocks
        if not 0 <= num_blocks <= self.max_blocks:
            raise ValueError(f"{num_blocks} blocks outside 0..{self.max_blocks}")
        # 当前池里有多少个 block（id 为 0 ~ num_blocks−1）
        self.num_blocks: int = num_blocks
        self._on_evict = on_evict
        # Called with a hash whose index entry moved to another block id
        # (republished elsewhere, `_adopt`): ids resolved before then are stale.
        # 【译】以一个 hash 为参数调用：它的索引条目移到了另一个 block id（在别处重新登记，或 _adopt）：
        #   在那之前解析出的 id 已经过时。
        self._on_remap = on_remap
        # 一次建好全部记录，下标就是 block 号（kv_block.py）
        self.blocks: list[Block] = [Block(i) for i in range(self.max_blocks)]
        # 内容寻址索引：链式 hash → block 号。只在 publish 时增加，在 _unindex / clear_index 时删除；
        #   一个 hash 只指向一个 block（同一内容算出了两份时，指向最后登记的那份，见 publish）
        self._hash_to_block_id: dict[int, int] = {}
        # Bumped when a hash leaves the index or moves to another id (not when
        # one is added): ids resolved at version v still hold while it is v.
        # 【译】当一个 hash 离开索引、或移到另一个 id 时加 1（新增时不加）：在版本 v 时解析出的 id，只要版本还是 v 就仍然有效。
        # 用途：BlockManager._take_admission_walk 判断准入探测时查到的那串 id 能否直接拿来 claim（版本没变就不用重查）。
        #   新增不加 1，是因为新增只会让命中变长，不会让已查到的 id 变错
        self.index_version: int = 0
        # Both containers may hold ids that were re-claimed straight off the
        # free list (`claim`) or retired, so membership in the set — not
        # presence in a container — is what makes an id free. `pop` skips the
        # stale entries.
        # 【译】两个容器里都可能有已经被直接从空闲表认领走（claim）或被收回的 id，所以让一个 id 成为"空闲"的是
        #   它在 _free 集合里，而不是它出现在某个容器里。pop 会跳过这些陈旧条目。
        # Python 语法：heapq 在普通 list 上维护最小堆（heappop 总是取出最小值，O(log n)）；有序的 list 本身就满足堆的性质，
        #   所以 list(range(n)) 不用再 heapify。OrderedDict[int, None] 当"有序集合"用，值没有意义：
        #   popitem(last=False) 弹出最早插入的那个，pop(key) 以 O(1) 删掉任意一个
        self._vacant: list[int] = list(range(num_blocks))
        self._cached: OrderedDict[int, None] = OrderedDict()
        self._free: set[int] = set(range(num_blocks))
        self._used: set[int] = set()
        # Raw PAGE units reserved by multi-unit objects such as state checkpoints.
        # 【译】被多单元对象（例如状态 checkpoint）预留的原始 PAGE 单元。
        # 【分支】状态 checkpoint 存进分页池（page_unit_checkpoint.py，第 4 批 4b）：一份递归状态镜像要占若干个
        #   KV block 大小的"单元"。这里记 单元号 → (所有者, 第几片)；这些块不进 hash 索引，也不能用 free 释放（要用 release_units）
        self._raw_unit_owner: dict[int, tuple[Hashable, int]] = {}
        # Reusable content this pool destroyed, split by what destroyed it.
        # Both are evictions in the sense that a later prefix hit is now
        # impossible, and they read the same in a hit rate, but they want
        # opposite fixes — the same reason `StateSlotPool` keeps `evicted`
        # and `orphaned` apart:
        #   `blocks_evicted`  the pool was out of vacant blocks and spent a
        #                     cached one. Says the paged pool is too small.
        #   `blocks_retired`  the boundary moved down over cached content.
        #                     Says the split is wrong, not the total.
        # Counted here rather than derived from `on_evict` because that hook
        # also fires for relocation (`_adopt`), which destroys nothing: the
        # hash moves to the block that adopted it.
        # 【译】这个池销毁掉的可复用内容，按销毁的原因分开计数。两者都是淘汰——之后不可能再命中，在命中率里看起来也一样——
        #   但它们需要相反的修复，这和 StateSlotPool 把 evicted 与 orphaned 分开是同一个原因：
        #     blocks_evicted  池子没有空块了，花掉了一个有缓存的块。说明分页池太小。
        #     blocks_retired  边界下移，盖过了有缓存的内容。说明划分不对，而不是总量不对。
        #   在这里计数而不是从 on_evict 推出来，因为那个钩子在搬迁（_adopt）时也会触发，而搬迁什么也没销毁：
        #   hash 移到了接收它的那个 block。
        # 观察：当前代码里被搬走的 hash 触发的是 on_remap 而不是 on_evict（_adopt 只在目标块原有的内容被销毁时经 _unindex
        #   触发 on_evict，那属于真正的销毁，计入 blocks_retired）；不过 publish 撤销旧 hash 时也会触发 on_evict，
        #   那不是容量压力，所以单独计数仍然必要
        self.blocks_evicted: int = 0
        self.blocks_retired: int = 0

    # ------------------------------- counts -------------------------------- #
    # 【译】------------------------------- 计数 -------------------------------- #
    # num_free 包括 _cached / _protected 里可淘汰的块：惰性淘汰下它们和空块一样能立刻分出去，
    #   所以准入判断 has_free(n)（BlockManager._has_page_units）把它们算作可用。上例第 5 步 num_free = 6
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    @property
    def num_indexed(self) -> int:
        """Blocks reachable by content hash, live or merely not-yet-overwritten."""
        # 【译】能通过内容 hash 找到的 block 数，不论是正在用的，还是仅仅还没被覆盖的。
        return len(self._hash_to_block_id)

    def has_free(self, n: int) -> bool:
        return len(self._free) >= n

    def is_used(self, block_id: int) -> bool:
        return block_id in self._used

    def block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    @property
    def num_reusable_free(self) -> int:
        """Free blocks still holding content a prefix hit could claim.

        The pool's headroom before the *next* allocation has to evict: while
        vacant blocks remain this is slack, and once they are gone every
        allocation spends one of these. `num_free - num_reusable_free` is the
        vacant count, which is the number that actually has to reach zero
        before `blocks_evicted` can start moving.
        """
        # 【译】仍然装着前缀命中可以认领的内容的空闲 block 数。
        #
        #   这是池子在"下一次分配必须淘汰"之前的余量：只要还有空块，这就是富余；空块用完后，每次分配都会花掉其中一个。
        #   num_free − num_reusable_free 是空块数，它才是 blocks_evicted 开始增长之前真正要降到 0 的那个数。
        # 例：上例第 5 步 num_free = 6、num_reusable_free = 4（2、3、1、0）→ 空块 2 个，所以第 6 步第 3 次分配才开始淘汰。
        #   调用方：eviction_stats；engine_utility.py 的统计（kv_blocks_evictable，kv_blocks_vacant = num_free − 它）
        return len(self._cached) + len(self._protected)

    def eviction_stats(self) -> dict[str, int]:
        """Content this pool destroyed, and the headroom it has left.

        Counters, not rates, for the same reason `EngineStats.cache_statistics`
        hands back counts: a rate cannot be summed across DP ranks that saw
        different traffic.
        """
        # 【译】这个池销毁掉的内容，以及它剩下的余量。
        #
        #   给的是计数而不是比率，原因和 EngineStats.cache_statistics 返回计数相同：比率不能在流量不同的 DP rank 之间相加。
        # 例：rank0 分配 100 次淘汰 10 次、rank1 分配 1000 次淘汰 10 次，比率 10% 和 1% 平均成 5.5% 是错的，
        #   计数相加才对。调用方：BlockManager.pool_pressure（周期性的统计行）
        return {
            "blocks_evicted": self.blocks_evicted,
            "blocks_retired": self.blocks_retired,
            "blocks_total": self.num_blocks,
            "blocks_used": self.num_used,
            "blocks_free": self.num_free,
            "blocks_free_reusable": self.num_reusable_free,
            "blocks_indexed": self.num_indexed,
        }

    # ------------------------------- index --------------------------------- #
    # 【译】------------------------------- 索引 --------------------------------- #
    def lookup(self, h: int) -> int:
        """Block id indexed under content hash `h`, or -1."""
        # 【译】登记在内容 hash h 下的 block id，没有则返回 -1。
        # 例：上例第 4 步 lookup(h0) = 0、lookup(h1) = 1；C 的第 0 块 lookup(H(c0..c3)) = -1。
        #   调用方：BlockManager._walk_prefix（逐块遍历，ATOM_PREFIX_FAST_WALK=0）、allocate（准入探测的结果不能整段直接用时，剩下的块逐个查）、
        #   _memoised_walk（记下第一个未命中的边界块），以及 offload / 远端前缀等【分支】
        return self._hash_to_block_id.get(h, -1)

    @property
    def index_get(self):
        """The index's own `dict.get`: block id or None, for C-speed lookups
        of a whole chain with `map`. Read it per walk, not once per pool."""
        # 【译】索引自己的 dict.get：返回 block id 或 None，用来配合 map 以 C 速度查一整条链。每次遍历读一次，
        #   不要每个池只读一次。
        # 用法：BlockManager._hit_walk 里 list(map(get, 一串 hash))，循环在 C 里跑；结果里第一个 None 就是第一个未命中。
        #   @property 返回的是绑定方法本身。"每次遍历读一次"是防御性的约定：现在只有 clear() 原地清空，不会换掉 dict 对象
        return self._hash_to_block_id.get

    def count_used(self, block_ids: list[int], n: int) -> int:
        """How many of the first `n` of `block_ids` (distinct) some live
        sequence holds. No slice; the set it builds holds only the used ids,
        and a waiting head's hit blocks are mostly free."""
        # 【译】block_ids（互不相同）的前 n 个里，有多少个正被某条活着的序列持有。不做切片；它建的集合只含正在用的 id，
        #   而一个排在等待队头的请求，命中的块大多是空闲的。
        # 用途：BlockManager._evaluate（can_allocate）算要从空闲表拿多少块 = 总块数 − 已在用的命中块（共享不占空闲名额）
        #   + 草稿 lookahead。例：上例第 4 步 count_used([0,1], 2) = 0；"变体"里 = 2。
        #   Python 语法：islice(block_ids, n) 只迭代前 n 个、不复制；set.intersection(可迭代对象) 只保留两边都有的元素
        return len(self._used.intersection(islice(block_ids, n)))

    def claim_many(self, block_ids) -> None:
        """`claim` each of `block_ids`, in order, with the lookups hoisted."""
        # 【译】依次对 block_ids 的每一个做 claim，把属性查找提到循环外面。
        # 调用方：BlockManager.allocate——把准入探测（can_allocate 里的 _hit_walk）找到的那串命中 id 一次认领，
        #   结果与逐个调 claim 完全相同。上例第 4 步：claim_many([0, 1])
        # 局部变量绑定：循环里不再每次查 self.blocks / self._used / self._take_named
        blocks, used, take = self.blocks, self._used, self._take_named
        slru = self.cache_policy == "slru"
        for block_id in block_ids:
            block = blocks[block_id]
            # 【分支】slru：有 hash 的块被命中复用 → 记下，释放时进受保护段
            if slru and block.hash != -1:
                self._reused.add(block_id)
            # 已有序列在用（例如两条请求同时用同一段 system prompt）→ 只加引用计数，共享同一页 KV；
            #   否则它在空闲表里：空闲块的引用计数必须是 0（断言），置 1、从空闲表摘下（_take_named）、加入 _used
            if block_id in used:
                block.ref_count += 1
            else:
                assert block.ref_count == 0
                block.ref_count = 1
                take(block_id)
                used.add(block_id)

    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        """Index `block_id` under the content hash of the tokens it now holds.
        A different hash it was indexed under goes, as on eviction: indexed
        content never changes."""
        # 【译】把 block_id 登记在它现在装的 token 的内容 hash 下。它之前登记过的另一个 hash 要撤掉，就像淘汰时一样：
        #   已登记的内容永远不变。
        # 调用方：BlockManager.hash_blocks（prefill / decode 写满一块、KV 已算好之后，在 postprocess 里），
        #   以及远端前缀 / offload 加载等【分支】。三步：
        #   ① 这个块之前以另一个 hash 登记、且索引还指向它 → 删掉旧条目（index_version + 1、on_evict）：索引里的内容必须与块里的一致
        #   ② block.update 写入 hash 和 token
        #   ③ 索引里 h 已经指向这个块 → 结束；否则 index[h] = block_id。若 h 原来指向另一个块（同一内容在两个块里各算了一份），
        #      就改指到这个新块，index_version + 1、on_remap(h)
        # 例：上例第 3 步之后，来了一条与 A 完全相同的 12 token 请求 A'：只能命中 0、1（最后一块不复用），第 2 块要重算到一个
        #   新块 3 里；publish(3, h2) 时 h2 原来指向 2 → 改指 3，on_remap(h2)。
        # 观察：改指之后，旧块 2 的 Block.hash 仍是 h2（publish 只更新新块），它已经不可能被命中，却仍被当作"有内容"留在
        #   _cached 里（空块用完才轮到它），并计入 num_reusable_free（统计里的 kv_blocks_evictable）；最终被分配时 _unindex
        #   发现索引不指向它，不计入 blocks_evicted。clear_index 之后仍被序列持有的块也一样（它们保留 hash，释放时进 _cached）
        block = self.blocks[block_id]
        index = self._hash_to_block_id
        old = block.hash
        if old not in (-1, h) and index.get(old) == block_id:
            del index[old]
            self.index_version += 1
            if self._on_evict is not None:
                self._on_evict(old)
        block.update(h, token_ids)
        prev = index.get(h)
        if prev == block_id:
            return
        index[h] = block_id
        if prev is not None:
            self.index_version += 1
            if self._on_remap is not None:
                self._on_remap(h)

    def clear_index(self) -> None:
        """Drop every content-hash entry, keeping blocks live sequences hold.

        Those stay valid through their block_table refs; they are simply no
        longer reachable by hash, so no future request can claim them.
        """
        # 【译】删掉所有内容 hash 条目，保留活着的序列持有的 block。
        #
        #   那些块通过 block_table 的引用依然有效；只是不再能通过 hash 找到，所以以后的请求都不能认领它们。
        # 调用方：BlockManager.clear_cache（docstring 称供 /reset_prefix_cache 一类的管理接口用；atom/ 里目前只有测试调用）。
        #   只清空闲块的 hash / token；
        #   在用的块保留自己的 hash（后续 hash_blocks 还要读它当父 hash），但索引里已经没有它了
        self._hash_to_block_id.clear()
        self.index_version += 1
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = array.array("i")
        # Every free block is vacant now, and which container an id is in is
        # only ever decided from its hash — so the split has to be redrawn
        # here rather than left to drift.
        # 【译】现在每个空闲块都是空块，而一个 id 该在哪个容器只由它的 hash 决定——所以划分必须在这里重画，不能任其漂移。
        #   （sorted 的结果本身就是合法的堆，heapify 只是保险）
        self._cached.clear()
        self._protected.clear()
        self._reused.clear()
        self._vacant = sorted(self._free)
        heapify(self._vacant)

    def _unindex(self, block_id: int) -> bool:
        """Drop `block_id`'s index entry and forget what it held.

        Returns whether an index entry actually went — i.e. whether reusable
        content was destroyed. A block with no hash, or one whose hash the
        index has since re-pointed elsewhere, costs nothing to drop, and the
        callers that count evictions must not count those.
        """
        # 【译】删掉 block_id 的索引条目，并忘掉它装过什么。
        #
        #   返回是否真的删掉了一个索引条目——也就是是否销毁了可复用的内容。一个没有 hash 的块，或者它的 hash 已被索引
        #   改指到别处的块，删掉它没有代价，统计淘汰的调用方不能把这些算进去。
        # 调用方：allocate（淘汰，计 blocks_evicted）、retire_top / _adopt（收回，计 blocks_retired）。
        #   index.get(hash) == block_id 这个判断，防的正是 publish 里"观察"说的那种被改指过的旧块
        block = self.blocks[block_id]
        # 【分支】slru：不管它在不在受保护段，都清掉标记
        self._protected.pop(block_id, None)
        self._reused.discard(block_id)
        dropped = False
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
            self.index_version += 1
            dropped = True
            if self._on_evict is not None:
                self._on_evict(block.hash)
        block.hash = -1
        block.token_ids = array.array("i")
        return dropped

    # ---------------------------- allocation ------------------------------- #
    # 【译】---------------------------- 分配 ------------------------------- #
    def _take_free(self) -> int:
        """Next free block id, or -1. Vacant before cached; see the class doc.

        An entry is stale when the block has since been taken, or when it has
        gained or lost a hash and so belongs in the other container — an id can
        sit in both at once, and testing only that it is free would hand a
        cached block out of the vacant half. Both conditions are the ones that
        decide which half it belongs to anyway, so the test is the definition
        rather than a guard bolted on top.
        """
        # 【译】下一个空闲 block id，没有则返回 -1。空块先于有缓存的块；见类 docstring。
        #
        #   条目陈旧有两种情况：这个块后来已被拿走；或者它后来获得或失去了 hash，因而属于另一个容器——一个 id 可以同时出现
        #   在两个容器里，只检查它是否空闲，就可能从空块那半边给出一个有缓存的块。这两个条件本来就是决定它属于哪半边的条件，
        #   所以这个检查就是定义本身，而不是额外加上去的防护。
        # 三个循环按优先级依次试：弹出一个条目，"在 _free 里、且 hash 状态与容器相符"才算数，否则丢掉这个陈旧条目继续弹。
        #   上例第 6 步：前两次从堆里给出 4、5；第三次堆已空 → _cached.popitem(last=False) 弹出最左端的 2。
        #   陈旧条目的来源：_take_named 只把块移出 _free 和两个有序容器，不从堆里删（堆删任意元素要 O(n)）
        while self._vacant:
            block_id = heappop(self._vacant)
            if block_id in self._free and self.blocks[block_id].hash == -1:
                self._free.discard(block_id)
                return block_id
        while self._cached:
            block_id, _ = self._cached.popitem(last=False)
            if block_id in self._free and self.blocks[block_id].hash != -1:
                self._free.discard(block_id)
                return block_id
        while self._protected:
            block_id, _ = self._protected.popitem(last=False)
            if block_id in self._free and self.blocks[block_id].hash != -1:
                self._free.discard(block_id)
                return block_id
        return -1

    # 取出一个空闲块号（只移出空闲表，还没有加入 _used）：调用方随后必须调 allocate(block_id)。
    #   pop + allocate 两步合起来相当于 nano-vllm 的 _allocate_block；BlockManager._fresh_block 和 reserve_units 都这样用。
    #   没有空闲块时抛 AssertionError：经过 Scheduler 的准入 / can_append 检查后不可达（见 BlockManager._fresh_block 的 docstring）
    def pop(self) -> int:
        block_id = self._take_free()
        if block_id < 0:
            raise AssertionError("No free blocks available")
        return block_id

    def _take_named(self, block_id: int) -> None:
        """Take one specific block off the free list, content and all.

        The cached half is ordered by when a block was released, so an id left
        in it after being taken would come back at its old position rather than
        the end — see the class doc. The vacant half is ordered by id, where a
        leftover entry is only a wasted pop.
        """
        # 【译】把一个指定的 block 连同它的内容从空闲表拿走。
        #
        #   有缓存的那半边按块的释放时间排序，所以一个被拿走后仍留在里面的 id，会回到它的旧位置而不是末尾——见类 docstring。
        #   空块那半边按 id 排序，留在那里的条目只会造成一次白费的 pop。
        self._free.discard(block_id)
        self._cached.pop(block_id, None)
        self._protected.pop(block_id, None)

    def allocate(self, block_id: int) -> Block:
        """Take `block_id` for fresh content, evicting whatever it held."""
        # 【译】为新内容拿下 block_id，淘汰它原来装的东西。
        # 调用方：BlockManager._fresh_block（allocate 补新块、may_append 给 decode 补块、disown_claimed_prefix）与 reserve_units，
        #   都是先 pop 再调这里。与 nano-vllm 的 _allocate_block 对应：先删 hash 索引、再 reset、再记为在用
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 调用方给的必须是空闲块（上面的断言）。_unindex 返回 True = 销毁了可被命中的内容 → 这才是一次真正的淘汰（上例第 6 步的 2 号）
        if self._unindex(block_id):
            self.blocks_evicted += 1
        block.reset()
        # 上一行 reset 把 ref_count 置 1、清掉 hash 和 token。从 pop 拿来的块已经不在 _free 里，下面的 _take_named 对它
        #   基本是空操作，保证不经 pop、直接指名调用时也正确
        self._take_named(block_id)
        self._used.add(block_id)
        return block

    def claim(self, block_id: int) -> Block:
        """Take a share of `block_id` for the content it already holds.

        The cache-hit counterpart of `allocate`, and deliberately not built on
        it: `allocate`'s reset would drop the hash and destroy the entry for
        every other request that could still hit it.
        """
        # 【译】为 block_id 已经装着的内容拿一份引用。
        #
        #   这是 allocate 的命中版本，而且故意不建立在 allocate 之上：allocate 的 reset 会丢掉 hash，毁掉其他所有
        #   还可能命中它的请求要用的那个索引条目。
        # 调用方：BlockManager.allocate（准入探测的结果不能整段直接用时，剩下的块逐个认领）、deallocate_partial / acquire_offload_prefix
        #   （【分支】offload 保存的租约）等。逻辑与 claim_many 的循环体相同
        block = self.blocks[block_id]
        if self.cache_policy == "slru" and block.hash != -1:
            self._reused.add(block_id)
        if block_id in self._used:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self._take_named(block_id)
            self._used.add(block_id)
        return block

    # 释放一份引用。调用方：BlockManager.deallocate（逆序遍历 block_table，见文件头"淘汰顺序"）、disown_claimed_prefix、
    #   free_leased_blocks、release_units。引用计数没减到 0（还有别的持有者）就什么都不变；减到 0 才回到空闲表，
    #   并且不清 hash、不清内容（惰性淘汰）：有 hash 进 _cached 右端（slru 下被复用过的进 _protected），没有 hash 进 _vacant 堆。
    # 观察：重复释放不会立刻报错——ref_count 已为 0 的块再 free 一次会变成 -1，`if block.ref_count:` 为真于是直接返回；
    #   要等它下次被 allocate / claim 时的 `assert ref_count == 0` 才暴露（BlockManager.free_leased_blocks 的 docstring
    #   也说连调两次就是重复释放）
    def free(self, block_id: int) -> None:
        if block_id in self._raw_unit_owner:
            raise AssertionError(
                f"block {block_id} is a reserved raw unit; use release_units"
            )
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count:
            return
        self._used.remove(block_id)
        self._free.add(block_id)
        if block.hash != -1:
            if block_id in self._reused:
                self._protected[block_id] = None
                self._trim_protected()
            else:
                self._cached[block_id] = None
            return
        heappush(self._vacant, block_id)
        # Stale entries are skipped, not removed, so the heap can outgrow the
        # pool under churn. Rebuilding costs one pass and buys at least
        # `num_blocks` pushes.
        # 【译】陈旧条目是跳过而不是删除的，所以频繁进出时堆可能长得比池子还大。重建只要扫一遍，换来的是至少 num_blocks 次 push。
        if len(self._vacant) > 2 * self.num_blocks + 2:
            self._vacant = [b for b in self._free if self.blocks[b].hash == -1]
            heapify(self._vacant)

    def _trim_protected(self) -> None:
        # Bound protection so new prefixes can compete for cache space.
        # 【译】限制受保护段的大小，让新前缀也能竞争缓存空间。
        # 【分支】slru：超出 num_blocks × protected_ratio 时，最旧的降回 _cached 的右端（最新的位置），并清掉"复用过"的标记。
        #   例：6 块、比例 0.5 → 最多 3 块受保护
        limit = int(self.num_blocks * self.protected_ratio)
        while len(self._protected) > limit:
            block_id, _ = self._protected.popitem(last=False)
            self._reused.discard(block_id)
            self._cached[block_id] = None

    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None:
        """Reserve arbitrary PAGE-sized units for raw storage."""
        # 【译】为原始存储预留任意个 PAGE 大小的单元。
        # 【分支】状态 checkpoint 存进分页池（调用方 page_unit_checkpoint.py）：每个单元走 pop + allocate（可能淘汰有缓存的块），
        #   并记下所有者和片号；空闲块不够时返回 None，什么都不拿
        if count < 0:
            raise ValueError(f"unit count must be non-negative, got {count}")
        if owner is None:
            raise ValueError("a raw-unit reservation needs an owner")
        if not self.has_free(count):
            return None
        unit_ids: list[int] = []
        for piece_index in range(count):
            block_id = self.pop()
            self.allocate(block_id)
            self._raw_unit_owner[block_id] = (owner, piece_index)
            unit_ids.append(block_id)
        return unit_ids

    def release_units(self, unit_ids: Iterable[int], owner: Hashable) -> None:
        """Release a complete raw-unit reservation back to the PAGE pool."""
        # 【译】把一整份原始单元预留还给 PAGE 池。
        # 先全部校验（无重复、每个单元的所有者和片号都对），再全部释放：要么全还，要么一个都不动
        ids = list(unit_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("a raw-unit release contains duplicate ids")
        # Validate ownership before releasing any unit.
        # 【译】先校验所有权，再释放任何一个单元。
        for piece_index, block_id in enumerate(ids):
            actual = self._raw_unit_owner.get(block_id)
            expected = (owner, piece_index)
            if actual != expected:
                raise AssertionError(
                    f"raw unit {block_id} belongs to {actual!r}, not {expected!r}"
                )
        for block_id in ids:
            del self._raw_unit_owner[block_id]
            self.free(block_id)

    def rekey_units(
        self,
        unit_ids: Iterable[int],
        old_owner: Hashable,
        new_owner: Hashable,
    ) -> None:
        """Atomically transfer a raw reservation between logical owners.

        The units remain allocated throughout the hand-off.  This is required
        when an external-transfer scratch image becomes a reusable checkpoint:
        releasing and re-reserving would briefly expose the units through the
        free list, allowing another allocation to overwrite bytes that the new
        checkpoint index already advertises.
        """
        # 【译】把一份原始预留原子地从一个逻辑所有者转给另一个。
        #
        #   单元在交接期间始终保持已分配。当一份外部传输用的临时镜像变成可复用的 checkpoint 时需要这样做：先释放再重新预留
        #   会让这些单元短暂地出现在空闲表里，另一次分配就可能覆盖新 checkpoint 索引已经对外宣布的那些字节。
        ids = list(unit_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("a raw-unit ownership transfer contains duplicate ids")
        if old_owner is None or new_owner is None:
            raise ValueError("raw-unit ownership transfer needs both owners")
        for piece_index, block_id in enumerate(ids):
            actual = self._raw_unit_owner.get(block_id)
            expected = (old_owner, piece_index)
            if actual != expected:
                raise AssertionError(
                    f"raw unit {block_id} belongs to {actual!r}, not {expected!r}"
                )
        for piece_index, block_id in enumerate(ids):
            self._raw_unit_owner[block_id] = (new_owner, piece_index)

    # ------------------------------ resizing ------------------------------- #
    # 【译】------------------------------ 扩缩池 ------------------------------- #
    # 【分支】分页 KV 池与状态池共用一块固定显存时，靠移动边界在两者之间调配；v0.1.8 的 atom/ 里没有调用方（见 __init__ 的"观察"）
    def extend(self, count: int) -> int:
        """Grow the pool by up to `count` blocks; returns how many it took.

        Capped at `max_blocks`, which is how many blocks the plane can address
        — beyond it there is nothing to hand out even if the caller has bytes.
        """
        # 【译】把池子扩大最多 count 个 block；返回实际扩了多少。
        #
        #   上限是 max_blocks，也就是这个平面能寻址的 block 数——超过它就没有东西可以分出去了，哪怕调用方还有字节。
        # 新加的块号进 _free 和 _vacant 堆（还没装过东西）
        taken = min(count, self.max_blocks - self.num_blocks)
        for block_id in range(self.num_blocks, self.num_blocks + taken):
            self._free.add(block_id)
            heappush(self._vacant, block_id)
        self.num_blocks += taken
        return taken

    def retire_top(self) -> BlockRetirement | None:
        """Take the highest block id out of the pool, or None if it cannot.

        The highest specifically, because the ids the pool gives up have to be
        the ones the boundary is about to cover — any free block will not do.
        A block that is merely cached costs nothing to retire; the index entry
        goes and the content with it. One that a sequence still holds has to
        move, and its new id is reported so its holders can follow.

        Only fails when the top block is in use and nothing is free to move it
        into, which is the same condition that would block admitting the
        request in the first place.
        """
        # 【译】把最高号的 block 移出池子；不能移出时返回 None。
        #
        #   特意是最高号，因为池子交出去的 id 必须正好是边界将要盖住的那些——随便哪个空闲块都不行。一个仅仅是有缓存的块，
        #   收回它没有代价：索引条目删掉，内容随之作废。一个仍被序列持有的块必须搬走，它的新 id 会报告出去，让持有者跟着改。
        #
        #   只有当顶部的块正在使用、而又没有空闲块可以把它搬进去时才会失败，这与"一开始就无法接纳这个请求"是同一个条件。
        # 两种情况：顶部块空闲 → 直接摘下（有缓存的计 blocks_retired），moved_to = -1；
        #   顶部块在用 → 用 _take_free 找一个空闲块作目的地，_adopt 把身份搬过去（字节由调用方搬，block_table 由调用方改）。
        #   最后 num_blocks − 1，并按新的大小重新限制受保护段
        top = self.num_blocks - 1
        if top < 0:
            return None
        # Raw units cannot move without updating their owning record.
        # 【译】原始单元不更新它的所有者记录就不能搬。
        if top in self._raw_unit_owner:
            return None
        if top in self._free:
            self._take_named(top)
            if self._unindex(top):
                self.blocks_retired += 1
            destination = -1
        else:
            destination = self._take_free()
            if destination < 0:
                return None
            self._adopt(destination, top)
        self.num_blocks -= 1
        self._trim_protected()
        return BlockRetirement(top, destination)

    def _adopt(self, destination: int, source: int) -> None:
        """Give `source`'s identity to `destination`, leaving source empty.

        Ref count, hash and tokens all move: a relocation is invisible to the
        sequences holding the block except for the id itself, which the caller
        rewrites. The bytes are the caller's to move too — this is the
        bookkeeping half, and the two have to happen in the same pass.
        """
        # 【译】把 source 的身份交给 destination，并把 source 清空。
        #
        #   引用计数、hash 和 token 全部搬走：对持有这个块的序列来说，除了 id 本身（由调用方改写）以外，搬迁是看不见的。
        #   字节也由调用方搬——这里只是记账的那一半，两半必须在同一轮里完成。
        # The destination may have come off the cached half of the free list,
        # in which case making room for the relocation destroyed its content.
        # `retire_top` is the only caller, so the boundary is what spent it.
        # 【译】目的地可能来自空闲表里有缓存的那半边，那样的话，为搬迁腾地方就销毁了它的内容。retire_top 是唯一的调用方，
        #   所以花掉它的是边界移动。
        if self._unindex(destination):
            self.blocks_retired += 1
        src, dst = self.blocks[source], self.blocks[destination]
        if source in self._reused:
            self._reused.discard(source)
            self._reused.add(destination)
        dst.ref_count, dst.hash, dst.token_ids = src.ref_count, src.hash, src.token_ids
        if src.hash != -1 and self._hash_to_block_id.get(src.hash) == source:
            self._hash_to_block_id[src.hash] = destination
            self.index_version += 1
            if self._on_remap is not None:
                self._on_remap(src.hash)
        self._used.discard(source)
        self._used.add(destination)
        src.ref_count, src.hash, src.token_ids = 0, -1, array.array("i")
