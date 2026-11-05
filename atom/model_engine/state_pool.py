# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# =====================================================================================
# 【导读】state_pool.py —— 每请求状态的"活跃槽位池"（Active Slot）与 fork 式状态 checkpoint 索引（第4批 4b，B级）
#   调用方：只有 BlockManager（block_manager.py __init__ 里 self.state = StateSlotPool(...)）；调度器只经 BlockManager 间接使用
#   本文件开头的【科普】①~⑤ 是"没有逐 token KV 的模型怎样做前缀缓存"的规范讲解位置，
#   state_cache.py / state_runtime.py / state_offload.py / page_unit_checkpoint.py 都指回这里。
#
# 4b 五个文件的阅读路线（建议顺序）：
#   1. 本文件开头【科普】①~⑤：概念、图、数字
#   2. state_cache.py：所有"状态类"共同的协议（resumable_hit / checkpoint / successor_room / readable_midstep）
#   3. state_runtime.py：后端声明的"状态怎么交给另一个槽位"（StateTransfer：fork / copy / replay / none）与 midstep 几何
#   4. 本文件 StateSlotPool：活跃槽位的分配与释放、fork checkpoint 的登记 / 命中 / 惰性淘汰、钉住（pin）
#   5. page_unit_checkpoint.py：copy 方式——把状态快照切成 KV block 大小的"页单元"存进 KV 池
#   6. state_offload.py：【分支】快照的 CPU 层（LMCache）索引，见第9批
#   前置：分页 KV 与 block 见 config.py 文件开头"【科普】KV cache 与分页（paged）KV"；
#     前缀缓存与链式 hash 见 sequence.py Sequence 类上方"科普：前缀缓存"和 arg_utils.py enable_prefix_caching 上方的【科普】；
#     只讲概念的概要版见 arg_utils.py EngineArgs.state_checkpoint_interval_tokens 上方的【科普】、
#     sequence.py Replay 类上方"科普：状态checkpoint与有界重放"（本文件讲的是它们背后的实现）。
#
# 术语：活跃槽位（Active Slot）= 状态张量的一个下标，装一条请求一份完整状态（跨所有线性注意力层）；
#   checkpoint（快照）= 某个 hash block 边界处的状态副本；fork / copy / replay 见下面④(b)；
#   页单元（PAGE unit）= KV 池里一个 block 的全部字节，见⑤；钉住（pin）= 本步有 forward 要读它，暂不回空闲表。
# =====================================================================================
#
# 【科普】没有逐 token KV 的模型，前缀缓存靠什么？—— 从 MHA / GQA 出发，五步
#
# ① 出发点：MHA / GQA 的 KV 按 token 存，前缀命中 = 直接复用 block
#   每个 token 每层留一行 K、V，按 block（默认 16 个 token）分页放在 KV 池里。前缀缓存给每个填满的 block 算链式 hash
#   （h_i = hash(h_{i-1}, 第 i 块的 token)），新请求前 N 块的 hash 相同，就把这 N 个 block 的 ref_count 加 1 直接借用：
#
#       请求A  block_table = [blk7][blk3][blk12]        （位置 0~15 | 16~31 | 32~47）
#       请求B  与 A 前 32 个 token 相同 → block_table = [blk7][blk3][新块]，从位置 32 开始 prefill
#
#   能这样做的根本原因：KV 可以"按位置切"——位置 p 的 K/V 只由 token 0..p 决定，与 p 之后无关，
#   所以在任何 block 边界都能接上。nano-vllm 的 BlockManager 做的就是这件事，也只做这件事。
#
# ② 线性注意力（GDN / KDA）与 Mamba：每层每请求只有"一个"定长递归状态，概括了之前的全部 token
#   线性注意力去掉 softmax 后：o_t = S_t · q_t，S_t = f(S_{t-1}, k_t, v_t)。全部历史被压进一个固定大小的矩阵 S，
#   每来一个 token 原地更新一次（公式推导见 model_config/qwen3_next.py 文件头【科普】②）。
#   以 Qwen3-Next（48 层 = 36 层 GDN + 12 层全注意力，TP=1，bf16）为例，一层 GDN 每条请求要存：
#       temporal_state  [32 个 v 头, 128, 128]                 = 524,288 ≈ 52万个数
#       conv_state      [3, 8192]（短卷积要的最近 3 个输入）    =  24,576 个数
#     全部 36 层 ≈ 36 × 548,864 ≈ 1976万个数 ≈ 37.7 MiB/请求 —— 上下文 1 千还是 10 万都是这么大
#     对比：12 层全注意力每 token 的 KV = 12 层 × 2(K,V) × 2 个 kv 头 × 256 维 × 2 字节 = 24 KiB
#       → 一份 GDN 状态 ≈ 1608 个 token 的 KV（算法同 sequence.py Replay 类上方科普②）
#   问题：S 在第 p 个 token 之后马上被覆盖成 S_{p+1}，"位置 p 时的 S"只存在一瞬间。KV 能按位置切，S 不能——
#     手里有 S_20000，也推不回 S_16384。DeepSeek-V4 的滑动窗口环形缓冲 + 压缩器没凑满的 token 也是这种"滚动状态"。
#   所以对这类层，"KV block 命中 N 个 token"还不够，还必须有"恰好算到第 N 个 token 时的状态"。
#
# ③ 解法：在选定位置给状态拍快照（checkpoint），命中只能退到"≤ 命中点的最近一个快照"，其余重算
#   快照位置由 BlockManager 统一决定（所有状态类共用一套"阶梯"；位置都在 hash block 边界上，快照登记在
#   "该边界前最后一个 block 的链式 hash"名下，所以和 KV block 用同一个 hash 寻址）：
#     - 阶梯（ladder）：每 --state-checkpoint-interval-tokens 个 token 一级，默认 8192
#       （BlockManager.checkpoint_limit / checkpoint_cut / checkpointers_at）；-1 = 关阶梯、只留下面两种；0 = 整个关闭
#     - prompt 末尾锚点（anchor）：本 prompt 结尾向下取整到 hash block 边界、且不越过最后一个可匹配的 block
#       （BlockManager._record_checkpoint_end）。多轮对话下一轮会把整段 prompt 重发再往后接，
#       代码注释里的实测：cc-traces 上 93.5% 的恢复落在上一轮 prompt 末尾，落在 8192 阶梯上的是 0.0%
#     - 按需一级（demand rung）：某次命中因"缺快照"被截短时，在被拒的位置给自己补一级，后来的请求就能整段命中
#       （BlockManager._record_checkpoint_demand；--state-checkpoint-demand / ATOM_STATE_CHECKPOINT_DEMAND，默认开）
#     （三种位置怎样把 prefill chunk 切开、forward 结束时存不存，带数字的逐步走查——40000 token 的请求 A/B/C——见
#      block_manager.py checkpoint_limit 上方的【分支】状态checkpoint阶梯；这里只讲概念）
#   命中判定：BlockManager._gated_hit 把 KV 命中长度交给每个状态类的 resumable_hit（本文件 StateSlotPool.resumable_hit、
#     page_unit_checkpoint.py 的同名方法），从命中点往左扫，停在第一个"有快照"的边界：
#
#       上一轮 prompt A（20000 token）留下的快照：  8192 ▲       16384 ▲         19984 ▲ ← 锚点（20000 向下取整到16的倍数后，
#                                                                                         再卡在最后一个可匹配 block 之前）
#       新请求 B 与 A 共享前 18432 个 token：
#         位置   0 ───────────────── 16384 ──────── 18432 ─────────── B 的末尾
#                [复用 KV，状态从 16384 的快照恢复]│[重算 2048 个]│[B 自己的新内容]
#         → KV block 命中 18432，但状态只能从 16384 接上：num_cached_tokens = 16384，重算 18432 − 16384 = 2048 个 token
#         → B 的命中被截短了 2048，于是 B 在 18432 记一个按需快照；第三个同前缀请求就能从 18432 接上
#
#   代价：①快照占地方；②非 midstep 后端只能在 forward 的最后一个 token 处拿到状态，所以 prefill 要在快照位置"切一刀"
#     （多一次 forward，BlockManager.checkpoint_cut）。间隔因此不能太密，短于一个间隔的 prompt 阶梯一级都不放。
#     midstep 例外：chunk kernel 本来就在每 64 个 token 处算出中间状态，可直接拷出来，不必切（state_runtime.py 的 midstep 几何）。
#
# ④ 快照住在哪、槽位怎么分配 / 分叉 / 释放、怎样和全注意力层的分页 KV 保持一致
#   (a) 活跃槽位：ModelRunner 启动时按 max_num_seqs × 每请求槽位数 预分配一块状态张量（Pool.STATE，先于 KV 池扣除，
#       见 model_ops/attentions/pool_layout/sub_pool_spec.py plan_pools），每个下标 = 一份完整状态。
#       allocate 时从本类 pop_many(1 + num_spec) 个下标放进 seq.state_slots（[0] = 已提交状态，其余 = 投机解码回滚槽位，
#       见 sequence.py has_per_req_cache 处的科普），deallocate 时 release_many 全部还回。下标不要求相邻。
#   (b) 快照放哪，由后端声明的 StateTransfer 决定（state_runtime.py；v0.1.8 各后端的 state_transfer() 实际返回）：
#
#       方式     谁用                                                  快照存在                      恢复时
#       fork(1)  Qwen4-exp；GDN / KDA 家族在 PP>1 或 RapidServe 时       空闲表里"内容仍有效"的槽位本身  下一次 forward 读它（只读）、写新槽位
#       copy     Qwen3-Next / Qwen3.5（GDN）、Kimi-K3 / GLM-5.3（KDA），  KV 池里若干个 block（页单元）  把页单元 gather 回新分配的活跃槽位
#                以上均要求 PP=1 且非 RapidServe；DeepSeek-V4；           见⑤ 与 page_unit_checkpoint.py
#                DeepSeek-V4.1 关闭 encoder 侧有界重放时
#       replay   DeepSeek-V4.1（--enable-encoder-swa-bounded-replay，默认）  不存                        从命中点往回退 1 个 hash block 重算
#       none     没有每请求状态的模型（默认）                              —                             —
#
#     本类在三种情况下的角色：fork → 既分配槽位又管 fork 快照（BlockManager.state_caches = (self.state,)）；
#       copy → BlockManager 用 StateTransfer.none() 构造本类，只分配槽位（外加 midstep 用的临时 scratch 槽位），
#       快照归 PagedStateCheckpointCoordinator；replay → 只分配槽位，resumable_hit 原样放行。
#   (c) fork 方式下一个快照的一生（本文件；num_spec=0，s 表示 seq.state_slot）：
#
#       请求A prefill 到 8192（forward 恰好结束在这里）→ hash_blocks → checkpoint(A, 512, h_8192)：
#           hash_to_slot[h_8192] = 3（A 原来的槽位 3 变成只读快照，并被钉住）；A 改用新槽位 s=5，state_fork_src=3
#       下一轮调度：A 的 forward 读槽位 3、写槽位 5（fork）；这一轮开头的 release_pins 还留着 3（reader_is_next_batch）
#       再下一轮：release_pins 把 3 放回空闲表"有内容"的一半（_checkpointed，LRU 序）——空闲但仍可被命中
#       请求B 命中 8192：lookup(h_8192)=3 → claim(3) 并钉住，B 拿新槽位 s=7，第一次 forward 读 3、写 7；
#           下一轮 release_pins 后 3 回到 LRU 尾部（被读过 = 被用过，排到最后才淘汰）
#       没空槽位时：pop() 先拿"无内容"的一半（_vacant，下标小的先），拿完才花掉 LRU 头部的快照——分配才是淘汰事件
#
#     所以 fork 快照不占额外容量：它就是"暂时没人用、内容还有效"的槽位。代价：槽位总数 = max_num_seqs × 每请求槽位数，
#     满并发时空闲槽位为 0，快照会被挤光（gdn_attn.py GDNStateMixin.state_spec 的 docstring 说：max_num_seqs 越小、
#     命中率反而越差）。这正是 Qwen / K3 默认改走 copy 的原因：快照放进 KV 池，能存多少由 KV 池大小决定。
#   (d) 与分页 KV 的一致性：混合模型（Qwen3-Next：12 层全注意力有 KV block，36 层 GDN 有状态）的一次前缀命中是对
#       "两边同时成立"的声明——全注意力层的 KV block 在 KV 索引里，GDN 层的快照也在。两边用同一条链式 hash 寻址：
#         - 命中长度 = KV 命中再经状态门控截短（_gated_hit），allocate 只认领到快照位置为止的 KV block，其余重算
#         - KV block 被淘汰时（BlockPool 的 on_evict 回调 → BlockManager._record_evicted），同一 hash 的快照立刻 unindex：
#           这一环的 KV 没了，快照永远不会再被命中，留着只会挤掉还有用的快照（本类 unindex 的 docstring）
#         - 清空前缀缓存时两边一起清（BlockManager.clear_cache 依次调 kv.clear_index 和 状态类.clear_index；这个管理接口在
#           v0.1.8 的 atom/ 里只有测试调用）
#
# ⑤ 页单元 checkpoint（page_unit_checkpoint.py，copy 方式）
#   页单元 = KV 池（Pool.PAGE）里一个 block 的全部字节（跨所有全注意力层，ModelRunner 里 PagedStateCheckpointSpec.page_unit_bytes）。
#   一份快照的字节流（image_bytes，可以只是槽位的一部分）切成 units_per_checkpoint = ceil(image_bytes / page_unit_bytes) 段，
#   分散写进从 KV 池借来的这么多个 block（BlockPool.reserve_units，不要求连续）：
#
#       活跃槽位 s（一段连续字节）  ──scatter（CheckpointStoreOp）──>  KV 池的 block #812、#45、#3071 …（约一百个）
#       新请求的活跃槽位 s'         <──gather （CheckpointRestoreOp）──  同一组 block，按顺序拼回
#
#   数字：Kimi-K3 一个 KDA 槽位 53.6 MiB，存成页单元是 127 个普通 KV block、约占分页池的 0.112%
#     （kimi_mla_gdn_attn.py state_transfer 的 docstring）；Qwen3-Next 按②估算（bf16 KV、block_size=16）：
#     一个 block = 16 × 24 KiB = 384 KiB，37.7 MiB / 384 KiB ≈ 101 个 block。
#   和 KV 怎么抢空间：页单元从 KV 池空闲表里拿（先拿没用过的 block，再拿只是缓存着旧前缀的 block）；只有空闲表不够时，
#     才按快照自己的 LRU 淘汰最旧的整份快照（PageUnitCheckpointStore.ensure_free_units）；反过来，快照所在边界的
#     KV block 被淘汰时，快照随之作废（④(d)）。
#   为什么要它：fork 快照只能住在空闲槽位里，数量被并发卡死；页单元把快照放进 KV 池，数量随 KV 池走，不需要大段连续显存，
#     CPU 层（state_offload.py）也能按页单元直接搬运。
#   它和 V4 / V4.1 的"有界重放"是两回事：有界重放是"不存快照、命中时往回重算一小段"（sequence.py Replay 类上方科普④、
#     state_runtime.py 的 REPLAY）；页单元 checkpoint 是"存快照"的一种存法。DeepSeek-V4 默认用页单元存它的
#     滑动窗口 + 压缩器状态（只存 image_bytes 那一部分，见 docs/scheduling_kv_cache_guide.md "State checkpoints"）。
#
# 对比nano-vllm：nano-vllm 只支持全注意力模型，没有每请求状态、没有槽位池、也没有状态 checkpoint，前缀缓存只做①。
#   ATOM 的整个 4b（本文件约 800 行 + page_unit_checkpoint.py 约 1400 行 + 另外三个文件）都是为了让②里的模型也能做前缀缓存；
#   BlockManager 里的 checkpoint 阶梯、_gated_hit、joint boundary 等是它们在 4a 一侧的接口。
# =====================================================================================
# heapq：Python 标准库的最小堆（在普通 list 上原地维护），heappop 总是弹出最小值，heappush / heapify 维护堆序
from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from atom.model_engine.state_runtime import StateTransfer


# 用途：retire_top（缩小状态池时交还最高下标）的返回值，告诉调用方要不要搬一次字节
# dataclass(frozen=True)：自动生成 __init__ 等方法，且实例不可修改（给字段赋值会抛错）
# 观察：v0.1.8 中 retire_top / extend / record_relocation 在 atom/ 下没有调用方（只有 tests/test_state_checkpoint.py 用），
#   "KV 与状态池之间挪边界"的通路尚未接上；take_relocations 每批都被 BlockManager.take_state_maintenance_ops 取一次，结果总为空
@dataclass(frozen=True)
class SlotRetirement:
    """What `retire_top` did, and what the caller still owes.

    `relocated_to` of -1 means the top slot was empty or was itself the thing
    worth spending, so no bytes move. Otherwise the caller must copy
    `retired` → `relocated_to` before the next forward, and then either re-point
    the owning sequence (`held_checkpoint` false) or nothing at all — the index
    has already been re-filed here, because it is this class's to keep
    consistent.
    """

    # 【译】retire_top 做了什么，以及调用方还欠什么。
    #   relocated_to 为 -1 表示顶部槽位是空的，或者它本身就是该被花掉的那个，所以不搬任何字节。
    #   否则调用方必须在下一次 forward 之前把 retired 拷到 relocated_to，然后：held_checkpoint 为 False 时
    #   让拥有它的序列改指向新槽位；为 True 时什么都不用做——索引已经在这里重新登记好了，因为保持它一致是本类的职责。

    retired: int
    relocated_to: int
    held_checkpoint: bool


class StateSlotPool:
    """Own Active Slot allocation and the fork-checkpoint index.

    A *slot* is one index of the pre-allocated state tensor: one complete
    recurrent state, across every layer. Slots are handed out *per need*, not in
    fixed-width groups. A live request under speculative decoding holds
    `1 + num_spec` of them — one committed state plus a rollback slot per
    speculated token — while a checkpoint holds exactly one, because the
    rollback slots are scratch a resumed prefix has no use for. That asymmetry
    is the point: allocating the wide shape unconditionally would cost a
    checkpoint three slots at `--num-speculative-tokens 2` rather than one, out
    of the same budget as the KV cache.

    The slots one request holds need not be adjacent. Nothing downstream reads
    them by arithmetic on a base: the ssm kernel gathers each index out of the
    indices tensor and the conv path is handed column 0 alone, so a request's
    slots reach the kernels as a list and scattered is as good as contiguous.
    """

    # 【译】负责活跃槽位（Active Slot）的分配和 fork checkpoint 索引。
    #   一个"槽位"是预分配状态张量的一个下标：一份完整的递归状态，跨所有层。槽位按需发放，而不是按固定宽度成组发放。
    #   开投机解码的在跑请求持有 1 + num_spec 个——一个已提交状态，外加每个投机 token 一个回滚槽位；而一个 checkpoint
    #   恰好只占一个，因为回滚槽位是草稿用的临时空间，恢复出来的前缀用不上。这种不对称正是要点：如果无条件按"宽"形状分配，
    #   --num-speculative-tokens 2 时一个 checkpoint 要花 3 个槽位而不是 1 个，而它们和 KV cache 出自同一份预算。
    #
    #   一个请求持有的槽位不必相邻。下游没有任何地方用"基址 + 偏移"去算它们：ssm kernel 从索引张量里逐个 gather 下标，
    #   conv 路径只拿到第 0 列，所以请求的槽位以列表形式交给 kernel，分散的和连续的一样好用。
    # 在 BlockManager 里的构造：StateSlotPool(num_state_slots, transfer=(copy 时为 StateTransfer.none()，否则为后端的 transfer),
    #   hash_block_size=BlockManager.hash_block_size, enabled=enable_prefix_caching)；三种 transfer 下的角色见文件开头④(b)
    # 数据结构一览（num_slots=8 的例子；"快照"= 空闲但内容是有效 checkpoint）：
    #   槽位：         0     1     2     3     4      5     6     7
    #   状态：        空闲  空闲  空闲  在用  钉住   快照  快照  快照
    #   slot_hash     -1    -1    -1    -1    h4     h2    h1    h3   （每个槽位登记的 hash）
    #   _free         = {0,1,2,5,6,7}          两半的并集，"空闲多少个"只看它（准入门 has_free 读它）
    #   _vacant       = 最小堆[0,1,2]           空闲且无内容；可能残留过期项，_pop_vacant 弹出时丢掉
    #   _checkpointed = deque([6,5,7])          空闲且有快照；左端 = 最久没用（先花掉），右端 = 最近用过
    #   hash_to_slot  = {h1:6, h2:5, h3:7, h4:4}
    #   _pinned       = {4: 1}                  4 是刚交出的 fork 源，下一个 forward 要读它，读完才进 _checkpointed
    #   pop() 依次给出：0 → 1 → 2 → 6（花掉 h1，计 evicted）→ 5 → 7
    # 对比nano-vllm：nano-vllm 的 BlockManager 用 free_block_ids + hash_to_block_id 对 KV block 做惰性淘汰
    #   （释放时内容和 hash 都保留，重新分配时才删 hash）。本类把同一模式搬到状态槽位上，多了三点：
    #   空闲表分成"无内容 / 有快照"两半、无内容的永远先发；快照是只读的 fork 源，用 pin 计数挡住"边读边被别人写"；
    #   猜测性的快照（阶梯、按需）排在 LRU 头部先被淘汰。

    def __init__(
        self,
        num_slots: int,
        transfer: StateTransfer | None = None,
        hash_block_size: int = 1,
        enabled: bool = True,
    ):
        # num_slots：状态张量的槽位总数（= config.pool_entries["state"] = max_num_seqs × 每请求槽位数，ModelRunner 规划好后回填）
        # transfer：后端的 StateTransfer（state_runtime.py）；hash_block_size：前缀缓存 hash 粒度（block_size × DCP）
        # enabled：是否开启前缀缓存；关闭时所有 checkpoint 相关方法都退化为空操作，只剩槽位分配
        self.enabled: bool = enabled and num_slots > 0
        self.num_slots: int = num_slots
        self.transfer: StateTransfer = transfer or StateTransfer.none()
        # Committed tokens needed to make a fork destination self-contained.
        # 【译】让 fork 的目标槽位"自给自足"所需的已提交 token 数。
        # 解释：fork(1) 时为 1——checkpoint 之后的下一个 forward 至少要带 1 个 token，才能把新槽位写成完整状态
        #   （conv 写路径会把整个卷积窗口写进输出槽位）；successor_room 是同一个数的 float 版，none / replay 时为 inf
        self.min_fork_tokens: int = self.transfer.fork_tokens
        self.successor_room: float = self.transfer.successor_room
        # Whether a checkpoint may sit anywhere inside a forward rather than
        # only at its last token. False is the conservative answer and the
        # reason `BlockManager` still cuts prefill chunks onto rungs; see
        # `StateTransfer.readable_midstep`.
        # 【译】检查点是否可以位于一次 forward 内部的任意位置，而不只是它的最后一个 token。
        #   False 是保守的答案，也是 BlockManager 仍然把 prefill chunk 切在阶梯上的原因；见 StateTransfer.readable_midstep。
        self.readable_midstep: bool = self.transfer.readable_midstep
        self.hash_block_size: int = hash_block_size
        if self.transfer.copies:
            raise ValueError("PAGE-copy checkpoints do not belong to StateSlotPool")
            # copy 方式的快照由 page_unit_checkpoint.py 管；BlockManager 在 copy 时给本类传 StateTransfer.none()，所以这里只防误用
        # The free list, split by whether the slot still carries content worth
        # something. Two containers rather than one queue because the two halves
        # want opposite orders and mixing them serves neither:
        #
        #   `_vacant`       nothing to lose, so order is free and it is chosen
        #                   to pack allocation towards index 0 — which is what
        #                   lets the pool shrink from the top without having to
        #                   move anybody.
        #   `_checkpointed` content is a resumable boundary, so order is LRU
        #                   and the head is what a shortage spends first.
        #
        # Vacant is always drawn from first, so a checkpoint can only be evicted
        # once there is nothing free left to take. A single release-ordered FIFO
        # cannot express that: a checkpoint released before a never-used slot
        # sits ahead of it and is spent first.
        #
        # Which half a free slot belongs to is a function of `slot_hash`, not
        # a third piece of state, so `_set_hash` is the one place that moves a
        # slot across and nobody else has to remember to. `_free` is membership
        # for both; `_vacant`'s heap keeps entries a `claim` took out and
        # `_pop_vacant` drops them when they surface.
        # 【译】空闲表，按"槽位里是否还有值钱的内容"拆成两部分。用两个容器而不是一个队列，是因为两半想要相反的顺序，
        #   混在一起两边都照顾不好：
        #     _vacant        没东西可丢，顺序随意，于是选成"把分配往下标 0 压紧"——这样池子可以从顶部缩小而不用搬动任何人。
        #     _checkpointed  内容是可恢复的边界，所以按 LRU 排序，队头是短缺时最先被花掉的。
        #   总是先从 vacant 里拿，所以只有在已经没有别的空闲槽位可拿时，checkpoint 才会被淘汰。
        #   单个按释放顺序排列的 FIFO 表达不了这一点：一个比"从没用过的槽位"更早释放的 checkpoint 会排在它前面、先被花掉。
        #
        #   一个空闲槽位属于哪一半，是 slot_hash 的函数，而不是第三份状态，所以 _set_hash 是唯一把槽位从一半挪到另一半的地方，
        #   其他人不用记着去做。_free 是两半共同的成员集合；_vacant 的堆里会留着被 claim 取走的条目，由 _pop_vacant 在它们
        #   浮到堆顶时丢掉。
        self._free: set[int] = set(range(num_slots))
        self._vacant: list[int] = list(range(num_slots))
        self._checkpointed: deque[int] = deque()
        # Slots whose checkpoint is a guess rather than this prompt's own end;
        # `mark_speculative` explains what the distinction is worth. Cleared
        # when the guess pays off (`promote`) or the content goes (`release` of
        # a slot carrying nothing) — both of which end the slot's claim to be
        # spent first.
        # 【译】checkpoint 只是"猜测"、而不是本 prompt 自己末尾的那些槽位；这种区分值多少，见 mark_speculative。
        #   猜测兑现时（promote）或内容消失时（release 一个不带任何东西的槽位）清除——两者都终结了"该槽位应被先花掉"的资格。
        self._speculative: set[int] = set()
        # hash -> slot holding the state as of that block boundary.
        # 【译】hash -> 保存着"到该 block 边界为止的状态"的槽位。
        self.hash_to_slot: dict[int, int] = {}
        # Reverse map, for lazy eviction when a slot is handed out. -1 = the
        # slot carries no checkpoint.
        # 【译】反向映射，用于槽位被发出去时的惰性淘汰。-1 = 该槽位不带 checkpoint。
        self.slot_hash: list[int] = [-1] * num_slots
        # Groups serving as a fork source in the in-flight step, counted by how
        # many requests fork off each. They stay off the free list until
        # `release_pins`, so the step that reads them cannot race a request that
        # was handed the same slot. The count matters because a checkpoint is
        # read-only and several requests may share one: the slot goes back only
        # once, and no reader may take it over while another still reads it.
        # 【译】在进行中的这一步里充当 fork 源的槽位，计数 = 有多少个请求从它 fork。它们在 release_pins 之前不回空闲表，
        #   这样读它们的那一步不会和"被发到同一个槽位的请求"发生竞争。计数有意义，是因为 checkpoint 是只读的、
        #   可以被多个请求共享：槽位只还一次，而且只要还有人在读，任何读者都不能把它接管过去。
        self._pinned: dict[int, int] = {}
        # Of those, the ones whose reader is the batch the *next* pass builds
        # rather than the one just built. `checkpoint` runs in postprocess,
        # after its batch went out, so the forward that reads the source it
        # hands over is one pass further off than a resume's. Carried as a set
        # rather than a second count because the depth is only ever one pass
        # more — see `release_pins`.
        # 【译】其中那些读者是"下一轮才构造的 batch"、而不是刚构造好的这一批的。checkpoint 在 postprocess 里运行，
        #   那时它所在的 batch 已经发出，所以读它交出的源的那次 forward 比恢复（resume）的情况还要再远一轮。
        #   用一个集合而不是第二个计数，因为深度永远只多一轮——见 release_pins。
        self._deferred: set[int] = set()
        self._relocations: list[tuple[int, int]] = []
        # 待执行的槽位搬迁 (src, dst)，由 take_relocations 取走、经 StateMaintenanceOps 交给 runner（见上方 SlotRetirement 处的观察）
        # `dropped` had no slot to go to; `evicted` landed and was later
        # spent on an allocation. Counted apart because they read the same in a
        # hit rate and want opposite fixes — the first says the pool is too
        # small for the rate of keeping, the second for how long a checkpoint
        # has to last.
        # 【译】dropped 表示没有槽位可去；evicted 表示落了地、后来被一次分配花掉。分开计数，是因为它们在命中率里看起来一样，
        #   却要相反的修法——前者说明池子相对"保存的速率"太小，后者说明池子相对"一个 checkpoint 需要活多久"太小。
        self.checkpoints_kept: int = 0
        self.checkpoints_dropped: int = 0
        self.checkpoints_evicted: int = 0
        # Died because the prefix it was filed under left the KV index first,
        # so nothing could ever have resumed off it. Apart from `evicted`
        # because they want opposite fixes: `evicted` says this pool is too
        # small for how long a checkpoint has to last, `orphaned` says the KV
        # pool is too small for the same span.
        # 【译】因为它登记所依附的前缀先离开了 KV 索引而死掉，所以根本不可能有人从它恢复。和 evicted 分开，
        #   因为修法相反：evicted 说明本池相对 checkpoint 需要存活的时长太小，orphaned 说明 KV 池相对同样的时长太小。
        self.checkpoints_orphaned: int = 0

    # ------------------------------ free list ------------------------------ #
    # 【译】空闲表
    def has_free(self, count: int = 1) -> bool:
        """Whether `count` slots can be handed out.

        A checkpoint on the free list counts: `pop` spends it rather than
        refusing, so it holds no capacity back. This is the admission gate, and
        the caller asks for what it will actually take — `1 + num_spec` for a
        request that will speculate, 1 for one that will not.
        """
        # 【译】能否发出 count 个槽位。
        #   空闲表上的 checkpoint 也算数：pop 会花掉它而不是拒绝，所以它不占住任何容量。这是准入门，调用方问的是它实际会拿的数量——
        #   会做投机的请求问 1 + num_spec，不会的问 1。
        # 调用方：BlockManager._evaluate（seq.has_per_req_cache 时 has_free(state_slots_per_req)，不够就返回 -1 拒绝准入）、
        #   BlockManager._attach_state_slots（恢复时能否再给一整套）、Scheduler._warn_if_unschedulable；本类 reserve_midstep / checkpoint
        return len(self._free) >= count

    def num_free(self) -> int:
        return len(self._free)

    def is_free(self, slot: int) -> bool:
        return slot in self._free

    def holds_checkpoint(self, slot: int) -> bool:
        """Whether `slot` is on the free list *and* still worth something."""
        # 【译】slot 是否在空闲表上，并且仍然值点什么（带着 checkpoint）。
        return slot in self._free and self.slot_hash[slot] != -1

    def pop(self) -> int:
        """Hand out a slot, evicting its checkpoint if it carried one.

        Vacant first, lowest index first; only when nothing is vacant does a
        checkpoint get spent, and then it is the least recently used one.

        The state twin of `BlockManager._pop_free_block`: slots sit on the free
        list carrying whatever the last owner left in them, and re-allocation —
        not the free — is the eviction event.

        Lowest-index-first is not a fairness choice, it is what keeps the top of
        the pool cold: a slot high in the range is only reached at a
        high-water-mark of concurrency, so after the peak passes the top holds
        the least recently touched things — exactly what shrinking should spend.

        One slot. A caller needing a request's whole set asks `pop_many`.
        """
        # 【译】发出一个槽位；如果它带着 checkpoint，就把那个 checkpoint 淘汰掉。
        #   先 vacant、下标最小的先；只有没有 vacant 时才花掉一个 checkpoint，而且是最久没用的那个。
        #
        #   它是 BlockManager._pop_free_block 的状态版"双胞胎"：槽位待在空闲表上时还带着上一任主人留下的内容，
        #   淘汰事件是重新分配，而不是释放。
        #
        #   下标最小优先不是为了公平，而是为了让池子顶部保持"冷"：下标高的槽位只在并发达到高水位时才会被用到，
        #   所以高峰过后，顶部放的都是最久没碰过的东西——正好是缩池时该花掉的。
        #
        #   只发一个。需要一个请求整套槽位的调用方用 pop_many。
        # 观察：docstring 里的 BlockManager._pop_free_block 在 v0.1.8 中已不存在（KV 侧对应的是 BlockPool.pop / BlockManager._fresh_block）
        # 例：_vacant=[] 且 _checkpointed=deque([6,5,7]) 时，pop() 拿走 6，checkpoints_evicted += 1，invalidate(6) 删掉 hash_to_slot 里的 h1
        slot = self._pop_vacant()
        if slot < 0:
            slot = self._checkpointed.popleft()
            self._free.discard(slot)
            self.checkpoints_evicted += 1
        # Whatever guess this slot carried is spent with it. Not folded into
        # `invalidate`, which also runs on the re-file inside `_index` and would
        # clear a mark that was only just made.
        # 【译】这个槽位带着的任何"猜测"标记随它一起花掉。不并进 invalidate，因为 invalidate 也会在 _index 内部重新登记时运行，
        #   那样会把刚打上的标记清掉。
        self._speculative.discard(slot)
        self.invalidate(slot)
        return slot

    def pop_many(self, count: int) -> list[int]:
        """Hand out `count` slots as one request's set. Caller checked `has_free`.

        Returned in allocation order, and the caller keeps that order: element 0
        is the committed state every path reads and writes, and elements 1..n
        are speculation rollback, which only the spec-decode path touches. The
        set is not adjacent and nothing may assume it is.

        Not atomic, because it does not need to be: `has_free(count)` was
        checked against a free list only this thread mutates, and `pop` never
        refuses — with nothing vacant it spends the LRU checkpoint. The worst
        case is a wide request evicting checkpoints for slots it then keeps,
        which is what admitting it means.
        """
        # 【译】为一个请求发出 count 个槽位作为一整套。调用方已经检查过 has_free。
        #   按分配顺序返回，调用方也保持这个顺序：第 0 个是每条路径都读写的已提交状态，1..n 是投机回滚，只有投机解码路径会碰。
        #   这一套不相邻，任何地方都不能假设它相邻。
        #
        #   不是原子的，因为不需要：has_free(count) 是对着一个只有本线程会修改的空闲表检查的，而 pop 从不拒绝——没有 vacant 时
        #   它花掉 LRU checkpoint。最坏情况是一个"宽"请求淘汰掉若干 checkpoint 来腾出它随后占着的槽位，而这本来就是准入它的含义。
        # 调用方：BlockManager._attach_state_slots（冷启动、copy 恢复、fork 恢复、adopt 四条路径）
        return [self.pop() for _ in range(count)]

    def _pop_vacant(self) -> int:
        """Lowest vacant index, or -1, dropping entries that are stale.

        An entry is stale when the slot has since been claimed or has taken a
        hash, both of which leave the heap untouched — cheaper than an exact
        removal, and the two conditions below are the same ones that decide
        which half a slot belongs to anyway.
        """
        # 【译】最小的 vacant 下标，或 -1；顺手丢掉过期条目。
        #   条目过期是指该槽位后来被 claim 走了、或者拿到了一个 hash——这两种情况都不去动堆，比精确删除便宜；
        #   而下面这两个条件，本来就是判断槽位属于哪一半的条件。
        # 解释：这是"惰性删除"的堆——删除时不改堆，弹出时再检查是否仍然有效
        while self._vacant:
            slot = heappop(self._vacant)
            if slot in self._free and self.slot_hash[slot] == -1:
                self._free.discard(slot)
                return slot
        return -1

    def claim(self, slot: int) -> None:
        """Take one specific free slot off the list, content and all.

        Linear in `_checkpointed` when the slot holds a checkpoint, which is
        the case the resume path takes. That is deliberate: the free list stays
        the single source of truth for "how many slots are free", which
        admission and every caller of `has_free` rely on. The scan is bounded by
        max_num_seqs and runs once per resuming request, against a
        `can_allocate` that already hashed every block of the prompt.
        """
        # 【译】把一个指定的空闲槽位连同内容一起从空闲表上拿下来。
        #   槽位带 checkpoint 时，对 _checkpointed 是线性扫描，恢复路径走的正是这种情况。这是故意的：空闲表保持为
        #   "空闲槽位有多少"的唯一真相来源，准入和每个调用 has_free 的地方都依赖它。扫描长度受 max_num_seqs 限制，
        #   每个恢复请求只做一次，而且它前面的 can_allocate 已经把 prompt 的每个 block 都 hash 过了。
        # 注意：claim 不清 slot_hash——快照被读之后经 release 回到 LRU 尾部，"被复用"就算"被使用"
        # 调用方：BlockManager._attach_state_slots（fork 命中、源未被别人钉住时）；本类 _set_hash、retire_top、_take_relocation_target
        self._free.discard(slot)
        if self.slot_hash[slot] != -1:
            self._checkpointed.remove(slot)

    def mark_speculative(self, slot: int) -> None:
        """File `slot` at the LRU *head* when it is released, not the tail.

        The next shortage then spends it before every checkpoint already held.
        For a placement that guesses where reuse will resume rather than
        knowing: the ladder rung and the demand guess, the prompt-end anchor
        knows.

        The guess and the knowledge are not close in value. Measured on the
        cc-traces at conc 4, prompt-end anchors are read back 85.2% of the time
        and demand rungs 2.8%, while the demand is 47% of all writes (1,370 of
        2,919) — so under plain LRU the placement almost never read evicts the
        one almost always read. Demoting rather than dropping keeps the demand's
        own purpose: it exists to fill a gap once, and one spent before it is
        read cost nothing a checkpoint never taken would have saved.

        A mark, not an argument to `release`, because the slot that carries a
        checkpoint under `fork` is *pinned* rather than released — it goes back
        through `release_pins` two passes later, which knows only the slot
        index. Marking is the one mechanism both paths can reach.
        """
        # 【译】让 slot 在被释放时排到 LRU 的"头部"而不是尾部。
        #   于是下一次短缺会先于所有已持有的 checkpoint 花掉它。用于那些"猜测"复用会从哪里恢复、而不是"知道"的放置：
        #   阶梯和按需是猜测，prompt 末尾锚点是知道。
        #
        #   猜测和知道的价值相差悬殊。在 cc-traces、并发 4 下实测：prompt 末尾锚点有 85.2% 会被读回，按需快照只有 2.8%，
        #   而按需快照占全部写入的 47%（2,919 次中的 1,370 次）——所以在朴素 LRU 下，几乎从不被读的放置会淘汰几乎总被读的。
        #   降级而不是直接丢弃，保留了按需快照本身的用意：它是为了填一次空缺而存在的；在被读之前就被花掉，
        #   代价也不超过"从没存过这个 checkpoint"。
        #
        #   做成一个标记而不是 release 的参数，因为 fork 下携带 checkpoint 的槽位是被钉住（pinned）而不是被释放的——
        #   它两轮之后经 release_pins 回去，而 release_pins 只知道槽位下标。打标记是两条路径都够得着的唯一机制。
        # 调用方：本类 checkpoint（fork）与 publish_midstep
        if self.enabled:
            self._speculative.add(slot)

    @staticmethod
    def _anchor_of(seq) -> int | None:
        """The one position this seq *knows* is a resume point, or None.

        None and 0 are different answers and the distinction is the whole
        policy: 0 would be a position, None means this seq has no known
        position at all and therefore everything it files is a guess.
        `_record_checkpoint_end` leaves `checkpoint_end_pos` at 0 on four
        paths, so reading the attribute directly conflates "not the anchor"
        with "no anchor exists" and demotes nothing in the second case.

        None compares unequal to every real position, so a caller that tests
        `pos != known` demotes everything -- which is what a seq with no known
        resume point should do. Whether to demote at all is the caller's
        separate question; see `publish_midstep` for `seq=None`.
        """
        # 【译】这条序列"知道"是恢复点的那一个位置，或 None。
        #   None 和 0 是不同的答案，而这种区分就是整个策略：0 是一个位置，None 表示这条序列根本没有已知位置，
        #   因此它登记的一切都是猜测。_record_checkpoint_end 在四条路径上把 checkpoint_end_pos 留为 0，
        #   所以直接读这个属性会把"不是锚点"和"不存在锚点"混为一谈，在后一种情况下什么也不降级。
        #
        #   None 与任何真实位置都不相等，所以调用方测试 pos != known 时会把所有东西都降级——这正是一条没有已知恢复点的序列
        #   应该做的。到底要不要降级是调用方另外的问题；seq=None 的情况见 publish_midstep。
        # staticmethod：不需要 self 的方法，可以用 StateSlotPool._anchor_of(seq) 调用
        # midstep_anchor_pos 优先：midstep 时锚点会被挪到 forward 的网格上（BlockManager.midstep_anchor），比较要用挪过的位置
        pos = 0
        if seq is not None:
            pos = getattr(seq, "midstep_anchor_pos", 0) or getattr(
                seq, "checkpoint_end_pos", 0
            )
        return pos or None

    def promote(self, slot: int) -> None:
        """A guess paid off: stop spending this one first.

        Called from the resume path when a request actually reads `slot`. The
        slot may be on the free list right now, so move it in `_checkpointed`
        too rather than only clearing the mark — otherwise the promotion would
        not take effect until something released it again.
        """
        # 【译】一次猜测兑现了：别再先花掉它。
        #   在恢复路径上、某个请求真的读了 slot 时调用。此刻 slot 可能就在空闲表上，所以也要在 _checkpointed 里挪动它，
        #   而不只是清标记——否则提升要等到下次有人释放它时才会生效。
        # 调用方：BlockManager._attach_state_slots（fork 命中时）
        if not self.enabled or slot not in self._speculative:
            return
        self._speculative.discard(slot)
        if slot in self._free and self.slot_hash[slot] != -1:
            self._checkpointed.remove(slot)
            self._checkpointed.append(slot)

    def release(self, slot: int) -> None:
        """Hand a slot back, to whichever half its content puts it in.

        A slot still carrying a checkpoint goes to the LRU tail, so being
        resumed from refreshes it — `claim` deliberately leaves the hash in
        place, which is what makes reuse count as use. One marked by
        `mark_speculative` goes to the head instead.
        """
        # 【译】把槽位还回去，放进它的内容所对应的那一半。
        #   仍带着 checkpoint 的槽位放到 LRU 尾部，所以"被恢复"会刷新它——claim 故意保留 hash，正是这一点让复用算作使用。
        #   被 mark_speculative 标记过的则放到头部。
        # 调用方：本类 release_many / drop_reader / release_pins / publish_midstep / cancel_midstep / _set_hash / extend；
        #   BlockManager.cancel_state_fork；page_unit_checkpoint.py 归还 midstep scratch 槽位（self._scratch.release）
        self._free.add(slot)
        if self.slot_hash[slot] != -1:
            if slot in self._speculative:
                self._checkpointed.appendleft(slot)
            else:
                self._checkpointed.append(slot)
            return
        self._speculative.discard(slot)
        heappush(self._vacant, slot)
        # Every slot that took a hash while vacant left an entry behind, so on
        # a long-lived server the stale ones outnumber the live ones without
        # this. Rebuilding costs one pass and buys at least `num_slots` pushes.
        # 【译】每个在 vacant 期间拿到过 hash 的槽位都会在堆里留下一个条目，所以在长期运行的服务上，不这么做的话过期条目会比
        #   有效条目还多。重建花一次遍历，换来至少 num_slots 次 push 的余量。
        if len(self._vacant) > 2 * self.num_slots + 2:
            self._vacant = [g for g in self._free if self.slot_hash[g] == -1]
            heapify(self._vacant)

    def release_many(self, slots) -> None:
        """Hand back a whole set. Tolerates -1 and an empty list.

        The -1 tolerance is what lets a caller release a request's slots without
        first asking whether it ever got any: a seq that was never admitted, or
        one whose set was already taken over by an adopt, carries the sentinel.
        """
        # 【译】还回一整套。容忍 -1 和空列表。
        #   容忍 -1 让调用方可以释放一个请求的槽位，而不必先问它到底有没有拿到过：从未被准入的序列，或者它那一套已经被
        #   adopt 接管的序列，带的都是这个哨兵值。
        # 调用方：BlockManager.deallocate（一次还回 state_slots 的全部，或 state_slots[1:]）
        for slot in slots or ():
            if slot >= 0:
                self.release(slot)

    def _set_hash(self, slot: int, h: int) -> None:
        """Change what an existing slot backs, re-filing it if it is free.

        Going through here is what lets the two halves of the free list be a
        function of `slot_hash` rather than a third thing to keep in step.
        `h` of -1 means the slot backs nothing.
        """
        # 【译】改变一个已有槽位所支撑的内容；如果它在空闲表上，就把它重新归档。
        #   都从这里走，才能让空闲表的两半是 slot_hash 的函数，而不是又一样需要保持同步的东西。h 为 -1 表示该槽位不支撑任何东西。
        # 实现：空闲时先 claim（拿出来）、改 hash、再 release（按新内容放进对应的一半）；不空闲时直接改
        if slot not in self._free:
            self.slot_hash[slot] = h
            return
        self.claim(slot)
        self.slot_hash[slot] = h
        self.release(slot)

    # ----------------------------- resizing -------------------------------- #
    # 【译】调整大小
    # 【分支】下面三个方法（extend / retire_top 及其辅助）用于在 KV 池与状态池之间挪动边界；v0.1.8 无生产调用方，见 SlotRetirement 上方的观察
    def extend(self, count: int) -> None:
        """Add `count` slot indices at the top. The caller freed the bytes.

        `slot_hash` is grown but never shrunk, so an index that was retired and
        then handed back out reuses its slot rather than appending a second one.
        """
        # 【译】在顶部增加 count 个槽位下标。字节由调用方腾出。
        #   slot_hash 只增不减，所以一个被退役后又被重新发出的下标会复用它原来的那一项，而不是再追加一项。
        for slot in range(self.num_slots, self.num_slots + count):
            if slot < len(self.slot_hash):
                self.slot_hash[slot] = -1
            else:
                self.slot_hash.append(-1)
            self.release(slot)
        self.num_slots += count

    def retire_top(self) -> "SlotRetirement | None":
        """Give up the highest slot index, relocating whatever sits on it.

        Shrinking is index-forced — the bytes being handed back are the ones the
        top slot occupies — but the policy is not: what gets *spent* is the
        least recently used checkpoint, whichever index that is. So the top
        slot's content moves to a target taken in the same order `pop` uses
        (vacant first, LRU checkpoint only if nothing is vacant) and the caller
        issues one copy.

        Without that move, shrinking would be anti-LRU: a slot's index records
        the concurrency high-water mark when it was handed out and is never
        refreshed by use, so a checkpoint resumed from every second could sit at
        the top and be spent ahead of one nothing has touched in minutes.

        Returns `None` when the top slot cannot be given up this pass — it is
        pinned as a fork source, or it is live and there is nowhere to move it.
        Both clear on their own, so the caller retries rather than blocks.
        """
        # 【译】交出最高的槽位下标，把上面放着的东西搬走。
        #   缩池是"下标强制"的——交还的字节就是顶部槽位占的那些——但策略不是：真正被"花掉"的是最久没用的 checkpoint，
        #   不管它在哪个下标。所以顶部槽位的内容被搬到一个按 pop 同样顺序选出的目标（先 vacant，没有 vacant 才是 LRU checkpoint），
        #   调用方发起一次拷贝。
        #
        #   不搬的话，缩池就是"反 LRU"的：槽位的下标记录的是它被发出时的并发高水位，之后不会因使用而刷新，
        #   所以一个每秒都被恢复的 checkpoint 可能正好坐在顶部，先于一个好几分钟没人碰过的被花掉。
        #
        #   顶部槽位这一轮交不出去时返回 None——它被钉住作为 fork 源，或者它正在使用而无处可搬。两者都会自行消失，
        #   所以调用方应重试而不是阻塞。
        # 三种结果：顶部空闲且无内容 → 直接收掉（relocated_to=-1）；顶部有快照或在用 → 找目标搬过去（调用方拷字节）；
        #   哪里都没空位且顶部是快照 → 它就是最后一个快照，按 LRU 本来也该花掉（relocated_to=-1、held=True）
        top = self.num_slots - 1
        if top < 0 or self.is_pinned(top):
            return None
        held = self.holds_checkpoint(top)
        if self.is_free(top) and not held:
            self.claim(top)
            self.num_slots -= 1
            return SlotRetirement(top, -1, False)

        dst = self._take_relocation_target(exclude=top)
        if dst < 0 and not held:
            return None
        if dst < 0:
            # Nothing free anywhere, so the top slot is by elimination the only
            # checkpoint left — spending it is what LRU asks for regardless.
            # 【译】哪里都没有空闲的，所以顶部槽位按排除法就是仅剩的 checkpoint——无论如何 LRU 都要求花掉它。
            self.claim(top)
            self.invalidate(top)
        elif held:
            self._rehome_checkpoint(top, dst)
        self.num_slots -= 1
        return SlotRetirement(top, dst, held)

    def _take_relocation_target(self, exclude: int) -> int:
        """A slot to move content into, in `pop`'s order but skipping `exclude`."""
        # 【译】一个可以把内容搬进去的槽位，顺序同 pop，但跳过 exclude。
        slot = self._pop_vacant()
        if slot >= 0:
            return slot
        for slot in self._checkpointed:
            if slot != exclude:
                self.claim(slot)
                self.invalidate(slot)
                return slot
        return -1

    def _rehome_checkpoint(self, src: int, dst: int) -> None:
        """Give a checkpoint a different slot index, LRU position included.

        Written against the containers rather than through `claim`/`release`
        because keeping the position is the whole point: released slots go to
        the tail, and a checkpoint that only changed address has not been used.
        """
        # 【译】给一个 checkpoint 换一个槽位下标，连同它在 LRU 里的位置一起。
        #   直接对容器操作而不经过 claim/release，因为保住位置正是全部用意：被释放的槽位会去尾部，
        #   而一个只是换了地址的 checkpoint 并没有被使用。
        h = self.slot_hash[src]
        self._checkpointed[self._checkpointed.index(src)] = dst
        self._free.discard(src)
        self._free.add(dst)
        self.slot_hash[src] = -1
        self.slot_hash[dst] = h
        self.hash_to_slot[h] = dst

    # ---------------------------- applicability ---------------------------- #
    # 【译】适用性
    def applies(self, seq) -> bool:
        """Whether this class gates or checkpoints anything for `seq`.

        Per-request state is declared by the attention type, so a seq on a model
        without one carries no slot and this class has no say over its hits.
        """
        # 【译】本类是否对 seq 有门控或做 checkpoint。
        #   每请求状态由注意力类型声明，所以在没有这种状态的模型上，seq 不带槽位，本类对它的命中也没有发言权。
        # 调用方：本类 resumable_hit / reserve_midstep / checkpoint；BlockManager 的 checkpoint_limit、_record_checkpoint_end、
        #   _midstep_applies、checkpoint_cut、checkpointers_at（只在本类是 state_caches 成员即 fork / none / replay 时）；Scheduler._wait_for_inflight_prefix
        return self.enabled and seq.has_per_req_cache

    # ------------------------------- lookup -------------------------------- #
    # 【译】查找
    def resumable_hit(
        self,
        seq,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Shrink a compressed-prefix hit to the nearest recoverable boundary.

        Returns the largest `L <= hit` such that a checkpoint exists for the
        prefix of `L` blocks AND the forward resuming there can leave its own
        slot self-contained, scanning right-to-left so the hit is cut as little
        as possible. 0 is always valid — a request starting from scratch needs
        no prior state.

        The fork test is what `min_fork_tokens` buys: resuming reads the
        checkpoint and writes a fresh slot, and that forward has to leave the
        fresh slot whole. A boundary too close to the end of the prompt fails
        it and the scan keeps walking back.

        Without this a hit hands the resumed forward a slot freshly popped off
        the free list and it reads the previous occupant's state.

        `assume_checkpointed` drops the index lookup and keeps the fork test,
        which is the counterfactual the protocol describes: this class's ladder
        made dense, leaving only what a checkpoint could not have fixed.
        """
        # 【译】把一次压缩前缀命中缩到最近的可恢复边界。
        #   返回最大的 L <= hit，使得"前 L 个 block 的前缀有 checkpoint"并且"从那里恢复的 forward 能让它自己的槽位自给自足"，
        #   从右往左扫，让命中被砍得尽量少。0 总是合法的——从头开始的请求不需要任何先前状态。
        #
        #   fork 测试就是 min_fork_tokens 换来的东西：恢复时读 checkpoint、写一个新槽位，那次 forward 必须把新槽位写完整。
        #   离 prompt 末尾太近的边界通不过，扫描继续往回走。
        #
        #   没有这一步的话，命中会给恢复的 forward 一个刚从空闲表 pop 出来的槽位，它读到的是上一任住户的状态。
        #
        #   assume_checkpointed 去掉索引查找、保留 fork 测试，这就是协议描述的反事实：把本类的阶梯当成"每个边界都有"，
        #   只剩下 checkpoint 也修不了的部分。
        # 输入：hit = KV 侧（含其他门控）给出的命中 block 数；block_hashes = 这条 prompt 每个 hash block 的链式 hash
        # 输出：可以从中恢复的 block 数（≤ hit）。调用方：BlockManager._gated_hit（fork / none / replay 时本类是唯一的 state_caches 成员）
        # 例：hash_block_size=16，seq.num_tokens=20000，hit=1152（18432 个 token），快照只在第 1023 块（16384）：
        #   i=1151 … 1024 都不在 hash_to_slot → 跳过；i=1023 命中，且 20000 − 1024×16 = 3616 ≥ min_fork_tokens(1) → 返回 1024
        #   若同样的快照在 i=1249（19984 处），而 seq 恰好只有 19984 个 token：19984 − 19984 = 0 < 1 → 不能用，继续往左找
        if not self.applies(seq):
            return hit
        if self.transfer.replays:
            # Nothing is kept: the forward resuming at any boundary rebuilds the
            # state from the hash block below it (`StateTransfer.replay`).
            # 【译】什么都不保存：从任意边界恢复的 forward 都会用它下方的那个 hash block 重建状态（StateTransfer.replay）。
            return hit
        hbs = self.hash_block_size
        for i in range(hit - 1, -1, -1):
            checkpointed = block_hashes[i] in self.hash_to_slot
            if not assume_checkpointed and not checkpointed:
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        # 【译】保存着 hash h 的 checkpoint 的槽位，或 -1。
        # 调用方：BlockManager._attach_state_slots（fork 命中时找源槽位）
        if not self.enabled:
            return -1
        return self.hash_to_slot.get(h, -1)

    # ------------------------- midstep checkpointing ------------------------ #
    # 【译】midstep（forward 中途）checkpoint
    #
    # `checkpoint` below keeps state at the position a forward *ended* at, which
    # is the only position a backend can be assumed to expose. A backend that
    # declares `readable_midstep` exposes more: its chunk kernel already
    # materializes the recurrent state at every interior chunk boundary, so a
    # snapshot at any of them is a copy rather than a forward.
    #
    # That splits keeping a checkpoint into two moments that `checkpoint` runs
    # as one:
    #
    #   `reserve_midstep`  before the forward, take the destination slot and
    #                      name the position. The bytes do not exist yet.
    #   `publish_midstep`  after it, once the runner has copied them out, file
    #                      the hash so a resuming request can find it.
    #
    # Publishing at reservation time would index a slot over bytes the forward
    # had not written, and a request resuming there would read whatever the
    # previous tenant left. Reserving at publish time is the other way wrong:
    # the free list has to be committed before the batch is built, or the slot
    # the runner is told to write may have been handed to an admission in
    # between.
    # 【译】下面的 checkpoint 把状态保存在一次 forward"结束"的位置，这是唯一能假定所有后端都暴露出来的位置。
    #   声明了 readable_midstep 的后端暴露得更多：它的 chunk kernel 本来就在每个内部 chunk 边界上物化出递归状态，
    #   所以在其中任何一处拍快照只是一次拷贝，而不是一次 forward。
    #
    #   于是"保存一个 checkpoint"被拆成两个时刻，而 checkpoint 把它们当成一个：
    #     reserve_midstep  forward 之前：拿到目标槽位、确定位置。此时字节还不存在。
    #     publish_midstep  forward 之后：runner 把字节拷出来以后，登记 hash，让恢复的请求能找到它。
    #
    #   在预留时就发布，会把一个槽位索引到 forward 还没写的字节上，从那里恢复的请求会读到上一任住户留下的东西。
    #   在发布时才预留则是反方向的错：空闲表必须在构造 batch 之前就承诺出去，否则告诉 runner 去写的那个槽位
    #   可能在中间被发给了某个准入的请求。
    # 【分支】v0.1.8 中声明 readable_midstep=True 的只有 copy 方式的 GDN（ATOM_GDN_STATE_MIDSTEP=1，默认关）；
    #   fork 后端都声明 False（gdn_attn.py / kimi_mla_gdn_attn.py 的 state_transfer），所以本类这三个方法实际总是空操作，
    #   生效的是 page_unit_checkpoint.py 里的同名方法（它借本类的槽位当 scratch）。调用方：BlockManager.plan_midstep / commit_midstep / cancel_midstep

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        """Take a destination slot for each `(position, hash)` this step covers.

        Returns `(slot, position, hash)` per reservation actually made — the
        runner needs `slot` and `position`, `publish_midstep` needs the hash.
        Best-effort and order-preserving: reservations stop at the first one the
        free list cannot fill, so the earliest position (the one an earlier
        forward reaches first) is the one that survives a shortage.

        Nothing is indexed here. `pop` takes the slot off the free list, so an
        admission in the same pass cannot be handed it, and it carries no hash
        until `publish_midstep`, so a resuming request cannot find it either.

        A reservation therefore holds capacity that a `fork` checkpoint does
        not — that hands a slot back the moment it takes one. It is held for
        exactly one forward, and `cancel_midstep` returns it if that forward
        never runs. `has_free` is still the admission gate, so the worst case is
        an admission deferred a step, never one starved: the same best-effort
        contract `checkpoint` keeps.
        """
        # 【译】为这一步覆盖到的每个 (position, hash) 拿一个目标槽位。
        #   对每个实际做成的预留返回 (slot, position, hash)——runner 需要 slot 和 position，publish_midstep 需要 hash。
        #   尽力而为、保持顺序：预留在第一个空闲表填不满的位置处停止，所以短缺时留下的是最早的位置（更早的 forward 先到达的那个）。
        #
        #   这里不登记任何索引。pop 把槽位从空闲表上拿走，所以同一轮里的准入拿不到它；它在 publish_midstep 之前不带 hash，
        #   所以恢复的请求也找不到它。
        #
        #   因此预留占着容量，而 fork 的 checkpoint 不占——那边一拿到槽位马上就还回一个。预留恰好只占一次 forward，
        #   如果那次 forward 没跑，cancel_midstep 会把它还回去。has_free 仍然是准入门，所以最坏情况是某个准入被推迟一步，
        #   绝不会饿死：和 checkpoint 遵守的是同一个"尽力而为"约定。
        if not self.applies(seq) or not self.readable_midstep:
            return []
        out = []
        for pos, h in positions:
            if not self.has_free():
                self.checkpoints_dropped += 1
                break
            out.append((self.pop(), pos, h))
        return out

    def publish_midstep(self, reservations: list[tuple], seq=None) -> None:
        """File each reserved slot under its hash, now that its bytes exist.

        The slot goes back on the free list *as a checkpoint* — the same
        capacity-neutral move `checkpoint` makes, covered by the same lazy
        eviction: whoever pops it next invalidates the hash on the way out.

        A position that is not this prompt's own end is marked speculative and
        goes to the LRU head — the ladder rung and the demand guess where the
        next turn will resume, the anchor knows; `mark_speculative` gives what
        the difference measures.

        A seq with no anchor at all has *only* guesses, so all of them demote.
        Reading `anchor == 0` as "demote nothing" is the inversion this exists
        to prevent: `_record_checkpoint_end` leaves the anchor at 0 on four
        paths, including every prompt too short to have a keepable end — the
        majority of agentic first turns — and those seqs would then file ladder
        and demand rungs at the LRU tail beside the genuine anchors of long
        prompts, which are read back 85.2% of the time against their 2.8%.
        Under pressure the pool would spend the anchors first.

        `seq=None` demotes nothing, which is the other end of the same rule
        rather than an exception to it: a caller with no sequence in hand
        cannot tell a guess from knowledge, and over-keeping is the recoverable
        error where over-demoting is not.
        """
        # 【译】既然字节已经存在了，就把每个预留的槽位登记到它的 hash 名下。
        #   槽位"作为 checkpoint"回到空闲表——和 checkpoint 做的是同一个不占容量的动作，由同一套惰性淘汰兜底：
        #   下一个 pop 到它的人在拿走时让 hash 失效。
        #
        #   不是本 prompt 自己末尾的位置被标为猜测并放到 LRU 头部——阶梯和按需快照猜下一轮会从哪里恢复，锚点知道；
        #   两者差多少见 mark_speculative。
        #
        #   完全没有锚点的序列只有猜测，所以全部降级。把 anchor == 0 理解为"什么都不降级"正是这里要防止的颠倒：
        #   _record_checkpoint_end 在四条路径上把锚点留为 0，包括每一个短到没有可保存末尾的 prompt——这占了 agent 首轮的大多数——
        #   那样这些序列会把阶梯和按需快照放在 LRU 尾部，和长 prompt 的真锚点挨着；而真锚点有 85.2% 会被读回，它们只有 2.8%。
        #   有压力时池子就会先花掉锚点。
        #
        #   seq=None 什么都不降级，这是同一条规则的另一端而不是例外：手上没有序列的调用方分不清猜测和知道，
        #   多保留是可以挽回的错误，多降级则不是。
        demotes = seq is not None
        known = self._anchor_of(seq)
        for slot, pos, h in reservations:
            if demotes and pos != known:
                self.mark_speculative(slot)
            self._index(h, slot)
            self.release(slot)
            self.checkpoints_kept += 1

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Hand back reservations whose forward never ran (preempt, abort).

        Released vacant, not indexed: the bytes were never written, so the slot
        holds nothing anybody should be able to find.
        """
        # 【译】还回那些 forward 没有跑（被抢占、被中止）的预留。
        #   作为 vacant 释放、不登记索引：字节从没写过，所以槽位里没有任何应该被人找到的东西。
        for slot, _pos, _h in reservations:
            self.release(slot)

    # ---------------------------- checkpointing ---------------------------- #
    # 【译】checkpoint
    def checkpoint(self, seq, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of this boundary, filed under hash `h`.

        The owner moves to a fresh slot and the old slot becomes read-only.
        Only the committed slot moves: the owner keeps its rollback scratch, and
        the checkpoint is one slot wide because a resuming prefix has no
        speculation to roll back.

        `boundary_blocks` names the position, which is all this needs it for:
        whether the position is this prompt's own end decides where the
        checkpoint sits in the LRU order — see `mark_speculative`.
        """
        # 【译】把 seq 在这个边界处的状态保存下来，登记在 hash h 名下。
        #   拥有者换到一个新槽位，旧槽位变成只读。只有已提交槽位会挪：拥有者保留它的回滚临时槽位，
        #   checkpoint 只有一个槽位宽，因为恢复出来的前缀没有需要回滚的投机。
        #
        #   boundary_blocks 只用来说明位置，这里只需要它做一件事：这个位置是否就是本 prompt 自己的末尾，
        #   决定了 checkpoint 在 LRU 顺序里排在哪——见 mark_speculative。
        # 调用方：BlockManager.hash_blocks（forward 恰好结束在阶梯 / 按需 / 锚点位置，且 checkpointers_at 认可本类时，
        #   调 cache.checkpoint(seq, end, h)，h = 最后一个 block 的链式 hash）。只在 fork 方式下真正做事
        # 步骤（例：seq.state_slot=3，boundary_blocks=512，hash_block_size=16 → 位置 8192）：
        #   1. 没有空闲槽位就放弃（checkpoints_dropped += 1），后果只是以后某次命中会被拒
        #   2. seq.state_slot = pop()（例如 5），seq.state_fork_src = 3：下一次 forward 读 3、写 5
        #   3. _index(h, 3)：hash_to_slot[h] = 3；不是锚点位置就 mark_speculative(3)
        #   4. pin(3, reader_is_next_batch=True)：两轮之后才由 release_pins 放回空闲表（此后才可能被别人 pop 掉）
        if not self.applies(seq) or not self.transfer.forks:
            return
        old = seq.state_slot
        if old < 0:
            return
        if not self.has_free():
            self.checkpoints_dropped += 1
            return
        known = self._anchor_of(seq)
        guess = known is None or boundary_blocks * self.hash_block_size != known
        seq.state_slot = self.pop()
        seq.state_fork_src = old
        # Not released: `state_fork_src` names `old` as what this request's
        # NEXT forward reads, and that forward is two passes off — one to build
        # the batch carrying the fork, one to issue it. Handing it back now put
        # it on the free list during the pass that admits the requests which
        # could pop it, and then one kernel would read and write it at once.
        # `_index` files it either way; `_set_hash` leaves a non-free slot be.
        # 【译】不释放：state_fork_src 把 old 标为这个请求"下一次" forward 要读的东西，而那次 forward 还隔着两轮——一轮构造
        #   带着 fork 的 batch，一轮把它发出去。现在就还回去，会让它在"准入可能 pop 到它的请求"的那一轮里出现在空闲表上，
        #   然后一个 kernel 会同时读写它。_index 无论如何都会登记它；_set_hash 对不在空闲表上的槽位不做挪动。
        self._index(h, old)
        if guess:
            self.mark_speculative(old)
        self.pin(old, reader_is_next_batch=True)
        self.checkpoints_kept += 1

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of the checkpoints the ladder asked this pool to keep.

        `checkpoints_dropped` is not the capacity signal it looks like: a
        finished request hands its slot back, and `pop` never refuses — with
        nothing vacant it spends the LRU checkpoint and counts an eviction
        instead. This pool overwrites checkpoints, it does not turn them away.

        The capacity reading is `checkpoints_evicted` against
        `checkpoints_kept - num_slots`. When they match, every checkpoint but
        the resident set was destroyed to make room and more slots will buy
        hit rate directly. When eviction runs below that line, the pool is
        holding its working set and the misses are somewhere else — most
        likely `checkpoints_orphaned`, whose KV prefix died first and which
        more slots cannot prevent.
        """
        # 【译】阶梯要求本池保存的那些 checkpoint 后来怎样了。
        #   checkpoints_dropped 并不是它看上去那样的容量信号：结束的请求会还回它的槽位，而 pop 从不拒绝——没有 vacant 时它花掉
        #   LRU checkpoint 并记一次淘汰。本池是覆盖 checkpoint，而不是拒收它们。
        #
        #   容量的读法是拿 checkpoints_evicted 和 checkpoints_kept - num_slots 比。两者相等时，除了常驻的那一批，
        #   每个 checkpoint 都是为腾地方被毁掉的，多给槽位能直接换来命中率。淘汰低于这条线时，池子守住了它的工作集，
        #   未命中出在别处——最可能是 checkpoints_orphaned，它们的 KV 前缀先死了，再多槽位也救不了。
        # 调用方：BlockManager.state_checkpoint_fates（对每个 state_caches 成员 getattr 后求和，进周期性的 "state checkpoints:" 日志）
        return {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.checkpoints_evicted,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }

    def occupancy(self) -> dict[str, int]:
        """How much of this pool is live, held by checkpoints, or spare.

        Separate from `checkpoint_fates` because those are cumulative events
        and these are an instantaneous reading.

        `slots_vacant` at 0 is not a shortage. A slot with no live owner sits
        on the free list either way; vacant just means it also has no content
        left to lose. A steady state of all-held-none-vacant is this pool
        working, and admission never consults the split — `has_free` counts the
        whole free list.

        Nor is any field here the reading that says "too small" — see
        `checkpoint_fates`, which owns that question. `kept - num_slots ==
        evicted` is the thrash signal.
        """
        # 【译】本池有多少是在用的、被 checkpoint 占着的、或空闲的。
        #   和 checkpoint_fates 分开，因为那些是累计事件，这些是瞬时读数。
        #
        #   slots_vacant 为 0 不是短缺。没有活主人的槽位无论如何都在空闲表上；vacant 只是说它连可丢的内容都没有了。
        #   "全部被占着、一个都不 vacant"的稳态正是本池在正常工作，而准入从不看这个划分——has_free 数的是整个空闲表。
        #
        #   这里也没有哪个字段是"太小"的读数——见 checkpoint_fates，那个问题归它。kept - num_slots == evicted 才是抖动信号。
        # 调用方：BlockManager.pool_pressure（并进 KV 池的占用信息一起报告）
        held = sum(1 for g in self._free if self.slot_hash[g] != -1)
        return {
            "slots_total": self.num_slots,
            "slots_used": self.num_slots - len(self._free),
            "slots_held": held,
            "slots_vacant": len(self._free) - held,
        }

    def record_relocation(self, src: int, dst: int) -> None:
        """Schedule an Active Slot relocation for the next batch."""
        # 【译】为下一个 batch 安排一次活跃槽位搬迁。
        self._relocations.append((src, dst))

    def take_relocations(self) -> tuple[tuple[int, int], ...]:
        # 取走并清空待搬迁列表；调用方：BlockManager.take_state_maintenance_ops（放进 StateMaintenanceOps.relocations，
        #   由注意力 builder 在 forward 前执行 relocate_state_slots，见 model_ops/attentions/backends.py）
        relocations, self._relocations = self._relocations, []
        return tuple(relocations)

    def _index(self, h: int, slot: int) -> None:
        """File `slot` as the checkpoint for hash `h`.

        Caller guarantees `slot` holds the state as of that boundary and has
        been handed back to the free list — a checkpointed slot is never
        written again, so the request that produced it must have moved on.
        """
        # 【译】把 slot 登记为 hash h 的 checkpoint。
        #   调用方保证 slot 保存的是到那个边界为止的状态，并且已经被还回空闲表——被 checkpoint 的槽位不会再被写，
        #   所以产生它的请求必须已经挪走了。
        # 解释：fork 路径（checkpoint）调用时 slot 还被钉着、不在空闲表上，由 release_pins 稍后归还；docstring 的"已还回"只对 publish_midstep 严格成立
        if not self.enabled:
            return
        self.invalidate(slot)
        prev = self.hash_to_slot.get(h, -1)
        self.hash_to_slot[h] = slot
        self._set_hash(slot, h)
        # Re-filing a hash onto a different slot orphans the old one. Drop its
        # back-pointer directly rather than through `invalidate`, which would
        # take the entry just written with it.
        # 【译】把一个 hash 重新登记到另一个槽位，会让旧槽位成为孤儿。直接删掉它的反向指针，而不是经过 invalidate——
        #   那样会把刚写进去的这一项一起带走。
        if prev != -1 and prev != slot:
            self._set_hash(prev, -1)

    def unindex(self, h: int) -> None:
        """Drop the checkpoint filed under `h`. The dual of `_index`.

        Called when the KV block of that hash leaves the block index. The two
        pools are addressed by the same chained content hash and a prefix hit
        is a joint claim on both, so a checkpoint whose block is gone can never
        be reached again: `_gated_hit` caps at the last block still indexed.
        Until it is dropped it holds a slot and sits in the LRU queue ahead of
        checkpoints that are still worth something, so the pool spends a live
        one to make room for a dead one.

        Necessary but not sufficient for reachability — an *earlier* block in
        the chain may be the one that went — so this reclaims a subset. Making
        it exact would have the state pool watch every block of every prefix,
        which costs more than the tail it would catch.

        """
        # 【译】删掉登记在 h 名下的 checkpoint。它是 _index 的对偶。
        #   在该 hash 的 KV block 离开 block 索引时调用。两个池子用同一个链式内容 hash 寻址，一次前缀命中是对两者的联合声明，
        #   所以 block 已经没了的 checkpoint 再也不可能被够到：_gated_hit 最多截到最后一个仍在索引里的 block。
        #   在被删掉之前，它占着一个槽位、排在 LRU 队列里，挡在还有价值的 checkpoint 前面，于是池子会花掉一个活的去给一个死的腾地方。
        #
        #   这是"可达"的必要条件而不是充分条件——链上"更早"的某个 block 也可能已经没了——所以这里只回收一个子集。
        #   要做到精确，就得让状态池盯住每个前缀的每个 block，代价比它能抓到的尾巴还大。
        # 调用方：BlockManager._record_evicted（BlockPool 的 on_evict 回调：KV block 的 hash 在惰性淘汰中被丢弃时）
        slot = self.hash_to_slot.get(h, -1)
        if slot < 0:
            return
        self.invalidate(slot)
        self.checkpoints_orphaned += 1

    def clear_index(self) -> None:
        """Drop all checkpoint hashes, preserving only in-flight readers."""
        # 【译】丢掉所有 checkpoint hash，只保留正在进行中的读者。
        # 调用方：BlockManager.clear_cache（管理操作；v0.1.8 的 atom/ 里只有测试调用）；被钉住的源仍不在空闲表上，release_pins 照常归还
        for slot in list(self.hash_to_slot.values()):
            self.invalidate(slot)

    def invalidate(self, slot: int) -> None:
        """Drop `slot`'s checkpoint. Called when the slot is handed out."""
        # 【译】丢掉 slot 的 checkpoint。在槽位被发出时调用。
        # 调用方：本类 pop / _index / unindex / clear_index / retire_top；BlockManager._attach_state_slots（adopt 时）、cancel_state_fork
        if not self.enabled:
            return
        h = self.slot_hash[slot]
        if h == -1:
            return
        self._set_hash(slot, -1)
        if self.hash_to_slot.get(h) == slot:
            del self.hash_to_slot[h]

    # -------------------------------- pins --------------------------------- #
    # 【译】钉住（pin）
    # 时间线（一格 = 一轮 Scheduler.schedule；release_pins 在每轮开头由 BlockManager.complete_previous_state_batch 调用）：
    #   恢复（resume）：  第 N 轮准入时 pin(src)            → 第 N 轮构造的 batch 读 src → 第 N+1 轮开头 release_pins 归还
    #   checkpoint：     第 N 轮 forward 后 postprocess 里 pin(old, reader_is_next_batch=True)
    #                    → 第 N+1 轮开头 release_pins 跳过（只清 _deferred）→ 第 N+1 轮的 batch 读 old → 第 N+2 轮开头归还
    def pin(self, slot: int, *, reader_is_next_batch: bool = False) -> None:
        """Hold `slot` off the free list until the forward that reads it ran.

        The caller states which batch that is, not when to let go: a resume
        pins while its own batch is being built, `checkpoint` pins after its
        batch went out and the forward that reads what it handed over is the
        one the next pass builds. `release_pins` turns the two into passes.
        """
        # 【译】让 slot 不进空闲表，直到读它的那次 forward 已经跑过。
        #   调用方说明的是"哪个 batch 读它"，而不是何时放手：恢复在构造自己的 batch 时钉住；checkpoint 在它的 batch 发出之后钉住，
        #   读它交出之物的 forward 是下一轮构造的那个。release_pins 把这两种情况换算成轮数。
        # 调用方：BlockManager._attach_state_slots（fork 恢复）；本类 checkpoint
        self._pinned[slot] = self._pinned.get(slot, 0) + 1
        if reader_is_next_batch:
            self._deferred.add(slot)

    def is_pinned(self, slot: int) -> bool:
        return slot in self._pinned

    def pin_count(self, slot: int) -> int:
        """Requests forking off `slot` in the in-flight step."""
        # 【译】在进行中的这一步里从 slot fork 的请求数。
        # 调用方：BlockManager.cancel_state_fork（>1 时不能接管）
        return self._pinned.get(slot, 0)

    def unpin(self, slot: int) -> None:
        """Drop one reference without freeing — the caller took the slot over.

        Only legal on the last reference; a slot another request still reads
        cannot be taken over. `BlockManager.cancel_state_fork` enforces that.
        """
        # 【译】去掉一个引用但不释放——调用方接管了这个槽位。
        #   只在最后一个引用上合法；别的请求还在读的槽位不能被接管。BlockManager.cancel_state_fork 保证这一点。
        count = self._pinned.get(slot, 0)
        if count <= 1:
            self._pinned.pop(slot, None)
            self._deferred.discard(slot)
        else:
            self._pinned[slot] = count - 1

    def drop_reader(self, slot: int) -> None:
        """A reader of `slot` is gone before it read. Free it if it was last.

        A pin says some forward still has to read this slot. When the request
        that owed that forward is deallocated the obligation goes with it, and
        holding the slot to the clock would cost admission a slot for
        nothing — which is what keeps `checkpoint` capacity-neutral for a
        request that finishes or is preempted on the boundary it published.
        `release_pins` stays the backstop, for readers still to come.
        """
        # 【译】slot 的一个读者在读之前就走了。如果它是最后一个，就释放 slot。
        #   pin 表示还有某次 forward 要读这个槽位。欠这次 forward 的请求被 deallocate 时，这个义务随之消失；还按时钟扣着槽位，
        #   只会白白让准入少一个槽位——这正是让 checkpoint 对于"恰好在它发布的边界上结束或被抢占"的请求保持不占容量的原因。
        #   release_pins 仍然是兜底，用于还没到来的读者。
        # 调用方：BlockManager.deallocate（drop_reader(seq.state_fork_src)）
        if slot < 0 or not self.is_pinned(slot):
            return
        self.unpin(slot)
        if not self.is_pinned(slot):
            self.release(slot)

    def release_pins(self) -> None:
        """Return the sources whose reading forward has been issued.

        They go back to the free list, once each however many requests read
        them. Safe once that forward is out: a request handed one of these
        slots next pass runs its own forward after it, on the same stream.

        A pin taken while a batch was being built is read by that batch, so it
        clears here, one pass later. A pin marked `reader_is_next_batch` is
        read by the batch this pass is about to build, so it survives one more
        — and `checkpoint`'s source has to, because it is on nobody's free
        list to protect it during the very pass that admits the requests which
        could otherwise be handed it.

        Every pin clears within two passes whatever else happens. Nothing here
        waits on the obligation being consumed, so a request preempted between
        taking a checkpoint and its next batch cannot strand a slot — which
        is why this stays a clock rather than moving to `_consume_state_forks`,
        where it would be tighter and leak.
        """
        # 【译】归还那些"读它们的 forward 已经发出"的源。
        #   它们回到空闲表，不管有多少个请求读过，每个只还一次。forward 一发出就安全了：下一轮拿到这些槽位之一的请求，
        #   它自己的 forward 会排在后面、在同一个流上执行。
        #
        #   在构造某个 batch 时取的 pin 由那个 batch 读，所以在这里清掉，晚一轮。标了 reader_is_next_batch 的 pin 由
        #   这一轮即将构造的 batch 读，所以要多活一轮——checkpoint 的源必须如此，因为它不在任何人的空闲表上来保护它，
        #   而正是这一轮会准入那些本来可能被发到它的请求。
        #
        #   无论发生什么，每个 pin 都在两轮内清掉。这里不等待义务被消费，所以在拿了 checkpoint 和下一个 batch 之间被抢占的请求
        #   不会让槽位搁浅——这就是它保持为一个"时钟"、而不是挪到 _consume_state_forks 里的原因：挪过去会更紧凑，但会泄漏。
        # 调用方：BlockManager.complete_previous_state_batch（Scheduler.schedule 每轮开头）
        if not self._pinned:
            return
        held, self._deferred = self._deferred, set()
        for slot in sorted(self._pinned):
            if slot in held:
                continue
            self.release(slot)
            del self._pinned[slot]
