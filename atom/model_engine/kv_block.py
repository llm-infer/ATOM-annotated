# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】kv_block.py：一个物理 KV block 在 CPU 侧的账本记录（Block），以及状态槽位子池的名字（STATE_SLOT_CLASS）
#   位置：Scheduler → BlockManager（block_manager.py）→ BlockPool（block_pool.py：blocks[i] = Block(i)）→ 本文件
#   前置：分页 KV / block table → sequence.py BlockTable类上方"科普：KV cache分块"；
#         hash 字段里存的是什么（链式 hash）→ prefix_hash.py 文件头【科普】①②；
#         这些记录怎样在空闲表 / 使用中 / hash 索引之间流转 → block_pool.py 文件头的账本图和数字例子
#
# 一条 Block 记录对应 GPU 上的什么（GPU 侧只认 block 号，不知道这些记录存在）：
#
#   GPU：ModelRunner 启动时按 num_kvcache_blocks 一次分配，            CPU：调度器进程（本文件）
#        每个 KV 层一块按 block 号分页的缓存，每页装 block_size 个 token
#   第 0 层   ┌────┬────┬───┬────┬───┐                                blocks[7] = Block(7)
#            │ p0 │ p1 │ … │ p7 │ … │                                  ├ block_id  = 7     ← 就是左边的页号，也是 block_table 里存的数
#   第 1 层   ├────┼────┼───┼────┼───┤                                  ├ ref_count = 2     ← 有 2 条序列的 block_table 里有 7
#     …      │    │    │   │    │   │                                  ├ hash      = h1    ← 链式内容 hash，-1 表示未登记
#   第 L-1 层 └────┴────┴───┴────┴───┘                                  └ token_ids = 这一页装的 16 个 token（位置 16~31）
#   "block 7" = 每一层的第 7 页：BlockManager 分出一个 block，就是在所有层各占一页
#   （开 DCP 时每张卡的第 7 页只装自己那份 token，一条记录覆盖 block_size × DCP 个全局 token，见 prefix_hash.py ④）
#
# 例（block_size=16）：请求 A、B 共享开头 32 个 token 的 system prompt，物理 block 3 装位置 0~15、block 7 装位置 16~31：
#   A.block_table = [3, 7, 12]，B.block_table = [3, 7, 9]  →  blocks[7].ref_count = 2，
#   blocks[7].hash = H(blocks[3].hash, 位置16~31的token)，blocks[7].token_ids = 位置 16~31 的 token
#
# 一个 block 的几种状态（容器名是 BlockPool 的成员）：
#   状态                              ref_count   hash   token_ids   在哪个容器
#   空，从未登记过                         0         -1     空          _free 且在 _vacant 堆里
#   使用中，还没写满 / KV 还没算好          ≥1        -1     空          _used
#   使用中，已登记（publish）              ≥1        h      本页 token   _used，且 _hash_to_block_id[h] = 本号
#   空闲但内容仍可被命中（惰性淘汰）         0         h      本页 token   _free 且在 _cached（slru 下或 _protected）里
#   被重新分配给新内容（= 淘汰）             1         -1     空          _used；BlockPool.allocate 先删索引条目再 reset
#
# 对比nano-vllm：字段和方法与 nano-vllm 的 Block 一一对应（block_id / ref_count / hash=-1 / token_ids，update / reset 语义相同）。
#   区别只有两点：① token_ids 用 array('i') 而不是 list，并在 update 里强制这个类型（原因见 update 里的注释）；
#   ② 单独成一个模块（见 docstring）。复杂的部分——谁能拿走一个 block、什么时候淘汰——全在 BlockPool
import array


class Block:
    """One physical KV cache block of the compressed pool (BlockManager).

    Lives in its own module because `block_pool.py` and `block_manager.py` both
    need it and would otherwise import each other."""

    # 【译】BlockManager 的压缩池中的一个物理 KV cache block。
    #
    #   单独放在一个模块里，是因为 block_pool.py 和 block_manager.py 都需要它，否则二者会互相导入。
    # "压缩池"的叫法来自 DeepSeek-V4：它的分页池里存的是压缩后的 KV 行（滑动窗口那部分在每请求的环形缓冲区里，
    #   见 block_pool.py BlockPool 的 docstring）。对普通 MHA / GQA / MLA 模型，它就是唯一的分页 KV 池 BlockManager.kv

    # 由 BlockPool.__init__ 一次建好全部记录（blocks = [Block(i) for i in range(max_blocks)]），之后只改字段、不新建
    def __init__(self, block_id):
        # 物理 block 号 = GPU 上 KV 缓存的页号，也就是 block_table 里存的数。终身不变：
        #   【分支】池子收缩时 BlockPool._adopt 是把"身份"（ref_count / hash / token_ids）搬到另一条记录上，而不是改这个号
        self.block_id = block_id
        # 有几个持有者正在引用它：通常是序列的 block_table（前缀命中时多条序列共享同一个 block，>1），
        #   【分支】也可能是 offload 保存的租约（BlockManager.deallocate_partial）。0 = 空闲（在 BlockPool._free 里）
        self.ref_count = 0
        # 链式内容 hash（prefix_hash.block_hash）。xxh64 的结果在 [0, 2^64) 里，所以 -1 永远不是合法 hash，用作"未登记"。
        #   只有写满、且 KV 已经算好的 block 才登记（BlockManager.hash_blocks → BlockPool.publish）；
        #   空闲后仍保留，直到这一页被分给别的内容（惰性淘汰，见 block_pool.py）
        self.hash = -1
        # A slice of `Sequence.token_ids`, kept only so a hash hit can be
        # checked against the tokens it claims to stand for. See `update` for
        # why the type is pinned.
        # 【译】Sequence.token_ids 的一个切片，保存它只是为了让一次 hash 命中能和它声称代表的 token 核对。
        #   为什么固定成这个类型，见 update。
        # 用途：防 64 位 hash 碰撞——BlockManager 查到 hash 后再比较 token_ids（比哪些块见 prefix_hash.py ③）。
        #   Python 语法：array.array("i") 是紧凑的 C int32 数组（一块连续内存），不是一串 Python int 对象
        self.token_ids: array.array = array.array("i")

    # block 写满、KV 算好之后登记内容：调用方是 BlockPool.publish（BlockManager.hash_blocks 经它调用），
    #   以及【分支】BlockManager.publish_loaded_prefix（offload 加载回来的前缀）
    def update(self, hash: int, token_ids: array.array):
        # Pinned rather than accepted as either: a list here fails twice and
        # neither failure says so. It never compares equal to the `array("i")`
        # the other publish paths store, so every hit on this block reads as a
        # collision; and a list of ids is one traversal slot per token for the
        # collector, which across a full pool is a stop-the-world pause of
        # ~200ms per gen-2 pass against ~5ms for arrays.
        # 【译】固定成这个类型，而不是两种都接受：这里传 list 会错两次，而且哪一次都不报错。
        #   它和其他登记路径存的 array("i") 永远不相等，于是这个 block 上的每次命中都会被当成碰撞；
        #   而且一个 id 列表对垃圾回收器来说是每个 token 一个遍历槽位，整个池子加起来，每次第 2 代回收
        #   是约 200ms 的全停顿，用 array 时约 5ms。
        # 解释：① array('i', [1, 2]) == [1, 2] 的结果是 False，前缀缓存会静默失效（每次都"碰撞"、都重算）；
        #   ② list 是容器对象，CPython 的分代 GC 每次第 2 代回收都要逐个访问它的元素；array 内部是原始内存，没有可遍历的引用。
        #   200ms / 5ms 是上游注释给的量级
        assert isinstance(
            token_ids, array.array
        ), f"Block.token_ids must be an array('i'), got {type(token_ids).__name__}"
        self.hash = hash
        self.token_ids = token_ids

    # 分给新内容时重置：调用方是 BlockPool.allocate（pop 出一个空闲块之后）。ref_count 直接置 1——拿走它的那个持有者；
    #   hash 的索引条目不在这里删，由 allocate 先调 _unindex 删掉（只 reset 不删的话，索引会指向一个内容已变的块）。
    #   与 nano-vllm 的 Block.reset 相同
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = array.array("i")


# Name of the sub-pool sizing class backing the per-request state slots
# (GDN/Mamba recurrent state, the DeepSeek-V4 compressor ring). Owned next to
# the block/slot machinery that consumes the count, not in the sizing layer.
# 【译】支撑每请求状态槽位（GDN / Mamba 的递归状态、DeepSeek-V4 的压缩器环形缓冲）的子池规模类别名。
#   放在消费这个数量的 block / 槽位机制旁边，而不是放在规模计算层里。
# 前置：线性注意力层没有逐 token 的 KV，每条请求只有一份定长状态（Qwen3-Next 一层 GDN ≈ 52 万个数），
#   所以按"槽位"而不是按 block 分配（规范讲解见 state_pool.py 文件头【科普】②④(a)；概念概要见 arg_utils.py
#   EngineArgs.state_checkpoint_interval_tokens 上方的【科普】）。
# 用法：注意力后端声明子池 state_pool(STATE_SLOT_CLASS, 每槽字节数, entries_per_req=…)（gdn_attn.py、
#   deepseek_v41/backend.py 等）；ModelRunner 按声明切分显存，得到 {类别名: 条目数}，EngineCore 回填到
#   config.pool_entries；BlockManager.__init__ 读 pool_entries[STATE_SLOT_CLASS] 得到状态槽位总数 num_state_slots
STATE_SLOT_CLASS = "state"
