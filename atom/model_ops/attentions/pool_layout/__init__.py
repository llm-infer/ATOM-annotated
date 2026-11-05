# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a byte lives in the cache pools: sizing, addressing, and moving.

One topic across eight modules -- how many entries a byte budget buys
(`sub_pool_spec`), which row of the unified pool a DeepSeek-V4 layer's window
or compressed group occupies (`v4_pool_geometry`), what one of those rows and
one indexer block hold (`v4_pool_fields`), where a checkpoint image's bytes
land in the MLA paged pool (`page_unit_geometry`) or in an MHA paged pool
(`mha_page_unit_geometry`), what one entry holds and where those bytes are put
(`entry_arena`), what the row spaces are and which row of one each module was
given (`pool_rows`), and how one entry's run is scattered across PAGE units
and gathered back (`paged_state_copy`). All of it answers a question about the
pool, none of it about a particular step.

`..token_layout` is the other axis, and the two share a membership rule with
three parts.

**By topic.** The axis decides, not the dependencies. Half this tree imports
neither aiter nor atom without belonging anywhere near here.

**Arithmetic, not staging.** A member computes an answer; it does not write a
`forward_vars` mirror or upload one. `page_unit_geometry` is a mixin over
`self.model_runner` and still qualifies because it only reads -- taking `self`
is not the line, having a side effect on the step's buffers is. Move a stager
in and the package becomes a second builder.

**Reachable without aiter or the rest of atom**, so a member is importable on a
runner with no AITER build and no GPU. CI is such a runner, and one import
failure there aborts collection for every test rather than one, so
`tests/test_layout_packages.py` enforces this part over both packages. It is
the part a test can check, which is exactly why it must not be mistaken for the
whole rule.

Not everything on this topic is here, and one absence is worth naming:
`v4_kernels/pool_index.py` is `v4_pool_geometry`'s device-side half, the same
row formulas as `@triton.jit` device functions for the eight kernels that
address the pool. It stays with the kernels -- the split is by execution tier,
which is a real axis too -- but a layout change lands in both places, and
`tests/test_pool_index.py` is what pins the two sides together.

Nothing is re-exported here: a convenience import would pull all eight in
whenever one is wanted. A member importing a *sibling* is a different thing
and is allowed -- the sibling obeys the same rule, so nothing new is reached,
and a declaration is free to sit apart from the arithmetic over it.
"""

# 【译】一个字节在缓存池里的什么位置：定大小、寻址、搬运。
#
#   一个主题，分在八个模块里——一份字节预算能买多少个条目（sub_pool_spec）；DeepSeek-V4 某一层的
#   滑动窗口或压缩组占统一池的哪一行（v4_pool_geometry）；这样的一行、以及一个索引器 block 里装着什么
#   （v4_pool_fields）；一份 checkpoint 镜像的字节落在 MLA 分页池（page_unit_geometry）或 MHA 分页池
#   （mha_page_unit_geometry）的什么地方；一个条目装什么、这些字节放在哪（entry_arena）；有哪些行空间、
#   每个模块分到其中哪一行（pool_rows）；以及一个条目的一段连续字节怎样分散写进 PAGE 单元、再收集回来
#   （paged_state_copy）。这些回答的都是关于"池"的问题，没有一个是关于某一步（step）的。
#
#   ..token_layout 是另一条轴；两个包共用一条由三部分组成的"成员资格规则"。
#
#   按主题。决定一个模块属不属于这里的是它所在的轴，而不是它的依赖。这棵目录树里有一半的文件既不
#   import aiter 也不 import atom，但它们根本不该放到这里。
#
#   只做算术，不做暂存（staging）。成员只计算一个答案；它不写 forward_vars 的 CPU 镜像，也不上传它。
#   page_unit_geometry 是架在 self.model_runner 上的 mixin，但仍然符合条件，因为它只读——界线不在于
#   "是否接收 self"，而在于"是否对这一步的缓冲区产生副作用"。把一个做暂存的东西搬进来，这个包就成了
#   第二个 builder。
#
#   不依赖 aiter、也不依赖 atom 的其余部分就能导入，所以在没有 AITER 构建、没有 GPU 的机器上也能
#   import 成员。CI 就是这样的机器，在那里一次 import 失败会让所有测试的收集都中止，而不只是一个；所以
#   tests/test_layout_packages.py 对两个包都强制检查这一条。它是测试能检查的那一部分——也正因为如此，
#   不能把它误当成整条规则。
#
#   这个主题的东西并不全在这里，有一处缺席值得点名：v4_kernels/pool_index.py 是 v4_pool_geometry 在
#   device 侧的另一半——同样的行号公式写成 @triton.jit 的 device 函数，供寻址这个池的八个 kernel 使用。
#   它留在 kernel 那边（这是按执行层级划分的，这也是一条真实的轴），但布局一改两处都要改；把两边钉在
#   一起的是 tests/test_pool_index.py。
#
#   这里不重新导出任何东西：一个图方便的导入会让人只想要一个模块时把八个都拉进来。成员导入"兄弟"模块
#   是另一回事，是允许的——兄弟模块遵守同样的规则，所以不会触及新的依赖；一份声明也因此可以和作用于它的
#   算术分开放。
# 观察：docstring 说"八个模块"，列出的也是八个，但目录里实际有九个：v41_pool_geometry.py（DeepSeek-V4.1
#   的 PAGE / STATE 几何）没有被提到。
#
# =====================================================================================
# 【导读】pool_layout 包 —— "一个字节在 KV pool 里的哪个位置"（第5批 5b，B级）
#   定位：ModelRunner.get_num_blocks / allocate_kv_cache（第5批 model_runner.py）只做"按字节求和、分一大块显存"，
#     每种注意力后端（第6~8批 attentions/*.py）只声明"自己的一个条目要多少字节、长什么样"；
#     两者之间的全部算术——预算怎么切、一块显存怎么切成各层视图、某层某 token 的 KV 在第几行第几字节、
#     状态快照怎么拷进 KV 池——都在这个包里，而且不 import aiter，所以没有 GPU 也能单测。
#   姊妹包 token_layout（同目录）回答另一个问题："这一步的每个 token 写到哪个槽位、属于哪条序列"，
#     每步重算；本包回答的问题只取决于 config，启动时算一次。
#   前置：分页 KV 与 block_table 见 config.py 文件开头"【科普】KV cache 与分页（paged）KV"；
#     每 token 的 KV 字节数公式与 FP8 / scale 见 arg_utils.py EngineArgs.kv_cache_dtype 上方的【科普】。
# =====================================================================================
#
# 【科普】一个字节住在 KV pool 的哪里？—— 从 MHA 的 [num_blocks, block_size, H, D] 出发，四步
#
# ① 普通 MHA / GQA：每层一个 K、一个 V，按 block 分页
#   nano-vllm 的做法（engine/model_runner.py allocate_kv_cache）是开一个大张量
#     kv_cache[2, 层数, num_blocks, block_size, num_kv_heads, head_dim]，第 l 层的 K 就是 kv_cache[0, l]，
#     形状 [num_blocks, block_size, num_kv_heads, head_dim]，V 同理。
#   一个 token 占多少字节（以 Qwen3-8B 为例：36 层、8 个 KV 头、head_dim 128、bf16）：
#     每层 K = 8 × 128 × 2B = 2048 B，V 一样 → 每层 4096 B → 36 层 = 147,456 B = 144 KiB / token
#     block_size = 16 时一个 block（16 个 token、全部 36 层的 K 和 V）= 2,359,296 B = 2.25 MiB
#     -tp 8 时每卡只有 1 个 KV 头 → 每卡 18 KiB / token；fp8 时 K/V 减半为 72 KiB，另加每 (KV 头, token)
#     一个 fp32 scale（K、V 各 1 个）：2 × 36 × 8 × 4B = 2.25 KiB → 74.25 KiB / token
#     预算 60 GiB（bf16、单卡）→ 60 GiB // 2.25 MiB = 27,306 个 block ≈ 43.7 万个 token
#   token 的"槽位"（slot）：把 [num_blocks, block_size] 两维摊平成一维后的下标
#     slot = block_table[pos // block_size] × block_size + pos % block_size
#     例（与 config.py【科普】同一个例子）：block_size = 16，某请求 block_table = [7, 2, 15]，
#       第 35 个 token：35 // 16 = 2 → 物理 block 15；35 % 16 = 3 → 块内第 3 个 → slot = 15 × 16 + 3 = 243
#       写 KV 时 kernel 拿到 243，就知道写到每一层 K/V 的 [block 15, 偏移 3]；slot 由 token_layout/slots.py 算
#   ATOM 的实际字节布局（attentions/mha_kv_pool.py 的 MhaKvPool + 本包 entry_arena.py）比 nano-vllm 多两点：
#     (a) 所有层、PAGE 区的所有子池是"一次 uint8 分配"切出来的区域（ModelRunner._carve_paged_pool → 本包 carve /
#         carve_layer_major），而不是每层一个张量；同一字段内"层在外、block 在内"（layer-major）：
#           ┌─ cache 组 ──────────────────────────────────────┬─ scale 组（仅 fp8）──┬─ extra 组 ──────┐
#           │ k：层0[blk0│blk1│…│blkN-1] 层1[…] … 层35[…]       │ k_scale │ v_scale    │ 如 MiniMax-M3 的 │
#           │ v：同上（整片接在全部 K 之后）                    │ 每(头,token)1个fp32  │ 索引 key（部分层）│
#           └─────────────────────────────────────────────────┴─────────────────────┴─────────────────┘
#         于是"第 l 层、block b 的 K"从 k 区域的 (l × N + b) × 32 KiB 字节处开始（N = num_blocks，
#         32 KiB = 16 token × 8 头 × 128 维 × 2B），一层的全部 block 连续——这是注意力 kernel 要的形状
#     (b) 块内元素不是 [block_size, H, D] 顺序，而是 aiter 的 SHUFFLE 顺序（MhaKvPool.kv_views）：
#         K 视图 [N, H, D/x, block_size, x]、V 视图 [N, H, block_size/x, D, x]，x = 16B / 元素字节（bf16 为 8）。
#         slot 公式不变——变的只是 block 内部的排列，只有写 KV 的 kernel 和读 KV 的 kernel 需要知道
#   对比nano-vllm：nano-vllm 每层 K/V 各一个视图、显存预算只算 K/V；ATOM 一个 block 里还可能有 scale、
#     索引 key、草稿模型的 KV、甚至状态快照（见②④），所以要有本包这层"条目 = 若干字段"的描述（EntryField）。
#
# ② 子池：为什么"一个 block id"同时寻址好几块缓存
#   按"随什么增长"把缓存分成两个预算区（sub_pool_spec.py 的 Pool）：
#     PAGE ：随"缓存了多少历史 token"增长——分页 KV block。一个 block id 对应"这 block_size 个 token 的
#            所有逐 token 数据"：MHA 的 K/V + fp8 scale、MLA 的潜在向量 + DSA 索引 key（③）、EAGLE3 草稿模型
#            的 KV（spec_decode/draft_kv.py 用同名条目类 "kv" 加进来）、V4 的压缩 KV + 索引器 key（④）
#     STATE：随"同时在跑的请求数"增长、与序列长度无关——线性注意力（GDN / KDA / Mamba）的递归状态、
#            V4 的滑动窗口与压缩器环形缓冲；每条请求一份（投机解码时 1 + k 份，留给回滚）
#   为什么共用一个 block id：调度器 / BlockManager 每条请求只维护一张 block_table（第4批）；这些逐 token 的
#     缓存都是"每多 block_size 个 token 就多一块"，同步增长、同步释放，于是让它们共用一个编号空间，
#     把它们的 entry_bytes 加起来当作"一个 block 的价格"，就只需要一个 num_blocks（sub_pool_spec.merge_specs）。
#   预算怎么分（sub_pool_spec.plan_pools，ModelRunner.get_num_blocks 调用）：
#       available_for_kv（ModelRunner 算出的 KV 预算）
#       ├── 先扣 STATE：Σ (max_num_seqs × entries_per_req + extra_entries) × entry_bytes   ← 请求没它跑不了
#       └── 剩下全给 PAGE：num_blocks = 剩余 // 一个 block 的字节数（所有 "kv" 条目之和）
#     例（Qwen3-Next：12 层全注意力有 KV、36 层 GDN 有状态；bf16、block_size 16、不开 MTP、max_num_seqs 256）：
#       PAGE 条目 = 16 token × 12 层 × 2 头 × 256 维 × 2(K,V) × 2B = 384 KiB；
#       STATE 条目 = 36 层 × 548,864 个数 × 2B ≈ 37.7 MiB / 请求（第4批 §4 已核实）
#       预算 60 GiB → STATE 先扣 256 × 37.7 MiB ≈ 9.42 GiB → 剩 ≈ 50.6 GiB → 138,112 个 block ≈ 221 万 token
#       开 MTP k=1（每请求 2 个状态槽）→ STATE 翻倍到 ≈ 18.8 GiB，block 降到 112,384 个——这就是 STATE 的代价
#   一个条目里装几种张量（"字段"，entry_arena.EntryField）、它们怎么排：
#     PAGE 用"层在外"（LayerMajorArena）：一层的所有 block 连续，kernel 按层绑定；
#     STATE 用"条目在外"（EntryMajorArena）：一条请求的全部状态是一段连续字节，便于整份拷贝做 checkpoint、
#       走 RDMA、在池边界移动时搬家（entry_arena.py 文件头列的三个理由）。
#
# ③ MLA：每 token 每层一行 576 维（前置要点；完整讲解见 arg_utils.py EngineArgs.tensor_parallel_size 下方的【科普】MLA）
#   要点：MLA 不缓存各头的 K/V，只缓存一个所有头共用的潜在向量 c（kv_lora_rank = 512 维）+ 一份共用的
#     RoPE key（64 维）= 576 个数；decode 时所有 q 头读同一行（形式上是 MQA，只有 1 个"KV 头"）。
#     所以 MLA 的池里没有 K/V 之分、也没有 per-head scale：一个字段 "kv"，形状 [层, block, block_size, 576]
#     （attentions/mla_kv_pool.py 的 MlaKvPool）。
#   数字（DeepSeek-V3：61 层、bf16）：每 token 每层 1152 B → 61 层 70,272 B ≈ 68.6 KiB / token（Qwen3-8B 是 144 KiB），
#     block_size 16 时一个 block ≈ 1.07 MiB。DSA 模型（V3.2、GLM-5）再加第二个区域 "index"：
#     [索引层数, block, 每 block 行数, 144]（fp8 key 128 B + fp32 scale 4 B，补齐到 16 字节倍数，
#     aiter_mla.aligned_index_cache_dim）——同一个 block id、分开存放（索引器见 arg_utils.py index_cache_dtype 的【科普】DSA）。
#   注意 MLA kernel 通常按"页大小 1"读池（ATOM_MLA_PAGE_SIZE 默认 1，block_ratio = block_size），
#     但池的"条目"仍是调度器的 block：一个 block = block_ratio 个物理页首尾相连（mla_kv_pool.py 文件头）。
#
# ④ DeepSeek-V4 / V4.1：压缩 KV + 滑动窗口，和"页单元"
#   前置要点（完整讲解见 config.py Config.__post_init__ 第12步 V4 部分的【科普】CSA / HCA）：V4 的 KV 只有 1 个共享头
#     （head_dim 512），每层按 compress_ratios 取一种：0 = 只有滑动窗口（最近 128 个 token），4 = CSA（每 4 个 token
#     压成 1 行、再用索引器挑 top-k），128 = HCA（每 128 个 token 压成 1 行）；每层都另有 128 个 token 的滑动窗口。
#     block_size 被改成 256 → 一个 block 里 CSA 层 256/4 = 64 行、HCA 层 256/128 = 2 行。
#   V4 的布局（v4_pool_geometry.py）：两类 KV 一个"行空间"——压缩历史随历史增长（按 block 编号，PAGE），
#     滑动窗口每请求固定（按槽位编号，STATE）；block 从行 0 往上长、槽位从顶端往下长，中间的空隙两边都能用：
#       行 0                                                           plane_rows
#       ├── block 0 的"信封" │ block 1 │ … ──→      空隙      ←── … │ 组 1 的槽位 │ 组 0 的槽位 ┤
#           （每个信封 = 一个 block 在所有压缩层的行，按压缩类分组）      （每个槽位 = 一条请求的压缩器状态 + 全部层的窗口）
#       组 g（请求拿到的状态池下标）放在位置 slot_positions − 1 − g（physical_slot），所以新分配的往下长
#     一行 = 某层一个"压缩后 token"或一个窗口位置的 KV；fp8 时一行分两个平面存（NoPE 512 B + RoPE 128 B，v4_pool_fields.py），
#     bf16 时一个平面 1024 B。行号在两个平面里含义相同——读 KV 的 kernel 只拿一个共享的行号数组。
#     例（V4-Pro：30 层 CSA + 31 层 HCA + 1 层纯窗口，见 deepseek_v4_attn.py 文件头；窗口 128、不开投机）：
#       一个信封 = 31 × 2 + 30 × 64 = 1982 行 → fp8 时 1982 × 640 B ≈ 1.21 MiB / block（256 个 token，
#         约 4.8 KiB / token，另加索引器 key）——对比 DeepSeek-V3 的 MLA 68.6 KiB / token
#       一个槽位的窗口部分 = 62 层 × 128 = 7936 行 → fp8 时 ≈ 4.84 MiB / 请求（前面再加压缩器状态的 arena_rows 行）
#   V4.1（CSA2，v41_pool_geometry.py）：压缩率只取 0 / 1 / 2，只有少数"owner"层（docs/deepseek_v41_runtime.md：
#     层 2、8、14、20）真正在 PAGE 里存主 KV 和索引行，其余层复用 owner 的；block_size 改成 128。
#     PAGE 按区域排：[所有 page 的主 KV 行 │ 每个 owner 的索引平面 …]，同一个 block id 在各区域里各占一份；
#     STATE 条目 = 全部层的窗口环 + 压缩器环 + 一个游标（cursor）。
#   页单元（PAGE unit）：没有逐 token KV 的状态（GDN / KDA 的递归状态、V4 的窗口 + 压缩器状态）要做前缀缓存，
#     就在 hash block 边界给活跃槽位拍快照，并把快照字节切成"页单元"（= KV 池里一个 block 的全部字节）
#     存进借来的若干个普通 KV block。概念、命中退回规则和数字见第4批 state_pool.py 文件开头【科普】⑤、
#     page_unit_checkpoint.py 文件头；本包负责"字节落在哪"：
#       活跃槽位的若干段 ──plan_segmented_copy 求交──> 一组 (源地址, 目的地址, 长度) ──launch_copy_descriptor──> 页单元的各区域
#       源侧地址：各后端的 _checkpoint_slot_bases；目的侧地址：page_unit_geometry（MLA 池，Kimi-K3）、
#       mha_page_unit_geometry（MHA 池，Qwen3-Next / Qwen3.5 的 GDN）、V4 / V4.1 各自的后端；拷贝本身：paged_state_copy。
#
# 本包各模块（建议按这个顺序读）：
#   sub_pool_spec.py           预算：PAGE / STATE 两个区、条目类、plan_pools 把字节预算换成条目数（②）
#   entry_arena.py             一个条目里的字段怎么排（对齐、偏移），以及 entry-major / layer-major 两种物化方式（①②）
#   pool_rows.py               哪些 module 在哪个"行空间"里占第几行（按实际构建出的模型遍历，代替从 config 推算层数）
#   v4_pool_fields.py          V4 一行主 KV、一个索引器 block 里装什么（含 MQA_LOGITS_PRESHUFFLE_ROWS = 16）
#   v4_pool_geometry.py        V4 的统一行空间：信封、槽位、窗口行号公式（④）
#   v41_pool_geometry.py       V4.1 的 PAGE 区域与 STATE 条目（④）
#   paged_state_copy.py        按描述符逐段拷贝字节流（状态快照 ↔ 页单元）
#   page_unit_geometry.py      页单元在 MLA 池里的地址（Kimi-K3）
#   mha_page_unit_geometry.py  页单元在 MHA 池里的地址（Qwen3-Next / Qwen3.5 的 GDN）
# =====================================================================================
