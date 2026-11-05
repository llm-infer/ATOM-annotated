# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where this step's tokens go: the per-token index arrays a forward needs.

The counterpart of `..pool_layout`, on the other axis. That package answers
where a byte lives and is a function of the config; this one answers where a
token goes and is a function of the batch, so it is rebuilt every step. The
membership rule is the same one, stated in full in `..pool_layout`: by topic,
arithmetic rather than staging, and reachable without aiter or the rest of atom
-- which is what makes the arithmetic checkable against a naive reference while
the staging around it is not.

Four modules, split by what differs and what does not. `prefill` and `decode`
shape the token axis -- ragged for a chunk, rectangular for a speculative step
-- and that is the whole difference between them. `slots` and `batch_ids` are
what both sides then want off that axis: which KV slot each token is written
to, and which sequence it belongs to. Neither of those cares which side shaped
the axis, so neither is duplicated per side; both are the kind of answer that
goes wrong in silence, landing a write or a lookup on another request's row.

Not everything per-token is here. A shape that only one caller has stays with
that caller: V4's decode positions are ragged (a DSpark step verifies a
different number of drafts per request), the one-token decode slot mapping is
one position per row computed inline by its builders, and DCP's is a per-rank
filter rather than an address. Pulling any of them in would put two
derivations behind one name.

Nothing is re-exported.
"""

# 【译】这一步的 token 去哪里：一次 forward 需要的逐 token 下标数组。
#
#   ..pool_layout 的对应物，在另一条轴上。那个包回答"一个字节住在哪"，是 config 的函数；这个包回答"一个 token
#   去哪"，是 batch 的函数，所以每一步都要重建。成员资格规则是同一条，完整表述在 ..pool_layout 里：按主题、只做
#   算术不做暂存、不依赖 aiter 和 atom 其余部分就能导入——正是这一点让这些算术可以拿一个朴素的参考实现来核对，
#   而围绕它们的暂存代码做不到。
#
#   四个模块，按"什么不同、什么相同"来分。prefill 和 decode 决定 token 轴的形状——一个 chunk 是参差的（ragged），
#   投机解码的一步是矩形的——这就是它们之间的全部差别。slots 和 batch_ids 是两边随后都想从这条轴上得到的东西：
#   每个 token 的 KV 写进哪个槽位，以及它属于哪条序列。这两者都不关心轴是哪一边形成的，所以都不按边各写一份；
#   两者都属于那种出错时悄无声息的答案——把一次写入或一次查找落到另一个请求的行上。
#
#   并非所有逐 token 的东西都在这里。只有一个调用方用到的形状留在那个调用方那里：V4 的 decode positions 是参差的
#   （DSpark 的一步对每个请求验证的草稿数不同）；单 token decode 的 slot mapping 是每行一个位置，由各自的 builder
#   内联计算；DCP 的则是按 rank 的过滤而不是地址。把它们中任何一个拉进来，都会让一个名字背后有两种推导。
#
#   不重新导出任何东西。
# 观察：docstring 说"四个模块"，目录里还有第五个 gather.py（gather_rows：一次 launch 从几个张量里按行号取出若干
#   token 行，DeepSeek-V4.1 的 runtime / bounded_replay 用）；它 import 了 triton（但不依赖 aiter），没有在这里被提到。
#
# =====================================================================================
# 【导读】token_layout 包 —— "这一步的每个 token 写到哪、属于谁"（第5批 5b，B级）
#   定位：每一步 ModelRunner.prepare_inputs → 注意力 builder 的 prepare_prefill / prepare_decode（第5、6批）
#     需要把调度器给的"每条序列"的量（context_lens、num_cached_tokens、block_tables）展开成"每个 token"的数组，
#     写进预分配的锁页缓冲（forward_vars 里的 CpuGpuBuffer），再上传给 kernel。这里只放其中的纯算术。
#   对比nano-vllm：nano-vllm 的 prepare_prefill / prepare_decode 用 Python 循环逐条序列 extend 出 positions、
#     slot_mapping，每步新建张量；ATOM 用 numpy 向量化写进复用的缓冲区（大 batch 时 host 开销差一个数量级），
#     并多出 batch_id_per_q_token / batch_id_per_k_token（token → 序列）给不按 cu_seqlens 查表的 kernel 用。
#
# 【科普】一步的逐 token 下标——两条序列、block_size = 16，走一遍 prefill、decode 和投机 decode
#   前置：分页 KV、block_table、slot = block_table[pos // 16] × 16 + pos % 16，见 config.py 文件开头"【科普】KV cache 与
#     分页（paged）KV"和 pool_layout/__init__.py【科普】①；cu_seqlens（把各序列的 token 首尾拼成一条轴、用前缀和标边界）
#     见 nano-vllm 的 prepare_prefill（ATOM 的做法相同）。
#   两条序列：
#     A：新请求，prompt 20 个 token，没有前缀命中，block_table = [3, 8]
#     B：prompt 40 个 token，前 32 个（2 个整 block）命中前缀缓存，本步只算后 8 个，block_table = [5, 1, 9]
#   ① prefill 这一步（backends.py CommonAttentionBuilder.prepare_prefill；token 轴是"参差"的，长 28）：
#                          A 的部分                        B 的部分              拼接 / 说明
#     seqlens_q            20                              8                     本步 token 数；cu_seqlens_q = [0, 20, 28]
#     cached_lens          0                               32                    已缓存前缀 = context_lens − seqlens_q
#     context_lens         20                              40                    cu_seqlens_k = [0, 20, 60]，total_kv = 60
#     positions            0 1 … 19                        32 33 … 39            位置，prefill.prefill_positions
#     slot_mapping         48 … 63 │ 128 … 131             144 … 151             槽位，slots.slot_mapping
#                          （pos 0~15 → block 3；16~19 → block 8）（pos 32~39 → 逻辑 block 2 → block 9）
#     batch_id_per_q       0 × 20                          1 × 8                 batch_ids.build_batch_ids；若 pad_to = 32，尾部补 4 个 −1
#                          （由需要它的 builder 经 publish_batch_ids 生成，如 aiter_mla.py 的稀疏 prefill / decode）
#     batch_id_per_k       0 × 20                          1 × 40                build_batch_ids_device（长 60，在 device 上；只在有前缀命中或 TBO 时建）
#   ② 下一步普通 decode（每条序列 1 个 token；slot 由 builder 内联算，不在本包）：
#     A 送 pos 20（context 21）→ 逻辑 block 1 → block 8 → slot 8 × 16 + 4 = 132
#     B 送 pos 40（context 41）→ 逻辑 block 2 → block 9 → slot 9 × 16 + 8 = 152
#   ③ 若开 MTP k = 2（每条序列这一步送 1 + 2 = 3 个 token，token 轴是 2 × 3 的矩形；context_lens 已含草稿 = [23, 43]）：
#     位置：decode.decode_positions([23, 43], 3)      → [20, 21, 22, 40, 41, 42]
#     槽位：slots.slot_mapping(…, seqlens_q = [3, 3]) → [132, 133, 134, 152, 153, 154]
#     序列：batch_ids                                  → [0, 0, 0, 1, 1, 1]
#   边界与代价：
#     - 槽位算错不会报错，只会把 KV 写进别人的 block（所以 slot 只有一个公式、只在 slots.py 写一次）；
#     - CUDA Graph 回放时 token 轴按档位宽度（如 32）而不是实际 token 数运行，补位 token 必须标成 −1（"哪儿也不写、
#       不属于任何序列"），不能用 0——0 是真实的序列 0；
#     - batch_id_per_k 的长度 total_kv 没有上限（随上下文增长），所以它在 device 上直接生成，不走固定大小的 host 缓冲。
# =====================================================================================
