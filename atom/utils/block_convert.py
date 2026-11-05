# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# ======================================================================================================
# 【导读】block table 的"粒度换算"Triton kernel：调度器的block（kv_cache_block_size）→ 注意力kernel要的页（page）
#
# ① 两种block大小（nano-vllm里只有一种）
#   调度器的block：BlockManager、前缀缓存、block_table里的id都按它算，即 kv_cache_block_size（例如16、64、128）；
#   注意力kernel索引的"页"：每个backend自己定（CommonAttentionBuilder.block_size），可以更小：
#     MLA（aiter_mla.py）              页 = 1个token（ATOM_MLA_PAGE_SIZE，默认1）
#     MHA（aiter_attention.py）        页 = 16；调度器block是256 / 1024、或开了ATOM_USE_UNIFIED_ATTN时与调度器block相同
#     MiniMax-M3 稀疏注意力             页 = sparse_block_size = 128
#   ratio = block_ratio = 调度器block / kernel页（backends.py CommonAttentionBuilder.__init__，要求整除）。
#   KV池按kernel页排布：调度器的 block b 恰好就是kernel页 [b×ratio, (b+1)×ratio)，所以换算只是 b → b×ratio + r（0 ≤ r < ratio）：
#
#       调度器block_table一行（block_size=64）：[ 5 , 9 ]                    context_len = 100个token
#       MLA的页号（页=1，ratio=64）：           [320..383, 576..611]          共100个（5×64+0..63，9×64+0..35）
#       MHA的页号（页=16，ratio=4）：           [20, 21, 22, 23, 36, 37, 38]  共ceil(100/16) = 7页
#
# ② CSR格式：kv_indptr + kv_indices（aiter的paged attention / MLA decode kernel用的输入）
#   nano-vllm把补齐的 block_tables [bs, max_blocks] 直接交给flash-attn；aiter的不少kernel要的是"压平"的形式：
#     kv_indices  所有请求的页号首尾相接，一维 [Σ 各请求的页数]
#     kv_indptr   int32 [bs + 1]，第i个请求的页在 kv_indices[kv_indptr[i] : kv_indptr[i + 1]]
#   例（MHA页=16、ratio=4；A有100个token、B有20个token；block_tables = [[5, 9], [3, 0]]，0是补齐值）：
#     kv_indptr = [0, 7, 9]，kv_indices = [20, 21, 22, 23, 36, 37, 38,  12, 13]
#   kv_indices_generate_triton 就是从 block_tables 直接生成这个 kv_indices（kv_indptr由调用方先算好）。
#
# ③ DCP下的换算（前置：DCP）
#   MLA每token每层只缓存一个所有头共用的576维潜在向量，相当于只有1个KV头，TP按头切不开它：-tp 8时8张卡各存一整份。
#   DCP（decode上下文并行）让这几张卡按token轮流各存一部分：-dcp 4（interleave_size=1）时token t只存在第 t % 4 张卡上。
#   （block_table一项怎样覆盖 block_size × DCP 个全局token的图见 prefix_hash.py 文件头④；DCP本身的完整讲解：
#    arg_utils.py EngineArgs.decode_context_parallel_size上方的【科普】）
#   于是同一个block_table条目在4张卡上各放自己的block_size个token：一个条目覆盖 block_size × DCP 个全局token——
#   这也是前缀hash的块大小（prefix_hash.py 的 hash_block_size_for：16 × 4 = 64）。
#   本卡第 j 个本地token的位置 = table[j // block_size] × block_size + j % block_size，与不开DCP时的公式相同，
#   只是 j 和长度都换成本地的：所以DCP的MLA decode把"本地长度"算出的 kv_indptr 交给同一个 kv_indices_generate_triton
#   （aiter_mla.py prepare_mtp_decode 里的 dcp_local_rebuild）。
#   例：block_size=16，-dcp 4，100个token的请求，table = [7, 3]：条目0覆盖全局token 0~63，条目1覆盖64~99。
#     卡1持有 token 1, 5, 9, …, 97 共25个，本地第j个是全局token 4j+1：j=0 → 7×16+0 = 112；j=16（全局65）→ 3×16+0 = 48；
#     本地 kv_indptr = [0, 25]，kv_indices = [112..127, 48..56]（16 + 9 = 25个）
#   decompose_slots_triton 是 DCP + FP4索引器prefill 专用的：把本卡的槽位号拆成（页, 页内行, e8m0 scale行）。
#
# 本文件的函数与调用方（grep核实）：
#   block_table_convert_triton     [bs, cols] → [bs, cols×ratio] 的页表      aiter_attention.py _get_sparse_attention_block_tables
#                                                                             （MiniMax-M3稀疏注意力）
#   kv_indices_convert_triton      kv_indices（按调度器block）→ 按kernel页    只有本文件末尾的 __main__ 示例在用
#   kv_indices_generate_triton     block_tables → kv_indices（CSR）          aiter_attention.py、aiter_mla.py、gdn_attn.py（经
#                                                                             aiter_attention导入）、spec_decode/dspark_proposer.py、
#                                                                             plugin/vllm/attention/metadata.py（vLLM插件）
#   mtp_prepare_decode_mla_kernel  MTP草稿每一步的MLA元数据更新（融合成一个kernel） aiter_mla.py prepare_mtp_decode
#   decompose_slots_triton         槽位号 → (页, 行, scale行)                aiter_mla.py（__init__预热、_build_dcp_indexer_fp4_prefill_meta）
# ======================================================================================================

import torch
import triton
import triton.language as tl


# block_table_convert_triton 的kernel：一维grid，每个program处理 BLOCK_SIZE=256 个输入元素（压平的 [bs, n_cols] 页表），
#   每个输入元素展开成 ratio 个输出 → 一次写 [256, ratio] 个；输出第 c×ratio + r 列 = b×ratio + r（或-1）
@triton.jit
def block_table_convert_kernel(
    blk_table_ptr,
    output_ptr,
    context_lens_ptr,
    ratio: tl.constexpr,
    n_input_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_input_elements

    # Load original values
    # 【译】加载原始值。
    val = tl.load(blk_table_ptr + offsets, mask=mask, other=-1)

    # Compute row (batch) and column indices from flattened offsets
    # 【译】从压平的偏移算出行（batch）号和列号。
    row = offsets // n_cols
    col = offsets % n_cols

    # Load per-batch context length
    # 【译】加载每个batch（请求）的上下文长度。
    ctx = tl.load(context_lens_ptr + row, mask=mask, other=0)

    # Vectorized expansion over ratio
    # 【译】在ratio维上向量化展开。val_exp：[256, 1]，i：[ratio] → 广播成 [256, ratio]
    i = tl.arange(0, ratio)
    val_exp = val[:, None]
    is_neg_one = val_exp == -1
    new_val = val_exp * ratio + i[None, :]

    # For each expanded position, check against context length limit
    # 【译】对每个展开后的位置，检查是否超过上下文长度上限。
    expanded_col = col[:, None] * ratio + i[None, :]
    valid = expanded_col < ctx[:, None]

    # If original was -1 or exceeds context, write -1
    # 【译】原值是-1、或超出上下文时，写-1。
    write_val = tl.where(is_neg_one | (~valid), -1, new_val)

    # Compute output indices in flattened space
    # 【译】在压平的空间里算输出下标。
    out_idx = offsets[:, None] * ratio + i[None, :]
    tl.store(output_ptr + out_idx, write_val, mask=mask[:, None])


# 把调度器粒度的页表 [bs, cols] 展开成kernel粒度的页表 [bs, cols × ratio]：第i行第c列的 b → 第 c×ratio + r 列写 b×ratio + r
# 输入：block_table          int32 [bs, cols]（不连续时先 .contiguous()）
#       block_table_convert  输出缓冲区 int32 [bs, cols × ratio]（预分配，形状由assert检查）
#       context_lens         int32 [bs]；ratio = 调度器block / kernel页
# 输出：block_table_convert（原地写并返回）；原值为-1、或展开后的列号 ≥ context_lens[i] 的位置写-1
# 调用方：aiter_attention.py _get_sparse_attention_block_tables（MiniMax-M3稀疏注意力的页128比调度器block小、ratio > 1时）
# 例（见文件末尾 __main__）：ratio=4，一行 [0, 1, 2, -1]、context_lens=10 →
#   [0, 1, 2, 3,  4, 5, 6, 7,  8, 9, -1, -1,  -1, -1, -1, -1]
# 观察：kernel拿"展开后的列号"（kernel页的序号）和 context_lens 比较，__main__示例里 context_lens 的单位也是页；
#   而 MiniMax-M3 的调用方传的是 context_lens（token数）。于是列号落在 [ceil(ctx / 页), ctx) 之间的位置不会被置-1，
#   写进去的是源表在该处的展开值（ATOM的block_tables缓冲区在有效长度之后补0，见utils/block_tables.py，即 0×ratio + r）。
#   这些位置是否会被读到取决于下游稀疏kernel是否自己按seq_lens截断，本文件看不出来；代码未改。
def block_table_convert_triton(block_table, block_table_convert, context_lens, ratio):
    if not block_table.is_contiguous():
        block_table = block_table.contiguous()
    assert block_table.shape[1] * ratio == block_table_convert.shape[1]
    assert context_lens.shape[0] == block_table.shape[0]

    n_input_elements = block_table.numel()
    n_cols = block_table.shape[1]

    def grid(meta):
        return (triton.cdiv(n_input_elements, meta["BLOCK_SIZE"]),)

    block_table_convert_kernel[grid](
        block_table,
        block_table_convert,
        context_lens,
        ratio,
        n_input_elements,
        n_cols,
        BLOCK_SIZE=256,
    )

    return block_table_convert


# kv_indices_convert_triton 的kernel：输入是已经压平的 kv_indices（每个元素是一个调度器block），每个program处理256个元素。
#   某个元素属于哪个请求，靠循环全部bs个请求、累加每个请求的block数来定位（bs是constexpr：每个不同的bs编译一个版本）
# do_not_specialize：n_input_elements 每步都变，不让Triton按它的值（如能否被16整除）编译出不同的版本
@triton.jit(do_not_specialize=["n_input_elements"])
def kv_indices_convert_kernel(
    kv_indices_ptr,
    output_ptr,
    kv_indptr_convert_ptr,
    n_input_elements,
    ratio: tl.constexpr,
    ori_block_size: tl.constexpr,
    bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_input_elements

    # Load original values
    # 【译】加载原始值。
    val = tl.load(kv_indices_ptr + offsets, mask=mask, other=-1)

    col = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    ctx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    out_bs_start = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # 对每个输入元素求出：它在本请求内是第几个block（→ 输出列col = 块内序号 × ratio）、本请求的长度ctx（按kernel页）、
    #   本请求在输出里的起点out_bs_start = kv_indptr_convert[j]
    acc_block = 0
    for j in range(0, bs):
        start_j = tl.load(kv_indptr_convert_ptr + j)
        end_j = tl.load(kv_indptr_convert_ptr + (j + 1))
        ctx_len = end_j - start_j
        num_block = tl.cdiv(ctx_len, ori_block_size)
        in_this = (offsets >= acc_block) & (offsets < (acc_block + num_block)) & mask
        col = tl.where(in_this, (offsets - acc_block) * ratio, col)
        ctx = tl.where(in_this, ctx_len, ctx)
        out_bs_start = tl.where(in_this, start_j, out_bs_start)
        acc_block += num_block

    # 每个block展开成ratio个页号 val×ratio + r_i，只写在本请求长度之内的
    for r_i in range(0, ratio):
        cond = (col + r_i) < ctx
        cand = val * ratio + r_i
        tl.store(output_ptr + out_bs_start + col + r_i, cand, mask=mask & cond)


# 把按调度器block的 kv_indices（CSR，一维）展开成按kernel页的
# 输入：kv_indices          int32 [Σ 各请求的block数]（调度器粒度：block_size × ratio 个token一块）
#       kv_indices_convert  输出，大小 ≥ kv_indptr_convert[-1]
#       kv_indptr_convert   int32 [bs + 1]：按kernel页计的indptr（即各请求的长度按页累加）
#       ratio；block_size = kernel页大小（ori_block_size = block_size × ratio 就是调度器block）
# 输出：kv_indices_convert（原地写并返回）
# 调用方：grep只有本文件末尾的 __main__；现在的调用方都直接用 kv_indices_generate_triton（从block_tables生成，不用先压平）
# 例（__main__）：kv_indices=[0..5]，kv_indptr_convert=[0, 3, 9, 11, 18]，ratio=4，页=1 →
#   请求0 长3：block 0 → [0, 1, 2]；请求1 长6：block 1、2 → [4, 5, 6, 7, 8, 9]；请求2 长2：block 3 → [12, 13]；…
def kv_indices_convert_triton(
    kv_indices, kv_indices_convert, kv_indptr_convert, ratio, block_size
):
    ori_block_size = block_size * ratio
    bs = kv_indptr_convert.shape[0] - 1
    n_input_elements = kv_indices.numel()

    def grid(meta):
        return (triton.cdiv(n_input_elements, meta["BLOCK_SIZE"]),)

    kv_indices_convert_kernel[grid](
        kv_indices,
        kv_indices_convert,
        kv_indptr_convert,
        n_input_elements,
        ratio,
        ori_block_size,
        bs,
        BLOCK_SIZE=256,
    )

    return kv_indices_convert


@triton.jit
def kv_indices_generate_kernel(
    block_tables_ptr,
    output_ptr,
    kv_indptr_convert_ptr,
    n_cols: tl.constexpr,
    ratio: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr,
    RATIO_PAD: tl.constexpr,
):
    """Block-centric kernel: load each block table entry ONCE, expand to ratio outputs.
    ZERO integer division. Fully coalesced stores.
    2D grid: (tiles_over_blocks, bs).
    Each tile: load BLOCKS_PER_TILE block ids → write BLOCKS_PER_TILE * ratio outputs.
    """
    # 【译】以block为中心的kernel：每个block table条目只加载一次，展开成ratio个输出。
    #   没有整数除法。存储完全合并（coalesced）。
    #   二维grid：(覆盖block的tile数, bs)。
    #   每个tile：加载 BLOCKS_PER_TILE 个block id → 写 BLOCKS_PER_TILE × ratio 个输出。
    # program (tile_idx, batch_idx)：处理第batch_idx个请求的第 [tile_idx × BPT, (tile_idx+1) × BPT) 个block，
    #   写到 kv_indices[kv_indptr[batch_idx] + 块号 × ratio + r]；ctx_len = 本请求的长度（按kernel页，kv_indptr之差）
    batch_idx = tl.program_id(1)
    tile_idx = tl.program_id(0)

    out_start = tl.load(kv_indptr_convert_ptr + batch_idx)
    ctx_len = tl.load(kv_indptr_convert_ptr + batch_idx + 1) - out_start

    # Block columns this tile processes: [BPT]
    # 【译】本tile处理的block列：[BPT]
    block_start = tile_idx * BLOCKS_PER_TILE
    block_cols = block_start + tl.arange(0, BLOCKS_PER_TILE)
    load_mask = block_cols < n_cols

    # Load block table entries — each loaded ONCE, broadcast to ratio outputs: [BPT]
    # 【译】加载block table条目——每个只加载一次，广播到ratio个输出：[BPT]
    table_idx = batch_idx * n_cols + block_cols
    vals = tl.load(block_tables_ptr + table_idx, mask=load_mask, other=0)

    # 2D expansion: [BPT, RATIO_PAD] — zero division, pure add/mul
    # 【译】二维展开：[BPT, RATIO_PAD]——没有除法，只有加法和乘法
    # RATIO_PAD = ratio向上取2的幂：tl.arange的长度必须是2的幂，多出来的 sub ≥ ratio 由下面的掩码挡掉
    sub = tl.arange(0, RATIO_PAD)
    expanded_vals = vals[:, None] * ratio + sub[None, :]
    out_pos = block_cols[:, None] * ratio + sub[None, :]

    # Store mask: valid block × valid sub-index × within context length
    # 【译】存储掩码：有效的block × 有效的子下标 × 在上下文长度之内
    # 只写 out_pos < ctx_len 的位置：block_tables里有效长度之后的补齐值（0）不会被当成页号写出去
    store_mask = load_mask[:, None] & (sub[None, :] < ratio) & (out_pos < ctx_len)
    tl.store(output_ptr + out_start + out_pos, expanded_vals, mask=store_mask)


def kv_indices_generate_triton(
    block_tables, kv_indices_convert, kv_indptr_convert, ratio, max_ctx_len
):
    """Generate converted kv_indices directly from block_tables.

    Args:
        block_tables: [bs, max_blocks] int32 tensor of physical block ids
        kv_indices_convert: output tensor, size >= kv_indptr_convert[-1]
        kv_indptr_convert: [bs+1] int32, token-level indptr at converted page size
        ratio: block_size_old // block_size_new
        max_ctx_len: max context length across all batches (avoids GPU sync)
    """
    # 【译】直接从 block_tables 生成换算后的 kv_indices。
    #
    #   参数：
    #       block_tables：[bs, max_blocks] int32张量，物理block id
    #       kv_indices_convert：输出张量，大小 >= kv_indptr_convert[-1]
    #       kv_indptr_convert：[bs+1] int32，按换算后的页大小计的token级indptr
    #       ratio：block_size_old // block_size_new
    #       max_ctx_len：所有batch中最大的上下文长度（避免GPU同步）
    # 输出：kv_indices_convert（原地写并返回），格式见文件头②
    # max_ctx_len 只用来定grid大小（每个请求要几个tile），是主机上的数，所以不用从GPU读kv_indptr（.item()会同步）；
    #   偏大是安全的，只是多启动一些什么也不写的program。例：aiter_attention.py 传的是token数 context_lens.max()，
    #   而MHA的kv_indptr按16个token一页计，所以是上界；MLA页=1时两者相同
    # 调用方：aiter_attention.py（decode、TBO微batch）、aiter_mla.py（prepare_decode、TBO微batch decode、MTP草稿步含DCP本地重建、prefill）、
    #   gdn_attn.py prepare_mtp_decode、spec_decode/dspark_proposer.py、plugin/vllm/attention/metadata.py
    bs = block_tables.shape[0]
    n_cols = block_tables.shape[1]

    # 每个program最多写4096个输出：ratio=64（MLA页=1、block=64）时一个tile管64个block，ratio=4时管1024个
    ratio_pad = triton.next_power_of_2(ratio)
    blocks_per_tile = max(1, 4096 // ratio_pad)
    max_num_blocks = (max_ctx_len + ratio - 1) // ratio

    grid = (triton.cdiv(max_num_blocks, blocks_per_tile), bs)

    kv_indices_generate_kernel[grid](
        block_tables,
        kv_indices_convert,
        kv_indptr_convert,
        n_cols,
        ratio,
        BLOCKS_PER_TILE=blocks_per_tile,
        RATIO_PAD=ratio_pad,
    )

    return kv_indices_convert


# 【译】（下面各参数的行尾注释）
#   kv_indptr_ptr         int32 [bs+1] 读写：+= cu_seqlens_q
#   cu_seqlens_q_ptr      int32 [bs+1] 只读
#   sparse_kv_indptr_ptr  int32 [bs+1] 只写（仅IS_SPARSE）
#   positions_ptr         int64 [bs] 读写（仅UPDATE_POSITIONS）
#   context_lens_ptr      int32 [bs] 读写（仅UPDATE_CONTEXT_LENS）
# 背景：MTP（投机解码）的每个草稿步给每条序列的KV加1个token（本步每条序列1个query，cu_seqlens_q = [0, 1, 2, …]），
#   所以 MLA decode（页=1）的 CSR 要整体后移：kv_indptr[i] += cu_seqlens_q[i]（前i个请求一共多了i个token）。
#   例：bs=2，kv_indptr=[0, 100, 150]，cu_seqlens_q=[0, 1, 2] → [0, 101, 152]（A：100→101，B：50→51）
#   稀疏（DSA，见arg_utils.py EngineArgs里的【科普】DSA）：每行最多看index_topk个token：counts=[101, 51]，index_topk=64
#     → 截到index_topk：[64, 51] → 前缀和 sparse_kv_indptr = [0, 64, 115]
# 调用方：aiter_mla.py AiterMLAMetadataBuilder.prepare_mtp_decode（EagleProposer每个草稿步调用一次），grid=(1,)；
#   没用到的指针参数由调用方传kv_indptr占位，对应的constexpr分支整段被编译掉
@triton.jit
def mtp_prepare_decode_mla_kernel(
    kv_indptr_ptr,  # int32 [bs+1] in/out: += cu_seqlens_q
    cu_seqlens_q_ptr,  # int32 [bs+1] in
    sparse_kv_indptr_ptr,  # int32 [bs+1] out (IS_SPARSE only)
    positions_ptr,  # int64 [bs] in/out (UPDATE_POSITIONS only)
    context_lens_ptr,  # int32 [bs] in/out (UPDATE_CONTEXT_LENS only)
    bs,
    index_topk,
    position_stride,
    IS_SPARSE: tl.constexpr,
    UPDATE_POSITIONS: tl.constexpr,
    UPDATE_CONTEXT_LENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-program fused per-draft-step MTP-decode metadata for MLA.

    Collapses the elementwise chain that used to launch ~4 tiny kernels
    (``kv_indptr += cu_seqlens_q``; ``counts = diff(kv_indptr)``;
    ``clamp(index_topk)``; ``cumsum -> sparse_kv_indptr``) plus eagle's two
    per-step bumps (``positions += 1``, ``context_lens += 1``) into one launch.
    One program owns the whole batch so the sparse cumsum is a single in-register
    ``tl.cumsum``; the caller guarantees ``bs + 1 <= BLOCK`` (bs <= max_bs). All
    reads happen before any write, so the in-place ``kv_indptr`` update is
    hazard-free. ``sparse_kv_indptr[0]`` is written on-device (no Python-scalar
    H2D copy, which is what caused the original hot-path Host-Device sync).
    """
    # 【译】单个program、融合的、每个草稿步一次的MLA MTP-decode元数据更新。
    #
    #   把原来要启动约4个小kernel的逐元素链（kv_indptr += cu_seqlens_q；counts = diff(kv_indptr)；clamp(index_topk)；
    #   cumsum -> sparse_kv_indptr），加上eagle每步的两次自增（positions += 1、context_lens += 1），合并成一次启动。
    #   一个program负责整个batch，所以稀疏的cumsum就是寄存器里的一次 tl.cumsum；调用方保证 bs + 1 <= BLOCK（bs <= max_bs）。
    #   所有读都发生在任何写之前，所以原地更新 kv_indptr 没有读写冲突。sparse_kv_indptr[0] 在设备上写（没有Python标量的
    #   H2D拷贝——原来热路径上的主机-设备同步就是它造成的）。
    offs = tl.arange(0, BLOCK)
    # 【译】（下面两行的行尾注释）indptr数组的长度是bs+1；每序列的数组长度是bs
    mask_p1 = offs < (bs + 1)  # indptr arrays are length bs+1
    mask_bs = offs < bs  # per-seq arrays are length bs

    # Load originals first (into registers) so the kv_indptr store below cannot
    # clobber the shifted [i+1] read used for per-seq counts.
    # 【译】先把原始值加载进寄存器，这样下面对kv_indptr的写入不会冲掉"错位一格读[i+1]"的那次读取（它用来算每序列的计数）。
    kvi = tl.load(kv_indptr_ptr + offs, mask=mask_p1, other=0)
    cuq = tl.load(cu_seqlens_q_ptr + offs, mask=mask_p1, other=0)
    new_kvi = kvi + cuq

    if IS_SPARSE:
        kvi_next = tl.load(kv_indptr_ptr + offs + 1, mask=mask_bs, other=0)
        cuq_next = tl.load(cu_seqlens_q_ptr + offs + 1, mask=mask_bs, other=0)
        counts = tl.where(mask_bs, (kvi_next + cuq_next) - new_kvi, 0)
        clamped = tl.minimum(counts, index_topk)
        # inclusive scan: csum[i] == sparse_kv_indptr[i + 1]
        # 【译】包含式扫描（inclusive scan）：csum[i] == sparse_kv_indptr[i + 1]
        csum = tl.cumsum(clamped, axis=0)
        tl.store(sparse_kv_indptr_ptr + offs + 1, csum, mask=mask_bs)
        tl.store(sparse_kv_indptr_ptr + offs, tl.zeros_like(csum), mask=(offs == 0))

    tl.store(kv_indptr_ptr + offs, new_kvi, mask=mask_p1)

    # positions可能是一个带步长的视图（调用方传 positions_out.stride(0)），所以按 offs × position_stride 寻址
    if UPDATE_POSITIONS:
        pos = tl.load(positions_ptr + offs * position_stride, mask=mask_bs, other=0)
        tl.store(positions_ptr + offs * position_stride, pos + 1, mask=mask_bs)

    if UPDATE_CONTEXT_LENS:
        ctx = tl.load(context_lens_ptr + offs, mask=mask_bs, other=0)
        tl.store(context_lens_ptr + offs, ctx + 1, mask=mask_bs)


# decompose_slots_triton 的kernel：一维grid，每个program处理TILE=1024个位置，同时处理两路输入：
#   slots[offs]（offs < n_slots）与 iota = offs（offs < n_iota），各输出 (页, 页内行, row_lut[行]) 三个int32
@triton.jit(do_not_specialize=["n_slots", "n_iota"])
def decompose_slots_kernel(
    slots_ptr,
    n_slots,
    n_iota,
    row_lut_ptr,
    slot_page_ptr,
    slot_row_ptr,
    slot_lut_row_ptr,
    iota_page_ptr,
    iota_row_ptr,
    iota_lut_row_ptr,
    LOG2_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    # Widen before the multiply so large offsets don't wrap in i32.
    # 【译】先扩宽再相乘，这样大偏移不会在i32里溢出。
    offs = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    # Shift/mask, not // and %: triton's division truncates, torch's floors.
    # 【译】用移位/掩码，而不是 // 和 %：triton的除法向零截断，torch的是向下取整。
    # block是2的幂，slot = 页 × block + 行 → 页 = slot >> log2(block)，行 = slot & (block - 1)；与torch参考实现
    #   （aiter_mla.py 的回退分支用 // 和 %）对非负数结果相同
    mask = offs < n_slots
    slot = tl.load(slots_ptr + offs, mask=mask, other=0)
    row = slot & ((1 << LOG2_BLOCK) - 1)
    tl.store(slot_page_ptr + offs, slot >> LOG2_BLOCK, mask=mask)
    tl.store(slot_row_ptr + offs, row, mask=mask)
    tl.store(slot_lut_row_ptr + offs, tl.load(row_lut_ptr + row), mask=mask)

    mask = offs < n_iota
    token = offs.to(tl.int32)
    row = token & ((1 << LOG2_BLOCK) - 1)
    tl.store(iota_page_ptr + offs, token >> LOG2_BLOCK, mask=mask)
    tl.store(iota_row_ptr + offs, row, mask=mask)
    tl.store(iota_lut_row_ptr + offs, tl.load(row_lut_ptr + row), mask=mask)


def decompose_slots_triton(
    slots: torch.Tensor, n_iota: int, block_size: int, row_lut: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Split int32 `slots` and `arange(n_iota)` into (page, row, row_lut[row]).

    Returns six int32 tensors, `slots`' three then the iota's, in one launch.
    Sizes are not specialized; call once at startup to keep the JIT out of serving.
    """
    # 【译】把int32的 slots 和 arange(n_iota) 拆成 (page, row, row_lut[row])。
    #
    #   返回六个int32张量：先是slots的三个，再是iota的三个，一次启动完成。
    #   尺寸不做特化；在启动时调用一次，让JIT编译不出现在服务期间。
    # 用途：DCP + FP4索引器的prefill（aiter_mla.py _build_dcp_indexer_fp4_prefill_meta）。本卡要把自己持有的索引key从分页缓存
    #   "读"出来（slots = 本卡的槽位号），再"存"进一块按恒等页表排布的暂存区（第t个token → 页 t // block、行 t % block）。
    #   FP4索引缓存的e8m0 scale平面在每个block内的行是转置排布的，row_lut = fp4_index_scale_rows（model_ops/sparse_indexer_fp4.py）
    #   给出"逻辑行 → scale平面上的行"，所以每个位置还要多查一次表
    # 输入：slots int32 [n_slots]；n_iota（本步的总token数 total_kv）；block_size（必须是2的幂）；row_lut int32 [block_size]
    # 输出：(slot_page, slot_row, slot_lut_row, iota_page, iota_row, iota_lut_row)，前三个 [n_slots]，后三个 [n_iota]
    # 例：block=64，row_lut[r] = (r % 16) × 4 + r // 16：slot = 7×64 + 5 = 453 → 页7、行5、scale行20；iota的 t=70 → 页1、行6、scale行24
    # 调用方：aiter_mla.py：__init__ 里用1个元素预热编译（注释说预热跑不到长的DCP prefill），
    #   _build_dcp_indexer_fp4_prefill_meta 每次forward调用一次
    if block_size & (block_size - 1) or block_size <= 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if slots.dtype != torch.int32 or row_lut.dtype != torch.int32:
        raise TypeError(
            f"slots and row_lut must be int32, got {slots.dtype} and {row_lut.dtype}"
        )
    if n_iota < 0:
        raise ValueError(f"n_iota must be non-negative, got {n_iota}")
    if row_lut.numel() != block_size:
        raise ValueError(
            f"row_lut must have {block_size} entries, got {row_lut.numel()}"
        )
    slots = slots.contiguous()
    n_slots = slots.numel()
    dev = slots.device
    out = tuple(
        torch.empty(n, dtype=torch.int32, device=dev)
        for n in (n_slots, n_slots, n_slots, n_iota, n_iota, n_iota)
    )
    tile = 1024
    grid = (max(triton.cdiv(max(n_slots, n_iota), tile), 1),)
    decompose_slots_kernel[grid](
        slots,
        n_slots,
        n_iota,
        row_lut,
        *out,
        LOG2_BLOCK=block_size.bit_length() - 1,
        TILE=tile,
    )
    return out


if __name__ == "__main__":
    # Example usage and test
    # 【译】示例用法与测试。
    # 需要GPU（张量都 .cuda()）；打印 block_table_convert_triton 和 kv_indices_convert_triton 的输入输出，两个例子的结果见上面各函数的注释

    block_table = torch.tensor(
        [[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]], dtype=torch.int32
    ).cuda()
    context_lens = torch.tensor([10, 5, 15], dtype=torch.int32).cuda()
    ratio = 4
    block_table_converted = torch.empty(
        (block_table.shape[0], block_table.shape[1] * ratio),
        dtype=torch.int32,
    ).cuda()

    block_table_convert_triton(block_table, block_table_converted, context_lens, ratio)

    print("Original Block Table:")
    print(block_table.cpu().numpy())
    print("Converted Block Table:")
    print(block_table_converted.cpu().numpy())

    kv_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32).cuda()
    kv_indptr_convert = torch.tensor([0, 3, 9, 11, 18], dtype=torch.int32).cuda()
    ori_block_size = 4
    kv_indices_converted = torch.zeros(
        (kv_indices.shape[0] * ratio,), dtype=torch.int32
    ).cuda()
    kv_indices_convert_triton(
        kv_indices, kv_indices_converted, kv_indptr_convert, ratio, 1
    )
    print("Original KV Indices:")
    print(kv_indices.cpu().numpy())
    print("Converted KV Indices:")
    print(kv_indices_converted.cpu().numpy())
