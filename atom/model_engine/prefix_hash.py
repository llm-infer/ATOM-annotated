# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chained prompt-block hash, shared by the frontend and the BlockManager
(`BlockManager.compute_hash` is `block_hash`)."""

# 【译】链式的 prompt block hash，由前端和 BlockManager 共用（BlockManager.compute_hash 就是 block_hash）。
#
# 【导读】prefix_hash.py：前缀缓存的 hash 函数本身，以及"前端先把 prompt 的 hash 链算好、随请求发给引擎"这条路
#   谁用什么：
#     前端（主进程 / API 进程的 InputOutputProcessor，llm_engine.py）
#         __init__ → hash_block_size_for；preprocess_fanout → ship_prompt_hash_chain
#     engine_core_mgr.py 的 _add_frame（CoreManager 发 ADD 请求时打包）→ take_shipped_chains：把链从 seq 上摘下，作为帧的第 3 个元素
#     EngineCore 输入线程（engine_core.py）→ attach_shipped_chains：把链装回反序列化出来的 seq
#     BlockManager（block_manager.py）→ block_hash（即 compute_hash）、chained_block_hashes（_hit_walk）、
#         full_prompt_blocks / prompt_fingerprint（_adopt_shipped_chain 核对前端送来的链）、hash_block_size_for（__init__）
#   前置：前缀缓存的整体思路（命中多少块就跳过多少块的 prefill、最后一块永远不复用）见 sequence.py Sequence类上方
#     "科普：前缀缓存"；命中后 block 怎样共享、释放后何时才真正失效见 block_pool.py 文件头
#
# 【科普】前缀缓存的 hash：hash 了什么、为什么链式、会不会碰撞、开 DCP 时块为什么变大、为什么在前端算
#
# ① hash 了什么
#   一个 hash block = 连续 hash_block_size 个 token（不开 DCP 时就是 block_size：默认 16，DeepSeek-V4 / V4.1 / GLM-5.3 被 Config 改写成 256 / 128 / 64；DCP 见④）。第 i 块的 hash：
#
#       h_i = xxh64( h_{i-1} 的 8 字节小端  ‖  第 i 块 token 的 int64 字节 )     （‖ = 字节拼接；xxh64 的种子参数为 0）
#       h_0 = xxh64( cache_seed 的 8 字节   ‖  第 0 块 token 的 int64 字节 )     请求带 cache_seed（≠ -1）时
#       h_0 = xxh64(                          第 0 块 token 的 int64 字节 )     纯文本请求（cache_seed = -1）时
#
#   结果是 [0, 2^64) 里的无符号整数，所以 -1 永远不是合法 hash，被用作"没有父块 / 未登记"（kv_block.Block.hash）。
#   例：一块 16 个 token → 16 × 8 = 128 字节；有父块时前面再拼 8 字节，共 136 字节交给 xxh64。
#   只有写满、且 KV 已经算好的块才会被登记（BlockManager.hash_blocks），不满的最后一块内容还会变，不 hash。
#   链的根（"种子"）只有一个来源：Sequence.cache_seed
#     · 纯文本请求：-1，什么都不拼
#     · 多模态请求：multimodal_runtime.multimodal_cache_seed 对 pixel_values / image_grid_thw / token_types /
#       embedding_spans 算的 64 位 blake2b 指纹（Sequence.__init__ 里算，或由 DeepSeek-V4.1 的图像预处理预先放进数据）。
#       为什么需要：图片在 prompt 里是一串 id 相同的占位 token，只看 token 会让"同样的文字 + 不同的图"误命中别人的 KV
#     · ATOM 原生引擎不支持 LoRA 适配器，也没有 vLLM 那种按请求的 cache_salt，所以没有别的种子来源
#   代价：种子作用在第 0 块上，带图请求的整条链都和纯文本请求不同——哪怕图片之前那段 system prompt 一字不差，
#     带图请求也命中不了纯文本请求的 KV；不同图片的请求之间也不共享
#
# ② 为什么要链式：一个块的身份取决于它之前的全部内容
#   因果注意力里，从第 1 层起每个位置的 hidden 都已混入了它前面所有 token 的信息，所以一块 token 的 K/V 不只取决于
#   这块 token 本身，还取决于它前面的全部 token。两个 prompt 的第 1 块 token 一模一样、第 0 块不同（block_size = 4）：
#
#       prompt P： 块0 = [S1 S2 S3 S4]   块1 = [x y z w]      h0P = H(S1..S4)      h1P = H(h0P, x y z w)
#       prompt Q： 块0 = [T1 T2 T3 T4]   块1 = [x y z w]      h0Q = H(T1..T4)      h1Q = H(h0Q, x y z w)
#                                                            h0P ≠ h0Q   ⇒   h1P ≠ h1Q   ← 正确：两块的 KV 本来就不同
#     若只 hash 本块 token，就会得到 h1P = h1Q，Q 复用 P 的块 1 的 KV → 静默算错。
#   反过来，链式还给了一个很强的性质：命中 h_i 就意味着块 0..i 全部相同（不计碰撞）。所以 BlockManager 只要从第 0 块
#   往后查，遇到第一个未命中就停（之后的块不可能再命中），不需要逐块比较整段前缀。
#
# ③ 碰撞
#   64 位 hash，按生日问题估算：索引里有 N 个不同的块时，出现任意一对碰撞的概率约为 N² / 2^65；N = 100 万块时约 2.7×10⁻⁸。
#   BlockManager 命中后会把 Block.token_ids 和本块 token 比对：默认的 _hit_walk 只比最后一个命中块，不一致时再逐块比；
#     ATOM_PREFIX_FAST_WALK=0 的逐块遍历每块都比。这能抓住"hash 相同、本块 token 不同"；而"本块 token 相同、
#     前缀不同、hash 又恰好相同"比对不出来，只能靠 2^-64 的概率。（nano-vllm 每块都比；据 _hit_walk 的 docstring，vLLM 一块都不比）
#   xxh64 不是密码学 hash（不抗刻意构造的碰撞），追求的是快和分布均匀
#
# ④ 开 DCP 时，一个 hash block = block_size × DCP 个 token（hash_block_size_for；本批讲"DCP 对 block 账本的影响"的规范位置，
#   block_manager.py 的【前置：DCP】、block_convert.py 文件头③只写要点并指回这里）
#   前置（完整讲解见 arg_utils.py EngineArgs.decode_context_parallel_size 上方的【科普】DCP、tensor_parallel_size 下方的【科普】MLA）：
#     MLA（DeepSeek-V3 等）每个 token 每层只缓存一份 576 个数、所有头共用，相当于只有 1 个 KV 头，没法再按 TP 切头，
#     所以 -tp 8 时 8 张卡各存一整份同样的 KV。DCP 改成按 token 切：-dcp 4（interleave_size = 1）时第 i 个 token 的 KV
#     只存在 DCP 组的第 i % 4 张卡上；每张卡对自己那份 token 算局部注意力，再用 LSE 合并，结果与不切完全相同。
#   对 block 账本的影响：各卡共用同一张 block_table（BlockManager 只在调度器上跑一份），但每张卡的第 k 页只装自己那份 token：
#
#       block_size = 16，-dcp 4，block_table = [7, …]
#         卡0 的第 7 页：全局 token 0, 4, 8, …, 60          卡1 的第 7 页：1, 5, 9, …, 61
#         卡2 的第 7 页：2, 6, 10, …, 62                    卡3 的第 7 页：3, 7, 11, …, 63
#       → block_table 的一项（物理 block 7）覆盖全局 token 0~63，共 16 × 4 = 64 个
#       各卡的写入槽位 = block_table[pos // 64] × 16 + 本卡本地序号 % 16（dcp_ops.dcp_prefill_slot_mapping）；
#       KV 池按每卡的本地 block 数计：1000 个 token → ⌈1000/64⌉ = 16 个条目，每卡本地 250 个 token 也是 16 个物理 block
#       （BlockManager.num_pool_blocks）
#
#     前缀复用以 block_table 的一项为单位，所以 hash 也必须按 64 个全局 token 一块来算；DCP = 1 时就等于 block_size。
#     interleave_size = S > 1（每 S 个 token 一组轮流分；Config 断言 block_size 是 S 的倍数）时，第 k 项同样覆盖
#     全局 token [64k, 64k + 64)。
#
# ⑤ 为什么在前端算 prompt 的 hash 链（ATOM_FRONTEND_PREFIX_HASH，默认 1）
#   EngineCore 的调度是单线程串行的（busy_loop），在那里多做一点计算，下一步就晚一点发出；而 prompt 的 hash 只取决于 token，
#   分完词就能算。例：128K token 的 prompt、hash block = 16 → 8192 块、8192 次 xxh64，交给前端的 executor / 提交线程去算，
#   调度线程只做几次核对。
#
#     前端（InputOutputProcessor.preprocess_fanout）                       EngineCore 进程
#     ship_prompt_hash_chain(seq, hbs)：只算被 prompt 填满的块
#       seq.prompt_hash_chain = ShippedChain(hbs, 种子, prompt 长度, 指纹, hashes)
#       （n > 1 的兄弟序列同一个 prompt、同一个种子，共用这一条）
#     engine_core_mgr._add_frame → take_shipped_chains：把链从 seq 上摘下
#       （Sequence 被 pickle 时 __getstate__ 本来也会把它置 None）
#       帧 = (ADD, seqs, chains) ══════════════ ZMQ ══════════════>  输入线程：attach_shipped_chains 装回各 seq
#                                                                    BlockManager._prompt_hashes → _adopt_shipped_chain 核对：
#                                                                      ① hbs、种子、块数与本引擎一致
#                                                                      ② prompt 长度和 prompt_fingerprint 一致（链确实是这串 token 的）
#                                                                      ③ 抽查第 0、n//2、n−1 块，本进程重算结果相同（同一个 hash 函数）
#                                                                      全过 → 整条采用；否则告警一次、丢弃、自己算（只慢不错）
#   为什么只算"被 prompt 填满的块"（full_prompt_blocks）：prompt 的 token 不会再变；含生成 token 的块可能还挂着延迟输出 /
#     投机解码的占位 token，最后一个不满的块也还会变。生成部分的块由 BlockManager.hash_decode_blocks 在 KV 算好之后再 hash。
#
# 对比nano-vllm：nano-vllm 的 compute_hash 用 xxhash.xxh64() 依次 update(前一块 hash 的 8 字节小端) 和
#   update(np.array(token_ids).tobytes())（Linux 上是 int64）——喂进去的字节流和这里完全相同，所以纯文本 prompt 的 hash 值也一样。
#   ATOM 多出来的：链的根可以是多模态种子（①）；DCP 下 hash block 变大（④）；prompt 链在前端算好、引擎核对后采用（⑤）；
#   一次算一段的 chained_block_hashes（整段只做一次 int64 转换）；_token_bytes 强制 int64。另外 nano-vllm 在 can_allocate 和
#   allocate 里各 hash 一遍，ATOM 的 BlockManager 把每条序列的 prompt 链缓存起来（_prompt_hash_chains），只算一次

import array
from typing import NamedTuple

import numpy as np
import xxhash

from atom.utils import envs

# xxhash.xxh64_intdigest(data, seed=0)：一次算完 64 位 xxHash，直接返回 [0, 2^64) 的 Python int
#   （nano-vllm 是先建 xxh64() 对象再 update，结果相同）。绑定成模块级名字，热路径上省一次属性查找
_xxh64 = xxhash.xxh64_intdigest


def hash_block_size_for(block_size: int, dcp_world_size: int) -> int:
    """Tokens per prefix-cache hash block: one block-table entry under DCP."""
    # 【译】每个前缀缓存 hash block 的 token 数：DCP 下 block_table 的一项。
    # 原理见文件头④。例：block_size 16、DCP 1 → 16；block_size 16、-dcp 4 → 64。
    #   调用方：BlockManager.__init__（self.hash_block_size）、InputOutputProcessor.__init__（前端的
    #   prefix_hash_block_size；前缀缓存关闭时前端不调它，留 0，表示不在前端算链）
    return block_size * dcp_world_size


def full_prompt_blocks(seq, hash_block_size: int) -> int:
    """Hash blocks `seq`'s prompt fills: the part of its chain that cannot change."""
    # 【译】seq 的 prompt 填满了多少个 hash block：它的链里不会再变的那一段。
    # 例：prompt 70 个 token、hash block 16 → 70 // 16 = 4（位置 0~63）；64~69 这 6 个不满一块，不算。
    #   它和准入时查命中的块数不是一回事：查命中只查 ⌈len(seq)/16⌉ − 1 块（最后一块永远不复用），prompt 恰好 64 个 token 时
    #   这里是 4、查命中只查 3 块；链里多出的块留给 prefill 算完后 hash_blocks 登记时直接取用。
    #   调用方：ship_prompt_hash_chain、BlockManager 的 _adopt_shipped_chain / _block_hash / _hit_walk
    return seq.num_prompt_tokens // hash_block_size


def _token_bytes(token_ids) -> bytes:
    # int64 whatever the caller passes: `np.asarray` alone infers int32 from an
    # `array("i")` and int64 from a list.
    # 【译】不管调用方传什么都按 int64：单用 np.asarray 时，array("i") 会被推断成 int32，list 会被推断成 int64。
    # 解释：Sequence.token_ids 是 array("i")（int32），也有调用方传 list；不统一的话同一串 token 会得到两种字节、两种 hash，
    #   前端算的链和引擎算的就对不上。例：token 5 按 int32 是 05 00 00 00，按 int64 是 05 00 00 00 00 00 00 00。
    #   int64 也正是 nano-vllm 的字节格式
    return np.asarray(token_ids, dtype=np.int64).tobytes()


def _link(prev: int, raw) -> int:
    """Hash of one block's token bytes `raw` after `prev` (-1: no parent)."""
    # 【译】一个块的 token 字节 raw 接在 prev 之后的 hash（prev 为 -1：没有父块）。
    # 公式见文件头①。prev.to_bytes(8, "little") 要求 prev 非负：父 hash 是 xxh64 的结果，多模态种子是无符号 64 位的
    #   blake2b 结果，都满足。raw 可以是 bytes 或 memoryview（chained_block_hashes 传切片）：bytes + memoryview 得到新的 bytes
    return _xxh64(raw) if prev == -1 else _xxh64(prev.to_bytes(8, "little") + raw)


def block_hash(token_ids, prev: int = -1) -> int:
    """Chained hash of one block's tokens."""
    # 【译】一个块的 token 的链式 hash。
    # 例：h0 = block_hash([1, 2, 3, 4])；h1 = block_hash([5, 6, 7, 8], prev=h0)。
    #   调用方：BlockManager.compute_hash（hash_blocks 登记、_block_hash 逐块查、_adopt_shipped_chain 抽查，
    #   以及 offload / 远端前缀等【分支】）
    return _link(prev, _token_bytes(token_ids))


def chained_block_hashes(
    token_ids, hash_block_size: int, start: int, end: int, prev: int
) -> list[int]:
    """`block_hash` of hash blocks `[start, end)` of `token_ids`, chained from
    `prev` (block `start - 1`'s hash, or the seed); a short final block hashes
    what it holds."""
    # 【译】token_ids 的第 [start, end) 个 hash block 的 block_hash，从 prev（第 start − 1 块的 hash，或种子）接着链；
    #   最后一块不满时就 hash 它现有的 token。
    # 结果与逐块调用 block_hash 完全相同，只是整段 token 只做一次 int64 转换，再用 memoryview 按块切片（不复制），
    #   循环里只剩 xxh64。例：hbs = 16、start = 2、end = 5：取 token[32:80] → 48 × 8 = 384 字节，每 128 字节一块切 3 段，
    #   h2 = H(prev, 段0)、h3 = H(h2, 段1)、h4 = H(h3, 段2)，返回 [h2, h3, h4]。
    #   调用方：ship_prompt_hash_chain（整条 prompt 链，start = 0、prev = cache_seed）、
    #   BlockManager._hit_walk（缓存的 prompt 链之外的部分，一轮一轮往后算）
    if end <= start:
        return []
    # 每块的字节数（每个 token 8 字节）
    step = hash_block_size * 8
    raw = memoryview(
        _token_bytes(token_ids[start * hash_block_size : end * hash_block_size])
    )
    out: list[int] = []
    # 把绑定方法存成局部变量，省掉循环里每次的属性查找（Python 常见的微优化）
    append = out.append
    h = prev
    for off in range(0, (end - start) * step, step):
        h = _link(h, raw[off : off + step])
        append(h)
    return out


def prompt_fingerprint(seq) -> int:
    """xxh64 of `seq`'s prompt token bytes, without a copy."""
    # 【译】seq 的 prompt token 字节的 xxh64，不拷贝。
    # memoryview(seq.token_ids) 直接看 array("i") 的底层内存（int32），切前 num_prompt_tokens 个，不复制。
    #   它只是"这条链是不是从这串 prompt 算出来的"的指纹，前后端都对同样的 array("i") 计算，所以不需要转 int64。
    #   用途：ShippedChain.fingerprint，引擎在 BlockManager._adopt_shipped_chain 里重算对比（只多一次整段 xxh64）
    return _xxh64(memoryview(seq.token_ids)[: seq.num_prompt_tokens])


class ShippedChain(NamedTuple):
    """`seq.prompt_hash_chain`: its full-prompt-block chain as the frontend
    hashed it, and what it was minted from."""

    # 【译】seq.prompt_hash_chain：前端算出的整条"满 prompt 块"链，以及它是从什么算出来的。
    # Python 语法：NamedTuple 是带字段名的不可变元组。各字段都是给引擎核对用的（见文件头⑤的①②）；
    #   hashes 是 array("Q")（无符号 64 位 C 数组，每个 hash 8 字节），pickle 时是一整段连续字节。
    #   例：prompt 70 个 token、hbs 16、纯文本 → ShippedChain(16, -1, 70, 指纹, array("Q", [h0, h1, h2, h3]))
    hash_block_size: int
    cache_seed: int
    num_prompt_tokens: int
    fingerprint: int  # `prompt_fingerprint`
    hashes: array.array  # "Q"


def ship_prompt_hash_chain(seq, hash_block_size: int) -> None:
    """Attach `seq`'s chain for `BlockManager._adopt_shipped_chain`."""
    # 【译】给 seq 挂上它的链，供 BlockManager._adopt_shipped_chain 使用。
    # 调用方：InputOutputProcessor.preprocess_fanout（每个请求算一次，n > 1 的兄弟序列共享结果）
    # 开关关掉（ATOM_FRONTEND_PREFIX_HASH=0）或前端没开前缀缓存（hbs = 0）时不挂链，全部由引擎自己算
    if not envs.ATOM_FRONTEND_PREFIX_HASH or hash_block_size <= 0:
        return
    # 只算被 prompt 填满的块，从种子 cache_seed 开始链（见文件头⑤最后一段）
    n = full_prompt_blocks(seq, hash_block_size)
    chain = chained_block_hashes(seq.token_ids, hash_block_size, 0, n, seq.cache_seed)
    seq.prompt_hash_chain = ShippedChain(
        hash_block_size,
        seq.cache_seed,
        seq.num_prompt_tokens,
        prompt_fingerprint(seq),
        array.array("Q", chain),
    )


def take_shipped_chains(seqs) -> list | None:
    """`seqs`' shipped chains, detached for the ADD frame that carries them
    beside the seqs (a pickled `Sequence` drops its chain); None if none has
    one. The frontend keeps no copy."""
    # 【译】把 seqs 的随行链摘下来，交给在 seqs 旁边携带它们的 ADD 帧（pickle 后的 Sequence 会丢掉自己的链）；
    #   一个都没有时返回 None。前端不保留副本。
    # 调用方：engine_core_mgr.py 的 _add_frame（CoreManager 打包 ADD 帧）。返回的列表与 seqs 一一对应（某些位置可以是 None）；
    #   any(chains)：NamedTuple 是非空元组、为真，None 为假。返回 None 时帧里就不带第 3 个元素
    chains = [getattr(seq, "prompt_hash_chain", None) for seq in seqs]
    if not any(chains):
        return None
    # 前端这边的 Sequence 对象还留在 requests 表里直到请求结束，摘掉链免得它一直占着内存
    for seq in seqs:
        seq.prompt_hash_chain = None
    return chains


def attach_shipped_chains(seqs, chains=None) -> None:
    """Hand an ADD frame's chains (`take_shipped_chains`) back to its seqs."""
    # 【译】把一个 ADD 帧里的链（take_shipped_chains 的结果）交还给它的 seqs。
    # 调用方：EngineCore 的输入线程 attach_shipped_chains(reqs, *chains)——chains 是帧在 seqs 之后的部分，
    #   帧里没带链时为空，这里 chains 取默认值 None，什么都不做。
    #   zip(..., strict=True)（Python 3.10+）：两边长度不同就抛 ValueError，而不是悄悄截断
    if chains:
        for seq, chain in zip(seqs, chains, strict=True):
            seq.prompt_hash_chain = chain
