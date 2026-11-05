# 【导读】lm_head_argmax.py：词表被切到多张卡上时，"每个分片先各自求 argmax、只交换 (最大值, token id)"的 Triton kernel
#   （第 5 批 5d，B 级）
#   前置：lm_head 怎样按词表切。lm_head 是 hidden [N, H] × 权重 [H, V] → logits [N, V]。TP=tp 时 ParallelLMHead 把权重
#     按词表（输出维）切成 tp 份，rank r 只存 token id ∈ [r·V/tp, (r+1)·V/tp) 那一段，GEMM 只算出自己的 [N, V/tp]
#     （按输出维切 = 列切，直觉见 arg_utils.py 中 tensor_parallel_size 字段上方的【科普】）
#   普通路径（主模型采样用，ParallelLMHead.forward）：各 rank 的 [N, V/tp] all-gather 成 [N, V]，每个 rank 一份完整 logits → Sampler
#   只要贪心 argmax 时不必拼出完整 logits（ParallelLMHead.compute_argmax_token）：
#
#     rank r：x [N, H] ──GEMM──> logits_r [N, V/tp] ──本 kernel──> packed_r [N, 2] = (本分片最大值, 它的全局 token id)
#                                                                        │ all_gather（只交换 [N, 2]）
#                                                                        ▼
#             [tp, N, 2] ── 对第 0 维 argmax，选出赢的 rank ──> 取它记录的 id ──> token [N]
#
#   注意：GEMM 本身没有和 argmax 融合——每个 rank 仍要写出、再读回自己那份 [N, V/tp] logits；
#     省下的是跨卡通信、完整 [N, V] 的落地，以及之后在完整词表上的那次 argmax
#   数字（Qwen3 词表 V=151936，tp=8，N=64 行，bf16）：
#     每个 rank 的分片 [64, 18992] ≈ 2.3 MiB（两条路径都要写、读这一份）
#     普通路径：每个 rank 收 7 份分片 ≈ 17.0 MB，拼出 [64, 151936] ≈ 18.5 MiB
#     本路径：每个 rank 发 [64, 2] fp32 = 512 B，收齐 [8, 64, 2] = 4 KiB——通信量约为普通路径的 1/4700
#   为什么打包成 fp32：值和 id 放进同一个张量，一次 all_gather 就够；bf16 的值转 fp32 无损，
#     token id < 2^24 = 16,777,216 时 fp32（24 位有效数字）能精确表示，词表远小于这个数
#   平局：分片内取 id 最小的；跨 rank 的 argmax 取第一个最大值 = rank 最小 = 词表区间最靠前，
#     整体就等于"全词表 argmax、平局取最小 id"，与普通路径逐位相同（理由见 embed_head.py 中 compute_argmax_token 的 docstring）
#
# 调用方（grep 核实）：
#   embed_head.py ParallelLMHead.compute_argmax_token 的 tp > 1 分支（tp = 1 时直接 topk_select，不经过这里）
#     ← 各草稿模型的 compute_draft_ids（deepseek_mtp.py、qwen3_next_mtp.py、eagle3_llama.py 等 9 个模型文件）、
#       vLLM 插件 model_wrapper.py、SGLang 插件 dflash_lm_head_bridge.py：贪心起草时每步只要 argmax
#   embed_head.py ParallelLMHead._dp_sharded_logits(mode="argmax")：纯 DP 时把词表按 DP rank 切开做草稿 argmax
#     （ATOM_DP_DRAFT_ARGMAX，默认开）。动机是带宽：草稿步只有很少几行，GEMM 耗时由读 lm_head 权重决定。
#     估算（DeepSeek-V3：V=129280、H=7168，bf16 权重约 1.85 GB；HBM ≈ 5.3 TB/s）：读完整权重约 0.35 ms，
#     dp=8 切分后每个 rank 只读 1/8，约 0.044 ms；交换的仍只是 [Σ行, 2]
#   主模型的 Sampler 不用它：主模型采样要完整 logits（温度 / top-k / top-p / logprobs），全贪心也是在 all-gather 之后做 topk_select
#   对比nano-vllm：nano-vllm 的 ParallelLMHead 用 dist.gather 把 logits 收到 rank 0，没有投机解码，也就没有这种只要 argmax 的场景
import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

# Cap the vector width; larger vocabularies stream through multiple tiles.
# 【译】给向量宽度设上限；更大的词表分成多个 tile 依次流过。
# 解释：一个 program 一次最多读 131072 个 logit；例：分片宽 M=18992 → BLOCK_M = next_pow2 = 32768，一个 tile 就够
_MAX_BLOCK_M = 131072


# Triton kernel：grid = (N,)，每个 program 负责一行 logits
#   输入：logits_ptr [N, M]（M 是本分片的词表宽度——注意这里 M 指列数，不是行数），vocab_start_idx = 本分片第一个 token 的全局 id
#   输出：packed_ptr [N, 2] fp32，第 0 列是最大值，第 1 列是全局 token id
#   做法：按 BLOCK_M 一块块扫过这一行；每块求最大值，和"等于最大值的位置里最小的下标"；再与之前的最优比较
@triton.jit
def _lm_head_argmax_pack_kernel(
    logits_ptr,
    packed_ptr,
    vocab_start_idx,
    M: tl.constexpr,
    stride_logits_n: tl.constexpr,
    stride_logits_m: tl.constexpr,
    stride_packed_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    block_offs = tl.arange(0, BLOCK_M)
    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    # 哨兵：下标 M 不是合法列，tl.min 时比任何合法下标都大
    invalid_idx = tl.full((), M, dtype=tl.int64)
    best_idx = invalid_idx

    for block_start in tl.range(0, M, BLOCK_M):
        offs = block_start + block_offs
        mask = offs < M
        vals = tl.load(
            logits_ptr + row * stride_logits_n + offs * stride_logits_m,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        # 越界位置读成 -inf，不会成为最大值；块内最大值，以及等于它的位置里最小的下标（块内平局取 id 小的）
        block_max = tl.max(vals, axis=0)
        idxs = offs.to(tl.int64)
        block_idx = tl.min(
            tl.where((vals == block_max) & mask, idxs, invalid_idx), axis=0
        )
        # 更大才换；相等时只在下标更小时换——后面的块下标都更大，所以相等时总是保留前面的，平局取最小 id
        take_block = (block_max > best_val) | (
            (block_max == best_val) & (block_idx < best_idx)
        )
        best_val = tl.where(take_block, block_max, best_val)
        best_idx = tl.where(take_block, block_idx, best_idx)

    # 局部下标 + 分片起点 = 全局 token id，转成 fp32 与最大值存在同一行
    global_idx = best_idx + vocab_start_idx
    tl.store(packed_ptr + row * stride_packed_n, best_val)
    tl.store(packed_ptr + row * stride_packed_n + 1, global_idx.to(tl.float32))


# fake 实现：torch.compile 追踪时只需要知道输出的形状和 dtype（[N, 2] fp32），不真正启动 kernel
def _lm_head_argmax_pack_fake(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 2), dtype=torch.float32, device=logits.device)


# torch_compile_guard（aiter）：把这个函数注册成 torch.compile 眼里不透明的自定义算子，Dynamo 不追踪进 Triton launch，
#   用上面的 fake 推断形状；草稿 forward 会被编译 / 录进 CUDA Graph，所以需要这层包装
# 入口：logits [N, M]（本分片），vocab_start_idx → [N, 2] fp32；调用方见文件开头
@torch_compile_guard(gen_fake=_lm_head_argmax_pack_fake)
def lm_head_argmax_pack(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    """Reduce and fp32-pack local LM-head argmax in one Triton launch."""
    # 【译】在一次 Triton launch 里完成本地 LM-head argmax 的规约和 fp32 打包。
    if logits.dim() != 2:
        raise ValueError("lm_head_argmax_pack expects a 2-D logits tensor")

    N, M = logits.shape
    if N == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=logits.device)
    if M == 0:
        raise ValueError("lm_head_argmax_pack requires a non-empty vocabulary")

    packed = torch.empty((N, 2), dtype=torch.float32, device=logits.device)
    # tile 宽度取不小于 M 的 2 的幂，最多 131072；宽 tile 用 8 个 warp
    block_m = min(_MAX_BLOCK_M, triton.next_power_of_2(M))
    num_warps = 8 if block_m >= 2048 else 4

    _lm_head_argmax_pack_kernel[(N,)](
        logits,
        packed,
        vocab_start_idx,
        M=M,
        stride_logits_n=logits.stride(0),
        stride_logits_m=logits.stride(1),
        stride_packed_n=packed.stride(0),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=2,
    )
    return packed
