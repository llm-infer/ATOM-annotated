# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】sampler.py：GPU 上的批量采样——logits [bs, V] → 每行一个 token id [bs]（第 5 批 5d，A 级）
#   位置（一步 forward 里，每个 TP rank 都执行）：
#     ModelRunner.forward
#       ├─ prepare_sample（CPU，numpy）：钳制温度、判断 all_greedy / 是否要过滤 / 过滤参数整批是否相同
#       ├─ run_model → logits（lm_head 输出；TP>1 时 ParallelLMHead 已 all-gather 成完整词表，每个 rank 都有一份）
#       └─ postprocess → self.sampler(...)                  ← 本文件 Sampler.forward
#             └─ sampled [bs] → tokenIDProcessor 异步拷回 CPU（延迟输出，见 engine_core.py 文件开头【3】）
#     每个 rank 各自采样（各用各的随机数）；开延迟输出或有草稿模型时，postprocess 随后把 rank 0 的结果 broadcast
#     给其他 rank，保证下一步各 rank 的输入相同；最终只有 rank 0 的结果回到 EngineCore
#   请求级的概念——温度 / top-k / top-p 怎样改变分布、两者取交集、重新归一化的 5-token 数值例子、三条路径的流程图、
#     "适用场景"表——都在 sampling_params.py 文件头，这里不重复；
#     本文件讲"在 GPU 上怎样批量实现"：张量形状、三条路径各用什么 kernel、为什么不能有 GPU→CPU 同步
#
# 阅读路线
#   ① 下面的【科普】①~③：argmax(p/E) 为什么等于按概率抽样；为什么采样里不能读回 GPU 的值；一个 3 行 batch 带形状走一遍
#   ② Sampler.forward：三条路径的分派（全贪心 / 只有温度 / 要过滤）
#   ③ _temperature_sample（不过滤，一个融合 kernel）→ _topk_topp_sample → _aiter_sample / _native_sample（要过滤）
#   ④ _apply_greedy_tokens：混合 batch 里 T=0 的行怎样换成 argmax
#   ⑤ 【分支】sample_verification_tokens：投机解码时目标模型在各草稿位置的采样（验证本身在 rejection_sampler.py，第 7 批）
#   logprobs 不在本文件：ModelRunner.postprocess 在采样之后对*未除温度*的 logits 做 log_softmax，再 gather 出采到的 token
#
# 张量形状（bs = 本步序列数；sample_verification_tokens 的行数是草稿 token 总数）
#   logits         [bs, V]  模型 dtype（通常 bf16），V = 完整词表（例：Qwen3 为 151936）
#   temperatures   [bs]     fp32，GPU，已钳制到 >= SAMPLER_EPS = 1e-10（请求 T=0 到这里变成 1e-10）
#   top_ks         None（没人过滤）/ int（整批相同，CPU 标量）/ [bs] int32 GPU 张量；-1 = 该行不过滤
#   top_ps         None / float / [bs] fp32 GPU 张量；1.0 = 该行不过滤
#   all_greedy     CPU 上的布尔值（numpy 算好：整批 T 都 == 0）
#   返回           [bs]     int32 token id
#
# 【科普】① 为什么 argmax(p / E) 就是"按概率 p 抽一个"（指数竞赛，也叫 Gumbel-max 技巧）
#   朴素做法：每行求累计和 cumsum(p)，抽 u ~ U(0,1)，找第一个 cumsum >= u 的位置（torch.multinomial 一类）。
#     要做前缀和 + 查找，按行有数据相关的控制流。
#   技巧：给词表里每个 token i 独立抽一个 E_i ~ Exp(1)（均值为 1 的指数分布），取 argmax_i p_i / E_i
#     它等于 argmin_i E_i / p_i；E_i / p_i 服从"速率为 p_i 的指数分布"，一组独立指数变量里谁最小，
#     恰好以 p_i / Σp 的概率是第 i 个 → 选中 i 的概率就是 p_i
#     取 log：argmax(log p_i − log E_i)，而 −log E_i 正是 Gumbel(0,1) 噪声，所以也叫 Gumbel-max
#   好处：只有逐元素除法和一次按行 argmax，没有前缀和与分支，一个 kernel 就能做完
#     （ATOM 用 aiter 的 mixed_sample_outer_exponential）；p 也不必归一化：softmax 的分母对一行里所有 token 相同，不改变 argmax
#   例（V=5，沿用 sampling_params.py 的 logits=[3, 2, 1, 0.5, -1]，T=1 → p ≈ [0.624, 0.229, 0.084, 0.051, 0.011]）：
#     抽到 E = [0.9, 0.2, 1.5, 0.05, 2.0] → p/E ≈ [0.69, 1.15, 0.06, 1.02, 0.01] → 选 1 号
#     换一组 E 再抽，选中 0 号的频率趋于 0.624、1 号趋于 0.229……
#   对比nano-vllm：layers/sampler.py 用的是同一技巧（softmax → probs.div_(exponential_) → argmax），但每步为 [B, V] 的
#     每个元素各抽一个 E；ATOM 默认整批共用一行 E（见 _temperature_sample）。其余差别见文件头末尾的"对比nano-vllm"
#
# 【科普】② 为什么本文件处处避开"在 Python 里读 GPU 上的值"
#   前置：GPU 异步执行、stream 和同步点，见 engine_core.py 文件开头【3】
#   采样发生在 postprocess 里，此时本步 forward 的 kernel 很可能还在 GPU 队列里排着。开着延迟输出时，CPU 要趁 GPU 还在算，
#     把采样 kernel 也发射出去、启动异步 D2H 拷贝，然后回到 EngineCore 处理上一步的结果、调度下一步（engine_core.py【3】的时间线）。
#   只要采样里出现 `if mask.any():`、`.item()`，或布尔下标 `x[mask]`（输出长度取决于数据，要先把 mask 读回 CPU），
#     CPU 就得停下来等本步 forward 全部跑完，这段重叠就没了。
#   所以：是否全贪心、是否需要过滤、过滤参数是否整批相同，都由 ModelRunner.prepare_sample 在 CPU 上用 numpy 判断，
#     传进来的是 Python 标量或 None；GPU 上只用 torch.where 这类形状固定的操作（_apply_greedy_tokens）
#
# 【科普】③ 一个 3 行 batch 在 GPU 上走一遍（V=5；三行 logits 都取 sampling_params.py 例子里的 [3.0, 2.0, 1.0, 0.5, -1.0]）
#   请求 A：T=0（贪心）   B：T=1.0   C：T=0.7；都没开 top-k / top-p
#   ┌ prepare_sample（CPU，numpy）─────────────────────────────────────────────────────────────
#   │ all_greedy = False（B、C 的 T 不是 0）     temperatures = max(T, 1e-10) = [1e-10, 1.0, 0.7] → GPU [3]
#   │ top_ks = top_ps = None（没人开过滤）         needs_independent_noise = False（batch 里没有 n>1 的兄弟序列）
#   └────────────────────────────────────────────────────────────────────────────────────────
#   Sampler.forward：不是全贪心、不需要过滤 → _temperature_sample
#     E = 一行 Exp(1) 噪声 [1, 5]，假设抽到 [0.9, 0.2, 1.5, 0.05, 2.0]，expand 成 [3, 5]（三行读同一块内存）
#     mixed_sample_outer_exponential：每行 argmax(softmax(logits / T) / E) → sampled [3] int32
#       B：p ≈ [0.624, 0.229, 0.084, 0.051, 0.011]，p/E ≈ [0.69, 1.15, 0.06, 1.02, 0.01] → 1
#       C：p ≈ [0.753, 0.180, 0.043, 0.021, 0.002]，p/E ≈ [0.84, 0.90, 0.03, 0.42, 0.00] → 1
#       A：T=1e-10，这里的结果无所谓，下一步会被覆盖
#   _apply_greedy_tokens：greedy = argmax(logits) = [0, 0, 0]；torch.where(T <= 1e-10, greedy, sampled) → [0, 1, 1]
#   全程没有读回 CPU；[3] int32 交给 tokenIDProcessor 异步拷回
#   注意 B、C 共用同一行 E，这一步它们的结果是相关的；单看任何一行，分布仍然正确
#   变化一：C 改成 top_p=0.9 → 三行 top_p = [1.0, 1.0, 0.9] 不相同，传 [3] 的 GPU 张量（三行都是 0.9 才传标量 0.9）
#     → 整批走 _topk_topp_sample：probs = softmax(logits / T) [3, 5] fp32 → aiter 的 top_p_sampling_from_probs → 贪心修正
#     A、B 两行 top_p=1.0 等于不过滤，但也一起离开了融合快路径（即 sampling_params.py 适用场景表末尾的"代价提示"）
#   变化二：B 是 n=2 的请求（兄弟 B1、B2，logits 和 T 都相同，batch 变成 4 行）→ 共用一行 E 时两者每步都采到同一个 token，
#     所以 prepare_sample 置 needs_independent_noise=True，E 变成 [4, 5]、每行独立
#
# 对比nano-vllm：nano-vllm 的 Sampler 只有温度一个参数（logits/T → softmax → argmax(p/E)，用 @torch.compile 融合），
#   不允许贪心（它的 SamplingParams 断言 temperature > 1e-10）、没有 top-k / top-p、没有投机解码验证。
#   ATOM 多出来的部分主要为了两件事：① 全程不在 Python 里读 GPU 的值（【科普】②）；② 平局与噪声的可复现

# 导入说明：
#   mixed_sample_outer_exponential：不过滤时的融合采样 kernel（argmax(p/E)，见【科普】①）
#   topk_select：按行取前 k 个（这里只用 k=1 当 argmax），tie="low" 指定平局取 id 最小的
#   aiter.ops.triton 的 softmax / topk：只被文件末尾的旧方法 greedy_sample / random_sample 使用
import warnings

import torch
from aiter import mixed_sample_outer_exponential, topk_select
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.topk import topk
from torch import nn

# Try to import aiter top-k/top-p sampling ops
# 【译】尝试导入 aiter 的 top-k / top-p 采样算子
# 解释：import aiter.ops.sampling 的作用是把 top_k_top_p_sampling_from_probs 等算子注册到 torch.ops.aiter 下；
#   导入失败时 AITER_TOPK_TOPP_AVAILABLE=False，要过滤的 batch 改走 _native_sample（纯 PyTorch 回退）
try:
    import aiter.ops.sampling  # noqa: F401

    aiter_ops = torch.ops.aiter
    AITER_TOPK_TOPP_AVAILABLE = True
except ImportError:
    AITER_TOPK_TOPP_AVAILABLE = False
    warnings.warn(
        "aiter.ops.sampling not available. Top-k/top-p sampling will use "
        "experimental native PyTorch implementation as fallback.",
        UserWarning,
        stacklevel=1,
    )

# Track whether we've already warned about native sampling being used
# 【译】记录是否已经就"正在使用原生采样"发出过警告
_NATIVE_SAMPLING_WARNING_ISSUED = False

# Epsilon value for numerical stability and to prevent division by 0
# 【译】用于数值稳定、防止除以 0 的 epsilon
# 解释：两处用途——ModelRunner.prepare_sample 用它钳制温度（np.maximum(T, SAMPLER_EPS)），
#   _apply_greedy_tokens 用 T <= SAMPLER_EPS 认出"请求的是 T=0"的贪心行
SAMPLER_EPS = 1e-10


# 混合 batch 的贪心修正：T=0 的行换成 argmax，其余行保留采样结果
#   probs：[bs, V]（不过滤路径传的是原始 logits，过滤路径传的是 softmax 后的 probs——两者 argmax 相同）
#   temperatures：[bs]（已钳制）；next_tokens：[bs] 或 [bs, 1]（torch.multinomial 返回 [bs, 1] int64）
#   返回 [bs] int32
#   调用方：Sampler.forward（不过滤路径）、_aiter_sample、_native_sample；全贪心 batch 不经过这里（forward 开头直接 argmax）
def _apply_greedy_tokens(
    probs: torch.Tensor, temperatures: torch.Tensor, next_tokens: torch.Tensor
) -> torch.Tensor:
    """Keep row selection on the device, with a fixed-size int32 result.

    A Python test of ``greedy_mask.any()`` waits for the GPU. Boolean indexing
    also needs a data-dependent output size. Reduce all rows and select with
    ``where`` so neither operation needs to read the mask back on the CPU.
    ``tie="low"`` preserves argmax's lowest-token-ID tie breaking.

    A greedy row is one at or below ``SAMPLER_EPS``: the runner clamps every
    temperature to it before the sampler sees them (``prepare_sample``), so a
    requested 0 never arrives as 0.
    """
    # 【译】把按行选择留在设备上完成，结果是固定大小的 int32 张量。
    #
    #   在 Python 里判断 ``greedy_mask.any()`` 要等 GPU。布尔下标还需要一个取决于数据的输出大小。
    #   这里对所有行都做规约，再用 ``where`` 选择，这样两个操作都不需要把 mask 读回 CPU。
    #   ``tie="low"`` 保持 argmax"平局取最小 token ID"的规则。
    #
    #   贪心行是温度 <= ``SAMPLER_EPS`` 的行：runner 在采样器看到温度之前（``prepare_sample``）就把每个温度
    #   钳到了这个值，所以请求的 0 到这里从来不是 0。
    # 每一行都算 argmax：贪心行用得上，其余行白算——一次 [bs, V] 的按行规约，比把 mask 读回 CPU 便宜得多（【科普】②）
    #   topk_select 返回 (值, 下标)，下标 [bs, 1] int32 → view(-1) → [bs]
    #   为什么贪心行不能直接用采样结果：T=1e-10 时 logits/T 让分布几乎是 one-hot，采样结果通常就是 argmax；
    #   但两个 token 的 logit 完全相等时概率各一半，噪声会随机挑一个——这里统一换成 id 小的那个，结果可复现
    #   （tests/test_sampler_greedy_rows.py 的 test_mixed_batch_greedy_rows_take_the_low_tie 测的就是这个平局）
    greedy_tokens = topk_select(probs, 1, tie="low")[1].view(-1)
    # torch.where(条件, a, b)：逐元素选择，形状固定，不产生同步
    #   例：temperatures = [1e-10, 1.0, 0.7]，greedy = [0, 0, 0]，sampled = [3, 1, 1] → [0, 1, 1]
    return torch.where(
        temperatures <= SAMPLER_EPS, greedy_tokens, next_tokens.view(-1).to(torch.int)
    )


# 生成一行 [1, V] 的 Exp(1) 噪声，供 _temperature_sample 广播给整批共用
#   调用方：只有 _temperature_sample（needs_independent_noise=False 时）
def get_per_token_exponential(vocab_size: int, device) -> torch.Tensor:
    """Returns a tensor of shape (1, vocab_size) filled with exponential random values.
    This is key to deterministic inference, as it ensures that the same random values are used for each token across different runs.
    """
    # 【译】返回形状为 (1, vocab_size)、填满指数分布随机值的张量。
    #   这是确定性推理的关键：它保证不同次运行中，每个 token 使用相同的随机值。
    # 观察：这里和 _temperature_sample 的 docstring 都把它说成"缓存的一行"，但实现每次调用都重新 exponential_(1)，
    #   并没有缓存，每步都是新的噪声。它真正带来的性质是：
    #   ① 每步只消耗 V 个随机数，与 batch 大小无关，随机数流的推进不受 batch 组成影响；
    #   ② 同一步里所有行读同一行 E，某一行的结果只取决于它自己的 logits、温度和这一步的 E，与它排第几行、batch 有几行无关
    #   推理路径里没有调用 torch.manual_seed（grep 只在两个 _selftest 里找到；rejection_sampler.py 只给合成接受率用的
    #   独立 generator 设种子），用的是 PyTorch 默认生成器的默认种子
    #   代价：同一步里 logits 和温度都相同的两行（n>1 的兄弟序列）必然采到同一个 token，所以它们要用独立噪声
    return torch.empty((1, vocab_size), dtype=torch.float, device=device).exponential_(
        1
    )


# 采样器：ModelRunner.__init__ 里创建一个（self.sampler = Sampler()），每步在 postprocess 里调用
#   继承 nn.Module 只是沿用习惯（与 nano-vllm 相同），它没有参数也没有 buffer；
#   与 nano-vllm 不同，forward 没有 @torch.compile——融合交给 aiter 现成的 kernel
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        # 传给融合 kernel 的 eps，以及 _native_sample 归一化分母的下限
        self.eps = SAMPLER_EPS

    # 【分支】投机解码（MTP / EAGLE / DSpark，第 7 批）：为目标模型在每个草稿位置的输出各采一个 token
    #   输入：logits [num_draft_rows, V]（postprocess 按 target_logits_indices 从本步 logits 里选出的目标行）
    #         cu_num_draft_tokens [bs]：各请求草稿数的包含式累计和，例：草稿数 [3, 0, 2] → [3, 3, 5]
    #         temperatures / top_ks / top_ps：按*请求*给（[bs] 张量或标量）
    #   输出：[num_draft_rows] int32；rejection_sampler 拿它和草稿 token 逐个比较
    #   调用方：ModelRunner.postprocess 的投机分支，只在 not all_greedy 时调用（全贪心时验证直接比 argmax）；
    #     调用后从 TP rank 0 broadcast，因为各 rank 的噪声不同
    def sample_verification_tokens(
        self,
        logits: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample each draft-conditioned target row with independent noise.

        Accepting a draft when it matches this draw and stopping at the first
        mismatch preserves the target distribution. Sharing noise across rows
        would condition later draws on earlier acceptance decisions.
        """
        # 【译】对每一个"以草稿为条件"的目标行，用独立的噪声采样。
        #
        #   草稿与这次采样结果相同就接受、在第一个不同处停下，这样输出仍服从目标模型的分布。
        #   若各行共用噪声，后面的采样就会以前面的接受决定为条件。
        # 解释：第 j 行是目标模型在"前缀 + 前 j 个草稿"之后的分布 p_j。草稿 d_j 被接受，当且仅当它等于从 p_j 独立采到的 s_j；
        #   第一次不等时输出 s_j 并停止。于是每个输出位置的 token 都是从对应的 p_j 里采的——与不投机时同分布。
        #   这要求各行的 s_j 相互独立：若共用一行 E，s_0 与 s_1 相关，"d_0 被接受"这件事就会改变 s_1 的分布。
        #   （这种"比对采样结果"的验证不需要草稿模型的概率 q，与经典的 min(1, p/q) 拒绝采样不同；
        #    RejectionSampler.forward 调 rejection_sample 时 draft_probs 传的就是 None）
        rows = logits.shape[0]
        if rows == 0:
            return torch.empty(0, device=logits.device, dtype=torch.int32)
        # 每一行属于哪个请求：searchsorted(right=True) 找第一个"cu > 行号"的位置
        #   例：cu = [3, 3, 5]，行号 [0, 1, 2, 3, 4] → [0, 0, 0, 2, 2]（请求 1 没有草稿，不占行）
        request_indices = torch.searchsorted(
            cu_num_draft_tokens,
            torch.arange(rows, device=logits.device, dtype=cu_num_draft_tokens.dtype),
            right=True,
        )

        # 把按请求给的参数展开成按行：标量或单元素张量原样返回（整批相同），[bs] 张量按 request_indices 取
        def expand(values):
            if not isinstance(values, torch.Tensor) or values.numel() == 1:
                return values
            return values[request_indices]

        # all_greedy=False：调用方已排除全贪心；needs_independent_noise=True：每行一份新噪声（见上面的解释）
        #   温度在 prepare_sample 里总是 [bs] 张量，所以直接按行展开
        return self(
            logits,
            temperatures[request_indices],
            expand(top_ks),
            expand(top_ps),
            all_greedy=False,
            needs_independent_noise=True,
        )

    # 主入口：ModelRunner.postprocess 每步调用（投机解码时还用于 bonus 行，以及经 sample_verification_tokens 调用）
    # 【译】（签名里的行尾注释）logits 形状 (num_tokens, vocab_size)；temperatures 为 (num_tokens,)；
    #   top_ks：(num_tokens,) int32，-1 表示禁用；top_ps：(num_tokens,) float32，1.0 表示禁用；
    #   all_greedy：所有温度都为 0 时为 True（在 CPU 上判断）；返回 (num_tokens,)
    def forward(
        self,
        logits: torch.Tensor,  # (num_tokens, vocab_size)
        temperatures: torch.Tensor,  # (num_tokens,)
        top_ks: (
            int | torch.Tensor | None
        ) = None,  # (num_tokens,) int32, -1 means disabled
        top_ps: (
            float | torch.Tensor | None
        ) = None,  # (num_tokens,) float32, 1.0 means disabled
        all_greedy: bool = False,  # True if all temperatures are 0 (checked on CPU)
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:  # (num_tokens,)
        """
        Sample tokens from logits using temperature or top-k top-p filtering.

        Args:
            logits: Raw logits from model (num_tokens, vocab_size)
            temperatures: Temperature for each token (num_tokens,), pre-clamped to eps
            top_ks: Uniform CPU scalar or per-token tensor; -1 means disabled
            top_ps: Uniform CPU scalar or per-token tensor; 1.0 means disabled
            all_greedy: True if all requests use greedy sampling (checked on CPU)
            needs_independent_noise: True when the batch contains fan-out
                siblings (SamplingParams.n>1). Forces fresh per-row random
                exponential noise so sibling sequences with identical logits
                do not produce identical tokens. Adds an O(bs * vocab_size)
                allocation, so only enabled when actually needed.

        Returns:
            Sampled token IDs (num_tokens,)
        """
        # 【译】用温度采样或 top-k / top-p 过滤，从 logits 中采样 token。
        #
        #   参数：
        #       logits：模型输出的原始 logits (num_tokens, vocab_size)
        #       temperatures：每个 token 的温度 (num_tokens,)，已预先钳制到 eps
        #       top_ks：整批相同时为 CPU 标量，否则为逐 token 的张量；-1 表示禁用
        #       top_ps：整批相同时为 CPU 标量，否则为逐 token 的张量；1.0 表示禁用
        #       all_greedy：所有请求都用贪心采样时为 True（在 CPU 上判断）
        #       needs_independent_noise：batch 中含有扇出的兄弟序列（SamplingParams.n>1）时为 True。
        #           强制为每行重新生成随机指数噪声，让 logits 相同的兄弟序列不会产生相同的 token。
        #           会多一次 O(bs * vocab_size) 的显存分配，所以只在确实需要时开启。
        #
        #   返回：
        #       采样得到的 token ID (num_tokens,)
        #
        # 三条路径（选哪条全在 CPU 上决定，见文件开头【科普】②）：
        #   all_greedy                     → topk_select(logits, 1)：一次按行 argmax，不做 softmax
        #   top_ks、top_ps 都是 None       → _temperature_sample：一个融合 kernel 做 argmax(p/E)，再做贪心修正
        #   否则（batch 里有人开了过滤）   → _topk_topp_sample：softmax 成 probs，再走 aiter 采样算子 / 回退实现
        # Every path starts here: an all-greedy batch is an argmax whatever its
        # filters. `tie="low"` is what makes this the pick `torch.argmax` made:
        # the default promises no direction among equal scores, so a tie would
        # resolve differently from one run (or TP rank) to the next. `logits` is
        # bf16 and stays bf16 -- the reduction widens each element as it reads it.
        # 【译】每条路径都从这里开始：全贪心的 batch 不管带什么过滤参数，都只是一次 argmax。`tie="low"` 才让它与
        #   `torch.argmax` 的选择一致：默认规则对相等的分数不保证取哪个方向，所以一次平局在不同的运行（或不同的
        #   TP rank）之间可能有不同的结果。`logits` 是 bf16 且保持 bf16——规约在读取每个元素时就把它加宽。
        # 解释：argmax(logits) = argmax(softmax(logits))，所以不需要 softmax，也就不用落地一份 [bs, V] 的 fp32 中间结果
        #   （例：bs=256、V=151936 的 fp32 张量约 148 MiB）
        if all_greedy:
            return topk_select(logits, 1, tie="low")[1].view(-1)

        # No Top-K Top-P parameters, perform temperature-based sampling
        # 【译】没有 Top-K / Top-P 参数，做基于温度的采样
        # 解释：融合 kernel 采完后，再用 _apply_greedy_tokens 把 T=0 的行换成 argmax（这里传的是原始 logits）
        if not self._needs_filtering(top_ks, top_ps):
            sampled = self._temperature_sample(
                logits, temperatures, needs_independent_noise=needs_independent_noise
            )
            return _apply_greedy_tokens(logits, temperatures, sampled)

        # Apply top-k/top-p filtering
        # 【译】做 top-k / top-p 过滤
        # 解释：贪心修正在 _aiter_sample / _native_sample 的最后一步完成
        return self._topk_topp_sample(
            logits,
            temperatures,
            top_ks,
            top_ps,
            needs_independent_noise=needs_independent_noise,
        )

    def _needs_filtering(
        self,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
    ) -> bool:
        """Check if any request needs top-k or top-p filtering.

        This check is O(1) - the actual filtering check is done on CPU in
        model_runner.prepare_sample(), which passes None if no filtering needed.
        """
        # 【译】检查是否有请求需要 top-k 或 top-p 过滤。
        #
        #   这个检查是 O(1) 的——真正"要不要过滤"的判断在 CPU 上的 model_runner.prepare_sample() 里完成，
        #   不需要过滤时它传入 None。
        return top_ks is not None or top_ps is not None

    def _temperature_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:
        """Temperature-based Gumbel-max sampling.

        When ``needs_independent_noise`` is True the per-row exponential noise
        tensor is freshly drawn with shape ``(num_tokens, vocab_size)`` so that
        fan-out siblings produced by ``SamplingParams.n > 1`` diverge instead
        of collapsing onto the same token when they share logits. Otherwise we
        keep the cached ``(1, vocab_size)`` row broadcasted across the batch,
        which preserves the existing run-to-run determinism optimization.
        """
        # 【译】基于温度的 Gumbel-max 采样。
        #
        #   ``needs_independent_noise`` 为 True 时，逐行的指数噪声张量重新生成，形状 ``(num_tokens, vocab_size)``，
        #   这样 ``SamplingParams.n > 1`` 扇出的兄弟序列在共享 logits 时会分叉，而不会塌缩到同一个 token。
        #   否则沿用缓存的 ``(1, vocab_size)`` 一行、在整批上广播，保留原有的"运行间确定性"优化。
        #   （"缓存"与实现不符，见 get_per_token_exponential 下的观察）
        num_tokens, vocab_size = logits.shape
        # 输出 [bs] int32，由下面的 kernel 写入
        sampled_tokens = torch.empty(num_tokens, dtype=torch.int, device=logits.device)
        if needs_independent_noise:
            # 每行各自一份噪声 [bs, V] fp32。显存：bs=256、V=151936 时约 148 MiB，所以只在有兄弟序列时才开
            exponential = torch.empty(
                (num_tokens, vocab_size), dtype=torch.float, device=logits.device
            ).exponential_(1)
        else:
            # 一行噪声 [1, V]（V=151936 时约 594 KiB），expand 成 [bs, V]：只把第 0 维的 stride 设为 0，不复制内存
            exponential = get_per_token_exponential(vocab_size, logits.device).expand(
                num_tokens, vocab_size
            )
        # aiter 融合 kernel，参数依次为：输出、logits [bs, V]、噪声 [bs, V]、温度 [bs]、eps
        #   每行求 softmax(logits / T) / E 的 argmax，写进 sampled_tokens；除温度、softmax、除噪声、argmax 在一次 launch 里完成
        #   （aiter 源码不在本仓库，以上按调用方式和本函数 docstring 推断；T=1e-10 那些行的结果随后被 _apply_greedy_tokens 覆盖）
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens

    def _topk_topp_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
        needs_independent_noise: bool = False,
    ) -> torch.Tensor:
        """Top-K/Top-P sampling with temperature scaling.

        The top-k/top-p paths below use ``torch.multinomial`` (native fallback)
        or aiter's sampling ops that draw from torch's default RNG stream per
        batch row, so fan-out siblings already diverge naturally and the
        ``needs_independent_noise`` flag is accepted purely for API symmetry.
        """
        # 【译】带温度缩放的 Top-K / Top-P 采样。
        #
        #   下面的 top-k / top-p 路径用 ``torch.multinomial``（原生回退）或 aiter 的采样算子，它们按 batch 的每一行
        #   从 torch 默认的随机数流里取随机数，所以扇出的兄弟序列本来就会分叉；
        #   接受 ``needs_independent_noise`` 这个参数纯粹是为了接口对称。
        # Accepted but unused here; see docstring.
        # 【译】在这里接受但不使用；见 docstring。
        del needs_independent_noise
        # Apply temperature scaling
        # Temperatures are pre-clamped to eps in model_runner.prepare_sample()
        # 【译】做温度缩放。温度已在 model_runner.prepare_sample() 里预先钳制到 eps
        # [bs, V] / [bs, 1] 广播：每行除以自己的温度；bf16 张量除以 fp32 张量，按类型提升规则结果是 fp32
        scaled_logits = logits / temperatures.unsqueeze(-1)
        # 每行 softmax → probs [bs, V] fp32，contiguous 保证行连续再交给 aiter / torch 算子
        #   这条路径要落地两份 [bs, V] fp32（scaled_logits 和 probs，bs=256、V=151936 时各约 148 MiB），
        #   这也是不过滤的融合路径更快的原因之一
        probs = scaled_logits.softmax(dim=-1, dtype=torch.float32).contiguous()

        # model_runner.prepare_sample passes None if filtering not needed for that type
        # 【译】某一类过滤不需要时，model_runner.prepare_sample 传 None
        has_topk = top_ks is not None
        has_topp = top_ps is not None

        if AITER_TOPK_TOPP_AVAILABLE:
            return self._aiter_sample(
                probs, top_ks, top_ps, has_topk, has_topp, temperatures
            )
        else:
            return self._native_sample(probs, top_ks, top_ps, temperatures)

    # 把过滤参数转成 aiter 接口要的 (张量, 标量) 一对：二者用其一，张量为 None 时 kernel 用标量
    #   None          → (None, 0)：不过滤（调用方另有 has_topk / has_topp 判断，这个 0 不会被当成 k=0 用）
    #   Python 标量   → (None, x)：整批相同（prepare_sample 的常见情况）
    #   单元素张量    → (None, x.item())：兼容旧调用方；.item() 是一次同步，runner 的主路径不会走到（整批相同时传的是标量）
    #   [bs] 张量     → (x, 0)：逐行参数
    def _to_tensor_scalar(self, x: float | torch.Tensor | None):
        """Adapt filters to AITER's tensor/scalar arguments.

        The runner supplies uniform filters as CPU scalars, avoiding a device
        readback. Singleton tensors remain supported for existing callers.
        """
        # 【译】把过滤参数适配成 AITER 的"张量 / 标量"参数。
        #
        #   runner 把整批相同的过滤参数作为 CPU 标量传入，避免一次设备回读。为兼容已有调用方，仍支持单元素张量。
        if x is None:
            return (None, 0)
        if not isinstance(x, torch.Tensor):
            return (None, x)
        if x.numel() == 1:
            return (None, x.item())
        return (x, 0)

    # aiter 采样算子路径（AITER_TOPK_TOPP_AVAILABLE 为真时）：probs [bs, V] fp32 → next_tokens → 贪心修正 → [bs] int32
    #   这几个算子与 FlashInfer 的采样 API 同名（aiter 源码不在本仓库，内部算法未核对）
    def _aiter_sample(
        self,
        probs: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
        has_topk: bool,
        has_topp: bool,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Use aiter optimized ops for top-k/top-p sampling."""
        # 【译】用 aiter 优化过的算子做 top-k / top-p 采样。
        # Convert to tensor/scalar format for aiter
        # 【译】转成 aiter 要的"张量 / 标量"格式
        k_tensor, k_scalar = self._to_tensor_scalar(top_ks)
        p_tensor, p_scalar = self._to_tensor_scalar(top_ps)

        if has_topk and has_topp:
            # Joint k+p path
            # 【译】k 和 p 同时开
            # 解释：一个算子同时满足两个条件，对应 _native_sample 里两个掩码取交集（数值例子见 sampling_params.py 文件头④）；
            #   参数按 (probs, None, k 张量, k 标量, p 张量, p 标量) 传入；deterministic=True 据参数名是要求结果可复现
            next_tokens = aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                k_tensor,
                k_scalar,
                p_tensor,
                p_scalar,
                deterministic=True,
            )
        elif has_topp:
            # Top-p only
            # 【译】只开 top-p
            next_tokens = aiter_ops.top_p_sampling_from_probs(
                probs, None, p_tensor, p_scalar, deterministic=True
            )
        elif has_topk:
            # Top-k only: renormalize and multinomial
            # 【译】只开 top-k：先重新归一化，再 multinomial
            # 解释：top_k_renorm_probs 把每行前 k 个之外的概率清零并重新归一化（[bs, V]）；
            #   torch.multinomial(·, 1) 按概率每行抽 1 个 → [bs, 1] int64
            renorm_probs = aiter_ops.top_k_renorm_probs(probs, k_tensor, k_scalar)
            next_tokens = torch.multinomial(renorm_probs, num_samples=1)
        else:
            # Neither - just multinomial from probs
            # 【译】都没开——直接按 probs 做 multinomial
            # 观察：forward 只在 _needs_filtering 为真（至少一个不是 None）时才进入这条路径，所以从 forward 走不到这个分支
            next_tokens = torch.multinomial(probs, num_samples=1)

        return _apply_greedy_tokens(probs, temperatures, next_tokens)

    # 回退实现（没有 aiter.ops.sampling 时）：整行排序 + 前缀和 + 掩码 + multinomial，全是普通 PyTorch 算子
    #   代价：每行要对 V 个元素做一次完整排序；数值例子（排序、排它累计和、取交集、重新归一化）见 sampling_params.py 文件头②~④
    def _native_sample(
        self,
        probs: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        EXPERIMENTAL: Native PyTorch fallback for top-k/top-p sampling.

        This implementation has not been thoroughly tested and may produce
        different results compared to the optimized aiter implementation.
        Use aiter.ops.sampling for production workloads.
        """
        # 【译】实验性：top-k / top-p 采样的原生 PyTorch 回退实现。
        #
        #   这个实现没有经过充分测试，结果可能与优化过的 aiter 实现不同。
        #   生产负载请使用 aiter.ops.sampling。
        global _NATIVE_SAMPLING_WARNING_ISSUED
        if not _NATIVE_SAMPLING_WARNING_ISSUED:
            warnings.warn(
                "Using experimental native top-k/top-p sampling. "
                "Install aiter.ops.sampling for optimized performance.",
                UserWarning,
                stacklevel=2,
            )
            _NATIVE_SAMPLING_WARNING_ISSUED = True

        vocab_size = probs.shape[-1]
        device = probs.device

        # Sort probs descending
        # 【译】按概率降序排序
        # sorted_probs、sorted_indices 都是 [bs, V]；sorted_indices[i, j] = 第 i 行概率第 j 大的 token id
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # 每行的包含式累计和 [bs, V]
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Top-p mask: keep tokens until cumsum exceeds top_p
        # The mask keeps tokens where cumsum - current_prob <= top_p
        # (i.e., before we exceed the threshold)
        # 【译】top-p 掩码：保留 token，直到累计和超过 top_p。
        #   掩码保留"累计和 − 当前概率 <= top_p"的 token（也就是越过阈值之前的那些）
        # 解释：cumsum − 当前 = 排它累计和（不含自己），第 0 个恒为 0，所以至少保留一个；
        #   p 是 [bs, 1]（逐行张量，广播到 [bs, V]）或标量
        if top_ps is not None:
            p = top_ps.unsqueeze(-1) if isinstance(top_ps, torch.Tensor) else top_ps
            topp_mask = (cumsum_probs - sorted_probs) <= p
        else:
            topp_mask = torch.ones_like(sorted_probs, dtype=torch.bool)

        # Top-k mask: keep first k tokens
        # 【译】top-k 掩码：保留前 k 个 token
        # indices [1, V] = 排序后的位置 0..V-1；effective_k 是 [bs, 1]（-1 换成 V，即不过滤）或标量
        #   → topk_mask 为 [bs, V]（标量 k 时是 [1, V]，与 topp_mask 相与时广播）
        if top_ks is not None:
            indices = torch.arange(vocab_size, device=device).unsqueeze(0)
            if isinstance(top_ks, torch.Tensor):
                effective_k = torch.where(top_ks == -1, vocab_size, top_ks).unsqueeze(
                    -1
                )
            else:
                effective_k = vocab_size if top_ks == -1 else top_ks
            topk_mask = indices < effective_k
        else:
            topk_mask = torch.ones_like(sorted_probs, dtype=torch.bool)

        # Combined filtering
        # 【译】合并两个过滤条件；（行尾注释）至少保留一个 token
        # 两个条件取交集 [bs, V]；第 0 列是每行概率最大的 token
        mask = topp_mask & topk_mask
        mask[:, 0] = True  # Always keep at least one token

        # 被滤掉的位置清零，再按行重新归一化（分母下限 eps 防止除 0）
        filtered_probs = sorted_probs * mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum(
            dim=-1, keepdim=True
        ).clamp(min=self.eps)

        # Sample and map back to original indices
        # 【译】采样，再映射回原来的 token id
        # sampled_idx [bs] 是排序后的位置；用 sorted_indices gather 回真正的 token id [bs]
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        next_tokens = sorted_indices.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)

        return _apply_greedy_tokens(probs, temperatures, next_tokens)

    # Legacy methods kept for reference
    # 【译】保留作参考的旧方法
    # 解释：grep 核实 atom/ 里没有调用方。greedy_sample = Triton topk 取 argmax；
    #   random_sample = 不带温度的 argmax(p / (E + eps))，与 nano-vllm 的 Sampler 几乎相同（nano-vllm 用 clamp_min_(1e-10)）
    def greedy_sample(
        self, logits: torch.Tensor  # (token_num, vocab_size)
    ) -> torch.Tensor:  # (token_num,)
        _, sampled_tokens = topk(logits, 1)
        return sampled_tokens.view(-1)

    def random_sample(
        self, logits: torch.Tensor  # (token_num, vocab_size)
    ) -> torch.Tensor:  # (token_num,)
        probs = softmax(logits)
        logits = probs.div_(torch.empty_like(probs).exponential_(1) + self.eps)
        _, sampled_tokens = topk(logits, 1)
        return sampled_tokens.view(-1)
