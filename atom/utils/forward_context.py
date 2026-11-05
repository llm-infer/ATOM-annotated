# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】forward_context.py：一次 forward 的"全局上下文"——模型深处的层不靠传参，就能拿到本步的批元数据
#   位置（一步 forward 里谁写、谁读；函数都在 model_runner.py）：
#     ModelRunner.forward(batch)                  每步一次
#       ├ _gate_staging_reuse                     本步的 pinned 元数据缓冲可以改写了（utils/h2d.py 文件头）
#       ├ prepare_model(batch)
#       │   ├ ForwardMode.decide(...)           ← 本文件：本步"走 eager 还是回放图、补到多宽"一次定下
#       │   │                                      （DP>1 时含本步唯一一次跨 DP 集合通信）
#       │   ├ prepare_sample / prepare_cu_seqlens_q / prepare_input_ids     CPU 写 pinned 缓冲 → H2D
#       │   └ prepare_inputs
#       │       ├ attn_metadata_builder.build → AttentionMetaData + positions   ← 本文件定义的结构，后端 builder 填
#       │       ├ Context(...)                  ← 本文件：本步的形状
#       │       └ set_forward_context(...)      ← 本文件：把模块级全局 _forward_context 换成本步的
#       ├ run_model → self.model(input_ids, positions)
#       │     └ 各层 forward 内 get_forward_context()：注意力读 .attn_metadata / .kv_cache_data，
#       │       MoE 读 .context.running_tokens* 和 .dp_metadata，LM head 读 .context.is_prefill 和 cu_seqlens_q ……
#       ├ postprocess（采样；投机解码读 .spec_decode_metadata）
#       └ reset_forward_context()               ← 本文件：清空，防止下一步误用上一步的元数据
#
# 为什么要"全局"上下文
#   model.forward 的主参数固定为 (input_ids, positions)（PP / 多模态另带少数关键字参数）：torch.compile 按这个签名抓图、
#   CUDA Graph 按它录制，几十个模型文件和 vLLM / SGLang 插件也都遵守它。可注意力层还要 slot_mapping、block_tables、cu_seqlens，
#   MoE 要知道 DP 各 rank 的 token 数，LM head 要知道这步是不是 prefill（只取每条序列最后一个 token）……
#   层层传参的话，每个模型的每层 forward 都得多带一串参数。于是在 forward 之前把它们放进一个模块级全局，用的地方自己取。
#   代价是隐式依赖：层的行为取决于"此刻全局里放的是什么"。所以只在 prepare_inputs 末尾 set、forward 结束 reset；
#   CUDA Graph 录制时（ModelRunner.capture_cudagraph）每个尺寸各 set 一个"捕获用"的上下文。
#
# 对比nano-vllm：nano-vllm 的 utils/context.py 只有一个 8 字段的 Context（is_prefill、cu_seqlens_q/k、max_seqlen_q/k、
#   slot_mapping、context_lens、block_tables）加 set_context / get_context / reset_context，用法与这里相同。ATOM 拆成几层：
#     ForwardContext     顶层容器，每步 set_forward_context 新建一个（≈ nano-vllm 的 _CONTEXT 本身）
#       ├ attn_metadata  AttentionMetaData：注意力要的逐 token / 逐序列张量（≈ nano-vllm Context 的全部字段，再加上
#       │                aiter kernel 要的 CSR 索引、persistent 调度表、DSA 稀疏索引、前缀缓存信息）
#       ├ context        Context：本步的"形状"——本 rank 调度了多少（scheduled_*），补齐后跑多少（running_*）
#       │   └ forward_mode  ForwardMode：本步走 eager 还是回放 CUDA Graph（nano-vllm 在 run_model 里只有一行 if）
#       ├ dp_metadata    DPMetadata：DP 各 rank 本步的 token 数（nano-vllm 没有 DP）
#       ├ kv_cache_data  每层的 KV cache 张量（nano-vllm 是直接挂在每个 Attention 模块上：module.k_cache = ...）
#       └ spec_decode_metadata / ubatch_slices / main_stream / in_hipgraph / cudagraph_runtime_mode …
#                        （投机解码、TBO 微批、多 stream、分段 CUDA Graph 的分派）
#
# 本步形状的"两个单位"（全文件最重要的概念；ForwardMode、Context、AttentionMetaData 都按它组织）
#   前置：CUDA Graph 回放时，kernel 序列、张量形状和显存地址都是录制时定死的，所以 batch 要补齐（padding）到某个
#     录过的尺寸，所有输入都要写进事先分配好的固定 buffer（详见 config.py 中 CUDAGraphMode 上方的【科普】②）。
#   running_bs      补齐后的"序列数"：逐序列的数组（cu_seqlens_q、context_lens、block_tables……）按它定宽度，也是图的 key
#   running_tokens  补齐后的"行数"：逐 token 的数组（slot_mapping、hidden_states、MoE 的输入）按它定高度
#   scheduled_bs / scheduled_tokens：本 rank 真正调度的数量；补出来的序列 / 行是"假"的，结果最后切回 scheduled_*
#   例（不开 DP；CLI 默认阶梯 [1..8,16,32,48,64,128,256,512]；MTP k=1 → 每条序列 q=2 行）：
#     调度了 13 条 decode 序列 → scheduled_bs=13、scheduled_tokens=26
#     阶梯上 ≥13 的最小档是 16 → running_bs=16、running_tokens=16×2=32，回放 key=(16, 2) 的图
#     cu_seqlens_q = [0,2,4,…,24,26, 26,26,26]（17 个数，3 条假序列长度为 0）
#     slot_mapping 32 行，第 26~31 行为 -1（不写 KV）；context_lens 16 个，后 3 个为 0
#   为什么两个都要存、不能一个用乘法算另一个：DSpark 的 ragged 步每条序列行数不同，running_tokens 是一个录过的
#     token 档位，不等于 running_bs × q（见 ForwardMode._running_tokens）。
#
# 阅读路线
#   ① AttnState → ② ForwardMode（docstring、decide 的流程图与数字例子、_running_tokens、assert_shape_contract）
#   ③ Context、AttentionMetaData（字段表）→ ④ DPMetadata（为什么要知道每个 DP rank 的 token 数）
#   ⑤ ForwardContext → ⑥ get_forward_context / set_forward_context / reset_forward_context 三个函数
#   【分支】SpecDecodeMetadata（投机解码，第7批）、GDNAttentionMetadata（线性注意力，第8批）、
#     BoundedReplay*（DeepSeek-V4.1 有界重放，第8批）、DP pad-row mask（MegaMoE，第7批）、
#     side_stream 与 TBO 的线程局部上下文（第9批）、cudagraph 运行模式（分段图的分派，第5批 utils/cuda_graph.py）、
#     KV connector（PD 分离 / KV offload，第9批）
#
import logging
import threading
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Union

import numpy as np
import torch

from atom.config import Config, CUDAGraphMode, KVCacheTensor, ParallelConfig


# AttnState：本步注意力走哪条分支，存在 AttentionMetaData.state 里
#   DECODE          每条序列 1+k 行（k = 投机 draft 数；不开投机时 k=0）
#   PREFILL_NATIVE  全新 prefill：每条序列都从位置 0 开始，没有已缓存的前缀要读
#   PREFILL_PREFIX  至少一条序列带前缀 → 注意力还要读 KV cache 里已有的部分。通用 builder（backends.py 的
#                   prepare_prefill）里"带前缀"= cached_lens>0，既包括 chunked prefill 的后续块，也包括前缀缓存命中
#   读取方：DeepSeek-V4 注意力（attentions/deepseek_v4_attn.py、models/deepseek_v4.py，第8批）和插件 bridge；
#     其余后端判断 prefill / decode 看的是 Context.is_prefill，不读它
# Python 语法：Enum 成员是单例；这里 value 用字符串，打印日志时可读
class AttnState(Enum):
    """Attention dispatch state — controls which kv-indices buffers are built
    and which forward branch fires.

    Backends that distinguish only "decode vs prefill" can treat any
    ``PREFILL_*`` value as prefill. Backends with chunked-prefill awareness
    (e.g. V4) further distinguish ``PREFILL_NATIVE`` from ``PREFILL_PREFIX``.

    - ``DECODE``: 1+K tokens/seq uniformly (decode + spec). Per-token decode
      kv-indices buffers are valid; prefill prefix buffers may be stale.
    - ``PREFILL_NATIVE``: fresh prefill — every seq starts at position 0 in
      this fwd. No prior-chunk KV history to read; the prefix region is
      empty per token.
    - ``PREFILL_PREFIX``: chunked prefill — at least one seq has
      ``chunk_start > 0`` and therefore reads its prior chunk's KV from
      the paged history (e.g. V4 SWA ring via ``kv_indices_prefix_swa``).
    """

    # 【译】注意力分派状态——决定构建哪些 kv-indices 缓冲区、走哪个 forward 分支。
    #
    #   只区分"decode 还是 prefill"的后端可以把任何 ``PREFILL_*`` 值都当作 prefill。能感知 chunked prefill 的后端
    #   （例如 V4）进一步区分 ``PREFILL_NATIVE`` 和 ``PREFILL_PREFIX``。
    #
    #   - ``DECODE``：每条序列统一 1+K 个 token（decode + 投机）。逐 token 的 decode kv-indices 缓冲区有效；
    #     prefill 的前缀缓冲区可能是旧的。
    #   - ``PREFILL_NATIVE``：全新 prefill——本次 forward 里每条序列都从位置 0 开始。没有之前块的 KV 历史要读；
    #     每个 token 的前缀区都是空的。
    #   - ``PREFILL_PREFIX``：chunked prefill——至少有一条序列 ``chunk_start > 0``，因此要从分页历史里读它之前块的 KV
    #     （例如 V4 的 SWA 环形缓冲，经由 ``kv_indices_prefix_swa``）。

    DECODE = "decode"
    PREFILL_NATIVE = "prefill_native"
    PREFILL_PREFIX = "prefill_prefix"


# 【分支】chunked MoE 用：把每个 rank 的 token 切成每块至多 max_num_tokens 个，算第 chunk_idx 块里各 rank 有几个
#   例：各 rank [300, 100]、每块 256 → 第 0 块 [256, 100]，第 1 块 [44, 1]（rank1 已做完，仍报 1 个以保持锁步）
#   沿用 vLLM forward_context.py 的同名函数；唯一调用方 DPMetadata.chunked_sizes 在本仓库目前没有调用方
def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: list[int], max_num_tokens: int, chunk_idx: int
) -> list[int]:
    dp_size = len(num_tokens_across_dp_cpu)

    local_size = [-1] * dp_size
    for i in range(dp_size):
        dp_tokens = num_tokens_across_dp_cpu[i]
        local_size[i] = min(max_num_tokens, dp_tokens - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            # 做完的 rank 也至少报 1 个 token，各 rank 才能在每块的集合通信里锁步（行尾英文：即使做完也保证锁步）
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


# 【科普】DPMetadata：为什么每个 rank 都要知道"每个 DP rank 本步有多少 token"
#   前置（DP-attention + MoE；完整图解见 engine_core_mgr.py CoreManager.__init__ 第2步的【科普】DP-attention，
#     锁步与 dummy batch 见 engine_core.py 中 DPEngineCoreProc 上方的说明）：
#     注意力按 DP 切——每个 DP rank 只算自己那批请求，各 rank 的 token 数不同；到了 MoE 层，专家分布在所有 rank 上，
#     每层都要把各 rank 的 token 汇到一起（all_gather / all-to-all），算完再分回去。
#   集合通信要按"每个 rank 发多少行"开缓冲、做切片：
#     变长版（eager）：拼出来的总行数 = Σ 各 rank 行数，第 r 个 rank 的那段 = [cu[r-1], cu[r])
#     定长版（CUDA Graph 回放必须用它，形状要固定）：先把每个 rank 补到全组 max，再做普通 all_gather
#   可是每个 rank 一开始只知道自己的数。num_tokens_across_dp 的办法：每个 rank 造一个只在自己位置非 0 的向量，
#     在 CPU（gloo）组上做一次 SUM all_reduce，于是每个 rank 都拿到完整的表——等于对一个标量做了 all_gather。
#     用 CPU 组：结果要在 CPU 上决定形状，放在 GPU 上还得再同步拷回来。
#   例：DP=4，本步各 rank 调度 [13, 70, 1（dummy）, 40] 个 token：
#     rank1 发 [0, 70, 0, 0]，其余同理，SUM 之后每个 rank 都得到 [13, 70, 1, 40]
#     max_tokens_across_dp = 70（定长 all_gather 的高度）；cu_tokens_across_dp_cpu = [13, 83, 84, 124]（变长切片的边界）
#   主路径上这张表不用再单独通信：ForwardMode.decide 的那次 all_gather（sync_dp_metadata 第 0 行）已经带回来了，
#     prepare_inputs 把它交给 set_forward_context → make(num_tokens_across_dp=表) 直接用。真正调用下面这次
#     all_reduce 的是没有现成表的地方：ModelRunner.get_dp_padding（CUDA Graph 录制循环）、make 既没拿到表
#     也没被告知 unified 时（drafter 的非 unified 情形、TBO 的每个微批）。
#   读取方（MoE 通信，第7批）：moe.py（naive_multicast 用 cu_tokens_across_dp_cpu；eager 变长 gather 用 get_sizes_across_dp）、
#     fused_moe/comm_fused_moe.py（max_tokens_across_dp）、fused_moe/rccl_prepare_finalize.py（get_sizes_across_dp）
#   对比nano-vllm：nano-vllm 只有 TP 没有 DP，没有对应结构；这个类沿用自 vLLM forward_context.py 的同名类
# 字段（都在 CPU 上）：
#   max_tokens_across_dp_cpu  0 维 int32 张量：全组最大 token 数
#   cu_tokens_across_dp_cpu   [dp_size] int64（torch.cumsum 对整数输入默认升到 int64）：各 rank token 数的（含本项）
#                             前缀和，上例为 [13, 83, 84, 124]
#   max_tokens_across_dp      同一个最大值的 Python int（make 里先 .item() 好）。行尾英文：为兼容 cudagraph 预先算好的 int——
#                             使用方直接拿常量，不必在 forward 里再从张量取值（.item() 在 torch.compile 抓图时会造成图中断）
#   local_sizes               只在 chunked_sizes 的 with 块里有效
@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp: int  # Pre-computed int value for cudagraph compatibility
    local_sizes: list[int] | None = None

    @staticmethod
    def num_tokens_across_dp(
        num_tokens: int, dp_size: int, dp_rank: int
    ) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        # 【译】在所有 DP rank 之间收集 num_tokens，结果放在一个大小为 dp_size 的 CPU 张量里返回。
        # 做法见类上方【科普】：one-hot 向量 + SUM all_reduce（get_dp_group().cpu_group 是 gloo 组）
        # 调用方：ModelRunner.get_dp_padding、下面的 make（没有表、又不是 unified 时）
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(
            num_tokens_across_dp, device="cpu", dtype=torch.int32
        )
        import torch.distributed as dist
        from aiter.dist.parallel_state import get_dp_group

        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        # 下一行是被注释掉的旧参数（vLLM 原版 make 的第二个参数）
        # attn_metadata: Any,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None = None,
        *,
        unified: bool = False,
    ) -> "DPMetadata":
        """Per-rank token counts for this pass, and the sizes derived from them.

        Three ways to a table, and the caller picks by what it knows. Hand one
        over and it is taken as given. Say `unified` and every rank runs
        `num_tokens`, so it is that number repeated. Say neither and it is
        discovered, at the cost of one CPU all_reduce.

        `unified` cannot be checked here -- the assert below reads this rank's
        own entry, which a repeated fill satisfies by construction. A wrong
        claim surfaces later, as a fixed-size collective posted at mismatched
        heights.
        """
        # 【译】本次 pass 各 rank 的 token 数，以及由它推出的大小。
        #
        #   有三种方式得到这张表，调用方按自己知道什么来选。直接递一张表过来，就照单全收。声明 `unified`，就表示每个 rank
        #   都跑 `num_tokens`，表就是这个数重复 dp_size 次。两者都没有，就去问，代价是一次 CPU all_reduce。
        #
        #   `unified` 在这里没法检查——下面的 assert 读的是本 rank 自己那一格，重复填充的表天然满足它。声明错了，
        #   问题会在之后暴露：各 rank 以不一致的高度发起定长集合通信。
        # 调用方与选用的方式：set_forward_context（主路径：递表）、spec_decode/drafter.py（unified=…）、
        #   utils/tbo/ubatch_wrapper.py（每个微批不带表 → all_reduce）、本文件 _bounded_replay_dp_stages（递表）
        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # A supplied table already says what every rank runs: its own entry is
        # this pass's height, and saying it again can only disagree.
        # 【译】递进来的表已经说明了每个 rank 跑多少：它自己那一格就是本次 pass 的高度，再声明一遍 unified 只可能矛盾。
        assert num_tokens_across_dp is None or (
            not unified and num_tokens_across_dp[dp_rank] == batchsize
        )
        if num_tokens_across_dp is None:
            num_tokens_across_dp = (
                torch.full((dp_size,), batchsize, dtype=torch.int32, device="cpu")
                if unified
                else DPMetadata.num_tokens_across_dp(batchsize, dp_size, dp_rank)
            )
        # 上例 [13, 70, 1, 40] → max=70，cumsum=[13, 83, 84, 124]
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        max_tokens_across_dp = (
            max_tokens_across_dp_cpu.item()
        )  # Pre-compute int for cudagraph
        return DPMetadata(
            max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu, max_tokens_across_dp
        )

    @contextmanager
    def chunked_sizes(self, max_chunk_size_per_rank: int, chunk_idx: int):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.
        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.
        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.
        It uses cumulative sizes (`cu_tokens_across_dp_cpu`) to derive the
        number of tokens per rank, and calls `_compute_chunked_local_num_tokens`
        to determine the chunk-wise split.
        `self.local_sizes` is only valid inside the context.
        Args:
            max_chunk_size_per_rank: The max number of tokens each rank is
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        # 【译】上下文管理器：在分块（chunked）forward 执行时，计算并临时设置某一块里各 rank 的本地 token 数。
        #   这是为了保证每个 DP rank 与其他 rank 锁步地处理分给它的那部分 token，即使各 rank 的 token 数不均、
        #   或者有些 rank 已经提前处理完。
        #   分块执行时，把每个 rank 上的全部 token 切成若干块（每块至多 `max_chunk_size_per_rank` 个）；对给定的 `chunk_idx`，
        #   这个上下文管理器把 `self.local_sizes` 设成该块里每个 rank 要处理的 token 数。
        #   它用累计大小（`cu_tokens_across_dp_cpu`）推出每个 rank 的 token 数，再调用 `_compute_chunked_local_num_tokens`
        #   决定按块的划分。
        #   `self.local_sizes` 只在这个上下文内有效。
        #   参数：
        #     max_chunk_size_per_rank：本块里每个 rank 最多允许处理的 token 数。
        #     chunk_idx：要计算的块的下标。
        # 【分支】本仓库目前没有调用方（沿用 vLLM）；Python 语法：@contextmanager 把生成器函数变成 with 语句用的上下文，
        #   yield 之前是进入、finally 里是退出时一定执行的清理
        cu_sizes = self.cu_tokens_across_dp_cpu
        num_tokens_across_dp_cpu = [
            (cu_sizes[i] - cu_sizes[i - 1]).item() if i > 0 else cu_sizes[0].item()
            for i in range(len(cu_sizes))
        ]
        self.local_sizes = _compute_chunked_local_num_tokens(
            num_tokens_across_dp_cpu, max_chunk_size_per_rank, chunk_idx
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        return self.local_sizes

    def get_sizes_across_dp(self) -> list[int]:
        """Per-rank token counts derived from cumulative tensor."""
        # 【译】由累计张量推出的各 rank token 数。
        # 上例 cu=[13, 83, 84, 124] → [13, 70, 1, 40]；MoE 的 eager 变长 gather / reduce_scatterv 用它
        cu = self.cu_tokens_across_dp_cpu
        return [(cu[i] - (cu[i - 1] if i > 0 else 0)).item() for i in range(len(cu))]


# 【分支】投机解码的验证元数据（第7批 spec_decode/drafter.py calc_spec_decode_metadata 构建，postprocess 与拒绝采样读）
#   例：k=2 个 draft、2 条序列，本步 input_ids = [a0,d0,d1 | b0,e0,e1]（a0/b0 是 anchor，d/e 是 draft）
#     target_logits_indices = [0, 1, 3, 4]   这些行的 logits 用来验证"下一行"的 draft（行 0 验 d0，行 1 验 d1 …）
#     bonus_logits_indices  = [2, 5]         每条序列最后一行：全部接受时再多采 1 个 bonus token
#     cu_num_draft_tokens   = [2, 4]         draft 数的前缀和；num_draft_tokens_np = [2, 2]（CPU 上的 numpy）
#   num_spec_steps = mtp_k
@dataclass
class SpecDecodeMetadata:
    # The step's input ids. Draft r is input_ids[target_logits_indices[r] + 1]:
    # logits row t verifies the token that follows it. Read in place by the
    # rejection sampler, on the forward stream the next step publishes its
    # input ids on, so they cannot be overwritten under it.
    # 【译】本步的 input ids。第 r 个 draft 是 input_ids[target_logits_indices[r] + 1]：logits 第 t 行验证的是紧跟在它
    #   后面的那个 token。拒绝采样器原地读取它；读取发生在 forward stream 上，而下一步也在这条 stream 上发布它的
    #   input ids，所以不会在读的过程中被覆盖。
    input_ids: torch.Tensor
    num_spec_steps: int
    num_draft_tokens_np: np.ndarray
    cu_num_draft_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor


# 只警告一次：decide 发现 (running_bs, q) 没录过图、只能 eager 时（只应在保留了 drafter 的 P/D producer 上出现）
_uncaptured_decode_shape_warned = False


def _warn_uncaptured_decode_shape(running_bs: int, max_seqlen_q: int) -> None:
    global _uncaptured_decode_shape_warned
    if _uncaptured_decode_shape_warned:
        return
    _uncaptured_decode_shape_warned = True
    _logger.warning(
        "No CUDA graph recorded at (bs=%d, q=%d): decode steps at an uncaptured "
        "shape run eagerly. Expected only on a P/D producer that keeps a drafter.",
        running_bs,
        max_seqlen_q,
    )


# ForwardMode：本步形状与分派的唯一决定（每步一次，ModelRunner.prepare_model 调 decide 生成；存进 Context.forward_mode）
#   读取方：prepare_inputs（running_bs / running_tokens / max_seqlen_q / sync）、run_model（use_cudagraph、
#     piecewise_captured、assert_shape_contract）、TBO 包装（第9批）
# Python 语法：@dataclass(frozen=True) 生成只读实例，给字段赋值会抛 FrozenInstanceError——定下来就不许改
@dataclass(frozen=True)
class ForwardMode:
    """One step's shape, settled once, in the only two units ATOM has.

    ``running_bs`` sizes everything per-sequence (attention above all);
    ``running_tokens`` everything per-row (MoE above all). ``scheduled_*`` is
    this rank's own, and what a result slices back to.

    They are agreed across DP to different extents, and the difference is the
    point. ``running_bs`` is agreed on EVERY step -- it is the reduction on a
    ladder every rank shares -- because a captured graph holds its collective
    at that batch and a rank arriving with another one hangs the group.
    ``running_tokens`` is agreed exactly when ``running_tokens_are_unified``:
    off it, MoE takes the variable-length gather, whose contract is that the
    rows it was handed are the rows THIS rank scheduled, so a group height
    there would claim rows nobody allocated. An over-wide per-sequence array
    costs a few sentinel rows; an over-wide height is a contract break.

    Nothing downstream recomputes a width. ``running_bs * q`` is NOT
    ``running_tokens`` on a ragged step, and every site that rederived one from
    the other is a site where the two came to disagree.

    ``decide`` owns the whole chain -- the collective, adopting its reductions
    onto the batch, the ladder, both widths -- because the steps are not
    separable: the reduced query length rewrites the batch, and everything else
    is read off it afterwards.
    """

    # 【译】一步的形状，只定一次，用 ATOM 仅有的两个单位表达。
    #
    #   ``running_bs`` 决定所有逐序列东西的宽度（首先是注意力）；``running_tokens`` 决定所有逐行东西的高度（首先是 MoE）。
    #   ``scheduled_*`` 是本 rank 自己的数量，也是结果最后切回去的大小。
    #
    #   两者在 DP 间达成一致的程度不同，这个差别正是要点。``running_bs`` 每一步都一致——它是在所有 rank 共享的阶梯上
    #   取的归约结果——因为录下来的图把其中的集合通信固定在那个 batch 上，带着别的 batch 来的 rank 会让整个组卡死。
    #   ``running_tokens`` 恰好在 ``running_tokens_are_unified`` 为真时一致：为假时 MoE 走变长 gather，它的约定是
    #   "交给它的行就是本 rank 调度的行"，此时若用全组的高度，就等于声称有一些谁也没分配的行。逐序列数组宽了，
    #   只是多几个哨兵值；高度宽了，就是违约。
    #
    #   下游不重新计算任何宽度。在 ragged 步上 ``running_bs * q`` 并不是 ``running_tokens``；凡是从一个推另一个的地方，
    #   最后都出现过两者不一致。
    #
    #   ``decide`` 负责整条链——集合通信、把归约结果采纳到本步上、阶梯、两个宽度——因为这些步骤分不开：
    #   归约后的 query 长度会改写 batch，其余一切都在那之后从它读出。

    # 字段（例子沿用文件头：13 条 decode、MTP k=1、阶梯取整到 16）
    #   use_cudagraph               True = 走 run_model 的"图"分支：手动 FULL 整图回放；PIECEWISE 模式下进入分段图分派
    #                               （是否真的回放分段图再看 piecewise_captured）
    #   is_prefill                  本步有没有 prefill token（一步只做 prefill 或只做 decode）
    #   scheduled_bs / _tokens      本 rank 调度的序列数 / token 数：13 / 26
    #   running_bs / running_tokens 补齐后的序列数 / 行数：16 / 32（DP 下是全组一致的值，见 decide）
    #   running_tokens_are_unified  全组都在 decode → running_tokens 是全组的高度，MoE 可以走定长 all_gather（才能进图）
    use_cudagraph: bool
    is_prefill: bool
    scheduled_bs: int
    scheduled_tokens: int
    running_bs: int
    running_tokens: int
    running_tokens_are_unified: bool
    # Query rows per sequence this step. `num_spec_step + 1` unless a DSpark
    # shrink cut it, then DP-MAX'd -- so it is the group's, like the widths.
    # 【译】本步每条序列的 query 行数。等于 `num_spec_step + 1`，除非 DSpark 把它缩短了；再在 DP 间取 MAX——
    #   所以它和两个宽度一样，是全组的值。
    max_seqlen_q: int
    # A PIECEWISE recording exists at `running_tokens`, so this step may replay.
    # 【译】在 `running_tokens` 这个高度上存在一份 PIECEWISE 录制，所以这一步可以回放。
    piecewise_captured: bool
    # TBO's cross-DP AND, or the local answer on one rank -- there being no peer
    # to veto it. Settled here so the gate has one source at every dp_size.
    # 【译】TBO 的跨 DP"与"结果；单个 rank 时就是本地的判断——没有同伴来否决它。在这里定下，任何 dp_size 下
    #   这个开关都只有一个来源。（TBO 见第9批）
    tbo_collective_active: bool
    # What else the one per-step collective returned: per-rank token counts, the
    # DSpark shape max. Not shape. None on a single rank.
    # 【译】每步那一次集合通信返回的其他东西：各 rank 的 token 数、DSpark 的形状最大值。不是形状本身。单 rank 时为 None。
    #   （类型是 utils/tbo/ubatching.py 的 DPSyncResult；prepare_inputs 从这里取 num_tokens_across_dp 交给 DPMetadata）
    sync: Any | None = None
    # This rank's own `BoundedReplayDPReport` for the step, if its model sends one.
    # 【译】本 rank 这一步自己的 `BoundedReplayDPReport`（如果它的模型会发的话）。（DeepSeek-V4.1，第8批）
    dp_stage_report: Any | None = None

    @classmethod
    def decide(
        cls,
        *,
        batch,
        dp_size: int,
        dp_group,
        enforce_eager: bool,
        step_needs_eager: bool,
        capture_sizes: list[int],
        captured_tokens: list[int] | None,
        is_block_drafter: bool,
        tbo_on: bool,
        local_tbo: tuple[bool, bool, int, int],
        max_seqlen_q: int,
        graph_shapes: Collection[tuple[int, int]] | None = None,
        dp_stage_report: "BoundedReplayDPReport | None" = None,
    ) -> "ForwardMode":
        """Run the step's DP collective and settle its shape from the result.

        Any new force-eager condition belongs here, not in a caller-side check.
        `step_needs_eager` is this rank's attention backend refusing a graph for
        the step (`AttentionMetadataBuilder.step_needs_eager`); the DP group
        agrees on it.
        """
        # 【译】执行本步的 DP 集合通信，并据其结果定下本步的形状。
        #
        #   任何新的"强制 eager"条件都属于这里，而不是在调用方另做检查。`step_needs_eager` 表示本 rank 的注意力后端
        #   拒绝这一步使用 graph（`AttentionMetadataBuilder.step_needs_eager`）；DP 组会对它达成一致。
        #
        # 参数（调用方只有 ModelRunner.prepare_model）：
        #   batch               调度器给本 rank 的 ScheduledBatch
        #   dp_size / dp_group  DP 组大小与它的 CPU（gloo）进程组；dp_size=1 时 dp_group=None
        #   enforce_eager       --enforce-eager：根本没录图
        #   step_needs_eager    注意力后端说"这一步不能用图"（目前只有 DeepSeek-V4.1 覆盖它：本步的行放不进 score workspace 时）
        #   capture_sizes       ModelRunner.capture_sizes_np：录过的 batch 阶梯，升序 int32；没录图时为 [0]
        #   captured_tokens     PIECEWISE 模式下录过的 token 档位（升序）；非 PIECEWISE 为 None
        #   is_block_drafter    DSpark 这类"一次出一整块 draft"的草稿器（第7批）：q 要跨 DP 取 max，token 布局是扁平的
        #   tbo_on / local_tbo  TBO 开关；本 rank 的 (够不够最少 token 数, 能否拆, 微批0 token 数, 微批1 token 数)（第9批）
        #   max_seqlen_q        本步每条序列的 query 行数 = num_spec_step+1；DSpark 缩 q 之后为缩后的值
        #   graph_shapes        只有 P/D producer 传：真正录过的 (bs, q) 集合（它保留了 drafter，录的是 q=k+1，跑的却都是 q=1）
        #   dp_stage_report     DeepSeek-V4.1 有界重放的各阶段 token 数，搭这次集合通信一起交换（第8批）
        #
        # 决策流程（编号与下面代码分段一致）：
        # ┌──────────────────────────────────────────────────────────────────────────────────────────┐
        # │ ① 本 rank 的事实：is_prefill（本步有没有 prefill token）、scheduled_bs、scheduled_tokens │
        # └──────────────────────────────────────────────────────────────────────────────────────────┘
        #                                               │
        #                   ┌───────────────────────────┴────────────────────┐
        #                   ▼ 单 rank（dp_size == 1）                        ▼ DP>1
        # ┌───────────────────────────────────┐  ┌───────────────────────────────────────────────────────┐
        # │ unified_bs = scheduled_bs         │  │ ② sync_dp_metadata：一次 CPU all_gather，在全组上取： │
        # │ unified    = True（自己就是全组） │  │    unified_bs       = MAX(scheduled_bs)               │
        # │ TBO = tbo_on 且本地够 token、能拆 │  │    unified          = 没有 rank 在 prefill            │
        # └───────────────────────────────────┘  │    step_needs_eager = OR(needs_eager)                 │
        #                   │                    │    max_seqlen_q     = MAX(q)（仅 block drafter）      │
        #                   │                    │    num_tokens_across_dp = 各 rank 的 token 数         │
        #                   │                    └───────────────────────────────────────────────────────┘
        #                   └───────────────────────────┬────────────────────┘
        #                                               ▼
        # ┌──────────────────────────────────────────────────────────────────────────────────────────┐
        # │ ③ 上阶梯（每步都做，prefill 也做）：rung = searchsorted(capture_sizes, unified_bs)       │
        # │    on_ladder  = rung < len(capture_sizes)                                                │
        # │    running_bs = capture_sizes[rung]（≥ unified_bs 的最小档），越出阶梯时 = unified_bs    │
        # └──────────────────────────────────────────────────────────────────────────────────────────┘
        #                                               ▼
        # ┌──────────────────────────────────────────────────────────────────────────────────────────┐
        # │ ④ use_cudagraph = not is_prefill              prefill 的行数不规则，从不回放             │
        # │               and unified                     有 rank 在 prefill → 全组 eager            │
        # │               and not (enforce_eager or step_needs_eager)     后者已在全组 OR 过         │
        # │               and on_ladder                   比最大档还大 → eager                       │
        # │               and (running_bs, q) 录过        只有 P/D producer 会查不到                 │
        # └──────────────────────────────────────────────────────────────────────────────────────────┘
        #                                               ▼
        # ┌──────────────────────────────────────────────────────────────────────────────────────────┐
        # │ ⑤ _running_tokens：not unified 或 prefill → scheduled_tokens（不补行）                   │
        # │    否则 running_bs × q；PIECEWISE 下再看这个高度录没录过 → piecewise_captured            │
        # │    （block drafter：取 ≥ 全组最长扁平行数、且能被 q 整除的最小已录档位）                 │
        # └──────────────────────────────────────────────────────────────────────────────────────────┘
        #
        # 带数字走几遍（阶梯 = CLI 默认 [1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]；不开投机 q=1，除非另说）：
        #   A 单卡 decode 13 条：③ rung=8 → running_bs=16；④ 全部满足 → use_cudagraph=True；⑤ running_tokens=16
        #     → 回放 key=(16, 1) 的图，3 条假序列长度为 0、slot 为 -1
        #   B 单卡 decode 600 条（把 max_num_seqs 调到 ≥600、阶梯仍止于 512 时）：③ rung=15=len → on_ladder=False，
        #     running_bs=600 → ④ eager（对应 §已核实："decode batch 超过最大档位时以 eager 运行"）
        #   C 单卡 prefill 10 条共 5000 token：③ 照样取整，running_bs=16（逐序列数组为随后的 draft pass 补齐）；
        #     ④ is_prefill → eager；⑤ running_tokens=5000（prefill 不补行）
        #   D DP=2，rank0 decode 13 条、rank1 decode 70 条：② unified_bs=70、unified=True → 两个 rank 都 running_bs=128、
        #     都回放 (128, 1)；rank0 补 115 条假序列。若各自取整（16 和 128），两张图里 MoE 定长 all_gather 的高度不同，
        #     组内集合通信对不上 → 整组卡死。这就是 running_bs 必须每步 DP 一致的原因
        #   E DP=2，rank0 decode 13 条、rank1 在 prefill：② unified=False → 两个 rank 都 eager；⑤ 各自的
        #     running_tokens = 自己的 scheduled_tokens，MoE 走变长 gather（DPMetadata 的表告诉它每个 rank 各几行）
        #   F --enforce-eager：没录图，capture_sizes=[0] → ③ searchsorted([0], 13)=1 → 永远越出阶梯，running_bs=unified_bs；
        #     ④ 永远 eager（torch.compile 仍然生效，见 config.py CUDAGraphMode 下的适用场景表）
        #   G MTP k=3（q=4）、单卡 decode 13 条：running_bs=16、running_tokens=64，回放 key=(16, 4)
        # 对比nano-vllm：nano-vllm 在 run_model 里只有一句
        #   `if is_prefill or enforce_eager or bs > 512: eager  else: graph_bs = 第一个 ≥bs 的档位`，
        #   ATOM 多出的是 DP 一致性（②）、注意力后端否决（step_needs_eager）、PIECEWISE 的 token 档位和 P/D producer 的形状检查
        #
        # Lazy: `atom.utils.tbo`'s package init reaches back into this module.
        # 【译】延迟导入：`atom.utils.tbo` 包的初始化会反过来 import 本模块（循环导入）。
        from atom.utils.tbo.ubatching import sync_dp_metadata

        # ① 本 rank 的事实。一步只做 prefill 或只做 decode（调度器排了 prefill 就直接返回），所以有 prefill token 就是 prefill 步
        is_prefill = batch.total_tokens_num_prefill > 0
        scheduled_tokens = batch.total_tokens_num
        # The whole batch, never a per-kind count: `schedule` returns its
        # prefill batch before it can add a decode row, so the two are equal
        # today, and this is the one that survives a mixed batch.
        # 【译】取整个 batch 的序列数，而不是某一类的数量：`schedule` 在能加入 decode 行之前就返回了 prefill batch，
        #   所以目前两者相等；将来出现混合 batch 时，这个才仍然正确。
        scheduled_bs = batch.total_seqs_num
        # 本 rank 的 TBO 预判（ModelRunner._local_tbo_eligibility）：是否达到最少 token 数、能否拆成两个微批、两个微批各几个 token
        meets_min, can_split, ub0, ub1 = local_tbo

        # ② DP>1：所有要跨 DP 对齐的标量打包成一次 CPU all_gather（sync_dp_metadata，utils/tbo/ubatching.py；
        #   行布局见那里的 docstring：第 0 行 scheduled_tokens、第 1 行 scheduled_bs、第 2 行 is_prefill、第 3 行 needs_eager……）。
        #   一步只发这一次：以前是最多 3 次 all_reduce（get_dp_padding / 这里 / UBatchWrapper 里各一次）
        sync = None
        if dp_size > 1:
            sync = sync_dp_metadata(
                dp_group=dp_group,
                dp_size=dp_size,
                scheduled_tokens=scheduled_tokens,
                scheduled_bs=scheduled_bs,
                is_prefill=is_prefill,
                tbo_on=tbo_on,
                needs_eager=step_needs_eager,
                local_meets_min_tokens=meets_min,
                local_can_split=can_split,
                local_ub_tokens=(ub0, ub1),
                # Only a block drafter needs the group to agree on it; every
                # other flavor keeps its own and the two extra wire rows are
                # not sent.
                # 【译】只有 block drafter 需要全组对它取一致；其他草稿方式各用各的 q，那几行额外数据也就不发送。
                max_seqlen_q=max_seqlen_q if is_block_drafter else None,
                # A model's per-stage token counts (`BoundedReplayDPReport`), so the
                # step still issues one collective.
                # 【译】模型的各阶段 token 数（`BoundedReplayDPReport`）也搭进来，这样本步仍然只发一次集合通信。
                dp_stage_row=(
                    None if dp_stage_report is None else dp_stage_report.wire()
                ),
            )
            # The group's query length, taken BEFORE the rows are read off it:
            # a rank still on its local q settles on a different `running_tokens`
            # than its peers.
            # 【译】全组的 query 长度，必须在用它算行数之前取：还用本地 q 的 rank 会算出和同伴不同的 `running_tokens`。
            if sync.max_seqlen_q_across_dp is not None:
                max_seqlen_q = sync.max_seqlen_q_across_dp
            # 全组最大的序列数：下面的阶梯取整从它出发，于是每个 rank 得到同一个 running_bs（上面例 D）
            unified_bs = sync.max_bs_across_dp
            tbo_collective_active = sync.tbo_collective_active
            # One meaning and only one: every rank is decoding, so the group max
            # pads them level and `scheduled_tokens` reduces to a decode count on
            # every rank. A prefilling peer makes the step ragged across the
            # group, and then each rank runs its own count through the
            # variable-length gather.
            # 【译】只有一个含义：每个 rank 都在 decode，所以全组的 max 能把它们补到一样高，而且每个 rank 的
            #   `scheduled_tokens` 都是 decode token 数。只要有同伴在 prefill，这一步在组内就是参差的（ragged），
            #   各 rank 只能按自己的数走变长 gather。（例 E）
            unified = not sync.any_rank_has_prefill
            # A graph replays its collectives: the group replays or runs
            # eagerly as one.
            # 【译】图回放时也会回放其中录下的集合通信：全组要么一起回放，要么一起 eager。
            #   所以任何一个 rank 的后端说"不能用图"，全组都改走 eager
            step_needs_eager = sync.any_rank_needs_eager
        else:
            # One rank pads to nobody, so it has one height by construction, and
            # its TBO answer stands unvetoed.
            # 【译】单个 rank 不需要向谁看齐，天然只有一个高度；它的 TBO 判断也没人否决。
            unified_bs, unified = scheduled_bs, True
            tbo_collective_active = tbo_on and meets_min and can_split

        # ---- the batch: one rule, every step ---------------------------
        # `unified_bs` is the reduction's own answer and the ladder is a rule
        # every rank shares, so this is DP-agreed by construction -- not by a
        # case analysis a future branch could get wrong. That is the whole
        # property: a graph key, a draft pass and an attention plan all read
        # this one field and cannot end up holding two widths.
        #
        # One lookup answers both of the step's questions. `capture_sizes` is
        # ascending int32 (`ModelRunner.capture_sizes_np`, sorted where it is
        # built and never mutated), so the insertion point IS the smallest size
        # that holds the batch, and running off the end IS "no recording can".
        # 【译】---- batch：一条规则，每一步都用 ----
        #   `unified_bs` 是归约本身的结果，阶梯是所有 rank 共享的规则，所以这里天然 DP 一致——不是靠分情况讨论
        #   （将来加分支时很容易写错）。这就是全部要点：图的 key、draft pass 和注意力的计划都读这一个字段，
        #   不可能各拿一个宽度。
        #
        #   一次查找回答本步的两个问题。`capture_sizes` 是升序 int32（`ModelRunner.capture_sizes_np`，在构建处排好序、
        #   之后不再修改），所以插入点就是能装下这个 batch 的最小档位，越过末尾就意味着"没有任何录制装得下"。
        # ③ 上阶梯。np.searchsorted(a, v) 默认 side='left'：返回第一个满足 a[i] >= v 的下标 i（全都小于 v 时返回 len(a)）
        #   例：a=[1,…,8,16,32,…,512]，v=13 → i=8（a[8]=16）；v=16 → 也是 8（正好有这一档，不用补）；v=600 → 15=len
        rung = int(np.searchsorted(capture_sizes, unified_bs))
        on_ladder = rung < len(capture_sizes)
        running_bs = int(capture_sizes[rung]) if on_ladder else unified_bs

        # ---- dispatch: a different question, asked separately ----------
        # Whether the TARGET replays. It no longer decides how wide anything
        # is; when the batch answered four ways depending on this, `running_bs`
        # was per-rank on two of them and every consumer had to know which.
        #
        # The ladder answers the batch question alone, but a recording is keyed
        # by `(running_bs, max_seqlen_q)`. Only a P/D producer passes
        # `graph_shapes`: it keeps its drafter, so capture holds the q=mtp_k+1
        # shapes, while the scheduler never speculates for it and every decode
        # it runs is q=1. That holds on every rank, its DP dummy included, and
        # `running_bs` is the group's, so the lookup is DP-agreed: the group
        # replays or runs eagerly as one.
        # 【译】---- 分派：另一个问题，单独问 ----
        #   问的是目标模型（TARGET）回不回放。它不再决定任何东西有多宽；以前 batch 宽度随这个答案有四种算法，
        #   其中两种下 `running_bs` 是每个 rank 各自的，每个使用方都得知道是哪一种。
        #
        #   阶梯只回答 batch 的问题，而一份录制的 key 是 `(running_bs, max_seqlen_q)`。只有 P/D producer 会传
        #   `graph_shapes`：它保留了 drafter，所以录的是 q=mtp_k+1 的形状，可调度器从不为它做投机，它跑的每个 decode 都是
        #   q=1。这在每个 rank 上都成立（包括它的 DP dummy），而 `running_bs` 是全组的，所以这次查找是 DP 一致的：
        #   全组一起回放或一起 eager。
        # ④ 能否回放：四个条件缺一不可（见上面流程图 ④）。enforce_eager 只关图，不关 torch.compile
        replayable = (
            not is_prefill
            and unified
            and not (enforce_eager or step_needs_eager)
            and on_ladder
        )
        shape_captured = (
            graph_shapes is None or (running_bs, max_seqlen_q) in graph_shapes
        )
        if replayable and not shape_captured:
            _warn_uncaptured_decode_shape(running_bs, max_seqlen_q)
        # 录过就回放；没录过的形状（只可能是 P/D producer）打一次 warning 后 eager
        use_cudagraph = replayable and shape_captured

        # ⑤ 逐行张量的高度；DP 下 num_tokens_across_dp 只给 block drafter 用（取全组最长的扁平行数）
        running_tokens, piecewise_captured = cls._running_tokens(
            is_prefill=is_prefill,
            unified=unified,
            running_bs=running_bs,
            scheduled_tokens=scheduled_tokens,
            num_tokens_across_dp=None if sync is None else sync.num_tokens_across_dp,
            captured_tokens=captured_tokens,
            is_block_drafter=is_block_drafter,
            max_seqlen_q=max_seqlen_q,
        )
        return cls(
            use_cudagraph=use_cudagraph,
            is_prefill=is_prefill,
            scheduled_bs=scheduled_bs,
            scheduled_tokens=scheduled_tokens,
            running_bs=running_bs,
            running_tokens=running_tokens,
            running_tokens_are_unified=unified,
            max_seqlen_q=max_seqlen_q,
            piecewise_captured=piecewise_captured,
            tbo_collective_active=tbo_collective_active,
            sync=sync,
            dp_stage_report=dp_stage_report,
        )

    @staticmethod
    def _running_tokens(
        *,
        is_prefill: bool,
        unified: bool,
        running_bs: int,
        scheduled_tokens: int,
        num_tokens_across_dp,
        captured_tokens,
        is_block_drafter: bool,
        max_seqlen_q: int,
    ) -> tuple[int, bool]:
        """``(running_tokens, piecewise_captured)``.

        `pad_for_all_gather` asserts the first equals the rows the tensor really
        has, so it is not "the biggest count anyone reported": it is what THIS
        forward will produce.
        """
        # 【译】返回 ``(running_tokens, piecewise_captured)``。
        #
        #   `pad_for_all_gather` 会断言前者等于张量实际的行数，所以它不是"任何人报告的最大数"，
        #   而是本次 forward 实际会产生的行数。
        # 前置（PIECEWISE 的 key）：分段编译模式下，注意力之间的稠密片段按"总 token 数"一维地录图（见 config.py
        #   CUDAGraphMode 上方【科普】④），所以这里要回答的是"running_tokens 这个高度有没有录过"，而不是 (bs, q)
        if not unified:
            # Nothing padded the tensor -- the varlen gather sends each rank its
            # own count -- so a height from the group would claim rows nobody
            # allocated.
            # 【译】没有任何东西补齐过这个张量——变长 gather 给每个 rank 发的是它自己的数——所以用全组的高度，
            #   就会声称有一些谁也没分配的行。
            return scheduled_tokens, False
        if is_prefill:
            # Reachable on one rank only: a group of one is unified whatever it
            # runs, while under DP any prefill clears `unified` above. Prompts
            # are ragged, so no product recovers a height and this rank's own
            # count is the whole answer.
            # 【译】只有单 rank 才会走到这里：一个人的"组"不管跑什么都是 unified，而 DP 下任何 prefill 都会在上面把
            #   `unified` 清掉。prompt 长短不一，任何乘积都还原不出高度，本 rank 自己的数就是全部答案。
            return scheduled_tokens, False

        # The batch was rounded onto the ladder AFTER the sync, so this runs rows
        # no rank reported; group-agreed all the same, both factors having come
        # out of the reduction.
        # 【译】batch 是在 sync 之后才取整到阶梯上的，所以这里会跑一些没有任何 rank 报告过的行；但它照样是全组一致的，
        #   因为两个因子都来自归约结果。例：DP 组 max bs=70、q=2 → running_bs=128、running_tokens=256
        q = max_seqlen_q
        running_tokens = running_bs * q
        if not captured_tokens:
            return running_tokens, False
            # 上面：非 PIECEWISE（captured_tokens 为 None 或空）——高度就是矩形 bs×q，没有分段图可回放
        if not is_block_drafter:
            # Rectangular layout: one row per (seq, query), which every per-token
            # buffer spans and `slot_mapping[i * q + j]` addresses through. Anything
            # under the product would leave the tail of that rectangle unwritten.
            # 【译】矩形布局：每个 (序列, query) 一行，每个逐 token 缓冲区都覆盖这个矩形，并通过 `slot_mapping[i * q + j]`
            #   寻址。比乘积小的任何高度都会让矩形尾部没人写。
            return running_tokens, running_tokens in captured_tokens

        # Marker-driven layout: the rows are one flat run, so the width is the
        # smallest recorded one that holds it, not a product -- q-divisible
        # because the per-seq structures tile the run at that stride. Every rank
        # is decoding here, so `scheduled_tokens` IS a decode count on each of
        # them and row 0's reduction is the group's run. `None` is one rank.
        # 【译】标记驱动的布局：这些行是一段扁平的连续行，所以宽度是装得下它的最小已录宽度，而不是乘积——
        #   要能被 q 整除，因为逐序列结构以 q 为步长铺满这段行。这里每个 rank 都在 decode，所以 `scheduled_tokens`
        #   在每个 rank 上都是 decode token 数，第 0 行的归约就是全组的行数。`None` 表示单 rank。
        # 【分支】DSpark ragged 步（第7批）。例：q=4、全组最长扁平行数 run=37、已录档位 [16, 32, 40, 48, 64]
        #   → 取 40（≥37 且 40%4==0），piecewise_captured=True
        run = (
            scheduled_tokens
            if num_tokens_across_dp is None
            else int(num_tokens_across_dp.max())
        )
        recorded = next((n for n in captured_tokens if n >= run and n % q == 0), None)
        if recorded is None:
            # Nothing recorded holds the run, so forward it eagerly rather than
            # claim a graph nobody made at this width.
            # 【译】没有任何录制装得下这段行，所以 eager 跑，而不是声称存在一张谁也没录过的这个宽度的图。
            return max(run, running_tokens), False
        return recorded, True

    def assert_shape_contract(
        self,
        input_ids: "torch.Tensor",
        attn_metadata: "AttentionMetaData",
    ) -> None:
        """Tie what the builder published to what this step settled.

        Covers the PADDED step, which is the only one that can be wrong: a
        width is either both units' or neither's, and every way they have come
        apart -- a builder on a stale batch, a per-seq array widened while the
        per-token ones were not -- shows up as one of the equalities below. The
        version that skipped whenever ``running_bs != scheduled_bs`` exempted
        exactly that case, so a mismatch reached the model and surfaced as lost
        accuracy rather than as a failure here.

        Shapes only, never values: reading a device tensor's contents on this
        path is a D2H sync per step.

        Prefill is still out. Its rows are the prompt's, so no product of a
        batch and a query length describes them, and its per-seq tail is padded
        for a following draft pass rather than for anything this step runs.
        """
        # 【译】把 builder 发布的东西和本步定下的形状对上。
        #
        #   检查的是补齐后的步，也只有它可能出错：一个宽度要么两个单位都用、要么都不用；两者分道扬镳的每一种方式——
        #   builder 用了过期的 batch、逐序列数组加宽了而逐 token 数组没有——都会表现为下面某个等式不成立。
        #   以前的版本在 ``running_bs != scheduled_bs`` 时跳过检查，恰好豁免了这种情况，于是不匹配一路传进模型，
        #   表现为精度下降，而不是在这里失败。
        #
        #   只查形状，不查值：在这条路径上读设备张量的内容，每步都会多一次 D2H 同步。
        #
        #   prefill 仍不检查。它的行是 prompt 的，任何 batch × query 长度的乘积都描述不了；它的逐序列尾部是为随后的
        #   draft pass 补齐的，而不是为本步要跑的任何东西。
        # 调用方：run_model 开头，每步一次。以文件头的例子（13 条、k=1）为准：input_ids 26 行 ≤ running_tokens 32，
        #   slot_mapping 32 行，cu_seqlens_q 17 = running_bs+1
        if self.is_prefill or input_ids is None or attn_metadata is None:
            return

        def _rows(name):
            t = getattr(attn_metadata, name, None)
            return None if t is None else int(t.shape[0])

        # `input_ids` is the scheduled prefix at the runner boundary. Uniform
        # decode exposes `running_tokens` to the model in both eager and graph
        # execution; a mixed prefill/decode step keeps its local token count.
        # 【译】在 runner 边界上，`input_ids` 是调度的那部分前缀。统一的 decode 不管 eager 还是回放，交给模型的都是
        #   `running_tokens` 行（run_model 里由 _padded_decode_inputs 或图的固定 buffer 补齐）；
        #   prefill/decode 混合的步保留本地 token 数。
        assert input_ids.shape[0] == self.scheduled_tokens, (
            f"input_ids length {input_ids.shape[0]} != scheduled_tokens="
            f"{self.scheduled_tokens} ({self})"
        )
        assert self.scheduled_tokens <= self.running_tokens, (
            f"running_tokens={self.running_tokens} is below the rows this rank "
            f"scheduled ({self.scheduled_tokens}) ({self})"
        )
        # Per-token: sized by `running_tokens`, the height everything per-row runs.
        # 【译】逐 token：按 `running_tokens` 定大小，这是所有逐行东西跑的高度。
        slot_rows = _rows("slot_mapping")
        assert slot_rows in (None, self.running_tokens), (
            f"slot_mapping length {slot_rows} != running_tokens="
            f"{self.running_tokens} ({self}); the builder used another width"
        )
        # Per-sequence: sized by `running_bs`. `cu_seqlens_q` carries the extra
        # boundary entry; the state slots are V4's and absent elsewhere.
        # 【译】逐序列：按 `running_bs` 定大小。`cu_seqlens_q` 多一个边界项；state slot 是 V4 的，别的后端没有。
        cu_rows = _rows("cu_seqlens_q")
        assert cu_rows in (None, self.running_bs + 1), (
            f"cu_seqlens_q length {cu_rows} != running_bs+1="
            f"{self.running_bs + 1} ({self})"
        )
        slot_out_rows = _rows("state_slot_out")
        assert slot_out_rows in (None, self.running_bs), (
            f"state_slot_out length {slot_out_rows} != running_bs="
            f"{self.running_bs} ({self})"
        )
        # No rectangle assertion: `_running_tokens` deliberately returns a
        # height ABOVE `running_bs * max_seqlen_q` when nothing recorded holds
        # the run.
        # 【译】不断言矩形：没有录制装得下扁平行时，`_running_tokens` 会故意返回比 `running_bs * max_seqlen_q` 更大的高度。


def running_tokens_from_bs(bs: int, *, is_prefill: bool, attn_metadata) -> int:
    """``running_tokens`` for a BRIDGE that only knows a request count.

    Native ATOM never calls this -- it reads `ForwardMode.running_tokens`. A
    plugin bridge has no ForwardMode, so it reconstructs the rectangle here and
    nowhere else.

    Prefill returns the count verbatim, which is NOT a height: prompt lengths
    are ragged, so no multiplier recovers one, and a number below the real
    height just leaves the batch unpadded -- correct there, since a prefill
    all_gather is variable-length.
    """
    # 【译】给只知道请求数的 BRIDGE 用的 ``running_tokens``。
    #
    #   原生 ATOM 从不调用它——它读的是 `ForwardMode.running_tokens`。插件 bridge 没有 ForwardMode，所以在这里、
    #   也只在这里重建矩形。
    #
    #   prefill 原样返回这个数，它不是高度：prompt 长短不一，任何乘数都还原不出高度；比真实高度小的数只是让 batch
    #   不补齐——在那里这是对的，因为 prefill 的 all_gather 是变长的。
    # 【分支】调用方：plugin/rtpllm/utils/forward_context.py、plugin/vllm/deepseek_v4_bridge.py（第10批）
    if is_prefill or attn_metadata is None:
        return bs
    return bs * int(attn_metadata.max_seqlen_q)


# Context：本步的"形状"与少量逐步数据（ForwardContext.context）
#   写入方：ModelRunner.prepare_inputs（真实的步，带 forward_mode）；capture_cudagraph 的 build_capture、
#     DP dummy、drafter 的 draft pass（is_draft=True）各自构造"声明形状"的 Context
#   读取方：run_model（scheduled_bs、positions、forward_mode）、MoE（scheduled_tokens、running_tokens*）、
#     LM head（is_prefill、is_draft）、注意力（is_dummy_run 等）、drafter / TBO 包装
# 对比nano-vllm：nano-vllm 的 Context 只有 is_prefill 这一个"形状"字段——它不补行（eager）或只在 run_model 里
#   临时补到 graph_bs；ATOM 把补齐后的形状算一次、存在这里，所有层读同一份
# Python 语法：类上写了 @dataclass 又手写了 __init__——dataclass 不会覆盖类里已经定义的方法，所以用的是手写版
#   （参数与字段一一对应），dataclass 仍然提供 __repr__、__eq__ 和 dataclasses.fields()
@dataclass
class Context:
    # This context is used to store the basic context of the forward.
    # 【译】这个上下文用来存放 forward 的基本信息。
    # positions：本步每个 token 的位置 [scheduled_tokens] int64（mRoPE 模型为 [3, N]）；
    #   prefill = 已缓存前缀长度 + 块内偏移；decode = context_len - q + j（j = 0..q-1，不投机时就是 context_len - 1，
    #   token_layout/decode.py decode_positions）。RoPE 和部分注意力后端读它
    # is_dummy_run：DP 锁步的陪跑 batch（engine_core.py DPEngineCoreProc 上方）；注意力据此直接跳过计算
    #   （attention_mha.py 返回 torch.empty_like(q)），只为陪同伴走完各层的集合通信
    positions: torch.Tensor
    is_prefill: bool = False
    is_dummy_run: bool = False
    # What this rank was handed. Duplicated from `forward_mode` because a
    # capture context has none; `scheduled_tokens` counts this rank's real rows.
    # 【译】本 rank 收到的数量。从 `forward_mode` 复制一份，因为录图用的上下文没有 forward_mode；
    #   `scheduled_tokens` 数的是本 rank 的真实行数。
    # 注意 use_dp_stage（ForwardContext）会在一个 with 块内临时改写 scheduled_tokens
    scheduled_bs: int = 0
    scheduled_tokens: int = 0
    # The step's DP-unified padded shape. `running_bs` counts SEQUENCES (graph
    # identity, the draft's pad width), `running_tokens` the hidden_states rows
    # MoE pads to. Both stored, because the ratio is not always max_seqlen_q --
    # a DSpark ragged step runs a packed width no rectangular bs*q recovers.
    # 【译】本步在 DP 间统一后的补齐形状。`running_bs` 数的是序列（决定用哪张图、draft 补到多宽），
    #   `running_tokens` 是 MoE 要补到的 hidden_states 行数。两个都存，因为它们的比值不一定是 max_seqlen_q——
    #   DSpark 的 ragged 步跑的是一个打包后的宽度，任何矩形 bs*q 都还原不出来。
    # is_draft：草稿模型的 forward（MTP / EAGLE / DSpark 的 draft pass，第7批）；LM head 在 draft 时不按 cu_seqlens_q 取末行
    running_bs: int = 0
    running_tokens: int = 0
    is_draft: bool = False
    # Every rank of the group is decoding, so `running_tokens` above is the
    # group's number and not this rank's own. That is what picks MoE's
    # fixed-size all_gather over the variable-length one, and so what makes a
    # captured graph usable at all. One meaning, deliberately: it once also
    # stood for "dp-attention is off", and readers needing the strong claim --
    # `_can_use_dp_sharded_head` above all -- silently got the weak one.
    # 【译】组里每个 rank 都在 decode，所以上面的 `running_tokens` 是全组的数，而不是本 rank 自己的。正是它让 MoE 选
    #   定长 all_gather 而不是变长的，也因此才让录下来的图能用。刻意只有一个含义：它曾经还表示"没开 dp-attention"，
    #   于是需要强含义的读取方——首先是 `_can_use_dp_sharded_head`——悄悄拿到了弱含义。
    running_tokens_are_unified: bool = True
    # `running_tokens` for every rank, unreduced; this rank's entry IS
    # `running_tokens`. Consumers reduce it, because they want different
    # reductions: padding each rank up to the group is a MAX, an all2all's
    # receive width holds what every rank sent and is a SUM. None where no DP
    # reduction produced one, which is where the two coincide.
    # 【译】每个 rank 的 `running_tokens`，未归约；本 rank 那一项就是 `running_tokens`。由使用方自己归约，因为它们要的
    #   归约不同：把每个 rank 补到全组高度是 MAX，all2all 的接收宽度要装下每个 rank 发来的东西，是 SUM。
    #   没有经过 DP 归约时为 None，那种情况下两者一致。
    # 写入方：set_forward_context（每次无条件覆盖，见那里）；DP=4、表 [13, 70, 1, 40] 时就是 (13, 70, 1, 40)
    running_tokens_across_dp: tuple[int, ...] | None = None
    # The step's whole shape decision. Set by `prepare_model` via
    # `ForwardMode.decide`; None only on a capture context, which declares its
    # shape rather than reading one.
    # 【译】本步完整的形状决定。由 `prepare_model` 经 `ForwardMode.decide` 设置；只有录图用的上下文为 None，
    #   它是声明形状，而不是读出形状。
    forward_mode: ForwardMode | None = None
    # Optional flat token ids for the current forward. Read by callbacks
    # invoked inside Dynamo-opaque custom ops (e.g. V4 MoE hash routing)
    # that need the token ids but cannot receive them as a function arg
    # (the op signature is fixed by the consumer's plugin contract).
    # 【译】可选：本次 forward 的扁平 token id。由 Dynamo 看不透的自定义算子内部调用的回调读取（例如 V4 MoE 的 hash 路由），
    #   它们需要 token id，却不能通过函数参数拿到（算子签名由使用方的插件约定固定）。
    # 写入方：models/deepseek_v4.py、model_runner.py 的 PCP 分支（第8、9批）
    input_ids: torch.Tensor | None = None
    # Row offset of this (micro-)batch's tokens on the full forward's token
    # axis. 0 except inside a TBO micro-batch. Anything writing into a
    # whole-forward buffer from inside the model must add it, or the ubatches
    # land on each other's rows. Read per-thread (the context is thread-local).
    # 【译】本（微）batch 的 token 在整个 forward 的 token 轴上的行偏移。除了 TBO 微批内部都是 0。模型内部任何往
    #   "整个 forward 共用的缓冲区"里写的代码都必须加上它，否则两个微批会写到彼此的行上。按线程读取（上下文是线程局部的）。
    ubatch_token_offset: int = 0

    # Optional speculative-decoding inputs staged alongside the target
    # forward's other host-to-device copies. Keeping these on the per-forward
    # context avoids launching pinned-buffer H2Ds later from postprocess.
    # 【译】可选：与目标模型 forward 的其他 H2D 拷贝一起暂存的投机解码输入。放在每次 forward 的上下文里，
    #   就不必之后在 postprocess 里再从 pinned 缓冲发起 H2D。
    # 写入方：prepare_model 末尾（drafter.anchors_to_gpu）；读取方：eagle_proposer.py（第7批）
    #   为什么不能在 postprocess 里再拷：那时本步的发布窗口已关（h2d_owner.finish），之后再改 pinned 源不受保护，见 utils/h2d.py
    draft_anchor_overrides: torch.Tensor | None = None

    def __init__(
        self,
        positions: torch.Tensor,
        is_prefill: bool = False,
        is_dummy_run: bool = False,
        scheduled_bs: int = 0,
        scheduled_tokens: int = 0,
        running_bs: int = 0,
        running_tokens: int = 0,
        is_draft: bool = False,
        running_tokens_are_unified: bool = True,
        running_tokens_across_dp: tuple[int, ...] | None = None,
        forward_mode: ForwardMode | None = None,
        input_ids: torch.Tensor | None = None,
        ubatch_token_offset: int = 0,
        draft_anchor_overrides: torch.Tensor | None = None,
    ):
        self.positions = positions
        self.is_prefill = is_prefill
        self.is_dummy_run = is_dummy_run
        self.scheduled_bs = scheduled_bs
        self.scheduled_tokens = scheduled_tokens
        self.running_bs = running_bs
        self.running_tokens = running_tokens
        self.is_draft = is_draft
        self.running_tokens_are_unified = running_tokens_are_unified
        self.running_tokens_across_dp = running_tokens_across_dp
        self.forward_mode = forward_mode
        self.input_ids = input_ids
        self.ubatch_token_offset = ubatch_token_offset
        self.draft_anchor_overrides = draft_anchor_overrides


# AttentionMetaData：注意力（和少数要逐 token 信息的层）本步要的全部元数据
#   填写方：model_ops/attentions/ 下各后端 builder 的 build → prepare_prefill / prepare_decode
#     （通用 prepare_prefill 在 backends.py；MHA 的 prepare_decode 在 aiter_attention.py；MLA / DSA 在 aiter_mla.py；
#      V4、V4.1、GDN 各有自己的，第6~8批）。张量都是 forward_vars 里固定 GPU buffer 的前缀切片（CUDA Graph 要固定地址），
#      CPU 先写 pinned 端，再经 utils/h2d.py 发布上去；Python int 字段直接在 CPU 上算
#   读取方：get_forward_context().attn_metadata —— attention_mha.py / attention_mla.py 等注意力实现、LM head（embed_head.py）、
#     投机解码的 proposer、TBO 拆分（utils/tbo/ubatch_splitting.py）、插件 bridge
#   形状记号：B = 逐序列宽度（decode 为 running_bs；prefill 为本步序列数，部分数组再补到 running_bs）
#             T = 逐 token 行数（decode 为 running_tokens；prefill 为 scheduled_tokens）   M = 每序列最多 block 数
#
#   ── nano-vllm 也有的（含义相同；nano-vllm Context 的第 8 个字段 is_prefill 在 ATOM 里放在 Context）──
#   cu_seqlens_q    int32 [B+1]  各序列 query 行数的前缀和，第 i 条序列的行 = [cu[i], cu[i+1])；补出来的序列长度为 0。
#                                唯一写入方 backends.py prepare_cu_seqlens_q（在 input ids 之前发布，decode_input_ids 的 kernel 也读）；
#                                prefill 时 LM head 用 cu_seqlens_q[1:]-1 取每条序列最后一行
#   cu_seqlens_k    int32 [B+1]  prefill：各序列 K 长度（已缓存前缀 + 本次新 token）的前缀和。例：缓存 256、本次 100 → q 100、k 356
#   max_seqlen_q/k  int          单条序列 q / k 的最大长度（flash-attention 用来规划）；decode 时 max_seqlen_q = 1+k，也是图 key 的第 2 维
#   slot_mapping    int64 [T]    每行 token 的 KV 写到哪个槽：slot = 物理 block 号 × block_size + 块内偏移；-1 = 不写（padding 行）
#   context_lens    int32 [B]    每条序列（含本步 token）的 KV 长度；补出来的序列为 0，读不到任何 KV
#   block_tables    int32 [B, M] block_tables[i][j] = 第 i 条序列第 j 个逻辑 block 的物理号（分页 KV 见 sequence.py
#                                BlockTable 上方的"科普：KV cache分块"）；decode 必需，prefill 只在有前缀命中时上传
#   ── ATOM 多出来的（按字段顺序）──
#   min_seqlen_q / dropout_p     prefill 写 0 / 0.0，原样传给 aiter 的 varlen kernel（推理不用 dropout，接口沿用 flash-attention）
#   causal          bool         只有 DSpark 的双向 draft block 为 False（第7批）
#   state           AttnState    见 AttnState 上方
#   kv_indptr / kv_indices       block_tables 的 CSR 压缩形式。前置：block_tables 是 [B, M] 的稠密表，大多数格子空着；
#                   int32 [B+1] / [Σ]   aiter 的 paged attention / MLA kernel 要的是压紧的"行指针 + 列表"。
#                                例：序列 0 占物理块 [7, 2, 9]、序列 1 占 [4] → kv_indptr=[0, 3, 4]、kv_indices=[7, 2, 9, 4]。
#                                MHA decode：kv_indptr 在 CPU 上累加 cdiv(context_lens, block_size)，kv_indices 由
#                                kv_indices_generate_triton 在 GPU 上从 block_tables 展开；MLA 默认 1 token 一页（见下一项）
#   qo_indptr       int32        q 侧的 CSR（≈ cu_seqlens_q）。基础 MHA 直接把 cu_seqlens_q 当 qo_indptr 传给 kernel；
#                                这个字段只由 DeepSeek-V4 注意力和插件 bridge 设置
#   kv_last_page_lens int32 [B]  每条序列最后一页里的有效 token 数。MLA 默认 1 token 一页（ATOM_MLA_PAGE_SIZE ≤ 1 时），
#                                此时恒为 1，aiter_mla.py 在初始化时整块填 1；页更大时写真实的末页填充数
#   cu_seqlen_ks / cu_seqlen_ke  DSA 索引器（prefill）：第 t 个 query token 可以打分的 key 区间 [ks, ke)（因果窗口，key 按序列拼接）。
#                   int32 [T]    aiter_mla.py 填，deepseek_v2.py 的 Indexer 读（DSA 见 arg_utils.py 的【科普】DSA）
#   sparse_kv_indptr int32 [T+1] DSA 主注意力：每个 query token 当成一条"序列"，它选中的 top-k 个 KV 位置组成 CSR
#   sparse_kv_last_page_lens     同上，每个 query token（decode 为每条序列）一个 1；与 kv_last_page_lens 分开，免得互相覆盖
#   dcp_local_context_lens / index_row_ends / dcp_token_block_tables 这三个：
#                                DSA + DCP（KV 按 rank 切开）/ FP4 索引器的逐 query token 元数据（第7、9批；
#                                DCP 见 arg_utils.py 中 decode_context_parallel_size 上方的【科普】）
#   work_meta_data / work_indptr / work_info_set / reduce_indptr / reduce_final_map / reduce_partial_map 这六个：
#                                aiter persistent decode kernel 的调度表。前置（split-KV / flash-decoding）：decode 时每条序列
#                                只有 1~(1+k) 行 q，若一个 workgroup 负责一条序列（MLA 只有 1 个 KV 头），bs=4 时全卡只有 4 份活，
#                                而 MI300X 有 304 个 CU（AMD 公布规格）。于是把每条序列的 KV 切成若干段分给不同 CU，各算一部分
#                                softmax·V，再按 log-sum-exp 合并（reduce）。这 6 个张量就是"怎么切、谁算哪段、结果合并到哪"，
#                                由 aiter 在 GPU 上生成（MLA：get_mla_metadata_v1，aiter_mla.py；MHA 在 block_size 为 256/1024 时：
#                                get_pa_metadata_v1，aiter_attention.py），大小事先由 *_info_v1 算好并在启动时分配，地址固定
#   has_cached / total_kv / num_cached_tokens / seq_starts     前缀缓存（prefill，见 "for prefix cache" 下面）
#   kpool_total_pools            GLM-5.3 k-pool：拼接后的 key 里共有几个整池（k-pool 见 config.py glm5_kpool_block_size 上方【科普】③）
#   sparse_cu_seqlens_q / batch_id_per_q_token / batch_id_per_k_token   见 __init__ 末尾（这三个不是 dataclass 字段）
#   此外 builder 还会动态挂一些后端私有属性（如 aiter_attention 的 flydsl_work_plan、aiter_mla 的 g_kv_indptr、dtype_q）：
#   类没有 __slots__，可以随意加属性
# 对比nano-vllm：nano-vllm 只有前 8 个，因为它只用 flash-attention 的 varlen / kvcache 两个接口；ATOM 的 aiter kernel
#   要 CSR 和预先算好的调度表，又支持 MLA / DSA / 前缀缓存，所以多出近 30 个字段
@dataclass
class AttentionMetaData:
    """Attention metadata for prefill and decode batched together."""

    # 【译】prefill 和 decode 一起批处理时的注意力元数据。
    # 解释：原生调度器一步只做 prefill 或只做 decode，"一起批处理"指的是接口设计（字段同时覆盖两种）

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    dropout_p: float = 0.0
    # True for standard causal attention; False only for DSpark's bidirectional
    # draft block. The MLA asm decode kernel selects a different .co by this flag.
    # 【译】标准因果注意力为 True；只有 DSpark 的双向 draft block 为 False。MLA 的汇编 decode kernel 按这个标志选不同的 .co
    #   （.co = 预编译好的 GPU 代码对象文件）。
    causal: bool = True

    state: AttnState = AttnState.PREFILL_NATIVE
    """One of `DECODE / PREFILL_NATIVE / PREFILL_PREFIX` — controls which
    kv-indices buffers downstream forward branches read. Default is
    `PREFILL_NATIVE`; every `prepare_*` path overrides explicitly.
    Backends that don't need the NATIVE/PREFIX distinction can treat
    `any PREFILL_*` as prefill. See ``AttnState`` for full semantics."""
    # 【译】`DECODE / PREFILL_NATIVE / PREFILL_PREFIX` 之一——决定下游 forward 分支读哪些 kv-indices 缓冲区。
    #   默认 `PREFILL_NATIVE`；每条 `prepare_*` 路径都会显式覆盖。不需要区分 NATIVE/PREFIX 的后端可以把
    #   任何 `PREFILL_*` 都当作 prefill。完整语义见 ``AttnState``。
    # 观察：并非"每条 prepare_* 都显式覆盖"。显式设置 state 的只有 backends.py 的 prepare_prefill 和 V4 / V4.1 的 builder；
    #   通用的 MHA / MLA decode（aiter_attention.py、aiter_mla.py 的 prepare_decode）构造 AttentionMetaData 时不传 state，
    #   decode 步的 state 仍是默认的 PREFILL_NATIVE。这些后端的注意力不读 state（判 decode 看 Context.is_prefill），
    #   所以不影响结果；但读 state 的新代码不能假定 decode 步一定是 DECODE。
    # Python 语法：字段下面紧跟的字符串是"属性 docstring"，只是一个表达式语句，不影响运行

    kv_indptr: torch.Tensor | None = None
    kv_indices: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    kv_last_page_lens: torch.Tensor | None = None
    cu_seqlen_ks: torch.Tensor | None = None
    cu_seqlen_ke: torch.Tensor | None = None
    sparse_kv_indptr: torch.Tensor | None = None
    # Last-page lens for sparse (DSA) attention: all 1s, one per query token in
    # prefill/MTP-verify and per seq in decode. Separate from kv_last_page_lens
    # (the dense per-seq buffer) so the two never clobber each other.
    # 【译】稀疏（DSA）注意力的最后一页长度：全为 1；prefill / MTP 验证时每个 query token 一个，decode 时每条序列一个。
    #   与 kv_last_page_lens（稠密的逐序列缓冲区）分开，两者就永远不会互相覆盖。
    sparse_kv_last_page_lens: torch.Tensor | None = None
    # Per-rank KV length of each query token's causal window, for the sparse
    # DSA + DCP indexer: MTP verify gives each draft position its own window.
    # The padded running-width buffer is shared by eager, graph and TBO paths.
    # 【译】每个 query token 因果窗口在本 rank 上的 KV 长度，供稀疏 DSA + DCP 索引器使用：MTP 验证让每个 draft 位置
    #   都有自己的窗口。按补齐后的 running 宽度分配的缓冲区由 eager、回放和 TBO 三条路径共用。
    dcp_local_context_lens: torch.Tensor | None = None
    # The FP4 sparse indexer's key bound for each query token (its position +
    # 1), published once per decode step for every full layer; a draft step's
    # aliases context_lens.
    # 【译】FP4 稀疏索引器对每个 query token 的 key 上界（它的位置 + 1），每个 decode 步为所有全注意力层发布一次；
    #   draft 步的这一项直接别名到 context_lens。
    index_row_ends: torch.Tensor | None = None
    # Block-table row per query token, for the same indexer -- both aiter ops
    # address the table by row. Aliases block_tables at one query per sequence.
    # Every producer that rewrites block_tables must rewrite this too: a stale
    # one has the right length and the wrong rows.
    # 【译】每个 query token 对应的 block table 行，供同一个索引器使用——两个 aiter 算子都按行寻址这张表。每条序列只有
    #   一个 query 时它就是 block_tables 的别名。每个改写 block_tables 的生产者也必须改写它：过期的版本长度对、行却是错的。
    # 注意：它是类级字段，但下面手写的 __init__ 没有这个参数——实例上读到的是类属性默认值 None，直到 builder
    #   （aiter_mla.py _publish_dcp_token_block_tables）给实例赋值
    dcp_token_block_tables: torch.Tensor | None = None

    # aiter persistent decode 的 6 张调度表（split-KV，见类上方字段表）
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # for prefix cache
    # 【译】用于前缀缓存
    #   has_cached         prefill：是否有序列带已缓存的前缀（cached_lens > 0，前缀命中或 chunked prefill 的后续块）
    #                      → 注意力要先把已缓存的 KV 从分页缓存 gather 出来，再与新 token 一起算
    #   total_kv           prefill：全 batch 的 KV 行数（已缓存 + 新）= cu_seqlens_k[B]；在 CPU 上算好，
    #                      gather 缓冲区据此开大小，不必在每层同步读 GPU
    #   kpool_total_pools  GLM-5.3 索引器：拼接后 key 里的整池数 = Σ (每序列 K 长度 // index_kpool)
    #   num_cached_tokens  int32 [prefill 序列数]：每条序列已缓存的前缀长度；没有任何命中时不上传、保持 None
    #                      （使用方据此判断"完全没有前缀"）
    #   seq_starts         int32 [prefill 序列数]：恒为 0，cp_mha_gather_cache 要的参数（前缀总从位置 0 开始）
    #   例：两条 prefill，A 命中 256 个 token、本次算 100 个，B 无命中算 50 个 → has_cached=True、
    #     cu_seqlens_k=[0, 356, 406]、total_kv=406、num_cached_tokens=[256, 0]、state=PREFILL_PREFIX
    has_cached: bool = False
    total_kv: int | None = None
    kpool_total_pools: int | None = None
    num_cached_tokens: torch.Tensor | None = None
    seq_starts: torch.Tensor | None = None

    # 手写 __init__（dataclass 不会覆盖它）：参数与上面的字段基本一一对应，另外多收 3 个非字段属性
    #   （sparse_cu_seqlens_q、batch_id_per_q_token、batch_id_per_k_token），少一个 dcp_token_block_tables
    def __init__(
        self,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        min_seqlen_q: int = 0,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = True,
        state: AttnState = AttnState.PREFILL_NATIVE,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        qo_indptr: torch.Tensor | None = None,
        kv_last_page_lens: torch.Tensor | None = None,
        cu_seqlen_ks: torch.Tensor | None = None,
        cu_seqlen_ke: torch.Tensor | None = None,
        sparse_kv_indptr: torch.Tensor | None = None,
        sparse_kv_last_page_lens: torch.Tensor | None = None,
        dcp_local_context_lens: torch.Tensor | None = None,
        index_row_ends: torch.Tensor | None = None,
        work_meta_data: torch.Tensor | None = None,
        work_indptr: torch.Tensor | None = None,
        work_info_set: torch.Tensor | None = None,
        reduce_indptr: torch.Tensor | None = None,
        reduce_final_map: torch.Tensor | None = None,
        reduce_partial_map: torch.Tensor | None = None,
        sparse_cu_seqlens_q: torch.Tensor | None = None,
        batch_id_per_q_token: torch.Tensor | None = None,
        batch_id_per_k_token: torch.Tensor | None = None,
        has_cached: bool = False,
        total_kv: int | None = None,
        kpool_total_pools: int | None = None,
        num_cached_tokens: torch.Tensor | None = None,
        seq_starts: torch.Tensor | None = None,
    ):
        self.has_cached = has_cached
        self.total_kv = total_kv
        self.kpool_total_pools = kpool_total_pools
        self.num_cached_tokens = num_cached_tokens
        self.seq_starts = seq_starts
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.min_seqlen_q = min_seqlen_q
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.dropout_p = dropout_p
        self.causal = causal
        self.state = state
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.kv_last_page_lens = kv_last_page_lens
        self.cu_seqlen_ks = cu_seqlen_ks
        self.cu_seqlen_ke = cu_seqlen_ke
        self.sparse_kv_indptr = sparse_kv_indptr
        self.sparse_kv_last_page_lens = sparse_kv_last_page_lens
        self.dcp_local_context_lens = dcp_local_context_lens
        self.index_row_ends = index_row_ends
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        # 下面 3 个没有在类体里声明成字段，只是普通实例属性：
        #   sparse_cu_seqlens_q   int32 [T+1]：DSA 把每个 query token 当成一段，恒为 arange(0..T)（aiter_mla.py 启动时写好）
        #   batch_id_per_q_token  int32 [T]：query token → 所属请求号；CUDA Graph 补齐的尾部为 -1（token_layout/batch_ids.py）
        #   batch_id_per_k_token  见下一行英文；prefill 有前缀或开 TBO 时由 backends.py prepare_prefill 在 GPU 上构建
        # 观察：因为不是字段，dataclasses.fields() 看不到它们，下面的 asdict_zerocopy 会漏掉这 3 个（目前本仓库没有调用
        #   asdict_zerocopy，所以没有实际影响）
        self.sparse_cu_seqlens_q = sparse_cu_seqlens_q
        self.batch_id_per_q_token = batch_id_per_q_token
        # [total_kv] int32 — the KV-axis twin, over the gathered cached+new KV.
        # Layer-invariant, so the prefill builder makes it once per fwd.
        # 【译】[total_kv] int32——KV 轴上的对应物，覆盖 gather 出来的"已缓存 + 新"KV。与层无关，所以 prefill builder
        #   每次 forward 只构建一次。
        self.batch_id_per_k_token = batch_id_per_k_token

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        # 【译】类似 dataclasses.asdict，但不做深拷贝。
        # 返回 {字段名: 原对象}（张量是同一个对象，不复制）；skip_fields 里的字段跳过。沿用 vLLM，本仓库目前没有调用方
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        # 【译】注意：如果以后把 dataclass 作为字段加进来，它们也需要类似的处理。
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in skip_fields
        }


def get_published_dcp_local_context_lens(
    attn_metadata: AttentionMetaData, num_rows: int
) -> torch.Tensor | None:
    """Return the valid prefix of a published sparse-DCP local-length buffer.

    Metadata is allocated at the engine's padded running width while PIECEWISE
    eager execution may score only the scheduled rows. The real requests are
    always the prefix and the producer zero-fills the padded tail.
    """
    # 【译】返回已发布的稀疏 DCP 本地长度缓冲区的有效前缀。
    #
    #   元数据按引擎补齐后的 running 宽度分配，而 PIECEWISE 的 eager 执行可能只给调度的那些行打分。
    #   真实请求总在前缀里，生产者会把补齐的尾部填 0。
    # 【分支】DSA + DCP（第9批）；调用方 model_ops/dcp_ops.py。缓冲区不够 num_rows 行时返回 None
    local_ctx = attn_metadata.dcp_local_context_lens
    if local_ctx is None or local_ctx.shape[0] < num_rows:
        return None
    if local_ctx.shape[0] == num_rows:
        return local_ctx
    return local_ctx[:num_rows]


# 【分支】GDN（Gated DeltaNet，Qwen3-Next / Qwen3.5 等的线性注意力层）的元数据，由 attentions/gdn_attn.py 构建（第8批）。
#   线性注意力没有逐 token 的 KV cache，每个请求只有一个定长状态（见 model_config/qwen3_next.py 文件头【科普】②），
#   所以这里存的是 prefill / decode / 投机 decode 各有几条、状态槽在哪（*_state_indices_tensor）、
#   query 起点（*_query_start_loc）。行尾英文是各张量的形状（shape: [...]）
@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    # 【译】下面这些属性用于 causal_conv1d 的 Triton 实现。
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


# 【分支】DeepSeek-V4.1 decoder 的有界重放（第8批 models/deepseek_v41/bounded_replay.py）：prefill 时部分 MoE 层跑的行
#   与本步的行不同，于是这些层的 DP 集合通信也要按"该阶段"的 token 数开缓冲。下面 3 个类把各阶段的 token 数
#   搭进 ForwardMode.decide 那一次 all_gather，再在 set_forward_context 里还原成每阶段一个 DPMetadata
#   报告（Report，本 rank 发出）→ 线上格式 wire()=(applies, active, *tokens) → 全组的表（Stages）→ ForwardContext.use_dp_stage
@dataclass(frozen=True)
class BoundedReplayDPReport:
    """A rank's report for the step's DP sync, from a model some of whose MoE
    layers run on other rows than the step (`AttentionMetadataBuilder.
    dp_stage_report`).

    ``names``/``tokens``: this rank's token count per stage, in an order the
    model fixes (every rank sends the same layout). ``applies``: this rank
    runs those stages on these counts this step; the group uses the stage
    tables when any rank does. ``active``: a model-defined flag, OR-reduced.
    """

    # 【译】一个 rank 为本步 DP sync 提交的报告，来自某些 MoE 层跑的行与本步不同的模型（`AttentionMetadataBuilder.
    #   dp_stage_report`）。
    #
    #   ``names``/``tokens``：本 rank 每个阶段的 token 数，顺序由模型固定（每个 rank 发同样的布局）。``applies``：本 rank
    #   这一步按这些数跑这些阶段；只要有 rank 这样做，全组就用阶段表。``active``：模型自定义的标志，做 OR 归约。

    names: tuple[str, ...]
    tokens: tuple[int, ...]
    applies: bool
    active: bool = True

    def wire(self) -> tuple[int, ...]:
        return (int(self.applies), int(self.active), *self.tokens)


@dataclass(frozen=True)
class BoundedReplayDPStage:
    """One stage's token counts across DP and the `DPMetadata` the MoE
    variable-length gather reads while it runs (`ForwardContext.use_dp_stage`)."""

    # 【译】一个阶段在 DP 间的 token 数，以及该阶段运行时 MoE 变长 gather 读的 `DPMetadata`（`ForwardContext.use_dp_stage`）。

    tokens_across_dp: tuple[int, ...]
    local_tokens: int
    dp_metadata: DPMetadata


@dataclass(frozen=True)
class BoundedReplayDPStages:
    """The step's stage tables, when its DP sync carried them.

    ``applies``: this rank's own report -- the one decision it acts on, so
    what it forwards is what it told the group. ``active``: OR over the
    group's flags."""

    # 【译】本步的阶段表（当本步的 DP sync 带了它们时）。
    #
    #   ``applies``：本 rank 自己的报告——它据以行动的那个决定，所以它实际跑的就是它告诉全组的。
    #   ``active``：全组标志的 OR。

    applies: bool
    active: bool
    stages: dict[str, BoundedReplayDPStage]


# 把 sync.dp_stage_rows（[2 + 阶段数][dp_size]：第 0 行 applies、第 1 行 active、之后每阶段一行）还原成每阶段的
#   DPMetadata；没有报告或本步 sync 没带时返回 None。调用方：set_forward_context
def _bounded_replay_dp_stages(
    parallel_config, report, sync
) -> "BoundedReplayDPStages | None":
    rows = None if sync is None else getattr(sync, "dp_stage_rows", None)
    if rows is None or report is None:
        return None
    rank = parallel_config.data_parallel_rank
    stages = {}
    for name, tokens in zip(report.names, rows[2:]):
        tokens = tuple(int(n) for n in tokens)
        stages[name] = BoundedReplayDPStage(
            tokens_across_dp=tokens,
            local_tokens=tokens[rank],
            dp_metadata=DPMetadata.make(
                parallel_config,
                tokens[rank],
                torch.tensor(tokens, dtype=torch.int32),
            ),
        )
    return BoundedReplayDPStages(
        applies=report.applies, active=any(rows[1]), stages=stages
    )


# ForwardContext：get_forward_context() 返回的顶层容器，每步由 set_forward_context 新建一个（不在原对象上改）
#   字段速览（本 rank 视角；例子沿用文件头：13 条 decode、MTP k=1、单卡）：
#     attn_metadata        本步的 AttentionMetaData（cu_seqlens_q 17 个数、slot_mapping 32 行……）
#     context              本步的 Context（scheduled 13/26、running 16/32、forward_mode）
#     kv_cache_data        {"layer_0": KVCacheTensor, …}，启动时 set_kv_cache_data 登记一次，每步复制引用
#     dp_metadata          DP>1 时的 DPMetadata；单卡为 None
#     spec_decode_metadata 开投机且本步要验证 draft 时的 SpecDecodeMetadata
#     其余                 TBO 微批、stream / 录图标志、分段图分派键、DP pad-row 记录、V4.1 阶段表（逐个见下）
# Python 语法：field(default_factory=dict) 让每个实例拿到自己的新 dict（直接写 = {} 会被所有实例共享，dataclass 也会报错）；
#   __post_init__ 是 dataclass 生成的 __init__ 末尾自动调用的钩子
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    # 【译】从 vllm_config.compilation_config.static_forward_context 复制过来
    # 解释：ATOM 里是 atom_config.compilation_config.static_forward_context（{层名: 模块}），set_forward_context 传的是
    #   同一个 dict 对象，不是拷贝。Dynamo 看不透的自定义算子（注意力、MoE，见 base_attention.py、module_dispatch_ops.py）
    #   在算子内部靠层名从 static_forward_context 找回自己的 module——它们读的是 compilation_config 上的那个，
    #   本仓库没有代码读 ForwardContext.no_compile_layers（字段沿用 vLLM）
    no_compile_layers: dict[int, Any] = field(default_factory=dict)

    # attn_metadata：原生 ATOM 每步一个 AttentionMetaData；类型里的 dict（层名 → 元数据）沿用 vLLM，
    #   __post_init__ 把 None 换成 {}（reset 后的"空"上下文就是这个状态）
    attn_metadata: Union["AttentionMetaData", dict[str, "AttentionMetaData"]] | None = (
        None
    )

    # kv_cache_data：{"layer_{n}": KVCacheTensor(k_cache, v_cache, k_scale, v_scale, …)}；注意力按自己的 layer_num 取
    #   （attention_mha.py：fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"].k_cache）。来源见 set_kv_cache_data
    kv_cache_data: dict[str, KVCacheTensor] = None

    # context / dp_metadata / spec_decode_metadata：见各自的类；ubatch_slices：TBO 把本步切成的微批（第9批）
    context: Context | None = None

    dp_metadata: DPMetadata | None = None

    spec_decode_metadata: SpecDecodeMetadata | None = None

    ubatch_slices: list[Any] | None = None

    # Cross-DP MAX of per-ubatch token counts, reduced in the step's one packed
    # all_gather (``ForwardMode.decide``) and propagated here so
    # ``UBatchWrapper._compute_ub_graph_bs`` no longer needs its own
    # per-ubatch all_reduce. Shape: tuple of length N == len(ubatch_slices).
    # None when DP is off or when TBO is not active this step.
    # 【译】每个微批 token 数的跨 DP 最大值，在本步唯一那次打包 all_gather（``ForwardMode.decide``）里归约，再传到这里，
    #   这样 ``UBatchWrapper._compute_ub_graph_bs`` 就不再需要自己按微批做 all_reduce。形状：长度 N == len(ubatch_slices)
    #   的 tuple。DP 关闭或本步 TBO 不生效时为 None。
    ub_max_tokens_across_dp: tuple | None = None
    # The same counts UNREDUCED: per ubatch, every rank's. The MAX above is one
    # reduction of it and an all2all's receive width wants the SUM, so consumers
    # take their own. Same None conditions as the MAX.
    # 【译】同一组数的未归约版本：每个微批、每个 rank 各一个。上面的 MAX 是它的一种归约，而 all2all 的接收宽度要的是 SUM，
    #   所以使用方自己归约。为 None 的条件与 MAX 相同。
    ub_tokens_across_dp: tuple | None = None

    # Cached current_stream() captured at set_forward_context() time, so
    # downstream code (V4 attention / MoE / metadata builder) doesn't have
    # to query torch.cuda.current_stream() repeatedly during a forward —
    # multiple call sites caching independent Stream handles was widening
    # the hipStream handle pool and complicating reasoning about which
    # logical stream each wait_stream() refers to. CG capture / TBO
    # threads each call set_forward_context() inside their own stream
    # context, so the cached value is correct for the captured graph or
    # active thread.
    # 【译】在 set_forward_context() 时缓存下来的 current_stream()，这样下游代码（V4 注意力 / MoE / metadata builder）
    #   不必在一次 forward 里反复查询 torch.cuda.current_stream()——多个调用点各自缓存独立的 Stream 句柄，曾让
    #   hipStream 句柄池变大，也让"每个 wait_stream() 指的是哪条逻辑 stream"难以推理。CUDA Graph 录制和 TBO 线程
    #   都在各自的 stream 上下文里调用 set_forward_context()，所以缓存的值对录下的图或当前线程都是对的。
    #   （stream 是什么见 engine_core.py 文件头【3】的前置）
    main_stream: torch.cuda.Stream | None = None

    # True only while the model forward runs inside a CUDAGraph capture
    # block (model_runner.capture_model loop). Components that gate
    # multi-stream side-launches (V4 main Compressor on alt_stream,
    # indexer.compressor on indexer_stream) check this flag: side-stream
    # work is safe to emit inside a captured graph (graph records the
    # fork-join edges and replay re-uses the same stream layout) but
    # racy in eager mode where launches accumulate across layers and
    # deadlock the hipStream queue. Replay does not re-execute Python
    # forward, so it ignores the flag entirely.
    # 【译】只有当模型 forward 在 CUDAGraph 录制块内运行时（model_runner 的录制循环）才为 True。会把工作分到旁路 stream
    #   上发射的组件（V4 主 Compressor 在 alt_stream 上、indexer.compressor 在 indexer_stream 上）检查这个标志：
    #   在录制的图里发射旁路 stream 的工作是安全的（图会记下 fork-join 的依赖边，回放沿用同样的 stream 布局），
    #   但在 eager 模式下有竞态——发射会跨层累积，把 hipStream 队列卡死。回放不会重新执行 Python 的 forward，
    #   所以完全不看这个标志。
    # 写入方：capture_cudagraph 及其 PIECEWISE / draft 录制循环里的 set_forward_context(in_hipgraph=True)；
    #   读取方：下面的 side_stream、models/deepseek_v4.py
    in_hipgraph: bool = False

    # Piecewise-cudagraph dispatch, read per forward by CUDAGraphWrapper:
    # cudagraph_runtime_mode picks the capture/replay mode, batch_descriptor is
    # the key (num_tokens). None defaults keep wrappers inert until model_runner
    # sets them. Typed Any to dodge a CUDAGraphMode circular import.
    # 【译】分段 cudagraph 的分派信息，CUDAGraphWrapper 每次 forward 都读：cudagraph_runtime_mode 选择录制/回放模式，
    #   batch_descriptor 是 key（num_tokens）。默认 None 让各 wrapper 在 model_runner 设置之前不起作用。
    #   类型写成 Any 是为了避开 CUDAGraphMode 的循环导入。
    # 写入方：run_model 的 PIECEWISE 分支在调 self.model 前设成 PIECEWISE（piecewise_captured 时）或 NONE、
    #   key=BatchDescriptor(num_tokens=running_tokens)，调完复位；读取方：utils/cuda_graph.py CUDAGraphWrapper.__call__
    cudagraph_runtime_mode: Any = None
    batch_descriptor: Any | None = None

    # The DP pad-row work this forward already did (see `step_pad_rows`).
    # 【译】本次 forward 已经做过的 DP pad-row 计算（见 `step_pad_rows`）。
    pad_rows_done: set = field(default_factory=set)

    # Stage tables for a model some of whose MoE layers run on other rows
    # than the step (`BoundedReplayDPReport`), built from the step's DP sync when the
    # forward context is set. None on every step that did not carry them.
    # 【译】给"某些 MoE 层跑的行与本步不同"的模型用的阶段表（`BoundedReplayDPReport`），在设置 forward context 时
    #   由本步的 DP sync 构建。没带这些表的步上都是 None。
    dp_stages: BoundedReplayDPStages | None = None

    # __post_init__：容错——即使有人显式传了 None，也换成空容器
    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}
        if self.attn_metadata is None:
            self.attn_metadata = {}

    @contextmanager
    def use_dp_stage(self, name: str | None):
        """Size the DP MoE collectives for stage ``name`` while this block runs.

        The variable-length gather reads every rank's count from
        ``dp_metadata`` and this rank's own from ``context.scheduled_tokens``;
        both are switched to the stage's, prepared with the forward context,
        and restored on exit. A no-op when ``name`` is None or this step has
        no such stage, so a caller outside DP runs unchanged.
        """
        # 【译】在这个块运行期间，按阶段 ``name`` 给 DP MoE 集合通信定大小。
        #
        #   变长 gather 从 ``dp_metadata`` 读每个 rank 的数，从 ``context.scheduled_tokens`` 读本 rank 的数；
        #   两者都换成该阶段的（与 forward context 一起准备好），退出时恢复。``name`` 为 None 或本步没有这个阶段时
        #   什么都不做，所以 DP 之外的调用方行为不变。
        # 【分支】DeepSeek-V4.1 有界重放（第8批）
        stage = (
            None
            if self.dp_stages is None or name is None
            else self.dp_stages.stages.get(name)
        )
        if stage is None:
            yield
            return
        context = self.context
        outer = (
            self.dp_metadata,
            context.scheduled_tokens,
            context.running_tokens_across_dp,
        )
        self.dp_metadata = stage.dp_metadata
        context.scheduled_tokens = stage.local_tokens
        context.running_tokens_across_dp = stage.tokens_across_dp
        try:
            yield
        finally:
            (
                self.dp_metadata,
                context.scheduled_tokens,
                context.running_tokens_across_dp,
            ) = outer


# 模块级全局（同一进程内所有层共享；ModelRunner 每个 TP rank 一个进程，所以每个 rank 各有一份）：
#   _forward_context           当前这一步的 ForwardContext；导入时就放一个空的，所以 get_forward_context 永远不返回 None
#   _forward_kv_cache_context  只用来长期保存 kv_cache_data（KV 张量在启动时分配一次、之后不变），
#                              每步 set_forward_context 从这里复制引用；reset 不会清掉它
# 对比nano-vllm：同 nano-vllm 的 `_CONTEXT = Context()`；nano-vllm 的 KV 挂在模块上，不需要第二个全局
_forward_context: ForwardContext | None = ForwardContext()
_forward_kv_cache_context: ForwardContext | None = ForwardContext()

# Cached once at module import — CUDA availability does not change at
# runtime, so we don't pay torch.cuda.is_available() per set_forward_context().
# 【译】在模块导入时缓存一次——CUDA 是否可用在运行时不会变，所以不必每次 set_forward_context() 都付 torch.cuda.is_available() 的开销。
_CUDA_AVAILABLE: bool = torch.cuda.is_available()

# DP pad-row mask for consumers recorded into CUDA graphs (MegaMoE). A replay
# runs no Python, so each pass derives its mask on the device, on its first call,
# from buffers every step already refreshes -- never from a host count:
#   target pass: rows at or past `cu_seqlens_q[running_bs]` (padded requests
#     have length 0, so that entry is this rank's real token count);
#   draft pass (`[running_bs, q]` rows): rows at or past q times the real request
#     count the step's target pass recorded.
# A target pass records that count first thing, whatever else it decides, so a
# draft never reads an older step's; one that cannot tell (no cu_seqlens_q, a TBO
# micro-batch) records "no padding". The buffers are persistent because graphs,
# and the pieces of a piecewise graph, record their addresses.
# 【译】给录进 CUDA Graph 的使用方（MegaMoE）用的 DP pad-row 掩码。回放不执行 Python，所以每次 pass 都在第一次调用时
#   在设备上、用每步本来就会刷新的缓冲区推出掩码——绝不用主机上的计数：
#     目标 pass：位于或越过 `cu_seqlens_q[running_bs]` 的行（补出来的请求长度为 0，所以这一项就是本 rank 的真实 token 数）；
#     draft pass（`[running_bs, q]` 行）：位于或越过"q × 本步目标 pass 记录的真实请求数"的行。
#   目标 pass 不管做别的什么决定，第一件事都是记录这个数，所以 draft 永远不会读到更早一步的；判断不了的
#   （没有 cu_seqlens_q、TBO 微批）记录"没有 padding"。这些缓冲区是常驻的，因为图（以及分段图的各片段）记录的是它们的地址。
# 【分支】MegaMoE（fused_moe/flydsl_mega_experts.py、mori_v2_prepare_finalize.py，第7批）：DP 补齐的假行不该参与路由和通信，
#   可图里的 kernel 每次回放跑的都是 running_tokens 行，只能在 GPU 上自己算出哪些行是假的。
#   例：running_bs=16、真实 13 条、q=1 → cu_seqlens_q[16]=13 → 行 13~15 的掩码为 True
#   _NO_PADDING = 2^30：一个大到不可能达到的行号，"没有 padding"时阈值取它
_NO_PADDING = 1 << 30
_row_index_device: torch.Tensor | None = None
_pad_rows_device: torch.Tensor | None = None
_real_requests_device: torch.Tensor | None = None


def enable_pad_rows_device(num_rows: int, device: torch.device) -> None:
    """Allocate (or grow) the mask buffers. Call before any capture."""
    # 【译】分配（或扩大）掩码缓冲区。必须在任何录制之前调用。
    # 调用方：flydsl_mega_experts.py；_row_index_device 是 [num_rows, 1] 的 0..num_rows-1，_pad_rows_device 是结果掩码
    global _row_index_device, _pad_rows_device, _real_requests_device
    if _pad_rows_device is None or _pad_rows_device.shape[0] < num_rows:
        # int64: the draft threshold (requests x q, or _NO_PADDING x q) is a 0-d
        # int64 tensor, which would be cast to the row index's dtype.
        # 【译】用 int64：draft 的阈值（请求数 × q，或 _NO_PADDING × q）是一个 0 维 int64 张量，会被转换成行号的 dtype。
        _row_index_device = torch.arange(
            num_rows, dtype=torch.int64, device=device
        ).unsqueeze(1)
        _pad_rows_device = torch.zeros((num_rows, 1), dtype=torch.bool, device=device)
        _logger.info("DP pad-row mask enabled for %d rows", num_rows)
    if _real_requests_device is None:
        _real_requests_device = torch.full(
            (), _NO_PADDING, dtype=torch.int64, device=device
        )


def step_pad_rows(num_rows: int) -> torch.Tensor | None:
    """The `[num_rows, 1]` DP pad-row mask for this pass, or None when no row can
    be padding: masking is off, a TBO micro-batch, the rows are not the pass's
    `running_tokens` (a PCP shard, a sub-slice), or an eager pass has no padding.
    While capturing it always returns the mask, so each replay follows its step."""
    # 【译】本次 pass 的 `[num_rows, 1]` DP pad-row 掩码；没有任何行可能是 padding 时返回 None：掩码功能没开、
    #   在 TBO 微批里、行数不是本 pass 的 `running_tokens`（PCP 分片、子切片），或者 eager pass 本来就没有补齐。
    #   录制期间总是返回掩码，这样每次回放都跟着它那一步走。
    # 实现：pad_rows_done 记录本次 forward 已经算过的 (类型, …, 是否录制中) 键，同一 forward 里每种只在 GPU 上算一次
    from atom.utils.tbo.ubatching import tbo_active

    forward_context = get_forward_context()
    context = forward_context.context
    if context is None or _pad_rows_device is None:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    done = forward_context.pad_rows_done
    cu = getattr(forward_context.attn_metadata, "cu_seqlens_q", None)
    running_bs = context.running_bs
    tbo = tbo_active()
    if not context.is_draft and ("requests", capturing) not in done:
        done.add(("requests", capturing))
        if tbo or cu is None or not 0 < running_bs < cu.shape[0]:
            _real_requests_device.fill_(_NO_PADDING)
        else:
            _real_requests_device.copy_(
                (cu[1 : running_bs + 1] > cu[:running_bs]).sum()
            )
    if tbo or num_rows != context.running_tokens:
        return None
    if not capturing and context.scheduled_tokens >= num_rows:
        return None
    if _pad_rows_device.shape[0] < num_rows:
        return None
    pad_rows = _pad_rows_device[:num_rows]
    key = ("mask", context.is_draft, num_rows, capturing)
    if key not in done:
        rows = _row_index_device[:num_rows]
        if context.is_draft:
            if running_bs <= 0 or num_rows % running_bs:
                return None
            torch.ge(
                rows, _real_requests_device * (num_rows // running_bs), out=pad_rows
            )
        else:
            if cu is None or not 0 < running_bs < cu.shape[0]:
                return None
            torch.ge(rows, cu[running_bs], out=pad_rows)
        done.add(key)
    return pad_rows


# Thread-local storage for TBO dual-thread execution
# 【译】TBO 双线程执行用的线程局部存储
# 【分支】TBO（第9批）：两个微批在两个 Python 线程里交替跑模型，每个线程要看到自己微批的 ForwardContext；
#   utils/tbo/ubatching.py 在每个微批线程里设 _forward_context_local.ctx，utils/tbo/ubatch_wrapper.py 跑完后清掉。
#   Python 语法：threading.local() 的属性每个线程各有一份，互不可见

_forward_context_local = threading.local()


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # 【译】获取当前的 forward context。
    # 调用方：几乎所有层（注意力、MoE、LM head、drafter）。对比nano-vllm：= get_context()，只是多了线程局部的优先查找
    # Check thread-local first (used by TBO threads)
    # 【译】先查线程局部的（TBO 线程用）
    ctx = getattr(_forward_context_local, "ctx", None)
    if ctx is not None:
        return ctx

    # 正常情况下不会触发：_forward_context 在导入时和 reset 后都是一个空 ForwardContext，不是 None
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


@contextmanager
def side_stream(stream: torch.cuda.Stream | None):
    """Issue the block beside the main stream instead of on it.

    Yields `(issuing, joining)`: the stream the block runs on, and the one to
    wait on it afterwards, `None` when nothing forked.

    Forks only inside the capture loop, the one window where a side stream is
    both safe and useful: eager launches pile up across layers with nothing to
    drain them, and a replay runs no Python, inheriting the recorded layout.
    `in_hipgraph` covers that loop's warmup forward too, which matters because
    AITER caches a kernel's scratch per `(device, stream)` and rejects a first
    allocation made during capture.

    The gate lives here rather than at each caller so that two branches of one
    layer cannot drift into disagreeing about when forking is allowed. A caller
    whose feature is switched off hands `None` and takes the same path, so the
    switch needs no second branch anywhere.
    """
    # 【译】把这个块发射到主 stream 旁边，而不是主 stream 上。
    #
    #   yield `(issuing, joining)`：块运行所在的 stream，以及之后要等它的那条 stream；没有分叉时 joining 为 `None`。
    #
    #   只在录制循环内分叉——这是旁路 stream 既安全又有用的唯一窗口：eager 时发射会跨层堆积而没有东西去消化它们，
    #   回放则不跑 Python，沿用录下的布局。`in_hipgraph` 也覆盖录制循环里的那次预热 forward，这很重要，因为 AITER 按
    #   `(device, stream)` 缓存 kernel 的临时缓冲区，并拒绝在录制期间做第一次分配。
    #
    #   这个判断放在这里而不是每个调用方，是为了让同一层的两个分支不会在"何时允许分叉"上渐渐不一致。功能被关掉的调用方
    #   传 `None`，走同一条路径，所以这个开关在任何地方都不需要第二个分支。
    # 【分支】调用方：models/deepseek_v41/attention.py（compress / index 两条旁路 stream，第8批）
    #   分叉的写法：stream.wait_stream(main) 让旁路 stream 先等主 stream 上已排的活（数据依赖）；调用方拿到 joining 后，
    #   在用到旁路结果之前让它 wait_stream(旁路 stream)（汇合）
    context = get_forward_context()
    if stream is None or not context.in_hipgraph:
        yield context.main_stream, None
        return
    stream.wait_stream(context.main_stream)
    with torch.cuda.stream(stream):
        yield stream, context.main_stream


def _normalize_cudagraph_runtime_mode(mode: Any) -> CUDAGraphMode | None:
    """Normalize a frontend runtime mode to ATOM's concrete enum.

    Frontends own their graph dispatch and therefore use distinct enum
    classes.  Match by name rather than value so their enum layouts can evolve
    independently.  Composite configuration modes are deliberately rejected:
    a forward context must describe the concrete NONE/PIECEWISE/FULL decision
    for the current batch.
    """
    # 【译】把前端的运行时模式规范化成 ATOM 的具体枚举。
    #
    #   各前端自己负责图的分派，所以用的是各自不同的枚举类。按名字而不是按值匹配，这样它们的枚举布局可以各自演变。
    #   组合配置模式被刻意拒绝：forward context 必须描述当前 batch 具体的 NONE/PIECEWISE/FULL 决定。
    # 例：vLLM 的 CUDAGraphMode.PIECEWISE → ATOM 的 CUDAGraphMode.PIECEWISE；FULL_AND_PIECEWISE 这类组合值 → None
    name = mode if isinstance(mode, str) else getattr(mode, "name", None)
    if name not in {"NONE", "PIECEWISE", "FULL"}:
        return None
    return CUDAGraphMode[name]


def get_current_cudagraph_runtime_mode() -> CUDAGraphMode:
    """Return the concrete graph mode for the active model forward.

    In vLLM plugin mode graph capture/replay is owned by vLLM, so its forward
    context is authoritative.  Native ATOM records the same decision on its
    own ForwardContext.  An unavailable/unknown context is treated as NONE:
    eager dual-stream execution is valid, and some vLLM runners expose NONE
    while a whole-model FULL graph is being captured.  Replay does not execute
    this Python dispatcher.
    """
    # 【译】返回当前这次模型 forward 的具体图模式。
    #
    #   在 vLLM 插件模式下，图的录制/回放由 vLLM 负责，所以以它的 forward context 为准。原生 ATOM 在自己的 ForwardContext
    #   上记录同样的决定。拿不到 / 不认识的上下文按 NONE 处理：eager 的双 stream 执行是合法的，而且有些 vLLM runner
    #   在录制整模型 FULL 图时对外报告的是 NONE。回放不会执行这个 Python 分派函数。
    # 调用方：model_ops/module_dispatch_ops.py（PIECEWISE 时决定自定义算子怎么走，第5批）；原生路径读的就是
    #   run_model 写进 ForwardContext.cudagraph_runtime_mode 的值
    from atom.plugin import is_vllm

    if is_vllm():
        try:
            from vllm.forward_context import (
                get_forward_context as get_vllm_forward_context,
            )
            from vllm.forward_context import (
                is_forward_context_available,
            )

            if is_forward_context_available():
                mode = _normalize_cudagraph_runtime_mode(
                    get_vllm_forward_context().cudagraph_runtime_mode
                )
                if mode is not None:
                    return mode
        except (ImportError, AttributeError, AssertionError):
            pass

    mode = _normalize_cudagraph_runtime_mode(
        getattr(get_forward_context(), "cudagraph_runtime_mode", None)
    )
    return mode if mode is not None else CUDAGraphMode.NONE


# set_forward_context：把模块级全局换成本步新建的 ForwardContext
#   调用方：ModelRunner.prepare_inputs（每个真实的步一次；DP dummy 也走 forward，同样经过这里）、capture_cudagraph 的
#     各录制循环（in_hipgraph=True，每个尺寸一次）、spec_decode/drafter.py、插件（rtpllm / SGLang / vLLM bridge，第10批）
#     （TBO 微批不调它，而是用上面 get_forward_context 之前定义的线程局部上下文 _forward_context_local）
#   参数：
#     attn_metadata         builder 刚建好的 AttentionMetaData
#     atom_config           全局 Config（读 parallel_config、compilation_config.static_forward_context）
#     context               本步的 Context
#     num_tokens            本 rank 的 token 数（prepare_inputs 传 scheduled_tokens）；None 则不建 DPMetadata
#     num_tokens_across_dp  各 rank token 数的表（ForwardMode.sync 带回来的，或录图循环里 get_dp_padding 给的）
#     spec_decode_metadata / ubatch_slices / ub_*_tokens_across_dp   投机解码与 TBO（第7、9批）
#     in_hipgraph           是否在 CUDA Graph 录制循环里
# 对比nano-vllm：= set_context(...)，nano-vllm 在 prepare_prefill / prepare_decode 末尾调；ATOM 在 prepare_inputs 末尾调
def set_forward_context(
    attn_metadata: AttentionMetaData,
    atom_config: Config,
    context: Context,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    spec_decode_metadata: SpecDecodeMetadata | None = None,
    ubatch_slices: list[Any] | None = None,
    in_hipgraph: bool = False,
    ub_max_tokens_across_dp: tuple | None = None,
    ub_tokens_across_dp: tuple | None = None,
) -> None:
    global _forward_context
    # 上一行 global：函数里要给模块级变量重新赋值，必须先声明 global，否则赋值只会创建一个同名局部变量
    # DP>1 才建 DPMetadata。主路径递进了 decide 得到的表 → make 不再通信（见 DPMetadata 上方【科普】）
    dp_metadata: DPMetadata | None = None
    if atom_config.parallel_config.data_parallel_size > 1 and num_tokens is not None:
        dp_metadata = DPMetadata.make(
            atom_config.parallel_config,
            # 下一行是被注释掉的旧参数
            # attn_metadata,
            num_tokens or 0,
            num_tokens_across_dp,
        )

    # Mirrored here rather than at each caller: the two going out of step is how
    # an all2all comes to bound its receive buffer by one rank's count. Assigned
    # unconditionally, because the capture loop reuses one Context across
    # buckets and a set-only write would leave the last table behind.
    # 【译】在这里镜像，而不是在每个调用方：两者不同步，正是 all2all 最后用某一个 rank 的数去限定接收缓冲区的原因。
    #   无条件赋值，因为录制循环在多个尺寸之间复用同一个 Context，只在有值时才写就会留下上一个尺寸的表。
    # .tolist()：CPU 张量 → Python 列表（表在 CPU 上，不触发 GPU 同步）
    context.running_tokens_across_dp = (
        None if num_tokens_across_dp is None else tuple(num_tokens_across_dp.tolist())
    )

    # 每步新建一个对象，而不是在旧对象上改字段：pad_rows_done（空集合）、cudagraph_runtime_mode / batch_descriptor（None）
    #   这些"每次 forward 从零开始"的字段因此每步都是干净的
    #   kv_cache_data 从 _forward_kv_cache_context 复制引用；main_stream 取调用时的当前 stream（录图循环里就是录制用的 stream）；
    #   dp_stages 只有 V4.1 有界重放、且 sync 带了阶段表时才非 None
    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        kv_cache_data=_forward_kv_cache_context.kv_cache_data,
        context=context,
        dp_metadata=dp_metadata,
        spec_decode_metadata=spec_decode_metadata,
        ubatch_slices=ubatch_slices,
        ub_max_tokens_across_dp=ub_max_tokens_across_dp,
        ub_tokens_across_dp=ub_tokens_across_dp,
        main_stream=(torch.cuda.current_stream() if _CUDA_AVAILABLE else None),
        in_hipgraph=in_hipgraph,
        dp_stages=(
            None
            if dp_metadata is None or context.forward_mode is None
            else _bounded_replay_dp_stages(
                atom_config.parallel_config,
                context.forward_mode.dp_stage_report,
                context.forward_mode.sync,
            )
        ),
    )  # _forward_context.attn_metadata = attn_metadata
    # _forward_context.no_compile_layers = atom_config.compilation_config.static_forward_context
    # _forward_context = ForwardContext(no_compile_layers=atom_config.compilation_config.static_forward_context, attn_metadata=attn_metadata)
    # 上面两行是被注释掉的旧写法


# reset_forward_context：forward 结束后换回空的 ForwardContext（attn_metadata={}、context=None）
#   调用方：ModelRunner.forward 的两个出口（中间 chunk 提前返回 / postprocess 之后；DP dummy 也走 forward）、
#     RapidServeModelRunner 的 forward / prefill_forward
#   为什么要清：层是从全局取元数据的，下一步（或一次录图 / dummy）若漏了 set，读到 None 会立刻报错，
#     而不是悄悄用上一步的 slot_mapping 把 KV 写错位置。kv_cache_data 存在另一个全局里，不受影响
# 对比nano-vllm：= reset_context()
def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


# ---------------------------------------------------------------------------
# KV Connector global instances (lazy initialization)
# ---------------------------------------------------------------------------
# 【译】KV Connector 的全局实例（惰性初始化）
# 【分支】PD 分离 / KV offload 的 KV 传输连接器（第9批 kv_transfer/）。放在本文件只是因为它和 kv_cache_data 一样是
#   "进程内全局"：worker 侧每个 TP rank 一个，调度器侧一个

_logger = logging.getLogger("atom")

_global_kvconnector: Any | None = None
_global_kvconnector_scheduler: Any | None = None


def get_kvconnector(role: str = "worker", config: Config | None = None) -> Any:
    """Get or lazily initialize the global KV connector instance.

    The connector is role-dependent:
      - ``"worker"``: Returns a :class:`KVConnectorBase` (worker-side, per TP rank).
      - ``"scheduler"``: Returns a :class:`KVConnectorSchedulerBase` (scheduler-side).

    The concrete backend is selected by :class:`KVConnectorFactory` based on
    ``config.kv_transfer_config["kv_connector"]`` (default: ``"moriio"``).

    Args:
        role: Either ``"worker"`` or ``"scheduler"``.
        config: Engine config; required on first call to trigger initialization.

    Returns:
        The KV connector instance, or ``None`` if KV transfer is not configured.
    """
    # 【译】获取（或惰性初始化）全局 KV connector 实例。
    #
    #   connector 依角色而定：
    #     - ``"worker"``：返回 :class:`KVConnectorBase`（worker 侧，每个 TP rank 一个）。
    #     - ``"scheduler"``：返回 :class:`KVConnectorSchedulerBase`（调度器侧）。
    #
    #   具体后端由 :class:`KVConnectorFactory` 按 ``config.kv_transfer_config["kv_connector"]`` 选择（默认 ``"moriio"``）。
    #
    #   参数：
    #     role：``"worker"`` 或 ``"scheduler"``。
    #     config：引擎配置；第一次调用时必须提供，用来触发初始化。
    #
    #   返回：KV connector 实例；没有配置 KV 传输时返回 ``None``。
    # 调用方：worker 侧 ModelRunner（不带 config，取已建好的实例）；调度器侧 scheduler.py（role="scheduler"，每次都新建）
    # 注意：不带 config 调用时第一行 if 直接返回已有的 worker 实例（可能为 None）
    global _global_kvconnector, _global_kvconnector_scheduler

    if not (hasattr(config, "kv_transfer_config") and config.kv_transfer_config):
        return _global_kvconnector

    if role == "worker":
        from aiter.dist.parallel_state import get_tp_group

        try:
            tp_rank = get_tp_group().rank_in_group
        except Exception:  # noqa: BLE001 -- logged; any failure means "no group"
            _logger.warning(
                "get_tp_group() failed (dist not initialized?), returning None"
            )
            return None

        if _global_kvconnector is None:
            from atom.kv_transfer.disaggregation import KVConnectorFactory

            _global_kvconnector = KVConnectorFactory.create_connector(
                config, role="worker"
            )
            _logger.debug("Initialized global KVConnector at tp_rank %d", tp_rank)

    elif role == "scheduler":
        from atom.kv_transfer.disaggregation import KVConnectorFactory

        _global_kvconnector_scheduler = KVConnectorFactory.create_connector(
            config, role="scheduler"
        )
        _logger.debug("Initialized global KVConnectorScheduler")
        return _global_kvconnector_scheduler

    else:
        raise ValueError(f"Unknown KV connector role: {role!r}")

    return _global_kvconnector


def set_kv_cache_data(
    kv_cache_data: dict[int, KVCacheTensor],
    config: Config | None = None,
    transfer_tensors: Any = None,
    num_blocks: int | None = None,
) -> None:
    """Register KV cache data globally and with the KV connector if enabled.

    ``num_blocks`` is the scheduler-visible KV block count (the ID space used
    by request block tables). The offload connector needs it to byte-slice
    MLA's token-major latent cache, where tensor.shape[0] is the page-size-1
    physical row count rather than the scheduler block count.
    """
    # 【译】把 KV cache 数据登记到全局，若启用了 KV connector 也登记给它。
    #
    #   ``num_blocks`` 是调度器可见的 KV block 数（请求 block table 用的那个编号空间）。offload connector 需要它来
    #   按字节切分 MLA 的 token-major latent 缓存，那里 tensor.shape[0] 是 page-size-1 的物理行数，而不是调度器的 block 数。
    # 调用方：ModelRunner.allocate_kv_cache（启动时一次）、RapidServeModelRunner._bind_kv_cache_to_modules、
    #   rollout/memory_manager.py（RL 释放显存时传 {}，第10批）、rtpllm 插件
    # 对比nano-vllm：nano-vllm 在 allocate_kv_cache 里把 k_cache / v_cache 直接赋给每个 Attention 模块；ATOM 放进这个全局字典，
    #   注意力层在 forward 时按层名取
    if hasattr(config, "kv_transfer_config") and config.kv_transfer_config:
        connector = get_kvconnector(config=config)
        if connector is not None:
            connector.register_kv_caches(
                kv_cache_data, transfer_tensors, num_blocks=num_blocks
            )

    _forward_kv_cache_context.kv_cache_data = kv_cache_data
