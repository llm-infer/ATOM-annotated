# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Centralized environment variable definitions for ATOM.

All ATOM-specific environment variables are defined in the
``environment_variables`` dict below.  Access them via attribute syntax::

    from atom.utils import envs
    if envs.ATOM_PROFILER_MORE:
        ...

Values are evaluated lazily on first access via ``__getattr__``.  To add a
new variable, append an entry to ``environment_variables`` with a lambda that
reads ``os.getenv`` and returns the typed value.

Third-party / dependency env vars (NCCL, torch, HuggingFace, AITER, FLA) are
documented at the bottom of this file but NOT managed here.
"""

# 【译】ATOM环境变量的集中定义。
#   所有ATOM专用的环境变量都定义在下面的``environment_variables``字典里。通过属性语法访问它们：
#       示例：from atom.utils import envs，然后 if envs.ATOM_PROFILER_MORE: ...
#   值在第一次访问时通过``__getattr__``惰性求值。要新增一个变量，就在``environment_variables``里追加一项，
#   值是一个读取``os.getenv``并返回带类型值的lambda。
#
#   第三方 / 依赖库的环境变量（NCCL、torch、HuggingFace、AITER、FLA）在本文件底部有说明，但不在这里管理。
#
# 本文件是ATOM的"环境变量总表"：两百多个变量，绝大多数是kernel路径选择、性能优化开关和调试开关
#   命令行参数（--max-num-seqs等，见atom/model_engine/arg_utils.py）管"引擎怎么配"；环境变量多用于"走哪条kernel路径、开不开某个优化"
#   大多数默认值就是生产推荐值，平时不用设；排查精度/性能问题时才按需打开或关闭，做A/B对比
# 机制：environment_variables是"变量名 -> 无参lambda"的字典；模块级__getattr__（PEP 562）在访问envs.XXX时调用对应lambda
#   注意并没有缓存：每次访问都重新读一次os.environ。热路径代码通常在模块导入时就把值存进模块级常量
#   （例如models/deepseek_v2.py顶部的ENABLE_DS_QKNORM_FUSION），所以这类开关必须在启动服务之前设置
# 布尔变量的写法不统一：多数是os.getenv(X, "0") == "1"，只有字符串"1"算开（"true"/"on"都不算）；
#   少数用_flag_env或.lower() in {...}，接受更多写法。设置时用1/0最稳妥
# 文件里原有的`# --- xxx ---`分节标题只反映大致位置，很多变量后来被加在不相关的分节里。
#   下面按功能重新分组；每个变量上方的中文注释都以【组名】开头，可以按组名搜索：
#   （组名后括号里的"第N批"指ANNOTATION_PLAN.md的阅读批次，即这组开关的读取方在哪一批里讲；"【分支】…见第N批"同义）
#
#   【服务】进程、API前端、日志、指标、GC、追踪（第2批、第10批）
#     ATOM_PROCESS_NAME_PREFIX、ATOM_SHUTDOWN_TIMEOUT_S、ATOM_LOG_MORE、
#     ATOM_METRICS_UPDATE_INTERVAL_S、ATOM_ENABLE_METRICS_DEVICE_TIMER、ATOM_GC_FREEZE / GC_DEBUG / GC_THRESHOLD、
#     ATOM_DETOKENIZER_DELTA_REUSE / AUDIT_EVERY、
#     ATOM_MESH_BUILD / MESH_FEATURES、USE_ATOMESH_ENTRYPOINTS
#   【调度】prefill合批与DP间prefill对齐（第3批）
#     ATOM_ENABLE_PREFILL_DELAYER、ATOM_PREFILL_DELAYER_*（7个）、ATOM_PREFILL_DECODE_INTERVAL、
#     ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT / SLACK / MIN_TOKENS（DP attention下的prefill成本均衡）
#   【前缀缓存】KV block账本、前缀缓存、KV事件（第4批）
#     ATOM_PREFIX_CACHE_POLICY / PROTECTED_RATIO、ATOM_PREFIX_WALK_MEMO、ATOM_PREFIX_FAST_WALK、ATOM_FRONTEND_PREFIX_HASH、
#     ATOM_KV_EVENTS_*（8个）
#   【状态缓存】线性注意力（GDN/KDA）的递归状态：状态checkpoint、ReplaySSM、状态精度（第4批、第8批）
#     ATOM_STATE_CHECKPOINT_DEMAND / END_MARGIN / INTERVAL_TOKENS、ATOM_GDN_STATE_MIDSTEP、ATOM_ENABLE_REPLAYSSM、
#     ATOM_REPLAYSSM_CACHE_LEN / ROUTE、ATOM_GDN_SSM_DTYPE、ATOM_ENABLE_GDN_DECODE_LOSSY_FAST、ATOM_DSV4_STATE_REPLAY
#   【加载】权重加载与在线量化（第5批）
#     ATOM_DISABLE_MMAP、ATOM_LOADER_NUM_THREADS / PREFETCH / PREFETCH_THREADS / PREFETCH_BLOCK_MB / FADVISE / STRICT_COVERAGE、
#     ATOM_ONLINE_QUANT_STREAMING / _THREADS / _HOST_STAGING、ATOM_REQUIRES_GRAD
#   【图/编译】CPU->GPU元数据传输（第5批）
#     ATOM_H2D_BACKEND
#   【GEMM】Linear/GEMM kernel与量化权重布局（第6批）
#     ATOM_USE_TRITON_GEMM、ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE、ATOM_GROUP32_WEIGHT_PRESHUFFLE、
#     ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE、ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM
#   【融合】把几个小算子合成一个kernel的开关，以及RMSNorm kernel选项（第6批、第7批）
#     ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION、ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS、ATOM_ENABLE_DS_*（4个）、
#     ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION、ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_*（2个）、
#     ATOM_FUSED_COMPRESS_USE_FLYDSL、ATOM_USE_FLYDSL_GATHER_KV_B_PROJ、ATOM_USE_MODEL_SENSITIVE_RMSNORM
#   【注意力】注意力后端与kernel选择（第6、7、8批）
#     MHA：ATOM_USE_UNIFIED_ATTN、ATOM_FORCE_ATTN_TRITON、ATOM_PA_FLYDSL、ATOM_AITER_FP8_PREFILL_ATTN
#     MLA：ATOM_USE_TRITON_MLA、ATOM_USE_TRITON_MLA_SHUFFLE_KV、ATOM_MLA_KV_SHUFFLED、ATOM_MLA_PAGE_SIZE、
#          ATOM_USE_TRITON_MXFP4_BMM、ATOM_USE_FLYDSL_FP8_PREFILL_ATTN
#     DeepSeek-V4：ATOM_FORCE_V4_PREFILL_OPUS、ATOM_USE_V4_PREFILL_ASM_FOR_DECODE、ATOM_V4_HCA_PERSIST / _MIN_ROWS、
#          ATOM_V4_BACKEND / _LAYERS（V4新旧注意力后端切换）
#   【索引器】DSA稀疏注意力的indexer（第7批、第8批）
#     ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB、ATOM_ENABLE_GLM_FUSED_INDEXER、
#     ATOM_GLM5_KPOOL、ATOM_M3_INDEXER_CP
#   【MoE】MoE kernel、专家并行通信、EPLB（第7批）
#     ATOM_USE_TRITON_MOE / _DECODE / _A4W4、ATOM_MOE_GU_ITLV、ATOM_FUSE_SHARED_EXPERT、
#     ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD、ATOM_DUAL_STREAM_PIECEWISE、ATOM_DISABLE_MORI_EP、ATOM_MORI_V2、ATOM_MORI_FP4_DISPATCH、ATOM_MORI_COMBINE_QUANT、
#     ATOM_MEGA_*（6个，gfx1250上的MegaMoE）、ATOM_EPLB_MAX_REBALANCES
#   【投机】投机解码MTP / EAGLE3 / DSpark（第7批）
#     ATOM_ENABLE_RELAXED_MTP、ATOM_DRAFT_CUDAGRAPH、ATOM_DSPARK_FUSED_CTX_KV、ATOM_DSPARK_FUSED_MARKOV_SAMPLE、
#     ATOM_DSPARK_DISABLE_COMPILE、ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL、ATOM_DP_DRAFT_ARGMAX / _MAX_ROWS
#   【并行】DP / TP通信 / PP / DCP / PCP / TBO / NUMA（第9批）
#     DP：ATOM_DP_RANK / RANK_LOCAL / SIZE / SIZE_LOCAL / MASTER_IP / MASTER_PORT / BASE_PORT、ATOM_DP_LB_REQ_EQUIV、
#         ATOM_DP_SESSION_AFFINITY、ATOM_DP_LM_HEAD_MODE
#     TP通信：ATOM_USE_CUSTOM_ALL_GATHER、ATOM_REPLICATE_VOCAB_EMBED
#     PP：ATOM_PP_SEND_ALLGATHER、ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR
#     DCP / PCP：ATOM_DCP_PREFILL_QREP、ATOM_DCP_SPARSE_PREFILL_NONPS、ATOM_PCP_MOE_MERGE
#     TBO：ATOM_TBO_PREFILL_TOKEN_SPLIT、ATOM_TBO_PREFILL_MIN_TOKENS、ATOM_TBO_TP_AR_MODE
#     NUMA：ATOM_NUMA_BIND、ATOM_AUTO_NUMA_BIND、ATOM_NUMA_NODE、ATOM_CRASH_ON_NUMA_BIND_FAILURE
#   【Mono】Mono融合decode（第9批）
#     ATOM_MONO_ENABLE、ATOM_GLM52_FUSED_INDEXER、ATOM_GLM52_INDEXER_CP（调试用的ATOM_MONO_CHECK等见【调试】）
#   【PD/offload】PD分离与KV offload（第9批）
#     ATOM_MOONCAKE_MATCHED_RAILS、ATOM_PD_MLA_STAGING / MLA_LANDING / FAILED_WRITE_HOLD_S、
#     ATOM_KV_OFFLOAD / _EXTRA_CONFIG、OFFLOAD_*（不带ATOM_前缀）、LMCACHE_MP_TRANSFER_MODE
#   【模型】只对某个模型生效的开关（第8批）
#     DeepSeek-V4/V4.1：ATOM_MHC_USE_BF16、ATOM_DSV41_SIDE_STREAMS、ATOM_ENGRAM_*（4个）
#     GLM-5.3：ATOM_GLM5_FORCE_DENSE_MLA、ATOM_GLM5_DISABLE_FUSED_MHC
#   【插件】vLLM / SGLang插件模式（第10批）
#     ATOM_DISABLE_VLLM_PLUGIN、ATOM_V4_CG_INSTR_DIR、ATOM_USE_GLUON_PA_DECODE、ATOM_USE_FLYDSL_GDR、ATOM_SGLANG_*（3个）
#   【调试】profiling、dump、校验（"出了问题怎么查"）
#     ATOM_TORCH_PROFILER_DIR、ATOM_PROFILER_*（5个）、ATOM_ENABLE_DETAILED_ANNOTATION、ATOM_RTL_TRACE_DIR、
#     ATOM_FWD_DUMP_*（5个）、ATOM_WEIGHT_DUMP_*（3个）、ATOM_DEBUG_TOPK / _PATH、ATOM_MONO_CHECK / TRACE / TIMELINE / DEBUG、
#     ATOM_STATE_CHECKPOINT_CHECK_BLOCKS、OFFLOAD_PROFILE
# 对比nano-vllm：nano-vllm没有环境变量，所有配置都在Config里；ATOM要在不同GPU架构、不同模型之间切换大量kernel路径，才有这张表
# 怎么用（补充上面的机制说明）：
#   设置：在启动命令前加，例如 `ATOM_MONO_ENABLE=0 AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server ...`
#   读取：代码里写 `from atom.utils import envs` 然后 `envs.ATOM_MONO_ENABLE`，拿到的是lambda转换后的Python值（bool/int/str...）
#     而不是原始字符串；`envs.is_set("ATOM_X")`（定义在文件末尾）用来判断"用户到底有没有显式设置"
#   多进程：ATOM用spawn启动EngineCore / ModelRunner等子进程，子进程继承父进程的环境变量，所以只要在启动命令里设一次即可；
#     反过来，服务起来以后再改os.environ，已在模块导入时读过的开关不会变
#   每个变量的权威说明表在 docs/environment_variables.md（含类型、默认值、适用条件），本文件的注释与之互为补充
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom")


# ---- 下面几个_xxx_env是解析辅助函数：统一处理"未设置 / 空字符串 / 非法值"，各自的容错策略不同 ----
# 正浮点数：非法、非有限或<=0时打警告并回退默认值（不报错）
def _positive_float_env(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
        if math.isfinite(value) and value > 0:
            return value
    except ValueError:
        pass
    logger.warning(
        "Invalid %s=%r: expected a finite positive number; using default %s",
        name,
        raw_value,
        default,
    )
    return float(default)


# 布尔开关：去掉首尾空白、转小写；空字符串和0/false/no/off为False，其他都为True
def _flag_env(name: str, default: str = "0") -> bool:
    # Stripped, and empty reads as off: `VAR=` is how a shell script clears a
    # flag inline, and a bare membership test reads the empty string as ON --
    # the opposite of what the operator wrote. `VAR="off "` did the same.
    # 【译】先strip，并把空值读作关闭：`VAR=`是shell脚本里内联清除一个开关的写法，而简单的成员判断会把空字符串读成"开"——
    #   与操作者写的意思正好相反。`VAR="off "`也有同样的问题。
    raw = os.getenv(name, default).strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


# profiler子选项：自身设置了就以自身为准，没设置（或为空）时跟随ATOM_PROFILER_MORE
def _profiler_detail_env(name: str) -> bool:
    # Unset or empty falls back to ATOM_PROFILER_MORE, so the per-option flag
    # always wins when given and ATOM_PROFILER_MORE=1 alone still enables all.
    # 【译】未设置或为空时退回ATOM_PROFILER_MORE，所以单独的选项开关一旦给出总是优先，而只设ATOM_PROFILER_MORE=1仍会全部开启。
    raw = os.getenv(name) or os.getenv("ATOM_PROFILER_MORE", "0")
    return raw == "1"


# 可选整数：未设置/空返回None；设置了就必须是整数（且>=min_value），否则直接抛ValueError
def _optional_int_env(name: str, *, min_value: int | None = None) -> int | None:
    """Unset or empty reads as None; anything else must be an integer."""
    # 【译】未设置或为空读作None；其他值必须是整数。
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


# 整数（严格版）：未设置/空用默认值，非法值抛ValueError
def _int_env(name: str, default: int) -> int:
    """Unset or empty reads as ``default``; anything else must be an integer."""
    # 【译】未设置或为空读作``default``；其他值必须是整数。
    value = _optional_int_env(name)
    return default if value is None else value


# 整数（宽松版）：非法值只打警告并回退默认值
def _int_env_or_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r: expected an integer; using default %d",
            name,
            raw,
            default,
        )
        return default


# 非负整数（宽松版）：非法值或负数都打警告并回退默认值
def _nonnegative_int_env(name: str, default: int) -> int:
    value = _int_env_or_default(name, default)
    if value < 0:
        logger.warning(
            "Invalid %s=%r: expected a nonnegative integer; using default %d",
            name,
            value,
            default,
        )
        return default
    return value


# 有限浮点数（严格版）：非法、非有限、或不满足非负/正数要求时抛ValueError
def _finite_float_env(name: str, default: float, *, allow_zero: bool) -> float:
    """Unset or empty reads as ``default``; a set value must be a valid float."""
    # 【译】未设置或为空读作``default``；设置了就必须是合法的浮点数。
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


# 变量名 -> 无参lambda；lambda里读os.getenv并转换成需要的类型（bool/int/float/str/tuple/None）
environment_variables: dict[str, Callable[[], Any]] = {
    # 科普·forward元数据H2D：每一步decode/prefill前，CPU上的调度器要算出一批小张量（positions、slot_mapping、block_tables、
    #   各序列长度等，nano-vllm里在ModelRunner.prepare_*里构建）再拷到GPU。几十个小张量各拷一次，每次都有固定开销；
    #   "packed"把它们先拼进一块连续的pinned内存，一次H2D拷完，再由一个Triton kernel把各段scatter到各自固定的GPU地址。
    #   实现见atom/utils/h2d.py（发布/检查协议）、utils/packed_h2d.py（打包拷贝）
    # 【图/编译】ATOM_H2D_BACKEND：每步forward的元数据怎么从CPU传到GPU，"packed"（默认）或"direct"，其他值启动时报错
    #   读取方：model_runner.py；机制见utils/h2d.py（第5批）
    # Forward metadata transport: direct or packed. Both keep source checks.
    # Single-member groups and strided bindings retain direct copies.
    # 【译】forward元数据的传输方式：direct或packed。两者都保留对数据源的检查。
    #   单成员的组和带步长的绑定仍然保持直接拷贝。
    "ATOM_H2D_BACKEND": lambda: os.getenv("ATOM_H2D_BACKEND", "packed"),
    # 科普·PD分离（Prefill/Decode disaggregation）：prefill是算力密集、decode是访存密集，放在同一组GPU上会互相拖累；
    #   PD分离让一组GPU（P节点）只做prefill，算完把KV cache通过RDMA网络传给另一组GPU（D节点）继续decode。
    #   Mooncake / MoRI-IO是负责这次KV传输的传输引擎；HCA即RDMA网卡（Host Channel Adapter；
    #   与DeepSeek-V4的"HCA压缩注意力层"同名不同义，后者见config.py __post_init__第12步V4部分的【科普】）。实现见atom/kv_transfer/disaggregation/（mooncake/、moriio/）
    # 【PD/offload】ATOM_MOONCAKE_MATCHED_RAILS：Mooncake传输引擎使用哪些RDMA网卡（HCA）；空=不启用；读取方：mooncake_connector.py
    # Opt-in single-HCA engine pool: "auto" or explicit comma-separated HCAs.
    # 【译】可选的单HCA引擎池："auto"，或显式给出逗号分隔的HCA列表。
    "ATOM_MOONCAKE_MATCHED_RAILS": lambda: os.getenv("ATOM_MOONCAKE_MATCHED_RAILS", ""),
    # 科普·MLA（Multi-head Latent Attention，DeepSeek-V2/V3、Kimi、GLM-5等使用）：标准MHA每个token要缓存2*头数*head_dim个数，
    #   MLA把K/V先压成一个所有头共享的低维latent向量（DeepSeek是512维 + 64维RoPE部分 = 576维）只缓存它，计算时再用
    #   kv_b_proj展开（或把展开矩阵"吸收"进q）。KV cache因此小一个数量级。实现见atom/model_ops/attention_mla.py、attentions/aiter_mla.py
    #   （带图和DeepSeek-V3数字的版本见arg_utils.py EngineArgs.tensor_parallel_size下方的【科普】MLA）
    # 科普·DCP（Decode Context Parallel）：长上下文decode时，把同一条序列的KV cache按token分散存到DCP组的多张卡上，
    #   每张卡只对自己那份KV算部分注意力，再用log-sum-exp合并结果；省的是每卡KV显存。实现见atom/distributed/dcp_utils.py、model_ops/dcp_ops
    #   （token分布图和LSE合并的数字例子见arg_utils.py EngineArgs.decode_context_parallel_size上方的【科普】）
    # 【PD/offload】ATOM_PD_MLA_STAGING：prefill端把某个DCP decode rank需要的MLA KV先聚到GPU暂存槽，再按整页发送（默认开）
    # Prefill (`kv_producer`, plain `mooncake`) only: gather the MLA tokens a
    # DCP decode rank owns into a GPU staging slot and send whole destination
    # pages instead of one RDMA descriptor per token. 0 = per-token path.
    # 【译】仅prefill端（`kv_producer`，普通`mooncake`）：把某个DCP decode rank拥有的MLA token收集到一个GPU暂存槽里，
    #   按整个目标页发送，而不是每个token一个RDMA描述符。0 = 逐token路径。
    "ATOM_PD_MLA_STAGING": lambda: _flag_env("ATOM_PD_MLA_STAGING", "1"),
    # 【PD/offload】ATOM_PD_MLA_LANDING：decode端（DCP>1）提供GPU"着陆池"，prefill把一个rank的MLA行打包写进一个槽（默认开）
    # Decode (`kv_consumer`, DCP > 1) only: give prefill stages partitions of a
    # GPU landing pool, first come, first served (see mla_landing.py). A stage
    # writes a rank's MLA rows packed into one slot with one RDMA descriptor,
    # the decode side scatters them into the paged KV and returns the slot.
    # 0 = staged per-page writes only.
    # 【译】仅decode端（`kv_consumer`，DCP > 1）：把GPU着陆池的分区按先到先得分给各prefill stage（见mla_landing.py）。
    #   一个stage把某个rank的MLA行打包写进一个槽，只用一个RDMA描述符；decode端再把它们scatter进分页KV，然后归还这个槽。
    #   0 = 只用分页暂存写入。
    "ATOM_PD_MLA_LANDING": lambda: _flag_env("ATOM_PD_MLA_LANDING", "1"),
    # 【PD/offload】ATOM_PD_FAILED_WRITE_HOLD_S：传输失败后，相关KV页 / 着陆槽再保留多少秒才释放（默认60），防止迟到的写入踩到已复用的内存
    # Mooncake P/D: how long a write Mooncake gave up on may still land. A
    # decode holds a failed receive's KV pages this long before reporting the
    # failure; a prefill keeps a landing slot whose write failed out of use
    # this long. 0 = release at once.
    # 【译】Mooncake的P/D：Mooncake放弃的一次写入，还可能在多长时间内落地。decode端在报告失败前，
    #   把失败接收的KV页保留这么久；prefill端把写入失败的着陆槽停用这么久。0 = 立即释放。
    "ATOM_PD_FAILED_WRITE_HOLD_S": lambda: float(
        os.getenv("ATOM_PD_FAILED_WRITE_HOLD_S", "60")
    ),
    # 科普·前缀缓存：和nano-vllm一样，KV cache按固定大小的block分页管理，每个满block按"前一个block的hash + 本block的token"算hash；
    #   新请求的prompt前缀如果和以前某个请求相同（同一个system prompt、多轮对话的历史），就直接复用那些block，跳过这部分prefill。
    #   空闲但仍保留内容的block组成可淘汰池，下面的POLICY决定先淘汰谁。实现见atom/model_engine/block_manager.py
    # 【前缀缓存】ATOM_PREFIX_CACHE_POLICY：空闲KV block的淘汰策略，"lru"（默认）或"slru"；读取方：block_manager.py -> block_pool.py
    #   slru（分段LRU）：被前缀命中复用过的block进"受保护段"，不会被一次性的长prefill扫描冲掉
    # Protect reused KV prefixes from one-off prefill scans. Opt-in.
    # 【译】保护被复用的KV前缀，不被一次性的prefill扫描冲掉。需主动开启。
    "ATOM_PREFIX_CACHE_POLICY": lambda: os.getenv("ATOM_PREFIX_CACHE_POLICY", "lru"),
    # 【前缀缓存】ATOM_PREFIX_CACHE_PROTECTED_RATIO：slru下受保护段最多占全部block的比例（默认0.5，必须在(0, 1)之间）
    "ATOM_PREFIX_CACHE_PROTECTED_RATIO": lambda: float(
        os.getenv("ATOM_PREFIX_CACHE_PROTECTED_RATIO", "0.5")
    ),
    # 背景：DeepSeek-V4的注意力有滑动窗口和"压缩器"（把每compress_ratio个相邻token的KV压成一条），它们的中间状态不是逐token存放的，
    #   所以前缀命中只复用KV block还不够，还要恢复这些状态；下面这个开关提供"只重算一小段感受野"的恢复方式（状态checkpoint的概念见下文"科普·线性注意力 / 递归状态"）
    # 【状态缓存】ATOM_DSV4_STATE_REPLAY：DeepSeek-V4前缀命中时用"有界重放"重建压缩状态（默认关）；读取方：config.py、deepseek_v4_replay.py
    # DeepSeek-V4 bounded replay: when the compressed history hits further than
    # the nearest state checkpoint by more than the window's receptive field,
    # keep the whole hit and recompute only that field (W + (L-1)(W-1) tokens)
    # to rebuild every layer's window and compressor state exactly.
    # 【译】DeepSeek-V4有界重放：当压缩历史的命中位置比最近的状态checkpoint远出窗口的感受野以上时，
    #   保留整个命中，只重算这段感受野（W + (L-1)(W-1)个token），从而精确重建每一层的窗口状态和压缩器状态。
    "ATOM_DSV4_STATE_REPLAY": lambda: os.getenv("ATOM_DSV4_STATE_REPLAY", "0").lower()
    in {"1", "true", "yes", "on"},
    # 科普·KV offload：GPU显存放不下的KV cache（例如很久以前的对话前缀）下放到CPU内存 / 本地盘 / 远端存储，下次前缀命中时再加载回GPU，
    #   相当于给前缀缓存加了一层更大更慢的"二级缓存"。LMCache和Mooncake Store是两个外部存储实现。实现见atom/kv_transfer/offload/
    # 【PD/offload】ATOM_KV_OFFLOAD：不写--kv-transfer-config也能开启KV offload；读取方：kv_transfer/offload/__init__.py、arg_utils.py
    # KV offload without --kv-transfer-config: "lmcache" (in-process LMCache),
    # "lmcache_mp" (standalone `lmcache server`) or "mooncake_store" (a Mooncake
    # Store, no LMCache). Unset = off.
    # 【译】不用--kv-transfer-config的KV offload："lmcache"（进程内LMCache）、"lmcache_mp"（独立的`lmcache server`）
    #   或"mooncake_store"（Mooncake Store，不经过LMCache）。未设置 = 关闭。
    "ATOM_KV_OFFLOAD": lambda: os.getenv("ATOM_KV_OFFLOAD", ""),
    # 【PD/offload】ATOM_KV_OFFLOAD_EXTRA_CONFIG：上面那个connector的额外配置（JSON字符串）
    # JSON object for that connector's kv_connector_extra_config, e.g.
    # {"lmcache.chunk_size": 256}, {"lmcache.mp.port": 5556} or
    # {"mooncake_store.master": "10.0.0.1:26051", ...}.
    # 【译】那个connector的kv_connector_extra_config，是一个JSON对象，例如
    #   {"lmcache.chunk_size": 256}、{"lmcache.mp.port": 5556}或{"mooncake_store.master": "10.0.0.1:26051", ...}。
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG": lambda: os.getenv(
        "ATOM_KV_OFFLOAD_EXTRA_CONFIG", ""
    ),
    # --- Data Parallelism ---
    # 【译】--- 数据并行 ---
    # 科普·DP与DP attention：DP（数据并行）= 起多份完整的引擎副本，各自服务不同请求，CoreManager按负载把请求分给某个DP rank。
    #   对MoE大模型常用"DP attention"：注意力部分每个rank只算自己的请求、存自己的KV（DP），而MoE部分所有rank合起来做专家并行（EP），
    #   因此所有rank必须每步同步一起forward（没活的rank也要跑一个空/填充batch），这也是下面prefill对齐、填充行等开关存在的原因
    # 【并行】ATOM_DP_RANK / ATOM_DP_RANK_LOCAL / ATOM_DP_SIZE / ATOM_DP_SIZE_LOCAL：本进程的DP rank（全局 / 本机）和DP规模（全局 / 本机）
    #   ATOM_DP_MASTER_IP / ATOM_DP_MASTER_PORT：DP进程组rendezvous的主节点地址和端口
    #   config.py的ParallelConfig.__post_init__：这些变量显式设置了才覆盖对应字段，否则以程序里的配置为准
    "ATOM_DP_RANK": lambda: int(os.getenv("ATOM_DP_RANK", "0")),
    "ATOM_DP_RANK_LOCAL": lambda: int(os.getenv("ATOM_DP_RANK_LOCAL", "0")),
    "ATOM_DP_SIZE": lambda: int(os.getenv("ATOM_DP_SIZE", "1")),
    "ATOM_DP_SIZE_LOCAL": lambda: int(os.getenv("ATOM_DP_SIZE_LOCAL", "1")),
    "ATOM_DP_MASTER_IP": lambda: os.getenv("ATOM_DP_MASTER_IP", "127.0.0.1"),
    "ATOM_DP_MASTER_PORT": lambda: int(os.getenv("ATOM_DP_MASTER_PORT", "29500")),
    # 【并行】ATOM_DP_BASE_PORT：DP rendezvous的基准端口（0=不指定）；读取方：config.py、engine_core_mgr.py
    # Rendezvous base port; set per role when prefill/decode share a node.
    # 【译】rendezvous的基准端口；prefill/decode共用一个节点时按角色分别设置。
    "ATOM_DP_BASE_PORT": lambda: int(os.getenv("ATOM_DP_BASE_PORT", "0")),
    # 【并行】ATOM_DP_LB_REQ_EQUIV："least_tokens" DP负载均衡里，一个在途请求折合多少个token（默认512）
    # Token-equivalent cost of one in-flight request for the "least_tokens" DP
    # load-balance strategy. The per-rank load score is
    #   sum(prompt_tokens) + ATOM_DP_LB_REQ_EQUIV * num_in_flight_requests
    # so a larger value biases routing toward request-count balance (decode
    # pressure) and a smaller value toward prompt-token balance (prefill
    # pressure). See engine_core_mgr.CoreManager._select_dp_rank_locked.
    # 【译】"least_tokens" DP负载均衡策略中，一个在途请求等价的token成本。每个rank的负载分数计算如下：
    #     负载分数 = sum(prompt_tokens) + ATOM_DP_LB_REQ_EQUIV * num_in_flight_requests
    #   所以值越大，路由越偏向按请求数均衡（decode压力）；值越小，越偏向按prompt token数均衡（prefill压力）。
    #   见engine_core_mgr.CoreManager._select_dp_rank_locked。
    "ATOM_DP_LB_REQ_EQUIV": lambda: int(os.getenv("ATOM_DP_LB_REQ_EQUIV", "512")),
    # 【并行】ATOM_DP_SESSION_AFFINITY：同一个agent会话的请求固定发往同一个DP rank，以复用该rank上的前缀缓存（默认关）
    # Place a new agent session on the lightest DP rank, then keep every later
    # request on that immutable cache owner. Existing sessions never spill;
    # child correlation ids are independently load-placed rather than
    # inheriting their parent's owner.
    # 【译】把新的agent会话放到最轻的DP rank上，之后该会话的每个请求都留在这个不可变的缓存属主上。
    #   已有会话永不溢出到别的rank；子correlation id独立按负载放置，而不是继承父会话的属主。
    "ATOM_DP_SESSION_AFFINITY": lambda: os.getenv(
        "ATOM_DP_SESSION_AFFINITY", "0"
    ).lower()
    in {"1", "true", "yes", "on"},
    # --- DP prefill cost balancing (DP attention with more than one rank) ---
    # 【译】--- DP prefill成本均衡（多于一个rank的DP attention）---
    # 【调度】ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT：DP attention下各rank的prefill成本均衡，成本公式里的每token固定权重；读取方：scheduler.py
    # Under DP attention every rank prefills in the same lockstep step and the
    # step lasts as long as the slowest rank. With the same token budget, a rank
    # whose chunk attends long contexts is that straggler. The ranks exchange
    # the estimated cost of their chunk on the PrefillDelayer's per-tick
    # all_reduce (cost = new tokens x (TOKEN_WEIGHT + prior context + new / 2);
    # estimated only by ranks with prefill work) and on a prefill step aim at
    # min(costliest, mean x (1 + SLACK)): a rank above it takes fewer tokens
    # (never fewer than MIN_TOKENS).
    # 【译】DP attention下，每个rank在同一个同步step里一起prefill，step的时长取决于最慢的rank。
    #   在相同token预算下，chunk要attend长上下文的rank就是那个拖后腿的。各rank在PrefillDelayer每个tick的all_reduce上
    #   交换各自chunk的估计成本（成本 = 新token数 x (TOKEN_WEIGHT + 已有上下文 + 新token数 / 2)；只有有prefill工作的rank才估计），
    #   并在prefill step上以min(最贵的成本, 平均值 x (1 + SLACK))为目标：高于它的rank少拿一些token（但不少于MIN_TOKENS）。
    "ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT": lambda: _int_env(
        "ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", 360000
    ),
    # 【调度】ATOM_DP_PREFILL_BALANCE_SLACK：允许超出平均成本的比例（默认0.1）
    "ATOM_DP_PREFILL_BALANCE_SLACK": lambda: float(
        os.getenv("ATOM_DP_PREFILL_BALANCE_SLACK", "0.1")
    ),
    # 【调度】ATOM_DP_PREFILL_BALANCE_MIN_TOKENS：被削减的rank至少保留的token数（默认4096）
    "ATOM_DP_PREFILL_BALANCE_MIN_TOKENS": lambda: _int_env(
        "ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", 4096
    ),
    # 【服务】ATOM_PROCESS_NAME_PREFIX：进程标题前缀（默认"ATOM"），见utils/__init__.py的set_process_title
    # Prefix for process titles set via set_process_title (shown in ps/top/rocm-smi)
    # 【译】通过set_process_title设置的进程标题的前缀（会显示在ps/top/rocm-smi里）
    "ATOM_PROCESS_NAME_PREFIX": lambda: os.getenv("ATOM_PROCESS_NAME_PREFIX", "ATOM"),
    # 科普·插件模式：ATOM除了自己的openai_server，还能作为vLLM / SGLang的"模型后端插件"运行——调度、API由vLLM/SGLang负责，
    #   模型forward和kernel用ATOM的实现。代码在atom/plugin/（vllm/、sglang/）。下面ATOM_SGLANG_*只在SGLang插件模式下生效
    # 科普·PCP（Prefill Context Parallel）：把一个很长prompt的token切成几段分给多张卡同时prefill（每卡算1/W的token），
    #   注意力时再交换K/V，从而缩短超长prompt的TTFT。实现见atom/distributed/pcp_utils.py
    # 【插件】ATOM_SGLANG_PCP_SIZE：SGLang插件里prefill上下文并行（PCP）的规模；读取方：plugin/config.py
    # SGLang's GLM-5.2 and DeepSeek V4 prefill CP paths still force
    # attention TP size to 1.
    # ATOM remaps the SGLang world into internal TP x PCP groups.
    # 0 means unset.
    # 【译】SGLang的GLM-5.2和DeepSeek V4 prefill CP路径仍然强制attention TP size为1。
    #   ATOM把SGLang的world重新映射成内部的TP x PCP组。
    #   0表示未设置。
    "ATOM_SGLANG_PCP_SIZE": lambda: int(os.getenv("ATOM_SGLANG_PCP_SIZE", "0") or "0"),
    # 科普·torch.compile与CUDA Graph：torch.compile（Dynamo抓图 + Inductor生成融合kernel）减少Python开销和小算子；
    #   CUDA Graph把一整串kernel launch"录像"下来，之后每步只需一次replay，省掉CPU逐个launch的时间（decode时这部分很可观），
    #   代价是输入形状必须固定，所以按batch大小分档（bucket）录多张图、不足的用padding补齐。PIECEWISE = 在注意力等"切分算子"处把图切开，分段录图（ATOM level 3的默认）。
    #   更准确地说：--level 3（默认level）是"分段编译"；原生命令行默认的--cudagraph-mode是FULL整图，PIECEWISE要显式指定
    #   （各模式的实际行为见config.py CUDAGraphMode上方的适用场景表）
    #   ATOM里见atom/utils/decorators.py（support_torch_compile）、docs/compilation_cudagraph_guide.md
    # 【插件】ATOM_SGLANG_PREFILL_COMPILE_ONLY：SGLang插件的prefill只编译、不捕获CUDA Graph（实验性）
    # Keep SGLang's tc_piecewise/Inductor compilation path for prefill, but
    # bypass its per-bucket CUDA Graph capture, static-buffer staging, and token
    # padding. This is an experimental ATOM SGLang-plugin-only mode.
    # 【译】保留SGLang对prefill的tc_piecewise/Inductor编译路径，但跳过它按bucket的CUDA Graph捕获、静态缓冲区暂存和token padding。
    #   这是实验性的、仅ATOM SGLang插件使用的模式。
    "ATOM_SGLANG_PREFILL_COMPILE_ONLY": lambda: (
        os.getenv("ATOM_SGLANG_PREFILL_COMPILE_ONLY", "0") == "1"
    ),
    # 科普·投机解码：用一个便宜的draft模型（或模型自带的MTP头）一次猜出k个后续token，再让大模型（target）在一次forward里并行验证，
    #   猜对的直接接受，从而一次forward产出多个token。EAGLE3 / MTP / DSpark是三种draft方式（DSpark一次并行猜出一整块，见recipes/DSpark.md）。
    #   实现见atom/spec_decode/、model_ops/rejection_sampler.py
    # 【插件】ATOM_SGLANG_EAGLE3_TP_VERIFY_BROADCAST：SGLang插件里EAGLE3验证结果从TP rank 0广播，保证各rank一致（默认关）
    # Broadcast EAGLE3 verification predictions and acceptance decisions from
    # TP rank 0 so every tensor-parallel rank advances with identical results.
    # SGLang lacks this broadcast path; the ATOM plugin adds it to match native
    # ATOM's tensor-parallel verification semantics.
    # Disabled by default because native SGLang normally keeps ranks in sync.
    # 【译】从TP rank 0广播EAGLE3验证的预测和接受决定，让每个张量并行rank以相同结果前进。
    #   SGLang没有这条广播路径；ATOM插件加上它，以匹配原生ATOM张量并行验证的语义。
    #   默认关闭，因为原生SGLang通常能保持各rank同步。
    "ATOM_SGLANG_EAGLE3_TP_VERIFY_BROADCAST": lambda: (
        os.getenv("ATOM_SGLANG_EAGLE3_TP_VERIFY_BROADCAST", "0") == "1"
    ),
    # 科普·PP（流水线并行）：把模型的层按顺序切给几组GPU（stage），一个batch像流水线一样依次流过各stage。
    #   长prompt的prefill被切成多个chunk送进流水线，"动态分块"按已有上下文长度调整每个chunk的大小，让各chunk耗时接近，减少流水线气泡
    # 【并行】ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR：PP动态分块prefill的平滑系数（默认0.75）；读取方：config.py、arg_utils.py
    #   【分支】动态分块PP prefill见第9批 dynamic_chunking.py
    # Dynamic CPP follows SGLang's quadratic chunk-size predictor. This factor
    # interpolates between the configured initial chunk size (0) and the fitted
    # equal-latency size (1). Values around 0.6-0.85 are typically most stable.
    # Read by Config only when dynamic chunking is on and no explicit factor
    # was given.
    # 【译】动态CPP沿用SGLang的二次chunk大小预测器。这个系数在配置的初始chunk大小（0）和拟合出的等延迟大小（1）之间插值。
    #   取0.6-0.85左右通常最稳定。只有动态分块开启、且没有显式给出系数时，Config才读取它。
    "ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR": lambda: float(
        os.getenv("ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR", "0.75")
    ),
    # --- Compilation & Execution ---
    # 【译】--- 编译与执行 ---
    # （这个分节下实际是GEMM、MLA、MoE等各种kernel选择开关，与torch.compile无关）
    # 科普·kernel后端（本文件大量开关都是在它们之间选）：
    #   AITER：AMD的算子库（ATOM的主要依赖），里面同一个算子常有多种实现：ASM（手写汇编，最快但只支持特定形状/架构）、
    #     CK（Composable Kernel，C++模板）、Triton（Python写的GPU kernel，可移植、易改）、gluon（Triton的底层方言）、FlyDSL（另一种kernel DSL）；
    #     OPUS也是aiter里的一类kernel实现。一般规律：ASM/FlyDSL快但条件苛刻，Triton通用，常作为后备或A/B对照
    #   GPU架构名：gfx942 = MI300X/MI325X，gfx950 = MI350X/MI355X，gfx1250是更新一代的架构（wave32）；很多开关只在某个架构上生效
    # 【GEMM】ATOM_USE_TRITON_GEMM：Linear用aiter的Triton GEMM（含FP4 preshuffle GEMM），也会影响MoE是否走Triton；读取方：linear.py、moe.py
    "ATOM_USE_TRITON_GEMM": lambda: os.getenv("ATOM_USE_TRITON_GEMM", "0") == "1",
    # 科普·FP8与block-scale量化：FP8只有8位，能表示的范围很窄，所以要配"scale"：实际值 ≈ fp8值 * scale。
    #   per-tensor = 整个张量一个scale；per-token = 每行一个；block-scale = 每128x128的权重块一个scale（DeepSeek-V3的FP8格式），
    #   粒度越细精度越好、开销越大。scale本身可以存成FP32，也可以存成E8M0（只有指数位，即只能是2的幂，硬件处理更省）
    # 【GEMM】ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE：FP8 block-scale量化的scale用E8M0（2的幂）格式
    #   显式设置时以它为准；没设置时config.py按checkpoint的scale_fmt（"ue8m0"）和GPU架构自动判断
    "ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE": lambda: (
        os.getenv("ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE", "0") == "1"
    ),
    # 【GEMM】ATOM_GROUP32_WEIGHT_PRESHUFFLE：gfx950上按32行分组量化的FP8权重是否做preshuffle（gfx1250总是做），默认开；读取方：linear.py
    #   更正：linear.py里这个开关只在gfx950上起作用，代码中没有"gfx1250总是做"的分支；"总是做"的是模块自己声明了
    #   needs_preshuffled_weight的权重（DeepSeek-V4.1的分组wo_a），与架构无关。"group32"指原生FP8 32x32分组scale的权重（DeepSeek-V4.1）
    #   科普·preshuffle：加载时把权重矩阵按kernel读取的顺序预先重排一次，GEMM运行时就能用连续、对齐的方式读取，换来更高带宽；
    #   代价是权重不再是普通的行主序，只有认这种布局的kernel能用，所以preshuffle开关和kernel选择必须配套
    "ATOM_GROUP32_WEIGHT_PRESHUFFLE": lambda: (
        os.getenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", "1") == "1"
    ),
    # 科普·MXFP4与aXwY命名：MXFP4是OCP的"微缩放"4位浮点格式——每32个元素共享一个E8M0 scale，每个元素只有4位（e2m1），
    #   权重体积约为BF16的1/4，MI355X（gfx950）有原生FP4矩阵指令。kernel名里的a8w4 = 激活8位、权重4位，a4w4 = 都是4位，a16w4 = 激活BF16。
    #   权重格式由checkpoint的量化配置决定（atom/quant_spec.py），这些开关只决定用哪个kernel去算
    # 【注意力】ATOM_USE_TRITON_MXFP4_BMM：MLA里的批量矩阵乘（bmm）用aiter Triton的MXFP4 kernel（默认关）；读取方：attention_mla.py、config.py
    "ATOM_USE_TRITON_MXFP4_BMM": lambda: (
        os.getenv("ATOM_USE_TRITON_MXFP4_BMM", "0") == "1"
    ),
    # 【注意力】ATOM_USE_TRITON_MLA：MLA注意力后端用TritonMLABackend，而不是默认的AiterMLABackend；读取方：utils/selector.py
    "ATOM_USE_TRITON_MLA": lambda: os.getenv("ATOM_USE_TRITON_MLA", "0") == "1",
    # 【注意力】ATOM_USE_TRITON_MLA_SHUFFLE_KV：旧开关，已被下面的ATOM_MLA_KV_SHUFFLED取代
    # Deprecated: use ATOM_MLA_KV_SHUFFLED. Only effective together with
    # ATOM_USE_TRITON_MLA=1.
    # 【译】已弃用：请用ATOM_MLA_KV_SHUFFLED。只在与ATOM_USE_TRITON_MLA=1一起设置时才生效。
    "ATOM_USE_TRITON_MLA_SHUFFLE_KV": lambda: (
        os.getenv("ATOM_USE_TRITON_MLA_SHUFFLE_KV", "0") == "1"
    ),
    # 【注意力】ATOM_MLA_KV_SHUFFLED：MLA的KV cache用"打乱"布局（每页64个token，按16x16小块存放），默认关
    # Shuffled MLA KV cache: block_size=64 pages, NoPE/RoPE split into 16x16
    # tiles. Independent of the decode kernel.
    # Also set by the legacy ATOM_USE_TRITON_MLA=1 + ATOM_USE_TRITON_MLA_SHUFFLE_KV=1.
    # 【译】打乱布局的MLA KV cache：block_size=64的页，NoPE/RoPE部分切成16x16的tile。与decode kernel无关。
    #   旧的ATOM_USE_TRITON_MLA=1 + ATOM_USE_TRITON_MLA_SHUFFLE_KV=1组合也会打开它。
    "ATOM_MLA_KV_SHUFFLED": lambda: (
        os.getenv("ATOM_MLA_KV_SHUFFLED", "0") == "1"
        or (
            os.getenv("ATOM_USE_TRITON_MLA", "0") == "1"
            and os.getenv("ATOM_USE_TRITON_MLA_SHUFFLE_KV", "0") == "1"
        )
    ),
    # 科普·MoE：把FFN换成很多个"专家"FFN，router为每个token选top-k个专家（如256选8），只算被选中的，参数量大而每token计算量小。
    #   "路由专家"是被选的那些；DeepSeek等还有每个token都会经过的"共享专家"。fused_moe = 把"按专家重排token + 两个GEMM + 激活 + 加权求和"
    #   合成少数几个kernel。实现见atom/model_ops/moe.py、fused_moe/
    # 【MoE】ATOM_USE_TRITON_MOE：路由专家改用aiter的Triton/gluon MoE kernel（否则用FlyDSL fused_moe）
    #   注意：lambda的默认值"0"只在显式读取时有意义；moe.py先用envs.is_set判断——未设置时按架构自动决定
    #   （非EP且gfx94x，或gfx95x且ATOM_USE_TRITON_GEMM=1时开启），设置了才以设置值为准
    #   准确地说（moe.py）：use_triton_moe = 非EP and (gfx94x or (gfx95x and ATOM_USE_TRITON_GEMM))，"非EP"对两种架构都适用
    # Run the routed experts with the aiter Triton/gluon MoE kernels instead of
    # FlyDSL fused_moe, on prefill and decode alike. For SiLU models on gfx1250
    # this selects the a8w4 GUGU (gate/up-interleaved) kernel -- the default --
    # which fuses SiLU into GEMM1's write-back and the MXFP8 requant into its
    # epilogue. SwiGLU models (GPT-OSS) and CDNA archs keep the general
    # moe_gemm_a16w4 / a4w4 / a8w4 path. Defaults to on for gfx94x, and for
    # gfx95x when ATOM_USE_TRITON_GEMM is set.
    # 【译】用aiter的Triton/gluon MoE kernel而不是FlyDSL fused_moe来跑路由专家，prefill和decode都一样。
    #   对gfx1250上的SiLU模型，这会选用a8w4 GUGU（gate/up交错）kernel——默认就是它——它把SiLU融合进GEMM1的写回，
    #   把MXFP8重量化融合进它的epilogue。SwiGLU模型（GPT-OSS）和CDNA架构仍走通用的moe_gemm_a16w4 / a4w4 / a8w4路径。
    #   在gfx94x上默认开启；在gfx95x上设置了ATOM_USE_TRITON_GEMM时默认开启。
    "ATOM_USE_TRITON_MOE": lambda: os.getenv("ATOM_USE_TRITON_MOE", "0") == "1",
    # 【MoE】ATOM_USE_TRITON_MOE_DECODE：按阶段拆分MoE：prefill用FlyDSL fused_moe，decode用Triton/gluon GUGU kernel（条件苛刻，见下）
    # Split the routed experts by phase: FlyDSL fused_moe on prefill, the Triton
    # /gluon GUGU kernel on decode. Needs ATOM_USE_TRITON_MOE=1 (it narrows that
    # flag, it cannot enable Triton on its own) plus gfx1250 + ATOM_MOE_GU_ITLV=1
    # + SiLU, because it keeps a single copy of the weights in the FlyDSL layout
    # and hands Triton a zero-copy view of it -- which is only valid where the
    # two preshuffles agree byte-for-byte -- which is what ATOM_MOE_GU_ITLV=1
    # buys, and why the prep asserts it: only the interleaved layout is shared,
    # so at ATOM_MOE_GU_ITLV=0 the FlyDSL prep and the Triton view disagree.
    # tests/test_mxfp4_triton_moe_decode.py runs both real preps and compares.
    #
    # Arms under EP as well as TP, at both EP entry points -- the modular-kernel
    # (transport) path and the local no-transport one build the same views.
    # 【译】按阶段拆分路由专家：prefill用FlyDSL fused_moe，decode用Triton/gluon GUGU kernel。需要ATOM_USE_TRITON_MOE=1
    #   （它只是收窄那个开关，自己不能单独开启Triton），外加gfx1250 + ATOM_MOE_GU_ITLV=1 + SiLU，因为它只保留一份FlyDSL布局的权重，
    #   交给Triton一个零拷贝视图——这只在两种preshuffle逐字节一致时才成立——而这正是ATOM_MOE_GU_ITLV=1带来的，
    #   也是权重准备阶段要断言它的原因：只有交错布局是共享的，所以在ATOM_MOE_GU_ITLV=0时，FlyDSL的准备结果和Triton视图对不上。
    #   tests/test_mxfp4_triton_moe_decode.py会运行两种真实的准备流程并比较。
    #
    #   在EP和TP下都会生效，两个EP入口都会——modular-kernel（有传输）路径和本地无传输路径构建的是同样的视图。
    "ATOM_USE_TRITON_MOE_DECODE": lambda: (
        os.getenv("ATOM_USE_TRITON_MOE_DECODE", "0") == "1"
    ),
    # 【MoE】ATOM_USE_TRITON_MOE_A4W4：Triton MoE里激活也量化成4bit（a4w4），默认是a8w4
    # Select the a4w4 Triton wrapper instead of the a8w4 default, on both the TP
    # and EP paths. Only chooses *which* wrapper runs -- it cannot enable the
    # Triton path on its own, and asserts if set without ATOM_USE_TRITON_MOE.
    # The weights are identical (both are w4); only the activation quant
    # differs, so no extra weight prep or memory is involved.
    # 【译】在TP和EP路径上都选用a4w4的Triton封装，而不是默认的a8w4。它只选择*哪个*封装运行——
    #   不能单独开启Triton路径，没设ATOM_USE_TRITON_MOE却设了它会触发断言。
    #   权重完全相同（都是w4），只有激活量化不同，所以不涉及额外的权重准备和内存。
    "ATOM_USE_TRITON_MOE_A4W4": lambda: (
        os.getenv("ATOM_USE_TRITON_MOE_A4W4", "0") == "1"
    ),
    # 科普·EP与all2all：EP（专家并行）把不同专家放在不同GPU上。每层MoE要先把每个token发到它选中的专家所在的卡（dispatch），
    #   算完再发回原卡加权求和（combine），这两次都是all-to-all通信。MoRI是AMD的通信库，提供这套dispatch/combine kernel；
    #   下面几个开关调它的版本（v1/v2）和传输精度（发FP8/FP4能省带宽，但有量化误差）。实现见atom/model_ops/fused_moe/mori_*.py
    # 【MoE】ATOM_DISABLE_MORI_EP：DP attention + EP时不用MoRI的all-to-all，改走集合通信的后备路径（做A/B对比用）
    # Force DP-attention + EP through the collective fallback even when mori is
    # installed. This is useful for controlled A/B tests and for deployments
    # where the mori shared-memory transport is unavailable or undesirable.
    # The fallback gathers hidden/router rows across DP ranks, computes only
    # the locally owned experts, then reduce-scatters the outputs.
    # 【译】即使安装了mori，也强制DP-attention + EP走集合通信后备路径。这对受控的A/B测试有用，
    #   也适用于mori共享内存传输不可用或不希望使用的部署。后备路径在各DP rank之间gather hidden/router行，
    #   只计算本地拥有的专家，然后对输出做reduce-scatter。
    "ATOM_DISABLE_MORI_EP": lambda: os.getenv("ATOM_DISABLE_MORI_EP", "0").lower()
    in {"1", "true", "yes", "on"},
    # 【MoE】ATOM_MORI_V2：EP+DP的MoE all2all改用mori的dispatch_combine_v2（能在gfx1250上运行）
    # Use mori dispatch_combine_v2 (FlyDSL/cco, gfx1250 wave32) instead of the
    # production mori v1 (mori.ops.EpDispatchCombineOp) for the EP+DP MoE
    # all2all. v1 is authored for gfx942/950 and does not run on gfx1250; v2 is
    # the gfx1250-capable path. Only takes effect when the mori all2all path is
    # active (dp_size>1 + expert-parallel + mori installed).
    # 【译】EP+DP MoE的all2all使用mori的dispatch_combine_v2（FlyDSL/cco，gfx1250 wave32），而不是生产用的mori v1
    #   （mori.ops.EpDispatchCombineOp）。v1是为gfx942/950写的，不能在gfx1250上运行；v2是能跑gfx1250的路径。
    #   只有mori all2all路径生效时（dp_size>1 + 专家并行 + 已安装mori）才起作用。
    "ATOM_MORI_V2": lambda: os.getenv("ATOM_MORI_V2", "0") == "1",
    # 【MoE】ATOM_MEGA_*（下面6个）：gfx1250上aiter MegaMoE的阶段融合与通信精度开关
    # 【MoE】ATOM_MEGA_STAGE2_FUSED：MegaMoE第2阶段（combine）是否由GEMM2的epilogue直接点对点写入对端（默认开）
    # MegaMoE stage 2 (gemm2-fused EP combine): the grouped gemm2 epilogue
    # P2P-writes its weighted per-(token,k) results straight into the peers'
    # combine staging, so combine only barriers + sums. Requires the mxfp4 weight
    # path; ignored unless ATOM_MORI_V2 is on.
    # This also selects the transport: 1 binds aiter's MegaMoEGfx1250, which owns
    # the fused dispatch/combine pair, 0 binds mori's v2 op-layer running plain
    # gather, i.e. the untouched upstream baseline. Formerly ATOM_MORI_V2_FUSED.
    # 【译】MegaMoE第2阶段（与gemm2融合的EP combine）：分组gemm2的epilogue把加权后的每个(token,k)结果
    #   点对点直接写进各对端的combine暂存区，于是combine只需要屏障+求和。需要mxfp4权重路径；ATOM_MORI_V2不开时忽略。
    #   它同时决定传输层：1绑定aiter的MegaMoEGfx1250（自带融合的dispatch/combine），0绑定mori的v2 op层、
    #   跑普通gather，即未改动的上游基线。以前叫ATOM_MORI_V2_FUSED。
    "ATOM_MEGA_STAGE2_FUSED": lambda: (os.getenv("ATOM_MEGA_STAGE2_FUSED", "1") == "1"),
    # 【MoE】ATOM_MEGA_COMBINE_WIRE：MegaMoE combine（回程）的传输精度：bf16 | fp8 | fp4
    # MegaMoE combine (return-trip) wire: bf16 | fp8 | fp4. Prefill-only; decode
    # always combines in bf16. Ignored unless ATOM_MEGA_STAGE2_FUSED is on.
    # 【译】MegaMoE combine（回程）的传输精度：bf16 | fp8 | fp4。只用于prefill；decode始终用bf16做combine。
    #   ATOM_MEGA_STAGE2_FUSED不开时忽略。
    "ATOM_MEGA_COMBINE_WIRE": lambda: os.getenv("ATOM_MEGA_COMBINE_WIRE", "bf16"),
    # 【MoE】ATOM_MEGA_STAGE1_FUSED：MegaMoE第1阶段（dispatch）是否用融合了路由规划的compact plan（默认开）
    # MegaMoE compact-plan stage 1 (aiter stage1_fused): routing/layout planning
    # fused with a flydsl TDM dispatch that writes the grouped GEMM's per-expert
    # rows directly. Needs ATOM_MEGA_STAGE2_FUSED and an fp8/fp4
    # MEGA_DISPATCH_WIRE; a bf16 wire keeps the token-major path regardless of
    # this flag.
    # 【译】MegaMoE compact-plan第1阶段（aiter stage1_fused）：路由/布局规划与flydsl TDM dispatch融合，
    #   直接写出分组GEMM每个专家的行。需要ATOM_MEGA_STAGE2_FUSED，以及fp8/fp4的MEGA_DISPATCH_WIRE；
    #   bf16传输时无论这个开关如何，都走按token排列的路径。
    "ATOM_MEGA_STAGE1_FUSED": lambda: (os.getenv("ATOM_MEGA_STAGE1_FUSED", "1") == "1"),
    # 【MoE】ATOM_MEGA_DECODE_FAST_PATH：小规模的decode/验证/draft forward复用一个小容量的MegaMoEV2实例（默认开）
    # Reuse a small MegaMoEV2 instance for native DP-unified small decode/
    # verify/draft forwards on the supported EP8, 48-experts-per-rank layout. Set to 0
    # to keep the configured max_num_batched_tokens capacity for every graph.
    # 【译】在支持的EP8、每rank 48个专家的布局上，原生DP-unified的小规模decode/verify/draft forward复用一个小的MegaMoEV2实例。
    #   设为0则每张graph都保持配置的max_num_batched_tokens容量。
    "ATOM_MEGA_DECODE_FAST_PATH": lambda: (
        os.getenv("ATOM_MEGA_DECODE_FAST_PATH", "1") == "1"
    ),
    # 【MoE】ATOM_MEGA_DECODE_MTPR：上面那个小实例每个rank的行数（默认128）
    # Rows per rank of that small instance: 128, 256, 512 or 1024 (see docs).
    # 【译】那个小实例每个rank的行数：128、256、512或1024（见文档）。
    "ATOM_MEGA_DECODE_MTPR": lambda: _int_env("ATOM_MEGA_DECODE_MTPR", 128),
    # 【MoE】ATOM_MEGA_MASK_PAD_ROWS：MegaMoE后端把DP填充行路由到专家-1直接跳过（默认开）
    # Route DP pad rows (past this rank's scheduled tokens) to expert -1 on the
    # MegaMoE backend, which skips them, so padding costs no transport or GEMM.
    # Set to 0 to retain the unmasked path.
    # 【译】在MegaMoE后端上把DP填充行（超出本rank已调度token的行）路由到专家-1，后端会跳过它们，
    #   所以填充不产生任何传输和GEMM开销。设为0保留不做mask的路径。
    "ATOM_MEGA_MASK_PAD_ROWS": lambda: os.getenv("ATOM_MEGA_MASK_PAD_ROWS", "1") == "1",
    # 【注意力】ATOM_MLA_PAGE_SIZE：MLA KV cache的页大小（token数），默认1；>1时AiterMLABackend用它作block_size（DCP的部分路径要求<=1）
    #   核实（attention_mla.py）：非Triton MLA路径上>1表示用分段（seg）kernel，并且必须恰好等于64、同时kv_cache_block_size也为64，
    #   否则启动报错；dcp_utils.py里DCP的persistent kernel和非persistent稀疏prefill要求它<=1
    "ATOM_MLA_PAGE_SIZE": lambda: int(os.getenv("ATOM_MLA_PAGE_SIZE", "1")),
    # 读取方：model_ops/attention_mha.py，以及SGLang插件的full_attention_backend.py。这里的FMHA就是flash-attention式的融合注意力kernel；
    #   转FP8算prefill注意力更快，代价是少量精度损失，怀疑精度问题时可设0做A/B
    # 【注意力】ATOM_AITER_FP8_PREFILL_ATTN：gfx950上MHA prefill把Q/K/V转成FP8，用AITER head_dim=256的FMHA kernel（默认开）
    # Match SGLang's gfx950 prefill fast path, with or without a cached prefix:
    # cast Q/K/V to FP8 and use AITER's head-dim-256 per-tensor FMHA kernel.
    # Set to 0 for the BF16 flash_attn_varlen_func fallback.
    # 【译】对齐SGLang在gfx950上的prefill快速路径，无论有没有缓存的前缀：把Q/K/V转成FP8，使用AITER的head_dim=256逐张量FMHA kernel。
    #   设为0则回退到BF16的flash_attn_varlen_func。
    "ATOM_AITER_FP8_PREFILL_ATTN": lambda: (
        os.getenv("ATOM_AITER_FP8_PREFILL_ATTN", "1") == "1"
    ),
    # 科普·mHC（Manifold-Constrained Hyper-Connections，DeepSeek-V4/V4.1）：
    #   普通Transformer（MHA、GQA都一样）每个子层只有一条残差流：x = x + Attn(Norm(x))。
    #   mHC把残差流加宽成hc_mult=4条并行的流，每个子层（attention或FFN）前后各做一次小混合：
    #
    #       4条残差流 [n, 4, dim] ──hc_pre：按门控权重pre把4条加权合成1条──> [n, dim] ──Norm + Attn/FFN（照常）──┐
    #             │                                                                                      │
    #             └──comb（4x4混合矩阵，作用在旧的4条流上）──> (+) <──hc_post：按门控权重post把输出分回4条──┘
    #
    #   pre、post、comb都是每个token从残差现算出来的：pre是sigmoid门、post是2*sigmoid门，comb经Sinkhorn迭代（默认20次）
    #   投影成"双随机矩阵"（每行、每列之和都≈1，这就是Manifold-Constrained），混合时不会把残差整体放大或缩小
    #   （model_ops/sparse_attn_v4.py的hc_split_sinkhorn）。
    #   例（数字是假设的）：dim=4096时，每个token的残差从4096个数变成4x4096=16384个数；attention/FFN本身的计算量不变，
    #   多出来的只是这些4维的小矩阵运算。实现见models/deepseek_v4.py的Block.hc_pre / hc_post
    #   （优先用aiter的mhc_pre / mhc_post融合kernel）。下面的开关决定这部分小计算用FP32还是BF16 hi/lo拆分
    # 【模型】ATOM_MHC_USE_BF16：DeepSeek-V4的mHC（超连接）模块加载后打包fn权重、用BF16 hi/lo计算（默认开）；读取方：models/deepseek_v4
    # Pack mHC fn weights once after loading and use BF16 hi/lo computation.
    # Set to 0 before model loading to retain FP32 fn and FP32 mHC computation.
    # 【译】加载后把mHC的fn权重打包一次，并使用BF16 hi/lo计算。在加载模型之前设为0，可保留FP32的fn和FP32的mHC计算。
    "ATOM_MHC_USE_BF16": lambda: os.getenv("ATOM_MHC_USE_BF16", "1") == "1",
    # --- Kernel Fusion Toggles ---
    # 【译】--- kernel融合开关 ---
    # 科普·kernel融合：一次kernel launch有固定开销（几微秒），而且每个kernel都要把结果写回显存（HBM）、下一个再读出来。
    #   把RMSNorm、RoPE、量化、写KV cache等相邻小算子合成一个kernel，中间结果留在寄存器里，省launch也省显存往返；decode时batch小，收益最明显。
    #   下面的"融合"开关默认值一般就是推荐值；关掉它们主要用于怀疑精度回归时做A/B
    # 读取方：model_ops/v4_kernels/fused_compress.py（DeepSeek-V4压缩器：把每RATIO个相邻token的KV压成一条）
    # 【融合】ATOM_FUSED_COMPRESS_USE_FLYDSL：DeepSeek-V4（V4-Pro）压缩器的融合kernel用Triton还是FlyDSL：auto / always / never
    # fused_compress_attn: switch between Triton (default historical) and a
    # flydsl drop-in for V4-Pro Compressor (Main BF16 + Indexer FP8) paths.
    # "auto" picks flydsl when shape matches the supported configs (D ∈
    # {128, 512}, RD=64, OVERLAP=1, RATIO=4); "always" forces it (errors on
    # unsupported); "never" pins Triton. flydsl pure-GPU time beats Triton
    # across the full range on V4-Pro (1.1x small N → 2-3x at N≥4096).
    # 【译】fused_compress_attn：在Triton（历来的默认）和FlyDSL替代实现之间切换，用于V4-Pro的Compressor（主路BF16 + Indexer FP8）路径。
    #   "auto"在形状匹配支持的配置时（D ∈ {128, 512}，RD=64，OVERLAP=1，RATIO=4）选FlyDSL；"always"强制用它（不支持时报错）；
    #   "never"固定用Triton。在V4-Pro上，FlyDSL的纯GPU时间在整个范围内都优于Triton（小N时1.1倍 → N≥4096时2-3倍）。
    "ATOM_FUSED_COMPRESS_USE_FLYDSL": lambda: os.getenv(
        "ATOM_FUSED_COMPRESS_USE_FLYDSL", "auto"
    ).lower(),
    # 读取方：model_ops/attention_mla.py。背景：命中前缀缓存后，prefill要对缓存里的latent做kv_b_proj展开成完整K/V，这一步先gather再GEMM
    # 【融合】ATOM_USE_FLYDSL_GATHER_KV_B_PROJ：MLA缓存前缀的展开（gather + kv_b_proj）用FlyDSL的a8w8 gather-GEMM（默认开）
    # gather_kv_b_proj (the MLA cached-prefix expansion): swap the Triton op for
    # the flydsl a8w8 gather-GEMM. Added 2026-09-09.
    # 【译】gather_kv_b_proj（MLA缓存前缀的展开）：把Triton算子换成FlyDSL的a8w8 gather-GEMM。2026-09-09加入。
    "ATOM_USE_FLYDSL_GATHER_KV_B_PROJ": lambda: (
        os.getenv("ATOM_USE_FLYDSL_GATHER_KV_B_PROJ", "1") == "1"
    ),
    # 【注意力】ATOM_USE_FLYDSL_FP8_PREFILL_ATTN：MLA prefill用FlyDSL的FP8 kernel（融合QKV量化），默认关；读取方：attention_mla.py
    # FlyDSL FP8 prefill with fused QKV quantization and direct FP8 gather output
    # where supported. Unsupported attention inputs raise. Added 2026-09-10.
    # 【译】FlyDSL FP8 prefill，在支持的地方融合QKV量化并直接输出FP8 gather结果。不支持的注意力输入会报错。2026-09-10加入。
    "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN": lambda: (
        os.getenv("ATOM_USE_FLYDSL_FP8_PREFILL_ATTN", "0") == "1"
    ),
    # 【融合】ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION：q/k norm + RoPE + 写KV cache + 量化合成一个kernel（默认关）
    #   读取方：models/qwen3_moe.py、qwen3_next.py、glm4_moe.py、gpt_oss.py等
    # QK-norm-rope-cache-quant fusion for Qwen3 dense and MoE; disabled by default.
    # 【译】Qwen3 dense和MoE的QK-norm-rope-cache-quant融合；默认关闭。
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION", "0") == "1"
    ),
    # 读取方：model_ops/attention_mha.py
    # 【融合】ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS：上面融合的token数上限（默认8192），超过就拆回分开的kernel
    # With that fusion on, batches above this token count run the Gemma q/k
    # norm, RoPE and cache write as separate kernels: the one-kernel version
    # loses at prefill sizes while saving launches at decode sizes (crossover
    # between 8192 and 16384 tokens for Qwen3.5 TP4 on MI355X).
    # 【译】开启那个融合后，token数超过这个值的batch把Gemma q/k norm、RoPE和写cache作为分开的kernel运行：
    #   单kernel版本在prefill规模上更慢，在decode规模上省去了launch（MI355X上Qwen3.5 TP4的交叉点在8192到16384个token之间）。
    "ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS": lambda: int(
        os.getenv("ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS", "8192")
    ),
    # 科普·DSA（DeepSeek Sparse Attention，DeepSeek-V3.2 / GLM-5系 / DeepSeek-V4使用）：每个query不再attend全部历史token，
    #   先由一个轻量的"indexer"（少量小头、FP8计算）给所有历史token打分，选出top-k（例如2048个），主注意力（MLA）只算这k个。
    #   长上下文下注意力从O(L)降到O(k)；indexer有自己的一份"index cache"（FP8的key）。实现见models/deepseek_v2.py的Indexer、
    #   model_ops/sparse_indexer_*.py
    # 【融合】下面4个ATOM_ENABLE_DS_*是DeepSeek系（models/deepseek_v2.py，覆盖V3 / V3.2 / GLM-5）的融合开关，默认都开，模块导入时读取：
    #   INPUT_RMSNORM_QUANT_FUSION：层输入RMSNorm + 量化；QKNORM_QUANT_FUSION：MLA的q/kv低秩分支norm + 量化（DeepSeek-V4也读）；
    #   QKNORM_FUSION：q/kv低秩分支norm融合；INDEXER_QK_ROPE_CACHE_FUSION：DSA indexer的q/k RoPE + 写index cache融合
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_QKNORM_QUANT_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_QKNORM_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_QKNORM_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION", "1") == "1"
    ),
    # 科普·线性注意力 / 递归状态（GDN、KDA）：Qwen3-Next / Qwen3.5的Gated DeltaNet（GDN）、Kimi的Kimi Delta Attention（KDA）这类层
    #   不保存逐token的K/V，而是每个头维护一个固定大小的状态矩阵，每来一个token就更新一次（像RNN）。好处是显存不随长度增长；
    #   坏处是前缀复用时没法"取前N个token的KV"，只能复用在第N个token处保存下来的状态快照，这就是"状态checkpoint"。
    #   这些模型通常是线性注意力层与普通注意力层混合。实现见model_ops/attention_gdn.py、attentions/gdn_attn.py、fla_ops/
    #   状态checkpoint怎样放置、命中时从哪里恢复（带图和数字），见arg_utils.py EngineArgs.state_checkpoint_interval_tokens上方的【科普】
    # 【状态缓存】ATOM_STATE_CHECKPOINT_DEMAND：前缀命中因没有合适的状态checkpoint而被拒绝时，是否按需补放一个checkpoint（默认开）
    #   背景见上面的科普；机制在第4批；读取方：block_manager.py
    # Set to 0 to stop a refused state-cache hit from placing a checkpoint of
    # its own, leaving the prompt-end anchor as the only placement. Overrides
    # --state-checkpoint-demand so the policy can be flipped without touching a
    # launch script. The rung's write traffic may cost more in evictions than
    # its reuse is worth — see `BlockManager._record_checkpoint_demand`.
    # 【译】设为0时，被拒绝的状态缓存命中不再放置它自己的checkpoint，只剩prompt末尾的锚点这一种放置。
    #   它会覆盖--state-checkpoint-demand，这样不改启动脚本就能切换策略。这一档checkpoint的写流量
    #   在淘汰上付出的代价可能超过它被复用的价值——见`BlockManager._record_checkpoint_demand`。
    "ATOM_STATE_CHECKPOINT_DEMAND": lambda: (
        os.getenv("ATOM_STATE_CHECKPOINT_DEMAND", "1") == "1"
    ),
    # 【状态缓存】ATOM_STATE_CHECKPOINT_END_MARGIN：prompt末尾的状态checkpoint至少离prompt结尾多少个token（默认0）
    # Tokens kept between the prompt-end state checkpoint and the prompt end,
    # at least; the backend's own successor room still applies when it is
    # larger. A chat template rewrites the tail of an assistant turn when the
    # next user turn is appended (Qwen3.5 drops the generation prompt's
    # `<think>`), so the next prompt diverges a few tokens before this one
    # ends; an anchor inside that suffix is never matched again. 0 keeps the
    # old placement. See docs/environment_variables.md for what a margin wider
    # than a block costs.
    # 【译】prompt末尾的状态checkpoint与prompt结尾之间至少保留的token数；如果backend自己要求的后继空间更大，以后者为准。
    #   追加下一轮user消息时，chat template会改写上一轮assistant消息的结尾（Qwen3.5会去掉生成提示里的`<think>`），
    #   所以下一个prompt会在这个prompt结尾前几个token处分叉；落在这段后缀里的锚点永远不会再被匹配上。
    #   0保持旧的放置方式。比一个block还宽的margin有什么代价，见docs/environment_variables.md。
    "ATOM_STATE_CHECKPOINT_END_MARGIN": lambda: max(
        0, int(os.getenv("ATOM_STATE_CHECKPOINT_END_MARGIN", "0"))
    ),
    # 【状态缓存】ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS：vLLM插件里状态checkpoint的间隔（token数）；读取方：plugin/vllm/paged_state_checkpoint_scheduler.py
    # vLLM plugin, state checkpoints: overrides the native checkpoint interval
    # (tokens; 0 = publish no state checkpoint, -1 = no rungs, anchor and demand only).
    # Unset: the model's native default.
    # 【译】vLLM插件，状态checkpoint：覆盖原生的checkpoint间隔（单位token；0 = 不发布状态checkpoint，
    #   -1 = 不设定期档位，只保留锚点和按需checkpoint）。未设置：用模型原生的默认值。
    "ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS": lambda: (
        int(v) if (v := os.getenv("ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS")) else None
    ),
    # 【调试】ATOM_STATE_CHECKPOINT_CHECK_BLOCKS：vLLM插件里每次调度后检查状态checkpoint占用的block数（诊断用）
    # vLLM plugin, state checkpoints: after every schedule, check that no
    # request holds more than two blocks per state checkpoint group (diagnostic).
    # 【译】vLLM插件，状态checkpoint：每次调度后检查没有请求在每个状态checkpoint组里持有超过两个block（诊断用）。
    "ATOM_STATE_CHECKPOINT_CHECK_BLOCKS": lambda: (
        os.getenv("ATOM_STATE_CHECKPOINT_CHECK_BLOCKS", "0") == "1"
    ),
    # 【状态缓存】ATOM_GDN_STATE_MIDSTEP：Qwen GDN的状态checkpoint直接取chunk kernel的中间状态，不再为对齐checkpoint而切分prefill chunk
    # Set to 1 to read Qwen GDN state checkpoints out of the chunk kernel's
    # interior states instead of cutting prefill chunks onto them. PAGE path
    # only (Qwen3-Next / Qwen3.5); see `PagedStateCheckpointCoordinator`.
    # 【译】设为1时，从chunk kernel的内部状态里读出Qwen GDN的状态checkpoint，而不是把prefill chunk切到checkpoint位置上。
    #   只用于PAGE路径（Qwen3-Next / Qwen3.5）；见`PagedStateCheckpointCoordinator`。
    "ATOM_GDN_STATE_MIDSTEP": lambda: (os.getenv("ATOM_GDN_STATE_MIDSTEP", "0") == "1"),
    # 【索引器】ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB：DSA indexer的dense logits缓冲区显存预算（MiB，默认2048），用来自动决定沿query维分块的大小
    #   读取方：model_ops/sparse_indexer_chunk.py、plugin/vllm/attention/layer_sparse_mla.py
    # DSA sparse-indexer prefill: KV-dimension chunk size (in tokens) for
    # `fp8_mqa_logits`. The dense logits buffer is [prefill_tokens, total_kv];
    # total_kv = sum of all co-scheduled prefill contexts and is NOT bounded by
    # max_num_batched_tokens, so a concurrency burst of long-context requests
    # can drive a single allocation to tens of GiB (see GLM-5.2 OOM #1376).
    # Target peak (in MiB) for the indexer's dense fp32 logits buffer during
    # prefill. The indexer chunks along the query (Q) dimension so the buffer
    # stays ~[q_chunk, total_kv] with q_chunk = budget_bytes // (total_kv * 4);
    # q_chunk shrinks automatically as total_kv (the unbounded KV dimension)
    # grows. Under chunked prefill num_rows is already capped by
    # max_num_batched_tokens, so a fixed row count would not adapt to total_kv
    # (see GLM-5.2 OOM #1376) — a memory budget does. Each chunk still scores
    # the full KV, so every row's top-k is exact (no cross-chunk merge). Set to
    # 0 to disable this soft budget; chunking still enforces aiter's hard 2 GiB
    # buffer-descriptor cap, above which the kernel fails to compile and aborts
    # the process (see atom/model_ops/sparse_indexer_chunk.py).
    # 【译】DSA稀疏indexer的prefill：fp8_mqa_logits在KV维上的分块大小（token数）。dense logits缓冲区是
    #   [prefill_tokens, total_kv]；total_kv是同批所有prefill上下文之和，不受max_num_batched_tokens限制，
    #   所以一波长上下文请求并发进来，单次分配就可能到几十GiB（见GLM-5.2 OOM #1376）。
    #   prefill期间indexer的dense fp32 logits缓冲区的目标峰值（MiB）。indexer沿query（Q）维分块，让缓冲区保持在
    #   约[q_chunk, total_kv]，其中q_chunk = budget_bytes // (total_kv * 4)；total_kv（无上界的KV维）越大，
    #   q_chunk自动越小。chunked prefill下num_rows已经被max_num_batched_tokens限住，所以固定行数适应不了
    #   total_kv（见GLM-5.2 OOM #1376），而显存预算可以。每块仍对完整KV打分，所以每行的top-k都是精确的
    #   （不需要跨块合并）。设为0关闭这个软预算；分块仍会执行aiter硬性的2 GiB缓冲区描述符上限，超过它kernel
    #   编译失败并终止进程（见atom/model_ops/sparse_indexer_chunk.py）。
    # 解释：第一段（"KV-dimension chunk size"）是旧实现留下的描述，现在的分块沿Q维进行，以第二段为准
    "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB": lambda: int(
        os.getenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", "2048")
    ),
    # 【并行】ATOM_DCP_PREFILL_QREP：DCP稀疏prefill时在本地算出整个DCP组的query头，代替AllGather Q（默认关）
    # DCP sparse (DSA) prefill: when QREP is active, produce the whole DCP
    # group's query heads locally, as decode does, instead of AllGather Q. The
    # pynccl gather lands rank-major, so the dim=1 reshape after it also copies
    # the whole gathered q back to token-major; this removes both. Costs a
    # group-wide q GEMM + W_K bmm per prefill token. Off by default.
    # 【译】DCP稀疏（DSA）prefill：QREP生效时，像decode那样在本地产生整个DCP组的query头，而不是AllGather Q。
    #   pynccl的gather按rank为主序落地，所以紧随其后的dim=1 reshape还要把整个gather来的q拷回token为主序；这样做省掉了这两步。
    #   代价是每个prefill token多一次组范围的q GEMM + W_K bmm。默认关闭。
    "ATOM_DCP_PREFILL_QREP": lambda: os.getenv("ATOM_DCP_PREFILL_QREP", "0") == "1",
    # 科普·persistent kernel：普通kernel按工作量发很多个block，GPU调度完就结束；persistent kernel只发"刚好占满所有计算单元"那么多block，
    #   每个block循环领取任务，需要事先算好一份"work metadata"（任务划分表）。它对不规则负载更均衡，但多了准备metadata的开销
    # 【并行】ATOM_DCP_SPARSE_PREFILL_NONPS：DCP稀疏prefill用非persistent的fp8 gqa64 MLA kernel（默认开）
    # DCP sparse prefill: use the non-persistent fp8 gqa64 MLA kernel, which
    # needs no work metadata. On by default; needs an aiter with ROCm/aiter#6132.
    # 【译】DCP稀疏prefill：使用非persistent的fp8 gqa64 MLA kernel，它不需要work metadata。默认开启；需要包含ROCm/aiter#6132的aiter。
    "ATOM_DCP_SPARSE_PREFILL_NONPS": lambda: os.getenv(
        "ATOM_DCP_SPARSE_PREFILL_NONPS", "1"
    )
    == "1",
    # 【索引器】ATOM_ENABLE_GLM_FUSED_INDEXER：GLM-5.2的indexer用与DeepSeek-V3.2相同的融合kernel（默认开）
    # GLM-5.2 (glm_moe_dsa): enable the fused indexer qk-rope + fp8-quant + kv-cache
    # kernel (indexer_qk_rope_quant_and_cache), same path DeepSeek-V3.2 uses. GLM's
    # indexer dims (index_head_dim=128, qk_rope_head_dim=64, per_1x128, neox rope) are
    # identical to V3.2, so the fusion is math-equivalent to the unfused path. Set to
    # "0" to fall back to the per-op Python path if a regression is suspected.
    # 【译】GLM-5.2（glm_moe_dsa）：启用融合的indexer qk-rope + fp8量化 + kv-cache kernel（indexer_qk_rope_quant_and_cache），
    #   与DeepSeek-V3.2用的路径相同。GLM的indexer维度（index_head_dim=128，qk_rope_head_dim=64，per_1x128，neox rope）与V3.2一致，
    #   所以融合在数学上等价于未融合路径。如果怀疑有回归，设为"0"回退到逐算子的Python路径。
    "ATOM_ENABLE_GLM_FUSED_INDEXER": lambda: (
        os.getenv("ATOM_ENABLE_GLM_FUSED_INDEXER", "1") == "1"
    ),
    # 【索引器】ATOM_GLM5_KPOOL：GLM-5.3的池化稀疏indexer（默认开）
    # GLM-5.3 pooled sparse indexer. Disabling it is an exact A/B only while
    # sequence length <= index_topk; the model refuses longer requests.
    # 【译】GLM-5.3的池化稀疏indexer。只有在序列长度 <= index_topk时，关闭它才是精确的A/B；更长的请求模型会拒绝。
    "ATOM_GLM5_KPOOL": lambda: os.getenv("ATOM_GLM5_KPOOL", "1") == "1",
    # 【模型】ATOM_GLM5_FORCE_DENSE_MLA / ATOM_GLM5_DISABLE_FUSED_MHC：GLM-5.3文本路径的调试开关：强制dense MLA、关闭融合mHC
    # Bring-up/debug controls for the GLM-5.3 text path.
    # 【译】GLM-5.3文本路径的bring-up/调试开关。
    "ATOM_GLM5_FORCE_DENSE_MLA": lambda: (
        os.getenv("ATOM_GLM5_FORCE_DENSE_MLA", "0") == "1"
    ),
    "ATOM_GLM5_DISABLE_FUSED_MHC": lambda: (
        os.getenv("ATOM_GLM5_DISABLE_FUSED_MHC", "0") == "1"
    ),
    # 【索引器】ATOM_M3_INDEXER_CP：MiniMax-M3只对indexer做上下文并行（覆盖dcp_config.indexer_dcp_only）；未设置时不改配置；读取方：config.py
    # MiniMax-M3 indexer-only context parallelism, as an ops override for
    # dcp_config.indexer_dcp_only. Selection is bit-identical to the TP path, so
    # this is an exact A/B: it trades a per-layer all-to-all for full MMA
    # occupancy in the block scorer, winning above ~1M batch*context tokens and
    # losing below. Unset leaves the config field alone.
    # 【译】MiniMax-M3仅indexer的上下文并行，作为dcp_config.indexer_dcp_only的运维覆盖项。选择结果与TP路径逐位相同，
    #   所以这是精确的A/B：它用每层一次all-to-all换取block scorer满额的MMA占用率，在batch*context约1M token以上时更快，
    #   以下时更慢。未设置时不改动该配置字段。
    "ATOM_M3_INDEXER_CP": lambda: os.getenv("ATOM_M3_INDEXER_CP"),
    # 科普·Mono融合decode：常规decode每层要launch几十个kernel（norm、GEMM、注意力、MoE、TP all-reduce...），小batch时launch和同步开销占大头。
    #   Mono把一整层（或层的一大段）写成一个大kernel，多卡之间的all-reduce也在kernel内部通过对端显存里的"mailbox"完成，
    #   一层一次launch。只对实测过的模型（MiniMax-M3、DeepSeek-V4.1、Kimi-K3、GLM-5.2）、TP规模和decode宽度启用，其他情况自动走普通路径。
    #   公共机制在atom/mono/，各模型的kernel在atom/models/<model>/mono/；读取方即这些目录下的config.py / dispatch.py
    # 【Mono】ATOM_MONO_ENABLE：Mono融合decode总开关（默认开）；只对测过的模型/部署/step形状生效，其他情况走生产路径
    #   【分支】Mono融合decode见第9批 atom/mono/
    # Fused decode for measured model/deployment/step geometries. Unsupported
    # configurations and forwards keep the production path. 0 disables it.
    # 【译】为测过的模型/部署/step形状启用融合decode。不支持的配置和forward保持生产路径。0关闭它。
    "ATOM_MONO_ENABLE": lambda: os.getenv("ATOM_MONO_ENABLE", "1") == "1",
    # 【Mono】ATOM_GLM52_FUSED_INDEXER：GLM-5.2 Mono里把indexer也放进融合kernel（默认关）
    # GLM-5.2 mono: run the "full" indexer layers (18 of 78) through the fused
    # MonoKernel (atom/models/glm52/mono/kernels/{kernel,layout,op}.py, built
    # with with_indexer=True), which does index Q/K/W projection, paged FP8
    # index-cache update, scoring and top-k selection inside the kernel launch
    # instead of ATOM's external Indexer module. Opt-in and off by default: the
    # measured baseline (0.9674 +/- 0.0049 on full GSM8K) is the
    # external-indexer path, and a deployment this build refuses (see
    # atom.models.glm52.mono.config.fused_indexer_refusal) falls back to it
    # unchanged. "Shared" layers (57 of 78) never use this kernel.
    # 【译】GLM-5.2 mono：让"full" indexer层（78层中的18层）通过融合的MonoKernel运行
    #   （atom/models/glm52/mono/kernels/{kernel,layout,op}.py，以with_indexer=True构建），它在kernel launch内部完成index的Q/K/W投影、
    #   分页FP8 index cache更新、打分和top-k选择，而不是用ATOM外部的Indexer模块。需主动开启，默认关闭：
    #   测得的基线（完整GSM8K上0.9674 +/- 0.0049）是外部indexer路径，而本构建拒绝的部署
    #   （见atom.models.glm52.mono.config.fused_indexer_refusal）会原样回退到它。"Shared"层（78层中的57层）从不使用这个kernel。
    "ATOM_GLM52_FUSED_INDEXER": lambda: (
        os.getenv("ATOM_GLM52_FUSED_INDEXER", "0") == "1"
    ),
    # 【Mono】ATOM_GLM52_INDEXER_CP：GLM-5.2融合indexer内部按TP rank分片打分（默认关，须同时开ATOM_GLM52_FUSED_INDEXER）
    # GLM-5.2 mono: shard index scoring and selection across the TP ranks
    # inside the fused indexer (FlyDSL PR #1256's indexer_cp, vendored at
    # 42670fb5). Each rank scores a contiguous 64-token-aligned shard of the
    # query's live context and the ranks exchange score shards through
    # in-kernel peer mailboxes; selection stays exact and the index caches
    # stay replicated -- CP distributes computation, not storage.
    #
    # Off by default, and strictly narrower than ATOM_GLM52_FUSED_INDEXER:
    # CP is only ever considered for layers the fused indexer already serves,
    # so this switch alone does nothing. What it cannot serve is refused by
    # name in atom.models.glm52.mono.config.indexer_cp_refusal, and a refusal
    # leaves the fused indexer running exactly as it does with CP off.
    # 【译】GLM-5.2 mono：在融合indexer内部把index打分和选择分片到各TP rank上（FlyDSL PR #1256的indexer_cp，vendored于42670fb5）。
    #   每个rank对query活跃上下文中一段连续的、按64 token对齐的分片打分，各rank通过kernel内的对端mailbox交换分数分片；
    #   选择结果仍然精确，index cache仍然是复制的——CP分摊的是计算，不是存储。
    #
    #   默认关闭，且严格窄于ATOM_GLM52_FUSED_INDEXER：CP只会对融合indexer已经接管的层考虑，所以单开这个开关
    #   什么也不做。它接管不了的情况由atom.models.glm52.mono.config.indexer_cp_refusal点名拒绝，
    #   被拒绝时融合indexer照常运行，与关闭CP时完全一样。
    "ATOM_GLM52_INDEXER_CP": lambda: (os.getenv("ATOM_GLM52_INDEXER_CP", "0") == "1"),
    # 【调试】ATOM_MONO_CHECK：每个mono层与原始层并排运行并记录差异
    # Debug: run each mono layer next to the original one and log the difference
    # (atom/models/minimax_m3/mono/check.py). Use with --enforce-eager.
    # 【译】调试：把每个mono层与原始层并排运行并记录差异（atom/models/minimax_m3/mono/check.py）。配合--enforce-eager使用。
    #   核实：除了minimax_m3/mono/check.py，DeepSeek-V4.1（models/deepseek_v41/mono/check.py、runner.py）也读这个开关
    "ATOM_MONO_CHECK": lambda: os.getenv("ATOM_MONO_CHECK", "0") == "1",
    # 【调试】ATOM_MONO_TRACE：把每个mono step的输入和top-2 logits追加写到这个文件
    # Debug: append every mono step's input tokens, positions and top-2 logits to
    # this file (rank 0; check.trace_logits). Use with --enforce-eager.
    # 【译】调试：把每个mono step的输入token、位置和top-2 logits追加写入这个文件（rank 0；check.trace_logits）。配合--enforce-eager使用。
    "ATOM_MONO_TRACE": lambda: os.getenv("ATOM_MONO_TRACE"),
    # 【调试】ATOM_MONO_TIMELINE：把mono kernel各阶段的时间戳保存到这个路径前缀下
    # Debug: the mono layer kernels' per-phase stamps, a few steps per decode
    # token count saved under this path prefix (mono/timeline.py). Use with
    # --enforce-eager.
    # 【译】调试：mono层kernel的各阶段时间戳，每种decode token数保存几个step，存到这个路径前缀下（mono/timeline.py）。配合--enforce-eager使用。
    #   注：英文里的mono/timeline.py实际路径是atom/mono/runtime/timeline.py；DeepSeek-V4.1和MiniMax-M3的mono runner会读它
    "ATOM_MONO_TIMELINE": lambda: os.getenv("ATOM_MONO_TIMELINE"),
    # 【调试】ATOM_MONO_DEBUG：mono kernel的mailbox等待加10秒上限，超时报错而不是让GPU挂死
    # Debug: the mono kernels bound every mailbox wait (10 s): a wait that gives up
    # records its region, pair and tags, and the runner raises with them at the end
    # of the step (mono/runner.py finish_step) instead of the GPU hanging. Use
    # with --enforce-eager.
    # 【译】调试：mono kernel对每次mailbox等待都设上限（10 s）：放弃等待时记录它的区域、配对和标签，
    #   runner在step结束时（mono/runner.py的finish_step）带着这些信息抛错，而不是让GPU挂住。配合--enforce-eager使用。
    "ATOM_MONO_DEBUG": lambda: os.getenv("ATOM_MONO_DEBUG", "0") == "1",
    # 科普·CUDA/HIP stream：GPU上的一条任务队列，同一stream里的kernel按顺序执行，不同stream里的kernel可以并发。
    #   把互不依赖的分支（如共享专家和路由专家、压缩器和indexer）放到"旁路stream"（alt_stream）上，可以让它们重叠执行，
    #   但需要额外的event同步；分支本身很小时，同步开销可能抵消收益（下面这个变量默认0就是这个原因）
    # 【模型】ATOM_DSV41_SIDE_STREAMS：DeepSeek-V4.1注意力层有几个分支放到旁路stream上并行（0/1/2，默认0）
    # DeepSeek-V4.1: how many of an attention layer's branches leave the main
    # stream. 0 none; 1 the compressor, on the MoE's `alt_stream`, waited at
    # the scorer that first reads it; 2 the indexer as well, on one of its own.
    # Default 0 because forking measured slower per layer, not faster -- the
    # periods and the noise floor under them are in the environment doc, and
    # end-to-end throughput is too coarse to see an effect that size.
    # 【译】DeepSeek-V4.1：一个注意力层有多少个分支离开主stream。0：没有；1：压缩器，放在MoE的`alt_stream`上，
    #   在第一个读取它的scorer处等待；2：indexer也离开，放在它自己的一个stream上。默认0，因为分叉测得每层更慢而不是更快——
    #   各段耗时及其下的噪声底在环境变量文档里，而端到端吞吐太粗，看不出这么小的效果。
    "ATOM_DSV41_SIDE_STREAMS": lambda: int(os.getenv("ATOM_DSV41_SIDE_STREAMS", "0")),
    # 【投机】ATOM_DSPARK_FUSED_CTX_KV：Kimi-K3 DSpark draft每层写上下文KV的多个kernel合成1个Triton kernel（默认开）
    # Kimi-K3 DSpark draft: fuse the per-layer context-row KV write
    # (K3DSparkMLAAttention.write_context_kv) into one Triton kernel --
    # RMSNorm(kv_c) + rope(k_pe) + concat + paged-cache store, versus today's
    # four launches plus a throwaway `empty_like` for the rope's query side.
    # That chain runs once per draft layer (5) per drafting step over every
    # scheduled target token, and each op re-reads the 576-wide latent from HBM
    # only to hand it to the next, so the win is launch count and round trips,
    # not FLOPs. Falls back per call when the cache layout or the rope is not
    # the plain one the kernel understands (see
    # MLAAttention.write_context_kv_latent). Measured on Kimi-K3 (MI355X, TP8,
    # fp8 KV): one 4.65us kernel replaces a 14us three-kernel chain, saving 39us
    # per drafting step at B=1 and ~36us at B=64 -- 0.1% of the step, since the
    # draft runs under a cudagraph and the launches this removes cost host time,
    # not device time. Against the per-op chain the stored latent differs on
    # ~1 element in 5M, where it is the fused kernel that matches an fp64
    # reference: aiter's RMSNorm rounds x*rstd to bf16 before applying w.
    # Set to "0" to force the per-op chain if a regression is suspected. That
    # chain is the eligibility fallback above, not debug code, so it stays
    # reachable (and runs the first write of every layer) either way.
    # 【译】Kimi-K3 DSpark draft：把每层上下文行的KV写入（K3DSparkMLAAttention.write_context_kv）融合成一个Triton kernel——
    #   RMSNorm(kv_c) + rope(k_pe) + concat + 分页cache存储，对比现在的四次launch外加给rope的query侧准备的一个用完即弃的`empty_like`。
    #   这条链在每个drafting step、每个draft层（5层）上，对每个调度的target token运行一次；每个算子都从HBM重读576宽的latent，
    #   只为了交给下一个算子，所以收益在launch次数和往返，而不在FLOPs。当cache布局或rope不是kernel能理解的普通形式时，
    #   按每次调用回退（见MLAAttention.write_context_kv_latent）。在Kimi-K3（MI355X，TP8，fp8 KV）上测得：一个4.65us的kernel
    #   替换一条14us的三kernel链，B=1时每个drafting step省39us，B=64时约36us——只占step的0.1%，因为draft在cudagraph下运行，
    #   去掉的launch花的是主机时间而不是设备时间。与逐算子链相比，存储的latent大约每500万个元素有1个不同，而此时是融合kernel
    #   与fp64参考一致：aiter的RMSNorm在乘w之前就把x*rstd舍入到bf16。
    #   如果怀疑有回归，设为"0"强制使用逐算子链。那条链是上面说的适用性回退，不是调试代码，所以无论如何都保持可达
    #   （并且每一层的第一次写入都走它）。
    "ATOM_DSPARK_FUSED_CTX_KV": lambda: (
        os.getenv("ATOM_DSPARK_FUSED_CTX_KV", "1") == "1"
    ),
    # 科普·TP all-reduce：张量并行把一个Linear的权重切到多卡，每卡算出部分和，注意力输出投影和FFN第二个GEMM之后都要all-reduce求和。
    #   all-reduce后紧接着就是"残差相加 + 下一个RMSNorm"，把三者合成一个kernel可省一次显存往返和launch。
    #   这些模型文件在模块导入时就把值存进模块级常量ENABLE_ALLREDUCE_RMSNORM_FUSION，所以必须在启动前设置
    # 【融合】ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION：TP的all-reduce与随后的残差加 + RMSNorm融合（默认开）
    #   读取方：models/deepseek_v2.py、qwen3_moe.py、glm4_moe.py、minimax_m2.py等模型文件（模块导入时读取）
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1") == "1"
    ),
    # 【投机】ATOM_DSPARK_FUSED_MARKOV_SAMPLE：DSpark块采样里Markov头的偏置GEMV + argmax融合成一个Triton kernel（默认开）
    # DSpark block sampling: replace the Markov head's
    #   bias = W1[x] @ W2.float().t() ; argmax(base_logits + bias)
    # with one fused Triton kernel (atom/model_ops/dspark_markov_sample.py).
    # The unfused spelling casts the whole [V, r] W2 table to fp32 INSIDE the
    # T-iteration loop (~252MB of traffic per iteration for a table that never
    # changes) and materializes two [B, V] fp32 tensors per iteration (42MB
    # each at B=64, V=163840) that only an argmax ever reads. The fused path
    # keeps W2 bf16, adds the base logits in the GEMM epilogue and reduces to
    # ids in registers, so W2 is read exactly once per block position and no
    # [B, V] intermediate exists. Covers both native DSpark block samplers,
    # Kimi-K3 (r=256) and DeepSeek-V4 (r=512); the op is shape-generic and
    # hands anything it cannot index back to the reference, but only K3 has
    # been run on hardware.
    # The bias GEMV moves from an fp32 matmul over an fp32 copy of W2 to bf16
    # MFMA with an fp32 accumulator. Every product is exact in fp32 either way,
    # so the result is equal to the reference up to accumulation order; whether
    # an argmax anywhere flips on a last-ulp tie is empirical. Measured on
    # Kimi-K3 (MI355X, TP8, fp8 KV, full GSM8K at 64 concurrency): acceptance
    # 87.08% against 87.06% unfused, accept-length distribution equal to within
    # 0.1pp, flexible-extract inside the run-to-run band (0.9477 / 0.9591 fused,
    # 0.9522 unfused). Saves 145 us per drafting step at B=1 and ~235 us at
    # B=64. Set to "0" to force the reference spelling if an acceptance-rate
    # regression is suspected -- the fastest way to rule this kernel in or out,
    # since the two paths are not bit-identical by construction. Read once when
    # the Markov head is built, so it must be set before the server starts.
    # 【译】DSpark块采样：把Markov头的
    #     bias = W1[x] @ W2.float().t() ; argmax(base_logits + bias)
    #   换成一个融合的Triton kernel（atom/model_ops/dspark_markov_sample.py）。未融合的写法在T次迭代的循环*内部*把整个[V, r]的W2表
    #   转成fp32（一张永不变化的表，每次迭代约252MB流量），并且每次迭代物化两个[B, V]的fp32张量（B=64、V=163840时各42MB），
    #   而它们只被一个argmax读取。融合路径保持W2为bf16，在GEMM的epilogue里加上base logits，并在寄存器里归约成id，
    #   所以W2在每个块位置只读一次，也不存在任何[B, V]中间结果。覆盖两个原生DSpark块采样器：Kimi-K3（r=256）和DeepSeek-V4（r=512）；
    #   这个算子与形状无关，无法处理的情况会交回参考实现，但只有K3在硬件上跑过。
    #   偏置GEMV从对W2的fp32副本做fp32 matmul，变成带fp32累加器的bf16 MFMA。两种方式下每个乘积在fp32里都是精确的，
    #   所以结果与参考实现相等，只差累加顺序；某处的argmax会不会因最后一位的平局而翻转，只能靠实测。在Kimi-K3
    #   （MI355X，TP8，fp8 KV，64并发的完整GSM8K）上测得：接受率87.08%对比未融合的87.06%，接受长度分布相差在0.1个百分点以内，
    #   flexible-extract在运行间波动范围内（融合0.9477 / 0.9591，未融合0.9522）。B=1时每个drafting step省145 us，B=64时约235 us。
    #   如果怀疑接受率回归，设为"0"强制使用参考写法——这是确认或排除这个kernel最快的办法，因为两条路径按构造就不是逐位一致的。
    #   在构建Markov头时读取一次，所以必须在启动服务前设置。
    "ATOM_DSPARK_FUSED_MARKOV_SAMPLE": lambda: (
        os.getenv("ATOM_DSPARK_FUSED_MARKOV_SAMPLE", "1") == "1"
    ),
    # 【并行】ATOM_REPLICATE_VOCAB_EMBED：词表embedding在每个TP rank上完整复制，省掉embedding之后的all-reduce（默认开）
    # Replicate the vocab embedding on every TP rank (full table per rank, purely
    # local lookup) instead of TP-sharding it — eliminates the post-embedding
    # all-reduce. Applies to BOTH the main/target model and the speculative draft
    # (EAGLE3 head + MTP draft steps), so the collective is dropped on every embed
    # on the critical path. Only used where the embedding is independent of the
    # still TP-sharded lm_head (EAGLE3 draft; GLM-5.2 target+MTP, whose
    # tie_word_embeddings=False), so the lookup is bit-identical to the sharded
    # masked-embedding + all-reduce path. Trades (tp-1)/tp of the embedding's
    # memory per rank for one fewer collective per embed. Default on; set "0" to
    # fall back to the sharded VocabParallelEmbedding.
    # 【译】在每个TP rank上复制词表embedding（每个rank一张完整表，纯本地查表），而不是按TP切分——省掉embedding之后的all-reduce。
    #   同时作用于主模型/target模型和投机draft（EAGLE3头 + MTP draft步），所以关键路径上每次embed的集合通信都被去掉。
    #   只用在embedding与仍按TP切分的lm_head相互独立的地方（EAGLE3 draft；GLM-5.2的target+MTP，它们tie_word_embeddings=False），
    #   因此查表结果与切分的masked-embedding + all-reduce路径逐位相同。用每个rank (tp-1)/tp的embedding显存换每次embed少一次集合通信。
    #   默认开启；设为"0"回退到切分的VocabParallelEmbedding。
    "ATOM_REPLICATE_VOCAB_EMBED": lambda: (
        os.getenv("ATOM_REPLICATE_VOCAB_EMBED", "1") == "1"
    ),
    # 【状态缓存】ATOM_ENABLE_GDN_DECODE_LOSSY_FAST：GDN纯decode步走一条有损的快速路径（默认关）
    #   读取方：attention_gdn.py——只在没有ReplaySSM、没有投机、batch里没有prefill时生效；它每步写回完整状态，与ReplaySSM二选一
    "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST": lambda: (
        os.getenv("ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", "0").lower() == "1"
    ),
    # --- ReplaySSM (linear-attention state) ---
    # 【译】--- ReplaySSM（线性注意力状态）---
    # 科普·ReplaySSM：投机解码一步要验证k+1个token，线性注意力（GDN/KDA）的状态S会被它们依次更新；
    #   后面的草稿被拒绝时，要退回到"只接受了前几个token"时的状态。以k=3为例：
    #
    #       朴素做法：  S0 → S1 → S2 → S3 → S4           每个位置存一份完整状态矩阵（k+1 = 4份），拒绝后挑对应那份
    #       ReplaySSM： S0 + 记录[(k,u,g)₁ (k,u,g)₂ (k,u,g)₃ (k,u,g)₄]   只记每个token的输入；拒绝 = 把游标往回挪，
    #                   需要时从S0按记录重放出S_i；记录攒满L个（ATOM_REPLAYSSM_CACHE_LEN）才把完整状态写回一次
    #
    #   状态池于是不再随投机宽度k成倍增长（数字见下面英文注释里Kimi-K3的例子）
    # 【状态缓存】ATOM_ENABLE_REPLAYSSM：投机解码时缓存SSM的输入，而不是每个投机token一份完整状态
    #   未设置（None）时跟随是否开启投机解码：有投机就开、没有就关；设置后强制开/关（gdn_attn.py）
    # Cache the SSM *inputs* (k, u, g) instead of one full recurrent state per
    # speculative token.  The state pool stops scaling with the MTP window --
    # e.g. Kimi-K3 at mtp_k=2, 64 seqs, tp=8 drops from 10.3 GiB to ~3.5 GiB --
    # and the full-state write moves off the per-step path (rewritten only when
    # the record buffer fills).  Rollback becomes a cursor move.
    # See atom/model_ops/fla_ops/replayssm.py.
    # 【译】缓存SSM的*输入*（k, u, g），而不是每个投机token一份完整的递归状态。状态池不再随MTP窗口扩大——
    #   例如Kimi-K3在mtp_k=2、64条序列、tp=8时从10.3 GiB降到约3.5 GiB——完整状态的写入也移出了每步的路径
    #   （只在记录缓冲区写满时才重写）。回滚变成移动一个游标。见atom/model_ops/fla_ops/replayssm.py。
    "ATOM_ENABLE_REPLAYSSM": lambda: (
        None
        if os.getenv("ATOM_ENABLE_REPLAYSSM") is None
        else os.getenv("ATOM_ENABLE_REPLAYSSM", "0").lower() == "1"
    ),
    # 【状态缓存】ATOM_REPLAYSSM_CACHE_LEN：ReplaySSM记录缓冲区深度L（默认16）
    # Record-buffer depth L.  Must be >= 2*(mtp_k+1); raised automatically if
    # set too low.  Larger L means fewer checkpoint write-backs but a longer
    # rebuild each step; upstream measured 8-16 as the sweet spot.
    # 【译】记录缓冲区深度L。必须>= 2*(mtp_k+1)；设得太低会被自动调高。L越大，checkpoint写回越少，但每步重建越长；
    #   上游测得8-16是最佳区间。
    "ATOM_REPLAYSSM_CACHE_LEN": lambda: int(
        os.getenv("ATOM_REPLAYSSM_CACHE_LEN", "16")
    ),
    # 【状态缓存】ATOM_REPLAYSSM_ROUTE：ReplaySSM验证阶段的计算路径：auto / serial / ut
    # "auto" | "serial" | "ut".  The UT-transform verify route only beats the
    # serial one at verify windows >= ~12 tokens (measured on gfx950), so
    # "auto" keeps practical MTP windows on the serial route.
    # 【译】"auto" | "serial" | "ut"。UT变换的verify路径只在verify窗口>=约12个token时才比serial路径快（在gfx950上测得），
    #   所以"auto"让实际使用的MTP窗口走serial路径。
    "ATOM_REPLAYSSM_ROUTE": lambda: os.getenv("ATOM_REPLAYSSM_ROUTE", "auto").lower(),
    # 【状态缓存】ATOM_GDN_SSM_DTYPE：线性注意力状态池的存储精度，fp32（默认）/ fp16 / bf16；读取方：gdn_attn.py、kimi_mla_gdn_attn.py、Kimi-K3 mono
    # "fp32" | "fp16" | "bf16".  Storage dtype of the KDA temporal state pool,
    # whose per-token traffic dominates KDA decode; the recurrence itself
    # always accumulates in fp32.  fp16 over bf16 when narrowing: the state is
    # O(1), so bf16's range buys nothing and its short mantissa costs accuracy.
    # 【译】"fp32" | "fp16" | "bf16"。KDA时间状态池的存储dtype，它的每token流量主导了KDA decode；递归计算本身总是用fp32累加。
    #   要收窄时选fp16而不是bf16：状态是O(1)量级，bf16的范围优势用不上，它短的尾数反而损失精度。
    "ATOM_GDN_SSM_DTYPE": lambda: os.getenv("ATOM_GDN_SSM_DTYPE", "fp32").lower(),
    # 【融合】ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT / ..._SILU_MUL_QUANT：Llama里RMSNorm + 量化、SiLU*Mul + 量化
    #   用aiter Triton融合kernel（默认都开）；读取方：models/llama.py
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT", "1") == "1"
    ),
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT", "1") == "1"
    ),
    # 【融合】ATOM_USE_MODEL_SENSITIVE_RMSNORM：作为use_model_sensitive_rmsnorm参数传给aiter的RMSNorm kernel（默认关）；读取方：layernorm.py
    "ATOM_USE_MODEL_SENSITIVE_RMSNORM": lambda: (
        os.getenv("ATOM_USE_MODEL_SENSITIVE_RMSNORM", "0") == "1"
    ),
    # --- Shutdown ---
    # 【译】--- 关闭 ---
    # 【服务】ATOM_SHUTDOWN_TIMEOUT_S：关闭时等待子进程退出的秒数（默认5），超时就强制终止；读取方：async_proc.py、engine_core.py、engine_core_mgr.py
    # How long a shutting-down process waits for its children (EngineCore,
    # ModelRunner workers) to exit before it terminates them.
    # 【译】正在关闭的进程等待它的子进程（EngineCore、ModelRunner worker）退出多久，超时就终止它们。
    "ATOM_SHUTDOWN_TIMEOUT_S": lambda: _positive_float_env(
        "ATOM_SHUTDOWN_TIMEOUT_S", "5"
    ),
    # --- Profiling & Logging ---
    # 【译】--- profiling与日志 ---
    # 【服务】ATOM_METRICS_UPDATE_INTERVAL_S：EngineCore推送指标、API server更新指标的间隔秒数（默认1.0）
    "ATOM_METRICS_UPDATE_INTERVAL_S": lambda: _positive_float_env(
        "ATOM_METRICS_UPDATE_INTERVAL_S", "1.0"
    ),
    # 【服务】ATOM_ENABLE_METRICS_DEVICE_TIMER：用CUDA event给forward计GPU时间并计入指标（默认关）；读取方：model_runner.py、engine_utility.py
    "ATOM_ENABLE_METRICS_DEVICE_TIMER": lambda: os.getenv(
        "ATOM_ENABLE_METRICS_DEVICE_TIMER", "0"
    )
    == "1",
    # 【调试】ATOM_TORCH_PROFILER_DIR：torch profiler的trace输出目录；config.py里Config.torch_profiler_dir字段的默认值取自它
    "ATOM_TORCH_PROFILER_DIR": lambda: os.getenv("ATOM_TORCH_PROFILER_DIR", None),
    # 【服务】ATOM_GC_FREEZE：预热完成后把启动期创建的大量对象移入Python GC的永久代，之后GC不再扫描它们（默认开）
    # Move the startup heap (model, compiled graph, tokenizer, KV block pool)
    # into CPython's permanent generation once warmup is done, so collections
    # stop scanning it.  On by default; set 0 to keep the old behaviour.
    # See freeze_gc_heap in atom/utils/gc_utils.py.
    # 【译】预热完成后，把启动期的堆（模型、编译后的图、tokenizer、KV block池）移进CPython的永久代，这样之后的垃圾回收不再扫描它。
    #   默认开启；设为0保持旧行为。见atom/utils/gc_utils.py里的freeze_gc_heap。
    "ATOM_GC_FREEZE": lambda: os.getenv("ATOM_GC_FREEZE", "1") == "1",
    # 【服务】ATOM_GC_DEBUG：记录每次垃圾回收的代、耗时和回收的对象数
    # Log every garbage collection: generation, duration, objects reclaimed.
    # 【译】记录每次垃圾回收：代数、耗时、回收的对象数。
    "ATOM_GC_DEBUG": lambda: os.getenv("ATOM_GC_DEBUG", "0") == "1",
    # 【服务】ATOM_GC_THRESHOLD："t0,t1,t2"形式的gc.set_threshold参数，空则保持CPython默认
    # "t0,t1,t2" for gc.set_threshold(); empty keeps CPython's default.
    # Thresholds are per-interpreter, but this is one variable, read by every
    # process that serves -- and only the frontend was measured to reclaim
    # nothing, so setting it tunes three others blind. t0 must be >= 1: zero
    # stops collection while the watch still reports the healthy shape.
    # See tune_gc in atom/utils/gc_utils.py.
    # 【译】给gc.set_threshold()的"t0,t1,t2"；为空则保持CPython的默认值。阈值是按解释器设置的，但这只是一个变量，
    #   每个参与服务的进程都会读取它——而只有前端进程被测出什么都回收不到，所以设置它等于盲调另外三个进程。
    #   t0必须>= 1：设为0会停止回收，而监控仍会报告一切正常的形态。见atom/utils/gc_utils.py里的tune_gc。
    "ATOM_GC_THRESHOLD": lambda: os.getenv("ATOM_GC_THRESHOLD", "").strip(),
    # 【服务】ATOM_DETOKENIZER_DELTA_REUSE：流式输出的增量反分词是否复用上次的结果，省一次decode（auto / on / off）
    # Whether the incremental detokenizer may reuse the delta it last emitted
    # in place of one of its two decodes per update. "auto" verifies at startup
    # that this tokenizer decodes a token span the same way wherever the window
    # starts; "on"/"off" pin it. Unrelated to the KV prefix cache.
    # See enable_delta_reuse in atom/entrypoints/openai/streaming_dispatch.py.
    # 【译】增量反分词器是否可以复用它上次输出的增量，代替每次更新时两次decode中的一次。"auto"在启动时验证这个tokenizer
    #   对一段token的解码结果与窗口从哪里开始无关；"on"/"off"固定它。与KV前缀缓存无关。
    #   见atom/entrypoints/openai/streaming_dispatch.py里的enable_delta_reuse。
    "ATOM_DETOKENIZER_DELTA_REUSE": lambda: os.getenv(
        "ATOM_DETOKENIZER_DELTA_REUSE", "auto"
    )
    .strip()
    .lower(),
    # 【服务】ATOM_DETOKENIZER_AUDIT_EVERY：开启复用后，每隔多少次更新用真实decode核对一次
    # How often the reused delta is checked against a real decode once reuse is
    # on. 1 checks every update; a mismatch corrects that call's output and
    # turns reuse off for the process.  Text, not int, for the reason
    # ATOM_GC_THRESHOLD is: it is read as an argument at a callsite that has
    # already loaded the weights, so it is parsed where a bad value can be
    # answered with a warning instead of a traceback.
    # 【译】开启复用后，多久用一次真实decode核对复用的增量。1表示每次更新都核对；发现不一致时修正这次调用的输出，
    #   并在本进程内关闭复用。之所以是文本而不是int，原因与ATOM_GC_THRESHOLD相同：它在一个已经加载完权重的调用点作为参数读取，
    #   所以在那里解析，坏值可以用一条警告回应，而不是一个traceback。
    "ATOM_DETOKENIZER_AUDIT_EVERY": lambda: os.getenv(
        "ATOM_DETOKENIZER_AUDIT_EVERY", ""
    ).strip(),
    # 【调试】ATOM_PROFILER_MORE：一次打开下面三个profiler选项（记录形状、调用栈、显存）；读取方：ModelRunner.start_profiler
    #   ATOM_PROFILER_RECORD_SHAPES / WITH_STACK / PROFILE_MEMORY：分别控制这三个选项；单独设置时优先，未设置时跟随ATOM_PROFILER_MORE
    "ATOM_PROFILER_MORE": lambda: os.getenv("ATOM_PROFILER_MORE", "0") == "1",
    "ATOM_PROFILER_RECORD_SHAPES": lambda: _profiler_detail_env(
        "ATOM_PROFILER_RECORD_SHAPES"
    ),
    "ATOM_PROFILER_WITH_STACK": lambda: _profiler_detail_env(
        "ATOM_PROFILER_WITH_STACK"
    ),
    "ATOM_PROFILER_PROFILE_MEMORY": lambda: _profiler_detail_env(
        "ATOM_PROFILER_PROFILE_MEMORY"
    ),
    # 【调试】ATOM_ENABLE_DETAILED_ANNOTATION：profiling时在trace标签里附加注意力的统计量
    # When profiling is active, append detailed attention aggregates (sqsq, sqsk, sk)
    # to the prefill[]/decode[] trace labels emitted by ModelRunner.run_model.
    # 【译】profiling开启时，在ModelRunner.run_model发出的prefill[]/decode[] trace标签后附加注意力的详细汇总量（sqsq、sqsk、sk）。
    "ATOM_ENABLE_DETAILED_ANNOTATION": lambda: (
        os.getenv("ATOM_ENABLE_DETAILED_ANNOTATION", "0") == "1"
    ),
    # 【调试】ATOM_PROFILER_TIMEOUT：LLMEngine.stop_profile等待各rank写完trace的超时秒数（默认300）
    "ATOM_PROFILER_TIMEOUT": lambda: float(os.getenv("ATOM_PROFILER_TIMEOUT", "300")),
    # 【服务】ATOM_LOG_MORE：非0时日志用详细格式（进程名、pid、文件路径行号、函数名）；读取方：utils/__init__.py的getLogger
    "ATOM_LOG_MORE": lambda: int(os.getenv("ATOM_LOG_MORE", "0")) != 0,
    # 【调试】ATOM_RTL_TRACE_DIR：rocm-trace-lite的GPU kernel trace输出目录；atom/下的Python代码不读它，读取方是CI脚本.github/scripts/atom_test.sh
    # RTL (rocm-trace-lite) GPU kernel tracing — set to output directory to enable.
    # When set, the server launch is wrapped with `rtl trace` to collect per-kernel
    # GPU timestamps for both prefill and decode phases.
    # 【译】RTL（rocm-trace-lite）GPU kernel追踪——设为输出目录即开启。设置后，服务启动会被包在`rtl trace`里，
    #   为prefill和decode两个阶段收集每个kernel的GPU时间戳。
    "ATOM_RTL_TRACE_DIR": lambda: os.getenv("ATOM_RTL_TRACE_DIR", None),
    # --- Model Loading ---
    # 【译】--- 模型加载 ---
    # 【加载】ATOM_DISABLE_MMAP：读safetensors时不用mmap（值为"true"才生效，默认false）；读取方：model_loader/*
    "ATOM_DISABLE_MMAP": lambda: (
        os.getenv("ATOM_DISABLE_MMAP", "false").lower() == "true"
    ),
    # 【加载】ATOM_LOADER_NUM_THREADS：权重加载的工作线程数（默认16）；读取方：model_loader/loading_core.py
    # Worker threads for weight loading. >1 (default 16) enables the batched
    # parallel loader (per-fused-param CPU staging flushed with one H2D copy)
    # with that many threads; set to 1 to fall back to the original sequential
    # per-expert path.
    # 【译】权重加载的工作线程数。>1（默认16）时用这么多线程启用批量并行加载器（按融合参数在CPU上暂存，再用一次H2D拷贝刷出）；
    #   设为1回退到原来逐专家的顺序路径。
    "ATOM_LOADER_NUM_THREADS": lambda: int(os.getenv("ATOM_LOADER_NUM_THREADS", "16")),
    # 【加载】ATOM_LOADER_PREFETCH：后台线程顺序预读checkpoint文件进page cache，加快冷启动（默认开，值为"true"才算开）
    # Warm the page cache with a background sequential reader instead of
    # leaving it to demand faults through the mmap. Measured on a local NVMe:
    # the fault-driven pattern sustains 3.2 GB/s where the device does 6.9 and
    # a single sequential reader alone reaches 6.06, so the gap is the access
    # pattern rather than queue depth. On DeepSeek-R1 MXFP4 (350 GiB, TP=4)
    # this takes a cold load from ~156s to ~68s and leaves the warm case
    # slightly better too, so it is on by default; turn it off for storage
    # where a second sequential reader competes with the loader instead of
    # feeding it.
    # 【译】用一个后台顺序读取线程预热page cache，而不是靠mmap的缺页按需读取。在本地NVMe上测得：缺页驱动的模式只能维持3.2 GB/s，
    #   设备能达到6.9，单个顺序读取线程就能达到6.06，所以差距在访问模式而不是队列深度。在DeepSeek-R1 MXFP4（350 GiB，TP=4）上，
    #   它把冷加载从约156s降到约68s，热加载也略有改善，所以默认开启；如果存储上第二个顺序读取者会与加载器争抢而不是喂给它，就关掉它。
    "ATOM_LOADER_PREFETCH": lambda: (
        os.getenv("ATOM_LOADER_PREFETCH", "true").lower() == "true"
    ),
    # 【加载】ATOM_LOADER_PREFETCH_THREADS：预读时并发读取的分片数（默认4）
    # Shards read concurrently by the prefetcher. Kept small: the device here
    # saturates at 2 streams, and every thread also competes with the loader.
    # 【译】预取线程同时读取的分片数。保持较小：这里的设备2个流就饱和了，而且每个线程也在和加载器争抢。
    "ATOM_LOADER_PREFETCH_THREADS": lambda: int(
        os.getenv("ATOM_LOADER_PREFETCH_THREADS", "4")
    ),
    # 【加载】ATOM_LOADER_PREFETCH_BLOCK_MB：预读每次读取的块大小（MB，默认16）
    "ATOM_LOADER_PREFETCH_BLOCK_MB": lambda: int(
        os.getenv("ATOM_LOADER_PREFETCH_BLOCK_MB", "16")
    ),
    # 【加载】ATOM_LOADER_FADVISE：用fadvise(WILLNEED)提示内核预读（默认关；开了预读线程时被忽略）
    # Hint the kernel to read each shard ahead. Off by default because it never
    # paid for itself once measured on its own: the read loop runs far ahead of
    # the workers, so WILLNEED is issued for the whole checkpoint within
    # seconds, and asking for 350 GiB of read-ahead is bookkeeping the kernel
    # largely discards. Cold load 193.0s vs 194.2s -- no effect; warm load
    # 27.3s vs 42.8s -- 15.5s of pure overhead. Kept as a switch rather than
    # deleted so the behaviour can be restored on storage that behaves
    # differently. Ignored when prefetching, which supersedes it.
    # 【译】提示内核对每个分片做预读。默认关闭，因为单独测量时它从来没有带来收益：读取循环远远跑在worker前面，
    #   所以几秒钟内就对整个checkpoint发出了WILLNEED，而要求350 GiB的预读只是内核大量丢弃的簿记工作。
    #   冷加载193.0s对比194.2s——没有效果；热加载27.3s对比42.8s——纯开销15.5s。保留为开关而不是删除，
    #   以便在表现不同的存储上恢复这一行为。开启预取时被忽略，预取取代了它。
    "ATOM_LOADER_FADVISE": lambda: (
        os.getenv("ATOM_LOADER_FADVISE", "false").lower() == "true"
    ),
    # 【加载】ATOM_LOADER_STRICT_COVERAGE：checkpoint缺少某些路由专家的权重时直接报错（默认开）
    # Fail loading when the checkpoint does not deliver every routed expert of
    # a fused MoE parameter. On by default: the alternative is a model that
    # loads happily with some expert slots left at their init values, which
    # only shows up much later as an accuracy drop. Set to false to downgrade
    # to a warning when bringing up a checkpoint that is known to be partial.
    # 【译】当checkpoint没有提供某个融合MoE参数的全部路由专家时，让加载失败。默认开启：否则模型会愉快地加载完，
    #   某些专家槽位保持初始化值，而这只会在很久以后表现为精度下降。启动一个已知不完整的checkpoint时，设为false降级为警告。
    "ATOM_LOADER_STRICT_COVERAGE": lambda: (
        os.getenv("ATOM_LOADER_STRICT_COVERAGE", "true").lower() == "true"
    ),
    # 科普·在线量化：checkpoint是BF16，但想用FP8/FP4跑时，ATOM可以在加载时现场量化（而不是事先离线转换checkpoint）。
    #   默认做法是先把BF16权重整个加载到显存再量化，峰值显存高；"流式"则逐模块加载即量化即释放。读取方：model_loader/online_quant_streaming.py、quant_spec.py
    # 【加载】ATOM_ONLINE_QUANT_STREAMING：在线量化时边加载边量化，降低显存峰值（默认关）
    # Quantize eligible modules as they load to reduce peak memory. Streaming
    # quantizes local TP shards, so results may differ slightly from offline.
    # 【译】在加载的同时量化符合条件的模块，以降低峰值内存。流式模式量化的是本地TP分片，所以结果可能与离线量化略有差异。
    "ATOM_ONLINE_QUANT_STREAMING": lambda: (
        os.getenv("ATOM_ONLINE_QUANT_STREAMING", "0").lower() in ("1", "true")
    ),
    # 【加载】ATOM_ONLINE_QUANT_STREAMING_THREADS：流式在线量化的尾部worker数（默认4）
    # Tail workers for H2D, quantization, and source release. More workers
    # increase overlap and in-flight memory; 0 runs inline.
    # 【译】负责H2D、量化和释放源数据的尾部worker数。worker越多，重叠越多，在途内存也越多；0表示内联执行。
    "ATOM_ONLINE_QUANT_STREAMING_THREADS": lambda: int(
        os.getenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "4")
    ),
    # 【加载】ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING：流式在线量化先在主机上暂存（默认开）
    # Stage on the host and upload once per completed parameter. Disabling this
    # buffers loader calls on meta and forces a single-threaded checkpoint walk.
    # 【译】在主机上暂存，每个参数完整后上传一次。关闭它会把加载器调用缓冲在meta设备上，并强制单线程遍历checkpoint。
    "ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING": lambda: (
        os.getenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1").lower()
        in ("1", "true")
    ),
    # --- Attention Backend ---
    # 【译】--- 注意力后端 ---
    # 【注意力】ATOM_USE_UNIFIED_ATTN：MHA的分页/prefill注意力用unified_attention（flash风格），而不是pa_decode_gluon
    # Use unified_attention (flash-style) for MHA paged/prefill attention instead
    # of pa_decode_gluon. Set to 1 to enable the unified_attention path.
    # 【译】MHA的分页/prefill注意力使用unified_attention（flash风格），而不是pa_decode_gluon。设为1启用unified_attention路径。
    "ATOM_USE_UNIFIED_ATTN": lambda: os.getenv("ATOM_USE_UNIFIED_ATTN", "0") == "1",
    # 【注意力】ATOM_FORCE_ATTN_TRITON：能回退到Triton的注意力都强制用Triton，绕开ASM/OPUS快速路径（调试用）
    # Force Triton attention fallbacks where available. Set to 1 to bypass
    # optional ASM/OPUS fast paths during debugging.
    # 【译】在有Triton后备实现的地方强制使用它。设为1可在调试时绕过可选的ASM/OPUS快速路径。
    "ATOM_FORCE_ATTN_TRITON": lambda: (os.getenv("ATOM_FORCE_ATTN_TRITON", "0") == "1"),
    # 【注意力】ATOM_FORCE_V4_PREFILL_OPUS：DeepSeek-V4 fp8稀疏prefill强制用OPUS kernel，而不是aiter的ASM kernel
    # Force the OPUS kernel for DeepSeek-V4 fp8 sparse prefill instead of the
    # aiter asm kernel (`mla_sparse_prefill_fp8_asm`, the default). Escape hatch
    # for the asm path; that kernel is gfx1250-only and hard-requires H == 128,
    # so smaller local head counts fall back to OPUS regardless of this flag.
    # 【译】DeepSeek-V4 fp8稀疏prefill强制使用OPUS kernel，而不是aiter的asm kernel（`mla_sparse_prefill_fp8_asm`，默认）。
    #   这是asm路径的逃生开关；那个kernel只支持gfx1250并且硬性要求H == 128，所以本地头数更少时无论这个开关如何都会回退到OPUS。
    "ATOM_FORCE_V4_PREFILL_OPUS": lambda: (
        os.getenv("ATOM_FORCE_V4_PREFILL_OPUS", "0") == "1"
    ),
    # 【注意力】ATOM_USE_V4_PREFILL_ASM_FOR_DECODE：DeepSeek-V4 fp8 decode复用gfx1250上H=128的稀疏prefill ASM kernel（默认关）
    # Reuse the gfx1250 H=128 sparse-prefill ASM kernel for DeepSeek-V4 fp8
    # decode. Ineligible shapes keep the dedicated decode ASM path.
    # 【译】DeepSeek-V4 fp8 decode复用gfx1250上H=128的稀疏prefill ASM kernel。不符合条件的形状保持专用的decode ASM路径。
    "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE": lambda: (
        os.getenv("ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", "0") == "1"
    ),
    # 【注意力】ATOM_V4_HCA_PERSIST：DeepSeek-V4 HCA层（压缩比128）的fp8 decode用aiter的persistent V4-NM kernel（默认开）
    #   CSA/HCA是什么见config.py __post_init__第12步V4部分的【科普】
    # DeepSeek-V4 HCA (compress_ratio 128) fp8 decode through aiter's
    # persistent V4-NM kernel (one launch, in-kernel split + merge) instead of
    # the decode ASM + split plan. gfx950, 128 local heads only; quietly off
    # when aiter lacks mla_decode_fwd_v4_nm_ps.
    # 【译】DeepSeek-V4 HCA（compress_ratio 128）的fp8 decode走aiter的persistent V4-NM kernel（一次launch，kernel内拆分+合并），
    #   而不是decode ASM + 拆分计划。只用于gfx950、本地128个头；aiter没有mla_decode_fwd_v4_nm_ps时静默关闭。
    "ATOM_V4_HCA_PERSIST": lambda: os.getenv("ATOM_V4_HCA_PERSIST", "1") == "1",
    # 【注意力】ATOM_V4_HCA_PERSIST_MIN_ROWS：q行数少于它（默认15）时仍走ASM路径
    # Calls with fewer q rows stay on the ASM path, which is faster there.
    # 【译】q行数更少的调用留在ASM路径上，那里它更快。
    "ATOM_V4_HCA_PERSIST_MIN_ROWS": lambda: int(
        os.getenv("ATOM_V4_HCA_PERSIST_MIN_ROWS", "15")
    ),
    # 【注意力】ATOM_PA_FLYDSL：分页decode注意力改用aiter的FlyDSL kernel（带work planner），默认关
    # Route the paged decode to aiter's FlyDSL kernel (with its work planner)
    # instead of gluon. Needs aiter with #5809.
    # 【译】把分页decode路由到aiter的FlyDSL kernel（带它的work planner），而不是gluon。需要包含#5809的aiter。
    "ATOM_PA_FLYDSL": lambda: (os.getenv("ATOM_PA_FLYDSL", "0") == "1"),
    # 【插件】ATOM_USE_GLUON_PA_DECODE：vLLM插件的MHA层decode用gluon的paged attention；读取方：plugin/vllm/attention/layer_mha.py
    # Use gluon pa decode for some models
    # 【译】某些模型使用gluon的pa decode
    "ATOM_USE_GLUON_PA_DECODE": lambda: (
        os.getenv("ATOM_USE_GLUON_PA_DECODE", "0") == "1"
    ),
    # --- Plugin Mode ---
    # 【译】--- 插件模式 ---
    # 【插件】ATOM_DISABLE_VLLM_PLUGIN：设为1时关闭ATOM的vLLM插件模式；读取方：plugin/vllm/register.py、platform.py
    "ATOM_DISABLE_VLLM_PLUGIN": lambda: (
        os.getenv("ATOM_DISABLE_VLLM_PLUGIN", "0").lower() == "1"
    ),
    # 【插件】ATOM_V4_CG_INSTR_DIR：vLLM插件中DeepSeek-V4的cudagraph分派计数器输出目录（诊断用）
    # vLLM plugin, DeepSeek-V4: directory for per-process cudagraph dispatch
    # counters (modes, overridden steps, guard trips), one JSON file per
    # process, rewritten every step. Unset: off.
    # 【译】vLLM插件，DeepSeek-V4：每个进程的cudagraph分派计数器（模式、被覆盖的step、保护触发次数）的输出目录，
    #   每个进程一个JSON文件，每步重写。未设置：关闭。
    "ATOM_V4_CG_INSTR_DIR": lambda: os.getenv("ATOM_V4_CG_INSTR_DIR") or None,
    # 【并行】ATOM_USE_CUSTOM_ALL_GATHER：TP all-gather（如lm_head的logits、indexer CP）用自定义通信实现（ca_comm），默认开
    "ATOM_USE_CUSTOM_ALL_GATHER": lambda: (
        os.getenv("ATOM_USE_CUSTOM_ALL_GATHER", "1").lower() == "1"
    ),
    # 【并行】ATOM_DP_LM_HEAD_MODE：纯DP decode时lm_head的计算方式：all2all（默认）/ allgather / default
    # Pure-DP LM head strategy (only active when the model TP group is size 1 and
    # the DP group is size > 1, i.e. pure DP attention on decode; every other
    # case — TP>1, DP off, prefill, ragged/draft rows — auto-falls back to the
    # replicated path via `_can_use_dp_sharded_head`):
    #   "all2all"   -> shard vocab across the DP group + gather hidden, then a
    #                  single all-to-all delivers each rank its own rows x full
    #                  vocab (config ③, minimal comm). DEFAULT.
    #   "allgather" -> same shard/gather, but vocab all-gather + scatter rows
    #                  (config ②); kept for A/B and as a fallback.
    #   "default"   -> replicated full-vocab GEMM per DP rank (big GEMM, no comm);
    #                  explicit opt-out.
    # 【译】纯DP时的LM head策略（只在模型TP组大小为1、DP组大小>1时生效，即decode时的纯DP attention；其他所有情况——
    #   TP>1、DP关闭、prefill、不规则/draft行——都通过`_can_use_dp_sharded_head`自动回退到复制路径）：
    #     "all2all"   -> 在DP组内切分词表 + gather hidden，然后一次all-to-all把每个rank自己的行 x 完整词表交给它
    #                    （配置③，通信最少）。默认。
    #     "allgather" -> 同样的切分/gather，但改为词表all-gather + 行scatter（配置②）；保留用于A/B和作为后备。
    #     "default"   -> 每个DP rank做复制的完整词表GEMM（大GEMM，无通信）；显式退出。
    "ATOM_DP_LM_HEAD_MODE": lambda: os.getenv(
        "ATOM_DP_LM_HEAD_MODE", "all2all"
    ).lower(),
    # 【投机】ATOM_DP_DRAFT_ARGMAX：纯DP下draft的贪心argmax按词表分片到各DP rank（默认开）
    # Pure-DP draft greedy argmax: shard the draft lm_head vocab across the DP
    # group so each rank reads [H, V/dp] instead of the full [H, V], exchanging
    # only the packed [N, 2]. Active only for a unified (rectangular) pure-DP
    # draft step; everything else falls back to the replicated local argmax.
    # 【译】纯DP的draft贪心argmax：把draft lm_head的词表切分到DP组内，每个rank读[H, V/dp]而不是完整的[H, V]，
    #   只交换打包的[N, 2]。只对统一（矩形）的纯DP draft step生效；其他情况回退到复制的本地argmax。
    "ATOM_DP_DRAFT_ARGMAX": lambda: os.getenv("ATOM_DP_DRAFT_ARGMAX", "1").lower()
    in ("1", "true"),
    # 【投机】ATOM_DP_DRAFT_ARGMAX_MAX_ROWS：行数超过它（默认256）就回退
    # Row count (running_tokens) above which it falls back: the hidden gather
    # grows with rows, so the shard only pays at small M. ~256 is the V4-Pro
    # crossover.
    # 【译】行数（running_tokens）超过它就回退：hidden的gather随行数增长，所以分片只在M较小时划算。约256是V4-Pro上的交叉点。
    "ATOM_DP_DRAFT_ARGMAX_MAX_ROWS": lambda: int(
        os.getenv("ATOM_DP_DRAFT_ARGMAX_MAX_ROWS", "256")
    ),
    # 【插件】ATOM_USE_FLYDSL_GDR：vLLM插件的GDN层decode用aiter的FlyDSL kernel（flydsl_gdr_decode），默认关；读取方：plugin/vllm/attention/layer_gdn.py
    "ATOM_USE_FLYDSL_GDR": lambda: os.getenv("ATOM_USE_FLYDSL_GDR", "0").lower() == "1",
    # 【投机】ATOM_DRAFT_CUDAGRAPH：draft模型的各个pass也按尺寸录成CUDA Graph（默认开）
    # Capture each declared draft pass into a per-captured-size CUDAGraph as it is
    # warmed, so the draft replays instead of relaunching every kernel. 0 drafts
    # eagerly. On by default: DSpark tp1 acceptance is 65.25% captured vs 65.21%
    # eager. Named for the draft, not a flavor -- see `DraftGraph.will_capture`.
    # 【译】每个声明的draft pass在预热时被捕获进按捕获尺寸划分的CUDAGraph，这样draft重放graph，而不是每个kernel重新launch。
    #   0表示eager地做draft。默认开启：DSpark tp1的接受率捕获时65.25%、eager时65.21%。
    #   名字针对的是draft，而不是某种具体实现——见`DraftGraph.will_capture`。
    "ATOM_DRAFT_CUDAGRAPH": lambda: (
        os.getenv("ATOM_DRAFT_CUDAGRAPH", "1").lower() == "1"
    ),
    # --- MoE (DeepSeek-style shared experts) ---
    # 【译】--- MoE（DeepSeek风格的共享专家）---
    # 【MoE】ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD：token数不超过它（默认1024）时，共享专家分支放到alt_stream上与路由专家并行
    #   decode受益、prefill不受益，所以按token数门控；读取方：model_ops/module_dispatch_ops.py等
    # Dual-stream MoE only when num_tokens <= threshold; 0 disables dual-stream registration.
    # 【译】只在num_tokens <= 阈值时使用双stream MoE；0表示不注册双stream。
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": lambda: int(
        os.getenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "1024")
    ),
    # 【MoE】ATOM_FUSE_SHARED_EXPERT：把共享专家当作一个额外的路由专家，融合进fused MoE一起计算（默认开）；读取方：topK.py
    # Fuse into a per-rank replica without EPLB. EPLB always fuses it as routed.
    # 【译】在没有EPLB时融合进每个rank的副本。EPLB总是把它作为路由专家来融合。
    "ATOM_FUSE_SHARED_EXPERT": lambda: (
        os.getenv("ATOM_FUSE_SHARED_EXPERT", "1").lower() == "1"
    ),
    # 【MoE】ATOM_DUAL_STREAM_PIECEWISE：在PIECEWISE捕获的片段里，让MoE的shared专家和routed专家分两条stream并行（默认关）
    # Opt-in: MoE shared||routed fork inside a PIECEWISE-captured piece. Capture
    # holds it and GSM8K is unmoved; off until a throughput win is shown. Shared
    # dispatcher, so this moves V2/V3.2/K3 too. See docs.
    # 【译】需手动开启：在PIECEWISE捕获的片段内把MoE的shared与routed分叉并行。捕获能容纳它，GSM8K精度不变；
    #   在证明有吞吐收益之前默认关闭。dispatcher是共用的，所以V2/V3.2/K3也会受影响。见文档。
    "ATOM_DUAL_STREAM_PIECEWISE": lambda: os.getenv("ATOM_DUAL_STREAM_PIECEWISE", "0")
    == "1",
    # 【MoE】ATOM_MOE_GU_ITLV：MoE权重里gate/up的排布，0=分开（默认），1=交错
    # Gate/Up interleave mode for MoE weight preshuffle and kernel gate_mode.
    # "0" (default) = SEPARATED layout; "1" = INTERLEAVE layout.
    # 【译】MoE权重preshuffle和kernel gate_mode使用的Gate/Up交错模式。"0"（默认）= SEPARATED分开布局；"1" = INTERLEAVE交错布局。
    "ATOM_MOE_GU_ITLV": lambda: os.getenv("ATOM_MOE_GU_ITLV", "0") == "1",
    # --- MoE all2all (MoRI) wire format ---
    # 【译】--- MoE all2all（MoRI）传输格式 ---
    # 【MoE】ATOM_MORI_FP4_DISPATCH：MoRI v1的dispatch把token以FP4格式发送（默认关；ATOM_MORI_V2开启时不生效）；读取方：moe.py
    "ATOM_MORI_FP4_DISPATCH": lambda: (os.getenv("ATOM_MORI_FP4_DISPATCH", "0") == "1"),
    # 【MoE】ATOM_MORI_COMBINE_QUANT：MoRI combine回程的量化方式：none（bf16）或fp8_blockwise
    # Combine-side codec. "none" (the MoRI default) sends bf16 back;
    # "fp8_blockwise" selects EpCombineIntraNodeKernel_*_fp8bwq_*.
    # 【译】combine侧的编码方式。"none"（MoRI的默认）回传bf16；"fp8_blockwise"选用EpCombineIntraNodeKernel_*_fp8bwq_*。
    "ATOM_MORI_COMBINE_QUANT": lambda: os.getenv("ATOM_MORI_COMBINE_QUANT", "none"),
    # --- EPLB ---
    # 【译】--- EPLB ---
    # 科普·EPLB（Expert Parallel Load Balancing）：EP下有些专家特别"热"，它所在的卡就成了瓶颈。EPLB统计各专家负载，
    #   给热专家多放几份冗余副本，并周期性地重新安排专家到卡的映射（重排要搬权重，有开销）。实现见atom/model_ops/eplb.py
    # 【MoE】ATOM_EPLB_MAX_REBALANCES：EPLB（专家并行负载均衡）最多重排几次，之后固定专家放置（默认0=不限）；读取方：eplb.py
    # Stop periodic EPLB rebalancing after this many rebalances have run and
    # keep that expert placement for the rest of the process. Rebalances the
    # balancedness gate skips do not count. 0 (default) = no limit.
    # 【译】周期性的EPLB重新均衡运行了这么多次之后就停止，在进程剩余时间里保持那时的专家放置。被均衡度门槛跳过的重排不计数。
    #   0（默认）= 不限制。
    "ATOM_EPLB_MAX_REBALANCES": lambda: _int_env("ATOM_EPLB_MAX_REBALANCES", 0),
    # --- MTP (relaxed mtp for quantized mtp) ---
    # 【译】--- MTP（为量化MTP准备的放宽验证）---
    # 【投机】ATOM_ENABLE_RELAXED_MTP：放宽投机token的接受条件（rejection_sampler.py里RELAXED_TOP_N=10、RELAXED_DELTA=0.6），
    #   用来缓解量化后MTP接受率下降；输出不再保证与逐token严格验证一致。DeepSeek-V4.1 DSpark要求严格验证，开启会报错
    "ATOM_ENABLE_RELAXED_MTP": lambda: (
        os.getenv("ATOM_ENABLE_RELAXED_MTP", "0").lower() == "1"
    ),
    # --- Atomesh ---
    # 【译】--- Atomesh ---
    # 【服务】ATOM_MESH_BUILD：从源码安装ATOM时是否构建Atomesh（atom/mesh，主体是Rust代码），由setup.py读取
    # Build atomesh when installing ATOM from source.
    # 【译】从源码安装ATOM时构建atomesh。
    "ATOM_MESH_BUILD": lambda: os.getenv("ATOM_MESH_BUILD", "0") == "1",
    # 【服务】ATOM_MESH_FEATURES：构建Atomesh时启用的Cargo features，由setup.py读取
    # Optional Cargo features for the Atomesh package build (e.g. "ext-proc").
    # 【译】Atomesh包构建时可选的Cargo features（例如"ext-proc"）。
    "ATOM_MESH_FEATURES": lambda: os.getenv("ATOM_MESH_FEATURES", "").strip(),
    # 【服务】USE_ATOMESH_ENTRYPOINTS：openai_server入口改走Atomesh；读取方：entrypoints/openai_server.py
    # Route the OpenAI-compatible server entrypoint through Atomesh.
    # 【译】让OpenAI兼容的服务入口经由Atomesh。
    "USE_ATOMESH_ENTRYPOINTS": lambda: (
        os.getenv("USE_ATOMESH_ENTRYPOINTS", "0") == "1"
    ),
    # --- Gradient Control ---
    # 【译】--- 梯度控制 ---
    # 【加载】ATOM_REQUIRES_GRAD：创建模型参数时是否开启requires_grad（推理保持0）；读取方：model_ops/utils.py的atom_parameter
    # Enable gradient tracking on model parameters.  Default "0" (disabled)
    # is correct for inference; set to "1" only for training / fine-tuning.
    # 【译】在模型参数上开启梯度跟踪。默认"0"（关闭）适合推理；只有训练/微调时才设为"1"。
    "ATOM_REQUIRES_GRAD": lambda: os.getenv("ATOM_REQUIRES_GRAD", "0") == "1",
    # --- Bpreshuffle for weight ---
    # 【译】--- 权重的B矩阵preshuffle ---
    # 【GEMM】ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE：FP8 block-scale权重预先重排成kernel友好的布局（默认开）；linear.py、layernorm.py等多处读取
    # Preshuffle weight.  Default "1" (enabled)
    # 【译】对权重做preshuffle。默认"1"（开启）
    "ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE": lambda: (
        os.getenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1") == "1"
    ),
    # 【GEMM】ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM：FP4 Linear用不做preshuffle的aiter Triton GEMM（gemm_afp4wfp4），默认关
    "ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM": lambda: (
        os.getenv("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", "0") == "1"
    ),
    # --- V4 Attention Backend Refactor (PR-A: kill .item(), unlock CUDAGraph) ---
    # 【译】--- V4注意力后端重构（PR-A：去掉.item()，解锁CUDAGraph）---
    # 【注意力】ATOM_V4_BACKEND：DeepSeek-V4注意力用legacy（默认，逐序列Python循环）还是new（批量Triton kernel，可捕获CUDA Graph）
    #   读取方：model_ops/v4_backend_gate.py
    # `legacy` (default) keeps the per-seq Python dispatch loop with .item()
    # syncs in deepseek_v4.py. `new` routes through V4AttentionBackend with
    # batched Triton kernels (no GPU→CPU sync, CUDAGraph-capturable).
    # During Phase 1/2 migration, individual sites can be flipped to `new`
    # for byte-equal A/B verification via dump-bisect.
    # 【译】`legacy`（默认）保留deepseek_v4.py里带.item()同步的逐序列Python分派循环。`new`经由V4AttentionBackend，
    #   使用批量的Triton kernel（没有GPU→CPU同步，可被CUDAGraph捕获）。在第1/2阶段迁移期间，可以把单个位置切到`new`，
    #   用dump-bisect做逐字节相等的A/B验证。
    "ATOM_V4_BACKEND": lambda: os.getenv("ATOM_V4_BACKEND", "legacy"),
    # 【注意力】ATOM_V4_BACKEND_LAYERS：只把这些层（逗号分隔的层号）切到new后端，用于逐层二分排查
    # Comma-separated layer ids to route through the new backend (others stay
    # legacy). Empty means: respect ATOM_V4_BACKEND for all layers. Used for
    # layer-by-layer bisect during migration. Example: "0,3,15,30".
    # 【译】逗号分隔的层号，这些层走新后端（其余保持legacy）。为空表示：所有层都遵循ATOM_V4_BACKEND。
    #   用于迁移期间逐层二分排查。示例："0,3,15,30"。
    "ATOM_V4_BACKEND_LAYERS": lambda: os.getenv("ATOM_V4_BACKEND_LAYERS", ""),
    # --- Debug Dump (atom/utils/debug_helper/) ---
    # 【译】--- 调试dump（atom/utils/debug_helper/）---
    # 【调试】下面这组用于数值问题的二分排查（逐层dump forward / 权重 / 采样器top-k），默认全部关闭；读取方：utils/debug_helper/dump.py
    # All disabled (empty / no-op) by default. Set to enable instrumentation
    # for forward / weight / sampler bisecting; safe to leave wired in
    # production paths.
    #
    # Forward hidden_state dump per Block.
    # 【译】默认全部关闭（空 / 不做任何事）。设置后启用forward / 权重 / 采样器二分排查的插桩；留在生产路径里也是安全的。
    #
    #   每个Block的forward hidden_state dump。
    # 【调试】ATOM_FWD_DUMP_DIR：dump输出目录（为空=关闭）；ATOM_FWD_DUMP_LAYERS：要dump的层
    "ATOM_FWD_DUMP_DIR": lambda: os.getenv("ATOM_FWD_DUMP_DIR", ""),
    "ATOM_FWD_DUMP_LAYERS": lambda: os.getenv("ATOM_FWD_DUMP_LAYERS", ""),
    # 【调试】ATOM_FWD_DUMP_BLOCK_CLASS：按哪个类名识别"一层"（默认DeepSeek的"Block"）
    # Override for non-DeepSeek models (e.g. "DecoderLayer" for Llama).
    # 【译】非DeepSeek模型用来覆盖的值（例如Llama用"DecoderLayer"）。
    "ATOM_FWD_DUMP_BLOCK_CLASS": lambda: os.getenv(
        "ATOM_FWD_DUMP_BLOCK_CLASS", "Block"
    ),
    # 【调试】ATOM_FWD_DUMP_LAYER_ATTR：从层对象上读层号用的属性名（默认"layer_id"）
    "ATOM_FWD_DUMP_LAYER_ATTR": lambda: os.getenv(
        "ATOM_FWD_DUMP_LAYER_ATTR", "layer_id"
    ),
    # 【调试】ATOM_FWD_DUMP_ONE_SHOT：只dump一次（默认开）
    "ATOM_FWD_DUMP_ONE_SHOT": lambda: os.getenv("ATOM_FWD_DUMP_ONE_SHOT", "1") == "1",
    # 【调试】ATOM_WEIGHT_DUMP_DIR / _LAYERS / _EXIT：按rank把权重dump到目录、要dump的层（默认"0"）、dump完是否退出进程（默认是）
    # Per-rank weight dump + sys.exit(0) — for byte-equal weight comparison.
    # 【译】按rank的权重dump + sys.exit(0)——用于逐字节比较权重。
    "ATOM_WEIGHT_DUMP_DIR": lambda: os.getenv("ATOM_WEIGHT_DUMP_DIR", ""),
    "ATOM_WEIGHT_DUMP_LAYERS": lambda: os.getenv("ATOM_WEIGHT_DUMP_LAYERS", "0"),
    "ATOM_WEIGHT_DUMP_EXIT": lambda: os.getenv("ATOM_WEIGHT_DUMP_EXIT", "1") == "1",
    # 【调试】ATOM_DEBUG_TOPK / _PATH：采样器每步记录top-K个logits（K=0关闭）以及日志路径
    # Sampler top-K logits log — int K, 0/empty disables.
    # 【译】采样器top-K logits日志——整数K，0或空表示关闭。
    "ATOM_DEBUG_TOPK": lambda: int(os.getenv("ATOM_DEBUG_TOPK", "0") or "0"),
    "ATOM_DEBUG_TOPK_PATH": lambda: os.getenv("ATOM_DEBUG_TOPK_PATH", ""),
    # 【前缀缓存】ATOM_KV_EVENTS_*：把KV block的存入 / 淘汰事件发布出去，供外部组件（如按KV命中做路由的调度器）订阅
    #   读取方：config.py（组装KVEventsConfig）、distributed/kv_events.py
    # KV cache event publisher (see atom/distributed/kv_events.py).
    # 【译】KV cache事件发布器（见atom/distributed/kv_events.py）。
    # 【前缀缓存】ENABLE：总开关；PUBLISHER：发布方式（默认zmq）；ENDPOINT：发布地址；TOPIC：消息主题
    "ATOM_KV_EVENTS_ENABLE": lambda: os.getenv("ATOM_KV_EVENTS_ENABLE", "0") == "1",
    "ATOM_KV_EVENTS_PUBLISHER": lambda: os.getenv("ATOM_KV_EVENTS_PUBLISHER", "zmq"),
    "ATOM_KV_EVENTS_ENDPOINT": lambda: os.getenv(
        "ATOM_KV_EVENTS_ENDPOINT", "tcp://127.0.0.1:5557"
    ),
    "ATOM_KV_EVENTS_TOPIC": lambda: os.getenv("ATOM_KV_EVENTS_TOPIC", ""),
    # 【前缀缓存】REPLAY_ENDPOINT：订阅方可以从这里请求重放错过的事件
    # ROUTER endpoint for the replay socket; empty string disables replay.
    # 【译】重放socket的ROUTER地址；空字符串表示关闭重放。
    "ATOM_KV_EVENTS_REPLAY_ENDPOINT": lambda: os.getenv(
        "ATOM_KV_EVENTS_REPLAY_ENDPOINT", ""
    ),
    # 【前缀缓存】REPLAY_BUFFER_STEPS：可重放的历史步数
    # Size of the replay ring buffer (distinct from the send queue depth).
    # 【译】重放环形缓冲区的大小（与发送队列深度不同）。
    "ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS": lambda: int(
        os.getenv("ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS", "10000") or "10000"
    ),
    # 【前缀缓存】HWM：发布socket的高水位线（0=不限）；BUFFER_STEPS：发送队列深度（满了就丢弃）
    "ATOM_KV_EVENTS_HWM": lambda: int(os.getenv("ATOM_KV_EVENTS_HWM", "0") or "0"),
    "ATOM_KV_EVENTS_BUFFER_STEPS": lambda: int(
        os.getenv("ATOM_KV_EVENTS_BUFFER_STEPS", "10000") or "10000"
    ),
    # 【投机】ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL：跳过draft模型、返回必被拒绝的哨兵token，复现100%拒绝的最坏情况（调试用）
    # Force-skip the draft-model forward in eagle/MTP propose() and return
    # sentinel draft token ids (int max) so rejection_sampler rejects all
    # speculative tokens. Used to reproduce 100% rejection behavior — the
    # worst case for ring-buffer aliasing in compressor state caches.
    # Default: False (run the draft model normally).
    # 【译】在eagle/MTP的propose()里强制跳过draft模型的forward，返回哨兵draft token id（int最大值），
    #   让rejection_sampler拒绝所有投机token。用来复现100%拒绝的行为——这是压缩器状态缓存里环形缓冲区别名问题的最坏情况。
    #   默认：False（正常运行draft模型）。
    "ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL": lambda: (
        os.getenv("ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL", "0") == "1"
    ),
    # 【投机】ATOM_DSPARK_DISABLE_COMPILE：DSpark draft模型不做torch.compile（target仍编译）
    # Run the DSpark draft model eager, bypassing torch.compile, while leaving
    # the target compiled. Rollback lever for the draft's compiled path; prefer
    # it over --level 0, which disables compilation for BOTH models. Note that
    # --enforce-eager does NOT disable it: support_torch_compile keys off
    # compilation_config.level only (see atom/utils/decorators.py:485).
    # Default: False (compile the draft).
    # 【译】DSpark draft模型以eager方式运行，绕过torch.compile，target保持编译。这是draft编译路径的回滚手段；
    #   比--level 0更好，后者会同时关闭两个模型的编译。注意--enforce-eager*不会*关闭它：support_torch_compile只看
    #   compilation_config.level（见atom/utils/decorators.py:485）。默认：False（编译draft）。
    "ATOM_DSPARK_DISABLE_COMPILE": lambda: (
        os.getenv("ATOM_DSPARK_DISABLE_COMPILE", "0") == "1"
    ),
    # NOTE: DSpark runtime knobs (confidence_schedule, ragged,
    # ragged_graph_sizes, q_buckets, disable_sps_calib) are no longer env vars.
    # They are configured via --dspark-config (JSON dict) and carried in
    # config.dspark (see atom/config.py DSparkConfig). See
    # recipes/DSpark.md.
    # 【译】注意：DSpark的运行时开关（confidence_schedule、ragged、ragged_graph_sizes、q_buckets、disable_sps_calib）不再是环境变量。
    #   它们通过--dspark-config（JSON字典）配置，存放在config.dspark里（见atom/config.py的DSparkConfig）。见recipes/DSpark.md。
    # --- PrefillDelayer (cross-DP prefill alignment) ---
    # 【译】--- PrefillDelayer（跨DP的prefill对齐）---
    # 【调度】ATOM_ENABLE_PREFILL_DELAYER：prefill合批器总开关（默认开）；读取方：engine_core.py、scheduler.py
    #   【分支】PrefillDelayer见第3批 prefill_delayer.py
    # Master switch; default on. Set "0" to disable construction.
    # The delayer is a prefill COALESCER: it holds back prefill admission under
    # DP-attention until the accumulated prefill fills a worthwhile forward, so
    # fragmented short-input prefills / small partial tail chunks batch into one
    # forward instead of firing many tiny ones.
    # 【译】总开关；默认开启。设为"0"则不构建它。
    #   这个delayer是一个prefill*合并器*：在DP-attention下扣住prefill准入，直到积累的prefill填满一次值得的forward，
    #   这样零碎的短输入prefill / 小的部分尾chunk会合并成一次forward，而不是发出许多很小的forward。
    "ATOM_ENABLE_PREFILL_DELAYER": lambda: (
        os.getenv("ATOM_ENABLE_PREFILL_DELAYER", "1") == "1"
    ),
    # 【调度】ATOM_PREFILL_DELAYER_TARGET_FILL：积累的待prefill token达到max_num_batched_tokens的这个比例（默认0.9）就放行
    # Fill target: release prefill once accumulated pending tokens reach
    # target_fill * max_num_batched_tokens (averaged across prefillable ranks).
    # In (0, 1]; higher batches harder (fewer, larger prefills) at some TTFT
    # cost. Default 0.9.
    # 【译】填充目标：当积累的待处理token达到target_fill * max_num_batched_tokens（在可prefill的rank间取平均）时放行prefill。
    #   取值在(0, 1]；越高合批越激进（prefill更少、更大），代价是一定的TTFT。默认0.9。
    "ATOM_PREFILL_DELAYER_TARGET_FILL": lambda: float(
        os.getenv("ATOM_PREFILL_DELAYER_TARGET_FILL", "0.9")
    ),
    # 【调度】ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS：一个prefill最多被扣住多少个调度tick（默认200），保证TTFT有上界
    # TTFT bound: max consecutive scheduler ticks a held prefill waits before
    # force-release (deterministic across ranks; replaces the old wall-clock +
    # pass-count pair).
    # 【译】TTFT上界：被扣住的prefill在强制放行前最多等待的连续调度tick数（各rank之间是确定的；取代了旧的墙钟时间 + 轮数组合）。
    "ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS", "200")
    ),
    # 【调度】ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS：chunked prefill做到一半被扣住时的更紧上界（默认100）
    # Tight bound (ticks) for a held mid-chunked-prefill: a partial holds already
    # allocated KV, so it force-releases sooner than a fresh prefill.
    # 【译】被扣住的、chunked prefill进行到一半的请求的更紧上界（tick数）：部分完成的请求已经占着分配好的KV，所以比新的prefill更早强制放行。
    "ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS", "100")
    ),
    # 【调度】ATOM_PREFILL_DELAYER_STALL_TICKS：整组工作连续这么多tick（默认10）没有变化，就放行未填满的prefill
    # Consecutive non-growing ticks after which the coalescer gives up waiting
    # (burst ended, more won't come) and releases.
    # 【译】连续多少个tick没有增长之后，合并器不再等待（突发结束，不会再来更多请求），直接放行。
    "ATOM_PREFILL_DELAYER_STALL_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_STALL_TICKS", "10")
    ),
    # 【调度】ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK：KV使用率达到它（默认0.9）就强制放行
    # KV high watermark: at/above this KV usage a prefillable rank force-releases
    # (can't accumulate a bigger batch anyway).
    # 【译】KV高水位：KV使用率达到或超过它时，可prefill的rank强制放行（反正也攒不出更大的batch了）。
    "ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK": lambda: float(
        os.getenv("ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK", "0.9")
    ),
    # 【调度】ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK：KV使用率低于它就强制放行（默认不启用）
    # Optional KV-usage low watermark: below it a prefillable rank force-releases
    # (GPU starving — feed it). Empty string => None => disabled.
    # 【译】可选的KV使用率低水位：低于它时可prefill的rank强制放行（GPU正在挨饿——喂它）。空字符串 => None => 关闭。
    "ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": lambda: (
        None
        if os.getenv("ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK", "") == ""
        else float(os.getenv("ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK"))
    ),
    # 【调度】ATOM_PREFILL_DELAYER_MAX_QUEUE_MS：按排队时间限制额外的合批（毫秒，默认不启用）
    # After decode protection, bound extra coalescing by queue age. Checkpoint
    # dependency waits use TTFT_MAX_TICKS; this is not an end-to-end TTFT bound.
    # Empty string => None => disabled.
    # 【译】在decode保护之后，按排队时长限制额外的合并。checkpoint依赖的等待使用TTFT_MAX_TICKS；这不是端到端的TTFT上界。
    #   空字符串 => None => 关闭。
    "ATOM_PREFILL_DELAYER_MAX_QUEUE_MS": lambda: (
        None
        if os.getenv("ATOM_PREFILL_DELAYER_MAX_QUEUE_MS", "") == ""
        else float(os.getenv("ATOM_PREFILL_DELAYER_MAX_QUEUE_MS"))
    ),
    # 【调度】ATOM_PREFILL_DECODE_INTERVAL：每次prefill forward之后，至少保护这么多次调度只做decode（默认0=不强制）
    # After a prefill forward, protect this many scheduler passes for decode
    # before allowing another prefill. Mirrors SGLang's
    # --prefill-decode-interval; 0 disables the hard interval.
    # A nonzero interval also enables local coalescing on TP without PP.
    # 【译】一次prefill forward之后，在允许下一次prefill之前保护这么多个调度轮次给decode。对应SGLang的--prefill-decode-interval；
    #   0关闭这个硬间隔。非零的间隔也会在不用PP的TP上启用本地合并。
    "ATOM_PREFILL_DECODE_INTERVAL": lambda: int(
        os.getenv("ATOM_PREFILL_DECODE_INTERVAL", "0")
    ),
    # 【前缀缓存】ATOM_PREFIX_WALK_MEMO：等待中请求的前缀匹配结果在各次探测间缓存，不必每个tick重走（默认开）
    # Memoise a waiting request's prefix walk across the per-tick probes (the
    # delayer, the DP chunk-cost estimate; `BlockManager.can_allocate_memoized`).
    # "0" re-walks every tick.
    # 【译】在每个tick的多次探测之间（delayer、DP chunk成本估计；`BlockManager.can_allocate_memoized`）记住一个等待中请求的前缀遍历结果。
    #   "0"表示每个tick都重新遍历。
    "ATOM_PREFIX_WALK_MEMO": lambda: (os.getenv("ATOM_PREFIX_WALK_MEMO", "1") == "1"),
    # 【前缀缓存】ATOM_FRONTEND_PREFIX_HASH：在前端（API进程）算好prompt的block hash链再交给引擎（默认开）
    # Hash the prompt block chain in the frontend and ship it to the engine.
    # "0" leaves all hashing to the engine.
    # 【译】在前端对prompt的block链做hash，并把结果发给引擎。"0"把所有hash都交给引擎去算。
    "ATOM_FRONTEND_PREFIX_HASH": lambda: (
        os.getenv("ATOM_FRONTEND_PREFIX_HASH", "1") == "1"
    ),
    # 【前缀缓存】ATOM_PREFIX_FAST_WALK：前缀命中用一次性遍历（默认开）
    # One-pass prefix-hit walk (`BlockManager._hit_walk`) whose ids `allocate`
    # claims. "0" walks block by block with a token compare per block.
    # 【译】一次性的前缀命中遍历（`BlockManager._hit_walk`），`allocate`直接认领它找到的id。"0"表示逐block遍历，每个block比较一次token。
    "ATOM_PREFIX_FAST_WALK": lambda: (os.getenv("ATOM_PREFIX_FAST_WALK", "1") == "1"),
    # --- TBO prefill ubatch splitting ---
    # 【译】--- TBO prefill微批拆分 ---
    # 科普·TBO（Two-Batch Overlap）：把一个batch拆成两个微批（ubatch）A、B交替执行：A在做all2all / all-reduce通信时，B在GPU上计算，
    #   反之亦然，用计算掩盖通信。前提是两个微批工作量差不多，所以拆分要尽量对半；batch太小时拆了反而不划算（见下面的MIN_TOKENS）。
    #   实现见atom/utils/tbo/
    # 【并行】ATOM_TBO_PREFILL_TOKEN_SPLIT：TBO拆prefill时按token数精确对半切（默认开）
    #   【分支】TBO见第9批 utils/tbo/
    # Split prefill ubatches at the exact token midpoint (vLLM-DBO style),
    # cutting through a request if needed for perfectly balanced 50/50 ubatches.
    # Default on; set "0" to fall back to the request-boundary balanced split.
    # 【译】在精确的token中点处拆分prefill微批（vLLM-DBO风格），必要时从一个请求中间切开，得到完全均衡的50/50微批。
    #   默认开启；设为"0"回退到在请求边界上做均衡拆分。
    "ATOM_TBO_PREFILL_TOKEN_SPLIT": lambda: (
        os.getenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "1") == "1"
    ),
    # 【并行】ATOM_TBO_PREFILL_MIN_TOKENS：每个rank的prefill token数至少这么多（默认8192）才做TBO拆分
    # Minimum prefill tokens (per rank) required to TBO-split.
    # 【译】做TBO拆分所需的最少prefill token数（每个rank）。
    "ATOM_TBO_PREFILL_MIN_TOKENS": lambda: int(
        os.getenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")
    ),
    # --- PCP MoE comm mode ---
    # 【译】--- PCP的MoE通信模式 ---
    # 【并行】ATOM_PCP_MOE_MERGE：PCP（prefill上下文并行）下MoE是否在完整token集合上计算（默认开）
    # Fold the PCP (prefill-context-parallel) dim into the MoE tp/ep sharding.
    # Only meaningful when prefill_context_parallel_size > 1;
    # Default "1": all-gather hidden 1/W -> full before MoE and slice
    # full -> 1/W after, so MoE sees the complete token set (MoE itself is
    # untouched / PCP-agnostic). Costs one extra hidden all-gather per layer.
    # "0": MoE runs on each rank's 1/W token shard with no extra comm.
    # 【译】把PCP（prefill上下文并行）维度折进MoE的tp/ep切分。只在prefill_context_parallel_size > 1时有意义；
    #   默认"1"：进入MoE前把hidden从1/W all-gather成完整的，MoE之后再从完整的切回1/W，所以MoE看到的是完整的token集合
    #   （MoE本身不改动，与PCP无关）。代价是每层多一次hidden的all-gather。
    #   "0"：MoE在每个rank自己的1/W token分片上运行，没有额外通信。
    "ATOM_PCP_MOE_MERGE": lambda: os.getenv("ATOM_PCP_MOE_MERGE", "1") == "1",
    # 【并行】ATOM_TBO_TP_AR_MODE：纯TP下TBO的all-reduce是否放到通信stream上，与另一个微批的计算重叠
    # Pure-TP TBO all_reduce overlap mode (see module_dispatch_ops.tbo_all_reduce):
    #   "overlap" (default): move the AR onto the comm stream so it overlaps the
    #             partner ubatch's compute. Per-ubatch pynccl comms keep the
    #             cross-rank enqueue order consistent (hang-free).
    #   "inline": run the AR on the current stream, no overlap. Plan-A baseline.
    # 【译】纯TP TBO的all_reduce重叠模式（见module_dispatch_ops.tbo_all_reduce）：
    #     "overlap"（默认）：把AR移到通信stream上，与伙伴微批的计算重叠。每个微批独立的pynccl通信器保证跨rank的入队顺序一致（不会挂死）。
    #     "inline"：在当前stream上运行AR，不重叠。方案A的基线。
    "ATOM_TBO_TP_AR_MODE": lambda: os.getenv("ATOM_TBO_TP_AR_MODE", "overlap"),
    # --- NUMA binding ---
    # 【译】--- NUMA绑定 ---
    # 科普·NUMA：多路CPU服务器上每个CPU插槽有自己"近"的内存和PCIe设备；GPU worker进程如果跑在离它的GPU"远"的CPU上，
    #   H2D拷贝和CPU侧调度都会变慢。NUMA绑定把进程固定到GPU所在节点的CPU核与内存上
    # 【并行】ATOM_NUMA_BIND：把每个GPU worker绑到离该GPU最近的NUMA节点的CPU核与内存上（默认关）；读取方：utils/numa_utils.py
    # Master switch: pin each GPU worker to its GPU-local NUMA node's CPU cores
    # and preferred memory. Default off so baseline/pinned A/B stays clean.
    # 【译】总开关：把每个GPU worker固定到它GPU所在NUMA节点的CPU核和首选内存上。默认关闭，让基线/绑定的A/B保持干净。
    "ATOM_NUMA_BIND": lambda: os.getenv("ATOM_NUMA_BIND", "0") == "1",
    # 【并行】ATOM_AUTO_NUMA_BIND：自动探测GPU到NUMA节点的映射（默认开）
    # Auto-detect the GPU->NUMA-node mapping (amdsmi first, sysfs fallback).
    # Default on, so `ATOM_NUMA_BIND=1` alone is zero-config.
    # 【译】自动检测GPU->NUMA节点的映射（先amdsmi，再退回sysfs）。默认开启，所以单设`ATOM_NUMA_BIND=1`即可零配置。
    "ATOM_AUTO_NUMA_BIND": lambda: os.getenv("ATOM_AUTO_NUMA_BIND", "1") == "1",
    # 【并行】ATOM_NUMA_NODE：手工指定每个全局rank的NUMA节点，覆盖自动探测
    # Explicit per-global-rank node ids (comma separated), overriding auto, e.g.
    # ATOM_NUMA_NODE="0,0,0,0,1,1,1,1". A single value applies to all ranks.
    # 【译】显式给出每个全局rank的节点id（逗号分隔），覆盖自动检测，例如ATOM_NUMA_NODE="0,0,0,0,1,1,1,1"。只给一个值时应用到所有rank。
    "ATOM_NUMA_NODE": lambda: os.getenv("ATOM_NUMA_NODE", ""),
    # 【并行】ATOM_CRASH_ON_NUMA_BIND_FAILURE：绑定失败时报错，而不是只警告
    # Raise instead of warn when binding fails.
    # 【译】绑定失败时抛异常，而不是警告。
    "ATOM_CRASH_ON_NUMA_BIND_FAILURE": lambda: (
        os.getenv("ATOM_CRASH_ON_NUMA_BIND_FAILURE", "0") == "1"
    ),
    # 【并行】ATOM_PP_SEND_ALLGATHER：PP相邻stage之间每个TP rank只发1/tp的hidden，接收端再all-gather（默认开）；读取方：distributed/pp_comm.py
    # PP-boundary hidden_states/residual are TP-replicated; when on, each rank
    # sends only its 1/tp_size slice and the receiver all-gathers, cutting PP
    # link traffic by tp_size. Default on; set "0" for full-tensor sends.
    # 【译】PP边界上的hidden_states/residual在TP间是复制的；开启时每个rank只发送它的1/tp_size切片，由接收端all-gather，
    #   把PP链路流量降低tp_size倍。默认开启；设为"0"发送完整张量。
    "ATOM_PP_SEND_ALLGATHER": lambda: os.getenv("ATOM_PP_SEND_ALLGATHER", "1") == "1",
    # 科普·Engram（DeepSeek-V4.1）：给模型加一张"按最近几个token查表"的记忆。它不是注意力（MHA/GQA那套q·k不参与），
    #   而是在第1、14层的attention之前，往残差流上加一个查表得到的向量（recipes/Engram.md）：
    #
    #       token序列（示意：一个字当一个token）:  ... 北  京  大  学
    #       当前token"学"往回看3个token，取各阶n-gram（n = 2..4）:  "大学"  "京大学"  "北京大学"
    #         每个n-gram x 每个hash头 ──hash──> 行号 ──> 在巨大的嵌入表里取一行 ──> 投影 + 门控 ──> 加到残差上
    #
    #   hash之前先把token id映射到一个更小的"规范化词表"（大小写、重音、空白不同的token算同一个，model_ops/engram/mapping.py的CompressedTokenizer）。
    #   表非常大（两张表合计约188.83GiB，E4M3数值 + E8M0 scale），所以放在主机内存里，每步只取用到的那几行。
    #   因为要往回看3个token，前缀命中后的重放也要带上命中点之前的3个token（config.py的replay_history_tokens = engram_max_ngram_size - 1）
    # 【模型】ATOM_ENGRAM_*（下面4个）：DeepSeek-V4.1的Engram（n-gram记忆，表放在主机内存里）相关开关；读取方：model_ops/engram/
    # 【模型】ATOM_ENGRAM_UVA：GPU通过UVA直接读主机内存里的n-gram表，而不是在主机上gather（默认开）
    # Engram: read the n-gram tables with a device kernel over UVA instead of
    # gathering them on the host. The tables stay in host memory (page-locked in
    # place, not copied to HBM); the GPU pulls only the rows a step names and
    # dequantizes them there. On by default: the host gather gives the same rows
    # but costs ~50 ms of CPU per decode step with the GPU idle behind it. Set
    # to 0 to fall back. Anything that would make it unsafe -- no CUDA, more TP
    # ranks than hash heads, a registration that will not fit -- falls back on
    # its own, so the switch is for taking the host path deliberately.
    # 【译】Engram：用设备kernel通过UVA读取n-gram表，而不是在主机上gather。表留在主机内存里（原地锁页，
    #   不拷到HBM）；GPU只拉取一步用到的行，并在GPU上反量化。默认开启：主机gather得到的行相同，
    #   但每个decode step要花约50 ms的CPU，GPU在后面空等。设为0回退。任何会让它不安全的情况——没有CUDA、TP rank多于hash头数、
    #   注册放不下——都会自己回退，所以这个开关是用来刻意走主机路径的。
    "ATOM_ENGRAM_UVA": lambda: os.getenv("ATOM_ENGRAM_UVA", "1") == "1",
    # 【模型】ATOM_ENGRAM_CACHE_DIR：压缩词表的表的缓存目录（默认~/.cache/atom/engram）
    # Where the compressed-vocab table is cached between runs. The table is
    # reproducible from the tokenizer, so this only trades startup time for
    # disk; point it at shared storage to let several servers build it once.
    # 【译】压缩词表的表在多次运行之间缓存在哪里。这张表可以从tokenizer重新生成，所以这只是用磁盘换启动时间；
    #   把它指向共享存储，可以让多个服务器只构建一次。
    "ATOM_ENGRAM_CACHE_DIR": lambda: os.getenv(
        "ATOM_ENGRAM_CACHE_DIR", str(Path.home() / ".cache" / "atom" / "engram")
    ),
    # 【模型】ATOM_ENGRAM_OVERLAP：hash/UVA查表和TP重组与前几层计算重叠（默认开）
    # Overlap hash/UVA lookup and TP reassembly with early layers, using private
    # IPC state where supported. Requires UVA; set 0 to disable.
    # 【译】把hash/UVA查表和TP重组与前面几层重叠执行，在支持的地方使用私有的IPC状态。需要UVA；设为0关闭。
    "ATOM_ENGRAM_OVERLAP": lambda: os.getenv("ATOM_ENGRAM_OVERLAP", "1") == "1",
    # 【模型】ATOM_ENGRAM_FUSED_GATE：Engram的FP32门控与残差相加融合成一个kernel（默认开）
    # Fuse FP32 post-wkv gating and residual addition; 0 selects the torch reference.
    # 【译】融合FP32的post-wkv门控和残差相加；0选用torch参考实现。
    "ATOM_ENGRAM_FUSED_GATE": lambda: os.getenv("ATOM_ENGRAM_FUSED_GATE", "1") == "1",
    # --- KV offload (LMCache) ---
    # 【译】--- KV offload（LMCache）---
    # 【PD/offload】下面OFFLOAD_*这组变量没有ATOM_前缀，控制KV offload的并发、阈值、暂存缓冲；读取方：kv_transfer/offload/*、scheduler.py
    #   【分支】KV offload见第9批 kv_transfer/offload/
    # Unset or empty always means the default, so a knob can be cleared inline.
    # A set but unusable value warns and falls back for the token thresholds
    # and lookup step counts, which only tune reuse; it is rejected at startup
    # for widths, sizes and timeouts, which shape memory and concurrency.
    # Save / load executor widths of the offload worker. DSV4 ignores
    # OFFLOAD_LOAD_WORKERS: its SLOT load path needs a serial load executor.
    # 【译】未设置或为空总是表示默认值，所以可以内联清除某个开关。设置了但无法使用的值：对token阈值和查询步数
    #   （它们只调节复用）会警告并回退；对宽度、大小和超时（它们决定内存和并发）会在启动时拒绝。
    #   offload worker的保存 / 加载执行器宽度。DSV4忽略OFFLOAD_LOAD_WORKERS：它的SLOT加载路径需要串行的加载执行器。
    # 【PD/offload】OFFLOAD_COPY_WORKERS / OFFLOAD_LOAD_WORKERS：保存 / 加载执行器的线程数（默认各1）
    "OFFLOAD_COPY_WORKERS": lambda: _int_env("OFFLOAD_COPY_WORKERS", 1),
    "OFFLOAD_LOAD_WORKERS": lambda: _int_env("OFFLOAD_LOAD_WORKERS", 1),
    # 【PD/offload】OFFLOAD_MAX_PENDING_SAVES：正在执行加排队的保存任务上限
    # Running-plus-queued save bound. None means unset: the connector then
    # derives max(2, 2 * OFFLOAD_COPY_WORKERS) and the scheduler's state tier
    # uses 2. A kv_connector_extra_config "max_pending_saves" takes precedence.
    # 【译】运行中加排队的保存任务上限。None表示未设置：此时connector推导为max(2, 2 * OFFLOAD_COPY_WORKERS)，调度器的状态层使用2。
    #   kv_connector_extra_config里的"max_pending_saves"优先。
    "OFFLOAD_MAX_PENDING_SAVES": lambda: _optional_int_env("OFFLOAD_MAX_PENDING_SAVES"),
    # 【PD/offload】OFFLOAD_MIN_LOAD_TOKENS / OFFLOAD_MIN_SAVE_TOKENS：外部存储命中至少多少token才值得加载 / 至少多长的前缀才保存（默认各8192）
    # Minimum external-tier hit worth loading, in tokens. And the shortest
    # prefix native lmcache_mp stores: an absolute boundary, for normal and
    # late saves alike (other connectors ignore it).
    # 【译】值得加载的最小外部层命中，单位token。同时也是原生lmcache_mp存储的最短前缀：一个绝对边界，
    #   对正常保存和延迟保存都一样（其他connector忽略它）。
    "OFFLOAD_MIN_LOAD_TOKENS": lambda: max(
        0, _int_env_or_default("OFFLOAD_MIN_LOAD_TOKENS", 8192)
    ),
    "OFFLOAD_MIN_SAVE_TOKENS": lambda: max(
        0, _int_env_or_default("OFFLOAD_MIN_SAVE_TOKENS", 8192)
    ),
    # 【PD/offload】OFFLOAD_LOOKUP_MEMO_STEPS / OFFLOAD_LOOKUP_RETRY_STEPS：外部存储查询结果缓存多少个调度步 / 查询失败后多少步内不再重试（默认各32）
    # Scheduler steps a memoised tier-lookup answer is replayed, and steps a
    # failed lookup suppresses the next attempt.
    # 【译】一个被记住的层查询结果被重放多少个调度步，以及一次失败的查询会压制下一次尝试多少步。
    "OFFLOAD_LOOKUP_MEMO_STEPS": lambda: _nonnegative_int_env(
        "OFFLOAD_LOOKUP_MEMO_STEPS", 32
    ),
    "OFFLOAD_LOOKUP_RETRY_STEPS": lambda: _nonnegative_int_env(
        "OFFLOAD_LOOKUP_RETRY_STEPS", 32
    ),
    # 【调试】OFFLOAD_PROFILE：打印每次offload传输的性能日志
    # Per-transfer offload profiling logs.
    # 【译】每次传输的offload性能分析日志。
    "OFFLOAD_PROFILE": lambda: _flag_env("OFFLOAD_PROFILE"),
    # 【PD/offload】OFFLOAD_ASYNC_LOOKUP_DEPTH：lmcache_mp每轮调度提前为等待队列头部多少个请求发查询（默认16）
    # lmcache_mp sends tier lookups for the head of the waiting queue ahead of
    # admission; this is how many waiting requests per scheduling pass.
    # 【译】lmcache_mp在准入之前就为等待队列头部的请求发送层查询；这是每个调度轮次处理的等待请求数。
    "OFFLOAD_ASYNC_LOOKUP_DEPTH": lambda: _nonnegative_int_env(
        "OFFLOAD_ASYNC_LOOKUP_DEPTH", 16
    ),
    # 【PD/offload】OFFLOAD_LOOKUP_DEFER_S：异步查询迟迟没回时，准入最多跳过该请求多少秒，之后改为等待结果（默认2.0）
    # How long admission may pass over a request whose async lookup has not
    # answered before it waits for the answer instead.
    # 【译】异步查询还没有回答时，准入可以跳过这个请求多久，之后改为等待答案。
    "OFFLOAD_LOOKUP_DEFER_S": lambda: _finite_float_env(
        "OFFLOAD_LOOKUP_DEFER_S", 2.0, allow_zero=True
    ),
    # 【PD/offload】OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT：显存里的前缀命中已经够多时，不等外部存储查询就准入（默认开）
    # Admit a prefill without waiting for its external-tier lookup when the HBM
    # prefix leaves fewer than OFFLOAD_MIN_LOAD_TOKENS uncached, so no tier hit
    # could be loaded. The lookup is still sent to refresh the tier's LRU.
    # On by default; set to 0 to always wait for the lookup.
    # 【译】当HBM里的前缀命中后剩余未缓存的token少于OFFLOAD_MIN_LOAD_TOKENS、因此不可能加载任何层命中时，
    #   不等外部层查询就准入prefill。查询仍会发出，以刷新层的LRU。默认开启；设为0总是等待查询结果。
    "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT": lambda: _flag_env(
        "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", "1"
    ),
    # 【PD/offload】OFFLOAD_SINGLE_STREAM：暂存打包和拷贝两步放在同一个stream上（实验性）
    # Experimental: run the staging pack and copy legs on one stream.
    # 【译】实验性：在同一个stream上运行暂存打包和拷贝这两段。
    "OFFLOAD_SINGLE_STREAM": lambda: _flag_env("OFFLOAD_SINGLE_STREAM"),
    # 【PD/offload】OFFLOAD_GPU_STAGING_CHUNKS / OFFLOAD_GPU_STAGING_MAX_BYTES：GPU暂存缓冲区的大小（LMCache chunk数）和字节上限
    # GPU staging buffer size in LMCache chunks, and an upper bound in bytes.
    # Unset derives the size from the KV geometry.
    # 【译】GPU暂存缓冲区的大小（以LMCache chunk计），以及以字节计的上限。未设置时根据KV几何形状推导大小。
    "OFFLOAD_GPU_STAGING_CHUNKS": lambda: _optional_int_env(
        "OFFLOAD_GPU_STAGING_CHUNKS", min_value=1
    ),
    "OFFLOAD_GPU_STAGING_MAX_BYTES": lambda: _optional_int_env(
        "OFFLOAD_GPU_STAGING_MAX_BYTES", min_value=1
    ),
    # 【PD/offload】OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER：每次传输后释放GPU暂存缓冲区
    # Free the GPU staging buffer after each transfer instead of keeping it.
    # 【译】每次传输后释放GPU暂存缓冲区，而不是一直保留。
    "OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER": lambda: _flag_env(
        "OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER"
    ),
    # 【PD/offload】OFFLOAD_SLOT_STAGING_SLOTS：DeepSeek-V4 SLOT sidecar的暂存行数
    # DSV4 SLOT sidecar staging rows; None means unset (1). A
    # kv_connector_extra_config "slot_sidecar_staging_slots" takes precedence.
    # 【译】DSV4 SLOT sidecar的暂存行数；None表示未设置（1）。kv_connector_extra_config里的"slot_sidecar_staging_slots"优先。
    "OFFLOAD_SLOT_STAGING_SLOTS": lambda: _optional_int_env(
        "OFFLOAD_SLOT_STAGING_SLOTS"
    ),
    # 【PD/offload】OFFLOAD_COMMITTED_SIDECAR_CAPACITY：DeepSeek-V4已提交SLOT sidecar的索引容量
    # DSV4 committed SLOT sidecar index capacity; None means unset (65536). A
    # kv_connector_extra_config "committed_sidecar_index_capacity" takes
    # precedence.
    # 【译】DSV4已提交SLOT sidecar的索引容量；None表示未设置（65536）。kv_connector_extra_config里的
    #   "committed_sidecar_index_capacity"优先。
    "OFFLOAD_COMMITTED_SIDECAR_CAPACITY": lambda: _optional_int_env(
        "OFFLOAD_COMMITTED_SIDECAR_CAPACITY"
    ),
    # 【PD/offload】OFFLOAD_PUBLICATION_TIMEOUT_S / _POLL_INTERVAL_S：等待已保存的SLOT sidecar可见的超时和轮询间隔
    # DSV4 wait for a saved SLOT sidecar to become visible, and its poll period.
    # 【译】DSV4等待一个已保存的SLOT sidecar变得可见的时长，以及轮询周期。
    "OFFLOAD_PUBLICATION_TIMEOUT_S": lambda: _finite_float_env(
        "OFFLOAD_PUBLICATION_TIMEOUT_S", 5.0, allow_zero=True
    ),
    "OFFLOAD_PUBLICATION_POLL_INTERVAL_S": lambda: _finite_float_env(
        "OFFLOAD_PUBLICATION_POLL_INTERVAL_S", 0.01, allow_zero=False
    ),
    # 【PD/offload】LMCACHE_MP_TRANSFER_MODE：LMCache MP传输模式：auto或lmcache_driven
    # LMCache MP transfer mode: auto or lmcache_driven (engine_driven is
    # rejected as not implemented). A kv_connector_extra_config
    # "lmcache.mp.mp_transfer_mode" takes precedence.
    # 【译】LMCache MP传输模式：auto或lmcache_driven（engine_driven会被拒绝，因为尚未实现）。
    #   kv_connector_extra_config里的"lmcache.mp.mp_transfer_mode"优先。
    "LMCACHE_MP_TRANSFER_MODE": lambda: os.getenv("LMCACHE_MP_TRANSFER_MODE", "auto"),
}


# 判断环境变量是否被显式设置。很多地方用它区分"用户没设、走自动判断"和"用户明确设了值"
#   例：moe.py里ATOM_USE_TRITON_MOE没设时按GPU架构自动决定；config.py里ATOM_DP_*设了才覆盖ParallelConfig
def is_set(name: str) -> bool:
    """Return True if the env var *name* is explicitly set (even if empty)."""
    # 【译】如果环境变量*name*被显式设置（即使为空）就返回True。
    # 注意：docstring说"即使为空"，但代码对空字符串返回False（val != ""），实际语义是"设置了且非空"
    val = os.getenv(name)
    return val is not None and val != ""


# 模块级__getattr__（PEP 562）：访问envs.XXX而XXX不是模块里真实存在的属性时才调用它
#   每次访问都重新执行lambda（没有缓存）；名字不在表里就抛AttributeError，拼错变量名会立刻报错，而不是悄悄返回None
def __getattr__(name: str):
    # lazy evaluation of environment variables
    # 【译】环境变量的惰性求值
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Third-party / dependency env vars (documented only, NOT managed here)
# ---------------------------------------------------------------------------
# MASTER_ADDR, MASTER_PORT        — PyTorch distributed; set in model_runner.py
# AITER_LOG_LEVEL                 — AITER library log verbosity
# AITER_QUICK_REDUCE_QUANTIZATION — AITER; set conditionally in model_runner.py
# TORCHINDUCTOR_CACHE_DIR         — PyTorch Inductor; set in compiler_inferface.py
# TRITON_CACHE_DIR                — Triton compiler; set in compiler_inferface.py
# HF_TOKEN                        — HuggingFace Hub auth token
# HF_HUB_ENABLE_HF_TRANSFER      — HuggingFace fast transfers
# NCCL_DEBUG, NCCL_TIMEOUT        — NCCL diagnostics
# FLA_COMPILER_MODE, FLA_CI_ENV,
#   FLA_GDN_FIX_BT, FLA_USE_CUDA_GRAPH,
#   FLA_TRIL_PRECISION             — FLA ops library
# VLLM_PP_LAYER_PARTITION         — vLLM legacy (still active in models/utils.py)
# VLLM_USE_MODELSCOPE             — vLLM legacy (benchmarks)
# LMCACHE_EC_PIN_TIMEOUT_SEC      — LMCache library's own source-pin timeout;
#                                   read in kv_transfer/offload/_offload_common.py
#                                   (offload_save_abandon_timeout_s) to derive the
#                                   engine's save-abandon window from it, so the
#                                   two stay ordered. The scheduler never reads the
#                                   env itself -- it asks the connector, via
#                                   save_abandon_timeout_s. ATOM does not own the
#                                   knob, hence no default of its own here.
# 【译】---------------------------------------------------------------------------
#   第三方 / 依赖库的环境变量（只在此说明，不由这里管理）
#   ---------------------------------------------------------------------------
#   MASTER_ADDR, MASTER_PORT        —— PyTorch分布式；在model_runner.py中设置
#   AITER_LOG_LEVEL                 —— AITER库的日志详细程度
#   AITER_QUICK_REDUCE_QUANTIZATION —— AITER；在model_runner.py中按条件设置
#   TORCHINDUCTOR_CACHE_DIR         —— PyTorch Inductor；在compiler_inferface.py中设置
#   TRITON_CACHE_DIR                —— Triton编译器；在compiler_inferface.py中设置
#   HF_TOKEN                        —— HuggingFace Hub的认证token
#   HF_HUB_ENABLE_HF_TRANSFER       —— HuggingFace快速传输
#   NCCL_DEBUG, NCCL_TIMEOUT        —— NCCL诊断
#   FLA_COMPILER_MODE, FLA_CI_ENV, FLA_GDN_FIX_BT, FLA_USE_CUDA_GRAPH, FLA_TRIL_PRECISION —— FLA算子库
#   VLLM_PP_LAYER_PARTITION         —— vLLM遗留变量（在models/utils.py中仍然生效）
#   VLLM_USE_MODELSCOPE             —— vLLM遗留变量（benchmarks）
#   LMCACHE_EC_PIN_TIMEOUT_SEC      —— LMCache库自己的源数据锁定超时；在kv_transfer/offload/_offload_common.py中读取
#                                      （offload_save_abandon_timeout_s），用来推导引擎的保存放弃窗口，使两者保持先后顺序。
#                                      调度器自己从不读这个环境变量——它通过save_abandon_timeout_s询问connector。
#                                      ATOM并不拥有这个开关，所以这里没有它自己的默认值。
# 补充：CLAUDE.md要求启动服务前设置AITER_LOG_LEVEL=WARNING，以免aiter kernel日志刷屏
#   另外ATOM_DISABLE_JIT_WEAKREF（utils/__init__.py）、ATOM_V4_USE_TRITON_FUSION（models/deepseek_v4）等少数变量
#   在代码里直接读os.environ，没有登记在本表中
