# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json
import logging
from dataclasses import dataclass, fields

from atom import LLMEngine
from atom.config import (
    CompilationConfig,
    CUDAGraphMode,
    DCPConfig,
    DSparkConfig,
    EPLBConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.model_engine.engine_core_mgr import DP_LB_DEFAULT, DP_LB_STRATEGIES
from atom.utils import envs

logger = logging.getLogger("atom")

# 本文件：引擎参数（EngineArgs），命令行与Config之间的一层
# 离线示例（simple_inference.py）和在线服务（openai_server.py）都通过它解析参数、创建引擎
#
# 一个参数从命令行到生效的路径：
#   EngineArgs.add_cli_args(parser)       在argparse上注册所有 --xxx 参数
#            │  parser.parse_args()
#            ▼
#   EngineArgs.from_cli_args(args)        从Namespace中挑出与EngineArgs字段同名的值
#            │
#            ▼
#   EngineArgs.create_engine()
#            │  _get_engine_kwargs()      字段转成kwargs，并做少量改名/打包（见下方）
#            ▼
#   LLMEngine(model, **kwargs)            只把"Config里有同名字段"的kwargs传给Config(model, **kwargs)
#            │                            Config里没有的kwargs会被直接忽略
#            ▼
#   Config.__post_init__                  读取HF config、推导和校验各项配置（见config.py）
#
# 对比nano-vllm：nano-vllm没有命令行层，直接LLM(path, **kwargs)，kwargs原样传给Config
# 除了命令行参数，ATOM还有两百多个ATOM_*环境变量（kernel路径选择、优化开关、调试开关），
#   按功能分组的索引见utils/envs.py文件开头
#
# 谁用本文件（已grep核实）：
#   add_cli_args + from_cli_args：entrypoints/openai/api_server.py（在线服务）、entrypoints/atomesh/server.py、
#     examples/simple_inference.py；tests/test_arg_utils_spec.py 测试它
#   kv_offload_connector_config / compose_kv_offload_config：只在本文件的_get_engine_kwargs里调用
# 调参排错技巧：启动日志里有一行"Engine kwargs: {...}"（_get_engine_kwargs末尾打印），
#   可以确认命令行参数最终变成了什么；但Config.__post_init__之后还会再改一些值（如index_cache_dtype），
#   那行日志打印的是改之前的值


# 把"[1,2,4,8]"这样的字符串解析成Python列表（用于--cudagraph-capture-sizes）
def parse_size_list(size_str: str) -> list[int]:
    """Parse a string representation of a list into a Python list."""
    # 【译】把列表的字符串表示解析成Python列表。
    import ast

    # ast.literal_eval只解析字面量（数字、列表、字典等），不执行任意代码，比eval安全
    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing size list: {size_str}") from e


# ---------------- 【分支】KV offload：把KV cache卸载到CPU内存/外部存储，见第9批 ----------------
# 通过环境变量ATOM_KV_OFFLOAD开启，最终会被折叠进--kv-transfer-config（见_get_engine_kwargs末尾）

# 【科普】KV offload：GPU显存放不下的KV cache先"存"到CPU内存或远端存储（LMCache / Mooncake Store），
#   以后同样前缀的请求再来时直接读回来，而不必重新prefill。相当于给前缀缓存加了一层更大、更慢的二级缓存。
#   例：多轮对话中，用户隔了几分钟才回复，GPU里的KV早已被淘汰，offload层还能命中，省掉整段历史的prefill。
#   实现在 atom/kv_transfer/offload/
# ATOM_KV_OFFLOAD value -> the offload connector it selects.
# 【译】ATOM_KV_OFFLOAD 的取值 -> 它选择的offload connector。
_KV_OFFLOAD_MODES = {
    "lmcache": "lmcache_offload",
    "lmcache_mp": "lmcache_mp",
    "mooncake_store": "mooncake_store",
}


def kv_offload_connector_config(mode: str, extra_config: str) -> dict | None:
    """The offload connector ATOM_KV_OFFLOAD selects, or None when it is off.

    ``lmcache`` is the in-process LMCache backend, ``lmcache_mp`` the standalone
    ``lmcache server`` one, and ``mooncake_store`` a Mooncake Store reached
    without LMCache. ATOM_KV_OFFLOAD_EXTRA_CONFIG, a JSON object, becomes the
    connector's ``kv_connector_extra_config``, which every backend reads.
    """
    # 【译】ATOM_KV_OFFLOAD 选择的offload connector；关闭时返回None。
    #   lmcache 是进程内的LMCache后端，lmcache_mp 是独立的 lmcache server 后端，
    #   mooncake_store 是不经过LMCache、直接访问的Mooncake Store。ATOM_KV_OFFLOAD_EXTRA_CONFIG
    #   是一个JSON对象，会成为connector的 kv_connector_extra_config，每个后端都会读取它。
    mode = mode.strip().lower()
    if mode in ("", "0", "off", "none"):
        if extra_config.strip():
            raise ValueError(
                "ATOM_KV_OFFLOAD_EXTRA_CONFIG is set but ATOM_KV_OFFLOAD is off"
            )
        return None
    if mode not in _KV_OFFLOAD_MODES:
        raise ValueError(
            f"ATOM_KV_OFFLOAD={mode!r}: expected one of {sorted(_KV_OFFLOAD_MODES)}"
        )
    offload = {"kv_connector": _KV_OFFLOAD_MODES[mode], "kv_role": "offload"}
    if extra_config.strip():
        extra = json.loads(extra_config)
        if not isinstance(extra, dict):
            raise TypeError("ATOM_KV_OFFLOAD_EXTRA_CONFIG must be a JSON object")
        offload["kv_connector_extra_config"] = extra
    return offload


def compose_kv_offload_config(kv_transfer_config: str, offload: dict | None) -> str:
    """Fold the ATOM_KV_OFFLOAD connector into the --kv-transfer-config JSON string.

    Without a transfer connector the offload connector stands alone; with one,
    both become subs of a ``multi`` connector (appended if it already is one).
    """
    # 【译】把 ATOM_KV_OFFLOAD 的connector合并进 --kv-transfer-config 的JSON字符串。
    #   没有传输connector时，offload connector单独存在；有的话，两者都成为一个 multi connector 的子项
    #   （如果它本来就是multi，就追加进去）。
    # 例：--kv-transfer-config是PD分离的mooncake connector，同时开了offload：
    #   {"kv_connector": "multi", "connectors": [{mooncake...}, {"kv_connector": "lmcache_offload", ...}]}
    if offload is None:
        return kv_transfer_config
    transfer = json.loads(kv_transfer_config or "{}")
    if not transfer:
        return json.dumps(offload)
    is_multi = _connector_name(transfer) == "multi"
    subs = transfer.get("connectors", []) if is_multi else [transfer]
    if any(_is_offload_connector(sub) for sub in subs):
        raise ValueError(
            "ATOM_KV_OFFLOAD conflicts with an offload connector already set in "
            "--kv-transfer-config; use only one of them"
        )
    if is_multi:
        return json.dumps({**transfer, "connectors": [*subs, offload]})
    return json.dumps({"kv_connector": "multi", "connectors": [transfer, offload]})


# 取出connector配置里的名字，统一成小写（casefold比lower更彻底，能处理一些特殊字符）
def _connector_name(cfg: object) -> str:
    name = cfg.get("kv_connector", "") if isinstance(cfg, dict) else ""
    return str(name).strip().casefold()


def _is_offload_connector(cfg: object) -> bool:
    """Whether a transfer config names a backend registered as an offload tier.

    Resolved by name or alias, in any case, as the factory resolves it when it
    builds the connector; a name it does not know is not an offload tier here
    (building it fails later, with the factory's error).
    """
    # 【译】一个传输配置所指定的后端，是否注册为offload层。
    #   按名字或别名解析、不区分大小写，与工厂构建connector时的解析方式相同；
    #   工厂不认识的名字在这里不算offload层（之后构建时会失败，报工厂的错误）。
    name = cfg.get("kv_connector") if isinstance(cfg, dict) else None
    try:
        canonical = KVConnectorFactory.canonical_name(name)
    except ValueError:
        return False
    return KVConnectorFactory.is_offload_backend(canonical)


# 引擎参数：每个字段对应一个命令行参数（--xxx-yyy 对应字段 xxx_yyy），默认值也与命令行一致
# 大部分字段同名传给Config；含义和取值影响在这里说明，Config里不再重复
@dataclass
class EngineArgs:
    """Arguments for configuring the LLM Engine."""

    # 【译】配置LLM引擎的参数。

    # ---------------- 模型 ----------------
    # 模型的本地路径或HuggingFace名字（不在本地时会下载）
    model: str = "Qwen/Qwen3-0.6B"
    # 是否允许执行模型仓库里的自定义Python代码（transformers的trust_remote_code）
    trust_remote_code: bool = False

    # ---------------- 并行（第9批详解；第1~8批只需关心tensor_parallel_size） ----------------
    # 张量并行（TP）：每层的权重切分到多张GPU，每张GPU一个ModelRunner进程，层内通过all-reduce合并结果
    # 【科普】直觉：Y = X·W，把W按列（或按行）切成tp份，每张卡只存、只算1/tp，再用通信把结果拼/加起来。
    #   注意力层按head切：例如32个head、-tp 8，每卡算4个head。
    #   例：70B模型bf16权重约140GB，-tp 8后每卡约17.5GB；代价是每层1~2次all-reduce，卡越多通信占比越大。
    #   实现在 model_ops/linear.py（ColumnParallelLinear / RowParallelLinear 等）
    tensor_parallel_size: int = 1
    # 【分支】decode上下文并行（DCP）：把一个请求的KV按token切分到TP组内的多张卡上，必须整除TP
    # 【科普】为什么需要：MLA这类模型KV只有1个"头"，TP切head切不动KV，于是TP组里每张卡都存一整份KV（重复）。
    #   DCP让这几张卡各存一部分token的KV，各自算局部注意力，再用LSE合并结果。
    #   例：-tp 8 -dcp 8，每张卡只存1/8的token的KV，同样显存能放约8倍长的上下文；代价是每层多一次通信。
    #   不增加卡数（复用TP的卡）。实现见 distributed/dcp_utils.py、model_ops/dcp_ops.py，文档docs/context_parallel_guide.md
    decode_context_parallel_size: int = 1
    # 【分支】流水线并行（PP）：按层切分模型，每个stage负责一段连续的层
    # 【科普】例：61层模型 -pp 2，stage0算前约一半层、stage1算后一半，中间只传一次hidden state（通信远少于TP），
    #   适合跨机；代价是"流水线气泡"：一个batch在stage0时stage1空闲，要靠多个batch交错填满。
    #   每个stage是一个独立的EngineCore进程，实现在 model_engine/pp_engine_core.py、distributed/pp_transport.py
    pipeline_parallel_size: int = 1
    # 【分支】prefill上下文并行（PCP）：prefill时把一条长序列切到多张卡上计算
    #   例：一条128K token的prompt、-pcp 2，每张卡各算64K token，缩短首token时间（TTFT）；
    #   与DCP不同，它会增加卡数（总卡数 = tp x pcp），见 distributed/pcp_utils.py
    prefill_context_parallel_size: int = 1
    # 数据并行（DP）：起多个EngineCore，每个有自己的调度器和一份完整模型（每份内部仍可TP），请求在它们之间分发（第2批）
    #   例：8张卡 -tp 2 -dp 4 = 4份模型副本、每份2卡；与 -tp 8 相比，单请求延迟更高（每份算力少），
    #   但没有跨8卡的all-reduce，总吞吐往往更高。前提是一份模型能放进tp张卡
    data_parallel_size: int = 1
    # 以下5个是多机DP用的：本节点跑几个DP rank、本节点第一个全局rank号、协调节点的地址和端口
    data_parallel_size_local: int | None = None
    data_parallel_rank: int = 0
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int = 29500
    data_parallel_base_port: int | None = None

    # ---------------- 执行方式、KV cache、调度 ----------------
    # 强制eager模式：不捕获CUDA Graph（调试时用，性能会明显下降）
    #   CUDA Graph是什么、为什么能加速decode，见config.py里CUDAGraphMode上方的【科普】
    #   注意：它只关CUDA Graph，不关torch.compile（--level仍生效）；CLAUDE.md提醒：被编译的函数里加调试代码
    #   即使--enforce-eager也会破坏Dynamo追踪
    enforce_eager: bool = False
    # 前缀缓存：不同请求的相同前缀（如系统提示词）复用已算好的KV block（第4批）
    # 【科普】做法：把prompt按block切开，每个block算一个"连同前面所有token"的链式hash；新请求的前几个block
    #   hash命中，就直接复用那些物理block，只prefill剩下的部分。
    #   例：2000 token的系统提示词 + 50 token的用户问题，第二个请求只需prefill约50 token。
    #   实现在 model_engine/prefix_hash.py（hash链）和 block_manager.py（命中/复用/淘汰）
    enable_prefix_caching: bool = True
    # 引擎内部通信端口：ModelRunner用它设置MASTER_PORT，作为TP进程组的集合点；不是HTTP服务端口
    #   更正/补充：torch.distributed主进程组的init_method用的其实是data_parallel_base_port
    #   （model_runner.py里get_distributed_init_method(master_ip, data_parallel_base_port)）；
    #   MASTER_PORT=port 是给读这个环境变量的组件用的（据utils/__init__.py的说明，custom all-reduce的
    #   rendezvous TCPStore绑在MASTER_PORT上）。HTTP端口是api_server的--server-port。
    #   同一台机器起两个ATOM实例时，--port要错开，否则会抢同一个端口
    port: int = 8006
    # KV cache的数据类型：bf16或fp8；fp8让KV cache占用减半，同样显存能容纳约两倍的token
    # 【科普】每个token的KV字节数 = 2(K和V) x 层数 x kv_head数 x head_dim x 每元素字节数。
    #   例：32层、8个kv head、head_dim 128：bf16是 2x32x8x128x2 = 128KB/token，fp8是64KB/token；
    #   60GB的KV空间，bf16约放49万token，fp8约放98万token。fp8写入时量化、读出时乘scale反量化，
    #   有轻微精度损失；decode又是读KV受限的，读的字节少一半通常还更快。CLAUDE.md的启动示例都带--kv-cache-dtype fp8
    kv_cache_dtype: str = "bf16"
    # 【分支】DSA稀疏注意力索引器的缓存类型（DeepSeek-V3.2/V4等），None表示跟随kv_cache_dtype，见第7、8批
    #   更正：None不总是跟随kv_cache_dtype——Config.__post_init__里：DeepSeek-V4原生单节点默认fp4（gfx942上fp8），
    #   V4插件模式或有KV传输时fp8，V4.1固定默认fp8，其他模型才等于kv_cache_dtype
    # 【科普】DSA（DeepSeek Sparse Attention）：每个query不看全部历史token，而是先用一个很小的"索引器"
    #   给所有历史token打分，挑出top-k（如2048个）再做真正的注意力。索引器自己也要缓存每个token的key，
    #   这就是"索引缓存"，与主KV cache分开存放。实现见 model_ops/sparse_indexer_*.py 等
    index_cache_dtype: str | None = None
    # 每个KV block容纳的token数（对比nano-vllm：nano-vllm默认256，ATOM默认16）
    #   block越小，内部碎片越少、前缀缓存的命中粒度越细，但block table越长、管理开销越大
    #   例：一条1000 token的序列，block_size 16需要63个block、最后一块浪费8个槽位；block_size 256需要4个block、浪费24个
    #   注意：部分模型会被Config强制改写，例如DeepSeek-V4.1强制128（见config.py __post_init__）
    block_size: int = 16
    # 最大上下文长度（prompt+生成）；None表示使用模型config的max_position_embeddings
    max_model_len: int | None = None
    # 单个step最多计算的token数（主要约束prefill；decode每条序列1个token，投机解码时是k+1个）
    #   调大：权重读取、kernel启动等固定开销被更多token摊薄，吞吐更高；
    #         但单步更慢（decode请求的ITL变大）、激活显存更大（可用KV block变少）
    #   调小：反之；配合chunked prefill，长prompt会被切成更多块
    #   它还限制了能录的CUDA Graph尺寸：ModelRunner会丢掉大于 min(max_num_seqs, 本值//(k+1)) 的捕获尺寸
    #   （k是投机token数，见model_runner.py capture_cudagraph前的检查）
    max_num_batched_tokens: int = 16384
    # chunked prefill时单个请求每步最多prefill多少token；0表示不限制（只受max_num_batched_tokens约束）
    #   调小可以让长prefill和decode更均匀地穿插，降低decode的ITL
    long_prefill_token_threshold: int = 0
    # MLA注意力内部做chunked prefill时的token预算（第7批）
    #   具体是：prefill时若"已缓存的前缀"（前缀缓存命中的部分或前面chunk已算的部分）超过这个token数，
    #   MLA要把压缩的latent KV解压成完整的K/V，就按这个大小分段解压、分段算注意力；
    #   它也决定了预分配的k/v workspace大小（aiter_mla.py里形状为[本值, head数, head_dim]）
    #   调大：分段少、更快，但workspace显存更大；调小：反之。<=0时不分配workspace
    #   help里说"默认使用max_num_batched_tokens"，实际是两者默认值都等于16384，并不联动
    attn_prefill_chunk_size: int = 16384
    # 【分支】带每请求状态的模型（DeepSeek-V4、GDN等），每隔N个token保存一次状态checkpoint，
    #   以后前缀命中时可以从checkpoint恢复；0关闭，-1只在"有需要的位置"和prompt末尾保存，见第4批
    state_checkpoint_interval_tokens: int = 8192
    # 【分支】是否允许"因缺checkpoint被拒绝的命中"为自己新增一个checkpoint位置
    state_checkpoint_demand: bool = True
    # 【分支】DeepSeek-V4.1专用的SWA有界重放优化（decoder侧/encoder侧），见第8批
    enable_decoder_swa_bounded_replay: bool = True
    enable_encoder_swa_bounded_replay: bool = True
    # chunked prefill：长prompt放不进单步预算时切成多块，分多个step算完
    # 【科普】不切的话，一条很长的prompt要独占一整个step，期间所有decode请求都停下等它（ITL出现尖峰）；
    #   切开后每步只算一块，剩余预算留给decode。
    #   例：max_num_batched_tokens=16384、一条40000 token的prompt，大约分3步：16384 + 16384 + 7232。
    #   切分逻辑在 model_engine/scheduler.py
    enable_chunked_prefill: bool = True

    # ---------------- 日志与统计 ----------------
    # 是否周期性打印引擎状态行（running/waiting请求数、KV使用率、前缀缓存命中率、吞吐）
    enable_log_stats: bool = True
    # 状态行的打印间隔（秒）
    throughput_log_interval: float = 10.0
    # 状态行里前缀缓存命中率的滑动窗口大小（按请求数）
    cache_hit_rate_window: int = 1000

    # ---------------- 【分支】PP动态分块，见第9批 ----------------
    # 让PP下一个prefill请求的各个chunk耗时相等，缩短流水线的填充/排空时间（需要pp>1）
    enable_dynamic_chunking: bool = False
    # None defers to ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR, read only once the
    # feature is on (see Config._resolve_dynamic_chunking).
    # 【译】None表示使用 ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR，只有功能开启后才读取它（见 Config._resolve_dynamic_chunking）。
    dynamic_chunking_smooth_factor: float | None = None
    dynamic_chunking_min_chunk_size: int = 4096

    # 调度延迟因子：在调度下一个prompt之前等待"上一个prompt延迟 x 因子"，攒更多请求一起prefill；0表示不等待
    #   例：上一次prefill耗时200ms、因子0.5，则新prompt至少等100ms再调度。用TTFT换吞吐（scheduler.py的_passed_delay）
    scheduler_delay_factor: float = 0.0
    # 单个step最多调度的序列数，主要影响decode
    #   decode是访存受限的：每步主要开销是读一遍权重，batch里多一条序列几乎是白赚一个token
    #   上限实际还受可用KV block约束：KV不够时调度器会抢占
    #   另外：sampling等buffer按它预分配（model_runner.py的max_bs），CUDA Graph捕获尺寸也会被它截断
    max_num_seqs: int = 512
    # 允许ATOM使用的显存比例：扣除权重和激活峰值后，剩余部分分配给KV cache（第5批）
    #   例：192GB的卡、0.9 -> 预算约172.8GB；若权重占40GB、激活峰值和CUDA Graph等占10GB，KV cache约得到122GB
    #   调高：KV更多、能并发更多请求；太高可能OOM（其他进程或碎片也要显存）
    gpu_memory_utilization: float = 0.9

    # ---------------- 编译与CUDA Graph（第5批） ----------------
    # decode阶段捕获CUDA Graph的batch size列表（字符串形式，_get_engine_kwargs里解析成列表）
    #   运行时batch向上取整到最近的捕获尺寸并回放；超过最大值的batch以eager模式运行
    #   （向上取整用np.searchsorted，见utils/forward_context.py；例：实际9条序列 -> 补到16条回放）
    #   列表越密，padding浪费越少，但录图时间和图占的显存越多
    cudagraph_capture_sizes: str = "[1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]"
    # torch.compile编译级别：0=eager，1=torch.compile，2=dynamo只追踪一次，3=分段编译（默认）
    #   分段（piecewise）编译的含义见config.py CUDAGraphMode上方【科普】④；编译结果缓存在 /root/.cache/atom/，
    #   CLAUDE.md要求改代码后重启前清掉它，否则可能静默用到旧的编译结果
    level: int = 3
    # CUDA Graph模式：NONE、FULL（整个forward捕获成一张graph，默认）、PIECEWISE（按编译切出的片段
    #   分别捕获，attention走eager，需要--level 3）、FULL_AND_PIECEWISE、AF_PIECEWISE（attention也单独捕获）
    cudagraph_mode: str = "FULL"
    # 用假权重代替读取checkpoint（只测性能/调试时用）：empty不初始化、zero全0、xavier随机初始化
    load_dummy: str | None = None

    # ---------------- MoE与专家并行（第7批） ----------------
    # 专家并行（EP）：MoE的专家分布到不同GPU上，token通过all-to-all发送到专家所在的GPU
    # 【科普】MoE：把Transformer的FFN换成很多个"专家"FFN，每个token由router只选其中top-k个（如256选8），
    #   参数量大但每个token的计算量小。EP：例如256个专家、8张卡，每卡放32个完整专家（而不是每个专家切成8份的TP）；
    #   每层要先把token发给它选中的专家所在的卡（dispatch），算完再收回来（combine），这两次就是all-to-all。
    #   实现在 model_ops/moe.py、model_ops/fused_moe/（mori/rccl的prepare_finalize）
    enable_expert_parallel: bool = False
    # 用均匀分布的假router logits替换真实路由，测量专家负载完全均衡时的性能上限
    fake_eplb: bool = False
    # torch profiler的trace输出目录；不设置时start_profile/stop_profile不采集
    torch_profiler_dir: str | None = None
    # DP attention：注意力部分按DP切分（每个rank只算自己的请求），MoE等部分在DP rank之间协同计算
    #   （通常配合--enable-expert-parallel），见第7、9批
    enable_dp_attention: bool = False
    # DP时CoreManager把请求分给哪个rank：least_requests（默认）、round_robin、least_tokens（第2批）
    dp_load_balance: str = DP_LB_DEFAULT
    # 【分支】TBO（Two-Batch Overlap）：把一个batch拆成两个微batch，让通信和计算重叠，见第9批
    #   例：微batch A在做MoE的all-to-all通信时，GPU同时算微batch B的注意力，通信时间被"藏"起来；
    #   代价是每个微batch更小、kernel效率略降。实现在 utils/tbo/
    enable_tbo: str | None = None
    # MoE的all-to-all通信后端：MORI的高吞吐/低延迟模式、RCCL，或none（退回到all-gather/reduce-scatter）
    #   取值取舍：high-throughput适合prefill的大batch；low-latency适合decode的小batch；
    #   none时每张卡收集全部token再各自只算本卡专家，通信量大但不依赖MORI。
    #   _get_engine_kwargs里拆成enable_low_latency + moe_all2all_backend两个Config字段
    all2all_backend: str | None = None
    # MoE实现：standard（prepare/GEMM/finalize三段）或mega（FlyDSL融合的MegaMoE）
    moe_backend: str = "standard"

    # ---------------- 投机解码（第7批） ----------------
    # 投机解码方法：mtp、eagle3、dspark；None表示不开启
    # 【科普】投机解码：decode每步只出1个token但要读一遍全部权重，很浪费。于是先用一个很小的"草稿模型"
    #   快速猜k个token，再让大模型一次forward同时验证这k个（相当于k+1个token的小prefill），猜对的全部接受。
    #   例：k=3，草稿猜"的 天气 很"，大模型验证前两个对、第三个错，则这一步得到2个草稿token + 大模型自己给出的1个
    #   = 3个token，而大模型只跑了1次。输出分布与不开投机解码一致（拒绝采样保证）。
    #   mtp：用模型checkpoint自带的MTP（Multi-Token Prediction）层当草稿（DeepSeek-V3等）；
    #   eagle3：单独训练的EAGLE-3草稿小模型，复用大模型的hidden state；两者都由spec_decode/eagle_proposer.py实现。
    #   dspark：块并行一次出多个草稿token，实现在 spec_decode/dspark_proposer.py（选择逻辑见spec_decode/factory.py）
    method: str | None = None
    # 每步草稿模型自回归生成几个草稿token（MTP的k）
    #   调大：猜对时一步收获更多token；但接受率随位置递减，验证的计算量和KV预留都按k+1增长，
    #   猜错的部分白算。常用1~3。设为0等于不开启（_get_engine_kwargs里method和k都非零才建SpeculativeConfig）
    num_speculative_tokens: int = 1

    # 【分支】KV传输配置（JSON字符串），用于PD分离、KV offload，见第9批
    # 【科普】PD分离（Prefill/Decode Disaggregation）：prefill算力受限、decode访存受限，混在一起会相互干扰。
    #   于是用两组机器：prefill节点（kv_role=kv_producer）算完prompt的KV，通过RDMA传给decode节点
    #   （kv_role=kv_consumer），后者只负责逐token生成。两边可以各自选最合适的并行方式和batch大小。
    #   例：'{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}'
    #   实现和用法见 atom/kv_transfer/disaggregation/（README.md）
    kv_transfer_config: str = "{}"
    # 独立的草稿模型路径：eagle3必填；mtp不支持（MTP层就在目标模型的checkpoint里）
    draft_model: str | None = None
    # 调试/压测用：强制固定的草稿接受率或接受长度，二者互斥
    spec_decode_acceptance_rate: float | None = None
    spec_decode_acceptance_length: float | None = None
    # 在计算图中插入graph_marker节点，方便profile时定位
    mark_trace: bool = False
    # 【分支】RapidServe：单卡内的prefill/decode分离（用不同的stream，可选按CU划分），见第5批
    #   与上面跨机的PD分离不同：同一张卡上prefill和decode并发跑在两条stream上，不传KV；
    #   CU（Compute Unit，AMD GPU的计算单元，类似NVIDIA的SM）屏蔽：让两条stream各自只用一部分CU，互不抢占
    enable_rapidserve: bool = False
    disagg_prefill_max_num_seqs: int | None = None
    disagg_constrained: bool = False
    # 在线量化配置（JSON）：加载时把权重量化成ptpc_fp8或mxfp4，可按层覆盖，见第6批
    # 【科普】离线量化是事先发布一个已量化的checkpoint；在线量化则读入bf16的checkpoint，在加载时逐层量化。
    #   ptpc_fp8：per-token（激活）+ per-channel（权重）的FP8，权重显存减半；mxfp4：每32个元素共享一个scale的4bit，
    #   约为bf16的1/4。代价：一定精度损失、加载变慢。常见做法是只把占大头的MoE专家量化成mxfp4。
    #   消费方：model_loader/loader.py；文档 docs/online_quantization_guide.md
    online_quant_config: dict | None = None
    # 加载HF config后覆盖其中的字段（JSON）
    hf_overrides: dict | None = None
    # DSpark投机解码的动态配置（JSON），解析成DSparkConfig
    dspark_config: dict | None = None

    # 【分支】EPLB（专家负载均衡）：运行时监控各专家负载，并迁移/复制热点专家，见第7批
    #   为什么需要：EP下每层要等最慢的那张卡；若某个专家特别热，它所在的卡收到的token远多于别人，其他卡空等。
    #   例：num_redundant_experts=8，就多出8个槽位放热点专家的副本，把它们的token分摊到多张卡上。
    #   实现在 model_ops/eplb.py
    eplb_enable: bool = False
    eplb_config: dict | None = None
    # 【分支】DCP的详细配置（JSON），解析成DCPConfig，见第9批
    dcp_config: dict | None = None

    # 在argparse上注册全部引擎参数；每个参数上方的【译·help】是其help文本的中文翻译
    # 参数名约定：大多用连字符（--max-num-seqs），少数沿用下划线（--kv_cache_dtype）；
    #   FlexibleArgumentParser两种写法都接受，argparse把它们存到同名的下划线属性里（args.max_num_seqs）
    # @staticmethod：不需要self/cls，可以直接EngineArgs.add_cli_args(parser)调用
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add engine arguments to an argument parser."""
        # 【译】把引擎参数添加到一个参数解析器上。
        # Model configuration
        # 【译】模型配置
        # 【译·help】模型名或路径。
        parser.add_argument(
            "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
        )
        # 【译·help】覆盖API返回的模型名；不指定时默认为--model的值。
        #   （注：EngineArgs没有这个字段，from_cli_args不会读取它，它由API server使用）
        #   具体：api_server.py里 model_name = args.served_model_name if args.served_model_name else args.model
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=None,
            help="Override the model name returned by the API. "
            "If not specified, defaults to the --model value.",
        )
        # 【译·help】加载模型时信任远程代码。
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code when loading model.",
        )
        # 【译·help】张量并行大小。
        parser.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=1,
            help="Tensor parallel size.",
        )
        # 【译·help】流水线并行大小。把模型的层切分到多个stage（总卡数 = tp x pp x pcp）。
        parser.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            type=int,
            default=1,
            help="Pipeline parallel size. Splits the model's layers across "
            "stages (world = tp x pp x pcp).",
        )
        # 【译·help】prefill上下文并行大小。独立的维度（总卡数 = tp x pcp）；prefill时切分序列。
        parser.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            type=int,
            default=1,
            help="Prefill context parallel size. Independent dimension "
            "(world = tp x pcp); splits the sequence during prefill.",
        )
        # 【译·help】数据并行大小。
        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
        )
        # 【译·help】在本节点上运行的DP rank数。默认等于--data-parallel-size（单节点）。
        #   设得更小，可以让本节点只承担多节点运行中的一部分。
        parser.add_argument(
            "--data-parallel-size-local",
            type=int,
            default=None,
            help=(
                "Number of data-parallel ranks to run on THIS node. Defaults "
                "to --data-parallel-size (single-node). Set it lower to give "
                "this node one slice of a multi-node run."
            ),
        )
        # 【译·help】本节点拥有的第一个全局DP rank。节点0用0；2x4运行中的第二个节点用4。
        parser.add_argument(
            "--data-parallel-rank",
            type=int,
            default=0,
            help=(
                "First GLOBAL data-parallel rank owned by this node. Node 0 "
                "uses 0; the second node of a 2x4 run uses 4."
            ),
        )
        # 【译·help】协调节点（全局DP rank 0）的IP。
        parser.add_argument(
            "--data-parallel-master-ip",
            type=str,
            default="127.0.0.1",
            help="IP of the coordinator node (global DP rank 0).",
        )
        # 【译·help】DP进程组的集合端口。引擎socket的端口从它推导（base = port + 100，每个DP rank 3个端口）。
        parser.add_argument(
            "--data-parallel-master-port",
            type=int,
            default=29500,
            help=(
                "Rendezvous port for the DP process group. Engine sockets are "
                "derived from it (base = port + 100, 3 ports per DP rank)."
            ),
        )
        # 【译·help】model runner分布式初始化的集合端口。多节点启动时需要显式设置。
        parser.add_argument(
            "--data-parallel-base-port",
            type=int,
            default=None,
            help=(
                "Rendezvous port for model-runner distributed init. Set "
                "explicitly for multi-node launches."
            ),
        )
        # 【译·help】decode上下文并行大小。必须整除tensor_parallel_size。
        parser.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            type=int,
            default=1,
            help="Decode context parallel size. Must divide tensor_parallel_size.",
        )
        # 【译·help】强制以eager模式执行。
        #   （action="store_true"：命令行出现该开关时为True，否则为False）
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Enforce eager mode execution.",
        )
        # 【译·help】开启前缀缓存（默认开启）。用--no-enable_prefix_caching关闭。
        #   （BooleanOptionalAction：自动生成 --xxx 和 --no-xxx 两个开关）
        #   配合FlexibleArgumentParser，实际注册了4个写法：--enable_prefix_caching、--enable-prefix-caching、
        #   --no-enable_prefix_caching、--no-enable-prefix-caching（已实测）
        #   关闭的代价：重复前缀（系统提示词、多轮历史）每次都要重新prefill，TTFT变大；好处是省掉hash计算，
        #   且做A/B性能对比时结果不受缓存命中影响
        parser.add_argument(
            "--enable_prefix_caching",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable prefix caching (default: enabled). "
            "Use --no-enable_prefix_caching to disable.",
        )
        # 【译·help】引擎内部端口
        parser.add_argument(
            "--port",
            type=int,
            default=8006,
            help="Engine internal port",
        )
        # 【译·help】KV cache类型。默认是'bf16'。
        parser.add_argument(
            "--kv_cache_dtype",
            choices=["bf16", "fp8"],
            type=str,
            default="bf16",
            help="KV cache type. Default is 'bf16'.",
        )
        # 【译·help】索引缓存类型。原生单节点DeepSeek-V4默认为'fp4'，但在gfx942（MI300系列）上默认为'fp8'；
        #   其他模型默认与--kv_cache_dtype相同。
        parser.add_argument(
            "--index-cache-dtype",
            "--index_cache_dtype",
            choices=["bf16", "fp8", "fp4"],
            type=str,
            default=None,
            help="Index cache type. Native single-node DeepSeek-V4 defaults to "
            "'fp4' except on gfx942, which defaults to 'fp8'; other models "
            "default to --kv_cache_dtype.",
        )
        # 【译·help】KV cache的block大小。
        parser.add_argument(
            "--block-size", type=int, default=16, help="KV cache block size."
        )
        # 【译·help】模型最大上下文长度，默认取hf_config.max_position_embeddings。
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Maximum model context length, the default is set to hf_config.max_position_embeddings.",
        )
        # 【译·help】捕获cudagraph的尺寸。例：[1,2,4,8,16]
        parser.add_argument(
            "--cudagraph-capture-sizes",
            type=str,
            default=EngineArgs.cudagraph_capture_sizes,
            help="Sizes to capture cudagraph. Example: [1,2,4,8,16]",
        )
        # 【译·help】编译级别（0-3）。
        parser.add_argument(
            "--level", type=int, default=3, help="The level of compilation (0-3)."
        )
        # 【译·help】CUDA graph运行模式。FULL = 手动捕获整个forward（默认，原有行为）。
        #   PIECEWISE = 按片段捕获cudagraph，attention走eager（需要--level 3）。
        #   AF_PIECEWISE = 在PIECEWISE的基础上，attention核心也用零拷贝缓冲区单独捕获成一张cudagraph（DeepSeek-V4 DSpark）。
        parser.add_argument(
            "--cudagraph-mode",
            type=str,
            default="FULL",
            choices=["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE", "AF_PIECEWISE"],
            help="CUDA graph runtime mode. FULL = manual whole-forward capture "
            "(default, existing behavior). PIECEWISE = per-piece cudagraph with "
            "attention eager (requires --level 3). AF_PIECEWISE = PIECEWISE where "
            "the attention core is also captured into its own cudagraph with "
            "zero-copy buffers (DeepSeek-V4 DSpark).",
        )
        # 【译·help】用假权重代替读取checkpoint。只写开关或'=empty'：跳过加载（未初始化，旧行为）。
        #   '=zero'：所有权重为0。'=xavier'：bf16权重用xavier_uniform_初始化，fp4/fp8打包权重用固定的
        #   目标幅值（有限值、大致接近真实尺度；fp4是验证过的路径）。
        #   （nargs="?" + const：只写--load_dummy不带值时取const，即"empty"）
        #   用途：没有真实checkpoint或不想等加载时测性能、测显存；输出的文字没有意义。
        #   empty是torch.empty未初始化的显存，内容随机（可能有NaN/Inf）；xavier给出有限、尺度接近真实的值
        parser.add_argument(
            "--load_dummy",
            nargs="?",
            const="empty",
            default=None,
            choices=["empty", "zero", "xavier"],
            help="Use dummy weights instead of reading the checkpoint. Bare flag "
            "or '=empty': skip loading (uninitialized, legacy behavior). '=zero': "
            "all weights 0. '=xavier': xavier_uniform_ for bf16 weights and a "
            "constant target magnitude for fp4/fp8 packed weights (finite, "
            "roughly real-scale; fp4 is the validated path).",
        )
        # 【译·help】开启专家并行（EP MoE）。
        parser.add_argument(
            "--enable-expert-parallel",
            action="store_true",
            help="Enable expert parallel(EP MoE).",
        )
        # 【译·help】用合成的均匀分布替换MoE的router logits，使每个专家被等概率选中。
        #   仅用于测量负载均衡时的性能上限。
        parser.add_argument(
            "--fake-eplb",
            action="store_true",
            help="Replace MoE router logits with a synthetic uniform "
            "distribution so every expert is selected equally. For "
            "benchmarking the balanced-load upper bound only.",
        )
        # 【译·help】保存torch profiler trace的目录
        parser.add_argument(
            "--torch-profiler-dir",
            type=str,
            default=None,
            help="Directory to save torch profiler traces",
        )
        # 【译·help】开启DP attention。
        #   取舍：MLA这类KV只有1个头的模型，纯TP时每张卡都存一整份KV；DP attention让每张卡只处理自己的请求、
        #   只存自己请求的KV，同样显存能容纳更多请求；代价是MoE前后需要在DP rank之间收发token（见--all2all-backend）
        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enable DP attention.",
        )
        # 【译·help】CoreManager把请求路由到某个DP引擎rank的策略。'round_robin'：旧的、不考虑请求数的轮转。
        #   'least_requests'（默认）：路由到在途请求最少的rank，平局时选在途prompt token负载更轻的。
        #   'least_tokens'：路由到在途token总负载最低的rank（prompt token + 每个请求折算的token当量，
        #   可通过ATOM_DP_LB_REQ_EQUIV调整）。data_parallel_size == 1时无效。
        parser.add_argument(
            "--dp-load-balance",
            type=str,
            default=DP_LB_DEFAULT,
            choices=list(DP_LB_STRATEGIES),
            help="Strategy the CoreManager uses to route a request to a DP "
            "engine rank. 'round_robin': legacy request-count-agnostic "
            "rotation. 'least_requests' (default): route to the rank with the "
            "fewest in-flight requests, breaking ties by the lighter in-flight "
            "prompt-token load. 'least_tokens': route to the rank with "
            "the lowest combined in-flight token load (prompt tokens + "
            "per-request token-equivalent, tunable via ATOM_DP_LB_REQ_EQUIV). "
            "Has no effect when data_parallel_size == 1.",
        )
        # 【译·help】开启TBO（Two-Batch Overlap）以重叠通信与计算。'--enable-tbo'或'--enable-tbo prefill'：
        #   只对prefill开启。'--enable-tbo all'：prefill和decode都开启。
        parser.add_argument(
            "--enable-tbo",
            nargs="?",
            const="prefill",
            default=None,
            choices=["prefill", "all"],
            help="Enable TBO (Two-Batch Overlap) for comm/compute overlap. "
            "'--enable-tbo' or '--enable-tbo prefill': TBO for prefill only. "
            "'--enable-tbo all': TBO for both prefill and decode.",
        )
        # 【译·help】路由MoE的传输方式。'high-throughput'和'low-latency'选择MORI的两种模式，
        #   'rccl'选择ATOM原生的RCCL MoE传输，'none'强制使用DP AllGather/ReduceScatter的回退方案。
        #   默认自动检测MORI。
        parser.add_argument(
            "--all2all-backend",
            nargs="?",
            const="high-throughput",
            default=None,
            choices=["high-throughput", "low-latency", "rccl", "none"],
            help="Routed MoE transport. 'high-throughput' and 'low-latency' "
            "select MORI modes, 'rccl' selects ATOM's native RCCL MoE "
            "transport, and 'none' forces the DP "
            "AllGather/ReduceScatter fallback. Default is auto-detect MORI.",
        )
        # 【译·help】MoE实现。'standard'使用现有的prepare/GEMM/finalize路径；'mega'使用融合的FlyDSL MegaMoE。
        parser.add_argument(
            "--moe-backend",
            type=str,
            default="standard",
            choices=["standard", "mega"],
            help="MoE implementation. 'standard' uses the existing "
            "prepare/GEMM/finalize path; 'mega' uses fused FlyDSL MegaMoE.",
        )
        # 【译·help】投机解码方法
        parser.add_argument(
            "--method",
            type=str,
            default=None,
            choices=["mtp", "eagle3", "dspark"],
            help="Speculative method",
        )
        # 【译·help】每次迭代生成的投机token数（草稿模型自回归运行这么多次）
        parser.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=1,
            help="Number of speculative tokens to generate per iteration (draft model runs this many times autoregressively)",
        )
        # 【译·help】独立草稿模型checkpoint的路径。--method eagle3时必填；--method dspark时可选
        #   （DFlash骨干的草稿模型如Kimi-K3-DSpark需要；V4-Pro-DSpark的草稿就在目标checkpoint里，不需要）。
        parser.add_argument(
            "--draft-model",
            type=str,
            default=None,
            help="Path to a standalone draft-model checkpoint. Required when "
            "--method eagle3; optional for --method dspark (needed for the "
            "DFlash-backbone drafts such as Kimi-K3-DSpark, omitted for "
            "V4-Pro-DSpark which ships inside the target checkpoint).",
        )
        # 【译·help】调试/压测开关：强制一个固定的投机解码平均接受长度（AL），范围[1, num_speculative_tokens + 1]。
        #   设置后，拒绝采样器忽略草稿与目标的真实一致性，强制接受草稿token，使测得的接受长度收敛到这个值。
        #   AL包含目标模型自己必定产生的那个token，因此与vLLM的synthetic_acceptance_length、
        #   SGLang的SGLANG_SIMULATE_ACC_LEN单位相同，公开的标准AL可以直接传入。
        #   只在开启投机解码时有意义；不设置即关闭。
        parser.add_argument(
            "--spec-decode-acceptance-length",
            type=float,
            default=None,
            help="Debug/benchmark knob: force a fixed speculative-decoding mean "
            "acceptance length (AL) in [1, num_speculative_tokens + 1]. When "
            "set, the rejection sampler ignores the real draft/target agreement "
            "and force-accepts draft tokens so the measured accept length "
            "converges to this value. AL counts the target's own guaranteed "
            "token, so it is the same unit as vLLM's synthetic_acceptance_length "
            "and SGLang's SGLANG_SIMULATE_ACC_LEN and a published golden AL can "
            "be passed through unchanged. Only meaningful with a speculative "
            "method; leave unset to disable.",
        )
        # 【译·help】与--spec-decode-acceptance-length是同一个开关，用[0, 1]的平均接受率表示
        #   （accepted_draft/total_draft），即 (length - 1) / num_speculative_tokens。
        #   与--spec-decode-acceptance-length互斥。
        parser.add_argument(
            "--spec-decode-acceptance-rate",
            type=float,
            default=None,
            help="The same knob as --spec-decode-acceptance-length, expressed as "
            "a mean acceptance rate in [0, 1] (accepted_draft/total_draft), i.e. "
            "(length - 1) / num_speculative_tokens. Mutually exclusive with "
            "--spec-decode-acceptance-length.",
        )
        # 【译·help】异步引擎中一起批处理的最大token数
        parser.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=16384,
            help="Maximum number of tokens to batch together in async engine",
        )
        # 【译·help】chunked prefill时，把单个请求每步的prefill量限制在这么多token。0表示不限制
        #   （请求只受max_num_batched_tokens约束）。可让长prefill与decode交错，降低ITL。
        parser.add_argument(
            "--long-prefill-token-threshold",
            type=int,
            default=0,
            help=(
                "For chunked prefill, cap a single request's per-step prefill "
                "size at this many tokens. 0 disables the cap (request is only "
                "bounded by max_num_batched_tokens). Useful to interleave long "
                "prefills with decode for lower ITL."
            ),
        )
        # 【译·help】MLA chunked prefill的预算（token数）。默认使用max_num_batched_tokens。
        #   （注：这里的默认值直接写成16384，恰好等于max_num_batched_tokens的默认值）
        parser.add_argument(
            "--attn-prefill-chunk-size",
            type=int,
            default=16384,
            help=(
                "MLA chunked-prefill budget in tokens. Default uses "
                "max_num_batched_tokens."
            ),
        )
        # 【译·help】对于有每请求状态的模型（DeepSeek-V4的压缩器环形缓冲区和滑动窗口、GDN的递归状态），
        #   每N个上下文token保存一次状态checkpoint，以后的前缀命中可以从那里恢复。
        #   比N短的prompt不发布任何checkpoint，这让该功能在从不复用前缀的负载上没有开销。
        #   必须是前缀缓存hash block大小的倍数。prefill的chunk会对齐到这些位置，所以它也会量化chunk边界。
        #   0完全关闭状态checkpoint。-1保持开启但不放置间隔档位：此时只在请求表现出需要的地方
        #   以及每个prompt自己的末尾保存checkpoint，而这正是agent流量实际恢复的位置——每个档位都要让
        #   保存它的prompt多做一个prefill chunk；在实测trace上，间隔档位的写入量约是另外两种放置方式的30倍，
        #   而那两种方式已经能达到同样的复用。
        parser.add_argument(
            "--state-checkpoint-interval-tokens",
            type=int,
            default=8192,
            help=(
                "For models with per-request state (DeepSeek-V4 compressor "
                "ring and sliding window, GDN recurrent state), keep a state "
                "checkpoint every N tokens of context so a later prefix hit "
                "can resume there. "
                "A prompt shorter than N publishes nothing, which is what keeps "
                "the feature free on workloads that never reuse a prefix. Must "
                "be a multiple of the prefix-cache hash block size. Prefill "
                "chunks are aligned to these positions, so this also quantizes "
                "chunk boundaries. "
                "0 disables state checkpointing entirely. -1 keeps it on but "
                "places no interval rungs: checkpoints are then taken only "
                "where a request is seen to want one and at each prompt's own "
                "end, which is where agentic traffic actually resumes — every "
                "rung costs the prompt that keeps it an extra prefill chunk, "
                "and on measured traces the interval ladder is ~30x the writes "
                "for reuse the other two placements already reach."
            ),
        )
        # 【译·help】DeepSeek-V4.1：prefill时，decoder（最后一个KV源层及其之后的层）只在每个请求最后一个窗口的行上运行，
        #   窗口在那里截断，最后一层只算最后一行（V4.1报告中的SWA Bounded Replay）。prefill更快；
        #   与完整prefill在数值上不等价。默认开启；--no-decoder-swa-bounded-replay关闭。对其他模型无效。
        #   （两个别名写法指向同一个字段，dest指定字段名）
        parser.add_argument(
            "--decoder-swa-bounded-replay",
            "--enable-decoder-swa-bounded-replay",
            dest="enable_decoder_swa_bounded_replay",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "DeepSeek-V4.1: in prefill, run the decoder (the last KV-source "
                "layer and those after it) only on each request's last window "
                "rows, their windows truncated there, and the last layer on its "
                "last row (the V4.1 report's SWA Bounded Replay). Faster "
                "prefill; not numerically equivalent to a full prefill. On by "
                "default; --no-decoder-swa-bounded-replay turns it off. No "
                "effect on other models."
            ),
        )
        # 【译·help】DeepSeek-V4.1：前缀命中时不依赖状态checkpoint恢复，而是重算命中点下方的hash block来重建
        #   滑动窗口状态（不改写任何已缓存的行）；不保存状态checkpoint，所以也不会为此切分prefill chunk。
        #   与从checkpoint恢复在数值上不等价。默认开启；--no-encoder-swa-bounded-replay则保留checkpoint。对其他模型无效。
        parser.add_argument(
            "--encoder-swa-bounded-replay",
            "--enable-encoder-swa-bounded-replay",
            dest="enable_encoder_swa_bounded_replay",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "DeepSeek-V4.1: a prefix hit resumes without a state checkpoint, "
                "rebuilding the sliding-window state by recomputing the hash "
                "block below the hit (no cached row is rewritten); no state "
                "checkpoint is kept, so no prefill chunk is cut for one. Not "
                "numerically equivalent to resuming a checkpoint. On by "
                "default; --no-encoder-swa-bounded-replay keeps checkpoints. No "
                "effect on other models."
            ),
        )
        # 【译·help】允许因缺少checkpoint而被拒绝的命中放置一个自己的档位。--no-state-checkpoint-demand
        #   则只保留prompt末尾锚点这一种放置方式。在实测trace上，需求档位占全部checkpoint写入的47%，
        #   但只有2.8%被读回，而锚点有85.2%，所以这种档位的写入流量造成的淘汰代价可能超过它带来的复用。
        #   （help里的%%是argparse对%的转义）
        parser.add_argument(
            "--state-checkpoint-demand",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Let a hit that was refused for want of a checkpoint place a "
                "rung of its own. --no-state-checkpoint-demand leaves the "
                "prompt-end anchor as the only placement. On measured traces a "
                "demand is 47%% of all checkpoint writes but reads back 2.8%% of "
                "the time, against 85.2%% for an anchor, so the rung's write "
                "traffic may cost more in evictions than its reuse is worth."
            ),
        )
        # 【译·help】开启chunked prefill（默认开启）。用--no-enable_chunked_prefill关闭。
        parser.add_argument(
            "--enable_chunked_prefill",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable chunked prefill (default: enabled). "
            "Use --no-enable_chunked_prefill to disable.",
        )
        # 【译·help】打印周期性的引擎状态行（running/waiting请求数、KV cache使用率、前缀缓存命中率、
        #   prompt/生成吞吐；默认开启）。用--no-enable-log-stats关闭。对离线LLM(...)和服务端都适用。
        #   只影响这一行：[MTP Stats]和[Cache Stats]行有各自的开关（--method mtp和--enable-prefix-caching）
        #   和各自的频率。
        parser.add_argument(
            "--enable-log-stats",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Log the periodic engine-status line (running/waiting reqs, "
            "KV cache usage, prefix cache hit rate, prompt/generation "
            "throughput; default: enabled). Use --no-enable-log-stats to "
            "disable. Applies to offline LLM(...) as well as to the server. "
            "Scoped to that line only: the [MTP Stats] and "
            "[Cache Stats] lines have their own gates (--method mtp and "
            "--enable-prefix-caching) and keep their own cadences.",
        )
        # 【译·help】引擎状态行之间的秒数（默认10，与vLLM一致）。必须>0。--no-enable-log-stats时忽略。
        parser.add_argument(
            "--throughput-log-interval",
            type=float,
            default=10.0,
            help="Seconds between engine-status lines (default: 10, matching "
            "vLLM). Must be > 0. Ignored when --no-enable-log-stats.",
        )
        # 【译·help】引擎状态行中前缀缓存命中率背后的滑动窗口请求数（默认1000，与vLLM一致）。必须>0。
        #   只有状态行是窗口化的；/metrics和[Cache Stats]保持累计值。
        parser.add_argument(
            "--cache-hit-rate-window",
            type=int,
            default=1000,
            help="Requests in the sliding window behind the engine-status "
            "line's prefix cache hit rate (default: 1000, matching vLLM). "
            "Must be > 0. Only the status line is windowed; /metrics and "
            "[Cache Stats] stay cumulative.",
        )
        # 【译·help】让PP下一个prefill请求的各chunk运行时间相等，以缩短流水线的填充/排空。
        #   需要--pipeline-parallel-size > 1。对单条长prefill有帮助，但多个请求同时prefill时会损失吞吐，
        #   所以面向延迟敏感的部署；见docs/dynamic_chunking_guide.md。
        parser.add_argument(
            "--enable-dynamic-chunking",
            action="store_true",
            help=(
                "Equalize the runtime of a PP prefill request's chunks to shorten "
                "pipeline fill/drain. Requires --pipeline-parallel-size > 1. Helps "
                "single-stream long prefill and costs throughput once several "
                "requests prefill at once, so it is meant for latency-bound "
                "deployments; see docs/dynamic_chunking_guide.md."
            ),
        )
        # 【译·help】在初始chunk大小（0）与等延迟预测值（1）之间插值。默认取
        #   ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR或0.75，只有设置了--enable-dynamic-chunking才读取。
        parser.add_argument(
            "--dynamic-chunking-smooth-factor",
            type=float,
            default=None,
            help=(
                "Interpolation between the initial chunk size (0) and the "
                "equal-latency prediction (1). Defaults to "
                "ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR or 0.75, read only when "
                "--enable-dynamic-chunking is set."
            ),
        )
        # 【译·help】求解出的chunk大小的下限。每多一个chunk就要再重建一次缓存的前缀，
        #   所以它限制了缩小chunk带来的额外工作量。
        parser.add_argument(
            "--dynamic-chunking-min-chunk-size",
            type=int,
            default=4096,
            help=(
                "Floor for solved chunk sizes. Every extra chunk rebuilds the "
                "cached prefix again, so this bounds how much work shrinking adds."
            ),
        )
        # 【译·help】一起批处理的最大序列数
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=512,
            help="Maximum number of sequences to batch together",
        )
        # 【译·help】GPU显存使用率（0.0到1.0）
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.9,
            help="GPU memory utilization (0.0 to 1.0)",
        )

        # 【译·help】KV传输配置，JSON字符串。
        parser.add_argument(
            "--kv-transfer-config",
            type=str,
            default="{}",
            help="KV transfer config as JSON string.",
        )

        # 【译·help】在调度下一个prompt之前施加一段延迟（延迟因子乘以上一个prompt的延迟）。
        parser.add_argument(
            "--scheduler-delay-factor",
            type=float,
            default=0.0,
            help="Apply a delay (of delay factor multiplied by previous"
            "prompt latency) before scheduling next prompt.",
        )
        # 【译·help】开启graph_marker节点，用于tracing/profile插桩。
        parser.add_argument(
            "--mark-trace",
            action="store_true",
            help="Enable graph_marker nodes for tracing/profile instrumentation.",
        )
        # 【译·help】开启单GPU内的prefill/decode分离。默认是无约束模式（普通的独立stream，不做CU屏蔽）。
        #   加上--disagg-constrained开启CU屏蔽的stream + 共享内存协调。
        parser.add_argument(
            "--enable-rapidserve",
            action="store_true",
            help="Enable intra-GPU prefill/decode disaggregation. "
            "Defaults to unconstrained mode (plain separate streams, "
            "no CU masking). Pass --disagg-constrained to enable "
            "CU-masked streams + shm coordination.",
        )
        # 【译·help】分离模式下每个prefill batch的最大序列数。不设置时默认为--max-num-seqs。
        parser.add_argument(
            "--disagg-prefill-max-num-seqs",
            type=int,
            default=None,
            help="Max sequences per prefill batch in disagg mode. "
            "Defaults to --max-num-seqs when not set.",
        )
        # 【译·help】与--enable-rapidserve一起使用，开启CU屏蔽的stream和基于共享内存的prefill/decode协调。
        #   默认（关闭）使用普通的独立stream，不做CU屏蔽。
        parser.add_argument(
            "--disagg-constrained",
            action="store_true",
            help="With --enable-rapidserve, enable CU-masked streams and "
            "shm-based prefill/decode coordination. Default (off) "
            "uses plain separate streams with no CU masking.",
        )
        # 【译·help】在线量化配置，JSON字符串。支持的量化格式：ptpc_fp8、mxfp4。JSON对象有三个字段（至少提供一个）：
        #     - global_quant_config：字符串，应用于所有层的默认量化格式。
        #     - layer_quant_config：字典，用glob模式作key的按层覆盖，对匹配的层覆盖global_quant_config。
        #     - exclude_layer：字符串或字符串列表，不做量化的层名模式。
        #   例：'{"global_quant_config": "ptpc_fp8", "layer_quant_config": {"*expert*": "mxfp4"}, "exclude_layer": "lm_head"}'
        #   （type=json.loads：argparse直接用json.loads把命令行字符串解析成dict）
        parser.add_argument(
            "--online_quant_config",
            type=json.loads,
            default=None,
            help=(
                "Online quantization config as a JSON string. "
                "Supported quantization formats: ptpc_fp8, mxfp4. "
                "The JSON object has three fields "
                "(at least one must be provided):\n"
                '  - "global_quant_config": str, default quantization '
                "format applied to all layers.\n"
                '  - "layer_quant_config": dict, per-layer overrides '
                "using glob patterns as keys. "
                "Overrides global_quant_config for matched layers.\n"
                '  - "exclude_layer": str or list[str], layer name '
                "patterns to exclude from quantization.\n"
                "Example:\n"
                """  '{"global_quant_config": "ptpc_fp8", """
                """"layer_quant_config": {"*expert*": "mxfp4"}, """
                """"exclude_layer": "lm_head"}'"""
            ),
        )
        # 【译·help】加载模型config后要覆盖的HF config属性，JSON对象。例：'{"use_index_cache": true, "index_topk_freq": 4}'
        parser.add_argument(
            "--hf-overrides",
            type=json.loads,
            default=None,
            help=(
                "JSON object of HF config attributes to override after loading "
                "the model config. Example: "
                '\'{"use_index_cache": true, "index_topk_freq": 4}\''
            ),
        )
        # 【译·help】DSpark动态配置，JSON字典，直接解析成DSparkConfig对象（不用环境变量）。支持的key：
        #     - confidence_schedule：bool，开启按置信度调度的验证（每个请求的验证长度ell_r）。
        #     - ragged：bool，开启每请求不等长的验证（batch级别不做q填充）。
        #     - ragged_graph_sizes：str，逗号分隔的每序列CUDA graph查询长度桶，例如"1,3,6"或"8"。
        #     - q_buckets：str，旧的batch统一q桶验证路径使用的CUDA graph查询长度桶。
        #     - disable_sps_calib：bool，跳过SPS校准，使用合成的桩。
        parser.add_argument(
            "--dspark-config",
            type=json.loads,
            default=None,
            help=(
                "DSpark dynamic config as a JSON dict, parsed straight into a "
                "DSparkConfig object (no env vars). Supported keys:\n"
                '  - "confidence_schedule": bool, enable confidence-scheduled '
                "verification (per-request verify length ell_r).\n"
                '  - "ragged": bool, enable per-request ragged verify '
                "(no batch-level q padding).\n"
                '  - "ragged_graph_sizes": str, comma-separated per-seq CUDA-graph '
                'query-length buckets to capture, e.g. "1,3,6" or "8".\n'
                '  - "q_buckets": str, CUDA-graph query-length buckets for the '
                "older batch-uniform q-bucket verify path.\n"
                '  - "disable_sps_calib": bool, skip SPS calibration and use the '
                "synthetic stub.\n"
                "Example:\n"
                """  '{"confidence_schedule": true, "ragged": true, """
                """"ragged_graph_sizes": "8"}'"""
            ),
        )
        dcp_group = parser.add_argument_group("DCP options")
        # 【译·help】DCP（decode上下文并行）的开关，一个JSON字典，直接解析成DCPConfig（不提供逐字段的命令行参数）；
        #   未知key会报错。细节和约束见docs/context_parallel_guide.md。
        #     interleave_size（int，1）：KV交错粒度S——token i位于rank (i // S) % W；1表示轮转。
        #     enable_query_replication（bool，TRUE）：在加载时复制q_proj，省掉每步decode的Q AllGather。
        #     enable_project_before_merge（bool，TRUE）：在输出合并之前先投影V，按kv_lora_rank/v_head_dim缩小它。
        #     comm_backend（str，a2a）：'a2a' = 一次all-to-all；'ag_rs' = AllGather LSE + ReduceScatter输出。
        #   后三个默认采用新行为（中间两个在不支持时自动关闭），所以对照组必须显式传入旧值——什么都不传会运行新路径。
        #   （add_argument_group：在--help里把这些参数单独分成一组显示）
        dcp_group.add_argument(
            "--dcp-config",
            type=json.loads,
            default=None,
            help=(
                "DCP (Decode Context Parallel) knobs as one JSON dict, parsed "
                "straight into DCPConfig (no per-field flags); unknown keys "
                "raise. Details and constraints: "
                "docs/context_parallel_guide.md.\n"
                '  "interleave_size" (int, 1): KV interleave granularity S -- '
                "token i lives on rank (i // S) %% W; 1 = round-robin.\n"
                '  "enable_query_replication" (bool, TRUE): drop the per-step '
                "decode AllGather Q by replicating q_proj at load time.\n"
                '  "enable_project_before_merge" (bool, TRUE): project V '
                "before the output merge, shrinking it by "
                "kv_lora_rank/v_head_dim.\n"
                "  \"comm_backend\" (str, a2a): 'a2a' = one all-to-all; "
                "'ag_rs' = AllGather LSE + ReduceScatter output.\n"
                "The last three default to the NEW behaviour (and the middle "
                "two auto-disable where unsupported), so a control run must "
                "pass the old values explicitly -- passing nothing re-runs the "
                "new path.\n"
                'Example: \'{"interleave_size": 16, "enable_query_replication": true}\''
            ),
        )
        eplb_group = parser.add_argument_group("EPLB options")
        # 【译·help】开启EPLB运行时负载监控和专家重平衡。
        eplb_group.add_argument(
            "--eplb-enable",
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB runtime load monitoring and expert rebalance.",
        )
        # 【译·help】EPLB配置，JSON字典，直接解析成EPLBConfig对象（不提供逐字段的命令行参数）。
        #   --eplb-enable开启EPLB；--eplb-config只用于调参。支持的key：
        #     - load_window_size：int，为EPLB负载统计累积的非dummy forward数。
        #     - rebalance_interval：int，EPLB重平衡门控的forward间隔。
        #     - rebalance_layers_per_chunk：int，每个重平衡chunk迁移的MoE层数。
        #     - num_redundant_experts：int，每个MoE层为EPLB副本额外准备的物理专家槽位数。
        #     - rebalance_min_balancedness：float，均衡度至少为该值时跳过重平衡。
        #     - rebalance_balancedness_agg："min"|"mean"，均衡度门控使用的层聚合方式。
        #     - p2p_batch_chunk_size：int，迁移专家权重时使用的P2P批次大小。
        #     - placement_policy："naive"|"biased"，如何使用冗余预算：'naive'（分散）或
        #       'biased'（把top-K最热的专家完整复制到所有GPU）。
        eplb_group.add_argument(
            "--eplb-config",
            type=json.loads,
            default=None,
            help=(
                "EPLB config as a JSON dict, parsed straight into an EPLBConfig "
                "object (no per-field flags). --eplb-enable turns EPLB on; "
                "--eplb-config only tunes it. Supported keys:\n"
                '  - "load_window_size": int, non-dummy forwards accumulated '
                "for EPLB load stats.\n"
                '  - "rebalance_interval": int, forward-pass interval for '
                "EPLB rebalance gating.\n"
                '  - "rebalance_layers_per_chunk": int, MoE layers migrated '
                "per EPLB rebalance chunk.\n"
                '  - "num_redundant_experts": int, extra physical expert '
                "slots per MoE layer for EPLB replicas.\n"
                '  - "rebalance_min_balancedness": float, skip EPLB '
                "rebalance when balancedness is at least this value.\n"
                '  - "rebalance_balancedness_agg": "min"|"mean", layer '
                "aggregation used by the EPLB balancedness gate.\n"
                '  - "p2p_batch_chunk_size": int, P2P batch chunk size used '
                "while migrating expert weights.\n"
                '  - "placement_policy": "naive"|"biased", how to spend the '
                "redundant budget: 'naive' (spread) or 'biased' (fully "
                "replicate top-K hottest experts to all GPUs).\n"
                "Example:\n"
                """  '{"num_redundant_experts": 8, "placement_policy": """
                """"biased"}'"""
            ),
        )

        return parser

    # 从argparse的解析结果构造EngineArgs
    # @classmethod：第一个参数是类本身（cls），常用来写"另一种构造方式"
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        """Create an EngineArgs instance from parsed command-line arguments."""
        # 【译】从解析好的命令行参数创建一个 EngineArgs 实例。
        # dataclasses.fields(cls)：列出dataclass的所有字段
        attrs = [attr.name for attr in fields(cls)]
        # 只取args里与字段同名的属性；示例自己加的参数（--temperature等）和--served-model-name不是字段，被忽略
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

    # EngineArgs -> 传给LLMEngine的kwargs
    # 大部分字段同名透传；少数需要改名、解析字符串、或打包成子配置对象（CompilationConfig、SpeculativeConfig等）
    def _get_engine_kwargs(self) -> dict:
        """Get common engine initialization kwargs.

        Most fields are directly passed through with the same name.
        Only handles special cases that need transformation.
        """
        # 【译】获取通用的引擎初始化kwargs。
        #   大多数字段直接以相同的名字透传。这里只处理需要转换的特殊情况。
        # model单独作为LLMEngine的第一个位置参数传入，所以从kwargs里排除
        kwargs = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "model"
        }

        # Handle special transformations
        # 【译】处理特殊的转换
        # 改名：命令行叫block_size，Config里叫kv_cache_block_size
        kwargs["kv_cache_block_size"] = kwargs.pop("block_size")
        # 三个编译相关参数打包成CompilationConfig（第5批）
        #   cudagraph_mode："FULL"字符串 -> CUDAGraphMode["FULL"]枚举成员
        #   cudagraph_capture_sizes："[1,2,4]"字符串 -> [1, 2, 4]；空字符串时为None（由Config决定默认值）
        kwargs["compilation_config"] = CompilationConfig(
            level=kwargs.pop("level"),
            cudagraph_mode=CUDAGraphMode[kwargs.pop("cudagraph_mode")],
            cudagraph_capture_sizes=(
                parse_size_list(kwargs.pop("cudagraph_capture_sizes"))
                if self.cudagraph_capture_sizes
                else None
            ),
        )
        # 投机解码参数打包成SpeculativeConfig；没开投机解码时把这几个kwargs都删掉，speculative_config=None
        # dict.pop(key)：取出并删除；这样它们不会以散的kwargs传给Config
        if self.method and self.num_speculative_tokens > 0:
            method = kwargs.pop("method")
            num_spec_tokens = kwargs.pop("num_speculative_tokens")
            draft_model = kwargs.pop("draft_model")
            synthetic_acceptance_rate = kwargs.pop("spec_decode_acceptance_rate")
            synthetic_acceptance_length = kwargs.pop("spec_decode_acceptance_length")
            if method == "eagle3" and not draft_model:
                raise ValueError("--draft-model is required when --method eagle3.")
            if draft_model and method == "mtp":
                raise ValueError(
                    "--draft-model is not supported with --method mtp: the MTP "
                    "draft is loaded from the target checkpoint."
                )
            # 没有独立草稿模型时（MTP），草稿层从目标模型自己的checkpoint加载，所以model填目标模型路径
            kwargs["speculative_config"] = SpeculativeConfig(
                method=method,
                model=draft_model or self.model,
                num_speculative_tokens=num_spec_tokens,
                synthetic_acceptance_rate=synthetic_acceptance_rate,
                synthetic_acceptance_length=synthetic_acceptance_length,
            )
        else:
            kwargs.pop("method")
            kwargs.pop("num_speculative_tokens")
            kwargs.pop("draft_model")
            kwargs.pop("spec_decode_acceptance_rate")
            kwargs.pop("spec_decode_acceptance_length")
            kwargs["speculative_config"] = None

        # --enable-tbo [prefill|all] → enable_tbo + enable_tbo_decode
        # 【译】--enable-tbo [prefill|all] → enable_tbo + enable_tbo_decode 两个布尔值
        # 不传 -> (False, False)；prefill -> (True, False)；all -> (True, True)
        tbo_mode = kwargs.pop("enable_tbo", None)
        kwargs["enable_tbo"] = tbo_mode is not None
        kwargs["enable_tbo_decode"] = tbo_mode == "all"

        # --all2all-backend的取值拆成两个Config字段：是否低延迟模式 + 用哪种传输
        #   不传(None) -> auto（自动检测MORI）；high-throughput/low-latency -> mori；rccl -> rccl；none -> none
        all2all_backend = kwargs.pop("all2all_backend", None)
        kwargs["enable_low_latency"] = all2all_backend == "low-latency"
        kwargs["moe_all2all_backend"] = {
            None: "auto",
            "high-throughput": "mori",
            "low-latency": "mori",
            "rccl": "rccl",
            "none": "none",
        }[all2all_backend]

        # ATOM_KV_OFFLOAD → folded into kv_transfer_config, so offload does not
        # compete with a P/D connector (or a launcher owning that flag) for it.
        # 【译】ATOM_KV_OFFLOAD → 合并进 kv_transfer_config，这样offload就不会与P/D connector
        #   （或掌管这个参数的启动脚本）争用它。
        offload = kv_offload_connector_config(
            envs.ATOM_KV_OFFLOAD, envs.ATOM_KV_OFFLOAD_EXTRA_CONFIG
        )
        kwargs["kv_transfer_config"] = compose_kv_offload_config(
            kwargs["kv_transfer_config"], offload
        )
        if offload is not None:
            logger.info(
                "ATOM_KV_OFFLOAD: kv_transfer_config=%s", kwargs["kv_transfer_config"]
            )

        # --dspark-config (JSON dict) → DSparkConfig object, passed through as
        # Config.dspark (no env vars).
        # 【译】--dspark-config（JSON字典）→ DSparkConfig对象，作为 Config.dspark 透传（不用环境变量）。
        kwargs["dspark"] = DSparkConfig.from_dict(kwargs.pop("dspark_config", None))
        # --eplb-config (JSON dict) → EPLBConfig object (--eplb-enable
        # is the master switch, --eplb-config only tunes it).
        # 【译】--eplb-config（JSON字典）→ EPLBConfig对象（--eplb-enable是总开关，--eplb-config只用于调参）。
        kwargs["eplb_config"] = EPLBConfig.from_dict(kwargs.pop("eplb_config"))
        # --dcp-config (JSON dict) → DCPConfig object, passed through as
        # Config.dcp_config.
        # 【译】--dcp-config（JSON字典）→ DCPConfig对象，作为 Config.dcp_config 透传。
        kwargs["dcp_config"] = DCPConfig.from_dict(kwargs.pop("dcp_config"))

        # DP topology -> ParallelConfig. `data_parallel_size` stays in kwargs
        # too: LLMEngine still reads the loose kwarg on the legacy path.
        # 【译】DP拓扑 -> ParallelConfig。data_parallel_size 也同时保留在kwargs里：
        #   LLMEngine 在旧路径上仍然读取这个散的kwarg。
        parallel_config_kwargs = {
            "data_parallel_size": kwargs["data_parallel_size"],
            "data_parallel_size_local": kwargs.pop("data_parallel_size_local"),
            "data_parallel_rank": kwargs.pop("data_parallel_rank"),
            "data_parallel_master_ip": kwargs.pop("data_parallel_master_ip"),
            "data_parallel_master_port": kwargs.pop("data_parallel_master_port"),
        }
        base_port = kwargs.pop("data_parallel_base_port")
        # 没指定时不传，让ParallelConfig用自己的默认值
        if base_port is not None:
            parallel_config_kwargs["data_parallel_base_port"] = base_port
        kwargs["parallel_config"] = ParallelConfig(**parallel_config_kwargs)

        # 启动日志里会打印这行，排查"某个参数到底有没有生效"时很有用
        logger.info(f"Engine kwargs: {kwargs}")

        return kwargs

    # 创建引擎：LLMEngine.__init__会构造Config，并启动EngineCore进程和每张GPU的ModelRunner进程（第2批）
    # tokenizer可以由调用方传入（在线服务已加载过tokenizer时复用），否则LLMEngine自己加载
    def create_engine(self, tokenizer=None) -> LLMEngine:
        """Create and return an LLMEngine instance with the configured parameters."""
        # 【译】用配置好的参数创建并返回一个 LLMEngine 实例。
        return LLMEngine(self.model, tokenizer=tokenizer, **self._get_engine_kwargs())
