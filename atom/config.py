# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import fnmatch
import hashlib
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.distributed import ProcessGroup, ReduceOp
from transformers import AutoConfig, GenerationConfig, PretrainedConfig

# plugin-related utilities
# 【译】插件相关的工具
# 解释：这行注释指的是下面的is_plugin_mode/is_vllm/PluginConfig；
#   紧挨着它的MQA_LOGITS_PRESHUFFLE_ROWS（=16）是后来插进来的，供glm5_kpool_block_size使用
from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    MQA_LOGITS_PRESHUFFLE_ROWS,
)
from atom.plugin import is_plugin_mode, is_vllm
from atom.plugin.config import PluginConfig
from atom.quant_spec import (
    NVFP4_DTYPE,
    LayerQuantConfig,
    get_quant_parser,
)
from atom.utils import envs
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group

# TYPE_CHECKING：只有类型检查器（mypy/pyright）分析代码时为True，运行时恒为False
#   所以这个import只服务于类型注解，运行时不会真正执行
if TYPE_CHECKING:
    # Annotation only. Importing AITER here would put a GPU kernel build behind
    # `import atom.config`, which is what `atom.quant_spec` defers on purpose.
    # 【译】只用于类型注解。如果在这里导入AITER，`import atom.config` 就会触发一次GPU kernel构建，
    #   而 atom.quant_spec 正是有意把这件事推迟了。
    from aiter import QuantType

logger = logging.getLogger("atom")

# 本文件：ATOM的"引擎级"配置中心
#   Config：一次启动的全部配置。LLMEngine.__init__用Config(model, **kwargs)构造它，之后它随进程启动
#     被pickle传给EngineCore进程和每个ModelRunner进程，调度器、BlockManager、ModelRunner、模型层都从它读配置
#   其余是Config的子配置和辅助函数（按在文件中的顺序）：
#     KVCacheTensor / KVCacheConfig          ModelRunner把每层KV cache张量交给注意力层时用的描述（第5批）
#     CUDAGraphMode / CompilationLevel / CompilationConfig    CUDA Graph与torch.compile（第5批）
#     QuantizationConfig                     模型的量化配置：哪一层用什么量化格式（第6批）
#     glm5_* / _normalize_minimax_m3_*       GLM-5.3、MiniMax-M3的config归一化（第8批）
#     get_hf_config / get_generation_config  读取模型目录下的config.json / generation_config.json
#     ParallelConfig                         DP拓扑与进程间通信地址（第2、9批）
#     SpeculativeConfig / DSparkConfig       投机解码（第7批）
#     KVEventsConfig                         KV cache事件发布（第4批）
#     EPLBConfig                             专家负载均衡（第7批）
#     DCPConfig + qrep_* / indexer_cp_*      decode上下文并行及其门控函数（第9批）
#     Config                                 本文件的主角，见类上方的总览
#   （括号里的"第N批"指ANNOTATION_PLAN.md的阅读批次：表示"细节在那一批的文件里讲"，不是本文件里的位置）
#     set/get_current_atom_config            进程内"当前Config"的全局变量
#
# 一个命令行参数怎样变成Config字段，见arg_utils.py的文件头；Config.__post_init__的步骤见Config类上方
#
# 对比nano-vllm：nano-vllm的config.py只有一个11个字段的Config，__post_init__只有5行；
#   ATOM的Config有上百个字段和一组子配置，因为它要支持多进程DP、PP/DCP/PCP、投机解码、量化、
#   PD分离、插件模式，以及几十种模型各自不统一的HF config
#
# 一个命令行参数的完整旅程（以 --kv_cache_dtype fp8 为例，每一跳都在代码里核实过）：
#   命令行  --kv_cache_dtype fp8
#      │  argparse（EngineArgs.add_cli_args注册，choices=["bf16", "fp8"]）
#      ▼
#   args.kv_cache_dtype = "fp8" ──EngineArgs.from_cli_args（按字段名挑）──> EngineArgs.kv_cache_dtype
#      │  _get_engine_kwargs：同名透传（它不在改名/打包的名单里）
#      ▼
#   LLMEngine.__init__：只留Config里有的字段 ──Config(model, **config_kwargs)──> Config.kv_cache_dtype = "fp8"
#      │  __post_init__第13步：index_cache_dtype为None且不是V4/V4.1 -> 跟随成"fp8"
#      ▼
#   CoreManager：launch_engine_core 写好本rank的 parallel_config.data_parallel_rank / control_address，
#      │  再把 config 作为 multiprocessing.Process 的kwargs（spawn启动，config被pickle进子进程）
#      ▼
#   EngineCore（子进程，持有一份副本）──AsyncIOProcManager再spawn──> ModelRunner × TP（各自又是一份副本）
#      │  ModelRunner.__init__：set_current_atom_config(config)
#      ▼
#   建模型：各模型把 atom_config.kv_cache_dtype 作为 cache_config 交给注意力层（如gpt_oss.py、glm4_moe.py）
#   分配KV：注意力builder按 dtypes.d_dtypes[config.kv_cache_dtype] 建KV池，fp8时还把k_scale/v_scale
#     绑到层上（aiter_attention.py；MLA的池在aiter_mla.py）
#
# 由此得到两条推论：
#   (1) Config跨进程是"值传递"：子进程改自己的副本，前端进程看不到，反之亦然。
#       所以num_kvcache_blocks这种运行期才算出来的值，要靠RPC显式回传给EngineCore（engine_core.py）
#   (2) 能放进Config的东西必须可pickle：lambda、锁、打开的socket都不行


# 【科普】KV cache 与分页（paged）KV —— 读本文件和第4、5批的前置知识
#   KV cache：decode时每生成一个token，都要和之前所有token的K、V做注意力。为了不每步重算，
#     把每个token每层算出的K、V存起来，这就是KV cache。它通常是推理时最占显存的东西。
#   分页：如果给每个请求按"最大长度"预留一整段连续显存，短请求会浪费大量空间。
#     PagedAttention的做法像操作系统的虚拟内存：把整块KV显存切成固定大小的block（例如每个block存16个token），
#     一个请求需要多少就领多少个block，block在显存里不必连续，用一张表（block_table）记下
#     "这个请求的第0、1、2…个block分别是物理上的第几号block"。
#   例：block_size=16，一个请求有40个token -> 需要3个block，block_table可能是[7, 2, 15]；
#     第35个token存在第35//16=2个逻辑block，即物理block 15里的第35%16=3个位置。
#   CPU侧的"账本"（谁占了哪些block）由第4批的BlockManager管，GPU侧整块显存由ModelRunner在启动时一次性分配（第5批）。
#
# KVCacheTensor：一层注意力的KV cache张量"句柄"
# 谁创建：ModelRunner.allocate_kv_cache分配好整块KV显存后，遍历模型（和草稿模型）的每个module，
#   调用注意力元数据builder的build_kv_cache_tensor(module)。builder把属于这一层的那片显存
#   绑定到module上（module.k_cache / v_cache / k_scale / v_scale），再返回一个KVCacheTensor；
#   不是注意力层的module返回None，被跳过
# 谁使用：ModelRunner把它们收集成字典 kv_cache_data = {"layer_0": KVCacheTensor, "layer_1": ...}，
#   经set_kv_cache_data放进forward context：注意力层forward时按 f"layer_{layer_num}" 取出；
#   PD分离/KV offload的connector也从这里找到要搬运的张量（第9批）
# 对比nano-vllm：nano-vllm没有这个类，allocate_kv_cache里直接给每层的module.k_cache/v_cache赋值
#   ATOM需要一个统一的描述，因为一层可能不止K、V两块：还有FP8的scale、DSA索引器的key缓存、
#   线性注意力的递归状态等，而且这些张量还要交给KV传输和offload模块
# @dataclass：根据下面的字段声明自动生成__init__、__repr__、__eq__（第1批sampling_params.py已讲过）
@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    # 【译】描述worker应如何初始化KV cache的类。
    # 解释：现在它更像"这一层已经分配好的KV张量的清单"，由ModelRunner在分配之后构造

    # 这一层的全局层号（PP下也是全局编号，不是本stage内的序号），即kv_cache_data的key里的数字
    layer_num: int
    # 这一层的K cache和V cache（整块KV pool中属于这一层的视图，具体形状由注意力后端决定，第5、6批）
    # field(default_factory=...)：每次创建实例时调用工厂函数生成默认值
    #   不能直接写 = torch.tensor([])：那样所有实例会共享同一个张量对象（可变默认值的经典陷阱）
    k_cache: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    v_cache: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    # kv_cache_dtype=fp8时K/V的反量化scale；bf16时为None
    k_scale: torch.Tensor = None
    v_scale: torch.Tensor = None
    # 【分支】以下字段只有特定模型的层才有，见第7、8批
    # DSA sparse layers (GLM-5.2 / DeepSeek-V3.2): indexer key cache, block-major.
    # Its per-block shape follows ``--index_cache_dtype``: the FP8 row is
    # ``(num_blocks, block_size, aligned_index_dim)``, while FP4 is the packed
    # E2M1 plane ``(num_blocks, k_tiles, 4, block_size, 16)`` that
    # ``fp4_index_block_shapes`` fixes, paired with ``index_scale`` below.
    # Omitted for non-DSA layers.
    # 【译】DSA稀疏层（GLM-5.2 / DeepSeek-V3.2）：索引器的key缓存，按block为主维排列。
    #   每个block的形状取决于 --index_cache_dtype：FP8时每行是 (num_blocks, block_size, aligned_index_dim)，
    #   FP4时是打包的E2M1平面 (num_blocks, k_tiles, 4, block_size, 16)（由 fp4_index_block_shapes 确定），
    #   并与下面的 index_scale 配对。非DSA层不设置。
    index_cache: torch.Tensor | None = None
    # The e8m0 scale plane paired with ``index_cache`` under
    # ``--index_cache_dtype fp4``, block-major over the same block axis. None on
    # every other layer, FP8 indexers included. The FP4 index region is two
    # planes, not one: a mover that takes ``index_cache`` alone restores keys
    # without their exponents, and every index it computes stays in bounds while
    # doing it. Whatever moves one must move the other.
    # 【译】--index_cache_dtype fp4 时与 index_cache 配对的e8m0 scale平面，同样按block为主维、
    #   使用同一个block轴。其他所有层上都是None，包括FP8索引器。FP4索引区域是两个平面而不是一个：
    #   只搬运 index_cache 的搬运器恢复出来的key缺少指数部分，而且它算出的每个下标都不会越界（所以不会报错）。
    #   搬运其中一个的代码，必须同时搬运另一个。
    index_scale: torch.Tensor | None = None
    # ReplaySSM record buffers for linear-attention layers: this layer's slice
    # of the (k, u, g) pools.  None for every other attention type.  Carried
    # here because the layer-id -> linear-attn-index mapping already lives in
    # the builder's `build_kv_cache_tensor`.
    # 【译】线性注意力层的ReplaySSM记录缓冲区：(k, u, g) 三个pool中属于这一层的切片。
    #   其他注意力类型都是None。放在这里，是因为"层号 -> 线性注意力序号"的映射本来就在
    #   builder的 build_kv_cache_tensor 里。
    # 解释：ReplaySSM是什么（只记每个token的输入、回滚只挪游标）见envs.py ATOM_ENABLE_REPLAYSSM上方的科普
    replay_buf_k: torch.Tensor = None
    replay_buf_u: torch.Tensor = None
    replay_buf_g: torch.Tensor = None
    # True when the tensors above are a hybrid's PER-REQUEST state (GDN/KDA
    # recurrent state) rather than paged KV. Only the GDN and KDA linear-attn
    # forwards set it (`gdn_attn.py`, `kimi_mla_gdn_attn.py`, and the rtpllm/
    # sglang plugin twins); no DeepSeek-V4 path does. They belong
    # in ``kv_cache_data`` -- the linear-attention forward reads them from there
    # -- but are addressed by request slot, so no block-addressed mover may
    # touch them. The dense codec skips them; the state tier reaches the same
    # bytes through ``state_entry_views``.
    # 【译】当上面的张量是混合模型的"每请求"状态（GDN/KDA的递归状态）而不是分页KV时为True。
    #   只有GDN和KDA线性注意力的forward会设置它（gdn_attn.py、kimi_mla_gdn_attn.py，以及rtpllm/sglang
    #   插件里的对应实现）；DeepSeek-V4的任何路径都不设置。它们属于 kv_cache_data（线性注意力的forward
    #   从那里读取），但按请求槽位寻址，所以任何按block寻址的搬运器都不能碰它们。dense codec会跳过它们；
    #   状态层通过 state_entry_views 访问同样的字节。
    # 解释：分页KV按block id寻址（block_table），递归状态按请求槽位寻址，两者的搬运方式完全不同（第4、9批）
    #   什么是"每请求状态"、它为什么不能像KV block那样直接复用：见arg_utils.py EngineArgs.state_checkpoint_interval_tokens上方的【科普】
    per_request_state: bool = False


# KVCacheConfig：整个模型的KV cache配置，即所有层KVCacheTensor的列表
# 注意：当前atom/里没有地方构造它。model_runner.py里虽然有"Build KVCacheConfig"的注释，
#   实际构造的是上面说的kv_cache_data字典；vLLM插件用的是vLLM自己的同名类
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    # 【译】一个模型的KV cache配置。

    kv_cache_tensors: list[KVCacheTensor]


# 【科普】CUDA Graph 与 torch.compile —— 下面三个类（CUDAGraphMode、CompilationLevel、CompilationConfig）的前置知识
#   ① 为什么decode会被"CPU卡住"：GPU上每个算子都是一次kernel launch，由CPU发起。一次forward有几百到几千个kernel，
#     decode时batch小、每个kernel在GPU上只跑几微秒，CPU发命令的时间（每次也是几微秒）反而成了瓶颈，GPU在空等。
#   ② CUDA Graph：先把一次forward的全部kernel launch"录"下来（capture），之后用一条命令整体"回放"（replay），
#     CPU开销从几千次launch变成1次。限制：录下来的是固定的kernel序列、固定的显存地址和固定的张量形状。
#     所以：(a) 输入必须写进事先分配好的固定buffer；(b) 每种batch size要单独录一张图，
#     运行时把batch补齐（padding）到最近的已录尺寸。例：录了[1,2,4,8,16]，实际有5条序列 -> 补到8条回放。
#     (c) 每张图都要占一些显存，所以只录有限几个尺寸（cudagraph_capture_sizes）。
#   ③ torch.compile：Dynamo在运行时读Python字节码，把模型forward"抓"成一张计算图（FX graph）；
#     Inductor再把图里相邻的小算子融合、生成更快的kernel（例如把 RMSNorm 的几步合成一个kernel）。
#     guard：Dynamo记下"这张图成立的前提"（形状、dtype等），每次调用先检查，不满足就重新抓图。
#   ④ 分段（piecewise）：注意力算子每步的输入元数据（序列长度、block_table）变化大，不适合放进编译图和CUDA Graph。
#     于是在注意力处把图切开：注意力之间的稠密部分（投影、MLP、norm）各自编译并录成CUDA Graph，
#     注意力本身在两段之间正常执行。这就是 --level 3 和 PIECEWISE 模式。
#     FULL模式则是连注意力一起录成一张整图（需要注意力kernel本身支持被录制）。
#   ATOM里的实现：录图在 ModelRunner.capture_cudagraph，编译后端在 utils/backends.py（第5批）。
#
# CUDAGraphMode：CUDA Graph的录制方式（--cudagraph-mode；各模式在原生ATOM里的实际行为以下面的适用场景表为准）
# 背景见上面【科普】②（为什么能省CPU launch开销、为什么要按batch size分桶录制）
# 取值分两类：
#   单一模式：NONE / PIECEWISE / FULL，value是一个整数
#   组合模式：value是(decode用的模式, 混合batch用的模式)二元组，例如FULL_DECODE_ONLY=(FULL, NONE)
#     "混合batch"（prefill和decode在同一个batch里）是vLLM的概念；ATOM原生调度器一步只做prefill或只做decode
#     （scheduler.py的_schedule：先试prefill，没有才做decode），所以原生下只区分"decode batch"和"prefill batch"
#     注意：在Enum类体内部，FULL、NONE此时还只是普通整数2和0，所以FULL_DECODE_ONLY.value == (2, 0)
# 各成员调用下面方法的结果（已用Python验证）：
#   成员                value    decode_mode  mixed_mode  requires_piecewise_compilation  has_full_cudagraphs
#   NONE                0        NONE         NONE        否                              否
#   PIECEWISE           1        PIECEWISE    PIECEWISE   是                              否
#   FULL                2        FULL         FULL        否                              是
#   FULL_DECODE_ONLY    (2, 0)   FULL         NONE        否                              是
#   FULL_AND_PIECEWISE  (2, 1)   FULL         PIECEWISE   是                              是
#   AF_PIECEWISE        (1, 1)   PIECEWISE    PIECEWISE   是                              否
#   is_attn_ffn_piecewise()只对AF_PIECEWISE为True
# 原生ModelRunner用requires_piecewise_compilation()（_piecewise_cg_active）区分两条捕获路径：
#   True走逐片段捕获（编译切出的各段各自录制，attention在段之间eager运行），False走手动FULL整图捕获；
#   AF_PIECEWISE再用is_attn_ffn_piecewise()额外捕获attention核心（_capture_attn_ffn_graphs）
#   要不要捕获由enforce_eager决定（EngineCore在not enforce_eager时才调用capture_cudagraph），
#   据代码，原生路径里没有对NONE的单独判断（第5批）
# 命令行开放NONE、PIECEWISE、FULL、FULL_AND_PIECEWISE、AF_PIECEWISE五种，FULL_DECODE_ONLY来自vLLM，命令行不能选
# 对比nano-vllm：nano-vllm没有模式选择，只有enforce_eager开关，开启时只捕获decode整图（相当于FULL）
#
# 适用场景（原生模式；命令行默认FULL；"实际行为"按ModelRunner._piecewise_cg_active/capture_cudagraph核实）：
#   模式                实际行为（原生）                          什么时候选                             好处 / 代价
#   NONE                与FULL相同：走手动整图捕获                几乎不用；想真正关图请用--enforce-eager    —（原生ModelRunner不认NONE；只有
#                       （ModelRunner不单独判断NONE）                                                     backends.py的copy_inputs分支看它）
#   FULL                decode按capture_sizes录整张forward图      默认；多数模型的标准选择；                 好处：CPU launch开销最低，一次回放
#                       prefill batch不走图（仍跑编译后代码）     DeepSeek-V4.1必须用它（Engram stream       代价：attention kernel必须可录制；
#                                                                 fork/join要求整图），recipe里多数是FULL     超过最大capture size的batch退回eager
#   PIECEWISE           按level 3切出的稠密片段各录各的图，        attention元数据形状多变、不宜整图录的场景， 好处：attention可以是任意eager实现
#                       attention在片段之间eager运行；            如DSpark的ragged verify（recipes/DSpark.md  代价：每层attention仍有launch开销；
#                       手动整图捕获关闭                          用它）；必须--level 3                     片段间张量要放进静态buffer
#   FULL_AND_PIECEWISE  原生下等同PIECEWISE（只要含PIECEWISE就    与vLLM行为对齐时；原生下没有额外收益        同PIECEWISE
#                       走逐片段路径，decode也不录整图）
#   AF_PIECEWISE        PIECEWISE + 再单独录attention核心的图     DeepSeek-V4 + DSpark（.github/benchmark/    好处：attention也省掉launch开销
#                       （_capture_attn_ffn_graphs）              models.json的V4-Pro-DSpark配置）           代价：多一组图的录制时间和显存
#   不支持PIECEWISE类的模型会直接拒绝：例如qwen4_exp.py（Qwen3.8-Flash-Next）在非eager且requires_piecewise_compilation()时报错
#
# 选择流程：
#   要调试/排查graph问题？ ── 是 ──> --enforce-eager（不录任何图，--level照常生效）
#          │ 否
#          ▼
#   模型是DeepSeek-V4.1？ ── 是 ──> FULL（Config在cudagraph_mode为None时也会替它选FULL）
#          │ 否
#          ▼
#   用DSpark（ragged verify）？ ── 是 ──> PIECEWISE；V4系列想再压launch开销 -> AF_PIECEWISE（都要--level 3）
#          │ 否
#          ▼
#   FULL（默认）
class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    # 【译】CompilationConfig中cudagraph模式的常量。
    #   同时，NONE、PIECEWISE、FULL这几个单一成员也被当作具体的运行时模式，用于cudagraph的运行时分发。
    # 解释："运行时模式"指forward_context.cudagraph_runtime_mode：ModelRunner捕获/回放时把它设成PIECEWISE或NONE，
    #   module_dispatch_ops.py等据此判断当前是否处在分段graph里（第5批）

    # vLLM语义是"不录制CUDA Graph"；但原生ModelRunner不认NONE，仍走手动FULL整图捕获（见上面适用场景表），
    #   原生下真正不录图要用--enforce-eager
    NONE = 0
    # 分段录制：编译切出的各段（attention之间的稠密部分）各自录制成graph，attention本身在段之间eager运行
    #   需要--level 3的分段编译（详见下面CompilationConfig.cudagraph_mode的docstring，第5批）
    PIECEWISE = 1
    # 整个forward录制成一张graph
    FULL = 2
    # decode batch用FULL，prefill/decode混合batch不用graph
    FULL_DECODE_ONLY = (FULL, NONE)
    # decode用FULL，混合batch用PIECEWISE；但原生ModelRunner只要requires_piecewise_compilation()为True
    #   就走逐片段捕获路径，decode也不捕获整图
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
    # AF_PIECEWISE ("attention/FFN-wise"): PIECEWISE + attention core in its own
    # cudagraph (DSpark). Tuple so decode/mixed_mode resolve to PIECEWISE (existing
    # == PIECEWISE checks treat it as such); extra capture gated by is_attn_ffn_piecewise().
    # 【译】AF_PIECEWISE（"attention/FFN-wise"）：PIECEWISE + attention核心单独录制一张cudagraph（DSpark）。
    #   用二元组是为了让decode/mixed_mode都解析成PIECEWISE（已有的 == PIECEWISE 判断会把它当作PIECEWISE）；
    #   额外的捕获由 is_attn_ffn_piecewise() 控制。
    # 解释：value是(1, 1)而不是1，否则Enum会把它当成PIECEWISE的别名，is_attn_ffn_piecewise()就区分不出来了
    AF_PIECEWISE = (PIECEWISE, PIECEWISE)

    # decode batch使用的模式：组合模式取二元组第0项，单一模式就是自己
    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    # 混合（prefill+decode）batch使用的模式：组合模式取第1项
    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    # 只有AF_PIECEWISE返回True：在标准的逐片段graph之外，再额外捕获attention核心的graph
    # 调用方：ModelRunner.capture_cudagraph、deepseek_v4.py
    def is_attn_ffn_piecewise(self) -> bool:
        """True only for AF_PIECEWISE — gates the extra attention-core cudagraph
        (attention/FFN-wise capture) on top of the standard piecewise pieces."""
        # 【译】只对AF_PIECEWISE为True——控制在标准逐片段graph之上额外捕获attention核心的cudagraph
        #   （attention/FFN-wise捕获）。
        return self is CUDAGraphMode.AF_PIECEWISE

    # decode或混合batch任一用到PIECEWISE，就返回True
    #   （ModelRunner._piecewise_cg_active、utils/backends.py等据此选择分段录制路径）
    def requires_piecewise_compilation(self) -> bool:
        return (
            self.decode_mode() == CUDAGraphMode.PIECEWISE
            or self.mixed_mode() == CUDAGraphMode.PIECEWISE
        )

    # 组合模式里"最强"的那个：按整数值取max（FULL=2最大）
    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    # 是否有任何路径会录制FULL整图（目前只有vLLM插件的补丁调用它）
    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    # value是二元组，即decode和混合batch分开处理的组合模式
    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)


# CompilationLevel：torch.compile的编译级别（--level，含义见arg_utils.py的EngineArgs.level）
# 这里只是4个整数常量，CompilationConfig.level存的就是这些整数；各级别在代码里的落点（第5批详解）：
#   0 NO_COMPILATION  不编译，模型以eager方式运行
#   1 DYNAMO_AS_IS    ModelRunner对整个模型调用torch.compile(backend="eager")：只做Dynamo图捕获，不生成新kernel
#   2 DYNAMO_ONCE     @support_torch_compile装饰的模型交给ATOM自己的编译后端（utils/backends.py的VllmBackend），
#                     Dynamo只追踪一次，之后由自定义分发器直接调用编译好的代码，不再走Dynamo的guard检查
#   3 PIECEWISE       分段编译（默认值）：在注意力等算子处把图切开、各段分别编译；
#                     Config.__post_init__只在这一级调用set_splitting_ops_for_v1和init_with_cudagraph_sizes
# 注意：编译级别和CUDA Graph模式基本正交：FULL整图在任何--level下都能录制，
#   但PIECEWISE类模式依赖--level 3的分段编译（见下面cudagraph_mode的docstring）
#
# 适用场景：
#   --level  什么时候选                                   好处                                代价
#   0        排查"是不是编译导致的"数值/崩溃问题；           启动最快，没有Dynamo追踪的限制          没有算子融合；没有编译片段，选PIECEWISE类模式
#            小模型做功能验证（如recipes/Qwen3-8B-FP8.md）                                           时等于不录图（据代码推断：手动整图捕获被关掉、
#                                                                                                    又没有片段wrapper）
#   1        只想确认模型能被Dynamo完整抓图                 能暴露graph break                      backend="eager"不生成新kernel，几乎没有加速
#   2        少见；同样经VllmBackend编译（按标了spliting_op   Dynamo只追踪一次，之后不走guard检查      Config只在level 3才设置splitting_ops、
#            的注意力算子切图），但跳过Config的level 3设置                                            compile_sizes和cudagraph_mode默认值（第8步）
#   3        生产默认                                       分段编译 + 允许任意cudagraph_mode        首次启动编译慢（之后读/root/.cache/atom/）；
#                                                                                                    改代码后必须清缓存（CLAUDE.md）
#   另：EAGLE3草稿、以及带独立--draft-model的DSpark草稿，会把自己那份config浅拷贝后把level改成0
#     （eagle_proposer.py、dspark_proposer.py的_build_draft_model），所以草稿模型不受--level影响
# 例：--level 0 --cudagraph-mode FULL 是合法组合：不编译，但decode仍录整图（FULL录图不依赖编译）
class CompilationLevel:
    # constants for the levels of the compilation process
    # 【译】编译过程各级别的常量
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


# CompilationConfig：torch.compile与CUDA Graph的子配置，挂在Config.compilation_config上
# 来源：原生模式下由arg_utils.py的_get_engine_kwargs用--level、--cudagraph-mode、--cudagraph-capture-sizes
#   三个参数构造；插件模式由atom/plugin/config.py构造（cudagraph_mode=None，交给Config.__post_init__决定）
# 字段分两类：
#   用户可设的（init=True）：level、cudagraph_mode、cudagraph_capture_sizes等
#   运行时由编译后端回填的（init=False，不出现在__init__参数里）：local_cache_dir、traced_files、
#     compilation_time、static_forward_context
# 这个类和下面几个compute_hash的写法沿用自vLLM（VllmBackend等名字也是），所以有些字段/注释是vLLM的遗留
# 对比nano-vllm：nano-vllm不用torch.compile，CUDA Graph只由enforce_eager一个布尔值控制
@dataclass
class CompilationConfig:
    # 编译级别0~3（CompilationLevel里的常量）；dataclass默认0，但命令行默认传3
    # 调大调小的取舍：level 3启动更慢（首次要编译，之后读缓存）但运行最快；排查编译相关问题时用0（eager）
    level: int = 0
    """The level of compilation:

    - 0: no compilation.
    - 1: dynamo as is.
    - 2: dynamo once.
    - 3: piecewise compilation."""
    # 【译】编译级别：
    #
    #   - 0：不编译。
    #   - 1：dynamo原样运行。
    #   - 2：dynamo只追踪一次。
    #   - 3：分段编译。
    # （以下是被注释掉的旧代码）
    # use_cudagraph: bool = field(default_factory=lambda: 0)

    # 编译期检查用：为True时，decorators.py的字节码钩子发现forward里修改nn.Module的buffer就报错，
    #   因为CUDA Graph回放时这种修改会被静默丢掉
    use_cudagraph: bool = True

    # 本rank的编译缓存子目录（cache_dir/rank_<设备号>/<图标签>），由VllmBackend编译时回填
    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    # 录制CUDA Graph的batch size列表，来自--cudagraph-capture-sizes（默认[1,2,...,8,16,32,48,64,128,256,512]）
    #   为None时退回到下面的cuda_graph_sizes；ModelRunner通过Config.capture_sizes读取，见那里的说明
    cudagraph_capture_sizes: list[int] | None = None

    # cudagraph_capture_sizes为空时的后备；__post_init__把空列表补成[512]
    cuda_graph_sizes: list[int] = field(default_factory=list)
    """Cuda graph capture sizes
    1. if none provided, then default set to [min(max_num_seqs * 2, 512)]
    2. if one value is provided, then the capture list would follow the
    pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]
    3. more than one value (e.g. 1 2 128) is provided, then the capture list
    will follow the provided list."""
    # 【译】CUDA graph的捕获尺寸
    #   1. 如果没有提供，默认设为 [min(max_num_seqs * 2, 512)]
    #   2. 如果提供了一个值，捕获列表按如下模式生成：[1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]
    #   3. 如果提供了多个值（例如 1 2 128），捕获列表就使用给定的列表。
    # 解释：这段docstring来自vLLM，和ATOM的实际代码不一致：
    #   1. 没提供时，下面的__post_init__直接补成[512]，不看max_num_seqs；
    #   2. 只有一个值N时，Config.capture_sizes生成的是 [1..8] + range(16, N+1, 16)，步长16而不是8；
    #   3. 多个值时按给定列表，这一条一致
    # 编译调试输出目录：level 3下非空时，用depyf把Dynamo反编译出的代码写到这里（decorators.py）
    debug_dump_path: str = ""
    """The path to dump the debug information."""
    # 【译】转储调试信息的路径。

    """custom ops that are disabled"""
    # 【译】被禁用的自定义算子
    # 解释：这是一句游离的字符串（vLLM遗留），并不描述紧接着的traced_files，ATOM里也没有对应字段
    # 编译时Dynamo追踪过的所有源文件（decorators.py逐个加入），VllmBackend把它们的内容hash进缓存目录名：
    #   改了模型代码，缓存目录就会变，不会误用旧的编译产物
    traced_files: set[str] = field(default_factory=set, init=False)

    # torch.compile缓存根目录；为空时VllmBackend按"Config.compute_hash() + 追踪文件hash + 编译器hash"
    #   生成 ~/.cache/atom/torch_compile_cache/<10位hash>（CLAUDE.md要求重启前清理的就是~/.cache/atom）
    cache_dir: str = ""

    # True时用Inductor编译各段子图（backends.py的make_compiler）
    use_inductor: bool = True

    # CudaGraph compilation
    # 【译】CudaGraph编译
    # CUDA Graph模式；None表示"由Config.__post_init__决定"：level 3下默认PIECEWISE，DeepSeek-V4.1默认FULL
    #   原生模式命令行总会传一个值（默认FULL），None只出现在插件模式或直接构造Config时
    cudagraph_mode: CUDAGraphMode | None = None
    """
    The mode of the cudagraph:

    - NONE, no cudagraph capture.
    - PIECEWISE. (v1 default)
    - FULL.
    - FULL_DECODE_ONLY.
    - FULL_AND_PIECEWISE.
    - AF_PIECEWISE.

    AF_PIECEWISE ("attention/FFN-wise") mode: PIECEWISE where the attention
    core is ALSO captured into its own cudagraph with zero-copy public buffers
    (DeepSeek-V4 DSpark), so small-batch decode is all-replay with no eager
    attention gap between the dense pieces. Falls back to plain PIECEWISE
    behavior for models that don't implement the attention-core capture.

    PIECEWISE mode build piecewise cudagraph only, keeping the cudagraph
    incompatiable ops (i.e. some attention ops) outside the cudagraph
    for general flexibility.
    This is the default mode.

    FULL mode: Capture full cudagraph for all batches. Can be good for small
    models or workloads with small prompts; not supported by many backends.
    Generally for performance FULL_AND_PIECEWISE is better.

    FULL_DECODE_ONLY mode: Capture full cudagraph for decode batches only.
    Mixed prefill-decode batches are run without cudagraphs. Can be good for
    decode instances in a P/D setup where prefill is not as important so we
    can save some memory.

    FULL_AND_PIECEWISE mode: Capture full cudagraph for decode batches and
    piecewise cudagraph for prefill and mixed prefill-decode batches.
    This is like the most performant mode for most models.

    Currently, the cudagraph mode is only used for the v1 engine.
    Note that the cudagraph logic is generally orthogonal to the
    compilation logic. While piecewise cudagraphs require piecewise
    compilation (level=PIECEWISE and non-empty splitting_ops), full
    cudagraphs are supported with and without compilation.

    Warning: This flag is new and subject to change in addition
    more modes may be added.
    """
    # 【译】cudagraph的模式：
    #
    #   - NONE：不捕获cudagraph。
    #   - PIECEWISE。（v1的默认值）
    #   - FULL。
    #   - FULL_DECODE_ONLY。
    #   - FULL_AND_PIECEWISE。
    #   - AF_PIECEWISE。
    #
    #   AF_PIECEWISE（"attention/FFN-wise"）模式：在PIECEWISE的基础上，attention核心也用零拷贝的公共缓冲区
    #   单独捕获成一张cudagraph（DeepSeek-V4 DSpark），这样小batch的decode全程都是回放，稠密片段之间
    #   没有eager运行的attention空档。对没有实现attention核心捕获的模型，行为退回到普通PIECEWISE。
    #
    #   PIECEWISE模式只构建分段cudagraph，把与cudagraph不兼容的算子（即某些attention算子）留在cudagraph之外，
    #   以保持通用性。这是默认模式。
    #
    #   FULL模式：为所有batch捕获完整的cudagraph。对小模型或prompt较短的负载可能有好处；很多后端不支持。
    #   一般来说，FULL_AND_PIECEWISE的性能更好。
    #
    #   FULL_DECODE_ONLY模式：只为decode batch捕获完整cudagraph。prefill-decode混合batch不使用cudagraph。
    #   适合P/D分离中的decode实例：那里prefill没那么重要，因此可以省一些显存。
    #
    #   FULL_AND_PIECEWISE模式：为decode batch捕获完整cudagraph，为prefill和prefill-decode混合batch
    #   捕获分段cudagraph。对大多数模型来说，这大概是性能最好的模式。
    #
    #   目前cudagraph模式只用于v1引擎。
    #   注意，cudagraph的逻辑与编译逻辑大体正交。分段cudagraph需要分段编译（level=PIECEWISE且splitting_ops非空），
    #   而完整cudagraph无论是否编译都支持。
    #
    #   警告：这个参数是新加的，可能会变化，也可能增加更多模式。
    # 解释：这段docstring大部分来自vLLM，"v1""默认模式"说的是vLLM。ATOM命令行默认传FULL，
    #   只有传入None（插件模式或直接构造Config）时，Config.__post_init__才在level 3下落到PIECEWISE
    # 解释：ATOM原生ModelRunner对FULL_AND_PIECEWISE走的是逐片段捕获路径（_piecewise_cg_active为True），
    #   decode并不捕获整图；上面译文里"NONE：不捕获"在原生下也不成立：NONE同FULL走整图捕获，
    #   只有--enforce-eager才不录图。见上面CUDAGraphMode的适用场景表（第5批）

    # 累计的编译耗时（秒），由编译后端在每次编译后累加，用于日志
    compilation_time: float = field(default=0.0, init=False)

    # 分段编译时在哪些算子处切图；None时由set_splitting_ops_for_v1填成4个注意力算子
    splitting_ops: list[str] | None = None
    """A list of ops to split the full graph into subgraphs, used in piecewise
    compilation."""
    # 【译】分段编译时用来把完整计算图切成多个子图的算子列表。

    # （以下是被注释掉的旧代码）
    # splitting_ops: Optional[list[str]] = field(default_factory=list)

    cudagraph_copy_inputs: bool = False
    """Whether to copy input tensors for
    cudagraph. If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True, and the compiler will copy the input to an
    internally managed buffer. Default is False.
    Note that this flag is only effective when cudagraph_mode is PIECEWISE.
    """
    # 【译】是否为cudagraph复制输入张量。如果调用方能保证始终使用同一组输入缓冲区，可以设为False；
    #   否则应设为True，编译器会把输入复制到内部管理的缓冲区里。默认是False。
    #   注意这个参数只在cudagraph_mode为PIECEWISE时生效。
    # 解释：ATOM的ModelRunner预先分配固定的输入缓冲区（forward_vars），每步把数据写进去，所以不需要复制

    inductor_compile_config: dict = field(default_factory=dict)
    """Additional configurations for inductor.
    - None: use default configurations."""
    # 【译】传给inductor的额外配置。
    #   - None：使用默认配置。

    compile_sizes: list[int | str] | None = None
    """Sizes to compile for inductor. In addition
    to integers, it also supports "cudagraph_capture_sizes" to
    specify the sizes for cudagraph capture."""
    # 【译】让inductor（针对具体形状）编译的尺寸。除了整数，还支持写字符串 "cudagraph_capture_sizes"，
    #   表示使用cudagraph的捕获尺寸。
    # 解释：列出的尺寸会被cuda_piecewise_backend.py按静态形状单独编译一份（其余形状共用动态形状版本）

    # 科普：自定义算子（custom op）——用 torch.library 注册成"一个整体"的算子，torch.compile把它当成黑盒，
    #   不会钻进去追踪。ATOM把注意力包装成这样的算子（如 aiter.unified_attention_with_output），
    #   既能在它这里切图（见set_splitting_ops_for_v1），又避免Dynamo去追踪AITER kernel内部
    # 层名 -> 层对象的注册表。注意力层构造时把自己登记进来（例如paged_attention.py：MHA_<层号>/MLA_<层号>），
    #   torch.compile图里的自定义算子只携带layer_name字符串，运行时再到这里查回module本身，
    #   这样module内部的状态不进入图的参数列表（见module_dispatch_ops.py文件头）
    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)

    # 把compile_sizes里的字符串"cudagraph_capture_sizes"展开成实际的整数列表
    # 调用方：Config.__post_init__，只在level 3（PIECEWISE）时调用
    # 例：compile_sizes=[1, "cudagraph_capture_sizes"]，cudagraph_capture_sizes=[1,2,4]
    #   -> 去重后逐个展开 -> [1, 1, 2, 4]（展开后的结果没有再次去重）
    #   compile_sizes为None（默认）时，结果是空列表[]
    def init_with_cudagraph_sizes(self) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""
        # 【译】要完成config的初始化，需要知道cudagraph的尺寸。
        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            # 【译】对config提供的尺寸去重
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", (
                        "Unrecognized size type in compile_sizes, "
                        f"expect 'cudagraph_capture_sizes', got {x}"
                    )
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

    # 构造后的校验：level只能是0~3；cuda_graph_sizes为空时补成[512]
    def __post_init__(self):
        if self.level not in {0, 1, 2, 3}:
            raise ValueError("level must in 0-3")
        if not self.cuda_graph_sizes:
            self.cuda_graph_sizes = [512]

    # 本子配置的hash，被Config.compute_hash汇总，最终决定torch.compile缓存目录名
    # 只放"会改变计算图结构"的字段：改了level或捕获尺寸，就要用新的缓存目录
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # 【译】警告：每当这个config新增字段时，如果它会影响计算图，务必把它加进factors列表。
        #
        #   提供一个hash，唯一标识所有会影响"从input ids/embeddings到最终hidden states"这段计算图结构的配置，
        #   不包括input ids/embeddings之前和最终hidden states之后的部分。
        factors: list[Any] = []
        factors.append(self.level)
        factors.append(self.use_cudagraph)
        factors.append(self.local_cache_dir)
        factors.append(self.cudagraph_capture_sizes)
        factors.append(self.cuda_graph_sizes)

        # 把各因素转成字符串再做sha256；顺序固定，所以同样的配置总得到同样的hash
        return hashlib.sha256(str(factors).encode()).hexdigest()

    # level 3时设置默认的切图算子：4个注意力自定义算子（原生MHA、原生MLA、vLLM插件下的MHA/MLA）
    # 为什么在注意力处切：注意力的kernel参数依赖每步的元数据（序列长度、block table），
    #   放在编译段之外更灵活（第5批详解）
    # 调用方：Config.__post_init__；用户显式指定了splitting_ops时不覆盖
    def set_splitting_ops_for_v1(self):
        # NOTE: this function needs to be called only when level is
        # CompilationLevel.PIECEWISE
        # 【译】注意：只有level为 CompilationLevel.PIECEWISE 时才需要调用这个函数
        assert self.level == CompilationLevel.PIECEWISE, (
            "set_splitting_ops_for_v1 should only be called when "
            "level is CompilationLevel.PIECEWISE"
        )

        if self.splitting_ops is None:
            self.splitting_ops = [
                "aiter.unified_attention_with_output",
                "aiter.mla_attention",
                "aiter.atom_vllm_mha_attention",
                "aiter.atom_vllm_mla_attention",
            ]


# 【分支】QuantizationConfig：整个模型的"量化规格表"（B级导读；规格本身LayerQuantConfig、各解析器
#   在quant_spec.py，各层怎么按规格建参数/选kernel在第6批linear.py、第7批moe.py）
#
# 它解决的问题：同一个模型里不同层的量化方式往往不同——例如attention投影是FP8 blockscale，
#   路由专家是MXFP4，lm_head、router gate、MTP的eh_proj不量化。而每个层在__init__里只知道自己的层名
#   （prefix，例如"model.layers.3.mlp.experts"），于是需要一个全模型共享的对象来回答：
#   "名叫X的层，权重在磁盘上是什么格式？加载后要不要转成别的格式？"——这就是本类。
#   它自己不碰任何张量，只做"层名 -> LayerQuantConfig"的查表。
#
# 内部是两张结构相同的表，每张表 = 三件套：
#   - checkpoint表（global_spec / layer_pattern_specs / exclude_layers）：描述checkpoint里权重"本来"的格式，
#     来自 hf_config.quantization_config。层按它分配参数（形状、dtype、scale参数），才能把权重原样装进去
#   - 在线表（online_global_spec / online_layer_pattern_specs / online_exclude_layers）：描述加载完后
#     "要转成"的格式，来自命令行 --online_quant_config。只有 online_quant=True 时才有意义
#   三件套的含义：global_spec是默认规格；layer_pattern_specs是按层名模式覆盖默认值的列表；
#     exclude_layers里的层一律不量化（保持torch_dtype，通常是bf16）
#
# 生命周期（谁在什么时候用它）：
#   1. 构造：Config.__post_init__ 调 QuantizationConfig(hf_config, online_quant_config)，挂在 config.quant_config 上；
#      插件模式下若checkpoint没给排除项，Config会再用vLLM的 ignored_layers 填 exclude_layers。
#      个别模型会再包一层或重建（deepseek_v4.py 覆写 get_layer_quant_config、qwen4_exp.py 的子类、
#      kimi_k3/model.py 重建、sglang插件 runtime/model_arch.py 重建）
#   2. 改名：ModelRunner.__init__ 在构造模型"之前"调 remap_layer_name，把表里的checkpoint层名改写成
#      ATOM融合后的层名（gate_proj/up_proj -> gate_up_proj 等）；插件在 plugin/prepare.py、
#      sglang/runtime/model_arch.py 等处做同样的事。必须在构造前做，因为第3步的查表发生在构造期间
#   3. 查表：构造模型时，LinearBase（linear.py）、FusedMoE（moe.py）、RMSNorm（layernorm.py）、
#      激活（activation.py）、共享专家融合判断（topK.py）等各自调 get_layer_quant_config(prefix)，
#      按返回的规格决定参数的dtype和形状（例如FP4权重打包成 [N, K/2] 的uint8）、scale参数形状，
#      以及forward里调用哪个AITER GEMM / MoE kernel
#   4. 在线量化：权重加载完后（或流式加载过程中），loader.py 发现 online_quant=True，就让各层执行
#      online_quantize_weight（Linear）/ _online_quant（FusedMoE）/ online_quantize_activation（RMSNorm）；
#      这些方法用 get_layer_quant_config(prefix, use_online_quant=True) 查在线表，把权重重新量化成目标格式
#   5. 缓存键：compute_hash 被 Config.compute_hash 汇总，量化规格变化会让torch.compile缓存失效
#
# 例：DeepSeek-R1 FP8 checkpoint + --online_quant_config '{"global_quant_config": "mxfp4", ...}'
#   第3步按checkpoint表把专家建成FP8 blockscale参数，加载FP8权重；第4步按在线表把它们转成MXFP4，
#   之后推理走MXFP4 MoE kernel
#
# 适用场景：常见量化格式一览（QuantType取值见quant_spec.py；"在线名"是--online_quant_config里能写的字符串，
#   来自quant_spec.py的FORMAT_MAP；硬件结论来自docs/online_quantization_best_practices.md）
#   格式             QuantType   在线名                          scale粒度                             权重显存(相对bf16)
#   BF16（不量化）    No          —（写进exclude_layer的层）       无                                     1
#   FP8 per-tensor   per_Tensor  —（只能来自checkpoint）          整个权重1个scale，激活也是整张量1个      约1/2
#   FP8 per-token    per_Token   ptpc_fp8                        权重每输出通道1个，激活每token1个（动态） 约1/2
#   FP8 blockscale   per_1x128   per_block_fp8/per_block128_fp8  权重每128x128块1个，激活每1x128块1个      约1/2（另加少量scale）
#   MXFP8            per_1x32    mxfp8                           每32个元素共享1个E8M0 scale              约1/2
#   MXFP4            per_1x32    mxfp4                           每32个元素共享1个E8M0 scale，两个值打包   约1/4（4bit + 每32个元素8bit scale）
#                                                                成1字节（权重形状[N, K/2]）
#
#   格式             适用场景                                         好处                          代价
#   BF16             lm_head、router gate、注意力等精度敏感层；基线    无精度损失                     显存/带宽最大
#   FP8 per-tensor   checkpoint本来就是这种格式                       scale开销最小，GEMM最简单       一个离群值会拉大整张量的scale，FP8里精度最差
#   FP8 per-token    在线量化的保守通用默认（ptpc_fp8）；MI300上       逐行scale抗离群值               激活scale每步动态计算，有少量开销
#                    想省显存时的首选
#   FP8 blockscale   DeepSeek-V3/R1这类官方FP8 checkpoint             粒度介于上两者之间，             scale存储与反量化更复杂；
#                                                                    与DeepSeek发布格式一致           scale格式还分fp32/e8m0（blockscale_e8m0_scale）
#   MXFP4            占参数大头的MoE路由专家                          显存、带宽最省，decode最快       精度风险最大；只有MI350/MI355（CDNA4）有原生
#                                                                                                   MXFP4加速，MI300（CDNA3）上应改用ptpc_fp8
#   NVFP4（checkpoint）NVIDIA的FP4 checkpoint                        —                              AITER不能直接算，必须在线转成MXFP4
#                                                                                                   （见下面_validate_nvfp4_online_quant）
#   FP8格式本身也随架构变化：MI300是e4m3fnuz，MI350是e4m3fn
#   常见组合：MoE专家mxfp4 + 其余ptpc_fp8 + 排除lm_head/gate/共享专家（best_practices文档的逐步策略）
class QuantizationConfig:
    """Model-wide quantization configuration.

    API:
    - ``get_layer_quant_config(prefix)`` -> :class:`LayerQuantConfig`
    - ``global_quant_config`` property -> :class:`LayerQuantConfig`
    - ``quant_type``, ``quant_dtype``, ``is_dynamic`` convenience properties
    """

    # 【译】模型级的量化配置。
    #
    #   API：
    #   - get_layer_quant_config(prefix) -> LayerQuantConfig（按层名查规格）
    #   - global_quant_config 属性 -> LayerQuantConfig
    #   - quant_type、quant_dtype、is_dynamic 三个便捷属性

    # 把两份"原始量化描述"解析成上面说的两张表
    # config：HF config，量化描述在它的 quantization_config 字段（checkpoint导出工具写进config.json的）
    # online_quant_config：--online_quant_config 的JSON解析成的dict（EngineArgs -> Config.online_quant_config），
    #   格式见 arg_utils.py 的说明，例如 {"global_quant_config": "ptpc_fp8", "layer_quant_config": {...}, "exclude_layer": ...}
    def __init__(
        self,
        config: PretrainedConfig = None,
        online_quant_config: dict | None = None,
    ):
        # 没有HF config：构造一个"什么都不量化"的空表后直接返回。
        # 用途：需要自己从零拼一份规格表的场合，例如 deepseek_v41/layers.py 的 native_quant_config
        #   先 QuantizationConfig() 拿空表，再直接给 global_spec 赋值为V4.1原生的FP8/FP4规格；
        #   测试里也常这样构造
        if config is None:
            self.torch_dtype = torch.bfloat16
            self.hf_quant_config = None
            self.global_spec: LayerQuantConfig = LayerQuantConfig()
            self.layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
            self.exclude_layers: list[str] = []
            self.quant_method = ""
            self.online_quant = False
            self.online_quant_config_raw = None
            self.online_global_spec: LayerQuantConfig = LayerQuantConfig()
            self.online_layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
            self.online_exclude_layers: list[str] = []
            return

        # Some HF configs set torch_dtype=None; normalize to bf16 default.
        # 【译】有些HF config把torch_dtype设成None；统一成默认的bf16。
        # torch_dtype：未量化层的参数dtype；被排除的层（_is_excluded）也用它
        self.torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        # 原样保留的checkpoint量化描述；之后blockscale_e8m0_scale还会从里面读scale_fmt
        self.hf_quant_config = getattr(config, "quantization_config", None)

        # checkpoint没有量化（BF16/FP16模型）：checkpoint表只有一条"不量化、用torch_dtype"的全局规格
        if self.hf_quant_config is None:
            self.global_spec = LayerQuantConfig.no_quant(self.torch_dtype)
            self.layer_pattern_specs = []
            self.exclude_layers = []
            self.quant_method = ""
        else:
            # 量化工具/格式名，例如"fp8"、"quark"、"modelopt"、"compressed-tensors"。
            # 用途：(1) 本函数末尾据此选解析器；(2) 下面判断能否叠加在线量化；
            #   (3) 层和模型据此走工具特有的分支，例如 moe.py 读 quant_method，
            #   deepseek_v2.py 和 sglang插件对"quark"有特殊的权重处理
            self.quant_method = self.hf_quant_config.get("quant_method", "")

        # Online quantization: re-quantize float / FP8 / MXFP4 / MXFP8 / Quark
        # models at load time.
        # 【译】在线量化：在加载时对float / FP8 / MXFP4 / MXFP8 / Quark模型重新量化。
        # 只有quant_method在下面列表里时，--online_quant_config才生效；否则被静默忽略
        #   （列表外的格式，例如gguf，没有实现从它出发的重新量化）
        # online_quant：总开关。loader.py 用它判断加载后要不要跑在线量化那一遍；
        #   quant_spec.will_online_requant / validate_nvfp4_online_target 也先看它
        self.online_quant = False
        # 原始dict留一份：loader.py 在线量化结束后，把它和各层的量化结果一起写进记录文件
        #   （_save_online_quant_info），便于事后核对"这次到底按什么配置量化的"
        self.online_quant_config_raw = online_quant_config
        self.online_global_spec: LayerQuantConfig = LayerQuantConfig()
        self.online_layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        self.online_exclude_layers: list[str] = []
        if online_quant_config and self.quant_method in [
            "",
            "fp8",
            "mxfp4",
            "mxfp8",
            "quark",
            "modelopt",
            "compressed-tensors",
        ]:
            self.online_quant = True
            if self.quant_method == "compressed-tensors":
                logger.warning(
                    "Online quant with compressed-tensors is not fully supported. "
                    "Be careful about the online quant config setting when launching "
                    "the server."
                )
            # 在线表的格式由ATOM自己定义（global_quant_config / layer_quant_config / exclude_layer），
            #   所以固定用 quant_spec.py 里注册名为"online_quant"的解析器
            online_parser = get_quant_parser("online_quant")
            online_parsed_quant_config = online_parser.parse(online_quant_config)
            self.online_global_spec = online_parsed_quant_config.global_spec
            self.online_layer_pattern_specs = (
                online_parsed_quant_config.layer_pattern_specs
            )
            self.online_exclude_layers = list(online_parsed_quant_config.exclude_layers)

        if self.quant_method == "":
            return
        # Use the parser registry to build a structured ParsedQuantConfig
        # 【译】用解析器注册表构建结构化的 ParsedQuantConfig
        # 不同量化工具（fp8、quark、modelopt……）的quantization_config格式各不相同，
        #   quant_spec.py 里每种格式用 @register_quant_parser 注册一个解析器，没注册的走"_generic"通用解析器；
        #   解析器负责把各家写法统一成同一种三件套
        parser = get_quant_parser(self.quant_method)
        parsed_quant_config = parser.parse(self.hf_quant_config)
        self.global_spec = parsed_quant_config.global_spec
        self.layer_pattern_specs = parsed_quant_config.layer_pattern_specs
        self.exclude_layers = list(parsed_quant_config.exclude_layers)
        self._validate_nvfp4_online_quant()

    # 启动期的第一道检查：checkpoint含NVFP4层却完全没给在线量化配置时，立即报错
    # 背景：NVFP4是NVIDIA的FP4格式（16个元素一组、FP8 scale），AITER没有直接算它的kernel，
    #   ATOM只能在加载时把它转成AMD的MXFP4（32个元素一组、E8M0 scale）再跑
    # 两道检查的分工：这里只看"有没有在线配置"，在解析阶段就失败，用户不用等模型构建；
    #   第二道 quant_spec.validate_nvfp4_online_target 由 linear.py / moe.py 在构建每个NVFP4层时调用，
    #   检查在线配置是否真的给"这一层"指定了MXFP4目标（可能被exclude或配成别的格式）
    def _validate_nvfp4_online_quant(self):
        """Reject NVFP4 checkpoint specs when no online config was given.

        NVFP4 runs only after online conversion to MXFP4. This fails while the
        config is parsed, before any layer is built;
        `validate_nvfp4_online_target` checks each NVFP4 layer's target as it
        is built.
        """
        # 【译】没有给出在线量化配置时，拒绝NVFP4的checkpoint规格。
        #
        #   NVFP4只有在线转换成MXFP4后才能运行。这里在解析config时就失败，早于任何层的构建；
        #   validate_nvfp4_online_target 会在构建每个NVFP4层时检查它的目标格式。
        if self.online_quant:
            return
        specs = [self.global_spec, *(spec for _, spec in self.layer_pattern_specs)]
        if any(spec.quant_dtype == NVFP4_DTYPE for spec in specs):
            raise ValueError(
                "The checkpoint has NVFP4 layers, but no online quantization "
                "config was given. NVFP4 weights cannot run directly: pass "
                "--online_quant_config with an mxfp4 target, for example "
                '\'{"global_quant_config": "mxfp4"}\'.'
            )

    # FP8 blockscale（每128x128权重块 / 每1x128激活块一个scale）的scale用哪种格式存
    # 谁用：
    #   - linear.py：FP8 blockscale的LinearBase据此把weight_scale参数建成fp8_e8m0还是fp32，
    #     并决定激活量化时产出哪种scale
    #   - layernorm.py：RMSNorm融合"norm + 激活量化"时，产出的激活scale格式必须和后面GEMM期望的一致
    #   - deepseek_v4.py：分配E8M0 scale时依赖这个判断
    # 为什么要区分：E8M0 scale更小、某些AITER kernel只接受它；但它只能表示2的幂，
    #   不是2的幂的FP32 scale装进去会被舍入，等于改了权重，所以默认只对声明了ue8m0的checkpoint开启
    @property
    def blockscale_e8m0_scale(self) -> bool:
        """Whether 128x128 FP8 block scales (weight and activation) are E8M0.

        ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE decides when set. Otherwise E8M0 on
        every arch but gfx942, and only for a checkpoint whose scales are
        declared ``scale_fmt: ue8m0`` (DeepSeek-V4): those are powers of two, so
        E8M0 restates them exactly, while an FP32 scale loaded into an E8M0
        parameter would be rounded, altering the weight.
        """
        # 【译】128x128 FP8 block scale（权重和激活）是否使用E8M0格式。
        #
        #   设置了 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE 时由它决定。否则：除gfx942以外的所有架构上用E8M0，
        #   而且只针对scale声明为 scale_fmt: ue8m0 的checkpoint（DeepSeek-V4）：这些scale都是2的幂，
        #   E8M0能精确表示；而把FP32的scale装进E8M0参数会被舍入，从而改变权重。
        # 解释：E8M0只有8位指数、没有尾数，只能表示2的幂
        if envs.is_set("ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE"):
            return envs.ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE
        if (self.hf_quant_config or {}).get("scale_fmt") != "ue8m0":
            return False
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() != "gfx942"

    # -- typed API (preferred) ----------------------------------------------
    # 【译】-- 带类型的API（推荐使用） --

    # checkpoint表的默认规格，即"整个模型主要是什么格式"
    # 谁用：不关心具体层、只想知道模型整体是否量化的地方，例如 qwen3_5.py 和vLLM/SGLang插件的
    #   qwen3_5 适配用 global_quant_config.quant_dtype == bf16 判断是不是未量化模型；
    #   topK.py 在找不到共享专家的逐层规格时也退回到它
    @property
    def global_quant_config(self) -> LayerQuantConfig:
        """Alias for ``global_spec``."""
        # 【译】global_spec 的别名。
        return self.global_spec

    # 核心接口：层名 -> 这一层的LayerQuantConfig
    # 谁调用：几乎所有带权重的层在 __init__ 里调用一次，传入自己的完整层名（构造时层层传下来的prefix，
    #   如"model.layers.3.self_attn.qkv_proj"）——LinearBase、FusedMoE、RMSNorm（融合量化）、
    #   SiluAndMul（融合量化）、topK（判断共享专家能否与路由专家融合）、各模型里的特殊层
    # 参数：
    #   use_online_quant=False：查checkpoint表，用于建参数、装权重
    #   use_online_quant=True：查在线表，用于加载后的重新量化（各层的online_quantize_*方法）
    #   check_children=True：FusedMoE专用。它的prefix是专家容器（"...mlp.experts"），而排除项常写到
    #     单个专家（"...mlp.experts.0.gate_up_proj"），需要"子模块被排除也算排除"，见_is_excluded
    # 返回值被层保存下来（如 self.layer_quant_config），之后forward选kernel都看它
    def get_layer_quant_config(
        self,
        layer_name: str,
        use_online_quant: bool = False,
        *,
        check_children: bool = False,
    ) -> LayerQuantConfig:
        """Return the :class:`LayerQuantConfig` for *layer_name*.

        Resolution order:
        1. Check exclude list -> ``LayerQuantConfig.no_quant()``.
        2. fnmatch-style pattern match in ``layer_pattern_specs``.
        3. Fall back to ``global_spec``.
        """
        # 【译】返回 layer_name 对应的 LayerQuantConfig。
        #
        #   解析顺序：
        #   1. 检查排除列表 -> LayerQuantConfig.no_quant()。
        #   2. 在 layer_pattern_specs 中做fnmatch风格的模式匹配。
        #   3. 退回到 global_spec。
        if use_online_quant:
            layer_pattern_specs = self.online_layer_pattern_specs
            global_spec = self.online_global_spec
            exclude_layers = self.online_exclude_layers
        else:
            layer_pattern_specs = self.layer_pattern_specs
            global_spec = self.global_spec
            exclude_layers = self.exclude_layers

        # 1. Exclude list
        # 【译】1. 排除列表
        # 排除优先级最高：即使下面某个模式也能匹配到，被排除的层也不量化
        #   （例如全局FP8，但 lm_head、router gate 被排除，保持bf16）
        if self._is_excluded(layer_name, exclude_layers, check_children=check_children):
            return LayerQuantConfig(quant_dtype=self.torch_dtype)

        # 2. Pattern match
        # 【译】2. 模式匹配
        # 不含*的模式按"层名是模式的子串"判断（注意方向：layer_name in pattern），含*的用fnmatch通配
        # 按列表顺序取第一个命中的，所以列表里靠前的模式优先；
        #   这也是 _validate_packed_layer_specs 要拦截"同名模式规格不一致"的原因
        # 例：在线配置 layer_quant_config={"*expert*": "mxfp4"}，层名"model.layers.3.mlp.experts"
        #   命中，返回MXFP4规格；"model.layers.3.self_attn.qkv_proj"不命中，落到第3步
        for pattern, spec in layer_pattern_specs:
            if "*" not in pattern:
                if layer_name in pattern:
                    return spec
            elif fnmatch.fnmatch(layer_name, pattern):
                return spec

        # 3. Global default
        # 【译】3. 全局默认值
        return global_spec

    # -- convenience properties (delegate to global_spec) ---------------------
    # 【译】-- 便捷属性（委托给 global_spec） --
    # 下面三个只反映"全局默认规格"，不代表某一层的实际规格；按层判断应调用 get_layer_quant_config。
    # 它们是早期"整模型只有一种量化格式"时代的接口，保留给只需整体信息的老代码
    #   is_dynamic：激活scale是推理时现算（动态），还是checkpoint里存好的静态scale

    @property
    def quant_type(self) -> "QuantType":
        return self.global_spec.quant_type

    @property
    def quant_dtype(self) -> torch.dtype | str:
        return self.global_spec.quant_dtype

    @property
    def is_dynamic(self) -> bool:
        return self.global_spec.is_dynamic

    # -- other methods ------------------------------------------------------
    # 【译】-- 其他方法 --

    # 唯一调用方：Config.compute_hash，汇总后决定torch.compile缓存目录（~/.cache/atom/下）
    # 为什么量化要进hash：量化规格决定每层调用哪个GEMM/MoE op、参数dtype和形状，图结构随之不同；
    #   同一个模型换一种 --online_quant_config 启动，如果命中旧缓存就会跑错图
    # 注意只把两张表的三件套放进去：torch_dtype、quant_method 已经体现在规格里了
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # 【译】警告：每当这个config新增字段时，如果它会影响计算图，务必把它加进factors列表。
        #
        #   提供一个hash，唯一标识所有会影响"从input ids/embeddings到最终hidden states"这段计算图结构的配置，
        #   不包括input ids/embeddings之前和最终hidden states之后的部分。
        factors: list[Any] = []
        factors.append(self.global_spec)
        factors.append(self.layer_pattern_specs)
        factors.append(self.exclude_layers)
        if self.online_quant:
            factors.append(self.online_layer_pattern_specs)
            factors.append(self.online_global_spec)
            factors.append(self.online_exclude_layers)
        hash_value = hashlib.sha256(str(factors).encode()).hexdigest()
        return hash_value

    # 返回quant_method。对齐vLLM QuantizationConfig.get_name()的接口，让从vLLM移植来的模型代码
    #   能原样写 quant_config.get_name() == "quark"（如 sglang插件的 deepseek_mla_forward.py、llama.py 判断gguf）
    def get_name(self):
        """Returns the quantization method name."""
        # 【译】返回量化方法的名字。
        return self.quant_method

    # -- internal helpers ---------------------------------------------------
    # 【译】-- 内部辅助函数 --

    # layer_name是否在排除列表里（排除 = 不量化，保持torch_dtype）
    # 调用方：get_layer_quant_config 第1步；_validate_packed_layer_specs
    # exclude_layers默认是checkpoint表的排除列表，查在线表时由调用方传入online_exclude_layers
    def _is_excluded(
        self,
        layer_name: str,
        exclude_layers: list[str] | None = None,
        *,
        check_children: bool = False,
    ) -> bool:
        if exclude_layers is None:
            exclude_layers = self.exclude_layers
        if layer_name is None or not exclude_layers:
            return False
        prefix = layer_name + "."
        for ignore_str in exclude_layers:
            if self._matches_exclude(layer_name, ignore_str):
                return True
            # When check_children is True, also match if any exclude entry
            # is a child of layer_name.  This is needed by container modules
            # like FusedMoE whose prefix (e.g. "mtp.layers.60.mlp.experts")
            # is a parent of the leaf-level exclude entries (e.g.
            # "mtp.layers.60.mlp.experts.0.gate_up_proj").
            # 【译】check_children为True时，只要某个排除项是layer_name的子模块，也算匹配。
            #   FusedMoE这类容器模块需要它：它的前缀（例如 "mtp.layers.60.mlp.experts"）
            #   是叶子级排除项（例如 "mtp.layers.60.mlp.experts.0.gate_up_proj"）的父级。
            if check_children and ignore_str.startswith(prefix):
                return True
        return False

    # 单条排除项的匹配规则。之所以支持这么多写法，是因为各量化工具导出的排除项写法各不相同：
    #   有的写完整层名（"model.layers.0.mlp.gate"），有的写模块前缀（"lm_head"），
    #   有的写通配（"*.eh_proj"），compressed-tensors还会写正则（"re:.*mlp.gate$"）
    # check_contains：反方向的子串匹配（层名包含在排除项里，忽略大小写）。
    #   本类自己不传True；vLLM插件 kimi_k25.py 直接调用这个静态方法时用它
    @staticmethod
    def _matches_exclude(
        layer_name: str, ignore_str: str, check_contains: bool = False
    ) -> bool:
        """Match the target string or regular expression.

        Supports exact match, prefix match (layer under an excluded module),
        fnmatch glob patterns (``*`` / ``?``), and ``re:`` regex patterns.
        """
        # 【译】匹配目标字符串或正则表达式。
        #
        #   支持精确匹配、前缀匹配（位于被排除模块之下的层）、fnmatch通配模式（* / ?）以及 re: 开头的正则模式。
        if ignore_str.startswith("re:"):
            pattern = ignore_str[3:]
            if re.search(pattern, layer_name):
                return True
        elif "*" in ignore_str or "?" in ignore_str:
            # Glob pattern: match exact or as prefix of deeper sub-modules
            # 【译】通配模式：精确匹配，或者作为更深层子模块的前缀匹配
            if fnmatch.fnmatch(layer_name, ignore_str):
                return True
            if fnmatch.fnmatch(layer_name, ignore_str + ".*"):
                return True
        elif check_contains:
            return layer_name.lower() in ignore_str.lower()
        else:
            # Exact match or prefix match (e.g. "lm_head" excludes "lm_head.weight")
            # 【译】精确匹配或前缀匹配（例如 "lm_head" 会排除 "lm_head.weight"）
            if layer_name == ignore_str or layer_name.startswith(ignore_str + "."):
                return True
        return False

    # 把排除项里的HF层名片段按字符串替换成ATOM的模块路径（只改排除列表，不改逐层规格）
    # 为什么需要：有些模型的ATOM模块树和HF不同（例如多模态模型把语言模型挂在另一个路径下），
    #   checkpoint写的排除项用HF路径，不替换就匹配不到，本该保持bf16的层会被当成量化层去装权重
    # 例：Kimi-K2.5 / MiniMax-M3 的VL checkpoint按纯文本方式加载时，映射为
    #   {"language_model.model.": "model.", "language_model.lm_head": "lm_head"}，
    #   排除项"language_model.lm_head" -> "lm_head"，才能命中ATOM里的lm_head层
    # 谁调用：remap_layer_name 的最后一步（模型类声明了 quant_exclude_name_mapping 时）；
    #   vLLM插件 model_wrapper.py 在插件模式下也直接调用
    def apply_exclude_name_mapping(self, mapping: dict[str, str]):
        if not mapping or not self.exclude_layers:
            return
        new_excludes = []
        for name in self.exclude_layers:
            for old, new in mapping.items():
                name = name.replace(old, new)
            new_excludes.append(name)
        # dict.fromkeys(列表)：利用dict保持插入顺序且key唯一，实现"保序去重"
        self.exclude_layers = list(dict.fromkeys(new_excludes))

    # 追加"这个模型无论checkpoint怎么写都不该量化"的层（已在列表里的不重复添加）
    # 为什么需要：有的checkpoint漏写了排除项，但ATOM的实现要求某些层保持bf16，
    #   例如MTP层的eh_proj（拼接embedding与hidden后的投影）
    # 谁调用：deepseek_mtp.py、mimo_v2_mtp.py 追加 "*.eh_proj"；
    #   vLLM插件 model_wrapper.py、SGLang插件 prepare.py / runtime/model_arch.py 追加模型类声明的默认排除项
    def apply_default_exclude_layers(self, excludes: list[str]):
        if not excludes:
            return
        for exclude in excludes:
            if exclude not in self.exclude_layers:
                self.exclude_layers.append(exclude)

    # 把量化表里的层名从"checkpoint的命名"改写成"ATOM融合后的命名"
    # 背景：ATOM会把多个投影融合成一个参数（gate_proj+up_proj -> gate_up_proj），
    #   而checkpoint的quantization_config按融合前的名字写排除项/逐层规格。不改写的话，
    #   融合层用自己的名字（gate_up_proj）去查表永远查不到，会错误地落到全局规格
    # 例：排除项"model.layers.0.mlp.gate_proj" -> "model.layers.0.mlp.gate_up_proj"
    # 调用方：ModelRunner.__init__ 在构造模型之前（model_runner.py，必须在查表之前完成），
    #   以及插件的准备代码（plugin/prepare.py、sglang/runtime/model_arch.py、sglang各模型适配等）
    # 参数：
    #   packed_modules_mapping：模型类的类属性，声明融合关系，
    #     值为 (融合名, 分片序号) 表示"分量名 -> 融合名"；值为列表表示"融合名 -> 拆成各分量名"
    #   weights_mapper：插件模式下vLLM的 WeightsMapper（有 _map_name 方法），先把排除项按它的规则改名
    #   quant_exclude_name_mapping：模型类的类属性，见 apply_exclude_name_mapping
    # 副作用：把最终用的映射存到 self.packed_modules_mapping，SGLang插件的qwen3_5适配会改写它
    def remap_layer_name(
        self,
        hf_config: PretrainedConfig,
        packed_modules_mapping: dict | None = None,
        weights_mapper: dict | None = None,
        quant_exclude_name_mapping: dict[str, str] | None = None,
    ):
        model_type = hf_config.model_type
        weights_mapper = weights_mapper or {}
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        # for special models
        # 【译】针对特殊模型
        # DeepSeek系MLA模型：有q_lora_rank时，q_a_proj和kv_a_proj_with_mqa融合成fused_qkv_a_proj（第7批）
        # 这些模型的映射写死在这里、覆盖模型类传进来的映射，因为是否融合q_a_proj取决于hf_config
        #   （V2-Lite这类没有q_lora_rank的模型没有q_a_proj），类属性表达不了这种条件
        if model_type in ("deepseek_mtp", "deepseek_v3", "kimi_k2", "glm_moe_dsa"):
            if hasattr(hf_config, "q_lora_rank") and hf_config.q_lora_rank is not None:
                self.packed_modules_mapping = {
                    "q_a_proj": ("fused_qkv_a_proj", 0),
                    "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                    "gate_proj": ("gate_up_proj", 0),
                    "up_proj": ("gate_up_proj", 1),
                }
            else:
                self.packed_modules_mapping = {
                    "gate_proj": ("gate_up_proj", 0),
                    "up_proj": ("gate_up_proj", 1),
                }
        elif model_type in ("qwen3_moe", "qwen3_next") and getattr(
            hf_config, "mlp_only_layers", []
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        # 解释：上面这个是反方向的映射（融合名 -> 分量名），把表里写作 gate_up_proj 的条目拆成
        #   gate_proj、up_proj 两条。Qwen3MoeForCausalLM.__init__ 里有同样一句（"Only perform the following
        #   mapping when Qwen3MoeMLP exists"，即有 mlp_only_layers 这种dense MLP层时）；
        #   这里要提前补上，因为remap发生在模型构造之前，那时模型类的__init__还没执行

        if weights_mapper:
            self.exclude_layers = [
                weights_mapper._map_name(name) for name in self.exclude_layers
            ]

        # remap
        # 【译】重映射
        # 一个名字可能映射成多个名字（融合层拆回各个分量时），所以返回列表
        def _remap_layer_name(name: str) -> list[str]:
            for packed_key, packed_value in self.packed_modules_mapping.items():
                # for self_attn.up_proj and self_attn.gate_up_proj
                # up_proj in gate_up_proj, so add prefix .
                # 【译】为了区分 self_attn.up_proj 和 self_attn.gate_up_proj：
                #   up_proj 是 gate_up_proj 的子串，所以加上前缀"."
                match_key = (
                    packed_key if packed_key.startswith(".") else f".{packed_key}"
                )
                if match_key in name:
                    if isinstance(packed_value, list):
                        # "gate_up_proj" → ["gate_proj", "up_proj"]
                        # 【译】"gate_up_proj" → ["gate_proj", "up_proj"]（融合名拆成各分量名）
                        return [
                            name.replace(packed_key, part, 1) for part in packed_value
                        ]
                    else:
                        # "gate_proj" → ("gate_up_proj", 0)
                        # 【译】"gate_proj" → ("gate_up_proj", 0)（分量名改成融合名，0是它在融合参数里的分片序号）
                        packed_remap_part, _ = packed_value
                        return [name.replace(packed_key, packed_remap_part, 1)]
            return [name]

        # 对checkpoint表的逐层规格和排除项做同样的改名；排除项改名后去重
        #   （gate_proj、up_proj都被排除时，会得到两条相同的gate_up_proj）
        new_pattern_specs = []
        for pattern, spec in self.layer_pattern_specs:
            for remapped in _remap_layer_name(pattern):
                new_pattern_specs.append((remapped, spec))
        self.layer_pattern_specs = new_pattern_specs

        new_exclude = []
        for name in self.exclude_layers:
            new_exclude.extend(_remap_layer_name(name))
        self.exclude_layers = list(dict.fromkeys(new_exclude))

        # Both lists have been through the same remap here, which is what makes
        # them comparable; `quant_exclude_name_mapping` below rewrites only the
        # exclude side.
        # 【译】此时两个列表都经过了同样的重映射，所以可以相互比较；下面的 quant_exclude_name_mapping
        #   只改写排除列表这一边。
        if self.quant_method == "modelopt":
            self._validate_packed_layer_specs()

        # 在线表用户可能也按HF名字写（例如 layer_quant_config 的key写 "*gate_proj*"），同样改名
        if self.online_quant:
            new_online_pattern_specs = []
            for pattern, spec in self.online_layer_pattern_specs:
                for remapped in _remap_layer_name(pattern):
                    new_online_pattern_specs.append((remapped, spec))
            self.online_layer_pattern_specs = new_online_pattern_specs

            new_online_exclude = []
            for name in self.online_exclude_layers:
                new_online_exclude.extend(_remap_layer_name(name))
            self.online_exclude_layers = list(dict.fromkeys(new_online_exclude))

        # Apply model-declared HF-name to ATOM-path translations for exclude entries.
        # Models that have a mismatch between their HF quant config names and ATOM
        # module paths declare `quant_exclude_name_mapping` as a class attribute.
        # 【译】对排除项应用模型声明的"HF名 -> ATOM路径"转换。
        #   HF量化config中的名字与ATOM模块路径不一致的模型，会把 quant_exclude_name_mapping 声明为类属性。
        if quant_exclude_name_mapping:
            self.apply_exclude_name_mapping(quant_exclude_name_mapping)

    # 只由 remap_layer_name 在 quant_method=="modelopt" 时调用：
    #   检查"融合进同一个ATOM参数的几个投影"量化规格是否一致，不一致就在启动时报错
    # 为什么只查ModelOpt：它按每个源投影逐条写规格（q/k/v各一条），最容易出现兄弟投影格式不同
    #   （例如q_proj是NVFP4、k_proj是FP8）；而ATOM一个融合参数只能有一种格式
    # 不查的后果：remap后出现两条同名但规格不同的模式，get_layer_quant_config取第一条，
    #   另一种格式的权重被硬装进去，不报错但结果是错的
    def _validate_packed_layer_specs(self):
        """Reject fused parameters whose source projections disagree.

        ModelOpt records one entry per source projection, while ATOM builds one
        packed parameter per fused name. Siblings that disagree collapse into
        duplicate patterns during the remap above, and `get_layer_quant_config`
        returns whichever comes first -- so a packed (N, K/2) uint8 NVFP4 weight
        can be loaded into an (N, K) fp8 parameter with nothing said.

        A projection absent from the config is deliberately not an error: a
        model's packed mapping lists every projection that *may* fuse, and real
        exports legitimately omit the ones a given layer does not have (M3 has
        `index_q_proj`/`index_k_proj` only on its indexer layers).
        """
        # 【译】拒绝那些源投影之间规格不一致的融合参数。
        #
        #   ModelOpt为每个源投影记录一个条目，而ATOM为每个融合名构建一个打包参数。规格不一致的兄弟投影
        #   在上面的重映射中会合并成重复的模式，而 get_layer_quant_config 返回最先出现的那个——
        #   于是一个打包的 (N, K/2) uint8 NVFP4权重可能被悄无声息地装进一个 (N, K) 的fp8参数里。
        #
        #   config中缺少某个投影则有意不算错误：模型的融合映射列出的是所有"可能"融合的投影，
        #   而真实导出的checkpoint会合理地省略某一层并不具备的那些（M3只有索引器层才有 index_q_proj/index_k_proj）。
        seen: dict[str, LayerQuantConfig] = {}
        for pattern, spec in self.layer_pattern_specs:
            previous = seen.setdefault(pattern, spec)
            if previous != spec:
                raise ValueError(
                    f"Conflicting quantization specs for packed layer {pattern!r}: "
                    f"{previous} vs {spec}. Source projections fused into one ATOM "
                    "parameter must share one spec."
                )
        for pattern in seen:
            # FusedMoE resolves its experts container with check_children=True,
            # so an exclude entry naming one expert de-quantizes all of them.
            # 【译】FusedMoE用 check_children=True 解析它的专家容器，所以一个点名某个专家的排除项
            #   会让全部专家都反量化（不量化）。
            if self._is_excluded(pattern, self.exclude_layers, check_children=True):
                raise ValueError(
                    f"Packed layer {pattern!r} is quantized by at least one source "
                    "projection and excluded by another. Exclude all of them or "
                    "none: a partial exclusion de-quantizes the whole fused layer "
                    "to bf16 while the checkpoint still holds packed weights."
                )


# =====================================================================================
# 【科普】什么是 k-pool？—— 从你熟悉的 MHA / GQA 出发，四步讲清楚
#   （GLM-5.3-Flash 的稀疏注意力索引器；模型实现见第8批 models/glm5_next.py，
#    kernel 见 model_ops/glm5_next/kpool.py、indexer.py，数字取自 recipes/GLM-5.3-Flash.md）
# =====================================================================================
#
# ① 出发点：MHA / GQA 的注意力要看"全部"历史 token
#   decode 时，新 token 的 q 要和 KV cache 里每一个历史 token 的 k 做点积，再对 v 加权求和：
#
#       q_新 ──点积──> k_0  k_1  k_2  ...  k_{L-1}      （L 个，全都要算）
#                       │    │    │          │
#                    softmax 后加权求和 v_0 ... v_{L-1}
#
#   GQA 只是让多个 q 头共用一组 K/V 头，省的是 KV cache 的显存，"每个 q 要看 L 个 token"这件事没变。
#   上下文 L = 100万 时，每生成一个 token、每层都要读 100万 个 k/v —— 这是长上下文的主要开销。
#
# ② 稀疏注意力（DSA）：先"粗筛"，只对最相关的 2048 个 token 做真正的注意力
#   想法：大多数历史 token 对当前 token 几乎没贡献。如果能便宜地挑出最相关的 top-k 个（这里 k = index_topk = 2048），
#   主注意力就只在这 2048 个 token 上算，代价从"随 L 增长"变成固定的 2048。
#   挑选靠一个额外的小模块：索引器（indexer）。它本身就是一个"迷你注意力"：
#     - 32 个索引 q 头，每头 128 维；但 k 只有 1 个头、128 维 —— 正是 GQA 的极端情况 MQA（所有 q 头共用一份 k）。
#       所以索引器也要缓存一份"索引 key"：每个 token 只存 128 维（FP8），比主 KV 小得多
#     - 打分：新 token 的 32 个索引 q 头分别和每个历史 token 的索引 key 点积，按每头权重加起来，得到一个分数
#     - 取分数最高的 2048 个 token 的下标，交给主注意力（MLA）只算这些位置
#
#       新token ──> 索引器：给 L 个历史 token 各打一个分（便宜：1个k头、128维、FP8）
#                     │
#                     └─> top-2048 的下标 ──> 主注意力（MLA）只读这 2048 个 token 的 KV（贵的部分被限住了）
#
#   剩下的问题：索引器自己仍然要给"全部 L 个"token 打分、缓存"全部 L 个"索引 key —— 虽然便宜，仍随 L 线性增长。
#
# ③ k-pool：把"每个 token 一个索引 key"压成"每 index_kpool 个 token 一个"
#   GLM-5.3 的 index_kpool = 4：把连续 4 个 token 的索引 key 合并（pool，池化）成 1 个，索引器只给这些"池"打分。
#
#       token:      0   1   2   3 | 4   5   6   7 | 8   9  10  11 | 12  13
#       索引key:    k0  k1  k2  k3| k4  k5  k6  k7| k8  k9  k10 k11| k12 k13
#                   └────┬────┘   └────┬────┘   └────┬─────┘   └─┬─┘
#       池化后:         P0            P1             P2         尾巴（还没凑满4个，不池化）
#
#   合并方式不是简单平均，而是"按维度的加权平均"：对 128 维里的每一维，4 个 token 各有一个打分
#     （由一个可学习的门 gate 投影 + 每个槽位一个可学习偏置 ape 算出），在这 4 个 token 之间做 softmax 得到权重，
#     再加权求和。于是每一维都能自己决定"更听这 4 个 token 里哪一个的"。
#   打分与选择也在"池"的粒度上进行：
#     - 要选 2048 个 token，就选 2048 / 4 = 512 个池，再把每个池展开回它覆盖的 4 个 token
#     - 尾巴（最新的、还没凑满 4 个的 0~3 个 token）总是被选上（always_select_tail），所以最新的 token 永远不会被漏掉；
#       尾巴的原始 key 暂存在每个请求自己的小缓存里（tail cache），凑满 4 个时才池化写入索引缓存
#   效果：索引缓存缩小到 1/4，索引器的打分量也缩小到 1/4。
#   一个重要性质：序列不超过 2048 个 token 时，512 个池足以覆盖全部历史 → 每个 token 都被选中 →
#     结果和不做稀疏的普通（dense）注意力完全一样。只有超过 2048 时，池化选择才真正决定"看哪些 token"。
#   （池化后的 key 还会做一次 Hadamard-128 旋转再量化成 FP8，只是为了让 FP8 量化更准，不改变点积结果。）
#
# ④ 这和 KV block 大小有什么关系？—— 就是下面这个函数要解决的
#   索引 key 和主 KV 共用同一张 block_table（见文件开头【科普】分页 KV）：一个 KV block 装 B 个 token 的主 KV，
#   同时也装这 B 个 token 对应的索引 key。池化后，这 B 个 token 只占 B / 4 行索引。
#   索引打分 kernel（deepgemm_fp8_paged_mqa_logits）按"每 16 行一组"的预重排（preshuffled）布局读每个 block，
#   所以每个 block 的索引行数必须是 16 的倍数（MQA_LOGITS_PRESHUFFLE_ROWS = 16）：
#
#       B = 16（默认）:  主KV 16个token │ 索引 16/4 = 4 行   ✗ 不够16行 → 只能退回"每个token一行"，k-pool省的全浪费了
#       B = 64        :  主KV 64个token │ 索引 64/4 = 16 行  ✓ 正好一组
#       B = 128       :  主KV 128个token│ 索引 32 行         ✓ 也行，但 block 越大：最后一个 block 平均浪费越多、
#                                                             前缀缓存复用的粒度越粗
#
#   所以取满足条件的最小值 B = index_kpool × 16 = 64。
# =====================================================================================
#
# 调用方：只有 Config.__post_init__：检测到Glm5Next架构且index_kpool>1时，用返回值覆盖 kv_cache_block_size
#   （用户的 --block-size 被改写），之后 BlockManager、slot_mapping、注意力builder 都按这个block大小工作
# 例：index_kpool=4，MQA_LOGITS_PRESHUFFLE_ROWS=16 -> block_size=64，每个block正好16行索引
def glm5_kpool_block_size(index_kpool: int) -> int:
    """Tokens per KV block that lets GLM-5.3's pooled index cache be exact.

    A block of B tokens needs ``B // index_kpool`` index rows, and that count
    must be a multiple of `MQA_LOGITS_PRESHUFFLE_ROWS`. The smallest B that
    satisfies it is the product, and the smallest is what we want: a larger
    block only adds paging waste, while a smaller one forces the cache to be
    padded back up to one row per token -- which is the whole cost being
    removed here.
    """
    # 【译】让GLM-5.3的池化索引缓存恰好对齐的每个KV block的token数。
    #
    #   一个B个token的block需要 B // index_kpool 行索引，这个行数必须是 MQA_LOGITS_PRESHUFFLE_ROWS 的倍数。
    #   满足条件的最小B就是两者的乘积，而我们要的正是最小值：更大的block只会增加分页浪费，
    #   更小的block则迫使缓存被填充回每个token一行——而那正是这里要消除的全部开销。
    return index_kpool * MQA_LOGITS_PRESHUFFLE_ROWS


# 【分支】GLM-5.3-Flash还不支持的并行方式（PCP/DCP/TBO），见第8、9批
# 作用：列出当前配置里开启了、但GLM-5.3还没适配的并行方式。这三种方式都会把一条序列的token/KV
#   或一个batch切到不同rank / micro-batch上处理，而上面说的k-pool池化索引状态还没有按这种切分维护，
#   硬跑会得到错误的索引结果而不是报错
#   例：PCP把一条prompt的token切给两张卡，第4~7个token可能被拆到两边，"凑满4个才池化"和"尾巴缓存"
#   这两件事都要求同一个池的4个token在同一处处理
# 调用方：只有 Config.__post_init__：返回非空就抛 ValueError，在启动时拒绝，而不是跑出错误答案
def _glm5_next_unsupported_features(config: "Config") -> list[str]:
    """Return parallel modes that do not yet preserve GLM-5.3 k-pool state."""
    # 【译】返回尚不能正确保持GLM-5.3 k-pool状态的并行模式。
    unsupported = []
    if config.prefill_context_parallel_size > 1:
        unsupported.append("PCP")
    if config.decode_context_parallel_size > 1:
        unsupported.append("DCP")
    if config.enable_tbo or config.enable_tbo_decode:
        unsupported.append("TBO")
    return unsupported


# ---------------- 以下几张表供get_hf_config使用：决定一个model_type用哪种方式加载HF config ----------------
# 背景：transformers靠config.json里的model_type找到对应的Config类；有些新模型transformers不认识，
#   或者ATOM想复用另一个模型的Config类，就需要这几张表来"改道"
# 这几张表都是模块私有的，只有本文件的 get_hf_config 读；支持一个新模型时，如果它的config.json
#   transformers加载不了，通常就是往这里加一行（见 /add-model）

# model_type -> 实际用来解析它的transformers config类型
# 例：DeepSeek-V3.2、V4、GLM-5、Kimi-K2都借用deepseek_v3的Config类（结构相近，额外字段在get_hf_config里补回）
# 用在get_hf_config的路径2（多模态模型的text_config，如Kimi-K2.5的text_config是kimi_k2）和路径4
# 副作用：借用后 hf_config.model_type 变成被借用者（"deepseek_v3"），所以下游判断具体模型时
#   要看 architectures（见 Config.__post_init__ 对V4的检测），而 QuantizationConfig.remap_layer_name
#   用 model_type 判断"DeepSeek系"恰好能把这几个借用者都覆盖进来
_CONFIG_REGISTRY: dict[str, str] = {
    "deepseek_v32": "deepseek_v3",
    # 【译】（行尾注释及其续行）V4复用V3的schema；V4特有的字段（compress_ratios、num_hash_layers、
    #   hc_mult、swiglu_limit……）作为额外的config属性透传，在 DeepseekV4Args.from_hf_config 中读取。
    "deepseek_v4": "deepseek_v3",  # V4 reuses V3 schema; V4-specific fields
    # (compress_ratios, num_hash_layers, hc_mult, swiglu_limit, ...) flow
    # through as extra config attrs and are read in DeepseekV4Args.from_hf_config.
    # 【译】（行尾注释）GLM 5.0 MoE，结构与DeepSeek v3.2相似
    "glm_moe_dsa": "deepseek_v3",  # GLM 5.0 MoE, structure similar to DeepSeek v3.2
    "kimi_k2": "deepseek_v3",
    "qwen3_next": "qwen3_next",
}


# model_types that exist only as speculative-draft checkpoints. transformers
# has no config class for them, so AutoConfig.from_pretrained raises; load them
# as a bare PretrainedConfig instead (see get_hf_config).
# 【译】只作为投机解码草稿checkpoint存在的model_type。transformers没有对应的config类，
#   所以 AutoConfig.from_pretrained 会抛异常；改为把它们加载成一个裸的 PretrainedConfig（见 get_hf_config）。
# frozenset：不可变的集合，适合做模块级常量（不会被意外修改）
# k3_dspark：Kimi-K3的DSpark草稿模型。它的config由 SpeculativeConfig.__post_init__ 通过 get_hf_config 读取，
#   之后 _normalize_draft_dspark_config 补齐字段，草稿模型类直接getattr读取
_PLAIN_CONFIG_MODEL_TYPES: frozenset[str] = frozenset({"k3_dspark"})

# 多模态模型：ATOM（目前只跑文本部分）取出config.json里的text_config子配置当作hf_config
# 为什么：引擎里的通用代码（KV cache大小计算、ModelRunner、调度器）都直接读 hf_config.num_hidden_layers、
#   num_key_value_heads 等文本模型字段；VL模型的根config里这些字段在text_config下面，不抽出来就读不到
# 不在表里的多模态模型（如MiniMax-M3 VL）保留根config，由各自的归一化函数把text_config字段补到根上
_MULTIMODAL_MODEL_TYPES: dict[str, str] = {
    # Maps multimodal model_type -> key in config_dict for the text sub-config
    # 【译】多模态model_type -> 文本子配置在config_dict中的key
    "kimi_k3": "text_config",
    "kimi_k25": "text_config",
    "qwen3_5": "text_config",
    "qwen3_5_moe": "text_config",
    "mistral3": "text_config",
    # 【译】（行尾注释）GLM-5.3-Flash：纯文本的KDA/DSA混合运行时
    "glm5_next": "text_config",  # GLM-5.3-Flash: text-only hybrid KDA/DSA runtime
    "qwen4_exp": "text_config",
}

# Text sub-config model_types that this image's transformers has no class for.
# Loaded as a bare PretrainedConfig; the ATOM model normalizes the aliases it
# needs at construction time.
# 【译】当前镜像里的transformers没有对应类的文本子配置model_type。
#   加载成裸的 PretrainedConfig；ATOM模型在构造时自行归一化它需要的别名字段。
_PLAIN_TEXT_CONFIG_MODEL_TYPES: frozenset[str] = frozenset(
    {"kimi_linear", "glm5_next_text", "qwen4_exp_text"}
)

# multimodal models fully supported by plugin mode
# 【译】插件模式完整支持的多模态模型
# vLLM插件模式下，这几个模型不抽取text_config（vLLM自己处理多模态），见get_hf_config开头
# 原因：插件模式下视觉编码器、图像预处理都由vLLM负责，ATOM的插件模型类（plugin/vllm/models/qwen3_5.py、
#   kimi_k3.py 等）需要完整的多模态config；ATOM原生模式不跑视觉部分，所以才需要抽文本子配置
_PLUGIN_SUPPORTED_MULTIMODAL_MODELS: set[str] = {
    "kimi_k25",
    "kimi_k3",
    "qwen3_5",
    "qwen3_5_moe",
}


# 读取模型目录（或HF仓库名）下的config.json，返回transformers的PretrainedConfig对象
# 调用方：Config.__post_init__（结果存为 Config.hf_config，几乎所有组件都从这里读模型结构）、
#   SpeculativeConfig.__post_init__（草稿模型的config，存为 draft_model_hf_config）、
#   vLLM插件的deepseek_v41_bridge
# 返回值的约定：无论走哪条路径，返回的都是"文本模型"的config，并且
#   - architectures 保持checkpoint原名（ModelRunner按 architectures[0] 在 support_model_arch_dict 选模型类）
#   - quantization_config 在返回对象上（QuantizationConfig 从这里读）
#   - 多模态模型额外带 _multimodal_config（完整config，Config.__post_init__ 取出放到 Config.multimodal_config）
# 对比nano-vllm：nano-vllm只有一行AutoConfig.from_pretrained(model)；ATOM要处理transformers不认识的模型、
#   多模态模型只取文本部分、借用别的模型的Config类等情况，分5条路径：
#   config_dict = 读出原始的config.json字典，按model_type分流：
#     1. deepseek_v41               交给V4.1自己的normalize_hf_config（第8批）
#     2. 多模态（_MULTIMODAL_MODEL_TYPES） 抽出text_config子字典，构造文本config，并补齐几个顶层字段
#     3. 纯草稿模型（k3_dspark）    裸PretrainedConfig，字段原样保留
#     4. 借用（_CONFIG_REGISTRY）   用被借用模型的Config类加载，再把被丢掉的字段补回来
#     5. 其余                      AutoConfig.from_pretrained；transformers不认识时，vLLM插件模式再试vLLM的加载器
def get_hf_config(model: str, trust_remote_code: bool = False) -> PretrainedConfig:
    # get_config_dict：只把config.json读成普通dict（本地目录或从HF下载），不构造任何Config类
    config_dict, _ = PretrainedConfig.get_config_dict(
        model,
    )
    model_type = config_dict.get("model_type")
    # 路径1：DeepSeek-V4.1，返回model_type为deepseek_v41_text的扁平文本config（第8批）
    if model_type == "deepseek_v41":
        from atom.models.deepseek_v41.config import normalize_hf_config

        return normalize_hf_config(config_dict)

    # 访问需要授权的HF仓库时用的token：环境变量HF_TOKEN非空才返回
    def _get_hf_token() -> str | None:
        token = os.getenv("HF_TOKEN")
        if token and token.strip():
            return token
        return None

    multimodal_model_types = _MULTIMODAL_MODEL_TYPES
    # vLLM插件模式：vLLM能完整处理的多模态模型不抽取text_config，从表里去掉
    if is_vllm():
        # Avoid mutating module-level state
        # 【译】避免修改模块级的状态
        # 解释：用字典推导式生成一个新字典，而不是在原表上删除元素
        multimodal_model_types = {
            name: text_key
            for name, text_key in _MULTIMODAL_MODEL_TYPES.items()
            if name not in _PLUGIN_SUPPORTED_MULTIMODAL_MODELS
        }
    # For multimodal models, extract the text sub-config so the rest of ATOM
    # (which is text-only today) works transparently.
    # 【译】对多模态模型，抽出文本子配置，让ATOM的其余部分（目前只处理文本）无感知地工作。
    # 路径2：多模态模型
    # 例：Qwen3.5的config.json = {"model_type": "qwen3_5", "text_config": {...}, "vision_config": {...},
    #   "quantization_config": {...}, "architectures": [...]}，hf_config由text_config构造
    if model_type in multimodal_model_types:
        text_config_key = multimodal_model_types[model_type]
        # .copy()：浅拷贝，下面的pop/赋值不会改动原始config_dict
        text_config_dict = config_dict.get(text_config_key, {}).copy()
        # Remove auto_map to avoid trust_remote_code issues
        # 【译】删除auto_map，避免 trust_remote_code 的问题
        # 解释：auto_map指向模型仓库里的自定义Python类，保留它会要求执行远程代码
        text_config_dict.pop("auto_map", None)
        # Propagate quantization_config from root level into text config
        # (quantization_config lives alongside text_config, not inside it).
        # 【译】把根级的 quantization_config 传播到文本config里
        #   （quantization_config 与 text_config 并列，而不在它里面）。
        if (
            "quantization_config" not in text_config_dict
            and "quantization_config" in config_dict
        ):
            text_config_dict["quantization_config"] = config_dict["quantization_config"]
        # 文本子配置自己的model_type（例如qwen3_5_text）；缺省时按deepseek_v3处理
        text_model_type = text_config_dict.get("model_type", "deepseek_v3")
        if text_model_type in _PLAIN_TEXT_CONFIG_MODEL_TYPES:
            # Transformers does not ship a config class for these in this image
            # (KimiLinearConfig, Glm5NextTextConfig). Keep the fields as plain
            # PretrainedConfig attrs; the ATOM model normalizes the aliases it
            # needs at construction time.
            # 【译】当前镜像里的transformers没有为它们提供config类（KimiLinearConfig、Glm5NextTextConfig）。
            #   把字段作为普通的 PretrainedConfig 属性保留；ATOM模型在构造时自行归一化它需要的别名字段。
            hf_config = PretrainedConfig.from_dict(text_config_dict)
        else:
            # AutoConfig.for_model(model_type)：按model_type找到transformers注册的Config类（不读文件）
            mapped_type = _CONFIG_REGISTRY.get(text_model_type, text_model_type)
            config_class = AutoConfig.for_model(mapped_type)
            hf_config = config_class.from_dict(text_config_dict)
        # Override architectures so that ATOM selects the correct model class
        # which can handle the multimodal weight prefix during loading.
        # 【译】覆盖 architectures，让ATOM选中正确的模型类：它能在加载时处理多模态权重的前缀。
        # 解释：ATOM按architectures[0]在model_runner.py的support_model_arch_dict里查模型类（第6批）；多模态checkpoint的权重名
        #   带"language_model."之类的前缀，要用多模态架构名对应的模型类才能正确映射
        original_arch = config_dict.get("architectures", [])
        if original_arch:
            hf_config.architectures = original_arch
        # Propagate top-level token IDs if missing in text config
        # 【译】文本config里缺少token ID时，从顶层传播过来
        # 注意：这里的循环变量field在函数内遮蔽了文件开头import的dataclasses.field，只在本函数内有效，无害
        for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(hf_config, field, None) is None and field in config_dict:
                setattr(hf_config, field, config_dict[field])
        # 让后续代码统一用 getattr(hf_config, "text_config", hf_config) 的写法：没有就指向自己
        if not hasattr(hf_config, "text_config"):
            hf_config.text_config = hf_config
        # Store full multimodal config (with vision_config) for vision encoder init
        # 【译】保存完整的多模态config（含vision_config），供视觉编码器初始化使用
        # Config.__post_init__把它取出来放到Config.multimodal_config
        try:
            full_config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code
            )
            hf_config._multimodal_config = full_config
        # 【译】（行尾注释）noqa: BLE001 - transformers在这里可能抛出任何异常
        except Exception:  # noqa: BLE001 - transformers raises anything here
            # Consumers use None to report that their full multimodal config
            # could not be loaded. A partial generic object can bypass those
            # checks and run with default token IDs. GLM-5.3 is text-only here,
            # so it has no reason to weaken that contract.
            # 【译】使用方以None来报告"完整的多模态config加载失败"。一个不完整的通用对象可能绕过这些检查，
            #   并以默认的token ID运行。GLM-5.3在这里是纯文本的，所以没有理由削弱这个约定。
            hf_config._multimodal_config = None
        return hf_config

    # 路径3：只作为草稿模型存在的model_type
    if model_type in _PLAIN_CONFIG_MODEL_TYPES:
        # Speculative-draft checkpoints ship their own model_type that
        # transformers has never heard of (and never will -- they are not
        # standalone LMs). There is no schema to map onto, and nothing here
        # needs one: the draft model reads plain attributes off the config.
        # Keep every field verbatim as a bare PretrainedConfig.
        # 【译】投机解码草稿checkpoint带着transformers从没听说过（以后也不会有——它们不是独立的语言模型）
        #   的model_type。没有可以映射的schema，这里也不需要：草稿模型直接从config上读普通属性。
        #   把所有字段原样保留在一个裸的 PretrainedConfig 里。
        return PretrainedConfig.from_dict(config_dict)

    # 路径4：借用另一个模型的Config类
    if model_type in _CONFIG_REGISTRY:
        config_class = AutoConfig.for_model(_CONFIG_REGISTRY[model_type])
        hf_config = config_class.from_pretrained(
            model,
            token=_get_hf_token(),
            trust_remote_code=trust_remote_code,
        )
        # transformers' from_pretrained strips fields that aren't in the target
        # config schema. For mapped types (e.g. deepseek_v4 → deepseek_v3) the
        # source-specific fields would be dropped. Re-inject them so V4-only
        # attrs (compress_ratios, num_hash_layers, hc_mult, swiglu_limit, ...)
        # remain accessible via getattr(hf_config, field) downstream.
        # 【译】transformers的 from_pretrained 会去掉目标config schema里没有的字段。对于映射过的类型
        #   （例如 deepseek_v4 → deepseek_v3），源模型特有的字段就会丢失。把它们重新注入，
        #   让V4独有的属性（compress_ratios、num_hash_layers、hc_mult、swiglu_limit……）
        #   在下游仍能通过 getattr(hf_config, field) 访问。
        # 注意：这样得到的hf_config.model_type是"deepseek_v3"（被借用者的类型），architectures仍是原名，
        #   所以Config.__post_init__判断是不是V4时看architectures而不是model_type
        for field, value in config_dict.items():
            if not hasattr(hf_config, field):
                setattr(hf_config, field, value)
        return hf_config
    # 路径5：普通模型（Qwen3、Llama等），transformers直接认识
    try:
        hf_config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
    except ValueError:
        # For the unsupported model in current transformers, try vllm if in plugin mode
        # 【译】当前transformers不支持的模型，如果处于插件模式，就尝试用vllm加载
        if is_vllm():
            from vllm.transformers_utils.config import get_config
            from vllm.transformers_utils.gguf_utils import (
                maybe_patch_hf_config_from_gguf,
            )

            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            hf_config = maybe_patch_hf_config_from_gguf(model, hf_config)
        else:
            # 裸raise：原样重新抛出当前正在处理的异常
            raise
    return hf_config


# 读取模型目录下的generation_config.json（模型作者推荐的生成参数，ATOM主要用其中的eos_token_id）
# 调用方：Config.__post_init__，结果存为 Config.generation_config，并用其中的eos_token_id填stop_token_ids；
#   LLMEngine 再从 stop_token_ids 里去掉 tokenizer 的 eos，调度器据此判断序列是否结束（第2、3批）
# 文件不存在时返回None（不是所有模型都有这个文件），此时只靠tokenizer的eos停止
# 对比nano-vllm：nano-vllm不读generation_config，eos只取tokenizer.eos_token_id
#   有些模型的generation_config.json里eos_token_id是一个列表（多个结束符），而tokenizer只给出其中一个
def get_generation_config(model: str) -> GenerationConfig:
    try:
        return GenerationConfig.from_pretrained(
            model,
        )
    # 【译】（行尾注释）找不到文件
    except OSError:  # Not found
        return None


# ---------------- 【分支】MiniMax-M3的config归一化，见第8批 ----------------
# 三个函数的关系：Config.__post_init__ -> _normalize_minimax_m3_text_config
#   -> _is_minimax_m3_config（判断是不是M3）+ _normalize_minimax_m3_mlp_layer_types（统一MoE层布局字段）
# 为什么要归一化：MiniMax-M3有纯文本和VL两种导出，各家量化工具导出的config.json写法也不统一
#   （层布局字段名不同、字段放在根上还是text_config里不同）。minimax_m3/model.py 只认一种写法，
#   所以在Config构造时一次性改写hf_config，让模型代码不必处理每种变体


# 根config或其text_config的architectures / model_type里含"minimax_m3"或"minimaxm3"（不区分大小写）就算M3
# 两处都查、两种拼写都认，是因为VL导出的根model_type和文本子配置不同、各导出工具的类名拼写也不同
def _is_minimax_m3_config(hf_config: PretrainedConfig) -> bool:
    text_config = getattr(hf_config, "text_config", None)
    for candidate in (hf_config, text_config):
        if candidate is None:
            continue
        architectures = getattr(candidate, "architectures", None) or ()
        if any(
            "minimax_m3" in str(arch).lower() or "minimaxm3" in str(arch).lower()
            for arch in architectures
        ):
            return True
        model_type = str(getattr(candidate, "model_type", "") or "").lower()
        if "minimax_m3" in model_type or "minimaxm3" in model_type:
            return True
    return False


# mlp_layer_types里的标签 -> moe_layer_freq里的值：dense层记0，sparse（MoE）层记1
_MINIMAX_M3_MLP_LAYER_TYPES = {"dense": 0, "sparse": 1}


# 把"每层是dense MLP还是MoE"统一写成 moe_layer_freq（直接改config对象，无返回值）
# 谁读结果：minimax_m3/model.py 按 moe_layer_freq[layer_id] != 0 决定这一层建 MoE 还是 dense MLP
# 例：mlp_layer_types=["dense", "sparse", "sparse"] -> moe_layer_freq=[0, 1, 1]
#   两个字段都有时必须一致，否则报错；长度必须等于num_hidden_layers
def _normalize_minimax_m3_mlp_layer_types(config: PretrainedConfig) -> None:
    """Express `mlp_layer_types` as the `moe_layer_freq` list the model reads.

    MiniMax-M3 exports carry one of the two fields, and a config without
    `moe_layer_freq` builds every layer as MoE, so a list that cannot be read
    exactly must fail here rather than build the wrong layers.
    """
    # 【译】把 mlp_layer_types 表达成模型读取的 moe_layer_freq 列表。
    #
    #   MiniMax-M3的导出文件带有这两个字段之一，而没有 moe_layer_freq 的config会把每一层都建成MoE，
    #   所以一个无法被精确解读的列表必须在这里失败，而不是建出错误的层。
    layer_types = getattr(config, "mlp_layer_types", None)
    if layer_types is None:
        return
    num_layers = config.num_hidden_layers
    if len(layer_types) != num_layers:
        raise ValueError(
            f"MiniMax-M3 mlp_layer_types has {len(layer_types)} entries, but "
            f"num_hidden_layers is {num_layers}."
        )
    unknown = sorted(
        {repr(label) for label in layer_types}
        - {repr(label) for label in _MINIMAX_M3_MLP_LAYER_TYPES}
    )
    if unknown:
        raise ValueError(
            f"MiniMax-M3 mlp_layer_types has unknown labels {', '.join(unknown)}; "
            "expected 'dense' or 'sparse'."
        )
    moe_layer_freq = [_MINIMAX_M3_MLP_LAYER_TYPES[label] for label in layer_types]
    existing = getattr(config, "moe_layer_freq", None)
    if existing is None:
        config.moe_layer_freq = moe_layer_freq
    elif (
        not isinstance(existing, (list, tuple))
        or [int(freq != 0) for freq in existing] != moe_layer_freq
    ):
        raise ValueError(
            "MiniMax-M3 config sets both mlp_layer_types and moe_layer_freq, and "
            "they disagree on which layers are MoE."
        )


# MiniMax-M3（含VL多模态版本）的config归一化；不是M3时直接返回（所以可以对所有模型无条件调用）
# 调用方：Config.__post_init__（在应用hf_overrides之后，所以 --hf-overrides 写在根上的字段也会被搬进text_config）
# 最后一步"text_config字段补到根上"的原因：M3 VL不在_MULTIMODAL_MODEL_TYPES里，hf_config是VL根config，
#   而引擎通用代码（KV大小计算、ModelRunner等）直接读 hf_config.num_hidden_layers 等字段
# 做的事：根上的层布局字段覆盖到text_config -> 统一MoE层布局
#   -> 补swiglu_beta默认值 -> 根上的索引器字段覆盖到text_config -> text_config的字段补到根上（根上没有的才补）
def _normalize_minimax_m3_text_config(hf_config: PretrainedConfig) -> None:
    if not _is_minimax_m3_config(hf_config):
        return
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and text_config is not hf_config:
        # Some exports (Quark NVFP4) carry the layer layout on the VL root; it
        # replaces the text config's own, then is checked like any other.
        # 【译】有些导出文件（Quark NVFP4）把层布局放在VL的根config上；它会替换文本config自己的布局，
        #   然后像其他情况一样接受检查。
        for attr_name in ("mlp_layer_types", "moe_layer_freq"):
            attr_value = getattr(hf_config, attr_name, None)
            if attr_value is not None:
                setattr(text_config, attr_name, attr_value)
    _normalize_minimax_m3_mlp_layer_types(
        text_config if text_config is not None else hf_config
    )
    if text_config is None or text_config is hf_config:
        return

    # 激活函数是swigluoai但没给swiglu_beta时，补默认值1.0
    if (
        getattr(text_config, "hidden_act", None) == "swigluoai"
        and getattr(text_config, "swiglu_beta", None) is None
    ):
        text_config.swiglu_beta = 1.0

    # 稀疏注意力索引器的几个开关：根上有就覆盖到text_config（--hf-overrides写在根上时由此生效）
    for attr_name in (
        "use_index_cache",
        "index_topk_freq",
        "index_topk_pattern",
        "index_skip_topk_offset",
    ):
        attr_value = getattr(hf_config, attr_name, None)
        if attr_value is not None:
            setattr(text_config, attr_name, attr_value)

    # vars(obj)：返回对象的__dict__（实例属性字典）；把text_config的公开字段补到根config上
    for attr_name, attr_value in vars(text_config).items():
        if attr_name.startswith("_") or getattr(hf_config, attr_name, None) is not None:
            continue
        setattr(hf_config, attr_name, attr_value)


# ParallelConfig：数据并行（DP）的拓扑，以及进程间通信用的地址/端口，挂在Config.parallel_config上
# 来源：arg_utils.py的_get_engine_kwargs用--data-parallel-*参数构造；之后：
#   __post_init__：ATOM_DP_*环境变量（若设置）覆盖对应字段，再做范围校验
#   CoreManager（第2批）：为每个DP rank复制一份Config，填入data_parallel_rank/_local、ZMQ地址、PP地址等
# 名词：DP rank = 一个EngineCore进程（自带调度器和一份完整模型，内部可以再TP）
#   例：-dp 2 -tp 4 共8张卡：2个EngineCore，每个管4个ModelRunner进程
#   多机：2台机器各4个DP rank（2x4），第二台的data_parallel_rank=4（起始全局rank）、data_parallel_size_local=4
# 字段里的"global rank"指全集群的DP编号，"local"指本机内的编号
# 注意：TP/PP/DCP的大小在Config上，这里主要是DP（下面的decode_context_parallel_size字段没有ATOM代码读取）
# 对比nano-vllm：nano-vllm只有单机TP，没有这个类
# 主要读者（按字段分组）：
#   拓扑（size / size_local / rank / rank_local）：CoreManager决定在本机spawn几个EngineCore、各自的rank；
#     DPEngineCoreProc 据此选GPU；moe.py 把 size_local 作为MoRI的 gpu_per_node / local_ep_size
#   ZMQ地址（control_address、pp_*）：CoreManager生成后写进每个rank的Config副本，
#     EngineCore / pp_engine_core.py 在子进程里按地址连接
#   端口（master_ip / master_port / base_port）：master_port给DP之间的gloo组，
#     base_port给ModelRunner的torch.distributed集合点（TCPStore）
#
# 适用场景：ATOM的几种并行方式对比（参数在arg_utils.py；限制条件均在代码里核实过）
#   方式          参数                         总GPU数        切的是什么
#   TP            -tp N（1..8）                N              每层权重（按列/按行/按注意力头）
#   DP            -dp M                        M x tp         请求：M个EngineCore，各有调度器和一份完整模型
#   DP-attention  --enable-dp-attention        dp x tp        注意力按请求切（CoreManager把dp改成dp x tp、tp改成1），
#                                                             MoE仍在全部卡上协同
#   EP            --enable-expert-parallel     不变           MoE专家整个分到各卡（跨dp x tp张卡）
#   PP            -pp S                        S x tp         按层切成S段，每段一个EngineCore
#   DCP           -dcp K（K整除tp）            不变（复用TP卡）一个请求的KV按token切到TP组内
#   PCP           -pcp P                       tp x P         prefill时一条长序列按token切
#
#   方式          什么时候用                                          代价 / 限制
#   TP            模型一张卡放不下；想降低单请求延迟                    每层1~2次all-reduce；Config断言tp<=8（单机内）
#   DP            模型放得进tp张卡，追求总吞吐                          每份副本都要一整份权重显存；PP>1时不能再DP>1（CoreManager断言）
#   DP-attention  MLA/KV头很少的MoE大模型高并发（DeepSeek、Kimi类）；    各DP rank每层MoE要锁步，路由不均时轻载rank空等；
#                 TP下每卡重复存同一份latent KV，改DP后KV容量约x卡数      不能与PP>1、PCP>1同用（LLMEngine/CoreManager报错）
#   EP            专家很多的MoE；通常与DP-attention一起开                每层dispatch+combine两次all-to-all；热点专家靠EPLB均衡
#   PP            跨机部署，或TP的all-reduce太贵                         流水线气泡（可用动态分块缓解，见第9批）；不能与DP>1、DP-attention同用
#   DCP           长上下文decode、KV显存紧张的MLA模型                    每层多一次通信（LSE合并）；投机解码+DCP只支持gfx950
#   PCP           超长prompt想缩短TTFT                                  要多出P倍的卡；不支持DP-attention
#
# 例：8张卡跑DeepSeek类MoE，两种常见起法
#   -tp 8                                          1个EngineCore、8个ModelRunner：权重最省，但8卡各存一份相同的latent KV
#   -tp 8 --enable-dp-attention --enable-expert-parallel
#                                                  CoreManager改写为dp=8、tp=1：8个EngineCore各管1张卡、各自调度请求；
#                                                  按engine_core_mgr.py里的估算，整机KV容量约是前者的8倍
@dataclass
class ParallelConfig:
    # DP的总rank数（全集群），对应-dp
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    # 【译】数据并行组的数量。MoE层将按 张量并行大小 x 数据并行大小 的乘积来切分。
    # 解释：开启专家并行时，MoE专家分布在全部 dp x tp 张卡上（第7批）
    data_parallel_size_local: int | None = None
    """DP ranks this node runs. Defaults to data_parallel_size, i.e. the
    single-node case where every global rank is local. Set it below the global
    size to give a node one slice of a multi-node run; it also reaches MoRI as
    `gpu_per_node` (see model_ops/moe.py), so it must describe real hardware."""
    # 【译】本节点运行的DP rank数。默认等于 data_parallel_size，即所有全局rank都在本地的单节点情况。
    #   把它设得小于全局大小，就让一个节点只承担多节点运行中的一部分；它还会作为 gpu_per_node 传给MoRI
    #   （见 model_ops/moe.py），所以必须如实描述硬件。
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    # 【译】数据并行组中的rank。
    # 解释：命令行传入时是"本节点第一个全局rank"；CoreManager为每个EngineCore复制Config后，改写成该进程自己的全局rank
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    # 【译】数据并行组中的本地rank，只在SPMD模式下设置。
    # 解释：ATOM里由CoreManager为每个DP EngineCore设置（本机内的编号），用于选择本机的GPU
    decode_context_parallel_size: int = 1
    """DCP group size. tp_size must be divisible by dcp_size.
    DCP does not increase world_size; it reuses TP GPUs."""
    # 【译】DCP组的大小。tp_size必须能被dcp_size整除。DCP不增加world_size，它复用TP的GPU。
    # 解释：实际生效的是Config.decode_context_parallel_size；这个字段ATOM代码不读（插件读的是vLLM自己的同名字段）
    pipeline_parallel_rank: int = 0
    """Pipeline stage index of this EngineCore (0 = first stage). Each PP stage
    runs as an independent EngineCore process; this identifies which stage."""
    # 【译】本EngineCore的流水线stage序号（0 = 第一个stage）。每个PP stage作为独立的EngineCore进程运行；
    #   这个字段标识它是哪一个stage。
    # 【分支】以下pp_*地址只在pp>1时由CoreManager填写，见第9批 pp_engine_core.py
    pp_meta_addrs: list = field(default_factory=list)
    """ZMQ endpoints (len == pp_size) where each downstream stage receives the
    scheduled batch from the head. Populated by CoreManager for pp_size > 1."""
    # 【译】ZMQ端点列表（长度 == pp_size），每个下游stage在这里从head接收调度好的batch。
    #   pp_size > 1 时由CoreManager填写。
    pp_token_addr: str = ""
    """ZMQ endpoint where the head receives sampled tokens back from the last
    stage. Populated by CoreManager for pp_size > 1."""
    # 【译】head从最后一个stage接收采样出的token的ZMQ端点。pp_size > 1 时由CoreManager填写。
    control_address: str = ""
    """ZMQ endpoint carrying control traffic (utility commands, abort, shutdown)
    for this EngineCore, separate from the request endpoint. Keeping the two
    apart leaves the request socket with a single writer thread, so admitting a
    request needs no synchronization. Populated by launch_engine_core."""
    # 【译】承载本EngineCore控制流量（工具命令、abort、关闭）的ZMQ端点，与请求端点分开。
    #   两者分开后，请求socket只有一个写线程，接纳请求就不需要同步。由 launch_engine_core 填写。
    # 解释：EngineCore启动时断言它非空（engine_core.py），第2批
    pp_kv_status_addr: str = ""
    """ZMQ endpoint where the head receives KV offload status from downstream
    PP stages. All downstream stages PUSH; the head PULLs."""
    # 【译】head从下游PP stage接收KV offload状态的ZMQ端点。所有下游stage PUSH，head PULL。
    # DP EngineCore之间gloo进程组的端口（用于DP rank之间同步状态，第2批）
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    # 【译】数据并行master的端口。

    # ModelRunner初始化torch.distributed时的集合点端口：CoreManager在这个端口上起一个TCPStore，
    #   所有ModelRunner作为客户端连上去；0表示让操作系统分配一个空闲端口（单机时可用，多机必须显式指定）
    data_parallel_base_port: int = 0
    """Model-runner rendezvous port. Zero requests an OS-assigned port locally."""
    # 【译】model runner的集合点端口。0表示在本地请求一个由操作系统分配的端口。
    # field(init=False, repr=False)：不出现在__init__参数里，也不出现在打印输出里；由CoreManager置True
    _managed_distributed_store: bool = field(default=False, init=False, repr=False)
    """CoreManager owns the store; every model runner connects as a client."""
    # 【译】store由CoreManager持有；每个model runner作为客户端连接它。

    # DP协调节点（全局DP rank 0所在机器）的IP
    data_parallel_master_ip: str = "127.0.0.1"

    # @property：把方法伪装成属性，调用时写pc.is_multinode_dp，不加括号
    # 谁用：CoreManager（engine_core_mgr.py）。多机时 (1) 本机spawn的EngineCore的全局rank要从
    #   data_parallel_rank 起算而不是从0；(2) 必须显式给 --data-parallel-base-port，且只有全局rank 0
    #   那台机器起TCPStore，其余机器作为客户端去连
    # 例：单机-dp 4：local=4、size=4、rank=0 -> False
    #   2x4多机：第一台local=4 < size=8 -> True；第二台rank=4 > 0 -> True
    @property
    def is_multinode_dp(self) -> bool:
        """Whether this node owns only part of the global DP group.

        Inferred from the topology rather than a separate flag: either this
        node runs fewer ranks than exist globally, or its slice starts at a
        non-zero global rank.
        """
        # 【译】本节点是否只拥有全局DP组的一部分。
        #
        #   由拓扑推断，而不是用单独的开关：要么本节点运行的rank数少于全局总数，
        #   要么本节点的那一段从非0的全局rank开始。
        # data_parallel_size_local is int | None in the declaration, but
        # __post_init__ always resolves it before any caller can reach here.
        # 【译】data_parallel_size_local 在声明里是 int | None，但 __post_init__ 总会在任何调用方
        #   到达这里之前把它解析成整数。
        assert self.data_parallel_size_local is not None
        return (
            self.data_parallel_size_local < self.data_parallel_size
            or self.data_parallel_rank > 0
        )

    # 返回当前的data_parallel_master_port，然后把它加上data_parallel_rank（rank 0时加0）
    # 只有下面的stateless_init_dp_group调用它
    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        # 【译】我们可能需要在多个进程里初始化与数据并行相关的进程组，例如worker和engine里都要，
        #   而它们可能位于不同的进程。为避免端口冲突，每次需要初始化一个新的DP相关进程组时，
        #   都从准备好的端口列表里取出一个新端口。
        # 解释：这段docstring来自vLLM；ATOM的实现并没有"端口列表"，只是返回当前值再自增data_parallel_rank
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += self.data_parallel_rank

        return answer

    # 建立DP EngineCore之间的gloo（CPU）进程组：所有DP rank连到master_ip:master_port
    # 为什么DP rank之间还要通信：forward里有跨DP rank的集合通信（各rank token数的all_reduce、
    #   MoE在dp×tp张卡上的all-to-all等），要求所有DP rank每步都一起跑forward（没请求的rank跑dummy batch陪跑），
    #   所以各rank每步要先同步"是否大家都空闲了"，见has_unfinished_dp；这个同步走CPU上的gloo，不占GPU
    # 调用方：DPEngineCoreProc._init_data_parallel（engine_core.py，第2批），
    #   之后DP rank之间用它做CPU上的all_reduce（例如同步"还有没有rank在跑"）
    # "stateless"：不使用torch.distributed的全局默认进程组，避免和ModelRunner里的NCCL进程组冲突
    def stateless_init_dp_group(self):
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        # 【译】注意：在高并发场景下，多个进程调用 get_open_port() 时可能因为竞争而选中同一个
        #   （当前空闲的）端口。第一个进程绑定该端口后，其他进程随后会以
        #   torch.distributed.DistNetworkError: EADDRINUSE 失败。为了让初始化更健壮，
        #   每当观察到这个特定错误时，我们就换一个新端口重试几次。
        # 解释：这段注释也来自vLLM，描述的重试逻辑并不在下面的代码里
        dp_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            self.get_next_dp_init_port(),
            self.data_parallel_rank,
            self.data_parallel_size,
            backend="gloo",
        )
        return dp_group

    # 多个DP rank汇总"是否还有未完成的请求"：任何一个rank还有，结果就是True
    # 调用方：DPEngineCoreProc._has_global_unfinished_reqs（engine_core.py，第2批），
    #   dp_group就是上面stateless_init_dp_group建的gloo组
    # @staticmethod：不需要self，直接用ParallelConfig.has_unfinished_dp(...)调用
    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        # 【译】dp rank 0：has_unfinished_seqs=True
        #   dp rank 1：has_unfinished_seqs=False
        #   汇总结果：has_unfinished_seqs=True
        #   所以这是一个"或"运算，在整数上就是MAX
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    # 被Config.compute_hash汇总（同CompilationConfig.compute_hash），决定torch.compile缓存目录
    # 注意：把data_parallel_rank、master_port也算了进去，所以不同DP rank的编译缓存目录不同
    #   （也就是说，各DP rank各自编译一份，不共享缓存；改端口也会导致重新编译）
    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # 【译】提供一个hash，唯一标识所有会影响"从input ids/embeddings到最终hidden states"这段计算图结构的配置，
        #   不包括input ids/embeddings之前和最终hidden states之后的部分。
        factors: list[Any] = []
        factors.append(self.data_parallel_size)
        factors.append(self.data_parallel_size_local)
        factors.append(self.data_parallel_rank)
        factors.append(self.data_parallel_rank_local)
        factors.append(self.data_parallel_master_ip)
        factors.append(self.data_parallel_master_port)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    # dataclass生成的__init__执行完后自动调用：环境变量覆盖 -> 补默认值 -> 校验
    # 校验的核心是"本节点的那一段不能越过全局DP组的末尾"：rank + size_local <= size
    #   例：-dp 8、本机data_parallel_rank=4、size_local=4 -> 4+4=8，合法；size_local=5 -> 报错
    def __post_init__(self) -> None:
        # Only override with env vars if explicitly set.
        # This allows programmatic configuration to take precedence.
        # 【译】只有显式设置了环境变量时才用它覆盖。这样以编程方式给出的配置优先生效。
        # 解释：envs.is_set检查环境变量是否存在；不能直接读envs.ATOM_DP_SIZE，因为没设置时它也有默认值，
        #   直接读会把调用方传进来的值冲掉
        if envs.is_set("ATOM_DP_SIZE"):
            self.data_parallel_size = envs.ATOM_DP_SIZE
        if envs.is_set("ATOM_DP_RANK"):
            self.data_parallel_rank = envs.ATOM_DP_RANK
        if envs.is_set("ATOM_DP_RANK_LOCAL"):
            self.data_parallel_rank_local = envs.ATOM_DP_RANK_LOCAL
        if envs.is_set("ATOM_DP_MASTER_IP"):
            self.data_parallel_master_ip = envs.ATOM_DP_MASTER_IP
        if envs.is_set("ATOM_DP_MASTER_PORT"):
            self.data_parallel_master_port = envs.ATOM_DP_MASTER_PORT
        if envs.is_set("ATOM_DP_BASE_PORT"):
            self.data_parallel_base_port = envs.ATOM_DP_BASE_PORT

        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be at least 1")

        if envs.is_set("ATOM_DP_SIZE_LOCAL"):
            self.data_parallel_size_local = envs.ATOM_DP_SIZE_LOCAL

        # Default the local slice to the whole group: on one node every global
        # rank is local, and that is the overwhelmingly common case.
        # 【译】本地这一段默认就是整个组：单节点时每个全局rank都在本地，这是绝大多数情况。
        if self.data_parallel_size_local is None:
            self.data_parallel_size_local = self.data_parallel_size

        if self.data_parallel_size_local < 1:
            raise ValueError("data_parallel_size_local must be at least 1")
        if self.data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        if (
            self.data_parallel_rank + self.data_parallel_size_local
            > self.data_parallel_size
        ):
            raise ValueError(
                f"data_parallel_rank ({self.data_parallel_rank}) + "
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must not exceed data_parallel_size "
                f"({self.data_parallel_size}): this node's slice would run off "
                f"the end of the global DP group"
            )


# ---------------- 【分支】投机解码相关的常量与辅助函数，见第7批 ----------------
# 下面5个常量是--num-speculative-tokens的上限默认值，在Config.__post_init__校验num_speculative_tokens时使用：
#   串行草稿（MTP / EAGLE3）：草稿checkpoint声明了上限就用它，否则MTP为8、EAGLE3为4
#   DSpark：独立草稿模型默认上限16，内嵌在V4目标模型里的草稿默认上限128（滑动窗口大小）
_SERIAL_MTP_DEFAULT_MAX_SPEC = 8
_EAGLE3_DEFAULT_MAX_SPEC = 4
# 草稿模型config里可能用来声明上限的字段名，按顺序查找
_SEQUENTIAL_DRAFTER_MAX_SPEC_ATTRS = (
    "max_speculative_tokens",
    "max_num_speculative_tokens",
    "max_draft_tokens",
)
_DSPARK_DEFAULT_MAX_BLOCK = 16
_DSPARK_DEFAULT_ROLLING_WINDOW = 128


# 串行草稿（MTP/EAGLE3）一次最多能起草几个token
# 调用方：Config.__post_init__，用户的 --num-speculative-tokens 超过返回值就报错。
#   "串行"指草稿器每起草一个token就要再跑一次草稿层（第k个token依赖第k-1个），层数/训练长度有限，
#   超过训练时的步数接受率会急剧下降，所以需要上限
def _resolve_sequential_drafter_max_spec(
    speculative_config: "SpeculativeConfig",
) -> int:
    """Return the supported draft-token horizon for serial MTP/EAGLE drafters."""
    # 【译】返回串行MTP/EAGLE草稿器支持的草稿token范围（上限）。
    draft_cfg = speculative_config.draft_model_hf_config
    for attr in _SEQUENTIAL_DRAFTER_MAX_SPEC_ATTRS:
        max_spec = getattr(draft_cfg, attr, None)
        if max_spec is not None:
            return int(max_spec)

    if speculative_config.method == "eagle3":
        return _EAGLE3_DEFAULT_MAX_SPEC
    return _SERIAL_MTP_DEFAULT_MAX_SPEC


# 调用方：SpeculativeConfig.hf_config_override（草稿模型的architectures[0]以DSparkModel结尾时）
# 输出：在hf_config上写入dspark_target_layer_ids、dspark_noise_token_id、dspark_markov_rank、dspark_with_draft
# 作用：把独立DSpark草稿checkpoint的字段名翻译成内嵌DSpark用的 dspark_* 名字，这样下游只有一套读取代码：
#   - dspark_target_layer_ids：目标模型ModelRunner要在这些层抽出hidden states（aux hidden）喂给草稿
#   - dspark_noise_token_id：DSparkProposer用它填充待起草的block（草稿一次并行生成整个block）
#   - dspark_markov_rank：dspark_markov_sample.py 的采样参数
#   - dspark_with_draft：SpeculativeConfig.use_dspark_with_draft() 读它，drafter.py、dspark_proposer.py、
#     deepseek_v4_replay.py 据此走"有独立草稿权重"的分支
def _normalize_draft_dspark_config(hf_config: PretrainedConfig) -> None:
    """Map a standalone DSpark draft config onto ATOM's canonical names.

    DSpark checkpoints come in two shapes and name the same quantities
    differently:

    - INLINE (e.g. V4-Pro-DSpark) ships inside the target checkpoint and already
      uses the ``dspark_*`` names the rest of ATOM reads. It never comes here.
    - STANDALONE drafts are their OWN checkpoint (``architectures:
      ["*DSparkModel"]``, e.g. Kimi-K3-DSpark's ``K3DSparkModel``) and carry
      their DSpark fields at the config top level: ``target_layer_ids``,
      ``mask_token_id``, ``markov_rank``, and optionally a training block width.
    """
    # 【译】把独立DSpark草稿模型的config映射到ATOM的规范字段名上。
    #
    #   DSpark的checkpoint有两种形态，对同一组量的命名不同：
    #
    #   - 内嵌（INLINE，例如V4-Pro-DSpark）：随目标模型checkpoint一起发布，已经使用ATOM其余部分读取的
    #     dspark_* 名字。它从不会走到这里。
    #   - 独立（STANDALONE）草稿：有自己的checkpoint（architectures: ["*DSparkModel"]，
    #     例如Kimi-K3-DSpark的 K3DSparkModel），DSpark字段直接放在config顶层：
    #     target_layer_ids、mask_token_id、markov_rank，以及可选的训练时block宽度。
    target_layer_ids = getattr(hf_config, "target_layer_ids", None)
    if not target_layer_ids:
        raise ValueError(
            "K3DSparkModel config is missing `target_layer_ids` (the target "
            "decoder layers whose hidden states the draft consumes). Without "
            "it the draft has no context input."
        )
    # 0-based target decoder-layer indices, matching ATOM's `layers[i]` aux-tap
    # convention: the reference indexes `hidden_states[layer_id + 1]`, i.e. the
    # OUTPUT of layer `layer_id`, which is the layer this convention taps.
    # 【译】0起始的目标模型decoder层序号，与ATOM layers[i] 的辅助抽头约定一致：参考实现索引的是
    #   hidden_states[layer_id + 1]，即第 layer_id 层的"输出"，也就是这个约定所抽取的那一层。
    # 解释：HF的hidden_states[0]是embedding输出，所以[layer_id+1]对应第layer_id层的输出，两种写法指同一层
    hf_config.dspark_target_layer_ids = [int(i) for i in target_layer_ids]

    mask_token_id = getattr(hf_config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "K3DSparkModel config is missing `mask_token_id` (the noise token "
            "the draft block is seeded with)."
        )
    hf_config.dspark_noise_token_id = int(mask_token_id)

    hf_config.dspark_markov_rank = int(getattr(hf_config, "markov_rank", 0) or 0)
    # Synthesized marker, not a checkpoint field: the flavor discriminator the
    # rest of the stack branches on (SpeculativeConfig.use_dspark_with_draft).
    # 【译】合成出来的标记，不是checkpoint字段：其余代码据此区分DSpark的形态
    #   （SpeculativeConfig.use_dspark_with_draft）。
    hf_config.dspark_with_draft = True

    # NOTE: no `dspark_block_size` here. Unlike V4-Pro-DSpark and the SpecForge
    # SpecForge DFlash checkpoints, this config carries no block width: the draft is
    # width-agnostic in its weights and the block is sized by
    # --num-speculative-tokens (7 in the checkpoint's own serving recipe).
    # DSparkProposer._resolve_mtp_k falls back to that.
    # 【译】注意：这里没有 dspark_block_size。与V4-Pro-DSpark和SpecForge的DFlash checkpoint不同，
    #   这个config不带block宽度：草稿模型的权重与宽度无关，block大小由 --num-speculative-tokens 决定
    #   （checkpoint自带的部署配方里是7）。DSparkProposer._resolve_mtp_k 会退回到这个值。

    logger.info(
        "Detected MLA DSpark drafter with a separate draft model "
        f"(markov_rank={hf_config.dspark_markov_rank}, "
        f"target_layers={hf_config.dspark_target_layer_ids}, "
        f"mask_token_id={hf_config.dspark_noise_token_id})"
    )


# 【分支】统一不同模型家族的MoE字段名，见第7批
# 调用方：Config.__post_init__（目标模型）、SpeculativeConfig.hf_config_override（草稿模型）
# 做的事：
#   路由专家数：有n_routed_experts用它，否则用num_experts（Qwen系的叫法），统一写成n_routed_experts
#   共享专家数：config里有n_shared_experts就保留；没有时扫描checkpoint，数出共享专家个数（>0才写）
# 写入text_config（有的话）和根config两处，后续代码读哪一个都一致
# 为什么需要：不同家族给同一个量起不同名字（DeepSeek叫n_routed_experts，Qwen/Mixtral叫num_experts）。
#   权重加载（loading_core.py / weight_names.py 把共享专家重命名成 experts.{n_routed_experts}.* 以融合进
#   FusedMoE，weight_dispatch.py 按专家数分发）只读 n_routed_experts / n_shared_experts 这一套名字
# 共享专家数扫描checkpoint：有些config（尤其MTP草稿）没写n_shared_experts，但checkpoint里有共享专家权重，
#   不补上的话这些权重会找不到落脚的参数
# hf_config.update(dict)：PretrainedConfig的方法，把dict里的键值逐个设为属性
def _normalize_moe_config_fields(
    hf_config: PretrainedConfig,
    model_path: str | None = None,
) -> None:
    """Normalize common MoE config field names across model families."""
    # 【译】跨模型家族统一常见的MoE config字段名。
    moe_config = getattr(hf_config, "text_config", hf_config)
    updates: dict[str, Any] = {}

    n_routed = getattr(
        moe_config,
        "n_routed_experts",
        getattr(moe_config, "num_experts", None),
    )
    if n_routed is not None:
        updates["n_routed_experts"] = n_routed

    existing_n_shared = getattr(moe_config, "n_shared_experts", None)
    if existing_n_shared is not None:
        updates["n_shared_experts"] = existing_n_shared
    elif n_routed is not None and model_path is not None:
        from atom.models.utils import ckpt_shared_expert_count

        n_shared = ckpt_shared_expert_count(model_path)
        if n_shared > 0:
            updates["n_shared_experts"] = n_shared

    if not updates:
        return

    moe_config.update(updates)
    if moe_config is not hf_config:
        hf_config.update(updates)


# 【分支】SpeculativeConfig：投机解码的子配置，挂在Config.speculative_config上（不开启时为None），第7批详解
# 投机解码、MTP / EAGLE3（串行草稿）与DSpark（块并行草稿）的入门对比见arg_utils.py EngineArgs.method上方的【科普】
# 来源：arg_utils.py的_get_engine_kwargs，在--method非空且--num-speculative-tokens>0时构造；
#   MTP没有独立草稿模型，model填目标模型路径（MTP层就在目标checkpoint里）
# __post_init__做三件事：
#   1. _resolve_synthetic_acceptance：校验并换算"强制接受率"这个调试开关
#   2. 读草稿模型的HF config（多模态时取text_config）
#   3. hf_config_override：把草稿config改写成ATOM草稿模型类认识的样子（改model_type/architectures等）
# Config.__post_init__随后再校验num_speculative_tokens不超过草稿模型支持的上限
# 谁读它（config.speculative_config）：
#   - spec_decode/factory.py：use_dspark() 决定建 DSparkProposer 还是 EagleProposer（MTP、EAGLE3都走后者）
#   - ModelRunner：按 draft_model_hf_config.architectures 建草稿模型；把 synthetic_acceptance_rates 传给拒绝采样器
#   - 调度器 / BlockManager / 各注意力backend：按 num_speculative_tokens 给每条序列预留 k 个草稿位置的KV slot、
#     决定decode每步每条序列的query长度（1 + k）、CUDA Graph捕获的形状
#   - eagle_proposer.py、vLLM插件 model_wrapper.py：读 method、use_aux_hidden_state、eagle3_aux_layer_ids
@dataclass
class SpeculativeConfig:
    # 投机解码方法："mtp"、"eagle3"、"dspark"
    method: str | None = ""
    # 草稿模型路径（MTP时就是目标模型路径）
    model: str | None = None
    # 每步起草的token数k（--num-speculative-tokens）
    num_speculative_tokens: int | None = None
    # 草稿模型的HF config；为None时__post_init__调用get_hf_config读取
    draft_model_hf_config: PretrainedConfig | None = None
    # EAGLE3专用：草稿模型是否吃目标模型中间层的hidden state，以及取哪几层
    use_aux_hidden_state: bool = False
    eagle3_aux_layer_ids: list[int] = field(default_factory=list)
    # Debug/benchmark knobs: force a speculative acceptance curve independent of
    # the real draft/target agreement, so a run can replay a published
    # acceptance-length figure (an InferenceX golden AL, say) while the draft head
    # is still training. Set at most one; both resolve into
    # `synthetic_acceptance_rates`. See ROCm/ATOM#555.
    #
    # Mean acceptance length in [1, num_speculative_tokens + 1], counting the
    # target's own guaranteed token -- the same unit as vLLM's
    # synthetic_acceptance_length and SGLang's SGLANG_SIMULATE_ACC_LEN, so a
    # published AL can be pasted in without conversion.
    # 【译】调试/压测开关：强制一条与草稿/目标模型真实一致程度无关的投机接受曲线，这样在草稿头还在训练时，
    #   就能复现一个公布的接受长度数字（比如InferenceX的标准AL）。最多设置其中一个；两者都会被解析成
    #   synthetic_acceptance_rates。见 ROCm/ATOM#555。
    #
    #   平均接受长度，取值范围 [1, num_speculative_tokens + 1]，计入目标模型自己保证产出的那个token——
    #   与vLLM的 synthetic_acceptance_length 和SGLang的 SGLANG_SIMULATE_ACC_LEN 单位相同，
    #   所以公布的AL可以直接填进来，不用换算。
    synthetic_acceptance_length: float | None = None
    # The same target expressed as a mean acceptance RATE in [0, 1]
    # (accepted_draft / total_draft), i.e. (length - 1) / num_speculative_tokens.
    # 【译】同一个目标，用平均接受"率"表达，范围 [0, 1]（被接受的草稿数 / 草稿总数），
    #   即 (length - 1) / num_speculative_tokens。
    # 例：k=3、接受率0.5 -> 接受长度 = 1 + 3 x 0.5 = 2.5
    synthetic_acceptance_rate: float | None = None
    # Resolved per-position *unconditional* acceptance rates (entry i = marginal
    # probability that the first i+1 draft tokens are all accepted), filled in by
    # __post_init__. None => real draft/target rejection sampling.
    # 【译】解析后的逐位置"无条件"接受率（第i项 = 前i+1个草稿token全部被接受的边缘概率），
    #   由 __post_init__ 填写。None => 使用真实的草稿/目标拒绝采样。
    # 解释：由rejection_sampler.py的acceptance_length_to_rates生成，采样器据此伪造接受结果
    synthetic_acceptance_rates: list[float] | None = None

    # ClassVar：类变量，不是dataclass字段（不进入__init__），所有实例共享
    # 为什么要改model_type：MTP的草稿config就是目标模型的config.json，model_type和目标一样；
    #   改成 *_mtp 后，下面 _MTP_CONFIG 才能给它换上草稿模型类的architectures。
    #   注意改写后的副作用：草稿层代码里再按model_type判断模型家族时看到的是"deepseek_mtp"，
    #   例如 deepseek_v2.py、sparse_indexer_fp4.py 为此改用GLM专有字段来识别GLM-5的MTP
    # model_type → mtp_model_type mapping
    # 【译】model_type → mtp_model_type 的映射
    # 例：目标模型是deepseek_v3，草稿config的model_type被改成deepseek_mtp，再按下表找到MTP模型类
    _MTP_TYPE_MAP: ClassVar[dict[str, str]] = {
        "deepseek_v3": "deepseek_mtp",
        "deepseek_v32": "deepseek_mtp",
        "deepseek_v4": "deepseek_v4_mtp",
        "glm_moe_dsa": "deepseek_mtp",
        "qwen3_next": "qwen3_next_mtp",
        "qwen3_5": "qwen3_5_mtp",
        "qwen3_5_moe": "qwen3_5_mtp",
        "qwen3_5_text": "qwen3_5_mtp",
        "qwen3_5_moe_text": "qwen3_5_mtp",
        "qwen4_exp_text": "qwen4_exp_mtp",
        "mimo_v2": "mimo_v2_mtp",
        "mimo_v2_flash": "mimo_v2_mtp",
        "glm5_next": "glm5_next_mtp",
        "glm5_next_text": "glm5_next_mtp",
    }

    # mtp_model_type → (n_predict_attr, architecture)
    # 【译】mtp_model_type → (n_predict属性名, 架构名)
    # 解释：n_predict属性名是各家config里"MTP层数"的字段名；架构名用来在registry里找草稿模型类
    _MTP_CONFIG: ClassVar[dict[str, tuple[str, str]]] = {
        "deepseek_mtp": ("num_nextn_predict_layers", "DeepSeekMTPModel"),
        "deepseek_v4_mtp": ("num_nextn_predict_layers", "DeepseekV4MTPModel"),
        "qwen3_next_mtp": ("num_nextn_predict_layers", "Qwen3NextMTPModel"),
        "qwen3_5_mtp": ("mtp_num_hidden_layers", "Qwen3_5MTPModel"),
        "glm5_next_mtp": ("num_nextn_predict_layers", "Glm5NextMTPModel"),
        "qwen4_exp_mtp": ("mtp_num_hidden_layers", "Qwen4ExpMTPModel"),
    }

    # 谁用：spec_decode/factory.py 据此选 DSparkProposer；scheduler.py 据此判断草稿器是否需要
    #   prefill chunk的下一个token（DSpark从aux hidden states起草，不需要；EAGLE/MTP需要）
    def use_dspark(self) -> bool:
        """DSpark semi-autoregressive block drafter (either flavor).

        DSpark is a parallel block drafter (parallel backbone + Markov
        sequential head + confidence head), NOT serial MTP. Two checkpoint
        flavors reach here, and both normalize to `dspark_block_size` in
        `hf_config_override`:

        - V4-Pro-DSpark: ships INSIDE the V4 target checkpoint under the same
          `mtp.*` namespace serial MTP uses, so only the DSpark-only
          `dspark_block_size` field distinguishes the two.
        - Kimi-K3-DSpark: a standalone MLA-backbone checkpoint with its own
          `architectures: ["K3DSparkModel"]`.

        We intentionally never silently fall back to MTP: a wrong fallback
        loads cleanly but measures the wrong algorithm.
        """
        # 【译】DSpark半自回归的块草稿器（两种形态都算）。
        #
        #   DSpark是并行的块草稿器（并行主干 + Markov串行头 + 置信度头），不是串行MTP。
        #   有两种checkpoint形态会走到这里，两者都在 hf_config_override 中被归一化到 dspark_block_size：
        #
        #   - V4-Pro-DSpark：放在V4目标模型的checkpoint里，使用与串行MTP相同的 mtp.* 命名空间，
        #     所以只有DSpark独有的 dspark_block_size 字段能区分二者。
        #   - Kimi-K3-DSpark：一个独立的、以MLA为主干的checkpoint，有自己的 architectures: ["K3DSparkModel"]。
        #
        #   我们有意绝不静默退回到MTP：错误的回退能正常加载，但测量的是错误的算法。
        # 三个条件任一成立即为DSpark：显式--method dspark、草稿config有dspark_block_size、或带dspark_with_draft标记
        cfg = self.draft_model_hf_config
        return (
            self.method == "dspark"
            or bool(getattr(cfg, "dspark_block_size", None))
            or bool(getattr(cfg, "dspark_with_draft", False))
        )

    # 谁用：drafter.py、dspark_proposer.py、deepseek_v4_replay.py——独立草稿有自己的权重和KV，
    #   而内嵌DSpark的草稿层挂在目标模型里，两者加载权重、分配草稿KV的方式不同
    def use_dspark_with_draft(self) -> bool:
        """True when DSpark was given a separate draft model (--draft-model).

        ``dspark_with_draft`` is NOT a checkpoint field -- do not go looking for
        it in config.json. It is synthesized by
        :func:`_normalize_draft_dspark_config`, which runs from
        :meth:`hf_config_override` when the draft's ``architectures`` is
        ``["K3DSparkModel"]``, so it is set for exactly the configs that went
        through that normalization.
        """
        # 【译】DSpark被给定了一个独立的草稿模型（--draft-model）时为True。
        #
        #   dspark_with_draft 不是checkpoint字段——不要去config.json里找它。它由 _normalize_draft_dspark_config
        #   合成；当草稿模型的 architectures 是 ["K3DSparkModel"] 时，hf_config_override 会调用那个函数，
        #   所以恰好是经过那次归一化的config才会带这个标记。
        cfg = self.draft_model_hf_config
        return bool(getattr(cfg, "dspark_with_draft", False))

    # 校验两个"强制接受"开关并换算成逐位置接受率；两个都没设时什么也不做
    # 结果 synthetic_acceptance_rates 由 ModelRunner 传给 rejection_sampler：采样器不再比较草稿和目标，
    #   而是按这组概率伪造"接受了几个"。只用于压测吞吐（输出文本无意义），对应 --spec-decode-acceptance-*
    # 例：k=3、synthetic_acceptance_rate=0.5 -> length=2.5 -> 交给acceptance_length_to_rates生成逐位置接受率
    def _resolve_synthetic_acceptance(self) -> None:
        """Validate the forced-acceptance knobs and resolve to per-position rates.

        Both knobs describe the same curve, so exactly one may be set; the rate
        form is converted to a length and everything downstream reads only
        ``synthetic_acceptance_rates``.
        """
        # 【译】校验强制接受的开关，并解析成逐位置的接受率。
        #
        #   两个开关描述的是同一条曲线，所以最多只能设置一个；接受率形式会被换算成接受长度，
        #   下游所有代码只读取 synthetic_acceptance_rates。
        # Local import: the schedule lives next to the kernel that consumes it,
        # and that module pulls in triton, which has no business loading just
        # because someone imported a config.
        # 【译】局部导入：这个调度表的实现放在使用它的kernel旁边，而那个模块会引入triton——
        #   仅仅因为有人import了一个config就加载triton，是没有道理的。
        from atom.model_ops.rejection_sampler import acceptance_length_to_rates

        if (
            self.synthetic_acceptance_length is not None
            and self.synthetic_acceptance_rate is not None
        ):
            raise ValueError(
                "--spec-decode-acceptance-length and --spec-decode-acceptance-rate "
                "describe the same curve; set at most one."
            )
        length = self.synthetic_acceptance_length
        if self.synthetic_acceptance_rate is None and length is None:
            return

        n = self.num_speculative_tokens
        if not n:
            raise ValueError(
                "Forced speculative acceptance needs --num-speculative-tokens, "
                f"but it is {n!r}."
            )
        if self.synthetic_acceptance_rate is not None:
            rate = self.synthetic_acceptance_rate
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    "synthetic_acceptance_rate (--spec-decode-acceptance-rate) "
                    f"must be in [0, 1], but got {rate}."
                )
            length = 1.0 + n * rate
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                "synthetic_acceptance_length (--spec-decode-acceptance-length) "
                f"must be in [1, {n + 1}] for num_speculative_tokens={n}, but got "
                f"{length}."
            )
        self.synthetic_acceptance_length = length
        self.synthetic_acceptance_rates = acceptance_length_to_rates(length, n)
        logger.info(
            "Forced speculative acceptance ON: mean acceptance length %.4f over "
            "%d draft positions (per-position rates %s). Throughput numbers from "
            "this run are synthetic; output text and accuracy are meaningless.",
            length,
            n,
            [round(r, 4) for r in self.synthetic_acceptance_rates],
        )

    def __post_init__(self):
        self._resolve_synthetic_acceptance()
        # 注意：草稿模型的config总是以trust_remote_code=True读取
        # 读出来的是一个新的config对象，和目标模型的Config.hf_config是两个对象（即使MTP用的是同一个目录）
        if self.draft_model_hf_config is None:
            self.draft_model_hf_config = get_hf_config(
                self.model, trust_remote_code=True
            )
        # For multimodal models, extract text_config
        # 【译】对多模态模型，取出text_config
        if hasattr(self.draft_model_hf_config, "text_config"):
            self.draft_model_hf_config = self.draft_model_hf_config.text_config
        self.hf_config_override(self.draft_model_hf_config, self.model)

        if self.method == "eagle3":
            # MLA drafts (kv_lora_rank set) route to Eagle3DeepseekMLAModel
            # via the arch rewrite in hf_config_override; no early reject.
            # Aux hidden state layers: prefer the draft checkpoint's
            # eagle_config; if absent or the list is empty, ModelRunner
            # falls back to model.get_eagle3_aux_hidden_state_layers(),
            # which defaults to 3 layers — early / middle / late
            # (see DeepseekV2ForCausalLM.get_eagle3_aux_hidden_state_layers,
            # returns `(2, num_layers // 2, num_layers - 3)`, aligned with vLLM).
            # 【译】MLA草稿模型（设置了kv_lora_rank）通过 hf_config_override 中的架构改写被路由到
            #   Eagle3DeepseekMLAModel；不在这里提前拒绝。
            #   辅助hidden state的层：优先用草稿checkpoint的 eagle_config；没有它或列表为空时，ModelRunner
            #   退回到 model.get_eagle3_aux_hidden_state_layers()，默认取3层——前/中/后
            #   （见 DeepseekV2ForCausalLM.get_eagle3_aux_hidden_state_layers，
            #   返回 (2, num_layers // 2, num_layers - 3)，与vLLM一致）。
            # 例：61层的目标模型 -> 默认取第2、30、58层的hidden state
            eagle_cfg = getattr(self.draft_model_hf_config, "eagle_config", None)
            if eagle_cfg:
                self.use_aux_hidden_state = eagle_cfg.get("use_aux_hidden_state", False)
                if self.use_aux_hidden_state and not self.eagle3_aux_layer_ids:
                    self.eagle3_aux_layer_ids = eagle_cfg.get(
                        "eagle_aux_hidden_state_layer_ids", []
                    )
            else:
                self.use_aux_hidden_state = True

    # 原地改写草稿模型的HF config，让ModelRunner能按architectures[0]找到正确的草稿模型类
    # 分支顺序：独立DSpark -> EAGLE3改架构名 -> 内嵌DSpark（V4/V4.1）-> 串行MTP改写 -> MiMo-V2 MTP
    #   内嵌DSpark必须排在MTP前面：它的model_type也是deepseek_v4，先走MTP分支就会被错当成串行MTP
    # @staticmethod：SGLang插件的 deepseek_nextn_wrapper.py / deepseek_v4_nextn_wrapper.py / qwen4_exp_nextn_wrapper.py
    #   也直接调用SpeculativeConfig.hf_config_override(cfg)改写SGLang给的草稿config，不需要实例
    @staticmethod
    def hf_config_override(
        hf_config: PretrainedConfig, model_path: str | None = None
    ) -> None:
        # Eagle3 architecture mapping (architecture-level, not model_type)
        # 【译】Eagle3的架构名映射（按架构名，而不是model_type）
        # (getattr(...) or [""])[0]：architectures缺失或为空列表时取""，避免IndexError
        arch = (getattr(hf_config, "architectures", None) or [""])[0]
        if arch.endswith("DSparkModel"):
            _normalize_draft_dspark_config(hf_config)
            return
        if arch == "LlamaForCausalLMEagle3":
            hf_config.architectures = ["Eagle3LlamaModel"]
        elif arch == "Eagle3DeepseekV2ForCausalLM":
            hf_config.architectures = ["Eagle3DeepseekMLAModel"]

        # DSpark detection (before MTP rewrite): the V4 DSpark checkpoint has
        # model_type=deepseek_v4 just like serial MTP, but carries DSpark-only
        # config fields. Route it to the DSpark draft model and skip the MTP
        # n_predict=1 rewrite (DSpark uses dspark_block_size, not n_predict).
        # 【译】DSpark检测（在MTP改写之前）：V4的DSpark checkpoint和串行MTP一样是 model_type=deepseek_v4，
        #   但带有DSpark独有的config字段。把它路由到DSpark草稿模型，并跳过MTP的 n_predict=1 改写
        #   （DSpark用的是 dspark_block_size，而不是n_predict）。
        if getattr(hf_config, "dspark_block_size", None):
            if hf_config.model_type in ("deepseek_v41", "deepseek_v41_text"):
                hf_config.model_type = "deepseek_v41_dspark"
                hf_config.architectures = ["DeepseekV41DSparkModel"]
                return
            hf_config.model_type = "deepseek_v4_dspark"
            hf_config.architectures = ["DeepseekV4DSparkModel"]
            logger.info(
                "Detected DeepSeek-V4 DSpark drafter "
                f"(block_size={hf_config.dspark_block_size}, "
                f"markov_rank={getattr(hf_config, 'dspark_markov_rank', None)}, "
                f"target_layers={getattr(hf_config, 'dspark_target_layer_ids', None)})"
            )
            _normalize_moe_config_fields(hf_config, model_path)
            return

        # Step 1: resolve model_type → mtp model_type
        # 【译】第1步：把model_type解析成MTP的model_type
        mtp_type = SpeculativeConfig._MTP_TYPE_MAP.get(hf_config.model_type)
        if mtp_type is not None:
            hf_config.model_type = mtp_type

        # Step 2: apply MTP-specific config overrides
        # 【译】第2步：应用MTP专属的config覆盖
        # 不管checkpoint里有几层MTP，都强制n_predict=1：只用1层MTP，循环k次生成k个草稿token
        entry = SpeculativeConfig._MTP_CONFIG.get(hf_config.model_type)
        if entry is not None:
            n_predict_attr, arch = entry
            n_predict = getattr(hf_config, n_predict_attr, 1)
            if n_predict != 1:
                logger.warning(
                    f"Overriding {n_predict_attr} from {n_predict} to 1 "
                    "(MTP typically uses 1 layer that gets reused)"
                )
                n_predict = 1

            updates: dict[str, Any] = {
                "n_predict": n_predict,
                "num_nextn_predict_layers": n_predict,
                "architectures": [arch],
            }
            hf_config.update(updates)

        # MiMo-V2 has not MTP related information in HF config.json,
        # override n_predict with the actual layer count (default 3).
        # 【译】MiMo-V2的HF config.json里没有MTP相关信息，用实际的层数（默认3）覆盖n_predict。
        # 注意：与上面不同，MiMo-V2保留多层MTP，不强制为1
        if hf_config.model_type == "mimo_v2_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 3)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2FlashMTPModel"],
                }
            )

        _normalize_moe_config_fields(hf_config, model_path)
        logger.info(f"hf config is: {hf_config}")

    # 自定义打印：只显示方法和k，不打印整个草稿HF config（太长）
    # f"{x=}"：Python 3.8+的写法，输出"x=值"
    def __repr__(self) -> str:
        method = self.method
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {num_spec_tokens=})"


# 【分支】KVEventsConfig：KV cache事件发布（block被存入/淘汰时对外广播），见第4批 distributed/kv_events.py
# 用途：让外部订阅者（例如KV感知的请求路由器）知道每个实例缓存了哪些block；消息格式与vLLM的kv_events兼容
# 来源：Config.kv_events_config的default_factory是from_env，即只由ATOM_KV_EVENTS_*环境变量配置，没有命令行参数
# 读取方：BlockManager 看 enable 决定是否在分配/释放block时记录事件（不开启时连事件列表都不建，零开销）；
#   Scheduler 用其余字段通过 distributed/kv_events.py 的 make_publisher 建发布器，每步把事件批量发出去（第3、4批）
# 典型场景：多实例部署时，前面的路由器订阅各实例的事件，把共享前缀的请求发到已经缓存了该前缀的实例上
@dataclass
class KVEventsConfig:
    """Configuration for KV cache event publishing."""

    # 【译】KV cache事件发布的配置。

    # 总开关
    enable: bool = False
    # 发布器类型
    # 【译】（行尾注释）可选值 "null" | "zmq"
    publisher: str = "zmq"  # "null" | "zmq"
    # ZMQ PUB socket绑定的地址，以及消息的topic前缀
    endpoint: str = "tcp://127.0.0.1:5557"
    topic: str = ""
    # ZMQ high-water-mark on the PUB socket (0 = unlimited).
    # 【译】PUB socket上的ZMQ高水位线（0 = 不限制）。
    # 解释：高水位线是ZMQ为每个订阅者缓存的最大消息数，超过后新消息被丢弃
    hwm: int = 0
    # Bounded in-process queue between scheduler and sender thread. When full,
    # oldest batch is dropped — KV events are advisory, never stall inference.
    # 【译】调度器与发送线程之间有界的进程内队列。队列满时丢弃最旧的一批——KV事件只是参考信息，
    #   绝不能拖住推理。
    buffer_steps: int = 10_000
    # New fields go after the pre-existing ones so positional constructor
    # calls keep binding the same way.
    # ROUTER endpoint subscribers use to request replay of missed batches by
    # sequence number. Empty string keeps replay disabled (PUB-only).
    # 【译】新字段放在已有字段之后，这样按位置传参的构造调用仍然绑定到相同的字段。
    #   订阅者用来按序号请求重放漏掉的批次的ROUTER端点。空字符串表示不开启重放（只有PUB）。
    replay_endpoint: str = ""
    # Size of the replay ring buffer (distinct from buffer_steps). Bounds the
    # long-lived retention of encoded payloads; only allocated when replay is
    # enabled. Each entry can be sizable (includes token_ids), so tune per the
    # expected event rate and memory budget.
    # 【译】重放环形缓冲区的大小（与buffer_steps不同）。它限制编码后的消息被长期保留的数量；
    #   只有开启重放时才分配。每个条目可能相当大（包含token_ids），所以要按预期的事件速率和内存预算来调。
    replay_buffer_steps: int = 10_000

    # @classmethod：第一个参数是类本身，常用来写"另一种构造方式"
    @classmethod
    def from_env(cls) -> "KVEventsConfig":
        """Build a config from `ATOM_KV_EVENTS_*` env vars. Provides an env-only
        opt-in path so containerized deployments can enable events without a
        CLI flag (see `atom/utils/envs.py`)."""
        # 【译】从 ATOM_KV_EVENTS_* 环境变量构建config。提供一条只靠环境变量开启的途径，
        #   让容器化部署无需命令行参数就能开启事件（见 atom/utils/envs.py）。
        return cls(
            enable=envs.ATOM_KV_EVENTS_ENABLE,
            publisher=envs.ATOM_KV_EVENTS_PUBLISHER,
            endpoint=envs.ATOM_KV_EVENTS_ENDPOINT,
            topic=envs.ATOM_KV_EVENTS_TOPIC,
            replay_endpoint=envs.ATOM_KV_EVENTS_REPLAY_ENDPOINT,
            hwm=envs.ATOM_KV_EVENTS_HWM,
            buffer_steps=envs.ATOM_KV_EVENTS_BUFFER_STEPS,
            replay_buffer_steps=envs.ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS,
        )


# 【分支】DSparkConfig：DSpark投机解码的运行时开关，挂在Config.dspark上，见第7批
# 来源：--dspark-config的JSON字典，经from_dict构造（arg_utils.py的_get_engine_kwargs）；默认全部关闭
# 各key的含义见arg_utils.py中--dspark-config的help（以及下面docstring的Fields）
# 为什么单独成一个类：这些开关横跨ModelRunner、DSpark调度/验证、V4注意力好几个模块，
#   以前散落在环境变量里；收成一个dataclass后，随Config一起pickle到子进程，各处读到的值保证一致
# 谁读哪个字段：
#   confidence_schedule  dspark_proposer.py：是否用置信度头为每个请求动态决定验证长度
#   ragged               model_runner.py、dspark_scheduler.py：每条序列验证长度不同、不补齐到同一长度
#   ragged_graph_sizes   model_runner.py：ragged路径要捕获哪些查询长度的CUDA Graph
#   q_buckets            model_runner.py、dspark_scheduler.py：旧的"整批统一长度"验证路径的Graph桶
#   disable_sps_calib    model_runner.py：预热时是否跳过自动校准（测量不同验证长度下的每步耗时，
#                        供置信度调度权衡"多验证几个token"和"每步变慢"）
#   calibration_profile  spec_decode/calibration.py：读离线测好的校准文件，并校验它与当前模型/TP/GPU/版本匹配
#   deepseek_v41/config.py 也会读其中几项，决定V4.1 DSpark的配置
@dataclass
class DSparkConfig:
    """Runtime configuration for DSpark speculative verification.

    Single source of truth for the DSpark knobs read across the model runner,
    the V4 attention op, and the Eagle proposer. It is built ONCE in the parent
    process from the ``--dspark-config`` JSON dict (see :meth:`from_dict`), then
    pickled into every engine-core worker subprocess as part of :class:`Config`,
    so all read sites observe the same resolved values via ``config.dspark.*``
    (no ``os.environ`` lookups).

    Fields:
      - confidence_schedule: use the DSpark confidence head to pick a per-request
        verify length ell_r (paper Algorithm 1) + variable-length verification.
      - ragged: per-request ragged verify (§5.2 avoid-padding); each decode seq
        forwards its own ell_r+1 tokens (no batch-level q padding).
      - ragged_graph_sizes: comma-separated per-seq CUDA-graph query-length
        buckets to capture for the ragged path (e.g. "1,3,6" or "8").
      - q_buckets: CUDA-graph query-length buckets for the (older) batch-uniform
        q-bucket verify path (independent of the ragged path).
      - disable_sps_calib: skip automatic SPS calibration (replays captured
        graphs at warmup). Without an explicit profile, use the synthetic stub.
      - calibration_profile: JSON file containing model/runtime-bound SPS and
        STS measurements. Takes precedence over automatic calibration.
    """

    # 【译】DSpark投机验证的运行时配置。
    #
    #   它是model runner、V4注意力算子和Eagle proposer读取的DSpark开关的唯一来源。它在父进程中
    #   由 --dspark-config 的JSON字典构建一次（见 from_dict），然后作为 Config 的一部分被pickle进
    #   每个engine-core工作子进程，所以所有读取点通过 config.dspark.* 看到的都是同一组解析好的值
    #   （不读 os.environ）。
    #
    #   字段：
    #     - confidence_schedule：用DSpark的置信度头为每个请求挑选验证长度ell_r（论文算法1）+ 变长验证。
    #     - ragged：每请求不等长的验证（§5.2 避免填充）；每条decode序列forward自己的ell_r+1个token
    #       （batch级别不做q填充）。
    #     - ragged_graph_sizes：逗号分隔的每序列CUDA graph查询长度桶，供ragged路径捕获（例如"1,3,6"或"8"）。
    #     - q_buckets：（较旧的）batch统一q桶验证路径使用的CUDA graph查询长度桶（与ragged路径相互独立）。
    #     - disable_sps_calib：跳过自动SPS校准（预热时回放已捕获的graph）。没有显式profile时使用合成的桩。
    #     - calibration_profile：包含与模型/运行时绑定的SPS和STS测量值的JSON文件。优先于自动校准。

    confidence_schedule: bool = False
    ragged: bool = False
    ragged_graph_sizes: str = ""
    q_buckets: str = ""
    disable_sps_calib: bool = False
    # Explicit offline measurements; compatibility is checked before use.
    # 【译】显式给出的离线测量值；使用前会检查兼容性。
    calibration_profile: str | None = None

    # 校验：给了calibration_profile就必须是非空字符串，且必须同时开启confidence_schedule
    def __post_init__(self):
        if self.calibration_profile is not None and (
            not isinstance(self.calibration_profile, str)
            or not self.calibration_profile
            or not self.confidence_schedule
        ):
            raise ValueError(
                "DSpark calibration_profile requires a path and confidence_schedule=True"
            )

    # 例：from_dict({"confidence_schedule": True, "ragged": True}) -> DSparkConfig(confidence_schedule=True, ragged=True)
    #   from_dict({"ragd": True}) -> ValueError（拼错的key立即报错，而不是被静默忽略）
    #   from_dict(None) -> 全部默认值
    # EPLBConfig、DCPConfig的from_dict是同样的写法
    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DSparkConfig":
        """Build from the ``--dspark-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        # 【译】从 --dspark-config 的JSON字典构建。
        #
        #   cfg 直接映射到这个dataclass的字段上；未知的key会报错，让拼写错误尽早失败。
        cfg = cfg or {}
        # dataclasses.fields(cls)：列出dataclass的所有字段，这里取出合法的字段名集合
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --dspark-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


# 【分支】EPLBConfig：专家负载均衡（Expert Parallel Load Balancing）的调参项，挂在Config.eplb_config上，见第7批
# 背景：MoE里有的专家很"热"，EP下放热点专家的GPU成为瓶颈；EPLB统计各专家负载，定期复制/迁移热点专家
# 开关是Config.eplb_enable（--eplb-enable），这里只是参数；来源是--eplb-config的JSON（from_dict）
# 各字段含义见arg_utils.py中--eplb-config的help，这里不重复
# 谁读：model_ops/eplb.py（负载统计窗口load_window_size、重平衡周期rebalance_interval、
#   每次重平衡分批迁移多少层、P2P搬权重的分块大小、冗余专家的放置策略）；
#   moe.py 按 num_redundant_experts 给每个rank多分配冗余专家的权重槽位（第7批）
# 字段分三组理解：
#   何时统计/何时重平衡：load_window_size、rebalance_interval、rebalance_min_balancedness、rebalance_balancedness_agg
#     （均衡度已经够好就跳过本次重平衡，避免无谓地搬权重）
#   冗余预算与放置：num_redundant_experts、placement_policy
#   搬运开销控制：rebalance_layers_per_chunk、p2p_batch_chunk_size（分批搬，避免一次性占满显存/带宽卡住推理）
@dataclass
class EPLBConfig:
    """EPLB sub-config (vLLM-style: enable + config object)."""

    # 【译】EPLB子配置（vLLM风格：一个开关 + 一个配置对象）。

    load_window_size: int = 1000
    rebalance_interval: int = 3000
    rebalance_layers_per_chunk: int = 64
    num_redundant_experts: int = 0
    rebalance_min_balancedness: float = 2.0
    rebalance_balancedness_agg: str = "min"
    p2p_batch_chunk_size: int = 32
    # Placement policy for spending the redundant-expert budget:
    #   "naive"  -> greedy replicate + balanced_packing (spread thinly)
    #   "biased" -> fully replicate top-K hottest experts to all GPUs
    #               (K = num_redundant // num_gpus, per-node in multi-node)
    # 【译】如何使用冗余专家预算的放置策略：
    #     "naive"  -> 贪心复制 + balanced_packing（分散地薄薄铺开）
    #     "biased" -> 把最热的top-K个专家完整复制到所有GPU
    #                 （K = num_redundant // num_gpus，多节点时按每个节点计算）
    placement_policy: str = "naive"

    # 校验并统一类型：JSON里传来的数字可能是字符串或浮点数，这里强制转成int/float，再检查取值范围
    #   例如要求rebalance_interval >= load_window_size（统计窗口还没攒满就不应该重平衡）
    def __post_init__(self):
        self.load_window_size = int(self.load_window_size)
        assert self.load_window_size > 0, "eplb.load_window_size must be > 0"
        self.rebalance_interval = int(self.rebalance_interval)
        assert self.rebalance_interval > 0, "eplb.rebalance_interval must be > 0"
        assert (
            self.rebalance_interval >= self.load_window_size
        ), "eplb.rebalance_interval must be >= eplb.load_window_size"
        self.rebalance_layers_per_chunk = int(self.rebalance_layers_per_chunk)
        assert (
            self.rebalance_layers_per_chunk > 0
        ), "eplb.rebalance_layers_per_chunk must be > 0"
        self.num_redundant_experts = int(self.num_redundant_experts)
        assert (
            self.num_redundant_experts >= 0
        ), "eplb.num_redundant_experts must be >= 0"
        self.rebalance_min_balancedness = float(self.rebalance_min_balancedness)
        self.rebalance_balancedness_agg = (
            str(self.rebalance_balancedness_agg).lower().strip()
        )
        assert self.rebalance_balancedness_agg in {
            "min",
            "mean",
        }, "eplb.rebalance_balancedness_agg must be one of {'min','mean'}"
        self.p2p_batch_chunk_size = int(self.p2p_batch_chunk_size)
        assert self.p2p_batch_chunk_size > 0, "eplb.p2p_batch_chunk_size must be > 0"
        self.placement_policy = str(self.placement_policy).lower().strip()
        assert self.placement_policy in {
            "naive",
            "biased",
        }, "eplb.placement_policy must be one of {'naive','biased'}"

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "EPLBConfig":
        """Build from the ``--eplb-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        # 【译】从 --eplb-config 的JSON字典构建。
        #
        #   cfg 直接映射到这个dataclass的字段上；未知的key会报错，让拼写错误尽早失败。
        cfg = cfg or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --eplb-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


# ---------------- 【分支】DCP（decode上下文并行）的子配置与门控函数，见第9批 ----------------
# DCP：decode时把一个请求的KV按token切分到TP组内的多张卡上，每张卡只存1/dcp的KV，
#   注意力分别算完再合并（合并输出需要一次集合通信）
#   MLA（kv_lora_rank、"KV只有1个头"）、DCP的token分布图和LSE合并的数字例子，见arg_utils.py EngineArgs.decode_context_parallel_size上方的【科普】

# DCP合并输出时可选的通信方式：ag_rs = AllGather LSE + ReduceScatter输出；a2a = 一次all-to-all
DCP_COMM_BACKENDS = ("ag_rs", "a2a")


# DCPConfig：挂在Config.dcp_config上；来源是--dcp-config的JSON（from_dict）
# 前4个字段的含义见arg_utils.py中--dcp-config的help，这里不重复
# 谁读：
#   interleave_size              deepseek_v2.py（KV按多少个token为一组轮流分给各DCP rank）、
#                                PD分离的mooncake_connector.py（传KV时要按同样的交错方式对齐）
#   enable_query_replication     attention_mla.py：是否开QREP（见下面几个门控函数），省掉decode时AllGather Q
#   enable_project_before_merge  attention_mla.py：合并各rank的部分注意力输出之前先乘W_V，每个头交换的数据
#                                从kv_lora_rank（512）维缩到v_head_dim（128）维，通信量变成约1/4；fp4 BMM时自动关闭
#   comm_backend                 attention_mla.py、dcp_ops.py：合并部分注意力输出用哪种集合通信
#   indexer_dcp_only             distributed/indexer_cp.py：MiniMax-M3只对索引器做上下文并行
# 只有 decode_context_parallel_size > 1（或indexer_dcp_only）时这些字段才有意义
@dataclass
class DCPConfig:
    """DCP (Decode Context Parallel) sub-config: interleave granularity, query
    replication, output-merge placement and the merge collective backend --
    knobs that would otherwise each grow the top-level CLI surface."""

    # 【译】DCP（decode上下文并行）子配置：交错粒度、query复制、输出合并的位置以及合并用的集合通信后端——
    #   这些开关如果各自做成命令行参数，会让顶层命令行越来越庞大。

    interleave_size: int = 1
    enable_query_replication: bool = True
    enable_project_before_merge: bool = True
    comm_backend: str = "a2a"
    # Context-parallelize the MiniMax-M3 lightning indexer INSTEAD of the KV
    # cache: every rank scores all index heads over 1/P of the blocks, then an
    # all-to-all routes each head's candidates to the rank that owns it. Sparse
    # attention, KV geometry and MoE stay TP. Requires
    # decode_context_parallel_size == 1 -- see indexer_cp_unsupported_reason.
    # 【译】对MiniMax-M3的lightning索引器做上下文并行，"代替"对KV cache做：每个rank在1/P的block上
    #   为全部索引头打分，然后一次all-to-all把每个头的候选发给拥有该头的rank。稀疏注意力、KV几何和MoE
    #   仍然是TP。要求 decode_context_parallel_size == 1——见 indexer_cp_unsupported_reason。
    # 解释：lightning索引器就是MiniMax-M3的DSA式稀疏注意力索引器（入门见glm5_kpool_block_size上方【科普】②）
    indexer_dcp_only: bool = False

    # 类型统一 + 取值校验；comm_backend必须在DCP_COMM_BACKENDS里
    def __post_init__(self):
        self.interleave_size = int(self.interleave_size)
        assert self.interleave_size >= 1, "dcp.interleave_size must be >= 1"
        self.indexer_dcp_only = bool(self.indexer_dcp_only)
        self.enable_query_replication = bool(self.enable_query_replication)
        self.enable_project_before_merge = bool(self.enable_project_before_merge)
        self.comm_backend = str(self.comm_backend)
        assert self.comm_backend in DCP_COMM_BACKENDS, (
            f"dcp.comm_backend must be one of {list(DCP_COMM_BACKENDS)}; "
            f"got {self.comm_backend!r}"
        )

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DCPConfig":
        """Build from the ``--dcp-config`` JSON dict.

        ``cfg`` maps directly onto this dataclass' fields; unknown keys raise so
        typos fail fast."""
        # 【译】从 --dcp-config 的JSON字典构建。
        #
        #   cfg 直接映射到这个dataclass的字段上；未知的key会报错，让拼写错误尽早失败。
        cfg = cfg or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown --dcp-config key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}"
            )
        return cls(**cfg)


# 下面5个函数都是DCP query复制（QREP）/ MiniMax-M3索引器CP的门控判断，写成模块级纯函数便于单测
# QREP是什么：DCP下每个rank本来只算自己那部分q头，decode时要AllGather一次Q；QREP在加载时把q_proj
#   复制/加宽，让每个rank本地就算出整个DCP组的q头，省掉这次AllGather（见arg_utils.py中--dcp-config的help）
# 调用关系：
#   qrep_unsupported_reason       Config.__post_init__：全局层面能不能开QREP，不能就关掉开关
#   qrep_enabled_for_layer        attention_mla.py：逐层判断（= 想开 且 q_proj确实被加宽 且 scale可按行切）
#     ├─ q_proj_is_qrep_widened
#     └─ q_proj_has_row_sliceable_scale
#   qrep_for_step                 attention_mla.py：每次forward判断本步是否走QREP
#   indexer_cp_unsupported_reason Config.__post_init__：MiniMax-M3索引器CP能不能开（不能就把indexer_dcp_only清成False，
#                                 之后 distributed/indexer_cp.py 在运行时只需读这个开关）
# 为什么放在config.py而不是attention_mla.py：attention_mla.py会import triton/aiter，而这些判断要能在
#   没有GPU库的CPU单测里测；Config.__post_init__ 也要用到其中两个


# 返回不能开QREP的原因字符串；能开时返回None
# 调用方：Config.__post_init__，返回非None时把 dcp_config.enable_query_replication 关掉并打日志说明原因，
#   这样后面的逐层判断和启动日志都和实际行为一致
def qrep_unsupported_reason(dcp_size: int, mxfp4_bmm: bool) -> str | None:
    """Why DCP query replication cannot run here, or None if it can.

    Kept a module-level pure function so it is unit-testable: the alternative,
    exercising it through ``Config.__post_init__``, needs a real model directory
    and an HF config. ``Config.__post_init__`` is its only production caller.
    """
    # 【译】DCP query复制在这里不能运行的原因；能运行则返回None。
    #
    #   保持为模块级的纯函数，以便单元测试：否则只能通过 Config.__post_init__ 来测，而那需要一个真实的
    #   模型目录和HF config。Config.__post_init__ 是它在生产代码中唯一的调用方。
    if dcp_size <= 1:
        # No DCP group means there is no AllGather Q to remove.
        # 【译】没有DCP组，就没有要省掉的AllGather Q。
        return "decode_context_parallel_size <= 1 (no DCP group)"
    if mxfp4_bmm:
        # fp4 (mxfp4) absorbed BMM has a different scale structure.
        # 【译】fp4（mxfp4）的吸收BMM有不同的scale结构。
        return "fp4 (mxfp4) BMM weights"
    return None


# 调用方：qrep_enabled_for_layer（即 attention_mla.py 的MLA层在 __init__/加载后做逐层判断时）
def q_proj_is_qrep_widened(q_proj, qrep_num_heads: int, qk_head_dim: int) -> bool:
    """Whether `q_proj` was actually built with `qrep_tp_override`.

    Requires both provenance (`q_proj.effective_tp_overridden`) and shape
    (`weight.shape[0] == qrep_num_heads * qk_head_dim`). Shape alone isn't
    enough: at `tp == dcp`, `override_tp_size` becomes 1 (a no-op), so a
    QREP-widened q_proj and a plain, never-sharded one end up the same
    width. Provenance alone isn't enough either: it says intent, not that
    the resulting shape is what `_local_q_proj`/`W_K_qrep` actually assume.

    Known residual gap: if `q_proj` is a wrapper that hides `.weight`
    (`getattr` returns None, e.g. GLM-5.3's `_ZeroRopePad`) while its
    wrapped linear actually was overridden, this returns False and the
    caller falls back to AllGather -- safe today because no wrapper of
    that shape is ever also wired to `qrep_tp_override`, but nothing here
    would catch a future model that combines both.

    Dependency-free so it stays importable, and testable, without triton/aiter.
    """
    # 【译】q_proj 是否真的是用 qrep_tp_override 构建的。
    #
    #   同时要求来源证明（q_proj.effective_tp_overridden）和形状（weight.shape[0] == qrep_num_heads * qk_head_dim）。
    #   只看形状不够：tp == dcp 时，override_tp_size 变成1（等于没覆盖），于是一个QREP加宽过的q_proj和
    #   一个普通的、从未切分过的q_proj宽度相同。只看来源证明也不够：它说明的是意图，而不能保证得到的形状
    #   就是 _local_q_proj / W_K_qrep 实际假定的那样。
    #
    #   已知的剩余漏洞：如果 q_proj 是一个隐藏了 .weight 的包装器（getattr 返回None，例如GLM-5.3的
    #   _ZeroRopePad），而它包住的linear确实被覆盖过，这里会返回False，调用方退回到AllGather——
    #   目前这是安全的，因为没有哪个这种形态的包装器同时接到 qrep_tp_override 上；但如果未来某个模型
    #   两者兼有，这里不会发现。
    #
    #   不依赖任何库，所以不需要triton/aiter也能导入和测试。
    if not getattr(q_proj, "effective_tp_overridden", False):
        return False
    weight = getattr(q_proj, "weight", None)
    return weight is not None and weight.shape[0] == qrep_num_heads * qk_head_dim


# Quant types `make_row_view` narrows safely for a single-partition q_proj
# (unquantized/per_Tensor never hit its per-output-channel branch there;
# per_1x128/per_Token are the two it explicitly handles). Named, not
# imported from aiter's QuantType enum, so this stays importable without
# triton/aiter; the aiter side is pybind11-bound but its members' `.name` is
# this same string.
# 【译】对单分区的q_proj，make_row_view 能安全按行收窄的量化类型（不量化/per_Tensor在那里从不会走到
#   按输出通道的分支；per_1x128/per_Token是它显式处理的两种）。这里写成名字字符串而不是从aiter的
#   QuantType枚举导入，这样不需要triton/aiter也能导入；aiter那边是pybind11绑定的，但其成员的 .name
#   就是同样的字符串。
_ROW_SLICEABLE_QUANT_TYPE_NAMES = frozenset(
    {"No", "per_Tensor", "per_1x128", "per_Token"}
)


# 调用方：qrep_enabled_for_layer。逐层拦截"q_proj的量化格式不支持按行切scale"的情况
#   （QREP下prefill要从加宽的q_proj里切出本rank那几行，scale也得能跟着切）
def q_proj_has_row_sliceable_scale(q_proj) -> bool:
    """Whether `_local_q_proj`'s row view can safely narrow `q_proj`'s scale.

    `make_row_view` narrows a 2D per-output-channel `weight_scale` only for
    `per_1x128`/`per_Token`; anything else (e.g. mxfp4's `per_1x32`) raises
    `NotImplementedError` on first use. `qrep_unsupported_reason`'s mxfp4
    gate only checks a global env var, not any layer's actual quant type, so
    this catches it per layer instead.

    Keyed on `quant_type` alone, not `weight_scale`'s presence/shape:
    `quant_type` is set unconditionally in `LinearBase.__init__`, but
    `weight_scale` is not -- the online-quant-streaming path
    (`source_quant_dtype` set) defers building it to
    `process_weights_after_loading`, well after this runs. Reading
    `weight_scale is None` as "safe" there would be checking a scale that
    does not exist yet, not one that will never exist.
    """
    # 【译】_local_q_proj 的行视图能否安全地收窄 q_proj 的scale。
    #
    #   make_row_view 只对 per_1x128/per_Token 收窄二维、按输出通道的 weight_scale；其他类型
    #   （例如mxfp4的per_1x32）第一次使用时会抛 NotImplementedError。qrep_unsupported_reason 对mxfp4的
    #   门控只检查一个全局环境变量，而不看任何层实际的量化类型，所以这里改为逐层拦截。
    #
    #   只按 quant_type 判断，而不看 weight_scale 是否存在/形状如何：quant_type 在 LinearBase.__init__ 中
    #   无条件设置，weight_scale 则不然——在线量化流式加载路径（设置了source_quant_dtype）会把它推迟到
    #   process_weights_after_loading 才构建，远晚于这里运行的时刻。在那里把 weight_scale is None 当作
    #   "安全"，检查的是一个"尚不存在"的scale，而不是一个"永远不会存在"的scale。
    name = getattr(getattr(q_proj, "quant_type", None), "name", None)
    return name is None or name in _ROW_SLICEABLE_QUANT_TYPE_NAMES


# 调用方：attention_mla.py 的MLA注意力层，结果存为 self.qrep_enabled，之后每步交给 qrep_for_step
# wants_qrep：全局开关（dcp_config.enable_query_replication 经 qrep_unsupported_reason 过滤后的结果）
def qrep_enabled_for_layer(
    wants_qrep: bool, q_proj, qrep_num_heads: int, qk_head_dim: int
) -> bool:
    """The per-layer QREP decision, kept testable on its own: asserting only
    `q_proj_is_qrep_widened`'s behavior can't catch the `and` being dropped
    (or weakened) at the call site. Also requires `q_proj_has_row_sliceable_scale`
    so a layer that is genuinely QREP-widened but whose on-disk quant type
    `_local_q_proj` cannot row-slice falls back to AllGather instead of
    raising the first time prefill runs.
    """
    # 【译】逐层的QREP决策，单独保持可测试：只断言 q_proj_is_qrep_widened 的行为，发现不了调用处的
    #   and 被删掉（或被削弱）。还要求 q_proj_has_row_sliceable_scale，这样一个确实被QREP加宽、但磁盘上的
    #   量化类型无法被 _local_q_proj 按行切分的层，会退回到AllGather，而不是在第一次跑prefill时报错。
    return (
        wants_qrep
        and q_proj_is_qrep_widened(q_proj, qrep_num_heads, qk_head_dim)
        and q_proj_has_row_sliceable_scale(q_proj)
    )


# 调用方：attention_mla.py 每次forward；决定本步是本地算全部q头，还是走AllGather Q
def qrep_for_step(
    qrep_enabled: bool, use_seg_mla: bool, is_prefill: bool, prefill_qrep: bool
) -> bool:
    """Whether this forward produces the DCP group's query heads locally (QREP)
    instead of gathering them.

    Decode always does when the layer has QREP. Prefill reaches the QREP-aware
    branch only as sparse (DSA) prefill -- dense prefill takes the MHA path and
    never gathers q -- and joins only under ``ATOM_DCP_PREFILL_QREP``
    (``prefill_qrep``). The seg path is excluded because its q_out is allocated
    at the per-rank head count. Pure so the prefill clause can be pinned by a
    test instead of living only as an inline expression.
    """
    # 【译】本次forward是否在本地产生DCP组的全部query头（QREP），而不是收集它们。
    #
    #   层开启了QREP时，decode总是这样做。prefill只有在稀疏（DSA）prefill时才会进入感知QREP的分支——
    #   dense prefill走MHA路径，从不收集q——并且只有在 ATOM_DCP_PREFILL_QREP（prefill_qrep）下才加入。
    #   seg路径被排除，因为它的q_out是按每个rank的头数分配的。写成纯函数，是为了让prefill这一条能被测试
    #   固定下来，而不是只作为一个内联表达式存在。
    return qrep_enabled and not use_seg_mla and (not is_prefill or prefill_qrep)


# 调用方：Config.__post_init__（dcp_config.indexer_dcp_only=True 时）；返回非None就把开关清掉并打日志
# 参数都是从Config / hf_config里取出来的标量，而不是整个Config，正是为了能在单测里直接构造
def indexer_cp_unsupported_reason(
    arches,
    tp_size: int,
    num_kv_heads: int,
    sparse_block_size: int,
    dcp_size: int,
    enable_tbo: bool,
    plugin_mode: bool = False,
) -> str | None:
    """Why MiniMax-M3 indexer-only context parallelism cannot run here, or None.

    Pure and module-level for the same reason as ``qrep_unsupported_reason``:
    the alternative is a real model directory and an HF config.

    Note the DCP relationship is EXCLUSIVE, not a prerequisite. This shards the
    indexer's block scoring; real DCP shards the KV cache itself (BlockManager
    scales the prefix-cache hash granularity by dcp_world_size), and M3 has no
    DCP-aware attention path at all. The two cannot both own the context axis.

    ``plugin_mode`` belongs here rather than only in ``indexer_cp_enabled``
    because this gate is what clears the FLAG. Leaving it set under a bridge
    left the config claiming CP while the runtime gate silently served TP, so
    the startup line an A/B is diagnosed from ("indexer_dcp_only enabled") said
    the opposite of what ran -- and ``compute_hash`` keyed on a value the model
    never used.
    """
    # 【译】MiniMax-M3仅索引器的上下文并行在这里不能运行的原因；能运行则返回None。
    #
    #   与 qrep_unsupported_reason 一样写成模块级纯函数，原因相同：否则需要真实的模型目录和HF config。
    #
    #   注意它和DCP的关系是"互斥"，而不是"前提"。这里切分的是索引器的block打分；真正的DCP切分的是
    #   KV cache本身（BlockManager按dcp_world_size放大前缀缓存的hash粒度），而M3根本没有感知DCP的注意力路径。
    #   两者不能同时占有上下文这一维。
    #
    #   plugin_mode 放在这里而不只放在 indexer_cp_enabled 里，是因为清除"开关"的正是这道门控。
    #   以前在bridge下保留这个开关，会让config声称开着CP，而运行时门控却静默地走TP，于是诊断A/B所依据的
    #   启动日志行（"indexer_dcp_only enabled"）说的和实际运行的正好相反——而且 compute_hash 依据的是
    #   一个模型从未使用的值。
    if not any("MiniMaxM3" in str(a) for a in arches):
        return "not a MiniMax-M3 model"
    if plugin_mode:
        # The vLLM/SGLang bridges run ATOM's own linear.py and minimax_m3/model.py, so
        # the flag would widen index_q underneath them -- and both reshape it
        # with a ``view`` on the width they assume, yielding 4x the rows with no
        # error. ``indexer_cp_enabled`` refuses them at runtime; this clears the
        # flag so nothing downstream (log line, compile hash) disagrees.
        # 【译】vLLM/SGLang的bridge运行的是ATOM自己的linear.py和minimax_m3/model.py，所以这个开关会在它们
        #   底下加宽index_q——而两者都按自己假定的宽度用 view 改变形状，得到4倍的行数且不报错。
        #   indexer_cp_enabled 在运行时拒绝它们；这里清除开关，让下游（日志行、编译hash）都不出现矛盾。
        return "plugin mode (the vLLM/SGLang bridges keep the TP indexer path)"
    if dcp_size > 1:
        return (
            "decode_context_parallel_size > 1 (KV-cache DCP owns the context "
            "axis; indexer_dcp_only replaces it, it does not extend it)"
        )
    if tp_size != num_kv_heads:
        # Below TP4 a rank holds >1 kv head, which the candidate merge and the
        # gluon decode kernel both reject (they assume per-rank num_kv_heads
        # == 1). Above it the group is a strided subset of TP -- not wired yet.
        # 【译】TP4以下时一个rank持有多于1个kv头，候选合并和gluon decode kernel都会拒绝
        #   （它们假定每个rank的num_kv_heads == 1）。TP4以上时这个组是TP的一个跨步子集——还没有接通。
        return (
            f"tensor_parallel_size ({tp_size}) != num_key_value_heads "
            f"({num_kv_heads}); v1 supports only the square case"
        )
    if sparse_block_size != 128:
        return f"sparse_block_size {sparse_block_size} != 128"
    # Speculative decoding is deliberately NOT rejected. The whole chain takes
    # max_query_len as a runtime argument and validates against it --
    # indexer_context_scores, local_candidate_keys, merge_candidate_keys --
    # and ``tests/model_ops/test_indexer_cp_parity.py`` pins torch.equal against
    # the native kernel at both qlen 1 and qlen 4 (EAGLE3 with 3 draft tokens).
    #
    # Rejecting it is also a throughput loss, though NOT for the reason an
    # earlier revision of this comment gave. That version claimed drafting
    # "fills the MMA tile" (1x1 = 6% -> 4x4 = 100%). Measured on MI355X, both
    # arms run the score kernel at 5.5-7.5 TB/s against an ~8 TB/s roof: the
    # kernel is BANDWIDTH-bound, never MMA-starved, and the CP win is the 4x
    # byte ratio (every rank reads 1/P of the blocks from a REPLICATED one-head
    # index cache). The tile width does matter, but with the opposite sign --
    # both kernels size the dot's N as max(16, next_pow2(heads * query_len)), so
    # TP's N is 16 for every qlen <= 16 (flat: 90.2/92.2/90.8us at q=1/4/8)
    # while CP's doubles at q=5 (20.3/20.4/24.0us). q=4 is CP's exact fill point
    # and free; q=8 costs it 17% for nothing. Pair this flag with MTP3.
    # 【译】投机解码有意不被拒绝。整条链路都把 max_query_len 作为运行时参数并据此校验——
    #   indexer_context_scores、local_candidate_keys、merge_candidate_keys——而且
    #   tests/model_ops/test_indexer_cp_parity.py 在qlen 1和qlen 4（EAGLE3带3个草稿token）下都用
    #   torch.equal 与原生kernel对比固定了结果。
    #
    #   拒绝它还会损失吞吐，不过原因"不是"这段注释早先版本给出的那个。那个版本声称起草能"填满MMA tile"
    #   （1x1 = 6% -> 4x4 = 100%）。在MI355X上实测，两个分支运行打分kernel都在5.5-7.5 TB/s，上限约8 TB/s：
    #   这个kernel是"带宽受限"的，从不缺MMA算力；CP的收益来自4倍的字节比（每个rank只从一份"复制的"单头
    #   索引缓存里读1/P的block）。tile宽度确实有影响，但符号相反——两个kernel都把点积的N取为
    #   max(16, next_pow2(heads * query_len))，所以TP的N在qlen <= 16时都是16（持平：q=1/4/8时
    #   90.2/92.2/90.8us），而CP的N在q=5时翻倍（20.3/20.4/24.0us）。q=4是CP恰好填满的点，不额外花钱；
    #   q=8则让它白白多花17%。请把这个开关和MTP3搭配使用。
    # 解释：MTP3即每步3个草稿token，加上目标模型自己的1个，qlen=4，正好是上面说的CP填满点
    if enable_tbo:
        # Two ubatch threads issuing all-to-alls on one group with no ordering
        # discipline deadlock.
        # 【译】两个ubatch线程在同一个通信组上发出all-to-all、又没有顺序约束，会死锁。
        return "TBO (two-batch overlap)"
    return None


# =====================================================================================
# Config：一次启动的全部配置（本文件的主角）
# =====================================================================================
# 在架构中的位置：
#   EngineArgs ──_get_engine_kwargs──> LLMEngine.__init__ ──Config(model, **kwargs)──> Config
#     （LLMEngine只把"Config里有同名字段"的kwargs传进来）          │ __post_init__：读HF config、推导、校验
#                                                                  ▼
#   LLMEngine补几个字段：bos/eos_token_id（来自tokenizer）、stop_token_ids去掉eos、DP大小（旧路径）
#                                                                  ▼
#   CoreManager（第2批）：每个DP rank复制一份Config，填ParallelConfig里的rank和地址，随进程启动pickle给EngineCore
#                                                                  ▼
#   EngineCore：把Config交给每个ModelRunner进程；ModelRunner算出能放多少KV block后，EngineCore回填
#     config.num_kvcache_blocks，再用这个Config构造Scheduler（其中的BlockManager按它建block池）
#                                                                  ▼
#   ModelRunner.__init__调用set_current_atom_config(config)，模型层用get_current_atom_config()读取
#
# 字段大致分组（声明顺序有些交错）：
#   调度/KV：max_num_batched_tokens、max_num_seqs、max_model_len、kv_cache_block_size、enable_prefix_caching、
#     enable_chunked_prefill、gpu_memory_utilization……（含义与调大调小的代价见arg_utils.py的EngineArgs）
#   状态缓存/有界重放：state_checkpoint_*、*_bounded_replay、min_final_prefill_rows……（第4、8批）
#   并行：tensor/pipeline/decode_context/prefill_context_parallel_size、parallel_config、enable_dp_attention……
#   派生字段（init=False，或由__post_init__/其他组件回填）：hf_config、generation_config、quant_config、
#     torch_dtype、num_kvcache_blocks、bos/eos_token_id、stop_token_ids
#   子配置：compilation_config、speculative_config、dcp_config、eplb_config、dspark、kv_events_config
#   插件/RapidServe/RL rollout专用：plugin_config、disagg_*、true_vocab_size、sleep_keeps_memory_resident
#
# 注意：这里的@dataclass没有slots=True（nano-vllm加了），所以__post_init__和其他组件还能给实例挂上
#   未声明的属性，例如multimodal_config、offload_page_hf_geometry，以及EngineCore回填的pool_entries
#
# __post_init__的步骤（★是主干，其余是分支，读第1批时可以先跳过；带报错/降级条件的流程图和一个走查例子在__post_init__开头）：
#   1. MoE后端字段的规范化与校验（moe_all2all_backend、moe_backend）                          第7批
#   2. 子配置dict→对象并重新校验（compilation_config、eplb_config、dcp_config）
#   3. 强制接受率与DSpark置信度调度互斥；RapidServe换runner类                                 第7、5批
#   4. ★ TP范围检查；DCP相关检查（整除、gfx950、interleave_size、QREP门控）                   DCP见第9批
#   5. ★ 读HF config（get_hf_config）+ hf_overrides + 各模型字段归一化 + rope参数补齐
#   6. ★ 读generation_config得到stop_token_ids；构造QuantizationConfig
#   7. ★ 推导max_model_len；校验long_prefill_token_threshold；创建profiler目录
#   8. ★ 编译配置：level 3时设置切图算子、cudagraph_mode默认值、compile_sizes
#   9. ★ torch_dtype；解析kv_transfer_config；offload快照（第9批）
#  10. 动态分块的校验/降级                                                                   第9批
#  11. 投机解码：num_speculative_tokens的上限校验                                              第7批
#  12. 按模型覆盖kv_cache_block_size（V4=256、GLM-5.3=index_kpool x 16）                       第8批
#  13. MiniMax-M3索引器CP门控；index_cache_dtype默认值；V4.1（block_size=128 + 重放参数）；V4重放参数  第8、9批
#
# 对比nano-vllm：nano-vllm的Config只有11个字段，__post_init__只做"读HF config + 截断max_model_len"
#   （对应上面第5、7步的一小部分）
@dataclass
class Config:
    # 模型的本地目录或HF仓库名（LLMEngine的第一个位置参数）
    model: str
    # 以下字段的含义和取值影响见arg_utils.py的EngineArgs中同名字段，这里只写一句话
    # 是否允许执行模型仓库里的自定义代码（get_hf_config、加载tokenizer时使用）
    trust_remote_code: bool = False
    # 单步最多计算的token数（调度器的token预算）
    max_num_batched_tokens: int = 16384
    # chunked prefill时单个请求每步最多prefill多少token，0不限制；__post_init__校验它在[block_size, max_model_len]内
    long_prefill_token_threshold: int = 0
    # MLA注意力内部做chunked prefill时的token预算（第7批，aiter_mla.py读取）
    attn_prefill_chunk_size: int = 16384
    # 【分支】以下到enable_encoder_swa_bounded_replay为止，是带每请求状态的模型（DeepSeek-V4/V4.1、GDN等）
    #   的状态缓存与有界重放，读取方是BlockManager和调度器，见第4、8批
    #   概念入门（每请求状态、状态checkpoint、SWA有界重放，带图和数字）见arg_utils.py EngineArgs.state_checkpoint_interval_tokens上方的【科普】
    # Tokens between rungs of the state-checkpoint ladder, shared by every
    # Pool.STATE class. Must be a multiple of the prefix-cache hash block size
    # (snapped, with a warning, in BlockManager).
    #   >0  a rung every N tokens
    #    0  state checkpointing off entirely
    #   -1  no interval rungs, but the demand rung and the prompt-end anchor
    #       still place checkpoints
    # See BlockManager.checkpointers_at.
    # 【译】状态checkpoint"阶梯"上相邻两级之间的token数，所有 Pool.STATE 类共用。必须是前缀缓存hash
    #   block大小的倍数（不是的话，BlockManager会对齐并给出警告）。
    #     >0  每N个token一级
    #      0  完全关闭状态checkpoint
    #     -1  不按间隔放置，但"按需"那一级和prompt末尾的锚点仍会放置checkpoint
    #   见 BlockManager.checkpointers_at。
    state_checkpoint_interval_tokens: int = 8192
    # Whether a refused hit may place a rung of its own. Off leaves the
    # prompt-end anchor as the only placement; the rung reads back far less
    # often than the anchor, so its worth is an open question — see
    # `StateSlotPool.mark_speculative` for the measurement and
    # `BlockManager._record_checkpoint_demand` for the placement.
    # 【译】一次被拒绝的命中是否可以为自己放置一级checkpoint。关闭时只剩prompt末尾的锚点这一种放置；
    #   这一级被读回的频率远低于锚点，所以它值不值得还没有定论——测量见 StateSlotPool.mark_speculative，
    #   放置见 BlockManager._record_checkpoint_demand。
    state_checkpoint_demand: bool = True
    # DeepSeek-V4.1 decoder SWA bounded replay
    # (models/deepseek_v41/bounded_replay.py). On by default;
    # --no-decoder-swa-bounded-replay turns it off. No effect on other models.
    # 【译】DeepSeek-V4.1 decoder侧的SWA有界重放（models/deepseek_v41/bounded_replay.py）。默认开启；
    #   --no-decoder-swa-bounded-replay 关闭它。对其他模型没有影响。
    enable_decoder_swa_bounded_replay: bool = True
    # The fewest rows the forward that finishes a prompt runs: one that resumes
    # closer to the prompt's end starts that much earlier, recomputing rows
    # the cache already holds (`BlockManager.replay_start`). Set by the model,
    # not a flag: V4.1's decoder bounded replay reads the last `window` rows.
    # 【译】完成一个prompt的那次forward至少要跑的行数：从更靠近prompt末尾处恢复的forward会相应地提前开始，
    #   重新计算缓存里已有的行（BlockManager.replay_start）。由模型设置，不是命令行开关：V4.1的decoder
    #   有界重放要读最后 window 行。
    # 解释：__post_init__末尾对V4.1设为hf_config.sliding_window；其他模型保持0
    min_final_prefill_rows: int = 0
    # The tokens before a replaying forward a backend reads (`Replay.history_ids`).
    # Set by the model: V4.1's Engram n-gram lookback.
    # 【译】一次重放forward之前、后端要读取的token（Replay.history_ids）。由模型设置：V4.1的Engram n-gram回看。
    # 解释：__post_init__末尾对V4.1设为engram_max_ngram_size - 1；调度器据此在Replay里带上历史token
    #   Engram是什么（按最近几个token的n-gram查一张大表）见envs.py ATOM_ENGRAM_*上方的科普
    replay_history_tokens: int = 0
    # How far below a prefix hit with no state checkpoint behind it a prefill
    # starts to rebuild that state exactly (`BlockManager.replay_start`), 0 =
    # such a hit is not taken. Set by the model, not a flag: DeepSeek-V4's
    # window receptive field under ATOM_DSV4_STATE_REPLAY (`deepseek_v4_replay`).
    # 【译】对一个背后没有状态checkpoint的前缀命中，prefill要从命中位置往前多远开始，才能精确重建那份状态
    #   （BlockManager.replay_start）；0 = 不接受这种命中。由模型设置，不是命令行开关：
    #   ATOM_DSV4_STATE_REPLAY 下DeepSeek-V4的窗口感受野（deepseek_v4_replay）。
    state_replay_tokens: int = 0
    # DeepSeek-V4.1 encoder SWA bounded replay: a prefix hit keeps no state
    # checkpoint and rebuilds it by recomputing the window rows below the hit
    # (`--encoder-swa-bounded-replay`). On by default;
    # --no-encoder-swa-bounded-replay keeps checkpoints. No effect on other models.
    # 【译】DeepSeek-V4.1 encoder侧的SWA有界重放：前缀命中不保留状态checkpoint，而是重新计算命中位置之下
    #   窗口内的行来重建它（--encoder-swa-bounded-replay）。默认开启；--no-encoder-swa-bounded-replay
    #   保留checkpoint。对其他模型没有影响。
    enable_encoder_swa_bounded_replay: bool = True
    # 调度延迟因子：攒更多请求再一起prefill，0不等待（调度器读取，第3批）
    scheduler_delay_factor: float = 0.0
    # 单步最多调度的序列数
    max_num_seqs: int = 512
    # 最大上下文长度；None表示用HF config的max_position_embeddings，__post_init__里解析成整数
    max_model_len: int | None = None
    # 允许使用的显存比例，扣掉权重和激活后剩下的给KV cache（ModelRunner.get_num_blocks，第5批）
    gpu_memory_utilization: float = 0.9
    # 张量并行大小；__post_init__要求在1~8之间（单机8卡）
    # 注意：开启enable_dp_attention时，CoreManager在Config构造之后把它改成1，并把data_parallel_size乘以原TP
    #   （每个TP rank变成一个独立的DP rank，engine_core_mgr.py，第2批）
    tensor_parallel_size: int = 1
    # 【分支】DCP大小（必须整除TP）及其子配置，见第9批
    decode_context_parallel_size: int = 1
    dcp_config: DCPConfig = field(default_factory=DCPConfig)
    # 【分支】流水线并行、prefill上下文并行的大小，见第9批
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    # True时不录制CUDA Graph（EngineCore据此跳过capture_cudagraph）
    enforce_eager: bool = False
    # Number of vocabulary positions that carry a real token. A checkpoint
    # whose embedding matrix is padded up to a friendlier width -- Qwen3 rounds
    # 151665 up to 151936 -- leaves the tail rows holding whatever the padding
    # was initialised to, which is neither zero nor -inf, so sampling can land
    # on an id the tokenizer cannot decode. Take it from the tokenizer, not
    # from the model card. 0 means "not padded", which is the right answer for
    # every model whose embedding matrix matches its tokenizer.
    # 【译】真正对应某个token的词表位置数。embedding矩阵被填充到更友好宽度的checkpoint——Qwen3把151665
    #   向上取整到151936——尾部那些行里是填充时初始化的任意值，既不是0也不是-inf，所以采样可能落到一个
    #   tokenizer无法解码的id上。这个值要从tokenizer取，而不是从模型卡上取。0表示"没有填充"，
    #   对embedding矩阵与tokenizer一致的所有模型这都是正确答案。
    # 【分支】目前只有RL rollout的RLHFModelRunner读取它（第10批），以kwargs传入
    true_vocab_size: int = 0
    # Sleep (`release_memory`) normally frees the weights and the KV pool and
    # recaptures the decode CUDA graphs on wake. Set this to keep both
    # allocated instead, so the addresses the graphs captured stay valid and
    # nothing is recaptured -- recapture is what faults under
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
    #
    # The cost is that sleep no longer returns that memory to the allocator, so
    # a colocated trainer that sleeps the rollout engine to get its memory back
    # will not get it back. That is why it is opt-in. No effect under
    # `enforce_eager`, where there are no graphs to keep valid.
    # 【译】休眠（release_memory）通常会释放权重和KV pool，唤醒时重新捕获decode的CUDA graph。设置它则改为
    #   两者都保持分配，这样graph捕获时记下的地址仍然有效，什么都不用重新捕获——在
    #   PYTORCH_CUDA_ALLOC_CONF=expandable_segments 下，正是重新捕获会出错。
    #
    #   代价是休眠不再把这些显存还给分配器，所以一个同机训练进程想通过让rollout引擎休眠来拿回显存，
    #   就拿不回来了。这就是它需要手动开启的原因。在 enforce_eager 下没有效果，那时没有需要保持有效的graph。
    # 【分支】RL rollout专用（atom/rollout/memory_manager.py），见第10批
    sleep_keeps_memory_resident: bool = False
    # ---- 派生字段：field(init=False)表示不出现在__init__的参数里，由__post_init__赋值 ----
    # 模型结构配置（层数、头数、hidden size等），由get_hf_config读取并经过各种归一化
    hf_config: PretrainedConfig = field(init=False)
    # generation_config.json的内容；文件不存在时为None
    generation_config: GenerationConfig = field(init=False)
    # DP拓扑与通信地址（见上面的ParallelConfig）
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # 开始/结束符的token id：-1表示未设置；Config构造后由LLMEngine用tokenizer的值覆盖，调度器读取
    bos_token_id: int = -1
    eos_token_id: int = -1
    # 其他停止token：__post_init__从generation_config的eos_token_id填入，LLMEngine再从中去掉eos_token_id
    #   例：generation_config的eos_token_id=[A, B]，tokenizer.eos_token_id=A -> eos_token_id=A，stop_token_ids=[B]
    stop_token_ids: list[int] = field(default_factory=list)
    # 每个KV block的token数（命令行叫--block-size，_get_engine_kwargs改名）
    #   注意：__post_init__会为部分模型强制覆盖它（V4=256、V4.1=128、GLM-5.3=index_kpool x 16），
    #   调度器、BlockManager、ModelRunner读到的都是覆盖后的值
    kv_cache_block_size: int = 16
    # KV block总数：-1表示未知。Config里从不计算它：ModelRunner.get_num_blocks按显存算出后，
    #   EngineCore回填（engine_core.py），之后BlockManager按它建block池（第4、5批）
    # 对比nano-vllm：nano-vllm里是ModelRunner直接写config.num_kvcache_blocks（同一进程内）；ATOM的ModelRunner
    #   在子进程里，写不到EngineCore那份Config，所以要通过RPC把数字传回来
    num_kvcache_blocks: int = -1
    # KV cache的数据类型（bf16/fp8）
    # 适用场景（命令行choices只有bf16、fp8；每token的KV字节数算法见arg_utils.py同名字段的【科普】）：
    #   取值   每token KV（32层、8个kv头、head_dim 128）  什么时候选                         代价/限制
    #   bf16   128KB                                    默认；精度最敏感的评测；              KV容量只有fp8的一半
    #                                                   DeepSeek-V4.1 DSpark与Qwen QSA强制要求
    #                                                   （deepseek_v41/config.py、qwen4_exp_attn.py会报错）
    #   fp8    64KB                                     长上下文、高并发、显存紧张；CLAUDE.md   写入时量化、读出时乘scale，有轻微精度损失；
    #                                                   和多数recipe的启动示例都用它            fp8格式随架构不同：MI300(gfx942)是e4m3fnuz、
    #                                                                                         MI350(gfx950)是e4m3fn（mooncake_store/keys.py）
    # 选fp8之后，index_cache_dtype不指定时也会跟着变成fp8（V4/V4.1另有默认值，见__post_init__第13步）
    kv_cache_dtype: str = "bf16"
    # 【分支】DSA索引器缓存的数据类型；None时__post_init__按模型决定默认值，见第7、8批
    index_cache_dtype: str | None = None
    # 前缀缓存、chunked prefill的开关（BlockManager、调度器读取，第3、4批）
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # 是否周期性打印引擎状态行；是否记录每请求的上下文指标
    enable_log_stats: bool = True
    # Seconds between engine-status lines. Validated > 0 by EngineStats.
    # 【译】引擎状态行之间的秒数。由EngineStats校验 > 0。
    throughput_log_interval: float = 10.0
    # Requests in the sliding window behind the status line's prefix-cache hit
    # rate. Validated > 0 by EngineStats.
    # 【译】状态行中前缀缓存命中率所用滑动窗口的请求数。由EngineStats校验 > 0。
    cache_hit_rate_window: int = 1000
    # 【分支】PP动态分块prefill，见第9批；__post_init__里的_resolve_dynamic_chunking会在不适用时关掉它
    enable_dynamic_chunking: bool = False
    # None reads ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR once the feature is on; see
    # `_resolve_dynamic_chunking`.
    # 【译】None表示功能开启后读取 ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR；见 _resolve_dynamic_chunking。
    dynamic_chunking_smooth_factor: float | None = None
    # Floor for solved chunks. Every extra chunk re-pays the cached-prefix
    # rebuild, so the floor is what bounds that amplification.
    # 【译】求解出的chunk大小的下限。每多一个chunk就要再付一次重建已缓存前缀的代价，
    #   所以这个下限就是限制这种放大的手段。
    dynamic_chunking_min_chunk_size: int = 4096
    # 引擎内部端口：ModelRunner把它写进环境变量MASTER_PORT（不是HTTP端口）；torch.distributed主进程组的集合点
    #   其实是parallel_config.data_parallel_master_ip / data_parallel_base_port，见arg_utils.py EngineArgs.port的说明
    port: int = 8006
    # torch profiler的trace输出目录；__post_init__在非插件模式下会自动创建它
    # 注意：default_factory在构造时读环境变量ATOM_TORCH_PROFILER_DIR，但只在调用方没传这个参数时生效；
    #   经EngineArgs启动时总会显式传入--torch-profiler-dir的值（默认None），此时环境变量不起作用
    torch_profiler_dir: str | None = field(
        default_factory=lambda: envs.ATOM_TORCH_PROFILER_DIR
    )
    # torch.compile与CUDA Graph的子配置（见上面的CompilationConfig）
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    # 量化配置，__post_init__构造（见上面的QuantizationConfig）
    quant_config: QuantizationConfig = field(init=False)
    # CoreManager是否用asyncio.Queue异步输出结果（第2批）；atom/里没有地方把它设成True，插件显式传False
    asyncio_mode: bool = False
    # 在计算图中插入graph_marker节点，方便profile定位
    # 读取方：ModelRunner.__init__ 据此打开graph_marker（set_graph_marker_enabled），各注意力/V4 kernel包装
    #   在trace里打上名字，抓到的profiler trace里能按层、按算子分段
    mark_trace: bool = False
    # 用假权重代替读取checkpoint（empty/zero/xavier），只测性能时用
    # 读取方：model_loader/loader.py 跳过读文件、按方式填充参数；moe.py 等处在dummy模式下跳过依赖真实权重的检查
    load_dummy: str | None = None
    # 【分支】专家并行；用均匀的假路由测负载完全均衡时的性能上限，见第7批
    # enable_expert_parallel 读取方：moe.py（专家按EP切分而不是每个专家按TP切）、ModelRunner、LLMEngine、插件注册
    enable_expert_parallel: bool = False
    # fake_eplb 读取方：moe.py / comm_fused_moe.py（用均匀假路由）、CoreManager、simulated_tp.py（见tp_world_size）
    fake_eplb: bool = False
    # Width the MoE shards experts for when DP-attention simulates a deployment
    # wider than the box (set by CoreManager); 0 = not simulating.
    # `parallel_config.data_parallel_size` stays the real rank count, since it
    # sizes the process group and the token collectives.
    # 【译】DP-attention模拟一个比本机更宽的部署时，MoE按这个宽度切分专家（由CoreManager设置）；0 = 不模拟。
    #   parallel_config.data_parallel_size 仍是真实的rank数，因为它决定进程组和token集合通信的大小。
    # 读取方：只有 moe.py（专家切分宽度）；写入方：CoreManager（engine_core_mgr.py）
    dp_logical_size: int = 0
    # ModelRunner把它写进环境变量MASTER_ADDR；与上面的port同理，它不是torch.distributed主进程组的集合地址
    master_addr: str = "127.0.0.1"
    # 【分支】DP attention：注意力按DP切分、MoE在DP rank间协同，见第7、9批
    # 用途：MLA这类KV头很少的模型，注意力做TP会让每张卡重复存同一份KV；改成每张卡独立处理不同请求（DP），
    #   MoE仍在全部卡上协同。读取方：CoreManager（把TP改写成DP）、LLMEngine、ModelRunner、
    #   moe.py / topK.py / fused_moe（MoE前后要在DP rank间gather/scatter token）
    enable_dp_attention: bool = False
    # DP request-routing strategy used by CoreManager to pick an engine rank:
    # "round_robin" | "least_requests" (default) | "least_tokens". Only has an
    # effect when more than one DP engine rank is launched. See
    # atom/model_engine/engine_core_mgr.py:_select_dp_rank_locked. The literal
    # default must stay in sync with engine_core_mgr.DP_LB_DEFAULT (config.py
    # cannot import it without a cycle: engine_core_mgr imports Config).
    # 【译】CoreManager选择engine rank时使用的DP请求路由策略："round_robin" | "least_requests"（默认）|
    #   "least_tokens"。只有启动了多于一个DP engine rank时才有效果。见
    #   atom/model_engine/engine_core_mgr.py:_select_dp_rank_locked。这里的字面默认值必须与
    #   engine_core_mgr.DP_LB_DEFAULT 保持一致（config.py不能导入它，否则会循环导入：engine_core_mgr导入了Config）。
    dp_load_balance: str = "least_requests"
    # MoE expert-parallel layout policy. When True, MoE EP computes ranks in the
    # flattened DP x TP device space (and shared-expert fusion is disabled,
    # because the fused shared expert assumes the per-DP MoE layout). The vLLM
    # plugin sets this when EP is enabled; native ATOM and other plugins use the
    # per-DP MoE layout and leave it False. Set by the frontend in
    # atom/plugin/config.py, not queried via is_vllm() at the call site.
    # 【译】MoE专家并行的布局策略。为True时，MoE EP在展平的 DP x TP 设备空间里计算rank（并且关闭共享专家融合，
    #   因为融合的共享专家假定的是按DP划分的MoE布局）。vLLM插件在开启EP时设置它；原生ATOM和其他插件
    #   使用按DP划分的MoE布局，保持False。它由前端在 atom/plugin/config.py 中设置，而不是在调用处用 is_vllm() 查询。
    moe_ep_flatten_tp_across_dp: bool = False
    # 模型权重的数据类型：__post_init__取hf_config.dtype，没有则bf16
    # 读取方：未量化层的参数dtype、激活dtype（linear.py、moe.py、各注意力backend分配buffer时）
    torch_dtype: torch.dtype = field(init=False)
    # 【分支】投机解码子配置，不开启时为None，见第7批
    speculative_config: SpeculativeConfig | None = None
    # 【分支】KV传输（PD分离/offload）配置，见第9批
    #   命令行传进来的是JSON字符串，__post_init__解析成dict（LLMEngine构造完Config后又会从kwargs重新解析一次）
    # 读取方：kv_transfer/disaggregation/factory.py 据此选connector（mooncake / moriio / multi），
    #   各connector读其中的角色（producer/consumer）、地址等；部分注意力backend据此决定KV布局是否要兼容传输
    kv_transfer_config: dict = field(default_factory=dict)
    # 【分支】KV事件发布，只由环境变量配置（见上面的KVEventsConfig）
    kv_events_config: KVEventsConfig = field(default_factory=KVEventsConfig.from_env)
    # DSpark runtime knobs. Built once in the parent from --dspark-config (see
    # EngineArgs) and pickled into every worker. Read sites use `config.dspark.*`
    # (no os.environ lookups). Defaults to all-off.
    # 【译】DSpark运行时开关。在父进程中由 --dspark-config 构建一次（见EngineArgs），并pickle进每个worker。
    #   读取处使用 config.dspark.*（不读os.environ）。默认全部关闭。
    dspark: DSparkConfig = field(default_factory=DSparkConfig)

    # 【分支】以下到eplb_config是TBO与MoE通信相关，见第7、9批
    # TBO（Two-Batch Overlap）：prefill是否开启、decode是否也开启（--enable-tbo [prefill|all]拆出来的两个布尔值）
    enable_tbo: bool = False
    enable_tbo_decode: bool = False
    # MoRI all-to-all是否用低延迟模式（--all2all-backend low-latency时为True）
    # 读取方：moe.py 创建MoRI dispatch/combine时选择模式（低延迟模式适合decode的小batch）
    enable_low_latency: bool = False
    # Routed MoE transport selection. ``auto`` preserves the historical
    # behavior (MoRI when importable, otherwise gather/scatter). ``rccl`` uses
    # a graph-safe pre-routed collective for uniform decode, variable-size
    # gather/scatter for matching DP/EP groups, and dynamic routed all-to-all
    # for other prefill/mixed topologies.
    # 【译】路由MoE的传输方式选择。auto 保持历史行为（能导入MoRI就用MoRI，否则用gather/scatter）。
    #   rccl 对uniform decode使用一种可录制进graph的预路由集合通信，对DP/EP组匹配的情况使用变长的
    #   gather/scatter，对其他prefill/混合拓扑使用动态路由的all-to-all。
    # 读取方：moe.py（建哪种prepare/finalize传输）、topK.py（RCCL路径下共享专家能否融合）；
    #   __post_init__第1步会规范化并校验它
    moe_all2all_backend: str = "auto"
    # Post-routing routed-MoE implementation. This is deliberately separate
    # from all2all backend/mode: Mega owns dispatch, both GEMMs, and combine.
    # 【译】路由之后的路由MoE实现。它有意与all2all的后端/模式分开：Mega自己负责dispatch、两次GEMM和combine。
    # 读取方：moe.py、fused_moe/comm_fused_moe.py；"mega"时用一个融合kernel包办dispatch→GEMM→combine
    moe_backend: str = "standard"
    # 每个GPU工作进程要实例化的runner类的完整路径（EngineCore交给AsyncIOProcManager按名字导入）
    #   默认ModelRunner；RapidServe时__post_init__换成RapidServeModelRunner；RL rollout换成RLHFModelRunner
    runner_qualname: str = "atom.model_engine.model_runner.ModelRunner"
    # EPLB master switch + sub-config
    # 【译】EPLB总开关 + 子配置
    # eplb_enable 读取方：moe.py（开EP时才生效：为冗余专家多分配槽位，并挂上负载统计）、deepseek_v41/config.py
    eplb_enable: bool = False
    eplb_config: EPLBConfig = field(default_factory=EPLBConfig)

    # only use for plugin mode
    # 【译】只在插件模式下使用
    # 【分支】插件模式：vLLM/SGLang等框架的配置被包装在这里，见第10批
    # 原生模式下为None；插件模式下由 plugin/config.py 生成，Config.__post_init__ 从中取vLLM的 ignored_layers 等
    plugin_config: PluginConfig | None = None
    # only for quark_online_quantization
    # 【译】只用于quark在线量化
    # 解释：现在它是通用的--online_quant_config（ptpc_fp8/mxfp4），传给QuantizationConfig
    online_quant_config: dict | None = None
    # 加载HF config后用这个dict覆盖其字段（--hf-overrides），在__post_init__里应用
    # 用途：不改checkpoint就试验模型结构参数（例如打开MiniMax-M3的 use_index_cache）；
    #   KV offload的 mooncake_store/keys.py 也把它编进缓存key，因为它可能改变KV的形状或含义
    hf_overrides: dict[str, Any] | None = None

    # Intra-GPU prefill/decode disaggregation
    # 【译】单GPU内的prefill/decode分离
    # 【分支】RapidServe：同一张卡上起prefill、decode两个进程，用不同的stream（可选按CU划分），
    #   下面的地址都由CoreManager（DisaggCoreManager）填写，见第5、9批
    # 读取方：EngineCore（建PUSH/PULL socket）、ModelRunner / RapidServeModelRunner（共享权重和KV的IPC握手）、
    #   调度器（disagg_cu_shm_name：交换两边batch大小以动态分配CU）
    enable_rapidserve: bool = False
    # ZMQ IPC address: decode PUSH → prefill PULL (BlockAssignment messages)
    # 【译】ZMQ IPC地址：decode PUSH → prefill PULL（BlockAssignment消息）
    disagg_d2p_addr: str = ""
    # ZMQ IPC address: prefill PUSH → decode PULL (PrefillDone messages)
    # 【译】ZMQ IPC地址：prefill PUSH → decode PULL（PrefillDone消息）
    disagg_p2d_addr: str = ""
    # Bootstrap round 1: prefill PUSH → decode PULL (weight IPC handles)
    # 【译】启动握手第1轮：prefill PUSH → decode PULL（权重的IPC句柄）
    disagg_weight_ipc_addr: str = ""
    # Bootstrap round 1 ACK: decode PUSH → prefill PULL (signals weights freed)
    # 【译】启动握手第1轮的确认：decode PUSH → prefill PULL（表示权重已释放）
    disagg_weight_ack_addr: str = ""
    # Bootstrap round 2: prefill PUSH → decode PULL (kvcache_args + num_blocks)
    # 【译】启动握手第2轮：prefill PUSH → decode PULL（kvcache_args + num_blocks）
    disagg_kvcache_ipc_addr: str = ""
    # True for the decode process in disagg mode: skip GPU weight/kvcache allocation.
    # 【译】分离模式下decode进程为True：跳过GPU上权重/KV cache的分配。
    disagg_is_decode: bool = False
    # Name of the shared-memory region used for dynamic CU partitioning.
    # Both prefill and decode processes open this to exchange batch sizes.
    # 【译】用于动态CU划分的共享内存区域名。prefill和decode两个进程都打开它来交换batch大小。
    disagg_cu_shm_name: str = ""
    # Override max_num_seqs for the prefill process in disagg mode.
    # When None, prefill inherits the base max_num_seqs.
    # 【译】分离模式下覆盖prefill进程的max_num_seqs。为None时prefill沿用基础的max_num_seqs。
    disagg_prefill_max_num_seqs: int | None = None
    # When True (and enable_rapidserve=True), use CU-masked streams + shm
    # coordination between prefill and decode. When False (default),
    # use plain separate streams with no CU masking.
    # 【译】为True（且enable_rapidserve=True）时，prefill和decode之间使用按CU屏蔽的stream + 共享内存协调。
    #   为False（默认）时，使用普通的独立stream，不做CU屏蔽。
    disagg_constrained: bool = False

    # 实际启动的TP工作进程数；正常情况下就等于tensor_parallel_size
    # 读取方：EngineCore（启动多少个ModelRunner进程 = tp_world_size x pcp）、ModelRunner、simulated_tp.py
    # 例：--fake-eplb -tp 8，但本机只有4张可见GPU -> tp_world_size=4（模拟8卡部署的前4张卡）
    @property
    def tp_world_size(self) -> int:
        """Number of TP worker processes actually launched.

        `tensor_parallel_size` is the *logical* width -- how many shards every
        weight is cut into. Under `--fake-eplb` on a box with fewer visible
        devices than `-tp`, only the first `tp_world_size` of those shards get
        a process, reproducing the first N devices of the larger deployment.
        See `atom/distributed/simulated_tp.py`.

        Gated on `fake_eplb` because such a run's output is garbage anyway;
        without it, an oversized `-tp` keeps raising in ModelRunner instead of
        silently running smaller. A property, not a field: `enable_dp_attention`
        rewrites `tensor_parallel_size` after Config is built.
        """
        # 【译】实际启动的TP工作进程数。
        #
        #   tensor_parallel_size 是"逻辑"宽度——每个权重被切成多少份。在 --fake-eplb 下、本机可见设备数少于
        #   -tp 时，只有前 tp_world_size 份会得到进程，以此复现更大部署中的前N个设备。
        #   见 atom/distributed/simulated_tp.py。
        #
        #   以 fake_eplb 为条件，是因为这种运行的输出本来就是无意义的；没有它时，过大的 -tp 仍会在ModelRunner里
        #   报错，而不是悄悄地以更小的规模运行。写成属性而不是字段：因为 enable_dp_attention 会在Config构建之后
        #   改写 tensor_parallel_size。
        tp = self.tensor_parallel_size
        if not self.fake_eplb:
            return tp
        # Does not create a CUDA context, so it is safe in the parent process.
        # 【译】不会创建CUDA上下文，所以在父进程里调用是安全的。
        visible = torch.cuda.device_count()
        return visible if 0 < visible < tp else tp

    # 声明的CUDA Graph捕获尺寸列表（按batch size，即decode时的序列数）
    # 读取方：ModelRunner.capture_cudagraph，拿到后再删掉调度器永远凑不出的尺寸（超过min(max_num_seqs,
    #   max_num_batched_tokens // (mtp_k+1))的那些），见第5批
    # 取值规则：
    #   1. --cudagraph-capture-sizes给了就用它（命令行默认[1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]）
    #   2. 否则看cuda_graph_sizes：只有一个值N时生成 [1..8] + [16, 32, ..., N]
    #      例：N=512（默认）-> [1,2,3,4,5,6,7,8,16,32,48,...,496,512]，共8+32=40个尺寸
    #   3. cuda_graph_sizes有多个值时原样使用
    # 运行时batch size向上取整到最近的捕获尺寸并回放；比最大尺寸还大的batch不走graph
    @property
    def capture_sizes(self) -> list[int]:
        """The declared CUDAGraph capture ladder, in batch sizes.

        Declared, not schedulable -- `ModelRunner` drops what its own token
        budget can never produce, a bound needing the drafter's resolved
        `mtp_k` that config cannot see. Returns a copy: the runner sorts and
        filters in place.
        """
        # 【译】声明的CUDAGraph捕获阶梯，以batch size计。
        #
        #   只是"声明的"，不代表都能被调度到——ModelRunner会删掉它自己的token预算永远凑不出来的尺寸，
        #   这个上限需要草稿器解析出的 mtp_k，而config看不到它。返回一个副本：runner会就地排序和过滤。
        declared = self.compilation_config.cudagraph_capture_sizes
        if declared:
            # list(...)：复制一份新列表，ModelRunner就地sort/过滤时不会改到配置里的原列表
            return list(declared)
        sizes = self.compilation_config.cuda_graph_sizes
        if len(sizes) == 1:
            return list(range(1, 9)) + list(range(16, sizes[0] + 1, 16))
        return list(sizes)

    def __post_init__(self):
        # 流程图（编号与下面各段"第N步"标题一致；✗ = raise/assert，启动即终止；"降级"= 只打warning并关掉功能）：
        #   ┌──────────────── 早期规范化与校验（还不需要HF config）────────────────┐
        #   │ 1  moe_all2all_backend / moe_backend 小写化、查取值      ──非法──> ✗    │
        #   │    rccl + TBO -> 降级关TBO；mega 但没开EP / mega + rccl  ─────────> ✗    │
        #   │ 2  compilation_config / eplb_config / dcp_config：dict -> 对象          │
        #   │ 3  强制接受率 + DSpark置信度调度 ─────────────────────────────> ✗      │
        #   │    RapidServe：runner_qualname换成RapidServeModelRunner                │
        #   │ 4  1 <= tp <= 8；dcp>1时 tp % dcp == 0、投机解码+DCP必须gfx950 ─否─> ✗  │
        #   └──────────────────────────────────────────────────────────────────────────┘
        #                                     ▼
        #   ┌──────────────── 读模型自己的配置 ──────────────────────────────────────┐
        #   │ 5  hf_config = get_hf_config(model) -> hf_overrides -> 各模型字段归一化   │
        #   │ 6  generation_config -> stop_token_ids；quant_config = QuantizationConfig │
        #   │ 7  max_model_len = min(用户值, max_position_embeddings)，None时取后者      │
        #   │    long_prefill_token_threshold 不在 [block_size, max_model_len] ──> ✗    │
        #   └──────────────────────────────────────────────────────────────────────────┘
        #                                     ▼
        #   8  (原生 或 vLLM插件) 且 level == 3 ?
        #        是 -> splitting_ops = 4个注意力算子；cudagraph_mode为None时：V4.1 -> FULL，其他 -> PIECEWISE
        #        否 -> 跳过（cudagraph_mode保持传进来的值）
        #                                     ▼
        #   9  torch_dtype = hf_config.dtype 或 bf16；kv_transfer_config字符串 -> dict；有KV传输时给cache几何拍快照
        #                                     ▼
        #   10 _resolve_dynamic_chunking()：条件不满足时校验报错或降级（见第9批）
        #   11 有投机解码？ num_speculative_tokens > 上限 ──> ✗
        #   12 V4 -> block_size = 256；GLM-5.3 -> 不支持的特性 ✗，index_kpool > 1时block_size由它算出
        #   13 indexer_dcp_only门控（不满足 -> 降级）；index_cache_dtype默认值；
        #      V4.1 -> block_size = 128 + validate_runtime_config；V4 -> state_replay_tokens
        #
        # 走一遍（数字是假设的）：Qwen3-8B，命令 -tp 2 --kv_cache_dtype fp8 --max-model-len 65536，
        #   且config.json里max_position_embeddings = 40960
        #   1~3 全部默认值，通过；4 tp=2 ∈ [1, 8]，dcp=1，跳过DCP检查
        #   5 读到hf_config（model_type=qwen3）；6 没有quantization_config -> 量化表只有一条"不量化、bf16"
        #   7 max_model_len = min(65536, 40960) = 40960（静默截断，不报错）
        #   8 原生 + level 3 -> 设好splitting_ops；cudagraph_mode已是命令行给的FULL，不改
        #   9 torch_dtype取hf_config.dtype（通常bf16）；kv_transfer_config是"{}"，解析成空dict，不拍快照
        #   10~12 没开PP动态分块、没开投机解码、不是V4/GLM-5.3，全部跳过
        #   13 index_cache_dtype为None且不是V4/V4.1 -> 等于kv_cache_dtype = "fp8"
        # ---------------- 第1步【分支】MoE后端字段的规范化与校验，见第7批 ----------------
        # 统一成小写、去空格；None/空字符串当作"auto"
        self.moe_all2all_backend = (
            str(self.moe_all2all_backend or "auto").strip().lower()
        )
        if self.moe_all2all_backend not in ("auto", "mori", "rccl", "none"):
            raise ValueError(
                "moe_all2all_backend must be one of "
                "{'auto', 'mori', 'rccl', 'none'}, "
                f"got {self.moe_all2all_backend!r}"
            )
        # RCCL路由MoE目前是同步实现，和TBO（通信与计算重叠）不兼容：降级关掉TBO
        if self.moe_all2all_backend == "rccl" and self.enable_tbo:
            logger.warning(
                "Disabling TBO for the experimental RCCL routed MoE backend; "
                "the first implementation is synchronous."
            )
            self.enable_tbo = False
            self.enable_tbo_decode = False

        # mega后端必须开EP，且自带MoRI传输，不能和rccl一起用
        self.moe_backend = self.moe_backend.strip().lower()
        if self.moe_backend not in ("standard", "mega"):
            raise ValueError(
                "moe_backend must be one of {'standard', 'mega'}, "
                f"got {self.moe_backend!r}"
            )
        if self.moe_backend == "mega" and not self.enable_expert_parallel:
            raise ValueError(
                "moe_backend='mega' requires expert parallelism; "
                "pass --enable-expert-parallel."
            )
        if self.moe_backend == "mega" and self.moe_all2all_backend == "rccl":
            raise ValueError(
                "moe_backend='mega' owns its MoRI transport and cannot use the "
                "experimental RCCL prepare/finalize backend"
            )

        # ---------------- 第2步：子配置dict→对象，并重新校验 ----------------
        # 直接用Python构造Config的调用方（测试、插件）可能传dict；统一转成对应的dataclass
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        elif isinstance(self.eplb_config, EPLBConfig):
            # Normalize/validate even when constructed programmatically.
            # 【译】即使是以编程方式构造的，也要规范化/校验一遍。
            # 解释：obj.__dict__是实例的字段字典；用它重新构造一次，就会再跑一遍EPLBConfig.__post_init__的校验
            #   （直接改过字段的对象，构造时的校验已经过期了）
            self.eplb_config = EPLBConfig(**self.eplb_config.__dict__)
        else:
            raise TypeError("eplb_config must be EPLBConfig or dict")
        if isinstance(self.dcp_config, dict):
            self.dcp_config = DCPConfig(**self.dcp_config)
        elif isinstance(self.dcp_config, DCPConfig):
            # Normalize/validate even when constructed programmatically.
            # 【译】即使是以编程方式构造的，也要规范化/校验一遍。
            self.dcp_config = DCPConfig(**self.dcp_config.__dict__)
        else:
            raise TypeError("dcp_config must be DCPConfig or dict")
        # （以下是被注释掉的旧代码：nano-vllm有同样的断言；注释掉之后model也可以是HF仓库名，
        #   EngineArgs的默认值Qwen/Qwen3-0.6B就是）
        # assert os.path.isdir(self.model)

        # ---------------- 第3步【分支】两个互斥/联动检查，见第7批、第5批 ----------------
        # The forced-acceptance schedule spends its whole budget on the first
        # ceil(length - 1) positions, and the sampler can only accept what was
        # actually drafted. The DSpark confidence scheduler hands each request
        # its own verify length ell_r, so whenever ell_r falls below that many
        # positions the run quietly lands under the acceptance length it was
        # asked to reproduce (measured at length 3.78 over 7 positions: exact
        # while ell_r >= 3, 3.39 at ell_r = 2, 2.89 at ell_r = 1). ell_r is
        # chosen at runtime from the confidence head, so there is no upfront
        # check that would catch it -- and a benchmark reporting a number it
        # never hit is worse than one that refuses to start.
        # 【译】强制接受的调度表把全部预算花在前 ceil(length - 1) 个位置上，而采样器只能接受真正起草出来的token。
        #   DSpark置信度调度器给每个请求分配它自己的验证长度ell_r，所以只要ell_r小于那么多个位置，这次运行就会
        #   悄悄落到要求复现的接受长度之下（实测：长度3.78、7个位置时，ell_r >= 3 时精确，ell_r = 2 时为3.39，
        #   ell_r = 1 时为2.89）。ell_r是运行时由置信度头选出的，没有能提前发现它的检查——而一个报告了自己
        #   从未达到的数字的benchmark，比拒绝启动的更糟。
        if (
            self.speculative_config is not None
            and self.speculative_config.synthetic_acceptance_rates is not None
            and self.dspark.confidence_schedule
        ):
            raise ValueError(
                "Forced speculative acceptance (--spec-decode-acceptance-length "
                "/ --spec-decode-acceptance-rate) cannot be combined with the "
                "DSpark confidence scheduler (--dspark-config "
                "'{\"confidence_schedule\": true}'): it sizes each request's "
                "verify length at runtime, and a short one caps acceptance below "
                "the requested length with no way to detect it upfront. Drop "
                "confidence_schedule (and ragged, which needs it) for "
                "forced-acceptance runs."
            )

        # RapidServe (intra-GPU prefill/decode disagg) needs a specialized
        # runner in both the prefill and decode processes. Select it unless the
        # user explicitly overrode runner_qualname.
        # 【译】RapidServe（单GPU内的prefill/decode分离）在prefill和decode两个进程里都需要专门的runner。
        #   除非用户显式覆盖了 runner_qualname，否则选用它。
        if (
            self.enable_rapidserve
            and self.runner_qualname == "atom.model_engine.model_runner.ModelRunner"
        ):
            self.runner_qualname = (
                "atom.model_engine.model_runner.RapidServeModelRunner"
            )

        # ---------------- 第4步 ★ 并行度检查（DCP部分是【分支】，见第9批） ----------------
        # TP最多8：同nano-vllm，只支持单机内的张量并行
        assert 1 <= self.tensor_parallel_size <= 8
        if self.decode_context_parallel_size > 1:
            # DCP复用TP的GPU（不增加卡数），所以dcp必须整除tp，例如tp=8时dcp可取2、4、8
            assert self.tensor_parallel_size % self.decode_context_parallel_size == 0, (
                f"tp_size ({self.tensor_parallel_size}) must be divisible by "
                f"dcp_size ({self.decode_context_parallel_size})"
            )
            # Spec-decode + DCP arch gating. Any speculative method (mtp /
            # eagle3 / dspark) runs a q>1 verify pass, and DCP decode with q>1
            # uses the round-robin CP (cprr) MLA kernel, which is persistent-only
            # and ships only on gfx950; on gfx942 the non-persistent fallback
            # ignores the cprr masking and silently produces WRONG output.
            # 【译】投机解码 + DCP 的架构门控。任何投机方法（mtp / eagle3 / dspark）都会跑一次q>1的验证，
            #   而q>1的DCP decode使用轮转CP（cprr）的MLA kernel，它只有persistent版本、且只在gfx950上提供；
            #   在gfx942上，非persistent的后备实现会忽略cprr掩码，静默地产生"错误"的输出。
            # 解释：gfx950是MI350系列，gfx942是MI300系列；get_gfx()读当前GPU的架构名
            if self.speculative_config is not None:
                from aiter.jit.utils.chip_info import get_gfx

                gfx = get_gfx()
                assert gfx == "gfx950", (
                    f"Speculative decode + DCP is only supported on gfx950 (needs "
                    f"the persistent cprr MLA kernel); got {gfx}. Disable DCP or "
                    f"speculative decode on this GPU."
                )
                # TBO slices its ubatch work buffers by request, while the
                # sparse DCP indexer indexes them by query token. The two agree
                # only at one query per sequence.
                # 【译】TBO按请求切分它的ubatch工作缓冲区，而稀疏DCP索引器按query token索引它们。
                #   两者只有在每条序列一个query时才一致。
                assert not self.enable_tbo_decode, (
                    "Decode TBO (--enable-tbo all) combined with speculative "
                    "decode and DCP is unverified. Use --enable-tbo (prefill "
                    "only), or drop DCP or speculative decode."
                )
        # DCP KV-cache interleave granularity S. S=1 (default) = token-level
        # round-robin (unchanged). S>1 = block-level interleave; must divide the
        # KV block so each physical block holds an integer number of S-groups
        # (the (i//(S*W))*S + i%S local-index math relies on block_size % S == 0),
        # and only makes sense under DCP.
        # 【译】DCP KV cache的交错粒度S。S=1（默认）= 按token轮转（与原来相同）。S>1 = 按块交错；S必须整除
        #   KV block大小，这样每个物理block装整数个S组（本地下标公式 (i//(S*W))*S + i%S 依赖 block_size % S == 0），
        #   并且只在DCP下才有意义。
        # 例：S=1、W=2（dcp）：token 0,2,4...在rank0，1,3,5...在rank1；S=4：token 0-3在rank0，4-7在rank1，8-11在rank0……
        # 注意：这里检查的是"按模型覆盖之前"的kv_cache_block_size（V4/GLM-5.3的覆盖在后面第12步，V4.1在第13步）
        assert 1 <= self.dcp_config.interleave_size <= self.kv_cache_block_size, (
            f"dcp_config.interleave_size ({self.dcp_config.interleave_size}) must "
            f"be in [1, kv_cache_block_size={self.kv_cache_block_size}]"
        )
        if self.dcp_config.interleave_size > 1:
            assert self.kv_cache_block_size % self.dcp_config.interleave_size == 0, (
                f"kv_cache_block_size ({self.kv_cache_block_size}) must be divisible "
                f"by dcp_config.interleave_size ({self.dcp_config.interleave_size})"
            )
            assert self.decode_context_parallel_size > 1, (
                "dcp_config.interleave_size > 1 only applies under DCP "
                f"(decode_context_parallel_size={self.decode_context_parallel_size})"
            )
            assert self.speculative_config is None, (
                "dcp_config.interleave_size > 1 (block-level DCP interleave) is "
                "incompatible with speculative decode (MTP/eagle/dspark): the q>1 "
                "verify cprr MLA kernel assumes token-level interleave. Use "
                "dcp_config.interleave_size=1 with speculative decode, or disable "
                "speculative decode for block-level interleave."
            )

        # DCP Query Replication (QREP) gating: turn the flag OFF (warn, not
        # error) for combinations that cannot use it, so it can default to on
        # without breaking mixed runs.
        # 【译】DCP query复制（QREP）的门控：对不能使用它的组合，把开关关掉（警告，而不是报错），
        #   这样它可以默认开启，又不会让各种混合配置的运行出错。
        # 例：没开DCP（dcp=1）时enable_query_replication默认是True，这里会打一条警告并把它关掉
        if self.dcp_config.enable_query_replication:
            qrep_off = qrep_unsupported_reason(
                self.decode_context_parallel_size,
                envs.ATOM_USE_TRITON_MXFP4_BMM,
            )
            if qrep_off is not None:
                logger.warning(
                    "dcp_config.enable_query_replication disabled: %s not "
                    "supported.",
                    qrep_off,
                )
                self.dcp_config.enable_query_replication = False
        assert 1 <= self.pipeline_parallel_size

        # ---------------- 第5步 ★ 读取HF config并归一化 ----------------
        # 读config.json（5条加载路径见get_hf_config上方的说明）
        # 对比nano-vllm：self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config = get_hf_config(
            self.model, trust_remote_code=self.trust_remote_code
        )
        # PP按层切分，每个stage至少要分到一层
        num_hidden_layers = getattr(self.hf_config, "num_hidden_layers", None)
        if num_hidden_layers is not None:
            assert num_hidden_layers >= self.pipeline_parallel_size, (
                f"num_hidden_layers ({num_hidden_layers}) must be >= "
                f"pipeline_parallel_size ({self.pipeline_parallel_size})"
            )
        # --hf-overrides：在所有归一化之前应用，所以覆盖的字段也会参与后面的归一化
        if self.hf_overrides:
            self.hf_config.update(self.hf_overrides)
            logger.info("Applied HF config overrides: %s", self.hf_overrides)
        # 【分支】MiniMax-M3的字段归一化（不是M3时直接返回），见第8批
        _normalize_minimax_m3_text_config(self.hf_config)
        # Multimodal config (full config with vision_config) for vision encoder init
        # 【译】多模态config（含vision_config的完整config），供视觉编码器初始化使用
        # get_hf_config的多模态分支把它挂在hf_config._multimodal_config上；不是多模态模型时为None
        self.multimodal_config = getattr(self.hf_config, "_multimodal_config", None)
        # 统一MoE字段名（n_routed_experts / n_shared_experts），不是MoE模型时基本什么都不做
        _normalize_moe_config_fields(self.hf_config, self.model)
        # transformers 5+ exposes rope_parameters; <5 often only rope_scaling + rope_theta.
        # Synthesize when missing or None so GPT-OSS YaRN (rope_type in rope_scaling) is preserved.
        # 【译】transformers 5+ 提供 rope_parameters；<5 时通常只有 rope_scaling + rope_theta。
        #   缺失或为None时合成一个，从而保留GPT-OSS的YaRN设置（rope_type写在rope_scaling里）。
        # 解释：ATOM的模型文件（如qwen3.py、llama.py）从hf_config.rope_parameters读取RoPE参数（第6批），
        #   这里为旧版transformers补齐这个字段
        # 例：rope_scaling={"type": "yarn", "factor": 32}，rope_theta=150000
        #   结果 rope_parameters={"type": "yarn", "factor": 32, "rope_theta": 150000, "rope_type": "yarn"}
        if getattr(self.hf_config, "rope_parameters", None) is None:
            # Compatible with transformers < 5
            # 【译】兼容 transformers < 5
            rope_params = getattr(self.hf_config, "rope_scaling", None) or {}
            rope_params = dict(rope_params)
            # rope_theta: GPT-OSS / LLaMA-style configs keep it on the root in <5
            # 【译】rope_theta：GPT-OSS / LLaMA风格的config在<5版本里把它放在根上
            rope_params["rope_theta"] = getattr(self.hf_config, "rope_theta", None)
            # rope_type: must NOT overwrite rope_scaling["rope_type"] (e.g. GPT-OSS YaRN).
            # transformers 4.x has no top-level rope_type; getattr(..., "default") was wrong.
            # 【译】rope_type：绝不能覆盖 rope_scaling["rope_type"]（例如GPT-OSS的YaRN）。
            #   transformers 4.x没有顶层的rope_type；原来写的 getattr(..., "default") 是错的。
            # 优先级：rope_scaling["rope_type"] > rope_scaling["type"]（旧写法）> 根上的rope_type > "default"
            if "rope_type" not in rope_params and "type" in rope_params:
                rope_params["rope_type"] = rope_params["type"]
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = getattr(
                    self.hf_config, "rope_type", "default"
                )
            self.hf_config.rope_parameters = rope_params
        # 谁读 rope_parameters：模型文件建RotaryEmbedding时（例如qwen3.py取出rope_theta、rope_type等传给get_rope）、
        #   ModelRunner（看有没有 mrope_section 判断是否多模态M-RoPE）、KV offload（rope设置进KV编码快照，见第9步）

        # ---------------- 第6步 ★ 停止token与量化配置 ----------------
        # generation_config.json里的eos_token_id可能是一个整数，也可能是列表；统一成列表放进stop_token_ids
        # := 是"海象运算符"（Python 3.8+）：在表达式里同时赋值和取值，这里把getattr的结果存进eos_ids再判断
        # 之后LLMEngine会从stop_token_ids里去掉tokenizer的eos_token_id（见eos_token_id字段的说明）
        # 最终读者：Scheduler.__init__ 存为 self.stop_token_ids，每步postprocess检查新生成的token是否在其中，
        #   在就结束这条序列（第3批）。为什么要读generation_config：很多chat模型有多个结束符
        #   （例如<|im_end|>和<|endoftext|>），只靠tokenizer.eos_token_id会停不下来
        # generation_config本身还被 openai/api_server.py 读取（/v1/messages、/v1/responses 取其中推荐的采样参数）
        self.generation_config = get_generation_config(self.model)
        if self.generation_config is not None and (
            (eos_ids := getattr(self.generation_config, "eos_token_id", None))
            is not None
        ):
            self.stop_token_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        # 构造量化配置（第6批）；此时层名还是checkpoint的命名，ModelRunner建模型前会调用remap_layer_name改写
        # 必须在第5步之后：多模态模型的 quantization_config 是get_hf_config从根config搬进文本config的，
        #   --hf-overrides 也可能改写它；完整的生命周期见 QuantizationConfig 类上方的说明
        self.quant_config = QuantizationConfig(
            self.hf_config,
            self.online_quant_config,
        )
        # In plugin mode, supplement exclude_layers with vLLM's ignored_layers when
        # the HF quant config didn't produce any exclusions (non-quark quant methods).
        # 【译】插件模式下，如果HF量化config没有产生任何排除项（非quark的量化方法），就用vLLM的
        #   ignored_layers 补充 exclude_layers。
        # 【分支】插件模式，见第10批
        # 为什么只在"空"时补：ATOM自己的解析器已经解析出排除项时以它为准；解析结果为空时
        #   （英文注释说的非quark方法），用vLLM从同一checkpoint解析出的 ignored_layers 兜底，
        #   避免lm_head等本该bf16的层被当成量化层
        if (
            self.plugin_config is not None
            and self.plugin_config.vllm_config is not None
            and len(self.quant_config.exclude_layers) == 0
        ):
            vllm_ignored = getattr(
                self.plugin_config.vllm_config.quant_config, "ignored_layers", []
            )
            self.quant_config.exclude_layers = list(vllm_ignored)

        # ---------------- 第7步 ★ max_model_len与几个相关校验 ----------------
        # 不能超过模型自身支持的最大位置数；没传时直接取它（HF config里也没有时按8192）
        # 例：--max-model-len 32768，模型max_position_embeddings=40960 -> 32768
        #     不传 -> 40960；传65536 -> 截断为40960
        # 对比nano-vllm：nano-vllm默认4096再取min；ATOM默认None表示"用模型的上限"
        # 谁读max_model_len：
        #   - Scheduler：拒绝prompt超过max_model_len的请求（block_table放不下，接纳时就拒绝，而不是跑到一半崩溃）
        #   - ModelRunner / 各注意力backend：按它预分配block_table（每条序列 max_model_len / block_size 列）、
        #     arange等buffer和CUDA Graph里的元数据buffer
        #   - 本函数后面GLM-5.3的稀疏开关校验
        #   所以它越大，常驻的元数据buffer越大；它不直接决定KV cache有多大（那由gpu_memory_utilization决定）
        hf_config_max_position_embeddings = getattr(
            self.hf_config, "max_position_embeddings", 8192
        )
        if self.max_model_len is None:
            self.max_model_len = hf_config_max_position_embeddings
        else:
            self.max_model_len = min(
                self.max_model_len, hf_config_max_position_embeddings
            )
        # （以下是被注释掉的旧代码：注释掉之后，单步token预算可以小于max_model_len，
        #   放不进一步的长prompt由chunked prefill分多步算完）
        # assert self.max_num_batched_tokens >= self.max_model_len
        # 每步单请求的prefill上限必须在[一个block, max_model_len]之间
        # 谁读：Scheduler 在chunked prefill时，把单条长prompt每步的chunk截到这个长度，
        #   防止一条超长prompt独占整步token预算、挤掉同一步里的其他prefill请求
        #   （decode不在同一步里：ATOM一步只做prefill或只做decode，见arg_utils.py文件上方总览表的注①）
        if self.long_prefill_token_threshold > 0:
            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    f"long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than max_model_len ({self.max_model_len})."
                )
            if self.long_prefill_token_threshold < self.kv_cache_block_size:
                raise ValueError(
                    f"long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) must be >= "
                    f"kv_cache_block_size ({self.kv_cache_block_size})."
                )
        # 非插件模式下，profiler目录不存在就创建（exist_ok=True：已存在不报错）
        # 谁用这个目录：ModelRunner在下面按rank建子目录写trace（start/stop_profile时）；RapidServe时
        #   CoreManager给prefill进程另配一个子目录。在Config构造时检查，路径不可用会在启动时就报错，
        #   而不是等到抓trace时才发现
        if not is_plugin_mode():
            if self.torch_profiler_dir is not None:
                os.makedirs(self.torch_profiler_dir, exist_ok=True)
            assert self.torch_profiler_dir is None or os.path.isdir(
                self.torch_profiler_dir
            ), f"torch_profiler_dir {self.torch_profiler_dir} is not a valid directory"

        # ---------------- 第8步 ★ 编译与CUDA Graph配置（第5批详解） ----------------
        # only for server mode or plugin mode(vllm)
        # for torch compile policy, plugin mode(vllm) uses the ATOM compile policy
        # for cuda graph capture, plugin mode(vllm) uses the vLLM's cuda graph capture policy
        # 【译】只用于服务模式或插件模式（vllm）
        #   torch compile的策略：插件模式（vllm）使用ATOM的编译策略
        #   cuda graph的捕获：插件模式（vllm）使用vLLM的cuda graph捕获策略
        # 条件：原生模式或vLLM插件，且level==3（PIECEWISE）。原生默认就是level 3，所以通常都会进来
        # 这里填好的 splitting_ops / cudagraph_mode / compile_sizes 由 utils/backends.py（VllmBackend切图、
        #   决定哪些尺寸提前编译）和 ModelRunner.capture_cudagraph（按什么模式录图）读取（第5批）
        # SGLang/RTP-LLM插件不进来：它们用宿主框架自己的编译和图捕获
        if (
            not is_plugin_mode()
            or (self.plugin_config is not None and self.plugin_config.is_vllm)
        ) and self.compilation_config.level == CompilationLevel.PIECEWISE:
            # 设置默认的切图算子（4个注意力算子）
            self.compilation_config.set_splitting_ops_for_v1()
            # Default to PIECEWISE, except V4.1 whose Engram stream fork/join
            # currently requires a FULL graph. Explicit choices are preserved.
            # splitting_ops/sizes still configure the level-3 backend.
            # 【译】默认用PIECEWISE，V4.1除外：它的Engram stream fork/join目前要求FULL graph。
            #   显式指定的选择会被保留。splitting_ops/sizes仍然会配置level 3的后端。
            # 注意：原生模式下命令行总会传--cudagraph-mode（默认FULL），所以这个None分支实际上只在
            #   插件模式或直接构造Config时生效
            if self.compilation_config.cudagraph_mode is None:
                self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL
                    if self.hf_config.model_type == "deepseek_v41_text"
                    else CUDAGraphMode.PIECEWISE
                )
            # compile_sizes里的"cudagraph_capture_sizes"展开成整数（默认compile_sizes=None，结果为[]）
            self.compilation_config.init_with_cudagraph_sizes()

        # ---------------- 第9步 ★ 权重dtype与KV传输配置 ----------------
        # 模型权重的数据类型：HF config里声明了dtype就用它，否则bf16
        # 注意它和QuantizationConfig.torch_dtype是分别从hf_config算出来的同一个值；
        #   这里的给ModelRunner用：建模型前设成torch默认dtype、hidden states等buffer的dtype，各层也通过它分配激活buffer
        self.torch_dtype = (
            self.hf_config.dtype
            if getattr(self.hf_config, "dtype", None) is not None
            else torch.bfloat16
        )

        # 【分支】--kv-transfer-config从命令行来时是JSON字符串，这里解析成dict，见第9批
        #   json.loads失败时再用ast.literal_eval试一次（兼容Python字面量写法，例如单引号的dict）
        # 注意：import写在函数内部，只在需要时才导入（json/ast都是标准库，开销很小）
        if hasattr(self, "kv_transfer_config") and isinstance(
            self.kv_transfer_config, str
        ):
            import json

            try:
                self.kv_transfer_config = json.loads(self.kv_transfer_config)
            except json.JSONDecodeError:
                import ast

                self.kv_transfer_config = ast.literal_eval(self.kv_transfer_config)
        # 【分支】KV offload：在Config发往各进程之前，给模型的cache几何参数拍一个快照，见第9批
        # 这两个快照挂在Config上（未声明的属性），读取方是 kv_transfer/offload/config.py（建页命名空间）和
        #   mooncake_store/keys.py（存储key）。命名空间对不上的后果是"永远不命中"，不报错，很难发现
        if self.kv_transfer_config:
            # The offload namespace hashes the model's cache geometry. A worker
            # normalises `hf_config` while it builds the model (Kimi-K3 derives
            # `head_dim`, `ModelRunner.get_num_blocks` fills a missing one) and
            # the scheduler never does, so hashing the live config gives each a
            # different namespace and every scheduler lookup misses. Snapshot it
            # here, before the config is shipped to either process.
            # 【译】offload的命名空间是对模型cache几何参数做hash得到的。worker在构建模型时会规范化 hf_config
            #   （Kimi-K3会推导 head_dim，ModelRunner.get_num_blocks 会补上缺失的那个），而调度器从不这样做，
            #   所以对"活的"config做hash会让两边得到不同的命名空间，调度器的每次查找都会落空。
            #   在这里、在config被发往任一进程之前，给它拍一个快照。
            from atom.kv_transfer.offload.config import (
                snapshot_kv_encoding,
                snapshot_page_hf_geometry,
            )

            self.offload_page_hf_geometry = snapshot_page_hf_geometry(self.hf_config)
            # Likewise how K and V are encoded (the RoPE settings above all):
            # llama.py adds a key to the live `rope_parameters` while a worker
            # builds the model.
            # 【译】同样地，也给K和V的编码方式拍快照（首先是RoPE设置）：worker构建模型时，llama.py会往
            #   活的 rope_parameters 里加一个key。
            self.offload_kv_encoding = snapshot_kv_encoding(self.hf_config)

        # ---------------- 第10步【分支】动态分块prefill的校验/降级，见第9批 ----------------
        # 放在这里是因为要用到前面已经确定的 pipeline_parallel_size、enable_chunked_prefill、kv_transfer_config
        self._resolve_dynamic_chunking()

        # ---------------- 第11步【分支】投机解码：num_speculative_tokens的上限，见第7批 ----------------
        # 上限max_spec的来源：
        #   串行草稿（MTP/EAGLE3）：草稿checkpoint声明的上限，否则MTP 8、EAGLE3 4
        #   DSpark独立草稿：checkpoint的dspark_max_block，否则16
        #   DSpark内嵌草稿（V4）：草稿config的sliding_window，否则128
        # 例：--method mtp --num-speculative-tokens 10，checkpoint没声明上限 -> 10 > 8，报错
        # 为什么在Config里校验：k决定了草稿模型的图形状、每条序列预留的KV slot数、CUDA Graph的捕获尺寸，
        #   超限的k要么在建图/分配时深处报错，要么悄悄跑出很低的接受率；在启动入口拒绝最清楚
        # 放在这里而不是SpeculativeConfig.__post_init__：上限要看草稿config经过hf_config_override之后的字段
        if self.speculative_config is not None:
            num_spec = self.speculative_config.num_speculative_tokens
            # getattr(..., lambda: False)()：对象没有use_dspark方法时，用一个返回False的匿名函数代替再调用
            is_dspark = getattr(self.speculative_config, "use_dspark", lambda: False)()
            draft_cfg = self.speculative_config.draft_model_hf_config
            if not is_dspark:
                # Sequential drafters (MTP / Eagle): one drafted token per
                # backbone pass, and the drafter just repeats that pass, so the
                # horizon is whatever the checkpoint declares -- else 8 for
                # plain MTP (a headroom cap; recipes run 3-5 today) and the
                # historic 4 for EAGLE.
                # 【译】串行草稿器（MTP / Eagle）：每次主干forward起草一个token，草稿器只是重复这次forward，
                #   所以上限就是checkpoint声明的值——否则普通MTP为8（留有余量的上限；目前的部署配方用3-5），
                #   EAGLE沿用历史上的4。
                max_spec = _resolve_sequential_drafter_max_spec(self.speculative_config)
            else:
                # DSpark is a PARALLEL block drafter: all flavors
                # (inline V4, standalone K3 / Qwen3 / ...) share this path with
                # no per-model constants.
                # 【译】DSpark是"并行"的块草稿器：所有形态（内嵌V4、独立的K3 / Qwen3 / ……）都走这条路径，
                #   没有按模型区分的常量。
                train_block = getattr(draft_cfg, "dspark_block_size", None)
                if getattr(draft_cfg, "dspark_with_draft", False):
                    # Standalone draft: attends a paged sibling KV holding the
                    # full context, so the hard ceiling is a block cap the
                    # checkpoint may set, else a generous default.
                    # 【译】独立草稿：它关注的是一个装着完整上下文的、分页的兄弟KV，所以硬上限是checkpoint
                    #   可能设置的块上限，否则用一个宽松的默认值。
                    max_spec = int(
                        getattr(draft_cfg, "dspark_max_block", None)
                        or _DSPARK_DEFAULT_MAX_BLOCK
                    )
                else:
                    # Inline draft (V4): attends a ROLLING target-KV window, so
                    # the hard ceiling is that window -- beyond it the
                    # [window ++ draft] block attention no longer fits its
                    # context. `sliding_window` may be present-but-None.
                    # 【译】内嵌草稿（V4）：它关注的是目标模型KV的一个"滚动"窗口，所以硬上限就是这个窗口——
                    #   超过它，[窗口 ++ 草稿] 的块注意力就装不进它的上下文了。sliding_window 可能存在但为None。
                    max_spec = int(
                        getattr(draft_cfg, "sliding_window", None)
                        or _DSPARK_DEFAULT_ROLLING_WINDOW
                    )
                if train_block and num_spec is not None and num_spec > train_block:
                    logger.warning(
                        "num_speculative_tokens=%d exceeds the DSpark draft's "
                        "training block size (%d): accepted, but expect a lower "
                        "accepted length.",
                        num_spec,
                        train_block,
                    )
            if num_spec is None or num_spec < 1 or num_spec > max_spec:
                raise ValueError(
                    f"num_speculative_tokens must be between 1 and {max_spec}, "
                    f"got {num_spec}."
                )

        # ---------------- 第12步【分支】按模型强制覆盖kv_cache_block_size，见第8批 ----------------
        # 共同原因：BlockManager和slot_mapping都假定全局只有一个block_size，而这几个模型的kernel对它有硬要求，
        #   所以在Config里统一改掉，后面的调度器、BlockManager、ModelRunner读到的都是改后的值
        #   （用户传的--block-size会被静默覆盖，不报错也不警告）；V4.1的覆盖（128）在后面第13步
        # 改后的 kv_cache_block_size 的读取方很多：BlockManager（block池、前缀缓存hash粒度）、调度器、
        #   ModelRunner（block_table列数、slot_mapping）、各注意力backend（KV pool形状），
        #   以及PD分离/KV offload的connector（两端必须用相同的block大小才能对齐地搬KV）
        # DeepSeek V4: paper §3.6.1 mandates classical KV cache block_size =
        # a multiple of lcm(m, m'). For V4-Pro / V4-Flash lcm(4, 128) = 128;
        # we use 2*lcm = 256 so each block holds k1=256/4=64 CSA entries — the
        # FP4 paged-MQA-logits indexer kernels require kv_block_size=64 (so
        # NTPW=4 N-tiles share one physical block, N_PHYS=1). ATOM's
        # BlockManager + slot_mapping math assume one global block_size, so we
        # override `kv_cache_block_size` here when V4 is detected; the V4
        # attention builder enforces the same value.
        #
        # NOTE: cannot use `hf_config.model_type` for detection — `_CONFIG_REGISTRY`
        # maps "deepseek_v4" → "deepseek_v3" so model_type reads as "deepseek_v3".
        # Use the preserved `architectures` field (re-injected by get_hf_config,
        # line 567) which keeps the original "DeepseekV4ForCausalLM[NextN]" name.
        # 【译】DeepSeek V4：论文§3.6.1规定，经典KV cache的block_size必须是 lcm(m, m') 的倍数。对V4-Pro / V4-Flash，
        #   lcm(4, 128) = 128；我们用 2*lcm = 256，这样每个block容纳 k1=256/4=64 个CSA条目——FP4的分页MQA-logits
        #   索引器kernel要求 kv_block_size=64（这样 NTPW=4 个N-tile共享一个物理block，N_PHYS=1）。ATOM的
        #   BlockManager + slot_mapping 的计算假定全局只有一个block_size，所以检测到V4时在这里覆盖
        #   kv_cache_block_size；V4的注意力builder也强制使用同样的值。
        #
        #   注意：不能用 hf_config.model_type 来检测——_CONFIG_REGISTRY 把 "deepseek_v4" 映射成 "deepseek_v3"，
        #   所以model_type读出来是 "deepseek_v3"。要用保留下来的 architectures 字段（由get_hf_config重新注入，
        #   第567行），它保留着原始的 "DeepseekV4ForCausalLM[NextN]" 名字。
        # 解释：注释里的"第567行"是旧行号，指get_hf_config中_CONFIG_REGISTRY分支末尾"把丢掉的字段补回来"的循环；
        #   实际上architectures本来就是deepseek_v3 schema里的字段，不需要补回也会保留原名
        # 【科普】DeepSeek-V4的压缩注意力（CSA / HCA）与m、m'——从MHA/GQA出发
        #   V4的KV只有1个共享头（MQA：GQA的极端，所有q头读同一份KV，head_dim 512），每层按hf_config.compress_ratios取一种：
        #     ratio 0  ：只有滑动窗口（每个token只看最近window_size=128个原始token）
        #     ratio 4  ：CSA（Compressed Sparse Attention）：每4个相邻token的KV用可学习的门控池化压成1行（相邻组有重叠），
        #                再用索引器（DSA式，见glm5_kpool_block_size上方【科普】②）在压缩行里挑top-k
        #                （recipes/DeepSeek-V4.md：1024）做注意力，外加滑动窗口
        #     ratio 128：HCA：每128个token压成1行，不用索引器、看全部压缩行，外加滑动窗口
        #   （V4-Pro是30个CSA层 + 31个HCA层 + 1个纯窗口层，见model_ops/attentions/deepseek_v4_attn.py文件头）
        #   m = 4、m' = 128 就是这两种压缩率（models/deepseek_v4.py的DeepseekV4Attention）：
        #
        #       原始token:  0 1 2 3 | 4 5 6 7 | ...... | 252 253 254 255    （一个KV block = 256个原始token）
        #       CSA行:        c0        c1     ......        c63            -> 每block 256/4   = 64行
        #       HCA行:      h0（token 0~127）            h1（token 128~255）  -> 每block 256/128 = 2行
        #
        #   例：上下文 2^20 = 1048576 个token：CSA层有 262144 个压缩行、只读其中1024行；HCA层只有 8192 行
        #   block_size必须让两种压缩都落在整行上（lcm(4,128)=128的倍数）；取256是为了CSA每block正好64行（FP4索引kernel要求）
        #   V4.1的CSA2是同一思路的变体：每层压缩率只取0/1/2，后面的层可复用前面"KV源层"的压缩KV和top-k
        #     （deepseek_v41/config.py的build_attention_topology）；第8批详解
        arches = getattr(self.hf_config, "architectures", None) or []
        is_deepseek_v4 = any(
            str(a).startswith(("DeepseekV4For", "DeepseekV4MTP", "DeepseekV4DSpark"))
            for a in arches
        )
        if is_deepseek_v4:
            v4_block_size = 256
            if self.kv_cache_block_size != v4_block_size:
                self.kv_cache_block_size = v4_block_size

        # GLM-5.3-Flash's indexer caches one pooled key per `index_kpool`
        # tokens, so a KV block of B tokens needs only B // index_kpool index
        # rows. Those rows are what `deepgemm_fp8_paged_mqa_logits` pages over,
        # and it is correct only in its preshuffled layout, which requires the
        # row count per block to be a multiple of 16. B = 16 would give 4 rows
        # and force the index cache to be padded to one row per TOKEN, wasting
        # `index_kpool - 1` of every `index_kpool` rows. B = 64 gives exactly
        # 16. Same reasoning and same mechanism as the V4 override above: the
        # BlockManager and slot_mapping assume one global block size, so it is
        # set here and the attention builder sizes the index cache from it.
        # 【译】GLM-5.3-Flash的索引器每 index_kpool 个token缓存一个池化key，所以一个B个token的KV block只需要
        #   B // index_kpool 行索引。deepgemm_fp8_paged_mqa_logits 就在这些行上分页，而它只有在预重排
        #   （preshuffled）布局下才正确，该布局要求每个block的行数是16的倍数。B = 16只会得到4行，迫使索引缓存
        #   被填充成每个"token"一行，浪费每 index_kpool 行中的 index_kpool - 1 行。B = 64正好得到16行。
        #   理由和机制都与上面V4的覆盖相同：BlockManager和slot_mapping假定全局只有一个block大小，所以在这里设置，
        #   注意力builder据此确定索引缓存的大小。
        # 解释：上面"B=16得4行、B=64得16行"对应index_kpool=4
        # 什么是kpool、为什么要改block大小：见 glm5_kpool_block_size 上方的【科普】（四步图解）
        is_glm5_next = any("Glm5Next" in str(a) for a in arches)
        if is_glm5_next:
            unsupported_features = _glm5_next_unsupported_features(self)
            if unsupported_features:
                raise ValueError(
                    "GLM-5.3-Flash text serving does not yet support "
                    f"{', '.join(unsupported_features)}"
                )
            index_kpool = int(getattr(self.hf_config, "index_kpool", 1) or 1)
            if index_kpool > 1:
                glm5_block_size = glm5_kpool_block_size(index_kpool)
                if self.kv_cache_block_size != glm5_block_size:
                    self.kv_cache_block_size = glm5_block_size

                # Turning sparsity off is an exact A/B only at or below
                # `index_topk`; past it the dense/token-granular fallback is a
                # DIFFERENT answer, not a slower one. Validate that at startup
                # rather than per forward: `max_model_len` already bounds
                # `max_seqlen_k`, so one check here replaces a branch on a
                # 45-layer hot path, and it fails the launch cleanly instead of
                # raising inside the model and taking the engine down with it
                # mid-batch -- which is what both `envs.py` and
                # `docs/environment_variables.md` describe as refusing the
                # request.
                # 【译】只有在 index_topk 以内，关闭稀疏性才是精确的A/B对比；超过它，dense/按token粒度的后备实现给出的
                #   是一个"不同"的答案，而不是更慢的同一个答案。在启动时校验，而不是每次forward校验：max_model_len 已经
                #   限定了 max_seqlen_k，所以这里一次检查就能替代45层热路径上的一个分支；而且它会干净地让启动失败，
                #   而不是在模型内部抛异常、在batch中途把引擎一起拖垮——envs.py和docs/environment_variables.md
                #   把后者描述为"拒绝该请求"。
                # 用【科普】里的概念说：两个调试开关各关掉一部分，用来排查池化/稀疏路径的bug
                #   - ATOM_GLM5_KPOOL=0：不池化，索引器按"每个token"打分、选top-2048个token（不再按4个一池选512个池）
                #   - ATOM_GLM5_FORCE_DENSE_MLA=1：不跑索引器，主注意力直接看全部token（dense）
                #   序列 <= 2048 时，三种做法都会选中全部token，结果完全一样，所以可以拿来做精确的A/B对比：
                #     有差异就说明是池化/稀疏路径的bug；
                #   序列 > 2048 时，三者选中的token本来就不同，有差异是正常的，比较没有意义。
                #   所以开了这两个开关之一时，要求 max_model_len <= index_topk，否则启动就报错
                index_topk = int(getattr(self.hf_config, "index_topk", 0) or 0)
                sparsity_off = []
                if not envs.ATOM_GLM5_KPOOL:
                    sparsity_off.append("ATOM_GLM5_KPOOL=0")
                if envs.ATOM_GLM5_FORCE_DENSE_MLA:
                    sparsity_off.append("ATOM_GLM5_FORCE_DENSE_MLA=1")
                if sparsity_off and index_topk and self.max_model_len > index_topk:
                    raise ValueError(
                        f"GLM-5.3-Flash: {', '.join(sparsity_off)} turns the "
                        "pooled sparse selection off, which matches the pooled "
                        f"path only while sequences stay at or below "
                        f"index_topk={index_topk}. max_model_len is "
                        f"{self.max_model_len}. Lower --max-model-len to "
                        f"{index_topk} for the comparison, or drop the "
                        "override."
                    )

        # ---------------- 第13步【分支】模型相关的收尾：M3索引器CP、索引缓存dtype、V4.1的block_size与重放参数、V4重放参数 ----------------
        # MiniMax-M3 indexer-only context parallelism. Off by default: it trades
        # a fixed per-layer all-to-all for a large long-context score win, so it
        # is a loss below roughly batch*context ~ 1M tokens. Warn and fall back
        # rather than raise -- this is a performance flag, never correctness.
        # Ops override wins over the config field, in both directions.
        # 【译】MiniMax-M3仅索引器的上下文并行。默认关闭：它用每层固定一次all-to-all换取长上下文下打分的大幅收益，
        #   所以在大约 batch*context ~ 1M token以下是亏的。警告并回退而不是报错——这是一个性能开关，从不涉及正确性。
        #   运维用的环境变量覆盖优先于config字段，两个方向（开/关）都是。
        # 环境变量ATOM_M3_INDEXER_CP设置了就以它为准："1"开启，其他值关闭；没设置（None）时用dcp_config里的值
        # 结果 dcp_config.indexer_dcp_only 的读取方：distributed/indexer_cp.py（运行时是否走CP）、
        #   minimax_m3/model.py / linear.py（index_q投影要不要加宽）、本类的compute_hash（编译缓存key）
        # 必须放在第5步之后：要用到 hf_config 的架构名、kv头数和 sparse_attention_config
        if envs.ATOM_M3_INDEXER_CP is not None:
            self.dcp_config.indexer_dcp_only = envs.ATOM_M3_INDEXER_CP == "1"
        if self.dcp_config.indexer_dcp_only:
            text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
            sparse_cfg = getattr(text_cfg, "sparse_attention_config", None) or {}
            indexer_cp_off = indexer_cp_unsupported_reason(
                arches,
                self.tensor_parallel_size,
                getattr(text_cfg, "num_key_value_heads", 0),
                sparse_cfg.get("sparse_block_size", 0),
                self.decode_context_parallel_size,
                self.enable_tbo or self.enable_tbo_decode,
                is_plugin_mode(),
            )
            if indexer_cp_off is not None:
                logger.warning(
                    "dcp_config.indexer_dcp_only disabled: %s.", indexer_cp_off
                )
                self.dcp_config.indexer_dcp_only = False
            else:
                # Announce the ON case too. The "Engine kwargs" dump is emitted
                # before this runs (arg_utils.py), so it prints the pre-override
                # value -- reading it as the live setting is how an A/B ends up
                # comparing a config against itself.
                # 【译】开启的情况也打日志。"Engine kwargs" 那行日志在这之前输出（arg_utils.py），打印的是覆盖前的值——
                #   把它当成实际生效的设置，A/B对比就会变成拿一个配置和它自己比。
                logger.info(
                    "dcp_config.indexer_dcp_only enabled: MiniMax-M3 indexer "
                    "scores all %d index heads over 1/%d of the blocks.",
                    getattr(text_cfg, "num_key_value_heads", 0),
                    self.tensor_parallel_size,
                )

        # Keep ``None`` intact until the model architecture is known so an
        # omitted index-cache option remains distinguishable from an explicit
        # fp8/bf16 override. Native single-node V4 defaults to the FP4 indexer
        # except on gfx942. Plugin proxy pools and KV-transfer region maps do
        # not yet describe the separate FP4 scale pool, so those integrations
        # retain FP8 until their layouts support it. Every other model keeps
        # the historical KV-cache-dtype default.
        # 【译】在模型架构确定之前一直保持 None，这样"没有指定索引缓存选项"仍能与"显式指定fp8/bf16"区分开。
        #   原生单节点的V4默认使用FP4索引器，gfx942除外。插件的代理pool和KV传输的区域映射还不能描述单独的
        #   FP4 scale pool，所以这些集成在其布局支持之前保留FP8。其他所有模型沿用历史默认值：跟随KV cache的dtype。
        # index_cache_dtype 是DSA/CSA稀疏注意力的"索引器key缓存"（与主KV cache分开的一块pool）的格式。
        #   DSA/索引器入门见glm5_kpool_block_size上方【科普】②，CSA见上面第12步V4部分的【科普】
        # 谁读：各注意力backend（aiter_mla.py、deepseek_v41/backend.py、v4_kernels 等）按它分配索引pool并选打分kernel；
        #   pool_layout/v4_pool_fields.py 按它算每个block的字节布局；KV offload（offload/config.py、keys.py、
        #   kv_byte_codec.py）按它编解码、并编进存储key；compute_hash 也包含它
        # 决策表（只在用户没指定--index-cache-dtype时）：
        #   V4、原生、无KV传输：gfx942 -> fp8，其他 -> fp4；V4插件或有KV传输 -> fp8
        #   V4.1 -> fp8；其他模型 -> 等于kv_cache_dtype
        if self.index_cache_dtype is None:
            if is_deepseek_v4:
                if self.plugin_config is None and not self.kv_transfer_config:
                    from aiter.jit.utils.chip_info import get_gfx

                    self.index_cache_dtype = "fp8" if get_gfx() == "gfx942" else "fp4"
                else:
                    self.index_cache_dtype = "fp8"
            elif self.hf_config.model_type == "deepseek_v41_text":
                # CSA2 reads its index plane with the paged scorer and has no
                # other, so the plane's format does not follow the main pool's.
                # Only the default: an explicit value is carried through to the
                # V4.1 runtime gate, which refuses anything else before weights
                # load. Overriding it here would accept the flag and ignore it.
                # 【译】CSA2用分页打分器读取它的索引平面，没有别的读法，所以这个平面的格式不跟随主pool。
                #   这里只设默认值：显式给出的值会原样传到V4.1的运行时门控，它会在加载权重之前拒绝其他任何值。
                #   如果在这里覆盖，就等于接受了这个参数却又忽略它。
                self.index_cache_dtype = "fp8"
            else:
                self.index_cache_dtype = self.kv_cache_dtype

        # DeepSeek-V4.1：block_size强制128，并设置有界重放所需的两个字段，最后交给V4.1自己的运行时校验
        # 用model_type判断：V4.1的config由它自己的normalize_hf_config生成（get_hf_config路径1），
        #   model_type固定为deepseek_v41_text，不像V4那样被_CONFIG_REGISTRY改写
        if self.hf_config.model_type == "deepseek_v41_text":
            # CSA2's index plane is paged by the same scorer GLM's is, so a
            # PAGE again has to hold a whole number of `MQA_LOGITS_PRESHUFFLE_ROWS`
            # tiles -- and its ratio-2 owners halve the PAGE before that count
            # is taken, which makes the floor twice the tile. 128 rather than
            # that floor: the block table is `[max_num_seqs, max_model_len /
            # block_size]` int32 held twice and copied to the device every
            # forward, 17 MB here against 134 MB at the 16 this field defaults
            # to. Not the 256 DeepSeek-V4 takes: a PAGE is also the prefix
            # cache's hash block, so it is the granularity of prefix reuse and
            # of the rows encoder bounded replay recomputes below a hit: one
            # PAGE, which must cover the window - 1 rows a row at the hit reads.
            # 【译】CSA2的索引平面和GLM的一样由同一个打分器分页，所以一个PAGE同样必须装下整数个
            #   MQA_LOGITS_PRESHUFFLE_ROWS 的tile——而且它压缩率为2的那些层在计算这个数之前会把PAGE减半，
            #   所以下限是tile的两倍。用128而不是这个下限：block table是 [max_num_seqs, max_model_len / block_size]
            #   的int32，保存两份、每次forward都拷到设备上，在这里是17 MB，而用这个字段默认的16时是134 MB。
            #   也不用DeepSeek-V4的256：一个PAGE同时也是前缀缓存的hash block，所以它是前缀复用的粒度，也是encoder
            #   有界重放在命中位置之下重新计算的行数：一个PAGE，它必须覆盖命中位置那一行要读的 window - 1 行。
            # 解释：block_size越小block table越长（每条序列 max_model_len / block_size 项），
            #   越大则前缀复用越粗、重放的行越多，128是两者的折中
            self.kv_cache_block_size = 128
            # 这两个字段供BlockManager/调度器决定重放从哪里开始、带多少历史token（第4批）
            #   min_final_prefill_rows：BlockManager.replay_start 读，前缀命中后最后一次prefill至少从
            #     prompt末尾往前 sliding_window 行开始重算，SWA层才能拿到完整窗口（deepseek_v41/bounded_replay.py）
            #   replay_history_tokens：Scheduler 读，重放时额外把开始位置之前这么多个token一起带上，
            #     供Engram做n-gram查找（n-gram需要看前 n-1 个token）
            if self.enable_decoder_swa_bounded_replay:
                self.min_final_prefill_rows = self.hf_config.sliding_window
            self.replay_history_tokens = self.hf_config.engram_max_ngram_size - 1

            from atom.models.deepseek_v41.config import validate_runtime_config

            # 在加载权重之前拒绝V4.1尚未实现的执行模式（不支持的CUDA Graph模式、torch.compile level 1/2、
            #   PP、上下文并行、不带DP attention的DP/TBO、除lmcache_mp以外的KV传输等），
            #   这样配置错误几秒内就报出来，而不是等几分钟加载完权重才失败
            # 放在最后：它要检查的是本函数改写之后的最终值（block_size、cudagraph_mode等）
            validate_runtime_config(self)

        # DeepSeek-V4：根据窗口感受野算出state_replay_tokens（没设ATOM_DSV4_STATE_REPLAY或条件不满足时为0），第8批
        # 读取方：BlockManager（为0时，背后没有状态checkpoint的前缀命中直接放弃；>0时从命中位置往前这么多token
        #   开始重算来重建状态）、Scheduler、KV offload的 mp/native_state_* （决定状态能否按命中复用）
        # 放在最后：state_replay_tokens(self) 要读本函数已经定下来的 kv_cache_block_size 等字段
        if is_deepseek_v4:
            from atom.model_ops.attentions.deepseek_v4_replay import (
                state_replay_tokens,
            )

            self.state_replay_tokens = state_replay_tokens(self)

    # 动态分块prefill（PP专用）的校验与降级；调用方：__post_init__第10步
    # 先做"参数本身是否合法"的硬校验（raise），再看拓扑是否适用（不适用就警告并关掉）
    # 动态分块是什么：PP下一个长prompt被切成多个chunk依次流过各stage；chunk越靠后，要attend的前缀越长、
    #   算得越慢，固定chunk大小会让流水线出现气泡。动态分块按延迟模型给每个chunk算不同的大小，使各chunk耗时接近
    # 结果的读取方：
    #   enable_dynamic_chunking：EngineCore（启动时让ModelRunner做profiling，失败或非head stage时再关掉）、
    #     Scheduler（_dynamic_chunk_limit）、dynamic_chunking_worker.py（运行时计时）
    #   smooth_factor / min_chunk_size：Scheduler 求解chunk大小时用（平滑系数、chunk下限）
    # 为什么"警告并关掉"而不是报错：它只是性能开关，同一份启动脚本可能跑在不适用的拓扑上
    def _resolve_dynamic_chunking(self) -> None:
        """Validate dynamic chunking, or turn it off where it cannot size a chunk.

        Everything here, the env read included, runs only when the feature is
        on: a stale ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR in a shared launch script
        must not stop a server that never asked for dynamic chunking.
        """
        # 【译】校验动态分块，或者在它无法确定chunk大小的地方把它关掉。
        #
        #   这里的所有逻辑（包括读环境变量）都只在功能开启时运行：共享启动脚本里一个过时的
        #   ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR 不能让一个从未要求动态分块的服务器启动失败。
        if not self.enable_dynamic_chunking:
            return
        if self.dynamic_chunking_smooth_factor is None:
            try:
                self.dynamic_chunking_smooth_factor = (
                    envs.ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR
                )
            except ValueError as exc:
                raise ValueError(
                    f"ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR must be a number: {exc}"
                ) from None
        if not 0.0 <= self.dynamic_chunking_smooth_factor <= 1.0:
            raise ValueError(
                "dynamic_chunking_smooth_factor must be in [0, 1], got "
                f"{self.dynamic_chunking_smooth_factor}"
            )
        if self.dynamic_chunking_min_chunk_size <= 0:
            raise ValueError("dynamic_chunking_min_chunk_size must be positive")
        # 不适用的拓扑：没有PP、没开chunked prefill、RapidServe、PD分离的decode实例
        if self.pipeline_parallel_size <= 1:
            reason = "pipeline_parallel_size is 1"
        elif not self.enable_chunked_prefill:
            reason = "chunked prefill is off"
        elif self.enable_rapidserve:
            reason = "the RapidServe prefill scheduler never chunks"
        elif (self.kv_transfer_config or {}).get("kv_role") == "kv_consumer":
            reason = "a P/D decode instance does not run the prefills it would size"
        else:
            return
        # Same degrade-to-fixed path as a failed profile or a rejected fit.
        # Raising would make the flag a hard failure in a shared launch script.
        # 【译】与profile失败或拟合被拒绝时走同一条"降级为固定分块"的路径。报错会让这个参数在共享启动脚本里
        #   变成一个硬性故障。
        logger.warning(
            "Dynamic chunking disabled because %s; chunking stays fixed", reason
        )
        self.enable_dynamic_chunking = False

    # 整个Config的"编译相关"hash，返回10位十六进制字符串
    # 调用方：utils/backends.py的VllmBackend：没有指定cache_dir时，用它（再加上追踪到的源文件hash、编译器hash）
    #   生成torch.compile缓存目录 ~/.cache/atom/torch_compile_cache/<hash>/（第5批）
    # 为什么重要：两次启动的hash相同，就会直接加载上次编译好的产物；如果漏掉了某个会改变计算图的配置，
    #   就会加载到不匹配的旧产物，运行时报assert_size_stride之类的错误——下面多条英文注释说的都是这种事故
    # 收集的因素：三个子配置的hash + 一组会改变图结构/图中常量的字段和环境变量
    # 对比nano-vllm：nano-vllm不用torch.compile，没有这个方法
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # 【译】警告：每当这个config新增字段时，如果它会影响计算图，务必把它加进factors列表。
        #
        #   提供一个hash，唯一标识所有会影响"从input ids/embeddings到最终hidden states"这段计算图结构的配置，
        #   不包括input ids/embeddings之前和最终hidden states之后的部分。
        factors: list[Any] = []

        # summarize vllm config
        # 【译】汇总（沿用vLLM命名的）各子配置
        vllm_factors: list[Any] = []
        if self.quant_config:
            vllm_factors.append(self.quant_config.compute_hash())

        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())

        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())

        factors.append(vllm_factors)
        # TP大小决定每层权重切片的形状
        factors.append(self.tensor_parallel_size)
        # PCP changes the compiled graph: when pcp>1 the indexer runs through the
        # opaque `sparse_attn_indexer_pcp` op and takes the round-robin all-gather
        # / separate-rope path. A pcp1 vs pcp2 run over the same model+source otherwise hashes
        # identically, so without this factor pcp2 loads pcp1's cached artifact
        # (no indexer op) and trips copy_misaligned_inputs / assert_size_stride at
        # runtime — the same stale-artifact hazard documented for the vocab-embed
        # flag below.
        # 【译】PCP会改变编译出的图：pcp>1时索引器经由不透明的 sparse_attn_indexer_pcp 算子运行，并走轮转
        #   all-gather / 单独rope的路径。否则同一个模型+源码的pcp1和pcp2两次运行hash完全相同，没有这个因素的话，
        #   pcp2会加载pcp1缓存的产物（里面没有索引器算子），并在运行时触发 copy_misaligned_inputs /
        #   assert_size_stride 错误——与下面词表embedding开关所记录的"陈旧产物"隐患相同。
        factors.append(self.prefill_context_parallel_size)
        # MiniMax-M3 indexer-only CP changes the FUSED QKV OUTPUT WIDTH: index_q
        # is this rank's one head under TP and all `sparse_num_index_heads` of
        # them under CP (minimax_m3/model.py, linear.py), so the traced graph and the
        # captured buffer strides differ. Exactly the pcp hazard above -- two
        # runs of the same model and source otherwise hash identically, so
        # starting one mode after the other would load the opposite mode's
        # artifact and trip assert_size_stride at runtime.
        #
        # This reads the field AFTER __post_init__, which is the only correct
        # value: it already folds in the ATOM_M3_INDEXER_CP override AND every
        # indexer_cp_unsupported_reason fallback (plugin mode, TP != kv heads,
        # block != 128, DCP, TBO). Two topologies that both fall back therefore
        # share one artifact, as they should.
        # 【译】MiniMax-M3仅索引器CP会改变"融合QKV的输出宽度"：TP下index_q是本rank的一个头，CP下是全部
        #   sparse_num_index_heads 个头（minimax_m3/model.py、linear.py），所以追踪出的图和捕获的缓冲区步长都不同。
        #   和上面pcp的隐患完全一样——否则同一个模型和源码的两次运行hash相同，先后以两种模式启动，
        #   后者会加载前者模式的产物，并在运行时触发 assert_size_stride。
        #
        #   这里读取的是 __post_init__ "之后"的字段值，这是唯一正确的值：它已经折算了 ATOM_M3_INDEXER_CP 的覆盖，
        #   以及 indexer_cp_unsupported_reason 的每一种回退（插件模式、TP != kv头数、block != 128、DCP、TBO）。
        #   因此两种都回退了的拓扑会共享同一份产物，这正是应该的。
        factors.append(bool(getattr(self.dcp_config, "indexer_dcp_only", False)))
        factors.append(self.enable_dp_attention)
        factors.append(self.index_cache_dtype)
        # DeepSeek-V4.1 decoder SWA bounded replay compiles the backbone as two
        # graphs (early + late layers) instead of one; both compile under the
        # `backbone` tag, so the flag must key the artifacts.
        # 【译】DeepSeek-V4.1 decoder的SWA有界重放把主干编译成两张图（前段层 + 后段层）而不是一张；两张图都在
        #   backbone 标签下编译，所以这个开关必须作为产物的key。
        factors.append(getattr(self, "enable_decoder_swa_bounded_replay", True))
        # 稀疏注意力索引器的4个结构开关（MiniMax-M3等）：先查text_config，没有再查根config
        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        factors.append(
            (
                getattr(
                    text_config,
                    "use_index_cache",
                    getattr(self.hf_config, "use_index_cache", False),
                ),
                getattr(
                    text_config,
                    "index_topk_freq",
                    getattr(self.hf_config, "index_topk_freq", None),
                ),
                getattr(
                    text_config,
                    "index_topk_pattern",
                    getattr(self.hf_config, "index_topk_pattern", None),
                ),
                getattr(
                    text_config,
                    "index_skip_topk_offset",
                    getattr(self.hf_config, "index_skip_topk_offset", None),
                ),
            )
        )
        # Vocab-embedding replication (ATOM_REPLICATE_VOCAB_EMBED) changes both the
        # embed weight shape ([vocab] vs [vocab/tp]) and the embed op (local
        # F.embedding vs masked-embedding + all-reduce), so it alters the compiled
        # graph and MUST be part of its key. Without this, toggling the flag — or
        # deploying it on top of a cache built with the other setting — reuses a
        # stale artifact and trips assert_size_stride at runtime.
        # 【译】词表embedding复制（ATOM_REPLICATE_VOCAB_EMBED）既改变embedding权重的形状（[vocab] 对 [vocab/tp]），
        #   也改变embedding算子（本地 F.embedding 对 带掩码的embedding + all-reduce），所以它改变了编译出的图，
        #   "必须"成为key的一部分。没有它的话，切换这个开关——或者在用另一种设置建好的缓存上部署——就会复用陈旧的产物，
        #   并在运行时触发 assert_size_stride。
        factors.append(bool(envs.ATOM_REPLICATE_VOCAB_EMBED))
        # The drafter graph bakes its draft width in as a constant (DSpark traces
        # `[bs, num_draft]`), so a run that only changes --num-speculative-tokens
        # would otherwise load the other width's artifact: draft 6 over a draft 3
        # cache hands MoE half the rows it expects. Acceptance-length knobs only
        # steer sampling and stay out.
        # 【译】草稿模型的图把草稿宽度作为常量烘焙进去（DSpark追踪的是 [bs, num_draft]），所以一次只改了
        #   --num-speculative-tokens 的运行，否则会加载另一种宽度的产物：在草稿数3的缓存上跑草稿数6，
        #   交给MoE的行数只有它期望的一半。接受长度相关的开关只影响采样，不放进来。
        spec = self.speculative_config
        factors.append(
            None if spec is None else (spec.method, spec.num_speculative_tokens)
        )

        # md5只用来生成目录名，不涉及安全（usedforsecurity=False，在FIPS模式的系统上也能用）；取前10位
        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str


# ---------------- 进程内的"当前Config"全局变量 ----------------
# 为什么需要：模型层（注意力、Linear、MoE等）构造和forward时要读配置，但把config一层层作为参数传下去很麻烦，
#   于是每个进程设一个全局变量，层里直接调用get_current_atom_config()
# 谁设置：原生模式下ModelRunner.__init__一开始就调用set_current_atom_config(config)，所以ModelRunner进程里
#   之后构造的所有模型层都能读到；插件模式由plugin/config.py、SGLang的模型包装器设置
# 对比nano-vllm：nano-vllm的层不读Config（需要的参数在构造时传入），没有这个全局变量
_current_atom_config: Config | None = None


# global关键字：在函数内给模块级变量赋值时必须声明，否则Python会把它当成新的局部变量
def set_current_atom_config(atom_config: Config):
    global _current_atom_config
    _current_atom_config = atom_config


# 【分支】vLLM插件模式：先从vLLM当前的forward context里取atom_config，见第10批
# vLLM没装、或者当前不在forward中、或者取的过程中出现ImportError/AttributeError，都返回None（调用方退回全局变量）
def _get_current_atom_config_from_vllm_forward_context() -> Config | None:
    # In vLLM plugin mode (especially speculative decode), main/draft models
    # can coexist in one process. Resolve per-forward config first to avoid
    # reading a stale global singleton.
    # 【译】vLLM插件模式下（尤其是投机解码），主模型和草稿模型可能共存于同一个进程。先解析每次forward
    #   对应的config，避免读到一个过时的全局单例。
    try:
        from vllm.forward_context import (
            get_forward_context as get_vllm_forward_context,
        )
        from vllm.forward_context import (
            is_forward_context_available,
        )
    except (ImportError, AttributeError):
        return None
    if not is_forward_context_available():
        return None
    try:
        return get_vllm_forward_context().additional_kwargs.get("atom_config")
    except (ImportError, AttributeError):
        return None


# 读取当前Config；在一个从未设置过的进程里调用会触发断言错误
# 调用方：模型层、注意力后端、算子等，atom/下约50个文件调用它（例如moe.py读EP/EPLB开关、
#   deepseek_v2.py读dcp_config、eplb.py读eplb_config）
# 注意：只在ModelRunner进程（或插件的模型进程）里能用；EngineCore、调度器所在进程没有调用过set，
#   那里的代码应直接用传进来的config
def get_current_atom_config() -> Config:
    # Try to get the atom config from forward context first in vLLM plugin mode.
    # 【译】vLLM插件模式下，先尝试从forward context里取atom config。
    if is_vllm():
        forward_atom_config = _get_current_atom_config_from_vllm_forward_context()
        if forward_atom_config is not None:
            return forward_atom_config
    assert _current_atom_config is not None, "Current atom config is not set"
    return _current_atom_config


# 在with块内临时把"当前Config"换成custom_atom_config，离开时恢复原值
# 用法：with use_custom_atom_config(draft_config): 构造草稿模型
#   草稿模型的层构造时调用get_current_atom_config()，拿到的是草稿自己的Config（例如往它的
#   static_forward_context里登记层），而不是主模型的
# 调用方：vLLM插件 model_wrapper.py 构造投机解码草稿模型时（第10批）——vLLM插件下主模型和草稿模型在同一进程，
#   全局变量里是主模型的Config，不临时换掉的话草稿层会把自己登记到主模型的forward context里
# @contextmanager：把一个带yield的生成器函数变成上下文管理器：yield之前的代码在进入with时执行，
#   yield的值绑定到as后面的变量，finally里的代码在离开with时执行（即使块内抛了异常也会执行）
@contextmanager
def use_custom_atom_config(custom_atom_config: Config):
    # Temporarily masquerade the custom atom_config as the current atom_config
    # for the current context and restore upon exit
    # 【译】在当前上下文里临时让自定义的atom_config冒充当前的atom_config，退出时恢复
    global _current_atom_config
    prev = _current_atom_config
    _current_atom_config = custom_atom_config
    try:
        yield custom_atom_config
    finally:
        _current_atom_config = prev
