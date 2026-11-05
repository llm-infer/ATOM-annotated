# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】loader.py：权重加载的入口——把 checkpoint（*.safetensors）里的张量写进按 config 搭好的"空架子"模型，
#   再做加载后处理（在线量化、AITER 布局重排），之后模型才能 forward
#   位置：ModelRunner.__init__ → _build_and_load_model → load_model（本文件；model_runner.py）
#         草稿模型（MTP / EAGLE3 / DSpark）：spec_decode/drafter.py 的 load_model 再调一次本文件的 load_model
#         插件模式（vLLM / SGLang / RTP-LLM，第 10 批）：各插件模型的 load_weights → load_model_in_plugin_mode（本文件）
#   本目录各文件的分工（都属于第 5 批）：
#     loader.py                 入口；default_weight_loader；假权重；加载后处理与汇总日志          （本文件，A）
#     loading_core.py           加载主循环：读 → 改名 → 分发 → 等线程收尾 → 覆盖率检查            （A）
#     weight_iterator.py        从磁盘读张量：mmap / 整段 read、按 header 跳过分片、后台预读 page cache （A）
#     weight_names.py           checkpoint 名 → 参数名的 9 条改名规则（CheckpointNameRewriter）         （B）
#     weight_dispatch.py        改名后的张量交给哪个参数、带哪些参数调它的 weight_loader                 （B）
#     expert_staging.py         MoE 逐专家的小张量先拼进 pinned CPU 缓冲，凑齐再一次 H2D                 （B）
#     online_quant_streaming.py 边加载边在线量化，压低加载期的显存峰值                                  （B）
#     weight_utils.py           HF 下载、按 index.json 过滤分片、set_weight_attrs（改编自 vLLM）         （B）
#     deepseek_v41.py           DeepSeek-V4.1 的 Engram 主机表（不走参数路径）                          （C）
#
# 阅读路线
#   ① 【前置①~③】TP 列切/行切、融合参数与 packed_modules_mapping、MoE 专家堆叠——后面的例子要用到它们
#   ② 【全流程图】，再看【例】一个 k_proj 张量从磁盘走到 qkv_proj 的第几行（对照 loading_core → weight_names →
#      weight_dispatch → linear.py 推算过）
#   ③ 【要点①~④】并行加载、专家暂存、流式在线量化、假权重为什么存在、代价是什么
#   ④ 本文件的函数：load_model（主干）→ default_weight_loader → initialize_dummy_weights → load_model_in_plugin_mode
#
# ---------------------------------------------------------------------------------------------------------
# 【前置①】TP 的列切与行切（概念的完整讲解见 arg_utils.py tensor_parallel_size 上方【科普】；实现在第 6 批 linear.py）
#   nn.Linear 的权重存成 [out_features, in_features]，y = x · Wᵀ。以 Qwen3-8B（Llama-3-8B 形状相同）的注意力为例
#   （这里先把 q_proj 当作独立的层；ATOM 实际把 q/k/v 融合成一个参数，见【前置②】）：
#   hidden = 4096，32 个 q 头、8 个 KV 头（GQA），head_dim = 128；TP = 4：
#     列切（ColumnParallelLinear，tp_dim = 0，切输出维 = 权重的行）：
#       q_proj.weight [4096, 4096] → rank r 只存第 [1024r, 1024r+1024) 行 = [1024, 4096]，即第 8r ~ 8r+7 个 q 头
#       每卡只算自己那 8 个头的 q，不需要通信（输出按头分开，各管各的）
#     行切（RowParallelLinear，tp_dim = 1，切输入维 = 权重的列）：
#       o_proj.weight [4096, 4096] → rank r 只存第 [1024r, 1024r+1024) 列 = [4096, 1024]
#       输入正好是本卡 8 个头的注意力输出 [T, 1024]，算出的是部分和 [T, 4096]，最后 all-reduce 相加
#   对加载的含义：checkpoint 里存的永远是完整张量；每个 rank 都读完整张量，只把属于自己的那一片（narrow）拷进参数。
#     参数在构造时就已经是切过的形状（比 checkpoint 小 tp 倍），"取哪一片、写到哪"由参数自己的 weight_loader 决定
#
# 【前置②】融合参数与 packed_modules_mapping
#   q、k、v 三个投影的输入都是同一个 hidden，ATOM（和 nano-vllm 一样）把三者的权重按行拼成一个 qkv_proj，
#   一次 GEMM 代替三次（少两次 kernel launch，GEMM 更大、利用率更高）；MLP 的 gate_proj、up_proj 同理拼成 gate_up_proj。
#   checkpoint 里却仍是分开的 q_proj / k_proj / v_proj，名字对不上——模型类用类属性 packed_modules_mapping 声明对应关系
#   （qwen3.py、llama.py 原样如此）：
#       packed_modules_mapping = {
#           "q_proj": ("qkv_proj", "q"),          checkpoint 分量名 → (融合参数名, shard id)
#           "k_proj": ("qkv_proj", "k"),          shard id 告诉融合参数的 weight_loader：这是几份里的哪一份、写到哪几行
#           "v_proj": ("qkv_proj", "v"),          同上，v 那一份
#           "gate_proj": ("gate_up_proj", 0),     MLP：gate 是第 0 份、up 是第 1 份
#           "up_proj": ("gate_up_proj", 1),       （MergedColumnParallelLinear 按序号定位）
#       }
#   值也可以是列表（反方向："checkpoint 已融合、模型里分开"），见 weight_dispatch.py 的 _dispatch_packed。
#   量化那一侧也要跟着改名（checkpoint 的 quantization_config 按融合前的名字写排除项 / 逐层规格），在 config.py 的
#     QuantizationConfig.remap_layer_name，由 ModelRunner.__init__ 在构造模型之前调用；这里的加载只管张量
#
# 【前置③】MoE 的专家堆成一个 [E, ...] 张量（MoE 与 EP 的概念见 arg_utils.py enable_expert_parallel 上方【科普】；
#   FusedMoE 的实现与权重布局见第 7 批 moe.py）
#   MoE 层有很多个专家 FFN，例如 DeepSeek-V3 每层 256 个路由专家 + 1 个共享专家，每个专家都是 gate/up/down 三个线性层。
#   checkpoint 里每个专家的每个投影是单独一个张量：model.layers.3.mlp.experts.17.gate_proj.weight [2048, 7168] ……
#   ATOM 的 FusedMoE 把一层所有本地专家堆成一个张量，好让一个 fused-MoE kernel 一次算完所有专家：
#       w13_weight [E_local, 2·I/tp, H]   每个专家的 gate（w1）和 up（w3）按行拼在一起（同 gate_up_proj）
#       w2_weight  [E_local, H, I/tp]     每个专家的 down（w2）
#     （I = moe_intermediate_size，H = hidden_size；不开 EP 时 E_local = 全部专家、每个专家按 TP 切，开 EP 时按专家整个分）
#   所以每个专家张量要知道三件事：写进哪个堆叠参数（w13 / w2）、第几个专家（expert_id）、哪一份（w1 / w2 / w3）。
#     这由模型的 get_expert_mapping() 给出（FusedMoE.make_expert_params_mapping 生成，V3 为 (256+1) × 3 = 771 条）。
#   共享专家也可以当作第 n_routed 个专家塞进同一个堆叠张量（名字改写成 experts.256.*，见 weight_names.py）
#
# ---------------------------------------------------------------------------------------------------------
# 【全流程图】（顶格写的是流动的数据，`│` 后面是处理它的代码）
#
#   磁盘：model-00001-of-000163.safetensors …（每个分片 = 8 字节长度 + header(JSON：名字→dtype/shape/偏移) + 数据区）
#     │  weight_iterator.safetensors_weights_iterator（主线程上的一个生成器）
#     │    ├ 不是目录就当 HF hub id 下载；按 model.safetensors.index.json 去掉索引外的文件（weight_utils.py）
#     │    ├ 只读 header，丢掉"一个想要的张量都没有"的分片（草稿模型只要 MTP 那一层时，约 98% 的分片不用读）
#     │    ├ 后台线程把本节点分到的分片顺序读一遍、预热 page cache（ATOM_LOADER_PREFETCH，默认开）
#     │    └ 逐个 yield (checkpoint 名, CPU 张量)：默认 mmap；ATOM_DISABLE_MMAP=true 时整段 readinto 进内存
#     ▼
#   (ckpt_name, tensor)                       ← 名字和 CPU 张量
#     │  weight_names.CheckpointNameRewriter.rewrite：9 条规则按固定顺序改名，返回 None = 这个模型不要它
#     │    （模型声明的 WeightsMapper / weights_mapping、MTP 过滤与重映射、丢 kv_scale / inv_freq、
#     │     weight_scale_inv → weight_scale、丢超出层数的层、共享专家 → experts.{n_routed}）
#     ▼
#   (ckpt_name, param_name, tensor)           ← 多了改名后的参数名
#     │  weight_dispatch.WeightDispatcher.dispatch：按顺序试，前一条不认领才试下一条
#     │    packed        名字含 packed_modules_mapping 的 key → 融合参数名 + shard id          【前置②】
#     │    fused expert  checkpoint 把整层专家存成一个张量（Qwen3.5 BF16）→ 模型自带的加载函数
#     │    per expert    含 experts.{id}.{gate|up|down}_proj. → w13_ / w2_ 堆叠参数 + (expert_id, w1/w2/w3)  【前置③】
#     │    merged        experts.{id}.<其他名字> → 去掉 id 的合并参数
#     │    generic       同名参数（找不到就记进 dropped_ckpt_keys，加载完统一告警）
#     ▼
#   submit(weight_loader, param, tensor, [shard_id / expert_id …])   ATOM_LOADER_NUM_THREADS>1（默认 16）时进线程池
#     │    参数自己的 weight_loader（linear.py、moe.py、embed_head.py …；没有就用本文件的 default_weight_loader）：
#     │      按 tp_rank 从完整张量 narrow 出本 rank 的一片 → 按 shard id 定位到融合参数里的偏移 → copy_（CPU → GPU）
#     │    逐专家小张量 → expert_staging.ExpertStagingPool：先拼进 pinned CPU 缓冲，凑齐一个参数再一次 H2D   【要点②】
#     │    开流式在线量化时 → online_quant_streaming.OnlineQuantStreamer.run：一个模块到齐就量化、释放源权重 【要点③】
#     ▼
#   loading_core：等线程池排空 → 刷出暂存池里剩下的条目（缺专家就报错）→ 双向覆盖率报告
#     ▼
#   load_model（本文件）的加载后处理：按 named_modules 父先子后，对每个模块
#        module.process_weights_after_loading()          Linear：在线量化（若配置）+ AITER GEMM 要的 shuffle / pad
#        quant_method.process_weights_after_loading(m)   MoE 量化方法：把专家权重重排成 AITER fused-MoE kernel 的布局
#        FusedMoEMethodBase.init_prepare_finalize(m)      MoE 的 token dispatch / combine 初始化（第 7 批）
#     ▼
#   能 forward 的模型。之后 ModelRunner 才 warmup、测显存、算 KV block 数——warmup_model 开头会 reset 显存峰值，
#     所以加载期的峰值不影响 KV 预算，只决定"加载过程会不会 OOM"
#
# ---------------------------------------------------------------------------------------------------------
# 【例】一个张量走一遍：Qwen3-8B（36 层），TP = 4，checkpoint 张量
#   model.layers.3.self_attn.k_proj.weight   [1024, 4096] bf16（8 个 KV 头 × 128 = 1024 行）
#   1 读：weight_iterator 从它所在的分片 yield 出 (名字, CPU 上的 [1024, 4096])；4 个 rank 各读一遍同一个张量
#     （同一节点上共享 page cache，磁盘只读一次）
#   2 改名：CheckpointNameRewriter.rewrite——Qwen3 没声明 WeightsMapper / weights_mapping；名字不含 mtp、kv_scale、
#     inv_freq；第 3 层 < num_hidden_layers(36)；不是共享专家 → 原样返回（不是 None，所以迭代器才会把它读出来）
#   3 分发：WeightDispatcher._dispatch_packed 按 packed_modules_mapping 的顺序找子串："q_proj" 不在名字里，"k_proj" 在 →
#     值是元组 ("qkv_proj", "k") → param_name = "model.layers.3.self_attn.qkv_proj.weight"，shard_id = "k"
#     → model.get_parameter(param_name) → submit(param.weight_loader, param, tensor, "k")，交给线程池
#   4 写入：QKVParallelLinear.weight_loader(param, tensor, "k")（linear.py）。本 rank 的融合参数是
#       qkv_proj.weight [1536, 4096] = q 8 头 × 128（1024 行）+ k 2 头 × 128（256 行）+ v 2 头 × 128（256 行）
#     "k" 分支：shard_size = num_kv_heads(2) × 128 = 256；shard_offset = num_heads(8) × 128 = 1024；
#       shard_rank = tp_rank // num_kv_head_replicas(= 1，8 个 KV 头够 4 张卡分)
#     param.narrow(0, 1024, 256) ← tensor.narrow(0, 256 × tp_rank, 256)，再由 weight_loader_process 核对 dtype 后 copy_（H2D）
#
#       checkpoint k_proj.weight [1024, 4096]               rank r 的 qkv_proj.weight [1536, 4096]
#         行    0 ~  255   KV 头 0、1 ──► rank 0            行    0 ~ 1023   q：本卡 8 个 q 头（q_proj 的第 1024r ~ 1024r+1023 行）
#         行  256 ~  511   KV 头 2、3 ──► rank 1   ═════►   行 1024 ~ 1279   k：本卡 2 个 KV 头   ◄── 本例写的就是这 256 行
#         行  512 ~  767   KV 头 4、5 ──► rank 2            行 1280 ~ 1535   v：本卡 2 个 KV 头（v_proj 同样的切法）
#         行  768 ~ 1023   KV 头 6、7 ──► rank 3
#
#     例如 rank 2：checkpoint 的第 512 ~ 767 行 → 自己 qkv_proj 的第 1024 ~ 1279 行
#   5 记账：_record 把 param_name 记进 loaded_weights_record；q_proj、v_proj 到达时写同一个参数的另外两段
#   6 加载后处理：bf16 不量化时 LinearBase.process_weights_after_loading 不改它（只有量化权重才 shuffle 成 AITER 的布局）；
#     配了在线量化时，在这一步才把整个 [1536, 4096] 量化成目标格式
#   变体：KV 头数少于 tp 时（Llama-3-8B 的 8 个 KV 头、TP = 16；llama.py 允许，qwen3.py 断言 KV 头数能被 tp 整除、不允许）
#     每卡放 1 个 KV 头、两张卡共用一个：num_kv_head_replicas = 2，shard_rank = tp_rank // 2，
#     rank 0 和 rank 1 取的都是 checkpoint 的第 0 个 KV 头
#   MoE 专家的对应例子（per expert 路径、专家暂存）见 weight_dispatch.py 文件头
#
# ---------------------------------------------------------------------------------------------------------
# 【要点①】为什么用线程池并行加载（ATOM_LOADER_NUM_THREADS，默认 16；实现在 loading_core.py）
#   一个张量的写入 = narrow 出本 rank 的一片 + 从 CPU 拷到 GPU。主线程只"读名字、改名、找参数、提交"，
#   真正的 weight_loader 在工作线程里跑：
#     ・mmap 路径下 get_tensor 拿到的是映射在文件上的张量，真正读盘发生在 copy_ 第一次碰到这些页的时候（缺页）——
#       多个线程同时 copy_，就同时有多个读请求在途，冷启动时更能把磁盘带宽用满
#     ・从普通（pageable）内存拷到 GPU 对调用线程是同步的，而 PyTorch 的 Python 绑定在执行算子（包括 copy_）时释放 GIL，
#       所以多个线程能真正并行
#   规模（数字来自 loading_core.py 的源码注释）：DeepSeek-V3 / R1 每层 256 个专家 × 3 个投影，加上同样多的 scale，
#     全模型约 9 万个专家张量
#   代价：在途张量和暂存缓冲多占主机内存；设为 1 退回逐个顺序写入（也就关掉了下面的专家暂存），便于排查
#
# 【要点②】专家暂存（expert_staging.py）
#   逐专家的张量很小：V3、TP = 8 时每个专家的 gate_proj 本 rank 只有一片 [256, 7168] FP8 = 1.75 MiB，一个个 H2D 是
#     延迟受限而不是带宽受限的。于是先拷进一块和整个堆叠参数一样大的 pinned（页锁定）CPU 缓冲，等本 rank 的路由专家
#     都到齐（w13 要"本 rank 路由专家数 × 2"份 = gate + up，w2 要"路由专家数"份）再一次大拷贝写回 GPU
#   例：V3、TP = 8，一层 w13_weight 的路由专家部分本 rank 有 256 × 512 × 7168 字节 ≈ 0.875 GiB（FP8）→ 512 次小拷贝变成 1 次
#   只在开线程池时启用（loading_core.py：batching_enabled = executor is not None）
#
# 【要点③】流式在线量化（online_quant_streaming.py；ATOM_ONLINE_QUANT_STREAMING=1 才开，默认关）
#   在线量化（--online_quant_config；概念见 arg_utils.py online_quant_config 上方【科普】）默认"先全部加载、再逐模块量化"：
#     加载期显存峰值 ≈ 全部源格式权重。流式改成"一个模块的权重到齐 → 搬上 GPU → 量化 → 立刻释放源权重"：
#     峰值 ≈ 全部目标格式权重 + 少量在途模块
#   例（按参数量估算，非实测；配置取自 docs/online_quantization_streaming_guide.md）：DeepSeek-R1 FP8 checkpoint、-tp 8、
#     专家在线转 MXFP4。路由专家共 58 层 × 256 × 3 × 7168 × 2048 ≈ 6539 亿个参数，每卡 1/8 ≈ 817 亿个：
#       FP8 每个 1 字节 ≈ 81.7 GB；MXFP4 每个 4 + 8/32 = 4.25 位 ≈ 43.4 GB
#     非流式：每卡先装下约 82 GB 的 FP8 专家再量化；流式：约 43 GB + 在途的几层（每层每卡约 1.4 GB FP8）
#   代价：不省时间（量化计算量不变，只是和读盘重叠）；TP 下直接量化本地分片，scale 可能和非流式 / 离线略有不同；
#     有些模块认领不了，退回加载后处理（这些模块不省显存；日志会报 fallback 数）
#
# 【要点④】假权重（--load_dummy，见 arg_utils.py）
#   不读 checkpoint（loading_core.py 让迭代器的"想要"判断恒为 False），参数保持构造时的样子：
#     empty（只写 --load_dummy）= torch.empty 的随机内容，可能有 NaN / Inf；zero / xavier 由 initialize_dummy_weights 填有限值
#   用途：没有 checkpoint、只测性能 / 显存 / 启动流程；生成的文字没有意义
#
# ---------------------------------------------------------------------------------------------------------
# 对比nano-vllm（utils/loader.py，约 30 行）：
#   相同的骨架：遍历 *.safetensors → 名字命中 packed_modules_mapping 就换成融合参数名并带上 shard id →
#     model.get_parameter → param.weight_loader(param, tensor[, shard_id])；没有专属 loader 的参数用 default_weight_loader
#   ATOM 多出来的：
#     读盘     只读 header 跳过无关分片、按 index.json 过滤、后台预读 page cache、可选不用 mmap      weight_iterator.py
#     改名     9 条规则（模型声明的映射、MTP、共享专家融合、丢掉超出层数的层……）                   weight_names.py
#     分发     MoE 专家的几条路径（逐专家 / 整层堆叠 / merged），packed 只是其中一条                weight_dispatch.py
#     写入     线程池并行 + 专家暂存合并 H2D；流式在线量化                                        loading_core / expert_staging / online_quant_streaming
#     检查     缺失分片预检、缺失专家报错、"没加载的参数 / 被丢弃的张量"双向告警                     loading_core.py
#     后处理   process_weights_after_loading（在线量化、AITER 布局重排）、MoE dispatch 初始化       本文件 load_model
#     其他     假权重、插件入口
#   nano-vllm 名字对不上时 get_parameter 直接抛异常；ATOM 把它记为"被丢弃"，加载完统一告警，不中断加载
# ---------------------------------------------------------------------------------------------------------

import json
import logging
import os
import time

import safetensors
import safetensors.torch
import torch
from torch import nn
from transformers import AutoConfig

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `safetensors.torch.load(bytes)` path used when `ATOM_DISABLE_MMAP=true`
# raises `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
# 【译】safetensors<=0.7.0 的 Python `_TYPES` 字典缺少 `F8_E8M0`（MX 格式的 scale）这一项，尽管 torch 和 safetensors 的
#   Rust 二进制都支持它。走 mmap 的 `safe_open` 路径经过 Rust，没有问题；但 `ATOM_DISABLE_MMAP=true` 时用的
#   `safetensors.torch.load(bytes)` 路径会在 DeepSeek-V4-Pro 的分片上抛 `KeyError: 'F8_E8M0'`。
#   这里补登记这个 dtype 字符串，让两条路径行为一致
# E8M0：只有 8 位指数、没有尾数，只能表示 2 的整数次幂；MXFP4 / MXFP8 每 32 个元素共用一个这样的 scale
#   （格式见 arg_utils.py online_quant_config 上方【科普】）。weight_iterator.py 开头有同样的一段，两处都幂等
# 注：v0.1.8 不用 mmap 时实际走 weight_iterator._read_shard，它同样经 `_TYPES` 查 dtype（header 读不出来才退回 safetensors.torch.load）
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu

from aiter.dist.parallel_state import get_tp_group

from atom.model_loader.loading_core import load_weights_into_model, rank_tag
from atom.model_loader.online_quant_streaming import OnlineQuantStreamer
from atom.model_loader.weight_iterator import (
    safetensors_weights_iterator,
)

# Re-exported so the many `from atom.model_loader.loader import WeightsMapper`
# call sites (models, vLLM/SGLang/RTP-LLM plugins) keep working.
# 【译】在这里重新导出，让大量 `from atom.model_loader.loader import WeightsMapper` 的调用点
#   （各模型、vLLM / SGLang / RTP-LLM 插件）继续可用
from atom.model_loader.weight_names import WeightsMapper, WeightsMapping  # noqa: F401
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.moe import FusedMoEMethodBase
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.plugin.prepare import is_sglang
from atom.utils import envs

logger = logging.getLogger("atom")


# 报错用：返回"`参数名`（checkpoint `原名`）"这样的描述；调用方只有下面 default_weight_loader 的两条报错
def _weight_identity(param: nn.Parameter) -> str:
    """Name the weight a loader was handed, for failures inside one.

    `WeightDispatcher._parameter` records the runtime and checkpoint names on
    the parameter, since loaders take `(param, tensor)` and would otherwise have
    only shapes to report.
    """
    # 【译】给出 loader 拿到的是哪个权重，用于 loader 内部出错时的报错信息。
    #   `WeightDispatcher._parameter` 把运行时参数名和 checkpoint 名记在参数上，因为 loader 只接收 `(param, tensor)`，
    #   不这样做的话报错时只能报形状
    names = getattr(param, "_atom_load_names", None)
    if names is None:
        return "an unnamed parameter"
    param_name, ckpt_name = names
    if ckpt_name == param_name:
        return f"`{param_name}`"
    return f"`{param_name}` (checkpoint `{ckpt_name}`)"


# 默认的 weight_loader：参数自己没有 weight_loader 属性时用它
#   怎么接上的：load_model 把它作为 default_weight_loader 参数传给 loading_core → WeightDispatcher，后者在 generic / merged
#     路径上 getattr(param, "weight_loader", default_weight_loader)。典型使用者：RMSNorm 的 weight 等各 rank 存完整一份的小参数
#   三种情况：
#     ① 元素数相同 → 整体拷贝（就是 nano-vllm 的 default_weight_loader）
#     ② checkpoint 元素数 ÷ tp = 参数元素数 → 当作按 TP 均分，取本 rank 那一段"展平后的连续片"
#        例：checkpoint [64]、TP = 4、参数 [16] → rank 2 取 view(-1)[32:48]
#        只对"按第 0 维切、切完仍是连续一段"的张量成立；而 copy_ 要求形状可广播，把 [n] 拷进 [a, b] 只有 a = 1 时成立，
#        所以实际上只有 1 维参数能走通，多维参数会落进下面的 RuntimeError（提示它需要真正的分片 loader）
#     ③ 都不是 → 直接报错，而不是悄悄留着初始化值
#   对比nano-vllm：nano-vllm 只有 ①，形状不符时由 copy_ 自己报错；TP 切分全部交给各层的 weight_loader
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if loaded_weight.numel() == param.data.numel():
        param.data.copy_(loaded_weight)
    elif loaded_weight.numel() // get_tp_group().world_size == param.data.numel():
        loaded_weight_per_rank = loaded_weight.numel() // get_tp_group().world_size
        # Offset MUST use the TP-group-local rank (rank_in_group), NOT the global
        # rank: `.world_size` above is the TP group size, so the two must be from
        # the same (TP-group) frame. `.rank` is torch.distributed.get_rank()
        # (global). They coincide only when world == tp (pure TP); under PCP/DP/PP
        # the world splits into multiple TP groups, so a group's global ranks
        # (e.g. PCP rank 1 = global 4..7) exceed its world_size (4), making this
        # slice out of bounds → empty → copy_ fails.
        # 【译】偏移必须用 TP 组内的 rank（rank_in_group），不能用全局 rank：上面的 `.world_size` 是 TP 组的大小，两者必须出自
        #   同一个（TP 组的）坐标系。`.rank` 是 torch.distributed.get_rank()（全局）。只有 world == tp（纯 TP）时两者才相同；
        #   在 PCP / DP / PP 下 world 被分成多个 TP 组，一个组的全局 rank（例如 PCP rank 1 = 全局 4..7）会超过它的
        #   world_size（4），切片越界 → 得到空张量 → copy_ 失败
        tp_rank_start = loaded_weight_per_rank * get_tp_group().rank_in_group
        tp_rank_end = tp_rank_start + loaded_weight_per_rank
        try:
            param.data.copy_(loaded_weight.view(-1)[tp_rank_start:tp_rank_end])
        except RuntimeError as error:
            # The element counts matched, so this reached the flat-slice path,
            # but the destination is not laid out as one contiguous rank-major
            # run -- it needs a real sharded loader, or no sharding at all.
            # 【译】元素数对上了，所以走到了展平切片这条路；但目标参数并不是"按 rank 排成的一段连续区间"——
            #   它需要一个真正的分片 loader，或者根本不该切分
            raise RuntimeError(
                f"default_weight_loader: flat TP slice does not fit "
                f"{_weight_identity(param)}. param shape={tuple(param.shape)} "
                f"dtype={param.dtype} numel={param.data.numel()}; loaded "
                f"shape={tuple(loaded_weight.shape)} dtype={loaded_weight.dtype} "
                f"numel={loaded_weight.numel()}; tp_size="
                f"{get_tp_group().world_size} rank_in_group="
                f"{get_tp_group().rank_in_group} slice=[{tp_rank_start}:"
                f"{tp_rank_end}]"
            ) from error
    else:
        # Shape mismatch we cannot resolve — leaving the destination at its init
        # value is almost always a bug. The post-load check in load_model() will
        # catch this and warn (param will be in `unloaded` set since this loader
        # never wrote to it). Raise here so the failure is loud at copy time
        # too, instead of being masked by the default ones-init of RMSNorm etc.
        # 【译】无法解决的形状不匹配——让目标参数停在初始化值几乎总是 bug。load_model() 加载后的检查会发现并告警
        #   （这个 loader 从没写过它，所以参数会出现在 `unloaded` 集合里）。这里抛异常，让失败在拷贝时就大声暴露，
        #   而不是被 RMSNorm 等默认的全 1 初始化掩盖
        # 观察：括号里的说法与 v0.1.8 的实现不符——WeightDispatcher 在 submit 之后立刻 _record（不论 loader 成功与否），
        #   所以参数不会出现在 unloaded 里；而且这里一抛异常整个加载就中止（多线程时由 loading_core 排空 futures 时的
        #   future.result() 重新抛出），走不到覆盖率检查。真正起作用的就是这个 raise
        raise RuntimeError(
            f"default_weight_loader: shape mismatch on {_weight_identity(param)} — "
            f"param={tuple(param.shape)} "
            f"loaded={tuple(loaded_weight.shape)}. Cannot copy."
        )


# when plugin mode, model loader method is bind to model implementation
# thus call this interface to load the model, which leverages the load_model
# method
# 【译】插件模式下，模型加载方法绑定在模型实现上，所以调用这个接口来加载模型，它内部复用 load_model
# 【分支】插件模式（vLLM / SGLang / RTP-LLM，第 10 批）：宿主框架调模型的 load_weights(weights) 时，ATOM 的模型丢掉宿主
#   传进来的权重迭代器，改调这里、用 ATOM 自己的加载器读 checkpoint（见 qwen3.py Qwen3ForCausalLM.load_weights 的注释）
#   调用方：qwen3.py；plugin/vllm/model_wrapper.py、vllm/models/{qwen3_5, kimi_k25, kimi_k3}.py；
#     plugin/sglang/models/{base_model_wrapper, kimi_k25, qwen3_5, qwen4_exp}.py；plugin/rtpllm/models/{glm5, qwen3_5, qwen4_exp}.py
#   与原生路径的区别：模型路径从宿主的 model_config 取（vLLM model / SGLang model_path / RTP-LLM ckpt_path）；
#     前后各 empty_cache 一次；prefix 让记录的名字带上宿主的前缀；返回 loaded_weights_record 给宿主核对"权重是否都加载了"
#   hf_config_override / model_name_or_path_override：vLLM 插件加载草稿模型时用（model_wrapper.py 传草稿的 config 与路径）
def load_model_in_plugin_mode(
    model,
    config,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
    spec_decode: bool = False,
    hf_config_override: AutoConfig | None = None,
    model_name_or_path_override: str | None = None,
) -> set[str]:

    # during loading model, the outplace operation may consume more
    # GPU mem, which cached in torch caching allocator, here actively
    # call empty cache to free the extra reserved but not used memory
    # 【译】加载模型时，非原地（out-of-place）操作可能多占显存，这些显存留在 torch 的缓存分配器里；
    #   这里主动调 empty_cache，释放"已预留但没在用"的多余显存
    #   （缓存分配器里的空闲块对驱动来说仍是本进程占用的显存，之后用 mem_get_info 看剩余显存的一方会少算）
    def _empty_cache():
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    assert (
        config.plugin_config is not None and config.plugin_config.is_plugin_mode
    ), "ATOM is not running in plugin mode"
    if model_name_or_path_override is not None:
        model_name_or_path = model_name_or_path_override
    elif config.plugin_config.is_vllm:
        model_name_or_path = config.plugin_config.model_config.model
    elif config.plugin_config.is_sglang:
        model_name_or_path = config.plugin_config.model_config.model_path
    elif config.plugin_config.is_rtpllm:
        model_name_or_path = config.plugin_config.model_config.ckpt_path

    # 先清一次缓存；再选加载用的 hf_config：多模态模型只用 text_config（与原生 get_hf_config 把 text_config 当 hf_config
    #   一致），load_model 只从它读层数和专家数（loading_core.py）
    _empty_cache()
    if hf_config_override is not None:
        config_for_loading = getattr(
            hf_config_override, "hf_config", hf_config_override
        )
        if hasattr(config_for_loading, "text_config"):
            config_for_loading = config_for_loading.text_config
    else:
        config_for_loading = (
            config.hf_config.text_config
            if hasattr(config.hf_config, "text_config")
            else config.hf_config
        )
    loaded_weights_record = load_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=config_for_loading,
        load_dummy=config.load_dummy,
        spec_decode=spec_decode,
        prefix=prefix,
        is_plugin_mode=True,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
    )
    _empty_cache()
    return loaded_weights_record


# 在线量化的记录文件：每层量化前后的格式等（各模块的 _online_quant_info）+ 原始在线量化配置 + 耗时 + 显存峰值
#   只在 TP 组内 rank 0 写；目录是 ATOM_TORCH_PROFILER_DIR，没设就是当前工作目录——每次带在线量化的启动都会在 cwd
#   留下一个 online_quant_info_<时间>.json。调用方：load_model 末尾（只在有在线量化时）
def _save_online_quant_info(
    oq_layers: list[dict],
    model_name_or_path: str,
    elapsed_seconds: float,
    online_quant_config: dict,
    timing_scope: str,
    peak_gpu_memory_gb: float | None,
):
    """Save online quantization info to a JSON file (rank 0 only)."""
    # 【译】把在线量化信息保存到一个 JSON 文件（只在 rank 0）
    if get_tp_group().rank_in_group != 0:
        return
    output_dir = envs.ATOM_TORCH_PROFILER_DIR or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_ns = time.time_ns() % 1_000_000_000
    filepath = os.path.join(
        output_dir, f"online_quant_info_{timestamp}_{timestamp_ns:09d}.json"
    )

    payload = {
        "model": model_name_or_path,
        "online_quant_config": online_quant_config,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timing_scope": timing_scope,
        "peak_gpu_memory_gb": (
            round(peak_gpu_memory_gb, 3) if peak_gpu_memory_gb is not None else None
        ),
        "num_layers": len(oq_layers),
        "layers": oq_layers,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Online quantization info saved to %s", filepath)


# Dummy-weight init constants (see initialize_dummy_weights).
# 【译】假权重初始化用的常量（见 initialize_dummy_weights）
_DUMMY_WEIGHT_STD = 2.0**-4  # ~0.0625, a plausible transformer weight magnitude
_FP4_UNIT_BYTE = 0x22  # e2m1 fp4x2: both nibbles = 0b0010 = 1.0
_E8M0_UNIT_CODE = 123  # e8m0 exponent code for 2^(123-127) = 2^-4 = _DUMMY_WEIGHT_STD
# 【译】上面三行的行尾注释：约 0.0625，是 transformer 权重合理的量级；e2m1 的 fp4x2：两个半字节都是 0b0010 = 1.0；
#   e8m0 指数编码 123 表示 2^(123-127) = 2^-4 = _DUMMY_WEIGHT_STD
# 三者是配套的：fp4 权重 1.0 × e8m0 scale 2^-4 = 0.0625；fp8 权重 1.0 × 浮点 scale 0.0625 = 0.0625——反量化后的有效权重
#   都等于 _DUMMY_WEIGHT_STD，和 bf16 用 xavier_uniform_ 初始化的量级相当（[4096, 4096] 时上界 √(6/8192) ≈ 0.027）
#   E2M1 的 0b0010：符号 0、指数位 01、尾数 0 → 2^(1-1) × 1.0 = 1.0；E8M0 的指数偏置是 127


# --load_dummy 的 zero / xavier 模式：把参数原地填成有限值（empty 模式不调用它，参数保持 torch.empty 的随机内容）
# 调用方：load_model（在加载循环之后、加载后处理之前；假权重时加载循环一个张量都不读，见 loading_core.py）
# 为什么要在"后处理之前"填：后处理会对量化权重做 shuffle / swizzle（按 AITER kernel 的布局重排字节），
#   先填好常数，被重排的就是干净的常数；而一堆相同字节无论怎么重排都还是它自己
# 为什么量化权重填常数而不是随机数：常数保证 forward 里每个反量化后的权重都是 0.0625、激活不溢出成 inf / NaN；
#   docstring 也说明只有 FP4（MXFP4）这条是验证过的，FP8 等只保证有限、不保证分布像真实权重
def initialize_dummy_weights(model: nn.Module, mode: str) -> None:
    """Fill skipped-load (``--load_dummy``) params with finite values in place.

    ``mode="zero"``   -> every param zeroed (works for fp4/fp8/int/bf16 alike).
    ``mode="xavier"`` -> constant-magnitude init that keeps the forward finite and
    roughly at real-weight scale:

    - bf16/fp16/fp32 2D weight        -> ``xavier_uniform_``
    - 1D norm weight (non-bias)        -> 1.0
    - bias                             -> 0.0
    - float weight_scale              -> ``_DUMMY_WEIGHT_STD``
    - input_scale                      -> 1.0
    - fp8 packed weight               -> 1.0
    - fp4x2 packed weight (uint8-view) -> ``_FP4_UNIT_BYTE`` (each fp4 = 1.0)
    - e8m0 (uint8) block scale        -> ``_E8M0_UNIT_CODE`` (= 2^-4)

    Quantized weights are filled with a *constant* magnitude (not a true random
    distribution), so the effective weights survive the shuffle/swizzle in each
    quant method's ``process_weights_after_loading`` (a permutation of identical
    bytes is a no-op). FP4 (MXFP4) is the validated path; FP8 and other formats
    are made finite but not distribution-realistic.
    """
    # 【译】把跳过加载（``--load_dummy``）的参数原地填成有限值。
    #   ``mode="zero"``   -> 所有参数清零（对 fp4/fp8/int/bf16 都适用）。
    #   ``mode="xavier"`` -> 固定幅值的初始化，让 forward 保持有限、且大致处在真实权重的尺度：
    #     - bf16/fp16/fp32 的 2 维权重        -> ``xavier_uniform_``
    #     - 1 维 norm 权重（不是 bias）        -> 1.0
    #     - bias（偏置）                       -> 0.0
    #     - 浮点 weight_scale                 -> ``_DUMMY_WEIGHT_STD``
    #     - input_scale（静态激活 scale）       -> 1.0
    #     - fp8 打包权重                      -> 1.0
    #     - fp4x2 打包权重（按 uint8 看）      -> ``_FP4_UNIT_BYTE``（每个 fp4 = 1.0）
    #     - e8m0（uint8）块 scale             -> ``_E8M0_UNIT_CODE``（= 2^-4）
    #   量化权重填的是*固定*幅值（不是真正的随机分布），所以有效权重能经受住各量化方法 ``process_weights_after_loading``
    #   里的 shuffle/swizzle（相同字节的任意排列都是无操作）。FP4（MXFP4）是验证过的路径；FP8 和其他格式只保证有限，
    #   不保证分布接近真实
    for name, param in model.named_parameters():
        data = param.data
        if mode == "zero":
            # zero_() is the fast path: valid for every shape and every
            # standard dtype (fp8/int/bf16/...). Packed sub-byte dtypes
            # (e.g. Float4_e2m1fn_x2) have no CUDA fill kernel, so zero_()
            # raises ("fill_cuda" not implemented for that dtype); fall back
            # to zeroing the raw bytes instead (all-zero bytes == zero-valued
            # weights). Only packed weights reach the fallback, and those are
            # contiguous and >=1D, so the uint8 view is always valid there.
            # 【译】zero_() 是快速路径：对任何形状、任何标准 dtype（fp8/int/bf16/...）都有效。打包的亚字节 dtype
            #   （例如 Float4_e2m1fn_x2）没有 CUDA 的 fill kernel，zero_() 会抛异常（"fill_cuda" 对该 dtype 未实现）；
            #   这时退回到把原始字节清零（全 0 字节 == 值为 0 的权重）。只有打包权重会走到退路，而它们都是连续的、
            #   至少 1 维，所以那里的 uint8 view 总是合法的
            try:
                data.zero_()
            except (NotImplementedError, RuntimeError):
                data.view(torch.uint8).zero_()
            continue
        # mode == "xavier"
        # 【译】以下是 mode == "xavier"
        # 按参数名和 dtype 分派，顺序有意义（"input_scale" 要在 "scale" 之前判断）；各分支的行尾注释依次是：
        #   e8m0 块 scale（fp4 用）、fp8/bf16 的浮点 scale、1 维 norm 权重等、fp8 打包权重、fp4x2 打包权重（可按 uint8 看）
        dt = data.dtype
        if "input_scale" in name:
            data.fill_(1.0)
        elif "scale" in name:
            if dt == torch.uint8:  # e8m0 block scale (fp4)
                data.fill_(_E8M0_UNIT_CODE)
            else:  # fp8/bf16 float scale
                data.fill_(_DUMMY_WEIGHT_STD)
        elif dt in (torch.float32, torch.float16, torch.bfloat16):
            if data.dim() >= 2:
                nn.init.xavier_uniform_(data)
            elif "bias" in name:
                data.zero_()
            else:  # 1D norm weight etc.
                data.fill_(1.0)
        elif dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
            data.fill_(1.0)  # fp8 packed weight
        else:  # fp4x2 packed weight, viewable as uint8
            data.view(torch.uint8).fill_(_FP4_UNIT_BYTE)


# 【主干】加载入口
#   调用方：ModelRunner._build_and_load_model（model_runner.py，目标模型）；spec_decode/drafter.py 的 load_model（草稿模型，
#     第 5 个位置参数 spec_decode = not standalone_ckpt：MTP / V4 DSpark 的权重在目标 checkpoint 里）；
#     load_model_in_plugin_mode（插件）；SGLang 插件的几个 nextn wrapper（直接调，spec_decode=True）
#   参数：
#     model：已按 config 构造好的模型。ModelRunner 在 torch.set_default_device(GPU) 下构造，所以参数已在 GPU 上、形状已按 TP
#       切好、值无意义；流式在线量化的参数例外，建在 meta 设备上、还没有存储（见 online_quant_streaming.py）
#     model_name_or_path：本地目录或 HF hub id（hub id 会先下载，见 weight_iterator.py）
#     hf_config：只用来取层数、专家数（loading_core.py）
#     load_dummy：None / "empty" / "zero" / "xavier"（--load_dummy）
#     spec_decode：是否在加载草稿模型（改名规则随之变化：保留 mtp 权重、不按层数截断，见 weight_names.py）
#     prefix：插件宿主给模型参数名加的前缀，只用于记录名
#     weights_mapper / load_fused_expert_weights_fn：显式传入的改名器 / 整层专家加载函数
#       （ModelRunner 传 model.load_fused_expert_weights，只有 Qwen3.5 系列定义了它）
#   返回：写入过的参数名集合。原生目标模型不看它；草稿模型用它判断 embed / lm_head 是否随 checkpoint 加载、要不要共享目标
#     模型的（drafter.py）；插件宿主用它核对
#   步骤：
#     1 （可选）创建流式在线量化器                       OnlineQuantStreamer.maybe_create
#     2 读 checkpoint、改名、分发、写入                    loading_core.load_weights_into_model
#     3 （假权重 zero / xavier）填常数                     initialize_dummy_weights
#     4 （流式）收尾没被"到齐即量化"认领的模块              replay_stragglers_and_report
#     5 加载后处理：逐模块 process_weights_after_loading 等
#     6 日志：后处理耗时、在线量化记录文件、加载期显存峰值
def load_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: str | None = None,
    spec_decode: bool = False,
    prefix: str = "",
    is_plugin_mode: bool = False,
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
):
    """Load a checkpoint into `model` and run post-load weight processing.

    The checkpoint -> parameter logic lives in `loading_core`, which is kept
    free of AITER so it can be unit-tested without a GPU build; this wrapper
    supplies the pieces that do need AITER (TP group, quant-config-driven
    shared-expert fusion) and owns everything that happens after the weights
    have landed.

    `is_plugin_mode` is unused and kept for call-site compatibility.
    """
    # 【译】把 checkpoint 加载进 `model`，并做加载后的权重处理。
    #   "checkpoint → 参数"的逻辑在 `loading_core` 里，那里刻意不依赖 AITER，好在没有 GPU 构建的环境里做单元测试；
    #   这个包装函数提供需要 AITER 的那几样东西（TP 组、由量化配置决定的共享专家融合），并负责权重落地之后的一切。
    #   `is_plugin_mode` 已不再使用，保留只是为了调用点兼容

    # 共享专家要不要当作第 n_routed 个专家融合进 FusedMoE 的堆叠张量（weight_names.py 的 _maybe_fuse_shared_expert 问它）：
    #   判断要读量化配置（共享专家和路由专家量化格式相同才能放进同一个张量）和并行配置（DP + MoRI 等组合不融合），
    #   都在 topK.py（第 7 批）。优先用模型自己的 quant_config（有的模型包了一层或重建过，见 config.py
    #   QuantizationConfig 文件头"生命周期"第 1 步），没有再用全局的
    def _fuse_shared_expert(
        shared_expert_prefix: str, routed_expert_prefix: str
    ) -> bool:
        model_quant_config = getattr(
            getattr(model, "atom_config", None), "quant_config", None
        )
        if model_quant_config is None:
            model_quant_config = getattr(model, "quant_config", None)
        if model_quant_config is not None and hasattr(
            model_quant_config, "get_layer_quant_config"
        ):
            return is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                model_quant_config,
                shared_expert_prefix=shared_expert_prefix,
                routed_expert_prefix=routed_expert_prefix,
            )
        return is_rocm_aiter_fusion_shared_expert_enabled(
            shared_expert_prefix=shared_expert_prefix,
            routed_expert_prefix=routed_expert_prefix,
        )

    def _is_rank0() -> bool:
        # Diagnostics must not be the thing that breaks loading, and the TP
        # group may not exist yet (single-process tools, plugin hosts), so any
        # failure here degrades to "report from this rank".
        # 【译】诊断信息不能成为加载失败的原因，而且 TP 组可能还不存在（单进程工具、插件宿主），所以这里的任何失败
        #   都退化为"由本 rank 报告"
        # 注：`.rank` 是全局 rank（见 default_weight_loader 里的说明），所以 DP / PP 下只有全局 rank 0 报告
        try:
            return get_tp_group().rank == 0
        except Exception:  # noqa: BLE001
            return True

    # 从这里开始统计 PyTorch 分配器的显存峰值，第 6 步报告（覆盖加载 + 后处理）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Quantize eligible modules as soon as their source weights complete.
    # This must also run for speculative draft loads: those modules were built
    # with the same meta-backed streaming parameters as the target. `spec_decode`
    # only changes checkpoint-name selection; it must not disable materialization.
    # 【译】符合条件的模块一旦源权重到齐就立刻量化。草稿模型加载也必须这样做：那些模块和目标模型一样，是用 meta 设备上的
    #   流式参数构造的。`spec_decode` 只改变选哪些 checkpoint 名，不能因此关掉存储的实体化
    # 第 1 步。maybe_create 返回 None 的情况：ATOM_ONLINE_QUANT_STREAMING 没开（默认）、假权重、模型里没有可流式量化的模块
    online_quant_streamer = OnlineQuantStreamer.maybe_create(model, load_dummy)

    # 第 2 步：主循环在 loading_core.py；这里把需要 AITER 的几样东西以回调注入（default_weight_loader、_fuse_shared_expert、
    #   _is_rank0），读盘函数也作为参数传入（单元测试可以换成假的迭代器）
    loaded_weights_record = load_weights_into_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=hf_config,
        load_dummy=load_dummy,
        spec_decode=spec_decode,
        prefix=prefix,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        default_weight_loader=default_weight_loader,
        fuse_shared_expert=_fuse_shared_expert,
        is_rank0=_is_rank0,
        weights_iterator=safetensors_weights_iterator,
        online_quant_streamer=online_quant_streamer,
    )

    # Dummy modes other than "empty" fill the skipped-load params with finite
    # values before post-processing, so shuffle/swizzle runs on clean constants.
    # 【译】"empty" 以外的假权重模式在后处理之前把跳过加载的参数填成有限值，这样 shuffle/swizzle 处理的是干净的常数
    # 第 3 步
    # 观察：流式参数在构造时就建在 meta 上（只看 ATOM_ONLINE_QUANT_STREAMING 和在线量化配置，不看 load_dummy），而假权重时
    #   maybe_create 返回 None、没人把它们实体化；所以"流式 + 在线量化 + --load_dummy"组合下这些参数仍在 meta 上进入后处理
    #   （据代码推断，未在 GPU 上验证）
    if load_dummy and load_dummy != "empty":
        initialize_dummy_weights(model, load_dummy)

    # 第 4 步：还没被认领（没检测到"到齐"）的流式模块在这里统一实体化、重放缓冲的 loader 调用，然后走下面普通的后处理
    #   （这些模块就没有省下显存）
    if online_quant_streamer is not None:
        online_quant_streamer.replay_stragglers_and_report(_is_rank0())

    # 有没有在线量化：模块自己有 online_quant 属性，或它的 quant_config.online_quant 为真（--online_quant_config）。
    #   这里只决定日志措辞和要不要写记录文件，真正的量化发生在下面各模块的 process_weights_after_loading 里
    # streamed_done：加载期间已被流式量化器跑过 process_weights_after_loading 的模块（按 id），下面跳过它们的模块级钩子
    has_online_quant = any(
        getattr(m, "online_quant", False)
        or (
            getattr(m, "quant_config", None) is not None
            and getattr(m.quant_config, "online_quant", False)
        )
        for _, m in model.named_modules()
    )
    streamed_done = (
        online_quant_streamer.done_module_ids
        if online_quant_streamer is not None
        else frozenset()
    )
    stream_candidate_count = (
        len(online_quant_streamer.candidates)
        if online_quant_streamer is not None
        else 0
    )
    stream_fallback_count = stream_candidate_count - len(streamed_done)
    if online_quant_streamer is not None:
        logger.info(
            "[%s] Streaming online quantization: %d/%d eligible modules were "
            "quantized while loading; the post-load pass remains enabled to "
            "quantize %d fallback module(s) and finish weight processing",
            rank_tag(),
            len(streamed_done),
            stream_candidate_count,
            stream_fallback_count,
        )
    elif has_online_quant:
        logger.info(
            "[%s] Post-load online quantization and weight processing started",
            rank_tag(),
        )
    pp_start = time.perf_counter()

    # Parent-first traversal is significant for streaming-deferred children:
    # their parent first combines source weights, then the child's normal hook
    # online-quantizes the final fused weight.
    # 【译】父先子后的遍历顺序对"流式延后"的子模块很重要：它们的父模块先把源权重组合起来，然后子模块自己的普通钩子
    #   再对组合好的最终权重做在线量化
    #   例：Kimi-K3 的 KDA 层（get_streaming_deferred_modules 返回 in_proj、f_a_proj）——父模块的 process_weights_after_loading
    #   把 b_proj、f_a_proj 拼进 in_proj，之后遍历到 in_proj 时才量化拼好的整块。named_modules() 是先序遍历，正好先父后子
    # 第 5 步，对每个模块依次做：
    #   ① module.process_weights_after_loading()：LinearBase = 在线量化（若配置）+ AITER GEMM 要的 shuffle / pad；
    #      FusedMoE = 在线量化 + MoE backend 校验；模型自定义的模块也可以定义它（如上面 Kimi-K3 的融合）
    #   ② quant_method.process_weights_after_loading(module)：量化方法对象（主要是 MoE 的 Fp8 / Mxfp4 … MoEMethod）
    #      把专家权重重排成 AITER fused-MoE kernel 要的布局；流式模块的①已在加载期间做过，②仍在这里做
    #   ③ FusedMoEMethodBase.init_prepare_finalize(module)：建立 MoE 的 token dispatch / combine（EP 的 all-to-all 等，第 7 批）
    #   ④ 在线量化新建的参数记进 loaded_weights_record
    # 对比nano-vllm：nano-vllm 加载完直接能用、没有这一遍——它只跑 bf16，权重布局就是 checkpoint 的布局
    for module_name, module in model.named_modules():
        # Avoid repeating module post-processing already run by the streamer.
        # 【译】流式量化器已经跑过的模块，不要重复模块级的后处理
        if (
            hasattr(module, "process_weights_after_loading")
            and id(module) not in streamed_done
        ):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)

        # when running plugin mode for sglang, don't do the post process here
        # since sglang will call this func automatically after finishing loading
        # 【译】SGLang 插件模式下不在这里做量化方法的后处理，因为 SGLang 加载结束后会自动调用这个函数
        if isinstance(quant_method, QuantizeMethodBase) and not is_sglang():
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)

        # Online quantization creates new params (e.g. weight_scale) that are
        # not present in the source checkpoint. Record them as "loaded" so the
        # plugin host's strict weight tracking (e.g. vLLM's default loader)
        # does not flag them as uninitialized.
        # 【译】在线量化会新建 checkpoint 里没有的参数（例如 weight_scale）。把它们记为"已加载"，免得插件宿主严格的权重核对
        #   （例如 vLLM 的默认 loader）把它们标成未初始化
        if getattr(module, "_online_quant_info", None) is not None:
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                loaded_weights_record.add(prefix + full_name)

    # Post-processing (AITER shuffle/swizzle, per-quant-method hooks) runs inside
    # the load time the caller reports, so it is timed unconditionally: without
    # this line a shuffle-dominated load is indistinguishable from a slow read.
    # 【译】后处理（AITER shuffle/swizzle、各量化方法的钩子）算在调用方报告的加载时间里，所以无条件计时：没有这一行，
    #   一次被 shuffle 拖慢的加载和一次读盘慢的加载就分不出来
    # 第 6 步。max_memory_allocated：从开头 reset 以来 PyTorch 分配器的最高占用（不含缓存分配器预留未用的部分）
    pp_elapsed = time.perf_counter() - pp_start
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / (1 << 30)
        if torch.cuda.is_available()
        else None
    )
    if not has_online_quant:
        logger.info(
            "[%s] Weight post-processing done: %.2f seconds",
            rank_tag(),
            pp_elapsed,
        )
    else:
        # 收集各模块记下的 _online_quant_info，和第一个找到的原始在线量化配置一起写进记录文件
        oq_layers = []
        raw_online_quant_config = None
        for _, module in model.named_modules():
            info = getattr(module, "_online_quant_info", None)
            if info is not None:
                oq_layers.append(info)
            if raw_online_quant_config is None:
                qc = getattr(module, "quant_config", None)
                if qc is not None and hasattr(qc, "online_quant_config_raw"):
                    raw_online_quant_config = qc.online_quant_config_raw
        if online_quant_streamer is not None:
            logger.info(
                "[%s] Post-stream fallback and weight processing done: %.2f "
                "seconds; %d module(s) quantized while loading, %d fallback "
                "module(s) quantized after loading, %d layers online-quantized "
                "in total",
                rank_tag(),
                pp_elapsed,
                len(streamed_done),
                stream_fallback_count,
                len(oq_layers),
            )
            timing_scope = "post_stream_fallback_and_weight_processing"
        else:
            logger.info(
                "[%s] Post-load online quantization and weight processing done: "
                "%.2f seconds, %d layers online-quantized",
                rank_tag(),
                pp_elapsed,
                len(oq_layers),
            )
            timing_scope = "post_load_online_quantization_and_weight_processing"
        _save_online_quant_info(
            oq_layers,
            model_name_or_path,
            pp_elapsed,
            raw_online_quant_config or {},
            timing_scope,
            peak_gpu_memory_gb,
        )

    # Measure both loading and post-processing for comparable peak memory.
    # 【译】峰值显存同时覆盖加载和后处理，这样几种模式的数字才可比
    if peak_gpu_memory_gb is not None:
        if online_quant_streamer is not None:
            logger.info(
                "[%s] Peak GPU memory during streaming weight loading and "
                "online quantization: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )
        elif has_online_quant:
            logger.info(
                "[%s] Peak GPU memory during weight loading and post-load "
                "online quantization: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )
        else:
            logger.info(
                "[%s] Peak GPU memory during weight loading and "
                "post-processing: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )

    return loaded_weights_record
