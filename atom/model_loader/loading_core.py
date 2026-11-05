# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint -> parameter loading, with no GPU or AITER dependency.

`load_model` in `loader.py` is a thin wrapper that binds the host-specific
callables below and then runs post-processing; everything that decides *which*
checkpoint tensor lands in *which* parameter lives here.  The split exists so
this logic is unit-testable on a plain CPU runner — the unit-test gate has no
AITER build, and `loader.py` imports AITER at module level.
"""

# 【译】checkpoint → 参数的加载，不依赖 GPU 或 AITER。
#   `loader.py` 里的 `load_model` 是一层薄包装：它绑定下面这些依赖宿主环境的可调用对象，然后做后处理；决定"哪个
#   checkpoint 张量落进哪个参数"的一切都在这里。这样拆分是为了让这部分逻辑能在普通 CPU runner 上做单元测试——
#   单元测试门禁没有 AITER 构建，而 `loader.py` 在模块级就导入了 AITER
#
# 【导读】loading_core.py：加载主循环。全流程图、TP / 融合参数 / MoE 堆叠的前置知识和一个 k_proj 的完整例子都在
#   loader.py 文件头；本文件是图中"读 → 改名 → 分发 → 收尾 → 检查"这几格
#   主干：load_weights_into_model（A，逐段注释）；辅助：verify_shard_files_present（缺分片预检）、_report_coverage（双向告警）、
#     rank_tag（日志前缀）。唯一调用方：loader.py 的 load_model
#   load_weights_into_model 的结构：
#     1 收集模型声明的改名规则 → 构造 CheckpointNameRewriter（weight_names.py）
#     2 params_dict、专家映射的预索引、MoE 模块缓存、ExpertStagingPool（expert_staging.py）
#     3 线程池（ATOM_LOADER_NUM_THREADS，默认 16）与 _submit（同步 / 线程池 / 经流式量化器 三种提交方式）
#     4 构造 WeightDispatcher（weight_dispatch.py）
#     5 主循环：for name, tensor in weights_iterator(...)：查改名结果 → dispatcher.dispatch
#     6 收尾：排空线程池任务 → 排空流式量化的尾部 → 刷出暂存池（缺专家时报错）→ 关线程池
#     7 分阶段耗时日志 → _report_coverage
#   线程模型：主线程跑迭代器和 dispatch（读名字、改名、找参数、提交），工作线程跑 weight_loader / 暂存（真正的 narrow + copy_）。
#     线程池的任务队列没有上限，主线程会一路跑在工作线程前面；mmap 下排队的只是文件映射上的张量，不占多少内存，
#     ATOM_DISABLE_MMAP=true 时排队的张量引用着整个分片的读缓冲（见 weight_iterator._read_shard），主机内存会多占
# 对比nano-vllm：nano-vllm 的 load_model 是一个两层 for 循环、同步写入；这里拆成"改名器 + 分发器 + 线程池 + 收尾检查"

import concurrent.futures
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.expert_staging import ExpertStagingPool
from atom.model_loader.weight_dispatch import WeightDispatcher
from atom.model_loader.weight_names import (
    CheckpointNameRewriter,
    WeightsMapper,
)
from atom.utils import envs

if TYPE_CHECKING:
    from atom.model_loader.online_quant_streaming import OnlineQuantStreamer

logger = logging.getLogger("atom")


# 日志前缀 rank_N；调用方：loader.py 的各条日志、本文件的阶段耗时日志
def rank_tag() -> str:
    """`rank_N` for log lines, or `rank_?` before the process group exists.

    Mirrors how `weight_iterator` reads the rank: diagnostics must never be the
    thing that breaks a load, and single-process tools have no group at all.
    """
    # 【译】日志用的 `rank_N`，进程组还不存在时是 `rank_?`。
    #   与 `weight_iterator` 读 rank 的方式一致：诊断信息绝不能成为加载失败的原因，而单进程工具根本没有进程组
    # contextlib.suppress(Exception)：with 块里抛出的这些异常被吞掉，相当于 try/except: pass
    with contextlib.suppress(Exception):
        if torch.distributed.is_initialized():
            return f"rank_{torch.distributed.get_rank()}"
    return "rank_?"


# 缺分片预检：只做 stat，不读数据。调用方：load_weights_into_model（非假权重时，主循环之前）
# 例：index.json 的 weight_map 引用了 163 个分片，下载中断只落了 150 个 → 这里直接列出缺的 13 个文件名并报错
def verify_shard_files_present(model_name_or_path: str) -> None:
    """Fail early when the shard index references files that are not on disk.

    An interrupted download leaves a complete ``model.safetensors.index.json``
    next to an incomplete set of shards. Without this check the load happily
    skips every tensor those shards held, and the first symptom is the far
    downstream "MoE parameter(s) did not receive every routed expert" report --
    which reads like a loader or quantization bug rather than a missing file.
    Naming the absent shards here turns that into a one-line diagnosis.

    No-op for single-file checkpoints, for a bare HF repo id (nothing to stat),
    and for an unreadable index (the existing load path already reports that).
    """
    # 【译】分片索引引用了磁盘上不存在的文件时，尽早失败。
    #   下载中断后会留下完整的 ``model.safetensors.index.json``，旁边却只有一部分分片。没有这个检查，加载会若无其事地跳过
    #   那些分片里的所有张量，第一个症状要到很下游才出现——"MoE parameter(s) did not receive every routed expert"的报告，
    #   读起来像 loader 或量化的 bug，而不是缺文件。在这里点名缺失的分片，就变成一行就能诊断的问题。
    #   对单文件 checkpoint、裸的 HF repo id（没有东西可 stat）、读不了的索引（现有加载路径已经会报）不做任何事
    index_path = os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        return
    try:
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, ValueError):
        return
    shards = sorted(set(weight_map.values()))
    missing = [
        s for s in shards if not os.path.isfile(os.path.join(model_name_or_path, s))
    ]
    if not missing:
        return
    shown = "\n  ".join(missing[:20])
    elided = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
    raise FileNotFoundError(
        f"Checkpoint at {model_name_or_path} is incomplete: "
        f"{SAFE_WEIGHTS_INDEX_NAME} references {len(shards)} shard file(s), "
        f"but {len(missing)} of them are absent:\n  {shown}{elided}\n"
        "Re-download the checkpoint -- an interrupted `hf download` is the "
        "usual cause, and re-running it resumes the missing files."
    )


# 【主干】把 checkpoint 的每个张量写进它所属的模型参数（结构见文件头 1~7）。调用方：loader.py 的 load_model
#   前 8 个参数与 load_model 相同（含义见 loader.py 的 load_model 上方）；* 之后是只能按关键字传的回调——
#   这些都依赖 AITER 或宿主环境，本模块不自己导入，由 load_model 注入：
#     default_weight_loader  参数没有自己的 weight_loader 时的兜底（loader.py）
#     fuse_shared_expert     (共享专家前缀, 路由专家前缀) -> 是否融合（loader.py 的 _fuse_shared_expert → topK.py）
#     is_rank0               非 rank 0 不重复打印诊断
#     weights_iterator       (路径, disable_mmap, wants) -> 逐个 (名字, 张量)（weight_iterator.safetensors_weights_iterator）
#     online_quant_streamer  流式在线量化器，跟踪每个模块是否到齐（None = 不流式）
#   返回：写入过的参数名集合（WeightDispatcher.loaded_weights_record，带 prefix）
def load_weights_into_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: str | None = None,
    spec_decode: bool = False,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
    *,
    default_weight_loader: Callable,
    fuse_shared_expert: Callable[[str, str], bool],
    is_rank0: Callable[[], bool],
    weights_iterator: Callable[..., Iterable[tuple[str, torch.Tensor]]],
    online_quant_streamer: "OnlineQuantStreamer | None" = None,
) -> set[str]:
    """Copy every checkpoint tensor into the model parameter it belongs to.

    The four keyword-only callables are the host environment this module
    refuses to import itself:

    - ``default_weight_loader``  fallback copy for params without their own
    - ``fuse_shared_expert``     ``(shared_prefix, routed_prefix) -> fuse?``
    - ``is_rank0``               suppress duplicate diagnostics off rank 0
    - ``weights_iterator``       ``(path, disable_mmap, wants) -> (name, tensor)``

    ``online_quant_streamer`` tracks module completion during loading.
    """
    # 【译】把每个 checkpoint 张量拷进它所属的模型参数。
    #   四个只能按关键字传的可调用对象，是本模块拒绝自己导入的宿主环境：
    #   - ``default_weight_loader``  没有自己 loader 的参数的兜底拷贝
    #   - ``fuse_shared_expert``     ``(shared_prefix, routed_prefix) -> 是否融合``
    #   - ``is_rank0``               非 rank 0 时不重复输出诊断
    #   - ``weights_iterator``       ``(path, disable_mmap, wants) -> (name, tensor)``（逐个产出名字和张量）
    #   ``online_quant_streamer`` 在加载期间跟踪模块是否完整

    # 路由专家数：不同家族的字段名不同（DeepSeek 叫 n_routed_experts，Mixtral 叫 num_local_experts，Qwen 叫 num_experts）；
    #   只给共享专家改名用（experts.{n} 的 n，见 weight_names.py）
    def _n_routed_experts() -> int | None:
        return (
            getattr(hf_config, "n_routed_experts", None)
            or getattr(hf_config, "num_local_experts", None)
            or getattr(hf_config, "num_experts", None)
        )

    # need to record the loaded weight name for vllm load check
    # it is only used in plugin mode for vllm
    # 【译】需要记录已加载的权重名，供 vLLM 做加载检查；只在 vLLM 插件模式下用到
    #   （这两行注释现在对应的是 WeightDispatcher 里的 loaded_weights_record，此处已没有相应的代码）

    # Auto-detect weight mapper from model if not provided explicitly
    # 【译】调用方没显式提供时，从模型上自动探测改名器
    # ── 1 收集模型声明的改名规则 ──
    #   hf_to_atom_mapper：模型类属性，dict 时当作"前缀 → 新前缀"的映射包成 WeightsMapper，也可以直接是 WeightsMapper
    if weights_mapper is None:
        model_mapper = getattr(model, "hf_to_atom_mapper", None)
        if isinstance(model_mapper, dict):
            weights_mapper = WeightsMapper(orig_to_new_prefix=model_mapper)
        elif isinstance(model_mapper, WeightsMapper):
            weights_mapper = model_mapper

    # 其余规则都是模型类上的可选属性（没声明就是空），含义见 weight_names.py 的 CheckpointNameRewriter：
    #   packed_modules_mapping 交给分发器（loader.py 文件头【前置②】）；weights_mapping 是子串替换表；
    #   skip_weight_prefixes 是要丢掉的前缀（如多模态 checkpoint 只跑文本时的视觉塔）；remap_mtp_weight_name 是草稿模型的改名函数
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    skip_weight_prefixes = getattr(model, "skip_weight_prefixes", [])
    mtp_remap = getattr(model, "remap_mtp_weight_name", None)
    # Models can also expose a `weights_mapper` (WeightsMapper instance) for
    # precise prefix/suffix-anchored renames that the dumb substring-substitution
    # `weights_mapping` dict cannot express safely. If both are set they are
    # composed: weights_mapper applies first, then the legacy substring map.
    # 【译】模型也可以暴露一个 `weights_mapper`（WeightsMapper 实例），做前缀 / 后缀锚定的精确改名——这是粗糙的子串替换字典
    #   `weights_mapping` 无法安全表达的。两者都设置时会组合起来：先用 weights_mapper，再用老的子串映射
    # 例：DeepSeek-V4.1 的 weights_mapper 用后缀规则 ".gate.bias" → ".gate.e_score_correction_bias"，因为它还有一个
    #   ".gate.bias_vl"，子串规则会把后者也改坏（deepseek_v41/model.py）
    if weights_mapper is None:
        weights_mapper = getattr(model, "weights_mapper", None)
    # 改名器：9 条规则按固定顺序执行，返回 None 表示这个模型不要该张量（weight_names.py）
    rewriter = CheckpointNameRewriter(
        weights_mapper=weights_mapper,
        weights_mapping=weights_mapping,
        skip_weight_prefixes=skip_weight_prefixes,
        mtp_remap=mtp_remap,
        spec_decode=spec_decode,
        num_hidden_layers=hf_config.num_hidden_layers,
        n_routed_experts=_n_routed_experts(),
        fuse_shared_expert=fuse_shared_expert,
        # Stays False for models without the attribute (GLM4 etc.), so their
        # fused-shared path is unchanged.
        # 【译】对没有这个属性的模型（GLM4 等）保持 False，它们的共享专家融合路径不变
        disable_fused_shared_loading=getattr(
            model, "disable_fused_shared_loading", False
        ),
    )
    # ── 2 参数表、专家映射、暂存池 ──
    # params_dict：全名 → 参数，一次性建好，分发时 O(1) 查（专家路径用它；其余路径用 model.get_parameter）
    #   流式量化器也要拿着它：模块量化完后要把 params_dict 里的旧参数对象的存储释放掉（online_quant_streaming._finalize）
    params_dict = dict(model.named_parameters())
    if online_quant_streamer is not None:
        online_quant_streamer.bind_params_dict(params_dict)
    # Pre-index expert_mapping by weight_name_part for O(1) lookup.
    # Original code does O(N) scan of expert_mapping (768 entries) per tensor,
    # causing ~19s of CPU time for 90k expert tensors. This reduces it to O(1).
    # 【译】按 weight_name_part 预先给 expert_mapping 建索引，查找为 O(1)。原来的代码对每个张量都线性扫描一遍
    #   expert_mapping（768 项），9 万个专家张量要花约 19 秒 CPU 时间；这样降到 O(1)
    # get_expert_mapping() 的每一项是 (param_name_part, weight_name_part, expert_id, shard_id)，例如 DeepSeek-V3：
    #   ("experts.w13_", "experts.17.gate_proj.", 17, "w1")：名字里的 experts.17.gate_proj. 换成 experts.w13_ →
    #   model.layers.3.mlp.experts.17.gate_proj.weight → model.layers.3.mlp.experts.w13_weight，写第 17 个专家的 w1 那一半
    #   （V3：(256 路由 + 1 共享) × 3 = 771 项；源码注释里的 768 = 256 × 3）
    # 观察：命中哪个前缀仍要在 weight_dispatch._dispatch_per_expert 里线性扫描 expert_weight_prefixes（V3 每个张量最多 771 次
    #   子串判断，C 实现，很快）；O(1) 的是命中后取 (param_name_part, expert_id, shard_id) 这一步
    has_expert_mapping = hasattr(model, "get_expert_mapping")
    expert_index = {}  # {weight_name_part: (param_name_part, expert_id, shard_id)}
    expert_weight_prefixes = []  # sorted longest-first for prefix matching
    if has_expert_mapping:
        for (
            param_name_part,
            weight_name_part,
            expert_id,
            shard_id,
        ) in model.get_expert_mapping():
            expert_index[weight_name_part] = (param_name_part, expert_id, shard_id)
        # Sort by length descending so longer (more specific) prefixes match first
        # 【译】按长度降序排序，让更长（更具体）的前缀先匹配
        expert_weight_prefixes = sorted(expert_index.keys(), key=len, reverse=True)

    # Get fused expert mapping from model if it provides one
    # 【译】如果模型提供，就从模型取融合专家的映射
    #   （这行注释与紧接着的代码无关：取融合专家映射的是 WeightDispatcher._detect_fused_expert_format）
    # 下面的缓存：参数全名 → 它所属的模块（FusedMoE 等）。暂存池要通过它调 MoE 模块的暂存协议（expert_staging.py 文件头），
    #   每层的 w13/w2 会收到几百个张量，缓存后只 get_submodule 一次
    moe_module_cache: dict = {}

    def _lookup_moe_module(full_param_name: str):
        module_path = full_param_name.rsplit(".", 1)[0]
        if module_path not in moe_module_cache:
            moe_module_cache[module_path] = (
                model.get_submodule(module_path) if "." in full_param_name else None
            )
        return moe_module_cache[module_path]

    # 专家暂存池：逐专家的小张量先拼进 pinned CPU 缓冲，凑齐一个参数再一次 H2D（loader.py 文件头【要点②】、expert_staging.py）
    staging_pool = ExpertStagingPool(_lookup_moe_module)

    # ── 3 线程池 ──
    # 为什么并行：见 loader.py 文件头【要点①】。流式量化关掉主机暂存时只能单线程（resolve_num_threads 返回 1）；
    #   setup_online_quant_pool 另建流式量化自己的尾部线程池（量化 + 释放源权重）
    num_threads = envs.ATOM_LOADER_NUM_THREADS
    if online_quant_streamer is not None:
        num_threads = online_quant_streamer.resolve_num_threads(num_threads)
        online_quant_streamer.setup_online_quant_pool()
    if num_threads > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    else:
        executor = None
    futures = []

    def _submit(fn, *args):
        # All streamed writes pass through `run` for completion tracking.
        # 【译】所有流式写入都经过 `run`，以便跟踪模块是否完整
        #   （实际上只要有流式量化器，所有写入都先进 run，由 run 判断参数属不属于流式模块）
        # 三种提交方式：
        #   有流式量化器：run(fn, args)——先看参数属不属于流式模块，是就记录写了多少元素、到齐就提交量化；不是就直接 fn(*args)
        #   有线程池：丢进线程池，future 存起来，收尾时统一取结果（工作线程里的异常在那时重新抛出）
        #   都没有：当场同步执行（nano-vllm 的做法）
        if online_quant_streamer is not None:
            if executor is None:
                online_quant_streamer.run(fn, args)
            else:
                futures.append(executor.submit(online_quant_streamer.run, fn, args))
        elif executor is not None:
            futures.append(executor.submit(fn, *args))
        else:
            fn(*args)

    # 流式模块的参数不走专家暂存池（流式量化器有自己的主机暂存），由 batching_excluded 判断
    batching_excluded = None
    if online_quant_streamer is not None:
        batching_excluded = online_quant_streamer.manages_param

    # ── 4 分发器：决定每个改名后的张量交给哪个参数的哪个 loader（weight_dispatch.py 文件头的五条路径）──
    dispatcher = WeightDispatcher(
        model=model,
        params_dict=params_dict,
        hf_config=hf_config,
        prefix=prefix,
        spec_decode=spec_decode,
        submit=_submit,
        staging_pool=staging_pool,
        # The streamer and ExpertStagingPool own disjoint parameters. Streamed
        # parameters use the streamer's host staging so they can be quantized
        # and released as soon as their module completes. Non-streamed expert
        # parameters still need ExpertStagingPool; globally disabling it when a
        # streamer exists turns large per-expert checkpoints into tens of
        # thousands of small loader/H2D operations.
        # 【译】流式量化器和 ExpertStagingPool 管的参数互不相交。流式参数用流式量化器的主机暂存，这样所在模块一到齐就能量化
        #   并释放；非流式的专家参数仍然需要 ExpertStagingPool——只要存在流式量化器就全局关掉它，会把大型的逐专家 checkpoint
        #   变成几万次小的 loader / H2D 操作
        batching_enabled=executor is not None,
        batching_excluded=batching_excluded,
        default_weight_loader=default_weight_loader,
        packed_modules_mapping=packed_modules_mapping,
        expert_index=expert_index,
        expert_weight_prefixes=expert_weight_prefixes,
        has_expert_mapping=has_expert_mapping,
        detect_fused_expert_fn=getattr(model, "detect_fused_expert_format", None),
        get_fused_expert_mapping_fn=getattr(model, "get_fused_expert_mapping", None),
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        on_fused_param=(
            None
            if online_quant_streamer is None
            else online_quant_streamer.materialize_fused_param
        ),
    )

    # Rewriting a name is the same question as "is this tensor wanted", and the
    # iterator asks it before materializing anything -- so answer it once and
    # keep the answer. On a target-model load nearly every tensor is wanted, and
    # rewriting is a dozen substring scans plus a regex per tensor.
    # 【译】"改名"和"这个张量要不要"是同一个问题，而迭代器在实体化任何东西之前就会问——所以只回答一次、把答案存下来。
    #   加载目标模型时几乎每个张量都要，而每个张量的改名是十来次子串扫描加一次正则
    # rewritten：checkpoint 名 → 改名结果（None = 不要）。迭代器用 _wanted 决定读不读这个张量（weight_iterator.py），
    #   主循环再取同一个结果，每个名字只改名一次
    rewritten: dict[str, str | None] = {}

    def _wanted(ckpt_name: str) -> bool:
        if ckpt_name not in rewritten:
            rewritten[ckpt_name] = rewriter.rewrite(ckpt_name)
        return rewritten[ckpt_name] is not None

    # Cheap stat-only preflight: an incomplete download otherwise surfaces as a
    # confusing partial-expert-coverage error thousands of tensors later.
    # Skipped under --load_dummy, which never touches the checkpoint.
    # 【译】只做 stat 的低成本预检：否则一次不完整的下载要到几千个张量之后，才以令人困惑的"专家覆盖不全"错误表现出来。
    #   --load_dummy 时跳过，因为它从不碰 checkpoint
    # 观察："从不碰 checkpoint"不完全准确：假权重时下面仍会调用 weights_iterator——路径不是目录就会 download_weights_from_hf
    #   （hub 上的 *.safetensors 没缓存时会下载），并读每个分片的 header 来判断"有没有想要的张量"（答案恒为没有）
    if not load_dummy:
        verify_shard_files_present(model_name_or_path)

    # Phase timings. The caller reports one aggregate number for the whole load,
    # which cannot tell "the disk is slow" from "the per-tensor dispatch is
    # slow" -- and those have opposite fixes. With a thread pool the read loop
    # races ahead of the work it queues, so a small `read+queue` next to a large
    # `drain` locates the cost in the workers rather than in the read.
    # 【译】分阶段计时。调用方对整个加载只报一个总数，分不清"磁盘慢"和"逐张量分发慢"——两者的解决办法相反。有线程池时，
    #   读取循环会跑到它排队的工作前面，所以小的 `read+queue` 配上大的 `drain`，说明开销在工作线程而不在读取
    num_tensors = 0
    t_read = t_drain = t_quant_drain = t_flush = 0.0

    try:
        disable_mmap = envs.ATOM_DISABLE_MMAP
        # Reject by name before the tensor is materialized. A drafter load reads
        # the whole target checkpoint to pick out the MTP block, so most shards
        # can be skipped without being read at all. Under `--load_dummy` nothing
        # is loaded, so nothing is wanted -- and the rewriter, which is allowed
        # to raise on a checkpoint it cannot map, is never consulted.
        # 【译】在张量实体化之前就按名字拒绝。草稿模型加载会读整个目标 checkpoint 来挑出 MTP 块，所以大部分分片可以根本不读。
        #   `--load_dummy` 下什么都不加载，所以什么都不想要——改名器（遇到无法映射的 checkpoint 时允许抛异常）也就从不被调用
        # ── 5 主循环 ── 每个 (名字, CPU 张量)：取改名结果 → 不要就跳过 → dispatcher.dispatch（找参数、提交写入任务）
        _t0 = time.perf_counter()
        for name, weight_tensor in weights_iterator(
            model_name_or_path,
            disable_mmap,
            (lambda _: False) if load_dummy else _wanted,
        ):
            if load_dummy:
                continue
            _orig_ckpt_name = name  # preserve for ckpt-side coverage report
            # Normally a cache hit: the iterator just asked. Recomputed only if
            # a caller supplied an iterator that ignores the predicate.
            # 【译】通常是缓存命中：迭代器刚问过。只有调用方提供的迭代器不理会这个判断函数时才重新计算
            name = rewritten[name] if name in rewritten else rewriter.rewrite(name)
            if name is None:
                continue
            dispatcher.dispatch(_orig_ckpt_name, name, weight_tensor)
            num_tensors += 1
        t_read = time.perf_counter() - _t0

        # ── 6 收尾 ──
        # 6a 等线程池里所有写入任务完成；future.result() 会把工作线程里的异常（例如 default_weight_loader 的形状错误）在这里重新抛出
        _t0 = time.perf_counter()
        if executor is not None:
            # Drain all tasks (surfacing errors) before the safety flush.
            # 【译】在安全刷出（下面的 flush_pending）之前排空所有任务（让错误暴露出来）
            for future in concurrent.futures.as_completed(futures):
                future.result()
        t_drain = time.perf_counter() - _t0

        # Measure the non-overlapped streaming tail.
        # 【译】统计流式量化中没被重叠掉的尾部耗时
        # 6b 等流式量化的尾部线程把已提交的模块量化完（drain 里同样会重新抛出工作线程的异常）
        if online_quant_streamer is not None:
            _t0 = time.perf_counter()
            online_quant_streamer.drain()
            t_quant_drain = time.perf_counter() - _t0

        loaded_weights_record = dispatcher.loaded_weights_record
        dropped_ckpt_keys = dispatcher.dropped_ckpt_keys

        # Whatever the pool still holds is written back here; anything short of
        # its expected region count means the checkpoint never delivered some
        # routed base experts. The per-parameter check further down is too
        # coarse to see this -- it only knows whether a parameter was touched
        # at all -- so report it while the (slot, shard) detail is still around.
        # 【译】暂存池里还留着的东西在这里写回；任何不足预期区域数的条目，都意味着 checkpoint 没有提供某些路由基础专家。
        #   下面按参数的检查太粗，看不出这一点——它只知道参数有没有被碰过——所以趁 (slot, shard) 的细节还在时报告
        # 6c 例：一层 w13_weight 预期 512 份（256 个专家 × gate/up），只到了 510 份 → 写回已到的 510 份，报告这个参数，
        #   ATOM_LOADER_STRICT_COVERAGE（默认 true）时直接报错
        _t0 = time.perf_counter()
        staging_report = staging_pool.flush_pending()
        t_flush = time.perf_counter() - _t0
        if staging_report.incomplete:
            detail = "\n  ".join(staging_report.incomplete)
            message = (
                f"Batched loader: {len(staging_report.incomplete)} MoE "
                f"parameter(s) did not receive every routed expert from the "
                f"checkpoint:\n  {detail}"
            )
            if envs.ATOM_LOADER_STRICT_COVERAGE:
                raise RuntimeError(
                    f"{message}\nSet ATOM_LOADER_STRICT_COVERAGE=false to load "
                    "anyway, leaving those expert slots at their init values."
                )
            logger.warning("%s\nLoading anyway (strict coverage disabled).", message)
    # 6d 不论成功还是抛异常都关掉两个线程池（wait=True：等正在跑的任务结束）
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if online_quant_streamer is not None:
            online_quant_streamer.shutdown()

    # Every rank logs its own line: the spread between ranks is itself one of
    # the things these numbers exist to explain.
    # 【译】每个 rank 都打印自己的一行：rank 之间的差异本身就是这些数字要解释的东西之一
    # ── 7 分阶段耗时日志 ── 例（docs/online_quantization_streaming_guide.md）：
    #   日志 read+queue 140.71s | drain 2.01s | quant drain 0.00s | staging flush 0.00s | threads 16
    #   → 时间几乎都花在主线程的读取 / 排队上（冷读盘），工作线程跟得上
    if online_quant_streamer is not None:
        logger.info(
            "[%s] weight load phases (including streaming online quant): "
            "read+queue %.2fs (%d tensors) | drain %.2fs | quant drain %.2fs | "
            "staging flush %.2fs | threads %d",
            rank_tag(),
            t_read,
            num_tensors,
            t_drain,
            t_quant_drain,
            t_flush,
            num_threads,
        )
    else:
        logger.info(
            "[%s] weight load phases: read+queue %.2fs (%d tensors) | "
            "drain %.2fs | staging flush %.2fs | threads %d",
            rank_tag(),
            t_read,
            num_tensors,
            t_drain,
            t_flush,
            num_threads,
        )

    _report_coverage(
        loaded_weights_record=loaded_weights_record,
        params_dict=params_dict,
        dropped_ckpt_keys=dropped_ckpt_keys,
        prefix=prefix,
        is_rank0=is_rank0,
    )

    # Avoid holding stale Parameter refs that prevent storage release.
    # 【译】不要一直持有过期的 Parameter 引用，否则它们的存储释放不了
    #   （后处理会用新 Parameter 替换在线量化前的旧参数，params_dict 还拿着旧的就会让源权重的显存一直活着）
    if online_quant_streamer is not None:
        online_quant_streamer.release_params_dict()
    del params_dict

    return loaded_weights_record


# 双向覆盖率告警（只告警，不中断加载）。调用方：load_weights_into_model 末尾
#   方向一：模型参数里有谁一个张量都没收到 → 还停在初始化值（RMSNorm 全 1、其余是 torch.empty 的随机内容）
#   方向二：checkpoint 里有谁被改名成了一个模型没有的参数名 → 真实数据被丢掉（WeightDispatcher._parameter 收集）
#   两者通常是同一类 bug：改名规则产生了一个模型里不存在的名字
def _report_coverage(
    *,
    loaded_weights_record: set[str],
    params_dict: dict,
    dropped_ckpt_keys: list[tuple[str, str]],
    prefix: str,
    is_rank0: Callable[[], bool],
) -> None:
    """Warn about parameters nothing wrote, and checkpoint tensors nothing took.

    Both directions point at the same bug class -- a name-rewrite rule that
    produces a name the model does not have -- and both are silent otherwise:
    the destination keeps its init value (all-ones for RMSNorm, whatever
    `torch.empty` left elsewhere) and the forward pass is quietly wrong.
    """
    # 【译】对没被任何东西写过的参数、以及没被任何参数接收的 checkpoint 张量发出警告。
    #   两个方向指向同一类 bug——改名规则产生了模型里没有的名字——而且不告警的话两者都是静默的：目标参数保持初始化值
    #   （RMSNorm 是全 1，其他地方是 `torch.empty` 留下的东西），forward 悄悄地算错
    # Verify every model parameter actually got loaded from the checkpoint.
    # Without this check, weights_mapping bugs (e.g. a substring rule
    # accidentally rewriting `attn_norm.weight` → `attn_model.norm.weight`)
    # silently leave the destination parameter at its init value (all-ones for
    # RMSNorm, all-zeros for newly-allocated buffers), corrupting forward
    # outputs in ways that are extremely hard to diagnose. WARN loudly here
    # so the failure surfaces at load time instead of at generation time.
    # 【译】核对每个模型参数确实从 checkpoint 加载过。没有这个检查，weights_mapping 的 bug（例如一条子串规则意外地把
    #   `attn_norm.weight` 改成 `attn_model.norm.weight`）会让目标参数悄悄停在初始化值（RMSNorm 是全 1，新分配的 buffer 是
    #   全 0），以极难诊断的方式破坏 forward 输出。这里大声告警，让失败在加载时而不是生成时暴露
    # 注："加载过"按 loaded_weights_record 判断，而分发器在提交写入任务时就记录了（不等写完），所以这里查的是"有没有被分发到"
    #   例：Qwen3-8B 的 model.layers.3.self_attn.qkv_proj.weight 收到 q/k/v 三次分发，记录一次
    loaded_param_names = {
        n.removeprefix(prefix) if prefix else n for n in loaded_weights_record
    }
    expected_param_names = set(params_dict.keys())
    unloaded = sorted(expected_param_names - loaded_param_names)
    # Filter known-OK skips: post-load-derived params (e.g. FusedMoE shuffle
    # output buffers, weight_scale params merged from multiple checkpoint scales).
    # Heuristic: anything ending in `_shuffled`, `_packed`, etc. Conservative
    # default = report everything else.
    # 【译】过滤掉已知没问题的跳过项：加载后才派生出来的参数（例如 FusedMoE shuffle 输出的 buffer、由多个 checkpoint scale
    #   合并出来的 weight_scale 参数）。启发式：以 `_shuffled`、`_packed` 等结尾的名字。保守的默认做法 = 其余全部报告
    suppressed_suffixes = ("_shuffled", "_packed", "_meta_for_quant")
    truly_unloaded = [
        n for n in unloaded if not any(n.endswith(s) for s in suppressed_suffixes)
    ]
    # Only report from rank 0 (other ranks have the same view).
    # 【译】只在 rank 0 报告（其他 rank 看到的一样）
    if truly_unloaded and is_rank0():
        sample = truly_unloaded[:20]
        logger.warning(
            "load_model: %d/%d model parameters were NOT loaded from "
            "checkpoint and remain at their init values. This is almost "
            "always a bug (typically a `weights_mapping` substring rule "
            "that accidentally renames a param to something the model "
            "doesn't have). Fix the mapping or the on-disk → param name "
            "translation. First %d unloaded names: %s",
            len(truly_unloaded),
            len(expected_param_names),
            len(sample),
            sample,
        )

    # Reverse direction: ckpt names that were silently dropped by
    # `get_parameter` AttributeError. These are the actionable bug class —
    # the mapping rewrote the ckpt name to something the model has no slot for,
    # so legitimate ckpt data was thrown away. Filter known-benign families
    # (output_scale, kv_scale, etc.) so the warning is signal, not noise.
    # 【译】反方向：被 `get_parameter` 的 AttributeError 悄悄丢掉的 checkpoint 名。这是能直接行动的那类 bug——映射把 checkpoint 名
    #   改成了模型里没有位置的名字，合法的 checkpoint 数据被扔掉了。过滤掉已知无害的几类（output_scale、kv_scale 等），
    #   让这条告警是信号而不是噪声
    if dropped_ckpt_keys:
        benign_substrings = (
            "output_scale",
            "kv_scale",
            "inv_freq",
        )
        actionable_drops = [
            (orig, mapped)
            for orig, mapped in dropped_ckpt_keys
            if not any(s in orig or s in mapped for s in benign_substrings)
        ]
        if actionable_drops and is_rank0():
            sample = actionable_drops[:20]
            logger.warning(
                "load_model: %d checkpoint tensors were silently dropped "
                "because the rewritten name has no matching model parameter. "
                "This is a `weights_mapping` / `WeightsMapper` bug — real "
                "ckpt data is being thrown away. Fix the rewrite rule. "
                "First %d (orig_ckpt_name → rewritten_name): %s",
                len(actionable_drops),
                len(sample),
                sample,
            )
