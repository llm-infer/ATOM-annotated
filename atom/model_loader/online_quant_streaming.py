# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Stream online quantization module by module during checkpoint loading.

Eligible parameters start on meta, stage arrivals on the host, then move to the
device, quantize, and release their source storage once complete.
"""

# 【译】在加载 checkpoint 的过程中，逐个模块地流式做在线量化。
#   符合条件的参数从 meta 设备起步，到达的数据先暂存在主机上，模块完整后搬到设备、量化，然后释放源存储
#
# 【导读】online_quant_streaming.py：loader.py 文件头【要点③】的实现（ATOM_ONLINE_QUANT_STREAMING=1 才启用，默认关）
#   前置：在线量化是什么见 arg_utils.py online_quant_config 上方【科普】。meta 设备：只记形状和 dtype、不分配存储的"假设备"——
#     在 meta 上建参数不占任何内存，之后再换上真正的存储（这里用 torch.utils.swap_tensors 原地换，Parameter 对象本身不变，
#     所以 params_dict、模块属性里拿着的还是同一个对象）
#   哪些模块参与：构造时 quant_spec.should_stream_online_quant 为真的 LinearBase / FusedMoE（开关打开、源格式能在线反量化、
#     这一层确实要被在线量化成另一种格式）——它们把 _stream_online_quant 设为 True，源权重建在 meta 上（linear.py、moe.py）
#   每卡显存的时间线对比：
#     非流式：[读盘 + 写入：源格式权重逐渐填满显存 …… 峰值 ≈ 全部源权重] → [后处理：逐模块量化，源权重被逐个替换、释放]
#     流式：  [模块 A 到齐 → H2D → 量化 → 释放 A 的源权重][模块 B 到齐 → …] …   峰值 ≈ 全部目标格式权重 + 在途模块
#     （DeepSeek-R1 专家转 MXFP4 的估算：每卡约 82 GB → 约 43 GB，见 loader.py 文件头【要点③】）
#   一个 Linear 模块的一生（Qwen3-8B、TP = 4 的某层 qkv_proj 在线量化成 FP8；默认主机暂存；形状见 loader.py 文件头【例】）：
#     构造：weight [1536, 4096] 建在 meta 上（不占内存）；_stream_expected_numel = 1536 × 4096 = 6,291,456
#     q_proj 到达（遍历线程）：run → _ensure_storage_or_defer：在 CPU 上实体化一块清零的 [1536, 4096] bf16 →
#       在 _CopyCounter 下调原来的 weight_loader，数到 copy_ 写了 1024 × 4096 = 4,194,304 个元素 → 累计 4,194,304
#     k_proj、v_proj 到达：各 256 × 4096 = 1,048,576 → 累计 6,291,456 = 期望值 → 认领（_stream_claimed = True）
#     _submit_finalize → 尾部线程：_settle_on_device（CPU → GPU）→ module.process_weights_after_loading()（量化成 FP8，
#       weight 换成一个新的 Parameter）→ params_dict 里那个旧的 bf16 Parameter 的存储换成空张量，显存释放
#   退回加载后处理（fallback，日志会数出来，这些模块不省显存）的情况：子模块要等父模块组合源权重
#     （get_streaming_deferred_modules，如 Kimi-K3）；融合专家这类绕过 run 的写入（没有可靠的覆盖率）；覆盖率统计失效；
#     到加载结束都没凑齐
#   线程：遍历 checkpoint 的线程（loading_core 的线程池；主机暂存时保留 ATOM_LOADER_NUM_THREADS 个，关掉主机暂存时只能 1 个）
#     + 尾部线程（ATOM_ONLINE_QUANT_STREAMING_THREADS，默认 4，各用自己的 CUDA stream；信号量把在途任务限制在 2 × 线程数）
#   使用者：loader.py 的 load_model（maybe_create、replay_stragglers_and_report、done_module_ids、candidates）；loading_core
#     （bind / release_params_dict、resolve_num_threads、setup_online_quant_pool、run、manages_param、drain、shutdown）；
#     weight_dispatch（经 loading_core 传入的 materialize_fused_param）。文档：docs/online_quantization_streaming_guide.md；
#     单元测试：tests/test_online_quant_streaming.py

import concurrent.futures
import logging
import threading

import torch
import torch.utils._python_dispatch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")

_HOST = torch.device("cpu")
# 主机暂存的目标设备


# 数元素：在 `with counter:` 期间，本线程每一次 aten.copy_ 写了多少个元素
#   TorchDispatchMode：进入后，当前线程上每个张量算子在分发到具体 kernel 之前都会先调 __torch_dispatch__，可以观察或替换它；
#   这里只观察 copy_、照常执行。为什么数 copy_ 而不看 checkpoint 张量的形状：weight_loader 会 narrow、切 shard、映射专家，
#   一个 checkpoint 张量实际写进参数多少个元素，只有看真正的 copy_ 才知道（docs/online_quantization_streaming_guide.md §5）
#   moe_arrival：MoE 快速路径用它带回"写了哪个 (专家, shard) 区域、多少元素"，代替计数
class _CopyCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Count elements written by ``aten.copy_`` in the current thread."""

    # 【译】统计当前线程里 ``aten.copy_`` 写入的元素数

    def __init__(self):
        super().__init__()
        self.copied_numel = 0
        self.moe_arrival = None

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # TorchDispatchMode keeps its compile-internal state in process-global
        # booleans. Concurrent counter scopes can restore those booleans out of
        # order and leave Dynamo believing that a mode is still active.
        # 【译】TorchDispatchMode 把与编译相关的内部状态放在进程全局的布尔变量里。并发的计数作用域可能以错乱的顺序恢复这些变量，
        #   让 Dynamo 以为某个 mode 仍然处于激活状态
        # （多个加载线程同时 `with counter:`，所以返回 True 让这个 mode 不参与那套全局记账）
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()
        return func(*args, **kwargs)


# 一次加载的流式状态（目标模型、草稿模型各建一个）。状态大多挂在各候选模块自己身上（_stream_* 属性），由模块自己的锁保护
class OnlineQuantStreamer:
    """Per-load state for streaming online quantization."""

    # 【译】流式在线量化的每次加载状态

    @classmethod
    def maybe_create(cls, model: nn.Module, load_dummy: str | None):
        """Return a streamer when enabled and applicable."""
        # 【译】开关打开且适用时返回一个流式量化器
        # 返回 None：没开 ATOM_ONLINE_QUANT_STREAMING、假权重、没有 _stream_online_quant 的模块（例如没给 --online_quant_config）
        # deferred_module_ids：模型声明的"要等父模块组合完再量化"的子模块（Kimi-K3 的 in_proj、f_a_proj），它们照样主机暂存，
        #   但到齐后不在这里量化，留给加载后处理按父先子后的顺序做（loader.py 第 5 步）
        if not envs.ATOM_ONLINE_QUANT_STREAMING or load_dummy:
            return None
        candidates = [
            (mod_name, m)
            for mod_name, m in model.named_modules()
            if getattr(m, "_stream_online_quant", False)
        ]
        if not candidates:
            return None
        deferred_module_ids = {
            id(child)
            for parent in model.modules()
            if hasattr(parent, "get_streaming_deferred_modules")
            for child in parent.get_streaming_deferred_modules()
        }
        return cls(candidates, deferred_module_ids)

    def __init__(
        self,
        candidates: list[tuple[str, nn.Module]],
        deferred_module_ids: set[int] | None = None,
    ):
        # 量化器自身的状态：candidates（全部候选模块）；param_to_module（参数 id → 模块，判断一次写入属于哪个流式模块）；
        #   done_module_ids（已认领、已提交量化的模块）；excessive_loads（认领之后才到、被丢弃的写入，加载完告警）
        self.candidates = candidates
        self.deferred_module_ids = deferred_module_ids or set()
        self.param_to_module: dict[int, nn.Module] = {}
        # Claimed modules no longer accept loader arrivals.
        # 【译】已认领的模块不再接受 loader 的到达
        self.done_module_ids: set[int] = set()
        self.excessive_loads: list[str] = []

        self._host_staging = envs.ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING
        # Module claims are decided under each module's lock.
        # 【译】模块是否认领在各模块自己的锁下决定（这把锁只保护 done_module_ids 集合）
        self._done_lock = threading.Lock()
        self._params_dict: dict | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: list = []
        # Bound the executor's otherwise-unbounded queue and its memory usage.
        # 【译】限制 executor 原本无上限的队列及其内存占用（每个排队的模块都拿着已经搬上 GPU 的源权重）
        self._slots: threading.Semaphore | None = None
        # Side streams avoid serializing workers on the default stream.
        # 【译】旁路 stream 避免各工作线程在默认 stream 上串行
        # threading.local()：每个线程看到自己的一份属性（这里是各尾部线程自己的 CUDA stream）
        self._worker_stream = threading.local()

        # 给每个候选模块挂上流式状态：
        for mod_name, m in candidates:
            # Coverage comes from copy counts or explicit MoE regions.
            # 【译】覆盖率来自拷贝计数，或显式的 MoE 区域
            m._stream_loaded_numel = 0
            # Protects storage materialization and coverage accounting.
            # 【译】保护存储的实体化和覆盖率的记账
            m._stream_lock = threading.Lock()
            m._stream_claimed = False
            m._stream_on_host = self._host_staging
            # Used only when loader calls must be replayed from meta.
            # 【译】只在必须从 meta 重放 loader 调用时使用（关掉主机暂存时，参数还在 meta 上，写入调用先记下来，到齐后再重放）
            m._stream_buffer_list = []
            # Published after materialization under `_stream_lock`.
            # 【译】实体化之后在 `_stream_lock` 下发布（已有真实存储的参数 id）
            m._stream_materialized_param_ids = set()
            # Names of source parameters released by `_finalize`.
            # 【译】`_finalize` 要释放的源参数的名字：(params_dict 里的全名, 模块上的属性名)
            m._stream_param_names = [
                (f"{mod_name}.{p_name}" if mod_name else p_name, p_name)
                for p_name, p in m.named_parameters(recurse=False)
                if p is not None
            ]
            # Per-parameter `(expert, shard)` coverage from ExpertStagingPool.
            # 【译】来自 ExpertStagingPool 协议的、按参数记录的 `(专家, shard)` 覆盖情况
            m._stream_moe_arrivals = {}
            # Mixed semantic and generic accounting cannot trigger early.
            # 【译】语义（MoE 区域）记账和通用（拷贝计数）记账混在一起时，不能提前触发认领
            #   （一个参数一旦退回通用路径，就把 _stream_tracking_invalid 置位，这个模块只能等加载后处理）
            # _stream_expected_numel：模块直属参数（recurse=False）的元素总数，累计写入达到它才算到齐
            m._stream_moe_declined_param_ids = set()
            m._stream_tracking_invalid = False
            m._stream_expected_numel = sum(
                p.numel() for _, p in m.named_parameters(recurse=False) if p is not None
            )
            for _, p in m.named_parameters(recurse=False):
                if p is not None:
                    self.param_to_module[id(p)] = m

    # ── loading-loop wiring ───────────────────────────────────────────────

    # 调用方：loading_core 把它作为 batching_excluded 传给分发器——流式参数不进 ExpertStagingPool
    def manages_param(self, param: nn.Parameter) -> bool:
        """Whether this parameter belongs to a streamed quantization module."""
        # 【译】这个参数是否属于一个流式量化模块
        return id(param) in self.param_to_module

    # 拿住 loading_core 的 params_dict，_finalize 要从里面找到旧参数对象并释放它的存储；加载结束时 release_params_dict 放手
    def bind_params_dict(self, params_dict: dict) -> None:
        self._params_dict = params_dict

    def release_params_dict(self) -> None:
        """Drop the params_dict ref so its Parameters can be collected."""
        # 【译】放掉对 params_dict 的引用，让其中的 Parameter 能被回收
        self._params_dict = None

    def resolve_num_threads(self, num_threads: int) -> int:
        """Use concurrent checkpoint loading only with host staging."""
        # 【译】只有在主机暂存时才并发加载 checkpoint
        #   关掉主机暂存时写入调用要在 meta 上缓冲、按顺序重放，为了顺序和状态安全只能单线程遍历
        if self._host_staging:
            return num_threads
        if num_threads > 1:
            tail_threads = envs.ATOM_ONLINE_QUANT_STREAMING_THREADS
            logger.info(
                "Online-quant streaming enabled: the checkpoint walk runs "
                "single-threaded; per-module quantization %s.",
                (
                    f"is offloaded to {tail_threads} worker thread(s)"
                    if tail_threads > 0
                    else "runs inline on the walking thread"
                ),
            )
        return 1

    def setup_online_quant_pool(self) -> None:
        """Create the module-finalization pool.

        Worker threads must set their CUDA device and use independent streams.
        """
        # 【译】创建模块收尾（量化）用的线程池。
        #   工作线程必须设置好自己的 CUDA 设备，并使用独立的 stream
        # initializer：每个工作线程启动时执行一次。新线程默认在 cuda:0 上，所以要 set_device 到本 rank 的卡
        num_workers = envs.ATOM_ONLINE_QUANT_STREAMING_THREADS
        if num_workers <= 0:
            return
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        worker_stream = self._worker_stream

        def _worker_init():
            if device is not None:
                torch.cuda.set_device(device)
                worker_stream.s = torch.cuda.Stream(device=device)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="atom-stream-quant",
            initializer=_worker_init,
        )
        self._slots = threading.Semaphore(2 * num_workers)

    def drain(self) -> None:
        """Wait for all finalizers and surface worker exceptions."""
        # 【译】等所有收尾任务完成，并把工作线程里的异常抛出来。调用方：loading_core（排空遍历线程之后）
        for future in concurrent.futures.as_completed(self._futures):
            future.result()
        self._futures.clear()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ── trigger ───────────────────────────────────────────────────────────

    # 【主干】每一次写入调用都经过这里（loading_core 的 _submit，可能在多个遍历线程里并发）：
    #   ① 不属于流式模块 → 照常 fn(*args)
    #   ② 不归本 rank 的专家（EP）→ 丢弃；模块已认领 → 丢弃并记进 excessive_loads（不能覆盖已量化的存储）
    #   ③ _run_counted：确保存储（或决定缓冲重放）→ MoE 快速路径记区域，或通用路径数 copy_ 的元素
    #   ④ 在模块锁下累计覆盖率，达到期望值就认领 → 不在延后名单里的立刻 _submit_finalize
    def run(self, fn, args) -> None:
        """Run one loader call and record its physical or logical coverage."""
        # 【译】执行一次 loader 调用，并记录它的物理覆盖（拷贝的元素数）或逻辑覆盖（MoE 区域）
        param = args[0] if args else None
        module = self.param_to_module.get(id(param)) if param is not None else None
        if module is None:
            fn(*args)
            return
        if self._is_nonlocal_expert_arrival(module, fn, args):
            return
        if module._stream_claimed:
            # Never overwrite quantized storage with a late source arrival.
            # 【译】绝不用迟到的源数据覆盖已经量化好的存储
            self.excessive_loads.append(
                getattr(module, "prefix", type(module).__name__)
            )
            return
        counter = self._run_counted(module, param, fn, args)
        with module._stream_lock:
            moe_arrival = counter.moe_arrival
            if moe_arrival is not None:
                param_id, region, covered_numel = moe_arrival
                if param_id in module._stream_moe_declined_param_ids:
                    # A concurrent decline invalidated semantic accounting.
                    # 【译】一次并发的 decline 让语义记账失效了
                    module._stream_tracking_invalid = True
                elif region is not None:
                    filled = module._stream_moe_arrivals.setdefault(param_id, set())
                    if region not in filled:
                        filled.add(region)
                        module._stream_loaded_numel += covered_numel
            else:
                # Repeated copies into one parameter must not claim it early.
                # 【译】对同一参数的重复拷贝不能让它被提前认领（单次计数最多算到参数的元素数）
                module._stream_loaded_numel += min(
                    counter.copied_numel,
                    param.numel(),
                )
            claimed = (
                not module._stream_tracking_invalid
                and module._stream_loaded_numel >= module._stream_expected_numel
                and not module._stream_claimed
            )
            if claimed:
                module._stream_claimed = True
        # Some child modules must keep their source weights until a parent
        # post-load hook has combined them. They still use host staging here,
        # but intentionally fall back to the ordered post-load pass for
        # finalization and quantization.
        # 【译】有些子模块必须保留源权重，直到父模块的加载后钩子把它们组合起来。它们在这里照样使用主机暂存，但刻意退回到
        #   有序的加载后处理去做收尾和量化
        if claimed and id(module) not in self.deferred_module_ids:
            self._submit_finalize(module)

    # 识别 FusedMoE.weight_loader(param, tensor, name, shard_id, expert_id) 这种调用：fn 是模块自己的绑定方法
    #   （fn.__self__ is module），第 5 个参数 args[4] 是全局专家号，映射到本 rank 为 -1 就不归本 rank
    @staticmethod
    def _is_nonlocal_expert_arrival(module, fn, args) -> bool:
        """Whether a claimed EP module can ignore this checkpoint arrival."""
        # 【译】一个 EP 模块能否忽略这次 checkpoint 到达
        return (
            len(args) >= 5
            and getattr(fn, "__self__", None) is module
            and hasattr(module, "_map_global_expert_id_to_local_expert_id")
            and module._map_global_expert_id_to_local_expert_id(args[4]) == -1
        )

    def _run_counted(self, module, param, fn, args) -> _CopyCounter:
        """Apply one loader call and return its logical or physical coverage."""
        # 【译】执行一次 loader 调用，返回它的逻辑覆盖或物理覆盖
        # deferred = True：参数还在 meta 上且关掉了主机暂存，这次调用要先在 meta 上"空跑"一遍（计数），再记进缓冲列表，
        #   等到齐时 _materialize_and_replay 在设备上重放
        deferred = self._ensure_storage_or_defer(module, param)
        # MoE fast path
        # 【译】MoE 快速路径（直接用 ExpertStagingPool 的协议把这份专家写进模块自己的主机存储，并按区域记账）
        if not deferred:
            try:
                moe_arrival = self._try_stage_expert_arrival(module, param, fn, args)
            except NotImplementedError:
                if not module._stream_on_host:
                    raise
                # Retry on the load device when the CPU lacks a kernel.
                # 【译】CPU 上缺 kernel 时（例如打包 dtype 的某些算子），改到加载设备上重试——先把整个模块搬上去
                self._settle_on_device(module)
                moe_arrival = self._try_stage_expert_arrival(module, param, fn, args)
            if moe_arrival is not None:
                counter = _CopyCounter()
                counter.moe_arrival = moe_arrival
                return counter

        # Generic path
        # 【译】通用路径：在 _CopyCounter 下调原 loader，数它写了多少元素
        counter = _CopyCounter()
        try:
            with counter:
                fn(*args)
        except NotImplementedError:
            # Preserve staged data and retry where the dtype has kernels.
            # 【译】保住已暂存的数据，换到这个 dtype 有 kernel 的设备上重试
            if not module._stream_on_host:
                raise
            self._settle_on_device(module)
            counter = _CopyCounter()
            with counter:
                fn(*args)
        if deferred:
            module._stream_buffer_list.append((fn, args))
        return counter

    def _try_stage_expert_arrival(self, module, param, fn, args):
        """Try the ExpertStagingPool semantic protocol on stream storage."""
        # 【译】在流式存储上尝试 ExpertStagingPool 的语义协议
        #   与 expert_staging.py 的 stage 相同的协议，只是暂存目标就是参数自己的（主机）存储 param.data；
        #   返回 (参数 id, (本地专家号, shard) 或 None, 覆盖的元素数)，不适用时返回 None 走通用路径
        if (
            len(args) < 5
            or getattr(fn, "__self__", None) is not module
            or not hasattr(module, "stage_expert_weight")
            or not hasattr(module, "expected_batched_arrivals")
            or not hasattr(module, "_map_global_expert_id_to_local_expert_id")
            or not hasattr(module, "is_batched_expert_slot")
            or not hasattr(module, "batched_expert_region_numel")
        ):
            return None

        param_id = id(param)
        if param_id in module._stream_moe_declined_param_ids:
            return None

        _, loaded_weight, weight_name, shard_id, global_expert_id = args[:5]
        expected = module.expected_batched_arrivals(param)
        if not expected:
            return None

        local_expert_id = module._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )
        if local_expert_id == -1:
            return (param_id, None, 0)
        if not module.is_batched_expert_slot(local_expert_id):
            return None

        staged = module.stage_expert_weight(
            param=param,
            staging=param.data,
            loaded_weight=loaded_weight,
            local_expert_id=local_expert_id,
            shard_id=shard_id,
            weight_name=weight_name,
        )
        if staged:
            covered_numel = module.batched_expert_region_numel(
                param,
                local_expert_id,
                shard_id,
            )
            return (
                param_id,
                (local_expert_id, shard_id),
                covered_numel,
            )

        # False means no bytes were written, so generic fallback is safe.
        # 【译】返回 False 表示没写任何字节，所以退回通用路径是安全的；但这个参数从此改用通用记账，若它之前已有区域记账，
        #   两种记账混在一起就不可靠了 → 标记失效
        with module._stream_lock:
            module._stream_moe_declined_param_ids.add(param_id)
            if module._stream_moe_arrivals.get(param_id):
                module._stream_tracking_invalid = True
        return None

    def materialize_fused_param(self, param: nn.Parameter) -> None:
        """Materialize a fused parameter whose loader bypasses ``run``."""
        # 【译】为一个 loader 绕过 ``run`` 的融合参数实体化存储
        # 调用方：weight_dispatch._dispatch_fused_expert（经 on_fused_param），在模型的 load_fused_expert_weights 直接写参数之前
        param_id = id(param)
        module = self.param_to_module.get(param_id)
        if module is None:
            return
        with module._stream_lock:
            # Fused writes have no semantic coverage, so force post-load fallback.
            # 【译】融合写入没有语义覆盖信息，所以强制退回加载后处理
            module._stream_moe_declined_param_ids.add(param_id)
            module._stream_tracking_invalid = True
            if param.data.is_meta:
                self._materialize_meta_to_host_or_device(param, module._load_device)
            elif param.data.device != module._load_device:
                self._swap_storage(param, param.data.to(module._load_device))
            module._stream_materialized_param_ids.add(param_id)

    # ── tail ──────────────────────────────────────────────────────────────

    @staticmethod
    def _swap_storage(param: nn.Parameter, buf: torch.Tensor) -> None:
        """Replace storage while preserving the Parameter object and attributes.

        Copy attributes before ``swap_tensors`` so concurrent dispatch never
        observes a missing ``weight_loader``.
        """
        # 【译】替换存储，同时保留 Parameter 对象本身及其属性。
        #   在 ``swap_tensors`` 之前先拷贝属性，这样并发的分发永远不会看到一个缺了 ``weight_loader`` 的参数
        # torch.utils.swap_tensors(a, b)：交换两个张量对象的全部内容（存储、元数据、__dict__），a 这个 Python 对象不变——
        #   所有持有 a 的地方（params_dict、模块属性）自动看到新存储
        replacement = nn.Parameter(buf, requires_grad=param.requires_grad)
        replacement.__dict__.update(param.__dict__)
        torch.utils.swap_tensors(param, replacement)

    @classmethod
    def _materialize_meta_to_host_or_device(cls, param: nn.Parameter, device) -> None:
        """Materialize zeroed storage, including unwritten padding bytes."""
        # 【译】实体化一块清零的存储，连没被写到的 padding 字节也清零（按 uint8 清，任何 dtype 都适用）
        buf = torch.empty(tuple(param.shape), dtype=param.dtype, device=device)
        buf.view(torch.uint8).zero_()
        cls._swap_storage(param, buf)

    def _ensure_storage_or_defer(self, module: nn.Module, param: nn.Parameter) -> bool:
        """Materialize host storage or defer the loader call for replay.

        Storage inspection and swapping stay under the module lock so racing
        arrivals cannot discard each other's writes.
        """
        # 【译】实体化主机存储，或者把这次 loader 调用延后、留待重放。
        #   存储的检查和替换都在模块锁下进行，这样竞争的到达不会丢掉彼此的写入
        # 返回 False = 已有真实存储，可以直接写；True = 仍在 meta 上（关掉了主机暂存），调用方要缓冲这次调用
        param_id = id(param)
        if param_id in module._stream_materialized_param_ids:
            return False
        with module._stream_lock:
            # Recheck after waiting before touching a possibly swapped TensorImpl.
            # 【译】等到锁之后再检查一次，再去碰可能已被替换过的 TensorImpl（双重检查）
            if param_id in module._stream_materialized_param_ids:
                return False
            if not param.data.is_meta:
                module._stream_materialized_param_ids.add(param_id)
                return False
            if not module._stream_on_host:
                return True
            self._materialize_meta_to_host_or_device(param, _HOST)
            module._stream_materialized_param_ids.add(param_id)
            return False

    def _settle_on_device(self, module: nn.Module) -> None:
        """Move staged parameters to the module's load device."""
        # 【译】把暂存的参数搬到模块的加载设备上（_load_device：构造时的默认设备，即本 rank 的 GPU）
        device = module._load_device
        with module._stream_lock:
            if not module._stream_on_host:
                return
            module._stream_on_host = False
            for _, p in module.named_parameters(recurse=False):
                if p is not None and not p.data.is_meta and p.data.device != device:
                    self._swap_storage(p, p.data.to(device))
                if p is not None and not p.data.is_meta:
                    module._stream_materialized_param_ids.add(id(p))

    def _materialize_and_replay(self, module: nn.Module) -> None:
        """Materialize remaining parameters and replay deferred loader calls."""
        # 【译】实体化剩下的参数，并重放被延后的 loader 调用（仍在 meta 上的参数直接在设备上建清零存储）
        for _, p in module.named_parameters(recurse=False):
            if p is not None and p.data.is_meta:
                self._materialize_meta_to_host_or_device(p, module._load_device)
            if p is not None:
                module._stream_materialized_param_ids.add(id(p))
        for fn, args in module._stream_buffer_list:
            fn(*args)
        module._stream_buffer_list.clear()

    def _settle_weights(self, module: nn.Module) -> None:
        """Give the module device storage holding everything that arrived."""
        # 【译】给模块一份设备上的存储，装着所有已到达的数据
        self._settle_on_device(module)
        self._materialize_and_replay(module)

    def _finalize(self, module: nn.Module) -> None:
        """Move, quantize, and release one completed module."""
        # 【译】搬运、量化并释放一个已完成的模块
        # 调用方：_submit_finalize（尾部线程，或没有尾部线程时在遍历线程里同步执行）
        self._settle_weights(module)
        module.process_weights_after_loading()
        # Keep stale objects alive for id-based lookup, but release their storage.
        # 【译】保留过期的对象（按 id 查找还要用到它们），但释放它们的存储
        #   量化后 module.weight 是一个新 Parameter；params_dict 里那个旧对象若不释放，它的 bf16 显存就一直被占着
        params_dict = self._params_dict
        if params_dict is None:
            return
        for full_name, p_name in module._stream_param_names:
            stale = params_dict.get(full_name)
            if stale is not None and stale is not getattr(module, p_name, None):
                stale.data = torch.empty(0, dtype=stale.dtype, device=stale.device)

    def _submit_finalize(self, module: nn.Module) -> None:
        """Hand a completed module to the tail workers and return immediately."""
        # 【译】把一个已完成的模块交给尾部线程，然后立即返回
        # Mark claimed before the worker starts to prevent double post-processing.
        # 【译】在工作线程开始之前就标记为已认领，防止重复做后处理（loader.py 的后处理按 done_module_ids 跳过它们）
        with self._done_lock:
            self.done_module_ids.add(id(module))
        if self._executor is None:
            self._finalize(module)
            return

        def _task():
            try:
                # 在本尾部线程自己的 stream 上做 H2D + 量化，最后 synchronize：任务结束时结果已落在显存里，
                #   主线程不必知道这些 stream
                s = getattr(self._worker_stream, "s", None)
                if s is None:
                    self._finalize(module)
                else:
                    with torch.cuda.stream(s):
                        self._finalize(module)
                    s.synchronize()
            finally:
                # Always release the slot so worker failures cannot deadlock loading.
                # 【译】总是释放槽位，这样工作线程失败也不会让加载死锁
                self._slots.release()

        # 信号量：在途（已提交未完成）的收尾任务满 2 × 线程数时，提交方（遍历线程）在这里阻塞，限制占着显存的源权重数量
        self._slots.acquire()
        self._futures.append(self._executor.submit(_task))

    # ── post-load report ──────────────────────────────────────────────────

    # 收尾（调用方：loader.py load_model 第 4 步）：没被认领的候选模块统一实体化 / 重放，留给加载后处理量化；再报告
    #   从没加载过的参数（只能清零，模型几乎肯定是错的）、认领后才到被丢弃的写入、退回后处理的模块数
    def replay_stragglers_and_report(self, is_rank0: bool) -> None:
        """Settle unclaimed modules and report coverage fallbacks."""
        # 【译】安顿未被认领的模块，并报告覆盖率不足而退回的情况
        stranded = []
        for mod_name, m in self.candidates:
            # Detect missing params before `_settle_weights` zero-fills them.
            # 【译】在 `_settle_weights` 把缺失的参数清零之前把它们找出来（仍在 meta 上、又没有缓冲调用指向它的参数）
            targeted = {id(a[0]) for _, a in m._stream_buffer_list}
            for p_name, p in m.named_parameters(recurse=False):
                if p.data.is_meta and id(p) not in targeted:
                    stranded.append(f"{mod_name}.{p_name}")
            self._settle_weights(m)

        fell_back = [n for n, m in self.candidates if id(m) not in self.done_module_ids]
        if not is_rank0:
            return
        if stranded:
            logger.warning(
                "Online-quant streaming: %d parameter(s) were never loaded "
                "and stayed on the meta device; they have been zero-filled "
                "so post-processing can run, but the model is almost "
                "certainly wrong. First %d: %s",
                len(stranded),
                min(len(stranded), 20),
                stranded[:20],
            )
        if self.excessive_loads:
            unique = sorted(set(self.excessive_loads))
            logger.warning(
                "Online-quant streaming: dropped %d load(s) that arrived "
                "after their module was already quantized, across %d "
                "module(s). The checkpoint supplies more elements for these "
                "than _stream_expected_numel accounts for, so the surplus cannot "
                "be stored; verify the expected-size computation. First "
                "%d: %s",
                len(self.excessive_loads),
                len(unique),
                min(len(unique), 20),
                unique[:20],
            )
        log = logger.warning if fell_back else logger.info
        log(
            "Online-quant streaming: %d/%d eligible modules quantized during "
            "load, %d fell back to the post-load pass (no memory saving for "
            "those). First %d fallbacks: %s",
            len(self.candidates) - len(fell_back),
            len(self.candidates),
            len(fell_back),
            min(len(fell_back), 20),
            fell_back[:20],
        )
