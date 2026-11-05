# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【导读】cuda_graph.py：PIECEWISE 类 CUDA Graph 的录制 / 回放器（第 5 批 5e，B 级）
#   改编自 vLLM 的 vllm/compilation/cuda_graph.py（见上面的版权头和下面被注释掉的 vLLM import），
#   ATOM 另加了：按 num_tokens 分桶的内存池、StableOutputs、给 attention 核心单独录图的 CudagraphCaptureRunner
#
# 前置（只列本文件要用的三条要点。概念见 config.py 中 CUDAGraphMode 上方的【科普】②④；录哪些尺寸、录制顺序
#   （从大到小录、录完再排回升序）、FULL 整图共用的内存池见 model_runner.py capture_cudagraph 上方的说明）：
#   ① 地址固定：回放读写的是录制时那几块显存，新数据必须拷进同一块 buffer
#   ② 形状固定：每种 num_tokens 单独录一张，运行时补齐到能装下它的最小已录尺寸（ForwardMode.decide）
#   ③ 录制期间新分配的中间张量来自图的私有内存池（graph pool），回放时复用同样的位置；不会同时回放的图可以共用一个池
#
# ATOM 的两条录制路径（由 ModelRunner._piecewise_cg_active() 决定，读的是 --cudagraph-mode）
#   手动整图（FULL，默认；原生下 --cudagraph-mode NONE 也走这条）：ModelRunner.capture_cudagraph 自己用 torch.cuda.graph 录整个 forward，
#     存进 runner.graphs[(bs, max_q_len)]，所有图共用 runner.graph_pool——和 nano-vllm 一样，不经过本文件
#   逐段（PIECEWISE / FULL_AND_PIECEWISE / AF_PIECEWISE，需要 --level 3）：编译后端在切分算子（通常是 attention；
#     DeepSeek-V4 是压缩器 v4_attn_compress）处把模型切成若干段，backends.py 的 PiecewiseCompileInterpreter 把每个编译好的
#     稠密段包进一个 CUDAGraphWrapper（本文件）。ModelRunner.capture_cudagraph 对每个尺寸先预热一次，再以 PIECEWISE
#     模式跑一次 forward，各段在这一次里各自录下 num_tokens 对应的图；之后真实 decode 时同一 num_tokens 就回放
#
#   一次 PIECEWISE decode forward（num_tokens = 64；切分算子把模型切成若干稠密段）：
#
#     input_ids / positions [64]（runner 的固定 buffer，_padded_decode_inputs）
#       │
#       ▼
#     [段 0：embedding + 第 0 层切分算子之前的部分]   CUDAGraphWrapper#0 → entries[BatchDescriptor(64)].cudagraph.replay()
#       │ 输出在 num_tokens=64 这个桶的内存池里，地址录制时已定
#       ▼
#     第 0 层切分算子（eager 运行，每步都由 CPU 发射；AF_PIECEWISE 下可以有自己的图，见 attn_ffn_piecewise.py）
#       │ 它的输出就是下一段的输入。下一段录制时记住了这个地址，回放时它必须还在原地：
#       │ __call__ 每次回放前逐个比对张量参数的地址（is_debugging_mode 恒为 True），对不上就报错；
#       │ 设计上由 piecewise_core 把切分算子的输出拷进 StableOutputs 的固定 slot 来满足这一点，但 v0.1.8 没有构造它的地方
#       │ （唯一的使用者 V4 的切分算子不返回值，见下面类表 StableOutputs 一行）
#       ▼
#     [段 1：第 0 层余下部分（o_proj、MLP…）+ 第 1 层切分算子之前的部分]   CUDAGraphWrapper#1 → replay
#       ⋮
#     [最后一段：最后一层余下部分 + final norm]   CUDAGraphWrapper（is_last_graph：输出只留弱引用）
#       ▼
#     hidden_states [64, H] → compute_logits（图外）
#   切分点在哪里定、为什么切分算子留在图外：见 utils/backends.py 与 config.py【科普】④
#
# 本文件的类与使用方（grep 核实）
#   BatchDescriptor         图的键（num_tokens）。ModelRunner 的 run_model / capture_cudagraph / _capture_attn_ffn_graphs
#                           写进 forward_context.batch_descriptor，CUDAGraphWrapper.__call__ 读
#   CUDAGraphEntry          一个键的一条记录：graph、输出（弱引用）、录制时的输入地址
#   CUDAGraphOptions        wrapper 的构造选项（backends.py 按段在第几个设置）
#   CUDAGraphWrapper        每个编译好的稠密段一个，由 backends.py 构造；构造时在 graph_holders.py 登记，供 RL 休眠时释放
#   StableOutputs           切分算子输出的固定落点，attn_ffn_piecewise.piecewise_core 的 outputs 参数
#                           （v0.1.8 里 atom/ 没有地方构造它：唯一的 piecewise_core 使用者 V4 传 outputs=None，它的核心不返回值）
#   CudagraphCaptureRunner  AF_PIECEWISE 下给"attention 核心"单独录图；唯一实例是 deepseek_v4.py 的 v4_attn_runner
#   _graph_pools / _shared_graph_pool  PIECEWISE 稠密段的内存池（默认每个 num_tokens 桶一个，下面有实测数据）
#
# 对比nano-vllm：nano-vllm 只有"decode 整图 + 共用一个池"这一种（对应 ATOM 的手动 FULL 路径），没有编译切图，
#   也就没有逐段 wrapper、按桶分池和 attention 核心单独录图

import dataclasses
import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
from aiter import logger

from atom.config import Config, CUDAGraphMode
from atom.utils import compilation_counter, weak_ref_tensors
from atom.utils.forward_context import get_forward_context
from atom.utils.graph_holders import register_graph_holder

# from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
# from vllm.config import CUDAGraphMode, VllmConfig
# from vllm.forward_context import BatchDescriptor, get_forward_context
# from vllm.logger import init_logger
# from vllm.platforms import current_platform

# logger = init_logger(__name__)


# 图的键：只含补齐后的 token 数（NamedTuple：不可变、可哈希，能当 dict 的键）
#   例：BatchDescriptor(num_tokens=64)；decode bs=21 补到 32 个 token 时键是 BatchDescriptor(32)
class BatchDescriptor(NamedTuple):
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    # 【译】用于 cudagraph 分发的 batch 描述符。字段要尽量少，只要能正确、唯一地描述补齐后的 batch 即可。

    num_tokens: int
    # vLLM 遗留：ATOM 构造 BatchDescriptor 时只传 num_tokens，这个字段恒为 False（grep 核实，下面的 non_uniform 也没有调用方）
    uniform_decode: bool = False
    """
    False can also be used for an uniform decode batch to dispatch to the 
    cudagraph supporting non-uniform batches.
    """
    # 【译】uniform（每条序列 query 长度相同）的 decode batch 也可以用 False，以分发到支持非 uniform batch 的 cudagraph。

    @property
    def non_uniform(self) -> "BatchDescriptor":
        """
        Return a non-uniform version of current batch descriptor.
        """
        # 【译】返回当前 batch 描述符的非 uniform 版本。
        return BatchDescriptor(self.num_tokens, uniform_decode=False)


# CUDAGraphWrapper 里一个键（num_tokens）的记录
@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    # 录好的图；None = 还没录（release_graphs 后也会整条删掉）
    cudagraph: torch.cuda.CUDAGraph | None = None
    # 录制时的输出，存成弱引用：显存归图的内存池所有，回放时同一位置被重新写入
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    # 【译】用于 cudagraph 调试：录制时记下输入地址，回放时检查是否相同
    input_addresses: list[int] | None = None


# CUDAGraphWrapper 的构造选项；backends.py 按段的位置设置：
#   debug_log_enable = is_first_graph（只有第一段打"Capturing a cudagraph"日志）
#   gc_disable       = not is_first_graph（除第一段外，录制时把 gc.collect / empty_cache 换成空函数）
#   weak_ref_output  = is_last_graph（只有最后一段在录制时就把输出换成弱引用）
@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


# One cudagraph pool per num_tokens bucket (default). A bucket's graphs only
# ever reuse memory among themselves, so nothing one bucket's graph wrote can be
# handed to another bucket's.
#
# Sharing one pool across buckets corrupts DeepSeek-V4 decode. Measured on
# V4-Pro-DSpark tp8, GSM8K 3-shot 250q, --cudagraph-mode PIECEWISE:
#
#   pool shared, 11 buckets   0.664      pool shared, 2 buckets [32,64]  0.956
#   pool shared, 1 bucket     0.976      per-bucket, 11 buckets          0.960
#
# Generations stay coherent for a few decode tokens and then collapse, prefill
# is unaffected, and it is intermittent. Removing cross-bucket sharing by either
# route (one bucket, or one pool each) fixes it; the corrupted memory is inside
# the compiled pieces, not on the graph->eager boundary — pinning every boundary
# tensor of the one V4 split op changes nothing (0.636).
# 【译】默认每个 num_tokens 桶一个 cudagraph 内存池。一个桶的图只在彼此之间复用显存，
#   所以一个桶的图写过的东西，不可能被交给另一个桶的图。
#
#   让所有桶共用一个池会把 DeepSeek-V4 的 decode 弄坏。实测：V4-Pro-DSpark tp8，GSM8K 3-shot 250 题，
#   --cudagraph-mode PIECEWISE：
#
#     共用池，11 个桶   0.664      共用池，2 个桶 [32,64]   0.956
#     共用池，1 个桶    0.976      每桶一个池，11 个桶       0.960
#
#   生成的内容在头几个 decode token 还是通顺的，然后崩坏；prefill 不受影响，而且是间歇性的。
#   去掉跨桶共享——无论哪种办法（只留一个桶，或每桶一个池）——都能修好；被破坏的显存在编译出来的段内部，
#   不在 graph→eager 的边界上：把 V4 那个切分算子的每个边界张量都钉住，结果没有变化（0.636）。
# 解释：键是 BatchDescriptor.num_tokens；同一个桶里的各段（同一次 forward 里依次回放）共用一个池，
#   不同桶各用各的。由 __call__ 在每个桶第一张图录完后用 cudagraph.pool() 填入；release_graphs 时删除
_graph_pools: dict = {}

# ATOM_PER_BUCKET_POOL=0 restores the single shared pool. It buys little: on the
# config above the two modes differ by 1.11GB reserved per rank (202.83 ->
# 203.94GB) and the capture-time allocated delta is identical at 14.71GB, so the
# overlay was not reducing live footprint at all. The wider claim it was added
# for (~10GB vs ~35GB unshared on DSV4 TP8) does not reproduce there; a config
# with far more or larger buckets, or DP, may still see a real gap.
# 【译】ATOM_PER_BUCKET_POOL=0 恢复成单个共享池。它换来的很少：在上面的配置下，两种模式每个 rank 的保留显存只差 1.11GB
#   （202.83 → 203.94GB），录制期间的已分配增量完全相同（14.71GB），所以叠放复用根本没有减少实际占用。
#   当初引入它所声称的更大收益（DSV4 TP8 上约 10GB 对比不共享的约 35GB）在那里复现不出来；
#   桶多得多或大得多的配置、或者 DP，可能仍会看到真实的差距。
# 解释：ATOM_PER_BUCKET_POOL 由 __call__ 直接读 os.environ（不在 envs.py 里），默认 "1"
_shared_graph_pool: Any | None = None


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic.
    """

    # 【译】包装一个可调用对象（runnable），给它加上 CUDA graph 录制与回放的能力，并通过 `__getattr__` 提供对底层
    #   `runnable` 属性的访问。
    #
    #   这个 wrapper 在 cudagraph 分发中的工作流程：
    #   1. 初始化时给 wrapper 指定一个运行时模式（FULL 或 PIECEWISE）。
    #   2. 运行时，wrapper 从 forward context 拿到 runtime_mode 和 batch_descriptor（键），并无条件信任它们来做分发。
    #   3. runtime_mode 是 NONE，或与 wrapper 的模式不一致，就直接调用 runnable。
    #   4. 否则（runtime_mode 与 wrapper 的模式一致），wrapper 做 cudagraph 录制（键不存在时新建一条记录并缓存）
    #      或回放（键已存在）。
    #
    #   注意：CUDAGraphWrapper 不保存持久 buffer，也不把运行时输入拷进这些 buffer 以供回放。我们假设这些在 wrapper
    #   外部完成。这是因为我们对运行时输入的动态形状（batch size）不做任何假设，以此换取与编译逻辑保持正交。
    # 解释：ATOM 只以 PIECEWISE 模式构造它（backends.py），FULL 走 ModelRunner 的手动路径；
    #   "外部"在 ATOM 里就是 ModelRunner：input_ids / positions 用 runner 的固定 buffer 补齐到桶大小
    #   （_padded_decode_inputs），runtime_mode 和 batch_descriptor 也由 runner 在调用 self.model 前后写入 / 清空
    # runnable：backends.py 的 PiecewiseBackend（一个编译好的稠密段）；vllm_config：ATOM 的 Config（沿用 vLLM 的参数名）
    def __init__(
        self,
        runnable: Callable,
        vllm_config: Config,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = True

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        # 【译】断言 runtime_mode 不是 NONE（不录 cudagraph），否则根本不需要构造 CUDAGraphWrapper。
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        # 【译】TODO：将来如果要用多个 stream，共享一个全局池可能不安全。等用到多 stream 时再研究。

        # self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        # 【译】需要为之录制 cudagraph 的各个 batch 描述符对应的记录。
        # 例：capture_sizes 去重后的 num_tokens 有 15 个时，每个 wrapper 最终有 15 条记录
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        # Nothing else can find these graphs: the backend installs this wrapper
        # in a graph module's `__dict__`, off both `_modules` and any path from
        # the runner. See `atom/utils/graph_holders.py`.
        # 【译】除此之外没人能找到这些图：后端把这个 wrapper 装在一个 graph module 的 `__dict__` 里，
        #   既不在 `_modules` 中，也不在从 runner 出发的任何路径上。见 `atom/utils/graph_holders.py`。
        register_graph_holder(self)

    def release_graphs(self) -> int:
        """Drop every graph captured here, so what it captured can be freed.

        The entry, not just the graph: `output` is a live reference into the
        graph's private pool, which is the memory a release is trying to
        reclaim. Recapture needs no arranging -- `__call__` records on the first
        step whose entry has no graph -- but it must not happen mid-serve, so
        whoever calls this has to stop PIECEWISE being dispatched first (for the
        runner that is `_piecewise_captured_tokens`).
        """
        # 【译】丢掉这里录过的每一张图，让它们录下的东西可以被释放。
        #
        #   丢的是整条记录，而不只是图：`output` 是指向图私有内存池的活引用，而那正是释放想要回收的显存。
        #   重新录制不需要安排——`__call__` 在第一个"记录里没有图"的 step 上就会录——但它绝不能发生在服务中途，
        #   所以调用者必须先阻止 PIECEWISE 被分发（对 runner 来说就是 `_piecewise_captured_tokens`）。
        # 调用方：graph_holders.release_registered_graphs（← rollout/memory_manager.py 的 _release_piecewise_graphs）
        #   返回丢掉的图数
        global _shared_graph_pool

        dropped = 0
        for descriptor, entry in self.concrete_cudagraph_entries.items():
            if entry.cudagraph is not None:
                dropped += 1
            # Forget the pool this bucket recorded into. A pool whose last graph
            # has just gone away stays in the allocator's table with a use count
            # of zero until an `empty_cache()` erases it, and starting a capture
            # on an id in that state trips an internal assertion instead of
            # making a new pool. `runner.graph_pool = None` has always done this
            # for the manual store; these two globals are the piecewise
            # equivalents. Same expression `__call__` keys them by.
            # 【译】忘掉这个桶录制时用的池。最后一张图刚被删掉的池，会以使用计数 0 的状态留在分配器的表里，
            #   直到某次 `empty_cache()` 把它抹掉；在这种状态的 id 上开始录制会触发一个内部断言，而不是新建一个池。
            #   手动整图那边一直用 `runner.graph_pool = None` 做这件事；这两个全局变量是 PIECEWISE 的对应物。
            #   键的算法与 `__call__` 相同。
            # 各段共用同一组桶键，第一个 wrapper 释放时就把池删掉了，后面的 pop 得到 None，无害
            _graph_pools.pop(descriptor.num_tokens if descriptor else 0, None)
        self.concrete_cudagraph_entries.clear()
        _shared_graph_pool = None
        return dropped

    # 只在正常属性查找失败时才调用：把属性访问转给被包装的段
    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        # 【译】允许访问 runnable 的属性。
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        # 【译】以备需要访问原始 runnable。（atom/ 里没有调用方）
        return self.runnable

    # 每次这个稠密段被调用都进这里，三种结果：
    #   runtime_mode 不是 PIECEWISE（预热 forward、prefill、没录过的尺寸、FULL 路径…）→ 直接跑编译好的段
    #   PIECEWISE 且这个 num_tokens 还没有图（只应发生在 capture_cudagraph 的录制轮）→ 录制，并返回录制时的输出
    #   PIECEWISE 且已有图 → 检查输入地址 → replay → 返回录制时那块输出（弱引用）
    # 调用链：ModelRunner.run_model 设置 forward_context.cudagraph_runtime_mode / batch_descriptor → self.model(...)
    #   → 编译后的 split graph 依次调用各段 → 本方法
    #   （runner 只在 ForwardMode.decide 判定 piecewise_captured——补齐后的 token 数在 _piecewise_captured_tokens 里——
    #    时才设 PIECEWISE，否则设 NONE，避免服务中途在某个 rank 上临时录制、卡死在集合通信上）
    def __call__(self, *args, **kwargs):
        # Rebound on the first capture when the shared pool is opted into;
        # `_graph_pools` is only mutated in place, so it needs no declaration.
        # 【译】选用共享池时，它在第一次录制时被重新绑定（所以要 global 声明）；
        #   `_graph_pools` 只被原地修改，不需要声明。
        global _shared_graph_pool

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            # 【译】CUDAGraphMode.NONE 可能意味着 profile 运行、预热运行，或者不用 cudagraph 运行。
            #   runtime mode 不匹配时我们不触发录制 / 回放。这样在嵌套多个不同 runtime mode 的实例时，
            #   能正确分发到对应的那个 CUDAGraphWrapper。
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            # 【译】为这个 batch 描述符新建一条记录
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                # 【译】我们要为很多不同形状录 cudagraph，而录制很快，没必要每个形状都打日志。
                #   例如 piecewise 模式下只为第一个子图打日志。
                logger.info(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            # 【译】检查此刻录制 cudagraph 是否合法。（下面的检查在 ATOM 里被注释掉了，留作 TODO）
            # ================= TODO lirong ========================
            # 【译】待办（lirong）
            # validate_cudagraph_capturing_enabled()

            # 记下每个张量参数的地址，回放时逐个比对（前置①：图只认地址）
            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            # 一张空图，下面 with 块里发出的 GPU 操作都被录进去
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    # 【译】piecewise cudagraph 模式下，每次模型 forward 都要录很多段 cudagraph（大约每层一段）。
                    #   在层与层之间反复跑 gc 会让录制非常慢。所以只在第一张图时跑 gc，其余的图都禁用 gc。
                    # 解释：unittest.mock.patch 在 with 范围内把 gc.collect / torch.cuda.empty_cache 临时换成空函数
                    #   （PyTorch 的 graph 录制入口会调用它们），ExitStack 负责退出时还原
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                # Default: one pool per num_tokens bucket, so no bucket's graph
                # can be handed memory another bucket's graph wrote. See the
                # module header for the accuracy data behind this default and
                # for what ATOM_PER_BUCKET_POOL=0 costs.
                # 【译】默认每个 num_tokens 桶一个池，这样任何桶的图都不会拿到另一个桶的图写过的显存。
                #   这个默认值背后的精度数据，以及 ATOM_PER_BUCKET_POOL=0 的代价，见模块开头。
                # 每桶池：这个桶第一张图（第一段）录制时 _pool 为 None → PyTorch 新建一个池，录完再存进 _graph_pools；
                #   同一桶后面的段拿到这个池，在里面接着分配
                _per_bucket = os.environ.get("ATOM_PER_BUCKET_POOL", "1") == "1"
                _bkey = batch_descriptor.num_tokens if batch_descriptor else 0
                if _per_bucket:
                    _pool = _graph_pools.get(_bkey)
                else:
                    # Match vLLM (platforms/interface.py:get_global_graph_pool):
                    # use a DEDICATED shareable pool handle created ONCE up front
                    # for EVERY graph.
                    # 【译】与 vLLM 一致（platforms/interface.py 的 get_global_graph_pool）：用一个专门的、可共享的池句柄，
                    #   一开始就创建一次，给所有图用。
                    if _shared_graph_pool is None:
                        _shared_graph_pool = torch.cuda.graph_pool_handle()
                    _pool = _shared_graph_pool
                # thread_local, not the "global" default: global mode invalidates
                # the capture when ANY thread makes an unsafe HIP call, and under
                # DP attention the NCCL watchdog thread polls hipEventQuery on
                # outstanding works every ~100ms. Piecewise captures one small
                # graph per layer with eager (collective-issuing) sections in
                # between, so that poll reliably lands inside a capture and kills
                # the process with hipErrorStreamCaptureUnsupported. thread_local
                # restricts the check to the capturing thread, which is the only
                # one touching this stream.
                # Capture on graph_capture()'s stream (vLLM parity), not torch's
                # own side stream, so pieces + collectives stay coherent.
                # 【译】用 thread_local，而不是默认的 "global"：global 模式下只要*任何*线程做了一次不安全的 HIP 调用，
                #   录制就会失效；而 DP attention 下 NCCL 的 watchdog 线程大约每 100ms 会对未完成的工作调用一次
                #   hipEventQuery。piecewise 每层录一张小图，中间夹着 eager（会发起集合通信）的部分，
                #   所以那次轮询必然会落进某次录制里，以 hipErrorStreamCaptureUnsupported 杀掉进程。
                #   thread_local 把检查限制在正在录制的线程，而它是唯一碰这个 stream 的线程。
                #   在 graph_capture() 的 stream 上录制（与 vLLM 一致），而不是 torch 自己的副 stream，
                #   这样各段和集合通信保持一致。
                # 解释：graph_capture() 是 ModelRunner.capture_cudagraph 外层的上下文（它也为录制准备好 custom all-reduce），
                #   current_stream() 此时就是它的 stream
                with torch.cuda.graph(
                    cudagraph,
                    pool=_pool,
                    stream=torch.cuda.current_stream(),
                    capture_error_mode="thread_local",
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    # 【译】`output` 由 pytorch 的 cudagraph 池管理
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        # 【译】转成弱引用后，原来的 `output` 会立即被释放以节省显存。只有 piecewise cudagraph 模式下的
                        #   最后一张图才能这样做，因为最后一张图的输出不会被其他任何 cuda graph 使用。
                        # 解释：弱引用 = 与原张量共享显存、但不让它的 storage 保持存活（utils/__init__.py 的 weak_ref_tensor）；
                        #   中间段的输出还要被后面的段在录制时读，录制期间必须是强引用，否则那块显存会被池提前复用
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            # first graph of a per-bucket pool -> remember its pool. (shared pool
            # is created up front via graph_pool_handle() above, nothing to do.)
            # 【译】这里输出一律用弱引用，以节省显存。每桶池的第一张图 → 记住它的池。
            #   （共享池在上面已用 graph_pool_handle() 预先建好，这里什么都不用做。）
            if _per_bucket and _bkey not in _graph_pools:
                _graph_pools[_bkey] = cudagraph.pool()

            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            # 【译】重要：这里要返回输出本身，而不是它的弱引用，这样 pytorch 在 cuda graph 录制期间才能正确管理显存。
            return output

        # 回放分支。is_debugging_mode 在 __init__ 里写死为 True（vLLM 只在 DEBUG 日志级别下检查），所以每次回放都比对
        if self.is_debugging_mode:
            # check if the input addresses are the same
            # 【译】检查输入地址是否相同
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # 一条命令重放这一段的全部 kernel；结果写在录制时那块输出显存里，直接返回它（弱引用）
        entry.cudagraph.replay()
        return entry.output


class StableOutputs:
    """Persistent output slots for a split op running inside PIECEWISE.

    The dense graph piece downstream of the op was captured reading ONE fixed
    address per key. Whatever produces the op's output -- a replayed cudagraph
    or a plain eager call -- has to deliver it there, or the piece reads a
    tensor the allocator has since recycled.

    Deliberately not part of the capture runner: a slot is needed whenever the
    op sits inside PIECEWISE, whether or not anything was ever captured.
    """

    # 【译】PIECEWISE 中某个切分算子的持久输出 slot。
    #
    #   这个算子下游的稠密段录制时，每个键读的是*一个*固定地址。不管算子的输出是谁产生的——回放的 cudagraph 还是
    #   普通的 eager 调用——都必须送到那里，否则那一段读到的是分配器早已回收再用的张量。
    #
    #   刻意不放进 capture runner：只要算子位于 PIECEWISE 中就需要 slot，不管有没有录过任何东西。
    # 键由 piecewise_core 给出：(layer_name, num_tokens)；例：("model.layers.3.attn", 64) → 一块 [64, …] 的 buffer

    def __init__(self):
        self._slots: dict = {}

    def slot(self, key, sample: torch.Tensor) -> torch.Tensor:
        """This key's persistent buffer, shaped from `sample` on first use."""
        # 【译】这个键的持久 buffer；第一次使用时按 `sample` 的形状创建。
        buf = self._slots.get(key)
        if buf is None:
            buf = torch.empty_like(sample)
            self._slots[key] = buf
        return buf

    def deliver(self, key, out: torch.Tensor) -> torch.Tensor:
        """Copy an eagerly computed `out` into its slot and return the slot."""
        # 【译】把 eager 算出的 `out` 拷进它的 slot，并返回这个 slot。
        # 每步多一次拷贝，换来下游段永远在同一地址读到结果
        buf = self.slot(key, out)
        buf.copy_(out)
        return buf


class CudagraphCaptureRunner:
    """A cache of cudagraphs, keyed by whatever the caller decides identifies one.

    Give it a key, the inputs a graph should read, and the compute: it records
    once and replays after that. It does not know who is calling or why --
    deciding WHETHER a step should be captured, and where the output has to
    land, belongs to the caller (see `piecewise_core` in
    atom/utils/attn_ffn_piecewise.py, the attention-core user of this).

    Holds its own graph pool, isolated from the dense-piece pool.
    """

    # 【译】cudagraph 的缓存，键由调用方决定用什么来标识一张图。
    #
    #   给它一个键、图要读的输入、以及计算：它录一次，之后就回放。它不知道是谁在调用、为什么调用——
    #   决定某一步*是否*该录制、输出要落在哪里，归调用方负责（见 atom/utils/attn_ffn_piecewise.py 的 `piecewise_core`，
    #   它就是这个类在 attention 核心上的使用者）。
    #
    #   持有自己的图内存池，与稠密段的池隔离。
    # 与 CUDAGraphWrapper 的区别：wrapper 包住一个编译好的段、按 num_tokens 分键，由 forward_context 驱动；
    #   这里按调用方给的任意键（V4 是 (layer_name, num_tokens, bucket_bs, q_eff)），录制 / 回放由 piecewise_core 决定，
    #   而且自己管输入 buffer（input_buffers / replay 里的刷新），因为切分算子的输入不一定来自固定地址
    # 唯一实例：deepseek_v4.py 的模块级全局 v4_attn_runner（tests/test_graph_holders.py 另有测试用实例）

    def __init__(self, capture_error_mode: str = "thread_local"):
        # 键 → {"graph": 图, "in": 回放前要刷新的输入 buffer, "out": 输出 buffer 或 None}
        self._graphs: dict = {}
        # 所有核心图共用的一个池（第一次 capture 时创建）；与稠密段的 _graph_pools 隔离
        self._pool = None
        # 与 CUDAGraphWrapper 相同的理由默认 thread_local（见 __call__ 里的注释）
        self._capture_error_mode = capture_error_mode
        # A runner is a module-level global of the model that uses it
        # (`deepseek_v4.py`), passed to the core per step rather than held by
        # anything the releaser can walk. See `atom/utils/graph_holders.py`.
        # 【译】runner 是使用它的模型的模块级全局变量（`deepseek_v4.py`），每步作为参数传给核心，
        #   而不是被释放方能遍历到的任何对象持有。见 `atom/utils/graph_holders.py`。
        register_graph_holder(self)

    def release_graphs(self) -> int:
        """Drop every recording, so what it captured can be freed.

        These graphs hold the base of the KV pool like any decode graph does --
        an attention core writes the KV it attends. `has_graph` answers False
        afterwards, so the next step runs the core eagerly (delivering into its
        output slot) until a capture pass records again.
        """
        # 【译】丢掉所有录像，让它们录下的东西可以被释放。
        #
        #   这些图像任何 decode 图一样持有 KV pool 的基址——attention 核心会写它要关注的 KV。之后 `has_graph` 返回 False，
        #   所以下一步核心以 eager 方式运行（结果送进它的输出 slot），直到某次录制轮重新录下。
        dropped = len(self._graphs)
        self._graphs.clear()
        # The next capture makes its own; see `CUDAGraphWrapper.release_graphs`
        # for why reusing this handle is not safe.
        # 【译】下一次录制自己建一个；为什么复用这个句柄不安全，见 `CUDAGraphWrapper.release_graphs`。
        self._pool = None
        return dropped

    # piecewise_core 用它判断"录制"还是"回放"
    def has_graph(self, key) -> bool:
        return key in self._graphs

    def input_buffers(
        self, named_args: dict, input_names: tuple, zero_copy: frozenset
    ) -> tuple[dict, dict]:
        """Where a graph captured now will read each input from, and which of
        those this runner has to refresh at replay.

        A zero-copy input is captured on the caller's OWN tensor: that address
        goes into the graph and nothing refreshes it, so whoever produces it has
        to keep handing back the same address every step. Everything else gets a
        clone this runner owns and copies into at replay -- one copy per step,
        bought in exchange for the producer being free to allocate as it likes.

        Names rather than positions: an index set silently means something else
        the moment an argument is added, and cannot be read without counting.
        """
        # 【译】现在录制的图会从哪里读每个输入，以及其中哪些要由这个 runner 在回放时刷新。
        #
        #   零拷贝输入直接在调用方*自己的*张量上录制：那个地址进了图，没人刷新它，所以产生它的一方必须每步都交回同一个地址。
        #   其余输入各得到一份这个 runner 自己持有的克隆，回放时拷进去——每步一次拷贝，换来生产者想怎么分配都行。
        #
        #   用名字而不是位置：一旦增加一个参数，下标集合就悄悄变了意思，而且不数一遍就读不懂。
        # 返回 (read_from, refresh)：read_from 喂给 warmup 和 capture；refresh 存进图的记录，replay 时刷新
        #   例：输入 {x, idx_q_quant}，zero_copy = {x} → read_from = {x: 原张量, idx_q_quant: 克隆}，refresh = {idx_q_quant: 克隆}
        #   调用方：attn_ffn_piecewise.piecewise_core 的录制分支
        read_from: dict = {}
        refresh: dict = {}
        for n in input_names:
            src = named_args[n]
            if src is None or n in zero_copy:
                read_from[n] = src
                continue
            clone = src.clone()
            read_from[n] = clone
            refresh[n] = clone
        return read_from, refresh

    def capture(
        self,
        key,
        in_bufs: dict,
        refresh: dict,
        compute_fn: Callable,
        out_buf: torch.Tensor | None,
    ) -> None:
        """Record `compute_fn(**in_bufs)` under `key`, landing in `out_buf`.

        `out_buf` is None for a core that returns nothing -- one whose whole
        effect is on paged state. Then there is nothing to copy and nothing for
        replay to hand back.

        The output copy goes INSIDE the graph, so a later replay refreshes
        `out_buf` with no Python at all.

        Warm the compute up yourself first: workspace allocation, lazy init and
        autotuning are one-time work that must not be recorded. That warmup is
        also what gives you the sample `out_buf` is shaped from.
        """
        # 【译】把 `compute_fn(**in_bufs)` 录在 `key` 下，结果落在 `out_buf`。
        #
        #   对一个什么都不返回的核心——它的全部作用都在分页状态上——`out_buf` 为 None。这时没有要拷的东西，
        #   回放也没有要交回的东西。
        #
        #   输出拷贝放在图*里面*，这样之后的回放不用任何 Python 就能刷新 `out_buf`。
        #
        #   先自己把计算预热一遍：workspace 分配、惰性初始化和自动调优都是一次性的工作，不能被录进去。
        #   预热也正好给出 `out_buf` 用来定形状的样本。
        # 调用方：piecewise_core 的录制分支（它先 warm = core(**read_from)，再调这里）
        graph = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            stream=torch.cuda.current_stream(),
            capture_error_mode=self._capture_error_mode,
        ):
            result = compute_fn(**in_bufs)
            if out_buf is not None:
                out_buf.copy_(result)
        self._graphs[key] = {"graph": graph, "in": refresh, "out": out_buf}

    def replay(self, key, named_args: dict) -> torch.Tensor | None:
        """Refresh the inputs this runner owns, replay, and return the output."""
        # 【译】刷新这个 runner 持有的输入，回放，并返回输出。
        # 调用方：piecewise_core（AF_PIECEWISE 的真实 decode 步、键已录过时）
        entry = self._graphs[key]
        # Only the inputs this runner cloned are here. A zero-copy one is not:
        # the graph reads the producer's own tensor, so there is nothing to copy.
        # 【译】这里只有这个 runner 克隆过的输入。零拷贝输入不在其中：图读的是生产者自己的张量，没什么可拷的。
        for n, buf in entry["in"].items():
            src = named_args[n]
            if src is not None:
                buf.copy_(src)
        entry["graph"].replay()
        # 【译】（行尾注释）不返回值的核心为 None；否则是图拷进去的那块输出
        return entry["out"]  # None for a void core; else the graph copied it in
