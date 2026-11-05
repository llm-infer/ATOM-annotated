# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【导读】compiler_inferface.py（C 级）：把 Inductor 包装成"编译一个子图 / 把产物存盘 / 从盘上读回"的统一接口（文件名拼写沿用上游）
#   调用方：backends.py 的 make_compiler 选适配器，CompilerManager 对每个稠密子图调 compile / load / compute_hash
#   与主干的关系：编译子系统的最底层，即 utils/decorators.py 文件头流程图里"交给 Inductor 编译，或从缓存目录直接加载"那一步；
#     概念（缓存目录、为什么命中旧产物会出错）见那里的【科普】⑤
#   两个适配器：
#     InductorStandaloneAdaptor（PyTorch >= 2.8）：torch._inductor.standalone_compile 编出可单独 save / load 的 CompiledArtifact，
#       存到 <本地缓存目录>/artifact_shape_<形状>_subgraph_<i>/，句柄 = (键名, 路径)
#     InductorAdaptor（PyTorch 2.5~2.7）：打补丁钩住 Inductor 内部的 FX graph cache，拿到图的 hash 和生成的文件路径作为句柄
#   另外两件事：导入本模块时给 ROCm 上的 Triton 打补丁（_patch_triton_cluster_dims_for_rocm）；
#     --mark-trace 时改写 Inductor 生成的 Python 代码，给 graph_marker 标出的区间包上 record_function（见 decorators.py mark_trace）
#   名字和报错信息里的 vLLM 是沿用上游；报错里说的 ~/.cache/vllm/torch_compile_cache 在 ATOM 里是 ~/.cache/atom/torch_compile_cache

import contextlib
import copy
import hashlib
import logging
import os
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch._inductor.compile_fx
from torch import fx

from atom.config import Config
from atom.utils import compilation_counter, is_torch_equal_or_newer

logger = logging.getLogger("atom")
# 本进程是否已经打过一次"产物不能存盘"的警告（之后同类消息降为 debug 级别）
_non_saveable_artifact_warning_emitted = False


def _log_non_saveable_artifact(key: str, reason: str) -> None:
    """Warn once per process, then keep per-subgraph details at debug level."""
    # 【译】每个进程只警告一次，之后每个子图的细节都只记在 debug 级别。
    global _non_saveable_artifact_warning_emitted
    log = logger.debug if _non_saveable_artifact_warning_emitted else logger.warning
    log("Skipping standalone compiled graph save for %s: %s", key, reason)
    _non_saveable_artifact_warning_emitted = True


def _save_standalone_compiled_graph(
    compiled_graph: Any, path: str, key: str
) -> tuple[str, str] | None:
    """Persist a Torch standalone artifact when the graph is serializable.

    Torch 2.13 exposes ``CompiledArtifact.is_saveable()`` and raises a
    ``RuntimeError`` (rather than the older ``AssertionError``) when an
    Inductor graph has no AOT Autograd artifact. The compiled callable is still
    valid in memory, so a cache miss must not abort model startup.
    """
    # 【译】图可序列化时，把 Torch 的 standalone 产物持久化到磁盘。
    #
    #   Torch 2.13 提供了 ``CompiledArtifact.is_saveable()``，并且在一个 Inductor 图没有 AOT Autograd 产物时抛 ``RuntimeError``
    #   （而不是旧版本的 ``AssertionError``）。编译出的可调用对象在内存里仍然有效，所以"存不了缓存"不能中止模型启动。
    # 返回 (键名, 路径) 作为句柄；存不了时返回 None（CompilerManager 于是不记这个子图，下次启动重新编译它）
    # 调用方：InductorStandaloneAdaptor.compile；测试：tests/test_compiler_interface.py
    is_saveable = getattr(compiled_graph, "is_saveable", None)
    if callable(is_saveable) and not is_saveable():
        _log_non_saveable_artifact(key, "PyTorch did not emit a serializable artifact")
        return None

    try:
        compiled_graph.save(path=path, format="unpacked")
    except AssertionError:
        _log_non_saveable_artifact(
            key, "PyTorch did not emit a complete unpacked artifact"
        )
        return None
    except RuntimeError as exc:
        if "CompiledArtifact.save failed to save" not in str(exc):
            raise
        _log_non_saveable_artifact(key, str(exc))
        return None

    compilation_counter.num_compiled_artifacts_saved += 1
    return key, path


def _extend_metadata_with_cluster_dims(md):
    """Return ``md`` with a no-op ``cluster_dims=(1, 1, 1)`` appended if it's a
    namedtuple missing that field (AMD has no thread-block clusters); else ``md``
    unchanged."""
    # 【译】如果 ``md`` 是缺少 ``cluster_dims`` 字段的 namedtuple，就返回追加了无作用的 ``cluster_dims=(1, 1, 1)`` 的版本
    #   （AMD 没有 thread-block cluster）；否则原样返回 ``md``。
    if md is not None and hasattr(md, "_fields") and not hasattr(md, "cluster_dims"):
        from collections import namedtuple

        extended = namedtuple("KernelMetadata", list(md._fields) + ["cluster_dims"])
        return extended(*md, (1, 1, 1))
    return md


def _patch_triton_cluster_dims_for_rocm() -> None:
    """Make compile-time Triton autotuning work on ROCm.

    ROCm Triton's ``CompiledKernel.metadata`` namedtuple has no ``cluster_dims``
    field (thread-block clusters are an NVIDIA Hopper feature). torch Inductor's
    autotune-at-compile-time launcher
    (``torch._inductor.runtime.triton_heuristics.*.make_launcher``) reads
    ``binary.metadata.cluster_dims`` with no fallback, so *any* kernel autotuned
    at compile time -- which ``torch._inductor.standalone_compile`` forces on by
    patching ``triton.autotune_at_compile_time=True`` -- crashes on AMD with
    ``AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'``.

    Two entry paths create the metadata, and BOTH must be patched:

    1. **Same-process compile** (``compile_threads==1``): the kernel is built by
       ``CompiledKernel.__init__`` in this process -- patch ``__init__`` to inject
       the field.
    2. **Subprocess compile** (default ``compile_threads>1``): the kernel is
       compiled in an ``async_compile`` worker (which never imports atom, so the
       ``__init__`` patch is absent there), pickled back, and rebuilt in this
       process by ``TritonCompileResult.__setstate__`` via
       ``CompiledKernel.__new__`` (bypassing ``__init__`` entirely) +
       ``_deserialize_metadata`` (which reconstructs the namedtuple from the
       worker's field dict, still without ``cluster_dims``). Patch
       ``__setstate__`` to inject the field into the rebuilt ``.metadata``.

    Injecting a no-op ``cluster_dims=(1, 1, 1)`` (single cluster == AMD's only
    mode) lets the launcher's ``cta_args`` computation succeed. No-op on CUDA,
    where the field already exists. Idempotent.
    """
    # 【译】让编译期的 Triton 自动调优（autotune）能在 ROCm 上工作。
    #
    #   ROCm 版 Triton 的 ``CompiledKernel.metadata`` namedtuple 没有 ``cluster_dims`` 字段（thread-block cluster 是 NVIDIA Hopper 的特性）。
    #   torch Inductor 的"编译期自动调优"启动器（``torch._inductor.runtime.triton_heuristics.*.make_launcher``）读取
    #   ``binary.metadata.cluster_dims`` 时没有兜底，所以*任何*在编译期做自动调优的 kernel——``torch._inductor.standalone_compile``
    #   会通过打补丁 ``triton.autotune_at_compile_time=True`` 强制打开它——在 AMD 上都会崩溃，报
    #   ``AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'``。
    #
    #   metadata 有两条创建路径，两条都必须打补丁：
    #
    #   1. **同进程编译**（``compile_threads==1``）：kernel 在本进程里由 ``CompiledKernel.__init__`` 构建——给 ``__init__`` 打补丁注入这个字段。
    #   2. **子进程编译**（默认的 ``compile_threads>1``）：kernel 在 ``async_compile`` 的工作进程里编译（那个进程从不导入 atom，
    #      所以那里没有 ``__init__`` 补丁），pickle 回来后，在本进程里由 ``TritonCompileResult.__setstate__`` 重建：先经
    #      ``CompiledKernel.__new__``（完全绕过 ``__init__``），再经 ``_deserialize_metadata``（用工作进程传来的字段字典重建 namedtuple，
    #      仍然没有 ``cluster_dims``）。给 ``__setstate__`` 打补丁，往重建出的 ``.metadata`` 里注入这个字段。
    #
    #   注入一个无作用的 ``cluster_dims=(1, 1, 1)``（单个 cluster == AMD 唯一的模式）能让启动器的 ``cta_args`` 计算成功。
    #   在 CUDA 上不起作用（那里本来就有这个字段）。可以重复调用（幂等）。
    # 调用方：本模块导入时执行一次（见函数定义下面那一行）；torch.version.hip 为 None（非 ROCm）时直接返回
    if getattr(torch.version, "hip", None) is None:
        return

    # Path 1: same-process CompiledKernel.__init__.
    # 【译】路径 1：同进程的 CompiledKernel.__init__。
    try:
        import triton.compiler.compiler as _tcc
    except ImportError:
        _tcc = None
    if _tcc is not None and not getattr(
        _tcc.CompiledKernel, "_atom_cluster_dims_patched", False
    ):
        _orig_init = _tcc.CompiledKernel.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.metadata = _extend_metadata_with_cluster_dims(
                getattr(self, "metadata", None)
            )

        _tcc.CompiledKernel.__init__ = _init
        _tcc.CompiledKernel._atom_cluster_dims_patched = True

    # Path 2: subprocess-compiled kernels rebuilt via pickle (the default,
    # compile_threads>1 path -- this is what --cudagraph-mode PIECEWISE without
    # --level 0 hits during Inductor autotune).
    # 【译】路径 2：在子进程里编译、经 pickle 重建的 kernel（默认的 compile_threads>1 路径——--cudagraph-mode PIECEWISE
    #   且不是 --level 0 时，Inductor 自动调优走的就是这条）。
    try:
        from torch._inductor.runtime.triton_heuristics import TritonCompileResult
    except ImportError:
        TritonCompileResult = None
    if TritonCompileResult is not None and not getattr(
        TritonCompileResult, "_atom_cluster_dims_patched", False
    ):
        _orig_setstate = TritonCompileResult.__setstate__

        def _setstate(self, state):
            _orig_setstate(self, state)
            # Only fix .metadata (read by make_launcher's cta_args); leave
            # packed_metadata untouched -- triton's C launch expects its exact
            # packed layout, so appending a field there would corrupt it.
            # 【译】只修 .metadata（make_launcher 计算 cta_args 时读它）；packed_metadata 保持不动——triton 的 C 启动代码
            #   要求它的打包布局完全不变，往那里追加字段会把它弄坏。
            self.kernel.metadata = _extend_metadata_with_cluster_dims(
                getattr(self.kernel, "metadata", None)
            )

        TritonCompileResult.__setstate__ = _setstate
        TritonCompileResult._atom_cluster_dims_patched = True


_patch_triton_cluster_dims_for_rocm()


# 编译器适配器的接口（基类）：name、initialize_cache、compute_hash、compile、load；下面两个 Inductor 适配器实现它
class CompilerInterface:
    """
    The interface for a compiler that can be used by vLLM.
    """

    # 【译】可被 vLLM 使用的编译器的接口。

    # The name of the compiler, e.g. inductor.
    # This is a class-level attribute.
    # 【译】编译器的名字，例如 inductor。这是一个类属性。
    # 解释：它是 backends.py CompilerManager 缓存键 (runtime_shape, graph_index, 编译器名) 的第三项
    name: str

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        """
        when the vLLM process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.

        prefix can be used in combination with cache_dir to figure out the base
        cache directory, e.g. there're multiple parts of model being compiled,
        but we want to share the same cache directory for all of them.

        e.g.
        cache_dir = "/path/to/dir/backbone", prefix = "backbone"
        cache_dir = "/path/to/dir/eagle_head", prefix = "eagle_head"
        """
        # 【译】当 vLLM 进程使用 `cache_dir` 作为缓存目录时，编译器应当用这个目录初始化自己，例如把它自己的缓存目录重定向到一个子目录。
        #
        #   prefix 可以和 cache_dir 一起用来推出基础缓存目录，例如模型有多个部分被编译，但我们希望它们共用同一个缓存目录。
        #
        #   例如：
        #   cache_dir = "/path/to/dir/backbone"，prefix = "backbone"
        #   cache_dir = "/path/to/dir/eagle_head"，prefix = "eagle_head"
        pass

    def compute_hash(self, vllm_config: Config) -> str:
        """
        Gather all the relevant information from the vLLM config,
        to compute a hash so that we can cache the compiled model.

        See [`VllmConfig.compute_hash`][vllm.config.VllmConfig.compute_hash]
        to check what information
        is already considered by default. This function should only
        consider the information that is specific to the compiler.
        """
        # 【译】从 vLLM 的 config 中收集所有相关信息，计算一个 hash，以便缓存编译好的模型。
        #
        #   默认已经考虑了哪些信息，见 [`VllmConfig.compute_hash`]（ATOM 里是 config.py 的 Config.compute_hash）。
        #   这个函数只应考虑编译器特有的信息。
        return ""

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        """
        Compile the graph with the given example inputs and compiler config,
        with a runtime shape. If the `runtime_shape` is None, it means
        the `example_inputs` have a dynamic shape. Otherwise, the
        `runtime_shape` specifies the shape of the inputs. Right now we only
        support one variable shape for all inputs, which is the batchsize
        (number of tokens) during inference.

        Dynamo will make sure `graph(*example_inputs)` is valid.

        The function should return a compiled callable function, as well as
        a handle that can be used to directly load the compiled function.

        The handle should be a plain Python object, preferably a string or a
        file path for readability.

        If the compiler doesn't support caching, it should return None for the
        handle. If the compiler fails to compile the graph, it should return
        None for the compiled function as well.

        `key` is required for StandaloneInductorAdapter, it specifies where to
        save the compiled artifact. The compiled artifact gets saved to
        `cache_dir/key`.
        """
        # 【译】用给定的示例输入和编译器配置编译这张图，并指定运行时形状。`runtime_shape` 为 None 表示 `example_inputs`
        #   带的是动态形状；否则 `runtime_shape` 指定了输入的形状。目前所有输入只支持一个可变维度，即推理时的 batch size
        #   （token 数）。
        #
        #   Dynamo 会保证 `graph(*example_inputs)` 是合法的。
        #
        #   这个函数应当返回一个编译好的可调用函数，以及一个可以用来直接加载该函数的句柄（handle）。
        #
        #   句柄应当是一个普通的 Python 对象，为了可读性最好是字符串或文件路径。
        #
        #   如果编译器不支持缓存，句柄应返回 None。如果编译器编译失败，编译出的函数也应返回 None。
        #
        #   StandaloneInductorAdapter 需要 `key`，它指定编译产物存放的位置：产物会被存到 `cache_dir/key`。
        return None, None

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Callable:
        """
        Load the compiled function from the handle.
        Raises an error if the handle is invalid.

        The handle is the second return value of the `compile` function.
        """
        # 【译】从句柄加载编译好的函数。句柄无效时抛出错误。
        #
        #   句柄就是 `compile` 函数的第二个返回值。
        raise NotImplementedError("caching is not supported")


# 一个"永远命中"的假 ShapeEnv：让 InductorAdaptor 在 Dynamo 追踪上下文之外查 / 存 Inductor 的 FX graph cache（只给旧适配器用）
class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    # 【译】为什么需要这个类：
    #
    #   在常规的 `torch.compile` 用法里，每次编译都有一次 Dynamo 字节码编译和一次 Inductor 编译。Inductor 编译发生在
    #   Dynamo 字节码编译的上下文里，这个上下文被用来确定动态形状信息等。
    #
    #   在我们的用法里，Dynamo 字节码编译只跑一次，而 Inductor 编译要跑多次：不同的具体形状各一次，再加 general shape 一次。
    #   具体形状的编译发生在 Dynamo 字节码编译的上下文之外，那时我们没有 shape environment 可以提供给 Inductor，
    #   Inductor 的代码缓存查找就会失败。
    #
    #   提供一个永远命中的假 shape environment，就能让 Inductor 的代码缓存查找总是命中，从而按需为不同形状编译图。
    #
    #   下面这几个假方法是反复试错、直到能跑通为止得出来的。

    def __init__(self) -> None:
        self.guards: list[Any] = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""


# 编译器 hash 的因素：系统信息（CacheBase.get_system，含设备、Triton 等版本）+ torch 自身的 key（torch_key）
# 调用方：两个适配器的 compute_hash → backends.py 缓存目录 hash 的因素 3（升级 torch 后缓存目录自然就变了）
def get_inductor_factors() -> list[Any]:
    factors: list[Any] = []
    # summarize system state
    # 【译】汇总系统状态
    from torch._inductor.codecache import CacheBase

    system_factors = CacheBase.get_system()
    factors.append(system_factors)

    # summarize pytorch state
    # 【译】汇总 pytorch 的状态
    from torch._inductor.codecache import torch_key

    torch_factors = torch_key()
    factors.append(torch_factors)
    return factors


# 按编译场景补充 Inductor 配置：为具体尺寸（runtime_shape 是 int）编译时打开 max_autotune 和 coordinate_descent_tuning
#   （ATOM 默认 compile_sizes=[]，只编 general shape，所以默认不会触发）；--mark-trace 时关掉 size_asserts、单线程编译
# 调用方：两个适配器的 compile
def set_inductor_config(config, runtime_shape):
    if isinstance(runtime_shape, int):
        # for a specific batchsize, tuning triton kernel parameters
        # can be beneficial
        # 【译】对一个具体的 batch size，调优 triton kernel 的参数可能有好处
        config["max_autotune"] = True
        config["coordinate_descent_tuning"] = True

    try:
        from atom.utils.graph_marker import is_graph_marker_enabled

        if is_graph_marker_enabled():
            config["size_asserts"] = False
            config["compile_threads"] = 1
    except Exception:
        pass


# 旧版适配器（PyTorch 2.5~2.7）：调 compile_fx，并用一组补丁钩住 Inductor 内部，取出图的 hash 和生成文件的路径作为句柄
class InductorAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler, version 2.5, 2.6, 2.7.
    """

    # 【译】Inductor 编译器的适配器，适用于 2.5、2.6、2.7 版本。

    name = "inductor"

    # 编译器 hash：get_inductor_factors 的 md5 前 10 位
    def compute_hash(self, vllm_config: Config) -> str:
        factors = get_inductor_factors()
        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str

    # 记下缓存目录，并把 TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR 指到 rank 目录下的 inductor_cache/、triton_cache/
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.base_cache_dir = cache_dir[: -len(prefix)] if prefix else cache_dir
        if disable_cache:
            return
        # redirect the cache directory to a sub-directory
        # set flags so that Inductor and Triton store their cache
        # in the cache_dir, then users only need to copy the cache_dir
        # to another machine to reuse the cache.
        # 【译】把缓存目录重定向到一个子目录：设置环境变量，让 Inductor 和 Triton 把它们的缓存存进 cache_dir，
        #   这样用户只需要把 cache_dir 拷到另一台机器上就能复用缓存。
        # 解释：base_cache_dir 是去掉图标签之后的 rank_<GPU号>/ 目录，所以同一个 rank 的各张图共用 inductor_cache/、triton_cache/
        inductor_cache = os.path.join(self.base_cache_dir, "inductor_cache")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        triton_cache = os.path.join(self.base_cache_dir, "triton_cache")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache

    # 调 compile_fx 编译一个子图；用补丁拿到 FX graph cache 的 hash 和生成文件路径，作为句柄 (hash_str, file_path) 返回
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        compilation_counter.num_inductor_compiles += 1
        from torch._inductor.compile_fx import compile_fx

        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)

        # disable remote cache
        # 【译】关闭远程缓存
        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        set_inductor_config(current_config, runtime_shape)

        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        # 【译】inductor 可能原地修改图，所以要先复制一份（见上面的 pytorch issue 链接）
        graph = copy.deepcopy(graph)

        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.
        # 【译】这是第一次编译这张图。假设没有嵌套的 Inductor 编译：compiled_fx_graph_hash 只会被调用一次，
        #   于是可以钩住它，直接拿到编译出的图的 hash。

        hash_str, file_path = None, None
        from torch._inductor.codecache import FxGraphCache, compiled_fx_graph_hash

        if torch.__version__.startswith("2.5"):
            original_load = FxGraphCache.load
            original_load_name = "torch._inductor.codecache.FxGraphCache.load"

            def hijack_load(*args, **kwargs):
                inductor_compiled_graph = original_load(*args, **kwargs)
                nonlocal file_path
                compiled_fn = inductor_compiled_graph.current_callable
                file_path = compiled_fn.__code__.co_filename  # noqa
                if not file_path.startswith(self.base_cache_dir):
                    # hooked in the align_inputs_from_check_idxs function
                    # in torch/_inductor/utils.py
                    # 【译】（拿到的是）torch/_inductor/utils.py 里 align_inputs_from_check_idxs 函数包的一层，
                    #   要到它的闭包里找真正生成的文件
                    for cell in compiled_fn.__closure__:
                        if not callable(cell.cell_contents):
                            continue
                        if cell.cell_contents.__code__.co_filename.startswith(
                            self.base_cache_dir
                        ):
                            # this is the real file path compiled from Inductor
                            # 【译】这才是 Inductor 编译出的真实文件路径
                            file_path = cell.cell_contents.__code__.co_filename
                            break
                return inductor_compiled_graph

            hijacked_compile_fx_inner = (
                torch._inductor.compile_fx.compile_fx_inner
            )  # noqa
        elif torch.__version__ >= "2.6":
            # function renamed in 2.6
            # 【译】2.6 里这个函数改了名
            original_load_name = None

            def hijacked_compile_fx_inner(*args, **kwargs):
                output = torch._inductor.compile_fx.compile_fx_inner(*args, **kwargs)
                nonlocal hash_str
                inductor_compiled_graph = output
                if inductor_compiled_graph is not None:
                    nonlocal file_path
                    compiled_fn = inductor_compiled_graph.current_callable
                    file_path = compiled_fn.__code__.co_filename  # noqa
                    if not file_path.startswith(self.base_cache_dir):
                        # hooked in the align_inputs_from_check_idxs function
                        # in torch/_inductor/utils.py
                        # 【译】同上：拿到的是 align_inputs_from_check_idxs 包的一层，到闭包里找真正生成的文件
                        for cell in compiled_fn.__closure__:
                            if not callable(cell.cell_contents):
                                continue
                            code = cell.cell_contents.__code__
                            if code.co_filename.startswith(self.base_cache_dir):
                                # this is the real file path
                                # compiled from Inductor
                                # 【译】这才是 Inductor 编译出的真实文件路径
                                file_path = code.co_filename
                                break
                    hash_str = inductor_compiled_graph._fx_graph_cache_key
                return output

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            out = compiled_fx_graph_hash(*args, **kwargs)
            nonlocal hash_str
            hash_str = out[0]
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            # 【译】不报错就表示可以缓存。Inductor 拒绝在 Dynamo 追踪上下文之外缓存图，也会对含高阶算子的图禁用缓存。
            #   对 vLLM 来说，这两种情况我们都希望缓存这张图（见上面的 codecache.py 链接）。
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with ExitStack() as stack:
            # hijack to get the compiled graph itself
            # 【译】劫持（打补丁），以拿到编译出的图本身
            if original_load_name is not None:
                stack.enter_context(patch(original_load_name, hijack_load))

            # for hijacking the hash of the compiled graph
            # 【译】用于劫持编译出的图的 hash
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.compiled_fx_graph_hash",
                    hijack_compiled_fx_graph_hash,
                )
            )

            # for providing a dummy shape environment
            # 【译】用于提供一个假的 shape environment
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    _get_shape_env,
                )
            )

            from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            # 【译】torch 2.8+（main 分支）的 AOTAutogradCache 里也用到 _get_shape_env
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env,
                    )
                )

            # for forcing the graph to be cached
            # 【译】用于强制缓存这张图
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache,
                )
            )

            # Dynamo metrics context, see method for more details.
            # 【译】Dynamo 的 metrics 上下文，详见 metrics_context 方法。
            stack.enter_context(self.metrics_context())

            # Disable remote caching. When these are on, on remote cache-hit,
            # the monkey-patched functions never actually get called.
            # vLLM today assumes and requires the monkey-patched functions to
            # get hit.
            # TODO(zou3519): we're going to replace this all with
            # standalone_compile sometime.
            # 【译】关闭远程缓存。开着时，一旦远程缓存命中，打过补丁的函数根本不会被调用；而 vLLM 目前假定并要求这些
            #   补丁函数会被调用到。TODO(zou3519)：以后会用 standalone_compile 全部替换掉这些。
            if is_torch_equal_or_newer("2.6"):
                stack.enter_context(
                    torch._inductor.config.patch(fx_graph_remote_cache=False)
                )
                # InductorAdaptor (unfortunately) requires AOTAutogradCache
                # to be turned off to run. It will fail to acquire the hash_str
                # and error if not.
                # StandaloneInductorAdaptor (PyTorch 2.8+) fixes this problem.
                # 【译】InductorAdaptor（很遗憾）要求关掉 AOTAutogradCache 才能运行，否则拿不到 hash_str 而报错。
                #   StandaloneInductorAdaptor（PyTorch 2.8+）解决了这个问题。
                stack.enter_context(
                    torch._functorch.config.patch(enable_autograd_cache=False)
                )
                stack.enter_context(
                    torch._functorch.config.patch(enable_remote_autograd_cache=False)
                )

            # with pass_context(runtime_shape):
            # ======= TODO  lirong =======
            # 【译】（被注释掉的旧代码）待办（lirong）：vLLM 原本在这里套 pass_context(runtime_shape)，ATOM 没有自定义 pass，暂未启用
            compiled_graph = compile_fx(
                graph,
                example_inputs,
                inner_compile=hijacked_compile_fx_inner,
                config_patches=current_config,
            )

        # 【译·报错信息】vLLM 编译模型失败。最可能的原因是之前某次编译失败、留下了损坏的编译产物。
        #   建议删除 ~/.cache/vllm/torch_compile_cache（ATOM 里是 ~/.cache/atom/torch_compile_cache）后重试，以看到真正的问题。
        if hash_str is None:
            raise RuntimeError(
                "vLLM failed to compile the model. The most "
                "likely reason for this is that a previous compilation "
                "failed, leading to a corrupted compilation artifact. "
                "We recommend trying to "
                "remove ~/.cache/vllm/torch_compile_cache and try again "
                "to see the real issue. "
            )
        assert (
            file_path is not None
        ), "failed to get the file path of the compiled graph"
        # Best-effort post-process the generated wrapper file too (PyTorch <2.8 path).
        # 【译】也尽力对生成的 wrapper 文件做后处理（PyTorch <2.8 的路径）。
        try:
            # Only run post-processing when mark-trace is enabled (to avoid any
            # overhead / file churn in default runs).
            # 【译】只在打开 mark-trace 时做后处理（避免默认运行时有任何额外开销或文件改动）。
            from atom.utils.graph_marker import is_graph_marker_enabled

            if is_graph_marker_enabled():
                # Local import to avoid extra package-level side effects.
                # 【译】局部导入，避免额外的包级副作用。
                from .graph_marker_instrumentation import (
                    instrument_record_functions_in_file,
                )

                instrument_record_functions_in_file(file_path, strip_markers=False)
        except Exception:
            pass
        return compiled_graph, (hash_str, file_path)

    # 从句柄 (hash_str, 文件路径) 在 FxGraphCache 里查回编译好的图（用 AlwaysHitShapeEnv 绕过形状检查），包成 Dynamo 调用约定的函数
    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Callable:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        hash_str = handle[0]

        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.codecache import FxGraphCache

        with ExitStack() as exit_stack:
            exit_stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    lambda *args, **kwargs: AlwaysHitShapeEnv(),
                )
            )
            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            # 【译】torch 2.8+（main 分支）的 AOTAutogradCache 里也用到 _get_shape_env
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                exit_stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        lambda *args, **kwargs: AlwaysHitShapeEnv(),
                    )
                )

            # Dynamo metrics context, see method for more details.
            # 【译】Dynamo 的 metrics 上下文，详见 metrics_context 方法。
            exit_stack.enter_context(self.metrics_context())

            if torch.__version__.startswith("2.5"):
                inductor_compiled_graph = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, False
                )
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )
            elif torch.__version__ >= "2.6":
                from torch._inductor.output_code import CompiledFxGraphConstantsWithGm

                constants = CompiledFxGraphConstantsWithGm(graph)
                inductor_compiled_graph, _ = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, None, constants
                )
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any
        # 【译】Inductor 的调用约定（函数签名）：f(list) -> tuple；Dynamo 的调用约定（函数签名）：f(*args) -> Any

        # need to know if the graph returns a tuple
        # 【译】需要知道这张图是否返回 tuple
        from torch._inductor.compile_fx import graph_returns_tuple

        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        # 【译】这是我们返回给 Dynamo 去运行的可调用对象
        def compiled_graph(*args):
            # convert args to list
            # 【译】把 args 转成 list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            # 【译】需要的话把 tuple 拆开
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph

    # 返回 Dynamo 的 metrics 上下文（torch >= 2.6），否则空上下文；compile / load 打补丁时都要进入它
    def metrics_context(self) -> contextlib.AbstractContextManager:
        """
        This method returns the Dynamo metrics context (if it exists,
        otherwise a null context). It is used by various compile components.
        Present in torch>=2.6, it's used inside FxGraphCache in
        torch==2.6 (but not after). It might also be used in various other
        torch.compile internal functions.

        Because it is re-entrant, we always set it (even if entering via Dynamo
        and the context was already entered). We might want to revisit if it
        should be set at a different level of compilation.

        This is likely a bug in PyTorch: public APIs should not rely on
        manually setting up internal contexts. But we also rely on non-public
        APIs which might not provide these guarantees.
        """
        # 【译】这个方法返回 Dynamo 的 metrics 上下文（存在的话，否则返回一个空上下文）。各种编译组件会用到它。
        #   它在 torch>=2.6 中存在，torch==2.6 的 FxGraphCache 内部会用到（之后的版本不用）。它也可能被 torch.compile 的
        #   其他各种内部函数用到。
        #
        #   因为它是可重入的，我们总是设置它（即使是经 Dynamo 进来、上下文已经进入过）。以后也许要重新考虑它应该在
        #   编译的哪一层设置。
        #
        #   这很可能是 PyTorch 的一个 bug：公开 API 不应依赖手动设置内部上下文。但我们也依赖了一些非公开 API，
        #   它们未必提供这些保证。
        if is_torch_equal_or_newer("2.6"):
            import torch._dynamo.utils

            return torch._dynamo.utils.get_metrics_context()
        else:
            return contextlib.nullcontext()


# 新版适配器（PyTorch >= 2.8，backends.py make_compiler 的默认选择）：torch._inductor.standalone_compile 编译，
#   CompiledArtifact.save 存成 <本地缓存目录>/<键名>/ 目录，load 时 CompiledArtifact.load 读回
class InductorStandaloneAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler.
    Requires PyTorch 2.8+.
    This is not on by default yet, but we plan to turn it on by default for
    PyTorch 2.8.

    Use VLLM_USE_STANDALONE_COMPILE to toggle this on or off.
    """

    # 【译】Inductor 编译器的适配器。需要 PyTorch 2.8+。默认还没有启用，但计划在 PyTorch 2.8 上默认启用。
    #
    #   用 VLLM_USE_STANDALONE_COMPILE 打开或关闭它。
    # 解释：docstring 是 vLLM 的原文。ATOM 里没有 VLLM_USE_STANDALONE_COMPILE 这个开关：PyTorch >= 2.8 时总是用本适配器

    name = "inductor_standalone"

    # 编译器 hash：系统信息 + torch 版本（与 InductorAdaptor 相同）
    def compute_hash(self, vllm_config: Config) -> str:
        factors = get_inductor_factors()
        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str

    # 只记下本地缓存目录；产物路径由 compile 的 key 决定
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        self.cache_dir = cache_dir

    # standalone_compile 编译一个子图，存盘到 <本地缓存目录>/<key>，返回 (可调用对象, 句柄或 None)；mark-trace 时再改写生成的代码
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        compilation_counter.num_inductor_compiles += 1
        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)
        set_inductor_config(current_config, runtime_shape)

        # 具体尺寸：按示例输入的真实形状特化；general shape：沿用 Dynamo 追踪上下文里的符号形状（s0）
        if isinstance(runtime_shape, int):
            dynamic_shapes = "from_example_inputs"
        else:
            dynamic_shapes = "from_tracing_context"

        from torch._inductor import standalone_compile

        # TODO Lirong
        # with pass_context(runtime_shape):
        # 【译】待办（Lirong）：（被注释掉的旧代码）同上，pass_context 暂未启用
        compiled_graph = standalone_compile(
            graph,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={"config_patches": current_config},
        )

        # Save the compiled artifact to disk in the specified path
        # 【译】把编译产物存到磁盘上指定的路径
        assert key is not None
        path = os.path.join(self.cache_dir, key)
        handle = _save_standalone_compiled_graph(compiled_graph, path, key)

        # Post-process generated wrapper Python files: wrap regions between
        # <prefix>_start / <prefix>_end graph markers with record_function("<prefix>").
        # 【译】对生成的 wrapper Python 文件做后处理：把 <prefix>_start / <prefix>_end 两个 graph marker 之间的区域
        #   包进 record_function("<prefix>")。
        try:
            # Only run post-processing when mark-trace is enabled (to avoid any
            # overhead / file churn in default runs).
            # 【译】只在打开 mark-trace 时做后处理（避免默认运行时有任何额外开销或文件改动）。
            from atom.utils.graph_marker import is_graph_marker_enabled

            if is_graph_marker_enabled() and handle is not None:
                # Local import to avoid extra package-level side effects.
                # 【译】局部导入，避免额外的包级副作用。
                from .graph_marker_instrumentation import (
                    instrument_record_functions_in_dir,
                )

                instrument_record_functions_in_dir(path, strip_markers=False)
        except Exception:
            # Best-effort: never fail compilation due to instrumentation.
            # 【译】尽力而为：绝不因为插桩失败而让编译失败。
            pass
        return compiled_graph, handle

    # 从句柄 (键名, 路径) 用 CompiledArtifact.load 读回产物，包一层以符合 Dynamo 的调用约定（必要时拆开单元素 tuple）
    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Callable:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        path = handle[1]
        inductor_compiled_graph = torch._inductor.CompiledArtifact.load(
            path=path, format="unpacked"
        )
        from torch._inductor.compile_fx import graph_returns_tuple

        returns_tuple = graph_returns_tuple(graph)

        def compiled_graph_wrapper(*args):
            graph_output = inductor_compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            # 【译】需要的话把 tuple 拆开。TODO(rzou)：这是否意味着 vLLM 读 Python 字节码的方式不对？
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper
