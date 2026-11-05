# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【导读】backends.py：VllmBackend —— ATOM 给 torch.compile 的自定义后端：定缓存目录、切图、逐子图编译（类名沿用 vLLM）
#   位置：decorators.py 的 TorchCompileWrapperWithCustomDispatcher.__init__ 经 init_backend 为每个被装饰的模型实例建一个
#     VllmBackend，当作 torch.compile(forward, backend=...) 的 backend；Dynamo 抓完图后调用 VllmBackend.__call__(graph, example_inputs)，
#     每个实例只调这一次（--level 2 / 3；level 0 / 1 不会用到本文件）
#   概念（Dynamo 抓图、"编译一次直接分发"、切图、动态形状、缓存目录与失效）见 decorators.py 文件头【科普】①~⑥，这里只讲实现
#
# VllmBackend.__call__ 的流程（一个模型实例只走一遍，发生在 warmup 的第一次 forward 里）：
#   输入：graph = 整个 forward 的 FX 图；example_inputs = 这次调用的真实输入（含权重等被提升为图输入的张量）
#     │
#     ├─ ① 定缓存目录：cache_dir 为空 → md5([Config.compute_hash(), 追踪文件 hash, 编译器 hash])[:10]
#     │     → ~/.cache/atom/torch_compile_cache/<10位hash>/rank_<GPU号>/<图标签>/（【科普】⑤）
#     │     cache_dir 已经由同一模型的前一张图算好（DSpark 草稿、V4.1 的 late 层）→ _graph_tag 给它一个独立的子目录
#     ├─ ② CompilerManager.initialize_cache：读入上次留下的 vllm_compile_cache.py（产物句柄表），并让编译器指向这个目录
#     ├─ ③ split_graph：在 spliting_op 节点处切成 [submod_0, submod_1, …]（【科普】③：Qwen3-8B → 73 个子模块）
#     ├─ ④ PiecewiseCompileInterpreter(split_gm).run(*example_inputs)：用 fake 张量把切好的图"假跑"一遍，
#     │     每跑到一个稠密子图：CompilerManager.compile（缓存命中 → load；否则 Inductor 编译、存盘）
#     │       → 用 PiecewiseBackend 包住（PIECEWISE 类 --cudagraph-mode 再包一层 CUDAGraphWrapper）
#     │       → 替换掉 split_gm 里的这个子模块
#     ├─ ⑤ 第一次时把切好的图写成 computation_graph.py（可读的 Python，调试用）
#     └─ ⑥ 把 split_gm 返回给 Dynamo（cudagraph_copy_inputs=True 时返回带"先拷进静态 buffer"的包装 copy_and_call；ATOM 默认 False）
#   之后每一步：decorators.py 的 dispatch_to_code 执行改写后的字节码 → 调 split_gm(*args)
#     → 按顺序调用 submod_0（PiecewiseBackend）→ submod_1（attention 自定义算子，eager）→ submod_2 → …
#
# 阅读顺序：VllmBackend.__call__ → split_graph → PiecewiseCompileInterpreter.call_module → CompilerManager.compile / load
#   → cuda_piecewise_backend.py（运行时分发）
# 对比nano-vllm：nano-vllm 没有自定义后端，它的 @torch.compile 用默认的 Inductor 后端，不切图、不自己管理缓存目录

import ast
import dataclasses
import logging
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.utils import (
    compilation_counter,
    is_torch_equal_or_newer,
)
from torch._dispatch.python import enable_python_dispatcher

from .compiler_inferface import (
    CompilerInterface,
    InductorAdaptor,
    InductorStandaloneAdaptor,
)

logger = logging.getLogger("atom")


# 返回静态图包装类的路径字符串（vLLM 遗留接口）；ATOM 里没有调用方，PIECEWISE 包装直接 import CUDAGraphWrapper（见下面 call_module）
def get_static_graph_wrapper_cls(cls) -> str:
    return "atom.utils.cuda_graph.CUDAGraphWrapper"


# 选择编译器适配器（compiler_inferface.py）：use_inductor=True（默认，ATOM 里没有地方改它）时，
#   PyTorch >= 2.8 用 InductorStandaloneAdaptor（torch._inductor.standalone_compile，产物可单独存盘 / 加载），
#   更老的版本用 InductorAdaptor（钩住 Inductor 内部的 FX graph cache 来存取产物）
# 观察：use_inductor=False 时函数没有 return（EagerAdaptor 分支被注释掉了），返回 None，
#   之后 CompilerManager 访问 self.compiler.name / compute_hash 会报 AttributeError
# 调用方：CompilerManager.__init__
def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    # return InductorAdaptor()
    if compilation_config.use_inductor:
        if is_torch_equal_or_newer("2.8.0.dev"):
            logger.debug("Using InductorStandaloneAdaptor")
            return InductorStandaloneAdaptor()
        else:
            logger.debug("Using InductorAdaptor")
            return InductorAdaptor()
    # else:
    #     logger.info("Using EagerAdaptor")
    #     return EagerAdaptor()


class CompilerManager:
    """
    A manager to manage the compilation process, including
    caching the compiled graph, loading the compiled graph,
    and compiling the graph.

    The cache is a dict mapping
    `(runtime_shape, graph_index, backend_name)`
    to `any_data` returned from the compiler.

    When serializing the cache, we save it to a Python file
    for readability. We don't use json here because json doesn't
    support int as key.
    """

    # 【译】管理编译过程的管理器，包括缓存编译好的图、加载编译好的图、以及编译图。
    #
    #   缓存是一个 dict，把 `(runtime_shape, graph_index, backend_name)` 映射到编译器返回的 `any_data`。
    #
    #   序列化缓存时，我们把它存成一个 Python 文件，便于阅读。这里不用 json，是因为 json 不支持用 int 作键。
    # 解释：每个 VllmBackend 有一个 CompilerManager（也就是每张图一个）。缓存表的样子（PyTorch >= 2.8，Qwen3-8B 的 37 个稠密子图）：
    #   {   (None, 0, 'inductor_standalone'): ('artifact_shape_None_subgraph_0',          ← 键：(形状, 第几个子图, 编译器名)
    #                                          '~/.cache/atom/torch_compile_cache/<hash>/rank_0/backbone/artifact_shape_None_subgraph_0'),  ← 句柄：(键名, 产物路径)
    #       (None, 1, 'inductor_standalone'): (…subgraph_1 …),       ← 第 1 个稠密子图，以此类推
    #       …
    #       (None, 36, 'inductor_standalone'): (…subgraph_36 …)  }   ← 最后一个（第 36 个）稠密子图
    #   runtime_shape=None 表示 general shape（动态 token 数）版本；compile_sizes 里的具体尺寸会以整数出现（ATOM 默认没有）
    #   键里不含任何"图内容"的信息——命中就加载，这是 decorators.py【科普】⑤ 里"旧产物配新代码"的根源

    def __init__(self, compilation_config: CompilationConfig):
        # cache：上面说的句柄表，initialize_cache 时从 vllm_compile_cache.py 读入
        # is_cache_updated：本次有没有新编译出产物；没有就不用重写文件（save_to_file）
        self.cache: dict[tuple[Optional[int], int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)

    # 编译器自己的 hash（torch 版本、系统信息，见 compiler_inferface.py get_inductor_factors）；
    #   VllmBackend.__call__ 把它作为缓存目录 hash 的因素 3
    def compute_hash(self, vllm_config: Config) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        """
        Initialize the cache directory for the compiler.

        The organization of the cache directory is as follows:
        cache_dir=/path/to/hash_str/rank_i_j/prefix/
        inside cache_dir, there will be:
        - vllm_compile_cache.py
        - computation_graph.py
        - transformed_code.py

        for multiple prefixes, they can share the same
        base cache dir of /path/to/hash_str/rank_i_j/ ,
        to store some common compilation artifacts.
        """
        # 【译】为编译器初始化缓存目录。
        #
        #   缓存目录的组织方式如下：
        #   cache_dir=/path/to/hash_str/rank_i_j/prefix/
        #   cache_dir 里会有：
        #   - vllm_compile_cache.py（句柄表）
        #   - computation_graph.py（切好的图）
        #   - transformed_code.py（Dynamo 改写后的代码）
        #
        #   多个 prefix 可以共用同一个基础目录 /path/to/hash_str/rank_i_j/，用来存放一些公共的编译产物。
        # 解释：ATOM 里 rank 目录只有一级 rank_<GPU号>（没有 _j）；prefix 就是图标签（backbone / drafter-<hash> / backbone_late）。
        #   transformed_code.py 只有设置了 debug_dump_path 才会写（而且写在 debug_dump_path 下，见 decorators.py bytecode_hook）
        #   调用方：VllmBackend.__call__ 的第②步

        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            # load the cache from the file
            # 【译】从文件加载缓存
            with open(self.cache_file_path) as f:
                # we use ast.literal_eval to parse the data
                # because it is a safe way to parse Python literals.
                # do not use eval(), it is unsafe.
                # 【译】用 ast.literal_eval 解析数据，因为它能安全地解析 Python 字面量。不要用 eval()，那不安全。
                self.cache = ast.literal_eval(f.read())

        # 让编译器适配器也知道这个目录：standalone 版把产物存到 cache_dir/<key>；旧版 InductorAdaptor 把
        #   TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR 环境变量指到上一级目录（compiler_inferface.py）
        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    # 把句柄表用 pprint 写回 vllm_compile_cache.py（下次启动 initialize_cache 读它）；没有新产物或禁用缓存时不写
    # 调用方：cuda_piecewise_backend.py PiecewiseBackend.check_for_ending_compilation（最后一个子图第一次运行时）
    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    # 查句柄表：键 (runtime_shape, graph_index, 编译器名) 不在表里返回 None（未命中）；
    #   在表里就让编译器按句柄把产物加载成可调用对象。注意：只看键，不比对 graph 的内容
    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Optional[Callable]:
        if (runtime_shape, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via " "handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph

    # 编译一个子图（或从缓存加载），返回可调用对象
    # 输入：graph 一个稠密子图（fx.GraphModule）；example_inputs 它的示例输入（general shape 时是 fake 张量 + SymInt，
    #   具体尺寸时是真实张量）；graph_index / num_graphs 第几个、共几个稠密子图（Qwen3-8B：0~36 / 37）；
    #   runtime_shape None = general shape，整数 = compile_sizes 里的具体 token 数
    # 步骤：先试缓存 → 未命中则交给编译器 compile → 有句柄就记进缓存表 → mark-trace 时从产物重新加载 → 记耗时
    # 调用方：PiecewiseCompileInterpreter.call_module（general shape）、PiecewiseBackend.__call__（具体尺寸）
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        additional_inductor_config,
        compilation_config: CompilationConfig,
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
    ) -> Any:
        # print("runtime_shape", runtime_shape)
        if graph_index == 0:
            # before compiling the first graph, record the start time
            # 【译】编译第一个子图之前，记下开始时间
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        # 【译】先尝试从缓存加载
        compiled_graph = self.load(graph, example_inputs, graph_index, runtime_shape)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                # after loading the last graph for this shape, record the time.
                # there can be multiple graphs due to piecewise compilation.
                # 【译】加载完这个形状的最后一个子图后，记录耗时。分段编译时会有多个子图。
                # 解释：日志 "Directly load the compiled graph(s) ... from the cache" 就表示这次启动命中了缓存
                now = time.time()
                elapsed = now - compilation_start_time
                if runtime_shape is None:
                    logger.info(
                        "Directly load the compiled graph(s) for dynamic shape "
                        "from the cache, took %.3f s",
                        elapsed,
                    )
                else:
                    logger.info(
                        "Directly load the compiled graph(s) for shape %s "
                        "from the cache, took %.3f s",
                        str(runtime_shape),
                        elapsed,
                    )
            return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        # 【译】没有编译器缓存过这个图，或者缓存被禁用了，需要编译它
        if isinstance(self.compiler, InductorAdaptor):
            # Let compile_fx generate a key for us
            # 【译】让 compile_fx 替我们生成键
            maybe_key = None
        else:
            # standalone 版：产物存到 local_cache_dir/artifact_shape_None_subgraph_<i>（目录名就是这个键）
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, additional_inductor_config, runtime_shape, maybe_key
        )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        # 【译】把产物记进缓存
        # 解释：句柄为 None 表示产物不能序列化（compiler_inferface.py _save_standalone_compiled_graph 会打警告），
        #   这时内存里的编译结果照样能用，只是下次启动还要重新编译
        if handle is not None:
            self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                # 【译】只为第一个子图打 info 级别的日志
                if runtime_shape is None:
                    logger.info("Cache the graph for dynamic shape for later use")
                else:
                    logger.info(
                        "Cache the graph of shape %s for later use", str(runtime_shape)
                    )
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via " "handle %s",
                    graph_index,
                    self.compiler.name,
                    handle,
                )
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index,
                    str(runtime_shape),
                    self.compiler.name,
                    handle,
                )

        # When mark-trace is enabled, post-processing may rewrite generated
        # artifact sources. Force reloading from cache artifact so the current
        # process uses the rewritten artifact immediately.
        # 【译】mark-trace 打开时，后处理可能会改写生成的产物源码。强制从缓存产物重新加载，让当前进程立刻用上改写后的产物。
        # 解释：改写指 compiler_inferface.py 用 graph_marker_instrumentation 把 <名字>_start/_end 之间包上 record_function
        #   （见 decorators.py mark_trace 上方的说明）；内存里的 compiled_graph 是改写之前的版本
        try:
            from atom.utils.graph_marker import is_graph_marker_enabled

            force_reload = is_graph_marker_enabled()
        except Exception:
            force_reload = False
        if force_reload and handle is not None and not self.disable_cache:
            try:
                reloaded_graph = self.load(
                    graph, example_inputs, graph_index, runtime_shape
                )
                if reloaded_graph is not None:
                    compiled_graph = reloaded_graph
                    logger.info(
                        "Force reloaded compiled graph from cache artifact "
                        "(graph_index=%s, runtime_shape=%s).",
                        graph_index,
                        runtime_shape,
                    )
            except Exception:
                logger.exception(
                    "Failed to force reload compiled graph from cache artifact; "
                    "falling back to in-memory compiled callable."
                )

        # after compiling the last graph, record the end time
        # 【译】编译完最后一个子图后，记录结束时间
        # 解释：耗时累加进 compilation_config.compilation_time；日志 "Compiling a graph for dynamic shape takes X s"
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            if runtime_shape is None:
                logger.info("Compiling a graph for dynamic shape takes %.2f s", elapsed)
            else:
                logger.info(
                    "Compiling a graph for shape %s takes %.2f s",
                    runtime_shape,
                    elapsed,
                )

        return compiled_graph


# split_graph 输出的一项：一个子模块
#   submod_name         子模块名，如 "submod_2"（split_module 生成的属性名）
#   graph_id            名字里的数字 2，即 split_graph 分配的子图号
#   is_splitting_graph  True = 这是切分点（attention 自定义算子）本身，不编译；False = 稠密子图，要编译
#   graph               子模块对象（fx.GraphModule）
@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


# used to judge whether the node should be split or not
# 【译】用来判断一个节点要不要被切开（即是否为切分点）
def _split_judge_func(node: fx.Node) -> bool:
    # ATOM use mark_spliting_op to mark the attn as splitting op
    # 【译】ATOM 用 mark_spliting_op 把 attention 标成切分点
    # 解释：只认 call_function 节点（Dynamo 图里调用 torch.ops.aiter.xxx 的节点，target 是这个算子对象），
    #   且算子对象上有 spliting_op=True 属性——mark_spliting_op 在注册完自定义算子后给 torch.ops.aiter.<名字> 设的（utils/__init__.py）
    if node.op == "call_function" and (
        hasattr(node.target, "spliting_op") and (node.target.spliting_op)
    ):
        return True

    return False


# 在切分点把整张 FX 图切成一串子模块（概念与 Qwen3-8B 的完整切法见 decorators.py 文件头【科普】③）
# 输入：graph 整个 forward 的 FX 图；ops 本应是切分算子名列表（compilation_config.splitting_ops）
# 输出：(split_gm, [SplitItem…])：split_gm 是一个新的 GraphModule，它的 forward 依次调用 submod_0、submod_1……；
#   列表按子图号排好序
# 观察：ops 参数在函数体里没有被用到，切不切只看节点 target 上的 spliting_op 属性（_split_judge_func）。
#   Config 在 level 3 下填的 splitting_ops（"aiter.unified_attention_with_output"、"aiter.mla_attention" 等，见 config.py
#   set_splitting_ops_for_v1）与实际注册的切分算子名（如 aiter.unified_attention_with_output_base）也对不上——
#   这个列表在 ATOM 里只是摆设，改它不会改变切图
# 调用方：VllmBackend.__call__ 第③步
#
# 切分算法：按节点顺序扫一遍，给每个节点一个子图号 subgraph_id；遇到切分点，它独占一个号，前后的普通节点各自成段。
#   例（2 层的简化图）：
#     节点                              是切分点？   subgraph_id 怎么变                       分到
#     embed、ln0、qkv0                   否          0                                       0
#     attn0                              是          先 +1 → 1（attn0 用它，记进 split_op_graphs），再 +1 → 2   1
#     o0、ln0'、mlp0、ln1、qkv1           否          2                                       2
#     attn1                              是          → 3，再 → 4                              3
#     o1、mlp1、norm                     否          4                                       4
#   → split_module 生成 submod_0 ~ submod_4，split_op_graphs = [1, 3]；N 个切分点得到 2N+1 个子模块
#   某个号若一个节点都没有（例如图一开头就是切分点，号 0 为空），split_module 不会生成它，子图号就有空缺
#   placeholder（图输入）和 output 节点不参与编号，split_module 会自动把每个子模块需要的输入接好
def split_graph(
    graph: fx.GraphModule, ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    # 【译】按算子切图
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if _split_judge_func(node):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    # 【译】`keep_original_order` 很重要！否则 pytorch 可能会重排节点，图里有原地修改（mutation）时，图的语义就会改变。
    # 解释：例如 attention 算子会写 KV cache、某些 op 原地改 buffer，它们和读取方之间不一定有数据依赖边；
    #   保持原顺序，子模块就按原代码顺序被调用（deepseek_v4.py 的 v4_attn_compress 也依赖这一点保证先后）
    # split_module 的第三个参数：一个 节点 → 子图号 的函数
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    # named_modules() 会列出根模块（名字 ""）、直接子模块（"submod_0"…）和更深的子模块（带 "."，如 "submod_0.xxx"），只要直接子模块
    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            # 【译】递归的孙模块，或根模块本身
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by integer graph_id, rather than string name
    # 【译】按整数 graph_id 排序，而不是按字符串名字
    # 按字符串排的话 "submod_10" 会排在 "submod_2" 前面
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# 当前这批子图（同一个形状）编译 / 加载的开始时间：CompilerManager.compile 在 graph_index 0 时记下，最后一个子图时算耗时
compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    # 【译】代码改编自 `torch.fx.passes.shape_prop.ShapeProp`。它用 fake 输入运行给定的图，并用给定的编译配置编译
    #   `compile_submod_names` 指定的那些子模块。
    #
    #   注意：`compile_submod_names` 里的顺序很重要，它决定编译出的分段子图的顺序。第一个子图负责打日志，
    #   最后一个子图有一些特殊的 cudagraph 输出处理。
    # 解释：torch.fx.Interpreter 会逐个节点"执行"一张 FX 图；这里重写了 call_module：执行到 split_gm 的每个子模块时，
    #   先用原子模块（fake 张量）算出输出形状，好让下游子模块拿到正确的示例输入；若它是要编译的稠密子图，就编译它，
    #   并把 split_gm 上的这个子模块换成编译后的 PiecewiseBackend。"假跑"本身只用 fake 张量（attention 子模块走它注册的
    #   fake 实现），不跑真 kernel、不占显存；真正耗时的是其中对每个稠密子图的 Inductor 编译。
    #   "最后一个子图的特殊处理"：PIECEWISE 时它的 CUDAGraphWrapper 设 weak_ref_output（输出不被别的图读，可以只留弱引用省显存）
    # 调用方：VllmBackend.__call__ 第④步

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: Config,
        vllm_backend: "VllmBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        # Dynamo 抓图时用的 FakeTensorMode（fake 张量只有形状 / dtype / 设备，没有数据）；从当前追踪上下文里取出来复用
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        # 【译】为 True 时，出错会把整个 torch.fx.Graph 打印出来，很烦人。
        self.extra_traceback = False

    # 把真实示例输入转成 fake 张量后执行整张 split_gm；enable_python_dispatcher 让带符号形状（SymInt）的算子能在 Python 层分发
    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    # Interpreter 每执行到 split_gm 里的一个子模块调用（call_module 节点）就进这里；target 是子模块名，如 "submod_2"
    # 先照常（用 fake 张量）执行它得到 output 供下游使用，再对稠密子图做编译 + 替换；返回 output
    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        # 只处理要编译的稠密子图；切分点子模块（attention）保持原样，运行时 eager 调用
        # index 就是 graph_index：这个稠密子图在 compile_submod_names 里的位置（Qwen3-8B：submod_0 → 0，submod_2 → 1，…，submod_72 → 36）
        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            # 这个子模块的参数里哪些位置是 SymInt（符号整数，即 token 数 s0）：Dynamo 把用到的符号尺寸作为显式参数传进子模块，
            #   PiecewiseBackend 运行时从这个位置读出真实 token 数，用来按 compile_sizes 分发
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            # 编译（或从缓存加载）这个子图的 general shape 版本：runtime_shape=None，示例输入是 fake 张量 + SymInt
            compiled_graph_for_dynamic_shape = (
                self.vllm_backend.compiler_manager.compile(
                    submod,
                    args,
                    self.compilation_config.inductor_compile_config,
                    self.compilation_config,
                    graph_index=index,
                    num_graphs=len(self.compile_submod_names),
                    runtime_shape=None,
                )
            )
            # Lazy import here to avoid circular import
            # 【译】在这里延迟导入，避免循环导入
            from .cuda_piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.vllm_backend,
            )

            # Wrap each compiled dense piece in a PIECEWISE CUDAGraphWrapper so
            # it captures/replays its own cudagraph (attention runs eager between
            # pieces). Gated on cudagraph_mode requesting piecewise; otherwise
            # keep the bare piecewise_backend (compiled, no per-piece cudagraph —
            # the pre-existing behavior, e.g. FULL manual capture in the runner).
            # 【译】把每个编译好的稠密片段包进一个 PIECEWISE 的 CUDAGraphWrapper，让它捕获 / 回放自己的 cudagraph
            #   （attention 在片段之间 eager 运行）。只在 cudagraph_mode 要求分段时这样做；否则保留裸的 piecewise_backend
            #   （已编译，但没有逐片段的 cudagraph——原有行为，例如 runner 里手动的 FULL 整图捕获）。
            _cg_mode = self.compilation_config.cudagraph_mode
            # Driven by --cudagraph-mode. Default FULL -> requires_piecewise is
            # False -> bare piecewise_backend (compiled pieces inside the manual
            # FULL whole-forward capture, i.e. existing behavior). PIECEWISE ->
            # wrap each piece in its own cudagraph.
            # 【译】由 --cudagraph-mode 决定。默认 FULL -> requires_piecewise 为 False -> 裸的 piecewise_backend
            #   （编译好的片段被包在手动的 FULL 整个 forward 捕获里，即原有行为）。PIECEWISE -> 每个片段包进它自己的 cudagraph。
            # 解释：requires_piecewise_compilation() 对 PIECEWISE、FULL_AND_PIECEWISE、AF_PIECEWISE 为 True（config.py 的成员表）。
            #   这里在编译期就决定装不装 wrapper；--enforce-eager 时也照装，但 ModelRunner 不会把运行时模式设成 PIECEWISE，
            #   wrapper 每次都直接透传（cuda_graph.py CUDAGraphWrapper.__call__）
            _pw_cg = _cg_mode is not None and _cg_mode.requires_piecewise_compilation()
            if _pw_cg:
                from .cuda_graph import CUDAGraphOptions, CUDAGraphWrapper

                # CUDAGraphOptions：只有第一个片段打"正在捕获"的日志、只在第一个片段捕获前跑 gc / empty_cache
                #   （每层一张小图，每张都 gc 会让捕获非常慢），最后一个片段的输出只留弱引用
                # 为什么写 self.module.__dict__[target] 而不是 setattr：nn.Module 的 setattr 会把子模块登记进 _modules，
                #   而 PiecewiseBackend / CUDAGraphWrapper 不是 nn.Module；直接写进实例的 __dict__，属性查找会先找到它，
                #   split_gm 的 forward 里 self.submod_0(...) 调到的就是它（原来的子模块仍留在 _modules 里）
                self.module.__dict__[target] = CUDAGraphWrapper(
                    runnable=piecewise_backend,
                    vllm_config=self.vllm_config,
                    runtime_mode=CUDAGraphMode.PIECEWISE,
                    cudagraph_options=CUDAGraphOptions(
                        debug_log_enable=piecewise_backend.is_first_graph,
                        gc_disable=not piecewise_backend.is_first_graph,
                        weak_ref_output=piecewise_backend.is_last_graph,
                    ),
                )
            else:
                self.module.__dict__[target] = piecewise_backend

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


# the tag for the part of model being compiled,
# e.g. backbone/eagle_head
# 【译】正在被编译的那部分模型的标签，例如 backbone / eagle_head
# 解释：标签决定缓存子目录 rank_<GPU号>/<标签>/。VllmBackend 在构造时（即被装饰的模型 __init__ 时）读取当前值
model_tag: str = "backbone"


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    # 【译】设置模型标签的上下文管理器。
    # 调用方：ModelRunner.__init__ 在 with set_model_tag("drafter") 里构造投机解码的草稿模型；
    #   models/deepseek_v41/runtime.py 在 with set_model_tag("backbone_late") 里构造 V4.1 有界重放的后半段层
    #   ——同一个模型的两张图不能共用一个缓存子目录：它们会读写同一个 vllm_compile_cache.py，第二张图按
    #   (None, graph_index, 编译器名) 就会加载到第一张图的产物
    # 断言：不允许嵌套设置成同一个标签
    global model_tag
    assert (
        tag != model_tag
    ), f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


# 编译缓存的根目录（容器里以 root 运行时就是 CLAUDE.md 说的 /root/.cache/atom）；名字沿用 vLLM，ATOM 没有环境变量能改它
VLLM_CACHE_ROOT = os.path.expanduser("~/.cache/atom")
# Cache dirs `VllmBackend` derived from the config and the first graph's
# traced code, as opposed to one the user set (`_graph_tag`).
# 【译】由 `VllmBackend` 根据 config 和第一张图追踪到的代码推导出来的缓存目录，区别于用户自己设置的目录（见 `_graph_tag`）。
_DERIVED_CACHE_DIRS: set[str] = set()


def _hash_traced_files(files) -> str:
    """md5 over the paths and contents of the files Dynamo traced; the code
    factor of a graph's compile-cache key."""
    # 【译】对 Dynamo 追踪过的文件的路径和内容做 md5；这是一张图的编译缓存键里的"代码"因素。
    # 调用方：VllmBackend.__call__（缓存目录的因素 2）、_graph_tag（后续图的子目录后缀）
    import hashlib

    content = []
    for filepath in files:
        content.append(filepath)
        if filepath == "<string>" or filepath == "<frozen os>":
            # This means the function was dynamically generated, with
            # e.g. exec() or frozen os module. We can't actually check these.
            # 【译】这表示函数是动态生成的（例如用 exec()），或来自冻结的 os 模块。这些我们没法真正检查。
            continue
        with open(filepath) as f:
            content.append(f.read())
    return hashlib.md5("\n".join(content).encode(), usedforsecurity=False).hexdigest()


class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    # 【译】`torch.compile` 配合 vLLM 使用的编译后端。用于 `CompilationLevel.PIECEWISE` 编译级别，在那里我们定制编译过程。
    #
    #   这个后端的主要工作是把图切成分段子图，并把它们交给分段后端（piecewise backend）。
    #
    #   这个后端还会把 PostGradPassManager 加进 Inductor 的配置，用来处理 post-grad 的 pass。
    # 解释：ATOM 里 level 2 和 3 都用它（decorators.py 只在 level 0/1 时跳过）；PostGradPassManager 相关代码都被注释掉了，
    #   ATOM 没有自定义的 Inductor pass

    vllm_config: Config
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    # 【译】我们编译的那张图
    # 即整个 forward 的 FX 图
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    # 【译】把所有分段子图串起来的那个 graph module（split_gm）
    # piecewise_graphs：split_graph 输出的 SplitItem 列表
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    # 【译】在去函数化（defunctionalization）之前对图运行的 Inductor pass
    # ATOM 未使用 post_grad_passes；
    # sym_tensor_indices / input_buffers：只给 cudagraph_copy_inputs=True 的 copy_and_call 用（ATOM 默认不走）
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        vllm_config: Config,
        prefix: str = "",
    ):

        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        # 【译】如果模型初始化时带了非空的 prefix，通常用这个 prefix 就够了，例如 language_model、vision_model 等。
        #   当多个部分作为独立模型分别初始化时，需要用 model_tag 来区分它们，例如 backbone（默认）、eagle_head 等。
        # 解释：ATOM 里 init_backend 不传 prefix，所以 self.prefix 总是构造时的 model_tag（backbone / drafter / backbone_late）
        self.prefix = prefix or model_tag

        # Passes to run on the graph post-grad.
        # 【译】在 post-grad 阶段对图运行的 pass。
        # （下面一行是被注释掉的旧代码）
        # self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here
        # 【译】`torch.compile` 是即时（JIT）编译的，所以这里不需要做任何事
        # 解释：真正的工作都在 __call__ 里，等 Dynamo 第一次抓完图才发生

    # （以下是被注释掉的旧代码：配置 post-grad pass，ATOM 未启用）
    # def configure_post_pass(self):
    #     config = self.compilation_config
    #     self.post_grad_pass_manager.configure(self.vllm_config)

    #     # Post-grad custom passes are run using the post_grad_custom_post_pass
    #     # hook. If a pass for that hook exists, add it to the pass manager.
    #     inductor_config = config.inductor_compile_config
    #     PASS_KEY = "post_grad_custom_post_pass"
    #     if PASS_KEY in inductor_config:
    #         # Config should automatically wrap all inductor passes
    #         if isinstance(inductor_config[PASS_KEY], PostGradPassManager):
    #             assert (inductor_config[PASS_KEY].uuid() ==
    #                     self.post_grad_pass_manager.uuid())
    #         else:
    #             assert isinstance(inductor_config[PASS_KEY], InductorPass)
    #             self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
    #     inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def _graph_tag(self) -> str:
        """This graph's cache subdirectory when another graph of the model
        derived the cache dir first.

        The derived dir hashes only the files the FIRST graph traced, after
        which `traced_files` is cleared. A later graph of the same model (the
        DSpark drafter, V4.1's late layers under bounded replay) would
        otherwise load a stale artifact after an edit to a file only it
        traces, so its own traced files key its subdirectory. A cache dir the
        user configured keeps the plain tag.
        """
        # 【译】当这个模型的另一张图已经先推导出缓存目录时，本图的缓存子目录。
        #
        #   推导出的目录只 hash 了"第一张"图追踪到的文件，之后 `traced_files` 就被清空了。同一个模型后来的图
        #   （DSpark 草稿、V4.1 有界重放下的后半段层）在改动了只有它追踪的文件之后，本会加载到陈旧的产物，
        #   所以用它自己追踪到的文件来给它的子目录命名。用户自己配置的缓存目录则保持普通的标签。
        # 例：主模型先编译 → 目录 …/<hash>/rank_0/backbone/；草稿模型后编译 → …/<hash>/rank_0/drafter-<它的追踪文件 md5 前 10 位>/
        # 调用方：VllmBackend.__call__（cache_dir 已经有值时）
        if self.compilation_config.cache_dir not in _DERIVED_CACHE_DIRS:
            return self.prefix
        files = sorted(self.compilation_config.traced_files)
        self.compilation_config.traced_files.clear()
        if not files:
            return self.prefix
        code = _hash_traced_files(files)[:10]
        return f"{self.prefix}-{code}"

    # Dynamo 抓完图后调用的入口（每个实例只能调一次），六步流程见文件头
    # 输入：graph 整个 forward 的 FX 图；example_inputs 与图的输入一一对应（input_ids、positions、被提升为图输入的权重 / buffer、
    #   符号尺寸等，warmup 时的真实张量）
    # 输出：交给 Dynamo 放进改写后字节码里的可调用对象（通常就是 split_gm）
    # 调用方：Dynamo（torch.compile 的 backend 参数），时机是 decorators.py __call__ 的路径2
    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        vllm_config = self.vllm_config
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.
            # 【译】没有提供缓存目录，就根据已知的、会影响编译的因素生成一个。只要这些因素都不变，缓存目录就不变，
            #   这样就能复用编译好的图。
            # 解释：cache_dir 默认 ""，ATOM 没有命令行参数能设它，所以第一张图总会走这里；算出来后写回
            #   compilation_config.cache_dir，同一进程里后面的图（草稿、late 层）就走下面的 else 分支

            factors = []
            # 0. factors come from the env, for example, The values of
            # VLLM_PP_LAYER_PARTITION will affect the computation graph.
            # 【译】0. 来自环境变量的因素，例如 VLLM_PP_LAYER_PARTITION 的值会影响计算图。
            # 解释：下面两行被注释掉了——环境变量整体不进 hash。抓图时读到的 ATOM_* 开关会被冻结进图，
            #   改了开关却命中旧缓存，正是 decorators.py【科普】⑤ 说的"旧产物配新代码"（只有少数开关被 Config.compute_hash 单独收录）
            # env_hash = envs.compute_hash()
            # factors.append(env_hash)

            # 1. factors come from the vllm_config (it mainly summarizes how the
            #    model is created)
            # 【译】1. 来自 vllm_config 的因素（它主要概括了模型是怎样被创建的）
            # 解释：Config.compute_hash 汇总了量化 / 编译 / 并行子配置的 hash、TP、PCP、投机宽度等（详见 config.py 该方法）
            config_hash = vllm_config.compute_hash()
            factors.append(config_hash)

            # 2. factors come from the code files that are traced by Dynamo (
            #    it mainly summarizes how the model is used in forward pass)
            # 【译】2. 来自被 Dynamo 追踪的代码文件的因素（它主要概括了模型在 forward 中是怎样被使用的）
            # 解释：traced_files 由 decorators.py 的 patched_inline_call 在抓图期间收集；用完立即清空，
            #   后面的图（_graph_tag）就只看到它自己追踪的文件
            forward_code_files = list(sorted(self.compilation_config.traced_files))
            self.compilation_config.traced_files.clear()
            logger.debug(
                "Traced files (to be considered for compilation cache):\n%s",
                "\n".join(forward_code_files),
            )
            factors.append(_hash_traced_files(forward_code_files))

            # 3. compiler hash
            # 【译】3. 编译器的 hash
            # 解释：InductorStandaloneAdaptor / InductorAdaptor 的 compute_hash，即 torch 版本、系统信息（compiler_inferface.py）
            compiler_hash = self.compiler_manager.compute_hash(vllm_config)
            factors.append(compiler_hash)

            # combine all factors to generate the cache dir
            # 【译】把所有因素合起来生成缓存目录名
            # 例：factors = ['<Config hash 10位>', '<文件 md5 32位>', '<编译器 hash 10位>'] → str() 后再 md5，取前 10 位
            import hashlib

            hash_key = hashlib.md5(
                str(factors).encode(), usedforsecurity=False
            ).hexdigest()[:10]

            cache_dir = os.path.join(
                VLLM_CACHE_ROOT,
                "torch_compile_cache",
                hash_key,
                # "996ff366ca",
                # "e6292aa343",
            )
            # 记下"这个目录是推导出来的"，后面的图据此给自己加 hash 后缀（_graph_tag）
            self.compilation_config.cache_dir = cache_dir
            _DERIVED_CACHE_DIRS.add(cache_dir)
            graph_tag = self.prefix
        else:
            graph_tag = self._graph_tag()

        # 本图的本地缓存目录：<cache_dir>/rank_<当前 GPU 号>/<图标签>/
        #   用 torch.cuda.current_device()（设备号）而不是 TP rank（按 rank 的写法被注释掉了）：同一台机器上 DP / TP 的
        #   每个进程用不同的 GPU，各写各的子目录。local_cache_dir 写回 compilation_config 供日志 / 调试查看
        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir
        # rank = vllm_config.parallel_config.rank
        rank = torch.cuda.current_device()
        # dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}", graph_tag)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir

        # 写死为 False：ATOM 没有关闭编译缓存的开关，想不用缓存只能删目录（rm -rf /root/.cache/atom/*）
        disable_cache = False

        if disable_cache:
            logger.info("vLLM's torch.compile cache is disabled.")
        else:
            logger.info(
                "Using cache directory: %s for vLLM's torch.compile", local_cache_dir
            )

        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache, graph_tag
        )

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        # 【译】Dynamo 调用后端，意味着字节码改写和分析已经完成
        # 解释：从 decorators.py 的 start_monitoring_torch_compile 到这里的时间就是 Dynamo 抓图耗时（日志 "Dynamo bytecode transform time"）
        compilation_counter.num_graphs_seen += 1
        from atom.utils.decorators import torch_compile_start_time

        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        # 【译】编译过程由我们控制，每个实例只能被调用一次
        # 解释：被追踪的 forward 里发生 graph break 时，Dynamo 会带着第二张图再来调用同一个后端，就会在这里断言失败——
        #   这就是 CLAUDE.md "被追踪函数里不要加调试代码" 那条规则的报错现场（decorators.py 文件头）
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # self.configure_post_pass()

        # 第③步：切图（splitting_ops 实际不起作用，见 split_graph 上方的观察）
        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_config.splitting_ops
        )

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        # 【译】depyf 会钩住 lazy_format_graph_code 并把图转储出来用于调试，这里不需要打印图
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        # 只编译稠密子图（is_splitting_graph=False），顺序即 graph_index。切分点子模块从不编译，
        #   所以它也不会经过 AOT autograd 的死代码消除（deepseek_v4.py 的 v4_attn_compress 正是靠这一点）
        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        # 【译】把切好的图传播给分段后端，用符号形状编译各子模块
        # 解释：第④步。跑完之后 split_gm 上每个稠密子模块都已被替换成 PiecewiseBackend（或外面再包 CUDAGraphWrapper）
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*example_inputs)

        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
            # use `print_readable` because it can include submodules
            # 【译】代码改编自上面的 depyf 链接；用 `print_readable`，因为它能把子模块也打印出来
            # 解释：第⑤步。只在文件不存在时写，所以缓存目录里的 computation_graph.py 是第一次编译时的图
            src = (
                "from __future__ import annotations\nimport torch\n"
                + self.split_gm.print_readable(print_output=False)
            )
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.info("Computation graph saved to %s", graph_path)

        self._called = True

        # 第⑥步：ATOM 的常规出口。cudagraph_copy_inputs 默认 False 且没有地方改它：ModelRunner 自己预先分配了固定的输入
        #   buffer（forward_vars），每步把数据写进去再调模型，输入地址本来就固定，不需要在这里再拷一次。
        #   if 之后的 copy_and_call 分支在 ATOM 里不会执行（沿用 vLLM）
        if (
            self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or not self.compilation_config.cudagraph_copy_inputs
        ):
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        # 【译】如果需要为 cudagraph 复制输入缓冲区
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        # for weights and static buffers, they will have concrete shapes.
        # symbolic shape only happens for input tensors.
        # 【译】带符号形状（batch size）的张量的下标。权重和静态缓冲区的形状是具体的，符号形状只出现在输入张量上。
        from torch.fx.experimental.symbolic_shapes import is_symbolic

        self.sym_tensor_indices = [
            i
            for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
            and any(is_symbolic(d) for d in x.size())
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        # 【译】由编译器管理的 cudagraph 输入缓冲区。我们假设第一次带符号形状的运行在所有张量里尺寸最大。
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        # this is the callable we return to Dynamo to run
        # 【译】这是我们返回给 Dynamo 去运行的可调用对象
        # 解释：每次调用先把带符号形状的输入拷进静态 buffer 的前 runtime_shape 行，再用 buffer 调 split_gm，保证地址固定
        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                # 【译】把张量拷进静态缓冲区
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                # 【译】把 list_args 里的张量换成静态缓冲区
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call
