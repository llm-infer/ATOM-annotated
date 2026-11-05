# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【导读】cuda_piecewise_backend.py：PiecewiseBackend —— 一个稠密子图在运行时的"分发器"（类名里的 cuda 沿用 vLLM，本类不录 CUDA Graph）
#   位置：backends.py 的 PiecewiseCompileInterpreter.call_module 给每个稠密子图建一个 PiecewiseBackend，替换掉 split_gm 里
#     对应的子模块；之后 split_gm 每次被调用（即每一步 forward），这个子图都经过本类的 __call__：
#       FULL（命令行默认）：split_gm.submod_k = PiecewiseBackend
#       PIECEWISE 类模式  ：split_gm.submod_k = CUDAGraphWrapper(PiecewiseBackend)——录图 / 回放由外层的 cuda_graph.py 负责，
#                           本类只在外层 eager 透传或录图时被调用（回放时不调用）
#   本类手里有两种产物：
#     compiled_graph_for_general_shape  general shape 版本（token 数是符号 s0，任意 token 数都能跑），构造时就已编好
#     concrete_size_entries             compile_sizes 里每个具体 token 数一份静态形状版本，第一次遇到这个 token 数时才编译
#   概念（动态形状 vs compile_sizes）见 decorators.py 文件头【科普】④
#
# 运行时分发（__call__）：
#   第1次调用（就是 warmup 那次 forward，紧接在编译之后）
#     ─→ first_run_finished = True；若本子图是最后一个、且没有待编译的具体尺寸 → 把句柄表写盘、结束编译监控
#     ─→ 用 general 版本算
#   第2次及以后
#     ─→ runtime_shape = 本次 token 数（参数中第 sym_shape_indices[0] 个，运行时是普通 int）
#     ├─ 不在 compile_sizes 里 ─→ general 版本（ATOM 默认 compile_sizes=[]，所以永远走这里）
#     └─ 在 compile_sizes 里   ─→ 第一次遇到：用这次的真实输入编译该尺寸的静态版本（Inductor 会开 max_autotune）并记下；之后直接用
#   例：假设设置了 compile_sizes=[1, 2, 4]：Qwen3-8B 的 37 个稠密子图各有 1 份 general + 3 份静态 = 最多 148 份产物；
#     decode 批里 3 个 token → 3 不在列表里 → general；4 个 token → 第一次现场编译，之后直接用
# 对比nano-vllm：没有对应物——nano-vllm 的 @torch.compile 靠 Dynamo 的 guard 自己按形状重编译、分发

import dataclasses
from typing import Any, Callable

import torch.fx as fx

from atom.utils.backends import VllmBackend
from atom.utils.decorators import end_monitoring_torch_compile
from atom.config import Config


# compile_sizes 里一个具体 token 数的条目
#   runtime_shape  token 数；compiled  是否已为它编译过静态版本；
#   runnable       当前用来跑这个尺寸的可调用对象：初始是 general 版本，编译后换成静态版本
@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore


# PiecewiseBackend：见文件头。一个稠密子图一个实例（Qwen3-8B：37 个），由 backends.py 构造
class PiecewiseBackend:

    def __init__(
        self,
        graph: fx.GraphModule,
        vllm_config: Config,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        vllm_backend: VllmBackend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        # 【译】分段编译的后端。主要负责静态形状的编译，以及按运行时形状分发。
        #
        #   我们会为 general shape 编译 `self.graph` 一次，然后再为 `compilation_config.compile_sizes` 里指定的各个形状分别编译。
        # 参数：graph 本稠密子图（未编译的 fx.GraphModule，具体尺寸编译时还要用它）；piecewise_compile_index 它是第几个稠密子图；
        #   total_piecewise_compiles 稠密子图总数；sym_shape_indices 参数里 SymInt（token 数）的位置；
        #   compiled_graph_for_general_shape 已编译好的 general shape 版本；vllm_backend 所属的 VllmBackend（借用它的 CompilerManager）
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        # 第一个 / 最后一个子图：backends.py 据此设置 CUDAGraphOptions（只在第一个打日志、跑 gc；最后一个输出只留弱引用），
        #   最后一个子图还负责在编译全部结束时写盘（check_for_ending_compilation）
        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        # 整个模型只有一个稠密子图（没有切分点）；ATOM 里没有地方读它（vLLM 遗留）
        self.is_full_graph = total_piecewise_compiles == 1

        # 观察：compile_sizes 只在 Config.__post_init__ 的 level 3 分支里经 init_with_cudagraph_sizes 从 None 变成列表（默认 []）。
        #   --level 2 也会走到这里（decorators.py 只在 level 0/1 时跳过编译），而那时 compile_sizes 仍是 None，
        #   set(None) 会抛 TypeError——据代码推断 --level 2 在原生模式下会在第一次编译时失败（未在 GPU 上验证）
        self.compile_sizes: set[int] = set(self.compilation_config.compile_sizes)

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to compile
        # 【译】需要编译的各个形状的条目
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        # 【译】to_be_compiled_sizes 记录还剩哪些尺寸要编译，它会在编译过程中被更新，所以要复制一份
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()

        # We only keep compilation management inside this class directly.
        # 【译】编译的管理只直接放在这个类里。
        # 解释：每个尺寸的条目先指向 general 版本，等运行时第一次遇到该尺寸才真正编译（惰性编译）
        for shape in self.compile_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                runnable=self.compiled_graph_for_general_shape,
            )

    # 编译是否已全部结束：只有最后一个子图、且所有 compile_sizes 都编过了才算结束，
    #   这时把 CompilerManager 的句柄表写进 vllm_compile_cache.py，并关闭 depyf 转储（decorators.py end_monitoring_torch_compile）
    # 为什么由最后一个子图来做：子图按 0 → N-1 的顺序执行，最后一个子图第一次跑完时，前面所有子图都已编译 / 加载完毕
    # 调用方：本类 __call__（第一次调用时，以及最后一个具体尺寸编译完时）
    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            # 【译】没有具体尺寸要编译了；保存 inductor 图的 hash（句柄表），供下次运行使用
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    # 每一步 forward 调用一次（split_gm 按顺序调到本子图时）；args 是本子图的输入（张量 + token 数 int），返回子图输出
    # 分发规则见文件头
    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        # 本次的 token 数（例如 decode 批 16 条序列、不开投机 → 16）
        # 注意：这里假定每个稠密子图的参数里至少有一个 SymInt，否则 sym_shape_indices 为空会 IndexError
        runtime_shape = args[self.sym_shape_indices[0]]

        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            # 【译】这个形状不需要做任何特殊处理
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        # 这个尺寸第一次出现：用真实输入现场编译（或从缓存加载）它的静态版本，runtime_shape 作为缓存键的一部分
        if not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            # 【译】这里的 args 是真实参数
            # 解释：区别于 general shape 编译时用的 fake 张量；静态版本按这组真实张量的形状特化
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            # 【译】所有需要的形状都编译完了
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        return entry.runnable(*args)
