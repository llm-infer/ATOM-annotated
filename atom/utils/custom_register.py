# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】custom_register.py：direct_register_custom_op —— 把一个 Python 函数注册成 PyTorch 自定义算子 torch.ops.aiter.<名字>
#   为什么要注册成自定义算子、为什么参数里只带 layer_name 字符串：见 utils/decorators.py 文件头【科普】⑥
#   注册之后调用 torch.ops.aiter.<名字>(...)，PyTorch 的分发器按输入张量决定跑哪个实现：
#     真实的 GPU 张量（ROCm 上 HIP 设备同样走 "CUDA" 分发键）─→ op_func，即原来那个 Python 函数
#     fake 张量（Dynamo 抓图、backends.py PiecewiseCompileInterpreter 假跑时）─→ fake_impl，只造出形状 / dtype 正确的空张量
#     Dynamo 看到的永远只是一个节点 torch.ops.aiter.<名字>，不会钻进 op_func 里追踪
#   调用方（约 25 处）：utils/__init__.py 的 mark_spliting_op（注册 attention 等切分点）、model_ops/module_dispatch_ops.py、
#     model_ops 下的 moe.py、topK.py、glm5_next/indexer.py、kimi_k3/attention_residual.py、fused_moe/comm_fused_moe.py，
#     models 下的 deepseek_v2.py、deepseek_v4.py，以及 plugin 下的若干 attention / 稀疏索引器
#   另：graph_marker.py 用的是 PyTorch 自带的 @torch.library.custom_op 装饰器，效果相同
#   对比nano-vllm：nano-vllm 不注册自定义算子（它不编译整个模型，flash-attn 直接调用）

import torch
from torch.library import Library
from typing import Callable, Optional

# 一个 "aiter" 命名空间下的算子库句柄，本文件注册的算子都挂在这里，于是名字是 torch.ops.aiter.<名字>
#   "FRAGMENT"：往一个可能已经存在的命名空间里追加算子，而不是独占地新建它
#   （同一个 aiter 命名空间里还有别处注册的算子，例如 graph_marker.py 的 aiter::graph_marker）
aiter_lib = Library("aiter", "FRAGMENT")


# 注册一个自定义算子（三步：推断 schema → define 声明 → impl 绑定实现，可选再注册 fake 实现）
# 参数：
#   op_name       算子名，例如 "unified_attention_with_output_base" → torch.ops.aiter.unified_attention_with_output_base
#   op_func       真正的实现；schema 由它的类型注解推断，例如 base_attention.py 的
#                 签名 (q: Tensor, q_scale: Tensor | None, k, v, positions, layer_name: str, use_mla: bool, qkv) -> Tensor
#                 推断出 "(Tensor q, Tensor? q_scale, Tensor k, Tensor v, Tensor positions, str layer_name, bool use_mla, Tensor qkv) -> Tensor"
#   mutates_args  会被原地写入的参数名；schema 里标成 Tensor(a!)，编译器据此保留"先写后读"的顺序。
#                 两个方向都会出错：没写却声明了 → functionalization 插入多余的防御性拷贝（module_dispatch_ops.py 的注释）；
#                 写了却没声明 → 编译器以为没变，读到旧数据（deepseek_v2.py sparse_attn_indexer_pcp 的 docstring 记录过这种事故）
#   fake_impl     fake 实现：只根据输入形状造出输出（torch.empty 一类），供抓图和假跑推导形状；没有它，Dynamo 遇到这个算子无法推导输出
#   target_lib    注册到哪个库，默认 aiter_lib；ATOM 里没有调用方改它
#   dispatch_key  实现绑定的分发键，默认 "CUDA"；没有调用方改它，所以这些算子只有 GPU 实现
#   tags          算子标签，例如 torch.Tag.needs_fixed_stride_order：要求 Inductor 按 eager 时的步长传入参数，不为了布局优化改动它
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
):

    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        # 【译】兼容 pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or aiter_lib
    # Guard against double registration. torch's op registry (torch.ops) is a
    # C++ global that outlives Python module state, so re-importing the module
    # that runs this (e.g. after a sys.modules wipe in tests, or a second
    # import path) would call define() twice for the same op and raise. Skip
    # if the op already exists — production single-import flows are unaffected.
    # 【译】防止重复注册。torch 的算子注册表（torch.ops）是一个 C++ 全局对象，生命周期比 Python 模块状态更长，
    #   所以重新导入执行这段代码的模块（例如测试里清空 sys.modules 之后，或经第二条导入路径）会对同一个算子调用两次 define() 并报错。
    #   算子已存在就跳过——生产环境中只导入一次的流程不受影响。
    # 注意：跳过时连 impl 也不会重新绑定，算子仍指向第一次注册的那个函数对象
    op_namespace = getattr(torch.ops, my_lib.ns)
    if hasattr(op_namespace, op_name):
        return
    # define：在命名空间里声明 "名字 + schema"；impl：把 op_func 绑定为 CUDA 分发键上的实现；_register_fake：绑定 fake 实现
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
