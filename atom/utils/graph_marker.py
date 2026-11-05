# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
# A graph marker for profiling compiled code without copying activations.
# 【译】一个图标记：用来对编译后的代码做 profile，而不复制激活值。
#
# 【导读】graph_marker.py：--mark-trace 用的"区段标记"自定义算子（第 5 批 5e，C 级）
#   用途：抓 profiler trace 时让 kernel 能按"哪一层、哪个算子"分段。eager 代码里直接用 torch.profiler.record_function 就行；
#     但被 torch.compile 编译的部分，Inductor 重新生成一份 Python wrapper 代码、把小算子融合成新 kernel，原来的函数边界看不出来。
#     graph_marker 是一个什么都不做的自定义算子：编译后它作为一次不透明的调用原样留在生成的 wrapper 代码里，正好当边界记号
#   流程：
#     --mark-trace → Config.mark_trace → ModelRunner.__init__ 调 set_graph_marker_enabled(True)
#     utils/decorators.py 的 mark_trace 装饰器：Dynamo 正在编译时，在被装饰函数第一个张量输入上打 "<prefix>_start"、
#       在输出上打 "<prefix>_end"（不在编译时改用 record_function 直接计时）
#     Inductor 生成 wrapper 代码 → compiler_inferface.py 调 graph_marker_instrumentation.py，把一对记号之间的代码包进
#       with record_function("<prefix>"):，backends.py 再强制从缓存重新加载改写后的代码，当前进程立即生效
#   关闭时（默认）graph_marker 直接返回 x，图里什么都不插入，没有开销
#   调用方（grep 核实）：graph_marker 只被 utils/decorators.py 调用；set_graph_marker_enabled 只被 ModelRunner.__init__ 调用；
#     is_graph_marker_enabled 被 decorators.py、compiler_inferface.py（关掉 Inductor 的 size_asserts、编译线程设为 1，
#     并触发后处理）、backends.py（强制重新加载）读取

import torch

# 进程内全局开关；每个 ModelRunner 进程在 __init__ 里按 config.mark_trace 设置
_GRAPH_MARKER_ENABLED: bool = False


def set_graph_marker_enabled(enabled: bool) -> None:
    """Enable/disable graph markers globally (per-process)."""
    # 【译】全局（按进程）开启 / 关闭图标记。
    global _GRAPH_MARKER_ENABLED
    _GRAPH_MARKER_ENABLED = bool(enabled)


def is_graph_marker_enabled() -> bool:
    return _GRAPH_MARKER_ENABLED


# 注册成自定义算子 torch.ops.aiter.graph_marker：编译器看不到里面，只按 schema 对待它
#   mutates_args=("x",)：声明"会原地修改 x"（实际上不改），于是编译器既不会把它当成无用代码删掉，
#   也不会把它和读写 x 的其他操作重新排序——标记就留在原来的位置
@torch.library.custom_op("aiter::graph_marker", mutates_args=("x",))
def _graph_marker(x: torch.Tensor, name: str) -> None:
    # An in-place barrier keeps the marker ordered with operations on x and
    # prevents dead-code elimination. It does not actually change any data.
    # Return nothing: returning x from a custom op would introduce an output
    # alias that functionalization cannot represent with this schema.
    # 【译】一个原地的屏障让标记与 x 上的操作保持先后顺序，并防止被死代码消除。它实际上不改变任何数据。
    #   不返回任何东西：自定义算子若返回 x，会引入一个输出别名，而 functionalization 用这个 schema 表达不了。
    pass


# fake 实现：编译追踪时用，同样什么都不做
@_graph_marker.register_fake
def _graph_marker_fake(x: torch.Tensor, name: str) -> None:
    pass


def graph_marker(x: torch.Tensor, name: str) -> torch.Tensor:
    """Mark a tensor's position in the graph, preserving its identity and layout.

    The compiler sees an in-place barrier with no aliased output. Returning the
    original tensor here keeps its existing views and lets Inductor reuse its
    storage, instead of cloning an opaque custom op's returned alias.
    """
    # 【译】标记一个张量在图中的位置，保持它的身份和内存布局不变。
    #
    #   编译器看到的是一个没有别名输出的原地屏障。这里返回原张量，保留了它已有的 view，也让 Inductor 能复用它的 storage，
    #   而不是去克隆一个不透明自定义算子返回的别名。
    # 例：y = graph_marker(x, "mhc_start") 之后 y is x（tests/test_graph_marker.py 验证了不产生额外拷贝）
    # 调用方：utils/decorators.py 的 _graph_marker_first_tensor（mark_trace 在编译时使用）
    if _GRAPH_MARKER_ENABLED:
        _graph_marker(x, name)
    return x
