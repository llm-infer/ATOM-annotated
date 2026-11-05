# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Who is holding a captured cudagraph, for whoever frees what it captured.

A graph replays the addresses it recorded, so releasing the weights or the KV
pool invalidates every graph that captured them -- see
``atom/rollout/memory_manager.py``, which does the releasing. Finding those
graphs is the problem this module exists for.

``runner.graphs`` is only the manual whole-forward store. Under PIECEWISE the
graphs are one per compiled dense piece, each held by a ``CUDAGraphWrapper``
that the compile backend installs with ``module.__dict__[target] = ...``
(``atom/utils/backends.py``) on a submodule of a split ``fx.GraphModule`` that
only Dynamo's cache holds. Both halves of that defeat a walk: there is no path
from the runner to the graph module, and the ``__dict__`` assignment keeps the
wrapper out of ``_modules`` even for someone who has one. So a holder no walk
can reach registers itself here instead, and the releaser asks this module.

Registration is the last resort, not the convention: a store the releaser can
reach -- ``runner.graphs``, ``UBatchWrapper.tbo_graphs``,
``Drafter.draft_graphs`` -- is walked there, where the reader can see what is
being cleared.

Nothing here imports the compile stack, torch or aiter: the release path is
exercised on CPU, and a registry is names and lifetimes only.
"""

# 【译】谁持有录好的 cudagraph——给那个要释放"图所录内容"的人用。
#
#   图回放的是它录下的地址，所以释放权重或 KV pool 会让每张录过它们的图失效——见做释放的 ``atom/rollout/memory_manager.py``。
#   找到这些图，就是这个模块存在的原因。
#
#   ``runner.graphs`` 只是手动整图 forward 的存放处。PIECEWISE 下每个编译好的稠密段一张图，各由一个 ``CUDAGraphWrapper``
#   持有；编译后端用 ``module.__dict__[target] = ...``（``atom/utils/backends.py``）把它装到一个切分后的 ``fx.GraphModule``
#   的子模块上，而这个 GraphModule 只被 Dynamo 的缓存持有。这两点都让遍历失效：从 runner 走不到这个 graph module；
#   而 ``__dict__`` 赋值让 wrapper 不在 ``_modules`` 里，即使拿到了 module 也遍历不到。所以一个任何遍历都够不着的持有者
#   在这里自己登记，释放方来问这个模块。
#
#   登记是最后手段，不是惯例：释放方够得着的存放处——``runner.graphs``、``UBatchWrapper.tbo_graphs``、``Drafter.draft_graphs``——
#   就在那里遍历，读代码的人能看到清掉了什么。
#
#   这里不 import 编译栈、torch 或 aiter：释放路径要在 CPU 上测试，而一个登记表只关心名字和生命周期。
#
# 【导读】graph_holders.py：进程级的"CUDA Graph 持有者"登记表（第 5 批 5e，B 级）
#   前置：CUDA Graph 回放直接读写录制时固定下来的地址（见 cuda_graph.py 文件头前置①③），所以这些地址上的显存
#     （权重、KV pool、图自己的私有内存池）被释放后，旧图再回放就是读写已释放的显存——必须先把图丢掉
#   谁需要它：RL rollout（第 10 批）的 sleep / wake——训练阶段要把推理引擎的显存让出来，memory_manager.py 释放前
#     要清掉所有图（_release_piecewise_graphs 调 release_registered_graphs），wake 时由 memory_manager.py 的
#     _recapture_cudagraphs_if_needed 重新录
#   谁登记（grep 核实）：cuda_graph.py 的 CUDAGraphWrapper.__init__（每个 PIECEWISE 稠密段一个）
#     与 CudagraphCaptureRunner.__init__（deepseek_v4.py 的 v4_attn_runner）
#   协议：持有者实现 release_graphs() -> int（丢掉自己全部的图、返回丢了几张）
#
#     memory_manager._release_piecewise_graphs(runner)
#        ├─ runner._piecewise_captured_tokens.clear()   ← 先让下一步不再分派 PIECEWISE（否则会回放已释放的图，或在服务中途重录）
#        └─ release_registered_graphs()
#              ├─ CUDAGraphWrapper（段 0）.release_graphs()   → 丢掉各 num_tokens 桶的图，并忘掉 _graph_pools
#              ├─ …（每个稠密段一个）
#              └─ v4_attn_runner.release_graphs()             → 丢掉 attention 核心图，并忘掉它的池
#   对比nano-vllm：nano-vllm 只有 self.graphs 一个存放处，exit 时 del 掉即可；没有 RL 休眠，也没有散落在编译缓存里的图

import weakref
from typing import Any

# Weak: a holder lives and dies with the compiled module it hangs off, and a
# registry is the last thing that should be what keeps one alive.
# 【译】弱引用：持有者与它挂靠的编译后模块同生共死，登记表最不应该成为让它活着的那个东西。
# 解释：WeakSet 只保存弱引用，持有者被垃圾回收后自动从集合里消失
_holders: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_graph_holder(holder: Any) -> None:
    """Declare *holder* as owning cudagraphs, for `release_registered_graphs`.

    It must implement ``release_graphs() -> int``, dropping every graph it holds
    and returning how many were dropped, and be hashable by identity -- an
    `eq=True` dataclass is not, and is refused here rather than later.
    Registering twice is harmless.
    """
    # 【译】把 *holder* 声明为 cudagraph 的持有者，供 `release_registered_graphs` 使用。
    #
    #   它必须实现 ``release_graphs() -> int``（丢掉自己持有的每一张图，返回丢掉了几张），并且能按身份哈希——
    #   `eq=True` 的 dataclass 不能，在这里就会被拒绝，而不是等到以后。重复登记无害。
    # 解释：@dataclass 默认 eq=True 时会把 __hash__ 设成 None，WeakSet.add 对不可哈希的对象直接抛 TypeError
    # 调用方：CUDAGraphWrapper.__init__、CudagraphCaptureRunner.__init__（cuda_graph.py）
    _holders.add(holder)


def release_registered_graphs() -> int:
    """Drop every registered holder's graphs; return how many were dropped.

    Process-wide, which is as narrow as a registry gets: the holders are
    reached by nothing else, so there is no runner to ask for its own. The
    other graph stores in this process are already process-global for the same
    reason (`atom/utils/cuda_graph.py`'s graph pools, `deepseek_v4.py`'s
    attention-core capture runner), and a rollout worker holds one runner.
    """
    # 【译】让每个已登记的持有者丢掉它的图；返回一共丢掉了几张。
    #
    #   作用于整个进程，这已经是登记表能做到的最窄范围：这些持有者没有别的途径能够到，所以也没有哪个 runner 可以只问它自己的。
    #   这个进程里其他的图存放处出于同样的原因本来就是进程级全局的（`atom/utils/cuda_graph.py` 的图内存池、`deepseek_v4.py` 的
    #   attention 核心录制 runner），而一个 rollout worker 只持有一个 runner。
    # 调用方：rollout/memory_manager.py 的 _release_piecewise_graphs
    # list(_holders) 先拷一份：遍历过程中若有持有者被回收，WeakSet 会变小，直接遍历可能出错
    return sum(holder.release_graphs() for holder in list(_holders))
