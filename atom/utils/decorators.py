# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【导读】decorators.py：@support_torch_compile —— ATOM 使用 torch.compile 的入口，也是"编译子系统"讲解的规范位置
#   位置（一次编译从哪里来、到哪里去）：
#     models/qwen3.py 等：模型的主干类（如 Qwen3Model）被 @support_torch_compile 装饰（本文件）
#       └─ ModelRunner.__init__ → _maybe_warmup → warmup_model：第一次调用 self.model(...)
#            └─ 本文件 _support_torch_compile 里的 __call__：标动态维 → torch.compile(self.forward, backend=VllmBackend)
#                 └─ Dynamo 抓出整个 forward 的 FX 图，交给 backends.py 的 VllmBackend.__call__：
#                      ├─ 算编译缓存目录（下面【科普】⑤）
#                      ├─ split_graph：在 attention 等切分点把图切成一串子图（【科普】③）
#                      └─ PiecewiseCompileInterpreter：每个稠密子图
#                           ├─ compiler_inferface.py：交给 Inductor 编译，或从缓存目录直接加载
#                           ├─ cuda_piecewise_backend.py PiecewiseBackend：运行时按 token 数分发
#                           └─ 仅 PIECEWISE 类 --cudagraph-mode：再包 cuda_graph.py 的 CUDAGraphWrapper，每个子图各录各的 CUDA Graph
#            之后每一步：本文件的 dispatch_to_code 直接执行编译好的字节码，不再经过 Dynamo（【科普】②）
#     切分点与黑盒算子的注册：utils/__init__.py 的 mark_spliting_op（标切分点）、custom_register.py（注册 torch.ops.aiter.xxx）、
#       model_ops/module_dispatch_ops.py（只带 layer_name 字符串、运行时查回 module 的黑盒算子）
#
# 阅读路线（编译子系统共 6 个文件，建议按这个顺序）
#   ① 本文件下面的【科普】①~⑥：先建立整体图景（数字都以 Qwen3-8B 为例）
#   ② 本文件后半：support_torch_compile → _support_torch_compile（装饰器怎么改写模型类）
#      → TorchCompileWrapperWithCustomDispatcher（"编译一次、直接分发"）
#   ③ backends.py：VllmBackend.__call__（缓存目录 → 切图 → 逐子图编译）、split_graph、CompilerManager
#   ④ cuda_piecewise_backend.py：每个子图在运行时怎么分发
#   ⑤ custom_register.py、model_ops/module_dispatch_ops.py：自定义算子
#   ⑥ compiler_inferface.py（C 级）：Inductor 的适配层，平时不用读
#   本文件前半的 mark_trace 一族是 profiler 打点工具（【分支】），和编译主线无关，可以最后看
#   前置：config.py 中 CUDAGraphMode 上方的【科普】①~④（decode 为什么被 CPU 卡住、CUDA Graph 为什么要固定形状和地址、
#     torch.compile 与 guard、分段的一句话版）；下面的①~⑥是它③④的展开
#
# 【科普】ATOM 怎样使用 torch.compile —— 从"一次 forward 就是一串 Python 调用"讲起
#
#   ① Dynamo 抓图、guard 与重编译
#     回顾 config.py【科普】③的要点：torch.compile 分两步。Dynamo 在运行时读 forward 的 Python 字节码，
#     把其中的张量运算记成一张 FX 图（torch.fx.Graph：一个节点列表，每个节点是一次算子调用）；
#     Inductor 再把图里它看得懂的相邻小算子融合，生成更少、更快的 kernel。
#     图是"追踪"出来的：Dynamo 拿这一次调用的真实输入执行 Python，if 只记下实际走到的那一支，for 循环被展开
#     （36 层的 for → 图里有 36 份层的节点）。所以这张图只在某些前提下成立，这些前提叫 guard，例如：
#       forward 里写了 `if hidden_states.shape[0] <= 1024:` → Dynamo 记下 guard "token 数 <= 1024"，图里只有真分支
#     原生 torch.compile 每次调用先检查 guard，不满足就重新抓图（recompile），新图另存一份。
#     抓不了的东西（print / logging、.item() 或按张量的值决定分支、Dynamo 不认识的 C++ 扩展调用）会造成 graph break：
#     图被断成前后两段，中间退回 Python 执行。
#
#     Qwen3-8B（36 层）的 forward 抓出来是这样一张图（节点顺序就是执行顺序）：
#       输入 input_ids [s0]、positions [s0]                        ← s0 是"本步 token 数"的符号（见④）
#       embed_tokens ─→ 第0层 input_layernorm ─→ qkv_proj ─→ 切成 q / k / v ─→ ★attention(第0层)
#         ─→ o_proj ─→ post_attention_layernorm(含残差加) ─→ mlp(gate_up_proj ─→ SiluAndMul ─→ down_proj)
#         ─→ 第1层 input_layernorm ─→ … ─→ ★attention(第35层) ─→ … ─→ 最后的 norm ─→ 输出 hidden_states [s0, 4096]
#       ★ = torch.ops.aiter.unified_attention_with_output_base：一个自定义算子，在图里只是一个黑盒节点（见⑥）
#       lm_head / 采样不在图里：被装饰的是 Qwen3Model，compute_logits 在 Qwen3ForCausalLM 上、图外调用
#       （q_norm / k_norm 是图里的普通节点还是融合进 attention 算子，取决于 ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION，见 qwen3.py）
#
#   ② @support_torch_compile 与"编译一次、直接分发"（--level 2 / 3）
#     推理时每一步都调同一个 forward，每步唯一会变的是 token 数（prefill 几千、decode 几个到几百）。
#     所以 ATOM（沿用 vLLM 的做法）只让 Dynamo 抓一次图：
#       a. 第一次调用前，把 token 维标成动态（torch._dynamo.mark_dynamic），让图对任意 token 数都成立（④）
#       b. Dynamo 抓完图会生成一份"改写后的字节码"（transformed bytecode：取出参数、直接调用编译好的图），
#          本文件的 bytecode_hook 把它存进 compiled_codes
#       c. 之后每次调用，dispatch_to_code(0) 把 forward.__code__ 直接换成这份字节码再执行——
#          不检查 guard，也永远不会 recompile
#     好处：每步省掉 guard 检查（模型有成百上千个参数张量，每个都在 guard 里）；运行中不会因为某个形状突然触发重编译。
#     代价（重要）：抓图那一刻 Python 分支的取值被"冻结"。上面 `shape[0] <= 1024` 的例子里，warmup 用几千 token 的 prefill 抓图，
#       之后 decode 也永远走 > 1024 那一支，没有 guard 来发现这一点。所以凡是"每步要现场判断"的逻辑
#       （按 token 数选单 / 双 stream、prefill 还是 decode、是不是 dummy run），都要挪进黑盒自定义算子（⑥），
#       让它在运行时执行 Python。model_ops/module_dispatch_ops.py 的双 stream MoE 就是这样做的。
#     各 --level 在本文件里的落点（各级别的适用场景表见 config.py CompilationLevel 上方）：
#       level 0 / 1：本装饰器什么都不做（do_not_compile），__call__ 直接调 forward；
#                    level 1 另由 ModelRunner.__init__ 对整个模型套 torch.compile(fullgraph=True, backend="eager")
#       level 2 / 3：走上面 a~c；后端都是 backends.py 的 VllmBackend，两级在这里的代码路径完全相同，
#                    区别只在 Config.__post_init__ 第8步（只有 level 3 才填 cudagraph_mode / compile_sizes 等默认值）；
#                    level 2 的一个问题见 cuda_piecewise_backend.py PiecewiseBackend.__init__ 里的"观察"
#
#   ③ 分段编译：在切分点把整张图切成一串子图
#     Dynamo 抓到的是"整个 forward"一张图。VllmBackend 用 split_graph 在标了 spliting_op 属性的节点处把它切开。
#     切分点是 attention 一类的自定义算子，由 utils/__init__.py 的 mark_spliting_op 标记。Qwen3-8B 的切法：
#
#       子模块        内容                                                      角色        编译？
#       submod_0      embed_tokens + 第0层 attention 之前（input_layernorm、qkv_proj、切 q/k/v）  稠密子图 #0   Inductor 编译
#       submod_1      ★attention(第0层)                                         切分点      不编译，eager 调用
#       submod_2      第0层 attention 之后（o_proj、post_layernorm、MLP）+ 第1层 attention 之前   稠密子图 #1   编译
#       submod_3      ★attention(第1层)                                         切分点
#       …             …
#       submod_71     ★attention(第35层)                                        切分点
#       submod_72     第35层 attention 之后 + 最后的 norm                         稠密子图 #36  编译
#     → 共 73 个子模块 = 36 个切分点 + 37 个稠密子图（N 个切分点 → N+1 个稠密子图，前提是第一个切分点之前、
#       最后一个之后都还有计算；对 decoder 模型总成立：前面有 embedding，后面有 norm）。
#       37 个稠密子图按顺序编号 0~36（graph_index），各自交给 Inductor 编译一次，外面包一层 PiecewiseBackend。
#       哪些算子算切分点：标了 spliting_op 的都算，例如线性注意力（GDN）的 linear_attention_with_output_base、
#       DeepSeek-V4 的 v4_attention_with_output；没标的黑盒算子（如 maybe_dual_stream_forward）留在稠密子图里，当一个普通节点。
#       切分算法（节点怎么分到子图号）见 backends.py split_graph 上方的注释。
#
#     切开之后，每个稠密子图外面包什么，由 --cudagraph-mode 决定（backends.py PiecewiseCompileInterpreter.call_module）：
#       FULL（命令行默认）：只包 PiecewiseBackend，不单独录图；ModelRunner.capture_cudagraph 再把整步 forward
#         （37 个编译子图 + 36 次 attention）录成一张整图，每个 capture size 一张
#       PIECEWISE / FULL_AND_PIECEWISE / AF_PIECEWISE：每个稠密子图再包一个 CUDAGraphWrapper(PIECEWISE)，各录各的小图，
#         attention 在两张小图之间 eager 运行。例：默认 15 个 capture size（1~8、16、32、48、64、128、256、512）
#         × 37 个子图 = 最多 555 张小图（capture_cudagraph 会先去掉 token 预算达不到的档位）
#     为什么 attention 留在图外：attention 每步依赖的元数据都在变（每条序列的长度、block_table、prefill 和 decode 走不同 kernel），
#       Inductor 看不懂 AITER kernel、无从优化，按每步元数据选 kernel 的 Python 逻辑也不能被②冻结；
#       切开后，attention 在 Python 里现场读 forward_context 选 kernel，其余"形状只随 token 数变"的稠密部分照样编译、录图。
#
#   ④ 动态形状与 compile_sizes
#     a 步把 token 维标成符号 s0 后，Inductor 生成对任意 s0 都成立的 kernel（"general shape"/"dynamic shape"版本）。
#     Dynamo 有"0/1 特化"：示例值是 0 或 1 的维度不能标成动态，所以第一次抓图必须用 ≥ 2 个 token
#       （ModelRunner.warmup_model 用最大 token 预算的 prefill 抓图；DSpark 块草稿的行数 = 序列数，所以那里要求至少 2 条序列）。
#     compile_sizes：在 general 版本之外，再为几个具体的 token 数各编译一份静态形状版本（Inductor 会为它做 max_autotune），
#       运行时 token 数恰好等于其中之一时就用它（cuda_piecewise_backend.py）。
#       ATOM 没有能设置它的命令行参数，Config 展开后是 []（config.py CompilationConfig.init_with_cudagraph_sizes），
#       所以默认每个子图只有 general shape 一份产物。
#     和 CUDA Graph 的区别：CUDA Graph 按 capture size 分桶录（每张图形状固定），编译按符号 s0（一份代码适用所有形状）——两层互相独立。
#
#   ⑤ 编译缓存目录：为什么改了代码不清缓存会出事（CLAUDE.md："Clear compile cache before server restart"）
#     Dynamo 抓图每次启动都要重做（日志 "Dynamo bytecode transform time"），Inductor 编译的产物则缓存到磁盘，下次直接加载：
#       ~/.cache/atom/torch_compile_cache/<10位hash>/rank_<GPU号>/<图标签>/        图标签默认 backbone（backends.py set_model_tag）
#         ├─ vllm_compile_cache.py              {(runtime_shape, graph_index, 编译器名): 产物句柄}，例如键 (None, 0, 'inductor_standalone')
#         ├─ computation_graph.py               切图后的 FX 图（可读的 Python，调试用）
#         └─ artifact_shape_None_subgraph_<i>/  第 i 个稠密子图的产物（PyTorch ≥ 2.8 的 standalone 编译器）
#       （PyTorch < 2.8 用 InductorAdaptor，产物改放在 rank_<GPU号>/inductor_cache/ 和 triton_cache/ 下，见 compiler_inferface.py）
#       10位hash = md5([ Config.compute_hash(),                    ← 收了哪些配置因素见 config.py 的 Config.compute_hash
#                       Dynamo 追踪过的所有源文件"路径+内容"的 md5,  ← 由本文件 __call__ 里的 patched_inline_call 收集
#                       编译器 hash（torch 版本、系统信息）])         ← compiler_inferface.py get_inductor_factors
#     命中之后，只按 (None, graph_index, 编译器名) 取产物，不比对图的内容。于是：会改变计算图、却不在这三项里的改动，
#       都会让新代码配上旧产物：
#       - 环境变量：抓图时读到的 ATOM_* 开关被冻结进图（②），而环境变量的 hash 在 backends.py 里是被注释掉的
#         （只有 ATOM_REPLICATE_VOCAB_EMBED 等少数几项被 Config.compute_hash 单独收了进去）
#       - 没被 Dynamo 追踪的代码：模型 __init__、权重布局、配置解析不在 forward 里执行，改了它们会改变图里张量的形状或常量，
#         追踪文件的 hash 却可能不变
#       - 观察：Config.compute_hash 里没有模型路径，也没有 hf_config 的形状字段（层数、hidden_size）；据代码推断，
#         同一个模型文件、相同并行与量化配置的两个不同尺寸 checkpoint 会得到同一个缓存目录。--mark-trace 也不在 hash 里
#     后果：形状对不上时，运行时报 assert_size_stride 一类的错（config.py compute_hash 里的英文注释记录过这类事故）；
#       形状碰巧一致时则静默地按旧逻辑算——这就是"stale cache causes silent failures"。
#       所以改了代码，重启服务前要 rm -rf /root/.cache/atom/*（ANNOTATION_PLAN.md 也提醒：加注释会改变文件 hash，首次运行会重新编译）。
#
#   ⑥ 自定义算子：为什么层要通过"只带一个层名字符串"的算子来调用
#     attention 的 forward 里有很多 Dynamo 追不了、也不该追的东西：调用 AITER 的汇编 / C++ kernel，按 prefill / decode
#     选不同 kernel，读写 KV cache。做法是用 torch.library 把它注册成 torch.ops.aiter.xxx（custom_register.py）：
#       - Dynamo 把它当一个黑盒节点，不进入函数体，只靠 fake 实现知道"输出是什么形状和 dtype"；
#       - 参数里只放张量和一个 layer_name 字符串，函数体里再用 static_forward_context[layer_name] 查回 nn.Module 对象，
#         从它身上取 KV cache、impl、各种 buffer。static_forward_context 是 config.py CompilationConfig 的字段，
#         层在 __init__ 里把自己登记进去（例：paged_attention.py 用层名 prefix，缺省为 MHA_<层号> / MLA_<层号>）。
#     为什么不把 module 或 KV cache 直接当参数：
#       - module 不是张量，进不了算子的 schema；
#       - KV cache 和元数据 buffer 如果是参数，就要进 guard，functionalization 还会给"被原地修改的参数"插防御性拷贝；
#       - 依赖每步元数据（get_forward_context()）的 Python 逻辑留在黑盒里，每步现场执行，不会被②冻结。
#     两类黑盒算子：标了 spliting_op 的（attention，即③的切分点）和没标的（module_dispatch_ops.py 的双 stream MoE、
#       V4 索引器 top-k，它们留在稠密子图里，FULL / PIECEWISE 录图时随所在子图一起被录进去）。
#
#   CLAUDE.md 规则："@support_torch_compile 模型文件里的被追踪函数不要加调试代码"
#     print / logger / .item() / 断点 → graph break（①）→ Dynamo 抓出第二张图、第二次调用同一个 VllmBackend →
#     backends.py 的断言 "VllmBackend can only be called once"。--enforce-eager 只关 CUDA Graph、不关编译，照样会触发。
#     要加调试代码，放在调用处（ModelRunner.run_model、EagleProposer.propose 等）；想在编译后的代码里打 profiler 区间，用下面的 mark_trace。
#
#   对比nano-vllm：nano-vllm 只在几个小函数上写 @torch.compile（RMSNorm、SiluAndMul、RotaryEmbedding、Sampler 的 forward），
#     用默认 Inductor 后端、正常 guard；不编译整个模型、不切图、没有自定义算子（flash-attn 直接调用，decode 整步录成一张 CUDA Graph）。
#     ATOM 编译整个模型的 forward，于是要解决 nano-vllm 没有的问题：哪些部分不能进图（③⑥）、怎么避免每步检查 guard（②）、
#     动辄几分钟的编译怎么缓存、缓存什么时候失效（⑤）。

import inspect
import os
import sys
import time
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from types import CodeType
from typing import (
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
)
from typing import (
    overload as _overload,
)
from unittest.mock import patch

import torch
from torch import nn
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

from atom.config import CompilationConfig, CompilationLevel, Config
from atom.models.utils import IntermediateTensors
from atom.utils.graph_marker import graph_marker

# （下面是被注释掉的旧导入：start_monitoring_torch_compile 现在就定义在本文件里，见下文）
# from atom.utils import start_monitoring_torch_compile

_T = TypeVar("_T", bound=type[nn.Module])
_P = ParamSpec("_P")
_R = TypeVar("_R")

# depyf 调试转储的上下文（只在 level 3 且设置了 debug_dump_path 时才有值，见 start_monitoring_torch_compile）
context_manager = None
# 本次 torch.compile 的开始时间：__call__ 第一次编译前记下，backends.py VllmBackend.__call__ 用它算 Dynamo 抓图耗时
torch_compile_start_time: float = 0.0


# ---------------- 【分支】profiler 打点：mark_trace（--mark-trace；与编译主线无关，可观测性见第10批） ----------------
# 用途：让一个函数在 torch profiler 的 trace 里显示成一段有名字的区间（例如 "rope_cache"、"paged_attention_asm"）。
# 难点在被 torch.compile 编译过的函数：运行时执行的是 Inductor 生成的代码，原函数的 Python 根本不跑，
#   在原函数外面包 torch.profiler.record_function 没有用。所以有两种实现：
#   record_function 版（_decorate_record_function）：eager 调用时用 record_function("<名字>") 包住函数
#   graph_marker 版（_decorate_mark_trace_torch_compile）：抓图时在第一个输入张量上插 graph_marker(x, "<名字>_start")、
#     在第一个输出张量上插 "<名字>_end"。graph_marker（graph_marker.py）是一个声明"会原地修改 x"、实际什么都不做的自定义算子，
#     所以 Inductor 不会把它删掉或挪位置；编译完成后 compiler_inferface.py 改写生成的 Python 文件，
#     把 _start 与 _end 之间的代码包进 record_function("<名字>")
#   "auto"（默认）：Dynamo 抓图期间（_is_torch_compiling 为 True）用 graph_marker 版，否则用 record_function 版
# 开关：--mark-trace → Config.mark_trace → ModelRunner.__init__ 调 set_graph_marker_enabled；关闭时所有包装都直接调原函数
# 调用方：model_ops 下 21 个文件（linear.py、layernorm.py、moe.py、attention_mha.py、attention_mla.py、v4_kernels/* 等）
#   以及 deepseek_v2.py、deepseek_v4.py、kimi_k3/model.py
#
# 决定 profiler 区间的名字，按优先级：装饰器显式给的 prefix → 调用时的关键字参数 prefix → 按签名绑定出的 prefix 参数
#   → 第一个参数（通常是 self）的 get_trace_prefix(...)（如 linear.py）→ self.prefix 属性（层名，如 model.layers.3.mlp）→ 函数名
def _resolve_record_span_name(
    func: Callable,
    args,
    kwargs,
    explicit_prefix: str | None = None,
):
    if explicit_prefix is not None:
        return str(explicit_prefix)

    span_name = func.__name__
    runtime_prefix = kwargs.get("prefix")
    if isinstance(runtime_prefix, str) and runtime_prefix:
        return runtime_prefix

    try:
        base_sig = inspect.signature(inspect.unwrap(func))
        bound = base_sig.bind_partial(*args, **kwargs)
        runtime_prefix = bound.arguments.get("prefix")
    except Exception:
        runtime_prefix = None

    if isinstance(runtime_prefix, str) and runtime_prefix:
        return runtime_prefix

    if args:
        trace_prefix_fn = getattr(args[0], "get_trace_prefix", None)
        if callable(trace_prefix_fn):
            try:
                trace_prefix = trace_prefix_fn(*args[1:], **kwargs)
                if isinstance(trace_prefix, str) and trace_prefix:
                    return trace_prefix
            except Exception:
                pass
        owner_prefix = getattr(args[0], "prefix", None)
        if isinstance(owner_prefix, str) and owner_prefix:
            return owner_prefix
    return span_name


# record_function 版：调用时用 torch.profiler.record_function(区间名) 包住原函数；只对 eager 执行的代码有效
def _decorate_record_function(func: Callable, prefix: str | None = None):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        # Keep this decorator no-op unless mark-trace is enabled.
        # 【译】除非打开了 mark-trace，否则这个装饰器什么都不做。
        from atom.utils.graph_marker import is_graph_marker_enabled

        if not is_graph_marker_enabled():
            return func(*args, **kwargs)

        span_name = _resolve_record_span_name(func, args, kwargs, prefix)
        with torch.profiler.record_function(f"{span_name}"):
            return func(*args, **kwargs)

    return _wrapped


# 在 obj（可以是张量，或嵌套的 tuple / namedtuple / list / dict）里找到"第一个"张量，给它套上 graph_marker
# 返回 (替换后的 obj, 是否已经标过)；只标一个张量，避免一个函数在 trace 里出现多对 _start/_end
def _graph_marker_first_tensor(obj, name: str):
    if torch.is_tensor(obj):
        return graph_marker(obj, name=name), True
    if isinstance(obj, tuple):
        out = []
        marked_any = False
        for v in obj:
            if marked_any:
                out.append(v)
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out.append(vv)
        out_t = tuple(out)
        # namedtuple support
        # 【译】支持 namedtuple
        # 解释：namedtuple 有 _fields 属性，要用原类型重新构造，否则会退化成普通 tuple
        if hasattr(obj, "_fields"):
            return obj.__class__(*out_t), marked_any
        return out_t, marked_any
    if isinstance(obj, list):
        out = []
        marked_any = False
        for v in obj:
            if marked_any:
                out.append(v)
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out.append(vv)
        return out, marked_any
    if isinstance(obj, dict):
        out = {}
        marked_any = False
        for k, v in obj.items():
            if marked_any:
                out[k] = v
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out[k] = vv
        return out, marked_any
    return obj, False


# graph_marker 版：给函数的第一个张量输入插 <前缀>_start、第一个张量输出插 <前缀>_end 两个标记节点
#   前缀默认取 __qualname__ 的第一段（方法取类名，模块级函数就是函数名）；方法调用时优先用 self.prefix（层名），
#   调用时传了 prefix= 关键字则以它为准
def _decorate_mark_trace_torch_compile(func: Callable, prefix: str | None = None):
    # 已经包过一次（带 __mark_trace_wrapped__ 标记）就不再包，避免重复插标记
    if getattr(func, "__mark_trace_wrapped__", False):
        return func

    from atom.utils.graph_marker import is_graph_marker_enabled

    owner_name = func.__qualname__.split(".")[0]
    try:
        unwrapped = inspect.unwrap(func)
        params = list(inspect.signature(unwrapped).parameters.values())
        skip_first_arg = bool(params) and params[0].name in {"self", "cls"}
    except (TypeError, ValueError):
        skip_first_arg = False

    @wraps(func)
    def wrapped(*args, **kwargs):
        # When mark-trace is disabled, bypass all wrapping logic entirely.
        # 【译】mark-trace 关闭时，完全绕过所有包装逻辑。
        # 解释：这句在 Dynamo 抓图时执行，结果被冻结进图（见文件头【科普】②）：关闭时图里不会有 graph_marker 节点
        if not is_graph_marker_enabled():
            return func(*args, **kwargs)

        marker_prefix = str(prefix) if prefix is not None else owner_name
        start_idx = 0
        if skip_first_arg and args:
            if prefix is None:
                marker_prefix = getattr(args[0], "prefix", owner_name)
            else:
                marker_prefix = str(prefix)
            start_idx = 1
        if prefix is None:
            runtime_prefix = kwargs.get("prefix")
            if isinstance(runtime_prefix, str) and runtime_prefix:
                marker_prefix = runtime_prefix

        # Mark only the first tensor across args/kwargs, keeping names stable.
        # 【译】在 args/kwargs 中只标记第一个张量，保持名字稳定。
        args_l = list(args)
        marked = False
        for i in range(start_idx, len(args_l)):
            if marked:
                break
            aa, marked = _graph_marker_first_tensor(args_l[i], f"{marker_prefix}_start")
            args_l[i] = aa
        if not marked:
            for k, v in list(kwargs.items()):
                if marked:
                    break
                vv, marked = _graph_marker_first_tensor(v, f"{marker_prefix}_start")
                kwargs[k] = vv

        y = func(*tuple(args_l), **kwargs)
        yy, _ = _graph_marker_first_tensor(y, f"{marker_prefix}_end")
        return yy

    wrapped.__mark_trace_wrapped__ = True
    return wrapped


# 当前是否处在 Dynamo 抓图过程中（新旧两个 API 都试一次，任何异常都当作"不在编译"）
def _is_torch_compiling() -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and bool(compiler.is_compiling()):
            return True
    except Exception:
        pass
    try:
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


# "auto" 版：两种包装都先建好，每次调用时按"是否在抓图"二选一
#   抓图时走 graph_marker 版（标记进图）；eager 调用（--level 0、或不在被编译区域里）时走 record_function 版
def _decorate_mark_trace_auto(func: Callable, prefix: str | None = None):
    compiled_wrapped = _decorate_mark_trace_torch_compile(func, prefix)
    record_wrapped = _decorate_record_function(func, prefix)

    @wraps(func)
    def wrapped(*args, **kwargs):
        if _is_torch_compiling():
            return compiled_wrapped(*args, **kwargs)
        return record_wrapped(*args, **kwargs)

    wrapped.__mark_trace_wrapped__ = True
    return wrapped


# Overloads preserve the decorated callable's signature for pyright.
# Without them, `@mark_trace` instances were inferred as plain `Callable`,
# and class instances whose `__call__` was wrapped (DualRMSNorm, LinearBase,
# ...) were flagged as "not callable" at the call site.
# 【译】这些重载（overload）为 pyright 保留被装饰函数的签名。没有它们时，`@mark_trace` 的结果会被推断成普通的 `Callable`，
#   `__call__` 被包装过的类实例（DualRMSNorm、LinearBase……）在调用处会被标成"不可调用"。
# 解释：typing.overload 只给类型检查器看，运行时生效的是最后那个不带 @_overload 的 mark_trace 定义。
#   两种写法：@mark_trace（直接装饰）和 @mark_trace(torch_compile=..., prefix=...)（先调用、再装饰）
@_overload
def mark_trace(func: Callable[_P, _R], /) -> Callable[_P, _R]: ...
@_overload
def mark_trace(
    *,
    torch_compile: bool | Literal["auto"] = ...,
    prefix: str | None = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def mark_trace(
    func: Callable | None = None,
    *,
    torch_compile: bool | Literal["auto"] = "auto",
    prefix: str | None = None,
):
    """
    Unified trace decorator.

    - torch_compile=True: original graph_marker-based mark_trace behavior.
    - torch_compile=False: record_function behavior.
    - torch_compile="auto": graph_marker while Dynamo is compiling, otherwise
      record_function.
    """
    # 【译】统一的 trace 装饰器。
    #
    #   - torch_compile=True：原来那种基于 graph_marker 的 mark_trace 行为。
    #   - torch_compile=False：record_function 行为。
    #   - torch_compile="auto"：Dynamo 正在编译时用 graph_marker，否则用 record_function。
    # 调用方用法：attention_mha.py 等 eager 运行的 kernel 包装写 torch_compile=False；
    #   deepseek_v2.py 的 rmsnorm_quant 等写 torch_compile=True；其余用默认 "auto"

    def _decorate(target: Callable):
        if torch_compile == "auto":
            return _decorate_mark_trace_auto(target, prefix)
        if torch_compile:
            return _decorate_mark_trace_torch_compile(target, prefix)
        return _decorate_record_function(target, prefix)

    # Support both @mark_trace and @mark_trace(...)
    # 【译】同时支持 @mark_trace 和 @mark_trace(...) 两种写法
    # 解释：直接 @mark_trace 时 func 就是被装饰的函数；@mark_trace(...) 时 func 为 None，返回真正的装饰器 _decorate
    if func is not None:
        return _decorate(func)
    return _decorate


# ---------------- 编译开始 / 结束的钩子 ----------------
# We remove it from utils/__init__.py to avoid circular import
# 【译】把它从 utils/__init__.py 挪到这里，是为了避免循环导入
# 记下 torch.compile 的开始时间；若是 level 3 且设置了 debug_dump_path，再打开 depyf 的调试转储（把 Dynamo 改写后的字节码
#   反编译成可读的 Python 写到 <debug_dump_path>/rank_0/）。debug_dump_path 默认 ""，ATOM 没有命令行参数能设它
# 调用方：下面 _support_torch_compile 里的 __call__，在第一次编译前调用
def start_monitoring_torch_compile(vllm_config: Config):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = vllm_config.compilation_config
    if (
        compilation_config.level == CompilationLevel.PIECEWISE
        and compilation_config.debug_dump_path
    ):
        import depyf

        # 注意：子目录写死成 rank_0（原来按 rank 区分的写法被注释掉了），多卡时各 rank 写到同一个目录
        path = os.path.join(compilation_config.debug_dump_path, "rank_0")
        # f"rank_{vllm_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


# 编译全部结束时关闭 depyf 的转储上下文（没有打开过则什么都不做）
# 调用方：cuda_piecewise_backend.py PiecewiseBackend.check_for_ending_compilation（最后一个子图编译完时）
def end_monitoring_torch_compile(vllm_config: Config):
    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


# 为一个被装饰的模型实例新建一个编译后端（每个实例一个 VllmBackend，且只能被调用一次，见 backends.py）
# 调用方：TorchCompileWrapperWithCustomDispatcher.__init__；在函数里 import 是为了避免 decorators ↔ backends 的循环导入
def init_backend(config: Config):
    from .backends import VllmBackend

    return VllmBackend(config)


# TorchCompileWrapperWithCustomDispatcher：实现文件头【科普】②"编译一次、直接分发"的混入（mixin）类
#   它不会被直接继承：_support_torch_compile 在运行时把它追加进被装饰类的 __bases__（见下面）
#   它提供的三样东西：
#     compiled_callable  torch.compile(self.forward, backend=VllmBackend) 的结果，第一次调用时经它触发 Dynamo 抓图
#     compiled_codes     Dynamo 改写后的字节码列表（bytecode_hook 收集；正常只有 1 份）
#     dispatch_to_code   临时把 forward 的 __code__ 换成 compiled_codes[i] 执行，绕过 Dynamo 的 guard 检查
#   时序（Qwen3Model 为例）：
#     构造 ─→ __init__：建 VllmBackend、torch.compile 包装 forward、注册字节码钩子
#     第1次调用（warmup）─→ compiled_callable(...) ─→ Dynamo 抓图 ─→ VllmBackend 切图 + 编译 ─→ bytecode_hook 存下新字节码
#     第2次及以后 ─→ with dispatch_to_code(0): self.forward(...) ─→ 直接执行新字节码（里面调用编译好的子图）
#   测试：tests/test_dspark.py 检查 DSpark 草稿模型的内部类是否带上了这个基类
class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    # 【译】一个带自定义分发逻辑的 torch.compile 包装类。
    #   子类应当：
    #   1. 实现 forward 方法
    #   2. 在 __call__ 方法里实现分发逻辑
    #       可以用 `self.compiled_codes` 访问编译出的字节码，
    #       并用 `with self.dispatch_to_code(index):` 分发到编译后的代码。
    #   3. 实现 `__init__` 方法，决定怎样对 forward 方法调用 `torch.compile`。
    # 解释：ATOM 里这三步由 _support_torch_compile 替被装饰的模型类完成：forward 是模型自己的，
    #   __call__ 和 __init__ 由装饰器改写（见下面），所以本类的 __call__ 实际上不会被用到

    def __init__(
        self,
        vllm_config: Config,
        compiled_callable: Callable | None = None,
        compilation_level: int = 0,
    ):
        # 参数：vllm_config 即 ATOM 的 Config（沿用 vLLM 的参数名）；compilation_level 即 --level（0~3）
        self.vllm_config = vllm_config

        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method
            # 【译】默认的编译设置：编译 forward 方法
            # 解释：backend 传的是 VllmBackend 实例（一个可调用对象）：Dynamo 每抓出一张 FX 图就调用一次
            #   backend(graph, example_inputs)，由它决定怎么编译（切图、交给 Inductor、读缓存）。
            #   fullgraph=True 被注释掉了，所以 graph break 不会在这里报错，而是在 VllmBackend 的"只能调用一次"断言处暴露
            options = None
            backend = init_backend(vllm_config)
            compiled_callable = torch.compile(
                self.forward,
                # fullgraph=True,
                backend=backend,
                # dynamic=True,
                options=options,
            )

        self.compiled_callable = compiled_callable
        # forward 原本的代码对象：dispatch_to_code 用完要换回它；bytecode_hook 用它识别"是不是在编译我的 forward"
        # 注意取的是类上的 forward（self.__class__.forward），同一个类的所有实例共用这一个代码对象
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        # 让 Dynamo 每改写完一个函数的字节码就回调 self.bytecode_hook(旧代码, 新代码)（全局钩子，每个实例注册一个）
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        # 【译】读取环境变量，决定是否使用自定义分发器；子类可以用它在自定义分发器和 Dynamo 默认的 guard 机制之间切换。
        # 解释：注释是 vLLM 遗留，ATOM 里并不读环境变量，而是看 --level：>= 2（DYNAMO_ONCE）就用自定义分发器。
        #   本类只在 level 2/3 下才被初始化（level 0/1 时 _support_torch_compile 提前返回），所以在 ATOM 里它总是 True
        self.use_custom_dispatcher: bool = (
            compilation_level >= CompilationLevel.DYNAMO_ONCE
        )

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        # 【译】在 torch.compile 之上，在这里实现分发逻辑。
        #   注意：这个函数可以比 forward 方法多一些参数，用来直接分发到编译后的代码。
        # 解释：被装饰的模型类会被 _support_torch_compile 换上它自己的 __call__，这个默认实现不会被调用
        # print('compiled_callable=====================')
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    # Dynamo 的字节码钩子：Dynamo 每改写完一个函数都会调用它（全局生效，所有被编译的函数都会经过这里）
    # 输入：old_code 原函数的代码对象，new_code Dynamo 改写后的代码对象（里面调用编译好的图）
    # 作用：只挑出"本实例的 forward"那一次，把 new_code 存进 compiled_codes，供 dispatch_to_code 直接执行
    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        # 【译】钩子：保存编译出的字节码，以便直接执行。
        # 第1关：不是我这个类的 forward（比如 Dynamo 改写的是别的函数，或 graph break 产生的续接函数）→ 不管
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        # 【译】代码借自上面这个 depyf 链接
        # 第2关：同一个类可能有多个实例（代码对象相同），要确认被编译的是"我"。钩子的参数里没有 self，
        #   只好沿调用栈往上找到 Dynamo 的 _compile（convert_frame.py）那一帧，它的局部变量 frame 就是被编译的那一帧，
        #   再从那一帧的局部变量里取出 self 来比较
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return
        # 两关都过：这就是本实例 forward 的改写结果，存下来（dispatch_to_code(0) 用的就是第 0 份）
        # print("new_code", new_code)
        self.compiled_codes.append(new_code)
        # 可选：设置了 debug_dump_path 时，把改写后的字节码反编译成 Python 写到 transformed_code.py，方便看 Dynamo 做了什么
        debug_dump_dir = self.vllm_config.compilation_config.debug_dump_path
        if isinstance(debug_dump_dir, str) and debug_dump_dir != "":
            # rank = self.vllm_config.parallel_config.rank
            rank = 0
            decompiled_file = os.path.join(
                debug_dump_dir, f"rank_{rank}", "transformed_code.py"
            )
            if not os.path.exists(decompiled_file):
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    # 【译】对大多数模型，反编译通常都能成功，因为我们保证 Dynamo 做的是整图编译。
                    #   但不能百分之百保证，因为反编译不是一个可逆的过程。
                    import depyf

                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)
                except Exception:
                    pass

        # 安全检查（沿用 vLLM）：改写后的字节码里引用了名字 "update"，被当作"forward 里给 nn.Module 的 buffer 重新赋值"的迹象——
        #   这类副作用 Dynamo 放不进图，只能留在改写后的字节码里用 Python 执行。CUDA Graph 回放时只重放 kernel，
        #   这些 Python 侧的赋值不会再发生，结果会静默出错，所以直接报错。use_cudagraph 默认 True（插件模式传 False）
        if (
            self.vllm_config.compilation_config.use_cudagraph
            and "update" in new_code.co_names
        ):
            import depyf

            src = depyf.decompile(new_code)
            msg = (
                "Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler because it will cause silent errors. Please use eager mode or fix the code. The following code contains clues about which buffer is being modified (please search for the usage of the function `update`):\n"
                + src
            )
            # 【译·报错信息】使用编译器内部的 cudagraph 时，不允许在 forward 中给 nn.Module 的 buffer 赋值或修改它，
            #   因为这会导致静默错误。请使用 eager 模式或修改代码。下面的代码里有线索，能看出哪个 buffer 被修改了
            #   （请搜索函数 `update` 的用法）：
            raise RuntimeError(msg)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """
        # 【译】上下文管理器：分发到编译后的代码。
        #   为什么这样做可行？因为 Dynamo 保证编译出的字节码与原代码有完全相同的参数、cell 变量和自由变量。
        #   因此可以直接替换函数里的代码对象，然后调用它。
        #
        #   更多细节见上面的 dev-discuss 链接。
        # 解释：Python 函数对象 = 代码对象（__code__）+ 全局变量表 + 闭包。Dynamo 把编译好的图以 __compiled_fn_xxx 的名字
        #   放进了 forward 所在模块的全局变量表，改写后的字节码直接去调用它。所以换上 __code__ 后调用 self.forward，
        #   跑的就是"取参数 → 调用编译好的图 → 返回"，完全不经过 Dynamo 的 guard 检查（文件头【科普】②）。
        # 注意：改的是类上的 forward，with 块内同一类的所有实例都会执行这份代码；with 结束换回原代码。
        #   yield 没有放在 try/finally 里，所以块内抛异常时不会换回（之后若有人绕过 __call__ 直接调 forward，跑的也是编译后的代码）
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object


# ======================================================================================================
# @support_torch_compile：类装饰器，让一个 nn.Module 类的 forward 走文件头【科普】②的"编译一次、直接分发"
# 用法（两种都支持）：
#   @support_torch_compile                                   ← 不带参数：从 forward 的类型注解推断动态维
#   class LlamaModel(nn.Module): ...
#   @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})  ← 显式指定哪些参数的哪一维是动态的
#   class Qwen3Model(nn.Module): ...
# 参数 dynamic_arg_dims：{forward 的参数名: 动态维的下标（int 或 list）}；负数从末尾数，
#   例如 positions 的 -1：普通 RoPE 时 positions 是 [T]，就是第0维；M-RoPE 时是 [3, T]，就是第1维——两种都是 token 维
# 装饰的对象：各模型的主干类（Qwen3Model、LlamaModel、DeepseekV2Model……，models/ 下 23 处，另有 plugin/vllm/models/kimi_k25.py 1 处），
#   不是外层的 XxxForCausalLM，所以 lm_head、compute_logits、采样都在编译图之外
# 对比nano-vllm：nano-vllm 没有这个装饰器，只在 RMSNorm 等小函数上直接写 @torch.compile
def support_torch_compile(
    cls: _T | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
) -> Callable[[_T], _T] | _T:
    def cls_decorator_helper(cls: _T) -> _T:
        # helper to pass `dynamic_arg_dims`` to `_support_torch_compile``
        # to avoid too much indentation for `_support_torch_compile``
        # 【译】辅助函数：把 `dynamic_arg_dims` 传给 `_support_torch_compile`，免得 `_support_torch_compile` 缩进太深
        if not hasattr(cls, "forward"):
            raise TypeError("decorated class should have a forward method.")
        # 没有显式给 dynamic_arg_dims 时的推断规则：forward 里注解为 torch.Tensor 或 Optional[torch.Tensor] 的参数，
        #   第0维都当作动态（token 维）。例：LlamaModel.forward(input_ids: torch.Tensor, positions: torch.Tensor, ...)
        #   → 推断结果为 {"input_ids": 0, "positions": 0}
        # 注意：比较的是注解对象本身，文件里若写了 from __future__ import annotations（注解变成字符串）就推断不出来，会报下面的错
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [
                    torch.Tensor,
                    Optional[torch.Tensor],
                ]:
                    inferred_dynamic_arg_dims[k] = 0

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly."
            )

        # 显式给出的参数名必须真的是 forward 的参数，防止拼错名字后静默地什么都没标
        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}"
                )
        return _support_torch_compile(cls, inferred_dynamic_arg_dims)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        # 【译】把 `support_torch_compile` 当作不带参数的装饰器使用
        # 解释：@support_torch_compile 时 cls 就是被装饰的类；@support_torch_compile(...) 时 cls 为 None，返回 cls_decorator_helper
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, int | list[int]],
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    # 【译】一个装饰器：给类加上"编译其 forward 方法"的能力。
    # 它对类做三处原地修改，然后返回同一个类：
    #   1. 把 TorchCompileWrapperWithCustomDispatcher 追加进 cls.__bases__（运行时改基类）
    #   2. 包一层 __init__：先调原 __init__ 建好模型，再按 --level 决定要不要初始化编译包装
    #   3. 换掉 __call__：model(...) 不再走 nn.Module.__call__，而走下面的分发逻辑
    #      注意：nn.Module.__call__ 负责执行 forward hook；换掉之后，挂在这个顶层模块自身上的 hook 不会被调用
    #      （挂在内部各层上的 hook 不受影响）
    if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
        # support decorating multiple times
        # 【译】支持被装饰多次
        # 解释：例如子类继承了已装饰的父类、又被装饰一次；基类已经加过就不再改
        return cls
    # print("_support_torch_compile")
    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    # 【译】注意方法解析顺序（MRO）：确保 super().__init__ 调用的是 TorchCompileWrapperWithCustomDispatcher 以外的那个基类
    # 解释：追加在末尾，MRO 变成 [Qwen3Model, nn.Module, TorchCompileWrapperWithCustomDispatcher, object]，
    #   模型 __init__ 里的 super().__init__() 仍然先到 nn.Module；nn.Module.__init__ 默认不再往后调 super().__init__，
    #   所以混入类的 __init__ 不会被自动调用，要在下面的 __init__ 里显式调用
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher,)

    old_init = cls.__init__

    # 新的 __init__：要求模型类用关键字参数 atom_config 构造（各模型都是 __init__(self, *, atom_config, prefix="")）
    def __init__(self, atom_config: Config, **kwargs):
        old_init(self, atom_config=atom_config, **kwargs)
        self.atom_config = atom_config
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        # 【译】对 CompilationLevel.DYNAMO_AS_IS（level 1），由上层的 model runner 负责编译，这里什么都不用做。
        # 解释：level 1 由 ModelRunner.__init__ 对整个模型套 torch.compile(backend="eager")；level 0 不编译。
        #   这里只看 level，不看 enforce_eager：--enforce-eager 只关 CUDA Graph，编译照常（SGLang / RTP-LLM 插件
        #   是通过传 level=0 的 CompilationConfig 跳过这里的，见 plugin/config.py）
        self.do_not_compile = atom_config.compilation_config.level in [
            CompilationLevel.NO_COMPILATION,
            CompilationLevel.DYNAMO_AS_IS,
        ]
        # print("self.do_not_compile",self.do_not_compile)
        if self.do_not_compile:
            return

        # level 2/3：建 VllmBackend、用 torch.compile 包装 forward、注册字节码钩子（此时还没有抓图，抓图在第一次调用时）
        TorchCompileWrapperWithCustomDispatcher.__init__(
            self,
            vllm_config=atom_config,
            compilation_level=atom_config.compilation_config.level,
        )

    cls.__init__ = __init__

    # 新的 __call__：model(...) 的三条路径
    #   路径1 不编译：level 0/1，或已经处在别人的 Dynamo 抓图中 → 直接调 forward
    #   路径2 第一次调用：标动态维 → 经 compiled_callable 触发 Dynamo 抓图 + VllmBackend 编译（同时收集追踪到的源文件）
    #   路径3 之后每次：dispatch_to_code(0) 直接执行编译后的字节码
    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        # 【译】torch.compiler.is_compiling() 为真表示我们已经处在编译过程之中，
        #   例如 TPU 把编译逻辑放在 model runner 里，那就不需要在内部再编译模型。
        # 解释：ATOM 里对应 level 1：ModelRunner 用 torch.compile 包住整个模型，Dynamo 追踪到这里时 is_compiling() 为真，
        #   直接内联 forward 即可
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # print("self.compiled_codes", self.compiled_codes)
        # the first compilation needs to have dynamic shapes marked
        # 【译】第一次编译前，需要先标好动态形状
        # 例（Qwen3Model，假设 warmup 的 prefill 共 8192 个 token）：input_ids [8192] 标第0维、positions [8192] 标第0维
        #   → 两者长度相同，Dynamo 默认让它们共用一个符号 s0，生成的图对任意 token 数都成立（文件头【科普】④）
        #   sig.bind 把 *args/**kwargs 按 forward 的签名对应到参数名上，apply_defaults 补上没传的默认值
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        # 【译】处理用负数下标指定维度的情况
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        # print(arg.shape)
                        # print(f"torch._dynamo.mark_dynamic({arg, dims})")
                        torch._dynamo.mark_dynamic(arg, dims)
                    else:
                        raise ValueError(
                            "Unsupported dynamic dimensions"
                            f" {dims} for argument {k} with type {type(arg)}."
                        )
            # PP non-first stages receive activations inside an
            # IntermediateTensors container, so the Tensor-typed arg inference
            # above never marks them; mark the token dim of each contained
            # tensor here so inductor keeps the general-shape graph dynamic.
            # 【译】PP 的非首个 stage 收到的激活装在 IntermediateTensors 容器里，上面按 Tensor 类型推断的参数永远标不到它们；
            #   这里把容器里每个张量的 token 维标成动态，让 inductor 的 general shape 图保持动态。
            # 【分支】流水线并行（PP）：上一个 stage 传来的 hidden_states / residual（第9批）
            for arg in bound_args.arguments.values():
                if isinstance(arg, IntermediateTensors):
                    for t in arg.tensors.values():
                        if isinstance(t, torch.Tensor):
                            torch._dynamo.mark_dynamic(t, 0)
            # here, it is the starting point of the `torch.compile` process
            # 【译】这里是 `torch.compile` 过程的起点
            start_monitoring_torch_compile(self.atom_config)
            # print("Start compiling function %s",
            #              self.original_code_object)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        # 【译】如果不使用自定义分发器，可以直接调用编译后的函数，由 torch.compile 负责分发，
        #   代价是 guard 检查和重编译的开销。
        # 解释：ATOM 里能走到这里时 use_custom_dispatcher 总为 True，所以这个分支实际只在"还没有编译过"时进入（路径2）
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            # it seems Dynamo reuse the compilation across instances,
            # while we need to make sure the compiled code is not reused.
            # we need to control all the compilation of the model.
            # 【译】Dynamo 似乎会在多个实例之间复用编译结果，而我们需要确保编译后的代码不被复用：模型的每一次编译都要由我们掌控。
            # 解释：Dynamo 按代码对象缓存编译结果，同一个类若有第二个实例，第一次调用就会直接命中前一个实例的结果，
            #   不会调用本实例自己的 VllmBackend（缓存目录、图标签都是按实例算的）。先把缓存清掉，强制重新抓图
            torch._dynamo.eval_frame.remove_from_cache(self.original_code_object)

            # collect all relevant files traced by Dynamo,
            # so that the compilation cache can trigger re-compilation
            # properly when any of these files change.
            # 【译】收集 Dynamo 追踪过的所有相关文件，这样其中任何一个文件改动时，编译缓存都能正确地触发重新编译。
            # 解释：这些文件的"路径+内容"会被 hash 进缓存目录名（backends.py VllmBackend.__call__ 的因素 2，文件头【科普】⑤）。
            #   只有被 Dynamo 内联追踪的函数所在文件才会进来；自定义算子内部、模型 __init__ 里的代码不在其中

            # 1. the file containing the top-level forward function
            # 【译】1. 顶层 forward 函数所在的文件
            # 例如 Qwen3Model.forward 所在的 models/qwen3.py
            self.vllm_config.compilation_config.traced_files.add(
                self.original_code_object.co_filename
            )

            # 2. every time Dynamo sees a function call, it will inline
            # the function by calling InliningInstructionTranslator.inline_call
            # we hijack this function to know all the functions called
            # during Dynamo tracing, and their corresponding files
            # 【译】2. Dynamo 每遇到一次函数调用，都会调用 InliningInstructionTranslator.inline_call 把该函数内联进来；
            #   我们劫持这个函数，就能知道 Dynamo 追踪期间调用过的所有函数以及它们所在的文件。
            # 例：Qwen3Model.forward 内联 Qwen3DecoderLayer.forward（qwen3.py）、RMSNorm.forward（model_ops/layernorm.py）、
            #   linear.py 的 forward……这些文件路径都会被记进 traced_files
            # unittest.mock.patch.object：只在 with 块内把类属性换成新函数，退出时自动恢复
            inline_call = InliningInstructionTranslator.inline_call

            def patched_inline_call(
                parent, func, args, kwargs, *inline_args, **inline_kwargs
            ):
                code = func.get_code()
                self.vllm_config.compilation_config.traced_files.add(code.co_filename)
                return inline_call(
                    parent, func, args, kwargs, *inline_args, **inline_kwargs
                )

            with patch.object(
                InliningInstructionTranslator, "inline_call", patched_inline_call
            ):
                # print("self.compiled_callable to call torch compile")
                # 这一次调用里依次发生：Dynamo 抓图 → VllmBackend.__call__（算缓存目录、切图、逐子图编译或读缓存）
                #   → bytecode_hook 存下改写后的字节码 → 用真实输入跑一遍编译好的图，返回本次的输出
                output = self.compiled_callable(*args, **kwargs)
            return output

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        # 【译】通常只需捕获模型一次，之后就可以直接分发到编译后的代码，不经过 Dynamo 的 guard 机制。
        # 路径3（每一步 forward 都走这里，包括 CUDA Graph 录制时那次调用）：执行 compiled_codes[0]。
        #   若抓图时发生过 graph break，第二张图在 VllmBackend 里就已经报错了，所以这里只用第 0 份
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__
    return cls
