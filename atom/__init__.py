# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 本文件是atom包的入口，决定`from atom import ...`能拿到哪些名字
# 对外只导出三个名字（见__all__）：
#   SamplingParams           请求级采样参数，见本批sampling_params.py
#   LLMEngine                离线推理引擎，见第2批llm_engine.py（延迟导入，原因见下方英文注释）
#   prepare_model_for_sglang SGLang插件准备模型的入口（plugin/sglang/prepare.py里prepare_model的向后兼容别名）
# 调用方：examples/*.py、entrypoints/openai/api_server.py用`from atom import SamplingParams`；
#   model_engine/arg_utils.py用`from atom import LLMEngine`（EngineArgs.create_engine返回它）
# 对比nano-vllm：nanovllm/__init__.py在导入时就直接import LLM和SamplingParams；
#   ATOM的引擎依赖重得多（zmq、ModelRunner、AITER kernel），所以把LLMEngine改成按需导入
# 【分支】SGLang插件，见第10批 plugin/sglang
from atom.plugin.sglang import prepare_model_for_sglang
from atom.sampling_params import SamplingParams

# __all__：`from atom import *`会导出的名字，也被下面的__dir__用来列出公开属性
__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model_for_sglang",
]

# `LLMEngine` is resolved on first attribute access rather than at import time.
# Importing it here pulls the whole engine -- zmq sockets, the model runner,
# AITER kernels -- into *any* `import atom.<anything>`, because Python imports
# a package before its submodule. `import atom.config` paid for a GPU inference
# engine to read a dataclass, which is why the test suite hand-stubbed
# `atom.config` instead of importing it.
#
# The two names above stay eager: both cost only dataclasses, logging and
# typing.
# 【译】`LLMEngine`在第一次访问该属性时才解析，而不是在导入时。
#   在这里直接导入它，会把整个引擎（zmq socket、model runner、AITER kernel）
#   拖进*任何*`import atom.<anything>`，因为Python会先导入包、再导入子模块。
#   过去`import atom.config`为了读一个dataclass，要付出加载整个GPU推理引擎的代价，
#   这也是测试套件以前手工给`atom.config`打桩、而不直接导入它的原因。
#
#   上面两个名字保持立即导入：它们的开销只是dataclass、logging和typing。


# 模块级__getattr__（PEP 562，Python 3.7+）：访问atom.XXX而模块里找不到XXX时才会被调用
#   例：`from atom import SamplingParams`直接拿到上面已导入的名字，不经过这里；
#       `from atom import LLMEngine`（arg_utils.py里就是这么写的）找不到模块属性，于是调用__getattr__("LLMEngine")；
#       而`import atom.config`只执行本文件顶部那两个轻量导入，完全不碰引擎
# 输入：属性名；输出：LLMEngine类。只有LLMEngine走这里，第一次访问时才真正import引擎模块
# 之后Python的import缓存（sys.modules）让重复访问几乎无开销
def __getattr__(name: str):
    if name == "LLMEngine":
        from atom.model_engine.llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 模块级__dir__：dir(atom)和IDE补全只列出__all__，让延迟导入的LLMEngine也能被"看到"
def __dir__() -> list[str]:
    return sorted(__all__)
