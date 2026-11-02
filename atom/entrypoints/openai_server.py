# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backward-compatible entry point for the ATOM OpenAI API server.

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

# 【译】ATOM OpenAI API服务器的向后兼容入口。
#
#   用法：
#       python -m atom.entrypoints.openai_server --model <模型> [选项]
#
# 本文件：在线服务的命令行入口（CLAUDE.md里的启动命令就是它）。只有十几行，做三件事：
#   1. 作为主程序运行时，先调用initialize_metrics()设置Prometheus多进程目录（必须早于导入任何会创建指标的模块）
#   2. 按环境变量USE_ATOMESH_ENTRYPOINTS选择真正的main：
#        默认（不为"1"）           → atom/entrypoints/openai/api_server.py 的 main（本批2c的主角）
#        USE_ATOMESH_ENTRYPOINTS=1 → atom/entrypoints/atomesh/server.py 的 main（Atomesh路由方案）
#   3. 提高打开文件数的上限（set_ulimit），然后调用main()
# "向后兼容"指：实现都在openai/子包里，这个模块路径作为入口保留，旧的启动命令不用改
#
# 启动后的进程结构（详见第2批2a的llm_engine.py、engine_core_mgr.py）：
#   本进程 = API server：uvicorn事件循环 + LLMEngine/CoreManager + 每个DP rank一个输出线程
#     └─ZMQ─> EngineCore进程（每个DP rank一个）─ZMQ─> ModelRunner进程（每张GPU一个）
#
# 对比nano-vllm：nano-vllm没有在线服务，只有example.py里的离线LLM.generate
if __name__ == "__main__":
    # 只在直接运行本文件时执行，并且放在其余import之前：
    #   initialize_metrics()在没有设置PROMETHEUS_MULTIPROC_DIR时创建一个临时目录并写进这个环境变量，
    #   prometheus_client据此进入多进程模式；之后spawn出的EngineCore子进程继承该变量，指标写进同一目录
    from atom.metrics.prometheus import initialize_metrics

    initialize_metrics()

# 以下代码在"被import"和"直接运行"时都会执行：main在导入阶段就已确定
from atom.utils import envs, set_ulimit

# envs.X每次访问都重新读os.environ（见第1批utils/envs.py），所以这个开关要在启动前设置
if envs.USE_ATOMESH_ENTRYPOINTS:
    from atom.entrypoints.atomesh.server import main
else:
    from atom.entrypoints.openai.api_server import main

if __name__ == "__main__":
    # Raise the open-file soft limit before the server (and the engine-core
    # subprocesses it spawns) start, so high connection concurrency does not
    # exhaust file descriptors. Inherited by spawned children.
    # 【译】在服务器（以及它spawn出的engine-core子进程）启动之前提高打开文件数的软上限，
    #   这样高连接并发不会耗尽文件描述符。spawn出的子进程会继承这个上限。
    set_ulimit()
    # 进入真正的main：解析命令行 → 加载tokenizer、创建引擎 → uvicorn.run(...)，阻塞到服务退出
    main()
