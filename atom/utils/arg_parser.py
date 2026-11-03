# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A drop-in ArgumentParser that accepts both spellings of every long flag.

Historically some ATOM flags were registered kebab-case (``--tensor-parallel-size``)
and some snake_case (``--kv_cache_dtype``), so users had to remember which was
which. Rather than spelling out both forms on every ``add_argument`` call, this
parser adds the missing counterpart automatically.
"""

# 【译】一个可直接替换ArgumentParser的解析器：每个长参数的两种写法都接受。
#   历史上ATOM有的参数注册成kebab-case（``--tensor-parallel-size``），
#   有的注册成snake_case（``--kv_cache_dtype``），用户得记住哪个是哪个。
#   与其在每次``add_argument``调用里把两种写法都写出来，这个解析器会自动补上缺的那一种。
#
# 本文件只有一个类FlexibleArgumentParser，所有命令行入口都用它代替argparse.ArgumentParser：
#   examples/simple_inference.py、examples/multimodal_inference.py、examples/profile_offline.py、
#   entrypoints/openai/api_server.py、entrypoints/atomesh/server.py、benchmarks/benchmark_serving.py
# 引擎参数本身由EngineArgs.add_cli_args(parser)注册（见本批arg_utils.py），这里只负责"两种拼写都认"
# 例：注册了--max-num-seqs，那么--max_num_seqs 8也能用，二者写入同一个dest（max_num_seqs）
# 单元测试：tests/test_arg_utils_spec.py 的 TestFlexibleArgumentParser
# 为什么要这样做：vLLM/SGLang的命令行都是kebab-case，用户经常把它们的启动命令直接搬过来；
#   ATOM旧文档和脚本里又有不少snake_case写法（如CLAUDE.md里的--kv-cache-dtype和代码里注册的--kv_cache_dtype），
#   两边都认，就不会因为一个"-"和"_"之差报 unrecognized arguments
import argparse

__all__ = ["FlexibleArgumentParser"]


# 继承argparse.ArgumentParser，只覆盖add_argument这一个方法；解析行为（parse_args等）完全沿用父类
class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser accepting both ``--kebab-case`` and ``--snake_case`` flags.

    For each long option registered in one spelling, the dash/underscore
    counterpart is auto-registered as an alias for the same ``dest``. Only the
    flag *name* (after the leading ``--``) is transformed, so option *values* are
    never touched — this is safe for JSON-valued flags such as
    ``--online_quant_config '{"use_index_cache": true}'``. Short flags (``-tp``)
    and positionals are left untouched, and ``dest`` still derives from the first
    (original) option string, so existing callers are unaffected.
    """

    # 【译】同时接受``--kebab-case``和``--snake_case``两种写法的ArgumentParser。
    #   对每个以某一种写法注册的长参数，自动把短横线/下划线互换后的写法注册为同一个``dest``的别名。
    #   只变换参数*名*（开头``--``之后的部分），参数*值*不受影响——所以对JSON值的参数也安全，
    #   例如``--online_quant_config '{"use_index_cache": true}'``。短参数（``-tp``）和位置参数保持不变，
    #   ``dest``仍由第一个（原始）选项字符串推导，因此已有调用方不受影响。
    # 覆盖父类的add_argument：输入是调用方给的选项名（可能有多个，如"-tp", "--tensor-parallel-size"），
    #   在交给argparse之前，把每个长选项的另一种拼写追加进去；kwargs原样透传
    def add_argument(self, *names, **kwargs):
        expanded = list(names)
        for opt in names:
            # 短参数（-tp）和位置参数（不以--开头）不做变换
            if not opt.startswith("--"):
                continue
            stem = opt[2:]
            # 两个候选：全换成"-"、全换成"_"；其中一个就是原写法本身，下面的条件会把它跳过
            for alt in (
                "--" + stem.replace("_", "-"),
                "--" + stem.replace("-", "_"),
            ):
                # Skip the option itself, in-call duplicates, and any collision
                # with an already-registered flag (a real argument wins over an
                # auto-generated alias — never raise "conflicting option string").
                # 【译】跳过：选项本身、同一次调用里的重复项，以及与已注册参数冲突的写法
                #   （真实参数优先于自动生成的别名——绝不抛出"conflicting option string"）。
                # _option_string_actions是argparse内部的"选项字符串 -> Action"表，用来查是否已被注册
                if alt not in expanded and alt not in self._option_string_actions:
                    expanded.append(alt)
        # 与BooleanOptionalAction配合：父类会对expanded里每个--开头的名字都生成一个--no-版本，
        #   例：--enable_prefix_caching 最终注册了4个写法（--enable_prefix_caching、--enable-prefix-caching、
        #   --no-enable_prefix_caching、--no-enable-prefix-caching），已实测
        # 注意：只补"全-"和"全_"两种写法；例：注册--a_b-c，会再加--a-b-c和--a_b_c，但用户输入--a-b_c仍不认
        return super().add_argument(*expanded, **kwargs)
