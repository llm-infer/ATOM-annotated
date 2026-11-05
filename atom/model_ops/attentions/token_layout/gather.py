# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Some rows of a token axis, out of several tensors in one launch."""

# 【译】一次 launch，从几个张量里各取出 token 轴上的若干行。
#
# 【导读】gather.py —— 按行号从多个张量里抽 token 行（第5批 5b，B级；调用方在第8批）
#   用途：DeepSeek-V4.1 有时只需对一步里的部分 token 继续算（例如 decoder 只跑每个 prompt 的末尾几行、有界重放只重算
#     部分行），要把 hidden 和几个"缝合"张量按同一组行号同时抽出来；一个 Triton kernel 一次做完，代替 N 次 index_select。
#   调用方：models/deepseek_v41/runtime.py（plan.tail_rows；只为参加 DP 集合通信的占位行）、models/deepseek_v41/bounded_replay.py
#     （_gather_rows 延迟 import 本模块，让 bounded_replay 在没有 triton 时也能 import）
#   例：rows = [5, 2]，hidden 形状 [1, 8, 4096]（dim = 1）→ 输出 [1, 2, 4096]，第 0 行 = 原第 5 行，第 1 行 = 原第 2 行

import torch
import triton
import triton.language as tl


# _copy_row：把 source 的第 row 行（WIDTH 个 32 位字）拷到 output 的第 out 行，每次 BLOCK 个字，最后一段用 mask 截断
@triton.jit
def _copy_row(source, output, row, out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    for base in tl.static_range(0, WIDTH, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        word = tl.load(source + row * WIDTH + offs, mask)
        tl.store(output + out * WIDTH + offs, word, mask)


# _gather_rows：grid = (输出行数,)，每个 program 负责一个输出行 out：读 rows[out] 得到源行号，再对每个张量各拷一行；
#   WIDTHS 是 constexpr 元组，tl.static_range 在编译期展开对各张量的循环
@triton.jit
def _gather_rows(rows, sources, outputs, WIDTHS: tl.constexpr, BLOCK: tl.constexpr):
    out = tl.program_id(0)
    row = tl.load(rows + out).to(tl.int64)
    for i in tl.static_range(len(WIDTHS)):
        _copy_row(sources[i], outputs[i], row, out, WIDTHS[i], BLOCK)


def gather_rows(rows, *tensors, dim=0):
    """Each of `tensors` at `rows` of its token axis `dim`, every dim before
    it of size 1. A row is copied as 32-bit words, whatever its dtype."""
    # 【译】tensors 里每个张量在其 token 轴 dim 上取 rows 这些行，dim 之前的每一维大小都必须是 1。一行按 32 位字拷贝，
    #   不管它的 dtype 是什么。
    # 输入：rows = device 上的行号张量 [count]；tensors = 若干个形状 [1, …, T, …] 的张量；输出：同样多个 [1, …, count, …] 新张量
    # 要求：每行字节数是 4 的倍数（_words 把一行重解释成 int32），张量连续
    if not tensors:
        return ()
    count = rows.numel()
    outputs = tuple(
        t.new_empty(*t.shape[:dim], count, *t.shape[dim + 1 :]) for t in tensors
    )
    if count:
        _gather_rows[(count,)](
            rows,
            tuple(_words(t, dim) for t in tensors),
            tuple(_words(t, dim) for t in outputs),
            WIDTHS=tuple(_words(t, dim).shape[1] for t in tensors),
            BLOCK=1024,
        )
    return outputs


# _words：把张量看成 [dim 及之前各维之积（即 token 数）, 每行字节 / 4] 的 int32 视图（不拷贝）
def _words(t, dim):
    return t.view(t.shape[: dim + 1].numel(), -1).view(torch.int32)
