# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

# 本文件定义"请求级"的采样参数：每个请求可以不同，随请求一起提交
# 对比nano-vllm：nano-vllm只有temperature、max_tokens、ignore_eos三个字段
#   ATOM多了top_k/top_p、stop_strings、n（一个prompt采样多条）、logprobs
# "引擎级"配置（显存比例、batch上限、并行度等）在config.py的Config里，见本批config.py


# 请求级采样参数，每个请求一份
# 流向：LLMEngine.add_request / API server构造 -> InputOutputProcessor.preprocess拷贝到Sequence的字段上
#   -> ModelRunner.prepare_sample把一个batch里所有序列的参数拼成张量 -> Sampler在GPU上批量采样
# @dataclass：根据字段声明自动生成__init__、__repr__、__eq__，不用手写样板代码
@dataclass
class SamplingParams:
    # 温度：logits先除以temperature再softmax
    #   越小分布越尖锐，越接近贪心；越大越随机
    #   0表示贪心采样（argmax）：ModelRunner会把温度下限钳到一个很小的eps，采样器把<=eps的行按贪心处理
    temperature: float = 1.0
    # 只在概率最高的top_k个token中采样
    # 【译】（行尾注释）-1表示禁用（保留所有token）
    top_k: int = -1  # -1 means disabled (keep all tokens)
    # 核采样：按概率从高到低累加，只在累计概率达到top_p的最小集合中采样
    # 【译】（行尾注释）1.0表示禁用（保留所有token）
    top_p: float = 1.0  # 1.0 means disabled (keep all tokens)
    # 最多生成多少个token（不含prompt），达到后序列结束
    max_tokens: int = 64
    # 为True时遇到EOS也不停止，一直生成到max_tokens（压测时常用，保证输出长度固定）
    ignore_eos: bool = False
    # 停止字符串：InputOutputProcessor.preprocess把每个字符串分词成token序列，
    #   存入Sequence.stop_token_sequences，调度器在token层面匹配，命中即结束
    stop_strings: Optional[list[str]] = None
    # Number of independently sampled completions to return for a single
    # prompt. n == 1 preserves the historical single-sequence behavior.
    # n > 1 causes the engine to fan out N sibling sequences sharing the
    # same prompt; each uses independent random noise at the sampler so
    # outputs diverge when temperature > 0.
    # 【译】为一个prompt返回多少条独立采样的结果。n == 1 保持原来的单序列行为。
    #   n > 1 时引擎扇出N条共享同一prompt的兄弟序列；每条在采样器使用独立的随机噪声，
    #   所以 temperature > 0 时输出会不同。
    # 解释：扇出由InputOutputProcessor.preprocess_fanout完成，兄弟序列共享prompt，前缀缓存可以复用
    n: int = 1
    # 是否返回每个生成token的logprob；Sequence里只看真假（return_logprobs = bool(logprobs)）
    # 与投机解码不兼容，Sequence构造时会报错
    logprobs: Optional[Union[bool, int]] = None

    # 参数合法性检查
    def __post_init__(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
