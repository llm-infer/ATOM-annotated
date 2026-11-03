# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

# 本文件定义"请求级"的采样参数：每个请求可以不同，随请求一起提交
# 对比nano-vllm：nano-vllm只有temperature、max_tokens、ignore_eos三个字段
#   ATOM多了top_k/top_p、stop_strings、n（一个prompt采样多条）、logprobs
# "引擎级"配置（显存比例、batch上限、并行度等）在config.py的Config里，见本批config.py
# 注意：ATOM没有repetition_penalty / presence_penalty / frequency_penalty（"惩罚已出现过的token"一类参数）；
#   OpenAI协议里的presence_penalty、frequency_penalty只是被接受，不会传进这里（见api_server.py的_build_sampling_params）
#
# 科普：从logits到下一个token（每个解码步、对batch里每条序列各做一次）
#   1. 模型输出logits：词表上每个token一个分数，例如4个候选的logits=[2.0, 1.0, 0.5, -1.0]
#   2. 除以temperature后softmax得到概率；T=1时约为[0.61, 0.22, 0.14, 0.03]
#   3. top_k / top_p把概率低的尾部候选砍掉，剩下的重新归一化
#   4. 从剩下的分布里随机抽一个
#   ATOM的抽样写法：argmax(p / E)，E是逐token的Exp(1)随机噪声，效果等价于"按概率p抽样"
#     （见atom/model_ops/sampler.py；n>1的兄弟序列需要各自独立的E，见Sequence.needs_independent_noise）


# 请求级采样参数，每个请求一份
# 流向：LLMEngine.add_request / API server构造 -> InputOutputProcessor.preprocess拷贝到Sequence的字段上
#   -> ScheduledBatch构造时（scheduler.py）把batch里各序列的temperature/top_k/top_p收集成numpy数组
#   -> ModelRunner.prepare_sample把这些数组拷进GPU缓冲区（顺带钳制温度、判断是否全贪心/是否需要top-k、top-p）
#   -> Sampler在GPU上批量采样
# @dataclass：根据字段声明自动生成__init__、__repr__、__eq__，不用手写样板代码
@dataclass
class SamplingParams:
    # 温度：logits先除以temperature再softmax
    #   越小分布越尖锐，越接近贪心；越大越随机
    #   0表示贪心采样（argmax）：ModelRunner会把温度下限钳到一个很小的eps，采样器把<=eps的行按贪心处理
    #   例：logits=[2, 1]；T=1 -> 概率约[0.73, 0.27]；T=0.5 -> 等价logits=[4, 2] -> 约[0.88, 0.12]；T=0 -> 必选第0个
    #   prepare_sample在整个batch都是T=0时走纯贪心的快路径（all_greedy）
    temperature: float = 1.0
    # 只在概率最高的top_k个token中采样
    #   例：排序后概率[0.5, 0.3, 0.15, 0.05]，top_k=2 -> 只留前两个，归一化为[0.625, 0.375]
    # 【译】（行尾注释）-1表示禁用（保留所有token）
    top_k: int = -1  # -1 means disabled (keep all tokens)
    # 核采样：按概率从高到低累加，只在累计概率达到top_p的最小集合中采样
    #   例：排序后概率[0.5, 0.3, 0.15, 0.05]，累计和[0.5, 0.8, 0.95, 1.0]，top_p=0.9
    #     -> 保留前3个（第3个加入后累计才首次超过0.9），丢掉0.05那个
    #   实现上是"排它累计和 <= top_p"的token都保留（sampler.py里的cumsum - 当前概率 <= p）
    #   和top_k同时设置时两个条件都要满足；batch里只要有一条序列启用，prepare_sample就打开对应路径
    # 【译】（行尾注释）1.0表示禁用（保留所有token）
    top_p: float = 1.0  # 1.0 means disabled (keep all tokens)
    # 最多生成多少个token（不含prompt），达到后序列结束
    #   检查在Scheduler.postprocess里做，结束原因记为"max_tokens"；投机解码一步可能多出几个token，超出部分会被截掉
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
