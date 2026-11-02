# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 离线推理示例：读源码的起点（对应nano-vllm的example.py）
# 运行方式（见CLAUDE.md）：
#   python -m atom.examples.simple_inference --model <模型路径> --kv-cache-dtype fp8 -tp 8
#
# 整体流程：
#   命令行参数 ──EngineArgs.add_cli_args──> argparse解析 ──EngineArgs.from_cli_args──> EngineArgs
#        │
#        ▼
#   EngineArgs.create_engine() ──> 构造Config ──> LLMEngine（启动EngineCore进程、每张GPU一个ModelRunner进程）
#        │
#        ▼
#   用chat template把问题包装成prompt ──> llm.generate(prompts, sampling_params) ──> 每条prompt一个输出dict
#
# 对比nano-vllm：nano-vllm的example.py直接 LLM(path, tensor_parallel_size=1) 构造引擎；
#   ATOM的引擎参数很多（并行、编译、KV、投机解码……），统一由EngineArgs管理命令行和默认值
import argparse

from atom import SamplingParams
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser
from transformers import AutoTokenizer

# FlexibleArgumentParser：argparse.ArgumentParser的替代品，长参数两种写法都接受（--max-tokens 和 --max_tokens）
# RawTextHelpFormatter：--help输出时保留help文本里的换行
parser = FlexibleArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

# Add engine arguments
# 【译】添加引擎参数
# 所有引擎参数（--model、-tp、--kv-cache-dtype、--max-num-seqs等）都在EngineArgs.add_cli_args里定义，见arg_utils.py
EngineArgs.add_cli_args(parser)

# Add example-specific arguments
# 【译】添加本示例专用的参数
# 采样参数不属于引擎配置（每个请求可以不同），所以由示例自己定义，之后用来构造SamplingParams
parser.add_argument(
    "--temperature", type=float, default=0.6, help="temperature for sampling"
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=300,
    help="max sampled tokens per prompt",
)


# 生成不超过max_size的2的幂：max_size=6 -> [1, 2, 4]
# 用作CUDA Graph的捕获尺寸：decode时batch size向上取整到最近的捕获尺寸，回放对应的graph（第5批）
# 这里只有6条prompt，batch最大为6，没必要捕获默认的[1,...,512]那么多graph，可以缩短启动时间
# 注意：6条prompt得到[1, 2, 4]，最大捕获尺寸是4，所以batch为5、6的decode step不在捕获列表里，
#   会以eager模式运行（不用CUDA Graph）；batch为3时向上取整到4，回放batch=4的graph
#   （判断逻辑在forward_context.py的ForwardMode.decide，第5批）
def generate_cuda_graph_sizes(max_size):
    # This is for DP split batch size
    # 【译】这是为DP切分后的batch size准备的
    sizes = []
    power = 1
    while power <= max_size:
        sizes.append(power)
        power *= 2
    return sizes


def main():
    # 测试用的prompt：前4条是短问题；后2条是上千项的算式，prompt有几千个token，
    #   用来触发长prefill和chunked prefill（单步token预算放不下整个prompt时会被切成多个chunk）
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        "如何在一个月内增肌10公斤",
        "+".join([f"{i}-{i+1}" for i in range(1000)]) + "=? 最后结果是什么",
        "+".join([f"{i}+{i+1}" for i in range(1500)]) + "=? 最后结果是什么",
    ]
    args = parser.parse_args()
    # Generate power of 2 sizes for CUDA graph: [1, 2, 4, 8, ...]
    # 【译】为CUDA Graph生成2的幂次的尺寸：[1, 2, 4, 8, ...]
    # 覆盖命令行的--cudagraph-capture-sizes；转成字符串是因为EngineArgs里这个字段是"[1,2,4]"这样的字符串，之后再解析
    args.cudagraph_capture_sizes = str(generate_cuda_graph_sizes(len(prompts)))

    # Create engine from args
    # 【译】根据参数创建引擎
    # from_cli_args：从argparse的结果里挑出EngineArgs的字段，构造EngineArgs
    engine_args = EngineArgs.from_cli_args(args)
    # create_engine：EngineArgs -> Config -> LLMEngine
    #   这一步会启动子进程、加载权重、预热、分配KV cache、捕获CUDA Graph，是最耗时的一步
    llm = engine_args.create_engine()

    # 这里单独加载tokenizer只是为了套chat template；LLMEngine内部还有自己的tokenizer负责分词和反分词
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 所有prompt共用一份采样参数；也可以传一个列表，每个prompt一份
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    # 把用户问题包装成模型的对话格式（例如加上<|im_start|>user ... <|im_end|>）
    # 大多数模型用tokenizer自带的Jinja chat_template；少数模型在模型目录的encoding/encoding_*.py里
    #   提供自定义的消息编码器，load_custom_message_encoder找到它就优先使用，找不到返回None
    custom_encoder = load_custom_message_encoder(args.model)
    prompts = [
        apply_chat_template(tokenizer, custom_encoder, [{"role": "user", "content": p}])
        for p in prompts
    ]
    print("This is prompts:", prompts)
    # （以下是被注释掉的预热代码）
    # print("Warming up...")
    # _ = llm.generate(["warmup"], sampling_params)
    # print("Warm up done")

    print("\n" + "=" * 70)
    print("Starting profiling...")
    print("=" * 70)

    # 开始/停止torch profiler：向所有ModelRunner进程广播命令
    #   只有启动时指定了--torch-profiler-dir才会真正采集，trace文件写到该目录；否则这两个调用什么也不做
    llm.start_profile()
    # 提交所有prompt并阻塞等待全部完成
    # 返回值是一个列表，与prompts一一对应（n=1时），每项是一个dict：
    #   text（生成的文本）、token_ids、logprobs、latency、finish_reason、
    #   num_tokens_input、num_tokens_output、ttft（首token延迟）、tpot（每个输出token的耗时）
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    # 打印MTP投机解码的统计（草稿token接受率等）；没开投机解码时没有内容，见第7批
    llm.print_mtp_statistics()
    # 关闭引擎：通知EngineCore和ModelRunner子进程退出，释放GPU资源
    llm.close()


if __name__ == "__main__":
    main()
