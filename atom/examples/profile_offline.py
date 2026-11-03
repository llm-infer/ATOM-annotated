# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 【导读】离线profiling示例：建引擎、预热一次，然后在torch profiler包裹下跑一批请求，导出kernel trace
# 用法：python -m atom.examples.profile_offline --model <model> --bs 8 --input-length 128 --output-length 32 [--random-input]
# 调用方：开发者手动运行；结构与simple_inference.py（本批）相同。后者也调用start_profile / stop_profile，但没给
#   --torch-profiler-dir时什么也不采集；这里的区别是：默认把--torch-profiler-dir设为./profiler_traces、先预热一次、
#   用ignore_eos固定输出长度、把同一个prompt复制--bs份提交，并且不套chat template
# 与主干的关系：LLMEngine.start_profile经CoreManager向每个EngineCore广播"start_profile"工具命令，
#   最终由每个ModelRunner.start_profiler打开torch.profiler，trace按rank写到--torch-profiler-dir/<rank名>/下（见第2批、第5批）
# 科普·profiler trace：torch.profiler在start/stop之间记录每个CPU算子和GPU kernel的起止时间（ModelRunner里开启CPU+CUDA两类活动），
#   stop时每个rank导出一个<模型名>_ts_<时间戳>.pt.trace.json.gz（见model_runner.py的start_profiler）；
#   把它拖进ui.perfetto.dev或chrome://tracing即可按时间轴看kernel耗时、空泡和通信
#   ATOM_PROFILER_RECORD_SHAPES / ATOM_PROFILER_WITH_STACK / ATOM_PROFILER_PROFILE_MEMORY（或ATOM_PROFILER_MORE=1一次全开）
#   可额外记录张量形状、Python调用栈、显存分配，代价是trace更大、导出更慢（定义见utils/envs.py）
# 用户看到的输出：引擎日志 -> "Warming up..." -> "Starting profiling..." -> 各rank的"trace exported to ..."日志
#   -> trace保存目录 ->（非--random-input时）每个请求的生成文本
import argparse
import random
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

parser = FlexibleArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Offline profiling example for Atom LLM Engine",
)

EngineArgs.add_cli_args(parser)

parser.add_argument(
    "--input-length",
    type=int,
    default=128,
    help="Input prompt length in tokens (approximate, used with --random-input)",
)
parser.add_argument(
    "--output-length",
    type=int,
    default=32,
    help="Output generation length in tokens",
)
parser.add_argument(
    "--bs",
    type=int,
    default=1,
    help="Batch size (number of requests to process in parallel)",
)
parser.add_argument(
    "--random-input",
    action="store_true",
    help="Use random repeated words as input. Otherwise use a predefined meaningful text.",
)


# 主流程：解析参数 -> 建引擎 -> 构造prompt -> 预热 -> start_profile -> generate(bs个相同prompt) -> stop_profile -> 打印输出
def main():
    args = parser.parse_args()

    # Set default torch_profiler_dir to current directory if not provided
    # 【译】如果没给torch_profiler_dir，默认用当前目录
    # 必须在from_cli_args之前设置：Config里torch_profiler_dir为None时ModelRunner不建profiler目录，start_profiler就什么也不做
    if args.torch_profiler_dir is None:
        args.torch_profiler_dir = "./profiler_traces"
        print(
            "Warning: --torch-profiler-dir not specified, using default: ./profiler_traces"
        )

    print("\nInitializing LLM engine...")
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.random_input:
        vocab_size = tokenizer.vocab_size
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(args.input_length)
        ]
        prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
        # Re-encode to verify and truncate to exact length
        # 【译】重新编码以校验，并截断到精确长度
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)[
            : args.input_length
        ]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Using randomly generated input with length: {len(token_ids)} tokens")
    else:
        prompt = """Artificial intelligence has revolutionized numerous fields in recent years, transforming the way we live, work, and interact with technology. Machine learning algorithms have become increasingly sophisticated, enabling computers to perform tasks that were once thought to require human intelligence. From natural language processing to computer vision, AI systems are now capable of understanding and generating human language, recognizing objects in images, and even creating original artwork and music. The applications of AI continue to expand rapidly across industries."""
        print(f"Input_prompt: {prompt}")

    # ignore_eos=True：遇到EOS也不停，保证每个请求都正好生成--output-length个token，
    #   这样每次profile的decode步数固定，trace之间可比
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.output_length,
        ignore_eos=True,
    )

    # TODO: remove warm up and skip cuda capture process in engine while profiling.
    # 【译】TODO：profiling时去掉预热，并跳过引擎里的cuda graph捕获过程。
    # 预热：先跑一次，让首次调用才发生的开销（kernel JIT/调优、内存池扩张等）不落进trace里
    print("Warming up...")
    _ = llm.generate(["warmup"], sampling_params)
    print("Warm up done")

    print("\n" + "=" * 70)
    print("Starting profiling...")
    print("=" * 70)

    llm.start_profile()

    # Create batch of prompts based on batch size
    # 【译】按batch size构造一批prompt
    # --bs个完全相同的prompt一起提交：prompt较短时调度器通常会把它们放进同一个batch，trace里看到的就是batch size=bs时的kernel
    prompts = [prompt] * args.bs
    print(f"Processing batch of {args.bs} requests...")

    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    print("\n" + "=" * 70)
    print("Profiling completed!")
    print("Profiler traces have been saved to:", args.torch_profiler_dir)
    print("=" * 70)

    if not args.random_input:
        print("Generated Output:")
        for i, output in enumerate(outputs):
            generated_text = output["text"]
            if args.bs > 1:
                print(f"Output [{i+1}/{args.bs}]: {generated_text}\n")
            else:
                print(f"Output: {generated_text}\n")

    llm.close()


if __name__ == "__main__":
    main()
