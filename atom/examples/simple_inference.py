# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 离线推理示例：读源码的起点（对应nano-vllm的example.py）
# 运行方式（见CLAUDE.md）：
#   python -m atom.examples.simple_inference --model <模型路径> --kv-cache-dtype fp8 -tp 8
#   （CLAUDE.md里的命令不带-tp；-tp默认是1，即单卡，这里的-tp 8表示8卡张量并行）
#
# 本示例里常用的命令行参数（都由EngineArgs.add_cli_args注册，默认值见arg_utils.py的EngineArgs字段）：
#   --model             HF模型目录或hub名，权重、config.json、tokenizer都从这里读
#   -tp/--tensor-parallel-size  张量并行的GPU数（默认1），见下方TP科普
#   --kv-cache-dtype    KV cache的存储类型，bf16（默认）或fp8；fp8省一半显存、能多放请求，精度略降
#                       （注册名是--kv_cache_dtype，FlexibleArgumentParser让连字符写法也能用）
#   --enforce-eager     不捕获CUDA Graph，每步都逐个launch kernel；启动快、便于调试，decode更慢
#   --level             torch.compile级别，0=eager ... 3=piecewise编译+CUDA Graph（默认3）
#   --max-num-seqs      同时在跑的请求数上限（默认512）
#   --gpu-memory-utilization  允许使用的显存比例（默认0.9），扣掉权重和激活后剩下的都分给KV cache
#   --torch-profiler-dir      设了才会真正采集torch profiler trace（见下方start_profile处）
#   --temperature / --max-tokens  本脚本自己的采样参数，见下方
# 用户看到的输出：先是大量引擎日志（加载权重、分配KV cache、捕获CUDA Graph），然后打印套好模板的prompts，
#   最后对每条prompt打印一行"Prompt: ..."和一行"Completion: ..."（生成的文本）
#
# 科普·KV cache：自回归生成时，第t步的attention要用到前t-1个token的K和V；把它们缓存在显存里，
#   每步只需算新token的K/V，不必重算整个前缀。ATOM把KV cache切成固定大小的block（类似nano-vllm），
#   由调度器按需分配给各请求，见model_engine/block_manager.py和docs/scheduling_kv_cache_guide.md
# 科普·TP（张量并行）：把每层的权重矩阵按列/行切到多张GPU上，每张卡算一部分，再用all-reduce汇总；
#   例如-tp 8时每张卡只存1/8的attention头和1/8的MLP权重。每张卡对应一个ModelRunner进程，见docs/distributed_guide.md
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


# 科普·CUDA Graph：decode每步要launch几百个小kernel，CPU发射kernel的开销可能比GPU计算还长。
#   CUDA Graph先把"一整步forward的kernel序列"录制下来，之后每步只需一次replay；代价是输入形状（batch size）
#   必须是录制时的固定值，所以要事先为若干batch size各录一份。录制逻辑见utils/cuda_graph.py和model_runner.py
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
    # 科普·进程结构：当前进程（LLMEngine + CoreManager）用multiprocessing的spawn方式拉起EngineCore进程，
    #   EngineCore再为每张GPU拉起一个ModelRunner进程；进程间用ZMQ socket收发请求和结果
    #   （ZMQ科普见utils/__init__.py的make_zmq_socket处）。必须用spawn而非fork，原因见utils/__init__.py的get_mp_context
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
    #   （准确地说：命令仍会发到各进程，只是ModelRunner.start_profiler发现profiler_dir为None就直接跳过）
    # 科普·profiler trace：torch.profiler记录这段时间内每个CPU算子和GPU kernel的起止时间，
    #   stop时导出为<dir>/<rank名>/<模型名>_ts_<时间戳>.pt.trace.json.gz（每个rank一个文件，见model_runner.py的start_profiler），
    #   可以拖进ui.perfetto.dev或chrome://tracing按时间轴查看哪个kernel最耗时
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
    # 它是"发出去不等回复"的命令，统计打印在EngineCore进程的日志里，而不是这里的print输出
    llm.print_mtp_statistics()
    # 关闭引擎：通知EngineCore和ModelRunner子进程退出，释放GPU资源
    llm.close()


if __name__ == "__main__":
    main()
