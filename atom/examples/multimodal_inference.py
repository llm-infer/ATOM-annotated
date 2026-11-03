# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# 本文件是多模态（图片+文本）离线推理示例：用原生ATOM引擎跑一次"看图回答"
# 用法：python -m atom.examples.multimodal_inference --model <Qwen3.5或Kimi-K3> --image a.jpg [--image b.jpg] --prompt "..."
# 流程：
#   1. EngineArgs.add_cli_args注册全部引擎参数，再追加本脚本自己的--image/--prompt等参数
#   2. HF AutoProcessor负责图片预处理和chat template
#   3. EngineArgs.create_engine()创建LLMEngine（会拉起EngineCore、ModelRunner等子进程，见第2批）
#   4. prepare_multimodal_inputs把"对话+图片"变成(input_ids, multimodal_data)，与在线服务共用同一套代码
#   5. LLMEngine.generate_multimodal提交请求，阻塞到全部完成
# 与主干的关系：结构与simple_inference.py（本批）相同，区别只是输入多了图片、调用generate_multimodal
# 【分支】多模态输入的处理（media加载、各模型的input builder）见第2批 multimodal/*
#
# 【科普】图片占位token与视觉编码器 —— 从"语言模型第一层吃的是词嵌入向量"讲起
#   ① 纯文本时，每个token id先查词嵌入表变成一个hidden_size维的向量，这串向量再进第一层Transformer。
#     图片没有token id，但只要能变成"一串hidden_size维的向量"，就能和词嵌入排进同一条序列
#   ② 所以prompt里先为每张图留一段占位token（同一个特殊token重复N次；Qwen3.5里是image_token_id，默认248056），
#     forward时把这些位置的词嵌入换成视觉编码器的输出，之后的层（attention、KV cache）把它们当普通token处理：
#
#       token：  …user\n  [IMG] [IMG] … [IMG]（共196个）  描述这张图 …
#       嵌入：   词嵌入    视觉向量0 …… 视觉向量195        词嵌入 …
#                          ▲
#       图片448×448 ─切成16×16像素的patch→ 28×28=784个patch ─ViT（双向注意力的Transformer编码器）
#                   ─相邻2×2个合并、投影到hidden_size→ 196个视觉向量
#
#     processor（及模型的输入构建函数）负责前半段：把图片切patch、归一化成张量pixel_values，并把prompt里的占位token
#     展开成对应个数（ATOM要求每个视觉向量对应一个占位token，见multimodal/processing.py）；
#     image_grid_thw记录每张图在(时间t、高h、宽w)三个方向各有多少个patch，用来算占位token数和M-RoPE位置
#     （M-RoPE：图片token的位置用(t, h, w)三个编号而不是一维序号，见model_config/qwen3_5.py文件头【科普】②；
#      patch大小与2×2合并的出处也在那里的①）
#   ③ 在ModelRunner里：视觉编码器是models/qwen3_5_vl.py的Qwen3VisionTransformer；替换嵌入在qwen3_5.py的
#     merge_multimodal_embeddings（把input_ids == image_token_id的位置覆盖成视觉向量）。
#     占位token同样占KV cache：上面这张图在prompt里占196个位置，也占196个token的KV
#   DeepSeek-V4.1走另一条"span路径"（视觉向量缓存在ModelRunner里，按chunked prefill的chunk分批填入），
#     见model_engine/multimodal_runtime.py文件头
# 用户看到的输出：输入token数、pixel_values形状、image_grid_thw，然后是生成文本和延迟/TTFT/TPOT统计
import argparse
import json

from transformers import AutoProcessor

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.multimodal.processing import prepare_multimodal_inputs
from atom.utils.arg_parser import FlexibleArgumentParser

# 命令行解析器；description的英文大意：通用的图片+文本多模态离线推理，使用原生ATOM引擎，
#   已在Qwen3.5和Kimi-K3上验证；依赖模型的HF processor和chat template；
#   媒体归一化和原生输入准备与在线服务共用
# RawTextHelpFormatter：--help输出时保留description里的换行
parser = FlexibleArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Generic image+text multimodal offline inference using the native ATOM engine.\n"
        "Validated with Qwen3.5 and Kimi-K3. The script relies on the model's\n"
        "Hugging Face processor and chat template. Media normalization and native\n"
        "input preparation are shared with online serving."
    ),
)

# 注册全部引擎参数（--model、-tp、--kv-cache-dtype、--max-num-seqs等），定义见本批arg_utils.py
EngineArgs.add_cli_args(parser)

# 以下是本脚本自己的参数。它们不是EngineArgs的字段，EngineArgs.from_cli_args按字段名取值时会忽略它们
# --image：action="append"，可重复给出，每次追加一个路径，最终是list[str]；多张图即多图prompt
parser.add_argument(
    "--image",
    type=str,
    action="append",
    required=True,
    help="Path to an input image file. Repeat for multi-image prompts.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="Describe this image in detail.",
    help="Text prompt to accompany the image",
)
parser.add_argument(
    "--temperature", type=float, default=0.6, help="Temperature for sampling"
)
parser.add_argument(
    "--max-tokens", type=int, default=512, help="Max tokens to generate"
)
# chat template的额外参数（JSON字符串），原样传给processor.apply_chat_template
#   例如关闭Qwen3.5的思考模式：'{"enable_thinking": false}'
parser.add_argument(
    "--chat-template-kwargs",
    type=str,
    default="{}",
    help="JSON kwargs passed to processor.apply_chat_template, e.g. '{\"enable_thinking\": false}'",
)


# 主流程：解析参数 -> 加载processor -> 建引擎 -> 准备多模态输入 -> 生成 -> 打印 -> 关闭引擎
def main():
    args = parser.parse_args()
    chat_template_kwargs = json.loads(args.chat_template_kwargs)

    # Force eager mode and single-batch cudagraph sizes for simplicity
    # 【译】为简单起见，强制eager模式和单batch的cudagraph尺寸
    # 注意：代码实际只把cudagraph_capture_sizes设为"[1]"（只捕获batch=1的CUDA Graph），并没有设置enforce_eager
    #   EngineArgs里这个字段是字符串，_get_engine_kwargs用parse_size_list把它解析成列表
    args.cudagraph_capture_sizes = "[1]"

    # Load processor (handles media preprocessing and chat template)
    # 【译】加载processor（负责媒体预处理和chat template）
    # trust_remote_code=True：允许执行模型仓库自带的processor代码
    # 注意：这里直接用AutoProcessor；在线服务（api_server.py）走的是multimodal/registry.py的get_native_multimodal_processor，
    #   它会为DeepSeek-V4.1换成checkpoint自带的DeepseekV41ImageProcessor，本脚本没有这一步
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # The nth image marker refers to the nth entry in multi_modal_data["image"].
    # 【译】第n个图片标记对应multi_modal_data["image"]中的第n项。
    # 下面构造一条OpenAI风格的user消息：先放len(args.image)个{"type": "image"}占位，再放文本
    conversation = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in args.image),
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Create the engine first: the architecture-specific input builders are
    # selected from its resolved config.
    # 【译】先创建引擎：与具体架构相关的输入构造器（input builder）要根据引擎解析后的配置来选择。
    # llm.io_processor.config就是这份解析后的Config；get_multimodal_input_builder按
    #   hf_config.architectures[0]查表选构造器，查不到时退回Qwen的约定
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    # 输出：input_ids是已展开图片占位token、套好chat template的prompt token列表（list[int]）
    #   multimodal_data是图片张量等数据的字典，下面打印的pixel_values、image_grid_thw就在其中
    input_ids, multimodal_data = prepare_multimodal_inputs(
        llm.io_processor.config,
        processor,
        conversation,
        {"image": args.image},
        chat_template_kwargs,
    )

    print(f"Input token count: {len(input_ids)}")
    print(f"pixel_values shape: {multimodal_data['pixel_values'].shape}")
    print(f"image_grid_thw: {multimodal_data['image_grid_thw']}")

    # 本例所有请求共用一份采样参数（温度默认0.6，最多生成512个token）
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    # Run multimodal generation
    # 【译】执行多模态生成
    # 三个参数都是列表，一个元素对应一个请求；这里只提交一个请求
    # 返回值：按序列id排序的输出列表
    print("\nStarting multimodal inference...")
    outputs = llm.generate_multimodal(
        [input_ids],
        sampling_params,
        [multimodal_data],
    )

    # Print results
    # 【译】打印结果
    # 每个output是InputOutputProcessor.postprocess生成的dict：text、token_ids、latency、ttft、tpot、finish_reason等
    for output in outputs:
        print("\n" + "=" * 70)
        print(f"Generated text:\n{output['text']}")
        print(f"\nInput tokens: {output['num_tokens_input']}")
        print(f"Output tokens: {output['num_tokens_output']}")
        print(f"Latency: {output['latency']:.2f}s")
        print(f"TTFT: {output['ttft']:.3f}s")
        print(f"TPOT: {output['tpot']:.3f}s")
        print(f"Finish reason: {output['finish_reason']}")
        print("=" * 70)

    # 关闭引擎：停止CoreManager管理的子进程，释放GPU资源
    llm.close()


if __name__ == "__main__":
    main()
