# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
ATOM OpenAI-compatible API Server.

FastAPI-based server implementing OpenAI-compatible endpoints for chat
completions and text completions, with reasoning content separation for
thinking models (Kimi-K2, DeepSeek-R1, Qwen3, etc.).

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

# 【译】ATOM的OpenAI兼容API服务器。
#
#   基于FastAPI的服务器，实现OpenAI兼容的chat completions和text completions接口，
#   并为思考模型（Kimi-K2、DeepSeek-R1、Qwen3等）把推理内容单独分离出来。
#
#   用法：
#       python -m atom.entrypoints.openai_server --model <模型> [选项]
#
# =====================================================================================
# 本文件：在线服务的主体（第2批2c的主角）
# =====================================================================================
# 职责：把HTTP请求变成引擎里的Sequence，再把引擎每个step吐出的token变回HTTP响应。
#   它运行在API server进程里，和LLMEngine / CoreManager同一个进程；推理本身在EngineCore进程和ModelRunner进程里
#
# ---- 阅读路线（本文件约4000行，先看这一段）----
# 文件头（到import为止）依次是：接口一览 → 各接口的适用场景 → 首次出现的概念与术语 → 原理（服务层做哪五件事、为什么要流式、
#   为什么断开要立刻abort）→ 图1 流式chat请求的完整路径 → 图2 输出线程如何把结果交给事件循环 → 图3 时间线 → 图4 断开→abort
# 代码按五个"# ====="分节横幅组织（横幅里是英文标题和【译】）：
#   ① Global State：main()启动时写入的全局变量（engine、tokenizer、推理方言、工具调用格式……）、delivered_text、reasoning_channel；
#      其后是/v1/messages专用的anthropic_*辅助函数、Prometheus指标对象、_get_dp_session_affinity_ids
#   ② Request/Response Logging：_log_request_*，以及所有SSE流的最外层包装_client_stream
#   ③ Engine Interface：_build_sampling_params / _coerce_n / 长度校验 → 多模态预处理 → 输出线程上的流式回调与flush
#      → generate_async*（非流式）→ setup_streaming_request*（流式）→ cleanup_* → _race_disconnect（非流式的断开检测）
#   ④ FastAPI Application：lifespan、错误处理器、各接口函数（chat_completions、completions、anthropic_messages、
#      responses_create、/v1/models、/health、/metrics、/debug/*、/kv_transfer_info、/server_info、profiler开关）
#   ⑤ Main Entry Point：main()（解析参数 → tokenizer → 启动期探测chat template → create_engine → 投递器 → uvicorn.run）
# 建议先读主路径：图1 → chat_completions → setup_streaming_request → serving_chat.stream_chat_response
#   → streaming_dispatch.py → cleanup_stream；再读非流式的generate_async → _race_disconnect；最后读main()看全局变量从哪来。
#   anthropic_* / responses_create / *_fanout（n>1）/ *_multimodal 是B级分支，第一遍可以跳过
# 入门【科普】（都从"Transformer把token id序列映射成下一个token的logits"讲起，各讲一次，其他地方只指过去）：
#   chat template与特殊token → chat_encoders.py文件头；推理内容（<think>）→ 本文件chat_completions里；
#   工具调用格式、前缀缓存与cached_tokens → serving_chat.py文件头；增量反分词与偏移量 → streaming_dispatch.py的
#   IncrementalStreamDetokenizer前；图片占位token与视觉编码器 → examples/multimodal_inference.py文件头
# 注释里的记号（注释计划ANNOTATION_PLAN.md的约定）：
#   "第N批"=按主题划分的阅读批次，用来指路：第1批配置与请求对象、第2批引擎主循环与服务层（2b多模态、2c本服务层）、
#   第3批调度器、第4批KV cache、第5批ModelRunner、第7批投机解码、第8批新模型、第9批PD分离与分布式、第10批指标等周边；
#   "A级"=逐行详注的主干，"B级"=只注关键行；"【分支】"=主路径之外的功能分支，指向详细讲它的那一批
#
# 接口一览：
#   POST /v1/chat/completions   chat_completions     主路径（A级详注）
#   POST /v1/completions        completions          主路径（A级详注）
#   POST /v1/messages           anthropic_messages   Anthropic Messages API（格式转换见serving_anthropic.py）
#   POST /v1/responses          responses_create     OpenAI Responses API / Codex（见serving_responses.py）
#   GET  /v1/models、/health、/metrics、/server_info、/kv_transfer_info
#   GET  /debug/mtp_stats、/debug/cache_stats、/debug/gc_census；POST /start_profile、/stop_profile
#
# 各接口的适用场景（背后是同一个引擎、同一套SamplingParams，区别在线上格式、服务端做多少翻译、采样参数的默认值）：
#   接口                    典型调用方                    适用场景 / 注意
#   ──────────────────────  ────────────────────────────  ──────────────────────────────────────────────────────────
#   /v1/chat/completions    OpenAI SDK、聊天前端、Agent    对话/指令模型的默认选择：服务端套chat template，拆出reasoning_content、
#                                                         tool_calls；未传的采样参数用协议默认值（temperature=1.0、max_tokens=8192，
#                                                         不读generation_config）
#   /v1/completions         lm_eval的local-completions     裸prompt续写：base模型、客户端已自己拼好模板、或直接给token id
#                           （GSM8K评测）、PD路由、压测    （prompt传数组或prompt_token_ids）；不做推理内容/工具调用拆分
#   /v1/messages            Claude Code等Anthropic客户端    同一模型按Anthropic格式对外（content块、stop_reason、事件流）；
#                                                         未传的采样参数优先取generation_config的推荐值；max_tokens缺省4096
#   /v1/responses           Codex CLI                      OpenAI Responses格式（含Codex专有工具）；未给max_output_tokens时
#                                                         用serving_responses.DEFAULT_MAX_OUTPUT_TOKENS（4096）；采样参数默认值同上一行
#   /v1/models              客户端探测                     只返回一个ModelCard（id=model_name）
#   /health                 负载均衡、容器探针              只说明事件循环活着，不代表GPU上的模型正常（见/health前的注释）
#   /metrics                Prometheus                     周期刷新的缓存指标文本，抓取本身不访问引擎
#   /server_info            atomesh路由器                  DP宽度等元数据，用于按DP rank展开worker、做缓存感知路由
#   /kv_transfer_info       PD分离部署                     PD角色与握手端口
#   /debug/*、/start_profile、/stop_profile  开发者        投机解码/前缀缓存统计、GC普查、开关profiler
#   注意：没有vLLM / SGLang那种原生的 /generate 接口；想直接喂token id，用 /v1/completions 的 prompt_token_ids
#
# ---- 几个第一次出现的概念（只在这里讲一次）----
# asyncio：单线程的协作式并发。async def定义"协程"，执行到await时把控制权交还给"事件循环"（event loop），
#   事件循环去运行别的就绪协程，等await的东西准备好了再回来接着执行。于是一个线程能同时挂着几千个请求。
#   代价：任何同步的耗时操作（tokenize长prompt、下载图片、遍历大堆做GC统计）都会卡住所有请求，
#   所以本文件把这类工作用 loop.run_in_executor(...) 丢进线程池执行，事件循环只负责调度和收发
# FastAPI：基于Starlette的Web框架。@app.post("/路径") 把一个async函数注册成接口；
#   参数标注为pydantic模型（如ChatCompletionRequest）时，FastAPI自动把JSON请求体解析、校验成该对象，
#   校验失败时直接返回错误响应，不会进入函数体
# uvicorn：ASGI服务器（ASGI是Python异步Web服务器与应用之间的接口规范）。负责监听端口、解析HTTP、
#   在事件循环上驱动FastAPI应用；main()最后的uvicorn.run会一直阻塞。uvloop是libuv实现的更快的事件循环，装了就用
# SSE（服务端推送事件）：一种HTTP流式响应。响应头 Content-Type: text/event-stream，响应体不一次性返回，
#   而是一帧一帧地写，每帧形如 "data: {json}\n\n"；OpenAI的流式协议以 "data: [DONE]\n\n" 结尾。
#   FastAPI的StreamingResponse包装一个异步生成器：生成器每yield一个字符串，就往socket写一次
#   （具名帧"event: ...\ndata: ..."，以及为什么用SSE而不是WebSocket，见sse.py）
# 下文常用的引擎侧术语（一句话定义；带图的入门科普在括号里指向的位置）：
#   prefill / decode：prefill是一次前向处理整段prompt、写入KV cache并采样出第1个token；decode是之后每步只喂上一个token、
#     生成下一个（为什么一个计算受限、一个访存受限：model_engine/engine_core.py文件开头【2】）。
#     KV cache按固定大小的block分给各序列（examples/simple_inference.py的"科普·KV cache"，第4批）
#   投机解码（MTP等）：先由便宜的草稿头猜出后面几个token，主模型一次前向同时验证，猜对几个就一个step输出几个
#     （model_engine/arg_utils.py的"【科普】投机解码"，第7批）
#   Sequence：引擎里"一条生成序列"的对象，带prompt token、采样参数、已生成的token和KV block表（第1批sequence.py）
#   ZMQ：本进程与EngineCore进程之间收发请求、输出和控制命令（utility command）用的消息库（第2批engine_core_mgr.py）
#   DP（数据并行）：起多个EngineCore副本，每个是一个DP rank，各有自己的调度器和KV cache；CoreManager把每个请求分给某个rank，
#     并在本进程里为每个rank起一个输出线程（EngineCoreOutputThread-DP{rank}）
#   PD分离：prefill和decode放在不同节点上跑，prefill节点算完的KV传给decode节点，由PD路由器转发请求
#     （model_engine/arg_utils.py的"【科普】PD分离"，第9批）
#   atomesh：atom/mesh下的Rust路由网关（多worker负载均衡、PD路由），见entrypoints/atomesh/server.py文件头
#
# ---- 原理：一个"OpenAI兼容服务器"到底要做什么 ----
# 引擎只认token id和SamplingParams，客户端说的是OpenAI的JSON协议；服务层就是两者之间的翻译，主要五件事：
#   1. 输入：messages → chat template → prompt字符串 → token id（chat_encoders.py，文件头有【科普】；tokenize在preprocess里做）
#   2. 采样：temperature / top_p / top_k / max_tokens / stop / n → SamplingParams（_build_sampling_params）
#   3. 输出：token id → 文本（非流式一次decode，流式增量反分词），再拆出reasoning_content / tool_calls（serving_chat.py）
#   4. 结束：引擎的leave_reason → finish_reason（stop / length / tool_calls，见protocol.openai_stop_reason*）
#   5. 计量：usage = prompt_tokens（渲染模板后的prompt长度，含模板加的特殊token）/ completion_tokens /
#      prompt_tokens_details.cached_tokens（前缀缓存命中、不用重新prefill的prompt token数，科普见serving_chat.py文件头）
# 为什么要流式：自回归生成的总时延 ≈ TTFT + (输出token数 - 1) × TPOT
#   TTFT（首token时延）= 排队 + prefill；prefill是计算密集的，随prompt长度增长
#   TPOT（每个输出token的时间）≈ 一个decode step的耗时；decode是访存密集的（每步都要读一遍权重和整段KV），随batch和上下文缓慢增长
#   例：TTFT=0.3s、TPOT=20ms、输出500个token → 非流式要等 0.3 + 499×0.02 ≈ 10.3s 才看到第一个字；
#     流式0.3s就开始显示，之后每20ms左右来一段。用户感知的快慢主要看TTFT和ITL（相邻两次输出的间隔，即逐段测得的TPOT；
#     投机解码时一个step可能一次吐出多个token，ITL会"一阵一阵"的）
# 为什么用一个事件循环而不是每请求一个线程：一个请求绝大部分时间在等GPU，HTTP这边只做"等待 + 转发"，
#   一个线程挂几千个协程就够了；真正的批处理（continuous batching：每个step都有序列加入或离开batch，
#   见model_engine/engine_core.py文件开头【1】）在EngineCore进程的Scheduler里
# 为什么客户端断开要立刻abort：KV cache是推理服务最稀缺的资源，它决定了能同时跑多少条序列。
#   例：一条序列已有4000个token的KV，max_tokens还剩6000；客户端走了却不abort，它会继续占着KV block和batch里的一个位置
#   再decode 6000步——这期间等待中的请求因KV不够而进不来，TTFT被拉长。abort后调度器在下一个step结束它并释放block
# ---- 图1：一个流式chat请求的完整路径 ----
#   客户端  POST /v1/chat/completions  {"messages": [...], "stream": true}
#     │  uvicorn解析HTTP → FastAPI用pydantic校验JSON → ChatCompletionRequest
#     ▼
#   chat_completions(request, raw_request)                                   [事件循环线程]
#     │ ① 校验：validate_model、normalize_chat_tools、validate_chat_request（serving_chat.py）
#     │ ② 合并chat template参数：--default-chat-template-kwargs < 请求的chat_template_kwargs
#     │    < response_format / tool_choice / 思考开关
#     │ ③ _coerce_n + _build_sampling_params → SamplingParams
#     │ ④ apply_chat_template（chat_encoders.py）→ prompt字符串
#     │    （多模态 → 线程池里跑_prepare_multimodal_inputs；请求自带prompt_token_ids → 跳过模板）
#     │ ⑤ reasoning_channel(...)：这条回复是否从"思考段"内部开始
#     ▼
#   setup_streaming_request(...)                                             把流式请求交给引擎
#     │ ⑥ 新建StreamOutputCollector（这个请求的"收件箱"）和反分词状态；stream_callback闭包捕获它们
#     │ ⑦ run_in_executor(do_preprocess)：在线程池里 tokenize + 构造Sequence
#     │    （LLMEngine.io_processor.preprocess，见第2批llm_engine.py），stream_callback挂在Sequence上
#     │ ⑧ _validate_sequence_context_length：超过max_model_len或KV池容量 → 400
#     │ ⑨ engine.core_mgr.add_request([seq])：CoreManager登记 seq.id→回调，经ZMQ把Sequence发给EngineCore
#     ▼
#   return StreamingResponse(_client_stream(stream_chat_response(...)))        handler到此返回，响应头先发出
#     ┊
#     ┊  ═══ 其他进程：EngineCore → Scheduler（第3批）→ ModelRunner的forward与采样（第5批）═══
#     ┊  每个step结束，Scheduler.postprocess给有新token（或刚结束）的序列各建一个RequestOutput，
#     ┊  EngineCore把本step的[(seq_id, RequestOutput), ...]打成一条STREAM消息，经ZMQ发回本进程
#     ┊
#   CoreManager的输出线程 EngineCoreOutputThread-DP{rank}                   [引擎输出线程]
#     │ ⑩ 对每个(seq_id, RequestOutput)调用登记的流式回调 → _send_stream_chunk_direct
#     │    → StreamBatchDispatcher.enqueue：只把原始chunk（token id，还没反分词）放进线程本地缓冲
#     │ ⑪ 本step的回调全部跑完 → flush_stream_batch() → 每个事件循环只调一次call_soon_threadsafe
#     ▼
#   StreamBatchDispatcher._deliver(items)                                    [事件循环线程]
#     │ ⑫ 逐个StreamOutputCollector.put_nowait：消费者还没读走的chunk就合并token，然后Event.set()
#     ▼
#   stream_chat_response（serving_chat.py）里的 await collector.get() 醒来
#     │ ⑬ 增量反分词：token id → 新增文本（IncrementalStreamDetokenizer，处理被切开的多字节字符）
#     │ ⑭ 分离推理内容 / 正文 / 工具调用（reasoning.py、tool_parser/），create_chat_chunk → "data: {...}\n\n"
#     │ ⑮ _client_stream：记请求日志、量首个输出的时间、静默看门狗，然后交给uvicorn写socket
#     ▼
#   客户端依次收到：role帧 → 若干content / reasoning_content / tool_calls帧
#                  → 最后在一次发送里收到 finish帧 + usage帧 + data: [DONE]
#
# ---- 图2：引擎输出线程如何把结果交给事件循环 ----
#   asyncio的对象（Queue、Event、协程）不是线程安全的，只能在事件循环自己的线程里操作。
#   别的线程要"通知"事件循环，标准做法是 loop.call_soon_threadsafe(fn, *args)：
#   把fn放进事件循环的待执行队列并唤醒事件循环，fn稍后在事件循环线程里执行。
#
#   引擎输出线程（每个DP rank一个）                      事件循环线程（uvicorn，只有一个）
#   ┌──────────────────────────────────────┐            ┌──────────────────────────────────────────┐
#   │ recv STREAM: [(seq1,o1),(seq2,o2),…] │            │                                          │
#   │ for 每个输出: callback(o)            │            │                                          │
#   │   └ dispatcher.enqueue(...)          │            │                                          │
#   │     → threading.local缓冲 [c1,c2,…]  │            │                                          │
#   │ flush_stream_batch():                │            │                                          │
#   │   按目标事件循环分组                 │ 每step每个 │ _deliver(items):                         │
#   │   loop.call_soon_threadsafe(─────────┼─循环一次──▶│   for c: collector.put_nowait(c)         │
#   │       _deliver, items)               │            │     （上一个chunk还没被读走 → 合并token）│
#   └──────────────────────────────────────┘            │     Event.set()                          │
#                                                       │ stream_chat_response:                    │
#                                                       │   await collector.get() → 反分词 → yield │
#                                                       └──────────────────────────────────────────┘
#   为什么攒一个step再投递：并发1000个流时一个step就有1000个回调，逐个call_soon_threadsafe就是1000次跨线程唤醒、
#     1000个待执行回调；攒批后每个step只有一次。
#   为什么反分词放在事件循环的get()里而不是输出线程：见streaming_dispatch.py中flush的注释——多个输出线程各自反分词
#     会争抢GIL；而且消费者落后时，多个未读chunk先合并再反分词，只解码一次
#   非流式请求（generate_async）不经过dispatcher：每次回调直接
#     loop.call_soon_threadsafe(token_queue.put_nowait, {...})，handler里 await token_queue.get() 一直收集到结束
#
# ---- 图3：一个流式chat请求的具体时间线（数字为示意，取一台空闲服务器上8B级模型的典型量级）----
#   请求：POST /v1/chat/completions
#     {"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "用一句话介绍北京"}],
#      "stream": true, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}}
#   时间      在哪里                         发生了什么
#   ───────   ─────────────────────────────  ──────────────────────────────────────────────────────────────────
#   t=0ms     事件循环（chat_completions）   pydantic校验 → 合并模板参数 → SamplingParams(max_tokens=64) → 渲染prompt
#   t≈1ms     线程池（do_preprocess）        tokenize：假设渲染后的prompt是18个token → Sequence(num_prompt_tokens=18)
#             事件循环                       add_request经ZMQ发给EngineCore；返回StreamingResponse，先发出
#                                            "HTTP/1.1 200" + "content-type: text/event-stream"（此时还没有任何帧）
#   t≈2~30ms  EngineCore / ModelRunner       Scheduler在下一个step接纳它，prefill 18个token，采样出第1个token
#   t≈30ms    输出线程 → 事件循环            STREAM消息 → 回调 → enqueue → flush → _deliver → collector.put_nowait
#             stream_chat_response           get()反分词得到"北京" → 先yield role帧，再yield content帧：
#               data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,
#                      "delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}
#               data: {…"choices":[{"index":0,"delta":{"content":"北京"},"finish_reason":null,…}]}
#             _client_stream                 第一帧带生成内容 → timing.first_output：TTFT ≈ 30ms（含排队 + prefill）
#   t≈50ms    同上，下一个decode step        data: {…"delta":{"content":"是"}…}         ITL ≈ 20ms（≈一个decode step）
#   …         每≈20ms一帧                    中国的首都，…（事件循环忙时几个step会合并成一帧，见streaming_dispatch.py）
#   t≈470ms   最后一个step                   第23个token是EOS → chunk带finished=True、finish_reason="eos"
#             stream_chat_response           一次发送里拼好三帧：
#               data: {…"choices":[{"index":0,"delta":{},"finish_reason":"stop",…}]}      ← "eos"翻译成"stop"
#               data: {…"choices":[],"usage":{"prompt_tokens":18,"completion_tokens":23,"total_tokens":41,
#                      "prompt_tokens_details":{"cached_tokens":0}}}
#               data: [DONE]
#             finally                        cleanup_stream(seq_id, aborted=False)（不发abort）→ cleanup_request
#   总时延 ≈ TTFT + (23 - 1) × ITL ≈ 30 + 22 × 20 ≈ 470ms；客户端从30ms起就开始逐段显示
#   completion_tokens是各chunk的token_ids长度之和（engine计数，与"显示出来的字数"无关）；
#   同一段对话第二轮再发，前缀缓存命中整块的部分会体现在cached_tokens里
#
# ---- 图4：客户端断开 → abort（流式与非流式两条路，最后汇合到同一个abort_request）----
#                              客户端关闭连接（Ctrl+C、超时、浏览器关页）
#                                        │ uvicorn向应用投递ASGI消息 {"type": "http.disconnect"}
#                  ┌─────────────────────┴──────────────────────┐
#          流式：StreamingResponse                      非流式：handler还在await结果
#                  │                                            │
#   Starlette监听到断开，取消迭代生成器的任务         _race_disconnect：disconnect_task先完成
#                  │                                            │ handler_task.cancel()
#   stream_chat_response停在哪里？                              ▼
#     await collector.get() → CancelledError          generate_async里 await token_queue.get()
#     某个yield上 → 之后被aclose → GeneratorExit        → CancelledError
#                  │                                            │
#   finally：aborted仍为True（finished chunk没到）     finally：_finished_ok为False
#     → cleanup_stream(seq_id, aborted=True)             → core_mgr.abort_request(seq.id)
#         ├ core_mgr.abort_request(seq_id)               → io_processor.requests.pop(seq.id)
#         └ io_processor.requests.pop(seq_id)          _race_disconnect抛_ClientDisconnected
#     → cleanup_request(request_id)                    → 接口返回499（客户端已经收不到）
#                  └─────────────────────┬──────────────────────┘
#                                        ▼
#   CoreManager.abort_request：先_release_seq_load（DP负载记账），再broadcast_utility_command("abort_request")给每个DP rank
#                                        ▼  ZMQ控制通道（发出即不管）
#   各EngineCore → Scheduler.abort_request(seq_id)：
#     在waiting里 → 移出队列、status=ABORTED、_reject_aborted_waiting
#     在running里 → status=ABORTED，下一个step按正常停止路径结束它，释放KV block
#     都找不到（已经结束、或不在本rank）→ 返回False，什么也不做
#   好处：被放弃的请求最多再多跑一个step就让出KV和batch位置；代价：每次断开要向每个DP rank各发一条控制消息
#     （所以正常结束时aborted=False，不发这条注定无效的广播）
#
# 对比nano-vllm：nano-vllm没有服务层，LLM.generate在同一个线程里循环step直到全部完成；
#   ATOM的服务层要解决的是"引擎在别的进程/线程里异步产出token，而HTTP这边有几千个连接同时在等"——
#   回调 + 跨线程投递 + 每请求一个收件箱 + SSE，以及客户端断开时及时abort，避免GPU为没人读的输出空转

# 直接运行本文件时（python -m atom.entrypoints.openai.api_server），先设置PROMETHEUS_MULTIPROC_DIR，
#   再导入其余模块：下面的create_metrics_exporter()在导入期就会创建指标；该变量已设置、
#   但prometheus_client在设置之前就被导入了的话，metrics_setup.py会抛RuntimeError
#   （通过openai_server.py启动时，那里已经先做了同样的事）
if __name__ == "__main__":
    from atom.metrics.prometheus import initialize_metrics

    initialize_metrics()

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

# 第三方库：uvicorn（ASGI服务器）、FastAPI（Web框架）、transformers（tokenizer与多模态processor）
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoProcessor, AutoTokenizer

# ATOM引擎侧：SamplingParams（第1批sampling_params.py）、EngineArgs（第1批arg_utils.py）、
#   RequestOutput（第1批request.py：引擎每个step给每条序列的一份增量输出）、new_token_ids（第1批sequence.py：array("i")）
#   多模态预处理（第2批2b multimodal/*）、gc_utils（垃圾回收调优，第2批utils/gc_utils.py）
from atom import SamplingParams
from atom.entrypoints.chat_utils import has_multimodal_content, parse_chat_messages
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.llm_engine import _load_tokenizer
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import new_token_ids
from atom.multimodal.processing import prepare_multimodal_inputs
from atom.multimodal.registry import get_native_multimodal_processor
from atom.utils import envs
from atom.utils.arg_parser import FlexibleArgumentParser
from atom.utils.gc_utils import (
    arm_reclaim_watch,
    freeze_gc_heap,
    gc_census,
    maybe_attach_gc_debug_callback,
    reclaim_watch,
    tune_gc,
)

# 本包内的模块（分工见openai/__init__.py的文件头）
from .chat_encoders import (
    apply_chat_template,
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from .metrics_setup import create_metrics_exporter
from .protocol import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    validate_max_tokens,
)
from .reasoning import (
    ReasoningChannel,
    prompt_starts_in_reasoning,
    prompt_tokens_start_in_reasoning,
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from .reasoning_dialects import resolve_dialect
from .request_timing import (
    RequestTimingMiddleware,
    get_stream_timing,
    has_generated_output,
    record_nonstream_first_token,
)
from .serving_anthropic import (
    AnthropicBlocks,
    AnthropicMessagesRequest,
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    build_anthropic_response,
    completes_a_tool_call,
    read_whole_blocks,
    stream_failure_frames,
    stream_message_delta,
    stream_message_start,
    stream_message_stop,
    tool_event_frames,
)
from .serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    normalize_chat_tools,
    resolve_thinking,
    stream_chat_response,
    stream_chat_response_fanout,
    validate_chat_request,
    validate_tool_list,
)
from .serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    stream_completion_response,
    stream_completion_response_fanout,
)
from .serving_responses import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ResponsesRequest,
    ResponseStreamEmitter,
    build_responses_response,
    normalize_codex_request,
    responses_to_openai_messages,
    responses_tools_to_openai,
    unsupported_responses_parameter,
)
from .serving_responses import stream_failure_frames as responses_stream_failure_frames
from .sse import event_frame, iter_sse_data
from .streaming_dispatch import (
    SYNTHETIC_TOKEN_TEXT,
    FrameWait,
    StreamBatchDispatcher,
    StreamOutputCollector,
    enable_delta_reuse,
)
from .tool_parser import (
    ToolCallStreamParser,
    flatten_tool_events,
)
from .tool_parser.registry import (
    TOOL_CALL_PARSER_HELP,
    forbids_tool_calls,
    resolve_tool_call_parser,
)

# Configure logging
# 【译】配置日志
# 所有ATOM模块共用名为"atom"的logger
logger = logging.getLogger("atom")

# Constants
# 【译】常量
# --host默认0.0.0.0（监听所有网卡），--server-port默认8000（--port另有用途：引擎内部通信，见main()）
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


# ============================================================================
# Global State
# ============================================================================
# 【译】全局状态
# 为什么用模块级全局变量：FastAPI的接口函数定义在模块顶层，main()在启动时（uvicorn.run之前）把引擎、tokenizer等
#   写进这些全局变量（main里用global声明），之后各接口函数直接读取。
#   整个服务只有一个进程、一个事件循环，全局状态在启动后基本只读，不需要加锁

# LLMEngine实例（第2批llm_engine.py），main()中由engine_args.create_engine创建。本文件用到它的：
#   engine.io_processor：preprocess / preprocess_fanout（tokenize + 构造Sequence），以及requests字典（在途序列）
#   engine.core_mgr：CoreManager，add_request / abort_request，以及输出线程上的flush钩子
#   engine.config：Config（第1批），读max_model_len、generation_config等
engine = None
# HF tokenizer，main()中用_load_tokenizer加载，并通过create_engine(tokenizer=tokenizer)交给引擎：服务层和引擎共用同一个对象
#   服务层用它渲染chat template、解码非流式输出、给流式反分词
tokenizer: AutoTokenizer | None = None
# The tool-call format this model emits, resolved once at startup from its
# chat template. `None` means none was recognised and tool calls, if any, are
# delivered as plain text -- said out loud at startup, never discovered here.
# 【译】这个模型输出的工具调用格式，启动时根据它的chat template解析一次。None表示没有识别出任何格式，
#   工具调用（如果有的话）会作为普通文本交付——这一点在启动时就明确说出来，而不是到这里才发现。
# main()中由resolve_tool_call_parser（tool_parser/registry.py）给出；--tool-call-parser可以显式指定
tool_call_parser_cls: type | None = None
# Whether this model's output begins inside the reasoning channel even when
# nothing in the prompt or the output says so -- DeepSeek-R1 closes a block it
# never opens. Read from the chat template at startup, because a single
# response cannot tell you: its first token is already reasoning and reads
# like an answer.
# 【译】这个模型的输出是否从推理通道内部开始，即使prompt和输出里都没有任何标记说明这一点——
#   DeepSeek-R1会关闭一个它从未打开过的块。在启动时从chat template读出，因为单个响应无法告诉你：
#   它的第一个token就已经是推理了，读起来却像答案。
model_starts_in_reasoning: bool = False
# 这个模型的推理"方言"：思考段的开始/结束标记（例如<think>…</think>），
#   main()中由resolve_dialect（reasoning_dialects.py）根据chat template解析
reasoning_dialect: Any = None
# (kwarg, off-value) the chat template reads to switch reasoning off, resolved
# at startup by asking it; None when the template offers no such switch.
# 【译】chat template读取的、用来关闭推理的(kwarg, 关闭值)，启动时通过实际询问模板来解析；模板没有这种开关时为None。
# 解释：实际类型是三元组(kwarg名, 关闭值, 开启值)，例如Qwen3为("enable_thinking", False, True)，
#   由chat_encoders.resolve_reasoning_toggle渲染探测得到
reasoning_toggle: tuple[str, Any, Any] | None = None
# 多模态processor（图像/视频/音频预处理），第一次遇到多模态请求时由_get_multimodal_processor惰性加载
processor: Any | None = None
# API对外的模型名：--served-model-name，未指定时为--model；validate_model用它核对请求里的model字段，响应里也回填它
model_name: str = ""
# --default-chat-template-kwargs解析出的dict，每个请求先拷贝它，再叠加请求自己的chat_template_kwargs
default_chat_template_kwargs: dict[str, Any] = {}
# 模型自带的Python消息编码器（例如DeepSeek-V4的<model>/encoding/encoding_*.py），
#   由load_custom_message_encoder在启动时查找；None表示走tokenizer的Jinja chat template
custom_message_encoder: Any | None = None
# 以下三个字典按请求登记、在cleanup时删除：
#   _seq_id_to_request_id：引擎序列id（int）→ API请求id（"chatcmpl-..."），在setup_streaming_request*里登记
#   _stream_loops：API请求id → 该请求所在的事件循环
#   这两个字典在本文件里只登记和删除，没有按key查询的地方（回调实际通过闭包拿到事件循环）
_seq_id_to_request_id: dict[int, str] = {}
_stream_loops: dict[str, AbstractEventLoop] = {}
#   _request_start_times：API请求id → setup开始时的time.time()，_build_stream_chunk把它作为chunk的started_at
_request_start_times: dict[str, float] = {}
# --request-log打开时的JSONL请求日志logger；None表示关闭（各处先判断它，避免无谓的序列化）
_request_logger: logging.Logger | None = None
# 流式输出的跨线程批量投递器（streaming_dispatch.py），main()中创建；所有流共用一个
_stream_batch_dispatcher: StreamBatchDispatcher | None = None
# `SYNTHETIC_TOKEN_TEXT` while a run's own text is meaningless, None otherwise.
# One switch for both delivery modes: they decode in different places, and a
# response that read differently depending on `stream` is the asymmetry the
# reasoning and tool-call readers were unified to remove.
# 【译】当一次运行自己的文本没有意义时为 SYNTHETIC_TOKEN_TEXT，否则为None。
#   两种投递方式共用这一个开关：它们在不同的地方解码，而一个因 stream 取值不同而读起来不一样的响应，
#   正是推理内容读取器和工具调用读取器被统一时要消除的不对称。
# 只有开启"强制投机接受"的压测选项时才不为None（见main()）：草稿token不再与主模型比对，而是按设定的接受长度 / 接受率直接接受，
#   用来在压测里固定投机解码的收益，生成的文本因此不是模型真正的输出（docs/forced_acceptance_length.md）
synthetic_token_text: str | None = None


# 非流式路径（generate_async*）把收集到的全部token一次性解码成最终文本
#   skip_special_tokens=True：<eos>之类的特殊token不出现在返回给客户端的文本里
#   synthetic_token_text不为None时，照样解码（为了让压测的开销真实），但返回"每个token一段固定占位文本"
def delivered_text(token_ids) -> str:
    """The text a non-streaming response carries for these tokens.

    Decoded even when the answer is thrown away: the runs that stand their text
    in are measuring throughput, and skipping the work the measured server does
    would flatter it. The streaming half of this lives in
    `IncrementalStreamDetokenizer.update`.
    """
    # 【译】非流式响应为这些token携带的文本。
    #
    #   即使答案会被丢弃也照样解码：用占位文本替代真实文本的运行是在测吞吐，跳过被测服务器本来要做的工作会让结果虚高。
    #   流式的那一半在 IncrementalStreamDetokenizer.update 里。
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if synthetic_token_text is None:
        return decoded
    return synthetic_token_text * len(token_ids)


# 决定本请求的输出如何切分"思考段"和"正文"（什么是思考段、为什么要拆，见chat_completions里的【科普】推理内容）：返回ReasoningChannel（reasoning.py），
#   之后用它的.stream()得到流式过滤器，或用.split(text)一次性切分
#   prompt_opens：渲染出的prompt本身是否已经打开了思考段（由prompt_starts_in_reasoning等判断）
#   template_kwargs：合并后的chat template参数，用来判断推理是否被关掉
# 调用方：chat_completions、anthropic_messages、responses_create
def reasoning_channel(
    prompt_opens: bool, *, template_kwargs: dict[str, Any] | None
) -> ReasoningChannel:
    """How to read this request's reasoning channel.

    One place, because it is one answer and both endpoints and both delivery
    modes need the same one. The dialect is the model's, resolved at startup;
    what varies per request is whether the output begins inside the channel.

    The render is why that is not just the model-level fact. A prompt that
    switched reasoning off does not open the channel, and
    `model_starts_in_reasoning` -- which describes the template with reasoning
    *on* -- was OR-ed in regardless, so on a model that begins inside the
    channel implicitly an ordinary answer came back entirely as
    `reasoning_content` with `content` empty.

    ``template_kwargs``, not the request's own `thinking`: the server's
    defaults and the client's `chat_template_kwargs` switch it too, and only
    the merged dict has all three. Reasoning that was asked not to happen and
    happened anyway is still reasoning -- `anthropic_drop_reasoning` exists to
    withhold it, and it can only withhold what was separated.
    """
    # 【译】如何解读这个请求的推理通道（reasoning channel）。
    #
    #   放在一处，因为答案只有一个，而两个接口、两种投递方式都需要同一个答案。方言属于模型，启动时就已确定；
    #   每个请求不同的是：输出是否从推理通道内部开始。
    #
    #   之所以不能只看模型层面的事实，原因在渲染结果。一个关掉了推理的prompt不会打开推理通道，
    #   而 model_starts_in_reasoning（描述的是推理"开启"时的模板）以前不管怎样都被OR进来，
    #   于是在一个隐式从推理通道内开始的模型上，一个普通回答会整个作为 reasoning_content 返回，content 为空。
    #
    #   用 template_kwargs 而不是请求自己的 thinking：服务器默认值和客户端的 chat_template_kwargs 也能切换它，
    #   只有合并后的dict三者都包含。被要求不要发生却仍然发生了的推理依然是推理——
    #   anthropic_drop_reasoning 就是为了扣下它而存在的，而它只能扣下已经被分离出来的部分。
    # starts_open：prompt本身打开了思考段，或者"模型隐式从思考段开始且推理没被关掉"
    switched_off = thinking_switched_off(template_kwargs, reasoning_toggle)
    return ReasoningChannel(
        dialect=reasoning_dialect,
        starts_open=prompt_opens or (model_starts_in_reasoning and not switched_off),
    )


# 以下到_ANTHROPIC_PING_INTERVAL_SECONDS为止是/v1/messages（Anthropic协议）专用的辅助函数（B级）
# anthropic_thinking_enabled：请求是否开启了thinking（{"type": "disabled"}或字段缺失都算没开）
def anthropic_thinking_enabled(request: Any) -> bool:
    """Did this request ask for a reasoning channel?

    `thinking` is absent on most requests and `{"type": "disabled"}` is the
    spelling for switching it off, which is a non-empty dict and therefore
    truthy -- so `bool(request.thinking)` read the standard off-switch as on.
    """
    # 【译】这个请求是否要求了推理通道？
    #
    #   大多数请求没有 thinking 字段，而关闭它的写法是 {"type": "disabled"}——这是个非空dict，因而为真值，
    #   所以 bool(request.thinking) 会把标准的关闭开关读成开启。
    thinking = getattr(request, "thinking", None) or {}
    return bool(thinking) and thinking.get("type") != "disabled"


# 把Anthropic请求的thinking字段翻译成chat template参数（用解析出的reasoning_toggle），只在请求带了该字段时生效
def anthropic_template_kwargs(
    request: Any, toggle: tuple[str, Any, Any] | None
) -> dict:
    """How `thinking` reaches the model: by not asking it to think.

    `thinking: disabled` is answered where the answer costs nothing -- in the
    prompt. The chat template's own switch is set, so the model emits no
    reasoning, so there is none to separate, none to discard, and none for the
    tool parser to misread. This is what SGLang does with the same field and
    what vLLM gets structurally by having no such field at all.

    Everything downstream is then unconditional, which is the point. Three
    attempts to handle an unwanted chain of thought *after* generating it each
    broke something else: discarding it returned an empty message, relabelling
    it as `text` handed the client the thing it declined, and declining to
    separate it fed a chain of thought to the tool parser, which read one
    model's musing about `<function=NAME>` as a call to a tool named `NAME`.
    The reasoning that is never produced needs none of that.

    Only when the field is actually present, which is also what SGLang keys
    on. Anthropic's default is thinking-off, but reading an absent field as
    "switch this model's reasoning off" would silently change what every
    existing caller gets back from a reasoning model. Absent means unstated,
    and unstated leaves the model's own default alone.

    Both directions go through the resolved toggle. Writing a hardcoded
    `thinking=True` for the on direction is a no-op on any template that reads
    another name -- the whole Qwen family -- so an explicit opt-in was
    discarded silently against a server default of off.

    ``toggle`` is ``None`` for a model whose template has no switch. There is
    then nothing to put in the prompt and `anthropic_drop_reasoning` takes
    over.
    """
    # 【译】thinking 如何传达给模型：干脆不要求它思考。
    #
    #   thinking: disabled 在代价为零的地方回应——在prompt里。设置chat template自己的开关，模型就不输出推理，
    #   于是没有推理需要分离、需要丢弃，也没有推理会被工具解析器误读。SGLang对同一字段就是这么做的，
    #   vLLM则因为根本没有这个字段而在结构上得到同样的效果。
    #
    #   这样下游的一切都是无条件的，这正是目的所在。曾有三次尝试在生成"之后"处理不想要的思维链，每次都弄坏了别的东西：
    #   丢弃它会返回空消息；把它重新标成 text 会把客户端拒绝的东西交给客户端；不去分离它则会把思维链喂给工具解析器，
    #   后者把某个模型关于 <function=NAME> 的自言自语读成了对名为 NAME 的工具的调用。从未产生的推理就不需要这些处理。
    #
    #   只在该字段确实存在时才这么做，SGLang也以此为准。Anthropic的默认是关闭思考，但若把字段缺失读成
    #   "关掉这个模型的推理"，会悄悄改变所有现有调用方从推理模型拿到的结果。缺失意味着没有表态，没有表态就不动模型自己的默认值。
    #
    #   开和关两个方向都经过解析出的开关。为开启方向硬编码 thinking=True，在读取别的名字的模板上（整个Qwen家族）是空操作，
    #   于是在服务器默认关闭时，显式的开启请求被悄悄丢掉了。
    #
    #   对模板没有开关的模型，toggle 为 None。此时prompt里没什么可放，由 anthropic_drop_reasoning 接手。
    if getattr(request, "thinking", None) is None:
        return {}
    if toggle is None:
        return {}
    name, off_value, on_value = toggle
    return {name: on_value if anthropic_thinking_enabled(request) else off_value}


# 是否在响应里扣下推理内容（thinking块）不发给客户端：请求没有显式开启thinking时就扣下
def anthropic_drop_reasoning(request: Any) -> bool:
    """Must this response's reasoning be withheld from the client?

    Whenever the request did not ask for it. Answering in the prompt is
    strictly better and `anthropic_template_kwargs` does it wherever it can,
    but that only reaches models whose template has a switch -- and it only
    reaches the ones that *asked*, since an absent field leaves the model's
    default alone. This is everything else.

    Withheld, not left unseparated: the reasoning still goes through the
    reasoning filter, so the tool parser never sees a model musing about
    `<function=NAME>` and calls a tool named `NAME`. Only the `thinking`
    blocks are dropped. A response that was *nothing but* reasoning then ends
    on an empty text block, which is the honest answer to "do not think" --
    the previous three attempts to fix this downstream all failed by trying to
    salvage content out of it.

    Absent counts as off *here*, and that is deliberately not what the
    prompt-level answer does. The two are different questions. What to put in
    the prompt is about the model's own default, and overriding a default
    nobody asked about would change what every existing caller gets from a
    reasoning model. What to put in the *response* is about this protocol's
    default, and Anthropic's is thinking-off: a client that never sends the
    field has no reason to expect `thinking` blocks, and one that validates
    block types or verifies the signature rejects them. Reading absent as
    "show it" sent a random-signature `thinking` block to every Claude Code
    and Anthropic-SDK caller talking to a Qwen3 or DeepSeek deployment.

    Separated either way -- only whether the client is *shown* it is decided
    here. That is what keeps a chain of thought out of the tool parser, which
    is a second reader of the same text.
    """
    # 【译】这个响应的推理内容是否必须对客户端扣下不发？
    #
    #   只要请求没有要求它就扣下。在prompt里回应严格更好，anthropic_template_kwargs 能做的地方都做了，
    #   但那只覆盖模板有开关的模型——而且只覆盖"提出了要求"的请求，因为字段缺失时不动模型的默认值。这里处理其余所有情况。
    #
    #   是扣下，而不是不分离：推理内容仍然经过推理过滤器，所以工具解析器永远看不到模型关于 <function=NAME> 的自言自语，
    #   也就不会去调用名为 NAME 的工具。只丢弃 thinking 块。一个"只有"推理的响应于是以一个空的text块结束，
    #   这是对"不要思考"的诚实回答——之前三次在下游修复它的尝试，都败在试图从中抢救出内容。
    #
    #   在"这里"，缺失算作关闭，这刻意与prompt层面的处理不同。两者是不同的问题。往prompt里放什么关乎模型自己的默认值，
    #   覆盖一个没人问起的默认值会改变所有现有调用方从推理模型得到的结果。往"响应"里放什么关乎本协议的默认值，
    #   而Anthropic的默认是关闭思考：从不发送该字段的客户端没有理由期待 thinking 块，校验块类型或验证签名的客户端会拒收它们。
    #   把缺失读成"显示它"，曾让每个连接Qwen3或DeepSeek部署的Claude Code和Anthropic SDK调用方都收到带随机签名的 thinking 块。
    #
    #   无论如何都会分离——这里只决定是否"展示"给客户端。正是这一点让思维链远离工具解析器，后者是同一段文本的第二个读者。
    return not anthropic_thinking_enabled(request)


# The engine's `leave_reason` in Anthropic's vocabulary. The scheduler emits
# exactly seven shapes; the three here line up by name, `stop_<token_id>` is
# handled below, and `aborted` / `unschedulable: ...` / "" have no counterpart
# in Anthropic's vocabulary and keep the default.
# 【译】引擎的 leave_reason 在Anthropic词汇里的说法。调度器恰好产生七种形态；这里的三种按名字一一对应，
#   stop_<token_id> 在下面处理，aborted / unschedulable: ... / "" 在Anthropic的词汇里没有对应项，保持默认值。
_ANTHROPIC_STOP_REASON = {
    "eos": "end_turn",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}


# 有工具调用时stop_reason为tool_use，但max_tokens优先
def anthropic_stop_reason_with_calls(finish_reason: Any, has_calls: bool) -> str:
    """`max_tokens` outranks `tool_use`, for the reason `length` outranks
    `tool_calls` on the other endpoint (see
    :func:`~.protocol.openai_stop_reason_with_calls`): only one of the two is a
    warning, and a response cut off mid-call parses to a call with a silently
    truncated argument value."""
    # 【译】max_tokens 优先于 tool_use，理由与另一个接口上 length 优先于 tool_calls 相同（见 protocol.openai_stop_reason_with_calls）：
    #   两者中只有一个是警告，而在调用中途被截断的响应会解析出一个参数值被悄悄截断的调用。
    reason = anthropic_stop_reason(finish_reason)
    if reason == "max_tokens":
        return reason
    return "tool_use" if has_calls else reason


# 引擎的结束原因（Sequence.leave_reason）→ Anthropic的stop_reason
def anthropic_stop_reason(finish_reason: Any) -> str:
    """The engine's leave reason as Anthropic spells it.

    Not routed through `protocol.openai_stop_reason`: that maps into OpenAI's
    vocabulary (`stop` / `length`), which shares no member with the keys here,
    so chaining them would send every reason to the default.

    `stop_<token_id>` is *not* `stop_sequence`, though it was mapped to it.
    The two come from different branches of the scheduler and mean opposite
    things: `stop_sequence` is one of the client's own `stop_sequences`
    matching, and Anthropic pairs it with the matched string in the response;
    `stop_<id>` is a model end-of-turn token from `stop_token_ids` firing,
    which is an ordinary `end_turn`.

    `stop_token_ids` is `generation_config.eos_token_id` minus the single
    `tokenizer.eos_token_id`, so any model declaring more than one EOS reaches
    this branch in normal operation -- Qwen3, Qwen3.5, gpt-oss.
    """
    # 【译】按Anthropic的拼法给出引擎的结束原因。
    #
    #   不经过 protocol.openai_stop_reason：那个函数映射到OpenAI的词汇（stop / length），与这里的key没有交集，
    #   串起来会让每个原因都落到默认值。
    #
    #   stop_<token_id> "不是" stop_sequence，尽管以前被映射成它。两者来自调度器的不同分支，含义相反：
    #   stop_sequence 是客户端自己的某个 stop_sequences 匹配上了，Anthropic会在响应里配上匹配到的字符串；
    #   stop_<id> 是 stop_token_ids 里某个模型的回合结束token触发了，这是普通的 end_turn。
    #
    #   stop_token_ids 是 generation_config.eos_token_id 去掉唯一的 tokenizer.eos_token_id，
    #   所以任何声明了多个EOS的模型在正常运行中都会走到这个分支——Qwen3、Qwen3.5、gpt-oss。
    reason = finish_reason or ""
    if reason.startswith("stop_") and reason != "stop_sequence":
        return "end_turn"
    return _ANTHROPIC_STOP_REASON.get(reason, "end_turn")


# Anthropic's own keepalive. Sent in place of a reasoning segment the request
# asked not to see: the bytes are being generated either way, and with nothing
# going out the socket is silent for the whole chain of thought -- on an
# R1-shaped 5019-character trace the first client-visible frame arrived after
# 5016 of them. Long enough to trip proxy and SDK idle-read timeouts, and the
# stall watchdog can only report it, not prevent it.
# 【译】Anthropic协议自己的保活帧。用来代替请求不想看到的推理片段：这些字节反正都在生成，
#   如果什么都不发，整段思维链期间socket都是静默的——在一条R1形态、5019个字符的输出上，客户端可见的第一帧
#   直到其中5016个字符之后才到达。这足以触发代理和SDK的空闲读超时，而卡顿看门狗只能报告它，不能阻止它。
_ANTHROPIC_PING_FRAME = event_frame("ping", {"type": "ping"})
# Paced by the clock, not by the model: one ping per dropped reasoning
# *segment* is one per engine chunk, i.e. O(reasoning tokens) socket writes of
# 35 discarded bytes. The keepalive only has to beat proxy and SDK idle-read
# timeouts, which are tens of seconds.
# 【译】按时钟节拍发送，而不是按模型节拍：每丢弃一个推理"片段"发一次ping，就是每个引擎chunk一次，
#   即 O(推理token数) 次socket写、每次写35个被丢弃的字节。保活只需赶在代理和SDK的空闲读超时（几十秒）之前即可。
_ANTHROPIC_PING_INTERVAL_SECONDS = 5.0
# Prometheus指标（metrics_setup.py），在导入本模块时就创建：
#   _metrics_exporter：/metrics接口渲染的导出器；_request_metrics：请求级指标（如TTFT，由RequestTimingMiddleware上报）；
#   _stream_metrics：流级指标（如token间延迟ITL，由StreamBatchDispatcher._deliver上报）
#   【分支】指标体系见第10批metrics/
_metrics_exporter, _request_metrics, _stream_metrics = create_metrics_exporter()
# lifespan里启动的后台周期任务（刷新指标、GC监视），关闭时统一取消
_background_tasks: list[asyncio.Task] = []
# The watch compares two `gc.get_stats()` reads against something that moves on
# the scale of minutes, so it has no reason to ride the metrics cadence.
# 【译】这个监视比较两次 gc.get_stats() 的读数，被比较的东西以分钟为尺度变化，所以没有理由跟着指标的刷新节拍走。
_GC_WATCH_INTERVAL_SECONDS = 60.0

# Not the default pool: that one runs every request's prompt preprocessing, so
# a curl loop against `/debug/gc_census` would fill it with heap walks and block
# admission. One worker is also the single-flight -- each walk pins a gen-2
# snapshot. Starts no thread until first submit, and is never shut down: the
# interpreter joins it at exit, while a per-app teardown would leave the
# endpoint dead if the app ever restarted in-process.
# 【译】不用默认线程池：默认池要跑每个请求的prompt预处理，对 /debug/gc_census 的curl循环会用堆遍历把它塞满、阻塞请求准入。
#   只有一个worker也就保证了同一时刻只有一个遍历——每次遍历都会钉住一份第2代的快照。
#   第一次提交之前不启动线程，也从不关闭：解释器退出时会join它；而按应用生命周期关闭的话，
#   一旦应用在进程内重启，这个接口就失效了。
# 解释："默认线程池跑prompt预处理"指下文的loop.run_in_executor(None, do_preprocess)，None即事件循环的默认线程池
#   "第2代"：CPython的循环垃圾回收器把对象按存活时间分成第0、1、2代，越老的代越少被扫描；gc.freeze把对象移出这三代（见lifespan）
_gc_census_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gc-census")


# 主路径会调用（chat_completions / completions）：从HTTP头取会话ID，交给CoreManager的DP路由，
#   让同一会话的多轮请求落到同一个DP rank，复用那个rank上的前缀缓存
#   只在环境变量ATOM_DP_SESSION_AFFINITY为1/true/yes/on时生效（这里直接读os.environ，不经过envs.py）；
#   返回(session_id, parent_id)，没开或没有header时为(None, None)
#   【分支】DP路由策略见第2批engine_core_mgr.py
def _get_dp_session_affinity_ids(
    raw_request: Request | None,
) -> tuple[str | None, str | None]:
    """Extract AIPerf session lineage for CoreManager's DPA router.

    AIPerf keeps ``X-Correlation-ID`` stable across the turns of one session.
    When its Dynamo-affinity header option is enabled it also sends the parent
    session ID for forked agent trees. CoreManager retains that lineage for
    observability but deliberately assigns each child correlation ID its own
    strict cache owner, matching SGLang's agentic routing. Clients without a
    session header retain normal DP load balancing.
    """
    # 【译】为CoreManager的DPA路由器提取AIPerf的会话血缘。
    #
    #   AIPerf在一个会话的多轮之间保持 X-Correlation-ID 不变。启用它的Dynamo亲和性header选项时，
    #   它还会为分叉出来的agent树发送父会话ID。CoreManager保留这条血缘用于可观测性，但刻意给每个子correlation ID
    #   分配它自己的严格缓存归属者，与SGLang的agentic路由一致。没有会话header的客户端仍走正常的DP负载均衡。
    if os.environ.get("ATOM_DP_SESSION_AFFINITY", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, None
    if raw_request is None:
        return None, None

    session_id = raw_request.headers.get(
        "x-dynamo-session-id"
    ) or raw_request.headers.get("x-correlation-id")
    parent_id = raw_request.headers.get("x-dynamo-parent-session-id")
    return session_id, parent_id


# ============================================================================
# Request/Response Logging
# ============================================================================
# 【译】请求/响应日志


# --request-log打开时，请求、响应（流式时是每一帧）各写一行JSON；default=str让无法序列化的对象退化成字符串
def _log_request_event(event_type: str, request_id: str, data: Any) -> None:
    """Write a JSONL entry to the request log file (if enabled)."""
    # 【译】往请求日志文件写一条JSONL记录（如果启用了）。
    if _request_logger is None:
        return
    entry = {
        "timestamp": time.time(),
        "request_id": request_id,
        "type": event_type,
        "data": data,
    }
    _request_logger.info(json.dumps(entry, default=str))


def _log_request_model(event_type: str, request_id: str, model: Any) -> None:
    """:func:`_log_request_event` for a pydantic model, dumped only if logged.

    The guard in `_log_request_event` is in the callee, so
    ``_log_request_event("request", rid, request.model_dump())`` builds the
    dump before the call can decline it: every message and every tool schema
    serialised on the event loop and thrown away, on every request, with
    request logging off. Measured 20-26 us on an agent-shaped request against
    0.07 us for the guard.

    The client stream wrapper also skips parsing when no observer needs it.
    """
    # 【译】pydantic模型版的 _log_request_event：只有真要记录时才dump。
    #
    #   _log_request_event 的判断在被调用方里，所以 _log_request_event("request", rid, request.model_dump())
    #   会在调用能拒绝之前就把dump做完：即使关了请求日志，每个请求的每条消息、每个工具schema都要在事件循环上
    #   序列化一遍再扔掉。在agent形态的请求上实测要20-26微秒，而这个判断只要0.07微秒。
    #
    #   客户端流包装器同样在没有观察者需要时跳过解析。
    if _request_logger is None:
        return
    _log_request_event(event_type, request_id, model.model_dump())


# 所有流式接口的最外层包装：StreamingResponse(_client_stream(gen, request_id))
#   gen是serving_chat / serving_completion等产生SSE帧的异步生成器；这里逐帧透传，顺带做三件事：
#   1. 静默看门狗（FrameWait）：两帧之间等待太久时记日志，并让/metrics能看到"最饿的流等了多久"
#   2. TTFT：第一帧含生成内容的数据到达时记录首个输出的时间（只记一次）
#   3. --request-log：把每帧的JSON写进请求日志
# 语法：async def里含yield就是"异步生成器"，用async for迭代；这里手动调用__aiter__/__anext__，
#   效果等同于async for，但能让with FrameWait(...)只包住"等待下一帧"这一个await
async def _client_stream(
    gen: AsyncGenerator[str, None], request_id: str
) -> AsyncGenerator[str, None]:
    """Every SSE frame on its way to the client: logged, and timed.

    The last point a frame passes through before uvicorn writes it, which is
    why the silence watchdog lives here rather than at the collector it
    started at. Between the collector and this line sit the reasoning
    channel's read-ahead and the tool-call format's, and while either
    withholds, the collector keeps waking on schedule -- so the gauge read
    zero for exactly the stall it exists to report.

    Wrapping every streaming response and not two of the three: this was
    `_logged_stream`, and the Anthropic endpoint never used it. A watchdog
    with an endpoint-shaped hole in it is worse than none, because the zero it
    reports looks like an answer.
    """
    # 【译】每个发往客户端的SSE帧都经过这里：被记录、被计时。
    #
    #   这是帧在uvicorn写出之前经过的最后一站，所以静默看门狗放在这里，而不是它最初所在的collector。
    #   在collector和这一行之间还有推理通道的预读和工具调用格式的预读，只要其中任何一个扣着不发，
    #   collector仍会按时醒来——于是这个仪表恰恰在它要报告的卡顿期间读数为零。
    #
    #   包装每一个流式响应，而不是三个中的两个：它原来叫 _logged_stream，而Anthropic接口从没用过它。
    #   一个留着"接口形状的漏洞"的看门狗比没有还糟，因为它报告的零看起来像一个答案。
    # StreamingResponse sends its headers before iterating this generator, so
    # the middleware has already marked whether this response is eligible.
    # 【译】StreamingResponse在迭代这个生成器之前就发出了响应头，所以中间件已经标记好这个响应是否计入统计。
    # timing：RequestTimingMiddleware（request_timing.py）放进ContextVar的本请求计时对象，不需要量TTFT时为None
    timing = get_stream_timing()
    it = gen.__aiter__()
    # 是否已经交付过至少一帧；第一帧之前的等待是排队+prefill，不算"静默"
    delivered = False
    while True:
        # The first wait is the queue, not silence: every generator awaits the
        # collector before its opening frame.
        # 【译】第一次等待是在排队，不算静默：每个生成器在发出第一帧之前都要先await collector。
        # FrameWait（streaming_dispatch.py）：with块给这一次等待计时；armed=False（第一帧之前）时不计入静默统计
        # await it.__anext__() 取下一帧；生成器结束时抛StopAsyncIteration，这里直接return结束本生成器
        with FrameWait(request_id, armed=delivered):
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                return
        delivered = True
        # Parse each complete local frame once for both logging and TTFT.
        # After first output (or a terminal event), logging alone needs parsing.
        # 【译】每个完整的本地帧只解析一次，同时服务于日志和TTFT。首个输出（或终止事件）之后，只有日志还需要解析。
        # 既不量TTFT也没开请求日志时，完全不解析，直接透传（在事件循环上解析JSON也是开销）
        # iter_sse_data（sse.py）：从可能拼在一起的多帧字符串里逐个取出data部分
        if timing is not None or _request_logger is not None:
            for data in iter_sse_data(chunk):
                if data == "[DONE]":
                    timing = None
                    _log_request_event("stream_done", request_id, None)
                else:
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        # Diagnostics must not swallow malformed output.
                        # 【译】诊断代码不能吞掉格式错误的输出。
                        _log_request_event("stream_chunk_unparsed", request_id, data)
                        continue
                    _log_request_event("stream_chunk", request_id, payload)
                    if timing is not None:
                        # 错误帧不算首个输出：停止计时，不记录TTFT
                        if isinstance(payload, dict) and (
                            "error" in payload or payload.get("type") == "error"
                        ):
                            timing = None
                        # 第一帧带有生成内容（文本、推理内容或工具调用）：记录TTFT，之后不再计时
                        elif has_generated_output(payload):
                            timing.first_output(streaming=True)
                            timing = None
                # 两件事都不需要了，本chunk剩下的帧不必再解析
                if timing is None and _request_logger is None:
                    break
        # 原样交给StreamingResponse，由uvicorn写到socket
        yield chunk


# ============================================================================
# Engine Interface
# ============================================================================
# 【译】引擎接口
# 本节是服务层与引擎（LLMEngine / CoreManager）之间的全部接口：
#   请求进：_build_sampling_params、_coerce_n、长度校验 → generate_async*（非流式）/ setup_streaming_request*（流式）
#   结果出：引擎输出线程上的回调 → 非流式：call_soon_threadsafe进asyncio.Queue；流式：StreamBatchDispatcher批量投递
#   收尾：cleanup_stream / cleanup_request；客户端断开：_race_disconnect → abort_request


# 把OpenAI请求里的采样参数原样搬进引擎的SamplingParams（第1批sampling_params.py）
#   stop_strings：停止字符串，preprocess里会把它们tokenize成stop_token_sequences
#   SamplingParams.__post_init__会校验top_k / top_p，非法时抛ValueError → 接口返回400
# 注意：请求里的seed、presence_penalty、frequency_penalty不会传进来（protocol.py里只是"为兼容而接受"）
# 原理：采样参数的"默认值"有两种来源
#   协议默认值：OpenAI规范里temperature=1、top_p=1，即直接按模型原始分布采样；protocol.py的DEFAULT_*就是这一组
#   模型推荐值：模型作者写在generation_config.json里的值，例如Qwen3写的是temperature=0.6、top_p=0.95、top_k=20——
#     这是作者在该模型上调出的最好设置（推理模型在temperature=1时更容易发散、重复）
#   chat / completions接口在请求没传时用协议默认值（pydantic字段的默认值就是DEFAULT_*，到这里已分不清"没传"和"传了1.0"）；
#   /v1/messages和/v1/responses则优先取模型推荐值。所以评测或压测chat接口时，最好显式传采样参数
# logprobs：SamplingParams本身有logprobs字段（离线调用LLMEngine.generate时把它设为真，输出dict里才有每个token的logprob），但HTTP请求模型没有声明它
#   （extra="ignore"直接丢弃），这里也不传，所以serving_chat / serving_completion的帧里"logprobs"恒为None
def _build_sampling_params(
    temperature: float,
    max_tokens: int,
    stop_strings: list[str] | None,
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        ignore_eos=ignore_eos,
        n=n,
    )


# 计算实际的n（一个prompt采样几条回答）。例子：
#   n=None → 1；n="3" → 3；n=0 → 1
#   n=4、temperature=0（贪心）→ 1（4条结果必然一样，不值得占4份KV cache）
#   n=4、temperature=0.7 → 4，后续走扇出路径：preprocess_fanout建4个兄弟Sequence，共享同一个prompt
def _coerce_n(requested_n: int | None, temperature: float | None) -> int:
    """Return an effective ``n`` for a request.

    * ``None``/``<1`` coerce to ``1`` (matches OpenAI default).
    * ``n > 1`` combined with greedy sampling (``temperature <= 0``) is
      collapsed to ``1`` because all siblings would produce identical
      outputs — other runtimes (vLLM, TGI) silently do the same, and it
      avoids wasting KV cache on duplicate decodes.
    """
    # 【译】返回一个请求实际使用的 n。
    #
    #   * None 或 <1 强制为 1（与OpenAI的默认一致）。
    #   * n > 1 且为贪心采样（temperature <= 0）时折叠成 1，因为所有兄弟序列都会产生相同的输出——
    #     其他运行时（vLLM、TGI）也是悄悄这样做的，这也避免在重复的decode上浪费KV cache。
    # 解释：代码里temperature为None时也按贪心处理（折叠成1），docstring没有提到这一点
    n = requested_n if requested_n is not None else 1
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    n = max(n, 1)
    if n > 1 and (temperature is None or temperature <= 0.0):
        logger.info(
            "n=%s requested with temperature=%s; collapsing to n=1 because "
            "greedy sampling would produce identical siblings.",
            n,
            temperature,
        )
        n = 1
    return n


# 声明长度校验：prompt token数 + 请求的输出token数 > max_model_len 时抛ValueError（接口返回400，报错文本仿照OpenAI）
# 例：max_model_len=32768，prompt 30000个token，没传max_tokens（默认8192，见protocol.py）
#   → 30000 + 8192 = 38192 > 32768 → 拒绝；客户端要么缩短prompt，要么把max_tokens调小到2768以内
def _validate_context_length(
    num_prompt_tokens: int,
    max_tokens: int,
    max_model_len: int | None,
) -> None:
    if max_model_len is None:
        return

    requested_output_tokens = max(0, int(max_tokens or 0))
    total_tokens = int(num_prompt_tokens) + requested_output_tokens
    if total_tokens <= int(max_model_len):
        return

    raise ValueError(
        f"This model's maximum context length is {max_model_len} tokens. "
        f"However, you requested {requested_output_tokens} output tokens and "
        f"your prompt contains at least {num_prompt_tokens} input tokens, for "
        f"a total of at least {total_tokens} tokens. Please reduce the length "
        f"of the input prompt or the number of requested output tokens."
    )


# 取引擎的Config：优先engine.config，没有时退到engine.io_processor.config（用getattr链，测试里的替身引擎也能用）
def _get_engine_config():
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(getattr(engine, "io_processor", None), "config", None)
    return config


# 引擎还没建好时返回None，此时跳过声明长度校验
def _get_engine_max_model_len() -> int | None:
    return getattr(_get_engine_config(), "max_model_len", None)


# 物理容量：KV池能装下的最长prompt。CoreManager取所有rank在READY消息里报告的最小值（第2批engine_core_mgr.py）
def _get_engine_max_pool_tokens() -> int | None:
    """Longest prompt the KV pool can hold, as each engine rank reported it.

    None before the engine is up, or on a manager that never learned it, in
    which case the scheduler remains the only enforcer.
    """
    # 【译】KV池能容纳的最长prompt，取自各引擎rank的报告。
    #
    #   引擎启动前，或者在从未得知这个值的manager上，为None，此时调度器是唯一的把关者。
    return getattr(getattr(engine, "core_mgr", None), "max_pool_tokens", None)


# 物理容量校验：prompt的KV连一个完全空的KV池都放不下时，直接拒绝（400）
def _validate_pool_capacity(
    num_prompt_tokens: int, max_pool_tokens: int | None
) -> None:
    """Refuse a prompt whose KV cannot fit even a completely empty pool.

    `max_model_len` is a declared limit; this is a physical one, since the pool
    is sized from whatever device memory is free once the weights are loaded, so
    on a tight pool it binds first. The scheduler checks it too, but only once
    the request reaches the engine — by then the client holds a response it will
    never be answered on, so the request has to be turned away here instead.
    """
    # 【译】拒绝那些即使KV池完全空着也放不下其KV的prompt。
    #
    #   max_model_len 是声明出来的上限；这里是物理上限：KV池的大小取决于权重加载完后还剩多少显存，
    #   所以池子紧张时它先起作用。调度器也会检查，但要等请求到达引擎之后——那时客户端已经拿着一个
    #   永远等不到回答的响应了，所以必须在这里就把请求拒掉。
    if max_pool_tokens is None or int(num_prompt_tokens) <= int(max_pool_tokens):
        return

    raise ValueError(
        f"This server's KV cache holds at most {max_pool_tokens} tokens for a "
        f"single request, and your prompt contains at least {num_prompt_tokens} "
        f"input tokens. Please shorten the prompt, or restart the server with a "
        f"higher --gpu-memory-utilization to enlarge the cache."
    )


# 对preprocess产出的Sequence做两项长度校验；num_prompt_tokens是tokenize后的真实长度，max_tokens来自SamplingParams
# 调用方：generate_async*、setup_streaming_request*，都在preprocess之后、add_request之前调用；
#   校验失败时调用方负责把序列从io_processor.requests里删掉，再把异常往上抛
def _validate_sequence_context_length(seq) -> None:
    _validate_context_length(
        seq.num_prompt_tokens,
        seq.max_tokens,
        _get_engine_max_model_len(),
    )
    _validate_pool_capacity(seq.num_prompt_tokens, _get_engine_max_pool_tokens())


# 【分支】多模态（第2批2b multimodal/*）：以下三个函数处理带图像/视频/音频的chat请求
# 判断消息里是否有媒体内容片段（chat_utils.has_multimodal_content）
def _has_multimodal_content(messages: list[Any]) -> bool:
    return has_multimodal_content(
        [{"content": message.content} for message in messages]
    )


# 惰性加载多模态processor：先尝试ATOM原生实现（multimodal/registry.py），没有时退回HF的AutoProcessor
#   global声明：要给模块级变量processor赋值
def _get_multimodal_processor():
    global processor
    if processor is None:
        logger.info(f"Loading multimodal processor from {model_name}...")
        processor = get_native_multimodal_processor(
            _get_engine_config(), tokenizer, custom_message_encoder
        )
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
    return processor


# 把消息里的媒体抽出来（chat_utils.parse_chat_messages），再由prepare_multimodal_inputs渲染模板、
#   预处理媒体，返回(prompt token id列表, 多模态数据dict)；它可能下载URL，所以调用方放在线程池里执行
def _prepare_multimodal_inputs(
    messages: list[Any],
    chat_template_kwargs: dict[str, Any],
    tools: Any = None,
) -> tuple[list[int], dict[str, Any]]:
    conversation, media = parse_chat_messages(
        [
            {**message.to_template_dict(), "content": message.content}
            for message in messages
        ]
    )
    return prepare_multimodal_inputs(
        _get_engine_config(),
        _get_multimodal_processor(),
        conversation,
        media,
        chat_template_kwargs,
        tools=tools,
    )


# ── Batched stream dispatch ──────────────────────────────────────────────
# 【译】── 批量流式投递 ──
# 以下四个函数运行在引擎输出线程上（不是事件循环线程），由流式回调调用；机制见文件头的图2


# 把引擎的RequestOutput转成一个原始chunk（dict）。只放token id，文本由StreamOutputCollector.get()在事件循环上解码后补上"text"键
def _build_stream_chunk(request_output: RequestOutput, request_id: str) -> dict:
    """Build a raw chunk; detokenization happens once in the batch dispatcher."""
    # 【译】构造一个原始chunk；反分词在批量投递器里只做一次。
    # 解释：对StreamOutputCollector这类消费者，投递器只是把反分词器挂到chunk上，真正的解码发生在事件循环上的collector.get()里
    #   （见streaming_dispatch.py的StreamBatchDispatcher.flush）
    started_at = _request_start_times.get(request_id)
    # chunk各字段：
    #   token_ids：本step新产生的token id（投机解码时一个step可能有多个）
    #   finished / finish_reason：是否结束、引擎的结束原因（Sequence.leave_reason，如"eos"、"max_tokens"、"stop_<token_id>"）
    #   finished_at / started_at：时间戳（atom/下没有找到读取这两个键的代码）
    #   num_cached_tokens：前缀缓存命中的token数，最终写进usage.prompt_tokens_details.cached_tokens
    chunk_data = {
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
        "finished_at": time.time(),
        "started_at": started_at,
        "num_cached_tokens": getattr(request_output, "num_cached_tokens", 0),
    }
    # 【分支】PD分离：prefill节点把KV所在位置（kv_transfer_params）通过响应带回给PD路由器，见第9批
    if getattr(request_output, "kv_transfer_params_output", None):
        chunk_data["kv_transfer_params"] = request_output.kv_transfer_params_output
    return chunk_data


# 单序列流式回调的实际工作：把chunk交给投递器缓冲，等本step结束统一flush
#   state：setup_streaming_request里new_state()建的StreamState（反分词器 + 投递计时），每个流一个、全程复用
#   loop / stream_collector：这个请求所在的事件循环和它的收件箱
def _send_stream_chunk_direct(
    request_output: RequestOutput,
    request_id: str,
    stream_collector: StreamOutputCollector,
    loop: AbstractEventLoop,
    state: Any,
) -> None:
    """Buffer a single-request chunk for this engine step."""
    # 【译】为当前引擎step缓冲一个单请求chunk。
    assert _stream_batch_dispatcher is not None
    _stream_batch_dispatcher.enqueue(
        loop=loop,
        collector=stream_collector,
        state=state,
        chunk=_build_stream_chunk(request_output, request_id),
    )


# main()把它注册为engine.core_mgr._flush_stream_batch_fn；CoreManager的输出线程在处理完一条STREAM消息里的
#   所有回调后调用它（拒绝/abort的序列补发终止输出后也会调用）
# 缓冲在threading.local里，所以每个DP rank的输出线程只冲刷自己攒下的那一批
def flush_stream_batch() -> None:
    """Flush this output thread's engine-step batch to the stream collectors."""
    # 【译】把本输出线程攒下的这一个引擎step的批次刷给各个流的collector。
    if _stream_batch_dispatcher is not None:
        _stream_batch_dispatcher.flush()


# n>1扇出时的回调工作：和_send_stream_chunk_direct相同，只是多带一个tag=兄弟序号
def _send_stream_chunk_tagged(
    request_output: RequestOutput,
    request_id: str,
    sibling_index: int,
    stream_collector: StreamOutputCollector,
    loop: AbstractEventLoop,
    state: Any,
) -> None:
    """Variant of :func:`_send_stream_chunk_direct` for fan-out siblings.

    Pushes ``(sibling_index, chunk_data)`` tuples into a single shared
    collector so the merge-stream consumer in :mod:`serving_chat` /
    :mod:`serving_completion` can demultiplex by index. The collector folds
    per tag, so a lagging consumer never mixes two siblings' deltas.

    This path serves ``SamplingParams.n > 1`` by tagging each sibling's chunks
    so the shared stream consumer can merge them in order.
    """
    # 【译】_send_stream_chunk_direct 面向扇出兄弟序列的变体。
    #
    #   把 (sibling_index, chunk_data) 元组推进同一个共享collector，使 serving_chat / serving_completion 中
    #   合并流的消费者能按index拆分。collector按tag分别合并，所以落后的消费者永远不会把两个兄弟的增量混在一起。
    #
    #   这条路径服务于 SamplingParams.n > 1：给每个兄弟的chunk打上标签，让共享的流消费者能按顺序合并它们。
    assert _stream_batch_dispatcher is not None
    _stream_batch_dispatcher.enqueue(
        loop=loop,
        collector=stream_collector,
        state=state,
        chunk=_build_stream_chunk(request_output, request_id),
        tag=sibling_index,
    )


# =====================================================================================
# generate_async：非流式、n=1 的主路径（chat / completions / messages / responses 都会用到）
# =====================================================================================
# 形式上是异步生成器，但只在最后yield一次（完整结果dict）；调用方用_run_nonstream_with_disconnect驱动它
# 原理：非流式请求在整个生成期间，客户端收不到任何字节（响应头也要等到最后），所以长生成容易撞上
#   代理或SDK的读超时；而服务端这边也无从得知客户端是否还在——这就是下面_race_disconnect要单独监听断开事件的原因
# 步骤：
#   1. 建一个asyncio.Queue作为本请求的收件箱，定义在输出线程上运行的回调completion_callback
#   2. 线程池里preprocess：tokenize + 构造Sequence（回调挂在Sequence上）
#   3. 长度校验，通过后add_request交给引擎
#   4. await队列，累积token，直到finished
#   5. finally：没正常结束就abort；无论如何把Sequence从io_processor.requests删掉
#   6. 解码全文，算TTFT / TPOT，yield结果dict（交给serving_chat.build_chat_response等构造响应）
# 参数：prompt_or_tokens是prompt字符串或token id列表；kv_transfer_params是PD分离元数据（第9批）；
#   data_parallel_rank / dp_session_id / dp_parent_session_id是DP路由提示；return_token_ids表示要回显prompt的token id
async def generate_async(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
    return_token_ids: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate non-streaming output, optionally echoing local prompt token IDs."""
    # 【译】生成非流式输出，可选地回显本地的prompt token ID。
    # asyncio.Queue只能在事件循环线程里操作；loop记下当前事件循环，供输出线程上的回调跨线程投递
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # started_at在preprocess之前取，所以TTFT包含tokenize、排队和prefill
    started_at = time.time()
    first_token_at: float | None = None
    last_token_at: float | None = None
    # An array, not a list: this grows for the whole life of the request,
    # and one boxed PyInt per token is what the collector then walks on
    # every pass. It stays an array all the way out -- the dict below is
    # an internal hand-off to `build_*_response`, which reads `text` and
    # the counters and never `token_ids`, so nothing serializes it. A
    # consumer that starts reading that key has to convert.
    # 【译】用array而不是list：它在请求的整个生命周期里不断增长，而每个token一个装箱的PyInt，
    #   （垃圾）回收器每一轮都要遍历它们。它一路保持array直到最后——下面的dict是交给 build_*_response 的内部数据，
    #   后者读取 text 和各计数，从不读 token_ids，所以没有东西会序列化它。若有消费者开始读这个key，就得自己转换。
    all_token_ids = new_token_ids()
    # finish_reason：引擎的结束原因；seq：preprocess建出的Sequence（finally要用）；
    # kv_transfer_output_meta_info：PD分离时prefill节点返回的KV位置；num_cached_tokens_seen：前缀缓存命中数
    finish_reason: str | None = None
    seq = None
    kv_transfer_output_meta_info = None
    num_cached_tokens_seen = 0

    # 挂在Sequence上的stream_callback（非流式请求同样靠它收集token）：在CoreManager的输出线程上执行（不是事件循环线程），
    #   每个step本序列有新输出时调用一次
    def completion_callback(request_output: RequestOutput):
        # nonlocal：在内层函数里给外层函数的局部变量赋值（只读不需要声明）；这两个变量只由输出线程写，循环结束后才被读取
        nonlocal kv_transfer_output_meta_info, num_cached_tokens_seen
        kv_transfer_output_meta_info = getattr(
            request_output, "kv_transfer_params_output", None
        )
        _ct = getattr(request_output, "num_cached_tokens", 0)
        if _ct:
            num_cached_tokens_seen = _ct
        # 跨线程交接：asyncio.Queue不是线程安全的，不能在输出线程里直接put；
        #   call_soon_threadsafe把"token_queue.put_nowait(item)"排进事件循环的待执行队列并唤醒它，稍后在事件循环线程里执行
        # 这里把RequestOutput拆成普通dict，ts记录输出线程收到这批token的时刻
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    # 在线程池里执行：tokenize长prompt是CPU密集的同步操作，放在事件循环上会卡住所有请求
    #   preprocess（第2批llm_engine.py）：tokenize → 把stop_strings转成token序列 → 构造Sequence
    #   → 登记到io_processor.requests → 返回Sequence（开启前缀缓存时还会在这里算好prompt的block hash链，第4批）
    def do_preprocess():
        return engine.io_processor.preprocess(
            prompt_or_tokens,
            sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=kv_transfer_params,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    # run_in_executor(None, fn)：在事件循环的默认线程池里执行fn，await等它完成而不阻塞事件循环
    seq = await loop.run_in_executor(None, do_preprocess)
    # 校验失败时还没add_request，引擎不知道这条序列，只需从requests里删掉
    try:
        _validate_sequence_context_length(seq)
    except Exception:
        engine.io_processor.requests.pop(seq.id, None)
        raise
    # 交给引擎：CoreManager登记 seq.id → 回调，并把Sequence发给某个DP rank的EngineCore（第2批engine_core_mgr.py）
    engine.core_mgr.add_request([seq])

    # 是否收到了引擎的finished输出；没收到就离开（客户端断开导致任务被取消等），finally里要abort
    _finished_ok = False
    try:
        while True:
            # 等下一批输出：在这里挂起，事件循环去服务别的请求；回调把item放进队列后这里被唤醒
            item = await token_queue.get()
            token_ids = item.get("token_ids") or []
            if token_ids:
                # 首个非空输出：记录首token时间，并通知request_timing（非流式的TTFT统计）
                if first_token_at is None:
                    first_token_at = item.get("ts", time.time())
                    record_nonstream_first_token()
                last_token_at = item.get("ts", time.time())
                all_token_ids.extend(token_ids)
            # 引擎的最后一个输出带finished=True（可能同时带最后几个token，也可能为空列表）
            if item.get("finished", False):
                finish_reason = item.get("finish_reason")
                _finished_ok = True
                break
    finally:
        # Two responsibilities, on EVERY exit path:
        #   1) If we didn't finish (client disconnected / cancelled), tell the
        #      engine to stop so the seq doesn't run to max_tokens and burn GPU.
        #   2) Always drop the seq from io_processor.requests. The engine frees
        #      its own KV on finish, but this dict is only cleaned up here for
        #      non-stream requests -- without an unconditional pop, every
        #      completed non-stream request leaks a Sequence (pending grows
        #      forever). Streaming pops via cleanup_stream instead.
        # 【译】在"每一条"退出路径上都有两项职责：
        #     1) 如果没有正常结束（客户端断开 / 被取消），通知引擎停止，免得这条序列一直跑到max_tokens白白烧GPU。
        #     2) 总是把序列从 io_processor.requests 里删掉。引擎结束时会释放自己的KV，但对非流式请求，这个dict只在这里清理——
        #        如果不是无条件pop，每个完成的非流式请求都会泄漏一个Sequence（pending数永远增长）。流式请求则通过 cleanup_stream 删除。
        # contextlib.suppress(Exception)：相当于 try: ... except Exception: pass
        if seq is not None:
            if not _finished_ok:
                with contextlib.suppress(Exception):
                    engine.core_mgr.abort_request(seq.id)
            engine.io_processor.requests.pop(seq.id, None)

    # 一次性把全部输出token解码成文本（流式路径则是逐步增量解码）
    text = delivered_text(all_token_ids)
    # seq不会是None（preprocess失败会直接抛异常，走不到这里）；else分支是兜底
    num_tokens_input = (
        seq.num_prompt_tokens
        if seq is not None
        else len(
            prompt_or_tokens
            if isinstance(prompt_or_tokens, list)
            else tokenizer.encode(prompt_or_tokens)
        )
    )
    num_tokens_output = len(all_token_ids)
    # 时间统计（秒）：latency=总耗时；TTFT=从进入本函数到首个token；TPOT=首token之后平均每个token的间隔
    # 例：started_at=0.0，首token在0.5，末token在2.5，共输出21个token → TTFT=0.5，TPOT=(2.5-0.5)/20=0.1
    finished_at = time.time()
    latency = finished_at - started_at
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    # 结果dict，交给build_chat_response / build_completion_response等组装成OpenAI格式的响应
    response = {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": num_tokens_input,
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": latency,
        "num_cached_tokens": num_cached_tokens_seen,
    }
    if kv_transfer_output_meta_info is not None:
        response["kv_transfer_output_meta_info"] = kv_transfer_output_meta_info
    if return_token_ids and seq is not None:
        # Convert the token array to a JSON-serializable list.
        # 【译】把token数组转换成可JSON序列化的list。
        response["prompt_token_ids"] = list(seq.prompt_token_ids)
    yield response


# 多模态版的generate_async（B级）：输入已经是token id，另带multimodal_data（像素值等）；
#   与generate_async的差别：不支持kv_transfer_params，结果里没有num_cached_tokens，也不回显prompt token id
async def generate_async_multimodal(
    token_ids: list[int],
    multimodal_data: dict[str, Any],
    sampling_params: SamplingParams,
    request_id: str,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate text asynchronously for one multimodal request."""
    # 【译】为一个多模态请求异步生成文本。
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    first_token_at: float | None = None
    last_token_at: float | None = None
    all_token_ids = new_token_ids()
    finish_reason: str | None = None
    seq = None

    def completion_callback(request_output: RequestOutput):
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    def do_preprocess():
        return engine.io_processor.preprocess(
            token_ids,
            sampling_params,
            stream_callback=completion_callback,
            multimodal_data=multimodal_data,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    seq = await loop.run_in_executor(None, do_preprocess)
    try:
        _validate_sequence_context_length(seq)
    except Exception:
        engine.io_processor.requests.pop(seq.id, None)
        raise
    engine.core_mgr.add_request([seq])

    _finished_ok = False
    try:
        while True:
            item = await token_queue.get()
            token_ids_out = item.get("token_ids") or []
            if token_ids_out:
                if first_token_at is None:
                    first_token_at = item.get("ts", time.time())
                    record_nonstream_first_token()
                last_token_at = item.get("ts", time.time())
                all_token_ids.extend(token_ids_out)
            if item.get("finished", False):
                finish_reason = item.get("finish_reason")
                _finished_ok = True
                break
    finally:
        # See generate_async: abort on early exit, always pop to avoid leak.
        # 【译】见 generate_async：提前退出时abort，总是pop以避免泄漏。
        if seq is not None:
            if not _finished_ok:
                with contextlib.suppress(Exception):
                    engine.core_mgr.abort_request(seq.id)
            engine.io_processor.requests.pop(seq.id, None)

    text = delivered_text(all_token_ids)
    num_tokens_output = len(all_token_ids)
    finished_at = time.time()
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    yield {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": (
            seq.num_prompt_tokens if seq is not None else len(token_ids)
        ),
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": finished_at - started_at,
    }


# n>1的非流式路径（B级）：一个prompt扇出n个兄弟Sequence，全部完成后返回n个结果dict
#   它是普通协程（return列表），不是异步生成器，所以调用方直接用_race_disconnect包装
#   与generate_async的差别：所有兄弟共用一个队列，回调推(兄弟序号, item)；结果里没有num_cached_tokens和KV传输元数据
async def generate_async_fanout(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
    return_token_ids: bool = False,
) -> list[dict[str, Any]]:
    """Non-streaming n>1 path: fan out N siblings and await all of them.

    Returns a list of per-sibling output dicts in the same shape as
    :func:`generate_async` yields for n==1, so response builders can treat
    each entry the same way.

    Shared prompt token IDs are returned only in the first output.
    """
    # 【译】非流式 n>1 路径：扇出N个兄弟序列，等它们全部完成。
    #
    #   返回每个兄弟一个输出dict的列表，形状与 generate_async 在 n==1 时yield的相同，这样响应构造函数可以同样对待每一项。
    #
    #   共享的prompt token ID只在第一个输出里返回。

    n = int(sampling_params.n)
    assert n >= 1

    shared_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    per_tokens = [new_token_ids() for _ in range(n)]
    per_first_token_at: list[float | None] = [None] * n
    per_last_token_at: list[float | None] = [None] * n
    per_finish_reason: list[str | None] = [None] * n
    finished = [False] * n

    # 用工厂函数给每个兄弟生成回调：若直接在循环里定义闭包，idx会按"晚绑定"规则全都取到循环结束时的值
    def make_callback(idx: int):
        def _cb(request_output: RequestOutput) -> None:
            now = time.time()
            loop.call_soon_threadsafe(
                shared_queue.put_nowait,
                (
                    idx,
                    {
                        "token_ids": request_output.output_tokens,
                        "finished": request_output.finished,
                        "finish_reason": request_output.finish_reason,
                        "ts": now,
                    },
                ),
            )

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    # preprocess_fanout：只tokenize一次，构造n个兄弟Sequence（各自独立采样噪声），见第2批llm_engine.py
    def do_preprocess():
        return engine.io_processor.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )

    seqs = await loop.run_in_executor(None, do_preprocess)
    try:
        # 兄弟序列的prompt和max_tokens都相同，校验第一个即可
        _validate_sequence_context_length(seqs[0])
    except Exception:
        for seq in seqs:
            engine.io_processor.requests.pop(seq.id, None)
        raise
    engine.core_mgr.add_request(seqs)
    num_tokens_input = seqs[0].num_prompt_tokens

    _all_finished = False
    try:
        while not all(finished):
            idx, item = await shared_queue.get()
            # 已经结束的兄弟若再收到输出就忽略
            if finished[idx]:
                continue
            tokens = item.get("token_ids") or []
            if tokens:
                if per_first_token_at[idx] is None:
                    per_first_token_at[idx] = item.get("ts", time.time())
                    record_nonstream_first_token()
                per_last_token_at[idx] = item.get("ts", time.time())
                per_tokens[idx].extend(tokens)
            if item.get("finished", False):
                per_finish_reason[idx] = item.get("finish_reason")
                finished[idx] = True
        _all_finished = True
    finally:
        # Abort any sibling still running on early exit; always pop all seqs.
        # 【译】提前退出时abort所有仍在运行的兄弟序列；总是pop掉所有序列。
        for _seq in seqs:
            if not _all_finished:
                try:
                    engine.core_mgr.abort_request(_seq.id)
                except Exception:
                    logger.debug("abort_request failed on teardown", exc_info=True)
            engine.io_processor.requests.pop(_seq.id, None)

    finished_at = time.time()
    outputs: list[dict[str, Any]] = []
    for i in range(n):
        num_tokens_output = len(per_tokens[i])
        ttft = (
            per_first_token_at[i] - started_at
            if per_first_token_at[i] is not None
            else 0.0
        )
        tpot = (
            (per_last_token_at[i] - per_first_token_at[i]) / (num_tokens_output - 1)
            if per_first_token_at[i] is not None
            and per_last_token_at[i] is not None
            and num_tokens_output > 1
            else 0.0
        )
        outputs.append(
            {
                "text": delivered_text(per_tokens[i]),
                "token_ids": per_tokens[i],
                "finish_reason": per_finish_reason[i],
                "num_tokens_input": num_tokens_input,
                "num_tokens_output": num_tokens_output,
                "ttft": ttft,
                "tpot": tpot,
                "latency": finished_at - started_at,
            }
        )
    if return_token_ids and seqs:
        outputs[0]["prompt_token_ids"] = list(seqs[0].prompt_token_ids)
    return outputs


# 请求的model字段必须和服务的模型名一致（忽略末尾的"/"）；不传model则不检查
#   不一致时直接抛HTTPException(400)，FastAPI把它变成 {"detail": ...} 响应
def validate_model(requested_model: str | None) -> None:
    """Validate that the requested model matches the server's model."""
    # 【译】校验请求的模型名与服务器的模型名一致。
    if requested_model is None:
        return

    normalized_requested = requested_model.rstrip("/")
    normalized_served = model_name.rstrip("/")
    if normalized_requested != normalized_served:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{requested_model}' does not match "
            f"server model '{model_name}'",
        )


# return_token_ids只能用于非流式：流式响应没有"响应体"可以放prompt token id
def _validate_return_token_ids(
    return_token_ids: bool | None, stream: bool | None
) -> None:
    """Prompt token IDs can be returned only in non-streaming responses."""
    # 【译】prompt token ID只能在非流式响应里返回。
    if not return_token_ids:
        return
    if stream:
        raise ValueError(
            "return_token_ids is not supported with stream=true; the ids are "
            "returned on the response body, which a stream does not have"
        )


# 【分支】PD分离（第9批）：路由器可能在kv_transfer_params里带prompt_token_ids（vLLM兼容的写法）；
#   这些ID已经作为prompt交给引擎，就从元数据里去掉，不重复传
def _engine_kv_transfer_params(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Copy KV metadata without IDs already passed as the engine prompt."""
    # 【译】复制KV元数据，去掉已经作为引擎prompt传入的ID。
    if kv_transfer_params is None or "prompt_token_ids" not in kv_transfer_params:
        return kv_transfer_params
    return {k: v for k, v in kv_transfer_params.items() if k != "prompt_token_ids"}


# /v1/messages和/v1/responses在prompt过长时会截断（chat / completions则直接拒绝），这里算截断时给输出预留多少token
# 例：max_ctx=32768 → headroom=max(1024, 32768//8)=4096；无PD交接、max_tokens=2000时取min → 2000
# 注意：截断后的序列照样要过_validate_sequence_context_length，而预留量最多只有max(1024, max_ctx//8)。
#   max_tokens比它大时仍会被拒绝：max_ctx=32768、max_tokens=8000 → 预留4096，prompt截到约28672个token，28672 + 8000 > 32768 → 400
def _prompt_headroom(
    max_tokens: int, max_ctx: int, kv_transfer_params: dict[str, Any] | None
) -> int:
    """Output tokens to reserve when an over-long prompt is truncated.

    A PD router sends the prefill leg ``max_tokens=1`` and the decode leg the
    client's value. With a KV handoff the reservation therefore cannot depend
    on ``max_tokens``, or the two legs truncate to different prompts and the
    transferred blocks no longer match the decode-side sequence.
    """
    # 【译】截断过长prompt时要为输出预留的token数。
    #
    #   PD路由器给prefill那一路发 max_tokens=1，给decode那一路发客户端的值。因此有KV交接时，预留量不能依赖 max_tokens，
    #   否则两路会截断成不同的prompt，传过去的block就和decode侧的序列对不上了。
    headroom = max(1024, max_ctx // 8)
    if kv_transfer_params is None:
        headroom = min(max_tokens, headroom)
    return headroom


# 【分支】PD分离：把prefill节点返回的KV位置放进响应，给PD路由器转交decode节点（第9批）
def _with_kv_transfer_params(response: dict, final_output: dict) -> dict:
    """Return the prefill's KV location to the PD router, as chat does."""
    # 【译】像chat接口那样，把prefill的KV位置返回给PD路由器。
    if "kv_transfer_output_meta_info" in final_output:
        response["kv_transfer_params"] = final_output["kv_transfer_output_meta_info"]
    return response


# =====================================================================================
# setup_streaming_request：流式、n=1 的主路径（对应文件头图1的⑥~⑨）
# =====================================================================================
# 只负责"把请求交给引擎"，不等待任何输出：返回后，handler立刻用返回的collector构造SSE生成器
# 步骤：
#   1. 建StreamOutputCollector（本请求的收件箱）和StreamState（反分词器 + 计时），登记事件循环和开始时间
#   2. 定义stream_callback闭包：在输出线程上把chunk交给批量投递器
#   3. 线程池里preprocess（tokenize + 构造Sequence + 挂上回调），登记 seq.id → request_id
#   4. 长度校验；失败则撤销所有登记、把异常抛给handler（变成400）
#   5. add_request交给引擎，返回(seq_id, collector, prompt长度)
# 对比非流式的generate_async：那里每个回调直接call_soon_threadsafe一次，这里回调只缓冲、每step批量投递一次
# 原理：为什么要"先把请求交给引擎、再返回StreamingResponse"
#   StreamingResponse一开始迭代就会先发出 "200 + Content-Type: text/event-stream" 的响应头，之后状态码再也改不了。
#   所以一切可能失败、应当返回4xx的检查（模板渲染、tokenize、长度校验）都必须在这之前完成——本函数被handler直接await，
#   失败的异常才能在handler里变成400。流开始之后的错误只能放进流里告诉客户端（Anthropic路径的stream_failure_frames），
#   或者直接断流：chat / completions的流式生成器只有try/finally没有except，异常会让响应在没有[DONE]的情况下被截断
async def setup_streaming_request(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> tuple[int, StreamOutputCollector, int]:
    """Set up a streaming request with the engine.

    Returns ``(seq_id, stream_collector, num_prompt_tokens)``.
    ``num_prompt_tokens`` is the engine-computed prompt length so the stream
    response generator does not have to re-tokenize the prompt on the event
    loop.
    """
    # 【译】在引擎中建立一个流式请求。
    #
    #   返回 (seq_id, stream_collector, num_prompt_tokens)。num_prompt_tokens 是引擎算出的prompt长度，
    #   这样流式响应生成器不必在事件循环上重新tokenize一遍prompt。
    # collector：输出线程那边经批量投递放入chunk，事件循环这边的SSE生成器await它取出
    stream_collector = StreamOutputCollector(request_id)
    stream_loop = asyncio.get_running_loop()
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    # The detokenizer lives in this closure, so it is freed when the engine
    # drops the callback on the stream's last chunk -- no registry, no cleanup.
    # 【译】反分词器活在这个闭包里，所以引擎在这个流的最后一个chunk上丢掉回调时，它就被释放了——不需要注册表，也不需要清理。
    # 解释：变量名叫detokenizer，实际是StreamState（含反分词器和投递计时）；"引擎丢掉回调"指CoreManager在finished输出后
    #   从_seq_id_to_callback里pop掉它（还没被读走的chunk仍持有这个状态，读完后才可回收）
    assert _stream_batch_dispatcher is not None
    detokenizer = _stream_batch_dispatcher.new_state()

    # 在输出线程上执行：只缓冲，不做反分词、不碰asyncio对象
    def stream_callback(request_output: RequestOutput) -> None:
        _send_stream_chunk_direct(
            request_output, request_id, stream_collector, stream_loop, detokenizer
        )

    # 在协程里调用时，get_event_loop()返回的就是正在运行的事件循环（与上面的stream_loop是同一个）
    executor_loop = asyncio.get_event_loop()

    # 线程池里执行：tokenize + 构造Sequence（stream_callback挂在Sequence上，add_request时被CoreManager取下登记）
    def do_preprocess():
        seq = engine.io_processor.preprocess(
            prompt_or_tokens,
            sampling_params,
            stream_callback=stream_callback,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )
        _seq_id_to_request_id[seq.id] = request_id
        return seq

    seq = None
    try:
        seq = await executor_loop.run_in_executor(None, do_preprocess)
        _validate_sequence_context_length(seq)
    except Exception:
        # 预处理或校验失败：撤销本请求所有的登记，异常抛给handler（ValueError → 400）
        _stream_loops.pop(request_id, None)
        _request_start_times.pop(request_id, None)
        if seq is not None:
            _seq_id_to_request_id.pop(seq.id, None)
            engine.io_processor.requests.pop(seq.id, None)
        raise
    seq_id = seq.id

    # debug, not info: this runs once per request, on the event loop, and
    # logging takes a lock the engine's output threads are also contending for.
    # A loop-stall watchdog at concurrency 8192 caught 26 stalls over a run and
    # 9 of them were sitting on this line, up to 3.3 s each -- long enough that
    # the server accepts no new request at all and the GPUs run dry waiting for
    # work. Anything per-request logged from here has to stay off info.
    # %-style, not an f-string: the arguments are formatted only if the
    # record is emitted, and this runs once per request with debug off.
    # 【译】用debug而不是info：这行每个请求在事件循环上执行一次，而logging要拿一把锁，引擎的输出线程也在争这把锁。
    #   在并发8192下，一个事件循环卡顿看门狗在一次运行中抓到26次卡顿，其中9次就卡在这一行，每次最长3.3秒——
    #   长到服务器完全不接受新请求、GPU因为等不到活而空转。从这里打的任何逐请求日志都不能用info级别。
    #   用%风格而不是f-string：只有日志记录真被输出时才会格式化参数，而在关掉debug时这行每个请求都要执行一次。
    logger.debug("API: Created request_id=%s, seq_id=%s", request_id, seq_id)
    # 交给引擎；从这一刻起，输出线程随时可能开始往collector投递chunk
    engine.core_mgr.add_request([seq])

    return seq_id, stream_collector, seq.num_prompt_tokens


# 流结束时的清理（单个流 = 单个Sequence）。调用方：各流式生成器的finally（serving_chat.stream_chat_response等），
#   以回调参数的形式传过去，避免serving_chat反过来import api_server
# 步骤：删seq_id映射 → 若中途中断则abort → 从io_processor.requests删掉Sequence
def cleanup_stream(seq_id: int, aborted: bool = False) -> None:
    """Tear down one stream. A fan-out request runs this once per sibling.

    ``aborted`` says the stream did NOT reach its normal end (client disconnect
    or abnormal generator teardown), so the seq is likely still running in the
    engine and must be stopped. On normal completion pass False (the default):
    the engine has already dropped the seq, so an abort would be a guaranteed
    no-op that just floods the control path (one broadcast per engine core, per
    request).
    """
    # 【译】拆除一个流。扇出请求对每个兄弟序列各执行一次。
    #
    #   aborted 表示这个流"没有"走到正常结尾（客户端断开或生成器异常销毁），所以这条序列很可能还在引擎里运行，必须停掉。
    #   正常结束时传False（默认值）：引擎已经丢掉了这条序列，再abort必然是空操作，只会冲击控制通道
    #   （每个请求要向每个engine core广播一次）。
    _seq_id_to_request_id.pop(seq_id, None)
    if aborted:
        # abort_request：向所有DP rank广播"abort_request"工具RPC（utility command），持有该序列的调度器在下一个step
        #   把它按正常停止路径结束并释放KV block（第2批engine_core_mgr.py、第3批scheduler.py）
        try:
            engine.core_mgr.abort_request(seq_id)
        except Exception:
            logger.debug("abort_request failed on teardown", exc_info=True)
    # 流式请求的Sequence在这里从在途字典中删除（非流式在generate_async的finally里删）
    engine.io_processor.requests.pop(seq_id, None)


# 请求级（而非序列级）的清理：删掉按request_id登记的事件循环和开始时间
def cleanup_request(request_id: str) -> None:
    """Tear down what a request owns beyond its individual streams.

    Runs once, after every one of the request's streams has been cleaned up.
    Separate from :func:`cleanup_stream` because a fan-out has n streams but
    one request: folding both into one call meant these two pops ran n times,
    n-1 of them no-ops, and made a caller pass a seq id and a request id
    together when each half only needs one of them.
    """
    # 【译】拆除一个请求在各个流之外拥有的东西。
    #
    #   在该请求的所有流都清理完之后执行一次。与 cleanup_stream 分开，是因为扇出请求有n个流却只有一个请求：
    #   把两者合成一个调用，意味着这两个pop要执行n次（其中n-1次是空操作），还迫使调用方同时传seq id和request id，
    #   而每一半其实只需要其中一个。
    _stream_loops.pop(request_id, None)
    _request_start_times.pop(request_id, None)


# 以下三个函数处理"非流式请求的客户端中途断开"（流式请求的断开由StreamingResponse处理，见各流式生成器的finally）
# 异常类：_race_disconnect检测到断开时抛出，接口函数捕获后返回499
class _ClientDisconnected(Exception):
    """Raised when a non-streaming client hangs up mid-generation."""

    # 【译】非流式客户端在生成中途挂断时抛出。

    def __init__(self, request_id: str):
        super().__init__(request_id)
        self.request_id = request_id


# ASGI（Python异步Web的接口规范）里，request.receive()返回下一个来自客户端的消息：
#   请求体分片是"http.request"，客户端断开时收到"http.disconnect"。请求体早已读完，所以这里通常一直挂起，直到断开
async def _listen_for_disconnect(request) -> None:
    """Block until the client sends an ``http.disconnect`` ASGI event.

    Unlike polling ``request.is_disconnected()`` on a timer, this awaits the
    disconnect event directly, so detection is immediate and costs nothing while
    the client stays connected.
    """
    # 【译】阻塞直到客户端发来 http.disconnect 这个ASGI事件。
    #
    #   与定时轮询 request.is_disconnected() 不同，这里直接await断开事件，所以检测是即时的，客户端保持连接时没有任何开销。
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


# 让"生成结果"和"客户端断开"赛跑，谁先完成就以谁为准。时间线举例：
#   t=0   handler_task开始等引擎；disconnect_task等ASGI消息
#   t=2s  客户端断开 → disconnect_task先完成 → 取消handler_task
#         → generate_async里的 await token_queue.get() 抛CancelledError → 它的finally：abort_request + pop
#   随后本函数抛_ClientDisconnected → 接口返回499
async def _race_disconnect(coro, raw_request, request_id):
    """Race an awaitable against client disconnect (vLLM ``with_cancellation``
    style).

    Starlette does NOT cancel a *non-streaming* request handler when the client
    goes away (unlike StreamingResponse, which is cancelled on http.disconnect).
    Without this, an abandoned non-stream request keeps ``await``-ing the engine
    until it hits ``max_tokens`` -- burning GPU on output nobody will read AND
    leaking the seq(s) in ``io_processor.requests`` (their finally never fires).

    We run ``coro`` (which produces the final result) as a task alongside a task
    that awaits the ASGI ``http.disconnect`` event. Whichever finishes first
    wins; the loser is cancelled. On disconnect, the coro's cancellation
    propagates into its ``await`` points so its own ``try/finally`` runs ->
    ``abort_request`` + ``io_processor.requests.pop`` (for fan-out, this aborts
    every sibling). We then raise ``_ClientDisconnected``.

    ``request.receive()`` is safe here because FastAPI has already parsed the
    request body into a pydantic model before this handler runs, so there is no
    unread body for ``receive()`` to race against.
    """
    # 【译】让一个awaitable与客户端断开赛跑（vLLM with_cancellation 的风格）。
    #
    #   客户端离开时，Starlette"不会"取消"非流式"请求的handler（不同于StreamingResponse，后者在收到 http.disconnect 时会被取消）。
    #   没有这个函数，一个被放弃的非流式请求会一直await引擎直到达到 max_tokens——既为没人读的输出烧GPU，
    #   又在 io_processor.requests 里泄漏序列（它们的finally永远不会执行）。
    #
    #   我们把 coro（它产出最终结果）作为一个task运行，旁边再起一个等待ASGI http.disconnect 事件的task。
    #   先完成的获胜，输的那个被取消。断开时，coro的取消会传播到它的各个await点，于是它自己的 try/finally 得以执行 ->
    #   abort_request + io_processor.requests.pop（扇出时会abort每个兄弟序列）。然后我们抛出 _ClientDisconnected。
    #
    #   这里调用 request.receive() 是安全的，因为在这个handler运行之前FastAPI已经把请求体解析成了pydantic模型，
    #   所以没有未读的请求体会与 receive() 竞争。
    # asyncio.ensure_future(coro)：把协程包装成Task，交给事件循环调度（Task会在后台独立运行）
    handler_task = asyncio.ensure_future(coro)

    # No ASGI request object (e.g. internal call) -> just await the coro.
    # 【译】没有ASGI请求对象（例如内部调用）-> 直接await这个coro。
    if raw_request is None:
        return await handler_task

    disconnect_task = asyncio.ensure_future(_listen_for_disconnect(raw_request))

    # asyncio.wait(..., FIRST_COMPLETED)：任意一个task完成就返回 (done集合, pending集合)
    done, pending = await asyncio.wait(
        [handler_task, disconnect_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the loser and let its cancellation settle (drives the coro's own
    # finally -> abort_request when the handler is the loser). Only swallow the
    # expected CancelledError; log anything else, and let BaseException
    # (KeyboardInterrupt/SystemExit) propagate.
    # 【译】取消输的一方，并等它的取消处理完（handler输了时，这会驱动coro自己的finally -> abort_request）。
    #   只吞掉预期中的CancelledError；其他异常记日志；BaseException（KeyboardInterrupt/SystemExit）让它继续传播。
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                f"Error tearing down cancelled task for request {request_id}",
                exc_info=True,
            )

    # 生成先完成：返回结果（handler_task.result()在任务抛了异常时会重新抛出，例如校验失败的ValueError）
    if handler_task in done:
        return handler_task.result()

    # 断开先发生：生成任务已被取消并清理完毕
    logger.info(f"Client disconnected (non-stream), aborting request {request_id}")
    raise _ClientDisconnected(request_id)


# 驱动generate_async*这类"只yield一次"的异步生成器：取最后一次yield的值，同时监视断开
async def _run_nonstream_with_disconnect(agen, raw_request, request_id):
    """Drive a non-stream ``generate_async*`` async-*generator* while watching
    for client disconnect.

    Thin wrapper over :func:`_race_disconnect` that collects the generator's
    last yielded output. Use :func:`_race_disconnect` directly for the fan-out
    path, whose ``generate_async_fanout`` is a coroutine returning a list.
    """
    # 【译】驱动一个非流式的 generate_async* 异步"生成器"，同时监视客户端断开。
    #
    #   _race_disconnect 的薄包装，收集生成器最后一次yield的输出。扇出路径直接用 _race_disconnect，
    #   因为它的 generate_async_fanout 是一个返回list的协程。

    async def _collect():
        final_output = None
        async for output in agen:
            final_output = output
        return final_output

    return await _race_disconnect(_collect(), raw_request, request_id)


# n>1的流式路径（B级）：与setup_streaming_request相同，只是preprocess_fanout建n个兄弟Sequence，
#   共用一个collector，每个兄弟一个回调（各自的反分词状态），chunk带tag=兄弟序号
async def setup_streaming_request_fanout(
    prompt_or_tokens: str | list[int],
    sampling_params: SamplingParams,
    request_id: str,
    kv_transfer_params: dict[str, Any] | None = None,
    multimodal_data: dict[str, Any] | None = None,
    data_parallel_rank: int | None = None,
    dp_session_id: str | None = None,
    dp_parent_session_id: str | None = None,
) -> tuple[list[int], StreamOutputCollector, int]:
    """Fan-out variant of :func:`setup_streaming_request`.

    Creates ``sampling_params.n`` sibling sequences sharing one output
    collector. Every callback pushes ``(sibling_index, chunk_data)`` tuples so
    the merge-stream consumer can rewrite ``choices[0].index`` correctly.

    Returns ``(seq_ids, shared_collector, num_prompt_tokens)``. All siblings
    tokenize the same prompt once, so ``num_prompt_tokens`` is shared and lets
    the stream response generator skip re-tokenizing on the event loop.
    """
    # 【译】setup_streaming_request 的扇出变体。
    #
    #   创建 sampling_params.n 个共享同一个输出collector的兄弟序列。每个回调推送 (sibling_index, chunk_data) 元组，
    #   让合并流的消费者能正确改写 choices[0].index。
    #
    #   返回 (seq_ids, shared_collector, num_prompt_tokens)。所有兄弟只对同一个prompt tokenize一次，
    #   所以 num_prompt_tokens 是共享的，流式响应生成器因此可以跳过在事件循环上重新tokenize。
    n = int(sampling_params.n)
    assert n >= 1

    shared_collector = StreamOutputCollector(request_id)
    stream_loop = asyncio.get_running_loop()
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    assert _stream_batch_dispatcher is not None

    def make_callback(idx: int):
        # One detokenizer per sibling, held by the closure that feeds it.
        # 【译】每个兄弟一个反分词器，由给它喂数据的闭包持有。
        detokenizer = _stream_batch_dispatcher.new_state()

        def _cb(request_output: RequestOutput) -> None:
            _send_stream_chunk_tagged(
                request_output,
                request_id,
                idx,
                shared_collector,
                stream_loop,
                detokenizer,
            )

        return _cb

    stream_callbacks = [make_callback(i) for i in range(n)]

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seqs = engine.io_processor.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callbacks=stream_callbacks,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )
        for seq in seqs:
            _seq_id_to_request_id[seq.id] = request_id
        return seqs

    seqs = []
    try:
        seqs = await executor_loop.run_in_executor(None, do_preprocess)
        _validate_sequence_context_length(seqs[0])
    except Exception:
        _stream_loops.pop(request_id, None)
        _request_start_times.pop(request_id, None)
        for seq in seqs:
            _seq_id_to_request_id.pop(seq.id, None)
            engine.io_processor.requests.pop(seq.id, None)
        raise
    seq_ids = [seq.id for seq in seqs]
    # debug for the same reason as its single-sequence counterpart: per-request
    # logging on the event loop stalls it under load.
    # 【译】用debug的理由与单序列版本相同：在事件循环上逐请求打日志，高负载时会让它卡顿。
    # 解释：但这里用的是f-string，参数在调用logger.debug之前就已格式化（单序列版本特意用%风格来避免这一点）
    logger.debug(
        f"API: Created fan-out request_id={request_id}, n={n}, seq_ids={seq_ids}"
    )
    engine.core_mgr.add_request(seqs)
    return seq_ids, shared_collector, seqs[0].num_prompt_tokens


# ============================================================================
# FastAPI Application
# ============================================================================
# 【译】FastAPI应用


# 刷新一次Prometheus指标：engine.get_metrics_statistics()读的是CoreManager输出线程收到的METRICS快照
#   （EngineCore定期推送），纯本地读取，所以直接在事件循环上执行。【分支】指标见第10批
async def _refresh_metrics_once() -> None:
    """Take one snapshot from the engine and publish it.

    Owns its own accounting: `metrics_refresh_errors` means "the exported
    series are stale", which is this function not completing and nothing else.
    """
    # 【译】从引擎取一份快照并发布。
    #
    #   自己负责计数：metrics_refresh_errors 的含义是"导出的序列已经过时"，也就是这个函数没有完成，别无他意。
    if engine is None:
        return
    try:
        # A local read of the snapshots EngineCore pushes, so it runs inline on
        # the loop -- no executor thread, and no writer on the control socket.
        # 【译】这是对EngineCore推送来的快照的本地读取，所以直接在事件循环上执行——不需要线程池，也不会往控制socket写东西。
        _metrics_exporter.update(engine.get_metrics_statistics())
    except Exception:
        _metrics_exporter.record_refresh_error()
        logger.warning("Failed to refresh Prometheus metrics", exc_info=True)


async def _reclaim_watch_once() -> None:
    """Check whether this process's collector has started finding cycles."""
    # 【译】检查本进程的垃圾回收器是否开始发现循环引用。
    reclaim_watch("api_server")


# 后台周期任务的通用外壳：每隔interval秒执行一次step，step抛的异常记日志后继续，任务不会因此悄悄停掉
async def _periodic(interval: float, step: Callable[[], Awaitable[None]]) -> None:
    """Run `step` on a fixed cadence for the life of the process.

    The guard is the point, and there is one copy of it because that is the
    invariant a second copy loses. Nothing awaits these tasks until shutdown,
    so an exception escaping here is silent: asyncio records "never retrieved"
    and the job stops, leaving whatever it maintained frozen at its last value
    and still being served.

    `CancelledError` needs no clause -- it is a BaseException, so shutdown
    stops these regardless.
    """
    # 【译】在进程的整个生命周期里按固定节奏运行 step。
    #
    #   关键在于这层保护，而且只有一份，因为第二份拷贝正是会丢掉这个不变式的地方。在关闭之前没有人await这些task，
    #   所以从这里逃出去的异常是无声的：asyncio只记一句"never retrieved"，任务就停了，它维护的东西冻结在最后的值上，并继续被对外提供。
    #
    #   CancelledError 不需要单独的except子句——它是 BaseException，所以关闭时无论如何都能停掉这些task。
    while True:
        await asyncio.sleep(interval)
        try:
            await step()
        except Exception:
            logger.warning("%s failed", step.__name__, exc_info=True)


# lifespan：FastAPI的启动/关闭钩子。@asynccontextmanager把一个含单个yield的异步生成器变成异步上下文管理器：
#   yield之前的代码在uvicorn启动应用时执行（此时main()已经建好引擎，还没接收任何请求），
#   yield之后（finally里）的代码在服务关闭时执行
# 启动：调GC参数 → 首次刷新指标 → 冻结启动期的堆（gc.freeze）→ 起后台周期任务
# 关闭：取消后台任务 → 等指标渲染结束 → engine.close()（关闭EngineCore等子进程）
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # 【译】负责启动和关闭的lifespan上下文管理器。
    metrics_interval = envs.ATOM_METRICS_UPDATE_INTERVAL_S
    logger.info("Server started successfully and ready to accept requests")
    # tune_gc / maybe_attach_gc_debug_callback：调整本进程的垃圾回收阈值、按需挂上GC调试回调（utils/gc_utils.py）
    tune_gc()
    maybe_attach_gc_debug_callback("api_server")
    await _refresh_metrics_once()
    # The engine was built in `main()`, so this is the last point before the
    # first request at which everything reachable is still startup state.
    # 【译】引擎在 main() 里就建好了，所以这是第一个请求到来之前、一切可达对象都还是启动期状态的最后时刻。
    # freeze_gc_heap：先gc.collect()，再gc.freeze()把当前存活的对象全部移进永久代，之后的GC不再扫描它们
    #   （ATOM_GC_FREEZE=1时生效，默认开启）；服务期间GC只需扫描请求产生的新对象
    freeze_gc_heap("api_server")
    # After the freeze, or the baseline carries startup's own collections.
    # 【译】放在freeze之后，否则基线里会带上启动期自己的回收记录。
    arm_reclaim_watch()
    # _background_tasks[:] = [...]：原地替换列表内容（模块级列表对象不变，所以不需要global声明）
    # asyncio.create_task：把协程交给事件循环在后台运行；这里起两个周期任务：刷新指标、GC监视
    _background_tasks[:] = [
        asyncio.create_task(_periodic(interval, step), name=step.__name__)
        for interval, step in (
            (metrics_interval, _refresh_metrics_once),
            (_GC_WATCH_INTERVAL_SECONDS, _reclaim_watch_once),
        )
    ]
    # yield：把控制权交给uvicorn，服务开始接收请求，直到关闭时才回到这里
    try:
        yield
    finally:
        for task in _background_tasks:
            task.cancel()
        # `gather` rather than suppressing `CancelledError` around each await:
        # that swallows a cancellation aimed at shutdown itself, which makes
        # shutdown the thing that cannot be stopped.
        # 【译】用 gather 而不是在每个await外面压制 CancelledError：后者会吞掉针对关闭过程本身的取消，让关闭变成无法被停止的事情。
        await asyncio.gather(*_background_tasks, return_exceptions=True)
        _background_tasks.clear()
        await _metrics_exporter.wait_for_render()
        logger.info("Server shutting down, releasing resources...")
        # 关闭引擎：由CoreManager关闭EngineCore等子进程（第2批llm_engine.py / engine_core_mgr.py的close）
        if engine is not None:
            engine.close()


# 创建FastAPI应用，挂上lifespan；下面的@app.post / @app.get装饰器把各接口注册到这个app上
# RequestTimingMiddleware（request_timing.py）：ASGI中间件，在请求进入时开始计时，配合_client_stream /
#   record_nonstream_first_token把TTFT上报到_request_metrics
app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)
app.add_middleware(
    RequestTimingMiddleware,
    observe_ttft=_request_metrics.observe_time_to_first_token,
)


# ---- Error handlers ----
# 【译】---- 错误处理器 ----
# exception_handler：接口函数抛出未被捕获的指定类型异常时，FastAPI调用这里生成响应
#   注意：chat_completions / completions自己捕获ValueError并转成HTTPException(400)，
#   而HTTPException由FastAPI内置的处理器返回 {"detail": "..."}；这里的处理器兜住其余漏出来的异常


# 漏出来的ValueError → 400，OpenAI风格的错误体（responses_create也直接调用它）
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": 400,
            }
        },
    )


# 其他任何未捕获的异常 → 500，并记录完整的traceback
@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_server_error",
                "code": 500,
            }
        },
    )


# ---- Endpoints ----
# 【译】---- 接口 ----


# =====================================================================================
# POST /v1/chat/completions：OpenAI chat接口（主路径，流程见文件头图1）
# =====================================================================================
# 参数：request是FastAPI按pydantic模型解析校验好的请求体（protocol.ChatCompletionRequest）；
#   raw_request是底层的Starlette Request，用来读HTTP头（DP会话亲和）和监听客户端断开
# 返回：流式时是StreamingResponse（SSE）；非流式时是JSONResponse（ChatCompletionResponse的JSON）
# 一个具体的请求：
#   {"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "你好"}], "stream": true, "max_tokens": 128}
#   → 渲染成 "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" 这样的prompt（具体取决于模型的chat template）
#   → 构造SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, max_tokens=128, n=1) → setup_streaming_request → SSE
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests (OpenAI-compatible)."""
    # 【译】处理chat completion请求（OpenAI兼容）。

    # 放在try之外：模型名不符直接抛HTTPException(400)
    validate_model(request.model)

    try:
        # 兼容Anthropic风格的工具定义（name + input_schema），转成OpenAI的function工具（serving_chat.py）
        request.tools = normalize_chat_tools(request.tools)
        # 校验tools / tool_choice / response_format的形状，非法时抛ValueError → 400
        validate_chat_request(request)
        # 取消息列表：messages字段，或兼容的prompt字段
        messages = request.get_messages()

        # 合并chat template参数，后写的覆盖先写的：服务器默认值（--default-chat-template-kwargs）< 请求的chat_template_kwargs
        #   < 下面单独转发的response_format / tool_choice / 思考开关
        merged_kwargs = dict(default_chat_template_kwargs)
        if request.chat_template_kwargs:
            merged_kwargs.update(request.chat_template_kwargs)
        # Forward K3 template controls the chat template needs but that pydantic
        # does not otherwise thread through: structured-output response_format,
        # a string tool_choice ("auto"/"none"/"required"), and thinking/effort.
        # 【译】转发K3模板需要、而pydantic本身不会传下去的模板控制项：结构化输出的 response_format、
        #   字符串形式的 tool_choice（"auto"/"none"/"required"），以及 thinking/effort。
        # 解释：K3指Kimi-K3（它用模型目录里自带的Python消息编码器而不是Jinja模板，见chat_encoders.py）。
        #   这些键对所有模型都会写进merged_kwargs。Jinja模板会忽略不认识的kwarg；
        #   模型自带的Python encoder则由MessageEncoderAdapter按函数签名过滤（chat_encoder_adapters.py）
        if request.response_format is not None:
            merged_kwargs["response_format"] = request.response_format
        if isinstance(request.tool_choice, str):
            merged_kwargs["tool_choice"] = request.tool_choice
        # 从请求的thinking / reasoning_effort解析出(是否开启推理, 推理力度)；没表态时enabled为None
        _th_enabled, _th_effort = resolve_thinking(request)
        if request.thinking is not None or request.reasoning_effort is not None:
            # By the name this template actually reads. `thinking` was
            # hardcoded, which is right for Kimi-K3 and a silent no-op for the
            # whole Qwen family, whose templates read `enable_thinking` --
            # measured, `thinking=False` left the `<think>` prefill in place.
            # A template ignores a kwarg it does not know, so the failure was
            # invisible: the model reasoned anyway.
            # Only when the request said something about it. An effort is
            # not an opt-in, and this is merged after the server defaults and
            # after the client's own `chat_template_kwargs` -- so writing it
            # unconditionally overrode both.
            # 【译】用这个模板实际读取的名字。以前硬编码 thinking，这对Kimi-K3是对的，对整个Qwen家族却是无声的空操作——
            #   它们的模板读的是 enable_thinking；实测 thinking=False 时 <think> 预填充原样保留。模板会忽略它不认识的kwarg，
            #   所以这个失败是看不见的：模型照样推理。
            #   只在请求对此有所表示时才写。effort 不等于"开启"，而且这一步在服务器默认值和客户端自己的 chat_template_kwargs
            #   之后合并——无条件写入会把两者都覆盖掉。
            # 例：Qwen3的reasoning_toggle=("enable_thinking", False, True)，请求带 "thinking": {"type": "disabled"}
            #   → merged_kwargs["enable_thinking"] = False，模板渲染出关闭思考的prompt
            if reasoning_toggle is not None and _th_enabled is not None:
                name, off_value, on_value = reasoning_toggle
                merged_kwargs[name] = on_value if _th_enabled else off_value
            if _th_effort is not None:
                merged_kwargs["thinking_effort"] = _th_effort

        # 实际的n（贪心采样时n>1被折叠成1）；return_token_ids与stream互斥
        effective_n = _coerce_n(request.n, request.temperature)
        _validate_return_token_ids(request.return_token_ids, request.stream)
        # 构造SamplingParams：max_tokens优先取max_completion_tokens，其次max_tokens，都没有则默认8192（protocol.py）；
        #   stop → stop_strings。注意：这里不读模型generation_config里的推荐采样参数，
        #   未传的temperature / top_k / top_p取protocol.py里的默认值1.0 / -1 / 1.0（/v1/messages和/v1/responses则会读generation_config）
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.get_max_tokens(),
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )

        # API层的请求ID（"chatcmpl-"+32位十六进制，OpenAI风格），写进响应的id字段；与引擎内部的seq.id（整数）不是一回事
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        # DP路由提示：请求体里的data_parallel_rank（如atomesh插入的），以及从HTTP头取的会话ID；以**dp_routing展开传给下游
        dp_session_id, dp_parent_session_id = _get_dp_session_affinity_ids(raw_request)
        dp_routing = {
            "data_parallel_rank": request.data_parallel_rank,
            "dp_session_id": dp_session_id,
            "dp_parent_session_id": dp_parent_session_id,
        }

        _log_request_model("request", request_id, request)

        # 消息里有图像/视频/音频片段 → 走多模态预处理（【分支】第2批2b）
        is_multimodal = _has_multimodal_content(messages)
        multimodal_data = None
        kv_transfer_params = request.kv_transfer_params
        if is_multimodal:
            # Media loading (blocking network I/O, up to a 30s urlopen) plus
            # processor preprocessing are heavy and would stall the event loop;
            # run them in a worker thread. Warm the processor on the loop first
            # so concurrent cold-start requests don't race on its lazy init.
            # 【译】加载媒体（阻塞的网络I/O，urlopen最长30秒）加上processor预处理都很重，会卡住事件循环；放到工作线程里跑。
            #   先在事件循环上把processor预热好，免得并发的冷启动请求在它的惰性初始化上发生竞争。
            # run_in_executor的位置参数依次传给_prepare_multimodal_inputs；返回(prompt token id列表, 多模态数据)
            _get_multimodal_processor()
            loop = asyncio.get_running_loop()
            prompt_or_tokens, multimodal_data = await loop.run_in_executor(
                None,
                _prepare_multimodal_inputs,
                messages,
                merged_kwargs,
                request.tools,
            )
        elif (pretokenized := request.get_prompt_token_ids()) is not None:
            # Reuse prefill IDs to skip template rendering and tokenization.
            # 【译】复用prefill的ID，跳过模板渲染和tokenize。
            # 语法：elif (x := f()) is not None 是"海象运算符"，在条件判断里同时完成赋值
            # 例如PD分离的decode那一路，路由器把prefill节点用过的prompt_token_ids传过来，保证两边token完全一致（第9批）
            prompt_or_tokens = pretokenized
            kv_transfer_params = _engine_kv_transfer_params(kv_transfer_params)
        else:
            # 普通情况：在事件循环上直接渲染chat template，得到prompt字符串（Jinja路径固定tokenize=False、add_generation_prompt=True）；
            #   tokenize留到preprocess在线程池里做
            # chat template、特殊token、add_generation_prompt是什么，为什么必须用模型自己的模板，以及"两个BOS"的坑，
            #   都在chat_encoders.py文件头的【科普】里。有的模板还会顺手预填思考段的开头（如"<think>\n"），
            #   所以下面要检查prompt是否已处在思考段内（见下面的【科普】推理内容）
            prompt_or_tokens = apply_chat_template(
                tokenizer,
                custom_message_encoder,
                messages,
                tools=request.tools,
                **merged_kwargs,
            )

        # 【科普】推理内容（thinking / reasoning）—— 从"模型只是逐个预测下一个token"讲起
        #   ① 模型眼里只有一条token序列。推理模型训练时学会了先写一段"草稿"、再写答案，草稿用一对标记包住
        #     （Qwen3.5的<think>是一个特殊token，id 151667；特殊token是什么见chat_encoders.py文件头的【科普】）：
        #       ...<|im_start|>assistant\n │ <think>\n先算……\n</think>\n\n │ 答案是42 │ <|im_end|>
        #            （prompt到这里为止）     └──── 草稿 → reasoning_content ┘ └ content ┘
        #     对模型来说这些都是普通的"下一个token"，没有特殊的计算路径：草稿同样占KV cache、计入completion_tokens
        #   ② 服务端按方言（reasoning_dialect：开始 / 结束标记是什么，启动时从模板读出）把反分词后的文本切成两个字段：
        #       reasoning_content="先算……"，content="答案是42"
        #     为什么要拆：UI通常折叠思考过程；多轮对话里不少模板渲染历史assistant消息时会丢掉旧回合的思考，客户端回传时要能区分；
        #     工具调用解析只应看content，否则模型在思考里"设想"的调用会被当成真调用（见anthropic_template_kwargs的docstring）
        #   ③ 难点是"从哪开始"。有的模板会替模型把开始标记预填在prompt末尾：
        #       ...<|im_start|>assistant\n<think>\n │ 先算……\n</think>\n\n答案是42
        #                                         └ 模型的第一个token已在思考段内，输出里只会出现</think>
        #     所以要先判断渲染出的prompt是否已打开思考段（下面的prompt_starts_in_reasoning → ReasoningChannel的starts_open）；
        #     DeepSeek-R1这类模型更隐蔽：只看prompt和输出都看不出开头，启动时从模板源码判断（全局变量model_starts_in_reasoning）
        #   ④ 关掉推理：请求的thinking / reasoning_effort被翻译成模板开关（上面的reasoning_toggle，如Qwen的enable_thinking=False），
        #     模板预填一个空的思考段，模型直接写答案（见chat_encoders.resolve_reasoning_toggle前的例子）
        # The K3 template may open the reasoning channel in text or token IDs.
        # 【译】K3模板可能以文本或token ID的形式打开推理通道。
        # prompt是字符串时检查文本末尾；是token id时只解码末尾几个token再检查（reasoning.py）
        _reasoning = reasoning_channel(
            (
                prompt_starts_in_reasoning(prompt_or_tokens)
                if isinstance(prompt_or_tokens, str)
                else prompt_tokens_start_in_reasoning(
                    prompt_or_tokens, tokenizer.decode
                )
            ),
            template_kwargs=merged_kwargs,
        )

        # Streaming
        # 【译】流式
        if request.stream:
            # 流式分支：先把请求交给引擎（setup_*），再构造SSE生成器gen；创建异步生成器对象时其中的代码还不会执行，
            #   等StreamingResponse开始迭代它时才运行
            stream_input = prompt_or_tokens
            stream_multimodal_data = multimodal_data
            if effective_n > 1:
                # n>1：扇出n个兄弟序列共用一个流，每个chunk的choices[0].index标明是第几个
                seq_ids, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request_fanout(
                        stream_input,
                        sampling_params,
                        request_id,
                        multimodal_data=stream_multimodal_data,
                        kv_transfer_params=kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_chat_response_fanout(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_ids,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    reasoning=_reasoning,
                    tool_parser_cls=tool_call_parser_cls,
                )
            else:
                # n=1：主路径。返回时Sequence已经交给引擎了
                seq_id, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request(
                        stream_input,
                        sampling_params,
                        request_id,
                        multimodal_data=stream_multimodal_data,
                        kv_transfer_params=kv_transfer_params,
                        **dp_routing,
                    )
                )
                # stream_chat_response（serving_chat.py）：await collector → 分离推理内容 / 工具调用 → 产出SSE帧；
                #   结束（或客户端断开）时在finally里调用cleanup_stream / cleanup_request（以参数传入，避免循环导入）
                gen = stream_chat_response(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_id,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    reasoning=_reasoning,
                    tool_parser_cls=tool_call_parser_cls,
                )
            # handler到此返回；响应头先发出，之后uvicorn迭代生成器、逐帧写socket。
            #   注意：生成器里后续抛出的异常不会再经过本函数下面的except
            return StreamingResponse(
                _client_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        # 【译】非流式
        # 四个分支（多模态 × n>1）都用_race_disconnect / _run_nonstream_with_disconnect包装，客户端中途断开时及时abort
        if is_multimodal and effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    prompt_or_tokens,
                    sampling_params,
                    request_id,
                    multimodal_data=multimodal_data,
                    kv_transfer_params=kv_transfer_params,
                    return_token_ids=bool(request.return_token_ids),
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_chat_response_multi(
                request_id,
                model_name,
                outputs,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        elif is_multimodal:
            final_output = await _run_nonstream_with_disconnect(
                generate_async_multimodal(
                    prompt_or_tokens,
                    multimodal_data,
                    sampling_params,
                    request_id,
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if final_output is None:
                raise RuntimeError("No output generated")
            resp = build_chat_response(
                request_id,
                model_name,
                final_output["text"],
                final_output,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        elif effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    prompt_or_tokens,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                    return_token_ids=bool(request.return_token_ids),
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_chat_response_multi(
                request_id,
                model_name,
                outputs,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        else:
            final_output = await _run_nonstream_with_disconnect(
                generate_async(
                    prompt_or_tokens,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                    return_token_ids=bool(request.return_token_ids),
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if final_output is None:
                raise RuntimeError("No output generated")
            resp = build_chat_response(
                request_id,
                model_name,
                final_output["text"],
                final_output,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=_reasoning,
                tool_parser_cls=tool_call_parser_cls,
            )
        # serving_chat的build_*函数已经完成了推理内容 / 工具调用的分离和usage的组装
        _log_request_model("response", request_id, resp)
        # Bypass FastAPI's recursive conversion of the token ID list.
        # 【译】绕过FastAPI对token ID列表的递归转换。
        # 直接返回pydantic对象时FastAPI会递归地做jsonable_encoder转换；先model_dump(mode="json")再包JSONResponse更快
        return JSONResponse(content=resp.model_dump(mode="json"))

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        # 【译】客户端已挂断；序列已经abort并pop。没有什么可返回的。
        # 499是nginx的惯例状态码"Client Closed Request"，客户端反正收不到
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as e:
        # 参数错误（模板参数、长度超限、采样参数非法等）→ 400
        logger.error(f"Validation error in chat_completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 其他异常 → 500；raise ... from e 保留原始异常链
        logger.exception("Error in chat_completions")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =====================================================================================
# POST /v1/completions：OpenAI文本补全接口（主路径）
# =====================================================================================
# 与chat_completions的差别：没有chat template，prompt原样（字符串或token id列表）交给引擎；
#   没有推理内容 / 工具调用分离；不支持多模态；request_id以"cmpl-"开头。其余流程（采样参数、流式/非流式、断连）相同
@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request):
    """Handle text completion requests (OpenAI-compatible)."""
    # 【译】处理文本补全请求（OpenAI兼容）。

    validate_model(request.model)

    try:
        effective_n = _coerce_n(request.n, request.temperature)
        _validate_return_token_ids(request.return_token_ids, request.stream)
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.get_max_tokens(),
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=effective_n,
        )
        # Prefer pre-tokenized IDs; otherwise pass text to preprocessing.
        # 【译】优先使用预先tokenize好的ID；否则把文本交给预处理。
        prompt_or_tokens = request.get_prompt_or_tokens()
        # 与chat不同，这里无论prompt从哪来都去掉kv_transfer_params里的prompt_token_ids
        kv_transfer_params = _engine_kv_transfer_params(request.kv_transfer_params)

        request_id = f"cmpl-{uuid.uuid4().hex}"
        dp_session_id, dp_parent_session_id = _get_dp_session_affinity_ids(raw_request)
        dp_routing = {
            "data_parallel_rank": request.data_parallel_rank,
            "dp_session_id": dp_session_id,
            "dp_parent_session_id": dp_parent_session_id,
        }

        _log_request_model("request", request_id, request)

        # Streaming
        # 【译】流式
        if request.stream:
            if effective_n > 1:
                seq_ids, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request_fanout(
                        prompt_or_tokens,
                        sampling_params,
                        request_id,
                        kv_transfer_params=kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_completion_response_fanout(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_ids,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                )
            else:
                seq_id, stream_collector, num_prompt_tokens = (
                    await setup_streaming_request(
                        prompt_or_tokens,
                        sampling_params,
                        request_id,
                        kv_transfer_params=kv_transfer_params,
                        **dp_routing,
                    )
                )
                gen = stream_completion_response(
                    request_id,
                    model_name,
                    stream_collector,
                    seq_id,
                    num_prompt_tokens,
                    cleanup_stream,
                    cleanup_request,
                )
            return StreamingResponse(
                _client_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        # 【译】非流式
        if effective_n > 1:
            outputs = await _race_disconnect(
                generate_async_fanout(
                    prompt_or_tokens,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                    return_token_ids=bool(request.return_token_ids),
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )
            if not outputs:
                raise RuntimeError("No output generated")
            resp = build_completion_response_multi(request_id, model_name, outputs)
        else:
            final_output = await _run_nonstream_with_disconnect(
                generate_async(
                    prompt_or_tokens,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                    return_token_ids=bool(request.return_token_ids),
                    **dp_routing,
                ),
                raw_request,
                request_id,
            )

            if final_output is None:
                raise RuntimeError("No output generated")

            resp = build_completion_response(request_id, model_name, final_output)
        _log_request_model("response", request_id, resp)
        return JSONResponse(content=resp.model_dump(mode="json"))

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        # 【译】客户端已挂断；序列已经abort并pop。没有什么可返回的。
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as e:
        logger.error(f"Validation error in completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error in completions")
        raise HTTPException(status_code=500, detail=str(e)) from e


# POST /v1/messages：Anthropic Messages API（B级）。把Anthropic格式转成OpenAI消息，复用同一条引擎路径
#   （setup_streaming_request / generate_async），再按Anthropic格式输出；格式转换和帧构造在serving_anthropic.py
# 与chat接口的几处差别：
#   - thinking在prompt里回应（anthropic_template_kwargs），请求没开thinking时响应里扣下推理内容（anthropic_drop_reasoning）
#   - 未传的采样参数取模型generation_config的推荐值
#   - prompt过长时截断（保留开头的max_input个token），而不是像chat那样拒绝（max_tokens大于预留量时仍会400，见_prompt_headroom）
#   - 流式生成器定义在函数内部，自己处理异常帧、保活ping和最终的stop_reason
@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest, raw_request: Request):
    """Handle Anthropic Messages API requests.

    Translates Anthropic format to OpenAI format internally, runs inference,
    and returns Anthropic-formatted responses. Enables Claude Code and other
    Anthropic-compatible tools to use ATOM as a backend.
    """
    # 【译】处理Anthropic Messages API请求。
    #
    #   在内部把Anthropic格式转成OpenAI格式，执行推理，再返回Anthropic格式的响应。
    #   让Claude Code和其他兼容Anthropic的工具可以用ATOM做后端。
    # One validator over the shape both endpoints share: this path already
    # converts, so validating the conversion leaves a single rule rather than
    # a second one in Anthropic's spelling. It checked nothing before, so a
    # name `/v1/chat/completions` rejects was accepted here. Explicitly 400
    # and before the try, because the handler below turns every exception into
    # a 500 -- wrong for a malformed request, and once the response is
    # streaming it arrives after the client was told the request succeeded.
    # 【译】对两个接口共有的形状只用一个校验器：这条路径本来就要转换，校验转换结果就只留下一条规则，
    #   而不必再用Anthropic的拼法写第二条。以前这里什么都不查，所以 /v1/chat/completions 会拒绝的名字在这里被接受了。
    #   显式返回400，并且放在try之前，因为下面的处理会把每个异常都变成500——这对格式错误的请求是错的，
    #   而且一旦响应开始流式发送，错误会在客户端已被告知请求成功之后才到达。
    # 解释：下面的try其实也有except ValueError分支返回400；放在前面的作用是在任何处理开始前就拦下非法请求
    try:
        validate_tool_list(anthropic_to_openai_tools(request.tools))
        validate_max_tokens(request.max_tokens)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    try:
        # Convert Anthropic messages to OpenAI format
        # 【译】把Anthropic消息转换成OpenAI格式
        openai_messages = anthropic_to_openai_messages(request.messages, request.system)

        # Apply chat template
        # 【译】套用chat template
        from .protocol import ChatMessage

        messages = [ChatMessage(**m) for m in openai_messages]

        merged_kwargs = dict(default_chat_template_kwargs)
        # The request's `thinking` was dropped on the floor here, so the model
        # was never told not to think and the endpoint spent three attempts
        # dealing with a chain of thought it had asked for by omission.
        # 【译】请求的 thinking 以前在这里被直接丢掉，所以模型从未被告知不要思考，
        #   这个接口花了三次尝试去处理一条它因遗漏而自己招来的思维链。
        merged_kwargs.update(anthropic_template_kwargs(request, reasoning_toggle))
        drop_reasoning = anthropic_drop_reasoning(request)
        # Same answer-it-in-the-prompt rule as `thinking` above, and the chat
        # path already forwards its own spelling of this. `tool_choice` was
        # read off the Anthropic request and then used nowhere at all -- a
        # client that forbade tool calls got `tool_use` blocks and
        # `stop_reason: tool_use`. Translated to the string a chat template
        # expects; the object forms Anthropic has for *requiring* a call have
        # no such spelling, and are not answered here or on the chat path.
        # 【译】与上面 thinking 相同的"在prompt里回应"规则，chat路径已经转发了它自己的写法。以前 tool_choice 从Anthropic请求里读出来后
        #   哪儿都没用——禁止工具调用的客户端仍然收到 tool_use 块和 stop_reason: tool_use。这里翻译成chat template期望的字符串；
        #   Anthropic用来"要求"调用的对象形式没有这样的写法，这里和chat路径上都不予回应。
        if forbids_tool_calls(request.tool_choice):
            merged_kwargs["tool_choice"] = "none"
        prompt = apply_chat_template(
            tokenizer,
            custom_message_encoder,
            messages,
            tools=anthropic_to_openai_tools(request.tools),
            **merged_kwargs,
        )

        # 请求没给的采样参数取模型generation_config里的推荐值，再没有才用protocol.py的默认值
        generation_config = engine.config.generation_config
        model_temperature = getattr(generation_config, "temperature", None)
        model_top_p = getattr(generation_config, "top_p", None)
        model_top_k = getattr(generation_config, "top_k", None)
        if model_temperature is None:
            model_temperature = DEFAULT_TEMPERATURE
        if model_top_p is None:
            model_top_p = DEFAULT_TOP_P
        if model_top_k is None:
            model_top_k = DEFAULT_TOP_K

        sampling_params = _build_sampling_params(
            temperature=(
                request.temperature
                if request.temperature is not None
                else model_temperature
            ),
            max_tokens=request.max_tokens,
            stop_strings=request.stop_sequences,
            ignore_eos=False,
            top_k=(request.top_k if request.top_k is not None else model_top_k),
            top_p=(request.top_p if request.top_p is not None else model_top_p),
        )

        request_id = uuid.uuid4().hex[:24]
        kv_transfer_params = _engine_kv_transfer_params(request.kv_transfer_params)
        # 注意：这里在事件循环上tokenize了一次（chat路径的tokenize在线程池里），用于计算input_tokens和截断
        input_tokens = len(tokenizer.encode(prompt))

        # 依次尝试几处可能的max_model_len来源，都取不到时用30720；每个请求都会打一行warning日志
        max_ctx = None
        for _path in (
            lambda: engine.config.max_model_len,
            lambda: engine.model_config.max_model_len,
            lambda: engine.scheduler.max_model_len,
            lambda: engine.max_model_len,
        ):
            try:
                _v = _path()
                if _v:
                    max_ctx = int(_v)
                    break
            except Exception:
                logger.debug("max_ctx candidate rejected", exc_info=True)
                continue
        if not max_ctx:
            max_ctx = 30720
        logger.warning(f"[anthropic] resolved max_ctx={max_ctx}")
        # 过长时截断：保留开头的max_input个token（截掉的是末尾），再解码回字符串
        # 注意截掉的是末尾：也就是最新的消息和模板补上的assistant回合开头，模型看到的是一段没说完的对话；
        #   另外截断点可能落在一条消息中间，解码后再在preprocess里重新tokenize，token数也不一定恰好等于max_input
        headroom = _prompt_headroom(request.max_tokens, max_ctx, kv_transfer_params)
        max_input = max_ctx - headroom
        if input_tokens > max_input:
            logger.warning(
                f"Prompt too long ({input_tokens} > {max_input}), truncating"
            )
            token_ids = tokenizer.encode(prompt)[:max_input]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            input_tokens = max_input

        if request.stream:
            # Streaming response
            # 【译】流式响应
            seq_id, stream_collector, _num_prompt_tokens = (
                await setup_streaming_request(
                    prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                )
            )

            # Anthropic格式的SSE生成器：message_start → content_block_*（thinking / text / tool_use）→ message_delta → message_stop
            async def generate_anthropic_stream():
                # Unconditional, like the chat path and like both upstreams:
                # whatever reasoning arrives is separated and reported. The
                # request's `thinking` was answered in the prompt, so there is
                # nothing left here for it to decide -- and separating always
                # is what keeps the reasoning out of the tool parser, which is
                # a second reader of this same text.
                #
                # Asked of every dialect, not of one literal: the K3 template
                # opens with `<|open|>think<|sep|>`, which `.endswith("<think>")`
                # does not see, and the assignment it guarded skipped
                # `__post_init__` so the instance was in the thinking state
                # while claiming it did not start there. Which dialect closes
                # it is the model's, resolved at startup -- the filter used to
                # carry none and closed on any registered dialect's marker.
                # 【译】无条件执行，与chat路径和两个上游实现一样：到达的任何推理内容都会被分离并报告。
                #   请求的 thinking 已经在prompt里回应过了，这里没有什么还需要它来决定——而始终分离，正是让推理内容远离工具解析器的关键，
                #   后者是同一段文本的第二个读者。
                #
                #   对每种方言都询问，而不是只认一个字面量：K3模板以 <|open|>think<|sep|> 开头，.endswith("<think>") 看不到它；
                #   而它保护的那个赋值绕过了 __post_init__，于是实例处于思考状态，却声称自己不是从那里开始的。
                #   由哪种方言来关闭它属于模型，在启动时确定——这个过滤器以前不带方言，遇到任何已注册方言的标记都会关闭。
                reasoning_filter = reasoning_channel(
                    prompt_starts_in_reasoning(prompt),
                    template_kwargs=merged_kwargs,
                ).stream()
                tool_parser = ToolCallStreamParser(
                    parser_cls=tool_call_parser_cls,
                    suppress_calls=forbids_tool_calls(request.tool_choice),
                )
                tool_parser.tools = anthropic_to_openai_tools(request.tools)
                blocks = AnthropicBlocks()
                has_tool_calls = False
                output_tokens = 0
                # Overwritten by a tool call, and otherwise by whatever the
                # engine says. It used to be the constant `end_turn`, so a
                # response cut off at `max_tokens` claimed a normal ending --
                # and a reasoning model asked for no thinking, which produces
                # only reasoning and has all of it dropped, delivered an empty
                # message that also said nothing was wrong. The vocabularies
                # already line up; they were simply never connected.
                # Computed once, at the end, from the two facts that decide
                # it -- exactly as `serving_chat` does. Recomputing it at each
                # send point meant whichever fact arrived last won: a call
                # completing mid-stream froze it at `tool_use`, and the
                # `max_tokens` that arrived three chunks later could never be
                # folded in. Kimi-K2 is the one registered format whose region
                # closes mid-stream, so it is the one that reaches this.
                # 【译】被工具调用覆盖，否则取引擎给出的原因。它以前是常量 end_turn，所以在 max_tokens 处被截断的响应声称自己正常结束——
                #   而一个被要求不要思考的推理模型（只产生推理，且推理全部被丢弃）会交付一条空消息，同样声称一切正常。
                #   两边的词汇本来就能对上，只是从没接起来。
                #   在最后根据决定它的两个事实只计算一次——与 serving_chat 完全一样。在每个发送点重新计算，意味着最后到达的那个事实获胜：
                #   流中途完成的调用把它冻结在 tool_use，三个chunk之后到达的 max_tokens 就再也并不进来。
                #   Kimi-K2 是唯一一种区域会在流中途关闭的已注册格式，所以只有它会走到这里。
                engine_reason: Any = None

                message_started = False
                # See `_ANTHROPIC_PING_INTERVAL_SECONDS`.
                # 【译】见 _ANTHROPIC_PING_INTERVAL_SECONDS。
                last_ping = 0.0
                # Assume abort until we reach the normal end of the stream. If
                # the client disconnects, GeneratorExit unwinds through the
                # yields and the finally runs with this still True -> abort.
                # 【译】在到达流的正常结尾之前都假定是abort。如果客户端断开，GeneratorExit 沿着各个yield展开，
                #   finally执行时它仍为True -> abort。
                aborted = True

                try:
                    while True:
                        chunk_data = await stream_collector.get()
                        if not message_started:
                            cache_read = chunk_data.get("num_cached_tokens", 0)
                            yield stream_message_start(
                                request_id, model_name, input_tokens, cache_read
                            )
                            message_started = True
                        new_text = chunk_data["text"]
                        if chunk_data.get("finish_reason"):
                            engine_reason = chunk_data["finish_reason"]
                        output_tokens += len(chunk_data.get("token_ids", []))
                        finished = chunk_data.get("finished", False)

                        # Phase 1: Reasoning filter. Never None --
                        # `reasoning_channel(...).stream()` always returns a
                        # `ReasoningFilter`. Do not add a skip path back: it
                        # would feed an unseparated chain of thought straight
                        # into the tool parser.
                        # 【译】阶段1：推理过滤器。永远不是None——reasoning_channel(...).stream() 总是返回一个 ReasoningFilter。
                        #   不要再加回跳过它的路径：那会把未分离的思维链直接喂给工具解析器。
                        segments = reasoning_filter.process(new_text)
                        if finished:
                            segments.extend(reasoning_filter.flush())

                        for field, text in segments:
                            if not text:
                                continue

                            if field == "reasoning_content":
                                if drop_reasoning:
                                    now = time.monotonic()
                                    if now - last_ping >= (
                                        _ANTHROPIC_PING_INTERVAL_SECONDS
                                    ):
                                        last_ping = now
                                        yield _ANTHROPIC_PING_FRAME
                                    continue
                                for _frame in blocks.delta("thinking", text):
                                    yield _frame
                            else:
                                # Phase 2: Tool call detection on content
                                # 【译】阶段2：在正文上检测工具调用
                                events = tool_parser.process(text)
                                has_tool_calls = has_tool_calls or (
                                    completes_a_tool_call(events)
                                )
                                for _frame in tool_event_frames(events, blocks):
                                    yield _frame

                        if finished:
                            # Flush remaining tool call events
                            # 【译】冲刷剩余的工具调用事件
                            events = tool_parser.flush()
                            has_tool_calls = has_tool_calls or (
                                completes_a_tool_call(events)
                            )
                            for _frame in tool_event_frames(events, blocks):
                                yield _frame

                            # A response with no tool call must end on a text
                            # block even when it produced none, because a reply
                            # of pure reasoning still has to carry a `text`
                            # block for clients that read only that.
                            # 【译】没有工具调用的响应必须以一个text块结束，即使它没产生任何正文，因为纯推理的回复仍然要为只读text块的客户端带上一个text块。
                            if not has_tool_calls and blocks.kind != "text":
                                for _frame in blocks.open("text"):
                                    yield _frame
                            # Before the last frames, not after. The flag means
                            # "the engine sequence may still be running", and
                            # the engine is done the moment its finished chunk
                            # arrived -- a client hanging up between the two
                            # yields below fired a broadcast abort for a
                            # sequence that had already ended, which is the
                            # control-path flood the flag exists to prevent.
                            # `serving_chat` already sets it here.
                            # 【译】放在最后几帧之前，而不是之后。这个标志的意思是"引擎序列可能还在运行"，而引擎在它的finished chunk到达那一刻就结束了——
                            #   客户端若在下面两个yield之间挂断，就会为一条已经结束的序列广播一次abort，这正是这个标志要防止的控制通道洪泛。
                            #   serving_chat 已经在这个位置设置它了。
                            aborted = False
                            for _frame in blocks.close():
                                yield _frame
                            stop_reason = anthropic_stop_reason_with_calls(
                                engine_reason, has_tool_calls
                            )
                            yield stream_message_delta(stop_reason, output_tokens)
                            yield stream_message_stop()
                            break
                except Exception as exc:
                    # Every block this stream opened has to be closed, and the
                    # client has to be told why. There was no `except` at all:
                    # the endpoint's own handler has already returned by the
                    # time the generator runs, so a raise from the collector,
                    # the reasoning filter or a tool parser cut the response
                    # mid-frame with an open block and no terminator.
                    # 【译】这个流打开的每个块都必须关闭，并且要告诉客户端原因。以前这里根本没有 except：生成器运行时接口自己的handler早已返回，
                    #   所以collector、推理过滤器或工具解析器抛出的异常会在一帧中间截断响应，留下未关闭的块，也没有终止帧。
                    logger.exception("Error streaming anthropic response")
                    for _frame in stream_failure_frames(
                        exc,
                        blocks,
                        output_tokens,
                        opening=(
                            None
                            if message_started
                            else stream_message_start(
                                request_id, model_name, input_tokens, 0
                            )
                        ),
                    ):
                        yield _frame
                finally:
                    cleanup_stream(seq_id, aborted=aborted)
                    cleanup_request(request_id)

            return StreamingResponse(
                _client_stream(generate_anthropic_stream(), request_id),
                media_type="text/event-stream",
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-request-id": request_id,
                },
            )

        # Non-streaming response
        # 【译】非流式响应
        final_output = await _run_nonstream_with_disconnect(
            generate_async(
                prompt,
                sampling_params,
                request_id,
                kv_transfer_params=kv_transfer_params,
            ),
            raw_request,
            request_id,
        )
        if final_output is None:
            raise RuntimeError("No output generated")

        raw_text = final_output["text"]
        # Separating is unconditional -- the same call the chat path makes,
        # and what keeps a chain of thought out of the tool parser. Only
        # whether the client is shown it is a question, and the same one the
        # streaming branch above asks.
        # Both stages over one chunk, in the order the branch above streams
        # them: reasoning filter, then the tool parser on the content segments
        # it yields. Calling `.split()` here instead flattened the two into
        # `(reasoning, content)` and lost the interleaving at that line.
        # 【译】分离是无条件的——与chat路径做的是同一个调用，也正是它让思维链远离工具解析器。唯一的问题是是否展示给客户端，
        #   这和上面流式分支问的是同一个问题。
        #   两个阶段作用于同一段文本，顺序与上面流式分支一致：先推理过滤器，再对它产出的正文片段跑工具解析器。
        #   在这里改用 .split() 会把两者压平成 (reasoning, content)，在这一行丢掉交错顺序。
        events = read_whole_blocks(
            reasoning_channel(
                prompt_starts_in_reasoning(prompt),
                template_kwargs=merged_kwargs,
            ),
            tool_call_parser_cls,
            raw_text,
            anthropic_to_openai_tools(request.tools),
            suppress_calls=forbids_tool_calls(request.tool_choice),
        )
        if drop_reasoning:
            events = [e for e in events if e[0] != "reasoning"]
        _, tool_calls = flatten_tool_events(events)
        # 注意：这里的output_tokens是把解码后的文本重新tokenize得到的，不等于引擎实际生成的token数
        #   （final_output["num_tokens_output"]）：skip_special_tokens去掉的EOS等特殊token不再计入，重新分词的切法也可能不同
        output_tokens = len(tokenizer.encode(raw_text))
        cache_read_input_tokens = final_output.get("num_cached_tokens", 0)
        response = build_anthropic_response(
            request_id=request_id,
            model=model_name,
            events=events,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            # A tool call is its own ending; otherwise whatever the engine
            # said. Omitting this left the parameter's `end_turn` default in
            # place, so the same response cut off at `max_tokens` reported a
            # normal ending with `stream=false` and `max_tokens` with
            # `stream=true`.
            # 【译】工具调用本身就是一种结束；否则取引擎给出的原因。省略这个参数会留下参数默认值 end_turn，
            #   于是同一个在 max_tokens 处被截断的响应，stream=false 时报告正常结束，stream=true 时报告 max_tokens。
            stop_reason=anthropic_stop_reason_with_calls(
                final_output.get("finish_reason"), bool(tool_calls)
            ),
        )
        return _with_kv_transfer_params(response, final_output)

    except _ClientDisconnected:
        # Client hung up; seq already aborted + popped. Nothing to return.
        # 【译】客户端已挂断；序列已经abort并pop。没有什么可返回的。
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )
    except Exception as e:
        logger.exception("Error in anthropic_messages")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )


# POST /v1/responses：OpenAI Responses API（B级），Codex等客户端使用。结构与anthropic_messages基本相同：
#   规范化Codex的工具形态 → 转成OpenAI消息 → 渲染模板 → 同一条引擎路径 → 按Responses格式输出（serving_responses.py）
#   请求体直接从raw_request读JSON（没有用pydantic参数），因为要先做Codex规范化，再用ResponsesRequest校验
@app.post("/v1/responses")
async def responses_create(raw_request: Request):
    """Handle OpenAI Responses API requests, including Codex tool shapes.

    Codex sends custom tools, namespace groups, and client-side tool search.
    Those are normalized to function tools, run through the same engine path
    as ``/v1/messages``, then restored to the Responses / Codex wire format.
    """
    # 【译】处理OpenAI Responses API请求，包括Codex的工具形态。
    #
    #   Codex会发送自定义工具、命名空间分组和客户端侧的工具搜索。它们被规范化成function工具，走与 /v1/messages 相同的引擎路径，
    #   再还原成Responses / Codex的线上格式。
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be valid JSON",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Responses request must be a JSON object",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    # PD handoff metadata rather than a Responses parameter: take it out before
    # the body is checked against the Responses schema.
    # 【译】这是PD交接的元数据而不是Responses的参数：在按Responses的schema检查请求体之前先把它取出来。
    kv_transfer_params = body.pop("kv_transfer_params", None)
    if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "kv_transfer_params must be a JSON object",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )
    kv_transfer_params = _engine_kv_transfer_params(kv_transfer_params)

    # 不支持的Responses参数直接400，并指明是哪个参数
    unsupported = unsupported_responses_parameter(body)
    if unsupported:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": unsupported,
                    "type": "invalid_request_error",
                    "code": "unsupported_parameter",
                }
            },
        )

    try:
        body, codex_context = normalize_codex_request(body)
        request = ResponsesRequest.model_validate(body)
        openai_tools = responses_tools_to_openai(request.tools)
        validate_tool_list(openai_tools)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    try:
        openai_messages = responses_to_openai_messages(request)
        from .protocol import ChatMessage

        messages = [ChatMessage(**m) for m in openai_messages]

        merged_kwargs = dict(default_chat_template_kwargs)
        if isinstance(request.tool_choice, str):
            merged_kwargs["tool_choice"] = request.tool_choice
        elif forbids_tool_calls(request.tool_choice):
            merged_kwargs["tool_choice"] = "none"
        if isinstance(request.reasoning, dict):
            effort = request.reasoning.get("effort")
            if effort is not None:
                merged_kwargs["thinking_effort"] = effort

        prompt = apply_chat_template(
            tokenizer,
            custom_message_encoder,
            messages,
            tools=openai_tools,
            **merged_kwargs,
        )

        generation_config = engine.config.generation_config
        model_temperature = getattr(generation_config, "temperature", None)
        model_top_p = getattr(generation_config, "top_p", None)
        model_top_k = getattr(generation_config, "top_k", None)
        if model_temperature is None:
            model_temperature = DEFAULT_TEMPERATURE
        if model_top_p is None:
            model_top_p = DEFAULT_TOP_P
        if model_top_k is None:
            model_top_k = DEFAULT_TOP_K

        max_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else DEFAULT_MAX_OUTPUT_TOKENS
        )
        sampling_params = _build_sampling_params(
            temperature=(
                request.temperature
                if request.temperature is not None
                else model_temperature
            ),
            max_tokens=max_tokens,
            stop_strings=request.stop,
            ignore_eos=False,
            top_k=(request.top_k if request.top_k is not None else model_top_k),
            top_p=(request.top_p if request.top_p is not None else model_top_p),
        )

        request_id = f"resp_{uuid.uuid4().hex[:24]}"
        input_tokens = len(tokenizer.encode(prompt))

        # max_model_len的来源：依次尝试几个对象的max_model_len属性，都没有时用30720
        max_ctx = 30720
        for obj in (
            getattr(engine, "config", None),
            getattr(engine, "model_config", None),
            getattr(engine, "scheduler", None),
            engine,
        ):
            value = getattr(obj, "max_model_len", None) if obj is not None else None
            if value:
                max_ctx = int(value)
                break
        headroom = _prompt_headroom(max_tokens, max_ctx, kv_transfer_params)
        max_input = max_ctx - headroom
        if input_tokens > max_input:
            logger.warning(
                f"Prompt too long ({input_tokens} > {max_input}), truncating"
            )
            token_ids = tokenizer.encode(prompt)[:max_input]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            input_tokens = max_input

        created_at = int(time.time())
        if request.stream:
            seq_id, stream_collector, _num_prompt_tokens = (
                await setup_streaming_request(
                    prompt,
                    sampling_params,
                    request_id,
                    kv_transfer_params=kv_transfer_params,
                )
            )

            # Responses格式的SSE生成器，结构与generate_anthropic_stream相同：推理过滤器 → 工具解析器 → emitter生成帧
            async def generate_responses_stream():
                reasoning_filter = reasoning_channel(
                    prompt_starts_in_reasoning(prompt),
                    template_kwargs=merged_kwargs,
                ).stream()
                tool_parser = ToolCallStreamParser(
                    parser_cls=tool_call_parser_cls,
                    suppress_calls=forbids_tool_calls(request.tool_choice),
                )
                tool_parser.tools = openai_tools
                emitter = ResponseStreamEmitter(
                    request_id,
                    model_name,
                    context=codex_context,
                    request=request,
                    created_at=created_at,
                )
                has_tool_calls = False
                output_tokens = 0
                opened = False
                aborted = True
                try:
                    while True:
                        chunk_data = await stream_collector.get()
                        if not opened:
                            for _frame in emitter.opening_frames():
                                yield _frame
                            opened = True
                        new_text = chunk_data["text"]
                        output_tokens += len(chunk_data.get("token_ids", []))
                        finished = chunk_data.get("finished", False)

                        segments = reasoning_filter.process(new_text)
                        if finished:
                            segments.extend(reasoning_filter.flush())

                        for field, text in segments:
                            if not text:
                                continue
                            if field == "reasoning_content":
                                for _frame in emitter.frames_for_events(
                                    [("reasoning", text)]
                                ):
                                    yield _frame
                            else:
                                events = tool_parser.process(text)
                                has_tool_calls = has_tool_calls or (
                                    completes_a_tool_call(events)
                                )
                                for _frame in emitter.frames_for_events(events):
                                    yield _frame

                        if finished:
                            events = tool_parser.flush()
                            has_tool_calls = has_tool_calls or (
                                completes_a_tool_call(events)
                            )
                            for _frame in emitter.frames_for_events(events):
                                yield _frame
                            aborted = False
                            cache_read = chunk_data.get("num_cached_tokens", 0)
                            usage = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                                "input_tokens_details": {"cached_tokens": cache_read},
                            }
                            for _frame in emitter.finish_frames(
                                usage=usage,
                                has_tool_calls=has_tool_calls,
                            ):
                                yield _frame
                            break
                except Exception as exc:
                    logger.exception("Error streaming responses")
                    for _frame in responses_stream_failure_frames(
                        exc, emitter, opened=opened
                    ):
                        yield _frame
                finally:
                    cleanup_stream(seq_id, aborted=aborted)
                    cleanup_request(request_id)

            return StreamingResponse(
                _client_stream(generate_responses_stream(), request_id),
                media_type="text/event-stream",
                headers={"x-request-id": request_id},
            )

        final_output = await _run_nonstream_with_disconnect(
            generate_async(
                prompt,
                sampling_params,
                request_id,
                kv_transfer_params=kv_transfer_params,
            ),
            raw_request,
            request_id,
        )
        if final_output is None:
            raise RuntimeError("No output generated")

        raw_text = final_output["text"]
        events = read_whole_blocks(
            reasoning_channel(
                prompt_starts_in_reasoning(prompt),
                template_kwargs=merged_kwargs,
            ),
            tool_call_parser_cls,
            raw_text,
            openai_tools,
            suppress_calls=forbids_tool_calls(request.tool_choice),
        )
        # 与/v1/messages相同：重新tokenize得到的计数，不等于引擎实际生成的token数
        output_tokens = len(tokenizer.encode(raw_text))
        cache_read_input_tokens = final_output.get("num_cached_tokens", 0)
        response = build_responses_response(
            request_id=request_id,
            model=model_name,
            events=events,
            context=codex_context,
            request=request,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            created_at=created_at,
        )
        return _with_kv_transfer_params(response, final_output)

    except _ClientDisconnected:
        return JSONResponse(status_code=499, content={"detail": "client disconnected"})
    except ValueError as exc:
        return await value_error_handler(raw_request, exc)
    except Exception as e:
        logger.exception("Error in responses_create")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": 500,
                }
            },
        )


# GET /v1/models：只返回一个模型（model_name），OpenAI客户端常用它探测服务
@app.get("/v1/models")
async def list_models():
    """List available models."""
    # 【译】列出可用的模型。
    return ModelList(data=[ModelCard(id=model_name)])


# GET /health：只要事件循环还活着就返回ok，不检查引擎状态。
#   main()在uvicorn启动之前就创建好了引擎，所以能访问到它说明引擎至少启动成功过；
#   但之后EngineCore / ModelRunner进程出了问题，这里也察觉不到（CLAUDE.md建议用rocm-smi确认显存占用）
@app.get("/health")
async def health():
    """Health check endpoint."""
    # 【译】健康检查接口。
    return {"status": "ok"}


# GET/HEAD /metrics：Prometheus抓取接口；include_in_schema=False表示不出现在自动生成的OpenAPI文档里
#   render_async返回缓存好的指标文本（由_refresh_metrics_once周期刷新），【分支】见第10批
@app.api_route("/metrics", methods=["GET", "HEAD"], include_in_schema=False)
async def metrics():
    """Expose cached standalone-engine metrics in Prometheus text format."""
    # 【译】以Prometheus文本格式暴露缓存的独立引擎指标。
    return Response(
        content=await _metrics_exporter.render_async(),
        headers={"Content-Type": _metrics_exporter.content_type},
    )


# 以下是调试 / 运维接口（B级）
# GET /debug/mtp_stats：投机解码（MTP等）的接受率统计，【分支】见第7批
#   engine.get_mtp_statistics是同步调用：向各DP rank广播工具RPC并等待回复（默认超时30秒），等待期间事件循环被占用
@app.get("/debug/mtp_stats")
async def get_mtp_stats():
    """Return current speculative decoding acceptance statistics."""
    # 【译】返回当前投机解码的接受率统计。
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialized")
    try:
        return engine.get_mtp_statistics()
    except Exception as e:
        logger.exception("Failed to get MTP statistics")
        raise HTTPException(
            status_code=500, detail=f"Failed to get MTP statistics: {e!s}"
        )


# GET /debug/cache_stats：前缀缓存命中统计（同样是同步的工具RPC），【分支】前缀缓存见第4批
@app.get("/debug/cache_stats")
async def get_cache_stats():
    """Return cumulative prefix-cache reuse statistics."""
    # 【译】返回累计的前缀缓存复用统计。
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialized")
    try:
        return engine.get_cache_statistics()
    except Exception as e:
        logger.exception("Failed to get cache statistics")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {e}"
        ) from e


# GET /debug/gc_census：统计本进程（API server）垃圾回收器扫描集合里各类型对象的数量，在专用的单线程池里执行
@app.get("/debug/gc_census")
async def get_gc_census(top: int = 30, types_per_owner: int = 2):
    """Break the collector's scan set down by type, for this process.

    Frontend-local on purpose: the engine and the workers each have their own
    interpreter and their own answer, and it is this process whose scan set
    grows with in-flight streams.

    In a thread because the walk is seconds long at a serving heap and this
    loop is the one delivering every open SSE stream -- inline, one GET would
    stall token delivery for all of them, on a server whose product is
    inter-token latency. The GIL still makes it a pause; it becomes one the
    loop can interleave around rather than a single blocking call.
    """
    # 【译】按类型分解本进程垃圾回收器的扫描集合。
    #
    #   故意只看前端本地：引擎和各worker有各自的解释器、各自的答案，而扫描集合随在途流增长的正是本进程。
    #
    #   放在线程里，是因为在服务中的堆上遍历要花好几秒，而这个事件循环负责投递所有打开的SSE流——如果内联执行，
    #   一个GET就会让所有流的token投递停顿，而这个服务器的产品就是token间延迟。GIL仍然让它成为一次停顿，
    #   但变成了事件循环可以穿插进行的停顿，而不是一次整体阻塞的调用。
    try:
        return await asyncio.get_running_loop().run_in_executor(
            _gc_census_executor, gc_census, top, types_per_owner
        )
    except Exception as e:
        logger.exception("Failed to take a GC census")
        raise HTTPException(
            status_code=500, detail=f"Failed to take a GC census: {e}"
        ) from e


# 【分支】PD分离（第9批）：从kv_transfer_config里找出本节点的角色（kv_producer = prefill节点 / kv_consumer = decode节点）
#   和握手端口；multi connector时到它的子connector里找，优先返回producer
def _resolve_kv_transfer_role(kv_cfg: dict) -> tuple[str | None, int]:
    kv_role = kv_cfg.get("kv_role")
    handshake_port = kv_cfg.get("handshake_port", 6301)
    if kv_role is not None or kv_cfg.get("kv_connector") != "multi":
        return kv_role, handshake_port

    # MultiConnector wraps the real transfer connector. Surface the producer
    # role so a P/D proxy can recognize multi[mooncake-producer + offload] as a
    # prefill node.
    # 【译】MultiConnector包装着真正的传输connector。把producer角色暴露出来，
    #   让P/D代理能把 multi[mooncake-producer + offload] 识别为prefill节点。
    fallback_role = None
    fallback_port = handshake_port
    for sub_cfg in kv_cfg.get("connectors", []):
        sub_role = sub_cfg.get("kv_role")
        if sub_role is None:
            continue
        if fallback_role is None:
            fallback_role = sub_role
            fallback_port = sub_cfg.get("handshake_port", handshake_port)
        if sub_role == "kv_producer":
            return sub_role, sub_cfg.get("handshake_port", handshake_port)
    return fallback_role, fallback_port


# GET /kv_transfer_info：给PD代理查询本节点的TP/DP大小、KV传输角色和握手端口（第9批）
@app.get("/kv_transfer_info")
async def kv_transfer_info():
    cfg = engine.config
    kv_cfg = cfg.kv_transfer_config or {}
    kv_role, handshake_port = _resolve_kv_transfer_role(kv_cfg)
    return {
        "tp_size": cfg.tensor_parallel_size,
        "dp_size": cfg.parallel_config.data_parallel_size,
        "kv_role": kv_role,
        "handshake_port": handshake_port,
    }


# GET /server_info：给Atomesh路由器的元数据（模型名、TP/DP大小）
@app.get("/server_info")
async def server_info():
    """Server metadata for the Atomesh router.

    The router's dp-aware discovery reads ``dp_size`` here to expand the
    per-DP-rank worker set and enable cache-aware routing to the rank that
    holds a request's prefix.
    """
    # 【译】给Atomesh路由器的服务器元数据。
    #
    #   路由器的DP感知发现机制读取这里的 dp_size，展开成每个DP rank一个worker的集合，并启用缓存感知路由，
    #   把请求路由到持有其前缀的那个rank。
    cfg = engine.config
    return {
        "model_id": model_name,
        "served_model_name": model_name,
        "tp_size": cfg.tensor_parallel_size,
        "dp_size": cfg.parallel_config.data_parallel_size,
    }


# POST /start_profile、/stop_profile：通过工具RPC让各EngineCore开始 / 停止性能剖析（profiler），停止时返回trace文件信息
#   【分支】profiler见第5批、第10批
@app.post("/start_profile")
async def start_profile():
    """Start profiling the engine."""
    # 【译】开始对引擎做性能剖析。
    try:
        engine.start_profile()
        return {"status": "success", "message": "Profiling started"}
    except Exception as e:
        logger.exception("Failed to start profiling")
        raise HTTPException(status_code=500, detail=f"Failed to start profiling: {e!s}")


@app.post("/stop_profile")
async def stop_profile():
    """Stop profiling the engine."""
    # 【译】停止对引擎做性能剖析。
    try:
        traces = engine.stop_profile()
        return {
            "status": "success",
            "message": "Profiling stopped. Trace files generated.",
            "traces": traces,
        }
    except Exception as e:
        logger.exception("Failed to stop profiling")
        raise HTTPException(status_code=500, detail=f"Failed to stop profiling: {e!s}")


# ============================================================================
# Main Entry Point
# ============================================================================
# 【译】主入口


# =====================================================================================
# main()：服务器启动流程（A级）
# =====================================================================================
#   1. 解析命令行：EngineArgs.add_cli_args注册全部引擎参数（第1批arg_utils.py），再加上服务器自己的参数
#   2. 请求日志、默认chat template参数
#   3. 加载tokenizer（--chat-template可以覆盖模板）；确定model_name；查找模型自带的Python消息编码器
#   4. EngineArgs.from_cli_args；多节点DP中非0号节点：只起指标HTTP服务，然后在create_engine里阻塞，不跑API
#   5. 启动期探测chat template：推理方言、是否隐式从思考段开始、关闭推理的开关、工具调用格式
#   6. create_engine → LLMEngine：启动EngineCore和ModelRunner进程、加载权重、分配KV cache、捕获CUDA Graph（耗时最长的一步）
#   7. 强制投机接受模式的占位文本；创建StreamBatchDispatcher；探测增量反分词能否复用
#   8. 把flush_stream_batch挂到CoreManager的输出线程上
#   9. 注册SIGINT处理；uvicorn.run(app, ...)启动事件循环（阻塞到服务关闭，lifespan在其中执行）
# 对比nano-vllm：nano-vllm的入口只有 LLM(path) + llm.generate(...)，没有第5、7、8、9步
# 启动与关闭的时序（uvicorn 0.54的行为：运行期间uvicorn用自己的处理器接管SIGINT/SIGTERM，结束后恢复原处理器并重新触发信号）：
#   main()                                           uvicorn.run(app)
#   ──────────────────────────────────────────────   ──────────────────────────────────────────────────────────
#   解析参数、加载tokenizer、探测模板                  建事件循环（uvloop），绑定--host:--server-port
#   create_engine（几十秒到几分钟：权重、KV、Graph）   lifespan前半：tune_gc → 刷新一次指标 → freeze_gc_heap
#   StreamBatchDispatcher、enable_delta_reuse            → 起_periodic后台任务 → yield
#   挂flush钩子、注册_sigint_handler      ───────────▶  开始接收请求（在此之前请求连不上端口）
#                                                       … 服务中 …
#                                                     Ctrl+C：uvicorn停止接收新连接、等在途请求结束
#                                                     lifespan的finally：取消后台任务 → engine.close()
#   uvicorn.run返回后，被捕获的SIGINT重新触发    ◀───  恢复原信号处理器，raise_signal(SIGINT)
#   → _sigint_handler：再次engine.close()，最多等子进程2秒，仍活着的kill → SystemExit(0)
#   好处：先排空在途请求再关引擎；代价：有长请求在跑时Ctrl+C要等一会儿（再按一次Ctrl+C，uvicorn会强制退出）
def main():
    """Main entry point for the server."""
    # 【译】服务器的主入口。
    # global：要在函数里给模块级变量赋值，必须先用global声明（否则赋值会创建同名局部变量）
    global engine, tokenizer, model_name, default_chat_template_kwargs, _request_logger
    global tool_call_parser_cls, model_starts_in_reasoning, reasoning_toggle
    global reasoning_dialect, synthetic_token_text
    global custom_message_encoder, _stream_batch_dispatcher

    # FlexibleArgumentParser（第1批utils/arg_parser.py）：--a-b和--a_b两种写法都接受
    parser = FlexibleArgumentParser(description="ATOM OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    # 以下是API server自己的参数（引擎参数已由add_cli_args注册）
    # 【译·help】服务器监听地址。
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    # 【译·help】（help文本是tool_parser/registry.py中的常量TOOL_CALL_PARSER_HELP）工具调用的线上格式。
    #   'auto'（默认）在启动时从模型的chat template中读出；显式给出名字则覆盖它。两者都无法解析时，
    #   工具调用作为普通文本交付，并在启动日志里说明——格式永远不会从输出中猜测。
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        help=TOOL_CALL_PARSER_HELP,
    )
    # 【译·help】服务器端口（注意：--port 用于引擎内部通信）。
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (note: --port is used for internal engine communication)",
    )
    # 【译·help】服务器保持空闲keep-alive连接的秒数。连接池客户端保持它那一端的时间要长得多（aiohttp是15秒），
    #   所以停顿超过这个时间的调用方会复用一个服务器已经关闭的socket，不得不重发。把它调到超过调用方的空闲时间就能避免；
    #   这里的请求一跑就是几分钟，uvicorn默认的5秒偏短。
    # 解释：本参数的默认值仍是5
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=5,
        help=(
            "Seconds the server holds an idle keep-alive connection. Pooling "
            "clients hold their end far longer (aiohttp 15s), so a caller that "
            "pauses longer than this reuses a socket the server already closed "
            "and has to re-send. Raise it past the caller's idle time to stop "
            "that; requests here run for minutes, so uvicorn's 5s is short."
        ),
    )
    # 【译·help】不让uvicorn为每个HTTP请求打一行日志。它会复制一个LogRecord，在事件循环上写到引擎日志所用的同一个stdout，
    #   而且信息量还不如引擎自己的"Request N arrived"那一行。
    parser.add_argument(
        "--disable-uvicorn-access-log",
        action="store_true",
        help=(
            "Stop uvicorn logging a line per HTTP request. It copies a "
            "LogRecord and writes to the same stdout the engine logs to, on "
            "the event loop, and says less than the engine's own "
            "'Request N arrived' line."
        ),
    )
    # 【译·help】覆盖tokenizer的chat template。接受Jinja模板的文件路径或内联的Jinja字符串。适用于没有内置chat_template的基座模型。
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help=(
            "Override the tokenizer's chat template. "
            "Accepts a file path to a Jinja template or an inline Jinja string. "
            "Useful for base models that have no built-in chat_template."
        ),
    )
    # 【译·help】渲染chat template时的默认kwargs（JSON字符串）。与每个请求的 chat_template_kwargs 合并（请求优先）。
    #   例如：'{"enable_thinking": false}'
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=str,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
            "Example: '{\"enable_thinking\": false}'"
        ),
    )
    # 【译·help】记录所有API请求和响应的JSONL文件路径（调试用）。
    parser.add_argument(
        "--request-log",
        type=str,
        default=None,
        help="Path to JSONL file for logging all API requests and responses (debug)",
    )
    # 解析结果args同时包含引擎参数和上面的服务器参数
    args = parser.parse_args()

    # 请求日志：独立的logger，propagate=False让这些行不再传给上层logger（不混进stdout）；
    #   FileHandler以追加方式写文件，格式只保留消息本身（每行一个JSON）
    if args.request_log:
        _request_logger = logging.getLogger("atom.request_log")
        _request_logger.setLevel(logging.INFO)
        _request_logger.propagate = False
        fh = logging.FileHandler(args.request_log, mode="a")
        fh.setFormatter(logging.Formatter("%(message)s"))
        _request_logger.addHandler(fh)
        logger.info(f"Request logging enabled: {args.request_log}")

    if args.default_chat_template_kwargs:
        default_chat_template_kwargs = json.loads(args.default_chat_template_kwargs)
        logger.info(f"Default chat template kwargs: {default_chat_template_kwargs}")

    # 先加载tokenizer：启动期探测chat template要用，之后再交给引擎（create_engine(tokenizer=tokenizer)）
    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = _load_tokenizer(args.model, args.trust_remote_code)
    # --chat-template：是已存在的文件路径就读文件内容，否则当作内联的Jinja字符串
    if args.chat_template:
        if os.path.isfile(args.chat_template):
            with open(args.chat_template, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
            logger.info(f"Loaded chat template from file: {args.chat_template}")
        else:
            tokenizer.chat_template = args.chat_template
            logger.info("Using inline chat template from --chat-template argument")

    model_name = args.served_model_name if args.served_model_name else args.model
    # 查找<model>/encoding/下模型自带的Python消息编码器（chat_encoders.py），找不到为None
    custom_message_encoder = load_custom_message_encoder(args.model)

    logger.info(f"Initializing engine with model {args.model}...")
    # 从Namespace中挑出EngineArgs的字段（第1批arg_utils.py）
    engine_args = EngineArgs.from_cli_args(args)
    # Remote DP nodes block inside engine startup and never run the API loop.
    # Expose their local mmap metrics on the same configured HTTP port.
    # 【译】远程DP节点会阻塞在引擎启动里，从不运行API循环。把它们本地的mmap指标暴露在同一个配置好的HTTP端口上。
    # dp_rank：环境变量ATOM_DP_RANK优先，否则--data-parallel-rank；>0表示这是多节点DP中的非0号节点
    #   （【分支】多节点DP见第2批engine_core_mgr.py、第9批）。create_engine返回后关掉指标服务，main直接return
    dp_rank = (
        envs.ATOM_DP_RANK
        if envs.is_set("ATOM_DP_RANK")
        else engine_args.data_parallel_rank
    )
    if dp_rank > 0:
        from atom.metrics.prometheus import start_metrics_server

        server, _ = start_metrics_server(args.host, args.server_port)
        try:
            engine_args.create_engine(tokenizer=tokenizer)
        finally:
            server.shutdown()
            server.server_close()
        return
    # 启动期探测chat template（结果写进前面的全局变量，供每个请求使用）：
    #   _template_source：模板源码（Jinja文本，或Python encoder的源文件）
    #   resolve_dialect：推理方言（思考段的开始/结束标记）；_dialect_stated表示模板里是否明确写出了方言
    _template_source = chat_template_source(tokenizer, custom_message_encoder)
    reasoning_dialect, _dialect_stated = resolve_dialect(
        _template_source,
        render_probe_prompt(tokenizer, custom_message_encoder, tools=False) or "",
    )
    logger.info(
        "Reasoning channel: %s%s",
        reasoning_dialect.think_end_marker,
        "" if _dialect_stated else " (no dialect named in the chat template)",
    )
    # 模板是否会关闭一个从未打开的思考块（如DeepSeek-R1）：是的话，输出默认从思考段开始
    model_starts_in_reasoning = template_opens_reasoning_implicitly(_template_source)
    if model_starts_in_reasoning:
        logger.info(
            "Chat template closes a reasoning block it never opens; treating "
            "output as reasoning until the end marker."
        )

    # 渲染探测：哪个kwarg能关掉推理（enable_thinking / thinking / thinking_mode……），见chat_encoders.resolve_reasoning_toggle
    reasoning_toggle = resolve_reasoning_toggle(tokenizer, custom_message_encoder)
    if reasoning_toggle is not None:
        _name, _off, _on = reasoning_toggle
        logger.info(f"Reasoning switches on {_name}: {_on!r} on, {_off!r} off.")
    else:
        logger.info(
            "This chat template has no switch for reasoning, so a request "
            "asking for none cannot stop the model producing it. Any that "
            "arrives is still separated and reported, never discarded."
        )

    # 工具调用格式：--tool-call-parser为auto时从模板推断（tool_parser/registry.py）
    tool_call_parser_cls = resolve_tool_call_parser(
        args.tool_call_parser,
        tokenizer,
        custom_message_encoder,
        model=args.model,
    )

    # 创建引擎（耗时最长的一步）：LLMEngine.__init__构造Config、spawn EngineCore进程，EngineCore再拉起每张GPU一个ModelRunner，
    #   加载权重、确定KV block数、预热与捕获CUDA Graph，全部READY后才返回（第2批llm_engine.py → engine_core_mgr.py → engine_core.py）
    engine = engine_args.create_engine(tokenizer=tokenizer)
    # Forced acceptance emits draft tokens that nothing verified, and once the
    # context is long enough those degenerate into the dialect's own channel
    # framing and nothing else -- read as structure, correctly, that leaves a
    # response with no content at all. The mode already announces that its text is
    # meaningless, so stand in for it rather than parse it.
    # 【译】强制接受会输出没有经过任何验证的草稿token，上下文足够长后它们会退化成只剩方言自己的通道框架标记——
    #   被（正确地）当作结构来解析后，响应里就完全没有内容了。这个模式本来就声明了它的文本没有意义，所以用占位文本代替，而不去解析。
    # --spec-decode-acceptance-length / --spec-decode-acceptance-rate是压测用的开关：强制投机解码的接受长度/接受率（第7批）
    synthetic_token_text = (
        SYNTHETIC_TOKEN_TEXT
        if (
            args.spec_decode_acceptance_length is not None
            or args.spec_decode_acceptance_rate is not None
        )
        else None
    )
    if synthetic_token_text is not None:
        logger.warning(
            "Forced speculative acceptance is on, so every generated token is "
            "delivered as %r instead of its decoded text. Token counts, timings "
            "and throughput are unaffected; the text was already meaningless. "
            "Unset --spec-decode-acceptance-length / --spec-decode-acceptance-rate "
            "to read the model's own output again.",
            SYNTHETIC_TOKEN_TEXT,
        )
    # 创建全局的流式批量投递器：持有tokenizer（反分词）、占位文本设置，以及token间延迟（ITL）指标的上报函数
    _stream_batch_dispatcher = StreamBatchDispatcher(
        tokenizer,
        synthetic_text=synthetic_token_text,
        observe_inter_token_latency=_stream_metrics.observe_inter_token_latency,
    )
    # Here and not in the dispatcher's constructor: it replays a few thousand
    # updates, which every test that builds a dispatcher would then pay for.
    # 【译】放在这里而不是投递器的构造函数里：它要重放几千次更新，每个构造投递器的测试都得为此买单。
    # enable_delta_reuse：启动时用一组探测文本验证"复用上一次的增量文本、省掉一次decode"对这个tokenizer是否安全
    #   （ATOM_DETOKENIZER_DELTA_REUSE=auto/on/off，见streaming_dispatch.py）
    enable_delta_reuse(
        tokenizer,
        envs.ATOM_DETOKENIZER_DELTA_REUSE,
        envs.ATOM_DETOKENIZER_AUDIT_EVERY,
    )

    # Wire the batched stream-flush hook: per-seq stream callbacks only buffer
    # their chunks into a thread-local; the engine core manager's output thread
    # calls this flush after each step's callbacks to drain the buffer into the
    # per-request stream collectors (one call_soon_threadsafe per event loop).
    # Registered lazily here to avoid the api_server <-> engine_core_mgr import
    # cycle; the core manager leaves the hook as None until this resolves it.
    # 【译】接上批量流式冲刷的钩子：每条序列的流式回调只把chunk缓冲进线程本地存储；engine core manager的输出线程在每个step的回调
    #   跑完后调用这个flush，把缓冲倒进各请求的流collector（每个事件循环一次call_soon_threadsafe）。
    #   在这里延迟注册，是为了避免 api_server <-> engine_core_mgr 的循环导入；在这里设置之前，core manager让这个钩子保持为None。
    engine.core_mgr._flush_stream_batch_fn = flush_stream_batch

    # Ctrl+C（SIGINT）处理：关闭引擎，等子进程最多2秒，仍在运行的强制kill，然后以SystemExit(0)退出
    import signal

    def _sigint_handler(signum, frame):
        logger.info("Received SIGINT, shutting down engine...")
        engine.close()
        import psutil

        try:
            current = psutil.Process()
            children = current.children(recursive=True)
            psutil.wait_procs(children, timeout=2)
            alive = [c for c in children if c.is_running()]
            for c in alive:
                c.kill()
        except psutil.NoSuchProcess:
            pass
        logger.info("Engine shutdown complete.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # uvloop replaces the stdlib asyncio selector loop with a libuv-backed one,
    # which is markedly faster at the SSE socket I/O (sock.send / selector
    # register-unregister) that saturates the event loop under high streaming
    # concurrency. Fall back to the default loop if uvloop is unavailable.
    # 【译】uvloop用基于libuv的事件循环替换标准库asyncio的selector循环，它在SSE的socket I/O（sock.send / selector的注册与注销）上
    #   明显更快，而高并发流式时正是这些I/O把事件循环打满。uvloop不可用时退回默认循环。
    try:
        import uvloop  # noqa: F401

        loop_impl = "uvloop"
    except ImportError:
        loop_impl = "auto"
        logger.warning(
            "uvloop not installed; falling back to the default asyncio loop."
        )

    logger.info(
        f"Starting server on {args.host}:{args.server_port} (loop={loop_impl})..."
    )
    # 启动uvicorn：在事件循环上运行app，阻塞直到服务关闭。
    #   loop="uvloop"/"auto"选择事件循环实现；access_log控制每请求一行的访问日志；timeout_keep_alive见上面的参数说明
    uvicorn.run(
        app,
        host=args.host,
        port=args.server_port,
        loop=loop_impl,
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=args.timeout_keep_alive,
    )


# 直接运行本文件时的入口（通常经由atom/entrypoints/openai_server.py启动）
if __name__ == "__main__":
    main()
