# ATOM 源码注释计划（10 批，全量覆盖）

参考 `../nano-vllm-annotated` 的注释风格，为 ATOM 加中文注释。**`atom/` 下全部 787 个 .py 文件（约 34.4 万行）都分到了某一批**，按深度分为三级。读者按批次顺序阅读，可以从“一个请求怎么进来”一路读到 DeepSeek-V4、Mono 融合 kernel、PD 分离和 vLLM 插件。

本计划对应 **ATOM v0.1.8**（upstream commit `daa4f1bb8`）。所有文件数、行数和文件名都按这个版本统计。

- 逐文件清单（每个文件的批次、深度、行数，带勾选框，可用来记录进度）：[ANNOTATION_CHECKLIST.md](ANNOTATION_CHECKLIST.md)
- 分批规则由脚本生成并校验：没有遗漏，也没有重复

---

## 0. 总体思路

| | nano-vllm | ATOM |
|---|---|---|
| Python 文件 | 19 个 | 787 个 |
| 代码行数 | ~3,000 | ~344,000 |
| 最大单文件 | model_runner.py 605 行 | scheduler.py 5,346 行、moe.py 5,383 行 |

**全量覆盖，深度分级**。38 万行不可能全部逐行注释，读者也读不完。所以每个文件都会注释，但深度不同：

- 前 7 批是主干：引擎、调度、KV、执行器、算子、注意力、MoE/MLA/投机解码。以 **A 逐行**和 **B 精注**为主，对应 nano-vllm 的全部内容，再加上 ATOM 自己的多进程、编译、量化、MoE 等。
- 后 3 批是扩展：新模型、Mono、分布式、PD 分离、插件、diffusion。以 **B 精注**和 **C 导读**为主，每个文件都写清楚“它是什么、谁调用它、和主干哪里接上”。核心文件用 B 级展开，kernel 和适配代码用 C 级。

**阅读顺序**：先自顶向下读引擎（请求 → 进程 → 调度 → KV → 执行器），再自底向上读模型（算子 → 注意力 → 模型），最后读扩展。每一批只依赖前面的批次。

```
 第1批 配置/请求对象
    │
 第2批 服务层 → LLMEngine → CoreManager ═ZMQ═> EngineCore → AsyncIOProcManager ═ZMQ═> ModelRunner×TP
                                                 │                                       │
                                       第3批 Scheduler                  第5批 执行/加载/采样/CUDA Graph/编译
                                       第4批 BlockManager + 状态缓存                      │
                                                                                    model.forward()
                                                                                         │
                              第6批 Linear/量化/RMSNorm/RoPE/MHA 注意力 → Qwen3、Llama
                              第7批 MoE/EP、MLA、DSA、MTP/EAGLE/DSpark → DeepSeek-V3、Qwen3-MoE、GLM-4、GPT-OSS…
                              第8批 GDN 线性注意力、V4/V4.1、Kimi-K3、MiniMax-M3、GLM-5.x、Qwen4-exp
                              第9批 Mono 融合 decode │ PP/DCP/PCP/TBO │ PD 分离、KV offload
                              第10批 vLLM/SGLang/RTP-LLM 插件 │ RL rollout │ 指标/追踪/压测/调试 │ diffusion
```

---

## 1. 注释规范

### 1.1 三级深度

| 级别 | 写什么 | 用于 |
|---|---|---|
| **A 逐行详注** | nano-vllm-annotated 的完整风格：文件 / 类开头写总览（职责、架构位置、ASCII 流程图或时序图），成员变量逐个说明，方法按步骤分段解释，几乎每个非平凡语句都有注释 | 主干核心文件，共 52 个 |
| **B 精注** | 文件头总览；每个类和函数开头写用途、输入输出（张量形状）、调用方；函数体内只注释关键行：分支条件、数据流转折、不直观的技巧 | 主干的配套文件、扩展部分的核心文件，共 260 个 |
| **C 导读** | 文件头写职责、调用方、与主干的关系（替代或扩展了哪一批的哪个机制）；每个类和公开函数一行用途 | kernel、协议适配、补丁、工具，共 431 个 |

另有 35 个空的 `__init__.py` 或只有几行导出的文件不注释。9 个上游 vendored 文件不修改，见 1.5。

C 级示例：

```python
# 【导读】DeepSeek-V4 的分页 decode 注意力 Triton kernel
# 调用方：attentions/deepseek_v4_attn.py 的 decode 路径
# 与主干的关系：替代第6批 attention_mha.py 的 paged decode，读的是 V4 压缩后的 KV 行
```

### 1.2 写法（沿用 nano-vllm-annotated）

1. **中文 `#` 行注释**写在被解释代码的上方，很短的可以写在行尾。保留原有英文 docstring 和注释，**不改任何代码逻辑**。
2. **讲“为什么”，不复述代码**。Python 语法点（`dataclass(slots=True)`、`deque.extendleft` 等）第一次出现时顺带解释。
3. **张量标注形状**，例如 `# [num_tokens, num_heads * head_dim]`。
4. **用具体数字举例**，例如“running=[A,B,C,D]，空闲 block 为 0 时的抢占过程”。
5. **对照 nano-vllm**：用 `# 对比nano-vllm：...` 说明 ATOM 多了什么、为什么。
6. **跨批交叉引用**：主干文件里遇到扩展分支，在入口写一行，例如
   `# 【分支】PD分离：KV从prefill节点传过来，见第9批 kv_transfer/disaggregation`
   扩展文件的文件头反过来写它挂在主干的哪个位置。
7. **一个文件内深度可以混合**：清单里标的是文件的主要深度。例如 `config.py` 标 A，但核心类逐行注释，DSpark / EPLB 等配置只做 B 级。各批详情里会写明。
8. **英文注释一律附中文翻译**，与深度级别无关：
   - 英文 `#` 注释：在这段注释正下方加翻译，首行以 `# 【译】` 开头，续行用 `#   ` 缩进；
   - docstring：在结束的 `"""` 后面紧跟 `# 【译】` 注释（不能写进字符串里，否则会改变代码的 AST）；
   - 命令行参数的英文 help 文本是字符串、不是注释，但它是参数的说明文档，也翻译：在 `add_argument(` 上方写 `# 【译·help】`；
   - 自己补充的解释不加【译】标记，保证“翻译”和“解释”能区分开；
   - 不翻译：SPDX 版权头、被注释掉的旧代码、`# noqa` / `# type:` / `# fmt:` 等工具指令。

   示例：

   ```python
   # Built here rather than as a default argument: one instance shared by
   # every defaulting Sequence would be a mutable default in all but name.
   # 【译】在这里创建，而不是写成参数默认值：若所有使用默认值的 Sequence 共享同一个实例，
   #   那它实际上就是一个可变默认参数
   if sampling_params is None:
   ```

### 1.3 面向读者的写法（第 1、2 批回看后确定，之后每批照此执行）

**目标读者**：只懂标准 Transformer + MHA / GQA（大致读过 nano-vllm），不懂推理系统和新架构。注释的唯一标准是"这个读者能不能读懂"。

1. **新概念从 MHA / GQA 讲起，通俗易懂、图文并茂。** 遇到 kpool、DSA / 索引器、MLA、CSA、GDN / KDA、mHC、Engram、状态 checkpoint、投机解码、DP-attention、CUDA Graph 等机制名，不能只写名字，要在**第一次出现的最自然的位置**写一段【科普】：
   - 先说读者已知的东西（"MHA 里新 token 的 q 要和全部 L 个历史 k 点积…"），再说新机制改了哪一步、为什么；
   - 配 ASCII 图（数据怎么流、张量怎么排、token 怎么分组）；
   - 配具体数字（"index_kpool=4 → 2048 个 token 只需选 512 个池"）；
   - 写清"什么时候结果和朴素做法一样、什么时候不同"这类边界。
   - **先交代前置知识**：讲机制 A 时如果要用到读者不懂的机制 B（如讲 k-pool 要用到 MLA、讲 DCP 要用到 MLA 只有 1 个 KV 头），先用一小段、配**本模型的具体数字**交代 B 中理解 A 所必需的要点，再指向 B 的完整讲解；不要让读者在不懂 B 的情况下读 A。
   范例：`config.py` 中 `glm5_kpool_block_size` 上方的"【科普】什么是 k-pool"（① 全量注意力的开销 → ①′ 前置：GLM-5.3 的主注意力是 MLA → ② DSA 与索引器 → ③ k-pool → ④ 为什么 block 要变成 64，附一个 block 里装了什么的图）。
2. **按 nano-vllm-annotated 的结构组织**：文件 / 类开头先给**概念**，再给**流程图 / 状态机 / 内存布局图 / 时序图**（方框图，和代码一一对应），再**用一条具体请求带数字走一遍**，最后写**适用场景 / 好处 / 代价**（选项、模式、并行方式都要有"什么时候用、代价是什么"）。
3. **重点讲推理原理和"为什么这样设计"**，例如 prefill 计算受限 vs decode 带宽受限、为什么延迟输出能掩盖 CPU 时间、为什么 DP 要锁步。**简单代码略过**：读环境变量、getter、日志、自解释的 pydantic 字段不写。
4. **一个概念只讲一次**：在最自然的位置讲透，其他地方写一行指针（"见 xx.py 的【科普】③"），并确认指针目标存在。不要在三个文件里用三种措辞讲同一件事。
5. **长文件开头写阅读路线**：本文件分几部分、建议阅读顺序、需要的前置概念（指向对应【科普】）；必要时附术语表。
6. **事实必须核实**："谁调用 / 谁读取 / 默认值 / 生效条件"都要用 grep 或读实现确认，写不出出处的就不写；从硬件规格、论文得来的数字注明来源。图表中的数字要能按代码里的公式算出来。
7. **发现旧注释写错，直接改成正确说法**（必要时说明依据），不要留下"先写错、后面再更正"的两段；对上游代码本身的问题（疑似 bug、文档与实现不符），以"观察："写明事实，不改代码。
8. **跨文件保持一致**：每批结束时，对本批涉及的关键事实做一次"读者视角通读"，专门找：先用后解释的术语、前后矛盾、重复讲解、失效的交叉引用（旧行号、不存在的路径）、与代码不符的图表。第 1、2 批通读中统一过的事实见 §4"已核实的关键事实"。

### 1.4 每批的执行与验收

- 改动后对修改过的文件跑两个检查脚本（在 [annotation_tools/](annotation_tools/) 下，需 Python ≥ 3.12）：
  - `python annotation_tools/verify_comments_only.py [--base daa4f1bb8] <files>`：对比 git HEAD（或 `--base` 指定的未注释上游版本）的 AST，证明只加了注释、没改代码，并跑 black --check 和 ruff；
  - `python annotation_tools/untranslated.py <files>`：列出还没有 `# 【译】` 翻译的英文注释和 docstring（自己画的图、被注释掉的代码属于误报，可以忽略）。
- `annotation_tools/gen_checklist.py ANNOTATION_CHECKLIST.md --base daa4f1bb8` 生成清单（行数按未注释的上游版本统计）；重新生成会清空已勾选的进度，需要重新勾选。
- 每批完成后勾选 [ANNOTATION_CHECKLIST.md](ANNOTATION_CHECKLIST.md)，并提交一次，提交信息沿用 nano-vllm 格式：`【注释】第N批：调度器`。
- 后 3 批文件多，按小节（如 8a、8b）拆给多个子任务并行注释，最后统一检查术语和交叉引用。
- 注释不影响 Dynamo 追踪，但文件 hash 会变，`@support_torch_compile` 模型首次运行会重新编译。在 GPU 上运行前按 CLAUDE.md 清理 `/root/.cache/atom/*`。
- 推送：所有改动直接推到 `v0.1.8-annotated-claude`（快进推送，不改写已推送的历史）。
- 执行顺序（第 3 批起）：① 首次注释（翻译 + 1.2 / 1.3 的解释、【科普】、图表、适用场景）→ ② 读者视角通读修订（1.3 第 8 条）→ ③ 两个检查脚本 + 非注释 diff 为空 → ④ 勾选清单、提交。大文件可按函数区段拆给多个子任务并行，但每个子任务只改自己的区段，写入前重新读文件。
- 编辑注意：本仓库的编辑钩子会对 Edit 工具改过的文件跑 `ruff --fix`，可能删掉已有的 `# noqa`。用脚本做精确替换（`str.replace` 并断言只出现一次），改完检查 `git diff HEAD -- <files> | grep '^[-+]' | grep -v '^[-+]\s*#'` 为空，且 `# noqa`、【译】的数量没有减少。

### 1.5 不修改的文件

- `atom/models/glm52/mono/kernels/`（9 个文件）：从 FlyDSL 上游 vendored，须与上游保持字节一致（见 `pyproject.toml` 的说明），改动会破坏对齐，还可能改变 FlyDSL 重写器的生成结果。它们的导读写在 `glm52/mono/runner.py` 和 `dispatch.py` 的文件头里。
- 仓库中不属于 `atom/` 的部分（`tests/`、`tools/`、`scripts/`、`docker/`）不在计划内。`atom/mesh/` 主体是 Rust 代码，只注释其中一个 Python 脚本。

---

## 2. 总览

| 批次 | 主题 | 文件数 | 源码行数 | 深度构成（A / B / C） | 配套文档 |
|---|---|---|---|---|---|
| 1 | 入口、配置与请求对象 | 17 | 8.1K | 6 / 9 / 1 | configuration_guide、environment_variables |
| 2 | 引擎主循环、多进程架构与服务层 | 57 | 23.2K | 6 / 24 / 27 | architecture_guide、serving_benchmarking_guide |
| 3 | 调度器 | 2 | 5.9K | 1 / 1 / 0 | scheduling_kv_cache_guide |
| 4 | KV cache 账本与状态缓存 | 13 | 8.6K | 4 / 9 / 0 | scheduling_kv_cache_guide |
| 5 | ModelRunner：执行、权重加载、采样、CUDA Graph 与编译 | 43 | 17.8K | 9 / 30 / 4 | compilation_cudagraph_guide、h2d_publication |
| 6 | 基础算子、量化、注意力与 dense 模型 | 34 | 17.0K | 16 / 16 / 0 | model_ops_guide、model_support_guide |
| 7 | MoE、MLA、投机解码：DeepSeek-V3 与 MoE 模型家族 | 51 | 36.4K | 10 / 40 / 0 | eagle3_speculative_decoding、recipes/DeepSeek-R1、DSpark |
| 8 | 新一代模型：线性注意力、DeepSeek-V4 / V4.1 及其他 | 155 | 72.6K | 0 / 26 / 123 | deepseek_v41_*、recipes 中对应模型 |
| 9 | 进阶执行与分布式：Mono、并行扩展、PD 分离与 KV offload | 205 | 65.4K | 0 / 66 / 125 | distributed_guide、context_parallel_guide、dynamic_chunking_guide、recipes/TBO、pd_disaggregation_guide |
| 10 | 对外集成与周边：插件、RL rollout、可观测性、diffusion | 201 | 83.5K | 0 / 39 / 151 | vllm_plugin_backend_guide、metrics_guide、observability、recipes/MiniMax-H3 |
| — | 不修改（vendored） | 9 | 5.9K | — | — |

后 3 批行数多，但以 C 级为主，实际阅读量和前几批相当：读者主要读文件头导读和 B 级的核心文件。

---

## 3. 分批详情

下面只列 **A** 和 **B** 级的重点；每个文件的完整归属见清单。

### 第 1 批：入口、配置与请求对象

**目标**：知道怎么启动 ATOM，所有配置项从哪来，一个请求在系统里被表示成什么。

| 文件 | 深度与范围 |
|---|---|
| `examples/simple_inference.py` | A：离线推理入口，EngineArgs → create_engine → generate |
| `model_engine/arg_utils.py` | A：`EngineArgs`、`from_cli_args`、`_get_engine_kwargs`、`create_engine`；`add_cli_args` 按参数组注释 |
| `config.py` | A：`Config`、`KVCacheConfig`、`KVCacheTensor`、`CUDAGraphMode`、`CompilationLevel`、`CompilationConfig`、`ParallelConfig`、`get_hf_config`；B：`QuantizationConfig`、`SpeculativeConfig`、`DSparkConfig`、`EPLBConfig`、`DCPConfig`、`KVEventsConfig`，以及各模型的 config 归一化函数 |
| `sampling_params.py` | A：`SamplingParams` |
| `model_engine/sequence.py`、`request.py` | A：`Sequence`、`SequenceStatus`、`BlockTable`、`RequestOutput` 等 |
| `utils/envs.py` | B：按功能分组说明所有 `ATOM_*` 环境变量 |
| `utils/__init__.py`、`utils/arg_parser.py`、`model_config/*`、其余 examples | B |

**读完应能回答**
- `--max-num-batched-tokens`、`--max-num-seqs`、`--gpu-memory-utilization`、`-tp`、`--level`、`--cudagraph-mode` 分别影响什么？
- `Sequence` 比 nano-vllm 多了哪些字段？各自服务哪个特性？
- `model_config/` 为什么要给 Qwen3.5、Qwen3-Next、Kimi 单独写 HF config？

---

### 第 2 批：引擎主循环、多进程架构与服务层

**目标**：看清三层进程结构、请求和结果在进程间怎么流动，以及 HTTP 请求怎么变成 `Sequence`。这是 ATOM 与 nano-vllm 差别最大的部分：nano-vllm 用共享内存 + Event，ATOM 用 ZMQ + pickle。

**2a 离线引擎与多进程**

| 文件 | 深度与范围 |
|---|---|
| `llm_engine.py` | A：`LLMEngine`、`InputOutputProcessor.preprocess / postprocess`；B：统计接口、`preprocess_fanout` |
| `engine_core_mgr.py` | A：`CoreManager` 启动、`add_request`、`_dispatch_to_dp_ranks`、输出线程、`launch_engine_core`；B：DP 负载均衡、背压、weight sync、`DisaggCoreManager` |
| `engine_core.py` | A：`EngineCore.busy_loop`、`_process_engine_step_inner`、输入输出 socket 线程、`DPEngineCoreProc`；B：`PrefillEngineCore` / `DecodeEngineCore`（交叉引用第 9 批） |
| `async_proc.py` | A：`AsyncIOProc`、`AsyncIOProcManager` |
| `ipc_utils.py`、`engine_core_protocol.py`、`engine_utility.py`、`utils/numa_utils.py`、`utils/gc_utils.py`、`utils/network.py` | B |
| `collective_rpc.py`、`capabilities.py`、`engine_stats.py`、`run_labels.py` | C |

**2b 多模态输入**：`multimodal_runtime.py`、`multimodal/*`，均为 B。

**2c OpenAI 兼容服务层**

| 文件 | 深度与范围 |
|---|---|
| `entrypoints/openai_server.py`、`openai/api_server.py` | A：启动流程、`lifespan`、`chat_completions`、`completions`、`generate_async`、流式输出与断连处理；B：其余接口 |
| `serving_chat.py`、`serving_completion.py`、`protocol.py`、`sse.py`、`streaming_dispatch.py`、`chat_encoders.py`、`chat_utils.py` | B |
| `tool_parser/*`、`reasoning*.py`、`serving_anthropic.py`、`serving_responses.py`、`atomesh/*` 等 | C |

**读完应能回答**
- 画出 `LLMEngine → CoreManager → EngineCore → AsyncIOProcManager → ModelRunner` 的进程与 socket 图（ROUTER/DEALER、PUSH/PULL 各在哪一端）。
- EngineCore 的 I/O 线程和 busy_loop 主线程怎么分工？
- DP>1 时，一个 rank 有 prefill、另一个没有，为什么要跑 dummy batch？
- 一个流式 chat 请求从 HTTP 到第一个 SSE chunk 经过了哪些函数？

---

### 第 3 批：调度器

**目标**：理解 ATOM 每个 step 如何挑选 prefill / decode 请求，以及 `ScheduledBatch` 里装了什么交给 ModelRunner。

| 文件 | 深度与范围 |
|---|---|
| `model_engine/scheduler.py` | A：`ScheduledBatch`（所有字段）、`ScheduledBatchOutput`、`Scheduler.__init__`、`schedule` / `_schedule` 主干（prefill 优先、chunked prefill、decode、预算）、`_schedule_prefill_seq`、`_chunked_prefill_size`、`_finalize_prefill_chunk`、`preempt`、`postprocess`、`_mark_finished`、`abort_request`；B：`_connector_*`、remote KV、offload、state load/store、dynamic chunking、`PrefillScheduler`，每处写明对应第 4 批或第 9 批 |
| `model_engine/prefill_delayer.py` | B：本地 TP / 跨 DP 的 prefill 合并 |

`_schedule` 约 900 行：先在开头画一张决策流程图（同 nano-vllm），再在每个分支入口标注“主干 / 分支”。

**读完应能回答**
- ATOM 的调度策略与 nano-vllm 有哪些相同、哪些不同（chunked prefill、是否混合 prefill/decode、抢占方式）？
- `ScheduledBatch` 中哪些字段在 CPU 上算好，再传给 GPU 侧？
- `postprocess` 为什么要处理“延迟一步的输出”？

---

### 第 4 批：KV cache 账本与状态缓存

**目标**：理解 KV cache 在 CPU 侧的账本，以及混合架构模型（线性注意力、Mamba）的递归状态如何缓存。

**4a KV block 与前缀缓存**

| 文件 | 深度与范围 |
|---|---|
| `block_manager.py` | A：`can_allocate`、`_evaluate`、`_walk_prefix`、`allocate`、`hash_blocks`、`hash_decode_blocks`、`deallocate`、`can_append`、`may_append`、`draft_kv_lookahead`；B：state checkpoint、joint boundary、midstep、offload、remote prefix、KV events |
| `block_pool.py` | A：空闲链表、引用计数、内容寻址的 hash 索引、惰性淘汰 |
| `kv_block.py`、`prefix_hash.py` | A |
| `block_table_codec.py`、`utils/block_tables.py`、`utils/block_convert.py`、`distributed/kv_events.py` | B |

**4b 递归状态缓存**：`state_pool.py`、`state_cache.py`、`state_runtime.py`、`state_offload.py`、`page_unit_checkpoint.py`，均为 B。

**读完应能回答**
- 前缀缓存命中的判定条件是什么？同样是 xxhash 链式 hash，ATOM 比 nano-vllm 多处理了哪些情况？
- 一个 block 被释放后，它的 hash 和内容什么时候才真正失效？
- 线性注意力模型没有逐 token 的 KV，前缀缓存靠什么实现？

---

### 第 5 批：ModelRunner —— 执行、权重加载、采样、CUDA Graph 与编译

**目标**：跟着一个 `ScheduledBatch` 走完 GPU 侧的全部流程：初始化、显存预算、KV 分配、准备输入、forward、采样、异步回传，以及 CUDA Graph 与分段编译。

**5a ModelRunner 与 ForwardContext**

| 文件 | 深度与范围 |
|---|---|
| `model_runner.py` | A：`__init__`、`_setup_device_and_distributed`、`warmup_model`、`get_num_blocks`、`allocate_kv_cache`、`allocate_forward_vars`、`tokenIDProcessor`（延迟输出）、`prepare_inputs`、`prepare_model`、`run_model`、`postprocess`、`forward`、`capture_cudagraph`；B：`RapidServeModelRunner`、PCP、dspark、TBO、profiler |
| `utils/forward_context.py` | A：`Context`、`AttentionMetaData`、`ForwardMode.decide`、`DPMetadata`、`ForwardContext`、`set/get/reset_forward_context` |
| `utils/h2d.py`、`utils/packed_h2d.py`、`model_ops/decode_input_ids.py` | B：host 到 device 的元数据发布 |

**5b KV pool 内存布局与 token 索引**：`attentions/pool_layout/*`（一个字节在 KV pool 里的位置）、`attentions/token_layout/*`（slot_mapping 等逐 token 索引），均为 B。

**5c 权重加载**：A：`loader.py`、`loading_core.py`、`weight_iterator.py`；B：`weight_dispatch.py`、`weight_names.py`、`expert_staging.py`、`online_quant_streaming.py`、`weight_utils.py`。

**5d 采样**：A：`sampler.py`；B：`lm_head_argmax.py`。

**5e CUDA Graph 与 torch.compile**：A：`utils/decorators.py`、`utils/backends.py`、`utils/cuda_piecewise_backend.py`；B：`cuda_graph.py`、`attn_ffn_piecewise.py`、`graph_holders.py`、`custom_register.py`、`module_dispatch_ops.py`；C：`compiler_inferface.py`、`graph_marker*.py`。

**读完应能回答**
- ATOM 如何算出能分配多少 KV block？比 nano-vllm 多扣了哪些显存？
- 一次 decode step 中 CPU 和 GPU 各在做什么？延迟输出让哪一段 CPU 时间被 GPU 计算掩盖了？
- `--level 3`（piecewise）在哪里切图？为什么 attention 留在图外？
- 一个 checkpoint 权重名（如 `q_proj`）如何被映射并写入融合后的 `qkv_proj`？

---

### 第 6 批：基础算子、量化、注意力与 dense 模型

**目标**：自底向上读懂一个完整的 dense 模型，理解 TP 切分、量化 GEMM 分发，以及注意力的三件套：Backend、MetadataBuilder、Impl。

| 小节 | A 级 | B 级 |
|---|---|---|
| 6a 基础算子 | `linear.py`（`LinearBase` 与 Column / Row / QKV / Merged 并行层；量化分支展开 BF16、FP8）、`embed_head.py`、`layernorm.py`（RMSNorm 主干）、`rotary_embedding.py`、`activation.py`、`communication_op.py` | `utils.py`、`base_config.py`、`triton_rmsnorm_nw.py`；`linear.py` 中 Qwen3-Next / MiniMax 专用的并行层 |
| 6b 量化 | `quant_spec.py` | `quantization/quark/utils.py`、`blockscale.py`、`blockscale_kernels/*`、`triton_fused_qkv_quant.py` |
| 6c 注意力 | `utils/selector.py`、`base_attention.py`、`paged_attention.py`、`attention_mha.py`（`PagedAttentionImpl`）、`attentions/backends.py`、`attentions/aiter_attention.py`、`attentions/mha_kv_pool.py` | `triton_mha.py`、`triton_merge_attn_states.py`；`SparseMHAPagedAttentionImpl`、FlyDSL plan、ubatch |
| 6d dense 模型 | `models/utils.py`、`models/qwen3.py`；模型注册表 `support_model_arch_dict`（v0.1.8 中位于 `model_runner.py`） | `llama.py`、`mistral3.py`、`eagle3_llama.py`、`fused_aux_rmsnorm.py` |

**读完应能回答**
- Column / Row 并行如何搭配，使每层只需一次 all-reduce？
- 权重的量化类型在哪一步决定使用哪个 GEMM kernel？
- 一个 token 的 K/V 写到 KV cache 的哪个地址？prefill 和 decode 分别调用哪个 AITER kernel？
- 捕获 CUDA Graph 时，注意力元数据 buffer 为什么必须预先分配好？

---

### 第 7 批：MoE、MLA、投机解码 —— DeepSeek-V3 与 MoE 模型家族

**目标**：读懂 ATOM 的旗舰路径：融合 MoE 与专家并行、MLA 注意力、MTP / EAGLE / DSpark 投机解码，并覆盖同一家族的其他模型。

| 小节 | A 级 | B 级 |
|---|---|---|
| 7a MoE 与专家并行 | `moe.py`（`FusedMoE`、`UnquantizedFusedMoEMethod`、`Fp8MoEMethod`、`moe_forward`；MXFP4 / NVFP4 等量化方法做 B 级）、`topK.py`、`qwen3_moe.py` | `fused_moe_triton.py`、`fused_moe/*`（modular kernel、MORI / RCCL 的 dispatch 与 combine、all2all）、`eplb.py`、`mixtral.py` |
| 7b MLA 与 DeepSeek-V3 | `attention_mla.py`（V3 dense 路径：权重吸收、prefill 用 MHA 形式、decode 用 MQA 形式）、`attentions/aiter_mla.py`、`deepseek_v2.py`、`deepseek_mtp.py` | `triton_mla.py`、`triton_fused_mla_ctx_kv.py`、`mla_kv_pool.py`、`fused_mtp_prologue.py`、`eagle3_deepseek_mla.py` |
| 7c 投机解码框架 | `spec_decode/drafter.py`、`eagle_proposer.py`、`rejection_sampler.py` | `draft_graph.py`、`draft_kv.py`、`dspark_proposer.py`、`dspark_scheduler.py`、`dspark_verify.py`、`calibration.py`、`dspark_markov_sample.py`、`models/dspark_draft.py` |
| 7d DSA 稀疏索引器（V3.2 / GLM-5） | — | `sparse_indexer_*.py`（v0.1.8 没有 `model_ops/indexer/` 目录）、`fp4_mqa_ragged_metadata.py`；`deepseek_v2.py` 中的 `Indexer` |
| 7e 其他 MoE 模型 | — | `glm4_moe.py` 及其 MTP、`gpt_oss.py`、`swiglu_oai.py`、`minimax_m2.py`、`mimo_v2.py` 及其 MTP、`kimi_k25.py` |

本批也回头补全前几批里与投机解码相关的部分：`SpeculativeConfig`、`SpecDecodeMetadata`、`model_runner.propose_draft_token_ids`、`aiter_attention.prepare_mtp_decode`、`sampler.sample_verification_tokens`。

**读完应能回答**
- 一个 token 从 router logits 到 MoE 输出经历了哪几步？TP 和 EP 下专家权重分别放在哪些 GPU 上？
- MLA 的 KV cache 每个 token 存多少字节？为什么 decode 时要把 W_UK 吸收进 q、把 W_UV 吸收进输出投影？
- 一次 MTP step 中目标模型验证了几个 token？DSpark 与 EAGLE 的草稿方式有何不同？
- DSA 的索引器如何从全部历史 token 中选出 top-k 参与注意力？

---

### 第 8 批：新一代模型 —— 线性注意力、DeepSeek-V4 / V4.1 及其他

**目标**：在前 7 批的骨架上，看每个新模型改了哪一块：注意力换成线性注意力或压缩稀疏注意力、加了 Engram 记忆、KV 布局变化等。每个模型的 B 级文件写清“相对 DeepSeek-V3 / Qwen3 改了什么”，kernel 用 C 级。

| 小节 | 内容 | B 级重点（其余为 C） |
|---|---|---|
| 8a 线性注意力与混合架构（38 个文件） | Qwen3-Next、Qwen3.5 的 Gated DeltaNet（GDN）；FLA 的 chunk / recurrent kernel；Mamba 的 causal conv1d | `qwen3_next.py`、`qwen3_next_mtp.py`、`qwen3_5.py`、`attention_gdn.py`、`attentions/gdn_attn.py`、`fla_ops/chunk.py`、`fla_ops/fused_recurrent.py` |
| 8b DeepSeek-V4 / V4.1（77 个文件） | 压缩稀疏注意力（CSA）、hash 路由、V4 kernel 集、V4.1 的分页 runtime 与 Engram | `deepseek_v4.py`、`deepseek_v4_mtp.py`、`deepseek_v41/{model,runtime,attention,moe}.py`、`attentions/deepseek_v4_attn.py`、`attentions/deepseek_v41/{backend,cache}.py`、`v4_kernels/paged_decode.py`（作为 Triton kernel 的阅读示范）、`engram/host.py` |
| 8c Kimi-K3 / MiniMax-M3 / GLM-5.x / Qwen4-exp（40 个文件） | KDA + MLA 混合、attention residual、MiniMax-M3 稀疏注意力与索引器 CP、GLM-5.x kpool、Qwen4-exp 的 QSA 与 hyper-connection | `kimi_k3/model.py`、`attentions/kimi_mla_gdn_attn.py`、`minimax_m3/model.py`、`model_ops/minimax_m3/sparse_attn.py`、`glm5_next.py`、`glm5_next/kpool.py`、`qwen4_exp.py`、`attentions/qwen4_exp_attn.py` |

**读完应能回答**
- GDN 线性注意力的状态是什么形状？prefill 用 chunk 算法、decode 用 recurrent 算法的原因是什么？
- DeepSeek-V4 的 KV 为什么要压缩？压缩后的行怎么分页存放、怎么被稀疏注意力读取？
- 选一个新模型，说出它相对 DeepSeek-V3 或 Qwen3 改了哪几个模块。

---

### 第 9 批：进阶执行与分布式 —— Mono 融合 decode、并行扩展、PD 分离与 KV offload

**目标**：理解 ATOM 在单机 TP 之外的执行方式：把一整层 decode 融合成一个 kernel，跨更多 GPU 和节点并行，以及把 KV 在节点间或 CPU / 存储之间搬运。

| 小节 | 内容 | B 级重点（其余为 C） |
|---|---|---|
| 9a Mono 融合 decode（115 个文件） | `atom/mono/` 的公共机制（TP 共识、peer memory、mailbox、按宽度构建、编译与 trace 校验）与各模型的 `mono/` 目录 | `mono/runtime/*`、`mono/plan/*`、`mono/weights/{layer,config}.py`、`mono/dsa_indexer/{runner,dispatch,plan}.py`、各模型的 `mono/{runner,dispatch,plan,contract}.py`、`deepseek_v41/mono/*_plan.py` |
| 9b 并行扩展（17 个文件） | 流水线并行（PP）与动态分块 prefill、decode / prefill 上下文并行（DCP / PCP）、two-batch overlap（TBO） | `pp_engine_core.py`、`distributed/pp_*.py`、`dynamic_chunking*.py`、`dcp_utils.py`、`pcp_utils.py`、`dcp_ops.py`、`tbo/ubatching.py`、`tbo/ubatch_splitting.py`、`utils/distributed/utils.py` |
| 9c PD 分离与 KV offload（73 个文件） | prefill / decode 分离（Mooncake、MoRI-IO 传输）；KV offload 到 CPU 或 Mooncake Store（dense / hybrid / mp 三种实现） | `disagg_types.py`、`disaggregation/{base,factory,types,aggregator}.py`、`mooncake_connector.py`、`moriio_connector.py`、`offload/{connector,config,_offload_common,chunked_scheduler}.py`、`offload/dense/connector.py` |

本批也回头补全第 2 至 5 批里标记为 `【分支】` 的入口：`PrefillEngineCore` / `DecodeEngineCore`、调度器的 `_connector_*`、BlockManager 的 offload 与 remote prefix。

**读完应能回答**
- Mono 把一层 decode 融合成一个 kernel，TP 的 all-reduce 是怎么在 kernel 内完成的？
- PP 下 head stage 和 downstream stage 的循环有什么不同？动态分块如何决定每块 prefill 的大小？
- PD 分离时，KV 从 prefill 节点传到 decode 节点经过哪些组件？调度器在哪里等待它到达？

---

### 第 10 批：对外集成与周边 —— 插件、RL rollout、可观测性、diffusion

**目标**：理解 ATOM 如何作为 vLLM / SGLang / RTP-LLM 的后端运行，以及 RL rollout、监控、压测、调试工具和独立的 diffusion 子系统。

| 小节 | 内容 | B 级重点（其余为 C） |
|---|---|---|
| 10a 插件公共层（5 个文件） | 插件模式的配置转换与注册 | `plugin/{register,prepare,config}.py` |
| 10b vLLM 插件（58 个文件） | platform 注册、模型包装、注意力适配，以及各模型的 bridge 与补丁 | `vllm/{register,platform,model_wrapper}.py`、`vllm/attention/{backend,metadata,layer_mha,layer_mla}.py` |
| 10c SGLang 插件（63 个文件） | 注意力 backend、模型包装、forward context 转换、TBO 适配 | `sglang/{register,prepare}.py`、`models/base_model_wrapper.py`、`full_attention/full_attention_backend.py`、`runtime/forward_context.py` |
| 10d RTP-LLM 插件（16 个文件） | 同上，面向 RTP-LLM | `models/base_model_wrapper.py`、`utils/forward_context.py`、`rtp_full_attention.py` |
| 10e RL rollout（7 个文件） | 给 RL 训练框架用的异步引擎、权重热更新、显存释放与恢复 | `async_engine.py`、`weight_updater.py`、`weight_sync.py` |
| 10f 可观测性与工具（23 个文件） | Prometheus 指标、tracing、serving 压测、dump / compare 调试工具 | `metrics/exporter.py`、`benchmarks/benchmark_serving.py`、`debug_helper/{dump,compare}.py` |
| 10g diffusion 子系统（29 个文件） | 视频 + 音频生成，和 LLM 引擎平级：无 KV cache，固定步数去噪，单请求序列并行（Ulysses） | `diffusion/{pipeline,attention,ulysses}.py`、`engine/*`、`entrypoints/diffusion_server.py`、`models/minimax_h3/{pipeline,denoise,dit}.py` |

**读完应能回答**
- 插件模式下，vLLM 的调度器和 KV 管理还在吗？ATOM 替换了哪一层？
- ATOM 的 forward context 如何从 vLLM / SGLang 的元数据转换而来？
- RL 训练每轮更新权重时，ATOM 如何在不重启的情况下换掉权重并释放 KV 显存？
- diffusion 子系统与 LLM 引擎在进程结构上有什么异同？

---

## 4. 跨批备注（注释过程中发现、后续批次要交叉引用的事实）

### 已核实的关键事实（第 1、2 批通读时统一，后续批次注释与之保持一致）

- **一步只做 prefill 或只做 decode**：基础调度器 `_schedule` 排了 prefill 就直接返回，不把 prefill chunk 和 decode 拼进同一个 batch（混合批在源码中仍是 TODO）；做到一半的 chunked prefill 下一步优先续算。想让 decode 插进 chunk 之间，需 `ATOM_PREFILL_DECODE_INTERVAL>0` 且存在 PrefillDelayer（`ATOM_ENABLE_PREFILL_DELAYER=1` 默认开、非 RapidServe）：DP>1 时每个 rank 总会创建（`DPEngineCoreProc` 调 `_init_prefill_delayer(config, dp_group)`），DP=1 时仅在 PP=1 且间隔>0 时创建本地模式（`engine_core.py`）。
- **CUDA Graph**：原生模式下 `--cudagraph-mode NONE` 仍走整图捕获（`_piecewise_cg_active()` 为 False → 手动 FULL 路径），真正关闭捕获的是 `--enforce-eager`（它不关 torch.compile）；FULL_AND_PIECEWISE 在原生模式下等同 PIECEWISE。
- **延迟输出**默认开启，条件是 `not is_pipeline_parallel and not hands_off_recurrent_state`（`model_runner.py`）。
- **原生多模态**：`get_hf_config` 把 `text_config` 当 hf_config，完整 config 存进 `_multimodal_config`；Qwen3.5、Kimi-K3 原生会构建视觉塔，Mistral3、Kimi-K2.5、GLM-5.3 原生只跑文本。
- **`num_cached_tokens`** 在 prefill 结束后停在 prompt 长度，decode 时不再增长（与 nano-vllm 不同）；`RequestOutput.num_cached_tokens` 取自 `prefix_cache_hit_tokens`。
- **采样**：只有不做 top-k / top-p 过滤时才用 argmax(p/E)；开过滤时走 aiter 采样算子或 `torch.multinomial`。ATOM 没有 repetition / presence / frequency penalty。
- **默认值**：HTTP 请求 `max_tokens` 默认 chat/completions 8192、messages/responses 4096；`index_cache_dtype` 默认 V4 为 fp4（gfx942、插件或有 KV 传输时 fp8）、V4.1 为 fp8、其余跟随 kv_cache_dtype。
- **MLA + DSA 模型（V3.2、GLM-5 等）的索引缓存只按 fp8 或 fp4 分配**：`aiter_mla.py` 的 `_declare_kv_pool` 写死 `index_dtype=fp8`，fp4 另由 `_indexer_fp4` 控制；`--index_cache_dtype bf16` 时实际仍按 fp8 分配。
- **工具调用解析器**只有 qwen、kimi、kimi_k3、glm、dsml、dsml_v41、minimax，没有 Hermes。
- **MI300X 规格**（估算用）：HBM ≈ 5.3 TB/s，FP8 稠密 ≈ 2600 TFLOPS，BF16 稠密 ≈ 1300 TFLOPS。
- 不存在的路径：`model_ops/indexer`（DSA 索引器在 `deepseek_v2.Indexer` 和 `model_ops/sparse_indexer_*.py`；GLM-5.3 的在 `model_ops/glm5_next/`）。

### 来自第 1 批

（这些备注整理于把分支换到 v0.1.8 之前，个别细节和行号在 v0.1.8 中可能不同，引用前先核对。）


- **`kv_cache_block_size` 会被按模型改写**：DeepSeek-V4 改成 256，GLM-5.3 改成 `index_kpool × 16`，V4.1 改成 128（`config.py` 中 `Config.__post_init__` 的第 12、13 步）。调度器、BlockManager、ModelRunner 看到的都是改写后的值；DCP 的 `interleave_size` 检查发生在改写之前。→ 第 3、4、5、8 批
- **`Config.num_kvcache_blocks` 初始为 -1**：EngineCore 用 `ModelRunner.get_num_blocks` 的结果回填（`engine_core.py`），之后才构造 Scheduler / BlockManager。→ 第 2、4、5 批
- **`Config.capture_sizes` 是声明的 CUDA Graph 档位**，`ModelRunner.capture_cudagraph` 会再去掉 token 预算永远达不到的档位；decode batch 超过最大档位时以 eager 运行（`ForwardMode.decide` 中的 `on_ladder`）。→ 第 5 批
- **`stop_token_ids`** 来自 generation_config，`LLMEngine.__init__` 再从中去掉 tokenizer 的 eos，调度器读取的是处理后的结果。→ 第 2、3 批
- **`Config.compute_hash`** 决定 torch.compile 缓存目录 `~/.cache/atom/torch_compile_cache/<hash>`（`utils/backends.py`），包含投机宽度（v0.1.8里不含 `max_model_len`、`max_num_seqs`）。→ 第 5 批
- **`--enable-dp-attention` 会改写配置**：CoreManager 在构造后把 `tensor_parallel_size` 设为 1、`data_parallel_size` 乘以原 TP（`engine_core_mgr.py`）。→ 第 2 批
- **`LLMEngine` 只把 Config 中存在的同名 kwargs 传给 Config**，EngineArgs 中其余字段被静默丢弃。→ 第 2 批
- **延迟输出的占位 token**：`Scheduler.postprocess` 先往 `token_ids` 末尾追加 `eos_token_id` 占位，下一步原地覆盖；`num_placeholder_tokens` 记录宽度，`preempt` 据此删除。→ 第 3、5 批
- **ATOM 的序列在第一次被调度做 prefill 时就变成 RUNNING**（`_schedule_prefill_seq`），chunked prefill 未完成的序列留在 running 中由 Phase 1 继续，这一点与 nano-vllm 不同。→ 第 3 批
- **`envs.py` 的值不缓存**：每次访问 `envs.X` 都重新读 `os.environ`；热路径通常在模块导入时把值存成常量，所以这类开关必须在启动前设置。→ 所有批次
- 已发现的“注释与代码不符”（均已在注释中用“解释：”标出，代码未改）：`multimodal_inference.py` 的 “Force eager mode” 实际只设置了 `cudagraph_capture_sizes="[1]"`；`cuda_graph_sizes` 的 docstring 描述的是 vLLM 的规则；`CUDAGraphMode.NONE` 在原生 runner 中没有单独的判断，真正关闭捕获的是 `enforce_eager`。

### 来自第 2 批

- **AsyncIOProc 的回包规则**（`async_proc.py` 的 `busy_loop` / `call_func`）：runner 方法返回 None 时不回包，`wait_out=True` 调用它会卡住；所有 `call_func` 共用一个 FIFO，不等结果的调用必须返回 None，否则下一个等待的调用会拿到它的返回值。→ 第 5 批
- **只有 TP rank 0 有主输出通道**：每个 rank 都执行 forward，但只有 rank 0 的结果回到 EngineCore，其余 rank 的返回值被丢弃。→ 第 5 批
- **进程数是 TP × PCP**（不乘 PP：PP 的每个 stage 是独立的 EngineCore）。→ 第 5、9 批
- **SIGTERM 的处理链**：AsyncIOProcManager 的 `init_exit_handler` 替换了 EngineCore 的 SIGTERM 处理，先走 `manager.exit()` 再走 EngineCore 的 `exit()`。→ 第 9 批
- **前缀 hash 链在前端计算**：`ship_prompt_hash_chain`（受 `ATOM_FRONTEND_PREFIX_HASH` 控制），hash block 大小 = block_size × DCP；`_per_req_cache_model_types` 是 `has_per_req_cache` 的唯一来源，ModelRunner 启动时会核对。→ 第 4 批
- **停止条件**：EOS 从 `stop_token_ids` 中去掉，所以调度器分别报告 "eos" 和 "stop_<id>"；stop 字符串用 `add_special_tokens=False` 分词后按 token 匹配，分词方式不同的文本匹配不到。→ 第 3 批
- **在线服务绕过了 `LLMEngine.add_request` / `postprocess`**，自己从 `io_processor.requests` 中取结果。→ 第 2c 节
- **调度器每步的输入输出**（`engine_core.py` 的 `_process_engine_step_inner`）：`schedule()` 返回 `(ScheduledBatch, {seq_id: Sequence})`，没有请求时返回 None；被拒绝的序列通过 `take_rejected()` 取出并作为已结束推送；只有 `req_ids` 非空时才调用 `postprocess(seqs.values(), fwd_out, stream_output_queue=…, batch=…)`；STREAM 输出总是排在已结束列表之前，CoreManager 的输出线程依赖这个顺序；新请求只在每轮循环开头通过 `scheduler.extend()` 进入调度器，调度器只被 busy_loop 线程访问。→ 第 3 批
- **EngineCore 启动时对 ModelRunner 的调用顺序**：`get_num_blocks` → `allocate_kv_cache(num_blocks)` → （可选）`profile_dynamic_chunking` → `capture_cudagraph`（返回 cost、sizes、pool_bytes）→ `freeze_gc_heap`；`num_kvcache_blocks` 只回填到 EngineCore 自己的 Config 副本，主进程的 Config 仍是 -1（所以 READY 消息要带上 `max_pool_tokens`）；每一步调用 `forward(batch)`（`wait_out=True`），拿到 rank 0 的 `ScheduledBatchOutput`。→ 第 5 批
- **DP dummy 执行**：只要任一 DP rank 有工作，空闲 rank 就调用 `dummy_execution`；全局变为空闲后还会多跑一步；DP 下 `PrefillDelayer` 挂在 gloo 组上，在 `schedule()` 内部做跨 DP 的 all_reduce。→ 第 3、5 批
- **`--enable-dp-attention` 的配置改写**（v0.1.8 `engine_core_mgr.py` 约 584-596 行）：在 `iter_dp_rank_assignments` 已经按原 `-tp` 算完之后原地修改共享 Config：`data_parallel_size *= tp`，`data_parallel_size_local` 改为本地引擎数，多机时 `data_parallel_rank *= tp`，`tensor_parallel_size = 1`。→ 第 7、9 批
- **在线服务的结果通道**：只走 `seq.stream_callback` + `_flush_stream_batch_fn`（由 api_server 接上）；DP 负载计数在第一次 STREAM 输出时释放 prefill token 的计费（`_mark_seq_prefill_complete`）；RL 使用 `collective_rpc`、`weight_sync_session` 和 FINISH/DISCARD 协议。在线路径上没有人调用 `get_output`，`outputs_queue` 中的已结束批次会一直堆积（疑似泄漏，注释中只陈述事实）。→ 第 10 批
- **流式结果从引擎线程交给事件循环的机制**（`api_server.py`）：回调只把 chunk 存进 `StreamBatchDispatcher` 的 `threading.local` 列表；一个 step 的回调结束后，输出线程调用 `flush_stream_batch`，每个事件循环做一次 `call_soon_threadsafe(_deliver, items)`；反分词在事件循环里的 `StreamOutputCollector.get()` 中完成。非流式请求不走 dispatcher，每次回调直接 `call_soon_threadsafe(token_queue.put_nowait, ...)`。api_server 把 `engine.core_mgr._flush_stream_batch_fn` 设为 `flush_stream_batch`，离线运行时它为 None。→ 第 10 批
- **客户端断开**：`cleanup_stream(aborted=True)` 或 `generate_async` 的 finally 调用 `core_mgr.abort_request`，向所有 DP rank 广播 `abort_request` 工具命令，调度器在下一步结束该序列。→ 第 3 批
- **请求长度上限**：`_validate_pool_capacity` 对超过 `CoreManager.max_pool_tokens`（各 rank READY 消息中的最小值）的 prompt 返回 400；`RequestOutput.num_cached_tokens` 来自 `seq.prefix_cache_hit_tokens`，最终出现在 `usage.prompt_tokens_details.cached_tokens`。→ 第 4、5 批
- **PD 分离时的 prompt 来源**：`protocol.resolve_prompt_token_ids` 从 `kv_transfer_params["prompt_token_ids"]` 取 prompt id 并跳过 chat template；`_engine_kv_transfer_params` 把它们剥掉后再交给引擎。→ 第 9 批
- **采样默认值**：chat / completions 用 `protocol.py` 里的默认值（1.0 / -1 / 1.0），不读 `generation_config`；messages / responses 会读。强制投机接受率时，每个 token 的文本被替换成 `"synthetic "`。→ 第 5、7 批
- **工具调用解析器与模型的对应**：各模型 parser 的文件头用【分支】标出了对应模型，注册名（各 parser 类的 `NAME`）为 `kimi_k3`、`kimi`、`dsml`、`dsml_v41`、`minimax`、`glm`、`qwen`；`kimi_k3_tokens.py` 被 `kimi_k3_tool_parser.py` 和 `reasoning_dialects.py` 共用；`multimodal/processing.py` 的 `expand_media_placeholders` 由 `kimi_k3_vl` 调用；DeepSeek-V4.1 是 `embedding_spans` 的唯一产生者。→ 第 8 批
- **多模态与前缀缓存**：`multimodal_cache_seed` 参与前缀缓存 hash 链的种子（`block_manager.py`）；`multimodal_runtime.py` 与 `ModelRunner.vision_embeddings` 及 `run_model` 中的 span 路径对应。→ 第 4、5 批
- **atomesh**：Rust 写的路由网关（同一份构建也是 PyO3 模块 `atomesh_runner`），`USE_ATOMESH_ENTRYPOINTS=1` 时启用；HTTP 由 Rust 处理，只有 chat / completions 进入 Python，没有 `/v1/messages`。→ 第 10 批
- **TTFT 的口径**：`request_timing.py` 统计的 TTFT 包含预处理时间。→ 第 10 批

### 来自第 3 批

- **基础调度器的 decode 批里 `is_first_decode_without_local_prefill` 恒为 False**：decode 循环在构造 `ScheduledBatch` 之前就把 `seq.is_first_decode` 清掉了（只有 RapidServe 的 `DecodeScheduler._schedule` 用 `_consume_first_decode` 在构造后才清）。读这个字段的注意力代码（如 `aiter_mla.py`）在基础路径上不会触发相应分支。→ 第 6、7、9 批
- **投机解码一步多 token 时的判停**：stop 字符串从最新 token 往前扫，命中后用 `for…else` 跳过 eos 检查，所以同一步更早出现的 eos 可能留在输出里；eos 与 `stop_token_ids` 取的是最早位置。→ 第 7 批
- **延迟输出的占位 token 就是 eos id**，下一步原地覆盖，所以"结束"要晚一步才发现；投机解码稳态末尾有 2k+1 个占位（`num_placeholder_tokens`）。→ 第 5、7 批
- **PrefillDelayer**：DP>1 时每个 rank 总会创建（跨 DP 对齐模式），DP=1 时仅在 PP=1 且 `ATOM_PREFILL_DECODE_INTERVAL>0` 时创建（本地模式）；`_first` 不改变任何决策（与 SGLang 的 `skip_first_delayer` 语义不同）。→ 第 9 批
- **RapidServe 的 PrefillScheduler / DecodeScheduler** 有若干按代码推断的问题（超长 prompt 永久堵住、abort 对部分队列不生效、不复用前缀缓存、被抢占后不会再 prefill），均以"观察："写在 scheduler.py 中，读第 5、9 批时再核对。
