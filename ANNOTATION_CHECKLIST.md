# ATOM 注释清单（逐文件）

由脚本根据 [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md) 的分批规则生成，覆盖 `atom/` 下全部 787 个 .py 文件（344,268 行）。注释完一个文件就勾选一项。

深度：**A** 逐行详注；**B** 精注；**C** 导读；**—** 空文件或只有几行导出，不注释；**不改** 上游 vendored 代码。定义见计划第 1 节。


## 第 1 批：入口、配置与请求对象（17 个文件，8,065 行）


### 1 入口、配置与请求对象

- [x] **A** `atom/examples/simple_inference.py`（96 行）
- [x] **A** `atom/model_engine/arg_utils.py`（923 行）
- [x] **A** `atom/config.py`（2866 行）
- [x] **A** `atom/sampling_params.py`（30 行）
- [x] **A** `atom/model_engine/sequence.py`（633 行）
- [x] **A** `atom/model_engine/request.py`（17 行）
- [x] **B** `atom/__init__.py`（33 行）
- [x] **B** `atom/examples/multimodal_inference.py`（119 行）
- [x] **B** `atom/utils/envs.py`（1240 行）
- [x] **B** `atom/utils/arg_parser.py`（44 行）
- [x] **B** `atom/utils/__init__.py`（1091 行）
- [x] **B** `atom/model_config/kimi_k25.py`（175 行）
- [x] **B** `atom/model_config/qwen3_5.py`（195 行）
- [x] **B** `atom/model_config/qwen3_5_moe.py`（207 行）
- [x] **B** `atom/model_config/qwen3_next.py`（277 行）
- [x] **C** `atom/examples/profile_offline.py`（119 行）
- [x] — `atom/model_config/__init__.py`（0 行）

## 第 2 批：引擎主循环、多进程架构与服务层（57 个文件，23,220 行）


### 2a 离线引擎与多进程

- [x] **A** `atom/model_engine/llm_engine.py`（953 行）
- [x] **A** `atom/model_engine/engine_core_mgr.py`（2112 行）
- [x] **A** `atom/model_engine/engine_core.py`（1507 行）
- [x] **A** `atom/model_engine/async_proc.py`（1002 行）
- [x] **B** `atom/model_engine/engine_core_protocol.py`（35 行）
- [x] **B** `atom/model_engine/ipc_utils.py`（180 行）
- [x] **B** `atom/model_engine/engine_utility.py`（850 行）
- [x] **B** `atom/utils/network.py`（43 行）
- [x] **B** `atom/utils/numa_utils.py`（176 行）
- [x] **B** `atom/utils/gc_utils.py`（345 行）
- [x] **C** `atom/model_engine/collective_rpc.py`（204 行）
- [x] **C** `atom/model_engine/capabilities.py`（128 行）
- [x] **C** `atom/model_engine/engine_stats.py`（795 行）
- [x] **C** `atom/model_engine/run_labels.py`（105 行）

### 2b 分词与多模态输入

- [x] **B** `atom/model_engine/multimodal_runtime.py`（139 行）
- [x] **B** `atom/multimodal/__init__.py`（33 行）
- [x] **B** `atom/multimodal/inputs.py`（103 行）
- [x] **B** `atom/multimodal/media/__init__.py`（69 行）
- [x] **B** `atom/multimodal/media/connector.py`（36 行）
- [x] **B** `atom/multimodal/media/image.py`（31 行）
- [x] **B** `atom/multimodal/parse.py`（66 行）
- [x] **B** `atom/multimodal/processing.py`（122 行）
- [x] **B** `atom/multimodal/registry.py`（105 行）

### 2c OpenAI 兼容服务层

- [x] **A** `atom/entrypoints/openai/api_server.py`（3092 行）
- [x] **A** `atom/entrypoints/openai_server.py`（27 行）
- [x] **B** `atom/entrypoints/chat_utils.py`（104 行）
- [x] **B** `atom/entrypoints/openai/__init__.py`（17 行）
- [x] **B** `atom/entrypoints/openai/chat_encoder_adapters.py`（175 行）
- [x] **B** `atom/entrypoints/openai/chat_encoders.py`（312 行）
- [x] **B** `atom/entrypoints/openai/protocol.py`（409 行）
- [x] **B** `atom/entrypoints/openai/serving_chat.py`（720 行）
- [x] **B** `atom/entrypoints/openai/serving_completion.py`（316 行）
- [x] **B** `atom/entrypoints/openai/sse.py`（55 行）
- [x] **B** `atom/entrypoints/openai/streaming_dispatch.py`（642 行）
- [x] **C** `atom/entrypoints/atomesh/atom_standalone_service.py`（1727 行）
- [x] **C** `atom/entrypoints/atomesh/server.py`（327 行）
- [x] **C** `atom/entrypoints/openai/deepseek_v41_encoder.py`（41 行）
- [x] **C** `atom/entrypoints/openai/kimi_k3_tokens.py`（97 行）
- [x] **C** `atom/entrypoints/openai/marker_scanner.py`（232 行）
- [x] **C** `atom/entrypoints/openai/metrics_setup.py`（28 行）
- [x] **C** `atom/entrypoints/openai/reasoning.py`（340 行）
- [x] **C** `atom/entrypoints/openai/reasoning_dialects.py`（411 行）
- [x] **C** `atom/entrypoints/openai/request_timing.py`（133 行）
- [x] **C** `atom/entrypoints/openai/serving_anthropic.py`（615 行）
- [x] **C** `atom/entrypoints/openai/serving_responses.py`（1097 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/__init__.py`（81 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/deepseekv41_tool_parser.py`（38 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/deepseekv4_tool_parser.py`（370 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/glm_tool_parser.py`（169 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/kimi_k3_tool_parser.py`（282 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/kimi_tool_parser.py`（196 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/minimax_tool_parser.py`（213 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/qwen3_tool_parser.py`（189 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/registry.py`（247 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/schema.py`（106 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/stream.py`（786 行）
- [x] **C** `atom/entrypoints/openai/tool_parser/tool_parser.py`（487 行）

## 第 3 批：调度器（2 个文件，5,885 行）


### 3 调度器

- [x] **A** `atom/model_engine/scheduler.py`（5346 行）
- [x] **B** `atom/model_engine/prefill_delayer.py`（539 行）

## 第 4 批：KV cache 账本与状态缓存（13 个文件，8,552 行）


### 4a KV block 账本与前缀缓存

- [ ] **A** `atom/model_engine/block_manager.py`（3372 行）
- [ ] **A** `atom/model_engine/block_pool.py`（523 行）
- [ ] **A** `atom/model_engine/kv_block.py`（44 行）
- [ ] **A** `atom/model_engine/prefix_hash.py`（111 行）
- [ ] **B** `atom/model_engine/block_table_codec.py`（220 行）
- [ ] **B** `atom/utils/block_tables.py`（233 行）
- [ ] **B** `atom/utils/block_convert.py`（380 行）
- [ ] **B** `atom/distributed/kv_events.py`（549 行）

### 4b 递归状态缓存（GDN / Mamba / KDA 等混合模型）

- [ ] **B** `atom/model_engine/state_cache.py`（155 行）
- [ ] **B** `atom/model_engine/state_offload.py`（445 行）
- [ ] **B** `atom/model_engine/state_pool.py`（803 行）
- [ ] **B** `atom/model_engine/state_runtime.py`（348 行）
- [ ] **B** `atom/model_engine/page_unit_checkpoint.py`（1369 行）

## 第 5 批：ModelRunner：执行、权重加载、采样、CUDA Graph 与编译（43 个文件，17,801 行）


### 5a ModelRunner 与 ForwardContext

- [ ] **A** `atom/model_engine/model_runner.py`（4978 行）
- [ ] **A** `atom/utils/forward_context.py`（1332 行）
- [ ] **B** `atom/utils/h2d.py`（486 行）
- [ ] **B** `atom/utils/packed_h2d.py`（92 行）
- [ ] **B** `atom/model_ops/decode_input_ids.py`（194 行）

### 5b KV pool 内存布局与 token 索引

- [ ] **B** `atom/model_ops/attentions/pool_layout/__init__.py`（47 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/entry_arena.py`（584 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/mha_page_unit_geometry.py`（193 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/page_unit_geometry.py`（250 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/paged_state_copy.py`（387 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/pool_rows.py`（100 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/sub_pool_spec.py`（246 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/v41_pool_geometry.py`（411 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/v4_pool_fields.py`（132 行）
- [ ] **B** `atom/model_ops/attentions/pool_layout/v4_pool_geometry.py`（667 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/__init__.py`（30 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/batch_ids.py`（75 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/decode.py`（39 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/gather.py`（49 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/prefill.py`（34 行）
- [ ] **B** `atom/model_ops/attentions/token_layout/slots.py`（70 行）

### 5c 权重加载

- [ ] **A** `atom/model_loader/loader.py`（499 行）
- [ ] **A** `atom/model_loader/loading_core.py`（467 行）
- [ ] **A** `atom/model_loader/weight_iterator.py`（378 行）
- [ ] **B** `atom/model_loader/expert_staging.py`（386 行）
- [ ] **B** `atom/model_loader/online_quant_streaming.py`（515 行）
- [ ] **B** `atom/model_loader/weight_dispatch.py`（301 行）
- [ ] **B** `atom/model_loader/weight_names.py`（274 行）
- [ ] **B** `atom/model_loader/weight_utils.py`（203 行）
- [ ] **C** `atom/model_loader/deepseek_v41.py`（18 行）

### 5d 采样

- [ ] **A** `atom/model_ops/sampler.py`（371 行）
- [ ] **B** `atom/model_ops/lm_head_argmax.py`（87 行）

### 5e CUDA Graph 与 torch.compile

- [ ] **A** `atom/utils/decorators.py`（591 行）
- [ ] **A** `atom/utils/backends.py`（775 行）
- [ ] **A** `atom/utils/cuda_piecewise_backend.py`（115 行）
- [ ] **B** `atom/utils/graph_holders.py`（57 行）
- [ ] **B** `atom/utils/custom_register.py`（42 行）
- [ ] **B** `atom/model_ops/module_dispatch_ops.py`（270 行）
- [ ] **B** `atom/utils/cuda_graph.py`（470 行）
- [ ] **B** `atom/utils/attn_ffn_piecewise.py`（317 行）
- [ ] **C** `atom/utils/compiler_inferface.py`（765 行）
- [ ] **C** `atom/utils/graph_marker.py`（44 行）
- [ ] **C** `atom/utils/graph_marker_instrumentation.py`（460 行）

## 第 6 批：基础算子、量化、注意力与 dense 模型（34 个文件，16,983 行）


### 6a 基础算子与张量并行

- [ ] **A** `atom/model_ops/linear.py`（2641 行）
- [ ] **A** `atom/model_ops/embed_head.py`（496 行）
- [ ] **A** `atom/model_ops/layernorm.py`（1462 行）
- [ ] **A** `atom/model_ops/rotary_embedding.py`（165 行）
- [ ] **A** `atom/model_ops/activation.py`（112 行）
- [ ] **A** `atom/model_ops/communication_op.py`（58 行）
- [ ] **B** `atom/model_ops/__init__.py`（19 行）
- [ ] **B** `atom/model_ops/base_config.py`（55 行）
- [ ] **B** `atom/model_ops/utils.py`（338 行）
- [ ] **B** `atom/model_ops/triton_rmsnorm_nw.py`（60 行）

### 6b 量化

- [ ] **A** `atom/quant_spec.py`（807 行）
- [ ] **B** `atom/quantization/quark/utils.py`（758 行）
- [ ] **B** `atom/model_ops/blockscale.py`（319 行）
- [ ] **B** `atom/model_ops/blockscale_kernels/blockscale_gemm.py`（86 行）
- [ ] **B** `atom/model_ops/blockscale_kernels/fp8.py`（355 行）
- [ ] **B** `atom/model_ops/blockscale_kernels/quantization.py`（93 行）
- [ ] **B** `atom/model_ops/triton_fused_qkv_quant.py`（472 行）
- [ ] — `atom/quantization/quark/__init__.py`（0 行）
- [ ] — `atom/model_ops/blockscale_kernels/__init__.py`（2 行）

### 6c 注意力（MHA）

- [ ] **A** `atom/utils/selector.py`（171 行）
- [ ] **A** `atom/model_ops/base_attention.py`（1105 行）
- [ ] **A** `atom/model_ops/paged_attention.py`（124 行）
- [ ] **A** `atom/model_ops/attention_mha.py`（1756 行）
- [ ] **A** `atom/model_ops/attentions/backends.py`（1097 行）
- [ ] **A** `atom/model_ops/attentions/aiter_attention.py`（1579 行）
- [ ] **A** `atom/model_ops/attentions/mha_kv_pool.py`（296 行）
- [ ] **B** `atom/model_ops/attentions/triton_mha.py`（120 行）
- [ ] **B** `atom/model_ops/attentions/triton_merge_attn_states.py`（206 行）

### 6d dense 模型

- [ ] **A** `atom/models/utils.py`（559 行）
- [ ] **A** `atom/models/qwen3.py`（371 行）
- [ ] **B** `atom/models/llama.py`（516 行）
- [ ] **B** `atom/models/mistral3.py`（96 行）
- [ ] **B** `atom/models/eagle3_llama.py`（495 行）
- [ ] **B** `atom/model_ops/fused_aux_rmsnorm.py`（194 行）

## 第 7 批：MoE、MLA、投机解码：DeepSeek-V3 与 MoE 模型家族（51 个文件，36,441 行）


### 7a MoE 与专家并行

- [ ] **A** `atom/model_ops/moe.py`（5383 行）
- [ ] **A** `atom/model_ops/topK.py`（527 行）
- [ ] **A** `atom/models/qwen3_moe.py`（540 行）
- [ ] **B** `atom/model_ops/fused_moe_triton.py`（1114 行）
- [ ] **B** `atom/model_ops/fused_moe/comm_fused_moe.py`（479 行）
- [ ] **B** `atom/model_ops/fused_moe/config.py`（401 行）
- [ ] **B** `atom/model_ops/fused_moe/expert_layout.py`（245 行）
- [ ] **B** `atom/model_ops/fused_moe/flydsl_mega_experts.py`（452 行）
- [ ] **B** `atom/model_ops/fused_moe/modular_kernel.py`（649 行）
- [ ] **B** `atom/model_ops/fused_moe/mori_prepare_finalize.py`（470 行）
- [ ] **B** `atom/model_ops/fused_moe/mori_v2_prepare_finalize.py`（924 行）
- [ ] **B** `atom/model_ops/fused_moe/rccl_prepare_finalize.py`（539 行）
- [ ] **B** `atom/model_ops/fused_moe/routed_all2all.py`（261 行）
- [ ] **B** `atom/model_ops/fused_moe/shared_expert_dispatch.py`（159 行）
- [ ] **B** `atom/model_ops/fused_moe/triton_zero_pad_rows.py`（39 行）
- [ ] **B** `atom/model_ops/fused_moe/utils.py`（110 行）
- [ ] **B** `atom/model_ops/eplb.py`（2760 行）
- [ ] **B** `atom/models/mixtral.py`（390 行）

### 7b MLA 与 DeepSeek-V3

- [ ] **A** `atom/model_ops/attention_mla.py`（3675 行）
- [ ] **A** `atom/model_ops/attentions/aiter_mla.py`（3420 行）
- [ ] **A** `atom/models/deepseek_v2.py`（3593 行）
- [ ] **A** `atom/models/deepseek_mtp.py`（533 行）
- [ ] **B** `atom/model_ops/attentions/triton_mla.py`（165 行）
- [ ] **B** `atom/model_ops/triton_fused_mla_ctx_kv.py`（221 行）
- [ ] **B** `atom/model_ops/attentions/mla_kv_pool.py`（139 行）
- [ ] **B** `atom/model_ops/fused_mtp_prologue.py`（163 行）
- [ ] **B** `atom/models/eagle3_deepseek_mla.py`（399 行）

### 7c 投机解码框架（MTP / EAGLE / DSpark）

- [ ] **A** `atom/spec_decode/drafter.py`（735 行）
- [ ] **A** `atom/spec_decode/eagle_proposer.py`（876 行）
- [ ] **A** `atom/model_ops/rejection_sampler.py`（545 行）
- [ ] **B** `atom/spec_decode/calibration.py`（67 行）
- [ ] **B** `atom/spec_decode/draft_graph.py`（313 行）
- [ ] **B** `atom/spec_decode/draft_kv.py`（204 行）
- [ ] **B** `atom/spec_decode/dspark_proposer.py`（1099 行）
- [ ] **B** `atom/spec_decode/dspark_scheduler.py`（449 行）
- [ ] **B** `atom/spec_decode/dspark_verify.py`（244 行）
- [ ] **B** `atom/spec_decode/factory.py`（16 行）
- [ ] **B** `atom/model_ops/dspark_markov_sample.py`（531 行）
- [ ] **B** `atom/models/dspark_draft.py`（124 行）
- [ ] — `atom/spec_decode/__init__.py`（0 行）

### 7d DSA 稀疏注意力索引器（V3.2 / GLM-5）

- [ ] **B** `atom/model_ops/sparse_indexer_chunk.py`（95 行）
- [ ] **B** `atom/model_ops/sparse_indexer_fp4.py`（216 行）
- [ ] **B** `atom/model_ops/fp4_mqa_ragged_metadata.py`（54 行）

### 7e 其他 MoE 模型

- [ ] **B** `atom/models/glm4_moe.py`（669 行）
- [ ] **B** `atom/models/glm4_moe_mtp.py`（259 行）
- [ ] **B** `atom/models/gpt_oss.py`（492 行）
- [ ] **B** `atom/model_ops/swiglu_oai.py`（104 行）
- [ ] **B** `atom/models/minimax_m2.py`（529 行）
- [ ] **B** `atom/models/mimo_v2.py`（600 行）
- [ ] **B** `atom/models/mimo_v2_mtp.py`（368 行）
- [ ] **B** `atom/models/kimi_k25.py`（102 行）

## 第 8 批：新一代模型：线性注意力、DeepSeek-V4 / V4.1 及其他（155 个文件，72,555 行）


### 8a 线性注意力与混合架构（Qwen3-Next / Qwen3.5）

- [ ] **B** `atom/model_ops/attention_gdn.py`（671 行）
- [ ] **B** `atom/model_ops/attentions/gdn_attn.py`（2085 行）
- [ ] **B** `atom/model_ops/fla_ops/chunk.py`（390 行）
- [ ] **B** `atom/model_ops/fla_ops/fused_recurrent.py`（637 行）
- [ ] **B** `atom/models/qwen3_next.py`（1064 行）
- [ ] **B** `atom/models/qwen3_next_mtp.py`（234 行）
- [ ] **B** `atom/models/qwen3_5.py`（929 行）
- [ ] **C** `atom/model_ops/fla_ops/__init__.py`（21 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_delta_h.py`（418 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_delta_h_vk.py`（421 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_fused.py`（569 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_o.py`（250 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_o_vk.py`（227 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_scaled_dot_kkt.py`（159 行）
- [ ] **C** `atom/model_ops/fla_ops/chunk_vk.py`（273 行）
- [ ] **C** `atom/model_ops/fla_ops/cumsum.py`（265 行）
- [ ] **C** `atom/model_ops/fla_ops/fused_cumsum_kkt.py`（150 行）
- [ ] **C** `atom/model_ops/fla_ops/fused_merge_recompute.py`（426 行）
- [ ] **C** `atom/model_ops/fla_ops/fused_sigmoid_gating.py`（316 行）
- [ ] **C** `atom/model_ops/fla_ops/gdn_flydsl.py`（337 行）
- [ ] **C** `atom/model_ops/fla_ops/index.py`（41 行）
- [ ] **C** `atom/model_ops/fla_ops/l2norm.py`（157 行）
- [ ] **C** `atom/model_ops/fla_ops/layernorm_guard.py`（411 行）
- [ ] **C** `atom/model_ops/fla_ops/op.py`（65 行）
- [ ] **C** `atom/model_ops/fla_ops/replayssm.py`（1671 行）
- [ ] **C** `atom/model_ops/fla_ops/solve_tril.py`（585 行）
- [ ] **C** `atom/model_ops/fla_ops/state_checkpoint.py`（188 行）
- [ ] **C** `atom/model_ops/fla_ops/utils.py`（195 行）
- [ ] **C** `atom/model_ops/fla_ops/wy_fast.py`（159 行）
- [ ] **C** `atom/model_ops/mamba_ops/causal_conv1d.py`（1410 行）
- [ ] **C** `atom/model_ops/split_chunk.py`（557 行）
- [ ] **C** `atom/model_ops/triton_fused_qkv_norm_rope_cache.py`（447 行）
- [ ] **C** `atom/model_ops/triton_fused_sigmoid_mul_quant.py`（495 行）
- [ ] **C** `atom/model_ops/triton_gemma_rmsnorm.py`（142 行）
- [ ] **C** `atom/model_ops/triton_mrope.py`（319 行）
- [ ] **C** `atom/models/qwen3_5_mtp.py`（225 行）
- [ ] **C** `atom/models/qwen3_5_vl.py`（446 行）
- [ ] — `atom/model_ops/mamba_ops/__init__.py`（0 行）

### 8b DeepSeek-V4 / V4.1

- [ ] **B** `atom/models/deepseek_v4.py`（5216 行）
- [ ] **B** `atom/models/deepseek_v4_mtp.py`（341 行）
- [ ] **B** `atom/models/deepseek_v41/attention.py`（450 行）
- [ ] **B** `atom/models/deepseek_v41/model.py`（404 行）
- [ ] **B** `atom/models/deepseek_v41/moe.py`（137 行）
- [ ] **B** `atom/models/deepseek_v41/runtime.py`（431 行）
- [ ] **B** `atom/model_ops/v4_kernels/paged_decode.py`（1300 行）
- [ ] **B** `atom/model_ops/attentions/deepseek_v41/backend.py`（1211 行）
- [ ] **B** `atom/model_ops/attentions/deepseek_v41/cache.py`（910 行）
- [ ] **B** `atom/model_ops/attentions/deepseek_v4_attn.py`（5003 行）
- [ ] **B** `atom/model_ops/engram/host.py`（505 行）
- [ ] **C** `atom/models/deepseek_v4_dspark.py`（1423 行）
- [ ] **C** `atom/models/deepseek_v41/bounded_replay.py`（386 行）
- [ ] **C** `atom/models/deepseek_v41/config.py`（427 行）
- [ ] **C** `atom/models/deepseek_v41/dspark.py`（627 行）
- [ ] **C** `atom/models/deepseek_v41/image_processing.py`（204 行）
- [ ] **C** `atom/models/deepseek_v41/layers.py`（19 行）
- [ ] **C** `atom/models/deepseek_v41/multimodal.py`（56 行）
- [ ] **C** `atom/models/deepseek_v41/vision.py`（148 行）
- [ ] **C** `atom/models/deepseek_v41/weights.py`（416 行）
- [ ] **C** `atom/model_ops/v4_kernels/__init__.py`（160 行）
- [ ] **C** `atom/model_ops/v4_kernels/compress_plan.py`（398 行）
- [ ] **C** `atom/model_ops/v4_kernels/csa_translate_pack.py`（319 行）
- [ ] **C** `atom/model_ops/v4_kernels/dspark_fp8_indices.py`（268 行）
- [ ] **C** `atom/model_ops/v4_kernels/fused_compress.py`（853 行）
- [ ] **C** `atom/model_ops/v4_kernels/hca_persist.py`（217 行）
- [ ] **C** `atom/model_ops/v4_kernels/indexer_weights.py`（63 行）
- [ ] **C** `atom/model_ops/v4_kernels/inverse_rope.py`（123 行）
- [ ] **C** `atom/model_ops/v4_kernels/paged_2src.py`（730 行）
- [ ] **C** `atom/model_ops/v4_kernels/paged_decode_gluon.py`（218 行）
- [ ] **C** `atom/model_ops/v4_kernels/paged_decode_indices.py`（689 行）
- [ ] **C** `atom/model_ops/v4_kernels/paged_prefill_indices.py`（521 行）
- [ ] **C** `atom/model_ops/v4_kernels/pool_index.py`（118 行）
- [ ] **C** `atom/model_ops/v4_kernels/qk_norm_rope_maybe_quant.py`（836 行）
- [ ] **C** `atom/model_ops/v4_kernels/state_writes.py`（1006 行）
- [ ] **C** `atom/model_ops/v4_kernels/v4_quant.py`（549 行）
- [ ] **C** `atom/model_ops/deepseek_v41/candidate_table.py`（225 行）
- [ ] **C** `atom/model_ops/deepseek_v41/compressor.py`（170 行）
- [ ] **C** `atom/model_ops/deepseek_v41/draft_block.py`（45 行）
- [ ] **C** `atom/model_ops/deepseek_v41/dspark.py`（493 行）
- [ ] **C** `atom/model_ops/deepseek_v41/index_plane.py`（39 行）
- [ ] **C** `atom/model_ops/deepseek_v41/index_write.py`（241 行）
- [ ] **C** `atom/model_ops/deepseek_v41/indexer.py`（282 行）
- [ ] **C** `atom/model_ops/deepseek_v41/mhc.py`（240 行）
- [ ] **C** `atom/model_ops/deepseek_v41/mhc_pre_delayed.py`（135 行）
- [ ] **C** `atom/model_ops/deepseek_v41/paged_scoring.py`（412 行）
- [ ] **C** `atom/model_ops/deepseek_v41/projections.py`（39 行）
- [ ] **C** `atom/model_ops/deepseek_v41/rope_window.py`（228 行）
- [ ] **C** `atom/model_ops/deepseek_v41/rotary.py`（159 行）
- [ ] **C** `atom/model_ops/deepseek_v41/router.py`（188 行）
- [ ] **C** `atom/model_ops/deepseek_v41/score_workspace.py`（266 行）
- [ ] **C** `atom/model_ops/deepseek_v41/unit_table.py`（90 行）
- [ ] **C** `atom/model_ops/sparse_attn_v4.py`（644 行）
- [ ] **C** `atom/model_ops/quant_v4.py`（309 行）
- [ ] **C** `atom/model_ops/v4_backend_gate.py`（46 行）
- [ ] **C** `atom/model_ops/triton_hash_topk.py`（119 行）
- [ ] **C** `atom/model_ops/triton_mm_topk.py`（269 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/checkpoints.py`（155 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/indices.py`（433 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/metadata.py`（370 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/packed_attention.py`（81 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/packed_rows.py`（152 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/score_planner.py`（70 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v41/speculative.py`（117 行）
- [ ] **C** `atom/model_ops/attentions/deepseek_v4_replay.py`（111 行）
- [ ] **C** `atom/model_ops/engram/__init__.py`（17 行）
- [ ] **C** `atom/model_ops/engram/device/__init__.py`（6 行）
- [ ] **C** `atom/model_ops/engram/device/gate.py`（116 行）
- [ ] **C** `atom/model_ops/engram/device/hashing.py`（562 行）
- [ ] **C** `atom/model_ops/engram/device/layer.py`（160 行）
- [ ] **C** `atom/model_ops/engram/device/runtime.py`（284 行）
- [ ] **C** `atom/model_ops/engram/device/staging.py`（320 行）
- [ ] **C** `atom/model_ops/engram/device/uva.py`（122 行）
- [ ] **C** `atom/model_ops/engram/mapping.py`（391 行）
- [ ] **C** `atom/model_ops/engram/tables.py`（192 行）
- [ ] — `atom/models/deepseek_v41/__init__.py`（2 行）
- [ ] — `atom/model_ops/deepseek_v41/__init__.py`（2 行）

### 8c Kimi-K3 / MiniMax-M3 / GLM-5.x / Qwen4-exp

- [ ] **B** `atom/models/kimi_k3/model.py`（2038 行）
- [ ] **B** `atom/model_ops/attentions/kimi_mla_gdn_attn.py`（546 行）
- [ ] **B** `atom/models/minimax_m3/model.py`（948 行）
- [ ] **B** `atom/model_ops/minimax_m3/sparse_attn.py`（1549 行）
- [ ] **B** `atom/models/glm5_next.py`（1259 行）
- [ ] **B** `atom/model_ops/glm5_next/kpool.py`（803 行）
- [ ] **B** `atom/models/qwen4_exp.py`（1084 行）
- [ ] **B** `atom/model_ops/attentions/qwen4_exp_attn.py`（734 行）
- [ ] **C** `atom/models/kimi_k3/__init__.py`（26 行）
- [ ] **C** `atom/models/kimi_k3_dspark.py`（804 行）
- [ ] **C** `atom/models/kimi_k3_vl.py`（564 行）
- [ ] **C** `atom/model_ops/kimi_k3/__init__.py`（24 行）
- [ ] **C** `atom/model_ops/kimi_k3/activations.py`（472 行）
- [ ] **C** `atom/model_ops/kimi_k3/attention_residual.py`（573 行）
- [ ] **C** `atom/model_ops/kimi_k3/kda_state.py`（90 行）
- [ ] **C** `atom/model_ops/kimi_k3/quant.py`（142 行）
- [ ] **C** `atom/model_ops/attention_residual.py`（220 行）
- [ ] **C** `atom/models/minimax_m3/__init__.py`（33 行）
- [ ] **C** `atom/model_ops/minimax_m3/index_topk.py`（1325 行）
- [ ] **C** `atom/model_ops/minimax_m3/indexer_candidate_exchange.py`（329 行）
- [ ] **C** `atom/model_ops/minimax_m3/indexer_context_parallel.py`（226 行）
- [ ] **C** `atom/distributed/indexer_cp.py`（225 行）
- [ ] **C** `atom/models/glm5_next_mtp.py`（174 行）
- [ ] **C** `atom/model_ops/glm5_next/geometry.py`（94 行）
- [ ] **C** `atom/model_ops/glm5_next/indexer.py`（434 行）
- [ ] **C** `atom/model_ops/glm5_next/speculative.py`（351 行）
- [ ] **C** `atom/models/glm52/__init__.py`（24 行）
- [ ] **C** `atom/models/qwen4_exp_mtp.py`（162 行）
- [ ] **C** `atom/model_ops/qwen4_exp/hyperconnection.py`（195 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/gated.py`（320 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/hc_fused.py`（552 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/moe_decode.py`（213 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/ple.py`（553 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/qsa.py`（1604 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ops/qsa_flydsl_layout.py`（531 行）
- [ ] **C** `atom/model_ops/qwen4_exp/ple_layer.py`（533 行）
- [ ] **C** `atom/model_ops/qwen4_exp/qsa_attention.py`（457 行）
- [ ] — `atom/model_ops/glm5_next/__init__.py`（2 行）
- [ ] — `atom/model_ops/qwen4_exp/__init__.py`（0 行）
- [ ] — `atom/model_ops/qwen4_exp/ops/__init__.py`（3 行）

## 第 9 批：进阶执行与分布式：Mono 融合 decode、并行扩展、PD 分离与 KV offload（205 个文件，65,405 行）


### 9a Mono 融合 decode

- [ ] **B** `atom/mono/plan/__init__.py`（5 行）
- [ ] **B** `atom/mono/plan/arith.py`（23 行）
- [ ] **B** `atom/mono/plan/build_key.py`（70 行）
- [ ] **B** `atom/mono/plan/check.py`（101 行）
- [ ] **B** `atom/mono/plan/execution.py`（20 行）
- [ ] **B** `atom/mono/plan/layout.py`（19 行）
- [ ] **B** `atom/mono/plan/shard.py`（65 行）
- [ ] **B** `atom/mono/plan/tiles.py`（57 行）
- [ ] **B** `atom/mono/plan/trace.py`（73 行）
- [ ] **B** `atom/mono/runtime/__init__.py`（6 行）
- [ ] **B** `atom/mono/runtime/abi.py`（50 行）
- [ ] **B** `atom/mono/runtime/aux_taps.py`（69 行）
- [ ] **B** `atom/mono/runtime/compare.py`（15 行）
- [ ] **B** `atom/mono/runtime/compile.py`（174 行）
- [ ] **B** `atom/mono/runtime/consensus.py`（76 行）
- [ ] **B** `atom/mono/runtime/debug.py`（58 行）
- [ ] **B** `atom/mono/runtime/deployment.py`（120 行）
- [ ] **B** `atom/mono/runtime/hip_ipc.py`（260 行）
- [ ] **B** `atom/mono/runtime/lifecycle.py`（20 行）
- [ ] **B** `atom/mono/runtime/mailboxes.py`（44 行）
- [ ] **B** `atom/mono/runtime/peer_memory.py`（264 行）
- [ ] **B** `atom/mono/runtime/route.py`（70 行）
- [ ] **B** `atom/mono/runtime/step_begin.py`（216 行）
- [ ] **B** `atom/mono/runtime/timeline.py`（42 行）
- [ ] **B** `atom/mono/runtime/widths.py`（82 行）
- [ ] **B** `atom/mono/weights/config.py`（244 行）
- [ ] **B** `atom/mono/weights/layer.py`（456 行）
- [ ] **B** `atom/models/deepseek_v41/mono/attention_plan.py`（52 行）
- [ ] **B** `atom/models/deepseek_v41/mono/contract.py`（157 行）
- [ ] **B** `atom/models/deepseek_v41/mono/dispatch.py`（139 行）
- [ ] **B** `atom/models/deepseek_v41/mono/index_plan.py`（307 行）
- [ ] **B** `atom/models/deepseek_v41/mono/moe_plan.py`（52 行）
- [ ] **B** `atom/models/deepseek_v41/mono/runner.py`（746 行）
- [ ] **B** `atom/models/deepseek_v41/mono/walk_plan.py`（57 行）
- [ ] **B** `atom/models/glm52/mono/contract.py`（133 行）
- [ ] **B** `atom/models/glm52/mono/dispatch.py`（152 行）
- [ ] **B** `atom/models/glm52/mono/runner.py`（930 行）
- [ ] **B** `atom/models/kimi_k3/mono/dispatch.py`（128 行）
- [ ] **B** `atom/models/kimi_k3/mono/plan.py`（133 行）
- [ ] **B** `atom/models/kimi_k3/mono/runner.py`（480 行）
- [ ] **B** `atom/models/minimax_m3/mono/contract.py`（64 行）
- [ ] **B** `atom/models/minimax_m3/mono/dispatch.py`（143 行）
- [ ] **B** `atom/models/minimax_m3/mono/runner.py`（494 行）
- [ ] **C** `atom/mono/__init__.py`（12 行）
- [ ] **C** `atom/mono/device/mx.py`（81 行）
- [ ] **C** `atom/mono/device/ops.py`（617 行）
- [ ] **C** `atom/mono/device/ranks.py`（27 行）
- [ ] **C** `atom/mono/device/select.py`（100 行）
- [ ] **C** `atom/mono/device/stamps.py`（45 行）
- [ ] **C** `atom/mono/device/sync.py`（222 行）
- [ ] **C** `atom/mono/weights/__init__.py`（6 行）
- [ ] **C** `atom/mono/weights/formats.py`（146 行）
- [ ] **C** `atom/mono/weights/packing.py`（254 行）
- [ ] **C** `atom/models/deepseek_v41/mono/__init__.py`（9 行）
- [ ] **C** `atom/models/deepseek_v41/mono/check.py`（338 行）
- [ ] **C** `atom/models/deepseek_v41/mono/config.py`（73 行）
- [ ] **C** `atom/models/deepseek_v41/mono/draft_runner.py`（141 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/__init__.py`（8 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/attention.py`（483 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/attn_post.py`（766 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/attn_pre.py`（1380 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/debug.py`（30 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/dims.py`（111 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/index_query.py`（248 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/index_score.py`（412 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/index_score_fp4.py`（487 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/layer_post.py`（341 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/moe.py`（1629 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/moe_shape.py`（217 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/mx.py`（39 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/peer_layout.py`（41 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/token_par.py`（46 行）
- [ ] **C** `atom/models/deepseek_v41/mono/kernels/topk.py`（165 行）
- [ ] **C** `atom/models/deepseek_v41/mono/sources.py`（11 行）
- [ ] **C** `atom/models/deepseek_v41/mono/step_meta.py`（145 行）
- [ ] **C** `atom/models/deepseek_v41/mono/weights.py`（130 行）
- [ ] **C** `atom/models/glm52/mono/config.py`（428 行）
- [ ] **C** `atom/models/glm52/mono/weights.py`（376 行）
- [ ] **C** `atom/models/kimi_k3/mono/__init__.py`（8 行）
- [ ] **C** `atom/models/kimi_k3/mono/config.py`（335 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/__init__.py`（14 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/attention/back.py`（400 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/attention/kda.py`（1109 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/layer.py`（536 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/stages/gemv.py`（165 行）
- [ ] **C** `atom/models/kimi_k3/mono/kernels/stages/moe.py`（1434 行）
- [ ] **C** `atom/models/kimi_k3/mono/layouts.py`（75 行）
- [ ] **C** `atom/models/kimi_k3/mono/sources.py`（14 行）
- [ ] **C** `atom/models/kimi_k3/mono/weights.py`（435 行）
- [ ] **C** `atom/models/minimax_m3/mono/check.py`（160 行）
- [ ] **C** `atom/models/minimax_m3/mono/config.py`（125 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/common.py`（97 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/dense_post.py`（284 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/index_score.py`（475 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/post_attn.py`（631 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/pre_attn.py`（915 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/__init__.py`（10 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/attention.py`（596 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/dense_mlp.py`（466 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/ffn_reduce.py`（90 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/moe_common.py`（294 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/moe_narrow.py`（319 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/moe_wide.py`（506 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/router.py`（316 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/select.py`（358 行）
- [ ] **C** `atom/models/minimax_m3/mono/kernels/stages/shared.py`（55 行）
- [ ] **C** `atom/models/minimax_m3/mono/layout.py`（357 行）
- [ ] **C** `atom/models/minimax_m3/mono/sources.py`（8 行）
- [ ] **C** `atom/models/minimax_m3/mono/weights.py`（298 行）
- [ ] — `atom/mono/device/__init__.py`（4 行）
- [ ] — `atom/models/glm52/mono/__init__.py`（3 行）
- [ ] — `atom/models/kimi_k3/mono/kernels/attention/__init__.py`（3 行）
- [ ] — `atom/models/kimi_k3/mono/kernels/stages/__init__.py`（3 行）
- [ ] — `atom/models/minimax_m3/mono/__init__.py`（4 行）
- [ ] — `atom/models/minimax_m3/mono/kernels/__init__.py`（3 行）

### 9b 并行扩展：PP / DCP / PCP / TBO

- [ ] **B** `atom/model_engine/pp_engine_core.py`（449 行）
- [ ] **B** `atom/distributed/dcp_utils.py`（173 行）
- [ ] **B** `atom/distributed/pcp_utils.py`（285 行）
- [ ] **B** `atom/distributed/pp_comm.py`（164 行）
- [ ] **B** `atom/distributed/pp_transport.py`（167 行）
- [ ] **B** `atom/model_engine/dynamic_chunking.py`（574 行）
- [ ] **B** `atom/model_engine/dynamic_chunking_worker.py`（415 行）
- [ ] **B** `atom/model_ops/dcp_ops.py`（1557 行）
- [ ] **B** `atom/utils/tbo/ubatch_splitting.py`（505 行）
- [ ] **B** `atom/utils/tbo/ubatching.py`（674 行）
- [ ] **B** `atom/utils/distributed/utils.py`（149 行）
- [ ] **C** `atom/distributed/dcp_layout.py`（46 行）
- [ ] **C** `atom/distributed/simulated_tp.py`（198 行）
- [ ] **C** `atom/utils/tbo/__init__.py`（66 行）
- [ ] **C** `atom/utils/tbo/prefill_token_split.py`（89 行）
- [ ] **C** `atom/utils/tbo/ubatch_wrapper.py`（638 行）
- [ ] — `atom/distributed/__init__.py`（2 行）

### 9c PD 分离与 KV offload

- [ ] **B** `atom/kv_transfer/disaggregation/aggregator.py`（273 行）
- [ ] **B** `atom/kv_transfer/disaggregation/base.py`（149 行）
- [ ] **B** `atom/kv_transfer/disaggregation/factory.py`（395 行）
- [ ] **B** `atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py`（3605 行）
- [ ] **B** `atom/kv_transfer/disaggregation/moriio/moriio_connector.py`（1036 行）
- [ ] **B** `atom/kv_transfer/disaggregation/types.py`（605 行）
- [ ] **B** `atom/kv_transfer/offload/_offload_common.py`（1163 行）
- [ ] **B** `atom/kv_transfer/offload/chunked_scheduler.py`（1532 行）
- [ ] **B** `atom/kv_transfer/offload/config.py`（756 行）
- [ ] **B** `atom/kv_transfer/offload/connector.py`（403 行）
- [ ] **B** `atom/kv_transfer/offload/dense/connector.py`（526 行）
- [ ] **B** `atom/model_engine/disagg_types.py`（52 行）
- [ ] **C** `atom/kv_transfer/disaggregation/__init__.py`（43 行）
- [ ] **C** `atom/kv_transfer/disaggregation/index_staging.py`（146 行）
- [ ] **C** `atom/kv_transfer/disaggregation/landing_scatter.py`（185 行）
- [ ] **C** `atom/kv_transfer/disaggregation/mooncake/mla_landing.py`（1100 行）
- [ ] **C** `atom/kv_transfer/disaggregation/mooncake/rail_engine_pool.py`（120 行）
- [ ] **C** `atom/kv_transfer/disaggregation/moriio/__init__.py`（12 行）
- [ ] **C** `atom/kv_transfer/disaggregation/moriio/moriio_common.py`（174 行）
- [ ] **C** `atom/kv_transfer/disaggregation/moriio/moriio_engine.py`（360 行）
- [ ] **C** `atom/kv_transfer/disaggregation/multi/__init__.py`（9 行）
- [ ] **C** `atom/kv_transfer/disaggregation/multi/multi_connector.py`（899 行）
- [ ] **C** `atom/kv_transfer/disaggregation/page_region.py`（59 行）
- [ ] **C** `atom/kv_transfer/disaggregation/pd_producer.py`（174 行）
- [ ] **C** `atom/kv_transfer/disaggregation/port_offset.py`（50 行）
- [ ] **C** `atom/kv_transfer/disaggregation/pp_kv_aggregator.py`（185 行）
- [ ] **C** `atom/kv_transfer/disaggregation/sharded_transfer.py`（292 行）
- [ ] **C** `atom/kv_transfer/disaggregation/utils.py`（116 行）
- [ ] **C** `atom/kv_transfer/offload/__init__.py`（65 行）
- [ ] **C** `atom/kv_transfer/offload/_block_gpu_connector.py`（995 行）
- [ ] **C** `atom/kv_transfer/offload/atom_lmcache_staging.py`（159 行）
- [ ] **C** `atom/kv_transfer/offload/cache_policy.py`（83 行）
- [ ] **C** `atom/kv_transfer/offload/dense/kv_byte_codec.py`（449 行）
- [ ] **C** `atom/kv_transfer/offload/dense/triton_kv_staging.py`（788 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/dsv4/__init__.py`（54 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/dsv4/codec.py`（1682 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/dsv4/connector.py`（2800 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/dsv4/policy.py`（428 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py`（407 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/kimi_k3/connector.py`（914 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/kimi_k3/staging.py`（348 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/kimi_k3/state_object.py`（282 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/kimi_k3/state_tier.py`（491 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/m3/__init__.py`（11 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/m3/connector.py`（171 行）
- [ ] **C** `atom/kv_transfer/offload/hybrid/qwen/connector.py`（31 行）
- [ ] **C** `atom/kv_transfer/offload/metadata.py`（220 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/__init__.py`（21 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/client.py`（380 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/config.py`（279 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/keys.py`（252 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/nic.py`（160 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/pool.py`（365 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/scheduler.py`（686 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/taint.py`（136 行）
- [ ] **C** `atom/kv_transfer/offload/mooncake_store/worker.py`（1094 行）
- [ ] **C** `atom/kv_transfer/offload/mp/connector.py`（244 行）
- [ ] **C** `atom/kv_transfer/offload/mp/deployment.py`（363 行）
- [ ] **C** `atom/kv_transfer/offload/mp/lookup.py`（505 行）
- [ ] **C** `atom/kv_transfer/offload/mp/native_state_layout.py`（213 行）
- [ ] **C** `atom/kv_transfer/offload/mp/native_state_scheduler.py`（748 行）
- [ ] **C** `atom/kv_transfer/offload/mp/native_state_worker.py`（513 行）
- [ ] **C** `atom/kv_transfer/offload/mp/page_views.py`（158 行）
- [ ] **C** `atom/kv_transfer/offload/mp/scheduler.py`（296 行）
- [ ] **C** `atom/kv_transfer/offload/mp/transfer.py`（179 行）
- [ ] **C** `atom/kv_transfer/offload/mp/worker.py`（510 行）
- [ ] — `atom/kv_transfer/__init__.py`（0 行）
- [ ] — `atom/kv_transfer/disaggregation/mooncake/__init__.py`（0 行）
- [ ] — `atom/kv_transfer/offload/dense/__init__.py`（4 行）
- [ ] — `atom/kv_transfer/offload/hybrid/__init__.py`（4 行）
- [ ] — `atom/kv_transfer/offload/hybrid/kimi_k3/__init__.py`（4 行）
- [ ] — `atom/kv_transfer/offload/hybrid/qwen/__init__.py`（4 行）
- [ ] — `atom/kv_transfer/offload/mp/__init__.py`（4 行）

## 第 10 批：对外集成与周边：插件、RL rollout、可观测性、diffusion（201 个文件，83,483 行）


### 10a 插件公共层

- [ ] **B** `atom/plugin/config.py`（857 行）
- [ ] **B** `atom/plugin/prepare.py`（222 行）
- [ ] **B** `atom/plugin/register.py`（1222 行）
- [ ] **C** `atom/plugin/__init__.py`（13 行）
- [ ] **C** `atom/plugin/graph_capture_patch.py`（115 行）

### 10b vLLM 插件

- [ ] **B** `atom/plugin/vllm/attention/backend.py`（549 行）
- [ ] **B** `atom/plugin/vllm/attention/layer_mha.py`（1048 行）
- [ ] **B** `atom/plugin/vllm/attention/layer_mla.py`（1613 行）
- [ ] **B** `atom/plugin/vllm/attention/metadata.py`（2916 行）
- [ ] **B** `atom/plugin/vllm/model_wrapper.py`（1398 行）
- [ ] **B** `atom/plugin/vllm/platform.py`（513 行）
- [ ] **B** `atom/plugin/vllm/register.py`（579 行）
- [ ] **C** `atom/plugin/vllm/attention/layer.py`（85 行）
- [ ] **C** `atom/plugin/vllm/attention/layer_common.py`（11 行）
- [ ] **C** `atom/plugin/vllm/attention/layer_gdn.py`（675 行）
- [ ] **C** `atom/plugin/vllm/attention/layer_sparse_mla.py`（602 行）
- [ ] **C** `atom/plugin/vllm/attention/minimax_m3_attnetion.py`（1136 行）
- [ ] **C** `atom/plugin/vllm/attention/ops.py`（125 行）
- [ ] **C** `atom/plugin/vllm/attention_bridge.py`（62 行）
- [ ] **C** `atom/plugin/vllm/cudagraph_memory_profiler_patch.py`（73 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v41_bridge.py`（1945 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v41_state_reserve_patch.py`（306 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v4_bridge.py`（2455 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v4_cudagraph_patch.py`（305 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v4_ops.py`（259 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v4_prefix_patch.py`（264 行）
- [ ] **C** `atom/plugin/vllm/deepseek_v4_state_checkpoint.py`（573 行）
- [ ] **C** `atom/plugin/vllm/dist_store_patch.py`（57 行）
- [ ] **C** `atom/plugin/vllm/dspark_dcp_patch.py`（249 行）
- [ ] **C** `atom/plugin/vllm/gdn_backend.py`（160 行）
- [ ] **C** `atom/plugin/vllm/graph_capture_patch.py`（18 行）
- [ ] **C** `atom/plugin/vllm/kda_backend.py`（138 行）
- [ ] **C** `atom/plugin/vllm/kv_transfer/connector.py`（1517 行）
- [ ] **C** `atom/plugin/vllm/kv_transfer/kda_state.py`（1091 行）
- [ ] **C** `atom/plugin/vllm/kv_transfer/kv_cache_layout.py`（424 行）
- [ ] **C** `atom/plugin/vllm/kv_transfer/offload_config.py`（150 行）
- [ ] **C** `atom/plugin/vllm/kv_transfer/seq_view.py`（162 行）
- [ ] **C** `atom/plugin/vllm/models/deepseek_v4.py`（421 行）
- [ ] **C** `atom/plugin/vllm/models/deepseek_v41.py`（140 行）
- [ ] **C** `atom/plugin/vllm/models/deepseek_v4_mtp.py`（25 行）
- [ ] **C** `atom/plugin/vllm/models/kimi_k25.py`（318 行）
- [ ] **C** `atom/plugin/vllm/models/kimi_k3.py`（703 行）
- [ ] **C** `atom/plugin/vllm/models/kimi_k3_dspark.py`（129 行）
- [ ] **C** `atom/plugin/vllm/models/qwen3_5.py`（407 行）
- [ ] **C** `atom/plugin/vllm/models/qwen3_next.py`（117 行）
- [ ] **C** `atom/plugin/vllm/moe.py`（57 行）
- [ ] **C** `atom/plugin/vllm/mori_patch.py`（195 行）
- [ ] **C** `atom/plugin/vllm/paged_state_checkpoint.py`（1033 行）
- [ ] **C** `atom/plugin/vllm/paged_state_checkpoint_scheduler.py`（858 行）
- [ ] **C** `atom/plugin/vllm/qwen35_attention_patch.py`（181 行）
- [ ] **C** `atom/plugin/vllm/qwen35_plugin_scope.py`（64 行）
- [ ] **C** `atom/plugin/vllm/req_id_passthrough_patch.py`（247 行）
- [ ] **C** `atom/plugin/vllm/rocm_dcp_full_graph_patch.py`（50 行）
- [ ] **C** `atom/plugin/vllm/scheduler.py`（237 行）
- [ ] **C** `atom/plugin/vllm/spec_decode_patch.py`（841 行）
- [ ] **C** `atom/plugin/vllm/state_slot_allocator.py`（169 行）
- [ ] **C** `atom/plugin/vllm/tp_group_reuse.py`（346 行）
- [ ] **C** `atom/plugin/vllm/v41_core_overrides.py`（724 行）
- [ ] **C** `atom/plugin/vllm/v41_speculative.py`（339 行）
- [ ] **C** `atom/plugin/vllm/vllm_runtime.py`（178 行）
- [ ] — `atom/plugin/vllm/__init__.py`（5 行）
- [ ] — `atom/plugin/vllm/attention/__init__.py`（5 行）
- [ ] — `atom/plugin/vllm/kv_transfer/__init__.py`（2 行）

### 10c SGLang 插件

- [ ] **B** `atom/plugin/sglang/attention_backend/full_attention/full_attention_backend.py`（2940 行）
- [ ] **B** `atom/plugin/sglang/models/base_model_wrapper.py`（544 行）
- [ ] **B** `atom/plugin/sglang/prepare.py`（164 行）
- [ ] **B** `atom/plugin/sglang/register.py`（249 行）
- [ ] **B** `atom/plugin/sglang/runtime/forward_context.py`（902 行）
- [ ] **C** `atom/plugin/sglang/attention.py`（7 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/attention_gdn.py`（982 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/backend_resolver.py`（114 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/deepseek_v4_backend.py`（471 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/full_attention/__init__.py`（8 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/full_attention/kv_cache.py`（161 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/full_attention/metadata.py`（39 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/full_attention/pa_metadata.py`（126 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/full_attention/radix_attention.py`（242 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/gdn_replayssm.py`（271 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/glm52_dsa_backend.py`（486 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/kimi_k3_backend.py`（155 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/minimax_m3_sparse.py`（973 行）
- [ ] **C** `atom/plugin/sglang/attention_backend/sparse_mla_indexer.py`（1463 行）
- [ ] **C** `atom/plugin/sglang/deepseek_v4_bridge.py`（2564 行）
- [ ] **C** `atom/plugin/sglang/dflash_lm_head_bridge.py`（179 行）
- [ ] **C** `atom/plugin/sglang/eagle3_llama_bridge.py`（1101 行）
- [ ] **C** `atom/plugin/sglang/glm52_dsa_bridge.py`（1800 行）
- [ ] **C** `atom/plugin/sglang/kimi_k3_bridge.py`（639 行）
- [ ] **C** `atom/plugin/sglang/minimax_m3_bridge.py`（822 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_mla.py`（177 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_mla_attention.py`（444 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_mla_forward.py`（879 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_nextn_wrapper.py`（445 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_v4_attention.py`（363 行）
- [ ] **C** `atom/plugin/sglang/models/deepseek_v4_nextn_wrapper.py`（323 行）
- [ ] **C** `atom/plugin/sglang/models/glm52_dsa.py`（73 行）
- [ ] **C** `atom/plugin/sglang/models/glm52_dsa_attention.py`（135 行）
- [ ] **C** `atom/plugin/sglang/models/kimi_k25.py`（249 行）
- [ ] **C** `atom/plugin/sglang/models/kimi_k3_processor.py`（37 行）
- [ ] **C** `atom/plugin/sglang/models/kv_cache_utils.py`（13 行）
- [ ] **C** `atom/plugin/sglang/models/minimax_m3.py`（196 行）
- [ ] **C** `atom/plugin/sglang/models/minimax_m3_processor.py`（76 行）
- [ ] **C** `atom/plugin/sglang/models/qwen3_5.py`（501 行）
- [ ] **C** `atom/plugin/sglang/models/qwen3_5_attention.py`（823 行）
- [ ] **C** `atom/plugin/sglang/models/qwen4_exp.py`（454 行）
- [ ] **C** `atom/plugin/sglang/models/qwen4_exp_nextn_wrapper.py`（254 行）
- [ ] **C** `atom/plugin/sglang/patches/graph_capture_patch.py`（18 行）
- [ ] **C** `atom/plugin/sglang/patches/prefill_compile_only_patch.py`（192 行）
- [ ] **C** `atom/plugin/sglang/patches/qwen4_exp_gdn_pad.py`（43 行）
- [ ] **C** `atom/plugin/sglang/patches/qwen4_exp_recognition_patch.py`（177 行）
- [ ] **C** `atom/plugin/sglang/patches/qwen4_exp_rocm_patch.py`（189 行）
- [ ] **C** `atom/plugin/sglang/patches/triton_kernel_retention_patch.py`（68 行）
- [ ] **C** `atom/plugin/sglang/qwen4_exp_bridge.py`（1572 行）
- [ ] **C** `atom/plugin/sglang/runtime/__init__.py`（39 行）
- [ ] **C** `atom/plugin/sglang/runtime/attention_backend_resolver.py`（73 行）
- [ ] **C** `atom/plugin/sglang/runtime/context.py`（141 行）
- [ ] **C** `atom/plugin/sglang/runtime/load_config_patch.py`（33 行）
- [ ] **C** `atom/plugin/sglang/runtime/model_arch.py`（605 行）
- [ ] **C** `atom/plugin/sglang/tbo/__init__.py`（17 行）
- [ ] **C** `atom/plugin/sglang/tbo/adapter.py`（257 行）
- [ ] **C** `atom/plugin/sglang/tbo/forward_inputs.py`（101 行）
- [ ] **C** `atom/plugin/sglang/tbo/sglang_tbo_compat_patches.py`（106 行）
- [ ] **C** `atom/plugin/sglang/tbo/ubatch_wrapper.py`（229 行）
- [ ] — `atom/plugin/sglang/__init__.py`（3 行）
- [ ] — `atom/plugin/sglang/attention_backend/__init__.py`（0 行）
- [ ] — `atom/plugin/sglang/models/__init__.py`（5 行）
- [ ] — `atom/plugin/sglang/patches/__init__.py`（1 行）

### 10d RTP-LLM 插件

- [ ] **B** `atom/plugin/rtpllm/attention_backend/rtp_full_attention.py`（832 行）
- [ ] **B** `atom/plugin/rtpllm/models/base_model_wrapper.py`（44 行）
- [ ] **B** `atom/plugin/rtpllm/utils/forward_context.py`（2407 行）
- [ ] **C** `atom/plugin/rtpllm/__init__.py`（7 行）
- [ ] **C** `atom/plugin/rtpllm/attention_backend/__init__.py`（37 行）
- [ ] **C** `atom/plugin/rtpllm/attention_backend/attention_gdn.py`（168 行）
- [ ] **C** `atom/plugin/rtpllm/attention_backend/attention_switch.py`（27 行）
- [ ] **C** `atom/plugin/rtpllm/attention_backend/rtp_mla_attention.py`（270 行）
- [ ] **C** `atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py`（1908 行）
- [ ] **C** `atom/plugin/rtpllm/models/__init__.py`（20 行）
- [ ] **C** `atom/plugin/rtpllm/models/glm5.py`（815 行）
- [ ] **C** `atom/plugin/rtpllm/models/qwen3_5.py`（836 行）
- [ ] **C** `atom/plugin/rtpllm/models/qwen3_next.py`（207 行）
- [ ] **C** `atom/plugin/rtpllm/models/qwen4_exp.py`（339 行）
- [ ] **C** `atom/plugin/rtpllm/utils/__init__.py`（11 行）
- [ ] **C** `atom/plugin/rtpllm/utils/qwen4_exp_context.py`（315 行）

### 10e RL rollout

- [ ] **B** `atom/rollout/async_engine.py`（226 行）
- [ ] **B** `atom/rollout/weight_sync.py`（357 行）
- [ ] **B** `atom/rollout/weight_updater.py`（1328 行）
- [ ] **C** `atom/rollout/__init__.py`（32 行）
- [ ] **C** `atom/rollout/capabilities.py`（71 行）
- [ ] **C** `atom/rollout/memory_manager.py`（483 行）
- [ ] **C** `atom/rollout/model_runner_ext.py`（424 行）

### 10f 指标、追踪、压测与调试工具

- [ ] **B** `atom/metrics/exporter.py`（589 行）
- [ ] **B** `atom/benchmarks/benchmark_serving.py`（1315 行）
- [ ] **B** `atom/utils/debug_helper/compare.py`（418 行）
- [ ] **B** `atom/utils/debug_helper/dump.py`（315 行）
- [ ] **C** `atom/metrics/gpu.py`（167 行）
- [ ] **C** `atom/metrics/histogram.py`（51 行）
- [ ] **C** `atom/metrics/prometheus.py`（23 行）
- [ ] **C** `atom/metrics/request.py`（99 行）
- [ ] **C** `atom/metrics/scheduler.py`（237 行）
- [ ] **C** `atom/benchmarks/backend_request_func.py`（504 行）
- [ ] **C** `atom/benchmarks/benchmark_utils.py`（40 行）
- [ ] **C** `atom/benchmarks/results/__main__.py`（175 行）
- [ ] **C** `atom/benchmarks/results/aggregate.py`（433 行）
- [ ] **C** `atom/benchmarks/results/bundle.py`（607 行）
- [ ] **C** `atom/benchmarks/results/io.py`（117 行）
- [ ] **C** `atom/benchmarks/results/metadata.py`（425 行）
- [ ] **C** `atom/benchmarks/results/records.py`（207 行）
- [ ] **C** `atom/benchmarks/results/telemetry.py`（351 行）
- [ ] **C** `atom/utils/debug_helper/__init__.py`（64 行）
- [ ] **C** `atom/utils/debug_helper/ref_patch.py`（203 行）
- [ ] **C** `atom/mesh/proto/vendor_ext_proc.py`（187 行）
- [ ] — `atom/metrics/__init__.py`（1 行）
- [ ] — `atom/benchmarks/results/__init__.py`（5 行）

### 10g diffusion 子系统（视频 + 音频生成）

- [ ] **B** `atom/diffusion/attention.py`（156 行）
- [ ] **B** `atom/diffusion/engine/__init__.py`（29 行）
- [ ] **B** `atom/diffusion/engine/core_manager.py`（211 行）
- [ ] **B** `atom/diffusion/engine/diffusion_engine.py`（218 行）
- [ ] **B** `atom/diffusion/engine/engine_core.py`（235 行）
- [ ] **B** `atom/diffusion/engine/job_scheduler.py`（143 行）
- [ ] **B** `atom/diffusion/engine/pipeline_runner.py`（175 行）
- [ ] **B** `atom/diffusion/engine/protocol.py`（92 行）
- [ ] **B** `atom/diffusion/entrypoints/diffusion_server.py`（175 行）
- [ ] **B** `atom/diffusion/models/minimax_h3/denoise.py`（411 行）
- [ ] **B** `atom/diffusion/models/minimax_h3/dit.py`（879 行）
- [ ] **B** `atom/diffusion/models/minimax_h3/pipeline.py`（1014 行）
- [ ] **B** `atom/diffusion/pipeline.py`（307 行）
- [ ] **B** `atom/diffusion/ulysses.py`（125 行）
- [ ] **C** `atom/diffusion/__init__.py`（19 行）
- [ ] **C** `atom/diffusion/config.py`（83 行）
- [ ] **C** `atom/diffusion/entrypoints/video_api.py`（156 行）
- [ ] **C** `atom/diffusion/examples/dummy_pipeline.py`（231 行）
- [ ] **C** `atom/diffusion/models/__init__.py`（11 行）
- [ ] **C** `atom/diffusion/models/minimax_h3/__init__.py`（66 行）
- [ ] **C** `atom/diffusion/models/minimax_h3/arch.py`（118 行）
- [ ] **C** `atom/diffusion/models/minimax_h3/components.py`（597 行）
- [ ] **C** `atom/diffusion/models/minimax_h3/conditioning.py`（785 行）
- [ ] **C** `atom/diffusion/models/minimax_h3/layout.py`（845 行）
- [ ] **C** `atom/diffusion/mux.py`（126 行）
- [ ] **C** `atom/diffusion/registry.py`（70 行）
- [ ] **C** `atom/diffusion/request.py`（112 行）
- [ ] — `atom/diffusion/entrypoints/__init__.py`（2 行）
- [ ] — `atom/diffusion/examples/__init__.py`（4 行）

## 不修改的文件（9 个文件，5,878 行）

- [ ] 不改 `atom/models/glm52/mono/kernels/__init__.py`（33 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/geometry.py`（160 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/geometry_config.py`（200 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/kernel.py`（4216 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/layout.py`（546 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/op.py`（629 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/tp4_kernel.py`（14 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/tp4_layout.py`（6 行）
- [ ] 不改 `atom/models/glm52/mono/kernels/tp4_op.py`（74 行）
