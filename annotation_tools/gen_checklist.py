"""Assign every atom/*.py to a batch/section and a depth tier; emit the checklist."""
import collections, fnmatch, glob, sys

SECTIONS = [  # (section id, title, patterns) -- first match wins, order = reading order
 ("X",  "不修改：FlyDSL 上游代码（须与上游保持字节一致）", ["atom/models/glm52/mono/kernels/*"]),
 ("8d_mono", None, ["atom/models/*/mono/*"]),  # placeholder id, remapped to 9a below
 ("1",  "入口、配置与请求对象", ["atom/__init__.py", "atom/examples/*", "atom/model_engine/arg_utils.py", "atom/config.py",
        "atom/utils/envs.py", "atom/utils/arg_parser.py", "atom/utils/__init__.py", "atom/sampling_params.py",
        "atom/model_engine/sequence.py", "atom/model_engine/request.py", "atom/model_config/*"]),
 ("2a", "离线引擎与多进程", ["atom/model_engine/llm_engine.py", "atom/model_engine/engine_core_mgr.py",
        "atom/model_engine/engine_core.py", "atom/model_engine/async_proc.py", "atom/model_engine/engine_core_protocol.py",
        "atom/model_engine/ipc_utils.py", "atom/model_engine/output_transfer.py", "atom/model_engine/collective_rpc.py",
        "atom/model_engine/capabilities.py", "atom/model_engine/engine_utility.py", "atom/model_engine/engine_stats.py",
        "atom/model_engine/run_labels.py", "atom/utils/network.py", "atom/utils/numa_utils.py", "atom/utils/gc_utils.py"]),
 ("2b", "分词与多模态输入", ["atom/model_engine/segment_split.py", "atom/model_engine/segment_tokenizer.py",
        "atom/model_engine/multimodal_runtime.py", "atom/multimodal/*"]),
 ("2c", "OpenAI 兼容服务层", ["atom/entrypoints/*"]),
 ("3",  "调度器", ["atom/model_engine/scheduler.py", "atom/model_engine/prefill_delayer.py"]),
 ("4a", "KV block 账本与前缀缓存", ["atom/model_engine/block_manager.py", "atom/model_engine/block_pool.py",
        "atom/model_engine/kv_block.py", "atom/model_engine/prefix_hash.py", "atom/model_engine/prefix_block_hashes.py",
        "atom/model_engine/block_table_codec.py", "atom/model_engine/byte_lru.py", "atom/utils/block_tables.py",
        "atom/utils/block_convert.py", "atom/distributed/kv_events.py"]),
 ("4b", "递归状态缓存（GDN / Mamba / KDA 等混合模型）", ["atom/model_engine/state_*.py", "atom/model_engine/page_unit_checkpoint.py"]),
 ("5a", "ModelRunner 与 ForwardContext", ["atom/model_engine/model_runner.py", "atom/utils/forward_context.py",
        "atom/utils/h2d.py", "atom/utils/packed_h2d.py", "atom/model_ops/decode_input_ids.py"]),
 ("5b", "KV pool 内存布局与 token 索引", ["atom/model_ops/attentions/pool_layout/*", "atom/model_ops/attentions/token_layout/*"]),
 ("5c", "权重加载", ["atom/model_loader/*"]),
 ("5d", "采样", ["atom/model_ops/sampler.py", "atom/model_ops/lm_head_argmax.py", "atom/model_ops/required_channel.py"]),
 ("5e", "CUDA Graph 与 torch.compile", ["atom/utils/decorators.py", "atom/utils/backends.py",
        "atom/utils/cuda_piecewise_backend.py", "atom/utils/compiler_inferface.py", "atom/utils/region_graph.py",
        "atom/utils/graph_marker*.py", "atom/utils/graph_holders.py", "atom/utils/custom_register.py",
        "atom/model_ops/module_dispatch_ops.py", "atom/distributed/nccl_watchdog.py"]),
 ("6a", "基础算子与张量并行", ["atom/model_ops/__init__.py", "atom/model_ops/base_config.py", "atom/model_ops/utils.py",
        "atom/model_ops/linear.py", "atom/model_ops/embed_head.py", "atom/model_ops/layernorm.py",
        "atom/model_ops/rotary_embedding.py", "atom/model_ops/activation.py", "atom/model_ops/communication_op.py",
        "atom/model_ops/triton_rmsnorm_nw.py"]),
 ("6b", "量化", ["atom/quant_spec.py", "atom/quantization/*", "atom/model_ops/blockscale.py",
        "atom/model_ops/blockscale_kernels/*", "atom/model_ops/triton_fused_qkv_quant.py"]),
 ("6c", "注意力（MHA）", ["atom/utils/selector.py", "atom/model_ops/base_attention.py", "atom/model_ops/paged_attention.py",
        "atom/model_ops/attention_mha.py", "atom/model_ops/attentions/__init__.py", "atom/model_ops/attentions/backends.py",
        "atom/model_ops/attentions/aiter_attention.py", "atom/model_ops/attentions/triton_mha.py",
        "atom/model_ops/attentions/triton_merge_attn_states.py", "atom/model_ops/attentions/mha_kv_pool.py"]),
 ("6d", "dense 模型", ["atom/models/__init__.py", "atom/models/registry.py", "atom/models/utils.py", "atom/models/qwen3.py",
        "atom/models/llama.py", "atom/models/mistral3.py", "atom/models/eagle3_llama.py", "atom/model_ops/fused_aux_rmsnorm.py"]),
 ("7a", "MoE 与专家并行", ["atom/model_ops/moe.py", "atom/model_ops/fused_moe_triton.py", "atom/model_ops/fused_moe/*",
        "atom/model_ops/topK.py", "atom/model_ops/eplb.py", "atom/models/qwen3_moe.py", "atom/models/mixtral.py"]),
 ("7b", "MLA 与 DeepSeek-V3", ["atom/model_ops/attention_mla.py", "atom/model_ops/attentions/aiter_mla.py",
        "atom/model_ops/attentions/triton_mla.py", "atom/model_ops/triton_fused_mla_ctx_kv.py",
        "atom/model_ops/attentions/mla_kv_pool.py", "atom/model_ops/attentions/sparse_mla_head_groups.py",
        "atom/model_ops/fused_mtp_prologue.py", "atom/models/deepseek_v2.py", "atom/models/deepseek_mtp.py",
        "atom/models/eagle3_deepseek_mla.py"]),
 ("7c", "投机解码框架（MTP / EAGLE / DSpark）", ["atom/spec_decode/*", "atom/model_ops/rejection_sampler.py",
        "atom/model_ops/dspark_markov_sample.py", "atom/models/dspark_draft.py"]),
 ("7d", "DSA 稀疏注意力索引器（V3.2 / GLM-5）", ["atom/model_ops/indexer/*", "atom/model_ops/sparse_indexer_*.py",
        "atom/model_ops/fp4_mqa_ragged_metadata.py"]),
 ("7e", "其他 MoE 模型", ["atom/models/glm4_moe*.py", "atom/models/gpt_oss.py", "atom/model_ops/swiglu_oai.py",
        "atom/models/minimax_m2.py", "atom/models/mimo_v2*.py", "atom/models/kimi_k25.py"]),
 ("8a", "线性注意力与混合架构（Qwen3-Next / Qwen3.5）", ["atom/model_ops/attention_gdn.py",
        "atom/model_ops/attentions/gdn_attn.py", "atom/model_ops/fla_ops/*", "atom/model_ops/mamba_ops/*",
        "atom/model_ops/split_chunk.py", "atom/model_ops/triton_fused_qkv_norm_rope_cache.py",
        "atom/model_ops/triton_fused_sigmoid_mul_quant.py", "atom/model_ops/triton_gemma_rmsnorm.py",
        "atom/model_ops/triton_mrope.py", "atom/models/qwen3_next*.py", "atom/models/qwen3_5*.py"]),
 ("8b", "DeepSeek-V4 / V4.1", ["atom/models/deepseek_v4/*", "atom/models/deepseek_v4_*.py", "atom/models/deepseek_v41/*",
        "atom/model_ops/v4_kernels/*", "atom/model_ops/deepseek_v41/*", "atom/model_ops/sparse_attn_v4.py",
        "atom/model_ops/quant_v4.py", "atom/model_ops/v4_backend_gate.py", "atom/model_ops/triton_hash_topk.py",
        "atom/model_ops/triton_mm_topk.py", "atom/model_ops/attentions/deepseek_v4*.py",
        "atom/model_ops/attentions/deepseek_v41/*", "atom/model_ops/engram/*"]),
 ("8c", "Kimi-K3 / MiniMax-M3 / GLM-5.x / Qwen4-exp", ["atom/models/kimi_k3/*", "atom/models/kimi_k3_*.py",
        "atom/model_ops/kimi_k3/*", "atom/model_ops/attention_residual.py", "atom/model_ops/attentions/kimi_mla_gdn_attn.py",
        "atom/models/minimax_m3/*", "atom/model_ops/minimax_m3/*", "atom/distributed/indexer_cp.py",
        "atom/models/glm5_next*.py", "atom/model_ops/glm5_next/*", "atom/models/glm52/*",
        "atom/models/qwen4_exp*.py", "atom/model_ops/qwen4_exp/*", "atom/model_ops/attentions/qwen4_exp_attn.py"]),
 ("9a", "Mono 融合 decode", ["atom/mono/*"]),
 ("9b", "并行扩展：PP / DCP / PCP / TBO", ["atom/model_engine/pp_engine_core.py", "atom/distributed/*",
        "atom/model_engine/dynamic_chunking*.py", "atom/model_ops/dcp_ops.py", "atom/utils/tbo/*", "atom/utils/distributed/*"]),
 ("9c", "PD 分离与 KV offload", ["atom/kv_transfer/*", "atom/model_engine/disagg_types.py"]),
 ("10a", "插件公共层", ["atom/plugin/__init__.py", "atom/plugin/config.py", "atom/plugin/prepare.py", "atom/plugin/register.py",
         "atom/plugin/graph_capture_patch.py", "atom/plugin/deepseek_v4_index_rows.py"]),
 ("10b", "vLLM 插件", ["atom/plugin/vllm/*"]),
 ("10c", "SGLang 插件", ["atom/plugin/sglang/*"]),
 ("10d", "RTP-LLM 插件", ["atom/plugin/rtpllm/*"]),
 ("10e", "RL rollout", ["atom/rollout/*"]),
 ("10f", "指标、追踪、压测与调试工具", ["atom/metrics/*", "atom/observability/*", "atom/benchmarks/*",
         "atom/utils/debug_helper/*", "atom/mesh/*"]),
 ("10g", "diffusion 子系统（视频 + 音频生成）", ["atom/diffusion/*"]),
]
REMAP = {"8d_mono": "9a"}

A = [  # line-by-line, nano-vllm style
 "atom/examples/simple_inference.py", "atom/model_engine/arg_utils.py", "atom/config.py", "atom/sampling_params.py",
 "atom/model_engine/sequence.py", "atom/model_engine/request.py",
 "atom/model_engine/llm_engine.py", "atom/model_engine/engine_core_mgr.py", "atom/model_engine/engine_core.py",
 "atom/model_engine/async_proc.py", "atom/entrypoints/openai_server.py", "atom/entrypoints/openai/api_server.py",
 "atom/model_engine/scheduler.py",
 "atom/model_engine/block_manager.py", "atom/model_engine/block_pool.py", "atom/model_engine/kv_block.py",
 "atom/model_engine/prefix_hash.py",
 "atom/model_engine/model_runner.py", "atom/utils/forward_context.py", "atom/model_loader/loader.py",
 "atom/model_loader/loading_core.py", "atom/model_loader/weight_iterator.py", "atom/model_ops/sampler.py",
 "atom/utils/decorators.py", "atom/utils/backends.py", "atom/utils/cuda_piecewise_backend.py",
 "atom/model_ops/linear.py", "atom/model_ops/embed_head.py", "atom/model_ops/layernorm.py",
 "atom/model_ops/rotary_embedding.py", "atom/model_ops/activation.py", "atom/model_ops/communication_op.py",
 "atom/quant_spec.py", "atom/utils/selector.py", "atom/model_ops/base_attention.py", "atom/model_ops/paged_attention.py",
 "atom/model_ops/attention_mha.py", "atom/model_ops/attentions/backends.py", "atom/model_ops/attentions/aiter_attention.py",
 "atom/model_ops/attentions/mha_kv_pool.py", "atom/models/registry.py", "atom/models/utils.py", "atom/models/qwen3.py",
 "atom/model_ops/moe.py", "atom/model_ops/topK.py", "atom/models/qwen3_moe.py", "atom/model_ops/attention_mla.py",
 "atom/model_ops/attentions/aiter_mla.py", "atom/models/deepseek_v2.py", "atom/models/deepseek_mtp.py",
 "atom/spec_decode/drafter.py", "atom/spec_decode/eagle_proposer.py", "atom/model_ops/rejection_sampler.py",
]
C_IN_EARLY = [  # batches 1-7 default to B; these drop to C
 "atom/entrypoints/openai/tool_parser/*", "atom/entrypoints/openai/reasoning*.py",
 "atom/entrypoints/openai/serving_anthropic.py", "atom/entrypoints/openai/serving_responses.py",
 "atom/entrypoints/atomesh/*", "atom/entrypoints/openai/marker_scanner.py", "atom/entrypoints/openai/kimi_k3_tokens.py",
 "atom/entrypoints/openai/deepseek_v41_encoder.py", "atom/entrypoints/openai/api_compatibility.py",
 "atom/entrypoints/openai/metrics_setup.py", "atom/entrypoints/openai/request_timing.py",
 "atom/model_engine/engine_stats.py", "atom/model_engine/run_labels.py", "atom/model_engine/capabilities.py",
 "atom/model_engine/collective_rpc.py", "atom/utils/graph_marker_instrumentation.py", "atom/utils/graph_marker.py",
 "atom/utils/compiler_inferface.py", "atom/model_loader/deepseek_v41.py", "atom/examples/profile_offline.py",
]
B_IN_LATE = [  # batches 8-10 default to C; these rise to B
 # 8a
 "atom/models/qwen3_next.py", "atom/models/qwen3_5.py", "atom/model_ops/attention_gdn.py",
 "atom/model_ops/attentions/gdn_attn.py", "atom/model_ops/fla_ops/chunk.py", "atom/model_ops/fla_ops/fused_recurrent.py",
 "atom/models/qwen3_next_mtp.py",
 # 8b
 "atom/models/deepseek_v4/__init__.py", "atom/models/deepseek_v4_mtp.py", "atom/models/deepseek_v41/model.py",
 "atom/models/deepseek_v41/runtime.py", "atom/models/deepseek_v41/attention.py", "atom/models/deepseek_v41/moe.py",
 "atom/model_ops/attentions/deepseek_v4_attn.py", "atom/model_ops/attentions/deepseek_v41/backend.py",
 "atom/model_ops/attentions/deepseek_v41/cache.py", "atom/model_ops/v4_kernels/paged_decode.py",
 "atom/model_ops/engram/host.py",
 # 8c
 "atom/models/kimi_k3/model.py", "atom/model_ops/attentions/kimi_mla_gdn_attn.py", "atom/models/minimax_m3/model.py",
 "atom/model_ops/minimax_m3/sparse_attn.py", "atom/models/glm5_next.py", "atom/model_ops/glm5_next/kpool.py",
 "atom/models/qwen4_exp.py", "atom/model_ops/attentions/qwen4_exp_attn.py",
 # 9a
 "atom/mono/runtime/*", "atom/mono/plan/*", "atom/mono/weights/layer.py", "atom/mono/weights/config.py",
 "atom/mono/dsa_indexer/runner.py", "atom/mono/dsa_indexer/dispatch.py", "atom/mono/dsa_indexer/plan.py",
 "atom/models/*/mono/runner.py", "atom/models/*/mono/dispatch.py", "atom/models/*/mono/plan.py",
 "atom/models/deepseek_v41/mono/*_plan.py", "atom/models/*/mono/contract.py",
 # 9b
 "atom/model_engine/pp_engine_core.py", "atom/distributed/pp_*.py", "atom/model_engine/dynamic_chunking*.py",
 "atom/distributed/dcp_utils.py", "atom/distributed/pcp_utils.py", "atom/model_ops/dcp_ops.py",
 "atom/utils/tbo/ubatching.py", "atom/utils/tbo/ubatch_splitting.py", "atom/utils/distributed/utils.py",
 # 9c
 "atom/model_engine/disagg_types.py", "atom/kv_transfer/disaggregation/base.py", "atom/kv_transfer/disaggregation/factory.py",
 "atom/kv_transfer/disaggregation/types.py", "atom/kv_transfer/disaggregation/aggregator.py",
 "atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py", "atom/kv_transfer/disaggregation/moriio/moriio_connector.py",
 "atom/kv_transfer/offload/connector.py", "atom/kv_transfer/offload/config.py", "atom/kv_transfer/offload/_offload_common.py",
 "atom/kv_transfer/offload/chunked_scheduler.py", "atom/kv_transfer/offload/dense/connector.py",
 # 10
 "atom/plugin/register.py", "atom/plugin/prepare.py", "atom/plugin/config.py",
 "atom/plugin/vllm/register.py", "atom/plugin/vllm/platform.py", "atom/plugin/vllm/model_wrapper.py",
 "atom/plugin/vllm/attention/backend.py", "atom/plugin/vllm/attention/metadata.py",
 "atom/plugin/vllm/attention/layer_mha.py", "atom/plugin/vllm/attention/layer_mla.py",
 "atom/plugin/sglang/register.py", "atom/plugin/sglang/prepare.py", "atom/plugin/sglang/models/base_model_wrapper.py",
 "atom/plugin/sglang/attention_backend/full_attention/full_attention_backend.py",
 "atom/plugin/sglang/runtime/forward_context.py",
 "atom/plugin/rtpllm/models/base_model_wrapper.py", "atom/plugin/rtpllm/utils/forward_context.py",
 "atom/plugin/rtpllm/attention_backend/rtp_full_attention.py",
 "atom/rollout/async_engine.py", "atom/rollout/weight_updater.py", "atom/rollout/weight_sync.py",
 "atom/metrics/exporter.py", "atom/observability/tracing.py", "atom/benchmarks/benchmark_serving.py",
 "atom/utils/debug_helper/dump.py", "atom/utils/debug_helper/compare.py",
 "atom/diffusion/pipeline.py", "atom/diffusion/attention.py", "atom/diffusion/ulysses.py", "atom/diffusion/engine/*",
 "atom/diffusion/entrypoints/diffusion_server.py", "atom/diffusion/models/minimax_h3/pipeline.py",
 "atom/diffusion/models/minimax_h3/denoise.py", "atom/diffusion/models/minimax_h3/dit.py",
]

m = lambda f, pats: any(fnmatch.fnmatch(f, p) for p in pats)
files = sorted(glob.glob("atom/**/*.py", recursive=True))
lines = {f: sum(1 for _ in open(f)) for f in files}
code = {f: sum(1 for l in open(f) if l.strip()) for f in files}
sec_of = {}; idx_of = {}
for f in files:
    for sid, _, pats in SECTIONS:
        hit = [i for i, p in enumerate(pats) if fnmatch.fnmatch(f, p)]
        if hit:
            sec_of[f] = REMAP.get(sid, sid); idx_of[f] = hit[0] + (100 if sid in REMAP else 0); break
assert len(sec_of) == len(files), [f for f in files if f not in sec_of]
batch = lambda s: s if s == "X" else s.rstrip("abcdefg")
def tier(f):
    s = sec_of[f]
    if s == "X": return "不改"
    if code[f] <= 4: return "—"
    if f in A: return "A"
    b = int(batch(s))
    if b <= 7: return "C" if m(f, C_IN_EARLY) else "B"
    return "B" if m(f, B_IN_LATE) else "C"
missing_A = [f for f in A if f not in files]
assert not missing_A, missing_A

titles = {sid: t for sid, t, _ in SECTIONS if t}
order = [s for s, t, _ in SECTIONS if t and s != "X"] + ["X"]
BT = {"1": "入口、配置与请求对象", "2": "引擎主循环、多进程架构与服务层", "3": "调度器", "4": "KV cache 账本与状态缓存",
      "5": "ModelRunner：执行、权重加载、采样、CUDA Graph 与编译", "6": "基础算子、量化、注意力与 dense 模型",
      "7": "MoE、MLA、投机解码：DeepSeek-V3 与 MoE 模型家族", "8": "新一代模型：线性注意力、DeepSeek-V4 / V4.1 及其他",
      "9": "进阶执行与分布式：Mono 融合 decode、并行扩展、PD 分离与 KV offload",
      "10": "对外集成与周边：插件、RL rollout、可观测性、diffusion", "X": "不修改的文件"}

stats = collections.defaultdict(lambda: collections.Counter())
out = []
out.append("# ATOM 注释清单（逐文件）\n")
out.append("由脚本根据 [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md) 的分批规则生成，覆盖 `atom/` 下全部 "
           f"{len(files)} 个 .py 文件（{sum(lines.values()):,} 行）。注释完一个文件就勾选一项。\n")
out.append("深度：**A** 逐行详注；**B** 精注；**C** 导读；**—** 空文件或只有几行导出，不注释；**不改** 上游 vendored 代码。定义见计划第 1 节。\n")
cur_b = None
for s in order:
    rank = {"A": 0, "B": 1, "C": 2, "—": 3, "不改": 4}
    fs = sorted((f for f in files if sec_of[f] == s), key=lambda f: (rank[tier(f)], idx_of[f], f))
    b = batch(s)
    if b != cur_b:
        n = sum(1 for f in files if batch(sec_of[f]) == b); L = sum(lines[f] for f in files if batch(sec_of[f]) == b)
        out.append(f"\n## 第 {b} 批：{BT[b]}（{n} 个文件，{L:,} 行）\n" if b != "X" else f"\n## {BT[b]}（{n} 个文件，{L:,} 行）\n")
        cur_b = b
    if b != "X":
        out.append(f"\n### {s} {titles[s]}\n")
    for f in fs:
        t = tier(f); stats[b][t] += 1
        out.append(f"- [ ] {'**'+t+'**' if t in 'ABC' else t} `{f}`（{lines[f]} 行）")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
for b in [str(i) for i in range(1, 11)] + ["X"]:
    n = sum(1 for f in files if batch(sec_of[f]) == b); L = sum(lines[f] for f in files if batch(sec_of[f]) == b)
    print(b, n, L, dict(stats[b]))
for s in order:
    fs = [f for f in files if sec_of[f] == s]
    print("  ", s, len(fs), sum(lines[f] for f in fs))
