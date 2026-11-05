# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import gc
import inspect
import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, ClassVar, NamedTuple

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, init_dist_env
from aiter.dist.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
)
from aiter.dist.utils import get_distributed_init_method
from torch.profiler import record_function

from atom.config import Config, CUDAGraphMode, set_current_atom_config
from atom.distributed.pcp_utils import (
    PcpBalGroup,
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_pad_len,
    pcp_round_robin_split,
)
from atom.distributed.pp_comm import (
    async_send_intermediate_tensors,
    commit_pp_send_work,
    recv_intermediate_tensors,
)
from atom.distributed.simulated_tp import apply_simulated_tp, reject_simulated_tp
from atom.kv_transfer.disaggregation import (
    KVConnectorFactory,
    KVConnectorOutput,
    kv_config_has_producer,
)
from atom.metrics.gpu import GPUForwardMetrics, record_gpu_forward
from atom.model_engine.dynamic_chunking_worker import DynamicChunkingWorker
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.run_labels import build_run_label
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)
from atom.model_engine.state_runtime import StateRuntime
from atom.model_loader.loader import load_model
from atom.model_ops.attentions.pool_layout.entry_arena import plan_regions
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    InsufficientPoolBudget,
    Pool,
    PoolPlan,
    SubPoolSpec,
    plan_pools,
)
from atom.model_ops.decode_input_ids import (
    NEW_SEQUENCE,
    fill_deferred_decode_ids,
)
from atom.model_ops.eplb import (
    initialize_eplb_runtime,
    with_eplb_forward_monitor,
)
from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import SAMPLER_EPS, Sampler
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.factory import build_drafter
from atom.utils import (
    CpuGpuBuffer,
    envs,
    get_hf_text_config,
    init_exit_handler,
    resolve_obj_by_qualname,
    worker_process_name,
)
from atom.utils.cuda_graph import BatchDescriptor
from atom.utils.forward_context import (
    Context,
    DPMetadata,
    ForwardMode,
    get_forward_context,
    get_kvconnector,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)
from atom.utils.gc_utils import freeze_gc_heap
from atom.utils.h2d import h2d_producer
from atom.utils.selector import attn_family, get_attn_backend, has_mla_indexer
from atom.utils.tbo import (
    UBatchSlice,
    UBatchWrapper,
    local_tbo_precompute,
    maybe_create_ubatch_slices,
)

logger = logging.getLogger("atom")

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV41ForCausalLM": "atom.models.deepseek_v41.runtime.DeepseekV41RuntimeModel",
    "DeepseekV4ForCausalLM": "atom.models.deepseek_v4.DeepseekV4ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2.GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
    "Qwen4ExpForConditionalGeneration": (
        "atom.models.qwen4_exp.Qwen4ExpForConditionalGeneration"
    ),
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel",
    "Qwen3_5MoeForCausalLM": "atom.models.qwen3_5.Qwen3_5MoeForCausalLM",
    "KimiK25ForConditionalGeneration": "atom.models.kimi_k25.KimiK25ForCausalLM",
    "KimiK3ForConditionalGeneration": (
        "atom.models.kimi_k3.KimiK3ForConditionalGeneration"
    ),
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2.MiniMaxM2ForCausalLM",
    "MiMoV2ForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "MiMoV2FlashForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "Mistral3ForConditionalGeneration": "atom.models.mistral3.Mistral3TextOnly",
    "MistralForCausalLM": "atom.models.mistral3.Mistral3ForCausalLM",
    "MiniMaxM3SparseForCausalLM": "atom.models.minimax_m3.MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": "atom.models.minimax_m3.MiniMaxM3SparseForConditionalGeneration",
    "Glm5NextForConditionalGeneration": (
        "atom.models.glm5_next.Glm5NextForConditionalGeneration"
    ),
}
# Mono decode by wrapping the loaded model: the installer runs after the drafter
# armed its hooks on the unwrapped layers, and may add per-step buffers to the
# metadata builder before they are bound. Architectures not listed here route
# mono decode inside their own model class instead.
mono_decode_installers = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.mono.dispatch.install_mono_decode"
    ),
    "KimiK3ForConditionalGeneration": (
        "atom.models.kimi_k3.mono.dispatch.install_mono_decode"
    ),
}


def max_schedulable_decode_bs(
    max_num_seqs: int, max_num_batched_tokens: int, full_q_len: int
) -> int:
    """Largest decode batch the scheduler can admit.

    `Scheduler.schedule_decode` stops on either of two bounds: `max_num_seqs`
    sequences, or `max_num_batched_tokens` tokens. It charges every decode
    sequence the full speculative width `full_q_len` (== ``mtp_k + 1``) up
    front, whatever query length the step later replays, so dividing by
    `full_q_len` is exact rather than conservative — and it bounds every
    smaller q bucket too.

    Lives here so CUDAGraph capture can refuse to build a bucket the scheduler
    would never hand it. Must stay in step with `schedule_decode`'s
    `tokens_per_decode_seq`.
    """
    return min(max_num_seqs, max_num_batched_tokens // full_q_len)


# Re-exported under the old private name: this module is where the predicate
# used to live and where callers (and tests) still import it from.
_kv_config_has_producer = kv_config_has_producer


class TokenLocations(NamedTuple):
    """How each request in this decode batch gets its anchor token.

    `deferred_curr[k]` is a position in the CURRENT batch and
    `deferred_prev[k]` the row that request occupied in the previous forward,
    so its sampled id can be read from `prev_token_ids` without a D2H sync.
    `new_curr` holds the positions whose id the scheduler already put on the
    host. Together they cover the batch exactly once.

    A tuple of three same-typed arrays is easy to unpack in the wrong order;
    naming them makes that a typo the reader can see.
    """

    deferred_curr: np.ndarray
    deferred_prev: np.ndarray
    new_curr: np.ndarray


class tokenIDProcessor:

    def __init__(
        self,
        runner: "ModelRunner",
        max_num_batched_tokens: int,
        use_spec: bool = False,
        num_spec_tokens: int = 0,
    ):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        kv_cfg = getattr(runner.config, "kv_transfer_config", {}) or {}
        self.is_pipeline_parallel = (
            getattr(runner.config, "pipeline_parallel_size", 1) > 1
        )
        # The scheduler never speculates for a P/D producer, though the runner
        # keeps its drafter loaded: every decode it runs is q=1.
        self.is_pd_producer = _kv_config_has_producer(kv_cfg)
        # P/D hands off prompt-end state plus the first sampled token. Deferred
        # output makes the producer decode once more to surface that token, so
        # the consumer applies T0 to a state that already has it. Only a
        # recurrent state notices -- a paged write lands at T0's own position,
        # so repeating it is idempotent.
        hands_off_recurrent_state = (
            self.is_pd_producer and runner.attn_family.has_recurrent_state
        )
        self.is_deferred_out = (
            not self.is_pipeline_parallel and not hands_off_recurrent_state
        )

        self.runner = runner
        device = runner.device
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens + 1,
            dtype=torch.int32,
            device=device,
            publication_group="input_ids",
        )
        # One per request, not per token: where each request's anchor comes
        # from. Sized by tokens -- a batch can never hold more requests. The
        # matching prefix sum is `forward_vars["cu_seqlens_q"]`.
        self.decode_src = CpuGpuBuffer(
            max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
            publication_group="input_ids",
        )
        self.use_spec = use_spec
        self.num_spec_tokens = num_spec_tokens

        self.async_copy_stream = torch.cuda.Stream(runner.device)
        self.default_num_rejected_tokens = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.clean()

    def send_to_cpu_async(
        self,
        gpu_tensor: torch.Tensor,
        cpu_tensor_handle,
        data_ready: torch.cuda.Event,
        copy_done: torch.cuda.Event | None = None,
        gpu_logprobs: torch.Tensor | None = None,
    ):
        copy_done = copy_done or torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            cpu_logprobs = (
                gpu_logprobs.to("cpu", non_blocking=True)
                if gpu_logprobs is not None
                else None
            )
            copy_done.record(self.async_copy_stream)
        cpu_tensor_handle.append((cpu_tensor, copy_done))
        self.logprobs_cpu.append(cpu_logprobs)

    def recv_async_output(self, cpu_tensor_handle) -> torch.Tensor:
        if not cpu_tensor_handle:
            return torch.empty(0, dtype=torch.int32, device="cpu")
        cpu_tensor, event = cpu_tensor_handle.pop(0)
        event.synchronize()
        return cpu_tensor

    def recv_logprobs(self) -> list[float] | None:
        """Pop and return the earliest logprobs from the async copy queue.
        Must be called after recv_async_output (which synchronizes the event).
        """
        if not self.logprobs_cpu:
            return None
        logprob_tensor = self.logprobs_cpu.pop(0)
        if logprob_tensor is not None:
            return logprob_tensor.tolist()
        return None

    def send_to_cpu_async_draft(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.async_copy_stream)
        # No reverse wait here. `gpu_tensor` is a draft pass's captured output,
        # so its address is fixed and the NEXT replay rewrites it -- but that
        # replay sits behind a whole target forward on the default stream, while
        # this copy is a few KB. Ordering the default stream against it would
        # stall it for the copy every step, which is the overlap the side stream
        # exists to buy. If a future change ever puts a replay closer than one
        # forward away, the wait belongs immediately before THAT replay, not
        # here.
        self.draft_token_ids_cpu.append((cpu_tensor, event))

    def recv_async_output_draft(self) -> np.ndarray:
        if not self.draft_token_ids_cpu:
            return np.array([], dtype=np.int32)
        token_ids, event = self.draft_token_ids_cpu.pop(0)
        event.synchronize()
        return token_ids.numpy()

    def send_mtp_status_to_cpu_async(
        self,
        num_rejected: torch.Tensor,
        num_bonus: torch.Tensor,
        data_ready: torch.cuda.Event,
    ):
        # rejected num and bonus num are slightly different info for mtp
        # take mtp=1 for example:
        #   first decode after prefill have 0 rej, 0 bonus
        #   prev acc decode have 0 rej, 1 bonus
        #   prev rej decode have 1 rej, 0 bonus
        # It is clear that only rejected number is not sufficient for all status tracking, bonus number is also needed.
        # Single Event for both copies (vs. per-tensor send_to_cpu_async) so the
        # consumer pops one queue entry and synchronizes once instead of twice.
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_num_rejected = num_rejected.to("cpu", non_blocking=True)
            cpu_num_bonus = num_bonus.to("cpu", non_blocking=True)
            copy_done.record(self.async_copy_stream)
        self.pending_mtp_status_copies.append(
            (cpu_num_rejected, cpu_num_bonus, copy_done)
        )

    def recv_mtp_status_async(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.pending_mtp_status_copies:
            return None, None
        cpu_num_rejected, cpu_num_bonus, copy_done = self.pending_mtp_status_copies.pop(
            0
        )
        copy_done.synchronize()
        return cpu_num_rejected.numpy(), cpu_num_bonus.numpy()

    def clean(self):
        self.token_ids_cpu: list[torch.Tensor] = []
        self.logprobs_cpu: list[torch.Tensor | None] = []

        self.prev_batch: ScheduledBatch | None = None
        self.prev_token_ids: torch.Tensor | None = None

        self.pre_num_decode_token_per_seq = 1
        self.draft_token_ids: torch.Tensor | None = None
        self.draft_token_ids_cpu: list[torch.Tensor] = []
        # Queue of (cpu_num_rejected, cpu_num_bonus, copy_done_event) — async
        # D2H copies fired by send_mtp_status_to_cpu_async, drained by
        # recv_mtp_status_async after the event syncs.
        self.pending_mtp_status_copies: list[
            tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = []
        self.num_rejected: np.ndarray | None = None
        self.num_bonus: np.ndarray | None = None

    @staticmethod
    def _batch_process_token_ids(token_ids: list) -> list[tuple[int, ...]]:
        """Batch process token_ids: vectorized -1 truncation using numpy."""
        arr = np.array(token_ids, dtype=np.int64)
        mask = arr == -1
        if not mask.any():
            # No -1 sentinel in any row, convert each row to tuple directly
            return [tuple(row) for row in arr.tolist()]
        # Per-row: find first -1, truncate
        # Use argmax on mask; rows without -1 get 0, disambiguate with ~mask.any(axis=1)
        has_sentinel = mask.any(axis=1)
        first_neg = mask.argmax(axis=1)
        result = []
        rows = arr.tolist()
        for i, row in enumerate(rows):
            if has_sentinel[i]:
                result.append(tuple(row[: first_neg[i]]))
            else:
                result.append(tuple(row))
        return result

    def prepare_sampled_ids(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
        sampled_logprobs: torch.Tensor | None = None,
    ) -> tuple[dict[int, tuple[int, ...]], dict[int, float] | None]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed = self._batch_process_token_ids(token_ids)
            else:
                processed = [(tid,) for tid in token_ids]
            ret = dict(zip(req_ids, processed))
            ret[-1] = 0  # is_deferred_out flag
            logprobs_map = None
            if sampled_logprobs is not None:
                logprobs = sampled_logprobs.tolist()
                logprobs_map = {
                    seq_id: logprob for seq_id, logprob in zip(req_ids, logprobs)
                }
            return ret, logprobs_map

        token_ids = self.recv_async_output(self.token_ids_cpu)
        logprobs = self.recv_logprobs()
        self.send_to_cpu_async(
            sampled_token_ids,
            self.token_ids_cpu,
            sync_event,
            gpu_logprobs=sampled_logprobs,
        )
        token_id_dict = {}
        logprobs_map = None
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            token_ids_list = (
                token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            )
            if token_ids_list and isinstance(token_ids_list[0], list):
                processed = self._batch_process_token_ids(token_ids_list)
            else:
                processed = [(tid,) for tid in token_ids_list]
            token_id_dict = dict(zip(self.prev_req_ids, processed))
            if logprobs is not None:
                logprobs_map = {
                    seq_id: logprob
                    for seq_id, logprob in zip(self.prev_req_ids, logprobs)
                }
        else:
            # first time, no previous tokens
            token_ids = {}
            logprobs_map = None

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids
        token_id_dict[-1] = 1
        return token_id_dict, logprobs_map

    def get_token_locations(self, batch: ScheduledBatch) -> TokenLocations:
        prev_req_ids = self.prev_batch.req_ids
        cur_req_ids = batch.req_ids
        num_prev = len(prev_req_ids)
        num_cur = len(cur_req_ids)

        # A fabricated batch carries nothing over -- and the DP-sync dummy
        # reuses one id, so without this the next dummy matches it and reads a
        # request that never existed out of `prev_token_ids`/`draft_token_ids`.
        # Real ids are non-negative and never matched it either way.
        if self.prev_batch.is_dummy_run:
            none = np.empty(0, dtype=np.intp)
            return TokenLocations(none, none, np.arange(num_cur, dtype=np.intp))

        prev_id_to_idx = dict(zip(prev_req_ids, range(num_prev)))

        deferred_curr = np.empty(num_cur, dtype=np.intp)
        deferred_prev = np.empty(num_cur, dtype=np.intp)
        new_curr = np.empty(num_cur, dtype=np.intp)
        n_deferred = 0
        n_new = 0

        for cur_idx in range(num_cur):
            prev_idx = prev_id_to_idx.get(cur_req_ids[cur_idx])
            if prev_idx is not None:
                deferred_curr[n_deferred] = cur_idx
                deferred_prev[n_deferred] = prev_idx
                n_deferred += 1
            else:
                new_curr[n_new] = cur_idx
                n_new += 1

        deferred_curr = deferred_curr[:n_deferred]
        deferred_prev = deferred_prev[:n_deferred]
        new_curr = new_curr[:n_new]

        # Every request must be classified exactly once: `deferred_curr` and
        # `new_curr` are what the caller addresses the batch through, so a gap
        # or an overlap would leave a request reading whatever was staged for
        # it -- silently, and only in whatever batch shape produced the gap.
        assert (
            n_deferred + n_new == num_cur
        ), f"{n_deferred} deferred + {n_new} new != {num_cur} requests"
        return TokenLocations(deferred_curr, deferred_prev, new_curr)

    def _publish_input_ids(self, count, group):
        if group is None:
            return self.input_ids.copy_to_gpu(count)
        group.counts[group.indices["input_ids"]] = count
        group.counts[group.indices["decode_src"]] = None
        group.publish(group.counts)
        return self.input_ids.gpu[:count]

    @h2d_producer("input_ids", runner="runner")
    def prepare_input_ids(
        self,
        batch: ScheduledBatch,
        max_seqlen_q: int,
        *,
        publication_group=None,
    ) -> torch.Tensor:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.
        """
        scheduled_tokens = batch.scheduled_tokens  # tokens per req
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        # The MTP status queue is filled in postprocess but drained here, so a
        # step whose postprocess is skipped must not drain it: `forward()` bails
        # before postprocess when the batch produces no output (every prefill in
        # it is a middle chunk), and the status it popped belongs to the batch
        # whose deferred tokens the NEXT output-producing step will surface.
        # Draining it here would hand that step `num_rejected=None`.
        if batch.produces_output():
            self.prev_rejected_num, self.prev_bonus_num = self.recv_mtp_status_async()

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            # Decode does not publish an empty prefill prefix: an explicit
            # zero copy is still a publication in this forward's ledger.
            self.input_ids.np[:total_tokens_prefill] = scheduled_tokens[
                :total_tokens_prefill
            ]
            return self._publish_input_ids(total_tokens_prefill, publication_group)

        if not self.is_deferred_out:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            # No spec path supports pipeline parallel yet; it wants the deferred
            # branch's per-request staging. Gate on PP itself rather than on
            # `use_spec`: a P/D producer also reaches this branch, and the flag
            # is still set there (the runner keeps the drafter loaded to write
            # the draft's context KV at prefill) even though the scheduler
            # turned speculation off for it.
            if self.use_spec and self.is_pipeline_parallel:
                raise NotImplementedError("pipeline parallel + speculative decode")
            # PP verifies nothing locally and a P/D producer never speculates
            # (its DP dummy included), so no row here carries drafts. One that
            # did would need the deferred branch's per-request staging: under
            # DP a block drafter's `max_seqlen_q` is the group's maximum, not
            # this batch's row width.
            assert batch.num_spec_step == 0, (
                "non-deferred decode got a speculative batch "
                f"(num_spec_step={batch.num_spec_step})"
            )

            self.input_ids.np[:total_tokens_decode] = token_ids
            return self._publish_input_ids(total_tokens_decode, publication_group)

        # PD consumer first decode: no prior prefill step initialized
        # prev_batch, so use scheduled_tokens directly for this step.
        if self.prev_batch is None:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            return self._publish_input_ids(total_tokens_decode, publication_group)

        """for decode: input ids are from prev_sampled_token_ids"""
        locs = self.get_token_locations(batch)
        deferred_curr_indices = locs.deferred_curr
        deferred_prev_indices = locs.deferred_prev
        new_curr_indices = locs.new_curr
        num_deferred_seqs = len(deferred_curr_indices)
        num_new_seqs = len(new_curr_indices)

        # The GRAPH's row stride, for sizing the padded tail only. NOT a
        # per-request length: dp>1 raises it to the DP-wide maximum.
        tokens_per_seq = max_seqlen_q if self.use_spec else 1

        # Receive and map bonus_list to current batch order
        self.num_rejected = batch.num_rejected
        self.num_bonus = batch.num_bonus
        if num_deferred_seqs > 0 and self.prev_rejected_num is not None:
            # Remap prev step's rejected/bonus counts onto the current batch order
            # (prev_idx → curr_idx) for the deferred (carried-over) sequences.
            self.num_rejected[deferred_curr_indices] = self.prev_rejected_num[
                deferred_prev_indices
            ]
            self.num_bonus[deferred_curr_indices] = self.prev_bonus_num[
                deferred_prev_indices
            ]

        # ---- One path for every decode step -------------------------------
        # `num_scheduled_tokens` is the one statement of the per-request
        # lengths -- the ragged shrink rewrites it in place, so ragged and
        # rectangular are the same array. Its prefix sum is `cu_seqlens_q`,
        # already published; a second copy is what let the two disagree under DP.
        bs, lens, cu_np = self.runner.attn_metadata_builder.decode_spans(batch)

        # Stage the scheduler's ids over the whole region. For a request the
        # scheduler just admitted this is already its real anchor (and drafts);
        # for a carried-over one it is a placeholder the kernel overwrites.
        self.input_ids.np[:total_tokens_decode] = scheduled_tokens[
            total_tokens_prefill : total_tokens_prefill + total_tokens_decode
        ]
        # A newly admitted request has no row in `prev_token_ids`, so
        # `fill_deferred_decode_ids` skips it (`src < 0`) and its draft columns
        # must be staged here -- `scheduled_tokens` gave it committed history,
        # correct only at column 0. A batch without drafts (`num_spec_step ==
        # 0`: a P/D producer's decodes and DP dummies) has no draft table.
        #
        # Looped, not scattered: measured, the loop wins below ~24 admissions
        # per step (1.2us at 2) and a ragged scatter is a flat ~8us; above it
        # the scatter wins (12us vs 88us at bs=256). Swap if admissions stop
        # being few.
        if self.use_spec and batch.num_spec_step > 0 and num_new_seqs > 0:
            spec = batch.scheduled_spec_decode_tokens
            for i in new_curr_indices:
                n_draft = int(lens[i]) - 1
                if n_draft > 0:
                    s = int(cu_np[i]) + 1
                    self.input_ids.np[s : s + n_draft] = spec[i, :n_draft]
        src_np = self.decode_src.np[:bs]
        src_np.fill(NEW_SEQUENCE)
        src_np[deferred_curr_indices] = deferred_prev_indices
        group = (
            self.runner.h2d_groups["input_ids"]
            if publication_group is None
            else publication_group
        )
        counts = group.counts
        counts[group.indices["input_ids"]] = total_tokens_decode
        counts[group.indices["decode_src"]] = bs
        group.publish(counts)
        # How wide the forward reads: a replayed decode graph takes a fixed
        # `running_bs * tokens_per_seq` whatever the batch scheduled, and `bs`
        # sits between two captured buckets on most steps -- a 65-request batch
        # replays the 128 graph. The kernel zeroes the difference.
        fill_to = total_tokens_decode
        if not self.runner.enforce_eager:
            gbs = next(
                (g for g in reversed(self.runner.capture_sizes) if g >= bs), None
            )
            if gbs is not None:
                fill_to = max(fill_to, int(gbs) * tokens_per_seq)
        fill_deferred_decode_ids(
            self.input_ids.gpu,
            self.runner.forward_vars["cu_seqlens_q"].gpu[: bs + 1],
            self.decode_src.gpu[:bs],
            self.prev_token_ids,
            self.draft_token_ids if self.pre_num_decode_token_per_seq > 1 else None,
            max_tokens_per_seq=int(lens.max()) if bs else 1,
            width=fill_to,
        )

        # Slice by the width this path actually staged. Prefill returned above,
        # so the decode total is the whole batch; a worker-side speculative q
        # shrink rewrites it, and reading any other total here would leave the
        # attention metadata wider than input_ids.
        input_ids = self.input_ids.gpu[:total_tokens_decode]
        return input_ids

    def prepare_draft_ids(
        self, batch: ScheduledBatch, draft_token_ids: torch.Tensor
    ) -> np.ndarray:
        if not self.is_deferred_out:
            # propose() builds this on the drafter's device; the scheduler wants
            # host rows.
            ret = draft_token_ids.cpu().numpy()
        else:
            self.draft_token_ids = draft_token_ids
            self.pre_num_decode_token_per_seq = self.num_spec_tokens + 1
            token_ids = self.recv_async_output_draft()
            self.send_to_cpu_async_draft(draft_token_ids)
            ret = (
                token_ids
                if self.prev_req_ids is not None
                else np.array([], dtype=np.int32)
            )
        return ret


def _profile_runner_stage(method):
    """Annotate runner stages only while its torch profiler is active."""
    name = method.__name__
    label = f"ATOM::{name}"

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "profiler", None) is None:
            return method(self, *args, **kwargs)
        stage_label = label
        if name == "forward":
            batch = args[0] if args else kwargs["batch"]
            stage_label += (
                f" tokens={batch.total_tokens_num}"
                f" prefill={batch.total_seqs_num_prefill}"
                f" decode={batch.total_seqs_num_decode}"
            )
        with record_function(stage_label):
            return method(self, *args, **kwargs)

    return wrapped


# =====================================================================================
# ModelRunner：每张GPU一个进程里的"执行器"（第5批 5a，A级）
#   职责：初始化分布式 → 加载权重 → 预热测激活峰值 → 算KV预算 → 分配KV并绑定到各层 → 每步准备输入、forward、采样、异步回传
#   谁创建：AsyncIOProcManager为本stage的每个rank（共 tp_world_size × pcp 个）spawn一个worker进程，在其中构造
#     ModelRunner(rank, config)；类由config.runner_qualname决定（RapidServe时是本文件末尾的RapidServeModelRunner），见async_proc.py
#   谁调用：EngineCore用 runner_mgr.call_func("方法名", ...) 远程调用，命令经共享内存MessageQueue广播给每个rank，
#     每个rank执行同一个方法，只有rank 0回包（四种调用方式对照表见async_proc.py；为什么必须SPMD见engine_core.py文件开头【4】）
#
# 生命周期（每个rank都走一遍；冒号前是RPC名，也就是方法名）：
#   ┌─ __init__（本区段）────────────────────────────────────────────────────────────────┐
#   │ ① 读配置：block_size、KV dtype、注意力家族（MHA/MLA/GDN/V4…）与对应的注意力后端             │
#   │ ② _setup_device_and_distributed：绑定GPU，建NCCL world与TP/DP/PP/PCP/DCP等通信组            │
#   │ ③ tokenIDProcessor（延迟输出与input_ids组装）、Sampler                                    │
#   │ ④ 量化表层名重映射 → _build_and_load_model：构造模型 + load_model读权重（第5批5c）          │
#   │ ⑤ 【分支】草稿模型与拒绝采样器（投机解码，第7批）                                           │
#   │ ⑥ allocate_forward_vars + 注意力元数据builder：每步输入用的固定地址buffer                  │
#   │ ⑦ 【分支】TBO / Mono包装；forward_vars环与H2D发布组                                        │
#   │ ⑧ _maybe_warmup → warmup_model：一次满额的假prefill，让分配器记下激活峰值                  │
#   └──────────────────────────────────────────────────────────────────────────────────────┘
#        ▼ EngineCore.__init__依次RPC（engine_core.py __init__上方的启动流程图）
#   get_num_blocks：显存预算 → 每类子池放多少条目（本区段，带显存预算条）
#        ▼
#   allocate_kv_cache(num_blocks)：一次分配整块KV字节池，按层切片绑定到注意力模块（本区段）
#        ▼
#   [profile_dynamic_chunking]（【分支】动态分块，第9批）→ capture_cudagraph（录图，见该方法）→ freeze_gc_heap
#        ▼ 之后每一步
#   forward(batch)：prepare_model / prepare_inputs → run_model → postprocess（采样 + 异步回传），见本文件后半
#   DP>1且本rank这一步没活：dummy_execution（本区段）陪其他rank走完集合通信
#
# 本区段（__init__ 到 prepare_inputs 之前）的阅读顺序：
#   __init__ → _setup_device_and_distributed → warmup_model → get_num_blocks（预算条）→ allocate_kv_cache
#   → allocate_forward_vars / _init_forward_vars_ring / _init_h2d_publication（每步输入buffer怎样安全复用）
#   标【分支】的（profiler、dummy、TBO、DSpark、PP、RapidServe）第一遍可以跳过
#
# 对比nano-vllm：
#   nano-vllm的ModelRunner.__init__一口气做完 init_process_group → 建模型+load_model → warmup → allocate_kv_cache → capture_cudagraph，
#     rank 0就是主进程，由LLMEngine直接调用
#   ATOM的__init__只做到warmup为止；KV预算、KV分配、录图拆成三个RPC由EngineCore依次调用——block数要回传给EngineCore
#     去建Scheduler（Config跨进程是值传递，见config.py文件开头），中间还要能插入PD分离的握手、动态分块的测量
#   每个rank（包括rank 0）都是EngineCore之外的子进程；KV布局、每步元数据交给"注意力后端的builder"决定，runner只做调度与记账
# =====================================================================================
class ModelRunner:

    # 构造（在worker子进程里执行，耗时最长：建通信组、读权重、预热都在这里）
    #   rank：本worker在本stage里的编号，0 .. tp_world_size × pcp − 1（同一stage = 同一个EngineCore）
    #   config：EngineCore传来的Config（经pickle的副本，改它不影响其他进程，见config.py文件开头）
    #   步骤与类上方生命周期框图的①~⑧对应；成员变量的用途写在各自上方
    def __init__(self, rank: int, config: Config):
        self.config = config
        # 【分支】多模态：视觉编码结果按请求缓存，chunked prefill时分段散射进输入embedding
        #   （multimodal_runtime.py文件头【科普】；span路径只有DeepSeek-V4.1，第8批）
        from atom.model_engine.multimodal_runtime import VisionEmbeddingCache

        self.vision_embeddings = VisionEmbeddingCache()
        # 【分支】profiling：--mark-trace时打开graph_marker（utils/graph_marker.py，第5批C级）——模型里的graph_marker(x, name)
        #   才会真正插入一个不改数据的自定义算子，在编译后的图和profiler trace里标出位置；关闭时直接返回x
        self.mark_trace = getattr(config, "mark_trace", False)
        from atom.utils.graph_marker import set_graph_marker_enabled

        set_graph_marker_enabled(self.mark_trace)
        # 设成本进程的"当前Config"：之后构造的模型层、注意力后端、算子都用get_current_atom_config()读它（config.py）
        set_current_atom_config(config)
        hf_config = config.hf_config
        # 调度器的block大小（每个KV block装多少个token，默认16），已按模型改写过（V4=256、V4.1=128、GLM-5.3=index_kpool×16，
        #   config.py __post_init__第12、13步）；slot_mapping、block_table、KV池的形状都按它算
        self.block_size = config.kv_cache_block_size
        # KV cache的元素类型："bf16"或"fp8"。fp8时MHA每个(层, KV头, token)还要多存一个fp32 scale（见get_num_blocks上方②）
        self.kv_cache_dtype = config.kv_cache_dtype
        # True：不录CUDA Graph（EngineCore据此不调用capture_cudagraph，_estimate_cudagraph_overhead也返回0）；torch.compile照常
        self.enforce_eager = config.enforce_eager
        # world_size: the logical TP width, i.e. how many shards each weight is
        # cut into -- what the KV-head math below divides by.
        # tp_world_size: how many of those shards have a process.
        # They differ only under simulated TP.
        # 【译】world_size：逻辑上的TP宽度，即每个权重被切成几份——下面KV头数的计算除的就是它。
        #   tp_world_size：这些分片里有几份真正有进程。两者只在模拟TP下不同。
        # 解释：模拟TP（--fake-eplb下 -tp 8 但本机只有4张卡）只跑8卡部署里的前4个rank，每张卡的形状与显存和真实TP8一致，
        #   用来测kernel性能，输出无意义（distributed/simulated_tp.py文件头，第9批C级）
        self.world_size = config.tensor_parallel_size
        self.tp_world_size = config.tp_world_size
        self.rank = rank
        self.label = f"Model Runner{rank}/{self.tp_world_size}"
        # 【分支】动态分块的PP prefill（chunk大小按延迟模型求解）在runner侧的部分：启动时扫描测延迟、运行时计时，
        #   见第9批dynamic_chunking_worker.py；概念见engine_core.py里"【科普】动态分块"
        self._dynamic_chunking = DynamicChunkingWorker(self)
        # 文本模型的HF config：多模态模型取text_config，纯文本模型就是hf_config本身（层数、头数等都从这里读）
        self.hf_text_config = get_hf_text_config(hf_config)
        # Llama + bf16/fp16时设AITER_QUICK_REDUCE_QUANTIZATION=INT4：这是aiter自己的环境变量（envs.py末尾的第三方变量表），
        #   由aiter TP通信器里的QuickAllReduce（aiter.dist.device_communicators.quick_all_reduce）读取，
        #   INT4即把all-reduce要传的数据量化到4位，以少量精度换通信带宽；只对这两种模型/dtype组合打开
        if self.hf_text_config.model_type in ["llama"] and self.config.torch_dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
        # Which attention this model's config asks for. The runner holds the
        # answer, not the question: what makes a model MLA or a hybrid is the
        # selector's to know, and it is asked the same way of a draft.
        # 【译】这个模型的config要求哪种注意力。runner只保存答案，不负责提问：一个模型算MLA还是混合模型，
        #   由selector来判断，而且对草稿模型也用同样的方式提问。
        # 解释：attn_family返回utils/selector.py的Family枚举（按"最具体的优先"判断，第6批）：
        #   CSA2（DeepSeek-V4.1）、V4、QSA_GDN（Qwen4-exp）、KIMI_MLA（Kimi-K3等MLA+KDA混合）、MLA（有kv_lora_rank，
        #   DeepSeek-V3/V3.2、GLM-5等）、GDN（Qwen3-Next等全注意力+GDN混合）、MHA（其余，含GQA）。
        #   它决定用哪个注意力后端（下面的attn_backend）、KV池怎么布局、有没有每请求递归状态（Family.has_recurrent_state）
        self.attn_family = attn_family(self.hf_text_config)

        rope_parameters = getattr(self.hf_text_config, "rope_parameters", None) or {}
        # 【分支】多模态的M-RoPE（位置拆成时间/高/宽3个分量）：positions变成[3, N]，见allocate_forward_vars里的mrope_positions
        self.use_mrope = "mrope_section" in rope_parameters
        # MLA + DSA稀疏索引器（DeepSeek-V3.2、GLM-5等，config里有index_topk）：KV池的每个block多一段索引key缓存
        #   （DSA是什么见arg_utils.py index_cache_dtype附近的【科普】DSA；字节数见get_num_blocks上方②(b)）
        self.has_mla_indexer = has_mla_indexer(self.hf_text_config)
        # Initialize profiler for this rank (before _setup_device_and_distributed
        # so that dp config fields are still at their original values)
        # 【译】为本rank初始化profiler（放在_setup_device_and_distributed之前，这样DP相关的config字段还是原始值）
        # 观察：按v0.1.8的代码，_setup_device_and_distributed只读不写parallel_config的DP字段；--enable-dp-attention
        #   对DP字段的改写早在CoreManager里就完成了，ModelRunner看到的已是改写后的值
        #   （例：-tp 8 --enable-dp-attention → 8个EngineCore，rank_name依次是dp0_tp0 … dp7_tp0）
        # 【分支】torch profiler（start_profiler / stop_profiler，B级）：profiler在start_profiler里才创建
        self.profiler = None
        self.profiler_dir = None
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        if dp_rank_local > 0 or config.parallel_config.data_parallel_size > 1:
            self.rank_name = f"dp{dp_rank_local}_tp{rank}"
        else:
            self.rank_name = f"rank_{rank}"
        if config.torch_profiler_dir is not None:
            rank_name = self.rank_name
            if config.pipeline_parallel_size > 1:
                rank_name = (
                    f"pp{config.parallel_config.pipeline_parallel_rank}_{rank_name}"
                )
            self.profiler_dir = os.path.join(config.torch_profiler_dir, rank_name)
            os.makedirs(self.profiler_dir, exist_ok=True)

        # 注意力后端类（AttentionBackend子类，按family懒加载，例：MHA → aiter_attention.AiterBackend，MLA → aiter_mla.AiterMLABackend）。
        #   它的get_builder_cls()在下面构造"注意力元数据builder"：KV池怎么布局、每步的slot_mapping/block_table怎么准备，
        #   都由builder决定，ModelRunner只负责调用（第6批backends.py）
        self.attn_backend = get_attn_backend(self.attn_family)
        # ②：绑定GPU、建NCCL world和各并行通信组（见该方法上方）；必须在构造任何模型层之前，因为层的切分要读TP组大小
        self._setup_device_and_distributed(rank, config)

        # CUDA Graph的捕获档位（decode的batch size阶梯）。[0]是占位：还没录图（或enforce_eager）时只有0档，
        #   ForwardMode.decide找不到合适档位就走eager；capture_cudagraph录完后换成真正录到的档位（第5批capture_cudagraph）
        self.capture_sizes = [0]  # for eager fallback
        # The same ladder as an ASCENDING int32 array, which is what
        # `ForwardMode.decide` searches. Separate from the list because the list
        # is re-sorted in both directions during capture; rebound (never
        # mutated) once capture narrows it, so a search cannot read a transient
        # order.
        self.capture_sizes_np = np.asarray(self.capture_sizes, dtype=np.int32)
        # PIECEWISE cudagraph state, populated by capture_cudagraph. Empty when
        # capture never ran (enforce_eager), so the ragged-bucket paths no-op.
        # 【译】PIECEWISE cudagraph的状态，由capture_cudagraph填充。从没录过图（enforce_eager）时为空，于是ragged分桶路径什么都不做。
        # 解释：PIECEWISE模式按"本步总token数"分档录图，这里记下实际录到的token数档位（集合 + 排好序的列表）
        self._piecewise_captured_tokens: set[int] = set()
        self._piecewise_sorted_tokens: list[int] = []

        # 注册退出处理（utils/__init__.py）：置still_running=True；weakref.finalize在对象被回收时调用exit()；
        #   子进程忽略SIGINT（Ctrl+C交给主进程统一编排），收到SIGTERM时走exit()
        init_exit_handler(self)
        # 与nano-vllm相同的技巧：临时把默认dtype设成模型dtype、默认device设成本rank的GPU，
        #   之后创建的模型参数、buffer直接以模型dtype建在GPU上，省去.to()拷贝；构造和预热结束后再恢复（见下面"Restore the implicit CPU default"）
        default_dtype = self.config.torch_dtype
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(self.device)
        # 本rank是否参与投机解码：开了speculative_config，且是PP的最后一个stage（只有最后一个stage出logits、采样、跑草稿模型）
        use_spec = bool(self.config.speculative_config) and get_pp_group().is_last_rank
        # 每步每条序列的草稿token数k（MTP的k）；decode时每条序列的query长度是 1 + k
        self.num_spec_tokens = (
            self.config.speculative_config.num_speculative_tokens if use_spec else 0
        )

        # 【分支】PP：异步发往下一个stage的中间张量（hidden states）还没确认完成的send工作，下一次发送前或flush_pp_send时提交（pp_comm.py，第9批）
        self._pp_pending_send: list = []
        # ③ 延迟输出的GPU侧 + input_ids组装（tokenIDProcessor类，见本文件开头；原理见engine_core.py文件开头【3】）：
        #   它持有input_ids/decode_src这两个CpuGpuBuffer和独立的async_copy_stream（采样结果的D2H拷贝排在这条stream上）
        self.tokenID_processor = tokenIDProcessor(
            self,
            self.config.max_num_batched_tokens,
            use_spec,
            self.num_spec_tokens,
        )
        # 采样器：logits → token id（温度、top-k/top-p；第5批sampler.py）。投机解码的拒绝采样器在下面的草稿分支里建
        self.sampler = Sampler()
        # 预先建好的 [0, 1, 2, …] 数组，长度取"序列数+1、最大上下文、单步最多token数"三者的最大值，
        #   准备输入时直接切片当arange用（例如_get_cumsum_and_arange），避免每步np.arange重新分配
        self.arange_np = np.arange(
            max(
                self.config.max_num_seqs + 1,
                self.config.max_model_len,
                self.config.max_num_batched_tokens,
            ),
            dtype=np.int64,
        )

        # ④ 由HF config的architectures[0]查上面的support_model_arch_dict，拿到ATOM的模型类
        #   例："Qwen3ForCausalLM" → atom.models.qwen3.Qwen3ForCausalLM；DeepSeek-V3/V3.2共用deepseek_v2.DeepseekV2ForCausalLM
        model_class = resolve_obj_by_qualname(support_model_arch_dict[hf_config.architectures[0]])  # type: ignore
        # The model construction depends on quant_config,
        # so we must complete the remapping for layers before constructing the model.
        # 【译】模型的构造依赖quant_config，所以必须在构造模型之前完成层名的重映射。
        # 解释：ATOM把q/k/v、gate/up等投影融合成一个参数（qkv_proj、gate_up_proj），而checkpoint的量化表按融合前的名字写；
        #   每个线性层构造时按自己的（融合后的）名字查量化规格，所以要先把表里的名字改成融合后的
        #   （例："model.layers.0.mlp.gate_proj" → "...gate_up_proj"，见config.py QuantizationConfig.remap_layer_name上方）
        config.quant_config.remap_layer_name(
            config.hf_config,
            packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_class, "quant_exclude_name_mapping", {}
            ),
        )

        # 构造模型并从磁盘加载本rank的那份权重（见该方法上方）
        self._build_and_load_model(model_class)

        # Optional debug instrumentation; no-op when env vars unset.
        # See atom/utils/debug_helper/.
        # 【译】可选的调试插桩；相关环境变量没设置时什么也不做。见 atom/utils/debug_helper/。
        # 【分支】ATOM_FWD_DUMP等：给每个Block挂forward hook转储中间张量、或转储权重后直接退出（第10批debug_helper）
        from atom.utils.debug_helper import (
            install_block_forward_hooks,
            maybe_dump_weights_and_exit,
        )

        _n_fwd_hooks = install_block_forward_hooks(self.model)
        if _n_fwd_hooks > 0:
            logger.info(f"[ATOM_FWD_DUMP] {_n_fwd_hooks} Block forward hooks installed")
        maybe_dump_weights_and_exit(self.model)

        # ⑤ 【分支】投机解码（MTP / EAGLE3 / DSpark，第7批）：只在PP最后一个stage建草稿模型
        #   set_model_tag("drafter")：草稿模型的torch.compile缓存单独按tag分目录（utils/backends.py）
        #   build_drafter：spec_decode/factory.py按方法选EagleProposer（MTP、EAGLE3）或DSparkProposer
        #   RejectionSampler：验证草稿token、决定接受几个（第7批rejection_sampler.py）
        #   drafter.load_model(self.model)：加载草稿权重；MTP / V4 DSpark / Kimi-K3 DSpark与目标模型共享embedding和lm_head，
        #     EAGLE3用自己独立的（spec_decode/drafter.py）
        if self.config.speculative_config and get_pp_group().is_last_rank:
            from atom.utils.backends import set_model_tag

            torch.set_default_device(self.device)
            with set_model_tag("drafter"):
                self.drafter = build_drafter(self.config, self.device, self)
            self.rejection_sampler = RejectionSampler(
                synthetic_acceptance_rates=(
                    self.config.speculative_config.synthetic_acceptance_rates
                )
            )
            torch.set_default_device(None)
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)
            # NOTE: aux-hidden-state capture is armed AFTER the optional TBO wrap
            # below, so the drafter's capture hook lands on the object whose
            # forward returns the final (concatenated) output. See arm_aux_capture.
            # 【译】注意：辅助hidden state的捕获在下面可选的TBO包装之后才挂上，这样草稿器的捕获hook落在"forward返回
            #   最终（拼接后）输出"的那个对象上。见arm_aux_capture。
            # 解释：EAGLE3要用目标模型若干中间层的hidden state（aux hidden states）；TBO把batch拆成两半各跑一遍，
            #   包装器的forward才返回拼好的完整输出

        torch.set_default_device(self.device)
        # 一条独立的GPU stream（stream = GPU上的FIFO命令队列，见engine_core.py文件开头【3】的前置）
        # 观察：它只被aiter_mla.py、deepseek_v4_attn.py里builder的prep_stream属性返回，而v0.1.8全仓库没有任何地方读取prep_stream，
        #   这条stream实际上没有被使用
        self.async_execute_stream = torch.cuda.Stream(self.device)
        # ⑥ 每步forward输入的固定buffer（positions、采样参数、模型输出等，见该方法上方）；必须在builder之前，
        #   builder构造时会往同一个forward_vars字典里追加注意力元数据buffer（slot_mapping、block_tables、cu_seqlens_q/k…）
        self.allocate_forward_vars()
        # 注意力元数据builder（attn_backend.get_builder_cls()的实例，第6批backends.py）：
        #   ① 构造时向forward_vars追加自己的CpuGpuBuffer；② 声明KV池的子池规格（sub_pool_specs，get_num_blocks用）；
        #   ③ allocate_kv_cache时在给它的字节区里布局、并把每层的视图绑到注意力模块（build_kv_cache_tensor）；
        #   ④ 每步把ScheduledBatch翻译成注意力要的元数据（prepare_prefill / prepare_decode）
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            model_runner=self
        )
        # Sub-pool sizing needs a memory profile, so it cannot run until after
        # warmup. Install the empty plan now: `warmup_model` below drives the
        # builder through paths that ask for their entry counts, and those must
        # read 0 ("no pool yet") rather than trip over a missing attribute.
        # 【译】子池的规模计算需要一次显存profile，所以要等warmup之后才能做。先装一个空计划：下面的warmup_model
        #   会驱动builder走一些查询条目数的路径，它们必须读到0（"还没有池"），而不是因为属性不存在而出错。
        # 解释：pool_plan = 各子池（PAGE的KV block、STATE的每请求状态槽位）各有多少条目、每条多少字节；get_num_blocks里替换成真的
        self.pool_plan = PoolPlan.empty()
        # 每请求状态（GDN/KDA递归状态、V4滑动窗口等）怎样checkpoint/转移的运行时描述，默认"没有状态"；get_num_blocks里替换
        #   （state_runtime.py，第4批）
        self.state_runtime = StateRuntime()
        # Sanity-check: any builder that allocates a per-request cache must
        # have its model_type listed in `InputOutputProcessor`'s
        # `per_req_cache_model_types` set; otherwise sequences will be
        # constructed with `has_per_req_cache=False`, the BlockManager will
        # never assign them a slot, and the builder will silently read
        # tensor[-1] on first decode. Catch the misconfiguration up front
        # rather than producing wrong outputs at inference time.
        # 【译】完整性检查：任何分配了每请求cache的builder，其model_type都必须列在InputOutputProcessor的
        #   per_req_cache_model_types集合里；否则序列构造时has_per_req_cache=False，BlockManager永远不会给它们分配槽位，
        #   builder在第一次decode时会静默地读tensor[-1]。在启动时就抓住这种配置错误，而不是在推理时产生错误输出。
        # 解释：has_per_req_cache在前端进程构造Sequence时就决定了（llm_engine.py，第2批），runner这里只能核对、不能补救
        if self._state_pool_names():
            from atom.model_engine.llm_engine import InputOutputProcessor as _IOProc

            mt = self.config.hf_config.model_type
            known = _IOProc._per_req_cache_model_types()
            assert mt in known, (
                f"Attention builder {type(self.attn_metadata_builder).__name__} "
                f"declares a per-request state pool but model_type={mt!r} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({sorted(known)}). "
                "Add it to the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
        # ⑦ 【分支】TBO（两个微batch交错，通信与计算重叠；概念见arg_utils.py enable_tbo上方的【科普】）：
        #   用UBatchWrapper包住模型，forward时按_maybe_create_tbo_slices切出的两段分别执行（utils/tbo/，第9批）
        #   dp_gather_scatter：DP-attention但没开EP时，MoE前后要在DP rank间gather/scatter token
        if config.enable_tbo:
            dp_gather_scatter = (
                config.enable_dp_attention and not config.enable_expert_parallel
            )
            self.model = UBatchWrapper(
                self.model,
                attn_metadata_builder=self.attn_metadata_builder,
                dp_gather_scatter=dp_gather_scatter,
            )
            logger.info("TBO enabled: model wrapped with UBatchWrapper")
        # 草稿器在（可能已被TBO包装的）模型上挂辅助hidden state的捕获hook（上面NOTE说明了顺序为什么要在TBO之后）
        if getattr(self, "drafter", None) is not None:
            self.drafter.arm_aux_capture(self.model)
        # 【分支】Mono融合decode（每层一个融合kernel跑完decode，第9批）：DeepSeek-V4.1、Kimi-K3在加载后包装模型
        #   （installer见文件开头mono_decode_installers）；其他架构的mono路由在模型类内部
        installer = mono_decode_installers.get(hf_config.architectures[0])
        if installer is not None:
            self.model = resolve_obj_by_qualname(installer)(
                self.model,
                config,
                getattr(self, "drafter", None),
                self.attn_metadata_builder,
            )
        # 所有CpuGpuBuffer此时都已建好（runner的、builder的、mono installer可能追加的），才能：
        #   PP>1时克隆出每个在途微batch一份的forward_vars环；再把每个buffer登记进H2D发布组（两个方法上方有图）
        self._init_forward_vars_ring()
        self._init_h2d_publication()
        # 每次forward的采样结果写完后在当前stream上record（postprocess）；tokenIDProcessor的异步D2H拷贝在
        #   async_copy_stream上先wait这个event，保证拷走的是写完的token（延迟输出，engine_core.py文件开头【3】）
        self.forward_done_event = torch.cuda.Event()
        # 【分支】EPLB（专家负载均衡，第7批eplb.py）：开启时把本runner登记为运行时owner，没开时什么也不做
        initialize_eplb_runtime(self)
        # ⑧ 预热：跑一次满额的假prefill，让分配器记下激活峰值，get_num_blocks靠它算KV预算（见warmup_model上方）
        self._maybe_warmup()

        # Restore the implicit CPU default. An explicit "cpu" leaves a
        # DeviceContext intercepting every torch call during serving, including
        # tensor views and Triton's pointer/stride specialization queries.
        # 【译】恢复隐式的CPU默认值。显式设成"cpu"会留下一个DeviceContext，在服务期间拦截每一次torch调用，
        #   包括张量view以及Triton对指针/stride做特化时的查询。
        # 对比nano-vllm：nano-vllm在这里写的是torch.set_default_device("cpu")；ATOM改成None，原因就是上面这段：
        #   设成任何显式device都会让PyTorch在每次创建张量时多走一层DeviceContext分派，热路径上的开销不可忽略
        torch.set_default_device(None)
        torch.set_default_dtype(default_dtype)

        # --level 1（DYNAMO_AS_IS）：对整个模型直接torch.compile(backend="eager")，只做Dynamo抓图、不生成新kernel，
        #   用来暴露graph break；生产默认的--level 3由模型类上的@support_torch_compile在第一次调用时编译
        #   （各级别的落点见config.py CompilationLevel上方，实现见第5批utils/decorators.py、utils/backends.py）
        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
            if hasattr(self, "drafter"):
                self.drafter.model = torch.compile(
                    self.drafter.model, fullgraph=True, backend="eager"
                )

        # Install after initialization warmup, which is not request traffic.
        # Graph capture runs later via RPC and bypasses run_model, so its
        # timing decorator does not run during capture.
        # 【译】在初始化预热之后才装上，因为预热不是请求流量。录图稍后经RPC执行、不经过run_model，所以录图期间
        #   不会运行它的计时装饰器。
        # 【分支】ATOM_ENABLE_METRICS_DEVICE_TIMER=1（默认关）时用CUDA event给每次目标模型forward计GPU时间，
        #   导出为atom:gpu_forward_seconds直方图（metrics/gpu.py，第10批）；EngineUtilityHandler定期RPC poll_forward_metrics收集
        self.gpu_forward_metrics = None
        if envs.ATOM_ENABLE_METRICS_DEVICE_TIMER:
            self.gpu_forward_metrics = GPUForwardMetrics(
                lambda: torch.cuda.Event(enable_timing=True),
                dp_rank=config.parallel_config.data_parallel_rank,
                pp_rank=config.parallel_config.pipeline_parallel_rank,
                tp_rank=self.rank,
                engine_role=(
                    ("decode" if config.disagg_is_decode else "prefill")
                    if config.enable_rapidserve
                    else "default"
                ),
            )

    # 【分支】GPU forward计时指标：把已完成的CUDA event读出来记进直方图（metrics/gpu.py，第10批）
    def poll_forward_metrics(self):
        if self.gpu_forward_metrics is not None:
            self.gpu_forward_metrics.poll()

    # 构造模型 + 从checkpoint加载本rank的权重分片（__init__步骤④）
    #   前置：TP的列切/行切、q/k/v与gate/up融合成一个参数、checkpoint里的q_proj怎样写进qkv_proj的第0段，
    #     这些是权重加载的内容，规范讲解在model_loader/loader.py（第5批5c）；TP切分的直觉见arg_utils.py tensor_parallel_size上方的【科普】
    #   对比nano-vllm：nano-vllm是 Qwen3ForCausalLM(hf_config) + load_model(model, path)；ATOM按架构查表选模型类，
    #     并支持假权重（--load-dummy）、融合共享专家等钩子；RapidServe的decode进程覆写本方法，参数建在meta设备上、权重经CUDA IPC导入
    def _build_and_load_model(self, model_class):
        """Construct the model and load its weights from disk.

        Override point: subclasses (e.g. the rapidserve decode process) may
        construct on the meta device and import weights via IPC instead.
        """
        # 【译】构造模型并从磁盘加载它的权重。
        #
        #   覆写点：子类（例如rapidserve的decode进程）可以改为在meta设备上构造，再通过IPC导入权重。
        config = self.config
        # 构造模型：此时默认device已是本rank的GPU、默认dtype是模型dtype（见__init__），参数直接以未初始化的值建在GPU上，
        #   每个线性层只建本rank那一份（TP切分在层的构造函数里按get_tp_group().world_size算）
        self.model = model_class(config)
        # 【分支】有load_fused_expert_weights方法的模型（v0.1.8里是Qwen3.5）把它交给loader，用来加载融合后的专家权重
        fused_shared_expert_load_fn = None
        if hasattr(self.model, "load_fused_expert_weights"):
            fused_shared_expert_load_fn = self.model.load_fused_expert_weights
        # 加载期间不设默认device：loader读safetensors得到的是CPU张量，再按参数各自的weight_loader拷进GPU上的参数
        #   （__init__后面会再把默认device设回GPU）
        torch.set_default_device(None)
        load_start = time.perf_counter()
        # 参数：模型、checkpoint路径、HF config、load_dummy（empty/zero/xavier等假权重方式，只测性能时用；None = 读真实权重）
        load_model(
            self.model,
            config.model,
            config.hf_config,
            config.load_dummy,
            load_fused_expert_weights_fn=fused_shared_expert_load_fn,
        )
        load_elapsed = time.perf_counter() - load_start
        logger.info(
            f"[{self.rank_name}] Model load done: {config.model} "
            f"(weights loaded in {load_elapsed:.2f}s)"
        )

    def _maybe_warmup(self):
        """Run model warmup. Override point: the rapidserve decode process
        skips warmup since it imports weights/kvcache from prefill later."""
        # 【译】执行模型预热。覆写点：rapidserve的decode进程跳过预热，因为它稍后才从prefill导入权重和KV cache。
        self.warmup_model()
        logger.info(f"Model warmup done: {self.config.model}")

    def _kv_budget_extra_reserve(self, total_bytes: int) -> int:
        """Extra GPU bytes to hold back from the KV cache budget beyond the
        base overhead. Base runner reserves nothing; override point for
        setups that share the GPU with another process."""
        # 【译】在基本开销之外，再从KV cache预算里额外留出的GPU字节数。基类runner不留；这是给"与另一个进程共用GPU"的部署准备的覆写点。
        # 解释：RapidServeModelRunner覆写为 4 × (total × 2%) = 8%（prefill、decode两个进程共用同一组GPU），见get_num_blocks的预算条
        return 0

    # ② 绑定GPU、建立分布式通信（__init__里第一件重活，必须早于构造任何模型层）
    #   前置：集合通信、进程组（process group）、world、TCPStore集合点是什么，见engine_core_mgr.py _start_distributed_store里的
    #     【科普】（一句话：进程组 = 一起做集合通信的那组rank；world = 参加同一次init_process_group的全部rank；
    #     TP组、DP组、PP组等都是world的子集，组里每个rank都必须调用同一个集合操作才会返回）
    #
    #   ATOM一个NCCL world里的rank排布（外层到内层：DP → PP → PCP×TP），代码里的两条公式：
    #     global_rank       = (dp_rank × pp + pp_rank) × stage_span + rank           stage_span = tp_world_size × pcp
    #     local_device_rank = (dp_rank_local × pp + pp_rank) × stage_span + rank     （本机物理GPU号）
    #   例1：单机8卡 -dp 2 -tp 4（不开dp-attention）：stage_span = 4，world_size = 2 × 1 × 4 = 8
    #       GPU号：        0   1   2   3 │ 4   5   6   7
    #       DP rank：      0   0   0   0 │ 1   1   1   1      ← 两个EngineCore，各有自己的调度器
    #       rank（TP内）：  0   1   2   3 │ 0   1   2   3      ← 每个EngineCore下4个ModelRunner进程
    #       TP组：{0,1,2,3}、{4,5,6,7}（每层的all-reduce在组内做）；DP组大小为2（按这种排布是{0,4}、{1,5}…）
    #   例2：单机 -pp 2 -tp 4：PP的每个stage是独立的EngineCore，engine_index = pp_rank，stage 1在GPU 4~7
    #   例3：-tp 4 --prefill-context-parallel-size 2：stage_span = 8，一个EngineCore带8个worker（进程数 = TP × PCP，不乘PP）
    #   例4：2机×8卡 -dp 4 -tp 4：第2台机器上dp_rank=3（本机第1个DP，dp_rank_local=1）的TP rank 1 →
    #       local_device_rank = 1×4+1 = 5（本机GPU 5），global_rank = 3×4+1 = 13
    #   各通信组（由aiter的init_dist_env / PP时的init_pp_aware_dist_env在world里划分；之后用get_tp_group()等取用）：
    #     TP组：每层线性层的all-reduce / all-gather；DP组：DP rank之间对齐token数（get_dp_padding、DPMetadata）；
    #     PP组：stage之间传hidden states（第9批）；PCP组：prefill上下文并行（第9批）；DCP组：TP组里再按dcp切（第9批）；
    #     EP组：MoE的all-to-all（第7批）
    #   对比nano-vllm：nano-vllm只有TP：dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)，rank就是GPU号；
    #     ATOM多了DP/PP/PCP/DCP的rank换算和多机，集合点（TCPStore）由CoreManager持有，通信组交给aiter建
    def _setup_device_and_distributed(self, rank: int, config: Config):
        # Calculate local device rank considering DP, PP and PCP.
        # On a single node the physical GPU index equals the global distributed
        # rank in the DPxPPxPCPxTP layout: each EngineCore (one per (dp,pp)
        # stage) owns a contiguous tp*pcp GPU slice. `rank` is this worker's
        # local index (0..tp*pcp-1) within its stage.
        # 【译】计算本机设备号，考虑DP、PP和PCP。单机时，在DP×PP×PCP×TP的排布里物理GPU号等于全局分布式rank：
        #   每个EngineCore（每个(dp, pp) stage一个）拥有一段连续的tp*pcp张GPU。`rank`是本worker在其stage内的
        #   本地编号（0..tp*pcp-1）。
        # 解释：docs/distributed_guide.md里写的 dp_rank_local * tp + rank 是不含PP、PCP时的简化形式
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        pp_rank = config.parallel_config.pipeline_parallel_rank
        pp_size = config.pipeline_parallel_size
        if pp_size > 1:
            # Reject before any collective can wait for nonexistent ranks.
            # 【译】在任何集合通信去等待不存在的rank之前就拒绝。
            reject_simulated_tp(config, "pipeline parallel")
        # tp_world_size: how many GPUs this stage actually occupies.
        # 【译】tp_world_size：本stage实际占用的GPU数。
        # stage_span：一个stage（一个EngineCore）占几张卡 = TP × PCP
        stage_span = config.tp_world_size * config.prefill_context_parallel_size
        engine_index = dp_rank_local * pp_size + pp_rank
        local_device_rank = engine_index * stage_span + rank
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        self.device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, "
            f"pp_rank={pp_rank}, local_device_rank={local_device_rank}, "
            f"device={self.device}"
        )

        # 之后不带device的CUDA调用（新建CUDA张量、stream、event）都落在这张卡上
        torch.cuda.set_device(self.device)
        # 引擎内部的MASTER_ADDR/MASTER_PORT（config.master_addr、config.port，默认127.0.0.1:8006，不是HTTP端口）；
        #   真正的集合点是下面的data_parallel_master_ip:data_parallel_base_port（CoreManager起的TCPStore，见config.py port字段上方）
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        # "tcp://ip:port"形式的集合点地址，交给aiter建组用
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
        )
        # NCCL world的大小与本进程在其中的全局rank（公式与例子见本方法上方）；dp_rank是全局DP编号（多机时不从0重新数）
        dp_size = config.parallel_config.data_parallel_size
        world_size = dp_size * pp_size * stage_span
        dp_rank = config.parallel_config.data_parallel_rank
        global_rank = (dp_rank * pp_size + pp_rank) * stage_span + rank
        # 集合点由CoreManager持有（_managed_distributed_store=True，原生服务的常规情况）：本进程作为TCPStore客户端连上去，
        #   自己先建好默认进程组（NCCL world）；之后aiter复用这个默认组再划分各子组。
        #   为什么集合点放在CoreManager：world跨越所有EngineCore（DP、PP），store必须由它们之上的进程起、并活得比它们久
        #   （engine_core_mgr.py _start_distributed_store的【科普】①~③）
        #   init_process_group会阻塞到world里所有rank都连上为止
        if (
            config.parallel_config._managed_distributed_store
            and not torch.distributed.is_initialized()
        ):
            # Preserve AITER's environment setup when preinitializing its group.
            # 【译】在预先初始化AITER的组时，保留AITER原本会做的环境设置。
            os.environ.setdefault(
                "HIP_VISIBLE_DEVICES", ",".join(map(str, range(world_size)))
            )
            store = torch.distributed.TCPStore(
                config.parallel_config.data_parallel_master_ip,
                config.parallel_config.data_parallel_base_port,
                is_master=False,
            )
            torch.distributed.init_process_group(
                backend="nccl", store=store, rank=global_rank, world_size=world_size
            )
        # AITER reuses the default group and creates the model-parallel groups.
        # 【译】AITER复用默认组，并创建各个模型并行组。
        if config.pipeline_parallel_size > 1:
            # 【分支】PP：带PP组的建组函数（distributed/pp_comm.py，第9批）
            from atom.distributed.pp_comm import init_pp_aware_dist_env

            # No local_rank here, unlike the non-PP branch below. Safe only
            # because PP is single-node today: CoreManager rejects multi-node
            # DP when pp_size > 1, and asserts PP+DP out entirely, so
            # global_rank is already the physical device index. Revisit if
            # either restriction is lifted.
            # 【译】这里不传local_rank，这一点与下面的非PP分支不同。之所以安全，只是因为PP目前只支持单机：pp_size > 1时
            #   CoreManager拒绝多机DP，并且直接断言不允许PP+DP，所以global_rank就已经是物理设备号。
            #   如果放开其中任一限制，要重新审视这里。
            init_pp_aware_dist_env(
                tensor_model_parallel_size=config.tensor_parallel_size,
                pipeline_model_parallel_size=pp_size,
                global_rank=global_rank,
                world_size=world_size,
                distributed_init_method=distributed_init_method,
                backend="nccl",
                data_parallel_size=dp_size,
                prefill_context_model_parallel_size=config.prefill_context_parallel_size,
            )
        else:
            # The group spans the devices that exist; apply_simulated_tp then
            # makes it *report* the logical width so layers shard that many ways.
            # 【译】组只覆盖真实存在的设备；然后apply_simulated_tp让它*报告*逻辑宽度，使各层按那么多份切分。
            # aiter的init_dist_env：按TP大小、DP大小/编号、PCP、DCP在world里划出TP/DP/PCP/DCP/EP等组，
            #   并创建aiter自己的custom all-reduce等通信器（第一个参数是TP的真实进程数）
            init_dist_env(
                config.tp_world_size,
                rankID=rank,
                backend="nccl",
                distributed_init_method=distributed_init_method,
                # This node's physical device index. Without it aiter derives a
                # local rank from the DP-scaled global rank, which overruns the
                # device list on every node after the first.
                # 【译】本机的物理设备号。不传的话，aiter会从按DP放大后的全局rank推出本地rank，
                #   在第一台之后的每台机器上都会超出设备列表。
                local_rank=local_device_rank,
                data_parallel_size=config.parallel_config.data_parallel_size,
                data_parallel_rank=config.parallel_config.data_parallel_rank,
                prefill_context_model_parallel_size=config.prefill_context_parallel_size,
                decode_context_parallel_size=getattr(
                    config, "decode_context_parallel_size", 1
                ),
            )
            # 只在--fake-eplb的模拟TP下生效：让TP组报告逻辑宽度（见distributed/simulated_tp.py）；否则直接返回
            apply_simulated_tp(config)

    # 建一个pinned CPU + GPU成对的CpuGpuBuffer（utils/__init__.py）
    # 观察：v0.1.8全仓库没有任何地方调用_make_buffer，各处都直接构造CpuGpuBuffer
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    # 向量化的"分段arange"：给每段长度n_i，同时得到前缀和与每段内的0..n_i−1
    #   输入：num_tokens [B]（例如每条序列的草稿token数）；输出：(cu_num_tokens [B], arange [Σn_i])
    #   调用方：spec_decode/drafter.py（按序列展开草稿token的下标，第7批）
    #   用预分配的self.arange_np切片代替np.arange，再减去每个元素所在段的起点（np.repeat把段起点按段长展开）
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # 【译】求给定数组的前缀和，以及按段的arange。
        #   例如 [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        #   等价于 np.concatenate([np.arange(n) for n in num_tokens])，但更快。
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        # 【译】第1步：[2, 5, 3] -> [2, 7, 10]（前缀和，每段的终点）
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        # 【译】第2步：[2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]（每段起点 = 终点 − 段长，再按段长重复）
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # 【译】第3步：[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]（全局下标 − 所在段起点 = 段内下标）
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    # 【分支】多模态：EngineCore在请求结束/被拒绝时调用（call_func，不等回包），释放这些请求的视觉embedding租约
    def release_multimodal_requests(self, request_ids):
        self.vision_embeddings.release(request_ids)

    # 退出清理：由init_exit_handler装的weakref.finalize / SIGTERM处理调用，也可经RPC调用；still_running保证只执行一次
    #   顺序有讲究（见下面0~3）：先停KV connector的拷贝线程（它还在读KV池）→ 再销毁通信 → 释放CUDA Graph → 最后释放KV和模型
    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self.vision_embeddings.clear()
        # 0. Join any offload connector's copy threads. Its ThreadPoolExecutors
        #    are non-daemon, so leaving them running wedges interpreter shutdown
        #    or races an in-flight copy against atexit. Must run BEFORE the KV
        #    pool it copies out of is dropped and before the dist env goes away.
        #    Guarded: only offload workers define close() (moriio etc. do not).
        # 【译】0. 等待（join）offload connector的拷贝线程结束。它的ThreadPoolExecutor不是守护线程，让它们继续运行
        #   会卡住解释器退出，或让一次进行中的拷贝与atexit竞争。必须在它拷贝的源KV池被释放之前、在分布式环境消失之前执行。
        #   有判断保护：只有offload的worker定义了close()（moriio等没有）。
        connector = get_kvconnector()
        close = getattr(connector, "close", None) if connector is not None else None
        if callable(close):
            close()
        builder = getattr(self, "attn_metadata_builder", None)
        if builder is not None:
            builder.close()
        # 1. Destroy distributed env (NCCL + CustomAllreduce + process groups)
        #    Must happen while ops module is still alive for CustomAllreduce cleanup.
        # 【译】1. 销毁分布式环境（NCCL + CustomAllreduce + 各进程组）。必须趁ops模块还在时执行，CustomAllreduce的清理要用到它。
        destroy_dist_env()
        # 2. Release CUDA graphs
        # 【译】2. 释放CUDA graph（以及它们共用的私有显存池）
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        if isinstance(self.model, UBatchWrapper):
            self.model.tbo_graphs.clear()
        # 3. Release GPU tensors. `kv_cache` is the whole paged pool -- the
        # scales and any indexer cache are regions of it, not attributes.
        # 【译】3. 释放GPU张量。`kv_cache`就是整个分页池——scale和索引缓存都是它里面的区域，不是单独的属性。
        # 解释：mamba_k_cache / mamba_v_cache是GDN/KDA线性注意力的每请求状态池，kpool_tail_cache是GLM-5.3的k-pool尾部
        #   （都是builder的allocate_per_req_cache返回、在allocate_kv_cache里setattr上来的）
        for attr in (
            "kv_cache",
            "mamba_k_cache",
            "mamba_v_cache",
            "kpool_tail_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "drafter"):
            del self.drafter
        torch.cuda.empty_cache()
        return True

    # 【分支】torch profiler（B级）：开始抓本rank的trace
    #   调用方：EngineUtilityHandler._handle_start_profile 经 call_func("start_profiler", wait_out=True)（HTTP的start_profile命令，
    #     或examples/profile_offline.py）；只有--torch-profiler-dir非空时才真的开（profiler_dir在__init__里按rank名建好）
    #   抓CPU与GPU两类活动；stop时on_trace_ready把trace导出成
    #     <torch_profiler_dir>/<rank名>/<模型名>_ts_<时间戳>.pt.trace.json.gz（每个rank一个文件）
    #   trace_name：可选的文件名前缀（v0.1.8里的调用方都不传，用模型名）
    #   每一步forward、准备输入等阶段的名字由文件开头的_profile_runner_stage装饰器在profiler开着时打上
    def start_profiler(self, trace_name: str | None = None):
        """
        Start profiling for this rank.

        Set ATOM_PROFILER_RECORD_SHAPES, ATOM_PROFILER_WITH_STACK, or
        ATOM_PROFILER_PROFILE_MEMORY to "1"/"0" to enable/disable the matching
        profiler option. Any left unset falls back to ATOM_PROFILER_MORE, which
        enables all three when "1" (default: all disabled).
        """
        # 【译】为本rank开始profiling。
        #
        #   把ATOM_PROFILER_RECORD_SHAPES、ATOM_PROFILER_WITH_STACK或ATOM_PROFILER_PROFILE_MEMORY设成"1"/"0"，
        #   可以打开/关闭对应的profiler选项。没设置的选项退回ATOM_PROFILER_MORE：它为"1"时三个都打开（默认：全部关闭）。
        if self.profiler_dir is not None and self.profiler is None:
            record_shapes = envs.ATOM_PROFILER_RECORD_SHAPES
            with_stack = envs.ATOM_PROFILER_WITH_STACK
            profile_memory = envs.ATOM_PROFILER_PROFILE_MEMORY
            model_name = os.path.basename(self.config.model.rstrip("/"))
            safe_model_name = "".join(
                c if c.isalnum() or c in ("_", "-", ".") else "_" for c in model_name
            )
            worker_name = safe_model_name or "trace"
            if isinstance(trace_name, str) and trace_name:
                worker_name = "".join(
                    c if c.isalnum() or c in ("_", "-", ".") else "_"
                    for c in trace_name
                )
            if worker_name == "capture_graph" and safe_model_name:
                worker_name = f"{worker_name}_{safe_model_name}"
            output_prefix = os.path.join(self.profiler_dir, worker_name)

            def _on_trace_ready(prof):
                import gzip as _gzip

                # Use a short human-readable timestamp in file name.
                # 【译】文件名里用一个简短、人能读懂的时间戳。
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                ms = int((time.time() % 1) * 1000)
                output_path = f"{output_prefix}_ts_{ts}_{ms:03d}.pt.trace.json.gz"
                tmp_json_path = output_path[:-3]
                try:
                    t0 = time.monotonic()
                    prof.export_chrome_trace(tmp_json_path)
                    # Chunked gzip: read 64 MB at a time to avoid loading
                    # the entire JSON (~30 GB) into memory at once.
                    # 【译】分块gzip：每次读64MB，避免把整个JSON（约30GB）一次读进内存。
                    with (
                        open(tmp_json_path, "rb") as src,
                        _gzip.open(output_path, "wb") as dst,
                    ):
                        while chunk := src.read(64 * 1024 * 1024):
                            dst.write(chunk)
                    os.remove(tmp_json_path)
                    sz = os.path.getsize(output_path)
                    logger.info(
                        "Rank %d: trace exported to %s (%.1f MB, %.1fs)",
                        self.rank,
                        output_path,
                        sz / 1e6,
                        time.monotonic() - t0,
                    )
                except Exception:
                    logger.exception(
                        "Rank %d: failed to export trace to %s",
                        self.rank,
                        output_path,
                    )
                    for p in (tmp_json_path, output_path):
                        if os.path.exists(p):
                            os.remove(p)

            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=record_shapes,
                with_stack=with_stack,
                profile_memory=profile_memory,
                on_trace_ready=_on_trace_ready,
            )
            self.profiler.__enter__()
            logger.info(
                "Rank %d: profiler started "
                "(record_shapes=%s, with_stack=%s, profile_memory=%s, dir=%s)",
                self.rank,
                record_shapes,
                with_stack,
                profile_memory,
                self.profiler_dir,
            )
        return True

    # 【分支】停止profiler：__exit__时触发上面的on_trace_ready导出trace（可能很慢，大trace要几十秒）
    #   调用方：EngineUtilityHandler._handle_stop_profile（call_func(..., wait_out=True)）
    def stop_profiler(self):
        """Stop profiling for this rank.

        Returns a dict with ``trace_dir`` and ``elapsed`` so the caller
        can report where the trace was written.
        """
        # 【译】为本rank停止profiling。返回一个含 ``trace_dir`` 和 ``elapsed`` 的dict，调用方据此报告trace写到了哪里。
        if self.profiler is None:
            return {"trace_dir": self.profiler_dir, "elapsed": 0.0}
        t0 = time.monotonic()
        logger.info("Rank %d: stopping profiler...", self.rank)
        try:
            self.profiler.__exit__(None, None, None)
        except Exception:
            logger.exception("Rank %d: profiler stop failed", self.rank)
        finally:
            self.profiler = None
        elapsed = round(time.monotonic() - t0, 1)
        logger.info(
            "Rank %d: profiler stop completed in %.1fs",
            self.rank,
            elapsed,
        )
        return {"trace_dir": self.profiler_dir, "elapsed": elapsed}

    def debug(self, *args: Any):
        if self.rank == 0:
            logger.info(*args)

    # 【分支】DP锁步用的假decode：DP>1时只要任一DP rank还有活，本rank这一步没有batch也必须跑一次forward，
    #   陪其他rank完成MoE all-to-all等集合通信，否则它们会永远等在那里（原理与时序图见engine_core.py DPEngineCoreProc上方）
    #   调用方：DPEngineCoreProc._execute_dummy_batch → call_func("dummy_execution", wait_out=True)
    #   假batch：1条id=-1的序列，token全为0，长度 = mtp_k + 1（开MTP时与真实decode的query长度一致），block_table=[0]，
    #     is_dummy_run=True——各注意力builder看到它就把slot_mapping保持为-1，不写任何KV（只会读block 0，结果丢弃）
    #   返回True而不是forward的输出：返回None的RPC不回包，wait_out=True的调用方会卡死（async_proc.py的回包规则）
    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization.

        Two mechanisms lean on the fabricated id being -1: this pass flushes
        the previous real step's tokens (`prepare_sampled_ids` reports the
        batch before), and its own land on the key the deferred flag then
        overwrites. What -1 must NOT do is look like a request --
        `get_token_locations`.
        """
        # 【译】为DP同步执行一个假的decode batch。
        #
        #   有两个机制依赖"伪造的id是-1"：这一趟会把上一个真实step的token冲出来（延迟输出下prepare_sampled_ids报告的是前一个batch），
        #   而它自己的token落在key -1上，随后被延迟输出标志（token_id_dict[-1]）覆盖掉。-1绝不能做的是"看起来像一个请求"——
        #   见get_token_locations（它对prev_batch.is_dummy_run单独处理，不让下一步去匹配一个从未存在的请求）。
        # The scheduler's spec width, not the drafter's. On a P/D producer the
        # real decodes are q=1, and under DP an idle rank's dummy is unified
        # with them: a q=mtp_k+1 dummy would hand the uniform decode
        # collectives a different `running_tokens` than its peers'.
        # 【译】用调度器的投机宽度，而不是草稿器的。在P/D的producer上真实decode都是q=1，而DP下空闲rank的dummy要和它们统一：
        #   q=mtp_k+1的dummy会让统一的decode集合通信拿到一个与同伴不同的`running_tokens`。
        mtp_k = (
            self.drafter.mtp_k
            if hasattr(self, "drafter") and not self.tokenID_processor.is_pd_producer
            else 0
        )
        mtp_factor = mtp_k + 1
        num_tokens_original = mtp_factor

        seq = Sequence(
            [0] * num_tokens_original,
            block_size=self.block_size,
            id=-1,
        )
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = new_block_table([0])

        spec_tokens = {seq.id: np.zeros(mtp_k, dtype=np.int32)} if mtp_k > 0 else None
        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32),
            total_tokens_num=num_tokens_original,
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
            num_spec_step=mtp_k,
            scheduled_spec_decode_tokens=spec_tokens,
        )

        self.forward(dummy_batch)
        logger.debug(
            f"{self.label}: dummy batch executed with {dummy_batch.total_tokens_num} tokens"
        )
        return True

    # 预热forward要带多少个token：单步token预算max_num_batched_tokens（默认16384），PCP时每个PCP rank只拿到1/pcp
    #   调用方：warmup_model；dynamic_chunking_worker.py的启动扫描也用它当上限（第9批）
    # 观察：不开--enable-dp-attention的DP（例如 -dp 2 -tp 4）还要再除以DP组大小，代码和提交说明都没给出理由；
    #   而每个DP rank的调度器单步仍按完整的max_num_batched_tokens排batch（scheduler.py），若按这里测得的峰值定KV预算，
    #   满额prefill时激活可能超出预留
    def warmup_max_tokens(self) -> int:
        """Tokens one warmup forward carries: the activation peak KV sizing allows."""
        # 【译】一次预热forward携带的token数：即KV规模计算所允许的那个激活峰值。
        max_tokens = self.config.max_num_batched_tokens
        if not self.config.enable_dp_attention:
            max_tokens //= get_dp_group().world_size
        pcp_size = self.config.prefill_context_parallel_size
        if pcp_size > 1:
            max_tokens = max(1, max_tokens // pcp_size)
        return max_tokens

    # ⑧ 预热（A级）：用一次"单步能出现的最大prefill"跑一遍完整forward，测出激活的显存峰值
    #   前置：激活 = forward中的临时张量（每层的hidden、q/k/v、MLP中间结果、注意力workspace），一层用完下一层复用，
    #     占用的是"同一时刻活着的最大值"，并随本步token数线性增长（engine_core.py __init__上方"前置"给了70B模型的数字：
    #     16384个token时单是MLP的gate/up输出就≈1.9GB）
    #   为什么要在分KV之前做：KV cache只能分配在激活峰值之外。warmup开头reset_peak_memory_stats()清零峰值统计，
    #     跑完后分配器的allocated_bytes.all.peak = 常驻（权重等）+ 激活峰值；get_num_blocks读它（见那里的预算条）
    #   顺带的作用：第一次forward会触发torch.compile编译（--level 3）、Triton/aiter kernel的JIT与调优，这些一次性开销不会落到第一个请求上
    #   假batch怎么凑（T = warmup_max_tokens()，L = max_model_len）：
    #     num_seqs = min(T // L, max_num_seqs)，至少1条（DSpark块草稿器至少2条，见下面英文注释）；seq_len = min(L, T // num_seqs)
    #     例1：T=16384、L=4096 → 4条 × 4096；例2：T=16384、L=131072 → T//L=0 → 1条 × 16384（并打warning）
    #     序列没有block_table：builder把slot_mapping全设成-1，不写KV（此时KV还没分配，pool_plan是空计划）
    #   对比nano-vllm：凑batch的规则等价（nano-vllm：seq_len = min(T, L)，num_seqs = min(T // seq_len, max_num_seqs)）；
    #     ATOM多了DP/PCP下的token均分、DSpark的最少2条，并且走的是完整的forward（准备输入 → run_model → 采样），
    #     结束后要清掉延迟输出留下的状态
    def warmup_model(self):
        start_time = time.time()
        # 先把分配器缓存的空闲块还给驱动，再把峰值统计重置为"当前已分配量"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_model_len = self.config.max_model_len
        dp_size = get_dp_group().world_size
        warmup_max_tokens = self.warmup_max_tokens()

        num_seqs = min(warmup_max_tokens // max_model_len, self.config.max_num_seqs)

        # torch.compile's mark_dynamic can't make a size-1 batch dim dynamic, so
        # a DSpark block drafter (rows == num_seqs) must first-compile at B >= 2
        # (EAGLE gets that free -- its first draft step is a many-row prefill).
        # Other cases only need the usual >= 1 floor.
        # 【译】torch.compile的mark_dynamic不能把大小为1的batch维标成动态，所以DSpark块草稿器（行数 == num_seqs）
        #   第一次编译时必须B >= 2（EAGLE自然满足——它的第一步草稿就是多行的prefill）。其他情况只需要通常的 >= 1 下限。
        # 解释：Dynamo遇到大小为0或1的维度会把它特化成常量，之后B变成别的值就要重新抓图
        drafter = getattr(self, "drafter", None)
        min_seqs = 2 if getattr(drafter, "is_block_drafter", False) else 1
        num_seqs = max(num_seqs, min_seqs)

        # Split the token budget across the seqs so >1 sequences never exceed it
        # (peak memory unchanged); a lone seq keeps up to max_model_len.
        # 【译】把token预算分给各条序列，使多于1条序列时总数永远不超过预算（峰值显存不变）；只有一条序列时长度最多到max_model_len。
        seq_len = max(1, min(max_model_len, warmup_max_tokens // num_seqs))

        if warmup_max_tokens < max_model_len:
            logger.warning(
                f"{self.label}: dp_size={dp_size}, dp_attn={self.config.enable_dp_attention}, "
                f"warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
                f"Using {num_seqs} seq(s) with length {seq_len} for warmup."
            )

        seqs = [
            Sequence(
                [0] * seq_len,
                block_size=self.block_size,
            )
            for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([seq_len] * num_seqs, dtype=np.int32)
        total_tokens_num = int(num_scheduled_tokens.sum())

        # 一个纯prefill的假ScheduledBatch（字段含义见scheduler.py ScheduledBatch）；is_dummy_run=True让各处跳过只对真实请求有意义的步骤
        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num,
            total_tokens_num_prefill=total_tokens_num,
            total_seqs_num=num_seqs,
            total_seqs_num_prefill=num_seqs,
            is_dummy_run=True,
        )
        # 走与真实请求完全相同的forward（各阶段见forward上方），激活峰值就在这里被分配器记下
        self.forward(dummy_batch)
        # 延迟输出下forward把这一步记成了prev_batch、还挂着一份待取的异步D2H结果；清掉，免得第一个真实step把预热当成"上一步"
        self.tokenID_processor.clean()
        # 释放预热留在分配器缓存里的空闲块，让get_num_blocks看到的free / memory_reserved反映真实占用
        torch.cuda.empty_cache()
        logger.info(
            f"{self.label}: warmup_model {time.time() - start_time:.2f} seconds with {num_seqs} reqs {total_tokens_num} tokens"
        )

    # 【分支】动态分块（第9批）：EngineCore在allocate_kv_cache之后、录图之前调用一次，用启动时的假forward拟合chunk开销
    def profile_dynamic_chunking(self):
        """Fit chunk overhead from startup dummy forwards (engine RPC)."""
        # 【译】用启动时的假forward拟合chunk的开销（引擎RPC）。
        return self._dynamic_chunking.profile()

    # 【分支】动态分块：运行时取回新校准的延迟系数
    def take_dynamic_chunking_fit(self):
        """Return newly calibrated chunk latency coefficients (engine RPC)."""
        # 【译】返回新校准好的chunk延迟系数（引擎RPC）。
        return self._dynamic_chunking.take_fit()

    # ⑥ 每步forward输入/输出的固定buffer（__init__步骤⑥，B级）
    #   前置①（GPU异步执行与stream）：CPU发起的拷贝和kernel只是排进stream就返回，GPU按顺序执行（engine_core.py文件开头【3】的前置）
    #   前置②（锁页内存与异步H2D）：non_blocking=True的H2D拷贝要求源在锁页（pinned）内存，由DMA引擎直接读；
    #     拷贝真正执行时才去读源内存，所以排队期间CPU不能改源（utils/__init__.py _PAGEABLE_H2D_LIMIT上方的"科普·锁页内存与stream"）
    #   前置③（CUDA Graph要求固定地址）：录下的kernel参数里写死了张量地址，回放前只能把新数据拷进同一块显存
    #     （config.py CUDAGraphMode上方【科普】②(a)）。所以这些buffer按上限一次分配好，每步只用前n行
    #   CpuGpuBuffer = 一块pinned CPU张量 + 一块同形状的GPU张量 + 与CPU张量共享内存的numpy视图 .np：
    #     准备输入时用numpy写 .np[:n]，再异步拷到 .gpu[:n]（utils/__init__.py；按组发布见下面_init_h2d_publication）
    #
    #   self.forward_vars（本方法建的部分；注意力builder构造时会再追加slot_mapping、context_lens、block_tables、
    #     cu_seqlens_q/k、num_cached_tokens、batch_id_per_q_token等，草稿器追加它自己的metadata_buffers）：
    #   key               类型 [形状]（例：max_num_batched_tokens=16384、max_num_seqs=512）  H2D发布组    用途
    #   input_ids         CpuGpuBuffer int32 [16385]（tokenIDProcessor的那个）               input_ids   本步所有token的id，拼成一维
    #   decode_src        CpuGpuBuffer int32 [16384]（同上）                                 input_ids   延迟输出下每个请求的anchor从哪取
    #   positions         CpuGpuBuffer int64 [16384]                                        positions   每个token在自己序列里的位置（RoPE）
    #   temperatures      CpuGpuBuffer f32 [512]                                            sampling    每条序列的采样温度
    #   top_ks / top_ps   CpuGpuBuffer i32 / f32 [512]                                      sampling    每条序列的top-k / top-p
    #   outputs           GPU张量 [16384, (extra_output_dims), hidden]，模型dtype            —           模型输出hidden的固定落点（CUDA Graph的输出buffer）
    #   mrope_positions   CpuGpuBuffer int64 [3, 16384]（仅M-RoPE）                         mrope       【分支】多模态
    #   mtp_k、草稿metadata、num_accepted_tokens、draft_next_tokens（仅投机解码）                         【分支】第7批
    #   例：hidden=8192、bf16 → outputs = 16384 × 8192 × 2B = 256MB；它在warmup之前分配，算进get_num_blocks预算条里"常驻"的那一段
    #   对比nano-vllm：nano-vllm每步用torch.tensor(..., pin_memory=True).cuda(non_blocking=True)新建输入张量，只在capture_cudagraph里
    #     另建一套固定的graph_vars，回放前再把输入拷进去；ATOM让每步输入从一开始就写进同一套固定buffer，eager与CUDA Graph回放共用，
    #     省掉每步的分配，也省掉回放前那次拷贝
    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.torch_dtype
        # 单步最多的序列数 / token数：所有per-seq buffer按max_bs、per-token buffer按max_num_batched_tokens分配
        self.max_bs = self.config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}

        # TODO: remove it in forward_context
        # 【译】TODO：把它挪进forward_context里
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "decode_src": self.tokenID_processor.decode_src,
            "positions": CpuGpuBuffer(
                self.max_num_batched_tokens, publication_group="positions", **i64_kwargs
            ),
            "temperatures": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **f32_kwargs
            ),
            "top_ks": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **i32_kwargs
            ),
            "top_ps": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **f32_kwargs
            ),
            # Keep enough space for MTP decode (max_q_len > 1).
            # `extra_output_dims` lets a model insert dims between N and dim
            # (e.g. DeepSeek-V4 returns the un-reduced mHC residual
            # [N, hc_mult, dim] from forward, with hc_head + LM head deferred
            # to compute_logits). Default `()` keeps the standard 2D layout.
            # 【译】留够MTP decode（max_q_len > 1）用的空间。`extra_output_dims`允许模型在N和dim之间插入维度
            #   （例如DeepSeek-V4的forward返回未归约的mHC残差 [N, hc_mult, dim]，hc_head和LM head推迟到compute_logits里做）。
            #   默认 `()` 保持标准的二维布局。
            # 解释：按token数而不是序列数分配——MTP decode每条序列有1+k个token；mHC是V4的多路残差（第8批）
            "outputs": torch.empty(
                self.max_num_batched_tokens,
                *getattr(self.model, "extra_output_dims", ()),
                hidden_size,
                dtype=hidden_type,
            ),
        }
        if self.use_mrope:
            self.forward_vars["mrope_positions"] = CpuGpuBuffer(
                3,
                self.max_num_batched_tokens,
                publication_group="mrope",
                publication_unit="elements",
                **i64_kwargs,
            )
        if hasattr(self, "drafter"):
            self.forward_vars["mtp_k"] = self.drafter.mtp_k
            self.forward_vars.update(self.drafter.metadata_buffers)
            self.forward_vars["num_accepted_tokens"] = CpuGpuBuffer(
                self.max_bs, **i32_kwargs
            )
            # Per in-flight slot via forward_vars; PP ring clones it.
            # 【译】经forward_vars做到每个在途槽位一份；PP的环会克隆它。
            self.forward_vars["draft_next_tokens"] = CpuGpuBuffer(
                self.max_bs, publication_group="draft_anchors", **i32_kwargs
            )

    # forward_vars环（B级；PP>1时才有多格，【分支】PP见第9批）
    #   问题：PP的head stage会不等GPU、连续发出最多pp_size个微batch的forward；若共用一套forward_vars，
    #     第N+1个微batch的prepare_inputs会改写第N个微batch的kernel还在读的buffer
    #   做法：每个在途微batch一套buffer（只克隆CpuGpuBuffer），按轮转使用，复用一格之前先等这一格上次的forward做完：
    #     pp_size=2：槽位0：mb0 ──► mb2 ──► mb4 …    forward开头：_advance_forward_vars轮转到下一格 → _gate_staging_reuse等这一格的event
    #               槽位1：mb1 ──► mb3 ──► …        forward末尾：_record_forward_vars_event在stream上record这一格的event
    #   PP=1：环只有1格（就是原字典），advance是空操作；改用_stage_h2d_done这个event把CPU的领先限制在"一步的H2D"以内（见_gate_staging_reuse）
    #   PP>1要求--enforce-eager：CUDA Graph录下的是固定地址，换一套buffer回放就会读错地方
    def _init_forward_vars_ring(self):
        """Build a ring of independent ``forward_vars`` copies, one per possible
        in-flight pipeline microbatch.

        The head launches up to ``pp_size`` forwards back-to-back without a GPU
        sync, so a single reused ``forward_vars`` set would let microbatch N+1's
        ``prepare_inputs`` overwrite staging buffers microbatch N's kernels are
        still reading. Each in-flight slot gets its own buffer set; reuse of a
        slot is gated by a per-slot CUDA event (see ``_advance_forward_vars`` /
        ``_record_forward_vars_event``), bounding the CPU's GPU lead to the ring
        size even when the head runs middle-chunk or DP-sync dummy batches
        without a GPU sync.

        When ``pp_size == 1`` the ring is the single original dict and advance is
        a no-op, so behavior is unchanged.
        """
        # 【译】建一个由相互独立的 ``forward_vars`` 副本组成的环，每个可能同时在途的流水线微batch一份。
        #
        #   head会不经GPU同步、背靠背地发出最多 ``pp_size`` 个forward，所以只复用一套 ``forward_vars`` 的话，
        #   微batch N+1的 ``prepare_inputs`` 会覆盖微batch N的kernel仍在读取的暂存buffer。每个在途槽位有自己的一套buffer；
        #   一个槽位的复用由该槽位的CUDA event把关（见 ``_advance_forward_vars`` / ``_record_forward_vars_event``），
        #   即使head在跑middle chunk或DP同步用的dummy batch、不做GPU同步，CPU领先GPU的程度也被限制在环的大小以内。
        #
        #   ``pp_size == 1`` 时环就是那一个原始字典，advance是空操作，所以行为不变。
        pp_size = self.config.pipeline_parallel_size
        self._fv_idx = 0
        self._stage_h2d_done = None
        if pp_size <= 1:
            self._fv_ring = [self.forward_vars]
            self._fv_slot_events = None
            # Nothing to rotate to, so bound the lead in time instead. See
            # `_gate_staging_reuse`.
            # 【译】没有别的槽位可以轮转，所以改为在时间上限制领先。见 `_gate_staging_reuse`。
            self._stage_h2d_done = torch.cuda.Event()
            logger.info("forward_vars ring: 1 slot (staging reuse gated on its H2D)")
            return

        assert self.enforce_eager, (
            "pipeline_parallel_size > 1 requires eager execution "
            "(--enforce-eager): the forward_vars ring swaps metadata "
            "buffers per microbatch, which is incompatible with CUDAGraph replay."
        )

        def _clone_slot(src: dict) -> dict:
            # CpuGpuBuffers are the per-forward staging buffers, and only their
            # host half can be rewritten while an earlier microbatch's kernels
            # are still reading. Everything else is immutable, unused on the
            # eager PP path, or device-only, where the stream orders the writing
            # kernel after those readers.
            # 【译】CpuGpuBuffer是每次forward的暂存buffer，只有它们的主机那一半会在更早微batch的kernel仍在读取时被改写。
            #   其余的要么不可变，要么在eager的PP路径上用不到，要么只在设备上——那种情况由stream保证写它的kernel排在那些读者之后。
            return {
                k: (v.clone() if isinstance(v, CpuGpuBuffer) else v)
                for k, v in src.items()
            }

        self._fv_ring = [self.forward_vars] + [
            _clone_slot(self.forward_vars) for _ in range(pp_size - 1)
        ]
        # One event per slot, marking completion of the last forward that used
        # it. Fresh (never-recorded) events synchronize immediately, so the
        # first pass over the ring is unthrottled.
        # 【译】每个槽位一个event，标记上一次使用它的forward已经完成。新建（从未record过）的event会立即同步完成，
        #   所以第一圈使用环时不受节流。
        self._fv_slot_events = [torch.cuda.Event() for _ in range(pp_size)]
        logger.info(f"forward_vars ring: {pp_size} slots (pipeline parallel)")

    # H2D发布的接线（B级）：把每步要从CPU传到GPU的元数据buffer，按"第一个用到它的GPU kernel"分组发布
    #   机制（发布协议、检查、packed打包）的规范讲解在utils/h2d.py、utils/packed_h2d.py和docs/h2d_publication.md（第5批）；
    #   这里只看runner怎么接线：
    #   ① 环的每个槽位一个PublicationOwner，带着该槽位的复用event（PP=1时是_stage_h2d_done）；
    #      所有owner共用一个registry，登记时检查各buffer的源/目的地址区间互不重叠
    #   ② forward_vars里每个声明了publication_group的CpuGpuBuffer → owner.bind，按组名收集成组
    #   ③ ATOM_H2D_BACKEND=packed（默认）时再拼出两个大组：
    #        token_inputs   = sampling + early（cu_seqlens_q）+ input_ids（+ spec_decode）  ← 第一个GPU使用者是token组装
    #        prefill_inputs = prefill（slot_mapping/context_lens/block_tables…）+ positions  ← 第一个使用者是注意力准备
    #      builder还可以经h2d_group_members声明自己的组（例：MHA的mha_decode = slot_mapping、context_lens、block_tables、
    #      kv_indptr、positions，在生成KV下标的kernel之前一起就绪）
    #      packed：组内各buffer的有效部分先拼进一块连续的pinned arena，一次DMA传完，再由一个kernel把各段scatter回各自原来的
    #        固定GPU地址（地址不变，CUDA Graph照用）；direct：每个buffer各拷一次。几十个小张量各拷一次，每次都有固定开销
    #        （envs.py ATOM_H2D_BACKEND上方的"科普·forward元数据H2D"）
    #   ④ 每个owner最后record一次event，盖住构造期的上传、环的克隆和传输用的指针表
    #   每步的协议（见forward）：_advance_forward_vars → _gate_staging_reuse（owner.begin：等上一次的H2D执行完，CPU才能写）
    #     → prepare_* 写buffer、按组publish → _mark_staging_h2d_enqueued（owner.finish：在stream上record event）→ run_model
    def _init_h2d_publication(self):
        from atom.utils.h2d import PublicationOwner, PublicationRegistry

        registry = PublicationRegistry()
        self.publication_registry = registry
        transport = envs.ATOM_H2D_BACKEND
        if transport not in ("direct", "packed"):
            raise ValueError("ATOM_H2D_BACKEND must be direct or packed")
        self._h2d_owners = []
        self._h2d_groups = []
        for index, variables in enumerate(self._fv_ring):
            event = (
                self._stage_h2d_done
                if self._fv_slot_events is None
                else self._fv_slot_events[index]
            )
            owner = PublicationOwner(self.device, event, registry=registry)
            members = {}
            for name, buffer in variables.items():
                if isinstance(buffer, CpuGpuBuffer) and buffer.publication_group:
                    binding = owner.bind(buffer, name, unit=buffer.publication_unit)
                    members.setdefault(buffer.publication_group, []).append(binding)
            groups = {name: owner.group(name, items) for name, items in members.items()}
            if transport == "packed":
                # One upload immediately before token assembly's first GPU
                # consumer. Sampling has no earlier device consumer.
                # 【译】在token组装的第一个GPU使用者之前只做一次上传。采样参数没有更早的设备端使用者（所以可以并进来）。
                if all(name in groups for name in ("sampling", "early", "input_ids")):
                    token_members = ("sampling", "early", "input_ids")
                    if "spec_decode" in groups:
                        token_members += ("spec_decode",)
                    groups["token_inputs"] = owner.group(
                        "token_inputs",
                        [b for name in token_members for b in groups[name].members],
                    )
                if "prefill" in groups and "positions" in groups:
                    groups["prefill_inputs"] = owner.group(
                        "prefill_inputs",
                        groups["prefill"].members + groups["positions"].members,
                    )
            # Some buffers also publish together at a later consumer boundary
            # (e.g. MHA decode shares block tables with the prefill group).
            # 【译】有些buffer还会在更靠后的某个使用者边界一起发布（例如MHA decode与prefill组共用block table）。
            for name, names in getattr(
                getattr(self, "attn_metadata_builder", None), "h2d_group_members", {}
            ).items():
                groups[name] = owner.group(
                    name, [variables[item]._publication for item in names]
                )
            if transport == "packed":
                for group in owner.use_packed_transport():
                    logger.info(
                        "H2D group %s: %s%s",
                        group.name,
                        group.transport,
                        f" ({group.fallback_reason})" if group.fallback_reason else "",
                    )
            # Cover constructor uploads, ring clones and transport pointer
            # tables before the first preparation, even on another stream.
            # 【译】在第一次准备输入之前，把构造期的上传、环的克隆和传输用的指针表都盖住，即使第一次准备发生在另一条stream上。
            event.record()
            self._h2d_owners.append(owner)
            self._h2d_groups.append(groups)
        self.h2d_owner = self._h2d_owners[self._fv_idx]
        self.h2d_groups = self._h2d_groups[self._fv_idx]

    # forward的第一步：轮转到环的下一格（PP=1时空操作）。调用方：forward
    def _advance_forward_vars(self):
        """Rotate to the next in-flight slot before any buffer is written.

        Dummy forwards use the same staging buffers as real forwards, so they
        participate in the ring too. No-op when the ring has a single slot.
        """
        # 【译】在写任何buffer之前，轮转到下一个在途槽位。dummy forward与真实forward用同一套暂存buffer，
        #   所以它们也参与轮转。环只有一个槽位时是空操作。
        if len(self._fv_ring) == 1:
            return
        self._fv_idx = (self._fv_idx + 1) % len(self._fv_ring)
        # Select the slot now; _gate_staging_reuse waits on its existing event
        # before any producer writes. Avoid waiting twice on the same event.
        # 【译】现在就选定槽位；_gate_staging_reuse会在任何生产者写入之前等它已有的event。避免对同一个event等两次。
        self.forward_vars = self._fv_ring[self._fv_idx]
        self.h2d_owner = self._h2d_owners[self._fv_idx]
        self.h2d_groups = self._h2d_groups[self._fv_idx]
        # Token assembly holds aliases outside forward_vars; select both
        # payload and source-index mirrors from this slot before writing.
        # 【译】token组装在forward_vars之外还持有别名；写入之前，把数据本身和源下标这两个镜像都切到本槽位的。
        self.tokenID_processor.input_ids = self.forward_vars["input_ids"]
        self.tokenID_processor.decode_src = self.forward_vars["decode_src"]

    # forward的第二步：在CPU改写pinned暂存buffer之前，等上一次forward的H2D拷贝真正执行完（owner.begin()）
    #   为什么必须等（PP=1，一个buffer只有一份）：
    #     不加门：CPU [写buf←第N步][发H2D_N][发kernel_N][写buf←第N+1步] …    ← 中间没有任何同步点
    #             DMA                       ……………………[执行H2D_N：读buf]       ← 读到的已经是第N+1步的数据
    #     加门：  CPU [begin:等event][写buf][发H2D_N][finish:record event][发kernel_N] [begin:等H2D_N执行完][写buf] …
    #   decode每步本来就要同步取采样结果，稳态下这里不会真的等；只会在一串不采样的middle chunk（chunked prefill的中间块）时限流
    #   调用方：forward（也包括dummy forward，见docstring最后一段）
    @_profile_runner_stage
    def _gate_staging_reuse(self):
        """Block until the previous forward's staging H2Ds have executed.

        `_stage` / `CpuGpuBuffer.copy_to_gpu` copy `non_blocking=True` out of
        ONE pinned buffer per name, so the next forward's `buf.np[:n] = arr`
        races the previous forward's DMA. Sampling forwards close that window
        by accident, since `postprocess` synchronizes to read the sampled ids,
        but a chunked prefill's middle chunk returns before `postprocess` and
        closes nothing. Measured: the host reached 4042 packets ahead and the
        GPU read a `batch_id_per_q_token` from a later batch (id 3 in a `bs=2`
        batch), tripping the bounds assert in
        `cu_committed_gpu[batch_id_per_q_token]` -- which wedges the queue with
        no fault line and no traceback.

        One buffer admits one forward of lead, so the gate is depth-1. Decode
        already syncs every step, so it never blocks in steady state; it
        throttles only runs of middle chunks, where the unbounded lead was
        buying nothing. A never-recorded event passes, so the first forward is
        not held.

        DP-sync dummy forwards use these same buffers and can also return while
        their copies are in flight. They must therefore enter this gate and
        record the event just like real forwards.

        The publication owner uses the existing single-slot or PP slot event,
        then opens one ledger epoch before any producer writes host metadata.
        """
        # 【译】阻塞，直到上一次forward的暂存H2D拷贝已经执行完。
        #
        #   `_stage` / `CpuGpuBuffer.copy_to_gpu` 以 `non_blocking=True` 从每个名字对应的那*一个*pinned buffer拷贝，
        #   所以下一次forward的 `buf.np[:n] = arr` 会和上一次forward的DMA竞争。要采样的forward碰巧关上了这个窗口，
        #   因为 `postprocess` 为了读采样出的id会做同步；但chunked prefill的middle chunk在 `postprocess` 之前就返回了，
        #   什么也没关上。实测：主机领先了4042个包，GPU从一个更晚的batch里读到了 `batch_id_per_q_token`
        #   （在一个 `bs=2` 的batch里出现id 3），触发了 `cu_committed_gpu[batch_id_per_q_token]` 的越界断言——
        #   这会让队列卡死，既没有fault信息也没有traceback。
        #
        #   一个buffer只容许领先一次forward，所以这道门的深度是1。decode本来每步都同步，稳态下从不阻塞；
        #   它只给一连串middle chunk限流，那时无限制的领先本来就买不到任何东西。从未record过的event直接放行，所以第一次forward不会被卡住。
        #
        #   DP同步用的dummy forward也用这些buffer，也可能在拷贝仍在途时返回。所以它们必须和真实forward一样进入这道门、record event。
        #
        #   发布owner使用现有的单槽位event或PP槽位event，然后在任何生产者写主机元数据之前开启一个账本epoch。
        self.h2d_owner.begin()

    # forward里prepare_model之后、run_model之前调用：owner.finish()在当前stream上record event，
    #   标记"本步所有H2D都已排进stream"，下一次_gate_staging_reuse就等这个event
    @_profile_runner_stage
    def _mark_staging_h2d_enqueued(self):
        """Close the window the gate above waits on.

        Covers this preparation phase's source reads, including registered
        direct copies and grouped publishers. Late V4 TBO preparation resumes
        this epoch and records the completion again after its uploads.
        Independent subsystem uploads retain their own completion protocols.
        """
        # 【译】关上上面那道门所等待的窗口。
        #
        #   覆盖本次准备阶段对源buffer的读取，包括登记过的直接拷贝和分组发布者。V4 TBO较晚的准备会恢复这个epoch，
        #   在它的上传之后再record一次完成。独立子系统的上传保留它们自己的完成协议。
        self.h2d_owner.finish()

    # forward末尾调用：PP环里给当前槽位record event（PP=1时空操作）；record失败就把owner标成failed、拒绝再复用
    def _record_forward_vars_event(self):
        """Mark the current slot's forward as done on the GPU stream. Paired
        with the owner wait in ``_gate_staging_reuse``. Called at the end of
        every forward, including DP-sync dummies. No-op when the ring has a
        single slot."""
        # 【译】在GPU stream上标记当前槽位的forward已完成。与 ``_gate_staging_reuse`` 里owner的等待配对。
        #   每次forward结束时都调用，包括DP同步的dummy。环只有一个槽位时是空操作。
        if len(self._fv_ring) == 1:
            return
        try:
            self._fv_slot_events[self._fv_idx].record()
        except BaseException:
            self.h2d_owner.fail()
            raise

    # 本rank分到几个KV头（GQA的K/V头在TP下按头切分）：
    #   KV头数 ≥ TP：均分，例：8个KV头、-tp 4 → 每rank 2个
    #   KV头数 < TP：每rank 1个，同一个头复制到 TP / KV头数 个rank上，例：Qwen3-Next的2个KV头、-tp 4 → 每rank 1个、每个头2份
    #   调用方：qwen4_exp_attn.py；mha_kv_pool.py的MhaKvPool.from_hf_config按同一规则切（allocate_kv_cache里也有一份同样的计算）
    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads."""
        # 【译】返回每个rank的KV头数。
        hf_config = self.config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            return hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            return 1

    # 【分支】M-RoPE：把 [3, max_num_batched_tokens] 的GPU buffer看成紧凑的 [3, num_tokens]
    #   （as_strided把行步长设成num_tokens，三行首尾相接；与CPU侧按publication_unit="elements"发布的布局一致）
    #   调用方：注意力builder（backends.py）、run_model、capture_cudagraph
    def _mrope_positions_view(self, num_tokens: int) -> torch.Tensor:
        return self.forward_vars["mrope_positions"].gpu.as_strided(
            (3, num_tokens), (num_tokens, 1)
        )

    # 【分支】投机解码：草稿模型的注意力层是否直接放进目标模型的KV池（例：MTP层与目标同为MLA，就是池里多几行）
    #   草稿注意力与目标不同时（例：MLA目标 + MHA的EAGLE3草稿），eagle_proposer / dspark_proposer会设置
    #   runner.draft_kv_builder，草稿走自己的builder（仍共用同一套block id，见_sub_pool_specs）
    #   调用方：aiter_mla.py、pool_layout/pool_rows.py（决定数行时要不要把草稿的层也算进来）
    def draft_shares_kv_pool(self) -> bool:
        """Whether a draft's attention layers land in the target's pools.

        A draft whose flavor asked for a pool of its own has a
        `draft_kv_builder` and is priced and bound through that instead. One
        spelling, because the layer count and the walk that fills those layers
        have to answer this the same way.
        """
        # 【译】草稿模型的注意力层是否落在目标模型的池里。
        #
        #   一个要求自己单独有池的草稿会有一个 `draft_kv_builder`，改由它来计价和绑定。只写这一处，
        #   因为层数和"填这些层的遍历"必须对这个问题给出同样的回答。
        return bool(
            self.config.speculative_config
            and hasattr(self, "drafter")
            and not hasattr(self, "draft_kv_builder")
        )

    # 收集所有builder声明的子池规格（SubPoolSpec），交给get_num_blocks里的plan_pools
    #   SubPoolSpec = (哪个池：PAGE按历史长度增长 / STATE按在途请求数增长, 名字, 每条目字节数, 每请求条目数)
    #     （pool_layout/sub_pool_spec.py；例子见get_num_blocks上方②）
    #   同名规格在plan_pools里合并（字节相加、共用一套条目下标）：草稿的KV就这样"骑"在目标的block id上，而不是另成一个池
    def _sub_pool_specs(self) -> list[SubPoolSpec]:
        """Cache-class declarations from every builder attached to this runner.

        The target builder always, plus an optional `draft_kv_builder` when a
        spec-decode draft owns its own KV. Each builder knows its own tensor
        layout (MLA 576-dim packed, GDN-hybrid full-attn-only, MiMo-V2
        per-layer-type, standard MHA split-K/V); the runner only sums bytes.
        Specs sharing a name merge in `plan_pools`, which is how the draft KV
        joins the target's block ids instead of forming a second pool.
        """
        # 【译】挂在本runner上的每个builder声明的cache类别。
        #
        #   目标模型的builder总是有；投机解码的草稿自己持有KV时，再加一个可选的 `draft_kv_builder`。每个builder知道
        #   自己的张量布局（MLA是576维打包的一行、GDN混合模型只有全注意力层的行、MiMo-V2按层类型分、标准MHA是分开的K/V）；
        #   runner只把字节数加起来。同名的规格在 `plan_pools` 里合并——草稿KV就是这样加入目标的block id，而不是另成第二个池。
        specs = list(self.attn_metadata_builder.sub_pool_specs())
        if hasattr(self, "draft_kv_builder"):
            specs += self.draft_kv_builder.sub_pool_specs()
        return specs

    # 声明了哪些每请求STATE类子池（例：GDN的递归状态槽位、V4的压缩器环/滑动窗口）；没有就是空列表
    #   调用方：__init__里的per_req_cache核对、RapidServeModelRunner的PD检查
    def _state_pool_names(self) -> list[str]:
        """The per-request STATE classes the attached builders declare.

        Named rather than counted because both callers want to say which: one
        refuses a model whose `model_type` is missing from the per-req-cache
        set, the other refuses P/D for it.
        """
        # 【译】挂着的builder声明的每请求STATE类别。返回名字而不是个数，因为两个调用方都想说出是哪几个：
        #   一个拒绝 `model_type` 不在per-req-cache集合里的模型，另一个拒绝对它做P/D。
        return [s.name for s in self._sub_pool_specs() if s.pool is Pool.STATE]

    # CUDA Graph私有显存池的预估 = get_num_blocks预算条里"graph"那一段（B级）
    #   前置：录图时forward里的中间张量都从一个专用池分配，地址写死进graph、回放时原地复用，所以录完后这块显存一直被占着；
    #     各尺寸的graph共用一个池（图解见utils/__init__.py _ATOM_WEAKREF_OP上方的"覆盖复用"）。
    #     而录图必须在KV分配之后（graph要记下KV的最终地址），所以只能先估一个数、从KV预算里预先扣掉
    #   三种估法：
    #     enforce_eager                 → 0（不录图）
    #     FULL（默认--cudagraph-mode）   → 激活峰值 × 0.2（例：warmup峰值8GB → 1.6GB）；DSpark置信度调度每个q桶各录一套 → 再 × 桶数
    #     PIECEWISE类                   → 每token字节数 × Σ(要录的token数档位)，累加到约预算的15%为止：
    #       每token字节数 = hidden × dtype字节 × 层数 × 2.8（DP>1时再 × dp^0.6，见_piecewise_per_token_bytes）
    #       例：DeepSeek-V3（hidden 7168、bf16、61层）：7168 × 2 × 61 × 2.8 ≈ 2.34MB/token；默认档位[1..8,16,32,48,64,128,256,512]、
    #         不开投机（q=1）时 Σ = 36 + 1056 = 1092个token → ≈2.5GB；MTP k=3（q=4）时档位变成 bs×4，Σ = 4368 → ≈10GB
    #   低估是安全的：PIECEWISE录图时会按实际空闲显存逐档检查，放不下的档位直接跳过（_piecewise_skip_capture）
    #   对比nano-vllm：nano-vllm不预留这一块——KV按util填满后再录图，graph池从util之外的那部分显存里出
    def _estimate_cudagraph_overhead(self):
        """Estimate GPU memory consumed by CUDA graph capture.

        CUDA graphs allocate a shared memory pool for intermediate activations.
        The pool size is roughly the peak activation memory during a single
        forward pass. We estimate this from the gap between warmup peak and
        current (steady-state) allocation.

        Returns 0 when enforce_eager is set (no CUDA graphs).
        """
        # 【译】估计CUDA graph捕获要消耗的GPU显存。
        #
        #   CUDA graph为中间激活分配一个共享的显存池。池的大小大致是单次forward的激活峰值。我们用warmup峰值与
        #   当前（稳态）分配量之差来估计它。
        #
        #   设置了enforce_eager（不用CUDA graph）时返回0。
        if self.config.enforce_eager:
            return 0
        # peak − current = warmup那次满额prefill的激活峰值（current是权重等常驻部分，见get_num_blocks的预算条）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        activation_bytes = max(peak - current, 0)

        # PIECEWISE pool ~ per_token * Σ(captured num_tokens). per_token from model
        # geometry (hidden*dtype*layers*k), not a magic constant. Under-reserve is
        # safe: capture re-checks live free mem per bucket and skips oversized.
        # 【译】PIECEWISE的池 ≈ 每token字节数 × Σ(捕获的num_tokens)。每token字节数由模型几何（hidden*dtype*层数*k）算出，
        #   不是一个魔数。预留少了是安全的：捕获时会逐个桶重新检查实时空闲显存，跳过放不下的桶。
        if self._piecewise_cg_active():
            cap_sizes = self.config.compilation_config.cudagraph_capture_sizes or [
                self.config.max_num_seqs
            ]
            # Captured num_tokens shapes are bs * q. The capture loop uses
            # full_q_len = mtp_k+1 for ANY spec-decode drafter (plain MTP or
            # DSpark), and q=1 for non-spec. Mirror that here or the estimate
            # under-counts Σtok by a factor of q (plain MTP q=4 -> 4x under ->
            # pool est 8.5GB vs actual 33GB -> OOM). DSpark additionally captures
            # multiple q-buckets, so fold its whole bucket set in.
            # 【译】捕获的num_tokens形状是 bs * q。捕获循环对任何投机解码草稿器（普通MTP或DSpark）都用 full_q_len = mtp_k+1，
            #   不开投机时q=1。这里要照做，否则估出的Σtok少了q倍（普通MTP q=4 → 少4倍 → 池估计8.5GB而实际33GB → OOM）。
            #   DSpark还会捕获多个q桶，所以把它的整个桶集合也并进来。
            if hasattr(self, "drafter"):
                full_q = self.drafter.mtp_k + 1
                q_buckets = self._dspark_capture_q_buckets(full_q)
                if (
                    self.drafter.uses_confidence_schedule
                    and os.environ.get("ATOM_PIECEWISE_FINE_TOKENS", "0") == "1"
                ):
                    q_buckets = sorted(set(q_buckets) | set(range(1, full_q + 1)))
            else:
                q_buckets = [1]
            per_token_bytes = self._piecewise_per_token_bytes()
            dp_size = self.config.parallel_config.data_parallel_size
            # Cap the reserved buckets at a fraction of the KV budget so a huge
            # capture list can't starve KV. Use the utilization budget (not raw
            # total) as the reference — it tracks the configured memory envelope.
            # 【译】把预留的桶限制在KV预算的一个比例之内，免得一个很长的捕获列表把KV饿死。用utilization预算（而不是原始总显存）
            #   作参照——它跟随配置的显存范围。
            budget = self.config.gpu_memory_utilization * torch.cuda.mem_get_info()[1]
            target_reserve = 0.15 * budget
            all_shapes = sorted({bs * q for bs in cap_sizes for q in q_buckets})
            # Mirror the capture-loop token-budget skip: a bucket over
            # `max_num_batched_tokens` is not schedulable and is not captured,
            # so reserving for it would only shrink KV.
            # 【译】与捕获循环里的token预算跳过保持一致：超过 `max_num_batched_tokens` 的桶调度不到、也不会被捕获，
            #   为它预留只会让KV变小。
            all_shapes = [
                s for s in all_shapes if s <= self.config.max_num_batched_tokens
            ]
            # Mirror the capture-loop DP+spec num_tokens cap (see capture_cudagraph)
            # so the reservation only counts buckets we actually capture.
            # 【译】与捕获循环里DP+投机时的num_tokens上限保持一致（见capture_cudagraph），让预留只计算真正会捕获的桶。
            # 解释：ATOM_PIECEWISE_DP_MAX_TOKENS直接读os.environ（默认512），不在envs.py的登记表里
            if dp_size > 1 and hasattr(self, "drafter"):
                _dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
                all_shapes = [s for s in all_shapes if s <= _dp_cap]
            # 从小到大累加，加上下一档就超过target_reserve时停（至少保留一档）；停掉的大档位仍可能被录，由录图时的显存检查决定
            captured = []
            acc = 0
            for num_tokens in all_shapes:
                if captured and per_token_bytes * (acc + num_tokens) > target_reserve:
                    break
                captured.append(num_tokens)
                acc += num_tokens
            overhead = int(per_token_bytes * acc)
            logger.info(
                "PIECEWISE cudagraph mem estimate: n_shapes=%d/%d Σtok=%d "
                "per_token=%.3fMB -> overhead=%.2fGB",
                len(captured),
                len(all_shapes),
                acc,
                per_token_bytes / (1 << 20),
                overhead / (1 << 30),
            )
            return overhead
        # FULL整图（默认）：按激活峰值的20%估。各尺寸的整图共用一个池，最大的那张图决定池的大小
        overhead = activation_bytes * 0.2
        # DSpark RAGGED captures one graph set PER q-bucket, so scale by the
        # number of captured buckets (the pool grows ~linearly with bucket
        # count, each bucket ~one graph set). This stays a safe upper bound:
        # measured per-bucket pool (~1.4GB) << 0.2*act.
        # 【译】DSpark RAGGED为每个q桶各捕获一套graph，所以按捕获的桶数放大（池大约随桶数线性增长，每个桶约一套graph）。
        #   这仍是一个安全的上界：实测每个桶的池（约1.4GB）远小于 0.2*激活。
        if hasattr(self, "drafter") and self.drafter.uses_confidence_schedule:
            # Match the capture loop's bucket source so we count the graphs
            # actually captured; the pool grows ~linearly with bucket count.
            # 【译】与捕获循环的桶来源保持一致，这样数的就是真正捕获的graph；池大约随桶数线性增长。
            buckets = self._dspark_capture_q_buckets(self.drafter.mtp_k + 1)
            n_buckets = len(buckets)
            overhead = activation_bytes * 0.2 * n_buckets
            logger.info(
                "DSpark cudagraph mem estimate: buckets=%s n=%d act=%.2fGB "
                "-> overhead=%.2fGB",
                buckets,
                n_buckets,
                activation_bytes / (1 << 30),
                overhead / (1 << 30),
            )
        return int(overhead)

    # 启动的最后一步（EngineCore._freeze_after_startup经RPC调用）：把启动期建好的、要活一辈子的Python对象gc.freeze，
    #   之后的分代GC不再反复扫描它们（分代GC与gc.freeze见utils/gc_utils.py文件开头的【科普】）
    def freeze_gc_heap(self) -> int:
        """RPC target: freeze this worker's startup heap. Pauses here reached
        979 ms, the largest of any process.

        The count is returned because `busy_loop` replies only `if out is not
        None` -- an RPC target returning None hangs its `wait_out=True` caller.
        """
        # 【译】RPC目标：冻结本worker启动期的堆。这里的GC停顿曾达到979ms，是所有进程里最大的。
        #
        #   返回计数是因为 `busy_loop` 只在 `if out is not None` 时才回复——返回None的RPC目标会让 `wait_out=True` 的调用方挂住。
        return freeze_gc_heap(worker_process_name(self.config, self.rank))

    # =====================================================================================
    # get_num_blocks：KV cache能放多少个block（以及每请求状态池要多少槽位）——ATOM版的"显存预算"（A级）
    #   调用方：EngineCore.__init__ 经 call_func("get_num_blocks", wait_out=True)，只取rank 0的返回值；
    #     此时本rank的__init__（含warmup）已跑完，分配器的峰值统计里记着warmup那次满额prefill的激活峰值
    #   输出：dict（见末尾return），由EngineCore回填进它的Config，再交给allocate_kv_cache和Scheduler / BlockManager
    #   本段是显存预算条的规范位置（engine_core.py __init__上方的预算条是它的概要，数字与这里一致）
    #
    # ① 显存都花在哪：用PyTorch分配器的统计量把整张卡拆开（GB按2^30字节，与日志一致）
    #   例：192GB的卡、--gpu-memory-utilization 0.9、70B稠密模型FP8权重（TP=1，≈70GB）、80层、8个KV头、head_dim=128、
    #     --kv-cache-dtype fp8、block_size=16、默认--cudagraph-mode FULL；warmup（16384个token）的激活峰值≈8GB
    #
    #   0                                                                                                 192GB（total）
    #   ├──────── 常驻（权重等）70 ────────┼激活峰值 8┼非torch≈2┼graph 1.6┼余量3.84┼─────── KV ≈87.36 ───────┼不进预算 19.2┤
    #   │◄─────────── current ───────────►│          │                                                     │
    #   │◄────────── peak（= peak_torch）───────────►│                                                     │
    #   │◄──────────────────────────────── budget = total × util = 172.8 ─────────────────────────────►│
    #
    #   变量                     怎么得到                                                                  例（GB）
    #   current                 分配器当前已分配的字节：权重 + forward_vars等常驻buffer                         70
    #   peak                    warmup开头reset_peak_memory_stats之后分配器的最高点 = 常驻 + 激活峰值             78
    #   peak_torch              max(peak, current)                                                         78
    #   non_torch               (total − free) − memory_reserved()：不经PyTorch分配器的显存——RCCL/aiter custom
    #                           all-reduce的缓冲、HIP运行时，以及同卡上其他进程占的                                ≈2
    #   cudagraph_overhead      _estimate_cudagraph_overhead()：FULL时 (peak − current) × 0.2                  1.6
    #   safety_margin           total × 2%                                                                 3.84
    #   non_kv_overhead         peak_torch + non_torch + cudagraph_overhead + safety_margin               85.44
    #   available_for_kv_budget budget − non_kv_overhead − extra_reserve（RapidServe 8%）− connector_bytes   87.36
    #   free_for_kv             free − connector_bytes：物理上限，别的进程占着的显存不能算给KV                    ≈120
    #   available_for_kv        min(available_for_kv_budget, free_for_kv)                                  87.36
    #
    # ② 一个block多少字节：不再是一个写死的公式，而是每个注意力builder声明的子池规格（sub_pool_specs）
    #   (a) MHA/GQA（aiter_attention.py + mha_kv_pool.py）：一个block = 本rank所有全注意力层的K、V（fp8时再加scale）
    #         K、V        2 × 层数 × 本rank的KV头 × head_dim × block_size × 元素字节 = 2×80×8×128×16×1B = 2,621,440 B
    #         fp8 scale   每个(层, KV头, token)一个fp32：2×80×8×16×4B = 81,920 B（bf16 KV没有这一项）
    #         合计2,703,360 B ≈ 2.58MB（每个字段按256B对齐，这里恰好对齐）→ 87.36GB / 2.58MB ≈ 34,698个block ≈ 55.5万个token
    #   (b) MLA（aiter_mla.py + mla_kv_pool.py，DeepSeek-V3等）：每层每token只存一行576维latent（512维k_c + 64维k_pe），
    #         没有单独的V和scale；而且不按TP切（只有1个"KV头"，-tp 8时8张卡各存一整份，DP-attention就是为此而来，
    #         见engine_core_mgr.py的【科普】DP-attention）
    #         61层 × 16 × 576 × 1B（fp8）= 562,176 B ≈ 549KB/block（每token ≈ 34KB，约为(a)的1/5；开MTP时MTP层也在池里，62行）
    #         DSA索引器（V3.2、GLM-5等）每block再加：61 × 16 × 144B（128维fp8 key + 4B scale，补齐到16B）= 140,544 B
    #   (c) 混合模型（gdn_attn.py，Qwen3-Next：12层全注意力 + 36层GDN，TP=1、bf16 KV）：
    #         PAGE子池只装12层全注意力：2 × 12 × 2个KV头 × 256 × 16 × 2B = 393,216 B = 384KB/block
    #         STATE子池：每请求一份GDN递归状态 ≈ 37.7MB（state_pool.py【科普】②），每请求槽位数 = 1 + num_spec（ReplaySSM时为1）
    #       DeepSeek-V4的滑动窗口、压缩器环也是STATE类（deepseek_v4_attn.py，第8批）
    #
    # ③ plan_pools（pool_layout/sub_pool_spec.py）：先扣STATE（请求没有它就跑不了），剩下的全给PAGE
    #     STATE条目数 = max_num_seqs × entries_per_req + extra_entries；PAGE条目数 = 剩余字节 // PAGE每条目字节
    #     例（(c)，max_num_seqs=512、不开MTP）：先扣 512 × 37.7MB ≈ 18.8GB，剩下的才按384KB切成block；
    #     剩余 ≤ 0 时抛InsufficientPoolBudget，下面换算成"--gpu-memory-utilization至少要多大"的报错
    # ④ PP：各stage层数不同，算出的block数不同；而block id要在每个stage都合法，所以all_reduce(MIN)取最小值
    # ⑤ 【分支】copy式状态checkpoint（state_pool.py【科普】⑤）：把快照切成"页单元"存进PAGE池，这里算出页单元的规格
    #
    # 对比nano-vllm：nano-vllm是 total×util − used − peak + current（used = total − free，含权重和非torch显存），
    #   再除以 2×层数×block_size×KV头×head_dim×dtype字节。同一个例子它得到 172.8 − 72 − 8 = 92.8GB，ATOM是87.36GB。ATOM多扣了：
    #   ① CUDA Graph私有池的估计（1.6GB；nano-vllm先按util填满KV再录图，graph池落在util之外的那部分显存里）；② 2%安全余量（3.84GB）；
    #   ③ 每请求状态池（STATE，混合模型/V4）；④ KV connector缓冲（PD分离）、RapidServe的额外8%；
    #   还多了⑤ 实际空闲显存的上限（同卡还有别的进程时）和⑥ PP的跨stage取最小；block字节数也从写死的MHA公式变成各后端声明
    # =====================================================================================
    def get_num_blocks(self) -> dict[str, object]:
        # 与__init__同理，临时把默认device设成本GPU；显存算完后恢复None
        torch.set_default_device(self.device)
        config = self.config
        hf_config = config.hf_config
        # 有的HF config不带head_dim字段，按 hidden_size / 头数 补上（MhaKvPool.from_hf_config等按它算每个KV头的大小）
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        # 驱动视角：本卡空闲字节、总字节（空闲量已扣掉同卡其他进程的占用）
        free, total = torch.cuda.mem_get_info()
        # 分配器视角：peak = 自warmup开头reset以来已分配字节的最高点；current = 此刻已分配的字节（见上方①的表）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # weights + peak activation tensors (PyTorch allocator high-water).
        # 【译】权重 + 峰值时的激活张量（PyTorch分配器的最高水位）。
        peak_torch = max(peak, current)
        # RCCL/NCCL buffers etc. held outside the allocator: device-used minus
        # torch-reserved. Ignoring it over-allocates KV and OOMs at runtime.
        # 【译】RCCL/NCCL的缓冲等由分配器之外持有的显存：设备已用量减去torch已预留量。忽略它会把KV分多了，运行时OOM。
        # 解释：memory_reserved()是分配器向驱动要来的全部显存（已分配 + 缓存着的空闲块）；warmup末尾已empty_cache，所以它≈current
        non_torch = max((total - free) - torch.cuda.memory_reserved(), 0)

        # CUDA Graph私有池的预估（见_estimate_cudagraph_overhead上方）和整卡2%的安全余量
        cudagraph_overhead = self._estimate_cudagraph_overhead()
        safety_margin = int(total * 0.02)

        # 预算 = 整卡 × util；扣掉一切非KV开销，剩下的就是KV（含每请求状态池）能用的
        budget = int(total * config.gpu_memory_utilization)
        non_kv_overhead = peak_torch + non_torch + cudagraph_overhead + safety_margin
        available_for_kv_budget = budget - non_kv_overhead

        # Physical clamp: never exceed what's actually free on the GPU.
        # Subclasses may reserve extra headroom (override point).
        # 【译】物理上限：绝不超过GPU上实际空闲的量。子类可以额外预留余量（覆写点）。
        # 解释：这两句注释前一句说的是下面的min(..., free_for_kv)，后一句才是紧接着的_kv_budget_extra_reserve
        extra_reserve = self._kv_budget_extra_reserve(total)
        available_for_kv_budget -= extra_reserve
        # GPU buffers the KV connector allocates after the KV cache is sized.
        # 【译】KV connector在KV cache定好大小之后才分配的GPU缓冲。
        # 【分支】PD分离：例如mooncake connector的MLA staging / landing池（第9批），现在不扣，之后分配时就没地方了
        connector_bytes = KVConnectorFactory.kv_budget_reserve_bytes(config)
        if connector_bytes:
            logger.info(
                "Reserving %.2fGB for KV connector buffers", connector_bytes / (1 << 30)
            )
            available_for_kv_budget -= connector_bytes
        # This prevents OOM when other processes share the GPU. Free HBM must
        # cover the connector buffers too.
        # 【译】这能防止其他进程共用GPU时OOM。空闲的HBM还必须能装下connector的缓冲。
        free_for_kv = free - connector_bytes
        if connector_bytes and free_for_kv <= 0:
            # Otherwise the KV cache gets nothing and the connector allocates
            # from memory that is not there.
            # 【译】否则KV cache什么也拿不到，而connector要从并不存在的显存里分配。
            raise RuntimeError(
                f"Only {free / (1 << 30):.2f}GB is free on the GPU, less than "
                f"the {connector_bytes / (1 << 30):.2f}GB the KV connector "
                "buffers need after the KV cache is sized; free GPU memory or "
                "turn those buffers off (ATOM_PD_MLA_STAGING=0 / "
                "ATOM_PD_MLA_LANDING=0)."
            )
        # 预算口径和物理口径取小（上方①的表；例子里是87.36GB）
        available_for_kv = min(available_for_kv_budget, free_for_kv)

        torch.set_default_device(None)

        # 各builder声明的子池规格（_sub_pool_specs上方；每block字节数的例子见本方法上方②）
        specs = self._sub_pool_specs()

        # Sub-pool sizing is pure arithmetic over the byte budget — see
        # atom/model_ops/attentions/pool_layout/sub_pool_spec.py. STATE classes (GDN
        # recurrent state, the V4 compressor ring, the V4 sliding window) take
        # their floor first because a request cannot run without them; the
        # PAGE class absorbs the rest. Which classes exist, and what they are
        # called, is the backend's business — the runner sizes them and
        # publishes the counts, then every consumer looks up the class it
        # declared itself.
        # 【译】子池的规模计算只是对字节预算做算术——见 atom/model_ops/attentions/pool_layout/sub_pool_spec.py。
        #   STATE类（GDN递归状态、V4压缩器环、V4滑动窗口）先拿走它们的下限，因为请求没有它们就跑不起来；PAGE类吸收剩下的全部。
        #   有哪些类、叫什么名字，是后端的事——runner只负责算出规模、发布条目数，之后每个使用者按自己声明的类名去查。
        try:
            plan = plan_pools(specs, available_for_kv, config.max_num_seqs)
        except InsufficientPoolBudget as exc:
            # Minimum gpu_memory_utilization that makes the budget just cover the
            # per-request pools. Rounded UP to the next 0.01 so the printed value
            # is actually sufficient, not the exact threshold.
            # 【译】让预算恰好盖住每请求池所需的最小gpu_memory_utilization。向上取整到下一个0.01，这样打印出的值确实够用，
            #   而不是恰好卡在阈值上。
            # 解释：min_util = (非KV开销 + 额外预留 + connector + STATE下限) / total。例：192GB的卡、非KV开销85.44GB、
            #   STATE下限18.8GB（上方③的例子），若 --gpu-memory-utilization 0.5：预算96GB − 85.44GB = 10.56GB < 18.8GB → 抛异常；
            #   min_util = (85.44 + 18.8) / 192 ≈ 0.543 → 提示 >= 0.55（这只是让PAGE刚好不为负，要有像样的KV容量还得更高）
            min_util = (
                non_kv_overhead + extra_reserve + connector_bytes + exc.reserved_bytes
            ) / total
            min_util_hint = math.ceil(min_util * 100) / 100
            base_msg = (
                f"Per-request cache tensor "
                f"({exc.reserved_bytes / (1 << 30):.2f}GB for "
                f"{exc.entries} slots) exceeds available KV budget "
                f"({available_for_kv / (1 << 30):.2f}GB) at "
                f"--gpu-memory-utilization {config.gpu_memory_utilization:.2f}."
            )
            # 三种提示：①卡在物理空闲显存上（调util没用，只能腾显存或减max_num_seqs）；②调到min_util就够；③util=1.0也不够
            if (
                available_for_kv_budget > free_for_kv
                or free_for_kv <= exc.reserved_bytes
            ):
                # The physical free-memory clamp is the binding limit, now or
                # once the budget grows — raising --gpu-memory-utilization won't help.
                # 【译】真正卡住的是物理空闲显存这个上限（现在就是，或者预算增大之后也会是）——调高--gpu-memory-utilization没有用。
                connector_note = (
                    f"{connector_bytes / (1 << 30):.2f}GB of it reserved for KV "
                    f"connector buffers; "
                    if connector_bytes
                    else ""
                )
                fix_msg = (
                    f" Only {free / (1 << 30):.2f}GB is physically free on the GPU "
                    f"({connector_note}other processes may be holding memory); "
                    f"raising --gpu-memory-utilization will NOT help. Free GPU memory "
                    f"or reduce --max-num-seqs (currently {config.max_num_seqs})."
                )
            elif min_util_hint <= 1.0:
                fix_msg = (
                    f" Set --gpu-memory-utilization >= {min_util_hint:.2f} "
                    f"(this only zeroes out the deficit; use a higher value for "
                    f"actual KV capacity) or reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs})."
                )
            else:
                fix_msg = (
                    f" Even --gpu-memory-utilization 1.0 is insufficient "
                    f"(would need {min_util:.2f}); reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs}) or free GPU memory."
                )
            raise RuntimeError(base_msg + fix_msg) from exc

        # PP stages compute different block counts; block ids must be valid on
        # every stage's KV tensor, so reduce to the global minimum. Fold the
        # result back into the plan before publishing anything: the plan is the
        # single source for every entry count, so it must never disagree with
        # the number the pool is actually built at.
        # 【译】各PP stage算出的block数不同；block id必须在每个stage的KV张量上都合法，所以归约到全局最小值。在发布任何东西之前
        #   把结果折回plan里：plan是所有条目数的唯一来源，它绝不能和池真正建出来的数量不一致。
        # 例：-pp 2时两个stage的层数、权重大小都不同，各自算出的block数不同，取较小的那个
        # 观察：TP组内各rank也各自算了一遍，但这里只对PP做了归约；EngineCore只取TP rank 0的结果让所有rank按它分配，
        #   靠的是TP各rank的显存占用对称（同卡上若有别的进程，非0号rank可能分配失败）
        num_kvcache_blocks = plan.paged_entries
        if config.pipeline_parallel_size > 1 and torch.distributed.is_initialized():
            t = torch.tensor(
                [num_kvcache_blocks], dtype=torch.int64, device=self.device
            )
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            num_kvcache_blocks = int(t.item())
            plan = plan.with_paged_entries(num_kvcache_blocks)

        # 一个PAGE条目（= 一个调度器block）的字节数，例(a)里是2,703,360
        block_bytes = plan.entry_bytes[plan.paged_class]
        # The whole plan travels to the engine process; BlockManager, the
        # sliding-window pool and the attention builder each index it by the
        # class name they declared. Nothing here needs to know those names.
        # 【译】整个plan会传到引擎进程；BlockManager、滑动窗口池和注意力builder各自按自己声明的类名去索引它。这里不需要知道那些名字。
        # self.pool_plan给本进程的builder用（warmup时还是空计划，现在换成真的）；config上的两份只在本runner进程里可见，
        #   引擎进程那边靠下面return的dict拿到（engine_core.py里回填）
        self.pool_plan = plan
        config.pool_entries = dict(plan.entries)
        config.pool_entries_per_req = dict(plan.entries_per_req)
        # Keep runtime state metadata out of Config.
        # 【译】运行时的状态元数据不放进Config。
        # ⑤ 后端声明它的每请求状态怎样交接（state_runtime.py的StateTransfer：fork / copy / replay / none，第4批）；
        #   copy = 快照切成"页单元"存进PAGE池（state_pool.py【科普】⑤），需要在这里算出页单元的规格
        transfer = self.attn_metadata_builder.state_transfer()
        uses_paged_state = transfer.copies
        if uses_paged_state and config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "PAGE-backed state checkpoints do not yet support pipeline "
                "parallelism: every stage must first agree on one atomic "
                "checkpoint/unit ownership transaction"
            )
        if uses_paged_state and config.enable_rapidserve:
            raise RuntimeError(
                "PAGE-backed state checkpoints do not yet support RapidServe "
                "prefill/decode disaggregation"
            )
        checkpoint_spec = None
        if uses_paged_state:
            if plan.paged_class is None:
                raise RuntimeError(
                    "PAGE-backed state checkpoints require a PAGE sub-pool"
                )
            # 一个活跃槽位（一份完整状态）的字节数
            slot_bytes = int(plan.entry_bytes[STATE_SLOT_CLASS])
            # None means the backend has not narrowed its image: carry it all.
            # 【译】None表示后端没有缩小它的快照映像：整份都要带上。
            narrowed = self.attn_metadata_builder.checkpoint_image_bytes()
            # The charged entry can be larger than the bytes a transfer
            # publishes: MHA alignment padding is not a region, and a draft
            # pool merged into the same class is ordinary KV after the
            # checkpoint regions. The backend names the published unit.
            # 【译】计价的条目可能比一次传输发布的字节多：MHA的对齐填充不是一个区域；并进同一类的草稿池在checkpoint区域之后，
            #   只是普通的KV。由后端说明发布的单元是多大。
            # 例（state_pool.py【科普】⑤）：Qwen3-Next一个页单元 = 384KB的block，37.7MB的槽位 → 约101个页单元一份快照
            page_unit_bytes = int(plan.entry_bytes[plan.paged_class])
            published_unit = getattr(
                self.attn_metadata_builder, "checkpoint_page_unit_bytes", None
            )
            if callable(published_unit):
                published = published_unit()
                if published is not None:
                    published = int(published)
                    if published <= 0 or published > page_unit_bytes:
                        raise RuntimeError(
                            f"checkpoint PAGE unit is {published} B but the "
                            f"paged pool entry is {page_unit_bytes} B"
                        )
                    page_unit_bytes = published
            checkpoint_spec = PagedStateCheckpointSpec(
                page_unit_bytes=page_unit_bytes,
                slot_bytes=slot_bytes,
                image_bytes=slot_bytes if narrowed is None else int(narrowed),
                layout_id=transfer.paged_layout_id,
            )
            logger.info(
                "PAGE-backed state checkpoints enabled: unit_bytes=%d, "
                "slot_bytes=%d, image_bytes=%d (%.1f%% of a slot), "
                "units_per_checkpoint=%d, layout=%s",
                checkpoint_spec.page_unit_bytes,
                checkpoint_spec.slot_bytes,
                checkpoint_spec.image_bytes,
                100.0 * checkpoint_spec.image_bytes / checkpoint_spec.slot_bytes,
                checkpoint_spec.units_per_checkpoint,
                checkpoint_spec.layout_id,
            )
        # 状态的运行时描述：本进程留一份；序列化成wire格式随返回值交给EngineCore（Scheduler / BlockManager用，第4批）
        state_runtime = StateRuntime(
            transfer=transfer,
            checkpoint_spec=checkpoint_spec,
        )
        self.state_runtime = state_runtime
        for name in sorted(plan.entries):
            logger.info(
                f"sub-pool {name}: entries={plan.entries[name]}, "
                f"entry_bytes={plan.entry_bytes[name]}, "
                f"reserved={plan.reserved_bytes[name] / (1 << 30):.2f}GB"
            )

        logger.info(
            f"Memory budget: total_gpu={total / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB, "
            f"utilization={config.gpu_memory_utilization}, "
            f"budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"available_for_kv={available_for_kv / (1 << 30):.2f}GB, "
            f"block_bytes={block_bytes}, "
            f"num_kvcache_blocks={num_kvcache_blocks}"
        )
        # Concurrent-capacity table: at each context-length percentage of
        # max_model_len, how many requests can simultaneously hold their
        # KV in the pool. Per-req block usage = ceil(ctx_len/block_size).
        # Active Slots are reserved; PAGE checkpoints borrow from the paged pool.
        # 【译】并发容量表：在max_model_len的各个百分比的上下文长度下，池里能同时放下多少个请求的KV。
        #   每请求占用的block数 = ceil(ctx_len/block_size)。活跃槽位是预留好的；PAGE checkpoint从分页池里借。
        # 例：34,698个block、block_size=16、max_model_len=32768、max_num_seqs=512：
        #   10%（3276 tok）→ 205 blk/req → min(512, 34698 // 205 = 169) = 169，受block数限制；
        #   100%（32768 tok）→ 2048 blk/req → 16。DCP时每个rank只存1/dcp的token，blk/req按本rank计
        max_model_len = config.max_model_len
        cap = config.max_num_seqs
        dcp_w = max(1, getattr(config, "decode_context_parallel_size", 1) or 1)
        pct_lines = []
        for pct in (10, 30, 50, 70, 90, 100):
            ctx = max(1, max_model_len * pct // 100)
            local_ctx = math.ceil(ctx / dcp_w)
            blocks_per_req = math.ceil(local_ctx / self.block_size)
            block_bound = (
                num_kvcache_blocks // blocks_per_req if blocks_per_req > 0 else 0
            )
            max_conc = min(cap, block_bound) if cap > 0 else block_bound
            bound_label = (
                "slots" if cap > 0 and max_conc == cap < block_bound else "blocks"
            )
            local_note = f" (local {local_ctx:>7})" if dcp_w > 1 else ""
            pct_lines.append(
                f"  {pct:>3}% ({ctx:>7} tok){local_note}: {blocks_per_req:>6} blk/req "
                f"→ max_concurrent={max_conc:<5} (bound by {bound_label})"
            )
        logger.info(
            f"Concurrent capacity vs context length "
            f"(max_model_len={max_model_len}, block_size={self.block_size}, "
            f"max_slots={cap}, pool_blocks={num_kvcache_blocks}"
            + (f", dcp={dcp_w} (blk/req is per-rank)" if dcp_w > 1 else "")
            + "):\n"
            + "\n".join(pct_lines)
        )

        assert num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache with block size({self.block_size}). "
            f"At least 1 block ({block_bytes / (1 << 20):.2f}MB) is required, "
            f"but available_for_kv={available_for_kv / (1 << 20):.2f}MB "
            f"(budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB)"
        )
        # get_num_blocks runs in the RUNNER subprocess, so nothing it writes
        # to `config` is visible to the engine process that builds
        # BlockManager. Ship the whole per-class entry table across instead of
        # a hand-picked field per architecture; consumers over there look up
        # the class they declared.
        # 【译】get_num_blocks运行在RUNNER子进程里，所以它写进 `config` 的任何东西，构建BlockManager的引擎进程都看不到。
        #   把整张按类别的条目表传过去，而不是为每种架构挑一个字段；那边的使用者按自己声明的类名去查。
        # 返回值的去向（engine_core.py）：num_kvcache_blocks → allocate_kv_cache与config.num_kvcache_blocks（BlockManager的block池大小）；
        #   pool_entries / pool_entries_per_req → BlockManager、滑动窗口池；state_runtime → StateRuntime.from_wire → Scheduler
        # 对比nano-vllm：nano-vllm直接改config.num_kvcache_blocks——它的Scheduler在同一个进程里读同一个对象
        return {
            "num_kvcache_blocks": num_kvcache_blocks,
            "pool_entries": dict(plan.entries),
            "pool_entries_per_req": dict(plan.entries_per_req),
            "state_runtime": state_runtime.to_wire(),
        }

    def _carve_paged_pool(self, blocks: int, buf=None):
        """One allocation for every paged pool, and each builder's region of it.

        The target's KV, its scales, whatever indexer cache rides along and a
        draft's sibling pool were up to five allocations under five names. One
        buffer here, for the reason the declarations were collapsed: a block's
        bytes are one fact, and every name it is stored under is a place the
        next reader can disagree with it. `kv_cache` is what is left, which is
        what the IPC export ships and the sleep path frees.

        Returns `(buffer, regions, builders)`, positionally aligned. `buf` is
        the imported pool on the P/D decode side; that side carves with the
        same walk rather than being told the offsets, so the two cannot drift.
        A builder that allocates its own answers zero and gets an empty region.
        """
        builders = [self.attn_metadata_builder]
        if hasattr(self, "draft_kv_builder"):
            builders.append(self.draft_kv_builder)
        # Before pricing, not between pricing and filling: the price below
        # comes from the walk.
        for builder in builders:
            builder.invalidate_pool_rows()
        sizes = [b.paged_pool_bytes(blocks) for b in builders]
        # A builder either allocates its own pool or takes exactly what its own
        # PAGE spec was charged -- nothing between. It catches the two hooks
        # disagreeing (separate overrides; a field added to one), not a stale
        # walk, which both readings share. A startup check and not a unit test
        # because what it catches surfaces while serving, never as a bad number
        # here.
        for builder, size in zip(builders, sizes):
            if not size:
                continue
            charged = blocks * sum(
                s.entry_bytes for s in builder.sub_pool_specs() if s.pool is Pool.PAGE
            )
            if size != charged:
                raise ValueError(
                    f"{type(builder).__name__} wants {size} B of the paged pool "
                    f"at {blocks} blocks but was charged {charged} B"
                )
        offsets, total = plan_regions(sizes)
        if buf is None:
            buf = torch.zeros(total, dtype=torch.uint8, device=self.device)
        elif buf.numel() < total:
            raise ValueError(
                f"the imported paged pool holds {buf.numel()} B but this side's "
                f"declarations want {total} B at {blocks} blocks"
            )
        regions = [buf[o : o + n] for o, n in zip(offsets, sizes)]
        return buf, regions, builders

    def _back_paged_pools(self, blocks: int, buf=None):
        """Back every builder's paged pool and publish what comes back.

        The one place, because there are two callers: this side allocates the
        buffer and the decode side of a P/D pair imports it. What the hook
        returns -- the aligned indexer dimension, the compact layer maps,
        GLM-5.3's k-pool tail -- has to reach its readers as a runner attribute
        on both, and a path that only backed the pool leaves the other half
        dereferencing attributes nobody set.
        """
        pool, regions, builders = self._carve_paged_pool(blocks, buf=buf)
        for builder, region in zip(builders, regions):
            for name, value in builder.allocate_kv_cache_tensors(
                blocks=blocks, buf=region
            ).items():
                setattr(self, name, value)
            logger.info(
                "%s caches %s",
                type(builder).__name__,
                ", ".join(
                    f"{n} {kind} rows" for kind, n in builder.row_counts().items()
                )
                or "nothing",
            )
        return pool

    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            num_kv_heads = 1
        # Promote to self so attention builders' build_kv_cache_tensor()
        # hooks can access it without re-deriving from hf_config.
        self.num_kv_heads = num_kv_heads
        self.aligned_index_dim = None  # set below for DeepSeek-V3.2

        # Primary KV cache allocation: one buffer, one region per builder, each
        # owning the layout inside its own. A draft that cannot share the
        # target's pool is simply the next region -- no second allocation, and
        # nothing that asks whether its attention matches the target's.
        #
        # What comes back is only what a region cannot carry: scalars like
        # `aligned_index_dim` and the compact layer maps, setattr'd here so
        # their readers find them where they always have.
        pool = self._back_paged_pools(num_kvcache_blocks)
        if pool.numel():
            self.kv_cache = pool

        # Per-request cache allocation (model-agnostic, delegated to the
        # attention metadata builder). For GDN this returns
        # `{"mamba_k_cache": ..., "mamba_v_cache": ...}`; for stateless
        # attentions it returns an empty dict (no-op). Values are setattr'd
        # on `self` so model layers can access them as `model_runner.<name>`.
        per_req_state = self.attn_metadata_builder.allocate_per_req_cache(
            self.pool_plan.entries
        )
        for name, value in per_req_state.items():
            setattr(self, name, value)
        # The pools are reachable through `self` only now, which is the
        # earliest the builder can touch its own addresses — and the last
        # moment before a request could.
        self.attn_metadata_builder.warmup_per_req_cache()

        # Build KVCacheConfig
        # lirong TODO: This is a simple solution to build KVCacheConfig,
        # models with only one type of attention, but not support multi-type of attention models.
        # We need to support it by kv_cache_group in the future.

        # Prepare list of models to bind KV cache
        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        # Keyed by each module's own global layer_num, which is what it looks
        # up at forward time (`kv_cache_data[f"layer_{self.layer_num}"]`). The
        # walk's ordinal used to be passed down and stood in for this, for a
        # pool row and for an index-map key; under PP it is none of the three.
        kv_cache_data = {}
        for model_name, model in models_to_bind:
            logger.info("Binding KV cache for the %s model", model_name)
            for module in model.modules():
                # A draft that owns an independent KV pool binds through its
                # sibling builder first; for unrecognized modules it returns
                # None and we fall through to the target builder.
                kv_cache_tensor = None
                if model_name == "draft" and hasattr(self, "draft_kv_builder"):
                    kv_cache_tensor = self.draft_kv_builder.build_kv_cache_tensor(
                        module
                    )
                # Per-attention-type binding is owned by the attention
                # metadata builder; ModelRunner only walks modules and
                # collects the resulting KVCacheTensor entries. The builder
                # returns None for modules it does not recognize (so a
                # sibling module like nn.LayerNorm is silently skipped).
                if kv_cache_tensor is None:
                    kv_cache_tensor = self.attn_metadata_builder.build_kv_cache_tensor(
                        module
                    )
                if kv_cache_tensor is not None:
                    kv_cache_data[f"layer_{module.layer_num}"] = kv_cache_tensor

        transfer_tensors = self.attn_metadata_builder.get_kv_transfer_tensors()
        if transfer_tensors is not None:
            # The tier is built inside `register_kv_caches` and needs
            # `state_entry_views` to name the bytes it packs. This is the only
            # place the builder and the connector are both in scope.
            transfer_tensors.state_backend = self.attn_metadata_builder
        if hasattr(self, "draft_kv_builder") and transfer_tensors is not None:
            transfer_tensors.merge_pages(
                self.draft_kv_builder.get_kv_transfer_tensors()
            )
        if transfer_tensors is not None:
            # After the draft's regions are in, and here because this is the
            # only place holding both the complete region list and the
            # scheduler's block count. The builders cannot answer this: each
            # counts in its own page, and the transfer protocol addresses
            # ``req.block_ids``, which are the scheduler's.
            transfer_tensors.set_block_count(num_kvcache_blocks)
        # The same count reaches the codecs, which likewise index by scheduler
        # block: MLA's cache is allocated in page-size-1 rows, so its own count
        # is block_ratio times larger and would make a codec treat one token as
        # a complete scheduler block.
        set_kv_cache_data(
            kv_cache_data,
            config,
            transfer_tensors,
            num_blocks=num_kvcache_blocks,
        )

        # Cross-validate: compare estimated vs actual KV cache allocation.
        # `actual_kv_bytes` includes BOTH the unified pool tensors (counted by
        # `block_bytes × num_blocks`) AND the per-request cache tensors (state
        # buffers + SWA window prefix embedded in unified_kv). The budget
        # math in `get_num_blocks()` reserves both separately, so the cross-
        # check must mirror that — otherwise it spuriously fires for any
        # backend that declares a per-request state pool (V4, GDN).
        post_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        actual_kv_bytes = post_alloc - pre_alloc
        # Each sub-pool contributes `entry_bytes × entries`. The counts come
        # straight from the sizing plan — which already absorbed the pipeline-
        # parallel reconciliation — so this mirrors the budget by construction
        # rather than re-deriving it.
        expected_kv_bytes = self.pool_plan.total_reserved_bytes
        if expected_kv_bytes > 0:
            diff_pct = abs(actual_kv_bytes - expected_kv_bytes) / expected_kv_bytes
            # 3% threshold: budget formula matches allocation exactly, but the
            # measured `post_alloc - pre_alloc` includes allocator alignment
            # (round to 256 B / 16 MiB segments) and whatever transient the
            # builders touch while initializing their pools, accounting for
            # ~2% noise on multi-GiB pools. Lower thresholds spuriously fire.
            if diff_pct > 0.03:
                logger.warning(
                    f"KV cache allocation mismatch: "
                    f"expected={expected_kv_bytes / (1 << 30):.3f}GB, "
                    f"actual={actual_kv_bytes / (1 << 30):.3f}GB, "
                    f"diff={diff_pct:.1%}"
                )

        # Skip on single-rank: a world_size==1 barrier is a no-op but still
        # forces lazy NCCL communicator creation (CUDA-allocs its buffers),
        # which can OOM/fail on single-card runs. The process group stays
        # initialized so get_tp_group() and friends keep working.
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        return True

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1:
            # Early exit.
            return 0, None
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank
        )
        max_tokens_across_dp = int(torch.max(num_tokens_across_dp))

        return max_tokens_across_dp - num_tokens, num_tokens_across_dp

    def _maybe_create_tbo_slices(
        self,
        batch,
        is_prefill,
        scheduled_bs,
        scheduled_tokens,
        num_scheduled_tokens,
        tbo_collective_active: bool,
    ):
        """Create TBO ubatch slices when the collective DP decision is True.

        With the packed-reduce path the eligibility (local + cross-DP AND)
        is decided in ``ForwardMode.decide``; here we just realise the split.
        """
        if not tbo_collective_active:
            return None

        tbo_num_reqs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs
        # tbo_collective_active is the OR-reduced cross-DP decision: this rank
        # is committed to splitting even if it's below ATOM_TBO_PREFILL_MIN_TOKENS
        # (a peer cleared the bar). force=True bypasses the local min-token gate
        # so we don't desync from peers and hang.
        ubatch_slices = maybe_create_ubatch_slices(
            num_reqs=tbo_num_reqs,
            num_tokens=scheduled_tokens,
            is_prefill=is_prefill,
            num_scheduled_tokens=num_scheduled_tokens if is_prefill else None,
            force=True,
        )
        if ubatch_slices is not None:
            logger.debug(
                f"[TBO] splitting {'prefill' if is_prefill else 'decode'} batch: "
                f"num_reqs={tbo_num_reqs}, ubatches={len(ubatch_slices)}"
            )
        return ubatch_slices

    def _local_tbo_eligibility(
        self, batch: ScheduledBatch
    ) -> tuple[bool, bool, int, int]:
        """This rank's TBO answer, and the PCP flags that ride with it.

        `(meets_min_tokens, can_split, ub0_tokens, ub1_tokens)`. The first two
        are reduced across DP inside the step's one packed all_gather -- OR and
        AND respectively -- so this is the local half of that question, not a
        decision. Sets the PCP+TBO routing flags as a side effect because they
        come out of the same sizing.
        """
        is_prefill = batch.total_tokens_num_prefill > 0
        if not self.config.enable_tbo:
            self._pcp_tbo_balanced_active = False
            self._pcp_bal_groups = None
            return (False, False, 0, 0)

        local_tbo = local_tbo_precompute(
            self.config, batch, is_prefill, batch.num_scheduled_tokens
        )

        # PCP+TBO prefill: split requests into two GROUPS at a request boundary
        # (never split a sequence's tokens), so each ubatch = "non-TBO PCP on a
        # request subset". Requires num_reqs >= 2 (request-boundary split needs
        # two non-empty groups); bs=1 falls back to non-TBO.
        pcp_size = self.config.prefill_context_parallel_size
        # Read by build_ubatch / run_model / prepare_prefill to route the
        # per-group path; reset every step so a stale value cannot route one.
        self._pcp_tbo_balanced_active = False
        self._pcp_bal_groups = None
        if is_prefill and pcp_size > 1 and not batch.is_dummy_run:
            # PCP is always dp=1, so the single-rank path returns this verbatim
            # as `tbo_collective_active`; ub0/ub1 are only read under dp > 1.
            local_tokens = batch.total_tokens_num_prefill // pcp_size
            eligible = batch.total_seqs_num_prefill >= 2 and local_tokens >= 2
            ub0 = local_tokens // 2
            local_tbo = (eligible, eligible, ub0, local_tokens - ub0)
            self._pcp_tbo_balanced_active = eligible
        return local_tbo

    def _dspark_apply_q_bucket(self, batch: ScheduledBatch) -> int | None:
        """Shrink this decode step's verify length to one CUDA-graph bucket q,
        and return it -- or None when nothing shrank and the step keeps the
        configured `num_spec_step + 1`.

        q = quantize_up(max ell_i + 1) over the batch (ell_i = last step's
        per-req schedule). All seqs then forward q tokens (anchor + q-1 drafts)
        instead of mtp_k+1, and replay picks the (bs, q) graph; the dropped
        draft suffix is re-drafted next step -> lossless.

        Mutates only the worker's batch copy (counts + scheduled_spec_decode_
        tokens truncated to q-1); KV stays reserved at mtp_k+1. No-op unless
        DSpark confidence scheduling is on and this is a pure-decode batch.
        The bucket math lives in `apply_q_bucket` so CPU CI can run it without
        importing this module (which loads AITER).
        """
        from atom.spec_decode.dspark_scheduler import apply_q_bucket

        drafter = getattr(self, "drafter", None)
        dspark = getattr(getattr(self, "config", None), "dspark", None)
        return apply_q_bucket(
            batch,
            drafter,
            dspark,
            apply_ragged=lambda scheduled_bs, full_q, by_req: self._dspark_apply_ragged(
                batch, scheduled_bs, full_q, by_req
            ),
        )

    def _dspark_apply_ragged(self, batch, scheduled_bs, full_q, by_req):
        """DSpark per-request RAGGED verify (paper §5.2 avoid-padding).

        Sets num_scheduled_tokens[i] = max(ell_i, num_bonus_i) + 1, clamped to
        [1, full_q] -- both terms request i's own, so two requests that accepted
        2 and 5 forward 3 and 6 tokens, not 6 and 6. Downstream V4 attn is
        marker-driven (cu_seqlens etc.), so ragged lengths flow through
        unchanged; the dropped draft suffix is re-drafted next step (lossless).
        KV stays reserved at mtp_k+1.
        """
        old_nst = batch.num_scheduled_tokens

        tp = getattr(self, "tokenID_processor", None)
        prev_b = getattr(tp, "prev_batch", None) if tp is not None else None
        cur_req = list(batch.req_ids[:scheduled_bs])
        prev_req = list(prev_b.req_ids) if prev_b is not None else None
        # Shrinking needs the previous batch to be exactly this decode set in
        # the same order (no new/prefill seqs, no reorder), because the per-
        # request lengths below are read off that batch's `num_bonus`. Any
        # deviation → boundary step.
        if prev_req is None or prev_req != cur_req:
            return  # boundary / reorder step: skip ragged, stay rectangular

        # Positional, which is what the `prev_req == cur_req` guard buys: entry
        # i is request i's own count from the step that produced it, before
        # `prepare_input_ids` remaps the deferred rows.
        nb = batch.num_bonus[:scheduled_bs]

        from atom.spec_decode.dspark_scheduler import (
            quantize_to_bucket,
            ragged_verify_lens,
            resolve_q_buckets,
        )

        # Bounded per request by nb[i]+1 (its anchor must stay inside its OWN
        # segment) and old_nst[i] (a stale ell must not grow a seq past what the
        # scheduler scheduled). None -> some request's bounds cross.
        new_len = ragged_verify_lens(
            (by_req.get(rid) for rid in batch.req_ids[:scheduled_bs]),
            full_q,
            nb,
            old_nst[:scheduled_bs],
        )
        if new_len is None:
            return  # stay rectangular

        if not (new_len < old_nst[:scheduled_bs]).any():
            return  # nothing to shrink this step -> Phase-1 layout

        # q_eff, and the replay-shape feasibility check, BEFORE anything on the
        # batch is rewritten. Shrinking the flat token layout is only safe if the
        # replay can follow it down; when it cannot, the rebuild is what makes
        # the step unsafe, so the decision has to come first.
        #   * max_seqlen_q (scalar) q_eff : the PER-SEQ length bound
        #     (>= max(new_len), quantized up to a captured bucket). Per-seq
        #     structures (compressor grid, rectangular indexer) size by it, so no
        #     seq can overflow them. It is NOT the total compute size -- that is
        #     the flat num_tokens bucket (running_tokens), sized to the
        #     real sum, so a long-tail seq no longer inflates the batch row count.
        buckets = resolve_q_buckets(self.config.dspark.ragged_graph_sizes, full_q)
        if self.enforce_eager:
            # Eager: no graph → capacity == exact Σ (no bucket). Scalar = batch max
            # real len (positions/attn bound); layout is pure flat Σ.
            q_eff = int(new_len.max()) if scheduled_bs > 0 else full_q
        else:
            # Graph: q_eff = smallest bucket >= the real MAX per-seq len, so no
            # seq ever exceeds q_eff. Per-seq structures (compressor grid,
            # rectangular indexer) size by q_eff and can't overflow -- no separate
            # full_q cap needed. The TOTAL compute size is `running_tokens`,
            # settled apart from this by `ForwardMode`, so q_eff no longer needs
            # to track the sum/avg.
            q_eff = (
                quantize_to_bucket(int(new_len.max()), buckets)
                if scheduled_bs > 0
                else full_q
            )
            # Under FULL cudagraphs nothing flat is recorded, so a packed run has
            # no width to land on and shrinking buys nothing -- the replay is
            # rectangular either way. WHICH width it lands on is not asked here:
            # `ForwardMode.decide` settles that on the DP-agreed total, and a
            # second search on this rank's own count is how the two came to
            # disagree. A miss there forwards the run eagerly, which is what the
            # branch above already does under `enforce_eager`.
            if not self._piecewise_cg_active():
                return  # stay rectangular

        # Rebuild scheduled_tokens (flat) to the ragged per-seq layout: keep the
        # first new_len[i] of each seq's old segment (token[0]=anchor, rest=draft
        # placeholders already populated by the scheduler from seq.token_ids).
        sched = np.asarray(batch.scheduled_tokens)
        old_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(old_nst[:scheduled_bs], out=old_cu[1:])
        new_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(new_len, out=new_cu[1:])
        total_new = int(new_cu[-1])
        new_sched = np.empty(total_new, dtype=sched.dtype)
        for i in range(scheduled_bs):
            s_old = int(old_cu[i])
            s_new = int(new_cu[i])
            new_sched[s_new : s_new + new_len[i]] = sched[s_old : s_old + new_len[i]]
        batch.scheduled_tokens = new_sched

        nst = old_nst.copy()
        nst[:scheduled_bs] = new_len
        batch.num_scheduled_tokens = nst
        prefill_tok = int(batch.total_tokens_num_prefill)
        batch.total_tokens_num_decode = total_new
        batch.total_tokens_num = prefill_tok + total_new
        # No marker is published for "this step shrank": a shrunk step is
        # `num_scheduled_tokens` with smaller entries, and every reader reads
        # those. `scheduled_spec_decode_tokens` keeps its full width --
        # `prepare_input_ids` slices it per request (`spec[i, :len_i - 1]`).
        return int(q_eff)

    def _verifies_drafts(self, batch: ScheduledBatch, is_prefill: bool) -> bool:
        """Whether this forward scores drafts with the rejection sampler.

        A P/D producer keeps the drafter only to write its context KV; the
        scheduler never pads drafts for it, so its decodes (the step that
        surfaces a deferred T0) are q=1 and have nothing to verify.
        """
        return (
            not is_prefill
            and hasattr(self, "drafter")
            and not self.tokenID_processor.is_pd_producer
            and not batch.is_dummy_run
        )

    def prepare_inputs(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        spec_decode_indices: tuple[np.ndarray, int] | None = None,
    ):
        # ======================== prepare_inputs：把CPU上的批元数据变成GPU张量 ========================
        # 在一步里的位置（完整时间线见下面forward上方的图）：
        #   forward → prepare_model（定形状ForwardMode、采样参数、cu_seqlens_q、input_ids）→ 【本函数】注意力元数据 +
        #   设置ForwardContext → run_model → postprocess
        # 前置：这些GPU张量都"住"在固定的缓冲区里。CpuGpuBuffer = 一块锁页内存（.np / .cpu）+ 一块同样大小的显存（.gpu），
        #   启动时按max_num_batched_tokens / max_num_seqs一次分配好（allocate_forward_vars、backends.py CommonAttentionBuilder.__init__）。
        #   每步CPU只往.np[:n]写numpy数组，再把前n个元素异步拷进.gpu（H2D，排在forward所在的stream上；GPU的异步执行与stream
        #   见engine_core.py文件开头【3】的前置）。地址每步不变——这正是后面CUDA Graph回放不用再拷一遍的前提。
        #   ATOM_H2D_BACKEND=packed（默认）时，同一"发布组"的多段先拼进一块锁页内存、一次拷完，再由kernel分发到各自的地址
        #   （utils/h2d.py；envs.py里ATOM_H2D_BACKEND上方的科普）
        # 本函数做三件事：
        #   ① 投机解码的验证步：算出哪些logits行用来验证草稿（spec_decode_metadata，【分支】第7批）
        #   ② attn_metadata_builder.build：由注意力后端（MHA / MLA / GDN / V4……，第6~8批）把批里每序列、每token的元数据写成GPU张量
        #   ③ 打包Context，连同DP / TBO信息一起set_forward_context；模型里每层注意力用get_forward_context()取用，不层层传参
        # 下面两个例子是MHA后端（aiter_attention.py）的情形；ScheduledBatch字段 → GPU张量的总映射见scheduler.py ScheduledBatch上方的表
        #
        # 例1：prefill批，2条序列（block_size=16；字段名就是forward_vars里的名字）
        #   A：prompt 20个token，无缓存           num_cached=0， num_scheduled=20，context_len=20，block_table=[3, 7]
        #   B：prompt 40个token，前缀缓存命中32个  num_cached=32，num_scheduled=8， context_len=40，block_table=[5, 2, 9]
        #   字段（长度）          A的部分             B的部分          拼接后（GPU）          谁写
        #   ─────────────────────────────────────────────────────────────────────────────────────────────────
        #   input_ids [28]       a0..a19             b32..b39         28个token             tokenIDProcessor.prepare_input_ids
        #   positions [28]       0..19               32..39           [0..19, 32..39]       prefill_positions：缓存长度 + 区间内偏移
        #   cu_seqlens_q [3]     20                  8                [0, 20, 28]           prepare_cu_seqlens_q（prepare_model里先发）
        #   cu_seqlens_k [3]     20                  40（含缓存）     [0, 20, 60]           _publish_prefill_seq_lens
        #   context_lens [2]     20                  40               [20, 40]              同上
        #   slot_mapping [28]    48..63, 112..115    144..151         28个槽位              _write_prefill_slots
        #   num_cached_tokens    0                   32               [0, 32]               只在有前缀命中时上传
        #   block_tables [2, M]  [3, 7, ...]         [5, 2, 9, ...]   每行补到M列           每步都在CPU上打包（算slot要用），
        #                                                                                   只在有前缀命中时上传
        #   slot = block_table[p // 16] × 16 + p % 16（与nano-vllm相同）：
        #     A：p=0..15在逻辑块0 → 物理块3 → 48..63；p=16..19在逻辑块1 → 物理块7 → 112..115
        #     B：p=32..39在逻辑块2 → 物理块9 → 144..151；p=0..31已在缓存里，不算也不写
        #   标量：max_seqlen_q=20，max_seqlen_k=40，total_kv=60；B有缓存 → has_cached=True、state=PREFILL_PREFIX
        #   running_bs：ForwardMode.decide把bs=2放到capture_sizes阶梯上（默认阶梯里有2）→ 2，不补行
        #
        # 例2：紧接着的decode批，3条序列（不开投机解码，q=1；假设--cudagraph-capture-sizes [1,2,4,8]，阶梯上 3 → 4）
        #   例1那一步在GPU上采样出tA、tB，开延迟输出时CPU还不知道它们；C不在例1的batch里（假设例1之前那一步是含C的
        #   decode批），它上一个token tC已在例1那一步的forward里交回CPU，由调度器写进了scheduled_tokens
        #   A：context_len=21（新token在位置20），block_table=[3, 7]
        #   B：context_len=41（位置40），           block_table=[5, 2, 9]
        #   C：context_len=33（位置32，正好开新块），block_table=[4, 6, 8]（块8是调度器may_append刚分的）
        #   字段（长度）          A        B        C        补的第4行    说明
        #   ─────────────────────────────────────────────────────────────────────────────────────────
        #   input_ids [3]        tA       tB       tC       0（尾部清0）  先按scheduled_tokens写[占位, 占位, tC]，再由
        #                                                                fill_deferred_decode_ids按decode_src=[0, 1, -1]从GPU上的
        #                                                                prev_token_ids覆盖前两格（-1 = 新来的，保留CPU给的值）
        #   positions [3]        20       40       32       —            decode_positions：context_len − 1
        #   slot_mapping [4]     116      152      128      -1           7×16+4、9×16+8、8×16+0；-1 = 不写KV
        #   context_lens [4]     21       41       33       0            0 = 这一行什么都不读
        #   cu_seqlens_q [5]     [0, 1, 2, 3, 3]                         尾部填总token数（"平的"前缀和），补的行长度为0
        #   block_tables [4, M]  [3, 7]   [5, 2, 9] [4, 6, 8] 补行
        #   kv_indptr [5]        [0, 2, 5, 8, 8]                         每条序列的block数cdiv(21,16)=2、3、3的前缀和
        #   kv_indices           [3, 7, 5, 2, 9, 4, 6, 8]                按kv_indptr把block_tables摊平（给持久化attention kernel）
        #   running_bs=4、running_tokens=4；scheduled_bs=3、scheduled_tokens=3
        #   补的那一行：eager / PIECEWISE路径里由_padded_decode_inputs把它的id和position清0；FULL回放时graph直接读整块buffer的
        #     前4行，position尾部可能是旧值，但slot=-1、context_len=0，这一行不写也不读KV，算出的结果被丢掉
        #
        # 对比nano-vllm：
        #   ① nano-vllm的prepare_prefill / prepare_decode每步新建Python列表 → torch.tensor(pin_memory=True).cuda()；
        #     ATOM写进常驻的CpuGpuBuffer、按组发布（一次或几次H2D），地址固定，所以CUDA Graph回放前不必再拷进graph_vars
        #   ② decode的input_ids：nano-vllm取seq.last_token（CPU上已知）；ATOM开延迟输出时取GPU上的上一步采样结果
        #   ③ 元数据格式随注意力后端变（MHA要kv_indptr / kv_indices，MLA、GDN、V4各有各的），所以交给builder生成；
        #     nano-vllm只有flash-attn一种，直接在ModelRunner里算
        #   ④ 多了"补行"：DP下各rank要同形状、CUDA Graph要阶梯上的尺寸，于是有scheduled_*（本rank真实的）和
        #     running_*（实际跑的、补齐后的）两套宽度，见forward_context.py ForwardMode的docstring
        # Always supplied, settled in `prepare_model` (which is where the reason
        # lives). The q-bucket shrink ran there too, so `batch` is already
        # reduced here.
        # 【译】总会传进来，在prepare_model里定下（理由也写在那里）。q-bucket收缩也已在那里执行，所以这里的batch已经是缩减后的。
        # 解释：说的是forward_mode参数；"q-bucket收缩"是DSpark按置信度缩短本步验证长度（_dspark_apply_q_bucket，第7批）
        is_prefill = batch.total_tokens_num_prefill > 0
        scheduled_bs = batch.total_seqs_num
        scheduled_tokens = batch.total_tokens_num
        num_scheduled_tokens = batch.num_scheduled_tokens
        # sync：ForwardMode.decide里那次跨DP all_gather的结果（单DP时为None）；num_tokens_across_dp给MoE / DPMetadata补齐行数用
        sync = forward_mode.sync
        num_tokens_across_dp = None if sync is None else sync.num_tokens_across_dp
        tbo_collective_active = forward_mode.tbo_collective_active
        ub_max_tokens_across_dp = None if sync is None else sync.ub_max_tokens_across_dp
        ub_tokens_across_dp = None if sync is None else sync.ub_tokens_across_dp
        running_tokens_are_unified = forward_mode.running_tokens_are_unified

        # 本步不拆TBO微batch时，清掉PCP+TBO按请求切分的标志（它由_local_tbo_eligibility按本地情况设置）
        if not tbo_collective_active:
            self._pcp_tbo_balanced_active = False

        # The step's two units, read from where they were settled: `running_bs`
        # sizes everything per-sequence, `running_tokens` everything per-row.
        # 【译】本步的两个单位，从定下它们的地方读：running_bs决定所有按序列计的宽度，running_tokens决定所有按行（token）计的高度。
        running_bs = forward_mode.running_bs
        running_tokens = forward_mode.running_tokens
        spec_decode_metadata = None
        # ① 【分支】投机解码的验证步（第7批）：每条序列本步有[锚点, 草稿1..k]共k+1行，算出哪些logits行验证草稿、哪一行出bonus
        #   例（mtp_k=1，2条序列，每条2行）：cu=[0, 2, 4] → bonus_logits_indices = cu[1:]−1 = [1, 3]（每条的最后一行），
        #     target_logits_indices = [0, 2]（第t行的logits验证第t+1行的草稿），见spec_decode/drafter.py prepare_spec_decode_indices
        if self._verifies_drafts(batch, is_prefill):
            _, lens, cu = self.attn_metadata_builder.decode_spans(batch)
            # Inside the staging window: without the token group it publishes
            # the indices itself. Packed indices already share the token upload.
            # 【译】处在暂存窗口内：没有token组时它自己发布这些索引；packed模式下这些索引已经和token一起上传了。
            spec_decode_metadata = self.drafter.calc_spec_decode_metadata(
                lens, cu[1:], input_ids, prepared_indices=spec_decode_indices
            )
        # ② 注意力元数据：先在compute stream上执行每请求状态的维护（槽位搬迁、checkpoint存取，第4批），
        #   再按prefill / decode分派到后端的prepare_prefill / prepare_decode（backends.py CommonAttentionBuilder.build，
        #   上面的例1、例2）；返回 (AttentionMetaData, positions的GPU视图)
        attn_metadata, positions = self.attn_metadata_builder.build(
            batch=batch,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_seqlen_q=forward_mode.max_seqlen_q,
        )
        # token已就位后后端还要做的准备（默认什么都不做；DeepSeek-V4.1的后端重写了它，第8批）
        self.attn_metadata_builder.prepare_model_inputs(input_ids, attn_metadata)
        # ③ Context：本步与注意力后端无关的信息（positions、prefill / decode、两套宽度、ForwardMode本身），
        #   模型各层、LM head、MoE都从get_forward_context().context读；is_dummy_run：DP陪跑的假批（engine_core.py DPEngineCoreProc）
        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            is_dummy_run=batch.is_dummy_run,
            scheduled_bs=forward_mode.scheduled_bs,
            scheduled_tokens=scheduled_tokens,
            running_bs=running_bs,
            running_tokens=running_tokens,
            running_tokens_are_unified=running_tokens_are_unified,
            forward_mode=forward_mode,
        )

        # 【分支】TBO（把batch拆成两个微batch，第9批 utils/tbo/）：tbo_collective_active时给出两个微batch的切片；
        #   PCP+TBO的prefill改为按请求边界切（见下面_build_pcp_balanced_slices上方）
        pcp_size = self.config.prefill_context_parallel_size
        _pcp_tbo_balanced = (
            is_prefill
            and pcp_size > 1
            and tbo_collective_active
            and not batch.is_dummy_run
            and getattr(self, "_pcp_tbo_balanced_active", False)
        )
        if _pcp_tbo_balanced:
            # Request-boundary split for PCP+TBO prefill (see
            # _build_pcp_balanced_slices). forward_vars stay GLOBAL here.
            # 【译】PCP+TBO prefill按请求边界切分（见_build_pcp_balanced_slices）。这里forward_vars保持全局（不按组拆）。
            ubatch_slices, self._pcp_bal_groups = self._build_pcp_balanced_slices(
                batch, num_scheduled_tokens, pcp_size
            )
        else:
            ubatch_slices = self._maybe_create_tbo_slices(
                batch,
                is_prefill,
                scheduled_bs if not is_prefill else 0,
                scheduled_tokens,
                num_scheduled_tokens,
                tbo_collective_active,
            )

        # 装进全局ForwardContext（forward_context.py），本步结束时forward里reset_forward_context清掉；
        #   num_tokens是本rank真实的token数，num_tokens_across_dp给DPMetadata（MoE跨DP的all_gather按它补齐）
        # 对比nano-vllm：相当于nano-vllm的set_context(...)，多了DP、投机解码、TBO三类信息
        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=scheduled_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_decode_metadata=spec_decode_metadata,
            ubatch_slices=ubatch_slices,
            ub_max_tokens_across_dp=ub_max_tokens_across_dp,
            ub_tokens_across_dp=ub_tokens_across_dp,
        )

    # prepare_sample：把每条序列的采样参数放到GPU，并在CPU上提前回答"本批要不要做top-k / top-p过滤、是不是全贪心"
    #   输入：batch.temperatures / top_ks / top_ps（numpy，[bs]，来自各请求的SamplingParams，第1批）
    #   输出：(temperatures [bs] GPU张量, top_ks, top_ps, all_greedy, needs_independent_noise)
    #     top_ks / top_ps有三种形态：None（整批都不过滤）、Python标量（整批相同）、[bs] GPU张量（各不相同）
    #   为什么在CPU上判断：这些都是numpy数组，CPU上一眼就能看出；若放到GPU上判断（如(top_ks != -1).any()），
    #     sampler里就得.item()把结果读回CPU，形成一次同步，打断GPU队列（同步点见engine_core.py文件开头【3】的前置）
    #   例：3条序列 temperature=[0, 0.7, 1.0]、top_k=[-1, -1, -1]、top_p=[1.0, 0.9, 0.9]
    #     → all_greedy=False；没有top_k → top_ks=None；有top_p且不全相同（1.0 vs 0.9）→ top_ps是[3]的GPU张量
    #     → sampler走带过滤的采样（sampler.py）；若三条都是temperature=0 → all_greedy=True，直接argmax
    #   每个TP rank都执行（ATOM每个rank都采样，见postprocess）。调用方：prepare_model；packed H2D时传入"token_inputs"组，
    #     和cu_seqlens_q、input_ids一起一次上传
    # 对比nano-vllm：nano-vllm只有temperature（prepare_sample只拷一个[bs]张量，且只在rank 0上），ATOM多了top-k / top-p、
    #   n>1时的独立噪声，以及"整批相同就传标量"的优化
    @h2d_producer("sampling")
    def prepare_sample(
        self, batch: ScheduledBatch, *, publication_group=None
    ) -> tuple[
        torch.Tensor, int | torch.Tensor | None, float | torch.Tensor | None, bool, bool
    ]:
        bs = batch.total_seqs_num

        # Check on CPU whether all requests are greedy (temperature=0)
        # 【译】在CPU上检查是否所有请求都是贪心（temperature=0）
        all_greedy = (batch.temperatures == 0).all()

        # Check on CPU whether any fan-out sibling needs per-row random noise.
        # Missing attribute (e.g. dummy runs, older callers) -> False.
        # 【译】在CPU上检查是否有fan-out的兄弟序列需要逐行独立的随机噪声。没有这个属性（如dummy run、旧的调用方）→ False。
        # 解释：fan-out指SamplingParams.n>1，一个请求展开成n条logits相同的序列；共用同一份噪声它们会采出同一个token
        noise = getattr(batch, "needs_independent_noise", None)
        needs_independent_noise = noise is not None and bool(noise.any())

        temp_buffer = self.forward_vars["temperatures"]
        # Clamp temperatures on CPU to avoid division by zero in sampler
        # 【译】在CPU上把temperature钳到不小于SAMPLER_EPS，避免sampler里除以0
        # 解释：temperature=0被钳成SAMPLER_EPS，sampler把<= SAMPLER_EPS的行当作贪心行处理（sampler.py _apply_greedy_tokens）
        np.maximum(batch.temperatures, SAMPLER_EPS, out=temp_buffer.np[:bs])

        # Check on CPU whether filtering is needed to avoid GPU sync in sampler.
        # If no filtering needed, return None to skip GPU copy entirely.
        # 【译】在CPU上检查是否需要过滤，避免sampler里的GPU同步。不需要过滤时返回None，完全省掉GPU拷贝。
        # 解释：top_k=-1表示不限制，top_p=1.0表示不截断
        needs_topk = (batch.top_ks != -1).any()
        needs_topp = (batch.top_ps < 1.0).any()

        # Uniform filters are already known on the CPU. Keep them as scalars
        # for AITER's fast dispatch instead of uploading and reading them back.
        # 【译】整批相同的过滤参数在CPU上就已知。把它们保留成标量，走AITER的快速分派，而不是先上传再读回来。
        top_ks = top_ps = None
        top_k_count = top_p_count = None
        if needs_topk:
            if bs == 1 or (batch.top_ks == batch.top_ks[0]).all():
                top_ks = int(batch.top_ks[0])
            else:
                top_k_buffer = self.forward_vars["top_ks"]
                top_k_buffer.np[:bs] = batch.top_ks
                top_ks = top_k_buffer.gpu[:bs]
                top_k_count = bs

        if needs_topp:
            if bs == 1 or (batch.top_ps == batch.top_ps[0]).all():
                top_ps = float(np.float32(batch.top_ps[0]))
            else:
                top_p_buffer = self.forward_vars["top_ps"]
                top_p_buffer.np[:bs] = batch.top_ps
                top_ps = top_p_buffer.gpu[:bs]
                top_p_count = bs

        # H2D发布：counts[i] = 成员i本步上传的元素个数，None = 本步不上传；没有传入共享组（direct模式）时自己立即发布
        group = (
            self.h2d_groups["sampling"]
            if publication_group is None
            else publication_group
        )
        counts = group.counts
        counts[group.indices["temperatures"]] = bs
        counts[group.indices["top_ks"]] = top_k_count
        counts[group.indices["top_ps"]] = top_p_count
        if publication_group is None:
            group.publish(counts)
        temperatures = temp_buffer.gpu[:bs]

        return temperatures, top_ks, top_ps, all_greedy, needs_independent_noise

    # prepare_model：一步forward的"准备阶段"总入口（forward调用），按顺序做：
    #   ① _dspark_apply_q_bucket：DSpark按置信度把本步验证长度缩到某个q档（【分支】第7批；不开DSpark时返回None）
    #   ② ForwardMode.decide：定下本步的形状和执行方式（forward_context.py，规范位置）——
    #      DP>1时先做本步唯一一次跨DP集合通信（DP组gloo上的一次打包all_gather，交换各rank的token数、bs、是否prefill等），然后
    #        running_bs = capture_sizes阶梯上 >= 全组最大bs 的最小一档（超过最大档就等于原值）
    #        use_cudagraph = 非prefill 且 全组都在decode 且 未enforce_eager 且 后端不拒绝 且 落在阶梯上
    #      例（阶梯[1..8, 16, 32, 48, 64, 128, 256, 512]）：bs=5 → 5；bs=9 → 16；bs=65 → 128（补63行）；bs=600 → 600、走eager
    #      例（DP=2）：rank0有3条decode、rank1有5条 → 全组最大5 → 两个rank的running_bs都是5，rank0补2行
    #      --enforce-eager时capture_sizes保持[0]（__init__），任何bs都不在阶梯上 → 永远eager，单DP时running_bs = bs
    #   ③ prepare_sample：采样参数
    #   ④ cu_seqlens_q和input_ids：cu_seqlens_q必须先于input_ids，fill_deferred_decode_ids要靠它找到每个请求的区间
    #   ⑤ prepare_inputs：注意力元数据 + ForwardContext
    #   ⑥ 草稿模型的锚点覆盖（【分支】投机解码）
    #   返回 (input_ids, temperatures, top_ks, top_ps, all_greedy, needs_independent_noise)，交给run_model和postprocess
    # H2D的两种发布方式（ATOM_H2D_BACKEND，默认packed）：
    #   direct（token_group为None）：各组各拷各的——cu_seqlens_q一拷、input_ids一拷、采样参数一拷
    #   packed：采样参数、cu_seqlens_q、input_ids（以及投机解码的验证索引）合成一个"token_inputs"组，
    #     各生产者只填counts，由prepare_input_ids在启动它的kernel之前一次发布
    # 对比nano-vllm：nano-vllm的run()里只有prepare_prefill / prepare_decode + prepare_sample两行；ATOM多出的是
    #   形状决策（DP对齐、CUDA Graph阶梯）、H2D发布的编排，以及投机解码的准备
    @_profile_runner_stage
    def prepare_model(self, batch: ScheduledBatch):
        shrunk_q = self._dspark_apply_q_bucket(batch)
        # The step's shape, settled once. Here rather than in prepare_inputs
        # because DSpark under DP needs the reduced query length BEFORE
        # prepare_input_ids sizes the buffer, and because one call site is the
        # only way the step is guaranteed a single cross-DP collective.
        # 【译】本步的形状，只定一次。放在这里而不是prepare_inputs里，一是DP下的DSpark要在prepare_input_ids确定buffer大小之前
        #   就拿到缩减后的query长度；二是只有一个调用点，才能保证本步只有一次跨DP集合通信。
        # 参数：step_needs_eager = 后端只看这批就知道不能回放graph（默认False，deepseek_v41/backend.py等重写）；
        #   capture_sizes = 升序的已捕获bs阶梯；captured_tokens = PIECEWISE时录过的num_tokens档位；
        #   is_block_drafter = DSpark（块并行草稿，其q要跨DP取最大）；local_tbo = 本rank的TBO资格（【分支】第9批）；
        #   max_seqlen_q = 每条序列本步的query行数：不开投机解码为1，MTP为mtp_k+1，DSpark收缩后为shrunk_q；
        #   graph_shapes = 只有P/D producer传（见_producer_graph_shapes）；dp_stage_report = V4.1的有界重放（第8批）
        dp_size = self.config.parallel_config.data_parallel_size
        forward_mode = ForwardMode.decide(
            batch=batch,
            dp_size=dp_size,
            dp_group=get_dp_group().cpu_group if dp_size > 1 else None,
            enforce_eager=self.enforce_eager,
            step_needs_eager=self.attn_metadata_builder.step_needs_eager(batch),
            capture_sizes=self.capture_sizes_np,
            captured_tokens=(
                self._piecewise_sorted_tokens if self._piecewise_cg_active() else None
            ),
            is_block_drafter=(
                hasattr(self, "drafter") and self.drafter.is_block_drafter
            ),
            tbo_on=self.config.enable_tbo,
            local_tbo=self._local_tbo_eligibility(batch),
            max_seqlen_q=(batch.num_spec_step + 1 if shrunk_q is None else shrunk_q),
            graph_shapes=self._producer_graph_shapes(),
            dp_stage_report=(
                self.attn_metadata_builder.dp_stage_report(batch)
                if dp_size > 1
                else None
            ),
        )
        # Stash the DP-wide prefill OR for the EPLB prefill gate; reused free by
        # on_forward_pass_end when the DP group == the migration (EP) group.
        # 【译】把DP范围内"有没有rank在prefill"的OR结果存下来，给EPLB的prefill闸门用；DP组就是迁移（EP）组时，
        #   on_forward_pass_end可以免费复用它。
        # 【分支】EPLB（给热点专家放副本，第7批 model_ops/eplb.py）
        self._eplb_any_rank_has_prefill = (
            None
            if forward_mode.sync is None
            else forward_mode.sync.any_rank_has_prefill
        )
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        # token_group：packed模式下的"token_inputs"组（采样参数 + cu_seqlens_q + input_ids），direct模式下没有这个组
        token_group = self.h2d_groups.get("token_inputs")
        spec_decode_indices = None
        temperatures, top_ks, top_ps, all_greedy, needs_independent_noise = (
            self.prepare_sample(batch, publication_group=token_group)
        )
        # ④ cu_seqlens_q（running_bs+1项，补的行是"平的"前缀和）先于input_ids；两种发布方式只差在谁、何时真正发起H2D
        if token_group is None:
            self.attn_metadata_builder.publish_cu_seqlens_q(batch, forward_mode)
            input_ids = self.tokenID_processor.prepare_input_ids(
                batch, forward_mode.max_seqlen_q
            )
        else:
            cu_count = self.attn_metadata_builder.prepare_cu_seqlens_q(
                batch, forward_mode
            )
            token_group.counts[token_group.indices["cu_seqlens_q"]] = cu_count
            spec_group = self.h2d_groups.get("spec_decode")
            if spec_group is not None:
                if self._verifies_drafts(batch, batch.total_tokens_num_prefill > 0):
                    _, lens, cu = self.attn_metadata_builder.decode_spans(batch)
                    spec_decode_indices = self.drafter.prepare_spec_decode_indices(
                        lens, cu[1:], token_group
                    )
                else:
                    # A preceding decode may have filled these counts. Forwards
                    # that verify nothing must not publish its stale indices.
                    # 【译】之前的某次decode可能已经填过这些counts。不做验证的forward不能把那些过期的索引发布出去。
                    for binding in spec_group.members:
                        token_group.counts[token_group.indices[binding.name]] = None
            # No GPU consumer runs between these host producers and this
            # upload. Token assembly publishes before launching its kernel.
            # 【译】在这些主机端生产者和这次上传之间，没有任何GPU消费者在运行。token组装在启动自己的kernel之前发布。
            input_ids = self.tokenID_processor.prepare_input_ids(
                batch, forward_mode.max_seqlen_q, publication_group=token_group
            )
        self.prepare_inputs(
            batch,
            input_ids,
            forward_mode=forward_mode,
            spec_decode_indices=spec_decode_indices,
        )

        # Stage scheduler anchor overrides while this forward's staging window
        # is still open. This pinned mirror is reused, so copying it later from
        # postprocess would fall outside the preparation completion event.
        # 【译】趁本次forward的暂存窗口还开着，把调度器给的锚点覆盖值放上去。这块锁页镜像会被复用，如果晚到postprocess再拷，
        #   就落在"准备阶段完成"event的保护范围之外了。
        # 【分支】投机解码：还在prompt中间（chunked prefill的中间块）的序列没有可用的采样token，草稿的锚点改用调度器给的
        #   下一个prompt token（batch.next_token_ids，-1 = 用采样值），在propose_draft_token_ids里替换
        if hasattr(self, "drafter"):
            forward_context = get_forward_context()
            if batch.next_token_ids is not None:
                forward_context.context.draft_anchor_overrides = (
                    self.drafter.anchors_to_gpu(batch.next_token_ids)
                )
        return (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        )

    @staticmethod
    def _detailed_label_suffix(batch: ScheduledBatch | None) -> str:
        """Detailed attention aggregates for the trace label, or ``""``.

        These fields are only populated by
        `Scheduler.compute_detailed_aggregates` when profiling is active
        and ``ATOM_ENABLE_DETAILED_ANNOTATION`` is set, so on the normal
        (unprofiled) path this returns an empty string without any extra work.
        Appending here keeps the annotation on the ``prefill[]``/``decode[]``
        ``record_function`` (a GPU-recognized layer) instead of nesting an
        extra span above ``run_model``.
        """
        # 【译】给trace标签用的注意力聚合量明细，或空字符串""。
        #
        #   这些字段只有在profiling开启且设置了ATOM_ENABLE_DETAILED_ANNOTATION时，才由Scheduler.compute_detailed_aggregates填写，
        #   所以在正常（不profiling）的路径上它直接返回空串，不做额外工作。追加在这里，是为了让标注留在prefill[] / decode[]
        #   这个record_function上（GPU能识别的一层），而不是在run_model上面再嵌套一层span。
        # 【分支】profiler / 指标（第10批）：sqsq、sqsk、sk是注意力FLOP估算用的聚合量
        if batch is None or batch.detailed_sqsq is None:
            return ""
        return (
            f" sqsq={batch.detailed_sqsq}"
            f" sqsk={batch.detailed_sqsk}"
            f" sk={batch.detailed_sk}"
        )

    # ---------------- 【分支】PCP + TBO的prefill按请求边界切分（下面三个函数），见第9批 ----------------
    # 前置①PCP（prefill上下文并行，--prefill-context-parallel-size）：prefill时把本批的全部token轮询（round-robin）分给
    #   pcp个rank，token t归 t % pcp 号rank；每个rank只算自己那1/pcp个token的query，但保留完整的KV
    #   （distributed/pcp_utils.py文件头），算完再all_gather + 重排回原顺序（pcp_allgather_rerange）。
    #   为什么轮询而不是切成连续的几段：因果注意力里越靠后的token要看的历史越多，连续切分时拿到后半段的rank活重得多；
    #   轮询让每个rank分到早晚各半的token。token总数必须是pcp的倍数，不够就补dummy token（pcp_pad_len）。
    #   例：pcp=2、6个token → rank0算t0、t2、t4，rank1算t1、t3、t5。PCP为什么会增加卡数见arg_utils.py
    #   prefill_context_parallel_size上方
    # 前置②TBO：把一个batch拆成两个微batch交错执行，让通信与计算重叠，见arg_utils.py enable_tbo上方的【科普】
    # 问题：TBO一般在token中点切；与PCP同开时，一条序列可能被切进两个微batch，各自轮询分片后序列内的顺序就乱了
    #   （docstring里的R1/R2问题）。所以改成按"请求边界"切：每组是一个完整、独立的"非TBO的PCP小批"
    # 例（pcp=2，3条prefill序列，token数[101, 300, 200]，共601）：
    #   cum=[101, 401, 601]，target=300 → searchsorted得1 → split_idx=2：组0 = 请求[0, 2)共401个，组1 = 请求[2, 3)共200个
    #   组0补到402（pcp的倍数）→ 每个rank本地201个；组1已是200 → 每个rank本地100个
    #   每个rank上的本地拼接：[组0的201个 | 组1的100个] → ubatch_slices的token_slice = [0:201]、[201:301]
    #   groups = [PcpBalGroup(0, 2, 0, 401, 402), PcpBalGroup(2, 3, 401, 601, 200)]
    #   观察：注释说取"累计token数最接近target的请求边界"，代码实际取的是"第一个累计数 >= target的边界"，两者不总相同：
    #     [280, 100, 220]时target=300，代码选在380处（组0 = 380个），而280处离300更近
    # 调用方：prepare_inputs（_pcp_tbo_balanced为True时）；结果由run_model（_apply_pcp_balanced_stripe /
    #   _restore_pcp_balanced_output）和V4注意力后端的build_ubatch_prefill_metadata使用
    def _build_pcp_balanced_slices(
        self,
        batch: ScheduledBatch,
        num_scheduled_tokens: np.ndarray,
        pcp_size: int,
    ) -> "tuple[list[UBatchSlice], list[PcpBalGroup]]":
        """Build request-boundary-split ubatch slices for PCP+TBO prefill.

        Split REQUESTS into two groups at a request boundary near the token
        midpoint. Each group is an independent "non-TBO PCP mini-batch": padded
        to a pcp multiple and round-robin striped as a whole, so every sequence
        stays intact in one group (root-fixes token-split R1/R2). forward_vars
        stay GLOBAL here; build_ubatch_prefill_metadata slices the FULL
        (un-reindexed) metadata per group and calls _apply_pcp_reindex on it.

        Returns (ubatch_slices, groups): token_slice is in the LOCAL concat
        space [g0_local | g1_local] that run_model produces (see
        _apply_pcp_balanced_stripe); groups are the PcpBalGroup descriptors
        consumed by run_model (per-group stripe) and
        build_ubatch_prefill_metadata (slice + reindex).
        """
        # 【译】为PCP+TBO prefill构造按请求边界切分的微batch切片。
        #
        #   在token中点附近的某个请求边界处，把"请求"分成两组。每组是一个独立的"非TBO PCP小批"：整体补齐到pcp的倍数、
        #   整体做轮询分片，这样每条序列都完整地留在一个组里（从根上修复按token切分带来的R1/R2问题）。这里forward_vars
        #   保持全局；build_ubatch_prefill_metadata按组切出完整的（未重排索引的）元数据，再对它调用_apply_pcp_reindex。
        #
        #   返回(ubatch_slices, groups)：token_slice位于run_model产生的本地拼接空间[g0_local | g1_local]中
        #   （见_apply_pcp_balanced_stripe）；groups是PcpBalGroup描述符，由run_model（按组分片）和
        #   build_ubatch_prefill_metadata（切片 + 重排索引）使用。
        num_prefill_reqs = batch.total_seqs_num_prefill
        per_req = np.asarray(num_scheduled_tokens[:num_prefill_reqs], dtype=np.int64)
        total_tok = int(per_req.sum())
        # 【译】（行尾注释）cum[j] = 第0..j个请求的token数之和
        cum = np.cumsum(per_req)  # cum[j] = sum of reqs [0..j]
        target = total_tok // 2
        # request boundary whose cumulative token count is closest to target
        # 【译】累计token数最接近target的那个请求边界（实际取法见上面函数上方的"观察"）
        # 夹到[1, num_prefill_reqs−1]：保证两组都非空（_local_tbo_eligibility已要求至少2条prefill序列）
        split_idx = int(np.searchsorted(cum, target, side="left")) + 1
        split_idx = max(1, min(split_idx, num_prefill_reqs - 1))
        # global token count of group0 (reqs [0:split_idx])
        # 【译】组0（请求[0:split_idx]）的全局token数
        # h0 / h1：两组各自补齐到pcp倍数后的token数；l0 / l1：每个rank上分到的本地token数
        b0 = int(cum[split_idx - 1])
        h0 = pcp_pad_len(b0, pcp_size)
        h1 = pcp_pad_len(total_tok - b0, pcp_size)
        l0 = h0 // pcp_size
        l1 = h1 // pcp_size
        ubatch_slices = [
            UBatchSlice(
                request_slice=slice(0, split_idx),
                token_slice=slice(0, l0),
            ),
            UBatchSlice(
                request_slice=slice(split_idx, num_prefill_reqs),
                token_slice=slice(l0, l0 + l1),
            ),
        ]
        groups = [
            PcpBalGroup(0, split_idx, 0, b0, h0),
            PcpBalGroup(split_idx, num_prefill_reqs, b0, total_tok, h1),
        ]
        return ubatch_slices, groups

    def _apply_pcp_balanced_stripe(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
        forward_context,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """PCP+TBO prefill per-group round-robin stripe, before UBatchWrapper.

        Each request group is padded to a pcp multiple and round-robin striped
        as a WHOLE (so sequences stay intact per group), then the two groups'
        1/pcp shards are concatenated into [g0_local | g1_local]. token_slice
        (built in prepare_inputs) indexes into this concat. Returns the striped
        (input_ids, positions).
        """
        # 【译】PCP+TBO prefill在进入UBatchWrapper之前按组做轮询分片。
        #
        #   每个请求组补齐到pcp的倍数后整体轮询分片（这样序列在组内保持完整），再把两个组各自的1/pcp分片拼成
        #   [g0_local | g1_local]。token_slice（在prepare_inputs里构造）索引的就是这个拼接结果。返回分片后的(input_ids, positions)。
        # 例（接上例，rank0）：组0的402个token（401个真实 + 1个补的0）取偶数位 → 201个；组1的200个取偶数位 → 100个；拼成301个
        g_ids, g_pos = [], []
        for grp in groups:
            seg_ids = input_ids[grp.tok_start : grp.tok_end]
            seg_pos = positions[grp.tok_start : grp.tok_end]
            pad = grp.pad_total - (grp.tok_end - grp.tok_start)
            if pad > 0:
                seg_ids = torch.cat([seg_ids, seg_ids.new_zeros(pad)])
                seg_pos = torch.cat([seg_pos, seg_pos.new_zeros(pad)])
            g_ids.append(pcp_round_robin_split(seg_ids, pcp_size))
            g_pos.append(pcp_round_robin_split(seg_pos, pcp_size))
        input_ids = torch.cat(g_ids)
        positions = torch.cat(g_pos)
        # context.positions = local per-group concat so _make_ubatch_context
        # slices each ubatch's forward positions correctly.
        # 【译】context.positions设成按组拼接的本地结果，这样_make_ubatch_context才能为每个微batch正确切出forward用的positions。
        forward_context.context.positions = positions
        # Hash MoE: local per-group-concat ids. Each ForCausalLM.forward
        # allgathers its ubatch's slice (g_i local, H_i/pcp) across pcp ranks →
        # H_i ids, matching moe_pcp_merge_forward's per-ubatch hidden allgather.
        # 【译】Hash MoE：按组拼接的本地token id。每个ForCausalLM.forward把自己微batch的那一段（g_i的本地部分，H_i/pcp个）
        #   在pcp各rank之间all_gather → 得到H_i个id，与moe_pcp_merge_forward对每个微batch做的hidden all_gather对应。
        # 解释：Hash MoE指按token id哈希选专家（DeepSeek-V4，第8批），所以MoE要拿到和hidden同样排列的完整id；
        #   ATOM_PCP_MOE_MERGE默认开（MoE在完整token集合上算，envs.py）
        if envs.ATOM_PCP_MOE_MERGE:
            forward_context.context.input_ids = input_ids
        return input_ids, positions

    def _restore_pcp_balanced_output(
        self,
        mo: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
    ) -> torch.Tensor:
        """Restore PCP+TBO request-boundary-split output.

        UBatchWrapper concatenated the two groups' 1/pcp output shards
        [g0_local | g1_local]. Each group was striped independently, so restore
        per group: pcp_allgather_rerange its shard back to the group's global
        order, crop the per-group pad, then concat to the full global sequence.
        """
        # 【译】还原PCP+TBO按请求边界切分后的输出。
        #
        #   UBatchWrapper把两个组各自的1/pcp输出分片拼成了[g0_local | g1_local]。每组是独立分片的，所以按组还原：
        #   用pcp_allgather_rerange把该组的分片收齐并恢复成组内的全局顺序，裁掉组内补的pad，再拼成完整的全局序列。
        # 例（接上例）：mo = [201 | 100]行 → 组0：all_gather得402行、重排、裁成401行；组1：得200行；拼成601行（与原prefill顺序一致）
        outs = []
        off = 0
        for grp in groups:
            # 【译】（行尾注释）该组每个rank上的1/pcp个token数
            local_len = grp.pad_total // pcp_size  # group's 1/pcp token count
            seg = pcp_allgather_rerange(mo[off : off + local_len], pcp_size)
            # 【译】（行尾注释）裁掉组内补的pad
            outs.append(seg[: grp.tok_end - grp.tok_start])  # crop per-group pad
            off += local_len
        return torch.cat(outs)

    # ---------------- 【分支】PP + GLM-5.2 DSA IndexShare：跨stage传递稀疏top-k，见第8、9批 ----------------
    # 前置①DSA：每层的索引器为每个query从全部历史token里挑top-k个，主注意力只看这k个（入门见arg_utils.py
    #   index_cache_dtype上方的【科普】DSA，图解见config.py glm5_kpool_block_size上方【科普】②）
    # 前置②IndexShare：不是每层都跑索引器——"full"层自己算top-k、写进每个rank的暂存buffer _sparse_kv_indices_gpu，
    #   紧随其后的"shared"层直接复用这份top-k，省掉索引器的计算
    # 前置③PP：按层切成几个stage，每个stage是独立进程，stage之间只传hidden state（arg_utils.py pipeline_parallel_size上方）
    # 问题：PP边界如果正好落在一组shared层中间，下一个stage开头那几层shared要用的top-k还在上一个stage的buffer里
    #   → 上一个stage把top-k跟hidden_states一起发过去（run_model里的"sparse_kv_indices"），下一个stage收到后拷进自己的buffer
    # 例（假设index_topk_freq=4、index_skip_topk_offset=1，_is_shared(i) = max(i−1, 0) % 4 != 0）：
    #   层0、1、5、9、13…是full，其余是shared；-pp 2、边界在第39层：max(38, 0) % 4 = 2 ≠ 0 → 第39层是shared
    #   → stage0（层[0, 39)）_pp_send_needs_sparse=True，stage1（层[39, 78)）_pp_recv_needs_sparse=True
    # 只算一次（_pp_share_indexer_ready），由run_model在PP>1时调用
    def _setup_pp_shared_indexer(self):
        """Cache per-rank predicates for GLM-5.2 DSA IndexShare PP-boundary
        top-k transfer. Computed once.

        A "shared" attention layer reuses the prior "full" layer's sparse top-k
        via the per-rank scratch buffer ``_sparse_kv_indices_gpu``. When a PP
        boundary splits a shared group, the receiving rank's leading shared
        layers need the sending rank's top-k, so it is carried across the
        boundary. No-op for dense models, sparse models with no shared layers,
        pp=1, or when every rank starts on a "full" layer.
        """
        # 【译】为GLM-5.2 DSA IndexShare在PP边界上传递top-k，缓存每个rank的判断结果。只计算一次。
        #
        #   "shared"注意力层通过每个rank的暂存buffer _sparse_kv_indices_gpu复用前面"full"层的稀疏top-k。当PP边界把一组
        #   shared层切开时，接收方rank开头的那些shared层需要发送方rank的top-k，所以要把它带过边界。对稠密模型、没有shared层的
        #   稀疏模型、pp=1，或每个rank都从"full"层开始的情况，什么也不做。
        if getattr(self, "_pp_share_indexer_ready", False):
            return
        self._pp_share_indexer_ready = True
        self._pp_send_needs_sparse = False
        self._pp_recv_needs_sparse = False
        self._pp_index_topk = 0
        if not self.has_mla_indexer:
            return
        pp = get_pp_group()
        if pp.world_size <= 1:
            return
        # Unwrap to the module exposing the PP layer range (make_layers sets
        # start_layer/end_layer on the inner model; UBatchWrapper/CausalLM wrap it).
        # 【译】一层层剥开，找到暴露PP层范围的那个模块（make_layers在内层模型上设置start_layer / end_layer；外面包着
        #   UBatchWrapper / CausalLM）。
        inner = self.model
        while not hasattr(inner, "start_layer") and hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "start_layer"):
            return

        # Replicate the model's per-layer shared/full classification
        # (_should_skip_index_topk in deepseek_v2.py).
        # 【译】复刻模型里逐层的shared / full分类（deepseek_v2.py里的_should_skip_index_topk）。
        # 三种配置写法，按优先级：indexer_types逐层列出、index_topk_pattern用"S"标shared、否则按index_topk_freq周期推算
        hf = self.config.hf_config
        num_layers = int(hf.num_hidden_layers)
        indexer_types = getattr(hf, "indexer_types", None)
        index_topk_pattern = getattr(hf, "index_topk_pattern", None)
        index_topk_freq = int(getattr(hf, "index_topk_freq", 1))
        index_skip_topk_offset = int(getattr(hf, "index_skip_topk_offset", 1))

        def _is_shared(layer_idx):
            if not 0 <= layer_idx < num_layers:
                return False
            if indexer_types is not None:
                return indexer_types[layer_idx] == "shared"
            if index_topk_pattern is not None:
                return index_topk_pattern[layer_idx] == "S"
            if index_topk_freq <= 1:
                return False
            return max(layer_idx - index_skip_topk_offset, 0) % index_topk_freq != 0

        # This rank consumes the prior rank's top-k iff its first layer is shared.
        # 【译】当且仅当本rank的第一层是shared时，本rank要用前一个rank的top-k。
        self._pp_recv_needs_sparse = (not pp.is_first_rank) and _is_shared(
            inner.start_layer
        )
        # The next rank consumes this rank's top-k iff ITS first layer
        # (== this rank's end_layer) is shared.
        # 【译】当且仅当下一个rank的第一层（== 本rank的end_layer）是shared时，下一个rank要用本rank的top-k。
        self._pp_send_needs_sparse = (not pp.is_last_rank) and _is_shared(
            inner.end_layer
        )
        # Transfer the physical producer row width, not the logical top-k.
        # GLM-5.3 appends up to index_kpool-1 tail tokens and rounds each row
        # to 128 columns; slicing at index_topk would start every later row at
        # the wrong offset.
        # 【译】传的是生产方的物理行宽，而不是逻辑上的top-k。GLM-5.3会追加最多index_kpool-1个尾部token，并把每行凑整到128列；
        #   按index_topk切的话，后面每一行的起点都会错位。
        self._pp_index_topk = int(
            getattr(
                self.attn_metadata_builder,
                "index_topk_out",
                self.config.hf_config.index_topk,
            )
        )
        if self._pp_recv_needs_sparse or self._pp_send_needs_sparse:
            logger.info(
                "[%s] PP shared-indexer transfer: recv=%s send=%s "
                "(layers [%d,%d), index_topk=%d)",
                self.rank_name,
                self._pp_recv_needs_sparse,
                self._pp_send_needs_sparse,
                inner.start_layer,
                inner.end_layer,
                self._pp_index_topk,
            )

    # _padded_decode_inputs：decode时把模型看到的输入从scheduled_tokens行扩到running_tokens行（eager与PIECEWISE路径用）
    #   为什么要扩：running_bs / running_tokens是本步"实际跑"的宽度——DP下全组要同形状（MoE的all_gather要求各rank行数相同，
    #     见forward_context.py ForwardMode的docstring），PIECEWISE要对上录过的num_tokens档位；注意力元数据已经按running_*
    #     填好（见prepare_inputs例2），模型的激活也得是这个高度
    #   补的行要填合法的id / position（docstring原话），这里统一清成0；它们的slot_mapping=-1、context_len=0，不写也不读KV
    #   例：scheduled=3、running=4 → 返回input_ids.gpu[:4]、positions.gpu[:4]，第4行清0
    #   返回的是forward_vars里那两块固定buffer的切片（不拷贝），清0就是改buffer本身；FULL回放不调用它，graph直接读这两块buffer
    def _padded_decode_inputs(self, forward_mode: ForwardMode):
        """Expose the full decode run, independently of graph replay.

        Metadata and uniform DP collectives already use running_tokens. Keep
        model activations at that height too, with legal ids/positions in the
        unused tail. Mixed prefill/decode uses varlen collectives and must not
        call this helper.
        """
        # 【译】把完整的decode运行宽度暴露出来，与是否回放graph无关。
        #
        #   元数据和统一的DP集合通信已经用的是running_tokens。模型的激活也保持这个高度，没用到的尾部放合法的id / position。
        #   prefill / decode混合时用变长集合通信，不能调用这个函数。
        assert not forward_mode.is_prefill
        assert forward_mode.running_tokens_are_unified
        scheduled = forward_mode.scheduled_tokens
        running = forward_mode.running_tokens
        ids = self.forward_vars["input_ids"].gpu[:running]
        positions = (
            self._mrope_positions_view(running)
            if self.use_mrope
            else self.forward_vars["positions"].gpu[:running]
        )
        assert ids.shape[0] == running and positions.shape[-1] == running
        if running > scheduled:
            ids[scheduled:].zero_()
            positions[..., scheduled:].zero_()
        return ids, positions

    # ============================ run_model：执行一次模型forward ============================
    #   输入：input_ids（scheduled_tokens行，prepare_model的返回值）；batch只用于多模态、PP、动态分块和profiler标签。
    #     其余输入（positions、注意力元数据、Context）都从prepare_inputs设好的ForwardContext里取
    #   输出：(logits, hidden_states)。logits [行数, vocab]：每个TP rank都有完整的vocab（ParallelLMHead里做了all_gather），
    #     prefill时LM head只取每条序列最后一个token（cu_seqlens_q[1:] − 1，共running_bs行）；hidden_states [行数, hidden]
    #     给草稿模型用。PP的非最后stage两者都是None
    # 前置：CUDA Graph回放要求固定的kernel序列、固定的显存地址、固定的形状（config.py里CUDAGraphMode上方【科普】②），
    #   所以只有形状落在录过的档位上的decode才能回放。何时回放只由ForwardMode.decide一处决定（prepare_model里调用，
    #   forward_context.py），这里只看forward_mode.use_cudagraph
    # 分派流程（与下面的代码一一对应）：
    #   forward_mode.use_cudagraph？
    #     │否：prefill；--enforce-eager；后端step_needs_eager；DP组里有rank在prefill；bs超过最大档；P/D producer的形状没录
    #     ├──► eager：decode（全组统一）时先_padded_decode_inputs扩到running_tokens
    #     │      → 多模态prefill先算inputs_embeds（【分支】第2、8批）→ PP非首stage接收上一stage的hidden（【分支】第9批）
    #     │      → self.model(input_ids, positions)（--level 3时跑的仍是torch.compile编译过的代码，只是不走graph）
    #     │      → PP非最后stage：异步发给下一stage，返回(None, None)；chunked prefill的中间块：不算logits
    #     │      → 否则compute_logits
    #     │是
    #     ├──► _piecewise_cg_active()（--cudagraph-mode PIECEWISE / FULL_AND_PIECEWISE / AF_PIECEWISE）：
    #     │      设cudagraph_runtime_mode=PIECEWISE（running_tokens这档录过时；否则NONE，片段也eager）和
    #     │      BatchDescriptor(num_tokens=running_tokens) → self.model(...)：编译图里每个稠密片段的CUDAGraphWrapper
    #     │      按num_tokens找到自己录的图回放，attention在片段之间eager执行（utils/cuda_graph.py）
    #     └──► 否则（FULL；原生下的NONE也走这里）：self.graphs[(running_bs, max_seqlen_q)].replay()，
    #            结果在forward_vars["outputs"]的前scheduled_tokens行；TP=1且不开TBO时连LM head也录在图里（graph_logits）
    #   例（阶梯[1..8, 16, ..., 512]，不开投机解码）：decode bs=65 → running_bs=128 → 回放graphs[(128, 1)]，
    #     65行真实、63行补的；hidden_states只取前65行
    # 对比nano-vllm：nano-vllm的判断是 is_prefill or enforce_eager or bs > 512，回放前把输入拷进graph_vars[:bs]、
    #   slot_mapping / context_lens先fill_(-1) / zero_()；ATOM的输入在prepare阶段就已写进录图时用的那几块buffer（forward_vars），
    #   补行的-1 / 0也由builder写好，回放时不需要任何拷贝；graph的key多了max_seqlen_q（MTP每条序列q=mtp_k+1行）；
    #   PIECEWISE路径是nano-vllm没有的
    @_profile_runner_stage
    @record_gpu_forward
    def run_model(
        self,
        input_ids: torch.Tensor,
        batch: ScheduledBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.scheduled_bs
        is_prefill = context.is_prefill
        positions = context.positions

        # Dispatch is owned by ForwardMode.decide() (called in prepare_model).
        # Every run_model caller MUST go through prepare_inputs first, so
        # forward_mode is always set here.
        # 【译】分派由ForwardMode.decide()负责（在prepare_model里调用）。每个调用run_model的地方都必须先经过prepare_inputs，
        #   所以这里forward_mode一定已经设好。
        forward_mode = context.forward_mode
        assert forward_mode is not None, (
            "context.forward_mode is None; run_model invoked without going "
            "through prepare_model. Add ForwardMode.decide() at the new "
            "entry point instead of re-deriving the 4-OR dispatch here."
        )

        # Single canonical shape check; contract owned by ForwardMode, which
        # short-circuits only on prefill. The padded step is the one it is for.
        # 【译】唯一一处规范的形状检查；契约归ForwardMode所有，只在prefill时直接跳过。它就是为补过行的step准备的。
        # 解释：只比形状不读值（读GPU张量的值要同步）：input_ids是scheduled_tokens行，slot_mapping是running_tokens行，
        #   cu_seqlens_q是running_bs+1行（例2：3、4、5）
        forward_mode.assert_shape_contract(input_ids, forward_context.attn_metadata)

        # Profiler label. Kind (prefix) distinguishes real/dummy and
        # eager/cudagraph; `tbo=1` marks a step that ran TBO ubatches. See
        # `build_run_label`.
        # 【译】profiler标签。前缀区分真实 / dummy以及eager / cudagraph；tbo=1标记这一步跑了TBO微batch。见build_run_label。
        label = build_run_label(
            is_prefill=is_prefill,
            use_cudagraph=forward_mode.use_cudagraph,
            is_dummy=context.is_dummy_run,
            tbo_on=forward_context.ubatch_slices is not None,
            scheduled_bs=bs,
            running_bs=context.running_bs,
            batch=batch,
            detailed_suffix=self._detailed_label_suffix(batch),
        )

        # PCP+TBO prefill: per-group round-robin stripe before UBatchWrapper (see
        # _apply_pcp_balanced_stripe). _pcp_tbo_balanced also gates the per-group
        # output restore further below.
        # 【译】PCP+TBO prefill：进入UBatchWrapper之前按组做轮询分片（见_apply_pcp_balanced_stripe）。_pcp_tbo_balanced
        #   同时控制下面按组还原输出。（【分支】第9批）
        _pcp_size = self.config.prefill_context_parallel_size
        _pcp_bal_groups = getattr(self, "_pcp_bal_groups", None)
        _pcp_tbo_balanced = (
            _pcp_size > 1
            and isinstance(self.model, UBatchWrapper)
            and forward_context.ubatch_slices is not None
            and is_prefill
            and not forward_context.context.is_dummy_run
            and _pcp_bal_groups is not None
        )
        if _pcp_tbo_balanced:
            input_ids, positions = self._apply_pcp_balanced_stripe(
                input_ids, positions, _pcp_bal_groups, _pcp_size, forward_context
            )

        if not forward_mode.use_cudagraph:
            # prefill, or decode forced eager (enforce_eager / DP peer
            # prefill / bs above the largest captured graph).
            # 【译】prefill，或被强制eager的decode（enforce_eager / DP里有对端在prefill / bs超过最大的已录graph）。
            with record_function(label):
                # 全组都在decode时按running_tokens跑（DP下MoE用固定行数的all_gather）；有rank在prefill时各跑各的行数（变长gather）
                if not is_prefill and forward_mode.running_tokens_are_unified:
                    input_ids, positions = self._padded_decode_inputs(forward_mode)
                # The multimodal runtime owns request leases and span scatter.
                # 【译】请求的租约（lease）和span的散写归多模态运行时管。
                # 【分支】多模态prefill（第2b批 multimodal_runtime.py，模型侧第8批）：带embedding_spans的走embed_multimodal_batch
                #   （只有DeepSeek-V4.1产生这种数据）；否则是Qwen-VL式的pixel_values + image_grid_thw：先跑视觉塔，再把视觉embedding
                #   合并进文本embedding，得到inputs_embeds代替input_ids
                inputs_embeds = None
                if (
                    is_prefill
                    and hasattr(self.model, "get_vision_embeddings")
                    and batch is not None
                    and hasattr(batch, "multimodal_data")
                    and batch.multimodal_data
                ):
                    if all(
                        "embedding_spans" in data
                        for data in batch.multimodal_data.values()
                    ):
                        from atom.model_engine.multimodal_runtime import (
                            embed_multimodal_batch,
                        )

                        inputs_embeds = embed_multimodal_batch(
                            self.model,
                            self.vision_embeddings,
                            input_ids,
                            batch,
                            self.device,
                            self.config.torch_dtype,
                        )
                    else:
                        mm_data_values = list(batch.multimodal_data.values())
                        pixel_values = torch.cat(
                            [mm_data["pixel_values"] for mm_data in mm_data_values],
                            dim=0,
                        ).to(device=self.device, dtype=self.config.torch_dtype)
                        grid_thw = torch.cat(
                            [mm_data["image_grid_thw"] for mm_data in mm_data_values],
                            dim=0,
                        ).to(device=self.device)
                        vision_embeds = self.model.get_vision_embeddings(
                            pixel_values, grid_thw
                        )
                        text_embeds = self.model.embed_input_ids(input_ids)
                        inputs_embeds = self.model.merge_multimodal_embeddings(
                            input_ids, text_embeds, vision_embeds
                        )

                # 【分支】PP（第9批 distributed/pp_comm.py）：非首stage先收上一stage发来的hidden_states / residual
                pp_group = get_pp_group()
                pp_enabled = pp_group.world_size > 1
                if pp_enabled:
                    self._setup_pp_shared_indexer()

                intermediate_tensors = None
                if pp_enabled and not pp_group.is_first_rank:
                    intermediate_tensors = recv_intermediate_tensors()
                    # GLM-5.2 IndexShare: load prior rank's top-k for leading
                    # shared layers. Pop so compiled model sees only hidden_states.
                    # 【译】GLM-5.2 IndexShare：为开头的shared层载入前一个rank的top-k。pop掉，让编译后的模型只看到hidden_states。
                    recv_sparse = intermediate_tensors.tensors.pop(
                        "sparse_kv_indices", None
                    )
                    if recv_sparse is not None and self._pp_recv_needs_sparse:
                        tgt = self.attn_metadata_builder._sparse_kv_indices_gpu
                        tgt[: recv_sparse.numel()].copy_(recv_sparse)

                # Time only the chunk computation between PP receive and send.
                # 【译】只给PP接收与发送之间的这段chunk计算计时。
                # 【分支】动态分块：用这段时间拟合prefill的延迟模型（第9批 dynamic_chunking_worker.py；没开时是空操作）
                chunk_sample = self._dynamic_chunking.start_sample(batch)
                # 真正的模型forward（eager：CPU逐个发射kernel）；输出 [行数, hidden]（PP非最后stage是IntermediateTensors）
                if pp_enabled:
                    model_output = self.model(
                        input_ids,
                        positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                    )
                elif inputs_embeds is None:
                    model_output = self.model(input_ids, positions)
                else:
                    model_output = self.model(
                        input_ids, positions, inputs_embeds=inputs_embeds
                    )
                self._dynamic_chunking.finish_sample(chunk_sample)
                if pp_enabled and not pp_group.is_last_rank:
                    # GLM-5.2 IndexShare: carry top-k for next rank's shared layers.
                    # 【译】GLM-5.2 IndexShare：把top-k带给下一个rank的shared层。
                    if self._pp_send_needs_sparse:
                        # Use hidden_states rows (correct under PCP shard).
                        # 【译】用hidden_states的行数（在PCP分片下也正确）。
                        num_tokens = model_output.tensors["hidden_states"].shape[0]
                        n = num_tokens * self._pp_index_topk
                        model_output.tensors["sparse_kv_indices"] = (
                            self.attn_metadata_builder._sparse_kv_indices_gpu[:n]
                        )
                    # 异步发送（isend）；发之前先等上一次的发送完成（commit_pp_send_work），同一时刻只有一次在途
                    if self._pp_pending_send:
                        commit_pp_send_work(self._pp_pending_send)
                    self._pp_pending_send = async_send_intermediate_tensors(
                        model_output
                    )
                    hidden_states = None
                    logits = None
                elif self._is_pure_middle_chunk(batch):
                    if _pcp_tbo_balanced:
                        model_output = self._restore_pcp_balanced_output(
                            model_output, _pcp_bal_groups, _pcp_size
                        )
                    # Middle chunk: no logits, but drafter needs hidden states.
                    # 【译】中间块：不算logits，但草稿模型需要hidden states。
                    hidden_states = model_output
                    logits = None
                else:
                    if _pcp_tbo_balanced:
                        model_output = self._restore_pcp_balanced_output(
                            model_output, _pcp_bal_groups, _pcp_size
                        )
                    # PP carries the full run between stages. Only the last
                    # stage returns scheduled rows to sampling/draft consumers.
                    # 【译】PP在stage之间传的是完整的运行宽度。只有最后一个stage把scheduled的那些行交给采样 / 草稿的使用方。
                    # 解释：decode切回scheduled_tokens行（去掉补的行）；prefill不切——LM head自己按cu_seqlens_q挑每条序列最后一个token
                    hidden_states = (
                        model_output[: context.scheduled_tokens]
                        if not is_prefill
                        else model_output
                    )
                    logits = self.model.compute_logits(hidden_states)
        else:
            # decode[bs=128 tok=128 d=128] / decode[... p=2 d=126 spec=3] /
            # dummy_decode[...] — see build_run_label.
            # 【译】标签形如 decode[bs=128 tok=128 d=128] / decode[... p=2 d=126 spec=3] / dummy_decode[...]——见build_run_label。
            with record_function(label):
                running_bs = context.running_bs
                running_tokens = forward_mode.running_tokens
                scheduled_tokens = context.scheduled_tokens

                # PIECEWISE类：不调手动整图，照常调用编译后的模型；每个稠密片段的CUDAGraphWrapper读forward_context里的
                #   cudagraph_runtime_mode和batch_descriptor决定回放哪张图（utils/cuda_graph.py CUDAGraphWrapper.__call__）
                if self._piecewise_cg_active():
                    _ids, _pos = self._padded_decode_inputs(forward_mode)
                    forward_context.cudagraph_runtime_mode = (
                        CUDAGraphMode.PIECEWISE
                        if forward_mode.piecewise_captured
                        else CUDAGraphMode.NONE
                    )
                    forward_context.batch_descriptor = BatchDescriptor(
                        num_tokens=running_tokens
                    )
                    model_output = self.model(_ids, _pos)
                    # 用完恢复成NONE：CUDAGraphWrapper只在模式为PIECEWISE时才录 / 回放，NONE时直接执行原函数
                    forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    forward_context.batch_descriptor = None
                    # model_output is always a plain Tensor; drafter aux capture
                    # (if any) already wrote its own buffers inside the forward.
                    # Spec keeps the padded layout (postprocess/draft re-gather to
                    # bs via next_token_locs); non-spec cuts to the scheduled rows
                    # so pad rows never leak into sampled_token_ids ->
                    # prev_token_ids -> next-step shape mismatch.
                    # 【译】model_output总是一个普通Tensor；草稿模型的辅助hidden捕获（如果有）已经在forward里写进它自己的buffer。
                    #   投机解码保留补过行的布局（postprocess / 草稿再通过next_token_locs收回到bs行）；非投机解码则切到scheduled的行，
                    #   免得补的行漏进sampled_token_ids → prev_token_ids → 造成下一步的形状不匹配。
                    hidden_states = model_output[
                        : (
                            running_tokens
                            if hasattr(self, "drafter")
                            else scheduled_tokens
                        )
                    ]
                    logits = self.model.compute_logits(hidden_states)
                    return logits, hidden_states

                # FULL回放：graph按(running_bs, max_seqlen_q)存（capture_cudagraph录的key）。replay()只是一条命令，GPU照录好的kernel
                #   序列执行，读的是forward_vars里那些固定buffer——prepare阶段已经把本步的数据写进去了
                graph_key = (running_bs, forward_context.attn_metadata.max_seqlen_q)
                self.graphs[graph_key].replay()
                # graph把结果写进outputs的前running_tokens行（录图时outputs[:num_tokens] = model_output也录了进去），只取真实的前scheduled_tokens行
                hidden_states = self.forward_vars["outputs"][:scheduled_tokens]
                # Drafter aux buffers (if any) refresh on replay: their in-place
                # copy ops were captured into the graph.
                # 【译】草稿模型的辅助buffer（如果有）在回放时自动刷新：往它们里写的原地拷贝操作已经录进了graph。
                # logits_in_graph（TP=1且不开TBO，见capture_cudagraph）：LM head也在图里，直接取录图时固定下来的logits buffer
                if self.logits_in_graph:
                    logits = self.graph_logits[graph_key][:scheduled_tokens]
                else:
                    logits = self.model.compute_logits(hidden_states)

        return logits, hidden_states

    def flush_pp_send(self) -> bool:
        """Flush pending PP isend. Returns True for call_func wait_out."""
        # 【译】把挂起的PP isend全部冲刷完。返回True，供call_func的wait_out使用。
        # 【分支】PP（第9批）：返回True而不是None，因为AsyncIOProc里返回None的方法不回包，wait_out=True的调用会一直等
        #   （async_proc.py）。调用方：pp_engine_core.py（head空闲时、退出时）、dynamic_chunking_worker.py
        if self._pp_pending_send:
            commit_pp_send_work(self._pp_pending_send)
        return True

    # ============================ postprocess：采样 → 验证草稿 → 异步回传 ============================
    #   输入：logits（run_model的输出，每个TP rank都有完整vocab）、prepare_sample给的采样参数、hidden_states（给草稿模型）
    #   输出：ScheduledBatchOutput。只有TP rank 0的这份会回到EngineCore，其余rank的返回值被丢弃（async_proc.py）；
    #     开延迟输出时，其中的req_ids / token_ids属于*上一步*的batch（is_deferred_out=True）
    # 流程：
    #   ① 采样
    #      不投机：logits[:bs] → Sampler → sampled_tokens [bs]
    #        （先切到bs：prefill时LM head按running_bs输出行，补的行在这里去掉。sampler.py：全贪心时argmax；
    #          不过滤时Gumbel-max温度采样；有top-k / top-p时走过滤采样）
    #      投机解码的验证步（MTP / EAGLE3 / DSpark，【分支】第7批）：
    #        按spec_decode_metadata从logits里挑出target行（验证草稿）和bonus行（每条序列最后一行）
    #        → RejectionSampler从左到右逐个比对草稿，遇到第一个不接受的就停 → sampled_tokens [bs, mtp_k+1]（不足处填-1）
    #          和verdict（接受了几个、拒绝了几个、最后一个输出token在哪一行）
    #        例（mtp_k=3，某条序列的输入行 = [t, d1, d2, d3]，第i行的logits预测第i+1个位置）：
    #          d1、d2被接受，d3被拒 → 输出[d1, d2, x, -1]，x是第2行（d2之后）由目标模型给出的token；
    #          verdict：接受数2（= next_token_locs，下一步的锚点在第2列）、拒绝数1、锚点行 = 这条序列的起始行 + 2
    #   ② TP对齐：开延迟输出或有草稿模型时，把rank 0的sampled_tokens广播给所有rank
    #      ——每个rank都各自采样了，随机采样的结果可能不同；而下一步的input_ids是每个rank从自己GPU上的prev_token_ids取的，
    #        草稿模型的输入也一样，必须一致，否则各rank的forward就分叉了（SPMD，engine_core.py文件开头【4】）
    #   ③ logprobs（有请求要时）：log_softmax后取被采样token的那一项
    #   ④ D2H回传（tokenIDProcessor.prepare_sampled_ids，延迟输出的实现见本文件上方tokenIDProcessor）：
    #      forward_done_event.record()在计算stream上打一个"采样算完"的标记 →
    #        延迟输出：先等*上一步*的拷贝完成并取出它的token；再把本步的sampled_tokens排到async_copy_stream上异步拷回
    #          （拷贝等这个event）→ 本步的token下一次forward才取
    #        不延迟（PP、要交接递归状态的PD producer）：直接.tolist()，CPU在这里等GPU把本步算完
    #   ⑤ 草稿（有drafter时）：取每条序列最后被接受的token作为锚点，调propose_draft_token_ids生成下一步的草稿
    #   ⑥ 组装ScheduledBatchOutput：req_ids、token_ids、draft_token_ids、num_rejected / num_bonus（同样是上一步的）、logprobs
    # 对比nano-vllm：nano-vllm只在rank 0上 sampler(logits).tolist()，同步拷回、立刻返回本步的token；
    #   ATOM每个rank都采样再广播，拷回是异步的、晚一步交付，另外多了投机解码的验证和logprobs
    @_profile_runner_stage
    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
        all_greedy: bool,
        # following for draft
        # 【译】以下参数供草稿模型用
        hidden_states: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> ScheduledBatchOutput:
        spec_decode_metadata = get_forward_context().spec_decode_metadata
        bs = batch.total_seqs_num
        if spec_decode_metadata is None:
            # The LM head emitted one row per sequence the step FORWARDED,
            # which prefill pads to `running_bs` for the draft pass that
            # follows it. Cut to the scheduled batch here and nowhere else:
            # this is the boundary where a padded forward becomes a
            # per-request result, and everything below counts requests -- the
            # sampler's per-row parameters and the logprob gather both. Cut
            # after either and the pad rows divide `[running_bs, V]` by
            # `[scheduled_bs, 1]`.
            # 【译】LM head为本步forward过的每条序列输出一行，prefill为紧随其后的草稿pass把它补到了running_bs行。只在这里切回
            #   scheduled的batch、别处都不切：这里是"补过行的forward"变成"每请求结果"的边界，下面的一切都按请求计数——
            #   sampler的逐行参数和logprob的gather都是。在它们之后再切，补的行就会让[running_bs, V]去除以[scheduled_bs, 1]。
            logits = logits[:bs]
            sampled_tokens = self.sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise=needs_independent_noise,
            )
            # 不投机时：每条序列拒绝0个、下一步的锚点在第0列（一个全0的[bs]张量，启动时分配好）
            num_reject_tokens = self.tokenID_processor.default_num_rejected_tokens[:bs]
            next_token_locs = num_reject_tokens
            # No drafts scored -> no verdict; the drafter anchors on each
            # segment's last row.
            # 【译】没有给草稿打分 → 没有verdict；草稿模型以每段的最后一行为锚点。
            anchors = None
        else:
            assert logits is not None
            bonus_logits_indices = spec_decode_metadata.bonus_logits_indices
            target_logits_indices = spec_decode_metadata.target_logits_indices

            # 【分支】投机解码的验证（第7批 rejection_sampler.py）：bonus行 [bs, V]，target行 [草稿总数, V]
            bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
            target_logits = torch.index_select(logits, 0, target_logits_indices)
            target_token_ids = None
            # 随机采样时：每个target行用独立噪声各采一个token，草稿与之相同才接受（sampler.py sample_verification_tokens）；
            #   采样结果在TP各rank间以rank 0为准，接受个数才能一致
            if not all_greedy:
                target_token_ids = self.sampler.sample_verification_tokens(
                    target_logits,
                    spec_decode_metadata.cu_num_draft_tokens,
                    temperatures,
                    top_ks,
                    top_ps,
                )
                if target_token_ids.numel() and get_tp_group().world_size > 1:
                    target_token_ids = get_tp_group().broadcast(target_token_ids, src=0)
            bonus_token_ids = self.sampler(
                logits=bonus_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                all_greedy=all_greedy,
                needs_independent_noise=needs_independent_noise or not all_greedy,
            )
            sampled_tokens, verdict = self.rejection_sampler.forward(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                target_token_ids=target_token_ids,
            )
            # PCP ranks decode redundantly and are consistent only while their
            # kernels agree bit-for-bit -- they don't (hidden differs by ~1 bf16
            # ULP, flipping ~24% of the near-tie verify argmaxes). Accept counts
            # then differ per rank and the emitted streams fork. Sync the
            # decision instead: the ids and the verdict.
            # 【译】PCP的各rank冗余地做decode，只有当它们的kernel逐位一致时结果才一致——实际并不一致（hidden相差约1个bf16 ULP，
            #   接近平局的验证argmax中约24%会翻转）。于是各rank接受的个数不同，输出的token流就分叉了。所以改为同步决定本身：
            #   token id和verdict。
            if get_pcp_world_size() > 1 and hasattr(self, "drafter"):
                _g = get_pcp_group()
                sampled_tokens = _g.broadcast(sampled_tokens.contiguous(), src=0)
                verdict = _g.broadcast(verdict, src=0)
            num_bonus_tokens, num_reject_tokens, anchors = verdict
            next_token_locs = num_bonus_tokens

        # Drafter input must agree across TP ranks.
        # 【译】草稿模型的输入在各TP rank之间必须一致。
        # 解释：开延迟输出时下一步的input_ids也取自各rank自己GPU上的sampled_tokens，所以同样要一致（见上面流程②）
        if get_tp_group().world_size > 1 and (
            self.tokenID_processor.is_deferred_out or hasattr(self, "drafter")
        ):
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        # Compute logprobs if any sequence requested them
        # 【译】只要有序列要logprobs就计算
        # 例：logits [bs, V] → float → log_softmax → 每行取被采样token那一列 → [bs]；只算被采样token的logprob，没有top-n候选
        #   （投机解码时不支持logprobs：sequence.py构造Sequence时就拒绝，所以这里只会遇到[bs, V]）
        need_logprobs = any(batch.return_logprobs)
        sampled_logprobs = None
        if need_logprobs:
            logits_fp32 = logits.float()
            log_probs = torch.log_softmax(logits_fp32, dim=-1)
            sampled_logprobs = log_probs.gather(
                -1, sampled_tokens.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
                sampled_logprobs = get_tp_group().broadcast(sampled_logprobs, src=0)

        # ④ 在当前（计算）stream上记录"本步采样已算完"的event：async_copy_stream上的D2H拷贝等它，CPU不用等
        self.forward_done_event.record()
        # Capture before prepare_sampled_ids(), which advances self.prev_batch to current batch.
        # 【译】在prepare_sampled_ids()之前取出来，因为它会把self.prev_batch推进成当前batch。
        prev_batch = self.tokenID_processor.prev_batch
        token_id_dict, logprobs_map = self.tokenID_processor.prepare_sampled_ids(
            batch, sampled_tokens, self.forward_done_event, sampled_logprobs
        )
        # Extract req_ids and token_ids from dict (key -1 is the is_deferred_out flag)
        # 【译】从dict里取出req_ids和token_ids（key -1是is_deferred_out标志）
        # 例（延迟输出，接prepare_inputs的例2）：本步是decode [A, B, C]，返回的是例1那一步的结果 → req_ids_out = [A, B]、
        #   token_ids_out = [(tA,), (tB,)]；投机解码时每个元组里是本步接受的若干个token（-1之后截掉）
        req_ids_out = [k for k in token_id_dict if k != -1]
        token_ids_out = [token_id_dict[k] for k in req_ids_out]

        # ⑤ 草稿与接受数：开延迟输出时num_rejected / num_bonus也是上一步的（与token_ids对齐），本步的经异步D2H下一步才取
        draft_token_ids: np.ndarray | None = None
        if self.tokenID_processor.is_deferred_out:
            if hasattr(self, "drafter"):
                prev_rejected_num = self.tokenID_processor.prev_rejected_num
                prev_bonus_num = self.tokenID_processor.prev_bonus_num
                # 【译】（行尾注释）异步拷回CPU
                # 下面：每条序列取最后一个被接受的token（sampled_tokens第next_token_locs列）作为下一步的锚点，覆盖prev_token_ids；
                #   下一步的fill_deferred_decode_ids用它和本步的草稿拼出input_ids
                self.tokenID_processor.send_mtp_status_to_cpu_async(
                    num_reject_tokens, next_token_locs, self.forward_done_event
                )  # Async copy to CPU
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                self.tokenID_processor.prev_token_ids = next_token_ids
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                    anchors,
                )

            elif prev_batch is not None:
                prev_rejected_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
                prev_bonus_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
            else:
                # First forward pass: no deferred output yet, req_ids_out is empty
                # 【译】第一次forward：还没有延迟的输出，req_ids_out为空
                prev_rejected_num = np.zeros(0, dtype=np.int32)
                prev_bonus_num = np.zeros(0, dtype=np.int32)
        else:
            prev_rejected_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            prev_bonus_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            # PP stages and P/D producers still run the drafter.
            # 【译】PP的各stage和P/D producer仍然要跑草稿模型。
            if hasattr(self, "drafter"):
                # Mid-prompt sequences get their anchor corrected inside
                # propose_draft_token_ids, from `batch.next_token_ids`.
                # 【译】还在prompt中间的序列，其锚点在propose_draft_token_ids里用batch.next_token_ids纠正。
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                    anchors,
                )

        # DSpark Phase 2: carry this step's per-request ell back to the scheduler
        # as a {req_id: ell} dict (req_id-keyed avoids any output/draft batch
        # ordering ambiguity). The worker already fired this map in propose() via
        # verify_scheduler.record_ell(batch.req_ids).
        # 【译】DSpark第2阶段：把本步每个请求的ell以{req_id: ell}字典的形式带回调度器（按req_id作key，避免输出batch与草稿batch
        #   之间的顺序歧义）。worker已经在propose()里通过verify_scheduler.record_ell(batch.req_ids)发出了这个映射。
        # 解释：ell = 置信度调度为每个请求选的验证长度（第7批 spec_decode/dspark_scheduler.py）
        dspark_ell = None
        drafter = getattr(self, "drafter", None)
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            dspark_ell = verify_scheduler.ell_nonblocking()

        return ScheduledBatchOutput(
            req_ids=req_ids_out,
            token_ids=token_ids_out,
            draft_token_ids=draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=prev_rejected_num,
            num_bonus=prev_bonus_num,
            logprobs=logprobs_map,
            dspark_ell=dspark_ell,
        )

    def _record_kv_cache_ready(self, batch: ScheduledBatch) -> None:
        """Publish a GPU event for final prefill chunks to transfer connectors."""
        # 【译】为最后一个prefill chunk向传输connector发布一个GPU event。
        # 【分支】PD分离 / KV offload（第9批）：KV要等GPU真正写完才能传。connector（如mooncake的record_kv_cache_ready）在当前stream上
        #   record一个event，传输线程等这个event再读KV；只报is_final_chunk的序列（chunked prefill最后一块写完，整条prompt的KV才齐）
        if batch.total_seqs_num_prefill <= 0:
            return
        if batch.is_final_chunk is None:
            req_ids = batch.req_ids
        else:
            req_ids = [
                req_id
                for req_id, is_final in zip(
                    batch.req_ids, batch.is_final_chunk, strict=True
                )
                if is_final
            ]
        if not req_ids:
            return
        connector = get_kvconnector()
        callback = getattr(connector, "record_kv_cache_ready", None)
        if callable(callback):
            callback(req_ids)

    # ============================ forward：EngineCore每一步调用的入口 ============================
    # 调用链：EngineCore._process_engine_step_inner → runner_mgr.call_func("forward", batch, wait_out=True)
    #   → 共享内存广播 → 每个TP rank的ModelRunner.forward(batch)（同一个batch、同样的forward，SPMD，engine_core.py文件开头【4】）
    #   → 只有rank 0的返回值回到EngineCore → scheduler.postprocess
    #   装饰器：inference_mode关闭autograd记录；with_eplb_forward_monitor在开EPLB时统计专家负载（【分支】第7批）；
    #     _profile_runner_stage只在torch profiler开着时打标签
    # 一步的GPU侧全流程（与下面的代码一一对应）：
    #   ① _advance_forward_vars / _gate_staging_reuse：选一组暂存buffer（PP时轮换），并确认上一步从这些buffer发出的H2D
    #      已经执行完——否则CPU这次写.np会和上一步还没做完的DMA抢同一块锁页内存
    #   ② prepare_model：ForwardMode.decide（形状、是否回放graph、DP同步）→ 采样参数 → cu_seqlens_q、input_ids
    #      → prepare_inputs（注意力元数据、ForwardContext）
    #   ③ _mark_staging_h2d_enqueued：本步的H2D都已排进stream，关闭暂存窗口
    #   ④ run_model：eager / PIECEWISE / FULL回放 → logits、hidden_states
    #   ⑤ 草稿模型吸收本次forward的上下文（compute_draft_kv，【分支】投机解码第7批）
    #   ⑥ PP的非最后stage，或batch全是chunked prefill的中间块：没有token要采样，直接返回空输出（跳过postprocess）
    #   ⑦ postprocess：采样 / 验证 → 异步D2H → ScheduledBatchOutput
    #   ⑧ reset_forward_context；记录本组buffer的完成event；告诉KV connector哪些prefill的KV已写完（【分支】第9批）
    #
    # 延迟输出下的时间线（默认开启，条件见tokenIDProcessor.__init__；原理见engine_core.py文件开头【3】）：
    #   CPU（rank 0）：
    #     ─[forward(N)：①②准备N | ④发射N的kernel | ⑦发射采样N，等N−1的D2H拷完，排N的D2H，返回N−1的token]
    #     ─[EngineCore：postprocess(N−1) | schedule(N+1)]
    #     ─[forward(N+1)：①②准备N+1 | ④发射N+1 | ⑦等N的D2H拷完 ...]
    #   GPU计算stream：  ═══ N−1 ═══][═══════════════ forward N ═══════════════][═══════════════ forward N+1 ════
    #   async_copy_stream：          [D2H N−1]                                  [D2H N]
    #   被GPU掩盖的CPU时间：postprocess(N−1)、schedule(N+1)，以及forward(N+1)的准备阶段（ForwardMode.decide、注意力元数据、
    #     H2D排队、kernel发射）——它们都发生在GPU执行第N步的期间。之所以能提前准备，是因为第N+1步的input_ids直接从GPU上的
    #     prev_token_ids取（fill_deferred_decode_ids），不需要CPU先拿到第N步的token，GPU队列里始终有活
    #   CPU在主路径上等GPU的地方：⑦里等上一步的D2H（event.synchronize，这时第N步还在GPU上跑）；①的闸门等上一步的H2D
    #     （decode稳态下早已完成）；投机解码时还要取上一步的草稿和接受数（同样是上一步、早已在拷的）。不延迟时则在⑦等本步的.tolist()
    #   量级（engine_core.py【3】）：decode一步GPU 10ms量级、CPU的post + sched + 准备几ms → 不延迟时这几ms里GPU空转，延迟后被盖住
    # 对比nano-vllm：nano-vllm的run() = prepare → run_model → sampler → .tolist() → reset_context，一步完全同步；
    #   ATOM多了buffer轮换与H2D发布协议（①③）、DP形状同步（②）、投机解码草稿（⑤）、PP与chunked prefill中间块的提前返回（⑥）、
    #   延迟输出（⑦）
    @_profile_runner_stage
    @torch.inference_mode()
    @with_eplb_forward_monitor
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        # Make this forward's staging buffers safe to overwrite before
        # prepare_inputs writes them: rotate to a free slot if there is a ring,
        # otherwise wait out the previous forward's copies.
        # Dummy forwards use and asynchronously upload the same staging
        # buffers. Excluding them here leaves no event between a dummy and the
        # following real forward, allowing that real forward's CPU writes to
        # race the dummy's still-pending H2D copies.
        # 【译】在prepare_inputs写入之前，让本次forward的暂存buffer可以安全覆盖：有环（PP）就轮换到一个空闲的槽，否则等上一次
        #   forward的拷贝做完。dummy forward也使用并异步上传同一批暂存buffer。若在这里把它们排除，dummy和紧随其后的真实forward
        #   之间就没有event，真实forward的CPU写入会与dummy尚未完成的H2D拷贝竞争。
        self._advance_forward_vars()
        self._gate_staging_reuse()
        (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        ) = self.prepare_model(batch)
        self._mark_staging_h2d_enqueued()
        # ④ forward：此后GPU在跑，CPU只要不碰同步点就可以继续往下走
        logits, hidden_states = self.run_model(input_ids, batch)

        pp_group = get_pp_group()
        pp_non_last = pp_group.world_size > 1 and not pp_group.is_last_rank

        drafter = getattr(self, "drafter", None)
        # An output-less batch still runs propose() for its DP collectives.
        # 【译】不产出token的batch仍要为了其中的DP集合通信跑propose()。
        # 解释：DP attention下各rank的草稿pass里有集合通信，一个rank少跑一次，其他rank就会卡住（同【4】的SPMD道理）
        will_align_draft = (
            self._dp_draft_lockstep_active()
            and self._is_pure_middle_chunk(batch)
            and not batch.is_dummy_run
        )
        # Runs after EVERY target forward -- `postprocess` (hence propose) is
        # skipped for a middle chunk. Not on the aligning step: that pass is one
        # the peers never mirror.
        # 【译】每次目标模型forward之后都运行——中间块会跳过postprocess（因而也跳过propose）。对齐步上不运行：那一遍是对端
        #   从不镜像执行的。
        run_compute_draft_kv = (
            drafter is not None
            and not pp_non_last
            and not batch.is_dummy_run
            and not (will_align_draft and drafter.draft_kv_duplicates_propose)
        )
        if run_compute_draft_kv:
            drafter.compute_draft_kv(
                get_forward_context().context.positions,
                hidden_states,
                batch.next_token_ids,
            )
        if pp_non_last or self._is_pure_middle_chunk(batch):
            # This return skips `postprocess`, hence propose() and the DP
            # collectives it carries. Run it for those and drop the ids.
            # 【译】这个return跳过了postprocess，也就跳过了propose()和它携带的DP集合通信。为了这些通信跑一遍，然后丢掉生成的id。
            if will_align_draft:
                self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    torch.zeros(
                        batch.total_seqs_num, dtype=torch.int32, device=self.device
                    ),
                    torch.zeros(
                        batch.total_seqs_num, dtype=torch.int32, device=self.device
                    ),
                    # 【译】（行尾注释）没有做验证 → 用每段的最后一行
                    None,  # nothing verified -> segment's last row
                    align_only=True,
                )
            reset_forward_context()
            # Mark this slot's GPU work (attention consumed its metadata) done.
            # 【译】标记本槽位的GPU工作（注意力已消费完它的元数据）完成。
            self._record_forward_vars_event()
            self._record_kv_cache_ready(batch)
            # 空输出：token_ids为空（中间块没有采样出token；调度器侧怎么处理见scheduler.py ScheduledBatchOutput上方的说明）
            return ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            )

        # ⑦ 采样、验证、异步D2H（开延迟输出时这里返回的是上一步的token）
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )

        # ⑧ 清掉本步的ForwardContext（下一步重新set）；记录本组暂存buffer的完成event（PP环）；通知KV connector
        reset_forward_context()
        self._record_forward_vars_event()
        self._record_kv_cache_ready(batch)
        return fwd_output

    # batch里全是chunked prefill的中间块（没有序列到达prompt末尾）→ 本步不采样（ScheduledBatch.produces_output，第3批）
    @staticmethod
    def _is_pure_middle_chunk(batch) -> bool:
        return batch is not None and not batch.produces_output()

    def _dp_draft_lockstep_active(self) -> bool:
        """Are this rank's draft passes bound to what the DP peers run?

        Only under DP attention -- `_publish_draft_shape` returns early at
        `data_parallel_size <= 1`, where an output-less batch legitimately
        drafts nothing. PP is excluded, and asked about directly:
        `is_deferred_out` is also off on a P/D producer, a rank that keeps its
        drafter loaded and still owes its peers the collectives propose()
        carries.
        """
        # 【译】本rank的草稿pass是否必须与DP对端执行的保持一致？
        #
        #   只在DP attention下——data_parallel_size <= 1时_publish_draft_shape会提前返回，那时不产出token的batch本来就
        #   什么都不用起草。PP被排除在外，而且是直接问PP本身：is_deferred_out在P/D producer上也是关的，而这样的rank仍然加载着
        #   草稿模型，仍欠着对端propose()所携带的集合通信。
        return (
            hasattr(self, "drafter")
            and self.config.parallel_config.data_parallel_size > 1
            and not self.tokenID_processor.is_pipeline_parallel
        )

    def _producer_graph_shapes(self):
        """The recorded `(bs, q)` keys, for the one runner allowed to miss them.

        A P/D producer keeps its drafter, so capture records q=mtp_k+1, but
        every decode it runs is q=1 -- on every rank, its DP dummy included --
        so the lookup comes out the same on each and the group goes eager
        together. Anywhere else a missing key is a capture gap, and it stays a
        `KeyError` at replay.
        """
        # 【译】已录制的(bs, q) key，只提供给唯一一种允许查不到的runner。
        #
        #   P/D producer保留着草稿模型，所以捕获时录的是q=mtp_k+1，但它跑的每次decode都是q=1——每个rank上都是，包括DP dummy——
        #   所以每个rank查到的结果相同，整个组一起走eager。其他任何地方查不到key都说明捕获有缺口，回放时照样抛KeyError。
        # 【分支】PD分离（第9批）；返回值传给ForwardMode.decide的graph_shapes，其余情况返回None（不检查）
        if (
            not self.tokenID_processor.is_pd_producer
            or not hasattr(self, "drafter")
            or self.enforce_eager
            or self._piecewise_cg_active()
            or not hasattr(self, "graphs")
        ):
            return None
        return self.graphs.keys()

    # ---------------- 【分支】KV connector钩子（PD分离 / KV offload），见第9批 ----------------
    # KV connector：在实例之间、或GPU与CPU / 存储之间搬KV的插件（mooncake、LMCache等；PD分离见arg_utils.py kv_transfer_config上方【科普】）
    # EngineCore一步里的调用顺序（engine_core.py _process_engine_step_inner）：
    #   process_kvconnector_output（forward之前，不等待）→ forward（其中的_record_kv_cache_ready）→ async_proc_aggregation（forward之后轮询）
    @torch.inference_mode()
    def process_kvconnector_output(self, connector_meta_output):
        """Dispatch KV connector metadata to initiate async KV loading."""
        # 【译】分发KV connector元数据，启动异步的KV加载。
        # 调用方：EngineCore用call_func(...)且不等结果；所以这里返回None——AsyncIOProc里不等结果的调用必须返回None（async_proc.py）
        if connector_meta_output is not None:
            connector = get_kvconnector()
            if connector is not None:
                connector.start_load_kv(connector_meta_output)

    @torch.inference_mode()
    def async_proc_aggregation(self) -> KVConnectorOutput:
        """Collect finished send/recv status from the KV connector."""
        # 【译】从KV connector收集已完成的发送 / 接收状态。
        # 调用方：EngineCore._poll_kv_transfer_progress经call_func_with_aggregation（每个rank都回包，走单独的KV通道，async_proc.py）
        connector = get_kvconnector()
        if connector is None:
            return KVConnectorOutput()

        finished = connector.get_finished()
        # New connectors may return the full KVConnectorOutput so they can
        # report richer states. LMCache offload uses failed_recving to wake a
        # request for local recompute, and finished_saving to release blocks
        # whose free was deferred while a background save read their KV.
        # 【译】新的connector可以直接返回完整的KVConnectorOutput，以便报告更丰富的状态。LMCache offload用failed_recving唤醒一个
        #   请求去本地重算，用finished_saving释放那些因为后台保存正在读它们的KV而推迟释放的block。
        if isinstance(finished, KVConnectorOutput):
            return finished

        # Legacy P/D connectors still return the old
        # (done_sending, done_recving) tuple. Normalize it so EngineCore and
        # Scheduler only need to consume KVConnectorOutput.
        # 【译】旧的P/D connector仍返回老的(done_sending, done_recving)元组。在这里统一成KVConnectorOutput，让EngineCore和
        #   Scheduler只需要处理这一种。
        done_sending, done_recving = finished

        return KVConnectorOutput(
            finished_sending=done_sending, finished_recving=done_recving
        )

    # 【分支】投机解码：用草稿模型为下一步生成草稿token（MTP / EAGLE3 / DSpark，见第7批 spec_decode/）
    #   调用方：postprocess（采样 / 验证之后）；forward里batch全是中间块且DP需要对齐时（align_only=True，只为跑集合通信）
    #   输入：input_ids = tokenIDProcessor.input_ids.gpu[1 : total_tokens+1]——本步的输入整体左移一位，即每个位置的"下一个token"
    #     （草稿模型把 位置t+1的token 与 目标模型在t处的hidden_state 配对，锚点处再换成next_token_ids，eagle_proposer.py propose）；
    #     next_token_ids：每条序列最后被接受的token（锚点）；num_reject_tokens：要回收的KV行数；anchors：锚点所在的扁平行号
    #   输出：tokenIDProcessor.prepare_draft_ids的结果——开延迟输出时是*上一步*的草稿（numpy，经异步D2H），本步的草稿留在GPU上，
    #     下一步由fill_deferred_decode_ids直接填进input_ids
    def propose_draft_token_ids(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        # num_reject_tokens: KV rows to release, against the `mtp_k` RESERVATION.
        # anchors: each request's flat row of its last emitted token, from the
        # rejection verdict; None when nothing was verified.
        # 【译】num_reject_tokens：要释放的KV行数，相对于mtp_k的"预留"而言。anchors：每个请求最后一个输出token所在的扁平行号，
        #   来自拒绝采样的verdict；没有做验证时为None。
        num_reject_tokens: torch.Tensor,
        anchors: torch.Tensor | None,
        align_only: bool = False,
    ):
        """`align_only` runs the draft purely for its DP collectives.

        Its caller is the all-middle-chunk batch, where every seq is mid-prompt:
        the `batch.next_token_ids` override below replaces `next_token_ids`
        wholesale, so the zeros it passes are never read. The ids are dropped --
        the scheduler is not expecting a draft for a seq that produced no token.
        """
        # 【译】align_only为True时，跑草稿纯粹是为了其中的DP集合通信。
        #
        #   它的调用方是"全是中间块"的batch，其中每条序列都还在prompt中间：下面用batch.next_token_ids做的覆盖会整体替换
        #   next_token_ids，所以调用方传进来的0永远不会被读到。生成的id被丢弃——调度器不会为一条没产出token的序列等待草稿。
        forward_context = get_forward_context()

        # A sequence still mid-prompt samples nothing usable, so its anchor is
        # the scheduler's successor token instead. Per SEQUENCE, not per batch:
        # a middle chunk can sit beside one on its final chunk.
        # 【译】还在prompt中间的序列采样不出有用的东西，所以它的锚点改用调度器给的后继token。按序列而不是按batch：
        #   一个中间块可能和一个最后块在同一个batch里。
        nxt = batch.next_token_ids
        if nxt is not None:
            assert len(nxt) == next_token_ids.shape[0], (
                f"{len(nxt)} scheduler anchors != {next_token_ids.shape[0]} "
                "sampled -- they are matched positionally"
            )
            # -1 marks "sampling supplies it", so keep the sampled value there.
            # 【译】-1表示"由采样提供"，所以那些位置保留采样值。
            # 例：next_token_ids=[7, 9]、override=[-1, 42]（第2条还在prompt中间，下一个prompt token是42）→ [7, 42]
            override = forward_context.context.draft_anchor_overrides
            assert override is not None
            next_token_ids = torch.where(
                override >= 0, override.to(next_token_ids.dtype), next_token_ids
            )

        positions = forward_context.context.positions

        assert isinstance(self.drafter, Drafter)

        # 锚点行：没做验证时取每段最后一行（cu_seqlens_q[1:] − 1），做了验证时用verdict给的行号；
        #   commit_speculative_state：递归状态的后端在这里只保留被接受的前缀（第8批），分页KV的后端什么都不用做
        #   （被拒绝的行靠序列长度隐藏）
        last_token_indices = (
            self.drafter.prepare_inputs(batch.total_seqs_num)
            if anchors is None
            else anchors
        )
        self.attn_metadata_builder.commit_speculative_state(
            forward_context.attn_metadata, last_token_indices
        )

        draft_token = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=hidden_states,
            num_reject_tokens=num_reject_tokens,
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
        )
        # PCP runs the drafter on every rank and the ids come out different.
        # Align them before verification consumes them, so all ranks accept the
        # same count.
        # 【译】PCP在每个rank上都跑草稿模型，得到的id各不相同。在验证使用它们之前先对齐，让所有rank接受相同的个数。
        if draft_token is not None and get_pcp_world_size() > 1:
            draft_token = get_pcp_group().broadcast(draft_token.contiguous(), src=0)
        if align_only:
            return None
        # DSpark Phase 2: stash this step's scheduler-chosen ell keyed by req_id,
        # so next step's calc_spec_decode_metadata can re-map it onto the (possibly
        # reordered) batch. Keying by req_id (not batch position) is required:
        # continuous batching reorders requests between steps.
        # 【译】DSpark第2阶段：把本步调度器选定的ell按req_id存起来，让下一步的calc_spec_decode_metadata能把它重新映射到
        #   （可能已重排的）batch上。必须按req_id而不是batch中的位置做key：连续批处理会在步与步之间重排请求。
        verify_scheduler = getattr(self.drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            verify_scheduler.record_ell(batch.req_ids[: batch.total_seqs_num])
        return self.tokenID_processor.prepare_draft_ids(batch, draft_token)

    # ---------------- CUDA Graph捕获的辅助函数（下面到capture_cudagraph为止），捕获主流程见capture_cudagraph上方 ----------------
    # 【分支】profiler：给捕获阶段抓torch profiler trace（需设置torch_profiler_dir且开mark_trace），只用于性能排查
    def start_capture_profiler(self):
        """Set up the per-bs CUDA graph capture profiler (profiles in place).

        Profiles the capture phase as graphs are captured and writes one trace
        per (batch size, q-bucket), per rank
        (``bs_<bs>_q_<max_q_len>_rank<rank>.json.gz``). Enabled on every rank
        when a torch profiler dir is set and mark-trace is on.
        """
        # 【译】设置按bs划分的CUDA graph捕获profiler（就地profile）。
        #
        #   在捕获graph的同时profile捕获阶段，每个(batch size, q档)、每个rank写一份trace
        #   （bs_<bs>_q_<max_q_len>_rank<rank>.json.gz）。设置了torch profiler目录且开了mark-trace时，每个rank都启用。
        self._capture_profile_enabled = (
            self.profiler_dir is not None and self.mark_trace
        )
        if self._capture_profile_enabled:
            record_shapes = envs.ATOM_PROFILER_RECORD_SHAPES
            with_stack = envs.ATOM_PROFILER_WITH_STACK
            profile_memory = envs.ATOM_PROFILER_PROFILE_MEMORY
            self._capture_trace_tag = None
            self.capture_traces_dir = os.path.join(self.profiler_dir, "capture_traces")
            os.makedirs(self.capture_traces_dir, exist_ok=True)
            logger.info(
                "%s: Starting CUDA graph capture profiler "
                "(record_shapes=%s, with_stack=%s, profile_memory=%s)...",
                self.label,
                record_shapes,
                with_stack,
                profile_memory,
            )

            def on_trace_ready(prof):
                # The window is named from the tag the capture loop stashes
                # before each prof.step(), not from a step counter: batch sizes
                # are skipped (_piecewise_skip_capture) and repeated (once per
                # q-bucket), so any index into self.capture_sizes drifts out of sync
                # with what was actually captured.
                #
                # A cleared tag means this is the trailing window that opens
                # after the last step() and closes at __exit__ — it holds only
                # post-loop bookkeeping, so there is nothing worth writing.
                # 【译】trace窗口用捕获循环在每次prof.step()之前暂存的tag命名，而不是用步数计数：有的batch size会被跳过
                #   （_piecewise_skip_capture），有的会重复（每个q档一次），所以任何按self.capture_sizes下标的命名都会和实际捕获的对不上。
                #
                #   tag被清空，说明这是最后一次step()之后打开、在__exit__时关闭的尾窗口——里面只有循环结束后的收尾工作，没什么值得写的。
                tag = self._capture_trace_tag
                if tag is None:
                    return
                trace_file = os.path.join(
                    self.capture_traces_dir, f"{tag}_rank{self.rank}.json.gz"
                )
                prof.export_chrome_trace(trace_file)
                logger.info(f"Saved capture trace for {tag} to {trace_file}")

            self.capture_profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CUDA,
                    torch_profiler.ProfilerActivity.CPU,
                ],
                # wait=0: recording from __enter__, and every step() closes one
                # window and immediately opens the next, so each iteration of the
                # capture loop lands in its own file with nothing dropped between
                # them (wait>0 would silently skip alternate batch sizes).
                # 【译】wait=0：从__enter__开始就记录，每次step()关闭一个窗口并立刻打开下一个，所以捕获循环的每次迭代都落在自己的文件里，
                #   中间不丢任何东西（wait>0会悄悄跳过间隔的batch size）。
                schedule=torch_profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
                record_shapes=record_shapes,
                with_stack=with_stack,
                profile_memory=profile_memory,
                on_trace_ready=on_trace_ready,
            )
        else:
            self.capture_profiler = nullcontext()

    # _piecewise_cg_active：本次部署走"逐片段捕获"（True）还是"手动整图捕获"（False）
    #   真值表（原生ModelRunner；与config.py CUDAGraphMode上方的模式表一致）：
    #     --enforce-eager                                   → False（此时capture_cudagraph根本不会被调用：EngineCore只在
    #                                                          not enforce_eager时才调它）
    #     --cudagraph-mode FULL（默认）/ NONE               → False → 手动FULL整图捕获（原生路径没有单独判断NONE，照样录整图）
    #     PIECEWISE / FULL_AND_PIECEWISE / AF_PIECEWISE    → True（requires_piecewise_compilation()；需要--level 3在attention处切图）
    #     AF_PIECEWISE在此基础上再录attention核心（capture_cudagraph里的attn_ffn_piecewise）
    #   读取方：prepare_model（给ForwardMode.decide传captured_tokens）、run_model、capture_cudagraph、_estimate_cudagraph_overhead、
    #     _producer_graph_shapes
    @torch.inference_mode()
    def _piecewise_cg_active(self) -> bool:
        """True when the compiled model's dense pieces self-capture PIECEWISE
        cudagraphs (attention eager between them). In that mode the runner does
        NOT build the manual FULL whole-forward graphs — decode calls the model
        directly and the per-piece CUDAGraphWrapper handles capture/replay."""
        # 【译】当编译后模型的稠密片段自己捕获PIECEWISE cudagraph（片段之间的attention是eager）时为True。这种模式下runner
        #   不构建手动的FULL整forward graph——decode直接调用模型，由每个片段的CUDAGraphWrapper负责捕获 / 回放。
        if self.enforce_eager:
            return False
        # Driven by --cudagraph-mode (default FULL -> manual capture, unchanged).
        # PIECEWISE / FULL_AND_PIECEWISE -> per-piece cudagraph path.
        # 【译】由--cudagraph-mode决定（默认FULL → 手动捕获，不变）。PIECEWISE / FULL_AND_PIECEWISE → 逐片段cudagraph路径。
        # 解释：AF_PIECEWISE的value是(PIECEWISE, PIECEWISE)，requires_piecewise_compilation()同样为True
        mode = getattr(self.config.compilation_config, "cudagraph_mode", None)
        return mode is not None and mode.requires_piecewise_compilation()

    # 前置：aiter的custom all-reduce / all_gather / reduce_scatter（ca_comm）用CUDA IPC把同机各卡的一块缓冲互相映射，
    #   集合通信kernel直接读写对端显存、不走RCCL（术语见async_proc.py文件开头）。graph捕获时它有两种做法：
    #   registered：把graph里每次集合通信的输入地址在各rank之间互相登记，回放时kernel直接读对端的原始输入
    #   unreg（copy-in）：先把输入拷进一块预先登记好的池子，再做通信
    # 问题与修复见docstring：PIECEWISE把forward切成很多小图、中间夹着eager段，跨rank的"读完再覆盖"顺序不再成立，
    #   所以PIECEWISE时强制关掉registered路径。调用方：capture_cudagraph（_piecewise为True时）
    def _force_aiter_unreg_capture_for_piecewise(self):
        """PIECEWISE cudagraph + aiter custom all_gather/reduce_scatter: force the
        copy-in ('unreg') capture path instead of the direct-read ('registered')
        one.

        The registered path lets the collective kernel directly read each peer's
        ORIGINAL input pointer (cross-registered at register_graph_buffers). That
        is only safe under a single whole-forward FULL cudagraph, whose global
        read/overwrite ordering holds across all ranks. PIECEWISE splits the
        forward into many small graphs with eager sections between them, losing
        that ordering: a fast rank can overwrite its pool-recycled input via a
        later piece while a slow peer is still reading it -> stale cross-rank
        reads -> progressive hidden corruption -> repeated-token garbage
        (DP+PIECEWISE accuracy bug). The unreg path snapshots the input into a
        pre-registered pool before the collective, so it is order-independent.
        """
        # 【译】PIECEWISE cudagraph + aiter的custom all_gather / reduce_scatter：强制走拷入（"unreg"）的捕获路径，而不是直接读取
        #   （"registered"）的路径。
        #
        #   registered路径让集合通信kernel直接读每个对端的原始输入指针（在register_graph_buffers时互相登记）。这只有在单张
        #   整forward的FULL cudagraph下才安全，那时全局的"读 / 覆盖"顺序在所有rank上都成立。PIECEWISE把forward切成许多小图、
        #   之间夹着eager段，这个顺序就没了：快的rank可能通过后面的某个片段覆盖它那块从池里回收复用的输入，而慢的对端还在读它
        #   → 跨rank读到过期数据 → hidden逐渐被破坏 → 输出重复token的乱码（DP+PIECEWISE精度bug）。unreg路径在集合通信前把输入
        #   快照进一块预先登记的池子，所以与顺序无关。
        seen = set()
        for getter in ("get_tp_group", "get_dp_group", "get_ep_group"):
            try:
                from aiter.dist import parallel_state as _ps

                group = getattr(_ps, getter)()

            # initialised is the ordinary case here, and which exception says
            # so is the aiter build's business, not this probe's.
            # 【译】在这里，（组没有）初始化是常见情况；具体用哪种异常来表明这一点，是aiter构建版本自己的事，不归这个探测管。
            # 解释：某个组没建（例如没开EP）时getter会抛异常，这里直接跳过这个组；seen按ca_comm对象去重（几个组可能共用同一个）
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s is not available: %s", getter, exc)
                continue
            dc = getattr(group, "device_communicator", None)
            ca = getattr(dc, "ca_comm", None) if dc is not None else None
            if ca is None or id(ca) in seen:
                continue
            seen.add(id(ca))
            if getattr(ca, "enable_register_for_capturing", False):
                ca.enable_register_for_capturing = False
                logger.info(
                    "PIECEWISE: forced aiter ca_comm (%s) to unreg copy-in "
                    "capture path for cudagraph-safe DP collectives.",
                    getter,
                )

    # 【分支】DSpark置信度调度（第7批）：每个请求每步验证的草稿数ell_r由调度器决定，batch的max_q_len可能小于mtp_k+1，
    #   所以要为若干个q档各录一套graph。例：mtp_k=7（full_q=8）、q_buckets="4"→ 档位[4, 8]（resolve_q_buckets总会并入full_q）
    def _dspark_capture_q_buckets(self, full_q: int) -> list[int]:
        """DSpark query-length buckets to capture graphs for (paper Phase 2).

        Confidence scheduling replays a SMALLER max_q_len than full_q, so we
        capture one rectangular graph set per bucket. RAGGED and the older
        q-bucket path use independent size sets. Defaults to ``[full_q]`` (the
        Phase-1 single-graph behavior) when confidence scheduling is off.
        """
        # 【译】DSpark要为之捕获graph的query长度档位（论文第2阶段）。
        #
        #   置信度调度回放的max_q_len比full_q小，所以每个档位捕获一套矩形graph。RAGGED和较早的q-bucket路径使用各自独立的尺寸集合。
        #   不开置信度调度时默认是[full_q]（第1阶段的单graph行为）。
        if not (hasattr(self, "drafter") and self.drafter.uses_confidence_schedule):
            return [full_q]
        from atom.spec_decode.dspark_scheduler import resolve_q_buckets

        dspark = self.config.dspark
        sizes = dspark.ragged_graph_sizes if dspark.ragged else dspark.q_buckets
        return resolve_q_buckets(sizes, full_q)

    # PIECEWISE graph显存的估算斜率，读取方：_estimate_cudagraph_overhead（get_num_blocks算KV预算时先扣掉graph的显存）
    #   例（hidden=7168、bf16、61层、单DP）：7168 × 2 × 61 × 2.8 ≈ 2.45e6字节 ≈ 2.3MiB/token；dp=8时再 × 3.48
    def _piecewise_per_token_bytes(self) -> float:
        """Estimated GPU bytes a captured PIECEWISE graph retains per token.

        Derived from model geometry (hidden * dtype * layers * live-tensors/layer)
        so it holds for ANY model, not a magic per-token constant. Under DP the
        MoE all_gathers hidden to ~dp_size x local tokens, so each piece retains
        far more per local token than TP (measured DSV4: TP 2.32MB/tok vs DP
        7.7MB/tok, ~3.3x at dp=8). Attention doesn't amplify, so scale by a
        sub-linear dp**0.6 (8**0.6=3.48, just above the measured 3.3).
        """
        # 【译】估算一张已捕获的PIECEWISE graph每个token占住多少显存字节。
        #
        #   由模型几何（hidden × dtype字节数 × 层数 × 每层存活张量数）推出，所以对任何模型都成立，而不是一个拍脑袋的每token常数。
        #   DP下MoE把hidden all_gather成约dp_size倍的本地token，所以每个片段按本地token算占住的显存远多于TP（DSV4实测：
        #   TP 2.32MB/token，DP 7.7MB/token，dp=8时约3.3倍）。注意力不放大，所以按次线性的dp**0.6缩放（8**0.6=3.48，略高于实测的3.3）。
        hf = self.config.hf_config
        dtype_bytes = torch.finfo(self.config.torch_dtype).bits // 8
        _LIVE_TENSORS_PER_LAYER = 2.8
        per_token = (
            int(hf.hidden_size)
            * dtype_bytes
            * int(hf.num_hidden_layers)
            * _LIVE_TENSORS_PER_LAYER
        )
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1:
            per_token *= float(dp_size) ** 0.6
        return per_token

    # PIECEWISE捕获时决定跳过哪些num_tokens档（调用方：capture_cudagraph、_capture_attn_ffn_graphs）
    #   为什么每个rank的决定必须相同：捕获循环里每档都有跨DP的集合通信（get_dp_padding的all_reduce、模型里的all_gather），
    #     一个rank跳过、另一个不跳，两边的通信就对不上号
    #   例（显存闸，hidden=7168、num_tokens=512、单DP）：need = 600 × 7168 × 512 × 1.25 + 4GiB ≈ 2.56GiB + 4GiB ≈ 6.56GiB，
    #     按整GiB比较（>> 30）：空闲显存不足6GiB就跳过这档
    #   观察：ATOM_PIECEWISE_DP_MAX_TOKENS（默认512）直接读os.environ，没有登记在envs.py里
    def _piecewise_skip_capture(self, num_tokens: int) -> bool:
        """Whether to skip capturing a PIECEWISE bucket of ``num_tokens`` tokens.

        Two guards, both DP-safe (the decision must be identical on every rank,
        else capture loops desync and the next get_dp_padding all_reduce couples
        mismatched num_tokens -> "batch_id_per_q_token len < T"):

        1. DP+spec hard cap: big bs*q buckets never run under DP but bloat the
           pool and don't overlap comm, so cap at ATOM_PIECEWISE_DP_MAX_TOKENS.
        2. Memory guard: skip a bucket whose estimated capture footprint won't
           fit in free GPU memory (adapts to GPU size / config, no hardcoded
           cap). DP amplifies the retained per-token footprint (MoE all_gather
           ~dp_size x tokens); scale the slope by dp**0.6 to match
           _estimate_cudagraph_overhead. Free is min-reduced across DP so all
           ranks skip the same set.
        """
        # 【译】是否跳过捕获一个num_tokens个token的PIECEWISE档位。
        #
        #   两道闸，都是DP安全的（每个rank的决定必须完全相同，否则各rank的捕获循环会错开，下一次get_dp_padding的all_reduce
        #   会把不匹配的num_tokens配在一起 → 报"batch_id_per_q_token len < T"）：
        #
        #   1. DP + 投机解码的硬上限：大的bs*q档位在DP下永远跑不到，却会撑大内存池、也不能重叠通信，所以上限为
        #      ATOM_PIECEWISE_DP_MAX_TOKENS。
        #   2. 显存闸：估算的捕获占用放不进空闲显存时跳过该档（随GPU大小 / 配置自适应，没有写死的上限）。DP会放大每token
        #      占住的显存（MoE all_gather约dp_size倍token）；斜率按dp**0.6缩放，与_estimate_cudagraph_overhead一致。空闲显存在
        #      DP间取最小值，让所有rank跳过同一组档位。
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1 and hasattr(self, "drafter"):
            dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
            if num_tokens > dp_cap:
                if self.rank == 0:
                    logger.info(
                        "PIECEWISE DP-cap skip num_tokens=%d "
                        "(> %d = ATOM_PIECEWISE_DP_MAX_TOKENS)",
                        num_tokens,
                        dp_cap,
                    )
                return True

        # Memory guard slope: capture footprint grows ~linearly with hidden size.
        # Empirically ~600 bytes/token per hidden-dim (0.004GB/token measured at
        # hidden=7168 -> 0.004*2**30/7168 = 599B, rounded).
        # 【译】显存闸的斜率：捕获占用大致随hidden大小线性增长。经验值约每token每个hidden维600字节
        #   （hidden=7168时实测0.004GB/token → 0.004×2**30/7168 = 599B，取整）。
        _GUARD_BYTES_PER_TOKEN_PER_DIM = 600
        slope = _GUARD_BYTES_PER_TOKEN_PER_DIM * self.config.hf_config.hidden_size
        if dp_size > 1:
            slope *= float(dp_size) ** 0.6
        free = torch.cuda.mem_get_info()[0]
        if dp_size > 1:
            import torch.distributed as dist
            from aiter.dist.parallel_state import get_dp_group

            free_t = torch.tensor([free], device="cpu", dtype=torch.int64)
            dist.all_reduce(
                free_t, op=dist.ReduceOp.MIN, group=get_dp_group().cpu_group
            )
            free = int(free_t.item())
        # 需要的显存 = 斜率 × token数 × 1.25（余量）+ 4GiB（留给运行时的其他分配）
        need = slope * num_tokens * 1.25 + (4 << 30)
        if (free >> 30) < (int(need) >> 30):
            if self.rank == 0:
                logger.info(
                    "PIECEWISE skip num_tokens=%d: free=%.1fGB < need=%.1fGB",
                    num_tokens,
                    free / 1e9,
                    need / 1e9,
                )
            return True
        return False

    # 【分支】AF_PIECEWISE（DeepSeek-V4 + DSpark，第7、8批）：在PIECEWISE之外，把attention核心也录成图
    #   前置：DSpark的ragged验证——每条序列本步验证的草稿数各不相同，token被紧密拼成一条"扁平"的run，总数num_tokens_pad
    #     不再等于bs × q（矩形）。稠密片段只按num_tokens分档；attention核心的图还要区分bs和q
    #   例（bs=8、max_q_len=8、capture_sizes含1..8）：矩形是64个token（调用方已录）；这里再录b × 8 ∈ [8, 64)的档：
    #     b=1..7 → num_tokens_pad = 8、16、…、56，每档是"8条序列、长度之和等于num_tokens_pad"的合成batch
    #   调用方：capture_cudagraph（AF_PIECEWISE且后端的build_for_cudagraph_capture接受num_tokens_pad时）
    def _capture_attn_ffn_graphs(
        self, bs, max_q_len, rectangle_tokens, build_capture, input_ids
    ):
        """AF_PIECEWISE: capture the attn_ffn graphs for the smaller ragged buckets
        (num_tokens_pad = b*max_q_len < this bs's rectangle) a real ragged step at
        this bs may replay. Runs one PIECEWISE forward per new bucket on a ragged
        synthetic batch: dense pieces REPLAY (already captured, deduped by
        num_tokens); the attn_ffn op captures its fresh (bs, q_eff, num_tokens_pad)
        key. The rectangle bucket was already captured by the caller.
        """
        # 【译】AF_PIECEWISE：为这个bs下真实的ragged步可能回放的较小ragged档位（num_tokens_pad = b*max_q_len，小于本bs的矩形）
        #   捕获attn_ffn graph。对每个新档位，在一个ragged的合成batch上跑一次PIECEWISE forward：稠密片段走回放（已经捕获过，
        #   按num_tokens去重）；attn_ffn算子捕获它新的(bs, q_eff, num_tokens_pad) key。矩形档位已经由调用方捕获过。
        positions = self.forward_vars["positions"].gpu
        for b in self.capture_sizes:
            num_tokens_pad = b * max_q_len
            if num_tokens_pad >= rectangle_tokens or num_tokens_pad < bs:
                # >= rectangle: the rectangle case (already captured) or larger.
                # < bs: fewer than 1 token/seq — unreachable at real decode.
                # 【译】>= 矩形：就是矩形本身（已经捕获）或更大。< bs：每条序列不到1个token——真实decode到不了。
                continue
            if self._piecewise_skip_capture(num_tokens_pad):
                continue
            # Ragged synthetic metadata: bs seqs whose lengths sum to num_tokens_pad.
            # 【译】ragged合成元数据：bs条序列，长度之和为num_tokens_pad。
            attn_metadata, context = build_capture(
                bs=bs, max_q_len=max_q_len, num_tokens_pad=num_tokens_pad
            )
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens_pad)
            num_tokens_dp = num_tokens_pad + num_pad
            if num_tokens_across_dp is not None:
                num_tokens_across_dp = torch.full_like(
                    num_tokens_across_dp, num_tokens_dp
                )
            model_positions = (
                self._mrope_positions_view(num_tokens_dp)
                if self.use_mrope
                else positions[:num_tokens_dp]
            )
            set_forward_context(
                attn_metadata=attn_metadata,
                atom_config=self.config,
                context=context,
                num_tokens=num_tokens_dp,
                num_tokens_across_dp=num_tokens_across_dp,
                ubatch_slices=None,
                in_hipgraph=True,
            )
            # Warmup, then the PIECEWISE forward: dense pieces replay (deduped by
            # num_tokens); attn_ffn op captures its (bs, q_eff, num_tokens_pad) graph.
            # 【译】先预热，再跑PIECEWISE forward：稠密片段回放（按num_tokens去重）；attn_ffn算子捕获它的(bs, q_eff, num_tokens_pad) graph。
            self.model(input_ids[:num_tokens_dp], model_positions)
            fc = get_forward_context()
            fc.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            fc.batch_descriptor = BatchDescriptor(num_tokens=num_tokens_dp)
            self.model(input_ids[:num_tokens_dp], model_positions)
            fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
            fc.batch_descriptor = None
            self._piecewise_captured_tokens.add(num_tokens_dp)

    @torch.inference_mode()
    def capture_cudagraph(self):
        # M3 indexer-only CP puts an all-to-all in the captured decode path, and
        # NCCL sets up peer connections on a group's FIRST collective -- doing
        # that inside a capture hangs. warmup_model() does not cover it: its
        # dummy batch is prefill-only, and prefill stays on the TP path with no
        # all-to-all. No-op unless the flag is on.
        from atom.distributed.indexer_cp import warmup_exchange

        warmup_exchange(self.device)

        _piecewise = self._piecewise_cg_active()
        # AF_PIECEWISE: also capture the attn core (ragged combos below)
        cudagraph_mode = getattr(self.config.compilation_config, "cudagraph_mode", None)
        attn_ffn_piecewise = (
            cudagraph_mode is not None and cudagraph_mode.is_attn_ffn_piecewise()
        )
        if _piecewise:
            logger.info(
                "PIECEWISE cudagraph: capturing per-piece graphs (attention "
                "eager); manual FULL whole-forward capture disabled."
            )
            self._force_aiter_unreg_capture_for_piecewise()
        start_time = time.time()
        # Config owns the declared ladder; the runner only narrows it to what
        # this deployment can actually schedule (see the bound below).
        self.capture_sizes = self.config.capture_sizes
        self.capture_sizes.sort(reverse=True)

        # Drop any capture size the scheduler could never produce. `schedule_decode`
        # bounds a decode batch two ways: at most `max_num_seqs` sequences, and at
        # most `max_num_batched_tokens` tokens, charging `mtp_k + 1` tokens per
        # sequence whatever query length the step ends up replaying. So the
        # reachable batch size is the min of the two, and under speculation the
        # token budget is what binds first — mtp_k=3 turns 256 sequences into 1024
        # tokens.
        #
        # Filtering on the token budget is not just about avoiding a graph that is
        # never replayed. The per-token forward buffers (`positions`, `input_ids`,
        # `outputs`) are sized `max_num_batched_tokens`, so capture at a bs past
        # this bound writes out of bounds — it used to surface as a bare
        # `could not broadcast input array from shape (1024,) into shape (512,)`
        # out of `capture_cudagraph`, which reads like a shape bug rather than a
        # config one. Charging `mtp_k + 1` (never a smaller q bucket) also keeps
        # every (bs, max_q_len) pair the loop below visits within the bound, so
        # the runtime `self.graphs[(bs, max_q_len)]` lookup cannot miss.
        #
        # Warn rather than raise so a misconfig (default cuda_graph_sizes=[512]
        # vs e.g. max_num_seqs=16) stays recoverable.
        full_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
        max_seq_bs = self.config.max_num_seqs
        max_tok_bs = self.config.max_num_batched_tokens // full_q_len
        max_bs = max_schedulable_decode_bs(
            max_seq_bs, self.config.max_num_batched_tokens, full_q_len
        )
        oversized = [s for s in self.capture_sizes if s > max_bs]
        if oversized:
            self.capture_sizes = [s for s in self.capture_sizes if s <= max_bs]
            logger.warning(
                "cudagraph capture sizes %s exceed the schedulable batch size "
                "min(max_num_seqs=%d, max_num_batched_tokens=%d // (mtp_k+1)=%d "
                "= %d) = %d; dropping. Remaining: %s",
                oversized,
                max_seq_bs,
                self.config.max_num_batched_tokens,
                full_q_len,
                max_tok_bs,
                max_bs,
                self.capture_sizes,
            )
        assert self.capture_sizes, (
            f"no cudagraph capture sizes left: the scheduler can only reach "
            f"bs <= min(max_num_seqs={max_seq_bs}, "
            f"max_num_batched_tokens={self.config.max_num_batched_tokens} // "
            f"(mtp_k+1)={full_q_len} = {max_tok_bs}) = {max_bs}. Pass "
            f"--cudagraph-capture-sizes, raise --max-num-seqs, or raise "
            f"--max-num-batched-tokens."
        )

        # PIECEWISE: the set of num_tokens shapes whose dense pieces we captured
        # (reset here; initialized empty in __init__). run_model dispatches by
        # num_tokens; a shape NOT in here would force a runtime (uncoordinated)
        # capture that hangs on collectives, so run_model falls back to eager for
        # uncaptured shapes.
        self._piecewise_captured_tokens = set()

        if "kv_indptr" in self.forward_vars:
            self.forward_vars["kv_indptr"].gpu.zero_()
        # Present exactly when the model has an indexer -- the builder makes it
        # under the same answer -- so the buffer's own existence is the test.
        if "sparse_kv_indptr" in self.forward_vars:
            self.forward_vars["sparse_kv_indptr"].gpu.zero_()

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_logits: dict[tuple[int, int], torch.Tensor] = {}
        self.graph_pool = None
        is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
        # TBO graphs don't capture compute_logits, so disable logits_in_graph.
        self.logits_in_graph = self.world_size == 1 and not is_tbo

        # start capture profiler
        self.start_capture_profiler()

        @contextmanager
        def pause_gc():
            # No GC during capture: a finalizer's hipModuleUnload aborts it (HIP 900).
            gc.collect()
            gc.disable()
            try:
                yield
            finally:
                gc.enable()
                gc.collect()

        _rsv_before_capture = torch.cuda.memory_reserved()
        _alloc_before_capture = torch.cuda.memory_allocated()

        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]

        # Capture one graph per (bs, query-length bucket). Buckets default to
        # [full_q_len] (single-graph, classic per-bs capture); DSpark confidence
        # scheduling expands to the smaller q-buckets a decode step may replay.
        q_buckets = self._dspark_capture_q_buckets(full_q_len)
        if q_buckets != [full_q_len]:
            logger.info("DSpark CUDA-graph query buckets: %s", q_buckets)
        elif hasattr(self, "drafter") and self.drafter.uses_confidence_schedule:
            # resolve_q_buckets always folds full_q in, so a spec naming only
            # full_q (or nothing, or nothing valid) collapses to [full_q] and
            # every step replays at full length. The step still pays the
            # confidence schedule + ragged rebuild, so say so rather than look
            # like it is shrinking anything.
            dspark = self.config.dspark
            spec = dspark.ragged_graph_sizes if dspark.ragged else dspark.q_buckets
            logger.warning(
                "DSpark %s=%r resolves to [%d] (== full verify length), so no "
                "query-length shrink is possible and every decode step replays "
                "at full length. Pass sizes BELOW %d to get any benefit.",
                "ragged_graph_sizes" if dspark.ragged else "q_buckets",
                spec,
                full_q_len,
                full_q_len,
            )

        self.h2d_owner.begin()
        self.attn_metadata_builder.blank_cache_write_targets()
        self.h2d_owner.finish()

        # Whether this backend's capture builder supports a dynamic (per-bucket)
        build_capture = self.attn_metadata_builder.build_for_cudagraph_capture
        _build_params = inspect.signature(build_capture).parameters
        supports_dynamic_q_len = "max_q_len" in _build_params
        # Whether it supports a ragged num_tokens_pad (zero-copy-q attn-core graphs).
        supports_ragged_capture = "num_tokens_pad" in _build_params

        raw_build_capture = build_capture

        @wraps(raw_build_capture)
        def build_capture(*args, **kwargs):
            # Preparation is outside actual graph capture, including calls
            # made by the drafter and ragged bucket builder. Each invocation
            # owns a synthetic transaction; normal forwards never use it.
            bs = kwargs.get("bs", args[0] if args else None)
            q_len = kwargs.get("max_q_len", full_q_len)
            self.h2d_owner.begin()
            try:
                if not self.attn_metadata_builder.capture_owns_cu_seqlens_q:
                    cu = self.forward_vars["cu_seqlens_q"]
                    cu.np[: bs + 1] = np.arange(
                        0, (bs + 1) * q_len, q_len, dtype=np.int32
                    )
                    cu.copy_to_gpu(bs + 1)
                tokens = bs * q_len
                self.forward_vars["positions"].np[:tokens] = (
                    np.arange(tokens, dtype=np.int64) % q_len
                )
                result = raw_build_capture(*args, **kwargs)
                self.h2d_owner.finish()
                return result
            except BaseException:
                self.h2d_owner.fail()
                raise

        with pause_gc(), graph_capture() as capture_ctx, self.capture_profiler as prof:
            for max_q_len in q_buckets:
                capture_range = (
                    tqdm.tqdm(self.capture_sizes)
                    if self.rank == 0
                    else self.capture_sizes
                )
                for bs in capture_range:
                    if self.rank == 0:
                        capture_range.set_description(f"Capturing {bs=}, {max_q_len=}")

                    num_tokens = bs * max_q_len
                    if _piecewise and self._piecewise_skip_capture(num_tokens):
                        continue
                    # Names the capture trace this iteration will export. Set
                    # after the skip above so a skipped bs never claims a file;
                    # its handful of Python statements just fold into the next
                    # iteration's window.
                    self._capture_trace_tag = f"bs_{bs}_q_{max_q_len}"
                    if supports_dynamic_q_len:
                        attn_metadata, context = build_capture(
                            bs=bs, max_q_len=max_q_len
                        )
                    else:
                        attn_metadata, context = build_capture(bs=bs)
                    if self.use_mrope:
                        mrope_positions = self._mrope_positions_view(num_tokens)
                        mrope_positions.copy_(
                            positions[:num_tokens].unsqueeze(0).expand(3, -1)
                        )
                        context.positions = mrope_positions
                    num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                    num_tokens += num_pad
                    # get_dp_padding built num_tokens_across_dp from the PRE-pad
                    # count, but we just padded num_tokens. Capture is symmetric
                    # (every DP rank captures the same bs), so the padded count is
                    # uniform across ranks. Rebuild the tensor at the padded size so
                    # DPMetadata.make's `across_dp[rank] == num_tokens` holds.
                    if num_tokens_across_dp is not None:
                        num_tokens_across_dp = torch.full_like(
                            num_tokens_across_dp, num_tokens
                        )
                    # Create ubatch slices for TBO capture (need > 2 requests)
                    ubatch_slices = None
                    ub_tokens_across_dp = None
                    if is_tbo and self.config.enable_tbo_decode and bs > 2:
                        ubatch_slices = maybe_create_ubatch_slices(
                            num_reqs=bs,
                            num_tokens=num_tokens,
                        )
                        # The rebuild above's symmetry, one level down: every
                        # rank splits this bucket the same way, so a ubatch's
                        # per-rank counts are its own repeated. Stated, because
                        # a capture context declares its shape where a real step
                        # reduces one, and no consumer can tell an absent
                        # reduction from a uniform answer.
                        if ubatch_slices and num_tokens_across_dp is not None:
                            dp = len(num_tokens_across_dp)
                            ub_tokens_across_dp = tuple(
                                (s.token_slice.stop - s.token_slice.start,) * dp
                                for s in ubatch_slices
                            )

                    set_forward_context(
                        attn_metadata=attn_metadata,
                        atom_config=self.config,
                        context=context,
                        num_tokens=num_tokens,
                        num_tokens_across_dp=num_tokens_across_dp,
                        ubatch_slices=ubatch_slices,
                        ub_tokens_across_dp=ub_tokens_across_dp,
                        in_hipgraph=True,
                    )

                    # Warmup
                    model_positions = (
                        self._mrope_positions_view(num_tokens)
                        if self.use_mrope
                        else positions[:num_tokens]
                    )
                    model_output = self.model(input_ids[:num_tokens], model_positions)
                    outputs[:num_tokens] = model_output
                    if self.logits_in_graph:
                        self.model.compute_logits(outputs[:num_tokens])

                    if _piecewise:
                        # PIECEWISE: no manual whole-forward graph; the compiled
                        # per-piece wrappers self-capture. Replay once to register.
                        fc = get_forward_context()
                        fc.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                        fc.batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
                        self.model(input_ids[:num_tokens], model_positions)
                        fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
                        fc.batch_descriptor = None
                        self._piecewise_captured_tokens.add(num_tokens)
                        # also capture the attn_ffn graphs this bs can replay ragged
                        if attn_ffn_piecewise and supports_ragged_capture:
                            self._capture_attn_ffn_graphs(
                                bs=bs,
                                max_q_len=max_q_len,
                                rectangle_tokens=num_tokens,
                                build_capture=build_capture,
                                input_ids=input_ids,
                            )
                        if prof is not None:
                            # Drain before closing the window so this bs's
                            # kernels land in this bs's file. Profiling-only —
                            # the unprofiled PIECEWISE path stays sync-free.
                            torch.cuda.synchronize()
                            prof.step()
                            self._capture_trace_tag = None
                        continue

                    # Capture
                    with (
                        record_function(f"capture_graph_bs_{bs}")
                        if self.mark_trace
                        else nullcontext()
                    ):
                        if ubatch_slices is not None:
                            # TBO capture: threads + multi-stream captured in graph.
                            # Drafter aux (if any) is written inside the graph by
                            # capture_tbo_graph replaying the model's forward hooks.
                            graph, _ = self.model.capture_tbo_graph(
                                input_ids[:num_tokens],
                                positions[:num_tokens],
                                self.graph_pool,
                                capture_ctx.stream,
                                output_buffer=outputs[:num_tokens],
                            )
                        else:
                            # Standard single-stream capture. The drafter's aux
                            # capture hook runs inside the forward and writes its
                            # own buffers (captured in-graph); model_output is a
                            # plain Tensor.
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(
                                graph, self.graph_pool, stream=capture_ctx.stream
                            ):
                                model_output = self.model(
                                    input_ids[:num_tokens], model_positions
                                )
                                outputs[:num_tokens] = model_output
                                if self.logits_in_graph:
                                    graph_logits = self.model.compute_logits(
                                        outputs[:num_tokens]
                                    )
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(bs, max_q_len)] = graph
                    if self.logits_in_graph and ubatch_slices is None:
                        self.graph_logits[(bs, max_q_len)] = graph_logits
                    torch.cuda.synchronize()
                    # After the sync: the warmup forward's kernels must have
                    # completed before the window closes, or they spill into the
                    # next bs's file.
                    if prof is not None:
                        prof.step()
                        self._capture_trace_tag = None
            # Inside the `with`: graph_capture() arms the custom all-reduce for
            # capture and pause_gc() keeps the collector from aborting one. The
            # drafter gets the capture builder already bound to this backend's
            # signature -- which of them takes a `max_q_len` is the runner's to
            # know, and it is the same probe the loop above ran.
            if hasattr(self, "drafter"):
                self.drafter.warmup_draft_graphs(
                    (
                        partial(build_capture, max_q_len=full_q_len)
                        if supports_dynamic_q_len
                        else build_capture
                    ),
                    capture_ctx.stream,
                )
        self.capture_sizes.sort()
        self.capture_sizes_np = np.asarray(self.capture_sizes, dtype=np.int32)

        # PIECEWISE: sorted 1D num_tokens buckets for run_model's round_up_1d(Σ)
        # dispatch (bisect_left over this to pick the tightest captured shape).
        self._piecewise_sorted_tokens = sorted(self._piecewise_captured_tokens)
        if _piecewise and self.rank == 0:
            logger.info(
                "PIECEWISE captured %d num_tokens buckets: %s",
                len(self._piecewise_sorted_tokens),
                self._piecewise_sorted_tokens,
            )

        # DSpark Phase 2: calibrate the SPS(B) throughput profile from the just-
        # captured target graphs (each is a B = bs*max_q_len token forward, i.e.
        # exactly one verification step at batch B). Cheap, GPU-only, one-shot.
        self._maybe_calibrate_dspark_sps(full_q_len)

        # How much GPU memory the CUDA graph capture consumed (pool = reserved
        # delta; the allocated delta is what the graphs pin live).
        _pool_bytes = max(torch.cuda.memory_reserved() - _rsv_before_capture, 0)
        _alloc_bytes = max(torch.cuda.memory_allocated() - _alloc_before_capture, 0)
        if self.rank == 0:
            logger.info(
                "CUDA graph capture memory: %d graphs | pool(reserved)=%.2fGB "
                "allocated=%.2fGB",
                len(self.graphs) + len(self._piecewise_captured_tokens),
                _pool_bytes / (1 << 30),
                _alloc_bytes / (1 << 30),
            )

        # Post-init memory validation
        free_after, total_after = torch.cuda.mem_get_info()
        actual_usage = total_after - free_after
        target_usage = int(total_after * self.config.gpu_memory_utilization)
        usage_ratio = actual_usage / total_after
        logger.info(
            f"Post-init memory: "
            f"actual={actual_usage / (1 << 30):.2f}GB ({usage_ratio:.1%}), "
            f"target={target_usage / (1 << 30):.2f}GB "
            f"({self.config.gpu_memory_utilization:.0%}), "
            f"reserved={torch.cuda.memory_reserved() / (1 << 30):.2f}GB, "
            f"allocated={torch.cuda.memory_allocated() / (1 << 30):.2f}GB"
        )
        if usage_ratio > self.config.gpu_memory_utilization + 0.02:
            logger.warning(
                f"Actual GPU memory usage ({usage_ratio:.1%}) exceeds target "
                f"({self.config.gpu_memory_utilization:.0%}) by "
                f"{(usage_ratio - self.config.gpu_memory_utilization):.1%}. "
                f"Consider reducing gpu_memory_utilization."
            )

        return time.time() - start_time, self.capture_sizes, _pool_bytes

    @torch.inference_mode()
    def _maybe_calibrate_dspark_sps(self, max_q_len: int, n_iters: int = 20) -> None:
        """Profile SPS(B) by timing the captured target graphs, then hand a dense
        cost table to the DSpark drafter (paper §3.2.2, scheduler input).

        Each captured graph ``self.graphs[(bs, max_q_len)]`` is a forward over
        ``B = bs * max_q_len`` tokens — exactly one verification step at batch B.
        We replay each a few times, take the median step time, and densify the
        (B, steps/sec) samples into ``sps_table[B]``. No-op unless a DSpark
        drafter with confidence scheduling enabled is present.
        """
        drafter = getattr(self, "drafter", None)
        if drafter is None or not drafter.is_block_drafter:
            return
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is None:
            return
        if verify_scheduler.calibration_profile is not None:
            return  # Explicit offline measurements must not be overwritten.
        if not getattr(self, "graphs", None):
            return
        if self.config.dspark.disable_sps_calib:
            logger.info("DSpark SPS calibration disabled; using synthetic stub.")
            return

        from atom.spec_decode.dspark_scheduler import build_sps_table

        # DSpark RAGGED graph: replay-based SPS calibration is UNSAFE here. Each
        # `graph.replay()` runs the FULL decode graph (incl. SWA/KV writes) with
        # synthetic data at real cache slots [0:bs], polluting the KV cache real
        # requests then read. The scheduler only needs a monotone SPS(B) shape,
        # so use a synthetic table instead (matches the proven DISABLE_SPS_CALIB
        # path). Timed ragged calibration is a follow-up (needs a scratch KV pool
        # + buffer save/restore around the replays).
        if self.config.dspark.ragged:
            logger.info(
                "DSpark SPS calibration skipped under RAGGED graph "
                "(replay would pollute KV cache); using synthetic stub."
            )
            return

        token_points: list[int] = []
        sps_points: list[float] = []
        for bs in self.capture_sizes:
            graph = self.graphs.get((bs, max_q_len))
            if graph is None:
                continue
            B = bs * max_q_len
            # Every row of the recording is real here: time the whole of it
            # (the DP pad-row mask reads the real token count from here).
            cu = self.forward_vars["cu_seqlens_q"]
            cu.np[: bs + 1] = np.arange(0, B + 1, max_q_len, dtype=np.int32)
            cu.copy_to_gpu(bs + 1)
            # Warm replay, then timed replays (median for robustness to jitter).
            graph.replay()
            torch.cuda.synchronize()
            times_ms: list[float] = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times_ms.append(start.elapsed_time(end))
            times_ms.sort()
            median_ms = times_ms[len(times_ms) // 2]
            if median_ms <= 0:
                continue
            token_points.append(B)
            sps_points.append(1000.0 / median_ms)  # steps per second

        if not token_points:
            logger.warning("DSpark SPS calibration found no timeable graphs.")
            return

        max_b = self.config.max_num_seqs * max_q_len
        sps_table = build_sps_table(token_points, sps_points, max_b).to(self.device)
        verify_scheduler.sps_table = sps_table
        logger.info(
            "DSpark SPS calibrated over %d points (B=%d..%d), table size %d.",
            len(token_points),
            token_points[0],
            token_points[-1],
            sps_table.numel(),
        )


class RapidServeModelRunner(ModelRunner):
    """ModelRunner for intra-GPU prefill/decode disaggregation.

    The same class runs in both the prefill and decode processes; behavior that
    differs between them keys off config.disagg_is_decode.
    """

    def __init__(self, rank, config):
        if not config.disagg_is_decode:
            self.forward = self.prefill_forward
        super().__init__(rank, config)
        import hashlib

        # Session ID derived from the disagg IPC address so IPC handoff temp
        # files from different engine runs never collide.
        self._disagg_session_id = hashlib.md5(
            config.disagg_kvcache_ipc_addr.encode()
        ).hexdigest()[:12]

    @staticmethod
    @contextmanager
    def _init_weight_params_on_meta():
        """Construct a model with all `nn.Parameter`s on the meta device (no
        persistent GPU weight allocation), leaving the default device unchanged
        so init code that explicitly targets CUDA (e.g. aiter RoPE) and buffers
        work normally. Each parameter is briefly created on the real device then
        replaced with a meta tensor, so the transient peak is one parameter, not
        the whole model. Used by decode in disagg, which fills params from
        prefill via IPC.
        """
        orig_register = torch.nn.Module.register_parameter

        def register_parameter(self, name, param):
            orig_register(self, name, param)
            p = self._parameters.get(name)
            if p is not None:
                self._parameters[name] = torch.nn.Parameter(
                    p.detach().to("meta"), requires_grad=p.requires_grad
                )

        torch.nn.Module.register_parameter = register_parameter
        try:
            yield
        finally:
            torch.nn.Module.register_parameter = orig_register

    # ------------------------------------------------------------------
    # Base ModelRunner override points
    # ------------------------------------------------------------------

    def _build_and_load_model(self, model_class):
        config = self.config
        if not config.disagg_is_decode:
            super()._build_and_load_model(model_class)
            return
        # Decode imports prefill's weights via CUDA IPC and owns no weight
        # memory. Build on the meta device so construction allocates zero GPU
        # bytes (avoids the transient 2x-weights peak that OOMs at TP=4);
        # import_model_weight_ipc_handles() materializes params from prefill,
        # and RoPE caches are recomputed locally.
        with self._init_weight_params_on_meta():
            self.model = model_class(config)
        torch.set_default_device(None)

    def _maybe_warmup(self):
        # Decode owns no GPU memory yet (weights/kvcache imported from prefill
        # later), so warmup would run against meta-device tensors — skip it.
        if self.config.disagg_is_decode:
            return
        super()._maybe_warmup()

    def _kv_budget_extra_reserve(self, total_bytes: int) -> int:
        # Two processes share the GPU in disagg mode; reserve extra headroom so
        # NCCL/system allocs (e.g. the 512MB NCCL barrier buffer) don't OOM.
        safety_margin = int(total_bytes * 0.02)
        return 4 * safety_margin

    def get_num_blocks(self) -> dict[str, object]:
        # Decode in disagg mode owns no GPU memory — kvcache is imported from
        # prefill.
        if self.config.disagg_is_decode:
            transfer = self.attn_metadata_builder.state_transfer()
            if transfer.copies:
                raise RuntimeError(
                    "PAGE-backed state checkpoints do not yet support RapidServe "
                    "prefill/decode disaggregation"
                )
            return {
                "num_kvcache_blocks": 0,
                "state_runtime": StateRuntime(transfer=transfer).to_wire(),
            }
        return super().get_num_blocks()

    def allocate_kv_cache(self, num_kvcache_blocks):
        # Decode in disagg mode: kvcache is imported from prefill, not allocated.
        if self.config.disagg_is_decode:
            logger.info("decode skipping kv cache allocation")
            return True
        return super().allocate_kv_cache(num_kvcache_blocks)

    @_profile_runner_stage
    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        # Decode runs the model forward on a dynamically selected (optionally
        # CU-masked) stream so it doesn't contend with prefill on the shared
        # GPU; postprocess (sampling + async CPU copy) stays on the default
        # stream. The stream pool is created by DecodeEngineCore via
        # create_decode_stream_pool().
        stream = self._decode_streams[batch.cu_stream_fraction]
        self._done_event.record()
        stream.wait_event(self._done_event)
        with torch.cuda.stream(stream):
            self._advance_forward_vars()
            self._gate_staging_reuse()
            (
                input_ids,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise,
            ) = self.prepare_model(batch)
            self._mark_staging_h2d_enqueued()
            logits, hidden_states = self.run_model(input_ids, batch)
        self._model_fwd_event.record(stream)
        torch.cuda.current_stream().wait_event(self._model_fwd_event)

        # postprocess (sampling + async CPU copy) always runs on default stream.
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )

        self._record_forward_vars_event()
        reset_forward_context()
        return fwd_output

    # ------------------------------------------------------------------
    # Disagg IPC helpers (TP-aware)
    #
    # With TP>1, AsyncIOProcManager broadcasts each RPC to all ranks but only
    # rank 0's return value reaches the engine (the other ranks' output sockets
    # are wired to None).  For IPC handle exchange we therefore use a per-rank
    # temp-file rendezvous: every rank pickles its local handles to
    #   /tmp/atom_disagg_<tag>_rank<N>.pkl
    # Rank 0 waits until all N files exist, then returns the list of paths to
    # the engine.  On the import side every rank reads its own file (index=self.rank).
    # ------------------------------------------------------------------

    def _disagg_rank_file_path(self, tag: str, rank: int) -> str:
        import tempfile

        return os.path.join(
            tempfile.gettempdir(),
            f"atom_disagg_{self._disagg_session_id}_{tag}_rank{rank}.pkl",
        )

    def _disagg_write_rank_file(self, tag: str, payload) -> str:
        """Pickle *payload* to a session-unique per-rank temp file.

        Uses a session ID derived from the disagg IPC address so that files
        from different engine runs never collide.  No deletion: all ranks
        write concurrently when they receive the broadcast RPC, so rank 0
        must never delete sibling files.
        """
        import pickle

        path = self._disagg_rank_file_path(tag, self.rank)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def _disagg_collect_rank_files(self, tag: str) -> list[str] | None:
        """Rank 0: poll until all world_size rank files exist, return paths.
        Other ranks: return None immediately (rank 0 is the sole publisher).
        """
        import time

        if self.rank != 0:
            return None
        paths = [self._disagg_rank_file_path(tag, r) for r in range(self.tp_world_size)]
        deadline = time.monotonic() + 120  # 2 min timeout
        while time.monotonic() < deadline:
            if all(os.path.exists(p) for p in paths):
                return paths
            time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for disagg rank files for tag={tag!r}: "
            + str([p for p in paths if not os.path.exists(p)])
        )

    def export_model_weight_ipc_handles(self) -> list[str] | None:
        """Export all model parameters as CUDA IPC handles (prefill process only).

        TP-aware: each rank writes its own handles to a temp file.  Rank 0 waits
        for all ranks' files and returns the list of paths; other ranks return None.
        Decode calls import_model_weight_ipc_handles(paths) to replace its own
        weight tensors with zero-copy views into prefill's allocation.
        """
        from atom.model_engine.ipc_utils import export_model_weight_handles

        logger.info(f"ModelRunner rank {self.rank}: export_model_weight_ipc_handles")
        handles = export_model_weight_handles(self.model)
        self._disagg_write_rank_file("weights", handles)
        paths = self._disagg_collect_rank_files("weights")
        if paths is not None:
            logger.info(
                f"ModelRunner rank 0: all {self.tp_world_size} weight files ready"
            )
        return paths  # non-None only for rank 0

    def import_model_weight_ipc_handles(self, paths: list[str]) -> bool:
        """Replace model parameters with views into prefill's GPU allocation.

        TP-aware: each rank reads its own handles file (index=self.rank) and
        deletes it after import.  Returns True as sentinel for wait_out=True.
        """
        import gc
        import pickle

        from atom.model_engine.ipc_utils import import_model_weights

        path = paths[self.rank]
        logger.info(f"ModelRunner rank {self.rank}: reading weight handles from {path}")
        with open(path, "rb") as f:
            handles = pickle.load(f)
        os.remove(path)
        import_model_weights(self.model, handles)
        gc.collect()
        torch.cuda.empty_cache()
        # Surface any tensor prefill didn't export (would crash later in forward).
        leftover = [n for n, p in self.model.named_parameters() if p.is_meta] + [
            n
            for n, b in self.model.named_buffers()
            if isinstance(b, torch.Tensor) and b.is_meta
        ]
        if leftover:
            logger.warning(
                f"ModelRunner rank {self.rank}: {len(leftover)} tensors still on "
                f"meta after IPC import (not materialized): {leftover[:10]}"
            )
        logger.info(
            f"ModelRunner rank {self.rank}: weight IPC import complete — own weights freed"
        )
        return True

    def export_kv_cache_ipc_handle(self) -> list[str] | None:
        """Export self.kv_cache — the whole paged pool — as a CUDA IPC handle.

        TP-aware: each rank writes its handles to a temp file.  Rank 0 waits for
        all ranks and returns the list of paths; other ranks return None.
        """
        from atom.model_engine.ipc_utils import export_kv_cache_handle

        logger.info(f"ModelRunner rank {self.rank}: export_kv_cache_ipc_handle")
        handles = export_kv_cache_handle(self.kv_cache)
        self._disagg_write_rank_file("kvcache", handles)
        paths = self._disagg_collect_rank_files("kvcache")
        if paths is not None:
            logger.info(
                f"ModelRunner rank 0: all {self.tp_world_size} kvcache files ready"
            )
        return paths  # non-None only for rank 0

    def import_kv_cache_ipc_handle(
        self, paths: list[str], num_kvcache_blocks: int
    ) -> bool:
        """Import kvcache from prefill's GPU allocation into this (decode) process.

        TP-aware: each rank reads its own handles file (index=self.rank) and
        deletes it after import.  Also imports kv_scale when present (fp8).
        Returns True as sentinel for wait_out=True.
        """
        import pickle

        from atom.model_engine.ipc_utils import import_kv_cache

        # The count travels with the handle and reaches the builder through
        # `_bind_kv_cache_to_modules` -> `allocate_kv_cache_tensors` below; this
        # side never ran sizing, so there is nothing else it could come from.
        path = paths[self.rank]
        logger.info(
            f"ModelRunner rank {self.rank}: reading kvcache handles from {path}"
        )
        with open(path, "rb") as f:
            meta = pickle.load(f)
        os.remove(path)
        logger.info(f"ModelRunner rank {self.rank}: hipIpcOpenMemHandle for kvcache...")
        self.kv_cache = import_kv_cache(meta)
        logger.info(
            f"ModelRunner rank {self.rank}: kvcache IPC import done, binding..."
        )
        self._bind_kv_cache_to_modules(num_kvcache_blocks)
        logger.info(f"ModelRunner rank {self.rank}: import_kv_cache_ipc_handle done")
        return True

    def _bind_kv_cache_to_modules(self, num_kvcache_blocks: int):
        """Bind an IPC-imported KV pool to every attention module.

        The decode side of a P/D pair gets the pool as a handle, so it never
        reaches `allocate_kv_cache`'s bind loop. What it must not do is bind it
        a *second way*: this used to spell the views out again, with a 4-D V
        where every builder declared 5-D. Nothing could catch that — the two
        agree on every byte and differ only in what a reader branches on. So it
        runs the same loop over the same hook, down to backing the pools.

        The paged pool only. A per-request STATE class is sized by `pool_plan`,
        which is empty here -- the block count arrives with the handle and
        nothing else does -- so `allocate_per_req_cache` cannot run and the
        attributes it publishes do not exist. Refused up front rather than met
        as an `AttributeError` on whichever layer binds first.
        """
        stateful = self._state_pool_names()
        if stateful:
            raise NotImplementedError(
                f"per-request state {stateful} has no entry count on the decode "
                "side of a P/D pair: only the paged pool arrives with the "
                "handle, and this side never ran sizing. Run this model "
                "without disaggregation."
            )
        self._back_paged_pools(num_kvcache_blocks, buf=self.kv_cache)

        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        kv_cache_data = {}
        for model_name, model in models_to_bind:
            for module in model.modules():
                # Same dispatch as the allocating path's loop: a draft with a
                # pool of its own binds through its own builder, and its pool
                # is now a region of the same imported buffer.
                bound = None
                if model_name == "draft" and hasattr(self, "draft_kv_builder"):
                    bound = self.draft_kv_builder.build_kv_cache_tensor(module)
                if bound is None:
                    bound = self.attn_metadata_builder.build_kv_cache_tensor(module)
                if bound is not None:
                    # The same key the allocating path uses. Keyed by the walk's
                    # ordinal here until now, which agreed with it only while
                    # the first bound module was layer 0.
                    kv_cache_data[f"layer_{module.layer_num}"] = bound

        from atom.utils.forward_context import set_kv_cache_data

        set_kv_cache_data(kv_cache_data)

    # ------------------------------------------------------------------
    # CU-masked stream pools + prefill forward
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_with_cu_mask(mask_bits: list[int]) -> torch.cuda.ExternalStream:
        """Create a HIP stream restricted to the CUs described by mask_bits.

        mask_bits is a list of uint32 words; bit i of word w represents
        CU (w*32 + i).  Uses hipExtStreamCreateWithCUMask (ROCm only).
        """
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
        hip.hipExtStreamCreateWithCUMask.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        raw_stream = ctypes.c_void_p()
        mask_arr = (ctypes.c_uint * len(mask_bits))(*mask_bits)
        ret = hip.hipExtStreamCreateWithCUMask(
            ctypes.byref(raw_stream), len(mask_bits), mask_arr
        )
        assert ret == 0, f"HIP err {ret} creating masked stream"
        return torch.cuda.ExternalStream(raw_stream.value)

    @staticmethod
    def _cu_mask_for_fraction(fraction: float, upper: bool) -> list[int]:
        """Return CU mask bits for the given fraction.

        For upper=False (prefill): CUs [0, split).
        For upper=True  (decode):  CUs [split, total).
        split = round(total * fraction).
        """
        total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        split = max(1, min(total - 1, round(total * fraction)))
        start = split if upper else 0
        end = total if upper else split
        num_words = (total + 31) // 32
        words = [0] * num_words
        for cu in range(start, end):
            words[cu // 32] |= 1 << (cu % 32)
        side = "decode" if upper else "prefill"
        logger.info(
            f"CU mask ({side}): CUs [{start},{end}) "
            f"(fraction={fraction}, total={total})"
        )
        return words

    # CU fractions for which we pre-create masked streams.
    _CU_POOL_FRACTIONS: ClassVar[list[float]] = [0.5]

    def create_prefill_stream_pool(self) -> bool:
        """Create a pool of CUDA streams for disaggregated prefill.

        Called once by PrefillEngineCore._init_disagg() after IPC import.
        In constrained mode, pre-creates one CU-masked stream per fraction
        in _CU_POOL_FRACTIONS plus a full-CU fallback (None key);
        prefill_forward() selects the stream dynamically each iteration via
        _optimal_cu_fraction().  In unconstrained mode, only the plain
        full-CU stream (None key) is created.
        """
        self._prefill_streams = {}
        if getattr(self.config, "disagg_constrained", False):
            for f in self._CU_POOL_FRACTIONS:
                mask = self._cu_mask_for_fraction(f, upper=False)
                self._prefill_streams[f] = self._stream_with_cu_mask(mask)
        # Full-CU fallback (no mask) — always present, sole entry in unconstrained mode
        self._prefill_streams[None] = torch.cuda.Stream()
        logger.info(
            f"Prefill stream pool created: fractions={list(self._prefill_streams.keys())}"
        )
        return True

    def create_decode_stream_pool(self) -> bool:
        """Create a pool of CUDA streams for disaggregated decode.

        Called once by DecodeEngineCore._init_disagg().  In constrained mode,
        complementary to the prefill pool: for fraction F, decode gets CUs
        [F*total, total).  forward() selects the stream dynamically each
        iteration.  In unconstrained mode, only the plain full-CU stream
        (None key) is created.
        """
        self._decode_streams = {}
        if getattr(self.config, "disagg_constrained", False):
            for f in self._CU_POOL_FRACTIONS:
                mask = self._cu_mask_for_fraction(f, upper=True)
                self._decode_streams[f] = self._stream_with_cu_mask(mask)
        # Full-CU fallback (no mask) — always present, sole entry in unconstrained mode
        self._decode_streams[None] = torch.cuda.Stream()
        self._model_fwd_event = torch.cuda.Event()
        self._done_event = torch.cuda.Event()
        logger.info(
            f"Decode stream pool created: fractions={list(self._decode_streams.keys())}"
        )
        return True

    @torch.inference_mode()
    def prefill_forward(self, batch: ScheduledBatch) -> list[int]:
        """Run a prefill forward pass on a dynamically selected CU-masked stream.

        Writes KV for all prompt tokens, samples the first generated token for
        each sequence, then synchronizes before returning so decode's default
        stream sees all KV writes.  Returns a list of sampled token IDs (one
        per sequence, in batch order) — these are included in PrefillDone so
        the decode process can append them before the first decode step,
        matching the num_tokens state that non-disagg postprocess would produce.
        """
        prefill_streams = getattr(self, "_prefill_streams", None)
        if prefill_streams is not None:
            stream = self._prefill_streams[batch.cu_stream_fraction]
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self._advance_forward_vars()
            self._gate_staging_reuse()
            (
                input_ids,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                _needs_independent_noise,
            ) = self.prepare_model(batch)
            self._mark_staging_h2d_enqueued()
            logits, _ = self.run_model(input_ids, batch)
            # Sample the first generated token from each sequence's last logit
            sampled = self.sampler(logits, temperatures, top_ks, top_ps, all_greedy)
            sampled_cpu = sampled.view(-1).tolist()
            self._record_forward_vars_event()
        # Synchronize so decode's default stream sees all KV writes.
        stream.synchronize()
        self._record_kv_cache_ready(batch)
        reset_forward_context()
        return sampled_cpu
