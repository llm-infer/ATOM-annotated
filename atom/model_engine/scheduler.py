# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Scheduling logic for batching prefill and decode requests.

This module provides:

- :class:`ScheduledBatch`: A frozen snapshot of sequences selected for the
  next forward pass, together with their block tables and metadata.
- :class:`ScheduledBatchOutput`: Token-level outputs from a completed batch.
- :class:`Scheduler`: The main scheduling loop that manages *waiting* and
  *running* queues, coordinates block allocation, and integrates with the
  KV disaggregation connector for remote prefill/decode.

Every scheduler here owns an :class:`~atom.model_engine.engine_stats.EngineStats`
(``self.engine_stats``); the class itself lives in ``engine_stats.py``.
"""

from __future__ import annotations

import enum
import itertools
import logging
import math
import struct
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from atom.config import Config
from atom.kv_transfer.disaggregation import KVConnectorOutput, kv_config_has_producer
from atom.metrics.scheduler import SchedulerMetrics
from atom.model_engine.block_manager import BlockManager, draft_kv_lookahead
from atom.model_engine.dynamic_chunking import (
    CALIBRATION_SWEEP_RATIO,
    USEFUL_SHRINK_REFERENCE_CHUNKS,
    ChunkSizePredictor,
    chunk_alignment,
    has_sole_prefill,
)
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.multimodal_runtime import prefill_media_payload
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import (
    Replay,
    Sequence,
    SequenceStatus,
    SequenceType,
    new_token_ids,
    resumes_offload_prefill,
)
from atom.model_engine.state_runtime import (
    DEFAULT_STATE_RUNTIME,
    StateMaintenanceOps,
    StateRuntime,
)
from atom.utils import envs

if TYPE_CHECKING:
    from atom.model_engine.prefill_delayer import PrefillDelayer

logger = logging.getLogger("atom")

_DYNAMIC_CHUNKING_SUPPLY_WINDOW = 16


class ChunkKind(enum.Enum):
    """Which prefill chunk a sizing call is for, as dynamic chunking sees it.

    Required at every sizing call rather than defaulted, so a new call site has
    to say which one it is.
    """

    # A new request's first chunk: takes the next calibration sweep arm.
    FIRST = enum.auto()
    # A later chunk of a request already admitted: keeps the current arm.
    NEXT = enum.auto()
    # A prefill that must go whole (`_requires_atomic_prefill`): never limited.
    WHOLE = enum.auto()


# How often the stalled-save reconciler actually scans deferred_free_blocks. The
# engine polls KV progress every millisecond (KV_IDLE_DRAIN_INTERVAL_S), so the
# reconciler no-ops until this interval has passed.
_SAVE_RECONCILE_INTERVAL_S = 5.0

# Release attempts a deferred request may fail after its save was abandoned
# before `_reconcile_stalled_deferred_saves` calls it wedged. The reclaimer
# retries rather than force-freeing (a claimed source may be under a live RDMA
# read), so a claim nobody will ever retire no longer has a way out -- it has
# to be visible instead. Deliberately well past the abandon timeout: a P/D send
# that is merely slow must not trip it.
_RELEASE_WEDGE_ATTEMPTS = 12

# Memoised result of `_offload_max_pending_saves`, a process constant likewise.
_MAX_PENDING_OFFLOAD: int | None = None


def _offload_max_pending_saves() -> int:
    """How many offload transfers may be outstanding at once, engine-side.

    Shared with the KV leg's `OFFLOAD_MAX_PENDING_SAVES` rather than given a
    knob of its own: a KV save and a state store both hold the bytes they are
    reading out of the same pool while they run, so the queue depth is one
    question about one resource. Two independent numbers would let an operator
    raise one and unknowingly double what a slow backend can pin.
    """
    global _MAX_PENDING_OFFLOAD
    if _MAX_PENDING_OFFLOAD is not None:
        return _MAX_PENDING_OFFLOAD
    try:
        value = envs.OFFLOAD_MAX_PENDING_SAVES
    except ValueError as exc:
        logger.warning("%s; using 2 for the state tier", exc)
        value = None
    if value is None:
        value = 2
    _MAX_PENDING_OFFLOAD = max(1, value)
    return _MAX_PENDING_OFFLOAD


def _prompt_tokens_of(result) -> int:
    """Prompt tokens in whatever a `_schedule()` returned.

    The three schedulers disagree on the empty shape — bare `None` from
    `Scheduler` and `DecodeScheduler`, `(None, {})` from `PrefillScheduler` —
    so normalize here rather than at each caller. `total_tokens_num_prefill`
    is 0 on a decode batch, which is what the throughput window wants: its
    generation side is counted in `postprocess`, from tokens actually
    committed.
    """
    batch = result[0] if isinstance(result, tuple) else result
    return 0 if batch is None else batch.total_tokens_num_prefill


def _optimal_cu_fraction(
    decode_batch: int, prefill_waiting_tokens: int
) -> float | None:
    """Return the prefill CU fraction for the current workload, or None for no mask.

    Called by the DecodeScheduler, which has visibility into both the decode
    batch size and the total tokens queued in prefill_waiting.  The chosen
    fraction is written to shared memory so the PrefillScheduler can read it.

    Lookup table derived from empirical benchmarking across CU splits
    (30/50/60/70/80% prefill) on DeepSeek-R1 tp=8.  Prefill latency
    dominates in nearly all cases; decode tolerates CU reduction well
    at typical batch sizes.

    Returns None when CU masking provides no benefit (no pending prefill,
    tiny prefill).
    """
    assert prefill_waiting_tokens >= 0
    if prefill_waiting_tokens == 0 or decode_batch < 64:
        return None
    else:
        return 0.5


class ScheduledBatch:
    """Immutable snapshot of sequences selected for a single forward pass.

    Holds per-sequence metadata (block tables, context lengths, temperatures)
    and the flattened token array ready for the model runner.

    Args:
        seqs: Mapping from request ID to :class:`Sequence`.
        num_scheduled_tokens: Number of new tokens per sequence.
        total_tokens_num: Sum of all scheduled tokens (prefill + decode).
        connector_meta_output: Optional KV connector metadata for this batch.
        num_spec_step: Number of speculative decode steps (0 = disabled).
        scheduled_spec_decode_tokens: Draft token IDs per request for
            speculative decoding (must not use a mutable default).
        state_maintenance_ops: State moves that must execute before this batch.
    """

    def __init__(
        self,
        seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        total_tokens_num: int,
        total_tokens_num_prefill: int = 0,
        total_tokens_num_decode: int = 0,
        total_seqs_num: int = 0,
        total_seqs_num_prefill: int = 0,
        total_seqs_num_decode: int = 0,
        connector_meta_output=None,
        is_dummy_run: bool = False,
        num_spec_step: int = 0,
        scheduled_spec_decode_tokens: dict[int, np.ndarray] | None = None,
        cu_stream_fraction: float | None = None,
        remote_kv_block_ids: list[int] | None = None,
        remote_kv_seq_blocks: dict[int, list[int]] | None = None,
        num_cached_tokens: list[int] | None = None,
        is_final_chunk: list[bool] | None = None,
        next_token_ids: list[int] | None = None,
        state_maintenance_ops: StateMaintenanceOps | None = None,
    ):
        if scheduled_spec_decode_tokens is None:
            scheduled_spec_decode_tokens = {}
        self.remote_kv_block_ids = remote_kv_block_ids or []
        self.remote_kv_seq_blocks = remote_kv_seq_blocks or {}

        self.req_ids = list(seqs.keys())
        self.num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)
        self.temperatures = np.asarray(
            [seq.temperature for seq in seqs.values()], dtype=np.float32
        )
        self.return_logprobs = [seq.return_logprobs for seq in seqs.values()]
        # `context_lens` is set further down, once `num_cached_tokens` is known:
        # a chunked prefill's context ends at this chunk, not at the whole
        # prompt, so `seq.num_tokens` is only right for decode.
        # Kept as a list too: the offset loop below reads it per sequence, and
        # `arr[i]` builds a numpy scalar where a list index does not. The list
        # is what `np.asarray` consumes anyway, so holding onto it is free.
        num_rejected = [seq.num_rejected for seq in seqs.values()]
        self.num_rejected = np.asarray(num_rejected, dtype=np.int32)
        self.num_bonus = np.asarray(
            [seq.num_bonus_tokens for seq in seqs.values()], dtype=np.int32
        )
        # By batch index, the seqs whose forward starts below where they
        # resumed; empty in almost every batch.
        self.replaying = {
            i: seq.replay
            for i, seq in enumerate(seqs.values())
            if seq.replay is not None and seq.num_cached_tokens < seq.replay.until
        }
        # One entry per state-holding seq: that seq's whole slot set, in
        # allocation order. `[0]` is the committed state and `[1:]` is
        # speculation rollback, one slot per speculated token. The sets are not
        # adjacent and no backend may reconstruct them by arithmetic on a base
        # — see `StateSlotPool`.
        # Gated on `state_slot >= 0`, not on the list being non-empty: a seq
        # whose committed slot was never claimed carries the -1 sentinel in a
        # one-element list, which is truthy, and letting it through would shift
        # every list positionally aligned with this one.
        state_rows: list[int] = []
        state_seqs: list[Sequence] = []
        for row, seq in enumerate(seqs.values()):
            if seq.has_per_req_cache and seq.state_slot >= 0:
                state_rows.append(row)
                state_seqs.append(seq)
        # Batch row of each state-holding seq. Every list below is indexed by
        # state-holding seq; the token counts, `cu_seqlens` and the kernels'
        # rows are indexed by batch row. The two coincide only when every row
        # holds state, so a backend translating between them goes through this.
        self.state_rows = state_rows
        self.state_slots = [seq.state_slots for seq in state_seqs]
        # Column 0 broken out, because it is what every non-speculative backend
        # wants and rebuilding it per step in each of them would cost the same
        # Python loop several times over.
        self.state_slots_committed = [seq.state_slot for seq in state_seqs]
        # Read-side twin of the committed column, positionally aligned with it:
        # the slot this forward takes its incoming state from. Differs only on
        # the one forward after a state fork; -1 elsewhere, which attention
        # backends read as "same as the write slot".
        self.state_fork_srcs = [seq.state_fork_src for seq in state_seqs]
        # Midstep checkpoints this forward must write, `[(slot, position)]` per
        # seq, positionally aligned with `state_slots` like the fork sources
        # above. A list per seq, not one entry: a readable backend takes every
        # position the chunk covers rather than only the one it ends on.
        # Positions are absolute prompt offsets; the backend rebases them onto
        # the step's own tokens, which is the only frame its intermediates are
        # in. Empty everywhere except a `readable_midstep` prefill.
        self.state_save_all = [
            [(g, p) for g, p, _h in seq.midstep_reservations] for seq in state_seqs
        ]
        # Physical moves are drained once per real batch.
        self.state_maintenance_ops = (
            state_maintenance_ops
            if state_maintenance_ops is not None
            else StateMaintenanceOps()
        )
        self.top_ks = np.asarray([seq.top_k for seq in seqs.values()], dtype=np.int32)
        self.top_ps = np.asarray([seq.top_p for seq in seqs.values()], dtype=np.float32)
        # True if any seq in the batch is a fan-out child (SamplingParams.n>1)
        # and therefore requires fresh per-row random noise at the sampler
        # rather than the cached shared exponential tensor.
        self.needs_independent_noise = np.asarray(
            [getattr(seq, "needs_independent_noise", False) for seq in seqs.values()],
            dtype=bool,
        )

        self.is_first_decode_without_local_prefill = [
            seq.is_first_decode for seq in seqs.values()
        ]
        self.mrope_positions_by_req = {
            seq.id: seq.mrope_positions
            for seq in seqs.values()
            if getattr(seq, "mrope_positions", None) is not None
        }
        self.mrope_position_deltas = {
            seq.id: getattr(seq, "mrope_position_delta", 0)
            for seq in seqs.values()
            if getattr(seq, "mrope_positions", None) is not None
        }

        # num_cached_tokens for chunked prefill support
        self.num_cached_tokens = (
            num_cached_tokens
            if num_cached_tokens is not None
            else [seq.num_cached_tokens for seq in seqs.values()]
        )

        self.is_final_chunk = is_final_chunk
        # Per seq, the token following this forward where the scheduler knows it
        # (a middle prefill chunk's successor prompt token), -1 where sampling
        # supplies it. A drafter runs one position ahead of the target, so this
        # is its anchor on the chunks that never reach sampling.
        self.next_token_ids = next_token_ids

        # context_lens: for prefill seqs, use num_cached_tokens + num_scheduled_tokens
        self.context_lens = np.asarray(
            [
                (
                    self.num_cached_tokens[i] + num_scheduled_tokens[i]
                    if seq.type == SequenceType.PREFILL
                    else seq.num_tokens
                )
                for i, seq in enumerate(seqs.values())
            ],
            dtype=np.int32,
        )

        # Each sequence's window, staged into one array rather than assigned
        # per sequence: a numpy slice-assign costs ~245ns of dispatch whatever
        # its length, and at decode a window is a single token. `extend`
        # between two `array("i")` is a memcpy. 4x at bs=256.
        staged = new_token_ids()
        for i, (seq, num) in enumerate(zip(seqs.values(), num_scheduled_tokens)):
            if seq.type == SequenceType.PREFILL:
                offset = self.num_cached_tokens[i]
            else:
                offset = seq.num_tokens - num_rejected[i] - num
            staged.extend(seq.token_ids[offset : offset + num])
        # Checked here because nothing downstream will: the array below wraps
        # whatever length was staged, and a sequence too short to fill its
        # window would otherwise leave the batch quietly short.
        if len(staged) != total_tokens_num:
            raise ValueError(
                f"staged {len(staged)} tokens for a batch of {total_tokens_num}: "
                "a sequence is shorter than the window scheduled for it"
            )
        # Wrapped, not copied into a fresh array: consumers only ever read this
        # or rebind it, so the staging buffer can be the buffer.
        self.scheduled_tokens = np.frombuffer(staged, dtype=np.int32)

        if num_spec_step > 0 and scheduled_spec_decode_tokens is not None:
            # One row per sequence, in batch order. The caller's dict is keyed
            # by request id and holds only sequences that DO have drafts, so
            # densifying it with `list(...values())` drops the others and
            # shifts every later row onto the wrong sequence — consumers index
            # this by batch position (`prepare_input_ids`), which turns that
            # shift into one sequence silently reading another's drafts.
            #
            # Rows are collected and written in one assign: a per-row write
            # costs ~245ns of dispatch against a row this short, so the marshal
            # was mostly dispatch (3.7x at bs=256). Rows that are not
            # full width are padded rather than dropping the batch back to the
            # per-row form, because at a large batch a sequence joins decode
            # nearly every step and would take every other sequence with it.
            no_drafts = np.zeros(num_spec_step, dtype=np.int32)
            rows = []
            for req_id in self.req_ids:
                drafts = scheduled_spec_decode_tokens.get(req_id)
                if drafts is not None and drafts.size == num_spec_step:
                    rows.append(drafts)
                elif drafts is None or drafts.size == 0:
                    rows.append(no_drafts)
                else:  # short or long: onto a row of its own, never the shared one
                    padded = no_drafts.copy()
                    padded[: min(drafts.size, num_spec_step)] = drafts[:num_spec_step]
                    rows.append(padded)
            # `concatenate` rejects an empty list, which a warmup batch is.
            flat = (
                np.concatenate(rows, dtype=np.int32)
                if rows
                else np.empty(0, dtype=np.int32)
            )
            self.scheduled_spec_decode_tokens = flat.reshape(len(seqs), num_spec_step)
        self.block_tables = [
            seq.block_table for seq in seqs.values() if seq.block_table
        ]
        self.last_block_num_tokens = [
            _seq.last_block_num_tokens for _seq in seqs.values()
        ]

        # Total number of tokens scheduled for all requests.
        self.total_tokens_num = total_tokens_num
        self.total_tokens_num_prefill = total_tokens_num_prefill
        self.total_tokens_num_decode = total_tokens_num_decode

        # Total number of reqs scheduled for all requests.
        self.total_seqs_num = total_seqs_num
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode

        self.connector_meta_output = connector_meta_output
        self.finished_recving_kv_req_ids: list[int] = []

        self.is_dummy_run = is_dummy_run
        self.num_spec_step = num_spec_step
        # Set by the PP head on a batch whose completion it waits for although
        # the batch produces no output. It travels with the metadata, so the
        # last stage acks exactly the batches the head waits on.
        self.pp_completion_ack = False

        # Detailed attention aggregates (set by Scheduler.compute_detailed_aggregates
        # when profiling is active and ATOM_ENABLE_DETAILED_ANNOTATION is set).
        # None on the normal path; consumed by ModelRunner.run_model to extend
        # the prefill[]/decode[] trace labels.
        self.detailed_sqsq: int | None = None  # sum N_Q^2
        self.detailed_sqsk: int | None = None  # sum N_Q*N_KV
        self.detailed_sk: int | None = None  # sum N_KV

        # Key into ModelRunner's stream pool for CU-masked disagg streams.
        # None means full-CU fallback (no mask).
        self.cu_stream_fraction = cu_stream_fraction
        # Explicit spans support arbitrary chunks. Retain the CPU payload on
        # the request for retry, and omit pixels after workers acknowledge it.
        self.multimodal_data = {}
        for seq in seqs.values():
            data = getattr(seq, "multimodal_data", None)
            if data is None or seq.type != SequenceType.PREFILL:
                continue
            if "embedding_spans" in data:
                self.multimodal_data[seq.id] = prefill_media_payload(
                    data, seq.cache_seed, cache_ready=seq.multimodal_cache_ready
                )
            else:
                self.multimodal_data[seq.id] = data
                seq.multimodal_data = None
        self.external_request_ids = [seq.external_request_id for seq in seqs.values()]

        # Metrics reporting only: per-request prefill chunks for GPU timing.
        # Device timing is opt-in because recording GPU events adds overhead,
        # especially for short decode forwards; skip this bookkeeping when off.
        # Freeze request chunk boundaries before schedule-time advancement.
        # Workers receive copies of the batch, not the mutable Sequence objects.
        self.prefill_gpu_requests = []
        if (
            not is_dummy_run
            and total_seqs_num_prefill
            and envs.ATOM_ENABLE_METRICS_DEVICE_TIMER
        ):
            for i, seq in enumerate(seqs.values()):
                if i < total_seqs_num_decode or seq.prefill_gpu_complete:
                    continue
                # Re-prefill after generation has started is not initial prefill
                # (also excludes D recomputation following a remote prefill).
                if seq.num_tokens > seq.num_prompt_tokens:
                    continue
                final = (
                    bool(is_final_chunk[i])
                    if is_final_chunk is not None
                    else self.num_cached_tokens[i] + int(num_scheduled_tokens[i])
                    >= seq.num_prompt_tokens
                )
                seq.prefill_gpu_chunks += 1
                seq.prefill_gpu_complete = final
                self.prefill_gpu_requests.append(
                    (seq.id, seq.prefill_gpu_chunks, final)
                )

        # logger.info(f"{[el for el in scheduled_spec_decode_tokens.keys()]=}")
        # logger.info(f"{self.num_scheduled_tokens=}")
        # logger.info(f"{self.context_lens=}")
        # logger.info(f"{[len(blk)*16 for blk in self.block_tables]=}")
        # logger.info(f"{self.block_tables=}")

    def produces_output(self) -> bool:
        """True if this batch yields a token the head must consume.

        Decode batches always do. A pure-prefill batch yields a token only
        when at least one seq is on its final chunk; a batch of middle chunks
        produces nothing.

        A DP-sync dummy answers True and must: reporting lags a step, so the
        dummy is what flushes the last real one. Skip its `postprocess` and
        the request waking up after it has an anchor on neither side.
        """
        if self.total_seqs_num_decode > 0:
            return True
        if self.is_final_chunk is None:
            return True
        return any(self.is_final_chunk)


def _cut_floor(seq: Sequence, start: int, resume: int) -> int:
    """Where a checkpoint cut of a forward from `start` must land above: where
    the seq resumed (`resume`, or the `until` of the replay the forward
    continues), since below it a forward recomputing cached rows
    (`BlockManager.replay_start`) holds no state worth keeping. A DeepSeek-V4
    bounded replay has rebuilt the state whole by that point, so a cut may land
    right on it, and a later branch at the same hit restores from there instead
    of replaying again."""
    floor = max(start, resume, seq.replay_until)
    if seq.replays_hit and floor > start:
        floor -= 1
    return floor


def _scheduled_replay_until(batch, i: int) -> int:
    """Where batch entry `i`'s forward resumed if it started below that point
    (`ScheduledBatch.replaying`), as scheduled, 0 otherwise. Under pipeline
    parallelism a later chunk's scheduling may already have moved the live
    `Sequence.replay` by the time this batch is post-processed."""
    replay = (getattr(batch, "replaying", None) or {}).get(i)
    return 0 if replay is None else replay.until


class ScheduledBatchOutput:
    """Token-level results from a single forward pass.

    Attributes:
        token_ids: Mapping of request ID -> accepted token IDs.
        draft_token_ids: Speculative draft tokens (one row per request).
        num_rejected: Per-request count of rejected speculative tokens.
        num_bonus: Per-request count of bonus accepted tokens.
        is_deferred_out: Whether output was deferred from a previous step.
    """

    def __init__(
        self,
        req_ids: list[int],
        token_ids: list[tuple[int, ...]],
        num_rejected: np.ndarray | None,
        num_bonus: np.ndarray | None,
        draft_token_ids: np.ndarray | None,
        is_deferred_out: bool = False,
        is_prev_prefill=False,
        logprobs=None,
        dspark_ell: np.ndarray | None = None,
    ):
        self.req_ids = req_ids
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        self.is_deferred_out = is_deferred_out
        self.is_prev_prefill = is_prev_prefill
        self.logprobs = logprobs  # Optional[dict[int, float]]
        # DSpark Phase 2: {req_id: ell_r} from this step's propose() — the
        # scheduler-chosen verify length per request. Rides back to the
        # (main-process) scheduler so the NEXT step can size each request's
        # verification to ell_r+1. None when DSpark scheduling is off.
        self.dspark_ell = dspark_ell
        # O(1) lookup: req_id -> index (lazy-built on first access)
        self._req_id_to_idx: dict[int, int] | None = None

    def get_idx(self, req_id: int) -> int | None:
        """O(1) lookup of request index by id."""
        if self._req_id_to_idx is None:
            self._req_id_to_idx = {rid: i for i, rid in enumerate(self.req_ids)}
        return self._req_id_to_idx.get(req_id)


# =====================================================================================
# Scheduler：主调度器——每个engine step决定"这一步算哪些序列、各算多少token"
# =====================================================================================
# 架构位置（每个DP rank一个EngineCore进程，进程里只有busy_loop线程访问调度器，见engine_core.py）：
#   CoreManager ══ZMQ══> EngineCore.busy_loop，每轮：
#     scheduler.extend(新请求)             新请求只在每轮循环开头进入waiting
#     batch, seqs = scheduler.schedule()   本类：挑序列、分配KV block、打包ScheduledBatch；没有任何请求时返回None
#     scheduler.take_rejected()            准入时被拒绝（永远放不下 / 还没运行就被abort）的请求，直接作为"已结束"输出
#     out = ModelRunner.forward(batch)     所有TP rank执行同一个batch，只有rank 0回传ScheduledBatchOutput
#     scheduler.postprocess(seqs, out)     追加token、判断结束、释放block（延迟输出：拿到的是上一步的token）
#   每步的输入输出约定见engine_core.py的_process_engine_step_inner；连续批处理、prefill计算受限 / decode带宽受限、
#   延迟输出的原理见engine_core.py文件开头【1】【2】【3】，这里不再重复
#
# 调度策略（基础调度器；RapidServe（同一组GPU上的PD分离）用的PrefillScheduler / DecodeScheduler在文件末尾，【分支】第9批）：
#   1. prefill优先：只要有能做的prefill，这一步就只做prefill；一个也没有才做decode。
#      一个ScheduledBatch要么全是prefill、要么全是decode，从不混合（源码里留着"TODO for prefill/decode mixed batch"）
#   2. prefill分两段：Phase 1先续算running里做到一半的chunked prefill（is_partial_prefill=True），
#      Phase 2再按FIFO从waiting队头接纳新请求；队头放不下就停（break），不跳过它去接后面的，避免长请求被饿死
#      （例外：等前缀checkpoint / 等外部KV查询的请求可以暂时让路，此后最多再看16条，见_schedule）
#   3. chunked prefill：一条prompt放不进本步剩余的token预算时，切出放得下的那段（按max(block_size, 64)向下对齐），
#      剩下的留在running里由下一步的Phase 1续算。哪一条都可以被切，不限于本步第一条
#   4. decode：running里每条序列这一步算1个token（投机解码时1+mtp_k个）；KV block不够时从running队尾开始抢占
#      （重算式：释放block、放回waiting队头，以后重新prefill，见sequence.py的"科普：抢占"）
#   5. 永远放不下的prompt（_unschedulable_reason）不让它堵在队头，直接以FINISHED拒绝，经_rejected交给EngineCore
#   6. 可选的prefill合批（PrefillDelayer，prefill_delayer.py）：把零碎的小prefill攒成一批再放行；DP>1时各rank锁步决定
#   每一步完整的决策流程图见_schedule开头
#
# 两个预算——每一步的容量上限（默认值与调参建议见arg_utils.py EngineArgs的同名字段和该类上方的总览表）：
#   max_num_batched_tokens（默认16384）：一次forward最多计算多少个token
#     prefill：本步各序列的chunk之和 <= 16384；decode：本步各序列的(1+mtp_k)之和 <= 16384
#     decode通常碰不到它（512条 x (1+3) = 2048），但预算调小时会：GLM-5.2的agentic recipe用2048 + MTP3，
#       一步最多 2048 / 4 = 512 条decode
#   max_num_seqs（默认512）：running里同时驻留的序列数上限，也是一步prefill / decode的序列数上限
#     Phase 2发现 len(running) >= max_num_seqs 就不再接纳新请求，哪怕token预算还有剩
#   取舍（70B稠密FP8模型、MI300X理论峰值的粗估，口径同engine_core.py【2】：FP8稠密约2600 TFLOPS、HBM约5.3 TB/s，
#     忽略attention和KV读取，只用来看量级）：
#     token预算     一个满的prefill步耗时                   影响
#       2048        2 x 70e9 x 2048 / 2.6e15 ≈ 0.11s        已远超"约245个token转为计算受限"的分界点，算力基本吃满；长prompt切得更多
#       16384       ≈ 0.88s                                 固定开销摊得更薄；但激活显存更大（可用KV block变少），
#                                                           开了decode保护间隔时，每次prefill打断decode的时间也更长
#     seq预算（decode步：读一遍70GB权重 ≈ 13ms，这是不随batch变的下限）
#       B=64        计算≈3.4ms  → 一步≈13ms，约4900 token/s   带宽受限：B翻倍几乎白赚
#       B=256       计算≈13.8ms → 一步≈14ms，约18500 token/s  到了分界点附近
#       B=512       计算≈27.6ms → 一步≈28ms，约18500 token/s  吞吐不再涨，每个请求的ITL却翻倍
#     而且B实际上还被KV容量限住：每条序列都占KV block，每步还要读一遍它的KV；KV不够时就会抢占
#   注意：一步只做prefill或decode，且做到一半的chunked prefill下一步优先续算，所以单靠调小token预算并不能让decode
#     插进一条长prompt的chunk之间（只是把它切成更多步连续算完），见arg_utils.py EngineArgs上方总览表的注①
#
# 调度器持有的容器（序列的状态机见sequence.py SequenceStatus上方的状态流转图；这里只画"序列此刻在哪个容器里"）：
#
#   add / extend          Phase 2接纳（分配KV block）            postprocess判定结束（eos/stop/max_tokens/abort）
#   ────────────> waiting ────────────────────────> running ─────────────────────┬──> deallocate，block回空闲队列
#                 (deque)  <──────────────────────  (deque)                      │
#                   │  ▲      preempt：放回队头       ↺ 做到一半的prefill留在这里   └──> 【分支】connector还在读这些block
#                   │  │                              （is_partial_prefill）            （PD发送 / offload保存）：
#                   │  │                                                             deferred_free_blocks，读完再释放
#                   │  └── 【分支】等外部KV的请求（WAITING_FOR_REMOTE_KVS）也挂在waiting里，传完再接纳
#                   └────> _rejected（永远放不下，或还没运行就被abort）──> take_rejected()，作为已结束输出
#
# 本类的阅读路线（按代码顺序）：
#   ① __init__：预算、队列和各种【分支】状态，逐个说明
#   ② 给PrefillDelayer回答"本步有没有prefill可做、攒了多少"：_waiting_prefills、_local_prefill_pending_work /
#      _probe_local_prefill_work（TP本地合批）、_can_admit_head_prefill（DP）、_waiting_new_token_count、
#      _partial_prefill_remaining_tokens、_oldest_waiting_prefill_age_ms、_kv_usage
#   ③ chunk的起点与大小：_new_prefill_tokens（long_prefill_token_threshold）、_partial_prefill_chunk、_resume_prefill（有界重放）；
#      DP下各rank的prefill成本均衡：_chunk_cost、_prefill_room、_estimate_prefill_chunk_cost、_balance_prefill_step
#   ④ 准入拒绝：_requires_atomic_prefill、_unschedulable_reason、_warn_if_unschedulable、take_rejected
#   ⑤ 【分支】connector的延迟释放：_connector_*、_maybe_release_deferred、_reconcile_stalled_deferred_saves（第9批）
#   ⑥ schedule / _schedule（主干，开头有决策流程图）→ preempt → postprocess → _mark_finished → abort_request
#   第一遍建议按 ① → ④ → ⑥ 读，②③⑤ 遇到调用时再回来查
#
# 对比nano-vllm（nanovllm/engine/scheduler.py）：
#   相同：prefill优先、一步不混合prefill和decode、FIFO接纳、重算式抢占、两个预算的含义
#   不同：
#     - 状态：nano-vllm的序列prefill全部算完才进running；ATOM第一次被调度做prefill就变成RUNNING、进running，
#       没算完的用is_partial_prefill标记，由Phase 1续算
#     - chunked prefill：nano-vllm只允许本步第一条序列被切；ATOM哪一条都能切，还能用long_prefill_token_threshold
#       限制单条请求每步的量
#     - 抢占：nano-vllm总能抢；ATOM里connector还在读block的序列不能被抢（_is_preemptable）
#     - 永远放不下的请求：nano-vllm不做检查（会一直卡在waiting队头）；ATOM提前拒绝
#     - 另外多出：延迟输出的占位token、投机解码（每条decode 1+mtp_k个token）、前缀缓存 + 每请求状态的checkpoint /
#       有界重放、PrefillDelayer合批与DP锁步，以及【分支】PD分离 / KV offload / PP / 动态分块
class Scheduler:
    """Manages the lifecycle of inference requests through prefill and decode.

    The scheduler maintains two primary queues:

    - **waiting**: Newly arrived requests pending their first prefill.
    - **running**: Active requests that have completed prefill and are
      being decoded token-by-token.

    On each :meth:`schedule` call it selects a batch of sequences that
    fit within the token and sequence budget, allocates KV cache blocks
    via :class:`BlockManager`, and returns a :class:`ScheduledBatch`.

    Integration with the KV disaggregation connector is handled through
    :meth:`_update_waiting_for_remote_kv` (decode side) and
    :meth:`_update_from_kv_xfer_finished` (both sides).
    """

    # 【译】管理推理请求经过prefill和decode的生命周期。
    #
    #   调度器维护两个主要队列：
    #   - waiting：新到达、等待第一次prefill的请求。
    #   - running：已完成prefill、正在逐token decode的活跃请求。
    #
    #   每次调用 schedule 时，它选出一批放得进token预算和序列预算的序列，通过 BlockManager 分配KV cache block，
    #   并返回一个 ScheduledBatch。
    #
    #   与KV分离（disaggregation）connector的集成通过 _update_waiting_for_remote_kv（decode一侧）
    #   和 _update_from_kv_xfer_finished（两侧）完成。
    # 核对代码：running里还有做到一半的chunked prefill（is_partial_prefill=True），并非只有"已完成prefill"的请求；
    #   被抢占的序列和等外部KV的序列也在waiting里（见类上方的容器图）

    # 两个给子类覆盖的标签：_ENGINE_LABEL是引擎状态行的前缀（EngineStats的label），_METRICS_ROLE进指标标签engine_role
    #   以及EngineUtilityHandler.collect_metrics的"role"字段；子类DecodeScheduler（RapidServe的decode一侧，【分支】）
    #   设为"Decode " / "decode"；PrefillScheduler不是本类的子类，它自己定义_METRICS_ROLE="prefill"
    _ENGINE_LABEL = ""
    _METRICS_ROLE = ""

    # 构造：EngineCore.__init__在ModelRunner算出num_kvcache_blocks、回填进Config之后才创建调度器（engine_core.py），
    #   state_runtime由EngineCore传入（状态缓存的运行时，第4批）
    # 成员按用途分组（nano-vllm的__init__只有max_num_seqs、max_num_batched_tokens、eos、block_size、block_manager、
    #   waiting、running这几个）：
    #   ① 两个预算  ② DP prefill成本均衡  ③ 单条prefill的上限与停止条件  ④ 队列、BlockManager与指标  ⑤ 准入拒绝
    #   ⑥【分支】KV传输记账  ⑦ 攒批延迟（scheduler_delay_factor）  ⑧ 投机解码  ⑨ 统计计数
    #   ⑩ chunked prefill与【分支】动态分块  ⑪【分支】PP  ⑫ connector、KV事件、EngineStats  ⑬ PrefillDelayer
    # getattr(对象, 属性名, 默认值)：属性不存在时返回默认值；这里大量使用，是为了让测试里的简化Config也能构造调度器
    def __init__(
        self,
        config: Config,
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        # ---- ① 预算 ----（含义、默认值和取舍的数字见类上方"两个预算"）
        # 序列预算：running里最多驻留多少条序列，也是一步最多调度多少条（prefill、decode都受它约束）
        self.max_num_seqs = config.max_num_seqs
        # token预算：一次forward最多计算多少个token（prefill是各chunk之和，decode是各序列的1+mtp_k之和）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # ---- ② DP prefill成本均衡 ----
        # DP prefill cost balancing, on whenever DP attention runs more than
        # one rank: every rank estimates the cost of its chunk at the whole
        # batch budget; the ranks then aim at one cost target, a costlier rank
        # shrinking its chunk. The costs ride the PrefillDelayer's all_reduce,
        # so balancing needs the delayer (ATOM_ENABLE_PREFILL_DELAYER).
        # 【译】DP prefill成本均衡：只要DP attention运行着多于一个rank就开启：每个rank按整份batch预算估算自己chunk的成本；
        #   然后各rank瞄准同一个成本目标，成本较高的rank缩小自己的chunk。成本搭PrefillDelayer的all_reduce传递，
        #   所以均衡需要delayer（ATOM_ENABLE_PREFILL_DELAYER）。
        # 为什么需要：DP attention下各rank的prefill在同一个锁步step里执行（forward里有跨rank的MoE集合通信，
        #   见engine_core.py DPEngineCoreProc上方的说明），一步的时长取决于最慢的rank。token数相同时，
        #   chunk前面已有很长上下文的rank注意力更贵，就成了拖后腿的那个。成本公式与带数字的例子见_balance_prefill_step
        # dp_size是改写后的值：--enable-dp-attention会把data_parallel_size乘上原来的-tp（engine_core_mgr.py），
        #   所以 -tp 8 --enable-dp-attention 时这里是8，cost_balance为True
        dp_size = getattr(
            getattr(config, "parallel_config", None), "data_parallel_size", 1
        )
        self.cost_balance = (
            getattr(config, "enable_dp_attention", False) and dp_size > 1
        )
        # 成本公式里每token的固定权重W（默认360000）、允许超出平均成本的比例（默认0.1）、被削减时至少保留的token数（默认4096），
        #   定义见envs.py的ATOM_DP_PREFILL_BALANCE_*
        self._balance_token_weight = envs.ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT
        self._balance_slack = envs.ATOM_DP_PREFILL_BALANCE_SLACK
        self._balance_min_tokens = envs.ATOM_DP_PREFILL_BALANCE_MIN_TOKENS
        # This tick's prefill cost limit and what admission has spent of it;
        # see `_prefill_room`.
        # 【译】本tick的prefill成本上限，以及接纳已经花掉了多少；见 _prefill_room。
        # 每次_schedule开头重置为(inf, 0)；_balance_prefill_step算出目标后改写上限，_schedule_prefill_seq和Phase 1累加花费
        self._tick_cost_limit = math.inf
        self._tick_cost = 0.0
        # steps, tokens, own cost, target, steps shrunk
        # 【译】步数、token数、自身成本、目标、被削减的步数（_log_prefill_balance每50步打印一次）
        self._balance_stats = [0, 0, 0.0, 0.0, 0]
        # ---- ③ 单条prefill的上限与停止条件 ----
        # 单条请求每步最多prefill多少token（--long-prefill-token-threshold，默认0=不限），用法和例子见_new_prefill_tokens
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        # 最大上下文长度：超过它的prompt永远调度不了（_unschedulable_reason）
        self.max_model_len = config.max_model_len
        # 观察：bos_token_id在本文件里只被赋值、没有读取方
        self.bos_token_id = config.bos_token_id
        # postprocess读：判断是否采到EOS（结束原因"eos"）；它也是延迟输出的占位token——postprocess先往token_ids末尾
        #   追加eos占位、下一步原地覆盖，preempt按num_placeholder_tokens删掉没被覆盖的占位
        self.eos_token_id = config.eos_token_id
        # 其他停止token（来自generation_config；LLMEngine.__init__已从中去掉tokenizer的eos，所以结束原因分别报"eos"和"stop_<id>"）
        self.stop_token_ids = config.stop_token_ids
        # ---- ④ 队列、BlockManager与指标 ----
        # KV block的CPU侧账本：分配/释放、前缀缓存、每请求状态的checkpoint（第4批block_manager.py）；
        #   num_kvcache_blocks、block_size都是Config改写后的值（V4=256、V4.1=128、GLM-5.3=index_kpool×16，见config.py）
        self.block_manager = BlockManager(
            config,
            state_runtime=state_runtime,
        )
        # deque：两端增删都是O(1)的双端队列；waiting的队头是下一个要接纳的请求（preempt用appendleft插回队头）
        self.waiting: deque[Sequence] = deque()
        # 正在执行的序列：做到一半的prefill + decode中的序列，Phase 1和decode循环都扫它
        self.running: deque[Sequence] = deque()
        self.config = config
        # tokens a replaying forward carries from before it (`Replay.history_ids`)
        # 【译】一次重放forward要带上的、它起点之前的token数（Replay.history_ids）
        # 【分支】DeepSeek-V4.1的Engram：= engram_max_ngram_size - 1，其他模型为0；_resume_prefill用它截取history_ids（第8批）
        self.replay_history_tokens = getattr(config, "replay_history_tokens", 0)
        # Prometheus指标（排队时间、decode batch大小、PD传输耗时等直方图，atom/metrics/scheduler.py，第10批）；
        #   第一个参数是引擎编号 = dp_rank × pp_size + pp_rank，保证每个DP rank的每个PP stage各有一组标签
        pc = getattr(config, "parallel_config", None)
        self.metrics = SchedulerMetrics(
            getattr(pc, "data_parallel_rank", 0)
            * getattr(config, "pipeline_parallel_size", 1)
            + getattr(pc, "pipeline_parallel_rank", 0),
            self._METRICS_ROLE or "default",
        )

        # ---- ⑤ 准入拒绝 ----
        # Admit-rejected seqs (those `_unschedulable_reason` flags). Drained
        # by `take_rejected` each EngineCore step; routed through the same
        # output_queue path as forward-finished seqs.
        # 【译】准入时被拒绝的seq（_unschedulable_reason 标记的那些）。每个EngineCore step由 take_rejected 取空；
        #   和forward完成的seq走同一条 output_queue 路径。
        # 核对代码：还没运行就被abort的waiting序列（_reject_aborted_waiting，leave_reason="aborted"）也放进这里
        self._rejected: list[Sequence] = []

        # ---- ⑥【分支】KV传输记账（PD分离 / KV offload，第9批；PD、offload是什么见sequence.py SequenceStatus类内的科普）----
        # KV transfer bookkeeping
        # 【译】KV传输的记账
        # 外部KV已收齐 / 接收失败的请求id：_update_from_kv_xfer_finished写入；下一次_schedule时，开头的
        #   _promote_ready_remote_kv_requests先把这些挂起的请求挪到waiting前面，Phase 2再经_resolve_waiting_remote_kv消费：
        #   收齐的转回可调度；失败的，offload改为本地重算，PD的decode端则abort（见sequence.py状态流转表的②⑧）
        self.finished_recving_kv_req_ids: list[int] = []
        self.failed_recving_kv_req_ids: list[int] = []
        # {seq_id: seq}：已经结束、但connector还在读它的block（PD发送 / offload保存未完成），所以暂不释放的请求；
        #   为什么需要见下面_deferred_sequence上方的说明；is_finished()也把它算进去
        self.deferred_free_blocks: dict[int, Sequence] = {}
        # Reclamation bookkeeping for offload saves whose completion is never
        # reported (see `_reconcile_stalled_deferred_saves`).
        # 【译】给"完成报告一直没来"的offload保存做回收用的记账（见 _reconcile_stalled_deferred_saves）。
        self._abandoned_saves: int = 0
        # Set by the PP head: a lost stage report leaves a partial quorum in
        # its `PPKVAggregator`, which the scheduler owns no handle on. A direct
        # callback rather than a buffer -- under `pp_size == 1` nobody sets it,
        # and an unread buffer is its own slow leak.
        # 【译】由PP head设置：丢失的stage报告会在它的 PPKVAggregator 里留下一个不完整的法定数，而调度器拿不到那个对象。
        #   用直接回调而不是缓冲区——pp_size == 1 时没人设置它，一个没人读的缓冲区本身就是缓慢的泄漏。
        # （设置方：pp_engine_core.py里 self.scheduler.on_save_abandoned = self._forget_pp_save_quorum）
        self.on_save_abandoned = None
        # 下一次真正扫描卡住保存的时间（time.monotonic()），_reconcile_stalled_deferred_saves自我节流用
        self._next_save_reconcile_at: float = 0.0

        # ---- ⑦ 攒批延迟（--scheduler-delay-factor，默认0=关闭；逻辑在_passed_delay，借鉴vLLM PR #3279）----
        # Scheduling delay for batching efficiency
        # 【译】为提高批处理效率而做的调度延迟
        # delay_factor>0时，Phase 2只有在"waiting里最早到达的请求已等了超过 delay_factor × last_prompt_latency"
        #   或running为空时才接纳新请求。例：上一个prefill步约200ms、因子0.5 → 新请求至少攒100ms再一起prefill
        #   好处：一次prefill装更多请求，decode被打断的次数更少；代价：TTFT变长。它和PrefillDelayer是两套独立的机制
        # prev_time：上一次调用_passed_delay的时刻（time.time()）
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        # 【译】上一步是否调度了prompt（prefill）？（_schedule排出prefill batch时置True）
        self.prev_prompt = False
        # Latency of the last prompt step
        # 【译】上一个prompt步的延迟
        #   （_passed_delay在prev_prompt为True时用 now - prev_time 更新它，即从上一个prefill步的接纳到下一次接纳之间的墙钟间隔）
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor

        # ---- ⑧ 投机解码（MTP / EAGLE3 / DSpark，入门见arg_utils.py EngineArgs.method上方的【科普】）----
        #   投机解码下decode每条序列每步占 1+mtp_k 个token预算：1个上一步确定的token（锚点）+ mtp_k个草稿token一起验证
        #   例：mtp_k=3、512条decode → 一步2048个token
        # Speculative decoding. A P/D producer is excluded for the same reason
        # `PrefillScheduler` is: it prefills, samples T0 and hands the request
        # off, so it never proposes. That holds for every producer, so this
        # does not follow the runner's deferred-output decision, which
        # additionally asks whether the model keeps recurrent state.
        # Leaving speculation on here would still pad every sequence with
        # `mtp_k` placeholders and size its decode window to `mtp_k + 1`, and
        # the padding holds `num_tokens` `mtp_k` short of `max_tokens` so the
        # handoff never fires -- the producer keeps decoding a request whose
        # drafts do not exist. The drafter itself stays loaded: the producer
        # still writes the draft's context KV into the shared pool at prefill.
        # 【译】投机解码。P/D的producer被排除在外，原因和 PrefillScheduler 一样：它prefill、采样出T0，然后把请求交出去，
        #   所以它从不提议草稿。这对每个producer都成立，所以这里不跟随runner关于延迟输出的决定——后者还要额外看模型
        #   是否保留递归状态。在这里让投机保持开启的话，仍会给每条序列填上 mtp_k 个占位符、把decode窗口设为 mtp_k + 1，
        #   而这些填充让 num_tokens 始终比 max_tokens 少 mtp_k 个，于是交接永远不会触发——producer一直在decode一个
        #   草稿根本不存在的请求。草稿模型本身仍然加载：producer在prefill时仍把草稿的上下文KV写进共享池。
        is_pd_producer = kv_config_has_producer(
            getattr(config, "kv_transfer_config", {}) or {}
        )
        # use_spec：本引擎是否做投机解码（配置了speculative_config，且不是PD的producer）
        has_spec = config.speculative_config is not None
        self.use_spec = has_spec and not is_pd_producer
        # mtp_k：每条decode序列每步附带的草稿token数k（--num-speculative-tokens）；不做投机时为0
        self.mtp_k: int = (
            config.speculative_config.num_speculative_tokens if self.use_spec else 0
        )  # type: ignore
        # Successor-token metadata belongs to the drafter, not to whether this
        # engine verifies drafts. A P/D producer keeps the drafter loaded and
        # still writes its context KV, but `use_spec` stays off so it never
        # pads or verifies. Gating this on `use_spec` left EAGLE's middle
        # chunks without `next_token_ids`: compute_draft_kv returned without
        # writing draft KV, and the DP alignment pass kept its dummy anchor.
        # DSpark drafts from aux hidden states and does not read the successor.
        # 【译】后继token的元数据属于草稿模型，与本引擎是否验证草稿无关。P/D producer 仍加载着草稿模型、仍写它的上下文KV，
        #   但 use_spec 保持关闭，所以它从不填充占位、也不验证。若按 use_spec 控制这一项，EAGLE的中间chunk就拿不到
        #   next_token_ids：compute_draft_kv 不写草稿KV就返回了，DP对齐那一遍也保留着它的dummy锚点。
        #   DSpark从辅助hidden state起草，不读后继token。
        # 解释：MTP/EAGLE的草稿层在位置i的输入是"位置i的hidden state + 第i+1个token的embedding"。一个prefill chunk
        #   [s, e)的最后一个位置e-1需要第e个token——它属于下一个chunk、本步还没算，但prompt里早已知道，
        #   所以_schedule为每个prefill chunk填next_token_ids = token_ids[e]（chunk算到序列末尾时填-1）
        self.drafter_needs_next_token = has_spec and not (
            config.speculative_config.use_dspark()
        )
        # True when this engine both drafts and verifies; False under PP
        # (which only drafts for handoff to the decode node).
        # 【译】本引擎既起草又验证时为True；PP下为False（PP只为交给decode节点而起草）。
        # 为False时decode宽度退回1：_schedule里spec_width = mtp_k if spec_decode_local else 0
        pp_size = getattr(config, "pipeline_parallel_size", 1)
        self.spec_decode_local = self.use_spec and pp_size == 1
        # Positions past `len(seq)` each forward's draft round writes KV to.
        # Allocated before the forward, with the step's own blocks: see
        # `draft_kv_lookahead` for the arithmetic and what an unallocated one
        # costs.
        # 【译】每次forward的草稿轮次会写KV的、超出 len(seq) 的位置数。它在forward之前和本步自己的block一起分配：
        #   怎么算、没分配会有什么代价，见 draft_kv_lookahead。
        # 数值 = max(0, mtp_k - 1)：例如mtp_k=3时草稿要多写2个位置；没分配的位置会读到block table里补零的尾巴、
        #   写进物理block 0——那是别的请求的KV。所以can_append / may_append都要把它算进去（同时交给BlockManager）
        self.draft_lookahead = draft_kv_lookahead(self.mtp_k)
        self.block_manager.draft_lookahead = self.draft_lookahead
        if self.use_spec and not self.spec_decode_local:
            logger.info(
                "Speculative decoding: drafting only (pipeline_parallel_size=%d). "
                "Drafts are produced for handoff; this engine verifies none.",
                pp_size,
            )
        # ---- ⑨ 统计计数 ----
        # `engine_stats` (spec / cache / throughput sections) is constructed
        # below, once `dp_rank` — the throughput section's engine index — is
        # known.
        # Dashboard counters update only at request lifecycle boundaries.
        # 【译】engine_stats（投机 / 缓存 / 吞吐三部分）在下面构造，要等吞吐部分用作引擎编号的 dp_rank 确定之后。
        #   看板计数器只在请求生命周期的边界处更新。
        # 边界：_mark_finished（请求结束时累加完成数、prompt/生成token数）和preempt（抢占次数）；
        #   读取方：EngineUtilityHandler.collect_metrics（engine_utility.py）
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_finished_requests = 0
        self.total_preemptions = 0
        # profiler是否在运行：engine_utility.py的start/stop_profile翻转它；为True且开了ATOM_ENABLE_DETAILED_ANNOTATION时，
        #   compute_detailed_aggregates给每个batch附加注意力统计量（Σ N_Q²等），供trace标注
        self.profile_active = False
        # Cache the env flag once (env vars are fixed at process start) so the
        # per-iteration compute_detailed_aggregates never pays an os.getenv.
        # 【译】把环境变量标志缓存一次（环境变量在进程启动时就固定了），这样每次迭代的 compute_detailed_aggregates
        #   都不必付出 os.getenv 的开销。（envs.X每次访问都会重新读os.environ，见envs.py文件头）
        self._detailed_annotation_enabled = envs.ATOM_ENABLE_DETAILED_ANNOTATION

        # ---- ⑩ chunked prefill（切块prefill）与【分支】动态分块 ----
        # 默认开；关掉后不再切块，超过token预算的prompt会被_unschedulable_reason拒绝
        self.enable_chunked_prefill = config.enable_chunked_prefill
        # 【分支】动态分块（PP下让一条长prompt各chunk的耗时接近相等，--enable-dynamic-chunking且pp>1时才生效），第9批；
        #   原理见engine_core.py里"【科普】动态分块"（ModelRunner启动扫描处）。从这里到_dynamic_chunking_counts都是它的状态
        self.dynamic_chunking_smooth_factor = config.dynamic_chunking_smooth_factor
        self.dynamic_chunking_min_chunk_size = config.dynamic_chunking_min_chunk_size
        self.enable_dynamic_chunking = config.enable_dynamic_chunking
        # Installed after runtime calibration; None keeps fixed/sweep chunking.
        # 【译】运行时校准完成后安装；为None时保持固定分块 / 扫描分块。（install_chunk_latency_model安装）
        self.dynamic_chunk_predictor: ChunkSizePredictor | None = None
        # 校准期：新请求的首chunk在两种尺寸（两个"臂"：整份预算、预算的1/CALIBRATION_SWEEP_RATIO=1/4）之间交替，
        #   让worker测到不同大小的chunk耗时，用来拟合延迟模型的注意力项（_calibration_sweep_chunk）
        self._calibrating = self.enable_dynamic_chunking
        # Sweep arm of the request admitted last; a new request takes the other.
        # 【译】最近一个被接纳的请求用的扫描臂；新请求取另一个。
        self._calibration_small_arm = False
        # Whether the latest first-chunk sizing drew a sweep arm, so that
        # admitting that request moves the sweep on (see _commit_calibration_arm).
        # 【译】最近一次给首chunk定尺寸时是否抽取了扫描臂；这样只有真正接纳了该请求，扫描才前进（见 _commit_calibration_arm）。
        self._calibration_arm_drawn = False
        # Per-step prefill census (`_record_prefill_supply`); dynamic sizing
        # needs at most one prefill source over the whole window.
        # 【译】每步的prefill普查（_record_prefill_supply）；动态定尺寸要求整个窗口内最多只有一个prefill来源。
        #   （窗口长度_DYNAMIC_CHUNKING_SUPPLY_WINDOW=16步；deque(maxlen=N)满了以后再append会自动挤掉最老的一项）
        self._recent_prefill_supply: deque[int] = deque(
            maxlen=_DYNAMIC_CHUNKING_SUPPLY_WINDOW
        )
        # Sizing decisions by outcome, reported beside the engine-status line.
        # 【译】按结果分类的定尺寸决策计数，与引擎状态行一起报告。（Counter：缺省值为0的计数字典）
        self._dynamic_chunking_counts: Counter[str] = Counter()
        if self._calibrating and self._calibration_arms_collapsed():
            # The two sweep arms are the same aligned size, so the fit can
            # never see MIN_CALIBRATION_CHUNK_SIZES distinct chunks. Give up
            # now instead of polling for a model that cannot be identified.
            # 【译】两个扫描臂对齐后是同一个尺寸，所以拟合永远看不到 MIN_CALIBRATION_CHUNK_SIZES 个不同的chunk。
            #   现在就放弃，而不是一直轮询一个无法辨识的模型。
            # 例：--max-num-batched-tokens 4096、--dynamic-chunking-min-chunk-size 4096 → 小臂max(4096/4, 4096)=4096 = 大臂
            logger.warning(
                "Dynamic chunking calibration sweep has only one chunk size "
                "at max_num_batched_tokens=%d, min_chunk_size=%d; chunking "
                "stays fixed. Set --dynamic-chunking-min-chunk-size below "
                "--max-num-batched-tokens to enable it",
                self.max_num_batched_tokens,
                self.dynamic_chunking_min_chunk_size,
            )
            self._calibrating = False
        # Running seqs currently mid-prefill. Excludes `waiting`. The counter
        # lets schedule() skip the running-queue scan on pure-decode steps, and
        # `_record_prefill_supply` adds the admittable waiting seqs to it.
        # 【译】running里当前做到一半的prefill序列数，不含waiting。有了这个计数，纯decode步时schedule()可以跳过
        #   对running队列的扫描；_record_prefill_supply 会再加上可接纳的waiting序列数。
        # 维护者：postprocess（chunk没算完 +1 / 算完 -1）、preempt、结束移除、offload挂起（_park_ready_offload_partial_prefills）
        #   以及PP下的_advance_prefill_on_schedule；也作为"requests_partial_prefill"指标上报
        self._partial_prefill_count: int = 0
        # _schedule每调用一次加1：等待期限（_inflight_prefix_wait）按它计，周期性日志（每100 / 1000步）也看它
        self._schedule_tick: int = 0

        # 【分支】挂起等外部KV（PD的decode端 / offload加载）的请求数：它们已经分配了block，所以Phase 2用
        #   len(running) + 本计数 与max_num_seqs比较（_park_for_remote_load经_count_inflight_load +1；
        #   加载完成后被接纳、或被abort / 失败时经_uncount_inflight_load -1；也作为"requests_parked_kv_load"指标上报）
        self._num_parked_remote_kv: int = 0

        # ---- ⑪【分支】PP（流水线并行，第9批）----
        # Under PP the head keeps pp_size batches in flight, so schedule() must
        # advance chunked-prefill progress itself rather than defer it to
        # postprocess, else back-to-back schedules re-issue the same chunk.
        # 【译】PP下head同时有pp_size个batch在流水线里，所以schedule()必须自己推进chunked prefill的进度，
        #   而不是推迟到postprocess，否则背靠背的几次调度会重复发出同一个chunk。
        # 例：pp=4，chunk c0刚发给stage0、还没流到最后一个stage，head就要调度下一步了；若num_cached_tokens要等
        #   postprocess才前进，下一步算出的还是c0
        self.advance_on_schedule: bool = (
            getattr(config, "pipeline_parallel_size", 1) > 1
        )
        # Seq ids whose sampled token is in flight; the decode scheduler skips
        # them until the head releases the id after postprocess, so a seq is
        # never decoded against a token not yet appended.
        # 【译】采样出的token还在途中的seq id；decode调度跳过它们，直到head在postprocess之后释放该id，
        #   这样一条seq永远不会基于一个还没追加上来的token去decode。（mark_pp_inflight / release_pp_inflight维护）
        self._pp_inflight_token_block: set[int] = set()

        # ---- ⑫ connector、KV事件、EngineStats ----
        # 【分支】KV connector的scheduler一侧：--kv-transfer-config非空时才有（PD分离 / KV offload，第9批），否则为None；
        #   PD、offload是什么见sequence.py SequenceStatus类内的"科普：PD分离 / KV offload"和arg_utils.py
        #   kv_transfer_config字段上方的【科普】。本类调用connector时，要么先判断它是否为None，
        #   要么用getattr(connector, 方法名, None)探测（connector为None时同样探测不到）
        # 函数体内的import：执行到这里才导入
        from atom.utils.forward_context import get_kvconnector

        self.kv_connector = get_kvconnector("scheduler", config)
        # offload类connector要直接访问BlockManager（例如状态checkpoint的归属），在接纳任何请求之前绑定一次
        bind_block_manager = getattr(self.kv_connector, "bind_block_manager", None)
        if callable(bind_block_manager):
            bind_block_manager(self.block_manager)

        from atom.distributed.kv_events import (
            EventPublisher as _EventPublisher,
        )
        from atom.distributed.kv_events import (
            make_publisher as _make_publisher,
        )

        # 【分支】KV事件：block被存入/淘汰时对外广播，供KV感知的请求路由器使用（只由ATOM_KV_EVENTS_*环境变量配置，
        #   见config.py KVEventsConfig上方的说明，第4批）
        kv_events_cfg = getattr(config, "kv_events_config", None)
        parallel_cfg = getattr(config, "parallel_config", None)
        dp_rank = (
            getattr(parallel_cfg, "data_parallel_rank", None)
            if parallel_cfg is not None
            else None
        )
        # 引擎状态行与统计（engine_stats.py）：每throughput_log_interval秒（默认10）打印一行running/waiting请求数、
        #   KV使用率、前缀缓存命中率、吞吐、投机接受率；pool_pressure回调提供KV池与状态池的淘汰/占用数
        self.engine_stats = EngineStats(
            engine_index=dp_rank or 0,
            label=self._ENGINE_LABEL,
            use_spec=self.use_spec,
            mtp_k=self.mtp_k,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_log_stats=config.enable_log_stats,
            throughput_log_interval_s=config.throughput_log_interval,
            cache_hit_rate_window=config.cache_hit_rate_window,
            pool_pressure=self.block_manager.pool_pressure,
        )
        # 开了前缀缓存时把BlockManager交给它，用来读命中统计
        if config.enable_prefix_caching:
            self.engine_stats.block_manager = self.block_manager
        # Under pipeline parallelism every stage builds a Scheduler, but only
        # the head stage schedules and hashes blocks; the others never produce
        # events. Give them the null publisher so they neither bind the
        # per-DP-rank endpoints (a downstream stage on the same host would
        # collide with the head) nor emit duplicate streams.
        # 【译】流水线并行下每个stage都会构造一个Scheduler，但只有head stage调度、给block算hash，其他stage从不产生事件。
        #   给它们空发布器，这样它们既不会绑定按DP rank区分的端点（同一台机器上的下游stage会和head冲突），也不会发出重复的事件流。
        pp_rank = (
            getattr(parallel_cfg, "pipeline_parallel_rank", 0)
            if parallel_cfg is not None
            else 0
        )
        kv_events_on = kv_events_cfg is not None and kv_events_cfg.enable
        if kv_events_on and pp_rank:
            logger.info(
                "KV event publisher disabled on PP stage %s: only the head "
                "stage publishes",
                pp_rank,
            )
        if kv_events_on and not pp_rank:
            self.kv_event_publisher: _EventPublisher = _make_publisher(
                enabled=True,
                publisher_kind=kv_events_cfg.publisher,
                endpoint=kv_events_cfg.endpoint,
                topic=kv_events_cfg.topic,
                hwm=kv_events_cfg.hwm,
                buffer_steps=kv_events_cfg.buffer_steps,
                replay_endpoint=kv_events_cfg.replay_endpoint,
                replay_buffer_steps=kv_events_cfg.replay_buffer_steps,
                data_parallel_rank=dp_rank,
            )
            logger.info(
                "KV event publisher enabled: kind=%s endpoint=%s "
                "replay_endpoint=%s dp_rank=%s",
                kv_events_cfg.publisher,
                # The zmq publisher offsets its bind addresses per DP rank;
                # log what was actually bound, not the shared config value.
                # 【译】zmq发布器会按DP rank偏移它的绑定地址；这里记录实际绑定的地址，而不是共享的配置值。
                getattr(self.kv_event_publisher, "endpoint", kv_events_cfg.endpoint),
                getattr(
                    self.kv_event_publisher,
                    "replay_endpoint",
                    kv_events_cfg.replay_endpoint,
                ),
                dp_rank,
            )
        # 不开事件（或非head stage）时用"null"发布器：publish是空操作，调用方不必判断是否开启
        else:
            self.kv_event_publisher = _make_publisher(
                enabled=False,
                publisher_kind="null",
                endpoint="",
            )

        # ---- ⑬ PrefillDelayer（prefill合批，prefill_delayer.py）----
        # Set by EngineCore for cross-DP alignment or opt-in TP decode protection.
        # 【译】由EngineCore设置，用于跨DP对齐，或可选的TP decode保护。
        # 何时不为None（engine_core.py的_init_prefill_delayer，前提ATOM_ENABLE_PREFILL_DELAYER=1（默认）且非RapidServe）：
        #   DP>1：DPEngineCoreProc挂上带gloo组的delayer，每步在schedule()里做一次跨DP的all_reduce（锁步）
        #   DP=1、PP=1且ATOM_PREFILL_DECODE_INTERVAL>0（默认0）：挂一个"本地"delayer，做TP内的合批和decode保护
        #   其他情况为None：每步有prefill就直接做
        self.prefill_delayer: PrefillDelayer | None = None
        # delayer是"本地"的（is_local：dp_size=1且没有跨DP的进程组）时为True，_schedule据此选用_local_prefill_pending_work
        #   的精确探测，并启用_wait_for_inflight_prefix
        self._local_prefill_coalescing = False
        # {seq_id: 等待期限（_schedule_tick）}：_wait_for_inflight_prefix为"等别人的prompt末尾checkpoint"设的期限
        self._inflight_prefix_wait: dict[int, int] = {}

    # ==================== 给PrefillDelayer提供的探测（"本步有没有prefill可做、攒了多少"） ====================
    # PrefillDelayer是什么、它的FIRE/HOLD决策表见prefill_delayer.py文件头；调度器每步在_schedule开头调用
    #   delayer.should_allow_prefill(prefillable, pending_tokens, ...)，下面这些函数负责算出这些输入：
    #     本地合批（TP、DP=1）：_local_prefill_pending_work → (prefillable, pending)，与真正接纳逐条一致
    #     DP>1：_can_admit_head_prefill → prefillable；_waiting_new_token_count + _partial_prefill_remaining_tokens → pending
    #     两种都用：_kv_usage（KV水位）、_oldest_waiting_prefill_age_ms（排队时间上限）、_estimate_prefill_chunk_cost（成本均衡）
    #   设计要点：这些探测都不能有副作用（不分配block、不改seq），因为被扣住（HOLD）的那一步什么也不接纳
    #
    # 由EngineCore._init_prefill_delayer在启动时调用一次（单DP：EngineCore.__init__；DP>1：DPEngineCoreProc.__init__）
    def set_prefill_delayer(self, delayer: PrefillDelayer | None) -> None:
        self.prefill_delayer = delayer
        self._local_prefill_coalescing = delayer is not None and delayer.is_local
        self._inflight_prefix_wait = {}

    # 【分支】带每请求状态的混合模型（GDN / KDA线性注意力、DeepSeek-V4的SWA等），第4批；
    #   每请求状态、状态checkpoint、prompt末尾锚点是什么，见arg_utils.py EngineArgs.state_checkpoint_interval_tokens上方的
    #   【科普】和sequence.py Replay类上方的"科普：状态checkpoint与有界重放"
    # 调用方：只有_schedule的Phase 2，且只在本地合批（_local_prefill_coalescing）时；返回True = 本步先别接纳seq
    #   （放进prefix_waiters，按原顺序放回waiting队头，下一步再试）
    # 要解决的问题：两条共享一段长前缀的请求几乎同时到达（例：agent并发发出几个子任务，都带同一段60000 token的上下文）。
    #   A先被接纳、正在分块prefill，刚算到16384；B这时去查前缀缓存，状态只能从A已经存下的checkpoint（16384）恢复：
    #     B现在接纳：从16384开始，把A正在算的那约43600个token再算一遍
    #     B等一等：A算过自己的prompt末尾锚点（checkpoint_end_pos，约在60000）并存下checkpoint后，B从那里恢复，只算自己多出的部分
    #   等待有上限：第一次决定等时记下 当前tick + delayer.ttft_max_ticks（默认200），过期后正常接纳
    # 判定条件（全部满足才等）：
    #   ① 前缀缓存、状态缓存、checkpoint都开着，seq有每请求状态、不是多模态
    #   ② 还没过这条seq的等待期限
    #   ③ running里有一个"生产者"：仍是RUNNING、还没算到自己的锚点、锚点比seq当前的命中至少远出一整步token预算
    #      （省不到一步预算就不值得等）、锚点在seq的prompt之内、不是多模态、cache_seed相同
    #   ④ 生产者在锚点处会为seq需要的每一类状态都存checkpoint（checkpointers_at）
    #   ⑤ 两者的前anchor个token完全相同
    def _wait_for_inflight_prefix(self, seq: Sequence, cached_tokens: int) -> bool:
        """Bounded wait for a hybrid producer's planned prompt-end checkpoint."""
        # 【译】有界地等待一个混合模型"生产者"计划中的prompt末尾checkpoint。
        bm = self.block_manager
        delayer = self.prefill_delayer
        if (
            not bm.enable_prefix_caching
            or not bm.state.enabled
            or bm.state_checkpoint_interval_tokens == 0
            or not seq.has_per_req_cache
            or seq.multimodal_data is not None
        ):
            return False
        deadline = self._inflight_prefix_wait.get(seq.id)
        if deadline is not None and self._schedule_tick >= deadline:
            return False
        for producer in self.running:
            anchor = producer.checkpoint_end_pos
            if (
                producer.status != SequenceStatus.RUNNING
                or producer.num_cached_tokens >= anchor
                or anchor - cached_tokens < self.max_num_batched_tokens
                or anchor >= seq.num_prompt_tokens
                or producer.multimodal_data is not None
                or producer.cache_seed != seq.cache_seed
            ):
                continue
            keepers = bm.checkpointers_at(
                producer,
                anchor,
                # A fork needs room both to preserve the producer's state and
                # to restore it in the consumer's first forward.
                # 【译】一次fork既需要空间保住生产者的状态，也需要空间在消费者的第一个forward里恢复它。
                # （fork：消费者从生产者的checkpoint接手一份状态，第4批；所以"锚点之后下一个forward的token数"取两者各自剩余量的较小值）
                min(producer.num_prompt_tokens - anchor, seq.num_tokens - anchor),
            )
            if not keepers or any(
                cache.applies(seq) and cache not in keepers for cache in bm.state_caches
            ):
                continue
            # Reject divergent heads cheaply; equal long buffers stay in NumPy.
            # 【译】先便宜地排除开头就不同的；相等的长缓冲区交给NumPy比较。
            # token_ids是array("i")：memoryview切片比较不复制数据；np.frombuffer也是零拷贝地把它看成int32数组
            head = min(anchor, 64)
            if (
                memoryview(seq.token_ids)[:head]
                != memoryview(producer.token_ids)[:head]
            ):
                continue
            if np.array_equal(
                np.frombuffer(seq.token_ids, dtype=np.int32, count=anchor),
                np.frombuffer(producer.token_ids, dtype=np.int32, count=anchor),
            ):
                self._inflight_prefix_wait.setdefault(
                    seq.id, self._schedule_tick + delayer.ttft_max_ticks
                )
                return True
        return False

    # waiting里"可接纳"的请求：排除已abort的、挂起等外部KV的、永远调度不了的
    #   含yield的函数是生成器：调用时不执行，被迭代时才逐个产出；next(gen, None)只取第一个就停，不会扫完整个队列
    #   调用方：_probe_local_prefill_work、_estimate_prefill_chunk_cost，以及_schedule里"有没有prefill可做"的存在性检查
    def _waiting_prefills(self):
        for seq in self.waiting:
            if (
                seq.status
                not in (SequenceStatus.ABORTED, SequenceStatus.WAITING_FOR_REMOTE_KVS)
                and self._unschedulable_reason(seq) is None
            ):
                yield seq

    # 本地合批（TP、DP=1）时delayer的输入，_schedule调用；真正的工作在_probe_local_prefill_work
    def _local_prefill_pending_work(self) -> tuple[bool, int]:
        """Count probed work; a bounded scan never invents fill for unseen work."""
        # 【译】统计探测到的工作量；有界扫描绝不为没看到的工作虚构填充量。
        # The probe sizes the chunks admission is about to size; only
        # admission's outcomes belong in the sizing report.
        # 【译】探测会先替接纳把即将定尺寸的chunk定一遍尺寸；只有接纳的结果才应该进定尺寸报告。
        # 所以先拷一份动态分块的计数，finally里恢复（try/finally：无论是否抛异常都会执行finally）
        counts = self._dynamic_chunking_counts.copy()
        try:
            return self._probe_local_prefill_work()
        finally:
            self._dynamic_chunking_counts = counts

    # 返回 (这一步是否真有prefill可做, 这一步能装多少新token（封顶token预算）)
    #   做法：把_schedule的Phase 1 + Phase 2的尺寸计算原样走一遍（_partial_prefill_chunk、_prefill_chunk_for_budget、
    #   _preview_prefill_chunk），但不分配block、不改seq（can_allocate_memoized只探测不记录），
    #   所以delayer看到的"攒了多少"就是真正接纳时能装下的量
    # 例（预算16384，无前缀命中，普通模型）：running里一条partial还剩3000；waiting依次是8000、9000、500个token的新请求
    #   partial：3000 → pending=3000
    #   8000：剩余预算13384，放得下 → pending=11000
    #   9000：剩余预算5384，切成5384按64向下对齐 = 5376 → pending=16376
    #   500：已探测的prompt token（8000+9000=17000）达到预算，停止探测 → 返回(True, 16376)
    # 与真正接纳一样遇到第一个放不下的就停（break），不跳过它去看后面的
    def _probe_local_prefill_work(self) -> tuple[bool, int]:
        budget = self.max_num_batched_tokens
        pending = 0
        if self._partial_prefill_count:
            for seq in self.running:
                if not seq.is_partial_prefill:
                    continue
                resume = seq.num_cached_tokens
                start = self.block_manager.replay_start(seq, resume, hit=False)
                chunk = self._partial_prefill_chunk(
                    seq,
                    start,
                    resume,
                    pending,
                    cap_from=max(resume, seq.replay_until),
                )
                if not chunk:
                    break
                chunk = self._preview_prefill_chunk(seq, start, resume, chunk)
                pending += chunk
                if pending >= budget:
                    break
        prefillable = pending > 0
        if pending >= budget or len(self.running) >= self.max_num_seqs:
            return prefillable, min(pending, budget)
        probed_tokens = 0
        slots = self.max_num_seqs - len(self.running)
        for seq in self._waiting_prefills():
            offload_resume = self._is_offload_prefill_resume(seq)
            # 【分支】offload加载完成、待续算的请求：block早已持有，不再探测前缀缓存（第9批）
            if offload_resume:
                resume = self._offload_prefill_start(seq)
                # A replayed hit resumes as one, as admission starts it.
                # 【译】重放命中照样按重放命中恢复，和接纳时的起点一致。
                cached = self.block_manager.replay_start(
                    seq, resume, hit=seq.replays_hit
                )
            else:
                if probed_tokens >= budget:
                    break
                # The exact probe's answer; the walk is memoised across ticks.
                # 【译】与精确探测的答案相同；前缀遍历在各tick之间做了缓存。
                # 返回命中的hash block数；-1表示KV池放不下（BlockManager.can_allocate，第4批）
                cached_blocks = self.block_manager.can_allocate_memoized(seq)
                # probed_tokens限制探测量：探过的prompt总长到一份预算就停，避免每步把很深的waiting队列都探一遍
                probed_tokens += seq.num_tokens
                if cached_blocks < 0:
                    break  # Phase 2 also stops at the first allocation refusal.
                    # 【译】Phase 2 同样在第一次分配被拒时停下。
                if slots <= self._num_parked_remote_kv:
                    # A connector query can pin remote KV, so leave it to
                    # admission. Do not count uncertain remote-slot fit as
                    # fill or mistake an individually fitting request for idle.
                    # 【译】一次connector查询可能钉住远端KV，所以留给真正的接纳去做。不要把"远端槽位够不够"这种
                    #   不确定的东西算成填充量，也不要把一个单独放得下的请求误判为空闲。
                    prefillable = True
                    break
                resume = cached_blocks * self.block_manager.hash_block_size
                cached = self.block_manager.replay_start(seq, resume, hit=True)
            remaining = (
                seq.num_tokens - cached
                if offload_resume
                else self._new_prefill_tokens(seq, cached, resume)
            )
            atomic = self._requires_atomic_prefill(seq)
            if atomic:
                kind = ChunkKind.WHOLE
            else:
                kind = ChunkKind.NEXT if offload_resume else ChunkKind.FIRST
            chunk = self._prefill_chunk_for_budget(
                remaining, budget - pending, pending, history_len=cached, kind=kind
            )
            if (
                chunk is None
                or (atomic and chunk < remaining)
                or self._waits_below_resume(seq, cached, chunk, resume)
            ):
                break
            chunk = self._preview_prefill_chunk(seq, cached, resume, chunk)
            prefillable = True
            pending += chunk
            slots -= 1
            if pending >= budget:
                return True, budget
            if slots == 0:
                break
        return prefillable, pending

    # ==================== chunk的起点与大小 ====================
    # 先分清三个位置（普通模型三者相同；只有带每请求状态、要"重放"的模型才不同，见sequence.py Replay类上方的科普）：
    #   resume：这条seq"恢复"的位置——cache里它下面的每一行KV都已经有了（前缀命中点，或上一个chunk算到的位置）
    #   start：本次forward真正开始的位置，<= resume。重放时要从resume往前退一段重算，只为重建状态（KV不重写）
    #   start + chunk：本次forward结束的位置
    #       0 ........ start ──重放区（只重建状态）── resume ──真正的新token── start+chunk
    #
    # forward结束在resume或更低处 = 这一步只是在重算cache里已有的行，算了等于没前进
    #   例：重放从12291开始、恢复点12800，本步预算只剩256 → 12291+256 = 12547 <= 12800
    #   一般的重放（state transfer重放）这时要等一个预算更宽的step（返回True，调用方break）；
    #   DeepSeek-V4的有界重放（replays_hit）不等，可以跨chunk续算，下一步从12547接着重放
    @staticmethod
    def _waits_below_resume(seq: Sequence, start: int, chunk: int, resume: int) -> bool:
        """Whether a forward of `chunk` rows from `start` waits for a step
        that lets it pass `resume`, the point its seq resumed at: one ending
        there or below only recomputes rows the cache holds. A DeepSeek-V4
        bounded replay does not wait: it continues in the next forwards
        (`_resume_prefill`), and a budget narrower than the replay would
        otherwise hold it, and everything queued behind it, forever."""
        # 【译】从start开始、长chunk行的一次forward，是否要等一个能让它越过resume（这条seq恢复的位置）的step：
        #   结束在resume或更低处的forward只是在重算cache里已有的行。DeepSeek-V4的有界重放不等：它在接下来的forward里继续
        #   （_resume_prefill）；否则一个比重放区间还窄的预算会让它、以及排在它后面的所有请求永远卡住。
        return start + chunk <= resume and not seq.replays_hit

    # 【分支】KV offload（第9批）：offload加载完成、待续算的请求从哪里开始prefill
    #   通常就是加载后的num_cached_tokens；若外部层把整条序列都覆盖了（没有任何token要算），就把最后一个hash block
    #   交还给forward重算——首个decode要用这次prefill产出的logits，forward总得跑一次
    #   例：num_tokens=1000、hash_block_size=16 → (999 // 16) × 16 = 992，重算最后8个token
    def _offload_prefill_start(self, seq: Sequence) -> int:
        if seq.num_cached_tokens < seq.num_tokens:
            return seq.num_cached_tokens
        hbs = self.block_manager.hash_block_size
        return max(0, (seq.num_tokens - 1) // hbs * hbs)

    def _new_prefill_tokens(
        self, seq: Sequence, cached_tokens: int, resume: int | None = None
    ) -> int:
        """Rows a prefill from `cached_tokens` asks for. `resume`: where it
        resumed, at or above `cached_tokens` (`BlockManager.replay_start`); the
        long-prefill cap counts from there, the recomputed rows riding on top."""
        # 【译】一次从 cached_tokens 开始的prefill要算多少行。resume：它恢复的位置，不低于 cached_tokens
        #   （BlockManager.replay_start）；长prefill上限从那里开始算，重算的那些行额外搭在上面。
        # long_prefill_token_threshold（--long-prefill-token-threshold，默认0=不限）只在开了chunked prefill、且不是
        #   必须整条算完的多模态请求时生效：剩余量超过阈值，就只要"阈值 + 重放行数"
        #   例：threshold=8192，40000 token的新prompt → 8192；重放命中resume=12800、cached_tokens=12291 → 8192 + 509 = 8701
        # 为什么要它（预算16384，waiting = [40000, 2000, 3000]，无前缀命中）：
        #   不设：第1步40000切16384把预算吃满；第2步Phase 1续算16384；第3步7232 + 2000 + 3000 → 两条短请求第3步才prefill
        #   设4096：第1步 4096 + 2000 + 3000 → 短请求第1步就prefill完，TTFT短得多；长请求分10步算完
        #   注意：它不会让decode插进长请求的chunk之间（后面9步都是Phase 1续算的prefill步），见类上方"两个预算"的注意
        resume = cached_tokens if resume is None else resume
        remaining = seq.num_tokens - cached_tokens
        if (
            not self._requires_atomic_prefill(seq)
            and self.enable_chunked_prefill
            and 0 < self.long_prefill_token_threshold < seq.num_tokens - resume
        ):
            remaining = self.long_prefill_token_threshold + resume - cached_tokens
        return remaining

    def _resume_prefill(self, seq: Sequence, start: int, *, hit: bool) -> None:
        """Start `seq`'s prefill forward at `start`, where `BlockManager.
        replay_start` puts it for the point it resumes at, which `seq.replay`
        keeps when the two differ. `hit`: it resumes at a prefix hit, whose
        state a `replay` transfer never brings and a DeepSeek-V4 bounded replay
        (`Sequence.replays_hit`) did not. A DeepSeek-V4 bounded replay may span
        chunks: a forward picking up where one inside it ended continues it,
        writing nothing below the point it resumed at and reading no window row
        below where the replay started."""
        # 【译】让seq的prefill forward从start开始——start是 BlockManager.replay_start 为它的恢复点给出的位置；
        #   两者不同时由 seq.replay 记住恢复点。hit：它在一个前缀命中处恢复，而命中处的状态，replay方式的状态传输
        #   从不带来，DeepSeek-V4的有界重放（Sequence.replays_hit）也没带来。DeepSeek-V4的有界重放可以跨多个chunk：
        #   从重放内部某个forward结束处接着算的forward会延续这次重放，不写恢复点以下的任何东西，也不读重放起点以下的窗口行。
        # 调用方：_schedule的Phase 1（hit=False）、Phase 2的offload续算（hit=replays_hit）和新请求（hit=True）
        # 效果：把seq.num_cached_tokens改成forward的真实起点start（可能低于命中点）；start < 恢复点时记一条Replay，
        #   让attention backend知道[start, until)这段只为重建状态而重算
        # 例（沿用Replay类上方科普④的数字：state_replay_tokens=509）：KV命中12800，replay_start给出12291
        #   → seq.replay = Replay(until=12800, rebuilds=True, history_ids=起点前replay_history_tokens个token, start=12291)，
        #     num_cached_tokens = 12291
        #   若本步只算到12547（还没越过12800），下一步Phase 1再调用：start == resume == 12547 < until，走第一个分支，
        #     沿用原来的until / rebuilds / start，重放跨chunk继续
        #   普通模型：start == resume，seq.replay = None，num_cached_tokens不变
        resume = seq.num_cached_tokens
        replay = seq.replay
        if replay is not None and not hit and start == resume < replay.until:
            until, rebuilds, first = replay.until, replay.rebuilds, replay.start
        else:
            until, first = resume, start
            rebuilds = hit and (
                self.block_manager.state.transfer.replays or seq.replays_hit
            )
        seq.replay = (
            Replay(
                until,
                rebuilds,
                tuple(
                    seq.token_ids[max(0, start - self.replay_history_tokens) : start]
                ),
                first,
            )
            if start < until
            else None
        )
        seq.num_cached_tokens = start

    def _chunk_cost(self, start: int, chunk: int) -> float:
        """Estimated prefill cost of `chunk` new tokens after `start` cached
        ones: a per-token term plus the context each new token attends."""
        # 【译】在start个已缓存token之后prefill chunk个新token的估计成本：每个token一份固定项，加上每个新token要attend的上下文。
        # 成本 = chunk × (W + start + chunk/2)，单位相当于"一个新token attend一个上下文token"：
        #   W（ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT，默认360000）：每个新token与上下文无关的开销（线性层、MoE等），
        #     折算成"相当于attend多少个上下文token"
        #   start：每个新token都要看前面全部start个已缓存的token
        #   chunk/2：因果mask下块内平均每个token看前面一半（与engine_core.py"【科普】动态分块"里的2Lx + x²同一个量，差个系数2）
        #   例：chunk=8192，start=0 → 8192 × (360000 + 4096) ≈ 2.98e9；start=100000 → ≈ 3.80e9（贵27%）
        #   W很大，所以上下文要到十万量级注意力项才显著——正是长上下文rank拖慢整组的场景
        return chunk * (self._balance_token_weight + start + chunk / 2)

    def _prefill_room(self, batched_tokens: int, start: int) -> int:
        """New tokens a seq whose chunk starts at `start` may still add this
        tick: the batch budget, and under cost balancing the cost left to the
        step's target (largest k with `_chunk_cost(start, k)` <= what is left),
        but never below the balancing floor of `_balance_min_tokens`."""
        # 【译】一条chunk从start开始的seq本tick还能再加多少个新token：受batch预算约束；成本均衡下还受本步目标剩下的成本约束
        #   （满足 _chunk_cost(start, k) <= 剩余成本 的最大k），但不低于均衡的下限 _balance_min_tokens。
        # 调用方：Phase 1（经_partial_prefill_chunk）、Phase 2（新请求和offload续算）——每次给一条seq定chunk前都问一次
        # 没开均衡（_tick_cost_limit为inf）：就是token预算剩下的量
        # 开了均衡：解 k × (a + k/2) <= left（a = W + start）这个一元二次不等式，k = sqrt(a² + 2·left) − a；
        #   再保证本rank这一步至少有_balance_min_tokens（默认4096）个token（减去已经排进来的batched_tokens），
        #   最后不超过token预算剩余。数字例子见_balance_prefill_step
        cap_room = self.max_num_batched_tokens - batched_tokens
        if self._tick_cost_limit == math.inf:
            return max(cap_room, 0)
        left = self._tick_cost_limit - self._tick_cost
        room = 0
        if left > 0:
            a = self._balance_token_weight + start
            room = int(math.sqrt(a * a + 2.0 * left) - a)
        room = max(room, self._balance_min_tokens - batched_tokens)
        return max(min(room, cap_room), 0)

    def _estimate_prefill_chunk_cost(self) -> float:
        """Cost of the chunk this rank would prefill with the whole batch
        budget: resumable partials first, then the head of `waiting`, probing
        the prefix cache without recording (`can_allocate_memoized`). An
        estimate -- the real admission may park or skip some of these seqs."""
        # 【译】本rank用整份batch预算会prefill的chunk的成本：先算可续算的partial，再算waiting的队头，
        #   探测前缀缓存但不记录（can_allocate_memoized）。这是估计——真正的接纳可能会挂起或跳过其中一些seq。
        # 调用方：_schedule，只在cost_balance为True且本rank有prefill工作时；结果作为prefill_cost随delayer的all_reduce
        #   发给所有rank（各rank据此在_balance_prefill_step里算出同一个目标）
        token_cap = self.max_num_batched_tokens
        tokens, cost = 0, 0.0
        if self._partial_prefill_count > 0:
            for seq in self.running:
                if not seq.is_partial_prefill:
                    continue
                remaining = seq.num_tokens - seq.num_cached_tokens
                # A continued replay's rows below its `until` ride on top.
                # 【译】延续中的重放，其until以下的行额外搭在上面。
                cap_from = max(seq.num_cached_tokens, seq.replay_until)
                if 0 < self.long_prefill_token_threshold < seq.num_tokens - cap_from:
                    remaining = (
                        self.long_prefill_token_threshold
                        + cap_from
                        - seq.num_cached_tokens
                    )
                k = min(remaining, token_cap - tokens)
                if k <= 0:
                    return cost
                cost += self._chunk_cost(seq.num_cached_tokens, k)
                tokens += k
        for seq in itertools.islice(self._waiting_prefills(), self.max_num_seqs):
            if tokens >= token_cap:
                break
            if self._is_offload_prefill_resume(seq):
                resume = seq.num_cached_tokens
                start = self.block_manager.replay_start(
                    seq, resume, hit=seq.replays_hit
                )
            else:
                # Runs on every tick with prefill work, holds included: reuse
                # the walk across ticks; the answer is the exact probe's.
                # 【译】每个有prefill工作的tick都会运行（被扣住的tick也算）：跨tick复用前缀遍历；答案与精确探测相同。
                cached_blocks = self.block_manager.can_allocate_memoized(seq)
                if cached_blocks < 0:
                    break
                # From where the forward will really start, as admission
                # budgets it: a replayed hit starts below itself.
                # 【译】从forward真正开始的位置算，与接纳时的预算口径一致：重放命中从命中点下方开始。
                resume = cached_blocks * self.block_manager.hash_block_size
                start = self.block_manager.replay_start(seq, resume, hit=True)
            k = min(self._new_prefill_tokens(seq, start, resume), token_cap - tokens)
            if k > 0:
                cost += self._chunk_cost(start, k)
                tokens += k
        return cost

    def _balance_prefill_step(self, own: float) -> None:
        """Set this prefill step's cost target from every DP rank's chunk cost.

        The costs rode the delayer's all_reduce this tick (`own` is this rank's,
        0 when it estimated none), so every rank sees the same list. The target
        is the costliest chunk, lowered to `1 + slack` times the mean over the
        ranks that prefill: a rank above it shrinks its chunk. No rank
        reporting a cost means no rank has prefill work: nothing to balance."""
        # 【译】根据每个DP rank的chunk成本，设定本prefill step的成本目标。
        #
        #   这些成本在本tick搭着delayer的all_reduce传过来（own是本rank的，没估计时为0），所以每个rank看到的是同一个列表。
        #   目标是最贵的那个chunk，再压到"有prefill的rank的平均值 × (1 + slack)"：高于它的rank缩小自己的chunk。
        #   没有rank报告成本，说明没有rank有prefill工作：无需均衡。
        # 调用方：_schedule，cost_balance为True且delayer放行（delayer_allows）时
        # 例（2个DP rank，预算16384，W=360000，slack=0.1）：
        #   rank0：16384个token，start=0      → 成本 16384 × (360000 + 0 + 8192)      ≈ 6.03e9
        #   rank1：16384个token，start=200000 → 成本 16384 × (360000 + 200000 + 8192) ≈ 9.31e9
        #   平均 7.67e9 × 1.1 ≈ 8.44e9，目标 = min(9.31e9, 8.44e9) = 8.44e9
        #   rank0：成本低于目标，_prefill_room解出约22721 > 16384，仍按16384接纳
        #   rank1：_prefill_room解出 sqrt(560000² + 2 × 8.44e9) − 560000 ≈ 14870，按64向下对齐成14848
        #   效果：rank1少算约1500个token，两个rank的prefill耗时更接近，锁步时rank0少等一些
        # 每个rank用同一个列表算出同一个目标，不需要额外通信；_balance_stats只用于日志
        costs = [c for c in self.prefill_delayer.rank_prefill_costs if c > 0]
        if not costs:
            return
        target = min(max(costs), sum(costs) / len(costs) * (1 + self._balance_slack))
        self._tick_cost_limit = target
        st = self._balance_stats
        st[0] += 1
        st[2] += own
        st[3] += target
        st[4] += own > target

    # 成本均衡生效的步（_tick_cost_limit < inf）里由_schedule调用：累计token数，每50步打印一次平均token数、
    #   自身成本/目标的比值、被削减的步数
    def _log_prefill_balance(self, tokens: int) -> None:
        st = self._balance_stats
        st[1] += tokens
        if st[0] and st[0] % 50 == 0:
            logger.info(
                "prefill cost balance: steps=%d mean_tokens=%.0f "
                "own/target cost=%.3f shrunk=%d (cap %d tokens)",
                st[0],
                st[1] / st[0],
                st[2] / st[3] if st[3] else 0.0,
                st[4],
                self.max_num_batched_tokens,
            )

    def _partial_prefill_chunk(
        self,
        seq: Sequence,
        start: int,
        resume: int,
        batched_tokens: int,
        *,
        cap_from: int | None = None,
    ) -> int:
        """The next chunk of a running prefill that resumes at `resume` and
        starts at `start`, or 0. A chunk ending at or below `resume` only
        recomputes rows the cache holds, and the next pass would back off the
        same way again, forever: it waits for a step with room instead.
        `cap_from`: where the long-prefill cap counts from, `resume` unless the
        forward continues a bounded replay, whose rows below its `until` ride
        on top as its first forward's did."""
        # 【译】一个在resume处恢复、从start开始的、running中的prefill的下一个chunk，或者0。结束在resume或更低处的chunk
        #   只是在重算cache里已有的行，而下一遍又会同样地退回来，永远如此：所以它改为等待一个有空间的step。
        #   cap_from：长prefill上限从哪里开始算，通常是resume；但如果这个forward延续的是一次有界重放，
        #   那么它until以下的行要像它的第一个forward那样额外搭在上面。
        # 调用方：Phase 1，以及模仿Phase 1的_probe_local_prefill_work
        # 例：num_tokens=40000、已算16384、不设阈值、本步预算16384 → remaining=23616 → chunk=16384；下一步剩7232一次算完
        cap_from = resume if cap_from is None else cap_from
        remaining = seq.num_tokens - start
        if 0 < self.long_prefill_token_threshold < seq.num_tokens - cap_from:
            remaining = self.long_prefill_token_threshold + cap_from - start
        chunk = self._chunked_prefill_size(
            remaining,
            self._prefill_room(batched_tokens, start),
            batched_tokens,
            history_len=start,
            kind=ChunkKind.NEXT,
        )
        return chunk if start + chunk > resume else 0

    def _preview_prefill_chunk(
        self, seq: Sequence, start: int, resume: int, chunk: int
    ) -> int:
        # Estimation must not cancel a state fork. Only the actual admission
        # finalizes that decision after the delayer releases the batch.
        # 【译】估计不能取消一次状态fork。只有在delayer放行这批之后，真正的接纳才最终做出这个决定。
        # 即_finalize_prefill_chunk的无副作用版本：同样把chunk截到状态checkpoint的位置（checkpoint_cut，record=False不登记）
        #   或对齐到hash block边界（align_midstep_chunk），但不做fork检查；普通模型原样返回chunk
        if self._requires_atomic_prefill(seq):
            return chunk
        target = self.block_manager.checkpoint_cut(
            seq, _cut_floor(seq, start, resume), start + chunk, record=False
        )
        if target:
            return target - start
        return self.block_manager.align_midstep_chunk(seq, start, chunk)

    def _can_admit_head_prefill(self) -> bool:
        """Match SGL's `local_prefillable=True` semantics: report True iff
        this rank would *actually* admit a new prefill this tick.

        Just having `self.waiting` non-empty is too coarse — during a
        concurrent-burst workload (e.g. 1k/1k @ high concurrency) every
        DP rank has a full waiting queue, so `bool(self.waiting)` is
        ALWAYS True on all ranks → status="all" → delayer never engages.
        But only the 1-2 ranks with free KV blocks actually admit a
        prefill that tick; the other 6-7 ranks decode. That's the real
        "mixed" we need to delay.

        We peek the front of `waiting` (skipping a few unschedulable
        entries) and check `can_allocate` + token-budget, mirroring the
        same checks the admission while-loop runs below.
        """
        # 【译】对齐SGLang中 local_prefillable=True 的语义：当且仅当本rank在这个tick*真的*会接纳一个新prefill时返回True。
        #
        #   只看 self.waiting 非空太粗了——在并发突发的负载下（例如高并发的1k输入 / 1k输出），每个DP rank的waiting队列
        #   都是满的，所以 bool(self.waiting) 在所有rank上*总是*True → 状态为"all" → delayer永远不介入。
        #   但那个tick实际上只有1~2个有空闲KV block的rank会接纳prefill，其余6~7个rank在decode。这才是真正需要延迟的"混合"状态。
        #
        #   我们偷看waiting的队头（跳过少量不可调度的条目），检查 can_allocate + token预算，与下面接纳循环里做的检查相同。
        # 用在DP>1（delayer不是本地的）时：结果是delayer的prefillable输入，进跨DP的all_reduce
        # 准入判断流程：
        #   ┌────────────────────────────────────────────────────────────────────┐
        #   │ running里有做到一半的prefill？ ──是──> True（Phase 1一定会续算它） │
        #   │         │否                                                        │
        #   │ waiting为空？ ──是──> False                                        │
        #   │         │否                                                        │
        #   │ 依次看waiting的前4条：                                             │
        #   │   永远调度不了 / ABORTED / 等外部KV？ ──是──> 跳过，看下一条       │
        #   │   关了chunked prefill且新token数 > token预算？ ──是──> 跳过        │
        #   │   offload加载完成、待续算？ ──是──> True（block早已持有）          │
        #   │   否则就看这一条：KV池放得下吗？（can_allocate_memoized >= 0）     │
        #   │       放得下 ──> True        放不下 ──> False                      │
        #   │ 4条都被跳过 ──> False                                              │
        #   └────────────────────────────────────────────────────────────────────┘
        # 只看第一条合格的请求就下结论：Phase 2是FIFO，队头放不下就break，不会去接后面的
        # 观察：它没有检查Phase 2的"len(running) >= max_num_seqs就不再接纳"，running已满时也可能返回True
        #   （本地合批用的_probe_local_prefill_work检查了这一条）；另外"关了chunked prefill且新token数 > 预算"这一条
        #   实际不会触发——这样的prompt已被_unschedulable_reason的②判为永远调度不了，在第一条检查就跳过了
        if self._partial_prefill_count > 0:
            return True
        if not self.waiting:
            return False
        for i, seq in enumerate(self.waiting):
            if i >= 4:
                break
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status in (
                SequenceStatus.ABORTED,
                SequenceStatus.WAITING_FOR_REMOTE_KVS,
            ):
                continue
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                not self.enable_chunked_prefill
                and num_new_tokens > self.max_num_batched_tokens
            ):
                continue
            if self._is_offload_prefill_resume(seq):
                # Holds its blocks since before its load: nothing to allocate,
                # and a probe would recompute the hit flags it was admitted
                # with (`replays_hit`), which decide how it resumes.
                # 【译】它从加载之前就持有自己的block：没有要分配的；而且探测会重新计算它被接纳时的命中标志
                #   （replays_hit），这些标志决定它如何恢复。
                return True
            # KV-pressured requests definitely cannot prefill. A fit probe, not
            # an admission: it records nothing, and runs every tick, so the
            # walk is memoised across ticks.
            # 【译】KV吃紧的请求肯定不能prefill。这是一次"放不放得下"的探测，不是接纳：它什么也不记录，
            #   而且每个tick都跑，所以前缀遍历在tick之间做了缓存。
            return self.block_manager.can_allocate_memoized(seq) >= 0
        return False

    def _kv_usage(self) -> float:
        """Fraction of KV-cache blocks currently in use ∈ [0, 1].

        The `kv_usage` signal for PrefillDelayer's KV watermark bounds — both the
        high watermark (near-full → release, can't accumulate more) and the
        optional low watermark (GPU starving → release, feed it). Derived from
        BlockManager bookkeeping; cheap (no traversal of seq tables).
        """
        # 【译】当前在用的KV cache block比例，取值[0, 1]。
        #
        #   它是PrefillDelayer的KV水位界限所用的 kv_usage 信号——包括高水位（快满了 → 放行，反正攒不了更多）
        #   和可选的低水位（GPU在挨饿 → 放行，喂它）。由BlockManager的账本得出；很便宜（不遍历各seq的表）。
        # 其他读取方：引擎状态行（_record_throughput）、PD背压日志（_schedule）
        bm = self.block_manager
        total = bm.kv.num_blocks
        if total <= 0:
            return 0.0
        return bm.kv.num_used / total

    def _record_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        """Feed this tick's token counts into the throughput section of
        `engine_stats` and emit the periodic engine-status log line once
        `throughput_log_interval_s` has elapsed.

        No-op when `--no-enable-log-stats` disabled the section.

        The token counts must be accumulated on every call — they are what the
        line reports — but the three arguments below it are read fresh and then
        discarded on all but one call in `interval / step_time`, which at a 10s
        interval is about one in ten thousand. `window_expired` is a subtract
        and a compare, so gating on it keeps the per-step cost to the counter
        update.
        """
        # 【译】把本tick的token数喂给 engine_stats 的吞吐部分，并在 throughput_log_interval_s 过去之后打印周期性的引擎状态行。
        #
        #   被 --no-enable-log-stats 关掉时什么都不做。
        #
        #   token数必须每次调用都累加——状态行报告的就是它们——但下面那三个参数是现读的，每 interval / step_time 次调用里
        #   只有一次真正用到、其余都丢掉，在10秒间隔下大约是万分之一。window_expired 只是一次减法加一次比较，
        #   所以先用它挡一下，每步的开销就只剩计数器更新。
        # 调用方：schedule()（每步，传本步的prefill token数）、heartbeat_throughput（窗口到期时）；
        #   生成token数在postprocess里另行累加（engine_stats.update_throughput）
        stats = self.engine_stats
        if not stats.throughput_enabled:
            return
        stats.update_throughput(num_prompt_tokens, num_generation_tokens)
        if not stats.window_expired(time.monotonic()):
            return
        num_running_reqs, num_waiting_reqs = self.get_request_counts()
        stats.maybe_log_throughput(
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            kv_usage=self._kv_usage(),
        )
        if self._dynamic_chunking_counts:
            self._log_dynamic_chunking_counts()

    def heartbeat_throughput(self, now: float) -> None:
        """Close the throughput window on time while the engine sits idle."""
        # 【译】引擎空闲时也按时结算吞吐窗口。
        # EngineCore的busy_loop每轮都调用它，窗口到期才结算：引擎空闲时schedule()不跑，没有它状态行就会停在最后一次有活的时刻
        if self.engine_stats.window_expired(now):
            self._record_throughput()

    def _waiting_new_token_count(self) -> int:
        """Sum of new (uncached) tokens across the ADMITTABLE waiting queue,
        saturated at `max_num_batched_tokens`.

        Feeds PrefillDelayer's coalescer fill signal (accumulate a full prefill
        batch before releasing, instead of firing many tiny fragments). The cap
        early-exits the scan: one batch's worth is all the coalescer compares
        against, so there's no point summing a deep queue.

        Skips unschedulable, ABORTED, WAITING_FOR_REMOTE_KVS, and
        oversized-when-chunking-off sequences —
        so the "queued work" signal counts only tokens this rank could actually
        prefill this step. Counting remote-KV / unschedulable tokens here would
        inflate the cross-rank aggregate and reach the fill target before a real
        batch has accumulated. The `num_cached_tokens` discount is best-effort:
        an un-admitted seq has not been probed against the prefix cache yet, so
        this is an upper bound on new tokens for cache-hit prompts.
        """
        # 【译】可接纳的waiting队列里新（未缓存）token的总数，封顶为 max_num_batched_tokens。
        #
        #   它是PrefillDelayer合并器的填充信号（先攒满一整批prefill再放行，而不是发出许多很小的碎片）。
        #   封顶让扫描可以提前退出：合并器只拿一批的量做比较，所以没必要把很深的队列都加一遍。
        #
        #   跳过不可调度的、ABORTED、WAITING_FOR_REMOTE_KVS的，以及关掉chunking时过大的序列——这样"排队工作量"信号
        #   只统计本rank这一步真能prefill的token。把远端KV / 不可调度的token也算进来会抬高跨rank的总量，
        #   在真正攒够一批之前就达到填充目标。num_cached_tokens 的折扣只是尽力而为：尚未接纳的seq还没对前缀缓存做过探测，
        #   所以对能命中缓存的prompt来说，这是新token数的上界。
        # 调用方：_schedule的DP>1分支，pending_tokens = min(本函数 + _partial_prefill_remaining_tokens(), 预算)
        # 例（预算16384）：waiting = [3000, 5000, 20000] → 3000、8000、28000 → 封顶返回16384
        cap = self.max_num_batched_tokens
        total = 0
        for seq in self.waiting:
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status in (
                SequenceStatus.ABORTED,
                SequenceStatus.WAITING_FOR_REMOTE_KVS,
            ):
                continue
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                not self.enable_chunked_prefill
                and num_new_tokens > self.max_num_batched_tokens
            ):
                continue
            total += max(0, num_new_tokens)
            if total >= cap:
                return cap
        return total

    def _partial_prefill_remaining_tokens(self) -> int:
        """Sum of remaining (not-yet-computed) tokens across mid-chunked-prefill
        seqs in `running`, saturated at `max_num_batched_tokens`.

        Folded into the coalescer's pending-token signal so a small partial tail
        chunk does not force its own tiny prefill forward — it accumulates with
        fresh prefills instead, bounded by the coalescer's partial deadline.
        Skipped entirely in the common steady-state (no partial) via the
        `_partial_prefill_count` counter, and stops scanning once all
        `_partial_prefill_count` partials have been summed.

        This is an UPPER BOUND on what a partial can actually contribute in one
        forward: `remaining = num_tokens - num_cached_tokens` does NOT apply the
        per-step `long_prefill_token_threshold` chunk clamp that Phase-1
        scheduling uses, so when that threshold is set the coalescer fill signal
        may read slightly high. Acceptable for a coalescing heuristic; consistent
        with `_waiting_new_token_count`'s best-effort estimate.
        """
        # 【译】running里所有做到一半的chunked prefill序列剩余（尚未计算）token的总数，封顶为 max_num_batched_tokens。
        #
        #   它被并入合并器的待处理token信号，这样一个小的partial尾chunk不会单独触发一次很小的prefill forward——
        #   而是和新的prefill一起攒，受合并器的partial期限约束。在常见的稳态（没有partial）下靠 _partial_prefill_count
        #   计数器整个跳过；所有 _partial_prefill_count 个partial都加完后就停止扫描。
        #
        #   这是一个partial在一次forward里实际能贡献的量的*上界*：remaining = num_tokens - num_cached_tokens
        #   没有套用Phase 1调度用的每步 long_prefill_token_threshold 截断，所以设置了该阈值时合并器的填充信号可能略偏高。
        #   对合并启发式来说可以接受；与 _waiting_new_token_count 的尽力估计一致。
        # partial的"期限"：ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS（默认100个tick）——partial已经占着KV，比新请求更早强制放行
        if self._partial_prefill_count == 0:
            return 0
        cap = self.max_num_batched_tokens
        total = 0
        seen = 0
        for seq in self.running:
            if not seq.is_partial_prefill:
                continue
            total += max(0, seq.num_tokens - seq.num_cached_tokens)
            if total >= cap:
                return cap
            seen += 1
            if seen >= self._partial_prefill_count:
                break  # all partials summed; skip the rest of the decode tail
        return total

    def _oldest_waiting_prefill_age_ms(self) -> float:
        """Age in ms (since arrival) of the oldest ADMITTABLE waiting prefill,
        or 0.0 if none.

        After decode protection, max_queue_ms releases extra coalescing based
        on time since arrival, including backlog. This does not bound resource
        or checkpoint waits. ABORTED, remote-loading and statically rejected
        requests do not contribute.
        """
        # 【译】最老的可接纳waiting prefill（自到达起）的年龄，单位毫秒；没有时返回0.0。
        #
        #   在decode保护之后，max_queue_ms 按到达以来的时间（包括积压）放行额外的合并。它不限制资源等待或checkpoint等待。
        #   ABORTED、正在远端加载的以及被静态拒绝的请求不计入。
        # 只在设置了ATOM_PREFILL_DELAYER_MAX_QUEUE_MS（默认不启用）且不在decode保护窗口内时由_schedule调用；
        #   arrive_time是墙钟时间（time.time()），所以这里也用time.time()
        oldest_arrive = None
        for seq in self.waiting:
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status in (
                SequenceStatus.ABORTED,
                SequenceStatus.WAITING_FOR_REMOTE_KVS,
            ):
                continue
            if oldest_arrive is None or seq.arrive_time < oldest_arrive:
                oldest_arrive = seq.arrive_time
        if oldest_arrive is None:
            return 0.0
        return max(0.0, (time.time() - oldest_arrive) * 1000.0)

    def publish_kv_events(self) -> None:
        """Drain BlockManager's event log and publish as one EventBatch. Called
        by EngineCore at the end of each scheduler step. No-op when events are
        disabled (NullEventPublisher swallows the publish call)."""
        # 【译】取空BlockManager的事件日志，作为一个EventBatch发布出去。由EngineCore在每个调度步结束时调用。
        #   事件关闭时什么都不做（NullEventPublisher吞掉publish调用）。
        # 【分支】KV事件，见__init__里kv_events_config处
        events = self.block_manager.take_events()
        if events:
            self.kv_event_publisher.publish(events)

    def shutdown_kv_events(self) -> None:
        """Tear down the publisher background thread and ZMQ socket. Called
        by EngineCore on engine shutdown."""
        # 【译】拆掉发布器的后台线程和ZMQ socket。由EngineCore在引擎关闭时调用。
        # logger.exception：记录日志并附带当前异常的traceback；关闭阶段出错只记录、不再抛出
        try:
            self.kv_event_publisher.shutdown()
        except Exception:
            logger.exception("KV event publisher shutdown failed")

    def is_finished(self):
        # `_rejected` must be considered too: if a batch of seqs is all
        # oversized, schedule() moves them straight from `waiting` to
        # `_rejected`, leaving both `waiting` and `running` empty. Without
        # this check, busy_loop's `is_finished()` short-circuits to True
        # before EngineCore drains `_rejected` via take_rejected(), and
        # llm.generate() blocks forever.
        # 【译】_rejected 也必须考虑进来：如果一批seq全都过大，schedule() 会把它们直接从 waiting 移到 _rejected，
        #   waiting和running都变空。没有这个检查，busy_loop 的 is_finished() 会在EngineCore通过 take_rejected()
        #   取走 _rejected 之前就短路成True，llm.generate() 永远阻塞。
        # deferred_free_blocks同理：请求已结束、block还等着connector读完（见_deferred_sequence上方的说明），引擎不能算"空闲"
        # 对比nano-vllm：nano-vllm只看 not waiting and not running
        return (
            not self.waiting
            and not self.running
            and not self._rejected
            and not self.deferred_free_blocks
        )

    # 请求进入调度器：都只是放到waiting队尾，真正的接纳在下一次_schedule的Phase 2
    #   metrics.enqueue：给seq记排队开始时间（输入线程已经记过就保留原值），用于"排队时间"指标
    #   _warn_if_unschedulable：永远调度不了的请求在提交时就打一条警告（仍放进waiting，调度时再拒绝）
    #   add：单条；extend：批量，EngineCore每轮循环开头用它把输入队列里的新请求放进来（atom/内的生产代码只用extend）
    def add(self, seq: Sequence):
        self.metrics.enqueue(seq)
        self._warn_if_unschedulable(seq)
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        for seq in seqs:
            self.metrics.enqueue(seq)
            self._warn_if_unschedulable(seq)
        self.waiting.extend(seqs)

    # ==================== 【分支】connector的延迟释放（deferred free），第9批 kv_transfer/ ====================
    # PD分离、KV offload是什么：见sequence.py SequenceStatus类内的"科普：PD分离 / KV offload"，以及arg_utils.py
    #   开头的"【科普】KV offload"和kv_transfer_config字段上方的"【科普】PD分离"
    # 为什么请求结束了，它的block却不能马上释放：
    #   没有connector：postprocess发现请求结束 → BlockManager.deallocate → block回到空闲队列，下一个请求马上能拿去写自己的KV
    #   有connector：结束的那一刻，还有别人在异步地"读"这些block，而且不经过调度器——
    #     PD分离的prefill节点（producer）：这些block里的prompt KV正通过RDMA传给decode节点，收到finished_sending之前都可能在读
    #     KV offload：worker正把这些block的KV拷到CPU内存 / LMCache等外部层，收到finished_saving之前都在读
    #   这时立刻释放，新请求拿到同一批block写进自己的KV，正在进行的RDMA / 拷贝读到的就是别人的数据——
    #   decode节点或offload层拿到一份错的KV，而且不会有任何报错
    # 时间线（producer，请求A在第N步prefill完成并结束）：
    #   第N步postprocess：A结束；connector说"还在发送"（should_defer_free=True）→ A进deferred_free_blocks，block不释放
    #   第N+1 ~ N+k步：decode节点读A的block；调度器照常调度别的请求（A的block不在空闲队列里，不会被分出去）
    #   某一步_update_from_kv_xfer_finished收到finished_sending=[A] → _maybe_release_deferred(A) → 真正deallocate
    #   代价：这段时间A的block算"已用"，KV池的可用量变少；is_finished()把deferred_free_blocks算进去，引擎不会提前进入空闲
    #   同理，running中的序列若有保存正在读它的block，也不能被抢占（_is_preemptable）
    # 提前释放（early block release）：connector能说清"保存只读哪几个block"（protected_block_ids）时，postprocess只留下
    #   这几个block作为"租约"（lease），其余（decode阶段的block、没对齐的prompt尾巴、已经保存完的部分）立刻释放；
    #   租约block由_drain_source_safe_releases在保存完成后释放，这样的请求不进deferred_free_blocks
    # 本节的函数（都用getattr(connector, 方法名, None)探测：不同connector能力不同，没实现的方法就当"不支持"）：
    #   谓词（只问不改）：_connector_should_defer_free、_connector_protected_block_ids、
    #     _connector_can_partially_deallocate_state、_connector_waits_for_report
    #   通知connector（改connector自己的状态）：_connector_release_stalled_save、_connector_abandon_save、
    #     _connector_send_finished、_connector_source_blocks_released
    #   释放：_maybe_release_deferred（所有owner都放手后才deallocate）、_drain_source_safe_releases（租约block）
    #   兜底：_save_abandon_timeout_s + _reconcile_stalled_deferred_saves（完成报告永远不来时回收）
    #
    # 按req_id找延迟释放中的seq：connector一侧多用字符串id（本类调用abandon_save时也传str(seq.id)），
    #   所以finished_sending等报告里的req_id可能是str，按原值查不到就转成int再查一次
    def _deferred_sequence(self, req_id) -> Sequence | None:
        seq = self.deferred_free_blocks.get(req_id)
        if seq is not None:
            return seq
        try:
            return self.deferred_free_blocks.get(int(req_id))
        except (TypeError, ValueError):
            return None

    # connector是否还要用这条seq的block（PD发送 / offload保存未完成）；MultiConnector对各个子connector取"或"
    #   调用方：postprocess（结束时决定立即释放还是延迟）、_is_preemptable（能否抢占）、_maybe_release_deferred
    def _connector_should_defer_free(self, seq: Sequence) -> bool:
        callback = getattr(self.kv_connector, "should_defer_free", None)
        return bool(callable(callback) and callback(seq))

    def _connector_protected_block_ids(self, seq: Sequence) -> frozenset | None:
        """Exact block IDs a pending save still reads, or None if unsupported.

        None means the connector cannot narrow the protection (feature off, no
        offload connector, or this predicate not implemented, e.g. DSV4/K3) --
        the caller must fall back to deferring the whole request. See
        `DenseOffloadScheduler.protected_block_ids`.
        """
        # 【译】一次待完成的保存还要读的确切block ID；不支持时返回None。
        #
        #   None表示connector无法缩小保护范围（功能关闭、没有offload connector，或者没有实现这个谓词，例如DSV4 / K3）——
        #   调用方必须退回到"整个请求延迟释放"。见 DenseOffloadScheduler.protected_block_ids。
        # 调用方：postprocess，非None时走上面说的"提前释放"
        callback = getattr(self.kv_connector, "protected_block_ids", None)
        if not callable(callback):
            return None
        return callback(seq)

    def _connector_can_partially_deallocate_state(self, seq: Sequence) -> bool:
        """Require an explicit connector guarantee before releasing state.

        A PAGE block lease alone says nothing about a recurrent-state source.
        Connectors without this capability therefore remain on the original
        whole-request deferred-free path.
        """
        # 【译】释放状态之前，要求connector给出明确保证。
        #
        #   单有PAGE block的租约，并不能说明递归状态这一来源的情况。所以没有这项能力的connector仍走原来的
        #   "整个请求延迟释放"路径。
        # 只对有每请求状态的seq（has_per_req_cache）才问；必须返回严格的True（is True），返回别的值一律当作不保证
        callback = getattr(self.kv_connector, "can_partially_deallocate_state", None)
        return callable(callback) and callback(seq) is True

    def _drain_source_safe_releases(self) -> None:
        """Free block IDs the offload connector reports as newly source-safe.

        Polled after every `finished_saving` batch and by the stall
        reconciler. No-op for a connector without early block release (the
        method is simply absent).
        """
        # 【译】释放offload connector报告为"刚变得源安全"（不再有人读）的block ID。
        #
        #   在每批 finished_saving 之后以及由卡住回收器轮询。对于不支持提前释放block的connector什么都不做（它根本没有这个方法）。
        # 观察：v0.1.8里本函数只在_update_from_kv_xfer_finished末尾被调用（每次轮询KV进度都调，不论有没有finished_saving）；
        #   卡住回收器_reconcile_stalled_deferred_saves回收租约走的是connector.reclaim_stale_leases，并不调用本函数
        take = getattr(self.kv_connector, "take_source_safe_releases", None)
        if not callable(take):
            return
        for block_ids in take():
            if block_ids:
                self.block_manager.free_leased_blocks(block_ids)

    def _connector_release_stalled_save(self, seq: Sequence) -> None:
        """Let the connector drop a stall-escaped save this free surrenders.

        `preempt` frees blocks with `block_manager.deallocate` and no
        `request_finished`, so a K3 request whose `should_defer_free` returned
        False via its stall escape would keep its `_save_tracker` entry -- the
        base save loop could later emit a save reading those now-freed blocks.
        The connector's `should_defer_free` is a pure predicate, so this mutator
        does the cleanup exactly at the free. Guarded because not every connector
        offloads saves.
        """
        # 【译】让connector丢掉这次释放所交出的、已经靠"卡住逃生"放行的那次保存。
        #
        #   preempt 用 block_manager.deallocate 释放block，不调用 request_finished，所以一个 should_defer_free
        #   因卡住逃生而返回False的K3请求，会保留它在 _save_tracker 里的条目——基础的保存循环之后可能发出一次读取
        #   这些已释放block的保存。connector的 should_defer_free 是纯谓词，所以由这个修改函数恰好在释放时做清理。
        #   加了保护，因为不是每个connector都做offload保存。
        # K3：Kimi-K3混合模型的offload connector（atom/kv_transfer/offload/hybrid/kimi_k3/）；调用方：preempt
        callback = getattr(self.kv_connector, "release_stalled_save", None)
        if callable(callback):
            callback(seq)

    def _connector_waits_for_report(self, seq: Sequence) -> bool:
        """Whether the connector refuses clock-based reclaim for this request.

        A connector that cannot prove a stalled transfer stopped touching the
        blocks keeps them until a terminal report and bounds a missing report
        itself (LMCache MP fails stop at its transfer deadline). Abandoning it
        here would release nothing and only log a misleading wedge.
        """
        # 【译】connector是否拒绝对这个请求做"按时钟"的回收。
        #
        #   一个无法证明卡住的传输已停止访问这些block的connector，会一直保留它们直到收到终结报告，并自己给"报告迟迟不来"
        #   设上限（LMCache MP在传输截止时间处让它失败停止）。在这里放弃它什么也释放不了，只会打出一条误导性的"卡死"日志。
        callback = getattr(self.kv_connector, "waits_for_transfer_report", None)
        return bool(callback(seq)) if callable(callback) else False

    def _connector_abandon_save(self, seq: Sequence) -> None:
        """Tell the connector to drop a save this reclaim just abandoned.

        Without it the connector's `_save_inflight` (and, on K3, the stall
        latch) keeps the reclaimed request forever: `should_defer_free` stays
        True and `has_pending_kv_work()` never goes False, so the engine
        busy-loops. Guarded because not every connector implements offload
        saves.
        """
        # 【译】告诉connector丢掉这次回收刚刚放弃的保存。
        #
        #   没有它，connector的 _save_inflight（K3上还有卡住锁存）会永远留着这个被回收的请求：should_defer_free 一直为True，
        #   has_pending_kv_work() 永远不会变False，于是引擎空转。加了保护，因为不是每个connector都实现了offload保存。
        callback = getattr(self.kv_connector, "abandon_save", None)
        if callable(callback):
            callback(str(seq.id))

    def _connector_send_finished(self, req_id) -> None:
        """Drop the connector's send claim on a request's source blocks.

        The mutator half of `should_defer_free` for the send leg, as
        `_connector_abandon_save` is for the save leg. Guarded: only a producer
        connector tracks sends.
        """
        # 【译】撤销connector对一个请求源block的"发送占用"。
        #
        #   它是 should_defer_free 在发送这条腿上的"修改"一半，就像 _connector_abandon_save 之于保存那条腿。
        #   加了保护：只有producer connector跟踪发送。
        # 调用方：_update_from_kv_xfer_finished（收到finished_sending）、_mark_finished（abort：不会有对端来报告发送了）、
        #   abort后的加载清理
        callback = getattr(self.kv_connector, "send_finished", None)
        if callable(callback):
            callback(req_id)

    def _connector_source_blocks_released(self, seq: Sequence) -> None:
        """Tell the connector its source blocks are back in the pool.

        The terminal half of `request_finished` for a connector that deferred
        the free. Not `request_finished` a second time: that one also takes the
        P/D send claim, so re-invoking it would re-arm the claim this release
        just cleared. The base requires an explicit implementation; the guard
        also supports an absent connector and minimal scheduler test doubles.
        """
        # 【译】告诉connector它的源block已经回到池里。
        #
        #   对推迟了释放的connector，这是 request_finished 的"终结"那一半。不是再调一次 request_finished：
        #   那个函数还会拿走P/D的发送占用，再调一次会把这次释放刚清掉的占用重新挂上。基类要求显式实现；
        #   这里的保护也是为了支持没有connector以及最简化的调度器测试替身。
        callback = getattr(self.kv_connector, "source_blocks_released", None)
        if callable(callback):
            callback(seq)

    def _maybe_release_deferred(self, seq: Sequence) -> None:
        """Release once every connector has let go of the source blocks.

        `should_defer_free` is the one predicate; `MultiConnector` ORs it across
        its subs. It must stay a predicate rather than a tally of completion
        reports, because offload also owns chunks it has computed but not yet
        dispatched a save for.

        Deliberately does not call `request_finished` -- callers arrive with the
        request already finished, and a second call re-arms the P/D send claim
        this check just cleared. The block-lifetime cleanup that the second call
        used to carry is now its own hook, `source_blocks_released`, invoked
        below once the free has actually happened.
        """
        # 【译】等所有connector都放开源block之后再释放。
        #
        #   should_defer_free 是唯一的谓词；MultiConnector 对其各子connector取"或"。它必须保持为谓词，而不是完成报告的计数，
        #   因为offload还持有那些已经算完、但还没发出保存的chunk。
        #
        #   有意不调用 request_finished——调用方进来时请求已经结束了，再调一次会把本检查刚清掉的P/D发送占用重新挂上。
        #   以前由第二次调用顺带做的block生命周期清理，现在是单独的钩子 source_blocks_released，在下面真正释放之后调用。
        # 调用方：_update_from_kv_xfer_finished（finished_sending / finished_saving）、abort后的加载清理、卡住回收
        #   （_reconcile_stalled_deferred_saves）
        # 不释放的三种情况：不在deferred_free_blocks里；abort后还在等加载清理（_awaiting_aborted_load_cleanup）；connector仍要用
        if (
            seq.id not in self.deferred_free_blocks
            or getattr(seq, "_awaiting_aborted_load_cleanup", False)
            or self._connector_should_defer_free(seq)
        ):
            return

        self.deferred_free_blocks.pop(seq.id, None)
        self.block_manager.deallocate(seq)
        # After the free, not before: the connector may only forget the state
        # keyed on these blocks once nothing can read them again.
        # 【译】在释放之后、而不是之前：只有当再也没有谁能读这些block时，connector才能忘掉以它们为键的状态。
        self._connector_source_blocks_released(seq)

    def _save_abandon_timeout_s(self) -> float:
        """Seconds a deferred save may sit before reclamation, or 0 to disable.

        Sourced from the offload connector (`save_abandon_timeout_s`), not
        re-derived here: the window is a function of LMCache's own pin timeout,
        which is the connector's knowledge, and keeping the derivation in one
        place is what stops the safety ordering (reclaim strictly after LMCache
        force-unpins) from silently drifting. Returns 0 when no offload connector
        is attached, which disables reclamation -- there is nothing to reclaim.
        """
        # 【译】一次被推迟的保存在被回收前最多停留多少秒；0表示关闭回收。
        #
        #   取自offload connector（save_abandon_timeout_s），不在这里重新推导：这个窗口是LMCache自身pin超时的函数，
        #   那是connector掌握的知识；把推导放在一处，才能防止安全顺序（严格在LMCache强制解除pin之后才回收）悄悄漂移。
        #   没有挂offload connector时返回0，即关闭回收——没有东西可回收。
        # 具体值：LMCache为LMCACHE_EC_PIN_TIMEOUT_SEC + 30秒（multi_connector.py的save_abandon_timeout_s说明），
        #   Mooncake Store为它自己的配置项save_abandon_timeout_s
        callback = getattr(self.kv_connector, "save_abandon_timeout_s", None)
        return callback() if callable(callback) else 0.0

    def _reconcile_stalled_deferred_saves(self) -> int:
        """Reclaim blocks whose offload save has stalled past the abandon timeout.

        `_maybe_release_deferred` cannot do this: the connector still reports the
        save as pending (`should_defer_free` stays True), which is exactly why
        the blocks were never released. Once the save has been deferred longer
        than `_save_abandon_timeout_s()` -- set above LMCache's pin
        timeout, so upstream has force-unpinned and released the source bytes --
        the save's own hold on the blocks can go. Without this, a single
        never-reported save keeps `has_pending_kv_work()` True forever and the
        engine busy-loops with every GPU idle.

        Abandoning the save is not the same as freeing the blocks: the release
        still goes through `should_defer_free`, so an unreported P/D send keeps
        its source alive and `finished_sending` frees it later -- which is why
        the release attempt repeats every interval while the abandon happens
        once (`_save_abandoned`). The flip side is that this is no longer an
        unconditional escape: if `abandon_save` does not clear every claim (an
        active load does not go through it), the retry never succeeds, so
        `_RELEASE_WEDGE_ATTEMPTS` makes that state loud rather than silent.
        Reclamation does not re-invoke `request_finished`: it was already called
        when the request finished. It does notify the connector via
        `abandon_save` -- freeing the blocks here is not enough on its own: the
        connector holds the save in `_save_inflight` (and, on K3, in the stall
        latch), so `should_defer_free` would stay True and
        `has_pending_kv_work()` would never clear without that drop.

        The complement of the K3 connector's stall escape
        (`kimi_k3.connector.save_stall_seconds()`), not a duplicate of it: that
        one releases the blocks of a save the backend never took, on a shorter
        clock, and leaves a save already handed out alone -- precisely the case
        this reclaims once the report is not coming. Between them every deferred
        save has a way out -- as long as dropping the save drops the last claim.
        """
        # 【译】回收那些offload保存卡住超过放弃超时的block。
        #
        #   _maybe_release_deferred 做不到这一点：connector仍报告保存未完成（should_defer_free 一直为True），这正是block
        #   一直没被释放的原因。一旦保存被推迟的时间超过 _save_abandon_timeout_s()——它设得比LMCache的pin超时更长，
        #   所以上游已经强制解除pin、释放了源字节——保存自己对这些block的占用就可以撤掉了。没有这一步，一次永远不报告的
        #   保存会让 has_pending_kv_work() 永远为True，引擎在所有GPU都空闲的情况下空转。
        #
        #   放弃保存不等于释放block：释放仍要经过 should_defer_free，所以一次未报告的P/D发送会让它的源block继续存活，
        #   之后由 finished_sending 释放——这就是为什么释放尝试每个间隔都重复，而放弃只做一次（_save_abandoned）。
        #   反过来说，这已不再是无条件的逃生口：如果 abandon_save 没有清掉所有占用（进行中的加载不经过它），重试永远不会成功，
        #   所以用 _RELEASE_WEDGE_ATTEMPTS 让这种状态大声报出来，而不是悄无声息。回收不会再调用 request_finished：
        #   请求结束时已经调过了。它确实会通过 abandon_save 通知connector——光在这里释放block还不够：connector把这次保存
        #   记在 _save_inflight 里（K3上还在卡住锁存里），不撤掉的话 should_defer_free 会一直为True，
        #   has_pending_kv_work() 永远清不掉。
        #
        #   它是K3 connector的卡住逃生（kimi_k3.connector.save_stall_seconds()）的补充，而不是重复：那个机制按更短的时钟
        #   释放后端从未接手的保存的block，对已经交出去的保存不管——而那恰恰是本函数在报告不会来时回收的情形。
        #   两者合起来，每个被推迟的保存都有出路——只要丢掉保存就能丢掉最后一个占用。
        # 调用方：EngineCore._poll_kv_transfer_progress，每次轮询KV传输进度后都调（空闲时约每1ms一轮），
        #   本函数自己节流到每_SAVE_RECONCILE_INTERVAL_S=5秒才真正扫描一次
        # 时间线（设超时窗口为T）：
        #   t=0        A结束、保存未完成 → 进deferred_free_blocks，记_deferred_save_at=0
        #   t<T        每5秒扫一次：没到超时，跳过（报告来了就由_update_from_kv_xfer_finished正常释放）
        #   t>=T       abandon_save通知connector（只做一次）→ _maybe_release_deferred：
        #                没有别的占用 → 释放，完事
        #                还有P/D发送的占用 → 留着，此后每5秒再试一次（finished_sending晚到也能收走）
        #   又失败12次（_RELEASE_WEDGE_ATTEMPTS，约60秒）→ 打一次ERROR："这些block回不到池里了"，提示检查P/D后端
        # 返回值：释放的请求数 + 回收的租约block数（EngineCore不使用返回值）
        timeout = self._save_abandon_timeout_s()
        if timeout <= 0:
            return 0
        now = time.monotonic()
        if now < self._next_save_reconcile_at:
            return 0
        self._next_save_reconcile_at = now + _SAVE_RECONCILE_INTERVAL_S
        # Early-release leases (see `deallocate_partial`) live entirely on the
        # connector, keyed by save generation rather than by seq -- reclaim
        # those first since they are independent of `deferred_free_blocks`
        # below (an early-released request is never parked there).
        # 【译】提前释放的租约（见 deallocate_partial）完全保存在connector上，按保存的代次而不是按seq作键——先回收它们，
        #   因为它们与下面的 deferred_free_blocks 无关（提前释放的请求从不停放在那里）。
        reclaim_leases = getattr(self.kv_connector, "reclaim_stale_leases", None)
        lease_reclaims = 0
        if callable(reclaim_leases):
            for block_ids in reclaim_leases(timeout):
                if block_ids:
                    self.block_manager.free_leased_blocks(block_ids)
                    lease_reclaims += len(block_ids)
        if lease_reclaims:
            logger.warning(
                "Reclaimed %d offload save-lease block(s) whose save never "
                "reported after %.0fs.",
                lease_reclaims,
                timeout,
            )
        if not self.deferred_free_blocks:
            return lease_reclaims
        stalled = [
            seq
            for seq in list(self.deferred_free_blocks.values())
            if getattr(seq, "_deferred_save_at", None) is not None
            and now - seq._deferred_save_at >= timeout
        ]
        released = 0
        abandoned = 0
        wedged: list = []
        for seq in stalled:
            if self._connector_waits_for_report(seq):
                continue
            # The save is abandoned once; the release attempt is not. Whether
            # the blocks come back is a separate question -- a send that has
            # not reported still claims them -- and giving up on the first
            # attempt (by clearing the stamp) was how an unanswerable claim,
            # an abort's above all, lost its blocks for good. Retry each
            # interval instead, so a late `finished_sending`, or an explicit
            # retirement, is enough to collect them.
            # 【译】保存只放弃一次；释放尝试则不止一次。block能不能回来是另一个问题——一次还没报告的发送仍然占用着它们——
            #   而在第一次尝试后就放弃（清掉时间戳），正是一个无法应答的占用（尤其是abort产生的）永久丢掉它的block的原因。
            #   改为每个间隔重试一次，这样一个迟到的 finished_sending 或一次显式的撤销就足以把它们收回来。
            if not getattr(seq, "_save_abandoned", False):
                self._connector_abandon_save(seq)
                seq._save_abandoned = True
                self._abandoned_saves += 1
                observer = getattr(self, "on_save_abandoned", None)
                if callable(observer):
                    observer(seq.id)
                abandoned += 1
            self._maybe_release_deferred(seq)
            if seq.id not in self.deferred_free_blocks:
                seq._deferred_save_at = None
                released += 1
            else:
                seq._release_attempts = getattr(seq, "_release_attempts", 0) + 1
                if seq._release_attempts == _RELEASE_WEDGE_ATTEMPTS:
                    wedged.append(seq.id)
        if wedged:
            # Once: `_release_attempts` only equals the threshold on one pass.
            # 【译】只报一次：_release_attempts 只会在某一遍恰好等于阈值。
            logger.error(
                "Blocks for request(s) %s survived %d release attempts (%.0fs) "
                "after their save was abandoned -- a connector still claims the "
                "source and no completion report is coming to retire that "
                "claim. These blocks will not return to the pool and "
                "has_pending_kv_work() stays True. Check the P/D backend's "
                "send completions.",
                wedged,
                _RELEASE_WEDGE_ATTEMPTS,
                _RELEASE_WEDGE_ATTEMPTS * _SAVE_RECONCILE_INTERVAL_S,
            )
        if abandoned or released:
            logger.warning(
                "Abandoned %d offload save(s) still deferred after %.0fs with no "
                "completion report (LMCache force-unpins a stalled save without "
                "reporting it); released %d of %d stalled allocation(s), retrying "
                "the rest while a P/D send still owns the source. "
                "total abandoned_saves=%d",
                abandoned,
                timeout,
                released,
                len(stalled),
                self._abandoned_saves,
            )
        return released + lease_reclaims

    # ==================== 准入拒绝：哪些prompt"永远调度不了" ====================
    # 这条prefill必须一次算完、不能切块吗？——带多模态数据、但没有embedding_spans的请求（旧路径，如Qwen3.5-VL）
    #   原因：旧路径在第一次prefill时把整份图像数据交给ModelRunner做视觉编码，把embedding按位置散射到图像占位token上，
    #   之后调度器就清空multimodal_data；若被切块，后面chunk里的占位token拿不到embedding（会把原始的<|media_pad|>写进KV）
    #   有embedding_spans的span路径（v0.1.8里只有DeepSeek-V4.1）可以切块：每块只散射本块范围内的那部分，见multimodal_runtime.py文件头
    # 后果：这种prompt只能整条放进一步的token预算，超过max_num_batched_tokens就永远调度不了（_unschedulable_reason的②）；
    #   chunk计算时它用ChunkKind.WHOLE，不受动态分块和long_prefill_token_threshold限制
    # @staticmethod：不需要self的方法，用类名或实例都能调用
    @staticmethod
    def _requires_atomic_prefill(seq):
        data = getattr(seq, "multimodal_data", None)
        return data is not None and "embedding_spans" not in data

    def _unschedulable_reason(self, seq: Sequence) -> str | None:
        """Return a human-readable reason if `seq` is permanently unschedulable.

        Only checks static (configuration-time) capacity. Dynamic conditions
        that can clear up as other seqs finish (e.g. transiently full
        per-req-cache pool) are NOT checked here — they're warned at submit
        time (`_warn_if_unschedulable`) but not eligible for permanent drop
        at schedule time, since the prefill loop's existing `can_allocate`
        check will retry them later.

        Permanent failure modes (each leaves the seq stuck in `waiting`
        forever and would head-of-line block the prefill loop, which
        `break`s on the first oversized seq):
          - prompt longer than `max_model_len` → exceeds per-seq KV cache
            geometry; attention backends size `block_tables` as
            `max_model_len // block_size` cols and would crash with a
            broadcast error at prepare-time. (Checked first since it's the
            usual actionable cause.)
          - prompt longer than `max_num_batched_tokens` AND chunked prefill
            disabled → no single prefill forward can ever fit it (with chunked
            prefill enabled, the prompt is split across steps and this is fine)
          - prompt's KV blocks exceed the total pool size → never fits even on
            a fully empty pool. Counted per-rank (dcp-local, via
            `BlockManager.num_pool_blocks`) to match how the pool is sized and
            drawn, so a dcp>1 deployment admits prompts `dcp_world_size` times
            longer than the global block count alone would suggest.

        Called at submit time (`_warn_if_unschedulable`, which logs the
        reason and adds extra dynamic warnings) and at schedule time
        (drops the seq before it reaches the attention backend).
        """
        # 【译】如果seq永远无法调度，返回一个人能读懂的原因。
        #
        #   只检查静态（配置时就确定的）容量。会随着其他seq结束而消失的动态条件（例如暂时占满的每请求缓存池）不在这里检查——
        #   它们在提交时（_warn_if_unschedulable）给出警告，但在调度时不会被永久丢弃，因为prefill循环里已有的
        #   can_allocate 检查以后会重试它们。
        #
        #   永久失败的几种情况（每一种都会让seq永远卡在 waiting 里，并对prefill循环造成队头阻塞——循环遇到第一个过大的seq就 break）：
        #     - prompt长于 max_model_len → 超出每条序列的KV cache几何尺寸；注意力backend把 block_tables 的列数定为
        #       max_model_len // block_size，准备阶段会因broadcast错误而崩溃。（放在最前面检查，因为这通常是可以着手解决的原因。）
        #     - prompt长于 max_num_batched_tokens 且关闭了chunked prefill → 任何一次prefill forward都装不下它
        #       （开启chunked prefill时prompt会被分到多个step，这就没问题）
        #     - prompt的KV block数超过整个池的大小 → 即使池完全空着也放不下。按每个rank计数（DCP本地，通过
        #       BlockManager.num_pool_blocks），与池的划分和取用方式一致，所以 dcp>1 的部署能接纳的prompt，
        #       比只看全局block数所以为的长 dcp_world_size 倍。
        #
        #   在提交时（_warn_if_unschedulable，记录原因并附加额外的动态警告）和调度时（在seq到达注意力backend之前丢掉它）被调用。
        # 三种情况的数字例子（block_size=16）：
        #   ① max_model_len=32768，prompt 40000个token → 拒绝（调大--max-model-len或缩短prompt）
        #   ② 关了chunked prefill、预算16384，prompt 20000个token → 拒绝；开着chunked prefill就分两步：16384 + 3616
        #      旧路径的多模态prompt（_requires_atomic_prefill）开着chunked prefill也按这一条算：20000 > 16384 → 拒绝
        #   ③ KV池只有1000个block（能放16000个token），20000个token要1250个block → 拒绝，池全空也放不下
        #      DCP=4时每个rank只存约1/4（约5000个token → 313个block <= 1000）→ 可以接纳
        # 为什么要主动拒绝：Phase 2是FIFO，队头放不下就break。一个永远放不下的请求停在队头，后面的请求会全部被它堵死
        #   （nano-vllm没有这一步）。拒绝后它以FINISHED、leave_reason="unschedulable: ..."进_rejected，经take_rejected输出
        # 在线服务通常更早就拦下了：api_server的_validate_sequence_context_length（prompt + max_tokens超过max_model_len、
        #   或prompt超过KV池容量 → HTTP 400）；这里是离线LLMEngine等其他入口的防线，调度时每次也会重新检查
        # num_tokens（不是num_prompt_tokens）：被抢占后重新排队的序列，已生成的token也要重新prefill
        num_tokens = seq.num_tokens
        if num_tokens > self.max_model_len:
            return (
                f"input tokens={num_tokens} > max_model_len={self.max_model_len}. "
                f"Increase --max-model-len or shorten the prompt."
            )
        # Only processors with explicit spans and request leases can chunk.
        # 【译】只有提供显式span和请求租约的处理器才能切块。（即上面_requires_atomic_prefill的span路径）
        is_multimodal = self._requires_atomic_prefill(seq)
        if (
            not self.enable_chunked_prefill or is_multimodal
        ) and num_tokens > self.max_num_batched_tokens:
            remedy = (
                "Increase --max-num-batched-tokens or shorten the prompt "
                "(multimodal prompts are never chunked)."
                if is_multimodal
                else "Increase --max-num-batched-tokens, enable chunked "
                "prefill, or shorten the prompt."
            )
            return (
                f"input tokens={num_tokens} > max_num_batched_tokens="
                f"{self.max_num_batched_tokens}. {remedy}"
            )
        bm = self.block_manager
        total_blocks = bm.kv.num_blocks
        # `num_pool_blocks`, not `seq.num_blocks`: the latter counts global
        # blocks, while the pool is sized and drawn per-rank. Under dcp>1 the
        # two differ by `dcp_world_size`.
        # 【译】用 num_pool_blocks 而不是 seq.num_blocks：后者数的是全局block，而池是按每个rank划分和取用的。
        #   dcp>1时两者相差 dcp_world_size 倍。（DCP：decode上下文并行，把一条序列的KV按token交错分到多个rank，
        #   见arg_utils.py EngineArgs.decode_context_parallel_size上方的【科普】）
        # 每请求状态（GDN等的状态缓存）在自己预分配的张量里，不占这个池，所以这里不算它（报错信息里也这样说明）
        pool_blocks = bm.num_pool_blocks(num_tokens)
        if pool_blocks > total_blocks:
            scope = (
                f" (per-rank, dcp={bm.dcp_world_size})" if bm.dcp_world_size > 1 else ""
            )
            return (
                f"needs {pool_blocks} KV blocks{scope} for {num_tokens} input "
                f"tokens > total pool blocks={total_blocks}. Reduce prompt length "
                f"or raise --gpu-memory-utilization. (Per-req state cache lives in "
                f"its own pre-allocated tensor and does not consume pool blocks.)"
            )
        return None

    def _warn_if_unschedulable(self, seq: Sequence) -> None:
        """Log a single warning at submit time for permanently-unschedulable
        sequences. The seq still enters `waiting`; the prefill scheduler drops
        it later (see `schedule`).

        Also surfaces a dynamic configuration-time-only warning when the
        model was started with zero per-req-cache slots (max_num_seqs=0) —
        this is permanent if it holds at submit time, but is NOT eligible
        for schedule-time drop (a future config change could create slots).
        """
        # 【译】在提交时为永远无法调度的序列记录一条警告。seq仍然进入 waiting；之后由prefill调度把它丢掉（见 schedule）。
        #
        #   当模型启动时每请求缓存的槽位为零（max_num_seqs=0）时，还会给出一条仅与配置有关的动态警告——如果它在提交时成立，
        #   就是永久的，但它*不*在调度时被丢弃（以后的配置变更可能会创建槽位）。
        # 调用方：add / extend（每个新请求一次）
        reason = self._unschedulable_reason(seq)
        if reason is not None:
            logger.warning("Request %s will never be scheduled: %s", seq.id, reason)
            return
        bm = self.block_manager
        # No slots ever allocated (max_num_seqs=0 effectively) AND no slots
        # currently in use → seq with has_per_req_cache=True can never enter.
        # "No slots ever existed" is the permanent case, distinguished from
        # "all busy" by the pool's total capacity.
        # An empty free list means either "all slots in use" — which the
        # schedule loop handles by waiting — or "the pool can never hold one
        # request", the permanent case, and only that one is worth warning
        # about.
        #
        # Measured against what one request needs, not against zero. A pool of
        # 3 slots where a request takes `1 + num_spec` = 4 is as permanently
        # unschedulable as an empty one; `== 0` would let it wait forever in
        # silence. The group-shaped predicate this replaced divided before
        # comparing, so it caught that case for free.
        # 【译】从未分配过任何槽位（实际上 max_num_seqs=0）且当前也没有槽位在用 → has_per_req_cache=True 的seq永远进不来。
        #   "从来就没有槽位"是永久的情形，靠池的总容量与"全都在忙"区分开。空闲列表为空，要么是"所有槽位都在用"——
        #   调度循环会通过等待来处理——要么是"这个池连一个请求都装不下"，即永久情形，只有后者值得警告。
        #
        #   对照的是一个请求需要多少，而不是对照0。一个只有3个槽位、而每个请求要占 1 + num_spec = 4 个的池，
        #   和空池一样永远调度不了；用 == 0 会让它永远默默等待。被替换掉的那个按组形状的谓词在比较前先做了除法，
        #   所以顺带就覆盖了这种情况。
        # 槽位：线性注意力等每请求状态的格子，每个请求占1个"当前状态" + 投机解码每个草稿位置1个回滚槽位（见sequence.py的state_slot）
        if (
            seq.has_per_req_cache
            and not bm.state.has_free(bm.state_slots_per_req)
            and bm.num_state_slots < bm.state_slots_per_req
        ):
            logger.warning(
                "Request %s will never be scheduled: needs %d per-req cache "
                "slot(s) but the pool holds %d (max_num_seqs=0 for this model "
                "type, or --num-speculative-tokens too high for it).",
                seq.id,
                bm.state_slots_per_req,
                bm.num_state_slots,
            )

    def take_rejected(self) -> list[Sequence]:
        """Pop and return any seqs the prefill scheduler dropped because
        `_unschedulable_reason` flagged them (oversized prompt, exhausted
        pool, etc.). Caller (EngineCore) pushes them onto the same
        output_queue as forward-finished seqs so `llm.generate()` returns
        an output for them instead of blocking forever.
        """
        # 【译】弹出并返回prefill调度因 _unschedulable_reason 标记而丢掉的所有seq（prompt过大、池耗尽等）。
        #   调用方（EngineCore）把它们放进和forward完成的seq相同的 output_queue，这样 llm.generate() 会为它们返回一个输出，
        #   而不是永远阻塞。
        # 调用方：EngineCore._process_engine_step_inner，每步schedule()之后都取一次（不论本步有没有batch）；
        #   取出的还包括还没运行就被abort的waiting序列（leave_reason="aborted"）
        # 交换列表而不是copy+clear：把旧列表整个交出去，换一个新的空列表
        if not self._rejected:
            return []
        out = self._rejected
        self._rejected = []
        return out

    def schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]] | None:
        """Run a scheduling pass and close the throughput window.

        **Override `_schedule`, not this.** The window's 10s cadence is a
        whole-program invariant, and putting the tick here makes it depend on
        one call rather than on a `_record_throughput()` hand-placed at every
        early return inside `_schedule`. There are already several such
        returns; the next one added would otherwise stall the status line for
        as long as it fires, and nothing would fail — the log would just go
        quiet, which is indistinguishable from an idle engine.
        """
        # 【译】执行一次调度，并关闭（结算）吞吐统计窗口。
        #
        #   **要重写的是 `_schedule`，不是这个方法。** 统计窗口每10秒结算一次是全程序的不变量；把这一跳放在这里，
        #   它就只依赖这一次调用，而不必在 `_schedule` 里每个提前return处手工放一个 `_record_throughput()`。
        #   这样的return已经有好几处了，再加一处的话，它生效期间状态行会一直停更，而且什么都不会报错——
        #   日志只是安静下来，和引擎空闲没法区分。
        # 调用方：EngineCore._process_engine_step_inner（PP时是pp_engine_core.py的head stage）每步调用一次；
        #   返回 (ScheduledBatch, {seq_id: Sequence})，没有任何请求时返回None
        # DecodeScheduler（RapidServe的decode侧，本文件末尾）只重写_schedule，仍经过这里；
        #   PrefillScheduler不是Scheduler的子类，自己复制了同样的三行
        result = self._schedule()
        # 本步prefill的token数（decode批为0）计入吞吐窗口；生成token数在postprocess里按真正提交的token计
        self._record_throughput(num_prompt_tokens=_prompt_tokens_of(result))
        return result

    # ═══════════════════════ _schedule：一步调度的完整决策（本文件的核心） ═══════════════════════
    # 输入：waiting / running两个队列（含挂起等KV的、待拒绝的序列）；输出：(ScheduledBatch, {seq_id: Sequence})，或None
    # 一句话：先续算running里做到一半的chunked prefill（Phase 1），再从waiting准入新请求（Phase 2）；
    #   只要本步排上了任何prefill，就直接返回prefill批；一条prefill都没排上，才对running做一次decode（Decode scheduling）。
    #   所以一个ScheduledBatch要么全是prefill、要么全是decode（混合批在源码里仍是TODO，见下面"lip: TODO"处）。
    #   为什么prefill优先、代价是什么（prefill那一步所有decode停一拍 → ITL尖峰）：见engine_core.py文件开头【2】
    # 阅读建议：先看下面的流程图和带数字的例子；读代码时先只跟★主干，【分支】第一次读可以跳过
    #
    # 策略，逐条对比nano-vllm（nano-vllm注释版scheduler.py的Scheduler类上方列了前4条）：
    #   ① prefill优先，一步不混合prefill和decode ..... 相同
    #   ② chunked prefill ............................ 都有，但续算的位置不同：nano-vllm把没算完的序列留在waiting队头；
    #        ATOM的序列第一次被调度做prefill就变成RUNNING、进running队尾，没算完的（is_partial_prefill=True）
    #        由下一步的Phase 1从running里找出来优先续算（sequence.py状态表③）。nano-vllm只允许本步第一条被切，
    #        ATOM允许任意一条（通常是最后一条）被切去填满预算，切口向下对齐到max(block_size, 64)的整数倍
    #   ③ 预算 ...................................... 都有max_num_batched_tokens（token数）和max_num_seqs（条数）；
    #        ATOM还要求len(running) < max_num_seqs才准入（准入即进running，所以它限制的是"驻留"序列总数），
    #        PD / offload还要给挂起等KV的序列留名额；DP-attention下每步prefill还有一个"代价"上限（_tick_cost_limit）
    #   ④ 抢占 ...................................... 都是重算式、都抢running队尾；ATOM跳过KV正被offload / PD读取的序列，
    #        谁都抢不了时整个decode循环停下（见Decode scheduling处的例子）
    #   ⑤ 准入闸门 .................................. ATOM独有：PrefillDelayer可以整步否决prefill（Phase 1、2都跳过），
    #        让decode继续跑、把零散的prefill攒成一大批（prefill_delayer.py文件头）；DP>1时默认挂上，单DP默认不挂
    #
    # 决策流程（方框里每一行都对应下面的一段代码；★=主干，【分支】=扩展特性，代码里在入口处会再标一次）
    # ┌────────────────────────────── 准备（Phase 1之前，代码里没有小标题） ───────────────────────────────┐
    # │ ★ tick+=1；【分支】归还上一批借用的状态源（第4批）；本tick的prefill代价计数清零                    │
    # │ 【分支】remote KV / offload（第9批）：已传完KV的挂起请求挪到waiting前面；                          │
    # │         running里要等外部加载的半途prefill挂起回waiting                                            │
    # │ ★ waiting队头连续的ABORTED（客户端已断开）直接结束，送进_rejected                                  │
    # │ 【分支】动态分块记录本步prefill来源数（第9批）；offload为waiting头部预取前缀查询（第9批）          │
    # │ ★ 挂了PrefillDelayer？                                                                             │
    # │     是 → 估算本rank有没有能做的prefill、攒了多少token → should_allow_prefill() → delayer_allows    │
    # │          （DP>1时内部是跨DP的all_reduce：每个rank每步都必须调用，所以放在下面的return None之前）   │
    # │     否 → delayer_allows = True（默认的单DP部署）                                                   │
    # │ 【分支】DP代价均衡：用all_reduce带回的各rank chunk代价，定本步的prefill代价上限                    │
    # │ ★ running、waiting、_rejected都空？ 是 → return None（EngineCore本步不forward）                    │
    # └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    #                                              ▼
    # ┌────────────────────────── Phase 1: resume partial prefills from running ───────────────────────────┐
    # │ 条件：delayer_allows 且 有半途prefill（_partial_prefill_count > 0；纯decode的稳态直接跳过）        │
    # │ for seq in running（按队列顺序，先进先续）：                                                       │
    # │     本步prefill序列已达max_num_seqs → break                                                        │
    # │     不是半途prefill → continue                                                                     │
    # │     chunk = 下一块（剩余token、本步剩余预算、对齐、long_prefill上限）                              │
    # │     chunk <= 0（预算不够一个对齐单位，或只够重算已缓存的行）→ break                                │
    # │     记账，加入scheduled_seqs（type=PREFILL；序列留在running原位置，不挪动）                        │
    # └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    #                                              ▼
    # ┌──────────────────────────────── Phase 2: new requests from waiting ────────────────────────────────┐
    # │ while delayer_allows 且 (delay_factor<=0 或 已等够) 且 waiting非空                                 │
    # │       且 本步prefill序列 < max_num_seqs 且 token预算没满 且 代价没超上限：                         │
    # │     seq = waiting.popleft()                                                                        │
    # │     ★ ABORTED，或永远调度不了（prompt超长等）→ 结束它，送进_rejected，continue                     │
    # │     ★ len(running) >= max_num_seqs → 放回队头，break                                               │
    # │     【分支】还在等远端KV / offload加载 → 记进skipped，continue                                     │
    # │     【分支】offload前缀查询还没返回 → 记进prefix_waiters，continue                                 │
    # │     【分支】PD decode端：KV已传到 → 不做prefill，直接转为首个decode（进running），continue         │
    # │     【分支】offload加载完毕的续算（block早已分配）→ 只查预算：                                     │
    # │             放不下 → 放回队头，break；放得下 → 排上，continue                                      │
    # │     【分支】要加载远端KV但挂起名额已满 → 放回队头，break                                           │
    # │     ★ 查前缀缓存命中 + 池子够不够（can_allocate）：不够 → 放回队头，break                          │
    # │     【分支】running里另一条同前缀序列快要产出状态checkpoint → 记进prefix_waiters，continue         │
    # │     ★ 算chunk：预算放不下 / 多模态不能切开 / 只够重算命中部分 → 放回队头，break                    │
    # │     ★ allocate分配block：失败 → 释放、放回队头，break                                              │
    # │     【分支】要从远端 / LMCache加载KV或状态 → 挂起（WAITING_FOR_REMOTE_KVS），记进skipped，continue │
    # │     ★ chunk再按状态checkpoint / offload调整；多模态因此被切开 → 释放、放回队头，break              │
    # │     ★ _schedule_prefill_seq：置RUNNING、追加到running队尾、加入scheduled_seqs                      │
    # │ 循环结束：skipped → 放回waiting队尾；prefix_waiters → 放回waiting队头（保持到达顺序）              │
    # └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    #                                              ▼
    #                        本步排上了prefill（num_seqs_prefill > 0）？
    #                          是 → 组prefill批（is_final_chunk、next_token_ids、midstep checkpoint……）
    #                               → return prefill批（本步不做decode）
    #                          否 ▼
    # ┌──────────────────────────────────────── Decode scheduling ─────────────────────────────────────────┐
    # │ while running非空 且 本步decode序列 < max_num_seqs：                                               │
    # │     token预算放不下下一条（1+mtp_k个token）→ break                                                 │
    # │     seq = running.popleft()                                                                        │
    # │     半途prefill（例如本步prefill被delayer否决时）→ 记下，continue                                  │
    # │     【分支】PP：它上一步采样的token还在流水线里 → 记下，continue                                   │
    # │     while 池子不够它追加 1+mtp_k 个token：                                                         │
    # │         从running队尾往前找一条可抢占的 → 抢占它，再检查                                           │
    # │         找不到 → 抢占seq自己（本步不调度它）                                                       │
    # │         自己也不可抢（KV正被offload / PD读取）→ seq放回running队头，整个decode循环break            │
    # │     may_append补block，加入scheduled_seqs（type=DECODE）                                           │
    # │ scheduled放回running队头（顺序不变）；跳过的半途prefill、PP序列放到running队尾                     │
    # └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    #                                              ▼
    #           return decode批（可能为空：只为把connector元数据送出去，EngineCore不跑forward、不postprocess）
    #
    # 带数字走三步。默认配置：max_num_batched_tokens=16384、max_num_seqs=512、block_size=16 → chunk对齐单位max(16,64)=64；
    #   单DP且ATOM_PREFILL_DECODE_INTERVAL=0 → 没挂PrefillDelayer，delayer_allows恒为True；不开MTP；
    #   long_prefill_token_threshold=0（关）；为简单起见假设前缀缓存都没命中。
    #   初始：running=[A, B]（都在decode），waiting=[D(prompt 1000), C(prompt 20000)]
    #
    #   第1步  Phase 1：_partial_prefill_count=0，跳过
    #          Phase 2：D：剩余预算16384 ≥ 1000，整条放得下 → chunk=1000；allocate ceil(1000/16)=63个block；running=[A,B,D]
    #                   C：剩余预算16384-1000=15384 < 20000 → 截断，并向下对齐到64的倍数：15384-15384%64 = 15384-24 = 15360；
    #                      allocate一次就按整条序列分配20000/16=1250个block（同nano-vllm）；running=[A,B,D,C]
    #                   num_batched_tokens=16360 < 16384，但waiting已空，退出循环
    #          num_seqs_prefill=2 → 返回prefill批，A、B这一步不decode：
    #            req_ids=[D,C]  num_scheduled_tokens=[1000,15360]  num_cached_tokens=[0,0]  is_final_chunk=[True,False]
    #            total_tokens_num=total_tokens_num_prefill=16360  total_seqs_num_prefill=2  total_seqs_num_decode=0
    #          postprocess：D的prompt算完，得到第1个生成token；C.num_cached_tokens=15360、is_partial_prefill=True
    #            （_partial_prefill_count=1），C这一步采样出的token作废
    #   （第1步之后新请求E(prompt 500)到达：waiting=[E]）
    #   第2步  Phase 1：扫running，A、B、D不是半途prefill → continue；C剩20000-15360=4640 ≤ 16384 → chunk=4640
    #                   （是剩下的全部，不再对齐）
    #          Phase 2：E：剩余预算16384-4640=11744 → chunk=500；running=[A,B,D,C,E]
    #          返回prefill批：req_ids=[C,E]（Phase 1续算的排在前面）  num_scheduled_tokens=[4640,500]
    #            num_cached_tokens=[15360,0]  is_final_chunk=[True,True]；A、B、D又停一步
    #          postprocess：C的prompt算完，is_partial_prefill=False（计数回到0），C、E各得到第1个生成token
    #   第3步  Phase 1跳过；Phase 2：waiting为空；num_seqs_prefill=0 → Decode scheduling：
    #            running=[A,B,D,C,E]逐个popleft，block都够 → 每条1个token
    #            req_ids=[A,B,D,C,E]  num_scheduled_tokens=[1,1,1,1,1]  total_tokens_num_decode=5  total_seqs_num_decode=5
    #            num_spec_step=0（开MTP、mtp_k=3时每条4个token，num_spec_step=3）；scheduled按原顺序放回running队头
    #   延迟输出（默认开）只让postprocess晚一步拿到采样结果、序列末尾先放占位token，不改变上面的调度决策：
    #     见engine_core.py文件开头【3】、sequence.py的"延迟输出"一段
    def _schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]] | None:
        """Select the next batch of sequences for a forward pass.

        Tries prefill first; if no new prefills are ready, falls back to
        decoding already-running sequences.
        """
        # 【译】为下一次forward挑选一批序列。
        #
        #   先尝试prefill；如果没有就绪的新prefill，退而对已经在running中的序列做decode。
        # 解释：这里的"prefill"也包括Phase 1续算的半途chunk；决策全貌见上方流程图
        # ======== 准备（Phase 1之前；对应流程图第一个方框）========
        # 调度tick计数：用于每100 / 1000个tick打一次统计日志，以及_wait_for_inflight_prefix的等待截止tick
        self._schedule_tick += 1
        # Sources borrowed by the previous batch: its forward has been issued,
        # so they can go back on the free list.
        # 【译】上一批借用的源：它的forward已经发出，所以可以放回空闲列表了。
        # 【分支】每请求状态（GDN / Mamba等递归状态）的槽位钉住与checkpoint拷贝，见第4批 block_manager.py / state_pool.py
        self.block_manager.complete_previous_state_batch()
        # 本步选中的序列 {seq_id: Sequence}。dict保持插入顺序：Phase 1续算的在前、Phase 2新准入的在后，
        #   这个顺序就是batch里的行顺序（ScheduledBatch.req_ids）
        # num_seqs_prefill：本步prefill的序列数（受max_num_seqs限制）
        # num_batched_tokens：本步已用掉的token预算（受max_num_batched_tokens限制），同nano-vllm的num_batched_tokens
        scheduled_seqs = {}
        num_seqs_prefill = 0
        num_batched_tokens = 0
        # Before anything sizes a chunk this tick: the local coalescer's
        # pending-work probe (`_partial_prefill_chunk`) runs ahead of the
        # balancing decision and must not see the previous step's cost limit.
        # 【译】要在本tick任何地方计算chunk大小之前（重置）：本地合批器的待做工作探测（`_partial_prefill_chunk`）
        #   跑在均衡决策之前，不能看到上一步的代价上限。
        # 【分支】DP-attention的prefill代价均衡（cost_balance，DP部署见第7、9批）：_tick_cost_limit是本步prefill代价的上限
        #   （inf=不限，不开均衡时一直是inf），_tick_cost是本步已准入部分的代价；_prefill_room据此缩小每条的chunk
        self._tick_cost_limit = math.inf
        self._tick_cost = 0.0
        # 【分支】本步挂起等远端KV的序列（PD / offload，第9批），Phase 2结束后放回waiting队尾
        skipped_waiting_requests: deque[Sequence] = deque()
        # 与scheduled_seqs一一对应：每条序列本步算多少个token（prefill是chunk大小，decode是1+mtp_k）
        num_scheduled_tokens: list[int] = []
        # 【分支】MTP投机解码：{seq_id: 本步要验证的草稿token}，只在decode批里填（第7批）
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        # 【分支】remote KV / offload，见第9批：
        #   _promote_ready_remote_kv_requests：KV已传完（或传输失败）的挂起请求挪到waiting最前面。offload挂起的请求早已
        #     持有block；如果队头的新请求因池子不够而break，排在后面的它就永远轮不到，也就永远不释放block
        #   _park_ready_offload_partial_prefills：running里的半途prefill如果要等一段外部加载，挂起回waiting队头
        #     （WAITING_FOR_REMOTE_KVS，sequence.py状态表⑥）
        self._promote_ready_remote_kv_requests()
        self._park_ready_offload_partial_prefills()
        # Reclaim aborted heads even when decode protection vetoes Phase 2.
        # 【译】即使decode保护否决了Phase 2，也要回收队头的已取消请求。
        # 解释：Phase 2里也会处理ABORTED，但Phase 2可能整步被delayer跳过；这里只清队头连续的那几个。
        #   _reject_aborted_waiting：置FINISHED、leave_reason="aborted"，放进_rejected，由EngineCore当作已结束的请求送出
        while self.waiting and self.waiting[0].status == SequenceStatus.ABORTED:
            self._reject_aborted_waiting(self.waiting.popleft())
        # 【分支】动态分块（第9批）：记下本步有几个prefill来源（最多记到2）。只有最近16步里始终"至多一个prefill"，
        #   才按延迟模型调整chunk大小（_dynamic_chunk_limit）
        if self.enable_dynamic_chunking:
            self._record_prefill_supply()
        # 【分支】KV offload（第9批）：connector支持的话，为waiting头部最多OFFLOAD_ASYNC_LOOKUP_DEPTH（默认16）个请求
        #   提前发起外部存储的前缀查询（异步），等它们真正走到准入时，结果多半已经回来了
        if self.kv_connector is not None and self.waiting:
            prefetch = getattr(self.kv_connector, "prefetch_lookups", None)
            if prefetch is not None:
                # Includes requests whose lookup will be detached: the lookup
                # still refreshes the tier's LRU (`_can_detach_tier_lookup`).
                # 【译】也包括那些查询结果将被"分离"（准入时不等它）的请求：查询本身仍会刷新存储层的LRU（`_can_detach_tier_lookup`）。
                prefetch(
                    itertools.islice(self.waiting, 0, envs.OFFLOAD_ASYNC_LOOKUP_DEPTH)
                )

        # should_allow_prefill() runs a cross-DP all_reduce and MUST be called
        # every tick on every rank for lockstep — hence before the early-return.
        # 【译】should_allow_prefill() 内部做一次跨DP的all_reduce，为了锁步，每个rank每个tick都必须调用——所以放在提前return之前。
        # 解释：集合通信要求组里每个rank调用的次数一致；若某个rank因为没活，在下面return None而跳过了它，其他rank就会卡在
        #   all_reduce里。DP锁步的原理（以及空闲rank为什么要跑dummy batch）见engine_core.py DPEngineCoreProc类上方
        # 【分支】PrefillDelayer（prefill合批器，B级，FIRE / HOLD的完整判定规则见prefill_delayer.py文件头）：
        #   挂载条件（engine_core.py _init_prefill_delayer）：ATOM_ENABLE_PREFILL_DELAYER=1（默认）且不是RapidServe，并且
        #     DP>1（在DPEngineCoreProc里挂，跨DP对齐），或 DP=1、PP=1、ATOM_PREFILL_DECODE_INTERVAL>0（本地模式is_local）；
        #     所以默认的单DP部署不挂，走下面的else，delayer_allows恒为True
        #   作用：返回False（HOLD）时本步跳过Phase 1和Phase 2，只做decode；攒够一批或触发某个上限时放行（FIRE）
        if self.prefill_delayer is not None:
            # pending = fresh waiting new-tokens + resumable partials' remaining,
            # capped at the batch budget: the coalescer's accumulation signal.
            # 【译】pending = waiting里新请求的新token数 + 可续算的半途prefill剩余的token数，上限是一步的预算：
            #   合批器用它判断"攒够没有"。
            # running_decode_batch：真正在decode的序列数 = running里扣掉半途prefill（它们不是decode负载）
            running_decode_batch = max(
                0, len(self.running) - self._partial_prefill_count
            )
            # 本地模式下处于"每次prefill之后强制只做若干步decode"的保护窗口内（ATOM_PREFILL_DECODE_INTERVAL）
            # 下面三种情况分别算出"本rank有没有能做的prefill"（prefillable）和"攒了多少token"（pending_tokens）：
            #   ① 保护窗口内，或本地模式且没有decode：只看有没有活，不做昂贵的探测（探测结果改变不了这时的决定）
            #   ② 本地模式（is_local）：精确探测——按Phase 1 / 2的规则把chunk试算一遍（_local_prefill_pending_work，只试算、不分配）
            #   ③ DP模式：只看waiting队头前几条能否分配（_can_admit_head_prefill）+ 粗略的待算token总数
            protects_decode = self.prefill_delayer.protects_decode(running_decode_batch)
            if protects_decode or (
                self._local_prefill_coalescing and not running_decode_batch
            ):
                # Existence is sufficient during the hard protection window:
                # fit probes cannot change its decision. Coalescer hold bounds
                # start afterwards. With no decode, admission proceeds directly.
                # 【译】在硬保护窗口内，只需知道"有没有活"：能否放下的探测改变不了它的决定。合批器的扣留上限从窗口结束后才开始计。
                #   没有decode时，直接准入。
                prefillable = (
                    self._partial_prefill_count > 0
                    or next(self._waiting_prefills(), None) is not None
                )
                pending_tokens = 0
            elif self._local_prefill_coalescing:
                prefillable, pending_tokens = self._local_prefill_pending_work()
            else:
                prefillable = self._can_admit_head_prefill()
                pending_tokens = min(
                    self._waiting_new_token_count()
                    + self._partial_prefill_remaining_tokens(),
                    self.max_num_batched_tokens,
                )
            # DP prefill cost balancing: this rank's chunk cost rides the same
            # all_reduce, estimated only when this rank has prefill work. Not
            # skipped inside the decode-protection window: whether the delayer
            # holds there depends on every rank's decode load, known only after
            # the reduce (with no decode anywhere it fires through that window).
            # 【译】DP的prefill代价均衡：本rank的chunk代价搭同一次all_reduce的便车，只在本rank有prefill活时才估算。
            #   在decode保护窗口内也不跳过：delayer在窗口内是否扣留取决于所有rank的decode负载，这要reduce之后才知道
            #   （任何rank都没有decode时，它会直接穿过窗口放行）。
            # 【分支】cost_balance = enable_dp_attention 且 DP>1（见__init__）。估算的是"整步预算都给本rank"时的chunk代价，
            #   每个新token计一个常数项 + 它要attend的上下文长度（_chunk_cost）
            prefill_cost = (
                self._estimate_prefill_chunk_cost()
                if self.cost_balance and prefillable
                else 0.0
            )
            # FIRE / HOLD：decode保护窗口、没有decode必放、KV水位、排队时长、TTFT上限、DP对齐、填充率、停滞，见prefill_delayer.py文件头
            delayer_allows = self.prefill_delayer.should_allow_prefill(
                prefillable=prefillable,
                pending_tokens=pending_tokens,
                # decode-only: self.running also holds mid-chunked-prefill seqs,
                # which are NOT decode load — counting them would defeat the
                # coalescer's "no decode → fire" fast path.
                # 【译】只算decode：self.running里也有chunked prefill做到一半的序列，它们不是decode负载——把它们算进去会让
                #   合批器"没有decode → 直接放行"的快速路径失效。
                running_decode_batch=running_decode_batch,
                kv_usage=0.0 if protects_decode else self._kv_usage(),
                has_partial=self._partial_prefill_count > 0,
                oldest_waiting_age_ms=(
                    self._oldest_waiting_prefill_age_ms()
                    if not protects_decode
                    and self.prefill_delayer.max_queue_ms is not None
                    else 0.0
                ),
                prefill_cost=prefill_cost,
            )
        else:
            # 没挂delayer：永远放行
            delayer_allows = True
            prefill_cost = 0.0

        # 【分支】DP代价均衡：放行时，用all_reduce带回的各rank代价算出本步目标 = min(最大代价, 平均代价×(1+slack))，
        #   写进_tick_cost_limit；代价高于目标的rank在Phase 1 / 2里缩小chunk（_prefill_room），让各DP rank的prefill
        #   大致同时结束（DP rank之间是锁步的，最慢的rank决定这一步的耗时）
        if self.cost_balance and delayer_allows and self.prefill_delayer is not None:
            self._balance_prefill_step(prefill_cost)

        # Rejections may still need an empty batch to dispatch connector cleanup.
        # 【译】即使只剩被拒绝的序列，也可能需要一个空batch来下发connector的清理工作。
        # ★ 提前return：三者全空才返回None（EngineCore本步什么都不做）。只剩_rejected时继续往下走，最后得到一个空的decode批：
        #   EngineCore照样把它的connector_meta_output发给worker，但不跑forward、不postprocess（engine_core.py _process_engine_step_inner）
        if not self.running and not self.waiting and not self._rejected:
            return None

        # ---- Phase 1: resume partial prefills from running ----
        # Gated by `delayer_allows` so cross-DP alignment still holds when one
        # rank is mid-chunked-prefill: a delayer veto skips both Phase 1 and
        # Phase 2 in lockstep. Inside that, skip the running-queue scan entirely
        # when no seq is mid-prefill — the common steady-state decode case —
        # using the counter maintained by postprocess / preempt / finished-removal.
        # 【译】---- Phase 1：从running中续算做到一半的prefill ----
        #   受 `delayer_allows` 控制，这样某个rank正处于chunked prefill中途时，跨DP对齐依然成立：delayer的否决会让各rank
        #   同步地同时跳过Phase 1和Phase 2。在此之上，如果没有任何序列处于prefill中途——这是最常见的稳态decode情形——
        #   就借助 postprocess / preempt / 结束移除 维护的计数器，整个跳过对running队列的扫描。
        # ★ 主干。为什么续算排在新请求前面：半途prefill在准入时就已经为整条序列分配了全部block，越早算完越早进入decode、
        #   结束后释放；先接新请求只会让更多block被"半成品"占着。
        # 对比nano-vllm：nano-vllm的半途序列留在waiting队头，下一步的prefill循环自然最先看到它；
        #   ATOM的半途序列在running里（第一次被调度就变成RUNNING），所以要这一段专门把它们找出来
        if delayer_allows and self._partial_prefill_count > 0:
            for seq in self.running:
                # 本步prefill序列数已满
                if num_seqs_prefill >= self.max_num_seqs:
                    break
                # 正在decode的序列跳过；用continue而不是break：半途prefill可能排在running的任何位置
                if not seq.is_partial_prefill:
                    continue
                # resume：KV已经算到的位置（续算的起点）
                # start：forward实际开始的位置。普通模型start == resume；【分支】带每请求状态、要做"有界重放"的模型
                #   （DeepSeek-V4等）会往前退几行重算，见sequence.py Replay类上方的"科普：状态checkpoint与有界重放"（第4、8批）
                # chunk：下一块的大小（_partial_prefill_chunk → _chunked_prefill_size）= min(剩余token, 本步剩余预算[, 动态分块上限])，
                #   被预算截断时向下对齐到max(block_size, 64)；long_prefill_token_threshold>0时每步最多算这么多；
                #   cap_from：long_prefill上限从哪里算起（续一段有界重放时从它的until算起）；只能重算resume以下的行时返回0
                resume = seq.num_cached_tokens
                start = self.block_manager.replay_start(seq, resume, hit=False)
                chunk = self._partial_prefill_chunk(
                    seq,
                    start,
                    resume,
                    num_batched_tokens,
                    cap_from=max(resume, seq.replay_until),
                )
                # _resume_prefill：把num_cached_tokens设为start，并记下 / 清掉重放信息（普通模型等于什么都没变）
                # _finalize_prefill_chunk：【分支】带每请求状态的模型把chunk的终点对齐到状态checkpoint的"梯级"上（第4批）；
                #   普通模型、多模态原样返回
                if chunk:
                    self._resume_prefill(seq, start, hit=False)
                    chunk = self._finalize_prefill_chunk(seq, start, chunk)
                # chunk为0：通常是本步剩余预算连一个对齐单位都不够（或这条只能重算已缓存的行）。break而不是continue：
                #   后面的半途序列一般也拿不到预算；而且保持按running顺序先进先续，不让后面的插队
                if chunk <= 0:
                    break
                # 记账并加入本批。这里不调用_schedule_prefill_seq：序列已经是RUNNING、已在running里，不用再追加；
                #   前缀命中的统计也只在首次准入时记一次
                self._tick_cost += self._chunk_cost(seq.num_cached_tokens, chunk)
                num_batched_tokens += chunk
                num_seqs_prefill += 1
                seq.type = SequenceType.PREFILL
                scheduled_seqs[seq.id] = seq
                num_scheduled_tokens.append(chunk)

        # ---- Phase 2: new requests from waiting ----
        # 【译】---- Phase 2：从waiting中准入新请求 ----
        # ★ 主干，对应nano-vllm schedule()的prefill循环。waiting里除了新请求，还有被抢占回来的序列（要把prompt+已生成的token
        #   整体重算，见下面"Use num_tokens"处）和挂起等KV的序列（【分支】第9批）
        # 循环体里两种出口的约定（与nano-vllm同样的思路）：
        #   continue：这条序列已经被"处理掉"（结束 / 挂起 / 转为decode / 暂缓），不占本步预算，接着看下一条
        #   放回队头 + break：卡在资源上（running满、池子不够、预算放不下、分配失败）。后面的序列多半也卡在同一资源上；
        #     就算更小的能塞进去，让它们插队也会让队头的大请求一直饿着，所以保持FIFO，下一步从它重新开始
        # prefix_waiters：被暂缓的请求（offload前缀查询还没返回 / 等别的序列产出同前缀的状态checkpoint），
        #   循环结束后按原顺序放回waiting队头
        # prefix_bypass_left：一旦有请求被暂缓，最多再往后看min(16, max_num_seqs)条（有界前瞻），暂缓者不会被无限超车
        prefix_waiters: deque[Sequence] = deque()
        prefix_bypass_left = min(16, self.max_num_seqs)
        # 循环条件：delayer放行；【分支】scheduler_delay_factor（默认0，借自vLLM PR 3279）>0时，要等最早到达的请求
        #   已等待超过 delay_factor×上一次prefill的耗时（或running为空）才准入，让waiting多攒一些；waiting非空；
        #   本步prefill序列 < max_num_seqs；token预算没满；【分支】DP代价没超上限
        while (
            delayer_allows
            and (self.delay_factor <= 0 or self._passed_delay(time.time()))
            and self.waiting
            and num_seqs_prefill < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
            and self._tick_cost < self._tick_cost_limit
        ):
            # 有请求被暂缓后，每多看一条就扣一次前瞻额度，用完就停
            if prefix_waiters:
                if prefix_bypass_left == 0:
                    break
                prefix_bypass_left -= 1
            # 从队头取出（nano-vllm是先看队头、确定调度才popleft；ATOM先取出，没调度上的再appendleft放回原位）
            seq = self.waiting.popleft()

            # Client disconnected before this seq ever ran: it holds no KV yet
            # and needs no forward pass, so finish it outright and route via
            # `_rejected` (emits a finished RequestOutput through the same
            # output_queue). Must intercept here BEFORE the waiting->running
            # promotion below, which would overwrite ABORTED with RUNNING and
            # lose the abort intent.
            #
            # It does not follow that it holds nothing: a sequence parked for
            # a transfer was allocated before it parked, and holds a full block
            # table plus, for a hybrid, a state group. Leaking the group wedges
            # every hybrid request once enough disconnects accumulate, since
            # `can_allocate` refuses one outright with the pool empty.
            # `deallocate` is a no-op for a seq that really held nothing.
            # 【译】客户端在这条序列运行之前就断开了：它还没有KV，也不需要forward，所以直接结束它，经由 `_rejected` 送出
            #   （通过同一个output_queue发出一个已结束的RequestOutput）。必须在这里、在下面 waiting->running 的提升之前拦截，
            #   否则RUNNING会覆盖ABORTED，"取消"的意图就丢了。
            #
            #   但这不代表它什么都没持有：一条为等传输而挂起的序列在挂起前就已经分配过，持有一整张block table，混合模型还持有
            #   一个状态组。泄漏这个组，等断开的请求积累到一定数量，所有混合模型请求都会卡死，因为池子空了 `can_allocate` 会直接拒绝。
            #   对真的什么都没持有的序列，`deallocate` 是空操作。
            if seq.status == SequenceStatus.ABORTED:
                self._reject_aborted_waiting(seq)
                continue

            # Drop seqs the static-capacity check at submit-time flagged as
            # permanently unschedulable (oversized prompt, exhausted pool,
            # etc.). They've already been warned; mark FINISHED + record the
            # rejection reason and route them to `_rejected` so EngineCore
            # surfaces them through the same output_queue as forward-finished
            # seqs. Without this they'd reach the attention backend (where an
            # oversized prompt crashes with a broadcast error) AND
            # `llm.generate()` would block forever waiting for an output.
            # Re-check here (not just at submit) since pool state may change.
            # 【译】丢弃提交时静态容量检查标记为"永远无法调度"的序列（prompt过长、池子不够等）。它们已经被警告过；标记为FINISHED、
            #   记下拒绝原因，送进 `_rejected`，让EngineCore通过与forward正常结束的序列相同的output_queue把它们交出去。否则它们会
            #   进入注意力后端（超长prompt会在那里因广播错误崩溃），而且 `llm.generate()` 会永远阻塞等输出。
            #   在这里再检查一次（不只在提交时），因为池的状态可能变化。
            # 解释：三种永久性原因见_unschedulable_reason：超过max_model_len；关闭chunked prefill（或多模态不能切）时超过
            #   max_num_batched_tokens；所需block数超过整个池子。用continue：它不占资源，不该挡住后面的请求
            unschedulable = self._unschedulable_reason(seq)
            if unschedulable is not None:
                self._inflight_prefix_wait.pop(seq.id, None)
                seq.status = SequenceStatus.FINISHED
                seq.leave_reason = f"unschedulable: {unschedulable}"
                seq.multimodal_data = None
                self._rejected.append(seq)
                continue

            # ★ running已有max_num_seqs条 → 放回队头，break。
            # 对比nano-vllm：nano-vllm的max_num_seqs只限制每步调度的条数；ATOM准入即进running，这里限制的是同时驻留
            #   （已准入、未结束）的序列总数，因此decode批的条数也不会超过它
            if len(self.running) >= self.max_num_seqs:
                self.waiting.appendleft(seq)
                break

            # 【分支】PD分离 / KV offload（第9批）：处理WAITING_FOR_REMOTE_KVS状态的序列（其他序列直接返回False）
            #   None：还没传完，已记进skipped，continue；True：PD decode端的KV已到，下面直接转为首个decode；
            #   False：普通序列，或offload加载完成 / 失败的序列，继续正常准入
            remote_ready_for_decode = self._resolve_waiting_remote_kv(
                seq, skipped_waiting_requests
            )
            if remote_ready_for_decode is None:
                continue

            # 【分支】offload：加载已经完成、block在挂起前就分配好的序列，只需从后缀继续prefill（不再匹配前缀、不再分配）
            offload_resume = self._is_offload_prefill_resume(seq)
            # Side-effect free: the lookup is detached only after `allocate`
            # below, so a request held back by the budget or the pool keeps
            # its prefetched lookup for the next pass. The HBM probe it needs
            # is kept and reused by the admission probe below.
            # 【译】无副作用：查询只在下面的 `allocate` 之后才被分离，所以被预算或池子挡回去的请求，会为下一轮保留它预取的查询。
            #   它需要的HBM探测结果被保留下来，供下面的准入探测复用。
            # 【分支】offload的lookup detach（OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT，第9批）：HBM里的前缀命中已经够用时，
            #   不等外部存储的查询结果就准入
            hbm_probe = None
            probed_tier_kv = (0, 0)
            lookup_detachable = False
            if (
                not remote_ready_for_decode
                and not offload_resume
                and self._lookup_detach_enabled(seq)
            ):
                hbm_probe = self._probe_hbm_prefix(seq)
                probed_tier_kv = self._tier_hit(seq)
                lookup_detachable = self._can_detach_tier_lookup(seq, hbm_probe)
            # 【分支】offload的前缀查询还在路上：暂缓（进prefix_waiters），不阻塞后面的请求
            if (
                not remote_ready_for_decode
                and not offload_resume
                and not lookup_detachable
                and self.kv_connector is not None
                and self._offload_lookup_pending(seq)
            ):
                prefix_waiters.append(seq)
                continue
            # 【分支】问connector：这条prefill要不要先挂起、从远端 / LMCache把KV拉过来；顺便记下外部命中的长度，
            #   供下面的can_allocate对齐状态边界（第9批）
            needs_remote_load = self._query_connector_prefill_match(
                seq,
                skip=remote_ready_for_decode or offload_resume or lookup_detachable,
            )

            # 【分支】PD decode端（第9批）：prefill节点已经把KV和首token T0传过来，本地不做prefill：
            #   置RUNNING、is_first_decode=True、追加T0（开MTP时再补草稿占位），追加到running队尾。
            #   如果本步最终没有排上任何prefill，同一步的decode阶段就会调度它；否则等下一个decode步
            if remote_ready_for_decode:
                self._schedule_first_decode_after_remote_kv(seq)
                continue

            # 【分支】offload加载完成后的续算（第9批，B级）
            if offload_resume:
                # Blocks already held from the pre-park allocate; only re-check
                # the batch budget. No re-match / re-allocate / re-park.
                #
                # `num_tokens`, like Phase 1 above and the non-offload branch
                # below. A seq re-admitted after `preempt` owes KV for the
                # tokens it had already generated, and the offload load can
                # come back covering some of them -- `_mark_offload_load_ready`
                # sets `num_cached_tokens` to whatever the tier returned, which
                # is not bounded by the prompt. Sized against
                # `num_prompt_tokens` that is a non-positive width:
                # `_prefill_chunk_for_budget` answers None, the seq goes back
                # to the head of `waiting`, and the loop breaks -- every tick,
                # forever, starving everything queued behind it. With chunked
                # prefill off it is `_assert_positive_prefill_chunk` that fires
                # instead, in the engine loop.
                # 【译】block在挂起前的allocate时就已持有；这里只重新检查本步预算。不重新匹配 / 不重新分配 / 不重新挂起。
                #
                #   用 `num_tokens`，和上面的Phase 1以及下面的非offload分支一样。一条被 `preempt` 后重新准入的序列，欠着它已生成
                #   token的KV，而offload加载回来的部分可能覆盖其中一些——`_mark_offload_load_ready` 把 `num_cached_tokens` 设成存储层
                #   返回的长度，它不受prompt长度限制。若按 `num_prompt_tokens` 计算，宽度就是非正数：`_prefill_chunk_for_budget` 返回
                #   None，序列回到 `waiting` 队头，循环break——每个tick都这样，永远如此，饿死排在它后面的所有请求。关闭chunked prefill时，
                #   触发的则是引擎循环里的 `_assert_positive_prefill_chunk`。
                tier_hit = seq.num_cached_tokens
                if seq.num_tokens <= tier_hit:
                    # The tier can cover the request whole: its lookup runs
                    # over the entire prompt and is not bounded below it, where
                    # `can_allocate`'s HBM match stops one hash block short so
                    # that a prefill always has something to forward. A wholly
                    # resident request reaches here with nothing to size, which
                    # is the same non-positive width by another route.
                    #
                    # Hand back the trailing block, the one the HBM match would
                    # never have offered. Its KV is present, so recomputing it
                    # costs one block of forward, and the forward has to happen
                    # regardless: the first decode samples from the logits this
                    # prefill produces, and there is nowhere else to get them.
                    # 【译】存储层能覆盖整个请求：它的查询覆盖整个prompt，不在末尾之前停下；而 `can_allocate` 的HBM匹配会少匹配一个hash block，
                    #   保证prefill总有东西可以forward。一个整条都驻留的请求走到这里时没有任何token可算，这是从另一条路得到的同样的非正宽度。
                    #
                    #   把最后一个block交还出来——HBM匹配本来就不会提供的那一个。它的KV是在的，所以重算只花一个block的forward；而这次forward
                    #   无论如何都必须发生：首个decode要从这次prefill产出的logits里采样，没有别的地方能拿到。
                    seq.num_cached_tokens = self._offload_prefill_start(seq)
                # A replayed hit resumes as one: its state slot is fresh, so
                # the forward replays below where the load left
                # `num_cached_tokens` -- the KV-only reach, or the HBM hit if
                # the load did not run (`Sequence.replay_kv_load`).
                # 【译】重放的命中按重放的方式恢复：它的状态槽是新的，所以forward从加载留下的 `num_cached_tokens` 之下开始重放——
                #   即仅KV可达的位置；若加载没有执行，则是HBM命中（`Sequence.replay_kv_load`）。
                replayed = seq.replays_hit
                start = self.block_manager.replay_start(
                    seq, seq.num_cached_tokens, hit=replayed
                )
                num_new_tokens = seq.num_tokens - start
                budget_remaining = self._prefill_room(num_batched_tokens, start)
                chunk = self._prefill_chunk_for_budget(
                    num_new_tokens,
                    budget_remaining,
                    num_batched_tokens,
                    history_len=start,
                    # Admitted (and sized as FIRST) before it parked for the load.
                    # 【译】它在为加载而挂起之前就已经准入（并按FIRST算过大小）。
                    kind=(
                        ChunkKind.WHOLE
                        if self._requires_atomic_prefill(seq)
                        else ChunkKind.NEXT
                    ),
                )
                # 放不下（或只够重算已缓存的行）→ 放回队头，break（与主干同样的FIFO规则）
                if chunk is None or self._waits_below_resume(
                    seq, start, chunk, seq.num_cached_tokens
                ):
                    self.waiting.appendleft(seq)
                    break
                self._resume_prefill(seq, start, hit=replayed)
                chunk = self._finalize_prefill_chunk(seq, start, chunk)
                self._assert_positive_prefill_chunk(
                    chunk, num_new_tokens, budget_remaining
                )
                # This branch `continue`s past both assignments below, so the
                # refresh happens here too: `_mark_offload_load_ready` raised
                # `num_cached_tokens` to the post-load hit, and the stale field
                # is what reaches the API as `cached_tokens` -- the tier would
                # look like it returned nothing. Unconditional, because a seq
                # re-admitted after `preempt()` keeps the field.
                #
                # The tier's own figure, not the possibly-rolled-back one: what
                # it returned is what the API should report, and giving the
                # trailing block back to the forward is this scheduler's
                # decision rather than a smaller hit. A replay recomputes the
                # rows below the tier's figure: they are not a hit.
                # 【译】这个分支会 `continue`，跳过下面的两处赋值，所以这里也要刷新：`_mark_offload_load_ready` 把 `num_cached_tokens`
                #   抬到了加载后的命中长度，而这个过时的字段正是作为 `cached_tokens` 报给API的那个——存储层看起来就像什么都没返回。
                #   无条件刷新，因为 `preempt()` 后重新准入的序列还保留着这个字段。
                #
                #   用存储层自己的数字，而不是可能被回退过的那个：它返回了多少，API就该报告多少；把最后一个block交还给forward是本调度器
                #   的决定，不是命中变少了。重放会重算存储层数字以下的行：它们不算命中。
                seq.prefix_cache_hit_tokens = (
                    seq.num_cached_tokens if replayed else tier_hit
                )
                num_seqs_prefill, num_batched_tokens = self._schedule_prefill_seq(
                    seq,
                    chunk,
                    scheduled_seqs,
                    num_scheduled_tokens,
                    num_seqs_prefill,
                    num_batched_tokens,
                )
                # 排上本步之后，它不再算作"挂起等加载"的序列（挂起名额计数减1）
                self._uncount_inflight_load(seq)
                continue

            # 【分支】要挂起等远端KV：挂起的序列也占着block和一个驻留名额（running + 挂起 ≤ max_num_seqs），满了就放回队头，break
            #   （PD背压，下面每1000个tick打一次日志）
            if (
                needs_remote_load
                and len(self.running) + self._num_parked_remote_kv >= self.max_num_seqs
            ):
                self.waiting.appendleft(seq)
                break

            # Probe cache hits FIRST so budget check sees the real
            # (post-prefix-cache) remaining token count. V4 SWA correctness is
            # enforced inside can_allocate (_swa_bounded_hit bounds the hit to
            # where the trailing-window SWA is present); no post-hoc warmup trim.
            # The detach probe predates `_query_connector_prefill_match`, and
            # `can_allocate` reads the tier's answer (a replaying hit resumes at
            # the tier hit or extends to its KV-only reach) and records the
            # replay on `seq` -- for a replaying model only. Probe again if
            # that answer moved.
            # 【译】先探测缓存命中，这样预算检查看到的是真实的（扣掉前缀缓存之后的）剩余token数。V4的SWA正确性在can_allocate内部保证
            #   （_swa_bounded_hit把命中限制在尾部滑动窗口SWA还在的范围内）；不做事后的预热裁剪。detach探测发生在
            #   `_query_connector_prefill_match` 之前，而 `can_allocate` 会读取存储层的答案（重放命中从存储层命中处恢复，或延伸到
            #   仅KV可达的位置），并把重放记录在 `seq` 上——只对会重放的模型。如果那个答案变了，就重新探测。
            # ★ 主干，对应nano-vllm的block_manager.can_allocate(seq)：返回前缀缓存命中的block数（≥0），池子放不下整条序列时返回-1。
            #   _probe_hbm_prefix用record=False只探测、不提交，真正登记在下面的record_allocation / allocate。
            #   命中怎么算、为什么最后一个block永远不复用：见sequence.py的"科普：前缀缓存"和arg_utils.py enable_prefix_caching上方的【科普】
            #   上面detach路径已经探测过就复用；【分支】会重放的模型在connector的答案变化后重测
            if hbm_probe is None or (
                getattr(self.block_manager, "state_replay_tokens", 0)
                and seq.has_per_req_cache
                and self._tier_hit(seq) != probed_tier_kv
            ):
                hbm_probe = self._probe_hbm_prefix(seq)
            num_cached_blocks, block_hashes = hbm_probe
            # ★ 池子放不下 → 放回队头，break：和nano-vllm一样保持FIFO，防止队头的大请求饿死
            #   （后面的小请求也许塞得进去，但不让它们插队）
            if num_cached_blocks < 0:
                self.waiting.appendleft(seq)
                break
            if (
                seq.replay_kv_load
                and not needs_remote_load
                and len(self.running) + self._num_parked_remote_kv >= self.max_num_seqs
            ):
                # A replay that loads its KV first parks for that load like any
                # remote load (`needs_remote_load` knew only the state-carrying
                # hit), so it needs a parked slot too.
                # 【译】先加载KV的重放，和任何远端加载一样要为这次加载挂起（`needs_remote_load` 只知道携带状态的那个命中），
                #   所以它也需要一个挂起名额。（【分支】第4、9批）
                self.waiting.appendleft(seq)
                break

            # Use num_tokens (not num_prompt_tokens) so preempted seqs re-forward
            # their decoded tokens — preempt() frees their KV blocks but keeps
            # the token_ids, so num_tokens > num_prompt_tokens and those tokens
            # still need KV recomputed.
            # From where the forward will really start: a replayed hit starts
            # below itself (`BlockManager.replay_start`), and its budget counts
            # the rows it recomputes.
            # 【译】用num_tokens（而不是num_prompt_tokens），让被抢占的序列重新forward它已经decode出的token——preempt()释放了它的
            #   KV block，但保留了token_ids，所以num_tokens > num_prompt_tokens，这些token仍需要重算KV。
            #   从forward真正开始的地方算：重放的命中从自身之下开始（`BlockManager.replay_start`），它的预算要算上重算的那些行。
            # resume = 命中的token数（hash_block_size = block_size × DCP，DCP=1时就是block_size）。例：block_size=16、命中3个block
            #   → resume=48；普通模型start=48，本条要算 num_tokens-48 个token
            resume = num_cached_blocks * self.block_manager.hash_block_size
            start = self.block_manager.replay_start(seq, resume, hit=True)
            # 本条要算的token数 = num_tokens - start；long_prefill_token_threshold>0（默认0=关）且开了chunked prefill时，
            #   一条序列每步最多算这么多，长prompt不会一个人吃掉整步预算（_new_prefill_tokens）
            num_new_tokens = self._new_prefill_tokens(seq, start, resume)
            # 【分支】本地合批模式下的混合模型（每请求状态，第4批）：如果running里另一条同前缀的序列即将在prompt末尾附近
            #   产出状态checkpoint，就暂缓这条（最多ttft_max_ticks个tick），等它产出后再准入，就能命中前缀而不必从头重算
            if (
                self._local_prefill_coalescing
                and (not needs_remote_load or self._connector_flag("is_offload"))
                and self._wait_for_inflight_prefix(
                    seq,
                    max(
                        num_cached_blocks * self.block_manager.hash_block_size,
                        seq.offload_joint.kv_prefix_tokens,
                    ),
                )
            ):
                prefix_waiters.append(seq)
                continue
            # 多模态prompt（没有embedding_spans的）不能切开，必须一次整条prefill：视觉embedding要和位置一一对应
            #   （多模态输入见sequence.py里multimodal_data处，第2b批）
            atomic_prefill = self._requires_atomic_prefill(seq)
            # 本步还能给这条多少token：max_num_batched_tokens - 已用；【分支】DP代价均衡时还受代价上限约束
            budget_remaining = self._prefill_room(num_batched_tokens, start)
            # `start` comes from the hit `can_allocate` offered, before
            # `allocate` may disown the boundary. The worker fits
            # `batch.num_cached_tokens`, which `_resume_prefill` sets from the
            # hit kept after that. Sizing has to happen before allocate, so a
            # cache-hit's first chunk is solved against an upper bound on the
            # prefix it will actually run.
            # 【译】`start` 来自 `can_allocate` 提供的命中，那是在 `allocate` 可能放弃边界之前。worker对齐的是 `batch.num_cached_tokens`，
            #   它由 `_resume_prefill` 根据之后保留下来的命中设定。大小必须在allocate之前算出，所以命中缓存的请求，它的第一个chunk
            #   是针对它实际要跑的前缀的一个上界求出来的。
            # ★ 算chunk（_prefill_chunk_for_budget → _chunked_prefill_size）：
            #   开chunked prefill（默认）：chunk = min(要算的, 剩余预算[, 【分支】动态分块上限])；整条放得下就原样返回；
            #     被截断时向下对齐到max(block_size, 64)的整数倍，不够一个对齐单位就返回None（为这点碎片多一次forward不值）——
            #     但本步还空着（num_batched_tokens==0）时照样返回，保证每步总能发出东西
            #   关chunked prefill：放得下就整条；放不下且本步已有别的序列 → None；本步空着就整条返回（真超预算的早被
            #     _unschedulable_reason拦下了）
            # 对比nano-vllm：nano-vllm只允许本步第一条序列被切；ATOM允许本步任意一条被切去填满剩余预算（例子见_schedule上方第1步的C）
            chunk = self._prefill_chunk_for_budget(
                num_new_tokens,
                budget_remaining,
                num_batched_tokens,
                history_len=start,
                kind=ChunkKind.WHOLE if atomic_prefill else ChunkKind.FIRST,
            )
            # ★ 放不下 / 多模态会被切开 / 只够重算已命中的行（重放模型）→ 放回队头，break（还没allocate，不用释放）
            if (
                chunk is None
                or (atomic_prefill and chunk < num_new_tokens)
                or self._waits_below_resume(seq, start, chunk, resume)
            ):
                self.waiting.appendleft(seq)
                break
            # 提交前缀探测的结果（命中统计、joint边界等），复用上面探测时走过的hash链，不再走第二遍
            # ★ allocate：命中的前缀block直接共享（引用计数+1），其余block按整条序列（num_tokens）一次分配好——同nano-vllm，
            #   之后的chunk不再分配；返回False只发生在下面英文注释说的那种情况
            self.block_manager.record_allocation(seq, num_cached_blocks, block_hashes)
            if not self.block_manager.allocate(seq, num_cached_blocks):
                # A state-less joint boundary could not be privatised without
                # risking a shared decoding sequence's blocks (finding #2).
                # Return the seq to waiting for a clean recompute once the pool
                # has room, and stop admitting this pass.
                # 【译】一个不带状态的joint边界无法在不危及某条共享它的decode序列的block的前提下私有化（finding #2：从0重算会就地覆写
                #   共享的前缀block，撕坏另一个持有者的KV，见BlockManager.disown_claimed_prefix）。把序列放回waiting，等池子有空间时
                #   干净地重算，并停止本轮准入。
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break
            # 【分支】投机解码（第7批）：草稿模型会往prompt之后的几个位置写KV，这里提前把那几个位置的block也分配好
            self._allocate_prefill_draft_lookahead(seq)

            # Detached before the alloc notification, which reads the save
            # floor `detach_lookup` sets.
            # 【译】在分配通知之前分离查询，因为分配通知要读 `detach_lookup` 设定的保存下限。（【分支】offload lookup detach，第9批）
            if lookup_detachable and not self.kv_connector.detach_lookup(
                seq, int(seq.num_cached_tokens)
            ):
                # `allocate` kept less HBM prefix than the probe offered (e.g.
                # it disowned a boundary whose state checkpoint was lost), so
                # a tier load may now be worth it. Undo the admission and let
                # the next pass take the ordinary path, whose lookup and
                # remote-load capacity check run before `allocate`.
                # 【译】`allocate` 保留下来的HBM前缀比探测时提供的少（例如它放弃了一个状态checkpoint已丢失的边界），所以现在可能值得
                #   从存储层加载。撤销这次准入，让下一轮走普通路径——普通路径的查询和远端加载容量检查都在 `allocate` 之前进行。
                self._decline_lookup_detach(seq)
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break

            # The hit the request kept, not the one `can_allocate` offered:
            # `allocate` may disown the boundary, and `num_cached_blocks` is the
            # pre-disown count. `seq.num_cached_tokens` also gets the unit right
            # under DCP, and is what EngineStats uses, so the user-visible number
            # and the internal hit rate cannot disagree.
            #
            # Guard: PD decode consumer inherits hit from prefill node;
            # don't clobber with the local hit (always 0 on consumer).
            # 【译】用请求保留下来的命中，而不是 `can_allocate` 提供的那个：`allocate` 可能放弃边界，而 `num_cached_blocks` 是放弃之前
            #   的数。`seq.num_cached_tokens` 在DCP下单位也是对的，而且EngineStats用的就是它，所以用户看到的数字和内部命中率不会对不上。
            #
            #   保护：PD的decode消费端从prefill节点继承命中数；不要用本地命中（在消费端总是0）覆盖它。
            # 解释：prefix_cache_hit_tokens就是API usage里的cached_tokens（RequestOutput.num_cached_tokens取自它）
            if not seq.prefix_cache_hit_tokens:
                seq.prefix_cache_hit_tokens = seq.num_cached_tokens

            self._lookup_detach_declined.discard(seq.id)
            # 【分支】通知connector：block已分配（它据此决定要加载 / 保存哪些block，第9批）
            self._notify_connector_after_prefill_alloc(seq)

            # A replay that loads its KV first parks for that load whatever the
            # lookup answered: the answer was the state-carrying hit, which a
            # KV-only reach can exceed. Running the forward instead would start
            # under the replay's floors before the KV below them arrived.
            # 【译】先加载KV的重放，不管查询怎么回答，都要为那次加载挂起：那个答案是携带状态的命中，而仅KV可达的位置可能超过它。
            #   如果改为直接跑forward，forward会在重放的下限之下开始，而那下面的KV还没到。
            needs_remote_load = self._confirm_remote_load_after_alloc(
                seq, needs_remote_load or seq.replay_kv_load
            )

            # 【分支】要从远端 / LMCache加载：挂起（WAITING_FOR_REMOTE_KVS，sequence.py状态表①），记进skipped，本步不算它；
            #   加载完成后由_resolve_waiting_remote_kv唤醒（第9批）
            if needs_remote_load:
                oj = seq.offload_joint
                if oj.load_hash != -1 and not oj.boundary_tokens:
                    # Two transfers but one report: the first completion would
                    # unpark the request while the other is still writing. Only
                    # reachable when the legs were decided separately -- a joint
                    # load arrives as one event via `_JointPark`.
                    # 【译】两次传输却只有一次报告：第一个完成时就会把请求唤醒，而另一个还在写。只有两条腿是分开决定的时候才会走到这里——
                    #   joint加载经由 `_JointPark` 作为一个事件到达。
                    logger.warning(
                        "seq %s has both a remote KV load and a state load "
                        "pending, with no joint boundary; dropping the state "
                        "load.",
                        seq.id,
                    )
                    if not self.block_manager.cancel_state_load(seq):
                        # Disown could not be backed (finding #2); requeue for a
                        # clean recompute instead of parking a load into shared
                        # blocks.
                        # 【译】放弃边界所需的私有副本分配不出来（finding #2）；重新排队、干净地重算，而不是把一次加载挂到共享的block上。
                        self.block_manager.deallocate(seq)
                        self.waiting.appendleft(seq)
                        break
                self._park_for_remote_load(seq, skipped_waiting_requests)
                self._inflight_prefix_wait.pop(seq.id, None)
                continue

            # 【分支】KV offload的KV + 状态联合加载（joint，第9批，B级）
            if seq.offload_joint.boundary_tokens:
                # The KV leg was refused after the state leg was aimed at the
                # joint boundary, so the state claims a prefix whose KV nobody
                # will fetch. Disown it; `_decide_load_after_alloc` logs why.
                # 【译】KV这条腿在状态那条腿已经瞄准joint边界之后被拒绝了，于是状态认领了一段没人会去取KV的前缀。放弃它；
                #   `_decide_load_after_alloc` 会记录原因。
                logger.debug(
                    "[JOINT-DISOWN] seq %s: KV leg refused at boundary %d; "
                    "recomputing from %d",
                    seq.id,
                    seq.offload_joint.boundary_tokens,
                    seq.num_cached_tokens,
                )
                # Privatise the claimed prefix (finding #2) so recompute-from-0
                # cannot tear a shared decode. cancel_state_load does it when a
                # CPU state load was pending; otherwise disown directly.
                # 【译】把认领的前缀私有化（finding #2），这样从0重算就不会撕坏一个共享的decode。有CPU状态加载挂起时由cancel_state_load
                #   完成；否则直接放弃。
                if seq.offload_joint.load_hash != -1:
                    disowned = self.block_manager.cancel_state_load(seq)
                else:
                    disowned = self.block_manager.disown_claimed_prefix(seq)
                if not disowned:
                    self.block_manager.deallocate(seq)
                    self.waiting.appendleft(seq)
                    break
                seq.num_cached_tokens = 0
                # Partial reset by design: the KV/claim spans are left as-is,
                # matching the pre-record code -- not a full `reset_joint`.
                # 【译】有意只做部分重置：KV / 认领的范围保持原样，与记录化之前的代码一致——不是完整的 `reset_joint`。
                seq.offload_joint.boundary_tokens = 0
                seq.offload_joint.boundary_hash = -1

            if seq.offload_joint.load_hash != -1:
                # The kept state boundary is in LMCache, not HBM. It parks
                # exactly as a KV load does -- same queue, same backpressure,
                # same wake-up -- because to the scheduler both are one event:
                # a transfer into blocks this request already holds.
                # 【译】保留下来的状态边界在LMCache里，不在HBM里。它和KV加载完全一样地挂起——同一个队列、同样的背压、同样的唤醒——
                #   因为对调度器来说两者是同一种事件：往这个请求已经持有的block里做一次传输。
                self._park_for_remote_load(seq, skipped_waiting_requests)
                self._inflight_prefix_wait.pop(seq.id, None)
                continue

            # Refresh, not a duplicate of the set above: that one is guarded
            # on the field being empty, so a seq re-admitted after `preempt()`
            # would keep its previous admission's hit. Unconditional here, where
            # the request is certain to prefill locally. The only path reaching
            # this with an inherited PD hit is a *failed* remote recv, and there
            # the local hit is the truthful number.
            # 【译】这是刷新，不是上面那次赋值的重复：上面那次以字段为空为前提，所以 `preempt()` 后重新准入的序列会保留上一次准入的
            #   命中。这里无条件赋值，因为走到这里的请求肯定在本地prefill。唯一会带着继承来的PD命中走到这里的路径是一次*失败的*
            #   远端接收，而那时本地命中才是真实的数字。
            seq.prefix_cache_hit_tokens = seq.num_cached_tokens
            # ★ 设定forward的起点：num_cached_tokens = start（普通模型就是命中长度；【分支】重放模型会往前退几行）
            self._resume_prefill(
                seq,
                self.block_manager.replay_start(seq, seq.num_cached_tokens, hit=True),
                hit=True,
            )
            if seq.replays_hit:
                # A DeepSeek-V4 bounded replay recomputes the rows below its
                # hit: they are not served from the cache.
                # 【译】DeepSeek-V4的有界重放会重算命中以下的行：它们不是从缓存里拿的。
                seq.prefix_cache_hit_tokens = seq.num_cached_tokens

            # 【分支】offload connector可能推迟这条prefill的一部分（第9批）；【分支】带每请求状态的模型把chunk终点对齐到
            #   状态checkpoint梯级 / block网格（第4批）。普通模型两步都原样返回
            chunk = self._adjust_prefill_chunk_after_alloc(seq, chunk)
            chunk = self._finalize_prefill_chunk(seq, seq.num_cached_tokens, chunk)

            # Re-assert the atomic-prefill invariant against the *final* chunk.
            # The guard above ran on the pre-allocation size, but the two
            # adjusters just above can both shorten the chunk after the fact --
            # `_adjust_prefill_chunk_after_alloc` defers part of a prefill for an
            # offload load, and `_finalize_prefill_chunk` lands the chunk on a
            # state-checkpoint rung. Either can leave a multimodal prompt split
            # even though it passed the pre-alloc check, which would scatter the
            # vision embeddings against the wrong positions (or, on a later
            # chunk once `multimodal_data` is cleared, embed raw `<|media_pad|>`
            # into the KV cache). Take it whole or not at all: undo this
            # admission and wait for a step that can, exactly as the joint-disown
            # path above requeues. `num_cached_tokens` is post-allocation, so
            # this stays correct even if the prefix was disowned.
            # 【译】针对*最终*的chunk重新断言"原子prefill"不变量。上面的检查用的是分配前的大小，但刚才的两个调整器都可能事后缩短
            #   chunk——`_adjust_prefill_chunk_after_alloc` 为offload加载推迟一部分prefill，`_finalize_prefill_chunk` 把chunk落到状态
            #   checkpoint的梯级上。任何一个都可能让一个多模态prompt被切开，即使它通过了分配前的检查；那会把视觉embedding散到错误
            #   的位置上（或者在后面的chunk里、`multimodal_data` 已被清空时，把原始的 `<|media_pad|>` 嵌进KV cache）。要么整条，
            #   要么不做：撤销这次准入，等一个能放下的步，和上面joint放弃路径的重新排队一样。`num_cached_tokens` 是分配后的值，
            #   所以即使前缀被放弃了，这个判断也仍然正确。
            if (
                atomic_prefill and chunk < seq.num_tokens - seq.num_cached_tokens
            ) or self._waits_below_resume(
                seq, seq.num_cached_tokens, chunk, seq.replay_until
            ):
                # A replayed hit that cannot pass its hit this step waits too.
                # 【译】本步越不过自己命中点的重放命中，也要等。
                # 已经allocate过，所以要先deallocate再放回队头
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break

            # ★ 排上：_schedule_prefill_seq记前缀命中统计、累加代价和token预算、置RUNNING / type=PREFILL、追加到running队尾、
            #   加入scheduled_seqs。对比nano-vllm：nano-vllm要等prompt全部算完才把序列移进running
            self._assert_positive_prefill_chunk(chunk, num_new_tokens, budget_remaining)
            num_seqs_prefill, num_batched_tokens = self._schedule_prefill_seq(
                seq,
                chunk,
                scheduled_seqs,
                num_scheduled_tokens,
                num_seqs_prefill,
                num_batched_tokens,
            )
            # Commit only once the request is on the batch, so a requeue above
            # leaves the next sizing drawing the same arm.
            # 【译】只在请求真正进了batch之后才提交，这样上面任何一次重新排队，都会让下一次算大小时抽到同一个"臂"。
            # 【分支】动态分块的标定扫描（第9批）：新请求轮流用两种chunk大小（整步预算 / 它的一个分数），用来拟合延迟模型
            self._commit_calibration_arm()
            self._inflight_prefix_wait.pop(seq.id, None)

        # 【分支】挂起等KV的序列放回waiting队尾：它们这一步不可能就绪，放在队尾不挡新请求；
        #   就绪后由下一步开头的_promote_ready_remote_kv_requests挪到前面
        if skipped_waiting_requests:
            logger.debug(
                "Re-adding %d skipped requests back to waiting queue.",
                len(skipped_waiting_requests),
            )
            self.waiting.extend(skipped_waiting_requests)
        if prefix_waiters:
            # Keep deferred requests in arrival order while bounded lookahead
            # admits independent work without extending their deadlines.
            # 【译】让被暂缓的请求保持到达顺序，同时有界前瞻允许准入互不相关的工作，又不延长它们的截止时间。
            # extendleft会把元素逐个插到左端（顺序因此反过来），所以先reversed一次，放回后就是原顺序（同nano-vllm的写法）
            self.waiting.extendleft(reversed(prefix_waiters))

        # 【分支】PD背压日志：有挂起请求时每1000个tick打印一次
        if self._num_parked_remote_kv > 0 and self._schedule_tick % 1000 == 0:
            logger.info(
                "PD backpressure: parked=%d, waiting=%d, running=%d, "
                "resident=%d/%d, kv_usage=%.2f",
                self._num_parked_remote_kv,
                len(self.waiting),
                len(self.running),
                len(self.running) + self._num_parked_remote_kv,
                self.max_num_seqs,
                self._kv_usage(),
            )

        # 【分支】状态checkpoint的去向、joint KV的统计日志，每100个tick一次（第4、9批）
        if self._schedule_tick % 100 == 0:
            fates = self.block_manager.state_checkpoint_fates()
            if any(fates.values()):
                logger.info(
                    "state checkpoints: %s",
                    " ".join(f"{k}={v}" for k, v in sorted(fates.items())),
                )
            # Its own line rather than a key in the one above: those are what
            # became of a checkpoint, this is whether a prefix in LMCache could
            # be paired with one at all. Silent unless the joint load is on,
            # since both counters stay 0 without it.
            # 【译】单独一行，而不是上面那行里的一个键：上面那些是checkpoint的去向，这里是LMCache里的前缀到底能不能和某个checkpoint
            #   配上对。除非开了joint加载，否则不输出，因为没开时两个计数器都是0。
            chosen = self.block_manager.joint_boundaries
            skips = self.block_manager.joint_skips
            if chosen or skips:
                # The hbm/tier split is the actionable half: both are reuse,
                # but only `tier` costs an image-sized H2D and a park. Under
                # #2045 a ratio dominated by `tier` says the PAGED pool is too
                # small for this concurrency -- checkpoints are being squeezed
                # out by the KV write stream they now share a pool with -- and
                # that is a diagnosis the fork-era split could not make.
                # 【译】hbm / tier的拆分是可以据此采取行动的那一半：两者都是复用，但只有 `tier` 要付出一次镜像大小的H2D拷贝和一次挂起。
                #   在#2045之后，比例由 `tier` 主导说明PAGED池对这个并发度来说太小——checkpoint正被它们现在与之共用一个池的KV写入流
                #   挤出去——这是fork时代的拆分做不出的诊断。
                logger.info(
                    "joint kv: boundaries=%d (state_hbm=%d state_tier=%d) | %s",
                    chosen,
                    self.block_manager.state_hbm_boundaries,
                    self.block_manager.state_tier_boundaries,
                    " ".join(f"{k}={v}" for k, v in sorted(skips.items())),
                )

        total_tokens_num_prefill = sum(num_scheduled_tokens)
        # 【分支】DP代价均衡的统计日志
        if self._tick_cost_limit < math.inf:
            self._log_prefill_balance(total_tokens_num_prefill)

        # ★ prefill优先体现在这里：只要排上了prefill（Phase 1或Phase 2），就组prefill批直接return，本步不做decode
        #   （同nano-vllm的 `if scheduled_seqs: return scheduled_seqs, True`）。running里正在decode的序列这一步停一拍，
        #   这是不混合prefill和decode的代价（engine_core.py文件开头【2】）；想让decode插进chunk之间，
        #   需要PrefillDelayer的decode保护（ATOM_PREFILL_DECODE_INTERVAL>0，见prefill_delayer.py）
        if num_seqs_prefill > 0:
            # A cursor, not a hit count: it starts at the prefix-cache hit and
            # then advances by each finished chunk, so a chunked prompt logs the
            # same req_id repeatedly with this climbing by the previous `new`.
            # Logged as "done" so those repeats don't read as a growing hit.
            # 【译】这是一个游标，不是命中数：它从前缀缓存命中处开始，之后每完成一个chunk前进一次，所以一个被切块的prompt会反复用
            #   同一个req_id打日志，这个值每次增加上一次的 `new`。日志里写作"done"，这些重复就不会被误读成命中在增长。
            # num_cached_tokens_list：每条序列本chunk的起点（KV已算到的位置），也作为batch.num_cached_tokens交给ModelRunner
            num_cached_tokens_list = [
                seq.num_cached_tokens for seq in scheduled_seqs.values()
            ]
            logger.info(
                f"Scheduled prefill batch: {num_seqs_prefill} reqs, "
                f"{total_tokens_num_prefill} new tokens "
                f"(done: {num_cached_tokens_list}, new: {num_scheduled_tokens}), "
                f"req_ids: {tuple(scheduled_seqs.keys())}"
            )
            # 供scheduler_delay_factor（_passed_delay）测量上一次prefill的耗时
            self.prev_prompt = True
            # lip: TODO for prefill/decode mixed batch
            # 【译】lip：TODO，prefill / decode混合批
            # 解释：即"一个batch里同时有prefill chunk和decode序列"，目前未实现；vLLM V1等引擎支持（engine_core.py文件开头【2】）

            # 【分支】PD / offload：发布本步的状态加载 / 保存，生成本步的connector元数据（随batch发给worker，第9批）
            connector_meta_output = None
            if self.kv_connector is not None:
                self._publish_state_loads()
                self._publish_state_stores()
                connector_meta_output = self.kv_connector.build_connector_meta()

            # Freeze, per seq, whether this chunk is the last one this prefill
            # owes. Uses the pre-advance offsets so it is correct whether or not
            # schedule-time advancement runs below.
            #
            # Measured against `num_tokens`, for the same reason
            # `next_token_ids` below is: a seq re-admitted after `preempt`
            # re-forwards the tokens it had already generated, so its prefill
            # runs past the prompt boundary. Ending it at `num_prompt_tokens`
            # would call a middle chunk final and leave the rest of the context
            # with no KV -- the sequence would then decode against blocks still
            # holding the text of whichever request owned them last, and carry
            # on writing that request's answer. Equal to `num_prompt_tokens` on
            # a first admission, which is every seq that was never preempted.
            # 【译】为每条序列冻结"这个chunk是不是这次prefill欠的最后一块"。用推进之前的偏移量，所以不管下面是否在调度时就推进进度，
            #   结果都正确。
            #
            #   按 `num_tokens` 衡量，理由和下面的 `next_token_ids` 相同：`preempt` 后重新准入的序列要重新forward它已生成的token，
            #   所以它的prefill会越过prompt边界。若在 `num_prompt_tokens` 处结束，就会把中间的chunk当成最后一块，让剩下的上下文没有
            #   KV——这条序列接着会在仍存着上一个占用这些block的请求内容的block上decode，并接着写那个请求的回答。首次准入时它等于
            #   `num_prompt_tokens`，从未被抢占过的序列都是这种情况。
            # postprocess据此决定（见postprocess）：最后一块采样出的token才是真正的生成token；不是最后一块的，采样结果丢弃、
            #   序列保持is_partial_prefill=True，下一步由Phase 1续算
            is_final_chunk = [
                (num_cached_tokens_list[i] + int(num_scheduled_tokens[i]))
                >= seq.num_tokens
                for i, seq in enumerate(scheduled_seqs.values())
            ]
            # 【分支】投机解码的草稿模型（EAGLE / MTP，DSpark除外，第7批）：草稿模型比目标模型超前一个位置，中间chunk采不到token，
            #   就用prompt里紧接着的下一个token作它的锚点；最后一块填-1（由采样结果提供）
            next_token_ids = None
            if self.drafter_needs_next_token:
                next_token_ids = []
                for i, seq in enumerate(scheduled_seqs.values()):
                    end = num_cached_tokens_list[i] + int(num_scheduled_tokens[i])
                    next_token_ids.append(
                        -1 if end >= seq.num_tokens else int(seq.token_ids[end])
                    )

            # Reserve midstep checkpoint destinations for the chunks just
            # settled. Here rather than inside `_finalize_prefill_chunk`
            # because a reservation takes a slot off the free list, and
            # admission for this pass only finishes above — planning any
            # earlier would let a checkpoint's destination compete with a
            # request still to be let in. The batch below snapshots what this
            # leaves on each seq.
            # 【译】为刚确定下来的这些chunk预留"中途checkpoint"的目标槽。放在这里而不是 `_finalize_prefill_chunk` 里，是因为一次预留
            #   会从空闲列表拿走一个槽，而本轮的准入到上面才结束——更早规划，会让某个checkpoint的目标槽和一个还没准入的请求抢资源。
            #   下面的batch会快照这一步在每条序列上留下的东西。（【分支】每请求状态的checkpoint，第4批）
            for i, seq in enumerate(scheduled_seqs.values()):
                start = num_cached_tokens_list[i]
                self.block_manager.plan_midstep(
                    seq, start, start + int(num_scheduled_tokens[i])
                )
            # 【分支】第7批：检查草稿要写KV的位置都已经分配了block；没分配的位置会写进物理block 0、悄悄污染别人的数据，这里直接报错
            self._check_draft_writes_allocated(scheduled_seqs.values())

            # 组prefill批（各字段含义见ScheduledBatch类）：total_tokens_num == total_tokens_num_prefill，
            #   total_seqs_num == total_seqs_num_prefill，decode相关字段保持默认（0 / 空）；
            #   显式传入每条的起点num_cached_tokens、is_final_chunk、next_token_ids；
            #   【分支】state_maintenance_ops：取走本批forward之前要执行的状态搬移（第4批）
            prefill_batch = ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=total_tokens_num_prefill,
                total_tokens_num_prefill=total_tokens_num_prefill,
                total_seqs_num=num_seqs_prefill,
                total_seqs_num_prefill=num_seqs_prefill,
                connector_meta_output=connector_meta_output,
                num_cached_tokens=num_cached_tokens_list,
                is_final_chunk=is_final_chunk,
                next_token_ids=next_token_ids,
                state_maintenance_ops=self.block_manager.take_state_maintenance_ops(),
            )
            # batch已经快照了state_fork_src，立刻清掉序列上的fork标志（一次fork只描述这一次forward，见下方_consume_state_forks）
            self._consume_state_forks(scheduled_seqs)

            if self.advance_on_schedule:
                # Advance after batch build (so the batch keeps pre-advance
                # offsets) so the next schedule() issues the following chunk.
                # 【译】在组好batch之后推进（batch保留推进前的偏移量），好让下一次schedule()发出下一个chunk。
                # 【分支】PP（第9批）：head stage同时有pp_size个batch在流水线里，不能等postprocess再推进num_cached_tokens，
                #   否则连续几次schedule会重复发同一个chunk
                self._advance_prefill_on_schedule(
                    scheduled_seqs, num_scheduled_tokens, is_final_chunk
                )

            # ★ 提前return（prefill批）。running的顺序没有动：Phase 1的序列本来就在running里，Phase 2的已追加到队尾
            return (prefill_batch, scheduled_seqs)

        # --- Decode scheduling ---
        # 【译】--- decode调度 ---
        # ★ 主干。走到这里说明本步一条prefill都没排上（没有可做的prefill、都放不下，或delayer否决了prefill）。
        #   对每条running序列做一次decode：不开投机解码时每条1个token；开MTP（mtp_k个草稿）时每条1+mtp_k个
        #   （锚点 + 草稿一起验证，第7批）。此时num_seqs_prefill一定是0（>0的话上面已经return了）
        num_seqs_decode = 0
        num_decode_tokens = 0
        # anchor + drafts if verifying locally, anchor alone otherwise.
        # 【译】在本地验证草稿时是锚点+草稿，否则只有锚点。
        # 锚点 = 上一步采样出的真token；spec_decode_local = 开了投机解码、PP=1、且不是PD producer（见__init__）
        # tokens_per_decode_seq与num_new_tokens取值相同：前者用于预算检查，后者用于can_append / may_append和记账
        spec_width = self.mtp_k if self.spec_decode_local else 0
        tokens_per_decode_seq = spec_width + 1
        num_new_tokens = spec_width + 1
        # 【分支】PD decode端的首个decode：记下这些序列的block（KV是远端写进来的），放进batch（第9批）
        remote_kv_blocks: set[int] = set()
        remote_kv_seq_blocks: dict[int, list[int]] = {}
        # 半途prefill：本步不decode，循环结束后放回running队尾（为什么是队尾，见循环后的英文注释）
        skipped_partial_prefills: list[Sequence] = []
        # Pipeline-parallel: seqs whose sampled token is still in flight cannot
        # be decoded yet. Re-queue them at the tail (like partial prefills) so
        # they are reconsidered once the head releases them post-postprocess.
        # 【译】流水线并行：采样出的token还在流水线里的序列暂时不能decode。把它们重新排到队尾（和半途prefill一样），
        #   等head在postprocess之后释放它们时再考虑。（【分支】PP，第9批）
        skipped_pp_inflight: list[Sequence] = []
        _pp_block = self._pp_inflight_token_block
        # 对比nano-vllm：骨架完全相同——从running队头popleft，block不够就抢占队尾，调度完的再放回队头
        while self.running and num_seqs_decode < self.max_num_seqs:
            # token预算：decode每条只占1+mtp_k个token，默认16384的预算下几乎不会触发（max_num_seqs=512、mtp_k=3时最多512×4=2048个）；
            #   预算设得很小时才起作用。
            #   被截断时没轮到的序列留在running里，scheduled放回队头后它们仍排在后面，下一步还是从队头开始——
            #   和nano-vllm在max_num_seqs处截断的行为一样
            if num_decode_tokens + tokens_per_decode_seq > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            # prompt还没算完，不能decode：本步prefill被delayer否决、或Phase 1没能给它预算时会走到这里
            if seq.is_partial_prefill:
                skipped_partial_prefills.append(seq)
                continue
            # 【分支】PP：它上一步采样的token还在流水线里，还没追加到序列上
            if _pp_block and seq.id in _pp_block:
                skipped_pp_inflight.append(seq)
                continue
            # ★ 抢占：池子不够这条序列追加num_new_tokens个token（需要新block时）就腾地方
            #   ① _preempt_one_running：从running队尾往前找第一条"可抢占"的序列抢占（重算式：删掉占位token、释放全部block、
            #      放回waiting队头，见preempt和sequence.py的"科普：抢占"）；抢到了就continue，再检查一次够不够
            #   ② running里没有可抢的 → 抢占seq自己（preempted_current，本步不调度它）
            #   ③ 自己也不可抢 → 把seq放回running队头，整个decode循环break（blocked_by_pinned_save）
            #   "可抢占" = connector不要求推迟释放它的block（_is_preemptable）：KV正被offload保存 / 加载、或PD正在发送时，
            #     connector还拥有这些源block，提前释放会让对方读到被别人覆盖的数据，所以不能抢（【分支】第9批）；
            #     没有connector时总是可抢
            # 为什么抢队尾：running大致按准入顺序排列，队尾是最晚进来的，已生成的token最少，重算代价最小，也符合先来先服务
            # 例（同nano-vllm：running=[A,B,C,D]，空闲block为0，四条这一步都恰好要一个新block，假设抢占一条正好够一条用）：
            #   取A：不够 → 抢占队尾D（D回waiting队头）→ 够了 → A被调度
            #   取B：不够 → 抢占队尾C → 够了 → B被调度
            #   running空了，循环结束：本步decode [A,B]；waiting=[C,D,...]，C、D之后按"prompt+已生成token"整体重新prefill
            # ATOM多出来的情况：同样的条件下，若C、D的KV正被offload保存（不可抢）：
            #   取A：不够 → 从队尾往前找：D、C不可抢，B可抢 → 抢占B → A被调度；此时running=[C,D]
            #   取C：不够 → 唯一剩下的D不可抢 → 抢C自己也不行 → C放回running队头，break
            #   本步只decode [A]；之后running=[A,C,D]，waiting=[B,...]；D这一步根本没被看到，下一步再试
            # 对比nano-vllm：nano-vllm没有"不可抢"的序列，所以没有③，其余逻辑一致
            blocked_by_pinned_save = False
            preempted_current = False
            while not self.block_manager.can_append(
                seq, num_new_tokens, self.draft_lookahead
            ):
                if self._preempt_one_running():
                    continue
                if self.preempt(seq):
                    preempted_current = True
                    break
                self.running.appendleft(seq)
                blocked_by_pinned_save = True
                break
            if blocked_by_pinned_save:
                break
            # block够了：调度这条序列
            if not preempted_current:
                # 【分支】MTP：上一步草稿模型提出的草稿token，本步由目标模型一起验证（第7批）
                if self.spec_decode_local and seq.spec_token_ids.size > 0:
                    scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                num_seqs_decode += 1
                num_decode_tokens += num_new_tokens
                # For PD first-decode: if T0 was injected, may_append is
                # needed for the new position N. Without T0 injection,
                # blocks were already allocated during prefill.
                # 【译】PD的首个decode：如果注入了T0，就要为新位置N做may_append。没有注入T0时，block在prefill（传输）时就已经分配好了。
                #   （【分支】PD decode端，第9批）
                is_first = getattr(seq, "is_first_decode", False)
                if is_first and seq.block_table:
                    remote_kv_blocks.update(seq.block_table)
                    remote_kv_seq_blocks[seq.id] = list(seq.block_table)
                has_injected_t0 = (
                    is_first
                    and (seq.kv_transfer_params or {}).get("first_token_id") is not None
                )
                # The draft lookahead either way: the transfer filled the
                # prompt's blocks, not the positions past it the draft writes.
                # 【译】无论哪种情况都要分配草稿前瞻：传输填满的是prompt的block，而不是草稿要写的、prompt之后的那些位置。
                # ★ 必要时为新token追加block（同nano-vllm的may_append）；lookahead = 草稿要额外写KV的位置数（第7批）
                self.block_manager.may_append(
                    seq,
                    num_new_tokens if not is_first or has_injected_t0 else 0,
                    lookahead=self.draft_lookahead,
                )
                if is_first:
                    logger.debug(
                        "[PD-FIRST-DECODE] seq %s: num_tokens=%d, "
                        "blocks=%d, injected_t0=%s, "
                        "last_block_num=%d, context_will_be=%d",
                        seq.id,
                        seq.num_tokens,
                        len(seq.block_table),
                        has_injected_t0,
                        seq.last_block_num_tokens,
                        seq.num_tokens,
                    )
                # 观察：is_first_decode在这里、在下面构造ScheduledBatch之前就清掉了，而ScheduledBatch.__init__从seq.is_first_decode
                #   读取is_first_decode_without_local_prefill，所以基类这个decode批里该字段总是全False（aiter_mla.py里依赖它的分支在这条路径上不会触发）；
                #   专为"batch读完再清"设计的_consume_first_decode只在DecodeScheduler._schedule（RapidServe）里调用
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)
                seq.is_first_decode = False

        total_tokens_num_decode = sum(num_scheduled_tokens)

        # 本步调度的序列按原顺序放回running队头（同nano-vllm），默认它们下一步继续decode；结束时由postprocess移出
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs.values()))
        if skipped_partial_prefills:
            # Re-queue skipped partial prefills at the TAIL, not the head.
            #
            # A partial (chunked, prompt-not-done) prefill can land in this
            # decode loop when the cross-DP PrefillDelayer vetoes prefill for a
            # tick (Phase 1 skipped, so num_prefill==0 and the prefill-only
            # early-return doesn't fire). Re-inserting it at the head pins it
            # at running[0]; once it finishes prefill it becomes the batch's
            # position-0 *deferred* seq, which pushes the fresh decode seqs to
            # positions 1..N. TokenIDProcessor.prepare_input_ids then takes the
            # [deferred | new] path and indexes the (compacted)
            # scheduled_spec_decode_tokens array by those shifted positions —
            # running off the end (IndexError: index N out of bounds size N).
            #
            # Appending at the tail keeps the partial out of position 0 (its
            # prefill still resumes: Phase 1 scans all of `running`), so new
            # decode seqs stay contiguous from position 0 and the safe
            # [new | deferred] slice path is used.
            # 【译】把跳过的半途prefill重新排到队尾，而不是队头。
            #
            #   当跨DP的PrefillDelayer在某个tick否决prefill时（Phase 1被跳过，所以num_prefill==0，只做prefill的提前return不会触发），
            #   一个半途（分块、prompt还没算完）的prefill可能落进这个decode循环。把它重新插到队头会把它钉在running[0]；一旦它算完
            #   prefill，就成了batch里位置0的*延迟*序列，把新的decode序列挤到位置1..N。TokenIDProcessor.prepare_input_ids于是走
            #   [deferred | new] 路径，并用这些移了位的位置去索引（压缩过的）scheduled_spec_decode_tokens数组——越界
            #   （IndexError: index N out of bounds size N）。
            #
            #   追加到队尾能让半途序列不在位置0（它的prefill照样会续算：Phase 1扫描整个 `running`），于是新的decode序列从位置0开始
            #   保持连续，走安全的 [new | deferred] 切片路径。
            # 解释："延迟序列"指开着延迟输出时，上一步的token还在GPU上、本步才拿到的序列；输入拼装见第5批
            #   model_runner.py的tokenIDProcessor.prepare_input_ids（类名首字母小写，英文注释里写作TokenIDProcessor）
            self.running.extend(skipped_partial_prefills)
        # 【分支】PP：同样放到队尾
        if skipped_pp_inflight:
            self.running.extend(skipped_pp_inflight)

        # 【分支】PD / offload：与prefill批相同，发布状态加载 / 保存，生成connector元数据（第9批）
        connector_meta_output = None
        if self.kv_connector is not None:
            self._publish_state_loads()
            self._publish_state_stores()
            connector_meta_output = self.kv_connector.build_connector_meta()

        self._check_draft_writes_allocated(scheduled_seqs.values())
        # 组decode批：total_tokens_num == total_tokens_num_decode；total_seqs_num_prefill = num_seqs_prefill = 0；
        #   num_spec_step = mtp_k（本地验证草稿时），scheduled_spec_decode_tokens = 各条的草稿；
        #   【分支】remote_kv_block_ids / remote_kv_seq_blocks：PD首个decode的block
        #   （观察：本仓库里除了scheduler.py，没有别处读取这两个ScheduledBatch字段）
        decode_batch = ScheduledBatch(
            seqs=scheduled_seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num_decode,
            total_tokens_num_decode=total_tokens_num_decode,
            total_seqs_num=num_seqs_prefill + num_seqs_decode,
            total_seqs_num_prefill=num_seqs_prefill,
            total_seqs_num_decode=num_seqs_decode,
            connector_meta_output=connector_meta_output,
            num_spec_step=self.mtp_k if self.spec_decode_local else 0,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            remote_kv_block_ids=sorted(remote_kv_blocks) if remote_kv_blocks else [],
            remote_kv_seq_blocks=remote_kv_seq_blocks,
            # An empty batch cannot execute queued maintenance.
            # For spills that is not merely wasted work: a drained spill is
            # released only once its copy reaches a forward, so draining into a
            # batch that is never forwarded strands every slot it drained.
            # 【译】空batch执行不了排队的维护操作。对spill来说这不只是白费工夫：取出的spill要等它的拷贝进入一次forward才会释放，
            #   所以把它取进一个永远不会forward的batch，会让它取出的每个槽都悬空。（所以scheduled_seqs为空时传None，留给下一批）
            state_maintenance_ops=(
                self.block_manager.take_state_maintenance_ops()
                if scheduled_seqs
                else None
            ),
        )
        self._consume_state_forks(scheduled_seqs)
        # ★ return decode批。scheduled_seqs可能为空（只剩_rejected，或全部被抢占 / 跳过 / 卡在不可抢上）：
        #   EngineCore看到req_ids为空就不forward、不postprocess，只下发connector元数据
        return (decode_batch, scheduled_seqs)

    @staticmethod
    def _consume_state_forks(scheduled_seqs: dict[int, Sequence]) -> None:
        """Clear the fork flags the batch just snapshotted.

        A fork describes one forward: the batch carries `state_fork_src`, and
        every later batch for the same seq must read and write the same slot
        again. Cleared here rather than in the batch constructor so the snapshot
        stays free of side effects on Sequence.
        """
        # 【译】清掉batch刚刚快照过的fork标志。
        #
        #   一个fork描述的是一次forward：batch携带 `state_fork_src`，同一条序列之后的每个batch都必须再次读写同一个槽。
        #   在这里清除，而不是在batch构造函数里清除，是为了让快照对Sequence没有副作用。
        # 【分支】每请求状态的fork（第4批）：fork之后的那一次forward从另一个槽（state_fork_src）读入初始状态、写进自己的槽；
        #   之后都读写自己的槽。调用方：_schedule的prefill批和decode批构造完之后
        for seq in scheduled_seqs.values():
            seq.state_fork_src = -1

    @staticmethod
    def _consume_first_decode(scheduled_seqs: dict[int, Sequence]) -> None:
        """Clear the first-decode flags the batch just snapshotted.

        Same contract as `_consume_state_forks`: the flag describes one
        forward, and the batch constructor reads it off `Sequence`, so it can
        only be cleared once that read has happened.
        """
        # 【译】清掉batch刚刚快照过的首个decode标志。
        #
        #   与 `_consume_state_forks` 的约定相同：这个标志描述的是一次forward，而batch构造函数从 `Sequence` 上读取它，
        #   所以只能在读取发生之后清除。
        # 调用方：只有DecodeScheduler._schedule（RapidServe，本文件末尾）；基类_schedule的decode循环在构造batch之前
        #   就直接清掉了is_first_decode（见那里的"观察"）
        for seq in scheduled_seqs.values():
            seq.is_first_decode = False

    # -- Remote KV / offload admission helpers ------------------------------
    # 【译】-- 远端KV / offload 准入的辅助函数 --
    def _resolve_waiting_remote_kv(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> bool | None:
        """Resolve a ``WAITING_FOR_REMOTE_KVS`` request before admission.

        Returns:
          - ``None`` when the request is still blocked and has been requeued.
          - ``True`` when a P/D consumer should jump to first decode.
          - ``False`` when normal prefill admission should continue.
        """
        if seq.status != SequenceStatus.WAITING_FOR_REMOTE_KVS:
            return False

        if self._consume_failed_remote_kv(seq):
            return False

        if not self._update_waiting_for_remote_kv(seq):
            skipped_waiting_requests.append(seq)
            return None

        seq.status = SequenceStatus.WAITING
        if self._connector_flag("is_offload"):
            self._mark_offload_load_ready(seq)
            return False
        self._uncount_inflight_load(seq)
        return True

    def _consume_failed_remote_kv(self, seq: Sequence) -> bool:
        if not self._pop_req_id(self.failed_recving_kv_req_ids, seq.id):
            return False

        seq.status = SequenceStatus.WAITING
        if not self._connector_flag("is_offload"):
            self._uncount_inflight_load(seq)
        if seq.offload_joint.load_hash != -1 or seq.offload_joint.boundary_tokens:
            # The state never arrived, so the boundary is not this request's
            # history. Disown it exactly as `BlockManager.allocate` does at
            # admission, otherwise the resume runs with `num_cached_tokens > 0`
            # over a group nobody filled -- `has_initial_state` reads that as
            # "the recurrence continues". Before `offload_loaded_tokens` is set
            # below, so that field records the disowned figure. Privatise the
            # claimed prefix (finding #2): the offload resume path reuses this
            # exact block_table without re-allocating, so recompute-from-0 would
            # otherwise tear a shared decode.
            disowned = self.block_manager.disown_claimed_prefix(seq)
            seq.num_cached_tokens = 0
            seq.offload_joint.load_hash = -1
            # A joint boundary that failed on either leg is not this request's
            # history any more than a state-only one is, and leaving it set
            # would let the connector clamp a later lookup to it. Partial reset
            # by design: the KV/claim spans are left as-is, as before.
            seq.offload_joint.boundary_tokens = 0
            seq.offload_joint.boundary_hash = -1
            if not disowned:
                # Cannot back private copies; drop the whole table so the
                # re-admission rebuilds a clean one via can_allocate/allocate
                # (empty block_table disqualifies the offload-resume shortcut).
                self.block_manager.deallocate(seq)
        seq.offload_loaded = False
        seq.offload_loaded_tokens = seq.num_cached_tokens
        seq.offload_load_start_tokens = None
        seq.offload_load_failed = True
        return True

    def _reject_aborted_waiting(self, seq: Sequence) -> None:
        self._inflight_prefix_wait.pop(seq.id, None)
        self._lookup_detach_declined.discard(seq.id)
        has_inflight_load = bool(getattr(seq, "_counted_as_inflight_load", False))
        seq.status = SequenceStatus.FINISHED
        seq.leave_reason = "aborted"
        seq.multimodal_data = None
        self._rejected.append(seq)
        if not has_inflight_load:
            # A lookup can pin CPU KV before HBM allocation succeeds.
            if self._connector_flag("is_offload"):
                self.kv_connector.cancel_pending_load(seq)
            self.deferred_free_blocks[seq.id] = seq
            self._cleanup_aborted_load(seq)
            return

        self.deferred_free_blocks[seq.id] = seq
        terminal_queued = self._pop_req_id(
            self.finished_recving_kv_req_ids, seq.id
        ) or self._pop_req_id(self.failed_recving_kv_req_ids, seq.id)
        terminal_consumed = bool(
            getattr(seq, "offload_loaded", False)
            or getattr(seq, "offload_load_failed", False)
        )
        if terminal_queued or terminal_consumed:
            self._cleanup_aborted_load(seq)
            return
        seq._awaiting_aborted_load_cleanup = True

    def abort_request(self, req_id: int) -> bool:
        """Reclaim waiting cancellations on the event, independent of admission."""
        for seq in self.waiting:
            if seq.id == req_id:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.ABORTED
                self._reject_aborted_waiting(seq)
                return True
        for seq in self.running:
            if seq.id == req_id:
                seq.status = SequenceStatus.ABORTED
                return True
        return False

    def _abort_failed_remote_receive(self, req_id) -> None:
        """End a request whose P/D KV receive failed, as aborted.

        A decode does not prefill a P/D request itself: it is sized and
        compiled for decode, and the request's block table is still the one
        the transfer was writing, which re-admission cannot allocate over. The
        connector reports a failure only once no producer write can still land
        in those blocks (``ATOM_PD_FAILED_WRITE_HOLD_S``), so they are freed at
        once. Offload load failures (``failed_loading``) still resume their
        prefill on this node.
        """
        seq = next(
            (
                s
                for s in self.waiting
                if s.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
                and self._has_req_id([s.id], req_id)
            ),
            None,
        )
        if seq is None:
            logger.error(
                "KV receive failed for request %s, which is not waiting for it; "
                "nothing to abort",
                req_id,
            )
            return
        logger.error("KV receive failed for request %s; aborting it", req_id)
        self.waiting.remove(seq)
        seq.status = SequenceStatus.ABORTED
        self._reject_aborted_waiting(seq)
        # This failure is the terminal report the abort waits for.
        self._finish_aborted_load_cleanup(seq.id)

    def _cleanup_aborted_load(self, seq: Sequence) -> None:
        if hasattr(seq, "_awaiting_aborted_load_cleanup"):
            delattr(seq, "_awaiting_aborted_load_cleanup")
        self._uncount_inflight_load(seq)
        # The one path that never went through `postprocess`, so it finishes
        # the request here rather than before the park.
        callback = getattr(self.kv_connector, "request_finished", None)
        if callable(callback):
            callback(seq)
        # As in `_reject_aborted_waiting`: an abort never sends, so the
        # scheduler retires the send claim rather than waiting for a report.
        self._connector_send_finished(seq.id)
        self._maybe_release_deferred(seq)

    def _finish_aborted_load_cleanup(self, req_id) -> bool:
        seq = self._deferred_sequence(req_id)
        if seq is None or not getattr(seq, "_awaiting_aborted_load_cleanup", False):
            return False
        self._cleanup_aborted_load(seq)
        return True

    def _mark_offload_load_ready(self, seq: Sequence) -> None:
        """Turn a completed offload load into a suffix-prefill resume."""
        loaded = getattr(seq, "offload_loaded_tokens", None)
        load_start = getattr(seq, "offload_load_start_tokens", None)
        logger.debug(
            "[OFFLOAD-WAKE] seq %s: loaded=%s prev_cached=%d num_tokens=%d",
            seq.id,
            loaded,
            seq.num_cached_tokens,
            seq.num_tokens,
        )
        if loaded is not None and loaded > seq.num_cached_tokens:
            promoted = 0
            if load_start is not None and load_start < loaded:
                promoted = self.block_manager.publish_loaded_prefix(
                    seq,
                    start_token=load_start,
                    end_token=loaded,
                )
                logger.info(
                    "[OFFLOAD-PROMOTE] seq=%s loaded_range=%d:%d "
                    "gpu_indexed_tokens=%d",
                    seq.id,
                    load_start,
                    loaded,
                    promoted,
                )
            seq.offload_promoted_tokens = promoted
            seq.num_cached_tokens = loaded
            # Report the extended CPU-offload hit without reducing any
            # prefix-cache hit inherited from an upstream prefill node.
            seq.prefix_cache_hit_tokens = max(seq.prefix_cache_hit_tokens, loaded)
        seq.offload_load_start_tokens = None
        seq.offload_loaded = True
        # A state load moves no KV, so the block above does nothing for it --
        # `num_cached_tokens` is already the boundary the state covers, and all
        # that is left is to stop calling the load pending. A joint load is the
        # one case where both halves fire, and `_JointPark` held the wake until
        # both legs reported.
        # Partial reset by design: the KV/claim spans are left as-is, as the
        # pre-record code did -- not a full `reset_joint`.
        seq.offload_joint.load_hash = -1
        seq.offload_joint.boundary_tokens = 0
        seq.offload_joint.boundary_hash = -1

    def _is_offload_prefill_resume(self, seq: Sequence) -> bool:
        """True when offload already owns blocks and should resume suffix prefill.

        This avoids a second prefix lookup and, more importantly, avoids calling
        ``BlockManager.allocate`` again for a sequence whose block table was
        allocated before it parked for the LMCache load.
        """
        return self._connector_flag("is_offload") and resumes_offload_prefill(seq)

    def _lookup_detach_enabled(self, seq: Sequence) -> bool:
        """Whether `seq`'s tier lookup is a candidate for detaching at all.

        False when the feature is off, an earlier detach of this request was
        invalidated by `allocate`, the connector has no offload tier or no
        detach support, joint KV/state offload is active (a tier hit can
        change where it resumes), or the request loads remotely.
        """
        conn = self.kv_connector
        return not (
            not envs.OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT
            or seq.id in self._lookup_detach_declined
            or conn is None
            or not self._connector_flag("is_offload")
            or getattr(conn, "can_detach_lookup", None) is None
            or getattr(conn, "detach_lookup", None) is None
            or getattr(self.block_manager, "state_offload", None) is not None
            or (seq.kv_transfer_params or {}).get("do_remote_prefill")
        )

    def _probe_hbm_prefix(self, seq: Sequence) -> tuple[int, list[int]]:
        """`can_allocate(record=False)`: the HBM hit in blocks (-1 if the
        request does not fit) and the hash chain it walked.

        Done once per admission attempt; the result is shared by the detach
        decision and the admission itself, which run in the same pass with no
        pool change in between.
        """
        block_hashes: list[int] = []
        num_cached_blocks = self.block_manager.can_allocate(
            seq,
            record=False,
            block_hashes=block_hashes,
        )
        return num_cached_blocks, block_hashes

    @staticmethod
    def _tier_hit(seq: Sequence) -> tuple[int, int]:
        """The tier's answer as `can_allocate` reads it: the LMCache MP hit and
        its KV-only reach (`kv_prefix_tokens`, `kv_only_tokens`)."""
        joint = getattr(seq, "offload_joint", None)
        return (
            int(getattr(joint, "kv_prefix_tokens", 0) or 0),
            int(getattr(joint, "kv_only_tokens", 0) or 0),
        )

    def _can_detach_tier_lookup(
        self, seq: Sequence, hbm_probe: tuple[int, list[int]]
    ) -> bool:
        """Whether admission may proceed without waiting for `seq`'s lookup.

        Side-effect free. Requires an HBM hit and the connector's consent
        (`can_detach_lookup`, which applies `OFFLOAD_MIN_LOAD_TOKENS`). The
        probe is the one admission uses, so it sees the HBM prefix
        `_decide_load_after_alloc` will see.
        """
        hit_blocks = hbm_probe[0]
        if hit_blocks <= 0:
            return False
        num_hbm_tokens = hit_blocks * self.block_manager.hash_block_size
        return bool(self.kv_connector.can_detach_lookup(seq, num_hbm_tokens))

    @property
    def _lookup_detach_declined(self) -> set:
        """Requests whose detach `allocate` invalidated; they take the
        ordinary lookup path until admitted or aborted. Created on first use,
        so schedulers built without `__init__` (test doubles) work."""
        return self.__dict__.setdefault("_lookup_detach_declined_ids", set())

    def _decline_lookup_detach(self, seq: Sequence) -> None:
        self._lookup_detach_declined.add(seq.id)

    def _offload_lookup_pending(self, seq: Sequence) -> bool:
        """Whether the connector's lookup for `seq` is still in flight."""
        pending = getattr(self.kv_connector, "lookup_pending", None)
        return bool(pending(seq)) if pending is not None else False

    def _query_connector_prefill_match(self, seq: Sequence, *, skip: bool) -> bool:
        """Ask the connector whether this prefill should park for remote KV."""
        if skip or self.kv_connector is None:
            return False
        ext_tokens, needs_remote_load = self.kv_connector.get_num_new_matched_tokens(
            seq
        )
        # Keep the lookup's answer, not just the park flag: `can_allocate` runs
        # next and a hybrid's two legs have to be aimed at one boundary, for
        # which this is the KV leg's ceiling. In tokens, and absolute -- the
        # connector reports what it found *beyond* what the seq already has.
        seq.offload_joint.kv_prefix_tokens = int(ext_tokens) + int(
            seq.num_cached_tokens
        )
        return needs_remote_load

    def _schedule_first_decode_after_remote_kv(self, seq: Sequence) -> None:
        """P/D path: a remote prefill completed, so schedule first decode."""
        # num_cached_tokens still describes this consumer's pre-existing prefix.
        # Received suffix blocks and the producer's inherited API hit are not
        # local hits. Record before appending the producer's first output token.
        self._record_cache_reuse(seq)
        seq.status = SequenceStatus.RUNNING
        seq.is_first_decode = True
        first_token_id = (seq.kv_transfer_params or {}).get("first_token_id")
        if first_token_id is not None:
            seq.append_token(first_token_id)
            seq._injected_t0 = first_token_id
            if self.mtp_k > 0:
                drafts = list(
                    (seq.kv_transfer_params or {}).get("draft_token_ids") or []
                )[: self.mtp_k]
                for d in drafts:
                    seq.append_token(int(d))
                # Pad to the full verify window regardless of what the producer
                # sent -- normally nothing, since a producer does not
                # speculate. The first decode slices the trailing `mtp_k + 1`
                # tokens, so a short seq makes that slice reach back into the
                # prompt: T0 lands at the end of the window instead of at its
                # head and the consumer verifies prompt tokens as drafts.
                # Placeholders are what postprocess leaves on every other
                # running seq, and the target rejects any the drafter's absence
                # left it disagreeing with.
                for _ in range(self.mtp_k - len(drafts)):
                    seq.append_token(self.eos_token_id)
                seq.spec_token_ids = np.asarray(drafts, dtype=np.int32)
                # These trailing slots await verification by the first local
                # decode -- the same contract as postprocess's placeholders, so
                # record the same width. `preempt()` reads it to decide what to
                # strip, and the window is always `mtp_k` wide however few
                # drafts the remote sent: the padding above is `eos_token_id`,
                # so counting only `len(drafts)` would leave that EOS behind as
                # real context. T0 is excluded: it is a real generated token.
                seq.num_placeholder_tokens = self.mtp_k
        logger.debug(
            "[PD-TRANSITION] seq %s: num_tokens=%d, "
            "num_prompt=%d, blocks=%d, first_token=%s, "
            "last_5_tids=%s",
            seq.id,
            seq.num_tokens,
            seq.num_prompt_tokens,
            len(seq.block_table),
            first_token_id,
            seq.token_ids[-5:],
        )
        self.running.append(seq)

    def install_chunk_latency_model(self, predictor: ChunkSizePredictor) -> bool:
        """Install a useful calibrated model and end the calibration sweep.

        One-shot. A model that cannot shrink over a prefix several base chunks
        deep is rejected, leaving fixed chunking enabled. There is no refresh:
        the workers stop timing once they have answered with a fit.
        """
        self._calibrating = False
        reference_len = self.max_num_batched_tokens * USEFUL_SHRINK_REFERENCE_CHUNKS
        if self.max_model_len:
            reference_len = min(self.max_model_len, reference_len)
        if not predictor.predicts_useful_shrink(
            base_chunk_size=self.max_num_batched_tokens,
            history_len=reference_len,
        ):
            logger.warning(
                "Ignoring dynamic chunking calibration a=%.3e b=%.3e gamma=%.3e: "
                "it predicts no useful shrink after a %d-token prefix, so "
                "chunking stays fixed",
                predictor.quadratic_coeff,
                predictor.linear_coeff,
                predictor.prefix_coeff,
                reference_len,
            )
            return False
        self.dynamic_chunk_predictor = predictor
        logger.info(
            "Dynamic chunking latency model installed: a=%.3e b=%.3e c=%.3e gamma=%.3e",
            predictor.quadratic_coeff,
            predictor.linear_coeff,
            predictor.constant_coeff,
            predictor.prefix_coeff,
        )
        return True

    def abandon_chunk_latency_calibration(self) -> None:
        """End the calibration sweep without a model, leaving chunking fixed."""
        self._calibrating = False
        logger.warning(
            "Dynamic chunking calibration gave up; chunking stays fixed at %d tokens",
            self.max_num_batched_tokens,
        )

    @property
    def dynamic_chunking_active(self) -> bool:
        """Whether chunk sizes can still differ from the fixed budget."""
        return self.enable_dynamic_chunking and (
            self._calibrating or self.dynamic_chunk_predictor is not None
        )

    def _record_prefill_supply(self) -> None:
        """Sample this step's prefill sources into the sole-prefill window.

        Counts the requests mid-prefill plus the waiting ones admission could
        take, before this step admits any. Capped at 2: only "at most one"
        matters, and the cap keeps a long queue from costing a full scan.
        """
        sources = self._partial_prefill_count
        if sources < 2:
            waiting = itertools.islice(self._waiting_prefills(), 2 - sources)
            sources += sum(1 for _ in waiting)
        self._recent_prefill_supply.append(min(sources, 2))

    def _dynamic_chunk_limit(self, history_len: int, kind: ChunkKind) -> int | None:
        """Return the dynamic chunk ceiling, or None for fixed chunking.

        Dynamic sizing requires a single prefill source over the whole census
        window. Before calibration completes, it returns the sweep size.
        """
        if kind is not ChunkKind.NEXT:
            # A new request is being sized: forget any arm an earlier query drew.
            self._calibration_arm_drawn = False
        if kind is ChunkKind.WHOLE or not self.dynamic_chunking_active:
            return None
        if not has_sole_prefill(self._recent_prefill_supply):
            self._dynamic_chunking_counts["multi_prefill"] += 1
            return None
        if self.dynamic_chunk_predictor is None:
            self._dynamic_chunking_counts["sweep"] += 1
            return self._calibration_sweep_chunk(kind)
        try:
            limit = self.dynamic_chunk_predictor.predict(
                history_len=history_len,
                base_chunk_size=self.max_num_batched_tokens,
                smooth_factor=self.dynamic_chunking_smooth_factor,
                alignment=self._chunk_alignment(),
                max_chunk_size=self.max_num_batched_tokens,
                min_chunk_size=self.dynamic_chunking_min_chunk_size,
            )
        except ValueError:
            logger.exception(
                "Dynamic chunking prediction failed; chunking stays fixed at %d tokens",
                self.max_num_batched_tokens,
            )
            self.dynamic_chunk_predictor = None
            return None
        outcome = "unreachable" if limit is None else "solved"
        self._dynamic_chunking_counts[outcome] += 1
        return limit

    def _log_dynamic_chunking_counts(self) -> None:
        counts = self._dynamic_chunking_counts
        logger.info(
            "Dynamic chunking since last report: %d chunk(s) from the latency "
            "model, %d calibration-sweep chunk(s); fixed budget kept for %d "
            "unreachable, %d alongside other prefills",
            counts["solved"],
            counts["sweep"],
            counts["unreachable"],
            counts["multi_prefill"],
        )
        counts.clear()

    def _chunk_alignment(self) -> int:
        return chunk_alignment(self.block_manager.block_size)

    def _align_chunk_down(self, chunk: int) -> int:
        alignment = self._chunk_alignment()
        return max((chunk // alignment) * alignment, alignment)

    def _calibration_arms_collapsed(self) -> bool:
        """True when the two sweep arms are not two distinct aligned sizes."""
        budget = self.max_num_batched_tokens
        full = self._align_chunk_down(budget)
        small = self._align_chunk_down(
            max(
                budget // CALIBRATION_SWEEP_RATIO,
                self.dynamic_chunking_min_chunk_size,
            )
        )
        return small >= full

    def _commit_calibration_arm(self) -> None:
        """The request just admitted is the one last sized; its arm is now current."""
        if self._calibration_arm_drawn:
            self._calibration_small_arm = not self._calibration_small_arm
            self._calibration_arm_drawn = False

    def _calibration_sweep_chunk(self, kind: ChunkKind) -> int | None:
        """Select one of two per-request chunk sizes for an identifiable fit.

        Requests alternate between the full budget (None) and its
        ``CALIBRATION_SWEEP_RATIO`` fraction, aligned down the same way the
        full arm is. A first chunk takes the arm the last admitted request did
        not; sizing only reads that state, and `_commit_calibration_arm` moves
        it when the request is admitted. So a requeued request, or a fit
        probe, draws the same arm again rather than burning the other one.
        Later chunks keep the current arm, which is their own: the sole-prefill
        gate keeps a second request from being swept while one is mid-prefill.
        """
        small = self._calibration_small_arm
        if kind is ChunkKind.FIRST:
            small = not small
            self._calibration_arm_drawn = True
        if not small:
            return None
        return self._align_chunk_down(
            max(
                self.max_num_batched_tokens // CALIBRATION_SWEEP_RATIO,
                self.dynamic_chunking_min_chunk_size,
            )
        )

    def _chunked_prefill_size(
        self,
        num_new_tokens: int,
        budget_remaining: int,
        num_batched_tokens: int,
        *,
        history_len: int,
        kind: ChunkKind,
    ) -> int:
        """Tokens to forward this step, or 0 to leave the request for the next.

        A chunk cut short by the budget is floored onto the block grid. When
        what's left is under one aligned unit the answer is 0: that sliver buys
        the request nothing — it needs a later step to finish either way — so
        all it does is split the prefill into an extra forward, off the grid.
        Flooring is also what frees the remainder the sliver is made of, so
        without the 0 case the alignment manufactures its own tail; a
        16384-token budget was going out as `..., 640, 10`.

        Never 0 for an empty batch: something has to go out each step, or a
        `max_num_batched_tokens` below the alignment would stall forever.
        """
        dynamic_limit = self._dynamic_chunk_limit(history_len, kind)
        # None means the equal-latency chunk is unreachable for this prefix, so
        # the fixed budget stands. A reachable limit is the equal-latency or
        # sweep chunk size; the feature-off path is also None.
        chunk = min(
            num_new_tokens,
            budget_remaining,
            dynamic_limit if dynamic_limit is not None else num_new_tokens,
        )
        if chunk >= num_new_tokens:
            return chunk
        aligned = chunk - chunk % self._chunk_alignment()
        return aligned or (0 if num_batched_tokens else chunk)

    def _prefill_chunk_for_budget(
        self,
        num_new_tokens: int,
        budget_remaining: int,
        num_batched_tokens: int,
        *,
        history_len: int,
        kind: ChunkKind,
    ) -> int | None:
        if self.enable_chunked_prefill:
            return (
                self._chunked_prefill_size(
                    num_new_tokens,
                    budget_remaining,
                    num_batched_tokens,
                    history_len=history_len,
                    kind=kind,
                )
                or None
            )
        if num_new_tokens > budget_remaining and num_batched_tokens > 0:
            return None
        return num_new_tokens

    def _finalize_prefill_chunk(self, seq: Sequence, start: int, chunk: int) -> int:
        """Align a prefill chunk to the state checkpoint boundary and vet forks.

        Two jobs, both about `BlockManager`'s state checkpoints:

        1. A checkpoint can only be kept where a forward ends exactly on a rung
           of the ladder — otherwise the state is ahead of the hash it would be
           filed under. So land chunks on that grid (plus, at most once, a
           position this seq itself was seen to want), shortening to the
           previous rung; `BlockManager.checkpoint_cut` owns the arithmetic, so
           that it cannot drift from the rule deciding what actually gets kept.
        2. The forward carrying a fork has to fill the request's new slot by
           itself. If the budget left a chunk too short for that, drop the fork
           rather than the request — unless the source is shared with another
           request forking off it this step, which rules out taking it over. Then
           the fork stays and the chunk is held at `min_fork_tokens` instead;
           `can_allocate` only offered a resumable boundary with that many
           prompt tokens behind it, so the tokens are there to forward. Only the
           rolling state class forks, so this job asks it directly.

        No-op for models without per-request state, and for any prompt shorter
        than one checkpoint interval.
        """
        bm = self.block_manager
        if self._requires_atomic_prefill(seq):
            return chunk
        target = bm.checkpoint_cut(seq, _cut_floor(seq, start, start), start + chunk)
        if target:
            chunk = target - start
        else:
            # A readable class is never cut onto a rung; keep the next
            # chunk's start on the block grid instead, or every snapshot
            # position in it is off the grid too.
            chunk = bm.align_midstep_chunk(seq, start, chunk)
        # `cancel_state_fork` runs only when the first two hold, and returning
        # False is what leaves the fork in place — hence holding the chunk open.
        if (
            seq.state_fork_src >= 0
            and chunk < bm.state.min_fork_tokens
            and not bm.cancel_state_fork(seq)
        ):
            chunk = bm.state.min_fork_tokens
        return chunk

    def _checkpoint_room(self, seq: Sequence, finished: bool) -> int:
        """Tokens the forward after this one carries, for `hash_decode_blocks`.

        0 means "do not checkpoint here", for any of three reasons:

        - the request stops on this step, so there is nothing after it: no
          forward to fork into the slot a checkpoint would hand away, and no
          batch to issue a copy on either;
        - the seq is still on its prompt, where the prefill call site has
          already decided using the prompt's own remainder;
        - speculative decode, for a class that checkpoints by forking. Two
          reasons, and either is enough. The spec path's state index has no
          read-side counterpart (see
          `GDNAttentionMetadataBuilder.prepare_state_indices`), so a fork must
          never reach it. And a spec step commits `1 + accepted_drafts` tokens,
          which is what a fork's successor actually gets — the rest is rolled
          back and re-forwarded — so no promise made here can be kept.
          Checkpointing during *prefill* stays safe on the same models:
          `min_fork_tokens` guarantees prompt is left over, and prompt always
          forwards down the non-spec path.

        A class that checkpoints by copying is bound by none of that: the
        destination is complete when the copy lands, so any decode step will do.

        Otherwise plain decode carries exactly one token, and whether that is
        enough to fill a fresh slot is the backend's `min_fork_tokens` to say.
        """
        if finished or seq.type != SequenceType.DECODE:
            return 0
        if self.mtp_k and self.block_manager.state.transfer.forks:
            return 0
        return 1

    @staticmethod
    def _assert_positive_prefill_chunk(
        chunk: int, num_new_tokens: int, budget_remaining: int
    ) -> None:
        assert chunk > 0, (
            f"chunk must be positive: {chunk=}, "
            f"{num_new_tokens=}, {budget_remaining=}"
        )

    def _record_cache_reuse(self, seq: Sequence) -> None:
        """Account local prefix reuse at prefill or first PD decode admission."""
        if self.engine_stats.cache_enabled:
            # Hit counts are in hash blocks — one block_table entry spans
            # `block_size * dcp_world_size` tokens — so scaling by block_size
            # would under-report by the DCP factor.
            hbs = self.block_manager.hash_block_size
            # The reuse ceiling, mirroring `can_allocate`'s match loop, which
            # runs over `range(n_hash_blocks - 1)`: prefill must forward at
            # least one block to produce sampler logits, so the trailing block
            # is never a reuse candidate and no cache can be charged for it.
            # Floored at 0 for a sequence shorter than one hash block, whose
            # ceiling is genuinely zero — nothing about it is reusable.
            n_hash_blocks = (seq.num_tokens + hbs - 1) // hbs
            num_reusable_tokens = min(max(n_hash_blocks - 1, 0) * hbs, seq.num_tokens)
            # `num_cached_tokens` sits above the HBM walk once a load landed
            # -- that is the feature. Those tokens are not paged-pool reuse, so
            # they go in their own series rather than inflating `cached` and
            # making `wanted - cached` negative. `seq.num_cached_tokens` is
            # deliberately NOT clamped: the forward must skip what was loaded.
            wanted_tokens = seq.num_wanted_hit_blocks * hbs
            # Where it resumed, not where a replaying forward starts below it
            # -- unless it is a DeepSeek-V4 bounded replay, standing in for a
            # checkpoint the gates found missing: the rows it recomputes are
            # reuse lost for want of one (`wanted - cached`), not a hit.
            cached_tokens = (
                seq.num_cached_tokens
                if seq.replays_hit
                else max(seq.num_cached_tokens, seq.replay_until)
            )
            hbm_cached_tokens = min(cached_tokens, wanted_tokens)
            offload_tokens = cached_tokens - hbm_cached_tokens
            self.engine_stats.update_cache(
                hbm_cached_tokens,
                seq.num_tokens,
                seq.num_compressed_hit_blocks * hbs,
                wanted_tokens,
                num_reusable_tokens,
                num_offload_tokens=offload_tokens,
            )

    def _schedule_prefill_seq(
        self,
        seq: Sequence,
        chunk: int,
        scheduled_seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        num_seqs_prefill: int,
        num_batched_tokens: int,
    ) -> tuple[int, int]:
        num_seqs_prefill += 1
        self._record_cache_reuse(seq)
        self._tick_cost += self._chunk_cost(seq.num_cached_tokens, chunk)
        num_batched_tokens += chunk
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        self.running.append(seq)
        scheduled_seqs[seq.id] = seq
        num_scheduled_tokens.append(chunk)
        return num_seqs_prefill, num_batched_tokens

    def _allocate_prefill_draft_lookahead(self, seq: Sequence) -> None:
        """Allocate the blocks past the prompt that its draft round writes.

        Right after `allocate`, which sized the table on the prompt alone.
        `can_allocate` already charged these blocks (`draft_lookahead_blocks`),
        so the units are there. Skipped for a P/D remote prefill: the transfer
        pairs this table block for block with the producer's, and nothing
        prefills here; its first decode allocates the lookahead instead.
        """
        if not self.draft_lookahead or (seq.kv_transfer_params or {}).get(
            "do_remote_prefill"
        ):
            return
        self.block_manager.may_append(seq, 0, lookahead=self.draft_lookahead)

    def _check_draft_writes_allocated(self, seqs) -> None:
        """Refuse a batch whose draft round would write past a row's blocks.

        A position past the last allocated block reads block id 0 from the
        zeroed block-table tail, and the KV write lands in physical block 0 --
        silently, in whatever request or state checkpoint owns it. Every path
        that schedules a forward has to have allocated through
        `len(seq) + draft_lookahead` (`may_append(..., lookahead=)`); this is
        where one that did not fails loudly. Remote-prefill rows are skipped:
        nothing runs a forward for them here.
        """
        if not self.draft_lookahead:
            return
        bm = self.block_manager
        for seq in seqs:
            if bm.draft_writes_allocated(seq) or (seq.kv_transfer_params or {}).get(
                "do_remote_prefill"
            ):
                continue
            raise RuntimeError(
                f"seq {seq.id}: {len(seq.block_table)} blocks of "
                f"{bm._effective_block_size()} tokens do not reach position "
                f"{len(seq) + self.draft_lookahead - 1}, the end of this "
                f"forward's draft round (len(seq)={len(seq)}, draft lookahead "
                f"{self.draft_lookahead}); its KV write would land in block 0"
            )

    def _notify_connector_after_prefill_alloc(self, seq: Sequence) -> None:
        if self.kv_connector is not None:
            self.kv_connector.update_state_after_alloc(seq)

    def _confirm_remote_load_after_alloc(
        self, seq: Sequence, needs_remote_load: bool
    ) -> bool:
        if not needs_remote_load:
            return False
        if hasattr(self.kv_connector, "should_park_for_load_after_alloc"):
            return self.kv_connector.should_park_for_load_after_alloc(seq)
        return True

    def _settle_state_load(self, req_id, ok: bool) -> None:
        """Release the state group a load reserved. No-op for an id the state
        index never issued, and for a scheduler built without a block manager
        (the connector test doubles)."""
        settle = getattr(
            getattr(self, "block_manager", None), "settle_state_load", None
        )
        if settle is not None:
            settle(req_id, ok=ok)

    def _abandon_state_load(self, req_id) -> None:
        """Release a state group a load reserved *without* voting the outcome.

        The 'neither outcome' release, for a joint load whose state H2D landed
        intact but whose KV leg failed (an LMCache LRU miss on the KV chunk).
        Settling that as a failure would call `StateOffloadIndex.fail_load` and
        `forget(h)`, permanently un-advertising a state image whose bytes are
        still present -- the next request over that prefix could have reloaded
        it. `abandon_load` pops the pending reservation and releases the orphan
        slot but keeps the hash loadable. Same guards as `_settle_state_load`:
        no-op for an id the state index never issued and for a scheduler built
        without a block manager (the connector test doubles)."""
        abandon = getattr(
            getattr(self, "block_manager", None), "abandon_state_load", None
        )
        if abandon is not None:
            abandon(req_id)

    def _state_store_pending_cap(self) -> int:
        """The running-plus-queued cap for state-tier stores.

        Read off the connector's public `max_pending_saves` -- the canonical
        `_offload_common.max_pending_saves` value it computed from the
        process-wide `OFFLOAD_MAX_PENDING_SAVES` setting and
        `OFFLOAD_COPY_WORKERS`, and the exact bound the KV leg's
        `_may_emit_save` enforces. Falls back to the env reader for a connector
        (or test double) that never bounded its save queue (`max_pending_saves`
        is None, as on dense) and, under `kv_connector: multi`, reaches the
        bound through the state-tier sub -- `MultiConnectorScheduler` bounds
        nothing of its own, so without the sub probe the composite can silently
        use a different default from the state-tier implementation. The public
        accessor means neither read reaches through the delegating shell's
        `_impl` any more.
        """
        conn = self.kv_connector
        cap = getattr(conn, "max_pending_saves", None)
        if cap is None:
            # Under `multi` the bound lives on the state-tier sub (itself a
            # delegating shell), reached the same way the state face is routed.
            sub = getattr(conn, "_state_tier_sub", None)
            tier = sub() if callable(sub) else None
            if tier is not None:
                cap = getattr(tier, "max_pending_saves", None)
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            return cap
        return _offload_max_pending_saves()

    def _publish_state_stores(self) -> None:
        """Hand this pass's ready checkpoints to the connector for the CPU tier.

        Beside `_publish_state_loads` and on the same schedule, so one pass
        makes one decision about the tier. The cap is the connector's public
        `max_pending_saves` (see `_state_store_pending_cap`) -- the same bound
        the KV leg's `_may_emit_save` enforces, and for the same reason: each
        outstanding transfer holds the bytes it is reading out of the pool, so
        the queue depth is also how much of the pool a slow backend can pin.

        `take_state_stores` pins as it hands over, so anything the connector
        will not carry has to be settled here rather than left to the
        reconciler's full timeout -- the units are already held.
        """
        if self.kv_connector is None:
            return
        take = getattr(getattr(self, "block_manager", None), "take_state_stores", None)
        if take is None:
            return
        stores = take(self._state_store_pending_cap())
        if not stores:
            return
        accepted = False
        enqueue = getattr(self.kv_connector, "enqueue_state_stores", None)
        if enqueue is not None:
            accepted = bool(enqueue(stores))
        if accepted:
            return
        offload = getattr(getattr(self, "block_manager", None), "state_offload", None)
        if offload is not None:
            offload.stores_refused += len(stores)
        logger.warning(
            "state offload: %s did not carry %d state store(s); releasing "
            "their units now. Either this connector has no state tier (the "
            "index should not have been installed against it), or the "
            "delegating shell is missing an `enqueue_state_stores` forwarder "
            "to the implementation that does -- check both before assuming "
            "the configuration is wrong.",
            type(self.kv_connector).__name__,
            len(stores),
        )
        for op, _units in stores:
            # `attempted=False`: this refusal is already counted once above as
            # `stores_refused`; the units must still be released, but the store
            # never reached a worker, so it is not a `stores_failed`.
            self.block_manager.settle_state_store(op, ok=False, attempted=False)

    def _publish_state_loads(self) -> None:
        """Hand this pass's state-tier loads to the connector, before it builds
        its metadata.

        Drained here rather than at the park, because the connector publishes
        once per pass and a load handed over twice would write the same entry
        into a group the first transfer is already filling.
        """
        if self.kv_connector is None:
            return
        # Guarded like the `_settle_state_load` / `_publish_state_stores`
        # siblings: a scheduler built without a block manager (the connector
        # test doubles) would otherwise AttributeError on the bare deref.
        take = getattr(getattr(self, "block_manager", None), "take_state_loads", None)
        if take is None:
            return
        loads = take()
        if not loads:
            return
        accepted = False
        if hasattr(self.kv_connector, "enqueue_state_loads"):
            accepted = bool(self.kv_connector.enqueue_state_loads(loads))
        if accepted:
            return
        # Nothing will carry them and the requests are parked against a report
        # that would never come, so wake them for recompute -- `failed_loading`
        # means "recompute over the blocks already allocated".
        #
        # `abandon_state_load`, not `settle(ok=False)`: nothing was attempted,
        # so this is not a miss to count against the index.
        logger.warning(
            "state offload: %s did not carry %d state load(s); waking those "
            "requests to recompute. Either this connector has no state tier, "
            "or the delegating shell is missing an `enqueue_state_loads` "
            "forwarder -- the class named here is the shell, not the "
            "implementation behind it.",
            type(self.kv_connector).__name__,
            len(loads),
        )
        for req_id, _h, _group in loads:
            self.block_manager.abandon_state_load(req_id)
            self.failed_recving_kv_req_ids.append(req_id)

    def _park_for_remote_load(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> None:
        skipped_waiting_requests.append(seq)
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        self._count_inflight_load(seq)

    def _count_inflight_load(self, seq: Sequence) -> None:
        if not getattr(seq, "_counted_as_inflight_load", False):
            if metrics := getattr(self, "metrics", None):
                metrics.start_kv_wait(seq)
            self._num_parked_remote_kv += 1
            seq._counted_as_inflight_load = True

    def _uncount_inflight_load(self, seq: Sequence) -> None:
        if getattr(seq, "_counted_as_inflight_load", False):
            # Normal completions already closed the timer when worker quorum
            # arrived. Abort/rejection must also release its bookkeeping.
            if metrics := getattr(self, "metrics", None):
                metrics.finish_kv_wait(seq.id, succeeded=False)
            self._num_parked_remote_kv -= 1
            seq._counted_as_inflight_load = False

    def _adjust_prefill_chunk_after_alloc(self, seq: Sequence, chunk: int) -> int:
        if self.kv_connector is not None and hasattr(
            self.kv_connector, "adjust_prefill_chunk_after_alloc"
        ):
            return self.kv_connector.adjust_prefill_chunk_after_alloc(seq, chunk)
        return chunk

    def _is_preemptable(self, seq: Sequence) -> bool:
        return not self._connector_should_defer_free(seq)

    def _preempt_one_running(self) -> bool:
        for index in range(len(self.running) - 1, -1, -1):
            candidate = self.running[index]
            if not self._is_preemptable(candidate):
                continue
            del self.running[index]
            if self.preempt(candidate):
                return True
            self.running.insert(index, candidate)
        return False

    def preempt(self, seq: Sequence) -> bool:
        if not self._is_preemptable(seq):
            return False
        self.total_preemptions += 1
        seq.status = SequenceStatus.WAITING
        # Strip the placeholder tokens postprocess appended, so the recompute
        # restarts from real context only.
        #
        # Deferred output appends its placeholder on every decode step, not
        # only under speculation: it is on unless PP or a P/D handoff turns it
        # off, so a plain TP-only engine takes that path for every running
        # sequence. The placeholder is `eos_token_id`, and postprocess
        # overwrites it in place one step later -- a step this sequence will
        # never reach, because it is being preempted now. Left in place it
        # stops being a placeholder: the recompute prefills a context ending in
        # `<|endoftext|>` and the model duly starts a new document, and the
        # same token is handed back to the caller as generated output. With
        # `ignore_eos=False` the request just stops there, which reads as a
        # coherent answer that ends before it answers anything.
        #
        # `num_placeholder_tokens` is the only width that describes what is
        # actually there: postprocess appends `mtp_k + is_deferred_out -
        # num_rejected` slots and records exactly that count. Re-deriving it
        # here as `mtp_k + num_rejected` agrees only when deferred output is
        # off AND nothing was rejected -- on a TP-only MTP engine it leaves one
        # `eos_token_id` behind (the case above), and with `num_rejected = r`
        # it deletes `2r - 1` real tokens and their logprobs instead.
        strip = seq.num_placeholder_tokens
        # A truncating stop can move `num_tokens` back over the placeholders
        # (postprocess only rewrites the count, not the arrays), so bound the
        # deletion by what is present and by the prompt, which is never a
        # placeholder. Without this, `del token_ids[-strip:]` on an oversized
        # `strip` clears the whole list and drives `num_completion_tokens`
        # negative.
        strip = min(
            strip,
            len(seq.output_tokens),
            max(0, seq.num_tokens - seq.num_prompt_tokens),
        )
        if strip > 0:
            del seq.token_ids[-strip:]
            del seq.output_tokens[-strip:]
            seq.num_tokens -= strip
            # Each placeholder pushed a 0.0 alongside it (postprocess patches
            # that in place too), so they go together or the two lists stop
            # describing the same tokens.
            if seq.return_logprobs and len(seq.logprobs) >= strip:
                del seq.logprobs[-strip:]
        seq.num_rejected = 0
        seq.num_bonus_tokens = 0
        seq.num_placeholder_tokens = 0
        seq.spec_token_ids = np.array([], dtype=np.int32)
        seq.is_first_decode = False
        if seq.is_partial_prefill:
            seq.is_partial_prefill = False
            self._partial_prefill_count -= 1
        # This free runs no `request_finished`; drop any stall-escaped save
        # tracker now so the connector's save loop cannot later read these
        # surrendered blocks (the mutator half of `should_defer_free`'s escape).
        self._connector_release_stalled_save(seq)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return True

    def _advance_prefill_on_schedule(
        self,
        scheduled_seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        is_final_chunk: list[bool],
    ) -> None:
        """Advance chunked-prefill progress at schedule time (pipeline-parallel).

        Applies the num_cached_tokens / is_partial_prefill bookkeeping that
        postprocess() otherwise does, so the head can schedule the next chunk
        before this one's output returns. Hash registration is NOT done here — it
        stays in postprocess where the forward has computed the KV.
        """
        for i, seq in enumerate(scheduled_seqs.values()):
            seq.num_cached_tokens += int(num_scheduled_tokens[i])
            now_partial = not is_final_chunk[i]
            if now_partial != seq.is_partial_prefill:
                self._partial_prefill_count += 1 if now_partial else -1
                seq.is_partial_prefill = now_partial

    @staticmethod
    def _pp_inflight_req_ids(batch: ScheduledBatch):
        """Req_ids that will produce a not-yet-appended token in this batch.

        A whole decode batch, or the final chunk of a chunked prefill, yields a
        token the head has not appended yet. mark_pp_inflight() and
        release_pp_inflight() MUST use this same predicate so every add() has a
        matching discard(); a seq spanning two batches (middle then final chunk)
        is only blocked/released by its final-chunk batch.
        """
        final = batch.is_final_chunk
        if batch.total_seqs_num_decode > 0:
            yield from batch.req_ids
        elif final is not None:
            for i, req_id in enumerate(batch.req_ids):
                if final[i]:
                    yield req_id

    def mark_pp_inflight(self, batch: ScheduledBatch) -> None:
        """Head: block re-scheduling of seqs whose token is now in flight.

        Blocking them until release_pp_inflight() prevents decoding against a
        stale token while the pipeline is filled.
        """
        for req_id in self._pp_inflight_req_ids(batch):
            self._pp_inflight_token_block.add(req_id)

    def release_pp_inflight(self, batch: ScheduledBatch) -> None:
        """Head: release seqs blocked by mark_pp_inflight after postprocess.

        Discards exactly the set mark_pp_inflight() added (see
        _pp_inflight_req_ids) so a middle-chunk batch cannot clear a block that a
        later final-chunk batch set for the same seq.
        """
        for req_id in self._pp_inflight_req_ids(batch):
            self._pp_inflight_token_block.discard(req_id)

    def _batch_seq_lookup(self, seqs: Iterable[Sequence] | None) -> dict[int, Sequence]:
        """Look up batch seqs directly; falls back to ``running`` if None."""
        if seqs is not None:
            return {seq.id: seq for seq in seqs}
        return {seq.id: seq for seq in self.running}

    def register_prefill_hashes(
        self, batch: ScheduledBatch, seqs: Iterable[Sequence] | None = None
    ) -> None:
        """Hash blocks for middle chunked-prefill chunks that skip postprocess."""
        if not self.block_manager.enable_prefix_caching:
            return
        if batch.is_final_chunk is None:
            return
        seq_by_id = self._batch_seq_lookup(seqs)
        for i, req_id in enumerate(batch.req_ids):
            seq = seq_by_id.get(req_id)
            if seq is None or not seq.block_table:
                logger.warning(
                    "register_prefill_hashes: seq %s unavailable "
                    "(possible preemption leak under PP)",
                    req_id,
                )
                continue
            chunk = int(batch.num_scheduled_tokens[i])
            start_tokens = int(batch.num_cached_tokens[i])
            self.block_manager.hash_blocks(
                seq,
                chunk,
                start_tokens=start_tokens,
                replay_until=_scheduled_replay_until(batch, i),
            )

    def postprocess(
        self,
        seqs: list[Sequence],
        fwd_output: ScheduledBatchOutput,
        stream_output_queue=None,
        batch: ScheduledBatch = None,
    ) -> list[Sequence]:
        """Process model outputs: update tokens, check stop conditions, free blocks.

        Also updates num_cached_tokens for prefill seqs and tracks which seqs
        are still mid-prefill (partial chunks) so their sampled tokens can be
        discarded.
        """
        # 【译】处理模型输出：更新token、检查停止条件、释放block。
        #
        #   同时为prefill序列更新num_cached_tokens，并记录哪些序列仍处于prefill中途（部分chunk），以便丢弃它们采样出的token。
        #
        # ════════════════════════════════════ postprocess 总览 ════════════════════════════════════
        # 调用方（grep核对）：EngineCore._process_engine_step_inner（每步forward之后，传batch=本步的ScheduledBatch，只在req_ids非空时调用）、
        #   pp_engine_core.py（PP：某个batch的结果回到head时，同样传batch）、RapidServe的DecodeEngineCore._process_engine_step（不传batch）
        # 输入：seqs        本步调度的序列（schedule()返回的dict的values）
        #       fwd_output  ModelRunner rank 0返回的ScheduledBatchOutput：req_ids、每个请求得到的token元组、num_rejected、num_bonus、
        #                   草稿token、logprobs；用get_idx(req_id)按请求id查行号
        #       stream_output_queue  EngineCore的流式队列；batch  本步的ScheduledBatch
        # 输出：本步结束的序列列表；流式增量（每个请求一个RequestOutput）作为一个列表放进stream_output_queue
        # 前置概念（这里不重复）：
        #   为什么要延迟输出、它怎样把CPU时间藏到GPU时间后面 → engine_core.py文件开头【3】
        #   num_cached_tokens / num_placeholder_tokens等计数字段一步步怎么变 → sequence.py类开头的字段演化总表
        #   投机解码怎样接受/拒绝草稿、num_bonus_tokens与num_rejected的含义 → sequence.py里num_draft_tokens上方的"科普：投机解码"
        #
        # 对比nano-vllm：nano-vllm是zip(seqs, token_ids)逐条做 hash_blocks → num_cached_tokens+= → append_token → 判停（eos / max_tokens）
        #   → 释放。ATOM多出来的东西和原因：
        #   ① 延迟输出：fwd_output里的token属于"上一步"的batch，和本步的seqs不是同一批，所以改成遍历全部running、按req_id对号入座；
        #      还没拿到真实token的位置先放占位（值为eos_token_id），下一步原地覆盖（见下面时间线①）
        #   ② 投机解码：一步可能新增多个token（被接受的草稿 + 主模型自己的1个），要记num_rejected / num_bonus，停止位置可能落在中间
        #   ③ 停止条件更多：stop字符串（按token匹配）、generation_config里的stop_token_ids；一步多token时还可能越过max_tokens
        #   ④ 流式输出：每步把新增token打包成RequestOutput（nano-vllm只在结束时整体取结果）
        #   ⑤ chunked prefill的进度（num_cached_tokens）按本步batch在这里推进；PP时改为在调度时推进（advance_on_schedule）
        #   ⑥ 【分支】KV connector（PD分离 / KV offload，第9批）：结束的序列若还在被传输读取，block不能立刻释放
        #
        # 流程图（每一段对应下面代码里同名的一段）：
        #   ┌─ 第0段 按"本步batch"记账（不管这一步有没有拿到token）───────────────────────────────────────────┐
        #   │ 多模态：带embedding_spans的请求 → multimodal_cache_ready=True                                       │
        #   │ 【分支】offload：本步prefill的每请求状态已拷完 → _state_initialized_after_alloc=True                 │
        #   │ PP（advance_on_schedule）：进度已在调度时推进 → 只登记前缀hash、记下"中间chunk"                        │
        #   │ 非PP：每个PREFILL序列 hash_blocks → num_cached_tokens += chunk → 更新is_partial_prefill             │
        #   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        #                                                │
        #   ┌─ 第1段 for seq in self.running：给每条序列"对号入座"fwd_output里的结果 ────────────────────────────┐
        #   │ ABORTED？ ─是→ _mark_finished("aborted")，放进finished                                             │
        #   │ fwd_output里没有它（idx=None）？ ─是→ 跳过（例：延迟输出下刚做完prefill的那一步）                     │
        #   │ 延迟输出且它在本步做prefill？ ─是→ 跳过（那行结果属于抢占前的旧执行）                                 │
        #   │ 本步 / 上一步是中间chunk？ ─是→ 跳过（中间chunk采样出的token没有意义）                               │
        #   │ 写入token：有占位 → 原地覆盖末尾的占位（记num_rejected / num_bonus）；否则append                      │
        #   │ PD：把producer采样的T0补在本步输出前面；投机：spec_token_ids = 新一组草稿                            │
        #   │ num_tokens = 真实长度（去掉草稿占位和被拒绝的草稿）                                                   │
        #   │ 判停：stop序列 ─未命中→ eos ─未命中→ stop_token_ids（ignore_eos跳过后两项），再和max_tokens取更早者    │
        #   │       截断num_tokens和要发给客户端的new_tokens                                                      │
        #   │ TTFT、生成token计数、hash_decode_blocks；命中停止 → _mark_finished(原因)                              │
        #   │ 有新token或已结束 → RequestOutput进stream_outputs；结束的进finished_seqs                              │
        #   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        #                                                │
        #   stream_outputs作为一个列表放进stream_output_queue（一步一个）
        #                                                │
        #   ┌─ 第2段 给本步batch里仍RUNNING、prompt已算完的序列追加下一步的占位 ─────────────────────────────────────┐
        #   │ 追加 num_placeholder - num_rejected 个eos_token_id，num_placeholder_tokens加上这个数                   │
        #   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        #                                                │
        #   ┌─ 第3段 释放已结束的序列 ──────────────────────────────────────────────────────────────────────────┐
        #   │ 没有connector → deallocate；【分支】connector还在读这些block → 只留被读的block，或整条延迟释放        │
        #   │ running.remove(seq)                                                                               │
        #   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        #                                                │
        #   更新吞吐统计，return finished_seqs ─→ EngineCore先发("STREAM", stream_outputs)，再发finished_seqs
        #   （顺序由EngineCore保证：CoreManager的输出线程依赖"先交付最后一批token、再处理结束"，见engine_core.py调用本函数之后的注释）
        #
        # 时间线①：延迟输出、不开MTP（num_placeholder=1）。eos_token_id=2，prompt=[15,27,99]（位置0~2），模型依次生成40、41、2
        #   "fwd返回"是forward(第N步)交给本函数的结果——它属于第N-1步的batch；2*表示占位（值就是eos_token_id）；npt=num_placeholder_tokens
        #   step │ 本步调度                  │ fwd返回     │ postprocess之后的token_ids │ npt │ 说明
        #   ─────┼──────────────────────────┼────────────┼───────────────────────────┼─────┼─────────────────────────────────────────
        #    N   │ S的prefill（位置0~2）     │ {}，没有S   │ [15,27,99,2*]             │  1  │ idx=None跳过；第2段追加占位。GPU上第N步采出40
        #   N+1  │ S decode，位置3           │ {S:(40,)}   │ [15,27,99,40,2*]          │  1  │ 覆盖2*→40，再追加占位（代表GPU上正在采的那个）
        #        │ 输入不读token_ids[3]，     │             │                           │     │ 判停只看本步拿到的(40,)，占位不会被误判成eos
        #        │ 而是GPU上的prev_token_ids  │             │                           │     │
        #   N+2  │ S decode，位置4           │ {S:(41,)}   │ [15,27,99,40,41,2*]       │  1  │
        #   N+3  │ S decode，位置5           │ {S:(2,)}    │ [15,27,99,40,41,2]        │  0  │ 覆盖后命中eos → _mark_finished("eos")、释放block
        #        │ （调度时还不知道已结束）   │             │                           │     │ 本步替S发射的这次decode白算了
        #   N+4  │ S已不在running            │ {S:(…,)}    │ —                         │  —  │ 第N+3步替S算出的token没人认领，被丢弃
        #   结论：结束要晚一步才被发现、多算一步；token_ids末尾平时总挂着1个占位，preempt按npt把它删掉（见preempt里的说明）
        #
        # 时间线②：投机解码k=mtp_k=3 + 延迟输出（num_placeholder=k+1=4，num_placeholder_width=k=3）；E=占位（eos_token_id）
        #   每个验证步的窗口 = [锚点, 3个草稿]。CPU调度时还不知道上一步接受了几个，按"全部接受"排位置，
        #   由runner在GPU一侧按上一步的num_rejected回退（aiter_attention.py / aiter_mla.py里的context_lens -= num_rejected，第5、7批）
        #   prompt=[11,12,13,14]（位置0~3）
        #   ⓐ prefill那一步的postprocess：还没有S的输出，第2段追加4个E
        #        token_ids = [11 12 13 14  E  E  E  E]                      len=8   npt=4
        #                                  T0 d1 d2 d3 的位置                下一步V1验证[T0@4, d1@5, d2@6, d3@7]
        #   ⓑ 拿到prefill的输出(50,)：num_new_token=1、num_rejected=0 → offset=0，required_placeholders=4
        #        覆盖token_ids[-4]=50 → [11 12 13 14 50  E  E  E]          npt=4-1=3
        #        真实长度num_tokens = 8 - 3 - 0 = 5；spec_token_ids = V1正在验证的草稿[61 62 63]
        #        第2段追加4-0=4个E → [11 12 13 14 50  E E E E E E E]        len=12  npt=7
        #          （V1的输出最多4个，落在位置5~8；V2的3个草稿位置是9~11）
        #   ⓒ 拿到V1的输出：主模型接受61、62，在位置7给出70（草稿63被拒）
        #        这一行token = (61, 62, 70)：num_new_token=3，num_rejected=1，num_bonus=2
        #        offset=k=3（num_new_token+num_rejected=4≠1），required_placeholders=4+3=7，从末尾第7个E开始覆盖：
        #          前：[11 12 13 14 50  E  E  E  E E E E]
        #          后：[11 12 13 14 50 61 62 70  E E E E]                   npt=7-3=4（=k+num_rejected）
        #        真实长度num_tokens = 12 - 3 - 1 = 8（位置0~7）
        #        第2段追加4-1=3个E → len=15，npt=7
        #          （V2是在len=12时调度的，CPU按位置8~11排；GPU用V1的num_rejected=1回退成[70@7, 草稿@8~10]）
        #   稳态：每次postprocess之后末尾都是2k+1个E = 在途那一步的k+1个输出位 + 再下一步的k个草稿位
        #
        # 判停为什么按token比较、为什么命中的stop字符串留在输出里：
        #   EngineCore进程里没有tokenizer（反分词在前端），按文本匹配就得在调度热循环里每步反分词每条序列；
        #   所以前端先把stop字符串分词成stop_token_sequences（llm_engine.py的preprocess_fanout第4步，那里有"分词方式不同就匹配不上"的例子），
        #   这里只比较token_ids末尾的几个整数。截断时num_tokens只退到"stop序列最后一个token"之后，所以命中的stop序列本身、
        #   以及eos / stop_<id>那个token都留在输出里（OpenAI返回的文本不含stop字符串，见protocol.py里stop字段的注释）
        #
        # ---- 第0段：按本步batch记账 ----
        # Remember which seqs were already in the middle of chunked prefill
        # before this postprocess call mutates seq.is_partial_prefill below.
        #
        # In deferred-output mode, fwd_output.token_ids is one step late. If a
        # seq finishes its final prompt chunk in this call, the token we see
        # here is still from the previous partial chunk, not the real first
        # generated token. Keep the old partial state so we can drop that stale
        # token later in this loop.
        # 【译】记住哪些序列在本次postprocess修改下面的 seq.is_partial_prefill 之前，就已经处于chunked prefill的中途。
        #
        #   延迟输出模式下 fwd_output.token_ids 晚一步。如果某条序列在这次调用里算完了prompt的最后一个chunk，
        #   这里看到的token仍来自上一个部分chunk，而不是真正的第一个生成token。保留旧的partial状态，
        #   以便在后面的循环里丢掉这个过时的token。
        prev_partial_ids: set[int] = set()
        # Middle-chunk req_ids whose sampled token must be dropped, frozen from
        # batch.is_final_chunk (a later schedule() may have already flipped the
        # live seq.is_partial_prefill while this batch was in flight).
        # 【译】需要丢弃采样token的中间chunk的req_id，取自 batch.is_final_chunk 的冻结值
        #   （这个batch还在途时，后面的某次 schedule() 可能已经改写了实时的 seq.is_partial_prefill）。
        # 解释：只在PP（advance_on_schedule）时填充：PP的head同时有pp_size个batch在途，调度时就推进了进度
        pp_middle_chunk_ids: set[int] = set()
        # batch=None（RapidServe的DecodeEngineCore）时为空字典，下面按batch做的记账全部跳过
        running_by_id = {seq.id: seq for seq in self.running} if batch else {}
        if batch is not None:
            # 多模态：本步带着embedding_spans发出去的请求，ModelRunner已经缓存了它的视觉编码结果 →
            #   之后调度器只发span元数据、不再发像素张量（见sequence.py的multimodal_cache_ready、multimodal_runtime.py）
            for request_id, data in getattr(batch, "multimodal_data", {}).items():
                seq = running_by_id.get(request_id)
                if seq is not None and "embedding_spans" in data:
                    seq.multimodal_cache_ready = True
        # 本步做prefill的请求id（ScheduledBatch.req_ids的前total_seqs_num_prefill个）
        num_prefill = int(getattr(batch, "total_seqs_num_prefill", 0))
        prefill_ids = set(batch.req_ids[:num_prefill]) if num_prefill else set()
        # 【分支】KV offload的每请求状态（第4、9批）
        if self._connector_flag("is_offload") and num_prefill:
            for req_id in batch.req_ids[:num_prefill]:
                seq = running_by_id.get(req_id)
                if seq is not None and seq.has_per_req_cache:
                    # StateTransfer.copy runs inside this prefill forward. Mark
                    # the live destination ready only after that forward has
                    # returned, before connector metadata for a later batch.
                    # 【译】StateTransfer.copy 在这次prefill的forward内部执行。只有在这次forward返回之后、
                    #   为后面的batch生成connector元数据之前，才把实时的目标状态标为就绪。
                    seq._state_initialized_after_alloc = True
        if batch is not None and self.advance_on_schedule:
            # Progress already advanced at schedule time; publish prefix-cache
            # hashes at the chunk's pre-advance offset and record non-final chunks.
            # See _batch_seq_lookup: seq may have left running.
            # 【译】进度已在调度时推进；在chunk推进之前的偏移处发布前缀缓存hash，并记录非最终chunk。
            #   见 _batch_seq_lookup：seq可能已经离开了running。
            # 【分支】PP（第9批）：num_cached_tokens / is_partial_prefill已由_advance_prefill_on_schedule在调度时推进，
            #   这里只补"forward算完KV"之后才能做的hash登记（start_tokens传chunk推进前的偏移）
            seq_by_id = self._batch_seq_lookup(seqs)
            final = batch.is_final_chunk
            for i, req_id in enumerate(batch.req_ids):
                seq = seq_by_id.get(req_id)
                if seq is None or final is None or not seq.block_table:
                    continue
                is_final = final[i]
                chunk = int(batch.num_scheduled_tokens[i])
                start_tokens = int(batch.num_cached_tokens[i])
                self.block_manager.hash_blocks(
                    seq,
                    chunk,
                    start_tokens=start_tokens,
                    replay_until=_scheduled_replay_until(batch, i),
                )
                if not is_final:
                    pp_middle_chunk_ids.add(req_id)
        elif batch is not None:
            # 非PP（默认）：prefill进度在这里推进
            # 对比nano-vllm：nano-vllm在postprocess开头对每条seq做 hash_blocks + num_cached_tokens += num_scheduled_tokens；
            #   这里相同，只是只对PREFILL类型做（ATOM的decode不再累加num_cached_tokens，见sequence.py里num_cached_tokens的说明）
            for i, req_id in enumerate(batch.req_ids):
                seq = running_by_id.get(req_id)
                if seq is None or seq.type != SequenceType.PREFILL:
                    continue
                if seq.is_partial_prefill:
                    prev_partial_ids.add(seq.id)
                # chunk：本步给它算了多少个prompt token
                chunk = int(batch.num_scheduled_tokens[i])
                # Register prefix-cache hashes for blocks the prefill step
                # just finalized BEFORE advancing num_cached_tokens. Deferred
                # from BlockManager.allocate() so a hash is only published
                # once the block's KV has been computed by the forward —
                # correct under chunked-prefill where one block may span
                # multiple steps (hash_blocks clips to fully-filled blocks).
                # 【译】在推进 num_cached_tokens 之前，为prefill这一步刚写完的block登记前缀缓存hash。这一步从
                #   BlockManager.allocate() 推迟到这里，保证只有在forward算出block的KV之后才发布hash——在chunked prefill下
                #   一个block可能跨好几步，这样才正确（hash_blocks只截取完全填满的block）。
                # 解释：登记范围是[num_cached_tokens, num_cached_tokens+chunk)里填满的block（block_manager.hash_blocks）；
                #   replay_until是"有界重放"时本步实际开始重算的位置，它下面的block属于命中部分，不重复登记
                #   （有界重放见sequence.py里Replay类上方的"科普：状态checkpoint与有界重放"，第4批）
                self.block_manager.hash_blocks(
                    seq, chunk, replay_until=_scheduled_replay_until(batch, i)
                )
                seq.num_cached_tokens += chunk
                # Ask the scheduler what it decided rather than re-deriving it
                # from lengths here. It froze `is_final_chunk` at pre-advance
                # offsets, against the whole admitted length -- which for a
                # preempted seq runs past the prompt. By the time this runs
                # `num_tokens` may already have grown by this step's sampled
                # token, so no comparison against it is safe either. The length
                # fallback is for callers that hand a batch without the field;
                # the scheduler always sets it.
                # 【译】直接问调度器当时的决定，而不是在这里根据长度重新推导。调度器在chunk推进之前的偏移处、针对整个准入长度
                #   冻结了 is_final_chunk——对被抢占过的序列，这个长度会超过prompt。到这里运行时，num_tokens 可能已经因为本步采样的
                #   token而增长，所以和它比较也不安全。长度比较的兜底只给那些传入的batch没有这个字段的调用方用；调度器总是会设置它。
                # 解释：被抢占过的序列重新准入时，"prompt + 已生成的token"整体当作prompt重算，所以要算完的长度大于num_prompt_tokens
                final = getattr(batch, "is_final_chunk", None)
                if final is not None:
                    now_partial = not final[i]
                else:
                    now_partial = seq.num_cached_tokens < seq.num_prompt_tokens
                if now_partial != seq.is_partial_prefill:
                    # _partial_prefill_count：running里处于prefill中途的序列数，schedule()靠它在纯decode步跳过扫描running（见__init__）
                    self._partial_prefill_count += 1 if now_partial else -1
                    seq.is_partial_prefill = now_partial

        # ---- 取出forward的结果（延迟输出时它属于上一步的batch）----
        # prev_token_ids[idx]：第idx行请求得到的token元组（runner已按-1截掉被拒绝的位置）：
        #   普通decode和prefill是1个；投机验证步是"被接受的草稿 + 主模型自己的1个"，最多k+1个
        # draft_token_ids[idx]：与这行token同一步提出的k个草稿（延迟输出时同样晚一步到达，正是此刻GPU上在验证的那一组）；
        #   没有跑propose时为None
        # is_deferred_out：runner是否处在延迟输出模式（tokenIDProcessor.is_deferred_out，条件见engine_core.py文件开头【3】）
        # token_logprobs：{req_id: 本步采样token的logprob}；行尾英文注释是它的类型，没有请求要logprobs时为None
        prev_token_ids = fwd_output.token_ids
        draft_token_ids = fwd_output.draft_token_ids
        is_deferred_out = fwd_output.is_deferred_out
        token_logprobs = fwd_output.logprobs  # Optional[dict[int, float]]
        # update token_ids with the actual sampled token ids
        # 【译】用实际采样出的token id更新token_ids

        finished_seqs = []
        stream_outputs = []
        num_new_generation_tokens = 0

        # 需要占位：延迟输出（下一个真实token还在GPU上）或本引擎自己验证草稿（草稿位置要先占住）
        #   按__init__与model_runner.py的条件，spec_decode_local为True时延迟输出必然也开着（两者都要求非PP、非PD producer）
        need_placeholder = is_deferred_out or self.spec_decode_local
        # Drafts occupy trailing slots only on an engine that verifies them; a
        # drafting-only engine's tokens are all real.
        # 【译】只有在会验证草稿的引擎上，草稿才占用末尾的槽位；只出草稿（不验证）的引擎上，token全是真实的。
        # 三个宽度（k=mtp_k）：
        #   num_placeholder_width：算真实长度时要从num_tokens里减掉的草稿占位数（k；不在本地验证草稿时为0）
        #   num_placeholder：第2段每步要追加的占位数 = k个草稿位 + （延迟输出时）1个"在途token"位；
        #     例：不开MTP、延迟输出 → 1；k=3、延迟输出 → 4（见上面时间线②）
        #   覆盖时用的 required_placeholders = num_placeholder + offset，验证步的offset是k（见下面）
        num_placeholder_width = self.mtp_k if self.spec_decode_local else 0
        num_placeholder = self.mtp_k
        if is_deferred_out:
            num_placeholder += 1

        # ---- 第1段：遍历running，按req_id取这条序列的结果 ----
        # 为什么遍历self.running而不是seqs：延迟输出时fwd_output是上一步batch的结果，和本步的seqs不是同一批；
        #   running里是所有活着的序列，按req_id对号入座。已结束 / 已被抢占（不在running里）的序列迟到的token自然被丢弃
        for seq in self.running:
            # Cancellation does not require sampled output. Middle prefill
            # chunks and the first deferred step may have none; waiting for
            # one keeps an aborted request alive without advancing its tokens.
            # 【译】取消不需要等采样输出。中间的prefill chunk、以及延迟输出的第一步都可能没有输出；
            #   等一个输出会让已取消的请求一直活着，而它的token却不再前进。
            # 解释：ABORTED由abort_request设置（客户端断开，见engine_utility.py的_handle_abort_request）；
            #   传给_mark_finished的长度去掉了末尾占位
            if seq.status == SequenceStatus.ABORTED:
                self._mark_finished(
                    seq, "aborted", seq.num_tokens - seq.num_placeholder_tokens
                )
                finished_seqs.append(seq)
                continue
            # Update the running status
            # 【译】更新运行状态
            # idx：这条序列在fwd_output里的行号；None = 结果里没有它（例：延迟输出下它刚做完prefill，结果下一步才到）
            idx = fwd_output.get_idx(seq.id)
            if idx is None:
                continue
            # An immediate preempt/resume can retain the same request ID in
            # the runner's deferred output. That result belongs to execution
            # before this prefill and must not advance the replayed request.
            # 【译】一次"抢占后立即恢复"可能让runner的延迟输出里还留着同一个请求ID。那份结果属于这次prefill之前的执行，
            #   不能用来推进这个正在重放的请求。
            # 解释：延迟输出下，本步才做prefill的序列不可能已经拿到属于这次prefill的结果（要下一步才到），所以一律跳过
            if is_deferred_out and seq.id in prefill_ids:
                continue
            # Partial prefill: KV written but prefill not complete — discard
            # the sampled token. Prefix hashes are also deferred since
            # num_tokens < num_prompt_tokens until the prompt finishes.
            #
            # Under schedule-time advancement seq.is_partial_prefill reflects a
            # possibly-later schedule() and cannot gate THIS batch's output;
            # use the frozen middle-chunk set instead.
            # 【译】部分prefill：KV已写入但prefill尚未完成——丢弃采样出的token。前缀hash同样推迟，
            #   因为在prompt算完之前 num_tokens < num_prompt_tokens。
            #
            #   在调度时推进进度的模式下（PP），seq.is_partial_prefill 反映的可能是更晚的某次 schedule()，
            #   不能用来决定"这个batch"的输出；改用冻结下来的中间chunk集合。
            # 解释：英文里的num_tokens应理解为num_cached_tokens（num_tokens本身从不小于num_prompt_tokens）
            if self.advance_on_schedule:
                if seq.id in pp_middle_chunk_ids:
                    continue
            elif seq.is_partial_prefill:
                continue
            # With deferred output, a token visible after a normal chunked-
            # prefill step belongs to the previous partial chunk and must be
            # dropped. An offload handoff is different: parking removes the
            # request from at least one intervening model-runner batch, which
            # already discards its deferred partial output. The first output
            # after the request resumes is fresh and must be kept.
            # 【译】在延迟输出下，普通chunked prefill步之后看到的token属于上一个部分chunk，必须丢掉。offload交接不一样：
            #   停放（park）会让这个请求至少缺席一个中间的model-runner batch，那时它延迟的部分输出已经被丢弃；
            #   请求恢复后的第一个输出是新的，必须保留。
            # 观察：按代码推断，这个分支在现有调度下不会再命中——进入prev_partial_ids的序列都是本步batch里的PREFILL序列，
            #   也就在prefill_ids里，延迟输出时已被上面"is_deferred_out and seq.id in prefill_ids"跳过
            if is_deferred_out and seq.id in prev_partial_ids:
                continue
            # Register prefix-cache hashes for blocks the prefill step just
            # finalized. Deferred from BlockManager.allocate() so a hash is
            # only published after the block's KV has actually been computed
            # by the forward — keeps the block manager correct under chunked
            # prefill where one block may span multiple steps. Must run before
            # any seq state update so num_cached_tokens and block_table still
            # reflect the pre-step view.
            #
            # Gate is `not prefix_hashes_published`, not `seq.type ==
            # PREFILL`: ModelRunner runs in deferred-output mode by default
            # (tokenIDProcessor.is_deferred_out), so the prefill step's
            # postprocess sees idx=None and skips this seq (above). By the
            # time the prefill output surfaces, the next step's schedule has
            # already flipped seq.type to DECODE — the old PREFILL gate never
            # fires and the content index stays empty for prompt blocks (HBM
            # prefix cache silently dead). The flag gate fires once per seq
            # at the first postprocess with idx.
            #
            # `num_new` subtracts `num_placeholder` when deferred-output is
            # active: those slots are filled with the real prefill output
            # later in this loop, so they're not part of the prompt hash
            # chain — leaving them in would mint a stale partial-block hash.
            # 【译】为prefill这一步刚写完的block登记前缀缓存hash。这一步从 BlockManager.allocate() 推迟到这里，保证只有在forward
            #   真正算出block的KV之后才发布hash——在chunked prefill下一个block可能跨好几步，这样block manager才正确。
            #   必须在更新序列的任何状态之前执行，这样 num_cached_tokens 和 block_table 仍然反映这一步之前的视图。
            #
            #   判断条件是 `not prefix_hashes_published`，而不是 `seq.type == PREFILL`：ModelRunner默认运行在延迟输出模式
            #   （tokenIDProcessor.is_deferred_out），所以prefill那一步的postprocess看到 idx=None 就跳过了这条序列（见上面）。
            #   等prefill的输出出现时，下一步的调度已经把 seq.type 改成了DECODE——旧的PREFILL判断永远不会触发，
            #   prompt block的内容索引一直为空（HBM前缀缓存悄无声息地失效）。用这个标志，每条序列在第一次拿到idx的
            #   postprocess里触发一次。
            #
            #   延迟输出开启时，num_new 要减去 num_placeholder：这些槽位稍后在这个循环里才被真实的prefill输出填上，
            #   不属于prompt的hash链——留着它们会产生一个过时的、不完整block的hash。
            # 解释：传了batch时（EngineCore / PP），prompt的hash已经在第0段按chunk登记过，这里只把标志置True；
            #   只有batch=None（RapidServe的DecodeEngineCore）才在这里补登记
            if not seq.prefix_hashes_published:
                if batch is None:
                    _num_new = seq.num_tokens - seq.num_cached_tokens
                    if need_placeholder:
                        _num_new -= num_placeholder
                    self.block_manager.hash_blocks(seq, max(0, _num_new))
                seq.prefix_hashes_published = True
            # 这条序列本步得到的token元组；num_new_token = 它的个数（投机验证步 = num_bonus + 1）
            token_ids = prev_token_ids[idx]
            num_new_token = len(token_ids)
            token_logprob = None
            if token_logprobs is not None and seq.return_logprobs:
                token_logprob = token_logprobs.get(seq.id)

            # In-place overwrite only when placeholders already exist.
            # 【译】只有在已经有占位的情况下才原地覆盖。
            # 解释：占位是上一次第2段追加的；没有占位（不延迟、不验证草稿，例如PP）时走下面的else，和nano-vllm一样append
            if is_deferred_out or (
                self.spec_decode_local and seq.num_placeholder_tokens > 0
            ):
                # int() casts strip the np.int32 wrapper coming out of
                # fwd_output's np.ndarray indexing. Without these, the values
                # propagate into seq.num_rejected / seq.num_bonus_tokens, then
                # into seq.num_tokens via `preempt()`'s `-= mtp_k + num_rejected`,
                # contaminating downstream logs and arithmetic with np.int32.
                # 【译】int() 转换去掉了从 fwd_output 的 np.ndarray 索引出来的 np.int32 包装。没有它们，这些值会传播进
                #   seq.num_rejected / seq.num_bonus_tokens，再经由 preempt() 的 `-= mtp_k + num_rejected` 进入 seq.num_tokens，
                #   用np.int32污染下游的日志和算术。
                # 解释：英文里提到的preempt()写法已经过时——现在preempt按num_placeholder_tokens删占位（见preempt里的说明）
                # num_rejected：被拒绝的草稿数；num_bonus：被接受的草稿数（这一行token数 = num_bonus + 1）
                # offset决定要覆盖的那段占位有多长（required_placeholders = num_placeholder + offset）：
                #   num_new_token + num_rejected == 1 → prefill步的输出（1个token、没有验证），或不开MTP的decode：
                #     末尾的占位串长num_placeholder，offset=0
                #   否则是验证步（num_new_token + num_rejected = k+1）：末尾的占位串长2k+1（=num_placeholder+k），offset=k
                #   数字见函数开头时间线②的ⓑ、ⓒ
                num_rejected = int(fwd_output.num_rejected[idx])
                num_bonus = int(fwd_output.num_bonus[idx])
                offset = 0 if (num_new_token + num_rejected) == 1 else self.mtp_k
                # Align stats with vLLM: only count steps that actually ran
                # speculation (drafts proposed and validated). Skip the
                # prefill-only step where no draft tokens were scored against
                # the target — vLLM gates this via
                # `if scheduled_spec_token_ids and generated_token_ids`.
                # 【译】统计口径与vLLM对齐：只统计真正跑了投机（提出并验证了草稿）的步。跳过只有prefill的那一步——那一步没有草稿
                #   拿去和目标模型比对；vLLM用 `if scheduled_spec_token_ids and generated_token_ids` 做这个判断。
                if (
                    self.engine_stats.spec_enabled
                    and num_new_token > 0
                    and (num_new_token + num_rejected) > 1
                ):
                    self.engine_stats.update_spec(num_new_token)
                # 存到序列上：下一次调度时ScheduledBatch用seq.num_rejected算窗口起点、交给runner；第2段据它少追加占位
                seq.num_rejected = num_rejected
                seq.num_bonus_tokens = num_bonus
                # DSpark Phase 2: stash this step's scheduler-chosen ell on the
                # seq so the NEXT decode schedule can size its verification to
                # ell+1. Keyed by req_id (order-safe). Missing -> leave default
                # (mtp_k), so a request never gets under-verified.
                # 【译】DSpark第2阶段：把本步调度器选定的ell存到序列上，这样"下一次"decode调度可以把它的验证长度定为ell+1。
                #   按req_id取（与顺序无关）。没有值 → 保持默认（mtp_k），保证请求永远不会被少验证。
                # 【分支】DSpark投机解码（第7批），ell的含义见sequence.py的dspark_next_ell
                if fwd_output.dspark_ell is not None:
                    ell_r = fwd_output.dspark_ell.get(seq.id)
                    if ell_r is not None:
                        seq.dspark_next_ell = int(ell_r)
                required_placeholders = num_placeholder + offset
                # 末尾应有的占位数不够时（output_tokens比它短）打警告并用eos补齐，保证下面按负下标覆盖时只落在占位上
                missing_placeholders = required_placeholders - len(seq.output_tokens)
                if missing_placeholders > 0:
                    logger.warning(
                        "Repairing missing deferred-output placeholders for seq %s: "
                        "required=%d, available=%d",
                        seq.id,
                        required_placeholders,
                        len(seq.output_tokens),
                    )
                    for _ in range(missing_placeholders):
                        seq.append_token(self.eos_token_id)
                # 从占位串的开头起依次写入真实token（位置 = 长度 - required_placeholders + i），token_ids与output_tokens同步改
                for i, el in enumerate(token_ids):
                    seq.token_ids[-required_placeholders + i] = el
                    seq.output_tokens[-required_placeholders + i] = el
                # What the write did NOT consume of the window it was given.
                # Those slots are still `eos_token_id`, so they are still
                # placeholders, and on a verify step there are always some: the
                # window is `mtp_k` wider than the run appended for it (the
                # `offset` above) and the step hands back `mtp_k - num_rejected
                # + 1` tokens for it. Recorded rather than left to `preempt` to
                # re-derive -- `num_placeholder_tokens` is the width it reads,
                # and until this line it named only the run appended below.
                # Measured steady state, deferred output and drafts every step:
                # the trailing `eos` run is `2 * mtp_k + 1` while the recorded
                # width was `mtp_k + 1 - num_rejected`, so a preemption left
                # `mtp_k + num_rejected` of them in the recomputed context.
                # 【译】这次写入没有用掉的那部分窗口。这些槽位仍是 eos_token_id，所以仍然是占位；在验证步上总会剩一些：
                #   窗口比为它追加的那段宽 mtp_k（上面的 offset），而这一步为它交回 mtp_k - num_rejected + 1 个token。
                #   在这里记录下来，而不是留给 preempt 去重新推导——num_placeholder_tokens 就是它读取的宽度，而在这一行之前，
                #   它只描述了下面追加的那一段。实测的稳态（延迟输出、每步都有草稿）：末尾的 eos 串长 2*mtp_k+1，而之前记录的
                #   宽度是 mtp_k+1-num_rejected，所以一次抢占会在重算的上下文里留下 mtp_k+num_rejected 个eos。
                # 解释：验证步剩下 (2k+1) - (k+1-num_rejected) = k + num_rejected 个；第2段再追加 k+1-num_rejected 个，合计回到2k+1
                #   （时间线②的ⓒ：剩4个，追加3个）；不开MTP时剩0个
                seq.num_placeholder_tokens = max(
                    0, required_placeholders - len(token_ids)
                )
                # logprobs：每个占位追加时配了一个0.0，这里把末尾那一项改成真实值（不开MTP时末尾正好是刚覆盖的位置）
                if seq.return_logprobs and token_logprob is not None:
                    if seq.logprobs:
                        seq.logprobs[-1] = token_logprob
                    else:
                        seq.logprobs.append(token_logprob)
            else:
                num_rejected = 0
                num_bonus = 0
                # 没有占位：和nano-vllm一样，把本步的token直接append（num_tokens随之+1）
                for token_id in token_ids:
                    seq.append_token(token_id)
                if seq.return_logprobs and token_logprob is not None:
                    seq.logprobs.append(token_logprob)
            # new_tokens：本步要发给客户端的token，下面可能被截断
            new_tokens = token_ids

            # 【分支】PD分离的consumer（第9批）：第一次decode之前，_schedule_first_decode_after_remote_kv已把producer采样的T0
            #   append进序列并记在_injected_t0；它不在本步的fwd_output里，所以在这里补到要发出去的token前面，只补一次
            injected_t0 = getattr(seq, "_injected_t0", None)
            if injected_t0 is not None:
                new_tokens = [injected_t0] + list(new_tokens)
                seq._injected_t0 = None

            # `draft_token_ids` is None whenever the runner skipped propose(),
            # which a P/D producer does on every step: it disables deferred
            # output so the consumer consumes T0 exactly once (KDA state would
            # otherwise advance twice). The seq is handed off rather than
            # decoded here, so it keeps its empty spec_token_ids and the
            # consumer proposes for itself.
            # 【译】只要runner跳过了 propose()，draft_token_ids 就是None——P/D producer每一步都这样：它关掉了延迟输出，
            #   让consumer恰好只消费一次T0（否则KDA状态会前进两次）。这条序列会被交接出去，而不是在这里decode，
            #   所以它保持空的 spec_token_ids，由consumer自己提草稿。
            # 解释：spec_token_ids是下一步要验证的草稿，ScheduledBatch把它放进scheduled_spec_decode_tokens
            #   （延迟输出时，跨步的序列由runner直接用GPU上的草稿，见model_runner.py的prepare_input_ids）
            if self.mtp_k > 0 and draft_token_ids is not None:
                seq.spec_token_ids = draft_token_ids[idx]

            if seq.num_completion_tokens <= 3 and seq.kv_transfer_params:
                logger.debug(
                    "[PD-DECODE] seq %s: comp_tokens=%d, "
                    "new_token=%s, num_tokens=%d, blocks=%d",
                    seq.id,
                    seq.num_completion_tokens,
                    token_ids,
                    seq.num_tokens,
                    len(seq.block_table),
                )
            # 真实长度 = 数组长度 - 草稿占位(k) - 被拒绝的草稿数；对照时间线②：ⓑ 8-3-0=5，ⓒ 12-3-1=8
            #   不开MTP时覆盖之后已没有占位，num_tokens就是seq.num_tokens
            num_tokens = seq.num_tokens - num_placeholder_width - num_rejected
            # leave_reason：本步的结束原因，None表示继续；取值见_mark_finished上方的表
            leave_reason = None
            # MTP edge case: `rejection_sampler` does NOT inspect EOS — it
            # only compares draft vs target_argmax for acceptance. So when
            # the verified token is EOS the kernel still emits 1+ accepted
            # bonus tokens after EOS (often BOS, since the model naturally
            # starts a new sentence). Without truncating, those post-EOS
            # tokens leak into the detokenized output (e.g. "...6.<EOS><BOS>").
            # Empirically confirmed via DIAG: `token_ids=[EOS=1, BOS=0]`,
            # `eos_idx=0`, `num_new=2`, `num_rejected=0` for V4-Pro MTP-1.
            # Track the earliest stop position so `num_tokens` can drop the
            # spurious tail below.
            # 【译】MTP的边界情况：rejection_sampler 不检查EOS——它只比较草稿和目标模型的argmax来决定是否接受。所以当被验证的
            #   token是EOS时，kernel仍会在EOS之后吐出1个以上被接受的bonus token（常常是BOS，因为模型很自然地开始一个新句子）。
            #   不截断的话，这些EOS之后的token会漏进反分词后的输出（例如"...6.<EOS><BOS>"）。已用DIAG实测确认：
            #   V4-Pro MTP-1 下 token_ids=[EOS=1, BOS=0]、eos_idx=0、num_new=2、num_rejected=0。
            #   记录最早的停止位置，好让下面的 num_tokens 去掉这段多余的尾巴。
            # stop_at_idx：本步token（token_ids）里的下标，输出保留到它为止（含）；None表示本步没有停止
            stop_at_idx: int | None = None
            # Check if sequence ends with any stop sequence
            # 【译】检查序列是否以任何一个stop序列结尾
            # ① stop序列（stop字符串分词后的token序列）：对本步每个新token的位置，看"以它结尾"的那一段是否等于stop序列；
            #   offset = num_tokens - i，i=0是最后一个新token，所以从后往前找，找到就break
            #   例：stop序列=[271, 25]，本步新增(…, 271, 25)，num_tokens=20 → i=0时token_ids[18:20]==[271,25] → stop_at_idx=最后一个
            #   Python的for…else：循环没被break（没有任何stop序列命中，包括列表为空）时才执行else，所以eos只在stop序列都没命中时才检查
            # 观察：一步多token（MTP）时，选中的是"列表里第一个命中的stop序列在本步里最靠后的结束位置"，而且一旦命中就不再看eos；
            #   若同一步里更早的位置已出现eos或另一个stop序列，它们会留在输出里。不开MTP时每步只有1个新token，没有这个差别
            for stop_seq in seq.stop_token_sequences:
                stop_len = len(stop_seq)
                if num_tokens >= stop_len:
                    is_stop = False
                    for i in range(num_new_token):
                        offset = num_tokens - i
                        if seq.token_ids[offset - stop_len : offset] == stop_seq:
                            is_stop = True
                            # `i` counts back from the last sampled token
                            # (i=0 = last). Truncate to include this stop
                            # sequence (drop everything after it).
                            # 【译】i 从最后一个采样token往回数（i=0即最后一个）。截断到包含这个stop序列为止（丢掉它之后的一切）。
                            stop_at_idx = num_new_token - 1 - i
                            break
                    if is_stop:
                        leave_reason = "stop_sequence"
                        break
            else:
                # Check the last token in the list for EOS
                # 【译】检查列表里最后一个token是否为EOS
                # 解释：实际检查的是本步全部新token，取第一个eos的位置。② eos：只看本步拿到的真实token，不看token_ids末尾
                #   （那里的占位值也是eos）；③ stop_token_ids：generation_config里除eos以外的结束token（LLMEngine.__init__已把eos
                #   从中去掉），原因记为"stop_<id>"（如"stop_151645"）。ignore_eos=True时这两项都不查，stop序列和max_tokens照常生效
                if token_ids and not seq.ignore_eos and self.eos_token_id in token_ids:
                    leave_reason = "eos"
                    stop_at_idx = token_ids.index(self.eos_token_id)
                elif not seq.ignore_eos and any(
                    t in self.stop_token_ids for t in token_ids
                ):
                    stop_at_idx = next(
                        i for i, t in enumerate(token_ids) if t in self.stop_token_ids
                    )
                    leave_reason = f"stop_{token_ids[stop_at_idx]}"

            # ``num_tokens`` is the real post-verification length. One MTP
            # forward can accept multiple tokens, so the final batch can cross
            # max_tokens even though the request was below the cap when it was
            # scheduled. Select the earlier boundary between a natural stop
            # above and the output cap, then reuse the common truncation path
            # for both internal and client-visible tokens.
            # 【译】num_tokens 是验证之后的真实长度。一次MTP forward可以接受多个token，所以即使请求在被调度时还没到上限，
            #   最后这一批也可能越过 max_tokens。在上面的自然停止和输出上限之间选更早的那个边界，然后让内部token和
            #   客户端可见的token都走同一条截断路径。
            # 例：max_tokens=10，本步之前已生成8个，本步接受了(61,62,70) → completion_tokens=11，overflow=1
            #   → max_stop_at_idx = 3-1-1 = 1 → 只保留61、62，正好10个；本步没有自然停止时原因记为"max_tokens"
            #   自然停止恰好落在上限上（stop_at_idx == max_stop_at_idx）时保留自然原因，不改成max_tokens
            completion_tokens = num_tokens - seq.num_prompt_tokens
            if completion_tokens >= seq.max_tokens:
                overflow = completion_tokens - seq.max_tokens
                # ``stop_at_idx`` indexes this step's model output, excluding
                # an injected P/D T0.  -1 therefore keeps T0 but no model
                # output; when the cap retains no tokens, -2 is the boundary
                # before T0.
                # 【译】stop_at_idx 是本步模型输出里的下标，不含注入的P/D T0。所以 -1 表示保留T0、但不保留任何模型输出；
                #   当上限一个token都不让保留时，-2 是T0之前的边界。
                min_stop_at_idx = -1 - (1 if injected_t0 is not None else 0)
                max_stop_at_idx = max(min_stop_at_idx, num_new_token - 1 - overflow)
                if stop_at_idx is None or max_stop_at_idx < stop_at_idx:
                    stop_at_idx = max_stop_at_idx
                    leave_reason = "max_tokens"

            # Drop accepted-draft tokens past the stop position (MTP only —
            # for non-spec the sampler emits exactly 1 token so this is a
            # no-op).
            # 【译】丢掉停止位置之后被接受的草稿token（只有MTP会这样——非投机时采样器只吐1个token，这里什么也不做）。
            # 解释：例外是max_tokens<=0这类边界：stop_at_idx可以是-1，非投机时也会把本步的token去掉（但仍发一个结束事件，见下面）
            if stop_at_idx is not None and stop_at_idx < num_new_token - 1:
                num_tokens -= (num_new_token - 1) - stop_at_idx
                # The same truncation MUST apply to the EMITTED tokens, not just
                # the internal seq length. The client-visible text is built from
                # RequestOutput.output_tokens (an accumulation of `new_tokens`) by
                # generate_async / the streaming callback — NOT from
                # completion_token_ids (which the `seq.num_tokens` write above
                # governs). Without trimming `new_tokens` here, the post-stop
                # tokens the rejection sampler emits past EOS (it does not inspect
                # EOS) leak into the response: strict-match still finds the answer,
                # but flexible-extract's last-number picks up the leaked trailing
                # digit. `injected_t0` (if present) prepends one slot not counted
                # in stop_at_idx / num_new_token, so offset the cut by it.
                # 【译】同样的截断必须作用于"发出去的"token，而不仅是内部的序列长度。客户端看到的文本是由 generate_async /
                #   流式回调从 RequestOutput.output_tokens（new_tokens 的累积）拼出来的——不是从 completion_token_ids
                #   （它受上面写 seq.num_tokens 的那一步控制）。这里不裁剪 new_tokens 的话，拒绝采样器在EOS之后吐出的停止后token
                #   （它不检查EOS）就会漏进响应：strict-match仍然能找到答案，但flexible-extract取最后一个数字时会拿到漏出来的
                #   尾部数字。injected_t0（如果有）在前面多占一个槽位，不计入 stop_at_idx / num_new_token，所以截断位置要按它偏移。
                # 解释：strict-match / flexible-extract是lm_eval里GSM8K的两种答案抽取方式（CI精度门槛，见/ci-pr-guide）；
                #   seq.num_tokens实际是在下面的_mark_finished里写的
                keep = stop_at_idx + 1 + (1 if injected_t0 is not None else 0)
                new_tokens = new_tokens[:keep]
                if seq.return_logprobs:
                    # LLMEngine.postprocess returns the complete logprobs
                    # array rather than slicing it through num_tokens, so keep
                    # it aligned explicitly with the cropped completion.
                    # 【译】LLMEngine.postprocess 返回完整的logprobs数组，而不是按 num_tokens 切片，所以这里要显式地让它
                    #   与裁剪后的生成部分对齐。
                    del seq.logprobs[num_tokens - seq.num_prompt_tokens :]

            # Record TTFT from the finalized retained length, after rejected
            # speculative tokens and cap/stop overflow have been removed. A
            # terminal response with no completion tokens must keep TTFT zero.
            # 【译】在去掉被拒绝的投机token和超出上限/停止位置的部分之后，用最终保留的长度记录TTFT。
            #   一个没有任何生成token就结束的响应，TTFT必须保持为0。
            # 解释：num_finalized_tokens = 主机侧已确定的长度（不含占位，见sequence.py）；first_token_time只在第一次出现真实生成token时
            #   写一次（TTFT = first_token_time - arrive_time）
            seq.num_finalized_tokens = num_tokens
            if num_tokens - seq.num_prompt_tokens >= 1 and seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            # Counted here, not from `len(token_ids)` above: `new_tokens` is
            # what reaches RequestOutput, and it differs from the forward's
            # raw output in both directions — the truncation just above drops
            # accepted drafts past EOS/stop, and `injected_t0` prepends the
            # token the prefill process sampled. Counting the raw output made
            # the status line disagree with what the client received and with
            # `total_generation_tokens` below, which this same call derives
            # from the post-truncation length.
            # 【译】在这里计数，而不是用上面的 len(token_ids)：new_tokens 才是到达 RequestOutput 的东西，它和forward的原始输出
            #   在两个方向上都不同——上面的截断丢掉了EOS/stop之后被接受的草稿，injected_t0 又在前面加上了prefill进程采样的token。
            #   用原始输出计数会让状态行和客户端实际收到的不一致，也和下面的 total_generation_tokens 不一致
            #   （它在同一次调用里由截断后的长度算出）。
            num_new_generation_tokens += len(new_tokens)

            # Hash generated blocks. Deferred output: all tokens forwarded;
            # undeferred: last token not yet forwarded, so exclude it.
            # 【译】登记生成部分的block的hash。延迟输出：所有token都已经forward过；不延迟：最后一个token还没forward，所以排除它。
            # 解释：只有KV已写好的满block才能登记（hash_decode_blocks的committed_kv_len是一条硬线）。延迟输出时最后一个真实token
            #   正是GPU上在途那一步的输入，它的KV由那一步写；不延迟时刚append的token要下一步才算KV。这样下一轮对话
            #   （上轮prompt + 上轮回答）也能命中生成部分的block（sequence.py字段演化总表第4步）。
            #   next_forward_tokens：下一次forward会带几个token，给状态checkpoint用（_checkpoint_room，第4批）；本步就结束时为0
            self.block_manager.hash_decode_blocks(
                seq,
                num_tokens - (0 if is_deferred_out else 1),
                next_forward_tokens=self._checkpoint_room(
                    seq, leave_reason is not None
                ),
            )

            if leave_reason is not None:
                # Publish transfer metadata before constructing the terminal
                # output. Aborts without sampled output use the same helper.
                # 【译】在构造终止输出之前发布传输元数据。没有采样输出就被取消的请求也走同一个辅助函数。
                # 解释：_mark_finished会调用connector的request_finished，PD producer在那里写好kv_transfer_params_output
                #   （如mooncake_connector.py），下面的RequestOutput才能带上它
                self._mark_finished(seq, leave_reason, num_tokens)

            # Prepare stream output
            # A terminal event is required even when truncation leaves no
            # tokens (for example max_tokens <= 0). Async consumers wait for
            # this finished RequestOutput and would otherwise block forever.
            # 【译】准备流式输出。
            #   即使截断后一个token都不剩（例如 max_tokens <= 0），也必须发一个终止事件。异步消费者在等这个已结束的
            #   RequestOutput，否则会永远阻塞下去。
            # 解释：RequestOutput（request.py）：output_tokens=本步新增的token、finished、finish_reason=leave_reason、
            #   PD producer的kv_transfer_params_output、num_cached_tokens（取prefix_cache_hit_tokens，最终出现在usage里）
            if stream_output_queue is not None and (
                new_tokens or leave_reason is not None
            ):
                # new_tokens可能是runner给的tuple，也可能是上面拼接/切片得到的list；统一拷成一个新list
                output_tokens_list = (
                    list(new_tokens)
                    if isinstance(new_tokens, tuple)
                    else new_tokens.copy()
                )
                request_output = RequestOutput(
                    request_id=seq.id,
                    output_tokens=output_tokens_list,
                    finished=(leave_reason is not None),
                    finish_reason=leave_reason,
                    kv_transfer_params_output=getattr(
                        seq, "kv_transfer_params_output", None
                    ),
                    num_cached_tokens=getattr(seq, "prefix_cache_hit_tokens", 0),
                )

                if request_output.kv_transfer_params_output is not None:
                    logger.debug("KV transfer output present in stream output.")

                stream_outputs.append((seq.id, request_output))
                logger.debug(
                    f"Scheduler: Created stream output for seq_id={seq.id}, "
                    f"tokens={new_tokens}, finished={leave_reason is not None}"
                )

            if leave_reason is not None:
                # logger.info(
                #     f"Sequence {seq.id} finished with reason: {leave_reason}, {seq.token_ids[-8:]=}"
                # )
                finished_seqs.append(seq)

        # 一步只放一个列表：EngineCore把它作为一条("STREAM", 列表)消息发给CoreManager，之后才发finished_seqs
        if stream_output_queue is not None and stream_outputs:
            stream_output_queue.put_nowait(stream_outputs)

        # ---- 第2段：为下一步追加占位 ----
        # 只给本步batch里的序列（seqs）追加：它们此刻都有一步在GPU上在途（延迟输出），或下一步要验证k个草稿（投机）。
        #   已结束（FINISHED）的不追加；还在prompt中途的（is_partial_prefill）不追加，它的下一步仍是prefill chunk
        if need_placeholder:
            # placeholder for the each decode step
            # 【译】为每个decode步放置占位
            for seq in seqs:
                if seq.status == SequenceStatus.RUNNING and not seq.is_partial_prefill:
                    # Clamped because `preempt()` strips exactly this many:
                    # `range()` on a negative width appends nothing, so storing
                    # one unclamped would claim placeholders that are not there.
                    # 【译】做下限截断，因为 preempt() 恰好删掉这么多个：range() 遇到负数宽度什么也不追加，
                    #   所以不截断就存进去，会声称有并不存在的占位。
                    # 解释：num = num_placeholder - seq.num_rejected（刚拿到的那次验证里被拒绝的草稿数）。被拒绝的位置在上面覆盖时已经留成了占位
                    #   （剩下k+num_rejected个），这里少追加num_rejected个，末尾占位总数回到2k+1（时间线②的ⓒ）
                    num = max(0, num_placeholder - seq.num_rejected)
                    for _ in range(num):
                        seq.append_token(self.eos_token_id)
                        if seq.return_logprobs:
                            seq.logprobs.append(0.0)
                    # Added to what the overwrite above left unconsumed, not
                    # replacing it: both runs are trailing `eos_token_id` and
                    # both are what `preempt` has to strip. `=` also undercounted
                    # a sequence that went two steps without a row -- nothing
                    # consumed the first run, and this one appends beyond it.
                    # 【译】加到上面覆盖后没用掉的那部分上，而不是替换它：两段都是末尾的 eos_token_id，都是 preempt 要删的。
                    #   用 `=` 还会少算一条连续两步都没有结果行的序列——第一段没被消费，这一段又追加在它后面。
                    seq.num_placeholder_tokens += num
        # ---- 第3段：释放已结束序列的KV block ----
        # 对比nano-vllm：nano-vllm判停后立刻deallocate + running.remove；ATOM先收集finished_seqs，循环之后统一释放，
        #   中间多了connector的延迟释放（【分支】PD分离 / KV offload，第9批）
        for seq in finished_seqs:
            logger.debug("Freeing blocks for finished seq %s", seq.id)
            # 被取消时还在prompt中途：修正_partial_prefill_count
            if seq.is_partial_prefill:
                seq.is_partial_prefill = False
                self._partial_prefill_count -= 1
            if self.kv_connector is not None:
                # `request_finished` already ran once, above, while the reason
                # was being settled -- it must not run again now that the claim
                # it takes is a side effect.
                # 【译】request_finished 已经在上面确定结束原因时运行过一次（_mark_finished里）——它会认领block，这是一个副作用，
                #   所以现在不能再运行一次。
                # 【分支】KV connector（第9批）：should_defer_free为True表示还有人在读这些block（PD producer的RDMA发送还没完成，
                #   或KV offload还在把它们存到CPU/远端），不能马上放回空闲池
                if self._connector_should_defer_free(seq):
                    protected = self._connector_protected_block_ids(seq)
                    state_safe = (
                        not seq.has_per_req_cache
                        or self._connector_can_partially_deallocate_state(seq)
                    )
                    if protected is not None and state_safe:
                        # Early block release: only the save's exact source
                        # blocks stay pinned; everything else in this
                        # finished request's block_table -- decode blocks, an
                        # unaligned prompt tail, already-saved ranges -- frees
                        # now instead of waiting behind `deferred_free_blocks`.
                        # 【译】提前释放block：只有这次保存真正要读的源block保持钉住；这个已结束请求的block_table里其余的一切——
                        #   decode阶段的block、没对齐的prompt尾部、已经保存完的区段——现在就释放，而不必在 deferred_free_blocks 后面等。
                        # 解释：条件是connector能给出精确的受保护block集合（protected不为None），且每请求状态（若有）允许部分释放；
                        #   受保护的block以"租约"形式保留，保存完成后由_drain_source_safe_releases释放
                        before = len(seq.block_table)
                        # `state_safe` holds on this branch; restate it so the
                        # block manager's own fail-closed check stays armed.
                        # 【译】state_safe 在这个分支上必然成立；这里再显式传一遍，让block manager自己的"失败即关闭"（fail-closed）检查保持生效。
                        self.block_manager.deallocate_partial(
                            seq, protected, per_request_state_safe=True
                        )
                        activate = getattr(
                            self.kv_connector, "activate_block_leases", None
                        )
                        if callable(activate):
                            activate(seq, protected)
                        released = before - len(protected)
                        record = getattr(
                            self.kv_connector, "record_early_release", None
                        )
                        if callable(record):
                            record(released)
                        logger.debug(
                            "Early-released %d/%d block(s) for seq %s; %d "
                            "still leased to an in-flight save.",
                            released,
                            before,
                            seq.id,
                            len(protected),
                        )
                    else:
                        logger.debug(
                            "Deferring block free for seq %s until every KV "
                            "owner releases it.",
                            seq.id,
                        )
                        # Stamp when the save was deferred so the reconciler
                        # can reclaim it if the completion report never
                        # arrives.
                        # 【译】记下延迟保存开始的时间，这样如果完成报告一直不来，reconciler可以把它回收。
                        # 解释：整条请求放进deferred_free_blocks，等_update_from_kv_xfer_finished收到发送/保存完成的报告再释放；
                        #   报告丢了则由_reconcile_stalled_deferred_saves超时回收
                        seq._deferred_save_at = time.monotonic()
                        self.deferred_free_blocks[seq.id] = seq
                else:
                    self.block_manager.deallocate(seq)
            else:
                # 主干（没有connector）：引用计数减1，减到0的block回到空闲池；已登记hash的block内容留在显存里，
                #   之后同前缀的请求仍可能命中（同nano-vllm）
                self.block_manager.deallocate(seq)
            self.running.remove(seq)

        # 吞吐统计：本步发给客户端的生成token数（状态行的generation throughput）
        self.engine_stats.update_throughput(
            num_generation_tokens=num_new_generation_tokens
        )
        return finished_seqs

    # 把一条序列标记为结束（不释放block，释放在postprocess第3段）：写leave_reason → 通知connector → num_tokens截到真实长度
    #   → FINISHED → 累加看板计数。调用方：只有postprocess（判停命中时，和序列已是ABORTED时）
    # reason的取值（seq.leave_reason → RequestOutput.finish_reason → 对外的finish_reason）：
    #   reason           │ postprocess里在哪判定                               │ OpenAI finish_reason（protocol.openai_stop_reason）
    #   ─────────────────┼────────────────────────────────────────────────────┼──────────────────────────────
    #   "stop_sequence"  │ token_ids末尾命中某个stop字符串的token序列            │ "stop"
    #   "eos"            │ 本步token里有eos_token_id（ignore_eos时不查）         │ "stop"
    #   "stop_<id>"      │ 本步token里有stop_token_ids中的某个，如"stop_151645"   │ "stop"
    #   "max_tokens"     │ 生成数达到max_tokens（一步多token时截在上限处）        │ "length"
    #   "aborted"        │ 序列已被abort_request标成ABORTED（running里的）         │ "stop"
    #   不经过这里的两种：准入时被拒绝的"unschedulable: <原因>"（_schedule直接写leave_reason，经take_rejected交给EngineCore），
    #   以及还在waiting里就被取消的"aborted"（_reject_aborted_waiting）
    # 对比nano-vllm：nano-vllm只有eos和max_tokens两种结束，也不记录原因
    def _mark_finished(self, seq, reason, num_tokens):
        # Set the reason before the connector decides whether to claim the
        # source blocks. This runs exactly once, including early cancellation
        # without sampled output and requests without a stream queue.
        # 【译】在connector决定是否认领源block之前先设置结束原因。这个函数恰好运行一次，包括没有采样输出的提前取消、
        #   以及没有流式队列的请求。
        seq.leave_reason = reason
        if self.kv_connector is not None:
            # 【分支】KV connector（第9批）：request_finished让connector决定是否认领这些block
            #   （PD producer：登记待发送、写好kv_transfer_params_output）
            callback = getattr(self.kv_connector, "request_finished", None)
            if callable(callback):
                callback(seq)
            if reason == "aborted":
                # No peer will report a send for a cancelled request.
                # 【译】被取消的请求不会有对端来报告发送完成。
                # 解释：所以这里直接释放发送认领，否则这些block会一直等一个不会来的报告
                self._connector_send_finished(seq.id)
        # 长度截到真实长度：去掉末尾占位、被拒绝的草稿、停止位置之后的token。只改计数不删数组
        #   （completion_token_ids按num_tokens切片；preempt删占位时也以它为界）
        seq.num_tokens = num_tokens
        seq.multimodal_data = None
        seq.status = SequenceStatus.FINISHED
        # 看板计数只在请求生命周期的边界更新（见__init__）
        self.total_finished_requests += 1
        self.total_prompt_tokens += int(seq.num_prompt_tokens)
        self.total_generation_tokens += max(
            0, int(num_tokens) - int(seq.num_prompt_tokens)
        )

    def compute_detailed_aggregates(
        self,
        scheduled_batch: ScheduledBatch,
        seqs: dict[int, Sequence],
    ) -> None:
        """Attach detailed attention aggregates to *scheduled_batch* in place.

        Only the quadratic terms genuinely needed for a downstream attention-FLOP
        estimate are computed here. The request counts and total query tokens are
        already emitted by the ``prefill[]``/``decode[]`` labels in
        :meth:`ModelRunner.run_model`, so this avoids duplicating them.

        The following batch-level sums are stored on the batch and appended to
        those labels by the runner:

            sqsq  — sum of N_Q^2      (per request)
            sqsk  — sum of N_Q*N_KV   (per request)
            sk    — sum of N_KV       (per request)

        where ``N_Q`` is the number of query tokens scheduled for a request and
        ``N_KV`` is its KV length (cached + new tokens for prefill, full
        sequence length for decode). Aggregating over every request in the
        batch gives the total for that single forward, which is exactly the
        quantity a per-iteration attention-FLOP estimate needs. For MTP/spec-decode a
        decode step schedules ``mtp_k + 1`` query tokens, so the scheduled
        token count is used as ``N_Q`` for both branches (rather than a
        hardcoded 1) to avoid undercounting.

        This is a no-op (leaves the fields ``None``) unless profiling is active
        and ``ATOM_ENABLE_DETAILED_ANNOTATION`` is set.
        """
        # 【译】把详细的注意力聚合量原地挂到 scheduled_batch 上。
        #
        #   这里只计算下游估算注意力FLOP真正需要的二次项。请求数和总query token数已经由 ModelRunner.run_model 里的
        #   prefill[] / decode[] 标签输出了，所以这里不重复计算。
        #
        #   下面这几个batch级的和存放在batch上，由runner追加到那些标签后面：
        #
        #       sqsq —— N_Q^2 之和（按请求）
        #       sqsk —— N_Q*N_KV 之和（按请求）
        #       sk   —— N_KV 之和（按请求）
        #
        #   其中 N_Q 是为一个请求调度的query token数，N_KV 是它的KV长度（prefill为已缓存+新token，decode为整条序列长度）。
        #   对batch中每个请求求和，就得到这一次forward的总量，正好是按迭代估算注意力FLOP所需的量。对MTP/投机解码，一个decode步
        #   调度 mtp_k + 1 个query token，所以两个分支都用调度的token数作为 N_Q（而不是写死1），避免少算。
        #
        #   除非profiling处于激活状态并且设置了 ATOM_ENABLE_DETAILED_ANNOTATION，否则这是空操作（字段保持None）。
        # 调用方：EngineCore._process_engine_step_inner（forward之前）；结果由ModelRunner._detailed_label_suffix拼进trace标签，
        #   只服务于profile，不影响调度（第10批 指标）
        # 为什么sqsk是主项：每层每个头，打分QK^T和加权求和PV各约 2·N_Q·N_KV·head_dim 次浮点运算
        #   例：一条prefill，新算1000个token、前面已缓存3000个 → N_Q=1000、N_KV=4000 → sqsk贡献4e6；一条decode（N_Q=1）只贡献N_KV
        if not self.profile_active or not self._detailed_annotation_enabled:
            return

        sqsq = 0  # sum N_Q^2
        sqsk = 0  # sum N_Q*N_KV
        sk = 0  # sum N_KV
        for seq, num_tokens in zip(seqs.values(), scheduled_batch.num_scheduled_tokens):
            # Cast to Python int: num_scheduled_tokens is np.int32, so nq*nq /
            # nq*nkv would overflow once a prefill/chunk exceeds ~46341 tokens
            # (e.g. np.int32(65536)**2 == 0), silently corrupting the estimate.
            # 【译】转成Python int：num_scheduled_tokens 是 np.int32，一旦某个prefill/chunk超过约46341个token，nq*nq / nq*nkv
            #   就会溢出（例如 np.int32(65536)**2 == 0），悄无声息地破坏估算结果。
            nq = int(num_tokens)  # query tokens scheduled this forward
            if seq.type == SequenceType.DECODE:
                nkv = int(seq.num_tokens)  # full sequence length
            else:
                # PREFILL: KV length = cached + new tokens.
                # 【译】PREFILL：KV长度 = 已缓存 + 新token。
                nkv = int(seq.num_cached_tokens) + nq
            sqsq += nq * nq
            sqsk += nq * nkv
            sk += nkv

        scheduled_batch.detailed_sqsq = sqsq
        scheduled_batch.detailed_sqsk = sqsk
        scheduled_batch.detailed_sk = sk

    # 【分支】KV connector的小工具（第9批）：读connector上的布尔能力标志（is_producer / is_offload等），没有connector时为False
    def _connector_flag(self, name: str) -> bool:
        return bool(getattr(self.kv_connector, name, False))

    # connector报告的req_id可能是int，也可能是str（不同connector的key类型不同），所以原样、str、int三种写法都试一遍；
    #   _pop_req_id同理，找到就从列表里删掉
    @staticmethod
    def _has_req_id(req_ids: list, seq_id) -> bool:
        candidates = (seq_id, str(seq_id))
        for candidate in candidates:
            if candidate in req_ids:
                return True
        try:
            int_id = int(seq_id)
        except (TypeError, ValueError):
            return False
        return int_id in req_ids

    @staticmethod
    def _pop_req_id(req_ids: list, seq_id) -> bool:
        candidates = (seq_id, str(seq_id))
        for candidate in candidates:
            if candidate in req_ids:
                req_ids.remove(candidate)
                return True
        try:
            int_id = int(seq_id)
        except (TypeError, ValueError):
            return False
        if int_id in req_ids:
            req_ids.remove(int_id)
            return True
        return False

    def _update_waiting_for_remote_kv(self, seq: Sequence) -> bool:
        """Check whether a remote KV transfer for *seq* has completed.

        The ``finished_recving_kv_req_ids`` list is populated by
        :meth:`_update_from_kv_xfer_finished` during the previous
        scheduling step.  When ready, the sequence transitions back
        from ``WAITING_FOR_REMOTE_KVS`` to ``WAITING``.
        """
        # 【译】检查 seq 的远端KV传输是否已完成。
        #
        #   finished_recving_kv_req_ids 列表由 _update_from_kv_xfer_finished 在上一个调度步里填充。
        #   就绪时，序列从 WAITING_FOR_REMOTE_KVS 转回 WAITING。
        # 【分支】PD分离的consumer / KV offload的加载（第9批）
        # 调用方：_resolve_waiting_remote_kv（_schedule的Phase 2准入时）。返回True=传输已完成；False=还在等
        # 解释：docstring说的状态切换由调用方完成，这个函数本身只弹出req_id，并把收到的prompt block登记进前缀缓存 / 发出KV事件
        if not self._pop_req_id(self.finished_recving_kv_req_ids, seq.id):
            return False

        logger.debug("KV transfer finished for seq %s, ready for scheduling.", seq.id)

        # Hash received prompt blocks into prefix cache so the next turn
        # transfers only the delta. Decode never runs prefill forward, so
        # this is the only place these blocks get hashed.
        # 【译】把收到的prompt block登记进前缀缓存，这样下一轮只需传增量。decode从不跑prefill的forward，
        #   所以这是这些block唯一被登记hash的地方。
        bm = self.block_manager
        prefix_caching = getattr(bm, "enable_prefix_caching", False)
        kv_events = bm.kv_events_enabled
        if not (prefix_caching or kv_events):
            return True

        # num_cached_tokens is a global-token count. Under DCP each block-table
        # entry spans one virtual hash block, not one rank-local physical block.
        # 【译】num_cached_tokens 是全局token计数。在DCP下，每个block_table条目跨一个虚拟hash block，而不是一个rank本地的物理block。
        #   （DCP见arg_utils.py的decode_context_parallel_size上方的【科普】）
        num_cached_blocks = seq.num_cached_tokens // bm.hash_block_size
        # PD consumer only: full prompt KV arrived via RDMA, safe to hash all.
        # Offload/LMCache path skipped — suffix KV not yet computed.
        # 【译】只限PD consumer：整个prompt的KV已经通过RDMA到达，可以全部登记hash。offload/LMCache路径跳过——后缀的KV还没算出来。
        if prefix_caching and not self._connector_flag("is_offload"):
            bm.register_received_prefix(seq)

        # Emit BlockStored(REMOTE) for delta blocks so external consumers
        # can track remote-resident KV.
        # 【译】为增量block发出 BlockStored(REMOTE) 事件，让外部消费者能追踪驻留在远端的KV。
        # 解释：跳过前num_cached_blocks个（本地原有的前缀），把其余有hash的block按顺序收集起来，
        #   连同父hash和起始token位置一起交给block_manager.record_remote_store
        if kv_events:
            remote_hashes: list[int] = []
            remote_tokens: list[int] = []
            parent_block_hash: int | None = None
            prev_hash: int | None = None
            first_remote_index: int | None = None
            for i, block_id in enumerate(seq.block_table):
                blk = bm.kv.block(block_id)
                if blk.hash == -1:
                    continue
                if i < num_cached_blocks:
                    prev_hash = blk.hash
                    continue
                if not remote_hashes:
                    parent_block_hash = prev_hash
                    # Derive the token offset from the first remote block's
                    # actual block-table index, not num_cached_blocks: an
                    # unhashed block skipped above can push the first remote
                    # block past num_cached_blocks, which would otherwise
                    # undercount the offset by whole blocks.
                    # 【译】用第一个远端block在block_table里的实际下标推出token偏移，而不是用 num_cached_blocks：上面跳过的未登记hash的
                    #   block可能把第一个远端block推到 num_cached_blocks 之后，否则偏移会少算整数个block。
                    first_remote_index = i
                remote_hashes.append(blk.hash)
                remote_tokens.extend(blk.token_ids)
                prev_hash = blk.hash
            if remote_hashes:
                # first_remote_index is always set when remote_hashes is
                # non-empty; assert it so a future refactor can't silently
                # regress into `None * hash_block_size`.
                # 【译】remote_hashes 非空时 first_remote_index 一定已设置；这里断言一下，免得将来重构时悄悄退化成 None * hash_block_size。
                assert first_remote_index is not None
                bm.record_remote_store(
                    block_hashes=remote_hashes,
                    token_ids=remote_tokens,
                    parent_block_hash=parent_block_hash,
                    # Block-table index -> global token position: under DCP
                    # each entry spans hash_block_size tokens, not block_size.
                    # 【译】block_table下标 → 全局token位置：在DCP下每个条目跨 hash_block_size 个token，而不是 block_size 个。
                    token_offset=first_remote_index * bm.hash_block_size,
                )
        return True

    def _promote_ready_remote_kv_requests(self) -> None:
        """Move completed remote-KV waiters ahead of fresh admissions.

        Offload waiters already own allocated blocks. If a fresh request at the
        head cannot allocate while a completed waiter sits behind it, the waiter
        cannot finish and free blocks. Preserve FIFO order within the ready and
        blocked slots.
        """
        # 【译】把远端KV已完成的等待者移到新准入请求的前面。
        #
        #   offload的等待者已经持有分配好的block。如果队头的新请求分配不到block，而一个已完成的等待者排在它后面，
        #   这个等待者就无法完成、也就无法释放block。在"就绪"和"阻塞"两组内部各自保持FIFO顺序。
        # 【分支】PD分离 / KV offload（第9批）。调用方：_schedule开头（每步一次）
        # 例：waiting=[新A, 等KV的B（已完成）, 新C, 等KV的D（未完成）] → [B, A, C, D]
        if not self.waiting or not (
            self.finished_recving_kv_req_ids or self.failed_recving_kv_req_ids
        ):
            return

        ready: deque[Sequence] = deque()
        blocked: deque[Sequence] = deque()
        while self.waiting:
            seq = self.waiting.popleft()
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS and (
                self._has_req_id(self.finished_recving_kv_req_ids, seq.id)
                or self._has_req_id(self.failed_recving_kv_req_ids, seq.id)
            ):
                ready.append(seq)
            else:
                blocked.append(seq)

        if ready:
            self.waiting.extend(ready)
            self.waiting.extend(blocked)
        else:
            self.waiting.extend(blocked)

    # 【分支】KV offload（第9批）：running里做到一半的chunked prefill，若connector判断"剩下的前缀可以从offload层加载"
    #   （should_park_partial_prefill_for_load），就把它停放回waiting队头、状态改为WAITING_FOR_REMOTE_KVS，计入在途加载，
    #   等加载完成再继续，而不是接着一块块地重算。调用方：_schedule开头，紧跟_promote_ready_remote_kv_requests
    def _park_ready_offload_partial_prefills(self) -> None:
        if (
            not self.running
            or self.kv_connector is None
            or not hasattr(self.kv_connector, "should_park_partial_prefill_for_load")
        ):
            return

        parked: deque[Sequence] = deque()
        keep_running: deque[Sequence] = deque()
        while self.running:
            seq = self.running.popleft()
            should_park = self.kv_connector.should_park_partial_prefill_for_load(seq)
            if should_park:
                if seq.is_partial_prefill:
                    seq.is_partial_prefill = False
                    self._partial_prefill_count -= 1
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
                self._count_inflight_load(seq)
                parked.append(seq)
            else:
                keep_running.append(seq)

        self.running = keep_running
        if parked:
            self.waiting.extendleft(reversed(parked))

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """Reconcile scheduler state with completed KV transfers.

        * ``finished_recving``: marks requests as ready for decode scheduling.
        * ``finished_sending``: releases the producer's send ownership; blocks
          remain deferred while the connector still has offload work.
        """
        # 【译】让调度器状态与已完成的KV传输对齐。
        #
        #   * finished_recving：把请求标记为可以调度decode。
        #   * finished_sending：释放producer的发送所有权；connector还有offload工作时，block继续延迟释放。
        # 【分支】KV传输（PD分离 + KV offload，第9批）
        # 调用方：EngineCore._poll_kv_transfer_progress（每步forward之后）、pp_engine_core.py（PP head汇总各stage的报告）
        # 输入：各TP worker报告汇总后的KVConnectorOutput（kv_transfer/disaggregation/types.py，由聚合器合并各rank的报告）
        # 各通道的去向：
        #   finished_recving（consumer收完KV）  → finished_recving_kv_req_ids，下一步准入时由_update_waiting_for_remote_kv认领
        #   failed_recving（consumer接收失败）  → _abort_failed_remote_receive
        #   finished_loading（offload加载完成） → 结清状态加载，再进finished_recving_kv_req_ids（与PD接收共用一条路）
        #   failed_loading（offload加载失败）   → failed_recving_kv_req_ids，准入时退回普通prefill（_consume_failed_remote_kv）
        #   finished_sending（producer发完KV）  → 释放发送认领，_maybe_release_deferred释放延迟的block
        #   finished_saving（offload保存完成）  → _maybe_release_deferred
        #   状态offload的存储报告               → 结清pin，成功的才登记hash；最后做超时回收
        if kv_connector_output is None:
            return

        is_producer = self._connector_flag("is_producer")
        is_offload = self._connector_flag("is_offload")

        process_completions = getattr(self.kv_connector, "process_completions", None)
        if callable(process_completions):
            kv_connector_output = process_completions(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            assert not is_producer, "Only consumer should update recving KV status"
            logger.debug("Finished recving KV transfer for request %s", req_id)
            if self._finish_aborted_load_cleanup(req_id):
                continue
            if metrics := getattr(self, "metrics", None):
                metrics.finish_kv_wait(req_id, succeeded=True)
            self.finished_recving_kv_req_ids.append(req_id)

        for req_id in kv_connector_output.failed_recving or ():
            if metrics := getattr(self, "metrics", None):
                metrics.finish_kv_wait(req_id, succeeded=False)
            assert not is_producer, "Only consumer should update failed KV recv status"
            if self._finish_aborted_load_cleanup(req_id):
                continue
            self._abort_failed_remote_receive(req_id)

        # The two loading channels carry state-tier loads as well as KV ones,
        # which buys the state leg the aggregator's per-request quorum for free.
        # They cannot be confused: a KV load is refused for any sequence with a
        # per-request cache, and a state load exists only for such a sequence.
        # 【译】这两个加载通道既承载KV加载，也承载状态层的加载，这让状态那一路免费得到了聚合器按请求的法定人数（quorum）检查。
        #   两者不会混淆：任何带每请求缓存的序列都会被拒绝做KV加载，而状态加载只存在于这类序列。
        # 解释：finish_kv_wait的succeeded只影响PD传输耗时直方图（metrics/scheduler.py里只在timing.is_pd且成功时记录）
        for req_id in kv_connector_output.finished_loading or ():
            if metrics := getattr(self, "metrics", None):
                metrics.finish_kv_wait(req_id, succeeded=False)
            assert is_offload, "Only offload connector should update loading KV status"
            logger.debug("Finished offload KV load for request %s", req_id)
            # Ahead of the abort guard: an aborted request's state group has to
            # be released too, and the guard takes the `continue`.
            # 【译】放在取消检查之前：已取消请求的状态组也必须释放，而取消检查会走 continue。
            self._settle_state_load(req_id, ok=True)
            if self._finish_aborted_load_cleanup(req_id):
                continue
            self.finished_recving_kv_req_ids.append(req_id)

        # A joint load can fail on its KV leg alone while its state H2D landed
        # intact (an LMCache LRU miss dropping the KV chunk). The connector
        # advertises those survivors on a failure-dominant, TP-quorumed
        # disposition channel, drained here on the same step the KV failure
        # report arrives: a survivor is *abandoned* (pending released, hash
        # kept loadable) rather than *failed* (hash forgotten). Everything else
        # -- a genuine state-leg miss, or a connector that never advertises the
        # channel -- still settles as a failure.
        # 【译】一次联合加载可能只在KV那一路失败，而它的状态H2D完好地落地了（LMCache的LRU未命中丢掉了KV块）。connector在一个
        #   "失败优先、经TP法定人数确认"的处置通道上公布这些幸存者，并在KV失败报告到达的同一步在这里取走：幸存者被"放弃"
        #   （释放挂起的加载，hash仍可加载），而不是"失败"（忘掉hash）。其他所有情况——真正的状态那一路未命中，
        #   或者connector从不公布这个通道——仍按失败结算。
        state_survived = getattr(self.kv_connector, "take_state_load_survived", None)
        state_survived = state_survived() if state_survived is not None else set()
        for req_id in kv_connector_output.failed_loading or ():
            if metrics := getattr(self, "metrics", None):
                metrics.finish_kv_wait(req_id, succeeded=False)
            assert (
                is_offload
            ), "Only offload connector should update failed KV load status"
            logger.warning(
                "Offload KV load failed for request %s; falling back to prefill.",
                req_id,
            )
            if req_id in state_survived:
                self._abandon_state_load(req_id)
            else:
                self._settle_state_load(req_id, ok=False)
            if self._finish_aborted_load_cleanup(req_id):
                continue
            self.failed_recving_kv_req_ids.append(req_id)

        finished_saving = kv_connector_output.finished_saving or ()
        for req_id in kv_connector_output.finished_sending or ():
            assert (
                self.kv_connector.is_producer
            ), "Only producer should free blocks after sending KV"
            logger.debug("Finished sending KV transfer for request %s", req_id)
            # Before asking, not after: the release below queries the connector.
            # 【译】在询问之前、而不是之后：下面的释放会去问connector。
            self._connector_send_finished(req_id)
            seq = self._deferred_sequence(req_id)
            if seq is None:
                # Already reclaimed by `_reconcile_stalled_deferred_saves` after
                # a stall; a late completion report has nothing left to free.
                # 【译】已经在停滞之后被 _reconcile_stalled_deferred_saves 回收；迟到的完成报告已经没有东西可释放。
                continue
            self._maybe_release_deferred(seq)

        for req_id in finished_saving:
            seq = self._deferred_sequence(req_id)
            if seq is not None:
                self._maybe_release_deferred(seq)
        # Early-release requests are never in `deferred_free_blocks` (they were
        # fully torn down, minus their lease, at finish time), so the loops
        # above have nothing to find for them. Drain independently of producer
        # mode because a MultiConnector may contain both a producer and an
        # offload leg.
        # 【译】提前释放的请求从不在 deferred_free_blocks 里（它们在结束时就已经被完全拆掉，只留租约），所以上面的循环找不到它们。
        #   与是否为producer模式无关地排空，因为一个MultiConnector里可能同时有producer和offload两路。
        self._drain_source_safe_releases()

        # The state-offload store reports, not keyed by request: by the time a
        # store lands its owner is long gone and only the hash remains. They
        # ride the connector's completion channels, so they are drained from the
        # connector with TP quorum already taken.
        #
        # Indexed only on the report, never at submission: a hash advertised
        # before its bytes exist parks the next request over that prefix
        # against a `get` that must miss.
        # 【译】状态offload的存储报告不按请求索引：存储落地时它的主人早就不在了，只剩下hash。它们搭connector的完成通道，
        #   所以从connector取出时已经做过TP法定人数确认。
        #
        #   只在报告到达时登记索引，绝不在提交时登记：在字节存在之前就公布的hash，会让下一个使用这个前缀的请求停在一个
        #   注定未命中的get上。
        bm = getattr(self, "block_manager", None)
        offload = getattr(bm, "state_offload", None)
        take = getattr(self.kv_connector, "take_state_reports", None)
        if offload is not None and take is not None:
            indexed, failed = take()
            # Both settle the pin; only success indexes the hash. A hash whose
            # bytes are not really there must never be voted for, and a pin
            # exists to keep the source still during the copy -- which is over
            # either way.
            # The source release first, and separately: it hands the PAGE
            # units back the moment the GPU stopped reading them, which is
            # before the CPU put has been decided. Waiting for the store
            # report would keep a whole image out of the pool across an
            # operation that cannot touch it.
            # 【译】两种结果都会结清pin；只有成功才登记hash。字节并不真的存在的hash绝不能被投票选中；而pin的作用是在拷贝期间
            #   让源保持不动——无论哪种结果，拷贝都已经结束了。
            #   先单独释放源：GPU一停止读取PAGE单元就把它们交还，这发生在CPU侧的put有结果之前。等存储报告的话，
            #   会让一整份镜像在一个根本碰不到它的操作期间一直不在池子里。
            release = getattr(self.kv_connector, "take_state_source_releases", None)
            if release is not None:
                for op in release():
                    bm.release_state_store_source(op)
            for op in indexed:
                bm.settle_state_store(op, ok=True)
            for op in failed:
                bm.settle_state_store(op, ok=False)
            # Unit-side twin of `_reconcile_stalled_deferred_saves`, on the same
            # window and for the same reason: the pin lives in this process
            # while the D2H runs in the worker, so a report lost to a crashed
            # worker or a dropped completion would hold a whole image out of
            # the pool forever. Recovery is total -- a leaked pin breaks no
            # `BlockPool` invariant -- so zeroing the count restores the record
            # exactly.
            # 【译】_reconcile_stalled_deferred_saves 在单元一侧的孪生，窗口相同、理由相同：pin在这个进程里，而D2H在worker里执行，
            #   所以因worker崩溃或完成信号丢失而丢掉的报告，会让一整份镜像永远不在池子里。恢复是完全的——泄漏的pin不破坏任何
            #   BlockPool 不变量——所以把计数清零就能准确恢复记录。
            bm.reclaim_stale_state_store_pins(self._save_abandon_timeout_s())
            # Load-side twin of the same defence: `deallocate` parks a slot when
            # it tears down a request whose state load is still in flight, on the
            # promise that `settle_state_load` will hand it back. A crashed
            # worker or dropped completion breaks that promise and the slot sits
            # off the free list forever, wedging `can_allocate`'s state gate.
            # Same window, same "cannot tell lost from slow" caveat as the pins.
            # 【译】同一种防御在加载一侧的孪生：deallocate 在拆掉一个状态加载仍在途的请求时，会把槽位停放起来，指望
            #   settle_state_load 把它交还。worker崩溃或完成信号丢失会打破这个承诺，槽位就永远不在空闲列表上，卡住
            #   can_allocate 的状态闸门。窗口相同，也有和pin一样的"分不清是丢了还是慢"的局限。
            bm.reconcile_orphan_load_slots(self._save_abandon_timeout_s())

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # 【译】返回 (running请求数, waiting请求数)。
        # 调用方：引擎状态行（_record_throughput）和指标收集（engine_utility.py的collect_metrics）
        return len(self.running), len(self.waiting)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        # 【译】调度器内部队列里还有未完成的请求时返回True。
        return self.get_num_unfinished_requests() > 0

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        # 【译】有未完成的请求，或者有已结束但还没在 SchedulerOutputs 里返回的请求时，返回True。
        # 观察：ATOM没有SchedulerOutputs（这句沿用自vLLM）；实现只看waiting + running，不含_rejected和deferred_free_blocks
        #   （is_finished会看这两项）。atom/里只有RapidServe的PrefillEngineCore / DecodeEngineCore调用它
        return self.has_unfinished_requests()

    # 估算下一步的形状：(是否prefill, token数, 请求数)
    # 观察：atom/中没有调用方，只有tests/test_scheduler.py用到；它的条件也和_schedule不一致——decode序列的num_cached_tokens
    #   停在prompt长度（见sequence.py里num_cached_tokens的说明），追加过生成token后"num_cached_tokens < num_tokens"恒成立，
    #   会被当成"续算prefill"报告（测试里的序列没有追加过token，所以测不到）；decode分支也按每条1个token算，没考虑MTP的k+1
    def get_next_batch_info(self) -> tuple[bool, int, int]:
        # Check for partial prefills in running (chunked prefill resume)
        # 【译】检查running里有没有做到一半的prefill（chunked prefill续算）
        for seq in self.running:
            if seq.num_cached_tokens < seq.num_tokens:
                remaining = seq.num_tokens - seq.num_cached_tokens
                chunk = min(remaining, self.max_num_batched_tokens)
                return (True, chunk, 1)
        # Only consider waiting seqs that are not blocked on a remote KV
        # transfer (P/D disaggregation) when deciding if we can prefill.
        # 【译】判断能否做prefill时，只考虑没有被远端KV传输（P/D分离）阻塞的waiting序列。
        eligible_waiting = [
            seq
            for seq in self.waiting
            if seq.status
            not in (SequenceStatus.ABORTED, SequenceStatus.WAITING_FOR_REMOTE_KVS)
        ]
        if eligible_waiting:
            # new request is waiting, will do prefill
            # 【译】有新请求在等，下一步会做prefill
            num_reqs = 0
            total_tokens = 0
            for seq in eligible_waiting:
                tokens = seq.num_tokens - seq.num_cached_tokens
                if self.enable_chunked_prefill:
                    tokens = min(tokens, self.max_num_batched_tokens - total_tokens)
                if total_tokens + tokens > self.max_num_batched_tokens:
                    break
                if num_reqs >= self.max_num_seqs:
                    break
                total_tokens += tokens
                num_reqs += 1
            return (True, total_tokens, num_reqs)
        elif self.running:
            # decode
            # 【译】decode（每条running序列1个token）
            num_tokens = len(self.running)
            return (False, num_tokens, num_tokens)
        else:
            # No requests
            # 【译】没有请求
            return (False, 0, 0)

    # 调度延迟（--scheduler-delay-factor，默认0=关闭）：_schedule的Phase 2每准入一条新prefill前都问一次"等够了没有"
    #   目的：请求零星到达时，每来一条就做一次小prefill，会频繁打断decode（一步只做prefill或只做decode，见engine_core.py
    #   文件开头【2】）；稍等一会儿把几条请求攒成一批prefill，decode被打断的次数更少
    # 规则：最早到达的waiting请求已等待 > delay_factor × 上一个prefill步的耗时，或者running为空（没有decode可打断）→ 放行
    # last_prompt_latency的测法：上一步排过prefill时（prev_prompt=True，在_schedule的prefill分支里置位），
    #   用"这次调用 - 上次调用"的时间差近似上一个prefill步的耗时
    # 例：delay_factor=0.5，上一个prefill步耗时200ms → 最早的请求等满100ms才放行（running为空时立即放行）
    # 代价：TTFT最多增加 delay_factor × 一个prefill步的耗时；目的与PrefillDelayer（prefill_delayer.py）相近
    # 对比nano-vllm：nano-vllm没有调度延迟
    def _passed_delay(self, now: float) -> bool:
        # borrowed from https://github.com/vllm-project/vllm/pull/3279
        # if the earliest arrived request has waited long enough,
        # i.e., > delay_factor * last_prompt_latency (the latency of last prefill in unit of seconds),
        # new prefill should be scheduled immediately
        # 【译】借鉴自 https://github.com/vllm-project/vllm/pull/3279
        #   如果最早到达的请求已经等得够久，即超过 delay_factor * last_prompt_latency（上一次prefill的耗时，单位秒），
        #   就应该立刻调度新的prefill
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        # 【译】推迟调度prompt，让等待队列攒满一些
        if self.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min([seq.arrive_time for seq in self.waiting])
            passed_delay = (now - earliest_arrival_time) > (
                self.delay_factor * self.last_prompt_latency
            ) or not self.running
        else:
            passed_delay = True
        return passed_delay


# 【分支】RapidServe（同一组GPU上的PD分离）的prefill侧，见第9批；进程、通道、一个请求的时间线见engine_core.py里
#   PrefillEngineCore上方的"【分支】PD分离（同GPU的RapidServe）"；runner是model_runner.py的RapidServeModelRunner（第5批）
# 对比基础Scheduler：不切chunk（config.py里动态分块也因"the RapidServe prefill scheduler never chunks"被关掉）、不抢占、
#   没有自己的前缀缓存（block由decode侧的BlockManager分配，已缓存的token数随BlockAssignment.num_cached_tokens传来）
# 一个请求在这里的经历：ADD到达 → waiting → 收到BlockAssignment（PrefillEngineCore._apply_pending_assignments填好block_table）
#   → _schedule挑出来进running → prefill_forward → PrefillEngineCore发PrefillDone(第一个token)、标FINISHED、移出running
class PrefillScheduler:
    """Scheduler for the disaggregated prefill process.

    Key differences from the base Scheduler:
    - No BlockManager: KV blocks are pre-assigned by DecodeEngineCore and
      written into seq.block_table before schedule() is called.
    - schedule() only runs sequences that already have a non-empty block_table.
      Sequences still waiting for a BlockAssignment message stay in waiting.
    - postprocess() is a no-op: prefill produces no sampled tokens.
    - Decode scheduling is never performed.
    """

    # 【译】分离式prefill进程的调度器。
    #
    #   与基础Scheduler的主要区别：
    #   - 没有BlockManager：KV block由DecodeEngineCore预先分配，并在调用 schedule() 之前写进 seq.block_table。
    #   - schedule() 只运行已经有非空 block_table 的序列。还在等 BlockAssignment 消息的序列留在waiting里。
    #   - postprocess() 是空操作：prefill不产生采样token。
    #   - 从不做decode调度。
    # 解释：PrefillEngineCore._process_engine_step根本不调用postprocess：它自己把prefill_forward采样出的第一个token
    #   用PrefillDone发给decode

    # Every request here is also held by the decode side, whose queues span
    # the full lifetime; the aggregator drops this rank's counts so an
    # in-flight prefill is not counted on both. See `_METRICS_ROLE` on
    # `Scheduler`.
    # 【译】这里的每个请求decode侧也持有一份，而decode侧的队列覆盖请求的整个生命周期；聚合器会丢掉这个rank的计数，
    #   免得一个在途的prefill被两边都算一遍。见 Scheduler 上的 _METRICS_ROLE。
    _METRICS_ROLE = "prefill"

    def __init__(self, config: Config, disagg_cu_shm_name: str = ""):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = None  # blocks managed by decode process
        pc = getattr(config, "parallel_config", None)
        self.metrics = SchedulerMetrics(
            getattr(pc, "data_parallel_rank", 0)
            * getattr(config, "pipeline_parallel_size", 1)
            + getattr(pc, "pipeline_parallel_rank", 0),
            self._METRICS_ROLE or "default",
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # spec decode not used on prefill side
        # 【译】prefill侧不使用投机解码
        self.use_spec = False
        self.spec_decode_local = False
        self.mtp_k = 0
        # Only the throughput section applies to the prefill side: it does not
        # speculate, and it has no BlockManager to source prefix-cache hits.
        # Throughput follows config.enable_log_stats (default True), matching
        # the aggregated Scheduler and the decode side.
        # 【译】prefill侧只用到吞吐那一节：它不做投机，也没有BlockManager可以提供前缀缓存命中。吞吐统计跟随
        #   config.enable_log_stats（默认True），与合并式Scheduler和decode侧一致。
        parallel_cfg = getattr(config, "parallel_config", None)
        dp_rank = (
            getattr(parallel_cfg, "data_parallel_rank", None)
            if parallel_cfg is not None
            else None
        )
        self.engine_stats = EngineStats(
            engine_index=dp_rank or 0,
            label="Prefill ",
            enable_log_stats=config.enable_log_stats,
            throughput_log_interval_s=config.throughput_log_interval,
        )
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_finished_requests = 0
        self.total_preemptions = 0

        # Shared memory for dynamic CU partitioning.
        # Layout: [0:4] = decode_tokens (uint32)
        # 【译】用于动态CU划分的共享内存。布局：[0:4] = decode_tokens（uint32）。
        # 解释：只有disagg_constrained=True时DisaggCoreManager才创建它（engine_core_mgr.py）；decode侧每步写入本步的decode token数，
        #   prefill侧在_schedule里读出来、自己用_optimal_cu_fraction算CU比例（CU划分见engine_core.py里RapidServe说明的"代价①"）
        self._cu_shm = None
        if disagg_cu_shm_name:
            import multiprocessing.shared_memory

            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=disagg_cu_shm_name, create=False
            )
            logger.info("initialized shared memory")
        self._pending_lock = threading.Lock()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def publish_kv_events(self) -> None:
        # No-op: disagg prefill has no BlockManager — the decode process owns the
        # KV blocks and emits KV events. Defined so EngineCore.busy_loop teardown
        # (which calls this on every scheduler) works for PrefillScheduler.
        # 【译】空操作：分离式prefill没有BlockManager——decode进程拥有KV block并发出KV事件。定义它只是为了让
        #   EngineCore.busy_loop 的收尾（会对每种调度器调用它）对 PrefillScheduler 也能工作。
        pass

    def shutdown_kv_events(self) -> None:
        # No-op: see publish_kv_events.
        # 【译】空操作：见 publish_kv_events。
        pass

    def add(self, seq: Sequence):
        self.metrics.enqueue(seq)
        self.waiting.append(seq)

    def extend(self, seqs: list):
        for seq in seqs:
            self.metrics.enqueue(seq)
        self.waiting.extend(seqs)

    def schedule(self):
        """Run a scheduling pass and close the throughput window.

        Override `_schedule`, not this — see `Scheduler.schedule` for why the
        tick lives at the one entry point instead of at each early return.
        """
        # 【译】跑一遍调度，并结算吞吐统计窗口。
        #
        #   要重写的是 _schedule，不是这个——为什么把计时放在唯一的入口、而不是每个提前返回处，见 Scheduler.schedule。
        result = self._schedule()
        self._record_throughput(num_prompt_tokens=_prompt_tokens_of(result))
        return result

    def _schedule(self):
        """Schedule only sequences whose block_table has been populated.

        Sequences that do not yet have a block assignment (block_table is
        empty) remain in the waiting queue and will be reconsidered on the
        next call.

        Returns (ScheduledBatch, dict[seq_id, Sequence]) or (None, {}) when
        no sequence is ready.
        """
        # 【译】只调度 block_table 已经填好的序列。
        #
        #   还没有拿到block分配（block_table为空）的序列留在等待队列里，下次调用时再考虑。
        #
        #   返回 (ScheduledBatch, dict[seq_id, Sequence])；没有序列就绪时返回 (None, {})。
        # 对比nano-vllm：相当于nano-vllm的prefill阶段去掉block分配和chunked prefill——按FIFO凑batch，
        #   条数到max_num_seqs、或再加一条就超过max_num_batched_tokens时停
        scheduled_seqs = {}
        num_scheduled_tokens = []
        num_batched_tokens = 0
        num_seqs = 0

        with self._pending_lock:
            # Collect ready sequences (have received BlockAssignment from decode)
            # 【译】收集就绪的序列（已经从decode收到BlockAssignment的）
            # 锁：和PrefillEngineCore的后台收消息线程（_recv_block_assignments）共用，它在锁内登记收到的BlockAssignment
            ready = [s for s in self.waiting if s.block_table]

            for seq in ready:
                if num_seqs >= self.max_num_seqs:
                    break
                num_new_tokens = seq.num_tokens - seq.num_cached_tokens
                # 观察：这里不切chunk——若排在最前面的就绪序列本身就超过max_num_batched_tokens，第一轮就break，本步什么都不调度，
                #   之后每一步都一样，它会一直挡在最前面（按代码推断，未见有地方提前拒绝这种请求）
                if num_batched_tokens + num_new_tokens > self.max_num_batched_tokens:
                    break
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                seq.type = SequenceType.PREFILL
                self.running.append(seq)
                scheduled_seqs[seq.id] = seq
                num_scheduled_tokens.append(num_new_tokens)
                # 进running只为计数；forward之后PrefillEngineCore把它标成FINISHED并移出running
                num_batched_tokens += num_new_tokens
                num_seqs += 1

        if not scheduled_seqs:
            return None, {}

        # Read the decode tokens from decode process via shared memory.
        # 【译】通过共享内存读取decode进程的decode token数。
        # decode本步的token数 >= 64且本批有prefill token → 给prefill一半CU（_optimal_cu_fraction，目前只有0.5一档）；
        #   否则None = 不加CU掩码。cu_stream_fraction随batch交给runner选stream

        cu_fraction = None

        if self._cu_shm is not None:
            decode_tokens = struct.unpack_from("I", self._cu_shm.buf, 0)[0]
            cu_fraction = _optimal_cu_fraction(decode_tokens, num_batched_tokens)

        return (
            ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=num_batched_tokens,
                total_tokens_num_prefill=num_batched_tokens,
                total_seqs_num=num_seqs,
                total_seqs_num_prefill=num_seqs,
                cu_stream_fraction=cu_fraction,
            ),
            scheduled_seqs,
        )

    def _record_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        """`Scheduler._record_throughput`'s counterpart for the prefill side.

        This process schedules no decode, so generation stays 0, and the
        decode process owns the KV blocks — no local BlockManager means
        `kv_usage=None`, which the line reports as `n/a` rather than as a
        real, empty pool. Same `window_expired` gate as the base class, and
        for the same reason.
        """
        # 【译】prefill侧对应 Scheduler._record_throughput 的版本。
        #
        #   这个进程不调度decode，所以生成数一直为0；KV block归decode进程所有——本地没有BlockManager，所以 kv_usage=None，
        #   状态行把它报成 n/a，而不是一个真实的空池。与基类一样用 window_expired 做门控，理由也相同。
        stats = self.engine_stats
        if not stats.throughput_enabled:
            return
        stats.update_throughput(num_prompt_tokens, num_generation_tokens)
        if not stats.window_expired(time.monotonic()):
            return
        num_running_reqs, num_waiting_reqs = self.get_request_counts()
        stats.maybe_log_throughput(
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            kv_usage=None,
        )

    def heartbeat_throughput(self, now: float) -> None:
        """Idle-pass counterpart of `Scheduler.heartbeat_throughput`."""
        # 【译】空闲轮次时对应 Scheduler.heartbeat_throughput 的版本。
        if self.engine_stats.window_expired(now):
            self._record_throughput()

    def postprocess(self, seqs, fwd_output, stream_output_queue=None) -> list:
        """No-op: prefill produces no sampled tokens."""
        # 【译】空操作：prefill不产生采样token。
        return []

    def get_next_batch_info(self) -> tuple:
        # 与基类不同，返回二元组(是否有prefill, 队头还要算的token数)；atom/中没有调用方
        if self.waiting:
            seq = self.waiting[0]
            return (True, seq.num_tokens - seq.num_cached_tokens)
        return (False, 0)

    def get_request_counts(self) -> tuple[int, int]:
        """(running, waiting). Not a `Scheduler` subclass, so this is declared
        rather than inherited — the engine-status line and any metrics reader
        expect every scheduler to answer it."""
        # 【译】(running, waiting)。它不是 Scheduler 的子类，所以这里要自己声明而不是继承——引擎状态行和所有指标读取方
        #   都要求每种调度器都能回答它。
        return len(self.running), len(self.waiting)

    def get_num_unfinished_requests(self) -> int:
        return sum(self.get_request_counts())


# 【分支】RapidServe的decode侧（第9批）：继承基础Scheduler（有BlockManager、前缀缓存账本、抢占、同一个postprocess），只是自己不做prefill
# 队列与状态（核对代码）：
#   ADD ─extend─> waiting ─allocate_waiting（分配block，DecodeEngineCore发BlockAssignment）─> prefill_waiting {seq_id: seq}
#                                                                                                │ PrefillDone到达（后台线程
#                                                                                                │ on_prefill_done：追加T0和k个占位）
#                                                                                                ▼
#                              running  <───── _schedule开头（busy_loop线程）把状态改为RUNNING ───── prefill_done
#   running里的序列每步decode（每条mtp_k+1个token），DecodeEngineCore调用postprocess（不传batch）判停、释放
class DecodeScheduler(Scheduler):
    """Scheduler for the disaggregated decode process.

    Manages 3 queues:
    - waiting:         new requests pending block allocation
    - prefill_waiting: blocks allocated, BlockAssignment sent, awaiting PrefillDone
    - running:         ongoing decode sequences

    Block allocation is separated from scheduling: allocate_waiting() is called
    by DecodeEngineCore after draining the input queue, and returns newly
    allocated sequences so the engine can send BlockAssignment to prefill.

    on_prefill_done() promotes sequences directly from prefill_waiting to
    running.  schedule() only schedules the running queue as decode batches.
    """

    # 【译】分离式decode进程的调度器。
    #
    #   管理3个队列：
    #   - waiting：         等待分配block的新请求
    #   - prefill_waiting： 已分配block、已发出BlockAssignment，正在等PrefillDone
    #   - running：         正在decode的序列
    #
    #   block分配和调度是分开的：DecodeEngineCore 排空输入队列后调用 allocate_waiting()，它返回新分配到block的序列，
    #   好让引擎给prefill发BlockAssignment。
    #
    #   on_prefill_done() 把序列从 prefill_waiting 直接提升到running。schedule() 只把running队列调度成decode batch。
    # 解释：实际还有第4个队列prefill_done：on_prefill_done在后台线程里只把序列放进prefill_done，由_schedule在busy_loop线程里
    #   移进running（见上面的图）

    _ENGINE_LABEL = "Decode "
    _METRICS_ROLE = "decode"

    def abort_request(self, req_id: int) -> bool:
        # RapidServe does not drain the standard scheduler's rejection queue.
        # Preserve its mark-only cancellation path.
        # 【译】RapidServe不排空标准调度器的拒绝队列（_rejected）。保留它"只做标记"的取消方式。
        # 解释：running里被标成ABORTED的序列，下一次postprocess开头会以"aborted"结束
        # 观察（按代码推断）：这里只在running / waiting里找，处在prefill_waiting / prefill_done里的请求返回False，取消不生效；
        #   在waiting里被标记的，allocate_waiting不看状态照样分配，之后_schedule把prefill_done里的序列一律改成RUNNING，
        #   ABORTED也会被覆盖。atom/里也没有地方发送DisaggMsgType.ABORT（PrefillEngineCore只有接收分支）
        with self._prefill_lock:
            for seq in (*self.running, *self.waiting):
                if seq.id == req_id:
                    seq.status = SequenceStatus.ABORTED
                    return True
        return False

    def get_request_counts(self) -> tuple[int, int]:
        """Fold in the two queues this scheduler adds.

        `allocate_waiting()` drains `waiting` almost immediately, so a request
        spends most of its life in `prefill_waiting` (blocks assigned, awaiting
        PrefillDone) and lands in `prefill_done` before `schedule()` promotes
        it. Counting only the base pair reports `Running: 0, Waiting: 0` on an
        engine holding a full load of in-flight requests — on the status line
        and, because `/metrics` reads this same method, on the dashboard too.
        """
        # 【译】把本调度器新增的两个队列也算进来。
        #
        #   allocate_waiting() 几乎立刻就会排空 waiting，所以一个请求一生的大部分时间都在 prefill_waiting 里（已分配block、
        #   在等PrefillDone），在 schedule() 提升它之前还会落在 prefill_done 里。只数基类那两个队列的话，一个满载着在途请求的
        #   引擎会报 Running: 0, Waiting: 0——状态行上如此，而且因为 /metrics 读的也是这个方法，看板上也是如此。
        return (
            len(self.running) + len(self.prefill_done),
            len(self.waiting) + len(self.prefill_waiting),
        )

    def __init__(
        self,
        config: Config,
        disagg_cu_shm_name: str = "",
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        super().__init__(
            config,
            state_runtime=state_runtime,
        )
        # seq_id → Sequence; blocks allocated, BlockAssignment sent, awaiting PrefillDone.
        # 【译】seq_id → Sequence；已分配block、已发出BlockAssignment，在等PrefillDone。
        self.prefill_waiting: dict[int, Sequence] = {}
        self.prefill_done: deque[Sequence] = deque()
        # prefill_done：PrefillDone已到、还没被_schedule移进running的序列（后台线程写，busy_loop线程读）
        # Shared memory for dynamic CU partitioning.
        # 【译】用于动态CU划分的共享内存。
        self._cu_shm = None
        if disagg_cu_shm_name:
            # decode侧负责写：这里先清零，之后每步写本步decode的token数（见_schedule）
            import multiprocessing.shared_memory

            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=disagg_cu_shm_name, create=False
            )
            struct.pack_into("I", self._cu_shm.buf, 0, 0)

        # Protects prefill_waiting and running: on_prefill_done is called
        # from the _recv_prefill_done background thread.
        # 【译】保护 prefill_waiting 和 running：on_prefill_done 是从后台线程 _recv_prefill_done 里调用的。
        # 观察：on_prefill_done本身并不拿这把锁，它对prefill_waiting（dict.pop）和prefill_done（deque.append）的操作
        #   依赖CPython里这些单个操作的原子性；锁实际护住的是allocate_waiting里的分配和_schedule里的decode挑选
        self._prefill_lock = threading.Lock()
        self.cu_fraction: float | None = None
        # cu_fraction：本步给prefill的CU比例（只在有序列刚prefill完时重算），作为cu_stream_fraction随decode batch交给runner

    def is_finished(self) -> bool:
        # Kept explicit rather than derived from get_request_counts: unlike the
        # base it deliberately ignores `_rejected` and `deferred_free_blocks`.
        # If a queue is ever added to this scheduler it has to be listed here
        # *and* in get_request_counts.
        # 【译】显式列出而不是从 get_request_counts 推导：与基类不同，它有意不看 _rejected 和 deferred_free_blocks。
        #   将来给这个调度器加队列的话，必须在这里"和" get_request_counts 里都列上。
        return (
            not self.waiting
            and not self.prefill_waiting
            and not self.running
            and not self.prefill_done
        )

    def get_num_unfinished_requests(self) -> int:
        # Derived, so the queue inventory lives in get_request_counts alone.
        # `has_requests` / `has_unfinished_requests` come off the base's chain
        # through this method and need no override of their own.
        # 【译】由推导得到，这样队列清单只在 get_request_counts 里维护一份。has_requests / has_unfinished_requests
        #   通过基类的调用链经过这个方法，不需要各自重写。
        return sum(self.get_request_counts())

    def allocate_waiting(self) -> list[Sequence]:
        """Allocate KV blocks for sequences in waiting; move them to prefill_waiting.

        Returns newly allocated sequences so DecodeEngineCore can send a
        BlockAssignment message to the prefill process for each one.
        Called from the main busy_loop thread only.
        """
        # 【译】为waiting里的序列分配KV block，并把它们移到 prefill_waiting。
        #
        #   返回新分配到block的序列，好让 DecodeEngineCore 给prefill进程为每一条发一个 BlockAssignment 消息。
        #   只从主 busy_loop 线程调用。
        # 调用方：DecodeEngineCore.pull_and_process_input_queue（每轮收完新请求后）。FIFO：队头分配不到就停（同nano-vllm的准入）
        # 观察（按代码推断）：can_allocate的返回值（前缀缓存命中的block数）只用来判断够不够，没有传给allocate（默认0），
        #   所以这条路径不复用前缀缓存命中：num_cached_tokens=0，prefill侧整段重算
        newly_allocated = []
        while self.waiting:
            seq = self.waiting[0]
            with self._prefill_lock:
                if self.block_manager.can_allocate(seq) < 0:
                    logger.warning("Cannot allocate prefill")
                    break
                if not self.block_manager.allocate(seq):
                    # Disown could not be backed (finding #2); leave the seq at
                    # the front of waiting to retry once the pool has room.
                    # 【译】disown 没法得到支撑（finding #2）；把序列留在waiting队头，等池子有空间时重试。
                    # 解释：allocate返回False的含义见BlockManager.allocate的docstring（共享前缀的私有拷贝分不到block）
                    self.block_manager.deallocate(seq)
                    break
                # The prefill process drafts into this allocation too.
                # 【译】prefill进程的草稿也写进这份分配里。
                # 解释：MTP草稿层在prompt之后的draft_lookahead个位置写KV，见_allocate_prefill_draft_lookahead
                self._allocate_prefill_draft_lookahead(seq)
            self.waiting.popleft()

            # Shared-cache prefill writes into this allocation; there is no
            # KV transfer to time or retain a separate metrics reference for.
            # 【译】共享KV cache的prefill直接写进这份分配；没有KV传输需要计时，也不必为指标另外保留一个引用。
            self.prefill_waiting[seq.id] = seq
            newly_allocated.append(seq)
        return newly_allocated

    def on_prefill_done(
        self, seq_id: int, num_tokens_computed: int, sampled_token_id: int
    ) -> None:
        """Promote a sequence from prefill_waiting directly to running.

        Called from the _recv_prefill_done background thread.
        sampled_token_id is the first generated token sampled by the prefill
        process; it is appended here so that context_lens and slot_mapping
        match the non-disagg postprocess state before the first decode step.
        """
        # 【译】把一条序列从 prefill_waiting 直接提升到running。
        #
        #   从后台线程 _recv_prefill_done 调用。sampled_token_id 是prefill进程采样出的第一个生成token；在这里追加它，
        #   使得第一个decode步之前的 context_lens 和 slot_mapping 与非分离模式下postprocess之后的状态一致。
        # 解释：实际放进的是prefill_done，_schedule开头才移进running
        # 例（k=mtp_k=2，prompt 5个token，prefill采出T0=77）：token_ids=[p0..p4, 77, E, E]，num_cached_tokens=5；
        #   第一个decode的窗口是末尾k+1=3个位置[77, E, E]

        seq = self.prefill_waiting.pop(seq_id, None)
        if seq is not None:
            seq.num_cached_tokens = num_tokens_computed
            seq.append_token(sampled_token_id)
            seq.is_first_decode = True
            # Decode schedules a full target verification window.  The remote
            # prefill producer normally sends only T0, so pad the missing draft
            # positions exactly as the monolithic P/D handoff does.  Without
            # this, the trailing mtp_k+1 slice reaches back into prompt tokens.
            # 【译】decode会调度一个完整的目标模型验证窗口。远端的prefill producer通常只发来T0，所以像单体P/D交接那样把缺的
            #   草稿位置补齐。不补的话，末尾 mtp_k+1 个token的切片会往回伸进prompt里。
            # 观察：与单体P/D交接（_schedule_first_decode_after_remote_kv）不同，这里没有把num_placeholder_tokens记成mtp_k；
            #   第一次拿到输出之前若被抢占，preempt只删记下的宽度，这些eos会留在上下文里（第一次覆盖后postprocess会重新算出正确宽度）
            for _ in range(self.mtp_k):
                seq.append_token(self.eos_token_id)
            seq.first_token_time = time.time()
            self.prefill_done.append(seq)

    def _schedule(self):
        """Schedule decode-only batches.

        Sequences are promoted directly from prefill_waiting to running by
        on_prefill_done(); this method only schedules the running queue.

        Overrides the base `_schedule`, so the inherited `schedule()` still
        closes the throughput window on every one of the returns below.
        """
        # 【译】只调度decode batch。
        #
        #   序列由 on_prefill_done() 直接从 prefill_waiting 提升到running；这个方法只调度running队列。
        #
        #   它重写了基类的 _schedule，所以继承来的 schedule() 仍会在下面每一个返回处结算吞吐窗口。
        # 对比nano-vllm：就是nano-vllm的decode阶段（running队头取、block不够抢占队尾、实在不行抢占自己），
        #   只是每条序列调度mtp_k+1个token，且没有prefill阶段

        # This override does not call `super()._schedule()`, but it does route
        # through the same `block_manager.allocate` and the same `postprocess`,
        # so it owes the state pool the same two hooks. Without this one the
        # pins taken by every resume accumulate forever and admission starves.
        # 【译】这个重写不调用 super()._schedule()，但它走的是同一个 block_manager.allocate 和同一个 postprocess，
        #   所以欠状态池同样的两个钩子。没有这一个，每次恢复时拿的pin会永远累积下去，准入被饿死。
        self.block_manager.complete_previous_state_batch()

        prefill_finished = False
        while self.prefill_done:
            seq = self.prefill_done.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.type = SequenceType.DECODE
            # Append the first generated token sampled by the prefill process.
            # In non-disagg mode, Scheduler.postprocess() does this after the
            # prefill forward (is_deferred_out=True always appends one placeholder
            # that is later overwritten with the real token from the async queue).
            # In disagg mode the prefill process ran sampling and sent us the real
            # token; appending it here puts num_tokens, context_lens, and
            # slot_mapping in the same state as non-disagg before the first decode
            # step.
            # 【译】追加prefill进程采样出的第一个生成token。在非分离模式下，Scheduler.postprocess() 在prefill的forward之后做这件事
            #   （is_deferred_out=True 时总是追加一个占位，之后用异步队列里的真实token覆盖它）。在分离模式下，prefill进程做了采样
            #   并把真实token发给我们；在这里追加它，使 num_tokens、context_lens 和 slot_mapping 在第一个decode步之前与非分离模式
            #   处于相同状态。
            # 解释：这段英文说的追加实际已在on_prefill_done里完成，这里只改状态并放进running
            self.running.append(seq)
            prefill_finished = True

        if not self.running:
            self.cu_fraction = None
            if self._cu_shm is not None:
                struct.pack_into("I", self._cu_shm.buf, 0, 0)
            return None

        scheduled_seqs: dict[int, Sequence] = {}
        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        with self._prefill_lock:
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                # logger.warning("decode state: waiting=%d prefill_waiting=%d prefill_done=%d running=%d free_blocks=%d",
                #     len(self.waiting), len(self.prefill_waiting), len(self.prefill_done),
                #     len(self.running), self.block_manager.kv.num_free)
                while not self.block_manager.can_append(
                    seq, self.mtp_k + 1, self.draft_lookahead
                ):
                    logger.warning("Cannot allocate")
                    if self.running:
                        self.preempt(self.running.pop())
                        # 观察（按代码推断，未实测）：被抢占的序列回到waiting，下一次allocate_waiting会重新分配block、再发一次BlockAssignment；
                        #   但prefill进程那份序列在prefill完成后已移出队列（PrefillEngineCore._process_engine_step），
                        #   _apply_pending_assignments只匹配它waiting里的序列，于是这条请求不会再被prefill，会一直停在prefill_waiting
                    else:
                        self.preempt(seq)
                        break
                else:
                    if self.spec_decode_local and seq.spec_token_ids.size > 0:
                        scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                    num_new_tokens = self.mtp_k + 1
                    self.block_manager.may_append(
                        seq, num_new_tokens, lookahead=self.draft_lookahead
                    )
                    scheduled_seqs[seq.id] = seq
                    seq.type = SequenceType.DECODE
                    num_scheduled_tokens.append(num_new_tokens)

        if not scheduled_seqs:
            self.cu_fraction = None
            if self._cu_shm is not None:
                struct.pack_into("I", self._cu_shm.buf, 0, 0)
            return None

        # 同nano-vllm：调度过的序列放回running队头，保持原来的顺序
        self.running.extendleft(reversed(scheduled_seqs.values()))

        total_tokens_num_decode = sum(num_scheduled_tokens)

        # Dynamic CU partitioning: decode decides the fraction based on its
        # batch size and the total tokens queued for prefill, then writes the
        # decode tokens to shared memory for PrefillScheduler to read.
        # 【译】动态CU划分：decode根据自己的batch大小和prefill队列里排着的总token数决定比例，然后把decode token数写进共享内存，
        #   供PrefillScheduler读取。
        # 解释：两边各自用_optimal_cu_fraction算：decode侧用(本步decode token数, prefill_waiting里的总token数)，只在本步有序列
        #   刚prefill完时重算；prefill侧用(读到的decode token数, 它本批的prefill token数)

        if self._cu_shm is not None:
            struct.pack_into("I", self._cu_shm.buf, 0, total_tokens_num_decode)
            if prefill_finished:
                pwait = sum(seq.num_tokens for seq in self.prefill_waiting.values())
                self.cu_fraction = _optimal_cu_fraction(total_tokens_num_decode, pwait)

        # 安全检查：草稿轮要写KV的位置都已分配block（否则会写进0号block，见_check_draft_writes_allocated）
        self._check_draft_writes_allocated(scheduled_seqs.values())
        decode_batch = ScheduledBatch(
            seqs=scheduled_seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num_decode,
            total_tokens_num_decode=total_tokens_num_decode,
            total_seqs_num=len(scheduled_seqs),
            total_seqs_num_decode=len(scheduled_seqs),
            num_spec_step=self.mtp_k if self.spec_decode_local else 0,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            cu_stream_fraction=self.cu_fraction,
            state_maintenance_ops=self.block_manager.take_state_maintenance_ops(),
        )
        # batch构造时已读取is_first_decode，这里清掉
        self._consume_first_decode(scheduled_seqs)
        return (decode_batch, scheduled_seqs)
