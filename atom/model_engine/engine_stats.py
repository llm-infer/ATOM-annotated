# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unified engine statistics.

Holds :class:`EngineStats`, which folds in the former ``SpecStats`` and
``CacheStats`` and adds a throughput section alongside them. Kept out of
``scheduler.py`` because it is engine-level rather than scheduler-level: every
scheduler owns one, and ``engine_utility`` reads the one on the scheduler it
was handed to answer ``/debug/mtp_stats``, ``/debug/cache_stats`` and
``/metrics``.

Both P/D processes replace the scheduler that ``EngineCore.__init__`` built
the handler around, so each rewires ``utility_handler.scheduler`` to the one
it actually runs; the prefill side's snapshot then carries no ``kv_blocks_*``
keys, because the decode process owns the blocks.
"""

# 【译】统一的引擎统计。
#
#   包含 EngineStats，它合并了以前的 SpecStats 和 CacheStats，并在它们旁边加了一个吞吐部分。放在 scheduler.py 之外，
#   因为它是引擎级而不是调度器级的：每个调度器拥有一个，engine_utility 读取它拿到的那个调度器上的这一个，
#   来回答 /debug/mtp_stats、/debug/cache_stats 和 /metrics。
#
#   两个P/D进程都会替换掉 EngineCore.__init__ 构造handler时用的那个调度器，所以各自把 utility_handler.scheduler
#   重新接到自己真正运行的那个上；这样prefill一侧的快照就不带 kv_blocks_* 键，因为block归decode进程所有。

import logging
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("atom")


# 【导读】EngineStats：投机解码接受率、前缀缓存命中率、吞吐三类统计，及其周期性日志
# 调用方：Scheduler.__init__创建（scheduler.py，每个调度器一个，存为scheduler.engine_stats）；调度器在postprocess、
#   prefill准入时调用update_spec / update_cache / update_throughput；engine_utility.py读取它回答统计查询和推送指标
# 与主干的关系：只做统计和打日志，不影响调度；缓存命中各个计数器的含义属于第4批（BlockManager）、投机解码属于第7批
# 阅读路线（本文件约1000行，三部分互不依赖）：__init__（三组计数器）→ spec部分（update_spec / log_spec）
#   → cache部分（update_cache、五个命中率属性和_log_*，最长）→ throughput部分（maybe_log_throughput）
# 前置①投机解码（spec部分；完整讲解见arg_utils.py EngineArgs.method上方的"【科普】投机解码"）：草稿模型每步先猜mtp_k个token，
#   主模型一次forward同时验证，从第1个起连续猜对的被接受，再加上主模型自己给出的1个token，所以一步产出1~mtp_k+1个token。
#   update_spec收到的num_accepted_tokens含主模型那1个，num_bonus = 它 − 1 = 被接受的草稿数k，distribution[k]记这种步数。
#   例：mtp_k=3、100步共接受180个草稿 → 草稿总数300，接受率 = 180/300 = 60%，Average toks/fwd = 1 + 180/100 = 2.8
# 前置②前缀缓存的五个计数器（cache部分；前缀缓存与hash链见scheduler.py文件开头【前置③】，状态checkpoint见【前置⑤】）：
#   prompt按hash block沿hash链逐块查缓存，能复用的前缀一层层收窄，满足 cached <= wanted <= compressed <= reusable <= full：
#     full        prompt的全部token
#     reusable    去掉最后一个hash block（prefill至少要算一块才有logits可采样，它永远不复用）——所有命中率的分母
#     compressed  分页KV池里实际还在的那段前缀（状态门控之前；名字来自DeepSeek-V4的压缩KV池，普通模型就是分页KV的命中）
#     wanted      假如每个边界都有状态checkpoint，状态门控会放行到哪里
#     cached      真正被接受、prefill时跳过的前缀
#   例：1000个token、hash block=16 → full = 1000，reusable = (⌈1000/16⌉−1)×16 = 62块 = 992；池里还剩前50块 → compressed = 800；
#     递归状态只在第32块有checkpoint → cached = 512；假如每块都有checkpoint，SWA窗口仍完整的最远边界在第48块 → wanted = 768。
#     于是Hit = 512/992 = 51.6%，paged-hit = 800/992 = 80.6%，state-hit = 512/800 = 64%（80.6% × 64% = 51.6%），
#     Lost-to-checkpoint = 256/992 = 25.8%，Lost-unrecoverable = 32/992 = 3.2%，binding = state
#   没有状态缓存（SWA、递归状态等）的普通MHA/GQA模型没有状态门控：cached = wanted = compressed，有命中时state-hit = 100%


# 三部分各自开关、各自按自己的节奏打日志的统计对象
class EngineStats:
    """Unified engine statistics: the former ``SpecStats`` and ``CacheStats``
    folded together, with a throughput section added beside them.

    Three independent sections, each gated by its own enable flag and each
    emitting at its **own pace**:

    - **spec** — speculative-decoding acceptance. Logs every
      ``spec_log_interval * mtp_k`` draft tokens (count-based).
    - **cache** — prefix-caching hit statistics. Logs every
      ``cache_log_interval`` requests (count-based).
    - **throughput** — engine-status summary line (running/waiting requests,
      KV usage, prefix-cache hit rate, prompt/generation throughput),
      mirroring vLLM's ``LoggingStatLogger``. Logs every
      ``throughput_log_interval_s`` wall-clock seconds (time-based)::

          Engine 000: Avg prompt throughput: 254.4 tokens/s, Avg generation
          throughput: 0.3 tokens/s, Running: 6 reqs, Waiting: 0 reqs, GPU KV
          cache usage: 0.0%, Prefix cache hit rate: 0.0%

    A disabled section's ``update_*`` / ``maybe_log_*`` entry points are
    no-ops, so callers need not guard on the feature flag.
    """

    # 【译】统一的引擎统计：以前的 SpecStats 和 CacheStats 合并在一起，旁边加了一个吞吐部分。
    #
    #   三个相互独立的部分，各自由自己的开关控制，各自按**自己的节奏**输出：
    #
    #   - spec —— 投机解码接受情况。每 spec_log_interval * mtp_k 个草稿token打一次日志（按计数）。
    #   - cache —— 前缀缓存命中统计。每 cache_log_interval 个请求打一次日志（按计数）。
    #   - throughput —— 引擎状态摘要行（运行中/等待中的请求、KV使用率、前缀缓存命中率、prompt/生成吞吐），仿照vLLM的
    #     LoggingStatLogger。每 throughput_log_interval_s 墙钟秒打一次日志（按时间）：
    #
    #         Engine 000: Avg prompt throughput: 254.4 tokens/s, Avg generation
    #         throughput: 0.3 tokens/s, Running: 6 reqs, Waiting: 0 reqs, GPU KV
    #         cache usage: 0.0%, Prefix cache hit rate: 0.0%
    #
    #   被关闭的部分，其 update_* / maybe_log_* 入口都是空操作，所以调用方不需要先检查功能开关。

    # __slots__：固定实例属性集合，不建__dict__（省内存，拼错属性名会直接报错）
    __slots__ = (
        "_cache_interval_cached_tokens",
        "_cache_interval_compressed_tokens",
        "_cache_interval_evicted_base",
        "_cache_interval_full_tokens",
        "_cache_interval_offload_tokens",
        "_cache_interval_requests",
        "_cache_interval_reusable_tokens",
        "_cache_interval_wanted_tokens",
        "_cache_log_interval",
        "_pool_pressure",
        "_recent_cached_tokens",
        "_recent_hits",
        "_recent_reusable_tokens",
        "_recent_window",
        "_spec_interval_distribution",
        "_spec_interval_draft_tokens",
        "_spec_log_interval",
        "_throughput_last_log_time",
        "block_manager",
        "cache_enabled",
        "distribution",
        "engine_index",
        "label",
        "mtp_k",
        "num_generation_tokens",
        "num_prompt_tokens",
        "spec_enabled",
        "throughput_enabled",
        "throughput_log_interval_s",
        "total_cached_tokens",
        "total_compressed_tokens",
        "total_draft_tokens",
        "total_full_tokens",
        "total_offload_tokens",
        "total_requests",
        "total_reusable_tokens",
        "total_wanted_tokens",
    )

    def __init__(
        self,
        *,
        engine_index: int = 0,
        label: str = "",
        use_spec: bool = False,
        mtp_k: int = 0,
        enable_prefix_caching: bool = False,
        enable_log_stats: bool = False,
        spec_log_interval: int = 1000,
        cache_log_interval: int = 100,
        cache_hit_rate_window: int = 1000,
        pool_pressure: Callable[[], dict[str, int]] | None = None,
        throughput_log_interval_s: float = 10.0,
    ):
        self.spec_enabled = use_spec
        self.cache_enabled = enable_prefix_caching
        self.throughput_enabled = enable_log_stats

        # ── spec section ─────────────────────────────────────────────────
        # 【译】── spec部分 ──
        self.mtp_k = mtp_k
        # Log every spec_log_interval decode steps (in terms of draft tokens)
        # 【译】每 spec_log_interval 个decode step打一次日志（以草稿token数计）
        self._spec_log_interval = spec_log_interval * mtp_k
        self.total_draft_tokens: int = 0
        self.distribution: dict[int, int] = {k: 0 for k in range(mtp_k + 1)}
        # Per-interval tracking
        # 【译】按区间的跟踪
        self._spec_interval_draft_tokens: int = 0
        self._spec_interval_distribution: dict[int, int] = {
            k: 0 for k in range(mtp_k + 1)
        }

        # ── cache section ────────────────────────────────────────────────
        # 【译】── cache部分 ──
        self._cache_log_interval = cache_log_interval
        # Read at log time rather than passed per update: the free-list scan
        # behind it is O(free blocks), which is ~10k here and would be paid
        # once per request for a line printed once per `cache_log_interval`.
        # 【译】在打日志时读取，而不是每次更新时传入：它背后的空闲链表扫描是O(空闲block数)，这里约1万，若每个请求付一次，
        #   换来的却是每 cache_log_interval 才打印一次的一行日志。
        self._pool_pressure = pool_pressure
        self.total_requests: int = 0
        self.total_cached_tokens: int = 0
        self.total_full_tokens: int = 0
        # The reuse ceiling, and the only honest denominator for a hit rate.
        #
        # `full` is not reachable: `BlockManager.can_allocate` matches over
        # `range(n_hash_blocks - 1)`, because prefill must forward at least one
        # block to produce sampler logits, so a request's own trailing block
        # never comes from cache. Dividing by `full` therefore charges both
        # pools for a block neither was ever offered, and reports a ceiling of
        # 100% that no run can reach.
        #
        # It also silently rescales with sequence length -- the unreachable
        # block is a fixed `hash_block_size`, so it is ~13% of a 1k prompt and
        # ~0.05% of a 275k one. A hit rate divided by `full` thus moves with
        # the length mix even when both pools behave identically, which is
        # exactly the confound that makes two runs incomparable.
        # 【译】复用的上限，也是命中率唯一诚实的分母。
        #
        #   full 是达不到的：BlockManager.can_allocate 在 range(n_hash_blocks - 1) 上做匹配，因为prefill必须至少forward一个block
        #   才能产出采样器的logits，所以一个请求自己的最后一个block永远不会来自缓存。因此除以 full 会让两个池都为一个从未提供给它们的
        #   block背锅，并报告一个没有任何一次运行能达到的100%上限。
        #
        #   它还会随序列长度悄悄地改变比例——那个达不到的block是固定的 hash_block_size，所以它占1k prompt的约13%，
        #   占275k prompt的约0.05%。因此一个除以 full 的命中率，即使两个池表现完全相同，也会随长度分布而移动，
        #   这正是让两次运行无法比较的那个混杂因素。
        # 解释："1k prompt的约13%"是按hash_block_size=128算的（128/1000）；默认block_size=16、不开DCP时是16/1000 = 1.6%
        self.total_reusable_tokens: int = 0
        # Reuse the lmcache CPU offload tier served. Sits above the HBM walk
        # rather than inside the `cached <= ... <= reusable` chain (see
        # `update_cache`); shares only the `reusable` denominator with it.
        # 【译】由lmcache CPU offload层提供的复用。它位于HBM遍历之上，而不在 cached <= ... <= reusable 链之内（见 update_cache）；
        #   只与它共用 reusable 这个分母。
        self.total_offload_tokens: int = 0
        # Pre-gate compressed-prefix hit tokens, and the boundary between the
        # two pools: everything below it is the paged pool's doing, everything
        # between it and `cached` is the state gates'. `reusable - compressed`
        # is reuse the paged pool could not offer; `compressed - cached` is
        # reuse it offered and the Pool.STATE gates declined.
        # 【译】门控之前的压缩前缀命中token数，也是两个池之间的分界：它以下的一切是分页池的作为，它与 cached 之间的一切是状态门控的作为。
        #   reusable - compressed 是分页池没能提供的复用；compressed - cached 是它提供了、却被 Pool.STATE 门控拒绝的复用。
        self.total_compressed_tokens: int = 0
        # Where the gates would have landed with every state ladder dense. It
        # sits between cached and compressed and splits the declined reuse in
        # two: below it a checkpoint was missing, above it nothing would have
        # helped. Without the split "declined" is one number and whether
        # demand-driven checkpointing applies to a workload is unfalsifiable.
        # 【译】假如每条状态阶梯都是稠密的，门控会落在哪里。它位于 cached 和 compressed 之间，把被拒绝的复用一分为二：
        #   它以下是缺了一个checkpoint，它以上是什么都帮不上。没有这个拆分，"被拒绝"就只是一个数，
        #   按需checkpoint是否适用于某个负载就无法证伪。
        self.total_wanted_tokens: int = 0
        self._cache_interval_requests: int = 0
        self._cache_interval_cached_tokens: int = 0
        self._cache_interval_full_tokens: int = 0
        self._cache_interval_compressed_tokens: int = 0
        self._cache_interval_wanted_tokens: int = 0
        self._cache_interval_reusable_tokens: int = 0
        self._cache_interval_offload_tokens: int = 0
        # Sliding window behind `recent_cache_hit_rate`, the figure the
        # engine-status line reports. Mirrors vLLM's PrefixCachingMetrics: the
        # last N requests, aggregated incrementally so `observe` stays O(1).
        # 【译】recent_cache_hit_rate 背后的滑动窗口，也就是引擎状态行报告的那个数。仿照vLLM的PrefixCachingMetrics：
        #   最近N个请求，增量聚合，使 observe 保持O(1)。
        if cache_hit_rate_window <= 0:
            raise ValueError(
                f"cache_hit_rate_window must be > 0, got {cache_hit_rate_window}"
            )
        self._recent_window = cache_hit_rate_window
        self._recent_hits: deque[tuple[int, int]] = deque()
        self._recent_reusable_tokens: int = 0
        self._recent_cached_tokens: int = 0
        # Set by Scheduler for pool occupancy logging.
        # 【译】由Scheduler设置，用于打印池的占用情况。
        self.block_manager = None
        self._cache_interval_evicted_base: int = 0

        # ── throughput section ───────────────────────────────────────────
        # A real exception, not an assert: `python -O` strips asserts, and the
        # only thing between a non-positive interval and the division in
        # `maybe_log_throughput` is this check. Stripped, the engine would die
        # of ZeroDivisionError inside the scheduler loop.
        # 【译】── throughput部分 ──
        #   用真正的异常而不是assert：python -O 会去掉assert，而在一个非正的间隔和 maybe_log_throughput 里的除法之间，
        #   唯一挡着的就是这个检查。去掉它，引擎就会在调度循环里死于ZeroDivisionError。
        if throughput_log_interval_s <= 0:
            raise ValueError(
                "throughput_log_interval_s must be > 0, got "
                f"{throughput_log_interval_s}"
            )
        self.engine_index = engine_index
        # Distinguishes the P/D processes, which both run as engine index 0 and
        # usually log to the same place. "" for the aggregated engine.
        # 【译】区分P/D两个进程，它们都以engine index 0运行，而且通常把日志写到同一个地方。聚合引擎为 ""。
        self.label = label
        self.throughput_log_interval_s = throughput_log_interval_s
        self._throughput_last_log_time = time.monotonic()
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0

    # ══ spec section ═════════════════════════════════════════════════════
    # 【译】══ spec部分 ══

    # 记录一条序列在一个decode step里的接受结果（num_accepted_tokens含目标模型自己的那个token，所以bonus = 接受数 - 1）
    def update_spec(self, num_accepted_tokens: int) -> None:
        """Record acceptance result for one sequence in one decode step."""
        # 【译】记录一条序列在一个decode step里的接受结果。
        if not self.spec_enabled:
            return
        self.total_draft_tokens += self.mtp_k
        self._spec_interval_draft_tokens += self.mtp_k
        num_bonus = num_accepted_tokens - 1
        self.distribution[num_bonus] += 1
        self._spec_interval_distribution[num_bonus] += 1

        # 例：mtp_k=3、spec_log_interval=1000 → 每累计3000个草稿token打印一次
        if self.total_draft_tokens % self._spec_log_interval == 0:
            self.log_spec()
            self._reset_spec_interval()

    # 累计被接受的草稿token数 = sum(k x 步数)
    @property
    def total_accepted(self) -> int:
        """Total number of accepted bonus tokens across all steps."""
        # 【译】所有step中被接受的bonus token总数。
        return sum(k * v for k, v in self.distribution.items())

    # 记录过的decode step总数
    @property
    def total_steps(self) -> int:
        """Total number of decode steps recorded."""
        # 【译】记录过的decode step总数。
        return sum(self.distribution.values())

    # 接受率 = 被接受的草稿token / 草稿token总数
    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_draft_tokens

    # 汇总dict，供engine_utility回答get_mtp_statistics和推送指标
    def spec_statistics(self) -> dict:
        """Return a summary dict compatible with engine_core reporting."""
        # 【译】返回一个与engine_core上报格式兼容的汇总dict。
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "distribution": dict(self.distribution),
        }

    # 清零投机解码统计
    def reset_spec(self) -> None:
        self.total_draft_tokens = 0
        self.distribution = {k: 0 for k in range(self.mtp_k + 1)}
        self._reset_spec_interval()

    def _reset_spec_interval(self) -> None:
        self._spec_interval_draft_tokens = 0
        self._spec_interval_distribution = {k: 0 for k in range(self.mtp_k + 1)}

    # 打两行日志：本区间的、以及累计的（平均每次forward产出的token数、接受率、接受数分布）
    def log_spec(self) -> None:
        ts = self.total_steps
        if ts == 0:
            return
        # Interval stats
        # 【译】区间统计
        iv_steps = sum(self._spec_interval_distribution.values())
        if iv_steps == 0:
            self._reset_spec_interval()
            return
        iv_accepted = sum(k * v for k, v in self._spec_interval_distribution.items())
        iv_rate = (
            iv_accepted / self._spec_interval_draft_tokens
            if self._spec_interval_draft_tokens > 0
            else 0.0
        )
        logger.info(
            f"[MTP Stats Interval] Average toks/fwd: {1 + iv_accepted / iv_steps:.2f}, "
            f"Accepted/Total Draft tokens: {iv_accepted}/{self._spec_interval_draft_tokens}, "
            f"Acceptance rate: {iv_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / iv_steps:.2%}' for k, v in self._spec_interval_distribution.items()} }"
        )
        logger.info(
            f"[MTP Stats         ] Average toks/fwd: {1 + self.total_accepted / ts:.2f}, "
            f"Accepted/Total Draft tokens: {self.total_accepted}/{self.total_draft_tokens}, "
            f"Acceptance rate: {self.acceptance_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / ts:.2%}' for k, v in self.distribution.items()} }"
        )

    # ══ cache section ════════════════════════════════════════════════════
    # 【译】══ cache部分 ══

    # 记录一个prefill请求的缓存统计（调度器在准入prefill时调用；见第3、4批）
    def update_cache(
        self,
        num_cached_tokens: int,
        num_full_tokens: int,
        num_compressed_tokens: int,
        num_wanted_tokens: int,
        num_reusable_tokens: int,
        num_offload_tokens: int = 0,
    ) -> None:
        """Record cache stats for one prefill sequence.

        The first five are required because the reported rates are differences
        between them: `cached <= wanted <= compressed <= reusable <= full`. A
        defaulted argument would silently report a negative rate rather than a
        missing one.

        `reusable` is the caller's, not this class's, because the gap between
        it and `full` is a `BlockManager` matching detail (the trailing block
        has no stable hash, so it is never a reuse candidate). Recomputing it
        here would mean duplicating that rule in a second place and letting the
        two drift.

        `num_offload_tokens` is the exception, and defaults because it is not
        part of that chain: it is reuse the CPU tier served, which sits above
        the HBM walk rather than inside it. Counting it in `cached` is what
        would break the nesting -- see `_schedule_prefill_seq`.
        """
        # 【译】记录一条prefill序列的缓存统计。
        #
        #   前五个参数是必需的，因为报告的比率是它们之间的差：cached <= wanted <= compressed <= reusable <= full。
        #   一个带默认值的参数会悄悄报告一个负的比率，而不是一个缺失的比率。
        #
        #   reusable 由调用方给出，而不是由本类计算，因为它和 full 之间的差距是 BlockManager 的匹配细节（最后一个block
        #   没有稳定的hash，所以永远不是复用候选）。在这里重新计算意味着在第二个地方重复这条规则，并任由两者漂移。
        #
        #   num_offload_tokens 是例外，它有默认值，因为它不属于那条链：它是CPU层提供的复用，位于HBM遍历之上而不在其中。
        #   把它计入 cached 正是会破坏嵌套关系的做法——见 _schedule_prefill_seq。
        if not self.cache_enabled:
            return
        ordered = (
            num_cached_tokens
            <= num_wanted_tokens
            <= num_compressed_tokens
            <= num_reusable_tokens
            <= num_full_tokens
        )
        if not ordered:
            # Warned and clamped, not asserted. These are logging counters, and
            # `num_cached_tokens` has four independent writers -- the CPU-offload
            # wake at `_wake_offloaded_seq` sets it without touching the two
            # hit-block counters `can_allocate` derives the rest from, so an
            # LMCache resume that loads more prefix than the GPU index held
            # produces `cached > wanted` legitimately. Aborting `schedule()`
            # over it would take the engine down to protect a log line, and
            # would do so only in builds without `-O`, so the two would differ
            # in behaviour. Clamping keeps the rates monotone and the run alive.
            # 【译】告警并钳位，而不是断言。这些只是日志计数器，而 num_cached_tokens 有四个相互独立的写入者——在 _wake_offloaded_seq
            #   处的CPU offload唤醒会设置它，却不碰 can_allocate 推导其余各项所依据的那两个命中block计数器，所以一次加载的前缀
            #   多于GPU索引所持有的LMCache恢复，会合理地产生 cached > wanted。为此中止 schedule() 会为了保护一行日志而让引擎挂掉，
            #   而且只在不带 -O 的构建里这样做，于是两者的行为会不同。钳位让比率保持单调，也让运行继续。
            logger.warning(
                "Cache stats ordering violated, clamping: cached=%d wanted=%d "
                "compressed=%d reusable=%d full=%d",
                num_cached_tokens,
                num_wanted_tokens,
                num_compressed_tokens,
                num_reusable_tokens,
                num_full_tokens,
            )
            num_full_tokens = max(num_full_tokens, 0)
            num_reusable_tokens = min(max(num_reusable_tokens, 0), num_full_tokens)
            num_compressed_tokens = min(
                max(num_compressed_tokens, 0), num_reusable_tokens
            )
            num_wanted_tokens = min(max(num_wanted_tokens, 0), num_compressed_tokens)
            num_cached_tokens = min(max(num_cached_tokens, 0), num_wanted_tokens)
        self.total_requests += 1
        self.total_cached_tokens += num_cached_tokens
        self.total_full_tokens += num_full_tokens
        self.total_reusable_tokens += num_reusable_tokens
        self.total_compressed_tokens += num_compressed_tokens
        self.total_wanted_tokens += num_wanted_tokens
        self._cache_interval_requests += 1
        self._cache_interval_cached_tokens += num_cached_tokens
        self._cache_interval_full_tokens += num_full_tokens
        self._cache_interval_reusable_tokens += num_reusable_tokens
        self._cache_interval_compressed_tokens += num_compressed_tokens
        self._cache_interval_wanted_tokens += num_wanted_tokens
        # `num_offload_tokens` is outside the `cached <= ... <= reusable` chain,
        # so the ordering clamp above never bounded it -- but the rates pair it
        # with `cached` against the same `reusable` denominator (`cache_hit_rate`
        # and `lmcache_hit_rate` are meant to partition served reuse), and the
        # documented invariant is `cached + offload <= reusable`. A CPU-tier
        # resume can report more offload reuse than the HBM walk left room for;
        # clamp so the combined rate cannot exceed 100%.
        # 【译】num_offload_tokens 在 cached <= ... <= reusable 链之外，所以上面的排序钳位从未约束它——但比率把它与 cached 配对、
        #   用同一个 reusable 分母（cache_hit_rate 和 lmcache_hit_rate 的本意是划分被提供的复用），而文档写明的不变量是
        #   cached + offload <= reusable。一次CPU层的恢复可能报告比HBM遍历留出的空间更多的offload复用；钳位使合计比率不会超过100%。
        num_offload_tokens = max(
            0, min(num_offload_tokens, num_reusable_tokens - num_cached_tokens)
        )
        self.total_offload_tokens += num_offload_tokens
        self._cache_interval_offload_tokens += num_offload_tokens

        # Slide the recent-requests window. One call is one request here, so
        # the deque length is the request count directly. Denominated in
        # `reusable`, like every rate below it — see `total_reusable_tokens`.
        # 【译】滑动最近请求的窗口。这里一次调用就是一个请求，所以deque的长度就直接是请求数。分母是 reusable，
        #   和下面的每个比率一样——见 total_reusable_tokens。
        self._recent_hits.append((num_reusable_tokens, num_cached_tokens))
        self._recent_reusable_tokens += num_reusable_tokens
        self._recent_cached_tokens += num_cached_tokens
        if len(self._recent_hits) > self._recent_window:
            old_reusable, old_cached = self._recent_hits.popleft()
            self._recent_reusable_tokens -= old_reusable
            self._recent_cached_tokens -= old_cached

        if self.total_requests % self._cache_log_interval == 0:
            self._log_cache()
            self._reset_cache_interval()

    # 端到端命中率（累计）= cached / reusable
    @property
    def cache_hit_rate(self) -> float:
        """End-to-end reuse, against what was reusable at all.

        Lifetime figure: feeds `/debug/cache_stats`, `/metrics` and the
        cumulative row of the `[Cache Stats]` line, all cumulative by design.

        Not against `total_full_tokens`: that denominator includes each
        request's trailing block, which no cache is ever allowed to serve, so
        it reports a ceiling nothing can reach and drifts with the prompt
        length mix. `paged_hit_rate * state_hit_rate == hit_rate` exactly.
        """
        # 【译】端到端复用，相对于"一开始就可复用的量"。
        #
        #   累计数字：供 /debug/cache_stats、/metrics 以及 [Cache Stats] 行的累计那一行使用，它们按设计都是累计的。
        #
        #   不相对于 total_full_tokens：那个分母包含每个请求的最后一个block，任何缓存都不被允许提供它，所以它报告的上限
        #   永远达不到，还会随prompt长度分布漂移。paged_hit_rate * state_hit_rate == hit_rate，严格相等。
        return self._rate(self.total_cached_tokens, self.total_reusable_tokens)

    # 分页KV池自己的命中率 = compressed / reusable
    @property
    def paged_hit_rate(self) -> float:
        """The paged KV pool's own hit rate, with the state cache factored out.

        Denominator is what the paged pool was asked for (`reusable`);
        numerator is what it had (`compressed`). The state gates run strictly
        after this and cannot change either term, so this number is unaffected
        by checkpoint policy -- change `--state-checkpoint-*` and this should
        not move. It answers "is the prefix still in KV?" and nothing else.

        What it charges the pool for: eviction, capacity, and genuinely novel
        prefixes. That last one is a workload property, not a defect, so this
        rate has a ceiling below 100% set by how much of the traffic is new
        text -- compare it against the dataset's theoretical prefix hit, not
        against 100%.
        """
        # 【译】分页KV池自己的命中率，把状态缓存的因素排除掉。
        #
        #   分母是向分页池请求的量（reusable）；分子是它持有的量（compressed）。状态门控严格在它之后运行，改变不了任何一项，
        #   所以这个数不受checkpoint策略影响——改 --state-checkpoint-* 时它不应该动。它回答"前缀还在KV里吗？"，别的什么也不回答。
        #
        #   它为哪些事情扣分：淘汰、容量，以及真正全新的前缀。最后一项是负载的属性而不是缺陷，所以这个比率有一个低于100%的上限，
        #   由流量中有多少是新文本决定——要拿它和数据集的理论前缀命中率比，而不是和100%比。
        return self._rate(self.total_compressed_tokens, self.total_reusable_tokens)

    # 状态缓存自己的命中率 = cached / compressed
    @property
    def state_hit_rate(self) -> float:
        """The state cache's own hit rate, with the paged pool factored out.

        Denominator is what the paged pool actually offered (`compressed`),
        NOT `reusable` -- the state gates never see a prefix the paged pool
        already lost, and charging them for it would mean a KV eviction shows
        up as a state-cache failure and sends tuning at the wrong pool. This
        is the conditional probability: given the prefix was there, did a
        checkpoint let us resume from it?

        Unlike `paged_hit_rate`, 100% is genuinely reachable here: it means
        every boundary the paged pool offered had a resumable checkpoint. The
        gap decomposes into `state_recoverable_loss_rate` (a checkpoint would
        have fixed it) and the remainder (nothing would have).
        """
        # 【译】状态缓存自己的命中率，把分页池的因素排除掉。
        #
        #   分母是分页池实际提供的量（compressed），而**不是** reusable——状态门控永远看不到分页池已经丢掉的前缀，为此给它们扣分
        #   就意味着一次KV淘汰会表现为状态缓存的失败，把调优引向错误的池。这是条件概率：在前缀还在的前提下，有没有一个checkpoint
        #   让我们能从那里恢复？
        #
        #   与 paged_hit_rate 不同，这里100%是真正可以达到的：它意味着分页池提供的每个边界都有一个可恢复的checkpoint。
        #   差距分解为 state_recoverable_loss_rate（一个checkpoint本可以修好）和剩余部分（什么都修不好）。
        return self._rate(self.total_cached_tokens, self.total_compressed_tokens)

    # 状态缓存的未命中中、本可由checkpoint挽回的部分 = (wanted - cached) / compressed
    @property
    def state_recoverable_loss_rate(self) -> float:
        """The part of the state cache's miss that checkpoint placement owns.

        Same denominator as `state_hit_rate`, so the two compose:
        `state_hit_rate + state_recoverable_loss_rate` is the rate the state
        cache would reach with a dense ladder. The distance from that to 1.0
        is the part no checkpoint can buy, and so the honest cap on what any
        amount of checkpoint capacity is worth.
        """
        # 【译】状态缓存的未命中中，归checkpoint放置负责的那一部分。
        #
        #   与 state_hit_rate 分母相同，所以两者可以组合：state_hit_rate + state_recoverable_loss_rate 是状态缓存在稠密阶梯下
        #   能达到的比率。它与1.0的距离是任何checkpoint都买不到的部分，因此也是任意多的checkpoint容量所值的诚实上限。
        return self._rate(
            self.total_wanted_tokens - self.total_cached_tokens,
            self.total_compressed_tokens,
        )

    # lmcache（CPU offload层）提供的复用比例 = offload / reusable
    @property
    def lmcache_hit_rate(self) -> float:
        """Reuse the lmcache CPU offload tier served, as a share of all reuse.

        Disjoint from the HBM walk on purpose: `total_offload_tokens` counts
        reuse claimed *above* what the walk could offer (see `update_cache`), so
        it is exactly the reuse that becomes recompute the moment the tier is
        switched off -- the whole case for the tier on a given workload, in one
        number.

        **Pairs with `cache_hit_rate`, not with `paged_hit_rate`.** Both of
        those read as "the HBM number" and only `cache_hit_rate` is a served
        quantity: `paged_hit_rate`'s numerator is `compressed`, how far the
        prefix walk reached *before* the state gates cut it. `cached + offload
        <= reusable` by construction (the clamp in `update_cache` enforces it),
        so `cache_hit_rate` and this partition the served reuse; `compressed +
        offload` does not and can
        exceed the denominator.
        """
        # 【译】lmcache CPU offload层提供的复用，作为全部复用的一个份额。
        #
        #   有意与HBM遍历不相交：total_offload_tokens 统计的是在遍历所能提供的*之上*被认领的复用（见 update_cache），
        #   所以它恰好就是一关掉这一层就会变成重算的复用——这一层对给定负载的全部价值，用一个数表达。
        #
        #   **与 cache_hit_rate 配对，而不是与 paged_hit_rate 配对。** 两者读起来都像"HBM的那个数"，但只有 cache_hit_rate
        #   是一个被提供的量：paged_hit_rate 的分子是 compressed，即前缀遍历在状态门控截断它*之前*走到了多远。
        #   按构造 cached + offload <= reusable（update_cache 里的钳位保证它），所以 cache_hit_rate 和这个比率划分了被提供的复用；
        #   compressed + offload 则不是，而且可能超过分母。
        return self._rate(self.total_offload_tokens, self.total_reusable_tokens)

    # 最近cache_hit_rate_window个请求上的命中率（滑动窗口），引擎状态行用它；窗口为空时返回None
    @property
    def recent_cache_hit_rate(self) -> float | None:
        """`cache_hit_rate` over the last `cache_hit_rate_window` requests.

        What the engine-status line reports, because every other field on that
        line is windowed: a lifetime figure there stops tracking the workload,
        and on a long run it is dominated by traffic hours old. vLLM's
        `LoggingStatLogger` uses a 1000-request window for the same reason.
        Same `reusable` denominator as the lifetime rate, so the two are
        comparable.

        None when the window holds nothing, which the line renders `n/a`
        rather than as a measured 0%.
        """
        # 【译】最近 cache_hit_rate_window 个请求上的 cache_hit_rate。
        #
        #   引擎状态行报告的就是它，因为那一行上的其他每个字段都是按窗口的：一个累计数字放在那里会不再跟踪当前负载，在长时间运行中
        #   会被几个小时前的流量主导。vLLM的 LoggingStatLogger 出于同样的原因使用1000个请求的窗口。分母与累计比率同为 reusable，
        #   所以两者可以比较。
        #
        #   窗口里什么都没有时返回None，状态行把它显示为 n/a，而不是一个测出来的0%。
        if not self._recent_reusable_tokens:
            return None
        return self._recent_cached_tokens / self._recent_reusable_tokens

    # 返回原始计数器（不是比率），供跨DP rank求和（LLMEngine.get_cache_statistics再算比率）
    def cache_statistics(self) -> dict:
        """Counters, not rates — the caller derives those.

        Every rate this class reports is a ratio of two of these totals, and a
        rate cannot be aggregated across DP ranks that saw different token
        counts. Handing back the counts keeps the merge a sum.
        """
        # 【译】计数器，不是比率——比率由调用方推导。
        #
        #   本类报告的每个比率都是这些总数中两个的比值，而一个比率无法在看到不同token数的DP rank之间聚合。交还计数，
        #   合并就只是求和。
        return {
            "requests": self.total_requests,
            "cached_tokens": self.total_cached_tokens,
            "compressed_tokens": self.total_compressed_tokens,
            "wanted_tokens": self.total_wanted_tokens,
            # The denominator for every rate here. `full_tokens` is reported
            # too, but only so a consumer can see the unreachable gap; it is
            # not a hit-rate denominator -- see `total_reusable_tokens`.
            # 【译】这里每个比率的分母。full_tokens 也会报告，但只是为了让使用方看到那段达不到的差距；它不是命中率的分母——
            #   见 total_reusable_tokens。
            "reusable_tokens": self.total_reusable_tokens,
            "full_tokens": self.total_full_tokens,
            # Reuse from the CPU tier. Separate on purpose: it shares no
            # denominator with the HBM series above.
            # 【译】来自CPU层的复用。有意分开：它与上面的HBM系列不共用分母。
            "offload_tokens": self.total_offload_tokens,
        }

    def _reset_cache_interval(self) -> None:
        self._cache_interval_requests = 0
        self._cache_interval_cached_tokens = 0
        self._cache_interval_full_tokens = 0
        self._cache_interval_compressed_tokens = 0
        self._cache_interval_wanted_tokens = 0
        self._cache_interval_reusable_tokens = 0
        self._cache_interval_offload_tokens = 0

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    def _log_cache(self) -> None:
        # compressed = pre-gate prefix hit; cached = post-gate (admitted); the
        # two differ by what the Pool.STATE gates declined, and `wanted` splits
        # that difference where it matters:
        #   Lost-to-checkpoint  wanted - cached, reuse a checkpoint at that
        #                       boundary would have delivered. What the demand
        #                       rung goes after; expected to fall toward 0 on a
        #                       workload with a genuinely shared prefix.
        #   Lost-unrecoverable  compressed - wanted, declined for a reason no
        #                       checkpoint touches: the SWA tail is gone, or the
        #                       boundary is too near the prompt's end to fork.
        # (reusable - compressed) is the rest: compressed eviction, or no reuse.
        # 【译】compressed = 门控之前的前缀命中；cached = 门控之后（被接受的）；两者相差的是被 Pool.STATE 门控拒绝的部分，
        #   而 wanted 在关键处把这个差拆开：
        #     Lost-to-checkpoint  wanted - cached，在那个边界上有一个checkpoint本可以交付的复用。这是按需那一级所追求的；
        #                         在一个真正有共享前缀的负载上，预期会下降到接近0。
        #     Lost-unrecoverable  compressed - wanted，因任何checkpoint都影响不到的原因被拒绝：SWA的尾部已经没了，
        #                         或者边界离prompt末尾太近、无法分叉。
        #   （reusable - compressed）是其余部分：压缩后的淘汰，或者根本没有复用。
        self._cache_log_line(
            "Interval",
            self._cache_interval_requests,
            self._cache_interval_cached_tokens,
            self._cache_interval_compressed_tokens,
            self._cache_interval_wanted_tokens,
            self._cache_interval_reusable_tokens,
            self._cache_interval_full_tokens,
        )
        self._cache_log_line(
            "        ",
            self.total_requests,
            self.total_cached_tokens,
            self.total_compressed_tokens,
            self.total_wanted_tokens,
            self.total_reusable_tokens,
            self.total_full_tokens,
        )
        if self.block_manager is not None:
            occ = self.block_manager.pool_occupancy()
            evicted_iv = occ["evicted_total"] - self._cache_interval_evicted_base
            self._cache_interval_evicted_base = occ["evicted_total"]
            total = occ["total"] or 1
            logger.info(
                f"[Cache Pool          ] "
                f"used {occ['used']} ({occ['used'] / total:.0%}), "
                f"free {occ['free']} ({occ['free'] / total:.0%}), "
                f"retained-cache {occ['retained']}, "
                f"evicted this interval {evicted_iv} "
                f"(total {occ['evicted_total']})"
            )
            # Rejected chains are rehashed on the scheduler thread.
            # 【译】被拒绝的链会在调度线程上重新计算hash。
            bm = self.block_manager
            logger.info(
                f"[Prompt Chains] frontend-hashed: "
                f"adopted {bm.shipped_chains_adopted}, "
                f"rejected {bm.shipped_chains_rejected}"
            )
        self._log_pools()
        if self.total_offload_tokens:
            logger.info(
                "[Cache Stats offload] CPU-tier tokens: "
                f"interval={self._cache_interval_offload_tokens} "
                f"total={self.total_offload_tokens}"
            )
        if self._pool_pressure is not None:
            self._log_pressure(self._pool_pressure())

    def _log_pools(self) -> None:
        """Each pool's hit rate against its own denominator.

        The `[Cache Stats]` line reports one end-to-end rate, which cannot say
        which pool to fix: the same 85% is a KV pool that lost the prefix or a
        state cache that refused to resume from it, and those want opposite
        changes. Splitting needs two denominators, because the pools are in
        series and the second only ever sees what the first passed on:

            paged = compressed / reusable      "was the prefix still in KV?"
            state = cached     / compressed    "given it was, could we resume?"
            paged * state = cached / reusable = the end-to-end rate

        So the product is exact, and the smaller factor is the bottleneck --
        that comparison is the whole point of the line. Reading `state`
        against `reusable` instead would fold KV evictions into the state
        cache's score and point tuning at the wrong pool.

        `+ckpt` is where `state` would land if every ladder were dense. It is
        the ceiling on what checkpoint placement or more slots can buy; if it
        sits near `state`, the state cache is already doing all it can and the
        remaining loss is the paged pool's.
        """
        # 【译】每个池相对于自己分母的命中率。
        #
        #   [Cache Stats] 行报告一个端到端的比率，它说不出该修哪个池：同样的85%可能是一个丢了前缀的KV池，也可能是一个拒绝从它恢复的
        #   状态缓存，而这两者需要相反的改动。拆分需要两个分母，因为这两个池是串联的，第二个只能看到第一个传下来的东西：
        #
        #       paged = compressed / reusable      "前缀还在KV里吗？"
        #       state = cached     / compressed    "既然在，我们能恢复吗？"
        #       paged * state = cached / reusable = 端到端比率
        #
        #   所以乘积是精确的，而较小的那个因子就是瓶颈——这个比较就是这一行的全部意义。若相对 reusable 去读 state，就会把KV淘汰
        #   算进状态缓存的分数，把调优引向错误的池。
        #
        #   +ckpt 是假如每条阶梯都稠密时 state 会落在哪里。它是checkpoint放置或更多槽位所能买到的上限；如果它接近 state，
        #   状态缓存已经尽力了，剩下的损失归分页池。
        # 例：paged=90%、state=80% → combined=72%，binding=state（状态缓存是瓶颈）
        paged = self.paged_hit_rate
        state = self.state_hit_rate
        # Which factor is further from 1.0 loses more reuse, since the rates
        # multiply. Named here rather than left to the reader because the
        # comparison is against each other, not against 100%.
        # 【译】离1.0更远的那个因子损失的复用更多，因为比率是相乘的。在这里点名，而不是留给读者，因为比较是两者之间的，而不是与100%比。
        worse = "paged" if paged <= state else "state"
        logger.info(
            "[Cache Pools] "
            f"paged-hit: {paged:.2%} "
            f"({self.total_compressed_tokens}/{self.total_reusable_tokens}), "
            f"state-hit: {state:.2%} "
            f"({self.total_cached_tokens}/{self.total_compressed_tokens}), "
            f"state-hit+ckpt: {state + self.state_recoverable_loss_rate:.2%}, "
            f"combined: {self.cache_hit_rate:.2%}, "
            f"binding: {worse}"
        )
        # A second split, orthogonal to paged/state above: of all reuse, how much
        # came from HBM (paged pool) vs from the lmcache CPU offload tier. The
        # offload count sits above the HBM walk (see `update_cache`), so it is
        # exactly the reuse that reverts to recompute when the tier is switched
        # off -- i.e. the value of running lmcache on this workload, in one
        # number. Logged only when a tier is attached; the no-connector baseline
        # has total_offload_tokens==0 and its tier split is trivially HBM=all,
        # lmcache=0.
        #
        # The HBM half is `cache_hit_rate` (`cached`), NOT `paged_hit_rate`
        # (`compressed`). Both read as "the HBM number" and only one of them is
        # a *served* quantity: `compressed` is how far the prefix walk reached
        # before the state gates cut it, so it is reach, not reuse anybody got.
        # `cached + offload <= reusable` by construction (clamped in
        # `update_cache`), so these two do partition, and `combined` is exactly
        # the end-to-end rate.
        # 【译】第二种拆分，与上面的 paged/state 正交：在全部复用中，有多少来自HBM（分页池）、多少来自lmcache CPU offload层。
        #   offload计数位于HBM遍历之上（见 update_cache），所以它恰好是一关掉这一层就退回重算的复用——也就是在这个负载上跑lmcache的
        #   价值，用一个数表达。只在接了某个层时才打印；没有connector的基线 total_offload_tokens==0，它的层拆分平凡地是 HBM=全部、lmcache=0。
        #
        #   HBM那一半是 cache_hit_rate（cached），**不是** paged_hit_rate（compressed）。两者读起来都像"HBM的那个数"，但只有一个是
        #   *被提供*的量：compressed 是前缀遍历在状态门控截断它之前走到了多远，所以它是可达范围，不是任何人得到的复用。
        #   按构造 cached + offload <= reusable（在 update_cache 里钳位），所以这两者确实构成划分，combined 恰好就是端到端比率。
        if self.total_offload_tokens:
            served = self.total_cached_tokens + self.total_offload_tokens
            logger.info(
                "[Cache Tiers] "
                f"HBM-served (KV prefix resident): {self.cache_hit_rate:.2%} "
                f"({self.total_cached_tokens}/{self.total_reusable_tokens}), "
                f"lmcache-served (CPU tier): {self.lmcache_hit_rate:.2%} "
                f"({self.total_offload_tokens}/{self.total_reusable_tokens}), "
                f"combined: {self._rate(served, self.total_reusable_tokens):.2%}"
            )

    @staticmethod
    def _log_pressure(p: dict[str, int]) -> None:
        """The two pools' own account of what they destroyed.

        `full - compressed` in the line above is reuse the paged pool did not
        have, but it cannot say why — a prompt with no shared prefix and a
        prefix evicted an hour ago read identically. These counters separate
        them, and are the only evidence that eviction happened at all:
        `blocks_evicted == 0` at the end of a run means every miss above was
        absence of reuse, not loss of it.

        Vacant is called out because it is the leading indicator. Evictions
        can only begin once it reaches 0, so a run that ends with vacant
        blocks to spare never had paged pressure whatever its hit rate.
        """
        # 【译】两个池自己对它们销毁了什么的记录。
        #
        #   上一行里的 full - compressed 是分页池没有的复用，但它说不出原因——一个没有共享前缀的prompt和一个一小时前被淘汰的前缀
        #   读起来一模一样。这些计数器把两者分开，也是淘汰确实发生过的唯一证据：运行结束时 blocks_evicted == 0 意味着上面的每一次
        #   未命中都是复用的缺席，而不是它的丢失。
        #
        #   特别点出vacant，因为它是领先指标。只有它降到0之后淘汰才可能开始，所以一次结束时还有空余vacant block的运行，
        #   无论命中率如何，都从未有过分页压力。
        logger.info(
            "[Pool Pressure] "
            f"paged: {p['blocks_used']}/{p['blocks_total']} used, "
            f"{p['blocks_free_reusable']} reusable-free, "
            f"{p['blocks_free'] - p['blocks_free_reusable']} vacant, "
            f"{p['blocks_indexed']} indexed | "
            f"evicted: {p['blocks_evicted']}, retired: {p['blocks_retired']} | "
            f"state: {p['slots_used']}/{p['slots_total']} used, "
            f"{p['slots_held']} checkpointed, {p['slots_vacant']} vacant"
        )
        # The state pool's own losses, which `blocks_evicted` cannot express:
        # a checkpoint can die without any block dying (`evicted`, the pool ran
        # out of slots) or *because* a block died (`orphaned`, the prefix it
        # was filed under left the KV index first). The pair says which pool to
        # grow — see `StateSlotPool.__init__` for why they are kept apart.
        # 【译】状态池自己的损失，blocks_evicted 无法表达：一个checkpoint可以在没有任何block死亡的情况下死掉（evicted，池里的槽位用完了），
        #   也可以*因为*一个block死了而死掉（orphaned，它所登记的那个前缀先离开了KV索引）。这一对说明该扩大哪个池——
        #   为什么要把它们分开，见 StateSlotPool.__init__。
        logger.info(
            "[Checkpoint Fates] "
            f"kept: {p['checkpoints_kept']}, "
            f"dropped: {p['checkpoints_dropped']}, "
            f"evicted: {p['checkpoints_evicted']}, "
            f"orphaned: {p['checkpoints_orphaned']}"
        )

    @classmethod
    def _cache_log_line(
        cls,
        label: str,
        reqs: int,
        cached: int,
        compressed: int,
        wanted: int,
        reusable: int,
        full: int,
    ) -> None:
        """Every rate here is over `reusable`; `full` is shown, not divided by.

        `Unreachable` is the gap between them -- the trailing block of each
        request, which prefill must always compute. It is reported so the
        older `full`-denominated numbers in past logs remain translatable, and
        because a large value is itself a signal: it means short prompts
        dominate, and a run whose length mix differs this much is not
        comparable to another on hit rate alone.
        """
        # 【译】这里的每个比率都以 reusable 为分母；full 只显示，不做除数。
        #
        #   Unreachable 是两者之间的差距——每个请求的最后一个block，prefill必须总是计算它。报告它是为了让过去日志里那些以 full 为分母的
        #   旧数字仍然可以换算，也因为一个大的值本身就是一个信号：它意味着短prompt占主导，一次长度分布与之差别这么大的运行，
        #   不能只凭命中率与另一次比较。
        logger.info(
            f"[Cache Stats {label}] Reqs: {reqs}, "
            f"Cached/Reusable: {cached}/{reusable}, "
            f"Hit: {cls._rate(cached, reusable):.2%}, "
            f"Compressed-hit: {cls._rate(compressed, reusable):.2%}, "
            f"Lost-to-checkpoint: {cls._rate(wanted - cached, reusable):.2%}, "
            f"Lost-unrecoverable: {cls._rate(compressed - wanted, reusable):.2%}, "
            f"Unreachable: {cls._rate(full - reusable, full):.2%} of {full}"
        )

    # ══ throughput section ═══════════════════════════════════════════════
    # 【译】══ throughput部分 ══

    # 累加本区间的prompt / 生成token数（调度器每步调用）
    def update_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        if not self.throughput_enabled:
            return
        self.num_prompt_tokens += num_prompt_tokens
        self.num_generation_tokens += num_generation_tokens

    # 吞吐区间是否到期（busy_loop每轮都问，所以只做一次比较）
    def window_expired(self, now: float) -> bool:
        """Whether the throughput window is due to close.

        Split out so `heartbeat_throughput` can answer it without touching the
        queues or the KV pool. The busy loops call that on *every* pass, busy
        or idle, so this runs at loop frequency and says no all but once per
        interval; it takes the `now` the loop already read.
        """
        # 【译】吞吐窗口是否到了该关闭的时候。
        #
        #   单独拆出来，这样 heartbeat_throughput 不用碰队列或KV池就能回答它。busy loop每一轮——无论忙还是闲——都会调用那个函数，
        #   所以这里以循环的频率运行，除了每个区间一次以外都回答"否"；它接收循环已经读好的 now。
        return (
            self.throughput_enabled
            and now - self._throughput_last_log_time >= self.throughput_log_interval_s
        )

    # 区间到期时打印一行"Engine 000: Avg prompt throughput ..."，然后清零区间计数
    def maybe_log_throughput(
        self,
        num_running_reqs: int,
        num_waiting_reqs: int,
        kv_usage: float | None,
    ) -> None:
        if not self.throughput_enabled:
            return
        now = time.monotonic()
        elapsed = now - self._throughput_last_log_time
        # `elapsed <= 0` is unreachable while the interval holds the positive
        # value `__init__` validated, but the interval is a public attribute
        # and the division is two lines below; one compare keeps that division
        # safe whatever the attribute has been set to since.
        # 【译】只要间隔保持着 __init__ 验证过的正值，elapsed <= 0 就不可能发生，但间隔是一个公开属性，而除法就在两行之下；
        #   多一次比较就能让那次除法安全，不管这个属性之后被设成了什么。
        if elapsed < self.throughput_log_interval_s or elapsed <= 0:
            return
        # 完全空闲（没有token、没有请求）时不打印，只重置计时起点
        if (
            self.num_prompt_tokens == 0
            and self.num_generation_tokens == 0
            and num_running_reqs == 0
            and num_waiting_reqs == 0
        ):
            self._throughput_last_log_time = now
            return
        prompt_throughput = self.num_prompt_tokens / elapsed
        generation_throughput = self.num_generation_tokens / elapsed
        # The prefix-cache hit rate is owned by this same object now; pull it
        # from the cache section rather than have the caller thread it in.
        # Windowed, like every other field on this line, and None — rendered
        # `n/a` — when nothing has been measured. "0.0%" is a claim about
        # reuse, and a P/D decode engine never reaches `update_cache` at all.
        # 【译】前缀缓存命中率现在归这同一个对象所有；从cache部分取它，而不是让调用方传进来。和这一行的其他每个字段一样按窗口计算，
        #   没有测量时为None——显示为 n/a。"0.0%" 是一个关于复用的断言，而一个P/D decode引擎根本不会走到 update_cache。
        prefix_cache_hit_rate = (
            self.recent_cache_hit_rate if self.cache_enabled else None
        )
        logger.info(
            "%sEngine %03d: Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, Running: %d reqs, "
            "Waiting: %d reqs, GPU KV cache usage: %s, "
            "Prefix cache hit rate: %s",
            self.label,
            self.engine_index,
            prompt_throughput,
            generation_throughput,
            num_running_reqs,
            num_waiting_reqs,
            "n/a" if kv_usage is None else f"{kv_usage * 100:.1f}%",
            (
                "n/a"
                if prefix_cache_hit_rate is None
                else f"{prefix_cache_hit_rate * 100:.1f}%"
            ),
        )
        self._throughput_last_log_time = now
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0
