# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What an engine and its workers support, so a caller can ask instead of guess.

LumenRL currently probes ATOM with ``hasattr(self.engine, ...)`` in four places
and otherwise hard-codes behaviour per backend. That works only while the two
repos move together; the moment they do not, a missing method is discovered as a
runtime failure mid-rollout rather than at negotiation time.

Two layers, answering different questions:

- :class:`WorkerCapabilities` -- what one rank can do. Collected from every rank
  over ``collective_rpc``, so discovery uses the mechanism it describes.
- :class:`EngineCapabilities` -- the engine-wide answer: static topology plus the
  **intersection** of the per-rank feature sets.

Intersection, not union, is the load-bearing choice. A feature present on some
ranks cannot be driven by a collective: the ranks that have it would block in
one collective while the rest went elsewhere, which deadlocks the group with no
error. Advertising it would be worse than not knowing.

Kept free of heavy imports for the same reason as ``collective_rpc``: the
dispatch and negotiation layers must import on a machine with no GPU build.
"""

# 【译】一个引擎和它的worker支持什么，这样调用方可以询问而不是猜测。
#
#   LumenRL目前在四个地方用 hasattr(self.engine, ...) 探测ATOM，其余地方按后端硬编码行为。这只在两个仓库同步演进时才行得通；
#   一旦不同步，缺失的方法就会在rollout进行中作为运行时失败被发现，而不是在协商时。
#
#   两层，回答不同的问题：
#
#   - WorkerCapabilities —— 一个rank能做什么。通过 collective_rpc 从每个rank收集，所以发现机制使用的正是它所描述的那套机制。
#   - EngineCapabilities —— 引擎范围的答案：静态拓扑，加上各rank特性集合的**交集**。
#
#   交集而不是并集，这是承重的选择。一个只在部分rank上存在的特性无法用集合操作来驱动：拥有它的rank会阻塞在一个集合操作里，
#   而其余rank去了别处，这会让整个组死锁且没有任何报错。宣称支持它比"不知道"更糟。
#
#   与 collective_rpc 一样避免重量级导入，原因相同：分发层和协商层必须能在没有GPU构建的机器上导入。

from dataclasses import dataclass, field

# 【导读】能力发现（capability discovery）的数据类型：一个worker的自我描述，以及全引擎的汇总
# 调用方：rollout/async_engine.py 的 get_capabilities（经collective_rpc("get_worker_capabilities")收集每个rank，
#   再用 EngineCapabilities.from_workers 汇总）；worker一侧的回答在 rollout/capabilities.py（ModelRunner的mixin）
# 与主干的关系：不在推理主路径上；是【分支】RL rollout对接外部训练框架时的协商接口，见第10批

# Bump when the collective-RPC wire contract changes incompatibly. A consumer
# that pins a version can refuse to negotiate rather than fail mid-collective.
# 【译】当集合RPC的线路契约发生不兼容的变化时就递增它。固定了某个版本的使用方可以拒绝协商，而不是在集合操作进行中失败。
COLLECTIVE_RPC_PROTOCOL_VERSION = 1


# 一个rank的能力：协议版本、tp_rank、本地DP rank、支持的方法和特性
@dataclass(frozen=True)
class WorkerCapabilities:
    """One rank's view of itself."""

    # 【译】一个rank对自己的看法。

    protocol_version: int
    tp_rank: int
    dp_rank_local: int
    methods: frozenset[str] = frozenset()
    features: frozenset[str] = frozenset()

    @classmethod
    # 从worker经线路返回的dict构造（缺键时取默认值）
    def from_payload(cls, payload: object) -> "WorkerCapabilities":
        """Build from the dict a worker returned over the wire.

        Tolerant of missing keys on purpose: an older worker that predates a
        field should report less, not fail the whole negotiation.
        """
        # 【译】从worker经线路返回的dict构造。
        #
        #   有意容忍缺失的键：一个早于某字段的旧worker应当报告得少一些，而不是让整个协商失败。
        if not isinstance(payload, dict):
            raise TypeError(
                f"worker capabilities must be a dict, got {type(payload).__name__}"
            )
        return cls(
            protocol_version=int(payload.get("protocol_version", 0)),
            tp_rank=int(payload.get("tp_rank", -1)),
            dp_rank_local=int(payload.get("dp_rank_local", 0)),
            methods=frozenset(payload.get("methods", ())),
            features=frozenset(payload.get("features", ())),
        )


# 全引擎的能力：并行拓扑、KV dtype，以及所有rank都支持的方法/特性（交集）
@dataclass(frozen=True)
class EngineCapabilities:
    """The engine-wide answer a consumer should negotiate against."""

    # 【译】使用方应当据以协商的、引擎范围的答案。

    protocol_version: int
    tp_world_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    workers_per_engine: int
    kv_cache_dtype: str
    # Only what every rank reports. See the module docstring.
    # 【译】只包含每个rank都报告了的。见模块docstring。
    methods: frozenset[str] = frozenset()
    features: frozenset[str] = frozenset()
    worker_count: int = 0
    _mismatches: tuple[str, ...] = field(default=())

    @classmethod
    # 从各rank的WorkerCapabilities汇总：协议版本必须一致，方法/特性取交集，并记下"部分rank才有"的名字
    def from_workers(cls, *, config, workers) -> "EngineCapabilities":
        if not workers:
            raise ValueError("no worker capabilities to aggregate")

        versions = {w.protocol_version for w in workers}
        if len(versions) != 1:
            raise RuntimeError(
                f"workers disagree on the RPC protocol version: {sorted(versions)}"
            )

        methods = frozenset.intersection(*(w.methods for w in workers))
        features = frozenset.intersection(*(w.features for w in workers))

        # Record what was dropped, so a caller debugging a "missing" feature can
        # see it was present-but-not-universal rather than absent everywhere.
        # 【译】记录被丢掉的东西，这样一个排查"缺失"特性的调用方能看到它是"存在但不普遍"，而不是在每个rank上都不存在。
        union_methods = frozenset.union(*(w.methods for w in workers))
        union_features = frozenset.union(*(w.features for w in workers))
        mismatches = tuple(
            sorted((union_methods - methods) | (union_features - features))
        )

        parallel = getattr(config, "parallel_config", None)
        tp_world_size = int(getattr(config, "tp_world_size", len(workers)))
        pcp_size = int(getattr(config, "prefill_context_parallel_size", 1) or 1)
        return cls(
            protocol_version=versions.pop(),
            tp_world_size=tp_world_size,
            data_parallel_size=int(getattr(parallel, "data_parallel_size", 1) or 1),
            pipeline_parallel_size=int(
                getattr(config, "pipeline_parallel_size", 1) or 1
            ),
            prefill_context_parallel_size=pcp_size,
            workers_per_engine=tp_world_size * pcp_size,
            kv_cache_dtype=str(getattr(config, "kv_cache_dtype", "auto")),
            methods=methods,
            features=features,
            worker_count=len(workers),
            _mismatches=mismatches,
        )

    # 某个名字（方法或特性）是否在所有rank上都可用
    def supports(self, name: str) -> bool:
        """Whether *name* is usable on every rank, as a method or a feature."""
        # 【译】*name* 作为方法或特性，是否在每个rank上都可用。
        return name in self.methods or name in self.features

    # 只在部分rank上报告、因而没有被宣称支持的名字
    def partial(self) -> tuple[str, ...]:
        """Names some ranks reported but not all, hence not advertised."""
        # 【译】一部分rank报告了、但不是全部，因此没有被宣称的名字。
        return self._mismatches
