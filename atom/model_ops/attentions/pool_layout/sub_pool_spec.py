# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte-currency description of the cache classes an attention type needs.

There are exactly two pools, distinguished by what their size scales with:

  Pool.PAGE   scales with how much history is cached. The paged KV blocks.
  Pool.STATE  scales with in-flight requests, not with sequence length. The
              sliding window, the DeepSeek-V4 compressor ring, GDN/Mamba
              recurrent state. Classification follows the scaling, not the
              addressing: what puts a class in STATE is that its total is
              `max_num_seqs * per_req`, whatever the rows are indexed by.

A pool is a budget region, not a single entry size. Inside one pool there can
be several **entry classes**, each with its own index space, its own bytes
per entry, and — in the STATE pool — its own per-request multiplicity. The
sliding-window ring and the compressor ring live in the same pool and are
counted separately; collapsing them into one entry size would make the two
counts unrecoverable.

Entry classes are keyed by name. Two specs with the same name are the same
index space, so their `entry_bytes` add — that is how an Eagle3 draft KV pool
rides the target model's block ids, and how DeepSeek-V4's indexer cache rides
the same block table as its main compressed KV. Two specs with different names
are different index spaces even inside one pool and keep separate counts: the
sliding-window ring is `win_with_spec` rows per request while the compressor
ring is one entry per request, so they can never share a count.

This module deliberately defines no *architecture* vocabulary. A name is owned
by whatever consumes the count — `kv_block.py` names the per-request slot
class — and the backend that declares the spec imports it from there. The one name defined here, `PAGED_CLASS`, is
not an architecture's: it is the allocation regime itself, since PAGE is by
construction a single shared index space that every contributor adds bytes to.

Cross-request sharing is a property of the cache layer above, not of this
sizing. A STATE entry class is shared by copying it into the resuming
request's slot at a checkpoint, not by two requests pointing at one entry.
What is reserved here is the floor an in-flight request cannot run without;
retained-and-shared entries live in the same region and compete for the rest.
"""

# 【译】以"字节"为货币，描述一种注意力类型需要哪些缓存类。
#
#   恰好有两个池，按"大小随什么增长"区分：
#
#     Pool.PAGE   随缓存了多少历史增长。即分页 KV block。
#     Pool.STATE  随在途（in-flight）请求数增长，与序列长度无关。滑动窗口、DeepSeek-V4 的压缩器环、
#                 GDN/Mamba 的递归状态。分类看的是增长方式而不是寻址方式：让一个类进 STATE 的，是它的总量
#                 等于 max_num_seqs * 每请求量，不管它的行按什么编号。
#
#   池是一块预算区域，不是一个单一的条目大小。一个池里可以有好几个"条目类"，各有自己的编号空间、每条目
#   字节数，以及（在 STATE 池里）自己的每请求倍数。滑动窗口环和压缩器环住在同一个池里但分开计数；把它们
#   并成一个条目大小，两个计数就再也恢复不出来了。
#
#   条目类按名字区分。同名的两个 spec 是同一个编号空间，所以它们的 entry_bytes 相加——EAGLE3 草稿模型的
#   KV 池就是这样搭上目标模型的 block id 的，DeepSeek-V4 的索引器缓存也是这样和它的主压缩 KV 共用同一张
#   block table。不同名的两个 spec 即使在同一个池里也是不同的编号空间，各自计数：滑动窗口环每请求
#   win_with_spec 行，压缩器环每请求一个条目，所以它们永远不可能共用一个计数。
#
#   本模块有意不定义任何"架构"词汇。一个名字归使用这个计数的一方所有——kv_block.py 给每请求槽位类起名——
#   声明 spec 的后端从那里 import。这里唯一定义的名字 PAGED_CLASS 不属于任何架构：它就是分配方式本身，
#   因为 PAGE 按构造就是一个所有贡献者都往里加字节的共享编号空间。
#
#   跨请求共享是上层缓存层的性质，不是这里定大小时的事。STATE 条目类的共享方式是在 checkpoint 处把它拷进
#   续算请求的槽位，而不是让两个请求指向同一个条目。这里预留的是在途请求运行所必需的底线；被保留下来、
#   可共享的条目住在同一个区域里，争抢剩下的空间。
# 观察：v0.1.8 的 DeepSeek-V4（deepseek_v4_attn.py 的 sub_pool_specs）只声明一个 STATE 类 STATE_SLOT_CLASS
#   （entries_per_req=1）：滑动窗口和压缩器状态放在同一个槽位里（见 v4_pool_geometry.py 文件头）。上面
#   "两个环分开计数"描述的是旧布局；按名字区分编号空间这条规则本身仍然成立。
#
# =====================================================================================
# 【导读】sub_pool_spec.py —— KV 预算怎么切成"多少个 block、多少个状态槽"（第5批 5b，B级）
#   概念与带数字的例子（Qwen3-Next：PAGE 384 KiB / block、STATE 37.7 MiB / 请求）见 pool_layout/__init__.py【科普】②。
#   数据流：
#     各注意力 builder 的 sub_pool_specs()（+ 草稿模型的 draft_kv_builder）
#         └─> ModelRunner._sub_pool_specs() 收集成 list[SubPoolSpec]
#               └─> ModelRunner.get_num_blocks：plan_pools(specs, available_for_kv, max_num_seqs) -> PoolPlan
#                     ├── PP>1 时各 stage 对 block 数取全局最小 → plan.with_paged_entries(min)
#                     ├── self.pool_plan = plan（runner 进程内：V4 后端读 entries[STATE_SLOT_CLASS] 等）
#                     └── config.pool_entries / pool_entries_per_req = dict(...)（随 get_num_blocks 的返回值回到
#                         EngineCore，BlockManager 按 STATE_SLOT_CLASS 取状态槽位数）
#   对比nano-vllm：nano-vllm 的 allocate_kv_cache 只有一个公式 num_blocks = 可用显存 // block_bytes；
#     ATOM 把"一个 block 多少钱"交给各后端声明（可以是几个子池之和），并先扣掉每请求状态这块"底线"。
# =====================================================================================

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Pool(Enum):
    """Which budget region an entry class draws from."""

    # 【译】一个条目类从哪个预算区里取空间。

    PAGE = "page"
    STATE = "state"


# SubPoolSpec：一个后端对"我要一类条目"的声明（不可变）
#   谁构造：下面的 page_pool / state_pool；调用方是各注意力 builder 的 sub_pool_specs()——aiter_attention.py、
#     aiter_mla.py、gdn_attn.py、kimi_mla_gdn_attn.py、qwen4_exp_attn.py、deepseek_v4_attn.py、
#     deepseek_v41/backend.py，以及 spec_decode/draft_kv.py（草稿模型自己的 KV）
#   谁消费：plan_pools（定数量）；ModelRunner._carve_paged_pool 用 PAGE 类的 entry_bytes 核对各 builder 实际要的字节
@dataclass(frozen=True)
class SubPoolSpec:
    """One entry class: a pool, an index space, and a per-entry byte cost."""

    # 【译】一个条目类：一个池、一个编号空间、每条目的字节成本。

    pool: Pool
    name: str
    entry_bytes: int
    # STATE only. `entries_per_req` is what one in-flight request holds — 1
    # committed state, `1 + num_spec` when a rollback slot per speculated
    # token is kept, or the window block count for a sliding-window class.
    # `extra_entries` is a flat cushion on the class as a whole, on top of the
    # per-request term. What it covers is the declaring backend's business.
    # 【译】仅 STATE 用。entries_per_req 是一个在途请求持有的条目数——1 份已提交状态；为每个投机 token 保留
    #   一个回滚槽位时是 1 + num_spec；滑动窗口类则是窗口占的 block 数。extra_entries 是加在整个类上的
    #   固定余量，在"每请求"那一项之外。它用来覆盖什么由声明它的后端决定。
    # 例：GDN 不开 MTP 时 entries_per_req = 1，MTP k=3 时 = 4（gdn_attn.py slots_per_req）；
    #   v0.1.8 中没有后端传非 0 的 extra_entries（kimi_mla_gdn_attn.py 只是原样转发 GDN 的 0）
    entries_per_req: int = 0
    extra_entries: int = 0

    # 校验：STATE 类必须至少每请求 1 个条目；PAGE 类的数量完全由"剩余预算"决定，不允许声明倍数
    def __post_init__(self):
        if self.pool is Pool.STATE and self.entries_per_req < 1:
            raise ValueError(f"{self.name}: STATE entries need entries_per_req >= 1")
        if self.pool is Pool.PAGE and (self.entries_per_req or self.extra_entries):
            raise ValueError(f"{self.name}: PAGE entries are sized from the remainder")


# The paged class every backend contributes to. There is one block index
# space, so a draft KV pool or an indexer cache adds bytes to it rather than
# forming a pool of its own.
# 【译】每个后端都往里贡献字节的那个分页类。block 编号空间只有一个，所以草稿模型的 KV 池或索引器缓存
#   是往里加字节，而不是自己组成一个池。
PAGED_CLASS = "kv"


def page_pool(entry_bytes: int) -> SubPoolSpec:
    """The paged KV entry class: sized from whatever the STATE pool leaves."""
    # 【译】分页 KV 条目类：数量由 STATE 池用剩下的预算决定。
    # 输入 entry_bytes：一个 block（block_size 个 token）在这个后端所有层、所有 PAGE 字段上的总字节数
    #   （例：Qwen3-8B bf16、block_size 16 → 2,359,296 B，见 pool_layout/__init__.py【科普】①）
    return SubPoolSpec(Pool.PAGE, PAGED_CLASS, entry_bytes)


def state_pool(
    name: str,
    entry_bytes: int,
    *,
    entries_per_req: int,
    extra_entries: int = 0,
) -> SubPoolSpec:
    """A per-request state entry class."""
    # 【译】一个每请求的状态条目类。
    # 调用方：gdn_attn.py GDNStateMixin.state_spec（GDN 递归状态）、kimi_mla_gdn_attn.py（KDA）、qwen4_exp_attn.py、
    #   deepseek_v4_attn.py / deepseek_v41/backend.py（窗口 + 压缩器状态，名字都是 kv_block.STATE_SLOT_CLASS = "state"）
    return SubPoolSpec(Pool.STATE, name, entry_bytes, entries_per_req, extra_entries)


@dataclass(frozen=True)
class PoolPlan:
    """Sizing result: per entry class, how many entries and at what cost.

    Consumers index `entries` by the class name they themselves declared —
    the runner that produced the plan never needs to know the names.
    """

    # 【译】定大小的结果：每个条目类有多少个条目、每个多少钱。
    #   使用方用自己声明的类名去查 entries——产出这个计划的 runner 永远不需要知道这些名字。
    # 字段（都以类名为 key）：entries = 条目数；entry_bytes = 每条目字节；reserved_bytes = entries × entry_bytes；
    #   entries_per_req = 每请求倍数（PAGE 为 0）；paged_class = 唯一的 PAGE 类名（"kv"），没有 PAGE 类时为 None
    # 例（pool_layout/__init__.py【科普】②的 Qwen3-Next）：
    #   entries = {"kv": 138112, "state": 256}，entry_bytes = {"kv": 393216, "state": 39518208}，paged_class = "kv"

    entries: dict[str, int]
    entry_bytes: dict[str, int]
    reserved_bytes: dict[str, int]
    entries_per_req: dict[str, int]
    paged_class: str | None = None

    @classmethod
    def empty(cls) -> PoolPlan:
        """The plan before sizing has run.

        Every count reads back as 0, which is what "no pool exists yet" means
        to a consumer. Model warmup happens before the budget is known, so a
        builder asking for its entry count then must get 0 rather than an
        AttributeError — the runner installs this at construction time and
        replaces it in `get_num_blocks`.
        """
        # 【译】定大小之前的计划。
        #   每个计数读出来都是 0，这对使用方来说就是"池还不存在"。模型 warmup 发生在预算确定之前，所以 builder
        #   那时询问自己的条目数必须得到 0，而不是 AttributeError——runner 在构造时装上它，并在 get_num_blocks
        #   中替换掉。
        # 调用方：ModelRunner.__init__（self.pool_plan = PoolPlan.empty()）
        return cls(entries={}, entry_bytes={}, reserved_bytes={}, entries_per_req={})

    def with_paged_entries(self, count: int) -> PoolPlan:
        """Copy of the plan with the PAGE class resized to `count`.

        Pipeline-parallel stages reconcile their block counts to the global
        minimum after sizing; the plan has to follow so it stays the one place
        every consumer can read an entry count from.
        """
        # 【译】把 PAGE 类的数量改成 count 后的计划副本。
        #   流水线并行的各 stage 在定大小之后要把各自的 block 数统一成全局最小值；计划必须跟着改，才能继续是
        #   所有使用方读取条目数的唯一来源。
        # 调用方：ModelRunner.get_num_blocks（PP>1 时 all_reduce(MIN) 之后）
        # dataclasses.replace：复制一个 dataclass 实例并替换指定字段（frozen 实例不能原地改，只能这样"改"）
        if self.paged_class is None:
            raise ValueError("no PAGE entry class to resize")
        bytes_per = self.entry_bytes[self.paged_class]
        return replace(
            self,
            entries={**self.entries, self.paged_class: count},
            reserved_bytes={**self.reserved_bytes, self.paged_class: count * bytes_per},
        )

    @property
    def paged_entries(self) -> int:
        """Entries in the single PAGE class — the block count the scheduler
        and BlockManager are built around. Named-class-free so the runner can
        publish the plan without knowing any architecture's vocabulary."""
        # 【译】唯一的 PAGE 类里的条目数——调度器和 BlockManager 围绕它构建的 block 数。不依赖类名，
        #   这样 runner 不需要知道任何架构的词汇就能发布这个计划。
        # 读取方：ModelRunner.get_num_blocks（num_kvcache_blocks = plan.paged_entries）
        return self.entries[self.paged_class] if self.paged_class else 0

    # 所有类实际要分配的总字节；ModelRunner 分配 KV 之后拿它和实际分出的字节数核对（expected_kv_bytes）
    @property
    def total_reserved_bytes(self) -> int:
        return sum(self.reserved_bytes.values())


class InsufficientPoolBudget(RuntimeError):
    """The STATE floor alone exceeds the budget, leaving nothing to page with.

    Carries the numbers rather than a formatted message: the caller
    (ModelRunner) has the GPU-side context needed to suggest a fix.
    """

    # 【译】光 STATE 的底线就超过了预算，没有剩下任何可用于分页的空间。
    #   携带的是数字而不是格式化好的消息：调用方（ModelRunner）有 GPU 侧的上下文，能据此给出修复建议。
    # 抛出：plan_pools；捕获：ModelRunner.get_num_blocks，据 reserved_bytes 算出最低要多少 --gpu-memory-utilization，
    #   或提示调小 --max-num-seqs

    def __init__(self, reserved_bytes: int, available_bytes: int, entries: int):
        super().__init__(f"state pool needs {reserved_bytes}B of {available_bytes}B")
        self.reserved_bytes = reserved_bytes
        self.available_bytes = available_bytes
        self.entries = entries


def merge_specs(specs: list[SubPoolSpec]) -> dict[str, SubPoolSpec]:
    """Collapse specs naming the same entry class, summing `entry_bytes`.

    Same name means same index space, so pool and multiplicity must agree.
    Different names stay separate even within one pool.
    """
    # 【译】把同名的 spec 合并成一个条目类，entry_bytes 相加。
    #   同名意味着同一个编号空间，所以池和倍数必须一致；不同名的即使在一个池里也保持分开。
    # 例：目标模型 MHA 的 page_pool(2,359,296) + EAGLE3 草稿层 page_pool(65,536) → 一个 "kv" 类，每 block 2,424,832 B
    #   （草稿数字为示意：1 层、8 头、128 维、bf16、block_size 16 时 K+V = 2 × 16 × 8 × 128 × 2B）
    # 调用方：本文件 plan_pools
    merged: dict[str, SubPoolSpec] = {}
    for spec in specs:
        prev = merged.get(spec.name)
        if prev is None:
            merged[spec.name] = spec
            continue
        if (
            prev.pool,
            prev.entries_per_req,
            prev.extra_entries,
        ) != (
            spec.pool,
            spec.entries_per_req,
            spec.extra_entries,
        ):
            raise ValueError(
                f"entry class {spec.name!r} declared twice with different "
                f"pool/multiplicity: {prev} vs {spec}"
            )
        merged[spec.name] = SubPoolSpec(
            spec.pool,
            spec.name,
            prev.entry_bytes + spec.entry_bytes,
            spec.entries_per_req,
            spec.extra_entries,
        )
    return merged


def plan_pools(
    specs: list[SubPoolSpec], available_bytes: int, max_num_seqs: int
) -> PoolPlan:
    """Turn entry-class declarations plus a byte budget into entry counts.

    Pure function — no CUDA, no config, no env. The STATE floor is reserved
    first because a request cannot run without it; the PAGE pool absorbs what
    is left.

    Raises `InsufficientPoolBudget` if the floor leaves nothing to page with.
    """
    # 【译】把条目类声明加上一份字节预算，换算成条目数。
    #   纯函数——没有 CUDA、没有 config、没有环境变量。STATE 的底线先预留，因为请求没有它就跑不了；
    #   PAGE 池吸收剩下的部分。
    #   如果底线用完后没有剩下可分页的空间，抛 InsufficientPoolBudget。
    # 输入：specs = 所有 builder 的声明；available_bytes = ModelRunner 算出的 KV 预算（已扣权重、激活峰值、
    #   CUDA Graph 估计、connector 缓冲等，见 model_runner.py get_num_blocks）；max_num_seqs = 并发上限
    # 输出：PoolPlan
    # 调用方：ModelRunner.get_num_blocks（唯一调用方）
    merged = merge_specs(specs)
    entries: dict[str, int] = {}
    reserved: dict[str, int] = {}
    remaining = available_bytes

    # 第 1 步：每个 STATE 类按 max_num_seqs × entries_per_req + extra_entries 预留
    #   例：Qwen3-Next、max_num_seqs 256、不开 MTP → 256 个槽位 × 39,518,208 B ≈ 9.42 GiB
    state = {n: s for n, s in merged.items() if s.pool is Pool.STATE}
    for name, spec in state.items():
        count = max_num_seqs * spec.entries_per_req + spec.extra_entries
        cost = count * spec.entry_bytes
        entries[name], reserved[name] = count, cost
        remaining -= cost
    if state and remaining <= 0:
        raise InsufficientPoolBudget(
            reserved_bytes=sum(reserved.values()),
            available_bytes=available_bytes,
            entries=sum(entries.values()),
        )

    # 第 2 步：唯一的 PAGE 类拿走剩余的全部（向下取整到整 block）
    #   例：60 GiB − 9.42 GiB ≈ 50.6 GiB，// 393,216 B → 138,112 个 block
    #   只有 STATE、没有 PAGE 的模型（paged 为空）也合法：paged_class = None，paged_entries 读出 0
    paged = [n for n, s in merged.items() if s.pool is Pool.PAGE]
    if len(paged) > 1:
        raise ValueError(f"more than one PAGE entry class: {paged}")
    for name in paged:
        spec = merged[name]
        count = max(0, remaining // spec.entry_bytes)
        entries[name], reserved[name] = count, count * spec.entry_bytes

    return PoolPlan(
        entries=entries,
        entry_bytes={n: s.entry_bytes for n, s in merged.items()},
        reserved_bytes=reserved,
        entries_per_req={n: s.entries_per_req for n, s in merged.items()},
        paged_class=paged[0] if paged else None,
    )
