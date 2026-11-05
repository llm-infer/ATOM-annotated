# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# =====================================================================================
# 【导读】state_runtime.py —— 后端声明的"状态怎么交接"（StateTransfer）、它与页单元快照几何的组合（StateRuntime）、
#   每个 batch 前要执行的状态搬运（StateMaintenanceOps），以及 midstep checkpoint 的位置判定（第4批 4b，B级）
#   前置：fork / copy / replay / none 各是什么、谁在用——见 state_pool.py 文件开头【科普】④(b)；页单元见同处⑤
#   数据流（谁产生、谁消费）：
#     注意力后端 AttentionMetadataBuilder.state_transfer()        → StateTransfer（gdn_attn.py、kimi_mla_gdn_attn.py、
#                                                                  deepseek_v4_attn.py、deepseek_v41/backend.py；默认 none）
#     ModelRunner.get_num_blocks：规划完显存后组合成 StateRuntime（copy 时附 PagedStateCheckpointSpec），to_wire() 成普通 dict
#        ──READY 前的 get_num_blocks 返回值（跨进程）──> EngineCore：StateRuntime.from_wire(...)（engine_core.py）
#        ──> Scheduler(..., state_runtime) ──> BlockManager(..., state_runtime)：据此构造 StateSlotPool / PagedStateCheckpointCoordinator
#     每个 batch：BlockManager.take_state_maintenance_ops() → StateMaintenanceOps → ScheduledBatch.state_maintenance_ops
#        → AttentionMetadataBuilder.build 在 forward 之前、在计算流上执行（model_ops/attentions/backends.py）
#   对比nano-vllm：没有对应物——nano-vllm 没有每请求状态，也就没有"状态怎么交给另一个槽位"的问题。
# =====================================================================================
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)

FORK = "fork"
COPY = "copy"
NONE = "none"
# No state is carried or checkpointed: a forward resuming at a prefix hit
# rebuilds it by recomputing the hash block below the hit
# (`BlockManager.replay_start`), writing none of what the cache holds there.
# 【译】不携带、也不 checkpoint 任何状态：在前缀命中处恢复的 forward 通过重算命中点下方的那个 hash block 来重建状态
#   （BlockManager.replay_start），cache 在那里已有的内容一概不写。
# 例：V4.1 的 hash block = 128 个 token（正好盖住 128 的滑动窗口），命中在 10240 → 从 10112 开始 prefill，
#   10112~10239 这 128 个 token 只为重建窗口状态而算，它们的 KV 行不重写（见 arg_utils.py state_checkpoint_interval_tokens 上方【科普】）
REPLAY = "replay"
# 四种交接方式的字符串常量（也是 to_wire 里的 kind 字段）：
#   FORK   "交出去、换新的"：快照就是原槽位本身，下一次 forward 读旧槽位写新槽位（滚动状态，GDN/KDA 的 fork 路径）
#   COPY   "复制一份"：拥有者不动，快照是拷到页单元里的副本（Qwen GDN、Kimi-K3、GLM-5.3 默认，DeepSeek-V4）
#   NONE   没有可交接的状态（默认；successor_room = inf，从不放 checkpoint）
#   REPLAY 不保存、命中时重算（DeepSeek-V4.1 encoder 侧 SWA 有界重放）


@dataclass(frozen=True)
class StateTransfer:
    """How a backend transfers one request's state to another slot."""

    # 【译】一个后端如何把一条请求的状态交接到另一个槽位。
    # 字段：kind（上面四个常量之一）、fork_tokens（只对 FORK 有意义：successor_room）、paged_layout_id（只对 COPY：页单元快照的
    #   版本化布局 id，读写两端必须一致）、readable_midstep / midstep_grid / midstep_min_offset（midstep 几何，见下）
    # 合法组合（__post_init__ 校验，不合法直接抛错）：
    #   kind（方式） fork_tokens   paged_layout_id   readable_midstep
    #   fork     > 0           必须为 None       可 True / False
    #   copy     必须为 0      非空字符串         可 True / False
    #   none     必须为 0      必须为 None       必须 False
    #   replay   必须为 0      必须为 None       必须 False
    #   readable_midstep=True 时 midstep_grid 必须 > 0；为 False 时两个 midstep 数都必须为 0

    kind: str
    fork_tokens: int = 0
    paged_layout_id: str | None = None
    #: Whether this backend can extract a state snapshot at a position *inside*
    #: a forward, rather than only at the forward's last token.
    #:
    #: A separate axis from `kind`, and it must stay one. `kind` answers how a
    #: slot is handed to *another request*; this answers where within one
    #: forward a snapshot can be taken at all. GDN is `fork(1)` and — because
    #: its chunk kernel's per-chunk intermediates are copied out —
    #: midstep-readable; DeepSeek-V4 is `copy()` and is not, because its
    #: compressor ring is not materialized at interior boundaries the way a
    #: chunk kernel's `h` is. Deriving either from the other would silently
    #: turn the prefill chunk cut off for a backend that still needs it.
    #:
    #: Default False, so every backend keeps today's behavior until it has
    #: actually ported the copy-out.
    # 【译】这个后端能否在一次 forward"内部"的某个位置提取状态快照，而不只是在 forward 的最后一个 token 处。
    #
    #   它是独立于 kind 的另一根轴，而且必须保持独立。kind 回答的是槽位怎么交给"另一个请求"；这个回答的是在一次 forward 里
    #   到底能在哪里拍快照。GDN 是 fork(1)，并且——因为它 chunk kernel 的逐 chunk 中间结果会被拷出——是 midstep 可读的；
    #   DeepSeek-V4 是 copy()，而不可读，因为它的压缩器环形缓冲不会像 chunk kernel 的 h 那样在内部边界上物化出来。
    #   从一个推导另一个，会悄悄地把仍然需要 prefill chunk 切分的后端的切分关掉。
    #
    #   默认 False，所以每个后端在真正移植好拷出逻辑之前都保持现在的行为。
    # 观察：v0.1.8 的 fork GDN 声明的是 readable_midstep=False（gdn_attn.py state_transfer 的 docstring 说明了原因）；
    #   midstep 可读的只有 copy 方式的 Qwen GDN，且要 ATOM_GDN_STATE_MIDSTEP=1（默认 0）
    readable_midstep: bool = False
    #: Token spacing of the interior states a readable backend exposes, counted
    #: from the *start of the forward*, not from position 0. GDN's chunk kernel
    #: materializes `h` every 64 tokens of the step, so a forward over
    #: `(start, end]` can be read at `start + 64k` and nowhere else inside it.
    #: 0 when not readable.
    # 【译】可读后端暴露的内部状态之间的 token 间距，从"这次 forward 的起点"而不是从位置 0 数起。GDN 的 chunk kernel 在这一步的
    #   每 64 个 token 处物化一次 h，所以对 (start, end] 的 forward 只能在 start + 64k 处读取，内部别处都不行。不可读时为 0。
    midstep_grid: int = 0
    #: Tokens of this forward a snapshot needs behind it: GDN's conv window
    #: (`conv_kernel - 1`) is sliced out of the step's own conv input. 0 when
    #: not readable.
    # 【译】一次快照在这次 forward 里需要垫在它前面的 token 数：GDN 的卷积窗口（conv_kernel - 1）是从这一步自己的卷积输入里
    #   切出来的。不可读时为 0。
    # 例：Qwen3-Next 的 linear_conv_kernel_dim=4 → midstep_min_offset=3（gdn_attn.py _conv_state_len），midstep_grid=64（fla_ops/chunk.py CHUNK_SIZE）
    midstep_min_offset: int = 0

    # dataclass 生成 __init__ 之后自动调用；frozen=True 的实例不能改字段，这里只做校验（规则见类 docstring 下方的表）
    def __post_init__(self) -> None:
        if not isinstance(self.fork_tokens, int) or isinstance(self.fork_tokens, bool):
            raise TypeError("fork_tokens must be an integer")
        if not isinstance(self.readable_midstep, bool):
            raise TypeError("readable_midstep must be a bool")
        if self.kind in (NONE, REPLAY) and self.readable_midstep:
            raise ValueError(f"{self.kind} transfer has no state to read midstep")
        for name in ("midstep_grid", "midstep_min_offset"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if self.readable_midstep and self.midstep_grid <= 0:
            raise ValueError("a midstep-readable transfer must declare its grid")
        if not self.readable_midstep and (self.midstep_grid or self.midstep_min_offset):
            raise ValueError("midstep geometry on a transfer that is not readable")
        if self.kind == COPY:
            if self.fork_tokens != 0:
                raise ValueError("copy transfer cannot bind successor tokens")
            if not isinstance(self.paged_layout_id, str) or not self.paged_layout_id:
                raise ValueError("copy transfer requires a non-empty PAGE layout id")
            return
        if self.kind == FORK:
            if self.fork_tokens <= 0:
                raise ValueError("fork transfer requires positive fork_tokens")
            if self.paged_layout_id is not None:
                raise ValueError("fork transfer cannot declare a PAGE layout")
            return
        if self.kind in (NONE, REPLAY):
            if self.fork_tokens != 0 or self.paged_layout_id is not None:
                raise ValueError(
                    f"{self.kind} transfer cannot carry tokens or a PAGE layout"
                )
            return
        raise ValueError(f"unknown state transfer kind {self.kind!r}")

    # 下面四个 classmethod 是构造各 kind 的快捷方式（后端 state_transfer() 里写 StateTransfer.fork(1) / .copy(layout_id) 等）
    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def replay(cls) -> "StateTransfer":
        return cls(REPLAY)

    @classmethod
    def fork(
        cls,
        tokens: int,
        readable_midstep: bool = False,
        midstep_grid: int = 0,
        midstep_min_offset: int = 0,
    ) -> "StateTransfer":
        return cls(
            FORK,
            tokens,
            readable_midstep=readable_midstep,
            midstep_grid=midstep_grid,
            midstep_min_offset=midstep_min_offset,
        )

    @classmethod
    def copy(
        cls,
        layout_id: str,
        readable_midstep: bool = False,
        midstep_grid: int = 0,
        midstep_min_offset: int = 0,
    ) -> "StateTransfer":
        return cls(
            COPY,
            paged_layout_id=layout_id,
            readable_midstep=readable_midstep,
            midstep_grid=midstep_grid,
            midstep_min_offset=midstep_min_offset,
        )

    # to_wire / from_wire：与普通 dict 互转，用于从 ModelRunner 子进程（get_num_blocks 的返回值）传回 EngineCore；
    #   from_wire 要求字段集合与 to_wire 完全一致，多一个少一个都报错，防止两端版本不一致时被静默误读
    def to_wire(self) -> dict[str, str | int | bool | None]:
        return {
            "kind": self.kind,
            "fork_tokens": self.fork_tokens,
            "paged_layout_id": self.paged_layout_id,
            "readable_midstep": self.readable_midstep,
            "midstep_grid": self.midstep_grid,
            "midstep_min_offset": self.midstep_min_offset,
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateTransfer":
        if not isinstance(wire, Mapping):
            raise TypeError("state transfer capability must be a mapping")
        expected = {
            "kind",
            "fork_tokens",
            "paged_layout_id",
            "readable_midstep",
            "midstep_grid",
            "midstep_min_offset",
        }
        if set(wire) != expected:
            raise ValueError(
                "invalid state transfer capability fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            kind=wire["kind"],  # type: ignore[arg-type]
            fork_tokens=wire["fork_tokens"],  # type: ignore[arg-type]
            paged_layout_id=wire["paged_layout_id"],  # type: ignore[arg-type]
            readable_midstep=wire["readable_midstep"],  # type: ignore[arg-type]
            midstep_grid=wire["midstep_grid"],  # type: ignore[arg-type]
            midstep_min_offset=wire["midstep_min_offset"],  # type: ignore[arg-type]
        )

    # property：像属性一样读（transfer.copies），不用加括号调用
    @property
    def copies(self) -> bool:
        return self.kind == COPY

    @property
    def forks(self) -> bool:
        return self.kind == FORK

    @property
    def replays(self) -> bool:
        return self.kind == REPLAY

    @property
    # 状态类协议里的 successor_room（state_cache.py）：none / replay → inf（永不放 checkpoint）；fork(n) → n；copy → 0
    def successor_room(self) -> float:
        return inf if self.kind in (NONE, REPLAY) else float(self.fork_tokens)


@dataclass(frozen=True)
class StateRuntime:
    """Validated state transfer and optional checkpoint geometry."""

    # 【译】经过校验的状态交接方式，加上可选的 checkpoint 几何。
    # 规则：copy 必须带 PagedStateCheckpointSpec，且两者的 layout_id 相同；fork / none / replay 不能带 spec
    # 构造：ModelRunner.get_num_blocks（model_runner.py；copy 时先按 PAGE 子池的 entry 大小算出 page_unit_bytes 等再构造 spec）

    transfer: StateTransfer = field(default_factory=StateTransfer.none)
    checkpoint_spec: PagedStateCheckpointSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transfer, StateTransfer):
            raise TypeError("state runtime transfer must be a StateTransfer")
        if self.checkpoint_spec is not None and not isinstance(
            self.checkpoint_spec, PagedStateCheckpointSpec
        ):
            raise TypeError(
                "state runtime checkpoint_spec must be a PagedStateCheckpointSpec"
            )
        if self.transfer.copies:
            if self.checkpoint_spec is None:
                raise ValueError(
                    "StateTransfer.copy(layout_id) requires a PAGE checkpoint spec"
                )
            if self.transfer.paged_layout_id != self.checkpoint_spec.layout_id:
                raise ValueError(
                    "state runtime PAGE layout mismatch: "
                    f"transfer={self.transfer.paged_layout_id!r}, "
                    f"spec={self.checkpoint_spec.layout_id!r}"
                )
        elif self.checkpoint_spec is not None:
            raise ValueError(
                f"StateTransfer.{self.transfer.kind} cannot carry a PAGE checkpoint spec"
            )

    def to_wire(self) -> dict[str, object]:
        return {
            "transfer": self.transfer.to_wire(),
            "checkpoint_spec": (
                None if self.checkpoint_spec is None else self.checkpoint_spec.to_wire()
            ),
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateRuntime":
        if not isinstance(wire, Mapping):
            raise TypeError("state runtime must be a mapping")
        expected = {"transfer", "checkpoint_spec"}
        if set(wire) != expected:
            raise ValueError(
                "invalid state runtime fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        checkpoint_wire = wire["checkpoint_spec"]
        checkpoint_spec = (
            None
            if checkpoint_wire is None
            else PagedStateCheckpointSpec.from_wire(checkpoint_wire)
        )
        return cls(
            transfer=StateTransfer.from_wire(wire["transfer"]),
            checkpoint_spec=checkpoint_spec,
        )


DEFAULT_STATE_RUNTIME = StateRuntime()
# 默认值（transfer=none、无 spec）：Scheduler / BlockManager 的 state_runtime 参数默认用它，没有每请求状态的模型就是这个


@dataclass(frozen=True)
class StateMaintenanceOps:
    """State movement drained once before a model batch."""

    # 【译】在一个模型 batch 之前被取走（并执行）一次的状态搬运。
    # 三类操作：
    #   relocations          活跃槽位之间的搬迁 (src, dst)（StateSlotPool.take_relocations；v0.1.8 总为空，见 state_pool.py SlotRetirement 处的观察）
    #   checkpoint_stores    把活跃槽位 scatter 进页单元（存快照，CheckpointStoreOp）
    #   checkpoint_restores  把页单元 gather 回活跃槽位（恢复快照，CheckpointRestoreOp）
    # 产生：BlockManager.take_state_maintenance_ops（构造 batch 时；空 batch 不取，避免操作被"取走却没执行"）
    # 消费：AttentionMetadataBuilder.build（backends.py）在 forward 前执行 relocate_state_slots / execute_paged_state_copies

    relocations: tuple[tuple[int, int], ...] = ()
    checkpoint_stores: tuple[CheckpointStoreOp, ...] = ()
    checkpoint_restores: tuple[CheckpointRestoreOp, ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.relocations or self.checkpoint_stores or self.checkpoint_restores
        )


# ── midstep checkpoint geometry ──────────────────────────────────────────────
#
# One predicate for both ends of a midstep checkpoint. The scheduler reserves a
# position and publishes its hash after the forward; the runner writes the
# bytes during it. If the two disagree on which positions are writable, the
# scheduler files a hash over a destination nobody wrote, and a request that
# resumes there continues from the previous tenant's state with no error
# anywhere. So the scheduler filters with this and the runner asserts it.
# 【译】midstep checkpoint 两端共用的同一个判定。调度器预留一个位置、在 forward 之后发布它的 hash；runner 在 forward 期间写字节。
#   如果两端对"哪些位置可写"意见不一，调度器就会在一个没人写过的目标上登记 hash，从那里恢复的请求会接着上一任住户的状态
#   继续算，而且哪里都不报错。所以调度器用它来过滤，runner 用它来断言。
# 调用方：BlockManager.midstep_positions（midstep_target_ok + snap_midstep_position 过滤候选位置）；
#   runner 侧 gdn_attn.py GDN builder 的 _checkpoint_targets 调 midstep_checkpoint_targets（内部再用 midstep_target_ok 断言）


def midstep_target_ok(
    pos: int, start: int, end: int, grid: int, min_offset: int
) -> bool:
    """Whether a forward over `(start, end]` can snapshot the state at `pos`.

    Interior positions must sit on the forward's own grid, `start + grid*k`:
    that is where the chunk kernel left a state in `h`, and a position between
    two of them reads the earlier one while being filed under the later hash.
    `end` is always readable, from the runtime slot the forward leaves behind.

    `min_offset` tokens of this forward must precede `pos`, because the conv
    window is sliced out of this forward's conv input and has no earlier rows.
    """
    # 【译】对 (start, end] 的一次 forward 能否给 pos 处的状态拍快照。
    #   内部位置必须落在这次 forward 自己的网格 start + grid*k 上：chunk kernel 只在那里把状态留在 h 里；落在两个网格点之间的位置
    #   会读到较早那个网格点的状态，却被登记在较晚的 hash 名下。end 总是可读的，从 forward 留下的运行时槽位里读。
    #
    #   pos 之前必须有这次 forward 的 min_offset 个 token，因为卷积窗口是从这次 forward 的卷积输入里切出来的，它没有更早的行。
    # 例：forward 覆盖 (8000, 16384]，grid=64，min_offset=3：
    #   pos=8192 → 8192−8000=192，是 64 的倍数 → True；pos=8200 → 200 不是 64 的倍数 → False
    #   pos=8002 → 前面只有 2 个 token < 3 → False；pos=16384 = end → True
    if not start < pos <= end or pos - start < min_offset:
        return False
    return pos == end or (pos - start) % grid == 0


def snap_midstep_position(pos: int, start: int, end: int, grid: int) -> int:
    """`pos` moved back onto the grid of a forward over `(start, end]`.

    `end` stays where it is: the runtime slot holds it exactly. Anything else
    goes back to the last state the chunk kernel materialized at or before it,
    which costs the next resume at most `grid - 1` tokens of recompute.
    """
    # 【译】把 pos 挪回对 (start, end] 的 forward 的网格上。
    #   end 原地不动：运行时槽位恰好保存着它。其他位置退回到 chunk kernel 在它或它之前物化出的最后一个状态，
    #   代价是下一次恢复最多多重算 grid - 1 个 token。
    # 例（同上）：snap(8200) = 8000 + (200 // 64) × 64 = 8192；snap(16384) = 16384
    if pos == end:
        return pos
    return start + (pos - start) // grid * grid


def midstep_checkpoint_targets(
    state_rows: list[int],
    num_cached_tokens,
    num_scheduled_tokens,
    runtime_slots: list[int],
    saves: list[list[tuple[int, int]]],
    num_slots: int,
    grid: int,
    min_offset: int,
) -> list[tuple[int, int, int, int, int]]:
    """`(row, slot, offset, is_end, runtime_slot)` for every reservation.

    `saves`, `runtime_slots` and `state_rows` are indexed by state-holding
    sequence; `state_rows[j]` is that sequence's row in the batch, which is
    what the token counts and the kernel's `cu_seqlens` are indexed by. The
    two only coincide when every row holds state.

    Raises rather than skipping. Every reservation here has already been
    promised to `commit_midstep`, which publishes its hash after the forward;
    a skipped write is a findable checkpoint over unwritten bytes.
    """
    # 【译】为每个预留给出 (row, slot, offset, is_end, runtime_slot)。
    #   saves、runtime_slots 和 state_rows 按"持有状态的序列"编号；state_rows[j] 是该序列在 batch 中的行号，
    #   token 计数和 kernel 的 cu_seqlens 是按行号索引的。只有每一行都持有状态时两者才重合。
    #
    #   出错时抛异常而不是跳过。这里的每个预留都已经承诺给了 commit_midstep，它会在 forward 之后发布 hash；
    #   被跳过的写入就是一个能被找到、但字节没写过的 checkpoint。
    # 输入：num_cached_tokens / num_scheduled_tokens 按 batch 行号索引（start = 已缓存数，end = start + 本步调度数）；
    #   saves[j] = 该序列的 [(目标槽位, 位置), ...]（来自 seq.midstep_reservations）；num_slots = 本 rank 状态池的槽位数
    # 输出：offset = pos − start（第几个 token 处），is_end = 1 表示就是 forward 末尾（从 runtime_slot 直接拷），否则从 chunk kernel 的 h 里切
    # 例：一条序列在第 0 行，start=8000、本步 8384 个 token（end=16384），runtime_slots=[5]，saves=[[(17, 8192), (21, 16384)]]
    #   → [(0, 17, 192, 0, 5), (0, 21, 8384, 1, 5)]
    # 调用方：gdn_attn.py 里 GDN builder 的 _checkpoint_targets
    if not (len(state_rows) == len(runtime_slots) == len(saves)):
        raise RuntimeError(
            "midstep targets: state rows, runtime slots and saves disagree "
            f"({len(state_rows)}, {len(runtime_slots)}, {len(saves)})"
        )
    out = []
    for j, reservations in enumerate(saves):
        row = state_rows[j]
        start = int(num_cached_tokens[row])
        end = start + int(num_scheduled_tokens[row])
        for dst, pos in reservations:
            dst, pos = int(dst), int(pos)
            if not 0 <= dst < num_slots:
                raise RuntimeError(
                    f"midstep checkpoint slot {dst} outside this rank's "
                    f"{num_slots}-slot state pool"
                )
            if not midstep_target_ok(pos, start, end, grid, min_offset):
                raise RuntimeError(
                    f"midstep checkpoint at {pos} is not writable by a forward "
                    f"over ({start}, {end}] (grid {grid}, min offset {min_offset})"
                )
            out.append((row, dst, pos - start, int(pos == end), runtime_slots[j]))
    return out
