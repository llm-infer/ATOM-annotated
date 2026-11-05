# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# =====================================================================================
# 【导读】state_cache.py —— 所有"状态类"（Pool.STATE 里需要 checkpoint 才能做前缀缓存的缓存）共同遵守的协议（第4批 4b，B级）
#   前置：为什么线性注意力 / 滑动窗口这类状态需要 checkpoint、命中为什么要"退到最近的快照"——见 state_pool.py 文件开头【科普】②③；
#     fork / copy / replay 三种做法见同一处④(b)
#   实现者（v0.1.8）：
#     StateSlotPool（state_pool.py）                     fork / none / replay 方式；同时也是活跃槽位分配器
#     PagedStateCheckpointCoordinator（page_unit_checkpoint.py）   copy 方式（页单元快照）
#   使用者：只有 BlockManager。它把"当前生效的那一个"放进 self.state_caches（一个元素的元组）：
#       state_caches = (paged_state_checkpoints or state,)        copy 时是前者，否则是 StateSlotPool
#     并在这些地方调用协议方法：
#       _gated_hit ─────────────> resumable_hit        准入：KV 命中要被每个状态类"认可"（迭代到不动点）
#       hash_blocks ────────────> checkpoint           forward 恰好结束在阶梯 / 按需 / 锚点位置时保存快照
#       plan_midstep / commit_midstep / cancel_midstep ─> reserve_midstep / publish_midstep / cancel_midstep
#       checkpoint_limit / checkpointers_at / _record_checkpoint_end ─> successor_room、applies、readable_midstep
#       _record_evicted / clear_cache / state_checkpoint_fates ─> unindex / clear_index / checkpoint_fates（StateCheckpointCache）
#   对比nano-vllm：nano-vllm 只有 KV block 一种缓存，前缀命中只看 hash_to_block_id；ATOM 在它之上加了这一层"状态类可以否决命中"的协议。
# Protocol（typing）：结构化类型——任何"长得像"（有这些属性和方法）的类都算实现了它，不必继承；
#   runtime_checkable 让 isinstance(obj, StateCache) 在运行时按"有没有这些成员"判断
# =====================================================================================
from typing import Protocol, runtime_checkable

from atom.model_engine.sequence import Sequence


@runtime_checkable
class StateCache(Protocol):
    """One `Pool.STATE` cache class's checkpoint lifecycle.

    Pool.STATE holds several such classes (see `sub_pool_spec.py`): the sliding
    window, the DeepSeek-V4 compressor ring, GDN/Mamba recurrence. They have
    three things in common, and this protocol is exactly those three:

    - each scales with in-flight requests, so a boundary is only resumable if
      somebody deliberately kept its state there (`checkpoint`);
    - each can therefore veto a prefix-cache hit, by answering how far back the
      nearest boundary it *can* resume from is (`resumable_hit`);
    - *where* keeping one is worth its cost is the same question for all of
      them, so the ladder lives in `BlockManager`, not here.

    Vocabulary: *checkpoint*, noun and verb, is this — a boundary kept
    resumable, and the act of keeping it. *Publish* means something else and is
    never a synonym here: a block entering the content-addressed KV index.

    What differs between classes is only *how* a boundary is kept, and that
    follows from one property — mutability:

      immutable   a filled SWA block is never written again, so keeping it is
                  one extra ref; a reader shares it and needs nothing else.
      copyable    the DeepSeek-V4 compressor entry is a contiguous byte range,
                  so keeping it is a duplicate handed to the index and the owner
                  is never disturbed; the reader is handed a duplicate too.
      rolling     GDN recurrence is still being written by its owner and is not
                  one range to duplicate, so keeping it means handing it over
                  and taking a fresh one; the reader forks, and the forward
                  right after the hand-over has to refill the replacement.

    `successor_room` is that property, quantified — and it is the only thing the
    ladder needs to know about a class, which is why the rest of the difference
    can stay inside `checkpoint`.
    """

    # 【译】一个 Pool.STATE 缓存类的 checkpoint 生命周期。
    #   Pool.STATE 里有好几种这样的类（见 sub_pool_spec.py）：滑动窗口、DeepSeek-V4 的压缩器环形缓冲、GDN/Mamba 的递归状态。
    #   它们有三件共同的事，本协议恰好就是这三件：
    #   - 每一种都随在跑请求数扩展，所以只有有人特意把某个边界的状态保存下来（checkpoint），这个边界才可恢复；
    #   - 因此每一种都可以否决一次前缀缓存命中：回答"它能从多远之前最近的边界恢复"（resumable_hit）；
    #   - "在哪里保存一个 checkpoint 值得付出代价"对它们全都是同一个问题，所以阶梯放在 BlockManager 里，不在这里。
    #
    #   术语：checkpoint（名词和动词）就是指这个——一个被保持为可恢复的边界，以及保持它的动作。
    #   publish 指的是另一回事，在这里绝不是同义词：一个 block 进入内容寻址的 KV 索引。
    #
    #   各类之间不同的只是"怎么"保存一个边界，而这由一个性质决定——可变性：
    #     immutable（不可变）  填满的 SWA block 不会再被写，所以保存它只是多一个引用；读者共享它，别的什么都不需要。
    #     copyable（可复制）   DeepSeek-V4 压缩器条目是一段连续字节，所以保存它就是复制一份交给索引，拥有者完全不受打扰；
    #                          读者拿到的也是一份副本。
    #     rolling（滚动）      GDN 递归状态仍在被它的拥有者写、而且不是一段可以直接复制的范围，所以保存它意味着把它交出去、
    #                          自己换一个新的；读者 fork，交接之后紧接着的那次 forward 必须把替代品重新填满。
    #
    #   successor_room 就是这个性质的量化——它也是阶梯唯一需要知道的关于一个类的东西，这就是为什么其余的差别都可以留在 checkpoint 内部。
    # 解释（v0.1.8 现状）：SWA 已改成活跃槽位里的每请求环形缓冲、随状态快照一起保存，不再单独成类（block_manager.py 构造 state_caches 处的注释）；
    #   GDN 在 Qwen3-Next / 3.5 默认（PP=1、非 RapidServe）也走 copy 而不是 fork（gdn_attn.py _uses_paged_checkpoints），
    #   所以 state_caches 实际只有一个成员，"多个类迭代到不动点"的写法是为将来留的

    #: False when this class has nothing to say about any seq (not sized, or
    #: prefix caching off). Callers still invoke the methods — they are
    #: identity/no-op — so no `if enabled` appears at the call sites.
    # 【译】当这个类对任何 seq 都无话可说时为 False（没有分配大小，或前缀缓存关闭）。调用方照样调用这些方法——
    #   它们是恒等 / 空操作——所以调用处不会出现 if enabled。
    enabled: bool

    #: Tokens the forward *after* a checkpoint must carry for that checkpoint to
    #: come out whole. Three regimes, one comparison:
    #:
    #:   0     immutable or copyable — nothing is handed over, so no successor
    #:         is needed.
    #:   n>0   rolling — the successor has to refill the replacement group, and
    #:         this is how many committed tokens that takes.
    #:   inf   the class cannot be checkpointed at all, so no position ever
    #:         qualifies. Distinct from 0, and a backend says which by declaring
    #:         a `StateTransfer` rather than a bare token count.
    # 【译】checkpoint 之后的那次 forward 必须携带多少个 token，这个 checkpoint 才能完整地产生出来。三种情形，一个比较：
    #     0     不可变或可复制——什么都没交出去，所以不需要后继。
    #     n>0   滚动——后继要把替代的槽位重新填满，这是需要的已提交 token 数。
    #     inf   这个类根本不能做 checkpoint，所以没有任何位置合格。与 0 不同；后端通过声明一个 StateTransfer
    #           而不是一个裸 token 数来说明是哪一种。
    # 用法：BlockManager.checkpointers_at 里的 next_forward_tokens >= successor_room 就是"这个位置能不能保存"的全部判断
    #   （prefill 传 prompt 剩余 token 数，decode 传 1）；checkpoint_limit 把它解成"prompt 里最右一级阶梯"
    # 取值：fork(1) → 1.0；copy → 0.0（PagedStateCheckpointCoordinator 的类属性）；none / replay → inf（state_runtime.py successor_room）
    successor_room: float

    #: Whether this class can snapshot a boundary *inside* a forward instead of
    #: only at the forward's last token.
    #:
    #: True changes where the ladder's cost is paid, not what a checkpoint is
    #: worth. A class that answers false forces the scheduler to end a forward
    #: exactly on each rung (`BlockManager.checkpoint_cut`), so a prompt with N
    #: rungs on it is N shortened prefill chunks; one that answers true reads
    #: each rung out of intermediates its kernel already materialized, so the
    #: same N rungs are N copies inside one full-length forward.
    #:
    #: Independent of `successor_room`. That says what the forward *after* a
    #: checkpoint owes it; this says where within *this* forward it can be taken
    #: at all. A rolling class can be either — GDN forks and is readable — and so
    #: can a copying one, so neither implies the other.
    # 【译】这个类能否在一次 forward"内部"给某个边界拍快照，而不只是在 forward 的最后一个 token 处。
    #
    #   True 改变的是阶梯的代价在哪里付，而不是 checkpoint 值多少。回答 False 的类迫使调度器让 forward 恰好结束在每一级上
    #   （BlockManager.checkpoint_cut），所以一个上面有 N 级阶梯的 prompt 就是 N 个被截短的 prefill chunk；回答 True 的类
    #   从它的 kernel 本来就物化出来的中间结果里读每一级，同样的 N 级变成一次完整长度 forward 里的 N 次拷贝。
    #
    #   与 successor_room 无关。那个说的是 checkpoint"之后"的 forward 欠它什么；这个说的是在"这次" forward 里到底能不能拍。
    #   滚动类可以是任一种——GDN 是 fork 且可读——复制类也一样，所以两者谁也推不出谁。
    # 观察：上面说"GDN forks and is readable"，但 v0.1.8 的 fork GDN 声明 readable_midstep=False
    #   （gdn_attn.py state_transfer 的 docstring 解释了还缺哪些修复）；可读的只有 copy 方式且 ATOM_GDN_STATE_MIDSTEP=1 的 GDN
    readable_midstep: bool

    def applies(self, seq: Sequence) -> bool:
        """Whether this class gates or checkpoints anything for `seq`."""
        # 【译】这个类是否对 seq 有门控或做 checkpoint。
        ...

    def reserve_midstep(
        self, seq: Sequence, positions: list[tuple[int, int]]
    ) -> list[tuple]:
        """Take a destination for each `(position, hash)` the next forward covers.

        Returns `(destination, position, hash)` per reservation made, for the
        runner to write into and `publish_midstep` to file. Empty — not an
        error — for a class that is not `readable_midstep`, so the call site
        needs no branch.

        Best-effort and order-preserving: reservations stop at the first one
        capacity cannot fill, so the earliest position survives a shortage.
        """
        # 【译】为下一次 forward 覆盖到的每个 (position, hash) 拿一个目标。
        #   对每个做成的预留返回 (destination, position, hash)，给 runner 去写、给 publish_midstep 去登记。
        #   对不是 readable_midstep 的类返回空——不是错误——所以调用处不需要分支。
        #
        #   尽力而为、保持顺序：预留在第一个容量填不满的位置处停止，所以短缺时留下的是最早的位置。
        ...

    def publish_midstep(self, reservations: list[tuple], seq: Sequence = None) -> None:
        """File reservations whose bytes the completed forward has now written.

        The other half of `reserve_midstep`, and the split is the point:
        publishing before the forward would index a boundary over bytes nobody
        had written, and a request resuming there would read the destination's
        previous tenant.

        `seq` lets an implementation tell this prompt's own end — the position
        the next turn actually resumes at — from the ladder and demand rungs,
        which only guess. Optional: a class that ranks its checkpoints equally
        ignores it.
        """
        # 【译】登记那些"刚完成的 forward 已经写好了字节"的预留。
        #   它是 reserve_midstep 的另一半，拆开正是要点：在 forward 之前发布，会把一个边界索引到没人写过的字节上，
        #   从那里恢复的请求会读到目标位置上一任住户的东西。
        #
        #   seq 让实现能区分本 prompt 自己的末尾——下一轮真正会从那里恢复的位置——和只是猜测的阶梯与按需位置。
        #   可选：对所有 checkpoint 一视同仁的类可以忽略它。
        ...

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Release reservations whose forward never ran, holding nothing."""
        # 【译】释放那些 forward 从没跑过的预留，它们里面什么都没有。
        ...

    def resumable_hit(
        self,
        seq: Sequence,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Largest boundary `L <= hit` (in blocks) this class can resume from.

        Scanned right-to-left so the hit is cut as little as possible. 0 is
        always valid — a request starting from scratch needs no prior state.
        Identity when the class does not apply.

        `assume_checkpointed` asks the counterfactual instead: the answer this
        class would give if a checkpoint sat at every boundary. Whatever still
        cuts the hit under that assumption is a limit no amount of
        checkpointing can lift, which is what separates reuse worth arranging a
        checkpoint for from reuse that is simply gone.
        """
        # 【译】这个类能从中恢复的最大边界 L <= hit（以 block 计）。
        #   从右往左扫，让命中被砍得尽量少。0 总是合法的——从头开始的请求不需要任何先前状态。类不适用时为恒等。
        #
        #   assume_checkpointed 改问反事实：如果每个边界上都有一个 checkpoint，这个类会给出什么答案。在这个假设下仍然砍掉命中的部分，
        #   是再多 checkpoint 也解除不了的限制——这就把"值得为它安排一个 checkpoint 的复用"和"就是没了的复用"区分开了。
        # 调用方：BlockManager._gated_hit；assume_checkpointed=True 那次来自 _record_checkpoint_demand（两次答案之差就是按需快照的位置）
        # 例：hit=1152 块，快照只在第 1024 块 → 返回 1024；assume_checkpointed=True 时返回 1152（若 fork 测试通过），差 128 块 = 2048 个 token
        ...

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of `boundary_blocks`, filed under hash `h`.

        Called only at a ladder position `BlockManager` has already vetted
        against `successor_room`. Best-effort: a class out of room keeps
        nothing, and the hit it later declines is the only consequence.
        """
        # 【译】把 seq 在 boundary_blocks 处的状态保存下来，登记在 hash h 名下。
        #   只在 BlockManager 已经对照 successor_room 审核过的阶梯位置上调用。尽力而为：没有空间的类什么都不保存，
        #   唯一的后果是它以后会拒掉某次命中。
        ...


class StateCheckpointCache(StateCache, Protocol):
    """State cache lifecycle owned by BlockManager."""

    # 【译】由 BlockManager 持有的状态缓存生命周期（在 StateCache 之上多出索引维护与统计三个方法）。
    #   unindex(h)：h 对应的 KV block 被淘汰，同 hash 的快照作废（BlockManager._record_evicted 调用）
    #   clear_index()：清空全部快照索引（BlockManager.clear_cache）
    #   checkpoint_fates()：kept / dropped / evicted / orphaned 等累计计数（BlockManager.state_checkpoint_fates）
    # BlockManager._state_checkpoint_cache 的类型注解就是它；"..." 是 Protocol 方法体的惯用写法（只声明不实现）

    def unindex(self, h: int) -> None: ...

    def clear_index(self) -> None: ...

    def checkpoint_fates(self) -> dict[str, int]: ...
