# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import array
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Any, NamedTuple

import numpy as np

from atom.sampling_params import SamplingParams

# 本文件定义一条请求在引擎内部的表示：Sequence
# 对比nano-vllm：nano-vllm的Sequence只有十几个字段，ATOM的有上百个，
#   多出来的字段几乎都服务于某个特性：投机解码（MTP/DSpark）、延迟输出、PD分离、KV offload、
#   混合注意力模型的递归状态（GDN/Mamba）、状态checkpoint、多模态、n>1扇出、DP路由等
#   读第1批时只需掌握"主干字段"（token、计数、block_table、采样参数、状态），其余字段遇到对应批次再回看
# 另一个区别：nano-vllm的rank 0会把Sequence传给其他TP rank；ATOM里Sequence只在
#   LLMEngine -> CoreManager -> EngineCore（调度器）之间传递，ModelRunner拿到的是调度器打包好的ScheduledBatch
#
# 本文件里的几个辅助结构：
#   new_token_ids / BlockTable：token id和block id都用array("i")而不是list，原因见各自说明
#   Replay：DeepSeek-V4等模型的"有界重放"记录，见第4、8批
#   SequenceStatus / SequenceType：序列的生命周期状态、本step做prefill还是decode
#   OffloadJointRecord：KV offload联合加载的协议状态，见第9批


# 创建存放token id的数组：array("i")，即C语言int32的紧凑数组
# 对比nano-vllm：nano-vllm用list；ATOM改用array是为了长上下文下的拷贝速度和内存，详见下方docstring
def new_token_ids(token_ids=()) -> array.array:
    """A sequence's token ids.

    An `array("i")` rather than a list for two reasons that both scale with
    context. The scheduler copies a slice of this into the flat
    `scheduled_tokens` buffer every step -- a whole chunk of it on a prefill --
    and from a list that is one CPython int unboxing per token, 0.28 ms for a
    16k chunk against 0.001 ms from an array. And a list of 100k ids costs
    3.4 MiB of boxed ints where the array costs 0.38.

    Behaves as a list for append/pop/index/len/iterate/slice-delete, and holds
    negative ids (the exit sentinel is -1). It does NOT compare equal to a
    list, which is why `stop_token_sequences` below is converted too, and why
    `BlockManager.compute_hash` pins its dtype.
    """
    # 【译】一条序列的token id。
    #
    #   用 array("i") 而不是 list，有两个随上下文长度放大的原因。调度器每个step都要把它的一段
    #   拷进扁平的 scheduled_tokens 缓冲区（prefill时是一整个chunk），从list拷贝时每个token都要
    #   做一次CPython int拆箱：16k的chunk要0.28ms，而从array拷贝只要0.001ms。另外，10万个id的list
    #   要占3.4MiB的装箱int，array只要0.38MiB。
    #
    #   它在 append/pop/index/len/迭代/切片删除 上的行为和list一样，并能存负数id（退出哨兵是-1）。
    #   它与list比较永远不相等，这就是为什么下面的 stop_token_sequences 也要转换，
    #   以及为什么 BlockManager.compute_hash 要固定它的dtype。
    return array.array("i", token_ids)


# Drawn once per table and once per non-append mutation of one, never reused.
# A `version` therefore names an epoch during which one specific table only
# ever grew, which is what `block_table_codec` needs to ship a forward's block
# tables as appends alone -- see `BlockTable`.
# 【译】每张表创建时取一次，每次对表做非追加修改时再取一次，从不复用。
#   因此一个 version 标识了一段"某张表只增长过"的时期，这正是 block_table_codec 只用追加量
#   就能发送一次forward的block table所需要的——见 BlockTable。
# 全局版本号发生器：itertools.count()每次next()返回0、1、2...，全局唯一、永不重复
_block_table_versions = count()


# 一条序列的block table：逻辑block -> 物理block id的映射（同nano-vllm，类似操作系统的页表）
#   例如[5, 9, 12]：第0~15个token的KV在物理block 5，第16~31个在block 9，以此类推（block_size=16）
# 对比nano-vllm：nano-vllm用list，每步把整张表重新发给其他rank；ATOM做了两点优化
#   1. 继承array("i")：拼进int32缓冲区时是一次内存拷贝，不用逐个拆箱
#   2. 带一个version字段，配合block_table_codec只把"新追加的block"发给worker进程，而不是整张表
# version示例：
#   第1步：bt=[5,9]，version=7         -> 发送完整表[5,9]，接收方记下(version=7, 长度2)
#   第2步：bt.append(12)，version仍为7  -> 接收方发现版本相同，只需发送[12]
#   第3步：被抢占，del bt[:]，version=8 -> 版本变了，下次重新发送完整表
class BlockTable(array.array):
    """A sequence's physical block ids, tagged with a mutation `version`.

    An `array("i")` rather than a list because every forward marshals these
    into the int32 `block_tables` buffer, where a list costs one CPython int
    unboxing per block (~17k per step at 50 seqs x 100k ctx) and an array is a
    memcpy. It behaves as a list for append/pop/index/len/iterate; it has no
    `.clear()` (use `del bt[:]`) and no `.copy()` (use `list(bt)`).

    `version` is what makes the table's own growth self-describing: it is
    redrawn by every mutation that is *not* an append, so

        same version and a length of at least n  =>  the first n ids are the
        same ids they were when that version was last observed

    with no need to re-read the prefix. `BlockTableDeltaEncoder` turns that
    into "send only the ids appended since the last step". Appends deliberately
    do not redraw it, so the hot path (`BlockManager.allocate` appending one
    block per decode step) keeps the inherited C `array.append` with no Python
    frame on top -- 40 ns against 36 for a plain `array`, which is the subclass
    method lookup and nothing else.

    A version is a global draw rather than a per-table counter so that a fresh
    table for a recycled request id can never look like a known one:
    independent tables never share a version. The RPC decoder may retain
    a version when it copies a row to append: those immutable snapshots share
    one prefix lineage, and equal lengths still identify identical contents.
    """

    # 【译】一条序列的物理block id，带一个修改版本号 version。
    #
    #   用 array("i") 而不是 list，是因为每次forward都要把它们整理进int32的 block_tables 缓冲区：
    #   list每个block都要一次CPython int拆箱（50条序列 x 10万上下文时每步约1.7万次），array只是一次memcpy。
    #   它在 append/pop/index/len/迭代 上的行为和list一样；没有 .clear()（用 del bt[:]），
    #   也没有 .copy()（用 list(bt)）。
    #
    #   version 让表自身的增长可以自我描述：每次"不是追加"的修改都会重新取一个版本号，所以
    #
    #       版本号相同且长度至少为n  =>  前n个id与上次观察到该版本时完全相同
    #
    #   不需要重新读取前缀。BlockTableDeltaEncoder 据此做到"只发送上一步以来新追加的id"。
    #   追加操作故意不重新取版本号，这样热路径（BlockManager.allocate 每个decode step追加一个block）
    #   仍使用继承来的C实现 array.append，上面没有Python栈帧——40ns，普通array是36ns，
    #   差别只是子类的方法查找。
    #
    #   版本号是全局取号而不是每张表各自计数，这样为复用的请求id新建的表永远不会看起来像一张已知的表：
    #   相互独立的表从不共享版本号。RPC解码器在复制一行用于追加时可能保留版本号：
    #   这些不可变快照共享同一条前缀血缘，长度相同仍然意味着内容相同。

    # __slots__：只允许实例拥有version这一个额外属性，不创建__dict__，节省内存
    __slots__ = ("version",)

    # array是C实现的不可变类型构造方式：元素类型"i"必须在__new__里传给父类
    def __new__(cls, block_ids=()):
        return super().__new__(cls, "i", block_ids)

    # 每张新表都领取一个全局唯一的版本号
    def __init__(self, block_ids=()):
        self.version = next(_block_table_versions)

    # A copy is a different table and must draw its own version, or two tables
    # would answer for one. All three hooks are needed: `array` supplies its
    # own `__copy__`/`__deepcopy__`, which hand back a bare `array` and drop
    # the version silently, and its reconstructor leaves the attribute unset,
    # which surfaces much later as an AttributeError in whoever reads it.
    # 【译】拷贝出来的是另一张表，必须取自己的版本号，否则两张表会对应同一个版本号。
    #   三个钩子都需要：array自带 __copy__/__deepcopy__，它们返回一个普通array并悄悄丢掉version；
    #   而它的重建函数不会设置这个属性，问题要到很久以后才在读取它的地方以AttributeError暴露。

    # pickle序列化时调用：返回(重建函数, 参数)，反序列化时执行_rebuild_block_table(字节串)
    # Sequence通过ZMQ在进程间传递时会被pickle，block_table也随之序列化
    def __reduce_ex__(self, protocol):
        return _rebuild_block_table, (self.tobytes(),)

    # copy.copy(bt)时调用
    def __copy__(self):
        return _rebuild_block_table(self.tobytes())

    # copy.deepcopy(bt)时调用
    def __deepcopy__(self, memo):
        return _rebuild_block_table(self.tobytes())


# 从原始字节重建BlockTable：新建空表（领取新版本号），再把字节填进去
def _rebuild_block_table(raw: bytes) -> "BlockTable":
    table = BlockTable()
    table.frombytes(raw)
    return table


# 包装array的某个修改方法：先换新版本号，再调用原方法
def _revises_block_table(name: str) -> Callable:
    """Wrap `array.array.<name>` so it redraws the table's version first."""
    # 【译】包装 array.array.<name>，使它先为表重新取一个版本号。
    inherited = getattr(array.array, name)

    def revise(self, *args):
        self.version = next(_block_table_versions)
        return inherited(self, *args)

    revise.__name__ = name
    revise.__qualname__ = f"BlockTable.{name}"
    revise.__doc__ = f"`array.array.{name}`, plus a new version."
    return revise


# Every `array` mutator that can disturb ids already in the table. The ones
# left out -- `append`, `extend`, `frombytes`, `fromfile`, `fromlist` and
# `__iadd__` -- only ever add past the end, which is exactly the case the
# version is defined to survive. Assigned in a loop so the reviewable artifact
# is this list rather than eight near-identical bodies.
# 【译】所有可能改动表中已有id的array修改方法。没有列出的 append、extend、frombytes、fromfile、
#   fromlist 和 __iadd__ 只会在末尾之后添加，这恰好是版本号被定义为可以保持不变的情况。
#   用循环赋值，使需要审查的内容是这张列表，而不是八个几乎一样的函数体。
for _mutator in (
    "__setitem__",
    "__delitem__",
    "__imul__",
    "byteswap",
    "insert",
    "pop",
    "remove",
    "reverse",
):
    setattr(BlockTable, _mutator, _revises_block_table(_mutator))
del _mutator


def new_block_table(block_ids=()) -> BlockTable:
    """A sequence's physical block ids. See `BlockTable`."""
    # 【译】一条序列的物理block id。见 BlockTable。
    return BlockTable(block_ids)


# 【分支】有界重放（bounded replay），主要用于DeepSeek-V4等带滑动窗口/压缩状态的模型，见第4、8批
# 背景：前缀缓存命中后，KV可以直接复用，但有些模型还有"状态"（如滑动窗口、压缩器状态），
#   命中点上如果没有保存状态checkpoint，就要从命中点往前退一段重新算（重放），把状态重建出来
#   一条Replay记录描述这样一次prefill，四个字段的含义见下方docstring的翻译
# NamedTuple：带字段名的不可变元组，既能r.until访问，也能像元组一样解包
class Replay(NamedTuple):
    """A prefill forward starting below the point its seq resumed at: that
    point -- the cache holds every row below it --, whether the rows below are
    the seq's state (a `replay` state transfer brought none; otherwise the
    state is the one at `until`), the tokens just before the forward, for
    state a backend derives from the preceding tokens (V4.1's Engram n-gram
    history, `Config.replay_history_tokens` of them), and where the replay's
    first forward started: a DeepSeek-V4 bounded replay may span forwards, and
    no row below that start is in the state it rebuilds. `Scheduler.
    _resume_prefill` always records it; 0 (no floor) is for a record built
    elsewhere, which no V4 forward reads."""

    # 【译】一次从序列恢复点下方开始的prefill forward。它记录：
    #   until：该恢复点——cache保存了它下方的每一行；
    #   rebuilds：下方的行是否属于序列的状态（replay方式的状态传输不带来任何状态；否则状态就是until处的状态）；
    #   history_ids：forward之前紧邻的token，供backend从前文推导的状态使用
    #     （V4.1的Engram n-gram历史，共 Config.replay_history_tokens 个）；
    #   start：重放的第一个forward从哪里开始——DeepSeek-V4的有界重放可能跨多个forward，
    #     该起点以下的行都不在它重建的状态中。
    #   Scheduler._resume_prefill 总会记录它；0（没有下界）用于在别处构造的记录，V4的forward不会读取它。

    until: int
    rebuilds: bool
    history_ids: tuple[int, ...]
    start: int = 0


# 序列的生命周期状态
# auto()：自动给枚举成员分配值（1、2、3...），只关心名字不关心值时使用
# 状态流转图（主干是中间一行，其余为分支）：
#
#                       add_request
#                ┌───────────┴────────────┐
#   PD分离decode端│                        │普通请求
#                ▼                        ▼
#   WAITING_FOR_REMOTE_KVS ──KV到达──> WAITING ──被调度做prefill──> RUNNING ──EOS/max_tokens/stop──> FINISHED
#                                       ▲                            │
#                                       └────── 被抢占(preempt) ─────┘
#
# 对比nano-vllm：nano-vllm要等prefill全部算完才变成RUNNING；
#   ATOM在第一次被调度做prefill时就变成RUNNING并进入running队列，
#   chunked prefill没算完的序列留在running里（is_partial_prefill=True），下一步由调度器的Phase 1继续
# 客户端断开：WAITING/RUNNING -> ABORTED，调度器在下一步按正常结束路径释放资源 -> FINISHED
class SequenceStatus(Enum):
    # 【分支】PD分离的decode端：等待prefill节点把KV传过来（KV offload从外部加载时也会停在这里），见第9批
    WAITING_FOR_REMOTE_KVS = auto()
    # 在等待队列中，还没开始prefill（新请求、或被抢占后重新排队）
    WAITING = auto()
    # 在running队列中：正在prefill（可能分多个chunk）或decode
    RUNNING = auto()
    # Client disconnected: the seq is still live (its KV must be freed via the
    # normal stop path). The scheduler finishes it at the next step (running) or
    # drops it when popped from `waiting`. Distinct from FINISHED so it still
    # rides one cleanup pass; is_finished() stays False until then.
    # 【译】客户端断开：序列仍然存活（它的KV必须走正常的停止路径释放）。调度器会在下一步结束它
    #   （若在running中），或在它从waiting中取出时丢弃。与FINISHED区分开，是为了让它仍经过一次
    #   清理流程；在那之前 is_finished() 一直是False。
    ABORTED = auto()
    # 已结束，KV block已释放
    FINISHED = auto()
    # 不是真实请求：一个哨兵状态，放进EngineCore的队列，通知读队列的一方停止，见get_exit_sequence
    EXIT_ENGINE = auto()


# 本step这条序列做的是prefill还是decode，由调度器在每次调度时设置
# ScheduledBatch据此区分batch中的prefill序列和decode序列
class SequenceType(Enum):
    # 初始值：还没被调度过
    DUMMY = auto()
    PREFILL = auto()
    DECODE = auto()


# 构造退出哨兵，EngineCore在两处使用（第2批）：
#   输入线程收到SHUTDOWN命令 -> 放进input_queue -> 主循环取到后退出busy_loop
#   引擎退出时 -> 放进output_queue -> 输出线程据此结束，不会把它当作结果发回
def get_exit_sequence():
    exit_seq = Sequence([-1], 1)
    exit_seq.status = SequenceStatus.EXIT_ENGINE
    return exit_seq


# 【分支】KV offload，见第9批：offload加载已经结束且序列保留了block时返回True，
#   这样的请求直接从后缀继续prefill，不再重新匹配前缀
def resumes_offload_prefill(seq) -> bool:
    """True when an offload load already settled and `seq` keeps its blocks.

    Such a request resumes its suffix prefill without matching the prefix
    again. Shared by the scheduler's admission and the offload connector's
    lookup prefetch, which must agree: a lookup sent for a request that skips
    the match is never consumed and only holds read locks.
    """
    # 【译】当offload加载已经结束、且 seq 保留着它的block时返回True。
    #   这样的请求直接恢复后缀的prefill，不再重新匹配前缀。调度器的准入和offload connector的
    #   查询预取共用这个判断，两者必须一致：为跳过匹配的请求发出的查询永远不会被消费，只会白白占着读锁。
    return (
        getattr(seq, "offload_loaded", False)
        or getattr(seq, "offload_load_failed", False)
    ) and len(seq.block_table) > 0


# 【分支】KV offload的"联合加载"协议状态，见第9批
# 背景：混合模型的前缀复用需要两条腿同时到位：KV（从LMCache等外部存储加载）+ 状态（从状态层加载），
#   两者必须对齐到同一个边界位置，这个类记录这次联合加载的边界和各段范围
# 第1批只需知道它属于offload分支，字段细节读第9批时再细看
@dataclass
class OffloadJointRecord:
    """The KV-transfer offload/joint-load protocol state for one sequence.

    These six values were six flat attributes on `Sequence`, written from
    `BlockManager` and `Scheduler` and read back by the offload connector, with
    the joint subset re-initialised by hand in several places -- so a reset that
    forgot one field left a stale span the connector would then act on. Grouped
    here so the whole protocol state is one field (`Sequence.offload_joint`) and
    the joint boundary has one place to be dropped from (`reset_joint`).
    """

    # 【译】一条序列的KV传输offload/联合加载协议状态。
    #
    #   这六个值原来是 Sequence 上的六个散属性，由 BlockManager 和 Scheduler 写入、由offload connector
    #   读回，其中联合加载相关的几个要在好几处手动重新初始化——于是某次重置漏掉一个字段，
    #   就会留下一段过期的范围，connector随后还会据此行动。收拢到这里后，整个协议状态是一个字段
    #   （Sequence.offload_joint），联合边界也只有一个地方可以丢弃（reset_joint）。

    # Content hash of a checkpoint the tier was asked to fetch back into
    # `state_slot`, or -1. Tells the scheduler to park, and tells the failure
    # path that `num_cached_tokens` is claiming undelivered state.
    # 【译】已请求存储层取回到 state_slot 的checkpoint的内容hash，没有则为-1。它告诉调度器要把请求
    #   挂起等待，也告诉失败处理路径：num_cached_tokens 声称拥有的状态其实还没送达。
    load_hash: int = -1
    # The LMCache-resident KV prefix in tokens, as this admission's lookup
    # reported it -- the KV leg's ceiling. Written before `can_allocate`, which
    # is where the two legs agree on one boundary. 0 = no lookup.
    # 【译】本次准入查询报告的、驻留在LMCache中的KV前缀长度（token数），即KV这条腿的上限。
    #   在 can_allocate 之前写入，两条腿就是在 can_allocate 里商定同一个边界的。0 表示没有查询。
    kv_prefix_tokens: int = 0
    # The LMCache-resident KV prefix with no regard for window state, in
    # tokens: how far the full-attention KV alone reaches, at or past
    # `kv_prefix_tokens`. Asked only where a bounded replay could use it
    # (`Sequence.replay_kv_load`). 0 = not asked or not answered.
    # 【译】不考虑窗口状态时，驻留在LMCache中的KV前缀长度（token数）：仅全注意力KV能覆盖到多远，
    #   不小于 kv_prefix_tokens。只在有界重放可能用到时才查询（Sequence.replay_kv_load）。
    #   0 表示没问或没有答复。
    kv_only_tokens: int = 0
    # The boundary both legs of a joint load are aimed at, or 0. Chosen by
    # `can_allocate`: the rightmost checkpoint rung the LMCache KV prefix
    # covers. `num_cached_tokens` stays at the HBM prefix until both legs
    # report, which keeps the forward honest if either fails.
    # 【译】联合加载两条腿共同瞄准的边界，没有则为0。由 can_allocate 选定：LMCache KV前缀所覆盖的
    #   最右侧checkpoint档位。num_cached_tokens 停在HBM前缀处，直到两条腿都报告完成；
    #   这样任何一条腿失败，forward用到的也都是真实存在的数据。
    boundary_tokens: int = 0
    # Content hash of that boundary's last block, for the state leg.
    # 【译】该边界最后一个block的内容hash，供状态这条腿使用。
    boundary_hash: int = -1
    # How far the KV leg transfers, in tokens: the LMCache chunk that *covers*
    # the boundary above, which is at or past it. The request still only claims
    # the boundary -- see `_claim_after_load`.
    # 【译】KV这条腿传输到多远（token数）：覆盖上述边界的那个LMCache chunk，位于边界处或之后。
    #   请求仍然只认领到边界为止——见 _claim_after_load。
    kv_tokens: int = 0
    # How far `allocate` claimed straight out of the HBM prefix cache, in
    # tokens. Above `num_cached_tokens`, not below it: the blocks in
    # `(hit, compressed_hit]` hold this prompt's KV and are still indexed --
    # the state gate cut the hit, the KV never went anywhere. Claiming them is
    # what keeps the KV leg from paying LMCache to resend what the pool already
    # has, and it is where that leg starts.
    # 【译】allocate 直接从HBM前缀缓存中认领到多远（token数）。它在 num_cached_tokens 之上而不是之下：
    #   (hit, compressed_hit] 区间的block保存着这个prompt的KV，并且仍在索引中——是状态门控截短了命中，
    #   KV本身并没有丢。认领它们，KV这条腿就不必花钱让LMCache重发池里已有的内容；这里也是那条腿的起点。
    claim_tokens: int = 0

    def reset_joint(self) -> None:
        """Drop the joint boundary and both of its transfer spans.

        The four joint fields move together -- a boundary is meaningless
        without the KV span aimed at it and the prefix claimed below it -- so
        the sites that abandon a joint load in full clear them through here
        rather than by hand. Leaves `load_hash` and `kv_prefix_tokens`
        untouched: they are set on their own paths and are not part of the
        boundary this drops.
        """
        # 【译】丢弃联合边界及其两段传输范围。
        #   这四个联合字段总是一起变化——没有瞄准它的KV范围和它下方认领的前缀，边界就没有意义——
        #   所以完全放弃联合加载的地方都通过这里清空它们，而不是手动清。
        #   不动 load_hash 和 kv_prefix_tokens：它们在各自的路径上设置，不属于这里丢弃的边界。
        self.boundary_tokens = 0
        self.boundary_hash = -1
        self.kv_tokens = 0
        self.claim_tokens = 0


# 一条请求在引擎内部的表示：保存token、计数、KV block映射、采样参数和各特性的状态
# 由InputOutputProcessor.preprocess创建（第2批），之后由调度器（第3批）和BlockManager（第4批）读写
#
# 主干计数字段的变化示例（block_size=16，prompt长40，前缀缓存命中1个block，每步token预算16）：
#   创建：              num_tokens=40，num_prompt_tokens=40，num_cached_tokens=0，block_table=[]
#   第1次调度：          命中1个block -> num_cached_tokens=16；分配block_table=[7,3,9]（16+16+8=40个token）
#                       本次只算16个token（预算限制，chunked prefill），状态变为RUNNING，进入running队列
#   第1次postprocess：  num_cached_tokens=32，is_partial_prefill=True，这一步采样出的token无意义，丢弃
#   第2次调度：          算剩余8个token
#   第2次postprocess：  num_cached_tokens=40，prefill完成，追加第1个生成token -> num_tokens=41
#   之后每次decode：     每步追加1个token（投机解码时可能一步追加多个）
#
# 延迟输出（deferred output，第5批详解）对这些字段的影响：
#   GPU算完第N步之前，CPU就要调度第N+1步，此时第N步采样出的token还不知道，
#   postprocess先往token_ids末尾追加占位token（值为eos_token_id），等下一步拿到真实token后原地覆盖
#   num_placeholder_tokens记录末尾有几个这样的占位，抢占时要先把它们删掉
class Sequence:
    # 类变量，所有实例共享：全局自增器，为每条序列分配唯一id
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        sampling_params: SamplingParams | None = None,
        stop_token_sequences: list[list[int]] | None = None,
        stream_callback: Callable[[Any], None] | None = None,
        id=None,
        kv_transfer_params: dict | None = None,
        num_draft_tokens: int = 0,
        has_per_req_cache: bool = False,
        needs_independent_noise: bool = False,
        parent_request_id: str | None = None,
        sibling_index: int = 0,
        request_id: str | None = None,
        multimodal_data: dict | None = None,
        mrope_positions: np.ndarray | None = None,
        mrope_position_delta: int = 0,
        data_parallel_rank: int | None = None,
        dp_session_id: str | None = None,
        dp_parent_session_id: str | None = None,
    ):
        # Built here rather than as a default argument: one instance shared by
        # every defaulting Sequence would be a mutable default in all but name.
        # 【译】在这里创建，而不是写成参数默认值：若所有使用默认值的 Sequence 共享同一个实例，
        #   那它实际上就是一个可变默认参数。
        # 对比nano-vllm：nano-vllm直接写sampling_params=SamplingParams()作为默认参数，所有序列共享一个实例
        if sampling_params is None:
            sampling_params = SamplingParams()
        # 投机解码时，一步会验证多个草稿token，目前不支持为它们返回logprob
        if num_draft_tokens and sampling_params.logprobs:
            raise ValueError(
                "Token logprobs are not supported with speculative decoding"
            )
        # ---------------- 主干字段：身份、状态、token ----------------
        # 每个KV block容纳的token数，用于计算num_blocks和last_block_num_tokens
        # 对比nano-vllm：nano-vllm是类变量（所有序列共用），ATOM由调用方传入
        self.block_size = block_size
        # 引擎内部的序列id；调用方可以指定（例如n>1扇出时），否则自增
        self.id = id or next(Sequence.counter)
        # 用户侧的请求id（API层的request_id），用于日志和追踪
        self.external_request_id = request_id
        # 新序列进入等待队列
        self.status = SequenceStatus.WAITING
        # 本step做prefill还是decode，由调度器设置
        self.type = SequenceType.DUMMY
        # 全部token id：prompt + 已生成的token（array("i")，见new_token_ids）
        self.token_ids = new_token_ids(token_ids)
        # 最后一个token，decode时作为模型输入
        self.last_token = token_ids[-1]
        # 【分支】投机解码：每步的草稿token数（MTP的k），0表示不开投机解码，见第7批
        self.num_draft_tokens = num_draft_tokens
        # `has_per_req_cache=True` means this seq's attention type maintains
        # a per-request stateful buffer outside the paged KV pool (e.g. GDN
        # recurrent state, future DeepseekV4 ring-buffer + compressor state).
        # Triggers BlockManager to allocate a per-req cache slot in
        # allocate() / free it in deallocate().
        # 【译】has_per_req_cache=True 表示这条序列的注意力类型在分页KV池之外，还维护一份每个请求的
        #   有状态缓冲区（例如GDN的递归状态，以及将来DeepseekV4的环形缓冲区+压缩器状态）。
        #   它会让 BlockManager 在 allocate() 中分配一个每请求缓存槽位，在 deallocate() 中释放。
        # 解释：这类状态大小固定、不随token数增长，存在单独的状态池里，见第4批
        self.has_per_req_cache = has_per_req_cache
        # 【分支】多模态（图像等）输入，见第2批2b
        self.multimodal_data = multimodal_data
        self.multimodal_cache_ready = False
        # Immutable content identity survives payload release and preemption.
        # 【译】不可变的内容标识，在多模态数据被释放和序列被抢占后仍然保留。
        # 解释：它是多模态内容的指纹，参与前缀缓存的hash，保证"同样的文本+不同的图片"不会误命中
        self.cache_seed = -1
        if multimodal_data is not None:
            from atom.model_engine.multimodal_runtime import multimodal_cache_seed

            self.cache_seed = multimodal_data.get("cache_seed")
            if self.cache_seed is None:
                self.cache_seed = multimodal_cache_seed(multimodal_data)
        # 【分支】Qwen3.5等多模态模型的M-RoPE位置（文本/图像的三维位置编码），见第8批
        self.mrope_positions = mrope_positions
        self.mrope_position_delta = mrope_position_delta
        # 当前总token数（prompt + 已生成）
        # 注意它是property：赋值时会顺带更新num_blocks和last_block_num_tokens，见下方setter
        self.num_tokens = len(self.token_ids)
        # prompt的token数，用来区分prompt和生成部分
        self.num_prompt_tokens = len(token_ids)
        # Host-known prefix, excluding deferred outputs and draft placeholders.
        # Read by the DSpark state audit and the runtime benchmark; the width
        # preemption strips is `num_placeholder_tokens`, which is recorded
        # where the placeholders are appended.
        # 【译】主机侧已确定的前缀长度，不含延迟输出和草稿占位。DSpark状态审计和运行时benchmark会读取它；
        #   抢占时要删掉的宽度是 num_placeholder_tokens，它在追加占位的地方记录。
        self.num_finalized_tokens = len(token_ids)
        # Initial local prefill telemetry; kept across preemption/recomputation.
        # 【译】首次本地prefill的遥测数据；在抢占/重算后仍保留。
        self.prefill_gpu_chunks = 0
        self.prefill_gpu_complete = False
        # 【分支】投机解码：上一步被拒绝的草稿token数，见第7批
        self.num_rejected = 0
        # 已算好KV并写入KV cache的token数（同nano-vllm）
        # 来源：前缀缓存命中的部分 + 本序列各step计算过的部分
        # 一般在Scheduler.postprocess里累加；流水线并行时在调度时就累加（第9批）
        self.num_cached_tokens = 0
        # The prefill forward scheduled last, where it backed off below the
        # point it resumed at (`BlockManager.replay_start`); None when it did
        # not.
        # 【译】上一次调度的prefill forward：若它退到恢复点下方开始（BlockManager.replay_start），
        #   记录在这里；没有后退时为None。
        self.replay: Replay | None = None
        # Tokens whose blocks are registered in the prefix cache: through the
        # prompt as chunks finalize, then on through decode as generated blocks
        # fill (BlockManager.hash_decode_blocks). Distinct from
        # `num_cached_tokens`, which means "KV computed" and stops at the
        # prompt; this means "content hash published". They part ways the moment
        # generation starts.
        # 【译】其block已登记到前缀缓存的token数：随着chunk完成覆盖整个prompt，之后在decode中随着生成的block
        #   填满继续增长（BlockManager.hash_decode_blocks）。它与 num_cached_tokens 不同：后者表示
        #   "KV已算好"且止于prompt末尾；这个表示"内容hash已发布"。一开始生成，两者就分道扬镳。
        self.num_hashed_tokens = 0
        # 以下到prefix_cache_hit_tokens为止，都是前缀缓存和状态checkpoint的细节，第4批详解
        # 压缩命中的block数（DeepSeek-V4等压缩KV的模型）
        self.num_compressed_hit_blocks = 0
        # The same hit asked counterfactually: how far it would have reached
        # with a state checkpoint at every boundary. Equal to the admitted hit
        # when nothing was lost to a missing checkpoint, so the difference is
        # the reuse a checkpoint would have delivered — what the EngineStats
        # cache section reports as recoverable.
        # 【译】同一次命中的反事实版本：如果每个边界都有状态checkpoint，命中能延伸到多远。
        #   没有因为缺checkpoint而损失时，它等于实际准入的命中，所以两者之差就是checkpoint本可以带来的复用
        #   ——也就是 EngineStats 缓存部分报告的"可恢复"量。
        self.num_wanted_hit_blocks = 0
        # DeepSeek-V4 bounded replay: whether the hit `BlockManager.can_allocate`
        # admitted last has no checkpoint behind its state, so its prefill
        # starts `Config.state_replay_tokens` below it and rebuilds the state
        # there (`BlockManager.replay_start`; the forward is a `Replay`).
        # 【译】DeepSeek-V4有界重放：BlockManager.can_allocate 最近准入的命中，其状态背后是否没有checkpoint，
        #   因此它的prefill要从命中点下方 Config.state_replay_tokens 处开始，在那里重建状态
        #   （BlockManager.replay_start；对应的forward是一个 Replay）。
        self.replays_hit = False
        # That replay ends past the HBM hit, at LMCache MP's KV-only reach
        # (`OffloadJointRecord.kv_only_tokens`): the connector loads that KV
        # alone first, and once it lands the forward replays below the reach.
        # 【译】该重放终止于HBM命中之后、LMCache MP仅靠KV能到达的位置（OffloadJointRecord.kv_only_tokens）：
        #   connector先单独加载这段KV，加载完成后，forward在该位置下方重放。
        self.replay_kv_load = False
        # That gap as a prompt position, once it is worth a forward: the one
        # place off the checkpoint grid where this seq's prefill is cut so a
        # checkpoint can be kept. 0 = nowhere. Both written by
        # `BlockManager._record_checkpoint_demand` at admission; this one is
        # read by `checkpoint_cut` and `checkpointers_at`, which must agree.
        # 【译】当这个缺口值得单独做一次forward时，用prompt位置表示它：这是checkpoint网格之外唯一一处
        #   会切断该序列prefill以便保留checkpoint的位置。0表示没有。两者都由准入时的
        #   BlockManager._record_checkpoint_demand 写入；这个由 checkpoint_cut 和 checkpointers_at 读取，
        #   两者必须一致。
        self.checkpoint_demand_pos = 0
        # Which of the two demand counters this seq has already been put
        # against. `can_allocate` re-runs for a sequence the queue keeps
        # deferring, and the position above cannot serve as the marker for
        # either one: a declined demand writes 0 back, so it does not remember
        # the decline, and a decline retracts a demand the recorded counter had
        # already taken. Both are cleared by `deallocate`, so a re-admitted
        # request counts again, which it should.
        # 【译】这条序列已经计入了两个需求计数器中的哪一个。队列一直推迟的序列会反复执行 can_allocate，
        #   而上面的位置不能作为任一计数器的标记：被拒绝的需求会写回0，因此记不住拒绝；而拒绝又会撤回
        #   已被记录计数器计入的需求。两者都由 deallocate 清除，所以重新准入的请求会再计数一次，这是应该的。
        self.checkpoint_demand_counted = False
        self.checkpoint_demand_declined = False
        # The demand's sibling: this prompt's own end, floored to the hash
        # grid. 0 = nowhere.
        #
        # The demand is reactive — it only exists once a hit has already been
        # refused for want of a checkpoint, which is one request too late for
        # the position that serves the *next* turn of a conversation. On
        # agentic traffic that position is where nearly all the reuse is (see
        # `BlockManager._record_checkpoint_end`), so it is reserved up front
        # rather than waited for.
        #
        # Written by `BlockManager._record_checkpoint_end` at admission, read
        # by `checkpoint_cut` and `checkpointers_at` — which must agree, the
        # same contract `checkpoint_demand_pos` is held to.
        # 【译】需求位置的"兄弟"：这个prompt自身的末尾，向下对齐到hash网格。0表示没有。
        #
        #   需求是被动产生的——只有当某次命中因为缺checkpoint已经被拒绝后它才存在，
        #   而对于服务对话"下一轮"的那个位置来说，这已经晚了一个请求。在agent类流量中，
        #   几乎所有复用都发生在这个位置（见 BlockManager._record_checkpoint_end），所以提前预留，而不是等待。
        #
        #   由准入时的 BlockManager._record_checkpoint_end 写入，由 checkpoint_cut 和 checkpointers_at 读取
        #   ——两者必须一致，与 checkpoint_demand_pos 遵守同样的约定。
        self.checkpoint_end_pos = 0
        # The chained content hash of every block of this prompt, not just the
        # ones that hit. Empty unless the state backend reserves checkpoints
        # midstep (`StateTransfer.readable_midstep`), which is the only caller
        # that needs to name a position the forward has not reached yet — see
        # `BlockManager._extend_hash_chain` for why it cannot simply be the
        # `block_hashes` the admission scan built.
        # 【译】这个prompt每个block的链式内容hash，而不仅是命中的那些。只有当状态backend在forward中途
        #   预留checkpoint时（StateTransfer.readable_midstep）才非空，这是唯一需要指明forward尚未到达的
        #   位置的调用方——为什么它不能直接用准入扫描构建的 block_hashes，见 BlockManager._extend_hash_chain。
        self.block_hashes: list[int] = []
        # `prefix_hash.ShippedChain`, consumed by
        # `BlockManager._adopt_shipped_chain`; None = the engine hashes it. It
        # rides beside the seq in the ADD frame (`take_shipped_chains`), never
        # inside a pickled seq (`__getstate__`).
        # 【译】prefix_hash.ShippedChain，由 BlockManager._adopt_shipped_chain 消费；None表示由引擎自己计算hash。
        #   它在ADD消息帧里与seq并排传输（take_shipped_chains），从不放在pickle后的seq里（__getstate__）。
        # 解释：前端进程（LLMEngine）分词时顺便算好了prompt的hash链，随请求发给EngineCore，省得重复计算
        self.prompt_hash_chain: tuple | None = None
        # Slots taken for midstep checkpoints of the forward now in flight,
        # as `(slot, position, hash)` — one slot each, since a checkpoint never
        # carries speculation scratch. Filled by `BlockManager.plan_midstep`
        # before the batch is built, drained by `commit_midstep` after it, and
        # handed back by `cancel_midstep` if that forward never runs. Non-empty
        # only between those two points.
        # 【译】为正在执行的forward的中途checkpoint占用的槽位，形如 (slot, position, hash)——每个一个槽位，
        #   因为checkpoint从不携带投机解码的临时数据。在构建batch前由 BlockManager.plan_midstep 填充，
        #   之后由 commit_midstep 清空；如果那次forward没有执行，则由 cancel_midstep 归还。
        #   只在这两个时间点之间非空。
        self.midstep_reservations: list[tuple] = []
        # `checkpoint_end_pos` as the forward holding these reservations files
        # it: moved back onto that forward's snapshot grid. 0 when the forward
        # does not reach the anchor. Read by `StateSlotPool._anchor_of`.
        # 【译】持有这些预留的forward所登记的 checkpoint_end_pos：退回到该forward的快照网格上。
        #   forward到不了这个锚点时为0。由 StateSlotPool._anchor_of 读取。
        self.midstep_anchor_pos = 0
        # Where this seq last kept a checkpoint. Prefill lands on the grid so
        # this tracks it, but a speculative decode step lands wherever
        # `1 + accepted` puts it, and there the grid is unreachable — see
        # `BlockManager.checkpointers_at`, which measures spacing from here.
        # 【译】这条序列上一次保留checkpoint的位置。prefill落在网格上，所以它跟随网格；但投机解码的一步
        #   落在 1 + accepted 所在的任意位置，那里到不了网格——见 BlockManager.checkpointers_at，
        #   它从这里开始计算间距。
        self.last_checkpoint_pos = 0
        # 前缀缓存命中的token数（PD分离时由prefill端通过kv_transfer_params带过来，见下方）
        self.prefix_cache_hit_tokens = 0
        # True iff this seq is mid-prefill (chunked prefill produced KV for
        # some prompt tokens but not all). Maintained by the scheduler:
        # set in postprocess when an advance leaves prompt tokens remaining,
        # cleared when prefill completes or seq is preempted. Used to discard
        # garbage sampled tokens from intermediate chunks and to skip the
        # scheduler's Phase 1 scan when no partials exist.
        # 【译】当且仅当这条序列处于prefill中途时为True（chunked prefill只为部分prompt token算出了KV）。
        #   由调度器维护：postprocess中推进后仍有prompt token剩余时置位，prefill完成或序列被抢占时清除。
        #   用来丢弃中间chunk采样出的无效token，并在没有半途prefill时跳过调度器的Phase 1扫描。
        self.is_partial_prefill = False
        # `new_block_table` is main's: an array("i") rather than a list,
        # because every forward marshals these into the int32 buffer.
        # 【译】new_block_table 来自主线代码：用 array("i") 而不是list，因为每次forward都要把它整理进int32缓冲区。
        # KV逻辑block -> 物理block的映射表，由BlockManager.allocate/may_append填充，deallocate清空
        self.block_table = new_block_table()
        # Per-request state slots (filled by BlockManager.allocate()), indexing
        # the per-req cache tensors owned by ModelRunner (e.g. mamba_k_cache for
        # GDN). Empty = unallocated.
        #
        # `[0]` is the committed state, which every path reads and writes;
        # `[1:]` is one rollback slot per speculated token, which only the
        # spec-decode path touches. Held as a list rather than a base index
        # because the slots are allocated one at a time and need not be
        # adjacent — see `StateSlotPool`.
        # 【译】每请求的状态槽位（由 BlockManager.allocate() 填充），是 ModelRunner 持有的每请求缓存张量的下标
        #   （例如GDN的 mamba_k_cache）。为空表示尚未分配。
        #
        #   [0] 是已提交的状态，所有路径都读写它；[1:] 是每个投机token各一个的回滚槽位，只有投机解码路径会用到。
        #   用列表而不是"起始下标"保存，因为槽位是一个一个分配的，不一定相邻——见 StateSlotPool。
        self.state_slots: list[int] = []
        # Slot the NEXT forward reads its incoming state from, when that is not
        # the slot it writes (`state_slot`). Set by BlockManager on a state fork
        # — resuming from a checkpoint, or taking one — and cleared by the
        # scheduler once a batch has carried it, so it describes exactly one
        # forward. -1 = read and write the same slot, the case for every step in
        # between. Always a single slot: a checkpoint is one slot wide.
        # 【译】下一次forward读取输入状态的槽位，仅当它不同于写入的槽位（state_slot）时使用。由 BlockManager 在
        #   状态分叉时设置——从checkpoint恢复，或保存一个checkpoint——在某个batch携带它之后由调度器清除，
        #   所以它只描述一次forward。-1表示读写同一个槽位，中间的每一步都是这种情况。
        #   它总是单个槽位：一个checkpoint只占一个槽位。
        self.state_fork_src = -1
        # The KV-transfer offload/joint-load protocol state -- six values that
        # move together through admission and load. Grouped so a joint reset
        # cannot forget a field; see `OffloadJointRecord`.
        # 【译】KV传输offload/联合加载的协议状态——在准入和加载过程中一起变化的六个值。
        #   收拢在一起，联合重置时就不会漏掉字段；见 OffloadJointRecord。
        self.offload_joint = OffloadJointRecord()
        # ---------------- 主干字段：采样参数 ----------------
        # 把请求的采样参数拷贝到序列上，ModelRunner.prepare_sample按batch批量读取，拼成张量
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        # 最多生成多少个token，达到后结束
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略EOS继续生成
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_strings = sampling_params.stop_strings
        # Same type as `token_ids`, because the stop check compares a slice of
        # that against these and an `array("i")` never equals a list.
        # 【译】与 token_ids 类型相同，因为停止检查要拿 token_ids 的一段切片与它们比较，而 array("i") 永远不等于list。
        # 解释：这些是stop字符串分词后的token序列（由InputOutputProcessor.preprocess生成），
        #   调度器检查token_ids的末尾是否等于其中某一个，命中则以"stop_sequence"结束
        self.stop_token_sequences = [
            new_token_ids(s) for s in (stop_token_sequences or [])
        ]
        # 【分支】PD分离：decode端收到远端prefill的KV后，第一次decode时为True，见第9批
        self.is_first_decode = False
        # Set to True by Scheduler.postprocess after BlockManager.hash_blocks
        # has registered the prompt blocks for prefix caching. The trigger has
        # to be per-seq because in deferred-output mode the prefill step's
        # postprocess has no fwd_output entry for the seq (idx is None) — the
        # prefill output surfaces one step later, at which point seq.type has
        # already been flipped to DECODE. A seq.type / len(output_tokens) gate
        # would never fire for the prefill blocks; this flag does.
        # 【译】在 BlockManager.hash_blocks 把prompt的block登记到前缀缓存之后，由 Scheduler.postprocess 置为True。
        #   这个触发条件必须按序列记录，因为在延迟输出模式下，prefill那一步的postprocess里没有该序列的
        #   fwd_output项（idx为None）——prefill的输出要晚一步才出现，那时 seq.type 已经改成了DECODE。
        #   用 seq.type / len(output_tokens) 做判断永远不会为prefill的block触发；用这个标志就可以。
        self.prefix_hashes_published = False
        # 是否需要返回每个生成token的logprob
        self.return_logprobs = bool(getattr(sampling_params, "logprobs", False))
        # One entry per completion token, so the same reason `token_ids` is an
        # array applies: a list would box a PyFloat per token and hand the
        # collector a slot to walk for each one. `json.dumps` is the only
        # consumer that needs a list, and it converts at its own boundary.
        # 【译】每个生成token一项，所以 token_ids 用array的理由同样适用：list会为每个token装箱一个PyFloat，
        #   还给垃圾回收器每个token一个要遍历的槽位。json.dumps 是唯一需要list的使用方，它在自己的边界上转换。
        self.logprobs: array.array = array.array("d")
        # stream callback
        # 【译】流式回调
        # 每生成一批token调用一次，参数是RequestOutput（见request.py）；离线generate时为None
        self.stream_callback = stream_callback
        # The completion half of `token_ids`, kept in step with it by every
        # writer, so it is the same array type for the same reasons.
        # 【译】token_ids 中生成的那一半，每个写入方都让它与 token_ids 保持同步，所以出于同样的原因也是array。
        # 解释：等于token_ids[num_prompt_tokens:]，单独维护省得每次切片
        self.output_tokens = new_token_ids()
        # Placeholders from previous postprocess; overwritten in place.
        # 【译】上一次postprocess留下的占位token个数；会被原地覆盖。
        # 解释：延迟输出、投机解码时出现，见类开头的说明
        self.num_placeholder_tokens: int = 0

        # save speculative tokens if is_deferred_output = False or prefill is inter
        # 【译】保存投机token（当 is_deferred_output = False，或prefill处于中间chunk时）
        # 【分支】投机解码：上一步草稿模型提出的草稿token，下一步交给目标模型验证，见第7批
        self.spec_token_ids: np.ndarray = np.array([], dtype=np.int32)

        # DSpark Phase 2: scheduler-chosen verify length from the previous
        # decode step's propose(). None = no schedule yet -> verify mtp_k (full).
        # Next decode step sizes this seq's verification to dspark_next_ell+1.
        # 【译】DSpark第2阶段：调度器根据上一个decode step的 propose() 选定的验证长度。None表示还没有安排
        #   -> 验证全部 mtp_k 个。下一个decode step把这条序列的验证长度定为 dspark_next_ell+1。
        # 【分支】DSpark投机解码，见第7批
        self.dspark_next_ell: int | None = None

        # statistics fields
        # 【译】统计字段
        # InputOutputProcessor用来算延迟指标：
        #   TTFT（首token延迟）= first_token_time - arrive_time
        #   TPOT（每个输出token的平均耗时）= (leave_time - first_token_time) / (生成token数 - 1)
        self.arrive_time = 0.0
        self.first_token_time = 0.0
        self.leave_time = 0.0
        # 结束原因："eos"、"max_tokens"、"stop_sequence"、"stop_<token_id>"、"aborted"等
        self.leave_reason = ""

        # kv_transfer params
        # 【译】KV传输参数
        # 【分支】PD分离：输入是路由层传来的KV传输参数，输出是prefill端回给路由层的参数，见第9批
        self.kv_transfer_params = kv_transfer_params
        self.kv_transfer_params_output = None
        if kv_transfer_params:
            self.prefix_cache_hit_tokens = kv_transfer_params.get(
                "prefix_cache_hit_tokens", 0
            )

        # 注意：这里会再赋值一次，效果与上面的if分支相同（没有参数时为0），属于冗余代码
        self.prefix_cache_hit_tokens = (kv_transfer_params or {}).get(
            "prefix_cache_hit_tokens", 0
        )

        # accepted tokens for spec decode
        # 【译】投机解码中被接受的token
        # 【分支】投机解码：上一步额外得到的token数（被接受的草稿token），见第7批
        self.num_bonus_tokens = 0

        # Fan-out bookkeeping for SamplingParams.n > 1. When True, the sampler
        # must produce fresh, per-row random noise for this sequence instead
        # of reusing the cached shared exponential tensor, otherwise sibling
        # sequences with identical logits would emit identical tokens.
        # 【译】SamplingParams.n > 1 时的扇出记录。为True时，采样器必须为这条序列生成新的、逐行独立的随机噪声，
        #   而不是复用缓存的共享指数分布张量，否则logits相同的兄弟序列会输出相同的token。
        self.needs_independent_noise = needs_independent_noise
        # Parent request id (user-facing id from the API layer) and this
        # sequence's index within the fan-out group [0, n). Both default
        # to safe values for single-sample requests.
        # 【译】父请求id（来自API层、面向用户的id），以及这条序列在扇出组中的下标 [0, n)。
        #   对于单采样请求，两者都默认为安全值。
        self.parent_request_id = parent_request_id
        self.sibling_index = sibling_index
        # Explicitly requested DP rank, e.g. for cache aware DP routing.
        # Consumed by CoreManager._dispatch_to_dp_ranks as a routing hint.
        # 【译】显式指定的DP rank，例如用于感知缓存的DP路由。由 CoreManager._dispatch_to_dp_ranks 作为路由提示使用。
        # 解释：数据并行（DP）时有多个EngineCore，把同一会话路由到同一个rank能提高前缀缓存命中率，见第2批
        self.data_parallel_rank = data_parallel_rank
        # Optional client session lineage used by CoreManager's cache-aware,
        # token-load-aware DPA router.  These are routing metadata only; they
        # are deliberately kept out of the model-facing request payload.
        # 【译】可选的客户端会话血缘，供 CoreManager 中感知缓存、感知token负载的DPA路由器使用。
        #   它们只是路由元数据；有意不放进面向模型的请求负载中。
        self.dp_session_id = dp_session_id
        self.dp_parent_session_id = dp_parent_session_id

    # 以双下划线开头和结尾的方法叫"魔法方法"，会在特定语法下自动调用
    # len(seq) 等价于 seq.__len__()，返回token总数
    def __len__(self):
        return self._num_tokens

    # seq[3] -> self.token_ids[3]；seq[0:5] -> self.token_ids[0:5]
    def __getitem__(self, key):
        return self.token_ids[key]

    # 【分支】有界重放的起点，见Replay
    @property
    def replay_until(self) -> int:
        """Where the prefill forward scheduled last resumed, 0 when it did not
        back off below it."""
        # 【译】上一次调度的prefill forward在哪里恢复；如果它没有退到恢复点下方，返回0。
        return 0 if self.replay is None else self.replay.until

    # @property把方法变成"属性"：读seq.num_tokens时调用这个方法
    @property
    def num_tokens(self):
        """The total number of tokens in the sequence. i.e. prompt + completion"""
        # 【译】序列中的token总数，即 prompt + 生成部分
        return self._num_tokens

    # 写seq.num_tokens = x时调用：同时更新两个派生字段
    # 对比nano-vllm：nano-vllm的num_blocks、last_block_num_tokens是每次读取都重新计算的property；
    #   ATOM在写入时算好存起来（见文件末尾注释掉的旧property），调度器高频读取时省去重复计算
    @num_tokens.setter
    def num_tokens(self, value):
        self._num_tokens = value
        # 需要占用多少个block，向上取整；例如block_size=16，41个token -> 3个block
        self.num_blocks = (value + self.block_size - 1) // self.block_size
        # 最后一个block存了多少token；上例 41 - 2*16 = 9
        self.last_block_num_tokens = (
            self._num_tokens - (self.num_blocks - 1) * self.block_size
        )

    # 【分支】混合模型的状态槽位，见第4批：state_slots[0]是当前提交的状态，其余是投机解码的回滚槽位
    @property
    def state_slot(self) -> int:
        """The committed state slot, or -1 if this seq holds none.

        What every non-speculative path means by "the" slot: the one the
        forward reads and writes, the one a fork gives away, the one a
        checkpoint is. The rollback slots are `state_slots[1:]` and only the
        spec-decode path has any use for them.
        """
        # 【译】已提交的状态槽位；这条序列没有槽位时返回-1。
        #   这就是所有非投机路径所说的"那个"槽位：forward读写的那个、分叉时交出去的那个、checkpoint就是的那个。
        #   回滚槽位是 state_slots[1:]，只有投机解码路径会用到。
        return self.state_slots[0] if self.state_slots else -1

    @state_slot.setter
    def state_slot(self, slot: int) -> None:
        """Re-point the committed slot, keeping the rollback set.

        A fork moves where the request writes without disturbing its scratch:
        the speculation slots persist across forwards (step N's accepted slot
        is step N+1's initial state), so they belong to the request rather than
        to whichever slot it currently commits into.
        """
        # 【译】让已提交槽位指向新的位置，保留回滚槽位集合。
        #   分叉会改变请求写入的位置，但不影响它的临时槽位：投机槽位跨forward保留（第N步被接受的槽位
        #   就是第N+1步的初始状态），所以它们属于请求，而不属于请求当前提交写入的那个槽位。
        if self.state_slots:
            self.state_slots[0] = slot
        else:
            self.state_slots = [slot]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 已生成的token数，用来判断是否达到max_tokens
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    # 已生成的token；切片截止到num_tokens，因为停止条件可能把num_tokens回退到数组长度以内
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens : self.num_tokens]

    # （以下是被注释掉的旧代码：num_blocks、last_block_num_tokens原来是property，现在改为在num_tokens的setter里计算）
    # @property
    # def num_blocks(self):
    #     return (self.num_tokens + self.block_size - 1) // self.block_size

    # @property
    # def last_block_num_tokens(self):
    #     return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 取第i个逻辑block对应的token，用于计算前缀缓存的hash
    # 最后一个block可能不满，切片会自动截断
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # 追加一个生成的token：同时更新token_ids、output_tokens、last_token和num_tokens
    # num_tokens += 1会触发setter，顺带更新num_blocks、last_block_num_tokens
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.output_tokens.append(token_id)
        self.num_tokens += 1

    # 自定义pickle序列化：Sequence在LLMEngine -> CoreManager -> EngineCore之间通过ZMQ传递时会被pickle
    # 对比nano-vllm：nano-vllm只序列化前向需要的几个字段（发给其他TP rank）；
    #   ATOM的Sequence要交给另一个进程里的调度器完整管理，所以序列化全部字段（__dict__）
    #   唯一例外是prompt_hash_chain：它在ADD消息里和序列并排单独发送，这里置空避免重复传输
    def __getstate__(self):
        # The shipped prompt chain rides beside the seq (`prompt_hash_chain`).
        # 【译】要发送的prompt hash链与seq并排传输（prompt_hash_chain）。
        state = self.__dict__.copy()
        state["prompt_hash_chain"] = None
        return state

    # （以下是被注释掉的旧代码：与nano-vllm相同的"只序列化少数字段"实现，已被上面的__getstate__取代）
    # def __getstate__(self):
    #     return (
    #         self.num_tokens,
    #         self.num_prompt_tokens,
    #         self.num_cached_tokens,
    #         self.block_table,
    #         self.token_ids if self.num_completion_tokens == 0 else self.last_token,
    #     )

    # def __setstate__(self, state):
    #     (
    #         self.num_tokens,
    #         self.num_prompt_tokens,
    #         self.num_cached_tokens,
    #         self.block_table,
    #     ) = state[:-1]
    #     if self.num_completion_tokens == 0:
    #         self.token_ids = state[-1]
    #     else:
    #         self.last_token = state[-1]
