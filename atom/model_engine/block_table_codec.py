# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Ship a forward's block tables to the workers as appends alone.

Every forward RPC broadcasts one `ScheduledBatch` to every worker, and its
`block_tables` are the bulk of it: one row per running request, the whole row
every step, growing one block per decode. At 50 seqs x 100k context that is
~313k ids -- 1.2 MiB pickled and unpickled per rank per step -- to say
something whose new content is 50 ids.

The rows are `BlockTable`s, which carry a `version` that only a non-append
mutation redraws (see `atom/model_engine/sequence.py`). So the encoder does
not have to *infer* that a row merely grew, and never compares ids to find
out: it remembers `(version, length)` per request from the last step it sent,
and an unchanged version proves that everything up to that length is what the
workers already hold. Anything else -- a table cleared and refilled, a prefix
privatised in place by `BlockManager.disown_claimed_prefix`, a request seen
for the first time -- redraws the version and is sent in full.

The workers rebuild their rows from their own cache. Both sides are always on
and decoding is driven by the payload's type, so there is no setting a worker
could hold differently from the scheduler: whichever form a batch arrives in
is the form it is read in, and a step the encoder cannot account for arrives
as whole tables and resets both caches at once.
"""

# 【译】只用追加量把一次forward的block table发给各worker。
#
#   每次forward RPC都会把一个 ScheduledBatch 广播给每个worker，其中的 block_tables 占了大头：每个运行中的请求一行，
#   每一步都发整行，每次decode增长一个block。50条序列 x 10万上下文时约31.3万个id——每步每个rank要pickle、unpickle
#   约1.2 MiB——而它要表达的新内容只有50个id。
#
#   这些行是 BlockTable，它带一个只有"非追加修改"才会重新取的 version（见 atom/model_engine/sequence.py）。所以编码器
#   不必去*推断*某一行只是变长了，也从不比较id来确认：它记住上一步发出时每个请求的 (version, 长度)，版本号没变就证明
#   该长度以内的内容正是worker已经持有的。其他任何情况——表被清空再重填、前缀被 BlockManager.disown_claimed_prefix
#   原地私有化、第一次见到的请求——都会重新取版本号，于是整行发送。
#
#   worker用自己的缓存重建这些行。两端始终开启，解码由载荷的类型驱动，所以不存在worker与调度器设置不一致的可能：
#   batch以哪种形式到达就按哪种形式读；编码器无法说清的一步会以整张表到达，并同时重置两端的缓存。
#
# ======================================================================================================
# 【导读】block table的增量编码：EngineCore（调度器侧）发给各TP rank（worker）的forward指令里，只发"新追加的block id"
#
# 在架构中的位置（forward指令的完整时序图见 async_proc.py 文件开头）：
#
#   EngineCore进程（调度器）                        共享内存广播队列              每个TP rank进程（AsyncIOProc）
#   AsyncIOProcManager.call_func("forward", batch)  （aiter MessageQueue，
#     │ BlockTableDeltaEncoder.encode_rpc           写一次、每个rank都读）
#     │   block_tables: [BlockTable, ...]                                       AsyncIOProc.get_func
#     │   → BlockTableDelta              ──► ("forward", 编码后的batch) ──►      │ BlockTableDeltaDecoder.decode_rpc
#     ▼                                                                         │   → 还原成 [BlockTable, ...]
#   （调度器手里的batch不变，见encode里的copy.copy）                               ▼
#                                                                  ModelRunner.forward(batch)：注意力builder把这些行写进
#                                                                  [max_bs, 列数] 的int32缓冲区（utils/block_tables.py）
#
# 为什么要发增量（数字）：
#   block table每个请求一行，行长 = ceil(上下文token数 / block_size)。整表每步的大小 = Σ行长 × 4字节（int32）：
#     50个请求 × 10万token上下文，block_size=16 → 每行6250个id，共31.25万个id ≈ 1.19 MiB；
#     上界是 max_num_seqs × ceil(max_model_len / block_size) × 4字节：默认 max_num_seqs=512，若 max_model_len=131072、
#     block_size=16 且每个请求都跑满上下文 → 512 × 8192 × 4 = 16 MiB。
#   而decode一步里每行通常最多多出1个block（block_size=16时平均每16步才多1个），整批新增的id最多几十个。
#   这份数据在EngineCore里pickle一次、在每个rank里各unpickle一次，都在decode的关键路径上（一步只有~10ms）。
#   增量编码后每步只发：每行一个base长度（4字节）+ 一个偏移（4字节）+ 新id，50行约 50×4 + 51×4 + 新id×4 ≈ 0.4 KB。
#
# 发送方的版本账本（BlockTable.version 的定义与"version示例"见 sequence.py BlockTable类上方）：
#   编码器记 _published = {req_id: (上次发出时的version, 上次发出时的长度)}。下一步对每一行：
#     version 相同且长度没变短 → 前"上次长度"个id一定没变（version只有非追加修改才会换）→ base = 上次长度，只发后面的新id
#     其他情况（新请求、version变了）                         → base = 0，整行发送
#   worker侧的解码器记 _rows = {req_id: 上一步还原出的BlockTable}，按"保留前base个 + 追加新id"还原。
#   两边的账本都只保留"最近一次forward里出现的请求"，只要双方看到的forward序列相同（同一个FIFO广播队列保证），账本就一致；
#   编码器处理不了的batch（没有block table、行数不对、不是BlockTable）整张发出，并同时清空两边的账本。
#
# 两步的编码/解码示例（A、B、C是req_id，block_size=16）：
#   第1步：A.block_table=[5, 9]（version 7），B=[3]（version 8），编码器账本为空
#     encode：base_lengths=[0, 0]，tail_offsets=[0, 2, 3]，tail_values=[5, 9, 3]      （两行都整行发）
#             账本 = {A: (7, 2), B: (8, 1)}
#     decode：A = BlockTable([5, 9])，B = BlockTable([3])，缓存这两行
#   第2步：A追加了block 12（append不换version，仍是7，长度3）；B没变；新请求C=[4, 8]（version 11）
#     encode：A：账本(7, 2)，version相同、2 ≤ 3 → base=2，只发[12]
#             B：账本(8, 1)，没变       → base=1，什么也不发
#             C：账本里没有             → base=0，整行发[4, 8]
#             base_lengths=[2, 1, 0]，tail_offsets=[0, 1, 1, 3]，tail_values=[12, 4, 8]
#     decode：A：复制上一步的[5, 9]再追加12 → [5, 9, 12]（复制而不是原地追加，原因见decode里的注释）
#             B：直接共享上一步的[3]那个对象；C：新建[4, 8]
#   第3步：如果A被抢占（del bt[:]）或前缀被 disown_claimed_prefix 原地改写，version换了新的 → A整行重发
#
# 适用场景 / 代价：
#   好处在稳态decode：同一批请求连续多步出现，每步只发几个id。
#   边界：账本只记"上一次forward"里的请求。基础调度器一步只做prefill或只做decode（scheduler.py），所以一个prefill步
#     之后的decode步里，没出现在那个prefill步里的请求都要整行重发一次；DP空闲rank的dummy batch没有block table，
#     也会清空账本。这些情况只是退回"整表发送"，不影响正确性。
#
# 对比nano-vllm：nano-vllm的rank 0每步把 [方法名, seqs] 整个pickle进共享内存，Sequence.__getstate__ 里带着整张
#   block_table（list），每步全量重发；没有版本号，也没有增量编码。ATOM多出这个文件，只为省掉这部分CPU时间。
# ======================================================================================================

import copy
import logging
from dataclasses import dataclass

import numpy as np

from atom.model_engine.sequence import BlockTable

logger = logging.getLogger("atom")

# The one RPC whose payload is large enough and repetitive enough to be worth
# encoding. Named here rather than in the transport so that `async_proc` need
# not know anything about what a forward carries.
# 【译】唯一一个载荷又大、内容又足够重复、值得编码的RPC。名字定义在这里而不是在传输层里，
#   这样 async_proc 不需要知道forward里装了什么。
# encode_rpc / decode_rpc 只在方法名等于它时才改动参数，其余指令原样透传
FORWARD_RPC = "forward"

# block id在线上的类型：int32，与BlockTable（array("i")）的元素宽度相同，两者之间可以按字节零拷贝互转
_I32 = np.dtype(np.int32)


# 线上格式：替换 ScheduledBatch.block_tables 发给worker的对象
# @dataclass(frozen=True)：自动生成__init__等方法，且实例创建后字段不能再赋值（会抛FrozenInstanceError），表示一份只读消息
@dataclass(frozen=True)
class BlockTableDelta:
    """One step's block tables, as a shared prefix length plus new ids.

    Row `i` is ``workers_row_i[: base_lengths[i]]`` followed by
    ``tail_values[tail_offsets[i] : tail_offsets[i + 1]]``. A `base_lengths` of
    0 makes the row self-contained, which is how a request's first step, and
    every step after a table was disturbed, travel.
    """

    # 【译】一步的block table，表示为"共享前缀长度 + 新id"。
    #
    #   第 i 行 = workers_row_i[: base_lengths[i]] 后面接 tail_values[tail_offsets[i] : tail_offsets[i + 1]]。
    #   base_lengths 为0时这一行是自包含的——请求的第一步、以及表被扰动之后的每一步，都是这样传的。
    # 三个字段构成一个CSR（压缩行）结构：tail_offsets[i] ~ tail_offsets[i+1] 是第i行的新id在tail_values里的范围
    #   例（文件头第2步）：base_lengths=[2, 1, 0]，tail_offsets=[0, 1, 1, 3]，tail_values=[12, 4, 8]
    #     第0行 = 上次的前2个 + [12]；第1行 = 上次的前1个 + []；第2行 = [] + [4, 8]

    base_lengths: np.ndarray  # int32[num_rows]
    tail_offsets: np.ndarray  # int32[num_rows + 1]
    tail_values: np.ndarray  # int32[tail_offsets[-1]]


class BlockTableDeltaEncoder:
    """Scheduler side: replace a batch's block tables with a `BlockTableDelta`.

    One encoder per broadcast channel. It tracks what that channel's workers
    were last told, so it must see every forward on that channel and no other.
    """

    # 【译】调度器侧：把一个batch里的block table替换成 BlockTableDelta。
    #
    #   每个广播通道一个编码器。它记录的是该通道上的worker最近被告知了什么，所以它必须看到该通道上的每一条forward，
    #   而且只看这一个通道的。
    # 调用方：AsyncIOProcManager.__init__ 建一个（async_proc.py），call_func 对每条下发的指令调 encode_rpc
    # 为什么"必须看到每一条forward"：worker侧的解码器按同样的顺序消费同一串forward；若有forward绕过编码器发出，
    #   两边账本就会错开，解码器在 base 与缓存行的长度对不上时抛RuntimeError（见decode）

    def __init__(self):
        # req_id -> (version, length) as last published to the workers.
        # 【译】req_id -> 最近一次发布给worker时的 (version, 长度)。
        self._published: dict[int, tuple[int, int]] = {}
        # 最近一次"整表发送"的原因：同一原因只记一次debug日志（见_send_whole）；成功编码一次后清为None
        self._full_send_reason: str | None = None

    def encode_rpc(self, func_name: str, args: tuple) -> tuple:
        """Encode `args` if this is a forward, otherwise pass them through."""
        # 【译】如果是forward就编码 args，否则原样透传。
        # 输入：方法名、位置参数元组（forward时 args[0] 是 ScheduledBatch）；输出：新的参数元组（只替换args[0]）
        if func_name != FORWARD_RPC or not args:
            return args
        return (self.encode(args[0]), *args[1:])

    # 编码一个forward batch
    # 输入：ScheduledBatch，读其中的 block_tables: list[BlockTable] 与 req_ids: list[int]（两者按batch位置一一对应）
    # 输出：batch的浅拷贝 wire_batch，其 block_tables 换成 BlockTableDelta；处理不了时原样返回batch（整表发送）
    def encode(self, batch):
        rows = getattr(batch, "block_tables", None)
        req_ids = getattr(batch, "req_ids", None)
        if not rows or req_ids is None:
            # A warmup or dummy batch carries no tables at all.
            # 【译】warmup或dummy batch根本不带表。
            return self._send_whole(batch, "batch has no block tables")
        if len(rows) != len(req_ids):
            # `ScheduledBatch` drops rows for seqs with an empty table, which
            # would leave row i describing some other request. Downstream
            # consumers index the rows by batch position too, so this is not
            # expected to happen -- say so rather than silently sending full
            # tables forever.
            # 【译】ScheduledBatch 会丢掉空表序列的行，那样第i行描述的就是另一个请求了。下游消费者也按batch位置
            #   索引这些行，所以这种情况不应该发生——这里把它说出来，而不是悄悄地永远发整表。
            return self._send_whole(
                batch, f"{len(rows)} rows for {len(req_ids)} requests"
            )
        if not all(isinstance(row, BlockTable) for row in rows):
            # Rollout/disaggregation callers may build batches by hand.
            # 【译】rollout / PD分离的调用方可能手工构造batch。
            return self._send_whole(batch, "rows are not BlockTables")

        # base_lengths int32[num_rows]、tail_offsets int32[num_rows + 1]：见BlockTableDelta；
        #   tails收集每行新id的视图，最后一次concatenate成tail_values；published是本步结束后的新账本
        num_rows = len(rows)
        base_lengths = np.empty(num_rows, dtype=_I32)
        tail_offsets = np.empty(num_rows + 1, dtype=_I32)
        tail_offsets[0] = 0
        tails: list[np.ndarray] = []
        published: dict[int, tuple[int, int]] = {}

        for i, (req_id, row) in enumerate(zip(req_ids, rows, strict=True)):
            req_id = int(req_id)
            row_len = len(row)
            base = 0
            last = self._published.get(req_id)
            if last is not None:
                last_version, last_len = last
                # `last_len <= row_len` cannot fail while the version holds --
                # an unchanged version means only appends -- and is the one
                # cheap guard against a mutator that forgot to redraw it.
                # 【译】只要版本号不变，last_len <= row_len 就不可能不成立——版本号不变意味着只有追加——
                #   这一条是防止某个修改方忘了换版本号的、唯一一道廉价的保护。
                # 命中：前last_len个id就是worker手里那一行，只需发后面的部分；否则base保持0，整行发
                if last_version == row.version and last_len <= row_len:
                    base = last_len
            base_lengths[i] = base
            # Zero-copy view of the ids this step added; `np.concatenate`
            # below is the only pass over them.
            # 【译】本步新增id的零拷贝视图；下面的 np.concatenate 是对它们唯一的一次遍历。
            # BlockTable是array("i")，支持缓冲区协议：np.frombuffer直接看它的内存，offset按字节算（4 × base），
            #   count = 新id的个数（base == row_len 时是0，得到一个空视图）
            tails.append(
                np.frombuffer(
                    row, dtype=_I32, count=row_len - base, offset=_I32.itemsize * base
                )
            )
            tail_offsets[i + 1] = tail_offsets[i] + (row_len - base)
            # 新账本只记本batch里出现的请求：没出现的请求被忘掉，下次出现时整行发（worker侧的decode做同样的事）
            published[req_id] = (row.version, row_len)

        # 浅拷贝：只有发出去的副本换掉block_tables；调度器手里的batch不能改，forward返回后
        #   EngineCore还要拿它做postprocess（engine_core.py _process_engine_step_inner）
        wire_batch = copy.copy(batch)
        wire_batch.block_tables = BlockTableDelta(
            base_lengths=base_lengths,
            tail_offsets=tail_offsets,
            tail_values=(
                np.concatenate(tails, dtype=_I32)
                if tail_offsets[-1]
                else np.empty(0, dtype=_I32)
            ),
        )
        # Published only once the encoded batch exists: if building it raises,
        # the workers were told nothing and the cache must not claim otherwise.
        # 【译】只有在编码后的batch构造成功之后才更新账本：如果构造过程抛了异常，worker什么也没收到，
        #   缓存就不能声称它们收到了。
        self._published = published
        self._full_send_reason = None
        return wire_batch

    def _send_whole(self, batch, reason: str):
        """Fall back to the plain batch, and drop what the workers were told.

        The workers key off the payload's type, so an un-encoded batch also
        resets their side; the two caches cannot drift apart.
        """
        # 【译】退回到原样的batch，并丢弃"worker被告知了什么"的记录。
        #
        #   worker按载荷的类型判断，所以一个未编码的batch也会重置它们那一侧；两端的缓存不会错开。
        # 输入：batch、reason（只用于日志）；输出：原样的batch（block_tables仍是list，worker收到后清空自己的缓存）
        self._published.clear()
        if reason != self._full_send_reason:
            # Once per reason, not once per step: this is the difference
            # between "the optimization is off" and a log flood.
            # 【译】每种原因只记一次，而不是每步一次：这就是"优化被关掉了"和"日志刷屏"之间的区别。
            logger.debug("block-table delta off, sending whole tables: %s", reason)
            self._full_send_reason = reason
        return batch


class BlockTableDeltaDecoder:
    """Worker side: rebuild versioned rows from a `BlockTableDelta`."""

    # 【译】worker侧：从 BlockTableDelta 重建带版本号的行。
    # 调用方：每个TP rank的 AsyncIOProc.__init__ 建一个（async_proc.py），get_func 对每条从广播队列取出的指令调 decode_rpc

    def __init__(self):
        # req_id -> 上一步还原出的那一行，与编码器的 _published 一一对应
        self._rows: dict[int, BlockTable] = {}

    def decode_rpc(self, func_name: str, args: list) -> list:
        """Decode `args[0]` in place if this is an encoded forward."""
        # 【译】如果是编码过的forward，就原地解码 args[0]。
        # args是dequeue出来的list，所以可以原地替换args[0]；返回同一个list
        if func_name == FORWARD_RPC and args:
            args[0] = self.decode(args[0])
        return args

    # 输入：dequeue出来的batch，block_tables 是 BlockTableDelta（编码过）或普通list（整表发送）
    # 输出：同一个batch对象，block_tables 换回 list[BlockTable]，与 req_ids 按位置对应
    def decode(self, batch):
        delta = getattr(batch, "block_tables", None)
        if not isinstance(delta, BlockTableDelta):
            # Whole tables arrived, so whatever was cached is superseded --
            # the encoder cleared its side for the same step.
            # 【译】整张表到了，所以缓存里的任何东西都作废了——编码器在同一步也清掉了它那一侧。
            self._rows.clear()
            return batch

        req_ids = batch.req_ids
        if len(req_ids) != len(delta.base_lengths):
            raise RuntimeError(
                f"block-table delta has {len(delta.base_lengths)} rows for "
                f"{len(req_ids)} requests"
            )

        rows: list[BlockTable] = []
        cached: dict[int, BlockTable] = {}
        for i, req_id in enumerate(req_ids):
            req_id = int(req_id)
            base = int(delta.base_lengths[i])
            start = int(delta.tail_offsets[i])
            end = int(delta.tail_offsets[i + 1])
            # base > 0：前base个id取自上一步缓存的同一请求那一行。缓存里没有或长度对不上，说明两边账本错开了，
            #   直接报错，而不是拼出一张错误的表（错误的block id会让注意力读写别的请求的KV，且不会报错）
            if base:
                previous = self._rows.get(req_id)
                if previous is None or len(previous) != base:
                    raise RuntimeError(
                        f"missing block-table prefix for request {req_id}: "
                        f"need {base}, have "
                        f"{None if previous is None else len(previous)}"
                    )
                # Appending into the row the previous batch handed out would
                # rewrite history the token processor may still be reading, so
                # a row that grows is copied first. A row that did not grow is
                # shared, which is the decode-step-with-no-new-block case.
                # 【译】往上一个batch交出去的那一行里追加，会改写token processor可能还在读的历史，所以会增长的行
                #   要先复制。没有增长的行就共享，这对应"decode一步没有新block"的情况。
                # token processor：model_runner.py 的 tokenIDProcessor，它用 prev_batch 保存着上一步的batch（延迟输出）
                row = previous
                if end != start:
                    row = BlockTable(previous)
                    # This is the next immutable snapshot in the same append
                    # lineage. Equal version + length still means equal ids.
                    # 【译】这是同一条追加血缘里的下一个不可变快照。版本号相同 + 长度相同，仍然意味着id相同。
                    # BlockTable(previous) 会领一个新版本号（sequence.py），这里改回上一行的：worker侧的 BlockTableState
                    #   （utils/block_tables.py）据此认出"同一行只是变长了"，只把新id写进锁页缓冲区
                    row.version = previous.version
            else:
                row = BlockTable()
            # 追加本行的新id：frombytes 是追加操作，不换版本号；memoryview(...).cast("B") 把int32数组当成字节串交给它
            if end > start:
                row.frombytes(memoryview(delta.tail_values[start:end]).cast("B"))
            rows.append(row)
            cached[req_id] = row

        # 缓存换成本步的行（与编码器的 self._published = published 对称：没出现在本步的请求被忘掉）
        batch.block_tables = rows
        self._rows = cached
        return batch
