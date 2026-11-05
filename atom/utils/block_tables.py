# SPDX-License-Identifier: MIT
"""A forward slot's CPU page-table snapshot and its published GPU revision."""

# 【译】一个forward槽位的CPU页表快照，以及它已发布到GPU的版本。
#
# ======================================================================================================
# 【导读】worker侧：把一批序列的block table（每行长短不一的block id列表）写进固定形状的int32缓冲区，再传到GPU
#
# 在流程中的位置：
#   EngineCore ──forward(batch)──► 每个TP rank：block_table_codec.py 还原出 batch.block_tables = [BlockTable, ...]
#     ──► 注意力metadata builder（backends.py CommonAttentionBuilder.prepare_block_tables，各backend的prepare_decode等）
#           block_table_state(forward_vars["block_tables"]).prepare(rows, pad_to=running_bs)  ← 本文件：写锁页CPU缓冲区
#     ──► 用打包好的表算 slot_mapping（token_layout/slots.py）、kv_indices（utils/block_convert.py）
#     ──► block_table_state(...).publish(running_bs[, group])                                   ← 本文件：上传GPU（没变就不传）
#
# 缓冲区布局（forward_vars["block_tables"]，一个CpuGpuBuffer：锁页CPU内存 + 同形状的GPU显存，见 utils/__init__.py）：
#   形状 int32 [max_bs, block_table_cols]：max_bs = max_num_seqs（默认512），
#   block_table_cols ≈ ceil(max_model_len / kv_cache_block_size)（backends.py CommonAttentionBuilder.__init__）。
#   例：max_num_seqs=512、max_model_len=131072、block_size=16 → 512 × 8192 × 4字节 = 16 MiB（CPU、GPU各一份）。
#   形状固定是为了CUDA Graph：捕获时记住的是这块显存的地址和形状，之后每步只能原地改内容（见config.py CUDAGraphMode上方【科普】）
#
#   toy例子（block_size=16，cols=6；本步3个请求，CUDA Graph档位 running_bs=4）：
#              列0  列1  列2  列3  列4  列5      lengths  versions
#     row 0  [   5    9   12    0    0    0 ]      3        vA       A：40个token → ceil(40/16) = 3个block
#     row 1  [   3    0    0    0    0    0 ]      1        vB       B：10个token
#     row 2  [   4    8    0    0    0    0 ]      2        vC       C：20个token
#     row 3  [   0    0    0    0    0    0 ]      —        —        补齐行（scheduled_bs=3 < running_bs=4），整行为0
#   不变式：每行有效长度之后都是0，行 [len(lengths), zero_to) 全是0。
#     nano-vllm用-1补齐，这里用0：读这张表的代码按每行的context_lens / kv_indptr只取有效部分（例如block_convert.py的
#     kv_indices_generate_triton），补的值不会被当成KV位置；0可以一次清零写出，而且总是一个合法的block编号。
#   slot_mapping（本步每个token的KV写进KV cache的哪个槽位）由这张表算：slot = table[行, pos // block_size] × block_size + pos % block_size
#     例：A的第40个token（pos=39）→ table[0, 2] = 12 → slot = 12 × 16 + 7 = 199
#     （实现：model_ops/attentions/token_layout/slots.py 的 slot_mapping；DCP下另走 dcp_ops.dcp_prefill_slot_mapping，
#      只给本rank持有的token填槽位、其余填-1；DCP的前置知识见 prefix_hash.py 文件头④，本地下标的换算见 block_convert.py 文件头③）
#
# 增量写入：每步只写"变了的部分"
#   BlockTableState 不保存id的副本，只记每行的 (version, length)（version的定义见 sequence.py BlockTable类上方）。
#   prepare() 逐行比较新旧 (version, length)：
#     version、length 都相同          → 一个字节都不写
#     version相同、长度变长（只追加）  → 只写新增的那几个id（最常见：decode时某行多了1个block）
#     version不同（新请求 / 表被改写）  → 整行重写，比原来短出来的部分清0
#   全都没变时直接返回，revision不变，于是 publish() 也不上传。例：bs=1、block_size=16 的decode，16步里有15步
#     这一行都没变（每16个token才多1个block），这15步既不写CPU缓冲区也不做H2D拷贝。
#   接上面toy例子，下一步A追加了block 7（version仍是vA，长度4）：changes = [(行0, 长度4, 起点3, 旧长度3)]
#     → 只写 flat[0 × 6 + 3] = 7，revision+1；publish(4) 发现revision变了 → 把前4行拷到GPU。
#   注意：CPU侧只写新增的id，但上传拷的是前count行整行（count × cols × 4字节，异步DMA）。
#   这与 block_table_codec.py 是一条链：解码器让"只变长的行"保持原来的version，这里才能认出只需追加。
#
# 对比nano-vllm：nano-vllm每步 torch.tensor([每行补-1到本batch最长的行], pin_memory=True).cuda()——宽度随batch变、
#   每步新建张量、整表重拷；ATOM预分配固定形状的缓冲区（CUDA Graph需要），按版本号只写变化的部分，没变就不上传。
# ======================================================================================================

import numpy as np

from atom.model_engine.sequence import BlockTable


# 把一行（BlockTable / list / numpy数组）转成连续的一维int32 numpy数组；需要转换类型时顺带检查id非负且不超过int32
# 调用方：_normalize（没有version的行）、_validate_range（校验多于1个的新id）
def _int32_row(row):
    values = np.asarray(row)
    if values.ndim != 1:
        raise ValueError("Block-table rows must be one-dimensional")
    if values.dtype != np.int32:
        if values.size and (
            values.dtype.kind not in "iu"
            or values.min() < 0
            or values.max() > np.iinfo(np.int32).max
        ):
            raise ValueError("PAGE ids must be nonnegative int32 integers")
        values = values.astype(np.int32)
    return np.ascontiguousarray(values)


class BlockTableState:
    """Keep row versions and lengths beside the pinned snapshot, not its payload.

    All writers, including capture and padding, must use this object. Versions
    and lengths identify rows without payload snapshots or per-row key tuples.
    """

    # 【译】把每行的版本号和长度保存在锁页快照旁边，而不是保存载荷本身。
    #
    #   所有写者，包括（CUDA Graph）捕获和补齐，都必须经过这个对象。用版本号和长度来标识行，
    #   不需要载荷的快照，也不需要逐行的key元组。
    # 为什么"所有写者都必须经过它"：versions/lengths 是对缓冲区内容的承诺。有人绕过它直接改 buffer.np，
    #   下一步 prepare 会以为某行没变而跳过写入，GPU读到的就是被改坏的内容（不会报错）
    # 成员：
    #   buffer      CpuGpuBuffer（forward_vars的"block_tables"，或TBO微batch的"ub0_block_tables"等）
    #   cpu         buffer.np：锁页CPU内存的numpy视图，int32 [max_bs, cols]
    #   bytes       同一块内存按uint8看，[max_bs, cols × 4]：清零用（self.bytes[i, a*4 : b*4] = 0）
    #   flat        同一块内存的一维int32 memoryview：按"行号 × cols + 列号"直接写单个元素，比numpy的二维索引便宜
    #   versions / lengths   缓冲区里第i行当前对应的 (version, 长度)；version为object()的含义见_normalize
    #   zero_to     行 [len(lengths), zero_to) 已知全为0（补齐行）
    #   revision    每次真正写缓冲区就+1；publish用它判断GPU上的副本是否过期
    #   published   (上次上传到的gpu张量, (revision, count))
    #   page_limit / checked   只有DeepSeek-V4.1传page_limit；checked = {version: 已校验过的长度}，追加时只校验新尾巴

    def __init__(self, buffer):
        self.buffer = buffer
        self.cpu = buffer.np
        if (
            self.cpu.dtype != np.int32
            or self.cpu.ndim != 2
            or not self.cpu.flags.c_contiguous
        ):
            raise TypeError(
                "Block tables require a contiguous two-dimensional int32 buffer"
            )
        self.bytes = self.cpu.view(np.uint8)
        self.flat = memoryview(self.cpu).cast("B").cast("i") if self.cpu.size else ()
        self.versions = []
        self.lengths = []
        self.zero_to = 0
        self.revision = 0
        self.published = None
        self.page_limit = None
        self.checked = None

    # 缓冲区绑定了h2d发布者（utils/h2d.py，第5批）时，写之前先等"上一次从这块锁页内存出发的异步H2D拷贝"完成——
    #   否则改写的内容可能被还在进行的DMA读走
    def _acquire(self):
        binding = getattr(self.buffer, "_publication", None)
        if binding is not None:
            binding.acquire_write()

    # 每次真正写缓冲区之前调用一次：等异步拷贝、revision+1、作废"已发布"的记录
    def _begin_write(self):
        self._acquire()
        self.revision += 1
        self.published = None
        # A failed host write must not leave reusable CPU keys or padding.
        # 【译】一次失败的主机写入，不能留下可复用的CPU key或补齐信息。
        # 即：先清空versions/lengths/zero_to，写到一半抛异常时，下一步不会误以为缓冲区内容仍与旧key一致
        self.versions = []
        self.lengths = []
        self.zero_to = 0

    # 给没有version的行（普通list / numpy行：slice_to的源表、deepseek_v4_attn直接传的numpy表等）补一个version：
    #   内容与缓冲区当前第i行完全相同 → 沿用旧version（视为没变）；否则用一个新的 object()
    #   （object()只与自己相等，相当于"一个从没见过的版本号"，于是整行重写）
    # 输入：rows、versions（原地修改）、lengths；输出：rows的新列表（无version的行换成int32数组）
    def _normalize(self, rows, versions, lengths):
        rows = list(rows)
        for i, (row, version, length) in enumerate(zip(rows, versions, lengths)):
            if version is not None:
                continue
            rows[i] = values = _int32_row(row)
            unchanged = (
                i < len(self.lengths)
                and length == self.lengths[i]
                and np.array_equal(values, self.cpu[i, :length])
            )
            versions[i] = self.versions[i] if unchanged else object()
        return rows

    # 只有传了page_limit时调用（DeepSeek-V4.1：页数 num_pages，见 attentions/deepseek_v41/cache.py）：检查每个id在 [0, page_limit)
    # 输出：{version: 长度}，存成self.checked；同一version下次只校验 [已校验长度, 新长度) 这段新尾巴
    def _validate_range(self, rows, versions, lengths, page_limit):
        # Validation belongs to an append lineage, regardless of its row index.
        # Reordering requests must not re-read their already checked pages.
        # 【译】校验属于一条追加血缘，与行号无关。请求重新排序时，不能重新去读它们已经检查过的页。
        checked = self.checked if page_limit == self.page_limit else {}
        limit = min(page_limit, 1 << 31)
        for row, version, length in zip(rows, versions, lengths):
            start = checked.get(version, 0)
            if length <= start:
                continue
            if length - start == 1:
                invalid = not 0 <= row[start] < page_limit
            else:
                tail = _int32_row(row)[start:]
                # Signed negatives sort after every valid PAGE id as uint32,
                # so one reduction checks both bounds without a payload copy.
                # 【译】有符号的负数按uint32看会排在所有合法的PAGE id之后，所以一次归约就同时检查了上下界，也不用复制载荷。
                # 例：page_limit=1000：id 999合法；1000越界；-1按uint32是4294967295 ≥ 1000，同样越界
                invalid = tail.view(np.uint32).max() >= limit
            if invalid:
                raise ValueError("Request PAGE table is incomplete or out of range")
        return dict(zip(versions, lengths))

    # 把一批行写进CPU缓冲区（只写变了的部分），返回self（随后由调用方 publish 上传）
    # 输入：rows        list[BlockTable | list | np.ndarray]，长度 n = 本步的请求数（scheduled_bs）
    #       pad_to      CUDA Graph的档位running_bs（≥ n）：行 [n, pad_to) 清0；None表示不补齐
    #       page_limit  只有V4.1传：校验所有id < page_limit
    #       _keys       slice_to内部用：直接给出 (versions, lengths)，不从rows里读
    # 调用方：backends.py CommonAttentionBuilder.prepare_block_tables（各backend的decode、prefill的slot_mapping之前）、
    #   attentions/deepseek_v41/metadata.py、deepseek_v4_attn.py，以及本类的slice_to
    def prepare(self, rows, *, pad_to=None, page_limit=None, _keys=None):
        n = len(rows)
        capacity, columns = self.cpu.shape
        end = n if pad_to is None else pad_to
        if not 0 <= n <= end <= capacity:
            raise ValueError("Block-table rows exceed the declared capacity")
        if _keys is None:
            versions = [
                row.version if isinstance(row, BlockTable) else None for row in rows
            ]
            lengths = list(map(len, rows))
        else:
            versions, lengths = _keys
        # 快速路径：与上次写进去的完全相同（同样的行、同样的长度、补齐范围已经是0）→ 什么也不写，revision不变
        if (
            versions == self.versions
            and lengths == self.lengths
            and end <= self.zero_to
            and page_limit == self.page_limit
        ):
            return self
        if max(lengths, default=0) > columns:
            raise ValueError("Block-table row exceeds the destination columns")
        # 有行没有version（不是BlockTable）：按内容比较补上（_normalize）；所有校验都在第一次写之前做完
        if None in versions:
            rows = self._normalize(rows, versions, lengths)
        checked = (
            self._validate_range(rows, versions, lengths, page_limit)
            if page_limit is not None
            else None
        )

        # 比上次多出来的行：旧内容未知（可能是更早某步留下的），旧长度记作cols，于是新长度之后的整段都会被清0
        old_versions, old_lengths = self.versions, self.lengths
        old_n = len(old_lengths)
        if n > old_n:
            old_versions = old_versions + [None] * (n - old_n)
            old_lengths = old_lengths + [columns] * (n - old_n)
        # changes：(行号, 新长度, 从哪一列开始写, 旧长度)
        #   version相同且没变短 → 从旧长度开始，只写追加的部分；否则从第0列整行重写
        #   例：[(0, 4, 3, 3)] = 第0行从3变成4个，只写第3列
        changes = [
            (i, length, old if version == previous and length >= old else 0, old)
            for i, (version, previous, length, old) in enumerate(
                zip(versions, old_versions, lengths, old_lengths)
            )
            if version != previous or length != old
        ]
        # 要补齐（end > n），而且补齐行不一定全是0（行数变了，或补到了zero_to之外）→ 把 [n, end) 清0
        clear_tail = end > n and (n != old_n or end > self.zero_to)
        zero_to = max(end, self.zero_to) if n == old_n else end
        if changes or clear_tail:
            self._begin_write()  # all validation precedes the first write
            # A fully replaced batch can clear exposed tails in one pass.
            # Partial updates retain every unaffected row and copied prefix.
            # 【译】一个被整体替换的batch，可以一次清掉所有露出来的尾巴。部分更新则保留所有未受影响的行和已复制的前缀。
            # 条件：每一行都变了、且都是从第0列整行重写、长度和上次不同、且有行没占满 → 先把前n行整片清0，
            #   再逐行写，省掉逐行清尾巴
            bulk_clear = (
                len(changes) == n
                and lengths != old_lengths
                and min(lengths, default=columns) < columns
                and not any(start for _, _, start, _ in changes)
            )
            if bulk_clear:
                self.bytes[:n] = 0
            self._copy_changes(rows, changes, columns, bulk_clear)
            if clear_tail:
                self.bytes[n:end] = 0
        # 写完之后才记下新的key（_begin_write 已先清空旧key）
        self.versions, self.lengths = versions, lengths
        self.zero_to = zero_to
        self.page_limit = page_limit
        self.checked = checked
        return self

    # 按changes逐行写：先把比原来短出来的部分清0，再把 [start, length) 的id写进去；
    #   只多1个id时直接写单个元素（最常见的decode情况，省掉切片的开销）
    def _copy_changes(self, rows, changes, columns, bulk_clear):
        flat = self.flat
        for i, length, start, old in changes:
            if not bulk_clear and length < old:
                self.bytes[i, length * 4 : old * 4] = 0
            base = i * columns
            if length - start == 1:
                flat[base + start] = rows[i][start]
            elif length > start:
                flat[base + start : base + length] = (
                    rows[i][start:] if start else rows[i]
                )

    def pad(self, scheduled_bs, running_bs):
        """Finalize the padded range before publication, including draft rows."""
        # 【译】在发布之前定下补齐的范围，包括草稿行。
        # 输入：scheduled_bs（真实行数）、running_bs（补齐到的行数）；输出：self
        # 调用方：aiter_mla.py _prefill_block_table_rows（有drafter或有缓存前缀时）：prefill的表在第一次上传之前就补到running_bs，
        #   后面的消费者（MTP草稿等）直接复用这次上传，不必再改写一块可能正被DMA读着的锁页内存
        if not 0 <= scheduled_bs <= running_bs <= self.cpu.shape[0]:
            raise ValueError("Invalid block-table padding range")
        if scheduled_bs != len(self.lengths) or running_bs > self.zero_to:
            versions = self.versions[:scheduled_bs]
            lengths = self.lengths[:scheduled_bs]
            self._begin_write()
            self.bytes[scheduled_bs:running_bs] = 0
            self.versions, self.lengths = versions, lengths
            self.zero_to = running_bs
        return self

    def slice_to(self, buffer, start, count, *, pad_to):
        """Prepare a TBO destination using the source snapshot's row revisions."""
        # 【译】用源快照里各行的版本信息，准备一个TBO的目标缓冲区。
        # TBO（two-batch overlap）：把一个batch拆成两个微batch交替执行，让一个的通信与另一个的计算重叠（见llm_engine.py里的说明，
        #   第7、9批）。每个微batch有自己的 ub0_/ub1_block_tables 缓冲区；这里把整表的第 [start, start + count) 行连同
        #   (version, length) 交给目标缓冲区的prepare，于是目标也只写变化的行
        # 调用方：aiter_attention.py、aiter_mla.py、deepseek_v4_attn.py 的CUDA Graph TBO decode
        stop = start + count
        if stop > len(self.lengths):
            # Independent callers can supply an already packed CPU table.
            # 【译】独立的调用方可能直接提供一张已经打包好的CPU表。
            # 即源快照里没有这么多行的key（表不是经prepare写的）→ 退回按内容比较（_normalize）
            return block_table_state(buffer).prepare(
                self.cpu[start:stop], pad_to=pad_to
            )
        rows = [self.cpu[i, : self.lengths[i]] for i in range(start, stop)]
        keys = self.versions[start:stop], self.lengths[start:stop]
        return block_table_state(buffer).prepare(rows, pad_to=pad_to, _keys=keys)

    def publish(self, count, *, group=None):
        """Publish a changed table, optionally with the caller's metadata group.

        Omitting a table from a group does not omit its GPU view from metadata.
        Record the revision only after the entire publication succeeds.
        """
        # 【译】发布一张变化了的表，可以带上调用方的元数据组。
        #
        #   把一张表从组里省略，并不会把它的GPU视图从元数据里省略。只有在整个发布成功之后才记录revision。
        # 输入：count  要上传的行数（通常是running_bs）；None 表示这张表不传，只发布组里的其他缓冲区
        #       group  h2d发布组（utils/h2d.py，第5批）：几个元数据缓冲区一起上传
        # 输出：GPU视图 buffer.gpu[:count]，int32 [count, cols]
        # 调用方：aiter_attention.py、aiter_mla.py、deepseek_v4_attn.py、deepseek_v41/metadata.py、backends.py 等各backend的
        #   prepare_decode / prepare_prefill；同一张表在一步内可能被publish多次，没变化时只有第一次真正上传
        if count is None:
            if group is not None:
                group.publish(group.counts)
            return None
        # 需要上传（dirty）：从没上传过、GPU张量换了、或 (revision, count) 与上次不同；否则GPU上已经是这份内容
        key = (self.revision, count)
        previous = self.published
        dirty = (
            previous is None or previous[0] is not self.buffer.gpu or previous[1] != key
        )
        # 在组里：不脏时把这张表的count设为None（组发布时跳过它），但仍返回它的GPU视图
        if group is not None:
            group.set_count(self.buffer, count if dirty else None)
            group.publish(group.counts)
        elif dirty:
            self.buffer.copy_to_gpu(count)
        if dirty:
            self.published = (self.buffer.gpu, key)
        return self.buffer.gpu[:count]


def block_table_state(buffer):
    """Get the state belonging to this physical forward buffer/PP slot."""
    # 【译】取得属于这块物理forward缓冲区 / PP槽位的状态。
    # 状态挂在缓冲区对象自己身上（buffer._block_table），所以每块缓冲区各有一份账本：PP每个在途槽位clone出来的缓冲区
    #   （CpuGpuBuffer.clone）、TBO每个微batch的缓冲区互不干扰；state.cpu is not buffer.np 说明缓冲区的内存换过，重建状态
    # 调用方：backends.py、aiter_attention.py、aiter_mla.py、deepseek_v4_attn.py、qwen4_exp_attn.py、deepseek_v41/metadata.py
    state = getattr(buffer, "_block_table", None)
    if state is None or state.cpu is not buffer.np:
        state = buffer._block_table = BlockTableState(buffer)
    return state
