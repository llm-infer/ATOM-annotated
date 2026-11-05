# SPDX-License-Identifier: MIT
"""Owner-only PAGE fields and complete request STATE geometry for CSA2.

A PAGE holds the main latent of every owner; its index rows live in a region
of their own, one per owner, addressed by the same block id. The two planes
are bought together (`paged_bytes`) and read by the same block table, which is
the arrangement DeepSeek-V4's indexer already has -- see `sub_pool_spec`'s
"rides the same block table as its main compressed KV".

Keeping the index rows inside the PAGE instead would tie the finest
addressable group of them to the page: a paged scorer takes one block stride
and computes `id * stride`, so an index plane strided by the PAGE can only be
addressed a whole page's rows at a time. Region-major makes that stride the
rows' own, so a block id can name a group of rows finer than a page.

Everything here scales with cached history except the STATE entry, so the
regions are laid out main-pages, index planes, STATE -- one boundary between
the two currencies rather than one per plane.

What sets `block_size` is neither of the paddings this file rounds up. Both
are negligible: a field's own alignment comes out at zero for every shape CSA2
runs (a plane is `rows * row_bytes` and both factors are already coarse), and
an index plane is aligned once over the whole pool rather than per page, so
four owners cost at most four alignments. The two terms that do move are the
internal fragmentation of each request's last page -- `(block_size - 1)` tokens
at 3200 B (BF16 KV and index), so 406 KB per in-flight request at 128, ~0.05%
of a pool -- and the block table, `[max_num_seqs, max_model_len / block_size]`
int32 held twice and copied to the device every forward: 134 MB at block 16
against 16.8 MB at 128, for a million-token context. The block table is the
larger of the two and it is per-forward traffic, which is why CSA2 takes a
coarse page. What a coarse page costs is prefix reuse below its own size,
which no longer matches at all, and the rows encoder bounded replay recomputes
below a hit, a whole page; 128 keeps both at the window's size.
"""

# 【译】CSA2（DeepSeek-V4.1）的 PAGE 字段（只给 owner 层）与完整的请求 STATE 几何。
#
#   一个 PAGE 装着每个 owner 的主潜在向量；它的索引行住在自己的区域里，每个 owner 一个，用同一个 block id
#   寻址。两个平面一起购买（paged_bytes），由同一张 block table 读取——DeepSeek-V4 的索引器已经是这种安排，
#   见 sub_pool_spec 里的"和它的主压缩 KV 共用同一张 block table"。
#
#   若把索引行放在 PAGE 内部，它们能寻址的最小组就会被绑死在页上：分页打分器只接收一个 block 步长并计算
#   id * stride，所以按 PAGE 跨步的索引平面一次只能按整页的行来寻址。按区域排（region-major）让这个步长变成
#   这些行自己的，于是一个 block id 能指代比一页更细的一组行。
#
#   这里除了 STATE 条目，其余都随缓存的历史增长，所以区域按"主 KV 页、索引平面、STATE"的顺序排——两种货币
#   之间只有一条分界，而不是每个平面一条。
#
#   决定 block_size 的不是本文件向上取整的那两种填充。两者都可以忽略：对 CSA2 跑的每种形状，字段自身的对齐
#   填充都是零（一个平面是 rows * row_bytes，两个因子本来就很粗），而索引平面是在整个池上对齐一次、而不是
#   每页对齐，所以四个 owner 最多花四次对齐。真正会变的两项，一是每个请求最后一页的内部碎片——
#   (block_size - 1) 个 token、每个 3200 B（BF16 的 KV 和索引），所以 block 128 时每个在途请求 406 KB，约占池
#   的 0.05%——二是 block table：[max_num_seqs, max_model_len / block_size] 的 int32，保存两份，每次 forward
#   都拷到 device：百万 token 上下文时，block 16 要 134 MB，block 128 只要 16.8 MB。block table 是两者中较大
#   的，而且是每次 forward 的流量，所以 CSA2 用粗粒度的页。粗页的代价是：小于一页的前缀复用完全匹配不上，
#   命中点以下 encoder 有界重放要重算的行也是整整一页；128 让两者都等于窗口大小。
# 解释：134 MB / 16.8 MB 按 max_num_seqs = 256 算（256 × 2^20 / block × 4 B × 2 份）；406 KB = 127 × 3200 B。
#
# =====================================================================================
# 【导读】v41_pool_geometry.py —— DeepSeek-V4.1（CSA2）的 PAGE 与 STATE 字节布局（第5批 5b，B级；模型本身见第8批）
#   前置：V4 的 CSA / 索引器见 config.py Config.__post_init__ 第12步 V4 部分的【科普】；V4.1 的 CSA2 是同一思路的
#     变体——每层压缩率只取 0 / 1 / 2，只有 Full 模式的"owner"层（docs/deepseek_v41_runtime.md：层 2、8、14、20）
#     真正保存全局主 KV 和索引行，其余层复用 owner 的（models/deepseek_v41/config.py build_attention_topology）；
#     block_size 被 Config.__post_init__ 第13步改成 128。总览见 pool_layout/__init__.py【科普】④。
#   一次分配（deepseek_v41/cache.py 的 backing = paged_extents(pages)[1] + slots × state_bytes 字节）：
#     │ 主 KV：PAGE0[owner_a 行 │ owner_b 行 │ …] PAGE1[…] … │ owner_a 的索引平面 │ owner_b 的索引平面 │ … │ STATE：槽位0 │ 槽位1 │ …
#      ↑ 0（EntryMajorArena pages）                           ↑ paged_extents 给出的偏移                   ↑ paged_extents 的总数
#     同一个 block id 在主 KV 区和每个索引平面里各占一份（索引平面内按页紧密排列，步长是"这些行自己的"）
#   带数字走一遍（示意参数：4 个 owner = 1 个 ratio 1 + 3 个 ratio 2，head_dim 512，index_dim 128，主 KV bf16，
#     FP8 索引、index_block_rows 16，40 层，窗口 128，不开投机；这组取值若把索引也按 bf16 计，正好给出 docstring
#     的 3200 B / token。数字均按本文件的公式算出）：
#     page_fields：ratio 1 的 owner 128 行 × 1024 B = 128 KiB，ratio 2 的各 64 行 = 64 KiB → page_bytes = 327,680 B
#     index_planes：FP8 一个 16 行的 block = 16 × 128 + 16 × 4 = 2112 B → 每行 132 B
#     paged_bytes = 327,680 + (128 + 3 × 64) × 132 = 369,920 B / PAGE（128 个 token，≈ 2.8 KiB / token）
#     state_fields：window 40 层 × 128 × 1024 B = 5 MiB（进 checkpoint）；compress_kv / compress_score 各
#       4 个 owner × 2 × 512 × 4 B = 16 KiB（不进）；cursor 4 × 8 B（进）→ state_bytes = 5,276,672 B ≈ 5.03 MiB
#     checkpoint_image_ranges = [(0, 5,242,880), (5,275,648, 32)] → 镜像 5,242,912 B
#   谁用：deepseek_v41/backend.py（构造本类；sub_pool_specs 声明 page_pool(paged_bytes) 与 state_pool(state_bytes)）、
#     deepseek_v41/cache.py（按它切 backing）、deepseek_v41/checkpoints.py（状态快照）、plugin/vllm/deepseek_v41_bridge.py；
#     tests/test_v41_geometry_parity.py 核对原生与插件两处构造参数一致
# =====================================================================================

from dataclasses import dataclass

import torch

from .entry_arena import (
    EntryField,
    checkpoint_ranges_for,
    entry_bytes_for,
    field_extents,
    plan_regions,
)
from .v4_pool_fields import (
    MQA_LOGITS_PRESHUFFLE_ROWS,
    fp8_indexer_block_fields,
    indexer_block_regions,
)
from .v4_pool_geometry import WindowParams

# The packed main pool's FP4 grid. Finer than the index and window planes,
# which take `quantize_fp4`'s group-32 E8M0 default; a scatter that disagrees
# with `main_row_bytes` writes rows the readers cannot decode.
# 【译】打包主池的 FP4 网格。比索引平面和窗口平面更细——后两者用 quantize_fp4 默认的每 32 个一组的 E8M0；
#   一个与 main_row_bytes 不一致的 scatter 会写出读者解不了码的行。
# （每 16 个值一个 fp8 e4m3 scale：head_dim 512 时一行 = 256 B 数值 + 32 B scale = 288 B，见 main_row_bytes）
MAIN_FP4 = {"group_size": 16, "scale_dtype": torch.float8_e4m3fn}

# The FP8 index plane's scale format, in `indexer_k_quant_and_cache`'s spelling:
# a power-of-two scale, which is what V4's compressor writes for the same plane
# and what the published indexer quantizes with. The readers take the scale as
# fp32 either way, so this is free to change -- but a writer and an oracle that
# disagree about it differ by up to 2x per row and nothing about the bytes says
# which one meant what.
# 【译】FP8 索引平面的 scale 格式，用 indexer_k_quant_and_cache 的写法：2 的幂的 scale，也就是 V4 的压缩器给
#   同一个平面写的、发布的索引器量化时用的格式。读者无论如何都把 scale 当 fp32 读，所以它可以随意改——但写者
#   和参考实现（oracle）若对它看法不一，每行会差到 2 倍，而字节本身看不出谁是什么意思。
INDEX_FP8_SCALE_FMT = "ue8m0"

# The rows one page of the FP4 index plane holds: the row-group scorer's page-8
# layout, a page being a candidate block so a kept list is its block table.
# 【译】FP4 索引平面一页装的行数：行组打分器的 page-8 布局；一页就是一个候选块，所以保留下来的候选列表本身
#   就是它的 block table。（这就是 v4_pool_fields.MQA_LOGITS_PRESHUFFLE_ROWS 注释里说的"恰好 8 行"的短页）
FP4_INDEX_PAGE_ROWS = 8


# V41PoolGeometry：V4.1 一次配置的全部池几何（不可变；所有字节数都是由字段算出的 property）
#   构造方：deepseek_v41/backend.py 的 builder __init__、plugin/vllm/deepseek_v41_bridge.py
#   字段：layers = 层数（含草稿层）；owners = Full 模式层的 (全局层号, 压缩率 1 或 2)；block_size = 每 PAGE 的 token 数（128）；
#     window_size = 滑动窗口（hf_config.sliding_window）；head_dim / index_dim = 主 KV 与索引 key 的维度；
#     history_size = engram_max_ngram_size − 1（cursor 里保留的历史 ID 个数）；packed = 主 KV 是否 fp4 打包
#     （--kv-cache-dtype fp4）；speculative_tokens = 草稿 token 数
@dataclass(frozen=True)
class V41PoolGeometry:
    layers: int
    owners: tuple[tuple[int, int], ...]
    block_size: int
    window_size: int
    head_dim: int
    index_dim: int
    history_size: int = 3
    packed: bool = False
    speculative_tokens: int = 0
    # The distinct compression ratios this configuration's layers run, read
    # off the topology rather than enumerated, so a ratio no layer wants gets
    # no buffer. Every layer at a ratio reserves its rows the same way, so the
    # indptr is the ratio's and not the layer's -- the same thing V4's three
    # `kv_indptr_{swa,csa,hca}` are, at one fixed address each.
    # 【译】这个配置的各层实际用到的不同压缩率，从拓扑读出而不是枚举，所以没有层要的 ratio 不分配缓冲。同一
    #   ratio 的每层以同样方式预留行，所以 indptr 是"这个 ratio 的"而不是"这一层的"——和 V4 的三个
    #   kv_indptr_{swa,csa,hca} 是同一回事，各在一个固定地址。
    layer_ratios: tuple[int, ...] = ()
    # The width the scorer emits, whatever a row can see: a row short on
    # history is `-1` padded, not narrowed.
    # 【译】打分器输出的宽度，不管一行能看到多少：历史不够的行用 -1 填充，而不是变窄。
    index_topk: int = 0
    # Rows one index block id names: the `KVBlockSize` every reader of this
    # plane passes. A layout choice and not the kernel's tile -- 8 is what lets
    # a candidate list be a block table, the length candidates are picked in.
    # 【译】一个索引 block id 指代的行数：这个平面的每个读者传入的 KVBlockSize。这是布局选择而不是 kernel 的
    #   tile——8 让候选列表可以就是 block table，也就是候选被挑选的长度。
    # （backend 传入 hf_config.candidate_block_size；必须是 16 的倍数或恰好 8，见 __post_init__）
    index_block_rows: int = MQA_LOGITS_PRESHUFFLE_ROWS
    # The index plane's format: FP8 rows with their scales inside each block,
    # or the FP4 page-8 layout the row-group scorer reads, values and E8M0
    # scales in two planes (`index_planes`).
    # 【译】索引平面的格式：FP8 行、scale 在每个 block 内部；或者行组打分器读的 FP4 page-8 布局，数值和 E8M0 scale
    #   分在两个平面里（index_planes）。（backend 按 --index_cache_dtype == "fp4" 设置）
    index_fp4: bool = False

    # 构造时校验：维度为正且能被 32 整除；bf16 行要 256 B 对齐；owner 不重复；FP4 索引必须是 8 行一页、
    #   index_dim 是 128 的倍数；index_block_rows 是 16 的倍数或恰好 8；每个 owner 的压缩率是 1 或 2、能整除 block_size，
    #   且一个 PAGE 的行数能整除 index_block_rows
    def __post_init__(self):
        if self.speculative_tokens < 0:
            raise ValueError("Speculative window slack cannot be negative")
        if (
            min(
                self.layers,
                self.block_size,
                self.window_size,
                self.head_dim,
                self.index_dim,
                self.history_size,
            )
            <= 0
        ):
            raise ValueError("Cache dimensions must be positive")
        if self.head_dim % 32 or self.index_dim % 32:
            raise ValueError("Cache dimensions must be divisible by 32")
        if not self.packed and self.row_bytes % 256:
            raise ValueError("BF16 attention rows must align to 256 bytes")
        if len({owner for owner, _ in self.owners}) != len(self.owners):
            raise ValueError("Each global owner must be declared once")
        rows = self.index_block_rows
        if self.index_fp4 and (rows != FP4_INDEX_PAGE_ROWS or self.index_dim % 128):
            raise ValueError(
                f"The FP4 index plane is paged at {FP4_INDEX_PAGE_ROWS} rows of "
                f"whole 128-dim K tiles; got {rows} rows of {self.index_dim}"
            )
        if rows % MQA_LOGITS_PRESHUFFLE_ROWS and rows != 8:
            raise ValueError(
                f"An index block holds whole {MQA_LOGITS_PRESHUFFLE_ROWS}-row MFMA "
                f"tiles, or exactly 8 rows shuffled in groups of 8; got {rows}"
            )
        for owner, ratio in self.owners:
            if not 0 <= owner < self.layers or ratio not in (1, 2):
                raise ValueError("Invalid global owner or compression ratio")
            if self.block_size % ratio:
                raise ValueError("PAGE token count must divide every compression group")
            # A block id names `index_block_rows` rows, so a PAGE's rows have to
            # be a whole number of them -- the ratio-2 owners are what makes
            # that a statement about twice the PAGE.
            # 【译】一个 block id 指代 index_block_rows 行，所以一个 PAGE 的行数必须是它的整数倍——ratio 为 2 的 owner
            #   让这成了对"两倍 PAGE"的要求。
            # 例：index_block_rows 16、有 ratio 2 的 owner → block_size 至少 32（docs/deepseek_v41_runtime.md 说的"PAGE 的下限 32"）；
            #   实际用 128：ratio 2 时一页 64 行 = 4 个索引 block
            if self.rows_per_page(ratio) % rows:
                raise ValueError(
                    f"An index plane needs whole {rows}-row blocks: "
                    f"ratio {ratio} gives a PAGE {self.rows_per_page(ratio)} rows, "
                    f"so raise the PAGE token count to a multiple of "
                    f"{rows * max(r for _, r in self.owners)}"
                )

    # row_bytes = 一行 bf16 主 KV 的字节（head_dim × 2）；alignment = 字段与区域的对齐（打包时 256，否则一整行）；
    # main_row_bytes = 主 KV 一行在池里的实际字节：打包时 head_dim / 2（E2M1）+ head_dim / 16（每 16 个一个 fp8 scale）
    @property
    def row_bytes(self):
        return self.head_dim * 2

    @property
    def alignment(self):
        return 256 if self.packed else self.row_bytes

    @property
    def main_row_bytes(self):
        if not self.packed:
            return self.row_bytes
        return self.head_dim // 2 + self.head_dim // MAIN_FP4["group_size"]

    @property
    def index_planes(self):
        """`(name, bytes one row costs in it)` of each plane an owner's index
        rows take, in layout order.

        FP8 is one plane: under preshuffle a row's bytes are interleaved across
        its tile, so a row's cost is the block over its rows -- the convention
        `fp8_indexer_block_fields` states, and the reason a scale sits past the
        block's data rather than beside its own row. FP4 is two, the packed
        E2M1 values and their E8M0 scales, each dense in pages: the row-group
        scorer addresses page `i` of either at `i *` its page bytes.
        """
        # 【译】一个 owner 的索引行占用的每个平面的 (名字, 每行在其中的字节)，按布局顺序。
        #   FP8 是一个平面：在预重排下一行的字节交错分布在它的 tile 里，所以一行的成本是整个 block 除以行数——这是
        #   fp8_indexer_block_fields 声明的约定，也是 scale 放在 block 数据之后、而不是挨着各自那一行的原因。FP4 是
        #   两个平面：打包的 E2M1 数值和它们的 E8M0 scale，各自按页紧密排列：行组打分器在两者中寻址第 i 页都是
        #   i * 该平面的页字节数。
        # 例：FP8、16 行、index_dim 128 → (("index", 132),)；FP4、index_dim 128 → (("index", 64), ("index_scale", 4))
        if self.index_fp4:
            return (
                ("index", self.index_dim // 2),
                ("index_scale", self.index_dim // 32),
            )
        block = indexer_block_regions(
            fp8_indexer_block_fields(
                self.index_block_rows, self.index_dim, torch.float8_e4m3fn
            )
        )[1]
        return (("index", block // self.index_block_rows),)

    @property
    def index_row_bytes(self):
        """What one index row costs over all of its owner's index planes."""
        # 【译】一个索引行在它的 owner 的全部索引平面上的总成本（FP8 为 132 B，FP4 为 68 B）。
        return sum(width for _, width in self.index_planes)

    # 窗口一行：打包时 fp8 值 head_dim B + 每 32 个一个 e8m0 scale；否则 bf16 一行
    @property
    def window_row_bytes(self):
        return self.head_dim + self.head_dim // 32 if self.packed else self.row_bytes

    @property
    def ring_slots(self):
        # Verification includes one guaranteed input plus speculative tokens.
        # Slack retains the window behind every possible accepted prefix.
        # 【译】验证（verify）包括一个必定被接受的输入加上若干投机 token。多出的余量保证每一种可能被接受的前缀后面
        #   的窗口都还在。例：窗口 128、5 个草稿 → 环长 133
        return self.window_size + self.speculative_tokens

    @property
    def compress_owners(self):
        return tuple(owner for owner, _ in self.owners)

    @property
    def compress_ratios(self):
        """`(ratio, overlap)` per distinct ratio: CSA2 never overlaps."""
        # 【译】每个不同 ratio 一个 (ratio, overlap)：CSA2 从不重叠（V4 的 CSA 相邻组有重叠，这里没有）。
        return tuple(sorted({(ratio, False) for _, ratio in self.owners}))

    def batch_topk(self, ratio):
        """The selection width a batch gets at `ratio`, before a scorer runs.

        The scorer emits `index_topk` and pads what a row cannot see, so this
        is a constant rather than a batch's shape -- which is what lets the
        indptr reserve its rows up front, and keeps the scan it feeds to one
        compiled variant.
        """
        # 【译】在打分器运行之前，一个 batch 在 ratio 下得到的选择宽度。
        #   打分器输出 index_topk 个、把一行看不到的部分填充，所以这是一个常数而不是 batch 的形状——这让 indptr 可以
        #   预先预留行，也让它喂给的那个扫描只需一个编译变体。ratio 0（纯窗口层）不做选择，为 0。
        return self.index_topk if ratio else 0

    @property
    def compress_ring_slots(self):
        """V4's `STATE_SIZE`: the pool window plus rejected-draft slack.

        One width for every owner rather than one per ratio. The ring only has
        to be at least `K_pool = ratio` (no overlap in CSA2) and the widest
        owner sets that; a ratio-1 owner spending one extra row is cheaper than
        a second field and a second modulus to keep in step with the kernels.

        The slack is why a rejected draft cannot corrupt the next round: round
        R's discarded writes sit at most `speculative_tokens` ids past R+1's
        commit head, so they fall outside the `K_pool`-wide window R+1 reads.
        """
        # 【译】相当于 V4 的 STATE_SIZE：池化窗口加上被拒草稿的余量。
        #   所有 owner 共用一个宽度，而不是每个 ratio 一个。环只需至少有 K_pool = ratio 那么宽（CSA2 没有重叠），由最宽
        #   的 owner 决定；ratio 为 1 的 owner 多花一行，比多一个字段、多一个要和 kernel 保持同步的模数便宜。
        #   这个余量就是被拒草稿不会污染下一轮的原因：第 R 轮被丢弃的写入最多比第 R+1 轮的提交头超前
        #   speculative_tokens 个 id，所以它们落在第 R+1 轮读取的 K_pool 宽窗口之外。
        # 例：最宽 ratio 2、5 个草稿 → 7；不开投机 → 2
        widest = max((ratio for _, ratio in self.owners), default=1)
        return widest + self.speculative_tokens

    @property
    def page_fields(self):
        """The main latent of every owner. The index rows are a region apart."""
        # 【译】每个 owner 的主潜在向量。索引行在另一个区域。
        # 每个 owner 一个字段 main_<层号>，形状 (该 ratio 下每页行数, 每行宽度)，对齐到 alignment
        return [
            EntryField(
                f"main_{owner}",
                1,
                (
                    self.rows_per_page(ratio),
                    self.main_row_bytes if self.packed else self.head_dim,
                ),
                torch.uint8 if self.packed else torch.bfloat16,
                align=self.alignment,
            )
            for owner, ratio in self.owners
        ]

    def rows_per_page(self, ratio):
        """Rows one PAGE holds for an owner at this ratio, main and index alike."""
        # 【译】在这个 ratio 下，一个 PAGE 为一个 owner 装多少行，主 KV 和索引都一样。例：block 128 → ratio 1 为 128、ratio 2 为 64
        return self.block_size // ratio

    def index_blocks_per_page(self, ratio):
        """Index blocks one PAGE holds at this ratio: a block id is block
        `id % this` of the PAGE its request's column `id // this` names."""
        # 【译】在这个 ratio 下一个 PAGE 装多少个索引 block：一个（索引）block id 是"该请求 block table 第 id // this 列
        #   所指的 PAGE"里的第 id % this 个 block。
        # 例：ratio 2、64 行、index_block_rows 8 → 每页 8 个索引 block；索引 block id 19 → 第 2 列的 PAGE 里的第 3 个
        return self.rows_per_page(ratio) // self.index_block_rows

    @property
    def paged_bytes(self):
        """One PAGE's whole cost: its main rows and its index rows.

        What the pool budget is declared in. The two planes are one block id
        and one budget, laid out apart -- `paged_extents` says where.
        """
        # 【译】一个 PAGE 的全部成本：它的主 KV 行和索引行。
        #   池的预算就以它声明。两个平面是一个 block id、一份预算，分开摆放——paged_extents 说明放在哪。
        # 读取方：deepseek_v41/backend.py 的 sub_pool_specs（page_pool(self.geometry.paged_bytes)）
        return self.page_bytes + sum(
            self.rows_per_page(ratio) * self.index_row_bytes for _, ratio in self.owners
        )

    def paged_extents(self, pages):
        """`({owner: offset of each of its index_planes}, bytes the PAGE
        currency spans)`.

        An offset is from the pool's first PAGE and the total is where STATE
        begins, so this is the one walk that places everything scaling with
        history. Within a plane the pages are dense -- `index_rows *` its row
        bytes apart -- which is the stride a paged reader is handed; only the
        plane's own start is aligned.

        The total is rounded to `alignment` rather than to `plan_regions`' own,
        because the window ring behind it is addressed in whole rows.
        """
        # 【译】返回 ({owner: 它的各个 index_planes 的偏移}, PAGE 货币覆盖的字节数)。
        #   偏移从池的第一个 PAGE 算起，总数就是 STATE 开始的地方，所以这是唯一一次摆放"所有随历史增长的东西"的遍历。
        #   在一个平面内，页是紧密排列的——相隔 index_rows * 该平面的行字节——这就是交给分页读者的步长；只有平面
        #   自己的起点是对齐的。
        #   总数取整到 alignment 而不是 plan_regions 自己的对齐，因为它后面的窗口环以整行寻址。
        # 例（文件头的示意参数、pages = 1000）：区域大小 [327,680,000（主 KV）, 16,896,000, 8,448,000 × 3（各 owner 的索引平面）]
        #   → 偏移 {2: (327680000,), 8: (344576000,), 14: (353024000,), 20: (361472000,)}，总数 369,920,000（STATE 从这里开始）
        # 调用方：deepseek_v41/cache.py（切 backing）、本类 window
        offsets, total = plan_regions(
            [pages * self.page_bytes]
            + [
                pages * self.rows_per_page(ratio) * width
                for _, ratio in self.owners
                for _, width in self.index_planes
            ]
        )
        per_owner = len(self.index_planes)
        planes = offsets[1:]
        return (
            {
                owner: tuple(planes[i * per_owner : (i + 1) * per_owner])
                for i, owner in enumerate(self.compress_owners)
            },
            -(-total // self.alignment) * self.alignment,
        )

    @property
    def state_fields(self):
        window = EntryField(
            "window",
            self.layers,
            (self.ring_slots, self.window_row_bytes if self.packed else self.head_dim),
            torch.uint8 if self.packed else torch.bfloat16,
            align=self.alignment,
        )
        # The compressor's own ring, V4's `kv_state` / `score_state`: the last
        # `K_pool` raw projections per owner, so a pool window reaching back
        # before this forward reads them from here instead of the caller
        # carrying an incomplete group across the boundary.
        # Not checkpointed: CSA2 pools without overlap and a checkpoint sits on a
        # PAGE boundary (a multiple of every ratio), so a resumer reads none.
        # 【译】压缩器自己的环，相当于 V4 的 kv_state / score_state：每个 owner 最近 K_pool 个原始投影，这样回溯到本次
        #   forward 之前的池化窗口从这里读，而不用调用方把一个不完整的组跨边界带过去。
        #   不进 checkpoint：CSA2 的池化没有重叠，而 checkpoint 位于 PAGE 边界（每个 ratio 的倍数），所以续算方一个都不读。
        rings = [
            EntryField(
                name,
                len(self.owners),
                (self.compress_ring_slots, self.head_dim),
                torch.float32,
                align=self.alignment,
                in_checkpoint=False,
            )
            for name in ("compress_kv", "compress_score")
        ]
        # Position followed by compressed IDs/DEAD, oldest first.
        # 【译】先是位置，后面是压缩后的 ID / DEAD 标记，最旧的在前。（deepseek_v41/cache.py 初始化时把 ID 部分填成 -1）
        # 一个 STATE 条目的布局：
        #   │ window：layers × ring_slots 行（进 checkpoint）│ compress_kv │ compress_score（这两个不进）│ cursor：history_size + 1 个 int64（进）│
        cursor = EntryField(
            "cursor", 1, (self.history_size + 1,), torch.int64, align=self.alignment
        )
        return [window, *rings, cursor]

    # page_bytes / state_bytes：字段遍历的总字节再取整到 alignment（窗口环和主 KV 都按整行寻址）
    def _aligned_bytes(self, fields):
        return -(-entry_bytes_for(fields) // self.alignment) * self.alignment

    @property
    def page_bytes(self):
        return self._aligned_bytes(self.page_fields)

    @property
    def state_bytes(self):
        return self._aligned_bytes(self.state_fields)

    @property
    def checkpoint_image_ranges(self):
        """`(offset, num_bytes)` of a STATE entry a checkpoint image holds."""
        # 【译】checkpoint 镜像包含的 STATE 条目字节范围 (offset, num_bytes)。例（文件头参数）：[(0, 5242880), (5275648, 32)]
        return checkpoint_ranges_for(self.state_fields)

    @property
    def checkpoint_image_bytes(self):
        return sum(nbytes for _, nbytes in self.checkpoint_image_ranges)

    # main_offset：owner 的主 KV 字段在一个 PAGE 内的起点——bf16 时以"行"计，打包时以字节计（与 WindowParams 的单位一致）
    def main_offset(self, owner):
        return next(
            start // (1 if self.packed else self.row_bytes)
            for field, start, _ in field_extents(self.page_fields)
            if field.name == f"main_{owner}"
        )

    def window(self, layer, pages):
        if not 0 <= layer < self.layers:
            raise IndexError("Window layer is outside the cache")
        # Behind every PAGE-scaled region, main and index alike -- not just the
        # main pages.
        # 【译】在所有随 PAGE 增长的区域之后——主 KV 页和索引平面都算，而不只是主 KV 页。
        # 返回第 layer 层窗口环的 WindowParams（公式见 v4_pool_geometry.WindowParams），以整个 backing 为原点：
        #   bf16：单位是一行（row_bytes）——ring_start = STATE 起点的行号 + layer × ring_slots，slot_rows = state_bytes / row_bytes
        #   打包：单位是字节——ring_stride = 1、run_rows = window_row_bytes，于是 index() 直接给出字节偏移
        # 例（文件头参数）：bf16 时 slot_rows = 5,276,672 / 1024 = 5153
        state_start = self.paged_extents(pages)[1]
        if self.packed:
            return WindowParams(
                ring_start=state_start
                + layer * self.ring_slots * self.window_row_bytes,
                slot_rows=self.state_bytes,
                ring_slots=self.ring_slots,
                ring_stride=1,
                run_rows=self.window_row_bytes,
            )
        return WindowParams(
            ring_start=state_start // self.row_bytes + layer * self.ring_slots,
            slot_rows=self.state_bytes // self.row_bytes,
            ring_slots=self.ring_slots,
            ring_stride=self.ring_slots,
            run_rows=self.ring_slots,
        )

    @property
    def layout_id(self):
        identity = (
            # v2: the compressor's incomplete-group tail became a K_pool ring.
            # Every other term below was already the same in v1, so without the
            # bump a v1 image would read as compatible and restore into fields
            # that no longer mean what it holds.
            # v3: the image no longer carries the compressor tails, so it is
            # no longer the whole entry; a v2 image would restore its tail
            # bytes over the cursor.
            #
            # The index plane's format is here even though no STATE field holds
            # an index row: this string is what a resumed prefix is matched on,
            # and a plane at a different precision selects different rows, so
            # the prefix is a different computation. It is spelled out rather
            # than read off a field because FP8 is the only plane the scorer
            # takes -- an image from a build that had another must not match.
            # 【译】v2：压缩器的"不完整组尾巴"变成了一个 K_pool 环。下面其余各项在 v1 里就已经一样，所以不升版本号的话，
            #   v1 的镜像会被当作兼容，恢复进那些含义已经变了的字段里。
            #   v3：镜像不再携带压缩器尾巴，所以它不再是整个条目；v2 的镜像会把它的尾巴字节恢复到 cursor 上。
            #
            #   索引平面的格式也写在这里，尽管没有任何 STATE 字段装着索引行：续算的前缀就是按这个字符串匹配的，而不同精度
            #   的平面会选出不同的行，所以那个前缀是一次不同的计算。它被明确写出来，而不是从某个字段读出，因为 FP8 是打分器
            #   唯一接受的平面——来自曾有别的平面的构建的镜像不能匹配上。
            # layout_id 的用处：状态快照（页单元 checkpoint、KV offload）用它判断"这份镜像能否被这个 worker 按同样的布局读回"
            #   （第4批 page_unit_checkpoint.py 的 PagedStateCheckpointSpec.layout_id）
            # 观察：字符串里写死了 "index-fp8"，没有反映 index_fp4；而 V4.1 的运行时门控允许 --index_cache_dtype fp4
            #   （models/deepseek_v41/config.py："an index plane other than fp8 or fp4" 才拒绝），本类也支持 FP4 page-8 平面。
            #   于是 FP4 索引的构建与 FP8 索引的构建得到相同的 layout_id，与上面"不同精度的平面会选出不同的行，所以前缀是
            #   不同的计算"的理由不符（上面"FP8 是唯一接受的平面"的说法也已过时）
            f"dsv41-{'packed' if self.packed else 'bf16'}-index-fp8"
            f"-state-v3:layers={self.layers}:owners={self.owners}"
            f":block={self.block_size}:window={self.window_size}"
            f":dims={self.head_dim},{self.index_dim}:history={self.history_size}"
        )
        return identity + (
            f":spec={self.speculative_tokens}" if self.speculative_tokens else ""
        )
