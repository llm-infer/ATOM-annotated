# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in the MLA paged pool.

The destination side of a K3 PAGE-copy checkpoint: given logical block (PAGE
unit) ids, it gives the addresses and tensor views of the KV pool a checkpoint
copy lands on. `_KimiMLAGDNCommon` mixes it in over `GDNStateMixin`.

Pure over `self.model_runner`, and here rather than beside that builder for the
reason the package exists (see `pool_layout/__init__.py`): the geometry deciding
where every checkpoint byte goes is then testable on a GPU-less runner, which
the K3 builder's own module -- `importorskip`ped on CI -- is not.
"""

# 【译】一份 checkpoint 镜像的字节住在 MLA 分页池的什么地方。
#
#   Kimi-K3 PAGE 拷贝式 checkpoint 的目的侧：给定逻辑 block（PAGE 单元）号，给出 checkpoint 拷贝落到的 KV 池地址
#   和张量视图。_KimiMLAGDNCommon 把它混入在 GDNStateMixin 之上。
#
#   只读 self.model_runner；放在这里而不是那个 builder 旁边，理由就是这个包存在的理由（见 pool_layout/__init__.py）：
#   决定每个 checkpoint 字节去向的几何因此可以在没有 GPU 的机器上测试，而 K3 builder 自己的模块——在 CI 上被
#   importorskip 跳过——做不到。
#
# =====================================================================================
# 【导读】page_unit_geometry.py —— 页单元在 MLA 池里的地址（第5批 5b，B级）
#   前置：页单元 = KV 池里一个 block 的全部字节，一份状态快照占 units_per_checkpoint 个不必连续的 block（第4批
#     state_pool.py 文件开头【科普】⑤）；MLA 池的形状见 pool_layout/__init__.py【科普】③；拷贝本身见 paged_state_copy.py。
#   混入方：kimi_mla_gdn_attn.py 的 _KimiMLAGDNCommon（KIMI_MLA 家族：Kimi-K3、GLM-5.3-Flash 这类 KDA + MLA 混合模型），
#     MRO 在 GDNStateMixin 之前，所以 GDNStateMixin 里抛 NotImplementedError 的 _page_unit_regions 被这里的实现覆盖。
#   一个 MLA 池的 KV 视图 = [R 行（MLA 层）, 逻辑 block, block_size, 576]，层在外（LayerMajorArena），所以一个 block 的
#     字节散在 R 处、彼此相隔 row_stride = num_blocks × 一个区域：
#       行0：[blk0│blk1│…│blk u│…]  行1：[blk0│…│blk u│…]  …  （+ 带索引器时，每个索引层再一段）
#     → 页单元 u 的第 k 个区域起点 = bases[k] + u × sizes[k]（行内 block 首尾相连，步长恰好等于区域大小）
#   带数字（示意：R = 2、block_size 4、bf16、num_blocks 10，无索引器）：一个区域 = 4 × 576 × 2 B = 4608 B，
#     row_stride = 10 × 4608 = 46080 → bases = [b, b + 46080]，sizes = [4608, 4608]，页单元 = 9216 B；
#     一份占单元 [7, 3] 的镜像，目的段起点（单元为主序、区域为次序）= [b + 32256, b + 78336, b + 13824, b + 59904]
#   对外三件事：_page_unit_bases / _page_unit_stream_sizes 给 HBM 拷贝的描述符（GDNStateMixin 的 _checkpoint_copy_plan、
#     execute_paged_state_copies、warmup_per_req_cache 调用）；page_unit_views 给 CPU 状态层（KV offload）打包用
# =====================================================================================

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class PageUnitGeometryMixin:
    """PAGE-unit address/view arithmetic for the MLA paged pool.

    Reads only `self.model_runner` (`kv_cache`, `block_size`, `state_runtime`)
    and caches the pool-derived regions on `self._page_unit_region_cache`. No
    aiter, no CUDA -- the concrete K3 builder supplies the pool.
    """

    # 【译】MLA 分页池的 PAGE 单元地址 / 视图算术。
    #   只读 self.model_runner（kv_cache、block_size、state_runtime），并把由池推导出的区域缓存在
    #   self._page_unit_region_cache 上。没有 aiter、没有 CUDA——具体的 K3 builder 提供池。
    # （实际读的是 self.kv_pool.cache.view("kv")，见 _page_unit_kv_cache；"kv_cache" 是旧说法）

    def _page_unit_index_cache(self) -> torch.Tensor | None:
        """The sparse-indexer cache that rides this PAGE unit, or `None`.

        Plain MLA owns only its KV rows, so the generic geometry has no index
        cache. A sparse-indexer model (`runner.has_mla_indexer`) overrides this
        to return the pool's index view, whose layers `_page_unit_regions`
        appends after the MLA rows.
        """
        # 【译】搭在这个 PAGE 单元上的稀疏索引器缓存，没有则为 None。
        #   普通 MLA 只拥有它的 KV 行，所以通用几何没有索引缓存。稀疏索引器模型（runner.has_mla_indexer）覆盖这个方法，
        #   返回池的 index 视图，_page_unit_regions 把它的各层追加在 MLA 行之后。
        # 覆盖方：kimi_mla_gdn_attn.py _KimiMLAGDNCommon._page_unit_index_cache（has_mla_indexer 时返回 kv_pool.index.view("index")）
        return None

    def _page_unit_kv_cache(self):
        """The MLA rows a PAGE unit owns a region of.

        The pool's own view, `(rows, blocks, block_size, entry)`. A hook for the
        same reason `_page_unit_index_cache` is one: this module owns the
        arithmetic, not where the tensor comes from.
        """
        # 【译】PAGE 单元在其中各拥有一个区域的那些 MLA 行。
        #   即池自己的视图 (rows, blocks, block_size, entry)。做成钩子的理由和 _page_unit_index_cache 一样：本模块拥有的是
        #   算术，而不是张量从哪来。
        return self.kv_pool.cache.view("kv")

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        The pool's KV view is `(rows, blocks, block_size, entry)`: a block owns
        one contiguous region per row, rows a fixed stride apart. Affine in the
        block id and a pool property, so it is computed once.

        The units were the trap here, and the view is what takes it away: it is
        shaped in the same **logical** blocks `unit_ids` carries, so a region is
        `runner.block_size` tokens wide by construction rather than by
        remembering that the allocation pages at 1 (K3's `block_ratio` is 128).
        The granularity assertion below still checks the relation.
        """
        # 【译】一个 PAGE id 拥有的每个区域的基址和"每单元步长"。
        #   池的 KV 视图是 (rows, blocks, block_size, entry)：一个 block 在每一行里拥有一段连续区域，行与行之间相隔固定步长。
        #   它对 block id 是仿射的，又是池的性质，所以只算一次。
        #   这里的陷阱在于单位，而视图消除了它：视图按 unit_ids 所用的同一种"逻辑" block 排列，所以一个区域按构造就是
        #   runner.block_size 个 token 宽，而不必记住"分配时按页大小 1 分页"（K3 的 block_ratio 是 128）。下面的粒度断言
        #   仍然会检查这个关系。
        # 输出：(bases, sizes) 两个 int64 数组，长度 = R 个 MLA 行 + I 个索引层；页单元 u 的第 k 段在 bases[k] + u × sizes[k]
        # 调用方：本类 _page_unit_bases / _page_unit_stream_sizes / page_unit_views
        runner = self.model_runner
        cache = self._page_unit_kv_cache()
        index_cache = self._page_unit_index_cache()
        # Key on the full layout, not `data_ptr()` alone. The pool is rebuilt in
        # two places -- the P/D IPC import (`_bind_kv_cache_to_modules`) and the
        # rollout sleep/wake path (`rollout/memory_manager.py` releases it, then
        # `_resume_kv_cache` -> `allocate_kv_cache` with a possibly *reduced*
        # block count). Address
        # equality does not imply geometry equality: a same-start-address
        # reallocation keeps `data_ptr()` but changes `stride(0)`, so `row_stride`
        # (and thus every base for row > 0) goes stale while `rows`/`region`
        # survive -- and the granularity assertion below could not catch it even
        # if the early return did not skip it, since it never involves
        # `row_stride`. Keying on `(data_ptr, shape, stride, element_size)` makes
        # any of those changing a cache miss that rebuilds. Mirrors the DSV4
        # sibling (`deepseek_v4_attn.py`), which keys on the same evidence.
        #
        # A sparse-indexer model (`runner.has_mla_indexer`) rides its index cache
        # in the same PAGE unit, appended after the MLA rows. Its layout is part
        # of the identity too -- key on it (or `None`) so a reallocated or
        # resized index cache is a miss that rebuilds, exactly like the MLA pool.
        # 【译】以完整布局作为缓存的 key，而不只是 data_ptr()。池会在两个地方被重建——P/D 的 IPC 导入
        #   （_bind_kv_cache_to_modules），以及 rollout 的休眠/唤醒路径（rollout/memory_manager.py 释放它，然后
        #   _resume_kv_cache -> allocate_kv_cache，block 数可能"变少"）。地址相等不代表几何相等：起始地址相同的重新分配
        #   保留 data_ptr()，却改变了 stride(0)，于是 row_stride（以及 row > 0 的每个基址）变旧，而 rows / region 还是对的
        #   ——而且即使没有被提前返回跳过，下面的粒度断言也抓不到它，因为那个断言根本不涉及 row_stride。以
        #   (data_ptr, shape, stride, element_size) 为 key，任何一项变了都会变成缓存未命中、重新构建。这与 DSV4 的兄弟实现
        #   （deepseek_v4_attn.py）一致，它也以同样的证据为 key。
        #
        #   稀疏索引器模型（runner.has_mla_indexer）把它的索引缓存搭在同一个 PAGE 单元里，追加在 MLA 行之后。它的布局也是
        #   身份的一部分——以它（或 None）为 key，这样重新分配或改了大小的索引缓存也会像 MLA 池一样未命中并重建。
        index_owner = (
            None
            if index_cache is None
            else (
                index_cache.data_ptr(),
                tuple(index_cache.shape),
                tuple(index_cache.stride()),
                index_cache.element_size(),
            )
        )
        owner = (
            cache.data_ptr(),
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.element_size(),
            index_owner,
        )
        cached = getattr(self, "_page_unit_region_cache", None)
        if cached is not None and cached[0] == owner:
            return cached[1]

        if not cache.is_contiguous():
            raise RuntimeError("the MLA pool must be contiguous to be copied")
        item = cache.element_size()
        entry = cache.shape[3]
        rows = cache.shape[0]
        # One logical block's bytes inside one row.
        # 【译】一个逻辑 block 在一行里的字节数。（例：block_size 4 × 576 × 2 B = 4608；row_stride = stride(0) × 元素字节 = 一层的全部 block）
        region = runner.block_size * entry * item
        row_stride = cache.stride(0) * item

        base_addr = cache.data_ptr()
        bases = [base_addr + row * row_stride for row in range(rows)]
        sizes = [region] * rows

        index_layers = 0
        index_region = 0
        if index_cache is not None:
            if not index_cache.is_contiguous():
                raise RuntimeError("the index cache must be contiguous to be copied")
            index_item = index_cache.element_size()
            index_layers = index_cache.shape[0]
            layer_stride = index_cache.stride(0) * index_item
            index_region = index_cache.stride(1) * index_item
            bases += [
                index_cache.data_ptr() + layer * layer_stride
                for layer in range(index_layers)
            ]
            sizes += [index_region] * index_layers

        # 粒度核对：一个页单元在这里拥有的字节（R 个区域 + I 个索引区域）必须等于定大小时的 page_unit_bytes，
        #   否则说明"逻辑 block"与"物理页"的换算在某处错了
        runtime = getattr(runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        owned = sum(sizes)
        page_unit_bytes = spec.page_unit_bytes if spec is not None else owned
        if owned != page_unit_bytes:
            gives = f"{rows} rows x {region} B"
            if index_cache is not None:
                gives += f" + {index_layers} index layers x {index_region} B"
            raise RuntimeError(
                f"a PAGE unit is {page_unit_bytes} B but this pool gives a "
                f"logical block {gives} = {owned} B; the two disagree about "
                "block granularity"
            )
        regions = (
            np.array(bases, dtype=np.int64),
            np.array(sizes, dtype=np.int64),
        )
        self._page_unit_region_cache = (owner, regions)
        return regions

    def _page_unit_bases(self, unit_ids: Sequence[Sequence[int]]) -> np.ndarray:
        """Start address of every destination segment, one row per image.

        `unit_ids` is `(images, units_per_checkpoint)`. A unit's regions are
        each at `base + id * stride`, so one image's worth is an outer product
        and a batch's is the same product with an image axis in front. Unit
        major, region minor -- the order `_checkpoint_copy_plan` built the
        destination stream in.
        """
        # 【译】每个目的段的起始地址，每份镜像一行。
        #   unit_ids 的形状是 (images, units_per_checkpoint)。一个单元的各区域分别在 base + id * stride 处，所以一份镜像就是
        #   一个外积，一个 batch 就是在前面加一个镜像轴的同一个外积。单元为主序、区域为次序——就是 _checkpoint_copy_plan
        #   构建目的流时的顺序。
        # 输出形状 (images, units × 区域数)：就是 SegmentedCopyPlan.write_descriptor 的 dst_bases；例子见文件头
        # ids[..., None] * stride：广播成 (images, units, 区域数)，再 reshape 成每份镜像一行
        base, stride = self._page_unit_regions()
        ids = np.asarray(unit_ids, dtype=np.int64)
        return (base + ids[..., None] * stride).reshape(len(ids), -1)

    def _page_unit_stream_sizes(self, units: int) -> np.ndarray:
        """Bytes in each destination segment of an image of `units` units."""
        # 【译】一份由 units 个单元组成的镜像里，每个目的段的字节数。np.tile 把一个单元的各区域大小重复 units 次，
        #   作为 plan_segmented_copy 的 dst_sizes（和 _page_unit_bases 的展开顺序一致）
        return np.tile(self._page_unit_regions()[1], units)

    def page_unit_views(self, unit_ids: Sequence[int]) -> list[torch.Tensor]:
        """Tensor-view counterpart of `_page_unit_regions`, for the CPU tier.

        `_page_unit_regions` gives raw addresses for the Triton descriptor; the
        LMCache packer takes `list[torch.Tensor]`, so the same bytes need naming
        as views. Order is the contract: unit major, row minor, the same ravel
        as `_page_unit_bases` (cross-build safety is `layout_id` in
        `StateByteCodec.key`, not this order).

        `unit_ids` carries *logical* block ids while `kv_cache` is shaped in
        *physical* ones (K3's ratio is 128); indexing by `unit * block_size` is
        that conversion, range-checked against the logical count.

        `_page_unit_regions` runs first only for its side effect (the contiguity
        and granularity checks). Its return is for the Triton descriptor, not
        this view path -- called bare, not unpacked.
        """
        # 【译】_page_unit_regions 的张量视图版本，给 CPU 层用。
        #   _page_unit_regions 给 Triton 描述符的是裸地址；LMCache 打包器接收的是 list[torch.Tensor]，所以同样的字节需要
        #   以视图的形式命名。顺序就是契约：单元为主序、行为次序，与 _page_unit_bases 的展开顺序相同（跨构建的安全性靠
        #   StateByteCodec.key 里的 layout_id，而不是这个顺序）。
        #   unit_ids 携带的是"逻辑" block 号，而 kv_cache 是按"物理" block 排的（K3 的比例是 128）；按 unit * block_size
        #   索引就是这种换算，并对逻辑个数做范围检查。
        #   先调 _page_unit_regions 只是为了它的副作用（连续性与粒度检查）。它的返回值是给 Triton 描述符的，不是给这条
        #   视图路径的——所以只调用、不解包。
        # 解释：现在 _page_unit_kv_cache 返回的 MlaKvPool 视图本身就按逻辑 block 排（[行, 逻辑 block, block_size, 576]）；
        #   这里把每行摊平成 [token, 576] 再用 unit × block_size 定位，对两种形状结果相同
        # 调用方：kv_transfer/offload/hybrid/kimi_k3/state_object.py（CPU 状态层存储时打包这些视图）
        # 观察：这里的视图只遍历 MLA 行（planes 来自 _page_unit_kv_cache），不包含 _page_unit_regions 追加在每个单元里的
        #   索引层区域；对带索引器的模型（runner.has_mla_indexer，如同属 KIMI_MLA 家族的 GLM-5.3-Flash，见 utils/selector.py
        #   has_mla_indexer），CPU 层打包的视图流与 HBM 拷贝写下的"每单元：MLA 行 + 索引层"字节流顺序不同。kimi_k3 offload
        #   connector 没有针对 has_mla_indexer 的拒绝；这条路径实际会不会被走到，留待第9批（KV offload）核对
        self._page_unit_regions()
        cache = self._page_unit_kv_cache()
        rows, entry = cache.shape[0], cache.shape[3]
        block = self.model_runner.block_size
        # Logical blocks the pool holds, which is what a unit id indexes.
        # Range-checked against THIS count, not the physical one: the physical
        # count is `block_ratio` times smaller, so checking it would pass ids
        # that address past the end of the tensor.
        # 【译】池里的逻辑 block 数，也就是 unit id 所索引的东西。按"这个"数做范围检查，而不是按物理数：物理数比它小
        #   block_ratio 倍，用它检查会放过那些寻址越过张量末尾的 id。
        logical_blocks = (cache.shape[1] * cache.shape[2]) // block
        # planes[r] = 第 r 行摊平成 [逻辑 block 数 × block_size, 576]；单元 u 在每行里是 [u × block, (u + 1) × block) 这些 token
        planes = [cache[row].view(-1, entry) for row in range(rows)]
        views: list[torch.Tensor] = []
        for unit in unit_ids:
            unit = int(unit)
            if not 0 <= unit < logical_blocks:
                raise IndexError(
                    f"PAGE unit {unit} outside the pool's {logical_blocks} "
                    "logical blocks"
                )
            lo = unit * block
            views.extend(plane[lo : lo + block] for plane in planes)

        # ---- cut the padding tail off the final unit ----
        #
        # An image occupies `ceil(image_bytes / page_unit_bytes)` WHOLE units,
        # so the last one is mostly padding. The HBM copy already trims it
        # (`_checkpoint_copy_plan` passes `plan_segmented_copy` both the unit
        # stream and `spec.image_bytes`); gathering whole units for the CPU tier
        # skipped that and offered the packer the padding too, against a
        # MemoryObj `StateByteCodec.put` sizes at `entry_bytes` == `image_bytes`
        # -- so every store died with "MemoryObj tensor is too small".
        #
        # Byte-exact, not view-aligned: an image need not end on a row boundary,
        # so the boundary view is reinterpreted as `uint8` and sliced (the
        # packer sizes segments as `numel * element_size`). The load side needs
        # no counterpart -- it scatters into `state_entry_views`, which sums to
        # `entry_bytes`. Inline, not a helper: the tests drive this with a
        # `SimpleNamespace` `self`, and a helper would be one more stub each.
        # 【译】---- 把最后一个单元的填充尾巴切掉 ----
        #
        #   一份镜像占 ceil(image_bytes / page_unit_bytes) 个"整"单元，所以最后一个大部分是填充。HBM 拷贝已经裁掉了它
        #   （_checkpoint_copy_plan 把单元流和 spec.image_bytes 都传给 plan_segmented_copy）；为 CPU 层收集整单元时跳过了
        #   这一步，把填充也交给了打包器，而 StateByteCodec.put 按 entry_bytes == image_bytes 给 MemoryObj 定大小——于是每次
        #   存储都以 "MemoryObj tensor is too small" 失败。
        #
        #   按字节精确、而不是按视图对齐：一份镜像不一定在行边界结束，所以边界上的那个视图被重解释为 uint8 再切片（打包器
        #   按 numel * element_size 计算段大小）。加载侧不需要对应处理——它散写进 state_entry_views，后者加起来正好是
        #   entry_bytes。写成内联而不是辅助函数：测试用一个 SimpleNamespace 作 self 来驱动它，多一个辅助函数就要多一个桩。
        # 例：image_bytes = 10000、单元视图各 4608 B → 取前两个整视图（9216 B），第三个只取前 784 字节
        runtime = getattr(self.model_runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        # `getattr`: a fork build or a test stand-in carries no `image_bytes` --
        # no image size to trim against, so the whole-unit stream is correct.
        # 【译】用 getattr：fork 出来的构建或测试替身没有 image_bytes——没有镜像大小可以裁，所以整单元的流就是对的。
        budget = int(getattr(spec, "image_bytes", 0) or 0) if spec else 0
        if not budget:
            return views
        out: list[torch.Tensor] = []
        for view in views:
            nbytes = view.numel() * view.element_size()
            if budget >= nbytes:
                out.append(view)
                budget -= nbytes
                continue
            if budget > 0:
                out.append(view.reshape(-1).view(torch.uint8)[:budget])
                budget = 0
            break
        if budget:
            raise RuntimeError(
                f"a checkpoint image is {spec.image_bytes} B but its "
                f"{len(views)} unit views hold "
                f"{int(spec.image_bytes) - budget} B; "
                "the unit geometry and the image size disagree"
            )
        return out
