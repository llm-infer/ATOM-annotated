# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in an MHA paged pool.

The MHA counterpart of ``page_unit_geometry``. That module is a mixin over an
MLA cache ``(rows, blocks, block_size, entry)`` and only ``_KimiMLAGDNCommon``
mixes it in. An MHA pool publishes one contiguous region per field per layer,
with alignment padding between fields that is not a region, so these are free
functions over the pool's ``region_tensors`` rather than methods on
``self.model_runner``.

``GDNAttentionMetadataBuilder`` is the caller, the same role
``kimi_mla_gdn_attn`` plays for the MLA mixin: ``_page_unit_regions``,
``_page_unit_bases``, ``_page_unit_stream_sizes``, and ``page_unit_views``
there forward here. No aiter, so the byte arithmetic can be tested without a
GPU.
"""

# 【译】一份 checkpoint 镜像的字节住在 MHA 分页池的什么地方。
#
#   page_unit_geometry 的 MHA 对应物。那个模块是架在 MLA 缓存 (rows, blocks, block_size, entry) 上的 mixin，只有
#   _KimiMLAGDNCommon 混入它。MHA 池对每个字段的每一层发布一段连续区域，字段之间的对齐填充不算区域，所以这里是
#   作用于池的 region_tensors 的自由函数，而不是 self.model_runner 上的方法。
#
#   调用方是 GDNAttentionMetadataBuilder，和 kimi_mla_gdn_attn 对 MLA mixin 扮演的角色相同：那里的
#   _page_unit_regions、_page_unit_bases、_page_unit_stream_sizes 和 page_unit_views 都转发到这里。不依赖 aiter，
#   所以字节算术可以在没有 GPU 的情况下测试。
#
# =====================================================================================
# 【导读】mha_page_unit_geometry.py —— 页单元在 MHA 池里的地址（第5批 5b，B级）
#   前置：页单元见第4批 state_pool.py【科普】⑤；MHA 池的 layer-major 布局见 pool_layout/__init__.py【科普】① 与
#     entry_arena.LayerMajorArena；拷贝本身见 paged_state_copy.py。
#   使用场景：Qwen3-Next / Qwen3.5 这类 GDN 混合模型（gdn_attn.py GDNAttentionMetadataBuilder）用 copy 方式做状态快照时，
#     快照字节写进全注意力层的 MHA 池。MhaKvPool.region_tensors 按"一个字段的所有层，再下一个字段"给出区域，
#     每个区域 = 一个 (字段, 层) 的 [block, 每 block 字节]，一个 block 在其中是连续的一行。
#   带数字（Qwen3-Next：12 层全注意力、2 个 KV 头、head_dim 256、bf16、block_size 16，无 scale）：
#     区域 = k.layer_0 … k.layer_11, v.layer_0 … v.layer_11 共 24 个，每个区域每 block 16 × 2 × 256 × 2 B = 16 KiB
#     → 一个页单元 = 24 × 16 KiB = 384 KiB（= state_pool.py【科普】⑤ 的数字）；一份 37.7 MiB 的 GDN 状态镜像
#       = ceil(39,518,208 / 393,216) = 101 个页单元 = 101 × 24 = 2424 段；最后一个单元只用到一半（按 image_bytes 截断）
#   对比 MLA 版（page_unit_geometry）：那边区域按 MLA 行（层）分；这边按 (字段, 层) 分，fp8 时还有 k_scale / v_scale，
#     MiniMax-M3 一类还有 extra 字段——全由 MhaKvPool.region_tensors 决定，这里不认识任何字段名。
# =====================================================================================

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def mha_published_page_bytes(pools: Iterable) -> int:
    """Bytes one scheduler block publishes as MHA PAGE regions.

    ``MhaKvPool.entry_bytes`` also counts alignment padding between fields.
    Those gaps are not part of any ``region_tensors`` row, and ``lmcache_mp``
    requires the checkpoint unit to equal the sum of the published regions.
    """
    # 【译】一个调度器 block 作为 MHA PAGE 区域发布出去的字节数。
    #   MhaKvPool.entry_bytes 还算上了字段之间的对齐填充。那些空隙不属于任何 region_tensors 行，而 lmcache_mp 要求
    #   checkpoint 单元等于已发布区域之和。
    # 例（Qwen3-Next bf16）：24 个区域 × 16 KiB = 393,216 B（各字段字节本就是 256 的倍数时与 entry_bytes 相同）
    # 调用方：gdn_attn.py GDNAttentionMetadataBuilder.checkpoint_page_unit_bytes（收窄 PagedStateCheckpointSpec.page_unit_bytes）

    total = 0
    for pool in pools:
        for group in pool.field_groups:
            for field in group:
                total += field.bytes_per_entry
    return total


def mha_page_unit_regions(pools: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Base address and per-block stride of every published MHA region.

    Order is ``MhaKvPool.region_tensors``: all of one field, then the next,
    which is the order ``AiterAttentionMetadataBuilder.get_kv_transfer_tensors``
    publishes. One region is one layer of one field, and its stride is one
    scheduler block.
    """
    # 【译】每个已发布 MHA 区域的基址和"每 block 步长"。
    #   顺序就是 MhaKvPool.region_tensors 的顺序：一个字段的全部，然后下一个字段，也就是
    #   AiterAttentionMetadataBuilder.get_kv_transfer_tensors 发布的顺序。一个区域是一个字段的一层，它的步长是一个调度器 block。
    # 输出：(bases, sizes)，sizes[k] = 区域 k 每 block 的字节（= tensor.stride(0) × 元素字节）
    # 调用方：gdn_attn.py GDNAttentionMetadataBuilder._page_unit_regions（按全部区域张量的布局做缓存 key，并核对 page_unit_bytes）

    bases: list[int] = []
    sizes: list[int] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"MHA PAGE region {role} is not a tensor")
            if not tensor.is_contiguous():
                raise RuntimeError(
                    "an MHA PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            unit = tensor.stride(0) * tensor.element_size()
            if unit <= 0:
                raise RuntimeError(f"MHA PAGE region {role} has no bytes")
            bases.append(tensor.data_ptr())
            sizes.append(unit)
    if not bases:
        raise RuntimeError("GDN PAGE checkpoint found no MHA regions")
    return np.array(bases, dtype=np.int64), np.array(sizes, dtype=np.int64)


def mha_page_unit_bases(regions: tuple[np.ndarray, np.ndarray], unit_ids) -> np.ndarray:
    """Start address of every destination segment, one row per image.

    Same product as ``PageUnitGeometryMixin._page_unit_bases``. ``unit_ids``
    is ``(images, units_per_checkpoint)``, unit major and region minor, which
    is the order the copy plan tiled the destination stream in.
    """
    # 【译】每个目的段的起始地址，每份镜像一行。
    #   与 PageUnitGeometryMixin._page_unit_bases 是同一个乘积。unit_ids 的形状是 (images, units_per_checkpoint)，单元为
    #   主序、区域为次序，也就是拷贝计划铺设目的流的顺序。
    # 调用方：gdn_attn.py _page_unit_bases；本文件 mha_page_unit_segments

    base, stride = regions
    ids = np.asarray(unit_ids, dtype=np.int64)
    return (base + ids[..., None] * stride).reshape(len(ids), -1)


def mha_page_unit_stream_sizes(
    regions: tuple[np.ndarray, np.ndarray], units: int
) -> np.ndarray:
    """Bytes in each destination segment of an image of ``units`` units.

    Same tile as ``PageUnitGeometryMixin._page_unit_stream_sizes``.
    """
    # 【译】一份由 units 个单元组成的镜像里，每个目的段的字节数。
    #   与 PageUnitGeometryMixin._page_unit_stream_sizes 是同一种 tile。调用方：gdn_attn.py _page_unit_stream_sizes

    return np.tile(regions[1], units)


def mha_page_unit_segments(
    regions: tuple[np.ndarray, np.ndarray],
    unit_ids,
    *,
    entries: int,
    image_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Addresses and byte sizes of the stream ``mha_page_unit_views`` names.

    Same order and the same ``image_bytes`` trim, without one tensor view per
    (unit, region). An image is ~10k segments, and building those views held
    the store thread's GIL for tens of milliseconds per checkpoint, which the
    scheduler thread then waited out.
    """
    # 【译】mha_page_unit_views 所命名的那条流的地址和字节大小。
    #   顺序相同、image_bytes 的截断也相同，只是不为每个 (单元, 区域) 建一个张量视图。一份镜像大约 1 万段，建那些视图
    #   曾让存储线程每个 checkpoint 持有 GIL 几十毫秒，调度线程随后只好等它。
    # 截断：ends = 各段累计字节；searchsorted 找到第一个累计 ≥ image_bytes 的段，把它的大小减掉多出的部分
    # 例：段大小 [16384] × 2424、image_bytes 39,518,208 → 恰好是前 2412 段（16384 × 2412），最后一个单元只用到
    #   前 12 个区域，其余 12 段丢弃；若不整除，最后保留的那段会被减短
    # 调用方：gdn_attn.py GDNAttentionMetadataBuilder.page_unit_segments ← kv_transfer/offload/hybrid/kimi_k3/state_object.py（存储时的 gather）

    ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= entries):
        raise IndexError(f"PAGE unit outside the pool's {entries} scheduler blocks")
    ptrs = mha_page_unit_bases(regions, ids[None, :])[0]
    sizes = mha_page_unit_stream_sizes(regions, ids.size)
    budget = int(image_bytes or 0)
    if budget <= 0:
        return ptrs, sizes
    ends = np.cumsum(sizes)
    if not ends.size or ends[-1] < budget:
        held = int(ends[-1]) if ends.size else 0
        raise RuntimeError(
            f"a checkpoint image is {budget} B but its {sizes.size} unit "
            f"segments hold {held} B"
        )
    last = int(np.searchsorted(ends, budget))
    sizes = sizes[: last + 1].copy()
    sizes[-1] -= int(ends[last]) - budget
    return ptrs[: last + 1], sizes


def mha_page_unit_views(
    pools: Iterable, unit_ids, *, image_bytes: int = 0
) -> list[torch.Tensor]:
    """Tensor views of one checkpoint image, unit major and region minor.

    The MHA counterpart of ``PageUnitGeometryMixin.page_unit_views``. The
    in-process state tier packs these and later unpacks the same byte stream
    into an Active Slot. Order matches ``mha_page_unit_bases``. A whole unit
    is ``ceil(image / unit)`` scheduler blocks, so the tail past
    ``image_bytes`` is padding and must not be offered to the packer.
    """
    # 【译】一份 checkpoint 镜像的张量视图，单元为主序、区域为次序。
    #   PageUnitGeometryMixin.page_unit_views 的 MHA 对应物。进程内状态层打包这些视图，之后把同样的字节流解包进一个
    #   活跃槽位。顺序与 mha_page_unit_bases 一致。一个整单元是 ceil(image / unit) 个调度器 block，所以 image_bytes
    #   之后的尾部是填充，不能交给打包器。
    # 输出：每个 (单元, 区域) 一个连续视图 tensor[unit]（形状 [每 block 字节]），最后一个可能被截成 uint8 前缀
    # 调用方：gdn_attn.py GDNAttentionMetadataBuilder.page_unit_views

    rows: list[tuple[str, torch.Tensor]] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"MHA PAGE region {role} is not a tensor")
            if tensor.ndim < 1 or not tensor.is_contiguous():
                raise RuntimeError(
                    "an MHA PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            rows.append((role, tensor))
    if not rows:
        raise RuntimeError("GDN PAGE checkpoint found no MHA regions")
    entries = int(rows[0][1].shape[0])
    views: list[torch.Tensor] = []
    for unit in unit_ids:
        unit = int(unit)
        if not 0 <= unit < entries:
            raise IndexError(
                f"PAGE unit {unit} outside the pool's {entries} scheduler blocks"
            )
        for role, tensor in rows:
            row = tensor[unit]
            if not row.is_contiguous():
                raise RuntimeError(
                    f"MHA PAGE region {role} block {unit} is not contiguous"
                )
            views.append(row)
    budget = int(image_bytes or 0)
    if budget <= 0:
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
            f"a checkpoint image is {image_bytes} B but its {len(views)} "
            f"unit views hold {image_bytes - budget} B"
        )
    return out
