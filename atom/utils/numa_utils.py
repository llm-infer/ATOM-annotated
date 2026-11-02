# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""NUMA binding for GPU worker processes.

Pin each GPU worker to the CPU cores and preferred memory of its GPU's local
NUMA node. The GPU->node mapping is auto-detected (``amdsmi`` -- the ROCm analog
of NVML -- with a sysfs fallback) so the operator needs no per-machine config;
an explicit per-rank node list can override it.

Public entry point: :func:`numa_bind_to_node`, called once at worker start.
Knobs live in :mod:`atom.utils.envs` (``ATOM_NUMA_BIND``, ``ATOM_AUTO_NUMA_BIND``,
``ATOM_NUMA_NODE``, ``ATOM_CRASH_ON_NUMA_BIND_FAILURE``).
"""

# 【译】GPU worker进程的NUMA绑定。
#
#   把每个GPU worker固定到它的GPU所在NUMA本地节点的CPU核和首选内存上。GPU->节点的映射会自动检测
#   （amdsmi——ROCm版的NVML——并以sysfs作为后备），所以运维不需要逐台机器配置；也可以用显式的逐rank节点列表覆盖它。
#
#   公开入口：numa_bind_to_node，在worker启动时调用一次。
#   开关都在 atom.utils.envs 里（ATOM_NUMA_BIND、ATOM_AUTO_NUMA_BIND、ATOM_NUMA_NODE、ATOM_CRASH_ON_NUMA_BIND_FAILURE）。

import ctypes
import glob
import logging
import os

from atom.utils import envs

logger = logging.getLogger("atom")


# 本文件在系统中的位置：ModelRunner进程启动时（AsyncIOProc.__init__，在构造ModelRunner之前）调用numa_bind_to_node
#   为什么要绑：多路CPU服务器上，每张GPU挂在某一个CPU插槽（NUMA节点）下。worker线程跑在远端节点的核上、
#   或者内存分配在远端节点时，主机侧的数据准备和H2D拷贝要跨插槽访存，会变慢
#   默认关闭（ATOM_NUMA_BIND=0）；只影响性能，不影响正确性
# 决定节点的顺序：ATOM_NUMA_NODE显式列表 → amdsmi查询 → sysfs（/sys/class/drm）推断
#
# 例：8卡机器，GPU 0-3挂在节点0、4-7挂在节点1；节点0的cpulist为"0-63,128-191"
#   → rank 5的worker被绑到节点1的那些核上，内存优先从节点1分配


# 解析sysfs的cpulist格式，例如"0-2,8" → {0, 1, 2, 8}
def _parse_cpulist(s: str) -> set[int]:
    """Parse a sysfs cpulist string (e.g. ``"0-63,128-191"``) into a set."""
    # 【译】把sysfs的cpulist字符串（例如 "0-63,128-191"）解析成一个集合。
    out: set[int] = set()
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# 读出某个NUMA节点上的CPU编号集合
def _node_cpus(node: int) -> set[int]:
    """CPUs belonging to NUMA ``node`` (kernel-reported, no topology guessing)."""
    # 【译】属于NUMA节点 node 的CPU（内核报告的，不做拓扑猜测）。
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        return _parse_cpulist(f.read())


# 逻辑GPU编号 → 物理GPU编号：worker看到的0号卡，可能是物理上的3号卡（被*_VISIBLE_DEVICES过滤过）
def _physical_index(gpu_id: int) -> int:
    """Map a logical GPU id to the physical index, honoring *_VISIBLE_DEVICES.

    System tools (amdsmi) and sysfs enumerate every physical GPU regardless of
    the visible-device mask, so a logical worker id must be translated back to
    the physical device it actually drives.
    """
    # 【译】把逻辑GPU id映射到物理编号，遵循 *_VISIBLE_DEVICES。
    #
    #   系统工具（amdsmi）和sysfs会枚举每一块物理GPU，不管可见设备掩码如何，所以必须把逻辑worker id
    #   翻译回它实际驱动的那块物理设备。
    # ROCR first: it filters the device table before HIP sees it, so when both
    # are set HIP's indices are already relative to the filtered list. A recipe
    # that sets ROCR alone (the only correct way to split across NUMA nodes --
    # setting both cuts the count twice) would otherwise fall through to the
    # identity mapping and bind every rank to node 0.
    # 【译】先看ROCR：它在HIP看到设备表之前就做了过滤，所以两者都设置时，HIP的编号已经是相对过滤后列表的。
    #   一个只设置ROCR的配方（这是跨NUMA节点切分的唯一正确方式——两个都设会把数量切两次）否则就会落到恒等映射，
    #   把每个rank都绑到节点0。
    # 例：ROCR_VISIBLE_DEVICES="4,5,6,7"时，逻辑0号卡 → 物理4号卡
    visible = (
        os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if not visible:
        return gpu_id
    vis = [int(x) for x in visible.split(",") if x.strip() != ""]
    return vis[gpu_id] if gpu_id < len(vis) else gpu_id


# 通过amdsmi查物理GPU所在的NUMA节点；amdsmi不可用或查询失败时返回None
def _query_node_amdsmi(gpu_id: int) -> int | None:
    """Physical GPU -> NUMA node via amdsmi (ROCm analog of NVML affinity)."""
    # 【译】物理GPU -> NUMA节点，通过amdsmi（ROCm版的NVML亲和性查询）。
    try:
        import amdsmi
    except Exception:
        return None
    phys = _physical_index(gpu_id)
    try:
        amdsmi.amdsmi_init()
        try:
            handles = amdsmi.amdsmi_get_processor_handles()
            if phys >= len(handles):
                return None
            node = int(amdsmi.amdsmi_topo_get_numa_node_number(handles[phys]))
            return node if node >= 0 else None
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug(f"amdsmi NUMA query failed for gpu {gpu_id}: {e}")
        return None


# 后备方案：按PCI地址排序/sys/class/drm下的显卡，第phys块卡的numa_node就是答案
def _query_node_sysfs(gpu_id: int) -> int | None:
    """Fallback: physical GPU -> node via DRM cards sorted by PCI BDF.

    Assumes ROCm enumerates GPUs in PCI-BDF order, which a non-identity
    *_VISIBLE_DEVICES permutation can break -- so this is only used when amdsmi
    is unavailable. Prefer the explicit ``ATOM_NUMA_NODE`` list on such setups.
    """
    # 【译】后备方案：物理GPU -> 节点，通过按PCI BDF排序的DRM card。
    #
    #   假设ROCm按PCI BDF顺序枚举GPU，而一个非恒等的 *_VISIBLE_DEVICES 排列可能打破这个假设——所以只在amdsmi
    #   不可用时才使用。在这类机器上更推荐显式的 ATOM_NUMA_NODE 列表。
    phys = _physical_index(gpu_id)
    cards: list[tuple[str, int]] = []
    for dev in glob.glob("/sys/class/drm/card*/device"):
        nn = os.path.join(dev, "numa_node")
        if not os.path.exists(nn):
            continue
        bdf = os.path.basename(os.path.realpath(dev))
        with open(nn) as f:
            cards.append((bdf, int(f.read())))
    cards.sort()
    if phys >= len(cards):
        return None
    node = cards[phys][1]
    return node if node >= 0 else None


# 决定gpu_id该绑到哪个节点；返回None表示无法确定
def _resolve_node(gpu_id: int) -> int | None:
    """Explicit ``ATOM_NUMA_NODE`` wins; else auto-detect (amdsmi -> sysfs)."""
    # 【译】显式的 ATOM_NUMA_NODE 优先；否则自动检测（amdsmi -> sysfs）。
    explicit = [x for x in envs.ATOM_NUMA_NODE.split(",") if x.strip() != ""]
    if explicit:
        # 列表比gpu_id短时用最后一个值（所以只给一个值就是"所有rank都用这个节点"）
        idx = gpu_id if gpu_id < len(explicit) else len(explicit) - 1
        return int(explicit[idx])
    if not envs.ATOM_AUTO_NUMA_BIND:
        return None
    node = _query_node_amdsmi(gpu_id)
    if node is None:
        node = _query_node_sysfs(gpu_id)
    return node


# 让本进程优先从指定节点分配内存（libnuma的numa_set_preferred），失败只记debug日志
def _set_preferred_memory(node: int) -> None:
    """Best-effort memory binding via libnuma ``numa_set_preferred``.

    If libnuma is absent, first-touch on the pinned CPUs still lands memory on
    the local node, so this is an optimization, not a requirement.
    """
    # 【译】通过libnuma的 numa_set_preferred 做尽力而为的内存绑定。
    #
    #   如果没有libnuma，在已固定的CPU上的首次访问（first-touch）仍会把内存放在本地节点，所以这是优化，不是必需。
    # ctypes.CDLL：直接加载C共享库并调用其中的函数
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            return
        libnuma.numa_set_preferred(ctypes.c_int(node))
    except Exception as e:
        logger.debug(f"numa_set_preferred({node}) skipped: {e}")


# 公开入口：把当前进程绑到gpu_id所在NUMA节点的CPU和内存上
# 输入：gpu_id（本机内的逻辑GPU编号，由调用方算出）、label（日志标签）
# 调用方：AsyncIOProc.__init__（async_proc.py）
def numa_bind_to_node(gpu_id: int, label: str = "") -> None:
    """Bind the current process to its GPU's NUMA-local cores and memory.

    No-op unless ``ATOM_NUMA_BIND`` is enabled. Must run before any large
    allocation / native (e.g. mooncake RDMA) thread spawn so the affinity mask
    is inherited by child threads and Linux first-touch places memory on the
    node. The node's CPUs are intersected with the current affinity so an
    existing container cpuset is respected. On failure it warns (and raises only
    if ``ATOM_CRASH_ON_NUMA_BIND_FAILURE``).
    """
    # 【译】把当前进程绑定到它的GPU所在NUMA本地的核与内存上。
    #
    #   除非开启 ATOM_NUMA_BIND，否则什么也不做。必须在任何大块分配 / 原生（例如mooncake RDMA）线程创建之前运行，
    #   这样亲和性掩码会被子线程继承，Linux的首次访问也会把内存放在该节点上。节点的CPU会与当前亲和性取交集，
    #   以尊重容器已有的cpuset。失败时给出警告（只有在 ATOM_CRASH_ON_NUMA_BIND_FAILURE 时才抛异常）。
    if not envs.ATOM_NUMA_BIND:
        return
    tag = f" ({label})" if label else ""
    try:
        node = _resolve_node(gpu_id)
        if node is None or node < 0:
            raise RuntimeError(f"could not resolve NUMA node for gpu {gpu_id}")
        # 节点上的CPU ∩ 当前允许的CPU（容器的cpuset限制）；sched_getaffinity(0)中的0表示当前进程
        cpus = _node_cpus(node) & os.sched_getaffinity(0)
        if not cpus:
            raise RuntimeError(
                f"NUMA node {node} has no CPUs allowed by the current affinity"
            )
        # 设置CPU亲和性：之后本进程（以及之后创建的线程）只在这些核上运行
        os.sched_setaffinity(0, cpus)
        _set_preferred_memory(node)
        logger.info(f"NUMA bind{tag}: gpu={gpu_id} -> node {node} ({len(cpus)} cores)")
    except Exception as e:
        msg = (
            f"NUMA bind{tag} failed for gpu {gpu_id}: {e}. In docker add "
            f"--cap-add SYS_NICE, or set ATOM_NUMA_NODE explicitly."
        )
        if envs.ATOM_CRASH_ON_NUMA_BIND_FAILURE:
            raise RuntimeError(msg) from e
        logger.warning(msg)
