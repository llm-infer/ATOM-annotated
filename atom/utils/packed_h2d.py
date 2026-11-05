# SPDX-License-Identifier: MIT
"""Pack metadata into one pinned arena, upload once, scatter to stable addresses."""

# 【译】把元数据打包进一块 pinned arena，上传一次，再 scatter 到各自固定的地址。
#
# 【导读】packed_h2d.py：ATOM_H2D_BACKEND=packed（默认）时，一个发布组的多个小张量怎样只用一次 H2D 送上 GPU
#   调用方：h2d.py PublicationGroup.use_transport("packed") → PackedCopy.create（启动时，每个组合组一次）；
#     每步 PublicationGroup.publish → PackedCopy.submit
#   前置：pinned 内存、异步拷贝、为什么 DMA 执行前不能改写 pinned 源 → h2d.py 文件头的前置①②和"排了拷贝 ≠ 已经拷走"
#
# 为什么要打包：每次 H2D 都有一份与字节数无关的固定开销——主机端一次 hipMemcpyAsync 调用（驱动入队），GPU 端 DMA 引擎
#   再启动一次传输，量级是微秒。而每步的元数据大多只有几百字节：1 KB 在 PCIe Gen5 x16（PCIe 5.0 规范理论约 64 GB/s）上
#   传输本身只要约 16 ns。于是时间几乎全花在"次数"上。设每次固定开销为 t（与驱动、队列深度有关，本仓库没有给出实测值）：
#     direct：h2d.py 文件头那一步 decode 有 8~9 个非空成员 → 8~9 次拷贝 ≈ 8t~9t
#     packed：按"第一个 GPU 使用方"分成 2 组（token_inputs、mha_decode），每组 1 次 H2D + 1 个 scatter kernel ≈ 4t，
#       另加 CPU 上把各段字节拷进 arena 的几次小 memcpy（几百纳秒级）
#   V4.1 的普通 DSpark decode 整步也只有 2 次 H2D + 2 个 scatter（docs/h2d_publication.md）。该文档也提醒：少几次拷贝
#   不等于在所有场景都降低 TTFT，要用 profiler 从准备阶段的 GPU 工作量到第一个模型 kernel 实测。
#
# 为什么还要 scatter、不能让 kernel 直接读 arena：CUDA Graph 录下的是每个元数据缓冲的固定 GPU 地址（forward_vars 里的
#   CpuGpuBuffer.gpu），各算子也都按那些地址读；所以先把整块 arena 一次拷到 GPU 上的 device arena，再用一个 kernel
#   把每段字节写回它原本的 GPU 缓冲。
#
# arena 内存布局（例：h2d.py 文件头那一步的 token_inputs 组，6 个成员，bs=64、top_k / top_p 不上传）
#   host arena = pinned uint8，容量 = 16 × 成员数 + Σ(各成员容量 × 每单位字节数)；
#     默认配置下 = 96 + 3×2048（采样参数 [512] 各 4 B）+ 2052（cu_seqlens_q [513]）+ 65540（input_ids [16385]）
#     + 65536（decode_src [16384]）= 139368 B ≈ 136 KiB；GPU 上还有一块同样大的 device arena
#   字节偏移  0         16        32        48        64        80        96
#   header    │(96, 256)│(352, 0) │(352, 0) │(352, 260)│(612, 256)│(868, 256)│   每个成员 16 B = int64 (offset, length)
#   对应成员    temperatures top_ks  top_ps   cu_seqlens_q input_ids  decode_src
#   字节偏移  96                   352                  612               868              1124
#   payload   │ temperatures 256 B │ cu_seqlens_q 260 B │ input_ids 256 B │ decode_src 256 B │   只放非空成员，依次紧排
#   本步：一次 copy_(host[:1124] → device[:1124])，然后 scatter_packed_bytes 的 grid = (6 个成员, ⌈260/1024⌉ = 1)：
#     program (m, b) 读 header 第 m 项 → 把 device[offset + b×1024 + 0..1023] 中 < length 的字节写到 destinations[m] 处；
#     length=0 的成员（top_ks、top_ps）全部被 mask 掉，什么都不写
#   每步变化的 (offset, length) 跟数据走同一次 DMA，所以 kernel 的参数（两块 GPU 缓冲的地址）每步都一样
#
# 对比nano-vllm：nano-vllm 每个元数据张量单独 torch.tensor(..., pin_memory=True).cuda(non_blocking=True)，没有打包

import torch
import triton
import triton.language as tl


# scatter_packed_bytes：把 device arena 里每个成员的那段字节写到它自己的 GPU 缓冲
#   arena         uint8 [容量]：开头是 int64 header（每成员 (offset, length)），后面是 payload
#   destinations  int64 [成员数]：每个成员 GPU 缓冲的 data_ptr()（启动时建好，之后不变）
#   BLOCK         每个 program 搬的字节数（constexpr，固定 1024）
#   grid = (成员数, ⌈最长成员字节数 / BLOCK⌉)：program_id(0) 选成员，program_id(1) 选这个成员里的第几个 1 KB 块
# Triton 要点：tl.program_id(axis) 是本 program 在 grid 中的坐标；tl.arange(0, BLOCK) 是 [0, BLOCK) 的向量；
#   tl.load / tl.store 的第 2 个参数是掩码，越界（offset >= length）的位置不读不写；
#   .to(tl.pointer_type(...)) 把整数地址 / 字节指针重新解释成指定元素类型的指针
#   按字节搬运：不管成员是 int32、int64、float 还是 bool，都原样复制位模式
@triton.jit
def scatter_packed_bytes(arena, destinations, BLOCK: tl.constexpr):
    member = tl.program_id(0)
    # 把 arena 的开头当 int64 数组读：header[2m] = 本成员 payload 在 arena 里的字节偏移，header[2m+1] = 字节数
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 2 * member)
    length = tl.load(header + 2 * member + 1)
    # 目的地址是存在 int64 表里的整数，转成 uint8 指针
    destination = tl.load(destinations + member).to(tl.pointer_type(tl.uint8))
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arena + start + offset, offset < length, other=0)
    tl.store(destination + offset, values, offset < length)


class PackedCopy:
    """Persistent byte storage; the publication owner fences reuse of the arena."""

    # 【译】常驻的字节存储；arena 何时可以复用由发布的 owner 负责把关。
    #   （PackedCopy 自己不建 event：arena 的 host 端和组里其他 pinned 源一样，由 owner 的 completion event 保护——
    #    submit 返回 True 表示 arena 已借给在途 DMA，组会记下 _header_busy，直到下次 begin 才允许再改写）

    # create：启动时为一个组建 PackedCopy；不合格时返回 (None, 原因)，组退回 direct
    #   不合格：不是 GPU、成员少于 2 个（单成员本来就只要一次 DMA）、有成员不连续（带步长的区域只能逐个 direct 拷）
    # Python 语法：@classmethod 的第一个参数是类本身，cls() 新建实例；这里用它当"带校验的工厂函数"
    @classmethod
    def create(cls, members, device):
        if device.type != "cuda" or len(members) < 2:
            return None, "packing requires multiple GPU destinations"
        if any(
            not b.source.is_contiguous() or not b.destination.is_contiguous()
            for b in members
        ):
            return None, "strided regions use checked direct"
        result = cls()
        result.members = members
        result.header_bytes = 16 * len(members)  # int64 (offset, byte count)
        # 上一行行尾英文：每个成员一对 int64 (偏移, 字节数)
        # 容量按每个成员的最大容量算，所以任何一步的 counts 都放得下，arena 只分配一次
        capacity = result.header_bytes + sum(
            b.capacity * b.bytes_per_count for b in members
        )
        result.host = torch.empty(
            capacity, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        result.device = torch.empty(capacity, dtype=torch.uint8, device=device)
        result.header = result.host[: result.header_bytes].view(torch.int64).numpy()
        result.payload = memoryview(result.host.numpy())
        # Byte views avoid typed copies canonicalizing bool or casting BF16.
        # 【译】用字节视图，避免按类型拷贝时把 bool 规范化、或者对 BF16 做类型转换。
        #   （numpy 没有 bf16；bool 若按类型拷，非 0/1 的字节会被改成 0/1。按 uint8 拷就是原样的位模式）
        # memoryview：零拷贝地把 pinned 内存当字节数组切片赋值，payload[a:b] = sources[i][:n] 就是一次 CPU memcpy
        result.sources = tuple(
            memoryview(b.source.reshape(-1).view(torch.uint8).numpy()) for b in members
        )
        # 每个成员 GPU 缓冲的地址表，放在 GPU 上给 scatter kernel 读；这些缓冲的地址启动后不再变（CUDA Graph 也依赖这一点）
        result.destinations = torch.tensor(
            [b.destination.data_ptr() for b in members],
            dtype=torch.int64,
            device=device,
        )
        # prefixes：缓存 (host[:n], device[:n]) 视图，最多 4 个；kernel：第一次 launch 后缓存的已编译 kernel
        result.prefixes = {}
        result.launch = scatter_packed_bytes
        result.kernel = None
        return result, None

    def submit(self, counts):
        """Enqueue copies and return whether the packed arena is borrowed."""
        # 【译】把拷贝排进队列，并返回打包的 arena 是否被借用（是否有在途 DMA 在读它）。
        # 步骤：① 只有 1 个非空成员 → 直接 direct 拷它（不碰 arena，返回 False）；② 在 CPU 上填 header、把各成员字节
        #   依次拷进 payload；③ 一次 non_blocking H2D；④ launch scatter kernel。②③④ 之后返回 True
        active = [i for i, count in enumerate(counts) if count]
        if len(active) == 1:
            # A single live member already takes one DMA; no packing/scatter.
            # 【译】只有一个有内容的成员时，它本来就只需要一次 DMA；不打包、也不 scatter。
            i = active[0]
            self.members[i]._copy(counts[i])
            return False
        # ② payload 从 header 之后开始紧排；largest 记最长的一段，决定 grid 第二维
        offset = self.header_bytes
        largest = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            length = 0 if count is None else count * binding.bytes_per_count
            self.header[2 * i] = offset
            self.header[2 * i + 1] = length
            if length:
                self.payload[offset : offset + length] = self.sources[i][:length]
                offset += length
                largest = max(largest, length)
        # ③ offset 此刻 = header + 全部 payload 的总字节数，只拷这么多（上例 1124 B，而不是整个 136 KiB 的 arena）
        prefix = self.prefixes.get(offset)
        if prefix is None:
            prefix = self.host[:offset], self.device[:offset]
            if len(self.prefixes) == 4:
                del self.prefixes[next(iter(self.prefixes))]
            self.prefixes[offset] = prefix
        # Header and payload share this single asynchronous H2D.
        # 【译】header 和 payload 共用这一次异步 H2D。
        prefix[1].copy_(prefix[0], non_blocking=True)
        # ④ scatter：排在上面那次 H2D 之后（同一条 stream），所以读到的一定是刚拷上来的 arena
        grid = (len(self.members), (largest + 1023) // 1024, 1)
        if self.kernel is None:
            self.kernel = self.launch[grid](self.device, self.destinations, BLOCK=1024)
        else:
            # Both argument storages and their layouts are fixed for this
            # binding. Reuse its compiled kernel; counts and grid stay dynamic.
            # 【译】对这个绑定来说，两个参数的存储和布局都是固定的。复用它已编译的 kernel；计数和 grid 仍是动态的。
            # 第一次 self.launch[grid](...) 走 Triton JIT（绑定参数、算特化键、查编译缓存，必要时编译），返回编译好的 kernel；
            #   之后直接 self.kernel[grid](...) 调它，跳过这些 Python 端开销（BLOCK 已编进 kernel，不再传）
            self.kernel[grid](self.device, self.destinations)
        return True
