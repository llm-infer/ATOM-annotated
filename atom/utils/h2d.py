# SPDX-License-Identifier: MIT
"""Checked publication of persistent host buffers on an owner's stream.

The owner supplies its existing reuse event; this module owns no ring. CPU
writes must follow begin() or acquire_write(), before publish(). Tensor aliases
can still write around this protocol: producers must migrate with their buffers.
"""

# 【译】在所有者（owner）的 stream 上，对常驻 host 缓冲区做带检查的发布。
#
#   所有者提供它已有的"复用事件"（reuse event）；本模块自己不持有环形缓冲。CPU 的写入必须发生在 begin() 或 acquire_write()
#   之后、publish() 之前。张量别名仍然可以绕过这个协议去写：生产者必须带着它们的缓冲区一起迁移到这个协议上来。
#
# 【导读】h2d.py：每步 forward 的元数据（positions、slot_mapping、cu_seqlens、采样参数……）从 CPU 的 pinned 缓冲
#   发布到 GPU 上固定地址的"带检查的发布协议"
#   位置：启动时 ModelRunner._init_h2d_publication：每个 forward_vars slot 建一个 PublicationOwner，绑定所有带
#     publication_group 的 CpuGpuBuffer，按组名建 PublicationGroup（packed 模式下再拼出 token_inputs / prefill_inputs
#     等组合组并选打包传输）；每步 forward：_gate_staging_reuse = owner.begin() → 各生产者写 .np 并 group.publish(counts)
#     → _mark_staging_h2d_enqueued = owner.finish()（都在 model_runner.py）
#   打包传输的实现在 packed_h2d.py；开关 ATOM_H2D_BACKEND（默认 packed，或 direct，envs.py）；设计说明 docs/h2d_publication.md
#
# 前置①：GPU 异步执行与 stream（详见 engine_core.py 文件头【3】的前置）
#   CPU 调一个 GPU 操作只是把命令排进 stream（GPU 上的 FIFO 命令队列）就返回，GPU 按顺序执行。
#   gpu.copy_(cpu, non_blocking=True) 也一样：只是在 stream 里排了一条"DMA 拷贝"命令。
# 前置②：锁页（pinned）内存（详见 utils/__init__.py 中 _PAGEABLE_H2D_LIMIT 上方的"科普·锁页内存与stream"）
#   可分页内存不能直接被 DMA 读，驱动要先把字节拷进自己的暂存区（超过约 512 KB 时 CPU 要一直等到这次拷贝真正执行完）；
#   pinned 内存常驻物理内存，DMA 直接读它，所以 non_blocking 拷贝能真正"排完就走"。ATOM 每步的元数据都放在
#   预先分配的 pinned 缓冲里（CpuGpuBuffer.cpu，和 numpy 视图 .np 共享内存），GPU 端是同形状的固定缓冲（CpuGpuBuffer.gpu）。
#
# 由此带来的新问题：排了拷贝 ≠ 已经拷走
#   DMA 真正读 pinned 源，是在 GPU 执行到那条拷贝命令时——它排在前面所有 kernel 之后，可能是几毫秒以后。
#   在那之前 CPU 若改写了同一块 pinned 内存（比如已经开始准备下一步），GPU 拷走的就是新值，或者半新半旧。
#   （可分页内存没有这个问题：驱动在 copy_ 返回前已经把字节复制走了，代价就是上面说的同步等待）
#   GPU 目的端不用担心：下一步的拷贝和本步的 kernel 在同一条 stream 上，必然等本步 kernel 读完之后才执行。
#
#   时间线（非 PP、单个 slot；E = owner 的 completion event）：
#     CPU: [准备 N：写 pinned → 排 H2D N → record E][发射 fwd N] … [postprocess] … |begin：E.synchronize()|[准备 N+1：写 pinned …]
#     GPU: …[ fwd N-1 的 kernel ][H2D N][============ fwd N 的 kernel ============][H2D N+1][== fwd N+1 ==
#                                 ↑ DMA 在这一刻才读 pinned 源；E 紧跟在它后面完成
#   begin() 等的是 E：只要 H2D N 执行完，pinned 源就可以改写——不必等 fwd N 跑完，因为 fwd N 读的是 GPU 上的副本。
#   所以 GPU 还在跑 fwd N 时，CPU 就能准备第 N+1 步的元数据（一个 slot 允许 CPU 领先一步）。
#   稳态 decode 下这个等待几乎为 0：延迟输出让第 N 步的 postprocess 先等第 N-1 步的 token 拷回，那时 fwd N-1 已结束，
#   紧随其后的 H2D N 很快就执行完了。真正会被它卡住的是中间没有同步点的连续 forward：chunked prefill 的中间块
#   （forward 在 postprocess 之前就返回）和 DP dummy。没有这道闸时测到过 CPU 领先 4042 个包、GPU 读到了后面某个 batch 的
#   batch_id_per_q_token（见 model_runner.py _gate_staging_reuse 的 docstring）。
#
# 为什么还需要"环"（ring）：流水线并行（PP，第9批）时 head stage 连续发射最多 pp_size 个微批，中间不做 GPU 同步。
#   只有一个 slot 的话，第 k+1 个微批要等第 k 个微批的 H2D 执行完，而那条 H2D 排在第 k-1 个微批的整个 forward 后面，
#   CPU 的领先量就被压回 1 步。于是 ModelRunner._init_forward_vars_ring 给每个可能在途的微批一套独立的 forward_vars
#   （CpuGpuBuffer.clone：pinned 端和 GPU 端都复制），每套一个 event、一个 PublicationOwner，轮流使用，CPU 最多领先 pp_size 步。
#   本模块"不持有环"：环和 event 都属于 ModelRunner，这里只管单个 slot 内的规则。
#     例：pp_size=4 → 4 套缓冲。按默认 max_num_batched_tokens=16384、max_num_seqs=512 估算每套里的几个大头：
#       input_ids int32 [16385] ≈ 64 KiB、positions int64 [16384] = 128 KiB、slot_mapping int64 [16384] = 128 KiB、
#       block_tables int32 [512, max_model_len/block_size]（max_model_len=32768、block_size=16 时 512×2048×4 B = 4 MiB），
#       其余逐序列的小数组各 2 KiB 左右。每套都是"pinned 一份 + GPU 一份"，4 套就是 4 倍。
#       PP 必须 --enforce-eager：CUDA Graph 录下的是一套固定地址，没法跟着环切换。
#
# 本模块在此之上加的是"检查"：把上面的时序规则变成运行时断言，违规立刻抛 PublicationError，而不是悄悄读到错数据
#   对象关系：
#     PublicationRegistry（整个 runner 一个）  登记时检查任意两个绑定的 pinned 源区间、GPU 目的区间都不重叠
#     PublicationOwner（每个 slot 一个）       completion event、epoch 计数、状态机、所在线程 / 设备 / stream
#       ├ BufferPublication（每个绑定的 CpuGpuBuffer 一个）  源 / 目的、计数单位、"本 epoch 已发布"、"源可写"
#       └ PublicationGroup（每个 publication_group 名一个，外加 token_inputs / prefill_inputs / mha_decode 等组合组）
#            一次 publish 先校验全组，再统一提交：direct（每个成员一次 copy_）或 packed（PackedCopy：一次 DMA + 一个 scatter kernel）
#   owner 状态机（一次 forward = 一个 epoch）：
#     idle ──begin()──> active ──finish()──> sealed ──begin()──> active …
#                        ▲  │                  │
#                        │  └─任何提交失败─> failed ──drain()（排空后只能重建 owner）
#                        └─────resume()───────┘（同一 forward 的后续准备阶段，如 V4 TBO）
#     begin   等 completion event（上一个 epoch 排的 DMA 都已读完 pinned 源）→ epoch+1 → 所有源标记"可写"
#     publish 校验全组 → 标记"本 epoch 已发布"，count>0 的源标记"已借给 DMA、不可写" → 排拷贝
#     finish  在 stream 上 record completion event（不等 GPU）
#   一个 epoch 内的规则：每个源最多发布一次（再发布要给 republish_reason，并先 acquire_write 等 DMA 读完）；
#     只能在 begin 的那个线程、那个设备、那条 stream 上发布；不能在 CUDA Graph 录制期间发布。
#
# 一步 decode 带数字走一遍（非 PP、ATOM_H2D_BACKEND=packed、MHA 模型、不开投机、bs=64；64 在阶梯上 → running_bs=64）
#   forward(batch)                  一步
#    ├ _gate_staging_reuse → owner.begin()：E 是上一步 finish 时 record 的，稳态下早已完成，立即返回；epoch 41 → 42
#    ├ prepare_sample：写 temperatures.np[:64]（top_k / top_p 若全 batch 相同就作为 CPU 标量传，不上传，count=None）
#    ├ prepare_cu_seqlens_q：写 cu_seqlens_q.np[:65]
#    ├ prepare_input_ids：写 input_ids.np[:64]、decode_src.np[:64] → token_inputs.publish(counts)
#    │    packed：temperatures 256 B + cu_seqlens_q 260 B + input_ids 256 B + decode_src 256 B 拼进 arena → 1 次 H2D + 1 个 scatter
#    │    然后 fill_deferred_decode_ids kernel（读 GPU 上的 cu_seqlens_q / decode_src，见 model_ops/decode_input_ids.py）
#    ├ attn_metadata_builder.build → prepare_decode → mha_decode.publish：slot_mapping 512 B、context_lens 256 B、
#    │    block_tables（页表没变时不再上传）、kv_indptr 260 B、positions 512 B → 又是 1 次 H2D + 1 个 scatter
#    ├ _mark_staging_h2d_enqueued → owner.finish()：record E（排在这两次 H2D 之后）
#    └ run_model …（下一步的 begin() 等的就是这个 E）
#   direct 模式下同样的数据是 8~9 次 copy_（每个非空成员一次；block_tables 变化时 9 次）
#
# 对比nano-vllm：nano-vllm 每步用 torch.tensor(list, pin_memory=True).cuda(non_blocking=True) 现建现拷——每个张量
#   每步新分配一块 pinned 内存，PyTorch 的 pinned 缓存分配器要等这次 DMA 完成才会复用它，所以不存在"源被改写"的问题；
#   代价是每步分配、每个张量一次拷贝，而且 GPU 端地址每步都变（nano-vllm 的 CUDA Graph 因此要在回放前再拷进固定的
#   graph_vars）。ATOM 用常驻缓冲省掉了分配和二次拷贝，换来的就是本文件要管的"什么时候可以改写 pinned 源"。

from __future__ import annotations

import operator
import threading
from functools import wraps

import torch


class PublicationError(RuntimeError):
    """A publication violates ownership or asynchronous source lifetime."""

    # 【译】一次发布违反了所有权规则，或违反了异步拷贝源的生命周期。


# h2d_producer：装饰"写 pinned 源"的生产者方法，在方法体运行前检查它要写的那些组此刻可写
#   用法：@h2d_producer("sampling")（ModelRunner.prepare_sample）、@h2d_producer("input_ids", runner="runner")
#     （tokenIDProcessor.prepare_input_ids，runner 属性指向 ModelRunner）、@h2d_producer("prefill", "positions",
#     runner="model_runner")（backends.py prepare_prefill）等
#   为什么要在写之前查：一旦写坏了 pinned 源，就算后面发布时报错，正在飞的 DMA 也可能已经拷走了被改写的字节
#   每次调用都重新取 runner.h2d_groups：PP 下 _advance_forward_vars 每步会把它换成当前 slot 的组
def h2d_producer(*group_names, runner=None):
    """Declare source groups a method writes, checking before its body runs.

    ``runner`` names the attribute holding the runner, or None for its own
    methods. Resolve the current group on every call so PP slot rotation is
    respected. Standalone producers without a publication registry retain
    their unregistered behavior; a registered runner must provide the group.
    This neither publishes data nor grants permission to republish it.
    """
    # 【译】声明一个方法会写哪些源组，并在方法体运行之前做检查。
    #
    #   ``runner`` 给出持有 runner 的属性名；为 None 时表示就是 runner 自己的方法。每次调用都解析当前的组，
    #   以便尊重 PP 的 slot 轮换。没有发布注册表的独立生产者保持原来未注册时的行为；已注册的 runner 必须提供这个组。
    #   它既不发布数据，也不授予再次发布的许可。
    # Python 语法：带参数的装饰器 = 返回装饰器的函数；operator.attrgetter("runner") 生成一个取 obj.runner 的函数；
    #   functools.wraps 保留被装饰函数的名字和 docstring
    get_runner = operator.attrgetter(runner) if runner else lambda instance: instance

    def decorate(produce):
        @wraps(produce)
        def checked(instance, *args, **kwargs):
            groups = getattr(get_runner(instance), "h2d_groups", None)
            if groups is not None:
                for name in group_names:
                    groups[name].check_writable()
            return produce(instance, *args, **kwargs)

        return checked

    return decorate


# 校验并规范化一个发布计数：必须是 [0, capacity] 内的整数（拒绝 bool——True 也是 int 的子类）；operator.index 接受
#   int / numpy 整数，拒绝 float
def _count(value, capacity, name):
    if isinstance(value, bool):
        raise TypeError(f"{name}: count must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name}: count must be an integer") from exc
    if not 0 <= value <= capacity:
        raise ValueError(f"{name}: count {value} is outside [0, {capacity}]")
    return value


def _range(tensor):
    # Reserve the whole span for strided views; overlapping holes cannot be
    # registered as independent buffers. Direct still supports their layout.
    # 【译】对带步长的视图，保留它覆盖的整段区间；中间有空洞而互相重叠的视图不能登记成相互独立的缓冲区。
    #   direct 传输仍然支持这种布局。
    # 返回 [起始地址, 结束地址)：跨度 = 1 + Σ (每维长度-1)×步长 个元素。例：[4, 3] 的连续 int32 张量 → 1+3×3+2×1 = 12 个元素 = 48 B
    start = tensor.data_ptr()
    if not tensor.numel():
        return start, start
    span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return start, start + span * tensor.element_size()


# 两个半开区间 [a0, a1)、[b0, b1) 是否相交
def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


# republish_reason 必须是非空字符串：再次发布同一个源是例外情况，要求写明理由
def _reason(reason):
    return isinstance(reason, str) and bool(reason.strip())


class PublicationRegistry:
    """Registration-time alias checks shared by a runner's owners and slots."""

    # 【译】登记时的别名检查，由一个 runner 的所有 owner 和 slot 共用。
    # 为什么要查重叠：每个绑定各自记录"已发布 / 可写"，若两个绑定其实指向同一块内存，一个以为可写、另一个的 DMA 还在读，
    #   检查就失效了。PP 的各 slot 共用这一个 registry，所以跨 slot 的别名也能查出来

    def __init__(self):
        self._bindings = []

    def add(self, binding):
        for prior in self._bindings:
            if prior.device == binding.device and _overlaps(
                prior.destination_range, binding.destination_range
            ):
                raise ValueError(
                    f"{binding.name}: destination overlaps registered {prior.name}"
                )
            if _overlaps(prior.source_range, binding.source_range):
                raise ValueError(
                    f"{binding.name}: source overlaps registered {prior.name}"
                )
        self._bindings.append(binding)


class PublicationOwner:
    """One existing owner/slot, with an epoch shared by all its publish groups.

    finish() seals host preparation, retaining the ledger until the next begin.
    A later host phase must resume the SAME epoch, preserving duplicate checks.
    begin() starts a forward, initialization or capture-preparation epoch;
    an auxiliary group in the same forward must not reset it.
    """

    # 【译】一个已有的 owner / slot，它的所有发布组共用一个 epoch。
    #
    #   finish() 封存 host 端的准备工作，账本保留到下一次 begin。之后的 host 阶段必须 resume 同一个 epoch，
    #   这样重复发布的检查仍然有效。begin() 开启一个 forward、初始化或录图准备的 epoch；同一个 forward 里的
    #   辅助组不能把它重置。
    # 构造参数：device（GPU 设备）、completion（slot 的 torch.cuda.Event：非 PP 为 ModelRunner._stage_h2d_done，
    #   PP 为 _fv_slot_events[i]；GPU 发布必须提供）、registry（共用的 PublicationRegistry）

    def __init__(self, device, completion=None, *, registry=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type == "cuda" and completion is None:
            raise ValueError("GPU publication requires the owner's reuse event")
        self.completion = completion
        self.registry = registry if registry is not None else PublicationRegistry()
        self.epoch = 0
        self._state = "idle"
        self._stream = None
        self._thread = None
        self._stream_id = None
        # Compare PyTorch stream IDs without constructing a Stream wrapper.
        # Native handles can alias distinct PyTorch stream identities.
        # 【译】比较 PyTorch 的 stream ID，而不构造 Stream 包装对象。底层原生句柄可能对应多个不同的 PyTorch stream 身份。
        #   （每次发布都要检查一遍 stream，构造 Python 对象每次要几微秒；_cuda_getCurrentStream 直接返回 (id, 设备号, 类型)）
        self._get_current_stream = getattr(torch._C, "_cuda_getCurrentStream", None)
        self._bindings = []
        self._groups = []

    # bind：启动时把一个 CpuGpuBuffer 登记到本 owner（只能在 idle 状态）。之后 buffer.copy_to_gpu 会改走这个绑定
    #   （CpuGpuBuffer.copy_to_gpu 看到 _publication 就转交），并自动建一个只含它的单成员组
    #   unit：计数单位——"rows"（第 0 维的行数，默认）、"elements"（元素数）、"bytes"（字节数）
    def bind(self, buffer, name, *, unit="rows"):
        if self._state != "idle":
            raise PublicationError("bind buffers before starting publication")
        if getattr(buffer, "_publication", None) is not None:
            raise ValueError(f"{name}: buffer is already bound")
        if any(b.name == name for b in self._bindings):
            raise ValueError(f"duplicate logical name: {name}")
        binding = BufferPublication(self, buffer, name, unit)
        self.registry.add(binding)
        self._bindings.append(binding)
        buffer._publication = binding
        binding._single = self.group(name, (binding,))
        return binding

    # group：把若干绑定组成一个发布组（只能在 idle 状态；成员必须属于本 owner，不能重复）
    def group(self, name, members):
        if self._state != "idle":
            raise PublicationError("create groups during initialization")
        members = tuple(members)
        if any(b.owner is not self for b in members):
            raise ValueError("a group must belong to one owner/device")
        if len({id(b) for b in members}) != len(members):
            raise ValueError("a group cannot publish a destination twice")
        group = PublicationGroup(self, name, members)
        self._groups.append(group)
        return group

    def use_packed_transport(self):
        """Configure maximal eligible groups once; covered groups stay direct.

        The runner defines consumer boundaries. This layer chooses storage,
        avoiding duplicate arenas for producer groups covered by a boundary.
        Standalone calls to those groups still use checked direct copies.
        """
        # 【译】一次性为所有"极大的"合格组配置打包传输；被覆盖的组保持 direct。
        #
        #   runner 定义使用方的边界（哪些成员要在同一个 GPU 使用方之前一起就绪）。这一层选择存储，对已被某个边界覆盖的
        #   生产者组不再重复建 arena。对这些组的单独调用仍然走带检查的直接拷贝。
        # 规则：按成员数从多到少遍历，成员 ≥2 且不是某个已打包组的子集才打包。例（packed 模式）：token_inputs =
        #   sampling + early + input_ids 先被打包，sampling / early / input_ids 这三个小组就保持 direct
        #   （只在没有组合组的调用路径上单独用）；返回尝试过打包的组（含因不合格退回 direct 的），供启动日志打印
        if self._state != "idle":
            raise PublicationError("choose the transport during initialization")
        packed_members = []
        configured = []
        for group in sorted(self._groups, key=lambda g: -len(g.members)):
            members = set(group.members)
            if len(members) < 2 or any(members <= packed for packed in packed_members):
                continue
            group.use_transport("packed")
            if group.transport == "packed":
                packed_members.append(members)
            configured.append(group)
        return configured

    # _current_stream：取本 owner 设备上的当前 stream，并拒绝两种错误用法——在别的设备上，或正在录 CUDA Graph。
    #   为什么不能在录图时发布：这条 H2D 会被录进图里，每次回放都从同一块 pinned 内存再拷一遍，而回放不跑 Python，
    #   begin / finish 的检查全部失效；元数据必须在录制 / 回放之前发布好
    def _current_stream(self):
        if self.device.type != "cuda":
            return None
        if torch.cuda.current_device() != self.device.index:
            raise PublicationError("publication on the wrong device")
        if torch.cuda.is_current_stream_capturing():
            raise PublicationError("publish metadata before actual graph capture")
        return torch.cuda.current_stream(self.device)

    # _check_active：每次发布 / 写入前的检查——处于 active、同一线程、同一设备、不在录图、同一条 stream
    #   为什么要同一条 stream：completion event 是在 begin 时那条 stream 上 record 的，只覆盖排在那条 stream 上的拷贝；
    #   在别的 stream 上发的拷贝不受这个 event 保护，下一步 begin 时就可能提前改写它的源
    def _check_active(self):
        if self._state != "active":
            raise PublicationError(f"owner is {self._state}; acquire sources first")
        if threading.get_ident() != self._thread:
            raise PublicationError("publish on the owner thread")
        if self.device.type == "cuda":
            if torch.cuda.current_device() != self.device.index:
                raise PublicationError("publication on the wrong device")
            if torch.cuda.is_current_stream_capturing():
                raise PublicationError("publish metadata before actual graph capture")
            if self._get_current_stream is not None:
                same_stream = (
                    self._get_current_stream(self.device.index)[0] == self._stream_id
                )
            else:
                same_stream = torch.cuda.current_stream(self.device) == self._stream
            if not same_stream:
                raise PublicationError("publish on the owner's current compute stream")

    def begin(self):
        """Wait for source reuse BEFORE producer writes; start a new epoch."""
        # 【译】在生产者写之前等待源可以复用；开始一个新 epoch。
        # 步骤：completion.synchronize()（CPU 阻塞，直到上一个 epoch 排的 DMA 都读完 pinned 源；从没 record 过的 event 立即返回，
        #   所以第一次 forward 不会被卡）→ epoch+1 → 记下当前 stream / 线程 → 所有绑定标记可写、所有组的 header 标记空闲 → active
        if self._state not in ("idle", "sealed"):
            raise PublicationError(f"cannot begin: owner is {self._state}")
        stream = self._current_stream()
        if self.completion is not None:
            # The caller records constructor uploads/slot clones before first use.
            # 【译】调用方在第一次使用之前，已经为构造时的上传 / slot 克隆 record 过这个 event。
            #   （_init_h2d_publication 末尾的 event.record()）
            try:
                self.completion.synchronize()
            except BaseException:
                self._state = "failed"
                raise
        self.epoch += 1
        self._stream = stream
        self._stream_id = None if stream is None else stream.stream_id
        self._thread = threading.get_ident()
        for binding in self._bindings:
            binding._writable = True
        for group in self._groups:
            group._header_busy = False
        self._state = "active"

    def finish(self):
        """Seal preparation and cover every queued source read."""
        # 【译】封存准备阶段，并覆盖所有已排队的源读取。
        # 在 owner 的 stream 上 record completion event：它排在本 epoch 所有 H2D 之后，下一次 begin 等它即可。只排命令，不等 GPU
        self._check_active()
        try:
            if self.completion is not None:
                self.completion.record(self._stream)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "sealed"

    def resume(self):
        """Reopen a later phase without resetting the forward's ledger."""
        # 【译】重新打开同一 forward 的后续阶段，不重置这次 forward 的账本。
        # 调用方：deepseek_v4_attn.py（V4 TBO 的晚到准备，第8批）、plugin/vllm/v41_core_overrides.py
        if self._state != "sealed":
            raise PublicationError(f"cannot resume: owner is {self._state}")
        if threading.get_ident() != self._thread:
            raise PublicationError("resume on the original owner thread")
        if self._current_stream() != self._stream:
            raise PublicationError("resume on the original owner stream")
        self._state = "active"
        # Sources stay protected; acquire_write must precede their next write.
        # 【译】源仍然受保护；下一次写它们之前必须先 acquire_write。

    # _wait_sources：在 epoch 中途就要改写一个已借给 DMA 的源时调用（acquire_write）——立刻 record 再 synchronize，
    #   CPU 等到本 stream 上已排的所有工作完成。代价大（一次完整的 stream 同步），所以只用于明确声明了理由的再次发布
    def _wait_sources(self):
        self._check_active()
        try:
            if self.completion is not None:
                self.completion.record(self._stream)
                self.completion.synchronize()
        except BaseException:
            self._state = "failed"
            raise
        for group in self._groups:
            group._header_busy = False

    def fail(self):
        """Retain storage and reject reuse after partially submitted failures."""
        # 【译】部分提交后失败时：保留存储，并拒绝再复用。
        # 已经排出去的拷贝可能还在读 pinned 源，所以不能释放或重用；之后任何 begin 都会因状态不对而报错
        self._state = "failed"

    def drain(self):
        """Drain failed work; the owner remains failed and must be rebuilt."""
        # 【译】排空失败的工作；owner 仍处于 failed 状态，必须重建。
        if self._state != "failed":
            raise PublicationError("drain is only for failed publication owners")
        if self._stream is not None:
            self._stream.synchronize()


# BufferPublication：一个 CpuGpuBuffer 的绑定。记录源（buffer.cpu，pinned）、目的（buffer.gpu）、计数单位与容量，
#   以及两个状态位：_epoch（最后一次发布在哪个 epoch）、_writable（源此刻能不能写）
#   容量与 bytes_per_count 的例子：
#     rows 单位、block_tables [512, 2048] int32 → capacity=512 行，bytes_per_count=2048×4=8192
#     elements 单位、mrope_positions [3, 16384] int64 → capacity=49152 个元素，bytes_per_count=8
#   构造时检查：源在 CPU、目的在本 owner 的设备上，形状和 dtype 完全一致（发布不做 reshape / 类型转换）；
#     elements / bytes 单位要求两端连续（按扁平视图拷前缀）
class BufferPublication:
    def __init__(self, owner, buffer, name, unit):
        self.owner, self.buffer, self.name = owner, buffer, name
        self.source, self.destination = buffer.cpu, buffer.gpu
        self.device = self.destination.device
        if self.device != owner.device or self.source.device.type != "cpu":
            raise ValueError(f"{name}: incorrect source/destination device")
        if (
            self.source.shape != self.destination.shape
            or self.source.dtype != self.destination.dtype
        ):
            raise ValueError(f"{name}: publication cannot reshape or cast")
        if unit not in ("rows", "elements", "bytes"):
            raise ValueError(f"{name}: unknown count unit {unit}")
        if unit != "rows" and not (
            self.source.is_contiguous() and self.destination.is_contiguous()
        ):
            raise ValueError(f"{name}: flat/byte publication must be contiguous")
        self.unit = unit
        self.source_range = _range(self.source)
        self.destination_range = _range(self.destination)
        self.row_capacity = self.source.shape[0] if self.source.ndim else 1
        itemsize = self.source.element_size()
        if unit == "rows":
            self.capacity = self.row_capacity
            self.bytes_per_count = itemsize
            for width in self.source.shape[1:]:
                self.bytes_per_count *= width
            self._source_view, self._destination_view = self.source, self.destination
        else:
            self.capacity = self.source.numel() * (itemsize if unit == "bytes" else 1)
            self.bytes_per_count = 1 if unit == "bytes" else itemsize
            src, dst = self.source.reshape(-1), self.destination.reshape(-1)
            if unit == "bytes":
                src, dst = src.view(torch.uint8), dst.view(torch.uint8)
            self._source_view, self._destination_view = src, dst
        self._epoch = -1
        self._writable = False
        self._single = None
        # At most four private prefix pairs per binding, with FIFO eviction.
        # 【译】每个绑定最多缓存 4 对私有的前缀视图，先进先出淘汰。
        #   （source[:count] / destination[:count] 这种切片每次都会新建张量对象，花几微秒；同一个 count 常常连续出现）
        self._prefixes = {}

    def acquire_write(self, *, republish_reason=None):
        """Wait before rewriting a borrowed source, never afterwards."""
        # 【译】在改写一个已借出的源之前等待，而不是之后。
        # 本 epoch 已发布过它又要再写，必须给 republish_reason；源不可写时调 _wait_sources 等 DMA 读完。返回可写的 pinned 张量
        self.owner._check_active()
        if self._epoch == self.owner.epoch and not _reason(republish_reason):
            raise PublicationError(
                f"{self.name}: repeated write needs republish_reason"
            )
        if not self._writable:
            self.owner._wait_sources()
            self._writable = True
        return self.source

    # _validate：发布前的单成员检查——缓冲区没被换掉（CpuGpuBuffer 也禁止替换 cpu/gpu）、本 epoch 没发布过（或给了理由）、
    #   源处于可写状态（说明生产者是在 begin / acquire_write 之后写的）、count 合法
    def _validate(self, count, reason):
        if (
            self.buffer.cpu is not self.source
            or self.buffer.gpu is not self.destination
        ):
            raise PublicationError(
                f"{self.name}: binding changed; rebuild the buffer and owner"
            )
        if self._epoch == self.owner.epoch and not _reason(reason):
            raise PublicationError(
                f"{self.name}: repeated publish needs republish_reason"
            )
        if not self._writable:
            raise PublicationError(
                f"{self.name}: acquire_write before modifying source"
            )
        return _count(count, self.capacity, self.name)

    # _copy：direct 传输——拷前 count 个单位；整块时直接拷整个视图，否则用缓存的前缀视图。non_blocking=True 只排命令
    def _copy(self, count):
        if count == self.capacity:
            self._destination_view.copy_(self._source_view, non_blocking=True)
        elif count:
            prefix = self._prefixes.get(count)
            if prefix is None:
                prefix = (self._source_view[:count], self._destination_view[:count])
                if len(self._prefixes) == 4:
                    del self._prefixes[next(iter(self._prefixes))]
                self._prefixes[count] = prefix
            prefix[1].copy_(prefix[0], non_blocking=True)

    def copy_to_gpu(self, n=None, *, republish_reason=None):
        # Legacy n always means rows, including on a flat registered region.
        # 【译】旧接口的 n 永远表示行数，即使这个区域是按扁平单位登记的。
        # 例：mrope_positions [3, 16384] 按 elements 登记，n=2 行 → count = 2 × 49152 // 3 = 32768 个元素
        # 这是 CpuGpuBuffer.copy_to_gpu 的落脚点：经单成员组 publish，享受同样的检查
        if n is None:
            count = self.capacity
        else:
            rows = _count(n, self.row_capacity, self.name)
            count = (
                rows
                if self.unit == "rows"
                else (
                    rows * self.capacity // self.row_capacity
                    if self.row_capacity
                    else 0
                )
            )
        self._single.publish((count,), republish_reasons=(republish_reason,))
        return self.destination if n is None else self.destination[:n]


class PublicationGroup:
    """Fixed members validated together before any direct or kernel enqueue."""

    # 【译】一组固定的成员，在排任何直接拷贝或 kernel 之前一起校验。
    # 为什么要"整组先校验、再提交"：若第 3 个成员才发现违规，前 2 个的拷贝已经排出去了，状态就半新半旧；
    #   先全部校验通过再改账本、排拷贝，失败时要么什么都没发生，要么 owner 进入 failed
    # counts 用法：生产者把每个成员这次要发布多少填进 group.counts[group.indices[名字]]（None = 这次不发布该成员，
    #   0 = 显式发布空内容），最后调 group.publish(group.counts)。组合组的不同成员可以由不同的生产者先后填

    def __init__(self, owner, name, members):
        self.owner, self.name, self.members = owner, name, members
        self._single_member = members[0] if len(members) == 1 else None
        # CPU-only producer scratch; it is never borrowed by the GPU.
        # 【译】只在 CPU 上用的生产者草稿；GPU 从不借用它。
        self.counts = [None] * len(members)
        self.indices = {member.name: i for i, member in enumerate(members)}
        self._buffer_indices = {member.buffer: i for i, member in enumerate(members)}
        self._counts = [None] * len(members)
        self._header_busy = False
        self.transport = "direct"
        self.fallback_reason = None
        self._backend = None

    def set_count(self, buffer, count):
        """Record a producer's count without copying data or granting write access.

        This only fills CPU count scratch. The producer must acquire sources
        before writing them; publish validates the entire group before enqueue.
        Lookup by buffer identity keeps binding names private to registration.
        """
        # 【译】记录一个生产者的计数，不拷贝数据，也不授予写权限。
        #
        #   这只是填 CPU 上的计数草稿。生产者在写源之前必须先获取它们；publish 会在排队之前校验整个组。
        #   按缓冲区对象查找，让绑定名只在登记时使用。
        # 调用方：v4_kernels/compress_plan.py、utils/block_tables.py（页表没变时给 None，省掉这次上传）
        self.counts[self._buffer_indices[buffer]] = count

    def check_writable(self):
        """Check fresh sources before a producer writes any group member.

        The owner's begin() already acquired these regions. Check its common
        thread/device/stream once, then each binding without another GPU API
        query. This does not grant republish permission or wait for a reused
        source; explicit rewrites still use BufferPublication.acquire_write.
        """
        # 【译】在生产者写任何组成员之前，检查源是新鲜可写的。
        #
        #   owner 的 begin() 已经获取了这些区域。这里只检查一次 owner 公共的线程 / 设备 / stream，然后逐个检查绑定，
        #   不再额外调用 GPU API。它不授予再次发布的许可，也不等待被复用的源；显式改写仍然要用
        #   BufferPublication.acquire_write。
        # 调用方：h2d_producer 装饰器、各 builder 的 _check_metadata_writable
        self.owner._check_active()
        for binding in self.members:
            binding._validate(0, None)

    def use_transport(self, transport):
        """Choose direct or packed at initialization, before sources are used."""
        # 【译】在初始化时、源被使用之前选择 direct 或 packed。
        # packed 需要 GPU 目的端；PackedCopy.create 不合格时（成员 <2、有不连续的成员）返回原因，组退回 direct
        if self.owner._state != "idle":
            raise PublicationError("choose the transport during initialization")
        if transport == "packed":
            if self.owner.device.type != "cuda":
                backend, reason = None, "packing requires GPU destinations"
            else:
                from atom.utils.packed_h2d import PackedCopy

                backend, reason = PackedCopy.create(self.members, self.owner.device)
        elif transport == "direct":
            backend, reason = None, None
        else:
            raise ValueError("H2D transport must be direct or packed")
        self._backend, self.fallback_reason = backend, reason
        self.transport = "packed" if backend is not None else "direct"
        return self.transport

    # publish：每步真正的提交点。counts[i] 对应 members[i]（None 跳过、0 显式发布空、>0 拷前 count 个单位）
    #   流程：检查 owner → 单成员 direct 快速路径 → 逐个 _validate → （packed 时）检查 arena 的 header 没被在途 DMA 借用
    #     → 统一改账本（_epoch、_writable）→ 提交（packed：backend.submit；direct：逐个 _copy）
    #   任何一步在提交阶段抛异常 → owner.fail()
    def publish(self, counts, *, republish_reasons=None):
        self.owner._check_active()
        backend = self._backend
        if len(counts) != len(self.members):
            raise ValueError(f"{self.name}: one count required per member")
        if republish_reasons is not None and len(republish_reasons) != len(counts):
            raise ValueError(f"{self.name}: one reason entry required per member")
        # Legacy wrapper uploads use a fixed single-member group. Preserve its
        # checks, ledger and failure semantics without three general loops.
        # 【译】旧包装接口（CpuGpuBuffer.copy_to_gpu）的上传使用固定的单成员组。保留它的检查、账本和失败语义，
        #   但不走下面三个通用循环（这条路径每步要走很多次，省几个 Python 循环）。
        if backend is None and self._single_member is not None:
            count = counts[0]
            if count is None:
                self._counts[0] = None
                return
            binding = self._single_member
            count = binding._validate(
                count, None if republish_reasons is None else republish_reasons[0]
            )
            self._counts[0] = count
            binding._epoch = self.owner.epoch
            binding._writable = count == 0
            try:
                if count:
                    binding._copy(count)
            except BaseException:
                self.owner.fail()
                raise
            return
        active_members = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            self._counts[i] = (
                None
                if count is None
                else binding._validate(
                    count, None if republish_reasons is None else republish_reasons[i]
                )
            )
            active_members += self._counts[i] is not None and self._counts[i] > 0
        # packed 的 arena（含 header）是一块 pinned 内存：上一次打包提交的 DMA 还没读完时不能再改写它
        #   （begin / _wait_sources 会把 _header_busy 清掉）
        if backend is not None and active_members > 1 and self._header_busy:
            raise PublicationError(
                "transport counts are in flight; acquire sources first"
            )
        # No GPU-visible descriptor or ledger mutation until the WHOLE group
        # validates. Explicit zero counts count; omitted members do not.
        # 【译】整个组都校验通过之前，不改任何 GPU 可见的描述符或账本。显式给 0 的计数也算发布；省略（None）的成员不算。
        # count==0 的成员没有东西被 DMA 借走，所以保持可写
        for binding, count in zip(self.members, self._counts):
            if count is not None:
                binding._epoch = self.owner.epoch
                binding._writable = count == 0
        try:
            if backend is not None and active_members:
                # Direct fallback leaves any earlier packed read in flight.
                # 【译】（submit 内部）退回 direct 时，之前那次打包读取仍在途。
                #   submit 返回 arena 是否被借用；用 |= 是为了不把之前仍在途的借用状态清掉
                self._header_busy |= backend.submit(self._counts)
            elif backend is None:
                for binding, count in zip(self.members, self._counts):
                    if count:
                        binding._copy(count)
        except BaseException:
            self.owner.fail()
            raise
