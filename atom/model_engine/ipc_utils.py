# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
CUDA / ROCm IPC helpers for sharing GPU tensors across processes.

Uses tensor._share_cuda_() / UntypedStorage._new_shared_cuda() for the
low-level IPC handle path (hipIpcGetMemHandle / hipIpcOpenMemHandle on ROCm).
Both processes must be on the same physical GPU device.

Phase 1 (KV cache sharing):
  - export_kv_cache_handle  — called by PrefillEngineCore after allocate_kv_cache()
  - import_kv_cache         — called by DecodeEngineCore at startup

Phase 2 (weight sharing):
  - export_model_weight_handles  — called by PrefillEngineCore after load_model()
  - import_model_weights         — called by DecodeEngineCore at startup (frees own copy)
"""

# 【译】用于跨进程共享GPU张量的CUDA / ROCm IPC辅助函数。
#
#   底层IPC句柄路径使用 tensor._share_cuda_() / UntypedStorage._new_shared_cuda()（在ROCm上对应
#   hipIpcGetMemHandle / hipIpcOpenMemHandle）。两个进程必须在同一块物理GPU上。
#
#   第1阶段（KV cache共享）：
#     - export_kv_cache_handle  —— 由PrefillEngineCore在 allocate_kv_cache() 之后调用
#     - import_kv_cache         —— 由DecodeEngineCore在启动时调用
#
#   第2阶段（权重共享）：
#     - export_model_weight_handles  —— 由PrefillEngineCore在 load_model() 之后调用
#     - import_model_weights         —— 由DecodeEngineCore在启动时调用（释放它自己的那份）
#
# 本文件：【分支】RapidServe（同机PD分离）的显存共享工具，见第9批
#   同一组GPU上跑两个EngineCore：prefill一个、decode一个。decode端不另外加载权重、不另外分配KV池，
#   而是通过IPC句柄把prefill进程的那块显存映射进来，两边看到的是同一份权重和同一个KV池
# 调用方：ModelRunner.export_model_weight_ipc_handles / import_model_weight_ipc_handles /
#   export_kv_cache_ipc_handle / import_kv_cache_ipc_handle（model_runner.py，每个TP rank各自导出/导入自己那张卡的）；
#   它们再由PrefillEngineCore / DecodeEngineCore经call_func驱动（engine_core.py），句柄经临时文件和ZMQ传给对端
# 解释：所以准确地说，这些函数运行在prefill/decode两边的ModelRunner进程里，而不是EngineCore进程里
#
# 为什么同机PD分离要共享显存，而不是各自加载一份：
#   权重：例如一个70B模型的bf16权重约140GB，TP=8时每张卡约17.5GB；prefill、decode两个进程各放一份就是35GB/卡，
#     多出来的17.5GB全部要从KV cache预算里扣掉，直接减少能同时服务的请求数
#   KV cache：两边看到同一个分页KV池，prefill算完把KV写进某些block，decode拿着同样的block id直接读——
#     KV"传输"退化成传几个block编号。跨机器的PD分离则必须把这些KV经网络（RDMA等）真正搬过去
#   代价：两个进程跑在同一组GPU上，prefill和decode的kernel会互相争抢算力和显存带宽
#
# 生命周期约束：IPC句柄只是"另一个进程那块显存的一个引用"，必须由分配它的进程（prefill）一直持有原张量；
#   原张量一旦被释放（显存被缓存分配器回收或改作他用），decode端的视图就会读到别人的数据

import logging

import torch
from torch import nn

logger = logging.getLogger("atom")


# 把一个GPU张量变成可以pickle的描述（IPC句柄 + dtype/shape/stride/offset），供另一个进程重建
def _export_tensor(t: torch.Tensor) -> dict:
    """Serialize a CUDA tensor to a dict that can be pickled and sent cross-process.

    Uses tensor._share_cuda_() which calls hipIpcGetMemHandle on ROCm.
    Returns metadata needed to reconstruct the tensor on the other side.
    """
    # 【译】把一个CUDA张量序列化成一个dict，可以pickle后跨进程发送。
    #
    #   使用 tensor._share_cuda_()，它在ROCm上调用 hipIpcGetMemHandle。
    #   返回在另一端重建张量所需的元数据。
    # 注意：若t本来不连续，contiguous()会生成一份新拷贝，导出的就是这份拷贝而不是原张量
    t = t.contiguous()
    # IPC句柄针对的是整个底层storage（一次显存分配），视图信息另外记录在下面几项里
    #   更细一点：PyTorch的缓存分配器常把多个storage放在同一次hipMalloc得到的大段里，
    #   _share_cuda_返回的元组里既有这一整段的句柄，也有本storage在段内的字节偏移，导入端据此定位
    share_cuda_args = t.untyped_storage()._share_cuda_()
    return {
        "share_cuda_args": share_cuda_args,
        "dtype": t.dtype,
        "shape": t.shape,
        "stride": t.stride(),
        "storage_offset": t.storage_offset(),
    }


# _export_tensor的逆操作：打开IPC句柄得到同一块显存，再按记录的shape/stride/offset套上视图
def _import_tensor(meta: dict) -> torch.Tensor:
    """Reconstruct a CUDA tensor from the dict produced by _export_tensor.

    Calls UntypedStorage._new_shared_cuda() which calls hipIpcOpenMemHandle.
    """
    # 【译】从 _export_tensor 生成的dict重建一个CUDA张量。
    #
    #   调用 UntypedStorage._new_shared_cuda()，它会调用 hipIpcOpenMemHandle。
    storage = torch.UntypedStorage._new_shared_cuda(*meta["share_cuda_args"])
    # 先建一个空张量，再用set_把它指向导入的storage（零拷贝，不分配新显存）
    t = torch.empty(0, dtype=meta["dtype"], device="cuda")
    t.set_(storage, meta["storage_offset"], meta["shape"], meta["stride"])
    return t


# ---------------------------------------------------------------------------
# KV cache (Phase 1)
# ---------------------------------------------------------------------------
# 【译】KV cache（第1阶段）


# 导出整个分页KV池（一个大张量）的IPC句柄
def export_kv_cache_handle(kv_cache: torch.Tensor) -> dict:
    """Export the paged pool as a CUDA IPC handle.

    One handle because there is one allocation: the dequantization scales, any
    indexer key cache and a draft's sibling pool are regions of it, and the
    decode side finds them by carving with the same declarations rather than by
    being sent a tensor per region.

    Must be called from the process that allocated the tensor (prefill).
    Returns a dict that can be pickled and sent over ZMQ to the decode process.
    """
    # 【译】把分页池导出为一个CUDA IPC句柄。
    #
    #   只要一个句柄，因为只有一次分配：反量化scale、可能的索引器key cache以及草稿模型的兄弟池都是它的区域，
    #   decode端用同样的声明切分出它们，而不是为每个区域各收一个张量。
    #
    #   必须在分配了该张量的进程（prefill）中调用。返回一个dict，可以pickle后经ZMQ发给decode进程。
    return {"kv_cache": _export_tensor(kv_cache)}


# decode端：把prefill的KV池映射进来
def import_kv_cache(meta: dict) -> torch.Tensor:
    """Reconstruct the paged pool from its CUDA IPC handle.

    Must be called from the consumer process (decode). The returned tensor
    shares GPU memory with prefill's allocation — no copy.
    """
    # 【译】从分页池的CUDA IPC句柄重建它。
    #
    #   必须在消费者进程（decode）中调用。返回的张量与prefill的分配共享GPU显存——不拷贝。
    return _import_tensor(meta["kv_cache"])


# ---------------------------------------------------------------------------
# Model weights (Phase 2)
# ---------------------------------------------------------------------------
# 【译】模型权重（第2阶段）


# 导出模型的所有权重张量：参数、注册的buffer、以及模块上的普通张量属性
# 输出：{"__param__名字" / "__buf__名字" / "__attr__模块.属性": 元数据dict}，前缀告诉导入端该放回哪里
def export_model_weight_handles(model: nn.Module) -> dict:
    """Export all model parameter tensors as CUDA IPC handles.

    Also exports MLA weight-absorbed tensors (W_K/W_K_scale/W_V/W_V_scale)
    which are plain tensor attributes set by process_weights_after_loading(),
    not nn.Parameters, so named_parameters() misses them.

    Must be called from the process that allocated the weights (prefill),
    after load_model() completes.  Returns a dict {key: meta_dict}.
    """
    # 【译】把模型的所有参数张量导出为CUDA IPC句柄。
    #
    #   同时导出MLA权重吸收后的张量（W_K/W_K_scale/W_V/W_V_scale），它们是 process_weights_after_loading() 设置的
    #   普通张量属性，不是 nn.Parameter，所以 named_parameters() 找不到它们。
    #
    #   必须在分配了这些权重的进程（prefill）中、load_model() 完成之后调用。返回一个dict {key: meta_dict}。
    handles = {}
    # Parameters. remove_duplicate=False so a Parameter registered under multiple
    # names (e.g. e_score_correction_bias, shared by gate + experts) is exported
    # under EVERY name — otherwise the consumer only materializes one of the
    # aliased registrations and the other stays on meta.
    # 【译】参数。remove_duplicate=False，这样一个以多个名字注册的Parameter（例如被gate和experts共享的
    #   e_score_correction_bias）会以*每一个*名字导出——否则消费者只会实例化其中一个别名注册，另一个仍留在meta上。
    for name, param in model.named_parameters(remove_duplicate=False):
        handles[f"__param__{name}"] = _export_tensor(param.data)
    # Registered buffers (non-persistent included).
    # 【译】注册的buffer（包括非持久化的）。
    for name, buf in model.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda and buf.numel() > 0:
            handles[f"__buf__{name}"] = _export_tensor(buf)
    # Plain tensor attributes set by process_weights_after_loading() — e.g. the
    # MLA absorbed W_K/W_V — which are neither Parameters nor registered buffers.
    # 【译】process_weights_after_loading() 设置的普通张量属性——例如MLA吸收后的W_K/W_V——它们既不是Parameter，
    #   也不是注册的buffer。
    # 遍历每个模块的__dict__，找出GPU上、非空、又不是Parameter的张量
    for mod_name, mod in model.named_modules():
        for attr, val in list(mod.__dict__.items()):
            if (
                isinstance(val, torch.Tensor)
                and not isinstance(val, nn.Parameter)
                and val.is_cuda
                and val.numel() > 0
            ):
                key = f"{mod_name}.{attr}" if mod_name else attr
                handles[f"__attr__{key}"] = _export_tensor(val)

    return handles


# decode端：用prefill显存上的视图替换本进程模型的参数/buffer/属性，然后本进程原来的那份被释放
def import_model_weights(model: nn.Module, handles: dict) -> None:
    """Replace model parameters with views into another process's GPU allocation.

    Also restores MLA absorbed tensors exported by export_model_weight_handles.

    Must be called from the consumer process (decode) after receiving the
    handles dict from the producer (prefill).  After this call the decode
    model's parameters point into prefill's GPU memory — zero additional bytes
    are allocated.  The decode process's original weight tensors are freed when
    their reference counts drop to zero.
    """
    # 【译】用指向另一个进程GPU分配的视图替换模型参数。
    #
    #   同时恢复 export_model_weight_handles 导出的MLA吸收张量。
    #
    #   必须在消费者进程（decode）中、收到生产者（prefill）发来的handles dict之后调用。调用之后，decode模型的参数
    #   指向prefill的GPU显存——不额外分配任何字节。decode进程原来的权重张量在引用计数降为0时被释放。
    modules = dict(model.named_modules())
    # remove_duplicate=False to match the export and to materialize every
    # registration of a shared Parameter (see export note).
    # 【译】remove_duplicate=False，与导出保持一致，并实例化共享Parameter的每一个注册（见导出处的说明）。
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())

    for key, meta in handles.items():
        t = _import_tensor(meta)
        # 按key的前缀决定放回哪里
        if key.startswith("__param__"):
            # Rebuild the Parameter around the imported CUDA view (set_data fails
            # for meta->cuda). Create the slot if the consumer's meta model lacks
            # it (process_weights_after_loading may add params on the producer).
            # 【译】围绕导入的CUDA视图重建Parameter（从meta到cuda时set_data会失败）。如果消费者的meta模型缺少这个槽位
            #   就创建它（process_weights_after_loading 可能在生产者一侧新增了参数）。
            name = key[len("__param__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            rg = params[name].requires_grad if name in params else False
            mod._parameters[attr] = nn.Parameter(t, requires_grad=rg)
        elif key.startswith("__buf__"):
            # Keep decode's locally-built real buffers (e.g. RoPE caches built
            # during construction); only fill buffers it is missing or left on
            # meta (those created inside process_weights_after_loading).
            # 【译】保留decode本地构建的真实buffer（例如构造时生成的RoPE缓存）；只填充它缺少的、或留在meta上的buffer
            #   （那些在 process_weights_after_loading 里创建的）。
            name = key[len("__buf__") :]
            existing = buffers.get(name)
            if existing is None or existing.is_meta:
                parent, _, attr = name.rpartition(".")
                mod = modules.get(parent, model)
                if mod is not None:
                    mod._buffers[attr] = t
        elif key.startswith("__attr__"):
            # Plain tensor attribute (e.g. MLA W_K/W_V from process_weights).
            # 【译】普通张量属性（例如process_weights产生的MLA W_K/W_V）。
            name = key[len("__attr__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            if mod is not None:
                setattr(mod, attr, t)

    # 检查还有没有留在meta设备上的参数/buffer（生产者没有导出的），有的话告警，否则forward时才会崩溃
    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if isinstance(b, torch.Tensor) and b.is_meta
    ]
    if leftover:
        logger.warning(
            f"[WT-IMPORT] {len(leftover)} tensors still on meta after import "
            f"(not exported by producer): {leftover[:12]}"
        )
