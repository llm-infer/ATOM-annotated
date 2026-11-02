# SPDX-License-Identifier: MIT
"""Request leases for vision outputs and scatter into arbitrary prefill chunks.

The engine releases leases after the last in-flight consumer completes. The
scheduler retains CPU payloads for retry, but sends them only until a successful
prefill acknowledges the worker's lease. Decode carries no image payload.
"""

# 【译】视觉输出的请求租约（lease），以及把它们散射（scatter）进任意的prefill chunk。
#
#   引擎在最后一个在途的使用者完成之后释放租约。调度器保留CPU上的载荷以备重试，但只在一次成功的prefill
#   确认了worker端的租约之前发送它们。decode不携带图像载荷。
#
# 【导读】多模态prefill的运行时：图像经视觉编码器得到的embedding怎么缓存、怎么跨chunked prefill分批填进输入embedding
# 背景：多模态prompt里每张图对应一段连续的占位token（几百到上千个），prefill时要把这些位置的词嵌入替换成视觉编码器的输出。
#   ATOM支持chunked prefill，一张图的占位段可能被切进多个chunk，所以需要：
#   1. 视觉编码只做一次，结果在ModelRunner进程里缓存，直到该请求结束（VisionEmbeddingCache，按cache_seed去重）
#   2. 每个chunk只把"本chunk范围 ∩ 图像占位段"那部分embedding散射进去（embedding_indices）
#   3. 第一次prefill成功之后，调度器不再把像素张量发给worker，只发span元数据（prefill_media_payload）
# 数据：multimodal_data是一个dict，由模型的输入构建函数产出，含pixel_values（图像patch张量）、image_grid_thw（每张图的网格）、
#   token_types、embedding_spans=((起始位置, token数), ...)、cache_seed（内容指纹）
#   v0.1.8中只有DeepSeek-V4.1的image_processing.build_inputs会产出embedding_spans
# 调用关系（"span路径"：数据里有embedding_spans时）：
#   Sequence.__init__ / 模型的图像预处理 -> multimodal_cache_seed（第1批sequence.py）
#   Scheduler组batch -> prefill_media_payload；prefill完成后把seq.multimodal_cache_ready置True（第3批）
#   ModelRunner前向 -> embed_multimodal_batch -> VisionEmbeddingCache.acquire（第5批）
#   EngineCore在请求结束/被拒绝时 -> runner的release_multimodal_requests -> VisionEmbeddingCache.release（本批engine_core.py）
# 没有embedding_spans的旧路径（如Qwen3.5-VL）不经过本文件：调度器把整份数据随第一次prefill发出后就清空，
#   该请求的prefill也不能切块（Scheduler._requires_atomic_prefill），ModelRunner里直接拼接pixel_values做视觉编码
# 【分支】DeepSeek-V4.1的图像预处理与视觉模型见第8批

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


# 计算多模态内容的指纹（64位整数）：对pixel_values / image_grid_thw / token_types的dtype、形状和原始字节，以及embedding_spans做blake2b
#   输入：multimodal_data字典；输出：int
#   调用方：Sequence.__init__（数据里没带cache_seed时补算，第1批）、models/deepseek_v41/image_processing.py（第8批）
#   用途：①作为Sequence.cache_seed参与前缀缓存的block哈希链（BlockManager，第4批），使"同样的占位token + 不同图片"不会误命中；
#     ②作为VisionEmbeddingCache的键，相同图片内容的请求共享一份视觉embedding
def multimodal_cache_seed(data):
    """Hash processed media once; token-only prefix identity is insufficient.

    Include layout as well as values: an identical patch buffer arranged into
    different image grids need not produce the same language embeddings.
    """
    # 【译】对处理后的媒体只做一次哈希；仅凭token判断前缀是否相同是不够的。
    #
    #   布局和数值都要算进去：同一份patch缓冲区排成不同的图像网格，不一定产生相同的语言侧embedding。
    # 解释：person是blake2b的个性化参数，相当于给哈希加上命名空间和版本号
    digest = hashlib.blake2b(digest_size=8, person=b"ATOM-media-v1")
    for name in ("pixel_values", "image_grid_thw", "token_types"):
        value = data.get(name)
        if value is None:
            continue
        # torch张量：先拷到CPU、保证连续，再按uint8视图取原始字节（bf16等dtype无法直接转numpy）；其他情况按numpy数组处理
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            value = np.ascontiguousarray(value)
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
    digest.update(repr(data.get("embedding_spans", ())).encode())
    return int.from_bytes(digest.digest(), "little")


# 调度器为prefill序列组batch时调用（scheduler.py，第3批）：生成本次要发给ModelRunner的那份多模态数据
#   token_types总不发送；cache_ready=True（之前某次prefill已成功、worker端已持有租约）时连pixel_values / image_grid_thw也不发，
#   只剩embedding_spans等元数据；返回的新dict附带cache_seed，worker用它找到缓存的视觉embedding
def prefill_media_payload(data, cache_seed, *, cache_ready):
    """Keep span metadata; send media tensors only until the worker owns a lease."""
    # 【译】保留span元数据；只在worker持有租约之前发送媒体张量。
    omitted = {"token_types"}
    if cache_ready:
        omitted.update(("pixel_values", "image_grid_thw"))
    return {
        **{key: value for key, value in data.items() if key not in omitted},
        "cache_seed": cache_seed,
    }


# 计算本chunk中需要替换成视觉embedding的位置
#   spans：((起始位置, token数), ...)，是请求内的绝对token位置；position / length：本chunk在请求中的起点和长度
#   返回(query, source)：query是chunk内的相对位置，source是该请求视觉embedding（所有图按顺序拼接）中的行号
#   例：spans=((2, 4),)（位置2~5是图像），本chunk position=4、length=4（位置4~7）
#     -> 相交部分为位置4~5：query=[0, 1]，source=[2, 3]（这张图的第3、4行embedding）
def embedding_indices(spans, position, length):
    """Paired query/embedding indices for intersections with an input slice."""
    # 【译】与一段输入切片相交部分的成对索引：query索引与embedding索引。
    # 解释：offset是当前span在拼接后的视觉embedding中的起始行
    query, source, offset = [], [], 0
    for start, count in spans:
        first, end = max(start, position), min(start + count, position + length)
        if first < end:
            query.extend(range(first - position, end - position))
            source.extend(range(offset + first - start, offset + end - start))
        offset += count
    return query, source


# 缓存项：values是该内容的视觉embedding [所有图像token总数, hidden_size]；users是正在使用它的请求id集合
@dataclass
class VisionEntry:
    values: torch.Tensor
    users: set[int] = field(default_factory=set)


# ModelRunner进程内的视觉embedding缓存（每个ModelRunner一个，即model_runner.py的self.vision_embeddings，第5批）
#   entries：cache_seed -> VisionEntry，相同图片内容的多个请求共享一份
#   leases：请求id -> cache_seed，即"租约"：登记后该请求后续chunk不用再发像素
#   encodes：累计做过视觉编码的图片张数（统计用，测试里会检查）
class VisionEmbeddingCache:
    def __init__(self):
        self.entries: dict[int, VisionEntry] = {}
        self.leases: dict[int, int] = {}
        self.encodes = 0

    # 请求在某个prefill chunk中领取视觉embedding，缓存里没有时调用encode计算
    #   输入：request_id；data为本次batch里该请求的多模态数据（至少有cache_seed和embedding_spans，首次还带pixel_values）；
    #     encode为回调，(data) -> 视觉embedding张量
    #   输出：values张量；副作用：登记租约（同一请求重复acquire是幂等的）
    def acquire(self, request_id, data, encode):
        seed = data["cache_seed"]
        if request_id in self.leases and self.leases[request_id] != seed:
            raise ValueError("A live vision request cannot change its image identity")
        entry = self.entries.get(seed)
        if entry is None:
            # 调度器只有在prefill成功（租约已确认）后才停发像素；此时缓存里却没有，说明worker端状态已丢失，只能报错
            if "pixel_values" not in data:
                raise RuntimeError("Vision lease missing for an acknowledged request")
            values = encode(data)
            # 校验：视觉编码输出的行数必须等于所有图像占位token的总数，否则散射会错位
            expected = sum(count for _, count in data["embedding_spans"])
            if values.shape[0] != expected:
                raise ValueError(
                    "Vision output rows disagree with explicit image spans"
                )
            entry = VisionEntry(values)
            self.entries[seed] = entry
            self.encodes += len(data["embedding_spans"])
        entry.users.add(request_id)
        self.leases[request_id] = seed
        # record_stream：告诉CUDA缓存分配器这块显存还会在当前stream上使用，防止释放后被其他stream上的分配提前复用
        if entry.values.is_cuda:
            entry.values.record_stream(torch.cuda.current_stream(entry.values.device))
        return entry.values

    # 请求结束（完成/中止/被拒绝）时调用：EngineCore._release_multimodal_requests -> runner_mgr.call_func("release_multimodal_requests")
    #   -> 这里；撤销这些请求的租约，没有使用者的缓存项随之删除
    def release(self, request_ids):
        for request_id in request_ids:
            seed = self.leases.pop(request_id, None)
            if seed is None:
                continue
            entry = self.entries[seed]
            entry.users.remove(request_id)
            if not entry.users:
                del self.entries[seed]

    # ModelRunner.exit()时清空
    def clear(self):
        self.entries.clear()
        self.leases.clear()


# ModelRunner在prefill前向时调用（batch中所有多模态数据都带embedding_spans时，第5批run_model）：
#   先算整批的词嵌入，再按请求把视觉embedding覆盖到图像占位位置
#   输入：input_ids [本批token总数]；batch.req_ids / num_scheduled_tokens / context_lens按请求一一对应
#   输出：hidden即inputs_embeds [本批token总数, hidden_size]，交给模型forward（代替input_ids）
def embed_multimodal_batch(model, cache, input_ids, batch, device, dtype):
    """Consume request leases and scatter only the current chunk's intersections."""
    # 【译】使用请求租约，并只散射当前chunk相交的那部分。
    # 解释：offset是当前请求在本批扁平token序列中的起点
    hidden = model.embed_input_ids(input_ids)
    offset = 0
    for request_id, length, end in zip(
        batch.req_ids, batch.num_scheduled_tokens, batch.context_lens
    ):
        data = batch.multimodal_data.get(request_id)
        if data is not None:
            # Acquiring before the first image intersection lets later chunks
            # send descriptors only, even if the first chunk contains just text.
            # 【译】在第一次与图像相交之前就领取租约，使后续chunk只需发送描述信息，即使第一个chunk里只有文本。
            # 解释：本chunk不含图像位置也会调用acquire，所以视觉编码总是发生在该请求的第一个prefill chunk
            values = cache.acquire(
                request_id,
                data,
                lambda payload: model.get_vision_embeddings(
                    payload["pixel_values"].to(device=device, dtype=dtype),
                    payload["image_grid_thw"],
                ),
            )
            # context_lens是本chunk结束后的总长度（已缓存 + 本次调度），减去本chunk长度即本chunk在请求中的起点
            query, source = embedding_indices(
                data["embedding_spans"], int(end) - int(length), int(length)
            )
            if query:
                query = torch.tensor(query, device=device) + offset
                source = torch.tensor(source, device=device)
                # 把视觉embedding的对应行写进本批hidden的对应位置（query已加上请求在本批中的offset）
                hidden[query] = values[source]
        offset += int(length)
    return hidden
