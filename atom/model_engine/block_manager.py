# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from math import inf, isinf
from time import monotonic
from weakref import WeakKeyDictionary

import numpy as np

from atom.config import Config, DCPConfig
from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    checkpoint_copy_ops_per_batch,
)
from atom.model_engine.prefix_hash import (
    block_hash,
    chained_block_hashes,
    full_prompt_blocks,
    hash_block_size_for,
    prompt_fingerprint,
)
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_cache import StateCache, StateCheckpointCache
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import (
    DEFAULT_STATE_RUNTIME,
    StateMaintenanceOps,
    StateRuntime,
    StateTransfer,
    midstep_target_ok,
    snap_midstep_position,
)
from atom.utils import envs

logger = logging.getLogger("atom")


# =============================================================================================
# 【导读】block_manager.py：KV cache的CPU侧账本——block的分配 / 引用计数 / 释放 / 惰性淘汰、前缀缓存的命中判定，
#   以及带每请求状态的模型（线性注意力、DeepSeek-V4）的状态槽位与状态checkpoint
# =============================================================================================
# 一句话（同nano-vllm）：BlockManager在CPU上记账——哪个block空闲、被几条序列引用、装着哪段前缀（内容hash）；
#   只记元数据，从不碰显存。真正存KV的是ModelRunner在GPU上一次性分配好的显存（第5批）：
#
#   EngineCore进程（CPU，每个DP rank一个）               TP worker进程 × TP（各占一张GPU）
#     Scheduler                                          ModelRunner
#       └─ BlockManager（本文件）                          get_num_blocks：显存预算 → 多少个block、多少个状态槽位
#            self.kv     block空闲 / 引用计数 / hash索引    allocate_kv_cache：按这个数分配显存
#            self.state  状态槽位被谁占着（【分支】）       forward：按block table读写各层KV，按state_slots读写状态
#          给每条序列填 seq.block_table、seq.state_slots
#                    └──────── ScheduledBatch里只带block id / 槽位编号 ────────►
#
#   启动顺序（engine_core.py）：get_num_blocks → allocate_kv_cache → 把block数回填进config.num_kvcache_blocks、各类条目数
#     放进config.pool_entries → 构造Scheduler，Scheduler.__init__再构造本类。所以这里的block id和显存里的block一一对应。
#     TP的各个worker上没有BlockManager；PP下只有head stage调度（第9批）
#
# GPU上有哪些池（预算只分两个区：atom/model_ops/attentions/pool_layout/sub_pool_spec.py的Pool；区里的"条目类"按名字区分编号空间）
#
#   预算区 / 条目类     一个条目装什么（一个rank、本PP stage的全部层）                  编号            CPU侧谁记账
#   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#   PAGE / "kv"        一个block = block_size个token在各层的KV：                       block id        self.kv
#   （随缓存了多少     · MHA/GQA：K、V（fp8时另有每token每KV头1个fp32 scale）          seq.block_table （BlockPool，
#    历史增长；先扣掉  · MLA：每token一行潜在向量（DeepSeek-V3：kv_lora_rank 512 + rope 64 = 576维）     block_pool.py）
#    STATE区，剩下的   · 搭车的同名条目：DSA索引器key（DeepSeek-V3.2 / GLM-5 / MiniMax-M3）、
#    全给它）            DeepSeek-V4的索引器缓存、Eagle3草稿KV——字节相加，共用同一个block id
#                      · 【分支】PAGE状态checkpoint镜像：占若干个block大小的"PAGE单元"
#   STATE / "state"    一条请求的每层定长状态：GDN / KDA的conv + ssm状态，               槽位编号        self.state
#   （随在途请求数      DeepSeek-V4 / V4.1的滑动窗口环 + 压缩器状态；GDN开MTP时每请求    seq.state_slots （StateSlotPool，
#    增长）             1 + k个槽位（多出的k个给草稿被拒时回滚；ReplaySSM下只要1个）                     state_pool.py）
#
#   不带状态的模型（Llama、Qwen3、DeepSeek-V3…）只有PAGE区，self.state的槽位数为0
#   以MHA/GQA为例（attentions/mha_kv_pool.py），PAGE区的字节按"字段 → 层 → block"排开：
#     [K 第0层：block 0 | block 1 | … | block NB−1][K 第1层：…] … [V 第0层：…] …（fp8另有k_scale、v_scale两段）
#     某一层的K视图 [NB, Hkv, D/x, BS, x]（x = 16字节 / 元素字节数，MFMA要的SHUFFLE排布；V是 [NB, Hkv, BS/x, D, x]）
#   和nano-vllm的 kv_cache[2, L, NB, BS, Hkv, D] 同一个思路：block id是跨层的逻辑单位，BlockManager分出1个block，
#     等于在每一层的K、V里各占第block_id页。（个别后端内部的block更小：一个调度器block = block_ratio个后端block，
#     见mha_kv_pool.py的blocks_per_entry；BlockManager只看调度器的block）
#
# 一条请求从头走到尾（block_size = 16，不开DCP / MTP，延迟输出默认开；KV池共100个block；每步prefill预算640个token）
#   调用顺序取自scheduler.py：准入 _probe_hbm_prefix → can_allocate(record=False) → record_allocation → allocate；
#   每步postprocess → hash_blocks（prefill）/ hash_decode_blocks（decode）；decode调度 can_append → may_append；结束 deallocate
#
#   初始：100个block全空闲且没有hash（BlockPool._vacant，id小的先发），hash索引为空
#   ① 请求A，prompt 1000个token：要 ⌈1000/16⌉ = 63个block，最后一个只装 1000 − 62×16 = 8个token
#      can_allocate(A)：沿hash链查前62个block（最后一个永远不复用），第一个就没命中 → 命中0；要63个新block ≤ 100 → 返回0
#      allocate(A, 0)：按整条prompt一次取63个新block → A.block_table = [0, 1, …, 62]，num_cached_tokens = 0；空闲37
#   ② 第1步prefill位置0~639（非最后一块chunk向下对齐到64），postprocess里 hash_blocks(A, 640)：
#      block 0~39写满了，登记 h0 = H(seed, 块0)、h1 = H(h0, 块1) … h39；num_cached_tokens = 640
#   ③ 第2步（Phase 1优先续算）位置640~999：hash_blocks登记block 40~61（h40 … h61）；block 62只装了992~999，不登记。
#      prompt算完；延迟输出下第1个生成token要到下一步才交回CPU，token_ids末尾先挂一个占位 → len(A) = 1001
#   ④ decode：每步算位置 len(A) − 1；can_append / may_append只在 len(A) % 16 == 1 时要一个新block：
#      len = 1001 → 位置1000落在block 62（管992~1007）的偏移8，不用新block；……len = 1009 → 位置1008开一个新块，
#      may_append取block 63 → A.block_table = [0, …, 63]。block 62的16个位置都写完KV后，hash_decode_blocks登记h62
#      （下一轮对话"上轮prompt + 上轮回答"也能命中生成的部分）
#   ⑤ A生成20个token后结束（len = 1020）：deallocate(A)倒序释放63、62、…、0，引用计数都归0。
#      有hash的block 0~62进"cached"空闲队列（越早释放越早被淘汰：62最先、0最后），没hash的block 63进"vacant"；
#      空闲回到100个，但hash索引里h0~h62还在、显存里的KV也没清——这就是惰性淘汰
#   ⑥ 请求B：A的前800个token + 300个新token，共1100个token（⌈1100/16⌉ = 69个block）
#      can_allocate(B)：h0~h49命中（block 0~49，token 0~799），h50（800~815）不同 → 停（链式hash，之后不可能再命中）。
#      命中的50个block都躺在空闲队列里，取回来也占空闲名额：要69个 ≤ 100 → 返回50
#      allocate(B, 50)：block 0~49从空闲队列取回（ref_count = 1，内容原样），其余19个新block优先取vacant的63~81；
#      num_cached_tokens = 50 × 16 = 800，B只prefill 800~1099这300个token
#   ⑦ 之后某个请求要的新block超过vacant存量时，才从cached队头拿block 62（最早释放的）：直到这一刻h62才真正失效
#      （BlockPool.allocate里触发on_evict → _record_evicted）；在此之前任何请求都还能命中它
#   字段的逐步变化（num_cached_tokens、num_hashed_tokens、block_table）见sequence.py Sequence类上方的"字段演化总表"
#
# 前置概念（读者懂MHA/GQA、读过nano-vllm即可；每条先给本文件用到的要点，再指向完整讲解）
#   分页KV / block table、前缀缓存的链式hash：nano-vllm已有。要点见scheduler.py文件开头【前置①③】，完整讲解见sequence.py
#     BlockTable类上方"科普：KV cache分块"、Sequence类上方"科普：前缀缓存"
#   每请求状态与状态checkpoint：线性注意力（GDN：Qwen3-Next / Qwen3.5；KDA：Kimi-K3、GLM-5.3）和DeepSeek-V4 / V4.1的
#     滑动窗口不给每个token留一行KV，而是每请求一份定长状态、每来一个token原地覆盖（Qwen3-Next一层GDN约52万个数，
#     ≈ 一层全注意力里512个token的KV）。前缀命中时KV行还在，"第N个token时的状态"早被覆盖了，只能事先存快照
#     （checkpoint：默认每8192个token一级 + prompt末尾的锚点），于是命中被截到最近的checkpoint（_gated_hit）。
#     例：与旧prompt共享前18432个token → KV命中18432，状态只能从16384恢复 → 从16384开始prefill。
#     要点见scheduler.py文件开头【前置⑤】，完整讲解见sequence.py Replay类上方"科普：状态checkpoint与有界重放"；
#     槽位与checkpoint的实现在本批state_pool.py、page_unit_checkpoint.py
#   DCP（hash block = block_size × DCP，前提是MLA只有1个KV头）：见下面_effective_block_size上方的【前置：DCP】
#   投机解码的草稿lookahead：草稿模型自己也往len(seq)之后写k − 1个位置的KV，见下面draft_kv_lookahead
#   KV offload分层（HBM前缀缓存下面再挂CPU内存 / LMCache）：见下面_joint_kv_boundary上方的【前置：KV offload分层】
#
# 全文件阅读路线（按代码顺序，用函数名作锚点；主干 = 只开TP的普通部署每步都会走，【分支】= 打开某个特性才走）
#   ① 文件开头 ~ BlockManager.__init__
#        _make_block_stored / _make_block_removed / _make_all_cleared .. 【分支】KV事件的构造（本批distributed/kv_events.py）
#        JointBoundaryDecision ........................................... 【分支】KV offload联合加载的边界（第9批）
#        draft_kv_lookahead .............................................. 主干（开投机解码时）：草稿轮多写几个KV位置
#        _PrefixWalk / _CachedWalk ....................................... 主干：一次前缀查找的结果、它的缓存
#        BlockManager.__init__ ........................................... 主干字段在前；状态 / offload字段是【分支】
#   ② 小工具与池子大小
#        compute_hash、_note_index_change、_record_evicted ............... 主干：hash函数、淘汰回调
#        complete_previous_state_batch / take_state_maintenance_ops / state_checkpoint_fates
#          ................................................................ 【分支】每请求状态（第4、8批）
#        _fresh_block .................................................... 主干：取一个新block（≈ nano-vllm的_allocate_block）
#        disown_claimed_prefix ........................................... 【分支】放弃共享前缀前先私有化
#        _checkpoint_has_room / _has_page_units / _ensure_page_units ..... 主干外壳：不开PAGE checkpoint时就是"空闲够不够"
#        num_pool_blocks / draft_lookahead_blocks / max_pool_tokens ...... 主干：一条序列占几个block（含DCP）、最长能收的prompt
#        _effective_block_size / _hash_block_size / _n_hash_blocks / _hash_block_tokens
#          ................................................................ 主干：hash block = block_size × DCP
#        _replay_hit / fall_back_to_hbm_replay ........................... 【分支】DeepSeek-V4有界重放（第8批）
#        _prompt_hashes / _adopt_shipped_chain / _reject_shipped_chain / _block_hash
#          ................................................................ 主干：prompt的hash链只算一次（可直接用前端算好的）
#        _gated_hit ...................................................... 【分支】状态门控：命中截到有checkpoint的位置
#        pool_occupancy .................................................. 统计
#        _joint_kv_boundary / _no_joint / _commit_joint_boundary / _chain_to / prefix_hash_chain
#          ................................................................ 【分支】KV offload联合加载（第9批）
#   ③ 准入：can_allocate → _evaluate ...................................... ★主干核心（≈ nano-vllm的can_allocate）
#        _walk_prefix / _admission_walk / _hit_walk / _take_admission_walk  主干：沿hash链找连续命中
#        can_allocate_memoized / _memoised_walk / _walk_still_valid ...... 【分支】PrefillDelayer、DP成本估算的反复探测
#        record_allocation ............................................... 提交joint边界（没开offload时只记一个"off"）
#        allocate ........................................................ ★主干核心（≈ nano-vllm的allocate）
#        replay_start .................................................... 【分支】重放时prefill从哪开始
#   ④ _state_leg_secured / _attach_state_slots ............................ 【分支】给序列分状态槽位、从checkpoint恢复
#      _tier_can_serve ~ take_state_loads ................................. 【分支】CPU状态层的加载 / 存储（第9批）
#   ⑤ _chain_parent_hash / hash_blocks / hash_decode_blocks ............... ★主干：forward算完KV后才登记hash
#      cancel_state_fork ~ checkpointers_at ............................... 【分支】checkpoint阶梯 / 按需级 / 末尾锚点 /
#                                                                            midstep（forward中途取快照）、统计
#   ⑥ publish_loaded_prefix / register_received_prefix ................... 【分支】offload加载回来的、PD传过来的前缀登记进索引
#      deallocate_partial / acquire_offload_prefix / free_leased_blocks ... 【分支】offload保存期间的提前释放与租约（第9批）
#   ⑦ deallocate / can_append / may_append ................................ ★主干（≈ nano-vllm的同名函数）
#      draft_writes_allocated ............................................. 主干（开投机解码时）：草稿要写的位置都有block了吗
#   ⑧ take_events / clear_cache / kv_events_enabled / record_remote_store  【分支】KV事件；clear_cache是管理接口
#   第一遍建议只读：本导读 → draft_kv_lookahead → __init__的主干字段 → _fresh_block、num_pool_blocks、hash block几个函数
#     → _evaluate、_walk_prefix、allocate → hash_blocks、hash_decode_blocks → ⑦
#
# 对比nano-vllm：nano-vllm的block_manager.py约100行代码（注释版240行）；本文件上游3372行（含大量英文注释），另外还拆出了
#   block_pool.py、prefix_hash.py、state_pool.py、page_unit_checkpoint.py。主干四个动作和nano-vllm一样：
#   can_allocate（沿链式hash找命中）→ allocate（命中的共享、其余新取）→ may_append（decode写满一块补一块）→
#   deallocate（倒序释放，hash留到block被别人拿去覆盖为止）。多出来的每一块解决一个nano-vllm没有的问题：
#
#   多出来的部分（锚点）                             解决什么问题                                          约多少行
#   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#   BlockPool（block_pool.py）                       空闲表分两半：没内容的vacant先发、有hash的cached按LRU   另一文件
#                                                    （可选SLRU）后发；命中的block O(1)从空闲表取回
#   _prompt_hashes、_adopt_shipped_chain、           长prompt的hash链只算一次（前端算好随请求送来），按轮成批    270
#   _hit_walk、can_allocate_memoized                 查表；同一个waiting请求每步被反复探测时复用上次结果
#   num_pool_blocks、max_pool_tokens、               DCP：一个block table条目 = block_size × DCP个全局token、   80
#   hash block一组                                   KV池按每卡本地block数算；API层提前拒绝放不进池子的prompt
#   draft_kv_lookahead、draft_lookahead_blocks、     投机解码：草稿层往len(seq)之后多写k − 1个位置，block要    55
#   draft_writes_allocated                           事先分好，否则写进物理block 0（别人的KV）
#   hash_decode_blocks                               生成的token也登记hash（多轮对话能命中上一轮回答）；投机     40
#                                                    解码下只登记token已确定、KV已写完的block
#   self.state、_gated_hit、_attach_state_slots、    线性注意力 / V4的每请求状态：状态槽位、命中截到有checkpoint 1000
#   checkpoint阶梯 / 按需级 / 锚点 / midstep、       的位置、checkpoint放哪（可借KV池的block存）、forward中途取
#   _replay_hit / replay_start                       快照、DeepSeek-V4用有界重放代替checkpoint
#   JointBoundaryDecision、_joint_kv_boundary、      KV offload：HBM里已淘汰的前缀从LMCache取KV、从CPU状态层取   730
#   CPU状态层的加载 / 存储、publish_loaded_prefix、  状态，两条腿对齐到同一个边界；offload保存期间block不能释放
#   deallocate_partial等
#   register_received_prefix、record_remote_store    PD分离：decode端把传过来的prompt block登记进前缀缓存          80
#   _make_block_*、take_events、clear_cache          KV事件：block登记 / 淘汰时对外广播，给KV感知的路由器用       65
#   disown_claimed_prefix、allocate里的兜底          放弃命中改为重算时，不能就地覆写别人正在读的共享block         40+
#   Sequence.cache_seed（链的种子）                   多模态：同样的文本 + 不同图片不能互相命中                     —
#   （行数按上游版本粗算，含英文注释；__init__另有约340行，大半是英文注释）


# 【分支】KV事件（本批distributed/kv_events.py；与vLLM的kv_events线格式兼容）：block被登记hash（BlockStored）、被淘汰
#   （BlockRemoved）、索引被清空（AllBlocksCleared）时记一条事件，Scheduler.publish_kv_events每步把它们发给外部的
#   KV感知路由器（只由ATOM_KV_EVENTS_*环境变量开启，见config.py KVEventsConfig上方的说明）。下面三个函数只是构造事件
# 调用方：hash_blocks / publish_loaded_prefix（medium=GPU）、record_remote_store（medium=REMOTE，PD传过来的block）
def _make_block_stored(
    hashes: list[int],
    tokens: list[int],
    parent: int | None,
    block_size: int,
    medium: str = MEDIUM_GPU,
    token_offset: int | None = None,
) -> BlockStored:
    """Construct a BlockStored event from a coalesced run of new blocks.

    `token_offset` is the sequence position of the first token of the run's
    first block, so consumers can map block i to
    `[token_offset + i*block_size, token_offset + (i+1)*block_size)`.
    `block_size` here is the hash block size (block_size * dcp_world_size),
    the span of one block-table entry in global tokens, so the offset must be
    computed in the same unit.
    """
    # 【译】用一段连续的新block构造一个BlockStored事件。
    #
    #   token_offset 是这段第一个block的第一个token在序列中的位置，这样消费者可以把第i个block映射到
    #   [token_offset + i*block_size, token_offset + (i+1)*block_size)。这里的 block_size 是hash block大小
    #   （block_size * dcp_world_size），即一个block table条目跨越的全局token数，所以offset也必须用同一个单位计算。
    # 例（-dcp 4、block_size 16 → hash block 64）：登记第3~5个条目时 token_offset = 3 × 64 = 192，三个block分别是
    #   [192, 256)、[256, 320)、[320, 384)
    # A list, not the `array("i")` the publish paths carry: the event is
    # msgpack-encoded and msgspec has no encoding for an array. The publisher
    # counts encode failures rather than raising, so an array here takes the
    # event stream down without stopping anything.
    # 【译】用list而不是发布路径携带的 array("i")：事件要经msgpack编码，而msgspec没法编码array。发布器只统计编码失败、
    #   不抛异常，所以这里混进一个array会让事件流悄无声息地断掉，其他一切照常运行。
    assert isinstance(
        tokens, list
    ), f"BlockStored.token_ids must be a list, got {type(tokens).__name__}"
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=block_size,
        medium=medium,
        token_offset=token_offset,
    )


def _make_block_removed(hashes: list[int]) -> BlockRemoved:
    return BlockRemoved(block_hashes=hashes, medium=MEDIUM_GPU)


def _make_all_cleared() -> AllBlocksCleared:
    return AllBlocksCleared()


# 【分支】KV offload联合加载（joint load）的边界，一个不可变的值（第9批；什么是联合加载见下面_joint_kv_boundary上方）
# @dataclass(frozen=True)：自动生成__init__等方法，且实例创建后字段不能再赋值（像元组一样不可变）——
#   _joint_kv_boundary只"算"出这个值，要不要"提交"到序列上由调用方决定（_commit_joint_boundary）
@dataclass(frozen=True)
class JointBoundaryDecision:
    """The outcome of the joint-boundary computation, as a value.

    `_joint_kv_boundary` is a pure function of the prompt and the current pool
    state: it reads the seq's KV-leg inputs and the caches, computes where (if
    anywhere) both legs can jointly resume, and returns this object WITHOUT
    touching the seq or any funnel counter. The caller decides whether to
    commit it -- a fit probe discards it, a real admission applies it via
    `_commit_joint_boundary`. That is what removed the `record` flag that used
    to thread down into the boundary computation: computing and recording are
    now two separate steps in two different places.

    `skip_reason is None` marks a real boundary and the four token/hash fields
    carry it; otherwise `skip_reason` names why no joint boundary was found and
    the other fields are unset. `state_from_hbm` tells the commit which tier the
    STATE leg came from (HBM image vs. an image-sized H2D), decided here because
    it is a read of the same cache state the boundary was computed against.
    """

    # 【译】联合边界计算的结果，作为一个值。
    #
    #   _joint_kv_boundary 是prompt和当前池状态的纯函数：它读取seq的KV腿输入和各个缓存，算出两条腿可以在哪里
    #   （如果有的话）共同恢复，然后返回这个对象，不碰seq，也不动任何漏斗计数器。是否提交由调用方决定——试探性的
    #   适配探测会丢弃它，真正的准入则通过 _commit_joint_boundary 应用它。正因如此，过去一路穿进边界计算的 record
    #   标志才被删掉：计算和记录现在是两个独立的步骤，发生在两个不同的地方。
    #
    #   skip_reason is None 表示一个真实的边界，由四个token / hash字段携带；否则 skip_reason 说明为什么没找到联合边界，
    #   其他字段不设置。state_from_hbm 告诉提交方状态腿来自哪一层（HBM镜像，还是一次镜像大小的H2D），在这里决定，
    #   因为它读的正是计算边界时所用的那份缓存状态。
    # 字段（单位都是全局token；图见sequence.py OffloadJointRecord类上方）：
    #   boundary_tokens / boundary_hash  两条腿共同瞄准的边界B，及B处最后一个hash block的链式hash（状态腿按它取checkpoint）
    #   kv_tokens                        KV腿实际传到哪：覆盖B的那个LMCache chunk的末尾（≥ B）
    #   claim_tokens                     allocate直接从HBM前缀缓存认领到哪（KV腿从这里开始取）

    boundary_tokens: int = 0
    boundary_hash: int = -1
    kv_tokens: int = 0
    claim_tokens: int = 0
    state_from_hbm: bool = False
    skip_reason: str | None = None


# Evicted or remapped hashes kept for `_CachedWalk` revalidation; a memo
# older than this many is re-walked.
# 【译】为 _CachedWalk 的重新校验而保留的、被淘汰或被重映射的hash；比这么多条更旧的memo会重新走一遍。
# 即_recent_index_changes这个环形队列的长度：两次校验之间索引变动超过8192次，就不逐条比对了，直接重走
_RECENT_INDEX_CHANGES = 8192


# ★ 投机解码（MTP / EAGLE）的草稿模型要往len(seq)之后多写几个位置的KV（scheduler.py文件开头【前置④】讲了投机解码本身）
#   调用方：Scheduler.__init__（self.draft_lookahead = draft_kv_lookahead(self.mtp_k)，再写进block_manager.draft_lookahead）
def draft_kv_lookahead(num_speculative_tokens: int) -> int:
    """KV positions past `len(seq)` that one forward's MTP draft round writes.

    `EagleProposer.propose` runs the draft `k = num_speculative_tokens` times
    per forward. Step 0 writes the target's own slots; each of the `k - 1`
    mid-steps after it bumps `context_lens` and writes one token at
    `context_lens - 1`. So a forward whose target context ends at `C` writes
    draft KV through position `C + k - 2`, and `C <= len(seq)` at schedule time
    (prefill: `C` is the chunk end; decode: `C = len(seq) - num_rejected`, the
    placeholders the scheduler appended assume full acceptance). The draft
    therefore needs `k - 1` positions past `len(seq)`.

    A position past the row's allocated blocks reads its block id from the
    zeroed tail of the block table, so the write lands in physical block 0 --
    another request's KV, or the units of a PAGE state checkpoint.
    """
    # 【译】一次forward的MTP草稿轮会写到len(seq)之后的多少个KV位置。
    #
    #   EagleProposer.propose 每次forward把草稿跑 k = num_speculative_tokens 次。第0步写目标模型自己的那些槽位；
    #   之后的 k − 1 个中间步，每步把 context_lens 加1，并在 context_lens − 1 处写一个token。所以一次目标上下文止于C的
    #   forward，草稿KV会写到位置 C + k − 2；而调度时 C <= len(seq)（prefill：C是chunk的末尾；decode：
    #   C = len(seq) − num_rejected，调度器追加的占位假设草稿全部被接受）。因此草稿需要len(seq)之后的 k − 1 个位置。
    #
    #   一个超出该行已分配block的位置，会从block table被清零的尾部读到block id（0），于是写入落在物理block 0上——
    #   那是另一个请求的KV，或者某个PAGE状态checkpoint的单元。
    # 图解（k = 3，prefill最后一块止于C = len(seq) = 1008，block_size = 16）：
    #   位置：      … 1006 1007 │ 1008 1009
    #   目标模型：  ■    ■      │                  第0步草稿也写在这些位置（与目标模型同一槽位）
    #   草稿中间步1：           │ ■                context_lens = 1009 → 写1008
    #   草稿中间步2：           │      ■           context_lens = 1010 → 写1009 = C + k − 2
    #   1008个token正好占满63个block（0~1007），1008、1009落在第64个block里：admission要多收1个block
    #   （draft_lookahead_blocks(1008) = 1），may_append(lookahead=2)把它分出来；若不分，block_table[63]读到的是
    #   ScheduledBatch里补零的尾部 → 物理block 0 → 覆写别人的KV
    #   k = 1（只有1个草稿token）时没有中间步，返回0；没开投机解码时num_speculative_tokens为0，也返回0
    # 对比nano-vllm：nano-vllm没有投机解码，没有这一项
    return max(0, int(num_speculative_tokens or 0) - 1)


# Blocks `_hit_walk` hashes per round past the cached prompt chain, doubling
# from MIN to MAX; bounds the hashing wasted past the first miss.
# 【译】_hit_walk 在已缓存的prompt链之外每轮hash的block数，从MIN开始翻倍到MAX；限制第一次未命中之后白算的hash数。
# 例：链没算到的部分按64、128、256、512、1024、1024…块一轮现算
_WALK_HASH_ROUND_MIN = 64
_WALK_HASH_ROUND_MAX = 1024
# Blocks `_hit_walk` looks up per round on the cached chain: the first block
# alone, then 16, growing 4x; bounds the lookups past the first miss to ~4x
# the hit.
# 【译】_hit_walk 在已缓存的链上每轮查表的block数：先只查第一个block，然后16个，每轮×4；把第一次未命中之后多查的次数
#   限制在命中长度的约4倍。
# 例（prompt链已由前端算好，一个128k token的prompt、block 16 → 约8000块）：冷prompt只查1次就知道没命中；
#   全命中时按1、16、64、256、1024、4096…块一轮轮查，几轮就走完
_WALK_LOOKUP_ROUND_MIN = 16


# @dataclass(slots=True)：实例没有__dict__，只能有这几个字段，省内存、属性访问也快一点
# 主干：一次前缀查找（_walk_prefix）的结果——从第0块开始连续命中的那一段：hashes[i]是第i块的链式hash，ids[i]是
#   当前持有它的物理block。读者：_evaluate（命中长度、哪些已被别的序列占用）、can_allocate_memoized（缓存它）、
#   allocate（快速查找下直接认领这些id，认领前_take_admission_walk会核对它们是否还有效）
#   num_tokens / cache_seed / index_version 记下"查找时的样子"，用来判断它是否已过期
@dataclass(slots=True)
class _PrefixWalk:
    """`_walk_prefix`'s contiguous HBM hit (`ids[i]` holds `hashes[i]`): what
    `_evaluate` reads, the delayer memoises and, on the fast walk, `allocate`
    claims (`_take_admission_walk` revalidates the claimed ids)."""

    # 【译】_walk_prefix 在HBM里找到的连续命中（ids[i] 持有 hashes[i]）：_evaluate 读它，delayer缓存它，
    #   在快速查找下 allocate 认领它（_take_admission_walk 会重新校验被认领的id）。

    num_tokens: int
    cache_seed: int
    index_version: int
    hashes: list[int]
    ids: list[int]


# 【分支】can_allocate_memoized的缓存条目（PrefillDelayer、DP成本估算每步对同一批waiting请求反复探测，见第3批）
#   index_clears / index_changes_seen：缓存时clear_cache的次数、索引变动的累计次数；之后变动过的hash里只要有一个在
#   watched里，这次查找的结果就可能变了，必须重走（_walk_still_valid）
#   boundary_hash / boundary_id：第一个没命中的块的hash和它当时查到的id（-1）；它后来被登记了，命中就会变长
@dataclass(slots=True)
class _CachedWalk:
    """A walk memoised for `can_allocate_memoized`. `watched`: every hash
    whose index entry it read (the hit and the first miss, `boundary_hash`,
    None if the hit reaches the last reusable block)."""

    # 【译】为 can_allocate_memoized 缓存的一次查找。watched：它读过的索引项对应的每个hash（命中的块和第一个未命中的块
    #   boundary_hash；如果命中一直延伸到最后一个可复用的块，boundary_hash为None）。

    walk: _PrefixWalk
    index_clears: int
    index_changes_seen: int
    watched: set[int]
    boundary_hash: int | None
    boundary_id: int


# 【主干】BlockManager：每个DP rank的EngineCore里一个，由Scheduler.__init__构造（scheduler.py "④ 队列、BlockManager与指标"）
#   只被EngineCore的busy_loop线程访问（调度器只在这个线程里跑），所以没有任何锁
# 账本的结构（nano-vllm的四个容器挪进了BlockPool）：
#   self.kv  BlockPool（block_pool.py）
#     blocks[i]           Block(block_id, ref_count, hash, token_ids)  每个物理block的元信息（kv_block.py）
#     _vacant / _cached   空闲block：没hash的按id从小到大先发；还有hash的按释放先后（LRU）后发，发出去那一刻才淘汰
#     _used               正被序列引用的block
#     _hash_to_block_id   前缀缓存索引 {链式hash: block id}
#   self.state                    StateSlotPool：每请求状态槽位（state_pool.py）——【分支】线性注意力 / V4
#   self.paged_state_checkpoints  PagedStateCheckpointCoordinator：借KV池的block存状态checkpoint（page_unit_checkpoint.py）——【分支】
#   self.state_offload            StateOffloadIndex：CPU状态层里有哪些checkpoint（state_offload.py）——【分支】KV offload
#   其余：prompt hash链的缓存、前缀查找的memo、KV事件日志、各种统计计数器
# 对比nano-vllm：nano-vllm的BlockManager自己持有blocks / free_block_ids / used_block_ids / hash_to_block_id；
#   ATOM把它们挪进BlockPool，本类只管"一条序列要哪些block、能命中多少、状态从哪接上"这一层
class BlockManager:
    def __init__(
        self,
        config: Config,
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        # 参数：config是EngineCore那份回填过num_kvcache_blocks、pool_entries的Config；state_runtime描述每请求状态怎么在槽位之间
        #   传递（StateTransfer）、PAGE checkpoint的几何（checkpoint_spec），由EngineCore从get_num_blocks的返回值还原
        #   （StateRuntime.from_wire）后经Scheduler传进来；没有每请求状态的模型用默认值DEFAULT_STATE_RUNTIME
        # 每个block装几个token：默认16；DeepSeek-V4 / V4.1 / GLM-5.3被Config改写成256 / 128 / 64（见scheduler.py文件开头【前置①】）
        block_size = config.kv_cache_block_size
        # KV池的block总数：ModelRunner.get_num_blocks按显存算出、EngineCore回填（Config里初始是-1，所以这里断言 > 0）
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        # 读者：num_pool_blocks、max_pool_tokens、_effective_block_size；调度器的chunk对齐（chunk_alignment(block_manager.block_size)）。
        #   同nano-vllm的block_size
        self.block_size = block_size
        # -dcp（--decode-context-parallel-size，默认1）。读者：num_pool_blocks、max_pool_tokens、_effective_block_size，调度器的日志。
        #   DCP是什么、为什么改变block的含义见下面_effective_block_size上方的【前置：DCP】
        self.dcp_world_size = config.decode_context_parallel_size
        # DCP KV-cache interleave granularity S (1 = token-level round-robin).
        # 【译】DCP下KV cache的交错粒度S（1 = 按token轮流分给各卡）。
        # 例（-dcp 2）：S = 1时token 0、2、4…在卡0，1、3、5…在卡1；S = 16时token 0~15在卡0、16~31在卡1、32~47又回到卡0……
        #   只有num_pool_blocks读它（传给get_dcp_local_seq_lens）；config里没有dcp_config时取DCPConfig()的默认值1
        self.cp_kv_cache_interleave_size = getattr(
            config, "dcp_config", DCPConfig()
        ).interleave_size
        # dcp_rank is always 0 here: BlockManager runs only on the scheduler
        # (rank 0). DCP rank is used only to compute local token counts for
        # memory reservation; the actual per-rank routing is done in the workers.
        # 【译】这里的dcp_rank恒为0：BlockManager只在调度器（rank 0）上运行。DCP rank只用来为显存预留计算本地token数；
        #   真正按rank分发是在各worker里做的。
        # 为什么按rank 0算就够：get_dcp_local_seq_lens把除不尽的余数从低rank往高rank分，rank 0分到的token总是最多。
        #   例：1001个token、-dcp 4、S = 1 → rank 0本地251个，rank 1~3各250个；按251记账，每张卡都放得下
        self.dcp_rank = 0
        # Prefix-cache hashing / reuse granularity: under DCP one block_table
        # entry maps to a virtual block of `block_size * dcp_world_size` global
        # tokens (see _hash_block_size). == block_size when DCP is off.
        # 【译】前缀缓存做hash / 复用的粒度：开DCP时一个block_table条目对应一个"虚拟block"，覆盖 block_size * dcp_world_size
        #   个全局token（见_hash_block_size）。不开DCP时等于block_size。
        # 例：block 16、-dcp 4 → 64。读者：本类所有按hash block切分的地方（_n_hash_blocks、_hash_block_tokens、hash_blocks、
        #   链式hash）、StateSlotPool（下面构造时传入）、下面checkpoint间隔的对齐；外部有调度器（命中块数 × hash_block_size =
        #   命中token数）和offload的native_state_scheduler。对比nano-vllm：nano-vllm只有一个block_size
        self.hash_block_size = hash_block_size_for(self.block_size, self.dcp_world_size)
        # `draft_kv_lookahead`: positions past `len(seq)` a forward's draft
        # round writes. Set by the scheduler, which knows whether this engine
        # drafts; admission charges for it and `may_append` allocates it.
        # 【译】draft_kv_lookahead：一次forward的草稿轮写到len(seq)之后的位置数。由调度器设置（它知道本引擎做不做草稿）；
        #   准入时按它收费，may_append按它分配。
        # 默认0（不开投机解码）；Scheduler.__init__设成draft_kv_lookahead(mtp_k)，例mtp_k = 3 → 2。读者：draft_lookahead_blocks
        #   （准入和max_pool_tokens多收的block）、draft_writes_allocated；may_append的lookahead参数由调度器传同一个值
        self.draft_lookahead = 0
        # 【分支】DeepSeek-V4.1 decoder侧SWA有界重放（第8批）：完成prompt的那次forward至少要算这么多行（Config.__post_init__
        #   对V4.1设为hf_config.sliding_window，其他模型0）。读者：replay_start、_record_checkpoint_end
        self.min_final_prefill_rows = getattr(config, "min_final_prefill_rows", 0)
        # 【分支】DeepSeek-V4有界重放（第8批）：ATOM_DSV4_STATE_REPLAY打开且条件满足时Config设成窗口的感受野，否则0；
        #   表示命中点之下要重算多少个token来重建状态（原理见sequence.py Replay类上方科普④）。读者：_replay_hit、allocate、replay_start
        self.state_replay_tokens = getattr(config, "state_replay_tokens", 0)
        # --enable-prefix-caching（默认True）。关掉后can_allocate只检查放不放得下、恒返回0，hash_blocks直接返回——
        #   相当于nano-vllm不做前缀缓存。外部读者有调度器和pp_engine_core
        self.enable_prefix_caching = config.enable_prefix_caching
        # Each seq's prompt-block hash chain, as `(cache_seed, chain)`
        # (`_prompt_hashes`). Content hashes only: pool hits and resource fit
        # are always rechecked. Weak keys keep this scheduler-side cache out of
        # request serialization.
        # 【译】每条seq的prompt block hash链，形如 (cache_seed, chain)（见_prompt_hashes）。只存内容hash：池里有没有命中、
        #   资源够不够，每次都重新检查。用弱引用作key，让这份调度器侧的缓存不进入请求的序列化。
        # WeakKeyDictionary：key是弱引用的dict——Sequence对象被回收时条目自动消失，不用手动删；缓存挂在BlockManager上而不是
        #   Sequence的属性上，pickle一个Sequence时就不会带上它。chain只含prompt里写满的block（full_prompt_blocks =
        #   num_prompt_tokens // hash_block_size个），这一段永远不会变。读者：_prompt_hashes、hash_blocks
        # 对比nano-vllm：nano-vllm在can_allocate和allocate里各现算一遍hash；一个128k token的prompt约有8000个block，
        #   排在满池子后面的请求每步都被探测一次，现算就很贵
        self._prompt_hash_chains: WeakKeyDictionary[Sequence, tuple[int, list[int]]] = (
            WeakKeyDictionary()
        )
        # `_adopt_shipped_chain` outcomes; rejections should stay at 0.
        # 【译】_adopt_shipped_chain 的结果计数；拒绝数应保持为0。
        # 读者：EngineStats的"[Prompt Chains] frontend-hashed: adopted …, rejected …"日志行。total_evicted_blocks：
        #   _record_evicted每被回调一次加1，pool_occupancy把它作为evicted_total报给EngineStats的"[Cache Pool]"日志行
        self.shipped_chains_adopted: int = 0
        self.shipped_chains_rejected: int = 0
        self.total_evicted_blocks: int = 0
        # `can_allocate_memoized`'s memo, validated against the hashes
        # evicted or remapped since (`_note_index_change`) and `clear_cache`.
        # 【译】can_allocate_memoized 的memo，用此后被淘汰或被重映射的hash（_note_index_change）以及 clear_cache 来校验。
        # 【分支】PrefillDelayer（调度器的_probe_local_prefill_work、_can_admit_head_prefill）和DP成本估算
        #   （_estimate_prefill_chunk_cost）每步都要问一遍"waiting里这条能命中多少"；索引没动时答案不变，就复用上次的查找
        self._cached_walks: WeakKeyDictionary[Sequence, _CachedWalk] = (
            WeakKeyDictionary()
        )
        # 下面三个给_walk_still_valid判断memo是否过期：最近被淘汰 / 重映射的hash（deque(maxlen=N)：满了以后从另一头自动
        #   丢掉最旧的，这里N = 8192）、索引变动的累计次数、clear_cache的次数。walk_memo：ATOM_PREFIX_WALK_MEMO（默认1），
        #   关掉则can_allocate_memoized每次都重走
        self._recent_index_changes: deque[int] = deque(maxlen=_RECENT_INDEX_CHANGES)
        self._index_changes = 0
        self._index_clears = 0
        self.walk_memo: bool = envs.ATOM_PREFIX_WALK_MEMO
        # `_hit_walk`, whose ids `allocate` claims; off = per-block walks.
        # 【译】_hit_walk（allocate会认领它找到的id）；关闭 = 逐块查找。
        # ATOM_PREFIX_FAST_WALK（默认1）。逐块查找就是nano-vllm的做法：每块查一次表、比对一次token；快速查找成批查表，
        #   只比对最后一个命中块的token（为什么够用见_hit_walk）
        self.fast_prefix_walk: bool = envs.ATOM_PREFIX_FAST_WALK
        # Each seq's last fast walk, for its `allocate` to claim.
        # 【译】每条seq最近一次快速查找的结果，留给它的 allocate 认领。
        # 这样allocate不用再走一遍hash链；认领前由_take_admission_walk核对这些id是否仍然有效
        self._admission_walks: WeakKeyDictionary[Sequence, _PrefixWalk] = (
            WeakKeyDictionary()
        )

        # 【分支】KV事件（见文件开头_make_block_stored上方）：config.kv_events_config由ATOM_KV_EVENTS_*环境变量构造
        #   （KVEventsConfig.from_env），enable为真才记录。_event_log：开启时是待发事件的列表，关闭时为None——各处用
        #   "_event_log is not None"判断要不要记；Scheduler.publish_kv_events每步经take_events取空它
        kv_events = getattr(config, "kv_events_config", None)
        self._events_enabled: bool = bool(kv_events and kv_events.enable)
        self._event_log: list[KVCacheEvent] | None = (
            [] if self._events_enabled else None
        )
        # The compressed KV blocks. Same class the sliding window uses for its
        # own index space — hash eviction has to happen at the same moment in
        # both or a prefix hit could be honoured by one pool and not the other.
        # 【译】压缩后的KV block。和滑动窗口自己的索引空间用的是同一个类——hash淘汰必须在两边同一时刻发生，否则一个前缀命中
        #   可能被一个池承认、另一个池不承认。
        # ★ 主干：整个KV池的账本（BlockPool，block_pool.py）。"compressed"沿用DeepSeek-V4的叫法（V4缓存的是压缩后的KV行）；
        #   对普通模型它就是全部KV block。
        #   观察：这段英文注释已过时——全仓库只有这一处构造BlockPool；block_pool.py的类说明写明滑动窗口现在是每请求的环、
        #   和压缩器状态共用一个state槽位（v4_pool_geometry.py），下面state_caches上方的英文注释也是这么说的
        # 参数：on_evict = _record_evicted（某个hash离开索引时回调：block被拿去装新内容，或publish把它改登记成新hash）；
        #   cache_policy = ATOM_PREFIX_CACHE_POLICY（默认"lru"；"slru"：被命中复用过的block释放后进受保护队列，最多占池子的
        #   protected_ratio = ATOM_PREFIX_CACHE_PROTECTED_RATIO（默认0.5），一串一次性的前缀冲不掉热前缀）；
        #   on_remap = _note_index_change（某个hash改指向另一个block时回调）
        # 读者：本类几乎每个函数；调度器读kv.num_blocks、num_used、block(id)；engine_utility
        # 对比nano-vllm：nano-vllm的blocks、free_block_ids、used_block_ids、hash_to_block_id都在这个对象里
        self.kv = BlockPool(
            num_blocks,
            on_evict=self._record_evicted,
            cache_policy=envs.ATOM_PREFIX_CACHE_POLICY,
            protected_ratio=envs.ATOM_PREFIX_CACHE_PROTECTED_RATIO,
            on_remap=self._note_index_change,
        )
        # Per-request cache slot pool. Used by attention types with a
        # stateful per-request buffer (GDN recurrent state, V4 compressor
        # state). The backing tensor is pre-allocated by ModelRunner and
        # excluded from `num_kvcache_blocks` at sizing time, so admission only
        # needs free slot indices from this list.
        #
        # Slots are counted raw, not divided into per-request groups. What one
        # request occupies is a property of the request — `1 + num_spec` while
        # it speculates, 1 otherwise — and a checkpoint occupies exactly one
        # whatever the model does, so there is no single width to divide by.
        # `state_slots_per_req` is what a live request asks for.
        # 【译】每请求缓存的槽位池。供带有每请求有状态缓冲区的注意力类型使用（GDN递归状态、V4压缩器状态）。底层张量由
        #   ModelRunner预先分配，计算规模时已从 num_kvcache_blocks 中扣除，所以准入只需要从这个列表里拿到空闲的槽位编号。
        #
        #   槽位按个数计，不分成每请求一组。一个请求占几个是请求自己的属性——投机时是 1 + num_spec，否则是1——而一个
        #   checkpoint不管模型怎样都恰好占1个，所以没有一个统一的宽度可以拿来除。state_slots_per_req 是一个在途请求要的个数。
        # 【分支】每请求状态（第4、8批；是什么见文件开头前置概念）：pool_entries / pool_entries_per_req是ModelRunner.get_num_blocks
        #   算出的"各条目类有多少条目 / 每请求要几条"，经EngineCore放进config；不带状态的模型里没有"state"这个键，下面取到0
        pool_entries: dict = getattr(config, "pool_entries", None) or {}
        pool_per_req: dict = getattr(config, "pool_entries_per_req", None) or {}
        # Total capacity, kept so callers can tell "all slots busy" (transient)
        # from "no slots were ever created" (permanent).
        # 【译】总容量，保留它是为了让调用方区分"槽位全忙"（暂时的）和"根本没建过槽位"（永久的）。
        # 读者：调度器的_unschedulable_reason（总数 < 每请求要的个数 → 永远放不下，直接拒绝）。state_slots_per_req：GDN开MTP且
        #   不用ReplaySSM时为1 + k，其余为1（"or 1"：取到0也按1算）；读者_evaluate（准入先看空闲槽位够不够）、
        #   _attach_state_slots、调度器
        self.num_state_slots = int(pool_entries.get(STATE_SLOT_CLASS, 0))
        self.state_slots_per_req = int(pool_per_req.get(STATE_SLOT_CLASS, 1)) or 1
        # Tokens between rungs of the checkpoint ladder, shared by every
        # Pool.STATE class (--state-checkpoint-interval-tokens).
        #
        # Three regimes, and the sign carries the distinction:
        #   >0  ladder on, a rung every N tokens.
        #    0  state checkpointing off entirely. Nothing is kept, anywhere.
        #   -1  ladder off, checkpointing on: the demand rung and the prompt-end
        #       anchor still place checkpoints, the interval grid does not.
        #
        # -1 rather than reusing 0 for this, even though "no interval" reads
        # like zero: 0 is the documented off switch and it is also *reachable
        # by accident* — the snap below rounds an off-grid interval down and
        # can land on 0, so a `--block-size` typo currently fails safe. Giving 0
        # a second meaning would make that typo silently enable a caching policy
        # instead of disabling one.
        # 【译】checkpoint阶梯相邻两级之间的token数，所有 Pool.STATE 类共用（--state-checkpoint-interval-tokens）。
        #
        #   三种状态，用符号区分：
        #     >0  阶梯开启，每N个token一级。
        #      0  完全关闭状态checkpoint，哪里都不保存。
        #     -1  阶梯关闭、checkpoint开启：按需的那一级和prompt末尾的锚点仍会放checkpoint，按间隔的网格不放。
        #
        #   用-1而不是复用0，尽管"没有间隔"读起来像0：0是文档写明的关闭开关，而且它*可能被意外得到*——下面的对齐把不在网格上
        #   的间隔向下取整，可能落到0，所以眼下一个写错的 --block-size 只会安全地失效。给0第二种含义，会让这种笔误悄悄开启
        #   一种缓存策略，而不是关闭它。
        # 【分支】默认8192（Config / EngineArgs）；max(-1, …)把小于-1的值都当-1。读者：checkpoint_limit、checkpoint_cut、
        #   checkpointers_at、_record_checkpoint_demand、_record_checkpoint_end、midstep_positions，以及调度器
        self.state_checkpoint_interval_tokens = max(
            -1, int(getattr(config, "state_checkpoint_interval_tokens", 0) or 0)
        )
        # See ATOM_STATE_CHECKPOINT_END_MARGIN and `_record_checkpoint_end`.
        # 【译】见 ATOM_STATE_CHECKPOINT_END_MARGIN 和 _record_checkpoint_end。
        # 默认0（envs.py）；只有_record_checkpoint_end读它
        self.state_checkpoint_end_margin = envs.ATOM_STATE_CHECKPOINT_END_MARGIN
        # Independent of the interval, because the demand rung is not part of
        # the grid: it is the one placement a *refused* hit makes for itself.
        # See `_record_checkpoint_demand` for what turning it off is testing.
        #
        # ATOM_STATE_CHECKPOINT_DEMAND wins over the config field when it is
        # exported, so the policy can be flipped for one run without editing a
        # launch script: =0 forces the rung off, =1 forces it on. Unset leaves
        # --state-checkpoint-demand in charge, so an unexported variable costs
        # nothing.
        # 【译】与间隔无关，因为按需的那一级不属于网格：它是一个*被拒绝*的命中为自己做的唯一一种放置。关闭它是在测试什么，
        #   见 _record_checkpoint_demand。
        #
        #   导出了 ATOM_STATE_CHECKPOINT_DEMAND 时它优先于config字段，这样不改启动脚本就能在某一次运行里切换策略：=0强制关闭
        #   这一级，=1强制开启。不设置则由 --state-checkpoint-demand 决定，所以一个没导出的变量没有任何代价。
        # 按需级：一次命中因为缺checkpoint被状态门控截短时，在它本来能命中的位置为下一个同前缀请求补放一级。config默认True；
        #   envs.is_set看的是"导出且非空"。读者：_record_checkpoint_demand
        self.state_checkpoint_demand = bool(
            getattr(config, "state_checkpoint_demand", True)
        )
        if envs.is_set("ATOM_STATE_CHECKPOINT_DEMAND"):
            self.state_checkpoint_demand = envs.ATOM_STATE_CHECKPOINT_DEMAND
        if not self.state_checkpoint_demand:
            logger.info(
                "[State Cache] demand rung disabled: the prompt-end anchor is "
                "the only checkpoint placement."
            )
        # checkpoint_spec不为None ⇔ 后端声明了PAGE方式的状态传输（StateTransfer.copy）：checkpoint镜像存进KV池的block。
        #   v0.1.8里用PAGE的：Qwen3-Next / Qwen3.5的GDN和Kimi的KDA（PP>1或开RapidServe时退回fork）、DeepSeek-V4、
        #   关掉encoder侧有界重放的V4.1（默认开着，走replay）；Qwen4-exp等其余带状态的后端走fork（checkpoint就是一个状态槽位）
        checkpoint_spec = state_runtime.checkpoint_spec
        # Read here rather than beside `state_offload` below, because the
        # coordinator has to know at construction whether anything can carry a
        # store: without a sink it must nominate nothing, since a pin nobody
        # releases holds a whole image out of the pool forever.
        # 【译】在这里读，而不是放在下面 state_offload 旁边，因为协调器在构造时就必须知道有没有东西能承接一次store：
        #   没有sink时它什么都不能提名，因为一个没人释放的pin会把一整个镜像永远扣在池子外面。
        from atom.model_engine.state_offload import (
            state_tier_capability,
            state_tier_chunk_tokens,
            state_tier_cpu_bytes,
        )

        # A capability derived from the whole config, not the connector's name.
        # The name only says which class is constructed; whether that class
        # builds a `StateOffloadTier` also depends on the layout it resolves,
        # the pipeline depth, and its role. Installing an index against a
        # worker that will refuse the tier is what left stores emitted with
        # nowhere to go.
        # 【译】一个从整个config推导出的能力描述，而不是看connector的名字。名字只说明构造哪个类；那个类会不会建出
        #   StateOffloadTier，还取决于它解析出的布局、流水线深度和它的角色。在一个会拒绝这一层的worker面前装上索引，
        #   正是之前store被发出却无处可去的原因。
        # 【分支】KV offload的CPU状态层（第9批；KV offload是什么见文件开头前置概念）：state_tier_capability(config)只看
        #   --kv-transfer-config等配置，回答"worker会不会建CPU状态层、能存吗、能取吗"。hosts_state_tier为False（绝大多数部署）
        #   时下面所有offload字段都不建
        self.state_tier_capability = state_tier_capability(config)
        kv_offload_enabled = self.state_tier_capability.hosts_state_tier
        # Only worth saying for a model that has state to offload. Every dense
        # engine would otherwise announce at startup that a tier it could never
        # use is off, which is true and useless.
        # 【译】只对有状态可卸载的模型才值得说。否则每个dense引擎启动时都会宣布一个它永远用不上的层是关着的——这是真的，但没用。
        if (
            not kv_offload_enabled
            and checkpoint_spec is not None
            and self.state_tier_capability.reason
        ):
            logger.info(
                "[State Cache] CPU state tier off: %s.",
                self.state_tier_capability.reason,
            )
        # 【分支】PAGE状态checkpoint的协调器（page_unit_checkpoint.py）：checkpoint镜像不占状态槽位，而是借KV池的block存。
        #   例（kimi_mla_gdn_attn.py的说明）：一个KDA槽位53.6 MiB，按PAGE存是127个普通KV block、约占KV池的0.112%，
        #   和普通KV从同一个空闲表里取、按同一个LRU淘汰。enabled = 开了前缀缓存且确实有状态槽位（关闭时对象仍在，只是不登记
        #   checkpoint）；readable_midstep：后端能否在forward中途取快照（GDN在ATOM_GDN_STATE_MIDSTEP=1时能）
        self.paged_state_checkpoints: PagedStateCheckpointCoordinator | None = None
        if checkpoint_spec is not None:
            enabled = self.enable_prefix_caching and self.num_state_slots > 0
            self.paged_state_checkpoints = PagedStateCheckpointCoordinator(
                self.kv,
                checkpoint_spec,
                enabled=enabled,
                readable_midstep=state_runtime.transfer.readable_midstep,
            )
        # The rolling state class: per-request slots plus a content index over
        # the free ones. A checkpoint IS a free slot whose content is still
        # valid, so it holds no capacity of its own and never blocks admission.
        # 【译】滚动状态类：每请求的槽位，加上一个建在空闲槽位之上的内容索引。一个checkpoint*就是*一个内容仍然有效的空闲槽位，
        #   所以它自己不占容量，也永远不会挡住准入。
        # self.state（StateSlotPool，state_pool.py）：在途请求的状态槽位总是从这里分。transfer：用PAGE时传StateTransfer.none()
        #   ——checkpoint由上面的协调器管，槽位池只分活跃槽位；否则用后端声明的方式（fork：checkpoint就是一个被释放、但内容仍
        #   登记在索引里的槽位，即上面英文注释说的情况）。enabled跟随enable_prefix_caching（槽位数为0时StateSlotPool自己也关）。
        #   读者：_evaluate（has_free）、allocate / deallocate、调度器（state.transfer、min_fork_tokens等）
        # 对比nano-vllm：nano-vllm没有每请求状态
        self.state = StateSlotPool(
            self.num_state_slots,
            transfer=(
                StateTransfer.none()
                if self.paged_state_checkpoints is not None
                else state_runtime.transfer
            ),
            hash_block_size=self.hash_block_size,
            enabled=self.enable_prefix_caching,
        )
        # checkpoint归谁管：有PAGE协调器就是它，否则是槽位池自己。读者：_record_evicted（KV block的hash被淘汰时同步删掉同一hash
        #   的checkpoint）、clear_cache、midstep相关函数，以及下面的state_caches
        self._state_checkpoint_cache: StateCheckpointCache = (
            self.paged_state_checkpoints or self.state
        )
        # Where a readable backend can snapshot inside a forward; see
        # `midstep_target_ok`. Zero for a backend that cannot.
        # 【译】一个可读的后端能在forward内部的哪些位置取快照；见 midstep_target_ok。不能这么做的后端为0。
        # 例：GDN开ATOM_GDN_STATE_MIDSTEP=1时midstep_grid = 64（chunk kernel每64个token物化一次中间状态h，从forward的起点算）、
        #   midstep_min_offset = conv窗口长度；其余后端都是0。读者：midstep_positions、midstep_anchor
        self.midstep_grid = state_runtime.transfer.midstep_grid
        self.midstep_min_offset = state_runtime.transfer.midstep_min_offset
        if self.paged_state_checkpoints is not None:
            # A PAGE midstep checkpoint is copied into a scratch slot from
            # this pool, then stored from there, within the copy budget the
            # runner's descriptor staging is sized to: one definition for both.
            # 【译】一个PAGE midstep checkpoint先拷进这个池的一个scratch槽位，再从那里存出去，拷贝个数不超过runner描述符暂存区
            #   所容纳的预算：两边用同一个定义（checkpoint_copy_ops_per_batch(max_num_seqs)，page_unit_checkpoint.py）。
            self.paged_state_checkpoints.attach_midstep_scratch(
                self.state,
                checkpoint_copy_ops_per_batch(getattr(config, "max_num_seqs", 0)),
            )
        # A checkpoint is filed under the content hash of the last block it
        # covers, so a rung that isn't a hash-block boundary can never be looked
        # up — the ladder would checkpoint into a void. The interval defaults to
        # 8192 while `hash_block_size` follows `--block-size` and
        # `--decode-context-parallel-size`, so ordinary flag combinations
        # (`--block-size 100`, dcp 3) land off the grid through no choice of the
        # user's. Snap down to the grid and say so, rather than refusing to
        # start — and rather than asserting, which `python -O` would drop and
        # leave the ladder cutting prefill chunks onto rungs nothing can reach.
        # `> 0` rather than truthy: -1 is not an interval to snap onto the grid,
        # it is the absence of one. Snapping it would give -4 and a warning
        # about a flag the user set deliberately.
        # 【译】一个checkpoint按它覆盖的最后一个block的内容hash登记，所以不在hash block边界上的一级永远查不到——阶梯就会把
        #   checkpoint存进虚空。间隔默认8192，而 hash_block_size 跟随 --block-size 和 --decode-context-parallel-size，所以普通的
        #   参数组合（--block-size 100、dcp 3）会在用户毫不知情时落到网格之外。向下对齐到网格并告知用户，而不是拒绝启动——也不用
        #   assert，因为python -O会去掉它，留下阶梯把prefill chunk切在永远够不着的级上。
        #   用 > 0 而不是真值判断：-1不是一个要对齐到网格的间隔，而是"没有间隔"。对齐它会得到-4，还会对一个用户故意设的参数报警。
        # 例：--block-size 100 → 8192 // 100 × 100 = 8100；block 16、-dcp 3 → hash block 48 → 8192 // 48 × 48 = 8160；
        #   hash block大于8192时对齐成0，等于关闭阶梯（警告里写"off (0)"）
        if (
            self.state.enabled
            and self.state_checkpoint_interval_tokens > 0
            and self.state_checkpoint_interval_tokens % self.hash_block_size
        ):
            snapped = (
                self.state_checkpoint_interval_tokens // self.hash_block_size
            ) * self.hash_block_size
            logger.warning(
                f"--state-checkpoint-interval-tokens="
                f"{self.state_checkpoint_interval_tokens} is not a multiple of "
                f"the prefix-cache hash block size {self.hash_block_size}; "
                f"snapping to {snapped or 'off (0)'}."
            )
            self.state_checkpoint_interval_tokens = snapped

        # Every Pool.STATE class. A tuple of one today — the sliding window
        # used to be the second member, back when it was a content-addressed
        # block pool that could gate a hit. It is now a per-request ring carried
        # by the state checkpoint, so it has nothing to say about hit length.
        # Kept plural because GDN's recurrent state is a second member the
        # moment it stops forking (see the state-cache protocol).
        # 【译】所有 Pool.STATE 类。现在是只有一个元素的元组——滑动窗口曾是第二个成员，那时它还是一个能对命中设门槛的内容寻址
        #   block池。现在它是一个由状态checkpoint携带的每请求环，对命中长度没有发言权。保持复数形式，是因为GDN的递归状态
        #   一旦不再fork，就会成为第二个成员（见state_cache.py的协议）。
        # 读者：_gated_hit（命中门控）、checkpointers_at / checkpoint_cut / checkpoint_limit（在哪存checkpoint）、
        #   _record_checkpoint_end、state_checkpoint_fates，以及调度器。不带状态的序列对每个类都"不适用"（applies为False），门控原样放行
        self.state_caches: tuple[StateCache, ...] = (self._state_checkpoint_cache,)

        # Class names already warned about in `state_checkpoint_fates`. See
        # there for why the warning latches.
        # 【译】已在 state_checkpoint_fates 里警告过的类名。为什么警告只发一次见那里。
        self._warned_no_checkpoint_fates: set[str] = set()
        from atom.model_engine.state_offload import StateOffloadIndex

        # Joint boundaries, split by what the state leg cost: `hbm` forked a
        # resident checkpoint, `tier` paid an entry-sized H2D and a park. Almost
        # all `tier` means the state pool is too small for the concurrency.
        # 【译】联合边界，按状态腿的代价分开计：hbm 是fork了一个驻留的checkpoint，tier 付出了一次条目大小的H2D和一次挂起。
        #   几乎全是 tier，说明状态池相对并发量太小。
        # 【分支】KV offload联合加载的统计（第9批），由_commit_joint_boundary累加；调度器的统计日志、llm_engine的指标读
        #   joint_boundaries、state_hbm_boundaries、state_tier_boundaries、joint_skips
        self.joint_boundaries = 0
        self.state_hbm_boundaries = 0
        # The number that says the tier is doing anything at all. Every other
        # counter here can be non-zero with the CPU tier switched off; this one
        # cannot, which makes it the only honest test of "did this feature run".
        # Suffixed `_boundaries` so an int counter no longer shares the bare name
        # `state_tier` with the state-tier object/module used across this PR
        # (`kimi_k3.state_tier`, `_JointPark`, `_state_tier`). The exported funnel
        # key and the log label keep the short `state_tier` string.
        # 【译】说明这一层到底有没有在干活的那个数。这里其他每个计数器在CPU层关闭时都可能非零；只有这个不会，所以它是
        #   "这个特性跑过没有"唯一可信的检验。
        #   加上 _boundaries 后缀，是为了让一个int计数器不再和本PR里到处用的状态层对象 / 模块（kimi_k3.state_tier、_JointPark、
        #   _state_tier）共用光秃秃的名字 state_tier。导出的漏斗键和日志标签保留短字符串 state_tier。
        self.state_tier_boundaries = 0
        # Admissions whose gated boundary neither tier could produce by the time
        # `allocate` ran. Non-zero is expected under pressure (the CPU index is
        # optimistic, and an HBM checkpoint can be unindexed inside the same
        # pass); a large fraction of `joint_boundaries` means the gate is
        # accepting boundaries that do not survive to attach.
        # 【译】门控选中了边界、但到 allocate 运行时两层都给不出来的准入数。有压力时非零是预期的（CPU索引是乐观的，而一个HBM
        #   checkpoint可能在同一轮里被取消索引）；如果它占 joint_boundaries 的比例很大，说明门控接受的边界撑不到attach那一步。
        # 由allocate、_attach_state_slots累加，checkpoint_funnel报出
        self.state_gate_lost_boundary = 0
        # Why the rest got none, keyed by the gate that stopped them.
        # 【译】其余的为什么没得到联合边界，按拦下它们的那道门分别计数。
        # 键就是_no_joint的reason：如"off"（没有CPU状态层；没开offload时每次准入都记这一条）、"lmcache_within_hbm"等
        self.joint_skips: dict[str, int] = {}
        # The LMCache chunk size in tokens, read where the config is rather than
        # off the connector object.
        # 【译】LMCache的chunk大小（token数），从config所在的地方读，而不是从connector对象上读。
        self._joint_chunk_tokens = 0
        # Only when a joint KV load is possible at all. Reading it reaches into
        # LMCache, an optional dependency: probing on an engine that hosts no
        # state tier logged a full `ModuleNotFoundError` traceback at WARNING
        # on every start, for a number that engine has no use for. The
        # capability above is the same gate the tier itself uses.
        # 【译】只在联合KV加载确有可能时才读。读它要伸进LMCache，而LMCache是可选依赖：在一个不托管状态层的引擎上探测，会在
        #   每次启动时以WARNING级别打出完整的 ModuleNotFoundError 回溯，只为一个那个引擎根本用不上的数。上面的能力描述就是
        #   这一层自己用的那道门。
        if kv_offload_enabled:
            self._joint_chunk_tokens = state_tier_chunk_tokens(config)
        # 【分支】StateOffloadIndex：engine侧记录"CPU状态层里存着哪些checkpoint"的索引（state_offload.py）；字节本身在worker
        #   那边的LMCache池里（OFFLOAD_STATE_CPU_SIZE从中划出）。没开offload时为None
        self.state_offload: StateOffloadIndex | None = None
        # (req_id, hash, target_group) admitted this pass and not yet handed to
        # the connector. Kept here rather than in the index because the slot
        # is this object's fact.
        # 【译】本轮已准入、还没交给connector的 (req_id, hash, target_group)。放在这里而不是索引里，因为槽位是这个对象掌握的事实。
        # take_state_loads每步取走、交给connector
        self._state_loads: list[tuple] = []
        # req_id -> [(slot, monotonic stamp), ...], for loads whose request was
        # deallocated before the bytes landed. The slot is off the free list
        # until the report comes back; see `deallocate`. A *list* per id, not a
        # single tuple: `seq.id` is per-request not per-admission, so a
        # preempt/re-admit can park the same id twice while the first load is
        # still in flight. A single-value entry would overwrite -- silently
        # dropping the first slot (leaked forever) and, worse, releasing the
        # wrong slot when the first report arrives (a slot still being written).
        # Append instead and release oldest-first: the worker submits and reports
        # a request's loads in admission order, so the FIFO pop in
        # `settle_state_load`/`abandon_state_load` matches slot to report. The
        # stamp trusts `settle_state_load` to "always come": it does not, if the
        # worker that owed the report crashed or its completion was dropped. Then
        # the slot would sit off the free list forever and the state gate in
        # `can_allocate` would wedge the pool once every slot is stranded, so
        # `reconcile_orphan_load_slots` reclaims it after the same abandon
        # window the store pins use -- the load-side twin of
        # `reclaim_stale_state_store_pins`.
        # 【译】req_id -> [(slot, monotonic时间戳), ...]，记那些字节送达之前请求就被释放了的加载。报告回来之前槽位不在空闲表上；
        #   见 deallocate。每个id对应一个*列表*而不是单个元组：seq.id是按请求而不是按准入分配的，所以抢占 / 重新准入可能在
        #   第一次加载还在途时把同一个id挂起两次。单值条目会被覆盖——悄悄丢掉第一个槽位（永远泄漏），更糟的是，第一份报告到达
        #   时会释放错的槽位（一个仍在被写入的槽位）。所以改为追加，并按最早优先释放：worker按准入顺序提交并报告一个请求的
        #   加载，所以 settle_state_load / abandon_state_load 里的FIFO弹出能让槽位和报告对上。时间戳相信 settle_state_load
        #   "总会到来"：可它不一定——欠这份报告的worker崩溃了，或者完成消息丢了。那时槽位会永远不在空闲表上，而一旦每个槽位
        #   都被困住，can_allocate 里的状态门控就会把池子卡死，所以 reconcile_orphan_load_slots 在store pin用的同一个放弃窗口
        #   之后回收它——它是 reclaim_stale_state_store_pins 在加载侧的孪生。
        # _orphan_load_slots_reclaimed：被reconcile_orphan_load_slots回收的个数，checkpoint_funnel报出
        self._orphan_load_slots: dict[object, list[tuple[int, float]]] = {}
        self._orphan_load_slots_reclaimed: int = 0
        # `kv_offload_enabled` is the whole switch: without a connector to
        # carry a transfer there is nothing beneath the pool, and a hash whose
        # KV left HBM could not be resumed from anyway.
        #
        # The index no longer reaches into `StateSlotPool`. A K3 checkpoint is
        # a PAGE image in the KV pool (#2045), so the spill source is a set of
        # units the coordinator owns, not a slot the pool is about to hand
        # away -- which is what the staging ring existed to rescue.
        # 【译】kv_offload_enabled 就是总开关：没有connector承载传输，池子下面什么都没有，一个KV已离开HBM的hash本来也无从恢复。
        #
        #   这个索引不再伸进 StateSlotPool。K3的checkpoint是KV池里的一个PAGE镜像（#2045），所以溢出的源是协调器拥有的一组
        #   单元，而不是槽位池马上要送出去的一个槽位——暂存环当初就是为了抢救那种槽位才存在的。
        if kv_offload_enabled:
            self.state_offload = StateOffloadIndex(
                can_store=self.state_tier_capability.can_store_state,
                can_load=self.state_tier_capability.can_load_state,
                # The codec stores one `image_bytes` image per entry (the
                # worker turns the tier off unless its entry is exactly that).
                # Not `slot_bytes`: under MTP a slot holds more than an image
                # (DSpark3: 42.45 vs 29.51 MiB), and dividing by it would cap
                # this index a third below the codec, hiding images it holds.
                # 【译】codec每个条目存一个 image_bytes 大小的镜像（除非条目恰好这么大，否则worker会关掉这一层）。不用 slot_bytes：
                #   MTP下一个槽位装的不止一个镜像（DSpark3：42.45 对 29.51 MiB），用它来除会让这个索引比codec的上限少三分之一，
                #   把它实际持有的镜像藏起来。
                # 例：OFFLOAD_STATE_CPU_SIZE默认32 GiB，镜像29.51 MiB时上限 32768 / 29.51 ≈ 1110个条目
                max_cpu_entries=(
                    state_tier_cpu_bytes() // checkpoint_spec.image_bytes
                    if checkpoint_spec is not None
                    else None
                ),
            )
            # Attached rather than passed at construction: the coordinator is
            # built before the switch is read (it needs `checkpoint_spec`,
            # which comes off `state_runtime`). One object, two uses -- the
            # coordinator votes off `hashes` and drains stores into it.
            # 【译】构造后挂上，而不是构造时传入：协调器是在读这个开关之前建的（它需要来自 state_runtime 的 checkpoint_spec）。
            #   一个对象两种用途——协调器按它的 hashes 投票，并把store排进它。
            if self.paged_state_checkpoints is not None:
                self.paged_state_checkpoints.attach_offload(
                    self.state_offload,
                    sink=self.state_tier_capability.can_store_state,
                )
        # The demand funnel: recorded at admission, cut for when a prefill
        # chunk is shortened to land on it, kept when the state pool files it.
        # Counted at all three because a gap between any two is a different
        # bug, and they are indistinguishable in the hit rate alone.
        # 【译】按需漏斗：准入时记recorded，为落在它上面而截短prefill chunk时记cut，状态池登记它时记kept。三处都计数，是因为
        #   任意两处之间的缺口都是不同的bug，而单看命中率根本分不出来。
        # 下面五个计数器的读者：checkpoint_funnel（engine_utility的统计接口）、llm_engine的指标
        self.demands_recorded: int = 0
        self.chunks_cut_for_demand: int = 0
        self.demands_declined_no_room: int = 0
        # The anchor's own cut counter, kept separate from the demand's: the
        # anchor fires on nearly every prompt and would drown the convergence
        # signal `chunks_cut_for_demand` exists to expose.
        # 【译】锚点自己的cut计数，与按需的分开：锚点几乎每个prompt都会触发，会淹没 chunks_cut_for_demand 要暴露的收敛信号。
        self.chunks_cut_for_end: int = 0
        # Ladder, demand or anchor positions a forward covered but could not
        # snapshot even after moving back onto its grid: too close to the
        # forward's start for the conv window, or the forward started off the
        # block grid. Each is a checkpoint the cutting ladder would have kept.
        # 【译】forward覆盖了、但即使挪回它的网格也没法取快照的阶梯 / 按需 / 锚点位置：离forward起点太近、conv窗口不够，
        #   或者forward的起点不在block网格上。每一个都是"截chunk的阶梯"本来会保留下来的checkpoint。
        self.midstep_positions_unreadable: int = 0

    # ★ 主干：一个block的链式hash（= prefix_hash.block_hash；前端预先算hash链用的也是同一个函数）
    #   做法：token id按int64的字节喂给xxh64；prefix ≠ -1时先拼上前一块hash的8个字节（小端序）：
    #     h0 = H(seed, 块0的token)，h1 = H(h0, 块1的token)，……，hi隐含了从开头到第i块的全部内容
    #   第0块的prefix是seq.cache_seed：纯文本请求为-1（什么都不拼），多模态请求是图片等内容的指纹——同样的文本 + 不同的图片，
    #     链从第0块起就不同，不会误命中（sequence.py cache_seed处）
    # @classmethod：第一个参数是类本身，所以不用实例也能调用（BlockManager.compute_hash(...)）
    # 对比nano-vllm：与nano-vllm的compute_hash相同（xxh64，前一块hash的8字节小端序 + token字节）
    @classmethod
    def compute_hash(cls, token_ids: array.array, prefix: int = -1):
        return block_hash(token_ids, prefix)

    def complete_previous_state_batch(self) -> None:
        """Complete state reads and copies issued by the previous batch."""
        # 【译】完成上一个batch发出的状态读取和拷贝。
        # 【分支】每请求状态（第4、8批）：Scheduler._schedule每步一开头调用（RapidServe的调度器也一样）。此时上一个batch的forward
        #   已经发出，它要读的源槽位（fork的源等）可以解除钉住、回到空闲表——下一个拿到它的请求的forward排在同一条stream上、
        #   一定在后面执行；PAGE协调器把上一批发出的checkpoint拷贝标为完成（镜像从COPYING变成READY、登记进索引，之后才能被
        #   恢复），借给midstep的scratch槽位也还回状态池
        self.state.release_pins()
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.complete_previous_batch()

    def take_state_maintenance_ops(self) -> StateMaintenanceOps:
        """Drain state maintenance for the batch being built."""
        # 【译】取出正在构建的这个batch要做的状态维护操作。
        # 【分支】调度器构造ScheduledBatch时调用（batch为空时不取，留给下一批），结果放进batch.state_maintenance_ops；worker在准备
        #   这个batch的注意力元数据之前先执行它们（attentions/backends.py的relocate_state_slots、execute_paged_state_copies）。
        #   三类：relocations（池子边界移动时槽位搬家，src → dst）、checkpoint_stores（活跃槽位 → PAGE单元：存一个checkpoint）、
        #   checkpoint_restores（PAGE单元 → 新请求的槽位：从checkpoint恢复）
        relocations = self.state.take_relocations()
        stores = restores = ()
        if self.paged_state_checkpoints is not None:
            stores, restores = self.paged_state_checkpoints.take_checkpoint_ops()
        return StateMaintenanceOps(
            relocations=relocations,
            checkpoint_stores=stores,
            checkpoint_restores=restores,
        )

    def state_checkpoint_fates(self) -> dict[str, int]:
        """Summed fates across every state class, for the periodic stats line.

        Accumulates whatever each class's ``checkpoint_fates()`` returns, so a
        new counter appears here with no change to this method. A class without
        the method is skipped with a warning, latched per class: the caller is a
        periodic stats line and the omission is a static property of the build.
        """
        # 【译】所有状态类的命运计数之和，给周期性的统计行用。
        #
        #   累加每个类的 checkpoint_fates() 返回的内容，所以新增一个计数器不用改这个方法。没有这个方法的类会被跳过并警告，
        #   按类只警告一次：调用方是周期性的统计行，而缺这个方法是构建时就定下的静态属性。
        # 调用方：调度器的周期统计、pool_pressure（kept / dropped / orphaned等计数由各类自己定义）
        totals: dict[str, int] = {}
        for cache in self.state_caches:
            fates_fn = getattr(cache, "checkpoint_fates", None)
            if fates_fn is None:
                name = type(cache).__name__
                if name not in self._warned_no_checkpoint_fates:
                    self._warned_no_checkpoint_fates.add(name)
                    logger.warning(
                        "state_checkpoint_fates: %s does not implement "
                        "checkpoint_fates(); its counters are excluded from "
                        "totals",
                        name,
                    )
                continue
            for k, v in fates_fn().items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def _note_index_change(self, h: int) -> None:
        """`h` left the KV index or moved to another block id."""
        # 【译】h 离开了KV索引，或者改指向了另一个block id。
        # 两条来路：BlockPool的on_remap回调（publish时同一个hash改登记到别的block上）、下面的_record_evicted。
        #   只服务can_allocate_memoized：一次被缓存的查找读过的hash只要在这里出现，那次查找就作废（_walk_still_valid）
        self._index_changes += 1
        self._recent_index_changes.append(h)

    def _record_evicted(self, h: int) -> None:
        """A hash the block pool just dropped: report it, and settle the state.

        The crossing belongs here rather than in either pool — the two are
        addressed by one chained content hash and a prefix hit claims both, so
        neither can be left holding a boundary the other can no longer honour.
        Without this the state pool keeps handing slots to checkpoints nothing
        can reach and spends live ones to make room for them.
        """
        # 【译】块池刚丢掉了一个hash：上报它，并把状态那边结清。
        #
        #   这件事放在这里，而不放在任何一个池子里——两者由同一个链式内容hash寻址，一次前缀命中同时认领两者，所以不能让任何
        #   一方留着一个另一方已经兑现不了的边界。没有这一步，状态池会不断把槽位分给根本到达不了的checkpoint，还会为了腾地方
        #   把活着的花掉。
        # ★ 主干部分：BlockPool的on_evict回调。触发时机就是惰性淘汰的那一刻：BlockPool.allocate把一个还带hash的空闲block拿去装
        #   新内容（文件开头例子的⑦）；另外publish把一个block改登记成新hash时，旧hash也走这里。做三件事：
        #   计数（total_evicted_blocks）；记一条BlockRemoved事件（开了KV事件时）；【分支】把同一hash上的状态checkpoint一并删掉
        #   ——"状态 ⊆ KV"：那个位置的KV都没了，那里的状态快照也就接不上了
        #   观察：因为publish改登记也算进来，total_evicted_blocks会比BlockPool.blocks_evicted（只数allocate里的淘汰）略大
        # 对比nano-vllm：nano-vllm在_allocate_block里直接 del hash_to_block_id[hash]，没有回调
        self.total_evicted_blocks += 1
        self._note_index_change(h)
        if self._event_log is not None:
            self._event_log.append(_make_block_removed([h]))
        self._state_checkpoint_cache.unindex(h)

    # ★ 主干：取一个block来装"这一步马上要算的新内容"（≈ nano-vllm的_allocate_block）
    #   kv.pop()：先发没hash的vacant（id小的先），没有了才发cached里最早释放的那个；kv.allocate(id)：它若还带着hash，就在这一刻
    #   把hash从索引删掉（惰性淘汰，回调_record_evicted），并把ref_count设为1
    #   _ensure_page_units(1)：不开PAGE checkpoint时就是"还有没有空闲block"；开了时，空闲不够就先淘汰一个checkpoint腾出单元
    #   调用方：allocate（prompt里没命中的block）、may_append（decode和草稿lookahead要的新block）、disown_claimed_prefix
    def _fresh_block(self) -> int:
        """Take a block for content this step is about to compute.

        The raise is unreachable through `Scheduler` and the checkpoint cache
        cannot make it reachable: a READY unpinned checkpoint counts as
        available, and both callers sit behind a pin-aware check in the same
        pass. `allocate` protects the one checkpoint it is about to pin and
        sees the pins taken before it; `may_append` runs only in a pass that
        scheduled no prefill (`scheduler.py`, `if num_seqs_prefill > 0` returns
        first), so every pin was already released at the top of that pass --
        except for the draft lookahead an admission allocates right after its
        `allocate`, which `can_allocate` charged in the same units check
        (`draft_lookahead_blocks`), against the same protected checkpoint.
        Under contention the reachable outcome is a refused admission, not
        this.

        That second half rests on prefill and decode never sharing a pass. If
        the mixed batch that `scheduler.py` has a TODO for lands, `may_append`
        starts running alongside this pass's pins and the argument has to be
        redone.
        """
        # 【译】为这一步马上要算的内容取一个block。
        #
        #   经由Scheduler到不了这里的raise，checkpoint缓存也不会让它变得可达：一个READY且没被钉住的checkpoint算作可用，而两个
        #   调用方都在同一轮里先过了一道考虑pin的检查。allocate保护它马上要钉住的那一个checkpoint，并且看得到在它之前取的pin；
        #   may_append只在一个没排prefill的轮次里运行（scheduler.py里 if num_seqs_prefill > 0 先返回了），所以每个pin在那一轮
        #   开头就已经释放了——唯一的例外是准入在 allocate 之后紧接着分配的草稿lookahead，而 can_allocate 在同一次单元检查里
        #   （draft_lookahead_blocks）、针对同一个被保护的checkpoint把它算进去了。竞争之下实际能发生的是一次准入被拒绝，
        #   而不是走到这里。
        #
        #   后半段论证依赖"prefill和decode永远不在同一轮"。如果 scheduler.py 里那个TODO的混合batch落地了，may_append就会和
        #   本轮的pin同时运行，这个论证就得重做。
        # "一步只做prefill或只做decode"：见scheduler.py _schedule上方的决策流程图
        if not self._ensure_page_units(1):
            raise AssertionError("No PAGE unit available for a fresh KV block")
        block_id = self.kv.pop()
        self.kv.allocate(block_id)
        return block_id

    # 【分支】放弃一段已认领的共享前缀（review "finding #2"）：准入时认领了别人也在用的前缀block，之后发现它背后没有可用的
    #   状态（joint边界的状态腿没到、checkpoint丢了……），只能把num_cached_tokens清零、从token 0重算。可block_table还指着那些
    #   共享block，重算会就地覆写别的序列正在读的KV，所以先把共享的换成私有的新block
    # 例：A正在decode，B准入时认领了A的block [7, 3]（两个ref_count都是2），后来B要从0重算：
    #   free(7)、free(3) → ref_count回到1（A照常用）；B.block_table[0:2]换成两个新block，如[20, 21]，表长不变；
    #   B后面那些hash == -1的新block本来就是私有的，不动
    # 调用方：allocate（状态没接上时）、调度器两处（joint的KV腿被拒、offload的状态没送到）；返回False时调用方deallocate后重新排队
    def disown_claimed_prefix(self, seq: Sequence, keep_blocks: int = 0) -> bool:
        """Make this seq's claimed canonical prefix private, in place.

        A disown sets ``num_cached_tokens = 0`` so the forward recomputes from
        token 0, but ``block_table`` still points at the hash-indexed blocks
        this seq took with ``kv.claim`` -- blocks another sequence may be
        decoding out of. Recomputing writes KV in place into those shared
        blocks and tears the other holder's values (non-reproducible
        corruption -- this is review finding #2). This hands each claimed
        prefix block in ``block_table[keep_blocks:]`` back (``kv.free``) and
        drops a fresh private block (``_fresh_block``) into the same slot, so
        the table length and every other position's mapping are unchanged. The
        fresh tail (``hash == -1``) is already private and is left untouched;
        so is a still-shared block below ``keep_blocks``.

        Returns ``False`` when the pool cannot back the private copies -- the
        caller must then abandon this admission (deallocate + requeue) rather
        than run a forward over shared blocks. Only blocks still shared at
        ``ref_count > 1`` cost a net PAGE unit here (``free`` releases nothing
        while another holder remains), and ``can_allocate`` never reserved them
        (it subtracts already-used hit blocks from ``num_new_blocks``); those
        are reserved up front. ``ref_count == 1`` blocks recycle their own unit
        and never fail.
        """
        # 【译】把这条seq已认领的规范前缀就地变成私有的。
        #
        #   一次disown把 num_cached_tokens 设为0，让forward从token 0重算，但 block_table 仍指向这条seq用 kv.claim 认领的、登记在
        #   hash索引里的block——别的序列可能正从这些block里decode。重算会就地把KV写进这些共享block，撕坏另一个持有者的值
        #   （不可复现的数据损坏——这就是review finding #2）。这里把 block_table[keep_blocks:] 里每个认领来的前缀block交还
        #   （kv.free），并在同一位置放一个新的私有block（_fresh_block），所以表长和其他每个位置的映射都不变。新的尾部
        #   （hash == -1）本来就是私有的，不动；keep_blocks 以下仍共享的block也不动。
        #
        #   池子撑不起这些私有副本时返回 False——调用方必须放弃这次准入（deallocate + 重新排队），而不是在共享block上跑一次
        #   forward。只有仍在 ref_count > 1 共享的block会在这里净耗一个PAGE单元（还有别的持有者时 free 什么都释放不了），而
        #   can_allocate 从没为它们预留过（它从 num_new_blocks 里减掉了已被占用的命中block）；所以先把它们预留出来。
        #   ref_count == 1 的block回收的是它自己的单元，永远不会失败。
        if not self.enable_prefix_caching:
            return True
        table = seq.block_table
        n_shared = sum(
            1
            for i in range(keep_blocks, len(table))
            if self.kv.block(table[i]).hash != -1
            and self.kv.block(table[i]).ref_count > 1
        )
        # n_shared：要净增几个单元——带hash（认领来的）且还有别人共享的block；先一次性确认够了再动手，不会换到一半才失败
        if n_shared and not self._ensure_page_units(n_shared):
            return False
        for i in range(keep_blocks, len(table)):
            # 【译】（下一行行尾）已经私有的新尾部（或之前已被disown过的位置）
            if self.kv.block(table[i]).hash == -1:
                continue  # already-private fresh tail (or a prior disown)
            self.kv.free(table[i])
            table[i] = self._fresh_block()
        return True

    # 下面三个函数回答"KV池还拿得出多少个PAGE单元"。PAGE单元就是KV池里的一个block：它要么装KV，要么（开了PAGE状态checkpoint
    #   时）是某个checkpoint镜像的一片。不开PAGE checkpoint时三者都退化成 kv.has_free(n)——"空闲block够不够"（空闲里包括还带
    #   hash的cached block，取用时才淘汰）；开了时，可以淘汰的checkpoint占着的单元也算可用，protected_hash是这次准入马上要
    #   恢复的那个checkpoint，不能算作可淘汰
    #   _checkpoint_has_room的调用方：_record_checkpoint_demand；_has_page_units：_evaluate、can_append（只问，不动池子）；
    #   _ensure_page_units：_fresh_block、disown_claimed_prefix（开了PAGE checkpoint时真的淘汰checkpoint，把空闲表补到count个）
    def _checkpoint_has_room(
        self, live_blocks: int = 0, protected_hash: int | None = None
    ) -> bool:
        """Whether an image still fits once `live_blocks` have been taken.

        `live_blocks` is what the admission asking this is about to allocate.
        Counting it is the difference between "there is room for an image" and
        "there is room for this request and an image", and only the second is
        the question: the request's blocks are taken first.

        `protected_hash` is the checkpoint the same admission is about to pin,
        excluded from what eviction could reclaim — the same argument
        `can_allocate` passes to `_has_page_units` on the next line, so the
        two gates in one pass agree on what is spendable.

        `True` when no PAGE-backed checkpoints exist at all: a fork checkpoint
        costs the pool nothing, so there is nothing to gate.
        """
        # 【译】在已拿走 live_blocks 之后，一个镜像是否还放得下。
        #
        #   live_blocks 是问这个问题的那次准入马上要分配的数量。把它算进去，是"有地方放一个镜像"和"有地方放这个请求*加*一个镜像"
        #   的区别，而只有后者才是要问的问题：请求的block先拿。
        #
        #   protected_hash 是同一次准入马上要钉住的checkpoint，不算在淘汰可回收的范围内——与 can_allocate 在下一行传给
        #   _has_page_units 的参数相同，所以同一轮里的两道门对"什么可以花"意见一致。
        #
        #   完全没有PAGE checkpoint时返回 True：fork checkpoint不花池子任何东西，所以没什么要把关的。
        # units_per_checkpoint = ⌈镜像字节数 / 一个block的字节数⌉，例：KDA一个镜像127个block（见__init__里paged_state_checkpoints处）
        if self.paged_state_checkpoints is None:
            return True
        return self.paged_state_checkpoints.has_available_units(
            live_blocks + self.paged_state_checkpoints.store.units_per_checkpoint,
            protected_hash=protected_hash,
        )

    def _has_page_units(
        self, count: int, protected_checkpoint_hash: int | None = None
    ) -> bool:
        if self.paged_state_checkpoints is None:
            return self.kv.has_free(count)
        return self.paged_state_checkpoints.has_available_units(
            count, protected_hash=protected_checkpoint_hash
        )

    def _ensure_page_units(self, count: int) -> bool:
        if self.paged_state_checkpoints is None:
            return self.kv.has_free(count)
        return self.paged_state_checkpoints.ensure_free_units(count)

    def num_pool_blocks(self, seq_len: int) -> int:
        """KV pool blocks a `seq_len`-token sequence occupies on this rank.

        Under DCP a rank stores only its interleaved shard, so this is a factor
        of `dcp_world_size` below the global `ceil(seq_len / block_size)`. The
        pool is sized in these same per-rank units, so this is the only count
        that may be compared against `kv.num_blocks` — whether to draw from the
        pool (`can_allocate`/`allocate`) or to reject a prompt as too large for
        it (`Scheduler._unschedulable_reason`).
        """
        # 【译】一条 seq_len 个token的序列在本rank上占KV池的多少个block。
        #
        #   开DCP时一个rank只存它交错分到的那一份，所以这个数比全局的 ceil(seq_len / block_size) 小 dcp_world_size 倍。池子正是
        #   按同样的每rank单位定的大小，所以只有这个数可以拿来和 kv.num_blocks 比——无论是决定要不要从池里取
        #   （can_allocate / allocate），还是判定一个prompt对这个池来说太大而拒绝（Scheduler._unschedulable_reason）。
        # 例（block 16）：不开DCP，1000个token → ⌈1000/16⌉ = 63个；-dcp 4、S = 1 → rank 0本地 ⌈1000/4⌉ = 250个token → 16个
        #   （DCP为什么这么算见下面_effective_block_size上方的【前置：DCP】）。对比nano-vllm：就是nano-vllm的seq.num_blocks
        if self.dcp_world_size <= 1:
            return (seq_len + self.block_size - 1) // self.block_size
        # 只有开DCP才用到，所以在函数里导入。dcp_rank恒为0：rank 0分到的本地token最多（见__init__里dcp_rank处）
        from atom.model_ops.dcp_ops import get_dcp_local_seq_lens

        local_len = get_dcp_local_seq_lens(
            np.array([seq_len]),
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )[0]
        return int((local_len + self.block_size - 1) // self.block_size)

    def draft_lookahead_blocks(self, seq_len: int) -> int:
        """Blocks past `num_pool_blocks(seq_len)` that `draft_lookahead` adds.

        What admission charges on top of the prompt, so the block `may_append`
        takes for a forward ending at `seq_len` was paid for in the same pass.
        """
        # 【译】draft_lookahead 在 num_pool_blocks(seq_len) 之外多出来的block数。
        #
        #   这是准入在prompt之外多收的部分，这样 may_append 为一次止于 seq_len 的forward取的block，在同一轮里已经付过了。
        # 例（block 16、draft_lookahead = 2）：seq_len = 1000 → num_pool_blocks(1002) − num_pool_blocks(1000) = 63 − 63 = 0；
        #   seq_len = 1008 → 64 − 63 = 1（见draft_kv_lookahead上方的图）。调用方：_evaluate、max_pool_tokens
        if not self.draft_lookahead:
            return 0
        return self.num_pool_blocks(
            seq_len + self.draft_lookahead
        ) - self.num_pool_blocks(seq_len)

    # @property：当属性读（bm.max_pool_tokens，不加括号），每次读都重新计算
    @property
    def max_pool_tokens(self) -> int:
        """Longest prompt, in global tokens, whose KV fits an entirely empty pool.

        Bisects `num_pool_blocks`, which is monotone in `seq_len`, rather than
        inverting it in closed form: under block-level interleaving
        (`cp_kv_cache_interleave_size > 1`) a rank's share is not a plain
        `seq_len / dcp_world_size`, and an inverse derived by hand would drift
        from the allocator as soon as that arithmetic moved. Runs once, at
        startup.

        Mirrors the ceiling `Scheduler._unschedulable_reason` enforces, so the
        frontend can predict that verdict and refuse an oversized prompt with an
        error while it is still answering the client, instead of leaving the
        scheduler to discover it once the client is already waiting. The API
        server needs it published because `num_kvcache_blocks` is measured in
        the engine subprocess and its own Config never learns the value.
        """
        # 【译】一个完全空的池子能装下其KV的最长prompt（全局token数）。
        #
        #   对 num_pool_blocks 做二分查找（它关于 seq_len 单调），而不是用闭式解求逆：在block级交错下
        #   （cp_kv_cache_interleave_size > 1），一个rank的份额并不是简单的 seq_len / dcp_world_size，手推的反函数只要那段算术
        #   一变就会和分配器对不上。只在启动时跑一次。
        #
        #   它与 Scheduler._unschedulable_reason 执行的上限一致，所以前端能预判那个结论，在还在应答客户端时就用一个错误拒绝过大
        #   的prompt，而不是等客户端已经在等待时才由调度器发现。API server需要把它公布出来，是因为 num_kvcache_blocks 是在引擎
        #   子进程里测出来的，它自己那份Config永远拿不到这个值。
        # 流向：EngineCore的READY消息（engine_core.py）→ CoreManager取各rank的最小值 → api_server的_validate_pool_capacity对更长
        #   的prompt返回400
        # 例：100个block、block 16、不开DCP / MTP → 1600；开MTP k = 3（draft_lookahead = 2）→ 1598（要给草稿留2个位置）；
        #   -dcp 4、S = 1 → 每卡1600个本地token → 6400个全局token
        capacity = self.kv.num_blocks
        # A prompt this long needs more than `capacity` blocks on some rank, so
        # it bounds the search from above: each rank holds at least
        # `1 / dcp_world_size` of it, i.e. over `capacity` blocks' worth.
        # 【译】这么长的prompt在某个rank上需要超过 capacity 个block，所以它是搜索的上界：每个rank至少存它的 1 / dcp_world_size，
        #   也就是超过 capacity 个block的量。
        # 二分：找最大的lo使"prompt的block + 草稿lookahead的block ≤ capacity"；mid取上中位数（+ 1），lo = mid时区间也在缩小
        hi = (capacity + 1) * self.block_size * max(1, self.dcp_world_size)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.num_pool_blocks(mid) + self.draft_lookahead_blocks(mid) <= capacity:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # =========================================================================================
    # 【前置：DCP】为什么开DCP后"hash block = block_size × DCP"（本文件里DCP只在这里讲；完整原理见arg_utils.py
    #   EngineArgs.tensor_parallel_size下方的【科普】MLA、decode_context_parallel_size上方的【科普】）
    # =========================================================================================
    #   ① 前提：MLA只有1个KV头。MLA（DeepSeek-V3、Kimi-K2、GLM-5等）每token每层只缓存一行所有头共用的576维潜在向量
    #     （DeepSeek-V3若用MHA要存 128头 × (192 + 128) = 40960个数），相当于只有1个KV头。TP按头切KV，1个头切不开：
    #     -tp 8时8张卡各存一整份同样的KV
    #   ② DCP（decode上下文并行）：让TP组里的几张卡按token轮流各存一份，不增加卡数。-dcp 4、interleave = 1时token t只存在
    #     第t % 4张卡上，每卡只存1/4；注意力各卡先对自己那部分算，再用LSE合并（结果与在全部token上算一次相同）
    #   ③ 对账本的影响：4张卡用同一个block id，各在自己的KV池里放自己那16个token（dcp_ops.dcp_prefill_slot_mapping：
    #     槽位 = block_table[pos // (16 × 4)] × 16 + 本地序号 % 16），所以一个block table条目覆盖 16 × 4 = 64个全局token：
    #
    #       全局token        0  1  2  3  4  5  6  7  …  60 61 62 63 │ 64 65 …
    #       卡0的block 7     0           4           …  60          │          （卡0本地的第0~15个token）
    #       卡1的block 7        1           5        …     61       │          （卡1本地的第0~15个token）
    #       卡2、卡3同理                                             │
    #       → block_table[0] = 7 覆盖全局token 0~63，block_table[1]从64开始
    #
    #     于是本类里有两个数：
    #       block_size（16）                 一个rank上的物理block装几个token；KV池按"每卡本地token数 / block_size"计
    #                                        （num_pool_blocks、max_pool_tokens）
    #       hash_block_size = _effective_block_size() = block_size × DCP（64）
    #                                        一个block table条目覆盖几个全局token：前缀hash、命中、登记（_n_hash_blocks、
    #                                        _hash_block_tokens），以及can_append / may_append判断何时要加条目，都用它
    #     例：1000个token、-dcp 4：⌈1000/64⌉ = 16个条目；每卡本地 ⌈1000/4⌉ = 250个token → 也是16个物理block
    #   不开DCP（默认）时两个数相等，和nano-vllm一样
    #
    # 一个block table条目覆盖的全局token数。读者：can_append、may_append、draft_writes_allocated，以及调度器的一条报错信息。
    #   观察：和hash_block_size是同一个数的两种写法（这里每次现乘，hash_block_size在__init__里由prefix_hash.hash_block_size_for算好）
    def _effective_block_size(self):
        return self.block_size * self.dcp_world_size

    # 等价于直接读self.hash_block_size（allocate、hash_blocks、_joint_kv_boundary等用这个写法）
    def _hash_block_size(self) -> int:
        return self.hash_block_size

    # 这条序列一共要几个block table条目：⌈len(seq) / hash_block_size⌉；len(seq)即num_tokens，延迟输出 / MTP时含末尾的占位。
    #   例（不开DCP）：1000个token → 63个，最后一个可能没写满（≈ nano-vllm的seq.num_blocks）
    def _n_hash_blocks(self, seq: Sequence) -> int:
        hbs = self.hash_block_size
        return (len(seq) + hbs - 1) // hbs

    # 第i个条目覆盖的token id：token_ids[i × hbs : (i + 1) × hbs]（array("i")的切片，最后一块可能不满）。
    #   用来算链式hash，以及命中后比对内容（≈ nano-vllm的seq.block(i)）
    def _hash_block_tokens(self, seq: Sequence, i: int) -> array.array:
        hbs = self.hash_block_size
        return seq.token_ids[i * hbs : (i + 1) * hbs]

    # 【分支】DeepSeek-V4的有界重放（ATOM_DSV4_STATE_REPLAY打开时Config.state_replay_tokens > 0，第8批；原理与数字见sequence.py
    #   Replay类上方"科普：状态checkpoint与有界重放"④）。_evaluate在状态门控之后调用它，返回值就是最终的命中块数
    # 两种恢复方式比代价：
    #   不重放：从checkpoint（hit）处接上，要重算 [hit, 末尾)，hit之上那段KV命中白白浪费
    #   重放：保住整个KV命中（compressed_hit），只从它往前退state_replay_tokens个token重算一遍，把窗口状态精确重建出来
    # 例（hbs = 256，state_replay_tokens = R = 509，不开offload）：hit = 32块（8192），compressed_hit = 50块（12800）
    #   resume = 8192，end = 12800，end − resume = 4608 > 509 → 重放更省：返回50，seq.replays_hit = True；
    #   之后replay_start把forward的起点放到 12800 − 509 = 12291
    #   若compressed_hit只有33块（8448）：8448 − 8192 = 256 ≤ 509 → 不如直接从checkpoint接上，返回32
    # 开了LMCache MP时两头都可能被推远（推远的量不足OFFLOAD_MIN_LOAD_TOKENS（默认8192）就不算，因为connector本来就会跳过
    #   这么小的加载）：resume → kv_prefix_tokens（LMCache连KV带状态镜像都有的位置）；end → kv_only_tokens（只看全注意力KV
    #   能到多远），后者超过HBM命中时要先把那段KV加载回来（seq.replay_kv_load = True）
    def _replay_hit(self, seq: Sequence, compressed_hit: int, hit: int) -> int:
        """`hit` widened to `compressed_hit` when replaying is cheaper.

        Without a replay the forward resumes at the checkpoint `hit`, or at the
        LMCache MP hit (`kv_prefix_tokens`) if that is further, which brings its
        state image along. A replay resumes `state_replay_tokens` below the
        furthest full-attention KV: the HBM `compressed_hit`, or LMCache's
        KV-only reach (`kv_only_tokens`) when that is further, in which case
        the connector loads the KV past the HBM hit first (`replay_kv_load`).
        Records on `seq` whether the hit it returns replays (`replay_start`).
        """
        # 【译】当重放更便宜时，把 hit 放宽到 compressed_hit。
        #
        #   不重放时，forward从checkpoint hit 处恢复；如果LMCache MP的命中（kv_prefix_tokens）更远就从那里恢复，它会把自己的
        #   状态镜像一起带回来。重放则从最远的全注意力KV之下 state_replay_tokens 处恢复：HBM的 compressed_hit，或者LMCache
        #   仅KV能到达的位置（kv_only_tokens）——如果那更远，connector会先把HBM命中之后的KV加载回来（replay_kv_load）。
        #   在 seq 上记下它返回的命中是否要重放（replay_start 读它）。
        # 没开重放、或序列不带状态：原样返回门控后的hit
        hbs = self.hash_block_size
        seq.replays_hit = seq.replay_kv_load = False
        if not self.state_replay_tokens or not seq.has_per_req_cache:
            return hit
        # The connector skips a load under OFFLOAD_MIN_LOAD_TOKENS, so neither
        # LMCache figure counts unless it clears that above where it starts.
        # 【译】connector会跳过小于 OFFLOAD_MIN_LOAD_TOKENS 的加载，所以两个LMCache的数都必须比各自的起点多出这么多才算数。
        min_load = envs.OFFLOAD_MIN_LOAD_TOKENS
        resume = hit * hbs
        tier_hit = int(seq.offload_joint.kv_prefix_tokens or 0)
        if tier_hit - resume >= min_load:
            resume = tier_hit
        in_hbm = compressed_hit * hbs
        end = in_hbm
        tier_kv = int(seq.offload_joint.kv_only_tokens or 0)
        if tier_kv - in_hbm >= min_load:
            end = tier_kv
        if end - resume <= self.state_replay_tokens:
            return hit
        seq.replays_hit = True
        seq.replay_kv_load = end > in_hbm
        return compressed_hit

    def fall_back_to_hbm_replay(self, seq: Sequence) -> None:
        """A replay's KV-only load will not run: replay below the HBM hit.

        The state slot is fresh (`allocate` restored no checkpoint), so the HBM
        hit is no place to resume; the forward replays below it instead, and
        rewrites the blocks past it the load would have filled.
        """
        # 【译】一次重放的仅KV加载不会执行了：改为在HBM命中之下重放。
        #
        #   状态槽位是新的（allocate没有恢复任何checkpoint），所以HBM命中处不是能直接恢复的位置；forward改在它之下重放，并重写
        #   这次加载本该填上的、HBM命中之后的那些block。
        # 【分支】调用方：LMCache MP的native_state_scheduler（offload的KV加载不执行或失败时，第9批）
        seq.replay_kv_load = False

    # ★ 主干：一条序列的prompt hash链，每个cache_seed只算一次（缓存在_prompt_hash_chains里），优先用前端算好送来的
    #   （_adopt_shipped_chain）。链只覆盖prompt里写满的block：再往后的block还会随生成而变。读者：_walk_prefix / _hit_walk、
    #   _block_hash、allocate、_chain_to、hash_blocks（登记时直接取用，不再重算）
    def _prompt_hashes(self, seq: Sequence) -> list[int]:
        """`seq`'s prompt-block hash chain, computed once per `cache_seed`;
        starts from the frontend's when it checks out (`_adopt_shipped_chain`)."""
        # 【译】seq 的prompt block hash链，每个 cache_seed 只算一次；前端送来的链核对无误时就从它开始（_adopt_shipped_chain）。
        # 返回的是缓存里那个list本身：_block_hash、_hit_walk会往它后面追加新算出的prompt块hash（只追加prompt里写满的块）
        seed = seq.cache_seed
        cached = self._prompt_hash_chains.get(seq)
        if cached is None or cached[0] != seed:
            cached = (seed, self._adopt_shipped_chain(seq))
            self._prompt_hash_chains[seq] = cached
        return cached[1]

    # 前端预先算好的hash链（prefix_hash.ship_prompt_hash_chain，ATOM_FRONTEND_PREFIX_HASH默认开）：API进程在preprocess时就把
    #   prompt的整条链算好，随ADD消息经ZMQ送到EngineCore（Sequence本身被pickle时不带它），调度线程就不用为长prompt算几千次xxhash。
    #   但前端和引擎的配置可能不一致，所以核对后才采用：
    #   ① hash block大小、cache_seed、块数（= num_prompt_tokens // hbs）都对得上
    #   ② 链确实是从这些token算出来的：prompt token数、整个prompt的指纹（prompt_fingerprint，一次xxh64）都相等
    #   ③ 抽查第0块、中间一块、最后一块：在本地重算，链式hash相等（同一个hash函数）
    #   全过就整条信任；任何一条不过就丢掉、在本地重算（命中结果不受影响，只是调度线程多算hash）
    #   例：128k token的prompt、hbs 16 → 约8000块的链，只重算3块
    def _adopt_shipped_chain(self, seq: Sequence) -> list[int]:
        """The frontend's chain for `seq` (consumed), or `[]` to hash here.
        Adopted only if it was minted for this block size and seed from these
        prompt tokens (`prompt_fingerprint`), and its first, middle and last
        links rehash equal here (the same hash function). Then trusted whole:
        walks, `allocate` and `hash_blocks` read it."""
        # 【译】前端为 seq 算的链（取用后即清除），或者 [] 表示在这里重算。只有当它是为这个block大小和种子、从这些prompt token
        #   算出来的（prompt_fingerprint），并且它的第一、中间、最后三个链节在这里重算相等（同一个hash函数）时才采用。
        #   采用后就整条信任：各种查找、allocate 和 hash_blocks 都读它。
        shipped = getattr(seq, "prompt_hash_chain", None)
        if shipped is None:
            return []
        # 取用后即清：链只在这里用一次（结果存进_prompt_hash_chains）
        seq.prompt_hash_chain = None
        hbs, seed = self.hash_block_size, seq.cache_seed
        n = full_prompt_blocks(seq, hbs)
        packed = shipped.hashes
        if (shipped.hash_block_size, shipped.cache_seed, len(packed)) != (hbs, seed, n):
            return self._reject_shipped_chain(
                f"minted for hash block {shipped.hash_block_size}, seed "
                f"{shipped.cache_seed}, {len(packed)} blocks; this engine wants "
                f"{hbs}, {seed}, {n}"
            )
        if (shipped.num_prompt_tokens, shipped.fingerprint) != (
            seq.num_prompt_tokens,
            prompt_fingerprint(seq),
        ):
            return self._reject_shipped_chain("it was minted from other tokens")
        chain = packed.tolist()

        # link_holds(i)：第i块的hash是否等于 H(第i−1块的hash（i = 0时为种子）, 第i块的token)；{0, n // 2, n − 1}用集合去重
        #   （n很小时三个下标可能相同）
        def link_holds(i: int) -> bool:
            prev = chain[i - 1] if i else seed
            return chain[i] == self.compute_hash(self._hash_block_tokens(seq, i), prev)

        if n and not all(map(link_holds, {0, n // 2, n - 1})):
            return self._reject_shipped_chain("its hashes differ from this engine's")
        self.shipped_chains_adopted += 1
        return chain

    def _reject_shipped_chain(self, why: str) -> list[int]:
        # Warn once: a frontend/engine mismatch repeats on every request.
        # 【译】只警告一次：前端和引擎不一致会在每个请求上重复出现。
        if not self.shipped_chains_rejected:
            logger.warning(
                "Prefix cache: rehashing a prompt chain the frontend shipped, "
                "%s. Prefix hits are unaffected; the scheduler thread pays "
                "the hashing.",
                why,
            )
        self.shipped_chains_rejected += 1
        return []

    def _block_hash(self, seq: Sequence, chain: list[int], i: int, prev: int) -> int:
        """Chained hash of block `i` given block `i - 1`'s hash `prev`, read
        from or appended to `chain` (`_prompt_hashes`, full prompt blocks)."""
        # 【译】已知第 i − 1 块的hash为 prev 时第 i 块的链式hash：从 chain（_prompt_hashes，prompt里写满的块）中读出，
        #   或者算出来后追加进去。
        # 只有恰好接在链尾（i == len(chain)）且仍在prompt写满的范围内才追加，所以链里永远是一段连续的前缀
        if i < len(chain):
            return chain[i]
        h = self.compute_hash(self._hash_block_tokens(seq, i), prev)
        if i == len(chain) and i < full_prompt_blocks(seq, self.hash_block_size):
            chain.append(h)
        return h

    # 【分支】状态门控（第4、8批；为什么需要见文件开头前置概念的"每请求状态与状态checkpoint"）：KV前缀命中到compressed_hit块，
    #   但带每请求状态的序列只能在"有状态checkpoint"的位置接上。每个状态类回答"≤ X的位置里我能接受的最右一个"，本函数在
    #   各类之间反复传递候选值直到不再变化（不动点）
    # 例（hbs = 256）：compressed_hit = 72块（18432个token），checkpoint在32块（8192）和64块（16384）→ 返回64
    #   → 命中从18432截到16384，从16384开始prefill（与scheduler.py文件开头【前置⑤】的例子一致）
    # 调用方：_evaluate（准入）、_joint_kv_boundary（往HBM命中之外找边界）、_record_checkpoint_demand（assume_checkpointed=True
    #   的反事实）。不带状态的序列：每个类的resumable_hit直接原样返回（applies为False），命中不变——和nano-vllm一样
    def _gated_hit(
        self,
        seq: Sequence,
        compressed_hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Rightmost boundary every Pool.STATE class can resume from.

        Each class answers "the rightmost boundary <= X that I accept", and no
        class is monotone in another's answer, so they cannot be applied in
        series: the largest SWA-complete boundary need not carry a state
        checkpoint, and walking back to one that does can land on a boundary
        whose trailing SWA window is gone. `allocate` then calls
        `swa.claim_cached` for a hash the SWA pool never promised — which is
        exactly the guarantee that method's docstring asks the caller for.

        So run to a fixpoint: keep passing the candidate around the classes
        until a full round changes nothing. Every answer is <= its input, so
        each round either terminates or strictly decreases; 0 is absorbing.
        Classes that do not apply are the identity, so a build with one class
        settles on the first round.

        `assume_checkpointed` passes straight through to every class, giving the
        joint counterfactual: not "the answer minus one class's gate" but "the
        answer if every ladder were dense". A boundary the other classes decline
        anyway is one no checkpoint would rescue.
        """
        # 【译】每个 Pool.STATE 类都能从那里恢复的最右边界。
        #
        #   每个类回答"<= X 的边界里我能接受的最右一个"，而没有哪个类对另一个类的答案是单调的，所以不能串起来依次套用：最大的
        #   SWA完整边界不一定带着状态checkpoint，往回走到一个带checkpoint的边界，又可能落在一个尾部SWA窗口已经没了的边界上。
        #   那样 allocate 就会为一个SWA池从没承诺过的hash调用 swa.claim_cached——而这正是那个方法的docstring要求调用方保证
        #   不会发生的。
        #
        #   所以跑到不动点：把候选值在各个类之间传来传去，直到完整一轮什么都没改变。每个答案都 <= 它的输入，所以每一轮要么
        #   结束、要么严格减小；0是吸收态。不适用的类是恒等映射，所以只有一个类的构建在第一轮就收敛。
        #
        #   assume_checkpointed 原样传给每个类，给出联合的反事实："如果每条阶梯都是满的，答案是多少"，而不是"去掉某一个类的门槛
        #   之后的答案"。一个其他类无论如何都会拒绝的边界，是任何checkpoint都救不回来的。
        # 观察：docstring里的"SWA池""swa.claim_cached"是旧设计，仓库里已没有claim_cached；滑动窗口现在不是单独的类
        #   （见__init__里state_caches处），state_caches只有一个成员，循环第一轮就收敛
        boundary = compressed_hit
        while boundary > 0:
            settled = True
            for cache in self.state_caches:
                accepted = cache.resumable_hit(
                    seq,
                    boundary,
                    block_hashes,
                    assume_checkpointed=assume_checkpointed,
                )
                if accepted != boundary:
                    boundary = accepted
                    settled = False
                    if boundary == 0:
                        return 0
            if settled:
                break
        return boundary

    # 统计：KV池的占用快照，EngineStats每个统计周期打一行"[Cache Pool] used … free … retained-cache … evicted this interval …"
    #   used = 被序列引用的block，free = 空闲（含还能命中的cached），hashed = 索引里的hash数（在用的 + 空闲但还没被覆盖的）
    #   观察：retained = hashed − used 只是"空闲但还能命中的block"的近似：used里还有没登记hash的block（每条序列末尾没写满的块、
    #   decode新分到的块、PAGE checkpoint占的单元），会把它压低；更准的是BlockPool.num_reusable_free（cached和protected两个
    #   队列的长度）
    def pool_occupancy(self) -> dict[str, int]:
        used = self.kv.num_used
        free = self.kv.num_free
        hashed = self.kv.num_indexed
        return {
            "used": used,
            "free": free,
            "total": self.kv.num_blocks,
            "hashed": hashed,
            "retained": max(0, hashed - used),
            "evicted_total": self.total_evicted_blocks,
        }

    # =========================================================================================
    # 【分支】KV offload的联合加载（joint load）边界（B级；connector侧的实现见第9批 atom/kv_transfer/offload/）
    # =========================================================================================
    # 【前置：KV offload分层】（完整讲解：sequence.py SequenceStatus类上方"科普：KV offload"、arg_utils.py文件开头的【科普】KV offload）
    #   L1 HBM前缀缓存（本类的self.kv）── 算好的KV在后台另存一份 ──► L2 CPU内存 / L3 NVMe（LMCache、Mooncake Store，可到TB级）
    #   HBM里已被淘汰的前缀，以后可以从L2加载回显存、免去重算（2万个token的前缀约2.6GB，读回约0.06s、重算约0.25s，数字见
    #   sequence.py那段）。LMCache按chunk搬运（LMCACHE_CHUNK_SIZE，kv_transfer/offload/__init__.py里的例子是256个token），只能
    #   整chunk读写。带每请求状态的模型还多一条腿：除了KV，还要取回边界处的状态checkpoint（CPU状态层，self.state_offload）。
    #   两条腿必须落在同一个位置B——KV取到B、状态是B处的快照——从B接着算才是对的
    #
    # 为什么需要它：can_allocate只走HBM的前缀缓存，HBM里第一个缺失的block就停；状态门控还会把命中截到HBM里最近的checkpoint。
    #   本函数往HBM命中之外看：LMCache里还有KV、状态层里还有checkpoint的话，选一个两边都到得了的最右边界
    #
    # 例（hbs = 64，LMCache chunk = 256，checkpoint间隔8192；prompt 40000个token）：
    #   HBM：KV前缀命中70块（4480个token），但4480以下没有checkpoint → 门控后的HBM命中hbm_boundary = 0（只能冷启动）
    #   LMCache报告有前20000个token的KV（kv_prefix_tokens）→ 向下取整到chunk：19968
    #   cap = min(19968 / 64 = 312, ⌈40000/64⌉ − 1 = 624) = 312；hash链续到312块，_gated_hit问312块以内最右的可恢复点：
    #     设CPU状态层有16384处的checkpoint → candidate = 256块 = 16384个token
    #   kv_tokens = 16384向上取整到chunk = 16384（≤ 19968，KV腿够得着）
    #   claim：HBM里那70块免费认领，向下取整到chunk：4480 → 4352（17个chunk）
    #   结果：allocate认领HBM的 [0, 4352)，KV腿从LMCache取 [4352, 16384)，状态腿取16384处的checkpoint；两腿都到齐后
    #     num_cached_tokens才从0抬到16384（seq.offload_joint几个字段的位置关系见sequence.py OffloadJointRecord类上方的图）
    # 依次过的门（不过就返回_no_joint(原因)，原因由_commit_joint_boundary计入joint_skips）：
    #   off（没有CPU状态层）→ no_paged_checkpoints（序列不带状态，或没用PAGE checkpoint）→ no_chunk_size →
    #   lmcache_within_hbm（LMCache并不比HBM多）→ hbm_off_chunk_grid → no_room_above_hbm → no_rung_above_hbm →
    #   covering_chunk_beyond_lookup → claim_off_hash_grid
    # 调用方：record_allocation（_evaluate在record=True时，或调度器在等待 / 预算检查通过后调用）
    def _joint_kv_boundary(
        self,
        seq: Sequence,
        hbm_boundary: int,
        block_hashes: list[int],
    ) -> JointBoundaryDecision:
        """The boundary above the HBM hit that BOTH legs can reach, as a value.

        A pure computation: it reads the seq's KV-leg inputs and the caches and
        returns a `JointBoundaryDecision`, but writes nothing on the seq and
        moves no funnel counter. `can_allocate` decides whether to commit the
        result -- a real admission calls `_commit_joint_boundary`, a fit probe
        discards it. Separating the two is what let the `record` flag that used
        to thread down through here disappear.

        `can_allocate` walks the HBM prefix cache and nothing else, so an
        evicted prefix stops at the first miss even when LMCache holds every
        block. This looks past that: the KV leg fetches `[hbm, B)` from LMCache
        while the state leg fetches B's checkpoint from the tier, and the two
        land on one boundary or neither runs.

        The grids do not line up -- a state rung is a hash-block boundary, the
        KV leg moves whole chunks. Rather than discard every unaligned rung, the
        KV leg is aimed at the chunk *covering* B while the request claims only
        B: overshooting costs one chunk into blocks the forward is about to
        rewrite, undershooting would be silent wrong output.
        """
        # 【译】HBM命中之上、两条腿都能到达的边界，作为一个值返回。
        #
        #   纯计算：它读取seq的KV腿输入和各个缓存，返回一个 JointBoundaryDecision，但不在seq上写任何东西，也不动任何漏斗计数器。
        #   要不要提交由 can_allocate 决定——真正的准入调用 _commit_joint_boundary，适配探测则丢弃它。正是把两者分开，才让过去
        #   一路穿到这里的 record 标志消失。
        #
        #   can_allocate 只走HBM的前缀缓存，别的都不看，所以一个被淘汰的前缀会停在第一个未命中处，哪怕LMCache里每个block都在。
        #   这里看得更远：KV腿从LMCache取 [hbm, B)，状态腿从状态层取B处的checkpoint，两者落在同一个边界上，要么都执行，要么都
        #   不执行。
        #
        #   两边的网格对不齐——状态的一级是一个hash block边界，KV腿按整chunk搬运。与其丢掉每个没对齐的级，不如让KV腿瞄准*覆盖*B
        #   的那个chunk，而请求只认领到B：多取的代价是一个chunk的数据写进forward马上要重写的block，取少了则是悄无声息的错误输出。
        # 现在的提交点是record_allocation（见上方"调用方"），docstring里的"can_allocate决定"指的就是这条路径
        if self.state_offload is None:
            return self._no_joint("off")
        # PAGE is now the served class, not the excluded one. This used to
        # read `is not None` and refuse -- correct while a K3 checkpoint was an
        # Active Slot the tier spilled out of `StateSlotPool`. #2045 moved the
        # image into the KV pool, and with it the whole reason the joint path
        # exists: HBM's `state ⊆ KV` (`_record_evicted` unindexes a checkpoint
        # the instant its boundary block is spent) means a checkpoint can no
        # longer outlive its KV in HBM -- so when LMCache hands the KV back,
        # nothing hands the state back unless the two are fetched together.
        # 【译】PAGE现在是被服务的那一类，而不是被排除的那一类。这里过去写的是 is not None 并拒绝——那时K3的checkpoint还是一个由
        #   状态层从 StateSlotPool 里溢出的活跃槽位，这样做是对的。#2045 把镜像搬进了KV池，联合路径存在的全部理由也随之而来：
        #   HBM里"状态 ⊆ KV"（_record_evicted 在一个checkpoint的边界block被花掉的瞬间就取消它的索引）意味着在HBM里checkpoint再也
        #   活不过它的KV——所以当LMCache把KV送回来时，除非两者一起取，否则没有什么会把状态送回来。
        if not seq.has_per_req_cache or self.paged_state_checkpoints is None:
            return self._no_joint("no_paged_checkpoints")
        hbs = self._hash_block_size()
        # Resolved once from config at construction (`state_tier_chunk_tokens`),
        # not off the connector object: the scheduler holds whatever
        # `get_kvconnector` returned, and one without `chunk_size` would zero
        # this and disable the feature silently.
        # 【译】构造时从config解析一次（state_tier_chunk_tokens），而不是从connector对象上读：调度器持有的是 get_kvconnector 返回的
        #   任何东西，一个没有 chunk_size 的connector会把它置零，悄悄关掉这个特性。
        chunk = self._joint_chunk_tokens
        lmc_tokens = int(seq.offload_joint.kv_prefix_tokens or 0)
        if chunk <= 0:
            return self._no_joint("no_chunk_size")
        # Floored to the grid everything below compares against, so the
        # covering-chunk check becomes true by construction rather than by
        # luck. (`get_num_new_matched_tokens` withholds one token on a
        # full-prompt hit, which takes the lookup off the grid.)
        # 【译】向下取整到下面所有比较所用的网格上，这样"覆盖它的chunk"这个检查由构造保证为真，而不是靠运气。
        #   （get_num_new_matched_tokens 在整个prompt都命中时会扣下一个token，使查询结果偏离网格。）
        # 例：LMCache报告20000个token、chunk 256 → 19968
        lmc_tokens = (lmc_tokens // chunk) * chunk
        if lmc_tokens <= hbm_boundary * hbs:
            return self._no_joint("lmcache_within_hbm")
        # The KV leg moves whole chunks and the blocks below the HBM prefix are
        # shared, so an unaligned start cannot be rounded down.
        # 【译】KV腿按整chunk搬运，而HBM前缀以下的block是共享的，所以一个没对齐的起点不能向下取整。
        # 例（hbs 64、chunk 256）：HBM命中3块 = 192个token，192 % 256 ≠ 0 → 放弃（向下取到0就要往别人共享的block里写）
        if (hbm_boundary * hbs) % chunk != 0:
            return self._no_joint("hbm_off_chunk_grid")
        # cap：两条腿能到的上限（以hash block计）——不超过LMCache有的，也不到最后一块（最后一块永远要自己算，同can_allocate）
        cap = min(lmc_tokens // hbs, self._n_hash_blocks(seq) - 1)
        if cap <= hbm_boundary:
            return self._no_joint("no_room_above_hbm")
        chain = self._chain_to(seq, block_hashes, cap)
        # One scan: `_gated_hit` already returns the rightmost rung
        # `_resumable_from` accepts, so a decrement-and-rescan walk would spend
        # a fixpoint pass per rung to reach the same answer.
        # 【译】扫一次就够：_gated_hit 已经返回 _resumable_from 接受的最右一级，逐级递减再重扫的做法要为每一级花一轮不动点计算，
        #   才得到同样的答案。
        candidate = self._gated_hit(seq, cap, chain)
        if candidate <= hbm_boundary:
            return self._no_joint("no_rung_above_hbm")
        tokens = candidate * hbs
        h = chain[candidate - 1]
        # Cannot fail given the two premises above; kept as their assertion.
        # 【译】在上面两个前提下不可能失败；保留下来当作它们的断言。
        # -(-a // b)：Python里不经浮点的向上取整，即把边界B向上取整到chunk
        kv_tokens = -(-tokens // chunk) * chunk
        if kv_tokens > lmc_tokens:
            return self._no_joint("covering_chunk_beyond_lookup")
        # Everything up to the compressed hit is this prompt's KV and is still
        # in the pool -- what `_gated_hit` cut was resumability, not residency.
        # Below the joint boundary those blocks are read-only for this request
        # (the forward starts at the boundary), so claiming them is free, and
        # it is the difference between the KV leg fetching `[hit, B)` and
        # fetching `[0, B)` with the front half already sitting in HBM.
        #
        # `candidate > hbm_boundary` and `hbm_boundary` is the RIGHTMOST rung
        # `<= compressed_hit`, so the boundary is always past the compressed
        # hit and this min is always the compressed hit. Written as a min
        # anyway: the invariant belongs to `_gated_hit`, not here.
        # 【译】直到压缩命中为止的一切都是这个prompt的KV，并且仍在池里——_gated_hit 截掉的是可恢复性，不是驻留。在联合边界之下
        #   这些block对这个请求是只读的（forward从边界开始），所以认领它们不花钱；这就是KV腿取 [hit, B) 还是取 [0, B)（而前一半
        #   明明已经在HBM里）的区别。
        #
        #   candidate > hbm_boundary，而 hbm_boundary 是 <= compressed_hit 的*最右*一级，所以边界总在压缩命中之后，这个min
        #   永远等于压缩命中。还是写成min：这个不变式属于 _gated_hit，不属于这里。
        claim_blocks = min(len(block_hashes), candidate)
        # Floored to the chunk grid so the KV leg starts aligned and writes
        # only into blocks below it that nobody else can be reading.
        # 【译】向下取整到chunk网格，这样KV腿的起点是对齐的，而且只写进（边界以下）没有别人可能在读的block。
        # 例：70块 × 64 = 4480 → 4352（17个chunk）；不低于HBM命中（max）
        claim_tokens = ((claim_blocks * hbs) // chunk) * chunk
        claim_tokens = max(claim_tokens, hbm_boundary * hbs)
        # `allocate` claims the boundary in whole hash blocks --
        # `claim_tokens // hbs` -- but the value above is floored to
        # the *chunk* grid, not the hash-block grid. When `hbs` does not divide
        # `chunk`, a chunk-aligned claim is not hash-block-aligned: the floor
        # drops the `[floor(claim_tokens/hbs)*hbs, claim_tokens)` tail into a
        # fresh, unfilled block that `num_cached_tokens` (raised to the boundary
        # once the KV leg lands) then counts as computed -- silent wrong output,
        # with no exception. It is invisible whenever `hbs | chunk`, which is the
        # shipping case (`block_size * dcp_world_size == LMCACHE_CHUNK_SIZE`), so
        # it only surfaces at DCP world sizes that take `hbs` off the chunk grid.
        # Refuse the joint and recompute the tail rather than resume over a gap.
        # 【译】allocate 以整hash block为单位认领边界——claim_tokens // hbs——但上面的值是向下取整到*chunk*网格，而不是hash block
        #   网格。当 hbs 不整除 chunk 时，一个chunk对齐的认领不一定是hash block对齐的：取整会把
        #   [floor(claim_tokens/hbs)*hbs, claim_tokens) 这一段尾巴丢进一个新的、没填过的block，而 num_cached_tokens（KV腿到达后
        #   会被抬到边界）随后把它算作已计算——悄无声息的错误输出，没有任何异常。只要 hbs | chunk 它就看不出来，这正是出货配置
        #   的情况（block_size * dcp_world_size == LMCACHE_CHUNK_SIZE），所以它只在让 hbs 偏离chunk网格的DCP规模下才会出现。
        #   宁可拒绝联合、重算尾巴，也不要越过一个缺口去恢复。
        # 例：block 16、-dcp 6 → hbs = 96，chunk 256：70块 × 96 = 6720 → 向下取整到chunk = 6656，6656 % 96 = 32 ≠ 0 → 拒绝
        if claim_tokens % hbs != 0:
            return self._no_joint("claim_off_hash_grid")
        # Which tier the STATE leg came from. The KV leg's own tier is decided
        # separately by `_decide_load_after_alloc`, which is why this is not
        # named for the boundary as a whole. Read here, off the same cache the
        # boundary was computed against, and carried on the decision so the
        # commit does not have to look it up again.
        # 【译】状态腿来自哪一层。KV腿自己的层由 _decide_load_after_alloc 另行决定，所以这个字段不按整个边界命名。在这里、从计算
        #   边界所用的同一份缓存读出来，并带在决策上，这样提交时不必再查一次。
        # state_from_hbm：这个checkpoint此刻就在HBM的PAGE单元里（contains），否则要从CPU状态层取
        return JointBoundaryDecision(
            boundary_tokens=tokens,
            boundary_hash=h,
            kv_tokens=kv_tokens,
            claim_tokens=claim_tokens,
            state_from_hbm=self.paged_state_checkpoints.contains(h),
        )

    # @staticmethod：不需要self也不需要类，只是放在类里的普通函数
    @staticmethod
    def _no_joint(reason: str) -> JointBoundaryDecision:
        """The empty decision, naming why no joint boundary was found.
        Every reason is counted at commit time, including "this build is not
        even trying", because a silent zero is indistinguishable from a feature
        that ran and found nothing.
        """
        # 【译】空的决策，并写明为什么没找到联合边界。每个原因都在提交时计数，包括"这个构建根本没在尝试"，因为一个沉默的0和
        #   "特性跑过、什么都没找到"是没法区分的。
        return JointBoundaryDecision(skip_reason=reason)

    def _commit_joint_boundary(
        self, seq: Sequence, decision: JointBoundaryDecision
    ) -> None:
        """Apply a `JointBoundaryDecision` to the seq and the funnel counters.

        The recording half of what `_joint_kv_boundary` used to do inline under
        `record=True`: a real admission calls this, a fit probe does not, so the
        counters move once per admission and never for a seq that only probed.
        `reset_joint` first so a boundary that resolved to a skip this pass
        leaves no stale span from a prior one -- the same clean-slate the old
        code got from resetting at the top of every boundary walk.
        """
        # 【译】把一个 JointBoundaryDecision 应用到seq和漏斗计数器上。
        #
        #   这是 _joint_kv_boundary 过去在 record=True 下内联做的"记录"那一半：真正的准入调用它，适配探测不调用，所以计数器每次
        #   准入只动一次，只探测过的seq永远不动。先 reset_joint，这样一个在本轮落成skip的边界不会留下上一轮的过期范围——与旧代码
        #   在每次边界查找开头重置得到的干净状态相同。
        # 调用方：record_allocation。写进seq.offload_joint的四个字段之后由allocate（认领claim_tokens）和KV connector（取kv_tokens、
        #   状态腿取boundary_hash）读取
        seq.offload_joint.reset_joint()
        if decision.skip_reason is not None:
            self.joint_skips[decision.skip_reason] = (
                self.joint_skips.get(decision.skip_reason, 0) + 1
            )
            return
        seq.offload_joint.boundary_tokens = decision.boundary_tokens
        seq.offload_joint.boundary_hash = decision.boundary_hash
        seq.offload_joint.kv_tokens = decision.kv_tokens
        seq.offload_joint.claim_tokens = decision.claim_tokens
        self.joint_boundaries += 1
        if decision.state_from_hbm:
            self.state_hbm_boundaries += 1
        else:
            self.state_tier_boundaries += 1

    def _chain_to(
        self, seq: Sequence, block_hashes: list[int], blocks: int
    ) -> list[int]:
        """`block_hashes` continued to `blocks` entries, hashes only.
        A chained hash is a function of the prompt alone, so it is computable
        past where the HBM cache stops -- which is what makes an LMCache-only
        boundary addressable. Resumes from the last APPENDED hash: on a miss the
        caller's loop variable holds a hash that is not in the chain.
        """
        # 【译】把 block_hashes 续到 blocks 个条目，只算hash。
        #   链式hash只是prompt的函数，所以在HBM缓存停下的地方之后照样能算——这就是一个只在LMCache里的边界能被寻址的原因。
        #   从最后一个*追加进去*的hash接着算：未命中时，调用方的循环变量里放的是一个不在链里的hash。
        # 先尽量用prompt链（_prompt_hashes）里现成的；首尾对不上（block_hashes不是这条prompt链的前缀）就不用它、全部现算。
        #   调用方：_joint_kv_boundary、_extend_hash_chain、prefix_hash_chain
        prompt = self._prompt_hashes(seq)
        chain = list(block_hashes)
        k = min(len(chain), len(prompt))
        if k and (chain[0], chain[k - 1]) != (prompt[0], prompt[k - 1]):
            # 【译】（下一行行尾）不是prompt链的前缀：越过它现算。
            prompt = []  # Not a prefix of the prompt chain: hash past it.
        chain += prompt[len(chain) : blocks]
        h = chain[-1] if chain else seq.cache_seed
        for i in range(len(chain), blocks):
            h = self._block_hash(seq, prompt, i, h)
            chain.append(h)
        return chain

    def prefix_hash_chain(
        self, seq: Sequence, block_hashes: list[int], blocks: int
    ) -> list[int]:
        """`block_hashes` continued to `blocks` entries, for callers outside
        the manager. Seeded from `seq.cache_seed` like every chain it mints."""
        # 【译】把 block_hashes 续到 blocks 个条目，给管理器之外的调用方用。和它生成的每条链一样，以 seq.cache_seed 为种子。
        # 【分支】调用方：LMCache MP的native_state_scheduler（算checkpoint边界的hash，第9批）
        return self._chain_to(seq, block_hashes, blocks)

    def can_allocate(
        self,
        seq: Sequence,
        record: bool = True,
        *,
        block_hashes: list[int] | None = None,
    ) -> int:
        """Return number of cache-hit blocks (>=0) if seq fits, else -1.

        `record=False` returns the same fit and HBM hit without committing
        joint-load fields or joint-boundary counters. Checkpoint demand/end,
        hit instrumentation and `Sequence.replays_hit` (whether the hit it
        returns replays, which `replay_start` reads for it) still refresh;
        demand counters deduplicate per request. An optional `block_hashes` output
        lets the scheduler defer `record_allocation` until after its wait and
        token-budget checks, reusing this probe's chain without another walk.
        `_admission_walk` then `_evaluate`; `can_allocate_memoized` evaluates a
        memoised walk the same way.

        The hit count is the contiguous run of cache hits starting at the
        prompt's first block. On the first miss we break: subsequent blocks
        cannot match either (hash is chained, so a divergent token breaks the
        chain for the rest of the prompt). The last block is never considered
        for reuse — prefill must forward at least one block to produce
        sampler logits, so it always comes from the free pool.

        Caller (scheduler) passes the returned hit count to `allocate()`,
        avoiding a second hash pass.
        """
        # 【译】seq放得下时返回前缀缓存命中的block数（>=0），否则返回-1。
        #
        #   record=False 返回同样的"放不放得下"和HBM命中，但不提交联合加载（joint load）字段和联合边界计数器。
        #   checkpoint的demand/end、命中统计和 Sequence.replays_hit（它返回的命中是否要重放，replay_start 会替它读这个值）
        #   照样刷新；demand计数器按请求去重。可选的 block_hashes 输出参数让调度器能把 record_allocation 推迟到它的
        #   等待检查和token预算检查之后，复用这次探测的hash链，不必再遍历一次。
        #   流程是 _admission_walk 然后 _evaluate；can_allocate_memoized 以同样的方式评估一个缓存下来的遍历结果。
        #
        #   命中数是从prompt第一个block开始、连续命中的那一段。遇到第一个未命中就停：后面的block也不可能命中
        #   （hash是链式的，一个不同的token会让此后整个prompt的链都断开）。最后一个block从不参与复用——prefill
        #   至少要forward一个block才能产出供采样的logits，所以它总是从空闲池里取。
        #
        #   调用方（调度器）把返回的命中数传给 allocate()，避免第二次hash遍历。
        #
        # ==================== 准入（admission）：本段（can_allocate ~ _chain_parent_hash）的总览 ====================
        # 这里是"新请求能不能进来、进来时复用哪些block"的实现，即前缀缓存命中判定的规范位置。
        #   概念（hash为什么链式、为什么最后一个block不复用）见sequence.py Sequence类上方"科普：前缀缓存"、
        #   scheduler.py文件开头【前置③】；hash本身（种子、碰撞、DCP下块变大、前端算链）见prefix_hash.py文件头【科普】①~⑤；
        #   命中的block怎样共享、释放后何时才真正失效（惰性淘汰）见block_pool.py BlockPool类。这里讲准入的实现。
        #   调度器怎样调用：scheduler.py _schedule的Phase 2（"★ 主干，对应nano-vllm的block_manager.can_allocate(seq)"一段）
        #     _probe_hbm_prefix: can_allocate(seq, record=False, block_hashes=[])  -->  token预算 / chunk检查
        #       -->  record_allocation(seq, 命中块数, block_hashes)  -->  allocate(seq, 命中块数)
        #     任何一步不通过：seq放回waiting队头、本轮停止准入（还没allocate，不用释放）
        #   记号：hbs = hash_block_size = block_size × DCP（DCP=1时就是block_size；为什么乘DCP见scheduler.py文件开头【前置⑥】），
        #     一个hash block = block_table的一项；n = ⌈len(seq) / hbs⌉ 是整条序列的块数
        #
        # 方法一览（主干A级 / 【分支】B级）：
        #   can_allocate ................................ 入口：探测 + 容量检查，返回命中块数或-1
        #   _evaluate ................................... 真正的判定（can_allocate 和 can_allocate_memoized 共用）
        #   _walk_prefix / _hit_walk .................... 前缀遍历：沿hash链查表，第一次未命中就停
        #   _admission_walk / _take_admission_walk ...... 把探测查到的block id留给allocate直接认领
        #   can_allocate_memoized / _memoised_walk / _walk_still_valid ... 给PrefillDelayer等"只探测"方的跨tick遍历缓存
        #   record_allocation ........................... 【分支】提交联合边界（LMCache offload，第9批）
        #   allocate .................................... 认领命中块（引用计数+1）、取新块、【分支】给状态槽位
        #   replay_start ................................ 【分支】有状态的模型从命中点往前退多少开始prefill
        #   _state_leg_secured ~ take_state_loads ....... 【分支】每请求状态槽位、CPU状态层的加载 / 存储（第9批）
        #   _chain_parent_hash .......................... hash_blocks的辅助：取前一个block的hash
        #
        # 【前置：每请求状态与状态checkpoint】（下面(4)、allocate的状态槽位、replay_start都要用到；完整讲解见arg_utils.py
        #   EngineArgs.state_checkpoint_interval_tokens上方的【科普】、sequence.py Replay类上方"科普：状态checkpoint与有界重放"）
        #   线性注意力层（GDN：Qwen3-Next / Qwen3.5；KDA：Kimi-K3、GLM-5.3）不给每个token留一行K/V，而是每条请求每层一个
        #   定长状态矩阵（Qwen3-Next一层GDN约 32头 × 128 × 128 ≈ 52万个数），每来一个token原地覆盖；DeepSeek-V4 / V4.1的
        #   128窗口滑动注意力、压缩器还没凑满一组的token，也是这种"每请求状态"。它们放在单独的状态池里，按"槽位"编号。
        #   于是前缀命中时：KV行还在，"第N个token时的状态"却早被覆盖了 → 只能从事先存下的快照（状态checkpoint）接上；
        #   checkpoint只存在少数位置：每 state_checkpoint_interval_tokens（默认8192）个token一级 + prompt末尾锚点 + 按需补的一级。
        #   例（Qwen3-Next，block 16）：新请求与旧prompt共享前18432个token，旧prompt的checkpoint还在8192、16384
        #     → KV遍历命中 18432 / 16 = 1152块（compressed_hit），但状态只在第512块（8192）、第1024块（16384）有快照
        #     → _gated_hit 截到1024块 → num_cached_blocks = 1024，从16384开始prefill（比KV命中多算2048个token）；
        #     同时 _record_checkpoint_demand 记下"想要的"1152块（demand = 18432；池子放得下一份镜像时）：本请求prefill时在
        #     18432处多切一刀、存一个checkpoint，下一个同前缀的请求就能整段命中
        #   普通模型（没有每请求状态）没有这一步：KV命中多少块就复用多少块
        #
        # 流程图（与下面代码一一对应；普通模型只走不带【分支】的格子）：
        #   ┌─ can_allocate(seq) -> _evaluate(seq, _admission_walk, record, block_hashes) ────────────────────┐
        #   │ (1) 【分支】有每请求状态 且 空闲状态槽位 < state_slots_per_req ?   是 -> return -1              │
        #   │ (2) 没开前缀缓存 ?  是 -> 池子够 num_pool_blocks(len) + 草稿lookahead 块 ?  够 -> 0，不够 -> -1 │
        #   │ (3) 前缀遍历 walk = _walk_prefix(seq)，只看前 n-1 块（n = ceil(len / hbs)，最后一块不复用）：   │
        #   │       for i in 0 .. n-2:                                                                        │
        #   │           h_i = H(h_{i-1}, 第i块的token)      （h_{-1} = cache_seed；链已缓存时直接读）         │
        #   │           id  = 索引.get(h_i)；查不到 或 token对不上 -> break（后面的块不可能再命中）           │
        #   │           记下 (h_i, id)                                                                        │
        #   │     compressed_hit = 记下的块数                                                                 │
        #   │ (4) 【分支】_gated_hit：退到有状态checkpoint的边界；_replay_hit：checkpoint太远时改用有界重放   │
        #   │     （普通模型两步都原样返回） -> num_cached_blocks                                             │
        #   │ (5) num_new_blocks = n - 命中块里正被别的序列用着的块数 + 草稿lookahead块数                     │
        #   │ (6) 【分支】_record_checkpoint_demand / _record_checkpoint_end：记下想放checkpoint的位置        │
        #   │ (7) 空闲block（+ 可淘汰的PAGE checkpoint单元）< num_new_blocks ?   是 -> return -1              │
        #   │ (8) record=True -> record_allocation（【分支】联合边界）                                        │
        #   │ (9) return num_cached_blocks                                                                    │
        #   └─────────────────────────────────────────────────────────────────────────────────────────────────┘
        #                  │ >= 0：调度器再做token预算 / chunk检查（scheduler.py _schedule Phase 2），通过后先record_allocation，再
        #                  ▼
        #   ┌─ allocate(seq, num_cached_blocks) ─────────────────────────────────────────────────────────────┐
        #   │ (1) claim_blocks = num_cached_blocks（【分支】有联合边界时放宽到 claim_tokens / hbs）          │
        #   │ (2) 探测留下的walk仍有效（_take_admission_walk）? 是 -> claim_many(前claimed个id)：            │
        #   │       已被别的序列用着 -> ref_count + 1；在空闲表里 -> 取出来，ref_count = 1（内容和hash原样） │
        #   │     其余的（walk失效时就是全部）逐块：h = 链上的hash，id = 索引[h]，claim(id)                  │
        #   │       查不到 -> 截断命中、break                                                                │
        #   │ (3) 【分支】V4有界重放：hit_hash = -1（拿新槽位，靠重算重建状态）                              │
        #   │ (4) 【分支】PAGE checkpoint：_attach_state_slots（排好恢复、钉住checkpoint，必须在取新块之前） │
        #   │ (5) 补齐到 num_pool_blocks(len) 块：_fresh_block()                                             │
        #   │       先取没有hash的空块（id小的先），没有了才取最久没用的缓存块（这一刻才淘汰它的hash）       │
        #   │ (6) num_cached_tokens = num_cached_blocks * hbs                                                │
        #   │ (7) 【分支】fork状态槽位：_attach_state_slots                                                  │
        #   │ (8) 【分支】状态对不上 -> disown_claimed_prefix（私有化共享前缀），num_cached_tokens = 0       │
        #   │       私有化所需的块不够 -> return False（调度器 deallocate + 放回waiting）                    │
        #   │ (9) return True                                                                                │
        #   └────────────────────────────────────────────────────────────────────────────────────────────────┘
        #   hash不在这里登记：forward算完KV后，postprocess → hash_blocks 只登记写满的块（与nano-vllm的hash_blocks相同）
        #
        # 一条请求带数字走一遍（block_size=16、无DCP → hbs=16；池子20个block，id 0~19起初都空；普通模型、不开MTP）：
        #   请求A：prompt 100个token → n = ⌈100/16⌉ = 7块（前6块满，第7块只有4个token）
        #     can_allocate：索引是空的，_hit_walk第一轮只查第0块就未命中 → 命中0；要7块，空闲20 ≥ 7 → 返回0
        #     allocate：没有可认领的；_fresh_block取7块（空块按id从小到大）→ A.block_table=[0,1,2,3,4,5,6]，空闲20→13
        #     prefill算完后hash_blocks登记写满的块0~5：索引 {h0:0, h1:1, …, h5:5}；第7块只有4个token，不登记
        #   请求B：prompt也是100个token，前70个与A相同，70~99不同
        #     块0 [0,16)、块1 [16,32)、块2 [32,48)、块3 [48,64)：内容和前缀都相同 → h0~h3 命中
        #     块4 [64,80)：64~69相同、70~79不同 → hash不同 → 未命中，停（链式hash，块5也不可能再命中）
        #     → 命中4块 = 64个token。64~69这6个token虽然相同也要重算：复用以整块为单位，"半块"不算
        #     （前端已送来B的prompt链（6个满块的hash），_hit_walk做了2轮查表：第1轮只查h0；第2轮一次查h1~h5共5个，在h4处未命中）
        #   (a) A还在运行（占着0~6）：命中的4块都被A用着 → num_new_blocks = 7 − 4 = 3；空闲13 ≥ 3 → 返回4
        #       allocate：claim_many([0,1,2,3]) 把它们的ref_count 1→2；再取3个新块7、8、9
        #       → B.block_table=[0,1,2,3,7,8,9]，空闲13→10，num_cached_tokens = 4×16 = 64，B只prefill 100−64 = 36个token
        #   (b) A已结束：deallocate倒序释放，块0~5带着hash进"缓存空闲"队列，没hash的块6进"空块"堆；空闲 = 20
        #       命中的4块都在空闲表里 → num_new_blocks = 7 − 0 = 7（从空闲表捞回来也占名额）；20 ≥ 7 → 返回4
        #       allocate：claim_many把0~3从空闲表取出（ref_count 0→1，内容和hash原样保留）；新块先取空块6、7、8
        #       → B.block_table=[0,1,2,3,6,7,8]，空闲20→13；A的块4、5（h4、h5）仍在缓存空闲队列里，没有被淘汰
        #   为什么最后一块总是重算：请求C的prompt恰好是A的前96个token（6个满块）
        #     n=6，遍历只看前5块 → 命中5块 = 80个token；块5虽然就在缓存里，也要重算这16个token：prefill至少要forward
        #     最后一个位置，才有logits采样第1个输出token，整条prompt都命中就没有东西可forward了
        #     （A的100个token原样再来一遍：n=7，看前6块 → 命中6块 = 96，只重算最后4个token）
        #
        # 对比nano-vllm（block_manager.py的can_allocate / allocate，两个函数合计约40行代码）：骨架相同——沿链式hash逐块查、第一次未命中就停、
        #   最后一块不复用、命中块里"正被别人用着的"不占空闲名额、allocate先认领命中块（ref_count+1）再取新块。ATOM多出来的：
        #   ① hash链只算一次：_prompt_hashes把prompt的链缓存在seq上，默认由前端（API进程）算好随请求送来
        #      （ATOM_FRONTEND_PREFIX_HASH，见prefix_hash.py【科普】⑤）；nano-vllm在can_allocate和allocate里各hash一遍
        #   ② 批量查表（_hit_walk，ATOM_PREFIX_FAST_WALK默认开）：用map按轮次查，只比对最后一个命中块的token；nano-vllm每块都比对
        #   ③ allocate直接认领探测时查到的block id（_take_admission_walk），不再"算hash → 查表"第二遍；索引在中间变过才逐块重查
        #   ④ 跨tick缓存遍历（can_allocate_memoized）：PrefillDelayer、DP代价均衡每个tick都要探一遍waiting队列
        #   ⑤ hash block = block_size × DCP（开DCP时block_table的一项装 16×DCP 个全局token）
        #   ⑥ 【分支】每请求状态：状态槽位够不够也是准入条件；命中要被状态checkpoint"门控"截短（_gated_hit），或改用有界重放
        #      （_replay_hit / replay_start）；PAGE checkpoint的镜像和KV共用一个池子，所以容量检查要算上可淘汰的checkpoint
        #   ⑦ 【分支】投机解码：MTP草稿层要多写len(seq)之后的k−1个位置，准入时一并计费（draft_lookahead_blocks）
        #   ⑧ 【分支】LMCache offload的联合边界（record_allocation、claim_tokens，第9批）
        #   ⑨ 空闲块的选取：nano-vllm是一个FIFO（先释放的先被重用）；ATOM先取没有hash的空块、再按LRU花掉缓存块
        #      （block_pool.py BlockPool._take_free），淘汰发生在块被重新分配的那一刻（惰性淘汰）
        #   ⑩ allocate可能失败（返回False）：【分支】状态对不上、又给不出私有化共享前缀所需的块时，见allocate末尾
        # =====================================================================================================
        # 精确探测：遍历用 _admission_walk（快速遍历时把查到的block id留给本seq的allocate直接认领），其余判定全在 _evaluate
        return self._evaluate(seq, self._admission_walk, record, block_hashes)

    def _evaluate(
        self,
        seq: Sequence,
        walk_of: Callable[[Sequence], _PrefixWalk],
        record: bool,
        block_hashes: list[int] | None,
    ) -> int:
        """`can_allocate` over the walk `walk_of(seq)` returns; everything but
        the walk is asked afresh on every call."""
        # 【译】在 walk_of(seq) 返回的遍历结果上做 can_allocate；除了遍历本身，其余每一项每次调用都重新询问。
        # walk_of 有两种：can_allocate 传 _admission_walk（每次都重新遍历），can_allocate_memoized 传 _memoised_walk
        #   （遍历结果跨tick缓存；ATOM_PREFIX_WALK_MEMO=0 时传 _walk_prefix）。"其余每一项"= 状态槽位、门控、重放、命中块里几块正被使用、池子够不够——它们随池子变化，
        #   不能缓存；只有"沿hash链能走多远"这件最贵的事可以缓存（有效条件见 _walk_still_valid）
        # 下面的 (1)~(9) 与 can_allocate 里流程图的编号一一对应
        # Active Slots are preallocated; PAGE checkpoints share the KV pool.
        #
        # The full per-request width, because that is what `allocate` will take:
        # gating on one slot would admit a request the pool cannot give a
        # rollback set to.
        # 【译】Active Slot（请求正在用的状态槽位）是预先分配好的；PAGE checkpoint则与KV共用一个池子。
        #
        #   按每请求的完整宽度判断，因为allocate就要拿这么多：只按1个槽位判断，会放进一个池子给不出整组回滚槽位的请求。
        # 这段英文说的是下面第二个if（状态闸门）。在它之前先清空输出参数 block_hashes：下面会把本次遍历的hash链填进去，
        #   调度器稍后原样交给 record_allocation，不必再遍历一次
        if block_hashes is not None:
            block_hashes.clear()
        # (1) 【分支】状态槽位闸门（槽位是什么见下方 _state_leg_secured 上方的【分支】总览）：一条有每请求状态的请求要拿
        #   state_slots_per_req = 1 + 投机token数 个槽位（MTP k=3 时4个：1个已提交状态 + 3个回滚）。空闲槽位里也算上
        #   "仍存着fork checkpoint的空闲槽位"（pop会花掉它），所以checkpoint不占准入容量。普通模型 has_per_req_cache=False，跳过
        if seq.has_per_req_cache and not self.state.has_free(self.state_slots_per_req):
            return -1
        # (2) 没开前缀缓存：不遍历、不命中，只看池子装不装得下整条序列 + 草稿lookahead（单位是本rank的KV block，见num_pool_blocks；
        #   有PAGE checkpoint时 _has_page_units 还把可淘汰的checkpoint单元算作可用）
        if not self.enable_prefix_caching:
            if not self._has_page_units(
                self.num_pool_blocks(len(seq)) + self.draft_lookahead_blocks(len(seq))
            ):
                return -1
            return 0
        # Step 1: compressed prefix (CSA/HCA/indexer share the block hash and
        # read the WHOLE history, so this stays a full front-to-back chained
        # match). Record each block's hash for the SWA scan below.
        # 【译】第1步：压缩前缀（CSA/HCA/索引器共用block hash，而且读取完整的历史，所以这里仍然是从头到尾的完整链式匹配）。
        #   记下每个block的hash，供下面的SWA扫描使用。
        # (3) 前缀遍历。"压缩前缀"是DeepSeek-V4的说法：它的KV池存的是压缩后的行（CSA / HCA，见config.py __post_init__里
        #   "【科普】DeepSeek-V4的压缩注意力（CSA / HCA）"）；对普通模型就是普通的KV block。只看前 n−1 块、第一次未命中就停，
        #   实现见 _walk_prefix / _hit_walk。walk.hashes[i] 是第i块的链式hash，walk.ids[i] 是索引里存着它的物理block
        # 观察：英文里的"SWA扫描"是旧设计——滑动窗口（SWA）曾是第二个按内容寻址的block池；现在它是每请求的环形缓冲、
        #   由状态checkpoint携带（见__init__里 state_caches 上方的英文），下面的门控只剩 _gated_hit 一处
        walk = walk_of(seq)
        # 调用方没给输出列表时直接引用walk里的列表（行尾英文：只读，下面不会往里追加）；给了就把hash链复制进去（+= 原地扩展）
        if block_hashes is None:
            block_hashes = walk.hashes  # read, never extended, below
        else:
            block_hashes += walk.hashes
        # 门控之前的命中块数。例：前70个token相同、block 16 → 4
        compressed_hit = len(walk.hashes)
        # Step 2: SWA only needs the trailing window before the boundary to be
        # present (SWA is local). Scan right-to-left within the compressed prefix
        # for the largest boundary whose window is SWA-cached (vLLM
        # SlidingWindowManager; simple-hybrid one pass). Reduces compressed_hit
        # → num_cached_blocks so we never reuse a block whose in-window SWA is
        # gone (#1417), while out-of-window front blocks (SWA-freed) don't block
        # the hit —
        # plus step 3, the per-request state: neither the SSM recurrent state nor
        # the V4 compressor ring can be rebuilt from cached blocks — the cache
        # holds the compressor's output, the state is its rolling input window —
        # so a boundary is only resumable where somebody checkpointed the state.
        # `_gated_hit` settles the two gates jointly; neither can be applied to
        # the other's answer.
        # 【译】第2步：SWA只要求边界之前的尾部窗口还在（SWA是局部的）。在压缩前缀内从右往左扫，找窗口仍在SWA缓存里的最大边界
        #   （vLLM的SlidingWindowManager；简单混合模型扫一遍即可）。这把 compressed_hit 缩减为 num_cached_blocks，保证绝不复用
        #   窗口内SWA已经不在的block（#1417），而窗口之外、SWA已释放的前部block不妨碍命中——
        #   再加上第3步，每请求状态：SSM的递归状态和V4压缩器的环形缓冲都无法从缓存的block重建——缓存里存的是压缩器的输出，
        #   状态则是它滚动的输入窗口——所以只有在有人给状态存过checkpoint的地方，边界才能恢复。
        #   _gated_hit 把两道门控一起求解；哪一道都不能直接作用在另一道的结果上。
        # (4a) 【分支】状态门控（前置见上面 can_allocate 里的【前置：每请求状态与状态checkpoint】）：在 [0, compressed_hit] 里找
        #   "每个状态类都能从这里恢复"的最右边界，各类轮流回答直到不动点。对没有每请求状态的seq，各类都不适用，原样返回。
        #   例：compressed_hit = 1152块、checkpoint在第512、1024块 → resumable = 1024
        resumable = self._gated_hit(seq, compressed_hit, block_hashes)
        # A checkpoint far below the compressed hit loses to a bounded replay,
        # which keeps the whole hit and rebuilds the state at its end.
        # 【译】离压缩命中太远的checkpoint，输给有界重放：重放保留整个命中，在命中末端重建状态。
        # (4b) 【分支】DeepSeek-V4有界重放（ATOM_DSV4_STATE_REPLAY，Config.state_replay_tokens > 0 时）：若"从checkpoint接上要多算的
        #   长度"超过 state_replay_tokens，就改为保留整个KV命中，从命中点往前退 state_replay_tokens 个token重算来重建状态
        #   （seq.replays_hit = True，起点由 replay_start 算）。例（sequence.py Replay类上方科普④的数字）：KV命中12800、checkpoint
        #   在8192、state_replay_tokens = 509 → 12800 − 8192 = 4608 > 509 → 保留12800。其他模型原样返回 resumable
        num_cached_blocks = self._replay_hit(seq, compressed_hit, resumable)
        # Instrumentation: the pre-gate hit, so EngineStats can separate reuse
        # the gates declined (compressed_hit - num_cached_blocks) from reuse
        # lost to compressed eviction (everything above compressed_hit).
        # 【译】统计用：门控之前的命中，让EngineStats能把"被门控拒绝的复用"（compressed_hit − num_cached_blocks）和
        #   "因压缩KV被淘汰而丢掉的复用"（compressed_hit以上的全部）区分开。
        seq.num_compressed_hit_blocks = compressed_hit
        # Free-pool demand: blocks we actually reuse minus those already used
        # (shared ref); blocks we drop from the hit become fresh → counted.
        # Plus the draft round's lookahead past the prompt, which the scheduler
        # allocates right after `allocate` (`may_append(..., lookahead=)`).
        # 【译】对空闲池的需求：真正复用的block减去已经在被使用的（共享引用即可）；从命中里丢掉的block变成新block → 要计入。
        #   再加上草稿轮在prompt之后的lookahead，调度器在 allocate 之后立刻分配它（may_append(..., lookahead=)）。
        # (5) 要从空闲池拿多少块：总块数 n − 前num_cached_blocks个命中块里"正被别的序列用着"的块数（它们只需ref_count+1，不占空闲名额）。
        #   命中但躺在空闲表里的块仍要计入：claim 会把它从空闲表取出来（同nano-vllm的 `if block_id in self.used_block_ids`）。
        #   例（block 16，prompt 100 → n=7，命中4块）：4块都被正在运行的A占着 → 7 − 4 = 3；A已结束、4块都在空闲表里 → 7 − 0 = 7
        #   【分支】草稿lookahead（MTP k=3 → 多写 k−1 = 2 个位置）：prompt 96个token时要写到位置97 → 多1块；
        #     prompt 100个token时位置100、101仍在第7块里 → 多0块
        num_new_blocks = (
            self._n_hash_blocks(seq)
            - self.kv.count_used(walk.ids, num_cached_blocks)
            + self.draft_lookahead_blocks(len(seq))
        )
        # 命中边界那一块的hash。【分支】有PAGE checkpoint时，allocate要从这个hash名下的checkpoint恢复状态；下面的容量检查和
        #   demand的空间检查都把它排除在"可淘汰"之外，否则可能为了腾地方把自己马上要用的checkpoint淘汰掉
        protected_hash = (
            block_hashes[num_cached_blocks - 1] if num_cached_blocks else None
        )
        # After `num_new_blocks`, not before: the demand's room check has to
        # account for what this very admission is about to take, or it reads a
        # pool it then drains itself.
        #
        # `resumable`, not the replayed hit: the counterfactual is asked of the
        # gates, whose answer a replay does not change.
        # 【译】放在 num_new_blocks 之后而不是之前：demand的空间检查必须算上本次准入马上要拿走的block，否则它读到的池子随后会被自己耗掉。
        #
        #   传 resumable 而不是重放后的命中：反事实问题是问门控的，而重放不改变门控的回答。
        # (6) 【分支】checkpoint放置：demand = 门控截短命中时"如果这里有checkpoint，能命中到哪"（上面例子里的18432），
        #   prompt末尾锚点 = 本prompt末尾附近（多轮对话的下一轮最常从这里接上）。普通模型 demand = 0、end直接返回
        self._record_checkpoint_demand(
            seq,
            hit=resumable,
            compressed_hit=compressed_hit,
            block_hashes=block_hashes,
            live_blocks=num_new_blocks,
            protected_hash=protected_hash,
        )
        self._record_checkpoint_end(seq)
        # (7) 容量检查。没有PAGE checkpoint时就是"空闲block数 ≥ num_new_blocks"（空闲块包括仍带hash的缓存块，取用时才淘汰）；
        #   有PAGE checkpoint时，状态镜像也占KV池的单元：READY且没被钉住的checkpoint单元也算可用（取新块时按需淘汰），
        #   protected_hash那一份除外。对比nano-vllm：`if len(self.free_block_ids) < num_new_blocks: return -1` 那一行
        if not self._has_page_units(num_new_blocks, protected_hash):
            return -1
        # A direct admission commits its joint boundary here. The scheduler
        # probes first and calls record_allocation only after dependency and
        # budget checks; refused probes avoid the extra LMCache hash walk.
        # 【译】直接准入在这里提交它的联合边界。调度器则先探测，在依赖检查和预算检查之后才调用 record_allocation；
        #   被拒绝的探测因此省掉了额外那次LMCache hash遍历。
        # (8) record=True（默认，直接准入的路径，如RapidServe）立刻提交【分支】联合边界；主路径的 _probe_hbm_prefix 传 record=False
        if record:
            self.record_allocation(seq, num_cached_blocks, block_hashes)
        # After the refusal, not before it. The chain is O(prompt) xxhash plus
        # two temporaries per block, and a refused admission discards it — a
        # 128k prompt queued behind a full pool paid ~2000 rounds per waiting
        # request per scheduling pass for a list nobody read. Nothing above
        # consumes it: `_record_checkpoint_demand` takes `block_hashes` as an
        # argument, and `_record_checkpoint_end` derives its position from
        # `num_prompt_tokens` alone. Its one reader is `midstep_positions`.
        #
        # Free unless a backend declares `readable_midstep`: the call returns
        # at its first line otherwise. Qwen GDN does under
        # `ATOM_GDN_STATE_MIDSTEP=1`, and then every admission pays it.
        # 【译】放在拒绝之后，而不是之前。这条链是 O(prompt长度) 次xxhash外加每块两个临时对象，被拒绝的准入会把它丢掉——
        #   一个排在满池子后面的128k prompt，每个调度轮次、每个等待中的请求都要付约2000轮，算出一个没人读的列表。上面没有
        #   任何地方用它：_record_checkpoint_demand 把 block_hashes 当参数接收，_record_checkpoint_end 只从 num_prompt_tokens
        #   推出位置。它唯一的读者是 midstep_positions。
        #
        #   除非后端声明了 readable_midstep，否则这一步不花钱：调用在第一行就返回。Qwen GDN 在 ATOM_GDN_STATE_MIDSTEP=1 时
        #   声明了它，那时每次准入都要付这笔开销。
        # 【分支】只有midstep checkpoint（在一次forward中途取状态快照，见本文件 midstep_positions 一带）需要整条prompt的hash链，
        #   写到 seq.block_hashes；其余情况把 seq.block_hashes 清空就返回
        self._extend_hash_chain(seq, block_hashes)
        # (9) 返回门控 / 重放之后的命中块数；调度器据此算 resume = num_cached_blocks × hbs（例：4块 → 64）
        return num_cached_blocks

    def _walk_prefix(self, seq: Sequence) -> _PrefixWalk:
        """`seq`'s contiguous HBM prefix hit, the costly half of `can_allocate`."""
        # 【译】seq在HBM里连续命中的前缀，can_allocate中开销大的那一半。
        # "HBM"指GPU显存里的KV池（与【分支】CPU / LMCache那一层相对）。两种实现，结果相同：
        #   快速遍历 _hit_walk（ATOM_PREFIX_FAST_WALK，默认开）：批量查表，只比对最后一个命中块的token
        #   逐块遍历（关掉快速遍历时）：下面的for循环，和nano-vllm的can_allocate循环一样（每块都比对token）
        # prompt = 本seq的prompt hash链（_prompt_hashes：每个seq只算一次；默认用前端算好送来的链，核对后采用）
        prompt = self._prompt_hashes(seq)
        if self.fast_prefix_walk:
            hashes, ids = self._hit_walk(seq, prompt)
        else:
            # h从cache_seed开始：纯文本为-1（第0块不拼前缀）；多模态是对图像数据算出的种子，同样的文字配不同的图不会误命中
            hashes, ids, h = [], [], seq.cache_seed
            # 只看前 n−1 块：最后一块不复用（见 can_allocate 里的例子）
            for i in range(self._n_hash_blocks(seq) - 1):
                # 链上已有就直接读，没有就现算，并追加到链上（只追加prompt的满块）
                h = self._block_hash(seq, prompt, i, h)
                block_id = self.kv.lookup(h)
                # 查不到，或hash碰撞（索引里那块的token和本块不同）→ 停：链式hash下后面的块也不可能命中
                if block_id == -1 or self.kv.block(block_id).token_ids != (
                    self._hash_block_tokens(seq, i)
                ):
                    break
                hashes.append(h)
                ids.append(block_id)
        # 连同"在什么条件下遍历的"一起返回：token数、种子、索引版本号——allocate（_take_admission_walk）据此判断能否直接用这些id
        return _PrefixWalk(
            seq.num_tokens, seq.cache_seed, self.kv.index_version, hashes, ids
        )

    def _admission_walk(self, seq: Sequence) -> _PrefixWalk:
        """The exact probe's walk, kept on the fast walk for `seq`'s `allocate`
        (`_take_admission_walk`)."""
        # 【译】精确探测的遍历结果；快速遍历模式下把它存起来，留给这条seq的 allocate 使用（_take_admission_walk）。
        # 只在快速遍历时保存：allocate 直接 claim_many 这些id，省掉第二遍"算hash → 查表"（nano-vllm的allocate会把命中块
        #   重新hash、重新查一遍）；关掉快速遍历时 allocate 逐块重查。_admission_walks 是 WeakKeyDictionary（以seq为弱引用键，
        #   seq被回收时条目自动消失），每次精确探测覆盖、allocate 时取走
        walk = self._walk_prefix(seq)
        if self.fast_prefix_walk:
            self._admission_walks[seq] = walk
        return walk

    def _hit_walk(self, seq: Sequence, prompt: list[int]) -> tuple[list, list]:
        """`(hashes, block ids)` of `seq`'s contiguous HBM prefix hit.

        Looks the cached prompt chain up with `map` in rounds growing from the
        first block alone (`_WALK_LOOKUP_ROUND_MIN`), so a cold prompt costs
        one lookup, and hashes ahead in rounds past it. Tokens are compared on
        the last hit block only: indexed contents cannot change (`publish`
        revokes a block's old hash, `allocate` unindexes first), so a mismatch
        is a 64-bit hash collision,
        and then every block is compared as the per-block walk did. A collision
        on an interior block alone goes undetected (vLLM compares none).
        """
        # 【译】seq在HBM里连续命中的前缀，返回 (hash列表, block id列表)。
        #
        #   对已缓存的prompt链用 map 分轮查表，每轮的块数从只查第一块开始增长（_WALK_LOOKUP_ROUND_MIN），所以一个冷prompt
        #   只花一次查表；越过缓存链之后，再分轮往前算hash。只比对最后一个命中块的token：已登记的内容不会改变（publish 会撤销
        #   block的旧hash，allocate 会先取消登记），所以比对不上只可能是64位hash碰撞，这时再像逐块遍历那样逐块比对。
        #   只发生在中间某块上的碰撞检测不到（vLLM一块都不比对）。
        # 分轮的例子（block 16）：
        #   查表轮（链已缓存：前端送来的，或之前算过的）：每轮查 1、16、64、256、1024、4096…块（第一轮只查1块，之后每轮 ×4、至少16）
        #     冷prompt：第1轮查第0块就未命中 → 只查1次
        #     128k prompt（8192块）命中3000块：前5轮累计查1361块全中；第6轮查4096块，在第3000块处未命中 → 共查5457次。
        #     每轮是一次 map(dict.get, …)，在C层循环，比Python逐块循环快得多；未命中之后多查的部分不超过命中的约4倍
        #   算hash轮（链没覆盖到的部分）：每轮算 64、128、256、512、1024、1024…块（chained_block_hashes 一次算一段），
        #     算出的若是prompt的满块就顺手追加到链上，下次不必再算；未命中之后白算的hash最多一轮
        # 碰撞概率与"为什么只比最后一块"的讨论见prefix_hash.py文件头【科普】③
        # 最后一块不复用：只遍历前 n−1 块
        n = self._n_hash_blocks(seq) - 1
        # 索引dict的get方法本身（查不到返回None），配合 map 在C层批量查表
        get = self.kv.index_get
        hashes: list[int] = []
        ids: list = []
        hash_round = _WALK_HASH_ROUND_MIN
        look = 1
        while len(hashes) < n:
            k = len(hashes)
            # 链上还有现成的hash：取下一轮的 look 个去查，然后把 look 放大到 max(4×look, 16)
            if k < len(prompt):
                new = prompt[k : min(n, k + look)]
                look = max(4 * look, _WALK_LOOKUP_ROUND_MIN)
            # 链用完了（前端没送链 / 链被拒，或已越过prompt的满块）：现算下一段hash，从上一块的hash（第0块则从cache_seed）接着链
            else:
                end = min(n, k + hash_round)
                new = chained_block_hashes(
                    seq.token_ids,
                    self.hash_block_size,
                    k,
                    end,
                    hashes[-1] if k else seq.cache_seed,
                )
                if k == len(prompt):
                    # Prompt blocks only: those past it can still change.
                    # 【译】只追加prompt的块：prompt之后的块还可能变。
                    # 链缓存只承诺"被prompt填满的块"这一段不变（prefix_hash.full_prompt_blocks；含生成token的块可能还挂着占位token，
                    #   见prefix_hash.py文件头【科普】⑤）。要 k == len(prompt) 才追加：保证新算的这段正好接在链的末尾
                    full = full_prompt_blocks(seq, self.hash_block_size)
                    prompt += new[: max(0, full - k)]
                hash_round = min(2 * hash_round, _WALK_HASH_ROUND_MAX)
            new_ids = list(map(get, new))
            try:
                # 本轮第一个未命中的位置
                miss = new_ids.index(None)
            except ValueError:
                # 本轮全部命中（list.index找不到时抛ValueError）：收下，继续下一轮
                hashes += new
                ids += new_ids
                continue
            hashes += new[:miss]
            ids += new_ids[:miss]
            break
        # 只比对最后一个命中块的token（理由见docstring）；对不上说明发生了hash碰撞，退回逐块比对，截在第一个不一致处
        last = len(ids) - 1
        if last >= 0 and self.kv.block(ids[last]).token_ids != (
            self._hash_block_tokens(seq, last)
        ):
            for i, block_id in enumerate(ids):
                if self.kv.block(block_id).token_ids != self._hash_block_tokens(seq, i):
                    del hashes[i:], ids[i:]
                    break
        return hashes, ids

    def _take_admission_walk(self, seq: Sequence, blocks: int) -> _PrefixWalk | None:
        """`seq`'s last fast walk if its first `blocks` ids, all `allocate`
        claims, still hold (consumed either way). Past its `index_version` they
        hold while their hashes still resolve to the same ids."""
        # 【译】如果seq上一次快速遍历的前 blocks 个id（即 allocate 要认领的全部）仍然有效，就返回它（无论有效与否都会被取走）。
        #   索引版本号变过之后，只要这些hash仍解析到同样的id，它们就仍然有效。
        # 为什么要验证——can_allocate 与 allocate 之间可能变了什么：
        #   BlockPool.index_version 在"某个hash离开索引或改指向另一个block"时+1（块被重新分配而淘汰、clear_cache、同一hash
        #   改登记到别的块）；新登记一个hash不+1（已经查到的id仍然对）。所以：
        #     版本号没变 → 这批id一定还对，O(1)放行；
        #     变了 → 用map把前k个hash重查一遍，与当时的id完全一致才放行，否则返回None，allocate 逐块重查（查不到就截断命中）
        #   主路径上同一条seq的探测和allocate紧挨着（中间只有预算 / chunk检查，不动池子），版本号通常不变；但这份结果存在
        #   字典里，可能来自更早的探测，也可能经过中间动过池子的路径，所以总要验证一次。token数或cache_seed变了（例如被抢占后
        #   带着已生成的token重新排队）则整份作废
        walk = self._admission_walks.pop(seq, None)
        if (
            walk is None
            or walk.num_tokens != seq.num_tokens
            or walk.cache_seed != seq.cache_seed
        ):
            return None
        # 只验证 allocate 真要认领的前k块
        k = min(blocks, len(walk.ids))
        if walk.index_version != self.kv.index_version and (
            list(map(self.kv.index_get, walk.hashes[:k])) != walk.ids[:k]
        ):
            return None
        return walk

    def can_allocate_memoized(self, seq: Sequence) -> int:
        """`can_allocate(seq, record=False)` for the delayer and the DP chunk
        cost, with the walk memoised per seq. The same `_evaluate`, so state
        room, the gates, the replay, checkpoint demand/end, the used hit blocks
        and the fit are re-asked on every call, and the answer is the exact
        probe's.

        The walk is reused while nothing it read moved: no `watched` hash was
        evicted or remapped to another block (`_note_index_change`; more than
        `_RECENT_INDEX_CHANGES` since the last check re-walks), no
        `clear_cache`, the first miss still resolves the same (a hit only grows
        by a publish of that hash), and token count and seed are unchanged.
        """
        # 【译】给PrefillDelayer和DP chunk代价估计用的 can_allocate(seq, record=False)，前缀遍历按seq缓存。走的是同一个 _evaluate，
        #   所以状态槽位空间、门控、重放、checkpoint的demand/end、命中块里正被使用的块数，以及放不放得下，每次调用都重新询问，
        #   答案与精确探测一致。
        #
        #   只要遍历读过的东西都没动，就复用这次遍历：watched 中没有hash被淘汰或改指向别的block（_note_index_change；自上次
        #   检查以来变化超过 _RECENT_INDEX_CHANGES 个就重新遍历），没有 clear_cache，第一个未命中块的hash仍解析成同样的结果
        #   （命中只会因为登记了这个hash而变长），token数和种子也没变。
        # 为什么要跨tick缓存：调用方（scheduler.py的 _probe_local_prefill_work、_estimate_prefill_chunk_cost、_can_admit_head_prefill）
        #   每个tick都要探一遍waiting队列，而PrefillDelayer为了攒批，可能让同一批请求在waiting里等好几个tick。一个128k的prompt
        #   是8192块，每tick重走一遍链就是成千上万次查表；缓存后每次只检查"自上次以来索引里变了哪些hash"，代价与索引变化量成正比、
        #   与prompt长度无关
        # tick之间可能变的：
        #   别的请求prefill完、hash_blocks登记了新hash → 命中可能变长（只有"第一个未命中块"的hash被登记时才会，所以只盯这一个）
        #   别的请求取新块淘汰了缓存块 → 命中可能变短（watched里的某个hash离开索引）
        #   clear_cache；本seq被抢占后token变多（num_tokens变了）
        # 只缓存遍历；"命中块里几块正被使用"等随池子变化的量仍每次重算（见 _evaluate）。ATOM_PREFIX_WALK_MEMO=0 时不缓存
        # 对比nano-vllm：没有这一层（nano-vllm没有delayer，每步只对真正要准入的请求调一次can_allocate）
        walk_of = self._memoised_walk if self.walk_memo else self._walk_prefix
        return self._evaluate(seq, walk_of, False, None)

    def _memoised_walk(self, seq: Sequence) -> _PrefixWalk:
        # 有缓存且仍有效 → 直接用；否则重新遍历一次，并记下以后用来验证的东西
        memo = self._cached_walks.get(seq)
        if memo is not None and self._walk_still_valid(seq, memo):
            return memo.walk
        walk = self._walk_prefix(seq)
        hit = len(walk.hashes)
        boundary_hash = None
        boundary_id = -1
        # 命中没走到最后一个可复用块（第n−2块）：记下"第一个未命中块"的hash和它此刻查到的id（通常是-1；hash碰撞时是那个
        #   token对不上的块）。以后它一旦被登记或改指，命中就可能变长 → 缓存作废。命中已到最后一个可复用块时不会再变长，记None
        if hit < self._n_hash_blocks(seq) - 1:
            # Read back from the prompt chain, where the walk left it.
            # 【译】从prompt链里读回来，遍历时已经把它留在那里了。
            prev = walk.hashes[-1] if hit else seq.cache_seed
            boundary_hash = self._block_hash(seq, self._prompt_hashes(seq), hit, prev)
            boundary_id = self.kv.lookup(boundary_hash)
        # watched = 这次遍历读过的全部索引项：命中的各块 + 第一个未命中块
        watched = set(walk.hashes)
        if boundary_hash is not None:
            watched.add(boundary_hash)
        # 同时记下此刻的 clear 次数和索引变化计数，作为下次 _walk_still_valid 的起点
        self._cached_walks[seq] = _CachedWalk(
            walk=walk,
            index_clears=self._index_clears,
            index_changes_seen=self._index_changes,
            watched=watched,
            boundary_hash=boundary_hash,
            boundary_id=boundary_id,
        )
        return walk

    def _walk_still_valid(self, seq: Sequence, memo: _CachedWalk) -> bool:
        """Whether the KV index still answers `memo`'s walk the same way. Costs
        the index changes since the last check, not the prompt length."""
        # 【译】KV索引对 memo 那次遍历的回答是否仍然一样。代价是自上次检查以来的索引变化数，而不是prompt长度。
        # 例：memo之后索引又变了30次（30个别的hash被淘汰），逐条看这30个是否在watched里，都不在 → 仍有效；
        #   代价是30次集合查询，而不是把几千块重走一遍
        # token数、种子变了，或其间 clear_cache 过 → 作废
        if (
            memo.walk.num_tokens != seq.num_tokens
            or memo.walk.cache_seed != seq.cache_seed
            or memo.index_clears != self._index_clears
        ):
            return False
        # 自上次检查以来新增的索引变化条数（_note_index_change：hash被淘汰、或改指向另一个block时各记一条）
        new = self._index_changes - memo.index_changes_seen
        if new:
            # 变化太多，最近变化的环形队列（deque(maxlen=8192)）已装不下 → 无法逐条核对，保守地作废、重新遍历
            if new > len(self._recent_index_changes):
                return False
            watched = memo.watched
            # 只看最近 new 条（islice(it, n)：取迭代器的前n个，不复制）：有一条落在watched里 → 遍历读过的某项变了 → 作废
            for h in islice(reversed(self._recent_index_changes), new):
                if h in watched:
                    return False
            # 都与本遍历无关：把检查起点推进到现在，下次只看更新的变化
            memo.index_changes_seen = self._index_changes
        # 最后看第一个未命中块：它的hash现在查到的id与当时相同（通常仍是-1），命中就没有变长
        return (
            memo.boundary_hash is None
            or self.kv.lookup(memo.boundary_hash) == memo.boundary_id
        )

    def record_allocation(
        self, seq: Sequence, num_cached_blocks: int, block_hashes: list[int]
    ) -> None:
        """Commit the fit probe before any allocation changes the pool."""
        # 【译】在任何分配改动池子之前，提交这次"放得下"的探测结果。
        # 【分支】联合边界（LMCache offload，有每请求状态的模型，第9批）：HBM里的前缀在第一次未命中处就停了，但LMCache可能还存着
        #   后面的KV和状态checkpoint；_joint_kv_boundary 找一个"KV和状态都能从LMCache取回"的更远边界，_commit_joint_boundary
        #   把它写进 seq.offload_joint（allocate 据 claim_tokens 多认领、据 boundary_hash 找状态）。没有状态层时（state_offload
        #   为None，普通部署都是）只记一次 joint_skips["off"] 并清空 seq.offload_joint 的联合字段
        # 为什么单独一步、放在allocate之前：边界要对着"分配前"的池子算（allocate取新块会淘汰缓存）；调度器在预算检查通过后才调它，
        #   被拒绝的探测就不必付这次计算
        if self.enable_prefix_caching:
            decision = self._joint_kv_boundary(seq, num_cached_blocks, block_hashes)
            self._commit_joint_boundary(seq, decision)

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> bool:
        """Allocate blocks for `seq`. `num_cached_blocks` is the hit count
        returned by `can_allocate` (0 if caller didn't call it).

        Hash registration is deferred to hash_blocks(), called from
        scheduler.postprocess() once the forward has computed each block's
        KV. This keeps the manager correct under future chunked-prefill
        scheduling: a block spanning multiple steps must not be published as
        a hash until fully filled.

        Returns ``True`` on success. Returns ``False`` only when a state-less
        joint boundary had to be disowned (finding #2) and the pool could not
        back the private prefix copies; the caller must then deallocate and
        requeue the seq rather than run a forward over shared blocks.
        """
        # 【译】为seq分配block。num_cached_blocks 是 can_allocate 返回的命中数（调用方没调它时为0）。
        #
        #   hash登记推迟到 hash_blocks()，由 scheduler.postprocess() 在forward算完每个block的KV之后调用。这让manager在将来的
        #   chunked prefill调度下也保持正确：一个跨越多步才写满的block，在完全填满之前不能以hash的形式发布。
        #
        #   成功返回 True。只有当一个没有状态支撑的联合边界必须被放弃（finding #2），而池子又给不出前缀的私有副本时才返回 False；
        #   这时调用方必须 deallocate 并把seq放回队列，而不是在共享block上跑forward。
        # 步骤编号对应 can_allocate 里第二个流程图的 (1)~(9)
        # 观察：docstring说"将来的chunked prefill"，而chunked prefill现在已默认开启（arg_utils.py enable_chunked_prefill=True），
        #   这条"写满才登记"的规则正是它所依赖的
        # 观察：返回False的实际条件是"state_holds为False 且 disown_claimed_prefix失败"。state_holds为False不只发生在联合边界上：
        #   普通HBM命中背后的checkpoint在探测之后没了（_attach_state_slots 里PAGE分支的"Neither tier can produce it"、fork分支的
        #   `return hit_hash == -1`）也会走到这里，比docstring说的范围宽
        # 调用方：scheduler.py _schedule Phase 2（返回False时 deallocate + 放回waiting队头 + 本轮停止准入）；
        #   【分支】RapidServe的 DecodeScheduler.allocate_waiting（不传 num_cached_blocks，即不复用前缀）
        assert not seq.block_table
        # Admitted: the delayer no longer probes this seq.
        # 【译】已准入：delayer不会再探测这条seq。
        # 所以丢掉它的跨tick遍历缓存（can_allocate_memoized 用的 _cached_walks）
        self._cached_walks.pop(seq, None)
        # Two extents, and they are not the same number. `num_cached_blocks` is
        # what the request may call cached -- it needs a resumable state behind
        # it. `claim_blocks` is what it may point its block table at: every
        # block the prefix walk matched, whether or not a checkpoint sits there.
        # The gap between them is real reuse that the state gate declined, and
        # taking it costs nothing: those blocks are below the joint boundary the
        # forward will start from, so nobody writes them.
        #
        # Only a joint boundary widens it -- without one there is nothing above
        # `num_cached_blocks` this request will ever treat as computed, and
        # claiming further would pin blocks the forward is about to overwrite.
        # 【译】两个范围，而且它们不是同一个数。num_cached_blocks 是请求可以称为"已缓存"的部分——背后需要一份可恢复的状态。
        #   claim_blocks 是它的block table可以指向的部分：前缀遍历匹配上的每个block，不论那里有没有checkpoint。两者之差是被状态
        #   门控拒绝的真实复用，拿下它们不花任何代价：这些block在forward将要开始的联合边界之下，没有人会写它们。
        #
        #   只有联合边界会放宽它——没有联合边界时，num_cached_blocks 之上没有任何东西会被这个请求当作已计算，认领更多只会
        #   钉住forward马上要覆盖的block。
        # (1) 普通情况 claim_blocks == num_cached_blocks；只有【分支】联合边界（record_allocation 写了 claim_tokens）才更大
        hbs = self._hash_block_size()
        # 记下调用方要的命中块数：下面若因淘汰截短了命中，【分支】V4重放据此判断
        requested_blocks = num_cached_blocks
        claim_blocks = max(
            num_cached_blocks,
            int(seq.offload_joint.claim_tokens or 0) // hbs,
        )
        # h：链上"上一块"的hash，从cache_seed开始；hit_hash：最后一个"算作已缓存"的块的hash，-1表示冷启动——
        #   【分支】状态槽位按它找checkpoint（普通模型用不到）
        h = seq.cache_seed
        hit_hash = -1
        # Claim the probe's ids as one run; the loop walks only the rest (a
        # joint tail past the HBM hit, or everything if the walk is stale).
        # 【译】把探测得到的id当作一整段认领；下面的循环只走剩下的部分（HBM命中之后的联合尾段，或walk已失效时的全部）。
        claimed = 0
        walk = self._take_admission_walk(seq, claim_blocks)
        # (2) `:=`（海象运算符）：在表达式里赋值并取值。claim_many：已被别的序列用着的块 ref_count+1；在空闲表里的块从空闲表
        #   取出、ref_count=1，内容和hash原样保留（BlockPool.claim；对比nano-vllm allocate里 `if block_id in self.used_block_ids`
        #   的两个分支）。例：can_allocate 里例子(a)的B → claim_many([0,1,2,3])，block_table = [0,1,2,3]
        if walk is not None and (claimed := min(claim_blocks, len(walk.ids))):
            ids = walk.ids[:claimed]
            self.kv.claim_many(ids)
            seq.block_table.extend(ids)
            # 下面的循环（如果还有要认领的）从第claimed块接着链；hit_hash = 已缓存部分最后一块的hash
            h = walk.hashes[claimed - 1]
            if cached := min(claimed, num_cached_blocks):
                hit_hash = walk.hashes[cached - 1]
        # 逐块认领剩下的：walk失效时就是全部命中块（和nano-vllm的allocate一样：算hash、查表、认领），否则只有联合尾段
        prompt = self._prompt_hashes(seq)
        for i in range(claimed, claim_blocks):
            h = self._block_hash(seq, prompt, i, h)
            block_id = self.kv.lookup(h)
            if block_id == -1:
                # Evicted between `can_allocate` and here. For the widened joint
                # tail this is expected -- LMCache or the state tier still holds
                # it and the forward refetches. For the cached range it should be
                # unreachable (nothing evicts inside one admission), but an
                # `assert i >= num_cached_blocks` vanishes under `python -O`, and
                # the old fall-through then recorded `num_cached_blocks * hbs` as
                # cached over a prefix it could not claim, left `hit_hash`
                # unassigned (cold-start state exit), and clamped only
                # `claim_tokens` while `boundary`/`kv` stayed above
                # the claimed region -- silent wrong output on both counts. Use
                # real control flow instead: never treat more than the `i` blocks
                # actually claimed as cached, and drop any joint boundary that
                # sits above them so the KV leg cannot resume over a gap.
                # 【译】在 can_allocate 和这里之间被淘汰了。对放宽出来的联合尾段这是预料之中的——LMCache或状态层仍持有它，forward会重新取。
                #   对已缓存的范围这本应不可达（一次准入内部不会淘汰任何东西），但 `assert i >= num_cached_blocks` 在 python -O 下会消失，
                #   旧代码于是一路执行下去：把 num_cached_blocks * hbs 记为已缓存、覆盖一个它没能认领的前缀，让 hit_hash 保持未赋值
                #   （走冷启动的状态出口），并且只钳制了 claim_tokens，而 boundary / kv 仍停在认领区域之上——两方面都是静默的错误输出。
                #   改用真正的控制流：已缓存的部分绝不超过实际认领到的 i 个block，并丢弃位于它们之上的任何联合边界，这样KV那条腿就
                #   不会越过一个缺口去恢复。
                # 命中截到实际认领到的 i 块
                claimed_tokens = i * hbs
                num_cached_blocks = min(num_cached_blocks, i)
                # Partial reset by design: the claim is clamped, not zeroed,
                # so this cannot go through `reset_joint`.
                # 【译】有意只做部分重置：认领范围是被钳制而不是清零，所以这里不能走 reset_joint。
                if int(seq.offload_joint.boundary_tokens or 0) > claimed_tokens:
                    seq.offload_joint.boundary_tokens = 0
                    seq.offload_joint.boundary_hash = -1
                    seq.offload_joint.kv_tokens = 0
                seq.offload_joint.claim_tokens = min(
                    int(seq.offload_joint.claim_tokens or 0),
                    claimed_tokens,
                )
                break
            self.kv.claim(block_id)
            seq.block_table.append(block_id)
            # Track the hash of the last claimed *cached* block as we go, not
            # only when the loop reaches num_cached_blocks-1. If the block_id==-1
            # branch above clamps num_cached_blocks to an earlier `i` (an
            # eviction inside the cached range), the `i + 1 == num_cached_blocks`
            # test never fired, leaving hit_hash == -1 while num_cached_tokens
            # stayed > 0 -- a cold-start state restore over a prefix that is in
            # fact cached. Assigning on every cached block leaves hit_hash on the
            # last one actually claimed, whichever way the loop exits.
            # 【译】一边走一边记下最后一个被认领的"已缓存"block的hash，而不是只在循环走到 num_cached_blocks-1 时才记。如果上面
            #   block_id==-1 的分支把 num_cached_blocks 钳到更早的 i（已缓存范围内发生了淘汰），`i + 1 == num_cached_blocks` 这个判断
            #   就永远不会触发，hit_hash 停在 -1 而 num_cached_tokens 仍 > 0——在一个实际已缓存的前缀上做冷启动的状态恢复。
            #   每个已缓存block都赋值一次，无论循环怎么退出，hit_hash 都停在最后一个真正认领到的block上。
            if i < num_cached_blocks:
                hit_hash = h
        if self.state_replay_tokens and seq.replays_hit:
            # A replay starts from a fresh state slot, never a checkpoint: it
            # rebuilds the state below whatever the claim above reached.
            # 【译】重放总是从一个新的状态槽位开始，从不从checkpoint开始：它在上面认领到的位置之下重建状态。
            # (3) 【分支】DeepSeek-V4有界重放（_replay_hit 判定重放时）：不找checkpoint，hit_hash 置-1 → 下面拿新槽位
            hit_hash = -1
            if num_cached_blocks < requested_blocks:
                # A KV-only load past the hit would start above a gap.
                # 【译】越过命中的"只加载KV"会从一个缺口之上开始。
                seq.replay_kv_load = False
        # Pin the restore before fresh blocks can evict its checkpoint.
        # 【译】在新block可能淘汰checkpoint之前，先把恢复钉住。
        # (4) 【分支】PAGE checkpoint的状态镜像就住在KV池的单元里，下面 _fresh_block 取新块时可能按LRU淘汰它；
        #   先 begin_restore（checkpoint的pin_count+1，不再是可淘汰对象），它就安全了
        state_holds = True
        if seq.has_per_req_cache and self.paged_state_checkpoints is not None:
            # The joint boundary, when there is one -- the same rule the fork
            # branch below already used, and the PAGE branch did not.
            #
            # `hit_hash` is the hash at the HBM hit, and a joint boundary sits
            # strictly above it by construction. Two ways that went wrong here:
            # with `num_cached_blocks == 0` the loop never assigned `hit_hash`
            # at all, so this took the `-1` cold-start exit and requested no
            # state whatsoever; with a non-zero hit it restored the checkpoint
            # covering `[0, hbm)`. Either way the KV leg then loaded to the
            # joint boundary and `_claim_after_load` raised `num_cached_tokens`
            # to it, so the forward resumed over a prefix the state does not
            # cover. Silent wrong output, no exception.
            # 【译】有联合边界时用联合边界——这正是下面fork分支早已采用、而PAGE分支以前没有采用的规则。
            #
            #   hit_hash 是HBM命中处的hash，而联合边界按构造严格位于它之上。这里出过两种错：num_cached_blocks == 0 时循环根本没有
            #   给 hit_hash 赋值，于是走了 -1 的冷启动出口，完全没有请求状态；命中非零时，它恢复的是覆盖 [0, hbm) 的那个checkpoint。
            #   无论哪种，KV那条腿随后都加载到联合边界，_claim_after_load 把 num_cached_tokens 提升到那里，于是forward在一个状态
            #   并不覆盖的前缀上恢复。静默的错误输出，没有异常。
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
        # (5) 补齐新块：从 len(block_table) 补到整条序列（len(seq) = num_tokens 个token）所需的本rank block数——和nano-vllm一样，
        #   准入时一次分配整条prompt（chunked prefill的后续chunk不再分配）；草稿lookahead由调度器随后另外分配
        #   （_allocate_prefill_draft_lookahead）。_fresh_block 从空闲池取：先给没有hash的空块（id小的先），没有了才给最久没用的
        #   缓存块，并在这一刻淘汰它的hash（惰性淘汰；淘汰回调 _record_evicted 会记KV事件、并作废同一hash名下的状态checkpoint）
        #   例：例子(a)的B补3块 → 7、8、9；例子(b)的B补3块 → 6、7、8
        for _ in range(len(seq.block_table), self.num_pool_blocks(len(seq))):
            seq.block_table.append(self._fresh_block())
        # (6) 复用了多少token，prefill从这里开始（【分支】有状态的模型可能再往前退，见 replay_start）。例：命中4块 → 64
        #   对比nano-vllm：`seq.num_cached_tokens = num_cached_blocks * self.block_size`；ATOM乘的是hbs（开DCP时 = block_size × DCP）
        seq.num_cached_tokens = num_cached_blocks * hbs

        # Per-request cache: claim this seq's slot indices from the
        # pre-allocated state tensor (e.g. GDN mamba_k_cache, the V4 compressor
        # ring). The state class took its bytes before the paged class was sized
        # in ModelRunner.get_num_blocks(), so admitting a seq adds no further
        # paged-block cost. The state pool's free list is the sole admission
        # bound for state cache.
        # 【译】每请求缓存：从预先分配好的状态张量里为这条seq认领槽位编号（例如GDN的 mamba_k_cache、V4的压缩器环形缓冲）。
        #   状态这一类在 ModelRunner.get_num_blocks() 给分页类定规模之前就拿走了自己的字节，所以准入一条seq不会再增加任何
        #   分页block的开销。状态池的空闲表是状态缓存唯一的准入上限。
        # (7) 【分支】fork类后端（checkpoint是状态池里的空闲槽位，不在KV池里，取新块不会淘汰它），所以放在取完新块之后
        if seq.has_per_req_cache and self.paged_state_checkpoints is None:
            # A joint load aims the state leg at its own boundary, which is
            # above the HBM hit by construction. `num_cached_tokens` stays at
            # the HBM prefix until the KV leg lands, so the forward covers
            # `[hbm, num_prompt)` if anything goes wrong from here.
            # 【译】联合加载让状态这条腿瞄准它自己的边界，而那个边界按构造就在HBM命中之上。KV那条腿送达之前，num_cached_tokens
            #   一直停在HBM前缀处，所以从这里开始无论哪里出了问题，forward都会覆盖 [hbm, num_prompt)。
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
        # 标记"本次allocate之后状态还没初始化"：【分支】offload连接器（kv_transfer/offload/hybrid/dsv4/connector.py）读它；
        #   调度器在这次prefill的forward返回之后把它置True（scheduler.py里 `_state_initialized_after_alloc = True` 处）
        if seq.has_per_req_cache:
            seq._state_initialized_after_alloc = False
        # A claimed joint boundary must have a state leg behind it. The two
        # decisions are made in different places -- `can_allocate` picks the
        # boundary, `_attach_state_slots` secures the state -- and every way
        # they have disagreed so far ends the same way: the KV leg loads to the
        # boundary, `_claim_after_load` claims it, and the forward runs over a
        # prefix no state covers. This is the backstop that turns any such
        # disagreement into a recompute instead, and it is deliberately a
        # separate check from `state_holds` rather than folded into it: it
        # holds even if the code above stops asking the right question.
        # 【译】一个被认领的联合边界背后必须有状态那条腿。两个决定是在不同地方做出的——can_allocate 选边界，_attach_state_slots
        #   保证状态——而迄今为止它们每一种不一致的结局都一样：KV那条腿加载到边界，_claim_after_load 认领它，forward在一个
        #   没有任何状态覆盖的前缀上运行。这是把任何此类不一致转成一次重算的兜底，而且有意作为一个独立于 state_holds 的检查，
        #   而不是并进去：即使上面的代码不再问对问题，它依然成立。
        if (
            state_holds
            and seq.has_per_req_cache
            and seq.offload_joint.boundary_hash != -1
            and not self._state_leg_secured(seq)
        ):
            self.state_gate_lost_boundary += 1
            logger.warning(
                "state offload: a joint boundary was admitted for request %s "
                "with no state restore or load behind it; disowning it. This "
                "is a bug in the joint gate, not a cache miss.",
                seq.id,
            )
            state_holds = False
        if not state_holds:
            # No state behind the boundary means it is not this request's
            # history. Disown it -- the forward recomputes from 0. Privatise the
            # claimed prefix first (finding #2): recomputing over the shared
            # canonical blocks would tear another sequence's decode. Keeping the
            # boundary is silent wrong output too, since `has_initial_state`
            # (`gdn_attn.py`) is `num_cached_tokens > 0`.
            # 【译】边界背后没有状态，就说明它不是这个请求的历史。放弃它——forward从0重算。先把认领的前缀私有化（finding #2）：
            #   在共享的规范block上重算，会撕坏另一条序列的decode。保留边界同样是静默的错误输出，因为 has_initial_state
            #   （gdn_attn.py）就是 num_cached_tokens > 0。
            # (8) 例（GDN）：命中1024块、本该从16384处的checkpoint恢复，但这份checkpoint在探测之后已不在 → state_holds=False →
            #   disown_claimed_prefix 把认领的共享前缀块逐个换成私有新块（ref_count>1 的块要多占一块），num_cached_tokens=0，从0重算
            if not self.disown_claimed_prefix(seq):
                # Pool cannot back the private copies; signal the caller to
                # deallocate and requeue for a clean recompute next pass.
                # 【译】池子给不出私有副本；通知调用方 deallocate 并重新排队，下一轮干净地重算。
                return False
            seq.num_cached_tokens = 0
            # Clears only the boundary here, as the pre-record code did -- a
            # partial reset, not a full `reset_joint`.
            # 【译】这里只清边界，和拆出record之前的代码一样——是部分重置，不是完整的 reset_joint。
            seq.offload_joint.boundary_tokens = 0
            seq.offload_joint.boundary_hash = -1
        return True

    def replay_start(self, seq: Sequence, start: int, *, hit: bool) -> int:
        """Where a prefill resuming at `start` begins, early enough to rebuild
        what its forward reads below `start`: under a `replay` state transfer
        and at a `hit`, the state (one hash block); and in a forward finishing
        the prompt, `min_final_prefill_rows` rows before its end (decoder
        bounded replay), on the hash-block grid. `start` itself everywhere
        else. And at a hit `can_allocate` admitted with no checkpoint behind
        its state (`_replay_hit`, DeepSeek-V4), the state's receptive field,
        `state_replay_tokens`."""
        # 【译】一次在 start 处恢复的prefill从哪里开始——要足够早，才能重建它的forward在 start 之下要读的东西：在 replay 状态
        #   传输方式下、且是命中时，重建状态（往前一个hash block）；在完成整个prompt的那次forward里，从末尾往前
        #   min_final_prefill_rows 行开始（decoder有界重放），并对齐到hash block网格。其他情况就是 start 本身。另外，对
        #   can_allocate 准入的、状态背后没有checkpoint的命中（_replay_hit，DeepSeek-V4），往前退状态的感受野，即 state_replay_tokens。
        # 【分支】前置：为什么有状态的模型要"往前退"——见上面 can_allocate 里的【前置：每请求状态与状态checkpoint】：KV命中到N，
        #   状态却只存在于少数checkpoint处。GDN / KDA / DeepSeek-V4（PAGE或fork checkpoint）在准入时就把命中截到checkpoint
        #   （_gated_hit：KV命中18432、checkpoint阶梯8192 → num_cached_blocks对应16384），所以 start 已经是checkpoint位置，
        #   本函数原样返回16384。本函数只处理"不靠checkpoint、靠重算一小段来重建状态"的三种情况：
        #   ① replay 状态传输（DeepSeek-V4.1 encoder SWA有界重放，--encoder-swa-bounded-replay 默认开：deepseek_v41/backend.py
        #      的 state_transfer 返回 StateTransfer.replay()）：不存checkpoint、不门控（StateSlotPool.resumable_hit 对replay原样
        #      返回命中），命中时往前退一个hash block重算，重建128行的滑动窗口。
        #      例（V4.1 block 128，不支持DCP → hbs=128）：命中80块 → start=10240 → 从10112开始
        #   ② DeepSeek-V4有界重放（ATOM_DSV4_STATE_REPLAY；_replay_hit 判定"重放比从checkpoint接上省"时 seq.replays_hit=True）：
        #      往前退 state_replay_tokens。例：命中12800、state_replay_tokens=509 → 从12291开始（不重放就要从8192接上）
        #   ③ 完成prompt的那次forward至少跑 min_final_prefill_rows 行（DeepSeek-V4.1 decoder SWA有界重放，= sliding_window = 128；
        #      命中与否都检查）：剩下的不够就往前退，并向下对齐到hbs。
        #      例（续算partial prefill，hit=False）：prompt 10300、上一个chunk算到10240 → 只剩60行，差68 →
        #        (10240−68)//128×128 = 10112 → 最后一次forward跑188行
        #   退回去的那段KV本来就在cache里，重算只为重建状态；hash_blocks 不会把 replay_until 以下的块重新登记
        # 调用方：scheduler.py的准入（hit=True，start = 命中块数×hbs）、续算partial prefill（hit=False）、PrefillDelayer的探测等，
        #   都只算起点、不改池子
        # 对比nano-vllm：没有这一步（prefill总是从 num_cached_tokens 开始）
        # 没有每请求状态（普通模型），或从0开始：原样返回
        if not (seq.has_per_req_cache and start):
            return start
        hbs = self.hash_block_size
        # ①（self.state 是StateSlotPool；有PAGE checkpoint时它被建成 StateTransfer.none()，replays 为False）
        if hit and self.state.transfer.replays:
            start -= hbs
        # ②
        elif hit and self.state_replay_tokens and seq.replays_hit:
            start -= self.state_replay_tokens
        # ③ short = 还差多少行才够 min_final_prefill_rows（≤0 表示够了；min_final_prefill_rows=0 的模型恒 ≤0）
        short = self.min_final_prefill_rows - (seq.num_tokens - start)
        if short > 0:
            start = (start - short) // hbs * hbs
        return max(0, start)

    # ================ 【分支】每请求状态槽位与CPU状态层（B级；LMCache / offload部分见第9批） ================
    # 前置：每请求状态是什么、为什么要checkpoint，见上面 can_allocate 里的【前置：每请求状态与状态checkpoint】
    # 本组函数：_state_leg_secured、_attach_state_slots（allocate 调用，给槽位、从checkpoint恢复）；
    #   _tier_can_serve、_request_state_load、cancel_state_load、abandon_state_load、_release_oldest_orphan_slot、
    #   settle_state_load、take_state_loads（CPU层加载）；take_state_stores、settle_state_store、release_state_store_source、
    #   has_pending_state_store_pins、reclaim_stale_state_store_pins（CPU层存储）；reconcile_orphan_load_slots（孤儿槽位回收）
    #
    # 什么是"槽位"（slot）：ModelRunner启动时按 pool_entries["state"] 预先分配一个状态张量，槽位号就是它的下标；
    #   一个槽位 = 一条请求在所有状态层的一份完整状态（例：Qwen3-Next每个GDN层一个约52万个数的矩阵，外加conv状态）。
    #   一条请求拿 state_slots_per_req = 1 + 投机token数 个槽位：state_slots[0] 是已提交状态（每步读写它），
    #   [1:] 是投机解码的回滚槽位（MTP k=3 → [s0, r1, r2, r3]，见sequence.py has_per_req_cache处的科普），彼此不必相邻。
    #   空闲表（state_pool.py StateSlotPool）也是"先给空槽位（号小的先）、再按LRU花掉存着checkpoint的槽位"，
    #   和KV的BlockPool是同一套惰性淘汰
    #
    # checkpoint放在哪里——三种后端，由注意力后端的 state_transfer() 决定（state_runtime.py StateTransfer）：
    #   PAGE（copy）：checkpoint的状态镜像切成若干"PAGE单元"存进KV池（和KV block是同一个BlockPool，
    #     PagedStateCheckpointCoordinator 管理）；恢复 = 下一个batch之前把镜像从这些单元gather到请求自己的新槽位（begin_restore 排队）。
    #     用于Qwen3-Next / Qwen3.5的GDN、Kimi-K3的KDA（这两类在PP或RapidServe下改用fork）、DeepSeek-V4
    #   fork：checkpoint就是一个"内容仍有效的空闲槽位"（StateSlotPool.hash_to_slot）；恢复 = 下一次forward从源槽位
    #     （seq.state_fork_src）读初始状态、写进请求自己的槽位，源槽位被pin住直到那次forward发出。用于上述模型在PP / RapidServe下、
    #     以及不在PAGE名单里的GDN模型（gdn_attn.py GDN_PAGE_CHECKPOINT_MODEL_TYPES 之外，如Qwen4Exp）
    #   replay：不存checkpoint，命中时重算一小段来重建（DeepSeek-V4.1默认，见 replay_start）
    #
    # CPU状态层（self.state_offload，只有配了能承载状态的KV connector——如LMCache——时才存在；第9批）：
    #   HBM里的checkpoint被淘汰后，它的镜像可能还在CPU / LMCache里。准入时门控也会为这些hash"投票"
    #   （PagedStateCheckpointCoordinator._reachable），allocate 在HBM里找不到时就请求加载（_request_state_load）
    #
    # 为什么加载 / 存储是异步的：一份镜像是几十MiB量级（例：DSpark3一份镜像29.51 MiB，见 __init__ 里 max_cpu_entries 处的英文），
    #   要经KV connector在worker进程里做H2D / D2H拷贝和LMCache读写，调度器不能为此停下一整步。所以：
    #   加载 = 准入时登记（_state_loads、seq.offload_joint.load_hash）→ 本轮 take_state_loads 交给connector → 请求挂起等待
    #     （WAITING_FOR_REMOTE_KVS）→ 若干步后worker报告 finished / failed_loading → settle_state_load
    #   存储 = take_state_stores 交出READY的checkpoint并pin住它的单元 → GPU读完（D2H排空）→ release_state_store_source 还单元
    #     → CPU侧put有了结果 → settle_state_store，成功才登记hash（两阶段，见各函数）
    #
    # "孤儿槽位"（orphan load slot）：请求在状态加载还在途时就被 deallocate（客户端断开 / 被抢占），它的已提交槽位正被worker写入，
    #   不能还回空闲表（否则下一条请求会拿到一个别人正在写的槽位）。deallocate 把它停放到 _orphan_load_slots[req_id]
    #   （带时间戳，同一请求可能停放多次，按准入顺序排队），由 settle_state_load / abandon_state_load 按FIFO还回；
    #   报告永远不来时（worker崩溃、完成信号丢失），reconcile_orphan_load_slots 在超时后回收
    #   （超时 = Scheduler._save_abandon_timeout_s，LMCache下为 LMCACHE_EC_PIN_TIMEOUT_SEC + 30秒）。
    #   注意与 StateSlotPool.checkpoints_orphaned 区分：那是"checkpoint所在前缀的KV已被淘汰、再也无法命中"的计数
    #
    # 对比nano-vllm：这一整组都没有（nano-vllm只支持逐token存KV的标准注意力，前缀命中即可直接接上）
    # =====================================================================================================
    def _state_leg_secured(self, seq: Sequence) -> bool:
        """Whether something will really put state behind this seq's boundary.

        Exactly three ways that can be true, and they are the three exits
        `_attach_state_slots` takes when it returns True with a boundary
        claimed: a PAGE restore was queued, a fork source was adopted or read,
        or a CPU load was requested. A cold start is not one of them -- it
        returns True as well, which is correct with no boundary and wrong with
        one.
        """
        # 【译】是否真的有东西会把状态放到这条seq的边界后面。
        #
        #   恰好有三种方式能让它成立，正是 _attach_state_slots 在认领了边界的情况下返回True的三个出口：排了一次PAGE恢复、
        #   采用或读取了一个fork源、或者请求了一次CPU加载。冷启动不在其中——它同样返回True，这在没有边界时正确，有边界时就错了。
        # 调用方：allocate 末尾的兜底检查（联合边界背后必须有状态那条腿）
        # 【译】（行尾注释）有一次CPU加载正为它在途
        if seq.offload_joint.load_hash != -1:
            return True  # a CPU load is in flight for it
        # 【译】（行尾注释）有一个fork checkpoint作为它的源
        if getattr(seq, "state_fork_src", -1) != -1:
            return True  # a fork checkpoint is its source
        # 剩下只有PAGE恢复一种：没有PAGE coordinator → False；有则看下一个batch是否排了一次gather恢复、目标正是本seq的已提交槽位
        if self.paged_state_checkpoints is None:
            return False
        return self.paged_state_checkpoints.restore_queued_for(seq.state_slot)

    def _attach_state_slots(self, seq: Sequence, hit_hash: int) -> bool:
        """Give `seq` its state slots, resuming from a checkpoint when one exists.

        Returns whether `hit_hash`'s state is really the one `seq` now holds.
        False means the caller must drop the boundary; see `allocate`.
        `hit_hash` is the content hash of the last reused block (-1 for a cold
        start). `can_allocate` already shrank the hit to a boundary that
        `_resumable_from` accepted, and that is not the same as "in HBM": the
        tier votes too. So a hash whose slot went to LMCache arrives as a miss
        and becomes a load (`_request_state_load`), and so does one whose bytes
        LMCache's own LRU has since dropped — which the tier cannot know until
        the fetch misses. The two are told apart by whether the tier still
        holds the hash, because only one of them may keep the boundary.

        A seq takes `state_slots_per_req` slots — one committed state plus a
        rollback slot per speculated token — however it starts. The checkpoint
        it resumes from is one slot wide, because speculation scratch is not
        state anybody resumes into, so a resume costs the pool exactly what a
        cold start does.

        Resuming shares: the checkpoint stays indexed and the request gets slots
        of its own, so a second request hitting the same prefix still finds it.
        How the state reaches the committed slot is the backend's
        `StateTransfer` — a fork reads the checkpoint for one forward, a copy is
        handed the bytes — and the two differ by one line here. When the pool
        cannot give a full set the request adopts the checkpoint as its
        committed slot instead: still correct, the state is exactly the one it
        wanted, it just spends the checkpoint rather than sharing it.

        A checkpoint is read-only, so several requests in one step may resume
        off the same one. The first takes it off the free list and the pin
        covers every reader until `release_state_pins`; a later one in that same
        step finds it already pinned and only needs slots to write into.
        Adopting is then off the table — the pin means someone else's forward
        still has to read it, or copy out of it.

        PAGE checkpoints gather into a fresh committed slot rather than being
        adopted: the bytes live in the KV pool, not in a state slot, so there is
        nothing here to hand over. Only fork checkpoints can be adopted.
        """
        # 【译】给seq分配它的状态槽位；有checkpoint时从checkpoint恢复。
        #
        #   返回 hit_hash 对应的状态是否真的就是seq现在持有的状态。False表示调用方必须丢掉这个边界；见 allocate。hit_hash 是
        #   最后一个被复用block的内容hash（冷启动为-1）。can_allocate 已经把命中缩到了 _resumable_from 接受的边界，而这和
        #   "在HBM中"不是一回事：存储层也会投票。所以一个槽位已转去LMCache的hash，到这里表现为未命中并变成一次加载
        #   （_request_state_load）；一个字节已被LMCache自己的LRU丢掉的hash也一样——存储层要等取数据落空才知道。二者靠
        #   "存储层是否仍持有这个hash"来区分，因为只有其中一种可以保留边界。
        #
        #   无论怎样开始，一条seq都占 state_slots_per_req 个槽位——一个已提交状态，外加每个投机token一个回滚槽位。它恢复所用的
        #   checkpoint只有一个槽位宽，因为投机用的草稿空间不是谁会恢复进去的状态，所以一次恢复花掉的槽位与冷启动完全相同。
        #
        #   恢复是共享的：checkpoint仍留在索引里，请求拿到自己的槽位，于是第二个命中同一前缀的请求仍能找到它。状态如何到达
        #   已提交槽位，由后端的 StateTransfer 决定——fork在一次forward里读取checkpoint，copy则是把字节交过去——两者在这里只差
        #   一行。当池子给不出完整的一组槽位时，请求改为把checkpoint采用为自己的已提交槽位：仍然正确，状态正是它想要的那份，
        #   只是花掉了checkpoint而不是共享它。
        #
        #   checkpoint是只读的，所以同一步里可以有多个请求从同一个checkpoint恢复。第一个把它从空闲表取走，pin覆盖所有读者直到
        #   release_state_pins；同一步里后来的请求发现它已被pin住，只需要可写入的槽位。这时就不能再采用它了——pin意味着别人的
        #   forward还要读它，或从它拷出数据。
        #
        #   PAGE checkpoint是gather到一个新的已提交槽位，而不是被采用：字节住在KV池里，不在某个状态槽位里，所以这里没有可以
        #   交出去的东西。只有fork checkpoint可以被采用。
        # 观察：docstring里的 _resumable_from 和 release_state_pins 在代码里都已不存在，对应的分别是各状态类的 resumable_hit
        #   （经 _gated_hit 调用）和 StateSlotPool.release_pins（经 complete_previous_state_batch 调用）
        # ---- PAGE后端：checkpoint镜像在KV池里。总是先拿一组新槽位，再把镜像gather进已提交槽位 ----
        width = self.state_slots_per_req
        if self.paged_state_checkpoints is not None:
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = -1
            # 【译】（行尾注释）冷启动：什么都没认领，也没有要恢复的
            if hit_hash == -1:
                return True  # cold start: nothing claimed, nothing to restore
            if self.paged_state_checkpoints.begin_restore(hit_hash, seq.state_slot):
                # Queued for the next batch, which gathers the image out of its
                # PAGE units into the committed slot. What the caller holds is
                # what it asked for.
                # 【译】排进下一个batch：那个batch会把镜像从它的PAGE单元gather到已提交槽位。调用方持有的正是它要的东西。
                return True
            # HBM does not have it. That used to be an invariant violation and
            # this used to raise -- `can_allocate` had shrunk the hit to a
            # boundary the HBM index carried, so a miss here meant the gate and
            # the store disagreed.
            #
            # It is a normal path now, and it has to be: the gate consults the
            # CPU tier too, so a hash it accepted may live only there. Leaving
            # the raise in would take the engine down on the first request that
            # actually used the tier.
            # 【译】HBM里没有它。这曾经算违反不变量，这里曾经直接raise——can_allocate 已经把命中缩到HBM索引里有的边界，所以这里
            #   未命中就意味着门控和存储意见不一。
            #
            #   现在这是正常路径，而且必须是：门控也会咨询CPU层，所以它接受的hash可能只存在于那里。留着raise的话，第一个真正用到
            #   存储层的请求就会让引擎崩溃。
            # 【分支】CPU状态层（第9批）：请求把镜像从LMCache加载进已提交槽位
            if self._request_state_load(seq, hit_hash):
                return True
            # Neither tier can produce it. Reachable rather than defensive:
            # the tier's index is optimistic (`hashes` means "was stored once"),
            # and an HBM checkpoint can be unindexed between `can_allocate` and
            # here by another seq's `_fresh_block` in the same pass. Disown the
            # boundary -- blocks stay claimed and the forward recomputes.
            # 【译】两层都给不出它。这是可达的而不是防御性代码：存储层的索引是乐观的（hashes 的意思是"曾经存过一次"），而且HBM里的
            #   checkpoint可能在 can_allocate 与这里之间，被同一轮里另一条seq的 _fresh_block 取消登记。放弃这个边界——block仍然
            #   认领着，forward重新计算。
            self.state_gate_lost_boundary += 1
            return False

        # ---- fork后端（以及replay）：checkpoint就是状态池里的一个空闲槽位 ----
        if self.state.transfer.replays:
            # Nothing to resume or load: a fresh slot, as for a cold start; the
            # forward rebuilds the state (`replay_start`).
            # 【译】没有可恢复或加载的：像冷启动一样给新槽位；forward会重建状态（replay_start）。
            hit_hash = -1
        # 找存着这个hash时刻状态的空闲槽位（StateSlotPool.hash_to_slot）；找不到时先看CPU层能否给，然后拿一组新槽位
        src = self.state.lookup(hit_hash) if hit_hash != -1 else -1
        if src < 0:
            wants_load = self._tier_can_serve(hit_hash)
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = -1
            if wants_load and self._request_state_load(seq, hit_hash):
                return True
            # A fresh slot holds the previous occupant's bytes. That is fine
            # for a cold start (nothing claims otherwise) and wrong for a hit,
            # so the hit only survives if there was none to begin with.
            # 【译】新槽位里装的是上一个占用者的字节。这对冷启动没问题（没有谁声称别的），对命中就是错的，所以只有本来就没有命中时，
            #   这个"命中"才算保得住。即：冷启动返回True；有命中却找不到checkpoint → False，allocate 放弃边界、从0重算
            return hit_hash == -1
        # Being resumed from is the evidence a guessed position was right, so a
        # checkpoint that pays off stops being spent first — see
        # `StateSlotPool.mark_speculative`. Here rather than in `claim`, which
        # `_set_hash` also calls to re-file a slot that nobody read.
        # 【译】被恢复使用，就证明当初猜的位置是对的，所以一个有回报的checkpoint不再优先被花掉——见 StateSlotPool.mark_speculative。
        #   放在这里而不是 claim 里，是因为 _set_hash 也会调用 claim，去重新归档一个没人读过的槽位。
        self.state.promote(src)
        # 同一步里已有别的请求从它恢复（已被pin住）→ 它已不在空闲表上，不必再取；否则用 claim 把这个指定槽位从空闲表取下
        #   （内容和hash保留：它仍是可被命中的checkpoint）
        shared = self.state.is_pinned(src)
        if not shared:
            self.state.claim(src)
        # 还拿得出一整组新槽位 → 共享式恢复：自己的槽位 + 下一次forward从src读初始状态
        if self.state.has_free(width):
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = src
            # Held off the free list until the forward that reads it is issued.
            # 【译】在读它的那次forward发出之前，一直不放回空闲表。
            # 例：width=1、空闲槽位{3, 5}，checkpoint在槽位5：claim(5) → 空闲{3} → 本请求拿3、fork源=5；5被pin住，
            #   那次forward发出后由 release_pins 放回空闲表（仍带着hash，下一个同前缀请求还能命中）
            self.state.pin(src)
            return True
        # `can_allocate` admitted this seq against a free list holding at least
        # `width`, and nothing else has run since, so the list can only be short
        # here if this seq's own `claim` above took the slot that made up the
        # count — which is `src`, unshared. Adopting it hands that slot straight
        # back to this seq, so the width is met.
        # 【译】can_allocate 是在空闲表至少有 width 个槽位时准入这条seq的，之后没有别的东西运行过，所以这里空闲表不够，只可能是
        #   这条seq自己上面的 claim 拿走了凑数的那个槽位——也就是 src，且没有被共享。采用它就把那个槽位直接交还给这条seq，
        #   宽度就够了。
        # 采用：把checkpoint槽位直接当成自己的已提交槽位（invalidate 去掉它的hash：内容马上会被改写，不能再被别人命中）。
        #   例：width=1、空闲表只剩槽位5（存着checkpoint）：claim(5) 后空闲表为空 → 采用5，state_slots=[5]
        assert not shared, "no slot to resume into and the source is being read"
        self.state.invalidate(src)
        seq.state_slots = [src] + self.state.pop_many(width - 1)
        seq.state_fork_src = -1
        return True

    def _tier_can_serve(self, hit_hash: int) -> bool:
        """Whether the tier could actually serve a load for `hit_hash`.

        Asked twice per admission -- once by `_attach_state_slots` to decide
        whether to try a load at all, and again inside `_request_state_load`.
        Both, and `request_load` itself, go through `StateOffloadIndex.
        could_serve`, so all three test the same capability+membership predicate
        and cannot drift. A bare `hit_hash in hashes` here dropped the `can_load`
        half: a store-only role (`kv_producer`) voted a hit it would then refuse,
        the resumable scan stopped at the tier rung and skipped a still-resident
        HBM rung, and the boundary was disowned into a full recompute.
        """
        # 【译】存储层能否真的为 hit_hash 提供一次加载。
        #
        #   每次准入问两次——一次是 _attach_state_slots 决定要不要尝试加载，一次在 _request_state_load 里面。这两处以及
        #   request_load 本身都走 StateOffloadIndex.could_serve，所以三者检验的是同一个"能力 + 成员"谓词，不会各说各话。
        #   这里若只写 `hit_hash in hashes`，就丢掉了 can_load 那一半：一个只存不取的角色（kv_producer）会为一个它随后又拒绝的
        #   命中投票，可恢复扫描停在存储层那一级、跳过了一个仍驻留在HBM里的级，边界随后被放弃，变成整段重算。
        return (
            hit_hash != -1
            and self.state_offload is not None
            and self.state_offload.could_serve(hit_hash)
        )

    def _request_state_load(self, seq: Sequence, hit_hash: int) -> bool:
        """Ask the tier to fetch `hit_hash` into the slot `seq` just took.

        Only reached when the HBM index missed -- the case the tier exists for.
        The bytes land in the committed slot, where the resuming forward reads
        them.

        `state_fork_src` stays -1. The loaded slot *is* the incoming state, and
        naming a source would send the forward to a different slot than the one
        being filled.

        False means the tier cannot serve it. `request_load` refuses an unknown
        hash because a load is resolved only by a report, so offering one for
        bytes no `get` can produce would park the request forever. The caller
        then disowns the boundary -- the pre-tier answer.
        """
        # 【译】请存储层把 hit_hash 取回到seq刚拿到的槽位里。
        #
        #   只有HBM索引未命中时才会走到这里——这正是存储层存在的意义。字节落在已提交槽位里，恢复的forward从那里读。
        #
        #   state_fork_src 保持-1。被加载的槽位本身就是输入状态，指定一个源会让forward去读另一个槽位，而不是正在被填充的那个。
        #
        #   False表示存储层给不了。request_load 会拒绝未知的hash，因为一次加载只能由一份报告来了结，为没有任何 get 能产出的
        #   字节发起加载，会让请求永远挂起。调用方随后放弃边界——也就是有存储层之前的做法。
        if not self._tier_can_serve(hit_hash):
            return False
        if not self.state_offload.request_load(seq.id, hit_hash):
            return False
        # 登记：load_hash 让调度器把这条seq挂起等加载（WAITING_FOR_REMOTE_KVS）；(req_id, hash, 目标槽位) 本轮由 take_state_loads 交给connector
        seq.offload_joint.load_hash = hit_hash
        self._state_loads.append((seq.id, hit_hash, seq.state_slot))
        return True

    def cancel_state_load(self, seq: Sequence) -> bool:
        """Withdraw a load requested this pass, before anything was issued.
        Only legal before `take_state_loads` handed it over; afterwards the
        bytes are on their way and the slot must be held. The boundary is
        disowned exactly as `allocate` would have -- including privatising the
        claimed prefix (finding #2) so the recompute cannot tear a shared
        decode.

        Returns ``False`` when the pool cannot back the private copies; the
        caller must abandon this admission rather than resume over the shared
        prefix. ``True`` when there was nothing to cancel or the disown held.
        """
        # 【译】撤回本轮请求的一次加载，此时还什么都没发出。只能在 take_state_loads 把它交出去之前这样做；交出之后字节已在路上，
        #   槽位必须一直留着。边界的放弃方式与 allocate 完全一样——包括把认领的前缀私有化（finding #2），这样重算就不会撕坏
        #   一个共享它的decode。
        #
        #   池子给不出私有副本时返回 False；调用方必须放弃这次准入，而不是在共享前缀上恢复。没有要取消的、或放弃成功时返回 True。
        # 调用方：scheduler.py 准入循环里 allocate 之后、同一轮 _publish_state_loads 之前（KV那条腿和状态加载对不上、
        #   或联合边界的KV腿被拒时）
        if seq.offload_joint.load_hash == -1:
            return True
        self._state_loads = [e for e in self._state_loads if e[0] != seq.id]
        if self.state_offload is not None:
            self.state_offload.abandon_load(seq.id)
        seq.offload_joint.load_hash = -1
        disowned = self.disown_claimed_prefix(seq)
        seq.num_cached_tokens = 0
        return disowned

    def abandon_state_load(self, req_id) -> None:
        """Give up on a load without blaming the bytes for it.
        For a load nothing could carry. `settle_state_load(ok=False)` would
        `forget` the hash on a miss that never happened, erasing the index one
        request at a time and inflating its false-positive counter.
        """
        # 【译】放弃一次加载，但不把责任归到那些字节上。用于没有任何东西能承载的加载。settle_state_load(ok=False) 会因为一次
        #   从未发生过的未命中而 forget 这个hash，一个请求一个请求地把索引抹掉，还会抬高它的误报计数。
        # 调用方：scheduler.py _publish_state_loads（connector不接这批加载时）、_abandon_state_load（KV腿失败但状态那一路完好时）。
        #   顺带按FIFO还回一个孤儿槽位（如果有）
        if self.state_offload is None:
            return
        self.state_offload.abandon_load(req_id)
        self._release_oldest_orphan_slot(req_id)

    def _release_oldest_orphan_slot(self, req_id) -> None:
        """Hand back the oldest orphan-parked slot for `req_id`, if any.

        FIFO: a preempt/re-admit can park a request's slot more than once, and
        the worker reports its loads in admission order, so the oldest park is
        the one this report settles. Releasing newest-first would free a slot a
        later load is still writing.
        """
        # 【译】如果 req_id 有被停放的孤儿槽位，交还最早停放的那个。
        #
        #   FIFO：一次抢占 / 重新准入可能让同一请求的槽位被停放不止一次，而worker按准入顺序报告加载，所以这份报告结清的是
        #   最早的那次停放。若先释放最新的，就会放掉一个后来的加载还在写的槽位。
        # 例：请求7第一次准入时向槽位12加载、加载未完就被抢占（12停放）；重新准入后向槽位20加载、又被abort（20停放）
        #   → _orphan_load_slots[7] = [(12, t1), (20, t2)]；第一份报告还12，第二份还20
        parked = self._orphan_load_slots.get(req_id)
        if not parked:
            return
        slot, _at = parked.pop(0)
        if not parked:
            del self._orphan_load_slots[req_id]
        self.state.release(slot)

    def settle_state_load(self, req_id, ok: bool) -> None:
        """Apply one worker load report. Keyed by request, like the KV load.

        Called for every `finished_loading`/`failed_loading` id, including the
        many that are plain KV loads -- a no-op for those, which keeps the
        scheduler from having to know which leg a report belongs to. An
        abandoned load is already out of the index, but its slot is still being
        written and comes back here, and only here.
        """
        # 【译】应用worker的一份加载报告。和KV加载一样按请求为键。
        #
        #   每个 finished_loading / failed_loading 的id都会调用它，其中很多只是普通的KV加载——对它们这是空操作，这样调度器
        #   就不必知道一份报告属于哪条腿。一次被放弃的加载已经不在索引里了，但它的槽位仍在被写入，会在这里、也只在这里还回来。
        # 调用方：scheduler.py _settle_state_load（_update_from_kv_xfer_finished 处理 finished_loading / failed_loading 时）。
        #   请求还活着时，加载好的槽位就是它的已提交槽位，这里不释放（没有孤儿项）；只有已被 deallocate 的请求才在这里还槽位
        if self.state_offload is None:
            return
        # 成功：complete_load；失败：fail_load（乐观索引的误报在这里自我纠正：忘掉这个hash，见 PagedStateCheckpointCoordinator._reachable）
        if ok:
            self.state_offload.complete_load(req_id)
        else:
            self.state_offload.fail_load(req_id)
        self._release_oldest_orphan_slot(req_id)

    def take_state_stores(self, max_inflight: int) -> list[tuple]:
        """`(operation, unit_ids)` for checkpoints to hand the CPU tier.

        Pins each as it hands it over; `settle_state_store` releases. Empty
        without a coordinator (the fork backends), which is correct rather than
        a gap: a fork checkpoint is a state slot, and #2045 is what moved K3's
        into the KV pool where a set of units can be read without being
        rescued first.
        """
        # 【译】要交给CPU层的checkpoint，形如 (operation, unit_ids)。
        #
        #   每交出一个就pin住它；settle_state_store 负责释放。没有coordinator（fork后端）时为空，这是正确的而不是缺口：fork
        #   checkpoint是一个状态槽位，而 #2045 正是把K3的checkpoint搬进KV池的那次改动，在KV池里一组单元可以直接被读取，
        #   不必先抢救出来。
        # 调用方：scheduler.py _publish_state_stores（每轮一次，和 _publish_state_loads 一起；max_inflight 限制同时在途的存储数）
        if self.paged_state_checkpoints is None or self.state_offload is None:
            return []
        if not self.state_offload.can_store:
            # A load-only role. Handing over a store pins units against a
            # report the worker's save half will never produce.
            # 【译】只加载的角色。交出一次存储，会为一份worker的保存那一半永远不会产出的报告钉住单元。
            return []
        out = self.paged_state_checkpoints.take_offload_stores(max_inflight)
        self.state_offload.stores_attempted += len(out)
        return out

    def settle_state_store(self, op, ok: bool, attempted: bool = True) -> None:
        """One store reported. Release the units, and index the hash if it landed.

        Success and failure release the pin identically -- it existed to keep
        the bytes still during the copy, and the copy is over either way.
        Only the indexing differs, because only a hash whose bytes are really
        there may be voted for.

        `attempted` is `False` for a store a connector *refused* to carry: the
        pin still has to be released here, but the copy never reached a worker,
        so it is not a `stores_failed` -- the caller counts the refusal once as
        `stores_refused`, and bumping `stores_failed` too would double-count the
        same event under two names. For the same reason it backs the optimistic
        `stores_attempted` bump `take_state_stores` made at hand-out back out, so
        a refusal does not linger in `attempted - completed - failed`.

        `op` is a `StateStoreOperationId`: the pin is released for that exact
        generation, so a late report from a superseded attempt settles nothing.
        The index is keyed by hash, because what a resume asks is whether the
        prefix is in LMCache -- not which attempt put it there.
        """
        # 【译】一次存储报告回来了。释放单元；如果字节确实落地了，就登记这个hash。
        #
        #   成功和失败释放pin的方式完全相同——pin的作用是在拷贝期间让字节保持不动，而无论哪种结果，拷贝都已经结束了。
        #   只有登记不同，因为只有字节真的在那里的hash才能被投票选中。
        #
        #   对于connector拒绝承载的存储，attempted 为 False：pin仍然要在这里释放，但拷贝从未到达worker，所以它不算
        #   stores_failed——调用方把这次拒绝记一次 stores_refused，再加 stores_failed 就会用两个名字重复计数同一件事。出于同样
        #   的原因，它要把 take_state_stores 交出时乐观地加上的 stores_attempted 减回去，免得一次拒绝一直留在
        #   attempted − completed − failed 里。
        #
        #   op 是一个 StateStoreOperationId：pin按这个确切的代次释放，所以一份来自已被取代的尝试的迟到报告什么也不结清。
        #   索引按hash为键，因为恢复时问的是"这个前缀在不在LMCache里"——而不是哪次尝试把它放进去的。
        # 调用方：scheduler.py 处理connector的存储报告（成功 / 失败），以及 _publish_state_stores 里connector拒收时（attempted=False）
        reclaimed = False
        if self.paged_state_checkpoints is not None:
            reclaimed = self.paged_state_checkpoints.was_reclaimed(op)
            # Idempotent: normally the source release already returned these
            # units, and this is the backstop for a store that failed before
            # it ever read them.
            # 【译】幂等：通常源释放（release_state_store_source）已经把这些单元还回去了，这里是给"还没读到它们就失败了"的存储兜底。
            self.paged_state_checkpoints.settle_offload_store(op)
        if self.state_offload is None:
            return
        if ok and reclaimed:
            # The stale reclaimer took this store's units back before it
            # reported, and nothing can say whether the worker had stopped
            # reading them. If it had not, the pool may have handed them to
            # another request whose writes the gather picked up, making the CPU
            # image a mix of two prefixes filed under the first one's hash --
            # and a resume onto that is silent wrong output. The bytes may well
            # be fine; there is no way to know, so they are forfeited.
            # 【译】过期回收器在这次存储报告之前就把它的单元收回了，没有任何办法知道worker当时是否已停止读取它们。如果还没停，
            #   池子可能已把这些单元交给另一个请求，而gather读到了那个请求的写入，使CPU里的镜像成了两个前缀的混合、却登记在
            #   第一个前缀的hash下——从它恢复就是静默的错误输出。字节很可能没问题；但无从得知，所以放弃它们。
            self.state_offload.stores_untrusted += 1
            logger.warning(
                "state offload: %s reported stored, but its units were "
                "reclaimed while it ran; refusing to index it.",
                op,
            )
            return
        if ok:
            self.state_offload.stores_completed += 1
            self.state_offload.note_stored(int(op.prefix_hash))
        elif attempted:
            self.state_offload.stores_failed += 1
        else:
            # A refusal. `take_state_stores` bumped `stores_attempted` when it
            # handed this op over, but it never reached a worker, so back that
            # out: the event is counted once as `stores_refused`, and leaving it
            # in `stores_attempted` would read as permanently in-flight in the
            # `attempted - completed - failed` funnel.
            # 【译】一次拒绝。take_state_stores 交出这个op时把 stores_attempted 加了1，但它从未到达worker，所以减回去：这件事只记一次
            #   stores_refused，若留在 stores_attempted 里，在 attempted − completed − failed 这个漏斗里就会显示为永远在途。
            self.state_offload.stores_attempted -= 1

    def release_state_store_source(self, op) -> None:
        """The GPU has finished reading this store's PAGE units; hand them back.

        Separate from `settle_state_store` because the two answer different
        questions: the units are the KV pool's and are free as soon as the D2H
        drains, while whether the CPU put succeeded is decided afterwards and
        cannot touch them. Holding an image out of the pool across that would
        cost reuse for nothing.

        Phase one of the two-phase pin: it releases the units but leaves the
        operation pinned as dispatched-but-unreported, so `has_offload_pins`
        (and the liveness gate that ORs it in) keeps the engine polling until
        the index report actually lands. Retiring the pin here -- as it once
        did -- let liveness read idle while the store's report still sat
        undrained in the worker, so `note_stored` never ran and the checkpoint
        was never votable.
        """
        # 【译】GPU已读完这次存储的PAGE单元；把它们还回去。
        #
        #   与 settle_state_store 分开，是因为两者回答的是不同问题：单元属于KV池，D2H一排空就空出来了；而CPU侧的put是否成功
        #   是之后才决定的，与它们无关。在那期间把一整份镜像扣在池子外，只会白白损失复用。
        #
        #   两阶段pin的第一阶段：它释放单元，但让这次操作保持"已派发、未报告"的pin，于是 has_offload_pins（以及把它或进来的
        #   存活判断）会让引擎一直轮询，直到登记报告真正到达。像以前那样在这里就撤掉pin，会让存活判断在存储报告还没从worker
        #   取出时读成空闲，于是 note_stored 永远不会执行，这个checkpoint也永远不能被投票选中。
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.release_offload_store_source(op)

    def has_pending_state_store_pins(self) -> bool:
        """Whether a dispatched state store is still awaiting its report.

        The engine's liveness gate (`EngineCore.has_pending_kv_work`) ORs this
        in so a step whose only outstanding work is a state store already handed
        to the worker does not read as idle. Without it the store's PAGE units
        stay pinned out of the KV pool until they age out, because the report
        that would release them -- and the reclaim that is the report's only
        backstop -- are both drained by `_poll_kv_transfer_progress`, which the
        idle loop stops calling the moment liveness goes False. This is the
        unit-store twin of the `deferred_free_blocks` condition the gate already
        checks for the KV save leg.
        """
        # 【译】是否有一次已派发的状态存储还在等它的报告。
        #
        #   引擎的存活判断（EngineCore.has_pending_kv_work）把它或进去，这样当一步唯一剩下的工作是一次已交给worker的状态存储时，
        #   不会被读成空闲。没有它的话，这次存储的PAGE单元会一直被pin在KV池之外直到超时，因为能释放它们的报告——以及作为报告
        #   唯一兜底的回收——都由 _poll_kv_transfer_progress 取出，而存活判断一变成False，空闲循环就不再调用它。这是存活判断
        #   已经为KV保存那条腿检查的 deferred_free_blocks 条件在"单元存储"上的孪生。
        if self.paged_state_checkpoints is None:
            return False
        return self.paged_state_checkpoints.has_offload_pins()

    def reclaim_stale_state_store_pins(self, timeout_s: float) -> int:
        """Release store pins whose report never came. See the store."""
        # 【译】释放那些报告一直没来的存储pin。详见store（PageUnitCheckpointStore.reclaim_stale_offload_pins）。
        # 调用方：scheduler.py 处理完存储报告之后，超时用 _save_abandon_timeout_s()（与KV保存的回收同一个窗口）
        if self.paged_state_checkpoints is None:
            return 0
        return self.paged_state_checkpoints.reclaim_stale_offload_pins(timeout_s)

    def reconcile_orphan_load_slots(self, timeout_s: float) -> int:
        """Free load slots whose `settle_state_load` never came, after `timeout_s`.

        The load-side twin of `reclaim_stale_state_store_pins`. `deallocate`
        parks a slot in `_orphan_load_slots` when it tears down a request whose
        state load is still in flight, on the promise that the worker reports
        every load and `settle_state_load` will hand it back. A crashed worker
        or a dropped completion breaks that promise, and the slot then sits off
        the free list forever -- `can_allocate`'s state gate refuses new
        per-request work once enough slots are stranded, wedging the pool with
        no fault to point at.

        **This is a last resort and cannot tell a lost report from a slow
        worker still writing the slot** -- the same limitation the store-pin
        twin documents. Reclaiming under a live H2D hands the next request a
        buffer someone else is filling, with `has_initial_state` already true
        over it. So the window must be the abandon timeout, not a tight one:
        long enough that a report still in flight has already arrived. The
        index leg was abandoned back in `deallocate`, so nothing here touches
        it; a late report finds the slot already popped and releases nothing.
        """
        # 【译】在 timeout_s 之后，释放那些 settle_state_load 一直没来的加载槽位。
        #
        #   reclaim_stale_state_store_pins 在加载一侧的孪生。deallocate 拆掉一个状态加载仍在途的请求时，会把槽位停放到
        #   _orphan_load_slots，指望worker报告每一次加载、settle_state_load 会把它交还。worker崩溃或完成信号丢失会打破这个承诺，
        #   槽位就永远不在空闲表上——一旦搁浅的槽位够多，can_allocate 的状态闸门就会拒绝新的每请求工作，把池子卡死，而且找不到
        #   任何故障可以指认。
        #
        #   **这是最后手段，分不清"报告丢了"和"worker还在慢慢写这个槽位"**——和存储pin那个孪生记录的局限一样。在一次仍在进行的
        #   H2D之下回收，会把一块别人正在填充的缓冲区交给下一个请求，而它上面的 has_initial_state 已经为真。所以窗口必须是放弃
        #   超时，而不是一个很紧的值：要长到任何仍在途的报告都已经到达。索引那条腿早在 deallocate 里就放弃了，所以这里不碰它；
        #   迟到的报告会发现槽位已被取走，什么也不释放。
        # 观察：docstring最后一句说迟到报告"什么也不释放"，但下面循环里的英文说它"会释放下一个停放的槽位"。按
        #   _release_oldest_orphan_slot 的实现（弹出最早的一项），同一req_id还有更新的停放时，过期项被回收后迟到的报告会把那个
        #   更新的槽位提前还回空闲表——而它可能还在被后一次加载写入。只有同一请求被停放过不止一次时才会出现
        # 调用方：scheduler.py 处理KV传输进度时，与 reclaim_stale_state_store_pins 同窗口（_save_abandon_timeout_s；0 = 关闭）
        if timeout_s <= 0 or not self._orphan_load_slots:
            return 0
        # monotonic()：单调时钟秒数（不受系统时间调整影响），停放时记的也是它
        cutoff = monotonic() - timeout_s
        reclaimed = 0
        # list(...) 先拷一份键：循环里会删条目
        for req_id in list(self._orphan_load_slots):
            parked = self._orphan_load_slots[req_id]
            # Each admission is stamped and expires on its own age -- a stale
            # park is dropped even when a newer one for the same id is still
            # inside the window (kept in place).
            # 【译】每次准入各自带时间戳、按各自的年龄过期——即使同一id还有一个仍在窗口内的较新停放（原地保留），过期的那个也会被丢掉。
            fresh = []
            for slot, at in parked:
                if at <= cutoff:
                    # The index leg was already abandoned in `deallocate` when the
                    # slot was parked; only the slot itself outlived the report.
                    # Do not touch the index again here -- a second `abandon_load`
                    # would double-count the miss. A late report that still calls
                    # in finds this slot gone and releases the next parked one.
                    # 【译】槽位被停放时，索引那条腿已经在 deallocate 里放弃了；比报告活得更久的只有槽位本身。这里不要再碰索引——第二次
                    #   abandon_load 会把这次未命中重复计数。仍然打进来的迟到报告会发现这个槽位已不在，转而释放下一个被停放的槽位。
                    self.state.release(slot)
                    reclaimed += 1
                else:
                    fresh.append((slot, at))
            if fresh:
                self._orphan_load_slots[req_id] = fresh
            else:
                del self._orphan_load_slots[req_id]
        self._orphan_load_slots_reclaimed += reclaimed
        return reclaimed

    def take_state_loads(self) -> list[tuple]:
        """`(req_id, hash, target_slot)` for loads admitted since the last call.
        Drained once per pass by the scheduler, which hands them to the
        connector. Draining rather than reading is what keeps a load from being
        submitted twice into a slot the first transfer is already filling.
        """
        # 【译】自上次调用以来准入的加载，形如 (req_id, hash, target_slot)。调度器每轮取走一次、交给connector。用"取走"而不是
        #   "读取"，才能防止同一次加载被提交两次、写进第一次传输已经在填充的槽位。
        # 调用方：scheduler.py _publish_state_loads（build_connector_meta 之前）
        if self.state_offload is None:
            return []
        # 元组赋值：先取出旧列表、再换上新的空列表，一步完成"取走"
        out, self._state_loads = self._state_loads, []
        return out

    def _chain_parent_hash(self, seq: Sequence, start: int) -> int | None:
        """Return the chained hash of block ``start - 1``, or ``None`` on a gap.

        All source paths (register_prefill_hashes, postprocess, offload wake)
        are expected to hash blocks before this is called. A gap means a
        source-level bug; callers skip the range rather than mint false hashes.
        """
        # 【译】返回第 start−1 块的链式hash；中间有缺口时返回 None。
        #
        #   所有来源路径（register_prefill_hashes、postprocess、offload唤醒）都应在调用它之前已经hash过前面的block。出现缺口
        #   说明来源处有bug；调用方跳过这段范围，而不是造出错误的hash。
        # hash_blocks 要从第start块接着链，就需要第start−1块的hash：start=0 时用 cache_seed；否则读 block_table[start−1]
        #   那个物理块上登记的hash（-1 = 还没登记 → 缺口）。例：can_allocate 例子里的B命中4块，prefill算完后 hash_blocks 从第4块
        #   开始，父hash = 物理块3上的h3（A登记的，B共享）。对比nano-vllm：hash_blocks 里的
        #   `h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1`，ATOM多了缺口检查
        if start <= 0:
            return seq.cache_seed
        h = self.kv.block(seq.block_table[start - 1]).hash
        if h != -1:
            return h
        logger.error(
            "Unhashed parent block %d for seq %s — skipping hash "
            "registration for blocks %d onward",
            start - 1,
            seq.id,
            start,
        )
        return None

    def hash_blocks(
        self,
        seq: Sequence,
        num_new_tokens: int,
        start_tokens: int | None = None,
        next_forward_tokens: int | None = None,
        aimed: bool = True,
        replay_until: int | None = None,
    ) -> None:
        """Register hashes for blocks finalized by the most recent step.

        Called from scheduler.postprocess() after the forward completes, so a
        block's hash is only published once its KV is actually computed. The
        `[start, end)` range covers blocks fully filled by this step:
          start = first block whose first token was at num_cached_tokens
          end   = first block not yet fully filled (excludes the partial one)
        Caller passes `num_new_tokens` = tokens forwarded in this step. For
        single-shot prefill that's `seq.num_tokens - seq.num_cached_tokens`;
        chunked prefill will pass the per-chunk count.

        `start_tokens` overrides the token offset the range starts at. Pipeline-
        parallel schedule-time advancement already bumped seq.num_cached_tokens
        past this chunk, so the head passes the chunk's pre-advance offset here.

        `next_forward_tokens` reaches `checkpointers_at`; see there. Left
        unset it reads the prompt's remainder, which is the prefill answer.

        `replay_until`: where the forward resumed if it started below that
        point, as the batch was scheduled (`ScheduledBatch.replaying`). Nothing
        below it is published or checkpointed: those blocks are the hit's, and
        a forward ending below it leaves a replay's state still inexact. One
        ending right at it publishes nothing either, but a DeepSeek-V4 bounded
        replay has rebuilt the state whole there, so that may be checkpointed.
        Left unset it reads `seq.replay_until`, which under pipeline
        parallelism a later chunk's scheduling may already have moved.
        """
        # 【译】为最近一步"定稿"的block登记hash。
        #
        #   在forward完成之后由 scheduler.postprocess() 调用，所以一个block的hash只有在它的KV真正算出来之后才发布。
        #   `[start, end)` 这个范围覆盖本步完全填满的block：
        #     start = 第一个"首token位于num_cached_tokens处"的block
        #     end   = 第一个还没填满的block（不含那个半满的block）
        #   调用方传入 `num_new_tokens` = 本步forward的token数。一次性prefill就是 `seq.num_tokens - seq.num_cached_tokens`；
        #   chunked prefill传每个chunk的token数。
        #
        #   `start_tokens` 覆盖范围起点的token偏移。流水线并行在调度时就推进了进度，已经把 seq.num_cached_tokens 推过了这个chunk，
        #   所以head在这里传入该chunk推进之前的偏移。
        #
        #   `next_forward_tokens` 传给 `checkpointers_at`，见那里。不传时取prompt的剩余部分，这正是prefill时的答案。
        #
        #   `replay_until`：如果forward是从某个点下方开始的，这就是forward恢复的那个点，取batch被调度时的值
        #   （`ScheduledBatch.replaying`）。它下方的任何东西都不发布、也不做checkpoint：那些block属于命中部分，而一个结束在它
        #   下方的forward留下的重放状态仍不精确。正好结束在它上面的forward同样什么都不发布，但DeepSeek-V4的有界重放
        #   已经在那里把状态完整重建了，所以那里可以做checkpoint。
        #   不传时读 `seq.replay_until`，而在流水线并行下，后面某个chunk的调度可能已经把它挪动了。
        # 解释：start实际按 max(base, until) // hbs 算，即包含起点token的那个block——它可能在上一个chunk里写了一半，本步才写满
        #
        # ═══════════════ hash_blocks：把"KV已经算好的满block"登记进前缀缓存索引（A级） ═══════════════
        # 调用方（grep核对，都在forward之后）：
        #   Scheduler.postprocess          非PP：每条prefill序列，在 num_cached_tokens += chunk 之前；
        #                                  PP（advance_on_schedule）：用batch里冻结的chunk起点传start_tokens
        #   Scheduler.register_prefill_hashes  PP的中间chunk（不经postprocess），由pp_engine_core._flush_pending_prefix_hashes调用
        #   Scheduler.postprocess 的 batch=None 分支（RapidServe的DecodeEngineCore）：一次性登记整段prompt
        #   hash_decode_blocks（下面）      decode生成的block，aimed=False
        #
        # 链式hash（prefix_hash.block_hash；本类的compute_hash就是它）：
        #   h_-1 = seq.cache_seed   纯文本 = -1，表示"没有父hash"；多模态 = multimodal_cache_seed(媒体数据)：
        #                           像素、图像网格、token类型和embedding_spans的64位blake2b摘要（multimodal_runtime.py）
        #   h_i  = xxh64( h_{i-1}的8字节小端 ‖ 第i块token按int64排列的字节 )；h_{i-1} = -1 时只hash token字节
        #
        #   token：[0 ...... hbs-1][hbs .... 2hbs-1][2hbs ... 3hbs-1][3hbs .. 半满]
        #   hash ：  h0=H(种子,块0) ─► h1=H(h0,块1) ──► h2=H(h1,块2) ──► 不登记
        #   hbs = hash_block_size = block_size × DCP（不开DCP时就是block_size，默认16；DCP下一个block_table条目
        #     对应 block_size × DCP 个全局token，前置知识见scheduler.py文件开头【前置⑥】）
        #   h2相同 ⇔ 种子相同且块0~2的token全同：同一段文字配不同的图片，种子不同，从h0起就全不一样，不会误命中
        #
        # 为什么只在forward之后登记：allocate时只是把block号写进block_table，GPU上那些行还没写。这时就登记的话，别的请求可能
        #   复用一个KV还没写——甚至永远不会写（这条请求在forward之前被抢占、取消）——的block；chunked prefill下一个block还可能
        #   跨两步才写满。半满的block也不登记：它后面的token还没来，hash不是最终值，内容也不完整。
        #     步N 调度：A准入，物理block 9 = A的块3（未登记）     步N forward：GPU写块3的KV
        #     步N postprocess：本函数登记 h3 → block 9           之后准入的请求才能命中h3
        #
        # 例（hbs=16）：本步是chunk [40, 90)，即base = 40、num_new_tokens = 50
        #   start = 40//16 = 2（块2 = [32,48)：32~39是上一个chunk写的，40~47本步写，本步才满）
        #   end   = 90//16 = 5（块5 = [80,96)只写到89，半满，不登记）
        #   → 登记块2、3、4；num_hashed_tokens = 5×16 = 80；本次forward结束于pos = 90，再交给checkpointers_at判断要不要存状态
        # 对比nano-vllm：nano-vllm的hash_blocks也在postprocess里调、也只登记 [start, end) 里填满的block，这条规则相同。
        #   ATOM多处理了：PP在调度时就推进了num_cached_tokens（start_tokens）、有界重放（replay_until）、DCP（按hbs分块）、
        #   多模态种子、复用准入时 / 前端已算好的prompt hash链（known）、KV事件、状态checkpoint与midstep快照（函数末尾）
        if not self.enable_prefix_caching:
            return
        hbs = self._hash_block_size()
        base = seq.num_cached_tokens if start_tokens is None else start_tokens
        # A replaying forward's rows below where it resumed are published.
        # 【译】重放forward中位于它恢复点以下的那些行，是（早已）发布过的。
        # 解释：有界重放（sequence.py Replay类上方科普④）从命中点下方开始重算一段，只为重建状态；那段block属于命中部分，
        #   早就登记过，所以起点取 max(base, until)
        until = seq.replay_until if replay_until is None else replay_until
        start = max(base, until) // hbs
        end = (base + num_new_tokens) // hbs
        # A finished or preempted seq has had its block table released; the
        # deferred publish paths can still reach it with a stale token count.
        # 【译】已结束或被抢占的seq已经释放了它的block table；延迟发布的路径仍可能带着过时的token计数走到这里。
        # 解释：deallocate清空了block_table，end被截成0，下面直接返回
        end = min(end, len(seq.block_table))
        # 没有新填满的block就返回。唯一例外：DeepSeek-V4有界重放的forward正好结束在until上——不发布block，
        #   但可以在那里存状态checkpoint（见docstring）
        if start >= end and not (
            seq.replays_hit and until and base + num_new_tokens == until
        ):
            return
        # 父hash：start = 0时是cache_seed，否则是前一个block上登记的hash；前一个block没登记（-1）说明调用顺序有bug，
        #   整段跳过而不是造出错误的hash（_chain_parent_hash）
        h = self._chain_parent_hash(seq, start)
        if h is None:
            return
        # Watermark for the decode-side continuation, maintained here so every
        # prefill path feeds it without knowing about it.
        # 【译】给decode侧续登记用的水位线，在这里维护，这样每条prefill路径都会顺带更新它、而不必知道它的存在。
        # 解释：num_hashed_tokens = 已登记的token数（总是hbs的整数倍），hash_decode_blocks从这里接着登记
        seq.num_hashed_tokens = max(seq.num_hashed_tokens, end * hbs)
        # 【分支】KV事件（见文件末尾的KV event API）：把本次登记的一串block合成一条BlockStored。start = 0时父hash就是
        #   cache_seed：纯文本为-1 → 事件里写None；多模态则把媒体种子当作父hash发出去
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        # Prompt blocks the seq's chain covers (`_prompt_hashes`) are named
        # from it; only the rest are hashed here.
        # 【译】seq的hash链（`_prompt_hashes`）覆盖到的prompt block直接从链上取名字；只有其余的block才在这里现算hash。
        # 解释：known = 准入时为这条序列算好的（或前端算好随请求送来、_adopt_shipped_chain校验过的）整段prompt hash链，
        #   只含prompt里填满的block；种子对不上、或链上第start-1个与刚取到的父hash不一致时不用它。省掉调度线程里的一遍xxhash；
        #   链之外的（decode生成的block）照常现算
        known = self._prompt_hash_chains.get(seq)
        known = known[1] if known and known[0] == seq.cache_seed else ()
        if 0 < start <= len(known) and known[start - 1] != h:
            known = ()
        for i in range(start, end):
            token_ids = self._hash_block_tokens(seq, i)
            h = known[i] if i < len(known) else self.compute_hash(token_ids, h)
            # kv.publish（block_pool.py）：block记下(h, 本块token)，索引 h → 这个block。h已经指向别的block时改指向这个
            #   （后发布者胜出，并通知prefix walk的memo失效）；这个block原先挂着别的hash则先摘掉那个旧条目
            self.kv.publish(seq.block_table[i], h, token_ids)
            if record:
                store_run_hashes.append(h)
                store_run_tokens.extend(token_ids)
        if record and store_run_hashes:
            self._event_log.append(
                _make_block_stored(
                    store_run_hashes,
                    store_run_tokens,
                    store_run_parent,
                    hbs,
                    token_offset=start * hbs,
                )
            )
        # 【分支】状态checkpoint（见下方checkpoint_limit上方的【科普】状态checkpoint阶梯）：pos = 本次forward结束的位置，
        #   checkpointers_at选出要在这里存快照的状态类。快照归档在h名下，h = 刚登记的最后一个block（块end-1，恰好结束在pos）
        #   的hash——所以能存的pos都是hbs的整数倍；last_checkpoint_pos供decode侧"离上一个checkpoint够远"的间距规则使用
        pos = base + num_new_tokens
        kept = self.checkpointers_at(seq, pos, next_forward_tokens, aimed)
        for cache in kept:
            cache.checkpoint(seq, end, h)
        if kept:
            seq.last_checkpoint_pos = pos
        # The midstep half of the same moment. Here rather than at each of the
        # three prefill call sites because this is what they have in common:
        # the forward for `[base, base + num_new_tokens)` has completed, which
        # is precisely when the reserved bytes exist. A no-op with nothing
        # reserved, so decode and unreadable backends pay a branch.
        # 【译】同一时刻的midstep那一半。放在这里而不是三个prefill调用处各放一份，因为这正是它们的共同点：
        #   `[base, base + num_new_tokens)` 的forward已经完成，这恰恰是预留的字节已经存在的时刻。没有预留时是空操作，
        #   所以decode和不能中途读的后端只付出一次分支判断。
        self.commit_midstep(seq)

    # ═══════════════ hash_decode_blocks：decode生成的block写满后也登记进前缀缓存（A级） ═══════════════
    # 调用方：Scheduler.postprocess每步对每条序列调用一次，committed_kv_len = num_tokens − (延迟输出 ? 0 : 1)，
    #   next_forward_tokens = Scheduler._checkpoint_room（本步就结束、或投机解码 + fork类状态时为0，否则为1）
    # 为什么要登记生成部分：多轮对话下一轮的prompt = 上一轮prompt + 上一轮回答 + 新问题；只登记prompt的话，
    #   回答那一段每一轮都要重算
    # 例（hbs = 16，按不开延迟输出算，数字好读）：prompt 16380个token
    #   prefill：hash_blocks登记块0~1022（16380//16 = 1023块），num_hashed_tokens = 16368；
    #     块1023 = [16368, 16384)里只有12个prompt token，半满不登记
    #   decode生成g0~g3，依次写位置16380~16383：每步本函数都被调用，committed ≤ 16383时 start = end = 1023，什么都不登记
    #   算位置16383（输入g3）的那一步之后：采样出g4，num_tokens = 16385 → committed_kv_len = 16384
    #     → hash_blocks(seq, 16, start_tokens=16368, aimed=False) → 登记块1023：h1023 = H(h1022, 12个prompt token + g0~g3)，
    #       num_hashed_tokens = 16384；下一轮"本轮prompt + 回答 + …"的请求就能命中到这里
    #   开延迟输出时同一个block早一次postprocess登记：那时位置16383的KV由一个已经发出、排在同一条stream上的forward写，
    #     之后任何请求读它时它一定已经写完
    # 对比nano-vllm：nano-vllm的decode也走同一个hash_blocks（decode时num_scheduled_tokens = 1、num_cached_tokens每步+1），
    #   生成的block同样会登记；ATOM的decode不累加num_cached_tokens（sequence.py里num_cached_tokens的说明），所以单独用
    #   num_hashed_tokens做水位线，并用committed_kv_len这条硬线挡住延迟输出的在途token和被拒绝的投机草稿
    def hash_decode_blocks(
        self, seq: Sequence, committed_kv_len: int, next_forward_tokens: int = 0
    ) -> None:
        """Register hashes for generated blocks filled up to `committed_kv_len`.

        `may_append` allocates decode blocks without hashing them: at allocation
        time their tokens have not been sampled, and under speculative decoding
        part of what the forward writes is about to be rejected.

        `committed_kv_len` counts the tokens for which neither still applies —
        id final, KV computed — and is a hard line, not a hint. It stops short
        of any token no forward has read yet: that token's KV slot is written
        by the next forward, and a block published over an unwritten slot hands
        a later request KV that may never arrive at all (the seq can finish
        first). Prefill's `hash_blocks` draws the same line from its own side,
        at `num_cached_tokens + chunk`.

        Without this the prefix cache indexes prompt blocks only, and a
        follow-up turn — previous prompt plus previous answer — matches nothing
        beyond the original prompt.

        `next_forward_tokens` reaches `checkpointers_at`; see there. It
        defaults to "no next forward", i.e. hash but never checkpoint, so a
        caller opts into decode-point checkpointing rather than out of it.
        """
        # 【译】为已填到 `committed_kv_len` 的生成block登记hash。
        #
        #   `may_append` 分配decode block时不算hash：分配时这些token还没采样出来，而在投机解码下，forward写的一部分
        #   马上就要被拒绝。
        #
        #   `committed_kv_len` 统计的是这两条都不再成立的token——id已定、KV已算——它是一条硬线，不是提示。它停在任何还没被
        #   forward读过的token之前：那个token的KV槽位要由下一次forward来写，而一个盖在未写槽位上发布出去的block，会把一份可能
        #   永远不会到来的KV交给后来的请求（这条seq可能先结束）。prefill的 `hash_blocks` 从它自己那一侧划了同一条线，
        #   在 `num_cached_tokens + chunk` 处。
        #
        #   没有这个函数，前缀缓存只索引prompt block，后续的一轮——上一轮prompt加上一轮回答——除了原始prompt之外什么都匹配不上。
        #
        #   `next_forward_tokens` 传给 `checkpointers_at`，见那里。默认是"没有下一次forward"，即只登记hash、从不做checkpoint，
        #   所以调用方是主动选择在decode点做checkpoint，而不是主动退出。
        if not self.enable_prefix_caching:
            return
        base = seq.num_hashed_tokens
        if committed_kv_len > base:
            self.hash_blocks(
                seq,
                committed_kv_len - base,
                start_tokens=base,
                next_forward_tokens=next_forward_tokens,
                # Generation cannot choose where a step ends, least of all a
                # speculative one, so it is held to spacing rather than to the
                # grid.
                # 【译】生成阶段无法选择一步在哪里结束，投机步更是如此，所以它受间距约束而不受网格约束。
                # 解释：见checkpointers_at的unaimed分支（hash block边界 + 离上一个checkpoint ≥ interval）
                aimed=False,
            )

    # 【分支】每请求状态的fork（只用于"rolling"状态类StateSlotPool，即StateTransfer.fork；第4批state_pool.py）
    # 前置：GDN等线性注意力层每请求只有一份定长状态（Qwen3-Next一层约52万个数），放在状态池的一个"槽位"里。走fork的后端
    #   做checkpoint和恢复都靠"换槽位"，不另外拷贝：
    #     做checkpoint（StateSlotPool.checkpoint）：forward结束在边界P时，当前槽位S里正是P处的状态 → 把S交给hash索引，
    #       变成只读的checkpoint，序列换到新槽位S'；下一次forward从S读初始状态、写进S'
    #     从checkpoint恢复（_attach_state_slots）：新请求拿新槽位S'，第一次forward从checkpoint槽位S读、写进S'
    #   两种都叫fork：seq.state_fork_src = S。fork之后的那次forward至少要带min_fork_tokens个token才能把S'写完整
    # 本函数：那次forward太短时取消fork——不写新槽位，直接把S接过来当自己的槽位（S上的checkpoint随之作废）：
    #   做checkpoint的那种 → 等于没存；恢复的那种 → 变成独占式命中（S不再留给别人）
    #   S还被同一步的另一个请求读（pin_count > 1）时不能接管，返回False，调用方改为把chunk撑到min_fork_tokens
    # 调用方：Scheduler._finalize_prefill_chunk（chunk < state.min_fork_tokens 时）
    # 观察：现有走fork的后端都声明fork(1)（gdn_attn.py、kimi_mla_gdn_attn.py），min_fork_tokens = 1，而chunk总 ≥ 1，
    #   所以目前只有fork_tokens > 1的后端（如tests/test_state_checkpoint.py里的MIN_FORK = 8）才会走到取消这一步
    def cancel_state_fork(self, seq: Sequence) -> bool:
        """Undo a pending fork by adopting its source slot.

        Called when the forward that was going to carry the fork turns out too
        short to fill a fresh slot (`min_fork_tokens`). Both flavours collapse
        to the same move — take the source over as this seq's committed slot and
        spend its checkpoint: a resume becomes the non-sharing hit, a checkpoint
        becomes no checkpoint at all. Only the committed slot changes hands; the
        speculation scratch stays where it is, since the source never carried
        any.

        Returns False when the source cannot be taken over because another
        request in this same step forks off it too: adopting means writing into
        a slot that request's forward still has to read. The caller keeps the
        fork instead and must not shorten the forward below `min_fork_tokens`.
        """
        # 【译】通过接管源槽位来撤销一次待执行的fork。
        #
        #   当本来要承载这次fork的forward短到填不满一个新槽位（`min_fork_tokens`）时调用。两种fork都归结为同一个动作——
        #   把源槽位接过来当这条seq的已提交槽位，并花掉它上面的checkpoint：一次恢复变成不共享的命中，一次checkpoint
        #   变成根本没有checkpoint。只有已提交槽位易手；投机解码的scratch槽位留在原处，因为源槽位从来没带过它们。
        #
        #   如果同一步里还有另一个请求也要从这个源fork，就无法接管，返回False：接管意味着往一个那个请求的forward
        #   还要读的槽位里写。此时调用方保留fork，并且不能把forward缩短到 `min_fork_tokens` 以下。
        src = seq.state_fork_src
        if src < 0:
            return True
        if self.state.pin_count(src) > 1:
            return False
        # 新槽位S'还回空闲列表；invalidate：把S上的checkpoint从hash索引里摘掉（这条序列马上要接着往S里写）
        self.state.release(seq.state_slot)
        self.state.invalidate(src)
        # Both flavours of source are pinned — held off the free list for the
        # forward that has to read them — so taking one over is just dropping
        # this request's claim on it. It used to matter which flavour it was:
        # `checkpoint` handed its source straight back, so adopting it meant
        # claiming it off the free list, and `pin_count` then undercounted the
        # readers this refuses to overwrite.
        # 【译】两种源都是被pin住的——为了必须读它们的那次forward而不放回空闲列表——所以接管一个源只是丢掉本请求对它的认领。
        #   以前这和源是哪一种有关：`checkpoint` 会把它的源直接还回去，所以接管它意味着从空闲列表里把它认领回来，
        #   而 `pin_count` 随之会少算本函数拒绝覆盖的那些读者。
        self.state.unpin(src)
        seq.state_slot = src
        seq.state_fork_src = -1
        return True

    # ════════ 【分支】状态checkpoint阶梯（ladder）：存在哪些位置、什么时候切chunk、什么时候真的存（B级，第4批） ════════
    # 【科普】前置（要点；完整讲解见arg_utils.py EngineArgs.state_checkpoint_interval_tokens上方的【科普】、
    #   sequence.py Replay类上方的"科普：状态checkpoint与有界重放"）
    #   ① MHA/GQA的KV按token一行一行存，前缀命中到哪就从哪接着算。线性注意力层（GDN：Qwen3-Next / Qwen3.5；KDA：Kimi-K3）
    #      每请求只有一份定长状态S（Qwen3-Next一层GDN ≈ 32头×128×128 ≈ 52万个数），每来一个token原地覆盖；DeepSeek-V4的
    #      128窗口环形缓冲和压缩器没凑满的尾巴也一样。算到位置P时"P处的状态"只存在一瞬间
    #   ② 想让以后的请求从P接上，就得当时把状态另存一份 = 状态checkpoint，归档在"块P/hbs−1的链式hash"名下——和KV block
    #      共用一条hash链：KV命中到第k块后，再问状态池"h_{k-1}名下有没有快照"（_gated_hit），没有就往回找更短的
    #   ③ 状态一般只在forward的最后一个token处是完整的（中间位置的状态在kernel里一闪而过），所以要在P处存，就得让某次
    #      forward恰好结束在P：把prefill chunk截短到P——多一次forward，这是放checkpoint的主要代价
    #      （例外：能在forward中途读快照的后端，见下方midstep_positions上方的说明）
    #   ④ 每个checkpoint都占一份状态大小的存储（PAGE路径从KV池借单元，fork路径占一个状态槽位），所以只放在少数位置
    #
    #   三种位置（都是hash_block_size的整数倍，否则算不出hash，存了也找不到）：
    #     阶梯（rung）   每interval个token一级（--state-checkpoint-interval-tokens，默认8192）："猜"复用会从哪里接上。
    #                   一次forward只取不超过它终点的最右一级（同一个类，右边一级覆盖左边一级），且不超过checkpoint_limit
    #     需求（demand） 准入时KV命中被状态门控截短，就把"本来能命中到的位置"记成这条序列自己的一级（_record_checkpoint_demand）：
    #                   发现缺口的请求反正要从更低处重算，顺手在缺口处存一份，下一个同前缀的请求就能接满
    #     锚点（anchor） 本prompt的末尾，退到hash block边界（_record_checkpoint_end）：多轮对话的下一轮
    #                   "上一轮prompt + 回答 + 新问题"正好从这里接上（源码实测：cc-traces里93.5%的恢复落在上一轮prompt末尾）
    #   interval取值：>0 阶梯开；0 checkpoint全关；-1 只有demand和锚点（见__init__里state_checkpoint_interval_tokens处）
    #
    #   谁做什么：
    #     准入（can_allocate → _evaluate）    _record_checkpoint_demand、_record_checkpoint_end：记下demand / 锚点位置
    #     调度（Scheduler._finalize_prefill_chunk）  checkpoint_cut：在(起点, 终点]里挑最早的候选，把chunk截到那里
    #     forward之后（hash_blocks）          checkpointers_at：forward结束在pos，哪些状态类在这里存
    #     两边必须逐个位置一致：切了却不存 = 白多一次forward；该存的位置没切 = forward越过去，状态已不是那里的了
    #
    #   例（DeepSeek-V4：hash_block_size = 256，copy式PAGE checkpoint、successor_room = 0、不能中途读；interval 8192，
    #     每步预算max_num_batched_tokens = 16384，ATOM_STATE_CHECKPOINT_END_MARGIN = 0；数字按本区段各函数的公式推算）
    #   请求A：40000个token的prompt，冷缓存
    #     准入：checkpoint_limit = (40000 − 0)//8192×8192 = 32768；锚点 = (40000 − 0)//256×256 = 39936
    #           （上限 (ceil(40000/256) − 1)×256 = 156×256 = 39936，不变）；demand = 0（没有被拒绝的命中）
    #     ┌────┬───────┬──────────┬──────────────────────────────┬────────────────┬─────────────────┬────────────────────────┐
    #     │ 步 │ 起点  │ 预算终点 │ (起点, 终点]里的候选         │ checkpoint_cut │ 本步forward     │ 结束处存不存           │
    #     ├────┼───────┼──────────┼──────────────────────────────┼────────────────┼─────────────────┼────────────────────────┤
    #     │ 1  │ 0     │ 16384    │ 阶梯16384（8192被它覆盖）    │ 16384 = 终点   │ [0, 16384)      │ 存（在网格上）         │
    #     │ 2  │ 16384 │ 32768    │ 阶梯32768                    │ 32768 = 终点   │ [16384, 32768)  │ 存                     │
    #     │ 3  │ 32768 │ 40000    │ 锚点39936（32768不在区间内） │ 39936，截短    │ [32768, 39936)  │ 存（= 锚点）           │
    #     │ 4  │ 39936 │ 40000    │ 无                           │ 0              │ [39936, 40000)  │ 不存（不在网格上）     │
    #     └────┴───────┴──────────┴──────────────────────────────┴────────────────┴─────────────────┴────────────────────────┘
    #     → A留下3个checkpoint：16384、32768、39936（步3计入chunks_cut_for_end）；8192、24576从没成为某次forward的终点，所以没有
    #       （预算若够一步算完整个prompt：步1的候选是32768和39936，取最早的32768 → 只留32768、39936，共3次forward）
    #   请求B：与A共享前30000个token，总长也是40000
    #     准入：KV链逐块命中到块116（117块 = 29952个token；块117 = [29952, 30208)含分叉点30000，未命中）→ compressed_hit = 117
    #           状态门控：117块以内A只在16384（前64块的边界）有checkpoint → hit = 64块，B从16384开始prefill
    #           假设处处都有checkpoint（assume_checkpointed）能命中117块 → demand = 117×256 = 29952
    #     步1：起点16384、预算终点32768，候选{阶梯32768, demand 29952} → 截到29952（chunks_cut_for_demand + 1），存
    #     步2：[29952, 32768) 落在阶梯上，存；步3：[32768, 39936) 落在锚点上，存；步4：[39936, 40000)
    #   请求C：又一个共享前30000个token的请求 → 直接命中到29952（B在缺口处补的那一级），不再产生demand
    #     ——"发现一次、填一次、就收敛"（_record_checkpoint_demand的docstring）
    #   decode阶段（hash_decode_blocks，aimed=False）：不要求落在网格上，只要在hash block边界上、且离上一个checkpoint
    #     ≥ interval：A的prompt最后一个checkpoint在39936 → 生成到48128（= 39936 + 8192，正好是256的倍数）时再存一个
    # 对比nano-vllm：nano-vllm只支持纯KV的模型，前缀命中就是KV命中，没有这一整套
    def checkpoint_limit(self, seq: Sequence) -> int:
        """Rightmost prompt position any state class may checkpoint at, 0 none.

        `checkpointers_at` solved for prefill: the last rung of the ladder that
        still leaves the widest-reaching class its `successor_room` of prompt to
        forward. Kept as its own method because the scheduler needs the bound up
        front, to cut prefill chunks so they land on the ladder.

        0 means the grid places no rung on this prompt — every prompt shorter
        than one interval, among others. It does not mean the seq keeps
        nothing: a demand rung sits outside the grid and `checkpoint_cut`
        takes it either way.
        """
        # 【译】任何状态类可以做checkpoint的最右prompt位置，0表示没有。
        #
        #   这是针对prefill求解的 `checkpointers_at`：阶梯上的最后一级，它仍给伸得最远的那个类留下 `successor_room` 个prompt
        #   token去forward。单独做成一个方法，是因为调度器需要提前知道这个上界，好把prefill chunk切得落在阶梯上。
        #
        #   0表示网格在这条prompt上放不下任何一级——所有短于一个间隔的prompt都是如此，还有别的情况。它不表示这条seq什么都不存：
        #   demand那一级在网格之外，`checkpoint_cut` 无论如何都会取它。
        # 解释：successor_room = 存完checkpoint之后的那次forward至少要带的token数：PAGE（copy）类为0，fork(n)类为n，
        #   replay / none类为inf（根本不能存）。limit = (prompt长度 − room) // interval × interval
        #   例：P = 40000、PAGE类 → 40000//8192×8192 = 32768；P = 8192、fork(1)类 → (8192 − 1)//8192×8192 = 0——在8192处存的话，
        #     prompt已没有token留给后继forward去写新槽位；锚点、demand不受这个上限约束
        interval = self.state_checkpoint_interval_tokens
        if interval <= 0:
            return 0
        # The smallest room reaches furthest right, and `inf` — no class can
        # checkpoint this seq at all — falls out as 0 without a special case.
        # 【译】最小的room伸得最远；`inf`——没有任何类能给这条seq做checkpoint——不用特判就会得到0。
        # 观察：紧接着的 `if isinf(room)` 正是一个特判，而且必需：(P − inf)//interval 得到nan，int(nan) 会抛ValueError。
        #   没有任何类适用时 min 取 default=inf，同样走这条特判
        room = min(
            (c.successor_room for c in self.state_caches if c.applies(seq)),
            default=inf,
        )
        if isinf(room):
            return 0
        return max(int((seq.num_prompt_tokens - room) // interval) * interval, 0)

    def _record_checkpoint_demand(
        self,
        seq: Sequence,
        hit: int,
        compressed_hit: int,
        block_hashes: list[int],
        live_blocks: int,
        protected_hash: int | None,
    ) -> None:
        """Ask the hit counterfactually, and turn the gap into a rung.

        Whenever the gates cut a hit short, the same question is worth asking a
        second time with every ladder dense: how far would it have reached? What
        that recovers is reuse being declined only because nobody checkpointed
        there. What it does not recover is gone whatever anybody stores. The two
        land in `num_wanted_hit_blocks` (which `EngineStats` splits the declined
        reuse by) and `checkpoint_demand_pos` (which the ladder acts on).

        The demand is a rung of this seq's own, off the interval grid, and the
        seq that found the gap is the one best placed to fill it: it collects
        none of that reuse and has to compute the prefix regardless.

        Decided here, with both numbers in hand, rather than by the readers:
        `hit` survives only as `seq.num_cached_tokens`, which the scheduler
        advances as chunks land — under pipeline parallelism it is already past
        this chunk by the time `hash_blocks` runs, so a reader comparing against
        it would drop the demand on exactly the forward that was cut for it.

        A demand is not measured against the interval. The grid guesses
        where reuse will resume; a demand is reuse that was asked for and
        refused, and the granularity of the guess is no reason to discard it —
        gating one by the other left every prompt shorter than an interval
        declining all the reuse it had. The position comes from the same
        forkable test as the hit, so it always satisfies
        `num_prompt_tokens - pos >= successor_room`: somebody can really keep it.

        The shape that pays for this is a template header whose checkpoint is
        invalidated before anyone reaches it — there each request cuts a chunk
        and none collects. What bounds that is convergence rather than a
        threshold: found once, filled once, gone. `chunks_cut_for_demand`
        against `demands_recorded` is where it would show if it did not.

        `--no-state-checkpoint-demand` drops the rung and leaves the prompt-end
        anchor alone. Worth measuring: the rung is most of the write traffic
        and little of the read-back (`StateSlotPool.mark_speculative` has the
        numbers), and each write evicts something. Whether *removing* those
        writes beats merely demoting them is the open question — the CPU replay
        scores the two identically, which is the signature of a harness that
        models eviction order but not the cost of the write.
        """
        # 【译】以反事实的方式再问一次命中，并把差距变成一级阶梯。
        #
        #   每当门控把命中截短，都值得在"所有阶梯都稠密"的前提下把同一个问题再问一遍：本来能到多远？这样找回的是只因为没人
        #   在那里做checkpoint才被拒绝的复用；找不回的，是不管谁存什么都已经没了的部分。两者分别记进 `num_wanted_hit_blocks`
        #   （`EngineStats` 用它拆分被拒绝的复用）和 `checkpoint_demand_pos`（阶梯据此行动）。
        #
        #   demand是这条seq自己的一级，不在间隔网格上；而发现缺口的seq正是最适合填它的那一个：它拿不到这部分复用，
        #   反正都得把这段前缀算一遍。
        #
        #   在这里、两个数都在手边时就做决定，而不是交给读取方：`hit` 之后只以 `seq.num_cached_tokens` 的形式存在，而调度器会
        #   随着chunk落地推进它——在流水线并行下，等 `hash_blocks` 运行时它已经越过了这个chunk，所以拿它比较的读取方恰恰会在
        #   为它而截短的那次forward上把demand丢掉。
        #
        #   demand不用间隔来衡量。网格是在猜复用会从哪里接上；demand是请求过却被拒绝的复用，猜测的粒度不是丢弃它的理由——
        #   用一个去卡另一个，曾让所有短于一个间隔的prompt把它们本有的复用全部拒掉。这个位置和命中出自同一个可fork测试，
        #   所以总满足 `num_prompt_tokens - pos >= successor_room`：确实有人能存下它。
        #
        #   为此付代价的形状是一个模板头部，它的checkpoint在任何人用到之前就被作废了——此时每个请求都切一刀chunk，却没有一个
        #   能用上。约束它的是收敛而不是阈值：发现一次、填一次、就没了。如果不收敛，会体现在 `chunks_cut_for_demand`
        #   相对 `demands_recorded` 的比值上。
        #
        #   `--no-state-checkpoint-demand` 去掉这一级，只保留prompt末尾锚点。值得测一测：这一级占了大部分写流量，却只带来很少
        #   的读回（`StateSlotPool.mark_speculative` 里有数字），而且每次写都会淘汰点什么。到底*去掉*这些写更好，还是只*降级*
        #   它们更好，仍是悬而未决的问题——CPU重放给两者打了同样的分，这正是一个只模拟淘汰顺序、不模拟写入成本的测试框架的特征。
        # 解释（数字见上方checkpoint_limit处【科普】的请求B）：compressed_hit = 117块（KV命中）、hit = 64块（状态门控之后）
        #   → 反事实命中 wanted = _gated_hit(assume_checkpointed=True) = 117 → demand = 117 × hash_block_size = 29952
        #   调用方：_evaluate（can_allocate与can_allocate_memoized共用），在算完本次准入要拿的num_new_blocks之后
        wanted = (
            self._gated_hit(seq, compressed_hit, block_hashes, assume_checkpointed=True)
            if hit < compressed_hit
            else hit
        )
        seq.num_wanted_hit_blocks = wanted
        # Zero switches checkpointing off entirely — `checkpointers_at` keeps
        # nothing then, so a cut for a demand would buy nothing either. -1 is
        # the other thing: the interval grid is off but checkpointing is on, and
        # the demand rung is one of the two placements that then carries it.
        # 【译】0会彻底关掉checkpoint——那时 `checkpointers_at` 什么都不存，所以为demand切chunk也换不来任何东西。-1是另一回事：
        #   间隔网格关了但checkpoint还开着，此时demand这一级正是承担它的两种放置之一。
        interval_on = self.state_checkpoint_interval_tokens != 0
        demand = (
            wanted * self.hash_block_size
            if interval_on and self.state_checkpoint_demand and wanted > hit
            else 0
        )
        # A replayed hit (`_replay_hit`) reaches `wanted` already, and the
        # replay bounds what the missing checkpoint costs. A demand would cut
        # every replay admission's forward at the hit and write a checkpoint
        # there, for a branch at that exact hit that may never come:
        # conversations resume at prompt ends, which the anchor covers, the
        # hit included (`scheduler._cut_floor` lets it land there).
        # 【译】重放式命中（`_replay_hit`）本来就够到 `wanted`，而且重放限定了缺checkpoint的代价。再放demand的话，会把每个重放准入
        #   的forward都截在命中处、并在那里写一个checkpoint，只为了一个也许永远不会来的、恰好在这个命中处分叉的请求：对话都在
        #   prompt末尾恢复，那里由锚点覆盖，命中处也包括在内（`scheduler._cut_floor` 允许切点落在那里）。
        if seq.replays_hit:
            demand = 0
        # A demand is an instruction to cut a prefill chunk onto a rung, and
        # that cut costs the request a forward. Buying one for a store
        # `begin_store` is about to refuse is the only part of this funnel
        # that is pure loss — the attribution above stays either way, because
        # the reuse really was declined for want of a checkpoint.
        #
        # Asked afresh on every attempt, because that is the question: a
        # demand recorded while the pool had room is not still affordable once
        # it does not, and letting the earlier answer stand is exactly the cut
        # this gate exists to withhold. What must not repeat is the *counting*,
        # which is why the seq carries its own marker rather than the gate
        # reading the position it is about to overwrite.
        #
        # Asked with this admission's own blocks included, because they are
        # taken first: a pool with room for an image but not for the request
        # *and* the image would answer yes here and refuse at `begin_store`,
        # with the cut already bought and the funnel showing nothing.
        #
        # It is still a sample. The store happens many forwards later, at the
        # rung this cut creates, against a pool that has moved since — no
        # question asked here can be the one `begin_store` asks. What this
        # gate removes is the loss that was knowable at admission;
        # `checkpoints_dropped` is what counts the rest, and the two are meant
        # to be read together.
        # 【译】demand是"把一个prefill chunk截到某一级上"的指令，这一刀要请求多付一次forward。为一次 `begin_store` 马上就会拒绝的
        #   存储买这一刀，是整个漏斗里唯一纯粹的损失——上面的归因无论如何都保留，因为复用确实是因缺checkpoint而被拒绝的。
        #
        #   每次尝试都重新问，因为这才是要问的问题：池子有空间时记下的demand，在池子没空间之后就不再负担得起，让先前的答案继续
        #   生效，恰恰就是这道门要扣下的那一刀。不能重复的是*计数*，所以seq自己带一个标记，而不是让门去读它正要覆盖的位置。
        #
        #   问的时候把这次准入自己的block算进去，因为它们先被拿走：一个放得下一份镜像、却放不下"请求 + 镜像"的池子，在这里会
        #   回答"能"，然后在 `begin_store` 拒绝——那一刀已经买了，漏斗里却什么都看不出来。
        #
        #   它仍然只是一次采样。存储发生在许多次forward之后、在这一刀造出的那一级上，面对的池子早已变了——这里问的任何问题都不
        #   可能就是 `begin_store` 将来问的那个。这道门去掉的是准入时就能知道的损失；其余的由 `checkpoints_dropped` 计数，
        #   两者要放在一起看。
        # 解释：_checkpoint_has_room只对PAGE checkpoint有意义（镜像从KV池借单元）；fork类checkpoint不占额外容量，总返回True
        if demand and not self._checkpoint_has_room(live_blocks, protected_hash):
            self.demands_declined_no_room += not seq.checkpoint_demand_declined
            seq.checkpoint_demand_declined = True
            demand = 0
        seq.checkpoint_demand_pos = demand
        # Counted when the demand first appears rather than once per attempt —
        # otherwise one deferred request inflates the denominator the
        # convergence check above is read against. A separate marker from the
        # decline above: a decline zeroes the position, so the position alone
        # would let a recorded demand be counted twice the next time the pool
        # has room.
        # 【译】在demand第一次出现时计数，而不是每次尝试都计——否则一个被推迟的请求会抬高上面收敛检查所对照的分母。
        #   和上面的拒绝用不同的标记：一次拒绝会把位置清零，只看位置的话，一个已记录的demand会在下次池子有空间时被算两次。
        if demand:
            self.demands_recorded += not seq.checkpoint_demand_counted
            seq.checkpoint_demand_counted = True

    def _record_checkpoint_end(self, seq: Sequence) -> None:
        """Reserve this prompt's own end as a resume point.

        The ladder guesses where reuse will resume and the demand learns it one
        refusal late. Neither covers the case that dominates agentic traffic: a
        conversation whose next turn resends this whole prompt and continues
        past it. That resumes at *this* prompt's end, and the only request in a
        position to leave a checkpoint there is this one.

        Measured on the SemiAnalysis cc-traces (4,808 resumes with a nonzero KV
        hit): 93.5% land on a previous prompt end, 0.0% on the 8192 ladder.

        Off the grid by construction, like a demand: `checkpoint_limit` bounds
        the *ladder*, and a prompt shorter than one interval places no rung at
        all while still being a perfectly good thing to resume from. The
        `successor_room` test cannot be skipped though — it is what makes the
        position keepable, and `checkpointers_at` re-checks it against the
        forward that actually lands. Applying it here as well keeps the cut and
        the keep from disagreeing, which is the standing contract between them.
        """
        # 【译】把本prompt自己的末尾保留为一个恢复点。
        #
        #   阶梯在猜复用会从哪里接上，demand要晚一次拒绝才学到。两者都覆盖不了agentic流量里占主导的情形：一段对话的下一轮把
        #   整个prompt重发一遍、再接着往后写。那会在*这个*prompt的末尾恢复，而唯一有机会在那里留下checkpoint的请求就是这一个。
        #
        #   在SemiAnalysis的cc-traces上测得（4,808次KV命中非零的恢复）：93.5%落在上一个prompt的末尾，0.0%落在8192阶梯上。
        #
        #   和demand一样，它按构造就不在网格上：`checkpoint_limit` 约束的是*阶梯*，而一个短于一个间隔的prompt一级都放不下，
        #   却仍然完全可以作为恢复点。但 `successor_room` 测试不能省——正是它让这个位置能被存下，而 `checkpointers_at` 会针对
        #   真正落地的forward再检查一次。这里也做一次，能让"切"和"存"不打架，这是两者之间的长期约定。
        # 解释：锚点 = min( (P − room)//hbs×hbs , (ceil(P/hbs) − 1)×hbs )，room = max(successor_room, min_final_prefill_rows,
        #   ATOM_STATE_CHECKPOINT_END_MARGIN)。例：DeepSeek-V4，P = 40000、hbs = 256、room = 0 → 39936（上方【科普】的表）
        #   调用方：_evaluate，每次准入都重算；deallocate也会把它清零（理由见那里）
        seq.checkpoint_end_pos = 0
        if not self.state.enabled:
            return
        # 0 is the off switch for checkpointing as a whole, not just the grid,
        # and `checkpointers_at` enforces it by refusing every position. An
        # anchor recorded here anyway would still be cut for — a shortened
        # prefill chunk on every prompt, storing nothing, with no error to
        # show for it. Pinned by `test_interval_zero_anchors_nothing`.
        # 【译】0是整个checkpoint功能的关闭开关，而不只是关掉网格，`checkpointers_at` 通过拒绝每个位置来执行它。若照样在这里记下
        #   锚点，调度器仍会为它切chunk——每条prompt都多切一刀prefill chunk，什么都不存，也没有任何报错。
        #   由 `test_interval_zero_anchors_nothing` 钉住。
        if self.state_checkpoint_interval_tokens == 0:
            return
        room = min(
            (c.successor_room for c in self.state_caches if c.applies(seq)),
            default=inf,
        )
        if isinf(room):
            return
        # A forward resuming there and finishing the prompt with fewer rows
        # than the model's final forward runs would back off below it
        # (`replay_start`), recomputing what the checkpoint saves.
        # 【译】一个从那里恢复、而结束prompt时的行数少于模型最后一次forward所需行数的forward，会退到它下方（`replay_start`），
        #   把checkpoint省下的部分又重算一遍。
        # 解释：min_final_prefill_rows只有DeepSeek-V4.1设（= sliding_window，见config.py），其余模型为0
        room = max(room, self.min_final_prefill_rows)
        # The next turn re-renders this prompt's last few tokens (the chat
        # template's generation prompt), so an anchor must sit before them too.
        # Measured on Qwen3.5 AgentX: with no margin, prompts of length
        # 16k + 1 put the divergence inside the anchor's block, and their
        # follow-ups resumed from the anchor 1 time in 282 (95.6% otherwise).
        #
        # The larger of the room and the margin, not their sum. Against no
        # margin the anchor moves back by up to `ceil((margin - room) / block)`
        # blocks -- one for Qwen3.5's 5 over a PAGE room of 0 -- so a margin
        # wider than a block costs a verbatim resend several blocks, and a
        # prompt shorter than `margin + block` keeps no anchor (`end` below is
        # 0). Pinned by `TestPromptEndAnchorTemplateMargin`.
        # 【译】下一轮会重新渲染这个prompt的最后几个token（chat template的生成提示），所以锚点也必须落在它们之前。
        #   在Qwen3.5 AgentX上测得：不留余量时，长度为16k + 1的prompt把分叉点放进了锚点所在的block，它们的后续请求282次里
        #   只有1次从锚点恢复（其他情况下是95.6%）。
        #
        #   取room和余量中较大的一个，而不是两者之和。与不留余量相比，锚点最多后退 `ceil((margin - room) / block)` 个block——
        #   对Qwen3.5的5、PAGE的room 0来说是一个——所以比一个block还宽的余量，会让原样重发的请求少复用好几个block，
        #   而短于 `margin + block` 的prompt一个锚点都留不下（下面的 `end` 为0）。由 `TestPromptEndAnchorTemplateMargin` 钉住。
        # 解释：例（hbs = 16）：P = 16385。不留余量：锚点 = 16385//16×16 = 16384，而下一轮在16380附近就分叉，分叉点落在
        #   块1023 = [16368, 16384)里，锚点的hash永远对不上；余量5：(16385 − 5)//16×16 = 16368，落在分叉点之前
        room = max(int(room), self.state_checkpoint_end_margin)
        # The rightmost grid position that still leaves `room` behind it.
        #
        # Not simply `num_prompt_tokens` floored: a checkpoint at P binds the
        # forward after it to carry `room` tokens, and the floored end leaves
        # at most `hash_block_size - 1`. So for any class whose room reaches a
        # block or more — MIN_FORK 8 against block 4 in the tests, V4's 131
        # against 256 — the exact end is never keepable and an anchor placed
        # there would be cut for and then refused by `checkpointers_at`.
        #
        # Stepping back to the last keepable boundary costs at most one block
        # of the next turn's reuse and is what makes the anchor exist at all.
        # Measured on the cc-traces, the step-back anchor is worth +0.3 to +0.7
        # points over insisting on the exact end, and is never worse.
        # 【译】仍给身后留下 `room` 的最右网格位置。
        #
        #   不是简单地把 `num_prompt_tokens` 向下取整：在P处的checkpoint要求它之后的forward带上 `room` 个token，而取整后的末尾
        #   最多只留下 `hash_block_size - 1` 个。所以对于room达到一个block或更多的类——测试里的MIN_FORK 8对block 4、V4的131对256——
        #   精确的末尾永远存不下，放在那里的锚点会被切出来、然后被 `checkpointers_at` 拒绝。
        #
        #   退到最后一个能存下的边界，最多让下一轮少复用一个block，却是让锚点存在的前提。在cc-traces上测得：退一步的锚点
        #   比坚持精确末尾多+0.3到+0.7个百分点，而且从不更差。
        # 观察："V4的131"是旧情况：现在DeepSeek-V4声明StateTransfer.copy、走PAGE checkpoint，successor_room = 0
        #   （deepseek_v4_attn.py、page_unit_checkpoint.py），见checkpointers_at的docstring之后的说明
        end = (
            (seq.num_prompt_tokens - int(room)) // self.hash_block_size
        ) * self.hash_block_size
        # And never past the last block anybody can match on. `can_allocate`
        # stops one block short of the prompt (prefill must forward a block to
        # produce logits), so a checkpoint filed under the final block's hash is
        # one no scan ever looks up. For the continuation this anchor is aimed
        # at — the next turn resends this prompt and keeps going — that block is
        # interior and would match, which is why the ceiling is not obvious. But
        # a *re-sent identical* prompt caps at `n - 1`, and there the uncapped
        # anchor is not merely useless: it is stored, so it evicts the ladder
        # rung that would have served the resume, and the hit goes to 0 rather
        # than merely getting no better. Capping costs the continuation at most
        # one block of reuse — the same trade `room` above already takes.
        # 【译】而且永远不超过任何人能匹配上的最后一个block。`can_allocate` 在prompt之前一个block就停下（prefill必须forward一个block
        #   才能产出logits），所以归档在最后一个block的hash名下的checkpoint，是任何扫描都不会去查的。对这个锚点瞄准的续写——
        #   下一轮重发这个prompt并继续往后写——那个block是内部的，能匹配上，这就是为什么这个上限并不显然。但一个*原样重发*的
        #   prompt最多命中到 `n - 1`，那时不封顶的锚点不只是没用：它会被存下，于是淘汰掉本可服务这次恢复的那一级阶梯，命中直接
        #   变成0，而不只是没变好。封顶让续写最多少复用一个block——和上面 `room` 已经做的是同一种交换。
        # 解释：n = _n_hash_blocks(seq) = ceil(P/hbs)；P = 40000、hbs = 16时n = 2500，上限 = 2499×16 = 39984
        end = min(end, (self._n_hash_blocks(seq) - 1) * self.hash_block_size)
        if end > 0:
            seq.checkpoint_end_pos = end

    def _extend_hash_chain(self, seq: Sequence, block_hashes: list[int]) -> None:
        """Cache the chained hash of *every* block of the prompt on `seq`.

        `block_hashes` stops at the first cache miss — it is the *hit*, and the
        gates read it as one. A midstep reservation names a position the forward
        has not reached yet, which is past that miss for every reservation worth
        making, so the chain it is named by has to run further.

        A separate list rather than an extension of `block_hashes` in place.
        Nothing downstream would misread a longer list today — `_gated_hit` and
        the free-pool loop both index it under `compressed_hit` — but "the
        blocks that hit" and "the blocks of the prompt" are two facts, and one
        list holding both means the next reader of either has to know which. It
        is also cheap: this runs after the gates, so the copy is of a list the
        callers are already done with.

        Only for a backend that reserves midstep. Everywhere else the hash a
        checkpoint is filed under is computed by `hash_blocks` walking the
        blocks the forward just finished, and this would be a hash pass over the
        whole prompt on every admission, spent on nothing.
        """
        # 【译】在 `seq` 上缓存prompt *每一个* block的链式hash。
        #
        #   `block_hashes` 在第一次缓存未命中处就停了——它是*命中*，门控也把它当命中来读。midstep预留指定的是forward还没到达的
        #   位置，对每个值得做的预留来说都在那个未命中之后，所以给它取名的链要延伸得更远。
        #
        #   用一个单独的列表，而不是原地扩展 `block_hashes`。今天下游没有谁会误读一个更长的列表——`_gated_hit` 和空闲池循环都
        #   只在 `compressed_hit` 以内索引它——但"命中的block"和"prompt的block"是两个事实，让一个列表同时装两者，意味着它们
        #   各自的下一个读者都得知道是哪一个。它也便宜：这在门控之后运行，所以拷贝的是一个调用方已经用完的列表。
        #
        #   只对会做midstep预留的后端才做。其他情况下，checkpoint归档用的hash由 `hash_blocks` 在遍历forward刚写完的block时算出，
        #   这里就会变成每次准入都对整个prompt做一遍hash，白白浪费。
        # 解释：结果存进seq.block_hashes，唯一读者是midstep_positions（给forward里的每个快照位置取hash名）
        if not self._midstep_applies(seq):
            seq.block_hashes = []
            return
        # The continuation itself -- resume from the last APPENDED hash (on a
        # miss the loop's `h` holds the hash of the block that failed to match,
        # which was never appended and is not part of this chain) -- is exactly
        # `_chain_to`, run to the full prompt width. Delegate rather than
        # duplicate the loop; the only thing local to this path is the midstep
        # gate above and writing the result to `seq`.
        # 【译】续算本身——从最后一个*追加进来的*hash接着算（未命中时循环里的 `h` 是那个没匹配上的block的hash，它从没被追加，
        #   不属于这条链）——恰好就是一直跑到整个prompt宽度的 `_chain_to`。委托给它而不是重复那个循环；这条路径自己的只有
        #   上面的midstep门槛，以及把结果写到 `seq` 上。
        seq.block_hashes = self._chain_to(seq, block_hashes, self._n_hash_blocks(seq))

    # ════════ 【分支】midstep checkpoint：在一次forward的中间取状态快照，不再为checkpoint切chunk（B级，第4批） ════════
    # 前置：GDN的prefill用分块（chunk）算法（fla_ops/chunk.py，CHUNK_SIZE = 64）：把本次forward的token每64个分一块，
    #   块与块之间按顺序递推状态h（块内用矩阵乘法并行），所以kernel本来就会算出"起点 + 64k"每个块边界处的状态，只是用完就丢。
    #   能把这些中间状态拷出来的后端声明 StateTransfer.readable_midstep = True。目前只有Qwen GDN的PAGE路径在
    #   ATOM_GDN_STATE_MIDSTEP=1 时这样声明（gdn_attn.py）：midstep_grid = 64；midstep_min_offset = conv_kernel − 1 = 3
    #   （GDN前面的短卷积窗口要从本次forward自己的输入里切，所以快照位置之前至少要有本forward的3个token）
    #
    #   不能中途读（默认）：checkpoint只能取forward末尾的状态 → checkpoint_cut把chunk截到候选位置上，多一次forward
    #   能中途读：checkpoint_cut返回0，chunk只按预算切；候选位置（同样是阶梯、demand、锚点）由本函数在forward之前列出，
    #     挪到本forward的64网格上（snap_midstep_position：end原样不动，其余退到"起点 + 64k"中不超过它的最大者，
    #     最多少复用63个token），在forward里拷出来
    #
    #   流程：Scheduler调度完本轮 ─► plan_midstep(seq, start, end) ─► reserve_midstep：每个位置借一个scratch状态槽位
    #         ─► forward（runner把那一刻的h和卷积窗口拷进scratch）─► postprocess → hash_blocks → commit_midstep
    #         ─► publish_midstep：排队，下一批构建时begin_store成PAGE checkpoint，归档在该位置的hash名下
    #         forward不跑了（抢占、取消）─► deallocate → cancel_midstep：scratch原样还回
    #
    #   例（Qwen3.5、hbs = 16、interval 8192、预算16384、40000个token的prompt、冷缓存；
    #     checkpoint_limit = 32768，锚点 = min(40000//16×16, (2500 − 1)×16) = 39984）：
    #     不开midstep（切chunk，同scheduler.py _chunked_prefill_size上方的情形二）：
    #       [0, 16384) [16384, 32768) [32768, 39984) [39984, 40000)：4次forward，checkpoint在16384、32768、39984
    #     开midstep：chunk不切，[0, 16384) [16384, 32768) [32768, 40000)：3次forward
    #       forward 1、2：阶梯16384、32768正好是end → 原样
    #       forward 3：锚点39984在(32768, 40000]里但不是end → 退到 32768 + (39984 − 32768)//64×64 = 39936
    #         （39936是16的倍数、离起点 ≥ 3 → 可读）→ checkpoint在39936，归档在块2495（= 39936/16 − 1）的hash名下
    #     代价：锚点从39984退到39936，下一轮少复用48个token；收益：少一次forward
    #   为什么要align_midstep_chunk：快照位置 = 起点 + 64k，只有起点本身在hash block边界上，快照位置才可能是hbs的整数倍
    #     （才有hash可用），所以非最后一块chunk的终点要对齐到block边界
    def midstep_positions(self, seq: Sequence, start: int, end: int) -> list[tuple]:
        """`(position, hash)` for every checkpoint a forward over `(start, end]`
        can take without being cut short.

        The midstep twin of `checkpoint_cut`. Same candidate positions — the
        grid rung, the demand, the prompt-end anchor — but where `checkpoint_cut`
        must pick *one* and shorten the chunk onto it, this returns all of them:
        a backend that reads its chunk kernel's interior states takes a snapshot
        at each without the forward ending there.

        `checkpointers_at` is not consulted. That function answers "the forward
        ended here, keep or not", and its `successor_room` test is about a fork's
        successor forward — which a midstep snapshot does not have, because
        nothing is handed over and the owner keeps writing where it is. Applying
        it here would refuse exactly the block-aligned prompt end the anchor
        exists to reserve.

        Every candidate is a multiple of `hash_block_size` by construction: the
        interval is snapped onto that grid at construction and the rung is a
        multiple of the interval, the demand is `blocks * hash_block_size`, and
        the anchor is floored to it. The assertion below states that rather than
        trusting it, because a position off the grid indexes the wrong block's
        hash — a checkpoint filed under a name no resumer will ever compute, and
        nothing downstream that could notice.

        No `interval == 0` early-out, unlike the three placements this reads.
        Each already carries the off switch — `checkpoint_limit` returns 0 for
        any non-positive interval, `_record_checkpoint_demand` gates on
        `interval_on`, `_record_checkpoint_end` returns before setting the
        anchor — so under 0 all three candidates are 0 and the loop has nothing
        to walk. A fourth check would read as the one enforcing it, and be the
        one nobody dared remove. `test_interval_zero_reserves_nothing` pins the
        behavior end to end rather than the check.
        """
        # 【译】一次覆盖 `(start, end]` 的forward在不被截短的情况下能取的每个checkpoint的 `(position, hash)`。
        #
        #   `checkpoint_cut` 的midstep孪生版本。候选位置相同——网格上的一级、demand、prompt末尾锚点——但 `checkpoint_cut` 必须
        #   挑*一个*并把chunk截到那里，这里则全部返回：能读出chunk kernel内部状态的后端在每个位置各取一份快照，forward不必在那里结束。
        #
        #   不参考 `checkpointers_at`。那个函数回答的是"forward在这里结束了，存不存"，它的 `successor_room` 测试关心fork的后继
        #   forward——而midstep快照没有后继，因为什么都不交接，所有者接着在原处写。在这里套用它，恰恰会拒掉锚点要保留的那个
        #   对齐到block的prompt末尾。
        #
        #   每个候选按构造都是 `hash_block_size` 的倍数：间隔在构造时已对齐到这个网格、一级是间隔的倍数，demand是
        #   `blocks * hash_block_size`，锚点向下取整到它。下面的断言把这点写明而不是信任它，因为一个不在网格上的位置会取到
        #   错误block的hash——一个以谁都不会算出来的名字归档的checkpoint，下游也没有任何东西能察觉。
        #
        #   和它读取的三种放置不同，这里没有 `interval == 0` 的提前返回。三者各自已带着开关——`checkpoint_limit` 对任何非正的间隔
        #   返回0，`_record_checkpoint_demand` 以 `interval_on` 为门，`_record_checkpoint_end` 在设置锚点之前就返回——所以取0时
        #   三个候选都是0，循环无可遍历。再加第四个检查，读起来会像是它在执行这个开关，成为没人敢删的那一个。
        #   `test_interval_zero_reserves_nothing` 端到端地钉住这个行为，而不是钉住某个检查。
        # 解释：返回按位置排序的[(pos, hash)]；调用方是plan_midstep。例见上方【分支】midstep的说明
        if not self._midstep_applies(seq):
            return []
        interval = self.state_checkpoint_interval_tokens
        hbs = self.hash_block_size
        grid, min_offset = self.midstep_grid, self.midstep_min_offset
        hashes = seq.block_hashes
        # 网格上的候选：不超过end、也不超过checkpoint_limit的最右一级（和checkpoint_cut同一算法）；
        #   out用dict按挪动后的位置去重：两个候选可能退到同一个网格点
        rung = 0
        if limit := self.checkpoint_limit(seq):
            rung = min(end, limit)
            rung -= rung % interval
        out = {}
        for pos in sorted(
            {p for p in (rung, seq.checkpoint_demand_pos, seq.checkpoint_end_pos) if p}
        ):
            if not start < pos <= end:
                continue
            assert not pos % hbs, f"checkpoint position {pos} is off the {hbs} grid"
            # The state the kernel materialized at or before `pos`. It must
            # still be a block boundary to have a hash, which holds exactly
            # when this forward starts on one -- the scheduler keeps non-final
            # chunk ends on the block grid (`align_midstep_chunk`) for that.
            # 【译】kernel在 `pos` 处或之前物化出来的那个状态。它必须仍在block边界上才有hash，而这恰好在本forward从block边界开始时
            #   成立——调度器为此把非最后一块chunk的终点保持在block网格上（`align_midstep_chunk`）。
            snapped = snap_midstep_position(pos, start, end, grid)
            if snapped % hbs or not midstep_target_ok(
                snapped, start, end, grid, min_offset
            ):
                self.midstep_positions_unreadable += 1
                continue
            nblocks = snapped // hbs
            # A position past the chain is one `_extend_hash_chain` could not
            # name — nothing to file it under, so it cannot be found again.
            # 【译】超出这条链的位置，是 `_extend_hash_chain` 无法命名的——没有名字可以归档，也就再也找不回来。
            if nblocks > len(hashes):
                continue
            out[snapped] = hashes[nblocks - 1]
        return sorted(out.items())

    def midstep_anchor(self, seq: Sequence, start: int, end: int) -> int:
        """Where this forward files the prompt-end anchor, or 0.

        `checkpoint_end_pos` moved onto the forward's grid, as
        `midstep_positions` moves it. The class that ranks checkpoints by
        whether they are the anchor compares against this, not the unsnapped
        position, or the anchor it actually filed is ranked as a guess.
        """
        # 【译】这次forward把prompt末尾锚点归档在哪里，没有则为0。
        #
        #   `checkpoint_end_pos` 挪到forward的网格上，和 `midstep_positions` 挪它的方式一样。按"是不是锚点"给checkpoint排优先级的
        #   那个类拿这个值比较，而不是拿没挪过的位置，否则它真正归档的锚点会被当成一次猜测来排序。
        # 解释：例（上方midstep的例子）：锚点39984在forward (32768, 40000]里 → 返回39936。plan_midstep把它存进
        #   seq.midstep_anchor_pos，由StateSlotPool判断"是不是锚点"时读取（state_pool.py）
        anchor = seq.checkpoint_end_pos
        if not (anchor and start < anchor <= end):
            return 0
        return snap_midstep_position(anchor, start, end, self.midstep_grid)

    # 状态checkpoint类（PAGE协调器或StateSlotPool）能否在forward中途读快照、且适用于这条序列
    def _midstep_applies(self, seq: Sequence) -> bool:
        cache = self._state_checkpoint_cache
        return cache.readable_midstep and cache.applies(seq)

    def align_midstep_chunk(self, seq: Sequence, start: int, chunk: int) -> int:
        """`chunk` shortened so a non-final prefill chunk ends on a block boundary.

        A readable class needs no cut onto its rungs, so the budget alone ends
        a chunk -- anywhere. The next chunk would then start off the block
        grid, every position on *its* grid would be off it too, and
        `midstep_positions` would have to drop them all: for a prompt longer
        than one batch, the anchor along with them. Giving back under one
        block of this forward's budget keeps the next start on the grid.

        Unchanged when the chunk finishes the prompt, or when it is shorter
        than reaching the next boundary.
        """
        # 【译】把 `chunk` 缩短，让非最后一块的prefill chunk结束在block边界上。
        #
        #   能中途读的类不需要把chunk切到它的阶梯上，所以只有预算决定chunk在哪里结束——哪里都可能。那样下一块就会从block网格之外
        #   开始，*它的*网格上的每个位置也都不在block网格上，`midstep_positions` 只能把它们全部丢掉：对一条超过一个batch的prompt，
        #   锚点也一起丢了。让出本forward预算里不到一个block的量，就能让下一块的起点留在网格上。
        #
        #   chunk会结束这个prompt、或短到够不到下一个边界时，原样返回。
        # 解释：调用方Scheduler._finalize_prefill_chunk / _preview_prefill_chunk（checkpoint_cut返回0时）。
        #   例（hbs = 16）：start = 0、chunk = 10000（非最后一块）→ 10000//16×16 = 9984；start = 0、chunk = 10 → 对齐后为0 → 原样10
        if not self._midstep_applies(seq) or start + chunk >= seq.num_tokens:
            return chunk
        hbs = self.hash_block_size
        aligned = (start + chunk) // hbs * hbs - start
        return aligned if aligned > 0 else chunk

    def plan_midstep(self, seq: Sequence, start: int, end: int) -> None:
        """Reserve a destination slot for every checkpoint this forward covers.

        Called once the chunk `(start, end]` is settled and before the batch is
        built, which is the only window where both are true: the positions are
        known, and the free list can still be committed against without racing
        an admission later in the same scheduling pass.

        Overwrites rather than accumulates. A reservation is good for exactly
        one forward, so a second plan for the same seq means the first forward
        never ran — hand its slots back rather than leak them.
        """
        # 【译】为这次forward覆盖的每个checkpoint预留一个目标槽位。
        #
        #   在chunk `(start, end]` 已经确定、batch还没构建时调用，这是两件事同时成立的唯一窗口：位置已知，而空闲列表仍可以被承诺出去，
        #   不会和同一轮调度里稍后的准入抢资源。
        #
        #   覆盖而不是累积。一个预留只对恰好一次forward有效，所以同一条seq的第二次规划意味着第一次forward根本没跑——
        #   把它的槽位还回去，而不是泄漏掉。
        # 解释：调用方Scheduler._schedule（本轮所有准入结束之后、构建prefill batch之前，对每条调度到的序列调用）；
        #   seq.midstep_reservations = [(scratch槽位, 位置, hash), ...]，ScheduledBatch把(槽位, 位置)交给runner
        if not self._state_checkpoint_cache.readable_midstep:
            return
        if seq.midstep_reservations:
            self.cancel_midstep(seq)
        seq.midstep_reservations = self._state_checkpoint_cache.reserve_midstep(
            seq, self.midstep_positions(seq, start, end)
        )
        seq.midstep_anchor_pos = self.midstep_anchor(seq, start, end)

    def commit_midstep(self, seq: Sequence) -> None:
        """File this forward's reservations, now that its bytes exist.

        Drains the list: whatever is not committed here is not a checkpoint,
        and leaving it in place would let the next `plan_midstep` publish a
        position this seq has already moved past.
        """
        # 【译】把这次forward的预留归档，因为它的字节现在已经存在了。
        #
        #   会清空这个列表：这里没提交的都不算checkpoint，留着不动会让下一次 `plan_midstep` 发布一个这条seq早已越过的位置。
        # 解释：调用方是hash_blocks末尾（forward完成之后）
        if not seq.midstep_reservations:
            return
        self._state_checkpoint_cache.publish_midstep(seq.midstep_reservations, seq)
        # The rightmost published position, for the decode-side spacing rule in
        # `checkpointers_at`. Generation is never midstep-readable — its
        # forwards end where acceptance puts them — so it still measures from
        # here, and would otherwise re-checkpoint immediately after a prompt
        # that just filed one.
        # 【译】已发布位置里最右的那个，给 `checkpointers_at` 里decode侧的间距规则用。生成阶段从不中途读——它的forward在接受结果
        #   决定的地方结束——所以它仍从这里开始量，否则会在一个刚归档过checkpoint的prompt之后立刻又做一次checkpoint。
        seq.last_checkpoint_pos = max(
            seq.last_checkpoint_pos, max(p for _g, p, _h in seq.midstep_reservations)
        )
        seq.midstep_reservations = []
        seq.midstep_anchor_pos = 0

    def cancel_midstep(self, seq: Sequence) -> None:
        """Hand back reservations whose forward is not going to run."""
        # 【译】把那些forward不会再运行的预留交还回去。
        # 解释：调用方是plan_midstep（重新规划前）和deallocate（抢占 / 结束）；scratch槽位原样空着还回状态池
        if not seq.midstep_reservations:
            return
        self._state_checkpoint_cache.cancel_midstep(seq.midstep_reservations)
        seq.midstep_reservations = []
        seq.midstep_anchor_pos = 0

    def checkpoint_cut(
        self, seq: Sequence, start: int, end: int, *, record: bool = True
    ) -> int:
        """Earliest ladder position in `(start, end]`, or 0 if there is none.

        What a prefill chunk is cut at so its forward lands exactly on a rung.
        The counterpart of `checkpointers_at`, which decides what a forward
        ending there keeps: the two have to agree position for position, so the
        grid arithmetic lives here rather than at the scheduler's call site.

        Earliest, not latest, because a prompt can now want two positions and
        the chunk loop reaches the second one only by being handed the first.
        Within the grid a later rung still dominates an earlier one — one class,
        one resume point, take the rightmost — so the grid collapses to a single
        candidate before the choice is made. The prompt-end anchor is the
        exception that forces the loop: it does not dominate the rung and the
        rung does not dominate it. See `_record_checkpoint_end`.

        0 once every class that applies reads its checkpoints midstep. The cut
        exists only to put a forward's *end* on a rung, and a readable class
        does not need the end — `midstep_positions` hands it the same positions
        to snapshot inside a full-length chunk. Every, not any: one unreadable
        class still needs the forward to land there, and the readable ones lose
        nothing by being handed a position they would have taken anyway.

        This gate and the one in `checkpointers_at` are a pair and change
        together. Suppressing the cut alone leaves `checkpointers_at` refusing
        every off-grid position it is then handed — zero checkpoints kept, no
        error, no warning, and a hit rate that quietly collapses.
        """
        # 【译】`(start, end]` 里最早的阶梯位置，没有则返回0。
        #
        #   prefill chunk就截在这里，让它的forward恰好落在某一级上。它是 `checkpointers_at` 的对应物，后者决定结束在那里的forward
        #   存什么：两者必须逐个位置一致，所以网格算术放在这里，而不放在调度器的调用处。
        #
        #   取最早而不是最晚，因为一条prompt现在可能想要两个位置，而chunk循环只有先被交给第一个，才能走到第二个。在网格内部，
        #   靠后的一级仍然支配靠前的一级——一个类、一个恢复点，取最右的——所以网格在做选择之前先收缩成一个候选。prompt末尾锚点是
        #   迫使这个循环存在的例外：它不支配那一级，那一级也不支配它。见 `_record_checkpoint_end`。
        #
        #   一旦所有适用的类都在midstep读取它们的checkpoint，就返回0。截断只是为了把forward的*终点*放到一级上，而能中途读的类不需要
        #   终点——`midstep_positions` 会把同样的位置交给它，在一个完整长度的chunk里取快照。是"所有"而不是"任一"：只要有一个不能
        #   中途读的类，forward就仍要落在那里，而能中途读的类拿到一个它本来也会取的位置，什么也不损失。
        #
        #   这道门和 `checkpointers_at` 里的那道是一对，要一起改。只去掉这里的截断，会让 `checkpointers_at` 拒掉随后交给它的每一个
        #   不在网格上的位置——存下零个checkpoint，没有错误，没有警告，命中率悄悄崩塌。
        # 解释：调用方Scheduler._finalize_prefill_chunk（record=True）和_preview_prefill_chunk（record=False：只估计，不计数）；
        #   返回target时调用方把chunk改成 target − start。没有任何状态类适用（普通模型）时 all([]) 为True，也返回0。
        #   数字例子见上方checkpoint_limit处【科普】的表（请求A步3截到锚点39936、请求B步1截到demand 29952）
        applicable = [c for c in self.state_caches if c.applies(seq)]
        if all(c.readable_midstep for c in applicable):
            return 0
        rung = 0
        if limit := self.checkpoint_limit(seq):
            # `limit` is itself a multiple of the interval, so a chunk cut at
            # it needs no special case.
            # 【译】`limit` 本身就是间隔的倍数，所以截在它上面的chunk不需要特殊处理。
            # 解释：rung = 不超过end、也不超过limit的最右一级，例：end = 16384 → 16384（8192被它覆盖，不再是候选）
            rung = min(end, limit)
            rung -= rung % self.state_checkpoint_interval_tokens
        # A demand is capped by neither the grid nor `limit`. `limit` is the
        # last position on the *grid* that leaves the widest class its room; a
        # demand carries that room by construction, so it may sit to the right
        # of the last rung — or, on a prompt too short for the grid to place a
        # rung at all, be the only position either side has.
        # 【译】demand既不受网格约束，也不受 `limit` 约束。`limit` 是*网格*上仍给伸得最远的类留出room的最后一个位置；demand按构造
        #   就带着那份room，所以它可以在最后一级的右边——或者，在一个短到网格放不下任何一级的prompt上，成为两边唯一的位置。
        demand = seq.checkpoint_demand_pos
        # The prompt-end anchor sits outside the grid for the same reason and
        # on the same terms — see `_record_checkpoint_end`, which applies the
        # `successor_room` test that makes it keepable.
        # 【译】prompt末尾锚点出于同样的理由、按同样的条件也在网格之外——见 `_record_checkpoint_end`，它做了让锚点能被存下的
        #   `successor_room` 测试。
        anchor = seq.checkpoint_end_pos
        # The earliest candidate strictly inside the chunk. Taking the latest
        # instead is what a single-position ladder could afford and this one
        # cannot: with the anchor at 36 and a rung at 32, `max` cuts at 36 and
        # the forward never ends at 32, so the rung is not merely deferred —
        # it is never kept at all. A class the anchor is out of reach for then
        # loses the rung it used to resume from, permanently, and the demand
        # that fires in its place loses the same comparison on every following
        # request: reuse falls to zero and stays there while the demand counter
        # climbs. Pinned by
        # `test_reuse_another_class_declines_is_not_charged_to_the_ladder`.
        # 【译】chunk内部（严格在起点之后）最早的候选。取最晚的是只有一个位置的阶梯才负担得起的做法，这个阶梯负担不起：锚点在36、
        #   一级在32时，`max` 截在36，forward永远不会结束在32，所以那一级不只是被推迟——它根本不会被存下。对锚点够不着的类来说，
        #   它就永久失去了原本用来恢复的那一级，而顶替它触发的demand在之后的每个请求上都输掉同样的比较：复用跌到零并一直停在那里，
        #   demand计数器却不断上涨。由 `test_reuse_another_class_declines_is_not_charged_to_the_ladder` 钉住。
        candidates = [p for p in (rung, demand, anchor) if p and start < p <= end]
        if not candidates:
            return 0
        target = min(candidates)
        # Beating the grid means an off-grid position chose this cut and the
        # grid would not have. `target < end` is the other half: at `end` the
        # chunk is not shortened and the cut cost nothing, and counting those
        # made the funnel report cuts that never happened — which is the one
        # number meant to expose a shape that pays per request and never
        # converges.
        #
        # Attributed to whichever off-grid position actually chose `target`:
        # the anchor fires on nearly every prompt, so folding its cuts into
        # `chunks_cut_for_demand` would swamp the convergence signal that
        # counter exists to expose. The demand is checked first because when
        # the two coincide it is the demand that evidenced the position.
        # 【译】"胜过网格"意味着是一个网格之外的位置选中了这一刀，而网格本来不会这么切。`target < end` 是另一半条件：在 `end` 处
        #   chunk没被缩短，这一刀不花任何代价，把这种也计进去，曾让漏斗报出从未发生的截断——而这个数字正是用来暴露"每个请求都付
        #   代价、却永不收敛"那种形状的。
        #
        #   归到真正选中 `target` 的那个网格外位置名下：锚点几乎在每条prompt上都会触发，把它的截断并进 `chunks_cut_for_demand`，
        #   会淹没那个计数器要暴露的收敛信号。先检查demand，因为两者重合时，是demand证明了这个位置。
        if record and target != rung and target < end:
            if target == demand:
                self.chunks_cut_for_demand += 1
            else:
                self.chunks_cut_for_end += 1
        return target

    def checkpoint_funnel(self) -> dict[str, int]:
        """Every stage a wanted checkpoint passes through, in order.

        Assembled here because the stages live in two objects — the ladder
        decides what to ask for, the pool decides what survives — and a reader
        needs them side by side to tell which stage lost it.
        Pool-level fates are collected via ``state_checkpoint_fates()`` so that
        a second state class is automatically included — calling
        ``self.state.checkpoint_fates()`` directly would miss it.
        """
        # 【译】一个想要的checkpoint依次经过的每个阶段。
        #
        #   在这里汇总，是因为这些阶段分布在两个对象里——阶梯决定要什么，池子决定什么能留下——读者要把它们并排放着，才能看出是
        #   哪个阶段丢了它。池子一级的结局通过 ``state_checkpoint_fates()`` 收集，这样第二个状态类会被自动包括进来——直接调用
        #   ``self.state.checkpoint_fates()`` 会漏掉它。
        # 解释：读取方是EngineCore的get_cache_statistics工具命令和指标快照（engine_utility.py）
        return {
            "demands_recorded": self.demands_recorded,
            "demands_declined_no_room": self.demands_declined_no_room,
            "chunks_cut_for_demand": self.chunks_cut_for_demand,
            "chunks_cut_for_end": self.chunks_cut_for_end,
            "midstep_positions_unreadable": self.midstep_positions_unreadable,
            "joint_boundaries": self.joint_boundaries,
            "state_hbm": self.state_hbm_boundaries,
            "state_tier": self.state_tier_boundaries,
            "state_gate_lost_boundary": self.state_gate_lost_boundary,
            "orphan_load_slots_reclaimed": self._orphan_load_slots_reclaimed,
        } | self.state_checkpoint_fates()

    def pool_pressure(self) -> dict[str, int]:
        """Both pools' eviction counts and occupancy, side by side.

        The pair is the point. A hit rate says reuse was lost but not which
        pool lost it, and the two are sized against each other out of one
        budget (`plan_pools`) — so the actionable reading is always a
        comparison: paged evicting while state sits mostly vacant means the
        split is wrong, both evicting means the budget is.

        The fates come from `state_checkpoint_fates()`, the aggregator, not
        from `self.state` and not from a single `_state_checkpoint_cache`.
        Under PAGE `self.state` is built with `StateTransfer.none()` and never
        sees a `checkpoint()`, so reading it here printed four zeros for the
        life of the server; and reading `_state_checkpoint_cache` alone omits
        any second state class the aggregator folds in. Either way the number
        disagreed with the `checkpoint_funnel` line -- which already goes
        through the aggregator -- and two outputs with the same metric names
        disagreeing is how a tuning session concludes checkpointing never
        fires. Occupancy stays on `self.state`: that is the slot pool either
        way, and the coordinator has none.
        """
        # 【译】两个池子的淘汰计数和占用，并排列出。
        #
        #   重点就在"成对"。命中率只说明复用丢了，却不说是哪个池子丢的，而两个池子是从同一份预算里互相对照着分出来的
        #   （`plan_pools`）——所以可操作的读法总是一次对比：分页池在淘汰而状态池大半空着，说明划分不对；两个都在淘汰，说明总预算不够。
        #
        #   结局来自聚合器 `state_checkpoint_fates()`，而不是 `self.state`，也不是单个 `_state_checkpoint_cache`。在PAGE下 `self.state`
        #   是用 `StateTransfer.none()` 建的，从来收不到 `checkpoint()`，所以在这里读它，会在整个服务器生命周期里打印四个0；
        #   只读 `_state_checkpoint_cache` 又会漏掉聚合器纳入的任何第二个状态类。无论哪种，这个数都会和 `checkpoint_funnel`
        #   那一行对不上——那一行已经走聚合器了——而两个指标名相同的输出互相矛盾，正是一次调优得出"checkpoint从来没触发过"这种
        #   结论的原因。占用仍取自 `self.state`：无论哪种情况它都是那个槽位池，协调器没有槽位。
        # 解释：读取方是EngineStats的周期性日志（Scheduler构造EngineStats时把它作为pool_pressure回调传入）
        # `state_checkpoint_fates()`, matching `checkpoint_funnel`: it sums
        # every state class, so a build with a second class is not silently
        # undercounted here while the funnel line reports the full total.
        # `occupancy()` stays on the pool, which is the thing that has slots.
        # 【译】`state_checkpoint_fates()`，与 `checkpoint_funnel` 一致：它把每个状态类加总，所以有第二个类的构建不会在这里被悄悄少算，
        #   而漏斗那一行报的是全部总数。`occupancy()` 仍取自池子，有槽位的是它。
        return (
            self.kv.eviction_stats()
            | self.state.occupancy()
            | self.state_checkpoint_fates()
        )

    # 【分支】forward结束在pos时，哪些状态类在这里存checkpoint（数字例子见上方checkpoint_limit处【科普】的表）
    # 调用方：hash_blocks末尾（prefill：aimed=True，next_forward_tokens默认 = prompt剩余长度；decode经hash_decode_blocks：
    #   aimed=False，next_forward_tokens = Scheduler._checkpoint_room = 0或1）
    # 判定顺序：
    #   interval == 0 或 pos ≤ 0 → 不存
    #   aimed（prefill，chunk已被checkpoint_cut截到候选上）：pos必须在网格上，或等于这条seq的demand / 锚点
    #   unaimed（decode）：interval = −1 → 不存；否则pos要在hash block边界上，且 pos − last_checkpoint_pos ≥ interval
    #   最后逐个状态类筛：不是"prefill + 能中途读"（那种由midstep负责）、适用于这条seq、且 next_forward_tokens ≥ successor_room
    def checkpointers_at(
        self,
        seq: Sequence,
        pos: int,
        next_forward_tokens: int | None = None,
        aimed: bool = True,
    ) -> list[StateCache]:
        """State classes that should keep a checkpoint at `pos`, in class order.

        A ladder of resume points, one every `state_checkpoint_interval_tokens`
        of context, shared by every class. Keeping one is capacity-neutral for a
        rolling class (the slot handed away is replaced from the free list) and
        capacity-bounded for an immutable one (an LRU-capped pin), but either
        way it costs the *keeper* an extra forward — its prompt gets cut at the
        rung — so the interval is what keeps that cost amortized instead of
        per-request.

        `next_forward_tokens` is how many tokens the forward right after this
        one carries, and is what each class's `successor_room` is compared
        against. Unset means the prompt's remainder, the prefill answer; decode
        passes one. Everything else follows from that one number — a class
        needing a long hand-over (V4's ring, 131) simply never qualifies
        mid-generation, one that hands nothing over (a retaining SWA pool, 0)
        always does, one that cannot keep a checkpoint at all (`inf`) never
        does, and a request stopping on this step passes 0 and keeps nothing
        that nothing will ever resume from.

        The position must be exact. A checkpoint holds state as of the forward's
        last token, so a forward that overshoots a rung is ahead of the hash it
        would be filed under; the scheduler cuts prefill chunks to land here,
        and a path that doesn't simply keeps nothing.

        On top of the grid sit two rungs of this seq's own.
        `checkpoint_demand_pos` is a boundary this seq was denied for want of a
        checkpoint (`_record_checkpoint_demand`); `checkpoint_end_pos` is this
        prompt's own end, which is the position agentic traffic actually resumes
        at (see `_record_checkpoint_end`). Both
        are admitted on the same terms: off the grid, and still subject to the
        `successor_room` test below — which `_record_checkpoint_end` pre-checks,
        so the cut and the keep cannot disagree. `checkpoint_cut` reads the same
        two fields, so the cut and the keep cannot drift apart.

        `aimed` says whether the caller could place the forward's end. Prefill
        can — `checkpoint_cut` shortens the chunk — so it is held to the exact
        grid and the two agree position for position. A speculative decode step
        cannot: it commits `1 + accepted`, so it steps over most rungs, and
        holding it to the grid made a decode checkpoint a one-in-`toks/fwd`
        chance. Bounding the drafts would land it there, at the price of
        throwing away speculation and under-reporting the acceptance rate,
        which counts drafts offered as `mtp_k` regardless.

        Nothing needs that price. The grid exists to space checkpoints out, and
        a step that lands on any hash-block boundary far enough past the last
        one serves the purpose exactly as well — the position only has to be
        *findable*, and a resumer finds it by hash, never by arithmetic. So an
        unaimed caller is held to the spacing rather than the grid.

        What that buys scales with how many boundaries a rung spans. A step
        lands on any given boundary with probability `1 / toks_per_forward`, so
        the chance of keeping a checkpoint per rung is
        `1 - (1 - 1/toks_per_forward) ** (interval / hash_block_size)`: at
        DeepSeek-V4's 256-token block and 4.3 tokens a forward, 23% when the
        interval is one block and effectively certain at the 8192 default. The
        two rules coincide exactly when the interval *is* the block, which is
        also the finest grid V4 admits — so a test at that setting measures
        nothing, and `demand_config` exists to avoid it.

        The demand rung is absent from that branch by construction, not by
        omission: a demand is at most the prompt's own hit ceiling, and every
        unaimed position is at or past the end of the prompt, so generation
        cannot reach one. `checkpoint_end_pos` is absent for a near-identical
        reason: it is the prompt's end stepped back to the grid, and an unaimed
        position at or past the prompt end is past it too.
        """
        # 【译】应该在 `pos` 处保留一个checkpoint的状态类，按类的顺序。
        #
        #   一架恢复点的阶梯，每 `state_checkpoint_interval_tokens` 个token的上下文一级，所有类共用。存一份对rolling类是容量中性的
        #   （交出去的槽位由空闲列表补上），对不可变类是受容量约束的（一个有LRU上限的pin），但无论哪种，都要*存的一方*多付一次
        #   forward——它的prompt会在这一级被截开——所以正是间隔让这个代价被摊薄，而不是每个请求都付。
        #
        #   `next_forward_tokens` 是紧接这次之后的forward带多少token，拿来和每个类的 `successor_room` 比较。不传表示prompt的剩余部分，
        #   即prefill的答案；decode传1。其余一切都由这一个数推出——需要长交接的类（V4的环形缓冲，131）在生成过程中永远不合格，
        #   什么都不交接的类（保留式的SWA池，0）总是合格，根本不能存checkpoint的类（`inf`）永远不合格，而在这一步停止的请求传0，
        #   不保留任何永远不会有人从那里恢复的东西。
        #
        #   位置必须精确。checkpoint保存的是forward最后一个token时的状态，所以越过某一级的forward已经跑到了它要归档的那个hash前面；
        #   调度器把prefill chunk截到这里，没做到这一点的路径就什么都不存。
        #
        #   在网格之上还有这条seq自己的两级。`checkpoint_demand_pos` 是这条seq因缺checkpoint而被拒绝的一个边界
        #   （`_record_checkpoint_demand`）；`checkpoint_end_pos` 是本prompt自己的末尾，也就是agentic流量实际恢复的位置
        #   （见 `_record_checkpoint_end`）。两者按同样的条件准入：不在网格上，仍要通过下面的 `successor_room` 测试——
        #   `_record_checkpoint_end` 事先检查过它，所以切和存不会打架。`checkpoint_cut` 读的是同样两个字段，所以两者不会漂移开。
        #
        #   `aimed` 表示调用方能否决定forward在哪里结束。prefill能——`checkpoint_cut` 会截短chunk——所以它被要求精确落在网格上，
        #   两者逐个位置一致。投机decode步不能：它提交 `1 + accepted` 个token，会跨过大多数级，要求它落在网格上曾让decode
        #   checkpoint成了"每 `toks/fwd` 次才中一次"的事。限制草稿数可以让它落上去，代价是丢掉投机、并低报接受率——
        #   接受率不管怎样都按提供了 `mtp_k` 个草稿来算。
        #
        #   没有什么需要付这个代价。网格存在是为了把checkpoint隔开，而落在任何一个离上一个足够远的hash block边界上的一步，同样能
        #   达到目的——位置只需要能被*找到*，而恢复的一方靠hash找它，从不靠算术。所以unaimed的调用方只受间距约束，不受网格约束。
        #
        #   这能换来多少，取决于一级跨了多少个边界。一步落在某个给定边界上的概率是 `1 / toks_per_forward`，所以每一级能存下checkpoint
        #   的概率是 `1 - (1 - 1/toks_per_forward) ** (interval / hash_block_size)`：在DeepSeek-V4的256-token block、每次forward
        #   4.3个token时，间隔为一个block时是23%，在默认的8192时实际上是必然。间隔*就是*block时两条规则完全重合，这也是V4允许的
        #   最细网格——所以在那个设置下的测试什么都没测到，`demand_config` 就是为了避开它而存在。
        #
        #   demand那一级在这个分支里缺席是按构造，而不是遗漏：demand至多是prompt自己的命中上限，而每个unaimed位置都在prompt末尾
        #   或之后，所以生成永远到不了它。`checkpoint_end_pos` 缺席的理由几乎一样：它是退到网格上的prompt末尾，而在prompt末尾
        #   或之后的unaimed位置也在它之后。
        # 观察："V4的环形缓冲，131"是旧情况：现在DeepSeek-V4声明的是StateTransfer.copy（deepseek_v4_attn.py），走PAGE checkpoint，
        #   PagedStateCheckpointCoordinator.successor_room = 0（page_unit_checkpoint.py）；fork类的successor_room = fork_tokens
        #   （GDN为1），replay / none类为inf
        interval = self.state_checkpoint_interval_tokens
        if interval == 0 or pos <= 0:
            return []
        # -1: no grid, so `pos % interval` has nothing to say and asking it
        # would be a ZeroDivisionError's less honest cousin — -1 divides
        # everything, admitting every position as a rung. The demand and the
        # anchor are then the only two placements, which is the whole point.
        # 【译】-1：没有网格，所以 `pos % interval` 说明不了什么，硬要问它就是ZeroDivisionError那位不那么诚实的表亲——-1能整除一切，
        #   会把每个位置都当成一级。此时demand和锚点是仅有的两种放置，这正是-1的意义所在。
        on_grid = interval > 0 and not pos % interval
        if aimed:
            if (
                not on_grid
                and pos != seq.checkpoint_demand_pos
                and pos != seq.checkpoint_end_pos
            ):
                return []
        elif interval < 0:
            # Decode-side spacing has no grid to fall back on either. The two
            # aimed placements are both prompt positions, so nothing here is
            # reachable and generation keeps no checkpoints under -1.
            # 【译】decode侧的间距规则也没有网格可依。两种aimed放置都是prompt里的位置，所以这里什么都够不到，-1下生成阶段不存checkpoint。
            return []
        # unaimed（decode）：pos要在hash block边界上（才有hash可归档），且离上一个checkpoint至少一个interval。
        #   例（hbs = 256、interval 8192）：prompt最后一个checkpoint在39936 → 下一个在48128
        elif pos % self.hash_block_size or pos - seq.last_checkpoint_pos < interval:
            return []
        # 不传next_forward_tokens（prefill）：后继forward最多能带上prompt剩下的这么多token
        if next_forward_tokens is None:
            next_forward_tokens = seq.num_prompt_tokens - pos
        return [
            c
            for c in self.state_caches
            # A readable class checkpoints its prompt through
            # `plan_midstep`/`commit_midstep` instead, so keeping it here as
            # well would keep the same boundary twice. Both halves of that are
            # damage: two slots spent on one hash, the loser orphaned free but
            # unindexed — and, under `fork`, the seq hands its live slot away
            # and takes a fresh one, binding the next forward to refill a
            # replacement it had no reason to need.
            #
            # Only on the aimed path. Midstep positions are all prompt
            # positions, so generation is where `aimed=False` puts every class
            # back on this one. The twin of the `checkpoint_cut` gate; see
            # there for why the two cannot move separately.
            # 【译】能中途读的类通过 `plan_midstep`/`commit_midstep` 给它的prompt做checkpoint，在这里再存一次，同一个边界就被存了两次。
            #   两半都是损害：两个槽位花在一个hash上，输家空着、却没被索引，成了孤儿——而在 `fork` 下，seq会把它的活槽位交出去、
            #   换一个新的，迫使下一次forward去重新填一个它本不需要的替代品。
            #
            #   只在aimed路径上这样。midstep位置全都是prompt里的位置，所以在生成阶段 `aimed=False` 让每个类都回到这条路上。
            #   这是 `checkpoint_cut` 那道门的孪生；两者为什么不能分开动，见那里。
            if not (aimed and c.readable_midstep)
            and c.applies(seq)
            and next_forward_tokens >= c.successor_room
        ]

    # ════════ 【分支】外部来源的KV登记进HBM前缀缓存：offload加载回来的（publish_loaded_prefix）、
    #   PD传过来的（register_received_prefix）——都见第9批 ════════
    # 前置：这两条路径上，KV不是本机prefill forward算出来的，而是被直接写进已分配好的物理block：
    #   KV offload（第9批kv_transfer/offload）：GPU里早被淘汰的前缀，从CPU内存 / LMCache读回显存
    #     （概念见arg_utils.py _KV_OFFLOAD_MODES上方的【科普】KV offload）
    #   PD分离（第9批kv_transfer/disaggregation）：decode节点不做prefill，prompt的KV经RDMA从prefill节点写进来
    #     （概念见arg_utils.py EngineArgs.kv_transfer_config上方的【科普】PD分离）
    #   它们都不经过hash_blocks。不补登记的话，本请求能用这些KV，以后的请求却查不到，还会再从CPU / 远端读一遍
    #   对比nano-vllm：nano-vllm没有这两种外部来源
    def publish_loaded_prefix(
        self,
        seq: Sequence,
        start_token: int,
        end_token: int,
    ) -> int:
        """Publish a successfully loaded offload prefix into the GPU cache index.

        LMCache restores KV directly into already allocated physical blocks, so
        those blocks do not pass through ``hash_blocks()``. Without explicitly
        publishing them here, the current request can consume the restored KV,
        but later requests cannot discover it through ``can_allocate()`` and
        repeatedly load the same prefix from CPU.

        Only complete, hash-block-aligned loaded blocks are published. Existing
        canonical mappings win: concurrent requests may load the same prefix
        into different physical blocks, and replacing the canonical mapping
        would make its eventual eviction remove the wrong cache entry.
        """
        # 【译】把一段成功加载回来的offload前缀发布进GPU缓存索引。
        #
        #   LMCache把KV直接恢复进已经分配好的物理block，所以这些block不会经过 ``hash_blocks()``。不在这里显式发布的话，当前请求
        #   能用上恢复的KV，但以后的请求无法通过 ``can_allocate()`` 发现它，会一再从CPU加载同一段前缀。
        #
        #   只发布完整的、按hash block对齐的已加载block。已有的规范映射胜出：并发的请求可能把同一段前缀加载进不同的物理block，
        #   替换规范映射会让它日后被淘汰时删掉错误的缓存条目。
        # 解释：调用方Scheduler._mark_offload_load_ready（offload加载完成时）。例（hbs = 16）：prompt 20000个token，准入时HBM
        #   命中4096，LMCache加载回[4096, 12288) → 登记块256~767共512块，父hash取块255上的；返回8192（登记的token数）。
        #   规则和hash_blocks相反：hash_blocks里后发布的block接管索引；这里已有同hash的block时只给本block打标注（block.update），
        #   不动索引——两份内容相同的物理block，索引只指向原来那一份
        if not self.enable_prefix_caching:
            return 0

        start_token = max(0, int(start_token))
        end_token = min(int(end_token), int(seq.num_prompt_tokens))
        if end_token <= start_token:
            return 0
        # 起点必须对齐到hash block（否则算不出链上的hash）；终点向下取整，尾部半满的block不发布（同hash_blocks）
        hbs = self._hash_block_size()
        if start_token % hbs != 0:
            logger.warning(
                "Cannot publish offload prefix with unaligned start: "
                "seq=%s start=%d hash_block_size=%d",
                seq.id,
                start_token,
                hbs,
            )
            return 0

        start_block = start_token // hbs
        end_block = end_token // hbs
        if end_block <= start_block:
            return 0
        if end_block > len(seq.block_table):
            logger.warning(
                "Cannot publish offload prefix beyond block table: "
                "seq=%s end_block=%d blocks=%d",
                seq.id,
                end_block,
                len(seq.block_table),
            )
            return 0

        parent_hash = self._chain_parent_hash(seq, start_block)
        if parent_hash is None:
            return 0

        indexed_tokens = 0
        for i in range(start_block, end_block):
            token_ids = self._hash_block_tokens(seq, i)
            block_id = seq.block_table[i]
            block = self.kv.block(block_id)
            block_hash = self.compute_hash(token_ids, parent_hash)
            canonical_id = self.kv.lookup(block_hash)

            # 这个block已挂着另一个hash：不覆盖、停下（发布过的内容不会变）；索引里同hash的block内容不同 = hash碰撞，也停下
            if block.hash not in (-1, block_hash):
                logger.warning(
                    "Refusing to overwrite indexed block during offload "
                    "promotion: seq=%s block=%d",
                    seq.id,
                    block_id,
                )
                break

            if canonical_id != -1:
                canonical = self.kv.block(canonical_id)
                if canonical.token_ids != token_ids:
                    logger.warning(
                        "Hash collision while publishing offload prefix: "
                        "seq=%s block=%d canonical=%d",
                        seq.id,
                        block_id,
                        canonical_id,
                    )
                    break
                # Keep the canonical index entry, but annotate this request's
                # duplicate physical block as well: `hash_blocks()` needs the
                # final loaded block's hash as the parent when it publishes the
                # newly computed suffix. Annotating without indexing is what
                # separates `Block.update` from `BlockPool.publish` here, and it
                # is safe because `_unindex` only drops an entry that still
                # points at the block being reused.
                # 【译】保留规范的索引条目，但也给本请求这份重复的物理block打上标注：`hash_blocks()` 发布新算出的后缀时，需要最后一个
                #   加载block的hash作为父hash。"只标注、不索引"正是这里 `Block.update` 和 `BlockPool.publish` 的区别，而这样做是安全的，
                #   因为 `_unindex` 只删除仍指向被复用block的条目。
                block.update(block_hash, token_ids)
            else:
                self.kv.publish(block_id, block_hash, token_ids)
                if self._event_log is not None:
                    self._event_log.append(
                        _make_block_stored(
                            [block_hash],
                            # The only BlockStored site fed straight from
                            # `_hash_block_tokens`; the other two accumulate
                            # into a list already. See `_make_block_stored`.
                            # 【译】唯一一处直接由 `_hash_block_tokens` 喂数据的BlockStored；另外两处本来就累加成list了。
                            #   见 `_make_block_stored`。
                            list(token_ids),
                            parent_hash if parent_hash != -1 else None,
                            hbs,
                            token_offset=i * hbs,
                        )
                    )

            indexed_tokens += hbs
            parent_hash = block_hash

        return indexed_tokens

    def register_received_prefix(self, seq: Sequence) -> int:
        """Hash received prompt blocks into the prefix cache so subsequent
        turns can match them locally and transfer only the delta.

        Under DCP, one block-table entry represents ``dcp_world_size`` physical
        blocks and therefore ``hash_block_size`` global tokens. Hashing at the
        physical ``block_size`` would attach several incompatible token ranges
        to the same virtual block.

        Blocks before ``num_cached_tokens`` are already indexed locally; only
        the received suffix needs registration. The trailing partial hash block
        remains unpublished, matching ``hash_blocks``.

        Returns the number of complete received suffix blocks processed for
        this sequence. This includes blocks annotated from an existing canonical
        hash, so it is neither the total number of hashed prompt blocks nor
        necessarily the number of newly inserted hash-index entries.
        """
        # 【译】把收到的prompt block登记进前缀缓存，让后续几轮能在本地匹配上它们、只传增量。
        #
        #   DCP下，一个block-table条目代表 ``dcp_world_size`` 个物理block，因此是 ``hash_block_size`` 个全局token。按物理的
        #   ``block_size`` 来hash，会把好几段互不相容的token范围挂到同一个虚拟block上。
        #
        #   ``num_cached_tokens`` 之前的block已经在本地登记过；只有收到的后缀需要登记。末尾不满的hash block保持不发布，与
        #   ``hash_blocks`` 一致。
        #
        #   返回这条序列被处理的、完整的已接收后缀block数。其中包括按已有规范hash打标注的block，所以它既不是已登记的prompt block
        #   总数，也不一定是新插入的hash索引条目数。
        # 解释：调用方Scheduler._update_waiting_for_remote_kv（PD consumer收完KV、不是offload路径时）。
        #   例（hbs = 16）：prompt 1000个token，本地命中320（num_cached_tokens = 320）→ num_full = 1000//16 = 62，start = 20
        #     → 登记块20~61，返回42；[992, 1000)半满不登记。和publish_loaded_prefix的区别：hash碰撞时这里直接抛RuntimeError
        #     （那边记日志后停下）；最后把prefix_hashes_published置True，postprocess就不会再登记一遍
        if not self.enable_prefix_caching:
            return 0

        hbs = self._hash_block_size()
        num_full = seq.num_prompt_tokens // hbs
        num_full = min(num_full, len(seq.block_table))
        start = min(seq.num_cached_tokens // hbs, num_full)
        h = self._chain_parent_hash(seq, start)
        if h is None:
            return 0

        for i in range(start, num_full):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block_id = seq.block_table[i]
            block = self.kv.block(block_id)
            indexed_block_id = self.kv.lookup(h)
            if indexed_block_id == -1:
                self.kv.publish(block_id, h, token_ids)
            else:
                indexed_block = self.kv.block(indexed_block_id)
                if indexed_block.token_ids != token_ids:
                    raise RuntimeError(
                        "Hash collision while registering received prefix: "
                        f"seq={seq.id} block={block_id} indexed={indexed_block_id}"
                    )
                block.update(h, token_ids)

        seq.num_hashed_tokens = max(seq.num_hashed_tokens, num_full * hbs)
        # The decode consumer has no local prefill postprocess to publish these
        # prompt blocks. Mark that one-shot work complete so its first decode
        # output does not publish the same physical blocks again.
        # 【译】decode consumer没有本地prefill的postprocess来发布这些prompt block。把这项一次性的工作标为已完成，这样它第一次decode
        #   输出时就不会再发布一遍同样的物理block。
        seq.prefix_hashes_published = True
        return num_full - start

    # ════════ 【分支】KV offload的"租约"：后台保存还在读的block先别回收（第9批kv_transfer/offload） ════════
    # 前置：offload把算好的KV异步存到CPU / LMCache。请求结束时，后台保存可能还没读完它的block；这时把block放回空闲池，下一次
    #   _fresh_block就可能把它分给别人、覆盖掉还没存走的KV。原做法是把整条请求挂进deferred_free_blocks，等保存完成再deallocate；
    #   租约只扣住保存真正要读的那几个block：
    #     deallocate_partial：对受保护的block各claim一次（ref + 1，这份引用归"保存"所有），再正常deallocate整条请求
    #     free_leased_blocks：保存完成（或卡住超时被回收）后，把这份引用还掉
    #   例：block_table = [7, 3, 12, 20]，保存要读 {7, 3}
    #     claim 7、3 → ref 2；deallocate → 20、12归零回池（立刻能给别人），7、3降到1（租约还拿着）
    #     保存完成 → free_leased_blocks([7, 3]) → 归零回池（hash还在，进_cached，仍可被命中）
    #   acquire_offload_prefix是"晚保存"：请求结束、block_table没了之后才开始保存，按内容hash把仍驻留的前缀block找回来再claim
    def deallocate_partial(
        self,
        seq: Sequence,
        protected_block_ids: frozenset[int],
        *,
        per_request_state_safe: bool = False,
    ) -> None:
        """Deallocate `seq` now, except block IDs a pending offload save reads.

        Used by the offload early-block-release path (see
        `DenseOffloadScheduler.protected_block_ids`) instead of deferring the
        whole request behind `deferred_free_blocks`: everything in
        `seq.block_table` that is *not* in `protected_block_ids` -- decode
        blocks, an unaligned prompt tail, already-saved ranges, or any other
        block the in-flight save never reads -- is returned to `BlockPool`
        immediately, same as `deallocate`. `protected_block_ids` must already
        be frozen by the caller (from the connector's exact block-range for the
        in-flight `SaveOperationId`) before this clears `seq.block_table`.

        Each protected block is first ``claim``ed for the save lease, then the
        request is deallocated normally. This makes ownership explicit: shared
        prefix blocks retain their other owners, while the save owns exactly
        one refcount share until ``free_leased_blocks`` releases it.

        Per-request recurrent state is fail-closed: its active slots may be
        released here only when the connector explicitly confirms that every
        state source needed after request teardown has an independent lease.
        """
        # 【译】现在就释放 `seq`，但保留一次待执行的offload保存要读的block ID。
        #
        #   由offload的"提前释放block"路径使用（见 `DenseOffloadScheduler.protected_block_ids`），取代把整条请求挂在
        #   `deferred_free_blocks` 后面：`seq.block_table` 里所有*不在* `protected_block_ids` 中的block——decode block、没对齐的
        #   prompt尾部、已经保存完的区段，或任何在途保存从不读取的block——都立刻还给 `BlockPool`，和 `deallocate` 一样。
        #   `protected_block_ids` 必须在本函数清空 `seq.block_table` 之前，由调用方（根据connector给出的、在途 `SaveOperationId`
        #   的精确block范围）冻结好。
        #
        #   每个受保护的block先为保存租约 ``claim`` 一次，然后照常释放这条请求。这让所有权变得明确：共享的前缀block保留它们的其他
        #   主人，而保存恰好拥有一份引用计数，直到 ``free_leased_blocks`` 释放它。
        #
        #   每请求的递归状态采取"失败即关闭"：只有connector明确确认请求拆除之后还需要的每个状态源都有独立的租约时，
        #   才可以在这里释放它的活跃槽位。
        # 解释：调用方Scheduler.postprocess的请求结束路径（connector要求推迟释放、又能给出精确的受保护block集合时）
        if seq.has_per_req_cache and not per_request_state_safe:
            raise RuntimeError(
                "partial PAGE deallocation is unsupported for per-request "
                "state without an explicit connector safety capability"
            )
        table = set(seq.block_table)
        unknown = set(protected_block_ids) - table
        if unknown:
            raise ValueError(
                f"cannot lease blocks not owned by seq {seq.id}: {sorted(unknown)}"
            )
        for block_id in protected_block_ids:
            self.kv.claim(block_id)
        self.deallocate(seq)

    def acquire_offload_prefix(
        self,
        seq: Sequence,
        start_tokens: int,
        end_tokens: int,
    ) -> tuple[list[int], int, tuple[int, ...]]:
        """Claim the still-resident contiguous prefix used by a late save.

        A finished request no longer owns its old physical block table. Each
        block is therefore resolved again through the content hash index and
        token-checked before it is claimed. The scan stops at the first gap so
        the returned range can never describe a cache object with a hole.
        """
        # 【译】认领一次"晚保存"要用的、仍驻留在显存里的连续前缀。
        #
        #   已结束的请求不再拥有它原来的物理block table。所以每个block都要通过内容hash索引重新解析一遍，核对token之后再认领。
        #   扫描在第一个缺口处停下，这样返回的范围永远不会描述一个中间有洞的缓存对象。
        # 解释：调用方kv_transfer/offload/chunked_scheduler.py的_late_save_source。从块0开始按cache_seed重算整条链（block_table
        #   已没了，只能靠内容找），[start, end)里逐块lookup + 核对 + claim。返回 (每块的block id，没找到的为-1；连续可用的
        #   终点token；已claim的id元组)，调用方用不上的部分再用free_leased_blocks还掉
        hbs = self._hash_block_size()
        if (
            start_tokens < 0
            or end_tokens < start_tokens
            or start_tokens % hbs
            or end_tokens % hbs
        ):
            raise ValueError(
                f"offload source range [{start_tokens}, {end_tokens}) must "
                f"align to hash block size {hbs}"
            )
        end_block = end_tokens // hbs
        start_block = start_tokens // hbs
        block_ids = [-1] * end_block
        claimed: list[int] = []
        parent_hash = seq.cache_seed
        available_end = start_tokens
        for index in range(end_block):
            token_ids = self._hash_block_tokens(seq, index)
            block_hash = self.compute_hash(token_ids, parent_hash)
            parent_hash = block_hash
            if index < start_block:
                continue
            block_id = self.kv.lookup(block_hash)
            if block_id < 0:
                break
            block = self.kv.block(block_id)
            if block.hash != block_hash or block.token_ids != token_ids:
                break
            self.kv.claim(block_id)
            claimed.append(block_id)
            block_ids[index] = block_id
            available_end = (index + 1) * hbs
        # The caller may have to trim a partial LMCache chunk from the tail.
        # Preserve token order until that trim is complete; a set is suitable
        # only after the exact ordered prefix has been selected.
        # 【译】调用方可能要从尾部裁掉一个不完整的LMCache chunk。在裁剪完成之前保持token顺序；只有选定了精确的有序前缀之后，
        #   才适合用set。
        return block_ids, available_end, tuple(claimed)

    def free_leased_blocks(self, block_ids) -> None:
        """Return block IDs a `deallocate_partial` lease held back, to the pool.

        Called once the offload connector reports the exact save generation
        that read `block_ids` as source-safe (`take_source_safe_releases`) or
        reclaims it after a stall (`reclaim_stale_leases`). Idempotent only in
        the sense that the connector guarantees each block ID is surfaced by
        exactly one of `take_source_safe_releases` / `reclaim_stale_leases` for
        one lease -- calling this twice for the same ID double-frees the block,
        same as calling `BlockPool.free` twice would.
        """
        # 【译】把被 `deallocate_partial` 租约扣住的block ID还给池子。
        #
        #   在offload connector报告读取 `block_ids` 的那一代保存已经"源安全"（`take_source_safe_releases`）、或在卡住之后回收它
        #   （`reclaim_stale_leases`）时调用。"幂等"只在这个意义上成立：connector保证对一份租约，每个block ID只会经由
        #   `take_source_safe_releases` / `reclaim_stale_leases` 两者之一出现恰好一次——对同一个ID调用两次会重复释放这个block，
        #   和调用两次 `BlockPool.free` 一样。
        # 解释：kv.free = 引用计数 − 1，归零才回空闲池（见下方deallocate的图）。调用方Scheduler（排空租约释放、超时回收）和
        #   chunked_scheduler.py（晚保存用不上的block）
        for block_id in block_ids:
            self.kv.free(block_id)

    # ═══════════════ deallocate：请求结束或被抢占，归还它持有的全部资源（A级） ═══════════════
    # 调用方（grep核对，scheduler.py）：postprocess（请求结束；connector还在读这些block时改走deferred_free_blocks，之后由
    #   _maybe_release_deferred调用，或走deallocate_partial）、preempt（重算式抢占，之后整条重新prefill）、_schedule里
    #   准入之后回滚的几条路径、_consume_failed_remote_kv、allocate_waiting
    #
    # 核心是"引用计数 − 1，归零才回空闲池，hash和KV内容原样保留"（惰性淘汰）。例：hbs = 16，
    #   A.block_table = [7, 3, 12]：块0、1已登记（7 ↔ h0、3 ↔ h1），块2是半满的尾块（hash = −1）；block 7还被B共享（ref = 2）
    #   逆序释放 12 → 3 → 7（kv.free，block_pool.py）：
    #     block 12：ref 1→0，没有hash → 进 _vacant（按id排序的最小堆："空白"block，分配时最先被取走）
    #     block 3 ：ref 1→0，有hash   → 进 _cached 队尾（按释放先后排序的LRU）；索引 h1 → 3 不动，显存里的KV也不动
    #     block 7 ：ref 2→1，B还在用   → 什么都不变
    #
    #     空闲池   _vacant（最小堆）   { …, 12, … }            先取：拿走它什么都不损失
    #             _cached（LRU）      [ 最早释放 … , 3 ]       _vacant取空之后，才从最早释放的一端取
    #             _protected（SLRU）  只在ATOM_PREFIX_CACHE_POLICY=slru时用：被命中复用过的block放这里，最后才取
    #     hash索引 { h0:7, h1:3, … }    ← 不变：block 3虽然空闲，仍可被命中
    #
    #   之后block 3的两种命运：
    #     ① 新请求C的前缀命中h1 → kv.claim(3)：从_cached摘下、ref = 1，KV原样复用，不用重算
    #     ② 一直没人命中，某次_fresh_block（allocate未命中的部分 / may_append）把_vacant取空后轮到它 → kv.allocate(3)：
    #        _unindex删掉 h1 → 3，回调_record_evicted（计数、发BlockRemoved事件、作废挂在h1名下的状态checkpoint），
    #        block.reset()——这时h1才真正消失。另外几处也会抹掉hash：clear_cache（清空整个索引）、池子缩小时的retire_top
    #   为什么逆序释放：链是从前往后匹配的，块0没了块1就再也走不到；逆序释放让靠前的块后进LRU、最后被淘汰（同nano-vllm）
    # 对比nano-vllm：nano-vllm只有ref_count − 1和一个free deque（释放放队尾、分配从队头取），所以一个有hash的block可能
    #   在空白block之前就被拿走；ATOM把空闲block分成空白 / 可复用两半，空白的先用。函数后半部分重置的是ATOM多出来的每请求状态：
    #   checkpoint的demand / 锚点 / midstep预留、线性注意力的状态槽位（含正在加载中的槽位）、offload的联合边界
    def deallocate(self, seq: Sequence):
        # 逆序：链尾先进LRU、先被淘汰；kv.free = ref − 1，归零才回空闲池（hash和内容保留，见上图）
        for block_id in reversed(seq.block_table):
            self.kv.free(block_id)
        seq.num_cached_tokens = 0
        # The block table is gone, so nothing of this seq is hashed any more.
        # Covers preemption too, which frees through here and re-prefills.
        # 【译】block table没了，这条seq就不再有任何已登记hash的部分。抢占也包括在内：它通过这里释放，然后重新prefill。
        # 解释：block本身的hash仍留在池子里（上图），清零的只是这条seq的水位线；重新准入时can_allocate会重新命中它们
        seq.num_hashed_tokens = 0
        # 有界重放的记录只描述这一次准入（sequence.py Replay类）
        seq.replay = None
        seq.replays_hit = seq.replay_kv_load = False
        # Likewise the demand: it describes one admission against one cache
        # state, and a re-admitted seq gets a fresh answer from `can_allocate`
        # — including a fresh place in both funnel counters.
        # 【译】demand也一样：它描述的是某一次准入面对某一个缓存状态，重新准入的seq会从 `can_allocate` 得到新的答案——
        #   包括在两个漏斗计数器里的新位置。
        seq.checkpoint_demand_pos = 0
        seq.checkpoint_demand_counted = False
        seq.checkpoint_demand_declined = False
        # The anchor is derived from `num_prompt_tokens` alone, so it would
        # still be correct on re-admission — but `_record_checkpoint_end` runs
        # unconditionally in `can_allocate`, and leaving a stale value here
        # would let a seq that never reaches that path keep a position nothing
        # re-validated against the current `state_caches`.
        # 【译】锚点只由 `num_prompt_tokens` 推出，所以重新准入时它仍然正确——但 `_record_checkpoint_end` 在 `can_allocate` 里无条件运行，
        #   这里留下旧值的话，一条从未走到那条路径的seq就会保留一个没人针对当前 `state_caches` 重新验证过的位置。
        seq.checkpoint_end_pos = 0
        seq.last_checkpoint_pos = 0
        # An uncommitted checkpoint describes state in a slot that is about to
        # go back on the free list, so the intent dies with it.
        # 【译】一个还没提交的checkpoint描述的是一个马上要回到空闲列表的槽位里的状态，所以这个意图随之作废。
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.forget_pending(seq)
        # A midstep reservation is the same intent one step earlier: a slot
        # held for a forward that is now not going to run. Handed back vacant —
        # publishing it would file a hash over bytes nobody wrote. Before the
        # block table is cleared, because a preempted seq re-prefills through
        # `can_allocate` and would otherwise re-plan on top of a live list.
        # 【译】midstep预留是早一步的同一种意图：为一次现在不会再运行的forward保留的槽位。空着交回——发布它，等于把一个hash归档到
        #   一份没人写过的字节上。要在清空block table之前做，因为被抢占的seq会经 `can_allocate` 重新prefill，否则会在一张仍然
        #   有效的列表之上重新规划。
        self.cancel_midstep(seq)
        # 【译】（行尾注释）main分支上的 `array("i")` 没有 `.clear()`
        # 解释：block_table是BlockTable（array("i")的子类，sequence.py）。array在Python 3.13之前没有clear()（本项目要求 ≥ 3.10）；
        #   del bt[:] 还会让BlockTable重新取一个version（不是追加的修改都会换version）
        # 对比nano-vllm：nano-vllm在这里结束（block_table.clear()）；下面是ATOM的每请求状态槽位
        del seq.block_table[:]  # main's `array("i")` has no `.clear()`
        # 【分支】每请求状态（GDN的递归状态、V4的压缩器状态等，第4批state_pool.py）：state_slots = [已提交槽位, 投机回滚槽位…]
        if seq.has_per_req_cache and seq.state_slots:
            # Every slot the seq held — the committed one and its speculation
            # scratch, which is this request's alone and has no meaning to the
            # next one. A checkpoint it took is already back on the free list
            # under the state index; the source it was going to fork off is
            # dropped here rather than left to `release_state_pins`, because the
            # forward that owed the read is not going to happen and the slot
            # should not sit out a pass for a reader that no longer exists.
            # Unless a state load is in flight into the committed slot -- a
            # worker is writing it, so handing it back now would give the next
            # request a buffer someone else is filling, with `has_initial_state`
            # already true over it. Held until `settle_state_load`, which always
            # comes: the worker reports every load either way. The rollback
            # scratch is nobody's destination and goes back regardless.
            # 【译】这条seq持有的每个槽位——已提交的那个，以及它的投机scratch，后者只属于这个请求，对下一个请求毫无意义。它做过的
            #   checkpoint已经挂在状态索引下回到空闲列表；它本要fork的源在这里丢掉，而不是留给 `release_pins`，因为欠着那次读取的
            #   forward不会发生了，这个槽位不该为一个已经不存在的读者再空等一轮。
            #   除非有一次状态加载正在写进已提交槽位——一个worker正在写它，现在交回去，就会把一个别人正在填的缓冲区交给下一个请求，
            #   而它上面的 `has_initial_state` 已经是true。扣住直到 `settle_state_load`，它总会来：worker无论成败都会报告每次加载。
            #   回滚scratch不是任何人的目标，照常交回。
            # 解释：状态加载 = 【分支】KV offload的状态层（从CPU把一份状态镜像读回这个槽位，第9批state_offload.py）；
            #   _orphan_load_slots登记的槽位由settle_state_load / abandon_state_load / reconcile_orphan_load_slots最终释放
            if seq.offload_joint.load_hash != -1 and self.state_offload is not None:
                self._orphan_load_slots.setdefault(seq.id, []).append(
                    (seq.state_slot, monotonic())
                )
                # Abandoned, not failed: an abort says nothing about the bytes,
                # and forgetting the hash would cost the next request over this
                # prefix a full recompute.
                # 【译】是"放弃"而不是"失败"：取消并不说明那些字节有什么问题，忘掉这个hash会让下一个使用该前缀的请求付出一次完整重算。
                self.state_offload.abandon_load(seq.id)
                # Drop any not-yet-taken load queued for this request, exactly as
                # `cancel_state_load` does (finding #3). Without this, a requeued
                # or preempted admission leaves its entry in `_state_loads`, and
                # `_publish_state_loads` later hands the connector a load for a
                # request no longer in `metadata.requests` -- the tier writes into
                # a released or orphaned slot, and the completion can settle this
                # request's NEXT-generation park, attributing a state restore to
                # the wrong generation silently. A no-op once the load was taken
                # (in flight): the entry is already gone and the orphan-slot path
                # above holds the slot until `settle_state_load`.
                # 【译】丢掉为这个请求排队、还没被取走的任何加载，和 `cancel_state_load` 的做法完全一样（finding #3）。不这样做的话，
                #   一次重新排队或被抢占的准入会把它的条目留在 `_state_loads` 里，之后 `_publish_state_loads` 会把一个已不在
                #   `metadata.requests` 中的请求的加载交给connector——层把数据写进一个已释放或孤儿化的槽位，而完成报告可能结算掉
                #   这个请求*下一代*的停靠，把一次状态恢复悄悄算到错误的那一代上。加载一旦被取走（在途），这就是空操作：条目已经不在了，
                #   上面的孤儿槽位路径会扣住槽位直到 `settle_state_load`。
                self._state_loads = [e for e in self._state_loads if e[0] != seq.id]
                self.state.release_many(seq.state_slots[1:])
            else:
                self.state.release_many(seq.state_slots)
            # No next forward will read a pending fork source after deallocation.
            # 【译】释放之后，不会再有下一次forward去读一个待执行的fork源。
            # 解释：drop_reader：减掉这个读者的pin，没有别的读者了就把源槽位放回空闲列表（fork见cancel_state_fork上方）
            self.state.drop_reader(seq.state_fork_src)
            seq.state_slots = []
            seq.state_fork_src = -1

        # 【分支】KV offload的联合边界（KV + 状态一起从CPU层恢复，第9批）：清掉状态加载的hash和联合加载的范围
        seq.offload_joint.load_hash = -1
        seq.offload_joint.reset_joint()

    # ═══════════ can_append / may_append：decode（以及投机解码的草稿）要写KV的位置都得有block（A级） ═══════════
    # 调用方：Scheduler._schedule的decode循环——先 while not can_append(...) 抢占队尾直到够用，再may_append真正分配；
    #   准入之后Scheduler._allocate_prefill_draft_lookahead调 may_append(seq, 0, lookahead=draft_lookahead) 补草稿要写的块
    # ebs = _effective_block_size() = block_size × DCP：一个block_table条目覆盖的全局token数（不开DCP时就是block_size）
    def can_append(
        self, seq: Sequence, num_new_tokens: int = 1, lookahead: int = 0
    ) -> bool:
        """Whether the pool can back what `may_append` takes for this step.

        Sized on `len(seq) + max(num_new_tokens, lookahead)`. A decode step's
        `num_new_tokens` (`1 + mtp_k`) already exceeds its draft lookahead
        (`mtp_k - 1`), so the lookahead never raises the bar there; it is
        passed so the check states what it covers.
        """
        # 【译】池子能否供得起 `may_append` 这一步要拿的block。
        #
        #   按 `len(seq) + max(num_new_tokens, lookahead)` 估算。decode一步的 `num_new_tokens`（`1 + mtp_k`）总是大于它的草稿前瞻
        #   （`mtp_k - 1`），所以前瞻在那里从不抬高门槛；传进来是为了让这个检查写明它覆盖了什么。
        # 解释：needed_blocks = ceil(reach / ebs)，减去已有的块数；_has_page_units：没有PAGE checkpoint时就是"空闲block ≥ n"，
        #   有PAGE checkpoint时还把可以淘汰掉的checkpoint单元算作可用
        # 观察：may_append实际只按 len(seq)（目标写到 len(seq) − 1）和 len(seq) + lookahead 分配，这里却按 len(seq) + num_new_tokens 估，
        #   多估了num_new_tokens个位置。例（ebs = 16、不开MTP）：len = 32 时位置31是块1的最后一格，may_append不分配，这里却算出
        #   需要 ceil(33/16) = 3 块、要求1个空闲block——池子正好用满时会多抢占一条请求。偏保守，不会少分配
        # 对比nano-vllm：nano-vllm是 len(free) >= (len(seq) % block_size == 1)，和它的may_append逐位一致
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        reach = seq_len + max(num_new_tokens, lookahead)
        needed_blocks = (reach + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        return self._has_page_units(new_blocks_needed)

    def may_append(self, seq: Sequence, num_new_tokens: int = 1, lookahead: int = 0):
        """Allocate every block this step's KV writes land in.

        The target writes up to `len(seq) - 1`; the draft round writes
        `lookahead` positions past that (`draft_kv_lookahead`). Checking that
        the pool has room (`can_append`) is not the same as allocating it: a
        position past the last allocated block reads block id 0 from the zeroed
        block-table tail, and the write lands in whoever owns physical block 0.
        """
        # 【译】为这一步的KV写入会落到的每个block分配好。
        #
        #   目标模型写到 `len(seq) - 1`；草稿轮再往后写 `lookahead` 个位置（`draft_kv_lookahead`）。检查池子有空间（`can_append`）
        #   和真正分配它不是一回事：一个超出最后已分配block的位置，会从block table被清零的尾部读到block id 0，写入就落进物理block 0
        #   的主人那里。
        # 解释（ebs = 16，即block_size 16、不开DCP）：
        #   不开MTP（num_new_tokens = 1，lookahead = 0），本步算位置 len − 1：
        #     len = 33：33 % 16 = 1 → 位置32是新块的第一格 → needed = ceil(33/16) = 3，block_table从2块补到3块
        #     len = 34：34 % 16 = 2 > 1 → 位置33落在已有的块2里，不分配；len = 48：48 % 16 = 0 → 不分配（位置47是块2的最后一格）
        #   开MTP k = 3（num_new_tokens = 1 + k = 4，lookahead = draft_kv_lookahead(3) = k − 1 = 2）：
        #     len里已含本步要验证的草稿占位，目标写位置 len−4 ~ len−1；草稿模型在forward里还要跑k − 1个中间步，
        #     每步往后多写1个位置，最远写到 len + k − 2
        #     len = 34：34 % 16 = 2 ≤ 4 → 位置30~33跨进块2 → needed = 3；lookahead：ceil((34 + 2)/16) = 3 → 补到3块
        #     len = 47：47 % 16 = 15 > 4 → 目标的位置43~46都在块2里；但草稿写到 47 + 3 − 2 = 48 → 块3
        #       → lookahead：ceil((47 + 2)/16) = 4 → 补块3。没有lookahead的话，位置48会从block table清零的尾部读到block id 0，
        #       把草稿KV写进别人的block 0（draft_kv_lookahead的docstring）
        # 对比nano-vllm：nano-vllm只在 len(seq) % block_size == 1 时补一块；ATOM一步可能写多个位置（MTP验证1 + k个、草稿再多写
        #   k − 1个），所以条件放宽成"本步写的位置跨进了新块"，再用lookahead兜住草稿写的位置
        # Note: in disaggregated (P/D) mode the scheduler skips the
        # `num_new_tokens` half on the first decode step after remote prefill,
        # because those blocks were already allocated during the KV transfer.
        # 【译】注意：在分离式（P/D）模式下，远端prefill之后的第一个decode步，调度器会跳过 `num_new_tokens` 那一半，
        #   因为那些block在KV传输期间已经分配好了。
        # 【分支】PD分离（第9批）：调度器此时传 num_new_tokens = 0（注入了首token T0时除外），只靠lookahead补草稿要写的块
        block_table = seq.block_table
        seq_len = len(seq)
        # Check if we need to allocate a new block
        # When len(seq) % block_size == 1, we need a new block for the next token
        # When block_size == 1, every token needs a new block
        # 【译】检查是否需要分配新block：len(seq) % block_size == 1 时，下一个token需要一个新block；block_size == 1 时，每个token都需要新block。
        # 解释：代码已推广成 0 < len % ebs ≤ num_new_tokens（本步写的num_new_tokens个位置跨进了新块），例子见上面
        ebs = self._effective_block_size()
        needed_blocks = 0
        if 0 < seq_len % ebs <= num_new_tokens or ebs == 1:
            needed_blocks = (seq_len + ebs - 1) // ebs
        if lookahead:
            needed_blocks = max(needed_blocks, (seq_len + lookahead + ebs - 1) // ebs)
        while len(block_table) < needed_blocks:
            # Decode-generated blocks: token not finalized yet (depends on
            # sampling / speculative verification), so we cannot compute a
            # correct hash here.  Just allocate the block without hashing.
            # 【译】decode生成的block：token还没定下来（取决于采样 / 投机验证），所以这里算不出正确的hash，只分配、不登记hash。
            # 解释：登记留给hash_decode_blocks。_fresh_block = 必要时先腾出PAGE单元，再kv.pop()（空白block优先，其次是最久没用的
            #   可复用block——这时它的旧hash才被淘汰）+ kv.allocate()
            block_table.append(self._fresh_block())

    def draft_writes_allocated(self, seq: Sequence) -> bool:
        """Whether `seq`'s blocks reach the end of this forward's draft round.

        `len(seq) + draft_lookahead` positions, the bound `draft_kv_lookahead`
        derives. The scheduler checks it on every sequence it schedules.
        """
        # 【译】`seq` 的block是否覆盖到这次forward草稿轮写的终点。
        #
        #   即 `len(seq) + draft_lookahead` 个位置，这是 `draft_kv_lookahead` 推出的上界。调度器对它调度的每条序列都检查一次。
        # 解释：调用方Scheduler._check_draft_writes_allocated（不满足就报错，免得草稿KV写进block 0）。例：ebs = 16、len = 47、
        #   draft_lookahead = 2 → 需要 4×16 = 64 ≥ 49，即block_table至少4块
        ebs = self._effective_block_size()
        reach = len(seq) + self.draft_lookahead
        return len(seq.block_table) * ebs >= reach

    # ---------------- KV event API ---------------- #
    # 【译】---------------- KV事件API ----------------
    # 【分支】KV事件（第4批distributed/kv_events.py；配置见config.py KVEventsConfig上方的说明，只由ATOM_KV_EVENTS_*环境变量打开）
    # 前置：外部组件（例如按前缀命中情况路由请求的KV感知路由器、LMCache）想知道"哪台实例的GPU里缓存着哪些前缀block"。
    #   BlockManager在这几处往_event_log追加事件（消息格式兼容vLLM的kv_events）：
    #     存入：BlockStored(block_hashes, parent_block_hash, token_ids, block_size = hbs, medium, token_offset)
    #       hash_blocks、publish_loaded_prefix登记新block时（medium = GPU）；record_remote_store（PD decode端收到远端KV，REMOTE）
    #     淘汰：BlockRemoved([h])     _record_evicted：block被重新分配、它的hash被淘汰时
    #     清空：AllBlocksCleared()    clear_cache
    #   Scheduler.publish_kv_events在每个调度步结束时调take_events取走，交给发布器经ZMQ发出
    #   例（hbs = 16）：一个chunk登记了块2、3、4 → 一条 BlockStored(hashes=[h2, h3, h4], parent=h1, token_offset=2×16=32)

    def take_events(self) -> list[KVCacheEvent]:
        """Drain and return events accumulated since the last call."""
        # 【译】取走并返回自上次调用以来累积的事件。
        # 解释：直接换成一个新的空列表再返回旧的，不拷贝
        if self._event_log is None or not self._event_log:
            return []
        self._event_log, events = [], self._event_log
        return events

    def clear_cache(self) -> None:
        """Drop every prefix-cache entry. Used by `/reset_prefix_cache`-style
        admin APIs. Does NOT touch blocks currently held by live sequences —
        they remain valid via their block_table refs, just unhashable for
        future requests."""
        # 【译】丢掉所有前缀缓存条目。供 `/reset_prefix_cache` 一类的管理API使用。*不*影响活着的序列当前持有的block——它们通过
        #   block_table引用仍然有效，只是以后的请求无法再按hash命中它们。
        # 解释：同时清掉状态checkpoint的索引；_index_clears + 1 让can_allocate_memoized缓存的prefix walk全部失效
        # 观察：atom/下没有调用方（也没有 /reset_prefix_cache 这个HTTP接口），只有tests/里调用
        self.kv.clear_index()
        self._state_checkpoint_cache.clear_index()
        self._index_clears += 1
        if self._event_log is not None:
            self._event_log.append(_make_all_cleared())

    @property
    def kv_events_enabled(self) -> bool:
        """True iff KV events are being recorded."""
        # 【译】当且仅当正在记录KV事件时为True。
        return self._event_log is not None

    def record_remote_store(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_block_hash: int | None = None,
        token_offset: int | None = None,
    ) -> None:
        """Emit a BlockStored(medium=REMOTE) for blocks received from a remote
        KV transfer producer (Mooncake/MoriIO decode side). Called by the
        KVConnector worker once the transfer completes so external KV-cache
        consumers (LMCache, etc.) can track remote-resident blocks.

        `token_offset` is the sequence position of the first remote block."""
        # 【译】为从远端KV传输producer那里收到的block（Mooncake / MoriIO的decode端）发出一个 BlockStored(medium=REMOTE)。
        #   由KVConnector的worker在传输完成后调用，让外部的KV cache消费者（LMCache等）能追踪驻留在远端的block。
        #
        #   `token_offset` 是第一个远端block在序列中的位置。
        # 观察：docstring说由KVConnector的worker调用，实际唯一的调用方是Scheduler._update_waiting_for_remote_kv
        #   （PD consumer收完KV之后，紧接在register_received_prefix之后）
        if self._event_log is None or not block_hashes:
            return
        self._event_log.append(
            _make_block_stored(
                block_hashes,
                token_ids,
                parent_block_hash,
                self.hash_block_size,
                medium=MEDIUM_REMOTE,
                token_offset=token_offset,
            )
        )
