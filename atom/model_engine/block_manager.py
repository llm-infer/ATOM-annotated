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
_WALK_HASH_ROUND_MIN = 64
_WALK_HASH_ROUND_MAX = 1024
# Blocks `_hit_walk` looks up per round on the cached chain: the first block
# alone, then 16, growing 4x; bounds the lookups past the first miss to ~4x
# the hit.
# 【译】_hit_walk 在已缓存的prompt链之外每轮hash的block数，从MIN开始翻倍到MAX；限制第一次未命中之后白算的hash数。
#   _hit_walk 在已缓存的链上每轮查表的block数：先只查第一个block，然后16个，每轮×4；把第一次未命中之后多查的次数
#   限制在命中长度的约4倍。
# 例（prompt链已由前端算好，一个128k token的prompt、block 16 → 8000块）：冷prompt只查1次就知道没命中；
#   全命中时按1、16、64、256、1024、4096…一轮轮查，几轮就走完。链没算到的部分按64、128、256、512、1024、1024…块一轮现算
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

    @classmethod
    def compute_hash(cls, token_ids: array.array, prefix: int = -1):
        return block_hash(token_ids, prefix)

    def complete_previous_state_batch(self) -> None:
        """Complete state reads and copies issued by the previous batch."""
        self.state.release_pins()
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.complete_previous_batch()

    def take_state_maintenance_ops(self) -> StateMaintenanceOps:
        """Drain state maintenance for the batch being built."""
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
        self.total_evicted_blocks += 1
        self._note_index_change(h)
        if self._event_log is not None:
            self._event_log.append(_make_block_removed([h]))
        self._state_checkpoint_cache.unindex(h)

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
        if not self._ensure_page_units(1):
            raise AssertionError("No PAGE unit available for a fresh KV block")
        block_id = self.kv.pop()
        self.kv.allocate(block_id)
        return block_id

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
        if not self.enable_prefix_caching:
            return True
        table = seq.block_table
        n_shared = sum(
            1
            for i in range(keep_blocks, len(table))
            if self.kv.block(table[i]).hash != -1
            and self.kv.block(table[i]).ref_count > 1
        )
        if n_shared and not self._ensure_page_units(n_shared):
            return False
        for i in range(keep_blocks, len(table)):
            if self.kv.block(table[i]).hash == -1:
                continue  # already-private fresh tail (or a prior disown)
            self.kv.free(table[i])
            table[i] = self._fresh_block()
        return True

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
        if self.dcp_world_size <= 1:
            return (seq_len + self.block_size - 1) // self.block_size
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
        if not self.draft_lookahead:
            return 0
        return self.num_pool_blocks(
            seq_len + self.draft_lookahead
        ) - self.num_pool_blocks(seq_len)

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
        capacity = self.kv.num_blocks
        # A prompt this long needs more than `capacity` blocks on some rank, so
        # it bounds the search from above: each rank holds at least
        # `1 / dcp_world_size` of it, i.e. over `capacity` blocks' worth.
        hi = (capacity + 1) * self.block_size * max(1, self.dcp_world_size)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.num_pool_blocks(mid) + self.draft_lookahead_blocks(mid) <= capacity:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _effective_block_size(self):
        return self.block_size * self.dcp_world_size

    def _hash_block_size(self) -> int:
        return self.hash_block_size

    def _n_hash_blocks(self, seq: Sequence) -> int:
        hbs = self.hash_block_size
        return (len(seq) + hbs - 1) // hbs

    def _hash_block_tokens(self, seq: Sequence, i: int) -> array.array:
        hbs = self.hash_block_size
        return seq.token_ids[i * hbs : (i + 1) * hbs]

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
        hbs = self.hash_block_size
        seq.replays_hit = seq.replay_kv_load = False
        if not self.state_replay_tokens or not seq.has_per_req_cache:
            return hit
        # The connector skips a load under OFFLOAD_MIN_LOAD_TOKENS, so neither
        # LMCache figure counts unless it clears that above where it starts.
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
        seq.replay_kv_load = False

    def _prompt_hashes(self, seq: Sequence) -> list[int]:
        """`seq`'s prompt-block hash chain, computed once per `cache_seed`;
        starts from the frontend's when it checks out (`_adopt_shipped_chain`)."""
        seed = seq.cache_seed
        cached = self._prompt_hash_chains.get(seq)
        if cached is None or cached[0] != seed:
            cached = (seed, self._adopt_shipped_chain(seq))
            self._prompt_hash_chains[seq] = cached
        return cached[1]

    def _adopt_shipped_chain(self, seq: Sequence) -> list[int]:
        """The frontend's chain for `seq` (consumed), or `[]` to hash here.
        Adopted only if it was minted for this block size and seed from these
        prompt tokens (`prompt_fingerprint`), and its first, middle and last
        links rehash equal here (the same hash function). Then trusted whole:
        walks, `allocate` and `hash_blocks` read it."""
        shipped = getattr(seq, "prompt_hash_chain", None)
        if shipped is None:
            return []
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

        def link_holds(i: int) -> bool:
            prev = chain[i - 1] if i else seed
            return chain[i] == self.compute_hash(self._hash_block_tokens(seq, i), prev)

        if n and not all(map(link_holds, {0, n // 2, n - 1})):
            return self._reject_shipped_chain("its hashes differ from this engine's")
        self.shipped_chains_adopted += 1
        return chain

    def _reject_shipped_chain(self, why: str) -> list[int]:
        # Warn once: a frontend/engine mismatch repeats on every request.
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
        if i < len(chain):
            return chain[i]
        h = self.compute_hash(self._hash_block_tokens(seq, i), prev)
        if i == len(chain) and i < full_prompt_blocks(seq, self.hash_block_size):
            chain.append(h)
        return h

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
        if not seq.has_per_req_cache or self.paged_state_checkpoints is None:
            return self._no_joint("no_paged_checkpoints")
        hbs = self._hash_block_size()
        # Resolved once from config at construction (`state_tier_chunk_tokens`),
        # not off the connector object: the scheduler holds whatever
        # `get_kvconnector` returned, and one without `chunk_size` would zero
        # this and disable the feature silently.
        chunk = self._joint_chunk_tokens
        lmc_tokens = int(seq.offload_joint.kv_prefix_tokens or 0)
        if chunk <= 0:
            return self._no_joint("no_chunk_size")
        # Floored to the grid everything below compares against, so the
        # covering-chunk check becomes true by construction rather than by
        # luck. (`get_num_new_matched_tokens` withholds one token on a
        # full-prompt hit, which takes the lookup off the grid.)
        lmc_tokens = (lmc_tokens // chunk) * chunk
        if lmc_tokens <= hbm_boundary * hbs:
            return self._no_joint("lmcache_within_hbm")
        # The KV leg moves whole chunks and the blocks below the HBM prefix are
        # shared, so an unaligned start cannot be rounded down.
        if (hbm_boundary * hbs) % chunk != 0:
            return self._no_joint("hbm_off_chunk_grid")
        cap = min(lmc_tokens // hbs, self._n_hash_blocks(seq) - 1)
        if cap <= hbm_boundary:
            return self._no_joint("no_room_above_hbm")
        chain = self._chain_to(seq, block_hashes, cap)
        # One scan: `_gated_hit` already returns the rightmost rung
        # `_resumable_from` accepts, so a decrement-and-rescan walk would spend
        # a fixpoint pass per rung to reach the same answer.
        candidate = self._gated_hit(seq, cap, chain)
        if candidate <= hbm_boundary:
            return self._no_joint("no_rung_above_hbm")
        tokens = candidate * hbs
        h = chain[candidate - 1]
        # Cannot fail given the two premises above; kept as their assertion.
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
        claim_blocks = min(len(block_hashes), candidate)
        # Floored to the chunk grid so the KV leg starts aligned and writes
        # only into blocks below it that nobody else can be reading.
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
        if claim_tokens % hbs != 0:
            return self._no_joint("claim_off_hash_grid")
        # Which tier the STATE leg came from. The KV leg's own tier is decided
        # separately by `_decide_load_after_alloc`, which is why this is not
        # named for the boundary as a whole. Read here, off the same cache the
        # boundary was computed against, and carried on the decision so the
        # commit does not have to look it up again.
        return JointBoundaryDecision(
            boundary_tokens=tokens,
            boundary_hash=h,
            kv_tokens=kv_tokens,
            claim_tokens=claim_tokens,
            state_from_hbm=self.paged_state_checkpoints.contains(h),
        )

    @staticmethod
    def _no_joint(reason: str) -> JointBoundaryDecision:
        """The empty decision, naming why no joint boundary was found.
        Every reason is counted at commit time, including "this build is not
        even trying", because a silent zero is indistinguishable from a feature
        that ran and found nothing.
        """
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
        prompt = self._prompt_hashes(seq)
        chain = list(block_hashes)
        k = min(len(chain), len(prompt))
        if k and (chain[0], chain[k - 1]) != (prompt[0], prompt[k - 1]):
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
        # 对比nano-vllm（block_manager.py的can_allocate / allocate，共约50行）：骨架相同——沿链式hash逐块查、第一次未命中就停、
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
        #   （遍历结果跨tick缓存）。"其余每一项"= 状态槽位、门控、重放、命中块里几块正被使用、池子够不够——它们随池子变化，
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
        prompt = self._prompt_hashes(seq)
        if self.fast_prefix_walk:
            hashes, ids = self._hit_walk(seq, prompt)
        else:
            hashes, ids, h = [], [], seq.cache_seed
            for i in range(self._n_hash_blocks(seq) - 1):
                h = self._block_hash(seq, prompt, i, h)
                block_id = self.kv.lookup(h)
                if block_id == -1 or self.kv.block(block_id).token_ids != (
                    self._hash_block_tokens(seq, i)
                ):
                    break
                hashes.append(h)
                ids.append(block_id)
        return _PrefixWalk(
            seq.num_tokens, seq.cache_seed, self.kv.index_version, hashes, ids
        )

    def _admission_walk(self, seq: Sequence) -> _PrefixWalk:
        """The exact probe's walk, kept on the fast walk for `seq`'s `allocate`
        (`_take_admission_walk`)."""
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
        n = self._n_hash_blocks(seq) - 1
        get = self.kv.index_get
        hashes: list[int] = []
        ids: list = []
        hash_round = _WALK_HASH_ROUND_MIN
        look = 1
        while len(hashes) < n:
            k = len(hashes)
            if k < len(prompt):
                new = prompt[k : min(n, k + look)]
                look = max(4 * look, _WALK_LOOKUP_ROUND_MIN)
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
                    full = full_prompt_blocks(seq, self.hash_block_size)
                    prompt += new[: max(0, full - k)]
                hash_round = min(2 * hash_round, _WALK_HASH_ROUND_MAX)
            new_ids = list(map(get, new))
            try:
                miss = new_ids.index(None)
            except ValueError:
                hashes += new
                ids += new_ids
                continue
            hashes += new[:miss]
            ids += new_ids[:miss]
            break
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
        walk = self._admission_walks.pop(seq, None)
        if (
            walk is None
            or walk.num_tokens != seq.num_tokens
            or walk.cache_seed != seq.cache_seed
        ):
            return None
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
        walk_of = self._memoised_walk if self.walk_memo else self._walk_prefix
        return self._evaluate(seq, walk_of, False, None)

    def _memoised_walk(self, seq: Sequence) -> _PrefixWalk:
        memo = self._cached_walks.get(seq)
        if memo is not None and self._walk_still_valid(seq, memo):
            return memo.walk
        walk = self._walk_prefix(seq)
        hit = len(walk.hashes)
        boundary_hash = None
        boundary_id = -1
        if hit < self._n_hash_blocks(seq) - 1:
            # Read back from the prompt chain, where the walk left it.
            prev = walk.hashes[-1] if hit else seq.cache_seed
            boundary_hash = self._block_hash(seq, self._prompt_hashes(seq), hit, prev)
            boundary_id = self.kv.lookup(boundary_hash)
        watched = set(walk.hashes)
        if boundary_hash is not None:
            watched.add(boundary_hash)
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
        if (
            memo.walk.num_tokens != seq.num_tokens
            or memo.walk.cache_seed != seq.cache_seed
            or memo.index_clears != self._index_clears
        ):
            return False
        new = self._index_changes - memo.index_changes_seen
        if new:
            if new > len(self._recent_index_changes):
                return False
            watched = memo.watched
            for h in islice(reversed(self._recent_index_changes), new):
                if h in watched:
                    return False
            memo.index_changes_seen = self._index_changes
        return (
            memo.boundary_hash is None
            or self.kv.lookup(memo.boundary_hash) == memo.boundary_id
        )

    def record_allocation(
        self, seq: Sequence, num_cached_blocks: int, block_hashes: list[int]
    ) -> None:
        """Commit the fit probe before any allocation changes the pool."""
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
        assert not seq.block_table
        # Admitted: the delayer no longer probes this seq.
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
        hbs = self._hash_block_size()
        requested_blocks = num_cached_blocks
        claim_blocks = max(
            num_cached_blocks,
            int(seq.offload_joint.claim_tokens or 0) // hbs,
        )
        h = seq.cache_seed
        hit_hash = -1
        # Claim the probe's ids as one run; the loop walks only the rest (a
        # joint tail past the HBM hit, or everything if the walk is stale).
        claimed = 0
        walk = self._take_admission_walk(seq, claim_blocks)
        if walk is not None and (claimed := min(claim_blocks, len(walk.ids))):
            ids = walk.ids[:claimed]
            self.kv.claim_many(ids)
            seq.block_table.extend(ids)
            h = walk.hashes[claimed - 1]
            if cached := min(claimed, num_cached_blocks):
                hit_hash = walk.hashes[cached - 1]
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
                claimed_tokens = i * hbs
                num_cached_blocks = min(num_cached_blocks, i)
                # Partial reset by design: the claim is clamped, not zeroed,
                # so this cannot go through `reset_joint`.
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
            if i < num_cached_blocks:
                hit_hash = h
        if self.state_replay_tokens and seq.replays_hit:
            # A replay starts from a fresh state slot, never a checkpoint: it
            # rebuilds the state below whatever the claim above reached.
            hit_hash = -1
            if num_cached_blocks < requested_blocks:
                # A KV-only load past the hit would start above a gap.
                seq.replay_kv_load = False
        # Pin the restore before fresh blocks can evict its checkpoint.
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
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
        for _ in range(len(seq.block_table), self.num_pool_blocks(len(seq))):
            seq.block_table.append(self._fresh_block())
        seq.num_cached_tokens = num_cached_blocks * hbs

        # Per-request cache: claim this seq's slot indices from the
        # pre-allocated state tensor (e.g. GDN mamba_k_cache, the V4 compressor
        # ring). The state class took its bytes before the paged class was sized
        # in ModelRunner.get_num_blocks(), so admitting a seq adds no further
        # paged-block cost. The state pool's free list is the sole admission
        # bound for state cache.
        if seq.has_per_req_cache and self.paged_state_checkpoints is None:
            # A joint load aims the state leg at its own boundary, which is
            # above the HBM hit by construction. `num_cached_tokens` stays at
            # the HBM prefix until the KV leg lands, so the forward covers
            # `[hbm, num_prompt)` if anything goes wrong from here.
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
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
            if not self.disown_claimed_prefix(seq):
                # Pool cannot back the private copies; signal the caller to
                # deallocate and requeue for a clean recompute next pass.
                return False
            seq.num_cached_tokens = 0
            # Clears only the boundary here, as the pre-record code did -- a
            # partial reset, not a full `reset_joint`.
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
        if not (seq.has_per_req_cache and start):
            return start
        hbs = self.hash_block_size
        if hit and self.state.transfer.replays:
            start -= hbs
        elif hit and self.state_replay_tokens and seq.replays_hit:
            start -= self.state_replay_tokens
        short = self.min_final_prefill_rows - (seq.num_tokens - start)
        if short > 0:
            start = (start - short) // hbs * hbs
        return max(0, start)

    def _state_leg_secured(self, seq: Sequence) -> bool:
        """Whether something will really put state behind this seq's boundary.

        Exactly three ways that can be true, and they are the three exits
        `_attach_state_slots` takes when it returns True with a boundary
        claimed: a PAGE restore was queued, a fork source was adopted or read,
        or a CPU load was requested. A cold start is not one of them -- it
        returns True as well, which is correct with no boundary and wrong with
        one.
        """
        if seq.offload_joint.load_hash != -1:
            return True  # a CPU load is in flight for it
        if getattr(seq, "state_fork_src", -1) != -1:
            return True  # a fork checkpoint is its source
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
        width = self.state_slots_per_req
        if self.paged_state_checkpoints is not None:
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = -1
            if hit_hash == -1:
                return True  # cold start: nothing claimed, nothing to restore
            if self.paged_state_checkpoints.begin_restore(hit_hash, seq.state_slot):
                # Queued for the next batch, which gathers the image out of its
                # PAGE units into the committed slot. What the caller holds is
                # what it asked for.
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
            if self._request_state_load(seq, hit_hash):
                return True
            # Neither tier can produce it. Reachable rather than defensive:
            # the tier's index is optimistic (`hashes` means "was stored once"),
            # and an HBM checkpoint can be unindexed between `can_allocate` and
            # here by another seq's `_fresh_block` in the same pass. Disown the
            # boundary -- blocks stay claimed and the forward recomputes.
            self.state_gate_lost_boundary += 1
            return False

        if self.state.transfer.replays:
            # Nothing to resume or load: a fresh slot, as for a cold start; the
            # forward rebuilds the state (`replay_start`).
            hit_hash = -1
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
            return hit_hash == -1
        # Being resumed from is the evidence a guessed position was right, so a
        # checkpoint that pays off stops being spent first — see
        # `StateSlotPool.mark_speculative`. Here rather than in `claim`, which
        # `_set_hash` also calls to re-file a slot that nobody read.
        self.state.promote(src)
        shared = self.state.is_pinned(src)
        if not shared:
            self.state.claim(src)
        if self.state.has_free(width):
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = src
            # Held off the free list until the forward that reads it is issued.
            self.state.pin(src)
            return True
        # `can_allocate` admitted this seq against a free list holding at least
        # `width`, and nothing else has run since, so the list can only be short
        # here if this seq's own `claim` above took the slot that made up the
        # count — which is `src`, unshared. Adopting it hands that slot straight
        # back to this seq, so the width is met.
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
        if not self._tier_can_serve(hit_hash):
            return False
        if not self.state_offload.request_load(seq.id, hit_hash):
            return False
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
        if self.state_offload is None:
            return
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
        if self.paged_state_checkpoints is None or self.state_offload is None:
            return []
        if not self.state_offload.can_store:
            # A load-only role. Handing over a store pins units against a
            # report the worker's save half will never produce.
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
        reclaimed = False
        if self.paged_state_checkpoints is not None:
            reclaimed = self.paged_state_checkpoints.was_reclaimed(op)
            # Idempotent: normally the source release already returned these
            # units, and this is the backstop for a store that failed before
            # it ever read them.
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
        if self.paged_state_checkpoints is None:
            return False
        return self.paged_state_checkpoints.has_offload_pins()

    def reclaim_stale_state_store_pins(self, timeout_s: float) -> int:
        """Release store pins whose report never came. See the store."""
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
        if timeout_s <= 0 or not self._orphan_load_slots:
            return 0
        cutoff = monotonic() - timeout_s
        reclaimed = 0
        for req_id in list(self._orphan_load_slots):
            parked = self._orphan_load_slots[req_id]
            # Each admission is stamped and expires on its own age -- a stale
            # park is dropped even when a newer one for the same id is still
            # inside the window (kept in place).
            fresh = []
            for slot, at in parked:
                if at <= cutoff:
                    # The index leg was already abandoned in `deallocate` when the
                    # slot was parked; only the slot itself outlived the report.
                    # Do not touch the index again here -- a second `abandon_load`
                    # would double-count the miss. A late report that still calls
                    # in finds this slot gone and releases the next parked one.
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
        if self.state_offload is None:
            return []
        out, self._state_loads = self._state_loads, []
        return out

    def _chain_parent_hash(self, seq: Sequence, start: int) -> int | None:
        """Return the chained hash of block ``start - 1``, or ``None`` on a gap.

        All source paths (register_prefill_hashes, postprocess, offload wake)
        are expected to hash blocks before this is called. A gap means a
        source-level bug; callers skip the range rather than mint false hashes.
        """
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
        if not self.enable_prefix_caching:
            return
        hbs = self._hash_block_size()
        base = seq.num_cached_tokens if start_tokens is None else start_tokens
        # A replaying forward's rows below where it resumed are published.
        until = seq.replay_until if replay_until is None else replay_until
        start = max(base, until) // hbs
        end = (base + num_new_tokens) // hbs
        # A finished or preempted seq has had its block table released; the
        # deferred publish paths can still reach it with a stale token count.
        end = min(end, len(seq.block_table))
        if start >= end and not (
            seq.replays_hit and until and base + num_new_tokens == until
        ):
            return
        h = self._chain_parent_hash(seq, start)
        if h is None:
            return
        # Watermark for the decode-side continuation, maintained here so every
        # prefill path feeds it without knowing about it.
        seq.num_hashed_tokens = max(seq.num_hashed_tokens, end * hbs)
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        # Prompt blocks the seq's chain covers (`_prompt_hashes`) are named
        # from it; only the rest are hashed here.
        known = self._prompt_hash_chains.get(seq)
        known = known[1] if known and known[0] == seq.cache_seed else ()
        if 0 < start <= len(known) and known[start - 1] != h:
            known = ()
        for i in range(start, end):
            token_ids = self._hash_block_tokens(seq, i)
            h = known[i] if i < len(known) else self.compute_hash(token_ids, h)
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
        self.commit_midstep(seq)

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
                aimed=False,
            )

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
        src = seq.state_fork_src
        if src < 0:
            return True
        if self.state.pin_count(src) > 1:
            return False
        self.state.release(seq.state_slot)
        self.state.invalidate(src)
        # Both flavours of source are pinned — held off the free list for the
        # forward that has to read them — so taking one over is just dropping
        # this request's claim on it. It used to matter which flavour it was:
        # `checkpoint` handed its source straight back, so adopting it meant
        # claiming it off the free list, and `pin_count` then undercounted the
        # readers this refuses to overwrite.
        self.state.unpin(src)
        seq.state_slot = src
        seq.state_fork_src = -1
        return True

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
        interval = self.state_checkpoint_interval_tokens
        if interval <= 0:
            return 0
        # The smallest room reaches furthest right, and `inf` — no class can
        # checkpoint this seq at all — falls out as 0 without a special case.
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
        seq.checkpoint_end_pos = 0
        if not self.state.enabled:
            return
        # 0 is the off switch for checkpointing as a whole, not just the grid,
        # and `checkpointers_at` enforces it by refusing every position. An
        # anchor recorded here anyway would still be cut for — a shortened
        # prefill chunk on every prompt, storing nothing, with no error to
        # show for it. Pinned by `test_interval_zero_anchors_nothing`.
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
        if not self._midstep_applies(seq):
            seq.block_hashes = []
            return
        # The continuation itself -- resume from the last APPENDED hash (on a
        # miss the loop's `h` holds the hash of the block that failed to match,
        # which was never appended and is not part of this chain) -- is exactly
        # `_chain_to`, run to the full prompt width. Delegate rather than
        # duplicate the loop; the only thing local to this path is the midstep
        # gate above and writing the result to `seq`.
        seq.block_hashes = self._chain_to(seq, block_hashes, self._n_hash_blocks(seq))

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
        if not self._midstep_applies(seq):
            return []
        interval = self.state_checkpoint_interval_tokens
        hbs = self.hash_block_size
        grid, min_offset = self.midstep_grid, self.midstep_min_offset
        hashes = seq.block_hashes
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
            snapped = snap_midstep_position(pos, start, end, grid)
            if snapped % hbs or not midstep_target_ok(
                snapped, start, end, grid, min_offset
            ):
                self.midstep_positions_unreadable += 1
                continue
            nblocks = snapped // hbs
            # A position past the chain is one `_extend_hash_chain` could not
            # name — nothing to file it under, so it cannot be found again.
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
        anchor = seq.checkpoint_end_pos
        if not (anchor and start < anchor <= end):
            return 0
        return snap_midstep_position(anchor, start, end, self.midstep_grid)

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
        if not seq.midstep_reservations:
            return
        self._state_checkpoint_cache.publish_midstep(seq.midstep_reservations, seq)
        # The rightmost published position, for the decode-side spacing rule in
        # `checkpointers_at`. Generation is never midstep-readable — its
        # forwards end where acceptance puts them — so it still measures from
        # here, and would otherwise re-checkpoint immediately after a prompt
        # that just filed one.
        seq.last_checkpoint_pos = max(
            seq.last_checkpoint_pos, max(p for _g, p, _h in seq.midstep_reservations)
        )
        seq.midstep_reservations = []
        seq.midstep_anchor_pos = 0

    def cancel_midstep(self, seq: Sequence) -> None:
        """Hand back reservations whose forward is not going to run."""
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
        applicable = [c for c in self.state_caches if c.applies(seq)]
        if all(c.readable_midstep for c in applicable):
            return 0
        rung = 0
        if limit := self.checkpoint_limit(seq):
            # `limit` is itself a multiple of the interval, so a chunk cut at
            # it needs no special case.
            rung = min(end, limit)
            rung -= rung % self.state_checkpoint_interval_tokens
        # A demand is capped by neither the grid nor `limit`. `limit` is the
        # last position on the *grid* that leaves the widest class its room; a
        # demand carries that room by construction, so it may sit to the right
        # of the last rung — or, on a prompt too short for the grid to place a
        # rung at all, be the only position either side has.
        demand = seq.checkpoint_demand_pos
        # The prompt-end anchor sits outside the grid for the same reason and
        # on the same terms — see `_record_checkpoint_end`, which applies the
        # `successor_room` test that makes it keepable.
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
        # `state_checkpoint_fates()`, matching `checkpoint_funnel`: it sums
        # every state class, so a build with a second class is not silently
        # undercounted here while the funnel line reports the full total.
        # `occupancy()` stays on the pool, which is the thing that has slots.
        return (
            self.kv.eviction_stats()
            | self.state.occupancy()
            | self.state_checkpoint_fates()
        )

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
        interval = self.state_checkpoint_interval_tokens
        if interval == 0 or pos <= 0:
            return []
        # -1: no grid, so `pos % interval` has nothing to say and asking it
        # would be a ZeroDivisionError's less honest cousin — -1 divides
        # everything, admitting every position as a rung. The demand and the
        # anchor are then the only two placements, which is the whole point.
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
            return []
        elif pos % self.hash_block_size or pos - seq.last_checkpoint_pos < interval:
            return []
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
            if not (aimed and c.readable_midstep)
            and c.applies(seq)
            and next_forward_tokens >= c.successor_room
        ]

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
        if not self.enable_prefix_caching:
            return 0

        start_token = max(0, int(start_token))
        end_token = min(int(end_token), int(seq.num_prompt_tokens))
        if end_token <= start_token:
            return 0
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
        seq.prefix_hashes_published = True
        return num_full - start

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
        for block_id in block_ids:
            self.kv.free(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.kv.free(block_id)
        seq.num_cached_tokens = 0
        # The block table is gone, so nothing of this seq is hashed any more.
        # Covers preemption too, which frees through here and re-prefills.
        seq.num_hashed_tokens = 0
        seq.replay = None
        seq.replays_hit = seq.replay_kv_load = False
        # Likewise the demand: it describes one admission against one cache
        # state, and a re-admitted seq gets a fresh answer from `can_allocate`
        # — including a fresh place in both funnel counters.
        seq.checkpoint_demand_pos = 0
        seq.checkpoint_demand_counted = False
        seq.checkpoint_demand_declined = False
        # The anchor is derived from `num_prompt_tokens` alone, so it would
        # still be correct on re-admission — but `_record_checkpoint_end` runs
        # unconditionally in `can_allocate`, and leaving a stale value here
        # would let a seq that never reaches that path keep a position nothing
        # re-validated against the current `state_caches`.
        seq.checkpoint_end_pos = 0
        seq.last_checkpoint_pos = 0
        # An uncommitted checkpoint describes state in a slot that is about to
        # go back on the free list, so the intent dies with it.
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.forget_pending(seq)
        # A midstep reservation is the same intent one step earlier: a slot
        # held for a forward that is now not going to run. Handed back vacant —
        # publishing it would file a hash over bytes nobody wrote. Before the
        # block table is cleared, because a preempted seq re-prefills through
        # `can_allocate` and would otherwise re-plan on top of a live list.
        self.cancel_midstep(seq)
        del seq.block_table[:]  # main's `array("i")` has no `.clear()`
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
            if seq.offload_joint.load_hash != -1 and self.state_offload is not None:
                self._orphan_load_slots.setdefault(seq.id, []).append(
                    (seq.state_slot, monotonic())
                )
                # Abandoned, not failed: an abort says nothing about the bytes,
                # and forgetting the hash would cost the next request over this
                # prefix a full recompute.
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
                self._state_loads = [e for e in self._state_loads if e[0] != seq.id]
                self.state.release_many(seq.state_slots[1:])
            else:
                self.state.release_many(seq.state_slots)
            # No next forward will read a pending fork source after deallocation.
            self.state.drop_reader(seq.state_fork_src)
            seq.state_slots = []
            seq.state_fork_src = -1

        seq.offload_joint.load_hash = -1
        seq.offload_joint.reset_joint()

    def can_append(
        self, seq: Sequence, num_new_tokens: int = 1, lookahead: int = 0
    ) -> bool:
        """Whether the pool can back what `may_append` takes for this step.

        Sized on `len(seq) + max(num_new_tokens, lookahead)`. A decode step's
        `num_new_tokens` (`1 + mtp_k`) already exceeds its draft lookahead
        (`mtp_k - 1`), so the lookahead never raises the bar there; it is
        passed so the check states what it covers.
        """
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
        # Note: in disaggregated (P/D) mode the scheduler skips the
        # `num_new_tokens` half on the first decode step after remote prefill,
        # because those blocks were already allocated during the KV transfer.
        block_table = seq.block_table
        seq_len = len(seq)
        # Check if we need to allocate a new block
        # When len(seq) % block_size == 1, we need a new block for the next token
        # When block_size == 1, every token needs a new block
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
            block_table.append(self._fresh_block())

    def draft_writes_allocated(self, seq: Sequence) -> bool:
        """Whether `seq`'s blocks reach the end of this forward's draft round.

        `len(seq) + draft_lookahead` positions, the bound `draft_kv_lookahead`
        derives. The scheduler checks it on every sequence it schedules.
        """
        ebs = self._effective_block_size()
        reach = len(seq) + self.draft_lookahead
        return len(seq.block_table) * ebs >= reach

    # ---------------- KV event API ---------------- #

    def take_events(self) -> list[KVCacheEvent]:
        """Drain and return events accumulated since the last call."""
        if self._event_log is None or not self._event_log:
            return []
        self._event_log, events = [], self._event_log
        return events

    def clear_cache(self) -> None:
        """Drop every prefix-cache entry. Used by `/reset_prefix_cache`-style
        admin APIs. Does NOT touch blocks currently held by live sequences —
        they remain valid via their block_table refs, just unhashable for
        future requests."""
        self.kv.clear_index()
        self._state_checkpoint_cache.clear_index()
        self._index_clears += 1
        if self._event_log is not None:
            self._event_log.append(_make_all_cleared())

    @property
    def kv_events_enabled(self) -> bool:
        """True iff KV events are being recorded."""
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
