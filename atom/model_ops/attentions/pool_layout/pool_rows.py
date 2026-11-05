# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which row of a pool each module gets, and how many there are to buy.

One walk answers both. Sizing asks how many rows of each kind a builder needs
and binding asks which row a module holds, and they read the same assignment
rather than being two walks that agree by convention: the runner's bind walk
offers a draft's modules to the target builder whenever a sibling builder
declines them, and a counter following that walk would quietly number them into
rows nobody bought. Here that is a `KeyError`.

A mixin rather than a base class because the two things that need it are not
kin -- an attention metadata builder, and a draft's own KV pool, which is not
an attention builder and whose package promises to import no attention at all.
"""

# 【译】每个 module 在池里分到哪一行，以及总共要买多少行。
#
#   一次遍历同时回答两个问题。定大小时问"一个 builder 每种行要多少"，绑定时问"一个 module 占哪一行"，
#   两者读的是同一份分配结果，而不是两次靠约定保持一致的遍历：runner 的绑定遍历在兄弟 builder 不接某个
#   草稿模型的 module 时，会把它交给目标 builder；一个跟着那次遍历计数的计数器，就会悄悄把它们编进没人买过
#   的行里。在这里，那会是一个 KeyError。
#
#   做成 mixin 而不是基类，是因为需要它的两样东西没有亲缘关系——注意力元数据 builder，以及草稿模型自己的
#   KV 池；后者不是注意力 builder，而且它所在的包承诺完全不 import 任何注意力代码。
#
# =====================================================================================
# 【导读】pool_rows.py —— "行" = 池里某一层的那一份；哪个 module 是第几层，按实际构建出来的模型数（第5批 5b，B级）
#   这里的"行"指字段的层维（EntryField.layers 那一维，见 entry_arena.py）：MhaKvPool 的 k 视图是 [层, block, …]，
#   第 r 行就是第 r 个被缓存的注意力 module 的那一片。"行空间"（kind）= 一套独立的编号。
#   例（Qwen3-Next，TP=1：12 层全注意力 + 36 层 GDN）：
#     pool_rows = { KvGeometry(2, 256)（打印为 h2d256）: {全注意力 module_3: 0, module_7: 1, …, 共 12 个},
#                   "linear_state"（gdn_attn.LINEAR_STATE_ROWS）: {GDN module_0: 0, module_1: 1, …, 共 36 个} }
#     row_counts() = {h2d256: 12, "linear_state": 36} → 定大小时 MhaKvPool(layers=12)、状态池 36 层；
#     绑定时每个 module 用 pool_rows[kind][module] 取自己的行号，二者出自同一次遍历，不可能不一致
#   各后端的行空间：KvGeometry（aiter_attention.py，MHA，按 KV 头数和 head_dim 区分——一个模型有两种 KV 形状时
#     就有两个行空间、两个 MhaKvPool）、"index"（aiter_attention.py 的 INDEX_ROWS，MiniMax-M3 的索引 key）、
#     "mla"（aiter_mla.py MLA_ROWS）、"linear_state"（gdn_attn.py）、"qsa"（qwen4_exp_attn.py）、草稿 KV（spec_decode/draft_kv.py）
#   对比nano-vllm：nano-vllm 的 allocate_kv_cache 遍历 model.modules()，见到有 k_cache 的 module 就 layer_id += 1；
#     ATOM 把这个计数提前到定大小之前、并且和绑定共用，这样混合模型、草稿层共享 KV、PP 下只有部分层时都不用从 config 推算。
# =====================================================================================

from __future__ import annotations

from typing import NamedTuple


class KvGeometry(NamedTuple):
    """A KV row space, keyed by what one of its rows holds.

    A type rather than a bare tuple so that "which row spaces are pools" is a
    question with an answer -- a hybrid adds named row spaces of its own (the
    indexer's keys, a linear-attention slot), and telling them apart by
    exclusion means every new one has to be added to a list it does not know
    about.
    """

    # 【译】一个 KV 行空间，按它的一行装什么来作为 key。
    #   做成一个类型而不是裸 tuple，是为了让"哪些行空间是 KV 池"这个问题有答案——混合模型会加上自己的命名行空间
    #   （索引器的 key、线性注意力的槽位），如果靠排除法来区分它们，每新加一个都得被加进一个它根本不知道的列表里。
    # NamedTuple：不可变、可哈希，所以能当 dict 的 key；两个 module 的 (num_kv_heads, head_dim) 相同就落进同一个行空间
    # 构造方：aiter_attention.py 的 _module_kinds（KvGeometry(module.num_kv_heads, module.head_dim)，num_kv_heads 是本 rank 的头数）

    num_kv_heads: int
    head_dim: int

    def __str__(self) -> str:
        """What this row space is called, wherever one is named -- the startup
        log, and the `semantic_role` its transfer regions carry. Both are for a
        person: the default repr would print the class and field names."""
        # 【译】这个行空间在任何需要命名它的地方叫什么——启动日志，以及它的传输区域携带的 semantic_role。
        #   两者都是给人看的：默认的 repr 会打印类名和字段名。例：KvGeometry(8, 128) → "h8d128"
        return f"h{self.num_kv_heads}d{self.head_dim}"


class PoolRowsMixin:
    """`self.model_runner`'s modules, assigned to this builder's rows."""

    # 【译】把 self.model_runner 的各个 module 分配到本 builder 的各行。
    # 混入方：backends.py CommonAttentionBuilder（所以 MHA / MLA / GDN / V4 等 builder 都有）、spec_decode/draft_kv.py
    # 子类要实现的钩子：_module_kinds(module)；draft_kv.py 另外覆盖 _pooled_models（只遍历草稿模型）

    # Unassigned until something asks. On the class so a builder needs no
    # constructor of its own to be askable -- several are built by
    # `object.__new__` on paths that only want the geometry.
    # 【译】在被询问之前保持未分配。放在类上，这样 builder 不需要自己的构造函数也能被询问——有好几个 builder
    #   在只需要几何信息的路径上是用 object.__new__ 构造的。
    _pool_rows: dict | None = None

    def _pooled_models(self) -> list:
        """The models whose modules this builder's pools hold rows for."""
        # 【译】本 builder 的池为哪些模型的 module 保留行。
        #   草稿模型与目标模型共享 KV 池时（runner.draft_shares_kv_pool()），草稿模型的层接在目标模型之后编号
        runner = self.model_runner
        models = [runner.model]
        if runner.draft_shares_kv_pool():
            models.append(runner.drafter.model)
        return models

    def _module_kinds(self, module) -> tuple:
        """The row spaces this builder gives `module` a row in, if any.

        A row space is one numbering, not one pool: MiniMax-M3's indexer keys
        are a second row space over the same modules as its KV, and a hybrid's
        linear-attention state is a third the same builder hands out. Empty for
        a module it does not cache -- an `nn.LayerNorm`, another backend's
        attention -- which is also what keeps it out of the walk.
        """
        # 【译】本 builder 给 module 在哪些行空间里分配一行（如果有的话）。
        #   行空间是一套编号，不是一个池：MiniMax-M3 的索引器 key 是在和它的 KV 同一批 module 上的第二个行空间，
        #   混合模型的线性注意力状态是同一个 builder 分发的第三个。对它不缓存的 module——一个 nn.LayerNorm、另一个
        #   后端的注意力——返回空，这也正是把它挡在遍历之外的办法。
        # 覆盖方：aiter_attention.py、aiter_mla.py、gdn_attn.py、kimi_mla_gdn_attn.py、qwen4_exp_attn.py、spec_decode/draft_kv.py
        return ()

    @property
    def pool_rows(self) -> dict:
        """`{kind: {module: row}}`, in the order the modules are bound.

        Reading the layers off the modules is what turns every count that used
        to be derived from config -- which layers of a hybrid are full
        attention, where a draft's stack starts, what this PP stage holds --
        into a property of what was actually built.
        """
        # 【译】{kind: {module: row}}，按 module 被绑定的顺序。
        #   从 module 上读出层，就把过去每个从 config 推导出来的计数——混合模型里哪些层是全注意力、草稿模型的层从
        #   哪里开始、这个 PP stage 持有哪些层——变成了"实际构建出了什么"的属性。
        # 惰性计算并缓存：第一次访问时遍历 model.modules()（递归遍历所有子模块，顺序固定），之后直接返回
        # 读取方：各 builder 的 build_kv_cache_tensor / 绑定代码（如 aiter_attention.py pool_rows[geometry][module]、
        #   aiter_mla.py pool_rows[MLA_ROWS][module]、gdn_attn.py pool_rows[LINEAR_STATE_ROWS][module]）
        if self._pool_rows is None:
            rows: dict = {}
            for model in self._pooled_models():
                for module in model.modules():
                    for kind in self._module_kinds(module):
                        in_kind = rows.setdefault(kind, {})
                        # setdefault：kind 第一次出现时建空 dict；新 module 的行号 = 这个行空间里已有的个数（0, 1, 2, …）
                        in_kind[module] = len(in_kind)
            self._pool_rows = rows
        return self._pool_rows

    def row_counts(self) -> dict:
        """`{kind: rows}` -- what sizing charges for, from the same walk."""
        # 【译】{kind: 行数}——定大小时按这个收费，出自同一次遍历。
        # 调用方：各 builder 定大小（aiter_attention._declare_kv_pools、aiter_mla 的 _kv_pool_layers、gdn_attn.num_state_layers、
        #   qwen4_exp_attn、draft_kv）、ModelRunner 的启动日志（"… caches N <kind> rows"）
        return {kind: len(in_kind) for kind, in_kind in self.pool_rows.items()}

    def invalidate_pool_rows(self) -> None:
        """Drop the assignment, before a bind walk that may not match it.

        The runner's to call, because the model tree is its: a pool is re-bound
        on a P/D import and again on a rollout wake, and neither promises the
        same modules.
        """
        # 【译】在一次可能与它不符的绑定遍历之前，丢掉现有的分配。
        #   由 runner 来调用，因为模型树是它的：P/D 导入时、rollout 唤醒时池都会被重新绑定，两者都不保证 module 不变。
        # 调用方：ModelRunner._carve_paged_pool（定价之前，对每个 builder 调一次）；draft_kv.py 覆盖它以同时清掉自己的派生缓存
        self._pool_rows = None
