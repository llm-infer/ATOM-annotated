# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import itertools
import logging
import time
from collections import Counter
from dataclasses import fields
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.prefix_hash import hash_block_size_for, ship_prompt_hash_chain
from atom.model_engine.sequence import Sequence
from atom.multimodal.registry import get_mrope_input_positions
from atom.sampling_params import SamplingParams
from atom.utils import envs

logger = logging.getLogger("atom")


# 本文件：用户直接拿到的引擎对象LLMEngine，以及它的输入输出处理器InputOutputProcessor
#   谁创建LLMEngine：EngineArgs.create_engine()（第1批arg_utils.py），离线示例和在线服务都走这里
#   离线用法：llm.generate(prompts, sampling_params)（examples/simple_inference.py）
#   在线用法：api_server不调用generate，而是直接调engine.io_processor.preprocess / preprocess_fanout
#     把请求变成Sequence，再调engine.core_mgr.add_request；结果经流式回调送回（第2批openai/api_server.py）
#
# 一个请求的完整生命周期（单机、DP=1、TP=2为例；箭头上标的是传输方式）：
#
#   用户进程（本文件 + CoreManager）       EngineCore进程（每个DP rank一个）         ModelRunner进程×TP（每张GPU一个，由AsyncIOProc承载）
#   ────────────────────────────          ───────────────────────────────         ─────────────────────────────────
#   generate(prompts)  用户调用
#    │ add_request
#    │   io_processor.preprocess_fanout：
#    │     tokenize、stop字符串分词、建Sequence、记arrive_time
#    │   core_mgr.add_request(seqs) ──pickle + ZMQ ROUTER──> DEALER（输入线程）
#    │                                                      │ input_queue
#    │                                                      ▼
#    │                                                    busy_loop：
#    │                                                      scheduler.schedule() → ScheduledBatch
#    │                                                      runner_mgr.call_func("forward") ══共享内存广播══> TP0、TP1各自ModelRunner.forward()
#    │                                                                                                       （rank间all-reduce走RCCL）
#    │                                                      <══════════ 只有TP0的结果，ZMQ PUSH/PULL ══════════
#    │                                                      scheduler.postprocess()：追加token、判停、释放block
#    │                                                      已结束的序列 → output_queue → 输出线程
#    │   CoreManager输出线程 <──pickle + ZMQ PUSH/PULL──────┘
#    │     → outputs_queue
#    │ step() = core_mgr.get_output()：阻塞取回一批"已结束"的Sequence
#    │ io_processor.postprocess：detokenize、算TTFT/TPOT
#    ▼ 按seq id排序后返回
#
# 对比nano-vllm：nano-vllm的LLMEngine在自己进程里持有Scheduler和rank 0的ModelRunner，
#   step()就是"schedule → model_runner.call('run') → postprocess"一次完整迭代，由generate的while循环驱动；
#   ATOM的LLMEngine只做tokenize/detokenize和进程间收发：调度器在EngineCore进程里自己循环，
#   GPU计算在每张卡单独的ModelRunner进程里，step()不驱动任何计算，只是"等下一批结束的请求"
#   这样做的好处：tokenize、HTTP处理、调度、GPU计算分在不同进程，互不抢GIL；DP>1时可以有多个EngineCore
#
# 为什么tokenize / detokenize放在前端进程，而不放进GPU循环所在的EngineCore：
#   GPU循环的理想状态是"上一步forward一结束，下一步的输入已经备好"。一步decode往往只有几到几十毫秒，
#   而给一个几千token的prompt做tokenize、套chat模板、解析HTTP/JSON都是纯CPU的Python工作。
#   若它们和调度器在同一个进程，就要争同一把GIL（Python全局解释器锁：同一时刻只有一个线程在执行Python字节码），
#   这些CPU时间会直接变成GPU在两步之间的空闲气泡。放到前端后，EngineCore收到的已经是token id，回传的也只是token id
#   用多进程而不是多线程，才能真正绕开GIL；进程用spawn启动，避免fork继承父进程已初始化的CUDA/HIP上下文导致崩溃
#
# "step"在ATOM里有两层意思，容易混淆：
#   EngineCore里的一步 = busy_loop的一次迭代：schedule()选出一批 → 一次forward → postprocess。
#     与nano-vllm相同，一批要么全是prefill、要么全是decode（不混合；有可调度的prefill时通常先做prefill）；
#     decode批里每条序列这一步只长1个token（开投机解码时可以一次接受多个）
#   本文件的step() = 从outputs_queue取一项"已结束的序列列表"，与GPU的步调无关：EngineCore可能已经跑了几百步才回来一项


# 加载HF tokenizer；若"编码后再解码"不能还原探测字符串，就退回PreTrainedTokenizerFast
#   调用方：LLMEngine.__init__（调用方没有传入tokenizer时）
def _load_tokenizer(model: str, trust_remote_code: bool = False):
    # use_fast=True：用Rust实现的fast tokenizer（同nano-vllm）
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=True, trust_remote_code=trust_remote_code
    )
    # 探测字符串同时含英文、空格和中文：有些模型仓库的AutoTokenizer配置有问题，往返后会丢空格或乱码
    probe = "Hello world 你好"
    if tokenizer.decode(tokenizer.encode(probe), skip_special_tokens=True) != probe:
        logger.warning(
            "AutoTokenizer round-trip failed, falling back to PreTrainedTokenizerFast"
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    return tokenizer


# =====================================================================================
# LLMEngine：用户侧引擎（运行在调用者自己的进程里，不碰GPU）
# =====================================================================================
# 职责：
#   1. 构造Config（并补几个只有tokenizer知道的字段），做一组跨特性的兼容性检查
#   2. 创建InputOutputProcessor（tokenize / detokenize）和CoreManager（拉起EngineCore进程并与之通信）
#   3. 提供离线接口generate / add_request / step，以及统计、profile等转发到EngineCore的工具RPC（utility command）
#
# 成员变量：
#   config              本进程里的Config。CoreManager会为每个DP rank深拷贝一份再发给EngineCore进程，
#                       所以这里之后对config的修改（例如下面重新解析kv_transfer_config）不会影响已启动的进程
#   tokenizer           HF tokenizer，调用方可以传入现成的（api_server会这样做），否则从模型目录加载
#   data_parallel_size  DP rank数（最终以config.parallel_config为准）
#   io_processor        InputOutputProcessor，见本文件下半部分
#   core_mgr            CoreManager（普通模式）或DisaggCoreManager（RapidServe模式），见engine_core_mgr.py
#   _step_lock / _pending_results  初始化后仓库里没有任何地方读写，是遗留字段
class LLMEngine:

    # model：模型目录或HF仓库名；tokenizer：可选的现成tokenizer；kwargs：EngineArgs._get_engine_kwargs()的结果
    def __init__(self, model, tokenizer=None, **kwargs):
        # ---------- 1. 构造Config ----------
        # 同nano-vllm：只挑出Config里有同名字段的kwargs，其余被静默丢弃
        # dataclasses.fields(Config)：返回dataclass的字段描述列表，这里只取字段名
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # DP相关的两个kwargs不是Config字段，单独取出，供下面的"旧路径"使用
        data_parallel_size = kwargs.get("data_parallel_size", 1)
        data_parallel_master_port = kwargs.get("data_parallel_master_port", None)
        config = Config(model, **config_kwargs)
        self.config = config
        # ---------- 2. tokenizer，以及只有tokenizer才知道的字段 ----------
        self.tokenizer = tokenizer or _load_tokenizer(
            config.model, config.trust_remote_code
        )
        # 覆盖Config里的占位值（-1），调度器判停、延迟输出的占位token都要用eos_token_id
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        # stop_token_ids来自generation_config.json（Config.__post_init__里读入），可能也包含eos
        stop_token_ids = set(config.stop_token_ids)
        # separate eos_token_id from stop_token_ids
        # 【译】把eos_token_id从stop_token_ids中分离出来
        # 解释：调度器判停时先查eos_token_id（结束原因记为"eos"），再查stop_token_ids（记为"stop_<id>"），
        #   去掉重复项后两者各管一类
        stop_token_ids.discard(config.eos_token_id)
        config.stop_token_ids = list(stop_token_ids)
        # Legacy path only: callers that pass DP topology as loose kwargs
        # instead of a ParallelConfig. When a parallel_config was supplied it
        # is already authoritative, and overwriting it here would reset a
        # multi-node topology back to a single local rank.
        # 【译】只用于旧路径：调用方把DP拓扑当作零散的kwargs传入，而不是传一个ParallelConfig。
        #   如果传了parallel_config，它已经是权威的，在这里覆盖会把多机拓扑重置回单个本地rank。
        if "parallel_config" not in config_kwargs:
            config.parallel_config.data_parallel_size = data_parallel_size
            if data_parallel_master_port is not None:
                config.parallel_config.data_parallel_master_port = (
                    data_parallel_master_port
                )
        # 解释：CoreManager构造时若开启--enable-dp-attention，还会把parallel_config.data_parallel_size
        #   乘以原TP（见engine_core_mgr.py）；这里记录的是那之前的值
        # ---------- 3. 跨特性兼容性检查（TBO / PCP / PP / DP-attention），均为分支，见第9批 ----------
        self.data_parallel_size = config.parallel_config.data_parallel_size
        # TBO's two concurrent ubatches are supported by the mori EP
        # dispatch/combine path. The EP collective fallback instead issues
        # full DP all-gather/reduce-scatter operations from both ubatch streams;
        # those streams can enter the collectives in opposite order and hang.
        # Keep the fallback usable by serializing it until it has dedicated
        # per-ubatch communicators, analogous to the PCP/TP guard below.
        # 【译】TBO的两个并发ubatch由mori EP的dispatch/combine路径支持。EP的集合通信后备路径则会从两个
        #   ubatch流上各自发出完整的DP all-gather/reduce-scatter；两个流可能以相反的顺序进入这些集合通信而卡死。
        #   在它有专属的逐ubatch通信器之前，通过把它串行化来保持后备路径可用，与下面的PCP/TP保护类似。
        # 【分支】TBO（two-batch overlap）与EP/mori，见第7、9批
        # 解释：mori EP是否可用 = 后端选了auto/mori、没被环境变量禁用、且装了mori包
        mori_ep_enabled = (
            getattr(config, "moe_all2all_backend", "auto") in {"auto", "mori"}
            and not envs.ATOM_DISABLE_MORI_EP
            and importlib.util.find_spec("mori") is not None
        )
        # DP-attention + EP + TBO，但走的不是mori：关闭TBO（prefill和decode的TBO都关）
        if (
            config.enable_dp_attention
            and config.enable_expert_parallel
            and config.enable_tbo
            and not mori_ep_enabled
        ):
            logger.warning(
                "Disabling TBO for the DP-attention + EP collective fallback: "
                "concurrent all-gather/reduce-scatter ubatches can deadlock. "
                "Use mori EP or run the fallback without --enable-tbo."
            )
            config.enable_tbo = False
            config.enable_tbo_decode = False
        # PCP and DP-attention are not yet compatible: PCP stripe-splits
        # input_ids to 1/pcp_size in ForCausalLM.forward, but DP-attention's
        # `_gather_ids_for_dp` all-gathers using dp_metadata sizes computed on
        # the FULL (un-split) token count, so all_gatherv asserts
        # `1/pcp_size != full`.
        # 【译】PCP与DP-attention目前不兼容：PCP在ForCausalLM.forward里按条带把input_ids切成1/pcp_size，
        #   而DP-attention的 _gather_ids_for_dp 用基于"完整（未切分）token数"算出的dp_metadata大小做all-gather，
        #   所以all_gatherv会断言失败（`1/pcp_size != full`）。
        # 解释：这段注释说的是下面第二个if（PCP + DP-attention）；紧挨着它的这个if检查的其实是
        #   DP-attention + 流水线并行（pp>1），两者都直接报错
        if config.enable_dp_attention and config.pipeline_parallel_size > 1:
            raise ValueError(
                "--enable-dp-attention and pipeline-parallel (pp>1) "
                "cannot be used together."
            )
        if config.prefill_context_parallel_size > 1 and config.enable_dp_attention:
            raise ValueError(
                "prefill_context_parallel_size > 1 (-pcp) combined with "
                "--enable-dp-attention is not supported yet (may be supported "
                "in a future release): PCP splits tokens to 1/pcp_size while "
                "DP-attention's id-gather expects the full token count, "
                "causing an all_gatherv size mismatch. For now, disable one of "
                "them (use -tp N -pcp M without DP-attention, or -dp N "
                "--enable-dp-attention without -pcp)."
            )
        # 【分支】PCP（prefill上下文并行）与TBO的组合，见第9批
        # PCP + TBO prefill: supported via coordinated splitting.
        # PCP + TBO decode: not yet supported (pcp_all_reduce semantics under
        # per-request ubatch split are unverified).
        # 【译】PCP + TBO prefill：通过协调切分来支持。
        #   PCP + TBO decode：尚不支持（按请求切分ubatch时，pcp_all_reduce的语义尚未验证）。
        if config.prefill_context_parallel_size > 1 and config.enable_tbo:
            # TBO overlaps compute with the attn<->MoE PCP collectives, which
            # only exist in MoE merge mode (ATOM_PCP_MOE_MERGE=1). With
            # ATOM_PCP_MOE_MERGE=0, MoE runs on each rank's 1/W token shard
            # with NO extra comm between attn and MoE, so TBO has nothing
            # to overlap and only adds ubatch-splitting overhead. Force it off.
            # 【译】TBO让计算与attn<->MoE之间的PCP集合通信重叠，而这些通信只在MoE合并模式（ATOM_PCP_MOE_MERGE=1）下存在。
            #   ATOM_PCP_MOE_MERGE=0时，MoE在每个rank自己1/W的token分片上运行，attn与MoE之间没有额外通信，
            #   TBO没有可重叠的东西，只会增加ubatch切分开销。所以强制关闭它。
            if not envs.ATOM_PCP_MOE_MERGE:
                logger.warning(
                    "Disabling TBO because ATOM_PCP_MOE_MERGE=0: in this "
                    "situation it runs MoE on each rank's 1/W token shard with "
                    "no extra attn<->MoE communication, so TBO has nothing to "
                    "overlap and only adds overhead."
                )
                config.enable_tbo = False
                config.enable_tbo_decode = False
            elif config.tensor_parallel_size > 1:
                # Cross-communicator (PCP x TP) RCCL deadlock:
                # under TBO the PCP collectives (comm_stream) run concurrently
                # and UNORDERED with the TP-group all_reduces (compute_stream,
                # from attention wo_b / MoE RowParallelLinear). On the TPxPCP
                # rank grid with large collectives this forms a cross-rank
                # circular wait -> hang (reproduced MI355 TP4PCP2/TP2PCP4 merge
                # +TBO, 64k/c32). Serialized (non-TBO single stream) is fine, and
                # TP=1 has no TP communicator so it cannot form the cycle. Only
                # TP=1 + PCP keeps TBO; TP>1 falls back to non-TBO.
                # 【译】跨通信器（PCP x TP）的RCCL死锁：
                #   在TBO下，PCP集合通信（comm_stream）与TP组的all_reduce（compute_stream，来自attention的wo_b /
                #   MoE的RowParallelLinear）并发且无序地运行。在TPxPCP的rank网格上、集合通信又很大时，会形成跨rank的
                #   循环等待 -> 卡死（在MI355 TP4PCP2/TP2PCP4 merge+TBO、64k/c32下复现）。串行（非TBO单流）没问题，
                #   TP=1没有TP通信器，所以不可能形成环。只有TP=1 + PCP保留TBO；TP>1退回非TBO。
                logger.warning(
                    "Disabling TBO: prefill_context_parallel_size > 1 (-pcp) "
                    "with tensor_parallel_size > 1 (-tp) hangs under TBO due to "
                    "a PCP<->TP cross-communicator RCCL deadlock (concurrent "
                    "unordered collectives on comm/compute streams; see "
                    "PCP_TBO.md 14.4). TBO with PCP is only supported at -tp 1 "
                    "(e.g. -tp 1 -pcp 8)."
                )
                config.enable_tbo = False
                config.enable_tbo_decode = False
            elif config.enable_tbo_decode:
                raise ValueError(
                    "prefill_context_parallel_size > 1 (-pcp) combined with "
                    "--enable-tbo all (decode TBO) is not supported yet. "
                    "Use --enable-tbo (prefill only) with -pcp."
                )
            # Under PCP, TBO prefill uses a request-boundary split (the
            # non-default TBO split mode; never token-midpoint split), so
            # ATOM_TBO_PREFILL_TOKEN_SPLIT is ignored.
            # 【译】在PCP下，TBO prefill按请求边界切分（非默认的TBO切分模式；从不按token中点切分），
            #   所以ATOM_TBO_PREFILL_TOKEN_SPLIT被忽略。
            if config.enable_tbo and envs.ATOM_TBO_PREFILL_TOKEN_SPLIT:
                logger.warning(
                    "ATOM_TBO_PREFILL_TOKEN_SPLIT is ignored under PCP: TBO "
                    "prefill uses request-boundary balanced grouping."
                )
        # ---------- 4. 输入输出处理器与CoreManager ----------
        # InputOutputProcessor在本进程里做tokenize/detokenize，并记录"还没结束的请求"
        self.io_processor = InputOutputProcessor(
            config, self.tokenizer, config.kv_cache_block_size
        )
        # 【分支】RapidServe（同机PD分离：prefill与decode两个EngineCore共享KV/权重），见第9批
        if config.enable_rapidserve:
            self.core_mgr = DisaggCoreManager(config)
        else:
            # 普通路径：CoreManager构造时拉起所有EngineCore进程，并（在协调节点上）阻塞到每个EngineCore发回READY
            #   （也就是模型加载、KV cache分配、CUDA Graph捕获都完成之后）才返回，见engine_core_mgr.py
            self.core_mgr = CoreManager(config)
        # 下面两个字段在仓库中没有任何读写，属于遗留代码
        self._step_lock = None
        self._pending_results = {}
        # 函数体内的局部import（只在这里用到json）
        import json

        # 【分支】PD分离 / KV offload的KV传输配置（ATOM_KV_OFFLOAD也被折叠进这里，见第1批arg_utils.py），见第9批
        # 解释：Config.__post_init__已经解析过同一个kv_transfer_config，这里在本进程的config上再解析一次；
        #   此时EngineCore进程已经用各自的Config副本启动，不受影响。kwargs里没有这个键时结果为{}；
        #   注意这里只捕获JSONDecodeError，若kwargs传的是dict而不是JSON字符串，json.loads会抛TypeError
        kv_config_str = kwargs.get("kv_transfer_config", "{}")
        try:
            config.kv_transfer_config = json.loads(kv_config_str)
            logger.info(f"KV transfer config loaded: {config.kv_transfer_config}")
        except json.JSONDecodeError:
            config.kv_transfer_config = {}
        # （同一行日志打印了两次，原代码如此）
        logger.info(
            f"LLMEngine init with {self.data_parallel_size} data parallel ranks"
        )
        logger.info(
            f"LLMEngine init with {self.data_parallel_size} data parallel ranks"
        )

    def close(self):
        """Shut down engine and release all GPU resources."""
        # 【译】关闭引擎并释放所有GPU资源。
        # 关闭顺序：CoreManager.close()给每个EngineCore发SHUTDOWN → EngineCore退出时调用call_func("exit")
        #   关闭各ModelRunner进程（见async_proc.py）→ CoreManager等待/终止EngineCore进程
        # hasattr：__init__中途失败时core_mgr可能还没创建
        if hasattr(self, "core_mgr"):
            self.core_mgr.close()

    # 提交一批prompt（离线路径；在线路径直接调用io_processor + core_mgr.add_request，不经过这里）
    # 输入：
    #   prompt_or_tokens_list  文本或token id列表组成的列表
    #   sampling_params_list   单个SamplingParams（所有prompt共用）或与prompt等长的列表
    #   stream_callback        可选；单个回调（共用）或等长列表。每次有新token时在CoreManager输出线程里被调用
    #   multimodal_data_list   可选；每个prompt的多模态数据（见generate_multimodal）
    #   request_ids            可选；用户侧请求id，只用于日志/追踪
    # 输出：无；序列被交给CoreManager，结果要通过step()取回
    def add_request(
        self,
        prompt_or_tokens_list: list[str | list[int]],
        sampling_params_list: SamplingParams | list[SamplingParams],
        stream_callback=None,
        multimodal_data_list: list[dict] | None = None,
        request_ids: list[str] | None = None,
    ):
        """Submit one batch of prompts.

        ``SamplingParams.n > 1`` fans out: a prompt becomes ``n`` sibling
        sequences, and the batch handed to the scheduler is prompt-major --
        prompt 0's siblings in order, then prompt 1's. Sequence ids are assigned
        in that same order, which is what makes :meth:`generate`'s flat list
        line up with a prompt list expanded by ``n``. It is one request per
        sibling from here on; nothing downstream reassembles them.

        This used to reach ``io_processor.preprocess``, which returns a single
        sequence and refuses ``n > 1`` outright, so offline ``n > 1`` raised
        before a token was generated. That guard is still there and still right
        for a caller that expects one sequence back -- the two entry points
        differ in whether the caller is prepared for siblings, not in what they
        support.
        """
        # 【译】提交一批prompt。
        #
        #   SamplingParams.n > 1 时会扇出：一个prompt变成n个兄弟序列，交给调度器的这批序列按prompt为主序排列——
        #   先是prompt 0的所有兄弟（按顺序），再是prompt 1的。序列id也按同样的顺序分配，这正是 generate 返回的扁平列表
        #   能与"按n展开后的prompt列表"对齐的原因。从这里开始每个兄弟就是一个独立请求；下游没有任何地方会把它们重新组装。
        #
        #   这里过去调用的是 io_processor.preprocess，它只返回单个序列并直接拒绝 n > 1，所以离线的 n > 1 在生成任何token
        #   之前就报错。那个保护仍然存在，而且对"期望只拿回一个序列"的调用方仍然正确——两个入口的区别在于调用方是否准备好
        #   接收兄弟序列，而不在于它们支持什么。
        # 例：prompts=[P0, P1]，n=2 → 交给CoreManager的是[P0#0, P0#1, P1#0, P1#1]，seq id依次递增
        # if sampling params is not list, use it for all prompts
        # 【译】如果sampling params不是列表，就把它用于所有prompt
        # itertools.repeat(x)：无限重复x的迭代器，与下面的zip搭配时，长度由prompt列表决定
        if not isinstance(sampling_params_list, list):
            sampling_params_iter = itertools.repeat(sampling_params_list)
        else:
            # otherwise check num elements first
            # 【译】否则先检查元素个数
            if len(prompt_or_tokens_list) != len(sampling_params_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and sampling_params_list is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(sampling_params_list)=}"
                )
            sampling_params_iter = sampling_params_list

        # Handle stream_callback
        # 【译】处理stream_callback
        if stream_callback is not None and not isinstance(stream_callback, list):
            stream_callback_iter = itertools.repeat(stream_callback)
        elif isinstance(stream_callback, list):
            if len(stream_callback) != len(prompt_or_tokens_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and stream_callback is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(stream_callback)=}"
                )
            stream_callback_iter = stream_callback
        else:
            stream_callback_iter = itertools.repeat(None)

        # Handle multimodal data
        # 【译】处理多模态数据
        if multimodal_data_list is not None:
            if len(prompt_or_tokens_list) != len(multimodal_data_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and multimodal_data_list is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(multimodal_data_list)=}"
                )
            mm_data_iter = multimodal_data_list
        else:
            mm_data_iter = itertools.repeat(None)

        # Handle request_ids
        # 【译】处理request_ids
        if request_ids is not None:
            if len(request_ids) != len(prompt_or_tokens_list):
                raise ValueError(
                    "number of elements in prompt_or_tokens_list and request_ids is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(request_ids)=}"
                )
            request_id_iter = iter(request_ids)
        else:
            request_id_iter = itertools.repeat(None)

        # 逐个prompt做预处理：preprocess_fanout返回长度为n的Sequence列表（n=1时只有一个）
        reqs = []
        for prompt, sampling_param, callback, mm_data, request_id in zip(
            prompt_or_tokens_list,
            sampling_params_iter,
            stream_callback_iter,
            mm_data_iter,
            request_id_iter,
        ):
            fanout = self.io_processor.preprocess_fanout(
                prompt,
                sampling_param,
                stream_callback=callback,
                multimodal_data=mm_data,
                parent_request_id=request_id,
            )
            reqs.extend(fanout)
        # 一次性把整批序列交给CoreManager：pickle后经ZMQ发给EngineCore（DP>1时按负载均衡分到各DP rank）
        #   seq id在本进程里由Sequence.counter分配（Sequence.__init__），所以跨所有DP rank唯一，EngineCore直接沿用
        #   DP>1时每条序列单独选rank（_dispatch_to_dp_ranks），n>1的兄弟序列也可能落到不同rank
        # 背压（back-pressure）：这里没有显式的背压，add_request从不因为"引擎太忙"而阻塞或拒绝——
        #   make_zmq_socket把这些socket的SNDHWM/RCVHWM设为0（不设上限），outputs_queue也是无界的queue.Queue。
        #   超出GPU容量的请求会在EngineCore的scheduler.waiting里排队（受max_num_seqs、max_num_batched_tokens和空闲KV block限制），
        #   表现为TTFT变长；需要限流的话得在更上层（网关或客户端并发数）做
        self.core_mgr.add_request(reqs)

    # 取回下一批"已结束"的序列（阻塞直到有结果）
    # 对比nano-vllm：nano-vllm的step()执行一次"调度 + forward + 后处理"，返回本步结束的序列和吞吐数据；
    #   ATOM的EngineCore进程一直在自己的busy_loop里跑，这里的step()只是从CoreManager的outputs_queue里
    #   阻塞取一项。若取到的是异常对象（例如进程关闭时放入的），get_output会把它抛出
    # 从前端看continuous batching（示意：A、B一起提交，C稍后才到达；横轴是EngineCore的step编号）：
    #   step:  1        2 ........ 40 ....... 120 ...... 300
    #   A      prefill  decode ... 结束
    #   B      prefill  decode ...................... 结束(300)
    #   C               (第3步prefill加入) ... 结束(120)
    #   step()依次返回：[A] → [C] → [B]（同一步结束的会在同一个列表里）
    #   请求随时加入、随时离开，batch的成员每一步都可能变化；返回顺序取决于"谁先结束"，与提交顺序无关，
    #   所以generate最后要按seq id重新排序
    def step(self) -> list[Sequence]:
        seqs = self.core_mgr.get_output()
        return seqs

    # 本进程登记的请求是否都已经被postprocess取走（不是"EngineCore是否空闲"）
    def is_finished(self):
        return not self.io_processor.has_pending_requests()

    # 离线批量生成：提交全部prompt，循环取结果直到所有请求结束
    # 输入：prompts（文本或token id列表）、sampling_params（单个或等长列表）、request_ids（可选）
    # 输出：扁平列表，每项是postprocess产生的dict（text、token_ids、latency、ttft、tpot等）
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        request_ids: list[str] | None = None,
    ) -> list[str]:
        """Complete every prompt and return the outputs as ONE flat list.

        Not one entry per prompt: ``SamplingParams.n`` of them, prompt-major
        (see :meth:`add_request`), because the sort below is over sequence ids
        and those are assigned in fan-out order. A caller pairing prompts with
        outputs has to expand its own list by ``n`` to do it -- with ``n = 1``,
        the usual case, that is the identity and the list is 1:1 as before.
        """
        # 【译】完成每个prompt，并把输出作为一个扁平列表返回。
        #
        #   不是每个prompt一项：而是每个prompt SamplingParams.n 项，按prompt为主序（见 add_request），因为下面的排序
        #   针对的是序列id，而序列id是按扇出顺序分配的。调用方若要把prompt与输出配对，需要自己把prompt列表按n展开——
        #   通常 n = 1，此时展开就是原样，列表与之前一样一一对应。
        # Reset DP routing state (round-robin cursor + in-flight load) so a
        # fresh batch gets deterministic DP assignment and no leaked counts.
        # 【译】重置DP路由状态（轮询游标 + 在途负载），使新的一批请求得到确定性的DP分配，且不会遗留计数。
        self.core_mgr.reset_dp_router()

        self.add_request(prompts, sampling_params, request_ids=request_ids)
        # seq_id -> 输出dict
        outputs = {}
        # 循环条件：还有未结束的请求，并且（还有EngineCore进程活着，或者outputs_queue里还有没取的结果）
        #   后半个条件保证：EngineCore进程全部意外退出时循环能结束，而不是永远卡住
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            # 每次拿到的是一批已结束的序列（不是逐token）；detokenize后合并进outputs
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        # 按序列id排序：id在本进程里全局自增，排序后的顺序就是提交（扇出）的顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    # 多模态生成：与generate相同，只是每个prompt额外带上视觉数据（已经是token id）
    #   多模态输入的处理见第2批multimodal/；本方法由examples/multimodal_inference.py调用
    def generate_multimodal(
        self,
        token_ids_list: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        multimodal_data_list: list[dict],
    ) -> list[dict]:
        """Generate completions for multimodal inputs (token IDs + vision data)."""
        # 【译】为多模态输入（token ID + 视觉数据）生成补全。
        self.core_mgr.reset_dp_router()
        self.add_request(
            token_ids_list,
            sampling_params,
            multimodal_data_list=multimodal_data_list,
        )
        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    # ---------- 以下是转发到EngineCore的工具RPC（utility command），由EngineUtilityHandler执行，见engine_utility.py ----------
    # 开始profile：广播到每个DP rank并等待它们回复；EngineCore再通过call_func让各ModelRunner启动torch profiler
    def start_profile(self):
        self.core_mgr.broadcast_utility_command_sync("start_profile")
        logger.info("Profiling started")

    # 停止profile：等待时间由ATOM_PROFILER_TIMEOUT决定（写trace文件可能很慢）；返回每个DP rank的结果
    def stop_profile(self) -> list[dict[str, Any]]:
        responses = self.core_mgr.broadcast_utility_command_sync(
            "stop_profile", timeout=envs.ATOM_PROFILER_TIMEOUT
        )
        return [resp.get("result", {}) for resp in responses]

    # 让每个EngineCore把MTP统计打印到它自己的日志里；不等待回复（fire-and-forget）
    def print_mtp_statistics(self):
        self.core_mgr.send_utility_command("get_mtp_stats")

    # 投机解码（MTP/EAGLE/DSpark）接受率统计，跨DP rank聚合，见第7批
    #   调用方：api_server的/debug/mtp_stats接口
    def get_mtp_statistics(self, timeout: float = 30.0) -> dict[str, Any]:
        """Return aggregated speculative decoding statistics across DP ranks."""
        # 【译】返回跨DP rank聚合后的投机解码统计。
        responses = self.core_mgr.broadcast_utility_command_sync(
            "get_mtp_statistics", timeout=timeout
        )
        # 每个DP rank的回复是{"result": {...}}；只保留开启了投机解码的rank
        rank_stats = [
            resp.get("result", resp)
            for resp in responses
            if resp.get("result", resp).get("enabled", False)
        ]

        # distribution：一步接受了k个草稿token -> 发生了多少步；各rank按k求和
        # Counter：dict的子类，update时按键累加计数
        distribution: Counter[int] = Counter()
        for stats in rank_stats:
            distribution.update(
                {
                    int(accepted): int(steps)
                    for accepted, steps in stats.get("distribution", {}).items()
                }
            )

        total_draft_tokens = sum(
            int(stats.get("total_draft_tokens", 0)) for stats in rank_stats
        )
        total_accepted_tokens = sum(
            int(stats.get("total_accepted_tokens", 0)) for stats in rank_stats
        )
        # 总步数 = 各k的步数之和
        total_steps = sum(distribution.values())

        return {
            "enabled": bool(rank_stats),
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "acceptance_rate": (
                total_accepted_tokens / total_draft_tokens
                if total_draft_tokens
                else 0.0
            ),
            # 每次forward平均产出的token数 = 1（目标模型自己的token）+ 平均接受的草稿token数
            "average_tokens_per_forward": (
                1 + total_accepted_tokens / total_steps if total_steps else 0.0
            ),
            "distribution": dict(sorted(distribution.items())),
            "distribution_percent": {
                k: v / total_steps if total_steps else 0.0
                for k, v in sorted(distribution.items())
            },
        }

    # 前缀缓存命中统计，跨DP rank聚合；调用方：api_server的/debug/cache_stats接口
    #   各计数器的定义见engine_stats.py的EngineStats（缓存部分细节见第4批）
    def get_cache_statistics(self, timeout: float = 30.0) -> dict[str, Any]:
        """Return aggregated prefix-cache statistics across DP ranks.

        The rates are the ones `[Cache Stats]` and `[Cache Pools]` log,
        recomputed from summed counters rather than averaged: ranks admit
        different numbers of tokens, so the mean of their rates is not the rate
        of their union.

        `cached <= wanted <= compressed <= reusable <= full` by construction,
        which is what makes the differences below meaningful:
          hit                 reuse actually admitted
          compressed_hit      reuse the prefix index held, before the
                              per-request state classes had their say
          lost_to_checkpoint  declined only because no checkpoint existed at
                              that boundary — what a denser ladder recovers
          lost_unrecoverable  declined for a reason no checkpoint touches

        `paged_hit` and `state_hit` split that into one number per pool, so a
        caller can tell which to fix; `hit` alone cannot, since the same value
        arises from a KV pool that lost the prefix and from a state cache that
        refused to resume from it. They multiply back to `hit` exactly.
        """
        # 【译】返回跨DP rank聚合后的前缀缓存统计。
        #
        #   这些比率与 [Cache Stats] 和 [Cache Pools] 日志里打印的相同，但是从求和后的计数器重新计算的，而不是取平均：
        #   各rank接收的token数不同，所以各rank比率的平均值并不是它们合起来的比率。
        #
        #   按构造满足 `cached <= wanted <= compressed <= reusable <= full`，正因如此下面这些差值才有意义：
        #     hit                 实际被接受的复用
        #     compressed_hit      前缀索引里持有的复用，在逐请求的状态类给出意见之前
        #     lost_to_checkpoint  只因该边界上没有checkpoint而被拒绝——更密的checkpoint阶梯能找回的部分
        #     lost_unrecoverable  因checkpoint无法影响的原因而被拒绝
        #
        #   paged_hit 和 state_hit 把它拆成每个池各一个数，调用方就能分辨该修哪一个；只看 hit 做不到，因为同样的值
        #   既可能来自丢了前缀的KV池，也可能来自拒绝从该前缀恢复的状态缓存。二者相乘恰好还原 hit。
        # 解释：返回的dict里实际只有state_hit，没有paged_hit这个键——paged池那一半就是compressed_hit（见下方英文注释）
        responses = self.core_mgr.broadcast_utility_command_sync(
            "get_cache_statistics", timeout=timeout
        )
        rank_stats = [
            resp.get("result", resp)
            for resp in responses
            if resp.get("result", resp).get("enabled", False)
        ]
        # 先把各rank的计数器逐项求和，再在总和上算比率
        totals = {
            key: sum(int(stats.get(key, 0)) for stats in rank_stats)
            for key in (
                "requests",
                "cached_tokens",
                "compressed_tokens",
                "wanted_tokens",
                "reusable_tokens",
                "full_tokens",
                # Reuse served from the CPU offload tier. Every rank reports it
                # in `cache_statistics()`, but it was missing from this sum, so
                # the DP-aggregated snapshot dropped the one counter the offload
                # feature exists to move. Shares the `reusable` denominator with
                # the HBM series but no numerator -- scored separately below.
                # 【译】由CPU offload层提供的复用。每个rank都在 cache_statistics() 里报告它，但这里的求和曾经漏掉了它，
                #   所以按DP聚合的快照丢掉了offload特性存在的意义所在的那个计数器。它与HBM系列共用 reusable 分母，
                #   但不共用分子——在下面单独计分。
                "offload_tokens",
                "checkpoints_kept",
                "checkpoints_dropped",
                "checkpoints_evicted",
                # Says the *paged* pool is too small, where `evicted` says the
                # state pool is -- opposite fixes, so it cannot be folded in.
                # 【译】它说明的是*分页*池太小，而 evicted 说明的是状态池太小——修法相反，所以不能合并。
                "checkpoints_orphaned",
                "demands_recorded",
                "demands_declined_no_room",
                "chunks_cut_for_demand",
                # Both cut counters or neither: their ratio is what separates a
                # placement that converges from one that pays per request, and
                # one of them missing makes the other unreadable.
                # 【译】两个cut计数器要么都有要么都没有：它们的比值区分了"会收敛的放置"和"每个请求都要付代价的放置"，
                #   缺了其中一个，另一个就无法解读。
                "chunks_cut_for_end",
                # Whether the joint state+KV load found a boundary to work
                # with, and what the state leg cost when it did. Summed like
                # the rest: a boundary is per admission and every rank admits
                # the same request.
                # 【译】联合的状态+KV加载是否找到了可用的边界，以及找到时状态那一段的代价。和其他计数器一样求和：
                #   边界是按每次准入算的，每个rank都准入同一个请求。
                "joint_boundaries",
                "state_hbm",
                "state_tier",
            )
        }
        # `reusable`, not `full`: a request's trailing block is never a reuse
        # candidate (prefill must forward one block for logits), so `full`
        # charges both pools for tokens neither was offered and caps every
        # rate below 100%. See `EngineStats.total_reusable_tokens`.
        # 【译】用 reusable 而不是 full：请求的最后一个block从来不是复用候选（prefill必须至少forward一个block
        #   才能得到logits），所以用 full 会让两个池都为"从未提供给它们的token"背锅，并把所有比率压在100%以下。
        #   见 EngineStats.total_reusable_tokens。
        reusable = totals["reusable_tokens"]

        # 分母为0时返回0.0，避免除零
        def rate(num: int, den: int = reusable) -> float:
            return num / den if den else 0.0

        compressed = totals["compressed_tokens"]
        return {
            "enabled": bool(rank_stats),
            **totals,
            "hit": rate(totals["cached_tokens"]),
            "compressed_hit": rate(compressed),
            "lost_to_checkpoint": rate(
                totals["wanted_tokens"] - totals["cached_tokens"]
            ),
            "lost_unrecoverable": rate(compressed - totals["wanted_tokens"]),
            # The state cache's own rate, scored against what the paged pool
            # actually handed it rather than against `reusable` -- otherwise a
            # KV eviction reads as a state-cache miss and points tuning at the
            # wrong pool. `compressed_hit` above is already the paged pool's
            # half of the same split and the two multiply back to `hit`, so a
            # `paged_hit` alias for it was a second name for one number.
            # See `EngineStats.paged_hit_rate` / `state_hit_rate`.
            # 【译】状态缓存自己的比率，分母是分页池实际交给它的量，而不是 reusable——否则一次KV淘汰会被算成状态缓存未命中，
            #   把调优引向错误的池。上面的 compressed_hit 已经是同一拆分中分页池那一半，二者相乘还原 hit，
            #   所以再为它起一个 paged_hit 别名，只是同一个数的第二个名字。
            #   见 EngineStats.paged_hit_rate / state_hit_rate。
            "state_hit": rate(totals["cached_tokens"], compressed),
            "state_recoverable_loss": rate(
                totals["wanted_tokens"] - totals["cached_tokens"], compressed
            ),
            # CPU-tier reuse against the same `reusable` denominator as `hit`,
            # matching `EngineStats.offload_hit_rate`. Disjoint from the HBM
            # `hit` numerator (the offload tier serves what HBM missed), so it
            # is not part of the `hit`/`compressed_hit` product and stands on
            # its own.
            # 【译】CPU层的复用，与 hit 使用相同的 reusable 分母，对应 EngineStats.offload_hit_rate。
            #   它与HBM的 hit 分子不相交（offload层提供的是HBM没命中的部分），所以不属于 hit/compressed_hit 的乘积，单独成立。
            # 解释：engine_stats.py里并没有offload_hit_rate，对应的属性名是EngineStats.lmcache_hit_rate
            "offload_hit": rate(totals["offload_tokens"]),
        }

    # 给Prometheus导出器的DP聚合快照（指标细节见第10批metrics/）；调用方：api_server的/metrics刷新
    def get_metrics_statistics(self) -> dict[str, Any]:
        """Return a DP-aggregated snapshot for the Prometheus exporter.

        Reads the snapshots each EngineCore pushes on its own clock, so this is
        a local dict lookup: no round trip, no deadline, and nothing that can
        fail just because the engine is busy.
        """
        # 【译】返回给Prometheus导出器的、按DP聚合的快照。
        #
        #   读取的是每个EngineCore按自己的节奏推送来的快照，所以这只是一次本地dict查询：没有往返，没有截止时间，
        #   也不会仅仅因为引擎忙就失败。
        # 解释：EngineCore的busy_loop每隔ATOM_METRICS_UPDATE_INTERVAL_S推送一次METRICS消息，
        #   CoreManager的输出线程把它存进latest_metrics[dp_rank]
        latest_metrics = self.core_mgr.latest_metrics.copy()
        rank_stats = [
            stats for stats in latest_metrics.values() if stats.get("enabled", False)
        ]

        def summed(key: str) -> int:
            return sum(int(stats.get(key, 0)) for stats in rank_stats)

        # Queue depths cannot be summed across a P/D pair: one in-flight
        # request sits in the prefill rank's `running` and the decode rank's
        # `prefill_waiting` at once. The decode side's four queues already
        # span the whole lifetime, so it alone is the full picture. Falls back
        # to every rank when no decode rank reported — non-P/D, or before the
        # decode engine's first push, where nothing is duplicated yet.
        # 【译】队列深度不能在一对P/D之间求和：一个在途请求同时位于prefill rank的 running 和decode rank的
        #   prefill_waiting 中。decode端的四个队列已经覆盖了整个生命周期，所以只看它就是全貌。没有decode rank
        #   上报时（非P/D，或者decode引擎第一次推送之前，此时还没有重复）退回到所有rank。
        queue_stats = [s for s in rank_stats if s.get("role") == "decode"] or rank_stats

        def summed_queues(key: str) -> int:
            return sum(int(stats.get(key, 0)) for stats in queue_stats)

        kv_total = summed("kv_blocks_total")
        kv_used = summed("kv_blocks_used")

        mtp_rank_stats = [
            stats.get("mtp", {})
            for stats in rank_stats
            if stats.get("mtp", {}).get("enabled", False)
        ]
        mtp_distribution: Counter[int] = Counter()
        for stats in mtp_rank_stats:
            mtp_distribution.update(
                {
                    int(accepted): int(steps)
                    for accepted, steps in stats.get("distribution", {}).items()
                }
            )
        mtp_draft = sum(
            int(stats.get("total_draft_tokens", 0)) for stats in mtp_rank_stats
        )
        mtp_accepted = sum(
            int(stats.get("total_accepted_tokens", 0)) for stats in mtp_rank_stats
        )
        mtp_steps = sum(mtp_distribution.values())

        cache_rank_stats = [
            stats.get("cache", {})
            for stats in rank_stats
            if stats.get("cache", {}).get("enabled", False)
        ]
        cache_keys = (
            "requests",
            "cached_tokens",
            "compressed_tokens",
            "wanted_tokens",
            "full_tokens",
            "checkpoints_kept",
            "checkpoints_dropped",
            "checkpoints_evicted",
            "checkpoints_orphaned",
            "demands_recorded",
            "demands_declined_no_room",
            "chunks_cut_for_demand",
            "chunks_cut_for_end",
        )
        cache_totals = {
            key: sum(int(stats.get(key, 0)) for stats in cache_rank_stats)
            for key in cache_keys
        }
        # Keep the admitted supplemental reuse population aligned with the HBM
        # and input counters. Missing older snapshots are not a zero tier hit.
        # 【译】让被接受的补充复用总体与HBM及输入计数器保持一致。缺少该字段的旧快照不算作"层命中为0"。
        if all("offload_tokens" in stats for stats in cache_rank_stats):
            cache_totals["offload_tokens"] = sum(
                int(stats["offload_tokens"]) for stats in cache_rank_stats
            )
        # NOTE: `full`, while `get_cache_statistics` divides by `reusable`, so
        # this endpoint reads lower for the same engine — `full` counts the
        # trailing block no cache is offered, and that fixed size weighs more
        # on shorter prompts. Pre-existing; don't compare the two endpoints.
        # 【译】注意：这里用 full，而 get_cache_statistics 除以 reusable，所以同一个引擎在这个接口上读数更低——
        #   full 计入了任何缓存都不会被提供的最后一个block，这个固定大小在短prompt上权重更大。
        #   这是历史遗留；不要比较这两个接口。
        cache_full = cache_totals["full_tokens"]
        offload_rank_stats = [
            stats.get("offload", {}) for stats in rank_stats if stats.get("offload")
        ]
        offload_keys = (
            "load_requests",
            "loaded_tokens",
            "load_failures",
            "save_requests",
            "saved_tokens",
            "loads_pending",
            "saves_pending",
            # Why a waiting request is not costing a tier lookup per step, and
            # whether its one external-tier attempt was already spent. Both are
            # what an operator reads when the engine looks stalled with the
            # GPUs idle; see `OffloadSchedulerMixin._init_tier_hit_memo`.
            # 【译】为什么一个等待中的请求不会每步都做一次层查找，以及它唯一一次外部层尝试是否已经用掉。
            #   当引擎看起来卡住而GPU空闲时，运维要读的正是这两个；见 OffloadSchedulerMixin._init_tier_hit_memo。
            "lookups_skipped_by_memo",
            "suppressed_load_retries",
        )
        offload_totals = {
            key: sum(int(stats.get(key, 0)) for stats in offload_rank_stats)
            for key in offload_keys
        }

        return {
            "enabled": bool(rank_stats),
            "requests_running": summed_queues("requests_running"),
            "requests_waiting": summed_queues("requests_waiting"),
            "requests_parked_kv_load": summed("requests_parked_kv_load"),
            "requests_partial_prefill": summed("requests_partial_prefill"),
            "requests_finished": summed("requests_finished"),
            "prompt_tokens": summed("prompt_tokens"),
            "generation_tokens": summed("generation_tokens"),
            "preemptions": summed("preemptions"),
            "kv_blocks_used": kv_used,
            "kv_blocks_free": summed("kv_blocks_free"),
            "kv_blocks_total": kv_total,
            "kv_blocks_indexed": summed("kv_blocks_indexed"),
            "kv_cache_usage_ratio": kv_used / kv_total if kv_total else 0.0,
            # 每个DP rank一项的调度器指标（只有上报了scheduler_metrics的rank）
            "scheduler_metrics": [
                {
                    "dp_rank": rank,
                    "engine_role": stats.get("role") or "default",
                    **stats["scheduler_metrics"],
                    "kv_blocks": (
                        {
                            "used": stats["kv_blocks_used"],
                            "evictable": stats["kv_blocks_evictable"],
                            "vacant": stats["kv_blocks_vacant"],
                            "total": stats["kv_blocks_total"],
                        }
                        if "kv_blocks_total" in stats
                        else {}
                    ),
                }
                for rank, stats in latest_metrics.items()
                if stats.get("enabled") and "scheduler_metrics" in stats
            ],
            "mtp": {
                "enabled": bool(mtp_rank_stats),
                "total_draft_tokens": mtp_draft,
                "total_accepted_tokens": mtp_accepted,
                "acceptance_rate": mtp_accepted / mtp_draft if mtp_draft else 0.0,
                "average_tokens_per_forward": (
                    1 + mtp_accepted / mtp_steps if mtp_steps else 0.0
                ),
                "distribution": dict(sorted(mtp_distribution.items())),
            },
            "cache": {
                "enabled": bool(cache_rank_stats),
                **cache_totals,
                "hit": (
                    cache_totals["cached_tokens"] / cache_full if cache_full else 0.0
                ),
                "compressed_hit": (
                    cache_totals["compressed_tokens"] / cache_full
                    if cache_full
                    else 0.0
                ),
                "lost_to_checkpoint": (
                    (cache_totals["wanted_tokens"] - cache_totals["cached_tokens"])
                    / cache_full
                    if cache_full
                    else 0.0
                ),
                "lost_unrecoverable": (
                    (cache_totals["compressed_tokens"] - cache_totals["wanted_tokens"])
                    / cache_full
                    if cache_full
                    else 0.0
                ),
            },
            "offload": offload_totals,
            # DP路由器（CoreManager里的负载均衡）自己的统计
            "dp_router": self.core_mgr.get_dp_router_statistics(),
        }


# =====================================================================================
# InputOutputProcessor：本进程里的"文本 <-> Sequence"转换器
# =====================================================================================
# 职责：
#   preprocess / preprocess_fanout   prompt → token id → Sequence（stop字符串也在这里分词）
#   postprocess                      已结束的Sequence → 文本 + 延迟指标（TTFT/TPOT）
#   requests                         记录"已经发出、还没被postprocess取走"的请求，generate据此判断是否结束
# 调用方：LLMEngine.add_request / generate（离线）；api_server、atomesh服务直接调用preprocess*（在线）
#   在线路径不调用postprocess，而是自己从requests里pop（见api_server.py）
# 对比nano-vllm：nano-vllm在LLMEngine.add_request里直接tokenizer.encode + Sequence(...)，
#   generate末尾直接tokenizer.decode；ATOM把它们拆成一个类，因为在线服务也要复用同一套逻辑，
#   而且Sequence的构造参数多得多（投机解码、状态缓存、多模态、PD分离、扇出、DP路由……）
class InputOutputProcessor:
    # Class default: a processor built without `__init__` ships nothing.
    # 【译】类级默认值：一个没有经过 __init__ 构造的处理器不发送任何东西。
    # 解释：这里的"发送"指随请求一起把prompt的前缀hash链发给EngineCore（见下面__init__和preprocess_fanout）
    prefix_hash_block_size = 0

    # 输入：config、tokenizer、block_size（即config.kv_cache_block_size，已按模型改写过，见第1批）
    def __init__(self, config, tokenizer, block_size):
        # 成员变量逐个说明：
        #   config / tokenizer / block_size  见上
        #   requests                 seq.id -> Sequence：本进程发出、还没被取走的请求
        #   _external_to_internal    用户请求id -> seq.id；_internal_to_external 反过来（只在n=1且带了request_id时登记）
        #   has_per_req_cache        模型是否需要分页KV池之外的逐请求状态缓存（GDN/线性注意力、DeepSeek-V4等），见第4批
        #   num_speculative_tokens   投机解码每步的草稿token数（0表示未开启），见第7批
        #   prefix_hash_block_size   前缀缓存hash块的token数（0表示不在前端算hash链）
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = block_size
        # 在途请求表：seq.id -> Sequence
        self.requests = {}
        # `has_per_req_cache` flags model architectures that need a
        # per-request stateful buffer outside the paged KV pool. Sequences
        # constructed for these models trigger BlockManager to reserve a
        # per-req cache slot. Currently: GDN-based models (Qwen3-Next /
        # Qwen3.5). Future stateful models (DeepseekV4, etc.) extend the set.
        # 【译】has_per_req_cache 标记那些需要在分页KV池之外有一块逐请求有状态缓冲区的模型架构。为这些模型构造的
        #   Sequence会触发BlockManager预留一个逐请求的缓存槽位。目前是：基于GDN的模型（Qwen3-Next / Qwen3.5）。
        #   未来的有状态模型（DeepseekV4等）会扩充这个集合。
        # 解释：这段注释说的是下面第三行的has_per_req_cache，中间插进了两张id映射表；
        #   另外DeepSeek-V4等已经加进了_per_req_cache_model_types的集合里
        self._external_to_internal: dict[str, int] = {}
        self._internal_to_external: dict[int, str] = {}
        self.has_per_req_cache = False
        self.num_speculative_tokens = 0
        # 【分支】投机解码：每条序列要按草稿token数预留KV/占位，见第7批
        if (
            hasattr(self.config, "speculative_config")
            and self.config.speculative_config is not None
        ):
            self.num_speculative_tokens = (
                self.config.speculative_config.num_speculative_tokens
            )
        # model_type在集合里就标记为需要逐请求状态缓存（集合定义见下面_per_req_cache_model_types）
        if self.config.hf_config.model_type in self._per_req_cache_model_types():
            self.has_per_req_cache = True
        # `BlockManager.hash_block_size`, for shipping the prompt hash chain;
        # 0 = no prefix cache. The engine rehashes on a mismatch.
        # 【译】BlockManager.hash_block_size，用于发送prompt的hash链；0 = 没有前缀缓存。不一致时引擎会重新计算hash。
        # 解释：在前端（本进程）提前算好prompt的链式hash，随ADD消息一起发给EngineCore，省掉调度线程上的计算；
        #   hash块大小 = block_size x DCP大小（prefix_hash.hash_block_size_for），前缀缓存细节见第4批
        self.prefix_hash_block_size = 0
        if getattr(config, "enable_prefix_caching", False):
            self.prefix_hash_block_size = hash_block_size_for(
                block_size, config.decode_context_parallel_size
            )

    # 哪些model_type需要逐请求状态缓存（唯一的数据源）
    # @staticmethod：不需要self/cls的方法，可以通过类名直接调用（ModelRunner启动检查就是这样调用的）
    @staticmethod
    def _per_req_cache_model_types() -> frozenset[str]:
        """Single source of truth for which model_types use per-req cache.

        Read by Sequence-construction (here) AND by ModelRunner's startup
        sanity check, which asserts that any model whose attention builder
        declares a per-request state sub-pool has its model_type
        registered here. Adding a new stateful-attention model means
        adding its model_type to this set.
        """
        # 【译】哪些model_type使用逐请求缓存的唯一数据源。
        #
        #   由（这里的）Sequence构造读取，也由ModelRunner的启动自检读取：该自检断言，凡是注意力builder声明了逐请求状态
        #   子池的模型，其model_type都已在这里登记。新增一个有状态注意力的模型，就要把它的model_type加进这个集合。
        # frozenset：不可变集合，可以安全地作为"常量"返回
        return frozenset(
            {
                "qwen3_next",
                "qwen3_5_text",
                "qwen3_5_moe_text",
                "kimi_linear",
                "glm5_next_text",
                "deepseek_v4",
                "deepseek_v41_text",
                "qwen4_exp_text",
            }
        )

    # 单序列入口：把一个prompt变成一个Sequence
    # 输入：prompt（文本或token id）、采样参数、可选的流式回调、PD分离的kv_transfer_params、多模态数据、
    #   用户请求id、DP路由提示（data_parallel_rank / dp_session_id / dp_parent_session_id，见engine_core_mgr.py）
    # 输出：一个Sequence（已登记到self.requests）
    # 调用方：api_server、atomesh服务（在线）；内部直接复用preprocess_fanout
    def preprocess(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        kv_transfer_params=None,
        multimodal_data=None,
        request_id: str | None = None,
        data_parallel_rank: int | None = None,
        dp_session_id: str | None = None,
        dp_parent_session_id: str | None = None,
    ):
        """responsible for:
        1) Tokenize
        2) Create Sequence object

        Single-sequence entry point. Rejects ``sampling_params.n > 1`` so that
        callers which expect exactly one ``Sequence`` back cannot silently
        drop the other siblings. Use :meth:`preprocess_fanout` for n > 1.
        """
        # 【译】负责：
        #   1) 分词（Tokenize）
        #   2) 创建Sequence对象
        #
        #   单序列入口。拒绝 sampling_params.n > 1，这样期望只拿回一个 Sequence 的调用方不会悄悄丢掉其他兄弟序列。
        #   n > 1 时请使用 preprocess_fanout。
        # getattr(..., "n", 1)：兼容没有n字段的采样参数对象
        if getattr(sampling_params, "n", 1) > 1:
            raise ValueError(
                "preprocess() returns a single Sequence; for SamplingParams.n > 1 "
                "call preprocess_fanout() and manage the returned list."
            )
        # n == 1：preprocess_fanout返回长度为1的列表，取第0个
        seqs = self.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callback=stream_callback,
            kv_transfer_params=kv_transfer_params,
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
            dp_session_id=dp_session_id,
            dp_parent_session_id=dp_parent_session_id,
        )
        return seqs[0]

    # 扇出入口：tokenize一次，创建n个兄弟Sequence（n=1时就是普通的单序列路径）
    # 输入：同preprocess，另加stream_callbacks（每个兄弟一个回调，长度必须等于n）、parent_request_id
    # 输出：长度为n的Sequence列表（都已登记到self.requests）
    # 调用方：LLMEngine.add_request；api_server的n>1请求
    def preprocess_fanout(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        stream_callbacks: list | None = None,
        kv_transfer_params=None,
        multimodal_data=None,
        parent_request_id: str | None = None,
        data_parallel_rank: int | None = None,
        dp_session_id: str | None = None,
        dp_parent_session_id: str | None = None,
    ) -> list[Sequence]:
        """Tokenize once and materialize ``sampling_params.n`` Sequences.

        Returns a list of length ``n``. For ``n == 1`` this is functionally
        equivalent to the legacy single-sequence path. For ``n > 1``:

        * The prompt is tokenized a single time and the token list is copied
          into each sibling (``Sequence`` copies internally, so mutations stay
          isolated).
        * Every sibling is marked ``needs_independent_noise=True`` so the
          sampler generates fresh per-row noise instead of reusing the cached
          shared exponential tensor. Without this, siblings with identical
          logits would emit identical tokens.
        * Per-sibling ``stream_callbacks`` can be supplied to route streaming
          deltas to independent queues (one per choice index). Falls back to
          the scalar ``stream_callback`` for every sibling.
        """
        # 【译】tokenize一次，生成 sampling_params.n 个Sequence。
        #
        #   返回长度为 n 的列表。n == 1 时，功能上等同于旧的单序列路径。n > 1 时：
        #
        #   * prompt只tokenize一次，token列表被复制进每个兄弟（Sequence 内部会复制，所以各自的修改互不影响）。
        #   * 每个兄弟都标记 needs_independent_noise=True，让采样器为每行生成新的噪声，而不是复用缓存的共享指数分布张量。
        #     不这样做的话，logits完全相同的兄弟会输出完全相同的token。
        #   * 可以提供逐兄弟的 stream_callbacks，把流式增量送到各自独立的队列（每个choice index一个）。
        #     没有提供时，每个兄弟都退回使用标量 stream_callback。
        # 处理步骤：
        #   1. 模型级请求校验（可选）  2. tokenize  3. 多模态的M-RoPE位置  4. stop字符串分词
        #   5. 创建n个Sequence并登记  6. 前端计算prompt的hash链  7. 打日志
        # ---------- 1. 模型级请求校验 ----------
        # 只有少数模型的hf_config定义了validate_request（目前是DeepSeek-V4.1：DP-attention或DSpark下不接受多模态请求）
        request_validator = getattr(self.config.hf_config, "validate_request", None)
        if request_validator is not None:
            request_validator(
                num_draft_tokens=self.num_speculative_tokens,
                multimodal_data=multimodal_data,
                enable_dp_attention=self.config.enable_dp_attention,
            )
        # 扇出数量，至少为1
        n = max(1, int(getattr(sampling_params, "n", 1)))

        # ---------- 2. 分词（tokenize） ----------
        # 字符串才编码；已经是token id列表的原样使用。encode默认会加special token（如BOS），由tokenizer配置决定
        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )
        # ---------- 3. 多模态：M-RoPE位置 ----------
        # 【分支】多模态模型（目前是Qwen3.5、Qwen4-exp的多模态版）用M-RoPE：位置分时间/高/宽三维，
        #   图像token的位置不是简单的0,1,2...；这里按模型的接口预先算好整段位置和偏移delta，见第2批multimodal/
        mrope_positions = None
        mrope_position_delta = 0
        if multimodal_data is not None:
            mrope_positions, mrope_position_delta = get_mrope_input_positions(
                self.config,
                tokens,
                multimodal_data,
            )

        # ---------- 4. stop字符串 -> stop token序列 ----------
        # 调度器判停只比较token，所以要把stop字符串预先分词成token序列，判停时检查token_ids的末尾是否等于其中之一
        #   例：stop_strings=["\n\nUser:"] → 得到类似[[271, 1474, 25]]的token序列（具体id取决于tokenizer）
        # 注意这是token级匹配：同一段文本在生成时若被切成不同的token（例如与前一个字符合并），就匹配不上
        #   例：stop="\n\n"单独分词可能是一个token X；但模型生成时若吐出的是".\n\n"合并成的另一个token，
        #   token序列里就不会出现X，这次停不下来（直到eos或max_tokens）
        # 为什么不在EngineCore里按文本匹配：EngineCore没有tokenizer，按文本匹配就得在调度热循环里每步detokenize每条序列，
        #   正是上面"放到前端"想避免的CPU开销；token级匹配只需比较token_ids末尾的几个整数
        # 三种"停止"的区别（判停都在EngineCore的Scheduler.postprocess里）：
        #   eos_token_id        tokenizer的结束符，finish_reason="eos"；ignore_eos=True时这一项和下一项都不检查
        #   stop_token_ids      generation_config里的其他结束token（例如某些对话模型的轮次结束符），finish_reason="stop_<id>"
        #   stop_token_sequences 用户的stop字符串分词后的结果，finish_reason="stop_sequence"；匹配到的这段token保留在输出里
        stop_token_sequences = []
        if sampling_params.stop_strings:
            # 允许传单个字符串，统一成列表
            stops = (
                [sampling_params.stop_strings]
                if isinstance(sampling_params.stop_strings, str)
                else sampling_params.stop_strings
            )
            for stop_str in stops:
                # add_special_tokens=False：不要在stop序列前面加BOS之类的special token，否则永远匹配不上
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                # 空字符串分词后为空列表，跳过
                if stop_tokens:
                    stop_token_sequences.append(stop_tokens)

        if stream_callbacks is not None and len(stream_callbacks) != n:
            raise ValueError(
                f"stream_callbacks length {len(stream_callbacks)} does not match n={n}"
            )

        # ---------- 5. 创建n个兄弟Sequence ----------
        # n>1（parallel sampling，并行采样：同一个prompt采样出n个不同回答）时，兄弟之间共享的与不共享的：
        #   共享：tokenize结果、stop token序列、多模态数据与M-RoPE位置、前端算好的prompt hash链（见第6步）
        #   不共享：seq id、KV block（每个兄弟各自分配；prompt部分的KV能否复用只取决于前缀缓存是否开启并命中，
        #     引擎不做"fork一份block表"式的专门共享）、DP rank（见add_request处）
        #   从这里起每个兄弟就是一个普通请求，调度、判停、结束都各管各的
        # 对比nano-vllm：nano-vllm的SamplingParams没有n，要多个回答只能把同一个prompt提交多次
        seqs: list[Sequence] = []
        for i in range(n):
            # 每个兄弟用自己的回调（若给了stream_callbacks），否则共用stream_callback
            cb = (
                stream_callbacks[i] if stream_callbacks is not None else stream_callback
            )
            # Sequence的各参数含义见第1批sequence.py；这里传入的tokens会在Sequence内部复制成array("i")
            seq = Sequence(
                tokens,
                self.block_size,
                sampling_params,
                stop_token_sequences,
                stream_callback=cb,
                num_draft_tokens=self.num_speculative_tokens,
                has_per_req_cache=self.has_per_req_cache,
                kv_transfer_params=kv_transfer_params,
                multimodal_data=multimodal_data,
                mrope_positions=mrope_positions,
                mrope_position_delta=mrope_position_delta,
                # n>1时需要独立噪声，否则兄弟序列采样结果完全相同（见上面docstring的翻译）
                #   原理：采样器用Gumbel-max的等价形式 token = argmax_i(p_i / E_i)，E_i ~ Exp(1)，这与按概率p采样等价。
                #   默认为了省时只缓存一行(1, vocab)的噪声并广播给整批；兄弟序列prompt相同，第一步logits完全相同，
                #   共用同一行噪声就会选出同一个token，之后输入也一样，于是n个回答一字不差。
                #   needs_independent_noise=True时采样器为每行重新生成(num_tokens, vocab)的噪声（见model_ops/sampler.py）
                needs_independent_noise=(n > 1),
                parent_request_id=parent_request_id,
                sibling_index=i,
                # 只有n == 1时才把用户请求id设为序列的external_request_id；n>1时兄弟共用parent_request_id
                request_id=parent_request_id if n == 1 else None,
                data_parallel_rank=data_parallel_rank,
                dp_session_id=dp_session_id,
                dp_parent_session_id=dp_parent_session_id,
            )
            # 到达时间（墙钟）。first_token_time由EngineCore进程里的调度器记录，两者跨进程比较，所以都用time.time()
            #   注意测量边界：arrive_time在tokenize之后才记，所以这里的TTFT不含tokenize和HTTP处理；
            #   first_token_time在EngineCore的postprocess里记，不含结果传回前端的时间。
            #   若EngineCore在另一台机器上（多机DP），两台机器的时钟偏差会直接计入TTFT
            seq.arrive_time = time.time()
            # 登记为"在途请求"，postprocess时删除
            self.requests[seq.id] = seq
            # 记录用户请求id <-> seq.id的双向映射
            if seq.external_request_id is not None:
                self._external_to_internal[seq.external_request_id] = seq.id
                self._internal_to_external[seq.id] = seq.external_request_id
            seqs.append(seq)

        # ---------- 6. 前端计算prompt的hash链（前缀缓存，见第4批） ----------
        # Runs in an executor or submit thread, not the event loop. Siblings
        # share a prompt and seed, so one chain serves them all.
        # 【译】运行在executor或提交线程里，而不是事件循环里。兄弟序列共享同一个prompt和seed，所以一条链服务所有兄弟。
        # 解释：ship_prompt_hash_chain只有在ATOM_FRONTEND_PREFIX_HASH开启时才真正计算；
        #   链随ADD消息单独发送（CoreManager._add_frame），EngineCore的BlockManager直接采用，不一致时重算
        if self.prefix_hash_block_size:
            ship_prompt_hash_chain(seqs[0], self.prefix_hash_block_size)
            for seq in seqs[1:]:
                seq.prompt_hash_chain = seqs[0].prompt_hash_chain

        # ---------- 7. 日志 ----------
        if n == 1:
            logger.info(
                f"Request {seqs[0].id} arrived, input tokens: {len(tokens)}, "
                f"pending requests: {len(self.requests)}"
            )
        else:
            logger.info(
                f"Request {parent_request_id or seqs[0].id} fanned out into "
                f"{n} siblings ({seqs[0].id}..{seqs[-1].id}), "
                f"input tokens: {len(tokens)}, "
                f"pending requests: {len(self.requests)}"
            )
        return seqs

    # 后处理：把一批已结束的Sequence变成用户可读的输出
    # 输入：step()返回的已结束Sequence列表（从EngineCore进程pickle回来的副本）
    # 输出：{seq_id: 输出dict}
    # 调用方：LLMEngine.generate / generate_multimodal、rollout/async_engine.py（在线API不调用它）
    def postprocess(self, reqs: list[Sequence]):
        """responsible for:
        1) Compute stats for logging
        2) Detokenize"""
        # 【译】负责：
        #   1) 计算用于日志的统计数据
        #   2) 反分词（Detokenize）
        outputs = {}
        for req in reqs:
            # 从在途请求表中移除（不在表中会抛KeyError），同时清理用户请求id映射
            self.requests.pop(req.id)
            external_request_id = self._internal_to_external.pop(req.id, None)
            if external_request_id is not None:
                self._external_to_internal.pop(external_request_id, None)
            # detokenize：只解码生成的部分（completion_token_ids = token_ids[num_prompt_tokens:]）
            #   离线路径在结束时对整段一次性decode，所以没有增量反分词的麻烦。流式（在线）路径要边生成边吐文本，
            #   难点在于token与字符不是一一对应：byte-level BPE里一个汉字是3个UTF-8字节，一个emoji是4个，
            #   常被拆进多个token，单独decode其中一个只会得到"\ufffd"（替换字符）；合并规则也会让"前文+新token"的解码结果
            #   不等于两者分别解码再拼接。所以流式解码要保留一个滑动窗口，用"窗口整体解码 - 窗口前半解码"得出新增文本，
            #   末尾是"\ufffd"就先扣住不发——见entrypoints/openai/streaming_dispatch.py的IncrementalStreamDetokenizer
            #   另外这里没有传skip_special_tokens，默认会把eos等特殊token的文本（如"<|im_end|>"）也解码进text；
            #   流式路径则用skip_special_tokens=True
            output_str = self.tokenizer.decode(req.completion_token_ids)
            # 离开时间：在本进程记录，所以latency包含了结果从EngineCore传回来的时间
            req.leave_time = time.time()

            # Calculate TTFT (Time To First Token) and TPOT (Time Per Output Token)
            # 【译】计算TTFT（首token延迟）和TPOT（每个输出token的耗时）
            # 例：arrive=100.00s，first_token=100.25s，leave=102.25s，生成21个token
            #   → TTFT = 0.25s，TPOT = (102.25 - 100.25) / (21 - 1) = 0.1s
            ttft = 0.0
            tpot = 0.0
            # first_token_time为0说明一个token都没生成（例如被调度器拒绝或中止），两个指标都记0
            if req.first_token_time > 0:
                ttft = req.first_token_time - req.arrive_time
                # Calculate TPOT only if there are multiple output tokens
                # 【译】只有多于一个输出token时才计算TPOT
                # 第一个token的时间算在TTFT里，所以TPOT的分母是"生成token数 - 1"
                if req.num_completion_tokens > 1:
                    tpot = (req.leave_time - req.first_token_time) / (
                        req.num_completion_tokens - 1
                    )

            logger.info(
                f"Request {req.id} finished with reason {req.leave_reason}. "
                f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
                f"latency: {req.leave_time - req.arrive_time:.2f}s, "
                f"TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s"
            )
            outputs[req.id] = {
                "text": output_str,
                # `list`, not the `array("i")` slice: this is what `generate()`
                # hands a caller, and the storage type is ours to change.
                # 【译】用 list，而不是 array("i") 切片：这是 generate() 交给调用方的东西，而存储类型是我们自己可以改的。
                "token_ids": list(req.completion_token_ids),
                # `list` for the same reason as `token_ids` above.
                # 【译】用 list 的原因与上面的 token_ids 相同。
                "logprobs": list(req.logprobs) if req.return_logprobs else None,
                "latency": req.leave_time - req.arrive_time,
                "finish_reason": req.leave_reason,
                "num_tokens_input": req.num_prompt_tokens,
                "num_tokens_output": req.num_completion_tokens,
                # 【译】（行尾注释）首token延迟，单位为秒
                "ttft": ttft,  # Time to first token in seconds
                # 【译】（行尾注释）每个输出token的耗时，单位为秒
                "tpot": tpot,  # Time per output token in seconds
            }
        return outputs

    # 是否还有在途请求（generate的循环条件）
    def has_pending_requests(self):
        return len(self.requests) > 0
