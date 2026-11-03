"""
PrefillDelayer — local TP or cross-DP prefill coalescing for ATOM.

Purpose
-------
Under DP-attention each rank schedules independently. Left alone, ranks fire
many prefill forwards that each carry only a handful of tokens (a short fresh
prompt, or the small tail chunk of a chunked prefill). Every prefill forward
has ~fixed cost (kernel launch, pad-to-shape, the lockstep MoE all-to-all), so a
forward carrying 500 of a 16384-token budget wastes ~97% of that forward.

The delayer has two related jobs: **hold back prefill admission until the
accumulated prefill is worth a forward, then release** — Nagle's algorithm for
prefill — and optionally protect a fixed number of decode passes after every
prefill. While it holds, decode keeps running. Coalescing hold episodes are
bounded; the decode interval, backlog, and execution add to observed TTFT.

Single-rank / TP-only mode
--------------------------
With ``is_local`` (``dp_size=1`` and ``cpu_group=None``) the delayer runs the same coalescer
locally and skips the cross-rank ``all_reduce`` — a single scheduler drives all
TP workers, so there is no cross-rank phase to align. Only the batching value
remains (fill / stall / ttft / kv bounds decide FIRE/HOLD from this rank's own
counts); the ``n_prefillable < dp_size`` alignment gate is vacuous.
EngineCore opts TP/DCP into this mode when DP=1, PP=1, the master switch is
on, and ``ATOM_PREFILL_DECODE_INTERVAL > 0``. During local decode protection,
``protects_decode`` lets the scheduler skip cache probes and queue-age scans.
The interval precedes all coalescing bounds, including ``max_queue_ms``; those
bounds do not impose a hard end-to-end TTFT limit.

It is NOT about mixing prefill+decode in one forward. It DOES preserve cross-DP
phase alignment: it only releases when every rank is prefill-ready (so all ranks
enter prefill together and the MoE collective stays aligned), except when a
must-fire bound forces release regardless.

Decision (evaluated every tick, on every rank, in lockstep)
-----------------------------------------------------------
Each rank reports local state; a single cpu ``all_reduce(SUM)`` reduces it; then
every rank computes the SAME FIRE/HOLD from the reduced values:

  n_prefillable   = #ranks with admittable prefill (SUM of a 0/1 flag)
  G_pending       = total pending prefill tokens across ranks (fresh + partial
                    remaining, each rank capped at the token budget)
  G_running_dec   = total decode seqs across ranks
  any_kv_high/low = any prefillable rank at/above / below a KV watermark
  any_partial     = any rank mid-chunked-prefill
  any_prefill_ran = any rank completed a prefill forward on the previous tick

  if any_prefill_ran:                              arm decode interval
  if post-prefill decode interval is active and decode exists: HOLD
  if n_prefillable == 0:                          FIRE   # nothing to do (vacuous)
  # -- must-fire bounds (release even if unaligned / underfilled) --
  if G_running_dec == 0:                          FIRE   # no decode to hide the wait behind
  if any_kv_high or any_kv_low:                   FIRE   # KV pressure / starvation
  if any_queue_hot:                             FIRE   # queue age bound
  if hold_ticks >= ttft_max_ticks:                FIRE   # TTFT bound
  if any_partial and hold_ticks >= partial_max_ticks: FIRE  # partial holds KV — tight bound
  # -- alignment gate: never fire while some rank lacks prefill (anti-skew) --
  if n_prefillable < dp_size:                     HOLD
  # -- goal: fire once the aggregate fills a worthwhile forward --
  fill = G_pending / (n_prefillable * budget)
  if fill >= target_fill:                         FIRE
  # -- adaptive give-up: queue stopped growing, waiting longer is futile --
  if G_pending <= prev_G_pending:  stall += 1  else  stall = 0
  if stall >= stall_ticks:                        FIRE
  HOLD

Why SUM (not MAX)
-----------------
A MAX ("fire as soon as the busiest rank is ready") drags quiet ranks into
firing tiny prefills. A SUM/aggregate-fill target holds until the forward is
collectively worthwhile, so quiet ranks keep accumulating during the hold and
fire larger when release finally happens. Under-balanced load a quiet rank still
fires smaller — that is a request-routing problem, out of scope here.

Cross-DP comms
--------------
Single ``all_reduce(SUM)`` over an 8-int64 cpu buffer (gloo-safe). Booleans are
encoded as 0/1 and read back as ``sum > 0`` (logical OR). All timing is
tick-based (``hold_ticks``), which is deterministic across ranks — no per-rank
wall-clock, so ranks never diverge on a timeout boundary. Fail-open on a
collective error (a peer dying should not crash schedule()); the guarded
DP-state barrier drives shutdown.
"""

# 【译】PrefillDelayer —— ATOM 的本地 TP 或跨 DP 的 prefill 合并（coalescing）。
#
#   目的
#   ----
#   在 DP-attention 下，每个 rank 独立调度。放任不管的话，各 rank 会发出许多 prefill forward，每次只带寥寥几个 token
#   （一条很短的新 prompt，或 chunked prefill 最后那个小尾巴 chunk）。每次 prefill forward 都有大致固定的开销
#   （kernel 发射、按形状补齐、锁步的 MoE all-to-all），所以在 16384 token 的预算里只带 500 个 token 的一次 forward，
#   浪费了这次 forward 的约 97%。
#
#   delayer 有两项相关的工作：**扣住 prefill 准入，直到积累的 prefill 值得做一次 forward，再放行**——相当于 prefill 版的
#   Nagle 算法——以及（可选地）在每次 prefill 之后保护固定次数的 decode 轮次。扣住期间 decode 照常运行。
#   合并造成的扣住期（hold episode）是有界的；decode 间隔、积压和执行时间都会计入观测到的 TTFT。
#
#   单 rank / 仅 TP 模式
#   --------------------
#   ``is_local`` 时（``dp_size=1`` 且 ``cpu_group=None``），delayer 在本地运行同一个合并器，并跳过跨 rank 的 ``all_reduce``
#   ——一个调度器驱动所有 TP worker，不存在需要对齐的跨 rank 阶段。只剩下合批的价值（fill / stall / ttft / kv 这些上界
#   根据本 rank 自己的计数决定 FIRE/HOLD）；``n_prefillable < dp_size`` 这道对齐门恒不成立。
#   EngineCore 在 DP=1、PP=1、总开关打开且 ``ATOM_PREFILL_DECODE_INTERVAL > 0`` 时让 TP/DCP 进入这种模式。
#   本地 decode 保护期间，``protects_decode`` 让调度器跳过 cache 探测和排队时长扫描。
#   这个间隔优先于所有合并上界，包括 ``max_queue_ms``；这些上界并不构成端到端 TTFT 的硬上限。
#
#   它与"在同一次 forward 里混合 prefill+decode"无关。它*确实*会保持跨 DP 的阶段对齐：只有当每个 rank 都准备好 prefill
#   时才放行（这样所有 rank 一起进入 prefill，MoE 集合通信保持对齐），除非某个"必须放行"的上界无论如何都强制放行。
#
#   决策（每个 tick、在每个 rank 上、锁步地评估）
#   ---------------------------------------------
#   每个 rank 上报本地状态；一次 CPU 上的 ``all_reduce(SUM)`` 把它们归约；然后每个 rank 根据归约后的值算出*同一个* FIRE/HOLD：
#
#     n_prefillable   = 有可准入 prefill 的 rank 数（0/1 标志求和）
#     G_pending       = 所有 rank 的待 prefill token 总数（新请求 + 部分完成请求的剩余，每个 rank 封顶为 token 预算）
#     G_running_dec   = 所有 rank 的 decode 序列总数
#     any_kv_high/low = 是否有可 prefill 的 rank 处于/高于、低于某个 KV 水位
#     any_partial     = 是否有 rank 正处于 chunked prefill 中途
#     any_prefill_ran = 是否有 rank 在上一个 tick 完成了一次 prefill forward
#
#     若 any_prefill_ran：                                装填 decode 间隔
#     若 prefill 之后的 decode 间隔仍在生效且存在 decode：   HOLD
#     若 n_prefillable == 0：                             FIRE   （无事可做，空放行）
#     -- 必须放行的上界（即使未对齐 / 未填满也放行）--
#     若 G_running_dec == 0：                             FIRE   （没有 decode 可以把这段等待"藏"在后面）
#     若 any_kv_high 或 any_kv_low：                      FIRE   （KV 压力 / 饥饿）
#     若 any_queue_hot：                                  FIRE   （排队时长上界）
#     若 hold_ticks >= ttft_max_ticks：                   FIRE   （TTFT 上界）
#     若 any_partial 且 hold_ticks >= partial_max_ticks：  FIRE   （部分完成的请求占着 KV —— 更紧的上界）
#     -- 对齐门：只要还有 rank 没有 prefill 就不放行（防止错位）--
#     若 n_prefillable < dp_size：                        HOLD
#     -- 目标：聚合量填满一次值得做的 forward 就放行 --
#     fill = G_pending / (n_prefillable * budget)
#     若 fill >= target_fill：                            FIRE
#     -- 自适应放弃：队列不再增长，再等也没用 --
#     若 G_pending <= prev_G_pending：stall += 1，否则 stall = 0
#     若 stall >= stall_ticks：                           FIRE
#     其余情况：                                          HOLD
#
#   为什么用 SUM（而不是 MAX）
#   --------------------------
#   MAX（"最忙的 rank 一就绪就放行"）会把清闲的 rank 拖进去发出很小的 prefill。SUM / 聚合填充率目标会一直扣住，
#   直到这次 forward 在整体上值得做，于是清闲的 rank 在扣住期间继续积累，最终放行时发出更大的 prefill。
#   负载不均衡时，清闲的 rank 仍然会发出较小的 prefill——那是请求路由的问题，不在这里的处理范围内。
#
#   跨 DP 通信
#   ----------
#   对一个 8 个 int64 的 CPU 缓冲区做一次 ``all_reduce(SUM)``（gloo 可用）。布尔值编码为 0/1，读回时用 ``sum > 0``（逻辑或）。
#   所有计时都基于 tick（``hold_ticks``），它在各 rank 之间是确定的——没有各 rank 自己的墙钟时间，所以各 rank 永远不会在
#   超时边界上产生分歧。集合通信出错时 fail-open（直接放行）（一个 peer 挂掉不应让 schedule() 崩溃）；
#   由受保护的 DP 状态屏障来驱动关闭。
#
# 解释（对上面几处说法的补充）：
#   - "8 个 int64"：开了 exchange_prefill_costs（DP prefill 代价均衡）时是 8 + dp_size 个，见 __init__ 里的槽位表
#   - "所有计时都基于 tick"：唯一的墙钟量是 max_queue_ms，它对锁步也是安全的，原因见 __init__ 里 max_queue_ms 上方
#   - "受保护的 DP 状态屏障"：engine_core.py DPEngineCoreProc.busy_loop 里每轮一次的 _sync_dp_state；它失败后
#     _shutting_down 置 True，所有 rank 按"要关闭"处理
#   - "浪费了约 97%"指 500/16384≈3% 的预算被用上；真正白花的是那份与 token 数无关的固定开销，见下面【科普】①(b)
#   - 不只是 DP-attention：任何 DP>1 的部署默认都会创建跨 DP 的 delayer（见下面④）
#
# ============================== 以下是解释（不是翻译） ==============================
# 【导读】PrefillDelayer：调度器每个 tick 问一次"这一步能不能放 prefill 进来"的闸门
#   创建：engine_core.py EngineCore._init_prefill_delayer → Scheduler.set_prefill_delayer（什么时候创建见下面④）
#   调用方：scheduler.py Scheduler._schedule（每个 tick：先 protects_decode，再 should_allow_prefill）；
#           engine_core.py EngineCore._process_engine_step_inner（含 prefill 的 forward 返回后调 notify_prefill_executed）
#   与主干的关系：挂在调度器上的可选闸门。返回 False（HOLD）时 _schedule 跳过 Phase 1（续算 chunked prefill）和
#     Phase 2（准入新请求），这一步只做 decode。它只决定"这一步做不做 prefill"，从不把 prefill 和 decode 拼进同一个 batch
#   术语：tick = 一次 Scheduler._schedule() 调用 ≈ 一个 step（一次 forward）；FIRE = 本 tick 允许 prefill；
#     HOLD = 本 tick 不放 prefill；扣住期（episode）= 上一次 FIRE 之后连续 HOLD 的一段，_hold_ticks 数的就是它的长度
# 阅读路线：
#   ① 【科普】为什么要扣住 prefill（从"一步只做一种"出发）
#   ② 两种模式的时间线（本地 TP、跨 DP），带数字；为什么用 SUM
#   ③ FIRE/HOLD 决策流程图（和 should_allow_prefill 的代码一一对应）
#   ④ 什么时候会创建 delayer；各配置项管什么、适用场景与代价
#   ⑤ 调度器怎样用它（scheduler.py 的调用点，按函数名）
#   ⑥ 对比 nano-vllm / SGLang
#   之后读代码：__init__（参数检查、归约缓冲区的槽位）→ protects_decode → should_allow_prefill → _fire / _hold / _reset
#
# ① 【科普】为什么要"扣住"prefill
#   (a) 出发点：ATOM 的基础调度器一步要么全是 prefill、要么全是 decode（engine_core.py 文件开头【2】）。
#     做 prefill 的那一步里，所有正在 decode 的请求都拿不到新 token——这一步有多长，它们这一次的 ITL（相邻两个 token 的间隔）
#     就被拉长多少。量级（沿用 engine_core.py【2】的估算：70B 稠密 FP8，MI300X 理论峰值 FP8≈2600 TFLOPS、HBM≈5.3TB/s）：
#       decode B=64 一步 ≈ 13ms（读一遍 70GB 权重）
#       prefill 500 token   ≈ 2×70e9×500   / 2600e12 ≈ 27ms（已超过约245 token的分界，算力受限）
#       prefill 16384 token ≈ 2×70e9×16384 / 2600e12 ≈ 0.88s（一个满预算，max_num_batched_tokens 默认 16384）
#   (b) 很多次小 prefill vs 一次大 prefill：每次 prefill forward 除了与 token 数成正比的计算，还有一份与 token 数无关的
#     固定开销（prefill 走 eager、不回放 CUDA Graph，要逐个发射每一层的 kernel；DP attention 下每个 MoE 层都有一次
#     跨 rank 的 all-to-all；还有一次 CPU 调度 + postprocess）。
#     例：每隔约 5 个 step 来一条 500 token 的短 prompt，一共 30 条（15000 token，历时约 145 个 step）；设每次 prefill 的固定开销
#       为 c（示意）。按默认上界：pending 每 5 个 step 增长一次（stall 到不了 10），扣住期约 145 个 tick（< ttft_max_ticks 200）
#       来一条放一条：30 次 prefill forward → 30c 固定开销；正在 decode 的请求被打断 30 次（每次 ≈27ms+c）
#       攒齐再放：    1 次 15000 token 的 forward（15000/16384≈0.92 ≥ target_fill 0.9）→ 1c；decode 只被打断 1 次（≈0.81s+c）
#     按理论峰值，纯计算量相同（30×27ms ≈ 0.81s），省下的是 29c 的固定开销和 29 次打断。
#     代价：先到的请求要等后面的请求凑齐（第 1 条多等约 145 个 step ≈ 1.9s，TTFT 变长），而那一次打断很长（ITL 出现一个大尖峰）。
#     所以"等多久"必须有上界（stall / ttft / kv / 排队时长）——这就是下面决策流程里那一串 FIRE 条件。
#     这和 TCP 的 Nagle 算法是一个思路：小包先攒在缓冲区里，攒够一个包（或等到上一个包的 ACK）再发，用一点延迟换更少的包
#   (c) 本地 TP 的另一件事：保护 decode（decode 间隔）。chunked prefill 做到一半的请求下一步优先续算（scheduler.py
#     Scheduler._schedule 的 Phase 1），于是一条长 prompt 的所有 chunk 会背靠背占满连续几个 step，decode 在这期间完全停住。
#     ATOM_PREFILL_DECODE_INTERVAL=N 让每次真正执行过的 prefill 之后至少跟 N 个只做 decode 的 tick，decode 就能插进 chunk 之间
#   (d) 跨 DP 的问题：DP rank 之间锁步（原理图见 engine_core.py DPEngineCoreProc 上方）：每一步所有 rank 一起 forward，
#     MoE 层每层都要跨 rank 交换 token，一步的时长由最慢的 rank 决定。只要有一个 rank 在 prefill，这一步就是"混合步"：
#       正在 decode 的 rank：每层 MoE 都在等做 prefill 的 rank；而且整个 DP 组这一步都不能回放 CUDA Graph，decode 改走 eager
#         （utils/forward_context.py ForwardMode.decide：unified = not any_rank_has_prefill，只有 unified 时才可能回放）
#       没有任何请求的 rank：跑一个 dummy batch 陪跑（DPEngineCoreProc._execute_dummy_batch），同样要等满这一步
#     各 rank 的 prefill 分散在不同的 tick，就是好几个混合步；对齐到同一个 tick，就只有一个 prefill 步。
#     对齐的代价：有 prefill、自己却没有 decode 的 rank，在 HOLD 的 tick 里调度出的是空 batch，只能跑 dummy 陪别人 decode
#
# ② 两种模式的时间线（P=prefill 步，d=decode 步，-=这一步没有轮到它；时长沿用①(a)）
#   本地 TP（DP=1，-tp 8，ATOM_PREFILL_DECODE_INTERVAL=4，其余默认）：64 条请求在 decode，来了一条 40000 token 的长 prompt L，
#     分成 16384 + 16384 + 7232 三个 chunk（见 arg_utils.py enable_chunked_prefill 上方的【科普】）；满 chunk≈0.88s，7232≈0.39s
#     不设间隔（默认值 0，DP=1 时根本不创建 delayer）：Phase 1 让 L 的 chunk 背靠背续算
#       tick    1     2     3     4   5   6 ...
#       L       P1    P2    P3    d   d   d          P3 顺带采样出 L 的第 1 个 token
#       64条    -     -     -     d   d   d          decode 连停 3 步 ≈ 0.88+0.88+0.39 ≈ 2.15s
#     ATOM_PREFILL_DECODE_INTERVAL=4（创建本地 delayer）：
#       tick    1     2..5     6     7..10    11..20    21    22 ...
#       L       P1    -        P2    -        -         P3    d
#       64条    -     d x4     -     d x4     d x10     -     d
#       决策    FIRE  A:HOLD   FIRE  A:HOLD   F:HOLD    FIRE
#               fill           fill                     stall          （A、F 是③流程图里的框）
#       tick 1、6：pending=16384（一个满 chunk）→ fill=1.0 ≥ 0.9 → FIRE；之后 4 个 tick 是 decode 间隔
#       tick 11..20：只剩 7232 的尾巴 → fill≈0.44 < 0.9；pending 不再增长，stall 从 0 数到 9；tick 21 stall=10 → FIRE
#     → decode 最长一次停顿 2.15s → 0.88s；L 的 TTFT ≈ 2.15s + 18×13ms ≈ 2.38s（多等了 18 个 decode 步）。
#       这期间若有新 prompt 到达，会和尾巴 chunk 进同一个 prefill batch（Phase 1 与 Phase 2 在同一步），凑够 fill 时还会更早放行
#   跨 DP（--enable-dp-attention，DP=4，每个 rank 约 32 条 decode，其余默认、间隔 0）：tick 1..4 依次各有一条 4000 token 的
#     prompt 路由到 rank0..rank3
#     不开 delayer（ATOM_ENABLE_PREFILL_DELAYER=0）：
#       tick     1    2    3    4    5
#       rank0    P    d~   d~   d~   d
#       rank1    d~   P    d~   d~   d
#       rank2    d~   d~   P    d~   d
#       rank3    d~   d~   d~   P    d
#       d~ = 混合步里的 decode（eager，每层 MoE 都在等做 prefill 的那个 rank，这一步≈一次 4000 token prefill 那么长）
#       → 连续 4 个混合步：每个 rank 的 decode 都被拖慢了 4 次
#     默认（delayer 开）：
#       tick     1    2    3    4    5 .. 13   14   15
#       rank0    d    d    d    d    d .. d    P    d
#       rank1    d    d    d    d    d .. d    P    d
#       rank2    d    d    d    d    d .. d    P    d
#       rank3    d    d    d    d    d .. d    P    d
#       tick 1..3：n_prefillable=1、2、3 < 4 → D 对齐门 HOLD（不更新 stall）
#       tick 4..13：四个 rank 都可 prefill，G_pending=16000，fill=16000/(4×16384)≈0.24 < 0.9，G 不再增长：stall=0..9 → HOLD
#       tick 14：stall=10 → FIRE：四个 rank 在同一步 prefill
#     → 只有 1 个 prefill 步；tick 1..13 都是纯 decode 步（全组可以回放 CUDA Graph）。代价：rank0 的请求多等了 13 个 tick。
#       某个 rank 一直没有 prefill（负载不均）时，C④ 的 ttft_max_ticks=200 个 tick 之后不再等它，不对齐也放行
#   为什么用 SUM 而不是 MAX（DP=4，四个 rank 都可 prefill；锁步下一步的时长由最慢的 rank 决定）：
#       rank       0       1      2      3
#       pending    16384   1000   1000   1000
#     MAX 式（最忙的 rank 满了就放行）：这一步 rank0 要算 16384 个 token（≈0.88s），rank1~3 反正要陪它等满这一步，
#       却只用了自己预算的 6%，剩下的 prompt 以后还得再占 prefill 步
#     SUM 式：fill=(16384+3×1000)/(4×16384)≈0.30 < 0.9 → HOLD，rank1~3 继续积累，放行时各自带更大的 batch
#     反方向的问题（某个 rank 的 chunk 特别贵，拖长整步）由 DP prefill 代价均衡处理：各 rank 的代价搭同一次 all_reduce 交换，
#       贵的 rank 少拿 token（见 envs.py ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT 上方、scheduler.py Scheduler._balance_prefill_step）
#
# ③ FIRE/HOLD 决策流程（should_allow_prefill，自上而下与代码一一对应）
#   ┌─ 0 采集与归约 ──────────────────────────────────────────────────────────────────┐
#   │ kv_high / kv_low / queue_hot 只在本 rank prefillable 时才可能为 1               │
#   │ 填 8（+dp_size）个槽；cpu_group 非 None → all_reduce(SUM)；本地模式不通信       │
#   │   all_reduce 抛 RuntimeError → _reset，return True（fail-open）                 │
#   │ 读回；清掉本 rank 的 _prefill_executed_since_last_decision                      │
#   └─────────────────────────────────────────────────────────────────────────────────┘
#        │
#   prefill_ran_n > 0（上一 tick 有 rank 执行过 prefill）？ ── 是 ──> _decode_interval_remaining = prefill_decode_interval
#        │
#   _first？ ── 是 ──> _first=False，_reset，FIRE（一次性，不计统计）
#        │ 否
#   ┌─ A decode 间隔 ──────────────────────────────────────────────────────────────────────────────────┐
#   │ _decode_interval_remaining > 0？ 是：减 1；再看 n_prefillable>0 且 g_running_dec>0？ 是 ──> HOLD │
#   │   （不推进 _hold_ticks：合并器的 ttft / partial 计时从间隔结束后才开始）                         │
#   │   否（没人要 prefill，或没有 decode）：继续往下                                                  │
#   └──────────────────────────────────────────────────────────────────────────────────────────────────┘
#        │
#   B n_prefillable == 0？ ── 是 ──> FIRE vacuous（放行 = 空操作）
#        │ 否
#   ┌─ C 必放行上界（不管对齐、不管填满）─────────────────────────────────────────────┐
#   │ ① g_running_dec == 0                                     ──> FIRE nodecode      │
#   │ ② any_kv_high 或 any_kv_low                              ──> FIRE kv            │
#   │ ③ any_queue_hot                                          ──> FIRE queue_ms      │
#   │ ④ _hold_ticks >= ttft_max_ticks (200)                    ──> FIRE ttft          │
#   │ ⑤ any_partial 且 _hold_ticks >= partial_max_ticks (100)  ──> FIRE partial       │
#   └─────────────────────────────────────────────────────────────────────────────────┘
#        │ 都不满足
#   D 对齐门：n_prefillable < dp_size？ ── 是 ──> HOLD（等落后的 rank 也有 prefill；本地模式恒为否）
#        │ 否
#   E 填充目标：fill = g_pending / (n_prefillable × max_num_batched_tokens) >= target_fill（0.9）？ ── 是 ──> FIRE fill
#        │ 否
#   F 停滞放弃：g_pending <= _prev_pending？ 是：stall+1；否：stall=0；然后 _prev_pending = g_pending
#              _stall_count >= stall_ticks（10）？ ── 是 ──> FIRE stall
#        │ 否
#   其余情况：HOLD
#   FIRE（_fire）：对应原因的计数 +1，_reset（_hold_ticks=0、_stall_count=0、_prev_pending=-1），return True
#   HOLD（_hold）：_hold_ticks += 1，return False（框 A 的 HOLD 不走 _hold，不推进 _hold_ticks）
#
# ④ 什么时候会创建 delayer（engine_core.py；按 EngineCore.run_engine 选择的类）
#   PP>1  ──> PPEngineCoreProc：EngineCore.__init__ 里 PP≠1 不挂，PPEngineCoreProc 自己也不挂 ──> 没有 delayer
#   DP>1  ──> DPEngineCoreProc.__init__ ──> _init_prefill_delayer(config, self.dp_group) ──> 跨 DP 模式（cpu_group = DP 的 gloo 组）
#   其余  ──> EngineCore.__init__：DP==1 且 PP==1 且 ATOM_PREFILL_DECODE_INTERVAL>0
#               ──> _init_prefill_delayer(config) ──> 本地模式（cpu_group=None → is_local）
#   _init_prefill_delayer 里再挡掉三种：ATOM_ENABLE_PREFILL_DELAYER=0；enable_rapidserve（RapidServe 用自己的
#     PrefillScheduler / DecodeScheduler）；self.scheduler 为 None（同 GPU PD 分离的 decode 侧推迟创建调度器）
#   所以：DP>1（包括 --enable-dp-attention 把 DP 乘以 TP 之后）默认就有 delayer，ATOM_PREFILL_DECODE_INTERVAL 只决定有没有
#     decode 间隔；DP=1 时默认没有，设 ATOM_PREFILL_DECODE_INTERVAL>0 才同时打开 decode 间隔和本地合并器
#     （docs/environment_variables.md 的 "Prefill delayer" 一节）。构造参数全部来自 envs.py（见 __init__ 上方的对应表）
#   exchange_prefill_costs = scheduler.cost_balance（--enable-dp-attention 且 DP>1）：各 rank 的 prefill 代价搭同一次
#     all_reduce 交换，供 Scheduler._balance_prefill_step 使用；关掉 delayer 也就关掉了这项均衡
#   各配置项（envs.py【调度】一节；默认值；管的是③里哪个框；适用场景 / 代价）：
#     ATOM_ENABLE_PREFILL_DELAYER=1        总开关。设 0：任何部署都不创建 delayer（DP 下也不做 prefill 对齐和代价均衡）
#       适用：想要 prefill 来了就做；recipes/DeepSeek-V4-Agentic-PD-Max.md 在跨节点 PD 分离 + DP attention 的 prefill 节点
#         和 decode 节点上都设为 0（那里每个节点基本只承担一种负载）。代价：DP 下回到各 rank 各自 prefill 的混合步
#     ATOM_PREFILL_DECODE_INTERVAL=0       框 A：每次真正执行过的 prefill 之后，有 decode 时强制 N 个只做 decode 的 tick
#       适用：长 prompt 多、ITL/TPOT 敏感——decode 能插进 chunk 之间（②的本地 TP 例子）；单 DP 下它 >0 才创建 delayer
#       代价：长 prompt 的 TTFT 至少增加 N ×（chunk 数 − 1）个 decode 步，不满 fill 的尾巴 chunk 还可能再被合并器扣住
#         最多 stall_ticks 个 tick（②的例子：4×2 + 10 = 18）；单 DP 下还连带打开合并器
#     ATOM_PREFILL_DELAYER_TARGET_FILL=0.9 框 E：放行所需的填充率。越高 → prefill 越少越大、TTFT 越长；越低 → 越接近不合并；
#       截断到 (0, 1]
#     ATOM_PREFILL_DELAYER_STALL_TICKS=10  框 F：待 prefill 量连续多少个 tick 不增长就放弃凑满。越小 → 越早放行未满的 batch
#     ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS=200  框 C④：一个扣住期最多连续 HOLD 多少个 tick（按 decode≈13ms/tick 约 2.6s）；
#       不是端到端 TTFT 上界（不含 decode 间隔、积压和执行时间）。单 DP 下还是 Scheduler._wait_for_inflight_prefix
#       等待"在途前缀 checkpoint"的期限
#     ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS=100  框 C⑤：有 rank 在 chunked prefill 中途时更紧的上界——中途的请求已经占着
#       KV block，扣着它等于让这些 KV 空占；只有小于 TTFT_MAX_TICKS 才有意义
#     ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK=0.9  框 C②：可 prefill 的 rank KV 使用率 ≥ 它就放行——KV 快满了，再等也攒不出更大的 batch
#     ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK=（空，关闭）  框 C②：KV 使用率 < 它就放行——KV 用得少说明在跑的请求少、
#       GPU 吃不饱，先把 prefill 喂进去（对应 SGLang 的 token_usage_low_watermark）
#     ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=（空，关闭）  框 C③：最老的可准入等待请求从到达起排队满这么多 ms 就放行（墙钟时间，
#       含积压和之前的 HOLD）；排在框 A 之后，所以也不是 TTFT 的硬上限
#     ATOM_PREFILL_DELAYER_DEBUG=0、ATOM_PREFILL_DELAYER_LOG_EVERY=1000  每 tick 打印 FIRE/HOLD 明细 / 每 N 次决策打印一行统计；
#       这两个不在 envs.py 里，本文件直接读 os.environ（docs/environment_variables.md 有记录）
#
# ⑤ 调度器怎样用它（scheduler.py）
#   Scheduler.set_prefill_delayer：保存 delayer，并设 _local_prefill_coalescing = delayer.is_local（本地模式专用的分支看它）
#   Scheduler._schedule，在"没有请求就 return None"之前（否则 DP 下有的 rank 跳过 all_reduce，锁步就断了）：
#     running_decode_batch = len(running) − _partial_prefill_count（chunked prefill 中途的请求不算 decode 负载）
#     protects_decode = delayer.protects_decode(running_decode_batch)，然后按三种情况准备信号：
#       ┌ protects_decode 为 True，或本地模式且没有 decode：只报"有没有活"（_partial_prefill_count>0 或 _waiting_prefills() 非空），
#       │   pending_tokens=0——结果已经确定（框 A 必 HOLD / 框 C① 必 FIRE），不必做前缀 cache 探测
#       ├ 本地模式：_local_prefill_pending_work()——按准入时同样的 chunk 上限探测（扣掉 HBM 前缀命中），扫描量封顶一个预算
#       └ 跨 DP 模式：prefillable = _can_admit_head_prefill()（队头真能分到 block 才算），
#           pending = min(_waiting_new_token_count() + _partial_prefill_remaining_tokens(), max_num_batched_tokens)
#     kv_usage = _kv_usage()（保护期内传 0）；oldest_waiting_age_ms = _oldest_waiting_prefill_age_ms()（只在设了 max_queue_ms
#       且不在保护期时才算）；prefill_cost = _estimate_prefill_chunk_cost()（只在 cost_balance 且本 rank 有 prefill 时才算）
#     → 得到 delayer_allows = should_allow_prefill(...)
#       False：Phase 1（续算 partial）和 Phase 2（准入新请求）都跳过，这一步走 decode；被跳过的 partial 挪到 running 队尾
#       True 且 cost_balance：_balance_prefill_step(prefill_cost) 用 rank_prefill_costs 定本步的代价上限
#   Scheduler._wait_for_inflight_prefix（只在本地模式）：等待在途前缀 checkpoint 的期限 = 当前 tick + ttft_max_ticks
#   EngineCore._process_engine_step_inner：batch.total_seqs_num_prefill > 0 的 forward 返回后 → notify_prefill_executed()
#
# ⑥ 对比
#   对比nano-vllm：nano-vllm 没有这个机制——waiting 里只要有能调度的请求，这一步就做 prefill（prefill 优先），也没有 DP。
#     ATOM 的基础调度器同样是 prefill 优先；delayer 是唯一能让它"明明有 prefill 可做却先做 decode"的地方
#   对比SGLang：本文件的做法来自 SGLang 的 PrefillDelayer（python/sglang/srt/managers/prefill_delayer.py），代码里多处引用它
#     （skip_first、scheduler.py _can_admit_head_prefill 的 local_prefillable、decode 间隔的位置）。SGLang 那边（main 分支）：
#     各 DP rank 的局部状态做 all_gather，归成"全部可 prefill / 都不可 / 混合"三态，主要在混合时延迟、最多 max_delay_passes 轮，
#     另有 token_usage_low_watermark 安全阀和 skip_first_delayer；它的 PrefillDelayer 本身不处理 decode 间隔。ATOM 的不同：
#     ① 一次 all_reduce(SUM) 只传 8 个整数（加可选的代价槽），各 rank 在同一组归约值上算出同一个结论；
#     ② 全部可 prefill 时也不立刻放行，还要凑够 target_fill（合并器），并有 stall / ttft / partial / kv / 排队时长这些上界；
#     ③ 多了单 DP 的本地模式和 prefill 之后的 decode 间隔；④ 顺带交换各 rank 的 prefill 代价做均衡

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# ATOM_PREFILL_DELAYER_DEBUG=1：每次决策打印一行 FIRE/HOLD 明细（_fire / _hold 里）；模块导入时读一次，要在启动前设置
_DEBUG = os.environ.get("ATOM_PREFILL_DELAYER_DEBUG", "0") == "1"


# PrefillDelayer：每个调度器至多挂一个（Scheduler.prefill_delayer），只在 busy_loop 线程里被调用（schedule() 和
#   _process_engine_step_inner 都在那个线程），不需要加锁。状态分三类：配置（__init__ 参数）、扣住期状态、统计计数
class PrefillDelayer:
    # __slots__：声明实例只有这些属性，不为每个实例创建 __dict__——属性访问更快、更省内存；
    #   拼错属性名（比如给 self._hold_tick 赋值）会直接抛 AttributeError，而不是悄悄多出一个属性
    # 元组按字母序排列，下面两行分组注释只是大致位置：_first、_hold_ticks 等扣住期状态排在它们前面，cpu_group 之后基本是配置（rank_prefill_costs 是每次决策的输出）
    __slots__ = (
        "_cost_slots",
        "_decode_interval_remaining",
        "_first",
        "_hold_ticks",
        "_prefill_executed_since_last_decision",
        "_prev_pending",
        # buffer / episode state
        # 【译】缓冲区 / 扣住期状态
        "_rank",
        "_reduce_buf",
        "_stall_count",
        # stats
        # 【译】统计
        "_stat_fire_fill",
        "_stat_fire_kv",
        "_stat_fire_nodecode",
        "_stat_fire_partial",
        "_stat_fire_queue_ms",
        "_stat_fire_stall",
        "_stat_fire_ttft",
        "_stat_fire_vacuous",
        "_stat_hold",
        "_stat_hold_decode_interval",
        "_stat_log_every",
        "cpu_group",
        "dp_size",
        "kv_high_watermark",
        "max_num_batched_tokens",
        "max_queue_ms",
        "partial_max_ticks",
        "prefill_decode_interval",
        "rank_prefill_costs",
        "stall_ticks",
        "target_fill",
        "token_usage_low_watermark",
        "ttft_max_ticks",
    )

    # 参数与配置的对应（engine_core.py EngineCore._init_prefill_delayer 传入；含义、适用场景见文件开头④）：
    #   dp_size                ← config.parallel_config.data_parallel_size（--enable-dp-attention 改写之后的值）
    #   cpu_group              ← DP>1：DPEngineCoreProc.dp_group（各 EngineCore 进程之间的 gloo 组）；DP=1：None
    #   max_num_batched_tokens ← config.max_num_batched_tokens（默认 16384）：fill 的分母是"每个可 prefill 的 rank 一个满预算"
    #   target_fill、ttft_max_ticks、partial_max_ticks、stall_ticks、kv_high_watermark、token_usage_low_watermark、
    #     max_queue_ms         ← envs.py 的 ATOM_PREFILL_DELAYER_*（同名）
    #   prefill_decode_interval ← ATOM_PREFILL_DECODE_INTERVAL（envs.py，默认 0）
    #   exchange_prefill_costs ← scheduler.cost_balance（--enable-dp-attention 且 DP>1）
    #   这里的默认值与 envs.py 的默认值相同；atom/ 里唯一的构造处（engine_core.py）总是显式传入全部参数
    def __init__(
        self,
        dp_size: int,
        cpu_group,
        max_num_batched_tokens: int,
        target_fill: float = 0.9,
        ttft_max_ticks: int = 200,
        partial_max_ticks: int = 100,
        stall_ticks: int = 10,
        kv_high_watermark: float = 0.9,
        token_usage_low_watermark: float | None = None,
        max_queue_ms: float | None = None,
        prefill_decode_interval: int = 0,
        exchange_prefill_costs: bool = False,
    ):
        # 跨 DP 模式必须有通信组；本地模式（dp_size=1）的 cpu_group 为 None，跳过 all_reduce
        if dp_size > 1 and cpu_group is None:
            raise ValueError("Cross-DP prefill coalescing requires a CPU group")
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.max_num_batched_tokens = max_num_batched_tokens

        # target_fill is the fraction of a full prefill batch that must be
        # accumulated (across prefillable ranks) before we release. Must be in
        # (0, 1]: <= 0 would fire every tick (no coalescing); > 1 is unreachable
        # (fill is capped at 1) and would degrade to stall/timeout-only. Clamp.
        # 【译】target_fill 是放行之前必须积累（在可 prefill 的 rank 之间合计）的"满 prefill batch"的比例。必须在 (0, 1] 之内：
        #   <= 0 会每个 tick 都放行（没有合并）；> 1 不可能达到（fill 封顶为 1），会退化成只靠 stall / 超时放行。所以做截断。
        # 解释：fill ≤ 1 是因为调用方把每个 rank 的 pending_tokens 都封顶在 max_num_batched_tokens（scheduler.py 两条信号路径
        #   都封顶），而走到框 E 时所有 rank 都可 prefill，分母正好是 dp_size 个满预算
        if target_fill <= 0.0:
            logger.warning(
                f"target_fill={target_fill} <= 0 disables coalescing "
                "(fires every tick); clamping to a small positive 0.05."
            )
            target_fill = 0.05
        elif target_fill > 1.0:
            logger.warning(
                f"target_fill={target_fill} > 1.0 is unreachable (fill is capped "
                "at 1.0); clamping to 1.0."
            )
            target_fill = 1.0
        self.target_fill = target_fill

        # Tick bounds must be >= 1. 0 would make the corresponding `>=` fire on
        # the first tick, defeating the bound.
        # 【译】tick 上界必须 >= 1。取 0 会让对应的 `>=` 在第一个 tick 就触发，这个上界也就失去了意义。
        self.ttft_max_ticks = self._clamp_ticks("ttft_max_ticks", ttft_max_ticks)
        self.partial_max_ticks = self._clamp_ticks(
            "partial_max_ticks", partial_max_ticks
        )
        self.stall_ticks = self._clamp_ticks("stall_ticks", stall_ticks)
        # 两个 KV 水位不做范围检查：kv_usage ∈ [0, 1]，高水位设成 > 1 就永远不会因 KV 满而放行；低水位 None 表示关闭
        self.kv_high_watermark = kv_high_watermark
        self.token_usage_low_watermark = token_usage_low_watermark
        # TTFT SLA guard: if any rank's oldest schedulable waiting-prefill has
        # queued (since arrival) >= this many ms, force release regardless of the
        # fill target. None disables it. Wall-clock, but lockstep-safe: each rank
        # compares its OWN requests' ages to the threshold locally (disjoint
        # request sets across ranks), and only the OR of those per-rank booleans
        # crosses the collective — so all ranks act on the identical result the
        # same tick. (Contrast the removed per-rank hold-clock E1, which compared
        # each rank's own clock to a threshold for a GLOBAL decision → skew.)
        # 【译】TTFT SLA 保护：如果任一 rank 最老的可调度等待 prefill 从到达起已经排队 >= 这么多毫秒，就无视填充目标强制放行。
        #   None 表示关闭。它用的是墙钟时间，但对锁步是安全的：每个 rank 在本地拿*自己*请求的年龄和阈值比较（各 rank 的请求
        #   集合互不相交），只有这些逐 rank 布尔值的"或"经过集合通信——所以所有 rank 在同一个 tick 依据完全相同的结果行动。
        #   （对比已删除的逐 rank 扣住时钟 E1：它拿每个 rank 自己的时钟和阈值比较，却用来做*全局*决定 → 各 rank 产生错位。）
        # 解释：E1 是旧实现里按各 rank 自己的墙钟计时的扣住上界，已从代码中删除（envs.py TTFT_MAX_TICKS 的说明提到 tick 上界
        #   "取代了旧的墙钟时间 + 轮数组合"）。区别在于：E1 是"每个 rank 各自判断一个全局条件"，时钟稍有差别就会有的 rank FIRE、
        #   有的 HOLD，锁步被打破；这里是"每个 rank 判断自己的局部事实，再取或"，或的结果对所有 rank 相同。
        #   "无视填充目标"之外，它也排在 decode 间隔（框 A）之后：保护期内不生效
        self.max_queue_ms = max_queue_ms
        # 0 = 不设 decode 间隔；负数直接报错（不像 tick 上界那样截断成合法值）
        if prefill_decode_interval < 0:
            raise ValueError(
                "prefill_decode_interval must be non-negative; "
                f"got {prefill_decode_interval}"
            )
        self.prefill_decode_interval = prefill_decode_interval

        # 8-slot SUM-reduce buffer (gloo-safe). Encoding:
        #   slot 0 = prefillable        (SUM → n_prefillable)
        #   slot 1 = pending_tokens     (SUM → G_pending; fresh + partial remain)
        #   slot 2 = running_decode     (SUM → G_running_dec)
        #   slot 3 = kv_high flag       (SUM>0 → any prefillable rank KV-high)
        #   slot 4 = kv_low flag        (SUM>0 → any prefillable rank KV-low)
        #   slot 5 = has_partial flag   (SUM>0 → any rank mid-chunked-prefill)
        #   slot 6 = queue_hot flag     (SUM>0 → any rank's oldest waiting prefill
        #                                aged past max_queue_ms; TTFT SLA guard)
        #   slot 7 = prefill_ran flag   (SUM>0 → a rank completed a prefill
        #                                forward on the previous scheduler tick)
        # With exchange_prefill_costs (DP prefill cost balancing), dp_size more:
        #   slot 8 + r = rank r's prefill chunk cost (one-hot, SUM → every
        #                rank's cost), read back as rank_prefill_costs
        # 【译】8 槽的 SUM 归约缓冲区（gloo 可用）。编码：
        #   槽 0 = prefillable        （SUM → n_prefillable）
        #   槽 1 = pending_tokens     （SUM → G_pending；新请求 + 部分完成请求的剩余）
        #   槽 2 = running_decode     （SUM → G_running_dec）
        #   槽 3 = kv_high 标志       （SUM>0 → 有可 prefill 的 rank KV 偏高）
        #   槽 4 = kv_low 标志        （SUM>0 → 有可 prefill 的 rank KV 偏低）
        #   槽 5 = has_partial 标志   （SUM>0 → 有 rank 处于 chunked prefill 中途）
        #   槽 6 = queue_hot 标志     （SUM>0 → 有 rank 最老的等待 prefill 排队超过了 max_queue_ms；TTFT SLA 保护）
        #   槽 7 = prefill_ran 标志   （SUM>0 → 有 rank 在上一个调度 tick 完成了一次 prefill forward）
        #   开启 exchange_prefill_costs（DP prefill 代价均衡）时再加 dp_size 个：
        #   槽 8 + r = rank r 的 prefill chunk 代价（one-hot，SUM → 每个 rank 的代价），读回到 rank_prefill_costs
        # 例（DP=4，没开代价交换，设本扣住期 _hold_ticks < 100）：
        #   槽       0      1        2      3   4   5   6   7
        #   rank0    1      16384    32     0   0   0   0   0
        #   rank1    1      1000     30     0   0   0   0   0
        #   rank2    0      0        31     0   0   0   0   0
        #   rank3    1      7000     28     0   0   1   0   0      （rank3 有一个 chunked prefill 做到一半）
        #   SUM      3      24384    121    0   0   1   0   0
        #   → 框 C 都没触发（有 decode、KV 正常、partial 还没扣满 100 个 tick）→ 框 D：n_prefillable=3 < 4 → HOLD，等 rank2
        self._cost_slots = dp_size if exchange_prefill_costs else 0
        # The DP group is stateless (no default process group): ask it directly.
        # 【译】DP 组是无状态的（没有默认进程组）：直接问它自己。
        # 解释：这个组由 stateless_init_dp_group 创建，不是 torch.distributed 的默认组，所以不能用 torch.distributed.get_rank()，
        #   要调组对象自己的 rank()；只有交换代价时才需要知道自己写哪个槽，否则取 0
        self._rank = (
            cpu_group.rank() if self._cost_slots and cpu_group is not None else 0
        )
        # CPU 上的 int64 张量：gloo 组在 CPU 张量上做集合通信；每个 tick 复用同一块缓冲区，不重新分配
        self._reduce_buf = torch.zeros(
            8 + self._cost_slots, dtype=torch.int64, device="cpu"
        )
        # 每次决策后由归约结果改写：每个 rank 的 chunk 代价（没开代价交换时为空列表）；读取方：scheduler.py Scheduler._balance_prefill_step
        self.rank_prefill_costs: list[float] = []

        # Episode state. All ticks decide FIRE/HOLD in lockstep, so these evolve
        # identically on every rank (deterministic).
        # 【译】扣住期状态。所有 tick 都锁步地决定 FIRE/HOLD，所以这些状态在每个 rank 上的演化完全相同（确定性的）。
        # 解释：例外是 _prefill_executed_since_last_decision——它是本 rank 的局部事实，要经过下一次 all_reduce（槽 7）才变成
        #   所有 rank 一致的"有 rank 跑过 prefill"
        #   _hold_ticks：本扣住期已连续 HOLD 的 tick 数（框 C④⑤ 用）；_stall_count / _prev_pending：框 F 的停滞计数和上一次的
        #   G_pending（-1 表示本扣住期还没走到过框 F）
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        # First call fires immediately to seed the initial decode batch build-up
        # (mirrors SGL PR #19836's skip_first).
        # 【译】第一次调用立即放行，为最初的 decode batch 的积累打底（对应 SGL PR #19836 的 skip_first）。
        # 观察：在正常的启动路径上，第一次调用时一定还没有 decode：DP>1 时 engines_running 初始为 True，启动后的第一轮即使
        #   没有请求也会走一次 schedule()，_first 就在这个空 tick 上被用掉（n_prefillable=0，本来也是框 B 放行）；DP=1 时
        #   busy_loop 只在调度器里有请求时才调 schedule()，第一次调用时 running 为空（本来也是框 C① 放行）。所以它在 ATOM 里
        #   不改变决策，只是这一次不计入统计；SGLang 的 skip_first_delayer 跳过的是"第一次本该发生的延迟"，语义不同
        self._first = True
        # decode 间隔倒计时（框 A）：should_allow_prefill 看到"上一 tick 有 rank 跑过 prefill"时装填为 prefill_decode_interval，
        #   之后每次决策减 1
        self._decode_interval_remaining = 0
        # 本 rank 自上一次决策以来是否真的执行过 prefill forward：notify_prefill_executed 置位，下一次决策写进槽 7 后清零
        self._prefill_executed_since_last_decision = False

        # Per-exit fire counters + hold counter for periodic logging. Each exit
        # is counted separately so the log shows WHICH reason released prefill —
        # a high fire_stall/fire_ttft vs fire_fill means the queue rarely reaches
        # target and the coalescer is mostly giving up rather than batching.
        # 【译】每个出口一个 FIRE 计数器，外加 HOLD 计数器，用于周期性日志。每个出口单独计数，这样日志能看出是*哪个*原因
        #   放行了 prefill——fire_stall / fire_ttft 相对 fire_fill 偏高，说明队列很少达到目标，合并器多半是在"放弃"而不是在合批。
        # 解释：调参时看这一行：fire_fill 占多数 → 合批在起作用；fire_stall / fire_ttft 占多数 → target_fill 对当前负载太高，
        #   每次只是白白多等了 stall_ticks / ttft_max_ticks 个 tick。hold_decode_interval 是框 A 造成的 HOLD（同时也计入 hold）
        self._stat_fire_fill = 0
        self._stat_fire_stall = 0
        self._stat_fire_ttft = 0
        self._stat_fire_kv = 0
        self._stat_fire_partial = 0
        self._stat_fire_nodecode = 0
        self._stat_fire_queue_ms = 0
        self._stat_fire_vacuous = 0
        self._stat_hold = 0
        self._stat_hold_decode_interval = 0
        # ATOM_PREFILL_DELAYER_LOG_EVERY：每多少次决策打印一行统计（默认 1000，<= 0 关闭）；不在 envs.py 里，直接读 os.environ
        self._stat_log_every = int(
            os.environ.get("ATOM_PREFILL_DELAYER_LOG_EVERY", "1000")
        )

        logger.info(
            f"PrefillDelayer initialized: dp_size={dp_size} "
            f"max_num_batched_tokens={max_num_batched_tokens} "
            f"target_fill={self.target_fill} "
            f"ttft_max_ticks={self.ttft_max_ticks} "
            f"partial_max_ticks={self.partial_max_ticks} "
            f"stall_ticks={self.stall_ticks} "
            f"kv_high_watermark={kv_high_watermark} "
            f"token_usage_low_watermark={token_usage_low_watermark} "
            f"max_queue_ms={max_queue_ms} "
            f"prefill_decode_interval={prefill_decode_interval}"
        )

    # 把 < 1 的 tick 上界截断为 1 并告警（ttft_max_ticks / partial_max_ticks / stall_ticks 共用）
    #   @staticmethod：不需要 self 的工具函数，通过 self._clamp_ticks(...) 调用
    @staticmethod
    def _clamp_ticks(name: str, value: int) -> int:
        if value < 1:
            logger.warning(
                f"{name}={value} < 1 would fire on the first tick (bound "
                "disabled); clamping to 1."
            )
            return 1
        return value

    # 本地模式（单 DP 的 TP/DCP）：dp_size == 1 且没有通信组 → 不做 all_reduce，只用本 rank 的信号
    #   读取方：本类的 protects_decode；scheduler.py Scheduler.set_prefill_delayer（据此设 _local_prefill_coalescing）
    #   @property：像属性一样访问（delayer.is_local），不用加括号调用
    @property
    def is_local(self) -> bool:
        return self.dp_size == 1 and self.cpu_group is None

    def protects_decode(self, running_decode_batch: int) -> bool:
        """Whether the local decision can skip cache probes during protection."""
        # 【译】本地决策在保护期内是否可以跳过 cache 探测。
        # 解释：调度器在调 should_allow_prefill 之前先问这一句（scheduler.py Scheduler._schedule）。为 True 时本 tick 的结果
        #   已经确定是框 A 的 HOLD，调度器只需报告"有没有 prefill 的活"，不必为等待中的请求做前缀 cache 探测
        #   （can_allocate_memoized）、扫描排队时长、读 KV 使用率。四个条件：
        #     只在本地模式（DP 模式下各 rank 都要报真实信号参与归约）；不是第一次调用（第一次必 FIRE）；
        #     本 rank 有 decode（没有 decode 时框 A 不 HOLD）；
        #     倒计时 > 0，或者"刚跑过 prefill、倒计时要在接下来这次决策里才装填"
        #   例（interval=4）：tick 1 执行了 prefill → tick 2 调用时倒计时还是 0，但标志为 True → True；
        #     tick 2~5 都为 True（倒计时 4→3、3→2、2→1、1→0，都 HOLD），tick 6 为 False（回到合并器）
        return (
            self.is_local
            and not self._first
            and running_decode_batch > 0
            and (
                self._decode_interval_remaining > 0
                or (
                    self._prefill_executed_since_last_decision
                    and self.prefill_decode_interval > 0
                )
            )
        )

    # 每个 tick 一次的 FIRE/HOLD 决策，流程图见文件开头③；调用方：scheduler.py Scheduler._schedule（每个 tick、早退之前）
    def should_allow_prefill(
        self,
        prefillable: bool,
        pending_tokens: int,
        running_decode_batch: int = 0,
        kv_usage: float = 0.0,
        has_partial: bool = False,
        oldest_waiting_age_ms: float = 0.0,
        prefill_cost: float = 0.0,
    ) -> bool:
        """Return True iff this rank may admit new prefills this tick (FIRE).

        MUST be called every tick on every DP rank (it runs a cross-DP
        all_reduce) so ranks stay in lockstep.

        Args:
            prefillable: this rank has admittable prefill work (fresh head that
                can allocate, or a resumable partial). Only prefillable ranks
                count toward the fill target and the alignment gate. During
                local decode protection, the caller uses work existence alone:
                fit cannot change the hard interval decision.
            pending_tokens: this rank's accumulated prefill tokens — fresh
                waiting new-tokens PLUS the remaining tokens of resumable
                partials — already capped at max_num_batched_tokens by the
                caller. The local scheduler bounds its scan and uses chunk
                limits, so this signal can omit work beyond that scan.
            running_decode_batch: decode seqs running on this rank (NOT counting
                mid-chunked-prefill seqs). If no rank has decode, holding wastes
                GPU → fire.
            kv_usage: fraction of KV cache blocks in use ∈ [0, 1]. Drives the
                KV-high (can't accumulate more) and KV-low (GPU starving) bounds.
            has_partial: this rank has a mid-chunked-prefill seq in flight. Its
                remaining tokens are in pending_tokens; a partial only forces
                release once held for partial_max_ticks (it holds KV), counted
                after the decode-protection interval.
            oldest_waiting_age_ms: age (ms since arrival) of this rank's oldest
                schedulable waiting prefill. If it reaches max_queue_ms, this
                rank flags the age guard; all ranks release after decode
                protection. No waiting prefill / max_queue_ms=None → inactive.
            prefill_cost: this rank's estimated prefill chunk cost (0 if none).
                With exchange_prefill_costs every rank's value is in
                ``rank_prefill_costs`` after the call.
        """
        # 【译】当且仅当本 rank 在这个 tick 可以准入新的 prefill（FIRE）时返回 True。
        #
        #   必须在每个 DP rank 上每个 tick 都调用（它会执行一次跨 DP 的 all_reduce），这样各 rank 才能保持锁步。
        #
        #   参数：
        #     prefillable：本 rank 有可准入的 prefill 工作（能分配到资源的新请求队头，或可以续算的部分完成请求）。
        #       只有可 prefill 的 rank 才计入填充目标和对齐门。本地 decode 保护期间，调用方只看"有没有活"：
        #       放不放得下不会改变硬间隔的决定。
        #     pending_tokens：本 rank 积累的待 prefill token——等待中新请求的新 token *加上* 可续算的部分完成请求的剩余 token——
        #       调用方已经把它封顶在 max_num_batched_tokens。本地调度器限制了扫描范围并使用 chunk 上限，
        #       所以这个信号可能漏掉扫描范围之外的工作。
        #     running_decode_batch：本 rank 上正在 decode 的序列数（*不*包括 chunked prefill 中途的序列）。
        #       如果没有任何 rank 在 decode，扣住只会浪费 GPU → 放行。
        #     kv_usage：已用 KV cache block 的比例 ∈ [0, 1]。驱动 KV 高水位（没法再积累更多）和 KV 低水位（GPU 在挨饿）两个上界。
        #     has_partial：本 rank 有一个 chunked prefill 中途的序列在途。它的剩余 token 已计入 pending_tokens；部分完成的请求
        #       只有在被扣住满 partial_max_ticks 之后才会强制放行（它占着 KV），从 decode 保护间隔结束之后开始计。
        #     oldest_waiting_age_ms：本 rank 最老的可调度等待 prefill 的年龄（从到达起的毫秒数）。如果达到 max_queue_ms，
        #       本 rank 就打上年龄保护标志；所有 rank 在 decode 保护之后放行。没有等待中的 prefill / max_queue_ms=None → 不生效。
        #     prefill_cost：本 rank 估计的 prefill chunk 代价（没有则为 0）。开启 exchange_prefill_costs 时，调用之后每个 rank 的值
        #       都在 ``rank_prefill_costs`` 里。
        # 解释：各参数由调度器的哪个函数算出、三种信号路径（保护期只报有无 / 本地精确探测 / DP 粗估）见文件开头⑤
        # KV + queue-age bounds are gated on this rank actually having prefill to
        # push (firing when this rank can't admit anything would be a no-op).
        # 【译】KV 和排队时长这两类上界以"本 rank 确实有 prefill 要推"为前提（本 rank 什么都准入不了时放行只是空操作）。
        # 解释：这三个标志经过"或"之后会让*所有* rank 放行，所以只让真能做 prefill 的 rank 举手；例如某个没有可准入请求的
        #   rank KV 已经 95%，它举手只会让别的 rank 提前放行，自己却什么也做不了
        low = self.token_usage_low_watermark
        kv_high = prefillable and kv_usage >= self.kv_high_watermark
        kv_low = prefillable and low is not None and kv_usage < low
        queue_hot = (
            prefillable
            and self.max_queue_ms is not None
            and oldest_waiting_age_ms >= self.max_queue_ms
        )

        # 填槽（编码见 __init__ 的槽位表）；槽 7 带上"本 rank 上次决策后跑过 prefill"，让所有 rank 在这一次决策里一起装填 decode 间隔
        self._reduce_buf[0] = 1 if prefillable else 0
        self._reduce_buf[1] = int(pending_tokens)
        self._reduce_buf[2] = int(running_decode_batch)
        self._reduce_buf[3] = 1 if kv_high else 0
        self._reduce_buf[4] = 1 if kv_low else 0
        self._reduce_buf[5] = 1 if has_partial else 0
        self._reduce_buf[6] = 1 if queue_hot else 0
        self._reduce_buf[7] = 1 if self._prefill_executed_since_last_decision else 0
        # one-hot：只在自己的槽填代价、其余清零，SUM 之后每个 rank 都拿到全部 rank 的代价——一次 all_reduce 得到 all_gather 的效果。
        #   代价按 int64 传，小数部分被截掉（代价 = 新 token 数 × (ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT + 已有上下文 + 新 token 数/2)，
        #   默认权重 360000，量级很大，截断可以忽略）
        if self._cost_slots:
            self._reduce_buf[8:] = 0
            self._reduce_buf[8 + self._rank] = int(prefill_cost)
        # 本地模式（cpu_group=None）不通信：缓冲区里就是本 rank 自己的值，下面的"归约值"就是原值
        if self.cpu_group is not None:
            try:
                torch.distributed.all_reduce(
                    self._reduce_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.cpu_group,
                )
            # fail-open：某个 DP rank 进程挂掉时 gloo 的 all_reduce 会抛 RuntimeError。这里不让 schedule() 崩溃，而是放行 prefill，
            #   由 engine_core.py DPEngineCoreProc 每轮的 _sync_dp_state 发现失败、把所有 rank 带进关闭流程
            except RuntimeError:
                logger.warning(
                    "PrefillDelayer all_reduce failed (peer down?); admitting "
                    "prefill and deferring to the DP-state shutdown barrier."
                )
                self._reset()
                self.rank_prefill_costs = []
                return True

        # One readback for all slots (a single .tolist() beats per-slot .item()
        # boundary crossings on this per-tick hot path).
        # 【译】一次读回所有槽（在这条每个 tick 都要走的热路径上，一次 .tolist() 好过逐槽 .item() 多次跨越边界）。
        # 解释：每次 .item() 都是一次 Python ↔ C++ 的调用并新建一个 Python int；.tolist() 一次转完
        reduced = self._reduce_buf.tolist()
        (
            n_prefillable,
            g_pending,
            g_running_dec,
            kv_high_n,
            kv_low_n,
            partial_n,
            queue_hot_n,
            prefill_ran_n,
        ) = reduced[:8]
        self.rank_prefill_costs = [float(c) for c in reduced[8:]]
        # This local execution report has now participated in the global
        # decision. A later successful prefill forward sets it again.
        # 【译】本地的这份执行报告已经参与了这次全局决策。之后再有一次成功的 prefill forward 会重新置位它。
        self._prefill_executed_since_last_decision = False
        any_kv_high = kv_high_n > 0
        any_kv_low = kv_low_n > 0
        any_partial = partial_n > 0
        any_queue_hot = queue_hot_n > 0

        # A prefill that actually ran on the previous tick arms the same
        # countdown on every rank. Delayer FIRE is only permission to try:
        # allocation, remote-KV handling, or queue changes may still produce a
        # decode/empty batch, which must not start a decode-protection window.
        # 【译】上一个 tick 里真正跑过的 prefill，会在每个 rank 上装填同一个倒计时。delayer 的 FIRE 只是"允许尝试"：
        #   分配、远端 KV 的处理或队列变化仍可能让这一步变成 decode 批或空批，那种情况不能开启 decode 保护窗口。
        # 解释：所以装填看的是槽 7（engine_core.py 在含 prefill 的 forward 返回后调 notify_prefill_executed），而不是上一次的 FIRE；
        #   每次又有 prefill 跑过都会重新装满（不会叠加）
        if prefill_ran_n > 0:
            self._decode_interval_remaining = self.prefill_decode_interval

        # First call fires unconditionally (one-time warmup seed).
        # 【译】第一次调用无条件放行（一次性的预热打底）。
        if self._first:
            self._first = False
            self._reset()
            return True

        # Hard post-prefill decode protection. The countdown is identical on
        # all ranks because it is armed from the globally reduced execution
        # signal and schedule() calls this method in lockstep. Do not advance the
        # coalescer's own hold/TTFT episode: SGLang applies the interval before
        # invoking PrefillDelayer, so its delay budget starts afterwards.
        # 【译】prefill 之后的硬 decode 保护。倒计时在所有 rank 上都相同，因为它由全局归约后的执行信号装填，而 schedule()
        #   是锁步地调用本方法的。不要推进合并器自己的 hold / TTFT 扣住期计数：SGLang 在调用 PrefillDelayer 之前就先应用这个
        #   间隔，所以它的延迟预算从间隔之后才开始算。
        # 解释（框 A）：倒计时每次决策都减 1；只有"有人要 prefill 且有 decode"时才 HOLD——没人要 prefill 时往下走到框 B 放行
        #   （空操作），没有 decode 时往下走到框 C① 放行（没有可保护的 decode）。这里的 HOLD 不调 _hold，_hold_ticks 不变，
        #   所以 ttft_max_ticks / partial_max_ticks 都从间隔结束后才开始数
        if self._decode_interval_remaining > 0:
            self._decode_interval_remaining -= 1
            if n_prefillable > 0 and g_running_dec > 0:
                self._stat_hold += 1
                self._stat_hold_decode_interval += 1
                self._maybe_log()
                return False

        # Nothing to prefill anywhere → allow (vacuous), reset the episode.
        # 【译】哪里都没有要 prefill 的 → 放行（空放行），并重置扣住期。
        if n_prefillable == 0:
            self._reset()
            self._stat_fire_vacuous += 1
            self._maybe_log()
            return True

        # ---- must-fire bounds (release even if unaligned / underfilled) ----
        # 【译】---- 必须放行的上界（即使未对齐 / 未填满也放行）----
        # 框 C①：所有 rank 都没有 decode——没有 decode 可以把等待"藏"在后面，扣住只会让 GPU 空转
        if g_running_dec == 0:
            return self._fire("nodecode", g_pending)
        # 框 C②：有可 prefill 的 rank KV 太满（再等也攒不出更大的 batch）或太空（在跑的请求少，GPU 吃不饱）
        if any_kv_high or any_kv_low:
            return self._fire("kv", g_pending)
        # Queue age includes backlog and prior holds. This release applies
        # after decode protection; it is not an end-to-end TTFT guarantee.
        # 【译】排队时长包括积压和之前的扣住。这个放行在 decode 保护之后才生效；它不是端到端 TTFT 的保证。
        # 框 C③：有 rank 最老的可准入等待请求从到达起已排满 max_queue_ms（墙钟），见 __init__ 中它上方的说明
        if any_queue_hot:
            return self._fire("queue_ms", g_pending)
        # 框 C④：本扣住期已连续 HOLD 满 ttft_max_ticks 个 tick（对齐门和合并器的 HOLD 都算）
        if self._hold_ticks >= self.ttft_max_ticks:
            return self._fire("ttft", g_pending)
        # 框 C⑤：有 rank 在 chunked prefill 中途（已占着 KV block），用更紧的 partial_max_ticks
        if any_partial and self._hold_ticks >= self.partial_max_ticks:
            return self._fire("partial", g_pending)

        # ---- alignment gate: never fire while a rank lacks prefill (anti-skew).
        # Hold to let stragglers catch up; the ttft bound above bounds the wait.
        # 【译】---- 对齐门：只要还有 rank 没有 prefill 就不放行（防止错位）。扣住，让落后的 rank 赶上来；上面的 ttft 上界限制了
        #   这段等待。
        # 框 D：本地模式 dp_size=1，而走到这里时 n_prefillable ≥ 1，恒不成立。注意这里 HOLD 不更新框 F 的 stall 计数
        if n_prefillable < self.dp_size:
            return self._hold(g_pending)

        # ---- goal: fire once the aggregate fills a worthwhile forward ----
        # 【译】---- 目标：聚合量填满一次值得做的 forward 就放行 ----
        # 框 E：分母 = 可 prefill 的 rank 数 × 每 rank 的满预算（走到这里时就是 dp_size 个）；budget > 0 的判断只是防御
        #   例：DP=4、预算 16384 → 分母 65536，target_fill=0.9 要求 G_pending ≥ 58983；本地模式要求 ≥ 14746
        budget = n_prefillable * self.max_num_batched_tokens
        fill = g_pending / budget if budget > 0 else 1.0
        if fill >= self.target_fill:
            return self._fire("fill", g_pending)

        # ---- adaptive give-up: queue stopped growing → waiting is futile ----
        # 【译】---- 自适应放弃：队列不再增长 → 再等也没用 ----
        # 框 F："不增长"包括减少（例如有请求被 abort）；_prev_pending 在每次 FIRE 时重置为 -1，所以每个扣住期第一次走到这里
        #   一定算"增长"（stall=0），之后要连续 stall_ticks 个不增长的 tick 才放行——②的例子里正好是第 11 次走到这里
        if g_pending <= self._prev_pending:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._prev_pending = g_pending
        if self._stall_count >= self.stall_ticks:
            return self._fire("stall", g_pending)

        return self._hold(g_pending)

    # FIRE 的统一出口（框 C ①~⑤、E、F）：按原因计数、可选的调试日志、结束本扣住期（_reset），返回 True
    def _fire(self, reason: str, g_pending: int) -> bool:
        # reason ∈ {fill, stall, ttft, kv, partial, nodecode, vacuous}; each maps
        # to a dedicated _stat_fire_<reason> slot so the log shows what released.
        # 【译】reason ∈ {fill, stall, ttft, kv, partial, nodecode, vacuous}；每个原因对应一个专门的 _stat_fire_<reason> 计数槽，
        #   这样日志能看出是什么放行的。
        # 观察：实际传进来的 reason 是 nodecode、kv、queue_ms、ttft、partial、fill、stall；注释列表里漏了 queue_ms，
        #   而 vacuous 从不经过 _fire（框 B 在 should_allow_prefill 里直接给 _stat_fire_vacuous 加 1）
        # 解释：f"_stat_fire_{reason}" 拼出属性名，getattr / setattr 按名字读写；属性必须已在 __slots__ 里，拼错会抛 AttributeError
        attr = f"_stat_fire_{reason}"
        setattr(self, attr, getattr(self, attr) + 1)
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] FIRE ({reason}): g_pending={g_pending} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count}"
            )
        self._reset()
        self._maybe_log()
        return True

    def notify_prefill_executed(self) -> None:
        """Report a completed local prefill forward.

        The next lockstep decision folds this flag into the existing SUM
        reduction, so every DP rank arms the interval together without adding
        a second collective to the scheduler hot path.
        """
        # 【译】报告本地完成了一次 prefill forward。
        #
        #   下一次锁步决策会把这个标志并入已有的 SUM 归约，于是每个 DP rank 一起装填间隔，而不必在调度器的热路径上再加一次集合通信。
        # 解释：调用方 engine_core.py EngineCore._process_engine_step_inner：本批 total_seqs_num_prefill > 0、
        #   call_func("forward", ...) 返回之后。"完成"指 forward 调用返回；开延迟输出时（默认）GPU 可能还在执行这一步
        #   （见 engine_core.py 文件开头【3】），这不影响按 tick 计数。DP 下空闲 rank 跑的 dummy batch 不经过这里
        self._prefill_executed_since_last_decision = True

    # HOLD 的出口（框 D、框 F 之后）：本扣住期长度 +1（框 C④⑤ 据此判断），返回 False；框 A 的 HOLD 不走这里
    def _hold(self, g_pending: int) -> bool:
        self._hold_ticks += 1
        self._stat_hold += 1
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] HOLD: g_pending={g_pending} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count} "
                f"target_fill={self.target_fill}"
            )
        self._maybe_log()
        return False

    # 结束一个扣住期：FIRE（_fire、框 B）、第一次调用、all_reduce 失败时调用；不动 decode 间隔倒计时和统计
    def _reset(self) -> None:
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1

    # 周期性统计日志：决策总数恰好是 _stat_log_every 的倍数时打印一行（第一次调用的那次 FIRE 不计入 total）
    def _maybe_log(self) -> None:
        # Cheap guard first — skip the counter sum entirely when logging is
        # disabled (this runs on every FIRE/HOLD tick).
        # 【译】先做最便宜的判断——日志关闭时完全跳过计数器求和（这在每个 FIRE/HOLD 的 tick 上都会执行）。
        if self._stat_log_every <= 0:
            return
        total = (
            self._stat_fire_fill
            + self._stat_fire_stall
            + self._stat_fire_ttft
            + self._stat_fire_kv
            + self._stat_fire_partial
            + self._stat_fire_nodecode
            + self._stat_fire_queue_ms
            + self._stat_fire_vacuous
            + self._stat_hold
        )
        if total == 0:
            return
        if total % self._stat_log_every == 0:
            logger.info(
                f"[PrefillDelayer stats] total={total} "
                f"fire_fill={self._stat_fire_fill} "
                f"fire_stall={self._stat_fire_stall} "
                f"fire_ttft={self._stat_fire_ttft} "
                f"fire_kv={self._stat_fire_kv} "
                f"fire_partial={self._stat_fire_partial} "
                f"fire_nodecode={self._stat_fire_nodecode} "
                f"fire_queue_ms={self._stat_fire_queue_ms} "
                f"fire_vacuous={self._stat_fire_vacuous} "
                f"hold_decode_interval={self._stat_hold_decode_interval} "
                f"hold={self._stat_hold} "
                f"(hold_rate={self._stat_hold / total:.2%})"
            )
