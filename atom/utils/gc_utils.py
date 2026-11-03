# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Garbage-collector policy for the serving processes.

CPython's gen-2 pass is stop-the-world and traverses every tracked container,
so its cost tracks the live heap -- which here is almost all startup state
(model, compiled graph, tokenizer, KV block pool) that is never garbage.
Measured on DeepSeek-V4-Flash-DSpark tp1: 268 ms in the EngineCore, 979 ms in a
ModelRunner worker, 265 ms in the API server, each reclaiming 0 objects.

That is the argument for freezing rather than tuning thresholds: the collector
is not merely slow here, it finds nothing. Everything the hot loop allocates is
acyclic, so reference counting takes it.
"""

# 【译】服务进程的垃圾回收（GC）策略。
#
#   CPython的第2代回收会暂停整个解释器（stop-the-world），并遍历每一个被跟踪的容器对象，所以它的耗时与存活的堆大小成正比——
#   而这里的堆几乎全是启动时建立、永远不会变成垃圾的状态（模型、编译后的图、tokenizer、KV block池）。
#   在DeepSeek-V4-Flash-DSpark tp1上实测：EngineCore里268 ms，ModelRunner worker里979 ms，API server里265 ms，
#   每次都回收了0个对象。
#
#   这就是选择"冻结"而不是调阈值的理由：回收器在这里不只是慢，而是什么也找不到。热循环里分配的所有东西都没有循环引用，
#   引用计数就能回收它们。

import gc
import logging
import time
import types

logger = logging.getLogger("atom")


# 本文件在系统中的位置：每个服务进程（API server、EngineCore、每个ModelRunner worker、atomesh）在启动时调用
#   tune_gc()                      按ATOM_GC_THRESHOLD调整GC阈值（默认不设置，什么也不做）
#   maybe_attach_gc_debug_callback ATOM_GC_DEBUG=1时打印每次GC的耗时
#   freeze_gc_heap()               启动完成、开始接流量之前冻结当前堆（ATOM_GC_FREEZE默认开）
#     EngineCore：_freeze_after_startup先冻结自己，再call_func("freeze_gc_heap")让每个ModelRunner冻结
#   unfreeze_gc_heap()             EngineCore.exit时解冻
#   arm_reclaim_watch / reclaim_watch / gc_census   API server用来观察GC（/debug/gc_census接口）
#
# Python GC基础：CPython主要靠引用计数回收对象，分代GC只负责找"循环引用"的垃圾。新对象在第0代，
#   熬过一次回收就升一代；CPython 3.12及以前gc.get_threshold()默认是(700, 10, 10)，含义是第0代新增约700个对象触发一次第0代回收，
#   每10次第0代回收触发一次第1代，以此类推。第2代回收要扫描所有老对象，堆越大越慢
#   （CPython 3.13的默认值是(2000, 10, 10)）。第0代/第1代回收只扫描年轻对象，很便宜；贵的是第2代
#   gc.freeze()（Python 3.7+）把当前所有被跟踪的对象移进"永久代"，之后的回收都不再扫描它们
# 对比nano-vllm：nano-vllm没有任何GC调优
#
# 为什么一次GC停顿会直接变成尾延迟：
#   EngineCore一个线程驱动整个step循环，它在第2代回收里停住的这段时间不调度、不下发forward，所有TP worker都阻塞在
#     dequeue上空等；worker进程自己停住时，其他rank会卡在下一次all-reduce上等它（SPMD，见engine_core.py文件开头【4】）
#   时间线（decode每步10ms，用模块docstring里EngineCore实测的268ms）：
#     step:  |10ms|10ms|10ms|<-------- gen-2 GC 268ms -------->|10ms|10ms|
#     这一刻在途的每个请求都多等了约27步的时间：平均ITL几乎不变，P99/P999 ITL却被这一次停顿决定
#   而且触发时刻不受控：分配最密集的时候（一波prefill涌入、大量请求同时结束）最容易触发第2代回收，恰好是负载最高的时候
#   冻结的效果：被冻结的启动期对象（模型、编译后的图、KV block池……）不再参与任何一代的扫描，第2代只剩之后新分配、
#     熬过两轮回收的对象，一次第2代回收的耗时随之降到与"在途请求的状态"成正比，而不是与整个堆成正比


# 按ATOM_GC_THRESHOLD（如"50000,20,100"）设置本进程的GC阈值；值不合法时恢复原值并告警
def tune_gc() -> None:
    """Raise this interpreter's collection thresholds from `ATOM_GC_THRESHOLD`.

    Per-interpreter, so every process that serves calls it for itself -- an
    enumeration here has gone stale twice, so the rule is the documentation and
    `tests/test_gc_utils.py` is what checks it.

    No default: raising these spaces passes out without making one cheaper, so
    the same total scan lands in fewer, longer stop-the-world pauses -- a trade
    against tail latency that nothing here has measured. `freeze_gc_heap` is
    the lever that removes work rather than rescheduling it.
    """
    # 【译】按 ATOM_GC_THRESHOLD 调高本解释器的回收阈值。
    #
    #   阈值是每个解释器各自的，所以每个提供服务的进程都要自己调用它——这里曾两次列举调用者而列表过时了，
    #   所以规则本身就是文档，由 tests/test_gc_utils.py 来检查。
    #
    #   没有默认值：调高这些阈值只是把回收拉开间隔，并不让哪一次更便宜，所以同样的总扫描量会落在更少、更长的全停顿里——
    #   这是一种针对尾延迟的权衡，而这里没有任何测量支撑它。freeze_gc_heap 才是"消除工作"而不是"重新安排工作"的那根杠杆。
    # 在函数内导入envs；envs的值每次访问都重新读环境变量（见第1批utils/envs.py）
    from atom.utils import envs

    thresholds = envs.ATOM_GC_THRESHOLD
    if not thresholds:
        return
    old = gc.get_threshold()
    # Ignoring a bad value has to cover applying it too: this runs unguarded in
    # four processes, and in a ModelRunner worker a raise here lands between
    # "load model runner success" and "ready", where nothing reports it.
    # `OverflowError` is the one `set_threshold` actually raises, and it is an
    # ArithmeticError, so the other two do not cover it.
    # 【译】"忽略一个坏值"也必须覆盖"应用它"这一步：这段代码在四个进程里不加保护地运行，在ModelRunner worker里
    #   这里一旦抛异常，就落在"load model runner success"和"ready"之间，没有任何东西会报告它。
    #   OverflowError 是 set_threshold 真正会抛的那一种，而它是ArithmeticError，另外两种异常覆盖不到它。
    try:
        # 例："50000,20,100" → (50000, 20, 100)；个数必须与gc.get_threshold()一致（3个）
        t = tuple(int(x) for x in thresholds.split(","))
        if len(t) != len(old):
            raise ValueError(f"want {len(old)} values, got {len(t)}")
        # CPython accepts both of these and neither means what it looks like.
        # `t[0] == 0` stops automatic collection (measured: nothing reclaimed
        # over 50k cycles) while `reclaim_watch` and `atom:gc_collected` read
        # like the healthy case. The other two are ratios, where zero means
        # *more* collection, so only a negative one is meaningless.
        # 【译】CPython接受下面这两种值，但它们都不是看上去的意思。`t[0] == 0` 会停止自动回收（实测：5万个循环里什么都没回收），
        #   而 reclaim_watch 和 atom:gc_collected 读起来却像健康的情况。另外两个是比率，0意味着*更多*回收，
        #   所以只有负数才是无意义的。
        if t[0] < 1 or min(t) < 0:
            raise ValueError(f"t0 must be >= 1 and none negative, got {t}")
        gc.set_threshold(*t)
    except (ValueError, TypeError, OverflowError) as exc:
        # `set_threshold` converts and stores one argument at a time, so an
        # overflow on the second or third leaves the earlier ones applied:
        # `1,2**63,10` lands on `(1, 10, 10)` and logs that it ignored the
        # value. Restoring is what makes "ignored" true, whatever the reason --
        # and it is a no-op on the paths that never reached the call.
        # 【译】set_threshold 是逐个转换并保存参数的，所以第二个或第三个参数溢出时，前面的已经生效了：
        #   `1,2**63,10` 会落在 `(1, 10, 10)`，同时日志却说忽略了这个值。恢复原值才让"忽略"成为事实，不管原因是什么——
        #   而在还没走到那次调用的路径上，它是一个空操作。
        gc.set_threshold(*old)
        logger.warning("[gc] bad ATOM_GC_THRESHOLD=%r (%s), ignored", thresholds, exc)
        return
    logger.info("[gc] thresholds %s -> %s", old, gc.get_threshold())


# Whether this process's collector finds anything at all -- the number that
# decides whether spacing its passes out would be free, and the one thing about
# it that can change without anyone noticing. Baseline after the freeze,
# because everything before it reclaimed plenty.
# 【译】这个进程的回收器到底有没有找到任何东西——这个数决定了把它的回收拉开间隔是不是免费的，也是它身上唯一可能在
#   没人注意时发生变化的东西。基线取在冻结之后，因为在那之前回收的东西很多。
# 模块级全局变量：基线（各代累计回收数）和"是否已经告警过"
_reclaim_baseline: tuple[int, ...] | None = None
_reclaim_warned = False


# 记下各代目前为止的累计回收数，作为之后比较的基线；调用方：API server启动时（冻结之后）
def arm_reclaim_watch() -> None:
    """Record what has been reclaimed so far, so later growth is visible.

    Called once, after `freeze_gc_heap`. Before that point the counters carry
    startup's collections -- 18,826 objects in one measured run -- and a watch
    armed there would report that as steady-state growth on its first check.
    """
    # 【译】记录到目前为止已回收了多少，使之后的增长可见。
    #
    #   只调用一次，在 freeze_gc_heap 之后。在那之前计数器里带着启动阶段的回收——一次实测为18,826个对象——
    #   如果在那里布置观察，第一次检查就会把它报告成稳态增长。
    # global：在函数内给模块级变量赋值
    global _reclaim_baseline, _reclaim_warned
    _reclaim_baseline = tuple(s["collected"] for s in gc.get_stats())
    _reclaim_warned = False


# 自基线以来有几代回收到了东西（第一次发现时告警一次）；调用方：API server的定时任务
def reclaim_watch(context: str) -> int:
    """How many generations have reclaimed something since the watch was armed.

    Zero is the expected answer for the API server, measured over a full GSM8K
    run at its thresholds. A non-zero answer is not a fault: it says this
    process builds reference cycles, which is what would make spacing its
    collections out cost something rather than nothing.

    Blind to one case, and it is the one that matters most: a cycle promoted to
    gen-2 before it becomes garbage is reclaimed only by a gen-2 pass, so where
    those are rare it is neither collected nor counted here. `/debug/gc_census`
    reports the gen-2 size, which is what would show it.

    Warns once -- at four thousand streams a repeating line buries the log.
    `atom:gc_collected_total` is the continuous signal; this only makes someone
    look.
    """
    # 【译】自从布置观察以来，有多少代回收到了东西。
    #
    #   对API server而言，期望的答案是0，这是在它的阈值下跑完整个GSM8K测出来的。非零答案不是故障：它说明这个进程会构造
    #   循环引用，而这会让"把回收拉开间隔"变得有代价，而不是毫无代价。
    #
    #   它对一种情况是盲的，而那恰恰是最重要的情况：一个在变成垃圾之前就被提升到第2代的循环，只会被第2代回收清理，
    #   所以在第2代回收很少发生的地方，它既不会被回收，也不会在这里被计数。/debug/gc_census 报告第2代的大小，那才能看出它。
    #
    #   只告警一次——在四千条流时，重复的日志行会把日志淹没。atom:gc_collected_total 才是连续的信号；这里只是让人去看一眼。
    global _reclaim_warned
    if _reclaim_baseline is None:
        return 0
    # gc.get_stats()：每代一个dict，"collected"是该代累计回收的对象数
    now = tuple(s["collected"] for s in gc.get_stats())
    grew = [g for g, (was, is_) in enumerate(zip(_reclaim_baseline, now)) if is_ > was]
    if grew and not _reclaim_warned:
        _reclaim_warned = True
        logger.warning(
            "[gc] %s: generations %s reclaimed objects since startup (%s -> %s). "
            "This process builds reference cycles, so raising its "
            "ATOM_GC_THRESHOLD would defer real work rather than nothing. See "
            "atom:gc_collected_total.",
            context,
            grew,
            _reclaim_baseline,
            now,
        )
    return len(grew)


# 冻结当前堆：先完整回收一次，再gc.freeze()；返回永久代里的对象总数（ATOM_GC_FREEZE关闭时返回0）
# 调用方：EngineCore._freeze_after_startup、ModelRunner.freeze_gc_heap（经RPC）、API server、atomesh
def freeze_gc_heap(context: str) -> int:
    """Move everything alive now into the permanent generation, which
    collections skip. Call once, between "startup done" and "traffic starts".
    Returns the total frozen count.

    Collecting first is not optional: `gc.freeze()` takes every generation as
    it finds it, so current garbage would be made permanently unreclaimable.
    One full pass suffices -- `gc.collect()` covers all three generations, and
    freezing does not care which one an object ended up in.

    Objects created afterwards are still tracked and collected, so this is not
    `gc.disable()` -- a cycle written by later code is still caught. What it
    forfeits is anything alive *now* that later becomes garbage, which in these
    processes outlives the process anyway. `unfreeze_gc_heap` covers the one
    case where that is false: tearing an engine down in-process.
    """
    # 【译】把此刻存活的所有对象移进永久代，之后的回收会跳过它们。在"启动完成"和"流量开始"之间调用一次。
    #   返回冻结对象的总数。
    #
    #   先回收不是可选的：gc.freeze() 按原样接收每一代，所以当前的垃圾会被变成永远无法回收。一次完整回收就够了——
    #   gc.collect() 覆盖全部三代，而冻结并不关心一个对象最后落在哪一代。
    #
    #   之后创建的对象仍会被跟踪和回收，所以这不是 gc.disable()——之后的代码写出的循环仍然会被抓住。它放弃的是此刻存活、
    #   之后才变成垃圾的那些东西，而在这些进程里它们本来就会活得比进程还久。unfreeze_gc_heap 覆盖了这一点不成立的唯一情况：
    #   在进程内拆掉一个引擎。
    from atom.utils import envs

    if not envs.ATOM_GC_FREEZE:
        return 0
    gc.collect()
    before = gc.get_freeze_count()
    gc.freeze()
    total = gc.get_freeze_count()
    logger.info(
        "[gc] %s: froze %d objects (%d already frozen)", context, total - before, before
    )
    return total


# 解冻：把永久代交还给回收器；调用方：EngineCore.exit
def unfreeze_gc_heap() -> None:
    """Hand the permanent generation back. Required on engine shutdown: a
    frozen object is invisible to the collector, so an engine destroyed inside
    a live interpreter would leave its weights and KV cache unreachable *and*
    uncollectable, which presents as a GPU memory leak."""
    # 【译】交还永久代。引擎关闭时必须调用：被冻结的对象对回收器不可见，所以在一个仍存活的解释器里销毁的引擎，
    #   会让它的权重和KV cache既不可达*又*不可回收，表现为GPU显存泄漏。
    frozen = gc.get_freeze_count()
    if frozen:
        logger.info("[gc] unfroze %d objects", frozen)
    gc.unfreeze()


# ATOM_GC_DEBUG=1时，注册一个GC回调，打印每次回收的代数、耗时、回收数和被跟踪对象数
def maybe_attach_gc_debug_callback(context: str) -> None:
    """Under ``ATOM_GC_DEBUG``, log every collection.

    Deliberately expensive: it counts the tracked set on each pass, which cost
    ~90s of extra startup on a V4-Flash tp1. It is also the only way to see
    these pauses -- a stall in the EngineCore idles the workers, and an idle
    worker emits no trace event at all.
    """
    # 【译】在 ATOM_GC_DEBUG 下，记录每一次回收。
    #
    #   故意做得很昂贵：每次回收都要数一遍被跟踪的对象集合，在V4-Flash tp1上让启动多花了约90秒。它也是看到这些停顿的唯一办法——
    #   EngineCore里的一次停顿会让worker空闲，而空闲的worker根本不会产生任何trace事件。
    from atom.utils import envs

    if not envs.ATOM_GC_DEBUG:
        return

    # 闭包里用dict保存"开始时间"和"开始时的对象数"，start和stop两次回调之间共享
    started: dict[str, float | int] = {"at": 0.0, "tracked": 0}

    # gc.callbacks里的函数在每次回收开始（phase="start"）和结束（phase="stop"）时被调用，info里带generation、collected等
    def _log(phase: str, info: dict) -> None:
        gen = info.get("generation")
        if gen is None:
            return
        if phase == "start":
            started["at"] = time.perf_counter()
            started["tracked"] = len(gc.get_objects(gen))
            return
        logger.info(
            "[gc] %s: gen-%d took %.2f ms, reclaimed %s of %d tracked",
            context,
            gen,
            (time.perf_counter() - started["at"]) * 1e3,
            info.get("collected", "?"),
            started["tracked"],
        )

    gc.callbacks.append(_log)
    logger.info("[gc] %s: debug callback attached", context)


# ---------- 以下是gc_census（/debug/gc_census）的辅助函数 ----------
# 计数最多的n个键，渲染成"键 x次数"
def _most_common(counts: dict[str, int], n: int) -> list[str]:
    """The `n` most frequent keys, rendered `key xCOUNT`."""
    # 【译】出现最频繁的 n 个键，渲染成 `key xCOUNT`。
    return [f"{k} x{c}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


# (label, path marker, module prefix). Ordered: the first hit wins, so a more
# specific entry must come before any prefix of it.
# 【译】(标签, 路径标记, 模块前缀)。有顺序：第一个命中的胜出，所以更具体的条目必须排在它的任何前缀之前。
_OWNERS = tuple(
    (name, f"/{name}/", name)
    for name in (
        "atom",
        "fastapi",
        "starlette",
        "uvicorn",
        "h11",
        "anyio",
        "pydantic",
        "msgspec",
        "transformers",
    )
)

# Exact type -> where its code object lives. Keyed on `type(obj)` rather than
# probed with `getattr`, which would run the object's `__getattr__` and its
# descriptors -- for a census, executing the code it is measuring.
# 【译】精确类型 -> 它的code对象放在哪个属性上。用 type(obj) 作为键，而不是用 getattr 去探测，因为后者会运行对象的
#   __getattr__ 及其描述符——对一次普查来说，那等于执行它正在测量的代码。
_CODE_ATTR = {
    types.CoroutineType: "cr_code",
    types.AsyncGeneratorType: "ag_code",
    types.GeneratorType: "gi_code",
    types.FrameType: "f_code",
    types.FunctionType: "__code__",
}
# 无法确定归属的普通容器类型
_PLAIN_CONTAINERS = frozenset({tuple, list, dict, set, frozenset})


# 判断一个对象来自哪个库：带代码的对象看代码所在文件，其他实例看类的模块
def _owner(obj: object) -> str:
    """Which library this object came from, or why it cannot be said.

    Code-carrying objects go by the file their code compiled from, instances by
    their class's module.

    Plain containers have no owner: a `tuple` of two cells belongs to whoever
    built the closure, and that is not recoverable from the tuple. They are
    reported as `unattributable` rather than dropped -- a breakdown missing a
    third of the set would read as precision it does not have.
    """
    # 【译】这个对象来自哪个库，或者为什么说不出来。
    #
    #   带代码的对象按它们的代码编译自哪个文件来判断，实例按它们的类所在的模块来判断。
    #
    #   普通容器没有归属：一个装着两个cell的 tuple 属于构造了那个闭包的人，而这无法从tuple本身还原。它们被报告为
    #   unattributable 而不是被丢掉——一个缺了三分之一的分类会读起来像是有它并不具备的精确度。
    try:
        kind = type(obj)
        if kind is types.MethodType:
            # 【译】（行尾注释）代码在函数上
            obj, kind = obj.__func__, types.FunctionType  # code lives on the function
        attr = _CODE_ATTR.get(kind)
        if attr is not None:
            path = getattr(getattr(obj, attr, None), "co_filename", "")
            for name, marker, _ in _OWNERS:
                if marker in path:
                    return name
            return "site-other" if "/site-packages/" in path else "stdlib"
        if kind in _PLAIN_CONTAINERS:
            return "unattributable"
        module = kind.__module__ or ""
        # 【译】（行尾注释）不用startswith："atom"不能吞掉"atomic"
        root = module.split(".", 1)[0]  # not startswith: "atom" must not eat "atomic"
        for name, _, prefix in _OWNERS:
            if root == prefix:
                return name
        return f"module:{root}" if root else "unattributable"
    # （行尾的noqa说明：普查不能因为一个奇怪的对象而失败）
    except Exception:  # noqa: BLE001 - a census must not fail on one odd object
        return "error"


# 对第2代对象做一次普查：按类型、按来源库统计数量。调用方：API server的/debug/gc_census（放在线程池里跑）
# 输出：dict，含各代大小、冻结数、阈值、gc统计、按类型和按来源的排行
def gc_census(top: int = 30, types_per_owner: int = 2) -> dict:
    """What the collector actually rescans, broken down by type.

    The generation sizes say the collector is expensive; this says what it is
    expensive *on*, which is the only form of the answer you can act on. Read
    against `freeze_gc_heap`: everything counted here was allocated after the
    freeze, so at a serving concurrency it is per-request state, and dividing
    by the in-flight request count gives the per-stream object cost that
    thresholds do not change.

    On demand and never on a timer: it walks every tracked object, which takes
    roughly a second at a million of them, so the caller runs it off the event
    loop. That is also why `gc.get_stats()` is included -- those counters are
    free, so a caller that only needs "is anything being reclaimed" should read
    them and not this.

    Types and owners only. An earlier version also fingerprinted plain
    containers by their contents, which on this process meant serialising the
    keys of parsed request bodies into the response; naming what a dict holds
    is exactly the thing that cannot be reported from a live serving heap.

    What is absent is as informative as what is here. A dict or tuple whose
    contents are all atomic is *untracked* by the collector on its first pass,
    so it never appears and never costs a scan; one holding a list or an
    instance stays. Per-request state built out of scalars is therefore free,
    and the same state with one list in it is not -- which is the difference
    this census exists to find.
    """
    # 【译】回收器实际重复扫描的是什么，按类型分解。
    #
    #   各代的大小说明回收器很贵；这里说明它贵在*什么*上，而这才是你能据以行动的答案形式。要对照 freeze_gc_heap 来读：
    #   这里计到的所有东西都是在冻结之后分配的，所以在服务并发下它们是逐请求的状态，除以在途请求数就得到阈值改变不了的
    #   每条流的对象开销。
    #
    #   按需调用，从不定时：它要遍历每一个被跟踪的对象，在一百万个对象时大约要一秒，所以调用方把它放在事件循环之外跑。
    #   这也是为什么包含了 gc.get_stats()——那些计数器是免费的，只需要知道"有没有东西被回收"的调用方应该读它们而不是这个。
    #
    #   只给类型和归属。早先的版本还会按内容给普通容器做指纹，在这个进程上那意味着把解析后的请求体的键序列化进响应里；
    #   说出一个dict里装了什么，正是不能从一个正在服务的堆里报告的东西。
    #
    #   缺席的东西和在场的东西一样有信息量。一个内容全是原子值的dict或tuple，在第一次被扫描时就会被回收器*取消跟踪*，
    #   所以它永远不会出现，也永远不花扫描成本；而装着一个list或一个实例的则会留下。所以由标量构成的逐请求状态是免费的，
    #   而带一个list的同样状态就不是——这正是这次普查要找出的差别。
    # Slicing is silent about a nonsensical bound: `[:0]` returns nothing and
    # `[:-1]` drops the smallest row, both of which read as a real answer.
    # 【译】切片对无意义的边界不出声：`[:0]` 什么也不返回，`[:-1]` 会丢掉最小的一行，两者读起来都像真实的答案。
    top = max(1, top)
    types_per_owner = max(0, types_per_owner)
    # 只普查第2代：冻结之后仍留在这里的，就是每次第2代回收要扫描的对象
    gen2 = gc.get_objects(2)

    by_type: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    owner_types: dict[str, dict[str, int]] = {}
    for obj in gen2:
        name = type(obj).__qualname__
        by_type[name] = by_type.get(name, 0) + 1
        who = _owner(obj)
        by_owner[who] = by_owner.get(who, 0) + 1
        per_owner = owner_types.setdefault(who, {})
        per_owner[name] = per_owner.get(name, 0) + 1
    ranked = sorted(by_type.items(), key=lambda kv: -kv[1])[:top]

    return {
        # `len(gen2)` rather than a third `get_objects(2)`: that call
        # materialises the whole generation, and this one is already held.
        # 【译】用 len(gen2) 而不是第三次调用 get_objects(2)：那个调用会把整个代物化出来，而这一代已经拿在手里了。
        "generations": {
            "0": len(gc.get_objects(0)),
            "1": len(gc.get_objects(1)),
            "2": len(gen2),
        },
        # Walks the permanent generation, so it belongs to a caller that is
        # already walking. Nothing on a scrape path may ask for it.
        # 【译】它遍历永久代，所以属于已经在遍历的调用方。抓取（scrape）路径上的任何东西都不能请求它。
        "frozen": gc.get_freeze_count(),
        "thresholds": list(gc.get_threshold()),
        # `collected` is the one that decides whether raising thresholds is
        # free here or is deferring real work.
        # 【译】collected 是那个决定"调高阈值在这里是免费的，还是在推迟真正的工作"的数。
        "stats": gc.get_stats(),
        "by_type": [{"type": n, "count": c} for n, c in ranked],
        "by_owner": [
            {
                "owner": who,
                "count": c,
                "types": _most_common(owner_types[who], types_per_owner),
            }
            for who, c in sorted(by_owner.items(), key=lambda kv: -kv[1])
        ],
    }
