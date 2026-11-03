"""Python-owned ATOM standalone serving logic.

This adapter keeps OpenAI-compatible request semantics in Python so the Rust
standalone router only needs to bridge requests and responses.
"""

# 【译】由Python持有的ATOM standalone服务逻辑。
#
#   这个适配器把与OpenAI兼容的请求语义留在Python里，这样Rust的standalone路由器只需要在两边之间转接请求和响应。
#
# 【导读】atomesh standalone模式下真正干活的Python服务对象（atomesh是什么、与api_server.py的区别，见本目录 server.py 的文件头）
# 调用方：server.py 的 initialize_standalone_service 构造 AtomStandaloneService，交给Rust的 launch_mesh；
#   之后Rust的 AtomStandaloneRouter（atom/mesh/src/routers/atom_standalone.rs）经PyO3按名字调用它的方法：
#     非流式：chat_completions / completions，同步调用，返回完整的响应dict
#     流式：start_*_stream 返回stream_id -> Rust在一个阻塞线程里反复调用 drain_*_stream(stream_id, 16, 0.05) 取SSE帧，
#       空列表就继续轮询，拿到 "data: [DONE]" 后调用 close_*_stream；服务退出时调用 close
# 与主干的关系：它是 api_server.py 的"同步、轮询式"对应物，复用同一批构件——chat_encoders（chat template渲染）、
#   reasoning（推理内容分离）、tool_parser（工具调用解析）、serving_chat / serving_completion（拼响应和chunk）、
#   以及 api_server 的 _build_sampling_params / _coerce_n
#   不同之处：api_server 走 LLMEngine.generate_async（asyncio）；这里没有事件循环，由一个后台工作线程直接调用
#   engine.io_processor.preprocess（建Sequence）和 engine.core_mgr.add_request（交给CoreManager，本批），
#   每条序列挂一个回调；回调在CoreManager的输出线程（engine_core_mgr.py 的 _create_output_thread）里被调用，
#   把结果写进Future（非流式）或stream_queue（流式）
# 本文件的类（自底向上）：
#   AtomEngineService：唯一直接和引擎打交道的部分——请求队列 + 工作线程，以及流中止（abort）的簿记
#   SingleRequestState / FanoutRequestState：非流式请求（n=1 / n>1）的结果累加器，全部结束时完成Future
#   StreamRequestState：流式请求的回调，把每批新token解码后作为事件放进stream_queue
#   ChatCompletionStreamState / CompletionStreamState：drain时从stream_queue取事件，变成OpenAI格式的SSE chunk
#   AtomStandaloneService：对Rust暴露的门面，负责请求校验、chat template、采样参数和流注册表
# 阅读顺序（约1950行）：先读门面 AtomStandaloneService 的 chat_completions（非流式）和
#   start_chat_completions_stream / drain_chat_completions_stream（流式），再顺着调用往下：
#   AtomEngineService.generate / start_stream -> _worker_loop -> SingleRequestState / StreamRequestState.record
#   -> ChatCompletionStreamState.drain；completions 那一套与 chat 同构，可以跳过
# stream_queue里的事件有三种：{"index", "text", "token_ids", "finished", "finish_reason"[, "kv_transfer_params"]}、
#   {"error": 信息}、{"done": True}（所有兄弟序列都结束）

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import numbers
import queue
import threading
import time
import uuid
from typing import Any

from atom import SamplingParams
from atom.entrypoints.openai.api_server import _build_sampling_params, _coerce_n
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    STREAM_DONE_MESSAGE,
    TEXT_COMPLETION_OBJECT,
    CompletionRequest,
    openai_stop_reason,
    openai_stop_reason_with_calls,
)
from atom.entrypoints.openai.reasoning import (
    NO_REASONING,
    ReasoningChannel,
    prompt_starts_in_reasoning,
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    create_chat_chunk,
)
from atom.entrypoints.openai.serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    create_completion_chunk,
)
from atom.entrypoints.openai.sse import data_frame
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.registry import (
    forbids_tool_calls,
    resolve_tool_call_parser,
)
from atom.model_engine.sequence import new_token_ids

logger = logging.getLogger("atom")


# 非流式请求：交给工作线程的一项任务，结果通过future返回给阻塞等待的调用线程
@dataclasses.dataclass
class EngineRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    future: concurrent.futures.Future[list[dict[str, Any]]]
    kv_transfer_params: dict[str, Any] | None = None
    data_parallel_rank: int | None = None


# 流式请求：结果事件写进stream_queue，由drain_*_stream取走
@dataclasses.dataclass
class EngineStreamRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    stream_queue: queue.Queue[dict[str, Any]]
    kv_transfer_params: dict[str, Any] | None = None
    data_parallel_rank: int | None = None


# 引擎访问层：所有请求先放进_queue，由唯一的工作线程（_worker_loop）逐个做预处理并add_request给CoreManager
class AtomEngineService:
    def __init__(self, engine: Any, tokenizer: Any) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._queue: queue.Queue[EngineRequest | EngineStreamRequest | None] = (
            queue.Queue()
        )
        self._closed = threading.Event()
        self._active_futures: set[concurrent.futures.Future[list[dict[str, Any]]]] = (
            set()
        )
        self._active_futures_lock = threading.Lock()
        # 下面三个结构都是流中止的簿记，共用 _stream_seqs_lock：
        #   _stream_seqs：已提交给引擎的流 -> 它的序列列表；_abandoned：提交前就被关闭的流；_awaiting_submit：已入队、待提交的流
        # The engine sequences behind each open stream, so closing one can
        # stop it. Nothing here used to: `close_*_stream` popped the Python
        # state and the sequence kept decoding to `max_tokens` on the GPU,
        # putting into a queue no one would ever read again and holding its
        # KV blocks for the life of the process. The OpenAI server does this
        # from a generator's `finally`; this service is polled, so it has to
        # be done where the caller says it is done.
        # 【译】每个打开的流背后的引擎序列，这样关闭一个流时能停掉它们。以前这里什么都不做：close_*_stream 弹出Python侧的状态，
        #   序列却在GPU上继续decode到max_tokens，往一个再也没人读的队列里写，并在进程的整个生命周期内占着它的KV block。
        #   OpenAI服务在生成器的 finally 里做这件事；本服务是被轮询的，所以必须在调用方声明结束的地方做。
        self._stream_seqs: dict[str, list[Any]] = {}
        # Streams closed before the worker thread got round to submitting
        # them. `start_stream` only enqueues; the sequences do not exist until
        # `_submit_stream_request` runs, so a close in between had nothing to
        # abort and the sequence was added to the engine *afterwards* -- the
        # exact case the abort exists for, and the one it missed.
        # 【译】在工作线程来得及提交之前就被关闭的流。start_stream 只负责入队；直到 _submit_stream_request 运行之前序列都不存在，
        #   所以在这之间到来的关闭没有东西可中止，而序列随后又被加进了引擎——这恰恰是中止机制要处理的情况，却被它漏掉了。
        self._abandoned: set[str] = set()
        # Stream ids handed to the worker queue and not yet published or
        # forgotten -- the only ids for which "no sequence list" can still
        # mean "not submitted yet" rather than "already finished". Bounded by
        # requests actually in flight.
        # 【译】已交给工作队列、但还没有发布也没有被遗忘的流id——只有对这些id，"没有序列列表"才仍可能意味着"还没提交"
        #   而不是"已经结束"。其数量以实际在途的请求数为上限。
        # 解释："发布"指把序列列表写进 _stream_seqs（见 _submit_stream_request）
        self._awaiting_submit: set[str] = set()
        self._stream_seqs_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AtomStandaloneEngineService",
            daemon=True,
        )
        self._worker.start()

    # 非流式入口：入队，然后在调用线程里阻塞在 future.result() 上，直到所有序列生成结束
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        effective_n: int,
        kv_transfer_params: dict[str, Any] | None = None,
        data_parallel_rank: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._closed.is_set():
            raise RuntimeError("ATOM standalone engine service is closed")

        future: concurrent.futures.Future[list[dict[str, Any]]] = (
            concurrent.futures.Future()
        )
        with self._active_futures_lock:
            self._active_futures.add(future)

        self._queue.put(
            EngineRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                effective_n=effective_n,
                future=future,
                kv_transfer_params=kv_transfer_params,
                data_parallel_rank=data_parallel_rank,
            )
        )
        try:
            return future.result()
        finally:
            with self._active_futures_lock:
                self._active_futures.discard(future)

    # 关闭：放入None让工作线程退出，并把所有未完成的future设为异常
    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
        with self._active_futures_lock:
            futures = list(self._active_futures)
        for future in futures:
            if not future.done():
                future.set_exception(
                    RuntimeError("ATOM standalone engine service is closed")
                )
        if self._worker.is_alive():
            self._worker.join(timeout=1)

    # 工作线程主循环：取请求 -> 服务已关闭就回报错误 -> 否则提交（流式 / 非流式）；任何异常都转成该请求的错误，不让线程退出
    def _worker_loop(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                break
            if self._closed.is_set():
                if isinstance(request, EngineStreamRequest):
                    request.stream_queue.put(
                        {
                            "error": "ATOM standalone engine service is closed",
                        }
                    )
                    request.stream_queue.put({"done": True})
                    self._close_submit_window(request.request_id)
                else:
                    self._set_future_exception(
                        request.future,
                        RuntimeError("ATOM standalone engine service is closed"),
                    )
                continue

            try:
                if isinstance(request, EngineStreamRequest):
                    try:
                        self._submit_stream_request(request)
                    finally:
                        self._close_submit_window(request.request_id)
                else:
                    self._submit_request(request)
            except Exception as error:
                # Broad on purpose: this is the worker thread's outer loop, and
                # anything that escapes here kills it and hangs every request
                # after this one. The client learns what happened either way;
                # the server did not, so a failed request left no trace on the
                # side that could act on it.
                # 【译】故意捕获得很宽：这是工作线程的最外层循环，任何从这里逃逸的异常都会杀死它，并让之后的每个请求挂住。
                #   客户端无论如何都会知道发生了什么；服务端以前却不知道，所以一个失败的请求在能对它采取行动的那一侧没有留下任何痕迹。
                logger.exception(
                    "standalone engine request %s failed", request.request_id
                )
                if isinstance(request, EngineStreamRequest):
                    request.stream_queue.put(
                        {
                            "error": str(error),
                        }
                    )
                    request.stream_queue.put({"done": True})
                else:
                    self._set_future_exception(request.future, error)

    # 非流式提交：n>1走fanout（一个prompt展开成n条兄弟序列），然后交给CoreManager
    def _submit_request(self, request: EngineRequest) -> None:
        if request.effective_n > 1:
            seqs = self._preprocess_fanout_request(request)
        else:
            seqs = [self._preprocess_single_request(request)]
        self.engine.core_mgr.add_request(seqs)

    # 流式入口：登记为待提交、入队，立即返回stream_queue，不等待生成
    def start_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        effective_n: int,
        kv_transfer_params: dict[str, Any] | None = None,
        data_parallel_rank: int | None = None,
    ) -> queue.Queue[dict[str, Any]]:
        if self._closed.is_set():
            raise RuntimeError("ATOM standalone engine service is closed")

        stream_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._expect_stream(request_id)
        self._queue.put(
            EngineStreamRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                effective_n=effective_n,
                stream_queue=stream_queue,
                kv_transfer_params=kv_transfer_params,
                data_parallel_rank=data_parallel_rank,
            )
        )
        return stream_queue

    def _expect_stream(self, request_id: str) -> None:
        """This id is on the worker queue and its sequences do not exist yet.

        The one registrar, so that "could still be submitted" has a single
        definition; the window closes in `_submit_stream_request`.
        """
        # 【译】这个id已在工作队列里，它的序列还不存在。
        #
        #   这是唯一的登记处，让"仍可能被提交"只有一个定义；这个窗口在 _submit_stream_request 中关闭。
        # 解释：实际关闭窗口的是 _worker_loop 在提交之后调用的 _close_submit_window（见下一个方法的docstring），并不在 _submit_stream_request 内部
        with self._stream_seqs_lock:
            self._awaiting_submit.add(request_id)

    def _close_submit_window(self, request_id: str) -> None:
        """This id can no longer be submitted, however that came about.

        The counterpart to `_expect_stream`, and called from the worker loop
        rather than from inside `_submit_stream_request` because that function
        has four ways out -- abandoned at the top, abandoned mid-preprocess,
        published, or raising -- and bookkeeping spread over them discarded on
        two. A stream closed while it was preprocessing, and one whose
        preprocess raised, each left an id behind for the life of the process:
        the same leak `_awaiting_submit` was added to remove, one level down.

        It runs *after* the submit, never before, because the window has to
        stay open for an abort arriving while the sequences are still being
        built -- that abort has nothing to stop yet either.
        """
        # 【译】这个id已经不可能再被提交了，不管是怎么造成的。
        #
        #   它是 _expect_stream 的对应操作，由工作循环调用，而不是在 _submit_stream_request 内部调用，因为那个函数有四个出口——
        #   开头就发现已放弃、预处理中途被放弃、已发布、或抛出异常——把簿记分散到这些出口上时，只有其中两个做了清除。
        #   一个在预处理期间被关闭的流、一个预处理抛了异常的流，都会各自留下一个id，存活到进程结束：这正是加入
        #   _awaiting_submit 要消除的那种泄漏，只是低了一层。
        #
        #   它在提交之后运行，绝不在之前，因为对于在序列还在构建时到来的中止，窗口必须保持打开——那个中止此时也同样没有东西可停。
        with self._stream_seqs_lock:
            self._awaiting_submit.discard(request_id)

    # 流式提交：预处理前、预处理后各检查一次"是否已被放弃"；add_request之后再检查第三次，决定是登记序列还是立刻中止它们
    def _submit_stream_request(self, request: EngineStreamRequest) -> None:
        if self._take_abandoned(request.request_id):
            return
        if request.effective_n > 1:
            seqs = self._preprocess_fanout_stream_request(request)
        else:
            seqs = [self._preprocess_single_stream_request(request)]
        with self._stream_seqs_lock:
            # Checked again: the close may have arrived while this was
            # preprocessing. Sequences built and never added to the engine
            # cost nothing to drop -- there is no abort to do.
            # 【译】再检查一次：关闭可能在预处理期间到来。已经构建但从未加入引擎的序列丢掉不需要任何代价——没有需要做的中止。
            if request.request_id in self._abandoned:
                self._abandoned.discard(request.request_id)
                return
        self.engine.core_mgr.add_request(seqs)
        # Published *after* the engine has them, and the close re-checked
        # once more. Publishing first left a window where `abort_stream`
        # popped the list and aborted sequences the engine had not been told
        # about -- the abort raised into a swallowed `except`, the worker then
        # added them, and they decoded to `max_tokens` into a queue nobody
        # would read. Which side of `add_request` the publish sits on decides
        # whether a close in that window aborts or leaks; this way one of the
        # two branches below always fires.
        # 【译】在引擎拿到序列之后才发布，并且再重新检查一次关闭。先发布会留下一个窗口：abort_stream 弹出列表，中止了引擎还不知道的序列——
        #   中止在一个被吞掉的 except 里抛错，随后工作线程把它们加进引擎，它们一直decode到max_tokens，写进一个没人读的队列。
        #   发布放在 add_request 的哪一边，决定了在那个窗口内到来的关闭是被中止还是被泄漏；按现在的写法，下面两个分支总有一个会执行。
        with self._stream_seqs_lock:
            if request.request_id in self._abandoned:
                self._abandoned.discard(request.request_id)
                abandoned = seqs
            else:
                self._stream_seqs[request.request_id] = list(seqs)
                abandoned = ()
        for seq in abandoned:
            self._abort_seq(seq)

    # 如果这个流已被标记为放弃，就清除标记并返回True（调用方随即放弃提交）
    def _take_abandoned(self, request_id: str) -> bool:
        with self._stream_seqs_lock:
            if request_id not in self._abandoned:
                # Still to be submitted, so the window `_awaiting_submit`
                # marks stays open: an abort arriving while this request is
                # being preprocessed has no sequences to stop yet either, and
                # closing the window here let it through to the engine.
                # 【译】仍待提交，所以 _awaiting_submit 标记的窗口保持打开：在这个请求预处理期间到来的中止，此时同样没有序列可停，
                #   而在这里关闭窗口曾让它漏到了引擎那里。
                return False
            self._abandoned.discard(request_id)
            return True

    def forget_stream(self, request_id: str) -> None:
        """Drop what is remembered about a stream that ended on its own.

        Without this the sequence list for every completed stream stays for
        the life of the process; `abort_stream` only removes the ones a caller
        closes, and the engine has already forgotten those sequences by then
        anyway.
        """
        # 【译】丢掉关于一个自然结束的流所记住的一切。
        #
        #   没有这一步，每个已完成流的序列列表都会存活到进程结束；abort_stream 只会删除调用方关闭的那些，
        #   而且到那时引擎早已忘掉这些序列了。
        with self._stream_seqs_lock:
            self._stream_seqs.pop(request_id, None)
            self._abandoned.discard(request_id)
            self._awaiting_submit.discard(request_id)

    def abort_stream(self, request_id: str) -> None:
        """Stop the sequences behind a stream that is still live.

        Call this only for a stream the caller has just taken ownership of --
        `close_*_stream` does it under the same lock that removes the stream
        from the registry. It cannot be used as a blanket "make sure this is
        gone", because a missing sequence list is ambiguous: it means either
        "not submitted yet" or "already finished and forgotten", and this has
        to assume the first or the pre-submit abort window reopens. Calling it
        on a stream that already ended therefore remembers that id forever.
        """
        # 【译】停止一个仍然活着的流背后的序列。
        #
        #   只对调用方刚刚取得所有权的流调用它——close_*_stream 在把流从注册表移除的同一把锁下这样做。它不能当作笼统的
        #   "确保它消失"来用，因为缺少序列列表是有歧义的：它要么意味着"还没提交"，要么意味着"已经结束并被遗忘"，
        #   而这里必须假设是前者，否则提交前的中止窗口就会重新打开。因此对一个已经结束的流调用它，会永远记住那个id。
        # 解释：与代码有两处出入——close_*_stream 是在 _streams_lock 内弹出注册表项、释放锁之后才调用本方法；
        #   另外下面已改成只有id还在 _awaiting_submit 里时才加入 _abandoned，所以对已结束并被 forget_stream 清理过的流调用，不会再永久记住它
        with self._stream_seqs_lock:
            seqs = self._stream_seqs.pop(request_id, None)
            if seqs is None:
                # The ambiguity the docstring names: `_awaiting_submit` is the
                # only thing in here that can say "not submitted yet" rather
                # than "already finished", and remembering a finished id leaks.
                # 【译】这就是docstring所说的歧义：这里面只有 _awaiting_submit 能说出"还没提交"而不是"已经结束"，而记住一个已结束的id就是泄漏。
                if request_id in self._awaiting_submit:
                    self._abandoned.add(request_id)
                return
        for seq in seqs:
            self._abort_seq(seq)

    # 中止一条序列：通知CoreManager abort_request，并从 io_processor.requests 里删掉它的记录；失败只记debug日志
    def _abort_seq(self, seq: Any) -> None:
        seq_id = getattr(seq, "id", None)
        if seq_id is None:
            return
        try:
            self.engine.core_mgr.abort_request(seq_id)
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            logger.debug("abort_request(%s): %s", seq_id, exc)
        requests = getattr(self.engine.io_processor, "requests", None)
        if requests is not None:
            requests.pop(seq_id, None)

    # 流式n=1：io_processor.preprocess 建Sequence（第2批 llm_engine.py），挂上把输出转给StreamRequestState的回调
    def _preprocess_single_stream_request(self, request: EngineStreamRequest) -> Any:
        state = StreamRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            stream_queue=request.stream_queue,
            n=1,
        )

        def completion_callback(request_output: Any) -> None:
            state.record(0, request_output)

        return self.engine.io_processor.preprocess(
            request.prompt,
            request.sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=request.kv_transfer_params,
            data_parallel_rank=request.data_parallel_rank,
        )

    # 流式n>1：preprocess_fanout 把一个prompt展开成n条兄弟序列，每条的回调带自己的index
    def _preprocess_fanout_stream_request(
        self, request: EngineStreamRequest
    ) -> list[Any]:
        state = StreamRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            stream_queue=request.stream_queue,
            n=request.effective_n,
        )

        def make_callback(index: int):
            def completion_callback(request_output: Any) -> None:
                state.record(index, request_output)

            return completion_callback

        return self.engine.io_processor.preprocess_fanout(
            request.prompt,
            request.sampling_params,
            stream_callbacks=[
                make_callback(index) for index in range(request.effective_n)
            ],
            kv_transfer_params=request.kv_transfer_params,
            parent_request_id=request.request_id,
            data_parallel_rank=request.data_parallel_rank,
        )

    # 非流式n=1：回调把输出累加进SingleRequestState，并记下prompt的token数
    def _preprocess_single_request(self, request: EngineRequest) -> Any:
        state = SingleRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            future=request.future,
        )

        def completion_callback(request_output: Any) -> None:
            state.record(request_output)

        seq = self.engine.io_processor.preprocess(
            request.prompt,
            request.sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=request.kv_transfer_params,
            data_parallel_rank=request.data_parallel_rank,
        )
        state.set_num_tokens_input(seq.num_prompt_tokens)
        return seq

    # 非流式n>1：用FanoutRequestState等待n条兄弟序列全部结束
    def _preprocess_fanout_request(self, request: EngineRequest) -> list[Any]:
        state = FanoutRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            future=request.future,
            n=request.effective_n,
        )

        def make_callback(index: int):
            def completion_callback(request_output: Any) -> None:
                state.record(index, request_output)

            return completion_callback

        seqs = self.engine.io_processor.preprocess_fanout(
            request.prompt,
            request.sampling_params,
            stream_callbacks=[
                make_callback(index) for index in range(request.effective_n)
            ],
            kv_transfer_params=request.kv_transfer_params,
            parent_request_id=request.request_id,
            data_parallel_rank=request.data_parallel_rank,
        )
        if seqs:
            state.set_num_tokens_input(seqs[0].num_prompt_tokens)
        return seqs

    # future还没完成时才设置异常（对已完成的future再设置会抛InvalidStateError）
    @staticmethod
    def _set_future_exception(
        future: concurrent.futures.Future[list[dict[str, Any]]],
        error: BaseException,
    ) -> None:
        if not future.done():
            future.set_exception(error)


# 非流式n=1的结果累加器：每次回调把新token追加进token_ids，记录首个 / 最后一个token的时间；结束时一次性decode并完成future
class SingleRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        future: concurrent.futures.Future[list[dict[str, Any]]],
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.future = future
        self.started_at = time.time()
        self.first_token_at: float | None = None
        self.last_token_at: float | None = None
        # An array for the same reason the engine's `token_ids` is one. It
        # stays one: this dict goes to `build_*_response`, which reads
        # `text` and the counters and never `token_ids`.
        # 【译】用array的原因与引擎的 token_ids 用array相同。它一直保持为array：这个dict会交给 build_*_response，
        #   而后者只读 text 和各计数器，从不读 token_ids。
        # 解释：原因见第1批 sequence.py 的 new_token_ids
        self.token_ids = new_token_ids()
        self.finish_reason: str | None = None
        self.num_tokens_input = 0
        self.kv_transfer_output_meta_info: Any = None
        self._lock = threading.Lock()

    def set_num_tokens_input(self, num_tokens_input: int) -> None:
        self.num_tokens_input = num_tokens_input

    # 回调（在CoreManager的输出线程中被调用）：累加token；finished时构造输出dict并完成future
    def record(self, request_output: Any) -> None:
        with self._lock:
            if self.future.done():
                return
            self.kv_transfer_output_meta_info = getattr(
                request_output, "kv_transfer_params_output", None
            )
            now = time.time()
            output_tokens = request_output.output_tokens or []
            if output_tokens:
                if self.first_token_at is None:
                    self.first_token_at = now
                self.last_token_at = now
                self.token_ids.extend(output_tokens)
            if request_output.finished:
                self.finish_reason = request_output.finish_reason
                self.future.set_result([self._build_output(time.time())])

    # 输出dict：text、finish_reason、输入 / 输出token数、ttft（首token延迟）、tpot（相邻token平均间隔）、latency，
    #   交给 serving_chat / serving_completion 的 build_*_response 使用
    def _build_output(self, finished_at: float) -> dict[str, Any]:
        num_tokens_output = len(self.token_ids)
        ttft = (
            self.first_token_at - self.started_at
            if self.first_token_at is not None
            else 0.0
        )
        tpot = (
            (self.last_token_at - self.first_token_at) / (num_tokens_output - 1)
            if self.first_token_at is not None
            and self.last_token_at is not None
            and num_tokens_output > 1
            else 0.0
        )
        output = {
            "text": self.tokenizer.decode(self.token_ids, skip_special_tokens=True),
            "token_ids": self.token_ids,
            "finish_reason": self.finish_reason,
            "num_tokens_input": self.num_tokens_input,
            "num_tokens_output": num_tokens_output,
            "ttft": ttft,
            "tpot": tpot,
            "latency": finished_at - self.started_at,
        }
        if self.kv_transfer_output_meta_info is not None:
            output["kv_transfer_output_meta_info"] = self.kv_transfer_output_meta_info
        return output


# 非流式n>1：每条兄弟序列各自累加，全部finished后一起完成future
class FanoutRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        future: concurrent.futures.Future[list[dict[str, Any]]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.future = future
        self.n = n
        self.started_at = time.time()
        self.per_tokens = [new_token_ids() for _ in range(n)]
        self.per_first_token_at: list[float | None] = [None] * n
        self.per_last_token_at: list[float | None] = [None] * n
        self.per_finish_reason: list[str | None] = [None] * n
        self.finished = [False] * n
        self.num_tokens_input = 0
        self._lock = threading.Lock()

    def set_num_tokens_input(self, num_tokens_input: int) -> None:
        self.num_tokens_input = num_tokens_input

    def record(self, index: int, request_output: Any) -> None:
        with self._lock:
            if self.future.done() or self.finished[index]:
                return
            now = time.time()
            output_tokens = request_output.output_tokens or []
            if output_tokens:
                if self.per_first_token_at[index] is None:
                    self.per_first_token_at[index] = now
                self.per_last_token_at[index] = now
                self.per_tokens[index].extend(output_tokens)
            if request_output.finished:
                self.per_finish_reason[index] = request_output.finish_reason
                self.finished[index] = True
                if all(self.finished):
                    self.future.set_result(self._build_outputs(time.time()))

    def _build_outputs(self, finished_at: float) -> list[dict[str, Any]]:
        outputs = []
        for index in range(self.n):
            num_tokens_output = len(self.per_tokens[index])
            first_token_at = self.per_first_token_at[index]
            last_token_at = self.per_last_token_at[index]
            ttft = (
                first_token_at - self.started_at if first_token_at is not None else 0.0
            )
            tpot = (
                (last_token_at - first_token_at) / (num_tokens_output - 1)
                if first_token_at is not None
                and last_token_at is not None
                and num_tokens_output > 1
                else 0.0
            )
            outputs.append(
                {
                    "text": self.tokenizer.decode(
                        self.per_tokens[index], skip_special_tokens=True
                    ),
                    "token_ids": self.per_tokens[index],
                    "finish_reason": self.per_finish_reason[index],
                    "num_tokens_input": self.num_tokens_input,
                    "num_tokens_output": num_tokens_output,
                    "ttft": ttft,
                    "tpot": tpot,
                    "latency": finished_at - self.started_at,
                }
            )
        return outputs


# 流式回调：每批新token立即decode成文本，作为事件放进stream_queue；全部兄弟序列结束后再放入 {"done": True}
class StreamRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.stream_queue = stream_queue
        self.finished = [False] * n
        self._lock = threading.Lock()

    def record(self, index: int, request_output: Any) -> None:
        with self._lock:
            if self.finished[index]:
                return

            output_tokens = request_output.output_tokens or []
            text = (
                self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if output_tokens
                else ""
            )
            if output_tokens or request_output.finished:
                event = {
                    "index": index,
                    "text": text,
                    "token_ids": output_tokens,
                    "finished": request_output.finished,
                    "finish_reason": request_output.finish_reason,
                }
                if getattr(request_output, "kv_transfer_params_output", None):
                    event["kv_transfer_params"] = (
                        request_output.kv_transfer_params_output
                    )
                self.stream_queue.put(event)

            if request_output.finished:
                self.finished[index] = True
                if all(self.finished):
                    self.stream_queue.put({"done": True})


# 流式chat的状态：每次drain从stream_queue取事件，对每个兄弟序列分别做推理内容分离 + 工具调用解析，
#   产出OpenAI的 chat.completion.chunk SSE帧；最后补发finish_reason、usage和 [DONE]
class ChatCompletionStreamState:
    def __init__(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
        tool_parser_cls: type | None = None,
        reasoning: ReasoningChannel = NO_REASONING,
        tools: list | None = None,
        tool_choice: Any = None,
    ) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.stream_queue = stream_queue
        self.num_tokens_input = len(tokenizer.encode(prompt))
        self.num_tokens_output = [0] * n
        # Same prompt for every sibling, so the same starting state. Without
        # this a template that opens the reasoning channel in the prompt has
        # its whole trace delivered as the answer: state 0 no longer infers
        # reasoning from a bare end marker, because inferring it means waiting
        # for one.
        # 【译】每个兄弟序列的prompt相同，所以起始状态相同。没有这一步，一个在prompt里就打开推理通道的模板，会把整段推理过程
        #   当作回答返回：状态0不再根据一个孤立的结束标记推断出推理，因为要推断就得等那个标记出现。
        self.reasoning_filters = [reasoning.stream() for _ in range(n)]
        # `tool_choice="none"` forbids tool calls on this path too, and
        # `tools` is what type-coerces the arguments. The OpenAI server gated
        # and passed both; this one had neither, so the same request answered
        # differently depending on the entrypoint.
        # 【译】tool_choice="none" 在这条路径上同样禁止工具调用，而 tools 是用来给参数做类型转换的。OpenAI服务会做这道把关并
        #   传入两者；这里以前两者都没有，于是同一个请求因入口不同而得到不同的回答。
        self.tool_choice = tool_choice
        self.tool_parsers = [
            ToolCallStreamParser(
                tools=tools,
                parser_cls=tool_parser_cls,
                suppress_calls=forbids_tool_calls(tool_choice),
            )
            for _ in range(n)
        ]
        self.has_tool_calls = [False] * n
        self.finished = [False] * n
        self.role_sent = [False] * n
        self.completed = False
        self.closed = False
        self._pending_final_chunks: list[str] | None = None
        # Chunks built but not yet handed out. The drain takes `max_items` at
        # a time and this used to `break` out of the build loop when it hit
        # that -- discarding every event past the first, permanently, since
        # the parser had already yielded them. At `max_items=1` a tool call
        # lost its arguments and reported `finish_reason: stop`.
        # 【译】已经构建、但还没交出去的chunk。drain每次最多取 max_items 个，这里以前在达到上限时 break 出构建循环——
        #   把第一个之后的所有事件永久丢弃了，因为解析器已经把它们产出过了。在 max_items=1 时，一个工具调用会丢掉参数，
        #   并报告 finish_reason: stop。
        self._pending_chunks: list[str] = []
        # The engine's own reason per sibling; see serving_chat.
        # 【译】每个兄弟序列由引擎给出的结束原因；见 serving_chat。
        self.engine_finish_reasons: list[str | None] = [None] * n
        self._lock = threading.Lock()

    # 取一批SSE帧（最多max_items个）：先交出每个兄弟序列的role chunk和积压的chunk；没有可交的就在stream_queue上
    #   最多等timeout秒，取到事件后转成chunk（后续事件不再等待）；Rust侧以 (16, 0.05) 反复调用，返回空列表时继续轮询
    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            self._append_initial_role_chunks(chunks, max_items)
            if not self.completed:
                chunks.extend(self._hand_out(max_items - len(chunks)))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed:
                        chunks.extend(self._hand_out(max_items - len(chunks)))
                break

            with self._lock:
                chunks.extend(self._event_to_chunks(event, max_items - len(chunks)))
                if self.completed or self.closed:
                    break
        return chunks

    # 标记为已关闭；正在drain的线程看到closed后停止取新事件
    def close(self) -> None:
        with self._lock:
            self.closed = True

    # 每个兄弟序列的第一个chunk：delta里只有 role=assistant
    def _append_initial_role_chunks(self, chunks: list[str], max_items: int) -> None:
        for index, sent in enumerate(self.role_sent):
            if len(chunks) >= max_items:
                break
            if not sent:
                chunks.append(
                    create_chat_chunk(
                        self.request_id,
                        self.model_name,
                        delta={"role": "assistant", "content": ""},
                        index=index,
                    )
                )
                self.role_sent[index] = True

    # 一个队列事件 -> chunk：error事件安排 error帧 + [DONE] 作为结尾；正常事件先过推理内容分离，content部分再过工具调用解析器，
    #   finished时把两者都flush；产出的chunk先进 _pending_chunks，再经 _hand_out 按剩余容量交出
    def _event_to_chunks(
        self, event: dict[str, Any], remaining_capacity: int
    ) -> list[str]:
        if remaining_capacity <= 0:
            return []
        if event.get("error"):
            if self._pending_final_chunks is None:
                self._pending_final_chunks = [
                    self._error_chunk(str(event["error"])),
                    STREAM_DONE_MESSAGE,
                ]
            return self._hand_out(remaining_capacity)
        if event.get("done"):
            return [] if self.completed else self._hand_out(remaining_capacity)

        index = int(event["index"])
        if self.finished[index]:
            # Still hand out whatever is queued, and the final chunks once it
            # is empty -- returning early here left a fully drained stream
            # with no `finish_reason`, no usage and no `[DONE]`.
            # 【译】仍然交出已排队的内容，并在队列清空后交出最后的chunk——以前在这里提前返回，会让一个已完全消费的流
            #   没有 finish_reason、没有usage，也没有 [DONE]。
            return self._hand_out(remaining_capacity)

        chunks: list[str] = []
        text = event.get("text") or ""
        self.num_tokens_output[index] += len(event.get("token_ids", []))
        if event.get("finish_reason"):
            self.engine_finish_reasons[index] = event["finish_reason"]

        segments = self.reasoning_filters[index].process(text)
        if event.get("finished", False):
            segments.extend(self.reasoning_filters[index].flush())

        for field, segment_text in segments:
            if field == "reasoning_content":
                if segment_text:
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"reasoning_content": segment_text},
                            index=index,
                        )
                    )
            elif field == "content":
                for event_type, data in self.tool_parsers[index].process(segment_text):
                    if event_type == "content":
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"content": data},
                                index=index,
                            )
                        )
                    elif event_type == "tool_call_start":
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"tool_calls": [data]},
                                index=index,
                            )
                        )
                    elif event_type == "tool_call_args":
                        self.has_tool_calls[index] = True
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"tool_calls": [data]},
                                index=index,
                            )
                        )

        if event.get("finished", False):
            for event_type, data in self.tool_parsers[index].flush():
                if event_type == "content":
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"content": data},
                            index=index,
                        )
                    )
                elif event_type == "tool_call_start":
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
                elif event_type == "tool_call_args":
                    self.has_tool_calls[index] = True
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
            self.finished[index] = True

        self._pending_chunks.extend(chunks)
        return self._hand_out(remaining_capacity)

    def _build_final_chunks(self) -> list[str]:
        """The finish/usage/[DONE] tail, built once. Only `_hand_out` calls
        this, and only once nothing built is still queued."""
        # 【译】结尾部分 finish/usage/[DONE]，只构建一次。只有 _hand_out 调用它，而且只在已构建的chunk都发完之后。
        # 解释：每个兄弟序列的finish_reason由 openai_stop_reason_with_calls 决定：产出过工具调用参数就报tool_calls，否则用引擎给的原因
        chunks: list[str] = []
        for index, has_tool_calls in enumerate(self.has_tool_calls):
            finish_reason = openai_stop_reason_with_calls(
                self.engine_finish_reasons[index], has_tool_calls
            )
            chunks.append(
                create_chat_chunk(
                    self.request_id,
                    self.model_name,
                    finish_reason=finish_reason,
                    index=index,
                )
            )
        completion_tokens = sum(self.num_tokens_output)
        usage = {
            "prompt_tokens": self.num_tokens_input,
            "completion_tokens": completion_tokens,
            "total_tokens": self.num_tokens_input + completion_tokens,
        }
        if len(self.num_tokens_output) > 1:
            usage["num_choices"] = len(self.num_tokens_output)
        usage_chunk = {
            "id": self.request_id,
            "object": CHAT_COMPLETION_CHUNK_OBJECT,
            "created": int(time.time()),
            "model": self.model_name,
            "usage": usage,
        }
        chunks.append(data_frame(usage_chunk))
        chunks.append(STREAM_DONE_MESSAGE)
        return chunks

    def _take_pending(self, remaining_capacity: int) -> list[str]:
        """As many built chunks as the caller has room for; the rest wait."""
        # 【译】调用方有多少空间就给多少已构建的chunk；剩下的继续等待。
        if remaining_capacity <= 0:
            return []
        out = self._pending_chunks[:remaining_capacity]
        del self._pending_chunks[:remaining_capacity]
        return out

    def _hand_out(self, remaining_capacity: int) -> list[str]:
        """The one exit. Queued chunks first; terminal ones only after them.

        Every producer returns through here, and that is the whole of the
        rule: "every sibling finished" is not the same fact as "everything
        built has been sent", and three places used the first as a proxy for
        the second. The terminal chunks were then handed out while built
        chunks were still queued -- and once the finals go, `completed` is set
        and the caller pops the state, so the queued ones are gone. Measured
        at the production `max_items`: a response with nine tool calls
        delivered eight and reported `finish_reason: tool_calls`; at
        `max_items=1` a single call reported `tool_calls` having sent no
        `tool_calls` delta at all.
        """
        # 【译】唯一的出口。先发排队中的chunk；结尾的chunk只能在它们之后。
        #
        #   每个生产者都经由这里返回，这就是规则的全部："每个兄弟序列都已结束"和"所有已构建的东西都已发送"不是同一个事实，
        #   而有三个地方曾拿前者代替后者。于是结尾chunk在已构建的chunk还排着队时就被交了出去——而结尾chunk一旦发出，
        #   completed 就被置位，调用方弹出状态，排队中的chunk就没了。在生产环境的 max_items 下实测：一个带九个工具调用的响应
        #   只送达了八个，并报告 finish_reason: tool_calls；在 max_items=1 时，单个调用报告了 tool_calls，却一个 tool_calls 增量都没发。
        out = self._take_pending(remaining_capacity)
        # `_take_pending` never returns more than the caller asked for, so no
        # room left means chunks are still queued -- there is nothing to test
        # for beyond that, and a second condition here would be a guard that
        # cannot fail.
        # 【译】_take_pending 返回的数量从不超过调用方要求的，所以没有剩余空间就意味着还有chunk在排队——除此之外没有别的需要检查的，
        #   在这里再加第二个条件将会是一个永远不会失败的守卫。
        if len(out) >= remaining_capacity:
            return out
        if self._pending_final_chunks is None:
            if not all(self.finished):
                return out
            self._pending_final_chunks = self._build_final_chunks()
        return out + self._drain_pending_final_chunks(remaining_capacity - len(out))

    # 按剩余容量交出结尾chunk；全部交完时置 completed=True（调用方随后从注册表移除这个流）
    def _drain_pending_final_chunks(self, remaining_capacity: int) -> list[str]:
        if not self._pending_final_chunks or remaining_capacity <= 0:
            return []
        chunks = self._pending_final_chunks[:remaining_capacity]
        del self._pending_final_chunks[:remaining_capacity]
        if not self._pending_final_chunks:
            self.completed = True
        return chunks

    # 错误帧：data: {"error": {"message": ...}}
    @staticmethod
    def _error_chunk(message: str) -> str:
        return f"data: {json.dumps({'error': {'message': message}})}\n\n"


# 流式 /v1/completions 的状态：与 ChatCompletionStreamState 相同的 drain / _hand_out 结构，
#   但没有推理内容分离和工具调用解析，每个事件直接变成一个 text_completion chunk
class CompletionStreamState:
    def __init__(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.stream_queue = stream_queue
        self.num_tokens_input = len(tokenizer.encode(prompt))
        self.num_tokens_output = [0] * n
        self.finished = [False] * n
        self.completed = False
        self.closed = False
        self._pending_final_chunks: list[str] | None = None
        # Chunks built but not yet handed out. The drain takes `max_items` at
        # a time and this used to `break` out of the build loop when it hit
        # that -- discarding every event past the first, permanently, since
        # the parser had already yielded them. At `max_items=1` a tool call
        # lost its arguments and reported `finish_reason: stop`.
        # 【译】已经构建、但还没交出去的chunk。drain每次最多取 max_items 个，这里以前在达到上限时 break 出构建循环——
        #   把第一个之后的所有事件永久丢弃了，因为解析器已经把它们产出过了。在 max_items=1 时，一个工具调用会丢掉参数，
        #   并报告 finish_reason: stop。
        # 解释：这段注释是从 ChatCompletionStreamState 复制过来的；completion路径没有工具调用解析器，
        #   "解析器""工具调用"在这里并不适用，但"超出max_items的事件不能丢"的道理同样成立
        self._pending_chunks: list[str] = []
        # The engine's own reason per sibling; see serving_chat.
        # 【译】每个兄弟序列由引擎给出的结束原因；见 serving_chat。
        self.engine_finish_reasons: list[str | None] = [None] * n
        self._lock = threading.Lock()

    # 同 ChatCompletionStreamState.drain，只是没有开头的role chunk
    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            if not self.completed:
                chunks.extend(self._hand_out(max_items - len(chunks)))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed:
                        chunks.extend(self._hand_out(max_items - len(chunks)))
                break

            with self._lock:
                chunks.extend(self._event_to_chunks(event, max_items - len(chunks)))
                if self.completed or self.closed:
                    break
        return chunks

    # 标记为已关闭
    def close(self) -> None:
        with self._lock:
            self.closed = True

    # 一个队列事件 -> 一个completion chunk（finish_reason留到结尾chunk再报）；PD分离时带上kv_transfer_params
    def _event_to_chunks(
        self, event: dict[str, Any], remaining_capacity: int
    ) -> list[str]:
        if remaining_capacity <= 0:
            return []
        if event.get("error"):
            if self._pending_final_chunks is None:
                self._pending_final_chunks = [
                    self._error_chunk(str(event["error"])),
                    STREAM_DONE_MESSAGE,
                ]
            return self._hand_out(remaining_capacity)
        if event.get("done"):
            return [] if self.completed else self._hand_out(remaining_capacity)

        index = int(event["index"])
        if self.finished[index]:
            # Still hand out whatever is queued, and the final chunks once it
            # is empty -- returning early here left a fully drained stream
            # with no `finish_reason`, no usage and no `[DONE]`.
            # 【译】仍然交出已排队的内容，并在队列清空后交出最后的chunk——以前在这里提前返回，会让一个已完全消费的流
            #   没有 finish_reason、没有usage，也没有 [DONE]。
            return self._hand_out(remaining_capacity)

        extra_fields: dict[str, Any] = {}
        if "kv_transfer_params" in event:
            extra_fields["kv_transfer_params"] = event["kv_transfer_params"]

        self.num_tokens_output[index] += len(event.get("token_ids", []))
        if event.get("finish_reason"):
            self.engine_finish_reasons[index] = event["finish_reason"]
        chunks = [
            create_completion_chunk(
                self.request_id,
                self.model_name,
                event.get("text") or "",
                # The reason goes on the final chunk, as on every other
                # streaming path here and in the OpenAI server. Put here too
                # it was never translated, and `engine_finish_reasons` --
                # which exists to carry it to the terminal chunk -- was never
                # written at all.
                # 【译】结束原因放在最后的chunk上，与这里以及OpenAI服务中的其他所有流式路径一致。以前也放在这里时，它从未被翻译
                #   （转换成OpenAI的取值），而 engine_finish_reasons——它存在的意义就是把原因带到结尾chunk——则从未被写入。
                finish_reason=None,
                index=index,
                **extra_fields,
            )
        ]

        if event.get("finished", False):
            self.finished[index] = True

        self._pending_chunks.extend(chunks)
        return self._hand_out(remaining_capacity)

    def _build_final_chunks(self) -> list[str]:
        """The finish/usage/[DONE] tail, built once. Only `_hand_out` calls
        this, and only once nothing built is still queued."""
        # 【译】结尾部分 finish/usage/[DONE]，只构建一次。只有 _hand_out 调用它，而且只在已构建的chunk都发完之后。
        chunks: list[str] = []
        for index in range(len(self.num_tokens_output)):
            # The engine's own reason, as the chat path reports it. This
            # was recorded per sibling and then never read: every
            # completion said `stop`, including one the engine cut off at
            # `max_tokens`, which `stream=false` reported as `length`.
            # 【译】引擎自己给出的原因，与chat路径报告的方式一样。它以前按兄弟序列记录下来却从不读取：每个completion都说 stop，
            #   包括被引擎截断在 max_tokens 的那个——而 stream=false 时同样的情况会报告 length。
            chunks.append(
                create_completion_chunk(
                    self.request_id,
                    self.model_name,
                    "",
                    finish_reason=(
                        openai_stop_reason(self.engine_finish_reasons[index]) or "stop"
                    ),
                    index=index,
                )
            )
        completion_tokens = sum(self.num_tokens_output)
        usage = {
            "prompt_tokens": self.num_tokens_input,
            "completion_tokens": completion_tokens,
            "total_tokens": self.num_tokens_input + completion_tokens,
        }
        if len(self.num_tokens_output) > 1:
            usage["num_choices"] = len(self.num_tokens_output)
        usage_chunk = {
            "id": self.request_id,
            "object": TEXT_COMPLETION_OBJECT,
            "created": int(time.time()),
            "model": self.model_name,
            "usage": usage,
        }
        chunks.append(data_frame(usage_chunk))
        chunks.append(STREAM_DONE_MESSAGE)
        return chunks

    def _take_pending(self, remaining_capacity: int) -> list[str]:
        """As many built chunks as the caller has room for; the rest wait."""
        # 【译】调用方有多少空间就给多少已构建的chunk；剩下的继续等待。
        if remaining_capacity <= 0:
            return []
        out = self._pending_chunks[:remaining_capacity]
        del self._pending_chunks[:remaining_capacity]
        return out

    def _hand_out(self, remaining_capacity: int) -> list[str]:
        """The one exit. Queued chunks first; terminal ones only after them.

        Every producer returns through here, and that is the whole of the
        rule: "every sibling finished" is not the same fact as "everything
        built has been sent", and three places used the first as a proxy for
        the second. The terminal chunks were then handed out while built
        chunks were still queued -- and once the finals go, `completed` is set
        and the caller pops the state, so the queued ones are gone. Measured
        at the production `max_items`: a response with nine tool calls
        delivered eight and reported `finish_reason: tool_calls`; at
        `max_items=1` a single call reported `tool_calls` having sent no
        `tool_calls` delta at all.
        """
        # 【译】唯一的出口。先发排队中的chunk；结尾的chunk只能在它们之后。
        #
        #   每个生产者都经由这里返回，这就是规则的全部："每个兄弟序列都已结束"和"所有已构建的东西都已发送"不是同一个事实，
        #   而有三个地方曾拿前者代替后者。于是结尾chunk在已构建的chunk还排着队时就被交了出去——而结尾chunk一旦发出，
        #   completed 就被置位，调用方弹出状态，排队中的chunk就没了。在生产环境的 max_items 下实测：一个带九个工具调用的响应
        #   只送达了八个，并报告 finish_reason: tool_calls；在 max_items=1 时，单个调用报告了 tool_calls，却一个 tool_calls 增量都没发。
        # 解释：与chat路径共用的说明；completion路径没有工具调用，但同样的排序规则适用
        out = self._take_pending(remaining_capacity)
        # `_take_pending` never returns more than the caller asked for, so no
        # room left means chunks are still queued -- there is nothing to test
        # for beyond that, and a second condition here would be a guard that
        # cannot fail.
        # 【译】_take_pending 返回的数量从不超过调用方要求的，所以没有剩余空间就意味着还有chunk在排队——除此之外没有别的需要检查的，
        #   在这里再加第二个条件将会是一个永远不会失败的守卫。
        if len(out) >= remaining_capacity:
            return out
        if self._pending_final_chunks is None:
            if not all(self.finished):
                return out
            self._pending_final_chunks = self._build_final_chunks()
        return out + self._drain_pending_final_chunks(remaining_capacity - len(out))

    # 按剩余容量交出结尾chunk；全部交完时置 completed=True
    def _drain_pending_final_chunks(self, remaining_capacity: int) -> list[str]:
        if not self._pending_final_chunks or remaining_capacity <= 0:
            return []
        chunks = self._pending_final_chunks[:remaining_capacity]
        del self._pending_final_chunks[:remaining_capacity]
        if not self._pending_final_chunks:
            self.completed = True
        return chunks

    # 错误帧：data: {"error": {"message": ...}}
    @staticmethod
    def _error_chunk(message: str) -> str:
        return f"data: {json.dumps({'error': {'message': message}})}\n\n"


# 对Rust暴露的门面：Rust的 AtomStandaloneRouter 按名字调用它的 chat_completions / completions / start_*_stream /
#   drain_*_stream / close_*_stream / close；启动时解析一次推理方言、推理开关和工具调用解析器，之后每个请求共用
class AtomStandaloneService:
    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        model_name: str,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        tool_call_parser: str | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.custom_message_encoder = load_custom_message_encoder(model_name)
        # Resolved once here, for the same reason the OpenAI server resolves it
        # once: the chat template says which format this model emits, and
        # deciding from the output instead means deciding from a prefix.
        # 【译】在这里只解析一次，原因与OpenAI服务只解析一次相同：chat template说明了这个模型会输出哪种格式，
        #   而改从输出去判断就意味着要根据一个前缀来判断。
        # 解释：这段话说的是下面 resolve_tool_call_parser 解析工具调用格式；紧挨着的这一行只是取出chat template源码，供下面几处共用
        _template_source = chat_template_source(tokenizer, self.custom_message_encoder)
        # Which dialect this model's reasoning is written in, once, from the
        # same evidence and by the same function the OpenAI server uses. Both
        # paths through this service then read it the same way; they used to
        # share only a boolean, and the streaming half closed the channel on
        # any registered dialect's marker.
        # 【译】这个模型的推理内容用哪种方言书写，只解析一次，依据的证据和使用的函数都与OpenAI服务相同。之后本服务的两条路径
        #   都以同样的方式读取它；它们以前只共享一个布尔值，流式那一半会在任何一个已注册方言的标记上关闭通道。
        self.reasoning_dialect, _stated = resolve_dialect(
            _template_source,
            render_probe_prompt(tokenizer, self.custom_message_encoder, tools=False)
            or "",
        )
        self.model_starts_in_reasoning = template_opens_reasoning_implicitly(
            _template_source
        )
        # The kwarg that turns this model's reasoning off, so
        # `thinking_switched_off` can tell a switch the template reads from
        # one it ignores.
        # 【译】能关掉这个模型推理的那个kwarg，这样 thinking_switched_off 就能分辨模板会读取的开关和它会忽略的开关。
        self.reasoning_toggle = resolve_reasoning_toggle(
            tokenizer, self.custom_message_encoder
        )
        self.tool_parser_cls = resolve_tool_call_parser(
            tool_call_parser, tokenizer, self.custom_message_encoder, model=model_name
        )

        self.engine_service = AtomEngineService(engine, tokenizer)

        self._streams: dict[str, ChatCompletionStreamState] = {}
        self._completion_streams: dict[str, CompletionStreamState] = {}
        self._streams_lock = threading.Lock()

    # 为一个请求构造推理通道：方言固定为启动时解析的那个；starts_open 表示生成一开始是否已在推理内容里
    def _reasoning_channel(
        self, prompt: str, template_kwargs: dict[str, Any] | None = None
    ) -> ReasoningChannel:
        """How to read this response's reasoning channel.

        One function, read by both delivery modes, so they cannot be given
        different dialects -- which is what they were: the non-streaming split
        tried each registered dialect in turn and the streaming filter closed
        on the union of all their end markers.

        `model_starts_in_reasoning` describes the template with reasoning
        *on*, so it only holds while reasoning is on. This service takes no
        `thinking` field, but it does merge `--default-chat-template-kwargs`
        and the request's own `chat_template_kwargs` into the render -- so an
        operator or a client can set the template's switch directly, and then
        the rendered prompt does not open the channel. OR-ing the model-level
        fact in regardless made a DeepSeek-R1-shaped model return its whole
        answer as `reasoning_content` with `content` empty, on both delivery
        modes.
        """
        # 【译】如何读取这个响应的推理通道。
        #
        #   一个函数，两种交付方式都读它，所以它们不可能被赋予不同的方言——而以前正是这样：非流式的切分依次尝试每个已注册的方言，
        #   流式过滤器则在所有方言结束标记的并集上关闭通道。
        #
        #   model_starts_in_reasoning 描述的是推理"打开"时的模板，所以它只在推理打开时成立。本服务不接受 thinking 字段，
        #   但它确实会把 --default-chat-template-kwargs 和请求自带的 chat_template_kwargs 合并进渲染——所以运维人员或客户端
        #   可以直接设置模板的开关，这时渲染出的prompt并不会打开推理通道。不加区分地把模型层面的这个事实OR进去，
        #   曾让一个DeepSeek-R1形态的模型在两种交付方式下都把整个回答当作 reasoning_content 返回、content 为空。
        return ReasoningChannel(
            dialect=self.reasoning_dialect,
            starts_open=prompt_starts_in_reasoning(prompt)
            or (
                self.model_starts_in_reasoning
                and not thinking_switched_off(template_kwargs, self.reasoning_toggle)
            ),
        )

    # 非流式chat：规范化请求 -> 校验模型名 -> 合并template kwargs并渲染prompt -> 构造采样参数 -> engine_service.generate（阻塞）->
    #   build_chat_response(_multi) 做推理内容分离和工具调用解析 -> 转成可JSON序列化的dict返回给Rust
    # 注意：这条路径没有传 kv_transfer_params（completions 路径传了）；stream=True 会直接报错，流式走 start_chat_completions_stream
    def chat_completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        try:
            request_data = self._normalize_chat_request(request_data)
            self._validate_model_name(request_data.get("model"))

            if request_data.get("stream", False):
                raise NotImplementedError(
                    "Streaming chat completions are not implemented for ATOM standalone yet"
                )

            template_kwargs = dict(self.default_chat_template_kwargs)
            if request_data.get("chat_template_kwargs"):
                template_kwargs.update(request_data["chat_template_kwargs"])

            prompt = apply_chat_template(
                self.tokenizer,
                self.custom_message_encoder,
                [
                    self._chat_message_to_template_dict(msg)
                    for msg in self._get_chat_messages(request_data)
                ],
                tools=request_data.get("tools"),
                **template_kwargs,
            )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            data_parallel_rank = self._get_data_parallel_rank(request_data)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            if effective_n > 1:
                outputs = self.engine_service.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    effective_n,
                    data_parallel_rank=data_parallel_rank,
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                response = build_chat_response_multi(
                    request_id,
                    self.model_name,
                    outputs,
                    reasoning=self._reasoning_channel(prompt, template_kwargs),
                    tools=request_data.get("tools"),
                    tool_choice=request_data.get("tool_choice"),
                    tool_parser_cls=self.tool_parser_cls,
                )
            else:
                outputs = self.engine_service.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    effective_n,
                    data_parallel_rank=data_parallel_rank,
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                final_output = outputs[0]
                response = build_chat_response(
                    request_id,
                    self.model_name,
                    final_output["text"],
                    final_output,
                    reasoning=self._reasoning_channel(prompt, template_kwargs),
                    tools=request_data.get("tools"),
                    tool_choice=request_data.get("tool_choice"),
                    tool_parser_cls=self.tool_parser_cls,
                )
            return self._json_safe(response.model_dump(exclude_none=True))
        except Exception:
            logger.exception("ATOM standalone chat_completions failed")
            raise

    # 非流式completion：只支持一个prompt；kv_transfer_params 原样交给引擎
    # 【分支】PD分离：kv_transfer_params 是prefill / decode节点之间交接KV的参数，见第9批
    def completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        try:
            request_data = dict(request_data)
            self._validate_model_name(request_data.get("model"))

            if request_data.get("stream", False):
                raise ValueError(
                    "Use start_completions_stream for streaming completions"
                )

            prompts = self._get_completion_prompts(request_data)
            if len(prompts) != 1:
                raise ValueError(
                    "ATOM standalone /v1/completions currently supports exactly one prompt per request"
                )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"cmpl-{uuid.uuid4().hex}"
            outputs = self.engine_service.generate(
                prompts[0],
                sampling_params,
                request_id,
                effective_n,
                kv_transfer_params=request_data.get("kv_transfer_params"),
                data_parallel_rank=self._get_data_parallel_rank(request_data),
            )
            if not outputs:
                raise RuntimeError("No output generated")
            if effective_n > 1:
                response = build_completion_response_multi(
                    request_id, self.model_name, outputs
                )
            else:
                response = build_completion_response(
                    request_id, self.model_name, outputs[0]
                )
            return self._json_safe(response.model_dump(exclude_none=True))
        except Exception:
            logger.exception("ATOM standalone completions failed")
            raise

    # 流式completion的开始：提交给引擎服务，注册 CompletionStreamState，把stream_id返回给Rust
    def start_completions_stream(self, request_data: dict[str, Any]) -> str:
        try:
            request_data = dict(request_data)
            self._validate_model_name(request_data.get("model"))

            prompts = self._get_completion_prompts(request_data)
            if len(prompts) != 1:
                raise ValueError(
                    "ATOM standalone /v1/completions currently supports exactly one prompt per request"
                )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"cmpl-{uuid.uuid4().hex}"
            prompt = prompts[0]
            stream_queue = self.engine_service.start_stream(
                prompt,
                sampling_params,
                request_id,
                effective_n,
                kv_transfer_params=request_data.get("kv_transfer_params"),
                data_parallel_rank=self._get_data_parallel_rank(request_data),
            )
            stream_state = CompletionStreamState(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                tokenizer=self.tokenizer,
                stream_queue=stream_queue,
                n=effective_n,
            )
            with self._streams_lock:
                self._completion_streams[request_id] = stream_state
            return request_id
        except Exception:
            logger.exception("ATOM standalone start_completions_stream failed")
            raise

    # Rust轮询调用：取一批SSE帧；流已完成或已关闭时从注册表移除，并让引擎服务忘掉它；未知的stream_id直接返回 [DONE]
    def drain_completions_stream(
        self,
        stream_id: str,
        max_items: int = 16,
        timeout: float = 0.05,
    ) -> list[str]:
        with self._streams_lock:
            stream_state = self._completion_streams.get(stream_id)
        if stream_state is None:
            return [STREAM_DONE_MESSAGE]

        chunks = stream_state.drain(max_items=max_items, timeout=timeout)
        if stream_state.completed or stream_state.closed:
            with self._streams_lock:
                self._completion_streams.pop(stream_id, None)
            # A stream that ends on its own is never closed by the caller, so
            # this is the only place its sequence list is dropped. Without it
            # the engine service remembers every completed stream for the life
            # of the process.
            # 【译】一个自然结束的流从不会被调用方关闭，所以这里是丢掉它的序列列表的唯一地方。没有这一步，
            #   引擎服务会在进程的整个生命周期内记住每一个已完成的流。
            # 解释：Rust路由器拿到 [DONE] 后其实仍会调用 close_*_stream，但那时注册表里已没有这个流，close直接返回，
            #   不会再走 abort_stream（见 close_completions_stream 的docstring）
            self.engine_service.forget_stream(stream_id)
        return chunks

    # 一次只取一帧的便捷封装（max_items=1）；在 atom/ 和Rust路由器中都没有调用方
    def poll_completions_stream(
        self, stream_id: str, timeout: float = 1.0
    ) -> str | None:
        chunks = self.drain_completions_stream(
            stream_id,
            max_items=1,
            timeout=timeout,
        )
        if not chunks:
            return None
        return chunks[0]

    # 关闭流：只有真正从注册表弹出了这个流的那次调用，才去中止引擎里的序列
    def close_completions_stream(self, stream_id: str) -> None:
        """Close a stream the caller is finished with.

        The abort is conditional on this call being the one that removed the
        stream. A stream that ended on its own was already removed by
        `drain_*`, which called `forget_stream`; the router then closes it
        anyway, as it is documented to. Aborting unconditionally at that point
        reached `abort_stream` with nothing left to pop, which reads as "not
        submitted yet" and permanently remembered the id -- one leaked entry
        per *successful* request, which is the same unbounded growth
        `forget_stream` exists to prevent.

        `_completion_streams` is the liveness answer and it is exact: the entry is
        added before the engine request is submitted and removed either here
        or by the drain that saw the stream finish. So a stream closed during
        the pre-submit window still aborts, which is the case `_abandoned` was
        built for.
        """
        # 【译】关闭一个调用方已用完的流。
        #
        #   中止以"本次调用就是移除这个流的那一次"为条件。一个自然结束的流已经被 drain_* 移除了，drain_* 也调用过 forget_stream；
        #   之后路由器仍会关闭它，这是有文档说明的行为。在那时无条件中止，会让 abort_stream 没有东西可弹出，被解读为"还没提交"，
        #   于是永久记住这个id——每个成功的请求泄漏一项，这正是 forget_stream 要防止的那种无界增长。
        #
        #   _completion_streams 是"是否还活着"的答案，而且是精确的：这一项在引擎请求被提交之前加入，要么在这里移除，
        #   要么由看到流结束的那次drain移除。所以在提交前的窗口内被关闭的流仍然会被中止，这正是 _abandoned 要处理的情况。
        # 解释：严格说 start_completions_stream 是先调 start_stream 入队、再登记到 _completion_streams，工作线程可能在登记前就已提交；
        #   但Rust要等start返回stream_id之后才可能调用close，所以close总能在注册表里找到它
        with self._streams_lock:
            stream_state = self._completion_streams.pop(stream_id, None)
        if stream_state is None:
            return
        stream_state.close()
        self.engine_service.abort_stream(stream_id)

    # 流式chat的开始：渲染prompt、提交给引擎服务、注册 ChatCompletionStreamState（带推理通道和工具调用解析器），返回stream_id
    #   注意：这里同样没有传 kv_transfer_params
    def start_chat_completions_stream(self, request_data: dict[str, Any]) -> str:
        try:
            request_data = self._normalize_chat_request(request_data)
            self._validate_model_name(request_data.get("model"))

            template_kwargs = dict(self.default_chat_template_kwargs)
            if request_data.get("chat_template_kwargs"):
                template_kwargs.update(request_data["chat_template_kwargs"])

            prompt = apply_chat_template(
                self.tokenizer,
                self.custom_message_encoder,
                [
                    self._chat_message_to_template_dict(msg)
                    for msg in self._get_chat_messages(request_data)
                ],
                tools=request_data.get("tools"),
                **template_kwargs,
            )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            stream_queue = self.engine_service.start_stream(
                prompt,
                sampling_params,
                request_id,
                effective_n,
                data_parallel_rank=self._get_data_parallel_rank(request_data),
            )
            stream_state = ChatCompletionStreamState(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                tokenizer=self.tokenizer,
                stream_queue=stream_queue,
                n=effective_n,
                tool_parser_cls=self.tool_parser_cls,
                reasoning=self._reasoning_channel(prompt, template_kwargs),
                tools=request_data.get("tools"),
                tool_choice=request_data.get("tool_choice"),
            )
            with self._streams_lock:
                self._streams[request_id] = stream_state
            return request_id
        except Exception:
            logger.exception("ATOM standalone start_chat_completions_stream failed")
            raise

    # 同 drain_completions_stream，作用于 _streams
    def drain_chat_completions_stream(
        self,
        stream_id: str,
        max_items: int = 16,
        timeout: float = 0.05,
    ) -> list[str]:
        with self._streams_lock:
            stream_state = self._streams.get(stream_id)
        if stream_state is None:
            return [STREAM_DONE_MESSAGE]

        chunks = stream_state.drain(max_items=max_items, timeout=timeout)
        if stream_state.completed or stream_state.closed:
            with self._streams_lock:
                self._streams.pop(stream_id, None)
            # A stream that ends on its own is never closed by the caller, so
            # this is the only place its sequence list is dropped. Without it
            # the engine service remembers every completed stream for the life
            # of the process.
            # 【译】一个自然结束的流从不会被调用方关闭，所以这里是丢掉它的序列列表的唯一地方。没有这一步，
            #   引擎服务会在进程的整个生命周期内记住每一个已完成的流。
            self.engine_service.forget_stream(stream_id)
        return chunks

    # 一次只取一帧的便捷封装；在 atom/ 和Rust路由器中都没有调用方
    def poll_chat_completions_stream(
        self, stream_id: str, timeout: float = 1.0
    ) -> str | None:
        chunks = self.drain_chat_completions_stream(
            stream_id,
            max_items=1,
            timeout=timeout,
        )
        if not chunks:
            return None
        return chunks[0]

    def close_chat_completions_stream(self, stream_id: str) -> None:
        """Close a stream the caller is finished with.

        Identical to `close_completions_stream` over `_streams` instead of
        `_completion_streams` -- see there for why the abort is conditional on
        this call being the one that removed the stream.
        """
        # 【译】关闭一个调用方已用完的流。
        #
        #   与 close_completions_stream 相同，只是作用于 _streams 而不是 _completion_streams——为什么中止要以
        #   "本次调用就是移除这个流的那一次"为条件，见那里。
        with self._streams_lock:
            stream_state = self._streams.pop(stream_id, None)
        if stream_state is None:
            return
        stream_state.close()
        self.engine_service.abort_stream(stream_id)

    # 服务退出（Rust关闭或drop路由器时调用）：关闭并中止所有打开的流，然后关闭引擎服务和引擎
    def close(self) -> None:
        """Shut down, in the order `close_*_stream` uses.

        `stream_state.close()` is the only thing that sets `closed`, and a
        router thread already inside `drain()` does not hold `_streams_lock`:
        it wakes from its queue, reads `closed` as False and hands out chunks
        for a sequence this method just aborted -- and if `all(finished)`
        happens to hold, a finish/usage/`[DONE]` tail as well, so the caller
        records `finish_reason: stop` on a truncated answer. Clearing the
        registry is not what stops a drain; closing the state is.
        """
        # 【译】关闭服务，顺序与 close_*_stream 相同。
        #
        #   只有 stream_state.close() 会设置 closed，而一个已经在 drain() 里面的路由器线程并不持有 _streams_lock：
        #   它从队列上醒来，读到 closed 为False，就会为这个方法刚刚中止的序列交出chunk——如果恰好 all(finished) 成立，
        #   还会交出 finish/usage/[DONE] 结尾，于是调用方对一个被截断的回答记录下 finish_reason: stop。
        #   停止drain的不是清空注册表，而是关闭状态对象。
        with self._streams_lock:
            open_streams = [
                *self._streams.items(),
                *self._completion_streams.items(),
            ]
            self._streams.clear()
            self._completion_streams.clear()
        for stream_id, stream_state in open_streams:
            stream_state.close()
            self.engine_service.abort_stream(stream_id)
        if hasattr(self, "engine_service"):
            self.engine_service.close()
        if hasattr(self.engine, "close"):
            self.engine.close()

    # 兼容新字段名：只给了 max_completion_tokens 时把它当作 max_tokens
    @staticmethod
    def _normalize_chat_request(request_data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(request_data)
        if (
            normalized.get("max_tokens") is None
            and normalized.get("max_completion_tokens") is not None
        ):
            normalized["max_tokens"] = normalized["max_completion_tokens"]
        return normalized

    # 读取路由器注入的 data_parallel_rank（DP attention下指定由哪个DP rank处理）
    @staticmethod
    def _get_data_parallel_rank(request_data: dict[str, Any]) -> int | None:
        """Extract the DP-attention rank, if available.

        Routers can inject a ``data_parallel_rank`` field into the request body to
        indicate which DP rank to route to. Falls back to round-robin if not available.
        """
        # 【译】提取DP-attention的rank（如果有）。
        #
        #   路由器可以往请求体里注入一个 data_parallel_rank 字段，指明要路由到哪个DP rank。没有时退回到轮询（round-robin）。
        # 解释：返回None时由CoreManager按DP负载均衡策略选rank，默认策略是 least_requests
        #   （见本批 engine_core_mgr.py 的 DP_LB_DEFAULT），不一定是round-robin
        raw = request_data.get("data_parallel_rank")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"data_parallel_rank must be an integer, got {raw!r}")

    # 请求里的model必须与加载的模型名一致（None和"unknown"放行），否则抛ValueError（Rust侧把ValueError映射成400）
    def _validate_model_name(self, request_model: str | None) -> None:
        if (
            request_model is not None
            and request_model != "unknown"
            and request_model != self.model_name
        ):
            raise ValueError(
                f"requested model `{request_model}` does not match loaded model `{self.model_name}`"
            )

    # 取messages；没有时退而取prompt字段
    @staticmethod
    def _get_chat_messages(request_data: dict[str, Any]) -> list[dict[str, Any]]:
        messages = request_data.get("messages")
        if messages is None:
            messages = request_data.get("prompt")
        if messages is None:
            raise ValueError("Either 'messages' or 'prompt' field is required")
        return messages

    # prompt只接受字符串或字符串列表（调用方再限制为恰好一个）
    @staticmethod
    def _get_completion_prompts(request_data: dict[str, Any]) -> list[str]:
        prompt = request_data.get("prompt")
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            return prompt
        raise ValueError(
            "Completion request field 'prompt' must be a string or list of strings"
        )

    # 把一条chat消息整理成chat template要的dict：content是部件列表时只拼接text部件（图片等多模态部件被丢弃），
    #   并保留 tool_calls / tool_call_id / name / reasoning_content
    @staticmethod
    def _chat_message_to_template_dict(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif content is None:
            content = ""

        template_message = {
            "role": message.get("role"),
            "content": content,
        }
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
            if key in message:
                template_message[key] = message[key]
        return template_message

    # 同时兼容dict和对象的取字段；值为None时用默认值
    @staticmethod
    def _request_field(request: Any, field: str, default: Any = None) -> Any:
        if isinstance(request, dict):
            value = request.get(field, default)
        else:
            value = getattr(request, field, default)
        return default if value is None else value

    # 复用 api_server._build_sampling_params 构造SamplingParams（只取 temperature / max_tokens / stop / ignore_eos / top_k / top_p / n）
    def _build_sampling_params(
        self,
        request: dict[str, Any] | CompletionRequest,
        effective_n: int,
    ) -> SamplingParams:
        return _build_sampling_params(
            temperature=self._request_field(
                request, "temperature", DEFAULT_TEMPERATURE
            ),
            max_tokens=self._request_field(request, "max_tokens", DEFAULT_MAX_TOKENS),
            stop_strings=self._normalize_stop_strings(
                self._request_field(request, "stop")
            ),
            ignore_eos=self._request_field(request, "ignore_eos", False),
            top_k=self._request_field(request, "top_k", DEFAULT_TOP_K),
            top_p=self._request_field(request, "top_p", DEFAULT_TOP_P),
            n=effective_n,
        )

    # stop只接受字符串或字符串列表
    @staticmethod
    def _normalize_stop_strings(stop: Any) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
            return stop
        raise ValueError("Request field 'stop' must be a string or list of strings")

    # 把一个输出规范化成dict（字符串输出补齐各字段）；在 atom/ 内没有调用方
    @staticmethod
    def _normalize_output(output: Any) -> dict[str, Any]:
        if isinstance(output, str):
            return {
                "text": output,
                "finish_reason": None,
                "num_tokens_input": 0,
                "num_tokens_output": 0,
                "ttft": 0.0,
                "tpot": 0.0,
                "latency": 0.0,
            }
        return AtomStandaloneService._json_safe(dict(output))

    # 递归转换成可JSON序列化的原生类型：整数 / 实数（含numpy标量）转int / float，带 .item() 的对象（如张量标量）取其值，
    #   因为返回值要经PyO3转成Rust的JSON
    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): AtomStandaloneService._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [AtomStandaloneService._json_safe(item) for item in value]
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            return AtomStandaloneService._json_safe(value.item())
        return value
