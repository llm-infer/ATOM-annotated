# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pydantic request/response models for the OpenAI-compatible API."""

# 【译】OpenAI兼容API的pydantic请求/响应模型。
#
# 本文件（第2批2c，B级）：HTTP请求体 / 响应体的数据结构，以及几个与协议相关的小函数
#   请求模型：ChatCompletionRequest（/v1/chat/completions）、CompletionRequest（/v1/completions）、ChatMessage
#   响应模型：ChatCompletionResponse、CompletionResponse、ModelCard / ModelList（/v1/models）
#   协议函数：validate_max_tokens、resolve_prompt_token_ids、openai_stop_reason(_with_calls)
# pydantic（第一次出现，讲一次）：继承BaseModel、用类型注解声明字段的类。FastAPI接口的参数标注为这种类时，
#   会把JSON请求体按注解解析并校验（类型不对就直接返回错误响应），得到一个实例；model_dump()把实例转回dict。
#   model_config的extra控制未声明的字段："ignore"静默丢弃，"allow"保留（可从model_extra读取）
# 调用方：api_server.py（接口参数类型）、serving_chat.py / serving_completion.py（构造响应）
import json
import time
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    TypeAdapter,
    ValidationError,
)

# ============================================================================
# Constants
# ============================================================================
# 【译】常量
# 采样参数的默认值：请求没传时使用（chat / completions接口不读模型的generation_config）
#   temperature=1.0、top_k=-1（不限制）、top_p=1.0（不限制）、max_tokens=8192
# *_OBJECT：响应里object字段的固定取值；STREAM_DONE_MESSAGE：OpenAI流式响应的最后一帧

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = -1
DEFAULT_TOP_P = 1.0
# max_tokens的作用：给输出长度一个上限，同时也是API层长度校验的依据（prompt + max_tokens ≤ max_model_len，见api_server._validate_context_length）；
#   默认值8192意味着：没传max_tokens时，prompt超过 max_model_len - 8192 个token的请求会被直接拒绝（400）
DEFAULT_MAX_TOKENS = 8192
CHAT_COMPLETION_OBJECT = "chat.completion"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
TEXT_COMPLETION_OBJECT = "text_completion"
STREAM_DONE_MESSAGE = "data: [DONE]\n\n"


# Valid OpenAI ``tool_choice`` string values and the function-name constraint.
# Spec-level (not model-specific): the same for every model served.
# 【译】OpenAI tool_choice 合法的字符串取值，以及函数名约束。属于规范层面（与模型无关）：对所有被服务的模型都一样。
# 解释：这里只定义了字符串取值集合；函数名的校验在tool_parser的qualified_tool_name里（serving_chat.validate_chat_request调用）
TOOL_CHOICE_VALUES = frozenset({"auto", "none", "required"})


# max_tokens < 1 时抛ValueError（接口返回400）；调用方：两个请求模型的get_max_tokens、api_server.anthropic_messages
def validate_max_tokens(max_tokens: int) -> int:
    """Return a valid generation limit or raise a client-facing error."""
    # 【译】返回合法的生成上限，否则抛出面向客户端的错误。
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    return max_tokens


#: Validate token IDs with pydantic-core, avoiding a Python loop.
# 【译】用pydantic-core校验token ID，避免Python循环。
# 解释："#:"是Sphinx的属性文档注释写法。list[NonNegativeInt]要求每个元素是>=0的整数；
#   TypeAdapter让一个普通类型（不是BaseModel）也能用pydantic校验，下面用它校验kv_transfer_params里的ID
PromptTokenIds = list[NonNegativeInt]
_PROMPT_TOKEN_IDS_ADAPTER = TypeAdapter(PromptTokenIds)


# prompt的token ID可以来自两处：请求顶层的prompt_token_ids，或kv_transfer_params["prompt_token_ids"]
#   （PD路由器按vLLM的约定放在这里，【分支】第9批）。两处都没有 → None，调用方去渲染模板 / tokenize
# 例：prompt_token_ids=[1, 2, 3]且kv_transfer_params里也是[1, 2, 3] → [1, 2, 3]；两者不同 → ValueError；[] → ValueError
def resolve_prompt_token_ids(
    prompt_token_ids: PromptTokenIds | None,
    kv_transfer_params: dict[str, Any] | None,
) -> PromptTokenIds | None:
    """Resolve IDs from the top-level field or vLLM-compatible PD metadata.

    Reject empty lists and conflicting IDs between the two locations.
    """
    # 【译】从顶层字段或vLLM兼容的PD元数据中解析出ID。
    #
    #   拒绝空列表，以及两处位置的ID互相冲突的情况。
    from_kv = (kv_transfer_params or {}).get("prompt_token_ids")
    if prompt_token_ids is None and from_kv is None:
        return None

    if from_kv is not None:
        try:
            from_kv = _PROMPT_TOKEN_IDS_ADAPTER.validate_python(from_kv)
        except ValidationError as e:
            raise ValueError(
                "kv_transfer_params['prompt_token_ids'] must be a list of "
                f"non-negative integers: {e.errors()[0]['msg']}"
            ) from e
        if prompt_token_ids is not None and from_kv != prompt_token_ids:
            raise ValueError(
                "prompt_token_ids and kv_transfer_params['prompt_token_ids'] "
                "disagree; a request cannot carry two different prompts"
            )

    ids = prompt_token_ids if prompt_token_ids is not None else from_kv
    if not ids:
        raise ValueError("prompt_token_ids was given but is empty")
    return ids


# 引擎的结束原因（Sequence.leave_reason）→ OpenAI的finish_reason。例：
#   None → None（还没结束）；"max_tokens" → "length"；"eos" / "stop_151645" / "stop_sequence" / "aborted" → "stop"
def openai_stop_reason(finish_reason: str | None) -> str | None:
    """The engine's leave reason as OpenAI spells it.

    The engine says `eos` / `max_tokens` / `stop_sequence` / `stop_<token_id>`
    / `aborted` / `unschedulable: ...`; OpenAI clients understand only `stop` /
    `length` / `tool_calls`. `stop_<token_id>` is an ordinary end of turn --
    any model declaring more than one EOS reaches it in normal operation.

    Named for the vocabulary it maps *into*, and paired with
    `api_server.anthropic_stop_reason`. Two functions rather than one with a
    mode: the two vocabularies share no member, so chaining them would send
    every reason to the other's default.
    """
    # 【译】按OpenAI的拼法给出引擎的结束原因。
    #
    #   引擎给出的是 eos / max_tokens / stop_sequence / stop_<token_id> / aborted / unschedulable: ...；
    #   OpenAI客户端只认识 stop / length / tool_calls。stop_<token_id> 是普通的回合结束——
    #   任何声明了多个EOS的模型在正常运行中都会遇到。
    #
    #   以它映射"到"的词汇命名，与 api_server.anthropic_stop_reason 成对。用两个函数而不是一个带模式参数的函数：
    #   两套词汇没有共同成员，把它们串起来会让每个原因都落到对方的默认值。
    if finish_reason is None:
        return None
    if finish_reason in ("stop", "length", "tool_calls"):
        return finish_reason
    if finish_reason in ("max_tokens", "max_new_tokens"):
        return "length"
    # 其余一律报"stop"，包括aborted和unschedulable: ...——客户端从finish_reason上区分不出"自然结束"和"被引擎放弃"
    return "stop"


# 综合"引擎结束原因"和"是否解析出工具调用"得到最终的finish_reason：length > tool_calls > 其他
def openai_stop_reason_with_calls(engine_reason: str | None, has_calls: bool) -> str:
    """The reason to report when a call was parsed and the engine had its own.

    `length` outranks `tool_calls`, because they answer different questions
    and only one of them is a warning: `tool_calls` says "act on this", and
    `length` says "this is not all of it". A response cut off mid-call parses
    to a call with a silently truncated argument value -- every format's
    unclosed-region branch exists to salvage exactly that -- and reporting
    `tool_calls` for it told the client to run a tool with half its arguments
    and no indication anything was missing. OpenAI reports `length` for a
    truncated response whatever else is in it.
    """
    # 【译】解析出了工具调用、而引擎也有自己的结束原因时，应该报告哪个原因。
    #
    #   length 优先于 tool_calls，因为两者回答的是不同的问题，而且只有一个是警告：tool_calls 说"照此执行"，
    #   length 说"这不是全部"。在调用中途被截断的响应会解析出一个参数值被悄悄截断的调用——每种格式的"未闭合区域"分支
    #   正是为抢救这种情况而存在——而为它报告 tool_calls，等于让客户端用一半的参数去执行工具，还没有任何迹象表明缺了东西。
    #   对被截断的响应，不管里面还有什么，OpenAI都报告 length。
    normalized = openai_stop_reason(engine_reason)
    if normalized == "length":
        return "length"
    return "tool_calls" if has_calls else (normalized or "stop")


# ============================================================================
# Request Models
# ============================================================================
# 【译】请求模型


# 下面两个函数处理"客户端回传的历史工具调用"：把arguments从JSON字符串转回dict，好让chat template渲染
# 例：原文 {"re": "\d+"} 里的 \d 不是合法的JSON转义 → 改成 {"re": "\\d+"}，json.loads就能解析
def _fix_invalid_json_escapes(s: str) -> str:
    """Fix invalid JSON escapes in model-generated tool-call arguments.

    Models occasionally produce invalid escape sequences like ``\\k`` or
    ``\\p`` in function.arguments JSON. ``json.loads`` rejects these. This
    helper doubles any backslash not followed by a valid JSON escape char.
    """
    # 【译】修复模型生成的工具调用参数中非法的JSON转义。
    #
    #   模型偶尔会在 function.arguments 的JSON里产生 \k、\p 这样的非法转义序列，json.loads 会拒绝它们。
    #   这个辅助函数把后面没有跟合法JSON转义字符的反斜杠都改成双反斜杠。
    _VALID = frozenset('"\\bfnrtu/')
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            if i + 1 >= len(s):
                out.append("\\\\")
                i += 1
            elif s[i + 1] == "\\":
                out.append("\\\\")
                i += 2
            elif s[i + 1] in _VALID:
                out.append("\\")
                out.append(s[i + 1])
                i += 2
            else:
                out.append("\\\\")
                out.append(s[i + 1])
                i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _normalize_tool_call_arguments(tool_calls: Any) -> Any:
    """Deserialize ``function.arguments`` from a JSON string to a mapping.

    OpenAI clients send tool-call arguments as a JSON *string*, but chat
    templates (Qwen3 qwen3_coder/qwen3_xml, Hermes, etc.) iterate
    ``tool_call.arguments.items()`` and require a mapping. Mirrors how vLLM and
    SGLang deserialize arguments before applying the chat template.
    """
    # 【译】把 function.arguments 从JSON字符串反序列化成映射（dict）。
    #
    #   OpenAI客户端以JSON"字符串"发送工具调用参数，但chat template（Qwen3的qwen3_coder/qwen3_xml、Hermes等）会迭代
    #   tool_call.arguments.items()，要求是一个映射。这与vLLM和SGLang在套用chat template之前反序列化参数的做法一致。
    # 解析顺序：json.loads → 修复转义后再试 → 都失败则包成 {"_raw": 原字符串}
    if not isinstance(tool_calls, list):
        return tool_calls
    normalized = []
    for tc in tool_calls:
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
            fn = dict(tc["function"])
            if isinstance(fn.get("arguments"), str):
                raw = fn["arguments"]
                try:
                    fn["arguments"] = json.loads(raw)
                except (ValueError, TypeError):
                    try:
                        fn["arguments"] = json.loads(_fix_invalid_json_escapes(raw))
                    except (ValueError, TypeError):
                        fn["arguments"] = {"_raw": raw}
            tc = {**tc, "function": fn}
        normalized.append(tc)
    return normalized


# 一条chat消息：role（system / user / assistant / tool）+ content
#   content：字符串；或多模态的片段列表（[{"type": "text", ...}, {"type": "image_url", ...}]）；或None（只有工具调用的assistant消息）
#   extra="allow"：tool_calls、tool_call_id、name、reasoning_content、tools等未声明字段也会保留，可从model_extra读取
class ChatMessage(BaseModel):
    """Represents a single chat message."""

    # 【译】表示一条chat消息。

    role: str
    content: str | list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow")

    def get_content_text(self) -> str:
        """Extract text content, handling both string and multimodal content parts."""
        # 【译】提取文本内容，同时处理字符串和多模态内容片段两种形式。
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # OpenAI multimodal format: [{"type": "text", "text": "..."}, ...]
        # 【译】OpenAI的多模态格式：[{"type": "text", "text": "..."}, ...]
        # 只取text片段、用换行拼接，媒体片段被丢弃（多模态请求走的是chat_utils.parse_chat_messages，不经过这里）
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)

    # 转成交给chat template的dict（apply_chat_template调用）
    #   preserve_content=True时保留原始content（可能是片段列表），否则压成纯文本；由消息编码器的preserve_content决定
    #   （DeepSeek-V4.1的encoder为True，见chat_encoder_adapters.py）
    def to_template_dict(self, *, preserve_content: bool = False) -> dict[str, Any]:
        """Convert to dict for chat template, preserving tool-related fields.

        Returns a dict with role, content, and any extra fields (tool_calls,
        tool_call_id, name, reasoning_content, tools) that the chat template needs.
        """
        # 【译】转换成给chat template用的dict，保留与工具相关的字段。
        #
        #   返回的dict包含role、content，以及chat template需要的额外字段（tool_calls、tool_call_id、name、reasoning_content、tools）。
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content if preserve_content else self.get_content_text(),
        }
        # Preserve extra fields needed by chat templates (e.g. Kimi-K2/K3).
        # "tools" carries K3 dynamically-loaded tools declared inside a system
        # message; encoding_k3.build_chat_segments renders them per-message.
        # 【译】保留chat template需要的额外字段（例如Kimi-K2/K3）。
        #   "tools" 携带K3在system消息里声明的动态加载工具；encoding_k3.build_chat_segments 会按消息渲染它们。
        extras = self.model_extra or {}
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "tools"):
            if key in extras:
                d[key] = (
                    _normalize_tool_call_arguments(extras[key])
                    if key == "tool_calls"
                    else extras[key]
                )
        return d


# /v1/chat/completions的请求体。FastAPI据此解析校验JSON，api_server.chat_completions使用其中的字段
class ChatCompletionRequest(BaseModel):
    """Request model for chat completions (OpenAI-compatible)."""

    # 【译】chat completions的请求模型（OpenAI兼容）。

    # extra="ignore"：未声明的字段被静默丢弃——所以想支持的扩展字段（如thinking）必须在这里显式声明
    model_config = {"extra": "ignore"}

    model: str | None = None
    messages: list[ChatMessage] | None = None
    # 【译】（行尾注释）接受 'prompt' 作为别名
    prompt: list[ChatMessage] | None = None  # Accept 'prompt' as alias
    temperature: float | None = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    top_p: float | None = DEFAULT_TOP_P
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    # max_completion_tokens是OpenAI较新的字段名，优先于max_tokens（见get_max_tokens）
    max_completion_tokens: int | None = None
    # stop：停止字符串列表 → SamplingParams.stop_strings；ignore_eos：遇到EOS也不停（压测常用）
    # stop的语义与OpenAI有两点不同：①引擎做的是token级匹配（stop字符串先被tokenize成token序列，见llm_engine.py的preprocess），
    #   生成时同一段文字若被切成别的token就匹配不上；②按调度器的截断逻辑，命中的stop序列本身保留在输出里（OpenAI的返回文本不含它）。
    #   命中后leave_reason为"stop_sequence"，对外报告为finish_reason="stop"
    # ignore_eos：压测时用，让每个请求都正好生成max_tokens个token，输出长度固定，吞吐数字才可复现、可比较
    stop: list[str] | None = None
    ignore_eos: bool | None = False
    stream: bool | None = False
    # seed：接受但不使用（_build_sampling_params不传它）；chat_template_kwargs：渲染模板时额外传入的参数
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    # Tool calling
    # 【译】工具调用
    tools: list[dict[str, Any]] | None = None
    # 【译】（行尾注释）"auto"、"none"、"required"，或 {function: {name}}
    tool_choice: Any | None = None  # "auto", "none", "required", or {function: {name}}
    # Structured output: {"type": "text"|"json_object"|"json_schema", ...}
    # 【译】结构化输出：{"type": "text"|"json_object"|"json_schema", ...}
    # 只作为chat template参数转发（见api_server.chat_completions），这里没有做约束解码
    response_format: dict[str, Any] | None = None
    # V4.1 also accepts an exact integer budget; bool/float coercion is invalid.
    # 【译】V4.1还接受一个精确的整数预算；bool/float的强制转换是无效的。
    # Annotated[int, Field(strict=True, ge=1, le=100)]：严格模式下True、5.0这类值不会被转成int，取值范围1~100
    reasoning_effort: str | Annotated[int, Field(strict=True, ge=1, le=100)] | None = (
        None
    )
    # K3 thinking control (sent by clients via extra_body):
    # {"type": "enabled"|"disabled", "keep": "all", "effort": "low"|"high"|"max"}.
    # Without this field pydantic (extra="ignore") silently drops it, so effort
    # never reaches the template and the streaming reasoning gate never fires.
    # 【译】K3的思考控制（客户端通过extra_body发送）：
    #   {"type": "enabled"|"disabled", "keep": "all", "effort": "low"|"high"|"max"}。
    #   没有这个字段的话，pydantic（extra="ignore"）会悄悄丢掉它，于是effort永远到不了模板，流式推理门控也永远不会触发。
    thinking: dict[str, Any] | None = None
    # Accepted for compatibility, not actively used:
    # 【译】为兼容而接受，实际不使用：
    # 解释：这句只适用于下面两个penalty字段；n是会用到的（api_server._coerce_n，n>1时扇出多个兄弟序列）
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    n: int | None = 1
    # Optional KV-transfer metadata for P/D disaggregation.
    # 【译】可选的KV传输元数据，用于P/D分离。
    kv_transfer_params: dict[str, Any] | None = None
    # DP路由提示：指定由哪个DP rank处理（如atomesh插入），传给CoreManager
    data_parallel_rank: int | None = None
    # Pre-tokenized text prompt; messages are still required and validated.
    # 【译】预先tokenize好的文本prompt；messages仍然必填并会被校验。
    prompt_token_ids: PromptTokenIds | None = None
    # Echo prompt IDs in non-streaming responses, including n > 1.
    # 【译】在非流式响应（包括 n > 1）里回显prompt ID。
    return_token_ids: bool | None = None

    def get_prompt_token_ids(self) -> PromptTokenIds | None:
        """This request's pre-tokenized prompt, or None to render+tokenize."""
        # 【译】这个请求预先tokenize好的prompt；为None时需要渲染模板再tokenize。
        return resolve_prompt_token_ids(self.prompt_token_ids, self.kv_transfer_params)

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for OpenAI chat requests."""
        # 【译】返回OpenAI chat请求实际的生成上限。
        if self.max_completion_tokens is not None:
            return validate_max_tokens(self.max_completion_tokens)
        if self.max_tokens is not None:
            return validate_max_tokens(self.max_tokens)
        return DEFAULT_MAX_TOKENS

    def get_messages(self) -> list[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field."""
        # 【译】从 'messages' 或 'prompt' 字段取消息。
        if self.messages is not None:
            return self.messages
        elif self.prompt is not None:
            return self.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' field is required")


# /v1/completions的请求体：没有messages，prompt是字符串或token id列表
class CompletionRequest(BaseModel):
    """Request model for text completions (OpenAI-compatible)."""

    # 【译】文本补全的请求模型（OpenAI兼容）。

    model_config = {"extra": "ignore"}

    model: str | None = None
    # Required when no prompt_token_ids are supplied. Per the OpenAI spec this
    # may also be an array of token ids; that form is accepted and routed to the
    # pre-tokenized path (see get_prompt_or_tokens). The nested
    # array-of-token-arrays (batch) form is not supported.
    # 【译】没有提供 prompt_token_ids 时必填。按OpenAI规范，它也可以是token id数组；这种形式被接受，
    #   并走预tokenize路径（见 get_prompt_or_tokens）。不支持嵌套的token数组的数组（批量）形式。
    prompt: str | list[int] | None = None
    temperature: float | None = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    top_p: float | None = DEFAULT_TOP_P
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_completion_tokens: int | None = None
    stop: list[str] | None = None
    ignore_eos: bool | None = False
    stream: bool | None = False
    # Optional KV-transfer metadata for P/D disaggregation.
    # 【译】可选的KV传输元数据，用于P/D分离。
    kv_transfer_params: dict[str, Any] | None = None
    # Optional DPA routing hint inserted by atomesh for DP-aware workers.
    # 【译】可选的DPA路由提示，由atomesh为DP感知的worker插入。
    data_parallel_rank: int | None = None
    n: int | None = 1
    # See `ChatCompletionRequest` for both of these.
    # 【译】这两个字段见 ChatCompletionRequest。
    prompt_token_ids: PromptTokenIds | None = None
    return_token_ids: bool | None = None

    def get_prompt_token_ids(self) -> PromptTokenIds | None:
        """This request's pre-tokenized prompt, or None to tokenize `prompt`."""
        # 【译】这个请求预先tokenize好的prompt；为None时需要tokenize prompt。
        return resolve_prompt_token_ids(self.prompt_token_ids, self.kv_transfer_params)

    def get_prompt_or_tokens(self) -> "str | PromptTokenIds":
        """Return pre-tokenized IDs when supplied, otherwise require text."""
        # 【译】提供了预tokenize的ID就返回它们，否则要求有文本。
        # 优先级：prompt_token_ids（或kv_transfer_params里的）> prompt是token id列表 > prompt是字符串
        ids = self.get_prompt_token_ids()
        if ids is not None:
            return ids
        if self.prompt is None:
            raise ValueError("either 'prompt' or 'prompt_token_ids' is required")
        if isinstance(self.prompt, list):
            # OpenAI-spec `prompt` as an array of token ids: same meaning as
            # prompt_token_ids, so take the pre-tokenized path.
            # 【译】OpenAI规范里以token id数组形式给出的 prompt：含义与 prompt_token_ids 相同，所以走预tokenize路径。
            if not self.prompt:
                raise ValueError("'prompt' was given as a token array but is empty")
            return self.prompt
        return self.prompt

    def get_max_tokens(self) -> int:
        """Return the effective generation cap for completion requests."""
        # 【译】返回completion请求实际的生成上限。
        if self.max_completion_tokens is not None:
            return validate_max_tokens(self.max_completion_tokens)
        if self.max_tokens is not None:
            return validate_max_tokens(self.max_tokens)
        return DEFAULT_MAX_TOKENS


# ============================================================================
# Response Models
# ============================================================================
# 【译】响应模型
# 由serving_chat / serving_completion的build_*函数构造，api_server用model_dump(mode="json")序列化后返回


# 非流式chat响应：choices里每项是{"index", "message", "finish_reason"}（serving_chat._build_chat_choice构造）；
#   usage含prompt_tokens / completion_tokens / total_tokens，ATOM另加了ttft_s / tpot_s / latency_s等
class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    # 【译】chat completions的响应模型。

    id: str
    object: str = CHAT_COMPLETION_OBJECT
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]
    kv_transfer_params: dict[str, Any] | None = None
    # Echoed back only when the request set `return_token_ids`.
    # 【译】只在请求设置了 return_token_ids 时回显。
    prompt_token_ids: PromptTokenIds | None = None

    model_config = ConfigDict(extra="allow")


class CompletionResponse(BaseModel):
    """Response model for text completions."""

    # 【译】文本补全的响应模型。

    id: str
    object: str = TEXT_COMPLETION_OBJECT
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]
    # Optional KV-transfer metadata returned for P/D disaggregation.
    # 【译】为P/D分离返回的可选KV传输元数据。
    kv_transfer_params: dict[str, Any] | None = None
    # Echoed back only when the request set `return_token_ids`.
    # 【译】只在请求设置了 return_token_ids 时回显。
    prompt_token_ids: PromptTokenIds | None = None


class ModelCard(BaseModel):
    """Model card for /v1/models endpoint."""

    # 【译】/v1/models 接口的模型卡片。
    # Field(default_factory=...)：每次创建实例时调用一次，created取当时的时间戳

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "atom"


class ModelList(BaseModel):
    """Response for /v1/models endpoint."""

    # 【译】/v1/models 接口的响应。

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """OpenAI-format error response."""

    # 【译】OpenAI格式的错误响应。
    # 解释：atom/下没有找到使用这个类的地方，错误响应都是在api_server里直接拼dict

    error: dict[str, Any]
