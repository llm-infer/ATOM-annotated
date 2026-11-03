# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-scoped adapters for dynamically loaded chat encoders."""

# 【译】为动态加载的chat编码器提供的、按模型定制的适配器。
#
# 本文件（第2批2c，B级）：MessageEncoderAdapter——把模型仓库里附带的Python编码函数（encode_messages）包装成统一的可调用对象
#   前提：编码函数和Jinja chat template的作用相同——把消息列表拼成带特殊token的prompt字符串（例子、为什么必须用模型自己的格式，
#     见chat_encoders.py文件头的【科普】chat template与特殊token）；DeepSeek-V4、Kimi-K3这类模型随权重发布的是这样一个Python函数
#   1. prepare_messages：按模型改写消息（DeepSeek-V4：在最前面插一条携带工具的system消息；V4.1：deepseek_v41_encoder.py的钩子）
#   2. __call__：补默认参数、按模型改写kwargs，再按编码器的函数签名过滤掉它不接受的kwargs，最后调用编码器
#   3. supports_tools / preserve_content：是否支持工具、是否保留消息content的原始（列表）形式
# 调用方：chat_encoders._load_encoder_from_dir 用 build_message_encoder_adapter 构造它；chat_encoders.apply_chat_template 调用它
# 【分支】DeepSeek-V4 / V4.1的编码细节见第8批
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import deepseek_v41_encoder

logger = logging.getLogger("atom")

# 类型别名：编码器 (messages, **kwargs) -> prompt字符串；消息准备 (messages, tools) -> messages；kwargs准备 kwargs -> kwargs
MessageEncoder = Callable[..., str]
MessagePreparer = Callable[[list[dict], list[dict] | None], list[dict]]
KwargsPreparer = Callable[[dict[str, Any]], dict[str, Any]]


# 默认的kwargs准备：原样返回
def _identity_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs


def _copy_messages(
    messages: list[dict], _tools: list[dict] | None = None
) -> list[dict]:
    """Return shallow message copies without model-specific rewriting."""
    # 【译】返回消息的浅拷贝，不做任何模型特定的改写。
    return [dict(message) for message in messages]


# DeepSeek-V4：tools=[t]时 → [{"role": "system", "tools": [t]}, *原消息]
def _prepare_deepseek_v4_messages(
    messages: list[dict], tools: list[dict] | None
) -> list[dict]:
    """Prepare the internal message shape expected by DSV4 ``encode_messages``.

    The DeepSeek-V4 reference encoder reads tool schemas from a system message's
    ``tools`` field. Match vLLM's model-specific tokenizer wrapper by prepending
    a synthetic tool-carrying system message, without reordering or merging the
    caller's existing messages.
    """
    # 【译】准备DSV4的 encode_messages 期望的内部消息形状。
    #
    #   DeepSeek-V4的参考编码器从system消息的 tools 字段读取工具schema。这里与vLLM的模型专用tokenizer包装保持一致：
    #   在最前面插入一条携带工具的合成system消息，不重排、也不合并调用方已有的消息。
    prepared = _copy_messages(messages)
    if tools:
        prepared.insert(0, {"role": "system", "tools": tools})
    return prepared


# @dataclass(frozen=True)：不可变的数据类，构造后不能再给字段赋值
@dataclass(frozen=True)
class MessageEncoderAdapter:
    """A raw model encoder plus its model-specific message preparation."""

    # 【译】一个原始的模型编码器，加上它的模型特定的消息准备逻辑。

    # name：模块名（如"encoding_dsv4"）；encode：原始编码函数；prepare_messages：消息准备函数；supports_tools：能否接收工具
    name: str
    encode: MessageEncoder
    prepare_messages: MessagePreparer
    supports_tools: bool = False
    # The file the encoder was loaded from. `encode` is a closure defined in
    # `chat_encoders`, so `inspect.getmodule(encode)` names *that* module --
    # reading the model's own source needs the path carried here.
    # 【译】编码器是从哪个文件加载的。encode 是在 chat_encoders 里定义的闭包，所以 inspect.getmodule(encode) 给出的是"那个"模块——
    #   要读模型自己的源码，需要这里带着的路径。
    # 解释：当前的chat_encoders._load_encoder_from_dir直接把模型模块里的encode_messages传进来，并没有包闭包；
    #   source_path仍被chat_encoders.chat_template_source用来读取编码器源码
    source_path: str = ""
    # What this encoder's signature will accept, or ``None`` when it takes
    # `**kwargs` and so accepts anything. See `__call__`.
    # 【译】这个编码器的签名能接受哪些参数；它接受 **kwargs（因而什么都接受）时为 None。见 __call__。
    # field(compare=False)：这个字段不参与相等比较
    accepts: frozenset[str] | None = field(default=None, compare=False)
    # Values this model's loader wants set when the caller did not. Applied
    # after the filter and subject to it, so a default the encoder cannot take
    # is dropped like any other kwarg rather than raising.
    # 【译】这个模型的加载器希望在调用方没有设置时设上的值。在过滤之后应用并受过滤约束，
    #   所以编码器不能接受的默认值会像其他kwarg一样被丢掉，而不是抛异常。
    # 解释：从__call__的代码看，defaults是在过滤"之前"用setdefault加进kwargs的，再与其他kwarg一起接受过滤——效果仍是"不能接受的默认值被丢掉"
    defaults: dict[str, Any] = field(default_factory=dict, compare=False)
    # prepare_kwargs：按模型改写kwargs（V4.1用）；preserve_content：True时消息content保持原样（可能是片段列表），不压成纯文本
    prepare_kwargs: KwargsPreparer = _identity_kwargs
    preserve_content: bool = False

    # 例：编码器签名为 encode_messages(messages, thinking_mode="thinking") → accepts={"messages", "thinking_mode"}；
    #   请求带来 tool_choice="none"、thinking_effort="high" → 这两个被丢弃（打debug日志），只传thinking_mode
    def __call__(self, messages: list[dict], **kwargs: Any) -> str:
        """Render, passing on only the kwargs this encoder can take.

        A Jinja template silently ignores a kwarg it does not read; a Python
        encoder raises `TypeError`. The request handler forwards template
        controls it cannot know the model reads -- `response_format`,
        `tool_choice`, `thinking_effort`, and whatever a client puts in
        `chat_template_kwargs` -- so a request carrying any of them was a 500
        on every model that ships an encoder instead of a template, which on
        this box is DeepSeek-V4 and Kimi-K3. `tool_choice: "none"` reached it
        that way the moment the Anthropic endpoint began honouring the field.

        Dropped rather than raised, so the two rendering paths behave the same
        way: unread means unread. Logged at debug, because a kwarg a model
        cannot take is worth seeing when a template control appears not to
        work.
        """
        # 【译】渲染，只把这个编码器能接受的kwargs传下去。
        #
        #   Jinja模板会悄悄忽略它不读取的kwarg；Python编码器则抛 TypeError。请求处理函数会转发它无法知道模型是否读取的模板控制项——
        #   response_format、tool_choice、thinking_effort，以及客户端放进 chat_template_kwargs 的任何东西——所以携带其中任何一项的请求，
        #   在每个附带编码器而非模板的模型上都是500，在这台机器上就是DeepSeek-V4和Kimi-K3。Anthropic接口开始遵守 tool_choice 字段的那一刻，
        #   tool_choice: "none" 就这样到达了它。
        #
        #   丢弃而不是抛异常，这样两条渲染路径表现一致：不读就是不读。用debug级别记录，因为当某个模板控制项看起来不起作用时，
        #   一个模型不能接受的kwarg值得被看到。
        for name, value in self.defaults.items():
            kwargs.setdefault(name, value)
        kwargs = self.prepare_kwargs(kwargs)
        if self.accepts is not None:
            unread = [name for name in kwargs if name not in self.accepts]
            for name in unread:
                logger.debug(
                    "chat encoder %s does not take %r; ignoring it", self.name, name
                )
                kwargs.pop(name)
        return self.encode(messages, **kwargs)


# 一个模块名对应的"准备方案"：消息准备、kwargs准备、是否支持工具、是否保留content原样
@dataclass(frozen=True)
class _EncoderPreparation:
    messages: MessagePreparer = _copy_messages
    kwargs: KwargsPreparer = _identity_kwargs
    supports_tools: bool = False
    preserve_content: bool = False


# 已注册的编码器模块 → 准备方案；不在表里的模块用默认方案（浅拷贝消息、不改kwargs、不支持工具）
#   encoding_dsv4：DeepSeek-V4；encoding_dsv41：DeepSeek-V4.1（钩子在deepseek_v41_encoder.py）
_PREPARERS = {
    "encoding_dsv4": _EncoderPreparation(
        messages=_prepare_deepseek_v4_messages, supports_tools=True
    ),
    "encoding_dsv41": _EncoderPreparation(
        messages=deepseek_v41_encoder.prepare_messages,
        kwargs=deepseek_v41_encoder.prepare_kwargs,
        supports_tools=True,
        preserve_content=True,
    ),
}

# A generic filename does not identify a protocol. Only the checkpoint's
# explicit model_type selects the V4.1 hooks.
# 【译】一个通用的文件名并不能标识协议。只有checkpoint明确的 model_type 才会选中V4.1的钩子。
_MODEL_ENCODERS = {"deepseek_v41": "encoding_dsv41"}


# 用inspect.signature读出编码器能接受的关键字参数名
def _accepted_kwargs(encoder: MessageEncoder) -> frozenset[str] | None:
    """The keyword names ``encoder`` takes, or ``None`` for "anything".

    ``None`` also when the signature cannot be read at all -- a C callable, a
    functools wrapper without `__wrapped__`. Passing everything through is
    what happened before this existed, so an unreadable signature is no worse
    than it was, and refusing to load the encoder over it would be.
    """
    # 【译】encoder 接受的关键字参数名；"什么都接受"时为 None。
    #
    #   签名根本读不出来时也返回 None——比如C实现的可调用对象、没有 __wrapped__ 的functools包装。在有这个函数之前就是全部透传，
    #   所以读不出签名并不比以前更糟，而为此拒绝加载编码器反倒更糟。
    try:
        params = inspect.signature(encoder).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return frozenset(
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


# 构造适配器：按模块名（通用的"encoding"则按model_type映射）选出准备方案，读出签名，组装MessageEncoderAdapter
def build_message_encoder_adapter(
    module_name: str,
    encoder: MessageEncoder,
    source_path: str = "",
    accepts_from: MessageEncoder | None = None,
    defaults: dict[str, Any] | None = None,
    model_type: str | None = None,
) -> MessageEncoderAdapter:
    """Build an adapter registered for ``module_name`` or an identity adapter.

    ``accepts_from`` is the callable whose signature says what may be passed,
    when that is not ``encoder`` itself. The loader wraps the model's function
    in a closure taking ``**kwargs``, and reading the *wrapper* would answer
    "anything" for every encoder there is.
    """
    # 【译】构造为 module_name 注册的适配器，没有注册的就构造一个恒等适配器。
    #
    #   accepts_from 是用其签名来决定可以传什么的那个可调用对象（当它不是 encoder 本身时）。
    #   加载器把模型的函数包在一个接受 **kwargs 的闭包里，读"包装层"的签名会对每个编码器都回答"什么都接受"。
    # 解释：当前的加载器（chat_encoders._load_encoder_from_dir）没有包闭包，也没有传accepts_from，签名直接从encode_messages本身读取
    if module_name == "encoding":
        module_name = _MODEL_ENCODERS.get(model_type, module_name)
    preparation = _PREPARERS.get(module_name, _EncoderPreparation())
    return MessageEncoderAdapter(
        name=module_name,
        encode=encoder,
        prepare_messages=preparation.messages,
        prepare_kwargs=preparation.kwargs,
        supports_tools=preparation.supports_tools,
        preserve_content=preparation.preserve_content,
        source_path=source_path,
        accepts=_accepted_kwargs(accepts_from or encoder),
        defaults=dict(defaults or {}),
    )
