# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat-template dispatch for the OpenAI chat endpoint.

Some models (e.g. DeepSeek V4) ship a Python encoder under ``<model>/encoding/``
instead of a Jinja ``chat_template``. This module discovers such encoders at
server startup and provides :func:`apply_chat_template`, a single entry point
that the request handler calls — it transparently routes to the custom encoder
when one was found, or to ``tokenizer.apply_chat_template`` otherwise.
"""

# 【译】OpenAI chat接口的chat template分派。
#
#   有些模型（例如DeepSeek V4）在 <model>/encoding/ 下附带一个Python编码器，而不是Jinja的 chat_template。
#   这个模块在服务器启动时发现这类编码器，并提供 apply_chat_template 这一个入口供请求处理函数调用——
#   找到了自定义编码器就透明地交给它，否则交给 tokenizer.apply_chat_template。
#
# 本文件（第2批2c，B级）：把chat消息渲染成prompt字符串
# chat template（第一次出现，讲一次）：模型仓库tokenizer配置里的一段Jinja模板，把消息列表变成带特殊token的prompt；
#   add_generation_prompt=True会在末尾补上assistant回合的开头，让模型接着以assistant身份续写。例（Qwen风格）：
#   [{"role": "user", "content": "你好"}] → "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
# 本文件的函数：
#   启动时：load_custom_message_encoder（找Python编码器）；chat_template_source / render_probe_prompt /
#     resolve_reasoning_toggle（向模板"提问"：有哪些标记、怎样关掉推理）——结果由api_server.main存进全局变量
#   每个请求：apply_chat_template（api_server的chat / messages / responses接口、examples/simple_inference.py、atomesh都调用它）
# 对比nano-vllm：nano-vllm的example.py直接调用tokenizer.apply_chat_template，没有自定义编码器和启动期探测
import glob
import importlib.util
import json
import logging
import os
import pathlib
from typing import Any

from jinja2 import TemplateError

from atom.model_loader.weight_utils import local_model_dir

from .chat_encoder_adapters import (
    MessageEncoderAdapter,
    build_message_encoder_adapter,
)
from .protocol import ChatMessage

logger = logging.getLogger("atom")


# 在<model>/encoding/下找唯一一个encoding_*.py（或encoding.py），导入它并取出encode_messages，包成MessageEncoderAdapter
def _load_encoder_from_dir(model_path: str) -> MessageEncoderAdapter | None:
    """Look for ``<model>/encoding/encoding_*.py`` and load ``encode_messages``.

    Returns ``None`` when the directory or matching file is absent (model uses
    the standard Jinja path). Returns ``None`` and warns on ambiguity (multiple
    matches) or load failures.

    """
    # 【译】查找 <model>/encoding/encoding_*.py 并加载其中的 encode_messages。
    #
    #   目录或匹配的文件不存在时返回 None（模型走标准的Jinja路径）。有歧义（多个匹配）或加载失败时，返回 None 并给出警告。
    # 解释：代码也接受 encoding/encoding.py；对这个通用文件名，会读config.json的model_type来选择适配器（见下文）
    enc_dir = os.path.join(model_path, "encoding")
    if not os.path.isdir(enc_dir):
        return None

    candidates = sorted(glob.glob(os.path.join(enc_dir, "encoding_*.py")))
    standalone = os.path.join(enc_dir, "encoding.py")
    if os.path.isfile(standalone):
        candidates.append(standalone)
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            f"Multiple message encoders found in {enc_dir}, refusing to guess: "
            f"{[os.path.basename(p) for p in candidates]}"
        )
        return None

    enc_path = candidates[0]
    module_name = os.path.splitext(os.path.basename(enc_path))[0]
    try:
        # importlib按文件路径动态导入模块：spec_from_file_location → module_from_spec → exec_module（执行模块代码）
        spec = importlib.util.spec_from_file_location(module_name, enc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = mod.encode_messages
    except Exception:
        # Broad on purpose: this executes a file the operator supplied, so the
        # failure can be anything that module raises, and a bad encoder must
        # not stop the server. `exc_info` rather than `{e}` -- the message
        # alone was never enough to debug someone else's encoder, and the
        # traceback names the line inside it.
        # 【译】故意捕获得很宽：这里执行的是运维人员提供的文件，失败可能是那个模块抛出的任何异常，而一个坏的编码器不能让服务器停下。
        #   用 exc_info 而不是 {e}——光有消息从来不足以调试别人写的编码器，traceback会指出其中出错的那一行。
        logger.warning(f"Failed to load encoder from {enc_path}", exc_info=True)
        return None

    # 文件名是通用的encoding.py时，读模型的config.json取model_type，交给适配器决定用哪套钩子
    model_type = None
    if module_name == "encoding":
        try:
            with open(os.path.join(model_path, "config.json")) as config_file:
                model_type = json.load(config_file).get("model_type")
        except (OSError, ValueError, AttributeError):
            logger.debug("No model_type for encoder %s", enc_path, exc_info=True)

    logger.info(f"Loaded message encoder from {enc_path}")
    # Handed to the adapter rather than applied in a wrapper here. A wrapper
    # runs *after* the adapter has filtered kwargs against the encoder's
    # signature, so the one kwarg it adds is the one the filter cannot remove:
    # an encoder that does not take `thinking_mode` raised `TypeError` on
    # every real request while the startup probe -- which now counts TypeError
    # as a refusal -- reported only "tool calls will be delivered as plain
    # text". Silent at startup, 500 on every chat.
    # 【译】交给适配器处理，而不是在这里套一层包装来设置。包装层在适配器按编码器签名过滤kwargs"之后"才运行，所以它添加的那个kwarg
    #   恰好是过滤器无法移除的：一个不接受 thinking_mode 的编码器在每个真实请求上都抛 TypeError，而启动探测（现在把TypeError算作拒绝）
    #   只报告了"工具调用将作为普通文本交付"。启动时悄无声息，每个chat请求都是500。
    # defaults={"thinking_mode": "thinking"}：请求没指定时默认开启思考模式（编码器不接受这个参数时会被过滤掉）
    return build_message_encoder_adapter(
        module_name,
        raw,
        enc_path,
        defaults={"thinking_mode": "thinking"},
        model_type=model_type,
    )


# 启动时调用一次（api_server.main等）；local_model_dir（model_loader/weight_utils.py）把模型名解析成本地目录
def load_custom_message_encoder(model_path: str) -> MessageEncoderAdapter | None:
    """Probe ``model_path`` once at startup for a custom message encoder.

    Returns the encoder, or ``None`` when the model uses the standard Jinja
    ``chat_template`` path. Result should be cached by the caller — this does
    filesystem IO and a Python import.
    """
    # 【译】启动时探查一次 model_path 有没有自定义的消息编码器。
    #
    #   返回编码器；模型走标准的Jinja chat_template 路径时返回 None。调用方应缓存结果——这里有文件系统IO和一次Python导入。
    return _load_encoder_from_dir(local_model_dir(model_path))


# The smallest request that makes a template show its framing. Nothing is
# ever sent to the model; only what the template wraps a turn in matters.
# 【译】能让模板展示其框架的最小请求。它从不会发给模型；只关心模板把一个回合包装成什么样。
PROBE_MESSAGES: list[dict] = [{"role": "user", "content": "hi"}]
# 探测工具调用格式用的假工具（get_weather）：带上它渲染一次，看模板怎样向模型描述工具
PROBE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


# 取chat template的源码文本，供启动期按字符串搜索（推理方言、是否隐式打开思考段、工具调用格式）
def chat_template_source(
    tokenizer: Any, custom_encoder: MessageEncoderAdapter | None = None
) -> str:
    """The model's chat template as text to search, or ``""`` if it ships none.

    Deliberately not a rendered prompt, and the two are not interchangeable:
    a marker that appears only in the template's own logic is absent from a
    fresh prompt. Measured on this box -- Qwen3.5's source carries `<think>`
    and `</think>`, its rendered prompt carries only `<think>`, and
    Qwen3-8B's rendered prompt carries neither. A question about what the
    template *does with a reply* has to read the source; a question about what
    the prompt *tells the model* has to read the render
    (:func:`render_probe_prompt`).

    Two shapes bite anyone reaching for the raw attribute, which is why this
    exists rather than `getattr(tokenizer, "chat_template", "")`:
    multi-template tokenizers hold a ``dict``, and `"</think>" in <dict>`
    silently tests the *keys*; and it is ``None`` for every model that ships a
    Python encoder instead of Jinja (Kimi-K3, DeepSeek-V4), whose literals
    live in that module's source instead.
    """
    # 【译】把模型的chat template作为可搜索的文本返回；模型没有附带模板时返回 ""。
    #
    #   故意不是渲染后的prompt，两者不能互换：只出现在模板自身逻辑里的标记，不会出现在一个新渲染的prompt里。
    #   在这台机器上实测——Qwen3.5的源码里有 <think> 和 </think>，它渲染出的prompt里只有 <think>，而Qwen3-8B渲染出的prompt里两者都没有。
    #   关于模板"如何处理回复"的问题要读源码；关于prompt"告诉模型什么"的问题要读渲染结果（render_probe_prompt）。
    #
    #   直接取原始属性会被两种形态坑到，这就是为什么有这个函数，而不是 getattr(tokenizer, "chat_template", "")：
    #   多模板的tokenizer持有的是一个 dict，而 "</think>" in <dict> 会悄悄地检查"键"；对于附带Python编码器而不是Jinja的模型
    #   （Kimi-K3、DeepSeek-V4），它是 None，这些模型的字面量写在那个模块的源码里。
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template
    if isinstance(template, dict):
        # Every named variant, so a marker in any of them counts. Searching
        # the dict itself would have searched the names.
        # 【译】包含每个具名变体，这样任何一个里面出现的标记都算数。直接搜索dict本身只会搜到名字。
        return "\n".join(str(v) for v in template.values())
    if custom_encoder is not None and custom_encoder.source_path:
        try:
            return pathlib.Path(custom_encoder.source_path).read_text()
        except OSError as e:
            logger.warning("Could not read %s: %s", custom_encoder.source_path, e)
    return ""


# (kwarg, off-value, on-value) triples, tried in order. Every family on this
# box is covered: Qwen reads `enable_thinking`, Kimi-K3 `thinking`, and
# `thinking_mode` is read by two families with disjoint vocabularies --
# MiniMax-M3 wants "disabled"/"enabled" and DeepSeek-V4's encoder asserts on
# anything outside {"chat", "thinking"}. Hence pairs of the same name, and
# hence the order: "disabled" first, so MiniMax matches before V4's rejection
# of it sends the probe on to "chat".
#
# The on-value is the off-value's counterpart and is not probed for. It cannot
# be: most templates default to reasoning on, so passing their on-value renders
# identically to passing nothing, and "the render changed" -- the evidence the
# off-probe runs on -- is absent by construction. Only MiniMax, whose default
# is off, differs when switched on.
# 【译】(kwarg, 关闭值, 开启值) 三元组，按顺序尝试。这台机器上的每个家族都覆盖了：Qwen读 enable_thinking，Kimi-K3读 thinking，
#   而 thinking_mode 被两个词汇互不相交的家族读取——MiniMax-M3要 "disabled"/"enabled"，DeepSeek-V4的编码器则对
#   {"chat", "thinking"} 之外的任何值断言失败。因此有同名的两组，也因此是这个顺序："disabled" 在前，
#   这样MiniMax会先匹配上，而V4拒绝它之后探测会继续尝试 "chat"。
#
#   开启值是关闭值的对应项，不做探测。也没法探测：大多数模板默认开启推理，所以传它们的开启值和什么都不传渲染结果完全一样，
#   而关闭探测所依据的"渲染结果变了"这个证据，从构造上就不存在。只有默认关闭的MiniMax在开启时会有差别。
REASONING_TOGGLES: tuple[tuple[str, Any, Any], ...] = (
    ("enable_thinking", False, True),
    ("thinking", False, True),
    ("thinking_mode", "disabled", "enabled"),
    ("thinking_mode", "chat", "thinking"),
)

# A template refusing a probe value, by name. A model-shipped Python encoder
# validates with a bare `assert`; Jinja raises its own; a signature that does
# not take the kwarg raises TypeError. Refusal means "not this pair, try the
# next" -- anything else is a bug and is left to propagate.
# 【译】按名字列出"模板拒绝一个探测值"时会抛的异常。模型附带的Python编码器用裸 assert 做校验；Jinja抛它自己的异常；
#   签名不接受该kwarg时抛 TypeError。拒绝的意思是"不是这一组，试下一组"——其他任何异常都是bug，任其向上传播。
# 解释：元组里还有ValueError，原因见下面render_probe_prompt的docstring（没有chat template时transformers抛ValueError）
_PROBE_REFUSALS = (TemplateError, TypeError, ValueError, AssertionError)


# 启动时探测"哪个kwarg能关掉推理"：先渲染一次不带参数的基线，再逐个尝试REASONING_TOGGLES，渲染结果变了就是它
# 例：Qwen3的模板在enable_thinking=False时会在assistant开头后补一段空的<think></think>，与基线不同
#   → 返回("enable_thinking", False, True)，api_server把它存进全局reasoning_toggle
def resolve_reasoning_toggle(
    tokenizer: Any, custom_encoder: MessageEncoderAdapter | None = None
) -> tuple[str, Any, Any] | None:
    """The kwarg that turns this model's reasoning off, and the value, or None.

    Rendered twice and compared, rather than matched against a table of model
    families: a Jinja template silently ignores a kwarg it does not read, so
    "the prompt changed" *is* the evidence that it read this one. That silence
    is also why the question has to be asked at all -- the chat path passed a
    hardcoded ``thinking=`` to every model, which is correct for Kimi-K3 and a
    no-op for the entire Qwen family, whose templates read `enable_thinking`.
    A no-op here is invisible: the model reasons anyway and the client that
    asked for no reasoning simply pays for it.

    ``None`` means the template offers no switch, and for a model that begins
    inside the reasoning channel that means reasoning cannot be turned off at
    all -- worth saying out loud at startup, because the request can then only
    be honoured as far as separating the reasoning and reporting it, never by
    discarding it. On this box only gpt-oss and DeepSeek-R1 answer ``None``,
    and neither opens a channel this way to begin with.

    Verified against every model on this box: Qwen3/Qwen3.5 `enable_thinking`,
    Kimi-K3 `thinking`, MiniMax-M3 `thinking_mode="disabled"`, DeepSeek-V4
    `thinking_mode="chat"` -- and for all six that start inside the channel,
    applying the pair takes the rendered prompt back out of it.
    """
    # 【译】返回关闭这个模型推理的kwarg及其取值，或None。
    #
    #   渲染两次并比较，而不是去查一张模型家族表：Jinja模板会悄悄忽略它不读取的kwarg，所以"prompt变了"本身就是它读取了这个kwarg的证据。
    #   这种沉默也是必须问这个问题的原因——chat路径以前给每个模型都传硬编码的 thinking=，这对Kimi-K3是对的，
    #   对整个Qwen家族却是空操作，因为它们的模板读的是 enable_thinking。这里的空操作是看不见的：模型照样推理，
    #   要求不推理的客户端只是白白为它付费。
    #
    #   None 表示模板没有提供开关；对一个从推理通道内部开始的模型，这意味着推理根本关不掉——值得在启动时明确说出来，
    #   因为此时对请求的满足只能做到把推理分离出来并报告，永远不能丢弃它。在这台机器上只有gpt-oss和DeepSeek-R1返回 None，
    #   而它们本来就都不以这种方式打开通道。
    #
    #   已在这台机器上的每个模型上验证：Qwen3/Qwen3.5 是 enable_thinking，Kimi-K3 是 thinking，MiniMax-M3 是 thinking_mode="disabled"，
    #   DeepSeek-V4 是 thinking_mode="chat"——并且对全部六个从通道内部开始的模型，应用这一组参数都会让渲染出的prompt回到通道之外。
    baseline = render_probe_prompt(tokenizer, custom_encoder, tools=False)
    if baseline is None:
        return None
    for name, off_value, on_value in REASONING_TOGGLES:
        try:
            rendered = apply_chat_template(
                tokenizer, custom_encoder, PROBE_MESSAGES, **{name: off_value}
            )
        except _PROBE_REFUSALS:
            continue
        if rendered != baseline:
            return name, off_value, on_value
    return None


# 渲染一个探测回合（可选带上假工具），返回prompt字符串；模板拒绝时返回None。调用方：resolve_reasoning_toggle、api_server.main
def render_probe_prompt(
    tokenizer: Any,
    custom_encoder: MessageEncoderAdapter | None,
    *,
    tools: bool,
) -> str | None:
    """Render a throwaway turn, to ask the template about itself at startup.

    What a template renders *into the prompt* is the model's own instructions
    -- notably how to call a tool -- so a question about those is asked here
    and not of the template source. The reverse question reads the source
    instead; see :func:`chat_template_source` for why one is not a cheaper
    version of the other.

    ``None`` means the template refused the probe, which is a real answer: a
    template may reject the synthetic tools payload, and a model that does not
    do tool calls should still start. Refusal is `_PROBE_REFUSALS`, the same
    list the reasoning probe uses -- these two ask the same template the same
    kind of question and disagreeing about what counts as a refusal is how a
    model with no chat template at all stopped booting. `transformers` raises
    `ValueError` for that, this caught only `TemplateError` and `TypeError`,
    and both probes run *before* the engine is created, so a base checkpoint
    that used to serve `/v1/completions` perfectly well now died at startup.

    Nothing outside that list is caught -- an unexpected failure here is a
    bug, and a bug that silently turns into "this model has no tool-call
    format" is the class of silence this path exists to end.
    """
    # 【译】渲染一个一次性的回合，用来在启动时向模板询问它自己的情况。
    #
    #   模板渲染"进prompt"的内容是模型自己的指令——尤其是如何调用工具——所以关于这些的问题在这里问，而不是问模板源码。
    #   反过来的问题则读源码；为什么一个不是另一个的廉价版本，见 chat_template_source。
    #
    #   None 表示模板拒绝了探测，这是一个真实的答案：模板可能拒绝这个合成的工具载荷，而一个不做工具调用的模型仍然应该能启动。
    #   "拒绝"指 _PROBE_REFUSALS，与推理探测用的是同一个列表——这两个探测向同一个模板问同一类问题，而对什么算拒绝看法不一，
    #   正是一个完全没有chat template的模型启动不了的原因。transformers 对这种情况抛 ValueError，而这里以前只捕获 TemplateError
    #   和 TypeError，两个探测又都在引擎创建"之前"运行，于是一个原本能完美地服务 /v1/completions 的基座checkpoint在启动时就挂了。
    #
    #   列表之外的异常一概不捕获——这里的意外失败是bug，而一个悄悄变成"这个模型没有工具调用格式"的bug，正是这条路径要终结的那类沉默。
    try:
        return apply_chat_template(
            tokenizer,
            custom_encoder,
            PROBE_MESSAGES,
            tools=PROBE_TOOLS if tools else None,
        )
    except _PROBE_REFUSALS as e:
        logger.warning("The model's chat template refused the probe: %s", e)
        return None


# 渲染入口（每个chat请求都调用）：messages可以是ChatMessage列表或dict列表；tools是OpenAI格式的工具列表；
#   kwargs是合并后的chat template参数（enable_thinking、tool_choice等）。返回prompt字符串（不tokenize）
def apply_chat_template(
    tokenizer: Any,
    custom_encoder: MessageEncoderAdapter | None,
    messages: list[dict] | list[ChatMessage],
    *,
    tools: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    """Render ``messages`` to a prompt string.

    Dispatches to ``custom_encoder`` if one was discovered for this model,
    otherwise to ``tokenizer.apply_chat_template``. Jinja-only kwargs
    (``tokenize``, ``add_generation_prompt``) are stripped on the custom path.
    Model-scoped adapters prepare tools for custom encoders that support them;
    the generic path does not apply DeepSeek-V4-specific message rewriting.
    """
    # 【译】把 messages 渲染成prompt字符串。
    #
    #   如果为这个模型发现了 custom_encoder 就交给它，否则交给 tokenizer.apply_chat_template。在自定义路径上会去掉只属于Jinja的kwargs
    #   （tokenize、add_generation_prompt）。按模型定制的适配器会为支持工具的自定义编码器准备工具；通用路径不做DeepSeek-V4特有的消息改写。
    # ChatMessage先转成dict（protocol.ChatMessage.to_template_dict）
    if messages and isinstance(messages[0], ChatMessage):
        preserve_content = (
            custom_encoder is not None and custom_encoder.preserve_content
        )
        messages = [
            message.to_template_dict(preserve_content=preserve_content)
            for message in messages
        ]
    # 自定义编码器路径：按模型准备消息（例如DeepSeek-V4把工具放进一条system消息），再由适配器过滤kwargs后调用
    if custom_encoder is not None:
        for k in ("tokenize", "add_generation_prompt"):
            kwargs.pop(k, None)
        if tools and not custom_encoder.supports_tools:
            logger.warning(
                "tools= is not supported by custom message encoder %s; ignoring.",
                custom_encoder.name,
            )
        messages = custom_encoder.prepare_messages(messages, tools)
        return custom_encoder(messages, **kwargs)

    # Jinja路径：tokenize=False让它返回字符串；add_generation_prompt=True在末尾补上assistant回合的开头；工具交给模板渲染
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = True
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)
