"""Python entrypoint for running Atomesh with a Python-owned engine.

This module intentionally mirrors the engine/tokenizer initialization used by
``atom.entrypoints.openai.api_server``. The Rust side receives the already
constructed Python objects and uses them in ``AtomStandaloneRouter``.
"""

# 【译】以"引擎由Python持有"的方式运行Atomesh的Python入口。
#
#   本模块有意与 atom.entrypoints.openai.api_server 使用相同的引擎 / tokenizer 初始化方式。
#   Rust一侧接收已经构造好的Python对象，并在 AtomStandaloneRouter 中使用它们。
#
# 【导读】atomesh入口：环境变量 USE_ATOMESH_ENTRYPOINTS=1 时，openai_server.py 导入的不是 api_server.main，而是本文件的 main
# atomesh是什么：atom/mesh 下的Rust项目（Atomesh，见 atom/mesh/README.md），本身是面向PD分离部署的模型路由网关
#   （多worker负载均衡、重试、熔断、Prometheus指标）；编译产物 libmesh.so 同时是一个名为 atomesh_runner 的
#   PyO3扩展模块（atom/mesh/src/python.rs），提供 parse_from / launch_mesh 等函数给Python调用
#   PyO3：Rust与Python互相调用的绑定库——Rust代码编译成.so后Python可以直接import；Rust一侧也能先取得GIL
#     （atom_standalone.rs里的Python::attach），再调用Python对象的方法。standalone模式正是靠后者让Rust的HTTP线程调用Python服务对象
# 两种运行方式，由 main 按命令行里有没有 mesh-only 区分：
#   1. ATOM standalone（默认）：Python在本进程内建好 LLMEngine 和 tokenizer，包装成 AtomStandaloneService
#      （本目录 atom_standalone_service.py），交给Rust的HTTP服务；Rust负责HTTP、生命周期和可观测性，
#      收到请求后经PyO3调用该服务对象的方法
#   2. mesh-only：纯路由器，本进程不建引擎，把请求转发给外部的worker（可以是PD分离的prefill / decode节点）
#      （PD分离：prefill和decode放在不同实例上跑，prefill实例算完把KV cache传给decode实例；
#       入门见model_engine/arg_utils.py的【科普】PD分离）
#      【分支】PD分离的路由与KV交接，见第9批
# 与 api_server.py 的区别：api_server 是 FastAPI + uvicorn，HTTP层和OpenAI语义都在Python里；
#   standalone模式下HTTP层换成了Rust，Python只保留OpenAI语义（chat template、推理内容分离、工具调用解析）。
#   Rust的 AtomStandaloneRouter（atom/mesh/src/routers/atom_standalone.rs）只把 /v1/chat/completions 和
#   /v1/completions 转给Python服务，generate、responses 等返回"未实现"；Python服务也没有 /v1/messages 对应的方法

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom.entrypoints.openai.tool_parser.registry import (
    TOOL_CALL_PARSER_HELP,
    validate_tool_call_parser,
)
from atom.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
    tune_gc,
)

logger = logging.getLogger("atom")

engine: Any | None = None
tokenizer: Any | None = None


# 解析后的启动参数：engine_args交给EngineArgs建引擎，mesh_args交给Rust的 atomesh_runner.parse_from
@dataclass(frozen=True)
class StandaloneArgs:
    """Parsed standalone launch args split by their owning layer."""

    # 【译】解析后的standalone启动参数，按它们所属的层拆分。
    # 解释：两层指Python的引擎参数和Rust mesh的参数
    engine_args: argparse.Namespace
    mesh_args: list[str]
    default_chat_template_kwargs: dict[str, Any]


# 加载Rust扩展模块：先按普通模块名导入（已随ATOM安装时）；
#   找不到就去源码树里的 atom/mesh/target/debug、target/release 下找 libmesh.so 手动加载（debug优先）
def import_atomesh_runner() -> Any:
    # Provided by the Rust PyO3 module in atom/mesh/src/python.rs.
    # 【译】由 atom/mesh/src/python.rs 里的Rust PyO3模块提供。
    try:
        import atomesh_runner

        return atomesh_runner
    except ModuleNotFoundError as exc:
        if exc.name != "atomesh_runner":
            raise ModuleNotFoundError(f"Module named 'atomesh_runner' not found: {exc}")

    atom_source_root = Path(__file__).resolve().parents[3]
    mesh_root = atom_source_root / "atom" / "mesh"
    candidates = [
        mesh_root / "target" / "debug" / "libmesh.so",
        mesh_root / "target" / "release" / "libmesh.so",
    ]

    for library_path in candidates:
        if not library_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("atomesh_runner", library_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["atomesh_runner"] = module
        spec.loader.exec_module(module)
        return module

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise ModuleNotFoundError(
        "No module named 'atomesh_runner' and no loadable libmesh.so was found. "
        f"Searched:\n{searched}"
    )


# --version / --version-verbose：打印Rust侧的版本串；加载失败时只打印一行固定文字
def print_version(verbose: bool = False) -> None:
    try:
        atomesh_runner = import_atomesh_runner()
        version_fn = (
            atomesh_runner.version_verbose_string
            if verbose
            else atomesh_runner.version_string
        )
        print(version_fn())
    except Exception:  # noqa: BLE001 - a version banner must not fail the CLI
        print("Atomesh Python interface")


# 建tokenizer和LLMEngine（与api_server相同：EngineArgs.from_cli_args -> create_engine），结果存进模块级全局变量
def initialize_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import _load_tokenizer

    global engine, tokenizer

    logger.info("Loading tokenizer from %s...", args.model)
    tokenizer = _load_tokenizer(args.model, args.trust_remote_code)

    logger.info("Initializing engine with model %s...", args.model)
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(tokenizer=tokenizer)
    return engine, tokenizer


# 用已经建好的engine / tokenizer构造 AtomStandaloneService，这个对象随后交给Rust的 launch_mesh
def initialize_standalone_service(
    args: argparse.Namespace,
    default_chat_template_kwargs: dict[str, Any],
) -> Any:
    from atom.entrypoints.atomesh.atom_standalone_service import AtomStandaloneService

    return AtomStandaloneService(
        engine=engine,
        tokenizer=tokenizer,
        model_name=args.model,
        default_chat_template_kwargs=default_chat_template_kwargs,
        tool_call_parser=getattr(args, "tool_call_parser", None),
    )


# 先把 --host / --port / --server-port 从命令行里摘出来交给mesh，其余参数留给Python的解析器
def split_standalone_mesh_args(raw_args: list[str]) -> tuple[list[str], list[str]]:
    """Keep mesh-owned network args from being consumed by Python parsers.

    EngineArgs also defines --port for internal engine communication. In
    Atomesh standalone mode, the user-facing --port should configure the mesh
    HTTP router, matching the Rust CLI behavior. --server-port is accepted for
    compatibility with the classic OpenAI entrypoint and translated to --port.
    """
    # 【译】不让mesh持有的网络参数被Python的解析器吃掉。
    #
    #   EngineArgs也定义了 --port，用于引擎内部通信。在Atomesh standalone模式下，面向用户的 --port 应该配置
    #   mesh的HTTP路由器，与Rust CLI的行为一致。为了与经典的OpenAI入口兼容，也接受 --server-port，并把它转换成 --port。
    mesh_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mesh_parser.add_argument("--host")
    mesh_parser.add_argument("--port")
    mesh_parser.add_argument("--server-port")
    mesh_namespace, python_args = mesh_parser.parse_known_args(raw_args)

    mesh_args: list[str] = []
    if mesh_namespace.host is not None:
        mesh_args.extend(["--host", mesh_namespace.host])
    port = mesh_namespace.port or mesh_namespace.server_port
    if port is not None:
        mesh_args.extend(["--port", port])
    return python_args, mesh_args


# 判断 --tool-call-parser 的值是不是ATOM自己的工具调用解析器名字（tool_parser/registry.py 的 validate_tool_call_parser）
def _is_atom_tool_call_parser(value: str | None) -> bool:
    """Whether this name is one ATOM's own resolver understands.

    A question, not an assertion: the mesh router answers to a different set
    of names for the same flag, and neither side owns the other's.
    """
    # 【译】这个名字是否是ATOM自己的解析器能理解的名字。
    #
    #   这是一个询问而不是断言：对同一个参数，mesh路由器认的是另一套名字，双方都不拥有对方的那一套。
    if value is None:
        return False
    try:
        validate_tool_call_parser(value)
    except ValueError:
        return False
    return True


# 把 --tool-call-parser 原样再转发给Rust路由器
def _forward_tool_call_parser(value: str | None) -> list[str]:
    """Give the mesh router back the flag the Python parser just consumed.

    `--tool-call-parser` has two consumers with two vocabularies: the Rust
    router declares its own (`cliargs.rs`, sglang spelling) and reads it in
    `app_context`, and the Python service resolves it to a `ToolCallParser`.
    Until this entrypoint registered the flag it reached the router as an
    unrecognised arg and was passed straight through; registering it made
    `parse_known_args` swallow it, so the router silently lost a setting that
    had been reaching it. Both need it, so both get it.
    """
    # 【译】把Python解析器刚刚吃掉的参数还给mesh路由器。
    #
    #   --tool-call-parser 有两个使用者，各有一套词汇：Rust路由器声明了它自己的这个参数（cliargs.rs，sglang的写法），
    #   并在 app_context 中读取；Python服务则把它解析成一个 ToolCallParser。在本入口注册这个参数之前，它作为
    #   无法识别的参数直接透传给了路由器；注册之后 parse_known_args 会把它吞掉，于是路由器悄无声息地丢掉了一个
    #   原本能收到的设置。双方都需要它，所以双方都给。
    return [] if value is None else ["--tool-call-parser", value]


# argparse的type函数：把 --default-chat-template-kwargs 的值解析成JSON对象，格式不对时报参数错误
def json_object_arg(raw_value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--default-chat-template-kwargs must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "--default-chat-template-kwargs must decode to a JSON object"
        )
    return parsed


# 解析standalone模式的命令行：EngineArgs的全部参数 + --tool-call-parser + --default-chat-template-kwargs；
#   Python不认识的参数（parse_known_args剩下的）连同网络参数一起作为mesh_args交给Rust
def parse_standalone_args(raw_args: list[str]) -> StandaloneArgs:
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Atomesh Python interface",
        allow_abbrev=False,
    )
    EngineArgs.add_cli_args(parser)
    # 【译·help】工具调用的线上格式。'auto'（默认）在启动时从模型的chat template读出；显式给出名字会覆盖它。
    #   两者都解析不出时，工具调用作为普通文本返回，启动日志会说明——格式从不根据输出猜测。
    #   （help文本是 tool_parser/registry.py 的 TOOL_CALL_PARSER_HELP）
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        # Not "auto": that is what *unset* resolves to anyway, and telling the
        # two apart is what decides whether the mesh router is told as well.
        # 【译】默认值不用"auto"：未设置时本来就会解析成auto，而区分"未设置"与"显式给出"，决定了是否也要告诉mesh路由器。
        default=None,
        help=TOOL_CALL_PARSER_HELP,
    )
    # 【译·help】渲染chat template时的默认kwargs（JSON字符串）。与每个请求自带的chat_template_kwargs合并（请求的优先）。
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=json_object_arg,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
        ),
    )

    python_raw_args, mesh_network_args = split_standalone_mesh_args(raw_args)
    engine_args, mesh_args = parser.parse_known_args(python_raw_args)
    forwarded = engine_args.tool_call_parser

    # Not validated against ATOM's vocabulary here, and that is the point.
    # This flag has two consumers with disjoint vocabularies: the Rust router
    # declares its own (`json`, `python`, `xml`, `hermes`) and ATOM's
    # resolver takes `dsml`, `glm`, `kimi`, `kimi_k3`, `minimax`, `qwen`.
    # Raising on a name ATOM does not know would kill any existing standalone
    # deployment launched with a router name -- before this entrypoint
    # registered the flag at all, those passed straight through.
    #
    # So: a name ATOM knows binds ATOM's parser, anything else is the
    # router's and ATOM falls back to reading the chat template. Either way
    # the value reaches the router, and the choice is logged.
    # 【译】这里故意不按ATOM的词汇表做校验。这个参数有两个使用者，词汇表互不相交：Rust路由器声明了自己的一套
    #   （json、python、xml、hermes），ATOM的解析器接受 dsml、glm、kimi、kimi_k3、minimax、qwen。
    #   遇到ATOM不认识的名字就报错，会让所有用路由器的名字启动的现有standalone部署直接挂掉——
    #   在本入口注册这个参数之前，这些名字都是直接透传过去的。
    #
    #   所以：ATOM认识的名字绑定ATOM的解析器，其他名字归路由器，ATOM退回到从chat template读取格式。
    #   无论哪种情况，这个值都会传给路由器，而且这个选择会记进日志。
    # 解释：与本仓库代码不完全一致——atom/mesh/src/tool_parser/registry.rs 注册的名字是 passthrough、json、qwen、qwen_json、
    #   qwen_xml、qwen_coder、dsml、dsml_v41、glm、glm45_moe、glm47_moe、kimi、kimi_k3、kimik2、minimax，与ATOM的名字大量重合，
    #   并非互不相交；ATOM这边还有 dsml_v41。代码的行为不受影响：认识就用ATOM的解析器，不认识就置None，值总是转发给Rust
    if engine_args.tool_call_parser is not None and not _is_atom_tool_call_parser(
        engine_args.tool_call_parser
    ):
        logger.info(
            "--tool-call-parser=%r is not one of ATOM's formats, so it is "
            "forwarded to the mesh router and ATOM reads its own format from "
            "the chat template.",
            engine_args.tool_call_parser,
        )
        engine_args.tool_call_parser = None

    return StandaloneArgs(
        engine_args=engine_args,
        mesh_args=(
            mesh_args + mesh_network_args + _forward_tool_call_parser(forwarded)
        ),
        default_chat_template_kwargs=engine_args.default_chat_template_kwargs or {},
    )


# standalone启动流程：解析参数 -> Rust解析mesh参数 -> 建引擎和tokenizer -> 建服务对象 -> 调整GC ->
#   atomesh_runner.launch_mesh（阻塞，直到服务退出），Rust的HTTP服务把请求转给 standalone_service
def launch_atom_standalone(atomesh_runner: Any, raw_args: list[str]) -> None:
    standalone_args = parse_standalone_args(raw_args)
    parsed_args = atomesh_runner.parse_from(standalone_args.mesh_args)
    cli_args = parsed_args["cli_args"]
    initialize_engine(standalone_args.engine_args)
    standalone_service = initialize_standalone_service(
        standalone_args.engine_args,
        standalone_args.default_chat_template_kwargs,
    )

    # This frontend is the api_server's counterpart -- same tokenizer, same
    # per-request accumulators -- but it builds its engine itself and never
    # runs that FastAPI lifespan, so it has to apply the GC policy here.
    # 【译】这个前端是api_server的对应物——同样的tokenizer，同样的逐请求累加器——但它自己建引擎，
    #   从不运行api_server那个FastAPI的lifespan，所以必须在这里应用GC策略。
    tune_gc()
    maybe_attach_gc_debug_callback("atomesh")
    freeze_gc_heap("atomesh")

    print("\033[32mATOM starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    atomesh_runner.launch_mesh(
        server_config=parsed_args["server_config"],
        standalone_service=standalone_service,
    )


# mesh-only启动流程：只启动Rust路由器（standalone_service=None），打印模式（PD分离 / 常规）、负载均衡策略和worker地址
def launch_atomesh(atomesh_runner: Any, raw_args: list[str]) -> None:
    parsed_args = atomesh_runner.parse_from(
        [arg for arg in raw_args if arg != "mesh-only"]
    )
    cli_args = parsed_args["cli_args"]
    prefill_urls = parsed_args["prefill_urls"]
    decode_urls = parsed_args["decode_urls"]

    print("\033[32mAtomesh starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    mode = (
        "PD Disaggregated"
        if cli_args["pd_disaggregation"]
        else f"Regular ({cli_args['backend']})"
    )
    print(f"Mode: {mode}")
    print(f"Policy: {cli_args['policy']}")

    if cli_args["pd_disaggregation"] and prefill_urls:
        print(f"Prefill nodes: {prefill_urls}")
    if cli_args["pd_disaggregation"] and decode_urls:
        print(f"Decode nodes: {decode_urls}")

    atomesh_runner.launch_mesh(
        server_config=parsed_args["server_config"],
        standalone_service=None,
    )


# 入口：先处理 --version；命令行里没有 mesh-only 就走ATOM standalone，有就走纯路由
def main() -> None:
    raw_args = sys.argv[1:]
    for arg in raw_args:
        if arg in ("--version", "-V"):
            print_version(verbose=False)
            return
        if arg == "--version-verbose":
            print_version(verbose=True)
            return
    # `python xxx mesh-only ...` starts mesh routing;
    # other invocations default to ATOM standalone.
    # 【译】`python xxx mesh-only ...` 启动mesh路由；其他调用方式默认走ATOM standalone。
    use_atom_standalone = "mesh-only" not in raw_args
    # Import the mesh_python module.
    # 【译】导入mesh_python模块。
    # 解释：实际导入的模块名是 atomesh_runner，见 import_atomesh_runner
    atomesh_runner = import_atomesh_runner()

    if use_atom_standalone:
        launch_atom_standalone(atomesh_runner, raw_args)
    else:
        launch_atomesh(atomesh_runner, raw_args)


if __name__ == "__main__":
    main()
