# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.

# 【导读】graph_marker_instrumentation.py：改写 Inductor 生成的 wrapper 源码，把成对的 graph_marker 之间包进
#   record_function，让 profiler trace 里出现带名字的区段（第 5 批 5e，C 级）
#   前置：graph_marker 是什么、--mark-trace 的整条链路，见 graph_marker.py 文件头
#   调用方（grep 核实；都只在 is_graph_marker_enabled() 时调用，异常一律吞掉，不影响编译）：
#     compiler_inferface.py InductorAdaptor.compile → instrument_record_functions_in_file（PyTorch < 2.8，单个 wrapper 文件）
#     compiler_inferface.py InductorStandaloneAdaptor.compile → instrument_record_functions_in_dir（standalone_compile 的产物目录）
#     随后 backends.py 强制从缓存重新加载，让当前进程用上改写后的代码
#   与主干的关系：纯文本处理（正则 + 括号匹配），只在 profile 时用；不开 --mark-trace 时完全不运行
#
#   例（生成的 wrapper 片段，prefix = "mhc"；记号行写成简写 marker(...)，实际是 torch.ops.aiter 下的 graph_marker 调用）：
#     改写前                              改写后
#       marker(buf0, 'mhc_start')           marker(buf0, 'mhc_start')
#       buf1 = op_a(buf0)                   with record_function("mhc"):
#       buf2 = op_b(buf1)                       buf1 = op_a(buf0)
#       marker(buf2, 'mhc_end')                 buf2 = op_b(buf1)
#                                           marker(buf2, 'mhc_end')
#     并在 import torch 之后补一行 from torch.profiler import record_function
#   strip_markers=True 时再把记号调用换成 pass（旧式的 x = marker(y, …) 换成 x = y）；两个调用方都传 strip_markers=False，
#     所以实际运行时记号调用仍留在代码里（算子本身是空函数）
#
# 观察：_wrap_region_with_record_function 接收 layer_id（由 _layer_id_from_wrapper_path 从路径里算出），但函数体没有用它，
#   tag 就是 prefix；它的注释说的 "mlp" -> "layer_0_mlp" 升级在现有代码里不会发生。_already_wrapped 没有调用方

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

# 从产物路径里取子图编号：.../artifact_shape_<shape>_subgraph_<N>/...
_SUBGRAPH_ID_RE = re.compile(r"artifact_shape_[^/]+_subgraph_(\d+)")

# 匹配 "<缩进>变量 = 右边" 形式的一行，用于识别旧式的 x = marker(y, '...')

_ASSIGNMENT_RE = re.compile(
    r"""^(?P<indent>\s*)
        (?P<lhs>[A-Za-z_]\w*)\s*=\s*
        (?P<rhs>.+?)\s*$""",
    re.VERBOSE,
)


# 文件里的一个记号：行号、缩进、名字（如 "mhc_start"）
@dataclass(frozen=True)
class _Marker:
    idx: int
    indent: str
    name: str


# 解析出的旧式赋值 lhs = marker(arg, name)
@dataclass(frozen=True)
class _ParsedMarkerAssignment:
    indent: str
    lhs: str
    arg: str
    name: str


# Inductor 生成代码里记号调用的固定写法
_GRAPH_MARKER_PREFIX = "torch.ops.aiter.graph_marker.default("


# 从 open_idx 处的 "(" 开始找配对的 ")"，跳过字符串字面量里的括号；找不到返回 None
def _find_matching_paren(s: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_str: Optional[str] = None
    escaped = False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if in_str is not None:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


# 按顶层逗号切分参数列表（忽略括号 / 方括号 / 花括号和字符串内部的逗号）
def _split_top_level_args(s: str) -> list[str]:
    out: list[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_str: Optional[str] = None
    escaped = False
    start = 0
    for i, ch in enumerate(s):
        if in_str is not None:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren -= 1
            continue
        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket -= 1
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            depth_brace -= 1
            continue
        if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


# expr 整个就是一次记号调用时，返回 (第一个参数的源码, 名字)，否则返回 None
def _parse_graph_marker_call_expr(expr: str) -> Optional[tuple[str, str]]:
    idx = expr.find(_GRAPH_MARKER_PREFIX)
    if idx < 0:
        return None
    open_idx = idx + len(_GRAPH_MARKER_PREFIX) - 1
    close_idx = _find_matching_paren(expr, open_idx)
    if close_idx is None:
        return None
    call_src = expr[idx : close_idx + 1]
    if call_src.count("(") != call_src.count(")"):
        return None
    # Ensure the matched call is the whole RHS expression (allowing whitespace).
    # 【译】确保匹配到的调用就是整个右侧表达式（允许空白）。
    if expr[:idx].strip() or expr[close_idx + 1 :].strip():
        return None
    inner = call_src[len(_GRAPH_MARKER_PREFIX) : -1]
    args = _split_top_level_args(inner)
    if len(args) < 2:
        return None
    arg_expr = args[0]
    try:
        name = ast.literal_eval(args[1])
    except Exception:
        return None
    if not isinstance(name, str):
        return None
    return arg_expr, name


# 解析旧式的一行 x = marker(y, 'name')
def _parse_graph_marker_assignment_line(line: str) -> Optional[_ParsedMarkerAssignment]:
    m = _ASSIGNMENT_RE.match(line.rstrip("\n"))
    if not m:
        return None
    rhs = m.group("rhs")
    parsed = _parse_graph_marker_call_expr(rhs)
    if parsed is None:
        return None
    arg, name = parsed
    return _ParsedMarkerAssignment(
        indent=m.group("indent"),
        lhs=m.group("lhs"),
        arg=arg,
        name=name,
    )


# 这一行里若有记号调用，返回它的名字（不要求是整行）
def _extract_graph_marker_name(line: str) -> Optional[str]:
    idx = line.find(_GRAPH_MARKER_PREFIX)
    if idx < 0:
        return None
    open_idx = idx + len(_GRAPH_MARKER_PREFIX) - 1
    close_idx = _find_matching_paren(line, open_idx)
    if close_idx is None:
        return None
    call_src = line[idx : close_idx + 1]
    inner = call_src[len(_GRAPH_MARKER_PREFIX) : -1]
    args = _split_top_level_args(inner)
    if len(args) < 2:
        return None
    try:
        name = ast.literal_eval(args[1])
    except Exception:
        return None
    return name if isinstance(name, str) else None


# 递归列出目录下所有 .py 文件
def _iter_py_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _initial_docstring_span(lines: list[str]) -> Optional[tuple[int, int]]:
    """
    Best-effort detection of the initial module docstring / leading triple-quoted
    block, including variants like r\"\"\"...\"\"\" and '''...'''.
    """
    # 【译】尽力找出开头的模块 docstring / 最前面的三引号块，包括 r"""...""" 和 '''...''' 这类变体。
    # 返回 (起始行, 结束行)，没有则 None；用于插入 import 时避开 docstring
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r'^(?:[rRuUbBfF]+)?(?P<quote>"""|\'\'\')', stripped)
        if not m:
            return None
        quote = m.group("quote")
        if quote in stripped[m.end() :]:
            return (i, i)
        for j in range(i + 1, len(lines)):
            if quote in lines[j]:
                return (i, j)
        return (i, len(lines) - 1)
    return None


# 确保文件里有 from torch.profiler import record_function
def _ensure_record_function_import(lines: list[str]) -> None:
    doc_span = _initial_docstring_span(lines)

    def _iter_code_lines() -> Iterable[tuple[int, str]]:
        for i, line in enumerate(lines):
            if doc_span is not None and doc_span[0] <= i <= doc_span[1]:
                continue
            yield i, line

    # If already imported or referenced via qualified name in real code, do nothing.
    # 【译】如果真实代码里已经导入过，或已用全限定名引用，就什么都不做。
    if any(
        ("record_function" in line and ("import" in line or "from torch" in line))
        for _, line in _iter_code_lines()
    ):
        return

    # Insert `from torch.profiler import record_function` after the first
    # real `import torch` line outside the initial docstring.
    # 【译】在开头 docstring 之外的第一行真正的 `import torch` 之后，插入 `from torch.profiler import record_function`。
    for i, line in _iter_code_lines():
        if re.match(r"^\s*import\s+torch\b", line):
            lines.insert(i + 1, "from torch.profiler import record_function\n")
            return

    # Fallback: put it near the top (after any shebang/encoding if present).
    # 【译】兜底：放在靠近开头的地方（如果有 shebang / 编码声明，放在它之后）。
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if doc_span is not None:
        insert_at = max(insert_at, doc_span[1] + 1)
    lines.insert(insert_at, "from torch.profiler import record_function\n")


# 扫描全文，收集所有记号（行号、缩进、名字）
def _collect_markers(lines: list[str]) -> list[_Marker]:
    out: list[_Marker] = []
    for i, line in enumerate(lines):
        name = _extract_graph_marker_name(line)
        if name is None:
            continue
        indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
        out.append(_Marker(idx=i, indent=indent, name=name))
    return out


# "mhc_start" → ("mhc", "start")，"mhc_end" → ("mhc", "end")，其他名字 → None
def _prefix_and_kind(name: str) -> Optional[tuple[str, str]]:
    if name.endswith("_start"):
        return name[: -len("_start")], "start"
    if name.endswith("_end"):
        return name[: -len("_end")], "end"
    return None


# 区间内是否已有 with record_function("<prefix>"): 这一行（没有调用方，见文件头的观察）
def _already_wrapped(
    lines: list[str], indent: str, prefix: str, start_idx: int, end_idx: int
) -> bool:
    needle = f'{indent}with record_function("{prefix}"):\n'
    for i in range(start_idx, min(end_idx, len(lines))):
        if lines[i] == needle:
            return True
    return False


def _wrap_region_with_record_function(
    lines: list[str],
    *,
    start_marker_idx: int,
    end_marker_idx: int,
    prefix: str,
    indent: str,
    layer_id: Optional[int] = None,
) -> None:
    """
    Transform:
      <indent>... graph_marker(..., "<prefix>_start")
      <indent>LINE_A
      <indent>LINE_B
      <indent>... graph_marker(..., "<prefix>_end")
    Into:
      <indent>... graph_marker(..., "<prefix>_start")
      <indent>with record_function("<prefix>"):
      <indent>    LINE_A
      <indent>    LINE_B
      <indent>... graph_marker(..., "<prefix>_end")
    """
    # 【译】把上面"Transform:"下的形式改写成"Into:"下的形式：在 start 记号之后插入
    #   with record_function("<prefix>"):，并把两个记号之间的行多缩进 4 格（end 记号本身不缩进）。
    # 两个记号相邻（中间没有代码）时什么都不做
    if end_marker_idx <= start_marker_idx + 1:
        return

    tag = prefix
    with_line = f'{indent}with record_function("{tag}"):\n'
    insert_at = start_marker_idx + 1

    # If we already inserted a record_function line in a previous run, upgrade it
    # in-place (e.g. "mlp" -> "layer_0_mlp") and exit without touching indentation.
    # 【译】如果上一次运行已经插入过 record_function 行，就原地升级它（例如 "mlp" -> "layer_0_mlp"），不动缩进直接返回。
    #   （tag 就是 prefix、不含 layer_id，所以这里实际上只会原样保留，见文件头的观察）
    if insert_at < len(lines) and lines[insert_at].startswith(
        f"{indent}with record_function("
    ):
        if lines[insert_at] != with_line:
            lines[insert_at] = with_line
        return

    # Otherwise, insert a new record_function wrapper and indent the region.
    # 【译】否则插入一个新的 record_function 包装，并缩进这个区间。
    lines.insert(insert_at, with_line)

    # Re-indent the region between start_marker and end_marker (exclusive of end marker).
    # 【译】重新缩进 start 记号和 end 记号之间的区间（不含 end 记号）。
    region_start = insert_at + 1
    region_end = end_marker_idx + 1  # end marker shifted down by 1 due to insertion
    indent_prefix = indent
    extra = " " * 4
    for i in range(region_start, region_end):
        line = lines[i]
        if line.strip() == "":
            continue
        if line.startswith(indent_prefix):
            lines[i] = indent_prefix + extra + line[len(indent_prefix) :]


def _layer_id_from_wrapper_path(path: str) -> Optional[int]:
    """
    Derive layer id from wrapper file path:
      .../artifact_shape_<shape>_subgraph_<N>/... -> layer_id = N - 1
    Returns None if the pattern isn't found.
    """
    # 【译】从 wrapper 文件路径推出层号：
    #     .../artifact_shape_<shape>_subgraph_<N>/... -> layer_id = N - 1
    #   找不到这个模式时返回 None。
    m = _SUBGRAPH_ID_RE.search(path)
    if not m:
        return None
    try:
        subgraph_id = int(m.group(1))
    except ValueError:
        return None
    return subgraph_id - 1


def _strip_runtime_graph_markers(lines: list[str]) -> bool:
    """
    Remove runtime overhead of graph markers in generated wrapper code.

    - Remove standalone, void graph marker calls.
    - Replace legacy `x = torch.ops.aiter.graph_marker.default(y, '...')` with `x = y`
    - Drop assert_size_stride / assert_alignment lines that specifically refer to
      `torch.ops.aiter.graph_marker.default` (they become redundant).
    """
    # 【译】去掉生成的 wrapper 代码里图标记的运行时开销。
    #
    #   - 删除独立的、无返回值的图标记调用。
    #   - 把旧式的 `x = torch.ops.aiter.graph_marker.default(y, '...')` 替换成 `x = y`
    #   - 删除专门针对 `torch.ops.aiter.graph_marker.default` 的 assert_size_stride / assert_alignment 行（它们变得多余）。
    # 返回是否改动过；只有 strip_markers=True 时才被调用
    out: list[str] = []
    changed = False
    for line in lines:
        if _parse_graph_marker_call_expr(line.strip()) is not None:
            # Preserve a valid suite even when a marker is its only statement.
            # 【译】即使记号是某个代码块里唯一的语句，也要保持代码块合法（所以换成 pass 而不是删掉）。
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}pass  # graph marker\n")
            changed = True
            continue
        parsed = _parse_graph_marker_assignment_line(line)
        if parsed is not None:
            out.append(f"{parsed.indent}{parsed.lhs} = {parsed.arg}\n")
            changed = True
            continue

        if (
            "assert_size_stride" in line or "assert_alignment" in line
        ) and "torch.ops.aiter.graph_marker.default" in line:
            changed = True
            continue

        out.append(line)

    if changed:
        lines[:] = out
    return changed


def instrument_record_functions_in_file(
    path: str, *, strip_markers: bool = True
) -> bool:
    """
    Returns True if the file was modified.
    """
    # 【译】文件被修改过时返回 True。
    # 步骤：读文件 → 收集记号 → 按名字配对成区间 → 自下而上逐个包进 record_function →（可选）去掉记号 → 补 import → 写回
    # 调用方：compiler_inferface.py 的 InductorAdaptor.compile（strip_markers=False）、下面的 instrument_record_functions_in_dir
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return False

    markers = _collect_markers(lines)
    if not markers:
        return False

    # Build intervals by matching <prefix>_start / <prefix>_end.
    # 【译】把 <prefix>_start / <prefix>_end 配对，构成区间。
    # 用 dict 按 prefix 记最近一个 start；遇到同 prefix 的 end 就配成一对
    stack: dict[str, _Marker] = {}
    intervals: list[tuple[_Marker, _Marker, str]] = []
    for mk in markers:
        pk = _prefix_and_kind(mk.name)
        if pk is None:
            continue
        prefix, kind = pk
        if kind == "start":
            stack[prefix] = mk
        else:
            start_mk = stack.pop(prefix, None)
            if start_mk is None:
                continue
            # Use start indent as the wrapping indent (generated code is consistent).
            # 【译】用 start 记号的缩进作为包装的缩进（生成的代码缩进是一致的）。
            intervals.append((start_mk, mk, prefix))

    # Even if we can't form any intervals, we still might want to strip marker
    # calls from already-instrumented wrappers (best-effort).
    # 【译】即使一个区间都配不出来，我们可能仍想从已经插过桩的 wrapper 里去掉记号调用（尽力而为）。
    has_intervals = bool(intervals)

    layer_id = _layer_id_from_wrapper_path(path)

    # Apply from bottom to top so indices stay valid.
    # 【译】从下往上改，这样前面的行号保持有效（每次插入只会让后面的行下移）。
    wrapped_or_upgraded = False
    if has_intervals:
        for start_mk, end_mk, prefix in sorted(
            intervals, key=lambda t: t[0].idx, reverse=True
        ):
            _wrap_region_with_record_function(
                lines,
                start_marker_idx=start_mk.idx,
                end_marker_idx=end_mk.idx,
                prefix=prefix,
                indent=start_mk.indent,
                layer_id=layer_id,
            )
            wrapped_or_upgraded = True

    stripped = False
    if strip_markers:
        # Only strip marker calls if we either:
        # - wrapped/upgraded this run, or
        # - the file already contains record_function blocks (previous run)
        # 【译】只在以下两种情况之一时才去掉记号调用：本次运行包装 / 升级过；或者文件里已有 record_function 块（之前运行留下的）。
        already_has_record = any("with record_function(" in line for line in lines)
        if wrapped_or_upgraded or already_has_record:
            stripped = _strip_runtime_graph_markers(lines)

    changed = wrapped_or_upgraded or stripped
    if changed:
        if wrapped_or_upgraded:
            _ensure_record_function_import(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def instrument_record_functions_in_dir(root: str, *, strip_markers: bool = True) -> int:
    """
    Walk `root` and instrument all generated `.py` wrapper files.
    Returns the number of modified files.
    """
    # 【译】遍历 `root`，给所有生成的 `.py` wrapper 文件插桩。返回被修改的文件数。
    # 调用方：compiler_inferface.py 的 InductorStandaloneAdaptor.compile（strip_markers=False）
    changed = 0
    for fp in _iter_py_files(root):
        if instrument_record_functions_in_file(fp, strip_markers=strip_markers):
            changed += 1
    return changed
