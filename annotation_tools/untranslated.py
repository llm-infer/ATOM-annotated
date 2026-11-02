"""List English comments / docstrings that lack a following `# 【译】` translation.

Usage: python untranslated.py file1.py ...   Exit 1 if anything is missing.
- English comment block: consecutive `#` lines with no CJK; the next line must be a
  `# 【译】` comment. Skips SPDX/Copyright, pragmas (noqa/type:/fmt:/pragma), and
  blocks that look like commented-out code.
- Docstring (module/class/def, and attribute docstrings after a field): the first non-blank line after its closing quotes must
  be a `# 【译】` comment.
"""
import ast, re, sys

CJK = re.compile("[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2500-\u257f]")
CODE_HINT = re.compile(r"^#\s*(def |class |return\b|self\.|@\w|import |from \w+ import|\w+\s*=\s*\S|\)|\]|if .*:$|else:|for .* in .*:$)")
PRAGMA = re.compile(r"#\s*(noqa|type:|fmt:|pragma|pylint|SPDX|Copyright|All rights reserved)", re.I)
bad = 0
for path in sys.argv[1:]:
    src = open(path).read(); lines = src.splitlines()
    miss = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#") and not CJK.search(s) and re.search(r"[A-Za-z]{3,}", s) and not PRAGMA.search(s):
            j = i
            while j < len(lines) and lines[j].strip().startswith("#") and not CJK.search(lines[j]):
                j += 1
            block = [l.strip() for l in lines[i:j]]
            codeish = sum(bool(CODE_HINT.match(b)) for b in block) * 2 >= len(block)
            nxt = lines[j].strip() if j < len(lines) else ""
            if not codeish and "【译】" not in nxt:
                miss.append((i + 1, "comment", block[0][:70]))
            i = j
        else:
            i += 1
    tree = ast.parse(src)
    # every bare string statement: module/class/def docstrings and attribute docstrings
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        doc = node.value.value
        if not re.search(r"[A-Za-z]{3,}", doc) or CJK.search(doc):
            continue
        k = node.end_lineno  # 1-based line of closing quotes == 0-based index of next line
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k >= len(lines) or "【译】" not in lines[k]:
            miss.append((node.lineno, "docstring", doc.strip().splitlines()[0][:60]))
    miss.sort()
    bad += len(miss)
    print(f"{path}: {len(miss)} untranslated")
    for m in miss:
        print(f"  L{m[0]} {m[1]}: {m[2]}")
sys.exit(1 if bad else 0)
