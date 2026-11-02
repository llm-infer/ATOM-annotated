"""Check that annotated files differ from git HEAD only in comments/whitespace.

Usage: python verify.py file1.py file2.py ...   (paths relative to repo root)
AST (which carries docstrings but not comments) must be identical to HEAD's.
"""
import ast, subprocess, sys

bad = 0
for f in sys.argv[1:]:
    old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True, check=True).stdout
    new = open(f).read()
    a, b = ast.dump(ast.parse(old)), ast.dump(ast.parse(new))
    added = sum(1 for l in new.splitlines() if l.strip().startswith("#")) - sum(
        1 for l in old.splitlines() if l.strip().startswith("#"))
    status = "OK " if a == b else "AST CHANGED"
    bad += a != b
    print(f"{status} {f}  (+{added} comment lines, {len(old.splitlines())} -> {len(new.splitlines())} lines)")

r1 = subprocess.run(["black", "--check", *sys.argv[1:]], capture_output=True, text=True)
print("black:", "OK" if r1.returncode == 0 else r1.stderr)
r2 = subprocess.run(["ruff", "check", *sys.argv[1:]], capture_output=True, text=True)
print("ruff:", "OK" if r2.returncode == 0 else r2.stdout)
sys.exit(1 if bad or r1.returncode or r2.returncode else 0)
