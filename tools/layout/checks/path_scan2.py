"""Path expressions built from the file's own place or a root variable, checked against a tree (read-only, git objects only).

    python -I path_scan2.py <repo> <rev> <rev before the move, for the list of moved top-level names>

The first scan (path_scan.py) resolves quoted segment chains. It missed `Path(__file__).parents[1] / "include" / "x.h"`: a pathlib or
os.path expression whose base is `__file__` or a root variable and whose segments are string constants. This scan evaluates such
expressions in Python (ast), JS / MJS / TS (join / resolve / new URL with import.meta.dirname, __dirname or a root variable defined in the
same file) and shell (`dirname "$0"`, `$here`, `$repo` variables), turns each into a repository path, and reports it when the path does not
exist in <rev> and the first missing component is the name of a top-level folder or file the move took away (include, src, serve ...).
Paths whose first missing component is anything else (build, .venv, logs: made at run time) are not reported.
"""
import ast
import posixpath
import re
import subprocess
import sys

repo, rev, before = sys.argv[1:4]


def git(*a):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def top(r):
    return {x for x in git("ls-tree", "--name-only", r).split("\n") if x}


files = [f for f in git("ls-tree", "-r", "--name-only", rev).split("\n") if f]
fset = set(files)
dirs = set()
for f in files:
    d = posixpath.dirname(f)
    while d and d not in dirs:
        dirs.add(d)
        d = posixpath.dirname(d)
moved = (top(before) - top(rev)) | {"tools", "docs"}
moved = {m for m in moved if m}
SKIP = ("engine/third_party/", "docs/bench-results/", "docs/history/", "tools/layout/", "app/host/node_modules/", "tools/research/")


def exists(p):
    return p in fset or p in dirs or p == ""


def first_missing(p):
    parts = p.split("/")
    cur = ""
    for x in parts:
        nxt = (cur + "/" + x) if cur else x
        if not exists(nxt):
            return x, cur
        cur = nxt
    return None, cur


def judge(path_, where, line, out, what):
    if path_ is None or path_.startswith("..") or path_ == ".":
        return
    miss, parent = first_missing(path_)
    if miss and miss in moved and exists(parent):
        out.append("%s:%d: %s -> %s (the folder or file '%s' is not in %s/)" % (where, line, what, path_, miss, parent or "."))


# ------------------------------------------------------------------------------------------ Python
class P:      # a path value: posix repo-relative string, or None when unknown
    pass


def eval_py(node, env, here):
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return here
        return env.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ("const", node.value)
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in ("Path", "PurePath", "PosixPath") and len(node.args) == 1:
            v = eval_py(node.args[0], env, here)
            return v if isinstance(v, str) else None
        if name in ("resolve", "absolute", "expanduser", "abspath", "realpath", "normpath") and isinstance(fn, ast.Attribute):
            return eval_py(fn.value, env, here) if not node.args or name in ("abspath", "realpath", "normpath") else None
        if isinstance(fn, ast.Attribute) and name in ("abspath", "realpath", "normpath") and node.args:
            return eval_py(node.args[0], env, here)
        if name == "dirname" and len(node.args) == 1:
            v = eval_py(node.args[0], env, here)
            return posixpath.dirname(v) if isinstance(v, str) else None
        if name == "join" and node.args:
            v = eval_py(node.args[0], env, here)
            if not isinstance(v, str):
                return None
            for a in node.args[1:]:
                s = eval_py(a, env, here)
                if not (isinstance(s, tuple) and s[0] == "const"):
                    return None
                v = posixpath.normpath(posixpath.join(v, s[1]))
            return v
        return None
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            v = eval_py(node.value, env, here)
            return posixpath.dirname(v) if isinstance(v, str) else None
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents" and isinstance(node.slice, ast.Constant):
        v = eval_py(node.value.value, env, here)
        if not isinstance(v, str):
            return None
        for _ in range(node.slice.value + 1):
            v = posixpath.dirname(v)
        return v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        v = eval_py(node.left, env, here)
        s = eval_py(node.right, env, here)
        if isinstance(v, str) and isinstance(s, tuple) and s[0] == "const":
            return posixpath.normpath(posixpath.join(v, s[1]))
        return None
    return None


def scan_py(f, text, out):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    env = {}
    for node in tree.body:                       # module-level variables, in order
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = eval_py(node.value, env, f)
            if isinstance(v, str):
                env[node.targets[0].id] = v
    seen = set()

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local = dict(scope)                  # a function sees the module's variables and its own, not another function's
                for st in ast.walk(child):
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                        v = eval_py(st.value, local, f)
                        if isinstance(v, str):
                            local[st.targets[0].id] = v
                visit(child, local)
                continue
            if isinstance(child, (ast.BinOp, ast.Call)):
                v = eval_py(child, scope, f)
                if isinstance(v, str):               # the longest expression that is a path: its parts are not reported again
                    if (child.lineno, v) not in seen:
                        seen.add((child.lineno, v))
                        judge(v, f, child.lineno, out, "path expression")
                    continue
            visit(child, scope)

    visit(tree, env)


# ------------------------------------------------------------------------------------------ JS / MJS / TS
JS_ASSIGN = re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(.+?);?\s*$")
JS_CALL = re.compile(r"\b(?:path\.)?(join|resolve)\(\s*([^()]*?)\)")
JS_URL = re.compile(r"new URL\(\s*(['\"])([^'\"]+)\1\s*,\s*import\.meta\.url\s*\)")


def js_value(expr, env, here):
    expr = expr.strip()
    m = re.fullmatch(r"(?:path\.)?(?:join|resolve)\((.*)\)", expr)
    if m:
        parts = [x.strip() for x in re.split(r",(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)", m.group(1)) if x.strip()]
        return js_path(parts, env, here)
    m = re.fullmatch(r"(?:path\.)?dirname\((.*)\)", expr)
    if m:
        v = js_value(m.group(1), env, here)
        return posixpath.dirname(v) if isinstance(v, str) else None
    m = re.fullmatch(r"fileURLToPath\(\s*import\.meta\.url\s*\)", expr)
    if m:
        return here
    m = re.fullmatch(r"fileURLToPath\(\s*new URL\(\s*(['\"])([^'\"]+)\1\s*,\s*import\.meta\.url\s*\)\s*\)", expr)
    if m:
        return posixpath.normpath(posixpath.join(posixpath.dirname(here), m.group(2)))
    if expr in ("import.meta.dirname", "__dirname"):
        return posixpath.dirname(here)
    if expr == "import.meta.url":
        return here
    return env.get(expr)


def js_path(parts, env, here):
    if not parts:
        return None
    base = js_value(parts[0], env, here)
    if not isinstance(base, str):
        return None
    for p in parts[1:]:
        m = re.fullmatch(r"(['\"`])([^'\"`$]*)\1", p)
        if not m:
            return None
        base = posixpath.normpath(posixpath.join(base, m.group(2)))
    return base


def scan_js(f, text, out):
    env = {}
    lines = text.split("\n")
    for _pass in (1, 2):
        for n, l in enumerate(lines, 1):
            m = JS_ASSIGN.match(l.strip())
            if m and m.group(1) not in env:
                v = js_value(m.group(2), env, f)
                if isinstance(v, str):
                    env[m.group(1)] = v
    seen = set()
    for n, l in enumerate(lines, 1):
        for m in JS_CALL.finditer(l):
            v = js_value("%s(%s)" % (m.group(1), m.group(2)), env, f)
            if isinstance(v, str) and (n, v) not in seen:
                seen.add((n, v))
                judge(v, f, n, out, "path expression")
        for m in JS_URL.finditer(l):
            v = posixpath.normpath(posixpath.join(posixpath.dirname(f), m.group(2)))
            judge(v, f, n, out, "new URL")


# ------------------------------------------------------------------------------------------ shell
SH_VAR = re.compile(r"^\s*(?:export\s+|local\s+|readonly\s+)?([A-Za-z_]\w*)=(.+?)\s*$")


def sh_value(expr, env, here):
    e = expr.strip().strip('"').strip("'")
    # $(dirname "$0") / $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) / $(CDPATH= cd -- "$here/.." && pwd -P)
    if re.search(r"dirname\s+\"?\$\{?(?:0|BASH_SOURCE(?:\[0\])?)\}?\"?", e) and "&&" in e and "cd" in e:
        base = posixpath.dirname(here)
        m = re.search(r"cd\s+(?:--\s+)?\"?\$\(dirname[^)]*\)(/[^\"\s&]*)?\"?", e)
        if m and m.group(1):
            base = posixpath.normpath(posixpath.join(base, m.group(1).lstrip("/")))
        return base
    m = re.search(r"cd\s+(?:--\s+)?\"?\$\{?(\w+)\}?(/[^\"\s&]*)?\"?\s*&&", e)
    if m and m.group(1) in env:
        base = env[m.group(1)]
        if m.group(2):
            base = posixpath.normpath(posixpath.join(base, m.group(2).lstrip("/")))
        return base
    m = re.fullmatch(r"\$\(dirname\s+\"?\$\{?(?:0|BASH_SOURCE(?:\[0\])?)\}?\"?\)(/[^\"\s]*)?", e)
    if m:
        base = posixpath.dirname(here)
        if m.group(1):
            base = posixpath.normpath(posixpath.join(base, m.group(1).lstrip("/")))
        return base
    m = re.fullmatch(r"\$\{?(\w+)\}?(/[^\"\s]*)?", e)
    if m and m.group(1) in env:
        base = env[m.group(1)]
        if m.group(2):
            base = posixpath.normpath(posixpath.join(base, m.group(2).lstrip("/")))
        return base
    return None


def scan_sh(f, text, out):
    env = {}
    lines = text.split("\n")
    for n, l in enumerate(lines, 1):
        m = SH_VAR.match(l)
        if m:
            v = sh_value(m.group(2), env, f)
            if isinstance(v, str):
                env[m.group(1)] = v
    seen = set()
    for n, l in enumerate(lines, 1):
        if l.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\"?\$\{?(\w+)\}?/([A-Za-z0-9_.][A-Za-z0-9_./-]*)", l):
            if m.group(1) in env:
                v = posixpath.normpath(posixpath.join(env[m.group(1)], m.group(2)))
                if (n, v) not in seen:
                    seen.add((n, v))
                    judge(v, f, n, out, "$%s/%s" % (m.group(1), m.group(2)))
        for m in re.finditer(r"cd\s+(?:--\s+)?\"?\$\(dirname\s+\"?\$\{?(?:0|BASH_SOURCE(?:\[0\])?)\}?\"?\)/([A-Za-z0-9_./-]+)", l):
            v = posixpath.normpath(posixpath.join(posixpath.dirname(f), m.group(1)))
            if (n, v) not in seen:
                seen.add((n, v))
                judge(v, f, n, out, "cd dirname/%s" % m.group(1))


# ------------------------------------------------------------------------------------------ batch / PowerShell
BAT_SET = re.compile(r"^\s*set\s+\"?(\w+)=([^\"]*)\"?\s*$", re.I)


def bat_value(v, env, here):
    v = v.strip().replace("\\", "/")
    m = re.match(r"%~dp0(.*)$", v, re.I)
    if m:
        return posixpath.normpath(posixpath.join(posixpath.dirname(here), m.group(1).strip("/"))) if m.group(1).strip("/") else posixpath.dirname(here)
    m = re.match(r"%(\w+)%(.*)$", v)
    if m and m.group(1) in env:
        rest = m.group(2).strip("/")
        return posixpath.normpath(posixpath.join(env[m.group(1)], rest)) if rest else env[m.group(1)]
    return None


def scan_bat(f, text, out):
    env = {}
    lines = text.split("\n")
    for n, l in enumerate(lines, 1):
        m = BAT_SET.match(l.rstrip("\r"))
        if m:
            v = bat_value(m.group(2), env, f)
            if isinstance(v, str):
                env[m.group(1)] = v
    seen = set()
    for n, l in enumerate(lines, 1):
        if re.match(r"\s*(rem|::)\b", l, re.I):
            continue
        for m in re.finditer(r"(%~dp0|%(\w+)%)([\\/][A-Za-z0-9_.\\/ -]*[A-Za-z0-9_.])", l):
            base = bat_value(m.group(1), env, f) if m.group(1) != "%~dp0" else posixpath.dirname(f)
            if isinstance(base, str):
                v = posixpath.normpath(posixpath.join(base, m.group(3).replace("\\", "/").strip("/")))
                if (n, v) not in seen:
                    seen.add((n, v))
                    judge(v, f, n, out, m.group(0).strip()[:60])
        for m in re.finditer(r"cd\s+/d\s+\"?%~dp0([^\"\s&]*)", l, re.I):
            v = posixpath.normpath(posixpath.join(posixpath.dirname(f), m.group(1).replace("\\", "/").strip("/"))) if m.group(1).strip("\\/") else posixpath.dirname(f)
            if (n, v) not in seen:
                seen.add((n, v))
                judge(v, f, n, out, "cd /d %~dp0" + m.group(1))


def scan_ps1(f, text, out):
    seen = set()
    for n, l in enumerate(text.split("\n"), 1):
        for m in re.finditer(r"\$PSScriptRoot([\\/][A-Za-z0-9_.\\/-]*[A-Za-z0-9_.])", l):
            v = posixpath.normpath(posixpath.join(posixpath.dirname(f), m.group(1).replace("\\", "/").strip("/")))
            if (n, v) not in seen:
                seen.add((n, v))
                judge(v, f, n, out, m.group(0)[:60])
        for m in re.finditer(r"Join-Path\s+\$PSScriptRoot\s+['\"]([^'\"$]+)['\"]", l):
            v = posixpath.normpath(posixpath.join(posixpath.dirname(f), m.group(1).replace("\\", "/")))
            if (n, v) not in seen:
                seen.add((n, v))
                judge(v, f, n, out, "Join-Path $PSScriptRoot " + m.group(1))


# ------------------------------------------------------------------------------------------ CI workflows and Docker files: repository paths
YML_TOK = re.compile(r"(?<![\w./-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)(?![\w./-])")


def scan_yml(f, text, out):
    seen = set()
    for n, l in enumerate(text.split("\n"), 1):
        if "http" in l or l.lstrip().startswith("#"):
            continue
        for m in YML_TOK.finditer(l):
            t = m.group(1)
            if "*" in t or t.startswith((".", "~", "/")) or "$" in t:
                continue
            first = t.split("/")[0]
            if first in moved and not exists(t) and (n, t) not in seen:
                seen.add((n, t))
                miss, parent = first_missing(t)
                out.append("%s:%d: repository path %s does not exist (the folder or file '%s' is not in %s/)" % (f, n, t, miss, parent or "."))


# ------------------------------------------------------------------------------------------ driver
want = [f for f in files if not f.startswith(SKIP) and (f.endswith((".py", ".mjs", ".js", ".ts", ".tsx", ".sh", ".cjs", ".bat", ".cmd", ".ps1"))
        or (f.startswith(".github/") and f.endswith((".yml", ".yaml"))))]
proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out = []
count = 0
for f in want:
    proc.stdin.write(("%s:%s\n" % (rev, f)).encode())
    proc.stdin.flush()
    head = proc.stdout.readline().decode().split()
    if len(head) < 3:
        continue
    size = int(head[2])
    body = proc.stdout.read(size).decode("utf-8", errors="replace")
    proc.stdout.read(1)
    count += 1
    if f.endswith(".py"):
        scan_py(f, body, out)
    elif f.endswith(".sh"):
        scan_sh(f, body, out)
    elif f.endswith((".bat", ".cmd")):
        scan_bat(f, body, out)
    elif f.endswith(".ps1"):
        scan_ps1(f, body, out)
    elif f.endswith((".yml", ".yaml")):
        scan_yml(f, body, out)
    else:
        scan_js(f, body, out)
proc.stdin.close()
proc.wait()
for x in out:
    print(x)
print("path scan 2: %d files read at %s, moved names: %s; %d HITS" % (count, rev[:9], ", ".join(sorted(moved)), len(out)))
sys.exit(1 if out else 0)
