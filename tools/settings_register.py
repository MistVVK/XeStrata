#!/usr/bin/env python3
"""The settings register: every STRATA_* environment setting the engine reads, in one file (terminal/data/settings-register.json).

    python tools/settings_register.py             # regenerate terminal/data/settings-register.json (keeps the hand-kept fields)
    python tools/settings_register.py --check     # scan only; print what is missing from / extra in the register; exit 1 on a difference
    python tools/settings_register.py --names     # print the names the engine reads, one per line
    python tools/settings_register.py --no-git    # regenerate without the git history fields (fast)
    python tools/settings_register.py --docs      # write docs/SETTINGS.md from the register

What counts as "the engine reads it": the name is the string literal argument of a ``getenv`` call or of one of the engine's env
helpers (``env_on``, ``env_int``, ``pipe_dbg_env`` ...) in engine/src/, engine/include/, engine/metal/src/ or sycl/src/. Calls that write or remove a
variable (``_putenv_s``, ``put_env``, ``set_env``, ``unset_env``) are not reads; they are listed as ``engine_sets``.

Automatic fields (rebuilt on every run): backends, reads, parse, code_default, default_state, engine_sets, launch_sets, launch_reads,
doc_mentions, first/last commit and the first release that has the name.

Hand-kept fields (copied over from the old file on every run, never written by the scan): status, owner, note, retire_after, retired_in.
``status`` is one of STATUSES below. A new name comes in with status "" and the test refuses an empty status until somebody sets it.

Only the standard library is used.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "terminal" / "data" / "settings-register.json"
SETTINGS_PAGE = "docs/SETTINGS.md"

ENGINE_DIRS = ("engine/src", "engine/include", "engine/metal/src", "engine/sycl/src")
LAUNCH_PATHS = ("strata_control", "strata_cli", "terminal/setup.py", "app/host/server", "serve")
DOC_PATHS = ("docs", "README.md", "AGENTS.md")
LAUNCH_EXT = (".py", ".mjs", ".js", ".cjs", ".ts")
SKIP_PARTS = ("node_modules", "/tests/", "/test/", "__pycache__")

STATUSES = ("default", "rollback", "opt-in", "research", "retire-candidate", "unsure", "retired")
HAND_FIELDS = ("status", "owner", "note", "retire_after", "retired_in")

NAME_RE = re.compile(r"STRATA_[A-Z0-9_]+")
# a call whose first argument is the name literal: getenv, env_on, env_int, pipe_dbg_env, il_env_on, kva_env_flag ...
CALL_RE = re.compile(r'(\w*(?:getenv|env\w*|\w*_env\w*)\s*\(\s*)"(STRATA_[A-Z0-9_]+)"')
WRITE_FN_RE = re.compile(r"put|set_env|unset|setenv")

FLAG_FNS = {"env_on", "env_flag", "il_env_on", "kva_env_flag", "env_tri"}
INT_FNS = {"env_int", "env_int_or", "kva_env_i64", "a_env", "env_count", "lanes_env"}
FLOAT_FNS = {"env_num", "env_f"}

HIP_MACROS = {"STRATA_USE_HIP", "__HIPCC__", "STRATA_HIP_GFX906", "FNX_HIP", "__HIP_PLATFORM_AMD__", "STRATA_HIPBLASLT_AVAILABLE",
              "STRATA_ROCBLAS_AVAILABLE", "__HIP_DEVICE_COMPILE__", "STRATA_PREFILL_MMQ_HIP"}
CUDA_ONLY_MACROS = {"__CUDA_ARCH__", "CUDART_VERSION", "__NVPTX__", "STRATA_EXPERIMENTAL_SM60"}
METAL_MACROS = {"STRATA_METAL", "STRATA_GPU_METAL", "__APPLE__"}


# ---------------------------------------------------------------- file walking

def _files(rel_paths, exts=None):
    for rel in rel_paths:
        base = ROOT / rel
        if base.is_file():
            yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("node_modules", "__pycache__", ".git")]
            for fn in filenames:
                if exts and not fn.endswith(exts):
                    continue
                yield Path(dirpath) / fn


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _read_lines(p: Path):
    try:
        return p.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return []


# ---------------------------------------------------------------- preprocessor reachability (CUDA / HIP / Metal)

_TOKEN = re.compile(r"!|&&|\|\||\(|\)|defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)|[A-Za-z_]\w*|\d+")


def _eval(expr: str, cfg: str):
    """Three-valued evaluation of a preprocessor condition for one build flavour; None is unknown (treated as possibly true)."""
    toks = []
    for m in _TOKEN.finditer(expr):
        toks.append(m)
    pos = 0

    def val(sym: str):
        if sym in HIP_MACROS:
            return cfg == "hip"
        if sym in CUDA_ONLY_MACROS:
            return cfg == "cuda"
        if sym in METAL_MACROS:
            return cfg == "metal"
        return None

    def primary():
        nonlocal pos
        if pos >= len(toks):
            return None
        m = toks[pos]
        t = m.group(0)
        if t == "!":
            pos += 1
            v = primary()
            return None if v is None else (not v)
        if t == "(":
            pos += 1
            v = orx()
            if pos < len(toks) and toks[pos].group(0) == ")":
                pos += 1
            return v
        pos += 1
        if t.startswith("defined"):
            return val(m.group(1) or m.group(2))
        if t.isdigit():
            return None if t not in ("0", "1") else (t == "1")
        return val(t)

    def andx():
        nonlocal pos
        v = primary()
        while pos < len(toks) and toks[pos].group(0) == "&&":
            pos += 1
            w = primary()
            v = False if (v is False or w is False) else (None if (v is None or w is None) else True)
        return v

    def orx():
        nonlocal pos
        v = andx()
        while pos < len(toks) and toks[pos].group(0) == "||":
            pos += 1
            w = andx()
            v = True if (v is True or w is True) else (None if (v is None or w is None) else False)
        return v

    # comparisons (">=", "<") make the whole condition unknown; the tokenizer skips them, so give up on such a line
    if re.search(r"[<>=]=?|[+\-*/]", re.sub(r"&&|\|\||!defined|!\s*\(|!\s*\w", "", expr)):
        return None
    return orx()


def _pp_states(lines):
    """For every line, the set of build flavours in which it can be compiled: a subset of {"cuda", "hip", "metal"}."""
    flavours = ("cuda", "hip", "metal")
    stack = []   # each: (cond_values {cfg: bool|None}, parent_alive set) for the current branch
    out = []
    alive = set(flavours)
    for line in lines:
        s = line.lstrip()
        if s.startswith("#"):
            d = re.sub(r"^#\s*", "", s)
            m = re.match(r"(ifdef|ifndef|if|elif|else|endif)\b(.*)", d)
            if m:
                kw, rest = m.group(1), re.sub(r"(//.*|/\*.*)$", "", m.group(2)).strip()
                if kw in ("if", "ifdef", "ifndef"):
                    if kw == "ifdef":
                        cond = {c: _eval(f"defined({rest.split()[0]})" if rest.split() else "", c) for c in flavours}
                    elif kw == "ifndef":
                        cond = {c: _eval(f"!defined({rest.split()[0]})" if rest.split() else "", c) for c in flavours}
                    else:
                        cond = {c: _eval(rest, c) for c in flavours}
                    stack.append({"parent": set(alive), "cond": cond, "taken": {c: (cond[c] is True) for c in flavours}})
                    alive = {c for c in stack[-1]["parent"] if cond[c] is not False}
                elif kw == "elif" and stack:
                    top = stack[-1]
                    cond = {c: _eval(rest, c) for c in flavours}
                    alive = {c for c in top["parent"] if cond[c] is not False and not top["taken"][c]}
                    for c in flavours:
                        top["taken"][c] = top["taken"][c] or cond[c] is True
                elif kw == "else" and stack:
                    top = stack[-1]
                    alive = {c for c in top["parent"] if not top["taken"][c]}
                elif kw == "endif" and stack:
                    top = stack.pop()
                    alive = set(top["parent"])
        out.append(alive)
    return out


# ---------------------------------------------------------------- engine scan

def _backend_of(path: str, flavours: set[str]) -> list[str]:
    p = path
    if p.startswith("engine/sycl/src"):
        return ["sycl"]
    if p.startswith("engine/metal/src"):
        return ["metal"]
    if "/cpu/" in p or p.endswith(("_cpu.cpp", "_cpu.hpp")):
        return ["cpu"]
    fl = flavours
    if fl >= {"cuda", "hip"}:
        res = ["cuda-hip"]
    elif fl == {"cuda"}:
        res = ["cuda"]
    elif fl == {"hip"}:
        res = ["hip"]
    elif fl == {"metal"}:
        res = ["metal"]
    elif not fl:
        res = ["cuda-hip"]
    else:
        res = ["cuda-hip"] if "cuda" in fl or "hip" in fl else sorted(fl)
    base = os.path.basename(p).lower()
    if res == ["cuda-hip"]:
        if "hipblaslt" in base or "rocblas" in base or "/hip" in p or "hip_" in base:
            res = ["hip"]
        elif re.search(r"sm\d\d|nv_defaults|nvfp4|cublas|/nvidia", p.lower()) and "hip" not in base:
            res = ["cuda"]
    return res


_INT_RE = re.compile(r"\b(atoi|stoi|strtol|strtoll|strtoul|strtoull|atol|atoll|stoll|stoul|stoull)\b")
_FLOAT_RE = re.compile(r"\b(atof|strtod|strtof|stof|stod)\b")
_NUM = r"-?(?:0x[0-9a-fA-F]+|\d[\d.']*(?:e[+-]?\d+)?)[fFuUlL]*"


def _num_clean(s: str) -> str:
    return re.sub(r"[fFuUlL]+$", "", s.replace("'", ""))


def _analyze(lines, i, name, fn, col):
    """parse kind and code default for one read. i is the 0-based line index, col the column where the call starts."""
    fn0 = fn.strip().rstrip("(").strip()
    short = fn0.split("::")[-1]
    first = lines[i][col:]
    # arguments of a helper that takes a default: env_int("X", 5)
    argm = re.match(r'[^"]*"' + re.escape(name) + r'"\s*,\s*([^),;]+(?:\([^)]*\))?[^),;]*)\)', first)
    if short in FLAG_FNS or short in INT_FNS or short in FLOAT_FNS or short == "pipe_dbg_env":
        arg = argm.group(1).strip() if argm else None
        if short == "il_env_on":
            return "flag", "on", "set to 0 turns it off"
        if short in ("env_on", "env_flag"):
            return "flag", "off", "unset is off"
        if short == "kva_env_flag":
            d = None if arg is None else ("on" if arg in ("true", "1") else "off" if arg in ("false", "0") else None)
            return "flag", d, arg
        if short == "env_tri":
            return "flag", None, "unset / 0 / non-zero"
        if short in INT_FNS:
            return "int", None, (_num_clean(arg) if arg else None)
        if short in FLOAT_FNS:
            return "float", None, (_num_clean(arg) if arg else None)
        if short == "pipe_dbg_env":
            return "debug-gated", None, "read only when STRATA_PIPELINE_DEBUG=1"
    # std::getenv: look at the rest of the statement and the next two lines
    win = [first]
    for k in range(1, 4):
        if i + k < len(lines):
            win.append(lines[i + k])
    text = "\n".join(win)
    cut = re.search(r'getenv\(\s*"STRATA_', text[len(first.split(name, 1)[0]) + len(name):])
    if cut:
        text = text[: len(first.split(name, 1)[0]) + len(name) + cut.start()]
    prev = lines[i - 1] if i > 0 else ""
    kind = None
    if _FLOAT_RE.search(text):
        kind = "float"
    elif _INT_RE.search(text):
        kind = "int"
    elif re.search(r"'0'|\"0\"|!=\s*nullptr|==\s*nullptr|!=\s*NULL|==\s*NULL|\[0\]|\*\w+\s*(?:!=|==)|&&|\|\||\bif\s*\(|!\w", text):
        kind = "flag"
    else:
        kind = "string"
    default_state = None
    code_default = None
    if kind == "flag":
        # Only the exact shapes are read; anything else stays None ("not readable from the call").
        t = re.sub(r"\s+", " ", text)
        on = (re.search(r"(?:==\s*(?:nullptr|NULL)|!\s*\w+)\s*\|\|\s*\(?\w+(?:\[0\]|\s*\[\s*0\s*\])\s*!=\s*'0'", t)       # unset or not "0"
              or re.search(r"!\(\s*\w+\s*&&\s*\w+\[0\]\s*==\s*'0'\s*\)", t)                                               # !(e && e[0]=='0')
              or re.search(r"(?:==\s*(?:nullptr|NULL)|!\s*\w+)\s*\|\|\s*std::strcmp\(\s*\w+\s*,\s*\"0\"\s*\)\s*!=\s*0", t)
              or re.search(r"\b\w+\s*&&\s*\w+\[0\]\s*==\s*'0'\s*\)?\s*(?:return|\{)", t))                                # if (e && e[0]=='0') off
        off = (re.search(r"!=\s*(?:nullptr|NULL)\s*&&\s*\(?\w+\[0\]\s*!=\s*'0'", t)
               or re.search(r"!=\s*(?:nullptr|NULL)\s*&&\s*\*?\w+\s*!=\s*'0'", t)
               or re.search(r"\b\w+\s*&&\s*\w+\[0\]\s*(?:!=\s*'0'|==\s*'1')", t)
               or re.search(r"return\s+\w+\s*&&\s*std::atoi\(\s*\w+\s*\)\s*!=\s*0", t)
               or re.search(r"!=\s*(?:nullptr|NULL)\s*&&\s*std::atoi\(", t))
        if on and not off:
            default_state = "on"
        elif off and not on:
            default_state = "off"
        elif not on and not off:
            # presence only: getenv("X") != nullptr (set = on) / == nullptr (set = off)
            pres = re.match(r'getenv\("' + re.escape(name) + r'"\)\s*(!=|==)\s*(?:nullptr|NULL)\b(?!\s*&&)', t)
            if pres:
                default_state = "off" if pres.group(1) == "!=" else "on"
    elif kind in ("int", "float"):
        m = re.search(r"\)\s*:\s*(" + _NUM + r")", text) or re.search(r"\?\s*[^:;]*:\s*(" + _NUM + r")\s*[;,)]", text)
        if m:
            code_default = _num_clean(m.group(1))
        else:
            m = re.search(r"=\s*(" + _NUM + r")\s*;\s*$", prev)
            if m and re.search(r"\bif\s*\(", text):
                code_default = _num_clean(m.group(1))
    return kind, default_state, code_default


def scan_engine():
    reads = collections.defaultdict(list)
    sets = collections.defaultdict(list)
    for p in _files(ENGINE_DIRS):
        rel = _rel(p)
        lines = _read_lines(p)
        if not any("STRATA_" in l for l in lines):
            continue
        states = None
        for i, l in enumerate(lines):
            if "STRATA_" not in l:
                continue
            for m in CALL_RE.finditer(l):
                fn, name = m.group(1), m.group(2)
                if states is None:
                    states = _pp_states(lines)
                if WRITE_FN_RE.search(fn):
                    sets[name].append({"file": rel, "line": i + 1})
                    continue
                kind, dstate, dflt = _analyze(lines, i, name, fn, m.start())
                reads[name].append({
                    "file": rel, "line": i + 1,
                    "backend": _backend_of(rel, states[i])[0],
                    "fn": fn.strip().rstrip("(").strip().split("::")[-1],
                    "parse": kind, "default_state": dstate, "code_default": dflt,
                })
    return reads, sets


# ---------------------------------------------------------------- launch code

_SET_PATTERNS = [
    re.compile(r"""["'`]NAME["'`]\s*:\s*(?P<v>[^,}\n]+)"""),                      # {"NAME": "1"}
    re.compile(r"""\(\s*["']NAME["']\s*,\s*(?P<v>[^)\n]+)\)"""),                  # ("NAME", "1")
    re.compile(r"""\[\s*["']NAME["']\s*\]\s*=(?!=)\s*(?P<v>[^\n]+)"""),           # env["NAME"] = x
    re.compile(r"""setdefault\(\s*["']NAME["']\s*,\s*(?P<v>[^)\n]+)\)"""),
    re.compile(r"""\.NAME\s*=(?!=)\s*(?P<v>[^\n;]+)"""),                           # process.env.NAME = x
    re.compile(r"""(?<![A-Z0-9_])NAME=(?P<v>[^\s"'`,;)\]]+)"""),                   # NAME=1 in a string / command
    re.compile(r"""["'`]NAME["'`]\s*,\s*(?P<v>["'`][^"'`\n]*["'`])"""),            # ["NAME", "1"] / env.set("NAME","1")
]


def _is_test(rel: str) -> bool:
    b = os.path.basename(rel)
    return (b.startswith("test_") or b.endswith(("_test.py", ".test.mjs", ".test.js", ".spec.mjs")) or b == "terminal/conftest.py"
            or any(s in "/" + rel for s in SKIP_PARTS))


def scan_launch(names):
    sets = collections.defaultdict(list)
    reads = collections.defaultdict(list)
    named = collections.defaultdict(set)
    for p in _files(LAUNCH_PATHS, LAUNCH_EXT):
        rel = _rel(p)
        if _is_test(rel):
            continue
        lines = _read_lines(p)
        in_doc = False
        for i, l in enumerate(lines):
            was_doc = in_doc
            if p.suffix == ".py" and l.count('"""') % 2 == 1:
                in_doc = not in_doc
            if "STRATA_" not in l:
                continue
            stripped = l.strip()
            is_comment = stripped.startswith(("#", "//", "*", "/*")) or was_doc or (in_doc and not was_doc and '"""' in l and False)
            for nm in set(re.findall(r"STRATA_[A-Z0-9_]+(?![A-Z0-9_])", l)):
                if nm not in names:
                    continue
                named[nm].add(rel)
                got = None
                if not is_comment:
                    for pat in _SET_PATTERNS:
                        m = re.search(pat.pattern.replace("NAME", re.escape(nm)), l)
                        if m:
                            got = m.group("v").strip().rstrip(",;").strip()
                            break
                if got is not None and (got.startswith(("\"STRATA_", "'STRATA_")) or "\\d" in got or got.endswith(":") or got in ('"reason"', '"when"', '"value"')):
                    got = None
                if got is not None and not re.search(r"environ\.get|getenv|\.get\(", l[: l.find(nm)]):
                    sets[nm].append({"file": rel, "line": i + 1, "value": got[:80]})
                elif not is_comment and re.search(r"environ\.get|environ\[|getenv|process\.env|env\.get|env\[", l):
                    reads[nm].append({"file": rel, "line": i + 1})
    return sets, reads, named


# ---------------------------------------------------------------- engine default tables

DEFAULT_TABLE_FILES = ("engine/src/core/arch_defaults.cpp", "engine/src/core/nv_defaults.cpp", "src/core/amd_community.cpp", "src/core/gfx12_extra.cpp",
                       "src/core/defaults_026.cpp")


_ARCH_LABELS = (
    (re.compile(r"gfx_arch_is_gfx12\b|gfx12\b"), "gfx12"),
    (re.compile(r'gfx_arch_is_gfx1151|"gfx1151"'), "gfx1151"),
    (re.compile(r'"gfx1103"|"gfx1150"'), "gfx1103/gfx1150"),
    (re.compile(r"mj\s*==\s*8\s*&&\s*mn\s*==\s*6"), "sm_86"),
    (re.compile(r"mj\s*==\s*6\s*&&\s*mn\s*==\s*0"), "sm_60"),
    (re.compile(r"mj\s*==\s*12\s*&&\s*mn\s*==\s*1"), "sm_121"),
    (re.compile(r"nv_kq_default_env|cc_major_of\(min_major\)"), "NVIDIA cc>=8"),
)


def scan_engine_defaults(names):
    """Entries of the engine's own default tables: {"STRATA_X", "v"} / push_back({"STRATA_X", "v"}). The engine applies them for a GPU
    architecture when the person has not set the name. "arch" is the nearest architecture test above the entry (a label, to be checked
    by eye; "windows" is added when a Windows-only test sits between that test and the entry)."""
    out = collections.defaultdict(list)
    for rel in DEFAULT_TABLE_FILES:
        p = ROOT / rel
        lines = _read_lines(p)
        for i, l in enumerate(lines):
            for m in re.finditer(r'\{\s*"(STRATA_[A-Z0-9_]+)"\s*,\s*"([^"]*)"\s*\}', l):
                if m.group(1) not in names:
                    continue
                label, win = "", False
                for k in range(i, max(-1, i - 80), -1):
                    if re.search(r"\bwin\s*&&|\bwindows\b\s*(?:&&|\))|!windows", lines[k]) and not label:
                        win = True
                    hit = next((lab for rx, lab in _ARCH_LABELS if rx.search(lines[k])), None)
                    if hit:
                        label = hit
                        break
                    if lines[k].startswith(("std::vector", "bool ", "void ")) and k != i:
                        break
                out[m.group(1)].append({"file": rel, "line": i + 1, "value": m.group(2), "arch": (label + (" (Windows)" if win and label else "")) or "?"})
    return out


# ---------------------------------------------------------------- docs

def scan_docs(names):
    out = collections.defaultdict(list)
    for p in _files(DOC_PATHS, (".md",)):
        rel = _rel(p)
        if rel == SETTINGS_PAGE:   # the generated page lists the names itself; it is not a source of documentation
            continue
        for i, l in enumerate(_read_lines(p)):
            if "STRATA_" not in l:
                continue
            for nm in set(re.findall(r"STRATA_[A-Z0-9_]+(?![A-Z0-9_])", l)):
                if nm in names:
                    out[nm].append(f"{rel}:{i + 1}")
    return out


# ---------------------------------------------------------------- git history

def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def _tag_of_commits():
    """{commit: first release tag that contains it}, from the release tags in date order."""
    tags = [t for t in _git("tag", "--sort=creatordate").split() if re.fullmatch(r"v\d+\.\d+\.\d+(\.\d+)?", t)]
    seen = set()
    first = {}
    for t in tags:
        revs = _git("rev-list", t).split()
        for c in revs:
            if c not in seen:
                seen.add(c)
                first[c] = t
    return first


def scan_git(names):
    """first and last commit that adds or removes a line holding the name, in the engine directories."""
    out = _git("log", "--reverse", "--no-color", "-U0", "--format=@@C %H %cs", "-G", "STRATA_[A-Z0-9_]+", "-p", "--", *ENGINE_DIRS)
    tagmap = _tag_of_commits()
    first, last = {}, {}
    cur = None
    for line in out.split("\n"):
        if line.startswith("@@C "):
            _, h, d = line.split(" ", 2)
            cur = (h, d)
            continue
        if cur is None or not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        for nm in set(NAME_RE.findall(line)):
            if nm in names:
                if line[0] == "+" and nm not in first:
                    first[nm] = cur
                last[nm] = cur
    res = {}
    for nm in names:
        f, l = first.get(nm), last.get(nm)
        res[nm] = {
            "first_commit": f[0][:10] if f else None, "first_date": f[1] if f else None, "since": tagmap.get(f[0]) if f else None,
            "last_commit": l[0][:10] if l else None, "last_date": l[1] if l else None,
        }
    return res



# ---------------------------------------------------------------- the settings page (docs/SETTINGS.md)

_BACKEND_WORDS = {"cuda-hip": "NVIDIA and AMD", "cuda": "NVIDIA", "hip": "AMD", "sycl": "Intel Arc", "metal": "Mac", "cpu": "CPU"}


def _where(e):
    have = set(e["backends"])
    if "cuda-hip" in have:
        have -= {"cuda", "hip"}
    words = [_BACKEND_WORDS[b] for b in ("cuda-hip", "cuda", "hip", "sycl", "metal", "cpu") if b in have]
    return ", ".join(words)


def _kind(e):
    return {"flag": "on / off", "int": "number", "float": "number", "string": "text", "debug-gated": "on / off"}.get(e["parse"], e["parse"])


def _doc_links(e, limit=3):
    seen, out = set(), []
    for m in e["doc_mentions"]:
        f = m.rsplit(":", 1)[0]
        if f in seen or f.startswith(("docs/work", "docs/paper")):
            continue
        seen.add(f)
        out.append(f"[{os.path.basename(f)}]({os.path.relpath(f, 'docs').replace(os.sep, '/') if f.startswith('docs/') else '../' + f})")
        if len(out) == limit:
            break
    return ", ".join(out) or "-"


def render_page(entries):
    optin = [e for e in entries if e.get("status") == "opt-in"]
    roll = [e for e in entries if e.get("status") == "rollback"]
    out = [
        "# Engine settings",
        "",
        "<!-- Generated from terminal/data/settings-register.json by tools/settings_register.py --docs. Do not edit by hand; change the register. -->",
        "",
        "The engine reads its settings from environment variables whose names start with `STRATA_`. The Terminal and the app choose what",
        "your hardware needs, so most people never set one. This page lists the ones meant for people: switches that are off until you",
        "turn them on, and switches that take a newer default back to the earlier behaviour. The other settings are for measuring and",
        "debugging and are kept in `terminal/data/settings-register.json`.",
        "",
        "A number or a word is set like any environment variable, for example `STRATA_KVA=1`. In the Terminal, `strata launch --set NAME=value`",
        "puts it in the model's run configuration.",
        "",
        f"## Switches you can turn on ({len(optin)})",
        "",
        "Off unless you set them. The last column says where each one is described.",
        "",
        "| Setting | Value | Applies to | Described in |",
        "|---|---|---|---|",
    ]
    for e in optin:
        out.append(f"| `{e['name']}` | {_kind(e)} | {_where(e)} | {_doc_links(e)} |")
    out += [
        "",
        f"## Switches that go back to earlier behaviour ({len(roll)})",
        "",
        "These behaviours are on by default. Setting the switch returns the earlier behaviour; the notes in the register say which",
        "version made it the default.",
        "",
        "| Setting | Value | Applies to | Named in the engine since | Described in |",
        "|---|---|---|---|---|",
    ]
    for e in roll:
        out.append(f"| `{e['name']}` | {_kind(e)} | {_where(e)} | {e.get('since') or '-'} | {_doc_links(e)} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- build

def engine_names() -> set[str]:
    reads, _ = scan_engine()
    return set(reads)


def _merge_parse(sites):
    kinds = [s["parse"] for s in sites]
    kind = next((k for k in ("int", "float", "flag", "string", "debug-gated") if k in kinds), "string")
    if len(set(kinds)) > 1:
        kind = collections.Counter(kinds).most_common(1)[0][0]
    states = [s["default_state"] for s in sites if s["default_state"]]
    dstate = collections.Counter(states).most_common(1)[0][0] if states else None
    dflts = [s["code_default"] for s in sites if s["code_default"] and s["parse"] in ("int", "float")]
    dflt = collections.Counter(dflts).most_common(1)[0][0] if dflts else None
    return kind, dstate, dflt


def build(with_git=True):
    reads, esets = scan_engine()
    names = set(reads)
    lsets, lreads, lnamed = scan_launch(names)
    edefs = scan_engine_defaults(names)
    docs = scan_docs(names)
    git = scan_git(names) if with_git else {}
    old = {}
    if REGISTER.exists():
        try:
            old = {e["name"]: e for e in json.loads(REGISTER.read_text(encoding="utf-8"))["settings"]}
        except (OSError, ValueError, KeyError):
            old = {}
    entries = []
    for nm in sorted(names):
        sites = reads[nm]
        kind, dstate, dflt = _merge_parse(sites)
        backends = sorted({s["backend"] for s in sites})
        e = {
            "name": nm,
            "backends": backends,
            "reads": [{"file": s["file"], "line": s["line"], "backend": s["backend"], "fn": s["fn"]} for s in sites],
            "parse": kind,
            "default_state": dstate,
            "code_default": dflt,
            "engine_sets": esets.get(nm, []),
            "engine_defaults": edefs.get(nm, []),
            "launch_files": sorted(lnamed.get(nm, [])),
            "launch_sets": lsets.get(nm, []),
            "launch_reads": lreads.get(nm, []),
            "doc_mentions": docs.get(nm, []),
        }
        if with_git:
            e.update(git.get(nm, {}))
        elif nm in old:
            for k in ("first_commit", "first_date", "since", "last_commit", "last_date"):
                if k in old[nm]:
                    e[k] = old[nm][k]
        for k in HAND_FIELDS:
            e[k] = old.get(nm, {}).get(k, "")
        entries.append(e)
    # entries that the engine no longer reads but were retired on purpose stay (status "retired" with retired_in)
    for nm, o in sorted(old.items()):
        if nm not in names and o.get("status") == "retired" and o.get("retired_in"):
            entries.append(o)
    entries.sort(key=lambda e: e["name"])
    return {"version": 1, "scanned": list(ENGINE_DIRS), "launch": list(LAUNCH_PATHS), "statuses": list(STATUSES), "settings": entries}


def write(doc):
    REGISTER.parent.mkdir(parents=True, exist_ok=True)
    REGISTER.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--docs", action="store_true", help="write docs/SETTINGS.md from the register (opt-in and rollback entries)")
    a = ap.parse_args(argv)
    if a.names:
        print("\n".join(sorted(engine_names())))
        return 0
    if a.check:
        have = engine_names()
        entries = json.loads(REGISTER.read_text(encoding="utf-8"))["settings"]
        reg = {e["name"] for e in entries}
        retired = {e["name"] for e in entries if e.get("status") == "retired" and e.get("retired_in")}   # kept on purpose: not read any more
        missing, extra = sorted(have - reg), sorted(reg - have - retired)
        for n in missing:
            print("not in the register:", n)
        for n in extra:
            print("in the register, not read by the engine:", n)
        return 1 if (missing or extra) else 0
    if a.docs:
        entries = json.loads(REGISTER.read_text(encoding="utf-8"))["settings"]
        (ROOT / SETTINGS_PAGE).write_text(render_page(entries), encoding="utf-8", newline="\n")
        print(f"wrote {ROOT / SETTINGS_PAGE}")
        return 0
    doc = build(with_git=not a.no_git)
    write(doc)
    print(f"wrote {REGISTER} with {len(doc['settings'])} settings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
