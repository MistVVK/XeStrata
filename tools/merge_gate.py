#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The light checks right after a merge: what merge-tree and the tree comparison cannot see (a test whose premise the
other side broke, a member both sides declared).  The heavy gates (the whole build, CTest, the lints, parity) stay with
the dev head's final gate.

    python3 tools/merge_gate.py BASE [RESULT] [--build DIR]

Run from the root of a checkout at RESULT (default HEAD; the tracked files must be RESULT's).  BASE is the head before
the merge (RESULT's first parent).  Exits 1 when a check fails, 2 on wrong use.

1. The tools' unittest, whole: `<this python> -m unittest discover -s tools`.  A failure fails the gate, and so does an
   error, except one the environment causes: a ModuleNotFoundError for a module that is neither in this python nor a
   module of the repository at BASE or RESULT (regex and numpy on a python without them).  Those are counted apart.
   Run the gate with the data folder's venv python to have none.
2. The syntax of the files BASE..RESULT changed: .py by py_compile; .cpp by its command in DIR's compile_commands.json
   (default build/contrib-llvm) with -fsyntax-only; .hpp and .h inside a source that includes them (a changed one
   first, else the first in path order; through another header when no source includes it directly).  The commands'
   paths under the build's source folder are read as this checkout's.  A file the build does not compile, a header no
   source reaches, and every C++ file when DIR has no compile_commands.json are listed as unverified, not failed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import py_compile
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SEP = "=" * 70
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
MISSING = re.compile(r"^ModuleNotFoundError: No module named '([\w.]+)'$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def unittest_tools(repo_modules: set[str]) -> tuple[bool, str]:
    t0 = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tools"], env=env,
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    log = Path(tempfile.gettempdir()) / f"merge_gate-unittest-{os.getpid()}.log"
    log.write_text(out)
    ran = re.search(r"^Ran (\d+) tests? in ", out, re.M)
    if not ran:
        return False, f"unittest: no result line (rc {p.returncode}); output in {log}"
    fails: list[str] = []
    errs: list[str] = []
    env_errs: dict[str, list[str]] = {}
    for block in out.split(SEP + "\n")[1:]:
        head, _, body = block.partition("\n")
        body = body.split("\n" + "-" * 70 + "\nRan ")[0]
        last = body.strip().splitlines()[-1] if body.strip() else ""
        if head.startswith("FAIL: "):
            fails.append(head[6:])
        elif head.startswith("ERROR: "):
            m = MISSING.match(last)
            top = m.group(1).split(".")[0] if m else ""
            if m and top not in repo_modules and importlib.util.find_spec(top) is None:
                env_errs.setdefault(top, []).append(head[7:])
            else:
                errs.append(f"{head[7:]}: {last}")
    skipped = re.search(r"skipped=(\d+)", out.rsplit("\nRan ", 1)[-1])
    n_env = sum(len(v) for v in env_errs.values())
    lines = [f"unittest (tools, {sys.executable}): ran {ran.group(1)}, failures {len(fails)}, errors {len(errs)}, "
             f"environment errors {n_env}, skipped {skipped.group(1) if skipped else 0}, "
             f"{time.monotonic() - t0:.0f} s; output in {log}"]
    lines += [f"  FAIL {f}" for f in fails] + [f"  ERROR {e}" for e in errs]
    lines += [f"  environment: no module {k!r} ({len(v)}): {', '.join(v)}" for k, v in sorted(env_errs.items())]
    if p.returncode != 0 and not fails and not errs and not n_env:
        return False, "\n".join(lines + [f"  rc {p.returncode} with nothing listed"])
    return not fails and not errs, "\n".join(lines)


def py_syntax(files: list[str]) -> tuple[bool, str]:
    bad = []
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            try:
                py_compile.compile(f, cfile=os.path.join(d, "x.pyc"), doraise=True)
            except py_compile.PyCompileError as e:
                bad.append(f"  {e.msg.strip()}")
    return not bad, "\n".join([f"py_compile: {len(files)} files, {len(bad)} failed"] + bad)


def syntax_command(entry: dict, old: str, new: str, build: str) -> tuple[list[str], str]:
    """The entry's command with -fsyntax-only, its paths under the build's source folder read as this checkout's."""
    def move(t: str) -> str:
        i = t.find(old + "/")
        if i < 0 or t[i:].startswith(build + "/") or t[i:] == build:
            return t
        moved = new + t[i + len(old):]
        return t[:i] + moved if os.path.exists(moved) else t
    args = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    out, skip = [args[0], "-fsyntax-only"], False
    for t in args[1:]:
        if skip:
            skip = False
        elif t in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif t.startswith("-fsycl-targets="):
            out += sycl_targets(t)
        elif t not in ("-c", "-MD", "-MMD"):
            out.append(move(t))
    return out, entry["directory"]


def sycl_targets(arg: str) -> list[str]:
    """-fsycl-targets with its device aliases spelled as triples: with -fsyntax-only the driver takes nvidia_gpu_sm_89
    for an OpenMP target and stops ("OpenMP target is invalid"), so no SYCL source would be checked."""
    targets, extra = [], []
    for t in arg.split("=", 1)[1].split(","):
        if m := re.fullmatch(r"nvidia_gpu_(sm_\w+)", t):
            targets.append("nvptx64-nvidia-cuda")
            extra += ["-Xsycl-target-backend=nvptx64-nvidia-cuda", f"--cuda-gpu-arch={m.group(1)}"]
        elif m := re.fullmatch(r"amd_gpu_(gfx\w+)", t):
            targets.append("amdgcn-amd-amdhsa")
            extra += ["-Xsycl-target-backend=amdgcn-amd-amdhsa", f"--offload-arch={m.group(1)}"]
        else:
            targets.append(t)
    return ["-fsycl-targets=" + ",".join(targets), *extra]


def cpp_syntax(files: list[str], changed: set[str], build: Path) -> tuple[bool, str]:
    db = build / "compile_commands.json"
    if not db.is_file():
        return True, "\n".join([f"C++ syntax: unverified, {db} not found ({len(files)} files):"] +
                               [f"  {f}" for f in files])
    cache = (build / "CMakeCache.txt").read_text()
    home = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache, re.M)
    if not home:
        raise SystemExit(f"merge_gate: no CMAKE_HOME_DIRECTORY in {build}/CMakeCache.txt")
    old = home.group(1)
    new, b = os.getcwd(), str(build.resolve())
    tus: dict[str, dict] = {}
    for e in json.loads(db.read_text()):
        f = os.path.normpath(os.path.join(e["directory"], e["file"]))
        if f.startswith(old + "/"):
            tus.setdefault(f[len(old) + 1:], e)
    tracked = [f for f in git("ls-files").splitlines() if re.search(r"\.(cpp|cc|hpp|h|inc)$", f)]
    includers: dict[str, list[str]] = {}
    for f in tracked:
        try:
            text = Path(f).read_text(errors="replace")
        except OSError:
            continue
        for q, p in INCLUDE.findall(text):
            cands = ([os.path.normpath(os.path.join(os.path.dirname(f), p))] if q == '"' else []) + [f"include/{p}"]
            for c in cands:
                includers.setdefault(c, []).append(f)

    def source_of(h: str) -> str | None:   # breadth first up the includers, a changed source first at each level
        seen, level = {h}, [h]
        for _ in range(6):
            nxt = sorted({i for x in level for i in includers.get(x, []) if i not in seen})
            found = [i for i in nxt if i in tus]
            if found:
                return next((i for i in found if i in changed), found[0])
            seen.update(nxt)
            level = nxt
        return None

    jobs: dict[str, list[str]] = {}   # source -> the changed files it checks
    unverified = []
    for f in files:
        src = f if f in tus else (None if f.endswith((".cpp", ".cc")) else source_of(f))
        if src is None:
            unverified.append(f"  {f}: " + ("not compiled in this build" if f.endswith(".cpp")
                                            else "no source of this build includes it"))
        else:
            jobs.setdefault(src, []).append(f)

    def check(src: str) -> tuple[str, int | None, str]:
        cmd, cwd = syntax_command(tus[src], old, new, b)
        try:
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        except FileNotFoundError as e:   # the build's compiler or folder is gone
            return src, None, str(e)
        return src, p.returncode, p.stdout + p.stderr

    t0, bad, lines = time.monotonic(), 0, []
    with concurrent.futures.ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 2) // 2))) as ex:
        for src, rc, out in ex.map(check, sorted(jobs)):
            via = "" if jobs[src] == [src] else f" (checks {', '.join(f for f in jobs[src] if f != src)})"
            if rc is None:
                unverified += [f"  {f}: {out}" for f in jobs[src]]
                continue
            lines.append(f"  {'ok  ' if rc == 0 else 'FAIL'} {src}{via}")
            if rc != 0:
                bad += 1
                errors = [x for x in out.splitlines() if ": error: " in x or x.startswith("error: ")]
                lines += ["    " + x for x in (errors or out.splitlines()[:5])[:20]]
    head = (f"C++ syntax ({build}): {len(files) - len(unverified)} files checked in {len(jobs)} sources, "
            f"{bad} sources failed, {len(unverified)} unverified, {time.monotonic() - t0:.0f} s")
    return bad == 0, "\n".join([head] + lines + (["  unverified:"] + unverified if unverified else []))


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("result", nargs="?", default="HEAD")
    ap.add_argument("--build", default="build/contrib-llvm", help="the build folder with compile_commands.json")
    a = ap.parse_args()
    os.chdir(git("rev-parse", "--show-toplevel").strip())
    base, result = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (a.base, a.result))
    if result != git("rev-parse", "HEAD").strip():
        print(f"merge_gate: the checkout is not at {a.result}; check it out first", file=sys.stderr)
        return 2
    if git("status", "--porcelain", "--untracked-files=no"):
        print("merge_gate: tracked files differ from HEAD", file=sys.stderr)
        return 2
    changed = git("diff", "--name-only", "--diff-filter=d", base, result).splitlines()
    print(f"merge_gate: {base[:10]}..{result[:10]}, {len(changed)} files changed", flush=True)
    repo_modules = {part.removesuffix(".py") for r in (base, result)
                    for f in git("ls-tree", "-r", "--name-only", r).splitlines() if f.endswith(".py")
                    for part in Path(f).parts}
    results = [py_syntax([f for f in changed if f.endswith(".py")]),
               cpp_syntax([f for f in changed if f.endswith((".cpp", ".hpp", ".h"))], set(changed),
                          Path(a.build).resolve())]
    for _, text in results:
        print(text, flush=True)
    results.append(unittest_tools(repo_modules))
    print(results[-1][1])
    ok = all(r[0] for r in results)
    print(f"merge_gate: {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
