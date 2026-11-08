#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task scores of a quant, compared question by question with another run's (tools/quant_quality.py's companion):
GSM8K (grade-school math, 1,319 questions) and HumanEval (164 Python functions run against their tests), greedy,
through `xestrata-engine --serve`.

    python tools/task_check.py fetch --out DIR
        the two data files, pinned by commit and SHA-256 (openai/grade-school-math, openai/human-eval; MIT)
    python tools/task_check.py run --config CFG.json --tokenizer DIR --data DIR --out RUN.json
        [--tasks gsm8k,humaneval] [--every N] [--think] [--max-new N] [--env K=V ...]
        one engine (the config's exe, args, env, lib_dirs and cwd, as tools/quant_quality.py's), every question one
        request; RUN.json holds, per question, whether it passed and the answer.  Thinking is off (an empty think
        block) unless --think; a reply cut at --max-new counts as wrong.  HumanEval's answers are run as Python in a
        subprocess of their own (a temporary folder, a bare environment, 15 s): the model's code, on this machine.
    python tools/task_check.py paired A.json B.json
        per task and overall: both scores, the difference B - A with its 95% interval for paired proportions
        (Newcombe's method 10, from the Wilson intervals) and the exact McNemar test on the questions only one passed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DATA_FILES = {
    "gsm8k-test.jsonl": ("https://raw.githubusercontent.com/openai/grade-school-math/"
                         "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
                         "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
    "HumanEval.jsonl.gz": ("https://raw.githubusercontent.com/openai/human-eval/"
                           "b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz",
                           "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"),
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch(a) -> int:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, want) in DATA_FILES.items():
        p = out / name
        if not p.exists() or sha256(p) != want:
            tmp = p.with_suffix(p.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            got = sha256(tmp)
            if got != want:
                tmp.unlink()
                raise SystemExit(f"{name}: SHA-256 {got}, expected {want}")
            tmp.replace(p)
        print(f"{name}: {want}")
    return 0


# ------------------------------------------------------------------------------------------------ run
def load_data(d: pathlib.Path) -> None:
    for name, (_, want) in DATA_FILES.items():
        if not (d / name).exists() or sha256(d / name) != want:
            raise SystemExit(f"{d / name}: missing or not the pinned file (task_check.py fetch --out {d})")


def gsm8k_tasks(d: pathlib.Path) -> list[dict]:
    out = []
    for i, line in enumerate(open(d / "gsm8k-test.jsonl", encoding="utf-8")):
        q = json.loads(line)
        ref = q["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        out.append({"id": f"gsm8k/{i}", "kind": "gsm8k", "ref": ref,
                    "prompt": q["question"] + "\n\nSolve the problem step by step, and end with a line "
                                              "'Answer: <number>'."})
    return out


def humaneval_tasks(d: pathlib.Path) -> list[dict]:
    out = []
    for line in gzip.open(d / "HumanEval.jsonl.gz", "rt", encoding="utf-8"):
        p = json.loads(line)
        out.append({"id": p["task_id"], "kind": "humaneval", "ref": p,
                    "prompt": "Complete the following Python function. Reply with the whole function in one "
                              "```python code block.\n\n```python\n" + p["prompt"] + "```"})
    return out


def number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_ok(answer: str, ref: str) -> bool:
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", answer)
    got = number(m[-1]) if m else None
    want = number(ref)
    return got is not None and want is not None and abs(got - want) < 1e-6


def humaneval_ok(answer: str, p: dict) -> bool:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", answer, re.S)
    code = blocks[0] if blocks else answer
    prog = code if f"def {p['entry_point']}" in code else p["prompt"] + code
    prog += "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "t.py"
        f.write_text(prog, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=15,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def chat(q: str, think: bool) -> str:
    head = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return head if think else head + "\n</think>\n\n"


def engine_args(cfg: dict, context: int) -> list[str]:
    base, out, i = list(cfg["args"]), [], 0
    while i < len(base):
        if base[i] == "--serve":
            i += 1
            continue
        if base[i] == "--max-context":
            i += 2
            continue
        out.append(base[i])
        i += 1
    return [cfg["exe"], "--serve", *out, "--max-context", str(context)]


def cmd_run(a) -> int:
    sys.path.insert(0, str(HERE))
    import bench_core as B  # noqa: E402
    data = pathlib.Path(a.data)
    load_data(data)
    tasks: list[dict] = []
    for t in a.tasks.split(","):
        tasks += {"gsm8k": gsm8k_tasks, "humaneval": humaneval_tasks}[t](data)[::a.every]
    max_new = a.max_new or (16384 if a.think else 1024)
    tok = B.load_tokenizer(pathlib.Path(a.tokenizer))
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = engine_args(cfg, max_new + 2048)
    print("engine:", " ".join(args), flush=True)
    errf = open(str(out) + ".stderr.txt", "w")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    assert p.stdin is not None and p.stdout is not None
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")
    results: dict = {}
    meta = {"config": a.config, "args": args, "env": a.env or [], "think": a.think, "max_new": max_new,
            "every": a.every}
    for n, t in enumerate(tasks):
        ids = tok.encode(chat(t["prompt"], a.think), parse_special=True)
        p.stdin.write(f"GEN {max_new} {','.join(map(str, ids))}\n")
        p.stdin.flush()
        gen: list[int] = []
        done = None
        for line in p.stdout:
            if line.startswith("T "):
                gen.append(int(line.split()[1]))
            elif line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE"):
            p.kill()
            raise SystemExit(f"{t['id']}: the engine answered {done!r}; see {out}.stderr.txt")
        text = tok.decode(gen).replace("<|im_end|>", "")
        cut = len(gen) >= max_new
        answer = text.rsplit("</think>", 1)[-1] if a.think else text
        ok = not cut and (gsm8k_ok(answer, t["ref"]) if t["kind"] == "gsm8k" else humaneval_ok(answer, t["ref"]))
        results[t["id"]] = {"kind": t["kind"], "ok": ok, "cut": cut, "tokens": len(gen), "answer": answer[-2000:]}
        if (n + 1) % 20 == 0 or n + 1 == len(tasks):
            out.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1))
            passed = sum(r["ok"] for r in results.values())
            print(f"{n + 1}/{len(tasks)}: {passed} passed", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    return 0


# ------------------------------------------------------------------------------------------------ paired
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = z * z / n
    mid = (p + c / 2) / (1 + c)
    half = z * math.sqrt(p * (1 - p) / n + c / (4 * n)) / (1 + c)
    return mid - half, mid + half


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """B - A for paired outcomes (a: both passed, b: A only, c: B only, d: neither): the difference and its 95% interval
    (Newcombe 1998, method 10)."""
    n = a + b + c + d
    pa, pb = (a + b) / n, (a + c) / n
    la, ua = wilson(a + b, n)
    lb, ub = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    diff = pb - pa
    lo = diff - math.sqrt(max((pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2, 0.0))
    hi = diff + math.sqrt(max((ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2, 0.0))
    return diff, lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the discordant questions as Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cmd_paired(a) -> int:
    ra = json.load(open(a.a))["results"]
    rb = json.load(open(a.b))["results"]
    ids = sorted(set(ra) & set(rb))
    if not ids:
        raise SystemExit("no question in both runs")
    kinds = sorted({ra[i]["kind"] for i in ids})
    print(f"A: {a.a}\nB: {a.b}")
    print(f"{'task':10s} {'n':>5s} {'A':>7s} {'B':>7s} {'B - A':>7s} {'95% interval':>17s} {'A only':>6s} "
          f"{'B only':>6s} {'McNemar p':>9s}")
    for kind in kinds + (["all"] if len(kinds) > 1 else []):
        sel = [i for i in ids if kind == "all" or ra[i]["kind"] == kind]
        x = [(ra[i]["ok"], rb[i]["ok"]) for i in sel]
        both = sum(p and q for p, q in x)
        a_only = sum(p and not q for p, q in x)
        b_only = sum(q and not p for p, q in x)
        neither = len(x) - both - a_only - b_only
        diff, lo, hi = newcombe_paired(both, a_only, b_only, neither)
        n = len(x)
        print(f"{kind:10s} {n:5d} {100 * (both + a_only) / n:6.1f}% {100 * (both + b_only) / n:6.1f}% "
              f"{100 * diff:+6.1f} [{100 * lo:+6.1f}, {100 * hi:+6.1f}] {a_only:6d} {b_only:6d} "
              f"{mcnemar_exact(a_only, b_only):9.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="gsm8k,humaneval")
    p.add_argument("--every", type=int, default=1, help="every N-th question of each task (a quick check)")
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-new", type=int, default=0, help="the reply's cap (default 1024, 16384 with --think)")
    p.add_argument("--env", action="append")
    p = sub.add_parser("paired")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    return {"fetch": cmd_fetch, "run": cmd_run, "paired": cmd_paired}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
