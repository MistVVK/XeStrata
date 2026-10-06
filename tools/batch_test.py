#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe build/xe/strata --config xestrata-<model>.json --batch 4 --n 4 \
      --extra "--pcie-frac 0 --adapt-every 1000000"
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
]


def tokenizer(path):
    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args):
        args = list(cfg["args"])
        if batch:
            args += ["--batch", str(batch)]
        args += extra_args
        env = dict(os.environ, **extra_env)
        env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise SystemExit(f"the engine ended before READY - see {self.log_path}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for q in QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    env = {"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})}
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = Engine(a.exe, cfg, a.batch, env, a.extra.split())
    out = eng.lines()

    solo = []
    if not a.skip_solo:
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
            got, t0 = [], time.time()
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                    break
            solo.append(got)

    # batch: admit every prompt, then the slots decode together
    got = {i: [] for i in range(len(prompts))}
    done = {}
    t_admit = time.time()
    first_bt = None
    for i, ids in enumerate(prompts):
        eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
        for line in out:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                first_bt = first_bt or time.time()
            elif line.startswith("BDONE "):
                done[int(line.split()[1])] = line
            elif line.startswith("ERR"):
                print("batch:", line)
                return 1
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_admitted = time.time()
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            first_bt = first_bt or time.time()
        elif line.startswith("BDONE "):
            done[int(line.split()[1])] = line
            if len(done) == len(prompts):
                break
        elif line.startswith("ERR"):
            print("batch:", line)
            return 1
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    print(f"batch: {len(prompts)} slots, {total} tokens; admissions {t_admitted - t_admit:.1f}s; "
          f"then {t_end - t_admitted:.1f}s -> aggregate {total / max(t_end - t_admit, 1e-9):.1f} tok/s overall, "
          f"{total / max(t_end - (first_bt or t_admit), 1e-9):.1f} tok/s from the first batch token",
          flush=True)
    ok = True
    for i in range(len(prompts)):
        b = got[i]
        if solo:
            same = b == solo[i]
            ok &= same
            first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]), None)
            verdict = "IDENTICAL" if same else f"DIFFERS at {first_diff}"
            print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {verdict}")
        print("   ", ascii(tok.decode(b)[:160]))
    eng.send("QUIT")
    eng.p.wait(timeout=180)   # the next run needs the GPUs back
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
