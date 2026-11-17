#!/usr/bin/env python3
"""A conversation a batch slot held, continued after the slot was given to another request (#5401).

  1. engine 1 (--batch 2): A decodes to its end in slot 0 (BGEN); then B is admitted into slot 0, which overwrites what the
     slot held (before that, the engine parks A's conversation from the slot, or logs why it does not); then A's next turn
     (A, A's reply, a new question) is sent solo (GEN): it continues from whatever the conversation cache holds of A
  2. engine 2 (fresh, --batch 2): A solo, then the same next turn solo (it continues from the live session)
A's slot reply must equal its solo reply, and the next turn must be token for token the same in both engines: a parked
copy that does not hold the slot's real state (#5401: on Metal a slot's state lives in the decode window) would give
another answer if it were restored.  The engine log lines about the slot's park and the next turn's restore are printed.

Exact comparisons need the same settings as batch_test.py (this script sets STRATA_IQ_MT_MIN=1):
  python tools/batch_slot_park_test.py --exe build/strata --config strata-<model>.json --extra "--pcie-frac 0"
The engine logs go to $BATCH_TEST_LOG (default: the temp folder) with .1 / .2 for the two engines.
"""
import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402


def gen(eng, out, ids, n):
    eng.send(f"GEN {n} {','.join(map(str, ids))}")
    got = []
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return got
        elif line.startswith("ERR"):
            raise SystemExit("engine: " + line)
    raise SystemExit(f"the engine ended during a GEN (see {eng.log_path})")


def bgen(eng, out, slot, ids, n):
    """BGEN into `slot` and decode it to its end; returns its tokens."""
    eng.send(f"BGEN {slot} {n} {','.join(map(str, ids))}")
    got, cont = [], None
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("BADM "):
            cont = line.split()[2] == "1"
            break
        elif line.startswith("ERR"):
            raise SystemExit("engine: " + line)
    if cont is None:
        raise SystemExit(f"the engine ended during a BGEN (see {eng.log_path})")
    if not cont:
        return got
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            if int(s) == slot:
                got.append(int(y))
        elif line.startswith("BDONE ") and int(line.split()[1]) == slot:
            return got
        elif line.startswith("ERR"):
            raise SystemExit("engine: " + line)
    raise SystemExit(f"the engine ended during slot {slot}'s decode (see {eng.log_path})")


def cache_lines(path):
    keys = ("parked", "restored", "skip parking", "not parked", "discard", "borrowed")
    return [l for l in Path(path).read_text(errors="replace").splitlines()
            if "conversation cache:" in l and any(k in l for k in keys)]


def first_diff(x, y):
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    return i if i is not None else min(len(x), len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    A = chat("Write a Python function that merges overlapping intervals, then explain it.")
    B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    Q2 = chat("Now do the same in Rust.")
    M = a.max_new
    base_log = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
    env = {"STRATA_IQ_MT_MIN": "1"}

    # 1. the slot path
    os.environ["BATCH_TEST_LOG"] = base_log + ".1"
    e1 = Engine(a.exe, cfg, 2, env, a.extra.split())
    o1 = e1.lines()
    sA = bgen(e1, o1, 0, A, M)
    sB = bgen(e1, o1, 0, B, M)
    A2 = A + sA + Q2
    t = gen(e1, o1, A2, M)
    e1.send("QUIT")
    e1.p.wait(timeout=120)
    print(f"engine 1: A in slot 0 {len(sA)} tokens, B in slot 0 {len(sB)} tokens, A's next turn ({len(A2)} prompt tokens) "
          f"solo after the overwrite: {len(t)} tokens", flush=True)
    for line in cache_lines(base_log + ".1"):
        print("  log 1: " + line[:220], flush=True)

    # 2. a fresh engine, solo only
    os.environ["BATCH_TEST_LOG"] = base_log + ".2"
    e2 = Engine(a.exe, cfg, 2, env, a.extra.split())
    o2 = e2.lines()
    rA = gen(e2, o2, A, M)
    r = gen(e2, o2, A2, M)
    e2.send("QUIT")
    e2.p.wait(timeout=120)
    os.environ["BATCH_TEST_LOG"] = base_log

    ok_a, ok_t = sA == rA, t == r
    print(f"A in slot 0: {len(sA)} tokens, solo {len(rA)}: {'IDENTICAL' if ok_a else f'DIFFERS at {first_diff(sA, rA)}'}",
          flush=True)
    print(f"A's next turn after its slot was overwritten: {len(t)} tokens, fresh solo {len(r)}: "
          f"{'IDENTICAL' if ok_t else f'DIFFERS at {first_diff(t, r)}'}", flush=True)
    if not ok_t:
        print("  after the overwrite: " + repr(tok.decode(t)[:200]), flush=True)
        print("  fresh solo:          " + repr(tok.decode(r)[:200]), flush=True)
    print("PASS" if ok_a and ok_t else "FAIL", flush=True)
    sys.exit(0 if ok_a and ok_t else 1)


if __name__ == "__main__":
    main()
