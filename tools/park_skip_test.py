#!/usr/bin/env python3
"""The rewind keep (#5862) on a layer split with batch slots: the next turns of three conversations give the same tokens
whether the engine skips the park (--conversation-cache-rewind-keep N) or parks as before.

One engine start (the caller runs it twice, with and without the switch, and compares the dumps):
  t1  BGEN slot 0: conversation A, a long prompt                       (the slot holds A and its reply)
  t2  BGEN slot 1: conversation B                                      (the main session holds a conversation too)
  t3  BGEN slot 0: A's next turn, as a client that strips the reply renders it: A's prompt again plus a new question
  t4  BGEN slot 1: B's next turn the same way
  t5  GEN (solo): A's next turn again with another question
  t6  GEN (solo): B's next turn
  t7  GEN (solo): conversation C, a long prompt                        (the main session holds C and its reply)
  t8  GEN (solo): C's next turn the same way: its prompt again plus a new question; the reply is a private suffix of the
      live state, which --conversation-cache-rewind-keep leaves unparked ("park not needed (the rewind keeps ...")
Every turn is greedy; the dump holds the tokens of t1..t8.  The engine log lines about parks, skips and restores are printed.
A park only reads, so skipping one must not change a token; the unparked state is read again where it is needed.

  python3 tools/park_skip_test.py --cfg <json with args/env/lib_path> --exe <strata> --tok <tokenizer dir> --dump out.json \
      --log engine.log [--extra "--batch-groups 2 --conversation-cache-rewind-keep 4096"]
The cfg's args are the engine's own (model, pack, profile, split ...); --batch and --extra are added.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--exe", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--extra", default="")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--pad", type=int, default=2000, help="tokens of filler text in each conversation's first prompt")
ap.add_argument("--max-new", type=int, default=64)
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

cfg = json.load(open(a.cfg))
os.environ.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
for kv in a.env:
    k, _, v = kv.partition("=")
    os.environ[k] = v
os.environ["BATCH_TEST_LOG"] = a.log
tok = BT.tokenizer(a.tok)
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def first_prompt(qi):
    para = ("Note %d-%d: the survey team measured the northern ridge, logged the readings of station %d, and sent the "
            "figures to the office, where a clerk filed the sheets under reference %d. ")
    per = max(1, len(tok.encode(para % (qi, 100, 100, 100))))
    filler = "".join(para % (qi, i, (i * 7 + qi) % 97, i * 13 + qi) for i in range(a.pad // per + 1))
    return tok.encode(f"<|im_start|>user\n{filler}\n\n{BT.QUESTIONS[qi]}<|im_end|>\n{ASSISTANT}", parse_special=True)


def more(text):
    return tok.encode(f"{text}<|im_end|>\n{ASSISTANT}", parse_special=True)


eng = BT.Engine(a.exe, {"args": list(cfg["args"]), "lib_dirs": list(cfg.get("lib_path") or []), "cwd": None}, a.batch,
                {"STRATA_IQ_MT_MIN": "1"}, a.extra.split())
out = eng.lines()


def gen(ids):
    eng.send(f"GEN {a.max_new} {','.join(map(str, ids))}")
    got = []
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return got
        elif line.startswith("ERR"):
            raise SystemExit("engine: " + line)
    raise SystemExit(f"the engine ended during a GEN (see {a.log})")


def bgen(slot, ids):
    eng.send(f"BGEN {slot} {a.max_new} {','.join(map(str, ids))}")
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
        raise SystemExit(f"the engine ended during a BGEN (see {a.log})")
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
    raise SystemExit(f"the engine ended during a BGEN (see {a.log})")


pa, pb, pc = first_prompt(0), first_prompt(1), first_prompt(2)
turns = []
try:
    turns.append(("t1 slot0 A1", bgen(0, pa)))
    turns.append(("t2 slot1 B1", bgen(1, pb)))
    pa2 = pa + more("Thank you. One more question: what is the capital of France?")
    pb2 = pb + more("Thank you. One more question: what is the capital of Italy?")
    turns.append(("t3 slot0 A2", bgen(0, pa2)))
    turns.append(("t4 slot1 B2", bgen(1, pb2)))
    turns.append(("t5 solo A3", gen(pa + more("Thanks again. Please answer in one short sentence: who wrote Hamlet?"))))
    turns.append(("t6 solo B3", gen(pb + more("Thanks again. Please answer in one short sentence: who painted the Mona Lisa?"))))
    turns.append(("t7 solo C1", gen(pc)))
    turns.append(("t8 solo C2", gen(pc + more("Thank you. One more question: what is the capital of Spain?"))))
finally:
    eng.send("QUIT")
    try:
        eng.p.wait(timeout=180)
    except Exception:
        eng.p.kill()
Path(a.dump).write_text(json.dumps({n: t for n, t in turns}), encoding="utf-8")
for n, t in turns:
    print(f"{n}: {len(t)} tokens  {ascii(tok.decode(t)[:70])}")
text = Path(a.log).read_text(errors="replace")
pat = re.compile(r"conversation cache: (parked|restored|skipped|park not needed|skip parking|slot \d+ not parked)[^\n]*")
lines = [m.group(0)[:200] for m in pat.finditer(text)]
print(f"log lines about parks: {len(lines)}; park not needed: {sum('park not needed' in l for l in lines)}; "
      f"parked: {sum('parked' in l and 'not parked' not in l for l in lines)}; restored: {sum('restored' in l for l in lines)}")
for l in lines[:24]:
    print("  ", l)
