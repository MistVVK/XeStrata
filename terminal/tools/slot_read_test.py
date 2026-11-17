#!/usr/bin/env python3
"""Slot read test (STRATA_SLOT_READ): a conversation's next turn read in the batch slot that holds it gives the
tokens the same turn gives when it is admitted through the main sessions.

Two conversations in two slots, three turns each, run twice on a fresh engine:
  ref    every turn admitted the usual way (BGEN: read on the main sessions, copied into the slot)
  slot   turns 2 and 3 with slot_read=1 (the slot reads the new tokens in its own windows)
Both runs read every prompt through the verify windows (--short-read: the batched read's kernels round
differently) with the expert placement fixed (--pcie-frac 0 --adapt-every 1000000), so the rows are the same rows.

Conversation 0's later turns are rendered as a chat client renders them - the history again, the reply without the
template's thinking block - so they leave what the slot holds at the last turn boundary: the read restarts at the
slot's turn checkpoint (turn 3's is the one the slot's own read of turn 2 took).  Conversation 1's continue exactly
what its slot holds.  The two turns of a round are sent back to back, so the second arrives while the first is read.

The engine needs the pipelined groups with drafts (a layer split, --batch-mtp, STRATA_PIPED_MTP and
STRATA_BATCH_SELF_COMMIT=0 in the config's env or the shell); the test says so when it does not have them.

  python3 tools/slot_read_test.py --exe engine/strata --config strata-<model>.json \
      [--pad 2000] [--max-new 120] [--keys "temperature=0.7 seed=5 penalty_last_n=64 penalty_present=1.5"]

Exit 0: every later turn identical in both runs, each of them read in its slot in the slot run; 2: a turn differs
or was not read in its slot; 1: an engine error.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

LATER = ["Now give the three most important points of your answer, one line each.",
         "Rewrite those three points as a short table with two columns."]
ASSISTANT = "<|im_start|>assistant\n"
THINK = "<think>\n\n</think>\n\n"
READ_LINE = "reads its conversation's next turn in its own windows"


def first_prompts(tok, pad):
    """Two first turns: a question each behind about `pad` tokens of its own filler text."""
    para = ("Note %d-%d: the survey team measured the northern ridge, logged the readings of station %d, and sent the "
            "figures to the office, where a clerk filed the sheets under reference %d. ")
    out = []
    for qi in range(2):
        per = max(1, len(tok.encode(para % (qi, 100, 100, 100))))
        filler = "".join(para % (qi, i, (i * 7 + qi) % 97, i * 13 + qi) for i in range(pad // per + 1)) if pad > 0 else ""
        text = f"<|im_start|>user\n{filler}\n\n{BT.QUESTIONS[qi]}<|im_end|>\n{ASSISTANT}{THINK}"
        out.append(tok.encode(text, parse_special=True))
    return out


def run(a, cfg, tok, mode, log_path):
    """The three turns of both conversations on a fresh engine -> (tokens per turn and conversation, slot reads)."""
    os.environ["BATCH_TEST_LOG"] = log_path
    env = {"STRATA_DECODE_TIMING": "1", "STRATA_SLOT_READ": "1"}   # (opt-in: the engine then says INFO slot_read=1)
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    env.update(cfg.get("env") or {})
    eng = BT.Engine(a.exe, cfg, 2, env, a.extra.split())
    out = eng.lines()
    im_start = tok.encode("<|im_start|>", parse_special=True)[0]
    prompts = first_prompts(tok, a.pad)
    turns = []
    try:
        for t in range(3):
            got, done = {0: [], 1: []}, {}
            t0 = time.time()
            for i, ids in enumerate(prompts):
                key = "slot_read=1" if (mode == "slot" and t > 0) else ""
                eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, key, ",".join(map(str, ids))) if x))
                for line in out:
                    if line.startswith("T "):
                        got[i].append(int(line.split()[1]))
                    elif line.startswith("BT "):
                        _, s, y = line.split()
                        got[int(s)].append(int(y))
                    elif line.startswith("BDONE "):
                        done[int(line.split()[1])] = line
                    elif line.startswith("ERR"):
                        raise RuntimeError(line)
                    elif line.startswith("BADM "):
                        if line.split()[2] == "0":
                            done[i] = line
                        break
            for line in (out if len(done) < 2 else []):
                if line.startswith("BT "):
                    _, s, y = line.split()
                    got[int(s)].append(int(y))
                elif line.startswith("BDONE "):
                    done[int(line.split()[1])] = line
                    if len(done) == 2:
                        break
                elif line.startswith("ERR"):
                    raise RuntimeError(line)
            print(f"  {mode} turn {t + 1}: prompts {[len(p) for p in prompts]} tokens, {[len(got[i]) for i in (0, 1)]} "
                  f"generated in {time.time() - t0:.1f} s | {done.get(0, '')[:44]} | {done.get(1, '')[:44]}", flush=True)
            turns.append([got[0], got[1]])
            if t == 2:
                break
            nxt = tok.encode(f"<|im_end|>\n<|im_start|>user\n{LATER[t]}<|im_end|>\n{ASSISTANT}{THINK}", parse_special=True)
            # conversation 0, as a client renders the history: without the thinking block before the reply
            p0 = prompts[0]
            hdr = len(p0) - 1 - p0[::-1].index(im_start)
            prompts[0] = p0[:hdr] + tok.encode(ASSISTANT, parse_special=True) + got[0] + nxt
            # conversation 1: exactly what its slot holds, and on
            prompts[1] = prompts[1] + got[1] + nxt
    finally:
        eng.send("QUIT")
        try:
            eng.p.wait(timeout=180)
        except Exception:
            eng.p.kill()
    text = Path(log_path).read_text(errors="replace")
    return turns, text.count(READ_LINE), "STRATA_SLOT_READ:" in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pad", type=int, default=2000, help="tokens of filler text before each conversation's first question")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 seed=5"')
    ap.add_argument("--extra", default="--pcie-frac 0 --adapt-every 1000000 --short-read 100000",
                    help="more engine arguments in one string (the default fixes the expert placement and reads every "
                         "prompt through the verify windows, which the comparison needs)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--log-base", default=os.path.join(tempfile.gettempdir(), "slot_read_test_engine"),
                    help="the two engine logs are <log-base>.ref.log and <log-base>.slot.log")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    try:
        ref, ref_reads, _ = run(a, cfg, tok, "ref", a.log_base + ".ref.log")
        slot, slot_reads, able = run(a, cfg, tok, "slot", a.log_base + ".slot.log")
    except RuntimeError as e:
        print("engine:", e)
        return 1
    if not able:
        print("this engine does not read turns in its slots: it needs the pipelined groups with drafts (a layer split, "
              "--batch-mtp, STRATA_PIPED_MTP and STRATA_BATCH_SELF_COMMIT=0) - see", a.log_base + ".slot.log")
        return 2
    ok = True
    for t in (1, 2):
        for i in (0, 1):
            same = ref[t][i] == slot[t][i]
            ok &= same
            first = next((k for k, (x, y) in enumerate(zip(ref[t][i], slot[t][i])) if x != y),
                         min(len(ref[t][i]), len(slot[t][i])))
            print(f"turn {t + 1} conversation {i} ({'its turn checkpoint' if i == 0 else 'its live state'}): "
                  f"{len(slot[t][i])} tokens read in the slot, {len(ref[t][i])} through the main sessions: "
                  f"{'IDENTICAL' if same else f'DIFFERENT from token {first}'}")
            print("   ", ascii(tok.decode(slot[t][i])[:110]))
    print(f"turns read in their slot: {slot_reads} of 4 in the slot run, {ref_reads} in the reference run")
    if slot_reads != 4 or ref_reads != 0:
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
