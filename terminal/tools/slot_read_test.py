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
With --alone a round's turns are sent one after the other instead, each when the one before it has ended: a slot
then reads while the other group stands idle, where the engine sends its windows ahead through that group's rows
(STRATA_SLOT_READ_AHEAD); --later-pad makes the later turns long enough for that to happen many times, and
--expect-ahead fails the test when it did not happen at all.  --stop-read MS (with --alone) stops conversation 1's
later turns MS milliseconds into their read (BSTOP) and sends each again: the turn then goes on from what the slot
had read, and its reply must still be the reference's.

The engine needs the pipelined groups with drafts (a layer split, --batch-mtp, STRATA_PIPED_MTP and
STRATA_BATCH_SELF_COMMIT=0 in the config's env or the shell); the test says so when it does not have them.

  python3 tools/slot_read_test.py --exe engine/strata --config strata-<model>.json \
      [--pad 2000] [--max-new 120] [--keys "temperature=0.7 seed=5 penalty_last_n=64 penalty_present=1.5"] \
      [--alone --later-pad 400 --expect-ahead] [--stop-read 300]

Exit 0: every later turn identical in both runs, each of them read in its slot in the slot run; 2: a turn differs
or was not read in its slot; 1: an engine error.
"""
import argparse
import json
import os
import re
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
AHEAD = re.compile(r"read in the slots: \d+ prompt tokens, \d+ turns, (\d+) windows ahead")
PARA = ("Note %d-%d: the survey team measured the northern ridge, logged the readings of station %d, and sent the "
        "figures to the office, where a clerk filed the sheets under reference %d. ")


def filler(tok, n, salt):
    """About n tokens of text that no other call with another salt repeats."""
    if n <= 0:
        return ""
    per = max(1, len(tok.encode(PARA % (salt, 100, 100, 100))))
    return "".join(PARA % (salt, i, (i * 7 + salt) % 97, i * 13 + salt) for i in range(n // per + 1))


def first_prompts(tok, pad):
    """Two first turns: a question each behind about `pad` tokens of its own filler text."""
    out = []
    for qi in range(2):
        text = f"<|im_start|>user\n{filler(tok, pad, qi)}\n\n{BT.QUESTIONS[qi]}<|im_end|>\n{ASSISTANT}{THINK}"
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
    stops = []   # --stop-read: each stopped read's BDONE line
    try:
        for t in range(3):
            got, done = {0: [], 1: []}, {}
            t0 = time.time()
            for i, ids in enumerate(prompts):
                key = "slot_read=1" if (mode == "slot" and t > 0) else ""
                bgen = " ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, key, ",".join(map(str, ids))) if x)
                if key and i == 1 and a.stop_read > 0:
                    # the turn's read is stopped on its way, and the same turn sent again
                    eng.send(bgen)
                    for line in out:
                        if line.startswith("ERR"):
                            raise RuntimeError(line)
                        if line.startswith("BADM "):
                            break
                    time.sleep(a.stop_read / 1000.0)
                    eng.send(f"BSTOP {i}")
                    for line in out:
                        if line.startswith("ERR"):
                            raise RuntimeError(line)
                        if line.startswith(f"BDONE {i} "):
                            stops.append(line)
                            break
                eng.send(bgen)
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
                        if not (a.alone and i not in done):   # --alone: this turn ends before the next is sent
                            break
                    if a.alone and i in done:
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
            # (--later-pad: the later turns bring their own text, as a tool's output does; each conversation its own)
            nxts = []
            for i in range(2):
                pad = filler(tok, a.later_pad, 10 + 2 * t + i)
                nxts.append(tok.encode(f"<|im_end|>\n<|im_start|>user\n{pad}{chr(10) * 2 if pad else ''}{LATER[t]}"
                                       f"<|im_end|>\n{ASSISTANT}{THINK}", parse_special=True))
            # conversation 0, as a client renders the history: without the thinking block before the reply
            p0 = prompts[0]
            hdr = len(p0) - 1 - p0[::-1].index(im_start)
            prompts[0] = p0[:hdr] + tok.encode(ASSISTANT, parse_special=True) + got[0] + nxts[0]
            # conversation 1: exactly what its slot holds, and on
            prompts[1] = prompts[1] + got[1] + nxts[1]
    finally:
        eng.send("QUIT")
        try:
            eng.p.wait(timeout=180)
        except Exception:
            eng.p.kill()
    text = Path(log_path).read_text(errors="replace")
    return (turns, text.count(READ_LINE), "STRATA_SLOT_READ:" in text, sum(int(m) for m in AHEAD.findall(text)), stops)


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
    ap.add_argument("--alone", action="store_true",
                    help="a round's turns one after the other, each when the one before it has ended")
    ap.add_argument("--later-pad", type=int, default=0, help="tokens of new text in each later turn")
    ap.add_argument("--expect-ahead", action="store_true",
                    help="fail when the slot run sent no window ahead through an idle group (STRATA_SLOT_READ_AHEAD)")
    ap.add_argument("--stop-read", type=int, default=0,
                    help="with --alone: stop conversation 1's later turns this many ms into their read, then send them again")
    a = ap.parse_args()
    if a.stop_read > 0 and not a.alone:
        ap.error("--stop-read needs --alone (one request at a time)")
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    try:
        ref, ref_reads, _, _, _ = run(a, cfg, tok, "ref", a.log_base + ".ref.log")
        slot, slot_reads, able, ahead, stops = run(a, cfg, tok, "slot", a.log_base + ".slot.log")
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
    print(f"turns read in their slot: {slot_reads - len(stops)} of 4 in the slot run, {ref_reads} in the reference run; "
          f"{ahead} windows sent ahead through an idle group")
    for line in stops:   # (BDONE <slot> <tokens> cancel ...: 0 tokens = stopped while it read)
        print(f"a read stopped and sent again: {line[:60]}")
    if slot_reads - len(stops) != 4 or ref_reads != 0 or (a.expect_ahead and ahead == 0):
        ok = False
    if a.stop_read > 0 and (len(stops) != 2 or not all(" cancel" in s for s in stops)):
        print("--stop-read: the two reads were not stopped")
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
