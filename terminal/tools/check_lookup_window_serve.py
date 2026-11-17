#!/usr/bin/env python3
"""Gate check: a Qwen serve engine answers a request that makes prompt lookup propose long windows.

The #4040 default (--lookup-max-t 12) builds verify windows of up to 12 rows when the build holds 8: on Metal that was refused
with "metal window: window size out of range" at the first request that repeated text, and no gate request repeated any.  This
starts the engine as the Terminal would (serve, the build's defaults - no --lookup-max-t), sends a chat that asks for a text to be
repeated three times, twice, and fails on an ERR line or on a request that did not end with DONE.  With --need-drafts it also
requires that drafts were offered, i.e. that lookup windows really ran.

  check_lookup_window_serve.py --exe build/strata --pack-tokenizer PACK/tokenizer -- <engine args without --serve>
Exit 0 = pass, 1 = fail.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

TEXT = ("The quick brown fox jumps over the lazy dog near the quiet river bank, while the old farmer counts seventeen sheep, "
        "nine goats and four noisy geese before supper. The quick brown fox jumps over the lazy dog near the quiet river bank.")
PROMPT = ("<|im_start|>user\nRepeat the following text exactly, three times in a row, with no other words:\n\n" + TEXT +
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def prompt_ids(tokenizer_dir: Path) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import strata_tokenizer as ST
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tokenizer_dir / "token_type.json").read_text())
    try:
        tcfg = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    return list(tok.encode(PROMPT, parse_special=True))


def check(exe, engine_args, ids, max_new=96, start_timeout=900, request_timeout=1200, need_drafts=False, env=None, cwd=None):
    """Returns (ok, lines): one summary line per request (or the reason it failed)."""
    p = subprocess.Popen([exe, "--serve", *engine_args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, env=env, cwd=cwd)
    q = queue.Queue()

    def pump():
        for line in p.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def until(prefixes, timeout):
        end = time.monotonic() + timeout
        got = []
        while time.monotonic() < end:
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                got.append("<engine exited>")
                return got
            got.append(line)
            if line.startswith(prefixes):
                return got
        got.append("<timeout>")
        return got

    lines, ok = [], True
    try:
        r = until(("READY",), start_timeout)
        if not r[-1].startswith("READY"):
            return False, ["engine did not start: " + " | ".join(r[-3:])]
        for n in (1, 2):
            p.stdin.write(f"GEN {max_new} " + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            out = until(("DONE", "ERR"), request_timeout)
            last = out[-1]
            toks = sum(1 for x in out if x.startswith("T "))
            f = last.split()
            offered = int(f[7]) if last.startswith("DONE") and len(f) > 7 and f[7].isdigit() else 0
            good = last.startswith("DONE") and (offered > 0 or not need_drafts)
            ok = ok and good
            lines.append(f"request {n}: {last[:160]} | {toks} tokens | drafts offered {offered}" + ("" if good else "  <-- FAIL"))
            if not good:
                break
        return ok, lines
    finally:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
        except (OSError, ValueError):
            pass
        time.sleep(1)
        if p.poll() is None:
            p.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack-tokenizer", required=True, help="the pack's tokenizer folder (vocab.json, merges.txt, token_type.json)")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--need-drafts", action="store_true", help="also fail if no draft was offered (lookup windows did not run)")
    ap.add_argument("--cwd")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    args = a.engine_args[1:] if a.engine_args[:1] == ["--"] else a.engine_args
    ok, lines = check(a.exe, args, prompt_ids(Path(a.pack_tokenizer)), a.max_new, need_drafts=a.need_drafts, cwd=a.cwd)
    print("\n".join(lines))
    print("LOOKUP-WINDOW SERVE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
