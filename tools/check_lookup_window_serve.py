#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Gate check: a Qwen serve engine answers a request that makes prompt lookup propose long windows.

A prompt that repeats text makes prompt lookup propose verify windows of several rows (up to the build's lookup
limit).  A backend that cannot take that row count refuses at the first request that repeats text, and a gate whose
requests never repeat anything never sees it (Metal refused "window size out of range" at 12 rows on a build holding
8).  This starts the engine as a person's server would (--serve, the build's defaults, no lookup option), sends a chat
that asks for a text to be repeated three times, twice, and fails on an ERR line or on a request that did not end with
DONE.  With --need-drafts it also requires that drafts were offered, i.e. that lookup windows really ran.

  check_lookup_window_serve.py --exe build/contrib-llvm/xestrata-engine --pack-tokenizer PACK/tokenizer -- <engine args>
Exit 0 = pass, 1 = fail.
"""
import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

TEXT = ("The quick brown fox jumps over the lazy dog near the quiet river bank, while the old farmer counts seventeen "
        "sheep, nine goats and four noisy geese before supper. The quick brown fox jumps over the lazy dog near the "
        "quiet river bank.")
PROMPT = ("<|im_start|>user\nRepeat the following text exactly, three times in a row, with no other words:\n\n" + TEXT +
          "<|im_end|>\n<|im_start|>assistant\n")


def prompt_ids(tokenizer_dir: Path) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import strata_tokenizer as ST
    return list(ST.Tokenizer.from_dir(str(tokenizer_dir)).encode(PROMPT, parse_special=True))


def check(exe, engine_args, ids, max_new=96, start_timeout=900, request_timeout=1200, need_drafts=False, env=None,
          cwd=None):
    """Returns (ok, lines): one summary line per request (or the reason it failed)."""
    p = subprocess.Popen([exe, "--serve", *engine_args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, env=env, cwd=cwd)
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
            lines.append(f"request {n}: {last[:160]} | {toks} tokens | drafts offered {offered}" +
                         ("" if good else "  <-- FAIL"))
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
    ap.add_argument("--pack-tokenizer", required=True,
                    help="the pack's tokenizer folder (vocab.json, merges.txt, token_type.json)")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--need-drafts", action="store_true",
                    help="also fail if no draft was offered (lookup windows did not run)")
    ap.add_argument("--cwd")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    args = a.engine_args[1:] if a.engine_args[:1] == ["--"] else a.engine_args
    ok, lines = check(a.exe, args, prompt_ids(Path(a.pack_tokenizer)), a.max_new, need_drafts=a.need_drafts,
                      cwd=a.cwd)
    print("\n".join(lines))
    print("LOOKUP-WINDOW SERVE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
