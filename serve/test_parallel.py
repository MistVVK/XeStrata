# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, engine_args, parallel_args, serve

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args        # #997: the first window over two slots fails
STEP = 0.02
CH = 32
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else ""), flush=True)
print("READY 4096 stop", flush=True)
LONG = list(b"LONGREPLY")
def continued(ids):        # (the whole reply, how much of it the prompt already ends with: a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R, max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
def reply(ids):            # the rest of the reply after what the prompt already ends with
    R, k = continued(ids)
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
# #879: a prompt with NFSOLO / NFADMIT ends its first GEN / BGEN with DONE ... nonfinite (nothing emitted); one with
# NFSLOT ends its slot with BDONE ... nonfinite after its second token
fired, nf_slots = set(), set()
def window():        # one batch window: every active slot one token
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        if b in nf_slots and produced >= 2:
            nf_slots.discard(b)
            print(f"BDONE {b} {produced} nonfinite 1.0", flush=True)
            del active[b]
            continue
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
# DONE's 15th field: the prompt tokens the engine read (a continued prompt: the 1 it copies)
READF = "--readfield" in args
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # #1059: refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        # #1994: the engine's own view of its slots differs: BUSYSLOT is refused once, BUSYALL always
        pbx = bytes(int(x) & 255 for x in ids)
        if slot is not None and (b"BUSYALL" in pbx or (b"BUSYSLOT" in pbx and "busyslot" not in fired)):
            fired.add("busyslot")
            print(f"ERR BGEN: no such free slot (--batch {slots}) or a bad max_new", flush=True)
            if log:
                log.write(f"REFUSED {slot}\n"); log.flush()
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pb = bytes(int(x) & 255 for x in ids)
        mark = b"NFSOLO" if f[0] == "GEN" else b"NFADMIT"
        if mark in pb and mark not in fired:
            fired.add(mark)
            print(f"DONE 0 {len(ids)} 5.0 0.0 nonfinite 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"NFSLOT" in pb:
            nf_slots.add(slot)
        toks = reply(ids)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {reused(ids)}", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {reused(ids)}" +
              (f" 0 0 0 0 0.0 {len(ids) - reused(ids)}" if READF else ""), flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 20}, []), ["--batch", "20"])     # the engine warns and caps

    def test_parallel_above_the_window_says_so_unless_batch_mtp(self):
        import contextlib
        import io
        import os

        def said(n, args, env, wide_env=None):
            """What the server says of "parallel": n with these args, XESTRATA_BATCH_MTP and XESTRATA_WIDE_WINDOW."""
            buf = io.StringIO()
            old = {k: os.environ.get(k) for k in ("XESTRATA_BATCH_MTP", "XESTRATA_WIDE_WINDOW")}
            for k, v in (("XESTRATA_BATCH_MTP", env), ("XESTRATA_WIDE_WINDOW", wide_env)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": n}, args)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.assertEqual(r, ["--batch", str(n)])
            return buf.getvalue()
        # the window holds 16 rows (the wide windows' default, auto)
        self.assertNotIn("at most", said(12, [], None))
        self.assertIn("at most 16", said(20, [], None))
        self.assertIn("at most 16", said(20, [], "0"))
        # the wide windows off - by the args or the environment - hold it at 8, as before
        self.assertIn("at most 8", said(12, ["--wide-window", "off"], None))
        self.assertIn("at most 8", said(12, ["--wide-window", "0"], None))
        self.assertIn("at most 8", said(12, [], None, wide_env="0"))
        self.assertNotIn("at most", said(8, ["--wide-window", "off"], None))
        self.assertNotIn("at most", said(12, ["--wide-window", "on"], None, wide_env="0"))   # the args' own value wins
        # --batch-mtp waves more through the window
        self.assertNotIn("at most", said(20, ["--batch-mtp"], None))
        self.assertNotIn("at most", said(12, ["--wide-window", "off"], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])
        args = engine_args({"args": ["--pack", "p"], "wide_window": "off", "parallel": 2})
        self.assertIn("--wide-window", args)

    def test_auto_is_not_a_count(self):
        # the server fills "auto" in from the machine before the engine starts (serve/autotune.py); one that reaches
        # the arguments unresolved is one at a time
        self.assertEqual(parallel_args({"parallel": "auto"}, []), [])

    def test_batch_groups_and_the_conversation_cache(self):
        self.assertEqual(engine_args({"args": ["--pack", "p"], "parallel": 4, "batch_groups": "auto"})[-4:],
                         ["--batch", "4", "--batch-groups", "auto"])
        self.assertEqual(engine_args({"args": ["--pack", "p"], "batch_groups": 2})[-2:], ["--batch-groups", "2"])
        self.assertEqual(engine_args({"args": ["--pack", "p"], "conversation_cache_mib": 2048})[-2:],
                         ["--conversation-cache-mib", "2048"])
        self.assertEqual(engine_args({"args": ["--pack", "p"], "conversation_cache_mib": 0})[-2:],
                         ["--conversation-cache-mib", "0"])
        # "auto" leaves the engine's own sizing (from the RAM free once the model is loaded)
        self.assertNotIn("--conversation-cache-mib", engine_args({"args": ["--pack", "p"],
                                                                  "conversation_cache_mib": "auto"}))
        # the config's own engine arguments win
        mine = ["--pack", "p", "--batch-groups", "1", "--conversation-cache-mib", "512"]
        got = engine_args({"args": list(mine), "batch_groups": "auto", "conversation_cache_mib": 2048})
        self.assertEqual(got[:len(mine)], mine)
        self.assertEqual(got.count("--batch-groups"), 1)
        self.assertEqual(got.count("--conversation-cache-mib"), 1)


class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        e.slot_group = [0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt

    def test_a_rewritten_next_turn_goes_to_the_slot_of_its_chat(self):
        # #2518: two chats in two slots; each next turn renders the history again but not the reply it decoded or
        # the assistant header, so the held stream (prompt + reply) is no start of it.  The engine's turn checkpoint
        # sits at the last <|im_start|> of the old prompt: the prompt that agrees up to there belongs to that slot.
        # Before, both went to the least recently used slot in turn, and the first one overwrote the other chat's
        # cache.
        T = StrataEngine.TURN_ID
        hist_a, hist_b = list(range(1000, 1600)), list(range(2000, 2600))
        held_a = hist_a + [T, 7, 8, 9] + [50, 51, 52]                  # prompt (ends with the assistant header) + reply
        held_b = hist_b + [T, 7, 8, 9] + [60, 61]
        e = self.engine(2)
        e.slot_held = [held_a, held_b]
        e.slot_used = [10.0, 20.0]                                      # slot 0 was used longer ago
        next_a = hist_a + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))  # the history again, then new turns
        next_b = hist_b + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))
        self.assertEqual(e.pick_slot(next_b), 1)                        # the LRU slot 0 holds the other chat
        self.assertEqual(e.pick_slot(next_a), 0)
        e.slot_busy[0] = True                                           # a busy slot is never picked
        self.assertEqual(e.pick_slot(next_a), 1)
        e.slot_busy[0] = False
        # a chat that agrees with neither (a new one, or one that rewrote its history) keeps the old order
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        self.assertEqual(e.pick_slot(hist_a[:300] + [T, 5, 6] + list(range(3000, 3100))), 0)
        e.slot_held[0] = []                                             # an empty slot still comes before evicting one
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        # the whole held stream continued is still the longer match
        e.slot_held = [held_a, held_b + [T, 1, 2]]
        self.assertEqual(e.pick_slot(held_b + [T, 1, 2, 3, 4]), 1)

    def test_batch_groups_spread_requests_over_the_groups(self):
        # #1249: with --batch-groups 2 over 4 slots (groups {0,1} and {2,3}) a second request goes to the other group,
        # and a short held prefix does not pull it back into the busy one
        e = self.engine(4)
        e.slot_groups = 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 2)
        e.slot_held[1] = list(range(1, 601))                       # a long prefix is worth the unbalanced group
        self.assertEqual(e.pick_slot(list(range(1, 700))), 1)
        e.slot_groups, e.slot_group = 1, [0] * 4                   # without groups: any held prefix, as before
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 1)

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})


class ParallelService(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, reuse=False, more=()):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--reuse"] if reuse else []
        extra += list(more)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def message(self, text, max_tokens=64):
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def _two_at_once(self, text_b):
        """A long request first, then one that has to go to a batch slot (the engine is busy with the first)."""
        out = {}

        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, e.read().decode(), dict(e.headers))
        ta = threading.Thread(target=go, args=("a", "LONGREPLY please"))
        ta.start()
        time.sleep(0.15)
        tb = threading.Thread(target=go, args=("b", text_b))
        tb.start()
        ta.join(30)
        tb.join(30)
        return out

    def test_bgen_refused_by_the_engine_is_retried_not_a_400(self):
        """#1994: the engine refuses a BGEN ("no such free slot") although the server saw the slot free: the request
        goes to another slot (or waits) and is answered; the same conversation's next turn works too."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.2
        out = self._two_at_once("BUSYSLOT hello")
        self.assertEqual(out["b"][0], "ok", out["b"])
        self.assertIn("REFUSED", self.log.read_text())
        again = self.chat("BUSYSLOT hello")                     # the conversation is not poisoned
        self.assertEqual(again["choices"][0]["message"]["content"], "ok, done.")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])

    def test_bgen_refused_for_good_is_a_retryable_503_and_frees_the_slots(self):
        """#1994: when the engine keeps refusing, the answer is a 503 (retryable), never a 400, and no slot stays
        busy.  upstream b3c633868 (#3040): the refused admission keeps an error row but is not a completed request
        (no prompt tokens, no part in the totals)."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.05
        out = self._two_at_once("BUSYALL hello")
        self.assertEqual(out["b"][0], "http", out["b"])
        self.assertEqual(out["b"][1], 503, out["b"])
        metrics = self.get("/metrics")
        refused = [row for row in metrics["requests"] if row["output_tokens"] == 0]
        self.assertEqual(len(refused), 1, metrics["requests"])
        self.assertEqual((refused[0]["finish"], refused[0]["prompt_tokens"], refused[0]["prompt_read"]),
                         ("error", 0, None))
        self.assertEqual(metrics["totals"]["requests"], 1, "only the admitted request counts")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.chat("fine again")["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["totals"]["requests"], 2, "a later admitted request counts")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5 (4 x R9700 burst): the
        server listens with a deep backlog (XESTRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers."""
        import serve.server as server
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        self.start(2)
        errs, ok = [], []

        def one(i):
            try:
                self.chat(f"burst {i}", max_tokens=8)
                ok.append(i)
            except Exception as e:                      # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.assertEqual((len(ok), errs[:2]), (60, []))

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def wait_slots_free(self):
        """The HTTP reply can finish while a background BDONE drain still owns a slot (the handler's
        _release_slot_when_done frees it at the engine's BDONE, on a thread of its own): wait for its notification,
        but still fail if a slot never comes free - the leak or deadlock such a wait must not hide."""
        with self.engine.slot_cv:
            self.assertTrue(self.engine.slot_cv.wait_for(lambda: not any(self.engine.slot_busy), timeout=10),
                            f"slots did not clear: {self.engine.slot_busy}")

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.wait_slots_free()

    def test_three_requests_long_long_short_do_not_deadlock(self):
        """ENGINE_REVIEW finding 1 (0.1.41 check): two long prompts fill both slots, a short one arrives a moment later
        and wants one of them (the yield path).  The reviewer's repro: long, long, short at 0.25 s steps.  Every
        request must be answered; none may wait on the control lock that the yielding read holds."""
        self.start(2)
        texts = ["long " * 600, "lung " * 600, "short"]
        res, errors = {}, []

        def go(t):
            t0 = time.time()
            try:
                r = self.chat(t, max_tokens=64)
                res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
            except Exception as e:      # noqa: BLE001 - a timeout is the failure this test looks for
                errors.append((t[:6], repr(e)))
        threads = []
        for t in texts:
            th = threading.Thread(target=go, args=(t,))
            th.start()
            threads.append(th)
            time.sleep(0.25)
        for th in threads:
            th.join(45)
        self.assertFalse(any(th.is_alive() for th in threads), "a request never finished (deadlock)")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 3)
        # The HTTP reply can finish while a background BDONE drain still owns a slot.  Wait for its notification,
        # but still fail if a slot never comes free (the deadlock this test is meant to catch).
        with self.engine.slot_cv:
            self.assertTrue(self.engine.slot_cv.wait_for(lambda: not any(self.engine.slot_busy), timeout=10),
                            f"slots did not clear: {self.engine.slot_busy}")
            self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.wait_slots_free()

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held
                if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # #997 #890: a batch window fails while every request reads its own slot's lines, so no request reads the
        # engine's ERR line before it exits; the server's note on the death gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_nonfinite_in_batch_mode(self):
        """#879 in --batch mode: DONE ... nonfinite on the solo path or at an admission (nothing emitted yet) sends the
        request again once; BDONE ... nonfinite in a slot ends the reply that had started with an error (503)."""
        self.start(2)
        self.assertEqual(self.chat("NFSOLO alone")["choices"][0]["message"]["content"], "ok, done.")
        gens = [x for x in self.log.read_text().splitlines() if x.startswith("GEN")]
        self.assertEqual(len(gens), 2, gens)                              # sent again once
        res = self.race("first question LONGREPLY", "NFADMIT second")
        self.assertEqual(res["NFADMIT second"][0], "ok, done.")
        self.assertEqual(res["first question LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        out = {}

        def go(t):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 64, "reasoning_effort": "none"}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out[t] = json.loads(r.read().decode())["choices"][0]
            except urllib.error.HTTPError as e:
                with e:
                    out[t] = (e.code, json.loads(e.read().decode()))
        a = threading.Thread(target=go, args=("third question LONGREPLY",))
        a.start()
        time.sleep(0.15)
        go("NFSLOT fourth")
        a.join(30)
        code, err = out["NFSLOT fourth"]
        self.assertEqual((code, err["error"]["type"]), (503, "server_error"))
        self.assertIn("non-finite (NaN) after", err["error"]["message"])
        self.assertEqual(out["third question LONGREPLY"]["message"]["content"], "ok, " + "la " * 15 + "done.")
        # the engine dropped that slot's state: the server does not route the conversation's next turn to it
        self.assertFalse([h for h in self.engine.slot_held if bytes(t & 255 for t in h).find(b"NFSLOT") >= 0])
        with self.svc.status_lock:
            self.assertEqual(self.svc.live_reqs, {})

    def test_a_continued_request_reports_its_own_prompt_reuse(self):
        """A request STOPped on the solo path and continued in a slot (or back on the solo path) is sent again as its
        prompt + what it generated, and the engine holds all of that: the usage must still split the request's OWN
        prompt (input_tokens + cache_read_input_tokens = the prompt, input_tokens >= 1).  input_tokens 0 makes Claude
        Code count the prompt twice (message_start's input_tokens + cache_read_input_tokens): "Prompt is too long"
        at half the context."""
        self.start(2, slot_cache=True, reuse=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.message("LONGREPLY please", 200)))
        long.start()
        time.sleep(0.05)
        self.message("short", 200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)                    # it was continued: the case under test
        u = res["long"]["usage"]
        prompt = self.svc.history[[h["output_tokens"] for h in self.svc.history].index(
            max(h["output_tokens"] for h in self.svc.history))]["prompt_total"]
        self.assertGreaterEqual(u["input_tokens"], 1, u)
        self.assertEqual(u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"], prompt, u)
        self.assertEqual(u["cache_read_input_tokens"], 3, u)    # the first read's reuse, not the continuation's

    def test_a_slot_request_ended_by_its_stop_token_reports_the_slot_leg(self):
        """#1869: the consumer leaves at the stop token, before the BDONE: the request kept the admission's one token
        and its few ms.  Its figures are the slot leg's now (finish=stop was the broken case, finish=length read the
        BDONE)."""
        import serve.server as server
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):          # the fake engine's stop token
            a = threading.Thread(target=lambda: res.setdefault("a", self.chat("List three primes please", 200)))
            b = threading.Thread(target=lambda: res.setdefault("b", self.chat("Name two rivers in Germany", 200)))
            a.start()
            b.start()
            a.join(30)
            b.join(30)
        self.assertEqual(len(self.svc.history), 2)
        for h in self.svc.history:
            self.assertEqual(h["engine_generated"], h["output_tokens"], h)
            self.assertGreaterEqual(h["decode_ms"], 1.0, h)
        for r in res.values():
            t = r["timings"]
            self.assertEqual(t["prompt_n"] + t["cache_n"], r["usage"]["prompt_tokens"], t)
            self.assertEqual(t["predicted_n"], 10, t)
        self.assertFalse(any(self.engine.slot_busy))                 # the slots are free (no BDONE left for a wait)

    def test_a_request_moved_between_solo_and_slot_keeps_its_totals(self):
        """#1869/#1870: a request that read its prompt and decoded on the solo path, was STOPped when another arrived
        and went on in a slot (its admission copies the held prefix: prompt_read 1, one token) reports the prompt it
        read and the decode time and tokens of ALL its legs, not the last one's."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.25)                                   # a few solo tokens first
        self.chat("short", max_tokens=200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)   # solo, then continued in a slot: the case under test
        n = res["long"]["usage"]["completion_tokens"]
        row = max(self.svc.history, key=lambda h: h["output_tokens"])
        self.assertEqual(row["engine_generated"], n, row)
        self.assertGreater(row["decode_ms"], 20.0 * 3, row)          # the solo leg's STEP-long tokens are in it
        self.assertGreater(row["prompt_read"], 1, row)                # the first read, not the copy leg's 1
        t = res["long"]["timings"]
        self.assertEqual(t["prompt_n"] + t["cache_n"], res["long"]["usage"]["prompt_tokens"], t)
        self.assertEqual(t["cache_n"], 3, t)                          # the first read's reuse only
        self.assertGreater(t["predicted_ms"], 60.0, t)

    def test_a_leaked_busy_slot_does_not_hang_the_next_request(self):
        """#1603: a slot whose busy flag has no owner (no request holds it, no BDONE wait is running) kept every later
        request waiting for a slot nothing would free: running 0, queued 0, and the frozen-engine check never looked.
        The waiting request is now counted (admitting), the slot is given back after ORPHAN_SLOT_S, and it is
        answered."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.4
        with e.slot_cv:
            e.slot_busy[0] = e.slot_busy[1] = True       # the leak: busy, claimed by nobody, nothing ending them
        # the solo path needs a quiet engine; a busy flag sends the request to the slot path, which waits
        res, errs = {}, []

        def go():
            try:
                res["r"] = self.chat("after the leak", max_tokens=16)
            except Exception as ex:                       # noqa: BLE001
                errs.append(ex)
        t = threading.Thread(target=go)
        t0 = time.time()
        t.start()
        seen = []
        while t.is_alive() and time.time() - t0 < 20:
            live = self.get("/metrics")["live"]
            seen.append((live.get("running"), live.get("queued"), live.get("admitting")))
            time.sleep(0.05)
        t.join(5)
        self.assertEqual(errs, [])
        self.assertFalse(t.is_alive(), "the request hung on a slot nobody owns")
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        # while it waited it was visible: running as a request, and counted as waiting for a slot - never 0 / 0 / 0
        self.assertTrue(any(r == 1 and a == 1 for r, q, a in seen), seen)
        first = next(i for i, x in enumerate(seen) if sum(v or 0 for v in x))
        self.assertTrue(all(sum(v or 0 for v in x) for x in seen[first:-1]), seen)   # never invisible once it arrived
        self.assertEqual(e.admitting, 0)
        self.assertEqual(e.slot_claims, set())

    def test_an_owned_busy_slot_is_never_taken(self):
        """The leak check frees only slots with no owner: a slot a request holds stays busy however long it waits."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.0
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
            e.slot_busy[1] = True
            e.slot_releasing.add(1)
            self.assertEqual(e._reap_orphan_slots(), [])
            self.assertEqual(e.slot_busy, [True, True])
            e.slot_claims.discard(0)
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertEqual(e.slot_busy, [False, True])

    def test_a_parallel_request_does_not_wait_for_the_fifo_on_a_loaded_engine(self):
        """#1603: parallel requests used to take the FIFO (held by an image encode, /v1/vram, load) just to find the
        engine loaded; between 'queued' and 'running' they were neither.  Loaded: no fifo.  Unloaded: they wait for it
        and are counted as queued the whole time."""
        self.start(2)
        self.assertEqual(self.chat("warm")["choices"][0]["message"]["content"], "ok, done.")
        self.svc.fifo.acquire()
        try:
            r = self.chat("with the fifo held")             # would block here before
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        finally:
            self.svc.fifo.release()
        import serve.server as server
        script, real = Path(self.tmp.name) / "fake_strata.py", server.subprocess.Popen
        patch = mock.patch.object(server.subprocess, "Popen",      # the reload starts the fake again
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.unload()
        self.svc.fifo.acquire()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.chat("reloads", max_tokens=16)))
        t.start()
        try:
            deadline = time.time() + 10
            live = {}
            while time.time() < deadline:
                live = self.get("/metrics")["live"]
                if live.get("queued"):
                    break
                time.sleep(0.05)
            self.assertEqual(live.get("queued"), 1, live)    # waiting for the fifo: queued, not nothing
            self.assertEqual(live.get("running"), 0, live)
        finally:
            self.svc.fifo.release()
        t.join(60)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["live"]["queued"], 0)


if __name__ == "__main__":
    unittest.main()
