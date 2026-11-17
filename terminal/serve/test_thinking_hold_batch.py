"""The thinking budget's hold on a --batch engine (lane AGENT-TURN, P2b): GEN / BGEN hold=N:ID reaches the batch slots too.

The fake engine of test_parallel.py (a script speaking GEN / BGEN / T / DONE / BADM / BT / BDONE / BSTOP) is given the engine's side of the
hold, as generate.cpp has it: READY says "hold holdb"; a solo GEN ends after N emitted tokens unless token ID came first (DONE hold); a BGEN's
admission token counts toward N and its slot ends with BDONE hold at exactly N; once ID is emitted the hold is over.  What is tested is the
server: the key's count falls by what an earlier leg of the same pass already sent (the solo leg promoted to a slot), the finish "hold" ends
the pass, no key goes to an engine whose READY lacks holdb, and the tokens of a held pass are exactly the first N of the unheld reply."""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import HOLD_KEY  # noqa: E402
from serve.test_parallel import FAKE_BATCH, FakeEngineCase  # noqa: E402


def with_hold(script: str, says_holdb: bool = True) -> str:
    """FAKE_BATCH with the engine's hold (every replaced text must be there once: a change of the fake fails loudly here)."""
    def swap(text, old, new):
        assert text.count(old) == 1, old
        return text.replace(old, new)
    s = swap(script, 'print("READY 4096 stop", flush=True)', 'print("READY 4096 stop hold%s", flush=True)' % (" holdb" if says_holdb else ""))
    s = swap(s, '        ids = f[-1].split(",")\n',
             '        ids = f[-1].split(",")\n'
             '        hold = next(((int(x[5:].split(":")[0]), int(x[5:].split(":")[1])) for x in f if x.startswith("hold=")), None)\n')
    s = swap(s, 'log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}" + (" sr" if "slot_read=1" in f else "") + "\\n"); log.flush()',
             'log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}" + (" sr" if "slot_read=1" in f else "") + '
             'f" hold={hold[0] if hold else None}:{hold[1] if hold else None}\\n"); log.flush()')
    s = swap(s, "        out = 0\n        for t in toks[:n]:\n            if stop.is_set():\n                break\n            print(f\"T {t}\", flush=True)\n            out += 1\n            time.sleep(STEP)\n"
                "        fin = \"stop\" if out and toks[out - 1] == 257 else (\"cancel\" if stop.is_set() else \"length\")\n",
             "        out = 0\n        hold_on = hold is not None\n        for t in toks[:n]:\n            if stop.is_set():\n                break\n"
             "            if hold_on and out >= hold[0]:\n                break\n"
             "            print(f\"T {t}\", flush=True)\n            out += 1\n            if hold_on and t == hold[1]:\n                hold_on = False\n            time.sleep(STEP)\n"
             "        fin = \"stop\" if out and toks[out - 1] == 257 else (\"cancel\" if stop.is_set() else "
             "\"hold\" if hold_on and out >= hold[0] and (out < n or slot is not None) else \"length\")\n")
    s = swap(s, "                active[slot] = [toks[1:], max_new, 1]\n", "                active[slot] = [toks[1:], max_new, 1, hold, hold_on]\n")
    s = swap(s, "        left, max_new, produced = active[b]\n", "        left, max_new, produced, hold, hold_on = active[b]\n")
    s = swap(s, "        print(f\"BT {b} {t}\", flush=True)\n        if b in meta:\n            meta[b][1].append(t)\n"
                "        if t == 257 or produced >= max_new:\n"
                "            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'\n",
             "        print(f\"BT {b} {t}\", flush=True)\n        if b in meta:\n            meta[b][1].append(t)\n"
             "        if hold_on and t == hold[1]:\n            hold_on = False\n"
             "        if t == 257 or produced >= max_new or (hold_on and produced >= hold[0]):\n"
             "            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length' if produced >= max_new else 'hold'\n")
    s = swap(s, "            active[b] = [left, max_new, produced]\n", "            active[b] = [left, max_new, produced, hold, hold_on]\n")
    return s


REPLY = list(b"ok, " + b"la " * 15 + b"done.") + [257]     # what the fake engine says to a prompt with LONGREPLY in it (then <|im_end|>)
NO_ID = 199                                                 # a token the reply never has: the hold stays on
COMMA = ord(",")                                            # the reply's third token: the hold ends there


class HoldBatchCase(FakeEngineCase):
    def start_hold(self, slots=2, says_holdb=True, **kw):
        import serve.test_parallel as tp
        old = tp.FAKE_BATCH
        tp.FAKE_BATCH = with_hold(FAKE_BATCH, says_holdb)
        try:
            self.start(slots, **kw)
        finally:
            tp.FAKE_BATCH = old

    def run_pass(self, text, max_new, hold):
        """One engine pass through generate (the real StrataEngine.generate -> generate_batched): -> (tokens, finish)"""
        ids = list(text.encode())
        cancel = threading.Event()
        sampling = {HOLD_KEY: hold} if hold else {}
        out = [t for t in self.engine.generate(ids, max_new, sampling, cancel) if t is not None]
        return out, (self.engine.last or {}).get("finish")

    def requests(self):
        return [ln.split() for ln in self.log.read_text().splitlines()] if self.log.exists() else []


class SoloAndSlots(HoldBatchCase):
    def test_a_batch_engine_says_holdb_in_ready(self):
        self.start_hold()
        self.assertTrue(self.engine.can_hold)
        self.assertTrue(self.engine.can_hold_batch)

    def test_an_engine_without_holdb_says_so(self):
        self.start_hold(says_holdb=False)
        self.assertTrue(self.engine.can_hold)
        self.assertFalse(self.engine.can_hold_batch)

    def test_old_server_new_engine_the_tokens_are_ignored_and_no_key_means_no_hold(self):
        """Mixed versions, old server + new engine: the server before P2 reads READY with `"stop" in f[2:]` and sends no hold key.  The new
        engine's extra READY tokens change nothing it reads (can_stop, the context), and a request with no key runs as it always did,
        through the solo path and through a slot."""
        self.start_hold()
        self.assertTrue(self.engine.can_stop)
        self.assertEqual(self.engine.max_context, 4096)
        results = {}

        def first():
            results["first"] = self.run_pass("LONGREPLY first", 64, None)

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.15)
        results["second"] = self.run_pass("LONGREPLY second", 64, None)     # admitted into a slot
        t.join(30)
        self.assertEqual(results["first"][0], REPLY)
        self.assertEqual(results["second"][0], REPLY)
        self.assertEqual({results["first"][1], results["second"][1]}, {"stop"})
        self.assertTrue(any(r[0] == "BGEN" for r in self.requests()))
        self.assertTrue(all(r[-1] == "hold=None:None" for r in self.requests()))

    def test_alone_the_solo_path_ends_at_the_hold(self):
        self.start_hold()
        got, finish = self.run_pass("LONGREPLY please", 64, (12, NO_ID))
        self.assertEqual(got, REPLY[:12])
        self.assertEqual(finish, "hold")
        gen = [r for r in self.requests() if r[0] == "GEN"]
        self.assertEqual(len(gen), 1)
        self.assertEqual(gen[0][-1], "hold=12:%d" % NO_ID)

    def test_the_hold_ends_when_its_token_comes_first(self):
        self.start_hold()
        got, finish = self.run_pass("LONGREPLY please", 64, (12, COMMA))
        self.assertEqual(got, REPLY)           # </think> came at the third token: the pass runs on to its own end
        self.assertEqual(finish, "stop")

    def test_no_hold_no_key(self):
        self.start_hold()
        got, finish = self.run_pass("LONGREPLY please", 64, None)
        self.assertEqual(got, REPLY)
        self.assertTrue(all(r[-1] == "hold=None:None" for r in self.requests()))

    def test_a_request_that_goes_to_a_slot_ends_at_the_hold(self):
        """Another request holds the solo path: this one is admitted into a slot (BGEN) and its slot ends with BDONE hold at exactly N."""
        self.start_hold()
        results = {}

        def first():
            results["first"] = self.run_pass("LONGREPLY first", 64, None)

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.15)
        got, finish = self.run_pass("LONGREPLY second", 64, (20, NO_ID))
        t.join(30)
        self.assertEqual(got, REPLY[:20])
        self.assertEqual(finish, "hold")
        self.assertEqual(results["first"][0], REPLY)
        bgen = [r for r in self.requests() if r[0] == "BGEN" and r[-1].startswith("hold=20")]
        self.assertEqual(len(bgen), 1, self.requests())

    def test_a_solo_leg_promoted_to_a_slot_holds_what_is_left(self):
        """The pass starts alone (GEN, hold=N), another request arrives, the pass is stopped and goes on in a slot: its key counts the tokens
        the solo leg already sent, and the pass still ends at exactly N tokens in all."""
        self.start_hold()
        results = {}

        def first():
            results["first"] = self.run_pass("LONGREPLY first", 64, (40, NO_ID))

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.15)
        other = self.run_pass("LONGREPLY other", 64, None)
        t.join(30)
        got, finish = results["first"]
        self.assertEqual(got, REPLY[:40])
        self.assertEqual(finish, "hold")
        self.assertEqual(other[0], REPLY)
        keys = [r[-1] for r in self.requests() if r[0] in ("GEN", "BGEN") and r[-1] != "hold=None:None"]
        self.assertEqual(keys[0], "hold=40:%d" % NO_ID)
        legs = [int(k.split("=")[1].split(":")[0]) for k in keys]
        self.assertTrue(all(n <= 40 for n in legs) and legs == sorted(legs, reverse=True), keys)

    def test_the_tokens_of_a_held_pass_are_the_first_n_of_the_unheld_reply(self):
        self.start_hold()
        for n in (1, 2, 3, 10, 30, 53):
            with self.subTest(n=n):
                got, finish = self.run_pass("LONGREPLY please", 64, (n, NO_ID))
                self.assertEqual(got, REPLY[:n])
                self.assertEqual(finish, "hold")


if __name__ == "__main__":
    unittest.main()
