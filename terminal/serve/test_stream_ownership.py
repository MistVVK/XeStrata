"""P0 (round 0213): streamed output must be delivered only to the request that owns it.

#2892 (unbatched replicas cross DONE ownership) and #4451 (--batch: one client's stream reaches another connection).

The fake engine is the one of test_parallel.py with a canary reply: the reply to a prompt is the prompt's own canary
(two capital letters and 4 digits) repeated, so a token that reaches the wrong request is visible as a foreign canary.  Every
test forces a LEGAL schedule with events or a hooked lock (the way serve/test_replica_lifetime.py of PR #2886 does),
so a failure does not depend on winning a scheduler race.
"""
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.replicas import ReplicaEngine
from serve.server import StrataEngine
from serve.test_parallel import FAKE_BATCH

CANARY_REPEAT = 60
CANARY_RE = re.compile(rb"[A-Z]{2}\d{4}")

# the fake's `continued` (the whole reply, how much of it the prompt ends with) with a canary reply: the last canary
# of the prompt, repeated.  STEP is shortened so the many repetitions stay fast.
CANARY_PATCH = '''
import re as _re
_orig_continued = continued
def continued(ids):
    pb = bytes(int(x) & 255 for x in ids)
    found = _re.findall(rb"[A-Z]{2}\\d{4}", pb)
    if found:
        R = list((found[-1] + b" ") * %d) + [257]
        ints = [int(x) for x in ids]
        return R, max(k for k in range(len(R)) if k == 0 or ints[-k:] == R[:k])
    return _orig_continued(ids)
def reply(ids):
    R, k = continued(ids)
    return R[k:]
''' % CANARY_REPEAT


def canary_script(step=0.004):
    text = FAKE_BATCH.replace("STEP = 0.02", f"STEP = {step}")
    # SLOWPREFILL in a prompt: its read takes a while (a cancel can land inside it).  DEAFSTOP: the engine reads the
    # whole prompt whatever STOP says and then goes on as if it had not come (a drain that does not get its line in time).
    old = "        given = None\n        for q in range(CH, len(ids) - 1, CH):\n            time.sleep(STEP / 2)\n"
    assert old in text
    text = text.replace(old, "        given = None\n        pbm = bytes(int(x) & 255 for x in ids)\n"
                        "        slow, deaf = b\"SLOWPREFILL\" in pbm, b\"DEAFSTOP\" in pbm\n"
                        "        quiet = b\"SILENTREAD\" in pbm\n"
                        "        for q in range(CH, len(ids) - 1, CH):\n"
                        "            time.sleep(STEP / 2 + (0.02 if slow or deaf else 0))\n"
                        "            if quiet and q > CH:\n"
                        "                continue                  # SILENTREAD: after its first PP line the read says nothing\n", 1)
    old = "            if stop.is_set():\n                break\n            if active and (q // CH) % 4 == 0:"
    assert old in text
    text = text.replace(old, "            if stop.is_set() and not deaf:\n                break\n"
                        "            if active and (q // CH) % 4 == 0:", 1)
    old = "        if given:\n            if log:\n                log.write(f\"YIELD"
    assert old in text
    text = text.replace(old, "        if deaf:\n            stop.clear()\n" + old, 1)
    text = text.replace("while True:\n    try:\n        line = lines.get(timeout=STEP if active else None)",
                        CANARY_PATCH + "while True:\n    try:\n        line = lines.get(timeout=STEP if active else None)", 1)
    assert CANARY_PATCH in text
    # VRAM and SAVE / RESTORE answer on the control lines, in the engine's own words
    old = '    if line:\n        print("ERR unknown " + line[:20], flush=True)\n        continue\n'
    assert old in text
    text = text.replace(old, '    if line.startswith("VRAM"):\n        print("VRAM vram_free_mib=100 expert_slots=1", flush=True)\n        continue\n'
                        '    if line.startswith(("SAVE ", "RESTORE ")):\n'
                        '        print("SAVED 1 1 1.0" if line.startswith("SAVE") else "RESTORED 1 1 1.0", flush=True)\n'
                        '        continue\n' + old, 1)
    return text


def prompt_ids(canary: str, filler: int = 0) -> list[int]:
    return list(f"say {canary} ".encode()) + [ord("x")] * filler + list(f" {canary} please".encode())


def expected_reply(canary: str) -> list[int]:
    return list((canary.encode() + b" ") * CANARY_REPEAT) + [257]


def text_of(tokens) -> bytes:
    return bytes(t for t in tokens if 0 <= t < 256)


def foreign(tokens, own: str) -> list[bytes]:
    """The canaries in a token stream that are not `own`."""
    return [c for c in CANARY_RE.findall(text_of(tokens)) if c != own.encode()]


class FakeEngines:
    """Real StrataEngine objects (the reader/pump, the generate paths) over the canary fake process."""

    def make_engines(self, n, batch=0, step=0.004):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(canary_script(step), encoding="utf-8")
        real = server.subprocess.Popen
        # kept for the whole test: restart() starts the fake again
        self.popen_patch = mock.patch.object(server.subprocess, "Popen",
                                             lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen_patch.start()
        engines = []
        for i in range(n):
            extra = (["--batch", str(batch)] if batch else [])
            engines.append(StrataEngine("strata", extra))
        for e in engines:
            e.silence_s = 20
        return engines

    def cleanup_engines(self):
        for e in getattr(self, "engines", []):
            try:
                e.unload()
            except Exception:                                    # noqa: BLE001
                pass
        if getattr(self, "popen_patch", None):
            self.popen_patch.stop()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()


class HookLock:
    """A run lock that calls a hook right after it is released: the successor is then free to take the replica at the
    exact moment the real race needs (before the previous request's child generator has been finalized)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.after_release = None

    def acquire(self, *a, **k):
        return self._lock.acquire(*a, **k)

    def release(self):
        self._lock.release()
        hook, self.after_release = self.after_release, None
        if hook is not None:
            hook()

    def locked(self):
        return self._lock.locked()


def collect(gen, into: list) -> None:
    for t in gen:
        if t is not None:
            into.append(t)


class ReplicaOwnership(FakeEngines, unittest.TestCase):
    """#2892: an unbatched replica takes one request at a time under its run lock.  The lock must cover the previous
    request's whole native exchange (STOP, DONE), and the DONE statistics must belong to the request that read them."""

    def setUp(self):
        self.engines = self.make_engines(2)
        self.rep = ReplicaEngine(self.engines)
        self.rep.heal = lambda: None
        self.engines[1].ended = True          # only replica 0 is alive: every request goes to it
        self.lock = HookLock()
        self.rep.run_locks[0] = self.lock

    def tearDown(self):
        self.cleanup_engines()

    def wait_reading(self, engine, timeout=5.0):
        """True once some thread waits on the engine's line queue (the successor sent its GEN and reads)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if engine.lines.not_empty._waiters:
                return True
            time.sleep(0.001)
        return False

    def run_handoff(self, a_can, b_can, hold_s):
        """A reads two tokens and its consumer closes it; B takes the replica the moment the lock is free and reads
        for `hold_s` seconds before A's child is finalized (so A's tokens are already in the line queue)."""
        a_got, b_got = [], []
        threads = []

        def successor():
            def run():
                collect(self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got)
            t = threading.Thread(target=run, daemon=True)
            threads.append(t)
            t.start()
            self.wait_reading(self.engines[0])
            time.sleep(hold_s)
        self.lock.after_release = successor
        gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
        for t in gen_a:
            if t is not None:
                a_got.append(t)
            if len(a_got) >= 2:
                break
        gen_a.close()                           # what Service does on a stop token, a stop string or a disconnect
        del gen_a
        threads[0].join(30)
        self.assertFalse(threads[0].is_alive())
        return a_got, b_got

    def test_a_successor_never_reads_the_predecessors_done(self):
        """#2892: B is the next request on the replica and A's DONE is still unread."""
        a_can, b_can = "QA0001", "QB0002"
        a_got, b_got = self.run_handoff(a_can, b_can, 0)
        self.assertEqual(foreign(a_got, a_can), [])
        self.assertEqual(foreign(b_got, b_can), [], "B received A's tokens")
        self.assertEqual(b_got, expected_reply(b_can), "B's own reply is complete and exactly its own")

    def test_a_successor_never_reads_the_predecessors_tokens(self):
        """#2892: A's remaining tokens are already queued when B reads."""
        a_can, b_can = "QA0011", "QB0012"
        a_got, b_got = self.run_handoff(a_can, b_can, 0.08)
        self.assertEqual(foreign(b_got, b_can), [], "B received A's tokens")
        self.assertEqual(b_got, expected_reply(b_can), "B's own reply is complete and exactly its own")

    def test_many_forced_handoffs_never_cross(self):
        """The forced schedule above, 30 times with different hold times and different cut-off points."""
        import random
        rnd = random.Random(2892)
        bad = []
        for i in range(30):
            a_can, b_can = f"QA{i:04d}", f"QB{i:04d}"
            self.lock.after_release = None
            cut = rnd.choice((1, 2, 3, 10, 40))
            hold = rnd.choice((0, 0.002, 0.01, 0.05))
            a_got, b_got = [], []
            threads = []

            def successor(b_can=b_can, b_got=b_got, hold=hold, threads=threads):
                t = threading.Thread(target=lambda: collect(
                    self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got), daemon=True)
                threads.append(t)
                t.start()
                self.wait_reading(self.engines[0])
                time.sleep(hold)
            self.lock.after_release = successor
            gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
            for t in gen_a:
                if t is not None:
                    a_got.append(t)
                if len(a_got) >= cut:
                    break
            gen_a.close()
            del gen_a
            threads[0].join(30)
            if foreign(b_got, b_can) or b_got != expected_reply(b_can):
                bad.append((i, cut, hold, len(b_got), foreign(b_got, b_can)[:2]))
        self.assertEqual(bad, [], f"{len(bad)} of 30 handoffs crossed")

    def test_unforced_back_to_back_requests_never_cross(self):
        """No hook: B waits for the replica's lock while A is cut off at a random token; 60 repetitions."""
        import random
        rnd = random.Random(4451)
        self.rep.run_locks[0] = threading.Lock()
        bad = []
        for i in range(60):
            a_can, b_can = f"QA{i:04d}", f"QB{i:04d}"
            cut = rnd.choice((1, 2, 3, 5, 20, 100))
            a_got, b_got = [], []
            go = threading.Event()
            tb = threading.Thread(target=lambda: (go.wait(), collect(
                self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got)), daemon=True)
            tb.start()
            gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
            for t in gen_a:
                if t is not None:
                    a_got.append(t)
                if len(a_got) >= cut:
                    break
            go.set()
            gen_a.close()
            del gen_a
            tb.join(30)
            if foreign(b_got, b_can) or b_got != expected_reply(b_can):
                bad.append((i, cut, len(b_got), foreign(b_got, b_can)[:2]))
        self.assertEqual(bad, [], f"{len(bad)} of 60 crossed")

    def test_the_figures_of_a_request_are_its_own(self):
        """A runs to its max_tokens; before A's caller reads its DONE figures, B runs to completion on the replica."""
        a_can, b_can = "QA0003", "QB0004"
        done = []

        def successor():
            def run():
                collect(self.rep.generate(prompt_ids(b_can, 40), 25, {}, threading.Event()), [])
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(30)
            done.append(t)
        self.lock.after_release = successor
        got = []
        collect(self.rep.generate(prompt_ids(a_can), 10, {}, threading.Event()), got)
        self.assertTrue(done, "the successor ran")
        last = self.rep.last
        self.assertEqual(last.get("generated"), 10, last)
        self.assertEqual(last.get("prompt_tokens"), len(prompt_ids(a_can)), last)

    def test_an_undrainable_replica_is_not_handed_to_the_next_request(self):
        """The child cannot be drained (the engine never answers STOP): the replica is ended, not reused."""
        from serve.server import EngineSilent
        a_can = "QA0005"
        engine = self.engines[0]
        engine.silence_s = 1.0
        gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
        got = []
        for t in gen_a:
            if t is not None:
                got.append(t)
            if len(got) >= 2:
                break
        engine.can_stop = True
        # the engine stops answering: its process is frozen
        import os
        import signal
        if hasattr(signal, "SIGSTOP"):
            os.kill(engine.proc.pid, signal.SIGSTOP)
        else:
            self.skipTest("needs SIGSTOP to freeze the fake engine")
        try:
            try:
                gen_a.close()
            except EngineSilent:
                pass
        finally:
            if engine.proc.poll() is None:
                os.kill(engine.proc.pid, signal.SIGCONT)
        self.assertFalse(engine.alive(), "the replica that could not be drained is out of the routing")
        self.assertFalse(self.lock.locked(), "and its run lock is free")
        self.assertEqual(self.rep.route(prompt_ids("QA0006")), -1)


class Client(threading.Thread):
    """One streaming client of a --batch engine: its own canary, its own cancel Event.  `pause_after`/`resume`: the
    consumer stops reading after that many tokens until `resume` is set (a slow reader)."""

    def __init__(self, engine, canary, filler=0, max_new=500, pause_after=None, resume=None, extra="", leave_after_beats=None):
        super().__init__(daemon=True)
        self.leave_after_beats = leave_after_beats          # close the stream (as Service does) at this PP heartbeat
        self.engine, self.canary = engine, canary
        self.ids = prompt_ids(canary + extra, filler)
        self.max_new, self.pause_after, self.resume = max_new, pause_after, resume
        self.cancel = threading.Event()
        self.tokens, self.error, self.paused = [], None, threading.Event()

    def run(self):
        beats = 0
        gen = self.engine.generate(self.ids, self.max_new, {}, self.cancel)
        try:
            for t in gen:
                if t is None:
                    beats += 1
                    if self.leave_after_beats is not None and beats >= self.leave_after_beats:
                        gen.close()                              # the consumer leaves: GeneratorExit inside the engine's request
                        return
                    continue
                self.tokens.append(t)
                if self.pause_after is not None and len(self.tokens) == self.pause_after:
                    self.paused.set()
                    self.resume.wait(30)
        except BaseException as e:                               # noqa: BLE001
            self.error = e

    def foreign(self):
        return foreign(self.tokens, self.canary)

    def complete(self):
        return self.tokens == expected_reply(self.canary)


def wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


class BatchOwnership(FakeEngines, unittest.TestCase):
    """#4451: a --batch engine.  Every frame a request reads must be its own, whatever happened to the requests before
    it on the same slot, the same control lines or the same engine process."""

    def setUp(self):
        self.engines = self.make_engines(1, batch=4)
        self.engine = self.engines[0]
        # 0.3 s instead of 300 s for the wait that ends an interrupted exchange (a wrapper that works on every version)
        orig = self.engine._drain_control
        self.engine._drain_control = lambda until, timeout=None, born=None, lines_q=None: orig(
            until, timeout=0.3 if timeout is None else timeout, born=born,
            **({} if lines_q is None else {"lines_q": lines_q}))
        self.engine.QUARANTINE_S = 8.0

    def tearDown(self):
        self.cleanup_engines()

    def start_busy(self, n=1, tag="Y"):
        """Long requests that hold batch slots, so that the next requests are admitted into slots too."""
        bg = [Client(self.engine, f"{chr(ord(tag) + i)}B{9990 - i:04d}", max_new=500) for i in range(n)]
        for c in bg:
            c.start()
            time.sleep(0.05)
        wait_until(lambda: all(len(c.tokens) >= 3 for c in bg))
        return bg

    def finish(self, clients, timeout=60):
        for c in clients:
            c.join(timeout)
            self.assertFalse(c.is_alive())

    def test_interleaved_clients_get_their_own_replies(self):
        cs = [Client(self.engine, f"{chr(ord('A') + i)}{chr(ord('K') + i)}{i:04d}") for i in range(4)]
        for c in cs:
            c.start()
            time.sleep(0.02)
        self.finish(cs)
        for c in cs:
            self.assertIsNone(c.error)
            self.assertEqual(c.foreign(), [])
            self.assertTrue(c.complete(), c.canary)

    def test_a_cancel_in_the_middle_of_the_prompt_read_then_reuse(self):
        """The first of several interleaved clients is cancelled while its prompt is being read; the next requests
        reuse the slot and the control lines and must read only their own frames."""
        import random
        rnd = random.Random(4451)
        bg = self.start_busy(n=2)
        bad = []
        for i in range(12):
            a = Client(self.engine, f"KA{i:04d}", filler=700, extra=" SLOWPREFILL")
            a.start()
            time.sleep(rnd.choice((0.03, 0.08, 0.15, 0.3)))
            a.cancel.set()
            followers = [Client(self.engine, f"{chr(ord('M') + j)}{chr(ord('Q') + j)}{i:02d}{j:02d}", max_new=60)
                         for j in range(2)]
            for f in followers:
                f.start()
            self.finish([a] + followers)
            for f in followers:
                if f.error or f.foreign() or f.tokens != expected_reply(f.canary)[:60]:
                    bad.append((i, f.canary, repr(f.error), f.foreign()[:2], len(f.tokens)))
            self.assertEqual(a.foreign(), [])
        for c in bg:
            c.cancel.set()
        self.finish(bg)
        for c in bg:
            self.assertEqual(c.foreign(), [], "a background request received a foreign frame")
        self.assertEqual(bad, [])

    def test_a_drain_that_times_out_does_not_leak_into_the_next_request(self):
        """A leaves (its stream is closed, as Service does) in the middle of its prompt read; the engine says nothing for
        longer than the drain limit (SILENTREAD, DEAFSTOP: ~1 s of silence, then its DONE and BADM).  The control lines stay
        with A's exchange until they come; the request after it gets its own frames only."""
        bg = self.start_busy(n=1)
        a = Client(self.engine, "KA0001", filler=1300, extra=" DEAFSTOP SILENTREAD", leave_after_beats=1)
        a.start()
        time.sleep(0.6)                          # past the 0.3 s drain: A's request has left, its engine exchange has not ended
        # (B's prompt is not under half of A's: a much shorter waiting request would make A give way, BYIELD, at once)
        b = Client(self.engine, "LB0002", filler=900, max_new=80)
        b.start()
        self.finish([a, b])
        for c in bg:
            c.cancel.set()
        self.finish(bg)
        self.assertIsNone(b.error)
        self.assertEqual(b.foreign(), [], "B received A's frames")
        self.assertEqual(b.tokens, expected_reply("LB0002")[:80], "B's reply is its own, in full")
        self.assertEqual(bg[0].foreign(), [])
        self.assertTrue(self.engine.alive())

    def test_an_engine_that_keeps_talking_after_a_stop_is_not_given_up_on(self):
        """The limits count silence, never a read or a decode that still produces lines: the engine takes ~1.8 s to end A's
        exchange (DEAFSTOP, a PP line every chunk) with the drain at 0.3 s and the quarantine at 0.4 s.  A limit counted
        from the start would have ended the engine at ~0.7 s, with the other requests in it."""
        self.engine.QUARANTINE_S = 0.4
        gen_before = self.engine.gen
        bg = self.start_busy(n=1)
        a = Client(self.engine, "KA0001", filler=2600, extra=" DEAFSTOP", leave_after_beats=1)
        a.start()
        time.sleep(0.1)
        b = Client(self.engine, "LB0002", filler=1500, max_new=80)       # (not under half of A's prompt: no BYIELD)
        b.start()
        self.finish([a, b], timeout=60)
        self.assertEqual(self.engine.gen, gen_before, "the engine was ended although it kept talking")
        self.assertTrue(self.engine.alive())
        self.assertIsNone(b.error)
        self.assertEqual(b.tokens, expected_reply("LB0002")[:80])
        self.assertEqual(bg[0].foreign(), [])
        self.assertIsNone(bg[0].error, "a request running in another slot was ended with the engine")
        bg[0].cancel.set()
        self.finish(bg)

    def test_an_engine_that_goes_silent_after_a_stop_is_ended(self):
        """The other side of the limit: no line at all for DRAIN_TIMEOUT_S + QUARANTINE_S after the stop (here 0.3 + 0.4 s,
        the read is silent for ~1.8 s) is a wedged engine: it is ended and the next request starts it again."""
        self.engine.QUARANTINE_S = 0.4
        a = Client(self.engine, "KA0001", filler=2600, extra=" DEAFSTOP SILENTREAD", leave_after_beats=1)
        a.start()
        self.assertTrue(wait_until(lambda: not self.engine.alive(), 8.0), "a silent engine was not ended")
        self.finish([a])

    def test_the_engine_ending_during_a_drain_then_a_restart(self):
        """A is interrupted mid-prompt; the engine dies while A drains; after the restart the new requests (same slot
        numbers) must never be read by a request of the old process."""
        old = self.start_busy(n=2)
        slow = [Client(self.engine, f"{chr(ord('S') + i)}{chr(ord('T') + i)}{i:04d}", pause_after=3,
                       resume=threading.Event()) for i in range(2)]
        for c in slow:
            c.start()
            time.sleep(0.05)
        self.assertTrue(wait_until(lambda: all(c.paused.is_set() for c in slow)))
        a = Client(self.engine, "KA0001", filler=700, extra=" SLOWPREFILL")
        a.start()
        time.sleep(0.1)
        a.cancel.set()
        self.engine.proc.kill()                  # the engine ends while A's exchange is being drained
        self.engine.proc.wait()
        self.finish([a] + old)
        self.engine.restart()
        self.assertTrue(self.engine.alive())
        new = self.start_busy(n=2, tag="D")      # the new process: the same slot numbers again
        for c in slow:
            c.resume.set()
        # they read on: the old process's queued frames, then its end - never the new process's frames
        time.sleep(1.0)
        for c in slow:
            self.assertEqual(c.foreign(), [], f"{c.canary} (old process) read a frame of the new one")
        for c in new:
            c.cancel.set()
        self.finish(new)
        self.finish(slow)
        for c in slow:
            self.assertIsInstance(c.error, Exception, "a request of the dead engine ends with an error")
        for c in new:
            self.assertEqual(c.foreign(), [], "a request of the new process received a foreign frame")
            self.assertTrue(len(c.tokens) >= 3)

    def test_long_generations_are_never_cut_by_the_drain_limits(self):
        """The drain and quarantine limits (here 0.3 s and 0.4 s, far below a reply's ~1.7 s) apply after a STOP or BSTOP
        only: replies that are still streaming complete."""
        self.engine.QUARANTINE_S = 0.4
        cs = [Client(self.engine, f"{chr(ord('A') + i)}{chr(ord('K') + i)}{i:04d}") for i in range(4)]
        for c in cs:
            c.start()
            time.sleep(0.05)
        self.finish(cs)
        for c in cs:
            self.assertIsNone(c.error)
            self.assertTrue(c.complete(), c.canary)

    def test_vram_and_session_file_take_the_control_lines_like_a_request(self):
        """VRAM and SAVE read the control lines for their answer.  Run beside a request that holds them (its solo read
        or an admission), they took that request's T / PP / DONE lines and it lost its reply."""
        for what in ("vram", "save", "restore"):
            a = Client(self.engine, "KA0001")
            a.start()
            self.assertTrue(wait_until(lambda: len(a.tokens) >= 3))
            answers = []

            def call(what=what):
                try:
                    if what == "vram":
                        answers.append(self.engine.vram(None))
                    else:
                        answers.append(self.engine.session_file(what, "/tmp/x"))
                except BaseException as e:               # noqa: BLE001
                    answers.append(e)
            t = threading.Thread(target=call, daemon=True)
            t.start()
            self.finish([a])
            t.join(30)
            self.assertFalse(t.is_alive(), what)
            self.assertIsNone(a.error, what)
            self.assertTrue(a.complete(), f"{what}: the request lost lines to the VRAM / session call "
                                          f"({len(a.tokens)} of {len(expected_reply('KA0001'))} tokens)")
            self.assertIsInstance(answers[0], dict, answers)
            self.assertEqual(answers[0].get("vram_free_mib") if what == "vram" else answers[0].get("tokens"),
                             100 if what == "vram" else 1)

    def test_a_slot_is_not_handed_on_before_the_engine_confirms(self):
        """A consumer leaves a slot's stream; its BDONE has not come.  Whatever frames still arrive for that slot are
        not the next request's."""
        bg = self.start_busy(n=1)
        a = Client(self.engine, "KA0001", max_new=500)
        a.start()
        self.assertTrue(wait_until(lambda: len(a.tokens) >= 5))
        a.cancel.set()                           # BSTOP: A's slot ends at its next window
        followers = [Client(self.engine, f"{chr(ord('M') + j)}{chr(ord('Q') + j)}{j:04d}", max_new=60) for j in range(3)]
        for f in followers:
            f.start()
        self.finish([a] + followers)
        for f in followers:
            self.assertEqual(f.foreign(), [])
            self.assertEqual(f.tokens, expected_reply(f.canary)[:60])
        bg[0].cancel.set()
        self.finish(bg)


if __name__ == "__main__":
    unittest.main()
