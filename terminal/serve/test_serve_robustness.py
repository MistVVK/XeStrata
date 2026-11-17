"""Smaller serve-side robustness fixes of round 0213 (each test fails on the base a3a10b90a):

#3443 ConvCacheLog.poll without a lock, #4792 a failed engine start when another thread closed the process,
#4738 lost wake-ups of the control-line queue, #5044 a byte that is not UTF-8 on the engine's stdout.
The fake engine and helpers are those of test_stream_ownership.py.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine
from serve.test_stream_ownership import FakeEngines, canary_script, collect, expected_reply, prompt_ids


class ConvCachePoll(unittest.TestCase):
    """ConvCacheLog.poll ran outside any lock: a second poll that moved `pos` between the first one's size check and
    its read left a negative read length (ValueError).  Found by HWSTALL while triaging #3443."""
    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")

    def test_a_poll_that_overlaps_another_one_does_not_raise(self):
        import os
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK, encoding="utf-8")
            c = ConvCacheLog()
            c.poll(str(log), 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK)
            first_sized, first_may_go = threading.Event(), threading.Event()
            real_size = os.path.getsize
            calls = []

            def slow_first(path):
                size = real_size(path)
                if not calls and path == str(log):
                    calls.append(1)
                    first_sized.set()
                    first_may_go.wait(2.0)             # the first poll has its size; a second poll runs meanwhile
                return size
            errors = []

            def first():
                try:
                    c.poll(str(log), 0)
                except BaseException as e:             # noqa: BLE001
                    errors.append(e)
            with mock.patch("serve.server.os.path.getsize", slow_first):
                t = threading.Thread(target=first, daemon=True)
                t.start()
                self.assertTrue(first_sized.wait(5))
                with open(log, "a", encoding="utf-8") as f:
                    f.write(self.PARK * 2)
                second = threading.Thread(target=lambda: c.poll(str(log), 0), daemon=True)
                second.start()
                second.join(0.5)                       # (with the lock it waits for the first one)
                first_may_go.set()
                t.join(5)
                second.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(c.poll(str(log), 0)["parks"], 4, "every parked line counted once")


class FailedStart(unittest.TestCase):
    """#4792 (PR by a community author; the fix is written in our tree): a start that ends before READY must not crash
    when another thread closed the process (set self.proc to None) while the failed start cleaned up."""

    def test_close_from_another_thread_during_a_failed_start(self):
        import serve.server as server
        holder = {}

        class Pipe:
            def close(self):
                pass

            def __iter__(self):
                return iter(())                      # the engine's output ends: no READY

        class Proc:
            returncode = 1
            stdin, stdout = Pipe(), Pipe()
            pid = 0

            def wait(self, timeout=None):
                holder["engine"].proc = None         # restart()'s close() on another thread
                return 1

            def poll(self):
                return 1

            def kill(self):
                pass

        engine = StrataEngine.__new__(StrataEngine)
        holder["engine"] = engine
        with mock.patch.object(server, "popen", lambda *a, **k: Proc()), \
                mock.patch.object(server, "end_leftover_engines", lambda exe: None), \
                mock.patch.object(server, "transient_start_failure", lambda *a, **k: False):
            with self.assertRaises(server.EngineDied):
                engine.__init__("strata", [], retry_args=None)


class ControlNotify(unittest.TestCase):
    """#4738 (the PR's approach, written in our tree): a request that waits for the control lines must not miss a
    notification sent between its check of the turn and its wait, and a waiter that leaves tells the others."""

    class Cond(threading.Condition):
        def __init__(self):
            super().__init__()
            self.exits, self.after_check, self.waits, self.notifications = 0, None, [], 0

        def __exit__(self, *args):
            result = super().__exit__(*args)
            self.exits += 1
            if self.exits == 2 and self.after_check is not None:     # registered (1), turn read (2): now the producer
                hook, self.after_check = self.after_check, None
                hook()
            return result

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return True                                              # (never blocks: the test is the schedule)

        def notify_all(self):
            self.notifications += 1
            return super().notify_all()

    def engine(self):
        e = StrataEngine.__new__(StrataEngine)
        e.slot_cv, e.waiting, e.admitting = self.Cond(), 1, 0        # one other request is waiting
        e.wait_lens, e.arrivals, e.ctl_epoch, e.ctl = [], __import__("collections").deque(maxlen=64), 0, threading.Lock()
        e.proc, e.ended, e.starting = mock.Mock(**{"poll.return_value": None}), False, False
        return e

    def test_a_notification_between_the_check_and_the_wait_is_not_lost(self):
        e = self.engine()

        def producer():                                              # another request took the lines and gave them back
            with e.slot_cv:
                e.ctl_epoch += 1
                e.slot_cv.notify_all()
        e.slot_cv.after_check = producer
        gen = e._take_control(threading.Event(), 10, after_epoch=0)
        with self.assertRaises(StopIteration) as done:
            for _ in range(5):
                next(gen)
        self.assertTrue(done.exception.value)
        self.assertEqual(e.slot_cv.waits, [], "it waited although its turn had come in the meantime")

    def test_a_waiter_that_leaves_notifies_the_others(self):
        e = self.engine()
        e.ctl.acquire()                                              # the lines are taken: this one cannot have them
        cancel = threading.Event()
        cancel.set()
        gen = e._take_control(cancel, 10)
        with self.assertRaises(StopIteration) as done:
            for _ in range(3):
                next(gen)
        self.assertFalse(done.exception.value)
        self.assertGreaterEqual(e.slot_cv.notifications, 1)
        self.assertEqual(e.waiting, 1)


class StalledClient(unittest.TestCase):
    """#4271: a streaming client that stops reading, without closing its connection, must not hold the engine's turn
    for ever: the write that makes no progress ends the request as a disconnect and the next request is served."""

    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        import json
        import socket
        import time
        import urllib.request
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)   # (not one repeated token)
        svc = Service(engine, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and it never reads.  Wait until the server is writing into a full buffer, then until it lets go.
                end = time.monotonic() + 3
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()


    def test_a_client_that_reads_but_waits_for_a_slow_engine_is_not_cut_off(self):
        """The write limit is for a write that makes no progress (the client stopped reading), not for a client that is
        idle while the engine thinks: 0.3 s between tokens, a 0.1 s write limit, the whole reply arrives."""
        import http.client
        import json
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "abcdef", delay_s=0.3), tok,
                      ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.1, create=True):
                conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
                conn.request("POST", "/v1/chat/completions", json.dumps(
                    {"messages": [{"role": "user", "content": "go"}], "stream": True, "max_tokens": 50,
                     "reasoning_effort": "none"}), {"Content-Type": "application/json"})
                body = conn.getresponse().read().decode()
                conn.close()
            text = "".join(json.loads(l[5:])["choices"][0]["delta"].get("content") or ""
                           for l in body.splitlines() if l.startswith("data:") and "[DONE]" not in l
                           and json.loads(l[5:]).get("choices"))
            self.assertEqual(text, "abcdef")
            self.assertIn("data: [DONE]", body)
        finally:
            httpd.shutdown()
            httpd.server_close()


class BlockedConsole(unittest.TestCase):
    """#4479 / #2481: a console that stops taking output must not hold the thread that prints."""

    def pipe(self):
        import os
        r, w = os.pipe()
        self.addCleanup(lambda: [os.close(x) for x in (r,) if True])
        return r, os.fdopen(w, "w", encoding="utf-8", buffering=1)

    def test_a_print_to_a_full_pipe_blocks_without_the_wrapper(self):
        """The control: this is the dependency the request lifecycle had on the console."""
        r, w = self.pipe()
        done = threading.Event()

        def spam():
            for i in range(20000):
                w.write(f"[strata] progress line {i} " + "x" * 40 + "\n")
                w.flush()
            done.set()
        threading.Thread(target=spam, daemon=True).start()
        self.assertFalse(done.wait(1.0), "a pipe nobody reads must block a plain print")

    def test_the_wrapper_never_blocks_and_keeps_the_newest_lines(self):
        import os
        from serve.nonblocking_out import NonBlockingOut
        r, w = self.pipe()
        out = NonBlockingOut(w, max_bytes=64 * 1024, max_chunks=500)
        done = threading.Event()

        def spam():
            for i in range(20000):
                print(f"[strata] progress line {i} " + "x" * 40, file=out, flush=True)
            done.set()
        threading.Thread(target=spam, daemon=True).start()
        self.assertTrue(done.wait(5.0), "printing into a console nobody reads must not block")
        self.assertGreater(out.dropped_chunks, 0)
        self.assertLessEqual(len(out._q), 500)
        # the console wakes up: everything that was kept arrives, in order, ending with the newest line
        data = b""
        end = __import__("time").monotonic() + 10
        while __import__("time").monotonic() < end and b"progress line 19999" not in data:
            data += os.read(r, 65536)
        text = data.decode()
        self.assertIn("progress line 19999 ", text)
        seen = [int(l.split()[3]) for l in text.splitlines() if l.startswith("[strata] progress line")]
        self.assertEqual(seen, sorted(seen), "order is kept")
        out.close(1.0)

    def test_the_console_is_restored_and_drained(self):
        import io
        from serve.nonblocking_out import NonBlockingOut
        real = io.StringIO()
        out = NonBlockingOut(real)
        out.write("hello\n")
        out.write("world\n")
        self.assertTrue(out.drain(2.0))
        self.assertEqual(real.getvalue(), "hello\nworld\n")
        out.close()
        self.assertEqual(out.write("late\n"), 5)                     # a late write after close is not an error


class EngineOutputDecoding(FakeEngines, unittest.TestCase):
    """#5044: a byte that is not UTF-8 on the engine's stdout (Windows can print one) must not end the reader."""

    def tearDown(self):
        self.cleanup_engines()

    def test_a_non_utf8_byte_before_ready_and_in_a_reply(self):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        text = canary_script()
        bad = 'sys.stdout.buffer.write(b"INFO note=\\x8e\\n"); sys.stdout.flush()\nprint("READY 4096 stop", flush=True)'
        self.assertIn('print("READY 4096 stop", flush=True)', text)
        text = text.replace('print("READY 4096 stop", flush=True)', bad, 1)
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(text, encoding="utf-8")
        real = server.subprocess.Popen
        self.popen_patch = mock.patch.object(server.subprocess, "Popen",
                                             lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen_patch.start()
        engine = StrataEngine("strata", [])
        self.engines = [engine]
        self.assertTrue(engine.alive())
        got = []
        collect(engine.generate(prompt_ids("KA0001"), 500, {}, threading.Event()), got)
        self.assertEqual(got, expected_reply("KA0001"))


if __name__ == "__main__":
    unittest.main()
