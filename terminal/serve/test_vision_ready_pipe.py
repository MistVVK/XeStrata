"""The picture reader's READY pipe (serve/server.py Vision._start_reader / _await_ready).

A crashing reader can start a debugger (ggml's abort path forks gdb) and the debugger writes to the reader's stdout, the pipe the server reads the
READY line from.  The first line of that pipe is then not READY.  The server used to kill the reader on any first line that was not READY, while the
debugger was still attached to it.  Now: READY and ERR lines are answers; any other line is a stray line, kept and logged, and the wait goes on with
ONE deadline.  A reader that is only printing is never killed.  A reader whose process has ended counts as ended after READY_GRACE_S even when a
debugger or a child still holds the pipe open.  The readers here are small Python scripts run as real processes.

    python -m serve.test_vision_ready_pipe
"""
import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server


def plain_popen(what, args, **kw):                 # the real process, without the lifeline's anchor and watcher
    kw.pop("role", None)
    return subprocess.Popen(args, **kw)


class ReadyPipe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.procs = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
            for stream in (p.stdin, p.stdout):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

    def reader(self, body: str):
        """A Vision whose reader is a script with `body`; returns it (nothing started yet)."""
        script = Path(self.tmp.name) / ("reader%d.py" % len(list(Path(self.tmp.name).glob("reader*.py"))))
        script.write_text("import os, subprocess, sys, time\n" + textwrap.dedent(body), encoding="utf-8")
        v = server.Vision.__new__(server.Vision)
        v.spawn = ([sys.executable, "-u", str(script)], None, None)
        v.dir = Path(self.tmp.name)
        v.stopped, v.proc = True, None
        return v

    def start(self, v):
        """v._start_reader() with the process kept for cleanup; (the error or None, the text it printed, seconds taken)."""
        out, error, t0 = io.StringIO(), None, time.monotonic()
        real = plain_popen

        def keep(what, args, **kw):
            p = real(what, args, **kw)
            self.procs.append(p)
            return p

        with mock.patch.object(server, "popen", keep), contextlib.redirect_stdout(out):
            try:
                v._start_reader()
            except RuntimeError as e:
                error = e
        return error, out.getvalue(), time.monotonic() - t0

    def test_a_stray_line_before_READY_does_not_kill_the_reader(self):
        v = self.reader("""
            print("[New LWP 477206]", flush=True)
            time.sleep(1.0)
            print("READY 2560", flush=True)
            sys.stdin.readline()
        """)
        error, said, _ = self.start(v)
        self.assertIsNone(error)
        self.assertFalse(v.stopped)
        self.assertIsNone(v.proc.poll(), "a reader that was only printing is still running")
        self.assertIn("[New LWP 477206]", said)                      # logged
        self.assertIn("printed on its output", said)

    def test_the_encode_answers_after_READY_are_not_taken_by_the_start(self):
        v = self.reader("""
            print("noise", flush=True)
            print("READY 2560", flush=True)
            print("OK 7 1 1 1", flush=True)
            sys.stdin.readline()
        """)
        error, _, _ = self.start(v)
        self.assertIsNone(error)
        self.assertEqual(v._readline(5.0, "x"), "OK 7 1 1 1\n")

    def test_many_stray_lines_are_bounded_in_the_log_and_the_message(self):
        v = self.reader("""
            for i in range(200):
                print("stray %d" % i, flush=True)
            sys.exit(1)
        """)
        error, said, _ = self.start(v)
        self.assertIsNotNone(error)
        self.assertLessEqual(said.count("printed on its output"), server.Vision.STRAY_KEEP)
        self.assertIn("stray 199", str(error))                       # the last ones are in the message
        self.assertNotIn("stray 0 ", str(error) + " ")

    def test_a_stray_line_then_the_end_names_the_exit_and_the_line(self):
        v = self.reader("""
            print("[New LWP 5]", flush=True)
            sys.exit(1)
        """)
        error, _, took = self.start(v)
        self.assertIsNotNone(error)
        text = str(error)
        self.assertIn("did not start", text)
        self.assertIn("exit code 1", text)
        self.assertIn("[New LWP 5]", text)
        self.assertLess(took, 10)

    def test_a_stray_line_then_ERR_gives_the_ERR_line(self):
        v = self.reader("""
            print("[New LWP 5]", flush=True)
            print("ERR cannot load the vision encoder x.gguf", flush=True)
            sys.exit(1)
        """)
        error, _, _ = self.start(v)
        text = str(error)
        self.assertIn("ERR cannot load the vision encoder", text)
        self.assertIn("[New LWP 5]", text)

    def test_a_reader_that_only_prints_is_ended_at_the_deadline_with_its_lines_in_the_message(self):
        v = self.reader("""
            print("[New LWP 5]", flush=True)
            time.sleep(60)
        """)
        with mock.patch.object(server, "VISION_READY_S", 1.5):
            error, _, took = self.start(v)
        self.assertIsNotNone(error)
        self.assertIn("said nothing for", str(error))
        self.assertIn("[New LWP 5]", str(error))
        self.assertGreaterEqual(took, 1.4)
        self.assertLess(took, 8)
        self.assertIsNotNone(self.procs[-1].wait(timeout=10), "it was killed at the deadline (not before: the wait above took the whole deadline)")
        self.assertTrue(v.stopped)
        self.assertIsNone(v.proc)

    def test_a_reader_that_ended_while_a_child_holds_its_pipe_is_judged_after_the_grace(self):
        v = self.reader("""
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], stdout=sys.stdout)     # the "debugger": holds the pipe
            print("[New LWP 5]", flush=True)
            os._exit(3)
        """)
        with mock.patch.object(server.Vision, "READY_GRACE_S", 0.5):
            error, _, took = self.start(v)
        self.assertIsNotNone(error)
        self.assertIn("exit code 3", str(error))
        self.assertLess(took, 4.5, "the pipe's end (5 s) is not waited for: the process's end is what counts")

    def test_a_silent_reader_that_dies_is_the_same_as_before(self):
        v = self.reader("""
            sys.exit(7)
        """)
        error, _, _ = self.start(v)
        self.assertIn("exit code 7", str(error))
        self.assertIn("without saying anything", str(error))

    def test_the_cause_in_a_stray_line_is_named(self):
        v = self.reader("""
            print("cudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable", flush=True)
            sys.exit(1)
        """)
        error, _, _ = self.start(v)
        self.assertIn("Exclusive_Process", str(error))

    def test_no_deadline_means_wait_for_ever_not_fail_at_once(self):
        v = self.reader("""
            time.sleep(0.6)
            print("READY 2560", flush=True)
            sys.stdin.readline()
        """)
        with mock.patch.object(server, "VISION_READY_S", None):
            error, _, _ = self.start(v)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
