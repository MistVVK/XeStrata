"""A request whose consumer stops early sends the engine STOP from Engine.generate's `finally`.  When the engine is being
ended or restarted on another thread at that moment, its stdin can already be closed (Engine.close() closes it before it
clears `proc`) or `proc` can already be None.  The stop path must not raise in any of these states: the exception would
replace the request's own outcome and leave the generator's close() with an error (serve/replicas.py's gen.close(); seen
as the "'NoneType' object has no attribute 'stdin'" warning in serve/test_stream_ownership.py)."""

from __future__ import annotations

import io
import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


class FakeProc:
    def __init__(self):
        self.stdin = io.StringIO()

    def poll(self):
        return None


def engine() -> StrataEngine:
    e = StrataEngine.__new__(StrataEngine)
    e.alive = lambda: True
    e.batch, e.multi, e.silence_s, e.can_stop = 0, 0, 0, True
    e.lines = queue.Queue()
    e.proc = FakeProc()
    e.sampling_keys = lambda sampling: ""
    e.exit_code = lambda: 1
    return e


class StopOnAClosedPipe(unittest.TestCase):
    def run_case(self, break_pipe):
        e = engine()
        e.lines.put("T 7")
        with mock.patch.object(server.bias_api, "validate_request", lambda *a, **k: None), \
                mock.patch.object(server, "refuse_for_split", lambda *a, **k: None), \
                mock.patch.object(server, "ENGINE_STALL_S", 0):
            gen = e.generate([1, 2, 3], 8, {}, threading.Event())
            self.assertEqual(next(gen), 7)
            break_pipe(e)
            e.lines.put(None)                    # the engine's output ended (its reader thread's end mark)
            gen.close()                          # must not raise, whatever state the pipe is in

    def test_stdin_closed(self):
        self.run_case(lambda e: e.proc.stdin.close())          # ValueError: I/O operation on closed file

    def test_proc_gone(self):
        def gone(e):
            e.proc = None                                        # AttributeError: 'NoneType' ... 'stdin'
        self.run_case(gone)

    def test_broken_pipe_still_quiet(self):
        def broken(e):
            def write(_):
                raise BrokenPipeError(32, "Broken pipe")
            e.proc.stdin.write = write
        self.run_case(broken)

    def test_open_pipe_gets_the_stop(self):
        e = engine()
        e.lines.put("T 7")
        with mock.patch.object(server.bias_api, "validate_request", lambda *a, **k: None), \
                mock.patch.object(server, "refuse_for_split", lambda *a, **k: None), \
                mock.patch.object(server, "ENGINE_STALL_S", 0):
            gen = e.generate([1, 2, 3], 8, {}, threading.Event())
            self.assertEqual(next(gen), 7)
            e.lines.put("DONE 1 3 1.0 1.0 stop")
            gen.close()
        self.assertTrue(e.proc.stdin.getvalue().endswith("STOP\n"))


if __name__ == "__main__":
    unittest.main()
