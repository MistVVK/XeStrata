"""serve/test_fatal_gpu_restart.py: #4350.  After a fatal GPU error the DeepSeek engine ends itself (ERR fatal GPU error: ...) and the
server starts it again on the next request; three such deaths within ten minutes stop the restarts with a plain 503 text.
CPU only, no engine: the server's bookkeeping is called on a stand-in."""
import sys
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402


def stand_in(pid, last_err):
    return types.SimpleNamespace(engine=types.SimpleNamespace(last_err=last_err, proc=types.SimpleNamespace(pid=pid)))


class FatalGpuRestart(unittest.TestCase):
    def note(self, svc, code=3):
        S.Service._note_fatal_gpu_exit.__get__(svc)(code)

    def make(self, pid, err="fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered"):
        svc = stand_in(pid, err)
        svc.FATAL_GPU_MAX = S.Service.FATAL_GPU_MAX
        svc.FATAL_GPU_WINDOW_S = S.Service.FATAL_GPU_WINDOW_S
        return svc

    def test_one_fatal_death_is_restarted_with_one_plain_line(self):
        svc = self.make(100)
        self.note(svc)                      # no exception: the caller goes on to restart the engine

    def test_an_ordinary_death_is_not_counted(self):
        svc = self.make(100, err="the engine ran out of memory")
        for pid in range(10):
            svc.engine.proc.pid = pid
            self.note(svc)
        self.assertFalse(svc.__dict__.get("_fatal_gpu_times"))

    def test_the_same_death_seen_by_many_requests_counts_once(self):
        svc = self.make(100)
        for _ in range(8):
            self.note(svc)
        self.assertEqual(len(svc._fatal_gpu_times), 1)

    def test_three_fatal_deaths_in_ten_minutes_stop_the_restarts_and_say_so(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc.engine.proc.pid = 3
        with self.assertRaises(S.EngineDied) as cm:
            self.note(svc)
        self.assertIn("3 times in 10 minutes", str(cm.exception))
        self.assertIn("not started again", str(cm.exception))
        with self.assertRaises(S.EngineDied):          # the same dead engine asked again: still refused
            self.note(svc)

    def test_after_the_window_it_tries_again(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc._fatal_gpu_times[:] = [t - S.Service.FATAL_GPU_WINDOW_S - 1 for t in svc._fatal_gpu_times]
        svc.engine.proc.pid = 3
        self.note(svc)                      # the two old ones aged out: only this one counts

    def test_the_request_in_flight_fails_as_an_engine_death_not_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        with self.assertRaises(S.EngineDied) as cm:
            S.StrataEngine._raise_if_fatal_gpu(eng, "ERR fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered")
        self.assertIn("illegal memory access", str(cm.exception))
        self.assertTrue(eng.ended)                                  # the next request starts the engine at once
        self.assertTrue(eng.last_err.startswith("fatal GPU error"))

    def test_an_ordinary_engine_error_stays_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        S.StrataEngine._raise_if_fatal_gpu(eng, "ERR the prompt (9 tokens) and max_new 5 are more than the context 8")   # no exception
        self.assertFalse(eng.ended)


if __name__ == "__main__":
    unittest.main()
