"""A GPU picture reader whose start failed in its CUDA context ("CUDA error: out of memory ... in function ggml_cuda_init") on a Linux Arm machine with shared
CPU / GPU memory (DGX Spark) is started again before the CPU reader takes over: a new spawn may start only while less than 18 s have passed since the first one
started, and at most 12 times.  Every other failure goes the way it went.

    python -m serve.test_vision_ctx_retry
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from strata_control import vision_reader as VR

CTX = ("CUDA error: out of memory\n  current device: 0, in function ggml_cuda_init at ggml/src/ggml-cuda/ggml-cuda.cu:367\n"
       "ggml/src/ggml-cuda/ggml-cuda.cu:108: CUDA error\n")
ALLOC = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory\n"
DIED = "the vision encoder did not start (killed by signal 6)"


def rule(text=CTX, retries_done=0, next_start_s=0.0, **kw):
    args = dict(linux=True, arm=True, hip=False)
    args.update(kw)
    return VR.retry_gpu_start(text=text, retries_done=retries_done, next_start_s=next_start_s, **args)


class Rule(unittest.TestCase):
    def test_the_constants(self):
        self.assertEqual((VR.CONTEXT_RETRIES, VR.CONTEXT_RETRY_WAIT_S, VR.CONTEXT_RETRY_BUDGET_S), (12, 1.0, 18.0))

    def test_the_count_cap_ends_it(self):
        for done in range(12):
            self.assertTrue(rule(retries_done=done))
        self.assertFalse(rule(retries_done=12))
        self.assertFalse(rule(retries_done=20))

    def test_the_time_bound_is_the_start_of_the_next_spawn(self):
        self.assertTrue(rule(next_start_s=0.0))
        self.assertTrue(rule(next_start_s=17.9))
        self.assertFalse(rule(next_start_s=18.0))
        self.assertFalse(rule(next_start_s=40.0))

    def test_the_own_tower_wording_counts_too(self):
        self.assertTrue(rule("strata-vision: Strata's picture reader did not start (cannot use GPU 0: out of memory)"))

    def test_other_platforms_never_retry(self):
        self.assertFalse(rule(linux=False))           # Windows, Mac
        self.assertFalse(rule(arm=False))             # a PC with a card of its own
        self.assertFalse(rule(hip=True))              # the AMD reader

    def test_other_errors_never_retry(self):
        for text in (ALLOC,                                               # a card another program filled: no retry helps (reader_room reports it)
                     CTX + ALLOC,
                     "failed to allocate CUDA0 buffer of size 1048576 out of memory",
                     "CUDA error: device kernel image is invalid\n  in function ggml_cuda_init",
                     "CUDA error: CUDA-capable device(s) is/are busy or unavailable\n  in function ggml_cuda_init",
                     "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected",
                     "the vision encoder did not start: it said nothing for 300 s",
                     "out of memory"):                                     # no sign of the context
            with self.subTest(text=text[:50]):
                self.assertFalse(rule(text))


def make_vision(args=("/engine/strata-vision", "--gpu"), log_text=""):
    f = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    f.write(log_text)
    f.flush()
    v = server.Vision.__new__(server.Vision)
    v.spawn = (list(args), f, None)
    return v, f


class Clock:
    """A fake monotonic clock: `sleep` and every spawn (`attempt_s` long) move it.  It belongs to the thread that made it (the test's): `server.time` is the
    process-wide `time` module, so while it is patched any other thread of the pytest process that sleeps or reads the clock (on Windows a watcher or job thread
    that an earlier test left running) would call these.  Those threads get the real `time.sleep` / `time.monotonic` and are neither counted nor allowed to move the clock.
    Make it before patching, so the real functions it keeps are the unpatched ones."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.owner = threading.get_ident()
        self._real_sleep, self._real_monotonic = time.sleep, time.monotonic

    def monotonic(self):
        return self.now if threading.get_ident() == self.owner else self._real_monotonic()

    def sleep(self, s):
        if threading.get_ident() != self.owner:
            return self._real_sleep(s)
        self.sleeps.append(s)
        self.now += s


class Spawn(unittest.TestCase):
    def run_start(self, outcomes, args=("/engine/strata-vision", "--gpu"), system="linux", machine="aarch64", logged=CTX, attempt_s=1.4, hook=None):
        """_start_gpu_reader with `_start_reader` giving `outcomes` in turn (the last one repeats; an exception is raised, anything else returned) and taking `attempt_s`
        on a fake clock; returns (clock, calls, result or error)."""
        v, f = make_vision(args)
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        clock, calls = Clock(), []

        def start_reader():
            calls.append(clock.now)             # when this spawn started
            if hook is not None:
                hook()
            clock.now += attempt_s
            o = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(o, Exception):
                f.write(logged)
                f.flush()
                raise o
            return o

        v._start_reader = start_reader
        with mock.patch.object(server.sys, "platform", system), mock.patch.object(server.platform, "machine", return_value=machine), \
                mock.patch.object(server.time, "sleep", clock.sleep), mock.patch.object(server.time, "monotonic", clock.monotonic):
            try:
                res = v._start_gpu_reader()
            except RuntimeError as e:
                res = e
        return clock, calls, res

    def test_a_start_that_works_the_first_time_is_not_delayed(self):
        clock, calls, res = self.run_start([None])
        self.assertEqual((len(calls), clock.sleeps), (1, []))

    def test_a_success_ends_it_at_once(self):
        clock, calls, res = self.run_start([RuntimeError(DIED)] * 3 + [None])       # the measured case: 3 failures, the 4th spawn works
        self.assertEqual((len(calls), clock.sleeps, res), (4, [1.0, 1.0, 1.0], None))
        for got, want in zip(calls, [0.0, 2.4, 4.8, 7.2]):
            self.assertAlmostEqual(got, want)

    def test_the_five_measured_failures_are_covered(self):
        clock, calls, res = self.run_start([RuntimeError(DIED)] * 5 + [None])
        self.assertEqual((len(calls), res), (6, None))
        self.assertLess(calls[-1], 18.0)

    def test_a_context_that_never_works_ends_by_the_time_bound_and_raises_as_before(self):
        err = RuntimeError(DIED)
        clock, calls, res = self.run_start([err])
        self.assertIs(res, err)
        self.assertEqual(len(calls), 8)                       # spawns start at 0, 2.4, ... 16.8 s; the next would start at 19.2 s
        self.assertLess(calls[-1], 18.0)
        self.assertLess(clock.now, 20.0)                      # the worst wait before the CPU reader takes over

    def test_fast_attempts_end_by_the_count_cap(self):
        err = RuntimeError(DIED)
        clock, calls, res = self.run_start([err], attempt_s=0.1)
        self.assertIs(res, err)
        self.assertEqual((len(calls), len(clock.sleeps)), (13, 12))       # 1 spawn + 12 retries, 13.3 s: the cap, not the clock
        self.assertLess(clock.now, 18.0)

    def test_another_thread_that_sleeps_meanwhile_is_not_counted_and_does_not_move_the_clock(self):
        """`server.time` is the process-wide module: a thread of another part of the process that sleeps while the test has it patched (a winjob / watcher thread left
        by an earlier test on Windows counted 232,838 sleeps) must get the real sleep.  Here a thread sleeps in a loop and every spawn waits until it has slept once more."""
        stop, slept = threading.Event(), threading.Event()

        def other():
            while not stop.is_set():
                time.sleep(0.0002)              # the fake one when it is patched, and it must not count
                slept.set()

        def hook():
            slept.clear()
            slept.wait(2)

        t = threading.Thread(target=other, daemon=True)
        t.start()
        try:
            clock, calls, res = self.run_start([RuntimeError(DIED)], attempt_s=0.1, hook=hook)
        finally:
            stop.set()
            t.join(2)
        self.assertEqual((len(calls), len(clock.sleeps)), (13, 12))
        self.assertAlmostEqual(clock.now, 13 * 0.1 + 12 * 1.0)      # only this thread's pauses and spawns moved it

    def test_a_spawn_that_would_start_at_17_9_s_is_allowed_and_one_at_18_0_s_is_not(self):
        clock, calls, res = self.run_start([RuntimeError(DIED), None], attempt_s=16.9)        # fails at 16.9, next starts at 17.9
        self.assertEqual((len(calls), res), (2, None))
        self.assertAlmostEqual(calls[1], 17.9)
        clock, calls, res = self.run_start([RuntimeError(DIED), None], attempt_s=17.0)        # fails at 17.0, next would start at 18.0
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(res, RuntimeError)

    def test_the_diagnostic_line_only_with_STRATA_VIT_PROF_and_only_after_a_retry(self):
        for env, outcomes, expect in (({"STRATA_VIT_PROF": "1"}, [RuntimeError(DIED)] * 3 + [None], ["vit-prof: GPU reader started after 3 retries, 8.60 s"]),
                                      ({"STRATA_VIT_PROF": "1"}, [None], []),
                                      ({}, [RuntimeError(DIED)] * 3 + [None], []),
                                      ({"STRATA_VIT_PROF": "1"}, [RuntimeError(DIED)], ["vit-prof: GPU reader did not start after 7 retries, 18.20 s"])):
            with self.subTest(env=env, n=len(outcomes)), mock.patch.dict(server.os.environ, env), mock.patch("builtins.print") as pr:
                server.os.environ.pop("STRATA_VIT_PROF", None) if not env else None
                self.run_start(outcomes)
            # (print is patched for the whole process: only this module's diagnostic lines count, whatever another thread prints meanwhile)
            self.assertEqual([c.args[0] for c in pr.call_args_list if c.args and str(c.args[0]).startswith("vit-prof")], expect)

    def test_no_room_is_not_retried(self):
        err = server.VisionNoRoom("the vision encoder did not start: the graphics card has no room for it", None)
        clock, calls, res = self.run_start([err], logged=ALLOC)
        self.assertEqual((len(calls), clock.sleeps), (1, []))
        self.assertIs(res, err)

    def test_a_failure_that_is_not_the_context_is_not_retried(self):
        err = RuntimeError("the vision encoder did not start: ERR cannot open the model")
        clock, calls, res = self.run_start([err], logged="ERR cannot open the model\n")
        self.assertEqual((len(calls), clock.sleeps), (1, []))

    def test_other_systems_are_not_retried(self):
        err = RuntimeError("the vision encoder did not start")
        for kw in ({"system": "win32"}, {"system": "darwin"}, {"machine": "x86_64"}, {"machine": "AMD64"},
                   {"args": ("/engine/strata-vision-hip", "--gpu")}):
            with self.subTest(kw=kw):
                clock, calls, res = self.run_start([err], **kw)
                self.assertEqual((len(calls), clock.sleeps), (1, []))

    def test_an_old_failure_in_a_shared_log_does_not_count(self):
        v, f = make_vision(log_text=CTX)       # an earlier start's lines are in the file already
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        v._start_reader = mock.Mock(side_effect=RuntimeError("the vision encoder did not start: ERR cannot open the model"))
        clock = Clock()
        with mock.patch.object(server.sys, "platform", "linux"), mock.patch.object(server.platform, "machine", return_value="aarch64"), \
                mock.patch.object(server.time, "sleep", clock.sleep), mock.patch.object(server.time, "monotonic", clock.monotonic):
            with self.assertRaises(RuntimeError):
                v._start_gpu_reader()
        self.assertEqual(v._start_reader.call_count, 1)
        self.assertEqual(clock.sleeps, [])                    # no retry: this thread never slept

    def test_start_falls_back_to_the_cpu_reader_after_the_last_retry(self):
        v, f = make_vision()
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        v.fallback = (["/engine/strata-vision"], f, None)
        v.on_cpu, v.elastic = False, False
        starts = []
        clock = Clock()

        def start_reader():
            starts.append(v.on_cpu)
            if not v.on_cpu:
                f.write(CTX)
                f.flush()
                raise RuntimeError(DIED)

        v._start_reader = start_reader
        v.not_started_note = lambda: "note"
        with mock.patch.object(server.sys, "platform", "linux"), mock.patch.object(server.platform, "machine", return_value="aarch64"), \
                mock.patch.object(server.time, "sleep", clock.sleep), mock.patch.object(server.time, "monotonic", clock.monotonic), mock.patch("builtins.print"):
            v._start()
        self.assertEqual(starts, [False] * 13 + [True])        # fast attempts (the clock moves only by the pauses): the count cap
        self.assertTrue(v.on_cpu)


if __name__ == "__main__":
    unittest.main()
