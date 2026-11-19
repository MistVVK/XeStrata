"""P4b (0.2.16 retest): before the engine starts, the server drops the file cache an earlier start left for the model's files, so the engine's
40 GB of RAM do not have to be reclaimed page by page inside its allocation (serve/oomd_pacer.py make_room).

Measured on p1004 (rel0216/fix2plat/): natural cache state, the same load: user@ pressure peak 79.5 % (18 s above 50 %) without it, 1 % with the cache dropped.

    python -m unittest tests.terminal.test_oomd_room
"""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import oomd_pacer as op  # noqa: E402

GIB = 1 << 30
APPLICABLE = (Path("/x/memory.pressure"), 50.0)


class MakeRoom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.model = self.d / "model.gguf"
        self.model.write_bytes(b"x" * 1000)
        (self.d / "pack").mkdir()
        (self.d / "pack" / "t1.bin").write_bytes(b"y" * 500)
        (self.d / "pack" / "t2.bin").write_bytes(b"z" * 500)
        self.said, self.dropped = [], []
        patch = mock.patch.object(op, "_DONTNEED", 4)             # Windows has no posix_fadvise: the decisions are what is tested here, the drop is injected
        patch.start()
        self.addCleanup(patch.stop)
        self.cache = [14 * GIB]

    def room(self, *, cache=None, free=None, env=None, found=APPLICABLE, sizes=None, args=None):
        def drop(path):
            self.dropped.append(path.name)
            self.cache[0] = 0 if cache is None else cache
            return True
        return op.make_room(args if args is not None else ["--pack", str(self.d / "pack"), "--native", str(self.model), "--spec", "4", "relative/file", "--kv", "int8"],
                            self.said.append, env or {}, found=found, drop=drop, cache_of=lambda: self.cache[0],
                            free_of=lambda: free if free is not None else 10 * GIB, sizes=sizes or (lambda p: 40 * GIB // 3))

    def test_the_cache_of_the_files_named_in_the_arguments_is_dropped_and_said(self):
        freed = self.room()
        self.assertEqual(freed, 14 * GIB)
        self.assertEqual(sorted(self.dropped), ["model.gguf", "t1.bin", "t2.bin"])          # the folder's files and the file; "--spec", "4", "relative/file" are not paths
        self.assertEqual(len(self.said), 1)
        self.assertIn("freed 15 GB of memory that still held copies of this model's files", self.said[0])
        self.assertIn("systemd-oomd", self.said[0])

    def test_an_amount_below_1_GB_is_said_in_MB_not_as_0_GB(self):      # 0.2.17 item 18: 256 MiB to 0.5 GB was "freed 0 GB"
        for freed, words in ((256 << 20, "268 MB"), (400 * (1 << 20), "419 MB"), (500_000_000, "500 MB"), (999_000_000, "999 MB"),
                             (1_000_000_000, "1 GB"), (1_400_000_000, "1 GB"), (14 * GIB, "15 GB")):
            self.assertEqual(op.size_words(freed), words)
        self.said.clear()
        self.cache[0] = 3 * GIB
        self.assertEqual(self.room(cache=3 * GIB - (400 << 20)), 400 << 20)
        self.assertEqual(len(self.said), 1)
        self.assertIn("freed 419 MB of memory that still held copies", self.said[0])
        self.assertNotIn("0 GB", self.said[0])

    def boxed(self, *, free, cache=3 * GIB, in_box=True, env=None):
        """make_room as the server inside a container runs it: no user manager cgroup is visible, the figures are the host's /proc/meminfo."""
        self.cache[0] = cache

        def drop(path):
            self.dropped.append(path.name)
            self.cache[0] = 0
            return True
        with mock.patch.object(op, "applicable", return_value=None), mock.patch.object(op, "in_container", return_value=in_box):
            return op.make_room(["--pack", str(self.d / "pack"), "--native", str(self.model)], self.said.append, env or {}, drop=drop,
                                cache_of=lambda: self.cache[0], free_of=lambda: free, sizes=lambda p: 40 * GIB // 3)

    def test_in_a_container_the_cache_is_dropped_when_the_free_memory_does_not_cover_the_model_plus_10_percent(self):
        # 0.2.17, rootless Podman on Ubuntu (p1004): the container is ended by the HOST's systemd-oomd while 43 GB of 62 GB were free and 17 GB held the
        # earlier start's cache; the desktop rule (free < 0.8 x the model) would have said "enough room"
        freed = self.boxed(free=43 * GIB)
        self.assertEqual(freed, 3 * GIB)
        self.assertEqual(sorted(self.dropped), ["model.gguf", "t1.bin", "t2.bin"])
        self.assertEqual(len(self.said), 1)
        self.assertIn("does not end the container", self.said[0])
        self.assertNotIn("this desktop", self.said[0])
        self.said.clear()
        self.dropped.clear()
        self.assertEqual(self.boxed(free=45 * GIB), 0, "free memory covers the model plus 10 %: nothing to do")
        self.assertEqual((self.dropped, self.said), ([], []))

    def test_in_a_container_a_small_cache_and_a_switch_leave_it_alone(self):
        self.assertEqual(self.boxed(free=10 * GIB, cache=GIB // 2), 0)
        self.assertEqual(self.boxed(free=10 * GIB, env={"STRATA_OOMD_ROOM": "0"}), 0)
        self.assertEqual(self.boxed(free=10 * GIB, in_box=False), 0, "neither below a user manager nor in a container")
        self.assertEqual(self.dropped, [])

    def test_a_container_is_told_by_its_marker_files_or_the_container_variable(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("Linux only")
        with mock.patch.object(op.os.path, "exists", lambda p: False):
            self.assertFalse(op.in_container({}))
            self.assertTrue(op.in_container({"container": "podman"}))
        with mock.patch.object(op.os.path, "exists", lambda p: p == "/run/.containerenv"):
            self.assertTrue(op.in_container({}))

    def test_nothing_is_done_when_there_is_room_without_it(self):
        self.assertEqual(self.room(free=200 * GIB), 0)                                         # a big PC: free memory covers the model
        self.assertEqual(self.dropped, [])
        self.assertEqual(self.said, [])

    def test_a_small_cache_is_left_alone(self):
        self.cache[0] = GIB // 2
        self.assertEqual(self.room(), 0)
        self.assertEqual(self.dropped, [])

    def test_not_below_the_user_manager_or_with_oomd_off_it_does_nothing(self):
        with mock.patch.object(op, "applicable", return_value=None):
            self.assertEqual(op.make_room(["--native", str(self.model)], self.said.append, {}, drop=self.dropped.append), 0)
        self.assertEqual(self.room(env={"STRATA_OOMD_ROOM": "0"}), 0)
        self.assertEqual(self.dropped, [])

    def test_a_small_gain_is_not_announced(self):
        self.cache[0] = 2 * GIB
        self.assertEqual(self.room(cache=2 * GIB - 100 * (1 << 20)), 100 * (1 << 20))
        self.assertEqual(self.said, [])

    def test_unreadable_figures_and_errors_mean_nothing_done(self):
        self.assertEqual(op.make_room(["--native", str(self.model)], self.said.append, {}, found=APPLICABLE, cache_of=lambda: None, free_of=lambda: GIB), 0)
        self.assertEqual(op.make_room(["--native", str(self.model)], self.said.append, {}, found=APPLICABLE, cache_of=lambda: 14 * GIB, free_of=lambda: None), 0)
        self.assertEqual(op.make_room(["--native", str(self.model)], self.said.append, {}, found=APPLICABLE,
                                      cache_of=lambda: 1 / 0, free_of=lambda: GIB), 0)         # an exception never gets out
        self.assertEqual(self.said, [])

    def test_the_named_model_files_come_before_a_folder_of_thousands_of_small_files(self):
        # p1004: --pack <folder of ~24,000 expert files> comes first on the command line; a cap on the walk used to end before the 47 GB model file was reached
        for i in range(30):
            (self.d / "pack" / f"e{i}.bin").write_bytes(b"e")
        with mock.patch.object(op, "ROOM_MAX_FILES", 10):
            files = op._arg_files(["--pack", str(self.d / "pack"), "--native", str(self.model)])
        self.assertEqual(files[0], self.model)
        self.assertEqual(len(files), 10)

    def test_arguments_without_files_do_nothing(self):
        self.assertEqual(self.room(args=["--spec", "4", "--kv", "int8"]), 0)
        self.assertEqual(self.dropped, [])


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class StopTakesHold(unittest.TestCase):
    """A stop signal is acted on by one thread when it returns to user code.  While that thread is in one long system call (the engine placing its RAM,
    measured on p1004: state R, no thread ever T, pressure 43 % -> 87 % during the "pause") nothing stops: the pause is taken back at once."""

    def pacer(self, letters):
        self.clock, self.sent, self.said = Clock(), [], []
        self.loading = threading.Event()
        letters = iter(letters)
        p = op.Pacer(4242, lambda: 80.0, 50.0, self.said.append, alive=lambda: True, state=lambda: next(letters, "R"),
                     kill=lambda pid, sig: self.sent.append((sig, self.clock.t)), now=self.clock.now, sleep=self.clock.sleep)
        return p

    def test_an_engine_that_does_not_stop_is_not_paused_and_nothing_is_said(self):
        p = self.pacer(["R"] * 1000)
        p._pause(80.0, self.loading)
        self.assertEqual([s for s, _ in self.sent], [op.SIGSTOP, op.SIGCONT])           # taken back
        self.assertLessEqual(self.sent[1][1] - self.sent[0][1], op.STOP_CONFIRM_S + 0.5)
        self.assertEqual((p.pauses, p.unstoppable, self.said), (0, 1, []))
        self.assertEqual(p.paused_s, 0.0)
        self.assertFalse(p.stopped)

    def test_an_engine_that_stops_is_paused_and_it_is_said(self):
        p = self.pacer(["R", "R", "T"])
        orig = p.read
        readings = iter([80.0, 80.0, 10.0])
        p.read = lambda: next(readings, 10.0)
        p._pause(80.0, self.loading)
        self.assertEqual([s for s, _ in self.sent], [op.SIGSTOP, op.SIGCONT])
        self.assertEqual((p.pauses, p.unstoppable), (1, 0))
        self.assertEqual(len(self.said), 1)
        self.assertIn("pause 1", self.said[0])

    def test_a_state_that_cannot_be_read_counts_as_stopped(self):
        p = self.pacer([None])
        p._pause(80.0, self.loading)
        self.assertEqual(p.pauses, 1)

    def test_no_state_reader_keeps_the_old_behaviour(self):
        clock, sent, said = Clock(), [], []
        p = op.Pacer(1, lambda: 10.0, 50.0, said.append, alive=lambda: True, kill=lambda pid, sig: sent.append(sig), now=clock.now, sleep=clock.sleep)
        p._pause(80.0, threading.Event())
        self.assertEqual((p.pauses, len(said)), (1, 1))

    def test_the_real_state_reader(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("needs /proc")
        self.assertIn(op.process_state(os.getpid()), ("R", "S"))
        self.assertIsNone(op.process_state(2 ** 22 + 99991))


class TrimWhileLoading(unittest.TestCase):
    """Measured on p1004 (12 GiB of other programs): with the earlier cache already dropped, pressure still went to 90 % with 10 GB free, because every
    page the engine reads once stays cached and each new page is a reclaim stall.  While the pressure is 30 % of oomd's limit or more, the model file's
    cache is given back about once a second."""

    def pacer(self, value, trim):
        self.clock, self.said = Clock(), []
        return op.Pacer(1, lambda: value, 50.0, self.said.append, alive=lambda: True, now=self.clock.now, sleep=self.clock.sleep, trim=trim)

    def test_it_trims_once_a_second_from_15_percent_and_says_so_once(self):
        calls = []
        p = self.pacer(20.0, lambda: calls.append(p.now()))
        for _ in range(10):
            p._maybe_trim(20.0)
            self.clock.sleep(0.5)
        self.assertEqual(len(calls), 5)                                   # 5 s, one a second
        self.assertEqual(len(self.said), 1)
        self.assertIn("given back as it loads", self.said[0])

    def test_a_calm_desktop_is_not_trimmed(self):
        calls = []
        p = self.pacer(5.0, lambda: calls.append(1))
        p._maybe_trim(5.0)
        p._maybe_trim(None)
        self.assertEqual((calls, self.said), ([], []))

    def test_no_trim_function_and_a_failing_one_are_fine(self):
        self.pacer(40.0, None)._maybe_trim(40.0)

        def boom():
            raise OSError("x")
        p = self.pacer(40.0, boom)
        p._maybe_trim(40.0)
        self.assertEqual(self.said, [])

    def test_only_the_native_model_file_is_trimmed(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.gguf", Path(d) / "b.gguf"
            a.write_bytes(b"x")
            b.write_bytes(b"y")
            got = op.trim_files(["--pack", d, "--native", str(a), "--ple-gguf", str(b), "--spec", "4"])
            self.assertEqual(got, [a])
            self.assertEqual(op.trim_files(["--native", "relative.gguf"]), [])


@unittest.skipUnless(sys.platform.startswith("linux"), "posix_fadvise")
class RealDrop(unittest.TestCase):
    def test_a_real_file_is_dropped_without_changing_it(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(os.urandom(1 << 20))
            before = p.read_bytes()
            self.assertTrue(op._drop_cache(p))
            self.assertEqual(p.read_bytes(), before)
            self.assertFalse(op._drop_cache(Path(d) / "missing.bin"))
            self.assertEqual([x.name for x in op._arg_files(["--native", str(p), "--x", "no/such", d + "/missing"])], ["f.bin"])

    def test_the_server_calls_it_before_it_starts_the_engine(self):
        text = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        room = text.index("oomd_pacer.make_room(args)")
        self.assertLess(room, text.index("self.proc = popen(\"the Strata engine\""))
        self.assertGreater(room, text.index("end_leftover_engines(exe)"))


if __name__ == "__main__":
    unittest.main()
