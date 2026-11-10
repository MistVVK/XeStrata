"""#4260: `strata bench run --profile` ends the engine it started, with the program it started, and says so only when something is left.

The engine tree is the server (`strata serve`), the engine it starts and whatever those start. On Windows the tree is put in a job object this run owns,
so a per-process `taskkill` that is refused ("Access is denied" on a tree this very user started) no longer leaves it running; on Linux and macOS it is the
process group. An unrelated process with the same program name is never touched.

    python -m pytest tests/terminal/test_bench_stop.py
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import bench_standard

WINDOWS = os.name == "nt"
PARENT = ("import subprocess, sys, time\n"
          "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
          "open(sys.argv[1], 'w').write(str(kid.pid))\n"
          "time.sleep(120)\n")


def alive(pid: int) -> bool:
    if WINDOWS:
        from strata_control import winproc
        return winproc.pid_alive(pid, None)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie answers signal 0; only its parent has not collected it
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        # collected between the signal and the read (1 in 15 on a loaded p2006): gone, unless signal 0 still finds it (no /proc)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return not Path("/proc").is_dir()


class StopEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-benchstop-")
        self.addCleanup(self.temp.cleanup)
        self.pidfile = Path(self.temp.name) / "kid.pid"
        self.procs = []
        self.addCleanup(self._reap)

    def _reap(self):
        for pid in getattr(self, "pids", []):
            try:
                if alive(pid):
                    os.kill(pid, 9)
            except OSError:
                pass
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
        for proc in self.procs:
            handle = getattr(proc, "_strata_log", None)
            if handle is not None and not handle.closed:
                handle.close()
        time.sleep(0.5)

    def _start_tree(self):
        """What start_engine does: a child of this run, in its own group (POSIX) or process group + job (Windows), with a child of its own."""
        kwargs = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)} if WINDOWS else {"start_new_session": True}
        handle = open(Path(self.temp.name) / "engine.log", "wb")
        proc = subprocess.Popen([sys.executable, "-c", PARENT, str(self.pidfile)], stdout=handle, stderr=subprocess.STDOUT, **kwargs)
        proc._strata_log = handle
        self.procs.append(proc)
        if WINDOWS:
            from serve import winjob
            proc._strata_job = winjob.contain(proc)
        # wait for the whole pid, not only the file: under load the file can exist, still empty, for longer than a fixed pause
        deadline, kid = time.monotonic() + 60, None
        while kid is None and time.monotonic() < deadline:
            try:
                kid = int(self.pidfile.read_text().strip())
            except (OSError, ValueError):
                time.sleep(0.05)
        self.assertIsNotNone(kid, "the stand-in engine did not start its own child within 60 s")
        self.pids = [proc.pid, kid]
        return proc, kid

    def test_the_child_and_its_child_both_end(self):
        proc, kid = self._start_tree()
        outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])      # same program, not ours
        self.procs.append(outsider)
        self.assertTrue(alive(kid))
        self.assertIsNone(bench_standard.stop_engine(proc))
        deadline = time.monotonic() + 30                                  # a loaded box can take seconds to deliver and reap
        while alive(kid) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(alive(proc.pid), "the engine's server survived")
        self.assertFalse(alive(kid), "what the engine's server started survived")
        self.assertIsNone(outsider.poll(), "a process of the same program that this run did not start was ended")

    @unittest.skipUnless(WINDOWS, "Windows: a refused taskkill")
    def test_a_refused_taskkill_does_not_leave_the_tree_running(self):
        proc, kid = self._start_tree()
        refused = subprocess.CompletedProcess(["taskkill"], 128, b"", b"ERROR: The process could not be terminated.\r\nReason: Access is denied.\r\n")
        with mock.patch.object(bench_standard.subprocess, "run", return_value=refused):
            problem = bench_standard.stop_engine(proc)
        self.assertIsNone(problem, "the tree is gone, so there is nothing to report")
        deadline = time.monotonic() + 10
        while alive(kid) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(alive(kid))

    @unittest.skipUnless(WINDOWS, "Windows: every way of ending it refused")
    def test_a_tree_that_cannot_be_ended_is_still_reported(self):
        proc, kid = self._start_tree()
        refused = subprocess.CompletedProcess(["taskkill"], 128, b"", b"Access is denied.")
        with mock.patch.object(bench_standard.subprocess, "run", return_value=refused), \
                mock.patch("serve.winjob.terminate", return_value=False), mock.patch.object(proc, "kill", side_effect=OSError("denied")):
            problem = bench_standard.stop_engine(proc)
        self.assertIn("still running", problem or "")

    def test_a_process_that_already_ended_is_nothing_to_stop(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(20)
        proc._strata_log = open(os.devnull, "wb")
        self.assertIsNone(bench_standard.stop_engine(proc))


if __name__ == "__main__":
    unittest.main()
