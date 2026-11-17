"""Round 0213 QATUI: `strata bench run --profile` ended with SIGTERM must end the engine it started, like Ctrl+C does.

Found on l620 (Linux, CUDA) and m1001 (macOS): after `kill -TERM <bench pid>` the runner was gone and its server and engine kept the port and the GPU
(only KeyboardInterrupt was handled, so the `finally` that stops the engine never ran). The driver below runs the real `bench run` command line in a
child process with a stand-in engine (a sleeping process in its own session, as `start_engine` makes it) and the real `stop_engine`.

    python -m pytest tests/terminal/test_bench_term_0213_qatui.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli import bench_standard  # noqa: E402

POSIX = os.name != "nt"

DRIVER = r'''
import os, subprocess, sys, time
from pathlib import Path
from unittest import mock
sys.path.insert(0, sys.argv[1])
from strata_cli import bench_standard as bs, cli
mode, pidfile, ready = sys.argv[2], sys.argv[3], sys.argv[4]
kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
Path(pidfile).write_text(str(kid.pid))
plan = {"effective": {"host": "127.0.0.1", "port": 9}, "engine_argv": []}

def fake_start(ref, target, log_path, log, timeout_s):
    return kid

def fake_run(*a, **k):                         # the benchmark itself: still running when the signal comes
    Path(ready).write_text("x")
    time.sleep(120)

def slow_wait(*a, **k):                        # the model is still loading when the signal comes
    Path(ready).write_text("x")
    time.sleep(120)

patches = [mock.patch("strata_cli.cli._source", return_value=(Path("p.json"), None, None, {})),
           mock.patch("strata_cli.cli._launch_plan", return_value=(plan, [])),
           mock.patch.object(bs.Target, "get", lambda self, path, timeout=10: None)]
if mode == "loading":                          # the REAL start_engine, with the child it would start and a wait that never ends
    patches += [mock.patch.object(bs.subprocess, "Popen", lambda *a, **k: kid), mock.patch.object(bs, "_start_wait", slow_wait)]
else:
    patches += [mock.patch.object(bs, "start_engine", fake_start), mock.patch.object(bs, "run", fake_run)]
for p in patches:
    p.start()
raise SystemExit(cli.main(["bench", "run", "--profile", "p.json", "--out", os.path.join(os.path.dirname(pidfile), "out"), "--skip-model-hash"]))
'''


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


@unittest.skipUnless(POSIX, "SIGTERM is a POSIX signal; Windows ends the tree through a job object (test_bench_stop.py)")
class TermEndsTheEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-benchterm-")
        self.addCleanup(self.temp.cleanup)
        self.pidfile = Path(self.temp.name) / "kid.pid"
        self.ready = Path(self.temp.name) / "ready"
        self.driver = Path(self.temp.name) / "driver.py"
        self.driver.write_text(DRIVER, encoding="utf-8")
        self.addCleanup(self._reap)

    def _reap(self):
        try:
            pid = int(self.pidfile.read_text())
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass

    def _run(self, mode: str, sig=signal.SIGTERM):
        proc = subprocess.Popen([sys.executable, str(self.driver), str(ROOT), mode, str(self.pidfile), str(self.ready)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.time() + 60
        while time.time() < deadline and not self.ready.exists():
            time.sleep(0.1)
        self.assertTrue(self.ready.exists(), "the driver never got to the benchmark: " + (proc.stderr.read() if proc.poll() is not None else ""))
        kid = int(self.pidfile.read_text())
        self.assertTrue(alive(kid))
        proc.send_signal(sig)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("the runner did not end after the signal")
        end = time.time() + 30
        while time.time() < end and alive(kid):
            time.sleep(0.2)
        return code, kid, proc

    def test_sigterm_while_the_benchmark_runs_stops_the_engine(self):
        code, kid, proc = self._run("running")
        self.assertFalse(alive(kid), "the engine this run started is still running after SIGTERM")
        self.assertEqual(code, 130, proc.stderr.read())

    def test_sigint_while_the_benchmark_runs_stops_the_engine(self):
        code, kid, proc = self._run("running", signal.SIGINT)
        self.assertFalse(alive(kid), "the engine this run started is still running after SIGINT")

    def test_sigterm_while_the_model_loads_stops_the_engine(self):
        """The engine is started but not ready: start_engine owns it then, the caller never receives it."""
        code, kid, proc = self._run("loading")
        self.assertFalse(alive(kid), "the engine is still running after SIGTERM during the load")


class Handler(unittest.TestCase):
    @unittest.skipUnless(POSIX, "POSIX signal")
    def test_the_handler_is_installed_only_inside_and_put_back(self):
        before = signal.getsignal(signal.SIGTERM)
        with bench_standard.term_is_interrupt():
            self.assertIsNot(signal.getsignal(signal.SIGTERM), before)
            with self.assertRaises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGTERM)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_a_thread_other_than_the_main_one_is_left_alone(self):
        import threading
        seen = []

        def work():
            with bench_standard.term_is_interrupt():
                seen.append(True)
        t = threading.Thread(target=work)
        t.start()
        t.join()
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
