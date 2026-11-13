"""#5787: a native death of the serve process leaves the Python stack of the crashing thread in its own output.

    The field reports (ten automatic restarts over three days, on a Windows box, in one thread) all said the same
    thing: `python.exe` ended with `0xC0000005` or `0xC0000409`, the host log carried an exit code and nothing else,
    and it took the reporter's own `PYTHONFAULTHANDLER=1` in the launcher to get a stack.  So the server arms
    CPython's faulthandler itself, on its own output, in front of every launcher (the app, the Terminal, a source
    start).  It writes nothing while the process runs; it speaks only when the process is dying.

    python -m serve.test_faulthandler
"""
import os
import signal
import subprocess
import sys
import unittest
from unittest import mock

import serve.server as server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Arm(unittest.TestCase):
    def test_it_arms_with_every_thread(self):
        with mock.patch("faulthandler.enable") as enable:
            self.assertTrue(server.arm_faulthandler(env={}))
        enable.assert_called_once_with(all_threads=True)

    def test_the_off_switch_is_respected(self):
        with mock.patch("faulthandler.enable") as enable:
            self.assertFalse(server.arm_faulthandler(env={server.FAULTHANDLER_OFF: "1"}))
        enable.assert_not_called()

    def test_a_value_other_than_1_keeps_it_on(self):
        with mock.patch("faulthandler.enable") as enable:
            self.assertTrue(server.arm_faulthandler(env={server.FAULTHANDLER_OFF: "0"}))
        enable.assert_called_once_with(all_threads=True)

    def test_a_faulthandler_that_cannot_arm_never_stops_the_server(self):
        for failure in (OSError("the output is closed"), ImportError("no faulthandler"), ValueError("bad fd")):
            with mock.patch("faulthandler.enable", side_effect=failure):
                self.assertFalse(server.arm_faulthandler(env={}))

    def test_the_handler_is_left_off_by_a_closed_output(self):
        with mock.patch("faulthandler.enable", side_effect=OSError("closed")):
            self.assertFalse(server.arm_faulthandler())        # the real environment: it answers, it does not raise

    def test_serving_from_main_arms_it(self):
        """The call is in `main`, so every launcher gets it: the app's child, the Terminal, a source start.

        `_main` is stood in for, so this checks the wiring and starts no server."""
        with mock.patch.dict(os.environ, {server.FAULTHANDLER_OFF: ""}), \
                mock.patch("faulthandler.enable") as enable, mock.patch.object(server, "_main", return_value=0):
            self.assertEqual(server.main([]), 0)
        enable.assert_called_once_with(all_threads=True)

    def test_serving_from_main_arms_it_with_the_off_switch_set(self):
        with mock.patch.dict(os.environ, {server.FAULTHANDLER_OFF: "1"}), \
                mock.patch("faulthandler.enable") as enable, mock.patch.object(server, "_main", return_value=0):
            self.assertEqual(server.main([]), 0)
        enable.assert_not_called()


CRASHER = """
import os, signal, sys
sys.path.insert(0, os.environ["STRATA_ROOT"])
import serve.server as server

def the_line_that_dies():
    assert server.arm_faulthandler(env={})
    os.kill(os.getpid(), signal.SIGSEGV)

the_line_that_dies()
"""


class NativeDeath(unittest.TestCase):
    """What a reporter needs from a crash: the Python stack, in the output they already send."""

    @unittest.skipIf(os.name == "nt", "SIGSEGV is the POSIX signal; the Windows path is tested by CPython itself")
    def test_a_native_death_names_the_python_line_that_died(self):
        p = subprocess.run([sys.executable, "-c", CRASHER], cwd=ROOT, env={**os.environ, "STRATA_ROOT": ROOT},
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, -signal.SIGSEGV)              # it died by the signal, as before
        self.assertIn("Fatal Python error", p.stderr)                # faulthandler spoke
        self.assertIn("the_line_that_dies", p.stderr)                # and named the Python line
        self.assertIn("Current thread", p.stderr)

    @unittest.skipIf(os.name == "nt", "POSIX signal")
    def test_the_other_threads_are_shown_too(self):
        alive = "import threading\n" \
                "threading.Thread(target=lambda: threading.Event().wait(30), name='request-worker').start()\n" \
                "def the_line_that_dies():"
        crasher = CRASHER.replace("def the_line_that_dies():", alive)
        p = subprocess.run([sys.executable, "-c", crasher], cwd=ROOT, env={**os.environ, "STRATA_ROOT": ROOT},
                           capture_output=True, text=True, timeout=180)
        self.assertIn("Fatal Python error", p.stderr)
        self.assertIn("Thread 0x", p.stderr)                         # a live worker is in the report
        self.assertIn("request-worker", p.stderr)                    # and one can be told from the others


if __name__ == "__main__":
    unittest.main()
