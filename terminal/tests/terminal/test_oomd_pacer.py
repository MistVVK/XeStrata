"""P4 (0.2.16): the pacer that keeps systemd-oomd from ending a model load (serve/oomd_pacer.py).

Measured on p1004 (Ubuntu 24.04, systemd 255): a program in a desktop-terminal-like scope is ended by oomd when `user@UID.service`'s memory
pressure stays above 50 % for 20 s; excluding the engine with ManagedOOMPreference=omit only makes oomd end the terminal instead; a pacer that
stops the program while the pressure is above 35 % and continues it below 15 % kept it alive.
"""
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import oomd_pacer as op  # noqa: E402

TERMINAL_SCOPE = "0::/user.slice/user-1001.slice/user@1001.service/app.slice/vte-spawn-1b2c.scope\n"
LOGIN_SCOPE = "0::/user.slice/user-1001.slice/session-16676.scope\n"
SHOW_50 = "ManagedOOMMemoryPressure=kill\nManagedOOMMemoryPressureLimit=2147483648\n"
ACTIVE = "active\n"
CALM = "some avg10=1.00 avg60=0.00 avg300=0.00 total=1\n"


class Parsing(unittest.TestCase):
    def test_a_terminal_scope_is_below_the_user_manager(self):
        self.assertEqual(op.manager_cgroup(TERMINAL_SCOPE), "/user.slice/user-1001.slice/user@1001.service")
        self.assertEqual(op.manager_cgroup("0::/user.slice/user-1000.slice/user@1000.service\n"), "/user.slice/user-1000.slice/user@1000.service")

    def test_a_login_session_a_container_and_old_cgroups_are_not(self):
        for text in (LOGIN_SCOPE, "0::/\n", "0::/docker/abc123\n", "12:memory:/user.slice\n1:name=systemd:/user.slice/user-1000.slice/user@1000.service\n", ""):
            self.assertIsNone(op.manager_cgroup(text), text)

    def test_the_kill_limit(self):
        self.assertEqual(op.kill_limit_percent(SHOW_50), 50.0)
        self.assertEqual(op.kill_limit_percent("ManagedOOMMemoryPressure=kill\nManagedOOMMemoryPressureLimit=4294967295\n"), 100.0)
        self.assertEqual(op.kill_limit_percent("ManagedOOMMemoryPressure=kill\nManagedOOMMemoryPressureLimit=0\n"), 60.0)   # oomd's default
        self.assertIsNone(op.kill_limit_percent("ManagedOOMMemoryPressure=auto\nManagedOOMMemoryPressureLimit=2147483648\n"))
        self.assertIsNone(op.kill_limit_percent(""))
        self.assertIsNone(op.kill_limit_percent("ManagedOOMMemoryPressure=kill\nManagedOOMMemoryPressureLimit=lots\n"))

    def test_the_pressure_figure(self):
        self.assertEqual(op.avg10("some avg10=12.34 avg60=1.00 avg300=0.10 total=123\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"), 12.34)
        self.assertIsNone(op.avg10("garbage"))
        self.assertIsNone(op.avg10(""))


class Applicable(unittest.TestCase):
    CG = "/user.slice/user-1001.slice/user@1001.service"

    def tree(self):
        import tempfile
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        pressure = Path(d.name) / self.CG.lstrip("/") / "memory.pressure"
        pressure.parent.mkdir(parents=True)
        pressure.write_text(CALM, encoding="utf-8")
        return Path(d.name), pressure

    def test_not_off_linux_and_not_when_switched_off(self):
        with mock.patch.object(op.sys, "platform", "win32"):
            self.assertIsNone(op.applicable({}))
        with mock.patch.object(op.sys, "platform", "linux"):
            self.assertIsNone(op.applicable({"STRATA_OOMD_PACER": "0"}))

    def test_not_in_a_login_session(self):
        with mock.patch.object(op.sys, "platform", "linux"), mock.patch.object(op, "manager_cgroup", return_value=None):
            self.assertIsNone(op.applicable({}))

    def run_applicable(self, answers, root):
        with mock.patch.object(op.sys, "platform", "linux"), mock.patch.object(op, "manager_cgroup", return_value=self.CG),                 mock.patch.object(op, "CGROUP_ROOT", root), mock.patch.object(op, "_run", side_effect=lambda args: answers.get(args[1])):
            return op.applicable({})

    def test_not_when_oomd_is_not_running_or_not_set_to_kill(self):
        root, _ = self.tree()
        self.assertIsNone(self.run_applicable({"show": SHOW_50}, root))                                   # systemctl: systemd-oomd is not active
        self.assertIsNone(self.run_applicable({"is-active": ACTIVE, "show": "ManagedOOMMemoryPressure=auto\n"}, root))
        self.assertIsNone(self.run_applicable({"is-active": ACTIVE}, root))                          # no answer to show

    def test_not_when_the_pressure_file_cannot_be_read(self):
        import tempfile
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(self.run_applicable({"is-active": ACTIVE, "show": SHOW_50}, Path(empty)))

    def test_applicable_when_every_condition_holds(self):
        root, pressure = self.tree()
        self.assertEqual(self.run_applicable({"is-active": ACTIVE, "show": SHOW_50}, root), (pressure, 50.0))

    def test_start_never_raises(self):
        with mock.patch.object(op, "applicable", side_effect=RuntimeError("boom")):
            self.assertIsNone(op.start(mock.Mock(pid=1), threading.Event()))
        with mock.patch.object(op, "applicable", return_value=None):
            self.assertIsNone(op.start(mock.Mock(pid=1), threading.Event()))


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Loop(unittest.TestCase):
    def pacer(self, pressures, *, budget=600.0, limit=50.0, alive=None):
        """A pacer whose pressure follows `pressures(t)`, with a fake clock and a list of the signals it sent."""
        self.clock, self.sent, self.said = Clock(), [], []
        self.loading = threading.Event()
        return op.Pacer(4242, lambda: pressures(self.clock.t), limit, self.said.append, alive=alive or (lambda: True),
                        kill=lambda pid, sig: self.sent.append((pid, sig, self.clock.t)), now=self.clock.now, sleep=self.clock.sleep, budget=budget)

    def run_until(self, pacer, seconds):
        def ready():
            while self.clock.t < seconds:
                time.sleep(0)
            self.loading.set()
        # the fake clock only moves inside the pacer's sleeps: end the run by READY at the time asked for
        orig = pacer.sleep

        def sleep(s):
            orig(s)
            if self.clock.t >= seconds:
                self.loading.set()
        pacer.sleep = sleep
        pacer.run(self.loading)

    def test_it_stops_at_70_percent_of_the_limit_and_continues_at_30(self):
        # pressure climbs 3 points a second while the engine runs and falls 4 points a second while it is stopped (a 50 % limit: stop 35, go 15)
        state = {"v": 0.0, "stopped": False, "last": 0.0}

        def pressure(t):
            dt, state["last"] = t - state["last"], t
            state["v"] = max(0.0, state["v"] - 4 * dt) if state["stopped"] else state["v"] + 3 * dt
            return state["v"]
        p = self.pacer(pressure)
        orig = p.kill

        def kill(pid, sig):
            orig(pid, sig)
            state["stopped"] = sig == op.SIGSTOP
        p.kill = kill
        self.run_until(p, 120)
        stops = [s for s in self.sent if s[1] == op.SIGSTOP]
        conts = [s for s in self.sent if s[1] == op.SIGCONT]
        self.assertGreaterEqual(len(stops), 3)
        self.assertEqual(len(stops), len(conts))                          # every stop is followed by a continue
        self.assertEqual(self.sent[-1][1], op.SIGCONT)                # and it always ends continued
        self.assertTrue(all(s[0] == 4242 for s in self.sent))
        self.assertLess(state["v"], 50.0)                                  # the figure never reached oomd's limit
        self.assertEqual(sum(1 for line in self.said if "pause" in line and "waits a few seconds" in line), p.pauses)   # one line per pause

    def test_a_calm_desktop_is_never_touched(self):
        p = self.pacer(lambda t: 8.0)
        self.run_until(p, 60)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.said, [])

    def test_a_pause_is_never_longer_than_30_seconds(self):
        p = self.pacer(lambda t: 90.0)                                      # other programs keep the pressure up: our pause cannot lower it
        self.run_until(p, 100)
        stops = [s for s in self.sent if s[1] == op.SIGSTOP]
        conts = [s for s in self.sent if s[1] == op.SIGCONT]
        self.assertGreaterEqual(len(stops), 2)
        for a, b in zip(stops, conts):
            self.assertLessEqual(b[2] - a[2], op.PAUSE_MAX_S + 1.0)
        for i in range(1, len(stops)):
            self.assertGreaterEqual(stops[i][2] - conts[i - 1][2], op.RUN_MIN_S)   # it runs a little between two pauses

    def test_the_total_pause_is_capped_then_it_says_so_and_goes_on_unpaced(self):
        p = self.pacer(lambda t: 90.0, budget=100.0)
        self.run_until(p, 600)
        self.assertLessEqual(p.paused_s, 100.0 + 1.0)
        self.assertEqual(self.sent[-1][1], op.SIGCONT)
        self.assertTrue(any("is not paused any more" in line for line in self.said))
        n = len(self.sent)
        self.assertGreater(self.clock.t, 0)
        self.assertEqual(n, len([s for s in self.sent]))

    def test_ready_ends_it_with_a_continue_and_a_summary(self):
        state = {"stopped": False}
        p = self.pacer(lambda t: 3.0 if (t < 10 or t > 40) else 90.0)
        p.run_until = None
        orig_kill = p.kill

        def kill(pid, sig):
            orig_kill(pid, sig)
            if sig == op.SIGSTOP:
                self.loading.set()                                          # READY arrives while it is stopped
        p.kill = kill
        p.pressures = None
        self.clock.t = 20.0                                                 # start inside the high stretch
        p.run(self.loading)
        self.assertEqual([s[1] for s in self.sent], [op.SIGSTOP, op.SIGCONT])
        self.assertTrue(any("short pause" in line for line in self.said))

    def test_a_dead_engine_gets_no_signal(self):
        p = self.pacer(lambda t: 90.0, alive=lambda: False)
        p.run(self.loading)
        self.assertEqual(self.sent, [])

    def test_a_failing_signal_is_survived(self):
        p = self.pacer(lambda t: 90.0)

        def refuse(pid, sig):
            raise ProcessLookupError()
        p.kill = refuse
        self.run_until(p, 20)                                               # no exception, nothing marked as stopped
        self.assertFalse(p.stopped)

    def test_an_unreadable_figure_is_not_pressure(self):
        p = self.pacer(lambda t: None)
        self.run_until(p, 30)
        self.assertEqual(self.sent, [])


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class RealSignals(unittest.TestCase):
    def test_a_real_process_is_stopped_and_continued(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(child.kill)
        self.addCleanup(child.wait)

        def state():
            return Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        seen = []
        readings = iter([80.0] * 4 + [5.0] * 20)
        loading = threading.Event()

        def read():
            v = next(readings, 5.0)
            seen.append(state())
            return v
        pacer = op.Pacer(child.pid, read, 50.0, lambda text: None, alive=lambda: child.poll() is None, sleep=lambda s: time.sleep(0.02))
        t = threading.Thread(target=pacer.run, args=(loading,))
        t.start()
        time.sleep(1.0)
        loading.set()
        t.join(10)
        self.assertIn("T", seen)                                            # it was stopped at some point
        time.sleep(0.2)
        self.assertNotEqual(state(), "T")                                   # and is running again


class KilledWords(unittest.TestCase):
    def test_said_only_for_a_sigkill_on_a_watched_desktop(self):
        with mock.patch.object(op, "applicable", return_value=(Path("/x"), 50.0)):
            text = op.killed_words(-9)
            self.assertIn("systemd-oomd", text)
            self.assertIn("Close other programs", text)
            self.assertEqual(op.killed_words(-11), "")                 # another signal
            self.assertEqual(op.killed_words(1), "")
            self.assertEqual(op.killed_words(None), "")
        with mock.patch.object(op, "applicable", return_value=None):
            self.assertEqual(op.killed_words(-9), "")                  # a login session, no oomd, not Linux
        with mock.patch.object(op, "applicable", side_effect=RuntimeError("x")):
            self.assertEqual(op.killed_words(-9), "")

    def test_the_start_error_carries_it(self):
        from serve.server_parts import engines
        with mock.patch.object(op, "applicable", return_value=(Path("/x"), 50.0)), mock.patch.object(engines.sys, "platform", "linux"):
            self.assertIn("systemd-oomd", engines.start_exit_note(-9, None))


class WhereItIsCalled(unittest.TestCase):
    def test_the_server_starts_the_pacer_right_after_the_engine(self):
        text = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertIn("from serve import oomd_pacer", text)
        spawn = text.index('self.proc = popen("the Strata engine"')
        call = text.index("oomd_pacer.start(proc, loading)")
        self.assertGreater(call, spawn)
        self.assertLess(call - spawn, 1800)
        self.assertLess(call, text.index("loading.set()\n        if self.max_context <= 0"))

    def test_the_limits_are_the_agreed_ones(self):
        self.assertEqual(op.PAUSE_BUDGET_S, 600.0)
        self.assertEqual((op.STOP_FRACTION, op.RESUME_FRACTION), (0.70, 0.30))


if __name__ == "__main__":
    unittest.main()
