"""Leftover detection: a process is ended only when it is exactly what Strata recorded and its starter is gone."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_control import procs

SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Procs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-procs-")
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": self.temp.name, "XDG_STATE_HOME": self.temp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.started = []
        self.addCleanup(self._kill_all)

    def _kill_all(self):
        for proc in self.started:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def sleeper(self):
        proc = subprocess.Popen(SLEEP, creationflags=NO_WINDOW)
        self.started.append(proc)
        deadline = time.monotonic() + 10
        while procs.identify(proc.pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return proc

    def recorded_pair(self):
        """(starter, child, record): a recorded child of a starter that is then ended."""
        starter, child = self.sleeper(), self.sleeper()
        self.assertIsNotNone(procs.write_record(child.pid, "engine", profile="app-1", parent_pid=starter.pid))
        return starter, child

    def gone(self, proc, seconds=15):
        try:
            proc.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    def test_a_child_whose_starter_is_gone_is_a_leftover_and_is_ended(self):
        starter, child = self.recorded_pair()
        self.assertEqual(procs.leftovers()[0], [], "its starter is alive: not a leftover")
        starter.kill(); starter.wait(timeout=10)
        found, unverifiable = procs.leftovers()
        self.assertEqual([r["pid"] for r in found], [child.pid])
        self.assertEqual(unverifiable, [])
        result = procs.end_leftovers(grace=2)
        self.assertEqual([r["pid"] for r in result["ended"]], [child.pid])
        self.assertTrue(self.gone(child))
        self.assertEqual(procs.records(), [], "the record goes with the process")

    def test_a_child_with_a_live_starter_is_never_touched(self):
        starter, child = self.recorded_pair()
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll())

    def test_a_reused_pid_is_not_the_recorded_process(self):
        starter, child = self.recorded_pair()
        path = procs.procs_dir() / f"{child.pid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["token"] = "x:not-this-process"                  # what a later process with the same number looks like
        path.write_text(json.dumps(record), encoding="utf-8")
        starter.kill(); starter.wait(timeout=10)
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll(), "a stranger with the recorded PID must be left alone")
        self.assertEqual(procs.records(), [], "the stale record is dropped")

    def test_the_same_start_time_with_another_command_line_is_not_ended(self):
        starter, child = self.recorded_pair()
        path = procs.procs_dir() / f"{child.pid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["cmd"] = [sys.executable, "-c", "something else"]
        path.write_text(json.dumps(record), encoding="utf-8")
        starter.kill(); starter.wait(timeout=10)
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll())

    def test_a_profile_filter_only_ends_that_profiles_leftovers(self):
        starter, child = self.recorded_pair()
        other_starter, other = self.sleeper(), self.sleeper()
        procs.write_record(other.pid, "engine", profile="app-2", parent_pid=other_starter.pid)
        starter.kill(); other_starter.kill()
        starter.wait(timeout=10); other_starter.wait(timeout=10)
        result = procs.end_leftovers(profile="app-1", grace=2)
        self.assertEqual([r["pid"] for r in result["ended"]], [child.pid])
        self.assertIsNone(other.poll(), "another profile's leftover is not part of this stop")

    def test_garbage_in_the_records_folder_is_ignored(self):
        base = procs.procs_dir()
        (base / "1.json").write_text("{not json", encoding="utf-8")
        (base / "2.json").write_text(json.dumps({"schema": 1, "pid": "no"}), encoding="utf-8")
        (base / "3.json").write_text("x" * 40000, encoding="utf-8")
        self.assertEqual(procs.records(), [])
        self.assertEqual(procs.end_leftovers()["ended"], [])

    def test_an_engine_orphan_of_this_executable_is_found_and_a_live_one_is_not(self):
        # an "engine": this interpreter running --serve; its starter is gone after the middle process exits
        code = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)','--serve'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
                "creationflags=%d); print(p.pid, flush=True)" % NO_WINDOW)
        middle = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        orphan_pid = int(middle.stdout.split()[0])
        self.addCleanup(lambda: procs.start_token(orphan_pid) and os.kill(orphan_pid, 9))
        deadline = time.monotonic() + 10
        while procs.identify(orphan_pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        live = subprocess.Popen([*SLEEP, "--serve"], creationflags=NO_WINDOW)
        self.started.append(live)
        deadline = time.monotonic() + 10
        while procs.identify(live.pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        found = [i.pid for i in procs.orphans_of(sys.executable)]
        self.assertIn(orphan_pid, found)
        self.assertNotIn(live.pid, found, "this test process started it: a live parent")
        self.assertEqual(procs.orphans_of(str(Path(sys.executable).with_name("not-the-engine"))), [])
        ended = procs.end_orphans(sys.executable, grace=2)
        self.assertIn(orphan_pid, [e["pid"] for e in ended])
        self.assertIsNone(live.poll())
        self.assertIsNone(procs.start_token(orphan_pid))

class SubreaperOrphans(unittest.TestCase):
    """An engine adopted by `systemd --user` is an orphan (#3254), and a dying process is waited for by its start token."""

    class Gone(Exception):
        pass

    def fake_psutil(self, ancestors, parent_name):
        class Err(Exception):
            pass

        class NoSuch(Err):
            pass

        def proc(pid, name, ppid=1):
            info = {"pid": pid, "ppid": ppid, "cmdline": ["engine", "--serve"], "create_time": 100.0}
            return mock.Mock(pid=pid, **{"name.return_value": name, "exe.return_value": "/x/engine", "info": info,
                                         "as_dict.return_value": info})

        me = mock.Mock(**{"parents.return_value": [proc(p, n) for p, n in ancestors]})
        engine = proc(7000, "engine", ppid=5000)
        parent = proc(5000, parent_name)
        parent.create_time.return_value = 50.0
        parent.status.return_value = "running"
        fake = mock.Mock(Error=Err, NoSuchProcess=NoSuch, STATUS_ZOMBIE="zombie")
        fake.Process.side_effect = lambda pid=None: me if pid is None else parent
        fake.process_iter.return_value = [engine]
        return fake

    def found(self, ancestors, parent_name):
        fake = self.fake_psutil(ancestors, parent_name)
        with mock.patch.object(procs, "psutil", fake), mock.patch.object(procs, "same_file", return_value=True), \
                mock.patch.object(procs, "identify", side_effect=lambda pid: ("ident", pid)):
            return procs.orphans_of("/x/engine")

    def test_an_engine_adopted_by_systemd_user_is_an_orphan_even_when_it_is_an_ancestor_of_the_terminal(self):
        # systemd --user (pid 5000) is an ancestor of every desktop terminal and adopts what loses its parent
        self.assertEqual(self.found([(5000, "systemd"), (6000, "bash")], "systemd"), [("ident", 7000)])

    def test_an_engine_whose_parent_is_a_real_ancestor_is_not_an_orphan(self):
        # started by this server's own shell: not touched
        self.assertEqual(self.found([(5000, "bash"), (6000, "tmux")], "bash"), [])

    def test_ending_waits_on_the_start_token_not_on_an_empty_command_line(self):
        record = {"pid": 4242, "token": "t1", "cmd": ["engine", "--serve"]}
        answers = ["t1", "t1"]                                   # still exiting twice, then gone
        def token(pid):
            return answers.pop(0) if answers else None
        with mock.patch.object(procs, "WINDOWS", False), mock.patch.object(procs, "is_same", side_effect=[True, False, False]) as same, \
                mock.patch.object(procs, "_signal_verified"), mock.patch.object(procs, "start_token", side_effect=token), \
                mock.patch.object(procs, "drop_record"), mock.patch.object(procs.time, "sleep"):
            self.assertEqual(procs.end_one(record, grace=5), "ended")
        self.assertGreaterEqual(same.call_count, 1)

    def slow_exit(self, exits_after, **kw):
        """end_one of a process that is still there `exits_after` virtual seconds after the kill (a fake clock)."""
        record = {"pid": 4243, "token": "t1", "cmd": ["engine", "--serve"]}
        now = [0.0]
        said = []

        def sleep(s):
            now[0] += s

        def token(pid):
            return "t1" if now[0] < exits_after else None
        with mock.patch.object(procs, "WINDOWS", False), mock.patch.object(procs, "is_same", side_effect=lambda *a: token(0) is not None), \
                mock.patch.object(procs, "_signal_verified"), mock.patch.object(procs, "start_token", side_effect=token), \
                mock.patch.object(procs, "drop_record"), mock.patch.object(procs.signal, "SIGKILL", 9, create=True), \
                mock.patch.object(procs.time, "sleep", side_effect=sleep), \
                mock.patch.object(procs.time, "monotonic", side_effect=lambda: now[0]):
            return procs.end_one(record, say=said.append, **kw), said

    def test_an_engine_that_needs_longer_than_ten_seconds_to_give_its_memory_back_is_waited_for(self):
        # #4051: 50 GiB of pinned memory is released by the kernel after the signal; the next engine started beside it
        outcome, _ = self.slow_exit(25.0)                              # the old limit: 8 s grace + 10 s
        self.assertEqual(outcome, "stuck")
        outcome, said = self.slow_exit(40.0, settle=120.0)
        self.assertEqual(outcome, "ended")
        self.assertTrue(said and "still waiting for process 4243" in said[0], said)
        self.assertEqual(self.slow_exit(40.0, settle=10.0)[0], "stuck")


class WindowsOrphanScan(unittest.TestCase):
    """orphans_of on Windows reads names first and a command line only where the name can be the engine's file: 257 processes took
    4-9 s at every server start when every command line was read (30-150 ms for each protected process that refuses it)."""

    ENGINE = r"C:\Strata\engine\strata-vision-hip.exe"
    OTHER = r"C:\Old\engine\strata-vision-hip.exe"

    def table(self):
        e, o = self.ENGINE, self.OTHER
        # pid: (name or an exception, exe, cmdline or an exception)
        return {
            101: ("strata-vision-hip.exe", e, [e, "--serve"]),        # an orphan
            102: ("STRATA-VISION-HIP.EXE", e, [e, "--serve"]),        # upper case
            103: ("STRATA~1.EXE", e, [e, "--serve"]),                 # started through its 8.3 path
            104: (PermissionError, e, [e, "--serve"]),                # the name cannot be read: read in full
            105: ("svchost.exe", r"C:\Windows\System32\svchost.exe", PermissionError),
            106: ("lsass.exe", r"C:\Windows\System32\lsass.exe", PermissionError),
            107: ("strata-vision-hip.exe", o, [o, "--serve"]),        # another installation's engine
            108: ("strata-vision-hip.exe", e, [e]),                   # not serving
            109: ("chrome.exe", r"C:\Chrome\chrome.exe", ["chrome.exe", "--serve"]),
        }

    def fake_psutil(self, table, asked):
        class Err(Exception):
            pass

        class NoSuch(Err):
            pass

        class Denied(Err):
            pass

        def read(pid, attr):
            name, exe, cmd = table[pid]
            value = {"pid": pid, "ppid": 4242, "create_time": 100.0, "name": name, "cmdline": cmd}[attr]
            if attr == "cmdline":
                asked.append(pid)
            if isinstance(value, type) and issubclass(value, Exception):
                raise Denied()
            return value

        def as_dict(pid, attrs):
            out = {}
            for a in attrs:
                try:
                    out[a] = read(pid, a)
                except Denied:
                    out[a] = None                                    # as psutil: a refused value is None
            return out

        def proc(pid, attrs):
            p = mock.Mock(pid=pid, info=as_dict(pid, attrs))
            p.as_dict.side_effect = lambda attrs=None: as_dict(pid, attrs)
            p.exe.side_effect = lambda: table[pid][1]
            return p

        def process(pid=None):
            if pid is None:
                return mock.Mock(**{"parents.return_value": []})
            raise NoSuch()                                           # every parent (4242) is gone: an orphan

        fake = mock.Mock(Error=Err, NoSuchProcess=NoSuch, STATUS_ZOMBIE="zombie")
        fake.Process.side_effect = process
        fake.process_iter.side_effect = lambda attrs: [proc(pid, attrs) for pid in table]
        return fake

    def scan(self, windows, table=None):
        table = table or self.table()
        asked = []
        same = lambda a, b: bool(a) and bool(b) and a.lower() == b.lower()   # noqa: E731
        with mock.patch.object(procs, "psutil", self.fake_psutil(table, asked)), mock.patch.object(procs, "WINDOWS", windows), \
                mock.patch.object(procs, "same_file", side_effect=same), mock.patch.object(procs, "_short_name", return_value=None), \
                mock.patch.object(procs, "identify", side_effect=lambda pid: ("ident", pid)):
            found = sorted(i[1] for i in procs.orphans_of(self.ENGINE))
        return found, asked

    def test_the_same_orphans_as_reading_every_process(self):
        new, _ = self.scan(windows=True)
        old, _ = self.scan(windows=False)                            # the full read, as before (and as Linux and macOS still do)
        self.assertEqual(new, old)
        self.assertEqual(new, [101, 102, 103, 104])

    def test_a_process_that_cannot_be_the_engine_is_never_asked_for_its_command_line(self):
        _, asked = self.scan(windows=True)
        self.assertEqual(sorted(asked), [101, 102, 103, 104, 107, 108])
        _, asked_before = self.scan(windows=False)
        self.assertEqual(sorted(asked_before), sorted(self.table()))

    def test_the_files_own_short_name_matches(self):
        self.assertTrue(procs._name_may_be("SVIHIP~2.EXE", "strata-vision-hip.exe", "svihip~2.exe"))
        self.assertTrue(procs._name_may_be("STRATA~3.EXE", "strata-vision-hip.exe", None))
        self.assertFalse(procs._name_may_be("svchost.exe", "strata-vision-hip.exe", None))
        self.assertFalse(procs._name_may_be("OTHER~1.EXE", "strata-vision-hip.exe", None))

    def test_a_hard_link_under_another_name_is_left_to_the_full_read_only_off_windows(self):
        # The full read matches any name that is the same file (os.path.samefile); the name filter does not. Strata never links an
        # engine under another name (setup's links are model files under their own name), so this is a hand-made case.
        e = self.ENGINE
        table = {201: ("my-engine.exe", e, [e, "--serve"])}
        self.assertEqual(self.scan(windows=False, table=table)[0], [201])
        self.assertEqual(self.scan(windows=True, table=table)[0], [])


if __name__ == "__main__":
    unittest.main()
