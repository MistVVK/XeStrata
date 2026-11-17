"""#3789 (PR #3789's open point): an engine process whose parent or start time could not be read (psutil answers None for a value it
may not read, as_dict's default) is not provably an orphan. The base compared the parent's start time with None, which raised a
TypeError that orphans_of did not catch, or, with an unknown parent, called the process an orphan. Now it is left alone."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_control import procs


class Err(Exception):
    pass


class NoSuch(Err):
    pass


def fake_psutil(info: dict):
    engine = mock.Mock(pid=7000, info=info, **{"exe.return_value": "/x/engine", "as_dict.return_value": info})
    parent = mock.Mock(pid=5000, **{"name.return_value": "bash", "create_time.return_value": 50.0, "status.return_value": "running"})
    me = mock.Mock(**{"parents.return_value": []})
    fake = mock.Mock(Error=Err, NoSuchProcess=NoSuch, STATUS_ZOMBIE="zombie")
    fake.Process.side_effect = lambda pid=None: me if pid is None else parent
    fake.process_iter.return_value = [engine]
    return fake


class UnknownIdentity(unittest.TestCase):
    def found(self, **over):
        info = {"pid": 7000, "ppid": 5000, "cmdline": ["engine", "--serve"], "create_time": 100.0, **over}
        with mock.patch.object(procs, "psutil", fake_psutil(info)), mock.patch.object(procs, "WINDOWS", False), \
                mock.patch.object(procs, "same_file", return_value=True), \
                mock.patch.object(procs, "identify", side_effect=lambda pid: ("ident", pid)):
            return procs.orphans_of("/x/engine")

    def test_an_unknown_start_time_is_left_alone_and_does_not_raise(self):
        self.assertEqual(self.found(create_time=None), [])

    def test_an_unknown_parent_is_not_called_an_orphan(self):
        self.assertEqual(self.found(ppid=None), [])

    def test_a_live_parent_and_a_dead_parent_are_as_before(self):
        self.assertEqual(self.found(), [], "the parent (started at 50) is alive and older: started it")
        info = {"pid": 7000, "ppid": 5000, "cmdline": ["engine", "--serve"], "create_time": 100.0}
        fake = fake_psutil(info)
        fake.Process.side_effect = lambda pid=None: mock.Mock(**{"parents.return_value": []}) if pid is None else (_ for _ in ()).throw(NoSuch())
        with mock.patch.object(procs, "psutil", fake), mock.patch.object(procs, "WINDOWS", False), \
                mock.patch.object(procs, "same_file", return_value=True), \
                mock.patch.object(procs, "identify", side_effect=lambda pid: ("ident", pid)):
            self.assertEqual(procs.orphans_of("/x/engine"), [("ident", 7000)], "the parent is gone: an orphan")


if __name__ == "__main__":
    unittest.main()
