"""#5601: the start waits for the owner it launched to publish its state. The owner creates the state file as its lease and writes it a moment
later, so a read in between saw 0 bytes and the start (an update's test start) failed with "runtime state file is unreadable"."""

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import runtime
from strata_control.profiles import ProfileError


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class PendingState(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-state-pending-")
        self.addCleanup(temp.cleanup)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(Path(temp.name) / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        self.path = runtime._paths("updtest")[0]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: bytes):
        descriptor = runtime._write_new(self.path, data)       # the owner's own writer: a private file, the lease held
        self.addCleanup(os.close, descriptor)

    def test_an_empty_state_file_is_not_there_yet_for_the_waiting_start(self):
        self.write(b"")
        self.assertIsNone(runtime._read_state("updtest", pending_ok=True))
        with self.assertRaisesRegex(ProfileError, "unreadable|not private or is invalid"):
            runtime._read_state("updtest")                     # everywhere else an empty file is still an error, as before

    def test_a_half_written_state_file_is_not_there_yet_for_the_waiting_start(self):
        self.write(b'{"schema_version": 1, "prof')
        self.assertIsNone(runtime._read_state("updtest", pending_ok=True))
        with self.assertRaises(ProfileError):
            runtime._read_state("updtest")

    def test_a_written_state_of_another_shape_is_still_refused(self):
        self.write(b'{"schema_version": 2}\n')
        try:
            runtime._read_state("updtest", pending_ok=True)
        except ProfileError as exc:
            self.assertIn("runtime state file", str(exc))
        else:
            self.fail("a complete state file of the wrong shape must not read as waiting")

    def test_the_start_waits_with_pending_ok(self):
        src = inspect.getsource(runtime._start_once)                   # (#5400: start wraps _start_once, the one start)
        self.assertIn("while _read_state(name, pending_ok=True) is None and proc.poll() is None", src)
        self.assertIn("if _read_state(name, pending_ok=True) is None:", src)


if __name__ == "__main__":
    unittest.main()
