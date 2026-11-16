"""#5601, the start's own wait (model-free): the owner creates its state file as its lease and fills it a moment later. The start that launched
it must wait through an empty and a half-written file, end at once when that owner exits before it has published, end at its deadline when
the file never becomes valid, and never let two owners hold one profile's state."""

import contextlib
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_control import runtime
from strata_control.profiles import ProfileError

NAME = "updtest"
TOKEN = "ab" * 32


class _FakeOwner:
    """Stands in for the owner process `start` launches: writes the state file in steps (`chunks`, each after `gap` s) and then lives on,
    or exits after its last step (`exit_after`)."""

    def __init__(self, chunks, gap=0.15, exit_after=False):
        self.returncode = None
        self.fd = None
        self.done = threading.Event()
        self.chunks, self.gap, self.exit_after = chunks, gap, exit_after

    def launch(self, *args, **kwargs):
        """subprocess.Popen's stand-in: the owner begins when the start launches it."""
        path = runtime._paths(NAME)[0]
        chunks, gap, exit_after = self.chunks, self.gap, self.exit_after

        def run():
            self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # the lease: the file exists, still empty
            for chunk in chunks:
                time.sleep(gap)
                os.write(self.fd, chunk)
            if exit_after:
                os.close(self.fd)
                self.fd = None
                self.returncode = 1
            self.done.set()

        threading.Thread(target=run, daemon=True).start()
        return self

    def poll(self):
        return self.returncode

    def close(self):
        self.done.wait(5)
        if self.fd is not None:
            os.close(self.fd)


def _valid_state() -> bytes:
    socket = runtime._paths(NAME)[1]
    if os.name == "nt":
        socket = "tcp:127.0.0.1:50123"
    return (runtime.json.dumps({"schema_version": 1, "profile": NAME, "socket": socket, "token": TOKEN, "host_pid": 1}) + "\n").encode()


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class StartWaitsForItsOwner(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-state-wait-")
        self.addCleanup(temp.cleanup)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(Path(temp.name) / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        runtime._paths(NAME)[0].parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(runtime._paths(NAME)[0].parent, 0o700)
        self.addCleanup(runtime._STARTED_HOSTS.pop, NAME, None)
        for target, value in (("_refuse_deepseek", None), ("_engine_preflight", None)):
            patch = mock.patch.object(runtime, target, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        for target, value in (("_repair_config_paths", (None, None)), ("_check_start_candidate", ({"profile_revision": "0" * 16}, "d"))):
            patch = mock.patch.object(runtime, target, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        patches = [mock.patch.object(runtime.procs, "end_leftovers", return_value={}),
                   mock.patch.object(runtime.profiles, "_lock", lambda name: contextlib.nullcontext())]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _start_with(self, owner: _FakeOwner, **kw):
        self.addCleanup(owner.close)
        with mock.patch.object(runtime.subprocess, "Popen", side_effect=owner.launch):
            began = time.monotonic()
            try:
                return runtime.start(NAME, wait=0, **kw), time.monotonic() - began
            finally:
                owner.done.wait(5)

    def test_empty_then_half_written_then_valid_is_waited_for(self):
        data = _valid_state()
        result, _ = self._start_with(_FakeOwner([b"", data[:20], data[20:]]))
        self.assertEqual(result["state"], "starting")
        self.assertEqual(runtime._read_state(NAME)["token"], TOKEN)

    def test_an_owner_that_exits_before_it_published_fails_the_start_at_once(self):
        with self.assertRaisesRegex(ProfileError, "failed to start") as caught:
            self._start_with(_FakeOwner([b'{"schema_version": 1, "pro'], exit_after=True))
        self.assertEqual(caught.exception.exit_code, 6)

    def test_an_owner_that_exits_fails_the_start_well_before_the_deadline(self):
        began = time.monotonic()
        with self.assertRaises(ProfileError):
            self._start_with(_FakeOwner([b"{"], gap=0.05, exit_after=True))
        self.assertLess(time.monotonic() - began, 3.0)      # the deadline is 5 s: the start saw the exit, not the clock

    def test_a_file_that_never_becomes_valid_ends_at_the_deadline(self):
        began = time.monotonic()
        with self.assertRaisesRegex(ProfileError, "failed to start"):
            self._start_with(_FakeOwner([b'{"schema_version": 1, "pro']))
        self.assertGreaterEqual(time.monotonic() - began, 4.5)

    def test_a_second_owner_cannot_take_a_published_lease(self):
        first = runtime._write_new(runtime._paths(NAME)[0], _valid_state())
        self.addCleanup(os.close, first)
        with self.assertRaises(OSError):
            runtime._write_new(runtime._paths(NAME)[0], _valid_state())
        self.assertEqual(runtime._read_state(NAME)["token"], TOKEN)       # the first owner's file is untouched

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_a_link_in_place_of_the_state_file_is_refused_also_while_waiting(self):
        target = runtime._paths(NAME)[0].parent / "elsewhere.json"
        target.write_bytes(b"")
        os.symlink(target, runtime._paths(NAME)[0])
        with self.assertRaisesRegex(ProfileError, "not private or is invalid"):
            runtime._read_state(NAME, pending_ok=True)


if __name__ == "__main__":
    unittest.main()
