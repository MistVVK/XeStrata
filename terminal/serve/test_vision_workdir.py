"""#3370: the encoder's work folder is made again when a temp cleaner removed it while the encoder was unloaded.
#5755: and when it is removed while the encoder is running, that encoder's working directory is the removed folder,
so its files are invisible to it: it is started again before the next picture.

    python -m serve.test_vision_workdir
"""
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server


class WorkDir(unittest.TestCase):
    def test_start_makes_a_removed_work_folder_again(self):
        proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("READY\n"), kill=lambda: None,
                               wait=lambda timeout=None: 0, poll=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            v = server.Vision.__new__(server.Vision)
            v.spawn = (["strata-vision"], None, None)
            v.dir = Path(tmp) / "strata-vision-gone"
            v.cache = {"k": (v.dir / "k.sve", 4)}
            seen = []

            def fake_popen(*a, **k):
                seen.append(Path(k["cwd"]).is_dir())      # the real popen raises WinError 267 when it is not
                return proc

            with mock.patch.object(server, "popen", fake_popen):
                v._start()
            self.assertEqual(seen, [True])
            self.assertTrue(v.dir.is_dir())
            self.assertEqual(v.cache, {})                 # its cached embeddings went with the folder

    def test_an_existing_folder_and_its_cache_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = server.Vision.__new__(server.Vision)
            v.dir = Path(tmp)
            v.cache = {"k": (v.dir / "k.sve", 4)}
            v._ensure_dir()
            self.assertEqual(len(v.cache), 1)


@unittest.skipIf(sys.platform == "win32", "Windows does not let a folder go while a process runs in it, so the case cannot happen there")
class RunningReader(unittest.TestCase):
    """#5755: the temp cleaner removed the work folder while a reader was running.

    The reader runs with its `cwd` in the folder and the `ENC` lines name the files relative to it (#480), so a
    folder that is removed and made again is invisible to the reader that runs: its working directory is the
    removed one (its `/proc/<pid>/cwd` reads `(deleted)`).  Every picture then fails with
    `cannot read the image <hash>.img`, for as long as the server lives.  The reader here is a real process, so
    its `cwd` behaves exactly as on the server: it answers `OK` only when the file it was named is really there.
    """

    READER = (
        "import os, sys\n"
        "print('READY fake reader', flush=True)\n"
        "for line in sys.stdin:\n"
        "    p = line.split()\n"
        "    if p[:1] == ['QUIT']:\n"
        "        break\n"
        "    if p[:1] == ['ENC'] and len(p) == 3:\n"
        "        if os.path.exists(p[1]):\n"
        "            open(p[2], 'wb').write(b'sve')\n"
        "            print('OK 7', flush=True)\n"
        "        else:\n"
        "            print('ERR cannot read the image ' + p[1], flush=True)\n"
    )
    PNG = b"\x89PNG\r\n\x1a\n" + b"iVBORw0KGgo=" * 8

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.started: list[Path] = []            # the cwd of every reader the server started
        self.procs: list[subprocess.Popen] = []  # and the readers themselves

    def popen(self, what, args, role="engine", **kw):
        self.started.append(Path(kw["cwd"]))
        kw.pop("role", None)
        proc = subprocess.Popen(args, **kw)
        self.procs.append(proc)
        return proc

    def vision(self):
        """A reader as the server holds one: started, resident, with an empty cache in its own folder."""
        v = server.Vision.__new__(server.Vision)
        v.spawn = ([sys.executable, "-c", self.READER], None, None)
        v.fallback = None
        v.dir = self.tmp / "strata-vision-running"
        v.dir.mkdir()
        v.cache = {}
        v.lock = threading.Lock()
        v.max_tokens = 0
        v.proc = None
        v.stopped = True
        return v

    def test_a_picture_is_read_after_the_folder_of_a_running_reader_returns(self):
        v = self._start()
        first = self.procs[-1]
        shutil.rmtree(v.dir)                                    # what systemd-tmpfiles does after 10 idle days
        with mock.patch.object(server, "popen", self.popen):
            out, tokens = v.encode(self.PNG)
        self.assertEqual(tokens, 7)
        self.assertTrue(out.is_file())                          # the reader wrote where the server can read it
        self.assertEqual(self.started, [v.dir])                 # that reader's folder, made again
        self.assertEqual(len(self.procs), 2)                    # the reader was started again ...
        first.wait(timeout=10)
        self.assertIsNotNone(first.returncode)                  # ... and the one whose cwd was removed is gone

    def test_a_reader_whose_folder_is_intact_is_not_started_again(self):
        v = self._start()
        running = self.procs[-1]
        with mock.patch.object(server, "popen", self.popen):
            self.assertEqual(v.encode(self.PNG)[1], 7)
            self.assertEqual(v.encode(self.PNG)[1], 7)          # the second time from the cache
        self.assertEqual(self.started, [])                      # no reader started for these two pictures
        self.assertIs(v.proc, running)

    def test_the_read_only_all_cached_does_not_restart_a_reader(self):
        v = self._start()
        first = v.proc
        shutil.rmtree(v.dir)
        with mock.patch.object(server, "popen", self.popen):
            self.assertFalse(v.all_cached([self.PNG]))          # its cached embeddings went with the folder
        self.assertEqual(self.started, [])                      # a check that reads nothing starts nothing
        self.assertIs(v.proc, first)

    def _start(self):
        with mock.patch.object(server, "popen", self.popen):
            v = self.vision()
            v._start()
        self.assertTrue(v.alive())
        self.started.clear()                        # count only the starts the test is about
        self.addCleanup(v.close)
        return v


if __name__ == "__main__":
    unittest.main()
