# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""An uploaded file's name must not become another command of the image engine's line protocol.

    python -m unittest serve.test_videos
"""
from __future__ import annotations

import io
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import videos  # noqa: E402


class UploadName(unittest.TestCase):
    def test_the_extension_has_no_protocol_words(self):
        self.assertEqual(videos.safe_suffix("clip.mp4"), ".mp4")
        self.assertEqual(videos.safe_suffix("clip.2024 final"), ".2024final")
        self.assertEqual(videos.safe_suffix("x.a\nSHUTDOWN\nX"), ".aSHUTDOWNX")
        self.assertEqual(videos.safe_suffix("noext"), "")
        self.assertEqual(videos.safe_suffix("clip."), "")
        self.assertEqual(videos.safe_suffix("x.a b"), ".ab")

    def test_the_written_path_is_one_word(self):
        body = (b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.a\nSHUTDOWN\nX\"\r\n"
                b"Content-Type: video/mp4\r\n\r\nbytes\r\n--b--\r\n")
        with tempfile.TemporaryDirectory() as tmp:
            _, files = videos.stream_multipart(io.BytesIO(body), "multipart/form-data; boundary=b", len(body),
                                               Path(tmp), 1 << 20)
            self.assertEqual(len(files), 1)
            path = files[0]["path"]
            self.assertEqual(path.parent, Path(tmp))
            self.assertNotIn("\n", path.name)
            self.assertNotIn(" ", path.name)
            self.assertTrue(path.is_file())


class DeletedJob(unittest.TestCase):
    def test_a_deleted_job_does_not_stop_the_worker(self):
        # delete() waits at most 60 s for a running job, then removes its folder; the worker must survive writing
        # that job's final record and go on with the queue.
        def job(n):
            return {"width": 64, "height": 64, "n": n,
                    "source": {"width": 64, "height": 64, "fps": 30.0, "vfr": False, "frames": 1,
                               "duration": 0.03}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            started, release = threading.Event(), threading.Event()

            def run(job, progress, cancel, source, stem, model):
                if job["n"] == 1:
                    started.set()
                    release.wait(3)
                    shutil.rmtree(source.parent, ignore_errors=True)
                return {"path": str(stem / "out.mp4"), "mime": "video/mp4"}

            jobs = videos.VideoJobs(root, run)
            ids = []
            for n in (1, 2):
                vid, folder = jobs.new_dir()
                source = folder / "input.mp4"
                source.write_bytes(b"video")
                jobs.create(vid, "auto", job(n), source)
                ids.append(vid)
                if n == 1:
                    self.assertTrue(started.wait(2))
            release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and (jobs.get(ids[1]) or {}).get("status") != "completed":
                time.sleep(0.01)
            self.assertEqual(jobs.get(ids[1])["status"], "completed")
            self.assertIsNone(jobs.get(ids[0]))   # its folder went away with the delete


if __name__ == "__main__":
    unittest.main()
