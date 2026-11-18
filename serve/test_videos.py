# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""An uploaded file's name must not become another command of the image engine's line protocol.

    python -m unittest serve.test_videos
"""
from __future__ import annotations

import io
import sys
import tempfile
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
        self.assertEqual(videos.safe_suffix("clip."), ".")

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


if __name__ == "__main__":
    unittest.main()
