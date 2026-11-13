# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/dflash2_fetch.py: a pinned download behind a mocked urlopen (nothing is downloaded), the pack and its
selector conversion checked against their pins, a stopped download resumed, an offline fetch refused in plain words.

    python -m unittest tools.test_dflash2_fetch
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import dflash2_fetch as F  # noqa: E402
import dflash2_pack as P  # noqa: E402
from test_dflash2_pack import CFG, tensors, write_safetensors  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        # a small synthetic drafter and its packs (the real pins are 1.2 GB; the flow is the same)
        write_safetensors(cls.dir / "model.safetensors", tensors())
        (cls.dir / "config.json").write_text(json.dumps(CFG))
        cls.st = (cls.dir / "model.safetensors").read_bytes()
        cls.cfg = (cls.dir / "config.json").read_bytes()
        P.pack(str(cls.dir / "model.safetensors"), str(cls.dir / "config.json"), str(cls.dir / "q4.pack"),
               say=lambda *_: None, q4=True)
        P.convert_selector_q4(str(cls.dir / "q4.pack"), str(cls.dir / "selq4.pack"), say=lambda *_: None)
        cls.q4 = (cls.dir / "q4.pack").read_bytes()
        cls.selq4 = (cls.dir / "selq4.pack").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def drafter(self, pinned=True):
        """The synthetic drafter's pins (its repository and revision those of Qwen3.8's, so the pack's source line,
        and with it its bytes, are what the class's packs hold)."""
        q38 = F.DRAFTERS[F.DEFAULT]
        files = (("config.json", len(self.cfg), sha(self.cfg)), ("model.safetensors", len(self.st), sha(self.st)))
        return F.Drafter("test", q38.repo, q38.revision, files, "t.q4_0.pack",
                         (len(self.q4), sha(self.q4)) if pinned else None, "t.q4_0-selq4.pack",
                         (len(self.selq4), sha(self.selq4)) if pinned else None, 0.1, 0.1, "NOTICE.md")

    def setUp(self):
        self.out = self.dir / ("out-%s" % self._testMethodName)
        self.d = self.drafter()

    def serve(self, request, timeout=None):
        url = request.full_url
        name = url.rsplit("/", 1)[1]
        body = {"config.json": self.cfg, "model.safetensors": self.st}[name]
        rng = request.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return Response(body[start:], 206, {"Content-Range": "bytes %d-%d/%d" % (start, len(body) - 1, len(body))})
        return Response(body)

    def test_fetch_packs_converts_and_removes_the_download(self):
        with mock.patch("urllib.request.urlopen", side_effect=self.serve):
            F.fetch(self.d, self.out)
        got = F.ready(self.d, self.out)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], self.d.converted_name)
        self.assertEqual((self.out / self.d.converted_name).read_bytes(), self.selq4)
        self.assertEqual((self.out / self.d.pack_name).read_bytes(), self.q4)
        self.assertFalse((self.out / "download").exists(), "the 1.2 GB download is removed")
        self.assertTrue((self.out / "LICENSE").is_file())
        self.assertTrue((self.out / "NOTICE.md").is_file())
        self.assertEqual(F.verify(self.d, self.out), 0)

    def test_a_ready_install_is_not_fetched_again(self):
        (self.out).mkdir(parents=True, exist_ok=True)
        (self.out / self.d.converted_name).write_bytes(self.selq4)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no network for a ready install")):
            F.fetch(self.d, self.out)
        self.assertEqual(F.verify(self.d, self.out), 0)

    def test_a_q4_pack_is_converted_without_a_download(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / self.d.pack_name).write_bytes(self.q4)
        # ready() finds the Q4 pack; fetch still converts the selector (the Q4 pack alone is not the converted one)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no network: the Q4 pack is here")):
            F.fetch(self.d, self.out)
        self.assertEqual((self.out / self.d.converted_name).read_bytes(), self.selq4)

    def test_a_corrupt_pack_is_found_by_verify(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / self.d.converted_name).write_bytes(self.selq4[:-1] + b"\x00")
        self.assertIsNone(F.ready(self.d, self.out))
        self.assertEqual(F.verify(self.d, self.out), F.BAD)

    def test_a_stopped_download_continues_where_it_stopped(self):
        self.out.mkdir(parents=True, exist_ok=True)
        work = self.out / "download"
        work.mkdir()
        half = len(self.st) // 2
        (work / "model.safetensors.part").write_bytes(self.st[:half])
        seen = []

        def serve(request, timeout=None):
            seen.append(request.headers.get("Range"))
            return self.serve(request, timeout)

        with mock.patch("urllib.request.urlopen", side_effect=serve):
            F.fetch(self.d, self.out)
        self.assertIn("bytes=%d-" % half, seen, "the .part is resumed with a Range")
        self.assertEqual(F.verify(self.d, self.out), 0)

    def test_offline_is_refused_in_plain_words(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no route to host")):
            with self.assertRaises(SystemExit) as e:
                F.fetch(self.d, self.out)
        self.assertIn("MTP", str(e.exception))

    def test_an_unpinned_drafter_keeps_the_pack_it_made(self):
        # a drafter whose packs have no pin yet: the download is still checked, the packs are kept as made
        self.d = self.drafter(pinned=False)
        with mock.patch("urllib.request.urlopen", side_effect=self.serve):
            F.fetch(self.d, self.out)
        self.assertEqual(F.ready(self.d, self.out)[1], self.d.converted_name)
        self.assertEqual((self.out / self.d.converted_name).read_bytes(), self.selq4)
        self.assertFalse(any(p.suffix == ".tmp" for p in self.out.iterdir()))

    def test_wrong_bytes_are_refused(self):
        def serve(request, timeout=None):
            r = self.serve(request, timeout)
            if request.full_url.endswith("model.safetensors") and not request.headers.get("Range"):
                return Response(b"\x00" * len(self.st))
            return r

        with mock.patch("urllib.request.urlopen", side_effect=serve):
            with self.assertRaises(SystemExit):
                F.fetch(self.d, self.out)
        self.assertIsNone(F.ready(self.d, self.out))


if __name__ == "__main__":
    unittest.main()
