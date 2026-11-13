# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finish marks of a model folder that cannot be written are kept in XeStrata's own folder (so a big model file is
hashed once, not on every launch); a pinned file copied in by hand is checked here, without the network; and a
preparation tool fetches llama.cpp only when it runs without it (upstream b3c390914).

    python -m unittest tools.test_setup_marks
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class KeptMark(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xestrata-marks-"))
        self.state = self.tmp / "state"
        p = mock.patch.object(setup, "STATE", self.state)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: (os.chmod(self.tmp, 0o755), self._rm()))

    def _rm(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_read_only_folder_keeps_its_mark_in_the_state_folder(self):
        folder = self.tmp / "model"
        folder.mkdir()
        f = folder / "a.gguf"
        f.write_bytes(b"x" * 8)
        os.chmod(folder, 0o555)
        setup.mark(f, "sha256 abc")
        self.assertTrue(setup.done(f))
        self.assertEqual(setup.mark_text(f), "sha256 abc")
        kept = list((self.state / "marks").glob("*.done"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), "sha256 abc")

    def test_a_file_that_changed_has_no_mark(self):
        folder = self.tmp / "model"
        folder.mkdir()
        f = folder / "a.gguf"
        f.write_bytes(b"x" * 8)
        os.chmod(folder, 0o555)
        setup.mark(f, "sha256 abc")
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.assertFalse(setup.done(f))


class LocalPin(unittest.TestCase):
    def test_a_pinned_file_copied_in_is_checked_without_the_network(self):
        tmp = Path(tempfile.mkdtemp(prefix="xestrata-pin-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        data = b"GGUF" + b"\0" * 60
        dst = tmp / "m.gguf"
        dst.write_bytes(data)
        pin = (len(data), hashlib.sha256(data).hexdigest())
        with mock.patch.object(setup, "file_pin", return_value=pin), \
                mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            setup.download_file("https://example.invalid/m.gguf", dst)
        self.assertTrue(setup.done(dst))
        self.assertIn(f"sha256 {pin[1]}", setup.mark_text(dst))


class PreparationFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xestrata-llama-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_a_preparation_tool_fetches_llama_cpp_when_it_is_missing(self):
        llama = self.tmp / "third_party" / "main" / "llama.cpp"
        with mock.patch.object(setup, "ROOT", self.tmp), mock.patch.object(setup, "get_llama_cpp") as fetch, \
                mock.patch.object(setup, "say"):
            setup.run([sys.executable, "-c", "pass"], env={"XESTRATA_GGUF_PY": str(llama / "gguf-py")})
        fetch.assert_called_once()

    def test_a_gguf_py_of_the_person_s_own_asks_for_nothing(self):
        mine = self.tmp / "my-llama" / "gguf-py"
        with mock.patch.object(setup, "ROOT", self.tmp), mock.patch.object(setup, "get_llama_cpp") as fetch, \
                mock.patch.object(setup, "say"):
            setup.run([sys.executable, "-c", "pass"], env={"XESTRATA_GGUF_PY": str(mine)})
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
