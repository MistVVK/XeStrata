# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A model repository Hugging Face answers 401 for (private, or gone) does not stop a first install (the user's decision
of 2026-11-22): the step of FIRST_CHOICE that names it is passed over with a line that says which family and which
repository, and the next step is offered; setup stops, saying so, when no step is left.  A download of such a file
(a family chosen with --family) stops at once with the reason, not after five tries as a network failure.  setup takes
no Hugging Face login.  No network: the answers are mocked.

    python -m unittest tools.test_setup_closed_repo
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

FAMS = [f for f in setup.FAMILIES if f not in (setup.CUSTOM, "edge0")]


def room(ram: float, vram: float | None) -> setup.Room:
    return setup.room_of(ram, [] if vram is None else [{"vram_gb": vram, "integrated": False}])


def closed(*repos: str, other: Exception | None = None):
    """urlopen answering 401 for the API page of each of `repos`, `other` (raised) or 200 for the rest; the requests
    it saw are in its .seen"""
    seen: list = []

    def urlopen(req, timeout=None):
        seen.append(req)
        url = req.full_url
        if any(url.endswith(f"/api/models/{r}") for r in repos):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]
        if other is not None:
            raise other
        return io.BytesIO(b"")
    patch = mock.patch.object(setup.urllib.request, "urlopen", side_effect=urlopen)
    patch.seen = seen  # type: ignore[attr-defined]
    return patch


def model_repo(family: str) -> str:
    return next(r for role, r in setup.family_repos(family) if role == "model")


class FirstOffered(unittest.TestCase):
    def offered(self, patch, ram, vram, fams=FAMS) -> tuple[tuple, str]:
        out = io.StringIO()
        with patch, contextlib.redirect_stdout(out):
            got = setup.first_offered(fams, room(ram, vram))
        return got, out.getvalue()

    def test_open_repositories_change_nothing(self):
        for ram, vram in ((123.2, 32), (74.4, 16), (32, 12), (20, 16), (2, 1), (32, None)):
            with self.subTest(ram=ram, vram=vram):
                got, said = self.offered(closed(), ram, vram)
                self.assertEqual(got, setup.first_choice(FAMS, room(ram, vram)))
                self.assertEqual(said, "")

    def test_a_closed_step_is_passed_over_for_the_next(self):
        # 20 GB of RAM and a 16 GB card: Inkling-2-Mini (QSA), the 4th step, else SenseNova-Loop-9B (QSA), the 5th,
        # else Qwen3.5-2B, the 6th
        self.assertEqual(setup.first_choice(FAMS, room(20, 16))[0], "inkling-qsa")
        got, said = self.offered(closed(setup._INKLING), 20, 16)
        self.assertEqual(got, ("sensenova-loop-qsa", None))
        self.assertIn(setup.FAMILIES["inkling-qsa"]["title"], said)
        got, said = self.offered(closed(setup._INKLING, setup.SENSENOVA_QSA_REPO), 20, 16)
        self.assertEqual(got, ("qwen35-2b", None))
        self.assertIn(setup.FAMILIES["sensenova-loop-qsa"]["title"], said)
        self.assertIn(setup.SENSENOVA_QSA_REPO, said)
        self.assertIn("401", said)
        self.assertIn("passed over", said)

    def test_each_closed_step_in_turn(self):
        # 32 GB of RAM and a 16 GB card: Sync-S; Sync's, Inkling's and SenseNova's repositories closed -> Qwen3.5-2B
        r = room(32, 16)
        self.assertEqual(setup.first_choice(FAMS, r), ("sync", None))
        self.assertEqual(setup.first_choice([f for f in FAMS if f != "sync"], r)[0], "inkling-qsa")
        got, said = self.offered(closed(setup.SYNC_REPOS["Sync-S"], setup._INKLING, setup.SENSENOVA_QSA_REPO), 32, 16)
        self.assertEqual(got[0], "qwen35-2b")
        self.assertIn(setup.SYNC_REPOS["Sync-S"], said)
        self.assertIn(setup.SENSENOVA_QSA_REPO, said)

    def test_no_step_left_stops_with_the_reason(self):
        last = model_repo(setup.LAST_CHOICE)
        for ram, vram in ((20, 16), (32, None)):      # the last step after a closed one; on the CPU, the only one
            with self.subTest(ram=ram, vram=vram):
                with self.assertRaises(SystemExit) as stop:
                    self.offered(closed(setup._INKLING, setup.SENSENOVA_QSA_REPO, last), ram, vram)
                self.assertEqual(stop.exception.code, 1)
        out = io.StringIO()
        with closed(setup._INKLING, setup.SENSENOVA_QSA_REPO, last), contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit):
            setup.first_offered(FAMS, room(20, 16))
        said = out.getvalue()
        self.assertIn(setup.SENSENOVA_QSA_REPO, said)
        self.assertIn(last, said)
        self.assertIn("[X]", said)
        self.assertIn("--family", said)

    def test_no_answer_is_not_a_verdict(self):
        # offline, or a host that answers something else: the step stays; the download says what is wrong, as before
        missing = urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]
        for err in (urllib.error.URLError("no DNS"), TimeoutError(), missing):
            with self.subTest(err=err):
                got, said = self.offered(closed(other=err), 20, 16)
                self.assertEqual(got, ("inkling-qsa", None))
                self.assertEqual(said, "")

    def test_the_probe_is_a_head_without_a_login(self):
        patch = closed()
        with patch:
            self.assertFalse(setup.repo_closed(setup.SENSENOVA_QSA_REPO))
        (req,) = patch.seen
        self.assertEqual(req.get_method(), "HEAD")
        self.assertEqual(req.full_url, f"{setup.hf_endpoint()}/api/models/{setup.SENSENOVA_QSA_REPO}")
        self.assertFalse(req.has_header("Authorization"))


class ClosedDownload(unittest.TestCase):
    def test_a_401_stops_at_once_with_the_reason(self):
        url = setup.hf(setup.SENSENOVA_QSA_REPO) + "Q4_K_M-qsa.gguf"
        err = urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup.download_core, "open_url", side_effect=err) as head, \
                mock.patch.object(setup.time, "sleep") as sleep, \
                mock.patch.object(setup, "model_source", return_value="huggingface"), \
                contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.download_file(url, Path(tmp) / "Q4_K_M-qsa.gguf", "SenseNova-Loop-9B Q4_K_M")
        self.assertEqual(head.call_count, 1)
        sleep.assert_not_called()
        said = out.getvalue()
        self.assertIn(setup.SENSENOVA_QSA_REPO, said)
        self.assertIn("401", said)
        self.assertIn("private", said)
        self.assertNotIn("internet", said)


if __name__ == "__main__":
    unittest.main()
