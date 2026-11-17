"""0.2.16: the Terminal's words for an Intel Arc card. The pack step runs on Strata's own GGUF reader and downloads nothing (setup.py
prepare_pack_only; tests/terminal/test_intel_prepare.py), so the plan, the step line and the failure hint no longer speak of a helper fetched
from GitHub (owner-approved sentence, 11-16)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from tests.terminal.launch_support import HW_12, Home, catalog

INTEL_SENTENCE = "For the Intel card Strata builds the model's pack from your files itself, once; it takes a few minutes and downloads nothing."


class FakeChild:
    def __init__(self, code: int) -> None:
        self.code = code

    def wait(self) -> int:
        return self.code

    def poll(self) -> int:
        return self.code


class IntelWords(unittest.TestCase):
    def runner(self):
        from strata_cli.launch.runner import Runner
        events = []
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off"), catalog=catalog(HW_12),
                   emit=events.append)
        return r, events

    def test_the_plan_says_the_intel_pack_downloads_nothing(self):
        from strata_cli.launch.runner import Runner
        with Home(), mock.patch.object(Runner, "intel_chosen", return_value=True):
            r, _ = self.runner()
            text = r.download_plan()["text"]
        self.assertIn(" " + INTEL_SENTENCE, text)
        for gone in ("helper", "gguf-py", "GitHub"):
            self.assertNotIn(gone, text)

    def test_an_nvidia_plan_has_no_intel_sentence(self):
        from strata_cli.launch.runner import Runner
        with Home(), mock.patch.object(Runner, "intel_chosen", return_value=False):
            r, _ = self.runner()
            self.assertNotIn("Intel card", r.download_plan()["text"])

    def prepare(self, code: int, lines: list[str]):
        from strata_cli.launch.runner import Runner
        with Home(), mock.patch.object(Runner, "_spawn", return_value=FakeChild(code)), \
                mock.patch.object(Runner, "_read_lines", return_value=iter(lines)), \
                mock.patch.object(Runner, "keep_setup_output", return_value="LOG"):
            r, events = self.runner()
            try:
                r.prepare_pack(Path("models"), Path("pack"), [])
                err = None
            except Exception as e:                       # noqa: BLE001 - the test reads the LaunchError's words
                err = e
        return [e.text for e in events], err

    def test_the_step_line_says_nothing_is_downloaded(self):
        texts, err = self.prepare(0, ["[ok] model prepared"])
        self.assertIsNone(err)
        self.assertIn("Preparing the model files for this graphics card (once, a few minutes; nothing is downloaded)…", texts)
        self.assertFalse(any("helper" in t for t in texts))

    def test_a_failed_pack_points_at_the_log_and_not_at_a_download(self):
        from strata_cli.launch.runner import LaunchError
        _, err = self.prepare(1, ["[X] the pack tool stopped"])
        self.assertIsInstance(err, LaunchError)
        self.assertEqual(err.text, "Strata could not prepare the model files for this graphics card: the pack tool stopped")
        self.assertEqual(err.next, "The last lines of the preparation are in LOG.")


if __name__ == "__main__":
    unittest.main()
