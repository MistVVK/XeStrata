"""#5402: a PC with an AMD and an NVIDIA card. The person chose the AMD card in setup, and setup fetched the CUDA engine, because the `strata update` it
runs for the ready-made engine picks the first make it finds (NVIDIA first) and knew nothing of the choice. setup.engine_from_feed now hands that
update the engine family of the card setup chose (STRATA_ENGINE_BACKEND), unless the person set the variable themselves.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402


class FeedUpdateGetsTheChosenMake(unittest.TestCase):
    def feed_env(self, backend, environ=None):
        seen = {}

        def fake_run(cmd, **kw):
            seen["env"] = kw["env"]
            return mock.Mock(returncode=1, stdout="", stderr="stopped here")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(setup, "home", return_value=Path(tmp)), \
                mock.patch.object(setup.acquisition, "require"), mock.patch.object(setup.subprocess, "run", fake_run), \
                mock.patch.object(setup, "say"), mock.patch.object(setup, "warn"), \
                mock.patch.dict(os.environ, environ or {}, clear=False):
            if environ is None:
                os.environ.pop(ec.ENV_BACKEND, None)
            self.assertIsNone(setup.engine_from_feed(Path(tmp) / "engine", False, backend))
        return seen["env"]

    def test_the_amd_card_asks_for_the_hip_family(self):
        self.assertEqual(self.feed_env("hip").get(ec.ENV_BACKEND), "hip")

    def test_the_nvidia_card_asks_for_the_cuda_family(self):
        self.assertEqual(self.feed_env("cuda").get(ec.ENV_BACKEND), "cuda")

    def test_no_choice_leaves_the_update_to_find_the_make_as_before(self):
        self.assertNotIn(ec.ENV_BACKEND, self.feed_env(None))

    def test_a_family_set_by_hand_wins(self):
        self.assertEqual(self.feed_env("hip", {ec.ENV_BACKEND: "sycl"}).get(ec.ENV_BACKEND), "sycl")

    def test_the_callers_name_their_family(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('engine_from_feed(eng, updating, "hip")', src)
        self.assertEqual(src.count('engine_from_feed(eng, updating, "cuda")'), 2)
        self.assertNotIn("engine_from_feed(eng, updating)", src)


class TheUpdateHonoursIt(unittest.TestCase):
    def test_a_mixed_amd_and_nvidia_pc_takes_the_named_family(self):
        mixed = [{"vendor": "10de", "name": "NVIDIA GeForce RTX 5090"}, {"vendor": "1002", "name": "AMD Radeon AI PRO R9700"}]
        with mock.patch.object(ec, "windows_display_adapters", return_value=mixed), \
                mock.patch.object(ec, "host_platform", return_value="windows-x64"), \
                mock.patch.object(ec, "saved_backend", return_value=None):
            self.assertEqual(ec.detect_backend({})[0], "cuda")                          # without the variable: the first make (the old answer)
            self.assertEqual(ec.detect_backend({ec.ENV_BACKEND: "hip"}), ("hip", "environment"))


if __name__ == "__main__":
    unittest.main()
