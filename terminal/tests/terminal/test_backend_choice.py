"""0.2.10: a PC with cards of more than one make chooses the engine with `strata launch --backend` (kept in the settings file), is told that the
other cards exist, and a person who chose is never refused."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec, gpu_fit

NV = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 5060", "vram_gb": 8.0, "usable": True, "index": 0}
ARC = [{"vendor": "intel", "name": "Intel Arc Pro B65", "vram_gb": 32.0, "usable": True, "index": i} for i in (0, 1)]


class SavedChoice(unittest.TestCase):
    def env(self, d):
        return {"STRATA_SETTINGS_FILE": str(Path(d) / "settings.json")}

    def test_names_people_type(self):
        self.assertEqual([ec.clean_backend_name(x) for x in ("Intel", "nvidia", "AMD", "sycl", "arc", "toaster", "")],
                         ["sycl", "cuda", "hip", "sycl", "sycl", "", ""])

    def test_a_saved_choice_decides_and_keeps_the_other_settings(self):
        with tempfile.TemporaryDirectory() as d:
            env = self.env(d)
            Path(env["STRATA_SETTINGS_FILE"]).write_text(json.dumps({"data_dir": "/x"}))
            with mock.patch.object(ec, "gpu_makes_present", lambda: {"cuda", "sycl"}):
                ec.save_backend("intel", env)
                self.assertEqual(json.loads(Path(env["STRATA_SETTINGS_FILE"]).read_text()), {"data_dir": "/x", "engine_backend": "sycl"})
                self.assertEqual(ec.detect_backend(env), ("sycl", "saved choice"))
                self.assertEqual(ec.detect_backend({**env, ec.ENV_BACKEND: "cuda"}), ("cuda", "environment"), "the environment still wins")
                ec.save_backend(None, env)
                self.assertEqual(json.loads(Path(env["STRATA_SETTINGS_FILE"]).read_text()), {"data_dir": "/x"})

    def test_a_choice_for_a_make_this_pc_does_not_have_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            env = self.env(d)
            with mock.patch.object(ec, "gpu_makes_present", lambda: {"cuda"}):
                ec.save_backend("sycl", env)
                self.assertIsNone(ec.saved_backend(env))

    def test_the_update_make_guard_lets_a_saved_choice_through(self):
        from strata_control import updater
        ctx = mock.Mock()
        ctx.hardware = {"backend": "sycl", "backend_source": "saved choice"}
        self.assertIsNone(updater.make_guard(ctx))


class Notes(unittest.TestCase):
    def test_intel_cards_beside_nvidia_are_always_mentioned(self):
        note = gpu_fit.other_vendor_note([NV, *ARC], "nvidia")
        self.assertIn("2 Intel Arc cards (64 GB together)", note)
        self.assertIn("--backend intel", note)

    def test_one_intel_card_and_the_reverse_note(self):
        self.assertIn("an Intel Arc Pro B65 (32 GB)", gpu_fit.other_vendor_note([NV, ARC[0]], "nvidia"))
        back = gpu_fit.other_vendor_note([NV, *ARC], "intel")
        self.assertIn("NVIDIA", back)
        self.assertIn("--backend auto", back)

    def test_no_note_without_other_makes(self):
        self.assertIsNone(gpu_fit.other_vendor_note(ARC, "intel"))


class BrokenNvidia(unittest.TestCase):
    def test_says_what_is_wrong_and_the_next_step_only_when_it_is(self):
        from strata_cli.launch import cmd
        comp = mock.Mock(gpus=list(ARC))
        with mock.patch.object(ec, "detect_backend", lambda *a, **k: ("cuda", "sysfs pci vendor")),                 mock.patch.object(ec, "gpu_makes_present", lambda: {"cuda", "sycl"}):
            note = cmd._broken_nvidia_note(comp)
            self.assertIn("driver does not answer", note)
            self.assertIn("--backend intel", note)
            self.assertIsNone(cmd._broken_nvidia_note(mock.Mock(gpus=[NV, *ARC])), "a working NVIDIA card: nothing to say")
        with mock.patch.object(ec, "detect_backend", lambda *a, **k: ("sycl", "saved choice")),                 mock.patch.object(ec, "gpu_makes_present", lambda: {"cuda", "sycl"}):
            self.assertIsNone(cmd._broken_nvidia_note(comp), "the person already chose")


class SavedChoiceLine(unittest.TestCase):
    def test_a_launch_after_a_saved_choice_says_so(self):
        from strata_cli.launch import cmd, settings as S
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home(), mock.patch.object(ec, "detect_backend", lambda *a, **k: ("sycl", "saved choice")):
            _s, notes = cmd.apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS"), catalog(HW_12))
            hit = [n for n in notes if "your saved choice" in n.text]
            self.assertTrue(hit, [n.text for n in notes])
            self.assertIn("Intel cards", hit[0].text)
            self.assertIn("--backend auto", hit[0].text)


if __name__ == "__main__":
    unittest.main()
