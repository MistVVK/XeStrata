"""Changing the model must not carry a Drafting choice to a model that has none (owner, 2026-11-02): the Coder, Swift and NVFP4 builds have only the MTP.
The Terminal's wizard goes back to Auto when the model changes to one of them, and the Q4_0-selector pack is made only where the installed engine reads it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli.launch import modelstate
from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import drafting


class TheChoiceStaysWithItsFamily(unittest.TestCase):
    def test_qwen_gyro_and_unsloth_keep_it(self):
        for family in ("qwen", "gyro", "unsloth"):
            for choice in (None, "auto", "mtp", "dflash2", "off"):
                self.assertEqual(modelstate.drafting_for_family(choice, family), choice, (family, choice))

    def test_every_other_family_goes_back_to_auto(self):
        for family in ("coder", "swift", "nvfp4"):
            for choice in ("auto", "mtp", "dflash2", "off"):
                self.assertIsNone(modelstate.drafting_for_family(choice, family), (family, choice))
        self.assertEqual(modelstate.drafting_for_family("mtp", None), "mtp", "no family yet: nothing to decide")

    def test_the_model_list_does_it_when_the_person_changes_model(self):
        from strata_cli.launch.screens import ModelScreen
        rows = [SimpleNamespace(family=f, model="m", verdict="fits", title=f) for f in ("qwen", "coder", "gyro")]
        for target, choice, want in (("coder", "off", None), ("gyro", "dflash2", "dflash2"), ("qwen", "mtp", "mtp")):
            s = S.LaunchSettings(family="qwen", model="m", drafting=choice)
            fake = SimpleNamespace(ctx=SimpleNamespace(settings=s, notes=[]), rows=rows, input_open=lambda: False,
                                   menu=lambda target=target: SimpleNamespace(current=SimpleNamespace(key=(target, "m"))))
            ModelScreen.commit(fake)
            self.assertEqual((s.family, s.drafting), (target, want), (target, choice))
        # and back to Qwen after the Coder: Auto (the choice that was made for the other model is gone), which the person can change again
        s = S.LaunchSettings(family="qwen", model="m", drafting="mtp")
        for target in ("coder", "qwen"):
            fake = SimpleNamespace(ctx=SimpleNamespace(settings=s, notes=[]), rows=rows, input_open=lambda: False,
                                   menu=lambda target=target: SimpleNamespace(current=SimpleNamespace(key=(target, "m"))))
            ModelScreen.commit(fake)
        self.assertIsNone(s.drafting)


class AQwenLaunchWritesItsDraftingChoice(unittest.TestCase):
    def test_qwen_still_writes_it(self):
        base = {"exe": "component:strata-engine", "args": ["--pack", "/d/packs/x", "--mtp", "/d/mtp/rt"], "port": 8080, "model_name": "qwen3.8-flash-next-iq3_s"}
        for family in ("qwen", "gyro", "unsloth"):
            cfg = derive_config(base, S.LaunchSettings(family=family, drafting="dflash2"), bool_flags=frozenset())
            self.assertEqual(cfg["drafting"], "dflash2", family)


class TheSelectorPackIsMadeWhereTheEngineReadsIt(unittest.TestCase):
    def test_the_engine_decides(self):
        with tempfile.TemporaryDirectory() as tmp:
            reads = Path(tmp) / "reads.exe"
            reads.write_bytes(b"x" * 100 + drafting._MARK_SELQ4 + b"y" * 100)
            sycl = Path(tmp) / "sycl.exe"
            sycl.write_bytes(b"x" * 300)
            for exe, want in ((reads, True), (sycl, False), (None, False), (Path(tmp) / "missing.exe", False)):
                with mock.patch.object(drafting, "installed_engine_exe", return_value=exe):
                    self.assertEqual(drafting.engine_has_selector_q4(), want, exe)

    def test_a_launch_converts_only_for_such_an_engine(self):
        import inspect
        from strata_cli.launch import runner
        source = inspect.getsource(runner)
        self.assertIn("drafting.needs_conversion(**want) and drafting.engine_has_selector_q4()", source)


if __name__ == "__main__":
    unittest.main()
