"""#6015: on a PC whose integrated Radeon is HIP's device 0 (Ryzen 9800X3D + RX 9070), the first-run engine test (`<engine> --version`, no model
yet) ran with every card visible and failed the same way on every "Continue". The test now lets the engine see only the cards its package has
code for, a failure names the cards it ran with, and the same failure twice says that a retry will not help. Model-free: a fake engine package
whose `--version` fails when a card without code is visible, and a fake `strata-device --list-devices` with the mixed-card order."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.server import parse_hip_devices, pick_hip_devices
from strata_control import updater

MIXED = ("device 0: AMD Radeon(TM) Graphics\n  arch gfx1036, 0.5 GiB, wave32\n  cannot run: this engine is compiled for (gfx1201)\n"
         "device 1: AMD Radeon RX 9070\n  arch gfx1201, 15.9 GiB, wave32\n")
# the engine fails unless HIP_VISIBLE_DEVICES leaves device 0 (the integrated chip) out; it prints what it saw
ENGINE_NEEDS_HIDING = ('case "$HIP_VISIBLE_DEVICES" in\n  1) echo "strata 0.1.61 (saw $HIP_VISIBLE_DEVICES)"; exit 0;;\n'
                       '  *) echo "hipErrorNoBinaryForGpu: device 0 gfx1036" >&2; exit 1;;\nesac\n')
ENGINE_ALWAYS_FAILS = 'echo "rocblas: Cannot read TensileLibrary for gfx1201" >&2\nexit 3\n'


def script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@unittest.skipIf(os.name == "nt", "shell-script stand-ins for the engine; the Windows mixed-AMD case needs a real PC")
class HipTestCards(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="strata-hip-cards-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.engine = self.root / "engine"
        self.engine.mkdir()
        self.ctx = SimpleNamespace(components=self.root / "components")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    def package(self, engine_body: str, listing: str | None = MIXED):
        exe = script(self.engine / "strata", engine_body)
        if listing is not None:
            script(self.engine / "strata-device", f"cat <<'EOF'\n{listing}EOF\n")
        return SimpleNamespace(exe=exe, root=self.engine, version="0.1.61", backend="hip")

    def test_the_integrated_chip_without_code_is_hidden_and_the_test_passes(self):
        res = updater.binary_smoke(self.package(ENGINE_NEEDS_HIDING), ctx=self.ctx)
        self.assertEqual(res["mode"], "binary-only")
        self.assertIn("saw 1", res["output"])

    def test_a_failure_names_the_cards_it_ran_with(self):
        with self.assertRaises(updater.TestStartFailed) as caught:
            updater.binary_smoke(self.package(ENGINE_ALWAYS_FAILS), ctx=self.ctx)
        self.assertIn("GPU 1 (AMD Radeon RX 9070, gfx1201)", caught.exception.reason)
        self.assertNotIn("GPU 0", caught.exception.reason)
        self.assertIn("--force", str(caught.exception))                   # the first failure: try again

    def test_the_same_failure_twice_does_not_ask_for_the_same_retry(self):
        component = self.package(ENGINE_ALWAYS_FAILS)
        with self.assertRaises(updater.TestStartFailed) as first:
            updater.binary_smoke(component, ctx=self.ctx)
        with self.assertRaises(updater.TestStartFailed) as second:
            updater.binary_smoke(component, ctx=self.ctx)
        self.assertNotEqual(str(first.exception), str(second.exception))
        self.assertNotIn("--force", str(second.exception))
        self.assertIn("This is the same problem as last time, so trying again won't fix it. Please report it on GitHub and attach this log: ",
                      str(second.exception))

    def test_without_the_probe_nothing_changes(self):
        component = self.package('echo "strata 0.1.61 (saw [$HIP_VISIBLE_DEVICES])"; exit 0\n', listing=None)
        self.assertIn("saw []", updater.binary_smoke(component, ctx=self.ctx)["output"])

    def test_cards_the_person_named_are_kept(self):
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        component = self.package('echo "strata 0.1.61 (saw [$HIP_VISIBLE_DEVICES])"; exit 0\n')
        self.assertIn("saw [0]", updater.binary_smoke(component, ctx=self.ctx)["output"])

    def test_an_nvidia_package_is_not_probed(self):
        component = self.package('echo "strata 0.1.61 (saw [$HIP_VISIBLE_DEVICES])"; exit 0\n')
        component.backend = "cuda"
        self.assertIn("saw []", updater.binary_smoke(component, ctx=self.ctx)["output"])


class StartLeavesOutCardsWithoutCode(unittest.TestCase):
    def test_device_0_runs_and_a_card_without_code_behind_it_is_not_shown(self):
        devices = parse_hip_devices("device 0: AMD Radeon RX 9070\n  arch gfx1201, 15.9 GiB, wave32\n"
                                    "device 1: AMD Radeon(TM) Graphics\n  arch gfx1036, 0.5 GiB, wave32\n  cannot run: compiled for (gfx1201)\n")
        self.assertEqual(pick_hip_devices(devices), ([0], None))

    def test_the_mixed_order_runs_on_the_card(self):
        self.assertEqual(pick_hip_devices(parse_hip_devices(MIXED)), ([1], None))

    def test_cards_that_all_run_keep_the_runtime_order(self):
        two = parse_hip_devices("device 0: R9700\n  arch gfx1201, 31.9 GiB, wave32\ndevice 1: R9700\n  arch gfx1201, 31.9 GiB, wave32\n")
        self.assertEqual(pick_hip_devices(two), (None, None))


if __name__ == "__main__":
    unittest.main()
