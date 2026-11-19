"""#6203: `strata serve` ignored an option edited by hand in "args" (a value that differs from "args_written", with nothing in "user_args"), although
docs/TERMINAL_LAUNCH.md says such edits count as the person's own: drafting.apply read only "user_args" and reset --spec and --draft-conf to the backend policy.
engine_args now hands drafting.apply the person's options as user_args.capture finds them."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

GYRO = "qwen3.8-flash-next-gyro-gyro-s"
_HARDWARE = mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"]))


def setUpModule():
    _HARDWARE.start()


def tearDownModule():
    _HARDWARE.stop()


class HandEditsInArgs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        self.mtp = self.data / "mtp" / "rt"
        self.mtp.mkdir(parents=True)
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.mtp / "draft_vocab.bin").write_bytes(b"v")
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        self.args = ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", str(self.mtp), "--spec", "4", "--spec-min-p", "0.5"]
        for patcher in (mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data),
                        mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve_args(self, cfg):
        from serve.server import engine_args
        return engine_args(cfg, announce=False)

    def edited(self, **extra):
        args = [*self.args, "--draft-conf", "0.30"]
        args[args.index("--spec") + 1] = "8"
        return {"model_name": GYRO, "args": args, "args_written": list(self.args), "drafting": "dflash2", **extra}

    def value(self, args, flag):
        return args[args.index(flag) + 1]

    def test_a_hand_edit_of_spec_and_draft_conf_in_args_is_kept(self):
        args = self.serve_args(self.edited())
        self.assertEqual(self.value(args, "--spec"), "8")
        self.assertEqual(self.value(args, "--draft-conf"), "0.30")

    def test_the_same_edit_in_user_args_gives_the_same_arguments(self):
        by_hand = self.serve_args(self.edited())
        stored = self.serve_args({**self.edited(), "args": list(self.args), "user_args": ["--spec", "8", "--draft-conf", "0.30"]})
        for flag in ("--spec", "--draft-conf"):
            self.assertEqual(self.value(by_hand, flag), self.value(stored, flag))

    def test_a_config_nobody_edited_still_gets_the_policy_values(self):
        args = self.serve_args({"model_name": GYRO, "args": list(self.args), "args_written": list(self.args), "drafting": "dflash2"})
        self.assertNotEqual(self.value(args, "--draft-conf"), "0.30")
        self.assertNotEqual(self.value(args, "--spec"), "8")


if __name__ == "__main__":
    unittest.main()
