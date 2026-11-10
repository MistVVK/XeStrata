"""A run config that names files in a Control folder that is gone (the AppImage's /tmp/.mount_<random>) is pointed at this Control (QAGUI-LIN 0213).
Plain unittest with tempfile (pytest's tmp_path base folder cannot be read on the owner PC: WinError 5, 11-10)."""
import json
import tempfile
import unittest
from pathlib import Path

from strata_control import control_paths


def _control(root: Path, name: str) -> Path:
    here = root / name / "usr" / "lib" / "Strata" / "control"
    (here / "data").mkdir(parents=True)
    (here / "data" / "expert-profile.bin").write_bytes(b"x")
    return here


class ControlPaths(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_a_gone_mount_path_is_pointed_at_this_control(self):
        here = _control(self.root, ".mount_StrataNEW")
        gone = "/tmp/.mount_StratappjGco/usr/lib/Strata/control"
        cfg = {"cwd": gone, "args": ["--native", "/data/m/a.gguf", "--expert-profile", gone + "/data/expert-profile.bin", "--max-context", "32768"],
               "model_name": "Qwen"}
        out = control_paths.repoint(cfg, here)
        self.assertEqual(out["cwd"], str(here))
        self.assertEqual(out["args"][3], str(here / "data" / "expert-profile.bin"))
        self.assertEqual(out["args"][:3], cfg["args"][:3])
        self.assertEqual(out["args"][4:], cfg["args"][4:])
        self.assertEqual(cfg["cwd"], gone, "the input is not changed")

    def test_paths_that_exist_or_have_no_counterpart_stay(self):
        here = _control(self.root, "new")
        other = _control(self.root, "old")
        kept = str(other / "data" / "expert-profile.bin")
        self.assertEqual(control_paths.repoint_value(kept, here), kept, "an existing file is left alone")
        missing = "/tmp/.mount_X/usr/lib/Strata/control/data/not-in-this-control.bin"
        self.assertEqual(control_paths.repoint_value(missing, here), missing, "no counterpart: unchanged (the start reports it)")
        self.assertEqual(control_paths.repoint_value("/home/u/controller/x", here), "/home/u/controller/x")
        self.assertEqual(control_paths.repoint_value(32768, here), 32768)

    def test_the_update_test_start_uses_the_repointed_config(self):
        src = (Path(control_paths.__file__).parent / "updater.py").read_text(encoding="utf-8")
        at = src.index('cfg = json.loads(source.read_text(encoding="utf-8-sig"))')
        self.assertIn("control_paths.repoint(cfg)", src[at:at + 400])
        json.dumps(control_paths.repoint({"cwd": "x"}))


if __name__ == "__main__":
    unittest.main()
