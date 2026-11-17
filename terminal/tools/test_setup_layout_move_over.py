"""MAP-0215 Q7 (owner 11-16): a git clone made before 0.2.16 kept setup's state at its root (run configs, logs, run scripts, the model
and pack folders, the MCP state, and the engine setup put in engine/). After its pull to the 0.2.16 layout setup runs in terminal/,
and its first run carries that state over, once, with the paths inside the run configs and run scripts changed to the new places.
The engine's source entries in engine/ stay; nothing is replaced; nothing happens in the old layout or in an installed Terminal.

    python -m unittest tools.test_setup_layout_move_over
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class MoveOver(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        c = self.clone = self.tmp / "Strata"
        (c / "engine" / "src").mkdir(parents=True)
        (c / "engine" / "CMakeLists.txt").write_text("project(strata VERSION 0.1.61)\n", encoding="utf-8")
        (c / "engine" / "build-sycl").mkdir()                       # a build folder: stays
        (c / "engine" / "strata.exe").write_bytes(b"MZ")            # the engine the old setup downloaded
        (c / "engine" / "BUILD.json").write_text("{}", encoding="utf-8")
        (c / "engine" / "lib").mkdir()
        (c / "engine" / "lib" / "cudart.dll").write_bytes(b"dll")
        (c / "models").mkdir()
        (c / "models" / "note.txt").write_text("x", encoding="utf-8")
        (c / ".strata-mcp").mkdir()
        self.exe = c / "engine" / "strata.exe"
        (c / "strata-q2_0.json").write_text(json.dumps({
            "model_name": "Qwen Q2_0", "exe": str(self.exe), "args": ["--model", str(c / "models" / "q2.gguf"), "--mtp", str(c / "mtp" / "rt")],
            "env": {"PATH": str(c / "engine" / "lib")}, "elsewhere": str(c / "engine-sycl-other")}), encoding="utf-8")
        (c / "run-q2_0.bat").write_text(f'"{c / ".venv" / "Scripts" / "python.exe"}" "{c / "strata.py"}" launch\r\n', encoding="utf-8")
        (c / "strata-setup.log").write_text("log", encoding="utf-8")
        (c / "terminal").mkdir()
        (c / "terminal" / "setup.py").write_text("# new\n", encoding="utf-8")
        self.root = c / "terminal"

    def run_it(self, root=None):
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", root or self.root), contextlib.redirect_stdout(out):
            moved = setup.move_over_old_layout()
        return moved, out.getvalue()

    def test_the_old_state_moves_into_terminal_with_its_paths(self):
        moved, out = self.run_it()
        t, c = self.root, self.clone
        for rel in ("strata-q2_0.json", "run-q2_0.bat", "strata-setup.log", "models/note.txt", ".strata-mcp",
                    "engine/strata.exe", "engine/BUILD.json", "engine/lib/cudart.dll"):
            self.assertTrue((t / rel).exists(), rel)
        for rel in ("strata-q2_0.json", "run-q2_0.bat", "models", "engine/strata.exe", "engine/lib"):
            self.assertFalse((c / rel).exists(), rel)
        for rel in ("engine/CMakeLists.txt", "engine/src", "engine/build-sycl"):     # the engine's sources and a build folder stay
            self.assertTrue((c / rel).exists(), rel)
        cfg = json.loads((t / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["exe"], str(t / "engine" / "strata.exe"))
        self.assertEqual(cfg["args"][1], str(t / "models" / "q2.gguf"))
        self.assertEqual(cfg["env"]["PATH"], str(t / "engine" / "lib"))
        self.assertEqual(cfg["elsewhere"], str(c / "engine-sycl-other"), "only whole path parts are changed")
        bat = (t / "run-q2_0.bat").read_text(encoding="utf-8")
        self.assertIn(str(t / ".venv" / "Scripts" / "python.exe"), bat)
        self.assertIn(str(t / "strata.py"), bat)
        self.assertIn(str(c), out)
        self.assertEqual(len(moved), 8)

    def test_it_happens_once(self):
        self.run_it()
        (self.clone / "strata-late.json").write_text("{}", encoding="utf-8")
        moved, out = self.run_it()
        self.assertEqual((moved, out), ([], ""))
        self.assertTrue((self.clone / "strata-late.json").exists())

    def test_nothing_is_replaced(self):
        (self.root / "models").mkdir()
        (self.root / "models" / "mine.txt").write_text("keep", encoding="utf-8")
        self.run_it()
        self.assertTrue((self.clone / "models" / "note.txt").exists(), "an existing terminal/models is not replaced")
        self.assertEqual((self.root / "models" / "mine.txt").read_text(encoding="utf-8"), "keep")

    def test_models_and_packs_move_by_rename_only_and_stay_when_it_fails(self):
        """Coordinator 11-16: models/ and packs/ (hundreds of GB) are never copied. A rename that fails (another disk, files in use)
        leaves them where they are, with a warning, and the moved config still names them there."""
        (self.clone / "packs").mkdir()
        (self.clone / "packs" / "p.bin").write_bytes(b"pack")
        real = setup.os.rename

        def rename(src, dst):
            if Path(src).name in ("models", "packs"):
                raise OSError(18, "Invalid cross-device link")
            return real(src, dst)
        with mock.patch.object(setup.os, "rename", side_effect=rename), \
                mock.patch.object(setup.shutil, "move", side_effect=AssertionError("moved by copy")), \
                mock.patch.object(setup.shutil, "copytree", side_effect=AssertionError("copied")), \
                mock.patch.object(setup.shutil, "copy2", side_effect=AssertionError("copied")):
            moved, out = self.run_it()
        for name in ("models", "packs"):
            self.assertTrue((self.clone / name).is_dir(), name)
            self.assertFalse((self.root / name).exists(), name)
            self.assertIn(f"{self.clone / name} could not be moved", out)
        self.assertTrue((self.root / "strata-q2_0.json").exists(), "the rest still moves")
        cfg = json.loads((self.root / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["args"][1], str(self.clone / "models" / "q2.gguf"), "the config keeps naming the models where they are")
        self.assertEqual(cfg["exe"], str(self.root / "engine" / "strata.exe"))

    def test_no_move_in_the_old_layout_or_an_installed_terminal(self):
        self.assertEqual(self.run_it(self.clone), ([], ""))
        installed = self.tmp / "share" / "strata" / "terminal"
        installed.mkdir(parents=True)
        self.assertEqual(self.run_it(installed), ([], ""))
        self.assertTrue((self.clone / "strata-q2_0.json").exists())


class MainMovesFirst(unittest.TestCase):
    def test_main_moves_the_old_state_before_it_reads_any_config(self):
        order = []
        with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                mock.patch.object(setup, "move_over_old_layout", side_effect=lambda: order.append("move")), \
                mock.patch.object(setup, "update_pull", side_effect=lambda argv: order.append("pull") or 0):
            self.assertEqual(setup.main(), 0)
        self.assertEqual(order, ["move", "pull"])


if __name__ == "__main__":
    unittest.main()
