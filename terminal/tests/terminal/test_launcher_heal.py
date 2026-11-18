"""P1 (0.2.16): `strata` in PowerShell must be strata.cmd.

PowerShell runs a strata.ps1 before a strata.cmd in the same PATH folder, and the Windows default execution policy (Restricted) refuses
scripts. So the archive's PowerShell launcher is not named strata.ps1, and a Control that finds an old install's strata.ps1 renames it.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strata_cli import launcher_heal  # noqa: E402

OLD_PS1 = ("# Strata Terminal launcher for PowerShell. It finds a Python in a fixed order and hands over to strata.py.\n"
           "# It installs nothing and downloads nothing.\n$ErrorActionPreference = 'Stop'\n")


class Layout(unittest.TestCase):
    def test_no_strata_ps1_is_shipped(self):
        names = {p.name for p in (ROOT / "bin").iterdir()}
        self.assertNotIn("strata.ps1", names)
        self.assertIn("strata.cmd", names)
        self.assertIn(launcher_heal.RENAMED_TO, names)

    def test_the_packager_copies_the_renamed_launcher(self):
        text = (ROOT.parent / "tools" / "packaging" / "package_terminal.py").read_text(encoding="utf-8")
        self.assertIn('["strata.cmd", "strata-direct.ps1"]', text)
        self.assertNotIn('"strata.ps1"', text)

    def test_the_direct_launcher_says_why_it_is_not_called_strata_ps1(self):
        text = (ROOT / "bin" / launcher_heal.RENAMED_TO).read_text(encoding="utf-8")
        self.assertTrue(text.startswith(launcher_heal.HEADER))
        self.assertIn("-ExecutionPolicy Bypass -File bin\\strata-direct.ps1", text)


class Heal(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "bin"
        self.dir.mkdir()
        (self.dir / "strata.cmd").write_text("@echo off\r\n", encoding="utf-8")
        self.said = []

    def run_heal(self, path=None, **kw):
        return launcher_heal.heal(path if path is not None else str(self.dir), say=self.said.append, windows=True, **kw)

    def test_an_old_strata_ps1_is_renamed_and_said(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        self.assertEqual(self.run_heal(), [str(self.dir / "strata.ps1")])
        self.assertFalse((self.dir / "strata.ps1").exists())
        self.assertEqual((self.dir / "strata-direct.ps1").read_text(encoding="utf-8"), OLD_PS1)
        self.assertEqual(len(self.said), 1)
        self.assertIn("strata-direct.ps1", self.said[0])
        log = (self.dir / launcher_heal.LOG_NAME).read_text(encoding="utf-8")      # a line for support, also when nobody saw the one above
        self.assertEqual(len(log.splitlines()), 1)
        self.assertIn("strata.ps1 was renamed to strata-direct.ps1", log)
        self.assertEqual(self.run_heal(), [])                      # once: nothing left to do, nothing said again
        self.assertEqual(len(self.said), 1)

    def test_a_byte_order_mark_does_not_hide_the_header(self):
        (self.dir / "strata.ps1").write_bytes(b"\xef\xbb\xbf" + OLD_PS1.encode("utf-8"))
        self.assertEqual(len(self.run_heal()), 1)

    def test_an_older_direct_launcher_is_replaced(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        (self.dir / "strata-direct.ps1").write_text("old", encoding="utf-8")
        self.run_heal()
        self.assertEqual((self.dir / "strata-direct.ps1").read_text(encoding="utf-8"), OLD_PS1)

    def test_someone_elses_strata_ps1_is_left_alone(self):
        (self.dir / "strata.ps1").write_text("Write-Host 'my own wrapper'\n", encoding="utf-8")
        self.assertEqual(self.run_heal(), [])
        self.assertTrue((self.dir / "strata.ps1").exists())

    def test_a_ps1_without_our_cmd_beside_it_is_left_alone(self):
        (self.dir / "strata.cmd").unlink()
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        self.assertEqual(self.run_heal(), [])
        self.assertTrue((self.dir / "strata.ps1").exists())

    def test_the_install_folder_is_found_even_when_the_path_is_not_set(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        self.assertEqual(len(self.run_heal("", extra=[self.dir])), 1)

    def test_a_folder_listed_twice_is_healed_once(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        self.assertEqual(len(self.run_heal(os.pathsep.join([str(self.dir), str(self.dir)]))), 1)

    def test_other_systems_do_nothing(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        self.assertEqual(launcher_heal.heal(str(self.dir), say=self.said.append, windows=False), [])
        self.assertTrue((self.dir / "strata.ps1").exists())

    def test_a_failure_to_rename_is_swallowed(self):
        (self.dir / "strata.ps1").write_text(OLD_PS1, encoding="utf-8")
        real = os.replace

        def refuse(a, b):
            raise PermissionError("in use")
        os.replace = refuse
        try:
            self.assertEqual(self.run_heal(), [])
        finally:
            os.replace = real
        self.assertEqual(self.said, [])


if __name__ == "__main__":
    unittest.main()
