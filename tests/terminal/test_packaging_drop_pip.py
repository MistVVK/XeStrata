"""#5303: the Terminal bundle ships without pip. pip (and its pip/_vendor copies of certifi, urllib3, msgpack, setuptools that
scanners flag) and ensurepip's spare pip wheel are the build's tools: package_terminal.drop_pip removes them once the lock is
installed. These tests check the removal on a fake bundle of each layout, and that nothing in the app's install mode would run pip."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))
import package_terminal as pt  # noqa: E402
import setup  # noqa: E402
from strata_control import acquisition  # noqa: E402


def touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class DropPip(unittest.TestCase):
    def tree(self, windows: bool) -> tuple[Path, Path, Path]:
        root = Path(self.tmp.name) / "strata-terminal"
        lib = root / "python" / ("Lib" if windows else "lib/python3.12")
        site = lib / "site-packages"
        bindir = root / "python" / ("Scripts" if windows else "bin")
        touch(site / "pip" / "__init__.py")
        touch(site / "pip" / "_vendor" / "certifi" / "__init__.py")
        touch(site / "pip" / "_vendor" / "vendor.txt", "certifi==2026.6.17\n")
        touch(site / "pip-26.2.1.dist-info" / "METADATA", "Name: pip\nVersion: 26.2.1\n")
        touch(site / "certifi" / "__init__.py")                          # the runtime's own, locked copy: stays
        touch(site / "certifi-2026.10.19.dist-info" / "METADATA")
        touch(site / "pipeline_helper.py")                               # a name that only starts with "pip": stays
        touch(lib / "ensurepip" / "__init__.py")                         # the module stays
        touch(lib / "ensurepip" / "_bundled" / "pip-25.0.1-py3-none-any.whl")
        for name in (["pip.exe", "pip3.exe", "pip3.12.exe"] if windows else ["pip", "pip3", "pip3.12"]):
            touch(bindir / name)
        touch(bindir / ("python.exe" if windows else "python3"))
        return root, site, lib

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def check(self, windows: bool, target: str):
        root, site, lib = self.tree(windows)
        out = pt.drop_pip(root, target)
        self.assertEqual(out["version"], "26.2.1")
        self.assertFalse((site / "pip").exists())
        self.assertFalse(list(site.glob("pip-*.dist-info")))
        self.assertFalse(list((lib / "ensurepip" / "_bundled").glob("*.whl")))
        self.assertTrue((lib / "ensurepip" / "__init__.py").exists())
        self.assertTrue((site / "certifi" / "__init__.py").exists())
        self.assertTrue((site / "certifi-2026.10.19.dist-info" / "METADATA").exists())
        self.assertTrue((site / "pipeline_helper.py").exists())
        bindir = root / "python" / ("Scripts" if windows else "bin")
        self.assertEqual(sorted(p.name for p in bindir.iterdir()), ["python.exe" if windows else "python3"])
        self.assertEqual(len(out["paths"]), 6, out["paths"])
        self.assertTrue(all(not p.startswith("/") and ".." not in p for p in out["paths"]))

    def test_windows_layout(self):
        self.check(True, "windows-x86_64")

    def test_linux_and_mac_layout(self):
        for target in ("linux-x86_64", "linux-aarch64", "darwin-arm64"):
            with self.subTest(target=target):
                self.tmp.cleanup()
                self.tmp = tempfile.TemporaryDirectory()
                self.check(False, target)

    def test_a_bundle_without_pip_is_left_as_it_is(self):
        root = Path(self.tmp.name) / "empty"
        touch(root / "python" / "Lib" / "site-packages" / "certifi" / "__init__.py")
        self.assertEqual(pt.drop_pip(root, "windows-x86_64"), {"version": None, "paths": []})

    def test_the_build_removes_pip_after_the_locked_install_and_records_it(self):
        src = (ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8")
        install = src.index("installed = install_locked(")
        drop = src.index("pip_removed = drop_pip(root, target)")
        probe = src.index("probe = run_probe(py, root)")
        self.assertLess(install, drop)
        self.assertLess(drop, probe, "the import probe runs on the bundle without pip")
        self.assertIn('"pip_removed": pip_removed,', src)


class AppInstallModeNeverRunsPip(unittest.TestCase):
    """In the app's install mode (setup.CONTROL, the bundle) no path may start pip: the bundle has none any more."""

    def test_run_refuses_python_dash_m(self):
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "say"), \
                mock.patch("subprocess.run") as sub, mock.patch("subprocess.Popen") as popen:
            with self.assertRaises(acquisition.AcquisitionRefused):
                setup.run([sys.executable, "-m", "pip", "install", "x"])
        sub.assert_not_called()
        popen.assert_not_called()

    def test_pip_install_and_the_cuda_wheels_do_not_reach_pip(self):
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "control_missing_packages", return_value=[]), \
                mock.patch.object(setup, "ok"), mock.patch.object(setup, "run") as run:
            setup.pip_install(["definitely-not-installed==1.0"], "a test package")
            setup.pip_cuda_libs(13)
            setup.pip_cuda_libs(12)
        run.assert_not_called()

    def test_the_fp4_conversion_stops_before_pip(self):
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "NVFP4_LOCAL_MODULES", ("definitely_not_a_module_5303",)), \
                mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as fail, mock.patch.object(setup, "run") as run:
            with self.assertRaises(SystemExit):
                setup.ensure_nvfp4_local_packages()
        run.assert_not_called()
        self.assertIn("app does not include", fail.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
