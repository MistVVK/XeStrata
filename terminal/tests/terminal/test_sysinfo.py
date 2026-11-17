"""#4975 (PR #5185 is the reference): the OS and architecture are known without WMI on Windows.

On Windows with Python 3.12 `platform.uname()` (so `system()`, `machine()`, `release()`, `version()`, `processor()` and `node()`) asks WMI
(`platform._wmi_query`) with no timeout. A stuck `winmgmt` froze every Control command that resolves the engine component. A stuck WMI
cannot be produced from a test, so these tests make every `platform` call that could reach it fail loudly, and check that the Windows
answers still come out of the interpreter and the environment.

    python -m pytest tests/terminal/test_sysinfo.py
"""
import os
import re
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from strata_control import engine_component, sysinfo, targets

WinVer = namedtuple("WinVer", "major minor build")


def _stuck(name):
    def hang(*a, **k):
        raise AssertionError(f"platform.{name} reached WMI")
    return hang


def _windows(env, build=19045):
    """sys.platform is win32, the environment is `env`, and every platform call that asks WMI fails."""
    stack = [mock.patch.object(sys, "platform", "win32"),
             mock.patch.object(sys, "getwindowsversion", lambda: WinVer(10, 0, build), create=True),
             mock.patch.dict(os.environ, env, clear=False)]
    for name in ("uname", "system", "machine", "release", "version", "processor", "node", "platform", "win32_ver", "_wmi_query"):
        stack.append(mock.patch.object(sysinfo.platform, name, _stuck(name), create=True))
    return stack


class Stuck(unittest.TestCase):
    def run_in(self, env, fn, build=19045):
        stack = _windows(env, build)
        for item in stack:
            item.start()
        try:
            return fn()
        finally:
            for item in reversed(stack):
                item.stop()

    def test_windows_x64_host_platform_while_wmi_is_stuck(self):
        env = {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}
        self.assertEqual(self.run_in(env, targets.host_platform), "windows-x86_64")

    def test_windows_arm64_native(self):
        env = {"PROCESSOR_ARCHITECTURE": "ARM64", "PROCESSOR_ARCHITEW6432": ""}
        self.assertEqual(self.run_in(env, targets.host_platform), "windows-arm64")

    def test_x64_python_emulated_on_arm64_sees_amd64_and_gets_the_x64_engine(self):
        env = {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}
        self.assertEqual(self.run_in(env, sysinfo.machine), "AMD64")

    def test_a_32_bit_process_on_64_bit_windows_reads_the_real_architecture(self):
        for real, expect in (("AMD64", "windows-x86_64"), ("ARM64", "windows-arm64")):
            env = {"PROCESSOR_ARCHITECTURE": "x86", "PROCESSOR_ARCHITEW6432": real}
            self.assertEqual(self.run_in(env, targets.host_platform), expect)

    def test_without_the_environment_the_interpreter_build_answers(self):
        stack = _windows({})
        for item in stack:
            item.start()
        try:
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("sysconfig.get_platform", return_value="win-arm64"):
                self.assertEqual(sysinfo.machine(), "ARM64")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("sysconfig.get_platform", return_value="win-amd64"):
                self.assertEqual(sysinfo.machine(), "AMD64")
        finally:
            for item in reversed(stack):
                item.stop()

    def test_release_version_processor_and_node_need_no_wmi(self):
        env = {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_IDENTIFIER": "Intel64 Family 6 Model 183"}

        def facts():
            return (sysinfo.system(), sysinfo.release(), sysinfo.version(), sysinfo.processor(), bool(sysinfo.node()),
                    sysinfo.platform_string())
        self.assertEqual(self.run_in(env, facts, 22631),
                         ("Windows", "11", "10.0.22631", "Intel64 Family 6 Model 183", True, "Windows-11-10.0.22631"))
        self.assertEqual(self.run_in(env, facts, 19045)[1], "10")

    def test_the_engine_lookup_of_a_run_config_does_not_reach_wmi(self):
        # the very path of the report: engine_component.resolve() -> host_platform()
        env = {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}

        def look():
            with tempfile.TemporaryDirectory() as folder:
                try:
                    engine_component.resolve(Path(folder), backend="cpu")
                except engine_component.EngineMissing:
                    return "missing"
        self.assertEqual(self.run_in(env, look), "missing")


class OtherSystemsKeepTheirAnswers(unittest.TestCase):
    def test_macos_and_linux_use_the_platform_module(self):
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(sysinfo.platform, "system", return_value="Darwin"), \
                mock.patch.object(sysinfo.platform, "machine", return_value="arm64"):
            self.assertEqual(targets.host_platform(), "macos-arm64")
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(sysinfo.platform, "system", return_value="Linux"), \
                mock.patch.object(sysinfo.platform, "machine", return_value="aarch64"):
            self.assertEqual(targets.host_platform(), "linux-aarch64")

    def test_the_answer_is_a_key_of_the_backend_matrix(self):
        self.assertIn(targets.host_platform(), {t for ts in targets.BACKEND_MATRIX.values() for t in ts})

    def test_no_control_module_calls_the_uname_based_platform_functions_directly(self):
        # the callers that can run on Windows were moved to sysinfo; a new direct call would bring the hang back
        root = Path(__file__).resolve().parents[2]
        bad = []
        for rel in ("strata_control", "strata_cli", "serve"):
            for path in (root / rel).rglob("*.py"):
                # mac_fit.py is macOS only; engine_component's one call is behind sys.platform.startswith("linux") on the same line
                if path.name.startswith("test_") or path.name in ("sysinfo.py", "mac_fit.py"):
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if line.lstrip().startswith("#") or "`platform." in line:
                        continue
                    if re.search(r"(?<![\w.])platform\.(system|machine|release|version|processor|node|platform|uname)\(\)", line) \
                            and "linux" not in line:
                        bad.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
