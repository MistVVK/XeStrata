"""The Terminal / Control bundle holds every tools/ module that setup.py can import (#3613, #3467).

0.1.47 shipped without tools/calibrate.py: setup.py imports it after a tune, so every launch after a tune ended in
ModuleNotFoundError. The bundle is staged here the way packaging/package_terminal.py stages it (the payload, the read-only tools,
then add_installer), and setup.py's tool imports are resolved against tools/ as the bundle has it, not against the source tree.
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))
import package_terminal as pt  # noqa: E402


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


class BundleImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="strata-bundle-")
        cls.root = Path(cls.temp.name) / "strata-terminal"
        cls.root.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for item in pt.PAYLOAD:
            src, dst = pt.source_of(item), cls.root / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=ignore)
            else:
                shutil.copy2(src, dst)
        (cls.root / "tools").mkdir(exist_ok=True)
        for item in pt.TOOLS_SUBSET:
            shutil.copy2(ROOT / "tools" / item, cls.root / "tools" / item)
        cls.installer = pt.add_installer(cls.root, ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_gguf_py_and_the_converter_are_in_the_bundle(self):
        """OWN-1B: the Terminal and Control packages carry third_party/strata-gguf-py (the preparation tools' gguf-py and llama.cpp's converter), so
        preparing a model downloads no llama.cpp; the bundled copy is byte-identical to the source tree's."""
        here = self.root / "third_party" / "strata-gguf-py"
        for rel in ("gguf/__init__.py", "gguf/quants.py", "converter/convert_hf_to_gguf.py", "converter/conversion/qwen4exp.py", "LICENSE", "UPSTREAM.md"):
            self.assertTrue((here / rel).is_file(), rel)
            self.assertEqual((here / rel).read_bytes(), (ROOT.parent / "engine" / "third_party" / "strata-gguf-py" / rel).read_bytes(), rel)
        self.assertIn("third_party/strata-gguf-py/", self.installer["files"])
        probe = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(self.root / 'tools')!r}); import _paths; print(_paths.gguf_py())"],
                               capture_output=True, text=True, timeout=60, env={k: v for k, v in os.environ.items() if k != "STRATA_GGUF_PY"})
        self.assertEqual(probe.returncode, 0, probe.stderr[-500:])
        self.assertEqual(Path(probe.stdout.strip()).resolve(), here.resolve(), "a bundled tool finds the bundled copy")

    def test_calibrate_is_in_the_bundle(self):
        # the exact file #3613 and #3467 lacked
        self.assertTrue((self.root / "tools" / "calibrate.py").is_file())
        self.assertIn("calibrate", pt.INSTALLER_TOOLS)

    def test_every_tools_module_setup_py_imports_is_in_the_bundle(self):
        in_bundle = {p.stem for p in (self.root / "tools").glob("*.py")}
        in_source = {p.stem for p in (ROOT / "tools").glob("*.py")}
        wanted = imports_of(self.root / "setup.py") & in_source
        self.assertIn("calibrate", wanted, "setup.py no longer imports calibrate: update this test")
        self.assertEqual(sorted(wanted - in_bundle), [], "setup.py imports tools/ modules that the bundle does not carry")

    def test_the_bundled_tools_import_nothing_the_bundle_lacks(self):
        in_bundle = {p.stem for p in (self.root / "tools").glob("*.py")}
        in_source = {p.stem for p in (ROOT / "tools").glob("*.py")}
        for module in sorted(in_bundle):
            missing = sorted((imports_of(self.root / "tools" / f"{module}.py") & in_source) - in_bundle)
            self.assertEqual(missing, [], f"tools/{module}.py imports tools/ modules the bundle does not carry")

    def test_setup_py_and_each_of_its_tools_import_with_tools_limited_to_the_bundle(self):
        modules = sorted(set(pt.setup_tool_imports(self.root)) | set(pt.INSTALLER_TOOLS))
        probe = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(self.root / 'tools')!r}); import _paths; _paths.gguf_py()"],
                               capture_output=True, text=True, timeout=60, env={**os.environ, "STRATA_CONTROL_INSTALL": "1"})
        if probe.returncode != 0:      # a tool reads llama.cpp's gguf-py when it is imported: a test box without any llama.cpp checkout cannot import it
            self.skipTest("no llama.cpp gguf-py on this box (set STRATA_GGUF_PY to run this check)")
        code = ("import sys, os, importlib\n"
                f"sys.path[:0] = [{str(self.root)!r}, {str(self.root / 'tools')!r}]\n"
                "import setup\n"
                f"for name in {modules!r}:\n"
                "    module = importlib.import_module(name)\n"
                f"    assert os.path.samefile(os.path.dirname(module.__file__), {str(self.root / 'tools')!r}), (name, module.__file__)\n"
                "print('imports ok')\n")
        env = {**os.environ, "STRATA_CONTROL_INSTALL": "1", "PYTHONNOUSERSITE": "1"}
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        done = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(self.root.parent), env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, (done.stdout or "") + (done.stderr or "")[-1500:])
        self.assertIn("imports ok", done.stdout)

    def test_a_tool_setup_starts_to_import_is_added_without_editing_the_list(self):
        # setup_tool_imports reads setup.py, so INSTALLER_TOOLS cannot go stale the way it did for calibrate
        self.assertTrue(set(pt.setup_tool_imports(ROOT)) <= set(pt.local_imports(ROOT, tuple(sorted(
            set(pt.INSTALLER_TOOLS) | set(pt.setup_tool_imports(ROOT)))))))
        without = tuple(x for x in pt.INSTALLER_TOOLS if x != "calibrate")
        self.assertNotIn("calibrate", pt.local_imports(ROOT, without))      # the old list really lacked it
        self.assertIn("calibrate", pt.setup_tool_imports(ROOT))             # and the scan finds it anyway


class ProbeLeavesNoBytecode(unittest.TestCase):
    """The import probe of package_terminal.py must not write compiled files into the bundle, whoever runs the build."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-probe-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        (self.root / "probe_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
        self.code = "import probe_mod; print('probe ok', probe_mod.VALUE)"
        self.saved = {k: os.environ.pop(k, None) for k in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
        self.addCleanup(self.restore)

    def restore(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def leftovers(self, folder: Path):
        return sorted(str(p.relative_to(folder)) for p in list(folder.rglob("__pycache__")) + list(folder.rglob("*.pyc")))

    def test_the_check_itself_can_see_bytecode(self):
        # without the protection the same import does leave a compiled file, so the test below proves something
        plain = Path(self.temp.name) / "plain"
        plain.mkdir()
        (plain / "probe_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
        done = subprocess.run([sys.executable, "-c", self.code], cwd=str(plain), capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(self.leftovers(plain))

    def test_the_probe_writes_no_bytecode_when_the_caller_did_not_ask_for_that(self):
        done = pt.run_probe(Path(sys.executable), self.root, code=self.code)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("probe ok 7", done.stdout)
        self.assertEqual(self.leftovers(self.root), [])

    def test_the_probe_writes_no_bytecode_when_the_caller_asked_for_it(self):
        os.environ["PYTHONDONTWRITEBYTECODE"] = ""        # an empty value means "write"
        done = pt.run_probe(Path(sys.executable), self.root, code=self.code)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.leftovers(self.root), [])


if __name__ == "__main__":
    unittest.main()
