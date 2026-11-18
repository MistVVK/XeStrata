"""DIAG-WINHOST-0216: no Strata process on Windows may reach CPython 3.12's `_wmi`.

`platform.system()` / `uname()` ask WMI through `_wmi.exec_query` on Windows with Python 3.12. When WMI refuses the account (HRESULT 0x80041003, a
standard user over an ssh logon on the Strix test box) the query takes about 5 s, `exec_query` gives up after 100 ms and returns, and its worker
thread later reads a struct on the caller's gone stack: an access violation (0xC0000005) that ended the runtime host ("runtime control is
unavailable") and, with it, the server and the engine. `truststore`, imported by `strata_control/tls.py`, calls `platform.system()` at import.

The fix: every Strata entry point puts `sys.modules["_wmi"] = None` first (so `platform` falls back to the registry, `ver` and the environment),
`tls.install()` and `sysinfo.block_wmi()` guard a process that imported `platform` earlier, and the bundle's own sitecustomize.py does it for every
process of the bundled interpreter. These tests:

* check the guard is the FIRST thing each shipped entry file and package does (static, runs everywhere) and fails for a new entry point without it;
* check, on Windows, that a fresh process that imports each entry module never loads `_wmi` and still answers platform.system() == "Windows";
* check `sysinfo.block_wmi()` and the bundle's sitecustomize (anywhere, with sys.platform faked).

    python -m pytest tests/terminal/test_wmi_guard.py
"""
from __future__ import annotations

import ast
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]            # the terminal folder
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))
sys.path.insert(0, str(ROOT))
import package_terminal as pt  # noqa: E402
from strata_control import sysinfo  # noqa: E402

MARK = 'sys.modules["_wmi"] = None'

#: test fixtures that live inside a shipped package (run only by tests, never by a person); the package's own __init__ guard covers them anyway
NOT_SHIPPED = {"serve/mcp_fake_server.py", "serve/fixtures"}

#: the packages a process imports first when it runs `python -m <package>.<module>`: their __init__ must carry the guard too
PACKAGES = ("strata_control", "serve", "strata_cli")


def _has_main(text: str) -> bool:
    return "__name__ == \"__main__\"" in text or "__name__ == '__main__'" in text


def shipped_entry_files() -> list[str]:
    """Every Python file of the Terminal bundle that can be the start of a process: strata.py, the installer files, the tools the bundle runs by
    path or imports, and any file of the shipped packages with an `if __name__ == "__main__"` block (the `python -m` entries)."""
    found: set[str] = {"strata.py", "setup.py", "ref/load.py"}
    found.update(f"tools/{name}.py" for name in (*pt.INSTALLER_TOOLS, *pt.INSTALLER_SCRIPTS))
    found.update(f"tools/{name}" for name in pt.TOOLS_SUBSET)
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if any(rel == skip or rel.startswith(skip + "/") for skip in NOT_SHIPPED) or "/tests/" in rel or "__pycache__" in rel \
                    or path.name.startswith("test_"):
                continue
            if rel == f"{package}/__init__.py" or _has_main(path.read_text(encoding="utf-8-sig")):
                found.add(rel)           # (a sub-package's __init__ runs only after its package's, which carries the guard)
    return sorted(found)


def guard_problem(path: Path) -> str | None:
    """None when the guard is in `path` before any import that could pull in platform / truststore; else what is wrong."""
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    for node in tree.body:
        segment = ast.get_source_segment(text, node) or ""
        if isinstance(node, ast.If) and MARK in segment and "win32" in segment:
            return None
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue                                                     # the docstring
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import) and all(a.name in ("sys", "os") for a in node.names):
            continue
        return f"a statement comes before the guard (line {node.lineno}): {segment.splitlines()[0][:80] if segment else ast.dump(node)[:80]}"
    return f'no `if sys.platform == "win32": {MARK}` at the top level'


class EveryEntryPointHasTheGuard(unittest.TestCase):
    def test_the_list_is_not_empty_and_has_the_known_entries(self):
        files = shipped_entry_files()
        for must in ("strata.py", "setup.py", "serve/server.py", "strata_control/runtime.py", "strata_control/__init__.py", "serve/__init__.py",
                     "strata_cli/__init__.py", "tools/strata_mcp.py", "serve/mcp.py", "strata_control/management.py"):
            self.assertIn(must, files)
        for rel in files:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_each_entry_file_starts_with_the_guard(self):
        bad = {}
        for rel in shipped_entry_files():
            problem = guard_problem(ROOT / rel)
            if problem:
                bad[rel] = problem
        self.assertFalse(bad, "an entry point without the _wmi guard first (add the block from strata.py, after the docstring and any "
                              "`from __future__` line, before every other import):\n" + "\n".join(f"  {k}: {v}" for k, v in bad.items()))

    def test_a_file_without_the_guard_is_caught(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            late = Path(tmp) / "late.py"
            late.write_text('"""x"""\nimport platform\nimport sys\nif sys.platform == "win32":\n    sys.modules["_wmi"] = None\n', encoding="utf-8")
            none = Path(tmp) / "none.py"
            none.write_text('import sys\nprint(1)\nif __name__ == "__main__":\n    pass\n', encoding="utf-8")
            ok = Path(tmp) / "ok.py"
            ok.write_text('"""x"""\nfrom __future__ import annotations\nimport sys\nif sys.platform == "win32":\n    sys.modules["_wmi"] = None\n\nimport platform\n',
                          encoding="utf-8")
            self.assertIsNotNone(guard_problem(late))
            self.assertIsNotNone(guard_problem(none))
            self.assertIsNone(guard_problem(ok))

    def test_every_main_file_in_the_shipped_packages_is_listed(self):
        """A new `python -m` / script entry in a shipped package is found by shipped_entry_files() itself; this pins that nothing is skipped by accident."""
        listed = set(shipped_entry_files())
        for package in PACKAGES:
            for path in (ROOT / package).rglob("*.py"):
                rel = path.relative_to(ROOT).as_posix()
                if "/tests/" in rel or "__pycache__" in rel or path.name.startswith("test_") or any(rel == s or rel.startswith(s + "/") for s in NOT_SHIPPED):
                    continue
                if _has_main(path.read_text(encoding="utf-8-sig")):
                    self.assertIn(rel, listed)

    def test_the_bundle_gets_a_sitecustomize_with_the_guard(self):
        self.assertIn(MARK, pt.SITECUSTOMIZE)
        self.assertIsNone(_guard_problem_of_text(pt.SITECUSTOMIZE))
        import tempfile
        for target in ("windows-x86_64", "linux-x86_64"):
            with tempfile.TemporaryDirectory() as tmp:
                rel = pt.add_sitecustomize(Path(tmp), target)
                self.assertTrue(rel.endswith("site-packages/sitecustomize.py"), rel)
                self.assertEqual((Path(tmp) / rel).read_text(encoding="utf-8"), pt.SITECUSTOMIZE)
                pt.add_sitecustomize(Path(tmp), target)                       # again: same file, no error
                (Path(tmp) / rel).write_text("# somebody else's\n", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    pt.add_sitecustomize(Path(tmp), target)


def _guard_problem_of_text(text: str):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.py"
        p.write_text(text, encoding="utf-8")
        return guard_problem(p)


class BlockWmi(unittest.TestCase):
    """sysinfo.block_wmi(), with sys.platform faked (the real Windows run is below)."""

    def test_not_windows_does_nothing(self):
        with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(sys.modules):
            sys.modules.pop("_wmi", None)
            self.assertFalse(sysinfo.block_wmi())
            self.assertNotIn("_wmi", sys.modules)

    def test_windows_blocks_the_module_and_makes_platform_raise_the_oserror_it_handles(self):
        def asks_wmi(table, *keys):
            raise AssertionError("WMI was asked")
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(sys.modules), \
                mock.patch.object(platform, "_wmi_query", asks_wmi, create=True):
            self.assertTrue(sysinfo.block_wmi())
            self.assertIsNone(sys.modules["_wmi"])
            with self.assertRaises(OSError):
                platform._wmi_query("OS", "Version")
            first = platform._wmi_query
            self.assertTrue(sysinfo.block_wmi())                               # again: the same guard stays, nothing stacks
            self.assertIs(platform._wmi_query, first)

    def test_a_platform_without_wmi_query_is_left_alone(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(sys.modules):
            had = hasattr(platform, "_wmi_query")
            saved = getattr(platform, "_wmi_query", None)
            if had:
                delattr(platform, "_wmi_query")
            try:
                self.assertTrue(sysinfo.block_wmi())
                self.assertFalse(hasattr(platform, "_wmi_query"))
            finally:
                if had:
                    platform._wmi_query = saved

    def test_tls_install_blocks_wmi_before_truststore(self):
        from strata_control import tls
        seen = []
        real_import = __import__

        def spy(name, *a, **k):
            if name == "truststore":
                seen.append(sys.modules.get("_wmi", "absent"))
                raise ImportError("test: no truststore")
            return real_import(name, *a, **k)
        import ssl
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(sys.modules), mock.patch.object(tls, "mode", None), \
                mock.patch.object(ssl, "_create_default_https_context", ssl._create_default_https_context), \
                mock.patch("builtins.__import__", spy):
            tls.install()
        self.assertEqual(seen, [None], "truststore must be imported only after _wmi is blocked")


@unittest.skipUnless(sys.platform == "win32", "the real _wmi exists on Windows only")
class FreshProcessesNeverLoadWmi(unittest.TestCase):
    """Each entry module is imported in a fresh interpreter: `_wmi` is not loaded (blocked or absent), platform.system() is still "Windows"."""

    CODE = ("import importlib, sys\n"
            "name = sys.argv[1]\n"
            "importlib.import_module(name)\n"
            "import platform\n"
            "loaded = sys.modules.get('_wmi')\n"
            "assert loaded is None, ('_wmi is loaded', loaded)\n"
            "assert platform.system() == 'Windows', platform.system()\n"
            "assert platform.release() in ('10', '11') or platform.release(), platform.release()\n"
            "assert platform.version().count('.') >= 2, platform.version()\n"
            "assert platform.machine() in ('AMD64', 'ARM64', 'x86'), platform.machine()\n"
            "print('ok')\n")

    #: importable without starting anything (a module with a heavy or interactive import side effect is left to the static check)
    MODULES = ("strata_control", "serve", "strata_cli", "strata_control.runtime", "strata_control.management", "strata_control.pictures",
               "strata_control.app_policy", "strata_control.tls", "strata_control.sysinfo", "serve.server", "serve.mcp", "serve.cpu_affinity",
               "setup", "tools.strata_mcp", "tools.mtp_fetch", "tools.strata_tokenizer")

    def test_import_each_entry_in_a_fresh_process(self):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "STRATA_DIAG_LOG")}
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "tools")])
        for name in self.MODULES:
            with self.subTest(module=name):
                done = subprocess.run([sys.executable, "-c", self.CODE, name], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(done.returncode, 0, f"{name}: {done.stdout[-300:]} {done.stderr[-600:]}")
                self.assertIn("ok", done.stdout)

    def test_strata_py_by_path(self):
        done = subprocess.run([sys.executable, "-c",
                               "import runpy, sys; sys.argv = ['strata.py', '--version']\n"
                               "try:\n    runpy.run_path('strata.py', run_name='__main__')\n"
                               "except SystemExit:\n    pass\n"
                               "assert sys.modules.get('_wmi') is None, sys.modules.get('_wmi')\nprint('ok')"],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertIn("ok", done.stdout)


if __name__ == "__main__":
    unittest.main()
