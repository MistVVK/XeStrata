"""The CUDA Linux package scripts build strata_kq only when the build tree has it (packaging/engine_targets.sh).

A build configured with -DSTRATA_MMQ=OFF has no strata_kq target; `--target strata strata_kq` stopped such a package build with
"ninja: error: unknown target 'strata_kq'". The scripts now ask the build tree; a build with MMQ builds the same two targets as before.

    python -m unittest tools.test_engine_targets -v
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT.parent / "tools" / "packaging" / "engine_targets.sh"
SCRIPTS = {"tools/packaging/build_cuda_linux.sh": "cmake", "engine/spark/build_engine.sh": '"$cmake"'}

FAKE_CMAKE = """#!/bin/sh
echo "$*" >> "$FAKE_ARGS"
case "$FAKE_MODE" in
  kq) echo "[1/3] Building CUDA object libstrata_kq.so"; exit 0 ;;
  nokq) echo "ninja: error: unknown target 'strata_kq'"; exit 1 ;;
  nokq_hint) echo "ninja: error: unknown target 'strata_kq', did you mean 'strata'?"; exit 1 ;;
  make_nokq) echo "make: *** No rule to make target 'strata_kq'.  Stop."; exit 2 ;;
  *) echo "ninja: error: loading 'build.ninja': No such file or directory"; exit 1 ;;
esac
"""


def run_helper(cmake: str, build: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", '. "$0"; engine_targets "$1" "$2"', str(HELPER), cmake, build],
                          capture_output=True, text=True, env=env, timeout=600)


@unittest.skipIf(os.name == "nt" or shutil.which("sh") is None, "POSIX sh")
class EngineTargetsFake(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cmake = self.tmp / "cmake"
        self.cmake.write_text(FAKE_CMAKE, encoding="utf-8", newline="\n")
        self.cmake.chmod(self.cmake.stat().st_mode | stat.S_IXUSR)
        self.args = self.tmp / "args.txt"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, mode: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, FAKE_MODE=mode, FAKE_ARGS=str(self.args))
        return run_helper(str(self.cmake), "/b/build-x", env)

    def test_a_tree_with_strata_kq_builds_both_targets_as_before(self):
        r = self.call("kq")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "strata strata_kq\n")
        self.assertEqual(self.args.read_text().split(), ["--build", "/b/build-x", "--target", "strata_kq", "--", "-n"])

    def test_a_tree_without_strata_kq_builds_strata_alone(self):
        for mode in ("nokq", "nokq_hint", "make_nokq"):
            r = self.call(mode)
            self.assertEqual((r.returncode, r.stdout), (0, "strata\n"), mode)

    def test_any_other_failure_stops_the_build_and_shows_why(self):
        r = self.call("broken")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "")
        self.assertIn("loading 'build.ninja'", r.stderr)
        self.assertIn("cannot tell", r.stderr)


@unittest.skipIf(os.name == "nt" or not (shutil.which("sh") and shutil.which("cmake") and shutil.which("ninja")), "sh, cmake and ninja")
class EngineTargetsRealNinja(unittest.TestCase):
    """A real CMake + Ninja tree with and without a strata_kq target: the dry run builds nothing."""

    def tree(self, with_kq: bool) -> Path:
        src = self.tmp / ("src-kq" if with_kq else "src")
        src.mkdir()
        made = self.tmp / ("made-kq" if with_kq else "made")
        lines = ["cmake_minimum_required(VERSION 3.16)", "project(t NONE)",
                 "add_custom_target(strata COMMAND ${CMAKE_COMMAND} -E touch strata.out)"]
        if with_kq:
            lines.append(f'add_custom_target(strata_kq COMMAND ${{CMAKE_COMMAND}} -E touch "{made.as_posix()}")')
        (src / "CMakeLists.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        build = self.tmp / ("build-kq" if with_kq else "build")
        subprocess.run(["cmake", "-G", "Ninja", "-S", str(src), "-B", str(build)], check=True, capture_output=True, timeout=600)
        return build

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_trees(self):
        r = run_helper("cmake", str(self.tree(True)))
        self.assertEqual((r.returncode, r.stdout), (0, "strata strata_kq\n"), r.stderr)
        self.assertFalse((self.tmp / "made-kq").exists(), "the dry run ran a build command")
        r = run_helper("cmake", str(self.tree(False)))
        self.assertEqual((r.returncode, r.stdout), (0, "strata\n"), r.stderr)


class ScriptsUseTheHelper(unittest.TestCase):
    def test_the_cuda_linux_scripts_ask_the_build_tree(self):
        for rel, cmake in SCRIPTS.items():
            text = (ROOT.parent / rel).read_text(encoding="utf-8")
            self.assertNotIn("--target strata strata_kq", text, rel)
            self.assertIn('. "$repo/tools/packaging/engine_targets.sh"', text, rel)
            self.assertIn(f'targets=$(engine_targets {cmake} "$build")', text, rel)
            self.assertIn('--target $targets', text, rel)

    def test_the_helper_is_lf(self):
        self.assertNotIn(b"\r", HELPER.read_bytes())


if __name__ == "__main__":
    sys.exit(unittest.main())
