"""The source build's context (tools/docker/Dockerfile.src with its own ignore list) still holds what the build compiles.

Since the layout move `/engine/` is the engine's SOURCE (the built engine goes to /terminal/engine/), but the ignore list still carried the
old `/engine/` line, so every `docker build -f tools/docker/Dockerfile.src` copied a context without `engine/` and stopped at `cmake -S engine`.
This reads the ignore list the way docker does (every pattern is anchored at the context root, `*` stays inside one folder, a folder pattern
covers what is below it) and checks it against what git tracks."""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
IGNORE = REPO / "tools" / "docker" / "Dockerfile.src.dockerignore"
DOCKERFILE = REPO / "tools" / "docker" / "Dockerfile.src"
GIT = shutil.which("git")


def _regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
            continue
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
            continue
        out += "[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c)
        i += 1
    return re.compile(out + r"\Z")


def ignore_rules(text: str) -> list[tuple[bool, re.Pattern]]:
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        line = line[1:] if negate else line
        line = line.lstrip("/").rstrip("/")
        rules.append((negate, _regex(line)))
    return rules


def excluded(path: str, rules) -> bool:
    """docker's rule: the last pattern that matches the path or one of its parent folders decides."""
    parts = path.split("/")
    verdict = False
    for negate, rx in rules:
        if any(rx.match("/".join(parts[:n])) for n in range(1, len(parts) + 1)):
            verdict = not negate
    return verdict


def tracked() -> list[str]:
    out = subprocess.run([GIT, "-C", str(REPO), "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p]


class SourceBuildContext(unittest.TestCase):
    def setUp(self):
        self.rules = ignore_rules(IGNORE.read_text(encoding="utf-8"))

    def test_the_matcher_reads_the_list_the_way_docker_does(self):
        rules = ignore_rules("/engine/\n!/engine/keep\n*.log\nbuild/\n")
        self.assertTrue(excluded("engine/a/b.txt", rules))
        self.assertFalse(excluded("engine/keep", rules))
        self.assertTrue(excluded("x.log", rules))
        self.assertFalse(excluded("sub/x.log", rules))      # not gitignore: a pattern without a slash does not reach into folders
        self.assertTrue(excluded("build/y", rules))

    def test_the_engine_source_is_in_the_context(self):
        for path in ("engine/CMakeLists.txt", "engine/src", "engine/third_party/strata-ggml", "engine/third_party/strata-gguf-py",
                     "terminal/tools/vision/CMakeLists.txt", "terminal/requirements.txt", "tools/packaging/package_engine.py",
                     "tools/packaging/engine_targets.sh", "tools/docker/entrypoint.sh", "tools/docker/strata-wrapper.sh"):
            self.assertFalse(excluded(path, self.rules), f"{path} is cut out of the source build's context")

    def test_the_built_engine_folders_stay_out(self):
        for path in ("terminal/engine/strata", "terminal/engine-sycl/x", "terminal/engine-cuda12/x", "engine-sycl/x", "engine-cuda12/x",
                     "build/CMakeCache.txt", "engine/third_party/llama.cpp/x", ".git/config"):
            self.assertTrue(excluded(path, self.rules), f"{path} should not be copied into the build")

    @unittest.skipIf(GIT is None or not (REPO / ".git").exists(), "needs a git checkout")
    def test_no_tracked_file_the_dockerfile_uses_is_cut_out(self):
        """Everything git tracks under engine/, terminal/ (except its tests and data) and tools/ is part of some build stage; none may be dropped."""
        cut = [p for p in tracked() if p.split("/")[0] in ("engine", "tools") and excluded(p, self.rules)]
        cut += [p for p in tracked() if p.startswith("terminal/") and not p.startswith(("terminal/tests/", "terminal/data/", "terminal/bench/"))
                and excluded(p, self.rules)]
        self.assertEqual(cut, [], f"tracked files the ignore list drops: {cut[:10]}")

    def test_every_documented_source_build_names_the_dockerfile(self):
        """There is no Dockerfile at the repository root: a command without `-f tools/docker/Dockerfile.src` finds none (and the ignore list
        next to the Dockerfile is only used when it is named)."""
        named = r"-f tools/docker/Dockerfile\.src(?![.\w])"
        for path in (REPO / "docs" / "DOCKER.md", DOCKERFILE):
            text = path.read_text(encoding="utf-8")
            builds = len(re.findall(r"docker build -t strata:src", text))
            self.assertGreaterEqual(builds, 2, path.name)
            self.assertEqual(builds, len(re.findall(named, text)), f"{path.name}: a source build command has no -f")
        self.assertFalse((REPO / "Dockerfile").exists())

    def test_the_dockerfile_compiles_from_engine(self):
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("-S engine ", text)        # so `engine/` must be in the context; the check above is the guard


if __name__ == "__main__":
    unittest.main()
