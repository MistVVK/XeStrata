#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build intel/llvm's DPC++ (free software: Apache-2.0 with LLVM exceptions) from source into .tools/intel-llvm/, for a
free build of the engine where the distribution has no DPC++, or one whose SYCL runtime does not give the GPU its
matrix engines (Ubuntu 26.04's 6.2 lists none for the Arc Pro B70; intel/llvm added it in v7.0.0).

    tools/intel_llvm_build.py [--rebuild] [--keep-build] [--tag TAG] [--yes]

.tools/intel-llvm/src is the clone, build/ the build tree, install/ the toolchain (bin/clang++, lib/libsycl.so).  A
finished install/ carries XESTRATA.json (the tag, its commit, the date, the GPUs' matrix engines as tools/xmx_probe.cpp
saw them).  Run again, it uses a finished install of the same tag as it is, asks before building another tag over an
older one, and continues an interrupted build where it stopped.  build/ is deleted once install/ is done, unless
--keep-build: keeping it makes the next tag's build incremental.  v7.1.1 built in 13 minutes on 28 threads and keeps
3.5 GB (the clone 2.8, install/ 0.7).

Prints the toolchain's folder on its last line.  setup.py runs it for --intel-llvm-build."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / ".tools" / "intel-llvm"
SRC, BUILD, INSTALL = BASE / "src", BASE / "build", BASE / "install"
RECORD = INSTALL / "XESTRATA.json"
REPO = "https://github.com/intel/llvm.git"
# The release this script builds: one with the Arc Pro B70 in the runtime's matrix table (v7.0.0 or later).  Raised by
# a change to this line, not by following the newest release, so a run does not start a build by itself.
TAG = "v7.1.1"


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    sys.exit(1)


def ask(question, choices, default, yes):
    if yes or not sys.stdin.isatty():
        return default
    while True:
        a = input(f"{question} [{default}]: ").strip() or default
        if a in choices:
            return a
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if r.returncode != 0:
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")


def missing_prerequisites() -> list:
    """What intel/llvm's build needs (sycl/doc/GetStartedGuide.md), as Debian / Ubuntu packages."""
    need = []
    for tool, pkg in (("git", "git"), ("cmake", "cmake"), ("ninja", "ninja-build"), ("g++", "g++")):
        if shutil.which(tool) is None:
            need.append(pkg)
    if not Path("/usr/include/hwloc.h").exists():
        need.append("libhwloc-dev")
    return need


def probe(install: Path) -> list:
    """tools/xmx_probe.cpp built with the toolchain in `install` and run on its runtime: one dict per Level Zero GPU."""
    exe = BASE / "xmx_probe"
    env = dict(os.environ, LD_LIBRARY_PATH=str(install / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([str(install / "bin" / "clang++"), "-fsycl", str(ROOT / "tools" / "xmx_probe.cpp"), "-o",
                        str(exe)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return []
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    gpus = []
    for line in r.stdout.splitlines():
        f = dict(x.split("=", 1) for x in line.split(" name=")[0].split() if "=" in x)
        f["name"] = line.split(" name=", 1)[1] if " name=" in line else ""
        gpus.append(f)
    return gpus


def finished():
    if not RECORD.exists() or not (INSTALL / "bin" / "clang++").exists():
        return None
    try:
        return json.loads(RECORD.read_text())
    except (OSError, ValueError):
        return None


def build(tag: str, keep_build: bool) -> None:
    need = missing_prerequisites()
    if need:
        fail("intel/llvm's build needs: " + ", ".join(need), "install them: sudo apt install " + " ".join(need))
    BASE.mkdir(parents=True, exist_ok=True)
    if (SRC / ".git").exists():
        have = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--exact-match"], capture_output=True,
                              text=True).stdout.strip()
        if have != tag:
            run(["git", "-C", SRC, "fetch", "--depth", "1", "origin", "tag", tag])
            run(["git", "-C", SRC, "checkout", "--detach", tag])
    else:
        run(["git", "clone", "--depth", "1", "--branch", tag, REPO, SRC])
    # The configuration downloads what its release pins (Level Zero's headers and loader: the runtime adapter needs
    # newer ones than Ubuntu 26.04's 1.28; emhash), all free software; giving it the distribution's failed.
    if not (BUILD / "build.ninja").exists():
        run([sys.executable, SRC / "buildbot" / "configure.py", "-o", BUILD, "-t", "Release",
             f"--cmake-opt=-DCMAKE_INSTALL_PREFIX={INSTALL}"])
    jobs = max(2, min(os.cpu_count() or 4, int(ram_gb() // 3)))
    started = time.time()
    run([sys.executable, SRC / "buildbot" / "compile.py", "-o", BUILD, "-j", str(jobs)])
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpus = probe(INSTALL)
    RECORD.write_text(json.dumps({"tag": tag, "commit": commit, "built": time.strftime("%Y-%m-%d %H:%M"),
                                  "minutes": round((time.time() - started) / 60), "gpus": gpus}, indent=1))
    if not keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)


def ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 16.0


def version_key(tag: str) -> tuple:
    return tuple(int(x) for x in tag.lstrip("v").split(".") if x.isdigit())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=TAG, help=f"the intel/llvm release to build (default {TAG})")
    ap.add_argument("--rebuild", action="store_true", help="build again even when a finished build is there")
    ap.add_argument("--keep-build", action="store_true", help="keep the build tree for the next update")
    ap.add_argument("--yes", action="store_true", help="answer the questions with their defaults")
    a = ap.parse_args()
    have = finished()
    if have and not a.rebuild:
        if have.get("tag") == a.tag:
            say(f"intel/llvm {a.tag} is already built in {INSTALL}")
        elif version_key(have.get("tag", "")) < version_key(a.tag):
            choice = ask(f"intel/llvm {have.get('tag')} is built; build {a.tag} over it? "
                         "keep = use the one built, build = build the new one", ["keep", "build"], "keep", a.yes)
            if choice == "build":
                build(a.tag, a.keep_build)
        else:
            say(f"intel/llvm {have.get('tag')} is built (newer than {a.tag}): using it")
    else:
        if (BUILD / "build.ninja").exists() and not a.rebuild:
            say("an interrupted build is there: continuing it")
        say(f"building intel/llvm {a.tag} in {BASE} (13 minutes on 28 threads) ...")
        build(a.tag, a.keep_build)
    have = finished()
    if have is None:
        fail(f"no finished build in {INSTALL}")
    if not have.get("gpus"):
        say("note: this build's SYCL runtime lists no GPU (xmx_probe): the GPU's Level Zero driver (Intel's "
            "compute-runtime) is missing or too old for it")
    elif not any(g.get("fp16") == "1" and g.get("bf16") == "1" for g in have.get("gpus", [])):
        say("note: no GPU reports FP16 and BF16 matrix engines to this build (xmx_probe); the engine will run its "
            "products without XMX")
    say(str(INSTALL))


if __name__ == "__main__":
    main()
