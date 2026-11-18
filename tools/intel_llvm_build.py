#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2022-2026 The authors of Debian's rocm-llvm packaging (debian/*; named in NOTICE)
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Build intel/llvm's DPC++ (free software: Apache-2.0 with LLVM exceptions) from source into .tools/intel-llvm/, for a
free build of the engine where the distribution has no DPC++, or one whose SYCL runtime does not give the GPU its
matrix engines (Ubuntu 26.04's 6.2 lists none for the Arc Pro B70; intel/llvm added it in v7.0.0).

    tools/intel_llvm_build.py [--contrib] [--rebuild] [--keep-build] [--tag TAG] [--yes]
    tools/intel_llvm_build.py --windows [--xwin DIR] [--rebuild] [--keep-build] [--tag TAG]

.tools/intel-llvm/src is the clone, build/ the build tree, install/ the toolchain (bin/clang++, lib/libsycl.so).
Where ROCm's HIP is installed (free software), it builds the HIP target (AMD GPUs) as well, with ROCm's device
libraries built by the same compiler (DEVICE_LIBS_TAG, below).  --contrib builds the same
clone with the CUDA target too (NVIDIA GPUs; it needs NVIDIA's CUDA toolkit, which is not free software: the contrib
build mode), into .tools/intel-llvm-contrib/ (build/, install/).  --windows cross-builds the SYCL runtime for Windows
from the same clone, with the --contrib toolchain's tools, into .tools/intel-llvm-win/ (a Windows engine's headers,
import libraries and DLLs).  A finished install/ carries XESTRATA.json (the tag,
its commit, the date, the GPUs' matrix engines as tools/xmx_probe.cpp saw them).  Run again, it uses a finished install
of the same tag as it is, asks before building another tag over an older one, and continues an interrupted build where
it stopped.  build/ is deleted once install/ is done, unless --keep-build: keeping it makes the next tag's build
incremental.  v7.1.1 built in 13 minutes on 28 threads and keeps 3.5 GB (the clone 2.8, install/ 0.7).

Prints the toolchain's folder on its last line.  setup.py runs it for --intel-llvm-build."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / ".tools" / "intel-llvm"
SRC = BASE / "src"
# where the build goes: BASE, or the contrib one's folder (set_out)
OUT, BUILD, INSTALL, RECORD = BASE, BASE / "build", BASE / "install", BASE / "install" / "XESTRATA.json"

REPO = "https://github.com/intel/llvm.git"
# The release this script builds: one with the Arc Pro B70 in the runtime's matrix table (v7.0.0 or later).  Raised by
# a change to this line, not by following the newest release, so a run does not start a build by itself.
TAG = "v7.2.0"

# ROCm's device libraries (ocml, ockl and the others; the University of Illinois/NCSA license), which the AMD code
# links: built with the toolchain, into install/amdgcn/bitcode, since a clang reads bitcode of its own LLVM or older
# only - ROCm 7.14's (LLVM 23) stopped v7.1.1's (LLVM 22) build for an RX 7900 XTX (a field report).  CMake takes them
# where the installed ROCm's do not build (cmake/StrataHip.cmake).  amd/device-libs of
# ROCm's llvm-project at a release on TAG's LLVM, raised with TAG: v7.2.0 is LLVM 23, as ROCm 7.14 is (10.1 is LLVM
# 24).  ROCm's releases since 7.9 are TheRock's, which tags no llvm-project commit: the one its release tag pins
# (TheRock's compiler/amd-llvm at therock-7.14.1).  third_party/main/rocm-device-libs/.
DEVICE_LIBS_REPO = "https://github.com/ROCm/llvm-project.git"
DEVICE_LIBS_TAG = "f30ae3e6b680ad23dcf522e668c6d4c5c90838d0"   # ROCm 7.14.1 (TheRock's therock-7.14.1)
DEVICE_LIBS_SRC = BASE / "rocm-device-libs"

# Fixes the release lacks (its sycl branch too, on 2026-10-05): third_party/main/intel-llvm/patches/NN-<id>.patch,
# applied in order to the clone before the configuration (under intel/llvm's license, Apache-2.0 with LLVM
# exceptions).  A finished install records the ids and full-file hashes it was built with.
PATCHES = sorted((ROOT / "third_party" / "main" / "intel-llvm" / "patches").glob("[0-9][0-9]-*.patch"))


def patch_id(patch: Path) -> str:
    return patch.stem.split("-", 1)[1]


def patch_fingerprints() -> list[dict[str, str]]:
    """Names, order and contents identify the patches in a finished toolchain."""
    return [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in PATCHES]


def patched_files() -> tuple[list, list]:
    """The files the patches change and the files they add, as paths in the clone."""
    files, added, cur = set(), set(), ""
    for patch in PATCHES:
        for line in patch.read_text().splitlines():
            if line.startswith("diff --git a/"):
                cur = line.split()[2][2:]
                files.add(cur)
            elif line.startswith("new file mode") and cur:
                added.add(cur)
    return sorted(files - added), sorted(added)


def restore_release() -> None:
    """The files the patches change back to the release, and the files they add removed (git knows no such path to
    check out, and a patch does not apply over its own file)."""
    changed, added = patched_files()
    run(["git", "-C", SRC, "checkout", "--", *changed])
    for f in added:
        (SRC / f).unlink(missing_ok=True)


def set_out(out: Path) -> None:
    global OUT, BUILD, INSTALL, RECORD
    OUT, BUILD, INSTALL = out, out / "build", out / "install"
    RECORD = INSTALL / "XESTRATA.json"


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None) -> NoReturn:
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


def distro_family() -> str:
    """Which distribution's package names this PC's are ("ubuntu", "fedora", "arch"), as setup.py's distro_family
    reads it: ID_LIKE settles Arch's derivatives (CachyOS: ID=cachyos, ID_LIKE=arch), and a distribution none of the
    three names gets Ubuntu's list (apt)."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return "ubuntu"
    ids = " ".join(re.findall(r"^ID(?:_LIKE)?=\"?([^\"\n]*)", text, re.M)).split()
    for family in PREREQUISITES:
        if any(i == family or i.startswith(family) for i in ids):
            return family
    return "ubuntu"


# What the build needs (sycl/doc/GetStartedGuide.md) as the distribution's packages, and the command that installs
# them (setup.py's PACKAGES and INSTALLERS in the same names)
PREREQUISITES = {
    "ubuntu": {"git": "git", "cmake": "cmake", "ninja": "ninja-build", "g++": "g++", "ld.lld": "lld",
               "hwloc": "libhwloc-dev",
               "zstd": "libzstd-dev", "opencl": "opencl-headers", "cuda": "nvidia-cuda-toolkit",
               "how": "sudo apt install "},
    "fedora": {"git": "git", "cmake": "cmake", "ninja": "ninja-build", "g++": "gcc-c++", "ld.lld": "lld",
               "hwloc": "hwloc-devel",
               "zstd": "libzstd-devel", "opencl": "opencl-headers", "cuda": "cuda-toolkit-13-5",
               # (NVIDIA's CUDA repository, which Fedora's own do not carry the toolkit of: setup.py's NVIDIA_REPO.
               # It is fedora44's, which Fedora 45 uses as well - NVIDIA has no repository of its own yet; change the
               # URL when it publishes one, developer.download.nvidia.com/compute/cuda/repos/.  The toolkit's name
               # says the CUDA version: 13.4 on Fedora 44 as the packages build with, 13.5 on 45)
               "cuda_repo": "sudo dnf config-manager addrepo \\\n"
                            "    --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/"
                            "x86_64/cuda-fedora44.repo",
               "how": "sudo dnf install "},
    "arch": {"git": "git", "cmake": "cmake", "ninja": "ninja", "g++": "base-devel", "ld.lld": "lld",
             "hwloc": "hwloc",
             "zstd": "zstd", "opencl": "opencl-headers", "cuda": "cuda", "how": "sudo pacman -S --needed "},
}
# where CMake's find_library looks, per distribution
LIB_DIRS = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib")


def static_zstd() -> bool:
    """Whether the static zstd library, which the release's configure.py asks for (LLVM_USE_STATIC_ZSTD=ON), is here:
    Debian's libzstd-dev and Fedora's libzstd-static have it, Arch's zstd does not."""
    return any(Path(d, "libzstd.a").exists() for d in LIB_DIRS)


def missing_prerequisites() -> list:
    """What the build needs that is not installed, as this PC's distribution's packages."""
    p = PREREQUISITES[distro_family()]
    need = []
    for tool in ("git", "cmake", "ninja", "g++", "ld.lld"):
        if shutil.which(tool) is None:
            need.append(p[tool])
    if not Path("/usr/include/hwloc.h").exists():
        need.append(p["hwloc"])
    # intel/llvm's SYCL tools (sycl-ls, which the build installs) include CL/cl.h through sycl/detail/cl.h, and their
    # target gets no include for the OpenCL headers the configuration fetches (those belong to the runtime's own
    # objects), and no distribution's compiler brings the header in (a fresh Fedora lacks it as Arch does)
    if not Path("/usr/include/CL/cl.h").exists():
        need.append(p["opencl"])
    # the SYCL runtime reads zstd-compressed device images (oneMKL's, which oneMath calls in the contrib modes); without
    # the headers the configuration leaves zstd out silently.  With -DLLVM_ENABLE_ZSTD=FORCE_ON the configuration needs
    # the library too: configure.py asks for the static one (LLVM_USE_STATIC_ZSTD=ON), and where the distribution has
    # none (Arch) build() turns that off, which the release's own libraries and SYCL runtime take (intel/llvm#22294, in
    # v7.1.1: the shared library then).  Either way it is the devel package that has the headers.
    if not Path("/usr/include/zstd.h").exists() or not (
            static_zstd() or any(Path(d, "libzstd.so").exists() for d in LIB_DIRS)):
        need.append(p["zstd"])
    return need


def probe(install: Path) -> list:
    """tools/xmx_probe.cpp built with the toolchain in `install` and run on its runtime: one dict per Level Zero GPU."""
    exe = OUT / "xmx_probe"
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


def rocm_dirs() -> tuple | None:
    """ROCm's HIP for the HIP target: AMD's own tree (/opt/rocm, or the part ROCm 7.14's packages put it in,
    /opt/rocm/core-7.14, with nothing in /opt/rocm itself) or the distribution's (the headers in /usr/include, the
    libraries in Debian's and Ubuntu's multiarch folder or Fedora's /usr/lib64)."""
    # one tree, in setup.py's order (rocm_roots): /opt/rocm, its alternatives link core, the newest core-<version>
    cores = sorted(Path("/opt/rocm").glob("core-*"), key=lambda p: [int(n) for n in re.findall(r"\d+", p.name)],
                   reverse=True)
    for root in (Path("/opt/rocm"), Path("/opt/rocm/core"), *cores):
        if (root / "include/hip/hip_runtime_api.h").exists():
            return root, root / "lib"
    if Path("/usr/include/hip/hip_runtime_api.h").exists():
        for lib in [*sorted(Path("/usr/lib").glob("*/libamdhip64.so")), Path("/usr/lib64/libamdhip64.so")]:
            if lib.exists():
                return Path("/usr"), lib.parent
    return None


def target_options(contrib: bool) -> list:
    """configure.py's options for the GPU targets besides Intel's: HIP where ROCm is installed (free software, in both
    modes), and CUDA for the contrib toolchain."""
    opts: list = []
    if contrib:
        if shutil.which("nvcc") is None and not Path("/usr/local/cuda/include/cuda.h").exists():
            p = PREREQUISITES[distro_family()]
            fail("the CUDA target needs NVIDIA's CUDA toolkit (not free software)",
                 "install it:\n" + (f"  {p['cuda_repo']}\n" if "cuda_repo" in p else "") +
                 "  " + p["how"] + p["cuda"] +
                 (" (Ubuntu: multiverse; Debian: non-free)" if distro_family() == "ubuntu" else ""))
        opts.append("--cuda")
    rocm = rocm_dirs()
    if rocm is None:
        say("note: no ROCm HIP (hipcc's headers, libamdhip64): building without the HIP target (AMD GPUs)")
    else:
        # ROCm's device libraries as an external project: configure.py puts its external projects into
        # LLVM_ENABLE_PROJECTS too, whose folders LLVM takes from the clone's top (src/device-libs), so a link there.
        # They use AMDGPU builtins without an -mcpu, which upstream LLVM (and intel/llvm with it) allows only with
        # their target features: the four transcribed from Debian's rocm-llvm package (debian/rules of
        # 7.2.4+dfsg-1~exp1; Expat License, Copyright 2022-2026 its debian/ authors, see NOTICE), and msad-insts and
        # mqsad-pk-insts, which ROCm 7.14's ockl uses too (its media builtins; the others it takes in functions with
        # their own target attribute)
        link = SRC / "device-libs"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(fetch_device_libs())
        features = ";".join(f"-Xclang;-target-feature;-Xclang;+{f}"
                            for f in ("lerp-inst", "cube-insts", "qsad-insts", "sad-insts", "msad-insts",
                                      "mqsad-pk-insts"))
        opts += ["--hip", "--hip-platform", "AMD", f"--cmake-opt=-DUR_HIP_ROCM_DIR={rocm[0]}",
                 f"--cmake-opt=-DUR_HIP_LIB_DIR={rocm[1]}", "--llvm-external-projects", "device-libs",
                 f"--cmake-opt=-DCLANG_OPTIONS_APPEND={features};-Wno-error=attribute-alias"]
    # (libclc's targets, LLVM runtimes of their own since v7.2.0, are configure.py's for --cuda and --hip)
    return opts


def fetch_device_libs() -> Path:
    """ROCm's llvm-project at DEVICE_LIBS_TAG (a commit), amd/device-libs alone (a shallow, sparse clone: a few MB)."""
    if not (DEVICE_LIBS_SRC / ".git").exists():
        run(["git", "init", "-q", DEVICE_LIBS_SRC])
        run(["git", "-C", DEVICE_LIBS_SRC, "remote", "add", "origin", DEVICE_LIBS_REPO])
        run(["git", "-C", DEVICE_LIBS_SRC, "sparse-checkout", "set", "amd/device-libs"])
    have = subprocess.run(["git", "-C", str(DEVICE_LIBS_SRC), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if have != DEVICE_LIBS_TAG:
        run(["git", "-C", DEVICE_LIBS_SRC, "fetch", "--depth", "1", "--filter=blob:none", "origin", DEVICE_LIBS_TAG])
        run(["git", "-C", DEVICE_LIBS_SRC, "checkout", "--detach", DEVICE_LIBS_TAG])
    return DEVICE_LIBS_SRC / "amd" / "device-libs"


def build_device_libs(jobs: int) -> bool:
    """ROCm's device libraries, built in the toolchain's tree (an LLVM external project, as their README builds them)
    and copied into install/amdgcn/bitcode: those a SYCL program's AMD code links (ocml, ockl, oclc_*) and opencl,
    whose presence the driver asks for, not the HIP language's (hip) and the sanitizer's (asanrtl).  ROCm 7.14's
    opencl has aliases whose types differ from their aliasees', a warning (attribute-alias) its build takes for an
    error, which ROCm's clang does not give: not an error here (target_options).  Apart from compile.py's target, so a
    failure keeps the toolchain:
    CMake then takes a ROCm's device libraries the compiler reads, or says why none (cmake/StrataHip.cmake)."""
    r = subprocess.run(["ninja", "-C", str(BUILD), "-t", "targets", "all"], capture_output=True, text=True)
    libs = sorted({m.group(1) for m in re.finditer(r"^((?:ocml|ockl|oclc_\w+|opencl)):", r.stdout, re.M)})
    cmd = ["cmake", "--build", str(BUILD), "--target", *libs, "-j", str(jobs)]
    say("  > " + " ".join(cmd[:5]) + f" ... ({len(libs)} libraries)")
    if not libs or subprocess.run(cmd).returncode != 0:
        say(f"note: ROCm's device libraries ({DEVICE_LIBS_TAG}) did not build with this toolchain; the engine's "
            "AMD code takes the system's where this compiler reads them")
        return False
    dest = INSTALL / "amdgcn" / "bitcode"
    dest.mkdir(parents=True, exist_ok=True)
    for bc in BUILD.glob("**/amdgcn/bitcode/*.bc"):
        if bc.stem in libs:
            shutil.copy2(bc, dest / bc.name)
    return True


def fix_sources() -> None:
    # the patched files back to the release, then every patch in order
    restore_release()
    for patch in PATCHES:
        r = subprocess.run(["git", "-C", str(SRC), "apply", str(patch)], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"{patch.name} does not apply: {r.stderr.strip()}",
                 "a newer intel/llvm may have it fixed: check third_party/main/intel-llvm/patches")
        say(f"applied {patch.name}")


def prepare_source(tag: str) -> None:
    """The clone at `tag`, with the patches."""
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (SRC / ".git").exists():
        have = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--exact-match"], capture_output=True,
                              text=True).stdout.strip()
        if have != tag:
            run(["git", "-C", SRC, "fetch", "--depth", "1", "origin", "tag", tag])
            # the patched files back to the release first
            restore_release()
            run(["git", "-C", SRC, "checkout", "--detach", tag])
    else:
        run(["git", "clone", "--depth", "1", "--branch", tag, REPO, SRC])
    fix_sources()


def build(tag: str, keep_build: bool, contrib: bool, jobs: int | None = None) -> None:
    need = missing_prerequisites()
    if need:
        fail("intel/llvm's build needs: " + ", ".join(need),
             "install them: " + PREREQUISITES[distro_family()]["how"] + " ".join(need))
    prepare_source(tag)
    # The configuration downloads what its release pins (Level Zero's headers and loader: the runtime adapter needs
    # newer ones than Ubuntu 26.04's 1.28; emhash), all free software; giving it the distribution's failed.  Forced:
    # without pkg-config the adapter takes an installed loader without checking its version (Debian 13's 1.20 failed).
    # The links take the most memory: LLVM limits them by the free RAM at configure time (LLVM_RAM_PER_LINK_JOB, in
    # MB; 10000 is the value llvm/docs/CMake.rst suggests), and every other job runs on all the threads
    # Without assertions, as a distribution's clang is: they check the compiler's own state and do not change the code
    # it makes, and they slow every compile.  lld links LLVM's many large binaries faster, and with less memory, than
    # GNU ld.  Without sycl-jit (libsycl-jit, 157 MB): libsycl opens it only to compile kernels from source, which the
    # engine does not, and the packages never carried it.
    # configure.py asks for the static zstd library (LLVM_USE_STATIC_ZSTD=ON), which Debian's and Fedora's packages of
    # it have and Arch's does not: without one the shared library is linked instead, which the release's own libraries
    # and its SYCL runtime take (intel/llvm#22294, in v7.1.1).  A --cmake-opt comes after configure.py's own -D, so it
    # settles the answer.
    zstd = [] if static_zstd() else ["--cmake-opt=-DLLVM_USE_STATIC_ZSTD=OFF"]
    options = [f"--cmake-opt=-DCMAKE_INSTALL_PREFIX={INSTALL}", "--cmake-opt=-DSYCL_UR_FORCE_FETCH_LEVEL_ZERO=ON",
               "--cmake-opt=-DLLVM_ENABLE_ZSTD=FORCE_ON", "--cmake-opt=-DLLVM_RAM_PER_LINK_JOB=10000",
               "--cmake-opt=-DLLVM_ENABLE_ASSERTIONS=OFF", "--cmake-opt=-DLLVM_USE_LINKER=lld",
               "--disable-jit", "--cmake-opt=-DSYCL_ENABLE_EXTENSION_JIT=OFF"] + \
        zstd + target_options(contrib)
    # configured again when the tree was configured with other values of these options
    cache = BUILD / "CMakeCache.txt"
    text = cache.read_text(errors="replace") if cache.exists() else ""
    wanted = [o.removeprefix("--cmake-opt=-D").split("=", 1) for o in options if o.startswith("--cmake-opt=-D")]
    if not (BUILD / "build.ninja").exists() or \
            not all(any(line.startswith(name + ":") and line.endswith("=" + value) for line in text.splitlines())
                    for name, value in wanted):
        run([sys.executable, SRC / "buildbot" / "configure.py", "-o", BUILD, "-t", "Release"] + options)
    # The deploy writes one marker per component (install_manifest_<component>.txt) and its steps depend on nothing
    # but each other (sycl/CMakeLists.txt), so a marker that survived the build that is being redone makes ninja skip
    # installing what was built since (an older clang, as another build left it) and the toolchain in install/ stops
    # matching the build tree.  A build that runs installs its components again: the markers go.
    for marker in BUILD.glob("install_manifest_*.txt"):
        marker.unlink()
    jobs = jobs or os.cpu_count() or 4
    started = time.time()
    run([sys.executable, SRC / "buildbot" / "compile.py", "-o", BUILD, "-j", str(jobs)])
    # the tools a --windows build runs that the toolchain does not install (libdevice's)
    for tool in HOST_TOOLS_EXTRA:
        if (BUILD / "bin" / tool).exists():
            shutil.copy2(BUILD / "bin" / tool, INSTALL / "bin" / tool)
    device_libs = {"tag": DEVICE_LIBS_TAG, "built": build_device_libs(jobs)} if rocm_dirs() is not None else None
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpus = probe(INSTALL)
    backends = sorted(p.name.split("_adapter_")[1].split(".")[0] for p in (INSTALL / "lib").glob("libur_adapter_*.so"))
    RECORD.write_text(json.dumps({"tag": tag, "commit": commit, "fixes": [patch_id(p) for p in PATCHES],
                                  "patches": patch_fingerprints(), "zstd": True,
                                  "built": time.strftime("%Y-%m-%d %H:%M"),
                                  "minutes": round((time.time() - started) / 60), "backends": backends,
                                  "device_libs": device_libs,
                                  "gpus": gpus}, indent=1))
    if not keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)


# ---- the SYCL runtime for Windows (--windows): cross-built here with clang-cl and lld-link, against the MSVC CRT and
# the Windows SDK from xwin (docs/DEVTOOLS.md#windows-cross-build).  The device libraries are made with the contrib
# toolchain's tools, of the same tag and patches (LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR; its build copies in the three
# it does not install), so no compiler is built for Windows or for this machine: only the parts a Windows engine links
# and runs with (no Debug libraries, no clang.exe).
WIN_OUT = ROOT / ".tools" / "intel-llvm-win"
# the tools libdevice takes from LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR; the toolchain installs all but HOST_TOOLS_EXTRA,
# which build() copies from its build tree
HOST_TOOLS = ("clang", "llvm-ar", "append-file", "clang-offload-bundler", "llvm-offload-binary", "file-table-tform",
              "llvm-foreach", "llvm-spirv", "sycl-post-link", "llvm-link")
HOST_TOOLS_EXTRA = ("opt",)
# NVIDIA's Windows CUDA headers and import libraries (cuda.lib, nvml.lib) the CUDA adapter links: the toolkit's
# redistributable archives (not free software; downloaded, never bundled), of the version the contrib packages build
# for, which every newer driver runs
WIN_CUDA = "12.4.1"
WIN_CUDA_PARTS = ("cuda_cudart", "cuda_nvml_dev", "cuda_cccl")
WIN_CUDA_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
NVCC_STAND_IN = r"""#!/bin/sh
# FindCUDAToolkit's questions about the Windows CUDA tree {root} (headers and import libraries only)
case "$*" in *--version*) echo "Cuda compilation tools, release {version}"; exit 0 ;; esac
echo "#$ _HERE_={root}/bin"
echo "#$ _THERE_={root}/bin"
echo "#$ TOP={root}"
echo "#$ INCLUDES=\"-I{root}/include\""
echo "#$ LIBRARIES=\"-L{root}/lib/x64\""
"""
# AMD's HIP for the Windows HIP adapter (AMD GPUs): the HIP headers of this machine's ROCm (free software; AMD's
# Windows SDK carries the same) and an import library of the HIP 7 runtime AMD's Windows driver puts into System32
# (Adrenalin 26.5.1 and later ship only it, in place of amdhip64_6.dll), made with llvm-dlltool from the names the
# Linux runtime of the same major version exports
WIN_HIP_DLL = "amdhip64_7.dll"
# zstd for the Windows SYCL runtime (BSD-3-Clause; dual-licensed with GPL-2.0, taken under the BSD license), linked into
# sycl9.dll: oneMKL's SYCL DLLs carry zstd-compressed device images, which a runtime without it refuses as they
# register them in their DllMain - mkl_sycl_blas.6.dll then does not load (WinError 1114, XeStrata #59)
WIN_ZSTD = "1.5.7"
WIN_ZSTD_SHA256 = "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"
WIN_ZSTD_URL = f"https://github.com/facebook/zstd/releases/download/v{WIN_ZSTD}/zstd-{WIN_ZSTD}.tar.gz"
# what a Windows engine's SYCL runtime is: the headers and the device libraries, installed by their components, and
# the DLLs and import libraries, copied (their components install the Debug ones too, which are not built); the UR
# loader and the OpenCL adapter are linked into sycl9.dll since v7.2.0 (UR_STATIC_LOADER, UR_STATIC_ADAPTER_OPENCL)
WIN_COMPONENTS = ("sycl-headers", "sycl-headers-extras", "OpenCL-Headers", "ur_headers", "libsycldevice")
WIN_FILES = ("bin/sycl9.dll", "bin/ur_win_proxy_loader.dll",
             "bin/ur_adapter_cuda.dll", "bin/ur_adapter_level_zero.dll", "bin/ur_adapter_level_zero_v2.dll",
             "bin/xptifw.dll", "lib/sycl9.lib", "lib/sycl.lib",
             "lib/sycl-devicelib-host.lib", "lib/xpti.lib", "lib/xptifw.lib")
WIN_TARGETS = ("sycl9.dll", "ur_win_proxy_loader", "ur_adapter_cuda", "ur_adapter_level_zero",
               "ur_adapter_level_zero_v2", "xptifw", "libsycldevice")
# Windows names the build asks for in another case than the SDK's files (Linux's file names are case-sensitive)
WIN_CASE_HEADERS = {"Libloaderapi.h": "um/libloaderapi.h"}
WIN_CASE_LIBS = {"Shlwapi.lib": "um/x86_64/shlwapi.lib"}


def llvm_bin_with(tool: str) -> Path:
    """The newest of the distribution's LLVM folders with `tool` (llvm-lib, llvm-rc, llvm-mt, llvm-ml: Ubuntu's
    llvm-<version>; lld-link: lld-<version>), which intel/llvm does not install."""
    for d in sorted(Path("/usr/lib").glob("llvm-*/bin"), key=lambda p: version_key(p.parent.name), reverse=True):
        if (d / tool).exists():
            return d / tool
    fail(f"no {tool} (LLVM's)", "install it: sudo apt install llvm")


def fetch_win_cuda(dest: Path) -> Path:
    """NVIDIA's Windows CUDA headers and import libraries in `dest`, and an nvcc that answers CMake's FindCUDAToolkit
    (its version and folders: the Windows nvcc.exe does not run here)."""
    if not (dest / "lib" / "x64" / "cuda.lib").exists():
        with urllib.request.urlopen(f"{WIN_CUDA_URL}redistrib_{WIN_CUDA}.json", timeout=60) as r:
            meta = json.loads(r.read())
        for part in WIN_CUDA_PARTS:
            rel = meta[part]["windows-x86_64"]["relative_path"]
            z = OUT / Path(rel).name
            run(["curl", "-fsSL", "-o", z, WIN_CUDA_URL + rel])
            tmp = OUT / "cuda-unpack"
            shutil.rmtree(tmp, ignore_errors=True)
            run(["unzip", "-q", "-o", z, "-d", tmp])
            shutil.copytree(next(tmp.iterdir()), dest, dirs_exist_ok=True)
            shutil.rmtree(tmp)
            z.unlink()
    nvcc = dest / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True, exist_ok=True)
    nvcc.write_text(NVCC_STAND_IN.format(version=".".join(WIN_CUDA.split(".")[:2]), root=dest))
    nvcc.chmod(0o755)
    return dest


def win_hip(dest: Path) -> Path | None:
    """A HIP tree for the Windows HIP adapter in `dest` (include/ and lib/amdhip64.lib), or None when this machine has
    no ROCm HIP of WIN_HIP_DLL's major version."""
    rocm = rocm_dirs()
    major = WIN_HIP_DLL.removesuffix(".dll").rsplit("_", 1)[1]
    if rocm is None or not (rocm[1] / f"libamdhip64.so.{major}").exists():
        return None
    so = rocm[1] / f"libamdhip64.so.{major}"
    # the headers' license (MIT), for the Windows package's notices: the adapter carries their inline code
    lic = [*Path("/usr/share/doc/libamdhip64-dev").glob("copyright"),
           *Path("/usr/share/licenses").glob("rocm-hip*/LICENSE*"), *(rocm[0] / "share" / "doc").glob("hip*/LICENSE*")]
    if not lic:
        fail("ROCm's HIP headers are here, but not their license text", "install ROCm's HIP development package whole")
    inc = dest / "include"
    shutil.rmtree(dest, ignore_errors=True)
    for name in ("hip", "hsa"):
        if (rocm[0] / "include" / name).is_dir():
            shutil.copytree(rocm[0] / "include" / name, inc / name, symlinks=True)   # hsa/hsa: a link to itself
    out = subprocess.run(["nm", "-D", "--defined-only", "--format=posix", str(so)], capture_output=True, text=True,
                         check=True).stdout
    # the C functions, without their symbol versions (hipMalloc@@hip_4.2); the Itanium-mangled C++ ones have other
    # names in the Windows DLL
    names = sorted({f.split()[0].split("@")[0] for f in out.splitlines()
                    if len(f.split()) > 1 and f.split()[1] in ("T", "W") and not f.startswith("_Z")})
    (dest / "lib").mkdir(parents=True)
    shutil.copy2(lic[0], dest / "LICENSE-hip.txt")
    (dest / "amdhip64.def").write_text(f"LIBRARY {WIN_HIP_DLL}\nEXPORTS\n" + "".join(f"  {n}\n" for n in names))
    run([llvm_bin_with("llvm-dlltool"), "-m", "i386:x86-64", "-d", dest / "amdhip64.def", "-l",
         dest / "lib" / "amdhip64.lib"])
    return dest


def win_zstd(dest: Path, toolchain: Path) -> Path:
    """zstd's static library for Windows in `dest` (include/zstd.h, lib/zstd_static.lib, LICENSE), cross-built with
    the cross build's `toolchain`, WIN_ZSTD's release checked against its SHA-256."""
    if (dest / "lib" / "zstd_static.lib").exists() and (dest / "include" / "zstd.h").exists():
        return dest
    tar = OUT / f"zstd-{WIN_ZSTD}.tar.gz"
    run(["curl", "-fsSL", "-o", tar, WIN_ZSTD_URL])
    if hashlib.sha256(tar.read_bytes()).hexdigest() != WIN_ZSTD_SHA256:
        fail(f"zstd {WIN_ZSTD}'s release does not have the SHA-256 this script expects", f"look at {WIN_ZSTD_URL}")
    src, build = OUT / f"zstd-{WIN_ZSTD}", OUT / "zstd-build"
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(build, ignore_errors=True)
    run(["tar", "-xzf", tar, "-C", OUT])
    run(["cmake", "-G", "Ninja", "-S", src / "build" / "cmake", "-B", build, f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={dest}", "-DZSTD_BUILD_SHARED=OFF",
         "-DZSTD_BUILD_STATIC=ON", "-DZSTD_BUILD_PROGRAMS=OFF", "-DZSTD_BUILD_TESTS=OFF", "-DZSTD_LEGACY_SUPPORT=OFF"])
    run(["cmake", "--build", build])
    run(["cmake", "--install", build])
    shutil.copy2(src / "LICENSE", dest / "LICENSE")
    shutil.rmtree(src)
    shutil.rmtree(build)
    tar.unlink()
    return dest


def win_toolchain(xwin: Path, host_bin: Path) -> Path:
    """The CMake toolchain of the cross build: the host toolchain's clang-cl and lld-link, xwin's CRT and SDK."""
    case_inc, case_lib = OUT / "case" / "include", OUT / "case" / "lib"
    for d, links, root in ((case_inc, WIN_CASE_HEADERS, xwin / "sdk" / "include"),
                           (case_lib, WIN_CASE_LIBS, xwin / "sdk" / "lib")):
        d.mkdir(parents=True, exist_ok=True)
        for name, target in links.items():
            (d / name).unlink(missing_ok=True)
            (d / name).symlink_to(root / target)
    inc = [xwin / "crt" / "include", *(xwin / "sdk" / "include" / s for s in ("ucrt", "um", "shared")), case_inc]
    libs = [xwin / "crt" / "lib" / "x86_64", *(xwin / "sdk" / "lib" / s / "x86_64" for s in ("um", "ucrt")), case_lib]
    flags = " ".join(["--target=x86_64-pc-windows-msvc", "-Wno-unused-command-line-argument",
                      *(f"/imsvc{d}" for d in inc)])
    # the distribution's lld-link, which merges the DLLs' manifests itself (with libxml2); intel/llvm's, which it
    # installs where it builds the AMD target, looks for Windows' mt.exe for them and stopped sycl9.dll's link
    dist = [d for d in sorted(Path("/usr/lib").glob("llvm-*/bin/lld-link")) if d.exists()]
    linker = llvm_bin_with("lld-link") if dist else host_bin / "lld-link"
    tc = OUT / "toolchain.cmake"
    tc.write_text(f"""set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_C_COMPILER {host_bin}/clang-cl)
set(CMAKE_CXX_COMPILER {host_bin}/clang-cl)
set(CMAKE_LINKER {linker})
set(CMAKE_AR {llvm_bin_with("llvm-lib")})
set(CMAKE_RC_COMPILER {llvm_bin_with("llvm-rc")})
set(CMAKE_MT {llvm_bin_with("llvm-mt")})
set(CMAKE_ASM_MASM_COMPILER {llvm_bin_with("llvm-ml")})
set(CMAKE_C_FLAGS_INIT "{flags}")
set(CMAKE_CXX_FLAGS_INIT "{flags}")
set(CMAKE_RC_FLAGS_INIT "{" ".join(f"-I{d}" for d in inc)}")
foreach(kind EXE SHARED MODULE)
  set(CMAKE_${{kind}}_LINKER_FLAGS_INIT "{" ".join(f"/libpath:{d}" for d in libs)}")
endforeach()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# the checks' test programs as Release builds: xwin's splat has no Debug CRT unless asked for it
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
""")
    return tc


def build_windows(tag: str, xwin: Path, keep_build: bool, jobs: int | None) -> None:
    host = ROOT / ".tools" / "intel-llvm-contrib" / "install"
    rec = json.loads((host / "XESTRATA.json").read_text()) if (host / "XESTRATA.json").exists() else {}
    if rec.get("tag") != tag or "cuda" not in rec.get("backends", []):
        fail(f"--windows runs the tools of the contrib toolchain of {tag} on this machine, which is not built",
             f"build it first: tools/intel_llvm_build.py --contrib --tag {tag}")
    if rec.get("patches") != patch_fingerprints():
        fail("the contrib toolchain was built with different patches",
             "build it again: tools/intel_llvm_build.py --contrib --rebuild")
    lacking = [t for t in (*HOST_TOOLS, *HOST_TOOLS_EXTRA) if not (host / "bin" / t).exists()]
    if lacking:
        fail(f"the contrib toolchain lacks {', '.join(lacking)} (built before --windows needed them)",
             f"build it again: tools/intel_llvm_build.py --contrib --rebuild --tag {tag}")
    if not (xwin / "crt" / "include").is_dir():
        fail(f"no MSVC CRT and Windows SDK in {xwin}", "xwin's splat: docs/DEVTOOLS.md#windows-cross-build, or --xwin")
    prepare_source(tag)
    cuda = fetch_win_cuda(OUT / "cuda")
    hip = win_hip(OUT / "hip") if "hip" in rec.get("backends", []) else None
    if hip is None:
        say(f"note: no ROCm HIP for {WIN_HIP_DLL} here, or a contrib toolchain without the HIP target: the Windows "
            "runtime without AMD GPUs")
    hb = host / "bin"
    # the device libraries: the host compiler for Windows, with the CRT's and the SDK's headers after its own
    devflags = " ".join(["-nostdlibinc", *(f"-idirafter {d}" for d in (
        xwin / "crt" / "include", *(xwin / "sdk" / "include" / s for s in ("ucrt", "um", "shared"))))])
    s = SRC
    projects = "sycl;llvm-spirv;opencl;xpti;xptifw;libdevice"
    toolchain = win_toolchain(xwin, hb)
    zstd = win_zstd(OUT / "zstd", toolchain)
    options = [
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL}", "-DLLVM_ENABLE_ASSERTIONS=OFF",
        f"-DLLVM_TARGETS_TO_BUILD=X86;SPIRV;NVPTX{';AMDGPU' if hip else ''}",
        "-DLLVM_HOST_TRIPLE=x86_64-pc-windows-msvc",
        # BLAKE3's MASM files: LLVM's ml (llvm-ml) does not assemble them; the C versions are taken
        "-DLLVM_DISABLE_ASSEMBLY_FILES=ON",
        # no libclc: its libspirv for a Windows host (libspirv.l32.*, the Windows long) is the contrib toolchain's,
        # which builds every variant of each device target (LLVM runtimes since v7.2.0)
        f"-DLLVM_EXTERNAL_PROJECTS={projects}", f"-DLLVM_ENABLE_PROJECTS=clang;{projects}",
        f"-DLLVM_EXTERNAL_SYCL_SOURCE_DIR={s / 'sycl'}", f"-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR={s / 'llvm-spirv'}",
        f"-DLLVM_EXTERNAL_XPTI_SOURCE_DIR={s / 'xpti'}", f"-DXPTI_SOURCE_DIR={s / 'xpti'}",
        f"-DLLVM_EXTERNAL_XPTIFW_SOURCE_DIR={s / 'xptifw'}", f"-DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR={s / 'libdevice'}",
        # configure.py --cuda's options for Windows
        "-DLLVM_BUILD_TOOLS=ON", "-DSYCL_ENABLE_WERROR=OFF", "-DSYCL_INCLUDE_TESTS=OFF", "-DLLVM_ENABLE_DOXYGEN=OFF",
        "-DLLVM_ENABLE_SPHINX=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DSYCL_ENABLE_XPTI_TRACING=ON",
        "-DLLVM_ENABLE_LLD=OFF", "-DLLVM_SPIRV_ENABLE_LIBSPIRV_DIS=OFF", "-DXPTI_ENABLE_WERROR=OFF",
        f"-DSYCL_ENABLE_BACKENDS=level_zero;level_zero_v2;cuda;opencl{';hip' if hip else ''}",
        # no sycl-jit (the runtime compiler, kernel_compiler's), which the engine does not use: it would build most of
        # LLVM's code generation and clang for Windows
        "-DSYCL_ENABLE_EXTENSION_JIT=OFF",
        "-DSYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB=OFF",
        "-DSYCL_UR_FORCE_FETCH_LEVEL_ZERO=ON", f"-DCUDAToolkit_ROOT={cuda}", "-DLLVM_ENABLE_DIA_SDK=OFF",
        "-DLLVM_ENABLE_ZSTD=FORCE_ON", "-DLLVM_USE_STATIC_ZSTD=ON", f"-Dzstd_INCLUDE_DIR={zstd / 'include'}",
        f"-Dzstd_LIBRARY={zstd / 'lib' / 'zstd_static.lib'}",
        f"-Dzstd_STATIC_LIBRARY={zstd / 'lib' / 'zstd_static.lib'}",
        # the device libraries' tools, this machine's (patch build-libdevice-custom-tools); each also by its own
        # variable, else LLVM's get_host_tool_path sets its *_target in the cache to a build of it for this machine
        f"-DLIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR={hb}", f"-DSYCL_LIBDEVICE_CXX_FLAGS={devflags}",
        *(f"-D{tool.upper().replace('-', '_')}={hb / tool}" for tool in (*HOST_TOOLS, *HOST_TOOLS_EXTRA)),
        # build/NATIVE, LLVM's tree for this machine's tablegens: configured with these too; LLVM's cross build hands
        # it PYTHON_EXECUTABLE (empty unless given) over its own
        f"-DPYTHON_EXECUTABLE={sys.executable}", "-DCROSS_TOOLCHAIN_FLAGS_NATIVE=-DSYCL_ENABLE_EXTENSION_JIT=OFF",
    ]
    if hip:
        options += ["-DUR_HIP_PLATFORM=AMD", f"-DUR_HIP_ROCM_DIR={hip}", f"-DUR_HIP_INCLUDE_DIR={hip / 'include'}",
                    f"-DUR_HIP_HSA_INCLUDE_DIRS={hip / 'include'}", f"-DUR_HIP_LIB_DIR={hip / 'lib'}"]
    # llvm-rc preprocesses the resource scripts with the clang it finds on PATH: the host toolchain's
    env = dict(os.environ, PATH=f"{hb}{os.pathsep}{os.environ.get('PATH', '')}")
    started = time.time()
    # configured every time, so a kept build tree takes the options a change of this script gives (zstd)
    run(["cmake", "-G", "Ninja", "-S", s / "llvm", "-B", BUILD, *options], env=env)
    targets = [*WIN_TARGETS, *(["ur_adapter_hip"] if hip else [])]
    run(["cmake", "--build", BUILD, "--target", *targets, "--", "-j", str(jobs or os.cpu_count() or 4)], env=env)
    shutil.rmtree(INSTALL, ignore_errors=True)
    for component in WIN_COMPONENTS:
        run(["cmake", "--install", BUILD, "--component", component])
    for rel in (*WIN_FILES, *(["bin/ur_adapter_hip.dll"] if hip else [])):
        (INSTALL / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUILD / rel, INSTALL / rel)
    shutil.copytree(s / "xpti" / "include" / "xpti", INSTALL / "include" / "xpti", dirs_exist_ok=True)
    # Level Zero's headers (MIT, the loader's the build fetched), for the engine's calls into the loader the SYCL
    # runtime loads (the copy engine's queue, the device's status: src/core/device.cpp); no component installs them
    shutil.copytree(BUILD / "_deps" / "level-zero-loader-src" / "include" / "level_zero",
                    INSTALL / "include" / "level_zero", dirs_exist_ok=True)
    # the license texts of what the DLLs carry: intel/llvm's, and those of the projects its build fetched (UMF and the
    # hwloc it links statically, Level Zero's loader and headers, emhash, parallel-hashmap, OpenCL's headers and ICD
    # loader, vc-intrinsics), for the Windows package's notices
    lic = INSTALL / "licenses"
    lic.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s / "llvm" / "LICENSE.TXT", lic / "intel-llvm.txt")
    shutil.copy2(zstd / "LICENSE", lic / "zstd-LICENSE")
    if hip:
        shutil.copy2(hip / "LICENSE-hip.txt", lic / "hip-headers.txt")
    for dep in sorted((BUILD / "_deps").glob("*-src")):
        for f in sorted(dep.iterdir()):
            if f.is_file() and f.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                shutil.copy2(f, lic / f"{dep.name[:-4]}-{f.name}")
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    RECORD.write_text(json.dumps({"tag": tag, "commit": commit, "fixes": [patch_id(p) for p in PATCHES],
                                  "patches": patch_fingerprints(), "target": "x86_64-pc-windows-msvc",
                                  "cuda": WIN_CUDA, "zstd": WIN_ZSTD,
                                  "hip": WIN_HIP_DLL if hip else None,
                                  "built": time.strftime("%Y-%m-%d %H:%M"),
                                  "minutes": round((time.time() - started) / 60)}, indent=1))
    if not keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)


def version_key(tag: str) -> tuple:
    return tuple(int(x) for x in tag.lstrip("v").split(".") if x.isdigit())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrib", action="store_true",
                    help="with the CUDA target too (NVIDIA GPUs, needs NVIDIA's CUDA toolkit), into "
                         ".tools/intel-llvm-contrib")
    ap.add_argument("--windows", action="store_true",
                    help="the SYCL runtime for Windows (headers, import libraries and DLLs, for a Windows engine built "
                         "here), cross-built with the --contrib toolchain's tools, into .tools/intel-llvm-win")
    ap.add_argument("--xwin", type=Path, default=ROOT / ".tools" / "xwin" / "sdk",
                    help="with --windows: xwin's splat of the MSVC CRT and the Windows SDK (default .tools/xwin/sdk)")
    ap.add_argument("--tag", default=TAG, help=f"the intel/llvm release to build (default {TAG})")
    ap.add_argument("--rebuild", action="store_true", help="build again even when a finished build is there")
    ap.add_argument("--keep-build", action="store_true", help="keep the build tree for the next update")
    ap.add_argument("--yes", action="store_true", help="answer the questions with their defaults")
    ap.add_argument("--jobs", type=int, metavar="N",
                    help="compile N files at once (default: the threads; the links as many as the free RAM allows)")
    a = ap.parse_args()
    if a.windows:
        set_out(WIN_OUT)
        have = json.loads(RECORD.read_text()) if RECORD.exists() else {}
        # an install from before zstd and Level Zero's headers were in it is built again
        if a.rebuild or have.get("tag") != a.tag or have.get("patches") != patch_fingerprints() or \
                have.get("zstd") != WIN_ZSTD or not (INSTALL / "include" / "level_zero" / "ze_api.h").exists():
            say(f"building intel/llvm {a.tag}'s SYCL runtime for Windows in {OUT} ...")
            build_windows(a.tag, a.xwin.resolve(), a.keep_build, a.jobs)
        else:
            say(f"intel/llvm {a.tag}'s SYCL runtime for Windows is already built in {INSTALL}")
        say(str(INSTALL))
        return
    if a.contrib:
        set_out(ROOT / ".tools" / "intel-llvm-contrib")
    have = finished()
    if have and not a.rebuild:
        if have.get("tag") == a.tag:
            # A changed name, order or byte of any patch, including its description, changes the build input.
            # Old records have no hashes, so they ask for one rebuild too.
            hip = "hip" in have.get("backends", []) or rocm_dirs() is not None
            missing = [] if have.get("patches") == patch_fingerprints() else ["the current patches"]
            # an AMD GPU's code needs the HIP target, which the build gets only when ROCm's HIP is there while it is
            # built: a record whose backends have none was built before ROCm was installed (and its deploy may be
            # incomplete, as an interrupted one leaves the record of the build before it)
            if rocm_dirs() is not None and "hip" not in have.get("backends", []):
                missing.append("the HIP target")
            if not have.get("zstd"):
                missing.append("zstd")
            if hip and (have.get("device_libs") or {}).get("tag") != DEVICE_LIBS_TAG:
                missing.append(f"ROCm's device libraries {DEVICE_LIBS_TAG}")
            if not missing:
                say(f"intel/llvm {a.tag} is already built in {INSTALL}")
            elif ask(f"intel/llvm {a.tag} is built without: {', '.join(missing)}; build it again with them? "
                     "build = with them (with --keep-build's tree, minutes), keep = as it is", ["build", "keep"],
                     "build", a.yes) == "build":
                build(a.tag, a.keep_build, a.contrib, a.jobs)
        elif version_key(have.get("tag", "")) < version_key(a.tag):
            choice = ask(f"intel/llvm {have.get('tag')} is built; build {a.tag} over it? "
                         "keep = use the one built, build = build the new one", ["keep", "build"], "keep", a.yes)
            if choice == "build":
                build(a.tag, a.keep_build, a.contrib, a.jobs)
        else:
            say(f"intel/llvm {have.get('tag')} is built (newer than {a.tag}): using it")
    else:
        if (BUILD / "build.ninja").exists() and not a.rebuild:
            say("an interrupted build is there: continuing it")
        say(f"building intel/llvm {a.tag} in {OUT} (13 minutes on 28 threads; with --contrib longer) ...")
        build(a.tag, a.keep_build, a.contrib, a.jobs)
    have = finished()
    if have is None:
        fail(f"no finished build in {INSTALL}")
    if not have.get("gpus"):
        say("note: this build's SYCL runtime lists no Intel GPU (xmx_probe, which looks at Level Zero only): on a PC "
            "with one, its Level Zero driver (Intel's compute-runtime) is missing or too old for it")
    elif not any(g.get("fp16") == "1" and g.get("bf16") == "1" for g in have.get("gpus", [])):
        say("note: no GPU reports FP16 and BF16 matrix engines to this build (xmx_probe); the engine will run its "
            "products without XMX")
    say(str(INSTALL))


if __name__ == "__main__":
    main()
