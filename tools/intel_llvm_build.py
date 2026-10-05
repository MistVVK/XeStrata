#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build intel/llvm's DPC++ (free software: Apache-2.0 with LLVM exceptions) from source into .tools/intel-llvm/, for a
free build of the engine where the distribution has no DPC++, or one whose SYCL runtime does not give the GPU its
matrix engines (Ubuntu 26.04's 6.2 lists none for the Arc Pro B70; intel/llvm added it in v7.0.0).

    tools/intel_llvm_build.py [--contrib] [--rebuild] [--keep-build] [--tag TAG] [--yes]

.tools/intel-llvm/src is the clone, build/ the build tree, install/ the toolchain (bin/clang++, lib/libsycl.so).
--contrib builds the same clone with the CUDA target (NVIDIA GPUs; it needs NVIDIA's CUDA toolkit, which is not free
software: the contrib build mode) and, where ROCm's HIP is installed, the HIP target (AMD GPUs), into
.tools/intel-llvm-contrib/ (build/, install/).  A
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
SRC = BASE / "src"
# where the build goes: BASE, or the contrib one's folder (set_out)
OUT, BUILD, INSTALL, RECORD = BASE, BASE / "build", BASE / "install", BASE / "install" / "XESTRATA.json"

REPO = "https://github.com/intel/llvm.git"
# The release this script builds: one with the Arc Pro B70 in the runtime's matrix table (v7.0.0 or later).  Raised by
# a change to this line, not by following the newest release, so a run does not start a build by itself.
TAG = "v7.1.1"

# Fixes the release lacks (its sycl branch too, on 2026-10-05), applied to the clone before the configuration: an id,
# the file, the text it replaces and the replacement.  A finished install records the ids it was built with.
SOURCE_FIXES = [
    # the CUDA and HIP adapters copied the command-buffer's map of sync points for every node they added, so finalizing
    # a SYCL graph took the square of its node count (2600 kernels: 90 ms on an RTX 4070; Level Zero 2 ms)
    (f"{a}-sync-points", f"unified-runtime/source/adapters/{a}/command_buffer.cpp",
     "  auto SyncPoints = CommandBuffer->SyncPoints;\n", "  const auto &SyncPoints = CommandBuffer->SyncPoints;\n")
    for a in ("cuda", "hip")
] + [
    # the SYCL runtime gave every NVIDIA GPU the first NVIDIA image it found, whatever its architecture: with code for
    # several architectures (STRATA_CUDA_ARCHS), a GPU older than that image failed and a newer one ran older code.
    # The runtime now hands the CUDA adapter the image itself, as it does the HIP one, and the adapter takes the PTX
    # with the highest .target the device runs and refuses those for a newer one.
    ("cuda-select-binary", "unified-runtime/source/adapters/cuda/device.cpp", """\
UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t /*hDevice*/, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {

  // Look for an image for the NVPTX64 target, and return the first one that is
  // found
  for (uint32_t i = 0; i < NumBinaries; i++) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_NVPTX64) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }
""", """\
// The architecture a PTX image is for, from its ".target sm_XY" line: XY
// (sm_120: 120), with the suffix a or f in Suffix; -1 where the image is not
// PTX text.
static int ptxTargetSm(const unsigned char *Data, size_t Size, char &Suffix) {
  static const char Key[] = ".target sm_";
  const size_t KeyLen = sizeof(Key) - 1, End = Size < 65536 ? Size : 65536;
  for (size_t i = 0; i + KeyLen < End; i++) {
    if (memcmp(Data + i, Key, KeyLen) != 0)
      continue;
    size_t j = i + KeyLen;
    int Sm = -1;
    for (; j < End && Data[j] >= '0' && Data[j] <= '9'; j++)
      Sm = (Sm < 0 ? 0 : Sm * 10) + (Data[j] - '0');
    Suffix = j < End && (Data[j] == 'a' || Data[j] == 'f') ? Data[j] : 0;
    return Sm;
  }
  return -1;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {

  // Look for an image for the NVPTX64 target.  Where the SYCL runtime passes
  // the image itself (a {pointer, size} pair in pNext), take the PTX with the
  // highest .target the device runs (sm_XYa: that architecture only, sm_XYf:
  // its family) and refuse one for a newer architecture; otherwise the first
  // image found.
  int Major = 0, Minor = 0;
  UR_CHECK_ERROR(cuDeviceGetAttribute(
      &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));
  UR_CHECK_ERROR(cuDeviceGetAttribute(
      &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice->get()));
  const int DeviceSm = Major * 10 + Minor;
  int Best = -1, BestSm = -1, Unknown = -1;
  for (uint32_t i = 0; i < NumBinaries; i++) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_NVPTX64) != 0)
      continue;
    using BinaryBlobTy = std::pair<const unsigned char *, size_t>;
    const auto *Blob = static_cast<const BinaryBlobTy *>(pBinaries[i].pNext);
    char Suffix = 0;
    const int Sm = Blob ? ptxTargetSm(Blob->first, Blob->second, Suffix) : -1;
    if (Sm < 0) {
      if (Unknown < 0)
        Unknown = (int)i;
      continue;
    }
    const bool Runs = Suffix == 'a'   ? Sm == DeviceSm
                      : Suffix == 'f' ? Sm / 10 == Major && Sm <= DeviceSm
                                      : Sm <= DeviceSm;
    if (Runs && Sm > BestSm) {
      Best = (int)i;
      BestSm = Sm;
    }
  }
  if (Best >= 0 || Unknown >= 0) {
    *pSelectedBinary = (uint32_t)(Best >= 0 ? Best : Unknown);
    return UR_RESULT_SUCCESS;
  }
"""),
    ("cuda-binary-to-adapter", "sycl/source/detail/program_manager/program_manager.cpp", """\
  ur_device_binary_t UrBinary{};
  UrBinary.pDeviceTargetSpec = getUrDeviceTarget(DevBin.DeviceTargetSpec);
""", """\
  ur_device_binary_t UrBinary{};
  UrBinary.pDeviceTargetSpec = getUrDeviceTarget(DevBin.DeviceTargetSpec);
  // the CUDA adapter refuses PTX for a newer architecture than the device's
  std::pair<const unsigned char *, size_t> Blob{
      DevBin.BinaryStart, std::distance(DevBin.BinaryStart, DevBin.BinaryEnd)};
  if (DeviceImpl.getBackend() == backend::ext_oneapi_cuda)
    UrBinary.pNext = &Blob;
"""),
    ("cuda-binaries-to-adapter", "sycl/source/detail/program_manager/program_manager.cpp", """\
  // Pass extra information to the HIP adapter to aid in binary selection. We
  // pass it the raw binary as a {ptr, length} pair.
  std::vector<std::pair<const unsigned char *, size_t>> UrBinariesStorage;
  if (DeviceImpl.getBackend() == backend::ext_oneapi_hip)
    UrBinariesStorage.reserve(NumImgs);
""", """\
  // Pass extra information to the HIP and CUDA adapters to aid in binary
  // selection. We pass it the raw binary as a {ptr, length} pair.
  std::vector<std::pair<const unsigned char *, size_t>> UrBinariesStorage;
  const bool PassBinaries = DeviceImpl.getBackend() == backend::ext_oneapi_hip ||
                            DeviceImpl.getBackend() == backend::ext_oneapi_cuda;
  if (PassBinaries)
    UrBinariesStorage.reserve(NumImgs);
"""),
    # the CUDA adapter had urUSMImportExp and urUSMReleaseExp (sycl_ext_oneapi_copy_optimize) doing nothing and left
    # them out of its table, so the loader refused them (UR_RESULT_ERROR_UNINITIALIZED); now they page-lock the memory
    # with cuMemHostRegister, so copies from it are DMA transfers that overlap the GPU's work
    ("cuda-host-register", "unified-runtime/source/adapters/cuda/usm.cpp", """\
UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t, void *,
                                                   size_t Size) {
  UR_ASSERT(Size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t,
                                                    void *) {
  return UR_RESULT_SUCCESS;
}
""", """\
UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t hContext,
                                                   void *pMem, size_t Size) {
  UR_ASSERT(Size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  // page-locked for every context, so copies from and to it are DMA transfers
  try {
    ScopedContext Active(hContext->getDevices()[0]);
    const CUresult Result =
        cuMemHostRegister(pMem, Size, CU_MEMHOSTREGISTER_PORTABLE);
    if (Result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
      UR_CHECK_ERROR(Result);
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t hContext,
                                                    void *pMem) {
  try {
    ScopedContext Active(hContext->getDevices()[0]);
    const CUresult Result = cuMemHostUnregister(pMem);
    if (Result != CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
      UR_CHECK_ERROR(Result);
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}
"""),
    ("cuda-host-register-table", "unified-runtime/source/adapters/cuda/ur_interface_loader.cpp", """\
  pDdiTable->pfnContextMemcpyExp = urUSMContextMemcpyExp;
  return UR_RESULT_SUCCESS;
""", """\
  pDdiTable->pfnContextMemcpyExp = urUSMContextMemcpyExp;
  pDdiTable->pfnImportExp = urUSMImportExp;
  pDdiTable->pfnReleaseExp = urUSMReleaseExp;
  return UR_RESULT_SUCCESS;
"""),
    ("cuda-binaries-to-adapter-2", "sycl/source/detail/program_manager/program_manager.cpp", """\
    if (DeviceImpl.getBackend() == backend::ext_oneapi_hip) {
      UrBinariesStorage.emplace_back(
""", """\
    if (PassBinaries) {
      UrBinariesStorage.emplace_back(
"""),
]


def set_out(out: Path) -> None:
    global OUT, BUILD, INSTALL, RECORD
    OUT, BUILD, INSTALL = out, out / "build", out / "install"
    RECORD = INSTALL / "XESTRATA.json"


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
    # the SYCL runtime reads zstd-compressed device images (oneMKL's, which oneMath calls in the contrib modes); without
    # the headers the configuration leaves zstd out silently
    if not Path("/usr/include/zstd.h").exists():
        need.append("libzstd-dev")
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
    """ROCm's HIP for the HIP target: AMD's own tree (/opt/rocm) or the distribution's (Debian and Ubuntu put the
    headers in /usr/include, the libraries in the multiarch folder)."""
    if Path("/opt/rocm/include/hip/hip_runtime_api.h").exists():
        return Path("/opt/rocm"), Path("/opt/rocm/lib")
    if Path("/usr/include/hip/hip_runtime_api.h").exists():
        for lib in sorted(Path("/usr/lib").glob("*/libamdhip64.so")):
            return Path("/usr"), lib.parent
    return None


def contrib_options() -> list:
    """configure.py's options for the contrib toolchain: CUDA, and HIP where ROCm is installed."""
    if shutil.which("nvcc") is None and not Path("/usr/local/cuda/include/cuda.h").exists():
        fail("the CUDA target needs NVIDIA's CUDA toolkit (not free software)",
             "install it: sudo apt install nvidia-cuda-toolkit (Ubuntu: multiverse; Debian: non-free)")
    opts = ["--cuda"]
    libclc = ["nvptx64--nvidiacl"]
    rocm = rocm_dirs()
    if rocm is None:
        say("note: no ROCm HIP (hipcc's headers, libamdhip64): building without the HIP target (AMD GPUs)")
    else:
        opts += ["--hip", "--hip-platform", "AMD", f"--cmake-opt=-DUR_HIP_ROCM_DIR={rocm[0]}",
                 f"--cmake-opt=-DUR_HIP_LIB_DIR={rocm[1]}"]
        # v7.1.1's configure.py names the AMD target amdgcn--amdhsa, which libclc refuses (amdgcn-amd-amdhsa)
        libclc.append("amdgcn-amd-amdhsa")
    return opts + [f"--cmake-opt=-DLIBCLC_TARGETS_TO_BUILD={';'.join(libclc)}"]


def fix_sources() -> None:
    for _, rel, old, new in SOURCE_FIXES:
        path = SRC / rel
        text = path.read_text()
        if new in text:
            continue
        if old not in text:
            fail(f"{rel}: the text a fix replaces is not there",
                 "a newer intel/llvm may have it fixed: check SOURCE_FIXES")
        path.write_text(text.replace(old, new, 1))
        say(f"fixed {rel}")


def build(tag: str, keep_build: bool, contrib: bool) -> None:
    need = missing_prerequisites()
    if need:
        fail("intel/llvm's build needs: " + ", ".join(need), "install them: sudo apt install " + " ".join(need))
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (SRC / ".git").exists():
        have = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--exact-match"], capture_output=True,
                              text=True).stdout.strip()
        if have != tag:
            run(["git", "-C", SRC, "fetch", "--depth", "1", "origin", "tag", tag])
            # the fixed files back to the release first
            run(["git", "-C", SRC, "checkout", "--", *sorted({rel for _, rel, _, _ in SOURCE_FIXES})])
            run(["git", "-C", SRC, "checkout", "--detach", tag])
    else:
        run(["git", "clone", "--depth", "1", "--branch", tag, REPO, SRC])
    fix_sources()
    # The configuration downloads what its release pins (Level Zero's headers and loader: the runtime adapter needs
    # newer ones than Ubuntu 26.04's 1.28; emhash), all free software; giving it the distribution's failed.  Forced:
    # without pkg-config the adapter takes an installed loader without checking its version (Debian 13's 1.20 failed)
    cache = BUILD / "CMakeCache.txt"
    if not (BUILD / "build.ninja").exists() or not cache.exists() or \
            "SYCL_UR_FORCE_FETCH_LEVEL_ZERO:BOOL=ON" not in cache.read_text(errors="replace") or \
            "LLVM_ENABLE_ZSTD:STRING=FORCE_ON" not in cache.read_text(errors="replace"):
        run([sys.executable, SRC / "buildbot" / "configure.py", "-o", BUILD, "-t", "Release",
             f"--cmake-opt=-DCMAKE_INSTALL_PREFIX={INSTALL}", "--cmake-opt=-DSYCL_UR_FORCE_FETCH_LEVEL_ZERO=ON",
             "--cmake-opt=-DLLVM_ENABLE_ZSTD=FORCE_ON"]
            + (contrib_options() if contrib else []))
    jobs = max(2, min(os.cpu_count() or 4, int(ram_gb() // 3)))
    started = time.time()
    run([sys.executable, SRC / "buildbot" / "compile.py", "-o", BUILD, "-j", str(jobs)])
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpus = probe(INSTALL)
    backends = sorted(p.name.split("_adapter_")[1].split(".")[0] for p in (INSTALL / "lib").glob("libur_adapter_*.so"))
    RECORD.write_text(json.dumps({"tag": tag, "commit": commit, "fixes": [f[0] for f in SOURCE_FIXES], "zstd": True,
                                  "built": time.strftime("%Y-%m-%d %H:%M"),
                                  "minutes": round((time.time() - started) / 60), "backends": backends,
                                  "gpus": gpus}, indent=1))
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
    ap.add_argument("--contrib", action="store_true",
                    help="with the CUDA target (NVIDIA GPUs, needs NVIDIA's CUDA toolkit) and, where ROCm is "
                         "installed, HIP (AMD GPUs), into .tools/intel-llvm-contrib")
    ap.add_argument("--tag", default=TAG, help=f"the intel/llvm release to build (default {TAG})")
    ap.add_argument("--rebuild", action="store_true", help="build again even when a finished build is there")
    ap.add_argument("--keep-build", action="store_true", help="keep the build tree for the next update")
    ap.add_argument("--yes", action="store_true", help="answer the questions with their defaults")
    a = ap.parse_args()
    if a.contrib:
        set_out(ROOT / ".tools" / "intel-llvm-contrib")
    have = finished()
    if have and not a.rebuild:
        if have.get("tag") == a.tag:
            # the CUDA and HIP adapters' fixes do not ask the free toolchain, which has neither, to be built again
            missing = [f[0] for f in SOURCE_FIXES if f[0] not in have.get("fixes", [])
                       and (a.contrib or not f[0].startswith(("cuda-", "hip-")))]
            if not have.get("zstd"):
                missing.append("zstd")
            if not missing:
                say(f"intel/llvm {a.tag} is already built in {INSTALL}")
            elif ask(f"intel/llvm {a.tag} is built without the fixes {', '.join(missing)}; build it again with them? "
                     "build = with the fixes (with --keep-build's tree, minutes), keep = as it is", ["build", "keep"],
                     "build", a.yes) == "build":
                build(a.tag, a.keep_build, a.contrib)
        elif version_key(have.get("tag", "")) < version_key(a.tag):
            choice = ask(f"intel/llvm {have.get('tag')} is built; build {a.tag} over it? "
                         "keep = use the one built, build = build the new one", ["keep", "build"], "keep", a.yes)
            if choice == "build":
                build(a.tag, a.keep_build, a.contrib)
        else:
            say(f"intel/llvm {have.get('tag')} is built (newer than {a.tag}): using it")
    else:
        if (BUILD / "build.ninja").exists() and not a.rebuild:
            say("an interrupted build is there: continuing it")
        say(f"building intel/llvm {a.tag} in {OUT} (13 minutes on 28 threads; with --contrib longer) ...")
        build(a.tag, a.keep_build, a.contrib)
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
