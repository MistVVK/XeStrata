#!/usr/bin/env python3
"""Build the Windows NVIDIA engine package (the cuda13 and cuda12 engine zips of the release feed).

    py -3 packaging\\build_cuda_windows.py cuda13     CUDA 13.0: sm_75 / 86 / 89 / 120 machine code + compute_120 PTX
    py -3 packaging\\build_cuda_windows.py cuda12     CUDA 12.9: sm_60 / 61 / 70 / 75 / 80 / 86 / 89 + compute_89 PTX, -DSTRATA_EXPERIMENTAL_SM60=ON

Run it on a Windows PC with Visual Studio 2022 Build Tools (C++ workload), CMake + Ninja, Python 3.10+ and the CUDA toolkit of the variant
(CUDA_PATH_V13_0 / CUDA_PATH_V12_9 as its installer sets them, or CUDA_PATH pointing at it). ggml and the picture reader's sources are the
vendored third_party trees: nothing is downloaded. The package lands in OUT (default <repo>\\out-engine-win-<variant>). Nothing is uploaded.

The steps are the release's: the engine configure line below, `strata` (+ `strata_kq` when the build tree has it), the picture reader for the
same architectures (tools\\packaging\\build_vision_windows.bat), BUILD.json, then tools/packaging/package_engine.py (which copies the CUDA DLLs strata.exe
imports from the toolkit), --verify and binary_deps.py.

Settings (environment, all optional):
  STRATA_CMAKE_EXTRA  more engine configure arguments (they come last on the configure line)
  BUILD / BUILD_VISION_DIR / OUT   the engine build folder, the reader's build folder, the package folder
  ARCHS               the architecture list instead of the variant's (semicolons, e.g. 120; the last one also gets PTX), as in build_cuda_linux.sh
  BUILD_JOBS          compile jobs (default 6: the CUDA compiler takes several GB of RAM per job)
  COMMIT              the source commit for the build id (default: git rev-parse HEAD; required in a tree without .git)
  STRATA_GGML_DIR     a llama.cpp checkout to take ggml (and the reader's sources) from instead (an override, kept for two releases)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE_SRC = REPO / "engine"                      # the engine's sources: CMakeLists.txt is here since the 0.2.16 layout (the repository root has none)
PACKAGING = REPO / "tools" / "packaging"          # the scripts this one runs: build_vision_windows.bat, package_engine.py, binary_deps.py
VARIANTS = {
    # variant: (CUDA version, archs, min driver, extra configure arguments)
    "cuda13": ("13.0", (75, 86, 89, 120), "580.65", []),
    # cuda12: 120 before 89, so the PTX (for the last arch) stays compute_89 while a Blackwell card gets machine code (#5784)
    "cuda12": ("12.9", (60, 61, 70, 75, 80, 86, 120, 89), "576.02", ["-DSTRATA_EXPERIMENTAL_SM60=ON"]),
}
VCVARS = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / r"Microsoft Visual Studio\2022\BuildTools",
          Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / r"Microsoft Visual Studio\2022\Community",
          Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / r"Microsoft Visual Studio\2022\BuildTools"]


def fail(msg: str) -> None:
    raise SystemExit("build_cuda_windows.py: " + msg)


def cuda_arch_list(archs) -> str:
    """Every arch as machine code and PTX for the newest one: '75-real;86-real;89-real;120-real;120-virtual'."""
    return ";".join(["%d-real" % a for a in archs] + ["%d-virtual" % archs[-1]])


def parse_archs(text: str) -> tuple[int, ...]:
    """'75;86;89;120' (or commas / spaces) -> (75, 86, 89, 120); the last one also gets PTX."""
    try:
        archs = tuple(int(x) for x in re.split(r"[;,\s]+", text.strip()) if x)
    except ValueError:
        archs = ()
    if not archs:
        fail("ARCHS=%r is not a list of numbers like 75;86;89;120" % text)
    return archs


def configure_args(src: Path, build: Path, variant: str, cuda: Path, extra: list[str], ggml_dir: str = "",
                   archs: tuple[int, ...] | None = None) -> list[str]:
    """The release's engine configure line (CMake defaults for everything it does not name)."""
    cu = cuda.as_posix()
    args = ["-G", "Ninja", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON",
            "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_PORTABLE=ON", *VARIANTS[variant][3],
            "-DCMAKE_CUDA_ARCHITECTURES=" + cuda_arch_list(archs or VARIANTS[variant][1]), "-DCMAKE_CUDA_COMPILER=%s/bin/nvcc.exe" % cu,
            "-DCUDAToolkit_ROOT=" + cu]
    if ggml_dir:
        args.append("-DSTRATA_GGML_DIR=" + ggml_dir.replace("\\", "/"))
    return args + extra


def engine_targets(cmake: str, build: Path, env=None) -> list[str]:
    """['strata', 'strata_kq'] when the build tree has the strata_kq target (a build with llama.cpp's MMQ), ['strata'] when ninja names it
    unknown (STRATA_MMQ=OFF, the default). The tree is asked with a dry run, as packaging/engine_targets.sh does; any other failure stops the build."""
    r = subprocess.run([cmake, "--build", str(build), "--target", "strata_kq", "--", "-n"], capture_output=True, text=True, env=env,
                       errors="replace")
    if r.returncode == 0:
        return ["strata", "strata_kq"]
    text = (r.stdout or "") + (r.stderr or "")
    if "unknown target 'strata_kq'" in text or re.search(r"No rule to make target.*strata_kq", text):
        return ["strata"]
    sys.stderr.write(text + "\n")
    fail("cannot tell whether the build tree %s has the strata_kq target" % build)
    return []


def find_cuda(variant: str) -> Path:
    cu = VARIANTS[variant][0]
    for var in ("CUDA_PATH_V" + cu.replace(".", "_"), "CUDA_PATH"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc.exe").is_file():
            out = subprocess.run([str(Path(root) / "bin" / "nvcc.exe"), "--version"], capture_output=True, text=True).stdout
            m = re.search(r"release (\d+\.\d+)", out)
            if m and m.group(1) == cu:
                return Path(root)
    fail("%s needs the CUDA %s toolkit (CUDA_PATH_V%s or CUDA_PATH)" % (variant, cu, cu.replace(".", "_")))
    return Path()


def msvc_env() -> dict:
    """The environment of an x64 Native Tools prompt (vcvars64.bat), or this one when cl.exe is already on PATH."""
    env = {k.upper(): v for k, v in os.environ.items()}      # Windows names are case-insensitive: one key each (PATH, not Path and PATH)
    if shutil.which("cl", path=env.get("PATH")):
        return env
    for vs in VCVARS:
        bat = vs / r"VC\Auxiliary\Build\vcvars64.bat"
        if bat.is_file():
            r = subprocess.run('cmd /d /s /c ""%s" >nul && set"' % bat, capture_output=True, text=True, errors="replace")
            for line in r.stdout.splitlines():
                k, sep, v = line.partition("=")
                if sep and k:
                    env[k.upper()] = v
            if shutil.which("cl", path=env.get("PATH")):
                return env
    fail("cl.exe not found: install Visual Studio 2022 Build Tools with the C++ workload, or run this from an x64 Native Tools prompt")
    return env


def project_version(src: Path) -> str:
    m = re.search(r"project\([^)]*VERSION\s+([0-9][0-9.]*)", (src / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        fail("no project VERSION in CMakeLists.txt")
    return m.group(1)


def source_commit(src: Path) -> str:
    commit = os.environ.get("COMMIT", "")
    if not commit:
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{12,40}", commit):
        fail("set COMMIT=<hex> (no git commit found here)")
    return commit


def run(cmd: list[str], env: dict) -> None:
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    if subprocess.run(cmd, env=env).returncode != 0:
        fail("failed: " + cmd[0] + " " + " ".join(cmd[1:3]))


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in VARIANTS:
        print(__doc__)
        return 2
    variant = argv[0]
    if os.name != "nt":
        fail("this builds the Windows engine; on Linux use tools/packaging/build_cuda_linux.sh")
    cu, archs, mindriver, _ = VARIANTS[variant]
    if os.environ.get("ARCHS"):
        archs = parse_archs(os.environ["ARCHS"])
    src = REPO
    build = Path(os.environ.get("BUILD") or src / ("build-win-" + variant))
    vision_dir = Path(os.environ.get("BUILD_VISION_DIR") or src / ("build-vision-%s-win" % variant))
    out = Path(os.environ.get("OUT") or src / ("out-engine-win-" + variant))
    jobs = os.environ.get("BUILD_JOBS") or "6"
    extra = (os.environ.get("STRATA_CMAKE_EXTRA") or "").split()
    version, commit = project_version(ENGINE_SRC), source_commit(src)
    cuda = find_cuda(variant)
    env = msvc_env()
    env["CUDA_PATH"] = str(cuda)                         # package_engine.py copies the DLLs strata.exe imports from here
    env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    cmake = shutil.which("cmake", path=env["PATH"]) or fail("cmake not found")

    run([cmake, *configure_args(ENGINE_SRC, build, variant, cuda, extra, os.environ.get("STRATA_GGML_DIR", ""), archs)], env)
    targets = engine_targets(cmake, build, env)
    print("engine targets: " + " ".join(targets), flush=True)
    run([cmake, "--build", str(build), "-j", jobs, "--target", *targets], env)

    stage = Path(tempfile.mkdtemp(prefix="strata-engine-stage-"))
    try:
        for name in ("strata.exe", "strata_kq.dll"):
            if (build / name).is_file():
                shutil.copy2(build / name, stage / name)
            elif name in targets or name == "strata.exe":
                fail("the build made no %s" % name)
        (stage / "BUILD.json").write_text(json.dumps({"version": version, "source": "packaging/build_cuda_windows.py", "commit": commit[:12],
                                                      "archs": list(archs), "ptx": True, "cuda": cu, "vision": "gpu-cuda" + cu.split(".")[0],
                                                      "portable": True}, indent=1), encoding="utf-8")
        venv = dict(env, VISION_BUILD_DIR=str(vision_dir), BUILD_JOBS=jobs, VISION_ARCHS=";".join(str(a) for a in archs))
        run(["cmd", "/d", "/c", str(PACKAGING / "build_vision_windows.bat"), variant, str(stage)], venv)   # writes vision_archs
        out.mkdir(parents=True, exist_ok=True)
        py = sys.executable
        cmd = [py, str(PACKAGING / "package_engine.py"), "--src", str(stage), "--backend", "cuda", "--version", version, "--out", str(out),
               *[x for a in archs for x in ("--arch", "sm_%d" % a)], "--arch", "compute_%d" % archs[-1], "--min-driver", "nvidia=" + mindriver,
               "--variant", variant, "--platform", "windows-x86_64", "--build-id", "%s+cuda%s" % (commit[:12], cu),
               "--include", "strata-vision.exe", "--include", "strata_kq.dll"]
        run(cmd, env)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    folder = out / ("%s-windows-x86_64-cuda-%s" % (version, variant))
    run([sys.executable, str(PACKAGING / "package_engine.py"), "--verify", str(folder)], env)
    run([sys.executable, str(PACKAGING / "binary_deps.py"), str(folder)], env)
    print("built: %s" % folder)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
