# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream's engines for one GPU maker, built in the bench container (podman-bench.sh upstream) as upstream's own
setup builds them: its setup.py's build_engine (CUDA) or build_engine_hip (HIP) and build_image_engine (sd-cli: cuda,
hip, or vulkan on Intel), loaded through tools/bench_upstream.py; for Intel its sycl/ port (engine/sycl/ from 0.2.16)
with its tools/build.sh and the image's oneAPI DPC++ (AOT: the ahead-of-time device, as its docs/INTEL.md names it;
empty for SPIR-V).  The GPU's architecture for CUDA and HIP is read from the GPU (nvidia-smi, rocminfo) unless ARCH
gives it.

    python3 up_build.py TREE MAKER        # in the container; TREE: a writable copy of upstream's tag

Prints the variables the bench scripts take for the up arm (UP, UP_PROFILE, UP_ROOT, UP_IMAGE_BACKEND, and UP_TREE and
UP_SYCL_DIRECT for the sycl/ port), K=V a line, and writes them to TREE/UP.env as well.
"""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo / "tools"))
import bench_upstream as U  # noqa: E402

if len(sys.argv) != 3 or sys.argv[2] not in ("intel", "nvidia", "amd"):
    sys.exit(f"usage: {sys.argv[0]} TREE intel|nvidia|amd")
tree, maker = Path(sys.argv[1]).resolve(), sys.argv[2]


def arch() -> str:
    if os.environ.get("ARCH"):
        return os.environ["ARCH"]
    if maker == "nvidia":
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout.split()[0]
        return cap.replace(".", "")
    rocminfo = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "rocminfo"
    m = re.search(r"\bgfx[0-9a-f]+\b", subprocess.run([str(rocminfo)], capture_output=True, text=True).stdout)
    if not m:
        sys.exit("rocminfo shows no gfx architecture: give it as ARCH")
    return m.group(0)


def lib_path(eng: Path, extra: list[str]) -> str:
    try:
        dirs = json.loads((eng / "BUILD.json").read_text()).get("lib_dirs") or []
    except (OSError, ValueError):
        dirs = []
    return os.pathsep.join(d for d in dirs + extra if Path(d).is_dir())


os.chdir(tree)
# the builds' output (upstream's setup and its compilers) to stderr, so that stdout carries the variables alone
out = os.dup(1)
os.dup2(2, 1)
sys.stdout = os.fdopen(os.dup(2), "w", buffering=1)
# AMD's own ROCm (container/install.sh) is in /opt/rocm/core-<version>, where upstream's setup looks for it by ROCM_PATH
core = sorted(Path("/opt/rocm").glob("core-*"))
if core and not os.environ.get("ROCM_PATH"):
    os.environ["ROCM_PATH"] = str(core[-1])
gpu: dict[str, int | str] = {}
term = U.terminal(tree)  # its setup.py, serve/ and data/ (terminal/ from 0.2.16)
v = {"UP_ROOT": str(term), "UP_PROFILE": str(term / "data" / "expert-profile.bin")}
mod = U.load_setup(tree)
if maker == "intel":
    aot = os.environ.get("AOT", "")
    exe = tree / "build-sycl-aot" / "strata"  # upstream's engine
    env = Path(__file__).with_name("oneapi-env.sh")
    build = f"source {env}; bash {U.sycl_dir(tree).relative_to(tree)}/tools/build.sh strata"  # upstream's target
    subprocess.run(["bash", "-c", build], check=True,
                   env={**os.environ, "AOT": aot, "REPO": str(tree), "BUILD_DIR": str(tree / "build-sycl-aot")})
    if not exe.exists():
        sys.exit(f"upstream's sycl/ build made no {exe.name} (its build-sycl-aot/build.log)")
    v.update(UP=str(repo / ".claude/skills/release/scripts/up-sycl.sh"), UP_TREE=str(tree), UP_SYCL_DIRECT="1")
    backend = "vulkan"
else:
    a = arch()
    gpu = {"arch": int(a)} if maker == "nvidia" else {"arch": a}
    llama = U.call(mod, "get_llama_cpp")
    if maker == "nvidia":
        eng = Path(U.call(mod, "build_engine", gpu, "none", True, llama))
        libs = lib_path(eng, ["/usr/local/cuda/lib64"])
    else:
        eng = Path(U.call(mod, "build_engine_hip", gpu, llama))
        libs = lib_path(eng, [])
    v["UP"] = (f"env LD_LIBRARY_PATH={libs} " if libs else "") + str(eng / "strata")  # upstream's engine
    backend = "cuda" if maker == "nvidia" else "hip"
exe, _ = U.call(mod, "build_image_engine", backend, gpu)
v["UP_IMAGE_BACKEND"] = backend
lines = [f"{k}={shlex.quote(x)}" for k, x in v.items()]
(tree / "UP.env").write_text("\n".join(lines) + "\n")
sys.stdout.flush()
os.write(out, ("\n".join(lines) + "\n").encode())
print(f"up_build.py: upstream's sd-cli: {exe}", file=sys.stderr)
