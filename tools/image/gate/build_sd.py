"""OWN-5: build sd-cli on a test box the way setup.py's build_image_engine does (tools/image/qimage.py's CMake
definitions, Release, the native runner applied on CUDA / HIP), in this lane's own folder.

    python3 -I -B build_sd.py <root> <backend cuda|hip|vulkan|cpu> [arch] [--min-free-gb N] [--jobs N]

<root> holds repo.tgz (git archive of f-own-5) and sdsrc.tgz (the pinned stable-diffusion.cpp tree, .strata-pin
inside).  Results: <root>/repo, <root>/sd-<backend>, <root>/build-<backend>/bin/sd-cli, <root>/build-<backend>.json.
HIP: ROCm from the pip SDK in /data/Strata/.venv (as the engine builds on p620) unless ROCM_PATH is set.
Vulkan: the LunarG SDK (qimage.VULKAN_SDK_*) unpacked into <root>/vulkan-sdk when the box has no glslc."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def free_gb(p: Path) -> float:
    return shutil.disk_usage(p).free / 1e9


def sh(cmd, env=None, log=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, "w") as f:
            r = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            print(open(log, errors="replace").read()[-6000:])
    else:
        r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd[0]}")


def main() -> int:
    a = sys.argv[1:]
    root, backend = Path(a[0]).expanduser().resolve(), a[1]
    arch = a[2] if len(a) > 2 and not a[2].startswith("--") else None
    min_free = float(a[a.index("--min-free-gb") + 1]) if "--min-free-gb" in a else 15.0
    jobs = a[a.index("--jobs") + 1] if "--jobs" in a else str(max(2, (os.cpu_count() or 4) // 2))
    t0 = time.time()
    print(f"free before: {free_gb(root):.1f} GB (floor {min_free})", flush=True)
    if free_gb(root) < min_free + 3:
        raise SystemExit("ABORT: not enough disk")
    repo = root / "repo"
    if not (repo / "tools" / "image" / "qimage.py").is_file():
        repo.mkdir(exist_ok=True)
        with tarfile.open(root / "repo.tgz") as t:
            t.extractall(repo)
    sys.path.insert(0, str(repo / "tools" / "image"))
    import qimage as QI
    src = root / f"sd-{backend}"
    if not QI.source_ok(src):
        shutil.rmtree(src, ignore_errors=True)
        tmp = root / f"_sd-{backend}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        with tarfile.open(root / "sdsrc.tgz") as t:
            t.extractall(tmp)
        shutil.move(str(next(tmp.iterdir())), str(src))
        tmp.rmdir()
    if not QI.source_ok(src):
        raise SystemExit("the unpacked source does not carry the pinned .strata-pin")
    if backend in QI.NATIVE_BACKENDS:
        sh([sys.executable, "-I", "-B", repo / "tools" / "image" / "sdcpp_native" / "apply.py", src])
    env = dict(os.environ)
    extra = []
    if backend == "hip":
        hp = Path(env.get("ROCM_PATH") or "/data/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_devel")
        site = hp.parent
        libw = next((p for p in site.glob("_rocm_sdk_libraries_*")), None)
        core = site / "_rocm_sdk_core"
        env.update(HIP_PLATFORM="amd", ROCM_PATH=str(hp), HIP_PATH=str(hp))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(str(d) for d in (hp / "lib", libw / "lib" if libw else hp, core / "lib")
                                                 ) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["PATH"] = f"{hp / 'bin'}{os.pathsep}{env['PATH']}"
        defs = QI.cmake_defs("hip", arch=arch, rocm=hp)
        if libw:
            defs = [d.replace("-DCMAKE_PREFIX_PATH=", f"-DCMAKE_PREFIX_PATH={libw};") for d in defs]
    elif backend == "cuda":
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        cuda_root = Path(nvcc).resolve().parent.parent
        env["PATH"] = f"{cuda_root / 'bin'}{os.pathsep}{env['PATH']}"
        defs = QI.cmake_defs("cuda", arch=arch, cuda_root=cuda_root) + [f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    elif backend == "vulkan":
        defs = QI.cmake_defs("vulkan")
        if not shutil.which("glslc"):
            sdk = root / "vulkan-sdk" / QI.VULKAN_SDK_VERSION / "x86_64"
            if not (sdk / "bin" / "glslc").exists():
                tar = root / f"vulkansdk-{QI.VULKAN_SDK_VERSION}.tar.xz"
                if not tar.exists() or tar.stat().st_size != QI.VULKAN_SDK_SIZE:
                    print("downloading the LunarG Vulkan SDK", flush=True)
                    # urllib's default User-Agent gets 403 from sdk.lunarg.com; curl's is accepted
                    sh(["curl", "-L", "--fail", "--retry", "5", "-s", "-S", "-o", tar, QI.VULKAN_SDK_URL])
                h = hashlib.sha256()
                with open(tar, "rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                if tar.stat().st_size != QI.VULKAN_SDK_SIZE or h.hexdigest() != QI.VULKAN_SDK_SHA256:
                    raise SystemExit("the Vulkan SDK download does not match its pinned size / sha256")
                with tarfile.open(tar) as t:
                    t.extractall(root / "vulkan-sdk", filter="data")
                tar.unlink()
            env["VULKAN_SDK"] = str(sdk)
            env["PATH"] = f"{sdk / 'bin'}{os.pathsep}{env['PATH']}"
            loader = sdk / "lib" / "VulkanLoader" / "lib"
            env["LD_LIBRARY_PATH"] = f"{loader}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
            extra = [f"-DVulkan_LIBRARY={loader / 'libvulkan.so'}", f"-DVulkan_INCLUDE_DIR={sdk / 'include'}",
                     f"-DVulkan_GLSLC_EXECUTABLE={sdk / 'bin' / 'glslc'}"]
    else:
        defs = QI.cmake_defs("cpu")
    bdir = root / f"build-{backend}"
    gen = ["-G", "Ninja"] if shutil.which("ninja") else ["-G", "Unix Makefiles"]
    conf = ["cmake", *gen, "-S", src, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *defs, *extra]
    sh(conf, env=env, log=root / f"cfg-{backend}.log")
    sh(["nice", "-n", "15", "cmake", "--build", bdir, "--target", "sd-cli", "-j", jobs], env=env,
       log=root / f"build-{backend}.log")
    exe = bdir / "bin" / "sd-cli"
    if not exe.exists():
        raise SystemExit(f"no {exe}")
    info = {**QI.build_stamp(backend, arch), "configure": [str(c) for c in conf], "generator": gen[1],
            "minutes": round((time.time() - t0) / 60, 1), "exe": str(exe), "exe_bytes": exe.stat().st_size,
            "env_ld": env.get("LD_LIBRARY_PATH", ""), "env_path": env["PATH"]}
    (root / f"build-{backend}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("backend", "arch", "minutes", "exe_bytes", "generator")}))
    print(f"free after: {free_gb(root):.1f} GB", flush=True)
    print("BUILD-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
