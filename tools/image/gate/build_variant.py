"""tools/image/gate/build_variant.py - TEST-ONLY (OWN-5 gate harness): build the stage-hook variant of an existing
build_sd.py build with exactly its configure line.

    python3 -I -B build_variant.py <root> <backend> <variant> [--jobs N] [--def -DNAME=VALUE ...]
Makes <root>/sd-<variant> (stage_hooks.py applied to a copy of <root>/sd-<backend>), <root>/build-<variant> and
<root>/build-<variant>.json (what run_set.py --build <variant> reads)."""
import json
import os
import subprocess
import sys
from pathlib import Path

a = sys.argv[1:]
root, backend, var = Path(a[0]).expanduser().resolve(), a[1], a[2]
jobs = a[a.index("--jobs") + 1] if "--jobs" in a else "8"
extra = [a[i + 1] for i, x in enumerate(a) if x == "--def"]      # extra CMake definitions, e.g. -DGGML_CUDA_FORCE_CUBLAS=ON
info = json.loads((root / f"build-{backend}.json").read_text())
here = Path(__file__).resolve().parent
subprocess.run([sys.executable, "-I", "-B", str(here / "stage_hooks.py"), str(root / f"sd-{backend}"),
                str(root / f"sd-{var}")], check=True)
conf = [c.replace(str(root / f"sd-{backend}"), str(root / f"sd-{var}")).replace(str(root / f"build-{backend}"),
                                                                                str(root / f"build-{var}"))
        for c in info["configure"]] + extra
env = dict(os.environ, PATH=info["env_path"])
if info.get("env_ld"):
    env["LD_LIBRARY_PATH"] = info["env_ld"]
print("+", " ".join(conf), flush=True)
subprocess.run(conf, check=True, env=env, stdout=subprocess.DEVNULL)
subprocess.run(["nice", "-n", "15", "cmake", "--build", str(root / f"build-{var}"), "--target", "sd-cli", "-j", jobs],
               check=True, env=env, stdout=subprocess.DEVNULL)
exe = root / f"build-{var}" / "bin" / "sd-cli"
info.update(configure=conf, exe=str(exe), exe_bytes=exe.stat().st_size, variant_of=backend)
(root / f"build-{var}.json").write_text(json.dumps(info, indent=1))
print("VARIANT-OK", exe)
