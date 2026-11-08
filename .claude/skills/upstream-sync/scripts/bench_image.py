# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The image table of an upstream merge or a release on one GPU: XeStrata's and upstream's image generators
(Qwen-Image-2.1-Turbo), each as its own setup sets it up on this GPU and as its server runs it (serve/images.py), one
1024x1024 image of 8 steps from a fixed prompt and seed a run.  XeStrata: its resident image engine (xestrata-image, in
engine/ beside the text engine), measured warm (a warm-up image of another prompt first, in the same process);
upstream: stable-diffusion.cpp's sd-cli on the backend its setup picks for the GPU (CUDA on NVIDIA, HIP on AMD,
Vulkan on Intel; UP_IMAGE_BACKEND), built by its setup (build_image_engine), one process an image.  The runs are
tools/bench_core.py's (ab_image), which setup's --bench makes too.

    XE_ROOT=DIR UP_ROOT=DIR DATA=DIR QUANT=Q8_0 UP_IMAGE_BACKEND=cuda [XE_IMAGE_PCI=ADDR] [UP_IMAGE_GPU=N] \\
      python .claude/skills/upstream-sync/scripts/bench_image.py OUT GPU [--dry-run]     # from the repository root

XE_ROOT: XeStrata's tree, with engine/xestrata-image and the BUILD.json setup writes beside it (the "lib_dirs" of the
compiler's runtime); XE_IMAGE_EXE: a xestrata-image elsewhere (a build folder: then XE_LIB_DIRS, the runtime's folders,
separated as PATH is).  XE_IMAGE_PCI: the GPU's PCI address (XESTRATA_GPU_PCI, as setup's config names it).  UP_ROOT:
upstream's tree, with its sd-cli in engine-image/ and the BUILD.json its setup writes beside it (its backend, checked
against the one asked for, and the "lib_dirs" its config takes); UP_ROOT=none: XeStrata alone.  UP_SD_CLI: an sd-cli
elsewhere (its BUILD.json beside it).  DATA: the data folder (the image model's files in its models/).  QUANT: the
denoiser's quant, the same for both (upstream's setup picks it by the GPU's memory: Q8_0 above 12 GB, Q6_K up to 12,
Q4_K_M up to 8).  IMAGE_MODEL (default qwen-image-turbo): setup's --images name of the model; upstream has
Qwen-Image-2.1-Turbo alone, so another (mage-flow-turbo) takes UP_ROOT=none, and its steps are the model card's (4 for
Mage-Flow-Turbo).  UP_IMAGE_GPU: the GPU's index among the backend's devices where the PC has more than one (Vulkan
counts the processor's graphics too).  PY (default this Python): the Python that runs each tree's image code.  ROUNDS
(default 2): images per engine, alternating, after a discarded warm-up per engine.
OUT/image-runs/GPU-ARM-ROUND.json gets each run's times, .txt the log (sd-cli's, or the resident engine's figures) and
.png the image; sync_collect.py OUT reads them.  --dry-run: print the configs instead of running them.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
models = str(Path(B.need("DATA")) / "models")
quant = B.need("QUANT")
py = os.environ.get("PY", sys.executable)
model = os.environ.get("IMAGE_MODEL") or "qwen-image-turbo"


engines, cfgs, envs = [], {}, {}
for arm in ("xe", "up"):
    root = B.need(f"{arm.upper()}_ROOT")
    if root == "none":
        continue
    if arm == "up" and model != "qwen-image-turbo":
        sys.exit(f"up: upstream has no {model} (UP_ROOT=none)")
    e = B.Engine(arm, arm, [], "", root=Path(root), py=py)
    if arm == "xe":
        exe = Path(os.environ.get("XE_IMAGE_EXE") or
                   Path(root) / "engine" / ("xestrata-image.exe" if os.name == "nt" else "xestrata-image"))
        if os.environ.get("XE_IMAGE_EXE"):
            lib_dirs = [d for d in os.environ.get("XE_LIB_DIRS", "").split(os.pathsep) if d]
        else:
            try:
                lib_dirs = json.loads((exe.parent / "BUILD.json").read_text()).get("lib_dirs") or []
            except (OSError, ValueError) as err:
                sys.exit(f"xe: {exe.parent / 'BUILD.json'}: {err} (the engine its setup built)")
        env = {}
        if lib_dirs:
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = os.pathsep.join(lib_dirs + ([os.environ[var]] if os.environ.get(var) else []))
        if os.environ.get("XE_IMAGE_PCI"):
            env["XESTRATA_GPU_PCI"] = os.environ["XE_IMAGE_PCI"]
        cfg = B.image_config(e, "xestrata-image", quant, str(exe), models, model=model)
        envs["xe"] = env
    else:
        backend = os.environ.get("UP_IMAGE_BACKEND", "")
        if not backend:
            sys.exit("UP_IMAGE_BACKEND: the backend upstream's setup picks for this GPU (cuda, hip or vulkan)")
        exe = Path(os.environ.get("UP_SD_CLI") or
                   Path(root) / "engine-image" / ("sd-cli.exe" if os.name == "nt" else "sd-cli"))
        try:
            stamp = json.loads((exe.parent / "BUILD.json").read_text())
        except (OSError, ValueError) as err:
            sys.exit(f"up: {exe.parent / 'BUILD.json'}: {err} (the sd-cli its setup built)")
        if stamp.get("backend") != backend:
            sys.exit(f"up: {exe} is the {stamp.get('backend')} build, not {backend}")
        uv = os.environ.get("UP_IMAGE_GPU", "")
        cfg = B.image_config(e, backend, quant, str(exe), models, int(uv) if uv not in ("", "-") else None,
                             stamp.get("lib_dirs") or [])
    if "error" in cfg:
        sys.exit(f"{arm}: its image config: {cfg['error']}")
    print(f"{arm}: {cfg.get('backend') or 'xestrata-image'} {quant}, args {' '.join(cfg.get('args') or [])}, "
          f"env {cfg.get('env') or envs.get(arm) or {}}", flush=True)
    engines.append(e)
    cfgs[arm] = cfg
B.ab_image(Path(args[0]), args[1], engines, cfgs, int(os.environ.get("ROUNDS", str(B.ROUNDS))),
           dry="--dry-run" in sys.argv, envs=envs, model=model)
