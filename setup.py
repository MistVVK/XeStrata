#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strata setup and start (Linux, Intel Arc and other Intel GPUs).

    ./setup.sh installs Python if needed and runs this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts the model on http://127.0.0.1:8095 (OpenAI- and Anthropic-compatible
API; a small page there shows that it runs). Every later start skips straight to running the model: nothing that
is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: the GPU (Intel; NVIDIA with --license contrib) and its driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ...)
  4. compiles the Strata engine for the GPU with a SYCL compiler (see docs/XE.md), and the image encoder when
     images are wanted (on the CPU; on the GPU through Vulkan, or SYCL with oneMKL in the contrib-icpx mode)
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S, --context 32768, --rope-scaling none|linear|yarn
(--rope-scale F; past the trained 262144 setup adds yarn and the factor final context / 262144),
--vision yes|no|gpu|cpu, --vision-onednn on|off, --port 8095, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it from other devices on your network), --experimental-speed-projection on|off
(EXPERIMENTAL, off by default),
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --check (only check this PC).

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, a size it thinks will not fit - is kept, with what it risks.  With --yes, an
explicit flag (--model, --context) is the consent to a risk setup would otherwise stop at; --yes alone is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parent
# Every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when it was pinned, upstream #214), so a checkout installs the same files
# on any day.  A revision the repository no longer has falls back to its current files, with a message (download()).
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",   # 2026-09-24
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (upstream #214)

MODELS: dict[str, dict[str, Any]] = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # the original model only (Swift 1.5 has no IQ3_S): matches the full BF16 model on the published benchmarks
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # upstream #621: Unsloth's UD-IQ4_XS - IQ3_S gate/up experts with IQ4_NL (43 layers) or Q8_0 (5) downs, the dense
    # side as UD-Q4_K_XL's; three shards.  Its 59.5 GB of experts: a RAM budget of them, like UD-Q4_K_XL, but far
    # fewer read from the SSD on a 64 GB PC and none from ~80 GB of RAM.  Images: the same base model and encoder
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "vision": True},
    # EXPERIMENTAL (upstream's UD-Q4_K_XL support, 3889344 / 872af82): Unsloth's 4-bit file; its 77 GB of experts do not
    # fit a 64 GB PC, so the engine keeps a RAM budget of them (--resident-budget-gib, chosen below) and reads the rest
    # from the GGUF files in place
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), EXPERIMENTAL: the best quality, but on a 64 GB PC part of its "
                            "experts come from the SSD while it answers, so it is slower than the 2-3-bit "
                            "models", "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",),
                   "budget": True, "experimental": True},
}
# The experimental Unsloth file's four shards at the pinned revision: name -> (bytes, sha256), checked after the
# download (check_shards reads the other model files' directories; these are also hashed once).
UNSLOTH_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
}
# UD-IQ4_XS's three shards at the same revision (sizes and SHA-256: the Hub's LFS pointers; upstream #621)
UNSLOTH_IQ4_XS_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
}
UNSLOTH_RAM_LEFT_GB = 24        # RAM beside the budget: the OS, the engine, and the file cache the rest is read through
# Contexts past 262144 (the model's trained length) extend it by rope scaling (upstream #84): for the context it will
# serve the setup takes yarn (or asks, when interactive) and the factor final / 262144 (at least 1), keeps an explicit
# --rope-scaling / --rope-scale, and refuses an explicit --rope-scaling none there.
CONTEXTS = [8192, 32768, 65536, 131072, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES: dict[str, dict[str, Any]] = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's UD-IQ4_XS (three shards, upstream #621) and the EXPERIMENTAL UD-Q4_K_XL (four) of the original model;
    # "experimental" and "vision" are per model (MODELS)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's ~4-bit quantizations",
                "about": "UD-IQ4_XS: a 94 GB download; with less than ~80 GB of RAM part of its experts are read from "
                         "the SSD (UD-Q4_K_XL, 111 GB: experimental)",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "vision": False, "pack_args": ["--compat-bf16"],
                "sha256": {**UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS}},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
# under the Qwen Community License, so in third_party/nonfree (XeStrata runs without that folder)
ESP_VECTOR = (ROOT / "third_party" / "nonfree" / "experimental-speed-projection" /
              "Qwen3.8-Flash-Next-experimental-speed-projection.gguf")
# the image encoder on the GPU (~1.2 GB at 1024 image tokens) warms up before the engine starts, so the engine
# sizes its expert slots around it and the default reserve (700 MiB) is enough; engines before 0.1.2 need more
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
          "cpu": {"max_tokens": 300, "reserve_mib": 700}}
EXE = "strata"
# the image encoder, one binary per place it runs: the configs of different models may use different ones
# the image encoders: on the CPU, on the GPU through Vulkan (free), on the GPU through SYCL (oneMKL: contrib-icpx)
VEXE = {"cpu": "strata-vision-cpu", "vulkan": "strata-vision-vulkan", "gpu": "strata-vision-sycl"}


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None) -> NoReturn:
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def ask(question, choices, default, yes):
    if yes:
        return default
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the PC
def is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    try:
        txt = open("/proc/cpuinfo").read()
        flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        avx2 = "avx2" in flags
        avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
        m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
        name = m.group(1) if m else name
    except OSError:
        pass
    return name, avx2, avx512


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
ARGS = None                                             # the command line (--license decides which cards can be used)


def _drm_iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


def drm_memory(render: str, driver: str):
    """(VRAM bytes, of it visible to the CPU) as the kernel driver reports its memory regions (xe's
    DRM_IOCTL_XE_DEVICE_QUERY, i915's DRM_IOCTL_I915_QUERY), or None when it cannot be asked.  (0, 0) for a GPU that
    has only system memory (the processor's own graphics)."""
    import ctypes
    import fcntl
    try:
        fd = os.open(render, os.O_RDWR)
    except OSError:
        return None
    try:
        vram = vis = 0
        if driver == "xe":
            # struct drm_xe_device_query; DRM_XE_DEVICE_QUERY_MEM_REGIONS = 1; regions of 88 bytes after 8
            q = bytearray(struct.pack("QIIQQQ", 0, 1, 0, 0, 0, 0))
            req = _drm_iowr(0x40, len(q))
            fcntl.ioctl(fd, req, q)
            buf = ctypes.create_string_buffer(struct.unpack_from("I", q, 12)[0])
            struct.pack_into("Q", q, 16, ctypes.addressof(buf))
            fcntl.ioctl(fd, req, q)
            for i in range(struct.unpack_from("I", buf, 0)[0]):
                cls, _, _, total, _, cpu_vis, _ = struct.unpack_from("HHIQQQQ", buf, 8 + 88 * i)
                if cls == 1:                                # DRM_XE_MEM_REGION_CLASS_VRAM
                    vram, vis = vram + total, vis + cpu_vis
        elif driver == "i915":
            # struct drm_i915_query with one DRM_I915_QUERY_MEMORY_REGIONS (4) item; regions of 88 bytes after 16
            item = ctypes.create_string_buffer(struct.pack("QiIQ", 4, 0, 0, 0))
            q = bytearray(struct.pack("IIQ", 1, 0, ctypes.addressof(item)))
            req = _drm_iowr(0x79, len(q))
            fcntl.ioctl(fd, req, q)
            length = struct.unpack_from("i", item, 8)[0]
            if length <= 0:
                return None
            buf = ctypes.create_string_buffer(length)
            struct.pack_into("Q", item, 16, ctypes.addressof(buf))
            fcntl.ioctl(fd, req, q)
            for i in range(struct.unpack_from("I", buf, 0)[0]):
                cls, _, _, probed, _, cpu_vis = struct.unpack_from("HHIQQQ", buf, 16 + 88 * i)
                if cls == 1:                                # I915_MEMORY_CLASS_DEVICE
                    vram, vis = vram + probed, vis + (cpu_vis or probed)
        else:
            return None
        return vram, vis
    except (OSError, struct.error):
        return None
    finally:
        os.close(fd)


def pci_name(vendor: int, device: int) -> str:
    """The card's name from the PCI ID database (hwdata), or its IDs."""
    for f in ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"):
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"^%04x .*?$(.*?)^[0-9a-f]{4} " % vendor, text, re.M | re.S)
        if m:
            d = re.search(r"^\t%04x  (.+)$" % device, m.group(1), re.M)
            if d:
                n = d.group(1).strip()                  # "Battlemage G21 [Arc B580]", "Battlemage G31 [Intel Graphics]"
                b = re.search(r"\[(.+)\]", n)
                short = b.group(1) if b else n
                if short.startswith("Intel"):            # a bracket that names no product: the chip's name too
                    return f"{short} ({n[:n.find('[')].strip() or n})"
                return "Intel " + short
        break
    return f"Intel GPU 0x{device:04x}"


def host_link(dev: Path):
    """The PCIe link between the card and the CPU, from sysfs: that of the first device below the root port.  A B70
    sits behind its own switch, whose inner links report 2.5 GT/s x1."""
    parts = dev.resolve().parts
    i = next((k for k, x in enumerate(parts) if x.startswith("pci")), None)
    if i is None or len(parts) < i + 3:
        return None
    top = Path(*parts[:i + 3])
    try:
        return f"{(top / 'current_link_speed').read_text().strip()} x{(top / 'current_link_width').read_text().strip()}"
    except OSError:
        return None


def nvidia_smi() -> dict:
    """What NVIDIA's driver reports of each NVIDIA GPU, by PCI address ("0000:09:00.0"): its name, memory (MiB) and
    compute capability ("8.9").  Empty without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    try:
        r = subprocess.run([exe, "--query-gpu=pci.bus_id,name,memory.total,compute_cap",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 4:
            mib = float(f[2]) if re.fullmatch(r"[\d.]+", f[2]) else 0.0
            found[f[0].lower()[-12:]] = {"name": f[1], "mib": mib, "cc": f[3]}   # "00000000:09:00.0" -> "0000:09:00.0"
    return found


def gpus():
    """Every Intel GPU on the xe or i915 driver in PCI order (the order Level Zero numbers them), then every NVIDIA GPU
    on NVIDIA's driver (after them, so that the Intel cards keep the numbers saved configs know them by), from sysfs,
    the Intel driver's memory query and nvidia-smi: no tool of Strata's is needed before the engine is built.  Nothing
    is decided from the device ID: the engine checks at start what the card reports (AGENTS.md)."""
    found, nvidia = [], []
    smi = None
    cards = [c for c in Path("/sys/class/drm").glob("card*") if re.fullmatch(r"card\d+", c.name)]
    for card in sorted(cards, key=lambda c: (c / "device").resolve().name):
        dev = card / "device"
        try:
            driver = (dev / "driver").resolve().name
            vendor = (dev / "vendor").read_text().strip()
            if (vendor, driver) not in (("0x8086", "xe"), ("0x8086", "i915"), ("0x10de", "nvidia")):
                continue
            did = int((dev / "device").read_text().strip(), 16)
            # the BAR the CPU sees the memory through: BAR 2 on Intel's cards, BAR 1 on NVIDIA's
            bar = (dev / "resource").read_text().splitlines()[2 if vendor == "0x8086" else 1].split()
            bar_gb = (int(bar[1], 16) - int(bar[0], 16) + 1) / 2 ** 30 if int(bar[1], 16) else 0.0
        except (OSError, ValueError, IndexError):
            continue
        pci = dev.resolve().name
        if vendor == "0x10de":
            smi = nvidia_smi() if smi is None else smi
            info = smi.get(pci, {})
            vram = info.get("mib", 0.0) / 1024 or bar_gb
            node = "/dev/nvidiactl" if Path("/dev/nvidiactl").exists() else None
            nvidia.append({"name": info.get("name") or f"NVIDIA GPU 0x{did:04x}", "vram_gb": vram,
                           "device_id": did, "driver": driver, "integrated": False, "supported": True, "pci": pci,
                           "bar_gb": bar_gb, "link": host_link(dev), "render": node, "vendor": "nvidia",
                           "cc": info.get("cc")})
            continue
        render = next(iter(sorted((dev / "drm").glob("renderD*"))), None)
        node = f"/dev/dri/{render.name}" if render else None
        mem = drm_memory(node, driver) if node else None
        integrated = mem is not None and mem[0] == 0
        vram = mem[0] / 2 ** 30 if mem else bar_gb             # the BAR when the driver cannot be asked
        visible = mem[1] / 2 ** 30 if mem else bar_gb
        found.append({"index": len(found), "name": pci_name(0x8086, did), "vram_gb": vram, "device_id": did,
                      "driver": driver, "integrated": integrated, "supported": True, "pci": pci,
                      "bar_gb": visible, "link": host_link(dev), "render": node, "vendor": "intel"})
    return found + [{"index": len(found) + i, **g} for i, g in enumerate(nvidia)]


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can).  What the card can do (16-wide sub-groups,
    FP16, the matrix engines) is checked by the engine and the compiler's probe, not here."""
    if together:
        return "not supported together with other GPUs - the Xe engine runs on one GPU"
    if g.get("vendor") == "nvidia" and license_mode(ARGS) != "contrib":
        return "an NVIDIA GPU needs the contrib build (--license contrib: NVIDIA's CUDA toolkit, not free software)"
    return None


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, " + ("shares the system RAM" if g.get("integrated") else
                                                 f"{g['vram_gb']:.0f} GB") + ")"


def gpu_table(found) -> None:
    say("  Your GPUs:")
    for g in found:
        p = gpu_problem(g)
        mem = "no VRAM of its own" if g.get("integrated") else f"{g['vram_gb']:.0f} GB VRAM"
        say(f"    GPU {g['index']}: {g['name']}, {mem} - " + ("can be used" if p is None else p))


def parse_gpus(text, found) -> list:
    """--gpus: a layer split across several cards, which the Xe engine does not do."""
    fail(f"--gpus {text}: the Xe engine runs on one GPU",
         "use --gpu N (or leave it out for the card with the most VRAM)")


def check_gpus(sel, found, what="") -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can."""
    if len(sel) > 1:
        parse_gpus(",".join(str(i) for i in sel), found)
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g)
        if p is None:
            continue
        say()
        gpu_table(found)
        single = [x for x in found if gpu_problem(x) is None]
        hint = ("use " + " or ".join(f"--gpu {x['index']}" for x in single)) if single else \
            "Strata's Xe engine needs an Intel GPU on the xe or i915 driver, or with --license contrib an NVIDIA GPU"
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def choose_gpus(a, found) -> list:
    """Which card this install uses: --gpu, else the supported card with the most VRAM.  Returns its number in a
    list (the Xe engine runs on one GPU)."""
    if a.gpus:
        parse_gpus(a.gpus, found)
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run Strata",
             "it needs an Intel GPU on the xe or i915 driver, or with --license contrib an NVIDIA GPU")
    return [single[0]["index"]]


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (its number in gpus()) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no Intel GPU on the xe or i915 driver.  The dict also says how many there are
    ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_tool(name):
    """A tool on PATH, or the one pip installed next to this Python (cmake, ninja)."""
    p = shutil.which(name)
    if p:
        return p
    for d in (Path(sys.executable).parent, Path.home() / ".local" / "bin"):
        c = d / name
        if c.exists():
            return str(c)
    return None


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # the pinned revision is gone from the repository
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file")
                url = hf_unpinned(url)
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")


def verify_sha256(s: Path, size: int, sha: str) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file); the result is kept in its finish mark,
    so the minutes of hashing 111 GB happen once.  A wrong file is deleted, so the next run downloads it again."""
    m = s.with_name(s.name + ".done")
    if m.exists() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace"):
        return
    have = s.stat().st_size if s.exists() else -1
    if have != size:
        fail(f"{s.name} is {have:,} bytes, not {size:,}", "delete it and run setup again (the download restarts)")
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({h.hexdigest()}, expected {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = ROOT / "third_party" / "main" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir() and \
            (llama / "tools" / "mtmd" / "CMakeLists.txt").exists():   # a copy without mtmd cannot build the encoder
        return llama
    z = ROOT / "third_party" / "main" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "main" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    top = next(tmp.iterdir())
    if llama.exists():
        shutil.rmtree(llama)
    top.replace(llama)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """requirements.txt's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements recorded bare names: those packages are kept as they are (nothing is reinstalled), and the pinned
    dependencies it already has count as installed."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


# ------------------------------------------------------------------------------------------------ the engine
ONEAPI_ROOT = Path(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"))
_ONEAPI_ENV: dict | None = None


def oneapi_env() -> dict | None:
    """The environment oneAPI's setvars.sh sets up (its compilers, its libraries and the SYCL runtime on the paths), or
    None when oneAPI is not installed.  The engine is built with it."""
    global _ONEAPI_ENV
    if _ONEAPI_ENV is None:
        sv = ONEAPI_ROOT / "setvars.sh"
        if not sv.is_file():
            return None
        # the path goes in as $1: setvars.sh takes a $0 equal to its own path for "run, not sourced" and exits
        r = subprocess.run(["bash", "-c", 'source "$1" >/dev/null 2>&1; env -0', "_", str(sv)], capture_output=True)
        if r.returncode != 0:
            return None
        _ONEAPI_ENV = dict(x.split("=", 1) for x in r.stdout.decode(errors="replace").split("\0") if "=" in x)
    return _ONEAPI_ENV


def oneapi_lib_dirs() -> list:
    """The library folders the engine and the image encoders load at run time (the SYCL runtime; oneMKL for the SYCL
    image encoder), which the server puts on LD_LIBRARY_PATH: the server does not run in oneAPI's environment."""
    env = oneapi_env() or {}
    return [d for d in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d and Path(d).is_dir()]


# ------------------------------------------------------------------------------------------------ the SYCL compiler
# AGENTS.md, "Free and non-free builds": three build modes (--license).  free uses free software only:
# intel/llvm's DPC++ 7 or later, the distribution's (dpclang++) or one built from source (tools/intel_llvm_build.py,
# --intel-llvm-build; setup offers to build it when the distribution's is older), for Intel GPUs.  contrib (the default): intel/llvm
# built with its CUDA target (.tools/intel-llvm-contrib), for Intel and NVIDIA GPUs, with oneMKL and cuBLAS for the
# dense products.  contrib-icpx: Intel oneAPI's icpx and oneMKL,
# Intel GPUs only.  Whether the Intel GPU has its matrix engines (XMX) for a compiler is asked of the GPU itself
# (tools/xmx_probe.cpp), not read from a version number.  Without XMX the engine still runs, its prompt path about 1.6
# times slower (bench/results/2026-10-02-dp4a); setup asks before building it so.
LICENSES = ("free", "contrib", "contrib-icpx")
INTEL_LLVM_BUILD = ROOT / "tools" / "intel_llvm_build.py"
INTEL_LLVM_DEFAULT = ROOT / ".tools" / "intel-llvm" / "install"
INTEL_LLVM_CONTRIB = ROOT / ".tools" / "intel-llvm-contrib" / "install"
ICPX_HOW = ("Intel oneAPI's compiler (not free software; Intel's apt repository):\n"
            "         wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \\\n"
            "           | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null\n"
            "         echo \"deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] "
            "https://apt.repos.intel.com/oneapi all main\" | sudo tee /etc/apt/sources.list.d/oneAPI.list\n"
            "         sudo apt update && sudo apt install intel-oneapi-compiler-dpcpp-cpp")
FREE_HOW = ("intel/llvm's DPC++ 7 or later (free software): build it here: ./setup.sh --intel-llvm-build (13 minutes "
            "on 28 threads, 3.5 GB), or a distribution's package of version 7 or later")
MKL_HOW = ("oneMKL (not free software; Intel's apt repository, set up as for icpx in docs/XE.md#packages): "
           "sudo apt install intel-oneapi-mkl-sycl-devel")


def license_mode(a=None) -> str:
    """The build mode: --license, else the one kept in the settings.  Settings from before the three modes kept
    "nonfree": true for icpx, which is contrib-icpx now."""
    if a is not None and getattr(a, "license", None):
        return a.license
    st = load_settings()
    return st.get("license") or ("contrib-icpx" if st.get("nonfree") else "contrib")


def mkl_root() -> str | None:
    """Where oneMKL is (the contrib modes' dense products, the SYCL image encoder): oneAPI's MKLROOT, else oneAPI's
    default folder."""
    root = (oneapi_env() or {}).get("MKLROOT") or "/opt/intel/oneapi/mkl/latest"
    return root if Path(root, "lib").is_dir() else None


def compiler_version(cxx, env) -> str:
    r = subprocess.run([str(cxx), "--version"], capture_output=True, text=True, env=env)
    lines = [x.strip() for x in r.stdout.splitlines()] or ["?"]
    # dpclang's first line ends "build based on:", the clang version is on the next
    return f"{lines[0]} {lines[1]}" if lines[0].endswith(":") and len(lines) > 1 else lines[0]


def free_compiler(install: Path, license: str = "free") -> dict:
    """intel/llvm built from source (`install` holds bin/clang++ and lib/libsycl.so); "contrib": one with its CUDA
    target."""
    lib = install / "lib"
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join([str(lib), os.environ.get("LD_LIBRARY_PATH", "")]))
    return {"kind": "intel-llvm", "cxx": str(install / "bin" / "clang++"), "cc": str(install / "bin" / "clang"),
            "env": env, "lib_dirs": [str(lib)], "license": license, "cuda_archs": [], "onemkl": False}


def new_enough(comp: dict) -> bool:
    """Whether the compiler is intel/llvm 7 or later: its SYCL runtime is libsycl 9 (the distribution's dpclang++ 6.2
    has libsycl 8, and gives the Arc Pro B70 no XMX)."""
    r = subprocess.run([comp["cxx"], "-fsycl", "-dM", "-E", "-x", "c++", "-"], input="#include <sycl/sycl.hpp>\n",
                       capture_output=True, text=True, env=comp["env"])
    m = re.search(r"#define __LIBSYCL_MAJOR_VERSION (\d+)", r.stdout)
    return m is not None and int(m.group(1)) >= 9


def os_compiler() -> dict | None:
    """The distribution's intel/llvm DPC++ (dpclang++, or the newest dpclang++-N)."""
    cxx = shutil.which("dpclang++")
    if cxx is None:
        names = {p.name for d in os.environ.get("PATH", "").split(os.pathsep) if Path(d).is_dir()
                 for p in Path(d).glob("dpclang++-*")}
        for n in sorted(names, key=lambda n: [int(x) for x in re.findall(r"\d+", n)], reverse=True):
            cxx = shutil.which(n)
            if cxx:
                break
    if cxx is None:
        return None
    cc = shutil.which(Path(cxx).name.replace("dpclang++", "dpclang")) or cxx
    return {"kind": "os", "cxx": cxx, "cc": cc, "env": dict(os.environ), "lib_dirs": [], "license": "free",
            "cuda_archs": [], "onemkl": False}


def icpx_compiler() -> dict | None:
    env = oneapi_env()
    if env is None or shutil.which("icpx", path=env.get("PATH")) is None:
        return None
    return {"kind": "icpx", "cxx": shutil.which("icpx", path=env.get("PATH")),
            "cc": shutil.which("icx", path=env.get("PATH")) or "icx", "env": env, "lib_dirs": oneapi_lib_dirs(),
            "license": "contrib-icpx", "cuda_archs": [], "onemkl": True}


def intel_gpu_present() -> bool:
    """Whether this PC has an Intel GPU Strata can use: the contrib build then gets the oneMKL backend."""
    return any(g.get("vendor") == "intel" for g in gpus())


def cuda_toolkits() -> list:
    """The CUDA toolkits on this PC, as cmake/StrataCuda.cmake looks for them (which chooses the one the contrib build
    uses): CUDA_PATH / CUDA_HOME / CUDA_ROOT, /usr/local/cuda-X.Y, /opt/cuda, Debian's /usr/lib/cuda and the nvcc on
    PATH; a folder counts when it has the libdevice clang needs."""
    import glob
    dirs = [os.environ.get(v, "") for v in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")]
    dirs += sorted(glob.glob("/usr/local/cuda-*")) + sorted(glob.glob("/opt/cuda-*"))
    dirs += ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        dirs.append(str(Path(nvcc).resolve().parent.parent))
    found = []
    for d in dirs:
        if d and Path(d).is_dir() and glob.glob(str(Path(d) / "nvvm" / "libdevice" / "libdevice*.bc")):
            r = str(Path(d).resolve())
            if r not in found:
                found.append(r)
    return found


def cuda_archs() -> list:
    """The NVIDIA architectures of this PC's NVIDIA GPUs (sm_89 for compute capability 8.9), as nvidia-smi reports
    them: the contrib build makes their code (CMake's STRATA_CUDA_ARCHS=auto reads them again and maps one the
    compiler does not know to the newest it builds)."""
    return sorted({"sm_" + g["cc"].replace(".", "") for g in nvidia_smi().values()
                   if re.fullmatch(r"\d+\.\d+", g["cc"])})


def probe_gpu(comp: dict, pci: str | None) -> dict | None:
    """What the Intel GPU at `pci` reports to this compiler's SYCL runtime (tools/xmx_probe.cpp): its fields ("fp16",
    "bf16": "1" for the matrix engines), {} when the runtime lists no such GPU, None when the probe did not build."""
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "xmx_probe"
        r = subprocess.run([comp["cxx"], "-fsycl", str(ROOT / "tools" / "xmx_probe.cpp"), "-o", str(exe)],
                           capture_output=True, text=True, env=comp["env"])
        if r.returncode != 0:
            warn(f"{Path(comp['cxx']).name} could not build the XMX probe: {(r.stderr or r.stdout).strip()[-300:]}")
            return None
        r = subprocess.run([str(exe)], capture_output=True, text=True, env=comp["env"], timeout=120)
    seen = {}
    for line in r.stdout.splitlines():
        f = dict(x.split("=", 1) for x in line.split(" name=")[0].split() if "=" in x)
        if pci is None or f.get("pci") == pci:
            if has_xmx(f):
                return f
            seen = f
    return seen


def has_xmx(probed: dict | None) -> bool:
    return probed is not None and probed.get("fp16") == "1" and probed.get("bf16") == "1"


def build_intel_llvm(a, contrib: bool = False) -> Path:
    """tools/intel_llvm_build.py (--contrib: with the CUDA target): a finished build is used as it is, an interrupted
    one continued."""
    flags = [f for f, on in (("--contrib", contrib), ("--rebuild", a.intel_llvm_rebuild),
                             ("--keep-build", a.intel_llvm_keep_build), ("--yes", a.yes)) if on]
    run([sys.executable, INTEL_LLVM_BUILD, *flags])
    return INTEL_LLVM_CONTRIB if contrib else INTEL_LLVM_DEFAULT


def contrib_compiler(a, st: dict) -> tuple:
    """intel/llvm with its CUDA target: --intel-llvm DIR, the one kept in the settings, or the one built here (built
    now when there is none).  Returns it and its folder."""
    d = Path(a.intel_llvm or st.get("intel_llvm_contrib") or INTEL_LLVM_CONTRIB).expanduser().resolve()
    if not (d / "bin" / "clang++").exists():
        if a.intel_llvm:
            fail(f"--intel-llvm {d}: no bin/clang++ there", "give the folder intel/llvm was installed to")
        say("  The contrib build needs intel/llvm with its CUDA target (NVIDIA's CUDA toolkit, not free software); "
            "it is built here from source (about 25 minutes on 28 threads, 4 GB).")
        if ask("Build it now?", ["y", "n"], "y", a.yes) != "y":
            fail("stopped: no intel/llvm with the CUDA target", "python3 tools/intel_llvm_build.py --contrib")
        d = build_intel_llvm(a, contrib=True)
    if not any((d / "lib").glob("libur_adapter_cuda.so*")):
        fail(f"{d}: this intel/llvm has no CUDA target", "build one: python3 tools/intel_llvm_build.py --contrib")
    comp = free_compiler(d, "contrib")
    # every maker's GPUs on this PC: the NVIDIA ones' code and cuBLAS, the Intel ones' oneMKL
    comp["cuda_archs"] = cuda_archs()
    comp["onemkl"] = intel_gpu_present()
    if comp["cuda_archs"] and not cuda_toolkits():
        fail("NVIDIA's CUDA toolkit is missing (the contrib build makes the NVIDIA GPUs' code with it, and their "
             "dense products go to its cuBLAS)", "install it: sudo apt install nvidia-cuda-toolkit")
    return comp, d


def choose_compiler(a, gpu: dict) -> dict:
    """The SYCL compiler the engine is built with, for the build mode (--license; the order in docs/XE.md, "The SYCL
    compiler").  The mode, --intel-llvm DIR and an accepted build without XMX are kept in the settings for the next
    runs."""
    st = load_settings()
    mode = license_mode(a)
    pci = gpu["pci"]
    allow_no_xmx = a.allow_no_xmx or bool(st.get("allow_no_xmx"))
    kept = {}                                          # the intel/llvm folder to keep in the settings
    if mode == "contrib-icpx":
        comp = icpx_compiler()
        if comp is None:
            fail("--license contrib-icpx builds with Intel's icpx, which is not installed", ICPX_HOW)
    elif mode == "contrib" and cuda_archs():
        comp, d = contrib_compiler(a, st)
        kept = {"intel_llvm_contrib": str(d)}
    else:
        # free, and contrib on a PC without NVIDIA GPUs: no CUDA target is needed, so the free mode's compiler
        llvm_dir = a.intel_llvm or st.get("intel_llvm")
        comp = None
        if llvm_dir:
            d = Path(llvm_dir).expanduser().resolve()
            if not (d / "bin" / "clang++").exists():
                fail(f"--intel-llvm {d}: no bin/clang++ there", "give the folder intel/llvm was installed to")
            comp = free_compiler(d)
            if not new_enough(comp):
                fail(f"--intel-llvm {d}: older than intel/llvm 7", FREE_HOW)
        elif not a.intel_llvm_build and (c := os_compiler()) is not None:
            if new_enough(c):
                comp = c
            else:
                say(f"  {Path(c['cxx']).name} ({compiler_version(c['cxx'], c['env'])}) is older than intel/llvm 7, "
                    "which the engine needs")
        if comp is None:
            if not a.intel_llvm_build and ask("Build intel/llvm here now (13 minutes on 28 threads, 3.5 GB)?",
                                              ["y", "n"], "y", a.yes) != "y":
                fail("no SYCL compiler for the engine", FREE_HOW + "\n       (--license contrib-icpx builds with "
                     "Intel's icpx, which is not free software)")
            comp = free_compiler(build_intel_llvm(a))
            llvm_dir = str(INTEL_LLVM_DEFAULT)
        kept = {"intel_llvm": llvm_dir} if llvm_dir else {}
        if mode == "contrib":
            comp["license"], comp["onemkl"] = "contrib", intel_gpu_present()
    if comp["onemkl"] and mkl_root() is None:
        fail(f"--license {mode} hands the Intel GPU's dense matrix products to oneMKL, which is not installed", MKL_HOW)
    if gpu.get("vendor") == "nvidia":
        # the tensor cores through cuBLAS and joint_matrix: the XMX question is Intel's
        xmx = False
    else:
        probed = probe_gpu(comp, pci)
        if probed == {}:
            # no compiler makes the engine run on a GPU its driver does not list (seen on openSUSE Leap 16, whose
            # compute-runtime 25.18 lists no Arc Pro B70)
            fail(f"the SYCL runtime of {Path(comp['cxx']).name} lists no GPU at PCI {pci}" if pci else
                 f"the SYCL runtime of {Path(comp['cxx']).name} lists no GPU",
                 "the GPU's Level Zero driver (libze-intel-gpu1, Intel's compute-runtime) is missing or too old for "
                 "this GPU: install a newer one (docs/XE.md#packages) and run setup again")
        xmx = has_xmx(probed)
        if not xmx and not allow_no_xmx:
            say(f"\n  {Path(comp['cxx']).name} ({compiler_version(comp['cxx'], comp['env'])}) gives this GPU no XMX "
                "(its matrix engines): the engine would run, but its prompt processing about 1.6 times slower.")
            choices = {"1": "build without XMX", "2": "stop"}
            for k, v in choices.items():
                say(f"  {k}) {v}")
            if ask("Which?", list(choices), "2", a.yes) == "2":
                fail("stopped: no XMX for this GPU", "accept the slower prompt path (--allow-no-xmx)" +
                     ("; or --license contrib-icpx (Intel's icpx)" if mode != "contrib-icpx" else ""))
            allow_no_xmx = True
    st = {k: v for k, v in load_settings().items() if k != "nonfree"}
    save_settings({**st, "license": mode, "allow_no_xmx": allow_no_xmx, **kept})
    comp["version"] = compiler_version(comp["cxx"], comp["env"])
    comp["xmx"] = xmx
    what = ("NVIDIA's tensor cores" if gpu.get("vendor") == "nvidia" else
            "XMX" if xmx else "no XMX: the slower prompt path")
    ok(f"SYCL compiler: {comp['version']} ({mode}; {what}" +
       (f"; NVIDIA code for {', '.join(comp['cuda_archs'])}" if comp["cuda_archs"] else "") + ")")
    return comp


def check_build_tools(comp: dict, sycl_vision: bool) -> None:
    """What the build needs besides the SYCL compiler: the Level Zero headers for the engine, g++ for the CPU image
    encoder, glslc and the Vulkan headers for the Vulkan one, and oneMKL for the SYCL one (ggml-sycl, contrib-icpx).
    Setup does not install them (no apt or sudo from setup): it says what is missing."""
    missing = []
    if shutil.which("g++") is None:
        missing.append("build-essential")
    if not Path("/usr/include/level_zero/ze_api.h").exists():
        missing.append("libze-dev (the Level Zero headers the engine asks the GPU through)")
    if sycl_vision and not Path((oneapi_env() or {}).get("MKLROOT", "")).is_dir():
        missing.append("oneMKL for the SYCL image encoder (intel-oneapi-mkl-sycl-devel from Intel's apt repository)")
    if missing:
        fail("missing: " + ", ".join(missing), "install them, then run it again (docs/XE.md lists the packages)")


def vulkan_missing() -> list:
    """The packages the Vulkan image encoder (ggml-vulkan) needs that are not installed."""
    need = []
    if shutil.which("glslc") is None:
        need.append("glslc")
    if not Path("/usr/include/vulkan/vulkan.h").exists():
        need.append("libvulkan-dev")
    if not Path("/usr/include/spirv/unified1/spirv.hpp").exists():
        need.append("spirv-headers")
    return need


def cmake_build(src, bdir, target, defs, env=None):
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install cmake ninja")
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    # the tools on PATH too: ggml's Vulkan backend configures its shader generator as a separate CMake project
    # (ExternalProject) that asks for Ninja by name, which fails on a clean system where only pip's ninja exists
    env = dict(env or os.environ)
    env["PATH"] = os.pathsep.join(dict.fromkeys([str(Path(ninja).parent), str(Path(cmake).parent),
                                                 env.get("PATH", "")]))
    # a SYCL translation unit takes several GB to compile: as many jobs as the RAM allows, at most half the threads
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(ram_gb() // 12)))
    run(conf, env=env)
    run([cmake, "--build", str(bdir), "--target", target, "-j", str(jobs)], env=env)


ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/main/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def install_binary(src: Path, dst: Path) -> None:
    """Replace a running executable by renaming a complete copy over its old inode."""
    fd, name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".new", dir=dst.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def engine_is_xe(meta: dict) -> bool:
    """An engine in engine/ is this port's only if setup compiled it here for Xe: anything else there is upstream's
    CUDA build (a ready-made download or an older copy of Strata) and cannot run on an Intel GPU."""
    return meta.get("source") == "local" and meta.get("backend") == "xe"


def gpu_encoder(comp: dict) -> str:
    """The GPU image encoder this build makes: ggml-sycl ("gpu") in the contrib-icpx mode with oneMKL, else Vulkan."""
    env = oneapi_env() or {}
    if comp.get("license") == "contrib-icpx" and Path(env.get("MKLROOT", "")).is_dir():
        return "gpu"
    return "vulkan"


def build_engine(visions, llama, comp: dict) -> Path:
    """Compile the engine with `comp` (choose_compiler) and the image encoders in `visions` ("cpu", "vulkan", "gpu" =
    SYCL); the results go to engine/.  A compiled engine or encoder whose source files or compiler changed since (a
    `git pull`, another intel/llvm) is compiled again.  Each encoder has its own binary, so installing one never
    replaces the other."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    xe = engine_is_xe(meta)
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    used = {"cxx": comp["cxx"], "version": comp["version"], "kind": comp["kind"], "xmx": comp["xmx"],
            "license": comp["license"], "cuda_archs": comp["cuda_archs"], "onemkl": comp["onemkl"]}
    engine_ok = xe and (eng / EXE).exists() and meta.get("src") == src and meta.get("compiler") == used
    built = dict(meta.get("vision_srcs") or {}) if xe else {}
    missing = [v for v in dict.fromkeys(visions)
               if v in VEXE and not ((eng / VEXE[v]).exists() and built.get(v) == vsrc)]
    if engine_ok and not missing:
        ok("engine already built for this PC")
        return eng
    check_build_tools(comp, "gpu" in missing)
    if "vulkan" in missing and (need := vulkan_missing()):
        fail("the Vulkan image encoder needs: " + ", ".join(need), "install them: sudo apt install " + " ".join(need))
    if not engine_ok:
        bdir = ROOT / "build-xe"
        cache = bdir / "CMakeCache.txt"
        if cache.exists() and f"CMAKE_CXX_COMPILER:FILEPATH={comp['cxx']}" not in cache.read_text(errors="replace"):
            shutil.rmtree(bdir)                        # CMake keeps the compiler it configured with
        say("  The engine's source changed: compiling it again (only what changed) ..." if xe and (eng / EXE).exists()
            and bdir.exists() else "  Compiling the Strata engine for the GPU (20-40 minutes, once) ...")
        cmake_build(ROOT, bdir, "strata",
                    [f"-DCMAKE_CXX_COMPILER={comp['cxx']}", f"-DCMAKE_C_COMPILER={comp['cc']}",
                     f"-DSTRATA_LICENSE={comp['license']}",
                     f"-DSTRATA_CUDA_ARCHS={'auto' if comp['cuda_archs'] else ''}",
                     f"-DSTRATA_ONEMKL={'ON' if comp['onemkl'] else 'OFF'}",
                     *([f"-DMKL_ROOT={mkl_root()}"] if comp["onemkl"] else []), "-DSTRATA_ENABLE_XE=ON",
                     "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_PORTABLE=ON",
                     f"-DSTRATA_GGML_DIR={llama}"], comp["env"])
        install_binary(bdir / EXE, eng / EXE)
    for v in missing:
        llama_def = f"-DLLAMA_DIR={llama}"
        if v == "gpu":                                 # ggml-sycl (oneMKL), a separate build from the CPU encoder
            say("  Compiling the image encoder for the GPU (SYCL; 10-20 minutes, once) ...")
            bdir = ROOT / "build-vision-sycl"
            cmake_build(ROOT / "tools" / "vision", bdir, "strata-vision",
                        [llama_def, "-DSTRATA_LICENSE=contrib-icpx", "-DSTRATA_VISION_SYCL=ON", "-DSTRATA_PORTABLE=ON",
                         "-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"], oneapi_env())
        elif v == "vulkan":
            say("  Compiling the image encoder for the GPU (Vulkan; 5-15 minutes, once) ...")
            bdir = ROOT / "build-vision-vulkan"
            cmake_build(ROOT / "tools" / "vision", bdir, "strata-vision",
                        [llama_def, "-DSTRATA_VISION_VULKAN=ON", "-DSTRATA_PORTABLE=ON"])
        else:
            say("  Compiling the image encoder for the CPU ...")
            bdir = ROOT / "build-vision-cpu"
            cmake_build(ROOT / "tools" / "vision", bdir, "strata-vision", [llama_def, "-DSTRATA_PORTABLE=ON"])
        install_binary(bdir / "bin" / "strata-vision", eng / VEXE[v])
        built[v] = vsrc
    # the engine's runtime only: the SYCL image encoder's oneAPI folders go into the configs that use it (main)
    stamp.write_text(json.dumps({"source": "local", "backend": "xe", "version": source_version(), "compiler": used,
                                 "lib_dirs": comp["lib_dirs"], "src": src, "vision_srcs": built}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


def sycl_encoder_onednn() -> bool:
    """Whether the SYCL image encoder was built with oneDNN: ggml-sycl takes it when it finds it installed."""
    ninja = ROOT / "build-vision-sycl" / "build.ninja"
    return ninja.exists() and "GGML_SYCL_DNNL=1" in ninja.read_text(errors="replace")


def recorded_compiler(meta: dict) -> dict:
    """The compiler an installed engine was built with, for building it again after a `git pull` without asking.  An
    engine from before the free build (no record) was built with icpx: it stays so, as --license contrib-icpx."""
    rec = meta.get("compiler") or {"kind": "icpx", "xmx": True}
    if rec["kind"] == "intel-llvm":
        comp = free_compiler(Path(rec["cxx"]).parent.parent, rec.get("license", "free"))
        if comp["license"] == "contrib":
            comp["cuda_archs"] = cuda_archs()
            comp["onemkl"] = intel_gpu_present()
    elif rec["kind"] == "os":
        comp = os_compiler()
        if comp is not None and not new_enough(comp):
            fail(f"the engine was built with {Path(comp['cxx']).name}, older than intel/llvm 7 (which it now needs)",
                 "run ./setup.sh --setup to build it with intel/llvm 7 or later")
    else:
        comp = icpx_compiler()
        if comp is not None and "compiler" not in meta:
            save_settings({**{k: v for k, v in load_settings().items() if k != "nonfree"}, "license": "contrib-icpx"})
            say("  the installed engine was built with icpx (not free software): building it so again "
                "(--license contrib-icpx); ./setup.sh --setup --license free chooses a free compiler")
    if comp is None or not Path(comp["cxx"]).exists():
        fail("the compiler the engine was built with is gone", "run ./setup.sh --setup to choose one")
    comp["version"] = compiler_version(comp["cxx"], comp["env"])
    comp["xmx"] = bool(rec.get("xmx"))
    return comp


def update_installed_engine() -> None:
    """An engine or image encoder whose sources changed since it was compiled (a `git pull`) is compiled again before
    the model starts.  If that fails, a previous Xe engine is kept and starts as before; upstream's CUDA engine cannot
    run here."""
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if not info.exists() or not (eng / EXE).exists():
        return
    meta = json.loads(info.read_text())
    visions = list((meta.get("vision_srcs") or {}).keys())
    vsrc = source_hash(VISION_SOURCES)
    if engine_is_xe(meta) and meta.get("src") == source_hash(ENGINE_SOURCES) and \
            all(meta["vision_srcs"][v] == vsrc for v in visions):
        return
    try:
        build_engine(visions, get_llama_cpp(), recorded_compiler(meta))
    except (Exception, SystemExit) as e:
        why = "" if isinstance(e, SystemExit) else f" ({e})"
        if not engine_is_xe(meta):
            fail(f"the installed engine is not the Xe build and compiling it failed{why}", "run ./setup.sh --setup")
        warn(f"could not compile the updated engine{why}: starting the installed one")


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`XeStrata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """Setup recommends, it never forces (upstream #406).  A choice setup expects to fail or run badly is said plainly
    (msg), then asked (default n), or with --yes taken as consent when it was asked for explicitly (a flag such as
    --model): --yes alone keeps the stop (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def ctx_ram_need(model, ctx, low_ram=False):
    """The RAM (GB) setup estimates for a long context with IQ3_XXS / IQ3_S: their experts + the context's 8-bit KV
    cache + 24 GB of room for everything else.  None where the context does not count against RAM by this rule: the
    other sizes, and the low-RAM mode (its KV cache stays in VRAM)."""
    if model not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return MODELS[model]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24


def ram_ctx(model, ram, low_ram=False) -> int:
    """The longest context the RAM rule recommends: 128K, or longer where the estimate fits this PC's RAM.  It is part
    of the recommended default (the smaller of it and the GPU's rule); a longer choice is kept, with a note."""
    return max(c for c in CONTEXTS if c <= 131072 or (ctx_ram_need(model, c, low_ram) or 0) <= ram)


# The low-RAM mode (upstream de159b5 / 872af82): a PC whose GPU holds much of a model's experts but whose RAM cannot
# hold them all.  The engine reads the experts from the model files instead of copying them into RAM
# (--mmap-experts: a native pack's from the GGUF files in place, the AVX-512 Q2_0 pack's from its experts.bin), and
# with --resident-experts copies the ones the GPU does not hold into RAM once.
LOW_RAM_HEADROOM_GB = 10   # RAM beside the experts: the OS, the engine's other buffers, the server


SMALL_CARD_GB = 7.5            # #496: a card under 8 GB gets a tip (an 8 GB card lists 7.99)


def small_card_note(ctx: int, draft_vocab: str | None) -> list[str]:
    """#496 (upstream e1ee248, 4731a9b): what frees VRAM on a card under 8 GB when the start stops with "no VRAM is
    left for the expert cache" (the engine already lowers its own reserve on such a card) - a recommendation, setup
    changes none of it.  (The draft layer stays: the server needs it.)"""
    tips = []
    if ctx > 8192:
        tips.append("an 8K context (a smaller KV cache)")
    if draft_vocab != "en":
        tips.append("--draft-vocab en (a smaller draft head)")
    lines = ["If the start stops with \"no VRAM is left for the expert cache\" (the engine's log says how much is "
             "short):"]
    if tips:
        lines.append("  run ./setup.sh again with " + " and ".join(tips) +
                     ", or close other programs that use the GPU.")
    else:
        lines.append("  close other programs that use the GPU.")
    return lines


def gpu_expert_vram(gpu) -> float:
    """The VRAM (GB) that can hold experts instead of RAM: none on the processor's graphics (its memory is the RAM)."""
    return 0.0 if gpu.get("integrated") else gpu["vram_gb"]


def low_ram_needed(model, ram) -> bool:
    """The model's experts do not fit this PC's RAM with room left for the rest."""
    return ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB


def low_ram_gpu_gb(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how many GB of the model's experts the GPU's cache holds: its VRAM less ~5 GB for the dense weights,
    buffers and a 32K context's KV cache (upstream's figure, not measured on an Arc card: unverified), less the KV
    cache of a longer context (in VRAM in the low-RAM mode: its RAM has no room for KV streaming)."""
    kv_tok = 13 * (576 if kv == "q4_0" else 1056)       # bytes per context token: 12 QSA layers + the draft layer
    longer = max(0, ctx - 32768) * kv_tok / 1e9
    return max(0.0, min(MODELS[model]["arena_gb"], vram_gb - 5 - longer))


def low_ram_gpu_share(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how much of the model's experts the GPU holds."""
    return low_ram_gpu_gb(model, vram_gb, ctx, kv) / MODELS[model]["arena_gb"]


def low_ram_resident(model, ram, vram_gb, ctx=32768, kv="int8") -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM with the usual room beside them, so they are
    copied into RAM once (--resident-experts) instead of being read through the OS file cache (--mmap-experts, which
    a PC this short of RAM keeps re-reading from the SSD)."""
    rest = MODELS[model]["arena_gb"] - low_ram_gpu_gb(model, vram_gb, ctx, kv)
    return ram >= rest + LOW_RAM_HEADROOM_GB


def low_ram_fits(model, ram, vram_gb) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM left beside the rest (as file cache)."""
    return ram - 6 + max(0.0, vram_gb - 5) >= MODELS[model]["arena_gb"]


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name: the family's pattern, or the model's own (UD-IQ4_XS has three shards, not four)."""
    return MODELS.get(model, {}).get("file", fam["file"]).format(q=model, i=i)


def model_shards(fam: dict, model: str) -> int:
    return MODELS.get(model, {}).get("shards", fam.get("shards", 2))


def resident_budget_gib(model, ram, kv_ram_gb=0.0) -> int:
    """UD-Q4_K_XL, UD-IQ4_XS: the GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM (GB,
    ram_gb()) less 24 for the OS, the engine and the file cache the other experts are read through, less a KV cache
    streamed to RAM; at most all of them, at least 8.  A 64 GB PC (~62 GiB): 38, 36 with a 128K context
    (bench/results/2026-10-03-ud-low-ram)."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(MODELS[model]["arena_gb"] / 1.073741824)))


def budget_choice(model, ram, asked) -> float:
    """UD-Q4_K_XL's or UD-IQ4_XS's RAM budget: --resident-budget-gib N as given, else the recommendation
    (resident_budget_gib).  More than the recommendation is kept, with what it risks (setup recommends, it never
    forces)."""
    rec = resident_budget_gib(model, ram)
    if asked is None:
        return rec
    if asked > rec:
        warn(f"a {asked:g} GiB RAM budget is more than setup recommends for this PC ({rec} GiB: the RAM less "
             f"{UNSLOTH_RAM_LEFT_GB} GB for the OS, the engine and the file cache that reads the other experts). Kept "
             "as you chose: the engine clamps it to the RAM it finds free at start (less 8 GiB), and the file cache "
             "gets less room - it may be slower, or run the PC out of RAM under load")
    return int(asked) if asked == int(asked) else asked


def memlock_gib() -> float:
    """The RAM (GiB) this user may lock (ulimit -l): the engine locks its resident copy of the experts so the OS does
    not swap it out; with a lower limit the copy stays unlocked (the engine says so)."""
    import resource
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return float("inf") if soft == resource.RLIM_INFINITY else soft / 2 ** 30


def settings_path() -> Path:
    # its own, not upstream Strata's (~/.config/strata): the two keep their data folders apart
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "xestrata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                if v == o or v.startswith(o + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        cfg_file.write_text(json.dumps(new_cfg, indent=1), encoding="utf-8")


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder
    and of earlier Strata folders on the same drive into the data folder, and points their configs there."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "XeStrata-data"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    for folder in [ROOT, *other_installs(settings)]:
        if folder == dest or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in folder.glob("xestrata-*.json"):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += list(folder.glob("xestrata-*.json"))
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("xestrata-"):]
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), "qwen")
    model = tag[len(FAMILIES[family]["tag"]):].upper()     # unsloth-ud-iq4_xs: UD-IQ4_XS
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    rv = val("--vram-reserve-mib") or ""
    reserve = int(rv) if rv.isdigit() else None
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    return {"family": family, "model": model if model in MODELS else None,
            "context": int(val("--max-context")) if val("--max-context") else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0", "k8v4") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "vision_onednn": ("on" if vis.get("onednn") else "off") if isinstance(vis, dict) and vis.get("gpu") else None,
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"),
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": reserve if reserve is not None and (
                vis is None or reserve != VISION["gpu"]["reserve_mib"]) else None}


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549 (upstream 067bfda): a model's run config (a JSON object with "exe" and "args"). Any other xestrata-*.json
    in the folder (a file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping
    setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a XeStrata model config")
    return False


def installed_configs():
    return [p for p in sorted(ROOT.glob("xestrata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(XeStrata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache)."""
    sel = cfg.get("gpu")
    gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads). It takes about 5-10 minutes; the PC is busy meanwhile.")
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 32768, as the free VRAM allows - about 2x faster on long prompts)."""
    try:
        if engine_is_xe(json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text())):
            return cfg                                 # the Xe engine numbers its versions from 0.1.0 and reads --prefill auto
    except (OSError, ValueError):
        pass
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is updated")
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


# The draft layer's token subsets (data/, written by tools/draft_vocab.py; upstream 6e153c9, 0fc1a1e, cbd0522): cjk
# = with Chinese, Japanese and Korean (the default), en = English and code only (the subset before it), cyrillic =
# English, code and the Cyrillic script.  The ones setup copied before are replaced; a subset made by hand is kept.
DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}


def vision_tokens(asked: int | None, vision: str, earlier: Path) -> int:
    """The most image tokens a picture becomes (the config's vision.max_tokens; upstream #625): --vision-tokens N, else
    what this model's config chose earlier for the same encoder device (a setup run again keeps it), else the
    device's default (VISION).  More is allowed with a note on the time it takes: setup recommends, it does not cap."""
    default = VISION[vision]["max_tokens"]
    if asked is None and earlier.is_file():
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) and bool(v.get("gpu")) == (vision == "gpu") else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if vision == "cpu" and asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    elif asked > VISION["gpu"]["max_tokens"]:
        note = " - more than the encoder's default needs more VRAM and context per picture"
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


def saved_draft_vocab(cfg_path: Path) -> str | None:
    """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
    rewrites the config, and would otherwise put the default subset back."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in DRAFT_VOCABS else None


# the draft head's VRAM per subset (IQ3_S: the largest)
DRAFT_VOCAB_MIB = {"cjk": 348, "cyrillic": 193, "fr": 151, "en": 133}
SMALL_DRAFT_VRAM_GB = 14   # #474: below this the default subset's head can be what does not fit


def draft_vocab_note(vram_gb: float, chosen: str | None) -> list[str]:
    """#474 (upstream f2aaba1): on a card under 14 GB, the default draft subset (cjk, ~348 MiB of VRAM) can be what
    does not fit at the start ("the draft head does not fit"), and the engine's expert cache gets what a smaller one
    leaves.  Setup RECOMMENDS a smaller one here and changes nothing: a subset chosen with --draft-vocab, or kept from
    an earlier install, gets no note.  [] for every other case."""
    if chosen or not 0 < vram_gb < SMALL_DRAFT_VRAM_GB:
        return []
    return [f"Tip for a {vram_gb:.0f} GB card: the draft layer's default token subset (with Chinese, Japanese and "
            f"Korean) needs up to ~{DRAFT_VOCAB_MIB['cjk']} MiB of VRAM.",
            f"  For English and code answers, ./setup.sh --draft-vocab en needs up to ~{DRAFT_VOCAB_MIB['en']} MiB "
            f"(cyrillic: ~{DRAFT_VOCAB_MIB['cyrillic']}) and leaves the rest to the expert cache - and it is the",
            "  fix when the start stops with \"the draft head does not fit\". The model keeps the choice."]


def mtp_corrupt(mtp: Path, env=None) -> bool:
    """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
    which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
    shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
    if not (mtp / "tensors").is_dir():
        return False
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 3


def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
    """The chosen subset into the MTP folder: copied when missing or when another shipped subset is there."""
    new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
                   for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
        what = {"cjk": "with Chinese, Japanese and Korean", "cyrillic": "with the Cyrillic script", "fr": "for French"}
        ok("draft layer: the token subset " + what.get(choice, "for English and code"))
    shutil.copyfile(new, dst)


def update_install(have: list) -> int:
    """#475 (upstream 64987c8): `setup.py --update` (update.sh, after its git pull): what a plain ./setup.sh does to an
    install before it starts the model, without starting it - the Python packages, the engine compiled again when its
    source changed, each installed model's config upgrades and its draft subset.  No question is asked and the model
    files are not touched."""
    have = [p for p in have if model_config(p)]        # #549: an xestrata-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this XeStrata folder yet: run ./setup.sh to set it up - it finds an earlier")
        say("  install's model files next to it and reuses them.")
        return 0
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")
    update_installed_engine()
    for cfg_path in have:
        cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
        if "--mtp" in cfg["args"][:-1]:
            refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    say()
    ok("XeStrata is updated. Start the model with ./setup.sh when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564 (upstream ae6dd61): the settings a start uses, in one line: the config's engine options (the model's file
    paths left out) and the server's own fields, so a change made by hand to xestrata-<model>.json can be checked
    without the log.  The API key itself is never printed."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --expert-profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8095)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            v = ",".join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v
            srv.append(f"{k} {v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, reserve: int | None = None, browser: bool | None = None) -> int:
    """reserve: --vram-reserve-mib given on this start, kept in the model's args from now on (#493).  browser:
    --no-browser (False) or --browser (True) given on this start, kept in the config's "open_browser" (#609)."""
    cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    if reserve is not None:                            # #493 (upstream b9c2d5e): an engine argument, kept in the args
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if browser is not None and (cfg.get("open_browser") is not False) != browser:
        cfg["open_browser"] = browser
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        ok("saved for this model: " + ("the browser opens" if browser else "no browser"))
    cfg_path.touch()                                     # the most recently used model
    if "--mtp" in cfg["args"][:-1]:
        refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8095))]
    found = gpus()
    env = None
    usable = [x for x in found if gpu_problem(x) is None]
    if gpu is not None:                                # --gpu N: this start only, by the card's PCI address
        check_gpus(gpu if isinstance(gpu, list) else [gpu], found)
        g = next(x for x in found if x["index"] == (gpu[0] if isinstance(gpu, list) else gpu))
        env = dict(os.environ, STRATA_GPU_PCI=g["pci"])
        ok("GPU: " + gpu_name(g) + " (this start)")
    elif usable:
        g = next((x for x in usable if x["pci"] == cfg.get("gpu_pci")), usable[0])
        ok("GPU: " + gpu_name(g))
    open_browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false
    if open_browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'
    a_ = cfg["args"]
    if "--mmap-experts" in a_ and "--resident-budget-gib" not in a_ and "--resident-experts" not in a_:
        # #505: the mapped low-RAM mode loads nothing into RAM up front (the server's narrator says the same)
        say(f"  Starting {cfg.get('model_name', 'the model')}: it maps {size} of experts from the model files (the OS "
            "file cache reads them).")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {size} into RAM and locks part of it for the "
            "GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if open_browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    return subprocess.call(cmd, env=env)


def tiny_png(w=64, h=48) -> bytes:
    """A small two-colour PNG, for the warm-up's image request."""
    import zlib
    rows = b"".join(b"\x00" + b"".join(bytes((200, 30, 30) if x < w // 2 else (30, 30, 200)) for x in range(w))
                    for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))   # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def warm_up(cfg_path: Path) -> None:
    """Start the model once and answer a long prompt (and a picture): the GPU code is SPIR-V that the driver compiles
    on first use and keeps in its cache (~/.cache/neo_compiler_cache), so the first real request does not wait for it
    (on the B70 about 30 s more for the first reply, 11 s more to start: bench/results/2026-09-30-xe-setup).
    Best effort: a failure is reported and setup goes on."""
    import base64
    import socket
    import urllib.error
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    log = cfg_path.with_name(cfg_path.stem + "-warmup.log")
    base = f"http://127.0.0.1:{port}"
    say("  Compiling the GPU code for this model: starting it once and asking two questions (1-3 minutes) ...")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                 str(cfg_path), "--port", str(port)], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.time() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"the model stopped while starting (see {log.name})")
                try:
                    urllib.request.urlopen(base + "/v1/models", timeout=5).read()
                    break
                except (urllib.error.URLError, OSError):
                    if time.time() > deadline:
                        raise RuntimeError(f"the model did not start within 15 minutes (see {log.name})")
                    time.sleep(2)
            notes = " ".join(f"Note {i}: the {c} sensor read {i * 7 % 100} at step {i}."
                             for i, c in zip(range(60), ["red", "blue", "green", "amber"] * 15))
            asks = [[{"role": "user", "content": "Summarize these notes in two sentences.\n" + notes}]]
            if cfg.get("vision"):
                url = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
                asks.append([{"role": "user", "content": [{"type": "text", "text": "What colours are in this picture?"},
                                                          {"type": "image_url", "image_url": {"url": url}}]}])
            for messages in asks:
                req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"messages": messages, "max_tokens": 32,
                                                              "temperature": 0}).encode())
                json.loads(urllib.request.urlopen(req, timeout=900).read())
            ok(f"GPU code compiled for this model ({time.time() - t0:.0f} s)")
        except (RuntimeError, OSError, ValueError) as e:
            warn(f"the warm-up did not finish ({e}): the first request will take longer")
        finally:
            try:
                os.killpg(proc.pid, 15)
            except ProcessLookupError:                 # the server and its engine have ended already
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass
                proc.wait()


# upstream #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the
PARALLEL_MAX = 8               # #465: the engine's batch window holds at most 8 requests
PARALLEL_SHARE = 0.2           # #465: the slots' sessions may take this share of the VRAM the expert cache would hold
PARALLEL_HELD = 0.5            # #465: ... and only where the cache still holds this share of the experts beside them
PARALLEL_COST_NOTE = ("parallel N reduces waiting for several users but costs about 10-25% speed per request on this "
                      "card")


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """#465: the VRAM one batch slot's session takes: its KV cache (12 QSA layers; with KV streaming only the 32K
    positions the attention reads stay in VRAM) and the DeltaNet state (~0.17 GB).  Measured (upstream): 0.56 GiB at
    32K int8."""
    kv_tok = 12 * (576 if kv == "q4_0" else 1056)
    return (min(ctx, 32768) if streaming else ctx) * kv_tok / 1e9 + 0.17


def parallel_recommend(vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """#465: how many requests at once ("parallel") to recommend: 0 = none (one at a time).  Only where the experts
    mostly fit in VRAM - the expert cache (the card's VRAM less ~5 GB) still holds PARALLEL_HELD of the model's experts
    beside the slots' sessions, which take at most PARALLEL_SHARE of it, up to 4.  Where the experts mostly run on the
    CPU a batch reads about as many experts as the requests one by one and every slot's VRAM is expert cache lost
    (upstream measured a 12 GB card: a request alone 11-24% slower with 2-4 slots; docs/BATCHING.md)."""
    cache_gb = max(0.0, vram_gb - 5)
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in (2, 3, 4):
        if n * slot <= PARALLEL_SHARE * cache_gb and (cache_gb - n * slot) >= PARALLEL_HELD * arena_gb:
            best = n
    return best


def parallel_note(asked: int | None, vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool) -> list[str]:
    """#465: what setup says about "parallel": the recommendation (or, where it would cost speed, why it is left at
    one), or how the asked count compares with it (kept as asked: recommend, never force)."""
    rec = parallel_recommend(vram_gb, arena_gb, ctx, kv, streaming)
    slot = parallel_slot_gb(ctx, kv, streaming)
    if asked is None or asked <= 1:
        if not rec:
            return [f"Several requests at once: left at one at a time - {PARALLEL_COST_NOTE} (docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together instead of one "
                f"after the other (each takes ~{slot:.1f} GB of VRAM from the expert cache; docs/BATCHING.md)."]
    lines = [f"parallel requests: {asked} at once (each takes ~{slot:.1f} GB of VRAM from the expert cache, "
             f"{asked * slot:.1f} GB in all)"]
    if asked > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec:
        lines.append(f"recommended for this card: one at a time - {PARALLEL_COST_NOTE}; kept as you chose")
    elif asked > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


# user's - a "sampling" or "mcp_servers" block, "cors_origins", "open_browser" - and is kept when setup runs again, as
# are "host" and "api_key" when this run does not give them
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "gpu_pci",
                        "draft_vocab", "vision"})
SETUP_VISION = frozenset({"exe", "mmproj", "model", "gpu", "gpu_pci", "max_tokens", "threads", "onednn", "fallback"})


def carry_over(old: dict, cfg: dict) -> list[str]:
    """Setup run again for an installed model keeps what the user added to its run config: every key setup does not
    write (`SETUP_KEYS`), and in "vision" the keys setup does not write plus an mmproj of their own that still exists.
    `cfg` (the new config) is updated in place; the names of what was kept are returned.  Engine options added by
    hand to "args" are not merged (setup chooses those): `args_dropped` names them."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """The engine options of the earlier run config that the new one has no more (by flag name): options added by
    hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None) -> None:
    """Setup's run config, written over an earlier one for the same model without losing what the user added to it:
    the keys setup does not write are carried over (carry_over), and the earlier file is kept as
    xestrata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry the keys over from when
    this folder has none yet (a copy set up like the last one).  A line says what was kept, one what was not."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg) if old is not None else []
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    if kept and old_path is not None:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.sh: the server with this config; `open_browser` False (--no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    script = ROOT / f"run-{model.lower()}.sh"
    script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                      encoding="utf-8")
    script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ main
def derived_factor(ctx: int, trained: int = 262144) -> float:
    """The automatic extension factor: the final context over the trained one, at least 1 (upstream 04949fe)."""
    return max(1.0, float(ctx) / float(trained))


def resolve_rope(ctx: int, scaling, scale, trained: int = 262144):
    """The rope configuration for the context actually served: (scaling, scale); scaling None = no flags.

    An explicit --rope-scaling / --rope-scale wins, a given factor verbatim.  Past the trained range an omitted method
    is yarn and an omitted factor final / trained; inside it nothing turns on by itself and a chosen method without a
    factor gets 1 (the trained angles).  An explicit none past the trained range is refused, not overridden.
    """
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None
        return scaling, scale if scale is not None else derived_factor(ctx, trained)
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there - the model has never seen those positions, so setup "
                         "refuses the combination. Pick --rope-scaling yarn or linear, or rerun with --context "
                         "262144 or lower")
    return scaling or "yarn", scale if scale is not None else derived_factor(ctx, trained)


def main() -> int:
    if sys.platform != "linux":
        fail("Strata supports Linux only")
    if is_wsl():
        fail("Strata does not support WSL", "run it on a native Linux installation")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--rope-scaling", choices=["none", "linear", "yarn"],
                    help="the RoPE extension for a context past the model's trained 262144: linear (position "
                         "interpolation) or yarn, llama.cpp's types. Omitted with such a context, setup takes yarn; "
                         "none is refused there")
    ap.add_argument("--rope-scale", type=float,
                    help="the extension factor (default: the final context over the trained 262144, at least 1: "
                         "1.5 for 384K, 2 for 512K, 1 inside the trained range)")
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4"],
                    help="KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less "
                         "precise) or k8v4 (8-bit keys and 4-bit values: about three quarters of the memory)")
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (yes = gpu: the encoder on the GPU, the CPU encoder taking over "
                         "when it cannot start; cpu = on the CPU only)")
    ap.add_argument("--vision-onednn", choices=["on", "off"],
                    help="the SYCL image encoder with oneDNN, when built with it: a little faster, but an image's "
                         "embeddings vary slightly from run to run (default off; the server's --vision-onednn "
                         "overrides it for one start)")
    ap.add_argument("--vision-tokens", type=int, metavar="N",
                    help="the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on "
                         "the CPU): more reads small text and charts better, and takes longer to encode; remembered "
                         "for this model")
    ap.add_argument("--experimental-speed-projection", metavar="on|off|GGUF",
                    help="EXPERIMENTAL, off by default: the control vector in "
                         "third_party/nonfree/experimental-speed-projection "
                         "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8095 for a new one)")
    ap.add_argument("--gpu", help="the GPU, numbered in PCI order as --check lists them (default: the one with "
                                  "the most VRAM); the engine takes it by PCI address")
    # upstream's layer split across several GPUs: refused with an explanation, the Xe engine runs on one GPU
    ap.add_argument("--gpus", help=argparse.SUPPRESS)
    ap.add_argument("--layer-split", help=argparse.SUPPRESS)
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default XeStrata-data next to this folder, "
                                       "remembered for every Strata folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with the two shards)")
    ap.add_argument("--license", choices=LICENSES,
                    help="the build mode: free builds with free software only, for Intel GPUs; contrib (the default) "
                         "builds with intel/llvm's CUDA target for Intel and NVIDIA GPUs, and hands the dense matrix "
                         "products to oneMKL and cuBLAS; contrib-icpx builds with Intel oneAPI's icpx and oneMKL, for "
                         "Intel GPUs, with the SYCL image encoder. Kept for later runs")
    ap.add_argument("--intel-llvm", metavar="DIR",
                    help="build the engine with the intel/llvm installed in DIR (bin/clang++); kept for later runs")
    ap.add_argument("--intel-llvm-build", action="store_true",
                    help="build intel/llvm from source into .tools/intel-llvm and use it (13 minutes on 28 threads, "
                         "3.5 GB); a finished build is used again, an interrupted one continued")
    ap.add_argument("--intel-llvm-rebuild", action="store_true", help="with --intel-llvm-build: build it again")
    ap.add_argument("--intel-llvm-keep-build", action="store_true",
                    help="with --intel-llvm-build: keep its build tree so the next update is quicker")
    ap.add_argument("--allow-no-xmx", action="store_true",
                    help="build even when the compiler gives the GPU no XMX (prompt processing about 1.6 times slower)")
    ap.add_argument("--low-ram", choices=["auto", "on", "off", "resident", "mmap"], default="auto",
                    help="read the model's experts from its files instead of copying them all into RAM (for a PC with "
                         "a big GPU and little RAM); auto: when the experts would not fit the RAM. In this mode the "
                         "experts the GPU does not hold are copied into RAM once when they fit (resident), else read "
                         "through the OS file cache (mmap); resident / mmap force one of the two")
    ap.add_argument("--resident-budget-gib", type=float, metavar="N",
                    help="UD-Q4_K_XL, UD-IQ4_XS: the GiB of its experts kept in RAM (default: the RAM less 24 GB, "
                         "~38 on 64 GB; more is kept as you choose, with a note)")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's "
                         "default: 700); the expert cache takes that much less")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="up to N requests decode together (batch slots, opt-in; default: one at a time, the others "
                         "wait). Each slot takes VRAM from the expert cache; setup says what it recommends")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the engine, Python packages and model settings as a start would, without starting "
                         "the model (update.sh runs it after a git pull)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip starting the model once at the end of setup to compile its GPU code (the first "
                         "request then compiles it: ~30 s more)")
    ap.add_argument("--build", action="store_true", help=argparse.SUPPRESS)   # the engine is always compiled here
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--draft-vocab", choices=list(DRAFT_VOCABS),
                    help="the draft layer's tokens: cjk = with Chinese, Japanese and Korean (default), en = English "
                         "and code only (~110 MiB less VRAM), cyrillic = English, code and the Cyrillic script, fr = "
                         "English, code and French (French answers draft more)")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 5-10 minutes), then start the model")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    global ARGS
    ARGS = a
    if a.vision_tokens is not None and a.vision_tokens < 1:
        ap.error("--vision-tokens takes a number of image tokens, 1 or more, e.g. --vision-tokens 768")
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.gpu is not None:                              # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as --check lists them, e.g. --gpu 0, not {a.gpu!r}")
    say("Strata - Qwen3.8-Flash-Next on a normal PC (an Intel GPU + system RAM + CPU)")
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if a.update:                                       # #475: update.sh - never starts the model
        return update_install(have)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.vision_onednn = a.vision_onednn or ch["vision_onednn"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = (parse_gpus(a.gpus, gpus()) if a.gpus else None) or a.gpu
    port = a.port or 8095                              # a new install's port (issue #32: --port for an existing one)
    if have and a.calibrate and not (a.setup or a.model or a.family or a.check):
        update_installed_engine()
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        calibrate_config(pick_cfg)
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                                          reserve=a.vram_reserve_mib, browser=a.browser)
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        update_installed_engine()
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split, reserve=a.vram_reserve_mib,
                         browser=a.browser)
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                         reserve=a.vram_reserve_mib, browser=a.browser)

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    if not found:
        fail("no Intel GPU on the xe or i915 driver and no NVIDIA GPU on NVIDIA's driver found",
             "Strata's Xe engine needs an Intel GPU (Intel Arc, or the processor's graphics) on the xe or i915 kernel "
             "driver, or with --license contrib an NVIDIA GPU on NVIDIA's driver (lspci -k shows the driver)")
    if len(found) > 1 or gpu_problem(found[0]) is not None:
        gpu_table(found)
    gpu_chosen = a.gpu is not None                     # --gpu given: the config names the card
    sel = choose_gpus(a, found)
    a.gpu = sel[0]
    GPU_PICK = a.gpu
    gpu = gpu_info(a.gpu)
    chosen = [gpu]
    mem = "no VRAM of its own (it shares the system RAM)" if gpu.get("integrated") else f"{gpu['vram_gb']:.0f} GB VRAM"
    ok(f"GPU: {gpu['name']}, {mem}, PCI {gpu['pci']}, PCIe link " + (gpu["link"] or "not readable"))
    if gpu["render"] is None or not os.access(gpu["render"], os.R_OK | os.W_OK):
        fail(f"no access to the GPU ({gpu['render'] or 'no render node'})",
             "NVIDIA's driver is not loaded (nvidia-smi says why)" if gpu.get("vendor") == "nvidia" else
             "add yourself to the render group (sudo usermod -aG render $USER), log out and in, and run it again")
    if gpu["bar_gb"] < gpu["vram_gb"] - 0.5:
        warn(f"Resizable BAR looks off: the card shows {gpu['bar_gb']:.1f} GB of its {gpu['vram_gb']:.0f} GB to the CPU. "
             "Copies to the GPU will be slower; turn on Re-Size BAR (and Above 4G Decoding) in the BIOS")
    if gpu.get("integrated"):
        warn("the processor's own graphics: Strata will run, but slowly - its memory is the system RAM, and it has "
             "no matrix engines (XMX) on most processors")
    elif gpu["vram_gb"] < 11:
        warn("less than 12 GB of VRAM: Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    need = min(d["ram_gb"] for d in MODELS.values())
    low_ok = low_ram_fits("IQ1_M", ram, gpu_expert_vram(gpu)) and a.low_ram != "off"   # the smallest model, mapped
    if ram < need - 4 and not a.check and not low_ok:
        # every model keeps ALL its experts in RAM (23+ GB) unless the GPU holds enough of them for the low-RAM mode
        fail(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
             "Strata keeps the model's experts in RAM (23-50 GB) and the GPU holds a copy of the most-used ones: it "
             "needs 32 GB of RAM or more (48 GB for the full model), or a GPU with the VRAM to hold most of them")
    ok(f"RAM: {ram:.0f} GB" if ram >= need - 4 else
       f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)" +
       ("; the GPU's VRAM makes up for it (the low-RAM mode)" if low_ok else ""))
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    if not avx2:
        fail("this CPU has no AVX2; Strata needs at least AVX2")
    if a.check:
        say()
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            vram = gpu_expert_vram(gpu)
            if d.get("budget"):
                fits = f"fits with {resident_budget_gib(m, ram)} GiB of its experts in RAM, the rest read from the SSD"
                verdict = ("EXPERIMENTAL, " if d.get("experimental") else "") + fits if ram >= d["ram_gb"] \
                    else "does not fit"
            elif low_ram_needed(m, ram) and low_ram_fits(m, ram, vram) and a.low_ram != "off":
                verdict = (f"fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, vram):.0f}% of its "
                           "experts, " + ("the rest stays in RAM)" if low_ram_resident(m, ram, vram)
                                          else "the rest is read from the SSD as needed)"))
            say(f"  {m:10s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        say("\nThis PC can run Strata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = list(FAMILIES)
    if a.family:
        family = a.family
    else:
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}" +
                ("   [experimental]" if d.get("experimental") else ""))
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    # an experimental size sorts last (UD-IQ4_XS before UD-Q4_K_XL); --model takes the names as they are
    names = sorted((m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))),
                   key=lambda m: bool(MODELS[m].get("experimental")))
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names))
    vram = gpu_expert_vram(gpu)
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        if d.get("budget"):
            say(f"  {i}) {m} {d['about']}; download {d['download_gb']:.0f} GB, keeps ~"
                f"{resident_budget_gib(m, ram)} GB of its {d['arena_gb']:.0f} GB of experts in RAM{fit}")
            continue
        if low_ram_needed(m, ram) and low_ram_fits(m, ram, vram) and a.low_ram != "off":
            fit = (f"   <- fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, vram):.0f}%, "
                   + ("the rest in RAM)" if low_ram_resident(m, ram, vram) else "the rest from the SSD)"))
        say(f"  {i}) {m:8s} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index("IQ3_XXS") + 1) if ram >= 60 and "IQ3_XXS" in names else "1"
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    budget = None
    if MODELS[model].get("budget"):
        # Unsloth's UD-Q4_K_XL and UD-IQ4_XS: a RAM budget of their experts, the rest read from the GGUF files in place
        # (not the low-RAM mode's choice: the budget is set by the RAM, whatever the GPU holds)
        if MODELS[model].get("experimental"):
            warn(f"{model} is EXPERIMENTAL: part of its experts are read from the SSD while it answers, so it is "
                 "slower than the 2-3-bit models")
        if ram < MODELS[model]["ram_gb"]:
            confirm_risk(f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB: "
                         f"its RAM budget would be {resident_budget_gib(model, ram)} GiB, so nearly every expert is "
                         "read from the SSD while it answers (very slow), and it may run out of RAM",
                         bool(a.model), a.yes, f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC "
                         f"has {ram:.0f} GB", f"choose one of the 2-3-bit models, or --model {model} --yes to "
                         "install it anyway", "  Install it anyway?")
            warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
        budget = budget_choice(model, ram, a.resident_budget_gib)
        ok(f"RAM budget: {budget:g} GiB of {model}'s experts in RAM, the rest read from the model files on the SSD")
        if a.low_ram not in ("auto", "off"):
            warn(f"--low-ram {a.low_ram} does not apply to {model}: it always reads part of its experts from the files")
    elif a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib is for UD-Q4_K_XL and UD-IQ4_XS: {model} keeps all of its experts in RAM or in "
             "the low-RAM mode")
    low_ram = budget is None and (a.low_ram in ("on", "resident", "mmap") or
                                  (a.low_ram == "auto" and low_ram_needed(model, ram) and
                                   low_ram_fits(model, ram, vram)))
    # (the low-RAM mode's variant is decided once the context is known, below)
    if not low_ram and budget is None and ram < MODELS[model]["ram_gb"] - 4:
        # #125: a warning and a question, not a stop: the user may accept paging (asked, "no" by default, so an
        # unattended --yes install stops here unless the size was asked for with --model)
        need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
        confirm_risk(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
                     f"{arena:.0f} GB and must stay in RAM, so Linux will page part of them from disk. Expect it "
                     "to be much slower, and it may not start at all.\n       A smaller size (Q2_0 or IQ2_XS) fits; "
                     "more RAM fixes it.", bool(a.model), a.yes,
                     f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                     f"choose Q2_0 or IQ2_XS, or add RAM; or --model {model} --yes to install it anyway",
                     "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose" + (" (--model)" if a.model else ""))
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x["vram_gb"] for x in chosen)         # each card keeps its layers' KV of the whole context
    rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072
    # upstream #406: the RAM rule is part of the recommendation (the smaller of the two), not a cap over the choice
    rec_ctx = min(rec_ctx, ram_ctx(model, ram, low_ram))
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            need_c = ctx_ram_need(model, c, low_ram)
            say(f"  {i}) {c // 1024}K tokens" + ("   (recommended for your GPU)" if c == rec_ctx else "") +
                ("   (experimental: setup adds rope scaling)" if c > 262144 else "") +
                (f"   (needs ~{need_c:.0f} GB RAM, this PC has {ram:.0f}: may run out of memory)"
                 if c > 131072 and need_c is not None and need_c > ram else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    # upstream #406 #364: a context past the RAM rule (an explicit --context or a pick in the list) is kept, with
    # what it risks.  It used to become 128K: users ran 256K fine where setup's estimate said no.
    need_gb = ctx_ram_need(model, ctx, low_ram)
    if need_gb is not None and ram < need_gb and ctx > 131072:
        warn(f"{ctx // 1024}K with {model} needs ~{need_gb:.0f} GB of RAM by setup's estimate "
             f"({MODELS[model]['arena_gb']:.0f} GB of experts + the context + room for the rest); this PC has "
             f"{ram:.0f}. Kept as you chose: it may be slower or run out of RAM under load. {rec_ctx // 1024}K is the "
             "recommended size.")
    scaling = a.rope_scaling
    if ctx > 262144 and scaling is None and not a.yes:
        say()
        say(f"  A {ctx // 1024}K context runs the model past its trained 262,144 positions: the rotary angles")
        say("  get rescaled (llama.cpp's RoPE extension). yarn keeps the trained angles on the high-frequency")
        say("  pairs and corrects the magnitudes; linear shrinks every angle. Override any time with")
        say("  --rope-scaling.")
        scaling = ask("RoPE extension method?", ["yarn", "linear"], "yarn", a.yes)
    try:
        scaling, rope_scale = resolve_rope(ctx, scaling, a.rope_scale)
    except ValueError as e:
        fail(str(e))
    if scaling is not None:
        origin = ("final context / trained 262144; override with --rope-scale" if a.rope_scale is None
                  else "as requested")
        ok(f"rope scaling: {scaling}, factor {rope_scale:g} ({origin})")
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        say()
        say("  KV cache precision (the model's memory of the conversation):")
        say("  1) 8-bit   (recommended: what every published number was measured with)")
        say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
        say("             documents; long-context lookups (needle tests) still pass")
        kv = ["int8", "q4_0"][int(ask("KV cache?", ["1", "2"], "1", a.yes)) - 1]
    if ctx > 8192:
        ok("KV cache: " + {"int8": "8-bit", "q4_0": "4-bit (Hadamard-rotated)",
                           "k8v4": "8-bit keys, 4-bit values"}[kv])
    if MODELS[model].get("vision", fam.get("vision")) is False:     # UD-IQ4_XS: images, unlike UD-Q4_K_XL
        vision = "none"
        if a.vision not in (None, "no", "none"):
            warn(f"images are not available with {model} yet: off")
    elif a.vision:
        # yes = the GPU encoder, the CPU encoder its fallback.  Adopted without a tolerance on the embeddings (they
        # differ from the CPU encoder's by a few percent; the answers agree): bench/results/*-xe-vision-sycl
        vision = {"yes": "gpu", "no": "none"}.get(a.vision, a.vision)
    else:
        say()
        say("  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a 0.9 GB")
        say("  download; the image encoder runs on the GPU (on the CPU if it cannot start there).")
        vision = "gpu" if ask("Do you want images?", ["y", "n"], "n", a.yes) == "y" else "none"
    ok("images: " + {"none": "off", "gpu": "on (encoder on the GPU, the CPU encoder as fallback)",
                     "cpu": "on (encoder on the CPU)"}[vision])
    # The low-RAM mode's two variants.  resident: the experts the GPU's cache does not hold are copied from the model
    # files into RAM once (--resident-experts; the engine keeps what fits the RAM it finds free and reads the rest
    # from the files).  mmap: they are read through the OS file cache.  The GPU's share: its VRAM less the dense
    # weights and buffers, this context's KV cache and the image encoder's room.
    resident = False
    if low_ram:
        arena = MODELS[model]["arena_gb"]
        vram -= VISION[vision]["reserve_mib"] / 1024 if vision != "none" else 0
        share = low_ram_gpu_share(model, vram, ctx, kv)
        rest = arena - low_ram_gpu_gb(model, vram, ctx, kv)
        resident = a.low_ram == "resident" or (a.low_ram != "mmap" and low_ram_resident(model, ram, vram, ctx, kv))
        if resident:
            ok(f"low-RAM mode: the GPU holds ~{100 * share:.0f}% of {model}'s experts ({arena:.0f} GB) and the other "
               f"~{rest:.0f} GB stay in RAM ({ram:.0f} GB), copied once from the model files")
        else:
            ok(f"low-RAM mode: {model}'s experts ({arena:.0f} GB) are read from the model files through the OS file "
               f"cache instead of a copy in RAM ({ram:.0f} GB); the GPU holds ~{100 * share:.0f}% of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
    if (resident or budget is not None) and memlock_gib() < 4:
        warn("the experts kept in RAM cannot be locked (ulimit -l is small), so the OS may swap them out and it gets "
             "slower: raise the memlock limit for your user (memlock in /etc/security/limits.conf, or "
             "DefaultLimitMEMLOCK in systemd's user.conf) and log in again")
    # EXPERIMENTAL: the experimental-speed-projection control vector (ESP_VECTOR), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in ("qwen", "coder"):                   # the Coder: the same model's residual stream
        if not esp_choice:
            say()
            say("  EXPERIMENTAL - speed projection: a small control vector applied while the model runs (layers 4-44).")
            say("  It changes how the model answers: its package describes it as a refusal-direction projection (the")
            say("  model declines far fewer requests). Off unless you choose it; when on, the web app can switch it off")
            say("  per chat. Details: third_party/nonfree/experimental-speed-projection/README.md")
            esp_choice = "on" if ask("Turn on the experimental speed projection?", ["y", "n"], "n", a.yes) == "y" else "off"
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            esp = ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1") else Path(esp_choice).expanduser().resolve()
            if not esp.is_file() and esp == ESP_VECTOR:   # third_party/nonfree removed: the rest works without it
                warn(f"the experimental speed projection's vector is not here ({esp}): left off. Put the vector "
                     "there, or pass --experimental-speed-projection <its GGUF>")
                esp = None
            elif not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn("the experimental speed projection is made for the original Qwen3.8-Flash-Next, not Swift 1.5: left off"
             if family == "swift" else f"the experimental speed projection is not tested with {model}: left off")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)]
    if not a.gguf_dir and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    have_model = all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    need = (0 if a.gguf_dir or have_model else MODELS[model]["download_gb"]) + 8 + \
        (40 if model == "Q2_0" and avx512 and family == "qwen" else 0) + (1 if vision != "none" else 0)
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB", "use --models-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    comp = choose_compiler(a, gpu)
    enc = gpu_encoder(comp)
    eng = build_engine({"none": [], "gpu": [enc, "cpu"], "cpu": ["cpu"]}[vision], llama, comp)
    # the runtime of the compiler the engine was built with (none for the distribution's), and oneAPI's only when this
    # config uses the SYCL image encoder (oneMKL): a free install never puts oneAPI on the engine's library path
    lib_dirs = list(dict.fromkeys(comp["lib_dirs"] + (oneapi_lib_dirs() if vision == "gpu" and enc == "gpu" else [])))
    ok(f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if not a.gguf_dir:
        missing = [s.name for s in shards if not (s.exists() and done(s))]
        if missing:                                    # #495: files downloaded by hand go here, or --gguf-dir
            say(f"  The model files go in {models_dir}")
            say(f"  Files you already have: put them here with their original names ({', '.join(missing)}), or use "
                "--gguf-dir <their folder>.")
            if hf_endpoint() != HF_DEFAULT:
                say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes and the Coder: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family in ("qwen", "coder") and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    check_shards(shards)
    for s in shards:                                   # the experimental Unsloth file: pinned sizes and SHA-256
        if s.name in fam.get("sha256", {}):
            verify_sha256(s, *fam["sha256"][s.name])
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if not mmproj.exists():
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        ok(f"vision encoder: {mmproj}")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash"], env=env)
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    elif not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
        # (UD-Q4_K_XL: --compat-bf16 - its Q8_0 hyper-connection projections become BF16, the form the engine reads)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             *fam.get("pack_args", [])], env=env)
    ok(f"model prepared: {pack}")
    mtp = (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    corrupt = (rt / "experts.bin").exists() and mtp_corrupt(mtp, env)
    if corrupt:
        say("  The MTP tensors differ from the pinned checkpoint; fetching and rebuilding them.")
    if corrupt or not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    # a setup run again without --draft-vocab keeps the subset this model's config chose before
    draft_vocab = a.draft_vocab or saved_draft_vocab(ROOT / f"xestrata-{tag.lower()}.json")
    refresh_draft_vocab(rt, draft_vocab or "cjk")
    ok(f"MTP draft layer: {rt}")
    for line in draft_vocab_note(gpu_expert_vram(gpu), draft_vocab):   # #474: a recommendation, nothing changes
        say("  " + line)

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(ple),
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if scaling is not None:     # the resolved configuration: the explicit flags, or yarn and the derived factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if low_ram:   # the experts from the model files: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 7.5 KB with 4-bit, so only when it fits.
    kv_ram_gb = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9   # 12 QSA layers + the draft layer
    # k8v4 never streams its KV (the engine refuses --kv-resident with it)
    if kv != "k8v4" and ctx >= 65536 and ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1:
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
    elif kv != "k8v4" and ctx >= 65536:   # #620: say why, so a regenerated config without --kv-resident is no surprise
        ok(f"KV streaming off: it needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
           f"uses, and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts)")
    if budget is not None:     # UD-Q4_K_XL: the experts read from the GGUF in place, the most-used N GiB kept in RAM
        args += ["--resident-budget-gib", f"{budget:g}"]
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
    if a.vram_reserve_mib is not None:                 # #493: VRAM left free for other programs (only when given)
        if "--vram-reserve-mib" in args:
            i = args.index("--vram-reserve-mib") + 1
            if vision == "gpu" and a.vram_reserve_mib < int(args[i]):
                warn(f"--vram-reserve-mib {a.vram_reserve_mib}: the image encoder on the GPU needs ~{args[i]} MiB of "
                     "it; kept as you chose (it may run out of VRAM when it reads a picture)")
            args[i] = str(a.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    if 0 < gpu_expert_vram(gpu) < SMALL_CARD_GB:
        # #496: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when that
        # is what it takes, and says what is short when even that is not enough.  Setup only says what helps.
        for line in small_card_note(ctx, draft_vocab):   # a recommendation: nothing changes
            say("  " + line)
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"xestrata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    if gpu_chosen or sum(1 for x in found if gpu_problem(x) is None) > 1:
        cfg["gpu_pci"] = gpu["pci"]                    # the engine takes its GPU by PCI address (STRATA_GPU_PCI)
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            for i, line in enumerate(parallel_note(a.parallel, gpu_expert_vram(gpu), MODELS[model]["arena_gb"], ctx, kv,
                                                   streaming)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    else:                                              # the opt-in, said once (nothing changes)
        for line in parallel_note(None, gpu_expert_vram(gpu), MODELS[model]["arena_gb"], ctx, kv, streaming):
            say("  " + line)
    if vision != "none":
        vt = vision_tokens(a.vision_tokens, vision, ROOT / f"xestrata-{tag.lower()}.json")
        cpu_enc = {"exe": str(eng / VEXE["cpu"]), "gpu": False,
                   "max_tokens": vt if vision == "cpu" else VISION["cpu"]["max_tokens"],
                   "threads": max(1, (os.cpu_count() or 8) // 2)}
        cfg["vision"] = {"mmproj": str(mmproj), "model": str(shards[0]), **cpu_enc}
        if vision == "gpu":
            onednn = enc == "gpu" and a.vision_onednn == "on"   # oneDNN: ggml-sycl only
            if onednn and not sycl_encoder_onednn():
                warn("the SYCL image encoder was built without oneDNN (not installed): "
                     "--vision-onednn on has no effect")
                onednn = False
            # the GPU encoder takes the card by PCI address: with Vulkan the first GPU can be another card
            cfg["vision"] = {"mmproj": str(mmproj), "model": str(shards[0]), "exe": str(eng / VEXE[enc]),
                             "gpu": True, "gpu_pci": gpu["pci"], "max_tokens": vt,
                             "onednn": onednn,
                             "fallback": cpu_enc}   # the server starts it when the GPU encoder does not start
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    cfg_path = ROOT / f"xestrata-{tag.lower()}.json"
    cal = saved_calibration(cfg)
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None)
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    if not a.no_warmup:
        warm_up(cfg_path)
    # offered only when someone answers: --yes installs and adopted earlier installs are not held up by it
    if cal is None and not a.no_start and not a.yes and ask(
            "Tune Strata for this PC now? It measures a few engine settings (about 5-10 minutes; the PC is busy "
            "meanwhile; later: ./setup.sh --calibrate)", ["y", "n"], "y", a.yes) == "y":
        calibrate_config(cfg_path)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
