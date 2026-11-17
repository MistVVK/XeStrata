# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ComfyUI's arms of the image table on one GPU, beside bench_image.py's XeStrata and upstream: the same 1024x1024
image (tools/bench_core.py's IMAGE_REQUEST: its prompt and seed) through ComfyUI's HTTP API, measured as
bench_image.py measures XeStrata's resident engine: each run starts the server, makes a warm-up image of another
prompt (the models into memory, the GPU code compiled; a second one when it fails), then the measured one, and
stops the server.

    COMFY=DIR VENV=.venv-cuda IMAGE_MODEL=qwen-image-turbo [WEIGHTS="int8 bf16"] [PORT=8191] [ROUNDS=2] \\
      [COMFY_ARGS=...] \\
      python .claude/skills/upstream-sync/scripts/bench_comfy.py OUT GPU       # from the repository root

COMFY: the ComfyUI checkout, its models in its models/ folder (diffusion_models, text_encoders, vae).  VENV: its venv
for this GPU (relative to COMFY).  ONEAPI_DEVICE_SELECTOR or CUDA_VISIBLE_DEVICES in the environment pick the GPU.
IMAGE_MODEL: setup's --images name, qwen-image-turbo (8 steps; ComfyUI has its DiT as int8_convrot alone) or
mage-flow-turbo (4 steps; int8_convrot and bf16).  WEIGHTS: the DiT's weights to measure, an arm each ("comfy-int8",
"comfy-bf16"; one the model lacks is left out).  The text encoders are ComfyUI's own files: qwen3vl_8b_int8_convrot
(Qwen-Image), qwen3vl_4b_bf16 (Mage-Flow); PyTorch's attention (--use-pytorch-cross-attention).  COMFY_ARGS: more of
ComfyUI's own options (--reserve-vram 3, for one), recorded in the run's JSON.
OUT/image-runs/GPU-ARM-ROUND.json gets each run's times as bench_image.py writes them (the seconds of the measured
image from the request to its file; sampling from the KSampler's progress line; no VAE time, ComfyUI does not print
it), .txt the server's log and .png the image; sync_collect.py OUT reads them with bench_image.py's.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU")
out, gpu = Path(sys.argv[1]), sys.argv[2]
comfy = Path(B.need("COMFY"))
venv = B.need("VENV")
model = os.environ.get("IMAGE_MODEL") or "qwen-image-turbo"
port = int(os.environ.get("PORT", "8191"))
rounds = int(os.environ.get("ROUNDS", str(B.ROUNDS)))
WARM = "A lighthouse on a cliff at night, oil painting"     # IMAGE_RUNNER's warm-up prompt
extra = (os.environ.get("COMFY_ARGS") or "").split()

# the DiT files by weights, the steps, and the graph
DITS = {"qwen-image-turbo": ({"int8": "qwen_image_2.1_turbo_int8_convrot.safetensors"}, 8),
        "mage-flow-turbo": ({"int8": "mage_flow_turbo_int8_convrot.safetensors",
                             "bf16": "mage_flow_turbo_bf16.safetensors"}, 4)}
if model not in DITS:
    sys.exit(f"IMAGE_MODEL: one of {', '.join(DITS)}")
files, steps = DITS[model]


def graph(dit: str, prompt: str, seed: int, prefix: str) -> dict:
    """ComfyUI's API graph of one 1024x1024 image: euler, the simple schedule, CFG 1 (as both model cards)."""
    sampler = {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
               "denoise": 1.0}
    if model.startswith("mage"):
        return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader",
                      "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "mage", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "mage_flow_vae_bf16.safetensors"}},
                "4": {"class_type": "TextEncodeMageFlowEdit",
                      "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "width": 1024,
                                 "height": 1024, "batch_size": 1}},
                "5": {"class_type": "KSampler",
                      "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                                 "latent_image": ["4", 2], **sampler}},
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 0]}},
                "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}}}
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image",
                             "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
            "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["9", 0],
                             "latent_image": ["10", 0], **sampler}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 0]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}}}


def one(weights: str, rnd: int, stem: str) -> None:
    """A server, its warm-up image and the measured one -> OUT/image-runs/STEM.json, .txt and .png."""
    runs = out / "image-runs"
    runs.mkdir(parents=True, exist_ok=True)
    logf = runs / f"{stem}.txt"
    base = f"http://127.0.0.1:{port}"
    rec = {"gpu": gpu, "arm": f"comfy-{weights}", "round": rnd, "model": model, "steps": steps, "backend": "comfyui",
           "quant": weights, "args": extra}
    with open(logf, "w") as log:
        srv = subprocess.Popen([str(comfy / venv / "bin" / "python"), "main.py", "--listen", "127.0.0.1", "--port",
                                str(port), "--use-pytorch-cross-attention", "--preview-method", "none"] + extra +
                               (["--oneapi-device-selector", os.environ["ONEAPI_DEVICE_SELECTOR"]]
                                if "ONEAPI_DEVICE_SELECTOR" in os.environ else []),
                               cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                try:
                    urllib.request.urlopen(base + "/system_stats", timeout=2)
                    break
                except OSError:
                    time.sleep(1)
            r = {}
            # a second warm-up when the first fails: ComfyUI on the RX 9060 XT runs out of memory in the first image's
            # VAE decode, unloads its models and makes the next image (2026-10-21, 2026-10-23)
            jobs: list[tuple[str, int, str]] = [(WARM, 1, "warm"), (WARM, 2, "warm2"),
                                                (str(B.IMAGE_REQUEST["prompt"]),
                                                 int(cast(int, B.IMAGE_REQUEST["seed"])), "run")]
            for prompt, seed, tag in jobs:
                if tag == "warm2" and "cold_s" in r:
                    continue
                log.flush()
                size0 = logf.stat().st_size
                t0 = time.monotonic()
                req = urllib.request.Request(base + "/prompt", json.dumps(
                    {"prompt": graph(files[weights], prompt, seed, f"{stem}-{tag}")}).encode(),
                    {"Content-Type": "application/json"})
                pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
                while True:
                    h = json.loads(urllib.request.urlopen(f"{base}/history/{pid}", timeout=60).read()).get(pid)
                    if h and (h.get("status", {}).get("completed") or h.get("status", {}).get("status_str") == "error"):
                        break
                    if srv.poll() is not None:
                        h = {"status": {"status_str": "the server ended"}}
                        break
                    time.sleep(0.05)
                wall = time.monotonic() - t0
                log.flush()
                text = logf.read_text(errors="ignore")[size0:]
                status = h.get("status", {}).get("status_str")
                if status != "success":
                    if tag == "warm":
                        r["warm_error"] = status
                        continue
                    r = {"rc": 1, "error": f"{tag}: {status}"}
                    break
                it = re.findall(r"\|\s*(\d+)/(\d+) \[[0-9:]+<[0-9:?]+,\s*([0-9.]+)(s/it|it/s)\]", text)
                samp = None
                if it:
                    n, v, u = int(it[-1][1]), float(it[-1][2]), it[-1][3]
                    samp = round(n * (v if u == "s/it" else 1 / v), 2)
                if tag.startswith("warm"):
                    r["cold_s"] = round(wall, 2)
                    continue
                r.update({"rc": 0, "seconds": round(wall, 2), "text_s": None, "sampling_s": samp, "decode_s": None})
                for img in h.get("outputs", {}).get("7", {}).get("images", []):
                    src = comfy / "output" / img.get("subfolder", "") / img["filename"]
                    if src.exists():
                        shutil.move(str(src), str(runs / f"{stem}.png"))
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
    rec.update(r)
    (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    took = (f"{r['seconds']} s (sampling {r['sampling_s']} s)" if r.get("rc") == 0 else f"failed: {r.get('error')}")
    print(f"{gpu} comfy-{weights} image {rnd}: {took}", flush=True)


arms = [w for w in (os.environ.get("WEIGHTS") or "int8 bf16").split() if w in files]
head = subprocess.run(["git", "-C", str(comfy), "log", "-1", "--format=%h %cs"], capture_output=True, text=True)
print(f"ComfyUI {head.stdout.strip()}, {model}, {', '.join(arms)}", flush=True)
for r in range(1, rounds + 1):
    for w in (arms if r % 2 else arms[::-1]):
        one(w, r, f"{gpu}-comfy-{w}-{r}")
