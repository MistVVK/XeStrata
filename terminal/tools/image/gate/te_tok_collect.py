"""tools/image/gate/te_tok_collect.py - TEST-ONLY (OWN-5 gate harness): stable-diffusion.cpp's own token ids for every
case of te_tok_cases.py, from a stage_hooks.py build: one sd-cli process per prompt with OWN5_EXIT_AFTER_TOKENS=1, so
it stops right after LLMEmbedder built the ids (no model weights are loaded).

    python3 -I -B te_tok_collect.py <sd-cli> <models dir> <cases dir> <out.json>

out.json: [{"name", "prompt_b64", "ids": [...], "weights": [...], "n_system": int} or {..., "error": "<last log line>"}];
the command is the server's (the three model files, 8 steps, Turbo sigmas), at 256x256."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"


def main() -> int:
    exe, models, cdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    man = json.loads((cdir / "cases.json").read_text())
    res, errors = [], 0
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as td:
        for c in man:
            d = Path(td) / c["name"]
            d.mkdir()
            cmd = [exe, "--diffusion-model", str(models / "qwen-image-2.1-turbo-Q8_0.gguf"),
                   "--vae", str(models / "qwen_image_2.1_vae_bf16.safetensors"),
                   "--llm", str(models / "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
                   "--prompt-file", str(cdir / f"{c['name']}.txt"), "--steps", "8", "--cfg-scale", "1.0",
                   "--sampling-method", "euler", "-W", "256", "-H", "256", "--sigmas", SIGMAS, "-s", "42",
                   "-o", str(d / "img_%d.png"), "--diffusion-fa"]
            env = dict(os.environ, OWN5_DUMP_DIR=str(d), OWN5_EXIT_AFTER_TOKENS="1")
            p = subprocess.run(cmd, env=env, cwd=str(d), capture_output=True, text=True, errors="replace", timeout=600)
            r = dict(c)
            if (d / "tokens.bin").is_file():
                r["ids"] = np.fromfile(d / "tokens.bin", dtype="<i4").tolist()
                r["weights"] = [float(x) for x in np.fromfile(d / "tokens.weights", dtype="<f4")]
                r["n_system"] = int((d / "tokens.start").read_text().split()[0])
            else:
                lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()]
                r["error"] = (lines[-1] if lines else f"rc {p.returncode}")[:300]
                r["rc"] = p.returncode
                errors += 1
            res.append(r)
            shutil.rmtree(d, ignore_errors=True)
    out.write_text(json.dumps(res, ensure_ascii=True))
    print(f"{len(res)} cases, {len(res) - errors} with ids, {errors} without (sd.cpp refused or failed)")
    for r in res:
        if "error" in r:
            print(f"  {r['name']}: rc {r['rc']}: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
