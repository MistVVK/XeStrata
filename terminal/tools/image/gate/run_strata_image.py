"""tools/image/gate/run_strata_image.py - TEST-ONLY (OWN-5 gate harness, step 5): run the test program strata-image
(src/image/qi21_image_main.cpp) over a prompt x seed set, one process per picture, and write what run_set.py writes for
sd-cli, so compare.py, floor_table.py and speed_table.py read both alike.

    python3 -I -B run_strata_image.py <root> <tag> <W>x<H> <models_dir> <prompt-indices e.g. 0-7> <seeds e.g. 42,1234>
                                      --exe PATH [--dump] [--min-free-gb N] [--threads N]

models_dir holds the three files under their published names (tools/image/qimage.py FILES: the Q8_0 DiT, the VAE, the
text encoder).  The prompts are run_set.py's PROMPTS (the I1/I2 gate set).
Writes <root>/pics/<tag>/p<i>_s<seed>.png, .log and .json (strata-image --report) per picture, with --dump the stage
tensors in <root>/pics/<tag>/dump_p<i>_s<seed>/ (stage_hooks.py names), and <root>/pics/<tag>/results.jsonl: rc,
seconds, maxrss_mib, and from the report open_s, text_encode_s, sampling_s, vae_decode_s, png_s and the per-stage
memory (ours_gb: the parts' own device memory; driver_gb: the driver's in-use peak above the start; host_gb)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from run_set import PROMPTS  # noqa: E402

STAGE_KEYS = {"open": "open_s", "encode": "text_encode_s", "steps": "sampling_s", "decode": "vae_decode_s",
              "png": "png_s"}


def parse_range(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def record(tag: str, size: str, pi: int, seed: int, rc: int, secs: float, maxrss_kib: int, report: dict | None) -> dict:
    rec = {"tag": tag, "backend": "strata-image", "quant": "Q8_0", "size": size, "steps": 8, "prompt": pi,
           "seed": seed, "rc": rc, "seconds": round(secs, 2), "maxrss_mib": maxrss_kib >> 10}
    if report:
        rec["plan"] = report.get("plan")
        for st in report.get("stages", []):
            k = STAGE_KEYS.get(st.get("name"))
            if k:
                rec[k] = round(st["wall_ms"] / 1000.0, 3)
                stage = st["name"]
                for m in ("ours_gb", "driver_gb", "host_gb"):
                    rec[f"{stage}_{m}"] = round(st[m], 3)
    return rec


def main(argv: list[str]) -> int:
    if len(argv) < 6 or "--exe" not in argv:
        print(__doc__)
        return 2
    root, tag, size, models, pidx, seeds = argv[:6]
    root = Path(root).expanduser().resolve()
    exe = argv[argv.index("--exe") + 1]
    min_free = float(argv[argv.index("--min-free-gb") + 1]) if "--min-free-gb" in argv else 10.0
    threads = argv[argv.index("--threads") + 1] if "--threads" in argv else None
    w, _, h = size.partition("x")
    md = Path(models).expanduser()
    files = {"dit": md / "qwen-image-2.1-turbo-Q8_0.gguf", "vae": md / "qwen_image_2.1_vae_bf16.safetensors",
             "te": md / "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"}
    outd = root / "pics" / tag
    outd.mkdir(parents=True, exist_ok=True)
    for pi in parse_range(pidx):
        for seed in [int(s) for s in seeds.split(",") if s.strip()]:
            if shutil.disk_usage(root).free / 1e9 < min_free:
                raise SystemExit("ABORT: disk under the floor")
            key = f"p{pi}_s{seed}"
            rep = outd / f"{key}.json"
            rep.unlink(missing_ok=True)
            cmd = [exe, "--te", str(files["te"]), "--dit", str(files["dit"]), "--vae", str(files["vae"]),
                   "--prompt", PROMPTS[pi], "--width", w, "--height", h, "--seed", str(seed), "--steps", "8",
                   "-o", str(outd / f"{key}.png"), "--report", str(rep)]
            if threads:
                cmd += ["--threads", threads]
            if "--dump" in argv:
                cmd += ["--dump", str(outd / f"dump_{key}")]
            t0 = time.monotonic()
            with open(outd / f"{key}.log", "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                _, status, ru = os.wait4(proc.pid, 0)
            secs = time.monotonic() - t0
            rc = os.waitstatus_to_exitcode(status)
            report = json.loads(rep.read_text()) if rc == 0 and rep.is_file() else None
            rec = record(tag, size, pi, seed, rc, secs, ru.ru_maxrss, report)
            if rc != 0:
                rec["error"] = "\n".join((outd / f"{key}.log").read_text(errors="replace").strip().splitlines()[-8:])
            with open(outd / "results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    print("RUN-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
