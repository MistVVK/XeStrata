"""tools/image/gate/vae_decode_set.py - TEST-ONLY (OWN-5 gate harness): decode the fixed latents of a stage set with
Strata's own VAE decoder (strata-vae-decode, src/image/vae) and write the pictures as sd-cli would, so compare.py and
check_limits.py can put them next to sd.cpp's decode of the same latents.

    python -I -B vae_decode_set.py <strata-vae-decode> <vae.safetensors> <pics root> <latent tag> <out tag> [--repeat N] [--stream off]

<pics root>/<latent tag>/dump_<name>/latent.bin (run_set.py --dump) -> <pics root>/<out tag>/<name>.png (RGBA, sd.cpp's
8-bit conversion) and results.jsonl (decode ms, peak MiB, run-to-run identity when --repeat > 1).
The gate for the VAE stage (limits.py vae_same_tiling, PSNR >= 45 dB and SSIM >= 0.998 on every picture):
    python -I -B compare.py <pics root> <sd.cpp tag> <out tag>   then   check_limits.py vae_same_tiling <cmp json>
where <sd.cpp tag> is run_set.py --latent-from <latent tag> --steps 1 (sd.cpp's VAE on the same latents)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_ref as V  # noqa: E402

LINE = re.compile(r"decode (\d+)x(\d+) run (\d+): ([\d.]+) ms, peak (\d+) MiB \(weights\+work (\d+) MiB\)(.*)")
OPEN = re.compile(r"open ([\d.]+) ms")


def main(argv) -> int:
    if len(argv) < 5:
        print(__doc__)
        return 2
    exe, vae, root, src, dst = argv[0], argv[1], Path(argv[2]), argv[3], argv[4]
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    stream = argv[argv.index("--stream") + 1] if "--stream" in argv else "on"
    out = root / dst
    out.mkdir(parents=True, exist_ok=True)
    rc_all = 0
    with open(out / "results.jsonl", "w", encoding="utf-8") as res, tempfile.TemporaryDirectory() as tmp:
        for d in sorted((root / src).glob("dump_p*_s*")):
            name = d.name[len("dump_"):]
            f32 = Path(tmp) / f"{name}.f32"
            r = subprocess.run([exe, vae, str(d / "latent.bin"), str(f32), "--repeat", str(repeat),
                                "--stream", stream],
                               capture_output=True, text=True)
            runs = [m.groups() for m in map(LINE.match, r.stdout.splitlines()) if m]
            opened = [float(m.group(1)) for m in map(OPEN.match, r.stdout.splitlines()) if m]
            rec = {"name": name, "rc": r.returncode, "open_ms": opened[0] if opened else None, "runs": [
                {"ms": float(g[3]), "peak_mib": int(g[4]), "weights_work_mib": int(g[5]),
                 "identical_to_run0": None if g[2] == "0" else "identical" in g[6]} for g in runs]}
            if r.returncode != 0 or not runs:
                rec["stderr"] = r.stderr[-2000:]
                rc_all = 1
                res.write(json.dumps(rec) + "\n")
                print(f"{name}: FAILED rc {r.returncode} {r.stderr.strip()[-300:]}")
                continue
            W, H = int(runs[0][0]), int(runs[0][1])
            img = np.fromfile(f32, dtype=np.float32).reshape(4, H, W).transpose(1, 2, 0)
            Image.fromarray(V.to_u8(img), "RGBA").save(out / f"{name}.png")
            res.write(json.dumps(rec) + "\n")
            print(f"{name}: open {rec['open_ms']} ms, decode " + ", ".join(f"{x['ms']:.0f} ms" for x in rec["runs"]) +
                  f", peak {rec['runs'][0]['peak_mib']} MiB" +
                  ("" if repeat == 1 else f", run to run {'identical' if all(x['identical_to_run0'] for x in rec['runs'][1:]) else 'DIFFERENT'}"))
            if repeat > 1 and not all(x["identical_to_run0"] for x in rec["runs"][1:]):
                rc_all = 1
    return rc_all


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
