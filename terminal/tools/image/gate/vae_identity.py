"""tools/image/gate/vae_identity.py - TEST-ONLY (OWN-5 gate harness): the row-band decode (what Strata ships) against
the one-pass reference decode, byte for byte, for every fixed latent of the stage sets named; and run to run.

    python -I -B vae_identity.py <strata-vae-decode> <vae.safetensors> <pics root> <latent tag> [<latent tag> ...]

For each <pics root>/<tag>/dump_<name>/latent.bin: decode with --stream off (one pass) and --stream on --repeat 2
(row bands, twice in one process), and with two other band heights (--band-rows 48 and 100: other positions of the
pixels in their GEMM blocks than the default 128 rows); prints the peak memory and time and IDENTICAL / DIFFERENT;
exit 1 if anything differs (gate: all identical at 512, 1024 and 1536x1024, on CUDA and on HIP)."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

OTHER_BAND_ROWS = (48, 100)
LINE = re.compile(r"decode (\d+)x(\d+) run (\d+): ([\d.]+) ms, peak (\d+) MiB(?:.*(identical|DIFFERS))?")


def run(exe, vae, lat, out, extra):
    r = subprocess.run([exe, vae, str(lat), str(out)] + extra, capture_output=True, text=True)
    runs = [m.groups() for m in map(LINE.search, r.stdout.splitlines()) if m]
    return r.returncode, runs, r.stderr


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    exe, vae, root, tags = argv[0], argv[1], Path(argv[2]), argv[3:]
    bad = n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tag in tags:
            for d in sorted((root / tag).glob("dump_p*_s*")):
                name = f"{tag}/{d.name[len('dump_'):]}"
                one, band = Path(tmp) / "one.f32", Path(tmp) / "band.f32"
                rc1, r1, e1 = run(exe, vae, d / "latent.bin", one, ["--stream", "off"])
                rc2, r2, e2 = run(exe, vae, d / "latent.bin", band, ["--stream", "on", "--repeat", "2"])
                n += 1
                if rc1 or rc2 or not r1 or not r2:
                    bad += 1
                    print(f"{name}: FAILED rc {rc1}/{rc2} {(e1 + e2).strip()[-300:]}")
                    continue
                ref = one.read_bytes()
                same = ref == band.read_bytes()
                others = []
                for rows in OTHER_BAND_ROWS:
                    rc3, r3, e3 = run(exe, vae, d / "latent.bin", band, ["--stream", "on", "--band-rows", str(rows)])
                    ok = rc3 == 0 and bool(r3) and band.read_bytes() == ref
                    others.append(f"{rows} rows {'IDENTICAL' if ok else 'DIFFERENT'}")
                    same = same and ok
                rtr = all(g[5] == "identical" for g in r2[1:])
                bad += (not same) or (not rtr)
                print(f"{name} {r1[0][0]}x{r1[0][1]}: one pass {float(r1[0][3]):.0f} ms peak {r1[0][4]} MiB, "
                      f"bands {float(r2[0][3]):.0f} ms peak {r2[0][4]} MiB: "
                      f"{'IDENTICAL' if same else 'DIFFERENT'} ({', '.join(others)}), bands run to run "
                      f"{'identical' if rtr else 'DIFFERENT'}")
    print(f"VAE-IDENTITY {'PASS' if bad == 0 else 'FAIL'}: {n - bad} of {n}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
