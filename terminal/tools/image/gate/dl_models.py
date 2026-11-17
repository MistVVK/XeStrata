"""OWN-5: download Qwen-Image-2.1-Turbo's files at the pinned revisions (tools/image/qimage.py FILES/REVISIONS) and
check size + sha256, as setup does.  Uses curl (resumable).

    python3 -I -B dl_models.py <root> <models_dir> <quant> [--min-free-gb N] [--check-only]"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    a = sys.argv[1:]
    root, models, quant = Path(a[0]).expanduser(), Path(a[1]).expanduser(), a[2]
    min_free = float(a[a.index("--min-free-gb") + 1]) if "--min-free-gb" in a else 15.0
    sys.path.insert(0, str(root / "repo" / "tools" / "image"))
    import qimage as QI
    models.mkdir(parents=True, exist_ok=True)
    names = (quant, "vae", "llm")
    need = sum(QI.FILES[n][2] for n in names if not (models / Path(QI.FILES[n][1]).name).is_file()) / 1e9
    free = shutil.disk_usage(models).free / 1e9
    print(f"free {free:.1f} GB, to download about {need:.1f} GB, floor {min_free} GB", flush=True)
    if "--check-only" not in a and free - need < min_free:
        raise SystemExit("ABORT: the download would go under the disk floor")
    for n in names:
        repo, path, size, sha = QI.FILES[n]
        dst = models / Path(path).name
        if "--check-only" not in a and not (dst.is_file() and dst.stat().st_size == size):
            r = subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-s", "-S", "-o", str(dst),
                                QI.file_url(n)])
            if r.returncode:
                raise SystemExit(f"curl failed for {path} ({r.returncode})")
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        ok = dst.stat().st_size == size and h.hexdigest() == sha
        print(f"{'OK ' if ok else 'BAD'} {dst.name} {dst.stat().st_size} {h.hexdigest()}", flush=True)
        if not ok:
            raise SystemExit("sha256 / size mismatch")
    print(f"free after {shutil.disk_usage(models).free / 1e9:.1f} GB")
    print("DL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
