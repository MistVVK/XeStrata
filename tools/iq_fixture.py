# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Real rows of every i-quant format and their reference values, for src/kernels/iq_parity.cpp.

    python tools/iq_fixture.py --out logs/iq_fixture && build/xe/iq_parity logs/iq_fixture

For each format the parity test names, the first tensor stored in that format is found in a pinned revision of the
model repository, `--rows` of its rows are fetched with HTTP range requests (no shard is downloaded), and they are
dequantized by llama.cpp's gguf-py (tools/_paths.py finds it).  gguf-py has no Q2_0 (type 42), so Q2_0 rows use the
definition src/artifact/dequant.hpp mirrors from ggml's dequantize_row_q2_0: code {0,1,2,3} -> (code - 1) * d.

Each format writes <NAME>.bin (int32 type, rows, cols, then the raw rows) and <NAME>.f32 (rows * cols float32), and
manifest.json records where every row came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"   # bench/results/2026-09-30-b70/iq2-xs-baseline.json
# the model sizes searched, in order, for a tensor of each format
SIZES = ("IQ2_XS", "IQ3_XXS", "Q2_0", "IQ3_S")
FILE = "{q}/Qwen3.8-Flash-Next-GSQ-RCO-{q}-00001-of-00002.gguf"
HEADER_BYTES = 32 << 20                                   # every shard-1 header fits (11 MB for IQ2_XS)
NAMES = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "Q2_0", "Q3_K")


def url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def fetch(path: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url(path), headers={"Range": f"bytes={start}-{start + length - 1}",
                                                     "User-Agent": "xestrata-iq-fixture"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != length:
        raise RuntimeError(f"{path}: asked for {length} bytes at {start}, got {len(data)}")
    return data


def header(path: str, cache: pathlib.Path) -> GGUFFile:
    local = cache / (path.replace("/", "_") + ".header")
    if not local.exists():
        local.write_bytes(fetch(path, 0, HEADER_BYTES))
    return GGUFFile(local)


def q2_0_reference(raw: bytes, n: int):
    import numpy as np
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)          # (blocks, 1)
    qs = blocks[:, 2:]
    codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, 64).astype(np.float32)
    return ((codes - 1.0) * d).reshape(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/iq_fixture")
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    _paths.add_gguf_py()
    import numpy as np
    from gguf import GGMLQuantizationType, quants

    out = pathlib.Path(a.out)
    cache = out / "headers"
    cache.mkdir(parents=True, exist_ok=True)
    files = {q: FILE.format(q=q) for q in SIZES}
    heads = {q: header(p, cache) for q, p in files.items()}
    manifest = {"repository": REPO, "revision": REVISION, "rows_per_format": a.rows, "formats": {}}
    missing = []
    for name in NAMES:
        found = None
        for q in SIZES:
            for t in heads[q].tensors:
                if t.type_name == name and len(t.shape) >= 2 and t.shape[1] >= a.rows:
                    found = (q, t)
                    break
            if found:
                break
        if not found:
            missing.append(name)
            print(f"{name:8s} no tensor of this format in {', '.join(SIZES)}")
            continue
        q, t = found
        block_elems, block_bytes = BLOCK_GEOMETRY[name]
        cols = t.shape[0]
        row_bytes = cols // block_elems * block_bytes
        start = heads[q].data_start + t.offset
        raw = fetch(files[q], start, a.rows * row_bytes)
        if name == "Q2_0":
            ref = q2_0_reference(raw, a.rows * cols)
        else:
            qt = GGMLQuantizationType[name]
            ref = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), qt).astype(np.float32).reshape(-1)
        if ref.size != a.rows * cols or not np.isfinite(ref).all():
            raise RuntimeError(f"{name}: reference has {ref.size} values (expected {a.rows * cols}) or non-finite ones")
        (out / f"{name}.bin").write_bytes(np.array([t.type_id, a.rows, cols], dtype=np.int32).tobytes() + raw)
        (out / f"{name}.f32").write_bytes(ref.tobytes())
        manifest["formats"][name] = {"file": files[q], "tensor": t.name, "type_id": t.type_id, "rows": [0, a.rows],
                                     "cols": cols, "byte_offset": start, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                     "reference": "definition mirrored from ggml dequantize_row_q2_0" if name == "Q2_0"
                                                  else "gguf-py quants.dequantize"}
        print(f"{name:8s} {t.name} from {q}: {a.rows} x {cols}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
