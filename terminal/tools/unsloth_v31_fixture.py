"""Header-only fixtures of Unsloth's Qwen3.8-Flash-Next sets at revision d9620119 (2026-11-04).

tests/fixtures/unsloth_v31/headers.json.gz holds, for every set of the repository, each shard's pinned size and SHA-256 and, for
the shards that carry tensors, the split keys and the real tensor directory (name, ggml type id, shape, offset) read from the
first 4 MB of the file with an HTTP range request.  No weights are in it.  `materialize` writes a shard back as a GGUF that holds
only the header (the file ends where the data would begin), which gguf_reader.GGUFFile and tools/iq_pack.py's layout code read.

  python tools/unsloth_v31_fixture.py build <folder of the 4 MB heads, named SET__file.gguf> <unsloth-head-pins.json> <out.json.gz>
"""
from __future__ import annotations

import gzip
import json
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G                                       # noqa: E402

FIXTURE = HERE.parent / "tests" / "fixtures" / "unsloth_v31" / "headers.json.gz"


def load(path: pathlib.Path = FIXTURE) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build(heads: pathlib.Path, pins: pathlib.Path, out: pathlib.Path) -> dict:
    p = json.loads(pathlib.Path(pins).read_text(encoding="utf-8"))
    sets: dict = {}
    for key, v in p["files"].items():
        folder, _, name = key.partition("/")
        if not name or not name.endswith(".gguf") or "-of-" not in name:
            continue
        shard = {"name": name, "bytes": v["bytes"], "sha256": v["sha256"], "header": None}
        head = pathlib.Path(heads) / key.replace("/", "__")
        if head.exists():
            try:
                g = G.GGUFFile(head)
            except ValueError:
                g = None                                      # shard 1: metadata only, longer than the 4 MB read
            if g is not None:
                shard["header"] = {"split": [g.metadata.get("split.no"), g.metadata.get("split.count"),
                                             g.metadata.get("split.tensors.count")],
                                   "data_start": g.data_start,
                                   "tensors": [[t.name, t.type_id, t.shape, t.offset] for t in g.tensors]}
        sets.setdefault(folder, {"revision": p["revision"], "shards": []})["shards"].append(shard)
    doc = {"repo": p["repo"], "revision": p["revision"], "sets": sets}
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(doc, f, separators=(",", ":"), sort_keys=True)
    return doc


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def materialize(shard: dict, path: pathlib.Path, arch: str = "qwen4exp", first: bool = False) -> pathlib.Path:
    """A GGUF v3 file with the shard's tensor directory and no data: it ends at the aligned start of the data section."""
    h = shard["header"]
    no, count, total = h["split"]
    kv = []
    if first:
        kv.append(_str("general.architecture") + struct.pack("<I", 8) + _str(arch))
    for k, v in (("split.no", no), ("split.count", count)):
        kv.append(_str(k) + struct.pack("<IH", 2, v))
    kv.append(_str("split.tensors.count") + struct.pack("<Ii", 5, total))
    out = bytearray(struct.pack("<IIQQ", G.GGUF_MAGIC, 3, len(h["tensors"]), len(kv)))
    for e in kv:
        out += e
    for name, type_id, shape, offset in h["tensors"]:
        out += _str(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", type_id, offset)
    align = 32
    out += b"\0" * ((-len(out)) % align)
    pathlib.Path(path).write_bytes(bytes(out))
    return pathlib.Path(path)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "build":
        d = build(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
        print(len(d["sets"]), "sets")
    else:
        sys.exit(__doc__)
