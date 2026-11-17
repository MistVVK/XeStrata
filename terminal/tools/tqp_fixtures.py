"""Build the TQP parity fixtures from the real Gyro Coder-L file.

For each of TQP5/6/7, two tensors (one gate/up, one down) x 64 blocks: raw block
bytes + the fork CPU decoder's float32 output (produced by the standalone harness
ref/tqp/tqp_ref_c.c, cross-checked bit-for-bit against tools/tqp_ref.py), plus a
JSON index. Written under tests/data/tqp/.

Run: python3 tools/tqp_fixtures.py [--model PATH] [--harness PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tqp_ref as R

DEFAULT_MODEL = pathlib.Path.home() / "Strata-data/models/Gyro/Coder-L/Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf"
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "tqp"
NBLOCKS = 64
# one gate/up tensor and one down tensor per type, from different layers
TENSORS = {
    "TQP5": ("blk.32.ffn_gate_exps.weight", "blk.32.ffn_down_exps.weight"),
    "TQP6": ("blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"),
    "TQP7": ("blk.16.ffn_gate_exps.weight", "blk.16.ffn_down_exps.weight"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--harness", default="build-ref/tqp_ref_c")
    ap.add_argument("--scratch", default=None, help="dir for the harness outputs (default: a temp dir)")
    args = ap.parse_args()

    model = pathlib.Path(args.model)
    harness = pathlib.Path(args.harness)
    scratch = pathlib.Path(args.scratch) if args.scratch else pathlib.Path("/tmp/tqp-fixtures")
    scratch.mkdir(parents=True, exist_ok=True)
    gguf = R.open_gguf(model)
    table = R.lut()
    index = {"model": model.name, "nblocks": NBLOCKS, "fixtures": []}

    for name, tensors in TENSORS.items():
        k = R.TQP_BITS[name]
        bsize = R.TQP_GEOMETRY[name][1]
        for tensor in tensors:
            info = next(t for t in gguf.tensors if t.name == tensor)
            nblocks_total = info.elements // R.QK_TQP
            first = (nblocks_total // 2) - NBLOCKS // 2      # mid-tensor blocks
            raw = R.tensor_blocks(gguf, tensor, first, NBLOCKS)
            stem = f"{name}-{tensor}"
            bin_path = OUT / f"{stem}.bin"
            f32_path = OUT / f"{stem}.f32"
            bin_path.write_bytes(raw)
            c_out = scratch / f"{stem}.c.f32"
            subprocess.run([str(harness), str(k), str(bin_path), str(c_out)], check=True)
            c_f32 = np.frombuffer(c_out.read_bytes(), dtype=np.float32)
            py_f32 = R.dequant_row(raw, k, table)
            if not np.array_equal(c_f32.view(np.uint32), py_f32.view(np.uint32)):
                bad = int(np.count_nonzero(c_f32.view(np.uint32) != py_f32.view(np.uint32)))
                raise SystemExit(f"{stem}: C harness and tqp_ref.py differ in {bad} of {c_f32.size} floats")
            f32_path.write_bytes(py_f32.tobytes())
            index["fixtures"].append({
                "type": name, "type_id": R.TQP_IDS[name], "k": k,
                "tensor": tensor, "type_shape": info.shape,
                "block_bytes": bsize, "nblocks": NBLOCKS,
                "first_block": first, "byte_offset": gguf.data_start + info.offset + first * bsize,
                "raw": bin_path.name, "f32": f32_path.name,
            })
            print(f"{stem}: {len(raw)} B raw, {c_f32.size} floats, bit-identical")

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"index: {len(index['fixtures'])} fixtures -> {OUT / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
