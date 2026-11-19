#!/usr/bin/env python3
"""Q2-00: Pack Hy4-Flash-Engram-Q8.strata (67.58 GB / 62.94 GiB, 40 rows/4K page, lossless F8_E4M3 + f32 scale)
and Hy4-Flash-Vision.strata (0.62 GB SigLIP2-400M ViT + perceive projector) from tencent/Hy4-flash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

from .strata_format import (
    PAGE_SIZE,
    PROFILE_ENGRAM_SHARD,
    PROFILE_VISION_SHARD,
    StrataWriter,
    verify_strata_file,
)

ENGRAM_SEED_STRIDE = 10007
ENGRAM_ROW_DIM = 96
ENGRAM_ROW_BYTES = 100   # 4 B float32 per-row scale + 96 B F8_E4M3
ROWS_PER_4K_PAGE = 40    # floor(4096 / 100) = 40 whole rows/page (4000 B + 96 B zero pad)
FP8_E4M3_MAX = 448.0


def compute_engram_metadata(model_dir: Path) -> Dict:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tc = cfg["text_config"]
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = index["weight_map"]

    layer_ids_1based = sorted(int(i) for i in tc["engram_layer_ids"])  # [2, 12, 22, 32]
    layer_ids_0based = [i - 1 for i in layer_ids_1based]               # [1, 11, 21, 31]
    n_gram = int(tc["engram_max_ngram_size"])                          # 3
    n_head = int(tc["engram_n_head_per_ngram"])                        # 8
    head_dim = int(tc["engram_n_embed_per_ngram"]) // n_head           # 96
    assert head_dim == ENGRAM_ROW_DIM
    pad_id = int(tc["engram_pad_id"])                                  # 120002
    seed = int(tc.get("engram_seed", 0))
    vocab_size = int(tc["vocab_size"])                                 # 120832

    # Load lookup_table from first engram module in checkpoint (or engram_vocab_map.npz if present)
    npz_path = model_dir / "engram_vocab_map.npz"
    if npz_path.is_file():
        with np.load(npz_path) as z:
            table = z["lookup_table"].astype(np.int64)
            meta = json.loads(bytes(z["metadata"]).decode("utf-8").rstrip("\x00"))
            compressed = int(meta["compressed_vocab_size"])
    else:
        lt_key = f"model.layers.{layer_ids_0based[0]}.engram_layer.lookup_table"
        st_file = model_dir / wm[lt_key]
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            table = f.get_tensor(lt_key).to(torch.int64).numpy()
        compressed = int(table.max()) + 1

    token_map = [int(x) for x in table]
    if len(token_map) < vocab_size:
        token_map.extend(range(len(token_map), vocab_size))

    raw_primes = tc["engram_layer_head_vocab_sizes"]
    offsets: List[int] = []
    primes: List[int] = []
    multipliers: List[int] = []
    module_row_counts: List[int] = []
    base = 0
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    for lid1 in layer_ids_1based:
        groups = raw_primes[str(lid1)]
        flat_p = [int(p) for g in groups for p in g]
        module_rows = sum(flat_p)
        module_row_counts.append(module_rows)
        for p in flat_p:
            offsets.append(base)
            primes.append(p)
            base += p
        rng = np.random.default_rng(seed + ENGRAM_SEED_STRIDE * lid1)
        mults = [int(m) for m in rng.integers(0, bound, size=n_gram) * 2 + 1]
        multipliers.extend(mults)

    return {
        "model_type": "hy_v4_flash",
        "format": "Q8_INT8_SCALE_F32",
        "row_dim": ENGRAM_ROW_DIM,
        "row_bytes": ENGRAM_ROW_BYTES,
        "rows_per_4k_page": ROWS_PER_4K_PAGE,
        "total_rows": base,
        "total_pages": (base + ROWS_PER_4K_PAGE - 1) // ROWS_PER_4K_PAGE,
        "ple_layers": layer_ids_0based,
        "ple_ngram_size": n_gram,
        "ple_heads_per_ngram": n_head,
        "ple_head_dim": head_dim,
        "ple_layer_multipliers": multipliers,
        "ple_head_offsets": offsets,
        "ple_head_vocab_sizes": primes,
        "module_row_counts": module_row_counts,
        "engram_pad_id": pad_id,
        "compressed_vocab_size": compressed,
        "engram_token_map": token_map,
    }


def quantize_engram_rows_gpu(rows_bf16: torch.Tensor) -> np.ndarray:
    """Convert [M, 96] BF16/FP32 tensor -> uint8 [M, 100] (4 B f32 scale + 96 B symmetric int8).
    18x lower quantization error than FP8_E4M3 (cos=0.999981 vs 0.999668) at identical 100 B/row (40 rows/4K page)!
    """
    x = rows_bf16.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)  # [M, 1]
    scale = (amax / 127.0).clamp(min=1e-12)
    q_i8 = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    q_u8 = q_i8.view(torch.uint8)              # [M, 96]
    s_u8 = scale.squeeze(-1).contiguous().view(torch.uint8).reshape(-1, 4)  # [M, 4]
    packed = torch.cat([s_u8, q_u8], dim=1)    # [M, 100]
    return packed.cpu().numpy()


def pack_engram_q8(model_dir: Path, out_path: Path, device: str = "cuda", chunk_rows: int = 400_000) -> Dict:
    t0 = time.time()
    meta = compute_engram_metadata(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = index["weight_map"]

    assert chunk_rows % ROWS_PER_4K_PAGE == 0
    use_dev = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"[Engram-Q8] Packing {meta['total_rows']:,} rows ({meta['total_pages']:,} 4K pages) on {use_dev} -> {out_path}")

    writer = StrataWriter(str(out_path), profile=PROFILE_ENGRAM_SHARD)
    writer.write_manifest(meta)
    sec = writer.begin_streaming_section(section_id=4)  # Section 4 holds the 4K-page-framed Engram table

    carry = np.empty((0, ENGRAM_ROW_BYTES), dtype=np.uint8)
    page_pad = np.zeros((1, PAGE_SIZE - ROWS_PER_4K_PAGE * ENGRAM_ROW_BYTES), dtype=np.uint8)
    rows_written = 0
    pages_written = 0

    for mod_idx, lid0 in enumerate(meta["ple_layers"]):
        key = f"model.layers.{lid0}.engram_layer.multi_head_embedding.embedding.weight"
        st_path = model_dir / wm[key]
        print(f"  Module {mod_idx} (layer {lid0}): streaming {st_path.name} ...")
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            n_rows = sl.get_shape()[0]
            assert n_rows == meta["module_row_counts"][mod_idx], (
                f"Module {mod_idx} row count mismatch: tensor={n_rows} vs config={meta['module_row_counts'][mod_idx]}"
            )
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                t_chunk = sl[start:end].to(use_dev, non_blocking=True)
                packed = quantize_engram_rows_gpu(t_chunk)  # [M, 100]
                if carry.shape[0] > 0:
                    packed = np.concatenate([carry, packed], axis=0)
                n_full_pages = packed.shape[0] // ROWS_PER_4K_PAGE
                if n_full_pages > 0:
                    full_rows = n_full_pages * ROWS_PER_4K_PAGE
                    pages_view = packed[:full_rows].reshape(n_full_pages, ROWS_PER_4K_PAGE * ENGRAM_ROW_BYTES)
                    pad_block = np.broadcast_to(page_pad, (n_full_pages, page_pad.shape[1]))
                    full_4k = np.ascontiguousarray(np.concatenate([pages_view, pad_block], axis=1))
                    sec.write(memoryview(full_4k))
                    pages_written += n_full_pages
                    rows_written += full_rows
                    carry = packed[full_rows:]
                else:
                    carry = packed
                if (end == n_rows) or ((start // chunk_rows) % 40 == 0):
                    dt = time.time() - t0
                    gb = (pages_written * PAGE_SIZE) / 1e9
                    print(f"    layer {lid0}: {end:,}/{n_rows:,} rows | total {gb:.2f} GB ({gb/max(dt,1e-3):.2f} GB/s)", flush=True)

    # Flush final partial page
    if carry.shape[0] > 0:
        last_page = np.zeros(PAGE_SIZE, dtype=np.uint8)
        raw_tail = carry.reshape(-1)
        last_page[: raw_tail.size] = raw_tail
        sec.write(memoryview(last_page))
        pages_written += 1
        rows_written += carry.shape[0]

    assert rows_written == meta["total_rows"], f"Wrote {rows_written} != {meta['total_rows']}"
    assert pages_written == meta["total_pages"], f"Pages {pages_written} != {meta['total_pages']}"
    entry = sec.finish()
    writer.finalize()
    dt = time.time() - t0
    sz = os.path.getsize(out_path)
    print(f"[Engram-Q8] DONE in {dt:.1f}s: {sz/1e9:.2f} GB ({sz/(1024**3):.2f} GiB), section4_sha256={entry.sha256.hex()}")
    info = verify_strata_file(str(out_path), verify_hashes=False)
    return {"path": str(out_path), "size_bytes": sz, "section4_sha256": entry.sha256.hex(), "pages": pages_written}


def pack_vision_shard(model_dir: Path, out_path: Path) -> Dict:
    t0 = time.time()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = index["weight_map"]

    vit_keys = sorted(k for k in wm if k.startswith("vit."))
    print(f"[Vision] Packing {len(vit_keys)} vit.* tensors -> {out_path}")

    # Group by shard file to minimize file opens
    by_shard: Dict[str, List[str]] = {}
    for k in vit_keys:
        by_shard.setdefault(wm[k], []).append(k)

    # First pass: collect ordered tensors into Section 1 buffer
    tensor_index = []
    blobs = bytearray()
    for shard_name, keys in sorted(by_shard.items()):
        with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as f:
            for k in sorted(keys):
                t = f.get_tensor(k)
                # Keep norms/biases/pos_emb in FP32/FP16; weights in FP16 for compact 0.62 GB shard
                if t.ndim == 1 or "norm" in k or "bias" in k or "newline" in k:
                    t_out = t.to(torch.float32).contiguous()
                    dt_str = "F32"
                else:
                    t_out = t.to(torch.float16).contiguous()
                    dt_str = "F16"
                # 64-byte align each tensor inside Section 1
                rem = len(blobs) % 64
                if rem:
                    blobs.extend(b"\x00" * (64 - rem))
                off = len(blobs)
                raw = t_out.view(torch.uint8).numpy().tobytes()
                blobs.extend(raw)
                tensor_index.append({"name": k, "dtype": dt_str, "shape": list(t_out.shape), "offset": off, "bytes": len(raw)})

    manifest = {
        "model_type": "hy_v4_flash_vision",
        "vision_config": cfg["vision_config"],
        "spatial_merge_size": cfg.get("spatial_merge_size", 2),
        "image_token_id": cfg.get("image_token_id", 120120),
        "image_start_token_id": cfg.get("image_start_token_id", 120118),
        "image_end_token_id": cfg.get("image_end_token_id", 120119),
        "image_newline_token_id": cfg.get("image_newline_token_id", 120121),
        "tensors": tensor_index,
    }
    writer = StrataWriter(str(out_path), profile=PROFILE_VISION_SHARD)
    writer.write_manifest(manifest)
    writer.write_section_bytes(1, bytes(blobs))
    writer.finalize()
    verify_strata_file(str(out_path), verify_hashes=True)
    sz = os.path.getsize(out_path)
    print(f"[Vision] DONE in {time.time()-t0:.1f}s: {sz/1e9:.3f} GB ({sz/(1024**3):.3f} GiB), {len(tensor_index)} tensors verified.")
    return {"path": str(out_path), "size_bytes": sz, "n_tensors": len(tensor_index)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", choices=["all", "engram", "vision"], default="all")
    args = ap.parse_args()
    mdir = Path(args.model_dir)
    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "vision"):
        pack_vision_shard(mdir, odir / "Hy4-Flash-Vision.strata")
    if args.only in ("all", "engram"):
        pack_engram_q8(mdir, odir / "Hy4-Flash-Engram-Q8.strata", device=args.device)


if __name__ == "__main__":
    main()
