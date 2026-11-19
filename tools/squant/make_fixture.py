#!/usr/bin/env python3
"""Q1-13: Generate a mixed-code (gu=S2_K, dn=S3_K plus all 8 S_K codes) square [1536, 1536]
.strata v1.0 test fixture with Section 3 sq.rank_map, Section 4 expert bodies, and Section 5 sq.selfcheck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.codecs import (
    CODE_IDS,
    SUPERBLOCK_BYTES,
    dot_sk_q8_1,
    encode_sk,
    expert_body_bytes,
    fwht64_np,
)
from squant.strata_format import (
    PROFILE_TIERED,
    StrataExpertEntry,
    StrataWriter,
    pack_selfcheck_section,
    verify_strata_file,
)


def build_fixture(out_path: Path) -> None:
    rng = np.random.default_rng(20261118)
    n_latent = 1536
    pairs = [
        ("S2_K", "S3_K"),
        ("S4_K", "S4_K"),
        ("S3_K", "S3_K"),
        ("S1.25", "S1.25"),
        ("S1.5", "S1.5"),
        ("S5_K", "S6_K"),
        ("S6_K", "S8_K"),
    ]
    x_latent = rng.standard_normal(n_latent).astype(np.float32)
    x_latent_h = fwht64_np(x_latent)

    expected_outputs = []
    dir_entries = []
    bodies = bytearray()

    for idx, (gu_code, dn_code) in enumerate(pairs):
        w_gu = (rng.standard_normal((2 * n_latent, n_latent)) * 0.04).astype(np.float32)
        w_dn = (rng.standard_normal((n_latent, n_latent)) * 0.04).astype(np.float32)
        sq_xd_dn = (rng.uniform(0.7, 1.3, size=n_latent)).astype(np.float16)

        enc_gu = encode_sk(w_gu, gu_code)
        enc_dn = encode_sk(w_dn, dn_code)
        aligned_b, dn_off = expert_body_bytes(gu_code, dn_code)

        body_start = len(bodies)
        raw_body = sq_xd_dn.tobytes() + enc_gu + enc_dn
        assert len(raw_body) <= aligned_b
        pad = b"\x00" * (aligned_b - len(raw_body))
        bodies.extend(raw_body)
        bodies.extend(pad)

        # Compute reference Q8_1 dot product on first 4 rows of gu and first 4 rows of dn
        sb_gu = SUPERBLOCK_BYTES[gu_code]
        sb_dn = SUPERBLOCK_BYTES[dn_code]
        gu_row_b = 6 * sb_gu
        dn_row_b = 6 * sb_dn
        gu_dots = [dot_sk_q8_1(enc_gu[r * gu_row_b : (r + 1) * gu_row_b], gu_code, x_latent_h) for r in range(4)]
        dn_dots = [dot_sk_q8_1(enc_dn[r * dn_row_b : (r + 1) * dn_row_b], dn_code, x_latent) for r in range(4)]

        dir_entries.append(
            StrataExpertEntry(
                expert_id=idx,
                gu_code_id=CODE_IDS[gu_code],
                dn_code_id=CODE_IDS[dn_code],
                cluster_id=0,
                flags=0,
                body_offset=body_start,
                body_bytes=aligned_b,
                dn_offset=dn_off,
            )
        )
        expected_outputs.append(
            {
                "expert_id": idx,
                "gu_code": gu_code,
                "dn_code": dn_code,
                "gu_dots": gu_dots,
                "dn_dots": dn_dots,
            }
        )

    sec3 = b"".join(e.pack() for e in dir_entries)
    sec5 = pack_selfcheck_section(pairs)

    manifest = {
        "model_type": "hy_v4_flash_fixture",
        "n_latent": n_latent,
        "x_latent_f32_hex": x_latent.tobytes().hex(),
        "expected": expected_outputs,
    }
    writer = StrataWriter(str(out_path), profile=PROFILE_TIERED, engram_shard_sha256=hashlib.sha256(b"fixture").digest())
    writer.write_manifest(manifest)
    writer.write_section_bytes(1, b"\x00" * 4096)
    writer.write_section_bytes(2, b"\x00" * 4096)
    writer.write_section_bytes(3, sec3)
    writer.write_section_bytes(4, bytes(bodies))
    writer.write_section_bytes(5, sec5)
    writer.finalize()
    info = verify_strata_file(str(out_path), verify_hashes=True)
    print(f"Wrote & verified Q1-13 .strata v1.0 fixture -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB), {len(pairs)} mixed-code experts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/hy4_v1_fixture.strata")
    args = ap.parse_args()
    build_fixture(Path(args.out))
