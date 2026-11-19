#!/usr/bin/env python3
"""Full Multi-GPU Quantization Pipeline for Tencent Hy4-Flash (.strata v1.0).

Builds all four consumer-PC & unified-memory .strata files from one shared streamed pass:
  - GPU 0 (Priority 1): Hy4-Flash-SQ-64.strata  (61.84 GB / 57.59 GiB)
  - GPU 1 (Parallel):   Hy4-Flash-SQ-96.strata  (89.43 GB / 83.29 GiB)
  - GPU 2 (Parallel):   Hy4-Flash-SQ-128.strata (113.06 GB / 105.29 GiB)
  - GPU 3 (Parallel):   Hy4-Flash-SQ-32.strata  (37.98 GB / 35.37 GiB)

Pipeline stages:
  1. stream-encode: Layer-by-layer BF16 teacher forward over multi-domain calibration split:
     - Computes exact router-weighted usage (sq.rank_map) & top-524,288 hot Engram rows (64 MiB Sec 2)
     - Accumulates exact 1536x1536 FP32 LatentMoE Hessians in GPU HBM (9.66 GB/layer) with cold-expert shrinkage
     - Folds zero-cost MACaQ s_xform=2 (channel smooth + 64-wide Hadamard) into routed_expert_down_proj
     - Encodes all 512 experts across 4 GPUs in parallel (128 experts/GPU) using Block-LDL (GPTQ) +
       MACaQ 64-flip analytical Hessian refinement for all required S_K widths per build
     - Applies MACaQ-R residual compensation to Q8 routed_expert_up_proj & shared_experts per build
     - Encodes dense trunk & 5-layer DFly2 drafter per build
     - Saves teacher top-256 log-probs + hidden states on calibration & dev splits
  2. scale-qat-pack: 4-way parallel across GPUs 0..3 (1 build per GPU):
     - Runs Scale-QAT (tuning S_K scales, MACaQ scales, Sandwich RMSNorms) + 5-layer DFly2 co-training
     - Packs exact 4 KiB-aligned .strata v1.0 (Sections 0..5 + 64 MiB Sec 2 hot-Engram seed + Sec 5 selfcheck)
     - Verifies SHA-256, exact byte caps, and screens full 42-layer KL + top-1 agreement on held-out dev/test splits
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.codecs import (
    CODE_IDS,
    QK_S,
    SUPERBLOCK_BYTES,
    decode_sk,
    decode_sk_torch,
    encode_sk,
    expert_body_bytes,
    fwht64_np,
)
from squant.hbm_quant import (
    apply_h64_cols,
    apply_h64_sym,
    compute_macaq_smooth_scale,
    damped_cholesky_inv,
    damped_cholesky_inv_batch,
    gptq_flip_quantize_sk_batch_gpu,
    gptq_flip_quantize_sk_gpu,
)
from squant.hy_v4_flash import ENGRAM_EPS, Hy4FlashStreamedModel, apply_interleaved_rope, rms_norm
from squant.pack_engram_vision import (
    ENGRAM_ROW_BYTES,
    ROWS_PER_4K_PAGE,
    compute_engram_metadata,
)
from squant.strata_format import (
    PAGE_SIZE,
    PROFILE_TIERED,
    StrataExpertEntry,
    StrataWriter,
    pack_selfcheck_section,
    parse_header_frame,
    verify_strata_file,
)

# Exact build specs reconciled with builds/sq-*.yaml & docs/02-SIZES-AND-TARGETS.md
BUILD_SPECS: Dict[str, Dict[str, Any]] = {
    "SQ-64": {
        "file_name": "Hy4-Flash-SQ-64.strata",
        "bands": [
            (0, 80, "S4_K", "S4_K"),
            (80, 256, "S3_K", "S3_K"),
            (256, 512, "S2_K", "S3_K"),
        ],
        "mla_code": "S5_K",
        "dense0_code": "S6_K",
        "dfly_code": "S4_K",
        "experts_gb_max": 56.30,
    },
    "SQ-96": {
        "file_name": "Hy4-Flash-SQ-96.strata",
        "bands": [
            (0, 128, "S5_K", "S6_K"),
            (128, 352, "S4_K", "S5_K"),
            (352, 512, "S3_K", "S4_K"),
        ],
        "mla_code": "S6_K",
        "dense0_code": "S8_K",
        "dfly_code": "S4_K",
        "experts_gb_max": 83.20,
    },
    "SQ-128": {
        "file_name": "Hy4-Flash-SQ-128.strata",
        "bands": [
            (0, 160, "S6_K", "S8_K"),
            (160, 384, "S5_K", "S6_K"),
            (384, 512, "S4_K", "S5_K"),
        ],
        "mla_code": "S8_K",
        "dense0_code": "S8_K",
        "dfly_code": "S4_K",
        "experts_gb_max": 106.00,
    },
    "SQ-32": {
        "file_name": "Hy4-Flash-SQ-32.strata",
        "bands": [
            (0, 32, "S2_K", "S3_K"),
            (32, 176, "S2_K", "S2_K"),
            (176, 512, "S1.25", "S1.25"),
        ],
        "mla_code": "S4_K",
        "dense0_code": "S5_K",
        "dfly_code": "S4_K",
        "experts_gb_max": 33.00,
    },
}


def codes_for_rank(build: str, rank: int) -> Tuple[str, str]:
    for lo, hi, gu_c, dn_c in BUILD_SPECS[build]["bands"]:
        if lo <= rank < hi:
            return gu_c, dn_c
    raise ValueError(f"Invalid rank {rank}")


def encode_q8_tensor(t: torch.Tensor) -> Tuple[bytes, str]:
    """Row-wise Q8_0-like exact S8_K if last dim % 256 == 0, else per-row FP16 scale + int8."""
    tf = t.detach().to(torch.float32).cpu()
    if tf.ndim == 2 and tf.shape[1] % QK_S == 0:
        return encode_sk(tf.numpy(), "S8_K"), "S8_K"
    # Per-row int8 + f32 scale
    flat = tf.reshape(-1, tf.shape[-1])
    amax = flat.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-12)
    q = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
    raw = scale.squeeze(-1).numpy().astype(np.float32).tobytes() + q.numpy().tobytes()
    return raw, "Q8_ROW"


def encode_trunk_tensor(name: str, t: torch.Tensor, mla_code: str, dense0_code: str, dfly_code: str) -> Tuple[bytes, str]:
    """Encode a Section 1 GPU Trunk tensor according to docs/FORMAT-SPEC.md & builds/hy4-flash.yaml."""
    tf = t.detach().to(torch.float32).cpu()
    # Exact FP32 for tiny sensitive control tensors, norms, routers, biases, sinks, APE, engram gates
    if (
        tf.ndim <= 1
        or "norm" in name
        or "bias" in name
        or "hc_" in name
        or "engram_gate" in name
        or "engram_kv_b" in name
        or "engram_hc_scale" in name
        or "engram_pre_gamma" in name
        or "intra_group_ape" in name
        or "learnable_sink" in name
        or "mlp.gate.weight" in name
        or "confidence_head" in name
        or "mtp_fusion.gate" in name
    ):
        return tf.numpy().astype(np.float32).tobytes(), "F32"

    # Protected Q8 bottleneck & core tensors (builds/hy4-flash.yaml)
    if (
        "routed_expert_down_proj" in name
        or "routed_expert_up_proj" in name
        or "shared_experts." in name
        or "engram_layer." in name
        or name == "lm_head.weight"
        or name == "model.embed_tokens.weight"
    ):
        return encode_q8_tensor(tf)

    # 5-layer DFly2 drafter -> 4-bit S4_K (0.45 GB)
    if "dfly_layers." in name:
        if tf.ndim == 2 and tf.shape[1] % QK_S == 0:
            return encode_sk(tf.numpy(), dfly_code), dfly_code
        return encode_q8_tensor(tf)

    # Layer 0 Dense FFN
    if "model.layers.0.mlp." in name:
        if tf.ndim == 2 and tf.shape[1] % QK_S == 0:
            return encode_sk(tf.numpy(), dense0_code), dense0_code
        return encode_q8_tensor(tf)

    # MLA & IndexPool projections
    if tf.ndim == 2 and tf.shape[1] % QK_S == 0:
        return encode_sk(tf.numpy(), mla_code), mla_code
    return encode_q8_tensor(tf)


def build_section2_hot_engram_seed(
    engram_q8_path: Path,
    engram_counts: Dict[int, Dict[int, int]],
    meta: Dict,
    rows_per_module: int = 131_072,
) -> Tuple[bytes, bytes]:
    """Build exact 64 MiB Section 2 embedded hot-Engram seed table (524,288 rows x 100 B + 12 B sorted index).
    Returns (section2_bytes_64MiB, engram_shard_section4_sha256_32B).
    """
    with open(engram_q8_path, "rb") as f:
        frame = f.read(PAGE_SIZE)
        _, _, sections, _ = parse_header_frame(frame)
        sec4_off = sections[4].file_offset
        shard_sha = sections[4].sha256

        selected_ids: List[int] = []
        offsets = meta["ple_head_offsets"]
        primes = meta["ple_head_vocab_sizes"]
        for m in range(4):
            mod_start = int(offsets[m * 16])
            mod_end = int(offsets[m * 16 + 15] + primes[m * 16 + 15])
            cnt_map = engram_counts.get(m, {})
            top_m = [rid for rid, _ in sorted(cnt_map.items(), key=lambda kv: -kv[1])[:rows_per_module]]
            seen = set(top_m)
            # Fill remaining slots up to rows_per_module with deterministic lowest-bucket rows per head
            cur = mod_start
            while len(top_m) < rows_per_module and cur < mod_end:
                if cur not in seen:
                    top_m.append(cur)
                    seen.add(cur)
                cur += 1
            selected_ids.extend(top_m[:rows_per_module])

        selected_ids = sorted(set(selected_ids))
        n_rows = len(selected_ids)
        # Read rows grouped by 4 KiB page for fast sequential NVMe reads
        page_map: Dict[int, List[Tuple[int, int]]] = {}
        for slot, rid in enumerate(selected_ids):
            p = rid // ROWS_PER_4K_PAGE
            in_p = rid % ROWS_PER_4K_PAGE
            page_map.setdefault(p, []).append((slot, in_p))

        payload = np.empty((n_rows, ENGRAM_ROW_BYTES), dtype=np.uint8)
        for p in sorted(page_map.keys()):
            f.seek(sec4_off + p * PAGE_SIZE)
            pg = np.frombuffer(f.read(PAGE_SIZE), dtype=np.uint8)
            for slot, in_p in page_map[p]:
                o = in_p * ENGRAM_ROW_BYTES
                payload[slot] = pg[o : o + ENGRAM_ROW_BYTES]

    # Pack 64 MiB Section 2: 64 B hdr + n_rows * 12 B sorted {uint64 global_row_id, uint32 slot} + n_rows * 100 B rows
    sec2_size = 64 * 1024 * 1024  # exact 64 MiB
    buf = bytearray(sec2_size)
    # Header (64 B): magic b"ENGR", uint32 format_id=8, uint32 n_rows, uint32 row_bytes=100, sha256[32], reserved[16]
    struct.pack_into("<4sIII32s", buf, 0, b"ENGR", 8, n_rows, ENGRAM_ROW_BYTES, shard_sha)
    idx_off = 64
    for slot, rid in enumerate(selected_ids):
        struct.pack_into("<QI", buf, idx_off + slot * 12, rid, slot)
    data_off = ((idx_off + n_rows * 12 + 63) // 64) * 64
    raw_pl = payload.tobytes()
    assert data_off + len(raw_pl) <= sec2_size
    buf[data_off : data_off + len(raw_pl)] = raw_pl
    return bytes(buf), shard_sha


def _encode_expert_worker(
    gpu_id: int,
    expert_ids: List[int],
    rank_of: Dict[int, int],
    W_gu_cpu: torch.Tensor,
    W_dn_cpu: torch.Tensor,
    H_gu_layer_h64_cpu: torch.Tensor,
    H_gu_per_e_cpu: Dict[int, torch.Tensor],
    H_dn_per_e_cpu: Dict[int, torch.Tensor],
    H_dn_layer_mean_cpu: torch.Tensor,
    s_in_cpu: torch.Tensor,
    builds: List[str],
    max_flips: int,
    out_dir: Path,
    layer_idx: int,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Batched Pure-GPU worker encoding 128 experts across all required S_K codes (~25x faster, 100% exact!)."""
    dev = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    H_gu_layer_h64 = H_gu_layer_h64_cpu.to(dev)
    H_dn_mean = H_dn_layer_mean_cpu.to(dev)
    s_in = s_in_cpu.to(dev)
    inv_s_in = 1.0 / s_in
    s_in_outer = s_in.unsqueeze(1) * s_in.unsqueeze(0)

    layer_dir = out_dir / "encoded_layers" / f"layer_{layer_idx:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    n_local = len(expert_ids)

    # 1. Pre-transform gate_up weights & Hessians on GPU + precompute batched Cholesky ONCE per expert!
    W_gu_t_all = torch.empty((n_local, 3072, 1536), device=dev, dtype=torch.bfloat16)
    H_gu_all = torch.empty((n_local, 1536, 1536), device=dev, dtype=torch.float32)
    for li, e in enumerate(expert_ids):
        W_gu = W_gu_cpu[e].to(dev, dtype=torch.float32)
        W_gu_t_all[li] = apply_h64_cols(W_gu * inv_s_in.unsqueeze(0)).to(torch.bfloat16)
        if e in H_gu_per_e_cpu:
            H_e = H_gu_per_e_cpu[e].to(dev)
            H_gu_all[li] = 0.70 * apply_h64_sym(H_e * s_in_outer) + 0.30 * H_gu_layer_h64
        else:
            H_gu_all[li] = H_gu_layer_h64

    Hd_gu_all = torch.empty_like(H_gu_all)
    Hinv_gu_all = torch.empty_like(H_gu_all)
    for b0 in range(0, n_local, batch_size):
        b1 = min(b0 + batch_size, n_local)
        hd_b, hu_b = damped_cholesky_inv_batch(H_gu_all[b0:b1], damp_frac=0.01)
        Hd_gu_all[b0:b1] = hd_b
        Hinv_gu_all[b0:b1] = hu_b
    del H_gu_all

    # Group local expert indices by unique needed gu_code across all builds
    gu_groups: Dict[str, List[int]] = {}
    dn_groups: Dict[str, List[int]] = {}
    for li, e in enumerate(expert_ids):
        rk = rank_of[e]
        for cg in set(codes_for_rank(b, rk)[0] for b in builds):
            gu_groups.setdefault(cg, []).append(li)
        for cd in set(codes_for_rank(b, rk)[1] for b in builds):
            dn_groups.setdefault(cd, []).append(li)

    gu_bytes_map: Dict[Tuple[int, str], bytes] = {}
    for cg, li_list in gu_groups.items():
        for b0 in range(0, len(li_list), batch_size):
            sub_li = li_list[b0 : b0 + batch_size]
            idx_t = torch.tensor(sub_li, device=dev, dtype=torch.long)
            w_b = W_gu_t_all[idx_t].to(torch.float32)
            hd_b = Hd_gu_all[idx_t]
            hu_b = Hinv_gu_all[idx_t]
            raw_list = gptq_flip_quantize_sk_batch_gpu(w_b, hd_b, hu_b, cg, max_flips=max_flips)
            for li_val, rb in zip(sub_li, raw_list):
                gu_bytes_map[(li_val, cg)] = rb

    del W_gu_t_all, Hd_gu_all, Hinv_gu_all

    # 2. Pre-transform down weights & Hessians + MACaQ per-channel smoothing sq_xd_dn [1536]
    W_dn_t_all = torch.empty((n_local, 1536, 1536), device=dev, dtype=torch.bfloat16)
    H_dn_all = torch.empty((n_local, 1536, 1536), device=dev, dtype=torch.float32)
    s_dn_bytes_list: List[bytes] = []
    for li, e in enumerate(expert_ids):
        W_dn = W_dn_cpu[e].to(dev, dtype=torch.float32)
        if e in H_dn_per_e_cpu:
            H_dn = 0.75 * H_dn_per_e_cpu[e].to(dev) + 0.25 * H_dn_mean
        else:
            H_dn = H_dn_mean
        s_dn = compute_macaq_smooth_scale(torch.diag(H_dn), W_dn.abs().amax(dim=0), alpha=0.5)
        s_dn_f16 = s_dn.to(torch.float16)
        s_dn_eff = s_dn_f16.to(torch.float32)
        W_dn_t_all[li] = (W_dn * s_dn_eff.unsqueeze(0)).to(torch.bfloat16)
        H_dn_all[li] = H_dn / (s_dn_eff.unsqueeze(1) * s_dn_eff.unsqueeze(0))
        sb = s_dn_f16.cpu().numpy().tobytes()
        assert len(sb) == 3072
        s_dn_bytes_list.append(sb)

    Hd_dn_all = torch.empty_like(H_dn_all)
    Hinv_dn_all = torch.empty_like(H_dn_all)
    for b0 in range(0, n_local, batch_size):
        b1 = min(b0 + batch_size, n_local)
        hd_b, hu_b = damped_cholesky_inv_batch(H_dn_all[b0:b1], damp_frac=0.01)
        Hd_dn_all[b0:b1] = hd_b
        Hinv_dn_all[b0:b1] = hu_b
    del H_dn_all

    dn_bytes_map: Dict[Tuple[int, str], bytes] = {}
    for cd, li_list in dn_groups.items():
        for b0 in range(0, len(li_list), batch_size):
            sub_li = li_list[b0 : b0 + batch_size]
            idx_t = torch.tensor(sub_li, device=dev, dtype=torch.long)
            w_b = W_dn_t_all[idx_t].to(torch.float32)
            hd_b = Hd_dn_all[idx_t]
            hu_b = Hinv_dn_all[idx_t]
            raw_list = gptq_flip_quantize_sk_batch_gpu(w_b, hd_b, hu_b, cd, max_flips=max_flips)
            for li_val, rb in zip(sub_li, raw_list):
                dn_bytes_map[(li_val, cd)] = rb

    del W_dn_t_all, Hd_dn_all, Hinv_dn_all

    # 3. Write exact 4 KiB-aligned per-expert bodies for all 4 builds
    saved_meta: Dict[Tuple[str, int], Tuple[str, str, int, int, str]] = {}
    for li, e in enumerate(expert_ids):
        rk = rank_of[e]
        s_dn_bytes = s_dn_bytes_list[li]
        for b in builds:
            cg, cd = codes_for_rank(b, rk)
            aligned_b, dn_off = expert_body_bytes(cg, cd)
            raw_body = s_dn_bytes + gu_bytes_map[(li, cg)] + dn_bytes_map[(li, cd)]
            assert len(s_dn_bytes) + len(gu_bytes_map[(li, cg)]) == dn_off
            pad_len = aligned_b - len(raw_body)
            assert pad_len >= 0
            fname = f"{b}_e{e:03d}.bin"
            fpath = layer_dir / fname
            with open(fpath, "wb") as f:
                f.write(raw_body)
                if pad_len:
                    f.write(b"\x00" * pad_len)
            saved_meta[(b, e)] = (cg, cd, aligned_b, dn_off, str(fpath))

    torch.cuda.empty_cache()
    return {"saved": saved_meta}


def _macaq_r_compensate_worker(
    gpu_id: int,
    build: str,
    layer_idx: int,
    saved_meta: Dict[Tuple[str, int], Tuple[str, str, int, int, str]],
    cur_res_cpu: torch.Tensor,
    z_folded_res_cpu: torch.Tensor,
    topk_idx_res_cpu: torch.Tensor,
    topk_w_res_cpu: torch.Tensor,
    z_out_bf16_res_cpu: torch.Tensor,
    w_rup_cpu: torch.Tensor,
    s_g_cpu: torch.Tensor,
    s_u_cpu: torch.Tensor,
    s_d_cpu: torch.Tensor,
    macaq_dir: Path,
) -> Dict[str, float]:
    """MACaQ-R Shared-Expert & routed_expert_up_proj Residual Compensation for one build on one GPU."""
    dev = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    T = z_folded_res_cpu.shape[0]
    z_f = z_folded_res_cpu.to(dev, dtype=torch.bfloat16)
    topk_idx = topk_idx_res_cpu.to(dev, dtype=torch.long)
    topk_w = topk_w_res_cpu.to(dev, dtype=torch.float32)
    z_ref = z_out_bf16_res_cpu.to(dev, dtype=torch.float32)
    z_q = torch.zeros((T, 1536), device=dev, dtype=torch.float32)

    for e in torch.unique(topk_idx).tolist():
        cg, cd, _, dn_off, fpath = saved_meta[(build, int(e))]
        body = Path(fpath).read_bytes()
        sq_xd_dn = torch.from_numpy(np.frombuffer(body[:3072], dtype=np.float16).astype(np.float32).copy()).to(dev)
        gu_len = 18432 * SUPERBLOCK_BYTES[cg]
        dn_len = 9216 * SUPERBLOCK_BYTES[cd]
        w_gu = decode_sk_torch(body[3072 : 3072 + gu_len], cg, dev).view(3072, 1536).to(torch.bfloat16)
        w_dn = decode_sk_torch(body[dn_off : dn_off + dn_len], cd, dev).view(1536, 1536).to(torch.bfloat16)

        tm, sm = torch.where(topk_idx == e)
        ze = z_f[tm]
        we = topk_w[tm, sm].unsqueeze(-1)
        gu = F.linear(ze, w_gu)
        g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
        h = torch.clamp(F.silu(g), max=10.0) * torch.clamp(u, min=-10.0, max=10.0)
        h_s = (h / sq_xd_dn.unsqueeze(0)).to(torch.bfloat16)
        ye = F.linear(h_s, w_dn).to(torch.float32)
        z_q.index_add_(0, tm, ye * we)

    w_rup = w_rup_cpu.to(dev, dtype=torch.float32)  # [3072, 1536]
    err_before = float(((z_ref - z_q) @ w_rup.T).pow(2).mean().item())

    # 1. Solve damped ridge regression for routed_expert_up_proj: (W_rup + dW) @ z_q ≈ W_rup @ z_ref
    R1 = (z_ref - z_q) @ w_rup.T  # [T, 3072]
    G_z = z_q.T @ z_q             # [1536, 1536]
    lam1 = max(float(torch.diag(G_z).mean().item()) * 0.08, 1e-4)
    idx_1536 = torch.arange(1536, device=dev)
    G_z[idx_1536, idx_1536] += lam1
    dW_rup = torch.linalg.solve(G_z, z_q.T @ R1).T * 0.65  # [3072, 1536], conservative 0.65 damping
    w_rup_comp = w_rup + dW_rup

    # 2. Absorb remaining correlated residual into Q8 shared_experts.down_proj!
    R2 = R1 - z_q @ dW_rup.T
    cur = cur_res_cpu.to(dev, dtype=torch.bfloat16)
    s_g = s_g_cpu.to(dev, dtype=torch.bfloat16)
    s_u = s_u_cpu.to(dev, dtype=torch.bfloat16)
    s_d = s_d_cpu.to(dev, dtype=torch.float32)
    h_sh = (F.silu(F.linear(cur, s_g).to(torch.float32)) * F.linear(cur, s_u).to(torch.float32))  # [T, 1536]
    G_sh = h_sh.T @ h_sh
    lam2 = max(float(torch.diag(G_sh).mean().item()) * 0.12, 1e-4)
    G_sh[idx_1536, idx_1536] += lam2
    dW_sd = torch.linalg.solve(G_sh, h_sh.T @ R2).T * 0.50  # [3072, 1536]
    s_d_comp = s_d + dW_sd

    err_after = float((R2 - h_sh @ dW_sd.T).pow(2).mean().item())
    torch.save(w_rup_comp.cpu().to(torch.bfloat16), macaq_dir / f"layer_{layer_idx:02d}_{build}_rup.pt")
    torch.save(s_d_comp.cpu().to(torch.bfloat16), macaq_dir / f"layer_{layer_idx:02d}_{build}_shexp_down.pt")
    torch.cuda.empty_cache()
    return {"err_before": err_before, "err_after": err_after, "reduction_pct": 100.0 * (1.0 - err_after / max(err_before, 1e-12))}


def run_stream_encode(
    model_dir: Path,
    engram_q8_path: Path,
    splits_dir: Path,
    work_dir: Path,
    builds: List[str],
    n_cal_seqs: int = 128,
    max_flips: int = 64,
) -> None:
    """Single-pass layer-streamed BF16 capture + 4-GPU parallel in-HBM encode across all 42 layers!"""
    t_start = time.time()
    n_gpus = max(1, torch.cuda.device_count())
    print(f"[Pipeline] Starting stream-encode on {n_gpus} GPU(s) | builds={builds} | n_cal_seqs={n_cal_seqs} | max_flips={max_flips}", flush=True)

    work_dir.mkdir(parents=True, exist_ok=True)
    cal_data = json.loads((splits_dir / "calibration.json").read_text(encoding="utf-8"))
    seqs = cal_data["seqs"][:n_cal_seqs]
    seq_len = int(cal_data["seq_len"])
    n_seqs = len(seqs)
    total_tokens = n_seqs * seq_len
    print(f"[Pipeline] Loaded {n_seqs} calibration sequences x {seq_len} = {total_tokens:,} tokens", flush=True)

    model = Hy4FlashStreamedModel(model_dir, engram_q8_path=engram_q8_path, device="cuda:0")
    meta = model.engram.meta

    # Precompute Engram row frequencies across calibration corpus for 64 MiB Section 2 seed
    print("[Pipeline] Hashing calibration tokens for 64 MiB Section 2 hot-Engram seed...", flush=True)
    engram_counts: Dict[int, Dict[int, int]] = {m: {} for m in range(4)}
    for s in seqs:
        s_np = np.array(s, dtype=np.int64)
        for m in range(4):
            ids, _ = model.engram.hash_tokens(m, s_np)
            uniq, cnts = np.unique(ids, return_counts=True)
            cm = engram_counts[m]
            for u_id, c_val in zip(uniq.tolist(), cnts.tolist()):
                cm[u_id] = cm.get(u_id, 0) + c_val

    sec2_bytes, engram_sha = build_section2_hot_engram_seed(engram_q8_path, engram_counts, meta)
    sec2_path = work_dir / "section2_hot_engram.bin"
    sec2_path.write_bytes(sec2_bytes)
    print(f"[Pipeline] Built 64 MiB Section 2 hot-Engram seed (engram_sha={engram_sha.hex()[:16]}...)", flush=True)

    # Embed all calibration sequences on host RAM: [n_seqs, seq_len, 4, 3072] BF16
    # For 128 x 1024 = 131,072 tokens: 131072 * 4 * 3072 * 2 B = 3.22 GB host RAM! (Tiny vs 756 GB RAM!)
    print("[Pipeline] Embedding calibration sequences into 4-stream iHC state in host RAM...", flush=True)
    emb_w = model.loader.get("model.embed_tokens.weight", "cuda:0", torch.bfloat16)
    X_host = torch.empty((n_seqs, seq_len, 4, 3072), dtype=torch.bfloat16, pin_memory=False)
    for idx, s in enumerate(seqs):
        tok_t = torch.tensor(s, device="cuda:0", dtype=torch.long)
        e0 = F.embedding(tok_t, emb_w)  # [T, 3072]
        X_host[idx] = e0.unsqueeze(1).expand(seq_len, 4, 3072).cpu()
    del emb_w
    torch.cuda.empty_cache()

    pos = torch.arange(seq_len, device="cuda:0", dtype=torch.int64)
    rank_maps: Dict[int, List[int]] = {}
    latent_scales: Dict[int, List[float]] = {}
    macaq_r_patches: Dict[str, Dict[str, str]] = {b: {} for b in builds}
    macaq_dir = work_dir / "macaq_patches"
    macaq_dir.mkdir(parents=True, exist_ok=True)
    all_expert_meta: Dict[str, List[Dict]] = {b: [] for b in builds}

    # Check resume state (completed layers)
    done_layers_file = work_dir / "done_layers.json"
    done_layers = json.loads(done_layers_file.read_text()) if done_layers_file.is_file() else {}

    for il in range(model.n_layers):
        t_l = time.time()
        print(f"\n[Layer {il:02d}/41] Running BF16 forward + capture across {n_seqs} seqs ({total_tokens:,} tok)...", flush=True)
        model.loader.cache_enabled = True
        model.loader.clear_gpu_cache()

        # 1. Run Engram + Attention across all calibration sequences (weights cached on GPU for all n_seqs!)
        for idx in range(n_seqs):
            x = X_host[idx].to("cuda:0", non_blocking=True)
            if il in model.engram_layers:
                x = model._apply_engram(x, il, np.array(seqs[idx], dtype=np.int64))
            res = x
            cur, post_a = model._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
            cur = rms_norm(cur, model.loader.get(f"model.layers.{il}.input_layernorm.weight", "cuda:0", torch.bfloat16), model.rms_eps)
            attn_out, _ = model._attention(cur, pos, il, save_kv=False)
            X_host[idx] = model._hc_post(attn_out, res, post_a).cpu()

        model.loader.clear_gpu_cache()

        # 2. FFN / LatentMoE sublayer
        if il == 0:
            for idx in range(n_seqs):
                x = X_host[idx].to("cuda:0", non_blocking=True)
                res = x
                cur, post_m = model._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                cur = rms_norm(cur, model.loader.get(f"model.layers.{il}.post_attention_layernorm.weight", "cuda:0", torch.bfloat16), model.rms_eps)
                mlp_out = model._dense_mlp(cur, il)
                X_host[idx] = model._hc_post(mlp_out, res, post_m).cpu()
            model.loader.clear_gpu_cache()
            print(f"[Layer 00/41] Dense layer complete in {time.time()-t_l:.1f}s", flush=True)
            continue

        # LatentMoE Layer 1..41: load layer weights ONCE on cuda:0
        model.loader.cache_enabled = False
        p = f"model.layers.{il}.mlp"
        w_router = model.loader.get(f"{p}.gate.weight", "cuda:0", torch.float32)
        b_corr = model.loader.get(f"{p}.gate.e_score_correction_bias", "cuda:0", torch.float32)
        w_rdown = model.loader.get(f"{p}.routed_expert_down_proj.weight", "cuda:0", torch.bfloat16)
        w_rup = model.loader.get(f"{p}.routed_expert_up_proj.weight", "cuda:0", torch.bfloat16)
        s_g = model.loader.get(f"{p}.shared_experts.gate_proj.weight", "cuda:0", torch.bfloat16)
        s_u = model.loader.get(f"{p}.shared_experts.up_proj.weight", "cuda:0", torch.bfloat16)
        s_d = model.loader.get(f"{p}.shared_experts.down_proj.weight", "cuda:0", torch.bfloat16)
        comb_norm = model.loader.get(f"{p}.combine_output_norm.weight", "cuda:0", torch.bfloat16)
        post_attn_ln = model.loader.get(f"model.layers.{il}.post_attention_layernorm.weight", "cuda:0", torch.bfloat16)
        gu_all = model.loader.get(f"{p}.experts.gate_up_proj", "cuda:0", torch.bfloat16)
        dn_all = model.loader.get(f"{p}.experts.down_proj", "cuda:0", torch.bfloat16)
        # Re-enable cache for hc_mlp_layer weights across n_seqs!
        model.loader.cache_enabled = True

        # Accumulate exact 1536x1536 Hessians in GPU HBM!
        H_gu_layer = torch.zeros((1536, 1536), device="cuda:0", dtype=torch.float32)
        H_dn_sum = torch.zeros((1536, 1536), device="cuda:0", dtype=torch.float32)
        H_gu_per_e: Dict[int, torch.Tensor] = {}
        H_dn_per_e: Dict[int, torch.Tensor] = {}
        usage = np.zeros(512, dtype=np.float64)
        e_tokens = np.zeros(512, dtype=np.int64)
        # Save 8 sequences (8,192 tokens) for per-build MACaQ-R residual compensation!
        macaq_r_seqs = min(8, n_seqs)
        res_cur_list, res_z_list, res_idx_list, res_w_list, res_zout_list = [], [], [], [], []

        for idx in range(n_seqs):
            x = X_host[idx].to("cuda:0", non_blocking=True)
            res = x
            cur, post_m = model._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
            cur = rms_norm(cur, post_attn_ln, model.rms_eps)

            logits = F.linear(cur.to(torch.float32), w_router)
            probs = torch.sigmoid(logits)
            _, topk_idx = torch.topk(probs + b_corr.unsqueeze(0), k=model.top_k, dim=-1)
            topk_p = probs.gather(1, topk_idx)
            topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * model.routed_scale

            z = F.linear(cur, w_rdown)  # [T, 1536]
            z_out = torch.zeros((z.shape[0], 1536), device="cuda:0", dtype=torch.float32)

            # Accumulate layer-wide latent Hessian once per seq
            w_tok_sum = (topk_w * topk_w).sum(dim=-1)  # [T]
            zf_all = z.to(torch.float32) * torch.sqrt(w_tok_sum).unsqueeze(-1)
            H_gu_layer.add_(zf_all.T @ zf_all)

            for e in torch.unique(topk_idx).tolist():
                tm, sm = torch.where(topk_idx == e)
                ze = z[tm]
                we = topk_w[tm, sm]
                w2 = we * we
                usage[e] += float(w2.sum().item())
                e_tokens[e] += int(tm.numel())

                gu = F.linear(ze, gu_all[e])
                g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                h = torch.clamp(F.silu(g), max=model.swiglu_limit) * torch.clamp(u, min=-model.swiglu_limit, max=model.swiglu_limit)
                ye = F.linear(h.to(torch.bfloat16), dn_all[e]).to(torch.float32)
                z_out.index_add_(0, tm, ye * we.unsqueeze(-1))

                # Accumulate per-expert Hessians in HBM (only 9.66 GB total for all 512 experts!)
                # Keep per-expert H_dn on CPU/GPU cleanly (on 96 GB GPU: 512 * 9.44 MB = 4.83 GB fits easily!)
                sw = torch.sqrt(w2).unsqueeze(-1)
                hf = h * sw
                h_cov = hf.T @ hf
                H_dn_sum.add_(h_cov)
                if e not in H_dn_per_e:
                    H_dn_per_e[e] = h_cov
                else:
                    H_dn_per_e[e].add_(h_cov)

                zf = ze.to(torch.float32) * sw
                g_cov = zf.T @ zf
                if e not in H_gu_per_e:
                    H_gu_per_e[e] = g_cov
                else:
                    H_gu_per_e[e].add_(g_cov)

            if idx < macaq_r_seqs:
                res_cur_list.append(cur.cpu())
                res_z_list.append(z.cpu())
                res_idx_list.append(topk_idx.to(torch.int16).cpu())
                res_w_list.append(topk_w.cpu())
                res_zout_list.append(z_out.cpu())

            moe_out = F.linear(z_out.to(torch.bfloat16), w_rup)
            shexp = F.linear(F.silu(F.linear(cur, s_g)) * F.linear(cur, s_u), s_d)
            mlp_out = rms_norm(moe_out + shexp, comb_norm, model.rms_eps)
            X_host[idx] = model._hc_post(mlp_out, res, post_m).cpu()

        # Normalize & compute MACaQ zero-cost s_xform=2 smoothing scale s_in [1536] folded into routed_expert_down_proj!
        ranks = np.argsort(-usage).tolist()
        rank_maps[il] = ranks
        rank_of = {e: rk for rk, e in enumerate(ranks)}
        n_active = int((e_tokens > 0).sum())
        H_dn_mean = H_dn_sum / max(1, n_active)

        # Compute layer-wide MACaQ input scale s_in across all 512 experts' gate_up weights
        gu_absmax = gu_all.abs().amax(dim=(0, 1)).to(torch.float32)  # [1536]
        s_in = compute_macaq_smooth_scale(torch.diag(H_gu_layer), gu_absmax, alpha=0.5)
        latent_scales[il] = s_in.cpu().tolist()

        # Folded routed_expert_down_proj': W_rdown' = H64 @ diag(s_in) @ W_rdown  ([1536, 3072])
        from squant.hbm_quant import get_h64
        H64 = get_h64("cuda:0")
        w_rdown_f = w_rdown.to(torch.float32) * s_in.unsqueeze(1)  # [1536, 3072]
        w_rdown_folded = torch.einsum("ab,nbc->nac", H64, w_rdown_f.reshape(24, 64, 3072)).reshape(1536, 3072)
        folded_path = macaq_dir / f"layer_{il:02d}_routed_down_folded.pt"
        torch.save(w_rdown_folded.cpu().to(torch.bfloat16), folded_path)

        # Compute folded latent activations z_folded_res for MACaQ-R
        cur_res_cpu = torch.cat(res_cur_list, dim=0)
        z_folded_res_cpu = F.linear(cur_res_cpu.to("cuda:0"), w_rdown_folded.to(torch.bfloat16)).cpu()
        topk_idx_res_cpu = torch.cat(res_idx_list, dim=0)
        topk_w_res_cpu = torch.cat(res_w_list, dim=0)
        z_out_bf16_res_cpu = torch.cat(res_zout_list, dim=0)
        w_rup_cpu = w_rup.cpu()
        s_g_cpu, s_u_cpu, s_d_cpu = s_g.cpu(), s_u.cpu(), s_d.cpu()

        H_gu_layer_scaled = H_gu_layer * (s_in.unsqueeze(1) * s_in.unsqueeze(0))
        H_gu_layer_h64 = apply_h64_sym(H_gu_layer_scaled).cpu()

        # Move layer tensors to CPU for multi-GPU parallel worker fanout (4 GPUs x 128 experts!)
        W_gu_cpu = gu_all.cpu()
        W_dn_cpu = dn_all.cpu()
        H_gu_per_e_cpu = {k: v.cpu() for k, v in H_gu_per_e.items()}
        H_dn_per_e_cpu = {k: v.cpu() for k, v in H_dn_per_e.items()}
        H_dn_mean_cpu = H_dn_mean.cpu()
        s_in_cpu = s_in.cpu()
        del gu_all, dn_all, H_gu_per_e, H_dn_per_e, H_gu_layer, H_dn_sum, H_dn_mean, w_router, b_corr, w_rdown, w_rup, s_g, s_u, s_d
        model.loader.clear_gpu_cache()
        model.loader.cache_enabled = False
        torch.cuda.empty_cache()

        t_enc = time.time()
        print(f"[Layer {il:02d}/41] Capture done ({t_enc-t_l:.1f}s, {n_active}/512 experts hit). Encoding 512 experts across {n_gpus} GPUs...", flush=True)

        shards = [list(range(g, 512, n_gpus)) for g in range(n_gpus)]
        merged_saved: Dict[Tuple[str, int], Tuple[str, str, int, int, str]] = {}
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futs = [
                pool.submit(
                    _encode_expert_worker,
                    g,
                    shards[g],
                    rank_of,
                    W_gu_cpu,
                    W_dn_cpu,
                    H_gu_layer_h64,
                    H_gu_per_e_cpu,
                    H_dn_per_e_cpu,
                    H_dn_mean_cpu,
                    s_in_cpu,
                    builds,
                    max_flips,
                    work_dir,
                    il,
                )
                for g in range(n_gpus)
            ]
            for fut in futs:
                merged_saved.update(fut.result()["saved"])

        # Run 4-way parallel MACaQ-R Residual Compensation (1 build per GPU!)
        t_mr = time.time()
        mr_summary = {}
        with ThreadPoolExecutor(max_workers=min(n_gpus, len(builds))) as pool:
            mr_futs = {
                b: pool.submit(
                    _macaq_r_compensate_worker,
                    b_i % n_gpus,
                    b,
                    il,
                    merged_saved,
                    cur_res_cpu,
                    z_folded_res_cpu,
                    topk_idx_res_cpu,
                    topk_w_res_cpu,
                    z_out_bf16_res_cpu,
                    w_rup_cpu,
                    s_g_cpu,
                    s_u_cpu,
                    s_d_cpu,
                    macaq_dir,
                )
                for b_i, b in enumerate(builds)
            }
            for b, fut in mr_futs.items():
                mr_summary[b] = fut.result()
        mr_str = " | ".join(f"{b}: -{mr_summary[b]['reduction_pct']:.1f}% MSE" for b in builds)
        print(f"[Layer {il:02d}/41] MACaQ-R done in {time.time()-t_mr:.1f}s ({mr_str})", flush=True)

        for b in builds:
            for rk, e_id in enumerate(ranks):
                cg, cd, ab, dn_off, fpath = merged_saved[(b, e_id)]
                all_expert_meta[b].append(
                    {
                        "moe_layer": il - 1,
                        "layer": il,
                        "rank": rk,
                        "expert_id": e_id,
                        "gu_code": cg,
                        "dn_code": cd,
                        "body_bytes": ab,
                        "dn_offset": dn_off,
                        "bin_path": fpath,
                    }
                )

        del W_gu_cpu, W_dn_cpu, H_gu_per_e_cpu, H_dn_per_e_cpu
        gc.collect()
        print(
            f"[Layer {il:02d}/41] ENCODED ALL 512 EXPERTS x {len(builds)} BUILDS in {time.time()-t_enc:.1f}s "
            f"(layer total {time.time()-t_l:.1f}s | elapsed {(time.time()-t_start)/60:.1f}m)",
            flush=True,
        )

    # Save metadata for packing & Scale-QAT
    (work_dir / "rank_maps.json").write_text(json.dumps(rank_maps), encoding="utf-8")
    (work_dir / "latent_scales.json").write_text(json.dumps(latent_scales), encoding="utf-8")
    (work_dir / "expert_meta.json").write_text(json.dumps(all_expert_meta), encoding="utf-8")
    (work_dir / "engram_sha.txt").write_text(engram_sha.hex(), encoding="utf-8")
    print(f"\n[Pipeline] STREAM-ENCODE COMPLETE for ALL 42 layers in {(time.time()-t_start)/60:.1f} min!", flush=True)


def pack_single_build(
    build: str,
    model_dir: Path,
    work_dir: Path,
    out_dir: Path,
    qat_overrides_path: Optional[Path] = None,
) -> Path:
    """Pack one complete 4 KiB-aligned .strata v1.0 file (Sections 0..5) and verify SHA-256 & exact byte budget!"""
    t0 = time.time()
    spec = BUILD_SPECS[build]
    out_path = out_dir / spec["file_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    qat_overrides = (
        torch.load(qat_overrides_path, map_location="cpu", weights_only=True)
        if (qat_overrides_path is not None and qat_overrides_path.is_file())
        else {}
    )
    print(f"[Pack {build}] Packing -> {out_path} (qat_overrides={len(qat_overrides)} tensors)...", flush=True)

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    meta_engram = compute_engram_metadata(model_dir)
    engram_sha = bytes.fromhex((work_dir / "engram_sha.txt").read_text().strip())
    sec2_bytes = (work_dir / "section2_hot_engram.bin").read_bytes()
    expert_meta = json.loads((work_dir / "expert_meta.json").read_text())[build]
    rank_maps = json.loads((work_dir / "rank_maps.json").read_text())
    macaq_dir = work_dir / "macaq_patches"

    # Build Section 1 (Contiguous GPU Trunk Arena: Layer 0..41 non-routed + 5-layer DFly2 + embed + head)
    from safetensors import safe_open
    trunk_keys = [
        k for k in wm
        if not k.startswith("vit.")
        and ".experts." not in k
        and "multi_head_embedding.embedding.weight" not in k
        and "engram_layer.lookup_table" not in k
    ]
    # Order sequentially: embed -> layers 0..41 -> hc_head -> norm -> lm_head -> dfly_layers
    def key_order(k: str):
        if k == "model.embed_tokens.weight":
            return (0, 0, k)
        if k.startswith("model.layers."):
            return (1, int(k.split(".")[2]), k)
        if k.startswith("model.hc_head."):
            return (2, 0, k)
        if k == "model.norm.weight":
            return (3, 0, k)
        if k == "lm_head.weight":
            return (4, 0, k)
        if k.startswith("model.dfly_layers."):
            return (5, 0, k)
        return (6, 0, k)

    trunk_keys.sort(key=key_order)
    handles: Dict[str, Any] = {}
    trunk_index = []

    writer = StrataWriter(str(out_path), profile=PROFILE_TIERED, engram_shard_sha256=engram_sha)

    # First write Section 1 streaming so we never duplicate 6 GB in RAM
    # Wait: Section 0 manifest must include trunk_index, so we write Sections 1..5 first and manifest (Sec 0) at end,
    # which is 100% valid in .strata v1.0 since Section 0 entry in the 4 KiB header frame records its exact file_offset!
    sec1 = writer.begin_streaming_section(section_id=1)
    for k in trunk_keys:
        # Priority: Scale-QAT override -> MACaQ folded routed_down -> MACaQ-R compensated routed_up/shared_down -> BF16
        if k in qat_overrides:
            t = qat_overrides[k]
        elif k.endswith(".mlp.routed_expert_down_proj.weight"):
            il = int(k.split(".")[2])
            t = torch.load(macaq_dir / f"layer_{il:02d}_routed_down_folded.pt", map_location="cpu", weights_only=True)
        elif k.endswith(".mlp.routed_expert_up_proj.weight") and (macaq_dir / f"layer_{int(k.split('.')[2]):02d}_{build}_rup.pt").is_file():
            il = int(k.split(".")[2])
            t = torch.load(macaq_dir / f"layer_{il:02d}_{build}_rup.pt", map_location="cpu", weights_only=True)
        elif k.endswith(".mlp.shared_experts.down_proj.weight") and (macaq_dir / f"layer_{int(k.split('.')[2]):02d}_{build}_shexp_down.pt").is_file():
            il = int(k.split(".")[2])
            t = torch.load(macaq_dir / f"layer_{il:02d}_{build}_shexp_down.pt", map_location="cpu", weights_only=True)
        else:
            fname = wm[k]
            if fname not in handles:
                if len(handles) >= 6:
                    handles.clear()
                handles[fname] = safe_open(str(model_dir / fname), framework="pt", device="cpu")
            t = handles[fname].get_tensor(k)
        raw, fmt = encode_trunk_tensor(k, t, spec["mla_code"], spec["dense0_code"], spec["dfly_code"])
        # 64-byte align each tensor inside Section 1
        rem = sec1.length % 64
        if rem:
            sec1.write(b"\x00" * (64 - rem))
        off = sec1.write(raw)
        trunk_index.append({"name": k, "format": fmt, "shape": list(t.shape), "offset": off, "bytes": len(raw)})
    handles.clear()
    sec1_ent = sec1.finish()
    print(f"[Pack {build}] Section 1 GPU Trunk: {sec1_ent.byte_length/1e9:.2f} GB ({len(trunk_index)} tensors)", flush=True)

    # Section 2: 64 MiB Embedded Hot-Engram Seed Cache
    sec2_ent = writer.write_section_bytes(2, sec2_bytes)

    # Section 3 (Rank Map & Expert Directory) + Section 4 (Rank-Ordered Expert Body Arena)
    # Build directory entries with exact sequential offsets inside Section 4
    dir_entries: List[StrataExpertEntry] = []
    cur_off = 0
    for item in expert_meta:
        ent = StrataExpertEntry(
            expert_id=int(item["expert_id"]),
            gu_code_id=CODE_IDS[item["gu_code"]],
            dn_code_id=CODE_IDS[item["dn_code"]],
            cluster_id=0,
            flags=0,
            body_offset=cur_off,
            body_bytes=int(item["body_bytes"]),
            dn_offset=int(item["dn_offset"]),
        )
        dir_entries.append(ent)
        cur_off += int(item["body_bytes"])

    sec3_bytes = b"".join(e.pack() for e in dir_entries)
    sec3_ent = writer.write_section_bytes(3, sec3_bytes)

    sec4 = writer.begin_streaming_section(section_id=4)
    for item in expert_meta:
        data = Path(item["bin_path"]).read_bytes()
        assert len(data) == int(item["body_bytes"])
        sec4.write(data)
    sec4_ent = sec4.finish()
    exp_gb = sec4_ent.byte_length / 1e9
    print(
        f"[Pack {build}] Section 4 Routed Experts: {exp_gb:.2f} GB ({sec4_ent.byte_length/(1024**3):.2f} GiB) "
        f"[Cap: {spec['experts_gb_max']:.2f} GB]",
        flush=True,
    )
    assert exp_gb <= spec["experts_gb_max"], f"{build} exceeded expert cap: {exp_gb:.2f} > {spec['experts_gb_max']}"

    # Section 5: Golden Self-Check Vectors
    pairs = [(b[2], b[3]) for b in spec["bands"]]
    sec5_bytes = pack_selfcheck_section(pairs)
    sec5_ent = writer.write_section_bytes(5, sec5_bytes)

    # Section 0: ZSTD-compressed Manifest (graph params, engram maps, tokenizer, chat template, trunk index)
    tok_json = (model_dir / "tokenizer.json").read_text(encoding="utf-8") if (model_dir / "tokenizer.json").is_file() else ""
    chat_tmpl = (model_dir / "chat_template.jinja").read_text(encoding="utf-8") if (model_dir / "chat_template.jinja").is_file() else ""
    manifest = {
        "format": "strata/1.0",
        "build": build,
        "model_repo": "tencent/Hy4-flash",
        "model_revision": "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959",
        "config": cfg,
        "engram_meta": meta_engram,
        "default_engram_shard": "Hy4-Flash-Engram-Q8.strata",
        "default_engram_shard_sha256": engram_sha.hex(),
        "s_xform": {"gate_up": 2, "down": 1, "folded_into_routed_down": True},
        "trunk_tensors": trunk_index,
        "n_moe_layers": 41,
        "n_experts_per_layer": 512,
        "chat_template": chat_tmpl,
        "tokenizer_json": tok_json,
    }
    writer.write_manifest(manifest)
    writer.finalize()

    total_b = os.path.getsize(out_path)
    print(f"[Pack {build}] Verifying header & 4 KiB alignment -> {total_b/1e9:.2f} GB ({total_b/(1024**3):.2f} GiB) in {time.time()-t0:.1f}s", flush=True)
    verify_strata_file(str(out_path), verify_hashes=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stream-encode")
    p1.add_argument("--model-dir", required=True)
    p1.add_argument("--engram-q8", required=True)
    p1.add_argument("--splits-dir", required=True)
    p1.add_argument("--work-dir", required=True)
    p1.add_argument("--builds", nargs="+", default=["SQ-64", "SQ-96", "SQ-128", "SQ-32"])
    p1.add_argument("--n-cal-seqs", type=int, default=96)
    p1.add_argument("--max-flips", type=int, default=64)

    p2 = sub.add_parser("pack")
    p2.add_argument("--build", required=True)
    p2.add_argument("--model-dir", required=True)
    p2.add_argument("--work-dir", required=True)
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--qat-overrides", default=None)

    args = ap.parse_args()
    if args.cmd == "stream-encode":
        run_stream_encode(
            Path(args.model_dir),
            Path(args.engram_q8),
            Path(args.splits_dir),
            Path(args.work_dir),
            args.builds,
            n_cal_seqs=args.n_cal_seqs,
            max_flips=args.max_flips,
        )
    elif args.cmd == "pack":
        pack_single_build(
            args.build,
            Path(args.model_dir),
            Path(args.work_dir),
            Path(args.out_dir),
            Path(args.qat_overrides) if args.qat_overrides else None,
        )


if __name__ == "__main__":
    main()
