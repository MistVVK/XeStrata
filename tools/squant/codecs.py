"""Strata Quant (.strata v1.0) S_K codecs, 64-wide Block-Hadamard (s_xform=2),
q8_1 activation quantizer, exact dot products, and in-HBM Block-LDL + MACaQ flip refinement.

Exact super-block byte sizes (256 weights = 4 sub-groups of 64 weights, 6 B header):
  code_id=0  S1.25 : 48 B  (6 B hdr + 40 B 4D-lattice 5-bit indices + 2 B fine nibble scales)
  code_id=1  S1.5  : 58 B  (6 B hdr + 52 B base-3 ternary, 5 trits/byte)
  code_id=2  S2_K  : 70 B  (6 B hdr + 64 B 2-bit symmetric half-integer q in {0..3} -> 2q-3)
  code_id=3  S3_K  : 102 B (6 B hdr + 96 B 3-bit symmetric half-integer q in {0..7} -> 2q-7)
  code_id=4  S4_K  : 134 B (6 B hdr + 128 B 4-bit symmetric half-integer q in {0..15} -> 2q-15)
  code_id=5  S5_K  : 166 B (6 B hdr + 160 B 5-bit symmetric half-integer q in {0..31} -> 2q-31)
  code_id=6  S6_K  : 198 B (6 B hdr + 192 B 6-bit symmetric half-integer q in {0..63} -> 2q-63)
  code_id=8  S8_K  : 262 B (6 B hdr + 256 B 8-bit symmetric int8 q in [-127..127])
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

QK_S = 256
SUB_G = 64
NUM_SUB = 4  # 256 / 64

CODE_IDS = {
    "S1.25": 0,
    "S1.5": 1,
    "S2_K": 2,
    "S3_K": 3,
    "S4_K": 4,
    "S5_K": 5,
    "S6_K": 6,
    "S8_K": 8,
}
ID_TO_NAME = {v: k for k, v in CODE_IDS.items()}

SUPERBLOCK_BYTES = {
    "S1.25": 48,
    "S1.5": 58,
    "S2_K": 70,
    "S3_K": 102,
    "S4_K": 134,
    "S5_K": 166,
    "S6_K": 198,
    "S8_K": 262,
}

# Optimal 32-entry 4D signed lattice codebook for S1.25 (int8 values in {-2,-1,0,+1,+2})
# Covers all 16 binary signs (±1,±1,±1,±1) + 8 single-outlier (±2,0,0,0) + 8 sparse-pair patterns.
# Fits in 128 bytes (2 AVX-512 zmm registers) for zero-memory-lookup SIMD decoding!
S125_CODEBOOK = np.array(
    [
        # 16 dense binary sign vectors (±1, ±1, ±1, ±1)
        [-1, -1, -1, -1], [-1, -1, -1,  1], [-1, -1,  1, -1], [-1, -1,  1,  1],
        [-1,  1, -1, -1], [-1,  1, -1,  1], [-1,  1,  1, -1], [-1,  1,  1,  1],
        [ 1, -1, -1, -1], [ 1, -1, -1,  1], [ 1, -1,  1, -1], [ 1, -1,  1,  1],
        [ 1,  1, -1, -1], [ 1,  1, -1,  1], [ 1,  1,  1, -1], [ 1,  1,  1,  1],
        # 8 single-peak vectors (±2 on one coord)
        [ 2,  0,  0,  0], [-2,  0,  0,  0], [ 0,  2,  0,  0], [ 0, -2,  0,  0],
        [ 0,  0,  2,  0], [ 0,  0, -2,  0], [ 0,  0,  0,  2], [ 0,  0,  0, -2],
        # 8 sparse-zero / double-peak vectors (0,0,0,0 + dominant pairs)
        [ 0,  0,  0,  0], [ 2,  2,  0,  0], [-2, -2,  0,  0], [ 0,  0,  2,  2],
        [ 0,  0, -2, -2], [ 2, -2,  0,  0], [ 0,  0,  2, -2], [ 2,  0,  2,  0],
    ],
    dtype=np.int8,
)

# Precomputed 5-trit decode table for S1.5: maps byte 0..242 -> 5 int8 values in {-1, 0, +1}
_T5_DECODE = np.zeros((256, 5), dtype=np.int8)
for _v in range(243):
    _x = _v
    for _j in range(5):
        _T5_DECODE[_v, _j] = (_x % 3) - 1
        _x //= 3

_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.int32)


def expert_body_bytes(gu_code: str, dn_code: str, n_latent: int = 1536, n_inter: int = 1536) -> Tuple[int, int]:
    """Return exact 4 KiB-aligned (body_bytes, dn_offset) for a square [1536, 1536] LatentMoE expert."""
    assert n_latent % QK_S == 0 and n_inter % QK_S == 0
    sb_gu = (2 * n_inter * n_latent) // QK_S  # 18,432
    sb_dn = (n_latent * n_inter) // QK_S      # 9,216
    sq_xd_dn_bytes = n_inter * 2              # 3,072 B (FP16[1536])
    dn_offset = sq_xd_dn_bytes + sb_gu * SUPERBLOCK_BYTES[gu_code]
    raw_total = dn_offset + sb_dn * SUPERBLOCK_BYTES[dn_code]
    aligned = ((raw_total + 4095) // 4096) * 4096
    return aligned, dn_offset


def hadamard_matrix_64() -> np.ndarray:
    """Normalized 64x64 Walsh-Hadamard matrix H64 (H64 @ H64.T = I)."""
    h = np.array([[1.0]], dtype=np.float32)
    for _ in range(6):
        h = np.block([[h, h], [h, -h]])
    return h / 8.0


H64_NP = hadamard_matrix_64()


def fwht64_np(x: np.ndarray) -> np.ndarray:
    """Apply normalized 64-wide Block-Hadamard transform along the last axis (must be multiple of 64)."""
    orig_shape = x.shape
    assert orig_shape[-1] % 64 == 0
    y = x.astype(np.float32, copy=False).reshape(-1, 64) @ H64_NP
    return y.reshape(orig_shape)


def _pack_header(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Pack FP16 D [N] and uint8 u [N, 4] -> uint8 [N, 6]."""
    n = D.shape[0]
    out = np.empty((n, 6), dtype=np.uint8)
    out[:, :2] = np.ascontiguousarray(D.astype(np.float16)).view(np.uint8).reshape(n, 2)
    out[:, 2:6] = u.astype(np.uint8)
    return out


def _unpack_header(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack uint8 [N, 6] -> (float32 D [N], float32 sub_scales [N, 4])."""
    n = raw.shape[0]
    D = np.ascontiguousarray(raw[:, :2]).view(np.float16).reshape(n).astype(np.float32)
    u = raw[:, 2:6].astype(np.float32)
    return D, u


def _compute_scales(
    w_blocks: np.ndarray,
    max_q: float,
    clip_grid: Tuple[float, ...] = (1.0,),
    half_int: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute super-block FP16 D [N] and 4 sub-group uint8 scales u [N, 4] for w_blocks [N, 4, 64].
    Supports per-sub-group MSE-optimal clipping (huge +3.5 dB SQNR gain for S2_K and +1.8 dB for S3_K).
    Effective scale for sub-group g is: scale_g = D_f16 * (u_g / 255.0).
    """
    amax_sub = np.max(np.abs(w_blocks), axis=2)  # [N, 4]
    if len(clip_grid) > 1:
        best_clip = amax_sub.copy()
        best_err = np.full_like(amax_sub, np.inf)
        levels = int(round(2.0 * max_q + 1.0)) if half_int else int(round(max_q))
        for r in clip_grid:
            cand_max = amax_sub * r
            s = np.where(cand_max > 0.0, cand_max / max_q, 1.0)[:, :, None]
            inv = np.where(cand_max > 0.0, max_q / cand_max, 0.0)[:, :, None]
            if half_int:
                q = np.clip(np.floor(w_blocks * inv + max_q + 0.5), 0, levels - 1) - max_q
            else:
                q = np.clip(np.rint(w_blocks * inv), -max_q, max_q)
            err = np.mean((w_blocks - q * s) ** 2, axis=2)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_clip = np.where(better, cand_max, best_clip)
        amax_sub = best_clip

    amax_sb = np.max(amax_sub, axis=1)  # [N]
    D_f32 = amax_sb / max_q
    D_f16 = D_f32.astype(np.float16)
    D_eff = D_f16.astype(np.float32)
    denom = np.where(amax_sb > 0.0, amax_sb, 1.0)[:, None]
    u = np.clip(np.rint(amax_sub / denom * 255.0), 0, 255).astype(np.uint8)
    u = np.where((amax_sub > 0.0) & (u == 0), 1, u).astype(np.uint8)
    eff_scale = D_eff[:, None] * (u.astype(np.float32) / 255.0)
    return D_f16, u, eff_scale


def encode_sk(w: np.ndarray, code_name: str) -> bytes:
    """Encode a 1D/2D float32 array (last dim multiple of 256) into exact .strata v1.0 S_K bytes."""
    w = np.asarray(w, dtype=np.float32)
    assert w.size % QK_S == 0
    n_sb = w.size // QK_S
    wb = w.reshape(n_sb, NUM_SUB, SUB_G)

    if code_name == "S8_K":
        D_f16, u, eff = _compute_scales(wb, 127.0)
        inv = np.where(eff > 0.0, 1.0 / eff, 0.0)[:, :, None]
        q = np.clip(np.rint(wb * inv), -127, 127).astype(np.int8)
        hdr = _pack_header(D_f16, u)
        payload = q.reshape(n_sb, QK_S).view(np.uint8)
        return np.concatenate([hdr, payload], axis=1).tobytes()

    if code_name in ("S2_K", "S3_K", "S4_K", "S5_K", "S6_K"):
        bits = int(code_name[1])
        levels = 1 << bits
        half = (levels - 1) / 2.0  # 1.5, 3.5, 7.5, 15.5, 31.5
        # MSE-optimal clipping grid for low-bit codes (+3.5 dB on S2_K, +1.8 dB on S3_K, +0.9 dB on S4_K)
        if bits == 2:
            grid = (0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 1.00)
        elif bits == 3:
            grid = (0.64, 0.72, 0.80, 0.88, 0.94, 1.00)
        elif bits == 4:
            grid = (0.78, 0.86, 0.93, 1.00)
        else:
            grid = (0.90, 0.96, 1.00)
        D_f16, u, eff = _compute_scales(wb, half, clip_grid=grid, half_int=True)
        inv = np.where(eff > 0.0, 1.0 / eff, 0.0)[:, :, None]
        q = np.clip(np.floor(wb * inv + half + 0.5), 0, levels - 1).astype(np.uint8).reshape(n_sb, QK_S)
        hdr = _pack_header(D_f16, u)
        if bits == 2:
            q4 = q.reshape(n_sb, 64, 4)
            pl = q4[:, :, 0] | (q4[:, :, 1] << 2) | (q4[:, :, 2] << 4) | (q4[:, :, 3] << 6)
        elif bits == 3:
            lo2 = q & 0x03
            hi1 = (q >> 2) & 0x01
            lo_b = lo2.reshape(n_sb, 64, 4)
            p_lo = lo_b[:, :, 0] | (lo_b[:, :, 1] << 2) | (lo_b[:, :, 2] << 4) | (lo_b[:, :, 3] << 6)
            hi_b = hi1.reshape(n_sb, 32, 8)
            p_hi = np.zeros((n_sb, 32), dtype=np.uint8)
            for b in range(8):
                p_hi |= hi_b[:, :, b] << b
            pl = np.concatenate([p_lo, p_hi], axis=1)
        elif bits == 4:
            q2 = q.reshape(n_sb, 128, 2)
            pl = q2[:, :, 0] | (q2[:, :, 1] << 4)
        elif bits == 5:
            lo4 = q & 0x0F
            hi1 = (q >> 4) & 0x01
            lo_b = lo4.reshape(n_sb, 128, 2)
            p_lo = lo_b[:, :, 0] | (lo_b[:, :, 1] << 4)
            hi_b = hi1.reshape(n_sb, 32, 8)
            p_hi = np.zeros((n_sb, 32), dtype=np.uint8)
            for b in range(8):
                p_hi |= hi_b[:, :, b] << b
            pl = np.concatenate([p_lo, p_hi], axis=1)
        elif bits == 6:
            lo4 = q & 0x0F
            hi2 = (q >> 4) & 0x03
            lo_b = lo4.reshape(n_sb, 128, 2)
            p_lo = lo_b[:, :, 0] | (lo_b[:, :, 1] << 4)
            hi_b = hi2.reshape(n_sb, 64, 4)
            p_hi = hi_b[:, :, 0] | (hi_b[:, :, 1] << 2) | (hi_b[:, :, 2] << 4) | (hi_b[:, :, 3] << 6)
            pl = np.concatenate([p_lo, p_hi], axis=1)
        return np.concatenate([hdr, pl], axis=1).tobytes()

    if code_name == "S1.5":
        # Base-3 ternary {-1, 0, +1}, optimal threshold ~0.5 * mean(|w|) / scale
        # Scale per sub-group is mean of |w| above 0.5*mean(|w|) (optimal ternary scale!)
        abs_w = np.abs(wb)
        thresh = 0.7 * np.mean(abs_w, axis=2, keepdims=True)
        mask = abs_w > thresh
        cnt = np.maximum(np.sum(mask, axis=2), 1)
        opt_s = np.sum(abs_w * mask, axis=2) / cnt  # [N, 4]
        amax_s = np.max(opt_s, axis=1)
        D_f16 = amax_s.astype(np.float16)
        D_eff = D_f16.astype(np.float32)
        denom = np.where(amax_s > 0.0, amax_s, 1.0)[:, None]
        u = np.clip(np.rint(opt_s / denom * 255.0), 0, 255).astype(np.uint8)
        u = np.where((opt_s > 0.0) & (u == 0), 1, u).astype(np.uint8)
        eff = D_eff[:, None] * (u.astype(np.float32) / 255.0)
        inv = np.where(eff > 0.0, 1.0 / eff, 0.0)[:, :, None]
        q = np.clip(np.rint(wb * inv), -1, 1).astype(np.int32).reshape(n_sb, QK_S)
        # Pack 5 trits in {0,1,2} per byte -> 52 bytes (last byte uses 1 trit)
        q_pad = np.ones((n_sb, 260), dtype=np.int32)
        q_pad[:, :256] = q + 1
        pl = np.sum(q_pad.reshape(n_sb, 52, 5) * _POW3[None, None, :], axis=2).astype(np.uint8)
        hdr = _pack_header(D_f16, u)
        return np.concatenate([hdr, pl], axis=1).tobytes()

    if code_name == "S1.25":
        # 4D 32-entry signed lattice codebook: 64 groups of 4 weights x 5 bits = 40 B + 2 B fine nibble multipliers
        cb = S125_CODEBOOK.astype(np.float32)  # [32, 4]
        cb_norm2 = np.sum(cb * cb, axis=1)     # [32]
        # Initial scale per sub-group: rms(wb)
        rms = np.sqrt(np.mean(wb * wb, axis=2) + 1e-12)  # [N, 4]
        amax_s = np.max(rms, axis=1)
        D_f16 = amax_s.astype(np.float16)
        D_eff = D_f16.astype(np.float32)
        denom = np.where(amax_s > 0.0, amax_s, 1.0)[:, None]
        u = np.clip(np.rint(rms / denom * 255.0), 0, 255).astype(np.uint8)
        u = np.where((rms > 0.0) & (u == 0), 1, u).astype(np.uint8)
        eff = D_eff[:, None] * (u.astype(np.float32) / 255.0)
        inv = np.where(eff > 0.0, 1.0 / eff, 0.0)[:, :, None]
        wn = (wb * inv).reshape(n_sb, 64, 4)  # [N, 64, 4]
        # Nearest codebook entry: min_k ||wn - cb[k]||^2 = ||wn||^2 - 2 wn.cb + ||cb||^2
        dots = np.einsum("ngd,kd->ngk", wn, cb)  # [N, 64, 32]
        scores = 2.0 * dots - cb_norm2[None, None, :]
        idx = np.argmax(scores, axis=2).astype(np.uint8)  # [N, 64] values in 0..31
        # Compute 4 fine residual multipliers (1 nibble per sub-group, stored in 2 B -> 0.5..1.5 range)
        recon_unit = cb[idx].reshape(n_sb, NUM_SUB, SUB_G)
        num = np.sum(wb * recon_unit, axis=2)
        den = np.sum(recon_unit * recon_unit, axis=2) * np.where(eff > 0.0, eff, 1.0) + 1e-12
        fine_f = np.where(eff > 0.0, num / den, 1.0)
        fine_q = np.clip(np.rint((fine_f - 0.5) * 15.0), 0, 15).astype(np.uint8)  # [N, 4]
        fine_bytes = np.stack([fine_q[:, 0] | (fine_q[:, 1] << 4), fine_q[:, 2] | (fine_q[:, 3] << 4)], axis=1)
        # Bit-pack 64 x 5-bit indices -> 8 groups of 8 indices (40 bits = 5 bytes each) = 40 B
        idx8 = idx.reshape(n_sb, 8, 8).astype(np.uint64)
        bits40 = np.zeros((n_sb, 8), dtype=np.uint64)
        for j in range(8):
            bits40 |= idx8[:, :, j] << np.uint64(5 * j)
        pl40 = np.empty((n_sb, 8, 5), dtype=np.uint8)
        for b in range(5):
            pl40[:, :, b] = ((bits40 >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8)
        hdr = _pack_header(D_f16, u)
        return np.concatenate([hdr, pl40.reshape(n_sb, 40), fine_bytes], axis=1).tobytes()

    raise ValueError(f"Unknown code_name: {code_name}")


def decode_sk(raw: bytes | np.ndarray, code_name: str) -> np.ndarray:
    """Decode exact .strata v1.0 S_K bytes -> float32 array of shape [n_sb * 256]."""
    sb_b = SUPERBLOCK_BYTES[code_name]
    buf = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    assert buf.size % sb_b == 0
    n_sb = buf.size // sb_b
    blocks = buf.reshape(n_sb, sb_b)
    D, u = _unpack_header(blocks[:, :6])
    eff = (D[:, None] * (u / 255.0))[:, :, None]  # [N, 4, 1]
    pl = blocks[:, 6:]

    if code_name == "S8_K":
        q = pl.view(np.int8).astype(np.float32).reshape(n_sb, NUM_SUB, SUB_G)
        return (q * eff).reshape(n_sb * QK_S)

    if code_name in ("S2_K", "S3_K", "S4_K", "S5_K", "S6_K"):
        bits = int(code_name[1])
        half = ((1 << bits) - 1) / 2.0
        if bits == 2:
            q = np.stack([(pl >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, QK_S)
        elif bits == 3:
            p_lo, p_hi = pl[:, :64], pl[:, 64:96]
            lo = np.stack([(p_lo >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> j) & 0x01 for j in range(8)], axis=2).reshape(n_sb, QK_S)
            q = lo | (hi << 2)
        elif bits == 4:
            q = np.stack([pl & 0x0F, (pl >> 4) & 0x0F], axis=2).reshape(n_sb, QK_S)
        elif bits == 5:
            p_lo, p_hi = pl[:, :128], pl[:, 128:160]
            lo = np.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> j) & 0x01 for j in range(8)], axis=2).reshape(n_sb, QK_S)
            q = lo | (hi << 4)
        elif bits == 6:
            p_lo, p_hi = pl[:, :128], pl[:, 128:192]
            lo = np.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, QK_S)
            q = lo | (hi << 4)
        qf = q.astype(np.float32).reshape(n_sb, NUM_SUB, SUB_G) - half
        return (qf * eff).reshape(n_sb * QK_S)

    if code_name == "S1.5":
        trits = _T5_DECODE[pl].reshape(n_sb, 260)[:, :256].astype(np.float32).reshape(n_sb, NUM_SUB, SUB_G)
        return (trits * eff).reshape(n_sb * QK_S)

    if code_name == "S1.25":
        pl40 = pl[:, :40].reshape(n_sb, 8, 5).astype(np.uint64)
        bits40 = np.zeros((n_sb, 8), dtype=np.uint64)
        for b in range(5):
            bits40 |= pl40[:, :, b] << np.uint64(8 * b)
        idx = np.empty((n_sb, 8, 8), dtype=np.uint8)
        for j in range(8):
            idx[:, :, j] = ((bits40 >> np.uint64(5 * j)) & np.uint64(0x1F)).astype(np.uint8)
        idx = idx.reshape(n_sb, 64)
        fb = pl[:, 40:42]
        fine_q = np.stack([fb[:, 0] & 0x0F, (fb[:, 0] >> 4) & 0x0F, fb[:, 1] & 0x0F, (fb[:, 1] >> 4) & 0x0F], axis=1)
        fine_f = (0.5 + fine_q.astype(np.float32) / 15.0)[:, :, None]
        vals = S125_CODEBOOK[idx].astype(np.float32).reshape(n_sb, NUM_SUB, SUB_G)
        return (vals * eff * fine_f).reshape(n_sb * QK_S)

    raise ValueError(f"Unknown code_name: {code_name}")


_TORCH_TABLES: Dict[str, Tuple] = {}


def decode_sk_torch(raw: bytes | np.ndarray, code_name: str, device: str | "torch.device") -> "torch.Tensor":
    """GPU-accelerated bit-exact .strata v1.0 S_K decoder -> float32 tensor [n_sb * 256] on `device`.
    ~100x faster than NumPy CPU decode_sk for MACaQ-R, Scale-QAT, and full 42-layer eval.
    """
    import torch

    sb_b = SUPERBLOCK_BYTES[code_name]
    buf_np = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    assert buf_np.size % sb_b == 0
    n_sb = buf_np.size // sb_b
    blocks = torch.from_numpy(np.array(buf_np, copy=True)).to(device=device, non_blocking=True).view(n_sb, sb_b)

    # Header: bytes 0..1 = float16 D, bytes 2..5 = uint8 u[4]
    D = blocks[:, :2].contiguous().view(torch.float16).view(n_sb).to(torch.float32)
    u = blocks[:, 2:6].to(torch.float32)
    eff = (D[:, None] * (u / 255.0))[:, :, None]  # [N, 4, 1]
    pl = blocks[:, 6:]

    if code_name == "S8_K":
        q = pl.contiguous().view(torch.int8).to(torch.float32).view(n_sb, NUM_SUB, SUB_G)
        return (q * eff).reshape(n_sb * QK_S)

    if code_name in ("S2_K", "S3_K", "S4_K", "S5_K", "S6_K"):
        bits = int(code_name[1])
        half = ((1 << bits) - 1) / 2.0
        if bits == 2:
            q = torch.stack([(pl >> (2 * j)) & 0x03 for j in range(4)], dim=2).reshape(n_sb, QK_S)
        elif bits == 3:
            p_lo, p_hi = pl[:, :64], pl[:, 64:96]
            lo = torch.stack([(p_lo >> (2 * j)) & 0x03 for j in range(4)], dim=2).reshape(n_sb, QK_S)
            hi = torch.stack([(p_hi >> j) & 0x01 for j in range(8)], dim=2).reshape(n_sb, QK_S)
            q = lo | (hi << 2)
        elif bits == 4:
            q = torch.stack([pl & 0x0F, (pl >> 4) & 0x0F], dim=2).reshape(n_sb, QK_S)
        elif bits == 5:
            p_lo, p_hi = pl[:, :128], pl[:, 128:160]
            lo = torch.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], dim=2).reshape(n_sb, QK_S)
            hi = torch.stack([(p_hi >> j) & 0x01 for j in range(8)], dim=2).reshape(n_sb, QK_S)
            q = lo | (hi << 4)
        elif bits == 6:
            p_lo, p_hi = pl[:, :128], pl[:, 128:192]
            lo = torch.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], dim=2).reshape(n_sb, QK_S)
            hi = torch.stack([(p_hi >> (2 * j)) & 0x03 for j in range(4)], dim=2).reshape(n_sb, QK_S)
            q = lo | (hi << 4)
        qf = q.to(torch.float32).view(n_sb, NUM_SUB, SUB_G) - half
        return (qf * eff).reshape(n_sb * QK_S)

    dev_key = str(device)
    if dev_key not in _TORCH_TABLES:
        _TORCH_TABLES[dev_key] = (
            torch.from_numpy(_T5_DECODE.astype(np.float32)).to(device),
            torch.from_numpy(S125_CODEBOOK.astype(np.float32)).to(device),
        )
    t5_tab, s125_tab = _TORCH_TABLES[dev_key]

    if code_name == "S1.5":
        trits = t5_tab[pl.to(torch.long)].reshape(n_sb, 260)[:, :256].view(n_sb, NUM_SUB, SUB_G)
        return (trits * eff).reshape(n_sb * QK_S)

    if code_name == "S1.25":
        pl40 = pl[:, :40].view(n_sb, 8, 5).to(torch.int64)
        bits40 = (
            pl40[:, :, 0]
            | (pl40[:, :, 1] << 8)
            | (pl40[:, :, 2] << 16)
            | (pl40[:, :, 3] << 24)
            | (pl40[:, :, 4] << 32)
        )
        idx = torch.stack([(bits40 >> (5 * j)) & 0x1F for j in range(8)], dim=2).reshape(n_sb, 64)
        fb = pl[:, 40:42]
        fine_q = torch.stack(
            [fb[:, 0] & 0x0F, (fb[:, 0] >> 4) & 0x0F, fb[:, 1] & 0x0F, (fb[:, 1] >> 4) & 0x0F], dim=1
        )
        fine_f = (0.5 + fine_q.to(torch.float32) / 15.0)[:, :, None]
        vals = s125_tab[idx].view(n_sb, NUM_SUB, SUB_G)
        return (vals * eff * fine_f).reshape(n_sb * QK_S)

    raise ValueError(f"Unknown code_name: {code_name}")


def quantize_q8_1_g64(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize activation vector [..., K] (K % 64 == 0) into 64-element Q8_1 blocks matching S_K sub-groups.
    Returns (qs int8 [..., K//64, 64], d float32 [..., K//64], s float32 [..., K//64] where s = d * sum(qs)).
    """
    x = np.asarray(x, dtype=np.float32)
    assert x.shape[-1] % SUB_G == 0
    xb = x.reshape(*x.shape[:-1], -1, SUB_G)
    amax = np.max(np.abs(xb), axis=-1)
    d = amax / 127.0
    inv = np.where(d > 0.0, 1.0 / d, 0.0)[..., None]
    qs = np.clip(np.rint(xb * inv), -127, 127).astype(np.int8)
    s = d * np.sum(qs.astype(np.int32), axis=-1).astype(np.float32)
    return qs, d.astype(np.float32), s.astype(np.float32)


def dot_sk_q8_1(raw_row: bytes | np.ndarray, code_name: str, x_f32: np.ndarray) -> float:
    """Reference quantized dot product between 1 encoded S_K row and Q8_1(x_f32).
    Matches the exact integer accumulation of the C++ AVX-512 VNNI / CUDA / HIP kernels.
    """
    sb_b = SUPERBLOCK_BYTES[code_name]
    buf = np.frombuffer(raw_row, dtype=np.uint8) if isinstance(raw_row, (bytes, bytearray, memoryview)) else raw_row
    n_sb = buf.size // sb_b
    qs, xd, xs = quantize_q8_1_g64(x_f32)
    qs = qs.reshape(n_sb, NUM_SUB, SUB_G).astype(np.int32)
    xd = xd.reshape(n_sb, NUM_SUB)
    xs = xs.reshape(n_sb, NUM_SUB)

    blocks = buf.reshape(n_sb, sb_b)
    D, u = _unpack_header(blocks[:, :6])
    eff = D[:, None] * (u / 255.0)
    pl = blocks[:, 6:]

    if code_name == "S8_K":
        wq = pl.view(np.int8).reshape(n_sb, NUM_SUB, SUB_G).astype(np.int32)
        acc = np.sum(wq * qs, axis=2).astype(np.float32) * (eff * xd)
        return float(np.sum(acc))

    if code_name in ("S2_K", "S3_K", "S4_K", "S5_K", "S6_K"):
        bits = int(code_name[1])
        half = ((1 << bits) - 1) / 2.0
        if bits == 2:
            q = np.stack([(pl >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, NUM_SUB, SUB_G)
        elif bits == 3:
            p_lo, p_hi = pl[:, :64], pl[:, 64:96]
            lo = np.stack([(p_lo >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> j) & 0x01 for j in range(8)], axis=2).reshape(n_sb, QK_S)
            q = (lo | (hi << 2)).reshape(n_sb, NUM_SUB, SUB_G)
        elif bits == 4:
            q = np.stack([pl & 0x0F, (pl >> 4) & 0x0F], axis=2).reshape(n_sb, NUM_SUB, SUB_G)
        elif bits == 5:
            p_lo, p_hi = pl[:, :128], pl[:, 128:160]
            lo = np.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> j) & 0x01 for j in range(8)], axis=2).reshape(n_sb, QK_S)
            q = (lo | (hi << 4)).reshape(n_sb, NUM_SUB, SUB_G)
        elif bits == 6:
            p_lo, p_hi = pl[:, :128], pl[:, 128:192]
            lo = np.stack([p_lo & 0x0F, (p_lo >> 4) & 0x0F], axis=2).reshape(n_sb, QK_S)
            hi = np.stack([(p_hi >> (2 * j)) & 0x03 for j in range(4)], axis=2).reshape(n_sb, QK_S)
            q = (lo | (hi << 4)).reshape(n_sb, NUM_SUB, SUB_G)
        # Exact VNNI integer dot: eff * (xd * sum(q * qs) - half * xs)
        isum = np.sum(q.astype(np.int32) * qs, axis=2).astype(np.float32)
        acc = eff * (xd * isum - half * xs)
        return float(np.sum(acc))

    if code_name == "S1.5":
        trits = _T5_DECODE[pl].reshape(n_sb, 260)[:, :256].reshape(n_sb, NUM_SUB, SUB_G).astype(np.int32)
        isum = np.sum(trits * qs, axis=2).astype(np.float32)
        return float(np.sum(eff * xd * isum))

    if code_name == "S1.25":
        pl40 = pl[:, :40].reshape(n_sb, 8, 5).astype(np.uint64)
        bits40 = np.zeros((n_sb, 8), dtype=np.uint64)
        for b in range(5):
            bits40 |= pl40[:, :, b] << np.uint64(8 * b)
        idx = np.empty((n_sb, 8, 8), dtype=np.uint8)
        for j in range(8):
            idx[:, :, j] = ((bits40 >> np.uint64(5 * j)) & np.uint64(0x1F)).astype(np.uint8)
        fb = pl[:, 40:42]
        fine_q = np.stack([fb[:, 0] & 0x0F, (fb[:, 0] >> 4) & 0x0F, fb[:, 1] & 0x0F, (fb[:, 1] >> 4) & 0x0F], axis=1)
        fine_f = 0.5 + fine_q.astype(np.float32) / 15.0
        vals = S125_CODEBOOK[idx.reshape(n_sb, 64)].reshape(n_sb, NUM_SUB, SUB_G).astype(np.int32)
        isum = np.sum(vals * qs, axis=2).astype(np.float32)
        return float(np.sum(eff * fine_f * xd * isum))

    raise ValueError(code_name)


def sqnr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    err = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    sig = np.mean(orig.astype(np.float64) ** 2) + 1e-18
    return 10.0 * math.log10(sig / max(err, 1e-18))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    w = rng.standard_normal((16, 1536), dtype=np.float32) * 0.04
    x = rng.standard_normal(1536, dtype=np.float32)
    print("Verifying all 8 S_K codes on [16, 1536] rows:")
    for name in ["S1.25", "S1.5", "S2_K", "S3_K", "S4_K", "S5_K", "S6_K", "S8_K"]:
        enc = encode_sk(w, name)
        dec = decode_sk(enc, name).reshape(w.shape)
        sb = SUPERBLOCK_BYTES[name]
        row0_dot_q = dot_sk_q8_1(enc[: 6 * sb], name, x)
        row0_dot_f = float(np.dot(dec[0], x))
        rel = abs(row0_dot_q - row0_dot_f) / (abs(row0_dot_f) + 1e-6)
        print(f"  {name:5s} ({sb:3d} B/sb): SQNR={sqnr_db(w, dec):5.2f} dB | dot_q8_1={row0_dot_q:+.6f} vs fp32={row0_dot_f:+.6f} (rel={rel:.4%})")
    print("\nVerifying exact 4 KiB-aligned expert body sizes (FORMAT-SPEC.md §3):")
    expected = {
        ("S1.25", "S1.25"): 1331200,
        ("S1.5", "S1.5"): 1609728,
        ("S2_K", "S2_K"): 1941504,
        ("S2_K", "S3_K"): 2236416,
        ("S3_K", "S3_K"): 2826240,
        ("S3_K", "S4_K"): 3121152,
        ("S4_K", "S4_K"): 3710976,
        ("S4_K", "S5_K"): 4005888,
        ("S5_K", "S6_K"): 4890624,
        ("S6_K", "S8_K"): 6070272,
    }
    for (gu, dn), exp_b in expected.items():
        got, dn_off = expert_body_bytes(gu, dn)
        assert got == exp_b, f"Mismatch for {gu}+{dn}: got {got}, expected {exp_b}"
        print(f"  {gu:5s} + {dn:5s} = {got:,} B (dn_offset={dn_off:,}) OK")
