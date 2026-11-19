"""Layer-streamed PyTorch reference & capture forward for Tencent Hy4-Flash (hy_v4_flash) + 5-Layer DFly2.

Implements every exact invariant from reference/hy-v4-flash.cpp & config.json:
  - 4-stream iHC (hc_mult=4, hc_magnitude=2.0, hc_eps=1e-6) + Sandwich RMSNorms (rms_norm_eps=1e-5)
  - 4-module Engram (layers 1, 11, 21, 31 0-based; HY_V4_FLASH_ENGRAM_EPS = 1.1920929e-7f;
    rolling XOR hash with ENGRAM_SEED_STRIDE=10007; sqrt-sign gate; engram_pre_gamma; engram_hc_scale)
    supporting BOTH BF16 safetensors AND packed lossless Hy4-Flash-Engram-Q8.strata (40 rows/4K page)!
  - Hybrid MLA (12 Sparse 32-head rope_theta=1e7 + Compressed IndexPool kpool=4 group_topk=512;
    30 SWA 64-head rope_theta=1e4 window=512), interleaved MRoPE (text plane -> standard interleaved RoPE),
    learnable_sink_param, linear_gate sigmoid output gate
  - LatentMoE (layers 1..41): sigmoid router + e_score_correction_bias -> top-16 normalized * 3.998;
    routed_expert_down_proj [1536, 3072] -> 512 square [1536, 1536] experts with swiglu_limit=10.0 clamp ->
    routed_expert_up_proj [3072, 1536] + unclamped shared_experts
  - Full 5-layer DFly2 drafter (mtp_fusion, hidden_correction, dfly_layers.0..4 sharing MLA KV & IndexPool
    from target sparse layers [25, 29, 33, 37, 41], final_norm, confidence_head)
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .pack_engram_vision import (
    ENGRAM_ROW_BYTES,
    ENGRAM_ROW_DIM,
    ROWS_PER_4K_PAGE,
    compute_engram_metadata,
)
from .strata_format import PAGE_SIZE, parse_header_frame

ENGRAM_EPS = 1.1920929e-7  # FLT_EPSILON exact match to hy-v4-flash.cpp


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * inv_rms
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(x.dtype)


def apply_interleaved_rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Interleaved RoPE (GGML/MRoPE text mode: pairs (2i, 2i+1)) on last dim D=64.
    x: [..., T, H, D] or [T, H, D], pos: [T] int64.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    D = xf.shape[-1]
    half = D // 2
    freqs = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    angles = pos.to(torch.float32)[:, None] * freqs[None, :]  # [T, D/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # x0/x1 have shape [T, ..., D/2]; insert singleton dims between axis 0 (T) and last axis (D/2)
    while cos.ndim < xf.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack([out0, out1], dim=-1).flatten(-2)
    return out.to(orig_dtype)


class ShardLoader:
    """Fast LRU-cached safetensors tensor loader for streaming 440.5 GB checkpoints with low RAM."""

    def __init__(self, model_dir: Path, max_open: int = 8):
        self.model_dir = Path(model_dir)
        self.cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.tc = self.cfg["text_config"]
        idx = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.wm: Dict[str, str] = idx["weight_map"]
        self._handles: Dict[str, safe_open] = {}
        self._lru: List[str] = []
        self.max_open = max_open
        self._gpu_cache: Dict[Tuple[str, str, Optional[torch.dtype]], torch.Tensor] = {}
        self.cache_enabled = False

    def clear_gpu_cache(self) -> None:
        self._gpu_cache.clear()

    def _get_handle(self, fname: str):
        if fname in self._handles:
            self._lru.remove(fname)
            self._lru.append(fname)
            return self._handles[fname]
        if len(self._lru) >= self.max_open:
            old = self._lru.pop(0)
            self._handles.pop(old, None)
        h = safe_open(str(self.model_dir / fname), framework="pt", device="cpu")
        self._handles[fname] = h
        self._lru.append(fname)
        return h

    def get(self, key: str, device: str = "cuda", dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self.cache_enabled:
            ck = (key, str(device), dtype)
            if ck in self._gpu_cache:
                return self._gpu_cache[ck]
        h = self._get_handle(self.wm[key])
        t = h.get_tensor(key)
        if dtype is not None:
            t = t.to(dtype)
        out = t.to(device, non_blocking=True)
        if self.cache_enabled:
            self._gpu_cache[(key, str(device), dtype)] = out
        return out

    def get_slice(self, key: str):
        h = self._get_handle(self.wm[key])
        return h.get_slice(key)

    def has(self, key: str) -> bool:
        return key in self.wm


class EngramTableReader:
    """Reads 4-module Engram rows either from packed Hy4-Flash-Engram-Q8.strata (4K pages) or BF16 safetensors."""

    def __init__(self, model_dir: Path, engram_q8_path: Optional[Path] = None):
        self.model_dir = Path(model_dir)
        self.meta = compute_engram_metadata(self.model_dir)
        self.token_map = np.array(self.meta["engram_token_map"], dtype=np.int64)
        self.pad_cid = int(self.token_map[self.meta["engram_pad_id"]])
        self.n_gram = int(self.meta["ple_ngram_size"])
        self.n_head = int(self.meta["ple_heads_per_ngram"])
        self.n_cols = (self.n_gram - 1) * self.n_head  # 16
        self.ple_layers: List[int] = list(self.meta["ple_layers"])
        self.mults = np.array(self.meta["ple_layer_multipliers"], dtype=np.uint64).reshape(len(self.ple_layers), self.n_gram)
        self.primes = np.array(self.meta["ple_head_vocab_sizes"], dtype=np.uint64).reshape(len(self.ple_layers), self.n_cols)
        self.offsets = np.array(self.meta["ple_head_offsets"], dtype=np.int64).reshape(len(self.ple_layers), self.n_cols)
        self.module_base = [int(self.offsets[e, 0]) for e in range(len(self.ple_layers))]

        self.q8_file = None
        self.q8_sec4_offset = 0
        self.loader = None
        if engram_q8_path is not None and Path(engram_q8_path).is_file():
            self.q8_file = open(engram_q8_path, "rb")
            frame = self.q8_file.read(PAGE_SIZE)
            _, _, sections, _ = parse_header_frame(frame)
            self.q8_sec4_offset = sections[4].file_offset
        else:
            self.loader = ShardLoader(self.model_dir, max_open=4)

    def hash_tokens(self, mod_idx: int, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exact match to hy-v4-flash.cpp::engram_hash for a single sequence token_ids [T].
        Returns (global_row_ids [T, 16] int64, valid [T, 16] float32).
        """
        T = len(token_ids)
        mult = self.mults[mod_idx]
        primes = self.primes[mod_idx]
        offs = self.offsets[mod_idx]
        cids = np.where(
            (token_ids >= 0) & (token_ids < len(self.token_map)),
            self.token_map[np.clip(token_ids, 0, len(self.token_map) - 1)],
            -1,
        )
        ids = np.empty((T, self.n_cols), dtype=np.int64)
        valid = np.ones((T, self.n_cols), dtype=np.float32)
        for i in range(T):
            dead = cids[i] < 0
            t0 = np.uint64(self.pad_cid if dead else cids[i])
            t1 = np.uint64(self.pad_cid if (i - 1 < 0 or cids[i - 1] < 0) else cids[i - 1])
            t2 = np.uint64(self.pad_cid if (i - 2 < 0 or cids[i - 2] < 0) else cids[i - 2])
            r1 = (t0 * mult[0]) ^ (t1 * mult[1])
            r2 = r1 ^ (t2 * mult[2])
            for h in range(self.n_head):
                ids[i, h] = int(r1 % primes[h]) + int(offs[h])
                ids[i, self.n_head + h] = int(r2 % primes[self.n_head + h]) + int(offs[self.n_head + h])
            if dead:
                valid[i, :] = 0.0
        return ids, valid

    def gather_rows(self, mod_idx: int, global_ids: np.ndarray, device: str = "cuda") -> torch.Tensor:
        """Gather [T, 16, 96] float32 rows from either Engram-Q8.strata or BF16 safetensors."""
        shape = global_ids.shape
        flat_ids = global_ids.reshape(-1)
        if self.q8_file is not None:
            # Read 4 KiB pages (deduplicated) from Hy4-Flash-Engram-Q8.strata
            page_ids = flat_ids // ROWS_PER_4K_PAGE
            in_page = flat_ids % ROWS_PER_4K_PAGE
            uniq_pages, inv_idx = np.unique(page_ids, return_inverse=True)
            page_buf = np.empty((len(uniq_pages), PAGE_SIZE), dtype=np.uint8)
            for idx_u, p in enumerate(uniq_pages):
                self.q8_file.seek(self.q8_sec4_offset + int(p) * PAGE_SIZE)
                page_buf[idx_u] = np.frombuffer(self.q8_file.read(PAGE_SIZE), dtype=np.uint8)
            raw_rows = np.empty((len(flat_ids), ENGRAM_ROW_BYTES), dtype=np.uint8)
            for i in range(len(flat_ids)):
                o = int(in_page[i]) * ENGRAM_ROW_BYTES
                raw_rows[i] = page_buf[inv_idx[i], o : o + ENGRAM_ROW_BYTES]
            scales = np.ascontiguousarray(raw_rows[:, :4]).view(np.float32).reshape(-1, 1)
            q_i8 = torch.from_numpy(np.ascontiguousarray(raw_rows[:, 4:]).view(np.int8)).to(device)
            vals = q_i8.to(torch.float32) * torch.from_numpy(scales).to(device)
            return vals.reshape(*shape, ENGRAM_ROW_DIM)
        else:
            lid0 = self.ple_layers[mod_idx]
            local_ids = flat_ids - self.module_base[mod_idx]
            key = f"model.layers.{lid0}.engram_layer.multi_head_embedding.embedding.weight"
            sl = self.loader.get_slice(key)
            uniq_ids, inv_idx = np.unique(local_ids, return_inverse=True)
            # Gather unique rows from safetensors slice
            rows = torch.empty((len(uniq_ids), ENGRAM_ROW_DIM), dtype=torch.bfloat16)
            for idx_u, rid in enumerate(uniq_ids):
                rows[idx_u] = sl[int(rid)]
            out = rows[torch.from_numpy(inv_idx)].to(device=device, dtype=torch.float32)
            return out.reshape(*shape, ENGRAM_ROW_DIM)


class Hy4FlashStreamedModel:
    """Layer-streamed PyTorch forward for Tencent Hy4-Flash (42 layers + 5-layer DFly2).
    Runs full 220B BF16 inference cleanly on a single GPU (~8 GB VRAM peak) and supports:
      - Exact parity verification (Q0-02)
      - 3-layer pilot substitution (Q1-06)
      - Layer-by-layer LatentMoE Hessian & routing capture (Q2-03)
    """

    def __init__(self, model_dir: Path, engram_q8_path: Optional[Path] = None, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.loader = ShardLoader(self.model_dir, max_open=8)
        self.tc = self.loader.tc
        self.engram = EngramTableReader(self.model_dir, engram_q8_path=engram_q8_path)
        self.n_layers = int(self.tc["num_hidden_layers"])
        self.hidden_size = int(self.tc["hidden_size"])
        self.hc_mult = int(self.tc["hc_mult"])
        self.hc_eps = float(self.tc["hc_eps"])
        self.hc_magnitude = float(self.tc["hc_magnitude"])
        self.rms_eps = float(self.tc["rms_norm_eps"])
        self.q_lora_rank = int(self.tc["q_lora_rank"])
        self.kv_lora_rank = int(self.tc["kv_lora_rank"])
        self.qk_nope_dim = int(self.tc["qk_nope_head_dim"])
        self.qk_rope_dim = int(self.tc["qk_rope_head_dim"])
        self.qk_head_dim = int(self.tc["qk_head_dim"])
        self.v_head_dim = int(self.tc["v_head_dim"])
        self.swa_window = int(self.tc["sliding_window"])
        self.kpool = int(self.tc["compressed_indexer_compress_ratio"])
        self.index_topk = int(self.tc["index_topk"])
        self.n_idx_heads = int(self.tc["compressed_indexer_num_attention_heads"])
        self.idx_head_dim = int(self.tc["compressed_indexer_head_dim"])
        self.swiglu_limit = float(self.tc["swiglu_limit"])
        self.routed_scale = float(self.tc["routed_scaling_factor"])
        self.n_experts = int(self.tc["n_routed_experts"])
        self.top_k = int(self.tc["num_experts_per_tok"])
        self.layer_types = list(self.tc["layer_types"])
        self.engram_layers = {l: i for i, l in enumerate(self.engram.ple_layers)}

    def _hc_pre(self, x: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [T, 4, 3072] -> (cur [T, 3072], post [T, 4])."""
        T = x.shape[0]
        hc_fn = self.loader.get(f"{prefix}.hc_pre.hc_fn", self.device, torch.float32)
        hc_scale = self.loader.get(f"{prefix}.hc_pre.hc_scale", self.device, torch.float32)
        hc_base = self.loader.get(f"{prefix}.hc_pre.hc_base", self.device, torch.float32)
        flat = x.reshape(T, self.hc_mult * self.hidden_size)
        flat_norm = rms_norm(flat, None, self.rms_eps).to(torch.float32)
        mixes = F.linear(flat_norm, hc_fn)  # [T, 8]
        pre = torch.sigmoid(mixes[:, : self.hc_mult] * hc_scale[0] + hc_base[: self.hc_mult]) + self.hc_eps
        post = (
            self.hc_magnitude * torch.sigmoid(mixes[:, self.hc_mult :] * hc_scale[1] + hc_base[self.hc_mult :])
            + self.hc_eps
        )
        cur = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        return cur, post

    def _hc_post(self, cur: torch.Tensor, residual: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return (residual.to(torch.float32) + cur.to(torch.float32).unsqueeze(1) * post.unsqueeze(-1)).to(torch.bfloat16)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        hc_fn = self.loader.get("model.hc_head.hc_head_fn", self.device, torch.float32)
        hc_scale = self.loader.get("model.hc_head.hc_head_scale", self.device, torch.float32)
        hc_base = self.loader.get("model.hc_head.hc_head_base", self.device, torch.float32)
        flat = x.reshape(T, self.hc_mult * self.hidden_size)
        flat_norm = rms_norm(flat, None, self.rms_eps).to(torch.float32)
        mixes = F.linear(flat_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        return torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)

    def _apply_engram(self, x: torch.Tensor, il: int, token_ids_np: np.ndarray) -> torch.Tensor:
        mod_idx = self.engram_layers[il]
        ids, valid_np = self.engram.hash_tokens(mod_idx, token_ids_np)
        rows = self.engram.gather_rows(mod_idx, ids, device=self.device)  # [T, 16, 96] f32
        valid = torch.from_numpy(valid_np).to(self.device)                # [T, 16] f32
        T = x.shape[0]
        p = f"model.layers.{il}.engram_layer"

        ln_w = torch.stack(
            [self.loader.get(f"{p}.lookup_per_head_norms.{c}.weight", self.device, torch.float32) for c in range(16)],
            dim=0,
        )  # [16, 96]
        r = rms_norm(rows, None, ENGRAM_EPS) * ln_w.unsqueeze(0) * valid.unsqueeze(-1)
        r = r.reshape(T, 16 * ENGRAM_ROW_DIM)  # [T, 1536]

        kv_w = torch.cat(
            [self.loader.get(f"{p}.key_projs.{c}.weight", self.device, torch.float32) for c in range(self.hc_mult)]
            + [self.loader.get(f"{p}.value_proj.weight", self.device, torch.float32)],
            dim=0,
        )  # [5*3072, 1536]
        kv_b = torch.stack(
            [self.loader.get(f"{p}.key_projs.{c}.bias", self.device, torch.float32) for c in range(self.hc_mult)]
            + [self.loader.get(f"{p}.value_proj.bias", self.device, torch.float32)],
            dim=0,
        )  # [5, 3072]
        kv = F.linear(r, kv_w).reshape(T, self.hc_mult + 1, self.hidden_size) + kv_b.unsqueeze(0)
        key = kv[:, : self.hc_mult, :]    # [T, 4, 3072]
        value = kv[:, self.hc_mult :, :]  # [T, 1, 3072]

        nk = torch.stack([self.loader.get(f"{p}.norm_key.{c}.weight", self.device, torch.float32) for c in range(self.hc_mult)], dim=0)
        nq = torch.stack([self.loader.get(f"{p}.norm_query.{c}.weight", self.device, torch.float32) for c in range(self.hc_mult)], dim=0)
        egate = nk * nq  # [4, 3072]
        dot = (rms_norm(x.to(torch.float32), None, ENGRAM_EPS) * rms_norm(key, None, ENGRAM_EPS) * egate.unsqueeze(0)).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        mag = torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
        sign = torch.where(dot < 0.0, -1.0, 1.0)
        gate = torch.sigmoid(mag * sign)  # [T, 4, 1]

        pre_gamma = self.loader.get(f"{p}.unified_pre_gamma", self.device, torch.float32)  # [3072]
        hc_scale = torch.stack([self.loader.get(f"{p}.unified_per_hc_scales.{c}", self.device, torch.float32) for c in range(self.hc_mult)], dim=0)  # [4, 3072]
        signal = rms_norm(value, None, ENGRAM_EPS) * pre_gamma.view(1, 1, -1)  # [T, 1, 3072]
        delta = signal * (gate + hc_scale.unsqueeze(0))                        # [T, 4, 3072]
        any_valid = (valid.sum(dim=-1) > 0).to(torch.float32).view(T, 1, 1)
        return (x.to(torch.float32) + delta * any_valid).to(torch.bfloat16)

    def _indexpool_mask(self, cur: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, il: int, theta: float) -> Optional[torch.Tensor]:
        """Exact Compressed IndexPool mask [T, T] (0.0 for selected causal tokens, -inf elsewhere)."""
        T = cur.shape[0]
        if T <= self.index_topk:
            # Causal mask only when sequence <= index_topk (all previous groups fit within top-2048!)
            m = torch.triu(torch.full((T, T), float("-inf"), device=self.device, dtype=torch.float32), diagonal=1)
            return m
        p = f"model.layers.{il}.self_attn.compressed_indexer"
        w_q = self.loader.get(f"{p}.query_projection.weight", self.device, torch.bfloat16)
        w_k = self.loader.get(f"{p}.token_key_projection.weight", self.device, torch.bfloat16)
        kn = self.loader.get(f"{p}.token_key_norm.weight", self.device, torch.bfloat16)
        w_v = self.loader.get(f"{p}.key_value_projection.weight", self.device, torch.bfloat16)
        w_g = self.loader.get(f"{p}.key_gate_projection.weight", self.device, torch.bfloat16)
        ape = self.loader.get(f"{p}.intra_group_ape", self.device, torch.float32)  # [4, 128]
        pnorm = self.loader.get(f"{p}.key_norm.weight", self.device, torch.bfloat16)
        w_hw = self.loader.get(f"{p}.head_weight_projection.weight", self.device, torch.bfloat16)

        iq = F.linear(qr, w_q).reshape(T, self.n_idx_heads, self.idx_head_dim)
        iq_f, iq_b = iq[..., : self.idx_head_dim - self.qk_rope_dim], iq[..., self.idx_head_dim - self.qk_rope_dim :]
        iq = torch.cat([iq_f, apply_interleaved_rope(iq_b, pos, theta)], dim=-1).to(torch.float32)

        ik = rms_norm(F.linear(cur, w_k), kn, self.rms_eps)
        iv = F.linear(ik, w_v).to(torch.float32)
        ig = F.linear(ik, w_g).to(torch.float32)

        n_pools = T // self.kpool
        iv_p = iv[: n_pools * self.kpool].reshape(n_pools, self.kpool, self.idx_head_dim)
        ig_p = ig[: n_pools * self.kpool].reshape(n_pools, self.kpool, self.idx_head_dim)
        probs = torch.softmax(ig_p + ape.unsqueeze(0), dim=1)
        pooled = (probs * iv_p).sum(dim=1)  # [n_pools, 128]
        pooled = rms_norm(pooled, pnorm, self.rms_eps).to(torch.float32)
        pool_pos = torch.arange(0, n_pools * self.kpool, self.kpool, device=self.device, dtype=torch.int64)
        pf, pb = pooled[:, : self.idx_head_dim - self.qk_rope_dim], pooled[:, self.idx_head_dim - self.qk_rope_dim :]
        pb_r = apply_interleaved_rope(pb.unsqueeze(1), pool_pos, theta).squeeze(1)
        pooled = torch.cat([pf, pb_r], dim=-1)  # [n_pools, 128]

        hw = F.linear(cur, w_hw).to(torch.float32) / math.sqrt(self.idx_head_dim * self.n_idx_heads)  # [T, 16]
        # score[t, p] = sum_h relu(iq[t, h] . pooled[p]) * hw[t, h]
        kq = torch.einsum("thd,pd->thp", iq, pooled)
        score = (F.relu(kq) * hw.unsqueeze(-1)).sum(dim=1)  # [T, n_pools]
        # Only completed pools strictly before token's own group are scored; own group forced as tail
        tok_group = torch.arange(T, device=self.device) // self.kpool
        p_idx = torch.arange(n_pools, device=self.device)
        causal_pool = p_idx.unsqueeze(0) < tok_group.unsqueeze(1)
        score = score.masked_fill(~causal_pool, float("-inf"))
        n_top_pool = min(n_pools, self.index_topk // self.kpool - 1)
        _, top_p = torch.topk(score, k=n_top_pool, dim=-1)  # [T, n_top_pool]
        mask = torch.full((T, T), float("-inf"), device=self.device, dtype=torch.float32)
        # Unmask selected pools
        for off in range(self.kpool):
            tok_cols = top_p * self.kpool + off
            valid_c = tok_cols < torch.arange(T, device=self.device).unsqueeze(1)
            mask.scatter_(1, tok_cols.clamp(max=T - 1), torch.where(valid_c, 0.0, float("-inf")))
        # Unmask current token's own 4-token tail group (causal)
        for t in range(T):
            g0 = (t // self.kpool) * self.kpool
            mask[t, g0 : t + 1] = 0.0
        return mask

    def _attention(self, cur: torch.Tensor, pos: torch.Tensor, il: int, save_kv: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]:
        T = cur.shape[0]
        is_swa = self.layer_types[il] == "sliding_attention"
        n_head = int(self.tc["swa_num_attention_heads"] if is_swa else self.tc["num_attention_heads"])
        theta = float(self.tc["rope_parameters"]["sliding_attention" if is_swa else "deepseek_sparse_attention"]["rope_theta"])
        p = f"model.layers.{il}.self_attn"

        qr = rms_norm(
            F.linear(cur, self.loader.get(f"{p}.q_a_proj.weight", self.device, torch.bfloat16)),
            self.loader.get(f"{p}.q_a_layernorm.weight", self.device, torch.bfloat16),
            self.rms_eps,
        )
        q = F.linear(qr, self.loader.get(f"{p}.q_b_proj.weight", self.device, torch.bfloat16)).reshape(T, n_head, self.qk_head_dim)
        q_nope, q_pe = q[..., : self.qk_nope_dim], q[..., self.qk_nope_dim :]
        q_pe = apply_interleaved_rope(q_pe, pos, theta)

        kv_a = F.linear(cur, self.loader.get(f"{p}.kv_a_proj_with_mqa.weight", self.device, torch.bfloat16))
        kv_cmpr = rms_norm(
            kv_a[:, : self.kv_lora_rank],
            self.loader.get(f"{p}.kv_a_layernorm.weight", self.device, torch.bfloat16),
            self.rms_eps,
        )
        k_pe = apply_interleaved_rope(kv_a[:, self.kv_lora_rank :].unsqueeze(1), pos, theta)  # [T, 1, 64]

        kv_b = self.loader.get(f"{p}.kv_b_proj.weight", self.device, torch.bfloat16)
        kv_exp = F.linear(kv_cmpr, kv_b).reshape(T, n_head, self.qk_nope_dim + self.v_head_dim)
        k_nope = kv_exp[..., : self.qk_nope_dim]
        v = kv_exp[..., self.qk_nope_dim :]

        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)  # [H, T, 256]
        k_full = torch.cat([k_nope, k_pe.expand(T, n_head, self.qk_rope_dim)], dim=-1).transpose(0, 1)  # [H, T, 256]
        v_t = v.transpose(0, 1)  # [H, T, 256]

        scale = 1.0 / math.sqrt(self.qk_head_dim)
        scores = torch.matmul(q_full.to(torch.float32), k_full.to(torch.float32).transpose(-2, -1)) * scale  # [H, T, T]

        if is_swa:
            i_idx = torch.arange(T, device=self.device)[:, None]
            j_idx = torch.arange(T, device=self.device)[None, :]
            mask = torch.where((j_idx <= i_idx) & ((i_idx - j_idx) < self.swa_window), 0.0, float("-inf"))
            scores = scores + mask.unsqueeze(0)
            idx_mask = None
        else:
            idx_mask = self._indexpool_mask(cur, qr, pos, il, theta)
            scores = scores + idx_mask.unsqueeze(0)

        # Learnable attention sink (included in softmax denominator, dropped from weighted value sum!)
        sinks = self.loader.get(f"{p}.learnable_sink_param", self.device, torch.float32).view(n_head, 1, 1).expand(n_head, T, 1)
        scores_aug = torch.cat([scores, sinks], dim=-1)
        probs = torch.softmax(scores_aug, dim=-1)[..., :-1].to(torch.bfloat16)  # [H, T, T]
        attn = torch.matmul(probs, v_t).transpose(0, 1).reshape(T, n_head * self.v_head_dim)

        gate = torch.sigmoid(F.linear(cur, self.loader.get(f"{p}.linear_gate.weight", self.device, torch.bfloat16)).to(torch.float32)).to(torch.bfloat16)
        attn = attn * gate
        out = F.linear(attn, self.loader.get(f"{p}.o_proj.weight", self.device, torch.bfloat16))
        out = rms_norm(out, self.loader.get(f"{p}.post_proj_norm.weight", self.device, torch.bfloat16), self.rms_eps)
        kv_pack = (kv_cmpr.detach(), k_pe.detach(), idx_mask.detach() if idx_mask is not None else None) if save_kv else None
        return out, kv_pack

    def _dense_mlp(self, cur: torch.Tensor, il: int, quant_override: Optional[Dict] = None) -> torch.Tensor:
        p = f"model.layers.{il}.mlp"
        wg = quant_override["gate"] if (quant_override and "gate" in quant_override) else self.loader.get(f"{p}.gate_proj.weight", self.device, torch.bfloat16)
        wu = quant_override["up"] if (quant_override and "up" in quant_override) else self.loader.get(f"{p}.up_proj.weight", self.device, torch.bfloat16)
        wd = quant_override["down"] if (quant_override and "down" in quant_override) else self.loader.get(f"{p}.down_proj.weight", self.device, torch.bfloat16)
        out = F.linear(F.silu(F.linear(cur, wg)) * F.linear(cur, wu), wd)
        return rms_norm(out, self.loader.get(f"{p}.post_fc2_norm.weight", self.device, torch.bfloat16), self.rms_eps)

    def _latent_moe(
        self,
        cur: torch.Tensor,
        il: int,
        quant_override: Optional[Dict] = None,
        capture_hook: Optional[Callable] = None,
    ) -> torch.Tensor:
        T = cur.shape[0]
        p = f"model.layers.{il}.mlp"
        w_router = self.loader.get(f"{p}.gate.weight", self.device, torch.float32)
        b_corr = self.loader.get(f"{p}.gate.e_score_correction_bias", self.device, torch.float32)
        logits = F.linear(cur.to(torch.float32), w_router)  # [T, 512]
        probs = torch.sigmoid(logits)
        _, topk_idx = torch.topk(probs + b_corr.unsqueeze(0), k=self.top_k, dim=-1)  # [T, 16]
        topk_p = probs.gather(1, topk_idx)
        topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scale  # [T, 16]

        w_rdown = self.loader.get(f"{p}.routed_expert_down_proj.weight", self.device, torch.bfloat16)
        w_rup = self.loader.get(f"{p}.routed_expert_up_proj.weight", self.device, torch.bfloat16)
        z = F.linear(cur, w_rdown)  # [T, 1536]

        if quant_override is not None and "experts_fn" in quant_override:
            z_out = quant_override["experts_fn"](z, topk_idx, topk_w)
        else:
            gu_all = self.loader.get(f"{p}.experts.gate_up_proj", self.device, torch.bfloat16)  # [512, 3072, 1536]
            dn_all = self.loader.get(f"{p}.experts.down_proj", self.device, torch.bfloat16)     # [512, 1536, 1536]
            z_out = torch.zeros((T, z.shape[1]), device=self.device, dtype=torch.float32)
            active_e = torch.unique(topk_idx).tolist()
            for e in active_e:
                tok_m, slot_m = torch.where(topk_idx == e)
                ze = z[tok_m]  # [N_e, 1536]
                we = topk_w[tok_m, slot_m].unsqueeze(-1)  # [N_e, 1]
                gu = F.linear(ze, gu_all[e])              # [N_e, 3072]
                g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                # Exact swiglu_limit=10.0 clamping (routed experts only!)
                h = torch.clamp(F.silu(g), max=self.swiglu_limit) * torch.clamp(u, min=-self.swiglu_limit, max=self.swiglu_limit)
                if capture_hook is not None:
                    capture_hook(il, e, ze, h, we.squeeze(-1))
                ye = F.linear(h.to(torch.bfloat16), dn_all[e]).to(torch.float32)
                z_out.index_add_(0, tok_m, ye * we)
            del gu_all, dn_all

        moe_out = F.linear(z_out.to(torch.bfloat16), w_rup)
        # Unclamped Shared Expert
        s_g = self.loader.get(f"{p}.shared_experts.gate_proj.weight", self.device, torch.bfloat16)
        s_u = self.loader.get(f"{p}.shared_experts.up_proj.weight", self.device, torch.bfloat16)
        s_d = self.loader.get(f"{p}.shared_experts.down_proj.weight", self.device, torch.bfloat16)
        shexp = F.linear(F.silu(F.linear(cur, s_g)) * F.linear(cur, s_u), s_d)
        out = moe_out + shexp
        return rms_norm(out, self.loader.get(f"{p}.combine_output_norm.weight", self.device, torch.bfloat16), self.rms_eps)

    @torch.no_grad()
    def forward(
        self,
        token_ids: np.ndarray | torch.Tensor,
        layer_overrides: Optional[Dict[int, Dict]] = None,
        capture_hook: Optional[Callable] = None,
        return_dfly_kv: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(token_ids, torch.Tensor):
            tok_np = token_ids.detach().cpu().numpy().astype(np.int64)
            tok_t = token_ids.to(self.device, dtype=torch.long)
        else:
            tok_np = np.asarray(token_ids, dtype=np.int64)
            tok_t = torch.from_numpy(tok_np).to(self.device, dtype=torch.long)

        T = len(tok_np)
        pos = torch.arange(T, device=self.device, dtype=torch.int64)
        emb_w = self.loader.get("model.embed_tokens.weight", self.device, torch.bfloat16)
        x0 = F.embedding(tok_t, emb_w)  # [T, 3072]
        del emb_w
        x = x0.unsqueeze(1).expand(T, self.hc_mult, self.hidden_size).contiguous()  # [T, 4, 3072]

        dfly_kv_layers = set(self.loader.cfg.get("dfly_target_kv_layers", [25, 29, 33, 37, 41]))
        saved_kv = {}

        for il in range(self.n_layers):
            if il in self.engram_layers:
                x = self._apply_engram(x, il, tok_np)

            # Attention sublayer
            res = x
            cur, post_a = self._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
            cur = rms_norm(cur, self.loader.get(f"model.layers.{il}.input_layernorm.weight", self.device, torch.bfloat16), self.rms_eps)
            attn_out, kv_pack = self._attention(cur, pos, il, save_kv=(return_dfly_kv and il in dfly_kv_layers))
            if kv_pack is not None:
                saved_kv[il] = kv_pack
            x = self._hc_post(attn_out, res, post_a)

            # FFN / LatentMoE sublayer
            res = x
            cur, post_m = self._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
            cur = rms_norm(cur, self.loader.get(f"model.layers.{il}.post_attention_layernorm.weight", self.device, torch.bfloat16), self.rms_eps)
            ovr = layer_overrides.get(il) if layer_overrides else None
            if il == 0:
                mlp_out = self._dense_mlp(cur, il, quant_override=ovr)
            else:
                mlp_out = self._latent_moe(cur, il, quant_override=ovr, capture_hook=capture_hook)
            x = self._hc_post(mlp_out, res, post_m)

        h_head = self._hc_head(x)
        h_final = rms_norm(h_head, self.loader.get("model.norm.weight", self.device, torch.bfloat16), self.rms_eps)
        lm_head = self.loader.get("lm_head.weight", self.device, torch.float32)
        logits = F.linear(h_final.to(torch.float32), lm_head)
        del lm_head
        return {"logits": logits, "h_final": h_final, "dfly_kv": saved_kv}
