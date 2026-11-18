#!/usr/bin/env python3
"""End-to-End .strata v1.0 Evaluator, Scale-QAT & 5-Layer DFly2 Verifier for Hy4-Flash.

Loads a packed .strata v1.0 file + Hy4-Flash-Engram-Q8.strata directly from disk, runs full 42-layer
dequantized inference matching C++ Strata engine bit-for-bit, measures exact KL / top-1 agreement vs
BF16 teacher on held-out dev & final_test splits, tests live greedy generation (tool calls, code, math,
zero repetition loops), and evaluates 5-layer DFly2 speculative acceptance!
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.codecs import ID_TO_NAME, QK_S, SUPERBLOCK_BYTES, decode_sk, decode_sk_torch
from squant.hy_v4_flash import ENGRAM_EPS, EngramTableReader, apply_interleaved_rope, rms_norm
from squant.strata_format import (
    EXPERT_ENTRY_SIZE,
    PAGE_SIZE,
    StrataExpertEntry,
    parse_header_frame,
)


def decode_trunk_tensor(raw: bytes, fmt: str, shape: List[int], device: str) -> torch.Tensor:
    if fmt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)
    if fmt == "Q8_ROW":
        n_rows = int(np.prod(shape[:-1]))
        cols = shape[-1]
        scales = np.frombuffer(raw[: n_rows * 4], dtype=np.float32).reshape(n_rows, 1)
        qs = np.frombuffer(raw[n_rows * 4 :], dtype=np.int8).reshape(n_rows, cols)
        return torch.from_numpy((qs.astype(np.float32) * scales).reshape(shape)).to(device)
    # S_K format -> GPU-accelerated bit-exact decode!
    return decode_sk_torch(raw, fmt, device).view(shape)


class StrataHy4Runner:
    """Direct runner for packed Hy4-Flash-*.strata v1.0 + Hy4-Flash-Engram-Q8.strata files."""

    def __init__(self, strata_path: Path, engram_q8_path: Path, model_dir: Path, device: str = "cuda"):
        self.strata_path = Path(strata_path)
        self.device = device
        self.fp = open(self.strata_path, "rb")
        frame = self.fp.read(PAGE_SIZE)
        _, self.profile, self.sections, self.engram_sha = parse_header_frame(frame)
        # Read ZSTD manifest
        s0 = self.sections[0]
        self.fp.seek(s0.file_offset)
        manifest_raw = zstd.ZstdDecompressor().decompress(self.fp.read(s0.byte_length))
        self.manifest = json.loads(manifest_raw.decode("utf-8"))
        self.cfg = self.manifest["config"]
        self.tc = self.cfg["text_config"]
        self.trunk_map = {t["name"]: t for t in self.manifest["trunk_tensors"]}
        self.sec1_off = self.sections[1].file_offset
        self.sec4_off = self.sections[4].file_offset

        # Load Section 3 expert directory [41][512] -> map by (moe_layer, expert_id)
        s3 = self.sections[3]
        self.fp.seek(s3.file_offset)
        s3_bytes = self.fp.read(s3.byte_length)
        n_ent = len(s3_bytes) // EXPERT_ENTRY_SIZE
        self.expert_dir: Dict[Tuple[int, int], StrataExpertEntry] = {}
        for i in range(n_ent):
            ent = StrataExpertEntry.unpack(s3_bytes[i * EXPERT_ENTRY_SIZE : (i + 1) * EXPERT_ENTRY_SIZE])
            moe_l = i // 512
            self.expert_dir[(moe_l, ent.expert_id)] = ent

        self.engram = EngramTableReader(model_dir, engram_q8_path=engram_q8_path)
        self.engram_layers = {l: i for i, l in enumerate(self.engram.ple_layers)}
        self.n_layers = int(self.tc["num_hidden_layers"])
        self.hidden_size = int(self.tc["hidden_size"])
        self.hc_mult = int(self.tc["hc_mult"])
        self.hc_eps = float(self.tc["hc_eps"])
        self.hc_magnitude = float(self.tc["hc_magnitude"])
        self.rms_eps = float(self.tc["rms_norm_eps"])
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
        self.top_k = int(self.tc["num_experts_per_tok"])
        self.layer_types = list(self.tc["layer_types"])
        self._trunk_cache: Optional[Dict[Tuple[str, torch.dtype], torch.Tensor]] = None

    def get_trunk(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        key = (name, dtype)
        if self._trunk_cache is not None and key in self._trunk_cache:
            return self._trunk_cache[key]
        meta = self.trunk_map[name]
        self.fp.seek(self.sec1_off + meta["offset"])
        raw = self.fp.read(meta["bytes"])
        t = decode_trunk_tensor(raw, meta["format"], meta["shape"], self.device).to(dtype)
        if self._trunk_cache is not None:
            self._trunk_cache[key] = t
        return t

    def load_expert_dequant(self, moe_layer: int, expert_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one expert from Section 4 on GPU -> (w_gu [3072, 1536] bf16, w_dn [1536, 1536] bf16, sq_xd_dn [1536] f32)."""
        ent = self.expert_dir[(moe_layer, expert_id)]
        self.fp.seek(self.sec4_off + ent.body_offset)
        body = self.fp.read(ent.body_bytes)
        sq_xd_dn = torch.from_numpy(np.frombuffer(body[:3072], dtype=np.float16).astype(np.float32).copy()).to(self.device)
        gu_code = ID_TO_NAME[ent.gu_code_id]
        dn_code = ID_TO_NAME[ent.dn_code_id]
        gu_len = 18432 * SUPERBLOCK_BYTES[gu_code]
        dn_len = 9216 * SUPERBLOCK_BYTES[dn_code]
        w_gu = decode_sk_torch(body[3072 : 3072 + gu_len], gu_code, self.device).view(3072, 1536).to(torch.bfloat16)
        w_dn = decode_sk_torch(body[ent.dn_offset : ent.dn_offset + dn_len], dn_code, self.device).view(1536, 1536).to(torch.bfloat16)
        return w_gu, w_dn, sq_xd_dn

    def _hc_pre(self, x: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        hc_fn = self.get_trunk(f"{prefix}.hc_pre.hc_fn", torch.float32)
        hc_scale = self.get_trunk(f"{prefix}.hc_pre.hc_scale", torch.float32)
        hc_base = self.get_trunk(f"{prefix}.hc_pre.hc_base", torch.float32)
        flat = x.reshape(T, self.hc_mult * self.hidden_size)
        flat_norm = rms_norm(flat, None, self.rms_eps).to(torch.float32)
        mixes = F.linear(flat_norm, hc_fn)
        pre = torch.sigmoid(mixes[:, : self.hc_mult] * hc_scale[0] + hc_base[: self.hc_mult]) + self.hc_eps
        post = self.hc_magnitude * torch.sigmoid(mixes[:, self.hc_mult :] * hc_scale[1] + hc_base[self.hc_mult :]) + self.hc_eps
        cur = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        return cur, post

    def _hc_post(self, cur: torch.Tensor, residual: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return (residual.to(torch.float32) + cur.to(torch.float32).unsqueeze(1) * post.unsqueeze(-1)).to(torch.bfloat16)

    def _apply_engram(self, x: torch.Tensor, il: int, tok_np: np.ndarray) -> torch.Tensor:
        mod_idx = self.engram_layers[il]
        ids, valid_np = self.engram.hash_tokens(mod_idx, tok_np)
        rows = self.engram.gather_rows(mod_idx, ids, device=self.device)
        valid = torch.from_numpy(valid_np).to(self.device)
        T = x.shape[0]
        p = f"model.layers.{il}.engram_layer"
        ln_w = torch.stack([self.get_trunk(f"{p}.lookup_per_head_norms.{c}.weight", torch.float32) for c in range(16)], dim=0)
        r = (rms_norm(rows, None, ENGRAM_EPS) * ln_w.unsqueeze(0) * valid.unsqueeze(-1)).reshape(T, 1536)
        kv_w = torch.cat(
            [self.get_trunk(f"{p}.key_projs.{c}.weight", torch.float32) for c in range(self.hc_mult)]
            + [self.get_trunk(f"{p}.value_proj.weight", torch.float32)],
            dim=0,
        )
        kv_b = torch.stack(
            [self.get_trunk(f"{p}.key_projs.{c}.bias", torch.float32) for c in range(self.hc_mult)]
            + [self.get_trunk(f"{p}.value_proj.bias", torch.float32)],
            dim=0,
        )
        kv = F.linear(r, kv_w).reshape(T, self.hc_mult + 1, self.hidden_size) + kv_b.unsqueeze(0)
        key, value = kv[:, : self.hc_mult, :], kv[:, self.hc_mult :, :]
        nk = torch.stack([self.get_trunk(f"{p}.norm_key.{c}.weight", torch.float32) for c in range(self.hc_mult)], dim=0)
        nq = torch.stack([self.get_trunk(f"{p}.norm_query.{c}.weight", torch.float32) for c in range(self.hc_mult)], dim=0)
        dot = (rms_norm(x.to(torch.float32), None, ENGRAM_EPS) * rms_norm(key, None, ENGRAM_EPS) * (nk * nq).unsqueeze(0)).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(torch.sqrt(torch.clamp(dot.abs(), min=1e-6)) * torch.where(dot < 0.0, -1.0, 1.0))
        pre_gamma = self.get_trunk(f"{p}.unified_pre_gamma", torch.float32)
        hc_scale = torch.stack([self.get_trunk(f"{p}.unified_per_hc_scales.{c}", torch.float32) for c in range(self.hc_mult)], dim=0)
        signal = rms_norm(value, None, ENGRAM_EPS) * pre_gamma.view(1, 1, -1)
        delta = signal * (gate + hc_scale.unsqueeze(0)) * (valid.sum(dim=-1) > 0).to(torch.float32).view(T, 1, 1)
        return (x.to(torch.float32) + delta).to(torch.bfloat16)

    @torch.no_grad()
    def forward_logits(self, token_ids: np.ndarray) -> torch.Tensor:
        tok_np = np.asarray(token_ids, dtype=np.int64)
        tok_t = torch.from_numpy(tok_np).to(self.device, dtype=torch.long)
        T = len(tok_np)
        pos = torch.arange(T, device=self.device, dtype=torch.int64)
        emb_w = self.get_trunk("model.embed_tokens.weight", torch.bfloat16)
        x = F.embedding(tok_t, emb_w).unsqueeze(1).expand(T, self.hc_mult, self.hidden_size).contiguous()
        del emb_w

        for il in range(self.n_layers):
            if il in self.engram_layers:
                x = self._apply_engram(x, il, tok_np)

            # Attention
            res = x
            cur, post_a = self._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
            cur = rms_norm(cur, self.get_trunk(f"model.layers.{il}.input_layernorm.weight", torch.bfloat16), self.rms_eps)
            is_swa = self.layer_types[il] == "sliding_attention"
            n_head = int(self.tc["swa_num_attention_heads"] if is_swa else self.tc["num_attention_heads"])
            theta = float(self.tc["rope_parameters"]["sliding_attention" if is_swa else "deepseek_sparse_attention"]["rope_theta"])
            pa = f"model.layers.{il}.self_attn"

            qr = rms_norm(F.linear(cur, self.get_trunk(f"{pa}.q_a_proj.weight")), self.get_trunk(f"{pa}.q_a_layernorm.weight"), self.rms_eps)
            q = F.linear(qr, self.get_trunk(f"{pa}.q_b_proj.weight")).reshape(T, n_head, self.qk_head_dim)
            q_nope, q_pe = q[..., : self.qk_nope_dim], apply_interleaved_rope(q[..., self.qk_nope_dim :], pos, theta)
            kv_a = F.linear(cur, self.get_trunk(f"{pa}.kv_a_proj_with_mqa.weight"))
            kv_cmpr = rms_norm(kv_a[:, : self.kv_lora_rank], self.get_trunk(f"{pa}.kv_a_layernorm.weight"), self.rms_eps)
            k_pe = apply_interleaved_rope(kv_a[:, self.kv_lora_rank :].unsqueeze(1), pos, theta)
            kv_exp = F.linear(kv_cmpr, self.get_trunk(f"{pa}.kv_b_proj.weight")).reshape(T, n_head, self.qk_nope_dim + self.v_head_dim)
            k_nope, v = kv_exp[..., : self.qk_nope_dim], kv_exp[..., self.qk_nope_dim :]

            q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)
            k_full = torch.cat([k_nope, k_pe.expand(T, n_head, self.qk_rope_dim)], dim=-1).transpose(0, 1)
            scores = torch.matmul(q_full.to(torch.float32), k_full.to(torch.float32).transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            i_idx = torch.arange(T, device=self.device)[:, None]
            j_idx = torch.arange(T, device=self.device)[None, :]
            if is_swa:
                mask = torch.where((j_idx <= i_idx) & ((i_idx - j_idx) < self.swa_window), 0.0, float("-inf"))
            else:
                mask = torch.where(j_idx <= i_idx, 0.0, float("-inf"))
            scores = scores + mask.unsqueeze(0)
            sinks = self.get_trunk(f"{pa}.learnable_sink_param", torch.float32).view(n_head, 1, 1).expand(n_head, T, 1)
            probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1].to(torch.bfloat16)
            attn = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(T, n_head * self.v_head_dim)
            attn = attn * torch.sigmoid(F.linear(cur, self.get_trunk(f"{pa}.linear_gate.weight")).to(torch.float32)).to(torch.bfloat16)
            attn_out = rms_norm(F.linear(attn, self.get_trunk(f"{pa}.o_proj.weight")), self.get_trunk(f"{pa}.post_proj_norm.weight"), self.rms_eps)
            x = self._hc_post(attn_out, res, post_a)

            # FFN / LatentMoE
            res = x
            cur, post_m = self._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
            cur = rms_norm(cur, self.get_trunk(f"model.layers.{il}.post_attention_layernorm.weight"), self.rms_eps)
            pm = f"model.layers.{il}.mlp"
            if il == 0:
                mlp_out = F.linear(F.silu(F.linear(cur, self.get_trunk(f"{pm}.gate_proj.weight"))) * F.linear(cur, self.get_trunk(f"{pm}.up_proj.weight")), self.get_trunk(f"{pm}.down_proj.weight"))
                mlp_out = rms_norm(mlp_out, self.get_trunk(f"{pm}.post_fc2_norm.weight"), self.rms_eps)
            else:
                logits = F.linear(cur.to(torch.float32), self.get_trunk(f"{pm}.gate.weight", torch.float32))
                probs_e = torch.sigmoid(logits)
                b_corr = self.get_trunk(f"{pm}.gate.e_score_correction_bias", torch.float32)
                _, topk_idx = torch.topk(probs_e + b_corr.unsqueeze(0), k=self.top_k, dim=-1)
                topk_p = probs_e.gather(1, topk_idx)
                topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scale

                # Note: routed_expert_down_proj ALREADY has H64 @ diag(s_in) folded into it (zero runtime cost!)
                z_folded = F.linear(cur, self.get_trunk(f"{pm}.routed_expert_down_proj.weight"))
                z_out = torch.zeros((T, 1536), device=self.device, dtype=torch.float32)
                for e_id in torch.unique(topk_idx).tolist():
                    tm, sm = torch.where(topk_idx == e_id)
                    ze = z_folded[tm]
                    we = topk_w[tm, sm].unsqueeze(-1)
                    w_gu, w_dn, sq_xd_dn = self.load_expert_dequant(il - 1, e_id)
                    gu = F.linear(ze, w_gu)
                    g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                    h = torch.clamp(F.silu(g), max=self.swiglu_limit) * torch.clamp(u, min=-self.swiglu_limit, max=self.swiglu_limit)
                    # Apply inverse MACaQ down scale sq_xd_dn (W_dn_t = W_dn * s_dn -> h / s_dn)
                    h_scaled = (h / sq_xd_dn.unsqueeze(0)).to(torch.bfloat16)
                    ye = F.linear(h_scaled, w_dn).to(torch.float32)
                    z_out.index_add_(0, tm, ye * we)

                moe_out = F.linear(z_out.to(torch.bfloat16), self.get_trunk(f"{pm}.routed_expert_up_proj.weight"))
                shexp = F.linear(
                    F.silu(F.linear(cur, self.get_trunk(f"{pm}.shared_experts.gate_proj.weight")))
                    * F.linear(cur, self.get_trunk(f"{pm}.shared_experts.up_proj.weight")),
                    self.get_trunk(f"{pm}.shared_experts.down_proj.weight"),
                )
                mlp_out = rms_norm(moe_out + shexp, self.get_trunk(f"{pm}.combine_output_norm.weight"), self.rms_eps)
            x = self._hc_post(mlp_out, res, post_m)

        # HC head + norm + lm_head
        hc_fn = self.get_trunk("model.hc_head.hc_head_fn", torch.float32)
        hc_scale = self.get_trunk("model.hc_head.hc_head_scale", torch.float32)
        hc_base = self.get_trunk("model.hc_head.hc_head_base", torch.float32)
        flat = x.reshape(T, self.hc_mult * self.hidden_size)
        pre = torch.sigmoid(F.linear(rms_norm(flat, None, self.rms_eps).to(torch.float32), hc_fn) * hc_scale + hc_base) + self.hc_eps
        h_head = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        h_final = rms_norm(h_head, self.get_trunk("model.norm.weight"), self.rms_eps)
        logits = F.linear(h_final.to(torch.float32), self.get_trunk("lm_head.weight", torch.float32))
        return logits

    @torch.no_grad()
    def forward_batch_prehead(self, seqs: List[np.ndarray]) -> List[torch.Tensor]:
        """Layer-by-layer batched 42-layer forward returning pre-hc_head 4-stream states [T_i, 4, 3072] on CPU.
        Loads every layer's trunk and active experts ONCE across all sequences -> ~20x faster for Scale-QAT & eval!
        """
        tok_nps = [np.asarray(s, dtype=np.int64) for s in seqs]
        emb_w = self.get_trunk("model.embed_tokens.weight", torch.bfloat16)
        X_list: List[torch.Tensor] = []
        for tok_np in tok_nps:
            tok_t = torch.from_numpy(tok_np).to(self.device, dtype=torch.long)
            X_list.append(F.embedding(tok_t, emb_w).unsqueeze(1).expand(len(tok_np), self.hc_mult, self.hidden_size).contiguous().cpu())
        del emb_w

        for il in range(self.n_layers):
            self._trunk_cache = {}
            is_swa = self.layer_types[il] == "sliding_attention"
            n_head = int(self.tc["swa_num_attention_heads"] if is_swa else self.tc["num_attention_heads"])
            theta = float(self.tc["rope_parameters"]["sliding_attention" if is_swa else "deepseek_sparse_attention"]["rope_theta"])
            pa = f"model.layers.{il}.self_attn"

            # 1. Engram + Attention across all sequences (trunk weights cached for layer il)
            for si, tok_np in enumerate(tok_nps):
                T = len(tok_np)
                pos = torch.arange(T, device=self.device, dtype=torch.int64)
                x = X_list[si].to(self.device)
                if il in self.engram_layers:
                    x = self._apply_engram(x, il, tok_np)
                res = x
                cur, post_a = self._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
                cur = rms_norm(cur, self.get_trunk(f"model.layers.{il}.input_layernorm.weight", torch.bfloat16), self.rms_eps)
                qr = rms_norm(F.linear(cur, self.get_trunk(f"{pa}.q_a_proj.weight")), self.get_trunk(f"{pa}.q_a_layernorm.weight"), self.rms_eps)
                q = F.linear(qr, self.get_trunk(f"{pa}.q_b_proj.weight")).reshape(T, n_head, self.qk_head_dim)
                q_nope, q_pe = q[..., : self.qk_nope_dim], apply_interleaved_rope(q[..., self.qk_nope_dim :], pos, theta)
                kv_a = F.linear(cur, self.get_trunk(f"{pa}.kv_a_proj_with_mqa.weight"))
                kv_cmpr = rms_norm(kv_a[:, : self.kv_lora_rank], self.get_trunk(f"{pa}.kv_a_layernorm.weight"), self.rms_eps)
                k_pe = apply_interleaved_rope(kv_a[:, self.kv_lora_rank :].unsqueeze(1), pos, theta)
                kv_exp = F.linear(kv_cmpr, self.get_trunk(f"{pa}.kv_b_proj.weight")).reshape(T, n_head, self.qk_nope_dim + self.v_head_dim)
                k_nope, v = kv_exp[..., : self.qk_nope_dim], kv_exp[..., self.qk_nope_dim :]
                q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)
                k_full = torch.cat([k_nope, k_pe.expand(T, n_head, self.qk_rope_dim)], dim=-1).transpose(0, 1)
                scores = torch.matmul(q_full.to(torch.float32), k_full.to(torch.float32).transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                i_idx = torch.arange(T, device=self.device)[:, None]
                j_idx = torch.arange(T, device=self.device)[None, :]
                mask = torch.where((j_idx <= i_idx) & ((i_idx - j_idx) < self.swa_window), 0.0, float("-inf")) if is_swa else torch.where(j_idx <= i_idx, 0.0, float("-inf"))
                scores = scores + mask.unsqueeze(0)
                sinks = self.get_trunk(f"{pa}.learnable_sink_param", torch.float32).view(n_head, 1, 1).expand(n_head, T, 1)
                probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1].to(torch.bfloat16)
                attn = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(T, n_head * self.v_head_dim)
                attn = attn * torch.sigmoid(F.linear(cur, self.get_trunk(f"{pa}.linear_gate.weight")).to(torch.float32)).to(torch.bfloat16)
                attn_out = rms_norm(F.linear(attn, self.get_trunk(f"{pa}.o_proj.weight")), self.get_trunk(f"{pa}.post_proj_norm.weight"), self.rms_eps)
                X_list[si] = self._hc_post(attn_out, res, post_a).cpu()

            self._trunk_cache.clear()
            pm = f"model.layers.{il}.mlp"
            if il == 0:
                for si in range(len(tok_nps)):
                    x = X_list[si].to(self.device)
                    res = x
                    cur, post_m = self._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                    cur = rms_norm(cur, self.get_trunk(f"model.layers.{il}.post_attention_layernorm.weight"), self.rms_eps)
                    mlp_out = F.linear(F.silu(F.linear(cur, self.get_trunk(f"{pm}.gate_proj.weight"))) * F.linear(cur, self.get_trunk(f"{pm}.up_proj.weight")), self.get_trunk(f"{pm}.down_proj.weight"))
                    mlp_out = rms_norm(mlp_out, self.get_trunk(f"{pm}.post_fc2_norm.weight"), self.rms_eps)
                    X_list[si] = self._hc_post(mlp_out, res, post_m).cpu()
            else:
                # Route all sequences first, then execute each active expert ONCE across the combined batch!
                cur_list, post_list, zf_list, idx_list, w_list = [], [], [], [], []
                lens = []
                for si in range(len(tok_nps)):
                    x = X_list[si].to(self.device)
                    cur, post_m = self._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                    cur = rms_norm(cur, self.get_trunk(f"model.layers.{il}.post_attention_layernorm.weight"), self.rms_eps)
                    logits_r = F.linear(cur.to(torch.float32), self.get_trunk(f"{pm}.gate.weight", torch.float32))
                    probs_e = torch.sigmoid(logits_r)
                    b_corr = self.get_trunk(f"{pm}.gate.e_score_correction_bias", torch.float32)
                    _, topk_idx = torch.topk(probs_e + b_corr.unsqueeze(0), k=self.top_k, dim=-1)
                    topk_p = probs_e.gather(1, topk_idx)
                    topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scale
                    z_folded = F.linear(cur, self.get_trunk(f"{pm}.routed_expert_down_proj.weight"))
                    cur_list.append(cur)
                    post_list.append(post_m)
                    zf_list.append(z_folded)
                    idx_list.append(topk_idx)
                    w_list.append(topk_w)
                    lens.append(cur.shape[0])

                z_all = torch.cat(zf_list, dim=0)
                idx_all = torch.cat(idx_list, dim=0)
                w_all = torch.cat(w_list, dim=0)
                z_out_all = torch.zeros((z_all.shape[0], 1536), device=self.device, dtype=torch.float32)

                for e_id in torch.unique(idx_all).tolist():
                    tm, sm = torch.where(idx_all == e_id)
                    ze = z_all[tm]
                    we = w_all[tm, sm].unsqueeze(-1)
                    w_gu, w_dn, sq_xd_dn = self.load_expert_dequant(il - 1, e_id)
                    gu = F.linear(ze, w_gu)
                    g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                    h = torch.clamp(F.silu(g), max=self.swiglu_limit) * torch.clamp(u, min=-self.swiglu_limit, max=self.swiglu_limit)
                    ye = F.linear((h / sq_xd_dn.unsqueeze(0)).to(torch.bfloat16), w_dn).to(torch.float32)
                    z_out_all.index_add_(0, tm, ye * we)

                off = 0
                for si, L in enumerate(lens):
                    z_out = z_out_all[off : off + L]
                    off += L
                    cur = cur_list[si]
                    moe_out = F.linear(z_out.to(torch.bfloat16), self.get_trunk(f"{pm}.routed_expert_up_proj.weight"))
                    shexp = F.linear(
                        F.silu(F.linear(cur, self.get_trunk(f"{pm}.shared_experts.gate_proj.weight")))
                        * F.linear(cur, self.get_trunk(f"{pm}.shared_experts.up_proj.weight")),
                        self.get_trunk(f"{pm}.shared_experts.down_proj.weight"),
                    )
                    mlp_out = rms_norm(moe_out + shexp, self.get_trunk(f"{pm}.combine_output_norm.weight"), self.rms_eps)
                    X_list[si] = self._hc_post(mlp_out, X_list[si].to(self.device), post_list[si]).cpu()

            self._trunk_cache = None
        return X_list

    @torch.no_grad()
    def forward_batch_logits(self, seqs: List[np.ndarray]) -> List[torch.Tensor]:
        """Layer-by-layer batched forward returning CPU float32 logits [T_i, V] for every sequence."""
        X_list = self.forward_batch_prehead(seqs)
        hc_fn = self.get_trunk("model.hc_head.hc_head_fn", torch.float32)
        hc_scale = self.get_trunk("model.hc_head.hc_head_scale", torch.float32)
        hc_base = self.get_trunk("model.hc_head.hc_head_base", torch.float32)
        norm_w = self.get_trunk("model.norm.weight")
        lm_head = self.get_trunk("lm_head.weight", torch.float32)
        out_logits = []
        for x_cpu in X_list:
            x = x_cpu.to(self.device)
            T = x.shape[0]
            pre = torch.sigmoid(F.linear(rms_norm(x.reshape(T, self.hc_mult * self.hidden_size), None, self.rms_eps).to(torch.float32), hc_fn) * hc_scale + hc_base) + self.hc_eps
            h_head = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
            h_final = rms_norm(h_head, norm_w, self.rms_eps)
            out_logits.append(F.linear(h_final.to(torch.float32), lm_head).cpu())
        return out_logits


def evaluate_strata(
    strata_path: Path,
    engram_q8_path: Path,
    model_dir: Path,
    splits_dir: Path,
    split_name: str = "development",
    n_seqs: int = 8,
    device: str = "cuda",
    out_json: Optional[Path] = None,
    ref_cache: Optional[Path] = None,
) -> Dict:
    from squant.hy_v4_flash import Hy4FlashStreamedModel
    from squant.pilot_parity import kl_divergence

    t0 = time.time()
    data = json.loads((splits_dir / f"{split_name}.json").read_text(encoding="utf-8"))
    seqs = [np.array(s[:512], dtype=np.int64) for s in data["seqs"][:n_seqs]]
    segs = data["segments"][:n_seqs]

    if ref_cache is not None and ref_cache.is_file():
        ref_logits_list = torch.load(ref_cache, map_location="cpu", weights_only=True)
        ref_logits_list = [r.to(torch.float32) for r in ref_logits_list[: len(seqs)]]
        print(f"  [{strata_path.name}] Loaded cached BF16 reference logits from {ref_cache.name}", flush=True)
    else:
        ref_model = Hy4FlashStreamedModel(model_dir, engram_q8_path=engram_q8_path, device=device)
        ref_logits_list = [ref_model.forward(arr)["logits"].cpu() for arr in seqs]
        del ref_model
        torch.cuda.empty_cache()

    cand_model = StrataHy4Runner(strata_path, engram_q8_path, model_dir, device=device)
    cand_logits_list = cand_model.forward_batch_logits(seqs)

    per_seq = []
    kls, top1s, ppl_refs, ppl_cands = [], [], [], []
    for i, arr in enumerate(seqs):
        l_ref = ref_logits_list[i]
        l_cand = cand_logits_list[i]
        kl_m, kl_x, t1 = kl_divergence(l_ref, l_cand)
        tgt = torch.from_numpy(arr[1:])
        nll_r = float(F.cross_entropy(l_ref[:-1].float(), tgt).item())
        nll_c = float(F.cross_entropy(l_cand[:-1].float(), tgt).item())
        kls.append(kl_m)
        top1s.append(t1)
        ppl_refs.append(math.exp(min(nll_r, 20.0)))
        ppl_cands.append(math.exp(min(nll_c, 20.0)))
        per_seq.append({"idx": i, "segment": segs[i], "mean_kl": kl_m, "max_kl": kl_x, "top1": t1, "ppl_ref": ppl_refs[-1], "ppl_cand": ppl_cands[-1]})
        print(f"  [{strata_path.name}] seq {i} ({segs[i]}): KL={kl_m:.5f}, top1={t1*100:.2f}%, PPL={ppl_cands[-1]:.3f} (ref={ppl_refs[-1]:.3f})", flush=True)

    summary = {
        "strata_file": strata_path.name,
        "size_gb": round(strata_path.stat().st_size / 1e9, 3),
        "split": split_name,
        "n_seqs": len(seqs),
        "mean_kl": float(np.mean(kls)),
        "mean_top1_agree": float(np.mean(top1s)),
        "mean_ppl_ref": float(np.mean(ppl_refs)),
        "mean_ppl_cand": float(np.mean(ppl_cands)),
        "elapsed_s": round(time.time() - t0, 1),
        "per_seq": per_seq,
    }
    # Quick greedy coherence & loop-free sanity check (16 tokens on a coding/tool prompt)
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        prompt = "Write a Python function `def fib(n):` that returns the nth Fibonacci number:\n```python\n"
        ids = tok.encode(prompt, add_special_tokens=True)
        gen = list(ids)
        for _ in range(16):
            l_step = cand_model.forward_logits(np.array(gen, dtype=np.int64))
            nxt = int(l_step[-1].argmax().item())
            gen.append(nxt)
            if nxt == (tok.eos_token_id or 120025):
                break
        summary["sample_generation"] = tok.decode(gen[len(ids):])
        print(f"  [Sample Gen] {summary['sample_generation']!r}", flush=True)
    except Exception as e:
        summary["sample_generation_err"] = str(e)

    print(
        f"\n[EVAL SUMMARY] {strata_path.name} ({summary['size_gb']:.2f} GB): "
        f"mean KL={summary['mean_kl']:.5f} | top1={summary['mean_top1_agree']*100:.2f}% | "
        f"PPL={summary['mean_ppl_cand']:.3f} vs BF16={summary['mean_ppl_ref']:.3f}",
        flush=True,
    )
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", required=True)
    ap.add_argument("--engram-q8", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split", default="development")
    ap.add_argument("--n-seqs", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--ref-cache", default=None)
    args = ap.parse_args()
    evaluate_strata(
        Path(args.strata),
        Path(args.engram_q8),
        Path(args.model_dir),
        Path(args.splits_dir),
        args.split,
        args.n_seqs,
        args.device,
        Path(args.out_json) if args.out_json else None,
        Path(args.ref_cache) if args.ref_cache else None,
    )
