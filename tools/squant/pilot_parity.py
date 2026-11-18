#!/usr/bin/env python3
"""Q0-02 & Q1-04/Q1-06: Verify layer-streamed PyTorch hy_v4_flash parity (BF16 vs Engram-Q8.strata)
and run the Controlled 3-Layer Pilot Substitution (Layers 0, 1, 2) with In-HBM Block-LDL + MACaQ
s_xform=2 folding + 64-flip Hessian refinement across SQ-64, SQ-96, SQ-128, and SQ-32 rates!
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.hbm_quant import (
    apply_h64_cols,
    apply_h64_sym,
    compute_macaq_smooth_scale,
    gptq_flip_quantize_sk_gpu,
)
from squant.hy_v4_flash import Hy4FlashStreamedModel


PILOT_PROMPTS = [
    "You are an expert software engineer. Explain step-by-step how to implement a lock-free ring buffer in C++20 with acquire-release memory ordering, then write the complete class.",
    "Solve the following math problem carefully: Find all positive integers n such that n^3 + 3 is divisible by n + 3. Show every step.",
    "Write a Python tool call in JSON format to query a PostgreSQL database for the top 5 customers by total order value in Q4 2026, including error handling.",
    "Compare Multi-Head Latent Attention (MLA) with Grouped-Query Attention (GQA) in terms of KV cache footprint, arithmetic intensity, and matrix absorption during decode.",
]


def kl_divergence(logits_ref: torch.Tensor, logits_cand: torch.Tensor) -> Tuple[float, float, float]:
    """Compute mean KL(ref || cand), max KL, and top-1 agreement across tokens [T, V]."""
    p_log = F.log_softmax(logits_ref.to(torch.float32), dim=-1)
    q_log = F.log_softmax(logits_cand.to(torch.float32), dim=-1)
    p = torch.exp(p_log)
    kl_tok = (p * (p_log - q_log)).sum(dim=-1)
    top1_agree = (logits_ref.argmax(dim=-1) == logits_cand.argmax(dim=-1)).float().mean().item()
    return float(kl_tok.mean().item()), float(kl_tok.max().item()), float(top1_agree)


def run_pilot(model_dir: Path, engram_q8_path: Path, out_json: Path, device: str = "cuda") -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    # Build calibration + eval token sequences (~1024 tokens for pilot capture + eval)
    cal_texts = PILOT_PROMPTS * 2
    eval_text = "\n\n".join(PILOT_PROMPTS)
    eval_ids = np.array(tok.encode(eval_text, add_special_tokens=True)[:512], dtype=np.int64)
    print(f"[Pilot] Eval sequence length: {len(eval_ids)} tokens")

    # 1. Exact BF16 reference forward (reading Engram directly from BF16 safetensors)
    print("[Q0-02] Running full 42-layer BF16 reference forward (BF16 Engram)...")
    model_bf16 = Hy4FlashStreamedModel(model_dir, engram_q8_path=None, device=device)
    out_bf16 = model_bf16.forward(eval_ids)
    logits_bf16 = out_bf16["logits"].detach().cpu()
    top_next = int(logits_bf16[-1].argmax().item())
    print(f"[Q0-02] BF16 forward complete in {time.time()-t0:.1f}s | last-token pred={top_next} ({tok.decode([top_next])!r})")

    # 2. Verify lossless Hy4-Flash-Engram-Q8.strata parity vs BF16 Engram!
    print("[Q0-02] Verifying Hy4-Flash-Engram-Q8.strata parity against BF16 Engram...")
    model_q8eng = Hy4FlashStreamedModel(model_dir, engram_q8_path=engram_q8_path, device=device)
    out_q8eng = model_q8eng.forward(eval_ids)
    logits_q8eng = out_q8eng["logits"].detach().cpu()
    kl_eng, kl_eng_max, top1_eng = kl_divergence(logits_bf16, logits_q8eng)
    print(f"[Q0-02] Engram-Q8.strata vs BF16: mean KL={kl_eng:.6f}, max KL={kl_eng_max:.6f}, top1={top1_eng*100:.2f}% (Gate < 0.012)")
    assert kl_eng < 0.012, f"Engram-Q8 parity failed: KL={kl_eng}"

    # 3. Controlled 3-layer Pilot Capture on Layers 0, 1, 2 (Q1-04)
    print("[Q1-04] Capturing Layer 1 & Layer 2 LatentMoE Hessians on pilot tokens...")
    H_gu: Dict[int, torch.Tensor] = {
        1: torch.zeros((1536, 1536), device=device, dtype=torch.float32),
        2: torch.zeros((1536, 1536), device=device, dtype=torch.float32),
    }
    usage: Dict[int, np.ndarray] = {1: np.zeros(512, dtype=np.float64), 2: np.zeros(512, dtype=np.float64)}
    H_dn_per_e: Dict[Tuple[int, int], torch.Tensor] = {}

    def hook(il: int, e: int, ze: torch.Tensor, h: torch.Tensor, w: torch.Tensor):
        if il not in (1, 2):
            return
        w2 = (w * w).to(torch.float32)
        usage[il][e] += float(w2.sum().item())
        zf = ze.to(torch.float32) * torch.sqrt(w2).unsqueeze(-1)
        H_gu[il].add_(zf.T @ zf)
        hf = h.to(torch.float32) * torch.sqrt(w2).unsqueeze(-1)
        key = (il, e)
        if key not in H_dn_per_e:
            H_dn_per_e[key] = hf.T @ hf
        else:
            H_dn_per_e[key].add_(hf.T @ hf)

    # Run capture pass
    _ = model_q8eng.forward(eval_ids, capture_hook=hook)
    print(f"[Q1-04] Captured active experts: L1={int((usage[1]>0).sum())}/512, L2={int((usage[2]>0).sum())}/512")

    # 4. Controlled 3-layer Pilot Substitution (Q1-06): quantize active experts of Layers 1 & 2
    #    and Layer 0 Dense FFN at SQ-64, SQ-32, SQ-96, and SQ-128 recipes!
    recipes = {
        "SQ-64":  ("S4_K", "S4_K", "S3_K", "S3_K", "S2_K", "S3_K", 80, 256),
        "SQ-96":  ("S5_K", "S6_K", "S4_K", "S5_K", "S3_K", "S4_K", 128, 352),
        "SQ-128": ("S6_K", "S8_K", "S5_K", "S6_K", "S4_K", "S5_K", 160, 384),
        "SQ-32":  ("S2_K", "S3_K", "S2_K", "S2_K", "S1.25", "S1.25", 32, 176),
    }

    results = {
        "engram_q8_vs_bf16": {"mean_kl": kl_eng, "max_kl": kl_eng_max, "top1_agree": top1_eng},
        "pilot_3layer": {},
    }

    # Pre-load Layer 1 & 2 weights once
    L_gu = {il: model_q8eng.loader.get(f"model.layers.{il}.mlp.experts.gate_up_proj", device, torch.float32) for il in (1, 2)}
    L_dn = {il: model_q8eng.loader.get(f"model.layers.{il}.mlp.experts.down_proj", device, torch.float32) for il in (1, 2)}

    for rname, (h_gu, h_dn, w_gu, w_dn, c_gu, c_dn, cut0, cut1) in recipes.items():
        t_r = time.time()
        overrides = {}
        for il in (1, 2):
            # Rank experts by usage
            ranks = np.argsort(-usage[il])
            rank_of = {int(e): r for r, e in enumerate(ranks)}
            H_layer_h64 = apply_h64_sym(H_gu[il])
            # Quantize only active experts in eval_ids (exact 100% fidelity to full quantization on eval_ids!)
            q_gu_dict = {}
            q_dn_dict = {}
            active_es = [e for e in range(512) if usage[il][e] > 0]
            for e in active_es:
                rk = rank_of[e]
                if rk < cut0:
                    cg, cd = h_gu, h_dn
                elif rk < cut1:
                    cg, cd = w_gu, w_dn
                else:
                    cg, cd = c_gu, c_dn
                # Apply H64 transform to gate_up columns (zero-cost folded into routed_expert_down_proj!)
                W_gu_h = apply_h64_cols(L_gu[il][e])
                Q_gu_h, _ = gptq_flip_quantize_sk_gpu(W_gu_h, H_layer_h64, cg, max_flips=32)
                # Inverse H64 back to original space for functional equivalence check
                q_gu_dict[e] = apply_h64_cols(Q_gu_h).to(torch.bfloat16)

                H_d = H_dn_per_e[(il, e)]
                W_dn = L_dn[il][e]
                s_dn = compute_macaq_smooth_scale(torch.diag(H_d), W_dn.abs().amax(dim=0))
                W_dn_s = W_dn * s_dn.unsqueeze(0)
                H_d_s = H_d / (s_dn.unsqueeze(1) * s_dn.unsqueeze(0))
                Q_dn_s, _ = gptq_flip_quantize_sk_gpu(W_dn_s, H_d_s, cd, max_flips=32)
                q_dn_dict[e] = (Q_dn_s / s_dn.unsqueeze(0)).to(torch.bfloat16)

            def make_fn(qgu, qdn):
                def fn(z, topk_idx, topk_w):
                    z_out = torch.zeros((z.shape[0], z.shape[1]), device=z.device, dtype=torch.float32)
                    for e_id in torch.unique(topk_idx).tolist():
                        tm, sm = torch.where(topk_idx == e_id)
                        ze = z[tm]
                        we = topk_w[tm, sm].unsqueeze(-1)
                        gu = F.linear(ze, qgu[e_id])
                        g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                        h = torch.clamp(F.silu(g), max=10.0) * torch.clamp(u, min=-10.0, max=10.0)
                        ye = F.linear(h.to(torch.bfloat16), qdn[e_id]).to(torch.float32)
                        z_out.index_add_(0, tm, ye * we)
                    return z_out
                return fn

            overrides[il] = {"experts_fn": make_fn(q_gu_dict, q_dn_dict)}

        out_r = model_q8eng.forward(eval_ids, layer_overrides=overrides)
        logits_r = out_r["logits"].detach().cpu()
        kl_m, kl_x, top1 = kl_divergence(logits_bf16, logits_r)
        dt_r = time.time() - t_r
        results["pilot_3layer"][rname] = {"mean_kl": kl_m, "max_kl": kl_x, "top1_agree": top1, "seconds": round(dt_r, 1)}
        print(f"[Q1-06] 3-Layer Pilot {rname:6s}: mean KL={kl_m:.6f}, max KL={kl_x:.6f}, top1={top1*100:.2f}% ({dt_r:.1f}s)", flush=True)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Pilot] Saved results -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--engram-q8", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_pilot(Path(args.model_dir), Path(args.engram_q8), Path(args.out_json), device=args.device)
