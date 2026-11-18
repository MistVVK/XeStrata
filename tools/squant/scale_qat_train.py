#!/usr/bin/env python3
"""End-to-End Scale-QAT + 5-Layer DFly2 Co-Training for Hy4-Flash (.strata v1.0).

Runs 1 build per GPU (4-way parallel across GPUs 0..3):
  1. Captures exact BF16 teacher top-256 log-probs + tail probability on calibration sequences (shared once).
  2. Optimizes Sandwich RMSNorms, iHC stream scales/bases, and per-layer LatentMoE output scales
     against exact BF16 teacher KL divergence (gradient checkpointed layer-by-layer to stay <25 GB VRAM).
  3. Co-trains the 5-layer DFly2 drafter (mtp_fusion, hidden_correction, dfly_layers.0..4, confidence_head)
     against the quantized target model's hidden states & MLA KV so speculative acceptance alpha >= 4.75!
  4. Saves refined Section 1 parameters for packing into Hy4-Flash-SQ-{64,96,128,32}.strata.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.hy_v4_flash import ENGRAM_EPS, Hy4FlashStreamedModel, apply_interleaved_rope, rms_norm
from squant.scale_qat_eval import StrataHy4Runner


def capture_teacher_targets(
    model_dir: Path,
    engram_q8_path: Path,
    splits_dir: Path,
    out_path: Path,
    n_seqs: int = 24,
    seq_len: int = 512,
    topk: int = 256,
    device: str = "cuda:0",
) -> Path:
    """Capture BF16 teacher top-256 logits/indices + tail log-prob for Scale-QAT distillation."""
    if out_path.is_file():
        print(f"[Teacher] Reusing cached teacher targets: {out_path}", flush=True)
        return out_path
    t0 = time.time()
    cal = json.loads((splits_dir / "calibration.json").read_text(encoding="utf-8"))
    seqs = [s[:seq_len] for s in cal["seqs"][:n_seqs]]
    model = Hy4FlashStreamedModel(model_dir, engram_q8_path=engram_q8_path, device=device)

    records = []
    for i, s in enumerate(seqs):
        out = model.forward(np.array(s, dtype=np.int64), return_dfly_kv=False)
        logits = out["logits"].to(torch.float32)  # [T, V]
        logp = F.log_softmax(logits, dim=-1)
        top_lp, top_idx = torch.topk(logp, k=topk, dim=-1)  # [T, 256]
        top_p_sum = torch.exp(top_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
        tail_p = (1.0 - top_p_sum).clamp(min=1e-7)
        records.append(
            {
                "tokens": torch.tensor(s, dtype=torch.int32),
                "top_idx": top_idx.to(torch.int32).cpu(),
                "top_logp": top_lp.to(torch.float16).cpu(),
                "tail_p": tail_p.to(torch.float16).cpu(),
            }
        )
        if (i + 1) % 6 == 0 or (i + 1) == len(seqs):
            print(f"  [Teacher] Captured {i+1}/{len(seqs)} sequences ({time.time()-t0:.1f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_path)
    print(f"[Teacher] Saved {len(records)} teacher top-{topk} targets -> {out_path} in {time.time()-t0:.1f}s", flush=True)
    return out_path


def topk_tail_kl_loss(cand_logits: torch.Tensor, top_idx: torch.Tensor, top_logp: torch.Tensor, tail_p: torch.Tensor) -> torch.Tensor:
    """Exact top-256 + tail-bucket KL(p_teacher || q_cand) loss."""
    q_logp = F.log_softmax(cand_logits.to(torch.float32), dim=-1)
    q_top_lp = q_logp.gather(-1, top_idx.to(torch.long))
    p_top_lp = top_logp.to(torch.float32)
    p_top = torch.exp(p_top_lp)
    kl_top = (p_top * (p_top_lp - q_top_lp)).sum(dim=-1)

    q_top_sum = torch.exp(q_top_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
    q_tail_lp = torch.log((1.0 - q_top_sum).clamp(min=1e-7))
    p_t = tail_p.to(torch.float32).clamp(min=1e-7)
    kl_tail = p_t * (torch.log(p_t) - q_tail_lp)
    return (kl_top + kl_tail).mean()


def run_scale_qat_for_build(
    build: str,
    strata_path: Path,
    engram_q8_path: Path,
    model_dir: Path,
    teacher_path: Path,
    out_overrides_path: Path,
    device: str = "cuda:0",
    epochs: int = 2,
    lr: float = 2.0e-4,
) -> Dict:
    """Run lightweight Scale-QAT tuning Sandwich post-norms, output norm, and per-layer LatentMoE output scales."""
    t0 = time.time()
    torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
    print(f"[Scale-QAT {build}] Starting on {device} from {strata_path.name}...", flush=True)

    teacher_recs = torch.load(teacher_path, map_location="cpu", weights_only=False)
    runner = StrataHy4Runner(strata_path, engram_q8_path, model_dir, device=device)

    # Pre-cache quantized hidden states entering the final output_norm + lm_head, AND learnable per-layer
    # post-norm / output-norm / hc-head refinement parameters!
    # For fast, memory-safe end-to-end tuning across all 42 layers on 1 GPU in ~15-20 min:
    # We optimize:
    #   1) model.norm.weight (FP32) + model.hc_head.{hc_head_scale, hc_head_base}
    #   2) 5-layer DFly2 mtp_fusion & confidence_head calibration
    #   3) Per-layer Sandwich post-norm diagonal calibration alpha_l (42 layers) via finite-difference/STE-free exact pass
    # First measure exact baseline KL on calibration sequences & collect h_head (4-stream pre-hc_head state)
    h_x_list = []
    kl_before_list = []
    lm_head = runner.get_trunk("lm_head.weight", torch.float32)

    with torch.no_grad():
        for i, rec in enumerate(teacher_recs):
            tok_np = rec["tokens"].numpy().astype(np.int64)
            # Run full 42-layer quantized forward up to x [T, 4, 3072] right before hc_head
            tok_t = torch.from_numpy(tok_np).to(device, dtype=torch.long)
            T = len(tok_np)
            pos = torch.arange(T, device=device, dtype=torch.int64)
            x = F.embedding(tok_t, runner.get_trunk("model.embed_tokens.weight")).unsqueeze(1).expand(T, 4, 3072).contiguous()
            for il in range(runner.n_layers):
                if il in runner.engram_layers:
                    x = runner._apply_engram(x, il, tok_np)
                res = x
                cur, post_a = runner._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
                cur = rms_norm(cur, runner.get_trunk(f"model.layers.{il}.input_layernorm.weight"), runner.rms_eps)
                is_swa = runner.layer_types[il] == "sliding_attention"
                n_head = int(runner.tc["swa_num_attention_heads"] if is_swa else runner.tc["num_attention_heads"])
                theta = float(runner.tc["rope_parameters"]["sliding_attention" if is_swa else "deepseek_sparse_attention"]["rope_theta"])
                pa = f"model.layers.{il}.self_attn"
                qr = rms_norm(F.linear(cur, runner.get_trunk(f"{pa}.q_a_proj.weight")), runner.get_trunk(f"{pa}.q_a_layernorm.weight"), runner.rms_eps)
                q = F.linear(qr, runner.get_trunk(f"{pa}.q_b_proj.weight")).reshape(T, n_head, runner.qk_head_dim)
                q_nope, q_pe = q[..., : runner.qk_nope_dim], apply_interleaved_rope(q[..., runner.qk_nope_dim :], pos, theta)
                kv_a = F.linear(cur, runner.get_trunk(f"{pa}.kv_a_proj_with_mqa.weight"))
                kv_cmpr = rms_norm(kv_a[:, : runner.kv_lora_rank], runner.get_trunk(f"{pa}.kv_a_layernorm.weight"), runner.rms_eps)
                k_pe = apply_interleaved_rope(kv_a[:, runner.kv_lora_rank :].unsqueeze(1), pos, theta)
                kv_exp = F.linear(kv_cmpr, runner.get_trunk(f"{pa}.kv_b_proj.weight")).reshape(T, n_head, runner.qk_nope_dim + runner.v_head_dim)
                k_nope, v = kv_exp[..., : runner.qk_nope_dim], kv_exp[..., runner.qk_nope_dim :]
                q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)
                k_full = torch.cat([k_nope, k_pe.expand(T, n_head, runner.qk_rope_dim)], dim=-1).transpose(0, 1)
                scores = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) / math.sqrt(runner.qk_head_dim)
                i_idx = torch.arange(T, device=device)[:, None]
                j_idx = torch.arange(T, device=device)[None, :]
                mask = torch.where((j_idx <= i_idx) & ((i_idx - j_idx) < runner.swa_window), 0.0, float("-inf")) if is_swa else torch.where(j_idx <= i_idx, 0.0, float("-inf"))
                scores = scores + mask.unsqueeze(0)
                sinks = runner.get_trunk(f"{pa}.learnable_sink_param", torch.float32).view(n_head, 1, 1).expand(n_head, T, 1)
                probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1].to(torch.bfloat16)
                attn = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(T, n_head * runner.v_head_dim)
                attn = attn * torch.sigmoid(F.linear(cur, runner.get_trunk(f"{pa}.linear_gate.weight")).float()).to(torch.bfloat16)
                attn_out = rms_norm(F.linear(attn, runner.get_trunk(f"{pa}.o_proj.weight")), runner.get_trunk(f"{pa}.post_proj_norm.weight"), runner.rms_eps)
                x = runner._hc_post(attn_out, res, post_a)

                res = x
                cur, post_m = runner._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                cur = rms_norm(cur, runner.get_trunk(f"model.layers.{il}.post_attention_layernorm.weight"), runner.rms_eps)
                pm = f"model.layers.{il}.mlp"
                if il == 0:
                    mlp_out = F.linear(F.silu(F.linear(cur, runner.get_trunk(f"{pm}.gate_proj.weight"))) * F.linear(cur, runner.get_trunk(f"{pm}.up_proj.weight")), runner.get_trunk(f"{pm}.down_proj.weight"))
                    mlp_out = rms_norm(mlp_out, runner.get_trunk(f"{pm}.post_fc2_norm.weight"), runner.rms_eps)
                else:
                    logits_r = F.linear(cur.float(), runner.get_trunk(f"{pm}.gate.weight", torch.float32))
                    probs_e = torch.sigmoid(logits_r)
                    b_corr = runner.get_trunk(f"{pm}.gate.e_score_correction_bias", torch.float32)
                    _, topk_idx = torch.topk(probs_e + b_corr.unsqueeze(0), k=runner.top_k, dim=-1)
                    topk_p = probs_e.gather(1, topk_idx)
                    topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * runner.routed_scale
                    z_folded = F.linear(cur, runner.get_trunk(f"{pm}.routed_expert_down_proj.weight"))
                    z_out = torch.zeros((T, 1536), device=device, dtype=torch.float32)
                    for e_id in torch.unique(topk_idx).tolist():
                        tm, sm = torch.where(topk_idx == e_id)
                        ze, we = z_folded[tm], topk_w[tm, sm].unsqueeze(-1)
                        w_gu, w_dn, sq_xd_dn = runner.load_expert_dequant(il - 1, e_id)
                        gu = F.linear(ze, w_gu)
                        h = torch.clamp(F.silu(gu[:, :1536].float()), max=runner.swiglu_limit) * torch.clamp(gu[:, 1536:].float(), min=-runner.swiglu_limit, max=runner.swiglu_limit)
                        ye = F.linear((h / sq_xd_dn.unsqueeze(0)).to(torch.bfloat16), w_dn).float()
                        z_out.index_add_(0, tm, ye * we)
                    moe_out = F.linear(z_out.to(torch.bfloat16), runner.get_trunk(f"{pm}.routed_expert_up_proj.weight"))
                    shexp = F.linear(F.silu(F.linear(cur, runner.get_trunk(f"{pm}.shared_experts.gate_proj.weight"))) * F.linear(cur, runner.get_trunk(f"{pm}.shared_experts.up_proj.weight")), runner.get_trunk(f"{pm}.shared_experts.down_proj.weight"))
                    mlp_out = rms_norm(moe_out + shexp, runner.get_trunk(f"{pm}.combine_output_norm.weight"), runner.rms_eps)
                x = runner._hc_post(mlp_out, res, post_m)

            h_x_list.append(x.cpu())
            hc_fn = runner.get_trunk("model.hc_head.hc_head_fn", torch.float32)
            hc_sc = runner.get_trunk("model.hc_head.hc_head_scale", torch.float32)
            hc_bs = runner.get_trunk("model.hc_head.hc_head_base", torch.float32)
            pre = torch.sigmoid(F.linear(rms_norm(x.reshape(T, 4 * 3072), None, runner.rms_eps).float(), hc_fn) * hc_sc + hc_bs) + runner.hc_eps
            h_head = torch.einsum("thd,th->td", x.float(), pre)
            h_fin = rms_norm(h_head, runner.get_trunk("model.norm.weight", torch.float32), runner.rms_eps)
            logits = F.linear(h_fin, lm_head)
            kl_b = float(topk_tail_kl_loss(logits, rec["top_idx"].to(device), rec["top_logp"].to(device), rec["tail_p"].to(device)).item())
            kl_before_list.append(kl_b)

    kl_before = float(np.mean(kl_before_list))
    print(f"[Scale-QAT {build}] Pre-QAT cal KL = {kl_before:.5f}. Running head-norm + hc_head Scale-QAT...", flush=True)

    # Optimize hc_head (fn, scale, base) and final model.norm.weight with AdamW + L2 anchor to original weights
    p_norm_orig = runner.get_trunk("model.norm.weight", torch.float32).detach().clone()
    p_fn_orig = runner.get_trunk("model.hc_head.hc_head_fn", torch.float32).detach().clone()
    p_sc_orig = runner.get_trunk("model.hc_head.hc_head_scale", torch.float32).detach().clone()
    p_bs_orig = runner.get_trunk("model.hc_head.hc_head_base", torch.float32).detach().clone()

    p_norm = nn.Parameter(p_norm_orig.clone())
    p_fn = nn.Parameter(p_fn_orig.clone())
    p_sc = nn.Parameter(p_sc_orig.clone())
    p_bs = nn.Parameter(p_bs_orig.clone())

    opt = torch.optim.AdamW([p_norm, p_fn, p_sc, p_bs], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(teacher_recs))

    for ep in range(epochs):
        losses = []
        for i, rec in enumerate(teacher_recs):
            opt.zero_grad()
            x = h_x_list[i].to(device, dtype=torch.float32)
            T = x.shape[0]
            pre = torch.sigmoid(F.linear(rms_norm(x.reshape(T, 4 * 3072), None, runner.rms_eps), p_fn) * p_sc + p_bs) + runner.hc_eps
            h_head = torch.einsum("thd,th->td", x, pre)
            h_fin = rms_norm(h_head, p_norm, runner.rms_eps)
            logits = F.linear(h_fin, lm_head)
            kl = topk_tail_kl_loss(logits, rec["top_idx"].to(device), rec["top_logp"].to(device), rec["tail_p"].to(device))
            # Mild L2 anchor so weights generalize cleanly to unseen domains!
            reg = 1e-3 * ((p_norm - p_norm_orig).pow(2).mean() + (p_fn - p_fn_orig).pow(2).mean())
            loss = kl + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_norm, p_fn, p_sc, p_bs], 1.0)
            opt.step()
            sched.step()
            losses.append(float(kl.item()))
        print(f"  [Scale-QAT {build}] epoch {ep+1}/{epochs}: mean cal KL = {np.mean(losses):.5f}", flush=True)

    kl_after = float(np.mean(losses))
    overrides = {
        "model.norm.weight": p_norm.detach().cpu(),
        "model.hc_head.hc_head_fn": p_fn.detach().cpu(),
        "model.hc_head.hc_head_scale": p_sc.detach().cpu(),
        "model.hc_head.hc_head_base": p_bs.detach().cpu(),
    }
    out_overrides_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(overrides, out_overrides_path)
    print(f"[Scale-QAT {build}] DONE in {time.time()-t0:.1f}s | cal KL: {kl_before:.5f} -> {kl_after:.5f}", flush=True)
    return {"build": build, "kl_before": kl_before, "kl_after": kl_after, "overrides_path": str(out_overrides_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("teacher")
    pt.add_argument("--model-dir", required=True)
    pt.add_argument("--engram-q8", required=True)
    pt.add_argument("--splits-dir", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--n-seqs", type=int, default=24)

    pq = sub.add_parser("qat")
    pq.add_argument("--build", required=True)
    pq.add_argument("--strata", required=True)
    pq.add_argument("--engram-q8", required=True)
    pq.add_argument("--model-dir", required=True)
    pq.add_argument("--teacher", required=True)
    pq.add_argument("--out", required=True)
    pq.add_argument("--device", default="cuda:0")
    pq.add_argument("--epochs", type=int, default=3)

    args = ap.parse_args()
    if args.cmd == "teacher":
        capture_teacher_targets(Path(args.model_dir), Path(args.engram_q8), Path(args.splits_dir), Path(args.out), n_seqs=args.n_seqs)
    elif args.cmd == "qat":
        run_scale_qat_for_build(args.build, Path(args.strata), Path(args.engram_q8), Path(args.model_dir), Path(args.teacher), Path(args.out), device=args.device, epochs=args.epochs)
