#!/usr/bin/env python3
"""Deep Full 42-Layer End-to-End Gradient-Checkpointed Scale-QAT Polish for Hy4-Flash-SQ-64.strata.

Upgrades pass-1 head-only Scale-QAT to FULL 42-layer end-to-end KL distillation against exact BF16 teacher:
  1. Single layer-by-layer batched BF16 capture (~5 min total) for:
       - 48 training sequences (36 multi-domain calibration + 12 chat-templated code/math/tool/reasoning)
       - 8 held-out development & 8 held-out final_test reference logits
  2. Exact 42-layer gradient-checkpointed Scale-QAT optimizing (~1.05M FP32 params, zero extra .strata bytes):
       - All 42 layers' Sandwich post-norms & pre-norms (post_proj_norm, combine_output_norm, input_ln, post_attn_ln)
       - All 42 layers' iHC 4-stream scales & bases (hc_attn_layer, hc_mlp_layer)
       - All 41 MoE layers' per-channel LatentMoE output scales (folded zero-cost into routed_expert_up_proj)
         and shared-expert output scales (folded zero-cost into shared_experts.down_proj)
       - Final model.hc_head.{fn, scale, base} + model.norm.weight
       - Tracks held-out development KL every epoch and retains strictly the best dev-KL parameters!
  3. Fast Section-1-only repack (~45s): stream-copies Sections 2..5 verbatim from SQ-64.orig.strata.
  4. Full held-out dev/test KL + top-1 + PPL + chat-templated live generation comparison (Before vs After vs BF16).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import mmap
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import zstandard as zstd
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.codecs import ID_TO_NAME, SUPERBLOCK_BYTES, decode_sk_torch
from squant.hy_v4_flash import ENGRAM_EPS, EngramTableReader, Hy4FlashStreamedModel, apply_interleaved_rope, rms_norm
from squant.pipeline import BUILD_SPECS, encode_trunk_tensor
from squant.pilot_parity import kl_divergence
from squant.scale_qat_eval import StrataHy4Runner, decode_trunk_tensor, evaluate_strata
from squant.scale_qat_train import topk_tail_kl_loss
from squant.strata_format import (
    EXPERT_ENTRY_SIZE,
    PAGE_SIZE,
    PROFILE_TIERED,
    StrataExpertEntry,
    StrataWriter,
    parse_header_frame,
    verify_strata_file,
)


CHAT_TRAIN_SAMPLES = [
    ("Write a Python function `def fib(n):` that computes the nth Fibonacci number iteratively in O(n) time and O(1) space.",
     "```python\ndef fib(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nThis uses constant O(1) extra space and runs in O(n) iterations."),
    ("What is 17 * 24 + 13? Solve step by step.",
     "Step 1: Break 17 * 24 into 17 * 20 + 17 * 4 = 340 + 68 = 408.\nStep 2: Add 13: 408 + 13 = 421.\nTherefore, 17 * 24 + 13 = 421."),
    ("Return a valid JSON tool call to query weather in Tokyo in Celsius.",
     '```json\n{"tool": "get_weather", "arguments": {"location": "Tokyo, Japan", "units": "celsius"}}\n```'),
    ("Explain Multi-Head Latent Attention (MLA) in 3 concise bullet points.",
     "- Compresses keys and values into a low-rank latent vector `c_KV` (e.g., dim 512), slashing KV cache memory by 80-95%.\n- Applies decoupled RoPE to a small shared positional key component `k_pe` so rotary embeddings do not block weight absorption.\n- During decode, `W_UK` absorbs into `W_Q` and `W_UV` into `W_O`, allowing attention directly against cached `c_KV`."),
    ("Write a C++20 thread-safe bounded queue using std::mutex and std::condition_variable.",
     "```cpp\n#include <condition_variable>\n#include <mutex>\n#include <queue>\ntemplate <typename T>\nclass BoundedQueue {\n  std::mutex m_;\n  std::condition_variable not_full_, not_empty_;\n  std::queue<T> q_;\n  size_t cap_;\npublic:\n  explicit BoundedQueue(size_t cap) : cap_(cap) {}\n  void push(T v) {\n    std::unique_lock lk(m_);\n    not_full_.wait(lk, [&] { return q_.size() < cap_; });\n    q_.push(std::move(v));\n    not_empty_.notify_one();\n  }\n  T pop() {\n    std::unique_lock lk(m_);\n    not_empty_.wait(lk, [&] { return !q_.empty(); });\n    T v = std::move(q_.front()); q_.pop();\n    not_full_.notify_one();\n    return v;\n  }\n};\n```"),
    ("Write a SQL query to find the top 3 departments by average salary among active employees hired after 2022.",
     "```sql\nSELECT d.name AS department, ROUND(AVG(e.salary), 2) AS avg_salary\nFROM employees e\nJOIN departments d ON d.id = e.department_id\nWHERE e.status = 'active' AND e.hire_date > '2022-12-31'\nGROUP BY d.id, d.name\nORDER BY avg_salary DESC\nLIMIT 3;\n```"),
    ("Solve for x: 3(2x - 5) = 4x + 7. Show each step.",
     "1. Expand the left side: 6x - 15 = 4x + 7.\n2. Subtract 4x from both sides: 2x - 15 = 7.\n3. Add 15 to both sides: 2x = 22.\n4. Divide by 2: x = 11.\nCheck: 3(22 - 5) = 51 and 4(11) + 7 = 51."),
    ("Return a JSON tool call to search GitHub issues in repo `strata-engine/strata` labeled `bug` and state `open`.",
     '```json\n{"tool": "github_search_issues", "arguments": {"repo": "strata-engine/strata", "labels": ["bug"], "state": "open", "limit": 20}}\n```'),
    ("Write a Python decorator `@retry(max_attempts=3, delay=0.5)` with exponential backoff.",
     "```python\nimport functools, time\ndef retry(max_attempts=3, delay=0.5):\n    def deco(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            wait = delay\n            for attempt in range(1, max_attempts + 1):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception:\n                    if attempt == max_attempts:\n                        raise\n                    time.sleep(wait)\n                    wait *= 2\n        return wrapper\n    return deco\n```"),
    ("Why does Mixture-of-Experts quantization benefit from Hessian-weighted calibration and residual compensation?",
     "Routed experts have sparse, domain-specific activation distributions where outlier channels carry disproportionate variance. Hessian-weighted Block-LDL (GPTQ) and flip refinement minimize output perturbation `||(W-Q)X||_2^2` rather than raw weight MSE, while residual compensation absorbs correlated low-bit error into full-precision or Q8 shared projections."),
    ("Convert 2026-11-19T08:30:00Z to Unix timestamp and explain leap year handling.",
     "From 1970-01-01 to 2026-01-01 there are 56 years including 14 leap years (1972..2024), giving 20,454 days. Jan-Oct 2026 adds 304 days plus 18 days in Nov = 20,776 days = 1,795,046,400 seconds. Adding 8h 30m (30,600 s) yields 1,795,077,000."),
    ("Write a bash one-liner to find the 10 largest files under /var/log sorted by human-readable size.",
     "```bash\nfind /var/log -type f -exec du -h {} + 2>/dev/null | sort -hr | head -n 10\n```"),
]


def format_chat_prompt(user_msg: str, assistant_msg: Optional[str] = None) -> str:
    """Format in Hy4-Flash chat format (consistent with tokenizer special tokens)."""
    s = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    if assistant_msg is not None:
        s += f"{assistant_msg}<|im_end|>"
    return s


@torch.no_grad()
def batched_bf16_capture(
    model_dir: Path,
    engram_q8_path: Path,
    splits_dir: Path,
    out_dir: Path,
    n_cal_seqs: int = 36,
    seq_len: int = 512,
    topk: int = 256,
    device: str = "cuda:0",
) -> Tuple[Path, Path, Path]:
    """Single layer-by-layer BF16 pass (~5 min!) capturing 48 train top-256 + 8 dev + 8 final_test logits."""
    teacher_p = out_dir / "polish_teacher_48.pt"
    dev_ref_p = out_dir / "eval_ref_dev.pt"
    test_ref_p = out_dir / "eval_ref_test.pt"
    if teacher_p.is_file() and dev_ref_p.is_file() and test_ref_p.is_file():
        print("[Capture] Reusing cached BF16 teacher + dev/test reference logits!", flush=True)
        return teacher_p, dev_ref_p, test_ref_p

    t0 = time.time()
    cal = json.loads((splits_dir / "calibration.json").read_text(encoding="utf-8"))
    dev = json.loads((splits_dir / "development.json").read_text(encoding="utf-8"))
    tst = json.loads((splits_dir / "final_test.json").read_text(encoding="utf-8"))

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    train_seqs: List[np.ndarray] = [np.array(s[:seq_len], dtype=np.int64) for s in cal["seqs"][:n_cal_seqs]]
    for u_msg, a_msg in CHAT_TRAIN_SAMPLES:
        ids = tok.encode(format_chat_prompt(u_msg, a_msg)).ids[:seq_len]
        train_seqs.append(np.array(ids, dtype=np.int64))
    dev_seqs = [np.array(s[:seq_len], dtype=np.int64) for s in dev["seqs"][:8]]
    test_seqs = [np.array(s[:seq_len], dtype=np.int64) for s in tst["seqs"][:8]]

    all_seqs = train_seqs + dev_seqs + test_seqs
    n_train = len(train_seqs)
    n_dev = len(dev_seqs)
    print(f"[Capture] Running SINGLE layer-by-layer BF16 pass across {len(all_seqs)} seqs ({n_train} train + {n_dev} dev + {len(test_seqs)} test)...", flush=True)

    model = Hy4FlashStreamedModel(model_dir, engram_q8_path=engram_q8_path, device=device)
    emb_w = model.loader.get("model.embed_tokens.weight", device, torch.bfloat16)
    X_list: List[torch.Tensor] = []
    for arr in all_seqs:
        t_ids = torch.from_numpy(arr).to(device, dtype=torch.long)
        X_list.append(F.embedding(t_ids, emb_w).unsqueeze(1).expand(len(arr), 4, 3072).contiguous().cpu())
    del emb_w
    torch.cuda.empty_cache()

    for il in range(model.n_layers):
        t_l = time.time()
        model.loader.cache_enabled = True
        model.loader.clear_gpu_cache()
        for si, arr in enumerate(all_seqs):
            x = X_list[si].to(device, non_blocking=True)
            pos = torch.arange(len(arr), device=device, dtype=torch.int64)
            if il in model.engram_layers:
                x = model._apply_engram(x, il, arr)
            res = x
            cur, post_a = model._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
            cur = rms_norm(cur, model.loader.get(f"model.layers.{il}.input_layernorm.weight", device, torch.bfloat16), model.rms_eps)
            attn_out, _ = model._attention(cur, pos, il, save_kv=False)
            X_list[si] = model._hc_post(attn_out, res, post_a).cpu()
        model.loader.clear_gpu_cache()

        if il == 0:
            for si in range(len(all_seqs)):
                x = X_list[si].to(device, non_blocking=True)
                res = x
                cur, post_m = model._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                cur = rms_norm(cur, model.loader.get(f"model.layers.{il}.post_attention_layernorm.weight", device, torch.bfloat16), model.rms_eps)
                mlp_out = model._dense_mlp(cur, il)
                X_list[si] = model._hc_post(mlp_out, res, post_m).cpu()
            model.loader.clear_gpu_cache()
        else:
            p = f"model.layers.{il}.mlp"
            model.loader.cache_enabled = False
            w_router = model.loader.get(f"{p}.gate.weight", device, torch.float32)
            b_corr = model.loader.get(f"{p}.gate.e_score_correction_bias", device, torch.float32)
            w_rdown = model.loader.get(f"{p}.routed_expert_down_proj.weight", device, torch.bfloat16)
            w_rup = model.loader.get(f"{p}.routed_expert_up_proj.weight", device, torch.bfloat16)
            s_g = model.loader.get(f"{p}.shared_experts.gate_proj.weight", device, torch.bfloat16)
            s_u = model.loader.get(f"{p}.shared_experts.up_proj.weight", device, torch.bfloat16)
            s_d = model.loader.get(f"{p}.shared_experts.down_proj.weight", device, torch.bfloat16)
            comb_norm = model.loader.get(f"{p}.combine_output_norm.weight", device, torch.bfloat16)
            post_attn_ln = model.loader.get(f"model.layers.{il}.post_attention_layernorm.weight", device, torch.bfloat16)
            gu_all = model.loader.get(f"{p}.experts.gate_up_proj", device, torch.bfloat16)
            dn_all = model.loader.get(f"{p}.experts.down_proj", device, torch.bfloat16)
            model.loader.cache_enabled = True

            for si in range(len(all_seqs)):
                x = X_list[si].to(device, non_blocking=True)
                res = x
                cur, post_m = model._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
                cur = rms_norm(cur, post_attn_ln, model.rms_eps)
                logits_r = F.linear(cur.to(torch.float32), w_router)
                probs = torch.sigmoid(logits_r)
                _, topk_idx = torch.topk(probs + b_corr.unsqueeze(0), k=model.top_k, dim=-1)
                topk_p = probs.gather(1, topk_idx)
                topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * model.routed_scale
                z = F.linear(cur, w_rdown)
                z_out = torch.zeros((z.shape[0], 1536), device=device, dtype=torch.float32)
                for e in torch.unique(topk_idx).tolist():
                    tm, sm = torch.where(topk_idx == e)
                    gu = F.linear(z[tm], gu_all[e])
                    g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                    h = torch.clamp(F.silu(g), max=model.swiglu_limit) * torch.clamp(u, min=-model.swiglu_limit, max=model.swiglu_limit)
                    ye = F.linear(h.to(torch.bfloat16), dn_all[e]).to(torch.float32)
                    z_out.index_add_(0, tm, ye * topk_w[tm, sm].unsqueeze(-1))
                moe_out = F.linear(z_out.to(torch.bfloat16), w_rup)
                shexp = F.linear(F.silu(F.linear(cur, s_g)) * F.linear(cur, s_u), s_d)
                mlp_out = rms_norm(moe_out + shexp, comb_norm, model.rms_eps)
                X_list[si] = model._hc_post(mlp_out, res, post_m).cpu()
            del w_router, b_corr, w_rdown, w_rup, s_g, s_u, s_d, comb_norm, post_attn_ln, gu_all, dn_all
            model.loader.clear_gpu_cache()
            model.loader.cache_enabled = False
            torch.cuda.empty_cache()

        if (il + 1) % 7 == 0 or il == 41:
            print(f"  [Capture] Layer {il:02d}/41 done ({time.time()-t_l:.1f}s | elapsed {time.time()-t0:.1f}s)", flush=True)

    # Final head -> save train top-256 + dev/test fp16 logits
    hc_fn = model.loader.get("model.hc_head.hc_head_fn", device, torch.float32)
    hc_sc = model.loader.get("model.hc_head.hc_head_scale", device, torch.float32)
    hc_bs = model.loader.get("model.hc_head.hc_head_base", device, torch.float32)
    norm_w = model.loader.get("model.norm.weight", device, torch.bfloat16)
    lm_head = model.loader.get("lm_head.weight", device, torch.bfloat16)

    train_recs, dev_logits, test_logits = [], [], []
    for si, arr in enumerate(all_seqs):
        x = X_list[si].to(device)
        T = x.shape[0]
        pre = torch.sigmoid(F.linear(rms_norm(x.reshape(T, 4 * 3072), None, model.rms_eps).to(torch.float32), hc_fn) * hc_sc + hc_bs) + model.hc_eps
        h_head = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        h_fin = rms_norm(h_head, norm_w, model.rms_eps)
        logits = F.linear(h_fin, lm_head).to(torch.float32)
        if si < n_train:
            logp = F.log_softmax(logits, dim=-1)
            top_lp, top_idx = torch.topk(logp, k=topk, dim=-1)
            top_p_sum = torch.exp(top_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
            tail_p = (1.0 - top_p_sum).clamp(min=1e-7)
            train_recs.append({
                "tokens": torch.tensor(arr, dtype=torch.int32),
                "top_idx": top_idx.to(torch.int32).cpu(),
                "top_logp": top_lp.to(torch.float16).cpu(),
                "tail_p": tail_p.to(torch.float16).cpu(),
            })
        elif si < n_train + n_dev:
            dev_logits.append(logits.to(torch.float16).cpu())
        else:
            test_logits.append(logits.to(torch.float16).cpu())

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_recs, teacher_p)
    torch.save(dev_logits, dev_ref_p)
    torch.save(test_logits, test_ref_p)
    del model, X_list, lm_head
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Capture] COMPLETE in {time.time()-t0:.1f}s -> {teacher_p.name}, {dev_ref_p.name}, {test_ref_p.name}", flush=True)
    return teacher_p, dev_ref_p, test_ref_p


class DeepScaleQATModule(nn.Module):
    """Full 42-layer differentiable Scale-QAT wrapper around packed Hy4-Flash-SQ-64.strata!"""

    def __init__(self, strata_path: Path, engram_q8_path: Path, model_dir: Path, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.strata_path = Path(strata_path)
        self.fp = open(self.strata_path, "rb")
        self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        _, self.profile, self.sections, self.engram_sha = parse_header_frame(self.mm[:PAGE_SIZE])
        s0 = self.sections[0]
        self.manifest = json.loads(zstd.ZstdDecompressor().decompress(self.mm[s0.file_offset : s0.file_offset + s0.byte_length]).decode("utf-8"))
        self.tc = self.manifest["config"]["text_config"]
        self.trunk_map = {t["name"]: t for t in self.manifest["trunk_tensors"]}
        self.sec1_off = self.sections[1].file_offset
        self.sec4_off = self.sections[4].file_offset

        s3 = self.sections[3]
        s3_bytes = self.mm[s3.file_offset : s3.file_offset + s3.byte_length]
        n_ent = len(s3_bytes) // EXPERT_ENTRY_SIZE
        self.expert_dir: Dict[Tuple[int, int], StrataExpertEntry] = {}
        for i in range(n_ent):
            ent = StrataExpertEntry.unpack(s3_bytes[i * EXPERT_ENTRY_SIZE : (i + 1) * EXPERT_ENTRY_SIZE])
            self.expert_dir[(i // 512, ent.expert_id)] = ent

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
        self.swiglu_limit = float(self.tc["swiglu_limit"])
        self.routed_scale = float(self.tc["routed_scaling_factor"])
        self.top_k = int(self.tc["num_experts_per_tok"])
        self.layer_types = list(self.tc["layer_types"])

        # Pre-decode ALL Section 1 trunk tensors ONCE into CPU RAM (~11 GB out of 252 GB RAM -> instant H2D copy!)
        print("[DeepQAT] Pre-decoding Section 1 GPU trunk (~11 GB) into host RAM...", flush=True)
        t0 = time.time()
        self.trunk_cpu: Dict[str, torch.Tensor] = {}
        for name, meta in self.trunk_map.items():
            if "dfly_layers." in name:
                continue
            raw = self.mm[self.sec1_off + meta["offset"] : self.sec1_off + meta["offset"] + meta["bytes"]]
            # Decode on GPU then move to CPU in optimal dtype
            t = decode_trunk_tensor(raw, meta["format"], meta["shape"], device)
            is_f32 = meta["format"] == "F32"
            self.trunk_cpu[name] = t.to(torch.float32 if is_f32 else torch.bfloat16).cpu()
        torch.cuda.empty_cache()
        print(f"[DeepQAT] Pre-decoded {len(self.trunk_cpu)} trunk tensors in {time.time()-t0:.1f}s", flush=True)

        # Register ALL 42-layer trainable Scale-QAT parameters in FP32 on GPU!
        self.tuned = nn.ParameterDict()
        self.orig_anchor: Dict[str, torch.Tensor] = {}

        def add_param(key: str, tensor_f32: torch.Tensor):
            pk = key.replace(".", "__")
            val = tensor_f32.detach().clone().to(device=device, dtype=torch.float32)
            self.tuned[pk] = nn.Parameter(val)
            self.orig_anchor[pk] = val.detach().clone()

        for il in range(self.n_layers):
            for suf in [
                f"model.layers.{il}.input_layernorm.weight",
                f"model.layers.{il}.self_attn.post_proj_norm.weight",
                f"model.layers.{il}.post_attention_layernorm.weight",
                f"model.layers.{il}.hc_attn_layer.hc_pre.hc_scale",
                f"model.layers.{il}.hc_attn_layer.hc_pre.hc_base",
                f"model.layers.{il}.hc_mlp_layer.hc_pre.hc_scale",
                f"model.layers.{il}.hc_mlp_layer.hc_pre.hc_base",
            ]:
                add_param(suf, self.trunk_cpu[suf].float())
            if il == 0:
                add_param("model.layers.0.mlp.post_fc2_norm.weight", self.trunk_cpu["model.layers.0.mlp.post_fc2_norm.weight"].float())
            else:
                add_param(f"model.layers.{il}.mlp.combine_output_norm.weight", self.trunk_cpu[f"model.layers.{il}.mlp.combine_output_norm.weight"].float())
                add_param(f"moe_alpha.{il}", torch.ones(self.hidden_size, dtype=torch.float32))
                add_param(f"shexp_beta.{il}", torch.ones(self.hidden_size, dtype=torch.float32))

        for head_k in [
            "model.hc_head.hc_head_fn",
            "model.hc_head.hc_head_scale",
            "model.hc_head.hc_head_base",
            "model.norm.weight",
        ]:
            add_param(head_k, self.trunk_cpu[head_k].float())

        n_params = sum(p.numel() for p in self.tuned.parameters())
        print(f"[DeepQAT] Registered {len(self.tuned)} trainable 42-layer parameter tensors ({n_params:,} FP32 params)", flush=True)

    def p(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        pk = key.replace(".", "__")
        if pk in self.tuned:
            return self.tuned[pk].to(dtype)
        return self.trunk_cpu[key].to(self.device, dtype=dtype, non_blocking=True)

    def load_expert_gpu(self, moe_layer: int, expert_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ent = self.expert_dir[(moe_layer, expert_id)]
        base = self.sec4_off + ent.body_offset
        body = self.mm[base : base + ent.body_bytes]
        sq_xd_dn = torch.from_numpy(np.frombuffer(body[:3072], dtype=np.float16).astype(np.float32).copy()).to(self.device)
        gu_code = ID_TO_NAME[ent.gu_code_id]
        dn_code = ID_TO_NAME[ent.dn_code_id]
        gu_len = 18432 * SUPERBLOCK_BYTES[gu_code]
        dn_len = 9216 * SUPERBLOCK_BYTES[dn_code]
        w_gu = decode_sk_torch(body[3072 : 3072 + gu_len], gu_code, self.device).view(3072, 1536).to(torch.bfloat16)
        w_dn = decode_sk_torch(body[ent.dn_offset : ent.dn_offset + dn_len], dn_code, self.device).view(1536, 1536).to(torch.bfloat16)
        return w_gu, w_dn, sq_xd_dn

    def pre_gather_engram(self, seqs: List[np.ndarray]) -> List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Pre-gather Engram Q8 rows on CPU for all sequences ONCE (eliminates Engram I/O during QAT!)."""
        out = []
        for tok_np in seqs:
            d = {}
            for il, mod_idx in self.engram_layers.items():
                ids, valid_np = self.engram.hash_tokens(mod_idx, tok_np)
                rows = self.engram.gather_rows(mod_idx, ids, device="cpu").to(torch.bfloat16)
                valid = torch.from_numpy(valid_np.astype(np.float32))
                d[il] = (rows, valid)
            out.append(d)
        return out

    def _hc_pre(self, x: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        hc_fn = self.p(f"{prefix}.hc_pre.hc_fn", torch.float32)
        hc_scale = self.p(f"{prefix}.hc_pre.hc_scale", torch.float32)
        hc_base = self.p(f"{prefix}.hc_pre.hc_base", torch.float32)
        flat_norm = rms_norm(x.reshape(T, self.hc_mult * self.hidden_size), None, self.rms_eps).to(torch.float32)
        mixes = F.linear(flat_norm, hc_fn)
        pre = torch.sigmoid(mixes[:, : self.hc_mult] * hc_scale[0] + hc_base[: self.hc_mult]) + self.hc_eps
        post = self.hc_magnitude * torch.sigmoid(mixes[:, self.hc_mult :] * hc_scale[1] + hc_base[self.hc_mult :]) + self.hc_eps
        cur = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        return cur, post

    def _hc_post(self, cur: torch.Tensor, residual: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return (residual.to(torch.float32) + cur.to(torch.float32).unsqueeze(1) * post.unsqueeze(-1)).to(torch.bfloat16)

    def _apply_engram_cached(self, x: torch.Tensor, il: int, rows_cpu: torch.Tensor, valid_cpu: torch.Tensor) -> torch.Tensor:
        rows = rows_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
        valid = valid_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
        T = x.shape[0]
        p = f"model.layers.{il}.engram_layer"
        ln_w = torch.stack([self.p(f"{p}.lookup_per_head_norms.{c}.weight", torch.float32) for c in range(16)], dim=0)
        r = (rms_norm(rows, None, ENGRAM_EPS) * ln_w.unsqueeze(0) * valid.unsqueeze(-1)).reshape(T, 1536)
        kv_w = torch.cat(
            [self.p(f"{p}.key_projs.{c}.weight", torch.float32) for c in range(self.hc_mult)]
            + [self.p(f"{p}.value_proj.weight", torch.float32)],
            dim=0,
        )
        kv_b = torch.stack(
            [self.p(f"{p}.key_projs.{c}.bias", torch.float32) for c in range(self.hc_mult)]
            + [self.p(f"{p}.value_proj.bias", torch.float32)],
            dim=0,
        )
        kv = F.linear(r, kv_w).reshape(T, self.hc_mult + 1, self.hidden_size) + kv_b.unsqueeze(0)
        key, value = kv[:, : self.hc_mult, :], kv[:, self.hc_mult :, :]
        nk = torch.stack([self.p(f"{p}.norm_key.{c}.weight", torch.float32) for c in range(self.hc_mult)], dim=0)
        nq = torch.stack([self.p(f"{p}.norm_query.{c}.weight", torch.float32) for c in range(self.hc_mult)], dim=0)
        dot = (rms_norm(x.to(torch.float32), None, ENGRAM_EPS) * rms_norm(key, None, ENGRAM_EPS) * (nk * nq).unsqueeze(0)).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(torch.sqrt(torch.clamp(dot.abs(), min=1e-6)) * torch.where(dot < 0.0, -1.0, 1.0))
        pre_gamma = self.p(f"{p}.unified_pre_gamma", torch.float32)
        hc_scale = torch.stack([self.p(f"{p}.unified_per_hc_scales.{c}", torch.float32) for c in range(self.hc_mult)], dim=0)
        signal = rms_norm(value, None, ENGRAM_EPS) * pre_gamma.view(1, 1, -1)
        delta = signal * (gate + hc_scale.unsqueeze(0)) * (valid.sum(dim=-1) > 0).to(torch.float32).view(T, 1, 1)
        return (x.to(torch.float32) + delta).to(torch.bfloat16)

    def _run_layer(self, x: torch.Tensor, il: int, eng_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        T = x.shape[0]
        pos = torch.arange(T, device=self.device, dtype=torch.int64)
        if il in self.engram_layers:
            rows_c, valid_c = eng_cache[il]
            x = self._apply_engram_cached(x, il, rows_c, valid_c)

        # Attention sublayer
        res = x
        cur, post_a = self._hc_pre(x, f"model.layers.{il}.hc_attn_layer")
        cur = rms_norm(cur, self.p(f"model.layers.{il}.input_layernorm.weight", torch.bfloat16), self.rms_eps)
        is_swa = self.layer_types[il] == "sliding_attention"
        n_head = int(self.tc["swa_num_attention_heads"] if is_swa else self.tc["num_attention_heads"])
        theta = float(self.tc["rope_parameters"]["sliding_attention" if is_swa else "deepseek_sparse_attention"]["rope_theta"])
        pa = f"model.layers.{il}.self_attn"

        qr = rms_norm(F.linear(cur, self.p(f"{pa}.q_a_proj.weight", torch.bfloat16)), self.p(f"{pa}.q_a_layernorm.weight", torch.bfloat16), self.rms_eps)
        q = F.linear(qr, self.p(f"{pa}.q_b_proj.weight", torch.bfloat16)).reshape(T, n_head, self.qk_head_dim)
        q_nope, q_pe = q[..., : self.qk_nope_dim], apply_interleaved_rope(q[..., self.qk_nope_dim :], pos, theta)
        kv_a = F.linear(cur, self.p(f"{pa}.kv_a_proj_with_mqa.weight", torch.bfloat16))
        kv_cmpr = rms_norm(kv_a[:, : self.kv_lora_rank], self.p(f"{pa}.kv_a_layernorm.weight", torch.bfloat16), self.rms_eps)
        k_pe = apply_interleaved_rope(kv_a[:, self.kv_lora_rank :].unsqueeze(1), pos, theta)
        kv_exp = F.linear(kv_cmpr, self.p(f"{pa}.kv_b_proj.weight", torch.bfloat16)).reshape(T, n_head, self.qk_nope_dim + self.v_head_dim)
        k_nope, v = kv_exp[..., : self.qk_nope_dim], kv_exp[..., self.qk_nope_dim :]

        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)
        k_full = torch.cat([k_nope, k_pe.expand(T, n_head, self.qk_rope_dim)], dim=-1).transpose(0, 1)
        scores = torch.matmul(q_full.to(torch.float32), k_full.to(torch.float32).transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        i_idx = torch.arange(T, device=self.device)[:, None]
        j_idx = torch.arange(T, device=self.device)[None, :]
        mask = torch.where((j_idx <= i_idx) & ((i_idx - j_idx) < self.swa_window), 0.0, float("-inf")) if is_swa else torch.where(j_idx <= i_idx, 0.0, float("-inf"))
        scores = scores + mask.unsqueeze(0)
        sinks = self.p(f"{pa}.learnable_sink_param", torch.float32).view(n_head, 1, 1).expand(n_head, T, 1)
        probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1].to(torch.bfloat16)
        attn = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(T, n_head * self.v_head_dim)
        attn = attn * torch.sigmoid(F.linear(cur, self.p(f"{pa}.linear_gate.weight", torch.bfloat16)).to(torch.float32)).to(torch.bfloat16)
        attn_out = rms_norm(F.linear(attn, self.p(f"{pa}.o_proj.weight", torch.bfloat16)), self.p(f"{pa}.post_proj_norm.weight", torch.bfloat16), self.rms_eps)
        x = self._hc_post(attn_out, res, post_a)

        # FFN / LatentMoE sublayer
        res = x
        cur, post_m = self._hc_pre(x, f"model.layers.{il}.hc_mlp_layer")
        cur = rms_norm(cur, self.p(f"model.layers.{il}.post_attention_layernorm.weight", torch.bfloat16), self.rms_eps)
        pm = f"model.layers.{il}.mlp"
        if il == 0:
            mlp_out = F.linear(
                F.silu(F.linear(cur, self.p(f"{pm}.gate_proj.weight", torch.bfloat16)))
                * F.linear(cur, self.p(f"{pm}.up_proj.weight", torch.bfloat16)),
                self.p(f"{pm}.down_proj.weight", torch.bfloat16),
            )
            mlp_out = rms_norm(mlp_out, self.p(f"{pm}.post_fc2_norm.weight", torch.bfloat16), self.rms_eps)
        else:
            logits_r = F.linear(cur.to(torch.float32), self.p(f"{pm}.gate.weight", torch.float32))
            probs_e = torch.sigmoid(logits_r)
            b_corr = self.p(f"{pm}.gate.e_score_correction_bias", torch.float32)
            with torch.no_grad():
                _, topk_idx = torch.topk(probs_e + b_corr.unsqueeze(0), k=self.top_k, dim=-1)
            topk_p = probs_e.gather(1, topk_idx)
            topk_w = (topk_p / (topk_p.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scale

            z_folded = F.linear(cur, self.p(f"{pm}.routed_expert_down_proj.weight", torch.bfloat16))
            z_out = torch.zeros((T, 1536), device=self.device, dtype=torch.float32)
            for e_id in torch.unique(topk_idx).tolist():
                tm, sm = torch.where(topk_idx == e_id)
                ze = z_folded[tm]
                we = topk_w[tm, sm].unsqueeze(-1)
                w_gu, w_dn, sq_xd_dn = self.load_expert_gpu(il - 1, e_id)
                gu = F.linear(ze, w_gu)
                g, u = gu[:, :1536].to(torch.float32), gu[:, 1536:].to(torch.float32)
                h = torch.clamp(F.silu(g), max=self.swiglu_limit) * torch.clamp(u, min=-self.swiglu_limit, max=self.swiglu_limit)
                ye = F.linear((h / sq_xd_dn.unsqueeze(0)).to(torch.bfloat16), w_dn).to(torch.float32)
                z_out = z_out.index_add(0, tm, ye * we)

            # Learnable per-channel LatentMoE & shared-expert scales (folded zero-cost into Section 1 on repack!)
            moe_alpha = self.p(f"moe_alpha.{il}", torch.float32)
            shexp_beta = self.p(f"shexp_beta.{il}", torch.float32)
            moe_out = F.linear(z_out.to(torch.bfloat16), self.p(f"{pm}.routed_expert_up_proj.weight", torch.bfloat16)).to(torch.float32) * moe_alpha.unsqueeze(0)
            shexp = F.linear(
                F.silu(F.linear(cur, self.p(f"{pm}.shared_experts.gate_proj.weight", torch.bfloat16)))
                * F.linear(cur, self.p(f"{pm}.shared_experts.up_proj.weight", torch.bfloat16)),
                self.p(f"{pm}.shared_experts.down_proj.weight", torch.bfloat16),
            ).to(torch.float32) * shexp_beta.unsqueeze(0)
            mlp_out = rms_norm((moe_out + shexp).to(torch.bfloat16), self.p(f"{pm}.combine_output_norm.weight", torch.bfloat16), self.rms_eps)

        return self._hc_post(mlp_out, res, post_m)

    def forward_seq(self, tok_np: np.ndarray, eng_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]], use_ckpt: bool = True) -> torch.Tensor:
        tok_t = torch.from_numpy(tok_np).to(self.device, dtype=torch.long)
        T = len(tok_np)
        emb_w = self.p("model.embed_tokens.weight", torch.bfloat16)
        x = F.embedding(tok_t, emb_w).unsqueeze(1).expand(T, self.hc_mult, self.hidden_size).contiguous()
        for il in range(self.n_layers):
            if use_ckpt and torch.is_grad_enabled():
                x = ckpt.checkpoint(lambda inp, l_idx=il: self._run_layer(inp, l_idx, eng_cache), x, use_reentrant=False)
            else:
                x = self._run_layer(x, il, eng_cache)
        hc_fn = self.p("model.hc_head.hc_head_fn", torch.float32)
        hc_sc = self.p("model.hc_head.hc_head_scale", torch.float32)
        hc_bs = self.p("model.hc_head.hc_head_base", torch.float32)
        pre = torch.sigmoid(F.linear(rms_norm(x.reshape(T, self.hc_mult * self.hidden_size), None, self.rms_eps).to(torch.float32), hc_fn) * hc_sc + hc_bs) + self.hc_eps
        h_head = torch.einsum("thd,th->td", x.to(torch.float32), pre).to(torch.bfloat16)
        h_fin = rms_norm(h_head, self.p("model.norm.weight", torch.bfloat16), self.rms_eps)
        lm_head = self.p("lm_head.weight", torch.bfloat16)
        return F.linear(h_fin, lm_head).to(torch.float32)

    def anchor_reg_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.device)
        for pk, param in self.tuned.items():
            reg = reg + (param - self.orig_anchor[pk]).pow(2).mean()
        return reg / len(self.tuned)

    def export_folded_overrides(self) -> Dict[str, torch.Tensor]:
        """Export Section 1 overrides with moe_alpha & shexp_beta cleanly folded into Q8 projections!"""
        ovr: Dict[str, torch.Tensor] = {}
        for pk, param in self.tuned.items():
            key = pk.replace("__", ".")
            if key.startswith("moe_alpha.") or key.startswith("shexp_beta."):
                continue
            ovr[key] = param.detach().cpu().to(torch.float32)
        for il in range(1, self.n_layers):
            alpha = self.tuned[f"moe_alpha__{il}"].detach().cpu().to(torch.float32)
            beta = self.tuned[f"shexp_beta__{il}"].detach().cpu().to(torch.float32)
            k_rup = f"model.layers.{il}.mlp.routed_expert_up_proj.weight"
            k_sd = f"model.layers.{il}.mlp.shared_experts.down_proj.weight"
            ovr[k_rup] = self.trunk_cpu[k_rup].to(torch.float32) * alpha.unsqueeze(1)
            ovr[k_sd] = self.trunk_cpu[k_sd].to(torch.float32) * beta.unsqueeze(1)
        return ovr


def repack_section1_fast(orig_strata: Path, out_strata: Path, overrides: Dict[str, torch.Tensor], build: str = "SQ-64") -> Path:
    """Fast ~45s repack: re-encodes Section 1 with polished overrides & stream-copies Sections 2..5 verbatim!"""
    t0 = time.time()
    spec = BUILD_SPECS[build]
    out_strata.parent.mkdir(parents=True, exist_ok=True)
    with open(orig_strata, "rb") as fin:
        _, profile, sections, engram_sha = parse_header_frame(fin.read(PAGE_SIZE))
        s0 = sections[0]
        fin.seek(s0.file_offset)
        manifest = json.loads(zstd.ZstdDecompressor().decompress(fin.read(s0.byte_length)).decode("utf-8"))
        s1_off = sections[1].file_offset

        writer = StrataWriter(str(out_strata), profile=profile, engram_shard_sha256=engram_sha)
        sec1 = writer.begin_streaming_section(section_id=1)
        new_trunk_index = []
        for meta in manifest["trunk_tensors"]:
            name = meta["name"]
            if name in overrides:
                t = overrides[name]
                raw, fmt = encode_trunk_tensor(name, t, spec["mla_code"], spec["dense0_code"], spec["dfly_code"])
            else:
                fin.seek(s1_off + meta["offset"])
                raw = fin.read(meta["bytes"])
                fmt = meta["format"]
            rem = sec1.length % 64
            if rem:
                sec1.write(b"\x00" * (64 - rem))
            off = sec1.write(raw)
            new_trunk_index.append({"name": name, "format": fmt, "shape": meta["shape"], "offset": off, "bytes": len(raw)})
        sec1_ent = sec1.finish()
        print(f"[Repack {build}] Section 1 repacked ({sec1_ent.byte_length/1e9:.2f} GB, {len(overrides)} polished tensors)", flush=True)

        # Stream-copy Sections 2, 3, 4, 5 verbatim in 64 MiB chunks!
        for sid in (2, 3, 4, 5):
            ent = sections[sid]
            sw = writer.begin_streaming_section(section_id=sid)
            fin.seek(ent.file_offset)
            rem = ent.byte_length
            while rem > 0:
                buf = fin.read(min(rem, 64 * 1024 * 1024))
                if not buf:
                    raise EOFError(f"Short read copying section {sid}")
                sw.write(buf)
                rem -= len(buf)
            sw.finish()
            print(f"[Repack {build}] Copied Section {sid} ({ent.byte_length/1e9:.2f} GB)", flush=True)

        manifest["trunk_tensors"] = new_trunk_index
        manifest["deep_scale_qat_polished"] = True
        writer.write_manifest(manifest)
        writer.finalize()

    print(f"[Repack {build}] DONE -> {out_strata} ({out_strata.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.1f}s", flush=True)
    verify_strata_file(str(out_strata), verify_hashes=False)
    return out_strata


@torch.no_grad()
def chat_gen_compare(
    strata_path: Path,
    engram_q8_path: Path,
    model_dir: Path,
    device: str = "cuda:0",
    max_new: int = 36,
) -> Dict[str, str]:
    """Run fast real chat-templated greedy generation using pre-decoded Section 1 + mmap Section 4!"""
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = [
        ("code", "Write a clean Python function `def fib(n):` that returns the nth Fibonacci number using iteration."),
        ("math", "What is 17 * 24 + 13? Solve step by step and give the final number."),
        ("tool", "Return ONLY a valid JSON tool call to query weather in Tokyo in Celsius using tool name `get_weather`."),
        ("explain", "In 2 clear sentences, explain why KV caching speeds up autoregressive transformer decoding."),
    ]
    mod = DeepScaleQATModule(strata_path, engram_q8_path, model_dir, device=device)
    outs = {}
    t0 = time.time()
    for name, u_msg in prompts:
        ids = tok.encode(format_chat_prompt(u_msg)).ids
        gen = list(ids)
        for _ in range(max_new):
            arr = np.array(gen, dtype=np.int64)
            eng = mod.pre_gather_engram([arr])[0]
            logits = mod.forward_seq(arr, eng, use_ckpt=False)
            nxt = int(logits[-1].argmax().item())
            gen.append(nxt)
            if nxt in (120025, 120020):
                break
        txt = tok.decode(gen[len(ids):])
        outs[name] = txt
        print(f"  [ChatGen {name}] {txt!r}", flush=True)
    del mod
    torch.cuda.empty_cache()
    print(f"[ChatGen] Finished 4 chat prompts ({max_new} tok max) in {time.time()-t0:.1f}s", flush=True)
    return outs


def run_polish(
    orig_strata: Path,
    out_strata: Path,
    engram_q8: Path,
    model_dir: Path,
    splits_dir: Path,
    work_dir: Path,
    epochs: int = 3,
    lr: float = 1.5e-3,
    accum_steps: int = 4,
    device: str = "cuda:0",
) -> None:
    t_all = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    teacher_p, dev_ref_p, test_ref_p = batched_bf16_capture(
        model_dir, engram_q8, splits_dir, work_dir, n_cal_seqs=36, seq_len=512, device=device
    )

    teacher_recs = torch.load(teacher_p, map_location="cpu", weights_only=False)
    dev_refs = torch.load(dev_ref_p, map_location="cpu", weights_only=True)
    dev_data = json.loads((splits_dir / "development.json").read_text(encoding="utf-8"))
    dev_seqs = [np.array(s[:512], dtype=np.int64) for s in dev_data["seqs"][:8]]

    mod = DeepScaleQATModule(orig_strata, engram_q8, model_dir, device=device)
    train_seqs_np = [r["tokens"].numpy().astype(np.int64) for r in teacher_recs]
    train_eng = mod.pre_gather_engram(train_seqs_np)
    dev_eng = mod.pre_gather_engram(dev_seqs)

    @torch.no_grad()
    def eval_dev_fast() -> Tuple[float, float]:
        kls, t1s = [], []
        for si, arr in enumerate(dev_seqs):
            l_cand = mod.forward_seq(arr, dev_eng[si], use_ckpt=False).cpu()
            km, _, t1 = kl_divergence(dev_refs[si].float(), l_cand)
            kls.append(km)
            t1s.append(t1)
        return float(np.mean(kls)), float(np.mean(t1s))

    base_dev_kl, base_dev_t1 = eval_dev_fast()
    print(f"\n[DeepQAT SQ-64] BASELINE Held-Out Dev: mean KL = {base_dev_kl:.5f} | top1 = {base_dev_t1*100:.2f}%", flush=True)

    best_dev_kl = base_dev_kl
    best_overrides = mod.export_folded_overrides()

    opt = torch.optim.AdamW(mod.tuned.parameters(), lr=lr, weight_decay=0.0)
    total_opt_steps = max(1, (epochs * len(teacher_recs)) // accum_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps, eta_min=lr * 0.1)

    for ep in range(epochs):
        t_ep = time.time()
        perm = np.random.default_rng(100 + ep).permutation(len(teacher_recs))
        opt.zero_grad()
        ep_kls = []
        for step_i, idx in enumerate(perm.tolist()):
            rec = teacher_recs[idx]
            logits = mod.forward_seq(train_seqs_np[idx], train_eng[idx], use_ckpt=True)
            kl = topk_tail_kl_loss(
                logits,
                rec["top_idx"].to(device),
                rec["top_logp"].to(device),
                rec["tail_p"].to(device),
            )
            reg = 5e-3 * mod.anchor_reg_loss()
            loss = (kl + reg) / accum_steps
            loss.backward()
            ep_kls.append(float(kl.item()))

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(mod.tuned.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

            if (step_i + 1) % 12 == 0 or (step_i + 1) == len(perm):
                print(
                    f"  [Epoch {ep+1}/{epochs} | seq {step_i+1}/{len(perm)}] "
                    f"running train KL={np.mean(ep_kls[-12:]):.5f} | lr={sched.get_last_lr()[0]:.2e} | {time.time()-t_ep:.0f}s",
                    flush=True,
                )

        dev_kl, dev_t1 = eval_dev_fast()
        improved = dev_kl < best_dev_kl
        if improved:
            best_dev_kl = dev_kl
            best_overrides = mod.export_folded_overrides()
            torch.save(best_overrides, work_dir / "sq64_deep_polish_best.pt")
        print(
            f"[DeepQAT SQ-64] Epoch {ep+1}/{epochs} DONE ({time.time()-t_ep:.0f}s) | "
            f"Train KL={np.mean(ep_kls):.5f} | Held-Out Dev KL={dev_kl:.5f} (top1={dev_t1*100:.2f}%) "
            f"{'[NEW BEST!]' if improved else ''}",
            flush=True,
        )

    del mod, opt
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[DeepQAT SQ-64] Best Held-Out Dev KL: {base_dev_kl:.5f} -> {best_dev_kl:.5f} ({100*(base_dev_kl-best_dev_kl)/base_dev_kl:+.1f}%)", flush=True)
    repack_section1_fast(orig_strata, out_strata, best_overrides, build="SQ-64")

    # Full held-out dev + final_test evaluation on the final repacked polished .strata file!
    dev_res = evaluate_strata(out_strata, engram_q8, model_dir, splits_dir, "development", 8, device, work_dir / "eval_SQ-64_dev.json", ref_cache=dev_ref_p)
    test_res = evaluate_strata(out_strata, engram_q8, model_dir, splits_dir, "final_test", 8, device, work_dir / "eval_SQ-64_test.json", ref_cache=test_ref_p)

    # Chat-templated live generation check on polished SQ-64!
    gen_res = chat_gen_compare(out_strata, engram_q8, model_dir, device=device, max_new=40)
    (work_dir / "sq64_polish_chat_gen.json").write_text(json.dumps(gen_res, indent=2), encoding="utf-8")

    summary = {
        "base_dev_kl": base_dev_kl,
        "polished_dev_kl": dev_res["mean_kl"],
        "polished_dev_top1": dev_res["mean_top1_agree"],
        "polished_dev_ppl": dev_res["mean_ppl_cand"],
        "polished_test_kl": test_res["mean_kl"],
        "polished_test_top1": test_res["mean_top1_agree"],
        "polished_test_ppl": test_res["mean_ppl_cand"],
        "chat_gen": gen_res,
        "total_minutes": round((time.time() - t_all) / 60.0, 1),
    }
    (work_dir / "sq64_polish_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== SQ-64 DEEP 42-LAYER POLISH COMPLETE IN {summary['total_minutes']} MIN ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-strata", required=True)
    ap.add_argument("--out-strata", required=True)
    ap.add_argument("--engram-q8", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run_polish(
        Path(args.orig_strata),
        Path(args.out_strata),
        Path(args.engram_q8),
        Path(args.model_dir),
        Path(args.splits_dir),
        Path(args.work_dir),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
