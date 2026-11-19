#!/usr/bin/env python3
"""Full 5-Layer DFly2 Speculative Drafter (KV-Share + Index-Share) for Tencent Hy4-Flash.

Implements model.dfly_layers.*:
  - mtp_fusion (enorm, hnorm, eh_proj [3072, 6144], gate [3072])
  - hidden_correction (embed_norm, hidden_norm, gate_proj/up_proj [3072, 6144], down_proj [3072, 3072])
  - 5 dense MLA + SwiGLU layers (dfly_layers.0..4) sharing target MLA KV & IndexPool from layers [25, 29, 33, 37, 41]
  - final_norm [3072] + confidence_head.proj [1, 3072] + bias [1]
  - Shares target Q8 lm_head.weight (0 B extra KV cache!)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hy_v4_flash import apply_interleaved_rope, rms_norm


class DFly2Drafter(nn.Module):
    """5-layer DFly2 drafter supporting BF16 evaluation and S4_K QAT co-training against quantized target."""

    def __init__(self, weights: Dict[str, torch.Tensor], rms_eps: float = 1e-5, rope_theta: float = 1e7):
        super().__init__()
        self.rms_eps = rms_eps
        self.rope_theta = rope_theta
        self.n_layers = 5
        self.n_head = 32
        self.qk_nope = 192
        self.qk_rope = 64
        self.qk_dim = 256
        self.v_dim = 256
        self.kv_lora = 512
        self.target_kv_layers = [25, 29, 33, 37, 41]
        self.w = nn.ParameterDict({k.replace(".", "__"): nn.Parameter(v.to(torch.float32), requires_grad=False) for k, v in weights.items()})

    def _get(self, key: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return self.w[key.replace(".", "__")].to(dtype)

    def forward_step(
        self,
        tok_emb: torch.Tensor,         # [T, 3072] next-token embeddings
        h_target: torch.Tensor,        # [T, 3072] target model final hidden state (after output_norm)
        target_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        lm_head: torch.Tensor,         # [V, 3072]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run full 5-layer DFly2 forward reusing target MLA KV & IndexPool selections at layers [25,29,33,37,41].
        Returns (draft_logits [T, V], draft_hidden [T, 3072], confidence [T]).
        """
        T = tok_emb.shape[0]
        device = tok_emb.device
        pos = torch.arange(T, device=device, dtype=torch.int64)

        # 1. mtp_fusion (slot0): fuse enorm(tok_emb) & hnorm(h_target) via eh_proj + residual gate
        en = rms_norm(tok_emb, self._get("model.dfly_layers.mtp_fusion.enorm.weight"), self.rms_eps)
        hn = rms_norm(h_target, self._get("model.dfly_layers.mtp_fusion.hnorm.weight"), self.rms_eps)
        fused = F.linear(torch.cat([en, hn], dim=-1), self._get("model.dfly_layers.mtp_fusion.eh_proj.weight"))
        gate = torch.sigmoid(self._get("model.dfly_layers.mtp_fusion.gate", torch.float32)).to(torch.bfloat16)
        cur = h_target * (1.0 - gate) + fused * gate

        # 2. hidden_correction SwiGLU block
        hc_in = torch.cat(
            [
                rms_norm(tok_emb, self._get("model.dfly_layers.hidden_correction.embed_norm.weight"), self.rms_eps),
                rms_norm(cur, self._get("model.dfly_layers.hidden_correction.hidden_norm.weight"), self.rms_eps),
            ],
            dim=-1,
        )
        hc_g = F.linear(hc_in, self._get("model.dfly_layers.hidden_correction.gate_proj.weight"))
        hc_u = F.linear(hc_in, self._get("model.dfly_layers.hidden_correction.up_proj.weight"))
        cur = cur + F.linear(F.silu(hc_g) * hc_u, self._get("model.dfly_layers.hidden_correction.down_proj.weight"))

        # 3. 5 Dense MLA + SwiGLU DFly2 layers sharing target KV & IndexPool masks
        for d in range(self.n_layers):
            p = f"model.dfly_layers.{d}"
            t_layer = self.target_kv_layers[d]
            res = cur
            h = rms_norm(cur, self._get(f"{p}.input_layernorm.weight"), self.rms_eps)
            pa = f"{p}.self_attn"
            qr = rms_norm(F.linear(h, self._get(f"{pa}.q_a_proj.weight")), self._get(f"{pa}.q_a_layernorm.weight"), self.rms_eps)
            q = F.linear(qr, self._get(f"{pa}.q_b_proj.weight")).reshape(T, self.n_head, self.qk_dim)
            q_nope, q_pe = q[..., : self.qk_nope], apply_interleaved_rope(q[..., self.qk_nope :], pos, self.rope_theta)

            # KV Share: reuse target sparse layer's (kv_cmpr, k_pe, idx_mask) if provided; else own projection
            if t_layer in target_kv:
                kv_cmpr, k_pe, idx_mask = target_kv[t_layer]
            else:
                kv_a = F.linear(h, self._get(f"{pa}.kv_a_proj_with_mqa.weight"))
                kv_cmpr = rms_norm(kv_a[:, : self.kv_lora], self._get(f"{pa}.kv_a_layernorm.weight"), self.rms_eps)
                k_pe = apply_interleaved_rope(kv_a[:, self.kv_lora :].unsqueeze(1), pos, self.rope_theta)
                idx_mask = None

            kv_exp = F.linear(kv_cmpr, self._get(f"{pa}.kv_b_proj.weight")).reshape(T, self.n_head, self.qk_nope + self.v_dim)
            k_nope, v = kv_exp[..., : self.qk_nope], kv_exp[..., self.qk_nope :]
            q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1)
            k_full = torch.cat([k_nope, k_pe.expand(T, self.n_head, self.qk_rope)], dim=-1).transpose(0, 1)
            scores = torch.matmul(q_full.float(), k_full.float().transpose(-2, -1)) / math.sqrt(self.qk_dim)
            if idx_mask is not None:
                scores = scores + idx_mask.unsqueeze(0)
            else:
                cm = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
                scores = scores + cm.unsqueeze(0)
            sinks = self._get(f"{pa}.learnable_sink_param", torch.float32).view(self.n_head, 1, 1).expand(self.n_head, T, 1)
            probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :-1].to(torch.bfloat16)
            attn = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(T, self.n_head * self.v_dim)
            attn = attn * torch.sigmoid(F.linear(h, self._get(f"{pa}.linear_gate.weight")).float()).to(torch.bfloat16)
            attn_out = rms_norm(F.linear(attn, self._get(f"{pa}.o_proj.weight")), self._get(f"{pa}.post_proj_norm.weight"), self.rms_eps)
            cur = res + attn_out

            res = cur
            h = rms_norm(cur, self._get(f"{p}.post_attention_layernorm.weight"), self.rms_eps)
            pm = f"{p}.mlp"
            mlp_out = F.linear(F.silu(F.linear(h, self._get(f"{pm}.gate_proj.weight"))) * F.linear(h, self._get(f"{pm}.up_proj.weight")), self._get(f"{pm}.down_proj.weight"))
            cur = res + mlp_out

        h_out = rms_norm(cur, self._get("model.dfly_layers.final_norm.weight"), self.rms_eps)
        conf_w = self._get("model.dfly_layers.confidence_head.proj.weight", torch.float32)
        conf_b = self._get("model.dfly_layers.confidence_head.proj.bias", torch.float32)
        conf = torch.sigmoid(F.linear(h_out.float(), conf_w, conf_b)).squeeze(-1)
        logits = F.linear(h_out.float(), lm_head.float())
        return logits, h_out, conf
