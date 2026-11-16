#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi_train_torch_ref.py - one Qwen-Image-2.1 block of the engine's training step (qi_block) against
ai-toolkit's own block (QwenImage21TransformerBlock, its QwenImage21Rope and its attention processor), in FP32 with
torch's autograd: the same weights (FP16 values), the same LoRAs on its six products, the same input, modulation and
output gradient.  The block's output, its input's gradient and every LoRA factor's gradient are compared, and each
passes where the engine is at least as close to the FP32 reference as ai-toolkit's own block run in bf16 (what its
training computes in) is: the engine's products read FP16 rows, finer than bf16's.

ai-toolkit's checkout is AITK (default ~/.mainplane/refs/ai-toolkit); the engine's side is the training probe's
`qiblock` command.  Exits 77 (skipped) without torch, CUDA or the checkout.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit(77)

PRODUCTS = ("q", "k", "v", "o", "gate_up", "down")


def save(d: Path, name: str, a: np.ndarray, half: bool = False) -> None:
    a = np.ascontiguousarray(a, dtype=np.float16 if half else np.float32)
    a.tofile(d / f"{name}.bin")
    (d / f"{name}.json").write_text(json.dumps({"type": "f16" if half else "f32", "shape": list(a.shape[::-1])}))


def load(d: Path, name: str, shape) -> np.ndarray:
    return np.fromfile(d / f"{name}.bin", dtype=np.float32).reshape(shape)


def aitk_transformer(root: Path):
    sys.path.insert(0, str(root))
    path = root / "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py"
    spec = importlib.util.spec_from_file_location("aitk_qwen_image_2_transformer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} cannot be read")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dir", default="/tmp/qi-block")
    ap.add_argument("--aitk", default=os.environ.get("AITK", str(Path.home() / ".mainplane/refs/ai-toolkit")))
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--text", type=int, default=20, help="text rows")
    ap.add_argument("--h", type=int, default=8, help="the image's rows of tokens")
    ap.add_argument("--w", type=int, default=6, help="the image's columns of tokens")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--mod-scale", type=float, default=0.5, help="how large the modulation's scales and gates are")
    ap.add_argument("--skip-attn", action="store_true", help="the attention's gate 0: the block is norm + MLP only")
    a = ap.parse_args()
    if not torch.cuda.is_available() or not Path(a.aitk).is_dir():
        print("qi_train_torch_ref: no CUDA device or no ai-toolkit checkout, skipped")
        return 77
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = aitk_transformer(Path(a.aitk))
    hd, H = 128, a.heads
    C, F = H * hd, 3 * H * hd
    nt, ni = a.text, a.h * a.w
    R = nt + ni
    g = torch.Generator().manual_seed(20261109)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).half().float()   # FP16 values: what the engine reads

    block = m.QwenImage21TransformerBlock(dim=C, num_attention_heads=H, attention_head_dim=hd, mlp_ratio=3)
    with torch.no_grad():
        for p in block.parameters():
            p.copy_(rnd(*p.shape, scale=1.0 / p.shape[-1] ** 0.5) if p.dim() == 2 else 1.0 + rnd(*p.shape, scale=0.1))
    state = {k: v.clone() for k, v in block.state_dict().items()}
    shapes = {"q": (C, C), "k": (C, C), "v": (C, C), "o": (C, C), "gate_up": (2 * F, C), "down": (C, F)}
    factors = {n: (rnd(a.rank, K, scale=1.0 / K ** 0.5), rnd(N, a.rank, scale=0.05)) for n, (N, K) in shapes.items()}
    x0 = rnd(1, R, C)
    modulation = rnd(2, 4 * C, scale=a.mod_scale)   # row 0 the step's t, row 1 t = 0 (causal_condition)
    if a.skip_attn:
        modulation[:, C:2 * C] = 0
    dout = rnd(1, R, C)
    mask = torch.zeros(R, dtype=torch.bool)
    mask[nt:] = True
    rope = m.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    rot = rope([(1, a.h, a.w)], mask, device=torch.device("cuda"))   # [R][hd / 2] complex

    def run(dtype):
        """ai-toolkit's block in `dtype` (FP32: the reference; bf16: what its training computes in) with the LoRAs
        added after each product (alpha = rank: scale 1): its output, the input's and the factors' gradients"""
        b = m.QwenImage21TransformerBlock(dim=C, num_attention_heads=H, attention_head_dim=hd, mlp_ratio=3)
        b.load_state_dict(state)
        b = b.cuda().to(dtype)
        lin = {"q": b.attn.to_q, "k": b.attn.to_k, "v": b.attn.to_v, "o": b.attn.to_out[0],
               "gate_up": b.img_mlp.gate_up, "down": b.img_mlp.out}
        ad = {}
        for n, layer in lin.items():
            A, B = (t.cuda().to(dtype).requires_grad_(True) for t in factors[n])
            ad[n] = (A, B)
            layer.register_forward_hook(lambda mod, inp, out, A=A, B=B: out + (inp[0] @ A.T) @ B.T)
        x = x0.cuda().to(dtype).requires_grad_(True)
        out = b(hidden_states=x, modulation=modulation.cuda().to(dtype), rotary_emb=rot, attention_mask=None,
                target_token_mask=mask.cuda(), segments=[(0, nt, True)], key_valid=None)
        (out.float() * dout.cuda()).sum().backward()
        res = {"out": out[0], "dx": x.grad[0]}
        for n, (A, B) in ad.items():
            res[f"{n}.dA"], res[f"{n}.dB"] = A.grad, B.grad
        return {k: v.detach().float().cpu().numpy() for k, v in res.items()}, b

    ref, block = run(torch.float32)
    bf, _ = run(torch.bfloat16)

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    save(d, "dims", np.array([C, H, hd, F, R, nt]))
    save(d, "x", x0.numpy()[0])
    save(d, "cos", rot.real.float().cpu().numpy())
    save(d, "sin", rot.imag.float().cpu().numpy())
    mo = modulation.numpy().reshape(2, 4, C)       # [row][scale1, gate1, scale2, gate2][C]
    eng = np.stack([mo[:, 0], np.tanh(mo[:, 1]), mo[:, 2], np.tanh(mo[:, 3])])   # [4][row][C], gates through tanh
    save(d, "mod", eng)
    save(d, "nq", block.attn.norm_q.weight.detach().cpu().numpy())
    save(d, "nk", block.attn.norm_k.weight.detach().cpu().numpy())
    save(d, "dout", dout.numpy()[0])
    for name in PRODUCTS:
        key = {"q": "attn.to_q", "k": "attn.to_k", "v": "attn.to_v", "o": "attn.to_out.0",
               "gate_up": "img_mlp.gate_up", "down": "img_mlp.out"}[name]
        save(d, f"w_{name}", state[key + ".weight"].numpy(), half=True)
        A, B = factors[name]
        save(d, f"ad_{name}_a", A.numpy(), half=True)
        save(d, f"ad_{name}_b", B.numpy(), half=True)
        save(d, f"ad_{name}_cfg", np.array([a.rank, 1.0]))
    r = subprocess.run([a.probe, "qiblock", str(d)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        print("qi_train_torch_ref: the engine's block failed")
        return 1
    bad = 0

    # the engine passes where it is at least as close to the FP32 reference as ai-toolkit's own bf16 training is
    def check(tag: str, got: np.ndarray) -> None:
        nonlocal bad
        w = ref[tag].reshape(got.shape)
        scale = max(float(np.abs(w).max()), 1e-12)
        e = float(np.abs(got - w).max()) / scale
        eb = float(np.abs(bf[tag].reshape(got.shape) - w).max()) / scale
        ok = e <= eb
        bad += 0 if ok else 1
        print(f"qiblock  {tag:12} rel {e:.2e}  (ai-toolkit in bf16: {eb:.2e}) {'ok' if ok else 'FAIL'}")

    check("out", load(d, "out", (R, C)))
    check("dx", load(d, "dx", (R, C)))
    for name in PRODUCTS:
        N, K = shapes[name]
        check(f"{name}.dA", load(d, f"ad_{name}_dA", (a.rank, K)))
        check(f"{name}.dB", load(d, f"ad_{name}_dB", (N, a.rank)))
    print(f"qi_train_torch_ref: {bad} over tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
