#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi_train_step_ref.py - one whole training step of the engine's Qwen-Image-2.1 LoRA trainer
(xestrata-image-qi-train check) against ai-toolkit's own DiT (QwenImage21Transformer2DModel, run as its
run_transformer runs it for a text-to-image sample) with torch's autograd on the CPU in FP32: the same base weights (the
F16 GGUF), the same LoRA factors on all 32 x 6 products, the same text states, latent, noise and timestep.  The loss,
the prediction and every LoRA factor's gradient are compared; with --bf16 the same model is also run in bf16 (what
ai-toolkit trains in) and each of the engine's errors must be no larger than bf16's.

The model in FP32 takes about 30 GB of RAM (and another 15 GB with --bf16).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit(77)
try:
    import gguf
except ImportError:   # an environment without it (the XPU torch's) reads the file through --index's table
    gguf = None

NBLOCKS = 32
ROUND_TEXT = False
INT8 = False
PRODUCTS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")


def aitk_transformer(root: Path):
    sys.path.insert(0, str(root))
    path = root / "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py"
    spec = importlib.util.spec_from_file_location("aitk_qwen_image_2_transformer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} cannot be read")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gguf_index(path: str) -> list:
    """name, type, shape (GGUF order) and the data's offset and bytes of each tensor"""
    return [[t.name, t.tensor_type.name, [int(x) for x in t.shape], int(t.data_offset), int(t.n_bytes)]
            for t in gguf.GGUFReader(path).tensors]


def gguf_state(path: str, index: str = "") -> dict:
    if gguf is not None:
        table = gguf_index(path)
    else:
        table = json.loads(Path(index).read_text())
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    sd = {}
    for name, typ, shape, off, nb in table:
        raw = np.asarray(mm[off:off + nb])
        if typ == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif typ == "F16":
            a = raw.view(np.float16).astype(np.float32)
        else:
            a = raw.view(np.float32)
        sd[name.removeprefix("model.diffusion_model.")] = torch.from_numpy(
            a.reshape(list(reversed(shape))).copy())
    return sd


def build(m, sd: dict, dtype):
    with torch.device("meta"):
        model = m.QwenImage21Transformer2DModel()
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, assign=True, strict=True)
    # the parts __init__ made as plain tensors and buffers, on the meta device: made again for real
    model.pos_embed = m.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    model.time_text_embed.time_proj = m.QwenImage21TemporalTimesteps(timestep_dim=256)
    model.requires_grad_(False)
    model.transformer_blocks = model.transformer_blocks[:NBLOCKS]
    if ROUND_TEXT:
        for lin in (model.txt_in.in_layer, model.txt_in.out_layer):
            lin.register_forward_pre_hook(lambda mod, inp: (inp[0].bfloat16().to(inp[0].dtype),))
    return model.eval()


def run(m, sd, dtype, factors, states, xt, x0, noise, t_model, h, w, device="cpu"):
    """ai-toolkit's DiT in `dtype` with the LoRAs (alpha = rank: added at scale 1 after each product): the loss, the
    prediction [h w][64] and the factors' gradients, in factors' order"""
    t0 = time.time()
    model = build(m, sd, dtype).to(device)
    model.pos_embed = m.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    # ai-toolkit's quantize with qtype convrot8 on the blocks' linears (its excluded modules stay as they are)
    if INT8:
        from toolkit.util.convrot_quant import get_convrot_quantizer
        q = get_convrot_quantizer("convrot8")
        for blk in model.transformer_blocks:
            for lin in (blk.attn.to_q, blk.attn.to_k, blk.attn.to_v, blk.attn.to_out[0], blk.img_mlp.gate_up,
                        blk.img_mlp.out):
                q.quantize_(lin, lin.weight.detach().float())
                lin.forward = (lambda x, lin=lin: q.forward(lin, x))
    print(f"  built the DiT in {time.time() - t0:.0f} s", flush=True)
    leaves = []
    for b, blk in enumerate(model.transformer_blocks):
        mods = {"attn.to_q": blk.attn.to_q, "attn.to_k": blk.attn.to_k, "attn.to_v": blk.attn.to_v,
                "attn.to_out.0": blk.attn.to_out[0], "img_mlp.gate_up": blk.img_mlp.gate_up,
                "img_mlp.out": blk.img_mlp.out}
        for name in PRODUCTS:
            A, B = (torch.from_numpy(f).to(device, dtype).requires_grad_(True) for f in factors[(b, name)])
            leaves += [A, B]
            mods[name].register_forward_hook(lambda mod, inp, out, A=A, B=B: out + (inp[0] @ A.T) @ B.T)
    n = states.shape[0]
    img_mask = torch.cat([torch.zeros(1, n, dtype=torch.bool), torch.ones(1, h * w // 4, dtype=torch.bool)],
                         dim=1).to(device)
    out = model(hidden_states=torch.from_numpy(xt)[None].to(device, dtype),
                encoder_hidden_states=torch.from_numpy(states)[None].to(device, dtype),
                timestep=torch.tensor([t_model], dtype=dtype, device=device), img_shapes=[[(1, h, w)]],
                img_mask=img_mask, return_dict=False)[0]
    print(f"  forward in {time.time() - t0:.0f} s", flush=True)
    pred = out[0, -h * w:].float()
    target = (torch.from_numpy(noise) - torch.from_numpy(x0)).to(device)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    print(f"  backward in {time.time() - t0:.0f} s", flush=True)
    grads = [leaf.grad.float().cpu().numpy() for leaf in leaves]
    out_loss, out_pred = float(loss.detach()), pred.detach().cpu().numpy()
    del model, leaves, out, pred, loss
    if device == "xpu":
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    return out_loss, out_pred, grads


def bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16().float().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="xestrata-image-qi-train")
    ap.add_argument("--dit", required=True, help="qwen-image-2.1-F16.gguf")
    ap.add_argument("--dir", default="/tmp/qi-step")
    ap.add_argument("--aitk", default=os.environ.get("AITK", str(Path.home() / ".mainplane/refs/ai-toolkit")))
    ap.add_argument("--text", type=int, default=24)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--timestep", type=float, default=713.25)
    ap.add_argument("--round-text", action="store_true",
                    help="round the text projection's inputs to bf16 in the reference too, as the engine's bf16 "
                         "products read them (its weights are bf16)")
    ap.add_argument("--states", default="", help="a states cache (mf_pre.py --cache qi's): real text rows, not noise")
    ap.add_argument("--latent", default="", help="a latent cache (mf_pre.py --cache qi's): x0 its posterior mean")
    ap.add_argument("--int8", action="store_true",
                    help="a QLoRA: ai-toolkit's convrot8 on the blocks' products (its ConvRotInt8Quantizer), the "
                         "engine's --base int8")
    ap.add_argument("--blocks", type=int, default=32, help="the DiT cut to its first blocks (both sides)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="the bar: the FP32 reference run again on inputs moved by this relative noise")
    ap.add_argument("--bf16", action="store_true", help="also run ai-toolkit's DiT in bf16, the bar the engine meets")
    ap.add_argument("--bf16-device", default="cpu", help="where the bf16 run goes (xpu: an Intel GPU's torch)")
    ap.add_argument("--index", default="", help="the DiT's tensor table (--make-index's) where gguf is not installed")
    ap.add_argument("--make-index", default="", help="write the DiT's tensor table to this file and stop")
    a = ap.parse_args()
    if a.make_index:
        Path(a.make_index).write_text(json.dumps(gguf_index(a.dit)))
        return 0
    global NBLOCKS, ROUND_TEXT, INT8
    NBLOCKS, ROUND_TEXT, INT8 = a.blocks, a.round_text, a.int8
    torch.set_num_threads(os.cpu_count() or 8)
    m = aitk_transformer(Path(a.aitk))
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(20261109)
    n, h, w, r = a.text, a.h, a.w, a.rank
    states = (g.standard_normal((n, 4096)) * 4).astype(np.float32)
    x0 = g.standard_normal((h * w, 64)).astype(np.float32)
    if a.states:
        v = np.fromfile(a.states, np.float32)
        n, states = int(v[0]), v[1:].reshape(int(v[0]), 4096)
    if a.latent:
        v = np.fromfile(a.latent, np.float32)
        w, h = int(v[0]), int(v[1])
        x0 = np.ascontiguousarray(v[2:2 + 64 * h * w].reshape(64, h * w).T)
    noise = g.standard_normal((h * w, 64)).astype(np.float32)
    t01 = np.float32(a.timestep) / np.float32(1000)
    xt = bf16((1 - t01) * x0 + t01 * noise)            # the DiT's input, in bf16 as ai-toolkit gives it
    t_model = float(bf16(bf16(np.array([a.timestep], np.float32)) / np.float32(1000))[0])
    t0 = time.time()
    sd = gguf_state(a.dit, a.index)
    print(f"read the DiT in {time.time() - t0:.0f} s", flush=True)
    shapes = {}
    for b in range(a.blocks):
        for name in PRODUCTS:
            N, K = sd[f"transformer_blocks.{b}.{name}.weight"].shape
            shapes[(b, name)] = (N, K)
    factors = {k: ((g.standard_normal((r, K)) / np.sqrt(K)).astype(np.float32),
                   (g.standard_normal((N, r)) * 0.01).astype(np.float32)) for k, (N, K) in shapes.items()}
    np.array([n, h, w, a.timestep, r], np.float32).tofile(d / "params.bin")
    states.tofile(d / "states.bin")
    x0.tofile(d / "x0.bin")
    noise.tofile(d / "noise.bin")
    np.concatenate([np.concatenate([A.ravel(), B.ravel()]) for A, B in factors.values()]).tofile(d / "factors.bin")
    t0 = time.time()
    loss, pred, grads = run(m, sd, torch.float32, factors, states, xt, x0, noise, t_model, h, w)
    print(f"ai-toolkit FP32: loss {loss:.6f} in {time.time() - t0:.0f} s", flush=True)
    bars = None
    if a.jitter > 0:
        t0 = time.time()
        jit = np.random.default_rng(7)

        def move(x):
            return (x * (1 + a.jitter * jit.standard_normal(x.shape))).astype(np.float32)
        bars = run(m, sd, torch.float32, factors, move(states), bf16(move(xt)), x0, noise, t_model, h, w)
        print(f"ai-toolkit FP32, inputs moved by {a.jitter:g}: loss {bars[0]:.6f} in {time.time() - t0:.0f} s",
              flush=True)
    if a.bf16:
        t0 = time.time()
        bars = run(m, sd, torch.bfloat16, factors, states, xt, x0, noise, t_model, h, w, a.bf16_device)
        print(f"ai-toolkit bf16: loss {bars[0]:.6f} in {time.time() - t0:.0f} s", flush=True)
    del sd
    # the engine's own runtime (XS_LD_LIBRARY_PATH: intel/llvm's libsycl) for its process only: an XPU torch in this
    # one does not take it
    env = dict(os.environ)
    if os.environ.get("XS_LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ["XS_LD_LIBRARY_PATH"]
    res = subprocess.run([a.exe, "check", "--dit", a.dit, "--dir", str(d), "--blocks", str(a.blocks), "--base",
                          "int8" if a.int8 else "bf16"], capture_output=True, text=True, env=env)
    print(res.stdout + res.stderr, end="")
    if res.returncode != 0:
        return 1
    e_loss = float(np.fromfile(d / "eng_loss.bin", np.float32)[0])
    e_pred = np.fromfile(d / "eng_pred.bin", np.float32).reshape(pred.shape)
    e_grads = np.fromfile(d / "eng_grads.bin", np.float32)

    def rel(x, y):
        return float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-30))

    bad = 0
    rows = [("loss", abs(e_loss - loss) / abs(loss), None if bars is None else abs(bars[0] - loss) / abs(loss)),
            ("pred", rel(e_pred, pred), None if bars is None else rel(bars[1], pred))]
    off, worst, worst_bar, cos_num, nrm_e, nrm_r = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    worst_at = ""
    keys = [(k, part) for k in factors for part in ("A", "B")]
    for i, ref in enumerate(grads):
        got = e_grads[off:off + ref.size].reshape(ref.shape)
        off += ref.size
        e = rel(got, ref)
        if e > worst:
            worst, worst_at = e, f"{keys[i][0][0]}.{keys[i][0][1]}.{keys[i][1]}"
        if bars is not None:
            worst_bar = max(worst_bar, rel(bars[2][i], ref))
        cos_num += float((got.astype(np.float64) * ref).sum())
        nrm_e += float((got.astype(np.float64) ** 2).sum())
        nrm_r += float((ref.astype(np.float64) ** 2).sum())
    rows.append(("grads (worst tensor)", worst, worst_bar if bars is not None else None))

    def l2rel(gs):   # the whole gradient's relative L2 error against the reference
        num = sum(float(((x.astype(np.float64) - y) ** 2).sum()) for x, y in zip(gs, grads))
        return float(np.sqrt(num / sum(float((y.astype(np.float64) ** 2).sum()) for y in grads)))
    eng_list, off2 = [], 0
    for ref in grads:
        eng_list.append(e_grads[off2:off2 + ref.size].reshape(ref.shape))
        off2 += ref.size
    rows.append(("grads (relative L2)", l2rel(eng_list), None if bars is None else l2rel(bars[2])))
    for name, e, eb in rows:
        # one tensor's worst value is reported, the whole gradient is judged: deep in a DiT the reference's own
        # gradient moves by more than itself at one tensor's largest value for an input moved by 1e-3
        ok = eb is None or e <= eb or name == "grads (worst tensor)"
        bad += 0 if ok else 1
        bar = "" if eb is None else f"  (the bar: {eb:.2e}) {'ok' if ok else 'FAIL'}"
        print(f"step  {name:22} rel {e:.2e}{bar}")
    print(f"step  worst gradient at {worst_at}; all gradients' cosine {cos_num / np.sqrt(nrm_e * nrm_r):.8f}")
    print(f"qi_train_step_ref: {bad} over the bar")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
