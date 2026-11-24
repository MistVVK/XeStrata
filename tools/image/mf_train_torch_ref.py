# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/mf_train_torch_ref.py - one Mage-Flow block in torch, autograd, against the engine's own block
(include/strata/image/mf_train.hpp's mf_block, src/image/mf_train_block.cpp).

    PY=... python tools/image/mf_train_torch_ref.py --probe build/free/xestrata-image-mf-train-probe \
        [--dir /tmp/mf-block]

The block is the inference block's math, written the way src/image/mf_dit.cpp runs it: each stream's LayerNorm (no
affine, eps 1e-6) with the time's shift and scale, q, k and v, each head RMS-normed with its weight and the image's
rows rotated pairwise, one attention over [text, image] with every row seeing every key, the output projection and the
gated residual, then the second norm and the tanh GELU MLP with its gated residual.  A product reads FP16 rows divided
by a power of two a row and sums in FP32, so the reference does the same rounding here: what is compared is the
gradient's math, not FP16's noise (the two meet at 1e-5 on this block).

The weights are random and the same on both sides; the block's own time modulation is a value, not a trained tensor
(docs/LORA_TRAIN.md, "The four models" leaves the modulations out of v1's target modules).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:            # CTest's mf_block_parity skips where torch is not the interpreter's own
    sys.stderr.write("mf_train_torch_ref: no torch in this interpreter (CTest skips on exit 77)\n")
    sys.exit(77)

EPS_LN = 1e-6
ROW_LIM = 16384.0
PROD = ("q", "k", "v", "o", "mlp1", "mlp2")


def rnd(g, *shape):
    return torch.randn(*shape, device="cuda", generator=g)


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """dit_rows' pow2_scale, per row: the smallest power of two that brings the row under FP16's reach."""
    two = torch.tensor(2.0, device=amax.device)
    r = torch.where(amax > ROW_LIM, torch.pow(two, torch.ceil(torch.log2(amax / ROW_LIM))), torch.ones_like(amax))
    return torch.nan_to_num(r, nan=1.0, posinf=1.0).clamp(min=1.0)


def rows(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A product's input rows as the engine stores them: FP16 with a power of two a row."""
    rs = pow2_scale(y.abs().amax(dim=-1))
    return (y / rs.unsqueeze(-1)).half(), rs


def prod(x16: torch.Tensor, rs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return prod_nb(x16, rs, w) + b


def prod_nb(x16: torch.Tensor, rs, w: torch.Tensor) -> torch.Tensor:
    """a product's rows without its bias (an adapter joins between the rows and the bias)"""
    y = x16.float() @ w.float().T
    return y * rs.unsqueeze(-1) if rs is not None else y


AD_RANK = 8
AD_SCALE = 2.0   # s * alpha / rank, folded (s 1, alpha 16, rank 8)


def make_adapters(w16: dict, C: int, F: int, g) -> dict:
    """one run covers both paths: the img stream's q, o and mlp1 take a plain LoRA, the txt stream's k and mlp2 a
    DoRA, the rest no adapter.  A and B are the widened FP16 bits the engine reads, as autograd's leaves."""
    ad = {}
    for st, meth, prods in (("img", "lora", ("q", "o", "mlp1")), ("txt", "dora", ("k", "mlp2"))):
        for p in prods:
            N, K = {"mlp1": (F, C), "mlp2": (C, F)}.get(p, (C, C))
            A = (rnd(g, AD_RANK, K) * 0.05).half().cuda()
            B = (rnd(g, N, AD_RANK) * 0.05).half().cuda()
            e = {"method": meth, "A": A.float().requires_grad_(True), "B": B.float().requires_grad_(True),
                 "A16": A, "B16": B, "scale": AD_SCALE}
            if meth == "dora":
                e["m"] = (1.0 + rnd(g, N).cuda() * 0.1).requires_grad_(True)
                e["norms"] = torch.linalg.vector_norm(w16[f"w_{st}_{p}"].float(), dim=1)
            ad[(st, p)] = e
    return ad


def adapt(yb: torch.Tensor, x16: torch.Tensor, rs, w: torch.Tensor, e, b: torch.Tensor) -> torch.Tensor:
    """a product's rows with its adapter (None: the product as always): the base rows, the delta through A and B
    with the engine's own FP16 rounding on h, and the bias last.  h reads the input itself (x16 times its rows'
    scales), as inference's adapter does."""
    if e is None:
        return yb + b
    h = x16.float() @ e["A"].T
    if rs is not None:
        h = h * rs.unsqueeze(-1)
    h16, hr = rows(h)
    delta = h16.float() @ e["B"].T * hr.unsqueeze(-1)
    if e["method"] == "dora":
        # PEFT's DoRA (tuners/lora/dora.py): the norm of the composed weight W + s BA, recomputed from the current
        # factors and detached (the norm receives no gradient); at strength 1 the base rows take c, the delta's s c
        with torch.no_grad():
            n = torch.linalg.vector_norm(w.float() + e["scale"] * (e["B"] @ e["A"]), dim=1)
        c = e["m"] / (n + 1.1920929e-07)
        return c.unsqueeze(0) * yb + (e["scale"] * c).unsqueeze(0) * delta + b
    return yb + e["scale"] * delta + b


def head_norm(x: torch.Tensor, w: torch.Tensor, hd: int) -> torch.Tensor:
    """Each head RMS-normed over its own channels with the file's [head_dim] weight (attn.norm_q.weight is that wide),
    the same way qk_rope_v16 in src/image/mf_dit.cpp reads it."""
    xr = x.reshape(-1, hd)
    r = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + EPS_LN)
    return ((xr * r) * w.float()).reshape(x.shape)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)          # one angle a row's pair
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


def attn(q16, k16, v16, heads: int, hd: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = q16.shape[0]
    Q, K, V = [t.float().reshape(T, heads, hd).transpose(0, 1) for t in (q16, k16, v16)]
    S = Q @ K.transpose(-1, -2) / hd**0.5
    P = torch.softmax(S, dim=-1)
    o = (P @ V).transpose(0, 1).reshape(T, heads * hd)
    return o.half(), o


# the engine's tape (include/strata/image/mf_train.hpp's MfTapeSlice): every stream's own stages in this order, the
# image's rows first, then the text's, then the joint rows the attention read and wrote and their log-sum-exp
TAPE_STREAM = ("ln1", "q", "k", "v", "o", "h1", "ln2", "m1", "gelu", "m2", "out",
               "dln1", "dq", "dk", "dv", "dyo", "dh1", "dln2", "dm1", "dgelu", "dm2")
TAPE_JOINT = ("jq", "jk", "jv", "jo", "jdo", "jdq", "jdk", "jdv", "lse")
TAPE_F = (7, 8, 18, 19)   # mlp1's own and the GELU rows, forward and back: F wide; every other slice is C wide


def block_forward(w, ad, rows_in, C, heads, hd, F, Ri, Rt, cos, sin):
    """The block, one stream at a time, with the joint attention between them, and the stages the engine's tape holds
    beside them (`rows_in` is (img, txt)).  A stage is what the engine *stored* - the FP16 rows a product read, widened
    back - so a stage that disagrees names the kernel to look at, not the whole block."""
    img, txt = rows_in
    mod_i, mod_t = w["mod_img"], w["mod_txt"]
    joint, stage, gcap = {}, {"img": [], "txt": []}, {}

    def grab(tag, x):
        """keep autograd's gradient into `x` under `tag`: the engine's tape holds the same tensor's gradient"""
        x.register_hook(lambda g, tag=tag: gcap.__setitem__(tag, g.detach()))
        return x
    for name, x, mod, nq, nk in (("txt", txt, mod_t, "nq_txt", "nk_txt"), ("img", img, mod_i, "nq_img", "nk_img")):
        ln = grab(f"{name}.ln", torch.nn.functional.layer_norm(x, (C,), eps=EPS_LN) * (1 + mod[C:2 * C]) + mod[0:C])
        l16, rs1 = rows(ln)
        wq, wk, wv = w[f"w_{name}_q"], w[f"w_{name}_k"], w[f"w_{name}_v"]
        q = grab(f"{name}.q", adapt(prod_nb(l16, rs1, wq), l16, rs1, wq, ad.get((name, "q")), w[f"b_{name}_q"]))
        k = grab(f"{name}.k", adapt(prod_nb(l16, rs1, wk), l16, rs1, wk, ad.get((name, "k")), w[f"b_{name}_k"]))
        v = grab(f"{name}.v", adapt(prod_nb(l16, rs1, wv), l16, rs1, wv, ad.get((name, "v")), w[f"b_{name}_v"]))
        R = x.shape[0]
        qn = head_norm(q, w[nq], hd).reshape(R, heads, hd)
        kn = head_norm(k, w[nk], hd).reshape(R, heads, hd)
        if name == "img":
            qn, kn = rotate(qn, cos, sin), rotate(kn, cos, sin)
        joint[name] = {"q": qn.reshape(R, C).half(), "k": kn.reshape(R, C).half(), "v": v, "x": x, "ln": ln,
                       "mod": mod, "rs": rs1, "l16": l16}
        stage[name] = [l16.float(), q, k, v]
    # the joint rows: the text's first, then the image's
    jq = grab("jq", torch.cat([joint["txt"]["q"], joint["img"]["q"]], 0))
    jk = grab("jk", torch.cat([joint["txt"]["k"], joint["img"]["k"]], 0))
    # v's one power of two over both streams (the engine's v_scale): the attention reads v / vsc, and the output
    # projection takes vsc back as its rows' scale
    vcat = torch.cat([joint["txt"]["v"], joint["img"]["v"]], 0)
    vsc = pow2_scale(vcat.detach().abs().amax())
    print(f"v: absmax {vcat.detach().abs().amax().item():.6g}  scale {vsc.item():g}", flush=True)
    jv = grab("jv", (vcat / vsc).half())
    o16x, _ = attn(jq, jk, jv, heads, hd)
    o16 = grab("jo", o16x)
    T = Ri + Rt
    Q, K = [x.float().reshape(T, heads, hd).transpose(0, 1) for x in (jq, jk)]
    S = Q @ K.transpose(-1, -2) / hd**0.5
    lse = torch.logsumexp(S, dim=-1).transpose(0, 1).contiguous()   # the engine's lse[g]: one value a row's head
    joint["lse"] = lse
    # the engine's joint block: the four FP16 row sets, the four gradients (filled after the backward), the statistics
    stage["joint"] = [jq.float(), jk.float(), jv.float(), o16.float(), None, None, None, None, lse]
    outs = {}
    for name, mod in (("img", mod_i), ("txt", mod_t)):
        R = joint[name]["x"].shape[0]
        off = 0 if name == "txt" else Rt
        oa = o16[off:off + R]
        ors = vsc.expand(R)
        y = grab(f"{name}.y", adapt(prod_nb(oa, ors, w[f"w_{name}_o"]), oa, ors, w[f"w_{name}_o"],
                                    ad.get((name, "o")), w[f"b_{name}_o"]))
        h1 = grab(f"{name}.h1", joint[name]["x"] + mod[2 * C:3 * C] * y)
        ln2 = grab(f"{name}.ln2", torch.nn.functional.layer_norm(h1, (C,), eps=EPS_LN) * (1 + mod[4 * C:5 * C])
                   + mod[3 * C:4 * C])
        l16, rs2 = rows(ln2)
        m1 = grab(f"{name}.m1", adapt(prod_nb(l16, rs2, w[f"w_{name}_mlp1"]), l16, rs2, w[f"w_{name}_mlp1"],
                                      ad.get((name, "mlp1")), w[f"b_{name}_mlp1"]))
        # the engine's tanh GELU, differentiated by torch at the reference's own m1: no engine tape value enters
        # the reference's backward (at --big-norm2, clear_bend keeps m1 off the bend where the two sides' m1 differ)
        g = grab(f"{name}.g", torch.nn.functional.gelu(m1, approximate="tanh"))
        g16, rs3 = rows(g)
        m2 = grab(f"{name}.m2", adapt(prod_nb(g16, rs3, w[f"w_{name}_mlp2"]), g16, rs3, w[f"w_{name}_mlp2"],
                                      ad.get((name, "mlp2")), w[f"b_{name}_mlp2"]))
        out = h1 + mod[5 * C:6 * C] * m2
        outs[name] = {"out": out, "h1": h1, "m1": m1, "ln2": ln2, "mod": mod, "oa": oa, "rs2": rs2, "rs3": rs3,
                      "m2": m2}
        stage[name] += [y, h1, l16.float(), m1, g16.float(), m2, out]
    ref_rs = {st: {"rs": joint[st]["rs"], "rs2": outs[st]["rs2"], "rs3": outs[st]["rs3"]} for st in ("img", "txt")}
    return outs, stage, gcap, ref_rs


# how far clear_bend keeps m1 from GELU's bend: about three times the largest m1 the engine and the reference put
# apart at --big-norm2 1e5 (9.4 on the B70, 5.9 on the RTX 4070, 2026-11-24)
BEND_MARGIN = 32.0


def clear_bend(w, ad, rows_in, C, heads, hd, F, Ri, Rt, cos, sin) -> None:
    """Moves each mlp1 bias the least that keeps every row's m1 BEND_MARGIN from GELU's bend.  With the second norm's
    rows past FP16's reach (--big-norm2) the engine's LayerNorm and torch's reduce in their own orders, some of the
    rows' FP16 values land one step apart (77 of the image's 8192), and m1 moves by several units; where m1 is that near
    0, GELU's slope moves with it and the gradients behind it part by 1e-2.  Fed the engine's own FP16 rows, the
    reference agrees within every tolerance on the B70 and the RTX 4070 (2026-11-24): the gap is the reduction order,
    not the engine.  Off the bend the slope is 0 or 1 on both sides, and the rows' scales are tested as before."""
    outs = block_forward(w, ad, rows_in, C, heads, hd, F, Ri, Rt, cos, sin)[0]
    for st in ("img", "txt"):
        b = w[f"b_{st}_mlp1"]
        zero = (b - outs[st]["m1"]).detach().cpu()   # the bias that puts each row's m1 at 0, [R][F]
        new = b.detach().cpu().clone()
        moved = 0
        for j in range(zero.shape[1]):
            z = zero[:, j]
            if (z - new[j]).abs().min() >= BEND_MARGIN:
                continue
            cand = torch.cat([z - 1.001 * BEND_MARGIN, z + 1.001 * BEND_MARGIN])
            cand = cand[torch.stack([(z - c).abs().min() >= BEND_MARGIN for c in cand])]
            new[j] = cand[(cand - new[j]).abs().argmin()]
            moved += 1
        with torch.no_grad():
            b.copy_(new.to(b.device))
        print(f"{st}: {moved} of mlp1's {zero.shape[1]} biases moved off GELU's bend", flush=True)


def safetensors_read(path: str, names) -> dict:
    """the named tensors of a safetensors file as FP32 device tensors (the bf16 the DiT's blocks store widened)"""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        out = {}
        for nm in names:
            info = head[nm]   # a safetensors header lists its tensors at the top level
            f.seek(8 + n + info["data_offsets"][0])
            raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
            dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[info["dtype"]]
            a = torch.from_numpy(np.frombuffer(raw, dtype=dt).copy())
            if dt is np.uint16:
                a = a.view(torch.bfloat16)
            out[nm] = a.reshape(list(info["shape"])).float().cuda()   # safetensors lists a shape outer dim first
    return out


def real_block(g, path: str, temb_scale: float) -> tuple[dict, dict, int, int, int, int]:
    """Block 0 of the shipped DiT: its widths, its six products and their biases, its head norms, and the modulations
    as `img_mod.1`/`txt_mod.1` produce them from a synthetic time embedding (the file holds the linear, not its
    output).  The weights are the bf16 the file carries, read as the FP16 bits the engine reads: bf16's eight mantissa
    bits fit FP16's, so the two sides' weights stay the same numbers."""
    p0 = "transformer_blocks.0."
    t = safetensors_read(path, [p0 + s for s in (
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.to_q.weight", "attn.to_q.bias", "attn.to_k.weight",
        "attn.to_k.bias", "attn.to_v.weight", "attn.to_v.bias", "attn.to_out.0.weight", "attn.to_out.0.bias",
        "attn.add_q_proj.weight", "attn.add_q_proj.bias", "attn.add_k_proj.weight", "attn.add_k_proj.bias",
        "attn.add_v_proj.weight", "attn.add_v_proj.bias", "attn.to_add_out.weight", "attn.to_add_out.bias",
        "img_mlp.net.0.proj.weight", "img_mlp.net.0.proj.bias", "img_mlp.net.2.weight", "img_mlp.net.2.bias",
        "txt_mlp.net.0.proj.weight", "txt_mlp.net.0.proj.bias", "txt_mlp.net.2.weight", "txt_mlp.net.2.bias",
        "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "img_mod.1.weight", "img_mod.1.bias", "txt_mod.1.weight", "txt_mod.1.bias")])
    hd = int(t[p0 + "attn.norm_q.weight"].shape[0])
    C = int(t[p0 + "attn.to_q.weight"].shape[0])
    F = int(t[p0 + "img_mlp.net.0.proj.weight"].shape[0])
    heads = C // hd
    # the widths have to hang together the way the DiT's block does, or the file's shapes were read the wrong way round
    if (t[p0 + "attn.to_q.weight"].shape != (C, C) or t[p0 + "img_mlp.net.0.proj.weight"].shape != (F, C)
            or t[p0 + "img_mlp.net.2.weight"].shape != (C, F) or t[p0 + "img_mod.1.weight"].shape != (6 * C, C)
            or C % heads):
        shapes = {k.split(".")[-3]: tuple(v.shape) for k, v in list(t.items())[:4]}
        raise SystemExit(f"{path}: not a Mage-Flow DiT block of C x C attention and a {F} wide MLP: {shapes}")
    src = {"img": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.net.0.proj", "img_mlp.net.2",
                   "attn.norm_q", "attn.norm_k", "img_mod.1"),
           "txt": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
                   "txt_mlp.net.0.proj", "txt_mlp.net.2", "attn.norm_added_q", "attn.norm_added_k", "txt_mod.1")}
    w, w16 = {}, {}
    for st, names in src.items():
        for prod, nm in zip(PROD, names[:6]):
            h = t[p0 + nm + ".weight"].half()
            w16[f"w_{st}_{prod}"] = h
            w[f"w_{st}_{prod}"] = h.float().requires_grad_(True)
            w[f"b_{st}_{prod}"] = t[p0 + nm + ".bias"].clone().requires_grad_(True)
        w[f"nq_{st}"] = t[p0 + names[6] + ".weight"].clone().requires_grad_(True)
        w[f"nk_{st}"] = t[p0 + names[7] + ".weight"].clone().requires_grad_(True)
        lin, bias = t[p0 + names[8] + ".weight"], t[p0 + names[8] + ".bias"]
        temb = (torch.randn(lin.shape[1], device="cuda", generator=g) * temb_scale).float()
        w[f"mod_{st}"] = lin @ temb + bias
    return w, w16, C, heads, hd, F


def compare_stages(d: Path, stage, ref_rs, w16, C, heads, F, Ri, Rt) -> tuple[int, int, list]:
    """The engine's tape against the reference's own stages, one line each (the layout is MfTapeSlice's: a stream's
    slices, the image's rows first, then the text's, then the joint rows, their gradients and the log-sum-exp).  The
    magnitudes of both sides ride along: a stage where one side is zero is a different finding from one where both are
    large and differ."""
    tape = load(d, "tape").reshape(-1)
    off, bad, lines = 0, 0, []

    def one(tag, got, want, n, walk=True):
        nonlocal bad, off
        off += n if walk else 0
        if want.numel() != got.numel():
            lines.append((tag, float("nan"), False, got.abs().max().item() if got.numel() else 0.0, 0.0))
            bad += 1
            return
        e = (got - want).abs().max().item() / max(want.abs().max().item(), 1e-12)
        ok = e <= 3e-3
        bad += 0 if ok else 1
        lines.append((tag, e, ok, got.abs().max().item(), want.abs().max().item()))

    def side(v, n):    # a stage the backward has not reached yet (a closed gate) compares as zeros
        return torch.zeros(n) if v is None else v.reshape(-1)

    for st, R in (("img", Ri), ("txt", Rt)):
        for i, nm in enumerate(TAPE_STREAM):
            n = R * (F if i in TAPE_F else C)
            one(f"{st}.{nm}", tape[off:off + n], side(stage[st][i], n), n)
    T = Ri + Rt
    for i, nm in enumerate(TAPE_JOINT):
        n = T * C if nm != "lse" else T * heads
        one(f"joint.{nm}", tape[off:off + n], side(stage["joint"][i], n), n)
    # the rows' power-of-two scales, in the engine's order: ln1's two sets, ln2's two, the GELU rows' two, each the
    # image's rows first.  A product that came out as its own bias is either a zero weight or a zero scale, and this
    # is the half of the pair the products' dump cannot separate.
    for key, nm in (("rs", "ln1"), ("rs2", "ln2"), ("rs3", "gelu")):
        for st in ("img", "txt"):
            src = ref_rs[st][key].reshape(-1)
            one(f"{nm}.rs.{st}", tape[off:off + src.numel()], src, src.numel())
    for st in ("img", "txt"):   # q's weight as the engine's device really holds it
        want = w16[f"w_{st}_q"].float().reshape(-1)
        one(f"rd.{st}_q", load(d, f"rd_{st}_q").float().reshape(-1), want, want.numel(), walk=False)
    return off, bad, lines


def make_weights(C, heads, hd, F, g) -> tuple[dict, dict]:
    """The block's tensors, twice over where a product's weight is involved: `w16` holds the FP16 bits the engine reads
    (`MfBlockW::w` is uint16_t), `w` the same values in FP32 as autograd's leaves.  One tensor cannot be both: a dump
    written as f32 read as f16 is not the same weight, it is its exponent and mantissa split in two."""
    w, w16 = {}, {}
    for name, n, k in (("q", C, C), ("k", C, C), ("v", C, C), ("o", C, C), ("mlp1", F, C), ("mlp2", C, F)):
        for st in ("img", "txt"):
            h = (rnd(g, n, k) * 0.05).half().cuda()
            w16[f"w_{st}_{name}"] = h
            w[f"w_{st}_{name}"] = h.float().requires_grad_(True)
            w[f"b_{st}_{name}"] = (rnd(g, n).cuda() * 0.01).requires_grad_(True)
    for st in ("img", "txt"):
        # the file's attn.norm_q / norm_k weight is head_dim wide and shared by the heads
        w[f"nq_{st}"] = (1.0 + rnd(g, hd) * 0.1).cuda().requires_grad_(True)
        w[f"nk_{st}"] = (1.0 + rnd(g, hd) * 0.1).cuda().requires_grad_(True)
        # the modulations are a value, not a trained tensor (v1 leaves them out of the target modules)
        w[f"mod_{st}"] = rnd(g, 6 * C).cuda() * 0.2
    return w, w16


def save(d: Path, name: str, t: torch.Tensor) -> None:
    a = t.detach().to("cpu")
    if a.dtype == torch.float16:
        # view, not cast: to(int16) would round 0.05 to 0 and write a weight of zeros
        a = a.view(torch.int16).numpy().astype("<u2")
        tt = "f16"
    else:
        a = a.float().numpy()
        tt = "f32"
    a.tofile(d / f"{name}.bin")
    (d / f"{name}.json").write_text(json.dumps({"type": tt, "shape": list(reversed(a.shape))}))


def strata_tape_elems(C, heads, hd, F, Ri, Rt) -> int:
    """the same count as mf_block_tape_elems: each stream's 21 slices (17 of them C wide, m1 and the GELU's rows and
    their gradients F wide), then the joint rows and their gradients, the log-sum-exp, and the three scale pairs."""
    return (Ri + Rt) * (17 * C + 4 * F) + (Ri + Rt) * (8 * C + heads) + 3 * (Ri + Rt)


def load(d: Path, name: str) -> torch.Tensor:
    """the engine's dump form back as a device tensor: the file lists its shape fastest dim first"""
    meta = json.loads((d / f"{name}.json").read_text())
    shape = list(reversed(meta["shape"]))
    if meta["type"] == "f16":
        raw = np.fromfile(d / f"{name}.bin", dtype=np.int16)
        return torch.from_numpy(raw).view(torch.float16).reshape(shape).cuda()
    return torch.from_numpy(np.fromfile(d / f"{name}.bin", dtype=np.float32)).reshape(shape).float().cuda()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dir", default="/tmp/mf-block")
    ap.add_argument("--C", type=int, default=128)
    # head_dim 128: the shared attention kernel (kernels::flash_attn) takes 128 and 256 only
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--Ri", type=int, default=64)
    ap.add_argument("--Rt", type=int, default=8)
    ap.add_argument("--skip-attn", action="store_true", help="zero the attention's gate: the block is norm + MLP only")
    ap.add_argument("--skip-mlp", action="store_true", help="zero the MLP's gate: the block is norm + attention only")
    ap.add_argument("--stages", action="store_true", help="print the engine's tape against the reference's stages")
    ap.add_argument("--real", default="", help="a Mage-Flow DiT safetensors: block 0's real widths, weights and biases")
    ap.add_argument("--grad-tol", type=float, default=1e-4,
                    help="tolerance of the block's own output and input gradients: a LayerNorm's backward is a "
                         "difference of the row's large terms, so a cut with rows past FP16's reach needs its own; "
                         "the attention is the shared FP16 kernel, whose output carries FP16's rounding of P")
    ap.add_argument("--adapter-grad-tol", type=float, default=1e-4,
                    help="tolerance of the input gradients when adapters are on: the adapter path adds one more "
                         "FP16 rounding (h and delta through split FP16) and DoRA's row factors multiply the "
                         "row error by up to 1 + scale, so the plain 3e-5 has no margin (dimg 3.0e-5, dtxt 3.9-4.7e-5 "
                         "observed); 1e-4 is 3.3x the plain tol and 2.1x the worst observed")
    ap.add_argument("--temb-scale", type=float, default=1.0,
                    help="with --real: how large the time embedding that drives the block's modulations is")
    ap.add_argument("--big-norm2", type=float, default=0.0,
                    help="multiply the MLP's shift and scale by this: rows past dit_rows' kRowLim, so the products' "
                         "power-of-two scales and the wgrad's one global scale do real work")
    ap.add_argument("--big-norm1", type=float, default=0.0,
                    help="multiply the first norm's shift and scale by this: q, k and v read rows past kRowLim (their "
                         "adapters' rsx above 1) and v goes past FP16's 65504 (the block's v_scale), as Mage-Flow's "
                         "own block does at a high sigma")
    a = ap.parse_args()
    torch.manual_seed(20261103)
    g = torch.Generator(device="cuda").manual_seed(20261103)
    Ri, Rt = a.Ri, a.Rt
    if a.real:   # block 0 of the shipped DiT: real widths, real weights, a synthetic time embedding for the modulations
        w, w16, C, heads, hd, F = real_block(g, a.real, a.temb_scale)
        print(f"real block 0: C {C}  heads {heads}  head_dim {hd}  F {F}  rows {Ri} + {Rt}", flush=True)
    else:
        C, heads, hd, F = a.C, a.heads, a.C // a.heads, a.F
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    if not a.real:
        w, w16 = make_weights(C, heads, hd, F, g)   # synthetic, the same values on both sides
    img = (rnd(g, Ri, C) * 1.7).cuda()
    txt = (rnd(g, Rt, C) * 1.3).cuda()
    half = hd // 2
    ang = (rnd(g, Ri, half) * 2.0).cuda()
    cos, sin = torch.cos(ang), torch.sin(ang)
    for t in (img, txt, cos, sin):
        t.requires_grad_(False)
    imgg, txtg = img.clone().requires_grad_(True), txt.clone().requires_grad_(True)
    stage_bad = 0
    if a.big_norm2:
        # the second norm's shift and scale: its rows go past FP16's reach, so ln_mod16 has to divide by a power
        # of two and put it back, and the wgrad behind it folds one global power of two
        for st in ("img", "txt"):
            m = w[f"mod_{st}"]
            m[3 * C:5 * C] *= a.big_norm2
    if a.big_norm1:
        for st in ("img", "txt"):
            m = w[f"mod_{st}"]
            m[0:2 * C] *= a.big_norm1
    if a.skip_attn or a.skip_mlp:   # one gated residual closed: the other half of the block alone
        C2 = 2 * C
        for st in ("img", "txt"):
            m = w[f"mod_{st}"]
            if a.skip_attn:
                m[C2:C2 + C].zero_()
            if a.skip_mlp:
                m[C2 + 3 * C:C2 + 4 * C].zero_()
    ad = make_adapters(w16, C, F, g)
    if a.big_norm2:
        clear_bend(w, ad, (imgg, txtg), C, heads, hd, F, Ri, Rt, cos, sin)
    outs, stage, gcap, ref_rs = block_forward(w, ad, (imgg, txtg), C, heads, hd, F, Ri, Rt, cos, sin)
    d_out_img = (rnd(g, Ri, C) * 0.9).cuda()
    d_out_txt = (rnd(g, Rt, C) * 0.9).cuda()
    torch.autograd.backward([outs["img"]["out"], outs["txt"]["out"]], [d_out_img, d_out_txt])

    # the dumps the engine reads
    save(d, "dims", torch.tensor([C, heads, hd, F, Ri, Rt], dtype=torch.float32))
    for name in ("img", "txt"):
        save(d, name, img if name == "img" else txt)
        save(d, f"{name}_mod", w[f"mod_{name}"])
        save(d, f"{name}_nq", w[f"nq_{name}"])
        save(d, f"{name}_nk", w[f"nk_{name}"])
        for p in PROD:
            # the products' weights go as the FP16 bits the engine reads; [N][K] and its [K][N] transposed view
            save(d, f"{name}_{p}", w16[f"w_{name}_{p}"])
            save(d, f"{name}t_{p}", w16[f"w_{name}_{p}"].detach().T.contiguous())
            save(d, f"{name}_b_{p}", w[f"b_{name}_{p}"])
            if (name, p) in ad:
                # the adapter the same way: the FP16 bits and their transposed views, the DoRA's rows as f32
                e = ad[(name, p)]
                save(d, f"{name}_ad_{p}_a", e["A16"])
                save(d, f"{name}_ad_{p}_at", e["A16"].detach().T.contiguous())
                save(d, f"{name}_ad_{p}_b", e["B16"])
                save(d, f"{name}_ad_{p}_bt", e["B16"].detach().T.contiguous())
                save(d, f"{name}_ad_{p}_cfg",
                     torch.tensor([1.0 if e["method"] == "dora" else 0.0, AD_RANK, e["scale"]]))
                if e["method"] == "dora":
                    save(d, f"{name}_ad_{p}_m", e["m"].detach())
                    save(d, f"{name}_ad_{p}_norms", e["norms"])  # engine reads these; no grad through them here
    save(d, "cos", cos)
    save(d, "sin", sin)
    save(d, "d_out_img", d_out_img)
    save(d, "d_out_txt", d_out_txt)
    out = subprocess.run([a.probe, "block", str(d)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"the engine's block failed to run: {out.stderr.strip()}", file=sys.stderr)
        return 1
    if a.stages:
        for name in ("img", "txt"):   # the backward's slices, in the engine's order
            stage[name] += [gcap[f"{name}.{s}"] for s in ("ln", "q", "k", "v", "y", "h1", "ln2", "m1", "g", "m2")]
        for i, nm in enumerate(("jo", "jq", "jk", "jv")):
            stage["joint"][4 + i] = gcap[nm]
        used, sbad, lines = compare_stages(d, stage, ref_rs, w16, C, heads, F, Ri, Rt)
        for nm, e, ok, g_, wv in lines:
            print(f"stage     {nm:14} rel {e:.2e} {'ok' if ok else 'FAIL'}"
                  f"  engine {g_:.4e}  reference {wv:.4e}", flush=True)
        want_n = strata_tape_elems(C, heads, hd, F, Ri, Rt)
        # the walk must land exactly on the count the engine's layout says: a mismatch means the two layouts moved
        # apart, and then every line above compared the wrong slices
        if used != want_n:
            print(f"stage     tape: read {used} of {want_n} floats - THE TWO LAYOUTS DISAGREE", flush=True)
            stage_bad += 1
        else:
            print(f"stage     tape: read {used} of {want_n} floats, {sbad} over tolerance", flush=True)
        stage_bad += sbad
    got = {
        "out_img": load(d, "out_img"), "out_txt": load(d, "out_txt"),
        "dimg": load(d, "dimg"), "dtxt": load(d, "dtxt"),
    }
    want = {"out_img": outs["img"]["out"].detach(), "out_txt": outs["txt"]["out"].detach(), "dimg": imgg.grad,
            "dtxt": txtg.grad}
    bad = 0
    for name, t in want.items():
        print(f"  {name}: engine {got[name].abs().max().item():.4e}  reference {t.abs().max().item():.4e}")
        tol = a.adapter_grad_tol if name in ("dimg", "dtxt") else a.grad_tol
        e = (got[name] - t).abs().max().item() / max(t.abs().max().item(), 1e-12)
        ok = e <= tol
        bad += 0 if ok else 1
        print(f"block     {name:9} rel {e:.2e} (tol {tol:.0e}) {'ok' if ok else 'FAIL'}", flush=True)
    for st in ("img", "txt"):
        for p in PROD:
            gname, wname = f"gdw_{st}_{p}", f"w_{st}_{p}"
            gotw = load(d, gname)
            wantw = w[wname].grad if w[wname].grad is not None else torch.zeros_like(gotw)
            gb = load(d, f"gdb_{st}_{p}")
            wb = w[f"b_{st}_{p}"].grad
            dn = load(d, f"gdnq_{st}") if p == "q" else (load(d, f"gdnk_{st}") if p == "k" else None)
            pairs = [(f"{st}_{p}.w", gotw, wantw), (f"{st}_{p}.b", gb, wb)]
            if p == "q":
                pairs.append((f"{st}.nq", dn, w[f"nq_{st}"].grad))
            if p == "k":
                pairs.append((f"{st}.nk", dn, w[f"nk_{st}"].grad))
            for tag, g_, wv in pairs:
                if wv is None:
                    wv = torch.zeros_like(g_)
                g_, wv = g_.reshape(-1), wv.reshape(-1)
                e = (g_ - wv).abs().max().item() / max(wv.abs().max().item(), 1e-12)
                ok = e <= 3e-3
                bad += 0 if ok else 1
                print(f"block     {tag:14} rel {e:.2e} (tol 3e-03) {'ok' if ok else 'FAIL'}"
                      f"  engine {g_.abs().max().item():.4e}  reference {wv.abs().max().item():.4e}", flush=True)
    bad += stage_bad      # a stage that disagrees is a failure of the same block, counted with its gradients
    for (st, p), e in ad.items():
        pairs = [(f"{st}_{p}.dA", load(d, f"gdad_{st}_{p}"), e["A"].grad),
                 (f"{st}_{p}.dB", load(d, f"gdb_{st}_{p}_ad"), e["B"].grad)]
        if e["method"] == "dora":
            pairs.append((f"{st}_{p}.dm", load(d, f"gdm_{st}_{p}"), e["m"].grad))
        for tag, g_, wv in pairs:
            g_, wv = g_.reshape(-1), wv.reshape(-1)
            e_ = (g_ - wv).abs().max().item() / max(wv.abs().max().item(), 1e-12)
            ok = e_ <= 3e-3
            bad += 0 if ok else 1
            print(f"block     {tag:14} rel {e_:.2e} (tol 3e-03) {'ok' if ok else 'FAIL'}"
                  f"  engine {g_.abs().max().item():.4e}  reference {wv.abs().max().item():.4e}", flush=True)
    print(f"mf_train_torch_ref: {bad} over tolerance")
    return min(bad, 1)          # a CTest exit code counts to 255: one failure is enough to say so


if __name__ == "__main__":
    sys.exit(main())
