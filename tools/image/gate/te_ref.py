"""tools/image/gate/te_ref.py - TEST-ONLY (OWN-5 gate harness): a numpy reference of Qwen-Image-2.1's text encoder as
stable-diffusion.cpp 2988060 runs it for one prompt (src/conditioning/conditioner.hpp LLMEmbedder, VERSION_QWEN_IMAGE_2_1;
src/model/te/llm.hpp, LLMArch::QWEN3_VL, text only).

It is the oracle for our own GPU text encoder (step 3): the tokens, the per-layer activation ranges (what must fit the
f16 operands of the GEMMs), and the f32 condition our encoder is measured against besides sd.cpp's own (cond.bin of
run_set.py --dump).

What sd.cpp computes (read from the pinned tree, 11-11):
  prompt   = "<|im_start|>system\\nComprehend and analyze the provided prompt.<|im_end|>\\n" + "<|im_start|>user\\n"
             + (text or " ") + "<|im_end|>\\n<|im_start|>assistant\\n"
  tokens   = Qwen2 byte-level BPE (sd.cpp's own vocab: 256 bytes + merges in order + 31 special literals), the prompt
             split into [prefix | parse_prompt_attention(text) pieces | suffix], each piece encoded on its own;
             a piece's weight is its attention weight ("(word:1.2)" syntax), 1.0 for plain text
  model    = Qwen3-VL-8B text model: 36 layers, hidden 4096, 32 query heads / 8 KV heads of 128, FFN 12288 (SwiGLU),
             RMSNorm eps 1e-6 before attention and FFN, per-head RMSNorm on q and k (q_norm / k_norm), rope on q and k
             after the norms (ggml IMROPE, sections 24/20/20, base 5e6; with text-only positions t = h = w = i it is the
             plain NEOX rotation of halves), causal attention, scale 1/sqrt(128), no bias anywhere
  output   = the residual stream after layer 36 WITHOUT the final norm (out_layers = {36}), times the token weights
             (apply_token_weights; identity for plain text), minus the system tokens (prompt_template_encode_start_idx
             = the system prefix's token count)
  cond     = float32 [n_tokens - n_system, 4096]

Two arithmetic modes:
  "f32"    weights dequantized exactly to f32 (the GGUF block formats, f32 maths in ggml's order), every GEMM and the
           attention in float64 rounded to float32, the rest float32.  The "ideal" condition.
  "f16op"  what our GPU encoder computes: weights f32 -> f16 (round to nearest even), GEMM inputs rounded to f16,
           the products accumulated exactly (float64; the GPU accumulates in f32), Q / K / V rounded to f16, scores f32,
           probabilities rounded to f16 before P.V; the rest float32.

    python -I -B te_ref.py <te.gguf> --out-dir D [--prompts 0-7 | --prompt-file F] [--mode f32,f16op]
                           [--cmp NAME=/path/dump_p{p}_s42/cond.bin ...] [--stats]
        writes D/cond_p<p>_<mode>.npy (float32 [n, 4096]), D/tokens_p<p>.json, D/te_ref.json (comparisons, ranges)
    python -I -B te_ref.py <te.gguf> --tokens "some text"          the token ids sd.cpp would build, and n_system
    --tokens-from /path/dump_p{p}_s42   compare the ids with sd.cpp's own (stage_hooks.py tokens.bin / tokens.start)"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1]):          # tools/image/gate, tools (gguf_reader.py)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from gguf_reader import GGUFFile  # noqa: E402

SYSTEM = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
USER = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
# sd.cpp tokenizers/qwen2_tokenizer.cpp qwen2_special_tokens(): ids 151643 + index (after 256 bytes + 151387 merges)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
            "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
            "<think>", "</think>", "<|boi_token|>", "<|bor_token|>", "<|eor_token|>", "<|bot_token|>", "<|tms_token|>"]
QWEN2_SPLIT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
F16_MAX = 65504.0
F16_MIN_NORMAL = 6.103515625e-05
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


# ------------------------------------------------------------------ GGUF block formats -> f32 (ggml-quants.c order of operations)
def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<f2").astype(np.float32).reshape(b.shape[0])


def _scale_min_k4(s: np.ndarray):
    """get_scale_min_k4 for j = 0..7 over [nb, 12] scale bytes -> (sc [nb, 8], m [nb, 8]) as float32."""
    s = s.astype(np.int32)
    sc = np.empty((s.shape[0], 8), np.int32)
    mn = np.empty((s.shape[0], 8), np.int32)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def deq_q4_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 144)
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16])
    qs = b[:, 16:144].reshape(-1, 4, 32)
    dl = d[:, None] * sc                      # d1 = d * sc  (f32)
    ml = dmin[:, None] * mn                   # m1 = dmin * m
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    y = np.empty((b.shape[0], 4, 2, 32), np.float32)
    y[:, :, 0] = dl[:, 0::2, None] * lo - ml[:, 0::2, None]
    y[:, :, 1] = dl[:, 1::2, None] * hi - ml[:, 1::2, None]
    return y.reshape(-1)


def deq_q5_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 176)
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16])
    qh = b[:, 16:48]                          # [nb, 32]
    ql = b[:, 48:176].reshape(-1, 4, 32)
    dl = d[:, None] * sc
    ml = dmin[:, None] * mn
    y = np.empty((b.shape[0], 4, 2, 32), np.float32)
    for g in range(4):
        u1, u2 = 1 << (2 * g), 2 << (2 * g)
        lo = ((ql[:, g] & 0x0F) + np.where(qh & u1, 16, 0)).astype(np.float32)
        hi = ((ql[:, g] >> 4) + np.where(qh & u2, 16, 0)).astype(np.float32)
        y[:, g, 0] = dl[:, 2 * g, None] * lo - ml[:, 2 * g, None]
        y[:, g, 1] = dl[:, 2 * g + 1, None] * hi - ml[:, 2 * g + 1, None]
    return y.reshape(-1)


def deq_q6_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 210)
    ql, qh = b[:, 0:128].astype(np.int32), b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])
    y = np.empty((b.shape[0], 2, 4, 32), np.float32)
    for h in range(2):
        L, H = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32]
        q = [((L[:, 0:32] & 0xF) | (((H >> 0) & 3) << 4)) - 32, ((L[:, 32:64] & 0xF) | (((H >> 2) & 3) << 4)) - 32,
             ((L[:, 0:32] >> 4) | (((H >> 4) & 3) << 4)) - 32, ((L[:, 32:64] >> 4) | (((H >> 6) & 3) << 4)) - 32]
        for k in range(4):
            # y = d * (float)sc[is + 2k] * (float)q, is = l / 16
            s = np.repeat(sc[:, 8 * h + 2 * k:8 * h + 2 * k + 2], 16, axis=1)       # [nb, 32]
            y[:, h, k] = (d[:, None] * s) * q[k].astype(np.float32)
    return y.reshape(-1)


def deq_iq4_xs(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 136)
    d = _f16(b[:, 0:2])
    sh = b[:, 2:4].copy().view("<u2").astype(np.int32).reshape(-1)
    sl = b[:, 4:8].astype(np.int32)
    qs = b[:, 8:136].reshape(-1, 8, 16)
    y = np.empty((b.shape[0], 8, 2, 16), np.float32)
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32).astype(np.float32)
        y[:, ib, 0] = dl[:, None] * KV_IQ4NL[qs[:, ib] & 0x0F]
        y[:, ib, 1] = dl[:, None] * KV_IQ4NL[qs[:, ib] >> 4]
    return y.reshape(-1)


def deq_q8_0(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 34)
    return (_f16(b[:, 0:2])[:, None] * b[:, 2:34].view(np.int8).astype(np.float32)).reshape(-1)


DEQ = {"Q4_K": deq_q4_k, "Q5_K": deq_q5_k, "Q6_K": deq_q6_k, "IQ4_XS": deq_iq4_xs, "Q8_0": deq_q8_0,
       "F32": lambda r: r.view("<f4").astype(np.float32), "F16": lambda r: r.view("<f2").astype(np.float32),
       "BF16": lambda r: (r.view("<u2").astype(np.uint32) << 16).view(np.float32)}


class Weights:
    def __init__(self, path: Path):
        self.g = GGUFFile(path)
        self.t = {t.name: t for t in self.g.tensors}
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        md = self.g.metadata
        self.n_layer = int(md["qwen3vl.block_count"])
        self.n_embd = int(md["qwen3vl.embedding_length"])
        self.n_ff = int(md["qwen3vl.feed_forward_length"])
        self.n_head = int(md["qwen3vl.attention.head_count"])
        self.n_kv = int(md["qwen3vl.attention.head_count_kv"])
        self.hd = int(md["qwen3vl.attention.key_length"])
        self.theta = float(md["qwen3vl.rope.freq_base"])
        self.eps = 1e-6                                   # llm.hpp config.rms_norm_eps for QWEN3_VL (f32 1e-6f)

    def raw(self, name: str, row0: int = 0, rows: int | None = None) -> tuple[np.ndarray, list[int], str]:
        t = self.t[name]
        ne0 = t.shape[0]
        nrow = t.elements // ne0
        rb = t.expected_bytes() // nrow
        rows = nrow - row0 if rows is None else rows
        a = self.g.data_start + t.offset + row0 * rb
        return np.asarray(self.mm[a:a + rows * rb]), [ne0, rows], t.type_name

    def mat(self, name: str, row0: int = 0, rows: int | None = None) -> np.ndarray:
        """float32 [rows, ne0] (the GGUF's ne1 = output features, ne0 = input features)."""
        r, (ne0, n), ty = self.raw(name, row0, rows)
        return DEQ[ty](r).reshape(n, ne0)


# ------------------------------------------------------------------ tokens (sd.cpp's Qwen2Tokenizer + parse_prompt_attention)
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Qwen2Tok:
    def __init__(self, merges: list[str]):
        import regex
        bu = bytes_to_unicode()
        self.byte_enc = bu
        # sd.cpp load_from_merges: tokens = the 256 byte characters in bytes_to_unicode() ORDER, then merges, then specials
        order = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        order += [b for b in range(256) if b not in order]
        vocab = [bu[b] for b in order]
        pairs = [tuple(m.split(" ", 1)) for m in merges]
        vocab += [a + b for a, b in pairs]
        self.vocab = vocab
        self.enc = {}
        for i, tok in enumerate(vocab):
            self.enc[tok] = i                      # std::map assignment: a later duplicate wins
        for i, s in enumerate(SPECIALS):
            self.enc[s] = len(vocab) + i
        self.ranks = {p: r for r, p in enumerate(pairs)}
        self.pat = regex.compile(QWEN2_SPLIT)

    def bpe(self, word: str) -> list[str]:
        w = list(word)
        if len(w) < 2:
            return w
        while True:
            best, br = None, None
            for i in range(len(w) - 1):
                r = self.ranks.get((w[i], w[i + 1]))
                if r is not None and (br is None or r < br):
                    best, br = (w[i], w[i + 1]), r
            if best is None:
                break
            a, b = best
            nw, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(a + b)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            w = nw
            if len(w) == 1:
                break
        return w

    def encode(self, text: str) -> list[int]:
        out = []
        for piece in split_with_special_tokens(text, SPECIALS):
            if piece in SPECIALS:
                out.append(self.enc[piece])
                continue
            for m in self.pat.findall(piece):
                if not m:
                    continue
                u = "".join(self.byte_enc[b] for b in m.encode("utf-8"))
                out += [self.enc.get(s, 151643) for s in self.bpe(u)]
        return out


def split_with_special_tokens(text: str, specials: list[str]) -> list[str]:
    res, pos = [], 0
    while pos < len(text):
        nxt, tok = len(text), ""
        for s in specials:
            p = text.find(s, pos)
            if p != -1 and (p < nxt or (p == nxt and len(s) > len(tok))):
                nxt, tok = p, s
        if nxt > pos:
            res.append(text[pos:nxt])
        if tok:
            res.append(tok)
            pos = nxt + len(tok)
        else:
            break
    return res


_RE_ATT = re.compile(r"\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|\)|\]|\bBREAK\b|[^\\()\[\]:B]{1,1024}|:|\bB", re.ASCII)
_RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.ASCII)
SQ = np.float32(1) / np.float32(1.1)                 # square_bracket_multiplier = 1 / 1.1f (float division)


def parse_prompt_attention(text: str) -> list[list]:
    """core/util.cpp parse_prompt_attention (the A1111 syntax), weights as float32 products."""
    res, rb, sb = [], [], []
    f32 = np.float32

    def mul(start, m):
        for p in range(start, len(res)):
            res[p][1] = f32(res[p][1] * f32(m))

    rest = text
    while True:
        m = _RE_ATT.search(rest)
        if not m:
            break
        tok, suffix = m.group(0), rest[m.end():]
        if tok == ":":
            mw = re.match(r"([+-]?[0-9.]*)\)", suffix)
            if mw and mw.group(1) not in ("", "+", "-"):
                try:
                    v = float(mw.group(1))
                    if math.isfinite(v) and mw.group(1).count(".") <= 1 and mw.group(1) not in (".", "+.", "-."):
                        if rb:
                            mul(rb.pop(), v)
                        rest = suffix[mw.end():]
                        continue
                except ValueError:
                    pass
        if tok == "(":
            rb.append(len(res))
        elif tok == "[":
            sb.append(len(res))
        elif tok == ")" and rb:
            mul(rb.pop(), 1.1)
        elif tok == "]" and sb:
            mul(sb.pop(), SQ)
        elif tok == "\\(":
            res.append([tok[1:], f32(1.0)])
        elif _RE_BREAK.search(tok):
            res.append(["BREAK", f32(-1.0)])
        else:
            res.append([tok, f32(1.0)])
        rest = suffix
    for p in rb:
        mul(p, 1.1)
    for p in sb:
        mul(p, SQ)
    if not res:
        res.append(["", f32(1.0)])
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            del res[i + 1]
        else:
            i += 1
    return res


def build_tokens(tok: Qwen2Tok, text: str) -> tuple[list[int], list[float], int]:
    """LLMEmbedder::get_learned_condition (VERSION_QWEN_IMAGE_2_1, no reference images) + tokenize(): ids, weights,
    n_system (prompt_template_encode_start_idx)."""
    n_sys = len(tok.encode(SYSTEM))
    body = text if text else " "
    pieces = [[SYSTEM + USER, np.float32(1.0)]] + parse_prompt_attention(body) + [[SUFFIX, np.float32(1.0)]]
    ids, w = [], []
    for s, wt in pieces:
        t = tok.encode(s)
        ids += t
        w += [float(wt)] * len(t)
    return ids, w, n_sys


# ------------------------------------------------------------------ the model
def f16r(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    return ((x.astype(np.float64) / np.sqrt(ms + np.float64(np.float32(eps)))).astype(np.float32) * w).astype(np.float32)


def rope_neox(x: np.ndarray, theta: float) -> np.ndarray:
    """x [T, H, D]; NEOX: pairs (i, i + D/2), angle pos * theta^(-2i/D)."""
    T, H, D = x.shape
    i = np.arange(D // 2, dtype=np.float64)
    ang = np.arange(T, dtype=np.float64)[:, None] * theta ** (-2.0 * i / D)[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :D // 2].astype(np.float64), x[..., D // 2:].astype(np.float64)
    return np.concatenate([a * c - b * s, a * s + b * c], -1).astype(np.float32)


def gemm(x: np.ndarray, W64: np.ndarray, f16: bool) -> np.ndarray:
    xi = f16r(x) if f16 else x
    return (xi.astype(np.float64) @ W64.T).astype(np.float32)


def _rng(stats, key, x):
    if stats is None:
        return
    a = np.abs(x)
    s = stats.setdefault(key, {"max": 0.0, "n": 0, "over_f16": 0, "subnormal": 0})
    s["max"] = max(s["max"], float(a.max()))
    s["n"] += int(a.size)
    s["over_f16"] += int((a > F16_MAX).sum())
    s["subnormal"] += int(((a > 0) & (a < F16_MIN_NORMAL)).sum())


class Layer:
    def __init__(self, W: Weights, i: int, f16: bool):
        p = f"blk.{i}."
        conv = (lambda a: f16r(a).astype(np.float64)) if f16 else (lambda a: a.astype(np.float64))
        self.attn_norm = W.mat(p + "attn_norm.weight").reshape(-1)
        self.ffn_norm = W.mat(p + "ffn_norm.weight").reshape(-1)
        self.qn = W.mat(p + "attn_q_norm.weight").reshape(-1)
        self.kn = W.mat(p + "attn_k_norm.weight").reshape(-1)
        self.q, self.k, self.v = (conv(W.mat(p + f"attn_{n}.weight")) for n in "qkv")
        self.o = conv(W.mat(p + "attn_output.weight"))
        self.g, self.u, self.d = (conv(W.mat(p + f"ffn_{n}.weight")) for n in ("gate", "up", "down"))


def layer_forward(W: Weights, L: Layer, x: np.ndarray, f16: bool, stats: dict | None, li: int) -> np.ndarray:
    T = x.shape[0]
    H, KV, D = W.n_head, W.n_kv, W.hd
    h = rmsnorm(x, L.attn_norm, W.eps)
    _rng(stats, f"{li}.attn_in", h)
    q = gemm(h, L.q, f16).reshape(T, H, D)
    k = gemm(h, L.k, f16).reshape(T, KV, D)
    v = gemm(h, L.v, f16).reshape(T, KV, D)
    q = rope_neox(rmsnorm(q, L.qn, W.eps), W.theta)
    k = rope_neox(rmsnorm(k, L.kn, W.eps), W.theta)
    _rng(stats, f"{li}.q", q)
    _rng(stats, f"{li}.k", k)
    _rng(stats, f"{li}.v", v)
    if f16:
        q, k, v = f16r(q), f16r(k), f16r(v)
    scale = 1.0 / math.sqrt(D)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    out = np.empty((T, H, D), np.float32)
    rep = H // KV
    for hh in range(H):
        kv = hh // rep                                     # ggml broadcast: q head i uses kv head i / (H / KV)
        s = (q[:, hh].astype(np.float64) @ k[:, kv].astype(np.float64).T)
        s = (s.astype(np.float32) * np.float32(scale)).astype(np.float64) if f16 else s * scale
        s = s + mask
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(-1, keepdims=True)
        if f16:
            p = f16r(p.astype(np.float32)).astype(np.float64)
        out[:, hh] = (p @ v[:, kv].astype(np.float64)).astype(np.float32)
    a = out.reshape(T, H * D)
    _rng(stats, f"{li}.o_in", a)
    x = (x + gemm(a, L.o, f16)).astype(np.float32)
    h = rmsnorm(x, L.ffn_norm, W.eps)
    _rng(stats, f"{li}.ffn_in", h)
    g = gemm(h, L.g, f16)
    u = gemm(h, L.u, f16)
    a = ((g / (1.0 + np.exp(-g.astype(np.float64)))).astype(np.float32) * u).astype(np.float32)
    _rng(stats, f"{li}.down_in", a)
    x = (x + gemm(a, L.d, f16)).astype(np.float32)
    _rng(stats, f"{li}.resid", x)
    return x


def embed(W: Weights, ids: list[int]) -> np.ndarray:
    return np.stack([W.mat("token_embd.weight", i, 1)[0] for i in ids]).astype(np.float32)


def apply_token_weights(h: np.ndarray, w: list[float]) -> np.ndarray:
    wa = np.asarray(w, np.float32)
    if np.all(wa == 1.0):
        return h
    m0 = np.float32(h.astype(np.float64).mean())
    h = (h * wa[:, None]).astype(np.float32)
    m1 = np.float32(h.astype(np.float64).mean())
    return (h * np.float32(m0 / m1)).astype(np.float32)


# ------------------------------------------------------------------ comparison
def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).reshape(-1), b.astype(np.float64).reshape(-1)
    ma, mb = a.astype(np.float64), b.astype(np.float64)
    tc = (ma * mb).sum(1) / np.maximum(np.linalg.norm(ma, axis=1) * np.linalg.norm(mb, axis=1), 1e-30)
    return {"identical": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a64 - b64).max()),
            "rel_l2": float(np.linalg.norm(a64 - b64) / max(np.linalg.norm(b64), 1e-30)),
            "cos": float(a64 @ b64 / max(np.linalg.norm(a64) * np.linalg.norm(b64), 1e-30)),
            "token_cos_min": float(tc.min()), "token_cos_mean": float(tc.mean())}


def load_cond(path: str) -> np.ndarray:
    p = Path(path)
    shape = [int(x) for x in p.with_suffix(".shape").read_text().split()]
    return np.fromfile(p, dtype="<f4").reshape(shape[1], shape[0])


def prompt_list(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--out-dir")
    ap.add_argument("--prompts", default="0-7")
    ap.add_argument("--prompt-file", action="append", default=[])
    ap.add_argument("--mode", default="f32,f16op")
    ap.add_argument("--cmp", action="append", default=[], help="NAME=path with {p} for the prompt index")
    ap.add_argument("--tokens")
    ap.add_argument("--tokens-from", help="sd.cpp's stage dump dir with {p} (tokens.bin, tokens.start): ids must match")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--layers", type=int, default=0, help="debug: stop after this many layers")
    a = ap.parse_args(argv)
    W = Weights(Path(a.gguf))
    tok = Qwen2Tok(W.g.metadata["tokenizer.ggml.merges"])
    gv = W.g.metadata["tokenizer.ggml.tokens"]
    diff = sum(1 for x, y in zip(tok.vocab, gv) if x != y)
    spec = sum(1 for i, s in enumerate(SPECIALS) if len(gv) > len(tok.vocab) + i and gv[len(tok.vocab) + i] != s)
    print(f"vocab: sd.cpp construction {len(tok.vocab)} + {len(SPECIALS)} specials; differs from the GGUF's list at "
          f"{diff} regular and {spec} special ids", file=sys.stderr)
    if a.tokens is not None:
        ids, w, n_sys = build_tokens(tok, a.tokens)
        print(json.dumps({"ids": ids, "weights": w, "n_system": n_sys}))
        return 0
    from run_set import PROMPTS
    texts = {}
    if a.prompt_file:
        for i, f in enumerate(a.prompt_file):
            texts[i] = Path(f).read_text(encoding="utf-8")
    else:
        for i in prompt_list(a.prompts):
            texts[i] = PROMPTS[i]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = a.mode.split(",")
    toks = {}
    for i, t in texts.items():
        ids, w, n_sys = build_tokens(tok, t)
        toks[i] = (ids, w, n_sys)
        if a.tokens_from:
            d = Path(a.tokens_from.replace("{p}", str(i)))
            sd_ids = np.fromfile(d / "tokens.bin", dtype="<i4").tolist()
            sd_start = int((d / "tokens.start").read_text().split()[0])
            same = sd_ids == ids and sd_start == n_sys
            print(f"p{i}: TOKENS {'IDENTICAL' if same else 'DIFFER'} to sd.cpp ({len(sd_ids)} ids, start {sd_start})")
            if not same:
                return 3
        (out / f"tokens_p{i}.json").write_text(json.dumps({"text": t, "ids": ids, "weights": w, "n_system": n_sys}))
        print(f"p{i}: {len(ids)} tokens, n_system {n_sys}, cond rows {len(ids) - n_sys}", flush=True)
    xs = {(i, m): embed(W, toks[i][0]) for i in texts for m in modes}
    stats = {m: {} for m in modes} if a.stats else None
    nl = a.layers or W.n_layer
    t0 = time.time()
    for li in range(nl):
        for m in modes:
            L = Layer(W, li, m == "f16op")
            for i in texts:
                xs[(i, m)] = layer_forward(W, L, xs[(i, m)], m == "f16op", stats[m] if stats else None, li)
            del L
        print(f"layer {li} done {time.time() - t0:.0f} s", flush=True)
    res = {"prompts": {}, "modes": modes, "layers": nl}
    conds = {}
    for i in texts:
        ids, w, n_sys = toks[i]
        r = {"n_tokens": len(ids), "n_system": n_sys}
        for m in modes:
            c = apply_token_weights(xs[(i, m)], w)[n_sys:]
            conds[(i, m)] = c
            np.save(out / f"cond_p{i}_{m}.npy", c)
        if "f32" in modes and "f16op" in modes:
            r["f16op_vs_f32"] = compare(conds[(i, "f16op")], conds[(i, "f32")])
        for spec in a.cmp:
            name, path = spec.split("=", 1)
            pth = path.replace("{p}", str(i))
            if not Path(pth).is_file():
                continue
            ref = load_cond(pth)
            if ref.shape != conds[(i, modes[0])].shape:
                r[f"{name}_shape"] = [list(ref.shape), list(conds[(i, modes[0])].shape)]
                continue
            for m in modes:
                r[f"{name}_vs_{m}"] = compare(ref, conds[(i, m)])
        res["prompts"][str(i)] = r
        print(f"p{i}: " + json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
    # summaries per comparison
    keys = sorted({k for r in res["prompts"].values() for k, v in r.items() if isinstance(v, dict)})
    summ = {}
    for k in keys:
        rows = [r[k] for r in res["prompts"].values() if k in r]
        summ[k] = {"n": len(rows), "rel_l2_mean": float(np.mean([x["rel_l2"] for x in rows])),
                   "rel_l2_max": max(x["rel_l2"] for x in rows), "cos_min": min(x["cos"] for x in rows),
                   "token_cos_min": min(x["token_cos_min"] for x in rows), "max_abs": max(x["max_abs"] for x in rows),
                   "identical": sum(x["identical"] for x in rows)}
        print(f"SUMMARY {k}: {json.dumps(summ[k])}")
    res["summary"] = summ
    if stats:
        res["ranges"] = stats
        for m in modes:
            kinds = sorted({k.split(".", 1)[1] for k in stats[m]})
            for kd in kinds:
                vals = [(int(k.split(".")[0]), v) for k, v in stats[m].items() if k.split(".", 1)[1] == kd]
                top = max(vals, key=lambda t: t[1]["max"])
                sub = sum(v["subnormal"] for _, v in vals) / max(sum(v["n"] for _, v in vals), 1)
                over = sum(v["over_f16"] for _, v in vals)
                print(f"RANGE {m} {kd}: max {top[1]['max']:.4g} at layer {top[0]}, over f16 {over}, subnormal share {sub:.2e}")
    (out / "te_ref.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
