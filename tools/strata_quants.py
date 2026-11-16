"""tools/strata_quants.py - Strata's own numpy dequantizers and the few quantizers the pack tools need (replaces gguf-py's `quants`).

    dequantize(data, qtype) -> float32     data: uint8 array whose last axis holds whole rows of blocks (gguf-py's byte shape);
                                           F32 / F16 / BF16 are views and conversions of the raw values
    quantize(values, qtype) -> uint8       values: float array whose last axis is a multiple of the block size

Dequantize covers every type a GGUF file of ours can carry: BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, TQ1_0, TQ2_0, MXFP4,
NVFP4, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q2_0 (Strata's own 2-bit type). Quantize covers BF16, Q8_0, Q4_0,
Q4_1, Q5_0, Q5_1 (what the tools and tests use); anything else raises NotImplementedError.

The results are bit-identical to llama.cpp's gguf-py (the pin vendored in third_party/strata-gguf-py): every float32 operation is done in
the same order, which tools/test_strata_gguf.py proves against it. BLOCK below is the block geometry; tools/gen_quant_header.py checks the
engines' block structs (include/strata/quant/strata-quant-common.h) against it. The grids come from tools/strata_quant_tables.py.
"""
from __future__ import annotations

import numpy as np

import strata_quant_tables as T

# the block geometry (elements per block, bytes per block); strata_gguf.GGML_QUANT_SIZES is built from the same facts
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
         "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
         "TQ1_0": (256, 54), "TQ2_0": (256, 66), "MXFP4": (32, 17), "NVFP4": (64, 36),
         "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ3_XXS": (256, 98), "IQ3_S": (256, 110),
         "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "Q2_0": (64, 18)}

_CHUNK_ELEMS = 1 << 22        # blocks are processed in groups of about 4M values: the temporaries of the IQ types are 8x the output
f32 = np.float32


class QuantError(Exception):
    pass


def _name(qtype) -> str:
    return qtype.name if hasattr(qtype, "name") else str(qtype)


# ------------------------------------------------------------------------------------------------------------ small helpers
def _f16(b: np.ndarray, a: int) -> np.ndarray:
    """the fp16 at byte offset a of every block, as float32 (n, 1)"""
    return np.ascontiguousarray(b[:, a:a + 2]).view(np.float16).astype(f32)


def _nib(qs: np.ndarray) -> np.ndarray:
    """(n, k) bytes -> (n, 2k): all the low nibbles, then all the high nibbles (ggml's 4-bit order inside a block)"""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


def _bits(b: np.ndarray, width: int) -> np.ndarray:
    """(n, k) uint8 -> (n, k*8) bits, least significant first"""
    return ((b[:, :, None] >> np.arange(8, dtype=np.uint8)) & 1).reshape(b.shape[0], -1)


def _signs(bits: np.ndarray) -> np.ndarray:
    return np.where(bits == 0, f32(1), f32(-1))


def _u32(b: np.ndarray, a: int, n: int) -> np.ndarray:
    return np.ascontiguousarray(b[:, a:a + 4 * n]).view("<u4")


def _u16(b: np.ndarray, a: int, n: int) -> np.ndarray:
    return np.ascontiguousarray(b[:, a:a + 2 * n]).view("<u2")


_GRID = {}


def _grid(name: str) -> np.ndarray:
    """a grid table as float32 (entries, values per entry)"""
    if name not in _GRID:
        if name == "iq1s":
            g = np.frombuffer(T.IQ1S_GRID, dtype=np.int8).reshape(-1, 8).astype(f32)
        elif name in ("iq2xxs", "iq2xs", "iq2s"):
            g = np.frombuffer(getattr(T, name.upper() + "_GRID"), dtype=np.uint8).reshape(-1, 8).astype(f32)
        else:
            g = np.frombuffer(getattr(T, name.upper() + "_GRID"), dtype=np.uint8).reshape(-1, 4).astype(f32)
        _GRID[name] = g
    return _GRID[name]


# ------------------------------------------------------------------------------------------------------------ dequantizers (blocks -> float32)
def _bf16(b):
    return (np.ascontiguousarray(b).view("<i2").astype(np.int32) << 16).view(f32)


def _q4_0(b):
    d = _f16(b, 0)
    q = _nib(b[:, 2:]).astype(np.int8) - np.int8(8)
    return d * q.astype(f32)


def _q4_1(b):
    d, m = _f16(b, 0), _f16(b, 2)
    return (d * _nib(b[:, 4:]).astype(f32)) + m


def _q5_hi(b, a):
    """the fifth bit of the 32 values, from the 4 bytes at a, already shifted into place (n, 32)"""
    qh = _u32(b, a, 1)
    return (((qh >> np.arange(32, dtype=np.uint32)) & np.uint32(1)).astype(np.uint8)) << np.uint8(4)


def _q5_0(b):
    d = _f16(b, 0)
    q = (_nib(b[:, 6:]) | _q5_hi(b, 2)).astype(np.int8) - np.int8(16)
    return d * q.astype(f32)


def _q5_1(b):
    d, m = _f16(b, 0), _f16(b, 2)
    return (d * (_nib(b[:, 8:]) | _q5_hi(b, 4)).astype(f32)) + m


def _q8_0(b):
    return np.ascontiguousarray(b[:, 2:]).view(np.int8).astype(f32) * _f16(b, 0)


def _q2_k(b):
    n = b.shape[0]
    scales, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b, 80), _f16(b, 82)
    dl = (d * (scales & 0xF).astype(f32)).reshape(n, 16, 1)
    ml = (dmin * (scales >> 4).astype(f32)).reshape(n, 16, 1)
    q = ((qs.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & np.uint8(3))
    return (dl * q.reshape(n, 16, 16).astype(f32) - ml).reshape(n, 256)


def _q3_k(b):
    n = b.shape[0]
    hmask, qs, sc = b[:, :32], b[:, 32:96], b[:, 96:108]
    d = _f16(b, 108)
    lo = (sc[:, :8].reshape(n, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)).reshape(n, 16)
    hi = (sc[:, 8:].reshape(n, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)).reshape(n, 16)
    s = ((lo & np.uint8(0x0F)) | ((hi & np.uint8(3)) << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (d * s.astype(f32)).reshape(n, 16, 1)
    ql = (qs.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)).reshape(n, 16, 16) & np.uint8(3)
    qh = ((hmask.reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)).reshape(n, 16, 16) & np.uint8(1)) ^ np.uint8(1)
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(f32)
    return (dl * q).reshape(n, 256)


def _scale_min_k4(sc):
    """the 12 bytes of 6-bit scales and mins of Q4_K / Q5_K -> (scales, mins), each (n, 8)"""
    n = sc.shape[0]
    d, m, md = sc[:, 0:4], sc[:, 4:8], sc[:, 8:12]
    s = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=1)
    mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=1)
    return s.reshape(n, 8), mn.reshape(n, 8)


def _q4_k(b):
    n = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    s, m = _scale_min_k4(b[:, 4:16])
    dd = (d * s.astype(f32)).reshape(n, 8, 1)
    dm = (dmin * m.astype(f32)).reshape(n, 8, 1)
    q = ((b[:, 16:].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(n, 8, 32)
    return (dd * q.astype(f32) - dm).reshape(n, 256)


def _q5_k(b):
    n = b.shape[0]
    d, dmin = _f16(b, 0), _f16(b, 2)
    s, m = _scale_min_k4(b[:, 4:16])
    dd = (d * s.astype(f32)).reshape(n, 8, 1)
    dm = (dmin * m.astype(f32)).reshape(n, 8, 1)
    qh, qs = b[:, 16:48], b[:, 48:]
    ql = ((qs.reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(n, 8, 32)
    hb = ((qh.reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & np.uint8(1)).reshape(n, 8, 32)
    return (dd * (ql | (hb << np.uint8(4))).astype(f32) - dm).reshape(n, 256)


def _q6_k(b):
    n = b.shape[0]
    ql, qh, sc, d = b[:, :128], b[:, 128:192], b[:, 192:208], _f16(b, 208)
    dl = (d * np.ascontiguousarray(sc).view(np.int8).astype(f32)).reshape(n, 16, 1)
    lo = ((ql.reshape(n, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(n, 8, 32)
    hi = ((qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & np.uint8(3)).reshape(n, 8, 32)
    q = (lo | (hi << np.uint8(4))).astype(np.int8) - np.int8(32)
    return (dl * q.reshape(n, 16, 16).astype(f32)).reshape(n, 256)


_P3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)


def _tq1_0(b):
    n = b.shape[0]
    qs, qh, d = b[:, :48], b[:, 48:52], _f16(b, 52)
    a = (qs[:, :32].reshape(n, 1, 32) * _P3.reshape(1, 5, 1)).reshape(n, 160)           # uint8 wraps, as in the C code
    c = (qs[:, 32:].reshape(n, 1, 16) * _P3.reshape(1, 5, 1)).reshape(n, 80)
    e = (qh.reshape(n, 1, 4) * _P3[:4].reshape(1, 4, 1)).reshape(n, 16)
    q = np.concatenate([a, c, e], axis=1)
    q = ((q.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)
    return d * q.astype(f32)


def _tq2_0(b):
    n = b.shape[0]
    d = _f16(b, 64)
    q = ((b[:, :64].reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 256)
    return d * (q.astype(np.int8) - np.int8(1)).astype(f32)


_FP4 = np.frombuffer(T.KVALUES_FP4, dtype=np.int8)
_IQ4NL = np.frombuffer(T.KVALUES_IQ4NL, dtype=np.int8)


def _e8m0_half(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - np.uint32(1)) << np.uint32(23)).astype(np.uint32)
    return bits.view(f32)


def _mxfp4(b):
    d = _e8m0_half(b[:, :1])
    return d * _FP4[_nib(b[:, 1:])].astype(f32)


def _ue4m3_half(x: np.ndarray) -> np.ndarray:
    exp = (x >> 3).astype(np.int32) & 0xF
    man = (x & 0x7).astype(f32)
    raw = np.where(exp == 0, man * f32(2.0 ** -9), (f32(1.0) + man / f32(8.0)) * (f32(2.0) ** (exp.astype(f32) - f32(7))))
    return np.where((x == 0) | (x == 0x7F), f32(0.0), raw * f32(0.5)).astype(f32)


def _nvfp4(b):
    n = b.shape[0]
    d = _ue4m3_half(b[:, :4]).reshape(n, 4, 1)
    qs = b[:, 4:].reshape(n, 4, 8)
    v = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
    return (d * _FP4[v].astype(f32)).reshape(n, 64)


def _iq_signs_k(idx7: np.ndarray) -> np.ndarray:
    """7-bit sign indices -> (…, 8) float32 signs through ksigns_iq2xs"""
    ks = np.frombuffer(T.KSIGNS_IQ2XS, dtype=np.uint8)
    return _signs((ks[idx7][..., None] >> np.arange(8, dtype=np.uint8)) & np.uint8(1))


def _iq2_xxs(b):
    n = b.shape[0]
    d = _f16(b, 0)
    w = _u32(b, 2, 16).reshape(n, 8, 2)
    db = (d * (f32(0.5) + (w[..., 1] >> 28).astype(f32)) * f32(0.25)).reshape(n, 8, 1, 1)
    sidx = ((w[..., 1].reshape(n, 8, 1) >> np.array([0, 7, 14, 21], dtype=np.uint32)) & np.uint32(0x7F)).astype(np.intp)
    signs = _iq_signs_k(sidx).reshape(n, 8, 4, 8)
    gi = np.ascontiguousarray(w[..., 0]).view(np.uint8).reshape(n, 8, 4)
    grid = _grid("iq2xxs")[gi].reshape(n, 8, 4, 8)
    return (db * grid * signs).reshape(n, 256)


def _iq2_xs(b):
    n = b.shape[0]
    d = _f16(b, 0)
    q = _u16(b, 2, 32)
    sc = b[:, 66:74]
    s = ((sc.reshape(n, 8, 1) >> np.array([0, 4], dtype=np.uint8)) & 0x0F).reshape(n, 16)
    db = (d * (f32(0.5) + s) * f32(0.25)).reshape(n, 16, 1, 1)
    signs = _iq_signs_k((q >> 9).astype(np.intp)).reshape(n, 16, 2, 8)
    grid = _grid("iq2xs")[(q & np.uint16(511)).astype(np.intp)].reshape(n, 16, 2, 8)
    return (db * grid * signs).reshape(n, 256)


def _iq2_s(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs, sg, qh, sc = b[:, 2:34], b[:, 34:66], b[:, 66:74], b[:, 74:82]
    s = ((sc.reshape(n, 8, 1) >> np.array([0, 4], dtype=np.uint8)) & 0x0F).reshape(n, 16)
    db = (d * (f32(0.5) + s) * f32(0.25)).reshape(n, 16, 1, 1)
    signs = _signs(_bits(sg, 8)).reshape(n, 16, 2, 8)
    hi = ((qh.reshape(n, 8, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).astype(np.uint16).reshape(n, 32) << np.uint16(8)
    gi = (qs.astype(np.uint16) | hi).astype(np.intp)
    grid = _grid("iq2s")[gi].reshape(n, 16, 2, 8)
    return (db * grid * signs).reshape(n, 256)


def _iq3_xxs(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs = b[:, 2:66]
    w = _u32(b, 66, 8)
    db = (d * (f32(0.5) + (w >> 28).astype(f32)) * f32(0.5)).reshape(n, 8, 1, 1)
    sidx = ((w.reshape(n, 8, 1) >> np.array([0, 7, 14, 21], dtype=np.uint32)) & np.uint32(0x7F)).astype(np.intp)
    signs = _iq_signs_k(sidx).reshape(n, 8, 4, 8)
    grid = _grid("iq3xxs")[qs.astype(np.intp)].reshape(n, 8, 4, 8)
    return (db * grid * signs).reshape(n, 256)


def _iq3_s(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs, qh, sg, sc = b[:, 2:66], b[:, 66:74], b[:, 74:106], b[:, 106:110]
    s = ((sc.reshape(n, 4, 1) >> np.array([0, 4], dtype=np.uint8)) & 0x0F).reshape(n, 8)
    db = (d * (1 + 2 * s)).reshape(n, 8, 1, 1)
    signs = _signs(_bits(sg, 8)).reshape(n, 8, 4, 8)
    hi = _bits(qh, 8).astype(np.uint16) << np.uint16(8)
    gi = (qs.astype(np.uint16) | hi).astype(np.intp)
    grid = _grid("iq3s")[gi].reshape(n, 8, 4, 8)
    return (db * grid * signs).reshape(n, 256)


_DELTA = f32(0.125)


def _iq1_s(b):
    n = b.shape[0]
    d = _f16(b, 0)
    qs = b[:, 2:34]
    qh = _u16(b, 34, 8)
    dl = (d * (2 * ((qh >> 12) & 7) + 1)).reshape(n, 8, 1, 1)
    delta = np.where((qh & np.uint16(0x8000)) == 0, _DELTA, -_DELTA).reshape(n, 8, 1, 1)
    hi = ((qh.reshape(n, 8, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16)) & 7).reshape(n, 32) << np.uint16(8)
    gi = (qs.astype(np.uint16) | hi).astype(np.intp)
    grid = _grid("iq1s")[gi].reshape(n, 8, 4, 8)
    return (dl * (grid + delta)).reshape(n, 256)


def _iq1_m(b):
    n = b.shape[0]
    qs, qh = b[:, :32], b[:, 32:48]
    sc = _u16(b, 48, 4)
    dbits = ((sc & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16))
    dbits = (dbits[:, 0] | dbits[:, 1] | dbits[:, 2] | dbits[:, 3]).astype(np.uint16)
    d = dbits.view(np.float16).astype(f32).reshape(n, 1)
    s = ((sc.reshape(n, 4, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16)) & 7).reshape(n, 16)
    dl = (d * (2 * s + 1)).reshape(n, 8, 2, 1, 1)
    nib = (qh.reshape(n, 16, 1) >> np.array([0, 4], dtype=np.uint8)).reshape(n, 32)
    gi = (qs.astype(np.uint16) | ((nib & 7).astype(np.uint16) << np.uint16(8))).astype(np.intp)
    delta = np.where((nib & 0x08) == 0, _DELTA, -_DELTA).reshape(n, 8, 2, 2, 1)
    grid = _grid("iq1s")[gi].reshape(n, 8, 2, 2, 8)
    return (dl * (grid + delta)).reshape(n, 256)


def _iq4_nl(b):
    return _f16(b, 0) * _IQ4NL[_nib(b[:, 2:])].astype(f32)


def _iq4_xs(b):
    n = b.shape[0]
    d = _f16(b, 0)
    sh = _u16(b, 2, 1).reshape(n, 1)
    sl = b[:, 4:8]
    lo = ((sl.reshape(n, 4, 1) >> np.array([0, 4], dtype=np.uint8)) & np.uint8(0x0F)).reshape(n, 8)
    hi = ((sh >> (2 * np.arange(8, dtype=np.uint16))) & np.uint16(3)).astype(np.uint8)
    s = (lo | (hi << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (d * s.astype(f32)).reshape(n, 8, 1)
    qs = b[:, 8:].reshape(n, 8, 16)
    v = _IQ4NL[np.concatenate([qs & 0x0F, qs >> 4], axis=2)].astype(f32)
    return (dl * v).reshape(n, 256)


def _q2_0(b):
    d = _f16(b, 0)
    q = ((b[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(b.shape[0], 64)
    return (q.astype(f32) - f32(1.0)) * d


_DEQ = {"BF16": _bf16, "Q4_0": _q4_0, "Q4_1": _q4_1, "Q5_0": _q5_0, "Q5_1": _q5_1, "Q8_0": _q8_0, "Q2_K": _q2_k, "Q3_K": _q3_k,
        "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k, "TQ1_0": _tq1_0, "TQ2_0": _tq2_0, "MXFP4": _mxfp4, "NVFP4": _nvfp4,
        "IQ2_XXS": _iq2_xxs, "IQ2_XS": _iq2_xs, "IQ2_S": _iq2_s, "IQ3_XXS": _iq3_xxs, "IQ3_S": _iq3_s, "IQ1_S": _iq1_s,
        "IQ1_M": _iq1_m, "IQ4_NL": _iq4_nl, "IQ4_XS": _iq4_xs, "Q2_0": _q2_0}


def dequantize(data: np.ndarray, qtype) -> np.ndarray:
    name = _name(qtype)
    if name == "F32":
        return data.view(np.float32)
    if name == "F16":
        return data.view(np.float16).astype(np.float32)
    fn = _DEQ.get(name)
    if fn is None:
        raise NotImplementedError("Dequantization for %s is not yet implemented" % name)
    bs, ts = BLOCK[name]
    rows = np.asarray(data).view(np.uint8)
    shape = rows.shape
    if shape[-1] % ts:
        raise ValueError("Quantized tensor bytes per row (%d) is not a multiple of %s type size (%d)" % (shape[-1], name, ts))
    out_shape = (*shape[:-1], shape[-1] // ts * bs)
    blocks = rows.reshape(-1, ts)
    out = np.empty((blocks.shape[0], bs), dtype=np.float32)
    step = max(1, _CHUNK_ELEMS // bs)
    for i in range(0, blocks.shape[0], step):
        out[i:i + step] = fn(np.ascontiguousarray(blocks[i:i + step]))
    return out.reshape(out_shape)


# ------------------------------------------------------------------------------------------------------------ quantizers (float32 blocks -> bytes)
def _roundf(x: np.ndarray) -> np.ndarray:
    """round half away from zero, in float32 (ggml's roundf)"""
    a = np.abs(x)
    fl = np.floor(a)
    return np.sign(x) * (fl + np.floor(f32(2) * (a - fl)))


def _inv(d: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(d == 0, f32(0), f32(1) / d).astype(f32)


def _qbf16(x):
    n = np.ascontiguousarray(x).view(np.uint32)
    n = np.where((n & np.uint32(0x7fffffff)) > np.uint32(0x7f800000), (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)   # NaN -> quiet
    n = (n.astype(np.uint64) + (np.uint64(0x7fff) + ((n >> np.uint32(16)) & np.uint32(1)).astype(np.uint64))) >> np.uint64(16)  # nearest even
    return n.astype(np.uint16).view(np.uint8)


def _qq8_0(x):
    d = np.abs(x).max(axis=1, keepdims=True) / f32(127)
    q = _roundf(x * _inv(d))
    return np.concatenate([d.astype(np.float16).view(np.uint8), q.astype(np.int8).view(np.uint8)], axis=1)


def _pack_nib(q):
    n = q.shape[0]
    q = q.reshape(n, 2, 16)
    return (q[:, 0] & np.uint8(0x0F)) | (q[:, 1] << np.uint8(4))


def _trunc_u8(v, hi):
    return np.trunc(v).astype(np.uint8).clip(0, hi)


def _qq4_0(x):
    imax = np.abs(x).argmax(axis=1)[:, None]
    mx = np.take_along_axis(x, imax, axis=1)
    d = mx / f32(-8)
    q = _trunc_u8(x * _inv(d) + f32(8.5), 15)
    return np.concatenate([d.astype(np.float16).view(np.uint8), _pack_nib(q)], axis=1)


def _qq4_1(x):
    mx, mn = x.max(axis=1, keepdims=True), x.min(axis=1, keepdims=True)
    d = (mx - mn) / f32(15)
    q = _trunc_u8((x - mn) * _inv(d) + f32(0.5), 15)
    return np.concatenate([d.astype(np.float16).view(np.uint8), mn.astype(np.float16).view(np.uint8), _pack_nib(q)], axis=1)


def _qh(q):
    return np.packbits(q.reshape(q.shape[0], 1, 32) >> np.uint8(4), axis=-1, bitorder="little").reshape(q.shape[0], 4)


def _qq5_0(x):
    imax = np.abs(x).argmax(axis=1)[:, None]
    mx = np.take_along_axis(x, imax, axis=1)
    d = mx / f32(-16)
    q = _trunc_u8(x * _inv(d) + f32(16.5), 31)
    return np.concatenate([d.astype(np.float16).view(np.uint8), _qh(q), _pack_nib(q)], axis=1)


def _qq5_1(x):
    mx, mn = x.max(axis=1, keepdims=True), x.min(axis=1, keepdims=True)
    d = (mx - mn) / f32(31)
    q = _trunc_u8((x - mn) * _inv(d) + f32(0.5), 31)
    return np.concatenate([d.astype(np.float16).view(np.uint8), mn.astype(np.float16).view(np.uint8), _qh(q), _pack_nib(q)], axis=1)


_QUANT = {"BF16": _qbf16, "Q8_0": _qq8_0, "Q4_0": _qq4_0, "Q4_1": _qq4_1, "Q5_0": _qq5_0, "Q5_1": _qq5_1}


def quantize(data: np.ndarray, qtype) -> np.ndarray:
    name = _name(qtype)
    if name == "F32":
        return data.astype(np.float32, copy=False)
    if name == "F16":
        return data.astype(np.float16, copy=False)
    fn = _QUANT.get(name)
    if fn is None:
        raise NotImplementedError("Quantization for %s is not yet implemented" % name)
    bs, ts = BLOCK[name]
    x = np.asarray(data)
    if x.shape[-1] % bs:
        raise QuantError("Can't quantize tensor with shape %s to %s" % (x.shape, name))
    out_shape = (*x.shape[:-1], x.shape[-1] // bs * ts)
    blocks = x.astype(np.float32, copy=False).reshape(-1, bs)
    out = np.empty((blocks.shape[0], ts), dtype=np.uint8)
    step = max(1, _CHUNK_ELEMS // bs)
    for i in range(0, blocks.shape[0], step):
        part = fn(np.ascontiguousarray(blocks[i:i + step]))
        out[i:i + step] = part.reshape(-1, ts)
    return out.reshape(out_shape)
