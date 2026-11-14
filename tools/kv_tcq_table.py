# SPDX-FileCopyrightText: 2026 the YASAKANI authors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
"""tools/kv_tcq_table.py - the codebooks of the trellis-coded K/V cache (--kv tcq4 / tcq5, kv_tcq.hpp).

    python tools/kv_tcq_table.py init  --bits B                  a codebook from the normal quantiles (no training)
    python tools/kv_tcq_table.py train --bits B [--rows N]       trained on rows of normal values (numpy, the CPU)
    python tools/kv_tcq_table.py eval  --bits B [--table F] [--yasakani KVQP]   the error against q4_0 and INT8
    python tools/kv_tcq_table.py header --out H --bits4 F4 --bits5 F5    kv_tcq_tables.hpp

The trellis is YASAKANI's KV5 (kvq_fast.cuh): a state of 8 bits, B bits a step shifted in from the right
(s' = (s << B | f) & 255), chains of 64 values from state 0, one value a state (the codebook, int16 in 1/256 of the
row's scale), the row's scale its root mean square times the best of a few multipliers.  A codebook is trained the way
a trellis-coded quantizer's is: the Viterbi path of every training chain, then each state's value the mean of the
values its paths took, over and over.  The rows are of independent normal values: what the signed Hadamard rotation
makes of a head's K and V rows, divided by their root mean square.  `init`, `train` and `eval` write and read a
table as JSON ({"bits", "code": 256 ints, "mult": floats}); `--yasakani` reads the codebook and multipliers of
YASAKANI's kvq_u5.bin (KVQP v1: a 5-bit codebook, the same for K and V, and 9 multipliers).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

CHAIN = 64
ROW = 256
XI_MAX = 2048   # the encoder's clamp of a value in 1/256 of the scale (kvq_fast.cuh's)


def init_code(bits: int) -> list[int]:
    """The 2^B values of each context (the state's high 8 - B bits) at the normal quantiles, each context's shifted by
    its share of a step, so that the trellis has the values between the steps of one context in the next."""
    kq, ctx = 1 << bits, 1 << (8 - bits)
    nd = NormalDist()
    code = []
    for s in range(256):
        u, f = s >> bits, s & (kq - 1)
        p = (f + 0.5 + (u + 0.5) / ctx - 0.5) / kq
        code.append(int(round(256 * nd.inv_cdf(min(max(p, 1e-4), 1 - 1e-4)))))
    return code


def viterbi(x: np.ndarray, code: np.ndarray, bits: int) -> np.ndarray:
    """The states of the least-error path of each chain of x [N, 64] (ints, 1/256 of the scale) through the codebook
    (code [256] ints), ties to the lowest predecessor and the lowest final state, as the GPU's encoder."""
    n = x.shape[0]
    kq, uu = 1 << bits, 256 >> bits
    big = np.int64(1) << 40
    cost = np.full((n, 256), big, dtype=np.int64)
    cost[:, 0] = 0
    bp = np.empty((CHAIN, n, uu), dtype=np.uint8)
    u = np.arange(uu)
    c64 = code.astype(np.int64)
    for t in range(CHAIN):
        c3 = cost.reshape(n, kq, uu)          # c3[:, j, u]: predecessor j * uu + u of the states u * kq + f
        j = c3.argmin(axis=1)
        m = np.take_along_axis(c3, j[:, None, :], axis=1)[:, 0, :]
        bp[t] = (j * uu + u[None, :]).astype(np.uint8)
        d = x[:, t:t + 1].astype(np.int64) - c64[None, :]
        cost = np.repeat(m, kq, axis=1) + d * d
    st = cost.argmin(axis=1)
    out = np.empty((n, CHAIN), dtype=np.int64)
    rows = np.arange(n)
    for t in range(CHAIN - 1, -1, -1):
        out[:, t] = st
        st = bp[t, rows, st >> bits].astype(np.int64)
    return out


def to_ints(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Values r [N, 256] at scales s [N] as the encoder's ints: 1/256 of the scale, rounded, clamped."""
    return np.clip(np.rint(r / s[:, None] * 256.0), -XI_MAX, XI_MAX).astype(np.int64)


def f16(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float16).astype(np.float32)


def encode_rows(r: np.ndarray, code: list[int], mult: list[float], bits: int) -> np.ndarray:
    """Rows r [N, 256] coded and decoded: the scale of least error among the multipliers of the root mean square,
    rounded to FP16 as the encoder stores it (kvq_fast.cuh)."""
    c = np.asarray(code, dtype=np.int64)
    rms = np.sqrt((r.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)
    rms[rms == 0] = 1.0
    best_e = np.full(r.shape[0], np.inf)
    best = np.zeros_like(r, dtype=np.float32)
    for mu in mult:
        s = f16(rms * np.float32(mu))
        s[s == 0] = 1.0
        xi = to_ints(r, s)
        rec = np.empty_like(r, dtype=np.float32)
        for ch in range(ROW // CHAIN):
            st = viterbi(xi[:, ch * CHAIN:(ch + 1) * CHAIN], c, bits)
            rec[:, ch * CHAIN:(ch + 1) * CHAIN] = c[st].astype(np.float32) / 256.0 * s[:, None]
        e = ((r - rec) ** 2).sum(axis=1)
        take = e < best_e
        best_e[take] = e[take]
        best[take] = rec[take]
    return best


def q4_0_rows(r: np.ndarray) -> np.ndarray:
    """ggml's Q4_0 of rows r (kv_q4_parts.hpp): blocks of 32, d the signed value of largest magnitude over -8."""
    b = r.reshape(-1, 32)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(b.shape[0]), idx]
    d = f16(mx / -8.0)
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0)
    q = np.clip(np.trunc(b * inv[:, None] + 8.5), 0, 15)
    return ((q - 8) * d[:, None]).reshape(r.shape).astype(np.float32)


def int8_rows(r: np.ndarray) -> np.ndarray:
    """The INT8 cache (kv_q8.cpp): groups of 64, an FP16 scale of the largest magnitude over 127."""
    b = r.reshape(-1, 64)
    s = f16(np.abs(b).max(axis=1) / 127.0)
    q = np.clip(np.rint(b / np.where(s != 0, s, 1)[:, None]), -127, 127)
    return (q * s[:, None]).reshape(r.shape).astype(np.float32)


def rows(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, ROW)).astype(np.float32)


def train(bits: int, n_rows: int, iters: int, seed: int, code: list[int]) -> list[int]:
    """Lloyd's iterations on chains of normal rows divided by their root mean square (the multiplier 1)."""
    r = rows(n_rows, seed)
    r /= np.sqrt((r ** 2).mean(axis=1, keepdims=True))
    xi = to_ints(r, np.ones(n_rows, dtype=np.float32)).reshape(-1, CHAIN)
    c = np.asarray(code, dtype=np.int64)
    for it in range(iters):
        st = viterbi(xi, c, bits)
        err = float(((xi - c[st]) ** 2).mean()) / 65536.0
        sums = np.bincount(st.ravel(), weights=xi.ravel().astype(np.float64), minlength=256)
        cnt = np.bincount(st.ravel(), minlength=256)
        new = np.where(cnt > 0, np.rint(sums / np.maximum(cnt, 1)), c).astype(np.int64)
        print(f"iteration {it + 1}: mean squared error {err:.6f} (of a unit variance), "
              f"{int((cnt == 0).sum())} states unused", file=sys.stderr)
        if np.array_equal(new, c):
            break
        c = new
    return [int(v) for v in c]


def read_yasakani(path: str) -> dict:
    """YASAKANI's KVQP v1 (kvq_fast.hpp's tbf_parse_blob): the multipliers at the offset at byte 32, the tables'
    offsets at byte 36 (each: L 8, B, chain 64, F 8, an FP32 gain, 4 bytes, 256 int16), every group's 5 bits."""
    b = Path(path).read_bytes()
    if b[:4] != b"KVQP" or struct.unpack_from("<H", b, 4)[0] != 1:
        raise SystemExit(f"{path}: not a KVQP v1 profile")
    nm, nt = struct.unpack_from("<H", b, 24)[0], struct.unpack_from("<H", b, 22)[0]
    off_m, off_t = struct.unpack_from("<I", b, 32)[0], struct.unpack_from("<I", b, 36)[0]
    mult = [struct.unpack_from("<f", b, off_m + 4 * i)[0] for i in range(nm)]
    codes = []
    for t in range(nt):
        to = struct.unpack_from("<I", b, off_t + 4 * t)[0]
        if b[to] != 8 or b[to + 2] != CHAIN:
            raise SystemExit(f"{path}: table {t} is not a chain of 64 with 8-bit states")
        codes.append((b[to + 1], [struct.unpack_from("<h", b, to + 12 + 2 * i)[0] for i in range(256)]))
    if any(c != codes[0] for c in codes):
        raise SystemExit(f"{path}: the tables differ (one codebook for K and V expected)")
    return {"bits": codes[0][0], "code": codes[0][1], "mult": [round(m, 6) for m in mult]}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def nmse(r: np.ndarray, rec: np.ndarray) -> float:
    return float(((r - rec) ** 2).sum() / (r ** 2).sum())


def evaluate(tables: dict[str, dict], n_rows: int, seed: int) -> None:
    r = rows(n_rows, seed)
    out = [("q4_0 (4.5 bits)", nmse(r, q4_0_rows(r))), ("int8 (8.25 bits)", nmse(r, int8_rows(r)))]
    for name, t in tables.items():
        bits = 8 * (ROW * t["bits"] // 8 + 4) / ROW
        out.append((f"{name} ({bits:.3g} bits)", nmse(r, encode_rows(r, t["code"], t["mult"], t["bits"]))))
    for name, e in out:
        print(f"{name:32s} normalized squared error {e:.6f}  ({10 * math.log10(1 / e):.2f} dB)")


def emit(code: list[int]) -> str:
    lines = []
    for i in range(0, 256, 16):
        lines.append("    " + ", ".join(f"{v}" for v in code[i:i + 16]) + ",")
    return "\n".join(lines)


def header(out: str, t4: dict, t5: dict) -> None:
    def mults(m: list[float]) -> str:
        def lit(v: float) -> str:   # a float literal: 1 is written 1.0f, not 1f
            t = f"{v:.6g}"
            return (t if "." in t or "e" in t else t + ".0") + "f"
        return ", ".join(lit(v) for v in m)
    # REUSE-IgnoreStart
    text = [
        "// SPDX-FileCopyrightText: 2026 the YASAKANI authors",
        "// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors",
        "// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0",
        "// include/strata/kernels/kv_tcq_tables.hpp - the trellis-coded K/V cache's codebooks (kv_tcq.hpp),",
        "// written by tools/kv_tcq_table.py header: do not edit by hand.  A state's value in 1/256 of the row's",
        "// scale, and the multipliers of the row's root mean square the encoder tries for the scale.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace strata::kernels::tcq {",
        "",
        f"inline constexpr int16_t kCode4[256] = {{\n{emit(t4['code'])}\n}};",
        f"inline constexpr float kMult4[] = {{{mults(t4['mult'])}}};",
        f"inline constexpr int16_t kCode5[256] = {{\n{emit(t5['code'])}\n}};",
        f"inline constexpr float kMult5[] = {{{mults(t5['mult'])}}};",
    ]
    # REUSE-IgnoreEnd
    text += ["", "}  // namespace strata::kernels::tcq", ""]
    Path(out).write_text("\n".join(text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "train", "eval", "header"])
    ap.add_argument("--bits", type=int, choices=[4, 5], default=5)
    ap.add_argument("--rows", type=int, default=4096, help="rows of 256 normal values (train, eval)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mult", default="0.9,0.93,0.96,0.98,1.0,1.02,1.04,1.07,1.1",
                    help="the multipliers of the root mean square (init, train)")
    ap.add_argument("--table", action="append", default=[], help="a JSON table (eval; the start of train)")
    ap.add_argument("--yasakani", help="YASAKANI's kvq_u5.bin, compared by eval")
    ap.add_argument("--bits4"), ap.add_argument("--bits5"), ap.add_argument("--out")
    a = ap.parse_args()
    mult = [float(v) for v in a.mult.split(",")]
    if a.cmd == "init":
        print(json.dumps({"bits": a.bits, "code": init_code(a.bits), "mult": mult}))
    elif a.cmd == "train":
        start = load(a.table[0])["code"] if a.table else init_code(a.bits)
        print(json.dumps({"bits": a.bits, "code": train(a.bits, a.rows, a.iters, a.seed, start), "mult": mult}))
    elif a.cmd == "eval":
        tables = {Path(p).stem: load(p) for p in a.table}
        if a.yasakani:
            tables["yasakani"] = read_yasakani(a.yasakani)
        evaluate(tables, a.rows, a.seed + 1)   # rows other than the training's
    else:
        if not (a.out and a.bits4 and a.bits5):
            ap.error("header needs --out, --bits4 and --bits5")
        header(a.out, load(a.bits4), load(a.bits5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
