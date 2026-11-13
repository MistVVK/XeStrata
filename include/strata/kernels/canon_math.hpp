// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// include/strata/kernels/canon_math.hpp - the arithmetic an expert's rows take on the CPU and on every GPU alike.
//
// A verify window's expert runs on the GPU when it is in VRAM (or read over PCIe) and on the CPU when not, and which
// it is changes with the window (how many experts miss, the PCIe share): the two must give the same bits, or the
// greedy text depends on the drafting (#152).  Additions, multiplications and fused multiply-adds are exact IEEE
// operations on every CPU and GPU; a division, an exp and the hardware's reciprocal are not (each maker's precision
// differs).  So the expert path's SwiGLU and its activations' int8 scale are built from the first kind only, with
// the fused multiply-adds written out (a compiler that contracted `a * b + c` on one side and not on the other would
// round differently), and no value leaves the normal range (devices differ in how they flush subnormals).
//
// A row whose products are summed as integers (fx_*, below) takes no order of additions at all: what the CPU, a
// GPU's decode and a GPU's prompt products (the matrix engines' tiles, any split of K) add is exact, so any order
// gives the same bits.  The idea is YASAKANI's (github.com/yasakani-dev/yasakani, Apache-2.0: lnord.cu's XSUM rows,
// each block's scale rounded to a fixed-point integer and the int8 sums times it added in int64).
//
// Usable in SYCL device code and in plain host code (the CPU kernels are not SYCL translation units).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#else
#include <cmath>
#endif

// a * b + c is not fused into one rounding behind these functions' backs (clang contracts across statements for CUDA
// and HIP device code by default)
#if defined(__clang__)
#define STRATA_CANON_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define STRATA_CANON_NO_CONTRACT
#endif

namespace strata::kernels::canon {

inline float fma(float a, float b, float c) {
#if defined(SYCL_LANGUAGE_VERSION)
    return sycl::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

inline uint32_t bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}
inline float from_bits(uint32_t u) {
    float v;
    std::memcpy(&v, &u, sizeof v);
    return v;
}

/// 1 / v for a positive normal v whose reciprocal is normal too: a seed from the exponent bits, then three Newton
/// steps (relative error well below one ulp of the result; the same bits everywhere, not always the rounded quotient).
inline float rcp(float v) {
    STRATA_CANON_NO_CONTRACT
    float r = from_bits(0x7EF311C3u - bits(v));
    for (int i = 0; i < 3; ++i) {
        const float e = fma(-v, r, 1.0f);
        r = fma(r, e, r);
    }
    return r;
}

/// e^x for x in [-80, 80] (clamped): x = k ln2 + r, |r| <= ln2 / 2, a degree-6 polynomial in r, and k added to the
/// exponent bits.  Results stay normal (e^-80 = 1.8e-35).
inline float exp(float x) {
    STRATA_CANON_NO_CONTRACT
    x = x < -80.0f ? -80.0f : x > 80.0f ? 80.0f : x;
    const float kf = (x * 1.44269504f + 12582912.0f) - 12582912.0f;   // round(x / ln2), to nearest even
    float r = fma(-kf, 0.693145752f, x);                                // ln2's high bits: kf * hi is exact
    r = fma(-kf, 1.42860677e-06f, r);                                   // and its low bits
    float p = 1.0f / 720.0f;
    p = fma(p, r, 1.0f / 120.0f);
    p = fma(p, r, 1.0f / 24.0f);
    p = fma(p, r, 1.0f / 6.0f);
    p = fma(p, r, 0.5f);
    p = fma(p, r, 1.0f);
    p = fma(p, r, 1.0f);
    return from_bits(bits(p) + (uint32_t) ((int32_t) kf << 23));
}

/// The expert's SwiGLU: g / (1 + e^-g) * u.
inline float silu_mul(float g, float u) {
    STRATA_CANON_NO_CONTRACT
    return g * rcp(1.0f + exp(-g)) * u;
}

constexpr float kHalfMax = 65504.0f;

/// A q8_1 block's scale for the largest |x| `amax` of its 32 values, kept finite as an fp16 (#606).
inline float q8_scale(float amax) {
    STRATA_CANON_NO_CONTRACT
    const float d = amax * (1.0f / 127.0f);
    return d > kHalfMax ? kHalfMax : d;
}
/// The inverse of that scale; 0 (every value 0) for an all-zero block and below 1e-30, where rcp's seed and steps
/// would meet subnormals.
inline float q8_inv(float d) { return d >= 1e-30f ? rcp(d) : 0.0f; }
/// One value's int8: rounded half away from zero, held to [-127, 127].
inline int8_t q8_value(float x, float id) {
    STRATA_CANON_NO_CONTRACT
    const float t = x * id;
    const float r = t + (t < 0.0f ? -0.5f : 0.5f);
    const float c = r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r;
    return (int8_t) (int32_t) c;
}

// ---- the fixed-point sums
//
// A row of an i-quant's blocks of 256 against an activation with one fp16 scale a 256 (the gate/up rows):
//   y = K sum_b da_b dw_b I_b,  I_b = sum over the block's 32-value calls of (the call's integer scale) x (its dot)
// with da_b, dw_b fp16 values (the activation's and the weights' d) and K a power of two (the format's).  Each fp16
// value is m 2^e with an integer m below 2^11, so da_b dw_b = (ma mw) 2^(ea + ew) with ma mw below 2^22, exactly.  The
// row's sum takes one exponent, top = (the largest ea of the activation) + (the largest ew of the row): block b's
// product is the integer A_b = ma mw 2^(ea + ew - top + kFxHead), rounded half away from zero where the shift is to the
// right (a block far below the row's largest), below 2^30; A_b I_b (I_b below 2^26 for the i-quants' grids and scales)
// is added in int64, and the sum times 2^(top - kFxHead) K is the row's value, one conversion to FP32 (nearest).
// Every step is an integer one but that last, so the same blocks give the same bits on the CPU and every GPU, in any
// order.

/// The head room of a block's integer scale over the row's largest product: A_b below 2^(22 + kFxHead).
constexpr int kFxHead = 8;
/// The exponent fx_exp gives 0: below every fp16 value's, so a zero never sets a row's top.
constexpr int kFxNone = -1000;

/// An fp16 value v held as a float, as |v| = m 2^e (2^10 <= m < 2^11; m 0 and e kFxNone for 0) and its sign.  An fp16
/// value is a normal float whose 13 low mantissa bits are 0 (fp16's subnormals too).
struct FxPart {
    uint32_t m;
    int e;
    uint32_t neg;   // 1: negative
};
inline FxPart fx_part(float v) {
    const uint32_t u = bits(v), ef = (u >> 23) & 0xFFu;
    if (ef == 0) return {0u, kFxNone, 0u};
    return {((u & 0x7FFFFFu) | 0x800000u) >> 13, (int) ef - 137, u >> 31};
}
/// e of an fp16 value (kFxNone for 0): a row's top is the largest of its activation's plus the largest of its weights'
inline int fx_exp(float v) { return fx_part(v).e; }

/// A block's integer scale A for the parts of a (the activation's d) and w (the weights' d) in a row of exponent `top`:
/// (ma mw) 2^(ea + ew - top + kFxHead), rounded half away from zero where the shift is to the right.
inline int32_t fx_scale(const FxPart& a, const FxPart& w, int top) {
    if (a.m == 0u || w.m == 0u) return 0;
    const uint32_t m = a.m * w.m;
    const int s = a.e + w.e - top + kFxHead;   // at most kFxHead where top is the row's
    uint32_t r;
    if (s >= 0) r = m << s;
    else if (s < -23) r = 0u;   // m below 2^22: rounds to 0
    else r = (m + (1u << (-s - 1))) >> -s;
    return (a.neg ^ w.neg) != 0u ? -(int32_t) r : (int32_t) r;
}
inline int32_t fx_scale(float a, float w, int top) { return fx_scale(fx_part(a), fx_part(w), top); }

/// 2^k for k in [-126, 127]
inline float pow2(int k) { return from_bits((uint32_t) (k + 127) << 23); }

/// A row's value from its int64 sum (sum of A_b I_b), its `top` and log2 of its format's K: the sum to FP32 (nearest),
/// then times 2^(top - kFxHead + k_log2) in two exact steps.  0 for a 0 sum (a row or an activation of zeros, whose
/// top is not a number's).
inline float fx_value(int64_t sum, int top, int k_log2) {
    STRATA_CANON_NO_CONTRACT
    if (sum == 0) return 0.0f;
    const int k = top - kFxHead + k_log2, h = k / 2;
    return (float) sum * pow2(h) * pow2(k - h);
}

}  // namespace strata::kernels::canon
