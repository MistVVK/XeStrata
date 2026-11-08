// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/i8_gemm.cpp - see include/strata/kernels/cpu/i8_gemm.hpp: the packing that quantizes a BF16 or
// FP32 matrix to int8, the activation quantizer, the scalar reference kernel, the run-time ISA choice and the
// dispatch.  Compiled for the baseline x86-64, so everything here runs on any CPU; the AVX2 and AVX-VNNI kernels are
// in i8_gemm_vnni.cpp, their own translation unit.
//
// WHY THE SCALAR ROUTINES ARE THE DEFINITION HERE TOO, THOUGH THEY ARE SLOWER THAN THE LANE VERSIONS.  The int8
// path's answer is an integer sum per block plus a fixed chain of float operations, so the scalar and the vector
// copies can be held to the same bits exactly as bf16_gemm's two are - and the same rule applies: a bug in the lanes
// shows up as a difference on a machine that has them, not as a tolerance.  A CPU without AVX2 runs the scalar
// copies; XESTRATA_I8_GEMM_ISA=scalar or =avx2 caps the choice on a machine that has more.
#include "strata/kernels/cpu/i8_gemm.hpp"

#include "strata/kernels/bf16_bits.hpp"         // f32_from_bf16
#include "strata/kernels/cpu/expert_layout.hpp" // cpu_avx2_ok, cpu_avxvnni_ok
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

// The vector copies (i8_gemm_vnni.cpp) exist on x86-64 builds only; anything else runs the scalar loops, the way
// bf16_gemm.cpp keeps its AVX2 copy behind the same kind of guard.
#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_I8_GEMM_X86 1
#endif
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if !defined(STRATA_AVXVNNIINT8)
#define STRATA_AVXVNNIINT8 0
#endif

namespace strata::kernels::cpu {
namespace {

/// The level one value lands on: round-to-nearest-even (the mode every path here runs in), clamped to +/-`levels`.
/// `inv` is the block's levels / max|block|; a block of zeros keeps 0, and every level then lands on 0.
inline int i8_level(float v, float inv, int levels = kI8Levels) {
    int q = (int) std::nearbyintf(v * inv);
    if (q > levels) q = levels;
    if (q < -levels) q = -levels;
    return q;
}

/// A block's FP32 scale and the reciprocal its levels are made with.  Both come from the same maximum, so the
/// dequantization multiplies by exactly what the quantization divided by, up to the last bit of one rounding.
inline void i8_row_scale(float amax, float& scale, float& inv, int levels = kI8Levels) {
    scale = amax / (float) levels;
    inv = amax > 0.0f ? (float) levels / amax : 0.0f;
}

/// The k range of block `b` of a row of `n_in` (block 0: the whole row).
inline void block_range(int n_in, int block, int b, int& k0, int& k1) {
    k0 = block <= 0 ? 0 : b * block;
    k1 = block <= 0 ? n_in : std::min(n_in, k0 + block);
}

void i8_gemm_quantize_scalar(const float* x, size_t ldx, int n, int n_in, uint8_t* xq, size_t ldq, float* xs,
                             int block) {
    const int nb = i8_gemm_blocks(n_in, block);
    for (int t = 0; t < n; ++t) {
        const float* r = x + (size_t) t * ldx;
        uint8_t* q = xq + (size_t) t * ldq;
        for (int b = 0; b < nb; ++b) {
            int k0, k1;
            block_range(n_in, block, b, k0, k1);
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(r[k]));
            float scale = 0.0f, inv = 0.0f;
            i8_row_scale(amax, scale, inv);
            xs[(size_t) t * nb + b] = scale;
            for (int k = k0; k < k1; ++k) q[k] = (uint8_t) (i8_level(r[k], inv) + 128);   // vpdpbusd's unsigned side
        }
        for (int k = n_in; k < i8_gemm_act_row_bytes(n_in); ++k) q[k] = 0;    // the pad: its weight is 0 as well
    }
}

/// The definition of the product, in the order the header writes it.
void i8_gemm_rows_scalar(const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                         const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok, int p0,
                         int p1, const float* bias, int block) {
    const int quads = i8_gemm_quads(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    const int bq = block <= 0 ? quads : block / kI8Quad;   // quads a block
    for (int p = p0; p < p1; ++p) {
        const int base = p * kI8Panel;
        const int cols = std::min(kI8Panel, n_out - base);
        const uint8_t* wp = panel + (size_t) p * (size_t) quads * (size_t) kI8QuadBytes;
        const float* ws = wscale + (size_t) p * nb * kI8Panel;
        const int32_t* wc = wcorr + (size_t) p * nb * kI8Panel;
        for (int t = 0; t < n_tok; ++t) {
            const uint8_t* a = xq + (size_t) t * ldq;
            float f[kI8Panel] = {};
            for (int b = 0; b < nb; ++b) {
                int32_t acc[kI8Panel] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int kq = b * bq; kq < std::min(quads, (b + 1) * bq); ++kq) {
                    const uint8_t* w = wp + (size_t) kq * (size_t) kI8QuadBytes;
                    for (int j = 0; j < cols; ++j)
                        for (int qi = 0; qi < kI8Quad; ++qi)
                            acc[j] += (int32_t) a[kq * kI8Quad + qi] * (int32_t) (int8_t) w[j * kI8Quad + qi];
                }
                const float sa = xs[(size_t) t * nb + b];
                for (int j = 0; j < cols; ++j) {
                    const float d = (float) (acc[j] - wc[b * kI8Panel + j]);
                    const float s = sa * ws[b * kI8Panel + j];
                    if (b == 0) f[j] = bias != nullptr ? std::fmaf(s, d, bias[base + j]) : s * d;
                    else f[j] = std::fmaf(s, d, f[j]);
                }
            }
            float* yr = y + (size_t) t * ldy + base;
            for (int j = 0; j < cols; ++j) yr[j] = f[j];
        }
    }
}

/// One output row's levels into its panel lane `j`, and its scales and corrections, from `value(k)`.
template <class Value>
void pack_lane(Value value, bool real, int n_in, int quads, int nb, int block, int levels, uint8_t* dst, int j,
               float* ws, int32_t* wc) {
    for (int b = 0; b < nb; ++b) {
        int k0, k1;
        block_range(n_in, block, b, k0, k1);
        float amax = 0.0f;
        if (real)
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(value(k)));
        float scale = 0.0f, inv = 0.0f;
        i8_row_scale(amax, scale, inv, levels);
        ws[b * kI8Panel + j] = scale;
        // the block's quads: one block is every quad, the row's padding included
        const int q0 = block <= 0 ? 0 : k0 / kI8Quad;
        const int q1 = block <= 0 ? quads : (k1 + kI8Quad - 1) / kI8Quad;
        int32_t sum = 0;
        for (int kq = q0; kq < q1; ++kq) {
            uint8_t* lane = dst + (size_t) kq * (size_t) kI8QuadBytes + (size_t) j * kI8Quad;
            for (int qi = 0; qi < kI8Quad; ++qi) {
                const int k = kq * kI8Quad + qi;
                const int q = (real && k < n_in) ? i8_level(value(k), inv, levels) : 0;
                lane[qi] = (uint8_t) (int8_t) q;    // the signed side: two's complement bits
                sum += q;
            }
        }
        // |sum| <= 127 * n_in, and 128 * 127 * kI8MaxIn is 5.3e8 against int32's 2.1e9: no wrap.
        wc[b * kI8Panel + j] = sum * 128;
    }
}

template <class Row>
bool pack(Row row, int n_in, int n_out, uint8_t* panel, float* wscale, int32_t* wcorr, int p0, int p1, int block,
          int levels) {
    if (n_in > kI8MaxIn || !i8_gemm_block_ok(block) || !i8_gemm_levels_ok(levels)) return false;   // THE BOUNDS
    const int quads = i8_gemm_quads(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    for (int p = p0; p < p1; ++p) {
        uint8_t* dst = panel + (size_t) p * (size_t) quads * (size_t) kI8QuadBytes;
        const int base = p * kI8Panel;
        const int rows = std::min(kI8Panel, n_out - base);
        float* ws = wscale + (size_t) p * nb * kI8Panel;
        int32_t* wc = wcorr + (size_t) p * nb * kI8Panel;
        for (int j = 0; j < kI8Panel; ++j) {
            const int o = std::min(base + j, n_out - 1);   // a pad lane reads nothing (`real` false)
            pack_lane([&](int k) { return row(o, k); }, j < rows, n_in, quads, nb, block, levels, dst, j, ws, wc);
        }
    }
    return true;
}

}  // namespace

bool i8_gemm_isa_ok(I8GemmIsa isa) noexcept {
#if defined(STRATA_I8_GEMM_X86)
    if (isa == I8GemmIsa::avx512) return cpu_avx2_ok() && cpu_avx512vnni_ok();
    if (isa == I8GemmIsa::vnni) return STRATA_AVXVNNI && cpu_avx2_ok() && cpu_avxvnni_ok();
    if (isa == I8GemmIsa::vnni_int8) return STRATA_AVXVNNIINT8 && cpu_avx2_ok() && cpu_avxvnniint8_ok();
    if (isa == I8GemmIsa::avx2) return cpu_avx2_ok();
    return true;
#else
    return isa == I8GemmIsa::scalar;
#endif
}

I8GemmIsa i8_gemm_isa() {
    static const I8GemmIsa isa = [] {
        const char* e = strata::env_get("I8_GEMM_ISA");
        if (e != nullptr && std::strcmp(e, "scalar") == 0) return I8GemmIsa::scalar;
        const bool cap_avx2 = e != nullptr && std::strcmp(e, "avx2") == 0;
        const bool cap_vnni = cap_avx2 || (e != nullptr && std::strcmp(e, "vnni") == 0);
        const bool cap_int8 = cap_vnni || (e != nullptr && std::strcmp(e, "vnni_int8") == 0);
        if (!cap_int8 && i8_gemm_isa_ok(I8GemmIsa::avx512)) return I8GemmIsa::avx512;
        if (!cap_vnni && i8_gemm_isa_ok(I8GemmIsa::vnni_int8)) return I8GemmIsa::vnni_int8;
        if (!cap_avx2 && i8_gemm_isa_ok(I8GemmIsa::vnni)) return I8GemmIsa::vnni;
        return i8_gemm_isa_ok(I8GemmIsa::avx2) ? I8GemmIsa::avx2 : I8GemmIsa::scalar;
    }();
    return isa;
}

const char* i8_gemm_isa_name(I8GemmIsa isa) noexcept {
    return isa == I8GemmIsa::avx512 ? "avx512-vnni" : isa == I8GemmIsa::vnni_int8 ? "vnni-int8"
           : isa == I8GemmIsa::vnni ? "vnni"
           : isa == I8GemmIsa::avx2 ? "avx2" : "scalar";
}

int i8_gemm_weight_levels(I8GemmIsa isa) {
    static const bool eight = [] {
        const char* e = strata::env_get("I8_GEMM_WEIGHT_BITS");
        return e != nullptr && std::strcmp(e, "8") == 0;
    }();
    return isa == I8GemmIsa::avx2 && !eight ? kI8Levels7 : kI8Levels;
}

bool i8_gemm_pack(const uint16_t* w, size_t ld, int n_in, int n_out, uint8_t* panel, float* wscale, int32_t* wcorr,
                  int p0, int p1, int block, int levels) {
    return pack([&](int o, int k) { return strata::kernels::f32_from_bf16(w[(size_t) o * ld + (size_t) k]); }, n_in,
                n_out, panel, wscale, wcorr, p0, p1, block, levels);
}

bool i8_gemm_pack_f32(const float* w, size_t ld, int n_in, int n_out, uint8_t* panel, float* wscale, int32_t* wcorr,
                      int p0, int p1, int block, int levels) {
    return pack([&](int o, int k) { return w[(size_t) o * ld + (size_t) k]; }, n_in, n_out, panel, wscale, wcorr, p0,
                p1, block, levels);
}

void i8_gemm_quantize(I8GemmIsa isa, const float* x, size_t ldx, int n, int n_in, uint8_t* xq, size_t ldq,
                      float* xs, int block) {
    if (n <= 0 || n_in <= 0) return;
#if defined(STRATA_I8_GEMM_X86)
    // Only a vector ISA takes the lane copy: the scalar kernel is the definition, and its quantizer has to be the
    // one the reference ran through or the two would differ by the quantization as well as by the sum.
    if (isa != I8GemmIsa::scalar && cpu_avx2_ok()) {
        i8_gemm_detail::i8_gemm_quantize_avx2(x, ldx, n, n_in, xq, ldq, xs, block, isa == I8GemmIsa::vnni_int8);
        return;
    }
#else
    (void) isa;
#endif
    i8_gemm_quantize_scalar(x, ldx, n, n_in, xq, ldq, xs, block);
}

void i8_gemm_rows(I8GemmIsa isa, const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in,
                  int n_out, const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok,
                  int p0, int p1, const float* bias, int block, int levels) {
    if (p0 >= p1 || n_tok <= 0 || n_out <= 0) return;
#if defined(STRATA_I8_GEMM_X86)
    if (isa == I8GemmIsa::avx512) {
        i8_gemm_detail::i8_gemm_rows_avx512(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias,
                                            block);
        return;
    }
    if (isa != I8GemmIsa::scalar) {
        const int copy = isa == I8GemmIsa::vnni ? 1 : isa == I8GemmIsa::vnni_int8 ? 2 : 0;
        i8_gemm_detail::i8_gemm_rows_x86(copy, panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1,
                                         bias, block, levels);
        return;
    }
#else
    (void) isa;
#endif
    // the reference computes sum qu * qw - corr whatever the weights' levels: both forms are its definition
    (void) levels;
    i8_gemm_rows_scalar(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias, block);
}

}  // namespace strata::kernels::cpu
