// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/canon_expert.cpp - an expert's rows on the CPU in the GPU's arithmetic (canon_expert.hpp).
//
// The GPU's grouped experts (seg_dot_canon in iq_kernels.cpp) take the gate/up activations as q8_1 blocks whose scale
// is one for each 256 values (canon_math.hpp's), the down activations with one for each 32.  An i-quant gate/up row's
// eight 32-value calls of a 256-value block share a scale: their exact integer sums, (2 ls + 1)-weighted, add first,
// then the block's scale product times that sum is added with one fma into the even or the odd blocks' sum, and the
// two are added at the end (iq_avx2's iq256_canon_rows).  A down row's call (IQ4_NL: 16 values, Q2_0: 32) has a scale
// of its own: its scale product times its integer sum goes with one fma into the sum of the call's lane (the call
// index modulo the row's 8 or 4 lanes), and the lane sums are added as the GPU's butterfly (j += j ^ o, o = 4 .. 1).
// These rows take the same steps, so a row is the same bits wherever it ran; the integer sums are exact in any order,
// so the CPU keeps a down activation's values in the order its rows decode the weights (act_pos).
#include "strata/kernels/cpu/canon_expert.hpp"

#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/platform/env.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace strata::kernels::cpu {
namespace {

namespace canon = strata::kernels::canon;

inline float h2f(ggml_half h) { return _cvtsh_ss(h); }
inline ggml_half f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }

// The CPU's form of a canonical activation of n values with a scale for each `group` (the same values as the GPU's
// q8_1 blocks, laid out for aligned loads): the scales as floats (the fp16 value), each 32 values' integer sum (Q2_0's
// rows take code x q - sum q), then the int8 values from a 64-byte boundary.
inline size_t qs_offset(int64_t n, int group) { return (size_t) (((n / group) * 4 + (n / 32) * 4 + 63) / 64 * 64); }
inline const float* act_scales(const void* a) { return static_cast<const float*>(a); }
inline const int32_t* act_sums(const void* a, int64_t n, int group) {
    return reinterpret_cast<const int32_t*>(static_cast<const float*>(a) + n / group);
}
inline const int8_t* act_values(const void* a, int64_t n, int group) {
    return static_cast<const int8_t*>(a) + qs_offset(n, group);
}

// Where value j of an activation of n values goes in the order a down projection's rows read it (canon_quant's
// `order`).  Q2_0: four blocks' 256 values as 8 vectors of 32, lane k of vector 2 p + h (4 values) holding call k's
// values 16 h + 4 i + p, i = 0..3 (codes 4 bytes a lane, plane p); the last 128 values when the 256s do not fill n
// as two 64-value plane blocks (value 4 k + p at 16 p + k).  IQ4_NL: four blocks' 128 values as 4 vectors of 32, lane
// 2 b + c of vector m (4 values) holding block b's call c's m-th 4 values (a call: values 0-7 and 16-23, or 8-15 and
// 24-31).
constexpr int64_t act_pos(int64_t j, CanonOrder order, int64_t n) {
    if (order == CanonOrder::q2_planes) {
        if ((j & ~int64_t{255}) + 256 > n) return (j & ~int64_t{63}) + 16 * (j & 3) + ((j & 63) >> 2);   // the tail
        const int64_t v = j & 255, w = v & 31;   // call v / 32, its value w: 16 h + 4 i + p
        return (j & ~int64_t{255}) + 32 * (2 * (w & 3) + (w >> 4)) + 4 * (v >> 5) + ((w >> 2) & 3);
    }
    if (order == CanonOrder::iq4nl_calls) {
        const int64_t v = j & 31, r = v & 15, b = (j >> 5) & 3;
        const int64_t c = r >> 3, m = (v >= 16 ? 2 : 0) + ((r >> 2) & 1);
        return (j & ~int64_t{127}) + 32 * m + 4 * (2 * b + c) + (r & 3);
    }
    return j;
}

// act_pos within a pattern, as tables: Q2_0's 256 values (or the 64 of a last plane block), IQ4_NL's 128
struct PosTables {
    uint8_t q2[256], q2_tail[64], iq4[128];
    constexpr PosTables() : q2{}, q2_tail{}, iq4{} {
        for (int j = 0; j < 256; ++j) q2[j] = (uint8_t) act_pos(j, CanonOrder::q2_planes, 256);
        for (int j = 0; j < 64; ++j) q2_tail[j] = (uint8_t) act_pos(j, CanonOrder::q2_planes, 64);
        for (int j = 0; j < 128; ++j) iq4[j] = (uint8_t) act_pos(j, CanonOrder::iq4nl_calls, 128);
    }
};
constexpr PosTables kPos{};

// one IQ2_XS row of n values against an activation with a scale per 256 values (the reference loops)
float iq2xs_row(const block_iq2_xs* w, int n, const void* a) {
    STRATA_CANON_NO_CONTRACT
    const int8_t* xq = act_values(a, n, QK_K);
    const float* xd = act_scales(a);
    float s[2] = {0.0f, 0.0f};
    const int nb = n / QK_K;
    for (int kbx = 0; kbx < nb; ++kbx) {
        int t = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int8_t* q = xq + (ptrdiff_t) kbx * QK_K + 32 * (ptrdiff_t) ib;
            int sum[2] = {0, 0};
            for (int l = 0; l < 4; ++l) {
                const uint16_t qv = w[kbx].qs[4 * ib + l];
                const uint8_t* g = (const uint8_t*) (iq2xs_grid + (qv & 511));
                const uint8_t signs = ksigns_iq2xs[qv >> 9];
                for (int j = 0; j < 8; ++j) {
                    const int v = (int) g[j] * (int) q[8 * l + j];
                    sum[l / 2] += (signs >> j) & 1 ? -v : v;
                }
            }
            t += sum[0] * (2 * (w[kbx].scales[ib] & 0x0F) + 1) + sum[1] * (2 * (w[kbx].scales[ib] >> 4) + 1);
        }
        const float c = h2f(w[kbx].d) * xd[kbx] * 0.125f;
        s[kbx & 1] = canon::fma(c, (float) t, s[kbx & 1]);
    }
    return s[0] + s[1];
}

// one IQ4_NL row of n values against an activation with a scale per 32 values (the reference loops): the GPU's lane
// 2 (block % 4) + call, a call the block's values 0-7 and 16-23, or 8-15 and 24-31
float iq4nl_row(const block_iq4_nl* w, int n, const void* a) {
    const int8_t* xq = act_values(a, n, QK4_NL);
    const float* xd = act_scales(a);
    const int nb = n / QK4_NL;
    int ta = canon::kFxNone, tw = canon::kFxNone;   // canon_math.hpp's fixed-point sums: the row's top
    for (int kbx = 0; kbx < nb; ++kbx) {
        ta = std::max(ta, canon::fx_exp(xd[kbx]));
        tw = std::max(tw, canon::fx_exp(h2f(w[kbx].d)));
    }
    int64_t s = 0;
    for (int kbx = 0; kbx < nb; ++kbx) {
        int t = 0;
        for (int i = 0; i < 16; ++i) {
            t += (int) kvalues_iq4nl[w[kbx].qs[i] & 0x0F] * (int) xq[act_pos(32 * kbx + i, CanonOrder::iq4nl_calls, n)];
            t += (int) kvalues_iq4nl[w[kbx].qs[i] >> 4] *
                 (int) xq[act_pos(32 * kbx + i + 16, CanonOrder::iq4nl_calls, n)];
        }
        s += (int64_t) canon::fx_scale(xd[kbx], h2f(w[kbx].d), ta + tw) * t;
    }
    return canon::fx_value(s, ta + tw, 0);
}

// canon::silu_mul 8 at a time: the same operations in the same order (exp's range reduction, its polynomial and the
// exponent bits; rcp's seed and Newton steps), so the same bits
inline __m256 rcp8(__m256 v) {
    __m256 r = _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(0x7EF311C3), _mm256_castps_si256(v)));
    const __m256 one = _mm256_set1_ps(1.0f);
    for (int i = 0; i < 3; ++i) {
        const __m256 e = _mm256_fnmadd_ps(v, r, one);   // fma(-v, r, 1)
        r = _mm256_fmadd_ps(r, e, r);
    }
    return r;
}
inline __m256 exp8(__m256 x) {
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(80.0f)), _mm256_set1_ps(-80.0f));
    const __m256 big = _mm256_set1_ps(12582912.0f);
    const __m256 kf = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), big), big);
    __m256 r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.693145752f), x);
    r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(1.42860677e-06f), r);
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    const __m256i k = _mm256_slli_epi32(_mm256_cvttps_epi32(kf), 23);   // kf is an integer: the cast is exact
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), k));
}
inline __m256 silu_mul8(__m256 g, __m256 u) {
    const __m256 e = exp8(_mm256_xor_ps(g, _mm256_castsi256_ps(_mm256_set1_epi32((int) 0x80000000u))));
    return _mm256_mul_ps(_mm256_mul_ps(g, rcp8(_mm256_add_ps(_mm256_set1_ps(1.0f), e))), u);
}

// one Q2_0 gate/up row of n values against an activation in Q2_0's order with a scale per 256 values (the reference
// loops): call k (32 values) d x the scale x its exact (code - 1) x q sum into the sum of k modulo 16, the 16 added
// as the GPU's butterfly
float q20_gu_row(const block_q2_0* w, int n, const void* a) {
    STRATA_CANON_NO_CONTRACT
    const int8_t* xq = act_values(a, n, QK_K);
    const float* xd = act_scales(a);
    float s[16] = {};
    for (int k = 0; k < n / 32; ++k) {
        const block_q2_0& b = w[k / 2];
        int t = 0;
        for (int i = 0; i < 32; ++i) {
            const int v = 32 * (k % 2) + i;   // the value in its block
            const int code = (b.qs[v / 4] >> (2 * (v % 4))) & 3;
            t += (code - 1) * (int) xq[act_pos(32 * k + i, CanonOrder::q2_planes, n)];
        }
        const float c = h2f(b.d) * xd[(32 * k) / QK_K];
        s[k % 16] = canon::fma(c, (float) t, s[k % 16]);
    }
    for (int o = 8; o > 0; o >>= 1)
        for (int l = 0; l < o; ++l) s[l] += s[l + o];
    return s[0];
}

// A down activation's scales as canon_math.hpp's fx parts, for the IQ4_NL rows' fixed-point sums: each block's
// mantissa and exponent, and the largest exponent
struct FxAct {
    alignas(32) int32_t m[kNativeMaxFF / QK4_NL], e[kNativeMaxFF / QK4_NL];
    int top;
};
void fx_act(const float* dx, int nb, FxAct& a) {
    a.top = canon::kFxNone;
    for (int b = 0; b < nb; ++b) {
        const canon::FxPart p = canon::fx_part(dx[b]);   // a scale is never negative
        a.m[b] = (int32_t) p.m;
        a.e[b] = p.e;
        a.top = std::max(a.top, p.e);
    }
}

// ---- the AVX2 forms of the down rows (the gate/up rows: iq_avx2's iq256_canon_rows)
namespace plain {
#define STRATA_ROWS_VNNI 0
#define STRATA_ROWS_FN
#include "canon_expert_rows.inl"
#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_ROWS_VNNI 1
#define STRATA_ROWS_FN __attribute__((target("avxvnni")))
#include "canon_expert_rows.inl"
#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
}  // namespace vnni
#endif

bool vnni_on() {
#if STRATA_AVXVNNI
    static const bool on = cpu_avxvnni_ok();
    return on;
#else
    return false;
#endif
}

template <class F> void with_nt(int nt, F&& f) {
    switch (nt) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default:   // the arrays a job's tokens fill hold 8 (#45: a ninth wrote past them, and the pool faulted)
            std::fprintf(stderr, "xestrata: an expert job of %d tokens reached the CPU kernels, which take 1 to 8\n", nt);
            std::abort();
    }
}

}  // namespace

bool canon_layer(const NativeFmt& f) {
    static const bool on = [] {   // STRATA_CANON=0: the CPU's former kernels (comparisons; no longer the GPU's bits)
        const char* v = strata::env_get("CANON");
        return v == nullptr || v[0] != '0';
    }();
    const auto gu_ok = [](int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 23 || t == 42; };
    const auto fits = [](int64_t n, int group, size_t cap) { return (size_t) n + qs_offset(n, group) <= cap; };
    const int up = f.up_type < 0 ? f.gu_type : f.up_type;
    return on && gu_ok(f.gu_type) && gu_ok(up) && (f.gu_type == 42) == (up == 42) && (f.d_type == 20 || f.d_type == 42) &&
           f.n_embd % (int64_t{2} * QK_K) == 0 && f.n_ff % 128 == 0 && fits(f.n_embd, QK_K, kNativeActBytes) &&
           fits(f.n_ff, 32, kNativeHBytes);
}

void canon_quant(const float* x, int64_t n, int group, void* dst, CanonOrder order) {
    STRATA_CANON_NO_CONTRACT
    float* ds = static_cast<float*>(dst);
    int32_t* sums = reinterpret_cast<int32_t*>(ds + n / group);   // one for each 32
    int8_t* q = static_cast<int8_t*>(dst) + qs_offset(n, group);
    static const bool scalar = strata::env_get("CANON_SCALAR") != nullptr;   // the reference loops (A/B)
    if (scalar) {
        for (int64_t g0 = 0; g0 < n; g0 += group) {
            float amax = 0.0f;
            for (int i = 0; i < group; ++i) {
                const float a = std::fabs(x[g0 + i]);
                amax = a > amax ? a : amax;
            }
            const float d = canon::q8_scale(amax), id = canon::q8_inv(d);
            ds[g0 / group] = h2f(f2h(d));   // the fp16 value the GPU's blocks hold
            for (int i = 0; i < group; ++i) {
                const int8_t v = canon::q8_value(x[g0 + i], id);
                q[act_pos(g0 + i, order, n)] = v;
                if (i % 32 == 0) sums[(g0 + i) / 32] = 0;
                sums[(g0 + i) / 32] += v;
            }
        }
        return;
    }
    // canon::q8_value 8 at a time: t = x id, t + 0.5 with t's sign, held to [-127, 127], truncated (the same
    // operations, so the same bits); group a multiple of 32
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 signm = _mm256_castsi256_ps(_mm256_set1_epi32((int) 0x80000000u));
    const __m256 half = _mm256_set1_ps(0.5f), lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    alignas(32) int8_t tmp[256];
    for (int64_t g0 = 0; g0 < n; g0 += group) {
        const float* xg = x + g0;
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < group; i += 8) m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(xg + i), absm));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        const float amax = _mm_cvtss_f32(h);
        const float d = canon::q8_scale(amax), id = canon::q8_inv(d);
        ds[g0 / group] = h2f(f2h(d));   // the fp16 value the GPU's blocks hold
        const __m256 vid = _mm256_set1_ps(id);
        int8_t* out = order == CanonOrder::plain ? q + g0 : tmp;
        for (int i = 0; i < group; i += 32) {
            __m256i v[4], acc = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j) {
                const __m256 t = _mm256_mul_ps(_mm256_loadu_ps(xg + i + 8 * (ptrdiff_t) j), vid);
                const __m256 r = _mm256_add_ps(t, _mm256_or_ps(_mm256_and_ps(t, signm), half));
                v[j] = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(r, lo), hi));
                acc = _mm256_add_epi32(acc, v[j]);
            }
            const __m256i p8 = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_permutevar8x32_epi32(p8, perm));
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            sums[(g0 + i) / 32] = _mm_cvtsi128_si32(s4);
        }
        if (order != CanonOrder::plain) {
            if (group == 32 || group == 256) {   // a down activation's 32, a Q2_0 gate/up's 256: the pattern's table
                const bool q2 = order == CanonOrder::q2_planes;
                const int64_t span = q2 ? ((g0 & ~int64_t{255}) + 256 <= n ? 256 : 64) : 128;
                const uint8_t* tbl = (q2 ? (span == 256 ? kPos.q2 : kPos.q2_tail) : kPos.iq4) + (g0 & (span - 1));
                int8_t* base = q + (g0 & ~(span - 1));
                for (int i = 0; i < group; ++i) base[tbl[i]] = tmp[i];
            } else {
                for (int i = 0; i < group; ++i) q[act_pos(g0 + i, order, n)] = tmp[i];
            }
        }
    }
}

void canon_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0,
                   int r1) {
    const int n = (int) f.n_embd;
    const int8_t* x[8];
    const float* dx[8];
    for (int t = 0; t < nt; ++t) {
        x[t] = act_values(act[t], n, QK_K);
        dx[t] = act_scales(act[t]);
    }
    const int up_type = f.up_type < 0 ? f.gu_type : f.up_type;
    const size_t up_row = f.up_row != 0 ? f.up_row : f.gu_row;
    const int32_t* qs32[8];
    if (f.gu_type == 42)
        for (int t = 0; t < nt; ++t) qs32[t] = act_sums(act[t], n, QK_K);
    // iq_avx2's canonical rows (or the reference loops of IQ2_XS, A/B), a chunk of rows at a time for the gate and the
    // up rows, then the SwiGLU
    static const bool scalar_env = strata::env_get("CANON_SCALAR") != nullptr;
    const bool scalar = scalar_env && f.gu_type == up_type && (f.gu_type == 17 || f.gu_type == 42);
    constexpr int kChunk = 32;
    float gb[8][kChunk], ub[8][kChunk];
    for (int c0 = r0; c0 < r1; c0 += kChunk) {
        const int c1 = c0 + kChunk < r1 ? c0 + kChunk : r1;
        float* g[8];
        float* u[8];
        for (int t = 0; t < nt; ++t) {
            g[t] = gb[t] - c0;
            u[t] = ub[t] - c0;
        }
        if (scalar && f.gu_type == 42) {
            for (int r = c0; r < c1; ++r)
                for (int t = 0; t < nt; ++t) {
                    g[t][r] = q20_gu_row(reinterpret_cast<const block_q2_0*>(blob + (size_t) r * f.gu_row), n, act[t]);
                    u[t][r] = q20_gu_row(reinterpret_cast<const block_q2_0*>(blob + f.up_off + (size_t) r * up_row), n, act[t]);
                }
        } else if (scalar) {
            for (int r = c0; r < c1; ++r)
                for (int t = 0; t < nt; ++t) {
                    g[t][r] = iq2xs_row(reinterpret_cast<const block_iq2_xs*>(blob + (size_t) r * f.gu_row), n, act[t]);
                    u[t][r] = iq2xs_row(reinterpret_cast<const block_iq2_xs*>(blob + f.up_off + (size_t) r * up_row), n, act[t]);
                }
        } else if (f.gu_type == 42) {   // Q2_0 (the activation in its order)
            with_nt(nt, [&](auto NTc) {
                constexpr int NT = decltype(NTc)::value;
                float gr[NT], ur[NT];
                for (int r = c0; r < c1; ++r) {
#if STRATA_AVXVNNI
                    if (vnni_on()) {
                        vnni::q20_gu_rows_nt<NT>(blob + (size_t) r * f.gu_row, n, x, dx, qs32, gr);
                        vnni::q20_gu_rows_nt<NT>(blob + f.up_off + (size_t) r * up_row, n, x, dx, qs32, ur);
                    } else
#endif
                    {
                        plain::q20_gu_rows_nt<NT>(blob + (size_t) r * f.gu_row, n, x, dx, qs32, gr);
                        plain::q20_gu_rows_nt<NT>(blob + f.up_off + (size_t) r * up_row, n, x, dx, qs32, ur);
                    }
                    for (int t = 0; t < NT && t < nt; ++t) {
                        g[t][r] = gr[t];
                        u[t][r] = ur[t];
                    }
                }
            });
        } else {
            iq256_canon_rows(f.gu_type, blob, f.gu_row, n, x, dx, nt, g, c0, c1);
            iq256_canon_rows(up_type, blob + f.up_off, up_row, n, x, dx, nt, u, c0, c1);
        }
        for (int t = 0; t < nt; ++t) {
            int r = c0;
            for (; r + 8 <= c1; r += 8)
                _mm256_storeu_ps(ff[t] + r, silu_mul8(_mm256_loadu_ps(g[t] + r), _mm256_loadu_ps(u[t] + r)));
            for (; r < c1; ++r) ff[t][r] = canon::silu_mul(g[t][r], u[t][r]);
        }
    }
}

void canon_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0,
                     int r1) {
    const int n = (int) f.n_ff;
    const int8_t* x[8];
    const float* dx[8];
    for (int t = 0; t < nt; ++t) {
        x[t] = act_values(hq[t], n, QK4_NL);
        dx[t] = act_scales(hq[t]);
    }
    static const bool scalar = strata::env_get("CANON_SCALAR") != nullptr;
    if (f.d_type == 42) {   // Q2_0
        const int32_t* qs32[8];
        for (int t = 0; t < nt; ++t) qs32[t] = act_sums(hq[t], n, 32);
        with_nt(nt, [&](auto NTc) {
            constexpr int NT = decltype(NTc)::value;
            for (int r = r0; r < r1; ++r) {
                const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
                float o[NT];
#if STRATA_AVXVNNI
                if (vnni_on()) vnni::q20_rows_nt<NT>(dr, n, x, dx, qs32, o);
                else
#endif
                    plain::q20_rows_nt<NT>(dr, n, x, dx, qs32, o);
                for (int t = 0; t < NT; ++t) out[t][r] = o[t];
            }
        });
        return;
    }
    with_nt(nt, [&](auto NTc) {
        constexpr int NT = decltype(NTc)::value;
        FxAct fa[NT];   // the activations' parts once for every row
        if (!scalar)
            for (int t = 0; t < NT; ++t) fx_act(dx[t < nt ? t : 0], n / QK4_NL, fa[t]);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
            float o[NT];
            if (scalar)
                for (int t = 0; t < NT; ++t) o[t] = iq4nl_row(reinterpret_cast<const block_iq4_nl*>(dr), n, hq[t]);
            else
#if STRATA_AVXVNNI
                if (vnni_on()) vnni::iq4nl_rows_nt<NT>(dr, n, x, fa, o);
                else
#endif
                    plain::iq4nl_rows_nt<NT>(dr, n, x, fa, o);
            for (int t = 0; t < NT; ++t) out[t][r] = o[t];
        }
    });
}

}  // namespace strata::kernels::cpu
