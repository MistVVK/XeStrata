// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/vision_cpu_avx512.cpp - vision_cpu_avx2.cpp's loops sixteen floats wide: the copy vision_cpu.cpp
// takes where the int8 product runs its AVX-512 VNNI kernel (i8_gemm_isa()).  Compiled with -mavx512f (and the AVX2
// flags), and nothing else is; the same rule as there about headers with inline functions.
#include <immintrin.h>

#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu::vision_detail::avx512 {

namespace {

// e^x as vision_cpu_avx2.cpp's exp8, sixteen lanes
inline __m512 exp16(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.3f)), _mm512_set1_ps(88.3f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.f)));
    const __m512i e = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}

inline float gelu_tanh1(float v) {
    return 0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

}  // namespace

void layer_norm_row(const float* x, int64_t C, const float* w, const float* b, float eps, float* y) {
    int64_t c = 0;
    __m512 s16 = _mm512_setzero_ps();
    for (; c + 16 <= C; c += 16) s16 = _mm512_add_ps(s16, _mm512_loadu_ps(x + c));
    float s = _mm512_reduce_add_ps(s16);
    for (; c < C; ++c) s += x[c];
    const float mean = s / (float) C;
    const __m512 m16 = _mm512_set1_ps(mean);
    __m512 v16 = _mm512_setzero_ps();
    for (c = 0; c + 16 <= C; c += 16) {
        const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(x + c), m16);
        v16 = _mm512_fmadd_ps(d, d, v16);
    }
    float v = _mm512_reduce_add_ps(v16);
    for (; c < C; ++c) v += (x[c] - mean) * (x[c] - mean);
    const float inv = 1.f / std::sqrt(v / (float) C + eps);
    const __m512 i16 = _mm512_set1_ps(inv);
    for (c = 0; c + 16 <= C; c += 16) {
        const __m512 d = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + c), m16), i16);
        _mm512_storeu_ps(y + c, _mm512_fmadd_ps(d, _mm512_loadu_ps(w + c), _mm512_loadu_ps(b + c)));
    }
    for (; c < C; ++c) y[c] = (x[c] - mean) * inv * w[c] + b[c];
}

void gelu_tanh_span(float* x, int64_t n) {
    int64_t i = 0;
    const __m512 k0 = _mm512_set1_ps(2.f * 0.7978845608028654f), k1 = _mm512_set1_ps(0.044715f),
                 one = _mm512_set1_ps(1.f);
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        const __m512 u2 = _mm512_mul_ps(k0, _mm512_fmadd_ps(_mm512_mul_ps(k1, v), _mm512_mul_ps(v, v), v));
        const __m512 e = _mm512_add_ps(exp16(u2), one);
        _mm512_storeu_ps(x + i, _mm512_sub_ps(v, _mm512_div_ps(v, e)));
    }
    for (; i < n; ++i) x[i] = gelu_tanh1(x[i]);
}

void add_span(float* x, const float* v, int64_t n) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(v + i)));
    for (; i < n; ++i) x[i] += v[i];
}

float softmax_row(float* x, int64_t n) {
    int64_t i = 0;
    __m512 m16 = _mm512_set1_ps(-INFINITY);
    for (; i + 16 <= n; i += 16) m16 = _mm512_max_ps(m16, _mm512_loadu_ps(x + i));
    float m = _mm512_reduce_max_ps(m16);
    for (; i < n; ++i) m = std::fmax(m, x[i]);
    const __m512 mm = _mm512_set1_ps(m);
    __m512 s16 = _mm512_setzero_ps();
    for (i = 0; i + 16 <= n; i += 16) {
        const __m512 e = exp16(_mm512_sub_ps(_mm512_loadu_ps(x + i), mm));
        _mm512_storeu_ps(x + i, e);
        s16 = _mm512_add_ps(s16, e);
    }
    float s = _mm512_reduce_add_ps(s16);
    for (; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    return s;
}

#include "vision_cpu_rows.inl"

}  // namespace strata::kernels::cpu::vision_detail::avx512
