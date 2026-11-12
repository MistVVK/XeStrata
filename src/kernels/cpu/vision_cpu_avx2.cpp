// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/vision_cpu_avx2.cpp - the AVX2 loops of vision_cpu.cpp: a LayerNorm row, a span of tanh GELUs, a
// softmax row, a span of adds and the QKV's RoPE.  Compiled with -mavx2 -mfma, called only where cpu_avx2_ok() (the
// CPU reader refuses a CPU without); this file includes no header with inline functions another file also compiles,
// so no AVX2 copy of one can be the one the linker keeps.
#include <immintrin.h>

#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu::vision_detail::avx2 {

namespace {

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline float hmax(__m256 v) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// e^x: x = n ln2 + r, |r| <= ln2 / 2, e^r by Cephes' polynomial (expf's, relative error about 2e-7), 2^n in the
// exponent bits; x clamped to the floats' range (e^-88 is a denormal's neighbour: 0 for a softmax and a tanh)
inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

inline float gelu_tanh1(float v) {
    return 0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

}  // namespace

void layer_norm_row(const float* x, int64_t C, const float* w, const float* b, float eps, float* y) {
    int64_t c = 0;
    __m256 s8 = _mm256_setzero_ps();
    for (; c + 8 <= C; c += 8) s8 = _mm256_add_ps(s8, _mm256_loadu_ps(x + c));
    float s = hsum(s8);
    for (; c < C; ++c) s += x[c];
    const float mean = s / (float) C;
    const __m256 m8 = _mm256_set1_ps(mean);
    __m256 v8 = _mm256_setzero_ps();
    for (c = 0; c + 8 <= C; c += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + c), m8);
        v8 = _mm256_fmadd_ps(d, d, v8);
    }
    float v = hsum(v8);
    for (; c < C; ++c) v += (x[c] - mean) * (x[c] - mean);
    const float inv = 1.f / std::sqrt(v / (float) C + eps);
    const __m256 i8 = _mm256_set1_ps(inv);
    for (c = 0; c + 8 <= C; c += 8) {
        const __m256 d = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + c), m8), i8);
        _mm256_storeu_ps(y + c, _mm256_fmadd_ps(d, _mm256_loadu_ps(w + c), _mm256_loadu_ps(b + c)));
    }
    for (; c < C; ++c) y[c] = (x[c] - mean) * inv * w[c] + b[c];
}

// 0.5 v (1 + tanh(u)) with tanh(u) = 1 - 2 / (e^(2u) + 1): 0.5 v (2 - 2 / (e^(2u) + 1)) = v - v / (e^(2u) + 1)
void gelu_tanh_span(float* x, int64_t n) {
    int64_t i = 0;
    const __m256 k0 = _mm256_set1_ps(2.f * 0.7978845608028654f), k1 = _mm256_set1_ps(0.044715f),
                 one = _mm256_set1_ps(1.f);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 u2 = _mm256_mul_ps(k0, _mm256_fmadd_ps(_mm256_mul_ps(k1, v), _mm256_mul_ps(v, v), v));
        const __m256 e = _mm256_add_ps(exp8(u2), one);
        _mm256_storeu_ps(x + i, _mm256_sub_ps(v, _mm256_div_ps(v, e)));
    }
    for (; i < n; ++i) x[i] = gelu_tanh1(x[i]);
}

void add_span(float* x, const float* v, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(v + i)));
    for (; i < n; ++i) x[i] += v[i];
}

// x[i] = e^(x[i] - max), returns the sum
float softmax_row(float* x, int64_t n) {
    int64_t i = 0;
    __m256 m8 = _mm256_set1_ps(-INFINITY);
    for (; i + 8 <= n; i += 8) m8 = _mm256_max_ps(m8, _mm256_loadu_ps(x + i));
    float m = hmax(m8);
    for (; i < n; ++i) m = std::fmax(m, x[i]);
    const __m256 mm = _mm256_set1_ps(m);
    __m256 s8 = _mm256_setzero_ps();
    for (i = 0; i + 8 <= n; i += 8) {
        const __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), mm));
        _mm256_storeu_ps(x + i, e);
        s8 = _mm256_add_ps(s8, e);
    }
    float s = hsum(s8);
    for (; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    return s;
}

#include "vision_cpu_rows.inl"

}  // namespace strata::kernels::cpu::vision_detail::avx2
