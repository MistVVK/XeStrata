// tools/vision/own/cpu_kernels_avx2.cpp - the AVX2 + FMA + F16C tier (6 x 16 register tile).  Compiled with -mavx2 -mfma -mf16c (/arch:AVX2 on MSVC).
#include "cpu_kernels.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include <cmath>

namespace strata_vit {
namespace {
namespace kn {

struct V {
    static constexpr int W = 8;
    __m256 v;
    V() : v(_mm256_setzero_ps()) {}
    V(__m256 x) : v(x) {}
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    static V set1(float f) { return _mm256_set1_ps(f); }
    static V zero() { return _mm256_setzero_ps(); }
    static V fma(V a, V b, V c) { return _mm256_fmadd_ps(a.v, b.v, c.v); }
    static float reduce_add(V a) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }
    static float reduce_max(V a) {
        __m128 s = _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
        s = _mm_max_ps(s, _mm_movehl_ps(s, s));
        s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }
    friend V operator+(V a, V b) { return _mm256_add_ps(a.v, b.v); }
    friend V operator-(V a, V b) { return _mm256_sub_ps(a.v, b.v); }
    friend V operator*(V a, V b) { return _mm256_mul_ps(a.v, b.v); }
    friend V operator/(V a, V b) { return _mm256_div_ps(a.v, b.v); }
};
inline V vmax(V a, V b) { return _mm256_max_ps(a.v, b.v); }

// expf (Cephes): x = n ln2 + r, exp(x) = 2^n * p(r)
inline V vexp(V xv) {
    __m256 x = _mm256_min_ps(_mm256_max_ps(xv.v, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    int i = 0;
    if (kind == 2) {
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (p + i))));
        for (; i < n; ++i) out[i] = quant::f16_to_f32(p[i]);
    } else {
        for (; i + 8 <= n; i += 8) {
            const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (p + i))), 16);
            _mm256_storeu_ps(out + i, _mm256_castsi256_ps(w));
        }
        for (; i < n; ++i) out[i] = quant::bf16_to_f32(p[i]);
    }
}

}  // namespace kn
}  // namespace
}  // namespace strata_vit

#define STRATA_KN_MR 6
#define STRATA_KN_NAME "avx2"
#define STRATA_KN_GETTER cpu_ops_avx2
#include "cpu_kernels_impl.h"

#else   // not x86-64: the tier does not exist, the dispatcher never asks for it
namespace strata_vit {
const CpuOps& cpu_ops_avx2() { return cpu_ops_scalar(); }
}
#endif
