// tools/vision/own/cpu_kernels_avx2.cpp - the AVX2 + FMA + F16C tier (6 x 16 register tile).  Compiled with -mavx2 -mfma -mf16c (/arch:AVX2 on MSVC).
#include "cpu_tier.h"      // only this and <immintrin.h>: no standard-library code may be compiled with this file's ISA flags (cpu_tier.h)

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>


namespace strata_vit {
namespace {
namespace kn {

// one 16-bit value to float, bit exact (the hardware F16 conversion equals the software one for every finite value; BF16 is a shift)
inline float f16_one(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline float bf16_one(uint16_t h) { return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128((int) ((uint32_t) h << 16)))); }
// the helpers cpu_kernels_impl.h asks for: no library call is compiled with this file's flags
template <class T> inline T kmin(T a, T b) { return b < a ? b : a; }
template <class T> inline T kmax(T a, T b) { return a < b ? b : a; }
inline void kcopy(float* d, const float* s, int n) { for (int i = 0; i < n; ++i) d[i] = s[i]; }
inline void kzero(float* d, int n) { for (int i = 0; i < n; ++i) d[i] = 0.0f; }

struct V {
    static constexpr int W = 8;
    __m256 v;
    V() : v(_mm256_setzero_ps()) {}
    V(__m256 x) : v(x) {}
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    static V set1(float f) { return _mm256_set1_ps(f); }
    static V zero() { return _mm256_setzero_ps(); }
    static float sqrt1(float x) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))); }     // sqrtss: the correctly rounded root, as std::sqrt
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

// The product's micro-kernel on raw __m256 (no wrapper struct in the hot loop: MSVC kept the wrapper in memory, 1.46 to 1.82 of mtmd-CPU's time on Windows;
// RESULTS-OWN-3V.md 12.14).  The same operations in the same order as the generic micro of cpu_kernels_impl.h, so the results are bit-identical.
#define STRATA_KN_HAS_MICRO 1
template <int MR>
inline void micro(const float* A, size_t lda, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    __m256 acc[MR][2];
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) {
            __m256 v = _mm256_setzero_ps();
            if (init != 0) v = _mm256_loadu_ps(C + (size_t) r * ldc + c * 8);
            if (init != 1 && bias) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + c * 8));
            acc[r][c] = v;
        }
    for (int k = 0; k < K; ++k) {
        const __m256 b0 = _mm256_loadu_ps(B + (size_t) k * ldb), b1 = _mm256_loadu_ps(B + (size_t) k * ldb + 8);
        for (int r = 0; r < MR; ++r) {
            const __m256 a = _mm256_set1_ps(A[(size_t) r * lda + k]);
            acc[r][0] = _mm256_fmadd_ps(a, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(a, b1, acc[r][1]);
        }
    }
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) _mm256_storeu_ps(C + (size_t) r * ldc + c * 8, acc[r][c]);
}

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

// 8 floats rounded to the 16-bit type and back (BF16: integer round to nearest even; F16: the hardware conversion pair, also round to nearest even)
inline __m256 round8(__m256 x, int kind) {
    if (kind == 2) return _mm256_cvtph_ps(_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
    __m256i u = _mm256_castps_si256(x);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb));
    return _mm256_castsi256_ps(_mm256_and_si256(u, _mm256_set1_epi32((int) 0xffff0000u)));
}
inline void round_buf(float* p, size_t n, int kind) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(p + i, round8(_mm256_loadu_ps(p + i), kind));
    if (i < n) {
        float t[8] = {0};
        for (size_t j = i; j < n; ++j) t[j - i] = p[j];
        _mm256_storeu_ps(t, round8(_mm256_loadu_ps(t), kind));
        for (size_t j = i; j < n; ++j) p[j] = t[j - i];
    }
}

inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    int i = 0;
    if (kind == 2) {
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (p + i))));
        for (; i < n; ++i) out[i] = f16_one(p[i]);
    } else {
        for (; i + 8 <= n; i += 8) {
            const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (p + i))), 16);
            _mm256_storeu_ps(out + i, _mm256_castsi256_ps(w));
        }
        for (; i < n; ++i) out[i] = bf16_one(p[i]);
    }
}

}  // namespace kn
}  // namespace
}  // namespace strata_vit

#define STRATA_KN_MR 6
#define KN_MIN(a, b) kmin(a, b)
#define KN_MAX(a, b) kmax(a, b)
#define KN_COPY(d, s, n) kcopy((d), (s), (n))
#define KN_ZERO(d, n) kzero((d), (n))
#define STRATA_KN_NAME "avx2"
#define STRATA_KN_GETTER cpu_ops_avx2
#include "cpu_kernels_impl.h"

#else   // not x86-64: the tier does not exist, the dispatcher never asks for it
namespace strata_vit {
const CpuOps& cpu_ops_avx2() { return cpu_ops_scalar(); }
}
#endif
