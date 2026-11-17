// tools/vision/own/cpu_kernels_avx.cpp - the AVX tier without FMA, F16C or AVX2 (6 x 16 register tile: 12 ymm accumulators), for Sandy Bridge / Ivy
// Bridge and AMD's AVX CPUs before Excavator.  Compiled with -mavx (/arch:AVX on MSVC); only called when quant::isa_ok(Avx).
//
// As the SSE4.1 tier: a product term is a multiply then an add in the same k order as every tier (no FMA), F16 weights are converted in integer code
// (bit exact, no F16C: Sandy Bridge has none), exp is the Cephes polynomial with separate multiply and add.  AVX has no 256-bit integer operations, so the
// integer steps (exponent bits, conversions, rounding) run on 128-bit halves.
#include "cpu_tier.h"      // only this and <immintrin.h>: no standard-library code may be compiled with this file's ISA flags (cpu_tier.h)

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace strata_vit {
namespace {
namespace kn {

template <class T> inline T kmin(T a, T b) { return b < a ? b : a; }
template <class T> inline T kmax(T a, T b) { return a < b ? b : a; }
inline void kcopy(float* d, const float* s, int n) { for (int i = 0; i < n; ++i) d[i] = s[i]; }
inline void kzero(float* d, int n) { for (int i = 0; i < n; ++i) d[i] = 0.0f; }

#define STRATA_KN_RAW 1
constexpr int VW = 8;
typedef __m256 VR;
inline VR vr_load(const float* p) { return _mm256_loadu_ps(p); }
inline void vr_store(float* p, VR v) { _mm256_storeu_ps(p, v); }
inline VR vr_set1(float f) { return _mm256_set1_ps(f); }
inline VR vr_zero() { return _mm256_setzero_ps(); }
inline VR vr_add(VR a, VR b) { return _mm256_add_ps(a, b); }
inline VR vr_sub(VR a, VR b) { return _mm256_sub_ps(a, b); }
inline VR vr_mul(VR a, VR b) { return _mm256_mul_ps(a, b); }
inline VR vr_div(VR a, VR b) { return _mm256_div_ps(a, b); }
inline VR vr_fma(VR a, VR b, VR c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }      // two roundings (no FMA here)
inline VR vr_max(VR a, VR b) { return _mm256_max_ps(a, b); }
inline float vr_sqrt1(float x) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))); }
inline float vr_hadd(VR a) {      // the AVX2 tier's order
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
inline float vr_hmax(VR a) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

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
            acc[r][0] = _mm256_add_ps(_mm256_mul_ps(a, b0), acc[r][0]);
            acc[r][1] = _mm256_add_ps(_mm256_mul_ps(a, b1), acc[r][1]);
        }
    }
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) _mm256_storeu_ps(C + (size_t) r * ldc + c * 8, acc[r][c]);
}

inline __m256 combine(__m128 lo, __m128 hi) { return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1); }

// expf (Cephes, the AVX2 tier's constants), multiply and add separately; 2^n built on the two 128-bit halves
inline VR vr_exp(VR xv) {
    __m256 x = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    const __m256i n = _mm256_cvtps_epi32(fx);
    const __m128i b = _mm_set1_epi32(127);
    const __m128i elo = _mm_slli_epi32(_mm_add_epi32(_mm256_castsi256_si128(n), b), 23);
    const __m128i ehi = _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256(n, 1), b), 23);
    return _mm256_mul_ps(y, combine(_mm_castsi128_ps(elo), _mm_castsi128_ps(ehi)));
}

// rounding to 16 bits, bit for bit as the scalar tier (see cpu_kernels_sse41.cpp for the method), on 128-bit halves
inline __m128 round4(__m128 x, int kind) {
    const __m128i u = _mm_castps_si128(x);
    const __m128i absu = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
    if (kind == 1) {
        const __m128i lsb = _mm_and_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(1));
        const __m128i r = _mm_and_si128(_mm_add_epi32(u, _mm_add_epi32(_mm_set1_epi32(0x7fff), lsb)), _mm_set1_epi32((int) 0xffff0000u));
        const __m128i nan = _mm_cmpgt_epi32(absu, _mm_set1_epi32(0x7f800000));
        return _mm_castsi128_ps(_mm_blendv_epi8(r, u, nan));
    }
    const __m128 a = _mm_castsi128_ps(absu);
    const __m128i sign = _mm_and_si128(u, _mm_set1_epi32((int) 0x80000000u));
    const __m128i lsb = _mm_and_si128(_mm_srli_epi32(absu, 13), _mm_set1_epi32(1));
    const __m128i nrm = _mm_and_si128(_mm_add_epi32(absu, _mm_add_epi32(_mm_set1_epi32(0xfff), lsb)), _mm_set1_epi32((int) 0xffffe000u));
    const __m128 sub = _mm_sub_ps(_mm_add_ps(a, _mm_set1_ps(0.5f)), _mm_set1_ps(0.5f));
    const __m128 is_sub = _mm_cmplt_ps(a, _mm_set1_ps(6.103515625e-05f));
    __m128 q = _mm_blendv_ps(_mm_castsi128_ps(nrm), sub, is_sub);
    q = _mm_or_ps(q, _mm_castsi128_ps(sign));
    const __m128 keep = _mm_cmplt_ps(a, _mm_set1_ps(65520.0f));
    return _mm_blendv_ps(x, q, keep);
}
inline void round_buf(float* p, size_t n, int kind) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) _mm_storeu_ps(p + i, round4(_mm_loadu_ps(p + i), kind));
    if (i < n) {
        float t[4] = {0};
        for (size_t j = i; j < n; ++j) t[j - i] = p[j];
        _mm_storeu_ps(t, round4(_mm_loadu_ps(t), kind));
        for (size_t j = i; j < n; ++j) p[j] = t[j - i];
    }
}

inline __m128 h2f4(__m128i h) {
    const __m128i em = _mm_and_si128(h, _mm_set1_epi32(0x7fff));
    const __m128i sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
    const __m128i sh = _mm_slli_epi32(em, 13);
    const __m128 fin = _mm_mul_ps(_mm_castsi128_ps(sh), _mm_castsi128_ps(_mm_set1_epi32(0x77800000)));
    const __m128i inf = _mm_or_si128(sh, _mm_set1_epi32(0x7f800000));
    const __m128i big = _mm_cmpgt_epi32(em, _mm_set1_epi32(0x7bff));
    const __m128i r = _mm_blendv_epi8(_mm_castps_si128(fin), inf, big);
    return _mm_castsi128_ps(_mm_or_si128(r, sign));
}
inline float f16_one(uint16_t h) { return _mm_cvtss_f32(h2f4(_mm_cvtsi32_si128((int) h))); }
inline float bf16_one(uint16_t h) { return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128((int) ((uint32_t) h << 16)))); }

inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    int i = 0;
    if (kind == 2) {
        for (; i + 8 <= n; i += 8) {
            const __m128i w = _mm_loadu_si128((const __m128i*) (p + i));
            _mm256_storeu_ps(out + i, combine(h2f4(_mm_cvtepu16_epi32(w)), h2f4(_mm_cvtepu16_epi32(_mm_srli_si128(w, 8)))));
        }
        for (; i < n; ++i) out[i] = f16_one(p[i]);
    } else {
        const __m128i z = _mm_setzero_si128();
        for (; i + 8 <= n; i += 8) {
            const __m128i w = _mm_loadu_si128((const __m128i*) (p + i));
            _mm256_storeu_ps(out + i, combine(_mm_castsi128_ps(_mm_unpacklo_epi16(z, w)), _mm_castsi128_ps(_mm_unpackhi_epi16(z, w))));
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
#define STRATA_KN_NAME "avx"
#define STRATA_KN_GETTER cpu_ops_avx
#include "cpu_kernels_impl.h"

#else
namespace strata_vit {
const CpuOps& cpu_ops_avx() { return cpu_ops_scalar(); }
}
#endif
