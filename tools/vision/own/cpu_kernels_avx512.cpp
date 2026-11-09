// tools/vision/own/cpu_kernels_avx512.cpp - the AVX-512 tier (12 x 32 register tile: 24 accumulators).  Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl
// -mavx2 -mfma -mf16c (/arch:AVX512 on MSVC); only called when quant::isa_ok(Avx512).
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
    static constexpr int W = 16;
    __m512 v;
    V() : v(_mm512_setzero_ps()) {}
    V(__m512 x) : v(x) {}
    static V load(const float* p) { return _mm512_loadu_ps(p); }
    void store(float* p) const { _mm512_storeu_ps(p, v); }
    static V set1(float f) { return _mm512_set1_ps(f); }
    static V zero() { return _mm512_setzero_ps(); }
    static float sqrt1(float x) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))); }     // sqrtss: the correctly rounded root, as std::sqrt
    static V fma(V a, V b, V c) { return _mm512_fmadd_ps(a.v, b.v, c.v); }
    static float reduce_add(V a) { return _mm512_reduce_add_ps(a.v); }
    static float reduce_max(V a) { return _mm512_reduce_max_ps(a.v); }
    friend V operator+(V a, V b) { return _mm512_add_ps(a.v, b.v); }
    friend V operator-(V a, V b) { return _mm512_sub_ps(a.v, b.v); }
    friend V operator*(V a, V b) { return _mm512_mul_ps(a.v, b.v); }
    friend V operator/(V a, V b) { return _mm512_div_ps(a.v, b.v); }
};
inline V vmax(V a, V b) { return _mm512_max_ps(a.v, b.v); }

inline V vexp(V xv) {
    __m512 x = _mm512_min_ps(_mm512_max_ps(xv.v, _mm512_set1_ps(-87.3f)), _mm512_set1_ps(88.3f));
    __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
    __m512 y = _mm512_set1_ps(1.9875691500e-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(x, x), _mm512_add_ps(x, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(y, fx);
}

inline __m512 round16v(__m512 x, int kind) {
    if (kind == 2) return _mm512_cvtph_ps(_mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m512i u = _mm512_castps_si512(x);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb));
    return _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int) 0xffff0000u)));
}
inline void round_buf(float* p, size_t n, int kind) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(p + i, round16v(_mm512_loadu_ps(p + i), kind));
    if (i < n) {
        float t[16] = {0};
        for (size_t j = i; j < n; ++j) t[j - i] = p[j];
        _mm512_storeu_ps(t, round16v(_mm512_loadu_ps(t), kind));
        for (size_t j = i; j < n; ++j) p[j] = t[j - i];
    }
}

inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    int i = 0;
    if (kind == 2) {
        for (; i + 16 <= n; i += 16) _mm512_storeu_ps(out + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (p + i))));
        for (; i < n; ++i) out[i] = f16_one(p[i]);
    } else {
        for (; i + 16 <= n; i += 16) {
            const __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) (p + i))), 16);
            _mm512_storeu_ps(out + i, _mm512_castsi512_ps(w));
        }
        for (; i < n; ++i) out[i] = bf16_one(p[i]);
    }
}

}  // namespace kn
}  // namespace
}  // namespace strata_vit

#define STRATA_KN_MR 12
#define KN_MIN(a, b) kmin(a, b)
#define KN_MAX(a, b) kmax(a, b)
#define KN_COPY(d, s, n) kcopy((d), (s), (n))
#define KN_ZERO(d, n) kzero((d), (n))
#define STRATA_KN_NAME "avx512"
#define STRATA_KN_GETTER cpu_ops_avx512
#include "cpu_kernels_impl.h"

#else
namespace strata_vit {
const CpuOps& cpu_ops_avx512() { return cpu_ops_scalar(); }
}
#endif
