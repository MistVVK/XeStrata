// tools/vision/own/cpu_kernels_neon.cpp - the Arm NEON tier (8 x 8 register tile: 16 accumulators and 8 A vectors of the 32 vector registers).  Baseline AArch64 only (Advanced
// SIMD and the F16 conversions are in every ARMv8-A CPU), so no ISA flags; it keeps the contract of the x86 tiers anyway (cpu_tier.h: no standard-library code).
#include "cpu_tier.h"      // only this and <arm_neon.h> (cpu_tier.h)

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace strata_vit {
namespace {
namespace kn {

// one 16-bit value to float, bit exact
inline float f16_one(uint16_t h) { return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0); }
inline float bf16_one(uint16_t h) { return vgetq_lane_f32(vreinterpretq_f32_u32(vdupq_n_u32((uint32_t) h << 16)), 0); }
// the helpers cpu_kernels_impl.h asks for, internal to this file
template <class T> inline T kmin(T a, T b) { return b < a ? b : a; }
template <class T> inline T kmax(T a, T b) { return a < b ? b : a; }
inline void kcopy(float* d, const float* s, int n) { for (int i = 0; i < n; ++i) d[i] = s[i]; }
inline void kzero(float* d, int n) { for (int i = 0; i < n; ++i) d[i] = 0.0f; }

struct V {
    static constexpr int W = 4;
    float32x4_t v;
    V() : v(vdupq_n_f32(0.0f)) {}
    V(float32x4_t x) : v(x) {}
    static V load(const float* p) { return vld1q_f32(p); }
    void store(float* p) const { vst1q_f32(p, v); }
    static V set1(float f) { return vdupq_n_f32(f); }
    static V zero() { return vdupq_n_f32(0.0f); }
    static float sqrt1(float x) { return vgetq_lane_f32(vsqrtq_f32(vdupq_n_f32(x)), 0); }      // IEEE square root, correctly rounded
    static V fma(V a, V b, V c) { return vfmaq_f32(c.v, a.v, b.v); }
    static float reduce_add(V a) { return vaddvq_f32(a.v); }
    static float reduce_max(V a) { return vmaxvq_f32(a.v); }
    friend V operator+(V a, V b) { return vaddq_f32(a.v, b.v); }
    friend V operator-(V a, V b) { return vsubq_f32(a.v, b.v); }
    friend V operator*(V a, V b) { return vmulq_f32(a.v, b.v); }
    friend V operator/(V a, V b) { return vdivq_f32(a.v, b.v); }
};
inline V vmax(V a, V b) { return vmaxq_f32(a.v, b.v); }

// The product's micro-kernel with lane-indexed FMAs: four k steps take one vector load of each A row instead of four scalar broadcasts.  The same
// summation order as the generic micro of cpu_kernels_impl.h (each accumulator gets a[r][k] * b[k] in k order, fused), so the results are the same.
#define STRATA_KN_HAS_MICRO 1
#define KN_STEP(L)                                                                                     \
    {                                                                                                  \
        const float32x4_t b0 = vld1q_f32(B + (size_t) (k + L) * ldb), b1 = vld1q_f32(B + (size_t) (k + L) * ldb + 4); \
        for (int r = 0; r < MR; ++r) {                                                                 \
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, a[r], L);                                       \
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, a[r], L);                                       \
        }                                                                                              \
    }
template <int MR>
inline void micro(const float* A, size_t lda, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    float32x4_t acc[MR][2];
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) {
            float32x4_t v = vdupq_n_f32(0.0f);
            if (init != 0) v = vld1q_f32(C + (size_t) r * ldc + c * 4);
            if (init != 1 && bias) v = vaddq_f32(v, vld1q_f32(bias + c * 4));
            acc[r][c] = v;
        }
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t a[MR];
        for (int r = 0; r < MR; ++r) a[r] = vld1q_f32(A + (size_t) r * lda + k);
        KN_STEP(0) KN_STEP(1) KN_STEP(2) KN_STEP(3)
    }
    for (; k < K; ++k) {
        const float32x4_t b0 = vld1q_f32(B + (size_t) k * ldb), b1 = vld1q_f32(B + (size_t) k * ldb + 4);
        for (int r = 0; r < MR; ++r) {
            const float32x4_t a = vdupq_n_f32(A[(size_t) r * lda + k]);
            acc[r][0] = vfmaq_f32(acc[r][0], a, b0);
            acc[r][1] = vfmaq_f32(acc[r][1], a, b1);
        }
    }
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) vst1q_f32(C + (size_t) r * ldc + c * 4, acc[r][c]);
}
#undef KN_STEP

// expf (Cephes), the same steps and constants as the x86 tiers: x = n ln2 + r, exp(x) = 2^n * p(r)
inline V vexp(V xv) {
    float32x4_t x = vminq_f32(vmaxq_f32(xv.v, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
    const float32x4_t fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.0f)), y, vmulq_f32(x, x));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(fx), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

// 4 floats rounded to the 16-bit type and back (BF16: integer round to nearest even as the x86 tiers; F16: the hardware conversion pair, nearest even)
inline float32x4_t round4(float32x4_t x, int kind) {
    if (kind == 2) return vcvt_f32_f16(vcvt_f16_f32(x));
    uint32x4_t u = vreinterpretq_u32_f32(x);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    u = vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7fff), lsb));
    return vreinterpretq_f32_u32(vandq_u32(u, vdupq_n_u32(0xffff0000u)));
}
inline void round_buf(float* p, size_t n, int kind) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(p + i, round4(vld1q_f32(p + i), kind));
    if (i < n) {
        float t[4] = {0, 0, 0, 0};
        for (size_t j = i; j < n; ++j) t[j - i] = p[j];
        vst1q_f32(t, round4(vld1q_f32(t), kind));
        for (size_t j = i; j < n; ++j) p[j] = t[j - i];
    }
}

inline void conv16(const void* Wm, int kind, size_t off, int n, float* out) {
    const uint16_t* p = (const uint16_t*) Wm + off;
    int i = 0;
    if (kind == 2) {
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p + i))));
        for (; i < n; ++i) out[i] = f16_one(p[i]);
    } else {
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p + i), 16)));
        for (; i < n; ++i) out[i] = bf16_one(p[i]);
    }
}

}  // namespace kn
}  // namespace
}  // namespace strata_vit

#define STRATA_KN_MR 8
#define KN_MIN(a, b) kmin(a, b)
#define KN_MAX(a, b) kmax(a, b)
#define KN_COPY(d, s, n) kcopy((d), (s), (n))
#define KN_ZERO(d, n) kzero((d), (n))
#define STRATA_KN_NAME "neon"
#define STRATA_KN_GETTER cpu_ops_neon
#include "cpu_kernels_impl.h"

#else   // not AArch64: the tier does not exist, the dispatcher never asks for it
namespace strata_vit {
const CpuOps& cpu_ops_neon() { return cpu_ops_scalar(); }
}
#endif
