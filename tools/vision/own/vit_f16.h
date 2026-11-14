// tools/vision/own/vit_f16.h - the GPU reader's rules for holding a BF16 matrix as F16 (vit_gpu.cu), plain C++ so the CPU unit test
// (test_cpu_kernels.cpp) checks them without a GPU.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIP__)
#define VIT_F16_HD __host__ __device__
#else
#define VIT_F16_HD
#endif

namespace strata_vit {

// how a BF16 matrix reaches the GPU's products (STRATA_VIT_BF16_AS_F16)
enum Bf16Mode {
    BF16_STORED = 0,     // BF16 weights, BF16 inputs
    BF16_ROUNDED = 1,    // weights as F16, inputs rounded to BF16 and then written as F16 (the Pascal / gfx1151 kind: the products see BF16 values)
    BF16_TO_F16 = 2,     // weights as F16, inputs written as F16 (10-bit mantissa instead of BF16's 7)
};

// the mode from STRATA_VIT_BF16_AS_F16 (env, nullptr when unset) and the card's default: "2" is BF16_TO_F16; "0", an empty or a non-numeric value
// BF16_STORED and every other number BF16_ROUNDED, as before the F16 kind existed
inline int vit_gpu_bf16_mode(const char* env, int card_default) {
    if (!env) return card_default;
    long v = 0;
    bool neg = false;
    const char* p = env;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '+' || *p == '-') { neg = *p == '-'; ++p; }
    for (; *p >= '0' && *p <= '9'; ++p) { v = v * 10 + (*p - '0'); if (v > 1000000) break; }
    if (neg) v = -v;
    return v == 0 ? BF16_STORED : v == 2 ? BF16_TO_F16 : BF16_ROUNDED;
}

// true when the BF16 value b is an F16 value: finite, |v| <= 65504, and no mantissa bit below F16's step (2^-24 under 2^-14).  Every BF16 value
// with an exponent from -14 to 15 is one (the largest, 65,280, included); below 2^-14 only those whose lowest set bit is at 2^-24 or above.
VIT_F16_HD inline bool bf16_exact_in_f16(uint16_t b) {
    const int e = (b >> 7) & 0xff, m = b & 0x7f;
    if (e == 0xff) return false;                 // inf, nan
    if (e == 0) return m == 0;                   // +-0; BF16 subnormals (under 1.2e-38) are not F16 values
    const int E = e - 127;
    if (E > 15) return false;                    // 65,536 and above
    if (E >= -17) return true;                   // the 8 mantissa bits (with the leading one) land at 2^(E-7) >= 2^-24
    if (E < -24) return false;
    const int s = -17 - E;                       // 1..7 bits under 2^-24
    return (m & ((1 << s) - 1)) == 0;
}

// an activation clamped to F16's finite range (NaN stays NaN), before it is written as F16
VIT_F16_HD inline float f16_clamp(float v) { return v > 65504.0f ? 65504.0f : v < -65504.0f ? -65504.0f : v; }

// true when the BF16 value b lies beyond F16's range (|v| >= 65,536 by its exponent, inf, nan): a matrix with one stays BF16
VIT_F16_HD inline bool bf16_beyond_f16(uint16_t b) { return ((b >> 7) & 0xff) > 127 + 15; }
inline long long bf16_count_beyond_f16(const uint16_t* w, long long n) {
    long long c = 0;
    for (long long i = 0; i < n; ++i) c += bf16_beyond_f16(w[i]);
    return c;
}

// whether a BF16 matrix may be held as F16 in BF16_TO_F16 mode: no weight beyond F16's range.  Weights under 2^-14 that F16 holds only to its
// 2^-24 step (an error of at most 2^-25 each) are counted for the trace but do not keep the matrix in BF16.
inline bool bf16_matrix_fits_f16(long long n_overflow) { return n_overflow == 0; }

// the absolute error of a BF16 value F16 cannot hold exactly but holds inside its range (|v| < 2^-14): F16 rounds it to nearest even at its
// 2^-24 step, so the error is at most 2^-25.  0 for an F16 value or one beyond the range (those are not held as F16).
inline double bf16_f16_abs_error(uint16_t b) {
    const int e = (b >> 7) & 0xff, m = b & 0x7f;
    if (bf16_exact_in_f16(b) || bf16_beyond_f16(b)) return 0.0;
    if (e == 0) return std::ldexp((double) m, -133);                       // a BF16 subnormal: F16 holds 0
    const double q = std::ldexp((double) (0x80 | m), e - 127 - 7 + 24);   // the value in units of 2^-24
    return std::ldexp(std::fabs(q - std::nearbyint(q)), -24);
}

// counts of one BF16 matrix: weights F16 cannot hold exactly, of those the ones beyond its range (|w| > 65504, inf, nan), and the largest
// absolute error of the others
struct Bf16F16Count { long long not_exact = 0, overflow = 0; double max_abs_error = 0.0; };
inline Bf16F16Count bf16_count_f16(const uint16_t* w, long long n) {
    Bf16F16Count c;
    for (long long i = 0; i < n; ++i) {
        if (bf16_exact_in_f16(w[i])) continue;
        ++c.not_exact;
        if (bf16_beyond_f16(w[i])) { ++c.overflow; continue; }
        const double err = bf16_f16_abs_error(w[i]);
        if (err > c.max_abs_error) c.max_abs_error = err;
    }
    return c;
}

}  // namespace strata_vit
