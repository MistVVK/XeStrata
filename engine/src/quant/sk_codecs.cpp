// src/quant/sk_codecs.cpp - Native .strata v1.0 S_K codecs & AVX-512 VNNI / AVX2 / Scalar dot products
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
#include <immintrin.h>
#define STRATA_SK_AVX512_VNNI 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define STRATA_SK_AVX2 1
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#define STRATA_SK_NEON_DOTPROD 1
#endif

namespace strata::quant::sk {
namespace {

inline float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t e = 127 - 14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Exact match to S125_CODEBOOK in tools/squant/codecs.py
alignas(64) constexpr int8_t kS125Codebook[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1,  1}, {-1, -1,  1, -1}, {-1, -1,  1,  1},
    {-1,  1, -1, -1}, {-1,  1, -1,  1}, {-1,  1,  1, -1}, {-1,  1,  1,  1},
    { 1, -1, -1, -1}, { 1, -1, -1,  1}, { 1, -1,  1, -1}, { 1, -1,  1,  1},
    { 1,  1, -1, -1}, { 1,  1, -1,  1}, { 1,  1,  1, -1}, { 1,  1,  1,  1},
    { 2,  0,  0,  0}, {-2,  0,  0,  0}, { 0,  2,  0,  0}, { 0, -2,  0,  0},
    { 0,  0,  2,  0}, { 0,  0, -2,  0}, { 0,  0,  0,  2}, { 0,  0,  0, -2},
    { 0,  0,  0,  0}, { 2,  2,  0,  0}, {-2, -2,  0,  0}, { 0,  0,  2,  2},
    { 0,  0, -2, -2}, { 2, -2,  0,  0}, { 0,  0,  2, -2}, { 2,  0,  2,  0},
};

struct Trit5Table {
    int8_t v[256][5]{};
    constexpr Trit5Table() {
        for (int b = 0; b < 243; ++b) {
            int x = b;
            for (int j = 0; j < 5; ++j) {
                v[b][j] = static_cast<int8_t>((x % 3) - 1);
                x /= 3;
            }
        }
    }
};
constexpr Trit5Table kT5{};

#if defined(STRATA_SK_NEON_DOTPROD)
static inline bool sk_has_neon_dp() noexcept {
    static const bool v = []() {
#if defined(__linux__)
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__ARM_FEATURE_DOTPROD) || defined(__APPLE__)
        return true;
#else
        return false;
#endif
    }();
    return v;
}
#if !defined(__ARM_FEATURE_DOTPROD) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("+dotprod")))
#endif
static inline int32_t dot_u8_i8_64_neon_dp(const uint8_t* w, const int8_t* x) noexcept {
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vld1q_u8(w + 0)), vld1q_s8(x + 0));
    acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vld1q_u8(w + 16)), vld1q_s8(x + 16));
    acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vld1q_u8(w + 32)), vld1q_s8(x + 32));
    acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vld1q_u8(w + 48)), vld1q_s8(x + 48));
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}
#if !defined(__ARM_FEATURE_DOTPROD) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("+dotprod")))
#endif
static inline int32_t dot_i8_i8_64_neon_dp(const int8_t* w, const int8_t* x) noexcept {
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    acc0 = vdotq_s32(acc0, vld1q_s8(w + 0), vld1q_s8(x + 0));
    acc1 = vdotq_s32(acc1, vld1q_s8(w + 16), vld1q_s8(x + 16));
    acc0 = vdotq_s32(acc0, vld1q_s8(w + 32), vld1q_s8(x + 32));
    acc1 = vdotq_s32(acc1, vld1q_s8(w + 48), vld1q_s8(x + 48));
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}
#endif

inline int32_t dot_u8_i8_64(const uint8_t* w, const int8_t* x) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    __m512i vw = _mm512_loadu_si512(w);
    __m512i vx = _mm512_loadu_si512(x);
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vw, vx);
    return _mm512_reduce_add_epi32(acc);
#elif defined(STRATA_SK_AVX2)
    __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32));
    __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 32));
    __m256i p0 = _mm256_maddubs_epi16(w0, x0);
    __m256i p1 = _mm256_maddubs_epi16(w1, x1);
    __m256i ones = _mm256_set1_epi16(1);
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(p0, ones), _mm256_madd_epi16(p1, ones));
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i a = _mm_add_epi32(lo, hi);
    a = _mm_hadd_epi32(a, a);
    a = _mm_hadd_epi32(a, a);
    return _mm_cvtsi128_si32(a);
#elif defined(STRATA_SK_NEON_DOTPROD)
    if (sk_has_neon_dp()) return dot_u8_i8_64_neon_dp(w, x);
    int32_t s = 0;
    for (int i = 0; i < 64; ++i) s += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < 64; ++i) s += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
    return s;
#endif
}

inline int32_t dot_i8_i8_64(const int8_t* w, const int8_t* x) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    // Shift w from [-127..127] to u8 [1..255] via XOR 0x80 (or add 128), then subtract 128*sum(x)
    __m512i vw = _mm512_loadu_si512(w);
    __m512i vx = _mm512_loadu_si512(x);
    __m512i bias = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512i vwu = _mm512_xor_si512(vw, bias);  // w + 128 in u8
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vwu, vx);
    __m512i sumx = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_set1_epi8(1), vx);
    return _mm512_reduce_add_epi32(acc) - 128 * _mm512_reduce_add_epi32(sumx);
#elif defined(STRATA_SK_NEON_DOTPROD)
    if (sk_has_neon_dp()) return dot_i8_i8_64_neon_dp(w, x);
    int32_t s = 0;
    for (int i = 0; i < 64; ++i) s += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < 64; ++i) s += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
    return s;
#endif
}

inline void unpack_s2k_sb(const uint8_t* pl, uint8_t q[256]) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    // Load 64 bytes (256 x 2-bit values): unpack 4 lanes of 64 bytes in 8 instructions!
    const __m512i v = _mm512_loadu_si512(pl);
    const __m512i m3 = _mm512_set1_epi8(0x03);
    // Group order: byte i holds q[4i+0..3]; use 16-byte lane-wise shift-and-mask then interleave,
    // OR unpack 16 bytes -> 64 bytes via 4x _mm512_cvtepu8_epi32 + shifts!
    for (int c = 0; c < 4; ++c) {
        __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pl + c * 16));
        __m512i w = _mm512_cvtepu8_epi32(b16);  // 16 x uint32 (each holds 4 x 2-bit weights)
        __m512i q0 = _mm512_and_si512(w, _mm512_set1_epi32(0x03));
        __m512i q1 = _mm512_and_si512(_mm512_srli_epi32(w, 2), _mm512_set1_epi32(0x03));
        __m512i q2 = _mm512_and_si512(_mm512_srli_epi32(w, 4), _mm512_set1_epi32(0x03));
        __m512i q3 = _mm512_srli_epi32(w, 6);
        __m512i packed = _mm512_or_si512(
            _mm512_or_si512(q0, _mm512_slli_epi32(q1, 8)),
            _mm512_or_si512(_mm512_slli_epi32(q2, 16), _mm512_slli_epi32(q3, 24)));
        _mm512_storeu_si512(q + c * 64, packed);
    }
    (void)v; (void)m3;
#else
    for (int i = 0; i < 64; ++i) {
        const uint8_t b = pl[i];
        q[4 * i + 0] = b & 0x03u;
        q[4 * i + 1] = (b >> 2) & 0x03u;
        q[4 * i + 2] = (b >> 4) & 0x03u;
        q[4 * i + 3] = (b >> 6) & 0x03u;
    }
#endif
}

inline void unpack_s3k_sb(const uint8_t* pl, uint8_t q[256]) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    unpack_s2k_sb(pl, q);
    const uint8_t* p_hi = pl + 64;
    const __m512i bit4 = _mm512_set1_epi8(4);
    for (int c = 0; c < 4; ++c) {
        uint64_t mask64;
        std::memcpy(&mask64, p_hi + c * 8, 8);
        __m512i vq = _mm512_loadu_si512(q + c * 64);
        vq = _mm512_mask_or_epi32(vq, static_cast<__mmask16>(0), vq, vq); // no-op, use byte mask below
        __m512i hi = _mm512_maskz_mov_epi8(static_cast<__mmask64>(mask64), bit4);
        _mm512_storeu_si512(q + c * 64, _mm512_or_si512(vq, hi));
    }
#else
    const uint8_t* p_lo = pl;
    const uint8_t* p_hi = pl + 64;
    for (int i = 0; i < 64; ++i) {
        const uint8_t b = p_lo[i];
        q[4 * i + 0] = b & 0x03u;
        q[4 * i + 1] = (b >> 2) & 0x03u;
        q[4 * i + 2] = (b >> 4) & 0x03u;
        q[4 * i + 3] = (b >> 6) & 0x03u;
    }
    for (int i = 0; i < 32; ++i) {
        const uint8_t h = p_hi[i];
        for (int b = 0; b < 8; ++b) {
            q[8 * i + b] |= static_cast<uint8_t>(((h >> b) & 1u) << 2);
        }
    }
#endif
}

inline void unpack_s4k_sb(const uint8_t* pl, uint8_t q[256]) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    for (int c = 0; c < 4; ++c) {
        __m256i b32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pl + c * 32));
        __m512i w = _mm512_cvtepu8_epi16(b32);  // 32 x uint16
        __m512i q0 = _mm512_and_si512(w, _mm512_set1_epi16(0x000F));
        __m512i q1 = _mm512_slli_epi16(_mm512_srli_epi16(w, 4), 8);
        _mm512_storeu_si512(q + c * 64, _mm512_or_si512(q0, q1));
    }
#else
    for (int i = 0; i < 128; ++i) {
        const uint8_t b = pl[i];
        q[2 * i + 0] = b & 0x0Fu;
        q[2 * i + 1] = (b >> 4) & 0x0Fu;
    }
#endif
}

inline void unpack_s5k_sb(const uint8_t* pl, uint8_t q[256]) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    unpack_s4k_sb(pl, q);
    const uint8_t* p_hi = pl + 128;
    const __m512i bit16 = _mm512_set1_epi8(16);
    for (int c = 0; c < 4; ++c) {
        uint64_t mask64;
        std::memcpy(&mask64, p_hi + c * 8, 8);
        __m512i vq = _mm512_loadu_si512(q + c * 64);
        __m512i hi = _mm512_maskz_mov_epi8(static_cast<__mmask64>(mask64), bit16);
        _mm512_storeu_si512(q + c * 64, _mm512_or_si512(vq, hi));
    }
#else
    const uint8_t* p_lo = pl;
    const uint8_t* p_hi = pl + 128;
    for (int i = 0; i < 128; ++i) {
        const uint8_t b = p_lo[i];
        q[2 * i + 0] = b & 0x0Fu;
        q[2 * i + 1] = (b >> 4) & 0x0Fu;
    }
    for (int i = 0; i < 32; ++i) {
        const uint8_t h = p_hi[i];
        for (int b = 0; b < 8; ++b) {
            q[8 * i + b] |= static_cast<uint8_t>(((h >> b) & 1u) << 4);
        }
    }
#endif
}

inline void unpack_s6k_sb(const uint8_t* pl, uint8_t q[256]) noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    unpack_s4k_sb(pl, q);
    alignas(64) uint8_t hi[256];
    unpack_s2k_sb(pl + 128, hi);
    for (int c = 0; c < 4; ++c) {
        __m512i vq = _mm512_loadu_si512(q + c * 64);
        __m512i vh = _mm512_slli_epi16(_mm512_loadu_si512(hi + c * 64), 4);
        vh = _mm512_and_si512(vh, _mm512_set1_epi8(0x30));
        _mm512_storeu_si512(q + c * 64, _mm512_or_si512(vq, vh));
    }
#else
    const uint8_t* p_lo = pl;
    const uint8_t* p_hi = pl + 128;
    for (int i = 0; i < 128; ++i) {
        const uint8_t b = p_lo[i];
        q[2 * i + 0] = b & 0x0Fu;
        q[2 * i + 1] = (b >> 4) & 0x0Fu;
    }
    for (int i = 0; i < 64; ++i) {
        const uint8_t h = p_hi[i];
        q[4 * i + 0] |= static_cast<uint8_t>((h & 0x03u) << 4);
        q[4 * i + 1] |= static_cast<uint8_t>(((h >> 2) & 0x03u) << 4);
        q[4 * i + 2] |= static_cast<uint8_t>(((h >> 4) & 0x03u) << 4);
        q[4 * i + 3] |= static_cast<uint8_t>(((h >> 6) & 0x03u) << 4);
    }
#endif
}

inline void unpack_s15_sb(const uint8_t* pl, int8_t q[256]) noexcept {
    for (int i = 0; i < 51; ++i) {
        const int8_t* t = kT5.v[pl[i]];
        q[5 * i + 0] = t[0];
        q[5 * i + 1] = t[1];
        q[5 * i + 2] = t[2];
        q[5 * i + 3] = t[3];
        q[5 * i + 4] = t[4];
    }
    q[255] = kT5.v[pl[51]][0];
}

inline void unpack_s125_sb(const uint8_t* pl, int8_t q[256], float fine[4]) noexcept {
    for (int g = 0; g < 8; ++g) {
        const uint8_t* p = pl + g * 5;
        const uint64_t bits40 =
            (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
            ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
        for (int j = 0; j < 8; ++j) {
            const uint32_t idx = static_cast<uint32_t>((bits40 >> (5 * j)) & 0x1Fu);
            const int8_t* c = kS125Codebook[idx];
            const int base = (g * 8 + j) * 4;
            q[base + 0] = c[0];
            q[base + 1] = c[1];
            q[base + 2] = c[2];
            q[base + 3] = c[3];
        }
    }
    const uint8_t f0 = pl[40], f1 = pl[41];
    fine[0] = 0.5f + static_cast<float>(f0 & 0x0Fu) / 15.0f;
    fine[1] = 0.5f + static_cast<float>((f0 >> 4) & 0x0Fu) / 15.0f;
    fine[2] = 0.5f + static_cast<float>(f1 & 0x0Fu) / 15.0f;
    fine[3] = 0.5f + static_cast<float>((f1 >> 4) & 0x0Fu) / 15.0f;
}

}  // namespace

const char* active_sk_isa() noexcept {
#if defined(STRATA_SK_AVX512_VNNI)
    return "avx512-vnni";
#elif defined(STRATA_SK_AVX2)
    return "avx2";
#elif defined(STRATA_SK_NEON_DOTPROD)
    return sk_has_neon_dp() ? "neon-dotprod" : "scalar";
#else
    return "scalar";
#endif
}

void quantize_q8_1_g64(const float* x, BlockQ8_1_G64* out, size_t n) noexcept {
    const size_t nb = n / SUB_G;
    for (size_t b = 0; b < nb; ++b) {
        const float* xb = x + b * SUB_G;
        float amax = 0.0f;
        for (int i = 0; i < SUB_G; ++i) {
            amax = std::max(amax, std::fabs(xb[i]));
        }
        const float d = amax / 127.0f;
        const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
        int32_t sum = 0;
        for (int i = 0; i < SUB_G; ++i) {
            int v = static_cast<int>(std::nearbyintf(xb[i] * inv));
            v = std::clamp(v, -127, 127);
            out[b].qs[i] = static_cast<int8_t>(v);
            sum += v;
        }
        out[b].d = d;
        out[b].s = d * static_cast<float>(sum);
    }
}

void fwht64_f32(const float* in, float* out, size_t n) noexcept {
    const size_t nb = n / 64;
    constexpr float kInv8 = 0.125f;
    for (size_t b = 0; b < nb; ++b) {
        float tmp[64];
        std::memcpy(tmp, in + b * 64, 64 * sizeof(float));
        for (int len = 1; len < 64; len <<= 1) {
            for (int i = 0; i < 64; i += (len << 1)) {
                for (int j = 0; j < len; ++j) {
                    const float u = tmp[i + j];
                    const float v = tmp[i + j + len];
                    tmp[i + j] = u + v;
                    tmp[i + j + len] = u - v;
                }
            }
        }
        float* dst = out + b * 64;
        for (int i = 0; i < 64; ++i) dst[i] = tmp[i] * kInv8;
    }
}

void dequant_sk_row(const uint8_t* raw, SkCode code, float* out, size_t n) noexcept {
    const size_t n_sb = n / QK_S;
    const size_t sb_b = superblock_bytes(code);
    for (size_t s = 0; s < n_sb; ++s) {
        const uint8_t* blk = raw + s * sb_b;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float D = fp16_to_fp32(d_bits);
        const uint8_t* u = blk + 2;
        const uint8_t* pl = blk + 6;
        float* dst = out + s * QK_S;

        if (code == SkCode::S8_K) {
            const int8_t* q = reinterpret_cast<const int8_t*>(pl);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                for (int i = 0; i < SUB_G; ++i) {
                    dst[g * SUB_G + i] = static_cast<float>(q[g * SUB_G + i]) * eff;
                }
            }
        } else if (code == SkCode::S1_5) {
            alignas(64) int8_t q[256];
            unpack_s15_sb(pl, q);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                for (int i = 0; i < SUB_G; ++i) {
                    dst[g * SUB_G + i] = static_cast<float>(q[g * SUB_G + i]) * eff;
                }
            }
        } else if (code == SkCode::S1_25) {
            alignas(64) int8_t q[256];
            float fine[4];
            unpack_s125_sb(pl, q, fine);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f) * fine[g];
                for (int i = 0; i < SUB_G; ++i) {
                    dst[g * SUB_G + i] = static_cast<float>(q[g * SUB_G + i]) * eff;
                }
            }
        } else {
            alignas(64) uint8_t q[256];
            float half = 0.0f;
            switch (code) {
                case SkCode::S2_K: unpack_s2k_sb(pl, q); half = 1.5f;  break;
                case SkCode::S3_K: unpack_s3k_sb(pl, q); half = 3.5f;  break;
                case SkCode::S4_K: unpack_s4k_sb(pl, q); half = 7.5f;  break;
                case SkCode::S5_K: unpack_s5k_sb(pl, q); half = 15.5f; break;
                case SkCode::S6_K: unpack_s6k_sb(pl, q); half = 31.5f; break;
                default: return;
            }
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                for (int i = 0; i < SUB_G; ++i) {
                    dst[g * SUB_G + i] = (static_cast<float>(q[g * SUB_G + i]) - half) * eff;
                }
            }
        }
    }
}

float dot_sk_q8_1_row(const uint8_t* raw, SkCode code, const BlockQ8_1_G64* x, size_t n) noexcept {
    const size_t n_sb = n / QK_S;
    const size_t sb_b = superblock_bytes(code);
    float total = 0.0f;

    for (size_t s = 0; s < n_sb; ++s) {
        const uint8_t* blk = raw + s * sb_b;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float D = fp16_to_fp32(d_bits);
        const uint8_t* u = blk + 2;
        const uint8_t* pl = blk + 6;
        const BlockQ8_1_G64* xb = x + s * NUM_SUB;

        if (code == SkCode::S8_K) {
            const int8_t* q = reinterpret_cast<const int8_t*>(pl);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                const int32_t isum = dot_i8_i8_64(q + g * SUB_G, xb[g].qs);
                total += eff * xb[g].d * static_cast<float>(isum);
            }
        } else if (code == SkCode::S1_5) {
            alignas(64) int8_t q[256];
            unpack_s15_sb(pl, q);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                const int32_t isum = dot_i8_i8_64(q + g * SUB_G, xb[g].qs);
                total += eff * xb[g].d * static_cast<float>(isum);
            }
        } else if (code == SkCode::S1_25) {
            alignas(64) int8_t q[256];
            float fine[4];
            unpack_s125_sb(pl, q, fine);
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f) * fine[g];
                const int32_t isum = dot_i8_i8_64(q + g * SUB_G, xb[g].qs);
                total += eff * xb[g].d * static_cast<float>(isum);
            }
        } else {
            alignas(64) uint8_t q[256];
            float half = 0.0f;
            switch (code) {
                case SkCode::S2_K: unpack_s2k_sb(pl, q); half = 1.5f;  break;
                case SkCode::S3_K: unpack_s3k_sb(pl, q); half = 3.5f;  break;
                case SkCode::S4_K: unpack_s4k_sb(pl, q); half = 7.5f;  break;
                case SkCode::S5_K: unpack_s5k_sb(pl, q); half = 15.5f; break;
                case SkCode::S6_K: unpack_s6k_sb(pl, q); half = 31.5f; break;
                default: return 0.0f;
            }
            for (int g = 0; g < NUM_SUB; ++g) {
                const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                const int32_t isum = dot_u8_i8_64(q + g * SUB_G, xb[g].qs);
                total += eff * (xb[g].d * static_cast<float>(isum) - half * xb[g].s);
            }
        }
    }
    return total;
}

void gemm_sk_q8_1(
    const uint8_t* weights,
    SkCode code,
    const BlockQ8_1_G64* xs,
    float* out,
    size_t nrows,
    size_t ncols,
    size_t m) noexcept {
    const size_t n_sb = ncols / QK_S;
    const size_t sb_b = superblock_bytes(code);
    const size_t row_bytes = n_sb * sb_b;
    const size_t x_stride = ncols / SUB_G;

    // Decode each weight row once and dot against all m tokens (2x-8x weight reuse!)
    for (size_t r = 0; r < nrows; ++r) {
        const uint8_t* row = weights + r * row_bytes;
        if (m == 1) {
            out[r] = dot_sk_q8_1_row(row, code, xs, ncols);
            continue;
        }
        float acc[16]{};
        const size_t m_cap = std::min<size_t>(m, 16);
        for (size_t s = 0; s < n_sb; ++s) {
            const uint8_t* blk = row + s * sb_b;
            uint16_t d_bits;
            std::memcpy(&d_bits, blk, 2);
            const float D = fp16_to_fp32(d_bits);
            const uint8_t* u = blk + 2;
            const uint8_t* pl = blk + 6;

            if (code == SkCode::S8_K) {
                const int8_t* q = reinterpret_cast<const int8_t*>(pl);
                for (int g = 0; g < NUM_SUB; ++g) {
                    const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                    for (size_t t = 0; t < m_cap; ++t) {
                        const auto& xb = xs[t * x_stride + s * NUM_SUB + g];
                        acc[t] += eff * xb.d * static_cast<float>(dot_i8_i8_64(q + g * SUB_G, xb.qs));
                    }
                }
            } else if (code == SkCode::S1_5 || code == SkCode::S1_25) {
                alignas(64) int8_t q[256];
                float fine[4] = {1.f, 1.f, 1.f, 1.f};
                if (code == SkCode::S1_5) unpack_s15_sb(pl, q);
                else unpack_s125_sb(pl, q, fine);
                for (int g = 0; g < NUM_SUB; ++g) {
                    const float eff = D * (static_cast<float>(u[g]) / 255.0f) * fine[g];
                    for (size_t t = 0; t < m_cap; ++t) {
                        const auto& xb = xs[t * x_stride + s * NUM_SUB + g];
                        acc[t] += eff * xb.d * static_cast<float>(dot_i8_i8_64(q + g * SUB_G, xb.qs));
                    }
                }
            } else {
                alignas(64) uint8_t q[256];
                float half = 0.0f;
                switch (code) {
                    case SkCode::S2_K: unpack_s2k_sb(pl, q); half = 1.5f;  break;
                    case SkCode::S3_K: unpack_s3k_sb(pl, q); half = 3.5f;  break;
                    case SkCode::S4_K: unpack_s4k_sb(pl, q); half = 7.5f;  break;
                    case SkCode::S5_K: unpack_s5k_sb(pl, q); half = 15.5f; break;
                    case SkCode::S6_K: unpack_s6k_sb(pl, q); half = 31.5f; break;
                    default: break;
                }
                for (int g = 0; g < NUM_SUB; ++g) {
                    const float eff = D * (static_cast<float>(u[g]) / 255.0f);
                    for (size_t t = 0; t < m_cap; ++t) {
                        const auto& xb = xs[t * x_stride + s * NUM_SUB + g];
                        const int32_t isum = dot_u8_i8_64(q + g * SUB_G, xb.qs);
                        acc[t] += eff * (xb.d * static_cast<float>(isum) - half * xb.s);
                    }
                }
            }
        }
        for (size_t t = 0; t < m_cap; ++t) out[t * nrows + r] = acc[t];
    }
}

}  // namespace strata::quant::sk
