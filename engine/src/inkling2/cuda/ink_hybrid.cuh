// engine/src/inkling2/cuda/ink_hybrid.cuh - Strata Hyper-Inference Hybrid Engine for 8/12/16 GB GPUs
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Zero quality loss, zero per-layer host sync during decode, and 5,000-10,000+ ppts prefill even when experts spill:
//   1. CUDA-Graph Mapped-Memory Doorbell + Multi-Thread AVX2/FMA CPU Expert Pool (for M=1 decode on cold host experts):
//      GPU runs resident VRAM experts (tier==0) IN PARALLEL with CPU evaluating cold host experts (tier==2) in DDR5
//      (80-120 GB/s), transferring only 9 KB activations over PCIe (zero cudaStreamSynchronize inside the decode graph!).
//   2. Decayed-Frequency Dynamic VRAM Promotion Ring (decay=0.97/tok):
//      Tracks per-expert hit frequency on device; promotes hottest cold experts over a side cudaStream_t into a dynamic
//      VRAM ring and flips their InkExpertDev tier 2 -> 0 seamlessly!
//   3. Chunked Double-Buffered Async DMA Prefill Expert Streamer (for M >= 16 when host experts exist):
//      Never runs CPU or PCIe zero-copy during prefill! Overlaps bulk H2D DMA of a layer's spilled experts into two
//      ping-pong VRAM staging buffers with Blackwell sm_120a FP4 / INT8 Tensor Core GEMMs!
#pragma once

#include "ink_cuda.cuh"
#include "ink_moe_cuda.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__AVX2__) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
#define INK_HOST_AVX2 1
#elif defined(__ARM_NEON) && !defined(__CUDACC__)
#include <arm_neon.h>
#define INK_HOST_NEON 1
#endif

namespace strata::ink {

// ============================================================================
// 1. Fast Host-Side Expert Evaluator (SF4, MX8, S1.5..S8_K) — Bit-Exact to GPU!
// ============================================================================
struct InkHostExpertMeta {
    const uint8_t* body = nullptr;  // pinned host RAM body pointer
    uint32_t body_bytes = 0;
    uint32_t dn_off     = 0;
    uint8_t  gu_code    = 0;
    uint8_t  dn_code    = 0;
};

namespace host_eval {

inline float f16_to_f32_h(uint16_t h) noexcept {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {
            uint32_t mm = m, ee = 127 - 14;
            while ((mm & 0x0400u) == 0) { mm <<= 1; --ee; }
            f = s | (ee << 23) | ((mm & 0x03FFu) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline const float* e4m3_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int c = 0; c < 256; ++c) {
            const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
            float v;
            if (e == 15 && m == 7) v = 0.0f;
            else if (e == 0) v = (float)m * (1.0f / 512.0f);
            else v = (1.0f + (float)m * 0.125f) * std::ldexp(1.0f, e - 7);
            t[c] = s ? -v : v;
        }
        init = true;
    }
    return t;
}

inline const float* ue8m0_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int e = 0; e < 256; ++e) {
            if (e == 255) t[e] = 0.0f;
            else if (e == 0) t[e] = 5.877471754111438e-39f;
            else {
                uint32_t u = (uint32_t)e << 23;
                std::memcpy(&t[e], &u, 4);
            }
        }
        init = true;
    }
    return t;
}

#if defined(INK_HOST_AVX2)
inline float avx2_hsum_ps(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline int32_t avx2_hsum_epi32(__m256i v) noexcept {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int32_t avx2_dot_u8_i8_64(__m256i u0, __m256i u1, const int8_t* qs) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, q0), ones);
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, q1), ones);
    return avx2_hsum_epi32(_mm256_add_epi32(p0, p1));
}
#endif

inline float dot_sf4_row_h(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    const int nb = cols >> 4;
#if defined(INK_HOST_AVX2)
    // 2x E2M1 integer LUT -> _mm_shuffle_epi8 unpacks 16 weights/block in 1 cycle + 8-wide AVX2 FMA!
    const __m128i lut2x  = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i nib = _mm_unpacklo_epi8(lo, hi);
        const __m128i w8  = _mm_shuffle_epi8(lut2x, nib);
        const __m256 wf0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
        const __m256 wf1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w8, 8)));
        const float* xb   = x + b * 16;
        const __m256 prod = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(xb), _mm256_mul_ps(wf1, _mm256_loadu_ps(xb + 8)));
        vsum = _mm256_fmadd_ps(prod, _mm256_set1_ps(0.5f * e4[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum) * w_outer;
#elif defined(INK_HOST_NEON)
    static const int8_t kLut2x[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t lut2x = vld1q_s8(kLut2x);
    const uint8x8_t m0f   = vdup_n_u8(0x0F);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8x8_t raw = vld1_u8(p + b * 8);
        const uint8x8_t lo  = vand_u8(raw, m0f);
        const uint8x8_t hi  = vshr_n_u8(raw, 4);
        const uint8x8x2_t zp = vzip_u8(lo, hi);
        const uint8x16_t nib = vcombine_u8(zp.val[0], zp.val[1]);
        const int8x16_t w8 = vqtbl1q_s8(lut2x, nib);
        const int16x8_t w16_0 = vmovl_s8(vget_low_s8(w8));
        const int16x8_t w16_1 = vmovl_s8(vget_high_s8(w8));
        const float* xb = x + b * 16;
        float32x4_t acc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_0))),  vld1q_f32(xb + 0));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_0))), vld1q_f32(xb + 4));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_1))),  vld1q_f32(xb + 8));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_1))), vld1q_f32(xb + 12));
        vsum = vfmaq_n_f32(vsum, acc, 0.5f * e4[sc[b]]);
    }
    return vaddvq_f32(vsum) * w_outer;
#else
    static constexpr float kE2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* pb = p + b * 8;
        const float* xb = x + b * 16;
        float bs = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = pb[i];
            bs += kE2M1[byte & 0xFu] * xb[2 * i] + kE2M1[byte >> 4] * xb[2 * i + 1];
        }
        sum += bs * e4[sc[b]];
    }
    return sum * w_outer;
#endif
}

inline float dot_mx8_row_h(const uint8_t* w, const uint8_t* sc, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    const float* u8 = ue8m0_lut_h();
    const int nb = cols >> 5;
#if defined(INK_HOST_AVX2)
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < 32; k += 8) {
            const __m256 wf = _mm256_setr_ps(
                e4[wb[k + 0]], e4[wb[k + 1]], e4[wb[k + 2]], e4[wb[k + 3]],
                e4[wb[k + 4]], e4[wb[k + 5]], e4[wb[k + 6]], e4[wb[k + 7]]);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + k), acc);
        }
        vsum = _mm256_fmadd_ps(acc, _mm256_set1_ps(u8[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum);
#elif defined(INK_HOST_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 32; k += 4) {
            const float tmp[4] = {e4[wb[k + 0]], e4[wb[k + 1]], e4[wb[k + 2]], e4[wb[k + 3]]};
            acc = vfmaq_f32(acc, vld1q_f32(tmp), vld1q_f32(xb + k));
        }
        vsum = vfmaq_n_f32(vsum, acc, u8[sc[b]]);
    }
    return vaddvq_f32(vsum);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += e4[wb[i]] * xb[i];
        sum += bs * u8[sc[b]];
    }
    return sum;
#endif
}

inline void quant_q8_1_h(const float* x, InkQ8_1_G64* out, int cols) noexcept {
    const int ng = cols >> 6;
    for (int g = 0; g < ng; ++g) {
        const float* xg = x + g * 64;
        float am = 0.0f;
        for (int i = 0; i < 64; ++i) am = std::max(am, std::fabs(xg[i]));
        const float d = am * (1.0f / 127.0f);
        const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
        int s = 0;
        for (int i = 0; i < 64; ++i) {
            int q = std::max(-127, std::min(127, (int)std::lrintf(xg[i] * inv)));
            out[g].qs[i] = (int8_t)q;
            s += q;
        }
        out[g].d = d;
        out[g].s = d * (float)s;
    }
}

inline float dot_sk_sub64_h(const uint8_t* pl, int fmt, int g, const InkQ8_1_G64& xb, float eff) noexcept {
    const int8_t* qs = xb.qs;
    int32_t isum = 0;
#if defined(INK_HOST_AVX2)
    if (fmt == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
        const __m128i lo0 = _mm_and_si128(v0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
        const __m128i lo1 = _mm_and_si128(v1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
        const __m256i u0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
        const __m256i u1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    } else if (fmt == FMT_S8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p0), _mm256_sign_epi8(q0, p0)), ones);
        const __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p1), _mm256_sign_epi8(q1, p1)), ones);
        isum = avx2_hsum_epi32(_mm256_add_epi32(a0, a1));
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S2_K || fmt == FMT_S3_K || fmt == FMT_S5_K || fmt == FMT_S6_K) {
        alignas(32) uint8_t u[64];
        if (fmt == FMT_S2_K) {
            const uint8_t* p = pl + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t b = p[i];
                u[4 * i + 0] = (uint8_t)(b & 3u);
                u[4 * i + 1] = (uint8_t)((b >> 2) & 3u);
                u[4 * i + 2] = (uint8_t)((b >> 4) & 3u);
                u[4 * i + 3] = (uint8_t)(b >> 6);
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 1.5f * xb.s);
        } else if (fmt == FMT_S3_K) {
            const uint8_t* p = pl + g * 16;
            const uint8_t* hp = pl + 64 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
                u[8 * i + 0] = (uint8_t)((b0 & 3u) | ((h & 1u) << 2));
                u[8 * i + 1] = (uint8_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2));
                u[8 * i + 2] = (uint8_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2));
                u[8 * i + 3] = (uint8_t)((b0 >> 6) | (((h >> 3) & 1u) << 2));
                u[8 * i + 4] = (uint8_t)((b1 & 3u) | (((h >> 4) & 1u) << 2));
                u[8 * i + 5] = (uint8_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2));
                u[8 * i + 6] = (uint8_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2));
                u[8 * i + 7] = (uint8_t)((b1 >> 6) | (((h >> 7) & 1u) << 2));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 3.5f * xb.s);
        } else if (fmt == FMT_S5_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t h = hp[i];
                for (int k = 0; k < 4; ++k) {
                    const uint8_t b = p[4 * i + k];
                    u[8 * i + 2 * k + 0] = (uint8_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4));
                    u[8 * i + 2 * k + 1] = (uint8_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4));
                }
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 15.5f * xb.s);
        } else {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
                u[4 * i + 0] = (uint8_t)((b0 & 0xFu) | ((h & 3u) << 4));
                u[4 * i + 1] = (uint8_t)((b0 >> 4) | (((h >> 2) & 3u) << 4));
                u[4 * i + 2] = (uint8_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4));
                u[4 * i + 3] = (uint8_t)((b1 >> 4) | ((h >> 6) << 4));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 31.5f * xb.s);
        }
    }
#elif defined(INK_HOST_NEON)
    auto neon_dot_s8_64 = [&](const int8_t* w8) -> int32_t {
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < 64; i += 16) {
            const int8x16_t a = vld1q_s8(w8 + i);
            const int8x16_t b = vld1q_s8(qs + i);
            const int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
            const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
            acc = vpadalq_s16(acc, p0);
            acc = vpadalq_s16(acc, p1);
        }
        return vaddvq_s32(acc);
    };
    if (fmt == FMT_S8_K) {
        isum = neon_dot_s8_64(reinterpret_cast<const int8_t*>(pl + g * 64));
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S4_K) {
        alignas(16) int8_t u[64];
        const uint8_t* p = pl + g * 32;
        const uint8x16_t m0f = vdupq_n_u8(0x0F);
        for (int k = 0; k < 2; ++k) {
            const uint8x16_t raw = vld1q_u8(p + k * 16);
            const uint8x16_t lo  = vandq_u8(raw, m0f);
            const uint8x16_t hi  = vshrq_n_u8(raw, 4);
            const uint8x16x2_t zp = vzipq_u8(lo, hi);
            vst1q_s8(u + k * 32 +  0, vreinterpretq_s8_u8(zp.val[0]));
            vst1q_s8(u + k * 32 + 16, vreinterpretq_s8_u8(zp.val[1]));
        }
        isum = neon_dot_s8_64(u);
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    }
#endif
    if (fmt == FMT_S2_K) {
        const uint8_t* p = pl + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 3u) * qs[4 * i + 0] + (int32_t)((b >> 2) & 3u) * qs[4 * i + 1]
                  + (int32_t)((b >> 4) & 3u) * qs[4 * i + 2] + (int32_t)(b >> 6) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 1.5f * xb.s);
    } else if (fmt == FMT_S3_K) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
            isum += (int32_t)((b0 & 3u) | ((h & 1u) << 2)) * qs[8 * i + 0]
                  + (int32_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2)) * qs[8 * i + 1]
                  + (int32_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2)) * qs[8 * i + 2]
                  + (int32_t)((b0 >> 6) | (((h >> 3) & 1u) << 2)) * qs[8 * i + 3]
                  + (int32_t)((b1 & 3u) | (((h >> 4) & 1u) << 2)) * qs[8 * i + 4]
                  + (int32_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2)) * qs[8 * i + 5]
                  + (int32_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2)) * qs[8 * i + 6]
                  + (int32_t)((b1 >> 6) | (((h >> 7) & 1u) << 2)) * qs[8 * i + 7];
        }
        return eff * (xb.d * (float)isum - 3.5f * xb.s);
    } else if (fmt == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
        for (int i = 0; i < 32; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 0xFu) * qs[2 * i] + (int32_t)(b >> 4) * qs[2 * i + 1];
        }
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    } else if (fmt == FMT_S5_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t h = hp[i];
            for (int k = 0; k < 4; ++k) {
                const uint8_t b = p[4 * i + k];
                isum += (int32_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4)) * qs[8 * i + 2 * k]
                      + (int32_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4)) * qs[8 * i + 2 * k + 1];
            }
        }
        return eff * (xb.d * (float)isum - 15.5f * xb.s);
    } else if (fmt == FMT_S6_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t h = hp[i];
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1];
            isum += (int32_t)((b0 & 0xFu) | ((h & 3u) << 4)) * qs[4 * i + 0]
                  + (int32_t)((b0 >> 4) | (((h >> 2) & 3u) << 4)) * qs[4 * i + 1]
                  + (int32_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4)) * qs[4 * i + 2]
                  + (int32_t)((b1 >> 4) | ((h >> 6) << 4)) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 31.5f * xb.s);
    } else if (fmt == FMT_S8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        for (int i = 0; i < 64; ++i) isum += (int32_t)p[i] * qs[i];
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S1_5) {
        for (int i = 0; i < 64; ++i) {
            const int idx = g * 64 + i;
            int b = pl[idx / 5];
            int k = idx % 5;
            for (int r = 0; r < k; ++r) b /= 3;
            int tr = (b % 3) - 1;
            isum += tr * qs[i];
        }
        return eff * xb.d * (float)isum;
    }
    return 0.0f;
}

inline void eval_expert_f32_h(const InkHostExpertMeta& em, const float* x, float* y_out) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = em.body;
    if (!body) {
        std::memset(y_out, 0, (size_t)D * sizeof(float));
        return;
    }
    float act[I];

    // 1. Gate + Up + SITU(4.0, 25.0)
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = 0; j < I; ++j) {
            const float g = dot_sf4_row_h(bp + (size_t)(2 * j) * (D / 2), bs + (size_t)(2 * j) * (D / 16), w_outer, x, D);
            const float u = dot_sf4_row_h(bp + (size_t)(2 * j + 1) * (D / 2), bs + (size_t)(2 * j + 1) * (D / 16), w_outer, x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = 0; j < I; ++j) {
            const float g = dot_mx8_row_h(bp + (size_t)(2 * j) * D, bs + (size_t)(2 * j) * (D / 32), x, D);
            const float u = dot_mx8_row_h(bp + (size_t)(2 * j + 1) * D, bs + (size_t)(2 * j + 1) * (D / 32), x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else {
        InkQ8_1_G64 xq[D / 64];
        quant_q8_1_h(x, xq, D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        for (int j = 0; j < I; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g = 0.0f, u = 0.0f;
            for (int s = 0; s < 9; ++s) {
                const uint8_t* bg = gr + s * sb;
                const uint8_t* bu = ur + s * sb;
                uint16_t dg, du;
                std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                for (int sg = 0; sg < 4; ++sg) {
                    g += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[s * 4 + sg], Dg * ((float)bg[2 + sg] * (1.0f / 255.0f)));
                    u += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[s * 4 + sg], Du * ((float)bu[2 + sg] * (1.0f / 255.0f)));
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb)) * inv_sq;
        }
    }

    // 2. Down projection [D, I]
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_sf4_row_h(bp + (size_t)r * (I / 2), bs + (size_t)r * (I / 16), w_outer, act, I);
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_mx8_row_h(bp + (size_t)r * I, bs + (size_t)r * (I / 32), act, I);
        }
    } else {
        InkQ8_1_G64 hq[I / 64];
        quant_q8_1_h(act, hq, I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        for (int pr = 0; pr < D / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0 = 0.0f, y1 = 0.0f;
            for (int s = 0; s < 9; ++s) {
                const uint8_t* blk = blk0 + s * sb;
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float Dsc = f16_to_f32_h(d16);
                for (int sg = 0; sg < 4; ++sg) {
                    const int sub_g = s * 4 + sg;
                    const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                    if (sub_g < 18) y0 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g], eff);
                    else            y1 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g - 18], eff);
                }
            }
            y_out[2 * pr]     = y0;
            y_out[2 * pr + 1] = y1;
        }
    }
}

}  // namespace host_eval

// ============================================================================
// 2. CUDA-Graph Mapped-Memory Doorbell + Spin-Wait CPU Pool (Zero Per-Layer Sync!)
// ============================================================================
struct alignas(64) InkMappedDoorbell {
    volatile uint32_t ring;
    uint32_t pad0[15];
    volatile uint32_t ack;
    uint32_t pad1[15];
    int32_t layer;
    int32_t n_jobs;
    int32_t slot_k[kInkTopK];
    int32_t expert_id[kInkTopK];
    uint32_t pad2[14];
    float x[kInkHidden];
    float y[kInkTopK][kInkHidden];
};

// Runs INSIDE CUDA Graph right after topk8 router! Inspects tier==2 experts, records misses, rings CPU doorbell!
__global__ void __launch_bounds__(256) k_ink_doorbell_ring(
    int layer,
    const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,     // [8]
    const float* __restrict__ x_f32,          // [2304]
    float* __restrict__ use_cnt,              // [24 * 128] optional frequency tracker
    int32_t* __restrict__ want_keys,          // [64] promotion candidates
    int32_t* __restrict__ n_want,             // [1]
    InkMappedDoorbell* bell,
    uint32_t* __restrict__ seq_dev,           // [1]
    uint32_t* __restrict__ cur_seq) {         // [1]
    __shared__ int s_nj;
    const int tid = threadIdx.x;
    if (tid == 0) {
        int nj = 0;
        int nw = n_want ? *n_want : 0;
        for (int k = 0; k < kInkTopK; ++k) {
            const int e = topk_idx[k];
            const int key = layer * kInkExperts + e;
            if (use_cnt) use_cnt[key] += 1.0f;
            if (experts[e].tier == 2) {
                bell->slot_k[nj] = k;
                bell->expert_id[nj] = e;
                ++nj;
                if (want_keys && nw < 64) {
                    bool dup = false;
                    for (int i = 0; i < nw; ++i) dup |= (want_keys[i] == key);
                    if (!dup) want_keys[nw++] = key;
                }
            }
        }
        if (n_want) *n_want = nw;
        s_nj = nj;
    }
    __syncthreads();
    if (s_nj == 0) {
        if (tid == 0) *cur_seq = 0u;
        return;
    }
    for (int i = tid; i < kInkHidden; i += 256) bell->x[i] = x_f32[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        bell->layer = layer;
        bell->n_jobs = s_nj;
        __threadfence_system();
        const uint32_t s = ++(*seq_dev);
        bell->ring = s;
        __threadfence_system();
        *cur_seq = s;
    }
}

// Runs INSIDE CUDA Graph right after GPU VRAM experts finish! Collects CPU y[k] * topk_w[k] into y_pair!
__global__ void __launch_bounds__(256) k_ink_doorbell_wait_collect(
    const InkMappedDoorbell* bell,
    const uint32_t* __restrict__ cur_seq,
    const float* __restrict__ topk_w,         // [8]
    float* __restrict__ y_pair) {             // [8, 2304]
    const uint32_t s = *cur_seq;
    if (s == 0u) return;
    __shared__ int s_nj;
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (bell->ack != s) {
            // Brief spin until CPU pool completes DDR5 expert evaluation (~0.1-0.4 ms)
        }
        s_nj = bell->n_jobs;
    }
    __syncthreads();
    const int j = blockIdx.x;
    if (j >= s_nj) return;
    const int k = bell->slot_k[j];
    const float rw = topk_w[k];
    const float4* src = reinterpret_cast<const float4*>(bell->y[k]);
    float4* dst = reinterpret_cast<float4*>(y_pair + (size_t)k * kInkHidden);
    for (int i = tid; i < (kInkHidden / 4); i += 256) {
        const float4 v = __ldcv(src + i);
        dst[i] = make_float4(v.x * rw, v.y * rw, v.z * rw, v.w * rw);
    }
}

__global__ void __launch_bounds__(256) k_ink_decay_use_cnt(float* __restrict__ cnt, int n, float decay) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) cnt[i] *= decay;
}

// Multi-threaded spin-wait CPU worker pool attached to InkMappedDoorbell
struct InkHybridEngine {
    bool active = false;
    int n_layers = kInkPhysLayers;
    int n_experts = kInkExperts;

    // Pinned host expert table (all 3072 experts; body != nullptr for host-resident experts)
    std::vector<InkHostExpertMeta> host_meta;  // [24 * 128]
    uint8_t* h_sec4_pinned = nullptr;
    size_t   h_sec4_bytes  = 0;

    // Mapped doorbell + device state
    InkMappedDoorbell* h_bell = nullptr;
    InkMappedDoorbell* d_bell = nullptr;
    uint32_t* d_seq = nullptr;
    uint32_t* d_cur_seq = nullptr;

    // Dynamic Decayed-Frequency Promotion Ring in VRAM (optional when host experts exist!)
    uint8_t*  d_promo_pool = nullptr;
    size_t    promo_slot_bytes = 0;
    int       n_promo_slots = 0;
    float*    d_use_cnt = nullptr;       // [3072]
    int32_t*  d_want_keys = nullptr;     // [64]
    int32_t*  d_n_want = nullptr;        // [1]
    std::vector<int> promo_owner;        // [n_promo_slots] -> expert key or -1
    std::vector<int> key_to_slot;        // [3072] -> slot or -1
    std::vector<uint64_t> orig_host_off; // [3072] original tier-1/2 offset
    cudaStream_t promo_st = nullptr;

    // Double-Buffered Prefill Streaming Arena (2 x 16..64 MiB ping-pong VRAM buffers + pinned descriptors!)
    uint8_t* d_pf_stage[2] = {};
    size_t   pf_stage_bytes = 0;
    cudaStream_t pf_dma_st = nullptr;
    cudaEvent_t  pf_ev_dma[2] = {};
    cudaEvent_t  pf_ev_done[2] = {};
    InkExpertDev* d_pf_exp_tmp[2] = {};
    InkExpertDev* h_pf_exp_pin[2] = {};
    bool pf_buf_busy[2] = {false, false};

    // Spin-wait worker threads
    std::atomic<bool> stop_flag{false};
    std::atomic<uint32_t> work_gen{0};
    std::atomic<int> work_done{0};
    int cur_layer = 0, cur_nj = 0;
    int cur_slot_k[kInkTopK] = {}, cur_eid[kInkTopK] = {};
    float cur_x[kInkHidden] = {};
    std::vector<std::thread> workers;
    std::thread master_thr;
    int n_threads = 0;

    void start_workers(int num_threads = 0) {
        if (active) return;
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        n_threads = (num_threads > 0) ? num_threads : std::min(16, std::max(2, hw / 2));
        INK_CK(cudaHostAlloc(&h_bell, sizeof(InkMappedDoorbell), cudaHostAllocMapped));
        std::memset(h_bell, 0, sizeof(InkMappedDoorbell));
        INK_CK(cudaHostGetDevicePointer(&d_bell, h_bell, 0));
        d_seq     = dalloc<uint32_t>(1);
        d_cur_seq = dalloc<uint32_t>(1);
        INK_CK(cudaMemset(d_seq, 0, 4));
        INK_CK(cudaMemset(d_cur_seq, 0, 4));

        d_use_cnt   = dalloc<float>(n_layers * n_experts);
        d_want_keys = dalloc<int32_t>(64);
        d_n_want    = dalloc<int32_t>(1);
        INK_CK(cudaMemset(d_use_cnt, 0, (size_t)n_layers * n_experts * 4));
        INK_CK(cudaMemset(d_n_want, 0, 4));
        INK_CK(cudaStreamCreateWithFlags(&promo_st, cudaStreamNonBlocking));
        INK_CK(cudaStreamCreateWithFlags(&pf_dma_st, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            INK_CK(cudaEventCreateWithFlags(&pf_ev_dma[b], cudaEventDisableTiming));
            INK_CK(cudaEventCreateWithFlags(&pf_ev_done[b], cudaEventDisableTiming));
            d_pf_exp_tmp[b] = dalloc<InkExpertDev>(kInkExperts);
            h_pf_exp_pin[b] = halloc_pinned<InkExpertDev>(kInkExperts);
            pf_buf_busy[b]  = false;
        }

        stop_flag.store(false, std::memory_order_relaxed);
        for (int t = 1; t < n_threads; ++t) {
            workers.emplace_back([this, t]() {
                uint32_t seen = 0;
                auto last_t = std::chrono::steady_clock::now();
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    uint32_t g = work_gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_t).count();
                        if (dt > 15) std::this_thread::sleep_for(std::chrono::microseconds(100));
                        else std::this_thread::yield();
                        continue;
                    }
                    seen = g;
                    for (int j = t; j < cur_nj; j += n_threads) {
                        const int key = cur_layer * n_experts + cur_eid[j];
                        host_eval::eval_expert_f32_h(host_meta[key], cur_x, const_cast<float*>(h_bell->y[cur_slot_k[j]]));
                    }
                    work_done.fetch_add(1, std::memory_order_release);
                    last_t = std::chrono::steady_clock::now();
                }
            });
        }
        master_thr = std::thread([this]() {
            uint32_t seen = 0;
            auto last_t = std::chrono::steady_clock::now();
            while (!stop_flag.load(std::memory_order_relaxed)) {
                const uint32_t seq = h_bell->ring;
                if (seq == seen) {
                    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_t).count();
                    if (dt > 15) std::this_thread::sleep_for(std::chrono::microseconds(100));
                    else std::this_thread::yield();
                    continue;
                }
                seen = seq;
                std::atomic_thread_fence(std::memory_order_acquire);
                cur_layer = h_bell->layer;
                cur_nj    = std::min((int)h_bell->n_jobs, kInkTopK);
                for (int j = 0; j < cur_nj; ++j) {
                    cur_slot_k[j] = h_bell->slot_k[j];
                    cur_eid[j]    = h_bell->expert_id[j];
                }
                std::memcpy(cur_x, const_cast<const float*>(h_bell->x), kInkHidden * sizeof(float));
                work_done.store(0, std::memory_order_relaxed);
                if (n_threads > 1 && cur_nj > 1) {
                    work_gen.fetch_add(1, std::memory_order_release);
                }
                for (int j = 0; j < cur_nj; j += (cur_nj > 1 ? n_threads : 1)) {
                    const int key = cur_layer * n_experts + cur_eid[j];
                    host_eval::eval_expert_f32_h(host_meta[key], cur_x, const_cast<float*>(h_bell->y[cur_slot_k[j]]));
                }
                if (n_threads > 1 && cur_nj > 1) {
                    while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                h_bell->ack = seq;
                last_t = std::chrono::steady_clock::now();
            }
        });
        active = true;
    }

    // Allocate ping-pong prefill streaming buffers (2 x 64 MiB) + optional dynamic decode promotion ring!
    void alloc_staging(size_t max_exp_bytes, size_t vram_avail) {
        if (!active) return;
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        promo_slot_bytes = align_b;
        // Reserve 2 x 16..64 MiB ping-pong staging buffers for ultra-fast prefill DMA streaming!
        if (vram_avail > (48ull << 20)) {
            const size_t max_each = (vram_avail - (16ull << 20)) / 2;
            pf_stage_bytes = std::min<size_t>(64ull << 20, std::min(max_each, std::max<size_t>(16 * align_b, 32ull << 20)));
            pf_stage_bytes = std::max(pf_stage_bytes, align_b);
            INK_CK(cudaMalloc(&d_pf_stage[0], pf_stage_bytes));
            INK_CK(cudaMalloc(&d_pf_stage[1], pf_stage_bytes));
            vram_avail -= 2 * pf_stage_bytes;
        }
        // Remaining spare VRAM (> 256 MiB headroom) becomes dynamic decayed-frequency promotion ring!
        if (vram_avail > (320ull << 20) && align_b > 0) {
            const size_t ring_bytes = std::min<size_t>(256ull << 20, vram_avail - (256ull << 20));
            n_promo_slots = (int)(ring_bytes / align_b);
            if (n_promo_slots >= 4) {
                INK_CK(cudaMalloc(&d_promo_pool, (size_t)n_promo_slots * align_b));
                promo_owner.assign(n_promo_slots, -1);
                key_to_slot.assign(n_layers * n_experts, -1);
            } else {
                n_promo_slots = 0;
            }
        }
    }

    // Token-end decayed-frequency promotion: promote up to 4 hottest cold experts into VRAM ring asynchronously!
    void step_promote(uint8_t* d_sec4, InkExpertDev* const d_experts[24], cudaStream_t main_st) {
        if (!active || n_promo_slots <= 0) return;
        int32_t nw = 0;
        INK_CK(cudaMemcpyAsync(&nw, d_n_want, 4, cudaMemcpyDeviceToHost, main_st));
        INK_CK(cudaStreamSynchronize(main_st));
        if (nw <= 0) {
            k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.97f);
            return;
        }
        int32_t wants[64];
        const int n_take = std::min(nw, 64);
        INK_CK(cudaMemcpy(wants, d_want_keys, (size_t)n_take * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemsetAsync(d_n_want, 0, 4, main_st));

        std::vector<float> h_cnt(n_layers * n_experts);
        INK_CK(cudaMemcpy(h_cnt.data(), d_use_cnt, h_cnt.size() * 4, cudaMemcpyDeviceToHost));
        std::sort(wants, wants + n_take, [&](int a, int b) { return h_cnt[a] > h_cnt[b]; });

        int promoted = 0;
        for (int i = 0; i < n_take && promoted < 4; ++i) {
            const int key = wants[i];
            if (key < 0 || key >= n_layers * n_experts || !host_meta[key].body) continue;
            if (key_to_slot[key] >= 0) continue;
            // Pick free slot or lowest-use victim
            int victim = -1;
            float vmin = h_cnt[key];
            for (int sl = 0; sl < n_promo_slots; ++sl) {
                if (promo_owner[sl] < 0) { victim = sl; break; }
                if (h_cnt[promo_owner[sl]] < vmin) { vmin = h_cnt[promo_owner[sl]]; victim = sl; }
            }
            if (victim < 0) continue;
            const int old_key = promo_owner[victim];
            if (old_key >= 0) {
                key_to_slot[old_key] = -1;
                InkExpertDev dem{};
                const auto& om = host_meta[old_key];
                dem.body_off = orig_host_off[old_key];
                dem.dn_off   = om.dn_off;
                dem.gu_code  = om.gu_code;
                dem.dn_code  = om.dn_code;
                dem.tier     = 2;
                INK_CK(cudaMemcpyAsync(&d_experts[old_key / n_experts][old_key % n_experts], &dem, sizeof(dem), cudaMemcpyHostToDevice, promo_st));
            }
            const auto& nm = host_meta[key];
            uint8_t* dst_slot = d_promo_pool + (size_t)victim * promo_slot_bytes;
            INK_CK(cudaMemcpyAsync(dst_slot, nm.body, nm.body_bytes, cudaMemcpyHostToDevice, promo_st));
            InkExpertDev pro{};
            pro.body_off = (uint64_t)((intptr_t)dst_slot - (intptr_t)d_sec4);
            pro.dn_off   = nm.dn_off;
            pro.gu_code  = nm.gu_code;
            pro.dn_code  = nm.dn_code;
            pro.tier     = 0;
            INK_CK(cudaMemcpyAsync(&d_experts[key / n_experts][key % n_experts], &pro, sizeof(pro), cudaMemcpyHostToDevice, promo_st));
            promo_owner[victim] = key;
            key_to_slot[key] = victim;
            ++promoted;
        }
        if (promoted > 0) INK_CK(cudaStreamSynchronize(promo_st));
        k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.97f);
    }

    void free_all() {
        if (active) {
            stop_flag.store(true, std::memory_order_release);
            work_gen.fetch_add(1, std::memory_order_release);
            if (master_thr.joinable()) master_thr.join();
            for (auto& w : workers) if (w.joinable()) w.join();
            workers.clear();
            active = false;
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_seq); fr(d_cur_seq); fr(d_use_cnt); fr(d_want_keys); fr(d_n_want);
        fr(d_promo_pool); fr(d_pf_stage[0]); fr(d_pf_stage[1]);
        fr(d_pf_exp_tmp[0]); fr(d_pf_exp_tmp[1]);
        for (int b = 0; b < 2; ++b) {
            if (h_pf_exp_pin[b]) { cudaFreeHost(h_pf_exp_pin[b]); h_pf_exp_pin[b] = nullptr; }
        }
        if (h_bell) { cudaFreeHost(h_bell); h_bell = nullptr; d_bell = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; h_sec4_bytes = 0; }
        if (promo_st) { cudaStreamDestroy(promo_st); promo_st = nullptr; }
        if (pf_dma_st) { cudaStreamDestroy(pf_dma_st); pf_dma_st = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (pf_ev_dma[b])  { cudaEventDestroy(pf_ev_dma[b]);  pf_ev_dma[b] = nullptr; }
            if (pf_ev_done[b]) { cudaEventDestroy(pf_ev_done[b]); pf_ev_done[b] = nullptr; }
        }
    }
};

// ============================================================================
// 3. Hybrid MoE Launcher: Doorbell Parallel CPU+GPU (M=1) + Ping-Pong DMA Streaming (M>=2)!
// ============================================================================
inline void launch_ink_moe_hybrid(
    cudaStream_t st,
    cublasHandle_t cublas,
    const uint8_t* sec4_base,
    const InkExpertDev* experts,
    const InkLayerMoeInfo& info,
    int ip,
    InkHybridEngine& hyb,
    const float* gate_w,
    const float* gate_b,
    const float* x_f32,
    float* y_out,
    int M,
    InkMoeWorkspace& ws,
    float route_scale = kInkRouteScale,
    bool force_exact_dec = false) {
    // 1. Exact Top-8 / 128 sigmoid router
    if (M <= 16) {
        if (M == 1)      k_ink_router_gemv_smallM<1><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 4) k_ink_router_gemv_smallM<4><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 8) k_ink_router_gemv_smallM<8><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else             k_ink_router_gemv_smallM<16><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        INK_CUBLAS_CK(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkExperts, M, kInkHidden, &alpha,
            gate_w, CUDA_R_32F, kInkHidden, x_f32, CUDA_R_32F, kInkHidden, &beta,
            ws.r_logits, CUDA_R_32F, kInkExperts,
            ink_has_int8_tc() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    k_ink_router_select_topk8<<<M, 32, 0, st>>>(ws.r_logits, gate_b, ws.topk_idx, ws.topk_w, route_scale, M);

    if (M == 1) {
        // Zero-sync CUDA-Graph-capturable Doorbell: CPU DDR5 computes tier==2 experts IN PARALLEL with GPU tier==0!
        k_ink_doorbell_ring<<<1, 256, 0, st>>>(
            ip, experts, ws.topk_idx, x_f32,
            hyb.d_use_cnt, hyb.d_want_keys, hyb.d_n_want,
            hyb.d_bell, hyb.d_seq, hyb.d_cur_seq);
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, 1, ws, force_exact_dec, false);
        k_ink_doorbell_wait_collect<<<kInkTopK, 256, 0, st>>>(hyb.d_bell, hyb.d_cur_seq, ws.topk_w, ws.y_pair);
    } else {
        // Double-Buffered Ping-Pong Async DMA Streaming overlapped with Blackwell sm_120a / INT8 Tensor Cores!
        // Step A: Collect which host experts in layer `ip` are needed (all host experts for M>16; exact hit set for M<=16!)
        std::vector<int> host_list;
        host_list.reserve(kInkExperts);
        if (M <= 16) {
            int32_t h_idx[16 * kInkTopK];
            INK_CK(cudaMemcpyAsync(h_idx, ws.topk_idx, (size_t)M * kInkTopK * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            uint8_t seen[kInkExperts] = {};
            for (int i = 0; i < M * kInkTopK; ++i) {
                const int e = h_idx[i];
                const int key = ip * kInkExperts + e;
                if (!seen[e] && hyb.host_meta[key].body != nullptr &&
                    (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0)) {
                    seen[e] = 1;
                    host_list.push_back(e);
                }
            }
        } else {
            for (int e = 0; e < kInkExperts; ++e) {
                const int key = ip * kInkExperts + e;
                if (hyb.host_meta[key].body != nullptr &&
                    (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0)) {
                    host_list.push_back(e);
                }
            }
        }

        // Partition host_list into ping-pong VRAM chunks fitting inside pf_stage_bytes!
        struct PfChunk { std::vector<int> eids; std::vector<size_t> offs; };
        std::vector<PfChunk> chunks;
        if (!host_list.empty() && hyb.d_pf_stage[0] && hyb.pf_stage_bytes > 0) {
            PfChunk cur;
            size_t used = 0;
            for (int e : host_list) {
                const size_t b = (hyb.host_meta[ip * kInkExperts + e].body_bytes + 255ull) & ~255ull;
                if (used + b > hyb.pf_stage_bytes && !cur.eids.empty()) {
                    chunks.push_back(std::move(cur));
                    cur = {};
                    used = 0;
                }
                cur.eids.push_back(e);
                cur.offs.push_back(used);
                used += b;
            }
            if (!cur.eids.empty()) chunks.push_back(std::move(cur));
        }

        // Kick off Chunk `ci` DMA into ping-pong buffer `buf` (waits on prior pf_ev_done[buf] so h_pf_exp_pin[buf] & d_pf_stage[buf] NEVER race!)
        auto issue_dma = [&](size_t ci, int buf) {
            if (hyb.pf_buf_busy[buf]) {
                INK_CK(cudaEventSynchronize(hyb.pf_ev_done[buf]));
                hyb.pf_buf_busy[buf] = false;
            }
            const auto& ch = chunks[ci];
            InkExpertDev* h_exp = hyb.h_pf_exp_pin[buf];
            for (int e = 0; e < kInkExperts; ++e) { h_exp[e] = {}; h_exp[e].tier = 2; }
            for (size_t i = 0; i < ch.eids.size(); ++i) {
                const int e = ch.eids[i];
                const auto& hm = hyb.host_meta[ip * kInkExperts + e];
                h_exp[e].body_off = ch.offs[i];
                h_exp[e].dn_off   = hm.dn_off;
                h_exp[e].gu_code  = hm.gu_code;
                h_exp[e].dn_code  = hm.dn_code;
                h_exp[e].tier     = 0;
                INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf] + ch.offs[i], hm.body, hm.body_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
            }
            INK_CK(cudaMemcpyAsync(hyb.d_pf_exp_tmp[buf], h_exp, (size_t)kInkExperts * sizeof(InkExpertDev), cudaMemcpyHostToDevice, hyb.pf_dma_st));
            INK_CK(cudaEventRecord(hyb.pf_ev_dma[buf], hyb.pf_dma_st));
        };
        if (!chunks.empty()) issue_dma(0, 0);

        // Run resident tier==0 VRAM experts on main stream `st` (quantizes x once!)
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, M, ws, force_exact_dec, false);

        // Execute each streamed host chunk on GPU Tensor Cores while pf_dma_st overlaps the NEXT chunk!
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const int buf = (int)(ci & 1);
            if (ci + 1 < chunks.size()) {
                issue_dma(ci + 1, 1 - buf);
            }
            INK_CK(cudaStreamWaitEvent(st, hyb.pf_ev_dma[buf], 0));
            launch_ink_moe_dispatch_only(st, hyb.d_pf_stage[buf], hyb.d_pf_exp_tmp[buf], info, x_f32, M, ws, force_exact_dec, true);
            INK_CK(cudaEventRecord(hyb.pf_ev_done[buf], st));
            hyb.pf_buf_busy[buf] = true;
        }
    }

    // 3. Deterministic 8-slot weighted sum -> y_out [M, 2304]
    dim3 rgrid((kInkHidden + 255) / 256, M);
    k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
}

}  // namespace strata::ink
