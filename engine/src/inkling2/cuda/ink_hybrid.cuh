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
#include "../ink_cold_store.hpp"
#include "../ink_host_avx2.h"
#include "strata/core/moe_wave_plan.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdlib>
#include <sys/mman.h>
#endif

#if defined(__ARM_NEON) && !defined(__CUDACC__)
#include <arm_neon.h>
#define INK_HOST_NEON 1
#endif

namespace strata::ink {

inline uint64_t ink_ram_available_bytes() noexcept {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullAvailPhys : 0ull;
#else
    uint64_t avail = 0, mfree = 0, mbuf = 0, mcache = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) avail = kb * 1024ull;
            else if (std::sscanf(line, "MemFree: %llu kB", &kb) == 1) mfree = kb * 1024ull;
            else if (std::sscanf(line, "Buffers: %llu kB", &kb) == 1) mbuf = kb * 1024ull;
            else if (std::sscanf(line, "Cached: %llu kB", &kb) == 1) mcache = kb * 1024ull;
        }
        std::fclose(f);
    }
    if (avail == 0) avail = mfree + mbuf + mcache;
    if (FILE* fm = std::fopen("/sys/fs/cgroup/memory.max", "r")) {
        char buf[64] = {};
        if (std::fgets(buf, sizeof(buf), fm) && buf[0] >= '0' && buf[0] <= '9') {
            const uint64_t lim = std::strtoull(buf, nullptr, 10);
            uint64_t cur = 0, inact = 0;
            if (FILE* fc = std::fopen("/sys/fs/cgroup/memory.current", "r")) {
                (void)std::fscanf(fc, "%llu", (unsigned long long*)&cur);
                std::fclose(fc);
            }
            if (FILE* fs = std::fopen("/sys/fs/cgroup/memory.stat", "r")) {
                char line[256];
                unsigned long long v = 0;
                while (std::fgets(line, sizeof(line), fs))
                    if (std::sscanf(line, "inactive_file %llu", &v) == 1) { inact = v; break; }
                std::fclose(fs);
            }
            const uint64_t used = cur > inact ? (cur - inact) : 0;
            const uint64_t cg_av = lim > used ? (lim - used) : 0;
            if (lim > 0 && (avail == 0 || cg_av < avail)) avail = cg_av;
        }
        std::fclose(fm);
    }
    return avail;
#endif
}

inline int ink_logical_cpus() noexcept {
#ifdef _WIN32
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) return (int)n;
#endif
    return std::max(1, (int)std::thread::hardware_concurrency());
}

// ============================================================================
// 1. Fast Host-Side Expert Evaluator (SF4, MX8, S1.5..S8_K) — Bit-Exact to GPU!
// ============================================================================
struct InkHostExpertMeta {
    const uint8_t* body = nullptr;  // pinned host RAM body pointer (nullptr when cold_ssd == true)
    uint32_t body_bytes = 0;
    uint32_t dn_off     = 0;
    uint8_t  gu_code    = 0;
    uint8_t  dn_code    = 0;
    bool     cold_ssd   = false;    // true in GPU+SSD Only mode: streamed via InkColdStore pinned DMA ring!
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

inline float dot_sf4_row_h(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    if (host_avx2::available()) {
        return host_avx2::dot_sf4_row(p, sc, w_outer, x, cols, e4);
    }
    const int nb = cols >> 4;
#if defined(INK_HOST_NEON)
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
    if (host_avx2::available()) {
        return host_avx2::dot_mx8_row(w, sc, x, cols, e4, u8);
    }
    const int nb = cols >> 5;
#if defined(INK_HOST_NEON)
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

inline const int8_t (*s15_trit_lut_h() noexcept)[5] {
    static int8_t lut[256][5];
    static bool init = false;
    if (!init) {
        for (int b = 0; b < 256; ++b) {
            int v = b;
            for (int k = 0; k < 5; ++k) {
                lut[b][k] = (int8_t)((v % 3) - 1);
                v /= 3;
            }
        }
        init = true;
    }
    return lut;
}

inline void s15_unpack64_h(const uint8_t* pl, int g, int8_t out[64]) noexcept {
    const auto* lut = s15_trit_lut_h();
    const int base = g * 64;
    int i = 0;
    int byte_idx = base / 5;
    int rem = base % 5;
    if (rem != 0) {
        const int8_t* t = lut[pl[byte_idx++]];
        for (int k = rem; k < 5 && i < 64; ++k) out[i++] = t[k];
    }
    while (i + 5 <= 64) {
        const int8_t* t = lut[pl[byte_idx++]];
        out[i + 0] = t[0]; out[i + 1] = t[1]; out[i + 2] = t[2]; out[i + 3] = t[3]; out[i + 4] = t[4];
        i += 5;
    }
    if (i < 64) {
        const int8_t* t = lut[pl[byte_idx]];
        for (int k = 0; i < 64; ++k) out[i++] = t[k];
    }
}

inline float dot_sk_sub64_h(const uint8_t* pl, int fmt, int g, const InkQ8_1_G64& xb, float eff) noexcept {
    if (host_avx2::available()) {
        return host_avx2::dot_sk_sub64(pl, fmt, g, xb.qs, xb.d, xb.s, eff, s15_trit_lut_h());
    }
    const int8_t* qs = xb.qs;
    int32_t isum = 0;
#if defined(INK_HOST_NEON)
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
        alignas(16) int8_t tr[64];
        s15_unpack64_h(pl, g, tr);
#if defined(INK_HOST_NEON)
        isum = neon_dot_s8_64(tr);
#else
        for (int i = 0; i < 64; ++i) isum += (int32_t)tr[i] * (int32_t)qs[i];
#endif
        return eff * xb.d * (float)isum;
    }
    return 0.0f;
}

// Lossless SIMD host evaluator for native FMT_GGUF_RAW experts (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS, IQ2_XXS, IQ1_M/S, etc.)
// Uses strata::quant runtime CPUID SIMD kernels (AVX-512 VNNI 4-row / AVX2 / AVX1 / NEON / Scalar) with zero SF4 quality loss!
inline void eval_gguf_raw_batch_h(
    const InkHostExpertMeta& em,
    const uint8_t* body,
    const float* const* xs,
    float* const* ys,
    int T,
    void (*wait_down)(void*), void* wait_ctx) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    constexpr int MAX_T = 16;
    InkGgufExpertHeader hdr{};
    std::memcpy(&hdr, body, sizeof(hdr));
    float act[MAX_T][I];

    auto dot_f32_h = [](const float* a, const float* b, int n) noexcept -> float {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += a[i + 0] * b[i + 0]; s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    };

    auto deq_gguf_row_h = [](uint32_t gtype, const uint8_t* src, float* dst, int K) noexcept {
        if (gtype == 0) {
            std::memcpy(dst, src, (size_t)K * 4u);
        } else if (gtype == 1) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int c = 0; c < K; ++c) dst[c] = f16_to_f32_h(p[c]);
        } else if (gtype == 30) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int c = 0; c < K; ++c) {
                uint32_t u = (uint32_t)p[c] << 16;
                std::memcpy(&dst[c], &u, 4);
            }
        } else {
            strata::quant::dequant_row(static_cast<strata::quant::Type>(gtype), src, dst, K);
        }
    };

    const auto arith = strata::quant::default_arith();

    // 1. Gate + Up + SITU(4.0, 25.0)
    {
        const auto qt = static_cast<strata::quant::Type>(hdr.gu_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.gu_type, D);
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && ti->act != qt) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes((int)ti->act, D) : 0;
        const uint8_t* w_base = body + sizeof(InkGgufExpertHeader);
        if (dk && ak && act_rb > 0 && rb > 0) {
            alignas(32) uint8_t qa[MAX_T][3072];  // max Q8_K(2304) = 9 * 292 = 2628 B < 3072 B
            for (int t = 0; t < T; ++t) ak->fn(xs[t], qa[t], D);
            int j = 0;
            if (dk4) {
                for (; j + 4 <= I; j += 4) {
                    const uint8_t* wg = w_base + (size_t)j * rb;
                    const uint8_t* wu = w_base + (size_t)(I + j) * rb;
                    for (int t = 0; t < T; ++t) {
                        float g4[4], u4[4];
                        dk4->fn(D, g4, wg, rb, qa[t]);
                        dk4->fn(D, u4, wu, rb, qa[t]);
                        for (int k = 0; k < 4; ++k) {
                            const float g = g4[k] * hdr.gate_scale;
                            const float u = u4[k] * hdr.up_scale;
                            const float sig = 1.0f / (1.0f + std::exp(-g));
                            act[t][j + k] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                            (kInkSituLb * std::tanh(u / kInkSituLb));
                        }
                    }
                }
            }
            for (; j < I; ++j) {
                const uint8_t* wg = w_base + (size_t)j * rb;
                const uint8_t* wu = w_base + (size_t)(I + j) * rb;
                for (int t = 0; t < T; ++t) {
                    float gs = 0.0f, us = 0.0f;
                    dk->fn(D, &gs, wg, qa[t]);
                    dk->fn(D, &us, wu, qa[t]);
                    const float g = gs * hdr.gate_scale;
                    const float u = us * hdr.up_scale;
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                (kInkSituLb * std::tanh(u / kInkSituLb));
                }
            }
        } else {
            float wg_f[D], wu_f[D];
            for (int j = 0; j < I; ++j) {
                deq_gguf_row_h(hdr.gu_type, w_base + (size_t)j * rb, wg_f, D);
                deq_gguf_row_h(hdr.gu_type, w_base + (size_t)(I + j) * rb, wu_f, D);
                for (int t = 0; t < T; ++t) {
                    const float g = dot_f32_h(wg_f, xs[t], D) * hdr.gate_scale;
                    const float u = dot_f32_h(wu_f, xs[t], D) * hdr.up_scale;
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                (kInkSituLb * std::tanh(u / kInkSituLb));
                }
            }
        }
    }

    // 2. Down projection [D, I]
    if (wait_down) wait_down(wait_ctx);
    {
        const auto qt = static_cast<strata::quant::Type>(hdr.dn_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.dn_type, I);
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && ti->act != qt) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes((int)ti->act, I) : 0;
        const uint8_t* w_base = body + em.dn_off;
        if (dk && ak && act_rb > 0 && rb > 0) {
            alignas(32) uint8_t qa[MAX_T][2048];  // max Q8_0(1152) = 36 * 34 = 1224 B < 2048 B
            for (int t = 0; t < T; ++t) ak->fn(act[t], qa[t], I);
            int r = 0;
            if (dk4) {
                for (; r + 4 <= D; r += 4) {
                    const uint8_t* wr = w_base + (size_t)r * rb;
                    for (int t = 0; t < T; ++t) {
                        float s4[4];
                        dk4->fn(I, s4, wr, rb, qa[t]);
                        ys[t][r + 0] = s4[0] * hdr.dn_scale;
                        ys[t][r + 1] = s4[1] * hdr.dn_scale;
                        ys[t][r + 2] = s4[2] * hdr.dn_scale;
                        ys[t][r + 3] = s4[3] * hdr.dn_scale;
                    }
                }
            }
            for (; r < D; ++r) {
                const uint8_t* wr = w_base + (size_t)r * rb;
                for (int t = 0; t < T; ++t) {
                    float s = 0.0f;
                    dk->fn(I, &s, wr, qa[t]);
                    ys[t][r] = s * hdr.dn_scale;
                }
            }
        } else {
            float wd_f[I];
            for (int r = 0; r < D; ++r) {
                deq_gguf_row_h(hdr.dn_type, w_base + (size_t)r * rb, wd_f, I);
                for (int t = 0; t < T; ++t) {
                    ys[t][r] = dot_f32_h(wd_f, act[t], I) * hdr.dn_scale;
                }
            }
        }
    }
}

inline void eval_expert_f32_h(const InkHostExpertMeta& em, const float* x, float* y_out,
                              const uint8_t* body_override = nullptr,
                              void (*wait_down)(void*) = nullptr, void* wait_ctx = nullptr) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = body_override ? body_override : em.body;
    if (!body) {
        std::memset(y_out, 0, (size_t)D * sizeof(float));
        return;
    }
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        const float* xs[1] = {x};
        float* ys[1] = {y_out};
        eval_gguf_raw_batch_h(em, body, xs, ys, 1, wait_down, wait_ctx);
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

    // 2. Down projection [D, I] (overlaps gate+up+SITU compute with down NVMe read when wait_down is set!)
    if (wait_down) wait_down(wait_ctx);
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

// Expert-grouped multi-token CPU evaluator (M=2..16 verify): reads each weight block ONCE across all `cnt` routed tokens!
inline void eval_expert_batch_f32_h(
    const InkHostExpertMeta& em,
    const float* const* xs,
    float* const* ys,
    int cnt,
    const uint8_t* body_override = nullptr,
    void (*wait_down)(void*) = nullptr, void* wait_ctx = nullptr) noexcept {
    if (cnt <= 0) return;
    if (cnt == 1) {
        eval_expert_f32_h(em, xs[0], ys[0], body_override, wait_down, wait_ctx);
        return;
    }
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = body_override ? body_override : em.body;
    if (!body) {
        for (int t = 0; t < cnt; ++t) std::memset(ys[t], 0, (size_t)D * sizeof(float));
        return;
    }
    constexpr int MAX_T = 16;
    const int T = std::min(cnt, MAX_T);
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        eval_gguf_raw_batch_h(em, body, xs, ys, T, wait_down, wait_ctx);
        return;
    }
    float act[MAX_T][I];

    // 1. Gate + Up + SITU(4.0, 25.0) — reuse each weight row across all T tokens in L1!
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = 0; j < I; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * (D / 2);
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 16);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * (D / 2);
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 16);
            for (int t = 0; t < T; ++t) {
                const float g = dot_sf4_row_h(pg, sg, w_outer, xs[t], D);
                const float u = dot_sf4_row_h(pu, su, w_outer, xs[t], D);
                const float sig = 1.0f / (1.0f + std::exp(-g));
                act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
            }
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = 0; j < I; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * D;
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 32);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * D;
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 32);
            for (int t = 0; t < T; ++t) {
                const float g = dot_mx8_row_h(pg, sg, xs[t], D);
                const float u = dot_mx8_row_h(pu, su, xs[t], D);
                const float sig = 1.0f / (1.0f + std::exp(-g));
                act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
            }
        }
    } else {
        InkQ8_1_G64 xq[MAX_T][D / 64];
        for (int t = 0; t < T; ++t) quant_q8_1_h(xs[t], xq[t], D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        for (int j = 0; j < I; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g[MAX_T] = {}, u[MAX_T] = {};
            for (int s = 0; s < 9; ++s) {
                const uint8_t* bg = gr + s * sb;
                const uint8_t* bu = ur + s * sb;
                uint16_t dg, du;
                std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                for (int sg = 0; sg < 4; ++sg) {
                    const float effg = Dg * ((float)bg[2 + sg] * (1.0f / 255.0f));
                    const float effu = Du * ((float)bu[2 + sg] * (1.0f / 255.0f));
                    const int idx = s * 4 + sg;
                    for (int t = 0; t < T; ++t) {
                        g[t] += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[t][idx], effg);
                        u[t] += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[t][idx], effu);
                    }
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            for (int t = 0; t < T; ++t) {
                const float sig = 1.0f / (1.0f + std::exp(-g[t]));
                act[t][j] = (kInkSituBeta * std::tanh(g[t] / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u[t] / kInkSituLb)) * inv_sq;
            }
        }
    }

    // 2. Down projection [D, I]
    if (wait_down) wait_down(wait_ctx);
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = 0; r < D; ++r) {
            const uint8_t* pr = bp + (size_t)r * (I / 2);
            const uint8_t* sr = bs + (size_t)r * (I / 16);
            for (int t = 0; t < T; ++t) ys[t][r] = dot_sf4_row_h(pr, sr, w_outer, act[t], I);
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = 0; r < D; ++r) {
            const uint8_t* pr = bp + (size_t)r * I;
            const uint8_t* sr = bs + (size_t)r * (I / 32);
            for (int t = 0; t < T; ++t) ys[t][r] = dot_mx8_row_h(pr, sr, act[t], I);
        }
    } else {
        InkQ8_1_G64 hq[MAX_T][I / 64];
        for (int t = 0; t < T; ++t) quant_q8_1_h(act[t], hq[t], I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        for (int pr = 0; pr < D / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0[MAX_T] = {}, y1[MAX_T] = {};
            for (int s = 0; s < 9; ++s) {
                const uint8_t* blk = blk0 + s * sb;
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float Dsc = f16_to_f32_h(d16);
                for (int sg = 0; sg < 4; ++sg) {
                    const int sub_g = s * 4 + sg;
                    const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                    if (sub_g < 18) {
                        for (int t = 0; t < T; ++t) y0[t] += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[t][sub_g], eff);
                    } else {
                        for (int t = 0; t < T; ++t) y1[t] += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[t][sub_g - 18], eff);
                    }
                }
            }
            for (int t = 0; t < T; ++t) {
                ys[t][2 * pr]     = y0[t];
                ys[t][2 * pr + 1] = y1[t];
            }
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

    // Pinned or pageable host expert table (all 3072 experts; body != nullptr for RAM-resident, cold_ssd == true for GPU+SSD Only)
    // On Windows WDDM / tight RAM hosts (#6520), falls back cleanly to pageable VirtualAlloc/mmap + pinned bounce ring!
    std::vector<InkHostExpertMeta> host_meta;  // [24 * 128]
    uint8_t* h_sec4_pinned   = nullptr;
    uint8_t* h_sec4_pageable = nullptr;
    size_t   h_sec4_bytes    = 0;
    bool     sec4_is_pinned  = false;
    uint8_t* h_pf_bounce[2]  = {};
    uint8_t* h_promo_stg     = nullptr;
    std::unique_ptr<ink2::InkColdStore> cold_store;
    bool ssd_only = false;
    std::atomic<uint64_t> step_gen{1};
    uint8_t* h_copyback_buf = nullptr;
    size_t   h_copyback_bytes = 0;

    // Mapped doorbell + device state
    InkMappedDoorbell* h_bell = nullptr;
    InkMappedDoorbell* d_bell = nullptr;
    uint32_t* d_seq = nullptr;
    uint32_t* d_cur_seq = nullptr;

    // Simultaneous PCIe + CPU miss-sharing buffers for M=2..16 DSpark Block-8 verify
    float* h_v_x    = nullptr;   // pinned [16 * kInkHidden]
    float* h_v_y    = nullptr;   // pinned [16 * kInkTopK * kInkHidden]
    int32_t* h_v_p  = nullptr;   // pinned [16 * kInkTopK] pair indices written by CPU
    int32_t* h_v_idx = nullptr;  // pinned [16 * kInkTopK] topk_idx D2H staging (<= 512 B)
    float* d_v_y    = nullptr;   // device mirror for fast scatter
    int32_t* d_v_p  = nullptr;   // device pair indices
    double dec_pcie_share = -1.0; // -1 = auto, 0 = 100% CPU, 1 = 100% PCIe DMA (STRATA_INK_DEC_PCIE_SHARE)
    uint32_t promo_step = 0;      // cadence counter so step_promote does NOT sync every single token!
    bool l3_pf_enabled = true;    // STRATA_L3_PF: idle-worker next-layer L3 prefetch
    std::atomic<uint32_t> pf_gen{0};
    int pf_layer = -1, pf_n = 0;
    int32_t pf_eids[32] = {};
    int32_t prev_cpu_eids[24][32] = {};
    int prev_cpu_n[24] = {};

    // Allocate host expert arena safely (pinned if small & within 45% MemAvailable, else pageable + pinned bounce ring!)
    uint8_t* alloc_host_arena(size_t bytes) {
        h_sec4_bytes = bytes;
        const uint64_t avail = ink_ram_available_bytes();
#ifdef _WIN32
        const bool try_pin = (bytes <= (8ull << 30)) && (avail == 0 || (double)bytes < 0.35 * (double)avail);
#else
        const bool try_pin = (avail == 0 || (double)bytes < 0.45 * (double)avail);
#endif
        if (try_pin) {
            if (cudaHostAlloc(&h_sec4_pinned, bytes, cudaHostAllocDefault) == cudaSuccess) {
                sec4_is_pinned = true;
                return h_sec4_pinned;
            }
            (void)cudaGetLastError();
            h_sec4_pinned = nullptr;
        }
        sec4_is_pinned = false;
#ifdef _WIN32
        h_sec4_pageable = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
            madvise(p, bytes, MADV_HUGEPAGE);
#endif
            h_sec4_pageable = (uint8_t*)p;
        }
#endif
        std::fprintf(stderr, "[InkHybrid] host expert arena %.2f GB allocated as pageable RAM + pinned bounce ring (avail=%.1f GB)\n",
                     (double)bytes / 1e9, (double)avail / 1e9);
        return h_sec4_pageable;
    }

    void kick_l3_prefetch(int layer, const int32_t* eids, int n) noexcept {
        if (!l3_pf_enabled || !active || !eids || n <= 0 || layer < 0 || layer >= n_layers) return;
        const int take = std::min(n, 32);
        pf_layer = layer;
        pf_n = take;
        for (int i = 0; i < take; ++i) pf_eids[i] = eids[i];
        pf_gen.fetch_add(1, std::memory_order_release);
    }

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

    // Spin-wait worker threads (shared by M=1 doorbell AND M=2..16 expert-grouped verify — ZERO std::thread spawns!)
    struct CpuVerifyGroup {
        int eid = 0;
        int count = 0;
        int pids[16] = {};
        int out_slots[16] = {};
    };
    std::atomic<bool> stop_flag{false};
    std::atomic<uint32_t> work_gen{0};
    std::atomic<int> work_done{0};
    std::atomic<int> verify_next{0};
    std::atomic<int> workers_sleeping{0};
    std::mutex wake_mu;
    std::condition_variable wake_cv;
    int work_mode = 0;  // 0 = M=1 doorbell jobs, 1 = M=2..16 expert-grouped verify jobs
    int cur_layer = 0, cur_nj = 0, cur_ng = 0;
    int cur_slot_k[kInkTopK] = {}, cur_eid[kInkTopK] = {};
    int cur_cold_slot[kInkTopK] = {-1,-1,-1,-1,-1,-1,-1,-1};
    const uint8_t* cur_cold_ptr[kInkTopK] = {};
    float cur_x[kInkHidden] = {};
    CpuVerifyGroup verify_groups[kInkExperts];
    std::vector<std::thread> workers;
    std::thread master_thr;
    int n_threads = 0;

    void eval_verify_group(int gi) noexcept {
        const auto& grp = verify_groups[gi];
        const int key = cur_layer * n_experts + grp.eid;
        const auto& hm = host_meta[key];
        const float* xs[16];
        float* ys[16];
        for (int k = 0; k < grp.count; ++k) {
            const int pid = grp.pids[k];
            const int m   = pid / kInkTopK;
            const int os  = grp.out_slots[k];
            h_v_p[os] = pid;
            xs[k] = h_v_x + (size_t)m * kInkHidden;
            ys[k] = h_v_y + (size_t)os * kInkHidden;
        }
        if (hm.body != nullptr) {
            host_eval::eval_expert_batch_f32_h(hm, xs, ys, grp.count);
        } else if (hm.cold_ssd && cold_store) {
            int cslot = -1;
            const uint64_t g_now = step_gen.load(std::memory_order_relaxed);
            const uint8_t* cptr = cold_store->demand_begin(key, cur_layer, g_now, cslot);
            if (cptr && cslot >= 0) {
                struct DwCtx { ink2::InkColdStore* cs; int slot; };
                DwCtx ctx{cold_store.get(), cslot};
                auto dw_cb = +[](void* p) {
                    auto* c = static_cast<DwCtx*>(p);
                    c->cs->wait_down(c->slot);
                };
                if (cold_store->wait_gate_up(cslot)) {
                    host_eval::eval_expert_batch_f32_h(hm, xs, ys, grp.count, cptr, dw_cb, &ctx);
                } else {
                    for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
                }
                cold_store->unpin_slot(cslot);
            } else {
                for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
            }
        } else {
            for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
        }
    }

    void run_verify_groups(int layer, int ng) noexcept {
        if (ng <= 0) return;
        cur_layer = layer;
        cur_ng = ng;
        work_mode = 1;
        verify_next.store(0, std::memory_order_relaxed);
        work_done.store(0, std::memory_order_relaxed);
        if (n_threads > 1 && ng > 1) {
            work_gen.fetch_add(1, std::memory_order_release);
            if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                std::lock_guard<std::mutex> lk(wake_mu);
                wake_cv.notify_all();
            }
        }
        while (true) {
            const int gi = verify_next.fetch_add(1, std::memory_order_relaxed);
            if (gi >= ng) break;
            eval_verify_group(gi);
        }
        if (n_threads > 1 && ng > 1) {
            while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
        }
        work_mode = 0;
    }

    void eval_doorbell_job(int j) noexcept {
        const int key = cur_layer * n_experts + cur_eid[j];
        const auto& hm = host_meta[key];
        float* ydst = const_cast<float*>(h_bell->y[cur_slot_k[j]]);
        if (hm.body != nullptr) {
            host_eval::eval_expert_f32_h(hm, cur_x, ydst);
        } else if (hm.cold_ssd && cold_store && cur_cold_slot[j] >= 0 && cur_cold_ptr[j] != nullptr) {
            struct DwCtx { ink2::InkColdStore* cs; int slot; };
            DwCtx ctx{cold_store.get(), cur_cold_slot[j]};
            auto dw_cb = +[](void* p) {
                auto* c = static_cast<DwCtx*>(p);
                c->cs->wait_down(c->slot);
            };
            if (cold_store->wait_gate_up(cur_cold_slot[j])) {
                host_eval::eval_expert_f32_h(hm, cur_x, ydst, cur_cold_ptr[j], dw_cb, &ctx);
            } else {
                std::memset(ydst, 0, (size_t)kInkHidden * sizeof(float));
            }
        } else {
            std::memset(ydst, 0, (size_t)kInkHidden * sizeof(float));
        }
    }

    void start_workers(int num_threads = 0) {
        if (active) return;
        const int hw = ink_logical_cpus();
        n_threads = (num_threads > 0) ? num_threads : std::min(16, std::max(2, hw / 2));
        if (const char* e = std::getenv("STRATA_L3_PF")) l3_pf_enabled = (e[0] != '0');
        if (const char* e = std::getenv("STRATA_INK_DEC_PCIE_SHARE")) dec_pcie_share = std::max(0.0, std::min(1.0, std::atof(e)));
        INK_CK(cudaHostAlloc(&h_bell, sizeof(InkMappedDoorbell), cudaHostAllocMapped));
        std::memset(h_bell, 0, sizeof(InkMappedDoorbell));
        INK_CK(cudaHostGetDevicePointer(&d_bell, h_bell, 0));
        d_seq     = dalloc<uint32_t>(1);
        d_cur_seq = dalloc<uint32_t>(1);
        INK_CK(cudaMemset(d_seq, 0, 4));
        INK_CK(cudaMemset(d_cur_seq, 0, 4));
        h_v_x   = halloc_pinned<float>(16 * kInkHidden);
        h_v_y   = halloc_pinned<float>(16 * kInkTopK * kInkHidden);
        h_v_p   = halloc_pinned<int32_t>(16 * kInkTopK);
        h_v_idx = halloc_pinned<int32_t>(16 * kInkTopK);
        d_v_y   = dalloc<float>(16 * kInkTopK * kInkHidden);
        d_v_p   = dalloc<int32_t>(16 * kInkTopK);

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
                uint32_t seen = 0, pf_seen = 0;
                auto last_t = std::chrono::steady_clock::now();
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    uint32_t g = work_gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        if (l3_pf_enabled) {
                            const uint32_t pg = pf_gen.load(std::memory_order_acquire);
                            if (pg != pf_seen) {
                                pf_seen = pg;
                                const int pl = pf_layer, pn = pf_n;
                                for (int i = t - 1; i < pn; i += std::max(1, n_threads - 1)) {
                                    if (work_gen.load(std::memory_order_acquire) != seen || stop_flag.load(std::memory_order_relaxed)) break;
                                    const int key = pl * n_experts + pf_eids[i];
                                    if (key >= 0 && key < (int)host_meta.size() && host_meta[key].body) {
                                        const char* ptr = reinterpret_cast<const char*>(host_meta[key].body);
                                        const size_t len = host_meta[key].body_bytes;
                                        for (size_t o = 0; o < len; o += 64) {
                                            if ((o & 16383u) == 0 && work_gen.load(std::memory_order_acquire) != seen) break;
#if defined(__GNUC__) || defined(__clang__)
                                            __builtin_prefetch(ptr + o, 0, 2);
#endif
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_t).count();
                        if (dt > 80) {
                            std::unique_lock<std::mutex> lk(wake_mu);
                            workers_sleeping.fetch_add(1, std::memory_order_release);
                            wake_cv.wait(lk, [&]() {
                                return stop_flag.load(std::memory_order_acquire) ||
                                       work_gen.load(std::memory_order_acquire) != seen;
                            });
                            workers_sleeping.fetch_sub(1, std::memory_order_release);
                            last_t = std::chrono::steady_clock::now();
                        } else {
                            std::this_thread::yield();
                        }
                        continue;
                    }
                    seen = g;
                    if (work_mode == 1) {
                        while (true) {
                            const int gi = verify_next.fetch_add(1, std::memory_order_relaxed);
                            if (gi >= cur_ng) break;
                            eval_verify_group(gi);
                        }
                    } else {
                        for (int j = t; j < cur_nj; j += n_threads) {
                            eval_doorbell_job(j);
                        }
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
                const uint64_t g_now = step_gen.load(std::memory_order_relaxed);
                for (int j = 0; j < cur_nj; ++j) {
                    cur_slot_k[j]    = h_bell->slot_k[j];
                    cur_eid[j]       = h_bell->expert_id[j];
                    cur_cold_slot[j] = -1;
                    cur_cold_ptr[j]  = nullptr;
                    const int key = cur_layer * n_experts + cur_eid[j];
                    if (cold_store && host_meta[key].cold_ssd && !host_meta[key].body) {
                        cur_cold_ptr[j] = cold_store->demand_begin(key, cur_layer, g_now, cur_cold_slot[j]);
                    }
                }
                std::memcpy(cur_x, const_cast<const float*>(h_bell->x), kInkHidden * sizeof(float));
                work_done.store(0, std::memory_order_relaxed);
                if (n_threads > 1 && cur_nj > 1) {
                    work_gen.fetch_add(1, std::memory_order_release);
                    if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lk(wake_mu);
                        wake_cv.notify_all();
                    }
                }
                for (int j = 0; j < cur_nj; j += (cur_nj > 1 ? n_threads : 1)) {
                    eval_doorbell_job(j);
                }
                if (n_threads > 1 && cur_nj > 1) {
                    while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
                }
                if (cold_store) {
                    for (int j = 0; j < cur_nj; ++j) {
                        if (cur_cold_slot[j] >= 0) {
                            cold_store->unpin_slot(cur_cold_slot[j]);
                            cur_cold_slot[j] = -1;
                        }
                    }
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                h_bell->ack = seq;
                last_t = std::chrono::steady_clock::now();
            }
        });
        active = true;
    }

    // Size of each ping-pong prefill staging buffer for experts up to max_exp_bytes (32..64 MiB, at least 1 expert).
    static size_t pf_stage_size(size_t max_exp_bytes) noexcept {
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        return std::max(align_b, std::min<size_t>(64ull << 20, std::max<size_t>(16 * align_b, 32ull << 20)));
    }

    // #6617: Allocate the 2 ping-pong prefill staging buffers BEFORE d_sec4 and the host pinned arena!
    // Without them, multi-token MoE cannot stream host experts over PCIe; try full size then 16 MiB minimum.
    bool alloc_pf_stage(size_t max_exp_bytes) {
        if (d_pf_stage[0]) return true;
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        size_t tries[2] = {pf_stage_size(max_exp_bytes), std::max(align_b, (size_t)(16ull << 20))};
        for (size_t b : tries) {
            if (b == 0) continue;
            if (cudaMalloc(&d_pf_stage[0], b) == cudaSuccess && cudaMalloc(&d_pf_stage[1], b) == cudaSuccess) {
                pf_stage_bytes = b;
                return true;
            }
            (void)cudaGetLastError();
            if (d_pf_stage[0]) { cudaFree(d_pf_stage[0]); d_pf_stage[0] = nullptr; }
            if (d_pf_stage[1]) { cudaFree(d_pf_stage[1]); d_pf_stage[1] = nullptr; }
        }
        pf_stage_bytes = 0;
        return false;
    }

    // Allocate ping-pong prefill streaming buffers (unless alloc_pf_stage already did) + optional dynamic decode promotion ring!
    void alloc_staging(size_t max_exp_bytes, size_t vram_avail) {
        if (!active) return;
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        promo_slot_bytes = align_b;
        // Reserve 2 x 16..64 MiB ping-pong staging buffers for ultra-fast prefill DMA streaming!
        if (!d_pf_stage[0] && vram_avail > (48ull << 20)) {
            const size_t max_each = (vram_avail - (16ull << 20)) / 2;
            pf_stage_bytes = std::min<size_t>(64ull << 20, std::min(max_each, std::max<size_t>(16 * align_b, 32ull << 20)));
            pf_stage_bytes = std::max(pf_stage_bytes, align_b);
            INK_CK(cudaMalloc(&d_pf_stage[0], pf_stage_bytes));
            INK_CK(cudaMalloc(&d_pf_stage[1], pf_stage_bytes));
            vram_avail -= 2 * pf_stage_bytes;
        }
        if (d_pf_stage[0] && !sec4_is_pinned && !h_pf_bounce[0]) {
            for (int b = 0; b < 2; ++b) {
                if (cudaHostAlloc(&h_pf_bounce[b], pf_stage_bytes, cudaHostAllocDefault) != cudaSuccess) {
                    (void)cudaGetLastError();
                    h_pf_bounce[b] = nullptr;
                }
            }
        }
        // Remaining spare VRAM (> 256 MiB headroom) becomes dynamic decayed-frequency promotion ring!
        if (vram_avail > (320ull << 20) && align_b > 0) {
            const size_t ring_bytes = std::min<size_t>(256ull << 20, vram_avail - (256ull << 20));
            n_promo_slots = (int)(ring_bytes / align_b);
            if (n_promo_slots >= 4) {
                INK_CK(cudaMalloc(&d_promo_pool, (size_t)n_promo_slots * align_b));
                if (!sec4_is_pinned) {
                    if (cudaHostAlloc(&h_promo_stg, align_b, cudaHostAllocDefault) != cudaSuccess) {
                        (void)cudaGetLastError();
                        h_promo_stg = nullptr;
                    }
                }
                promo_owner.assign(n_promo_slots, -1);
                key_to_slot.assign(n_layers * n_experts, -1);
            } else {
                n_promo_slots = 0;
            }
        }
    }

    // Periodic decayed-frequency promotion: avoids per-token cudaStreamSynchronize stalls!
    // Fast warmup during first 32 steps (every 4 tokens, up to 16 swaps), then steady-state every 16 tokens (up to 4 swaps).
    void step_promote(uint8_t* d_sec4, InkExpertDev* const d_experts[24], cudaStream_t main_st) {
        if (!active || n_promo_slots <= 0) return;
        const uint32_t stp = ++promo_step;
        const bool in_warmup = (stp <= 32u);
        const uint32_t period = in_warmup ? 4u : 16u;
        if ((stp % period) != 0u) return;
        int32_t nw = 0;
        INK_CK(cudaMemcpyAsync(&nw, d_n_want, 4, cudaMemcpyDeviceToHost, main_st));
        INK_CK(cudaStreamSynchronize(main_st));
        if (nw <= 0) {
            k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.94f);
            return;
        }
        int32_t wants[64];
        const int n_take = std::min(nw, 64);
        INK_CK(cudaMemcpy(wants, d_want_keys, (size_t)n_take * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemsetAsync(d_n_want, 0, 4, main_st));

        std::vector<float> h_cnt(n_layers * n_experts);
        INK_CK(cudaMemcpy(h_cnt.data(), d_use_cnt, h_cnt.size() * 4, cudaMemcpyDeviceToHost));
        std::sort(wants, wants + n_take, [&](int a, int b) { return h_cnt[a] > h_cnt[b]; });

        const int max_swaps = in_warmup ? 16 : 4;
        int promoted = 0;
        std::vector<int> pinned_cold_slots;
        const uint64_t g_now = step_gen.fetch_add(1, std::memory_order_relaxed);
        for (int i = 0; i < n_take && promoted < max_swaps; ++i) {
            const int key = wants[i];
            if (key < 0 || key >= n_layers * n_experts) continue;
            const auto& nm = host_meta[key];
            if (!nm.body && !(nm.cold_ssd && cold_store)) continue;
            if (key_to_slot[key] >= 0) continue;
            // Pick free slot or lowest-use victim
            int victim = -1;
            float vmin = h_cnt[key];
            for (int sl = 0; sl < n_promo_slots; ++sl) {
                if (promo_owner[sl] < 0) { victim = sl; break; }
                if (h_cnt[promo_owner[sl]] < vmin) { vmin = h_cnt[promo_owner[sl]]; victim = sl; }
            }
            if (victim < 0) continue;
            const uint8_t* src_body = nm.body;
            int cold_slot = -1;
            if (!src_body && nm.cold_ssd && cold_store) {
                src_body = cold_store->demand_wait(key, key / n_experts, g_now, cold_slot);
                if (!src_body) continue;
                pinned_cold_slots.push_back(cold_slot);
            }
            const int old_key = promo_owner[victim];
            uint8_t* dst_slot = d_promo_pool + (size_t)victim * promo_slot_bytes;
            if (old_key >= 0) {
                // Full-duplex VRAM -> RAM copyback for SSD-backed evicted experts so future misses hit RAM ring!
                const auto& om = host_meta[old_key];
                if (om.cold_ssd && cold_store && h_copyback_buf && om.body_bytes <= h_copyback_bytes) {
                    INK_CK(cudaMemcpyAsync(h_copyback_buf, dst_slot, om.body_bytes, cudaMemcpyDeviceToHost, promo_st));
                    INK_CK(cudaStreamSynchronize(promo_st));
                    cold_store->insert_from_copyback(old_key, h_copyback_buf, om.body_bytes);
                }
                key_to_slot[old_key] = -1;
                InkExpertDev dem{};
                dem.body_off = orig_host_off[old_key];
                dem.dn_off   = om.dn_off;
                dem.gu_code  = om.gu_code;
                dem.dn_code  = om.dn_code;
                dem.tier     = 2;
                INK_CK(cudaMemcpyAsync(&d_experts[old_key / n_experts][old_key % n_experts], &dem, sizeof(dem), cudaMemcpyHostToDevice, promo_st));
            }
            if (cold_slot < 0 && !sec4_is_pinned && h_promo_stg) {
                INK_CK(cudaStreamSynchronize(promo_st));
                std::memcpy(h_promo_stg, src_body, nm.body_bytes);
                INK_CK(cudaMemcpyAsync(dst_slot, h_promo_stg, nm.body_bytes, cudaMemcpyHostToDevice, promo_st));
            } else {
                INK_CK(cudaMemcpyAsync(dst_slot, src_body, nm.body_bytes, cudaMemcpyHostToDevice, promo_st));
            }
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
        if (cold_store) {
            for (int cs : pinned_cold_slots) cold_store->unpin_slot(cs);
        }
        k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.97f);
    }

    void free_all() {
        if (active) {
            stop_flag.store(true, std::memory_order_release);
            work_gen.fetch_add(1, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(wake_mu);
            }
            wake_cv.notify_all();
            if (master_thr.joinable()) master_thr.join();
            for (auto& w : workers) if (w.joinable()) w.join();
            workers.clear();
            active = false;
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_seq); fr(d_cur_seq); fr(d_use_cnt); fr(d_want_keys); fr(d_n_want);
        fr(d_promo_pool); fr(d_pf_stage[0]); fr(d_pf_stage[1]);
        fr(d_pf_exp_tmp[0]); fr(d_pf_exp_tmp[1]);
        fr(d_v_y); fr(d_v_p);
        for (int b = 0; b < 2; ++b) {
            if (h_pf_exp_pin[b]) { cudaFreeHost(h_pf_exp_pin[b]); h_pf_exp_pin[b] = nullptr; }
            if (h_pf_bounce[b])  { cudaFreeHost(h_pf_bounce[b]);  h_pf_bounce[b]  = nullptr; }
        }
        if (h_promo_stg) { cudaFreeHost(h_promo_stg); h_promo_stg = nullptr; }
        if (h_v_x) { cudaFreeHost(h_v_x); h_v_x = nullptr; }
        if (h_v_y) { cudaFreeHost(h_v_y); h_v_y = nullptr; }
        if (h_v_p) { cudaFreeHost(h_v_p); h_v_p = nullptr; }
        if (h_v_idx) { cudaFreeHost(h_v_idx); h_v_idx = nullptr; }
        if (h_copyback_buf) { cudaFreeHost(h_copyback_buf); h_copyback_buf = nullptr; h_copyback_bytes = 0; }
        cold_store.reset();
        if (h_bell) { cudaFreeHost(h_bell); h_bell = nullptr; d_bell = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; h_sec4_bytes = 0; }
        if (h_sec4_pageable) {
#ifdef _WIN32
            VirtualFree(h_sec4_pageable, 0, MEM_RELEASE);
#else
            munmap(h_sec4_pageable, h_sec4_bytes);
#endif
            h_sec4_pageable = nullptr;
            h_sec4_bytes = 0;
        }
        sec4_is_pinned = false;
        if (promo_st) { cudaStreamDestroy(promo_st); promo_st = nullptr; }
        if (pf_dma_st) { cudaStreamDestroy(pf_dma_st); pf_dma_st = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (pf_ev_dma[b])  { cudaEventDestroy(pf_ev_dma[b]);  pf_ev_dma[b] = nullptr; }
            if (pf_ev_done[b]) { cudaEventDestroy(pf_ev_done[b]); pf_ev_done[b] = nullptr; }
        }
    }
};

__global__ void __launch_bounds__(256) k_ink_scatter_cpu_pairs(
    const float* __restrict__ src_y,       // [n_pairs, kInkHidden]
    const int32_t* __restrict__ pair_idx,  // [n_pairs]
    const float* __restrict__ topk_w,      // [M * 8]
    float* __restrict__ y_pair,            // [M * 8, kInkHidden]
    int n_pairs) {
    const int j = blockIdx.x;
    if (j >= n_pairs) return;
    const int pid = pair_idx[j];
    const float rw = topk_w[pid];
    const float4* s4 = reinterpret_cast<const float4*>(src_y + (size_t)j * kInkHidden);
    float4* d4 = reinterpret_cast<float4*>(y_pair + (size_t)pid * kInkHidden);
    for (int i = threadIdx.x; i < (kInkHidden / 4); i += 256) {
        const float4 v = s4[i];
        d4[i] = make_float4(v.x * rw, v.y * rw, v.z * rw, v.w * rw);
    }
}

// ============================================================================
// 3. Hybrid MoE Launcher: Doorbell Parallel CPU+GPU (M=1) + Simultaneous PCIe+CPU Miss-Sharing (M=2..16) + Ping-Pong DMA Streaming (M>16)!
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
        // #6617: Clear y_pair so no unwritten slot can ever carry an earlier layer's stale output!
        INK_CK(cudaMemsetAsync(ws.y_pair, 0, (size_t)M * kInkTopK * kInkHidden * sizeof(float), st));
        // Double-Buffered Ping-Pong Async DMA Streaming + Simultaneous PCIe/CPU Miss-Sharing (M=2..16) + GPU+SSD Wave Planning!
        std::vector<int> host_list;
        host_list.reserve(kInkExperts);
        std::vector<int> cpu_eids;
        std::vector<std::pair<int, int>> cpu_pairs;  // (pair_id, expert_id)
        auto is_off_gpu = [&](int key) -> bool {
            return (hyb.host_meta[key].body != nullptr || hyb.host_meta[key].cold_ssd) &&
                   (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0);
        };
        if (M <= 16 && !hyb.ssd_only) {
            int32_t* h_idx = hyb.h_v_idx;
            int32_t h_idx_fallback[16 * kInkTopK] = {};
            if (!h_idx) h_idx = h_idx_fallback;
            int ecnt[kInkExperts] = {};
            // Copy ONLY tiny topk_idx (<= 512 B) first so 0-miss visits and tiny-spill PCIe-only visits NEVER stall on x_f32 D2H!
            INK_CK(cudaMemcpyAsync(h_idx, ws.topk_idx, (size_t)M * kInkTopK * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            for (int i = 0; i < M * kInkTopK; ++i) {
                const int e = h_idx[i];
                const int key = ip * kInkExperts + e;
                if (e >= 0 && e < kInkExperts && hyb.host_meta[key].body != nullptr &&
                    (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0)) {
                    if (ecnt[e] == 0) host_list.push_back(e);
                    ++ecnt[e];
                }
            }
            // Zero-cliff small-spill fast path: when <= 6 cold experts miss (~2-6 MB) and PCIe staging is ready,
            // stream ALL of them asynchronously over PCIe H2D to GPU Tensor Cores (< 0.15 ms, zero x_f32 D2H, zero CPU wait!)
            const bool stream_all_pcie = (hyb.dec_pcie_share < 0.0) && (host_list.size() <= 6) &&
                                         (hyb.d_pf_stage[0] != nullptr) && (hyb.pf_stage_bytes > 0);
            const double share = (hyb.dec_pcie_share >= 0.0) ? hyb.dec_pcie_share
                               : (stream_all_pcie ? 1.0 : (hyb.sec4_is_pinned ? 0.65 : 0.55));
            if (hyb.active && hyb.h_v_x && hyb.h_v_y && host_list.size() >= 2 && share < 0.999) {
                std::sort(host_list.begin(), host_list.end(), [&](int a, int b) { return ecnt[a] > ecnt[b]; });
                int w_tot = 0;
                for (int e : host_list) w_tot += ecnt[e];
                const double w_goal = share * (double)w_tot;
                int w_acc = 0;
                size_t split = 0;
                while (split < host_list.size() && (double)w_acc < w_goal) {
                    w_acc += ecnt[host_list[split]];
                    ++split;
                }
                uint8_t is_cpu[kInkExperts] = {};
                for (size_t i = split; i < host_list.size(); ++i) {
                    is_cpu[host_list[i]] = 1;
                    cpu_eids.push_back(host_list[i]);
                }
                host_list.resize(split);
                for (int p = 0; p < M * kInkTopK; ++p) {
                    const int e = h_idx[p];
                    if (e >= 0 && e < kInkExperts && is_cpu[e]) cpu_pairs.push_back({p, e});
                }
                // Copy x_f32 D2H ONLY when CPU pool actually has cold misses to evaluate!
                if (!cpu_pairs.empty()) {
                    INK_CK(cudaMemcpyAsync(hyb.h_v_x, x_f32, (size_t)M * kInkHidden * 4, cudaMemcpyDeviceToHost, st));
                    INK_CK(cudaStreamSynchronize(st));
                }
            }
        } else if (M <= 16 || hyb.ssd_only) {
            std::vector<int32_t> h_idx((size_t)M * kInkTopK);
            INK_CK(cudaMemcpyAsync(h_idx.data(), ws.topk_idx, (size_t)M * kInkTopK * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            uint8_t seen[kInkExperts] = {};
            std::vector<int32_t> active_all;
            for (int i = 0; i < M * kInkTopK; ++i) {
                const int e = h_idx[i];
                if (e >= 0 && e < kInkExperts && !seen[e]) {
                    seen[e] = 1;
                    active_all.push_back(e);
                }
            }
            // Apply pe128 prompt-end recency wave ordering so hottest prompt-end experts stay resident in cold_store!
            const auto w_pe = strata::core::compute_pe128_scores(h_idx.data(), M, kInkTopK, kInkExperts);
            std::vector<uint8_t> is_res(kInkExperts, 0);
            for (int32_t e : active_all) {
                if (!is_off_gpu(ip * kInkExperts + e)) is_res[e] = 1;
            }
            const int cap_s = hyb.cold_store ? std::max(16, hyb.cold_store->num_slots()) : 64;
            const auto wplan = strata::core::plan_moe_waves(active_all, is_res, w_pe, cap_s);
            for (const auto& wv : wplan.cold_waves) {
                for (int32_t e : wv) host_list.push_back(e);
            }
        } else {
            for (int e = 0; e < kInkExperts; ++e) {
                if (is_off_gpu(ip * kInkExperts + e)) host_list.push_back(e);
            }
        }

        // Partition GPU-assigned host_list into ping-pong VRAM chunks fitting inside pf_stage_bytes (and cold_store slots/2 when SSD-only)!
        struct PfChunk { std::vector<int> eids; std::vector<size_t> offs; size_t total_bytes = 0; };
        std::vector<PfChunk> chunks;
        const size_t max_e_per_chunk = hyb.cold_store ? (size_t)std::max(1, hyb.cold_store->num_slots() / 2 - 2) : (size_t)kInkExperts;
        if (!host_list.empty() && hyb.d_pf_stage[0] && hyb.pf_stage_bytes > 0) {
            PfChunk cur;
            size_t used = 0;
            for (int e : host_list) {
                const size_t b = (hyb.host_meta[ip * kInkExperts + e].body_bytes + 255ull) & ~255ull;
                if ((used + b > hyb.pf_stage_bytes || cur.eids.size() >= max_e_per_chunk) && !cur.eids.empty()) {
                    cur.total_bytes = used;
                    chunks.push_back(std::move(cur));
                    cur = {};
                    used = 0;
                }
                cur.eids.push_back(e);
                cur.offs.push_back(used);
                used += b;
            }
            if (!cur.eids.empty()) {
                cur.total_bytes = used;
                chunks.push_back(std::move(cur));
            }
        }

        // Track cold_store pinned slots per ping-pong buffer so we unpin only after DMA finishes!
        if (hyb.cold_store) hyb.cold_store->notify_active();
        std::vector<int> buf_cold_slots[2];
        const uint64_t pf_gen = hyb.step_gen.fetch_add(1, std::memory_order_relaxed);
        auto unpin_buf_slots = [&](int buf) {
            if (hyb.cold_store) {
                for (int s : buf_cold_slots[buf]) hyb.cold_store->unpin_slot(s);
            }
            buf_cold_slots[buf].clear();
        };

        // Kick off Chunk `ci` DMA into ping-pong buffer `buf` (uses bounce ring cleanly when host arena is pageable!)
        auto issue_dma = [&](size_t ci, int buf) {
            if (hyb.pf_buf_busy[buf]) {
                INK_CK(cudaEventSynchronize(hyb.pf_ev_done[buf]));
                hyb.pf_buf_busy[buf] = false;
            }
            unpin_buf_slots(buf);
            const auto& ch = chunks[ci];
            InkExpertDev* h_exp = hyb.h_pf_exp_pin[buf];
            for (int e = 0; e < kInkExperts; ++e) { h_exp[e] = {}; h_exp[e].tier = 2; }
            const bool use_bounce = (!hyb.sec4_is_pinned && !hyb.ssd_only && hyb.h_pf_bounce[buf] != nullptr);
            // Pre-issue NVMe demand reads for all cold_ssd experts in this chunk in parallel!
            std::vector<const uint8_t*> ptrs(ch.eids.size(), nullptr);
            std::vector<int> cslots(ch.eids.size(), -1);
            for (size_t i = 0; i < ch.eids.size(); ++i) {
                const int key = ip * kInkExperts + ch.eids[i];
                const auto& hm = hyb.host_meta[key];
                if (hm.body) {
                    ptrs[i] = hm.body;
                } else if (hm.cold_ssd && hyb.cold_store) {
                    ptrs[i] = hyb.cold_store->demand_begin(key, ip, pf_gen, cslots[i]);
                    if (cslots[i] >= 0) buf_cold_slots[buf].push_back(cslots[i]);
                }
            }
            for (size_t i = 0; i < ch.eids.size(); ++i) {
                const int e = ch.eids[i];
                const auto& hm = hyb.host_meta[ip * kInkExperts + e];
                bool ok_io = true;
                if (cslots[i] >= 0 && hyb.cold_store) {
                    ok_io = hyb.cold_store->wait_gate_up(cslots[i]) && hyb.cold_store->wait_down(cslots[i]);
                }
                if (!ptrs[i] || !ok_io) {
                    std::fprintf(stderr, "[inkling-cuda] WARN cold_store expert key=%d ptr=%p ok_io=%d slot=%d\n",
                                 ip * kInkExperts + e, (const void*)ptrs[i], (int)ok_io, cslots[i]);
                    continue;
                }
                h_exp[e].body_off = ch.offs[i];
                h_exp[e].dn_off   = hm.dn_off;
                h_exp[e].gu_code  = hm.gu_code;
                h_exp[e].dn_code  = hm.dn_code;
                h_exp[e].tier     = 0;
                if (use_bounce) {
                    std::memcpy(hyb.h_pf_bounce[buf] + ch.offs[i], ptrs[i], hm.body_bytes);
                } else {
                    INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf] + ch.offs[i], ptrs[i], hm.body_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
                }
            }
            if (use_bounce && ch.total_bytes > 0) {
                INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf], hyb.h_pf_bounce[buf], ch.total_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
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

        // Concurrently evaluate CPU-assigned pairs for M=2..16 on the PERSISTENT worker pool (ZERO std::thread spawns!)
        // Deduplicates unique cold experts across M verify tokens -> ONE single weight read per expert!
        if (!cpu_pairs.empty()) {
            const int np = (int)cpu_pairs.size();
            int eid_to_grp[kInkExperts];
            std::memset(eid_to_grp, -1, sizeof(eid_to_grp));
            int ng = 0;
            for (int j = 0; j < np; ++j) {
                const int pid = cpu_pairs[j].first;
                const int e   = cpu_pairs[j].second;
                int gi = eid_to_grp[e];
                if (gi < 0) {
                    gi = ng++;
                    eid_to_grp[e] = gi;
                    hyb.verify_groups[gi].eid = e;
                    hyb.verify_groups[gi].count = 0;
                }
                auto& grp = hyb.verify_groups[gi];
                if (grp.count < 16) {
                    const int k = grp.count++;
                    grp.pids[k] = pid;
                    grp.out_slots[k] = j;
                }
            }
            hyb.run_verify_groups(ip, ng);
            INK_CK(cudaMemcpyAsync(hyb.d_v_y, hyb.h_v_y, (size_t)np * kInkHidden * 4, cudaMemcpyHostToDevice, st));
            INK_CK(cudaMemcpyAsync(hyb.d_v_p, hyb.h_v_p, (size_t)np * 4, cudaMemcpyHostToDevice, st));
            k_ink_scatter_cpu_pairs<<<np, 256, 0, st>>>(hyb.d_v_y, hyb.d_v_p, ws.topk_w, ws.y_pair, np);
            // Record CPU misses for layer ip and kick L3 prefetch for next layer (STRATA_L3_PF)
            const int cn = std::min((int)cpu_eids.size(), 32);
            hyb.prev_cpu_n[ip] = cn;
            for (int i = 0; i < cn; ++i) hyb.prev_cpu_eids[ip][i] = cpu_eids[i];
            if (ip + 1 < kInkPhysLayers && hyb.prev_cpu_n[ip + 1] > 0) {
                hyb.kick_l3_prefetch(ip + 1, hyb.prev_cpu_eids[ip + 1], hyb.prev_cpu_n[ip + 1]);
            }
        }
        // Unpin cold_store slots after the last chunk's DMA has completed (pf_ev_dma already waited by st).
        if (hyb.cold_store && (!buf_cold_slots[0].empty() || !buf_cold_slots[1].empty())) {
            INK_CK(cudaStreamSynchronize(hyb.pf_dma_st));
            unpin_buf_slots(0);
            unpin_buf_slots(1);
        }
    }

    // 3. Deterministic 8-slot weighted sum -> y_out [M, 2304]
    dim3 rgrid((kInkHidden + 255) / 256, M);
    k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
}

}  // namespace strata::ink
