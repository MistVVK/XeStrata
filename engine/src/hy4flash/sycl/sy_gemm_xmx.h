// sy_gemm_xmx.h - High-occupancy Intel Arc XMX prefill GEMM (FP16 multi-tile 8x8x16 + DPAS INT8 8x8x32 + Xe2 SG16 + SG32 SIMT fallback).
//   Y[m][n] = sum_k W[n][k] X[m][k];  W = S_K [N][K] (K % 256 == 0, N % 64 == 0) in raw wire or RP layout (sy_rp.h), X half [M][K], Y float [M][N].
// Optimizations vs original 10-11 TFLOP/s kernel:
//   1. BK = 64 (1 sub-group per SLM stage) -> 8 KiB SLM (4x smaller than 32 KiB) -> 4x higher Xe-core occupancy.
//   2. 128 threads/workgroup = 16 SG8 sub-groups (8 M-warps x 2 N-warps), BM=128 (or 64 for M<=64), BN=64 -> 2x token reuse, 2x compute/load ratio.
//   3. Cooperative vectorized dequant: 2 threads per weight row (32 weights/thread), coalesced vector loads, FMA (q*eff + bias), 32-bit half2 SLM stores.
//   4. Optional DPAS INT8 XMX (joint_matrix int8 x uint8/int8 -> int32, 8x8x32 = 2x XMX TOP/s, 4 KiB VNNI-4 SLM, integer-only unpack) via HY4S_XMX_I8=1|2.
//   5. Portable SG32 SIMT tiled GEMM fallback for non-XMX Intel iGPUs (Core Ultra / UHD).
#pragma once
#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifdef HY4S_USE_ONEMKL
#include <oneapi/mkl/blas.hpp>
#endif

#include "sy_common.h"
#include "sy_linear.h"
#include "sy_rp.h"

namespace hy4s {

namespace xm = sycl::ext::oneapi::experimental::matrix;

namespace xmx_detail {

inline void put_h2(half* p, float a, float b) {
    *reinterpret_cast<sycl::vec<half, 2>*>(p) = sycl::vec<half, 2>((half)a, (half)b);
}

inline uint32_t ld_u32_unaligned(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

constexpr int SB_FP16_LD = 136;  // 128 + 8 halves padding -> zero SLM bank conflicts!

// Cooperatively dequantize 32 weights (half_idx = 0 -> weights 0..31, half_idx = 1 -> weights 32..63) of 64-weight subgroup `g`
// of super-block `sbi` of row `row_local` (0..63) into VNNI-2 packed SLM `sb_ptr` [32][SB_FP16_LD] halves:
//   element (k in 0..63, n in 0..63) at sb_ptr[(k >> 1) * SB_FP16_LD + n * 2 + (k & 1)].
template <int FMT, bool RP>
inline void dq_half_group_fp16(const uint8_t* pl, float eff, int g, int half_idx, half* sb_ptr, int row_local) {
    const float bias = -sk_half<FMT>() * eff;
    const int k_base = half_idx * 32;  // 0 or 32 within the 64-weight subgroup
    half* dst = sb_ptr + (k_base >> 1) * SB_FP16_LD + row_local * 2;
    if constexpr (FMT == FMT_S4) {
        // 32 weights = 16 bytes
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        if constexpr (RP) {
            const sycl::uchar16 v = *reinterpret_cast<const sycl::uchar16*>(p);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t by = v[i];
                put_h2(dst + i * SB_FP16_LD, (float)(by & 15u) * eff + bias, (float)(by >> 4) * eff + bias);
            }
        } else {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t wd = ld_u32_unaligned(p + w * 4);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                    put_h2(dst + (w * 4 + j) * SB_FP16_LD, (float)(by & 15u) * eff + bias, (float)(by >> 4) * eff + bias);
                }
            }
        }
    } else if constexpr (FMT == FMT_S5) {
        // 32 low nibbles (16 B) + 32 high bits (4 B)
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint32_t hb = ld_u32_unaligned(pl + 128 + g * 8 + half_idx * 4);
        if constexpr (RP) {
            const sycl::uchar16 v = *reinterpret_cast<const sycl::uchar16*>(p);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t by = v[i];
                const uint32_t q0 = (by & 15u) | (((hb >> (2 * i)) & 1u) << 4);
                const uint32_t q1 = (by >> 4) | (((hb >> (2 * i + 1)) & 1u) << 4);
                put_h2(dst + i * SB_FP16_LD, (float)q0 * eff + bias, (float)q1 * eff + bias);
            }
        } else {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t wd = ld_u32_unaligned(p + w * 4);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int i = w * 4 + j;
                    const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                    const uint32_t q0 = (by & 15u) | (((hb >> (2 * i)) & 1u) << 4);
                    const uint32_t q1 = (by >> 4) | (((hb >> (2 * i + 1)) & 1u) << 4);
                    put_h2(dst + i * SB_FP16_LD, (float)q0 * eff + bias, (float)q1 * eff + bias);
                }
            }
        }
    } else if constexpr (FMT == FMT_S8) {
        // 32 int8 weights (32 B)
        const uint8_t* p = pl + g * 64 + half_idx * 32;
        if constexpr (RP) {
            const sycl::uchar16 v0 = *reinterpret_cast<const sycl::uchar16*>(p);
            const sycl::uchar16 v1 = *reinterpret_cast<const sycl::uchar16*>(p + 16);
#pragma unroll
            for (int i = 0; i < 8; ++i) put_h2(dst + i * SB_FP16_LD, (float)(int8_t)v0[2 * i] * eff, (float)(int8_t)v0[2 * i + 1] * eff);
#pragma unroll
            for (int i = 0; i < 8; ++i) put_h2(dst + (8 + i) * SB_FP16_LD, (float)(int8_t)v1[2 * i] * eff, (float)(int8_t)v1[2 * i + 1] * eff);
        } else {
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t wd = ld_u32_unaligned(p + w * 4);
                put_h2(dst + (2 * w) * SB_FP16_LD, (float)(int8_t)(wd & 0xFFu) * eff, (float)(int8_t)((wd >> 8) & 0xFFu) * eff);
                put_h2(dst + (2 * w + 1) * SB_FP16_LD, (float)(int8_t)((wd >> 16) & 0xFFu) * eff, (float)(int8_t)(wd >> 24) * eff);
            }
        }
    } else if constexpr (FMT == FMT_S2) {
        // 32 weights = 8 bytes (4 weights per byte)
        const uint8_t* p = pl + g * 16 + half_idx * 8;
        const uint32_t w0 = ld_u32_unaligned(p), w1 = ld_u32_unaligned(p + 4);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (w0 >> (8 * b)) & 0xFFu;
            put_h2(dst + (2 * b) * SB_FP16_LD, (float)(by & 3u) * eff + bias, (float)((by >> 2) & 3u) * eff + bias);
            put_h2(dst + (2 * b + 1) * SB_FP16_LD, (float)((by >> 4) & 3u) * eff + bias, (float)(by >> 6) * eff + bias);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (w1 >> (8 * b)) & 0xFFu;
            put_h2(dst + (8 + 2 * b) * SB_FP16_LD, (float)(by & 3u) * eff + bias, (float)((by >> 2) & 3u) * eff + bias);
            put_h2(dst + (8 + 2 * b + 1) * SB_FP16_LD, (float)((by >> 4) & 3u) * eff + bias, (float)(by >> 6) * eff + bias);
        }
    } else if constexpr (FMT == FMT_S3) {
        // 32 weights = 8 low bytes + 4 high-bit bytes
        const uint8_t* p = pl + g * 16 + half_idx * 8;
        const uint32_t w0 = ld_u32_unaligned(p), w1 = ld_u32_unaligned(p + 4);
        const uint32_t hb = ld_u32_unaligned(pl + 64 + g * 8 + half_idx * 4);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t by = ((b < 4 ? w0 : w1) >> (8 * (b & 3))) & 0xFFu;
            const uint32_t h4 = (hb >> (4 * b)) & 0xFu;
            const uint32_t q0 = (by & 3u) | ((h4 & 1u) << 2);
            const uint32_t q1 = ((by >> 2) & 3u) | (((h4 >> 1) & 1u) << 2);
            const uint32_t q2 = ((by >> 4) & 3u) | (((h4 >> 2) & 1u) << 2);
            const uint32_t q3 = (by >> 6) | (((h4 >> 3) & 1u) << 2);
            put_h2(dst + (2 * b) * SB_FP16_LD, (float)q0 * eff + bias, (float)q1 * eff + bias);
            put_h2(dst + (2 * b + 1) * SB_FP16_LD, (float)q2 * eff + bias, (float)q3 * eff + bias);
        }
    } else {
        // FMT_S6: 32 low nibbles (16 B) + 32 2-bit highs (8 B)
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint8_t* ph = pl + 128 + g * 16 + half_idx * 8;
        const uint32_t h0 = ld_u32_unaligned(ph), h1 = ld_u32_unaligned(ph + 4);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t wd = ld_u32_unaligned(p + w * 4);
            const uint32_t hw = (w < 2 ? h0 : h1) >> (16 * (w & 1));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                const uint32_t h4 = (hw >> (4 * j)) & 0xFu;
                const uint32_t q0 = (by & 15u) | ((h4 & 3u) << 4);
                const uint32_t q1 = (by >> 4) | (((h4 >> 2) & 3u) << 4);
                put_h2(dst + (w * 4 + j) * SB_FP16_LD, (float)q0 * eff + bias, (float)q1 * eff + bias);
            }
        }
    }
}

// Cooperatively unpack 32 integer weights (half_idx = 0 -> k=0..31, half_idx = 1 -> k=32..63) directly into VNNI-4 packed SLM
// uint32_t sB_u32[16][64]: element (k, n) is byte (k & 3) of sB_u32[k >> 2][n]. Zero float/half instructions!
template <int FMT, bool RP>
inline void unpack_half_group_u8(const uint8_t* pl, int g, int half_idx, uint32_t* sb_u32, int row_local) {
    uint32_t* dst = sb_u32 + (half_idx * 8) * 64 + row_local;
    if constexpr (FMT == FMT_S4) {
        const uint8_t* p = pl + g * 32 + half_idx * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t b0 = p[2 * i], b1 = p[2 * i + 1];
            const uint32_t pack = (b0 & 15u) | ((b0 >> 4) << 8) | ((b1 & 15u) << 16) | ((b1 >> 4) << 24);
            dst[i * 64] = pack;
        }
    } else if constexpr (FMT == FMT_S5) {
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint32_t hb = ld_u32_unaligned(pl + 128 + g * 8 + half_idx * 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t b0 = p[2 * i], b1 = p[2 * i + 1];
            const uint32_t h4 = (hb >> (4 * i)) & 0xFu;
            const uint32_t q0 = (b0 & 15u) | ((h4 & 1u) << 4);
            const uint32_t q1 = (b0 >> 4) | (((h4 >> 1) & 1u) << 4);
            const uint32_t q2 = (b1 & 15u) | (((h4 >> 2) & 1u) << 4);
            const uint32_t q3 = (b1 >> 4) | (((h4 >> 3) & 1u) << 4);
            dst[i * 64] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }
    } else if constexpr (FMT == FMT_S8) {
        const uint8_t* p = pl + g * 64 + half_idx * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i * 64] = ld_u32_unaligned(p + i * 4);
    } else if constexpr (FMT == FMT_S2) {
        const uint8_t* p = pl + g * 16 + half_idx * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t by = p[i];
            dst[i * 64] = (by & 3u) | (((by >> 2) & 3u) << 8) | (((by >> 4) & 3u) << 16) | ((by >> 6) << 24);
        }
    } else if constexpr (FMT == FMT_S3) {
        const uint8_t* p = pl + g * 16 + half_idx * 8;
        const uint32_t hb = ld_u32_unaligned(pl + 64 + g * 8 + half_idx * 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t by = p[i];
            const uint32_t h4 = (hb >> (4 * i)) & 0xFu;
            const uint32_t q0 = (by & 3u) | ((h4 & 1u) << 2);
            const uint32_t q1 = ((by >> 2) & 3u) | (((h4 >> 1) & 1u) << 2);
            const uint32_t q2 = ((by >> 4) & 3u) | (((h4 >> 2) & 1u) << 2);
            const uint32_t q3 = (by >> 6) | (((h4 >> 3) & 1u) << 2);
            dst[i * 64] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }
    } else {
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint8_t* ph = pl + 128 + g * 16 + half_idx * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t b0 = p[2 * i], b1 = p[2 * i + 1];
            const uint32_t h8 = ph[i];
            const uint32_t q0 = (b0 & 15u) | ((h8 & 3u) << 4);
            const uint32_t q1 = (b0 >> 4) | (((h8 >> 2) & 3u) << 4);
            const uint32_t q2 = (b1 & 15u) | (((h8 >> 4) & 3u) << 4);
            const uint32_t q3 = (b1 >> 4) | ((h8 >> 6) << 4);
            dst[i * 64] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }
    }
}

// Persistent scratch for INT8 DPAS activation quantization (grown lazily, never freed inside hot loop)
struct I8ActScratch {
    int8_t* xq = nullptr;
    float* xd = nullptr;
    float* xs = nullptr;
    int8_t* xq2 = nullptr;
    float* xd2 = nullptr;
    size_t cap_mk = 0;
    void ensure(size_t mk, bool two_term) {
        if (mk <= cap_mk && (!two_term || xq2 != nullptr)) return;
        wait();
        if (xq) { dfree(xq); dfree(xd); dfree(xs); }
        if (xq2) { dfree(xq2); dfree(xd2); }
        cap_mk = std::max(mk, (size_t)1024 * 3072);
        xq = dalloc<int8_t>(cap_mk);
        xd = dalloc<float>(cap_mk / 64);
        xs = dalloc<float>(cap_mk / 64);
        if (two_term) {
            xq2 = dalloc<int8_t>(cap_mk);
            xd2 = dalloc<float>(cap_mk / 64);
        } else {
            xq2 = nullptr;
            xd2 = nullptr;
        }
    }
};

inline I8ActScratch& i8_scratch() {
    static I8ActScratch s;
    return s;
}

// Quantize X (half [M][K], K % 64 == 0) into Q8_1_G64 (1-term or 2-term residual) on GPU.
// 1 SG32 subgroup per 64-element block (2 elements per lane).
template <bool TWO_TERM>
inline void quantize_x_q8_1(const half* X, int M, int K, int8_t* xq, float* xd, float* xs, int8_t* xq2, float* xd2) {
    const int ng = K >> 6;
    const size_t total_g = (size_t)M * ng;
    Q().parallel_for(NR((total_g + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const size_t gidx = (size_t)it.get_group(0) * 8 + (it.get_local_id(0) >> 5);
        if (gidx >= total_g) return;
        auto sg = it.get_sub_group();
        const int lane = (int)it.get_local_id(0) & 31;
        const size_t base = gidx << 6;
        const float v0 = (float)X[base + 2 * lane];
        const float v1 = (float)X[base + 2 * lane + 1];
        float amax = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const float d = amax * (1.0f / 127.0f);
        const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
        const int q0 = sycl::clamp((int)sycl::rint(v0 * inv), -127, 127);
        const int q1 = sycl::clamp((int)sycl::rint(v1 * inv), -127, 127);
        xq[base + 2 * lane] = (int8_t)q0;
        xq[base + 2 * lane + 1] = (int8_t)q1;
        const int qsum = sycl::reduce_over_group(sg, q0 + q1, sycl::plus<int>());
        if constexpr (!TWO_TERM) {
            if (lane == 0) {
                xd[gidx] = d;
                xs[gidx] = d * (float)qsum;
            }
        } else {
            const float r0 = v0 - d * (float)q0;
            const float r1 = v1 - d * (float)q1;
            float amax2 = sycl::fmax(sycl::fabs(r0), sycl::fabs(r1));
            amax2 = sycl::reduce_over_group(sg, amax2, sycl::maximum<float>());
            const float d2 = amax2 * (1.0f / 127.0f);
            const float inv2 = amax2 > 0.0f ? (127.0f / amax2) : 0.0f;
            const int b0 = sycl::clamp((int)sycl::rint(r0 * inv2), -127, 127);
            const int b1 = sycl::clamp((int)sycl::rint(r1 * inv2), -127, 127);
            xq2[base + 2 * lane] = (int8_t)b0;
            xq2[base + 2 * lane + 1] = (int8_t)b1;
            const int bsum = sycl::reduce_over_group(sg, b0 + b1, sycl::plus<int>());
            if (lane == 0) {
                xd[gidx] = d;
                xd2[gidx] = d2;
                xs[gidx] = d * (float)qsum + d2 * (float)bsum;
            }
        }
    });
}

}  // namespace xmx_detail

// ---- Path 1: High-occupancy multi-tile FP16 XMX GEMM (Alchemist SG=8, 8x8x16) ----
// BM_TILE in {64, 128}, BN = 64, BK = 64 (8 KiB SLM!). 128 threads = 16 SG8 subgroups (8 M-warps x 2 N-warps).
#ifndef HY4S_DISABLE_SG8
template <int FMT, bool RP, int BM_TILE>
inline void k_gemm_xmx_fp16(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    constexpr int RM = BM_TILE / 64;  // 1 (BM=64) or 2 (BM=128) M-tiles of 8 per subgroup
    const int nbm = (M + BM_TILE - 1) / BM_TILE, nbn = N / 64;
    const size_t pay_off = RP ? rp_pay_off(N, K >> 8) : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sB(32 * xmx_detail::SB_FP16_LD, h);  // 8.7 KiB bank-conflict-free VNNI-2 B tile for BK=64 (max Xe-core occupancy!)
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3;          // 0..15
            const int sg_m = sgi >> 1;         // 0..7
            const int sg_n = sgi & 1;          // 0..1 (each owns 32 N-cols = 4 tiles of 8)
            const int nsb = K >> 8;
            const int row_local = tid >> 1;    // 0..63
            const int half_idx = tid & 1;      // 0 or 1 (32 weights per thread)
            const int n_row = bn * 64 + row_local;

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma[RM];
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[RM][4];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[mi][j], 0.0f);

            int m_base[RM];
            bool act[RM];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi) {
                m_base[mi] = bm * BM_TILE + (mi * 8 + sg_m) * 8;
                act[mi] = (m_base[mi] + 8 <= M);
            }

            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            for (int sbi = 0; sbi < nsb; ++sbi) {
                const uint8_t* hd;
                const uint8_t* pl;
                if constexpr (RP) {
                    hd = W + ((size_t)n_row * nsb + sbi) * 8;
                    pl = W + pay_off + ((size_t)n_row * nsb + sbi) * PB;
                } else {
                    const uint8_t* blk = W + ((size_t)n_row * nsb + sbi) * SB;
                    hd = blk;
                    pl = blk + 6;
                }
                const float D = h2f_bits(ld16(hd));
                const float u0 = (float)hd[2] * (1.0f / 255.0f);
                const float u1 = (float)hd[3] * (1.0f / 255.0f);
                const float u2 = (float)hd[4] * (1.0f / 255.0f);
                const float u3 = (float)hd[5] * (1.0f / 255.0f);
                const float effs[4] = {D * u0, D * u1, D * u2, D * u3};

#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    xmx_detail::dq_half_group_fp16<FMT, RP>(pl, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());

                    const int k0 = sbi * 256 + g * 64;
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
#pragma unroll
                        for (int mi = 0; mi < RM; ++mi) {
                            if (act[mi]) {
                                auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    const_cast<half*>(X + (size_t)m_base[mi] * K + k0 + s * 16));
                                xm::joint_matrix_load(sg, ma[mi], xp, (size_t)K);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * xmx_detail::SB_FP16_LD + (size_t)(sg_n * 4 + j) * 16, (size_t)xmx_detail::SB_FP16_LD);
#pragma unroll
                            for (int mi = 0; mi < RM; ++mi) {
                                if (act[mi]) xm::joint_matrix_mad(sg, mc[mi][j], ma[mi], mb, mc[mi][j]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }

#pragma unroll
            for (int mi = 0; mi < RM; ++mi) {
                if (act[mi]) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            Y + (size_t)m_base[mi] * N + bn * 64 + (sg_n * 4 + j) * 8);
                        xm::joint_matrix_store(sg, mc[mi][j], yp, (size_t)N, xm::layout::row_major);
                    }
                }
            }
        });
    });
}

// ---- Path 2: DPAS INT8 XMX GEMM (Alchemist SG=8, 8x8x32 int8 x uint8/int8 -> int32) ----
// 2x peak XMX throughput (244 TOP/s on A750), 4 KiB VNNI-4 SLM, pure-integer weight unpack, optional 2-term residual ("hi").
template <int FMT, bool RP, bool TWO_TERM>
inline void k_gemm_xmx_i8(const uint8_t* W, int N, int K, const int8_t* Xq, const float* Xd, const float* Xs,
                          const int8_t* Xq2, const float* Xd2, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    constexpr float HVAL = sk_half<FMT>();
    constexpr bool IS_S8 = (FMT == FMT_S8);
    using WEl = std::conditional_t<IS_S8, int8_t, uint8_t>;
    const int nbm = (M + 63) / 64, nbn = N / 64;
    const int ng_tot = K >> 6;
    const size_t pay_off = RP ? rp_pay_off(N, K >> 8) : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sB(16 * 64, h);      // 4 KiB VNNI-4 B tile (BK=64 x BN=64)
        sycl::local_accessor<float, 1> sEff(64, h);            // 256 B per-row eff scale
        sycl::local_accessor<int32_t, 1> sTmp(16 * 64, h);     // 4 KiB scratch for 16 subgroups x (8x8) int32
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, lane = tid & 7;
            const int sg_m = sgi >> 1, sg_n = sgi & 1;
            const int nsb = K >> 8;
            const int row_local = tid >> 1, half_idx = tid & 1;
            const int n_row = bn * 64 + row_local;
            const int m0 = bm * 64 + sg_m * 8;
            const bool active = (m0 + 8 <= M);

            xm::joint_matrix<sycl::sub_group, WEl, xm::use::b, 32, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, int32_t, xm::use::accumulator, 8, 8> mc_i;

            // FP32 accumulators in registers: lane owns column (sg_n*32 + j*8 + lane), 8 rows r=0..7
            float facc[4][8];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int r = 0; r < 8; ++r) facc[j][r] = 0.0f;

            uint32_t* sb_u32 = &sB[0];
            WEl* sb_el = reinterpret_cast<WEl*>(sb_u32);
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_el);
            int32_t* my_tmp = &sTmp[sgi * 64];
            auto tmp_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_tmp);

            for (int sbi = 0; sbi < nsb; ++sbi) {
                const uint8_t* hd;
                const uint8_t* pl;
                if constexpr (RP) {
                    hd = W + ((size_t)n_row * nsb + sbi) * 8;
                    pl = W + pay_off + ((size_t)n_row * nsb + sbi) * PB;
                } else {
                    const uint8_t* blk = W + ((size_t)n_row * nsb + sbi) * SB;
                    hd = blk;
                    pl = blk + 6;
                }
                const float D = h2f_bits(ld16(hd));
                const float effs[4] = {
                    D * ((float)hd[2] * (1.0f / 255.0f)), D * ((float)hd[3] * (1.0f / 255.0f)),
                    D * ((float)hd[4] * (1.0f / 255.0f)), D * ((float)hd[5] * (1.0f / 255.0f))};

#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    if (half_idx == 0) sEff[row_local] = effs[g];
                    xmx_detail::unpack_half_group_u8<FMT, RP>(pl, g, half_idx, sb_u32, row_local);
                    sycl::group_barrier(it.get_group());

                    if (active) {
                        const int gt = sbi * 4 + g;
                        const int k0 = gt * 64;
                        float r_xd[8], r_xs[8], r_xd2[8];
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const size_t gi = (size_t)(m0 + r) * ng_tot + gt;
                            r_xd[r] = Xd[gi];
                            if constexpr (!IS_S8) r_xs[r] = HVAL * Xs[gi];
                            if constexpr (TWO_TERM) r_xd2[r] = Xd2[gi];
                        }
                        auto xp0 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            const_cast<int8_t*>(Xq + (size_t)m0 * K + k0));
                        auto xp1 = xp0 + 32;
                        xm::joint_matrix<sycl::sub_group, int8_t, xm::use::a, 8, 32, xm::layout::row_major> ma0, ma1;
                        xm::joint_matrix_load(sg, ma0, xp0, (size_t)K);
                        xm::joint_matrix_load(sg, ma1, xp1, (size_t)K);

#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int n_col = sg_n * 32 + j * 8 + lane;
                            const float ef = sEff[n_col];
                            xm::joint_matrix_fill(sg, mc_i, 0);
                            xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(sg_n * 4 + j) * 32, (size_t)256);
                            xm::joint_matrix_mad(sg, mc_i, ma0, mb, mc_i);
                            xm::joint_matrix_load(sg, mb, sB_mp + (size_t)8 * 256 + (size_t)(sg_n * 4 + j) * 32, (size_t)256);
                            xm::joint_matrix_mad(sg, mc_i, ma1, mb, mc_i);
                            xm::joint_matrix_store(sg, mc_i, tmp_mp, (size_t)8, xm::layout::row_major);
                            sycl::group_barrier(sg);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                float term = r_xd[r] * (float)my_tmp[r * 8 + lane];
                                if constexpr (!IS_S8) term -= r_xs[r];
                                facc[j][r] += ef * term;
                            }
                        }

                        if constexpr (TWO_TERM) {
                            auto xp20 = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                const_cast<int8_t*>(Xq2 + (size_t)m0 * K + k0));
                            auto xp21 = xp20 + 32;
                            xm::joint_matrix_load(sg, ma0, xp20, (size_t)K);
                            xm::joint_matrix_load(sg, ma1, xp21, (size_t)K);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int n_col = sg_n * 32 + j * 8 + lane;
                                const float ef = sEff[n_col];
                                xm::joint_matrix_fill(sg, mc_i, 0);
                                xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(sg_n * 4 + j) * 32, (size_t)256);
                                xm::joint_matrix_mad(sg, mc_i, ma0, mb, mc_i);
                                xm::joint_matrix_load(sg, mb, sB_mp + (size_t)8 * 256 + (size_t)(sg_n * 4 + j) * 32, (size_t)256);
                                xm::joint_matrix_mad(sg, mc_i, ma1, mb, mc_i);
                                xm::joint_matrix_store(sg, mc_i, tmp_mp, (size_t)8, xm::layout::row_major);
                                sycl::group_barrier(sg);
#pragma unroll
                                for (int r = 0; r < 8; ++r) facc[j][r] += ef * (r_xd2[r] * (float)my_tmp[r * 8 + lane]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }

            if (active) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = bn * 64 + sg_n * 32 + j * 8 + lane;
#pragma unroll
                    for (int r = 0; r < 8; ++r) Y[(size_t)(m0 + r) * N + col] = facc[j][r];
                }
            }
        });
    });
}
#endif  // !HY4S_DISABLE_SG8

// ---- Path 3: Xe2 Battlemage SG=16 XMX GEMM (8x16x16 fp16 -> fp32) ----
#ifdef HY4S_ENABLE_SG16_XMX
template <int FMT, bool RP>
inline void k_gemm_xmx_sg16(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    const int nbm = (M + 63) / 64, nbn = N / 64;
    const size_t pay_off = RP ? rp_pay_off(N, K >> 8) : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sB(32 * xmx_detail::SB_FP16_LD, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4;  // 0..7 (each subgroup owns 8 M-rows x 64 N-cols = 4 tiles of 8x16)
            const int nsb = K >> 8;
            const int row_local = tid >> 1, half_idx = tid & 1;
            const int n_row = bn * 64 + row_local;
            const int m0 = bm * 64 + sgi * 8;
            const bool active = (m0 + 8 <= M);

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);
            for (int sbi = 0; sbi < nsb; ++sbi) {
                const uint8_t* hd = RP ? W + ((size_t)n_row * nsb + sbi) * 8 : W + ((size_t)n_row * nsb + sbi) * SB;
                const uint8_t* pl = RP ? W + pay_off + ((size_t)n_row * nsb + sbi) * PB : hd + 6;
                const float D = h2f_bits(ld16(hd));
                const float effs[4] = {
                    D * ((float)hd[2] * (1.0f / 255.0f)), D * ((float)hd[3] * (1.0f / 255.0f)),
                    D * ((float)hd[4] * (1.0f / 255.0f)), D * ((float)hd[5] * (1.0f / 255.0f))};
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    xmx_detail::dq_half_group_fp16<FMT, RP>(pl, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());
                    if (active) {
                        const int k0 = sbi * 256 + g * 64;
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                const_cast<half*>(X + (size_t)m0 * K + k0 + s * 16));
                            xm::joint_matrix_load(sg, ma, xp, (size_t)K);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * xmx_detail::SB_FP16_LD + (size_t)j * 32, (size_t)xmx_detail::SB_FP16_LD);
                                xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
            if (active) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        Y + (size_t)m0 * N + bn * 64 + j * 16);
                    xm::joint_matrix_store(sg, mc[j], yp, (size_t)N, xm::layout::row_major);
                }
            }
        });
    });
}
#endif

// ---- Path 4: Portable SG32 SIMT tiled GEMM fallback (Intel Core Ultra / UHD iGPUs without XMX) ----
// Tile: BM=32 x BN=64 x BK=64, 128 threads = 4 SG32 sub-groups (each computes 8 rows x 64 cols).
template <int FMT, bool RP>
inline void k_gemm_simt_sg32(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    const int nbm = (M + 31) / 32, nbn = N / 64;
    const size_t pay_off = RP ? rp_pay_off(N, K >> 8) : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sB(32 * xmx_detail::SB_FP16_LD, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 128, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int nsb = K >> 8;
            const int row_local = tid >> 1, half_idx = tid & 1;
            const int n_row = bn * 64 + row_local;
            // Each thread computes 4 M-rows x 2 N-cols? Better: 128 threads = 4 M-groups (8 rows each) x 32 N-pairs (2 cols each = 64 cols)
            const int mg = tid >> 5, nc = (tid & 31) * 2;
            const int m0 = bm * 32 + mg * 8;
            float acc[8][2];
#pragma unroll
            for (int r = 0; r < 8; ++r) acc[r][0] = acc[r][1] = 0.0f;

            half* sb_ptr = &sB[0];
            for (int sbi = 0; sbi < nsb; ++sbi) {
                const uint8_t* hd = RP ? W + ((size_t)n_row * nsb + sbi) * 8 : W + ((size_t)n_row * nsb + sbi) * SB;
                const uint8_t* pl = RP ? W + pay_off + ((size_t)n_row * nsb + sbi) * PB : hd + 6;
                const float D = h2f_bits(ld16(hd));
                const float effs[4] = {
                    D * ((float)hd[2] * (1.0f / 255.0f)), D * ((float)hd[3] * (1.0f / 255.0f)),
                    D * ((float)hd[4] * (1.0f / 255.0f)), D * ((float)hd[5] * (1.0f / 255.0f))};
                for (int g = 0; g < 4; ++g) {
                    xmx_detail::dq_half_group_fp16<FMT, RP>(pl, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());
                    const int k0 = sbi * 256 + g * 64;
                    for (int kp = 0; kp < 32; ++kp) {
                        const half* bp0 = sb_ptr + kp * xmx_detail::SB_FP16_LD + nc * 2;
                        const float b00 = (float)bp0[0], b01 = (float)bp0[1];
                        const float b10 = (float)bp0[2], b11 = (float)bp0[3];
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            if (m0 + r < M) {
                                const half* xp = X + (size_t)(m0 + r) * K + k0 + 2 * kp;
                                const float x0 = (float)xp[0], x1 = (float)xp[1];
                                acc[r][0] += x0 * b00 + x1 * b01;
                                acc[r][1] += x0 * b10 + x1 * b11;
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (m0 + r < M) {
                    Y[(size_t)(m0 + r) * N + bn * 64 + nc] = acc[r][0];
                    Y[(size_t)(m0 + r) * N + bn * 64 + nc + 1] = acc[r][1];
                }
            }
        });
    });
}

// ---- Path 0 (optional, default ON when built with -DHY4S_USE_ONEMKL): Dequant-once S_K -> FP16 + oneMKL XMX GEMM ----
// Eliminates redundant per-M-tile S_K bit-unpacking on prefill (M >= 64), running Intel oneMKL's tuned Xe-HPG/Xe2 XMX GEMM
// directly with FP16 inputs and FP32 accumulation/output (zero quality loss, zero extra VRAM when aliased with Runner's dq_buf).
namespace xmx_detail {

struct DqScratch {
    half* external = nullptr;
    size_t external_cap = 0;
    half* owned = nullptr;
    size_t owned_cap = 0;
    void set_external(half* p, size_t cap_halves) {
        external = p;
        external_cap = cap_halves;
    }
    half* get(size_t& cap_halves) {
        if (external && external_cap > 0) {
            cap_halves = external_cap;
            return external;
        }
        if (!owned) {
            owned_cap = (size_t)16 * 1024 * 1024;  // 32 MiB fallback for standalone unit tests
            owned = dalloc<half>(owned_cap);
        }
        cap_halves = owned_cap;
        return owned;
    }
};

inline DqScratch& dq_scratch() {
    static DqScratch s;
    return s;
}

template <int FMT, bool RP>
inline void k_dequant_sk_to_fp16(const uint8_t* W, int N, int K, int n0, int nc, half* out_fp16) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    const int nsb = K >> 8;
    const size_t total_sb = (size_t)nc * nsb;
    const size_t pay_off = RP ? rp_pay_off(N, nsb) : 0;
    Q().parallel_for(NR((total_sb + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const size_t sb_idx = (size_t)it.get_group(0) * 8 + ((size_t)it.get_local_id(0) >> 5);
        if (sb_idx >= total_sb) return;
        const int lane = (int)it.get_local_id(0) & 31;
        const int r_local = (int)(sb_idx / (size_t)nsb);
        const int sbi = (int)(sb_idx % (size_t)nsb);
        const int row = n0 + r_local;
        const int g = lane >> 3;
        const int u = lane & 7;

        const uint8_t* hd = RP ? (W + ((size_t)row * nsb + sbi) * 8) : (W + ((size_t)row * nsb + sbi) * SB);
        const uint8_t* pl = RP ? (W + pay_off + ((size_t)row * nsb + sbi) * PB) : (hd + 6);
        const float D = h2f_bits(ld16(hd));
        const float eff = D * ((float)hd[2 + g] * (1.0f / 255.0f));
        const float bias = -sk_half<FMT>() * eff;

        sycl::vec<half, 8> vh;
        if constexpr (FMT == FMT_S4) {
            const uint32_t wd = ld_u32_unaligned(pl + g * 32 + u * 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (wd >> (8 * i)) & 0xFFu;
                vh[2 * i] = (half)((float)(by & 15u) * eff + bias);
                vh[2 * i + 1] = (half)((float)(by >> 4) * eff + bias);
            }
        } else if constexpr (FMT == FMT_S5) {
            const uint32_t wd = ld_u32_unaligned(pl + g * 32 + u * 4);
            const uint32_t hb = pl[128 + g * 8 + u];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (wd >> (8 * i)) & 0xFFu;
                const uint32_t q0 = (by & 15u) | (((hb >> (2 * i)) & 1u) << 4);
                const uint32_t q1 = (by >> 4) | (((hb >> (2 * i + 1)) & 1u) << 4);
                vh[2 * i] = (half)((float)q0 * eff + bias);
                vh[2 * i + 1] = (half)((float)q1 * eff + bias);
            }
        } else if constexpr (FMT == FMT_S8) {
            const uint8_t* p = pl + g * 64 + u * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) vh[i] = (half)((float)(int8_t)p[i] * eff);
        } else if constexpr (FMT == FMT_S2) {
            const uint8_t* p = pl + g * 16 + u * 2;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t by = p[i];
                vh[4 * i + 0] = (half)((float)(by & 3u) * eff + bias);
                vh[4 * i + 1] = (half)((float)((by >> 2) & 3u) * eff + bias);
                vh[4 * i + 2] = (half)((float)((by >> 4) & 3u) * eff + bias);
                vh[4 * i + 3] = (half)((float)(by >> 6) * eff + bias);
            }
        } else if constexpr (FMT == FMT_S3) {
            const uint8_t* p = pl + g * 16 + u * 2;
            const uint32_t hb = pl[64 + g * 8 + u];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t by = p[i];
                const uint32_t h4 = (hb >> (4 * i)) & 0xFu;
                const uint32_t q0 = (by & 3u) | ((h4 & 1u) << 2);
                const uint32_t q1 = ((by >> 2) & 3u) | (((h4 >> 1) & 1u) << 2);
                const uint32_t q2 = ((by >> 4) & 3u) | (((h4 >> 2) & 1u) << 2);
                const uint32_t q3 = (by >> 6) | (((h4 >> 3) & 1u) << 2);
                vh[4 * i + 0] = (half)((float)q0 * eff + bias);
                vh[4 * i + 1] = (half)((float)q1 * eff + bias);
                vh[4 * i + 2] = (half)((float)q2 * eff + bias);
                vh[4 * i + 3] = (half)((float)q3 * eff + bias);
            }
        } else {
            // FMT_S6
            const uint32_t wd = ld_u32_unaligned(pl + g * 32 + u * 4);
            const uint32_t hw = (uint32_t)pl[128 + g * 16 + u * 2] | ((uint32_t)pl[128 + g * 16 + u * 2 + 1] << 8);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t by = (wd >> (8 * i)) & 0xFFu;
                const uint32_t h4 = (hw >> (4 * i)) & 0xFu;
                const uint32_t q0 = (by & 15u) | ((h4 & 3u) << 4);
                const uint32_t q1 = (by >> 4) | (((h4 >> 2) & 3u) << 4);
                vh[2 * i] = (half)((float)q0 * eff + bias);
                vh[2 * i + 1] = (half)((float)q1 * eff + bias);
            }
        }
        *reinterpret_cast<sycl::vec<half, 8>*>(out_fp16 + (size_t)r_local * K + (size_t)sbi * 256 + lane * 8) = vh;
    });
}

}  // namespace xmx_detail

inline void register_dq_scratch(half* ptr, size_t cap_halves) {
    xmx_detail::dq_scratch().set_external(ptr, cap_halves);
}

inline int xmx_i8_mode() {
    static const int mode = [] {
        const char* e = std::getenv("HY4S_XMX_I8");
        if (!e) return 0;  // default 0 = fast multi-tile FP16 XMX (rel L2 ~2e-4); 1 = 1-term DPAS INT8; 2 = 2-term DPAS INT8
        return atoi(e);
    }();
    return mode;
}

inline bool onemkl_enabled() {
#ifdef HY4S_USE_ONEMKL
    static const bool on = [] {
        const char* e = std::getenv("HY4S_ONEMKL");
        return !(e && e[0] == '0');
    }();
    return on;
#else
    return false;
#endif
}

#ifdef HY4S_USE_ONEMKL
template <int FMT, bool RP>
inline bool gemm_onemkl_fmt(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    size_t cap_h = 0;
    half* buf = xmx_detail::dq_scratch().get(cap_h);
    int n_step = (int)std::min<size_t>((size_t)N, cap_h / (size_t)K);
    if (n_step >= 64) n_step &= ~63;
    if (n_step <= 0) return false;
    for (int n0 = 0; n0 < N; n0 += n_step) {
        const int nc = std::min(n_step, N - n0);
        xmx_detail::k_dequant_sk_to_fp16<FMT, RP>(W, N, K, n0, nc, buf);
        oneapi::mkl::blas::row_major::gemm(
            Q(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            (std::int64_t)M, (std::int64_t)nc, (std::int64_t)K,
            1.0f, X, (std::int64_t)K,
            buf, (std::int64_t)K,
            0.0f, Y + n0, (std::int64_t)N);
    }
    return true;
}

inline bool gemm_onemkl(int fmt, const void* Wv, int N, int K, const half* X, float* Y, int M, bool rp = false) {
    if (!onemkl_enabled() || M < 32 || (N & 63) || (K & 255) || xmx_i8_mode() != 0) return false;
    const uint8_t* W = (const uint8_t*)Wv;
    if (rp) {
        if (fmt == FMT_S4) return gemm_onemkl_fmt<FMT_S4, true>(W, N, K, X, Y, M);
        if (fmt == FMT_S5) return gemm_onemkl_fmt<FMT_S5, true>(W, N, K, X, Y, M);
        if (fmt == FMT_S8) return gemm_onemkl_fmt<FMT_S8, true>(W, N, K, X, Y, M);
        return false;
    }
    switch (fmt) {
        case FMT_S4: return gemm_onemkl_fmt<FMT_S4, false>(W, N, K, X, Y, M);
        case FMT_S5: return gemm_onemkl_fmt<FMT_S5, false>(W, N, K, X, Y, M);
        case FMT_S6: return gemm_onemkl_fmt<FMT_S6, false>(W, N, K, X, Y, M);
        case FMT_S8: return gemm_onemkl_fmt<FMT_S8, false>(W, N, K, X, Y, M);
        case FMT_S2: return gemm_onemkl_fmt<FMT_S2, false>(W, N, K, X, Y, M);
        case FMT_S3: return gemm_onemkl_fmt<FMT_S3, false>(W, N, K, X, Y, M);
        default: return false;
    }
}
#endif

template <int FMT, bool RP>
inline void dispatch_xmx_fmt(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    const auto& caps = dev_caps();
#ifndef HY4S_DISABLE_SG8
    if (caps.has_xmx_sg8) {
        const int i8m = xmx_i8_mode();
        if (i8m == 1 || i8m == 2) {
            auto& sc = xmx_detail::i8_scratch();
            sc.ensure((size_t)M * K, i8m == 2);
            if (i8m == 1) {
                xmx_detail::quantize_x_q8_1<false>(X, M, K, sc.xq, sc.xd, sc.xs, nullptr, nullptr);
                k_gemm_xmx_i8<FMT, RP, false>(W, N, K, sc.xq, sc.xd, sc.xs, nullptr, nullptr, Y, M);
            } else {
                xmx_detail::quantize_x_q8_1<true>(X, M, K, sc.xq, sc.xd, sc.xs, sc.xq2, sc.xd2);
                k_gemm_xmx_i8<FMT, RP, true>(W, N, K, sc.xq, sc.xd, sc.xs, sc.xq2, sc.xd2, Y, M);
            }
            return;
        }
        if (M >= 128) k_gemm_xmx_fp16<FMT, RP, 128>(W, N, K, X, Y, M);
        else k_gemm_xmx_fp16<FMT, RP, 64>(W, N, K, X, Y, M);
        return;
    }
#endif
#ifdef HY4S_ENABLE_SG16_XMX
    if (caps.has_xmx_sg16) {
        k_gemm_xmx_sg16<FMT, RP>(W, N, K, X, Y, M);
        return;
    }
#endif
    k_gemm_simt_sg32<FMT, RP>(W, N, K, X, Y, M);
}

inline bool gemm_xmx(int fmt, const void* Wv, int N, int K, const half* X, float* Y, int M, bool rp = false) {
#ifdef HY4S_USE_ONEMKL
    if (gemm_onemkl(fmt, Wv, N, K, X, Y, M, rp)) return true;
#endif
    const uint8_t* W = (const uint8_t*)Wv;
    if ((N & 63) || (K & 255) || (M & 7)) return false;
    if (rp) {
        if (fmt == FMT_S4) { dispatch_xmx_fmt<FMT_S4, true>(W, N, K, X, Y, M); return true; }
        if (fmt == FMT_S5) { dispatch_xmx_fmt<FMT_S5, true>(W, N, K, X, Y, M); return true; }
        if (fmt == FMT_S8) { dispatch_xmx_fmt<FMT_S8, true>(W, N, K, X, Y, M); return true; }
        return false;
    }
    switch (fmt) {
        case FMT_S4: dispatch_xmx_fmt<FMT_S4, false>(W, N, K, X, Y, M); return true;
        case FMT_S5: dispatch_xmx_fmt<FMT_S5, false>(W, N, K, X, Y, M); return true;
        case FMT_S6: dispatch_xmx_fmt<FMT_S6, false>(W, N, K, X, Y, M); return true;
        case FMT_S8: dispatch_xmx_fmt<FMT_S8, false>(W, N, K, X, Y, M); return true;
        case FMT_S2: dispatch_xmx_fmt<FMT_S2, false>(W, N, K, X, Y, M); return true;
        case FMT_S3: dispatch_xmx_fmt<FMT_S3, false>(W, N, K, X, Y, M); return true;
        default: return false;
    }
}

// production entry. rp: the weight is in the RP layout (S4/S5/S8 repacked at load time).
// M >= 64 (with oneMKL): dequant-once + oneMKL XMX GEMM for all M rows (no tail!).
// M >= 16 (fallback): fused XMX GEMM for the first (M & ~7) rows, tail by GEMV.
inline void linear_auto(int fmt, const void* W, int N, int K, const half* X, float* Y, int M, bool rp = false) {
#ifdef HY4S_USE_ONEMKL
    if (fmt != FMT_Q8ROW && fmt != FMT_F32 && gemm_onemkl(fmt, W, N, K, X, Y, M, rp)) return;
#endif
    int Mx = 0;
    if (M >= 16 && !(N & 63) && !(K & 255) && fmt != FMT_Q8ROW && fmt != FMT_F32) {
        Mx = M & ~7;
        if (!gemm_xmx(fmt, W, N, K, X, Y, Mx, rp)) Mx = 0;
    }
    for (int m0 = Mx; m0 < M; m0 += 8) {
        const int mm = std::min(8, M - m0);
        if (rp) {
            if (!gemv_rp(fmt, W, N, K, X + (size_t)m0 * K, Y + (size_t)m0 * N, mm)) { fprintf(stderr, "linear_auto: rp fmt %d unsupported\n", fmt); abort(); }
        } else {
            linear_h(fmt, W, N, K, X + (size_t)m0 * K, Y + (size_t)m0 * N, mm);
        }
    }
}

}  // namespace hy4s
