// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_xmx.h - Inkling-2-Mini's products of 16 rows and more on matrix engines: the weights
// dequantized to FP16 by the kernels here, the product through the engine's shared one (prefill::Gemm::f16: oneMath
// in contrib-llvm, XeStrata's own XMX / joint_matrix kernels in free; upstream called oneMKL directly).
//   1. the dense trunk and the vision encoder (BF16, Q8_ROW, S2_K..S8_K) for M >= 16 (prompts, the vision, 16..64 slots)
//   2. the experts (SF4, MX8, S2_K..S8_K and the paired 2-row w2) for an expert with 12 rows or more
// Taken where the device has FP16 matrix engines (strata::kernels::f16_matrix_engines, which STRATA_NO_XMX=1 turns
// off); elsewhere the 16-lane kernels of ink_sy_codecs.h run every product.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "ink_sy_codecs.h"
#include "ink_sy_common.h"
#include "strata/prefill/gemm.hpp"

namespace ink_sycl {

inline bool xmx_enabled() { return dev_caps().matrix_f16; }

// The fewest rows an expert sends through the dequant + GEMM path (fewer: the 16-lane kernels), upstream's 12 on the
// B70; XESTRATA_INKLING_MOE_GEMM_ROWS sets it.
inline int xmx_moe_min_n() {
    static const int v = [] {
        const char* e = strata::env_get("INKLING_MOE_GEMM_ROWS");
        if (e && e[0]) return std::max(1, std::atoi(e));
        return 12;
    }();
    return v;
}

// The shared product on the compute queue (stream null), no dequant scratch of its own (the weights come in FP16).
inline strata::prefill::Gemm& shared_gemm() {
    static std::unique_ptr<strata::prefill::Gemm> g = [] {
        auto p = std::make_unique<strata::prefill::Gemm>();
        std::string err;
        if (!p->init(nullptr, 0, err)) throw std::runtime_error("inkling2: " + err);
        return p;
    }();
    return *g;
}

// Y[M, N] (row stride ldy) = X[M, K] . W[N, K]^T, FP16 in, FP32 out
inline void gemm_f16(const half* X, const half* W, float* Y, int M, int N, int K, int ldy) {
    shared_gemm().f16(reinterpret_cast<const uint16_t*>(X), reinterpret_cast<const uint16_t*>(W), Y, M, N, K, ldy);
}

struct InkSyXmxScratch {
    half* w_f16 = nullptr;      // 16 MiB (8,388,608 halves) -> holds full w13 [2304,2304] or wq [3072,2304]
    size_t w_cap = 0;
    half* x_f16 = nullptr;      // [M, K] activation FP16 staging
    size_t x_cap = 0;
    float* y_gu = nullptr;      // [n, 2304] MoE gate+up output
    size_t y_gu_cap = 0;
    half* h_f16 = nullptr;      // [n, 1152] MoE post-SITU hidden
    size_t h_cap = 0;
    float* y_dn = nullptr;      // [n, 2304] MoE down output
    size_t y_dn_cap = 0;

    ~InkSyXmxScratch() { release(); }

    void release() {
        dfree(w_f16); dfree(x_f16); dfree(y_gu); dfree(h_f16); dfree(y_dn);
        w_f16 = nullptr; x_f16 = nullptr; y_gu = nullptr; h_f16 = nullptr; y_dn = nullptr;
        w_cap = x_cap = y_gu_cap = h_cap = y_dn_cap = 0;
    }

    void ensure_w(size_t need_halves) {
        if (need_halves <= w_cap) return;
        wait();
        dfree(w_f16);
        w_cap = std::max(need_halves, (size_t)8388608);  // 16 MiB
        w_f16 = dalloc<half>(w_cap);
    }

    void ensure_x(size_t need_halves) {
        if (need_halves <= x_cap) return;
        wait();
        dfree(x_f16);
        x_cap = std::max(need_halves, (size_t)512 * 4096);
        x_f16 = dalloc<half>(x_cap);
    }

    void ensure_moe(int n) {
        const size_t need_gu = (size_t)n * 2304;
        const size_t need_h  = (size_t)n * 1152;
        if (need_gu <= y_gu_cap && need_h <= h_cap) return;
        wait();
        dfree(y_gu); dfree(h_f16); dfree(y_dn);
        const size_t cap_n = std::max<size_t>((size_t)n, 512);
        y_gu_cap = cap_n * 2304;
        h_cap    = cap_n * 1152;
        y_dn_cap = cap_n * 2304;
        y_gu  = dalloc<float>(y_gu_cap);
        h_f16 = dalloc<half>(h_cap);
        y_dn  = dalloc<float>(y_dn_cap);
    }
};

inline InkSyXmxScratch& xmx_scratch() {
    static InkSyXmxScratch s;
    return s;
}

// ============================================================================
// 1. Fast SG16 Dequant-to-FP16 Kernels for Dense Trunk & Vision ViT
// ============================================================================

inline void k_f32_to_f16_sg16(const float* src, half* dst, size_t n) {
    Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const size_t i = (size_t)it.get_global_id(0);
        if (i < n) dst[i] = (half)src[i];
    });
}

inline void k_bf16_to_f16_rows_sg16(const uint16_t* W, int n0, int nc, int K, half* dst) {
    const size_t tot = (size_t)nc * K;
    const uint16_t* src = W + (size_t)n0 * K;
    Q().parallel_for(NR((tot + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const size_t i = (size_t)it.get_global_id(0);
        if (i < tot) dst[i] = (half)bf16_to_f32(src[i]);
    });
}

inline void k_q8row_to_f16_rows_sg16(const uint8_t* W, int n0, int nc, int K, int N_total, half* dst) {
    const float* scales = reinterpret_cast<const float*>(W) + n0;
    const int8_t* qw = reinterpret_cast<const int8_t*>(W + (size_t)N_total * 4) + (size_t)n0 * K;
    const size_t tot = (size_t)nc * K;
    Q().parallel_for(NR((tot + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const size_t i = (size_t)it.get_global_id(0);
        if (i < tot) {
            const int r = (int)(i / (size_t)K);
            dst[i] = (half)((float)qw[i] * scales[r]);
        }
    });
}

// 1 SG16 (16 lanes) per 256-weight superblock: each lane unpacks 16 consecutive weights -> 2x 128-bit half8 stores!
template <int F>
inline void k_sk_to_f16_rows_sg16(const uint8_t* W, int n0, int nc, int K, half* dst) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int nsb = K >> 8;
    const size_t total_sb = (size_t)nc * nsb;
    const uint8_t* W0 = W + (size_t)n0 * nsb * SB;
    Q().parallel_for(NR((total_sb + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const size_t sb_idx = (size_t)it.get_group(0) * 16 + ((size_t)it.get_local_id(0) >> 4);
        if (sb_idx >= total_sb) return;
        const int lane = (int)it.get_local_id(0) & 15;
        const int r = (int)(sb_idx / (size_t)nsb);
        const int sbi = (int)(sb_idx % (size_t)nsb);
        const int g = lane >> 2;
        const int j0 = (g << 6) + ((lane & 3) << 4);
        const uint8_t* blk = W0 + sb_idx * SB;
        const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
        const float bias = -H * eff;
        int q0[8], q1[8];
        sk_q8_sy<F>(blk + 6, j0, q0);
        sk_q8_sy<F>(blk + 6, j0 + 8, q1);
        sycl::vec<half, 8> v0, v1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            v0[t] = (half)((float)q0[t] * eff + bias);
            v1[t] = (half)((float)q1[t] * eff + bias);
        }
        half* out = dst + (size_t)r * K + (size_t)sbi * 256 + j0;
        *reinterpret_cast<sycl::vec<half, 8>*>(out)     = v0;
        *reinterpret_cast<sycl::vec<half, 8>*>(out + 8) = v1;
    });
}

// ============================================================================
// 2. Fast Vectorized SG16 Dequant-to-FP16 Kernels for MoE Experts (By-Value!)
// ============================================================================

// Dequant expert w13 [2304, 2304] -> half (row 2r = gate, 2r+1 = up), 128-bit coalesced vector stores!
inline void k_dequant_moe_gu_f16_sg16(InkSyExpertDev ed, half* w_f16) {
    Q().parallel_for(NR(144, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        const uint8_t* body = ed.body_ptr;
        half* dst = w_f16 + (size_t)row * 2304;

        if (ed.gu_code == FMT_SF4) {
            const float* hdr = reinterpret_cast<const float*>(body);
            const float outer = hdr[0] / hdr[1];
            const uint8_t* pb = body + 16;
            const uint8_t* sb = pb + 2654208;
            const uint8_t* pr = pb + (size_t)row * 1152;
            const uint8_t* sr = sb + (size_t)row * 144;
            for (int b = lane; b < 144; b += 16) {
                const float sc = e4m3_to_f32(sr[b]) * outer;
                const uint8_t* bp = pr + b * 8;
                sycl::vec<half, 8> v0, v1;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v0[2 * j]     = (half)(e2m1_lut(bp[j] & 15u) * sc);
                    v0[2 * j + 1] = (half)(e2m1_lut(bp[j] >> 4)  * sc);
                    v1[2 * j]     = (half)(e2m1_lut(bp[4 + j] & 15u) * sc);
                    v1[2 * j + 1] = (half)(e2m1_lut(bp[4 + j] >> 4)  * sc);
                }
                half* o = dst + b * 16;
                *reinterpret_cast<sycl::vec<half, 8>*>(o)     = v0;
                *reinterpret_cast<sycl::vec<half, 8>*>(o + 8) = v1;
            }
        } else if (ed.gu_code == FMT_MX8) {
            const uint8_t* wb = body;
            const uint8_t* sb = body + 5308416;
            const uint8_t* wr = wb + (size_t)row * 2304;
            const uint8_t* sr = sb + (size_t)row * 72;
            for (int b = lane; b < 72; b += 16) {
                const float sc = ue8m0_to_f32(sr[b]);
                const uint8_t* bp = wr + b * 32;
                half* o = dst + b * 32;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sycl::vec<half, 8> v;
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[j] = (half)(e4m3_to_f32(bp[c * 8 + j]) * sc);
                    *reinterpret_cast<sycl::vec<half, 8>*>(o + c * 8) = v;
                }
            }
        } else if (ed.gu_code == FMT_GGUF_RAW) {
            const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
            const float os = (row & 1) ? hdr->up_scale : hdr->gate_scale;
            const uint8_t* rr = body + 32 + (size_t)row * (72 * 34);
            for (int b = lane; b < 72; b += 16) {
                const uint8_t* blk = rr + b * 34;
                const float d = h2f_bits(ld16(blk)) * os;
                half* o = dst + b * 32;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sycl::vec<half, 8> v;
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[j] = (half)((float)(int8_t)blk[2 + c * 8 + j] * d);
                    *reinterpret_cast<sycl::vec<half, 8>*>(o + c * 8) = v;
                }
            }
        } else {
            const int F = ed.gu_code;
            const int SB = sk_sb_bytes(F);
            const float H = sk_half(F);
            const uint8_t* rr = body + 2304 + (size_t)row * 9ul * SB;
            const int g = lane >> 2;
            const int j0 = (g << 6) + ((lane & 3) << 4);
            for (int sbi = 0; sbi < 9; ++sbi) {
                const uint8_t* blk = rr + (size_t)sbi * SB;
                const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                const float bias = -H * eff;
                int q0[8], q1[8];
                if (F == FMT_S2) { sk_q8_sy<FMT_S2>(blk + 6, j0, q0); sk_q8_sy<FMT_S2>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S3) { sk_q8_sy<FMT_S3>(blk + 6, j0, q0); sk_q8_sy<FMT_S3>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S4) { sk_q8_sy<FMT_S4>(blk + 6, j0, q0); sk_q8_sy<FMT_S4>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S5) { sk_q8_sy<FMT_S5>(blk + 6, j0, q0); sk_q8_sy<FMT_S5>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S6) { sk_q8_sy<FMT_S6>(blk + 6, j0, q0); sk_q8_sy<FMT_S6>(blk + 6, j0 + 8, q1); }
                else { sk_q8_sy<FMT_S8>(blk + 6, j0, q0); sk_q8_sy<FMT_S8>(blk + 6, j0 + 8, q1); }
                sycl::vec<half, 8> v0, v1;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    v0[t] = (half)((float)q0[t] * eff + bias);
                    v1[t] = (half)((float)q1[t] * eff + bias);
                }
                half* o = dst + sbi * 256 + j0;
                *reinterpret_cast<sycl::vec<half, 8>*>(o)     = v0;
                *reinterpret_cast<sycl::vec<half, 8>*>(o + 8) = v1;
            }
        }
    });
}

// Dequant expert w2 [2304, 1152] -> half (handles SF4, MX8, and paired 2-row S_K SB_0..SB_8!), 128-bit stores!
inline void k_dequant_moe_dn_f16_sg16(InkSyExpertDev ed, half* w_f16) {
    Q().parallel_for(NR(72, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int rp = (int)it.get_group(0) * 16 + sg;
        const uint8_t* body = ed.body_ptr;
        half* d0 = w_f16 + (size_t)(2 * rp) * 1152;
        half* d1 = w_f16 + (size_t)(2 * rp + 1) * 1152;

        if (ed.dn_code == FMT_SF4) {
            const float* hdr = reinterpret_cast<const float*>(body);
            const float outer = hdr[2] / hdr[3];
            const uint8_t* pb = body + ed.dn_off;
            const uint8_t* sb = pb + 1327104;
            const uint8_t* p0 = pb + (size_t)(2 * rp) * 576;
            const uint8_t* s0 = sb + (size_t)(2 * rp) * 72;
            const uint8_t* p1 = pb + (size_t)(2 * rp + 1) * 576;
            const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 72;
            for (int b = lane; b < 72; b += 16) {
                const float sc0 = e4m3_to_f32(s0[b]) * outer;
                const float sc1 = e4m3_to_f32(s1[b]) * outer;
                const uint8_t* b0 = p0 + b * 8;
                const uint8_t* b1 = p1 + b * 8;
                sycl::vec<half, 8> u0, u1, w0, w1;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    u0[2 * j]     = (half)(e2m1_lut(b0[j] & 15u) * sc0);
                    u0[2 * j + 1] = (half)(e2m1_lut(b0[j] >> 4)  * sc0);
                    u1[2 * j]     = (half)(e2m1_lut(b0[4 + j] & 15u) * sc0);
                    u1[2 * j + 1] = (half)(e2m1_lut(b0[4 + j] >> 4)  * sc0);
                    w0[2 * j]     = (half)(e2m1_lut(b1[j] & 15u) * sc1);
                    w0[2 * j + 1] = (half)(e2m1_lut(b1[j] >> 4)  * sc1);
                    w1[2 * j]     = (half)(e2m1_lut(b1[4 + j] & 15u) * sc1);
                    w1[2 * j + 1] = (half)(e2m1_lut(b1[4 + j] >> 4)  * sc1);
                }
                *reinterpret_cast<sycl::vec<half, 8>*>(d0 + b * 16)     = u0;
                *reinterpret_cast<sycl::vec<half, 8>*>(d0 + b * 16 + 8) = u1;
                *reinterpret_cast<sycl::vec<half, 8>*>(d1 + b * 16)     = w0;
                *reinterpret_cast<sycl::vec<half, 8>*>(d1 + b * 16 + 8) = w1;
            }
        } else if (ed.dn_code == FMT_MX8) {
            const uint8_t* wb = body + ed.dn_off;
            const uint8_t* sb = wb + 2654208;
            const uint8_t* w0 = wb + (size_t)(2 * rp) * 1152;
            const uint8_t* s0 = sb + (size_t)(2 * rp) * 36;
            const uint8_t* w1 = wb + (size_t)(2 * rp + 1) * 1152;
            const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 36;
            for (int b = lane; b < 36; b += 16) {
                const float sc0 = ue8m0_to_f32(s0[b]);
                const float sc1 = ue8m0_to_f32(s1[b]);
                const uint8_t* b0 = w0 + b * 32;
                const uint8_t* b1 = w1 + b * 32;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sycl::vec<half, 8> u, v;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        u[j] = (half)(e4m3_to_f32(b0[c * 8 + j]) * sc0);
                        v[j] = (half)(e4m3_to_f32(b1[c * 8 + j]) * sc1);
                    }
                    *reinterpret_cast<sycl::vec<half, 8>*>(d0 + b * 32 + c * 8) = u;
                    *reinterpret_cast<sycl::vec<half, 8>*>(d1 + b * 32 + c * 8) = v;
                }
            }
        } else if (ed.dn_code == FMT_GGUF_RAW) {
            const auto* hdr = reinterpret_cast<const strata::quant::sk::GgufExpertHeader*>(body);
            const uint8_t* w0 = body + ed.dn_off + (size_t)(2 * rp) * (36 * 34);
            const uint8_t* w1 = w0 + 36 * 34;
            for (int b = lane; b < 36; b += 16) {
                const uint8_t* b0 = w0 + b * 34;
                const uint8_t* b1 = w1 + b * 34;
                const float s0 = h2f_bits(ld16(b0)) * hdr->dn_scale;
                const float s1 = h2f_bits(ld16(b1)) * hdr->dn_scale;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sycl::vec<half, 8> u, v;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        u[j] = (half)((float)(int8_t)b0[2 + c * 8 + j] * s0);
                        v[j] = (half)((float)(int8_t)b1[2 + c * 8 + j] * s1);
                    }
                    *reinterpret_cast<sycl::vec<half, 8>*>(d0 + b * 32 + c * 8) = u;
                    *reinterpret_cast<sycl::vec<half, 8>*>(d1 + b * 32 + c * 8) = v;
                }
            }
        } else {
            const int F = ed.dn_code;
            const int SB = sk_sb_bytes(F);
            const float H = sk_half(F);
            const uint8_t* pr = body + ed.dn_off + (size_t)rp * 9ul * SB;
            const int g = lane >> 2;
            const int j0 = (g << 6) + ((lane & 3) << 4);
            for (int sbi = 0; sbi < 9; ++sbi) {
                const uint8_t* blk = pr + (size_t)sbi * SB;
                const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                const float bias = -H * eff;
                int q0[8], q1[8];
                if (F == FMT_S2) { sk_q8_sy<FMT_S2>(blk + 6, j0, q0); sk_q8_sy<FMT_S2>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S3) { sk_q8_sy<FMT_S3>(blk + 6, j0, q0); sk_q8_sy<FMT_S3>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S4) { sk_q8_sy<FMT_S4>(blk + 6, j0, q0); sk_q8_sy<FMT_S4>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S5) { sk_q8_sy<FMT_S5>(blk + 6, j0, q0); sk_q8_sy<FMT_S5>(blk + 6, j0 + 8, q1); }
                else if (F == FMT_S6) { sk_q8_sy<FMT_S6>(blk + 6, j0, q0); sk_q8_sy<FMT_S6>(blk + 6, j0 + 8, q1); }
                else { sk_q8_sy<FMT_S8>(blk + 6, j0, q0); sk_q8_sy<FMT_S8>(blk + 6, j0 + 8, q1); }
                sycl::vec<half, 8> v0, v1;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    v0[t] = (half)((float)q0[t] * eff + bias);
                    v1[t] = (half)((float)q1[t] * eff + bias);
                }
                const int flat0 = sbi * 256 + j0;
                half* o = (flat0 < 1152) ? (d0 + flat0) : (d1 + (flat0 - 1152));
                *reinterpret_cast<sycl::vec<half, 8>*>(o)     = v0;
                *reinterpret_cast<sycl::vec<half, 8>*>(o + 8) = v1;
            }
        }
    });
}

inline void k_moe_gather_f16_sg16(const float* x_in, const int* pair_list, int off, int n, half* x_f16) {
    Q().parallel_for(NR(n, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int i = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int tk = pair_list[off + i] >> 3;
        const float* src = x_in + (size_t)tk * 2304;
        half* dst = x_f16 + (size_t)i * 2304;
        for (int d = tid; d < 2304; d += 256) dst[d] = (half)src[d];
    });
}

inline void k_moe_situ_invsq_f16_sg16(InkSyExpertDev ed, const float* y_gu, int n, half* h_f16) {
    Q().parallel_for(NR(n, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int i = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const uint8_t* body = ed.body_ptr;
        const bool is_sk_dn = (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8 && ed.dn_code != FMT_GGUF_RAW);
        const float* src = y_gu + (size_t)i * 2304;
        half* dst = h_f16 + (size_t)i * 1152;
        for (int r = tid; r < 1152; r += 256) {
            const float inv_sq = is_sk_dn ? (1.0f / h2f_bits(ld16(body + 2 * r))) : 1.0f;
            dst[r] = (half)(situ_act(src[2 * r], src[2 * r + 1]) * inv_sq);
        }
    });
}

inline sycl::event k_moe_scatter_w_sg16(const float* y_dn, const float* topk_w, const int* pair_list, int off, int n, float* y_pairs) {
    return Q().parallel_for(NR(n, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int i = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int pid = pair_list[off + i];
        const float rw = topk_w[pid];
        const float* src = y_dn + (size_t)i * 2304;
        float* dst = y_pairs + (size_t)pid * 2304;
        for (int d = tid; d < 2304; d += 256) dst[d] = src[d] * rw;
    });
}

// ============================================================================
// 3. The dequant + shared GEMM entry points (dense linear, vision, experts)
// ============================================================================

inline bool ink_xmx_linear_bf16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
    if (!xmx_enabled() || M < 16) return false;
    auto& sc = xmx_scratch();
    sc.ensure_w((size_t)std::min<int64_t>((int64_t)N * K, 8388608));
    sc.ensure_x((size_t)M * K);
    k_f32_to_f16_sg16(X, sc.x_f16, (size_t)M * K);
    int n_step = (int)std::min<size_t>((size_t)N, sc.w_cap / (size_t)K);
    if (n_step >= 64) n_step &= ~63;
    if (n_step <= 0) return false;
    for (int n0 = 0; n0 < N; n0 += n_step) {
        const int nc = std::min(n_step, N - n0);
        k_bf16_to_f16_rows_sg16(W, n0, nc, K, sc.w_f16);
        gemm_f16(sc.x_f16, sc.w_f16, Y + n0, M, nc, K, N);
    }
    return true;
}

inline void k_q8_0_to_f16_rows_sg16(const uint8_t* W, int n0, int nc, int K, int N_total, half* dst) {
    const size_t tot = (size_t)nc * K;
    const float* sc = reinterpret_cast<const float*>(W) + (size_t)n0 * (K >> 5);
    const int8_t* q = reinterpret_cast<const int8_t*>(W + (size_t)N_total * (K >> 5) * 4) + (size_t)n0 * K;
    Q().parallel_for(NR((tot + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const size_t i = (size_t)it.get_global_id(0);
        if (i < tot) dst[i] = (half)((float)q[i] * sc[i >> 5]);
    });
}

inline bool ink_xmx_linear_q8_0(const uint8_t* W, int N, int K, const float* X, float* Y, int M, int N_total) {
    if (!xmx_enabled() || M < 16) return false;
    auto& sc = xmx_scratch();
    sc.ensure_w((size_t)std::min<int64_t>((int64_t)N * K, 8388608));
    sc.ensure_x((size_t)M * K);
    k_f32_to_f16_sg16(X, sc.x_f16, (size_t)M * K);
    int n_step = (int)std::min<size_t>((size_t)N, sc.w_cap / (size_t)K);
    if (n_step >= 64) n_step &= ~63;
    if (n_step <= 0) return false;
    for (int n0 = 0; n0 < N; n0 += n_step) {
        const int nc = std::min(n_step, N - n0);
        k_q8_0_to_f16_rows_sg16(W, n0, nc, K, N_total, sc.w_f16);
        gemm_f16(sc.x_f16, sc.w_f16, Y + n0, M, nc, K, N);
    }
    return true;
}

inline bool ink_xmx_linear_q8row(const uint8_t* W, int N, int K, const float* X, float* Y, int M, int N_total) {
    if (!xmx_enabled() || M < 16) return false;
    auto& sc = xmx_scratch();
    sc.ensure_w((size_t)std::min<int64_t>((int64_t)N * K, 8388608));
    sc.ensure_x((size_t)M * K);
    k_f32_to_f16_sg16(X, sc.x_f16, (size_t)M * K);
    int n_step = (int)std::min<size_t>((size_t)N, sc.w_cap / (size_t)K);
    if (n_step >= 64) n_step &= ~63;
    if (n_step <= 0) return false;
    for (int n0 = 0; n0 < N; n0 += n_step) {
        const int nc = std::min(n_step, N - n0);
        k_q8row_to_f16_rows_sg16(W, n0, nc, K, N_total, sc.w_f16);
        gemm_f16(sc.x_f16, sc.w_f16, Y + n0, M, nc, K, N);
    }
    return true;
}

template <int F>
inline bool ink_xmx_linear_sk(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    if (!xmx_enabled() || M < 16 || (K & 255)) return false;
    auto& sc = xmx_scratch();
    sc.ensure_w((size_t)std::min<int64_t>((int64_t)N * K, 8388608));
    sc.ensure_x((size_t)M * K);
    k_f32_to_f16_sg16(X, sc.x_f16, (size_t)M * K);
    int n_step = (int)std::min<size_t>((size_t)N, sc.w_cap / (size_t)K);
    if (n_step >= 64) n_step &= ~63;
    if (n_step <= 0) return false;
    for (int n0 = 0; n0 < N; n0 += n_step) {
        const int nc = std::min(n_step, N - n0);
        k_sk_to_f16_rows_sg16<F>(W, n0, nc, K, sc.w_f16);
        gemm_f16(sc.x_f16, sc.w_f16, Y + n0, M, nc, K, N);
    }
    return true;
}

// One expert through the dequant + shared GEMM path, its metadata by value
inline bool ink_xmx_moe_single(
    InkSyExpertDev ed, const float* x_in, const float* topk_w,
    const int* pair_list, int off, int n, float* y_pairs, sycl::event* out_ev = nullptr) {
    if (!xmx_enabled() || n < xmx_moe_min_n()) return false;
    auto& sc = xmx_scratch();
    sc.ensure_w(2304ul * 2304ul);
    sc.ensure_x((size_t)n * 2304);
    sc.ensure_moe(n);

    k_moe_gather_f16_sg16(x_in, pair_list, off, n, sc.x_f16);
    k_dequant_moe_gu_f16_sg16(ed, sc.w_f16);
    gemm_f16(sc.x_f16, sc.w_f16, sc.y_gu, n, 2304, 2304, 2304);
    k_moe_situ_invsq_f16_sg16(ed, sc.y_gu, n, sc.h_f16);
    k_dequant_moe_dn_f16_sg16(ed, sc.w_f16);
    gemm_f16(sc.h_f16, sc.w_f16, sc.y_dn, n, 2304, 1152, 2304);
    sycl::event ev = k_moe_scatter_w_sg16(sc.y_dn, topk_w, pair_list, off, n, y_pairs);
    if (out_ev) *out_ev = ev;
    return true;
}

}  // namespace ink_sycl
