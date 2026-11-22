// sy_vram_cache.h - Multi-slab (<= 1.4 GiB/slab) Adaptive VRAM Expert Cache + Optimized GPU Decode/Verify evaluator for Intel Arc (SYCL).
// Key optimizations:
//   1. Pre-quantizes z_dev[0..T-1] and hmid[0..npairs-1] ONCE to Q8_1_G64 (16-lane SG16 kernel), eliminating 3,072+ redundant
//      quantizations per expert in GEMM1 and 1,536 per pair in GEMM2.
//   2. Deduplicates VRAM hits by DISTINCT EXPERT across verify tokens T = 1..16 (VramExpertGroup, up to 8 tokens/group),
//      reading each hot expert's weights ONCE per verify window instead of 6-8x redundantly.
//   3. Caches pre-quantized token activations in SLM (<= 12 KiB/WG) shared across all 16 SG16 output rows per workgroup,
//      fuses gate+up in GEMM1, uses exact int32 dot accumulation, and dispatches native SG16 (Xe2 Battlemage & Alchemist).
// Bitwise determinism (plain == DFly2):
//   * Every (token, expert) pair computes identical per-lane sums and SG16 reduction order regardless of window size T in [1..16].
//   * Per-token reduction sums pairs in ascending expert_id order.
//   * Adaptive slot promotions are committed ONLY between steps, never mid-step.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "hy4_io.h"
#include "strata/artifact/strata_reader.hpp"
#include "sy_common.h"

namespace hy4s {

namespace vcache_detail {

constexpr int VC_K = 1536;
constexpr int VC_NG = VC_K / 64;  // 24 groups of 64
constexpr int VC_NTOP = 16;
constexpr int VC_GRP_MAX_T = 8;

constexpr int8_t kS125[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
    {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
    {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},  {0, 0, 2, 0},    {0, 0, -2, 0},  {0, 0, 0, 2},   {0, 0, 0, -2},
    {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},   {0, 0, -2, -2},  {2, -2, 0, 0},  {0, 0, 2, -2},  {2, 0, 2, 0},
};

inline int vc_sb_bytes(int code) {
    return code == 0 ? 48 : code == 1 ? 58 : code == 2 ? 70 : code == 3 ? 102 : code == 4 ? 134 : code == 5 ? 166 : code == 6 ? 198 : 262;
}

inline int vc_s15_trit(uint32_t b, int d) {
    if (b >= 243u) return 0;
    const uint32_t m = d == 0 ? 65536u : d == 1 ? 21846u : d == 2 ? 7282u : d == 3 ? 2428u : 810u;
    const uint32_t x = (b * m) >> 16;
    const uint32_t x3 = (x * 21846u) >> 16;
    return (int)(x - 3u * x3) - 1;
}

inline uint32_t vc_ld_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Unpack 16 quantized integer weights at base_i = (g << 6) + (lane & 3) * 16 of superblock `blk` for code 0..7.
// Returns eff scale (including fine scale for code == 0) and zero-point half-offset H (0 for signed codes 0,1,7).
inline void vc_unpack16(int code, const uint8_t* blk, int g, int base_i, int8_t (&wq)[16], float& eff_out, float& h_out) {
    const float D = h2f_bits(ld16(blk));
    float eff = D * ((float)blk[2 + g] * (1.0f / 255.0f));
    const uint8_t* pl = blk + 6;
    h_out = 0.0f;
    if (code == 2) {
        const uint32_t wd = vc_ld_u32(pl + (base_i >> 2));
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (wd >> (8 * b)) & 0xFFu;
            wq[4 * b + 0] = (int8_t)(by & 3u);
            wq[4 * b + 1] = (int8_t)((by >> 2) & 3u);
            wq[4 * b + 2] = (int8_t)((by >> 4) & 3u);
            wq[4 * b + 3] = (int8_t)(by >> 6);
        }
        h_out = 1.5f;
    } else if (code == 3) {
        const uint32_t wd = vc_ld_u32(pl + (base_i >> 2));
        const uint32_t hi = (uint32_t)pl[64 + (base_i >> 3)] | ((uint32_t)pl[64 + (base_i >> 3) + 1] << 8);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (wd >> (8 * b)) & 0xFFu;
            const uint32_t h4 = (hi >> (4 * b)) & 0xFu;
            wq[4 * b + 0] = (int8_t)((by & 3u) | ((h4 & 1u) << 2));
            wq[4 * b + 1] = (int8_t)(((by >> 2) & 3u) | (((h4 >> 1) & 1u) << 2));
            wq[4 * b + 2] = (int8_t)(((by >> 4) & 3u) | (((h4 >> 2) & 1u) << 2));
            wq[4 * b + 3] = (int8_t)((by >> 6) | (((h4 >> 3) & 1u) << 2));
        }
        h_out = 3.5f;
    } else if (code == 4) {
        const uint32_t w0 = vc_ld_u32(pl + (base_i >> 1));
        const uint32_t w1 = vc_ld_u32(pl + (base_i >> 1) + 4);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t b0 = (w0 >> (8 * b)) & 0xFFu;
            const uint32_t b1 = (w1 >> (8 * b)) & 0xFFu;
            wq[2 * b + 0] = (int8_t)(b0 & 15u);
            wq[2 * b + 1] = (int8_t)(b0 >> 4);
            wq[8 + 2 * b + 0] = (int8_t)(b1 & 15u);
            wq[8 + 2 * b + 1] = (int8_t)(b1 >> 4);
        }
        h_out = 7.5f;
    } else if (code == 0) {
        const float fine = 0.5f + (float)((pl[40 + (g >> 1)] >> ((g & 1) * 4)) & 15u) * (1.0f / 15.0f);
        eff *= fine;
        const int grp = base_i >> 5;
        const uint8_t* p = pl + grp * 5;
        const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
        const int j_sub = (base_i & 31) >> 2;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t idx = (uint32_t)((bits >> (5 * (j_sub + q))) & 0x1Fu);
            wq[4 * q + 0] = kS125[idx][0];
            wq[4 * q + 1] = kS125[idx][1];
            wq[4 * q + 2] = kS125[idx][2];
            wq[4 * q + 3] = kS125[idx][3];
        }
    } else if (code == 1) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int idx = base_i + k;
            wq[k] = (int8_t)vc_s15_trit(pl[idx / 5], idx % 5);
        }
    } else if (code == 5) {
        const uint32_t w0 = vc_ld_u32(pl + (base_i >> 1));
        const uint32_t w1 = vc_ld_u32(pl + (base_i >> 1) + 4);
        const uint32_t hi = (uint32_t)pl[128 + (base_i >> 3)] | ((uint32_t)pl[128 + (base_i >> 3) + 1] << 8);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t b0 = (w0 >> (8 * b)) & 0xFFu;
            const uint32_t b1 = (w1 >> (8 * b)) & 0xFFu;
            wq[2 * b + 0] = (int8_t)((b0 & 15u) | (((hi >> (2 * b)) & 1u) << 4));
            wq[2 * b + 1] = (int8_t)((b0 >> 4) | (((hi >> (2 * b + 1)) & 1u) << 4));
            wq[8 + 2 * b + 0] = (int8_t)((b1 & 15u) | (((hi >> (8 + 2 * b)) & 1u) << 4));
            wq[8 + 2 * b + 1] = (int8_t)((b1 >> 4) | (((hi >> (8 + 2 * b + 1)) & 1u) << 4));
        }
        h_out = 15.5f;
    } else if (code == 6) {
        const uint32_t w0 = vc_ld_u32(pl + (base_i >> 1));
        const uint32_t w1 = vc_ld_u32(pl + (base_i >> 1) + 4);
        const uint32_t hi = vc_ld_u32(pl + 128 + (base_i >> 2));
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t b0 = (w0 >> (8 * b)) & 0xFFu;
            const uint32_t b1 = (w1 >> (8 * b)) & 0xFFu;
            wq[2 * b + 0] = (int8_t)((b0 & 15u) | (((hi >> (4 * b)) & 3u) << 4));
            wq[2 * b + 1] = (int8_t)((b0 >> 4) | (((hi >> (4 * b + 2)) & 3u) << 4));
            wq[8 + 2 * b + 0] = (int8_t)((b1 & 15u) | (((hi >> (16 + 4 * b)) & 3u) << 4));
            wq[8 + 2 * b + 1] = (int8_t)((b1 >> 4) | (((hi >> (16 + 4 * b + 2)) & 3u) << 4));
        }
        h_out = 31.5f;
    } else {
        const uint8_t* p = pl + base_i;
#pragma unroll
        for (int k = 0; k < 16; ++k) wq[k] = (int8_t)p[k];
    }
    eff_out = eff;
}

// Pre-quantize M rows of VC_K (1536) activations into Q8_1_G64 (identical to hy4::CpuExperts: d = amax/127, q = clamp(rint(x/d), -127, 127)).
// 1 SG16 subgroup per 64-element group (4 elements/lane), 16 groups per 256-thread WG.
template <class ActT>
inline void k_quant_q8_1_g64(const ActT* src, int M, int8_t* dst_q, float* dst_d) {
    const int total_g = M * VC_NG;
    Q().parallel_for(NR((total_g + 15) / 16, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
        const int gidx = (int)it.get_group(0) * 16 + ((int)it.get_local_id(0) >> 4);
        if (gidx >= total_g) return;
        auto sg = it.get_sub_group();
        const int lane = (int)it.get_local_id(0) & 15;
        const size_t base = (size_t)gidx * 64 + lane * 4;
        const float v0 = (float)src[base + 0];
        const float v1 = (float)src[base + 1];
        const float v2 = (float)src[base + 2];
        const float v3 = (float)src[base + 3];
        float amax = sycl::fmax(sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::fmax(sycl::fabs(v2), sycl::fabs(v3)));
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const float d = amax * (1.0f / 127.0f);
        const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
        sycl::vec<int8_t, 4> qv;
        qv[0] = (int8_t)sycl::clamp((int)sycl::rint(v0 * inv), -127, 127);
        qv[1] = (int8_t)sycl::clamp((int)sycl::rint(v1 * inv), -127, 127);
        qv[2] = (int8_t)sycl::clamp((int)sycl::rint(v2 * inv), -127, 127);
        qv[3] = (int8_t)sycl::clamp((int)sycl::rint(v3 * inv), -127, 127);
        *reinterpret_cast<sycl::vec<int8_t, 4>*>(dst_q + base) = qv;
        if (lane == 0) dst_d[gidx] = d;
    });
}

// Distinct-expert batch descriptor for decode/verify: evaluates up to 8 tokens sharing one VRAM expert in a single weight pass!
struct VramExpertGroup {
    const uint8_t* body;
    uint32_t dn_off;
    int32_t gu;
    int32_t dn;
    int32_t ntok;
    int32_t pad0;
    int32_t tok[VC_GRP_MAX_T];       // token index [0..T-1]
    int32_t pair_idx[VC_GRP_MAX_T];  // output pair slot [0..npairs-1]
};

// Peer-GPU (`sy_peer.h`) SG32 row dot product & pair descriptor (supported on both Alchemist & Battlemage).
template <class ActT, bool Q8_ACT>
inline float sg_dot_row(const sycl::nd_item<1>& it, int code, const uint8_t* rp, const ActT* x) {
    auto sg = it.get_sub_group();
    const int lane = (int)it.get_local_id(0) & 31;
    const int sby = vc_sb_bytes(code);
    const int g = lane >> 3;
    const int j0 = (lane & 7) << 3;
    const int base_i = (g << 6) + j0;
    float acc = 0.0f;
#pragma unroll
    for (int sbi = 0; sbi < 6; ++sbi) {
        const uint8_t* blk = rp + (size_t)sbi * sby;
        const float D = h2f_bits(ld16(blk));
        const float eff = D * ((float)blk[2 + g] / 255.0f);
        const uint8_t* pl = blk + 6;
        const ActT* xp = x + sbi * 256 + base_i;
        float xv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) xv[k] = (float)xp[k];
        float scale_x = 1.0f;
        if constexpr (Q8_ACT) {
            float amax = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) amax = sycl::fmax(amax, sycl::fabs(xv[k]));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 4));
            scale_x = amax * (1.0f / 127.0f);
            const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) xv[k] = (float)sycl::clamp((int)sycl::rint(xv[k] * inv), -127, 127);
        }
        const float effx = eff * scale_x;
        float sq = 0.0f, sx = 0.0f;
        if (code == 2) {
            const uint8_t b0 = pl[base_i >> 2], b1 = pl[(base_i >> 2) + 1];
            sq = (float)(b0 & 3u) * xv[0] + (float)((b0 >> 2) & 3u) * xv[1] + (float)((b0 >> 4) & 3u) * xv[2] + (float)(b0 >> 6) * xv[3] +
                 (float)(b1 & 3u) * xv[4] + (float)((b1 >> 2) & 3u) * xv[5] + (float)((b1 >> 4) & 3u) * xv[6] + (float)(b1 >> 6) * xv[7];
            sx = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
            acc += effx * (sq - 1.5f * sx);
        } else if (code == 3) {
            const uint8_t b0 = pl[base_i >> 2], b1 = pl[(base_i >> 2) + 1];
            const uint32_t hi = pl[64 + (base_i >> 3)];
            sq = (float)((b0 & 3u) | ((hi & 1u) << 2)) * xv[0] + (float)(((b0 >> 2) & 3u) | (((hi >> 1) & 1u) << 2)) * xv[1] +
                 (float)(((b0 >> 4) & 3u) | (((hi >> 2) & 1u) << 2)) * xv[2] + (float)((b0 >> 6) | (((hi >> 3) & 1u) << 2)) * xv[3] +
                 (float)((b1 & 3u) | (((hi >> 4) & 1u) << 2)) * xv[4] + (float)(((b1 >> 2) & 3u) | (((hi >> 5) & 1u) << 2)) * xv[5] +
                 (float)(((b1 >> 4) & 3u) | (((hi >> 6) & 1u) << 2)) * xv[6] + (float)((b1 >> 6) | (((hi >> 7) & 1u) << 2)) * xv[7];
            sx = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
            acc += effx * (sq - 3.5f * sx);
        } else if (code == 4) {
            const uint8_t* p = pl + (base_i >> 1);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                sq += (float)(by & 15u) * xv[2 * k] + (float)(by >> 4) * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 7.5f * sx);
        } else if (code == 0) {
            const float fine = 0.5f + (float)((pl[40 + (g >> 1)] >> ((g & 1) * 4)) & 15u) / 15.0f;
            const int grp = base_i >> 5;
            const uint8_t* p = pl + grp * 5;
            const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
            const int j_sub = (base_i & 31) >> 2;
            const uint32_t i0 = (uint32_t)((bits >> (5 * j_sub)) & 0x1Fu);
            const uint32_t i1 = (uint32_t)((bits >> (5 * (j_sub + 1))) & 0x1Fu);
            sq = (float)kS125[i0][0] * xv[0] + (float)kS125[i0][1] * xv[1] + (float)kS125[i0][2] * xv[2] + (float)kS125[i0][3] * xv[3] +
                 (float)kS125[i1][0] * xv[4] + (float)kS125[i1][1] * xv[5] + (float)kS125[i1][2] * xv[6] + (float)kS125[i1][3] * xv[7];
            acc += effx * fine * sq;
        } else if (code == 1) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int idx = base_i + k;
                sq += (float)vc_s15_trit(pl[idx / 5], idx % 5) * xv[k];
            }
            acc += effx * sq;
        } else if (code == 5) {
            const uint8_t* p = pl + (base_i >> 1);
            const uint32_t hi = pl[128 + (base_i >> 3)];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                const uint32_t q0 = (by & 15u) | (((hi >> (2 * k)) & 1u) << 4);
                const uint32_t q1 = (by >> 4) | (((hi >> (2 * k + 1)) & 1u) << 4);
                sq += (float)q0 * xv[2 * k] + (float)q1 * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 15.5f * sx);
        } else if (code == 6) {
            const uint8_t* p = pl + (base_i >> 1);
            const uint32_t h0 = pl[128 + (base_i >> 2)];
            const uint32_t h1 = pl[128 + (base_i >> 2) + 1];
            const uint32_t h16 = h0 | (h1 << 8);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                const uint32_t q0 = (by & 15u) | (((h16 >> (4 * k)) & 3u) << 4);
                const uint32_t q1 = (by >> 4) | (((h16 >> (4 * k + 2)) & 3u) << 4);
                sq += (float)q0 * xv[2 * k] + (float)q1 * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 31.5f * sx);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) sq += (float)(int8_t)pl[base_i + k] * xv[k];
            acc += effx * sq;
        }
    }
    return sg_sum(it, acc);
}

struct VramPair {
    const uint8_t* body;
    uint32_t dn_off;
    int32_t tok;
    int32_t gu;
    int32_t dn;
    float weight;
};

// GEMM1 (fused gate + up + SwiGLU/sq) with pre-quantized Q8_1_G64 activations cached in SLM (<= 12 KiB/WG) + native SG16.
// 96 WGs/expert x 256 threads (16 SG16 rows/WG) -> 1,536 rows. Each weight row loaded ONCE across all ntok <= NT tokens!
template <int NT>
inline void k_vcache_gemm1_q8(const VramExpertGroup* d_grps, int ngrps, const int8_t* d_zq, const float* d_zd, half* d_hmid) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> sXq(NT * VC_K, h);
        sycl::local_accessor<float, 1> sXd(NT * VC_NG, h);
        h.parallel_for(NR((size_t)ngrps * 96, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int gi = gid / 96, rb = gid % 96;
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 4, lane = tid & 15;
            const int d = rb * 16 + wave;
            const VramExpertGroup vg = d_grps[gi];
            const int ntok = vg.ntok;

            // Cooperatively load pre-quantized Q8 activations for the ntok tokens of this expert into SLM (16x global read reduction!)
            int8_t* sxq_ptr = &sXq[0];
            float* sxd_ptr = &sXd[0];
            const int total_v4 = (ntok * VC_K) >> 2;
            for (int idx = tid; idx < total_v4; idx += 256) {
                const int m = idx / (VC_K >> 2);
                const int off4 = (idx % (VC_K >> 2)) << 2;
                *reinterpret_cast<sycl::vec<int8_t, 4>*>(sxq_ptr + m * VC_K + off4) =
                    *reinterpret_cast<const sycl::vec<int8_t, 4>*>(d_zq + (size_t)vg.tok[m] * VC_K + off4);
            }
            const int total_d = ntok * VC_NG;
            for (int idx = tid; idx < total_d; idx += 256) {
                const int m = idx / VC_NG, g = idx % VC_NG;
                sxd_ptr[idx] = d_zd[(size_t)vg.tok[m] * VC_NG + g];
            }
            sycl::group_barrier(it.get_group());

            const int code = vg.gu;
            const int sby = vc_sb_bytes(code);
            const size_t row_b = (size_t)6 * sby;
            const uint8_t* wgu = vg.body + 3072;
            const uint8_t* rp_g = wgu + (size_t)d * row_b;
            const uint8_t* rp_u = wgu + (size_t)(VC_K + d) * row_b;
            const int g = lane >> 2;
            const int base_i = (g << 6) + ((lane & 3) << 4);

            float acc_g[NT], acc_u[NT];
#pragma unroll
            for (int m = 0; m < NT; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

#pragma unroll 1
            for (int sbi = 0; sbi < 6; ++sbi) {
                int8_t wg[16], wu[16];
                float eff_g = 0.0f, h_g = 0.0f, eff_u = 0.0f, h_u = 0.0f;
                vc_unpack16(code, rp_g + (size_t)sbi * sby, g, base_i, wg, eff_g, h_g);
                vc_unpack16(code, rp_u + (size_t)sbi * sby, g, base_i, wu, eff_u, h_u);
                const int xoff = sbi * 256 + base_i;
                const int doff = sbi * 4 + g;
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    if (m < ntok) {
                        const sycl::vec<int8_t, 16> xv = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(sxq_ptr + m * VC_K + xoff);
                        const float xd = sxd_ptr[m * VC_NG + doff];
                        int32_t sq_g = 0, sq_u = 0, sx = 0;
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            const int32_t xk = (int32_t)xv[k];
                            sq_g += (int32_t)wg[k] * xk;
                            sq_u += (int32_t)wu[k] * xk;
                            sx += xk;
                        }
                        acc_g[m] += (eff_g * xd) * ((float)sq_g - h_g * (float)sx);
                        acc_u[m] += (eff_u * xd) * ((float)sq_u - h_u * (float)sx);
                    }
                }
            }

            const float sq = (lane == 0) ? h2f_bits(ld16(vg.body + 2 * d)) : 1.0f;
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                if (m < ntok) {
                    const float gv = sg_sum(it, acc_g[m]);
                    const float uv = sg_sum(it, acc_u[m]);
                    if (lane == 0) {
                        const float s = sycl::fmin(gv / (1.0f + sycl::exp(-gv)), 10.0f);
                        const float uc = sycl::fmax(-10.0f, sycl::fmin(uv, 10.0f));
                        d_hmid[(size_t)vg.pair_idx[m] * VC_K + d] = (half)((s * uc) / sq);
                    }
                }
            }
        });
    });
}

// GEMM2 (down projection) with pre-quantized Q8_1_G64 hmid cached in SLM (<= 12 KiB/WG) + native SG16.
template <int NT>
inline void k_vcache_gemm2_q8(const VramExpertGroup* d_grps, int ngrps, const int8_t* d_hq, const float* d_hd, float* d_ye) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> sXq(NT * VC_K, h);
        sycl::local_accessor<float, 1> sXd(NT * VC_NG, h);
        h.parallel_for(NR((size_t)ngrps * 96, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int gi = gid / 96, rb = gid % 96;
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 4, lane = tid & 15;
            const int d = rb * 16 + wave;
            const VramExpertGroup vg = d_grps[gi];
            const int ntok = vg.ntok;

            int8_t* sxq_ptr = &sXq[0];
            float* sxd_ptr = &sXd[0];
            const int total_v4 = (ntok * VC_K) >> 2;
            for (int idx = tid; idx < total_v4; idx += 256) {
                const int m = idx / (VC_K >> 2);
                const int off4 = (idx % (VC_K >> 2)) << 2;
                *reinterpret_cast<sycl::vec<int8_t, 4>*>(sxq_ptr + m * VC_K + off4) =
                    *reinterpret_cast<const sycl::vec<int8_t, 4>*>(d_hq + (size_t)vg.pair_idx[m] * VC_K + off4);
            }
            const int total_d = ntok * VC_NG;
            for (int idx = tid; idx < total_d; idx += 256) {
                const int m = idx / VC_NG, g = idx % VC_NG;
                sxd_ptr[idx] = d_hd[(size_t)vg.pair_idx[m] * VC_NG + g];
            }
            sycl::group_barrier(it.get_group());

            const int code = vg.dn;
            const int sby = vc_sb_bytes(code);
            const size_t row_b = (size_t)6 * sby;
            const uint8_t* rp = vg.body + vg.dn_off + (size_t)d * row_b;
            const int g = lane >> 2;
            const int base_i = (g << 6) + ((lane & 3) << 4);

            float acc[NT];
#pragma unroll
            for (int m = 0; m < NT; ++m) acc[m] = 0.0f;

#pragma unroll 1
            for (int sbi = 0; sbi < 6; ++sbi) {
                int8_t wv[16];
                float eff = 0.0f, hval = 0.0f;
                vc_unpack16(code, rp + (size_t)sbi * sby, g, base_i, wv, eff, hval);
                const int xoff = sbi * 256 + base_i;
                const int doff = sbi * 4 + g;
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    if (m < ntok) {
                        const sycl::vec<int8_t, 16> xv = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(sxq_ptr + m * VC_K + xoff);
                        const float xd = sxd_ptr[m * VC_NG + doff];
                        int32_t sq = 0, sx = 0;
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            const int32_t xk = (int32_t)xv[k];
                            sq += (int32_t)wv[k] * xk;
                            sx += xk;
                        }
                        acc[m] += (eff * xd) * ((float)sq - hval * (float)sx);
                    }
                }
            }

#pragma unroll
            for (int m = 0; m < NT; ++m) {
                if (m < ntok) {
                    const float yv = sg_sum(it, acc[m]);
                    if (lane == 0) d_ye[(size_t)vg.pair_idx[m] * VC_K + d] = yv;
                }
            }
        });
    });
}

// High-precision FP32/FP16 activation path (when --cpu-hi / q8_act == false): still deduplicates experts & fuses gate+up on SG16!
template <int NT>
inline void k_vcache_gemm1_fp(const VramExpertGroup* d_grps, int ngrps, const float* z_dev, half* d_hmid) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sX(NT * 256, h);  // 1 superblock (256 floats/tok) <= 8 KiB SLM!
        h.parallel_for(NR((size_t)ngrps * 96, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int gi = gid / 96, rb = gid % 96;
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 4, lane = tid & 15;
            const int d = rb * 16 + wave;
            const VramExpertGroup vg = d_grps[gi];
            const int ntok = vg.ntok;
            const int code = vg.gu;
            const int sby = vc_sb_bytes(code);
            const size_t row_b = (size_t)6 * sby;
            const uint8_t* wgu = vg.body + 3072;
            const uint8_t* rp_g = wgu + (size_t)d * row_b;
            const uint8_t* rp_u = wgu + (size_t)(VC_K + d) * row_b;
            const int g = lane >> 2;
            const int base_i = (g << 6) + ((lane & 3) << 4);
            float* sx_ptr = &sX[0];

            float acc_g[NT], acc_u[NT];
#pragma unroll
            for (int m = 0; m < NT; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

#pragma unroll 1
            for (int sbi = 0; sbi < 6; ++sbi) {
                for (int m = 0; m < ntok; ++m) sx_ptr[m * 256 + tid] = z_dev[(size_t)vg.tok[m] * VC_K + sbi * 256 + tid];
                sycl::group_barrier(it.get_group());
                int8_t wg[16], wu[16];
                float eff_g = 0.0f, h_g = 0.0f, eff_u = 0.0f, h_u = 0.0f;
                vc_unpack16(code, rp_g + (size_t)sbi * sby, g, base_i, wg, eff_g, h_g);
                vc_unpack16(code, rp_u + (size_t)sbi * sby, g, base_i, wu, eff_u, h_u);
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    if (m < ntok) {
                        const float* xp = sx_ptr + m * 256 + base_i;
                        float sq_g = 0.0f, sq_u = 0.0f, sx = 0.0f;
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            const float xk = xp[k];
                            sq_g += (float)wg[k] * xk;
                            sq_u += (float)wu[k] * xk;
                            sx += xk;
                        }
                        acc_g[m] += eff_g * (sq_g - h_g * sx);
                        acc_u[m] += eff_u * (sq_u - h_u * sx);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            const float sq = (lane == 0) ? h2f_bits(ld16(vg.body + 2 * d)) : 1.0f;
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                if (m < ntok) {
                    const float gv = sg_sum(it, acc_g[m]);
                    const float uv = sg_sum(it, acc_u[m]);
                    if (lane == 0) {
                        const float s = sycl::fmin(gv / (1.0f + sycl::exp(-gv)), 10.0f);
                        const float uc = sycl::fmax(-10.0f, sycl::fmin(uv, 10.0f));
                        d_hmid[(size_t)vg.pair_idx[m] * VC_K + d] = (half)((s * uc) / sq);
                    }
                }
            }
        });
    });
}

template <int NT>
inline void k_vcache_gemm2_fp(const VramExpertGroup* d_grps, int ngrps, const half* d_hmid, float* d_ye) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sX(NT * 256, h);
        h.parallel_for(NR((size_t)ngrps * 96, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int gi = gid / 96, rb = gid % 96;
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 4, lane = tid & 15;
            const int d = rb * 16 + wave;
            const VramExpertGroup vg = d_grps[gi];
            const int ntok = vg.ntok;
            const int code = vg.dn;
            const int sby = vc_sb_bytes(code);
            const size_t row_b = (size_t)6 * sby;
            const uint8_t* rp = vg.body + vg.dn_off + (size_t)d * row_b;
            const int g = lane >> 2;
            const int base_i = (g << 6) + ((lane & 3) << 4);
            float* sx_ptr = &sX[0];

            float acc[NT];
#pragma unroll
            for (int m = 0; m < NT; ++m) acc[m] = 0.0f;

#pragma unroll 1
            for (int sbi = 0; sbi < 6; ++sbi) {
                for (int m = 0; m < ntok; ++m) sx_ptr[m * 256 + tid] = (float)d_hmid[(size_t)vg.pair_idx[m] * VC_K + sbi * 256 + tid];
                sycl::group_barrier(it.get_group());
                int8_t wv[16];
                float eff = 0.0f, hval = 0.0f;
                vc_unpack16(code, rp + (size_t)sbi * sby, g, base_i, wv, eff, hval);
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    if (m < ntok) {
                        const float* xp = sx_ptr + m * 256 + base_i;
                        float sq = 0.0f, sx = 0.0f;
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            const float xk = xp[k];
                            sq += (float)wv[k] * xk;
                            sx += xk;
                        }
                        acc[m] += eff * (sq - hval * sx);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

#pragma unroll
            for (int m = 0; m < NT; ++m) {
                if (m < ntok) {
                    const float yv = sg_sum(it, acc[m]);
                    if (lane == 0) d_ye[(size_t)vg.pair_idx[m] * VC_K + d] = yv;
                }
            }
        });
    });
}

}  // namespace vcache_detail

class SyVramExpertCache {
public:
    struct Meta {
        uint64_t file_off = 0;
        uint32_t dn_off = 0;
        uint32_t body_bytes = 0;
        uint8_t gu = 0, dn = 0;
        uint16_t rank = 0;
    };

    struct Slab {
        uint8_t* d = nullptr;
        size_t bytes = 0;
        int nslots = 0;
    };

    struct Slot {
        uint8_t* d_ptr = nullptr;
        int layer = -1;
        int expert = -1;
    };

    static constexpr int MAX_T = 16;
    static constexpr int MAX_PAIRS = MAX_T * vcache_detail::VC_NTOP;

    bool active() const { return !slots_.empty(); }
    bool adaptive() const { return active() && adaptive_; }
    int num_slots() const { return (int)slots_.size(); }
    size_t device_bytes() const { return dev_bytes_; }
    const uint8_t* expert_ptr(int L, int e) const {
        if ((unsigned)L >= (unsigned)nlayers_ || (unsigned)e >= 512u) return nullptr;
        return d_ptr_[(size_t)L * 512 + e];
    }
    const Meta& expert_meta(int L, int e) const { return meta_[(size_t)L * 512 + e]; }
    int n_layers() const { return nlayers_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t promotions() const { return promotions_; }
    uint64_t unique_expert_launches() const { return unique_launches_; }
    uint64_t total_pair_launches() const { return total_pair_launches_; }
    void reset_stats() { hits_ = misses_ = promotions_ = unique_launches_ = total_pair_launches_ = 0; }
    void set_q8_act(bool on) { q8_act_ = on; }
    bool q8_act() const { return q8_act_; }

    ~SyVramExpertCache() { release(); }

    void release() {
        if (cq_) { cq_->wait(); delete cq_; cq_ = nullptr; }
        for (auto& s : slabs_) if (s.d) dfree(s.d);
        slabs_.clear();
        slots_.clear();
        if (d_grps_) { dfree(d_grps_); d_grps_ = nullptr; }
        if (h_grps_) { sycl::free(h_grps_, Q()); h_grps_ = nullptr; }
        if (d_pair_w_) { dfree(d_pair_w_); d_pair_w_ = nullptr; }
        if (h_pair_w_) { sycl::free(h_pair_w_, Q()); h_pair_w_ = nullptr; }
        if (d_tok_off_) { dfree(d_tok_off_); d_tok_off_ = nullptr; }
        if (h_tok_off_) { sycl::free(h_tok_off_, Q()); h_tok_off_ = nullptr; }
        if (d_zq_) { dfree(d_zq_); d_zq_ = nullptr; }
        if (d_zd_) { dfree(d_zd_); d_zd_ = nullptr; }
        if (d_hq_) { dfree(d_hq_); d_hq_ = nullptr; }
        if (d_hd_) { dfree(d_hd_); d_hd_ = nullptr; }
        if (d_hmid_) { dfree(d_hmid_); d_hmid_ = nullptr; }
        if (d_ye_) { dfree(d_ye_); d_ye_ = nullptr; }
        if (h_stage_) { sycl::free(h_stage_, Q()); h_stage_ = nullptr; }
        dev_bytes_ = 0;
    }

    // Initialize multi-slab VRAM expert cache.
    bool init(const std::string& strata_path, size_t vram_budget_bytes, std::string& err, size_t slab_mb = 1400) {
        release();
        strata_path_ = strata_path;
        strata::artifact::StrataFileReader rd;
        if (!rd.open(strata_path, err)) return false;
        const auto& h = rd.header();
        const auto& dir = rd.expert_directory();
        if (dir.empty() || (dir.size() % 512) != 0) {
            err = "vram cache: invalid expert directory";
            return false;
        }
        nlayers_ = (int)(dir.size() / 512);
        meta_.assign(dir.size(), Meta{});
        d_ptr_.assign(dir.size(), nullptr);
        slot_of_.assign(dir.size(), -1);
        scores_.assign(dir.size(), 0.0f);
        rank_eid_.assign(dir.size(), 0);
        last_idx_.assign((size_t)nlayers_ * MAX_T * vcache_detail::VC_NTOP, -1);
        uint32_t max_body = 0;
        for (size_t i = 0; i < dir.size(); ++i) {
            const int L = (int)(i / 512), rk = (int)(i % 512);
            const auto& d = dir[i];
            Meta& m = meta_[(size_t)L * 512 + d.expert_id];
            m.file_off = h.sections[4].file_offset + d.body_offset;
            m.dn_off = d.dn_offset;
            m.body_bytes = d.body_bytes;
            m.gu = d.gu_code_id;
            m.dn = d.dn_code_id;
            m.rank = (uint16_t)rk;
            rank_eid_[i] = d.expert_id;
            max_body = std::max(max_body, d.body_bytes);
        }
        slot_bytes_ = ((size_t)max_body + 4095ull) & ~4095ull;
        if (vram_budget_bytes < slot_bytes_ * 41 || slot_bytes_ == 0) {
            return true;  // disabled cleanly
        }

        const size_t ws_bytes =
            (size_t)MAX_PAIRS * (sizeof(vcache_detail::VramExpertGroup) + sizeof(float) + vcache_detail::VC_K * (sizeof(half) + sizeof(float) + 1) + vcache_detail::VC_NG * sizeof(float)) +
            (size_t)MAX_T * (vcache_detail::VC_K + vcache_detail::VC_NG * sizeof(float)) + 8192;
        if (vram_budget_bytes <= ws_bytes + slot_bytes_) return true;
        size_t avail = vram_budget_bytes - ws_bytes;
        const size_t max_slots = std::min(dir.size(), avail / slot_bytes_);
        if (max_slots == 0) return true;

        const size_t slab_max = (slab_mb << 20);
        const int slots_per_slab_max = std::max(1, (int)(slab_max / slot_bytes_));
        size_t rem = max_slots;
        while (rem > 0) {
            const int ns = (int)std::min<size_t>(rem, (size_t)slots_per_slab_max);
            const size_t sb = (size_t)ns * slot_bytes_;
            uint8_t* ptr = sycl::malloc_device<uint8_t>(sb, Q());
            if (!ptr) break;
            dev_bytes_allocated() += sb;
            dev_bytes_ += sb;
            slabs_.push_back(Slab{ptr, sb, ns});
            for (int k = 0; k < ns; ++k) slots_.push_back(Slot{ptr + (size_t)k * slot_bytes_, -1, -1});
            rem -= (size_t)ns;
        }
        if (slots_.empty()) return true;

        // Allocate decode evaluator scratch (distinct-expert groups + pre-quant Q8 buffers)
        d_grps_ = dalloc<vcache_detail::VramExpertGroup>(MAX_PAIRS);
        h_grps_ = halloc<vcache_detail::VramExpertGroup>(MAX_PAIRS);
        d_pair_w_ = dalloc<float>(MAX_PAIRS);
        h_pair_w_ = halloc<float>(MAX_PAIRS);
        d_tok_off_ = dalloc<int32_t>(MAX_T + 1);
        h_tok_off_ = halloc<int32_t>(MAX_T + 1);
        d_zq_ = dalloc<int8_t>((size_t)MAX_T * vcache_detail::VC_K);
        d_zd_ = dalloc<float>((size_t)MAX_T * vcache_detail::VC_NG);
        d_hq_ = dalloc<int8_t>((size_t)MAX_PAIRS * vcache_detail::VC_K);
        d_hd_ = dalloc<float>((size_t)MAX_PAIRS * vcache_detail::VC_NG);
        d_hmid_ = dalloc<half>((size_t)MAX_PAIRS * vcache_detail::VC_K);
        d_ye_ = dalloc<float>((size_t)MAX_PAIRS * vcache_detail::VC_K);
        dev_bytes_ += ws_bytes;
        h_stage_ = sycl::aligned_alloc_host<uint8_t>(4096, slot_bytes_, Q());
        cq_ = new sycl::queue(Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order()});

        hy4::AlignedFile af;
        if (!af.open(strata_path, false)) {
            err = "vram cache: cannot open " + strata_path;
            return false;
        }
        const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        size_t si = 0;
        for (int rk = 0; rk < 512 && si < slots_.size(); ++rk) {
            for (int L = 0; L < nlayers_ && si < slots_.size(); ++L) {
                const int e = rank_eid_[(size_t)L * 512 + rk];
                const Meta& m = meta_[(size_t)L * 512 + e];
                if (af.read_at(m.file_off, h_stage_, m.body_bytes) < m.body_bytes) {
                    err = "vram cache: short read";
                    return false;
                }
                cq_->memcpy(slots_[si].d_ptr, h_stage_, m.body_bytes).wait();
                slots_[si].layer = L;
                slots_[si].expert = e;
                d_ptr_[(size_t)L * 512 + e] = slots_[si].d_ptr;
                slot_of_[(size_t)L * 512 + e] = (int)si;
                scores_[(size_t)L * 512 + e] = 2.0f;
                ++si;
            }
        }
        af.close();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
        adaptive_ = !(std::getenv("HY4S_VRAM_STATIC") && std::getenv("HY4S_VRAM_STATIC")[0] == '1');
        fprintf(stderr, "[vram-cache] %zu slots across %zu slabs (<= %zu MiB/slab) = %.2f GiB (%.1f experts/layer, %.1f s, adaptive=%d, sg16+dedup+preq8)\n",
                slots_.size(), slabs_.size(), slab_mb, dev_bytes_ / 1073741824.0, (double)slots_.size() / nlayers_, dt, (int)adaptive_);
        return true;
    }

    void build_vram_mask(std::vector<std::vector<bool>>& vram_mask) const {
        vram_mask.assign(nlayers_, std::vector<bool>(512, false));
        for (int L = 0; L < nlayers_; ++L)
            for (int e = 0; e < 512; ++e)
                if (d_ptr_[(size_t)L * 512 + e] != nullptr) vram_mask[L][e] = true;
    }

    // Step 1 (host-only, ~1 us): plan VRAM hits vs CPU misses, fill use_cpu[T*16], and deduplicate hits by distinct expert!
    // Call BEFORE ucx.submit() so CPU threads start immediately without waiting on SYCL kernel submission.
    int plan_decode(int L, int T, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu) {
        planned_pairs_ = 0;
        planned_groups_ = 0;
        planned_max_ntok_ = 0;
        planned_T_ = T;
        if (!active() || T <= 0 || T > MAX_T) {
            if (use_cpu) std::memset(use_cpu, 1, (size_t)T * vcache_detail::VC_NTOP);
            return 0;
        }
        int npairs = 0;
        h_tok_off_[0] = 0;
        int32_t* lbuf = &last_idx_[(size_t)L * MAX_T * vcache_detail::VC_NTOP];

        // Track distinct-expert groups in O(npairs) using a tiny stack map (expert -> group index)
        int16_t exp_grp[512];
        int16_t touched_e[MAX_PAIRS];
        int ntouched = 0;
        int ngroups = 0;
        int max_ntok = 0;
        for (int i = 0; i < 512; ++i) exp_grp[i] = -1;

        for (int t = 0; t < T; ++t) {
            int hit_k[vcache_detail::VC_NTOP], nh = 0;
            for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                lbuf[t * vcache_detail::VC_NTOP + k] = e;
                if ((unsigned)e < 512u) {
                    if (d_ptr_[(size_t)L * 512 + e] != nullptr) {
                        use_cpu[t * vcache_detail::VC_NTOP + k] = 0;
                        ++hits_;
                        int j = nh++;
                        while (j > 0 && idx_host[t * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                            hit_k[j] = hit_k[j - 1];
                            --j;
                        }
                        hit_k[j] = k;
                    } else {
                        use_cpu[t * vcache_detail::VC_NTOP + k] = 1;
                        ++misses_;
                    }
                } else {
                    use_cpu[t * vcache_detail::VC_NTOP + k] = 0;
                }
            }
            for (int a = 0; a < nh; ++a) {
                const int k = hit_k[a];
                const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                const int p = npairs++;
                h_pair_w_[p] = w_host[t * vcache_detail::VC_NTOP + k];
                int gi = exp_grp[e];
                if (gi < 0 || h_grps_[gi].ntok >= vcache_detail::VC_GRP_MAX_T) {
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    gi = ngroups++;
                    vcache_detail::VramExpertGroup& vg = h_grps_[gi];
                    vg.body = d_ptr_[(size_t)L * 512 + e];
                    vg.dn_off = m.dn_off;
                    vg.gu = m.gu;
                    vg.dn = m.dn;
                    vg.ntok = 0;
                    vg.pad0 = 0;
                    if (exp_grp[e] < 0) touched_e[ntouched++] = (int16_t)e;
                    exp_grp[e] = (int16_t)gi;
                }
                vcache_detail::VramExpertGroup& vg = h_grps_[gi];
                const int slot = vg.ntok++;
                vg.tok[slot] = t;
                vg.pair_idx[slot] = p;
                if (vg.ntok > max_ntok) max_ntok = vg.ntok;
            }
            h_tok_off_[t + 1] = npairs;
        }
        (void)touched_e;
        (void)ntouched;
        planned_pairs_ = npairs;
        planned_groups_ = ngroups;
        planned_max_ntok_ = max_ntok;
        unique_launches_ += (uint64_t)ngroups;
        total_pair_launches_ += (uint64_t)npairs;
        return npairs;
    }

    // Step 2 (async on Q()): enqueue GPU evaluation of deduplicated VRAM hits while CPU threads run in parallel.
    void launch_decode(const float* z_dev, float* zout_gpu_dev) {
        const int npairs = planned_pairs_;
        const int ngrps = planned_groups_;
        const int max_nt = planned_max_ntok_;
        const int T = planned_T_;
        if (npairs <= 0 || ngrps <= 0) return;

        Q().memcpy(d_grps_, h_grps_, (size_t)ngrps * sizeof(vcache_detail::VramExpertGroup));
        Q().memcpy(d_pair_w_, h_pair_w_, (size_t)npairs * sizeof(float));
        Q().memcpy(d_tok_off_, h_tok_off_, (size_t)(T + 1) * sizeof(int32_t));

        if (q8_act_) {
            // Pre-quantize z_dev[0..T-1] ONCE (eliminates 3,072 redundant quantizations per expert!)
            vcache_detail::k_quant_q8_1_g64<float>(z_dev, T, d_zq_, d_zd_);
            if (max_nt <= 1) vcache_detail::k_vcache_gemm1_q8<1>(d_grps_, ngrps, d_zq_, d_zd_, d_hmid_);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm1_q8<2>(d_grps_, ngrps, d_zq_, d_zd_, d_hmid_);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm1_q8<4>(d_grps_, ngrps, d_zq_, d_zd_, d_hmid_);
            else vcache_detail::k_vcache_gemm1_q8<8>(d_grps_, ngrps, d_zq_, d_zd_, d_hmid_);

            // Pre-quantize hmid[0..npairs-1] ONCE (eliminates 1,536 redundant quantizations per pair!)
            vcache_detail::k_quant_q8_1_g64<half>(d_hmid_, npairs, d_hq_, d_hd_);
            if (max_nt <= 1) vcache_detail::k_vcache_gemm2_q8<1>(d_grps_, ngrps, d_hq_, d_hd_, d_ye_);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm2_q8<2>(d_grps_, ngrps, d_hq_, d_hd_, d_ye_);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm2_q8<4>(d_grps_, ngrps, d_hq_, d_hd_, d_ye_);
            else vcache_detail::k_vcache_gemm2_q8<8>(d_grps_, ngrps, d_hq_, d_hd_, d_ye_);
        } else {
            if (max_nt <= 1) vcache_detail::k_vcache_gemm1_fp<1>(d_grps_, ngrps, z_dev, d_hmid_);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm1_fp<2>(d_grps_, ngrps, z_dev, d_hmid_);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm1_fp<4>(d_grps_, ngrps, z_dev, d_hmid_);
            else vcache_detail::k_vcache_gemm1_fp<8>(d_grps_, ngrps, z_dev, d_hmid_);

            if (max_nt <= 1) vcache_detail::k_vcache_gemm2_fp<1>(d_grps_, ngrps, d_hmid_, d_ye_);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm2_fp<2>(d_grps_, ngrps, d_hmid_, d_ye_);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm2_fp<4>(d_grps_, ngrps, d_hmid_, d_ye_);
            else vcache_detail::k_vcache_gemm2_fp<8>(d_grps_, ngrps, d_hmid_, d_ye_);
        }

        const float* ye = d_ye_;
        const float* pw = d_pair_w_;
        const int32_t* toff = d_tok_off_;
        Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = toff[t], p1 = toff[t + 1];
            for (int d = tid; d < vcache_detail::VC_K; d += 256) {
                float acc = 0.0f;
                for (int p = p0; p < p1; ++p) acc += ye[(size_t)p * vcache_detail::VC_K + d] * pw[p];
                zout_gpu_dev[(size_t)t * vcache_detail::VC_K + d] = acc;
            }
        });
    }

    int run_decode(int L, int T, const float* z_dev, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu, float* zout_gpu_dev) {
        const int n = plan_decode(L, T, idx_host, w_host, use_cpu);
        if (n > 0) launch_decode(z_dev, zout_gpu_dev);
        return n;
    }

    static void add_zout(float* dst, const float* src, size_t n) {
        Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const size_t i = it.get_global_id(0);
            if (i < n) dst[i] += src[i];
        });
    }

    // Committed-token LFU-decay update + periodic promotion whenever a 32-token boundary is crossed (no window clamping required!).
    // Optional `path`: for Tree-8 verify, path[i] gives the tree node index of the i-th committed token (nullptr = linear 0..n-1).
    template <class BodyFn>
    void commit_tokens(int n_committed, int next_pos, BodyFn&& get_host_body, int max_swaps = 41, const int* path = nullptr) {
        if (!active() || !adaptive_ || n_committed <= 0) return;
        const int n = std::min(n_committed, MAX_T);
        const int prev_pos = next_pos - n_committed;
        for (int i = 0; i < n; ++i) {
            const int t = path ? path[i] : i;
            for (float& s : scores_) s *= 0.965f;
            for (int L = 0; L < nlayers_; ++L) {
                const int32_t* row = &last_idx_[((size_t)L * MAX_T + t) * vcache_detail::VC_NTOP];
                for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                    const int e = row[k];
                    if ((unsigned)e < 512u) scores_[(size_t)L * 512 + e] += 1.0f;
                }
            }
        }
        if ((prev_pos >> 5) == (next_pos >> 5)) return;  // promote cleanly once per 32-token bucket crossing
        int swapped = 0;
        for (int L = 0; L < nlayers_ && swapped < max_swaps; ++L) {
            for (int rep = 0; rep < 2 && swapped < max_swaps; ++rep) {
                int best_e = -1;
                float best_sc = 1.8f;
                for (int e = 0; e < 512; ++e) {
                    if (d_ptr_[(size_t)L * 512 + e] != nullptr) continue;
                    const float sc = scores_[(size_t)L * 512 + e];
                    if (sc > best_sc) { best_sc = sc; best_e = e; }
                }
                if (best_e < 0) break;
                const uint8_t* hbody = get_host_body(L, best_e);
                if (!hbody) break;
                int victim_slot = -1;
                float worst_sc = best_sc / 1.25f;
                for (int ee = 0; ee < 512; ++ee) {
                    const int sl = slot_of_[(size_t)L * 512 + ee];
                    if (sl >= 0 && scores_[(size_t)L * 512 + ee] < worst_sc) {
                        worst_sc = scores_[(size_t)L * 512 + ee];
                        victim_slot = sl;
                    }
                }
                if (victim_slot < 0) break;
                const int old_e = slots_[victim_slot].expert;
                const Meta& m = meta_[(size_t)L * 512 + best_e];
                std::memcpy(h_stage_, hbody, m.body_bytes);
                cq_->memcpy(slots_[victim_slot].d_ptr, h_stage_, m.body_bytes).wait();
                d_ptr_[(size_t)L * 512 + old_e] = nullptr;
                slot_of_[(size_t)L * 512 + old_e] = -1;
                slots_[victim_slot].expert = best_e;
                d_ptr_[(size_t)L * 512 + best_e] = slots_[victim_slot].d_ptr;
                slot_of_[(size_t)L * 512 + best_e] = victim_slot;
                ++promotions_;
                ++swapped;
            }
        }
    }

    template <class BodyFn>
    void step_adapt(BodyFn&& get_host_body, int max_swaps = 16) {
        ++step_ctr_;
        commit_tokens(1, (int)step_ctr_, std::forward<BodyFn>(get_host_body), max_swaps);
    }

private:
    std::string strata_path_;
    int nlayers_ = 0;
    size_t slot_bytes_ = 0;
    size_t dev_bytes_ = 0;
    bool adaptive_ = true;
    bool q8_act_ = true;
    uint64_t step_ctr_ = 0;
    uint64_t hits_ = 0, misses_ = 0, promotions_ = 0;
    uint64_t unique_launches_ = 0, total_pair_launches_ = 0;
    std::vector<Meta> meta_;
    std::vector<uint16_t> rank_eid_;
    std::vector<Slab> slabs_;
    std::vector<Slot> slots_;
    std::vector<uint8_t*> d_ptr_;
    std::vector<int> slot_of_;
    std::vector<float> scores_;
    std::vector<int32_t> last_idx_;
    sycl::queue* cq_ = nullptr;
    uint8_t* h_stage_ = nullptr;
    vcache_detail::VramExpertGroup* d_grps_ = nullptr;
    vcache_detail::VramExpertGroup* h_grps_ = nullptr;
    float* d_pair_w_ = nullptr;
    float* h_pair_w_ = nullptr;
    int32_t* d_tok_off_ = nullptr;
    int32_t* h_tok_off_ = nullptr;
    int8_t* d_zq_ = nullptr;
    float* d_zd_ = nullptr;
    int8_t* d_hq_ = nullptr;
    float* d_hd_ = nullptr;
    half* d_hmid_ = nullptr;
    float* d_ye_ = nullptr;
    int planned_pairs_ = 0;
    int planned_groups_ = 0;
    int planned_max_ntok_ = 0;
    int planned_T_ = 0;
};

}  // namespace hy4s
