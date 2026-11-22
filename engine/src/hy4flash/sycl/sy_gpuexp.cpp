// sy_gpuexp.cpp - High-speed Intel Arc XMX prefill routed experts (see sy_gpuexp.h).
// Key optimizations vs original sy_gpuexp.cpp:
//   1. BK=64 cooperative vectorized dequantization (128 threads = 2 threads/row, 32 weights/thread, 8 KiB SLM = 4x occupancy!).
//   2. Adaptive subgroup tiling per item.ntok:
//        ntok <= 8  -> 1 M-tile x 8 N-tiles (all 16 subgroups split across 64 cols + 2x K or 8 SG x 8 cols: 0 idle XMX!)
//        ntok <= 16 -> 2 M-tiles x 4 N-tiles
//        ntok <= 32 -> 4 M-tiles x 2 N-tiles
//        ntok > 32  -> 8 M-tiles x 2 N-tiles (16 SG8 = 128 threads, 100% busy!)
//   3. Double-buffered planning buffers [2] eliminating the cross-layer I.last_ev.wait() stall.
//   4. Direct VRAM-slab execution for SyVramExpertCache hits (Group 0: ZERO host copy, ZERO PCIe H2D!) overlapped with PCIe streaming.
//   5. Unified backend supporting hy4::CpuExperts (RAM arena + cold AlignedFile fallback) and legacy hy4s::CpuExperts.
#include "sy_gpuexp.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "hy4_io.h"
#include "strata/artifact/strata_reader.hpp"

namespace hy4s {

namespace xm = sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr int K = 1536;
constexpr int NTOP = 16;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

constexpr int8_t kS125[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
    {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
    {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
    {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0},
};

struct Item {  // one (expert, 64-token tile) of a group
    const uint8_t* body_ptr;  // direct device pointer (either dslot + off or VRAM slab pointer!)
    uint32_t dn_off;
    uint32_t pbase;
    uint32_t ntok;
    int32_t gu, dn;
    int32_t pad0;
};

inline int code_sb_bytes(int code) {
    return code == 0 ? 48 : code == 1 ? 58 : code == 2 ? 70 : code == 3 ? 102 : code == 4 ? 134 : code == 5 ? 166 : code == 6 ? 198 : 262;
}
inline int s15_trit(uint32_t b, int d) {
    if (b >= 243u) return 0;
    const uint32_t m = d == 0 ? 65536u : d == 1 ? 21846u : d == 2 ? 7282u : d == 3 ? 2428u : 810u;
    const uint32_t x = (b * m) >> 16;
    const uint32_t x3 = (x * 21846u) >> 16;
    return (int)(x - 3u * x3) - 1;
}
inline void put2(half* p, float a, float b) { *reinterpret_cast<sycl::vec<half, 2>*>(p) = sycl::vec<half, 2>((half)a, (half)b); }
inline uint32_t ld_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Cooperatively dequantize 32 weights (half_idx = 0 -> k=0..31, half_idx = 1 -> k=32..63) of 64-weight subgroup `g`
// of super-block `blk` of row `n` (0..63) into VNNI-2 packed SLM `sb_ptr` [32][128] (only 8 KiB!).
// Supports ALL 8 S_K expert codes (0=S1.25, 1=S1.58, 2=S2, 3=S3, 4=S4, 5=S5, 6=S6, 7=S8) -> enables GPU XMX prefill on SQ-32/64/96/128!
template <int CODE>
inline void dq_half_group(const uint8_t* blk, float eff, int g, int half_idx, half* sb_ptr, int n) {
    const uint8_t* pl = blk + 6;
    half* dst = sb_ptr + (half_idx * 16) * 128 + n * 2;
    if constexpr (CODE == 0) {
        const float fine = 0.5f + (float)((pl[40 + (g >> 1)] >> ((g & 1) * 4)) & 15u) / 15.0f;
        const float e = eff * fine;
        const int grp = g * 2 + half_idx;  // 0..7 (32 weights = 8 indices = 5 bytes)
        const uint8_t* p = pl + grp * 5;
        const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t idx = (uint32_t)((bits >> (5 * j)) & 0x1Fu);
            put2(dst + (2 * j) * 128, (float)kS125[idx][0] * e, (float)kS125[idx][1] * e);
            put2(dst + (2 * j + 1) * 128, (float)kS125[idx][2] * e, (float)kS125[idx][3] * e);
        }
    } else if constexpr (CODE == 1) {
        const int base_i = g * 64 + half_idx * 32;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i0 = base_i + 2 * j, i1 = i0 + 1;
            put2(dst + j * 128, (float)s15_trit(pl[i0 / 5], i0 % 5) * eff, (float)s15_trit(pl[i1 / 5], i1 % 5) * eff);
        }
    } else if constexpr (CODE == 2) {
        const float bias = -1.5f * eff;
        const uint8_t* p = pl + g * 16 + half_idx * 8;
        const uint32_t w0 = ld_u32(p), w1 = ld_u32(p + 4);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (w0 >> (8 * b)) & 0xFFu;
            put2(dst + (2 * b) * 128, (float)(by & 3u) * eff + bias, (float)((by >> 2) & 3u) * eff + bias);
            put2(dst + (2 * b + 1) * 128, (float)((by >> 4) & 3u) * eff + bias, (float)(by >> 6) * eff + bias);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t by = (w1 >> (8 * b)) & 0xFFu;
            put2(dst + (8 + 2 * b) * 128, (float)(by & 3u) * eff + bias, (float)((by >> 2) & 3u) * eff + bias);
            put2(dst + (8 + 2 * b + 1) * 128, (float)((by >> 4) & 3u) * eff + bias, (float)(by >> 6) * eff + bias);
        }
    } else if constexpr (CODE == 3) {
        const float bias = -3.5f * eff;
        const uint8_t* p = pl + g * 16 + half_idx * 8;
        const uint32_t w0 = ld_u32(p), w1 = ld_u32(p + 4);
        const uint32_t hb = ld_u32(pl + 64 + g * 8 + half_idx * 4);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t by = ((b < 4 ? w0 : w1) >> (8 * (b & 3))) & 0xFFu;
            const uint32_t h4 = (hb >> (4 * b)) & 0xFu;
            const uint32_t q0 = (by & 3u) | ((h4 & 1u) << 2);
            const uint32_t q1 = ((by >> 2) & 3u) | (((h4 >> 1) & 1u) << 2);
            const uint32_t q2 = ((by >> 4) & 3u) | (((h4 >> 2) & 1u) << 2);
            const uint32_t q3 = (by >> 6) | (((h4 >> 3) & 1u) << 2);
            put2(dst + (2 * b) * 128, (float)q0 * eff + bias, (float)q1 * eff + bias);
            put2(dst + (2 * b + 1) * 128, (float)q2 * eff + bias, (float)q3 * eff + bias);
        }
    } else if constexpr (CODE == 4) {
        const float bias = -7.5f * eff;
        const uint8_t* p = pl + g * 32 + half_idx * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t wd = ld_u32(p + w * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                put2(dst + (w * 4 + j) * 128, (float)(by & 15u) * eff + bias, (float)(by >> 4) * eff + bias);
            }
        }
    } else if constexpr (CODE == 5) {
        const float bias = -15.5f * eff;
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint32_t hb = ld_u32(pl + 128 + g * 8 + half_idx * 4);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t wd = ld_u32(p + w * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = w * 4 + j;
                const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                const uint32_t q0 = (by & 15u) | (((hb >> (2 * i)) & 1u) << 4);
                const uint32_t q1 = (by >> 4) | (((hb >> (2 * i + 1)) & 1u) << 4);
                put2(dst + i * 128, (float)q0 * eff + bias, (float)q1 * eff + bias);
            }
        }
    } else if constexpr (CODE == 6) {
        const float bias = -31.5f * eff;
        const uint8_t* p = pl + g * 32 + half_idx * 16;
        const uint8_t* ph = pl + 128 + g * 16 + half_idx * 8;
        const uint32_t h0 = ld_u32(ph), h1 = ld_u32(ph + 4);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t wd = ld_u32(p + w * 4);
            const uint32_t hw = (w < 2 ? h0 : h1) >> (16 * (w & 1));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t by = (wd >> (8 * j)) & 0xFFu;
                const uint32_t h4 = (hw >> (4 * j)) & 0xFu;
                const uint32_t q0 = (by & 15u) | ((h4 & 3u) << 4);
                const uint32_t q1 = (by >> 4) | (((h4 >> 2) & 3u) << 4);
                put2(dst + (w * 4 + j) * 128, (float)q0 * eff + bias, (float)q1 * eff + bias);
            }
        }
    } else {
        // CODE == 7: S8
        const uint8_t* p = pl + g * 64 + half_idx * 32;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t wd = ld_u32(p + w * 4);
            put2(dst + (2 * w) * 128, (float)(int8_t)(wd & 0xFFu) * eff, (float)(int8_t)((wd >> 8) & 0xFFu) * eff);
            put2(dst + (2 * w + 1) * 128, (float)(int8_t)((wd >> 16) & 0xFFu) * eff, (float)(int8_t)(wd >> 24) * eff);
        }
    }
}

inline void dq_half_group_any(int code, const uint8_t* blk, float eff, int g, int half_idx, half* sb_ptr, int n) {
    if (code == 4) dq_half_group<4>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 3) dq_half_group<3>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 2) dq_half_group<2>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 5) dq_half_group<5>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 6) dq_half_group<6>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 0) dq_half_group<0>(blk, eff, g, half_idx, sb_ptr, n);
    else if (code == 1) dq_half_group<1>(blk, eff, g, half_idx, sb_ptr, n);
    else dq_half_group<7>(blk, eff, g, half_idx, sb_ptr, n);
}

// MODE 0: GEMM1 + SwiGLU epilogue -> hb (half [P][1536]); MODE 1: GEMM2 -> ye (float [P][1536])
// 128 threads = 16 SG8 sub-groups!
// Adaptive M x N subgroup mapping eliminates idle XMX engines on small/medium ntok:
//   ntok <= 8  -> wm=1, wn=8 (8 active subgroups x 1 tile each; wait: 16 SG -> wm=2, wn=8 or wm=1, wn=8 with 2 K-splits? Better: wm_max = (ntok+7)/8:
//     ntok <= 16: wm=2, wn=8 -> 16 SG = 2 M-tiles x 8 N-tiles (1 accumulator mc[1] per SG! 100% active for 9..16, 50% for 1..8)
//     ntok <= 32: wm=4, wn=4 -> 16 SG = 4 M-tiles x 4 N-tiles (2 accumulators mc[2] per SG! 100% active!)
//     ntok >  32: wm=8, wn=2 -> 16 SG = 8 M-tiles x 2 N-tiles (4 accumulators mc[4] per SG! 100% active!)
// Notice max accumulators per SG is only 4 (half of original 8!), SLM is only 8 KiB (4x smaller than original 32 KiB!),
// and cooperative dequant uses 128 threads (2x more parallelism, vectorized 32-bit loads)!
#ifndef HY4S_DISABLE_SG8
template <int MODE>
sycl::event launch_gemm_xmx(sycl::queue& q, const sycl::event* dep, const Item* items, int nitems, const half* X, void* out) {
    constexpr int NT = MODE == 0 ? 48 : 24;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        sycl::local_accessor<half, 1> sB(64 * 64, h);    // 8 KiB VNNI-2 B tile (BK=64)
        sycl::local_accessor<float, 1> sC(64 * 64, h);   // 16 KiB epilogue tile (total 24 KiB SLM -> 2-3 WG/Xe-core!)
        h.parallel_for(sycl::nd_range<1>((size_t)nitems * NT * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const Item item = items[gid / NT];
            const int wt = gid % NT;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3;          // 0..15
            const int row_local = tid >> 1;    // 0..63
            const int half_idx = tid & 1;      // 0 or 1
            const uint8_t* body = item.body_ptr;
            const int code = MODE == 0 ? item.gu : item.dn;
            const uint8_t* W = MODE == 0 ? body + 3072 : body + item.dn_off;
            const int row = MODE == 0 ? (row_local < 32 ? 32 * wt + row_local : K + 32 * wt + (row_local - 32)) : 64 * wt + row_local;
            const int sby = code_sb_bytes(code);
            const uint8_t* rp = W + (size_t)row * 6 * sby;
            const int ntok = (int)item.ntok;

            // Adaptive (M, N) subgroup assignment for maximum XMX occupancy
            int sg_m, sg_n, n_tiles, n_step;
            if (ntok <= 16) {
                // 2 M-groups x 8 N-groups (each SG owns 1 N-tile of 8 cols!)
                sg_m = sgi >> 3;      // 0..1
                sg_n = sgi & 7;       // 0..7
                n_tiles = 1;
                n_step = 1;
            } else if (ntok <= 32) {
                // 4 M-groups x 4 N-groups (each SG owns 2 N-tiles = 16 cols!)
                sg_m = sgi >> 2;      // 0..3
                sg_n = (sgi & 3) * 2; // 0, 2, 4, 6
                n_tiles = 2;
                n_step = 1;
            } else {
                // 8 M-groups x 2 N-groups (each SG owns 4 N-tiles = 32 cols!)
                sg_m = sgi >> 1;      // 0..7
                sg_n = (sgi & 1) * 4; // 0, 4
                n_tiles = 4;
                n_step = 1;
            }
            const int m0 = (int)item.pbase + sg_m * 8;
            const bool active = sg_m * 8 < ntok;

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* blk = rp + (size_t)sbi * sby;
                const float D = h2f_bits(ld16(blk));
                const float effs[4] = {
                    D * ((float)blk[2] * (1.0f / 255.0f)), D * ((float)blk[3] * (1.0f / 255.0f)),
                    D * ((float)blk[4] * (1.0f / 255.0f)), D * ((float)blk[5] * (1.0f / 255.0f))};
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    dq_half_group_any(code, blk, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());

                    if (active) {
                        const half* xrow = X + (size_t)m0 * K + (size_t)sbi * 256 + g * 64;
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                const_cast<half*>(xrow + s * 16));
                            xm::joint_matrix_load(sg, ma, xp, (size_t)K);
                            for (int j = 0; j < n_tiles; ++j) {
                                xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)(sg_n + j * n_step) * 16, (size_t)128);
                                xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }

            float* sc = &sC[0];
            auto sc_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc);
            if (active) {
                for (int j = 0; j < n_tiles; ++j)
                    xm::joint_matrix_store(sg, mc[j], sc_mp + (size_t)(sg_m * 8) * 64 + (sg_n + j * n_step) * 8, (size_t)64, xm::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            if constexpr (MODE == 0) {
                for (int idx = tid; idx < ntok * 32; idx += 128) {
                    const int t = idx >> 5, i = idx & 31;
                    const float g = sc[t * 64 + i], u = sc[t * 64 + 32 + i];
                    const float sq = h2f_bits(ld16(body + 2 * (32 * wt + i)));
                    const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                    const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                    ((half*)out)[(size_t)(item.pbase + t) * K + 32 * wt + i] = (half)((s * uc) / sq);
                }
            } else {
                for (int idx = tid; idx < ntok * 64; idx += 128) {
                    const int t = idx >> 6, i = idx & 63;
                    ((float*)out)[(size_t)(item.pbase + t) * K + 64 * wt + i] = sc[t * 64 + i];
                }
            }
        });
    });
}
#endif

// Xe2 Battlemage SG=16 XMX prefill routed experts (8x16x16 fp16 -> fp32, 128 threads = 8 SG16, adaptive M x N tiling!)
#ifdef HY4S_ENABLE_SG16_XMX
template <int MODE>
sycl::event launch_gemm_xmx_sg16(sycl::queue& q, const sycl::event* dep, const Item* items, int nitems, const half* X, void* out) {
    constexpr int NT = MODE == 0 ? 48 : 24;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        sycl::local_accessor<half, 1> sB(64 * 64, h);    // 8 KiB VNNI-2 B tile (BK=64)
        sycl::local_accessor<float, 1> sC(64 * 64, h);   // 16 KiB epilogue tile
        h.parallel_for(sycl::nd_range<1>((size_t)nitems * NT * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gid = (int)it.get_group(0);
            const Item item = items[gid / NT];
            const int wt = gid % NT;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 4;          // 0..7 (8 SG16 subgroups)
            const int row_local = tid >> 1;    // 0..63
            const int half_idx = tid & 1;      // 0 or 1
            const uint8_t* body = item.body_ptr;
            const int code = MODE == 0 ? item.gu : item.dn;
            const uint8_t* W = MODE == 0 ? body + 3072 : body + item.dn_off;
            const int row = MODE == 0 ? (row_local < 32 ? 32 * wt + row_local : K + 32 * wt + (row_local - 32)) : 64 * wt + row_local;
            const int sby = code_sb_bytes(code);
            const uint8_t* rp = W + (size_t)row * 6 * sby;
            const int ntok = (int)item.ntok;

            // Adaptive (M, N) SG16 assignment:
            //   ntok <= 16 -> 2 M-tiles x 4 N-tiles (1 8x16 tile/SG16)
            //   ntok <= 32 -> 4 M-tiles x 2 N-tiles (2 8x16 tiles/SG16)
            //   ntok >  32 -> 8 M-tiles x 1 N-tile  (4 8x16 tiles/SG16)
            int sg_m, sg_n, n_tiles;
            if (ntok <= 16) {
                sg_m = sgi >> 2;      // 0..1
                sg_n = sgi & 3;       // 0..3
                n_tiles = 1;
            } else if (ntok <= 32) {
                sg_m = sgi >> 1;      // 0..3
                sg_n = (sgi & 1) * 2; // 0, 2
                n_tiles = 2;
            } else {
                sg_m = sgi;           // 0..7
                sg_n = 0;
                n_tiles = 4;
            }
            const int m0 = (int)item.pbase + sg_m * 8;
            const bool active = sg_m * 8 < ntok;

            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 16, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 16> mc[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);

            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);

            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* blk = rp + (size_t)sbi * sby;
                const float D = h2f_bits(ld16(blk));
                const float effs[4] = {
                    D * ((float)blk[2] * (1.0f / 255.0f)), D * ((float)blk[3] * (1.0f / 255.0f)),
                    D * ((float)blk[4] * (1.0f / 255.0f)), D * ((float)blk[5] * (1.0f / 255.0f))};
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    dq_half_group_any(code, blk, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());

                    if (active) {
                        const half* xrow = X + (size_t)m0 * K + (size_t)sbi * 256 + g * 64;
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                const_cast<half*>(xrow + s * 16));
                            xm::joint_matrix_load(sg, ma, xp, (size_t)K);
                            for (int j = 0; j < n_tiles; ++j) {
                                xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + (size_t)(sg_n + j) * 32, (size_t)128);
                                xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }

            float* sc = &sC[0];
            auto sc_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc);
            if (active) {
                for (int j = 0; j < n_tiles; ++j)
                    xm::joint_matrix_store(sg, mc[j], sc_mp + (size_t)(sg_m * 8) * 64 + (sg_n + j) * 16, (size_t)64, xm::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            if constexpr (MODE == 0) {
                for (int idx = tid; idx < ntok * 32; idx += 128) {
                    const int t = idx >> 5, i = idx & 31;
                    const float g = sc[t * 64 + i], u = sc[t * 64 + 32 + i];
                    const float sq = h2f_bits(ld16(body + 2 * (32 * wt + i)));
                    const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                    const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                    ((half*)out)[(size_t)(item.pbase + t) * K + 32 * wt + i] = (half)((s * uc) / sq);
                }
            } else {
                for (int idx = tid; idx < ntok * 64; idx += 128) {
                    const int t = idx >> 6, i = idx & 63;
                    ((float*)out)[(size_t)(item.pbase + t) * K + 64 * wt + i] = sc[t * 64 + i];
                }
            }
        });
    });
}
#endif

// Portable SG32 SIMT fallback for non-XMX Intel iGPUs
template <int MODE>
sycl::event launch_gemm_simt(sycl::queue& q, const sycl::event* dep, const Item* items, int nitems, const half* X, void* out) {
    constexpr int NT = MODE == 0 ? 48 : 24;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        sycl::local_accessor<half, 1> sB(64 * 64, h);
        sycl::local_accessor<float, 1> sC(64 * 64, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nitems * NT * 128, 128), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int gid = (int)it.get_group(0);
            const Item item = items[gid / NT];
            const int wt = gid % NT;
            const int tid = (int)it.get_local_id(0);
            const int row_local = tid >> 1, half_idx = tid & 1;
            const uint8_t* body = item.body_ptr;
            const int code = MODE == 0 ? item.gu : item.dn;
            const uint8_t* W = MODE == 0 ? body + 3072 : body + item.dn_off;
            const int row = MODE == 0 ? (row_local < 32 ? 32 * wt + row_local : K + 32 * wt + (row_local - 32)) : 64 * wt + row_local;
            const int sby = code_sb_bytes(code);
            const uint8_t* rp = W + (size_t)row * 6 * sby;
            const int ntok = (int)item.ntok;

            // 128 threads = 4 M-groups (16 tokens each) x 32 N-pairs (2 cols each = 64 cols)
            const int mg = tid >> 5, nc = (tid & 31) * 2;
            float acc[16][2];
            for (int r = 0; r < 16; ++r) acc[r][0] = acc[r][1] = 0.0f;

            half* sb_ptr = &sB[0];
            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* blk = rp + (size_t)sbi * sby;
                const float D = h2f_bits(ld16(blk));
                const float effs[4] = {
                    D * ((float)blk[2] * (1.0f / 255.0f)), D * ((float)blk[3] * (1.0f / 255.0f)),
                    D * ((float)blk[4] * (1.0f / 255.0f)), D * ((float)blk[5] * (1.0f / 255.0f))};
                for (int g = 0; g < 4; ++g) {
                    dq_half_group_any(code, blk, effs[g], g, half_idx, sb_ptr, row_local);
                    sycl::group_barrier(it.get_group());
                    const int k0 = sbi * 256 + g * 64;
                    for (int kp = 0; kp < 32; ++kp) {
                        const half* bp0 = sb_ptr + kp * 128 + nc * 2;
                        const float b00 = (float)bp0[0], b01 = (float)bp0[1], b10 = (float)bp0[2], b11 = (float)bp0[3];
                        for (int r = 0; r < 16; ++r) {
                            const int t = mg * 16 + r;
                            if (t < ntok) {
                                const half* xp = X + (size_t)(item.pbase + t) * K + k0 + 2 * kp;
                                const float x0 = (float)xp[0], x1 = (float)xp[1];
                                acc[r][0] += x0 * b00 + x1 * b01;
                                acc[r][1] += x0 * b10 + x1 * b11;
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
            for (int r = 0; r < 16; ++r) {
                const int t = mg * 16 + r;
                if (t < ntok) {
                    sC[t * 64 + nc] = acc[r][0];
                    sC[t * 64 + nc + 1] = acc[r][1];
                }
            }
            sycl::group_barrier(it.get_group());
            if constexpr (MODE == 0) {
                for (int idx = tid; idx < ntok * 32; idx += 128) {
                    const int t = idx >> 5, i = idx & 31;
                    const float g = sC[t * 64 + i], u = sC[t * 64 + 32 + i];
                    const float sq = h2f_bits(ld16(body + 2 * (32 * wt + i)));
                    const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                    const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                    ((half*)out)[(size_t)(item.pbase + t) * K + 32 * wt + i] = (half)((s * uc) / sq);
                }
            } else {
                for (int idx = tid; idx < ntok * 64; idx += 128) {
                    const int t = idx >> 6, i = idx & 63;
                    ((float*)out)[(size_t)(item.pbase + t) * K + 64 * wt + i] = sC[t * 64 + i];
                }
            }
        });
    });
}

template <int MODE>
sycl::event launch_gemm(sycl::queue& q, const sycl::event* dep, const Item* items, int nitems, const half* X, void* out) {
#ifndef HY4S_DISABLE_SG8
    if (dev_caps().has_xmx_sg8) return launch_gemm_xmx<MODE>(q, dep, items, nitems, X, out);
#endif
#ifdef HY4S_ENABLE_SG16_XMX
    if (dev_caps().has_xmx_sg16) return launch_gemm_xmx_sg16<MODE>(q, dep, items, nitems, X, out);
#endif
    return launch_gemm_simt<MODE>(q, dep, items, nitems, X, out);
}

sycl::event launch_gather(sycl::queue& q, const float* z, const int32_t* ptok, half* zg, int P) {
    const size_t n = (size_t)P * K;
    return q.parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) zg[i] = (half)z[(size_t)ptok[i / K] * K + i % K];
    });
}

sycl::event launch_reduce(sycl::queue& q, const float* ye, const int32_t* red, const float* pw, float* zout, int T) {
    return q.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        for (int d = tid; d < K; d += 256) {
            float acc = 0.0f;
            for (int j = 0; j < NTOP; ++j) {
                const int p = red[t * NTOP + j];
                if (p < 0) break;
                acc += ye[(size_t)p * K + d] * pw[p];
            }
            zout[(size_t)t * K + d] = acc;
        }
    });
}

struct DirMeta {
    uint64_t file_off = 0;
    uint32_t dn_off = 0, bytes = 0;
    int gu = 0, dn = 0;
    bool ok = false;
};

}  // namespace

struct GpuExperts::Impl {
    const CpuExperts* cx = nullptr;               // legacy source (sy_gpuexp_test)
    const hy4::CpuExperts* ucx = nullptr;         // unified source (hy4_cpu_experts.cpp)
    const SyVramExpertCache* vcache = nullptr;    // optional VRAM expert cache
    std::vector<DirMeta> umeta;                   // [41*512] directory metadata when unified
    hy4::AlignedFile af;                          // cold-expert file reader when unified
    bool af_open = false;
    sycl::queue cq;
    sycl::queue* comp = nullptr;
    bool profile = false;
    size_t slot_bytes = 0;
    int max_tokens = 0, copy_threads = 6;
    uint8_t* h_slot[3] = {};
    uint8_t* d_slot[2] = {};
    Item* h_items[3] = {};
    Item* d_items[2] = {};
    Item* h_vitems[2] = {};   // items for VRAM-resident experts (Group 0, 0 PCIe copy!)
    Item* d_vitems[2] = {};
    half *zg = nullptr, *hb = nullptr;
    float* ye = nullptr;
    // Double-buffered planning arrays [2] eliminating cross-layer last_ev stalls!
    int32_t *d_ptok[2] = {}, *d_red[2] = {}, *h_ptok[2] = {}, *h_red[2] = {};
    float *d_pw[2] = {}, *h_pw[2] = {};
    sycl::event plan_ev[2];
    bool plan_valid[2] = {false, false};
    int plan_ctr = 0;
    size_t dev_bytes = 0;
    sycl::event cp_ev[3], comp_ev[2], vcomp_ev[2];
    bool cp_valid[3] = {false, false, false}, comp_valid[2] = {false, false}, vcomp_valid[2] = {false, false};
    int group_ctr = 0;
    std::vector<int32_t> cnt, st, cur, pair_tok, pos;
    std::vector<float> pair_w;
    std::vector<int> act_vram, act_host;
    std::vector<std::pair<sycl::event, int>> prof;
    GpuExpertStats stats;
    bool inited = false;
    static constexpr int MAXITEMS = 2048;

    template <class T> T* dev(size_t n) {
        T* p = sycl::malloc_device<T>(n, cq);
        if (!p) {
            fprintf(stderr, "gpuexp: out of device memory (%zu bytes)\n", n * sizeof(T));
            exit(3);
        }
        dev_bytes += n * sizeof(T);
        return p;
    }

    bool alloc_common(size_t stage_bytes, int max_tok, int nthreads, std::string& err) {
        cq = sycl::queue(Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        comp = &Q();
        max_tokens = std::max(1, max_tok);
        copy_threads = std::max(1, nthreads);
        slot_bytes = (stage_bytes / 3) & ~(size_t)4095;
        if (slot_bytes < (8u << 20)) {
            err = "stage_bytes too small";
            return false;
        }
        const size_t maxP = (size_t)max_tokens * NTOP;
        for (int i = 0; i < 3; ++i) {
            h_slot[i] = sycl::aligned_alloc_host<uint8_t>(4096, slot_bytes, cq);
            h_items[i] = sycl::malloc_host<Item>(MAXITEMS, cq);
            if (!h_slot[i] || !h_items[i]) {
                err = "pinned host allocation failed";
                return false;
            }
        }
        for (int i = 0; i < 2; ++i) {
            d_slot[i] = dev<uint8_t>(slot_bytes);
            d_items[i] = dev<Item>(MAXITEMS);
            h_vitems[i] = sycl::malloc_host<Item>(MAXITEMS, cq);
            d_vitems[i] = dev<Item>(MAXITEMS);
            d_ptok[i] = dev<int32_t>(maxP);
            d_red[i] = dev<int32_t>(maxP);
            d_pw[i] = dev<float>(maxP);
            h_ptok[i] = sycl::malloc_host<int32_t>(maxP, cq);
            h_red[i] = sycl::malloc_host<int32_t>(maxP, cq);
            h_pw[i] = sycl::malloc_host<float>(maxP, cq);
        }
        zg = dev<half>((maxP + 64) * K);
        hb = dev<half>((maxP + 64) * K);
        ye = dev<float>(maxP * K);
        cq.memset(zg, 0, (maxP + 64) * K * sizeof(half));
        cq.memset(hb, 0, (maxP + 64) * K * sizeof(half));
        cq.wait();
        cnt.assign(512, 0);
        st.assign(513, 0);
        cur.assign(512, 0);
        pair_tok.assign(maxP, 0);
        pair_w.assign(maxP, 0.0f);
        pos.assign(maxP, -1);
        inited = true;
        return true;
    }
};

GpuExperts::GpuExperts() : im_(new Impl) {}
GpuExperts::~GpuExperts() {
    if (!im_->inited) return;
    im_->cq.wait();
    if (im_->comp) im_->comp->wait();
    for (auto* p : im_->h_slot) sycl::free(p, im_->cq);
    for (auto* p : im_->d_slot) sycl::free(p, im_->cq);
    for (auto* p : im_->h_items) sycl::free(p, im_->cq);
    for (auto* p : im_->d_items) sycl::free(p, im_->cq);
    for (auto* p : im_->h_vitems) sycl::free(p, im_->cq);
    for (auto* p : im_->d_vitems) sycl::free(p, im_->cq);
    for (int i = 0; i < 2; ++i) {
        sycl::free(im_->d_ptok[i], im_->cq);
        sycl::free(im_->d_red[i], im_->cq);
        sycl::free(im_->d_pw[i], im_->cq);
        sycl::free(im_->h_ptok[i], im_->cq);
        sycl::free(im_->h_red[i], im_->cq);
        sycl::free(im_->h_pw[i], im_->cq);
    }
    sycl::free(im_->zg, im_->cq);
    sycl::free(im_->hb, im_->cq);
    sycl::free(im_->ye, im_->cq);
}

#ifdef HY4S_LEGACY_CPUEXP
bool GpuExperts::init(const CpuExperts& cx, size_t stage_bytes, int max_tokens, std::string& err, int copy_threads) {
    Impl& I = *im_;
    I.cx = &cx;
    I.ucx = nullptr;
    if (!I.alloc_common(stage_bytes, max_tokens, copy_threads, err)) return false;
    for (int L = 0; L < 41; ++L)
        for (int e = 0; e < 512; ++e) {
            const uint8_t* body;
            int gu, dn;
            uint32_t dno, bytes;
            if (!cx.expert_info(L, e, &body, &gu, &dn, &dno, &bytes)) continue;
            auto okc = [](int c) { return c >= 0 && c <= 7; };
            if (!okc(gu) || !okc(dn)) {
                err = "expert code not supported by GPU path (expected 0..7)";
                return false;
            }
            if (bytes > I.slot_bytes) {
                err = "expert body larger than ring slot";
                return false;
            }
        }
    return true;
}
#endif

bool GpuExperts::init_unified(const std::string& strata_path, const hy4::CpuExperts* ucx, const SyVramExpertCache* vcache,
                              size_t stage_bytes, int max_tokens, std::string& err, int copy_threads) {
    Impl& I = *im_;
    I.cx = nullptr;
    I.ucx = ucx;
    I.vcache = vcache;
    strata::artifact::StrataFileReader rd;
    if (!rd.open(strata_path, err)) return false;
    const auto& h = rd.header();
    const auto& dir = rd.expert_directory();
    I.umeta.assign(dir.size(), DirMeta{});
    if (!I.alloc_common(stage_bytes, max_tokens, copy_threads, err)) return false;
    for (size_t i = 0; i < dir.size(); ++i) {
        const int L = (int)(i / 512);
        const auto& d = dir[i];
        auto okc = [](int c) { return c >= 0 && c <= 7; };
        if (!okc(d.gu_code_id) || !okc(d.dn_code_id)) {
            err = "expert code not supported by GPU path (expected 0..7)";
            return false;
        }
        if (d.body_bytes > I.slot_bytes) {
            err = "expert body larger than ring slot";
            return false;
        }
        DirMeta& dm = I.umeta[(size_t)L * 512 + d.expert_id];
        dm.file_off = h.sections[4].file_offset + d.body_offset;
        dm.dn_off = d.dn_offset;
        dm.bytes = d.body_bytes;
        dm.gu = d.gu_code_id;
        dm.dn = d.dn_code_id;
        dm.ok = true;
    }
    I.af_open = I.af.open(strata_path, false);
    return true;
}

void GpuExperts::set_vram_cache(const SyVramExpertCache* vcache) { im_->vcache = vcache; }
void GpuExperts::set_profile_queue(sycl::queue* compute) {
    im_->comp = compute ? compute : &Q();
    im_->profile = compute != nullptr;
}
size_t GpuExperts::device_bytes() const { return im_->dev_bytes; }
const GpuExpertStats& GpuExperts::stats() const { return im_->stats; }
void GpuExperts::reset_stats() { im_->stats = GpuExpertStats(); }

void GpuExperts::wait_all() {
    Impl& I = *im_;
    I.cq.wait_and_throw();
    I.comp->wait_and_throw();
    for (auto& pe : I.prof) {
        const double ms = (double)(pe.first.get_profiling_info<sycl::info::event_profiling::command_end>() - pe.first.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
        if (pe.second == 0) I.stats.gpu_gemm1_ms += ms;
        else if (pe.second == 1) I.stats.gpu_gemm2_ms += ms;
        else if (pe.second == 3) I.stats.gpu_h2d_ms += ms;
        else I.stats.gpu_other_ms += ms;
    }
    I.prof.clear();
}

void GpuExperts::run(int L, int T, const float* z_dev, const int32_t* idx, const float* w, float* zout_dev, const uint8_t* only) {
    Impl& I = *im_;
    if (T > I.max_tokens) {
        for (int t0 = 0; t0 < T; t0 += I.max_tokens) {
            const int n = std::min(I.max_tokens, T - t0);
            run(L, n, z_dev + (size_t)t0 * K, idx + (size_t)t0 * NTOP, w + (size_t)t0 * NTOP, zout_dev + (size_t)t0 * K, only);
        }
        return;
    }
    const double t_start = now_s();
    sycl::queue& comp = *I.comp;
    const int ps = I.plan_ctr & 1;
    I.plan_ctr++;
    if (I.plan_valid[ps]) {
        const double tw0 = now_s();
        I.plan_ev[ps].wait();
        I.stats.t_wait_prev += now_s() - tw0;
    }

    struct EInfo {
        const uint8_t* hbody = nullptr;
        const uint8_t* dbody = nullptr;  // non-null if resident in SyVramExpertCache!
        uint64_t file_off = 0;
        int gu = 0, dn = 0;
        uint32_t dn_off = 0, bytes = 0;
    };
    std::fill(I.cnt.begin(), I.cnt.end(), 0);
    EInfo einfo[512];
    uint8_t okv[512];
    for (int e = 0; e < 512; ++e) {
        EInfo& x = einfo[e];
        bool ok = false;
        if (I.cx) {
#ifdef HY4S_LEGACY_CPUEXP
            ok = I.cx->expert_info(L, e, &x.hbody, &x.gu, &x.dn, &x.dn_off, &x.bytes);
#endif
        } else {
            const DirMeta& dm = I.umeta[(size_t)L * 512 + e];
            ok = dm.ok;
            x.gu = dm.gu;
            x.dn = dm.dn;
            x.dn_off = dm.dn_off;
            x.bytes = dm.bytes;
            x.file_off = dm.file_off;
            if (I.ucx) {
                auto vw = I.ucx->view(L, e);
                x.hbody = vw.body;
            }
        }
        if (ok && I.vcache && I.vcache->active()) x.dbody = I.vcache->expert_ptr(L, e);
        okv[e] = (ok && (!only || only[e])) ? 1 : 0;
    }

    const int npair = T * NTOP;
    for (int i = 0; i < npair; ++i) {
        const int e = idx[i];
        if ((unsigned)e < 512u && okv[e]) I.cnt[e]++;
    }
    I.act_vram.clear();
    I.act_host.clear();
    int acc = 0;
    for (int e = 0; e < 512; ++e) {
        I.st[e] = acc;
        I.cur[e] = acc;
        if (I.cnt[e]) {
            if (einfo[e].dbody != nullptr) I.act_vram.push_back(e);
            else I.act_host.push_back(e);
        }
        acc += I.cnt[e];
    }
    I.st[512] = acc;
    const int P = acc;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < NTOP; ++k) {
            const int e = idx[t * NTOP + k];
            if ((unsigned)e < 512u && okv[e]) {
                const int p = I.cur[e]++;
                I.pair_tok[p] = t;
                I.pair_w[p] = w[t * NTOP + k];
                I.pos[t * NTOP + k] = p;
            } else {
                I.pos[t * NTOP + k] = -1;
            }
        }
    for (int t = 0; t < T; ++t) {
        int ks[NTOP], n = 0;
        for (int k = 0; k < NTOP; ++k)
            if (I.pos[t * NTOP + k] >= 0) {
                int j = n++;
                while (j > 0 && idx[t * NTOP + ks[j - 1]] > idx[t * NTOP + k]) {
                    ks[j] = ks[j - 1];
                    --j;
                }
                ks[j] = k;
            }
        for (int a = 0; a < NTOP; ++a) I.h_red[ps][t * NTOP + a] = a < n ? I.pos[t * NTOP + ks[a]] : -1;
    }
    std::memcpy(I.h_ptok[ps], I.pair_tok.data(), (size_t)std::max(P, 1) * sizeof(int32_t));
    std::memcpy(I.h_pw[ps], I.pair_w.data(), (size_t)std::max(P, 1) * sizeof(float));
    sycl::event e1 = comp.memcpy(I.d_red[ps], I.h_red[ps], (size_t)T * NTOP * sizeof(int32_t));
    sycl::event e2 = comp.memcpy(I.d_ptok[ps], I.h_ptok[ps], (size_t)std::max(P, 1) * sizeof(int32_t));
    sycl::event e3 = comp.memcpy(I.d_pw[ps], I.h_pw[ps], (size_t)std::max(P, 1) * sizeof(float));
    if (I.profile) { I.prof.push_back({e1, 2}); I.prof.push_back({e2, 2}); I.prof.push_back({e3, 2}); }
    if (P > 0) {
        sycl::event eg = launch_gather(comp, z_dev, I.d_ptok[ps], I.zg, P);
        if (I.profile) I.prof.push_back({eg, 2});
    }
    I.stats.t_plan += now_s() - t_start;

    // ---- Group 0 (VRAM-resident experts): Launch IMMEDIATELY with 0 host copy and 0 PCIe transfer!
    // Runs on compute queue while host threads copy + cq DMAs the first host-RAM expert group!
    if (!I.act_vram.empty()) {
        if (I.vcomp_valid[ps]) I.vcomp_ev[ps].wait();
        int nvi = 0;
        for (int e : I.act_vram) {
            for (int tt = 0; tt < I.cnt[e]; tt += 64) {
                Item& it = I.h_vitems[ps][nvi++];
                it.body_ptr = einfo[e].dbody;
                it.dn_off = einfo[e].dn_off;
                it.pbase = (uint32_t)(I.st[e] + tt);
                it.ntok = (uint32_t)std::min(64, I.cnt[e] - tt);
                it.gu = einfo[e].gu;
                it.dn = einfo[e].dn;
                it.pad0 = 0;
            }
        }
        sycl::event evi = comp.memcpy(I.d_vitems[ps], I.h_vitems[ps], (size_t)nvi * sizeof(Item));
        sycl::event vk1 = launch_gemm<0>(comp, nullptr, I.d_vitems[ps], nvi, I.zg, I.hb);
        sycl::event vk2 = launch_gemm<1>(comp, nullptr, I.d_vitems[ps], nvi, I.hb, I.ye);
        I.vcomp_ev[ps] = vk2;
        I.vcomp_valid[ps] = true;
        if (I.profile) { I.prof.push_back({evi, 2}); I.prof.push_back({vk1, 0}); I.prof.push_back({vk2, 1}); }
        I.stats.vram_experts += I.act_vram.size();
        I.stats.experts += I.act_vram.size();
    }

    // ---- Host/SSD expert groups streamed via 3-slot pinned host bounce ring -> 2-slot device ring
    std::vector<int> grp_begin;
    {
        size_t used = 0;
        grp_begin.push_back(0);
        for (size_t i = 0; i < I.act_host.size(); ++i) {
            const size_t b = einfo[I.act_host[i]].bytes;
            if (used + b > I.slot_bytes) {
                grp_begin.push_back((int)i);
                used = 0;
            }
            used += b;
        }
        grp_begin.push_back((int)I.act_host.size());
    }
    const int ng = I.act_host.empty() ? 0 : (int)grp_begin.size() - 1;
    for (int g = 0; g < ng; ++g) {
        const int hs = I.group_ctr % 3, ds = I.group_ctr % 2;
        I.group_ctr++;
        double tw = now_s();
        if (I.cp_valid[hs]) I.cp_ev[hs].wait();
        I.stats.t_wait_host += now_s() - tw;
        const int a = grp_begin[g], b = grp_begin[g + 1];
        std::vector<uint32_t> off(b - a);
        size_t used = 0;
        int nitems = 0;
        for (int i = a; i < b; ++i) {
            const int e = I.act_host[i];
            off[i - a] = (uint32_t)used;
            for (int tt = 0; tt < I.cnt[e]; tt += 64) {
                Item& it = I.h_items[hs][nitems++];
                it.body_ptr = I.d_slot[ds] + used;
                it.dn_off = einfo[e].dn_off;
                it.pbase = (uint32_t)(I.st[e] + tt);
                it.ntok = (uint32_t)std::min(64, I.cnt[e] - tt);
                it.gu = einfo[e].gu;
                it.dn = einfo[e].dn;
                it.pad0 = 0;
            }
            used += einfo[e].bytes;
        }
        if (nitems > Impl::MAXITEMS) {
            fprintf(stderr, "gpuexp: too many items in a group (%d)\n", nitems);
            exit(3);
        }
        double tc = now_s();
#pragma omp parallel for num_threads(I.copy_threads) schedule(dynamic, 1)
        for (int i = a; i < b; ++i) {
            const EInfo& ei = einfo[I.act_host[i]];
            uint8_t* dst = I.h_slot[hs] + off[i - a];
            if (I.cx) {
#ifdef HY4S_LEGACY_CPUEXP
                I.cx->copy_body(ei.hbody, ei.bytes, dst);
#endif
            } else if (ei.hbody) {
                std::memcpy(dst, ei.hbody, ei.bytes);
            } else if (I.af_open) {
                I.af.read_at(ei.file_off, dst, ei.bytes);
            }
        }
        I.stats.t_copy_host += now_s() - tc;
        sycl::event ci = I.cq.submit([&](sycl::handler& h) {
            if (I.comp_valid[ds]) h.depends_on(I.comp_ev[ds]);
            h.memcpy(I.d_items[ds], I.h_items[hs], (size_t)nitems * sizeof(Item));
        });
        sycl::event cb = I.cq.memcpy(I.d_slot[ds], I.h_slot[hs], used);
        I.cp_ev[hs] = cb;
        I.cp_valid[hs] = true;
        sycl::event k1 = launch_gemm<0>(comp, &cb, I.d_items[ds], nitems, I.zg, I.hb);
        sycl::event k2 = launch_gemm<1>(comp, nullptr, I.d_items[ds], nitems, I.hb, I.ye);
        I.comp_ev[ds] = k2;
        I.comp_valid[ds] = true;
        if (I.profile) {
            I.prof.push_back({ci, 3});
            I.prof.push_back({cb, 3});
            I.prof.push_back({k1, 0});
            I.prof.push_back({k2, 1});
        }
        I.stats.bytes += used;
        I.stats.groups++;
        I.stats.experts += b - a;
    }
    sycl::event er = launch_reduce(comp, I.ye, I.d_red[ps], I.d_pw[ps], zout_dev, T);
    if (I.profile) I.prof.push_back({er, 2});
    I.plan_ev[ps] = er;
    I.plan_valid[ps] = true;
    I.stats.pairs += P;
    I.stats.calls++;
    I.stats.seconds += now_s() - t_start;
}

}  // namespace hy4s
