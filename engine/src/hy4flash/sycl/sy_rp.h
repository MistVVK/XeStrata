// sy_rp.h - "RP" repacked layout of S4_K / S5_K / S8_K trunk weights for Intel Arc GPUs (16-byte aligned payload loads), built on host at load time.
//   The .strata wire format has 134 / 166 / 262-byte super-blocks (2-byte aligned): RP splits every [N][K] weight into
//     headers  [N][nsb] x 8 B : D (fp16, 2 B), u[4] (4 B), 2 B zero          at offset 0
//     payload  [N][nsb] x PB  : the 128 / 160 / 256 payload bytes of block   at offset pay_off = roundup(N*nsb*8, 256)
//   Native SG16 (16 lanes/row, 16 rows/WG) gives 100% lane utilization on K=1536 (48=3*16 chunks), K=1024, K=3072, K=6144, K=8192, K=16384
//   and native SIMD16 execution on Xe2 Battlemage (B580 / Pro B60 / Pro B70) and Alchemist (A750 / A770).
//   k_gemv_multi fuses up to 3 projections sharing input width K across M = 1..8 (decode + DFly2 verify windows) for S4, S5, and S8.
#pragma once
#include <omp.h>

#include <cstdint>
#include <cstring>

#include "sy_common.h"

namespace hy4s {

inline bool rp_supported(int fmt, int K) { return (fmt == FMT_S4 || fmt == FMT_S5 || fmt == FMT_S8) && K % 256 == 0; }
inline int rp_pb(int fmt) { return fmt == FMT_S4 ? 128 : fmt == FMT_S5 ? 160 : 256; }  // S5: 128 B of low nibbles + 32 B of high bits
inline size_t rp_pay_off(int N, int nsb) { return ((size_t)N * nsb * 8 + 255) & ~(size_t)255; }
inline size_t rp_bytes(int fmt, int N, int K) { const int nsb = K / 256; return rp_pay_off(N, nsb) + (size_t)N * nsb * rp_pb(fmt); }

inline void rp_repack_host(int fmt, const uint8_t* src, uint8_t* dst, int N, int K) {
    const int nsb = K / 256, PB = rp_pb(fmt), SB = PB + 6;
    uint8_t* hdr = dst;
    uint8_t* pay = dst + rp_pay_off(N, nsb);
    std::memset(dst, 0, rp_pay_off(N, nsb));
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < N; ++r)
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* b = src + ((size_t)r * nsb + s) * SB;
            uint8_t* h = hdr + ((size_t)r * nsb + s) * 8;
            std::memcpy(h, b, 6);
            h[6] = h[7] = 0;
            std::memcpy(pay + ((size_t)r * nsb + s) * PB, b + 6, PB);
        }
}

// ---- device kernels on RP weights
// GEMV for M <= 8 tokens: one SG32 sub-group (32 lanes) per output row, 8 rows/WG (256 threads -> 2x WGs & 0 GRF spill on 32-Xe-core Arc Pro B70).
// Batch-invariant: every token m accumulates over identical chunk indices c0 = 0, 32, 64, ... -> bitwise identical M=1 vs M=2..8.
template <int FMT, int MT>
inline void k_gemv_rp(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;  // payload bytes per super-block
    constexpr int CPB = FMT == FMT_S8 ? 16 : 8;                          // 16-byte chunks (of nibbles / int8) per super-block
    constexpr int WPC = FMT == FMT_S8 ? 16 : 32;                         // weights per chunk
    const int nsb = K >> 8;
    const size_t pay_off = rp_pay_off(N, nsb);
    Q().parallel_for(NR((N + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const int row = (int)it.get_group(0) * 8 + wave;
        if (row >= N) return;
        const uint8_t* hdr = W + (size_t)row * nsb * 8;
        const uint8_t* pay = W + pay_off + (size_t)row * nsb * PB;
        const int nchunk = nsb * CPB;
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.f;
        for (int c0 = 0; c0 < nchunk; c0 += 32) {
            const int c = c0 + lane;
            if (c < nchunk) {
                const int sb = c / CPB, c_in = c % CPB, off = c_in * WPC;
                const uint8_t* pc = pay + (size_t)sb * PB;
                const sycl::uchar16 raw = *reinterpret_cast<const sycl::uchar16*>(pc + (size_t)c_in * 16);
                uint32_t hb = 0;
                if constexpr (FMT == FMT_S5) hb = *reinterpret_cast<const uint32_t*>(pc + 128 + (size_t)c_in * 4);
                const sycl::uchar8 hd = *reinterpret_cast<const sycl::uchar8*>(hdr + (size_t)sb * 8);
                const float D = h2f_bits((uint16_t)hd[0] | ((uint16_t)hd[1] << 8));
                const float eff = D * ((float)hd[2 + (off >> 6)] / 255.0f);
                const int k0 = sb * 256 + off;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const half* xp = X + (size_t)m * K + k0;
                        float sq = 0.f, sx = 0.f;
#pragma unroll
                        for (int v = 0; v < WPC / 8; ++v) {
                            const sycl::vec<half, 8> xv = *reinterpret_cast<const sycl::vec<half, 8>*>(xp + v * 8);
                            if constexpr (FMT == FMT_S4 || FMT == FMT_S5) {
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    const float x0 = (float)xv[2 * j], x1 = (float)xv[2 * j + 1];
                                    float q0 = (float)(raw[4 * v + j] & 15), q1 = (float)(raw[4 * v + j] >> 4);
                                    if constexpr (FMT == FMT_S5) {
                                        q0 += (float)(((hb >> (8 * v + 2 * j)) & 1u) << 4);
                                        q1 += (float)(((hb >> (8 * v + 2 * j + 1)) & 1u) << 4);
                                    }
                                    sq += q0 * x0 + q1 * x1;
                                    sx += x0 + x1;
                                }
                            } else {
#pragma unroll
                                for (int t = 0; t < 8; ++t) sq += (float)(int8_t)raw[8 * v + t] * (float)xv[t];
                            }
                        }
                        if constexpr (FMT == FMT_S8) acc[m] += eff * sq;
                        else acc[m] += eff * (sq - (FMT == FMT_S4 ? 7.5f : 15.5f) * sx);
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float v = sg_sum(it, acc[m]);
            if (lane == 0 && m < M) Y[(size_t)m * N + row] = v;
        }
    });
}

// ---- Multi-projection fused GEMV over up to 3 RP weights (S4, S5, or S8) of the SAME input width K across M = 1..8 tokens!
// Rows of the jobs are concatenated: q_a+kv_a+linear_gate, routed_down+shared_gate+shared_up, swiglu+shared_down, f2h+routed_up,
// and DFly2 drafter q_a+kv_a+linear_gate / gate+up / hidden_correction gate+up all fuse across BOTH decode (M=1) and verify (M=2..8).
// xmode 0: fp16 Xh[M][K]; xmode 1: fp32 Xf[M][K] rounded to fp16; xmode 2: half(silu(Xg)*Xu) of fp32 Xg[M][K], Xu[M][K].
// Arithmetic per row & token is bit-identical to k_gemv_rp<FMT, MT> and invariant across M = 1..8!
struct GemvJob {
    const uint8_t* W = nullptr;
    float* Y = nullptr;
    const half* Xh = nullptr;
    const float* Xf = nullptr;
    const float* Xg = nullptr;
    const float* Xu = nullptr;
    int N = 0, row0 = 0, xmode = 0, fmt = FMT_S4;
};
struct GemvMulti {
    GemvJob j[3];
    int n = 0, total = 0, K = 0;
};

template <int MT>
inline void k_gemv_multi_t(const GemvMulti& mj, int M) {
    const int nsb = mj.K >> 8;
    const int K = mj.K;
    Q().parallel_for(NR((mj.total + 7) / 8, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
        const int grow = (int)it.get_group(0) * 8 + wave;
        if (grow >= mj.total) return;
        int jj = 0;
        for (int q = 1; q < mj.n; ++q)
            if (grow >= mj.j[q].row0) jj = q;
        const GemvJob jb = mj.j[jj];
        const int row = grow - jb.row0;
        const size_t pay_off = rp_pay_off(jb.N, nsb);
        const uint8_t* hdr = jb.W + (size_t)row * nsb * 8;
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.f;

        auto load_x8 = [&](int m, int k_off, float (&xv)[8]) {
            if (jb.xmode == 0) {
                const sycl::vec<half, 8> vh = *reinterpret_cast<const sycl::vec<half, 8>*>(jb.Xh + (size_t)m * K + k_off);
#pragma unroll
                for (int t = 0; t < 8; ++t) xv[t] = (float)vh[t];
            } else if (jb.xmode == 1) {
                const float* p = jb.Xf + (size_t)m * K + k_off;
#pragma unroll
                for (int t = 0; t < 8; ++t) xv[t] = (float)(half)p[t];
            } else {
                const float* pg = jb.Xg + (size_t)m * K + k_off;
                const float* pu = jb.Xu + (size_t)m * K + k_off;
#pragma unroll
                for (int t = 0; t < 8; ++t) xv[t] = (float)(half)(siluf_(pg[t]) * pu[t]);
            }
        };

        if (jb.fmt == FMT_S8) {
            const uint8_t* pay = jb.W + pay_off + (size_t)row * nsb * 256;
            const int nchunk = nsb * 16;
            for (int c0 = 0; c0 < nchunk; c0 += 32) {
                const int c = c0 + lane;
                if (c < nchunk) {
                    const int sb = c >> 4, c_in = c & 15, off = c_in * 16;
                    const sycl::uchar16 raw = *reinterpret_cast<const sycl::uchar16*>(pay + (size_t)sb * 256 + (size_t)c_in * 16);
                    const sycl::uchar8 hd = *reinterpret_cast<const sycl::uchar8*>(hdr + (size_t)sb * 8);
                    const float D = h2f_bits((uint16_t)hd[0] | ((uint16_t)hd[1] << 8));
                    const float eff = D * ((float)hd[2 + (off >> 6)] / 255.0f);
                    const int k0 = sb * 256 + off;
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        if (m < M) {
                            float sq = 0.f;
#pragma unroll
                            for (int v = 0; v < 2; ++v) {
                                float xv[8];
                                load_x8(m, k0 + v * 8, xv);
#pragma unroll
                                for (int t = 0; t < 8; ++t) sq += (float)(int8_t)raw[8 * v + t] * xv[t];
                            }
                            acc[m] += eff * sq;
                        }
                    }
                }
            }
        } else {
            const bool is_s5 = (jb.fmt == FMT_S5);
            const int PB = is_s5 ? 160 : 128;
            const uint8_t* pay = jb.W + pay_off + (size_t)row * nsb * PB;
            const int nchunk = nsb * 8;
            const float hval = is_s5 ? 15.5f : 7.5f;
            for (int c0 = 0; c0 < nchunk; c0 += 32) {
                const int c = c0 + lane;
                if (c < nchunk) {
                    const int sb = c >> 3, c_in = c & 7, off = c_in * 32;
                    const uint8_t* pc = pay + (size_t)sb * PB;
                    const sycl::uchar16 raw = *reinterpret_cast<const sycl::uchar16*>(pc + (size_t)c_in * 16);
                    const uint32_t hb = is_s5 ? *reinterpret_cast<const uint32_t*>(pc + 128 + (size_t)c_in * 4) : 0u;
                    const sycl::uchar8 hd = *reinterpret_cast<const sycl::uchar8*>(hdr + (size_t)sb * 8);
                    const float D = h2f_bits((uint16_t)hd[0] | ((uint16_t)hd[1] << 8));
                    const float eff = D * ((float)hd[2 + (off >> 6)] / 255.0f);
                    const int k0 = sb * 256 + off;
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        if (m < M) {
                            float sq = 0.f, sx = 0.f;
#pragma unroll
                            for (int v = 0; v < 4; ++v) {
                                float xv[8];
                                load_x8(m, k0 + v * 8, xv);
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    const float x0 = xv[2 * j], x1 = xv[2 * j + 1];
                                    float q0 = (float)(raw[4 * v + j] & 15), q1 = (float)(raw[4 * v + j] >> 4);
                                    if (is_s5) {
                                        q0 += (float)(((hb >> (8 * v + 2 * j)) & 1u) << 4);
                                        q1 += (float)(((hb >> (8 * v + 2 * j + 1)) & 1u) << 4);
                                    }
                                    sq += q0 * x0 + q1 * x1;
                                    sx += x0 + x1;
                                }
                            }
                            acc[m] += eff * (sq - hval * sx);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float v = sg_sum(it, acc[m]);
            if (lane == 0 && m < M) jb.Y[(size_t)m * jb.N + row] = v;
        }
    });
}

inline void k_gemv_multi(const GemvMulti& mj, int M = 1) {
    if (M <= 1) k_gemv_multi_t<1>(mj, 1);
    else if (M == 2) k_gemv_multi_t<2>(mj, 2);
    else if (M <= 4) k_gemv_multi_t<4>(mj, M);
    else k_gemv_multi_t<8>(mj, M);
}

inline bool gemv_rp(int fmt, const void* W, int N, int K, const half* X, float* Y, int M) {
    const uint8_t* w = (const uint8_t*)W;
    if (M > 8) return false;
    if (fmt == FMT_S4) {
        if (M == 1) k_gemv_rp<FMT_S4, 1>(w, N, K, X, Y, M);
        else if (M == 2) k_gemv_rp<FMT_S4, 2>(w, N, K, X, Y, M);
        else if (M <= 4) k_gemv_rp<FMT_S4, 4>(w, N, K, X, Y, M);
        else k_gemv_rp<FMT_S4, 8>(w, N, K, X, Y, M);
        return true;
    }
    if (fmt == FMT_S5) {
        if (M == 1) k_gemv_rp<FMT_S5, 1>(w, N, K, X, Y, M);
        else if (M == 2) k_gemv_rp<FMT_S5, 2>(w, N, K, X, Y, M);
        else if (M <= 4) k_gemv_rp<FMT_S5, 4>(w, N, K, X, Y, M);
        else k_gemv_rp<FMT_S5, 8>(w, N, K, X, Y, M);
        return true;
    }
    if (fmt == FMT_S8) {
        if (M == 1) k_gemv_rp<FMT_S8, 1>(w, N, K, X, Y, M);
        else if (M == 2) k_gemv_rp<FMT_S8, 2>(w, N, K, X, Y, M);
        else if (M <= 4) k_gemv_rp<FMT_S8, 4>(w, N, K, X, Y, M);
        else k_gemv_rp<FMT_S8, 8>(w, N, K, X, Y, M);
        return true;
    }
    return false;
}

}  // namespace hy4s
