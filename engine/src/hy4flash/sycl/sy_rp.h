// sy_rp.h - "RP" repacked layout of S4_K / S8_K trunk weights for the Arc GPU (16-byte aligned payload loads), made on the host while loading.
//   The .strata wire format has 134 / 262-byte super-blocks (2-byte aligned): the GPU would need byte loads. RP splits every [N][K] weight into
//     headers  [N][nsb] x 8 B : D (fp16, 2 B), u[4] (4 B), 2 B zero          at offset 0
//     payload  [N][nsb] x PB  : the 128 / 256 payload bytes of the block     at offset pay_off = roundup(N*nsb*8, 256)       (PB = 128 for S4, 256 for S8)
//   so a lane can load one 16-byte chunk of payload and one 8-byte header per super-block. Dequant arithmetic is unchanged.
#pragma once
#include <omp.h>

#include <cstdint>
#include <cstring>

#include "sy_common.h"

namespace hy4s {

inline bool rp_supported(int fmt, int K) { return (fmt == FMT_S4 || fmt == FMT_S8) && K % 256 == 0; }
inline int rp_pb(int fmt) { return fmt == FMT_S4 ? 128 : 256; }
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
// GEMV for M <= 8 tokens: one sub-group (32 lanes) per output row, lane = 16-byte payload chunk.   Y[m][n] = sum_k W[n][k] X[m][k]
template <int FMT, int MT>
inline void k_gemv_rp(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int PB = FMT == FMT_S4 ? 128 : 256;
    constexpr int CPB = PB / 16;  // 16-byte chunks per super-block
    constexpr int WPC = FMT == FMT_S4 ? 32 : 16;  // weights per chunk
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
        for (int m = 0; m < MT; ++m) acc[m] = 0.f;
        for (int c0 = 0; c0 < nchunk; c0 += 32) {
            const int c = c0 + lane;
            if (c < nchunk) {
                const int sb = c / CPB, off = (c % CPB) * WPC;  // first weight of the chunk inside the super-block
                const sycl::uchar16 raw = *reinterpret_cast<const sycl::uchar16*>(pay + (size_t)c * 16);
                const sycl::uchar8 hd = *reinterpret_cast<const sycl::uchar8*>(hdr + (size_t)sb * 8);
                const float D = h2f_bits((uint16_t)hd[0] | ((uint16_t)hd[1] << 8));
                const float eff = D * ((float)hd[2 + (off >> 6)] / 255.0f);
                // no per-weight array (it spilled to private memory and ran ~8x slower than S8): dot = eff * (sum q*x - H * sum x), fully unrolled
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const half* xp = X + (size_t)m * K + (size_t)sb * 256 + off;
                        float sq = 0.f, sx = 0.f;
#pragma unroll
                        for (int v = 0; v < WPC / 8; ++v) {
                            const sycl::vec<half, 8> xv = *reinterpret_cast<const sycl::vec<half, 8>*>(xp + v * 8);
                            if constexpr (FMT == FMT_S4) {
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    const float x0 = (float)xv[2 * j], x1 = (float)xv[2 * j + 1];
                                    sq += (float)(raw[4 * v + j] & 15) * x0 + (float)(raw[4 * v + j] >> 4) * x1;
                                    sx += x0 + x1;
                                }
                            } else {
#pragma unroll
                                for (int t = 0; t < 8; ++t) sq += (float)(int8_t)raw[8 * v + t] * (float)xv[t];
                            }
                        }
                        acc[m] += eff * (sq - (FMT == FMT_S4 ? 7.5f : 0.0f) * sx);
                    }
                }
            }
        }
        for (int m = 0; m < MT; ++m) {
            const float v = sg_sum(it, acc[m]);
            if (lane == 0 && m < M) Y[(size_t)m * N + row] = v;
        }
    });
}

inline bool gemv_rp(int fmt, const void* W, int N, int K, const half* X, float* Y, int M) {
    const uint8_t* w = (const uint8_t*)W;
    if (M > 8) return false;
    if (fmt == FMT_S4) { if (M == 1) k_gemv_rp<FMT_S4, 1>(w, N, K, X, Y, M); else k_gemv_rp<FMT_S4, 8>(w, N, K, X, Y, M); return true; }
    if (fmt == FMT_S8) { if (M == 1) k_gemv_rp<FMT_S8, 1>(w, N, K, X, Y, M); else k_gemv_rp<FMT_S8, 8>(w, N, K, X, Y, M); return true; }
    return false;
}

}  // namespace hy4s
