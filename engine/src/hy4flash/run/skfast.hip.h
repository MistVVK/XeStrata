// skfast.hip.h - phase 2 kernels: bandwidth-oriented decode GEMV (M = 1..16) with packed fp16 dot (v_dot2_f32_f16) on gfx11
//
// y[m][n] = sum_k W[n][k] x[m][k]; W in .strata S_K wire format. One wave handles RW rows; the K loop runs in chunks of KC = 16384/MT halves so that
// the x tile of MT tokens (<= 32 KB) sits in LDS and is shared by the 8 waves of a block. Per 8-weight unit a lane decodes q (integer), builds
// half2 patterns (1024+q), subtracts the magic and the half-integer offset H in packed fp16 (exact), and uses fdot2 with fp32 accumulation.
// Includes skgemm.hip.h (reference kernels are kept as linear_h_ref).
#pragma once
#include "skgemm.hip.h"

namespace hy4 {
namespace skd {

__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) { return (uint32_t)ld16(p) | ((uint32_t)ld16(p + 2) << 16); }

__device__ __forceinline__ float dot2f(uint32_t a, uint32_t b, float c) {
    return h2_dot_f32(a, b, c);
}

// the raw payload words of the 8 weights 8u..8u+7 of a super-block (S2..S8 only): a/b = the loaded bytes, decoded by q_from_raw (split so a caller can issue its weight loads early)
template <int FMT>
__device__ __forceinline__ void load_q_raw(const uint8_t* pl, int u, uint32_t& a, uint32_t& b) {
    if (FMT == FMT_S2) { a = ld16(pl + 2 * u); b = 0; }
    else if (FMT == FMT_S3) { a = ld16(pl + 2 * u); b = pl[64 + u]; }
    else if (FMT == FMT_S4) { a = ld32u(pl + 4 * u); b = 0; }
    else if (FMT == FMT_S5) { a = ld32u(pl + 4 * u); b = pl[128 + u]; }
    else if (FMT == FMT_S6) { a = ld32u(pl + 4 * u); b = ld16(pl + 128 + 2 * u); }
    else { a = ld32u(pl + 8 * u); b = ld32u(pl + 8 * u + 4); }  // S8
}

// integer q of the 8 weights from the raw words (S8: byte ^ 0x80 = q + 128)
template <int FMT>
__device__ __forceinline__ void q_from_raw(uint32_t a, uint32_t b, uint32_t (&q)[8]) {
    if (FMT == FMT_S2) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = (a >> (2 * k)) & 3u;
    } else if (FMT == FMT_S3) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((a >> (2 * k)) & 3u) | (((b >> k) & 1u) << 2);
    } else if (FMT == FMT_S4) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = (a >> (4 * k)) & 15u;
    } else if (FMT == FMT_S5) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((a >> (4 * k)) & 15u) | (((b >> k) & 1u) << 4);
    } else if (FMT == FMT_S6) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((a >> (4 * k)) & 15u) | (((b >> (2 * k)) & 3u) << 4);
    } else {  // S8
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            q[k] = ((a >> (8 * k)) & 0xFFu) ^ 0x80u;
            q[4 + k] = ((b >> (8 * k)) & 0xFFu) ^ 0x80u;
        }
    }
}

// integer q of the 8 weights 8u..8u+7 of a super-block payload (S8: byte ^ 0x80 = q + 128)
template <int FMT>
__device__ __forceinline__ void decode_q8(const uint8_t* pl, int u, uint32_t (&q)[8]) {
    uint32_t a, b;
    load_q_raw<FMT>(pl, u, a, b);
    q_from_raw<FMT>(a, b, q);
}

// half2 weights (q-H pairs) from integer q (S2..S8)
template <int FMT>
__device__ __forceinline__ void w_from_q(const uint32_t (&q)[8], uint32_t (&wp)[4]) {
    const uint32_t mag = h2_splat(FMT == FMT_S8 ? 1152.0f : 1024.0f);
    const uint32_t hh = h2_splat(sk_half<FMT>());
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t pat = 0x64006400u | q[2 * i] | (q[2 * i + 1] << 16);
        uint32_t w = h2_sub(pat, mag);
        if (FMT != FMT_S8) w = h2_sub(w, hh);
        wp[i] = w;
    }
}

// half2 weights (q-H pairs) for the 8 weights of this lane
template <int FMT>
__device__ __forceinline__ void decode_w8(const uint8_t* pl, int u, uint32_t (&wp)[4]) {
    if constexpr (FMT == FMT_S1_25) {
        // weights 8u..8u+7 = codebook groups 2u, 2u+1 (one 10-bit field); values from the LDS codebook (s125_lut_init + barrier first)
        const int G = u >> 2, r = u & 3;
        const uint8_t* p = pl + G * 5 + ((10 * r) >> 3);
        const uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
        const uint32_t bits = (v >> ((10 * r) & 7)) & 0x3FFu;
        const uint2* lut = s125_lut();
        const uint2 a = lut[bits & 31u], b = lut[bits >> 5];
        wp[0] = a.x; wp[1] = a.y; wp[2] = b.x; wp[3] = b.y;
        return;
    }
    uint32_t q[8];
    if constexpr (FMT == FMT_S1_5) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = 8 * u + k;
            q[k] = (uint32_t)(s15_trit(pl[i / 5], i % 5) + 1);  // 0..2, magic 1025 below
        }
        const uint32_t mag = h2_splat(1025.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t pat = 0x64006400u | q[2 * i] | (q[2 * i + 1] << 16);
            wp[i] = h2_sub(pat, mag);
        }
        return;
    }
    decode_q8<FMT>(pl, u, q);
    const uint32_t mag = h2_splat(FMT == FMT_S8 ? 1152.0f : 1024.0f);
    const uint32_t hh = h2_splat(sk_half<FMT>());
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t pat = 0x64006400u | q[2 * i] | (q[2 * i + 1] << 16);
        uint32_t w = h2_sub(pat, mag);
        if (FMT != FMT_S8) w = h2_sub(w, hh);
        wp[i] = w;
    }
}

template <int FMT, int MT, int RW, int NW = 8>
__device__ __forceinline__ void gemv_body(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M, int bidx) {
#if HY4_CUDA
    // Tight LDS budget for small MT (8 KB at MT=1 vs 32 KB -> 4x higher SM occupancy on 128-188 SM Ada/Blackwell GPUs!)
    constexpr int KC = (MT <= 2) ? 4096 : (MT <= 4) ? 2048 : (MT <= 8) ? 1024 : 512;
#else
    // For MT >= 5, cap LDS at 16 KB (higher CU occupancy) and read xs per m instead of holding uint4 xv[MT] in VGPRs (eliminates 2x VGPR-pressure cliff at M=6..16).
    constexpr int KC = MT <= 4 ? (16384 / MT) : (MT <= 8 ? 1024 : 512);
#endif
    static_assert(KC >= 256 && (KC % 256) == 0, "KC must be a multiple of 256");
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int NTHR = NW * 32;
    __shared__ uint4 xs[MT * KC / 8];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (bidx * NW + wave) * RW;
    const int nsb = K >> 8;
    float acc[RW][MT];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
    const uint8_t* rp[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) rp[r] = W + (size_t)(row0 + r < N ? row0 + r : N - 1) * nsb * SB;
    if constexpr (FMT == FMT_S1_25) s125_lut_init();  // published by the barrier after the first x tile

    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kc = (K - k0) < KC ? (K - k0) : KC;
        const int n8 = kc >> 3;
        for (int i = threadIdx.x; i < MT * n8; i += NTHR) {
            const int mm = i / n8, c = i - mm * n8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (mm < M) v = *(const uint4*)(X + (size_t)mm * K + k0 + c * 8);
            xs[mm * (KC / 8) + c] = v;
        }
        __syncthreads();
        const int sb0 = k0 >> 8, nsbc = kc >> 8;
        if constexpr (MT <= 4) {
#pragma unroll 2
            for (int s = 0; s < nsbc; ++s) {
                uint4 xv[MT];
#pragma unroll
                for (int m = 0; m < MT; ++m) xv[m] = xs[m * (KC / 8) + s * 32 + lane];
#pragma unroll
                for (int r = 0; r < RW; ++r) {
                    const uint8_t* blk = rp[r] + (size_t)(sb0 + s) * SB;
                    const float eff = sk_eff<FMT>(blk, lane >> 3);
                    uint32_t wp[4];
                    decode_w8<FMT>(blk + 6, lane, wp);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        float p = dot2f(wp[0], xv[m].x, 0.f);
                        p = dot2f(wp[1], xv[m].y, p);
                        p = dot2f(wp[2], xv[m].z, p);
                        p = dot2f(wp[3], xv[m].w, p);
                        acc[r][m] += eff * p;
                    }
                }
            }
        } else {
            for (int s = 0; s < nsbc; ++s) {
#pragma unroll
                for (int r = 0; r < RW; ++r) {
                    const uint8_t* blk = rp[r] + (size_t)(sb0 + s) * SB;
                    const float eff = sk_eff<FMT>(blk, lane >> 3);
                    uint32_t wp[4];
                    decode_w8<FMT>(blk + 6, lane, wp);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const uint4 xv = xs[m * (KC / 8) + s * 32 + lane];
                        float p = dot2f(wp[0], xv.x, 0.f);
                        p = dot2f(wp[1], xv.y, p);
                        p = dot2f(wp[2], xv.z, p);
                        p = dot2f(wp[3], xv.w, p);
                        acc[r][m] += eff * p;
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[r][m];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0 && m < M && row0 + r < N) Y[(size_t)m * N + row0 + r] = v;
        }
}

template <int FMT, int MT, int RW, int NW = 8>
__global__ void __launch_bounds__(NW * 32) k_gemv(const uint8_t* __restrict__ W, int N, int K, const __half* __restrict__ X, float* __restrict__ Y, int M) {
    gemv_body<FMT, MT, RW, NW>(W, N, K, X, Y, M, blockIdx.x);
}

// several independent GEMVs on the same activations in ONE launch (q_a + kv_a + linear_gate, shared + latent, ...): jobs live in a device array
struct LinJob {
    const uint8_t* W;
    float* Y;
    int N;
    int blk0;  // first block of this job in the combined grid
};

template <int FMT, int MT, int RW, int NW = 8>
__global__ void __launch_bounds__(NW * 32) k_gemv_multi(const LinJob* __restrict__ jobs, int njobs, int K, const __half* __restrict__ X, int M) {
    int j = 0;
    while (j + 1 < njobs && (int)blockIdx.x >= jobs[j + 1].blk0) ++j;
    const LinJob jb = jobs[j];
    gemv_body<FMT, MT, RW, NW>(jb.W, jb.N, K, X, jb.Y, M, (int)blockIdx.x - jb.blk0);
}

template <int FMT, int MT>
inline void launch_gemv(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
#if HY4_CUDA
    if constexpr (MT <= 4) {
        // 4 rows/block (128 threads, 4 warps, 8-16 KB LDS): 4x more threadblocks to saturate 128-188 SM Ada/Blackwell GPUs
        constexpr int RW = 1, NW = 4;
        dim3 grid((N + NW * RW - 1) / (NW * RW)), block(NW * 32);
        hipLaunchKernelGGL((k_gemv<FMT, MT, RW, NW>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
        return;
    }
    constexpr int RW = 2, NW = 8;
    dim3 grid((N + NW * RW - 1) / (NW * RW)), block(NW * 32);
    hipLaunchKernelGGL((k_gemv<FMT, MT, RW, NW>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
#else
    constexpr int RW = MT <= 4 ? 2 : 1, NW = 8;
    dim3 grid((N + NW * RW - 1) / (NW * RW)), block(NW * 32);
    hipLaunchKernelGGL((k_gemv<FMT, MT, RW, NW>), grid, block, 0, s, (const uint8_t*)W, N, K, X, Y, M);
#endif
}

template <int FMT>
inline void gemv_by_m(hipStream_t s, const void* W, int N, int K, const __half* X, float* Y, int M) {
    if (M <= 1) launch_gemv<FMT, 1>(s, W, N, K, X, Y, M);
    else if (M <= 2) launch_gemv<FMT, 2>(s, W, N, K, X, Y, M);
    else if (M <= 4) launch_gemv<FMT, 4>(s, W, N, K, X, Y, M);
    else if (M <= 6) launch_gemv<FMT, 6>(s, W, N, K, X, Y, M);
    else if (M <= 8) launch_gemv<FMT, 8>(s, W, N, K, X, Y, M);
    else launch_gemv<FMT, 16>(s, W, N, K, X, Y, M);
}

}  // namespace skd

// ---- several GEMVs with the same activations and the same format in one launch. Build once at load time, call per token.
struct LinMulti {
    skd::LinJob* d_jobs = nullptr;
    skd::LinJob* d_jobs1 = nullptr;
    int njobs = 0, total_blocks = 0, total_blocks1 = 0, K = 0, fmt = 0;
};

#if HY4_CUDA
}  // namespace hy4
#include "sk_dp4a_cuda.hip.h"
namespace hy4 {
#endif

// production decode entry: S_K weights, 1 <= M <= 16 (K % 256 == 0, X 16-byte aligned)
inline void linear_h_gemv(hipStream_t s, int fmt, const void* W, int N, int K, const __half* X, float* Y, int M) {
#if HY4_CUDA
    if (skd::linear_h_gemv_dp4a(s, fmt, W, N, K, X, Y, M)) return;
#endif
    switch (fmt) {
        case FMT_S1_25: skd::gemv_by_m<FMT_S1_25>(s, W, N, K, X, Y, M); break;
        case FMT_S1_5: skd::gemv_by_m<FMT_S1_5>(s, W, N, K, X, Y, M); break;
        case FMT_S2: skd::gemv_by_m<FMT_S2>(s, W, N, K, X, Y, M); break;
        case FMT_S3: skd::gemv_by_m<FMT_S3>(s, W, N, K, X, Y, M); break;
        case FMT_S4: skd::gemv_by_m<FMT_S4>(s, W, N, K, X, Y, M); break;
        case FMT_S5: skd::gemv_by_m<FMT_S5>(s, W, N, K, X, Y, M); break;
        case FMT_S6: skd::gemv_by_m<FMT_S6>(s, W, N, K, X, Y, M); break;
        case FMT_S8: skd::gemv_by_m<FMT_S8>(s, W, N, K, X, Y, M); break;
        default: fprintf(stderr, "linear_h_gemv: unsupported fmt %d\n", fmt); abort();
    }
}

// ---- several GEMVs with the same activations and the same format in one launch. Build once at load time, call per token (1 <= M <= 16).
inline LinMulti linear_multi_build(int fmt, int K, int n, const void* const* W, float* const* Y, const int* N) {
    LinMulti d;
    d.fmt = fmt;
    d.K = K;
    d.njobs = n;
    skd::LinJob h2[16], h1[16];
    int blk2 = 0, blk1 = 0;
    for (int i = 0; i < n && i < 16; ++i) {
        h2[i].W = h1[i].W = (const uint8_t*)W[i];
        h2[i].Y = h1[i].Y = Y[i];
        h2[i].N = h1[i].N = N[i];
        h2[i].blk0 = blk2;
        h1[i].blk0 = blk1;
#if HY4_CUDA
        blk2 += (N[i] + 3) / 4;   // CUDA: RW = 1, NW = 4 (4 rows/block)
        blk1 += (N[i] + 3) / 4;
#else
        blk2 += (N[i] + 15) / 16;  // HIP M <= 4: RW = 2, NW = 8 (16 rows/block)
        blk1 += (N[i] + 7) / 8;    // HIP M >= 5: RW = 1, NW = 8 (8 rows/block, low VGPR)
#endif
    }
    d.total_blocks = blk2;
    d.total_blocks1 = blk1;
    if (hipMalloc((void**)&d.d_jobs, sizeof(skd::LinJob) * n) != hipSuccess) abort();
    if (hipMemcpy(d.d_jobs, h2, sizeof(skd::LinJob) * n, hipMemcpyHostToDevice) != hipSuccess) abort();
    if (hipMalloc((void**)&d.d_jobs1, sizeof(skd::LinJob) * n) != hipSuccess) abort();
    if (hipMemcpy(d.d_jobs1, h1, sizeof(skd::LinJob) * n, hipMemcpyHostToDevice) != hipSuccess) abort();
    return d;
}
inline void linear_multi_free(LinMulti& d) {
    if (d.d_jobs) (void)hipFree(d.d_jobs);
    if (d.d_jobs1) (void)hipFree(d.d_jobs1);
    d.d_jobs = d.d_jobs1 = nullptr;
}

namespace skd {
template <int FMT>
inline void multi_by_m(hipStream_t s, const LinMulti& d, const __half* X, int M) {
#if HY4_CUDA
#define HY4_MULTI(MT) hipLaunchKernelGGL((k_gemv_multi<FMT, MT, 1, 4>), dim3(d.total_blocks), dim3(128), 0, s, d.d_jobs, d.njobs, d.K, X, M)
    if (M <= 1) { HY4_MULTI(1); }
    else if (M <= 2) { HY4_MULTI(2); }
    else if (M <= 4) { HY4_MULTI(4); }
    else if (M <= 6) { HY4_MULTI(6); }
    else if (M <= 8) { HY4_MULTI(8); }
    else { HY4_MULTI(16); }
#undef HY4_MULTI
#else
    if (M <= 1) hipLaunchKernelGGL((k_gemv_multi<FMT, 1, 2, 8>), dim3(d.total_blocks), dim3(256), 0, s, d.d_jobs, d.njobs, d.K, X, M);
    else if (M <= 2) hipLaunchKernelGGL((k_gemv_multi<FMT, 2, 2, 8>), dim3(d.total_blocks), dim3(256), 0, s, d.d_jobs, d.njobs, d.K, X, M);
    else if (M <= 4) hipLaunchKernelGGL((k_gemv_multi<FMT, 4, 2, 8>), dim3(d.total_blocks), dim3(256), 0, s, d.d_jobs, d.njobs, d.K, X, M);
    else if (M <= 6) hipLaunchKernelGGL((k_gemv_multi<FMT, 6, 1, 8>), dim3(d.total_blocks1), dim3(256), 0, s, d.d_jobs1, d.njobs, d.K, X, M);
    else if (M <= 8) hipLaunchKernelGGL((k_gemv_multi<FMT, 8, 1, 8>), dim3(d.total_blocks1), dim3(256), 0, s, d.d_jobs1, d.njobs, d.K, X, M);
    else hipLaunchKernelGGL((k_gemv_multi<FMT, 16, 1, 8>), dim3(d.total_blocks1), dim3(256), 0, s, d.d_jobs1, d.njobs, d.K, X, M);
#endif
}

// ---- raw sub-group loader (shared by skwmma.hip.h and sk_rdna2.hip.h):
// 64 weights of sub-group dg of a super-block fetched with 4-byte aligned dword loads + byte realignment,
// one stage ahead of their use so global-memory latency hides behind the compute loop.
template <int FMT>
struct RawSub {
    static constexpr int NLO = FMT == FMT_S1_25 ? 3 : (FMT == FMT_S2 || FMT == FMT_S3 || FMT == FMT_S1_5) ? 4 : (FMT == FMT_S8 ? 16 : 8);
    static constexpr int NHI = FMT == FMT_S3 ? 2 : FMT == FMT_S5 ? 2 : FMT == FMT_S6 ? 4 : 1;
    uint32_t lo[NLO];
    uint32_t hi[NHI];
    uint32_t hdr[2];
};

template <int N>
__device__ __forceinline__ void load_bytes(const uint8_t* p, uint32_t (&o)[N]) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t ofs = (uint32_t)(a & 3);
    const uint32_t* q = (const uint32_t*)(a - ofs);
    uint32_t d[N + 1];
#pragma unroll
    for (int i = 0; i < N; ++i) d[i] = q[i];
    d[N] = ofs ? q[N] : 0u;
#pragma unroll
    for (int i = 0; i < N; ++i) o[i] = byte_funnel(d[i + 1], d[i], ofs);
}

template <int FMT>
__device__ __forceinline__ void load_raw(const uint8_t* blk, int dg, RawSub<FMT>& r) {
    load_bytes<2>(blk, r.hdr);
    const uint8_t* pl = blk + 6;
    if constexpr (FMT == FMT_S1_25) { load_bytes<3>(pl + dg * 10, r.lo); r.hi[0] = pl[40 + (dg >> 1)]; }
    else if constexpr (FMT == FMT_S1_5) { load_bytes<4>(pl + (64 * dg) / 5, r.lo); r.hi[0] = 64u * dg; }
    else if constexpr (FMT == FMT_S2) load_bytes<4>(pl + dg * 16, r.lo);
    else if constexpr (FMT == FMT_S3) { load_bytes<4>(pl + dg * 16, r.lo); load_bytes<2>(pl + 64 + dg * 8, r.hi); }
    else if constexpr (FMT == FMT_S4) load_bytes<8>(pl + dg * 32, r.lo);
    else if constexpr (FMT == FMT_S5) { load_bytes<8>(pl + dg * 32, r.lo); load_bytes<2>(pl + 128 + dg * 8, r.hi); }
    else if constexpr (FMT == FMT_S6) { load_bytes<8>(pl + dg * 32, r.lo); load_bytes<4>(pl + 128 + dg * 16, r.hi); }
    else load_bytes<16>(pl + dg * 64, r.lo);
}

template <int FMT>
__device__ __forceinline__ float raw_eff(const RawSub<FMT>& r, int dg) {
    const float D = h2f_bits((uint16_t)(r.hdr[0] & 0xFFFFu));
    const uint32_t u = dg < 2 ? (r.hdr[0] >> (16 + 8 * dg)) & 0xFFu : (r.hdr[1] >> (8 * (dg - 2))) & 0xFFu;
    float eff = D * ((float)u * (1.0f / 255.0f));
    if constexpr (FMT == FMT_S1_25) eff *= 0.5f + (float)((r.hi[0] >> ((dg & 1) * 4)) & 15u) / 15.0f;
    return eff;
}

__device__ __forceinline__ uint32_t byte_at4(const uint32_t (&w)[4], int o) {
    const int d = o >> 2;
    const uint32_t v = d == 0 ? w[0] : d == 1 ? w[1] : d == 2 ? w[2] : w[3];
    return (v >> ((o & 3) * 8)) & 0xFFu;
}
}  // namespace skd

// Y_i[m][n] = sum_k W_i[n][k] X[m][k] for all jobs i, 1 <= M <= 16, one launch. Each Y_i is a separate float [M][N_i] buffer.
inline void linear_h_multi(hipStream_t s, const LinMulti& d, const __half* X, int M) {
#if HY4_CUDA
    if (skd::linear_h_multi_dp4a(s, d, X, M)) return;
#endif
    switch (d.fmt) {
        case FMT_S1_25: skd::multi_by_m<FMT_S1_25>(s, d, X, M); break;
        case FMT_S1_5: skd::multi_by_m<FMT_S1_5>(s, d, X, M); break;
        case FMT_S2: skd::multi_by_m<FMT_S2>(s, d, X, M); break;
        case FMT_S3: skd::multi_by_m<FMT_S3>(s, d, X, M); break;
        case FMT_S4: skd::multi_by_m<FMT_S4>(s, d, X, M); break;
        case FMT_S5: skd::multi_by_m<FMT_S5>(s, d, X, M); break;
        case FMT_S6: skd::multi_by_m<FMT_S6>(s, d, X, M); break;
        case FMT_S8: skd::multi_by_m<FMT_S8>(s, d, X, M); break;
        default: fprintf(stderr, "linear_h_multi: unsupported fmt %d\n", d.fmt); abort();
    }
}

}  // namespace hy4
