// skmoe.hip.h - LatentMoE expert evaluation for one layer (square [1536,1536] experts, S_K bodies from .strata section 4)
//
//   for every (token t, slot s) with expert e = topk_idx[t*16+s], weight w = topk_w[t*16+s]:
//     gu = Wgu[e] @ z[t]            (3072 rows, K = 1536; rows 0..1535 gate, 1536..3071 up)
//     h  = min(silu(gu[:1536]), 10) * clamp(gu[1536:], -10, 10)
//     h2 = h / sq_xd_dn[e]          (fp16[1536] at the start of the body, element-wise by output index)
//     y  = Wdn[e] @ h2              (1536 rows, K = 1536)
//     z_out[t] += w * y             (summed over the 16 slots in slot order: deterministic)
//
// moe_experts(): all device pointers, no host sync, one workspace (moe_ws_bytes(M)).
//   M <= moe_dec_max_m() (default 4, valid up to 16; decode / speculative verify): per active expert, GEMV-style kernels (packed fp16 dot2, x tile in LDS,
//                         per-expert token count picks the inner width so an expert with 1 token costs a decode step)
//   above it (prefill, verify windows): per active expert, WMMA grouped GEMMs with a 16/32/64/128-token tile chosen per expert
// moe_experts_ref(): the simple phase-1.5 path (any M), kept for testing.
#pragma once
#include "skwmma.hip.h"

namespace hy4 {

constexpr int MOE_E = 512, MOE_TOPK = 16, MOE_D = 1536;

struct ExpertDev {             // per expert id (0..511) of one MoE layer
    uint64_t body_off;         // byte offset of the body relative to sec4_base
    uint32_t dn_off;           // byte offset of the down weight inside the body (= 3072 + 18432 * sb_bytes(gu_code))
    uint8_t gu_code, dn_code;  // Fmt values 2,3,4,5,6,8
    uint16_t pad;
};
static_assert(sizeof(ExpertDev) == 16, "ExpertDev");

inline size_t moe_align(size_t x) { return (x + 255) & ~(size_t)255; }
// workspace layout: ints [cnt 512 | off 513 | fill 512 | active 512 | nact 1] | pairs[P] (int) | h2[P][1536] (half) | y[P][1536] (float),  P = M*16
inline size_t moe_ws_bytes(int M) {
    const size_t P = (size_t)M * MOE_TOPK;
    return moe_align(sizeof(int) * 2056) + moe_align(P * 4) + moe_align(P * MOE_D * 2) + moe_align(P * MOE_D * 4);
}

namespace skd {

// ---------------------------------------------------------------- routing (single block, counting sort by expert)
__global__ void __launch_bounds__(512) k_moe_route(const ExpertDev* __restrict__ ex, const int32_t* __restrict__ idx, int P, int* __restrict__ cnt, int* __restrict__ off, int* __restrict__ pairs,
                                                    int* __restrict__ active, int* __restrict__ nact) {
    __shared__ int c[512], sc[512], fl[512], fill[512];
    const int t = threadIdx.x;
    c[t] = 0;
    fill[t] = 0;
    __syncthreads();
    // experts flagged pad != 0 live on the CPU (hybrid placement): their pairs are not evaluated here
    for (int i = t; i < P; i += 512)
        if (!ex[idx[i]].pad) atomicAdd(&c[idx[i]], 1);
    __syncthreads();
    const int cv = c[t];
    const int f = cv > 0;
    sc[t] = cv;
    fl[t] = f;
    __syncthreads();
    for (int d = 1; d < 512; d <<= 1) {
        int a = 0, b = 0;
        if (t >= d) { a = sc[t - d]; b = fl[t - d]; }
        __syncthreads();
        sc[t] += a;
        fl[t] += b;
        __syncthreads();
    }
    const int incl = sc[t], inclf = fl[t];
    cnt[t] = cv;
    off[t] = incl - cv;
    if (t == 511) { off[512] = incl; *nact = inclf; }
    if (f) active[inclf - 1] = t;
    __syncthreads();
    sc[t] = incl - cv;
    __syncthreads();
    for (int i = t; i < P; i += 512) {
        const int e = idx[i];
        if (ex[e].pad) continue;
        const int pos = atomicAdd(&fill[e], 1);
        pairs[sc[e] + pos] = i;
    }
}

// ---------------------------------------------------------------- decode path (M <= 16)
// dot products of one weight row (K = 1536, FMT) with MT x-vectors held in LDS as uint4[MT][192]
template <int FMT, int MT>
__device__ __forceinline__ void row_dot(const uint8_t* rp, const uint4* xs, int lane, float (&acc)[MT]) {
    constexpr int SB = sk_sb_bytes(FMT);
    uint32_t wp[6][4];
    float eff[6];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = rp + s * SB;
        eff[s] = sk_eff<FMT>(blk, lane >> 3);
        decode_w8<FMT>(blk + 6, lane, wp[s]);
    }
#pragma unroll
    for (int s = 0; s < 6; ++s) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint4 xv = xs[m * 192 + s * 32 + lane];
            float p = dot2f(wp[s][0], xv.x, 0.f);
            p = dot2f(wp[s][1], xv.y, p);
            p = dot2f(wp[s][2], xv.z, p);
            p = dot2f(wp[s][3], xv.w, p);
            acc[m] += eff[s] * p;
        }
    }
}

template <int MT>
__device__ __forceinline__ void row_dot_sw(int code, const uint8_t* rp, const uint4* xs, int lane, float (&acc)[MT]) {
    switch (code) {
        case FMT_S1_25: row_dot<FMT_S1_25, MT>(rp, xs, lane, acc); break;
        case FMT_S1_5: row_dot<FMT_S1_5, MT>(rp, xs, lane, acc); break;
        case FMT_S2: row_dot<FMT_S2, MT>(rp, xs, lane, acc); break;
        case FMT_S3: row_dot<FMT_S3, MT>(rp, xs, lane, acc); break;
        case FMT_S4: row_dot<FMT_S4, MT>(rp, xs, lane, acc); break;
        case FMT_S5: row_dot<FMT_S5, MT>(rp, xs, lane, acc); break;
        case FMT_S6: row_dot<FMT_S6, MT>(rp, xs, lane, acc); break;
        case FMT_S8: row_dot<FMT_S8, MT>(rp, xs, lane, acc); break;
        default: break;
    }
}

__device__ __forceinline__ int code_row_bytes(int code, int K) { return (K >> 8) * sk_sb_bytes(code); }

// the per-token work of the decode kernels only covers the tokens this expert really has (n <= MTE <= MT): an expert that sees 1 token in a
// verify window of 8 costs the same as in decode. Per token the arithmetic is the same as with MTE = MT (bit-identical results).
template <int MTE>
__device__ __forceinline__ void gu_dec_rows(const ExpertDev& ed, const uint8_t* body, int r, const uint4* xs, int lane, int n, int base, __half* h2) {
    const int rb = code_row_bytes(ed.gu_code, MOE_D);
    const uint8_t* gu = body + 3072;
    float ag[MTE], au[MTE];
#pragma unroll
    for (int m = 0; m < MTE; ++m) { ag[m] = 0.f; au[m] = 0.f; }
    row_dot_sw<MTE>(ed.gu_code, gu + (size_t)r * rb, xs, lane, ag);
    row_dot_sw<MTE>(ed.gu_code, gu + (size_t)(MOE_D + r) * rb, xs, lane, au);
    const float sq = __half2float(((const __half*)body)[r]);
#pragma unroll
    for (int m = 0; m < MTE; ++m) {
        float g = ag[m], u = au[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) { g += __shfl_xor(g, o); u += __shfl_xor(u, o); }
        if (lane == 0 && m < n) {
            const float sg = g / (1.0f + expf(-g));
            const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
            h2[(size_t)(base + m) * MOE_D + r] = __float2half(hh / sq);
        }
    }
}

template <int MTE>
__device__ __forceinline__ void dn_dec_rows(const ExpertDev& ed, const uint8_t* dn, int r0, const uint4* xs, int lane, int n, int base, const int* pairs,
                                            const float* topk_w, float* y) {
    const int rb = code_row_bytes(ed.dn_code, MOE_D);
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = r0 + rr;
        float a[MTE];
#pragma unroll
        for (int m = 0; m < MTE; ++m) a[m] = 0.f;
        row_dot_sw<MTE>(ed.dn_code, dn + (size_t)r * rb, xs, lane, a);
#pragma unroll
        for (int m = 0; m < MTE; ++m) {
            float v = a[m];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
            if (lane == 0 && m < n) {
                const int pid = pairs[base + m];
                y[(size_t)pid * MOE_D + r] = v * topk_w[pid];
            }
        }
    }
}

template <int A, int B> struct cmin { static constexpr int v = A < B ? A : B; };

// stage 1: gate/up + SwiGLU + /sq_xd_dn. grid = (192, min(512,P)); each wave = one (gate row r, up row 1536+r) pair
template <int MT>
__global__ void __launch_bounds__(256) k_moe_gu_dec(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                     const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                     const int* __restrict__ active, const int* __restrict__ nact, __half* __restrict__ h2) {
    if ((int)blockIdx.y >= *nact) return;
    const int e = active[blockIdx.y];
    const int n = cnt[e], base = off[e];
    __shared__ uint4 xs[MT * 192];
    s125_lut_init();  // the S1.25 codebook (published by the barrier below)
    for (int i = threadIdx.x; i < MT * 192; i += 256) {
        const int m = i / 192, c = i - m * 192;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < n) v = *(const uint4*)(z + (size_t)(pairs[base + m] >> 4) * MOE_D + c * 8);
        xs[i] = v;
    }
    __syncthreads();
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
    if (n <= 1) gu_dec_rows<1>(ed, body, r, xs, lane, n, base, h2);
    else if (n <= 2) gu_dec_rows<cmin<2, MT>::v>(ed, body, r, xs, lane, n, base, h2);
    else if (n <= 4) gu_dec_rows<cmin<4, MT>::v>(ed, body, r, xs, lane, n, base, h2);
    else if (n <= 8) gu_dec_rows<cmin<8, MT>::v>(ed, body, r, xs, lane, n, base, h2);
    else gu_dec_rows<MT>(ed, body, r, xs, lane, n, base, h2);
}

// stage 2: down. grid = (96, min(512,P)); each wave = 2 output rows
template <int MT>
__global__ void __launch_bounds__(256) k_moe_dn_dec(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                     const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                     const int* __restrict__ active, const int* __restrict__ nact, const __half* __restrict__ h2,
                                                     float* __restrict__ y) {
    if ((int)blockIdx.y >= *nact) return;
    const int e = active[blockIdx.y];
    const int n = cnt[e], base = off[e];
    __shared__ uint4 xs[MT * 192];
    s125_lut_init();
    for (int i = threadIdx.x; i < MT * 192; i += 256) {
        const int m = i / 192, c = i - m * 192;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < n) v = *(const uint4*)(h2 + (size_t)(base + m) * MOE_D + c * 8);
        xs[i] = v;
    }
    __syncthreads();
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const ExpertDev ed = ex[e];
    const uint8_t* dn = sec4 + ed.body_off + ed.dn_off;
    const int r0 = (blockIdx.x * 8 + wave) * 2;
    if (n <= 1) dn_dec_rows<1>(ed, dn, r0, xs, lane, n, base, pairs, topk_w, y);
    else if (n <= 2) dn_dec_rows<cmin<2, MT>::v>(ed, dn, r0, xs, lane, n, base, pairs, topk_w, y);
    else if (n <= 4) dn_dec_rows<cmin<4, MT>::v>(ed, dn, r0, xs, lane, n, base, pairs, topk_w, y);
    else if (n <= 8) dn_dec_rows<cmin<8, MT>::v>(ed, dn, r0, xs, lane, n, base, pairs, topk_w, y);
    else dn_dec_rows<MT>(ed, dn, r0, xs, lane, n, base, pairs, topk_w, y);
}

#if HY4_MMA
// ---------------------------------------------------------------- WMMA grouped GEMMs (prefill and small verify windows)
// Token tile BM = WM*MTW*16 (16, 32, 64 or 128): the kernel variant handles the experts whose token count n satisfies nlo < n <= nhi, so every expert
// is processed with the smallest tile that covers it (a 128-token tile wastes 16x the matrix-core work on an expert that sees 8 tokens).
template <int FMT, int WM, int MTW>
__device__ __forceinline__ void gu_gemm_body(const uint8_t* body, int bx, int ty, int n, int base, const int* pairs, const __half* z, __half* h2, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BMT = WM * MTW * 16;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;  // wn in 0..1 for the active waves (gate tile + up tile of 16 output columns each)
    const bool active = wn < 2;
    const int drow = tid >> 2;
    const int wrow = drow < 32 ? bx * 32 + drow : bx * 32 + drow + (MOE_D - 32);
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;
    const half_t* xrow[MTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
        int ml = ty * BMT + wm * MTW * 16 + i * 16 + (lane & 15);
        ml = ml < n ? ml : n - 1;
        xrow[i] = (const half_t*)z + (size_t)(pairs[base + ml] >> 4) * MOE_D;
    }
    fv8 acc[MTW][2];
    acc_zero(acc);
    const int brow[2] = {wn * 16, 32 + wn * 16};
    gemm_mainloop_g<FMT, MTW, 2>(wr, 6, xrow, ws, acc, brow, active);
    if (!active) return;
    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = bx * 32 + wn * 16 + acc_n(r, lane);
            const float sq = __half2float(sqx[c]);
            const int ml = ty * BMT + wm * MTW * 16 + i * 16 + acc_m(r, lane);
            if (ml < n) {
                const float g = acc[i][0][r], u = acc[i][1][r];
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + ml) * MOE_D + c] = __float2half(hh / sq);
            }
        }
}

// grid = (48, ceil(maxtokens/BM), min(512,P))
template <int WM, int MTW>
__global__ void __launch_bounds__(256) k_moe_gu_gemm(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                      const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                      const int* __restrict__ active, const int* __restrict__ nact, __half* __restrict__ h2, int nlo, int nhi) {
    constexpr int BMT = WM * MTW * 16;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BMT >= n) return;
    __shared__ __attribute__((aligned(16))) half_t ws[GEMM_BN * GEMM_LDW];
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
    switch (ed.gu_code) {
        case FMT_S1_25: gu_gemm_body<FMT_S1_25, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S1_5: gu_gemm_body<FMT_S1_5, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S2: gu_gemm_body<FMT_S2, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S3: gu_gemm_body<FMT_S3, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S4: gu_gemm_body<FMT_S4, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S5: gu_gemm_body<FMT_S5, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S6: gu_gemm_body<FMT_S6, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        case FMT_S8: gu_gemm_body<FMT_S8, WM, MTW>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, ws); break;
        default: break;
    }
}

template <int FMT, int WM, int MTW, int WN, int NTW>
__device__ __forceinline__ void dn_gemm_body(const uint8_t* dnw, int bx, int ty, int n, int base, const int* pairs, const __half* h2, const float* topk_w,
                                             float* y, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BMT = WM * MTW * 16;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const bool active = wn < WN;
    const uint8_t* wr = dnw + (size_t)(bx * GEMM_BN + (tid >> 2)) * 6 * SB;
    const half_t* xrow[MTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
        int ml = ty * BMT + wm * MTW * 16 + i * 16 + (lane & 15);
        ml = ml < n ? ml : n - 1;
        xrow[i] = (const half_t*)h2 + (size_t)(base + ml) * MOE_D;
    }
    fv8 acc[MTW][NTW];
    acc_zero(acc);
    int brow[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) brow[j] = (wn * NTW + j) * 16;
    gemm_mainloop_g<FMT, MTW, NTW>(wr, 6, xrow, ws, acc, brow, active);
    if (!active) return;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c = bx * GEMM_BN + (wn * NTW + j) * 16 + acc_n(r, lane);
                const int ml = ty * BMT + wm * MTW * 16 + i * 16 + acc_m(r, lane);
                if (ml < n) {
                    const int pid = pairs[base + ml];
                    y[(size_t)pid * MOE_D + c] = acc[i][j][r] * topk_w[pid];
                }
            }
        }
}

// grid = (24, ceil(maxtokens/BM), min(512,P))
template <int WM, int MTW, int WN, int NTW>
__global__ void __launch_bounds__(256) k_moe_dn_gemm(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                      const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                      const int* __restrict__ active, const int* __restrict__ nact, const __half* __restrict__ h2,
                                                      float* __restrict__ y, int nlo, int nhi) {
    constexpr int BMT = WM * MTW * 16;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BMT >= n) return;
    __shared__ __attribute__((aligned(16))) half_t ws[GEMM_BN * GEMM_LDW];
    const ExpertDev ed = ex[e];
    const uint8_t* dnw = sec4 + ed.body_off + ed.dn_off;
    switch (ed.dn_code) {
        case FMT_S1_25: dn_gemm_body<FMT_S1_25, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S1_5: dn_gemm_body<FMT_S1_5, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S2: dn_gemm_body<FMT_S2, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S3: dn_gemm_body<FMT_S3, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S4: dn_gemm_body<FMT_S4, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S5: dn_gemm_body<FMT_S5, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S6: dn_gemm_body<FMT_S6, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S8: dn_gemm_body<FMT_S8, WM, MTW, WN, NTW>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, ws); break;
        default: break;
    }
}

#endif  // HY4_MMA

__global__ void k_moe_reduce(const float* __restrict__ y, float* __restrict__ z_out) {
    const int t = blockIdx.x;
    for (int n = threadIdx.x; n < MOE_D; n += blockDim.x) {
        float a = 0.f;
#pragma unroll
        for (int s = 0; s < MOE_TOPK; ++s) a += y[((size_t)t * MOE_TOPK + s) * MOE_D + n];
        z_out[(size_t)t * MOE_D + n] = a;
    }
}

// ---------------------------------------------------------------- phase-1.5 reference path (any M, up to 8 pairs per weight pass)
template <int FMT, int MT>
__device__ __forceinline__ void wave_dot_row(const uint8_t* rp, int K, const __half* const (&xp)[MT], float (&acc)[MT], int lane) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int nsb = K >> 8;
    for (int unit = lane; unit < nsb * 32; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        const uint8_t* blk = rp + (size_t)sbi * SB;
        const float D = h2f_bits(ld16(blk));
        const float eff = D * ((float)blk[2 + g] / 255.0f);
        float w[8];
        if constexpr (FMT == FMT_S1_25 || FMT == FMT_S1_5) {
#pragma unroll
            for (int t = 0; t < 8; ++t) w[t] = sk_weight<FMT>(blk, j + t);
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) w[t] = ((float)sk_q<FMT>(blk + 6, j + t) - sk_half<FMT>()) * eff;
        }
#pragma unroll
        for (int mm = 0; mm < MT; ++mm) {
            if (xp[mm]) {
                const __half* x = xp[mm] + sbi * 256 + j;
                float a = 0.f;
#pragma unroll
                for (int t = 0; t < 8; ++t) a += w[t] * __half2float(x[t]);
                acc[mm] += a;
            }
        }
    }
}

template <int MT>
__device__ __forceinline__ void wave_dot_dispatch(int code, const uint8_t* rp, int K, const __half* const (&xp)[MT], float (&acc)[MT], int lane) {
    switch (code) {
        case FMT_S1_25: wave_dot_row<FMT_S1_25, MT>(rp, K, xp, acc, lane); break;
        case FMT_S1_5: wave_dot_row<FMT_S1_5, MT>(rp, K, xp, acc, lane); break;
        case FMT_S2: wave_dot_row<FMT_S2, MT>(rp, K, xp, acc, lane); break;
        case FMT_S3: wave_dot_row<FMT_S3, MT>(rp, K, xp, acc, lane); break;
        case FMT_S4: wave_dot_row<FMT_S4, MT>(rp, K, xp, acc, lane); break;
        case FMT_S5: wave_dot_row<FMT_S5, MT>(rp, K, xp, acc, lane); break;
        case FMT_S6: wave_dot_row<FMT_S6, MT>(rp, K, xp, acc, lane); break;
        case FMT_S8: wave_dot_row<FMT_S8, MT>(rp, K, xp, acc, lane); break;
        default: break;
    }
}

__global__ void k_moe_hist(const ExpertDev* __restrict__ ex, const int32_t* __restrict__ idx, int P, int* __restrict__ cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P && !ex[idx[i]].pad) atomicAdd(&cnt[idx[i]], 1);
}
__global__ void k_moe_scan(const int* __restrict__ cnt, int* __restrict__ off) {
    if (threadIdx.x == 0) {
        int a = 0;
        for (int e = 0; e < MOE_E; ++e) { off[e] = a; a += cnt[e]; }
        off[MOE_E] = a;
    }
}
__global__ void k_moe_scatter(const ExpertDev* __restrict__ ex, const int32_t* __restrict__ idx, int P, const int* __restrict__ off, int* __restrict__ fill, int* __restrict__ pairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P && !ex[idx[i]].pad) {
        const int e = idx[i];
        const int pos = atomicAdd(&fill[e], 1);
        pairs[off[e] + pos] = i;
    }
}

__global__ void __launch_bounds__(256) k_moe_gu_ref(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                     const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                     __half* __restrict__ h2) {
    const int e = blockIdx.y;
    const int n = cnt[e];
    if (n == 0) return;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
    const __half* sqx = (const __half*)body;
    const int rb = code_row_bytes(ed.gu_code, MOE_D);
    const uint8_t* gu = body + 3072;
    const uint8_t* grow = gu + (size_t)r * rb;
    const uint8_t* urow = gu + (size_t)(MOE_D + r) * rb;
    const int base = off[e];
    const float sq = __half2float(sqx[r]);
    for (int c0 = 0; c0 < n; c0 += 8) {
        const __half* xp[8];
        int pid[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (c0 + i < n) {
                pid[i] = pairs[base + c0 + i];
                xp[i] = z + (size_t)(pid[i] / MOE_TOPK) * MOE_D;
            } else {
                pid[i] = -1;
                xp[i] = nullptr;
            }
        }
        float ag[8], au[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { ag[i] = 0.f; au[i] = 0.f; }
        wave_dot_dispatch<8>(ed.gu_code, grow, MOE_D, xp, ag, lane);
        wave_dot_dispatch<8>(ed.gu_code, urow, MOE_D, xp, au, lane);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float g = ag[i], u = au[i];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) { g += __shfl_xor(g, o); u += __shfl_xor(u, o); }
            if (lane == 0 && pid[i] >= 0) {
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + c0 + i) * MOE_D + r] = __float2half(hh / sq);
            }
        }
    }
}

__global__ void __launch_bounds__(256) k_moe_dn_ref(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                     const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                     const __half* __restrict__ h2, float* __restrict__ y) {
    const int e = blockIdx.y;
    const int n = cnt[e];
    if (n == 0) return;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const ExpertDev ed = ex[e];
    const uint8_t* dn = sec4 + ed.body_off + ed.dn_off;
    const uint8_t* drow = dn + (size_t)r * code_row_bytes(ed.dn_code, MOE_D);
    const int base = off[e];
    for (int c0 = 0; c0 < n; c0 += 8) {
        const __half* xp[8];
        int pid[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (c0 + i < n) {
                pid[i] = pairs[base + c0 + i];
                xp[i] = h2 + (size_t)(base + c0 + i) * MOE_D;
            } else {
                pid[i] = -1;
                xp[i] = nullptr;
            }
        }
        float a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = 0.f;
        wave_dot_dispatch<8>(ed.dn_code, drow, MOE_D, xp, a, lane);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = a[i];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
            if (lane == 0 && pid[i] >= 0) y[(size_t)pid[i] * MOE_D + r] = v * topk_w[pid[i]];
        }
    }
}

struct MoeWs {
    int *cnt, *off, *fill, *active, *nact, *pairs;
    __half* h2;
    float* y;
};
inline MoeWs moe_ws_split(void* ws, int P) {
    uint8_t* w = (uint8_t*)ws;
    MoeWs s;
    s.cnt = (int*)w;
    s.off = s.cnt + 512;
    s.fill = s.off + 513;
    s.active = s.fill + 512;
    s.nact = s.active + 512;
    w += moe_align(sizeof(int) * 2056);
    s.pairs = (int*)w;
    w += moe_align((size_t)P * 4);
    s.h2 = (__half*)w;
    w += moe_align((size_t)P * MOE_D * 2);
    s.y = (float*)w;
    return s;
}

}  // namespace skd

// z: __half [M][1536] (= routed_expert_down_proj output; 16-byte aligned rows), topk_idx int32 [M][16], topk_w float [M][16] (normalized * routed_scaling_factor),
// z_out: float [M][1536] (overwritten = sum over the 16 slots of w * y). ws: moe_ws_bytes(M) bytes of device memory.
// tuning knob: for M <= moe_dec_max_m() the GEMV-style decode kernels run, above it the grouped WMMA kernels with the smallest token tile per expert
namespace skd {
inline int& moe_has_cpu() { static int v = 0; return v; }  // 1 = some experts are CPU-resident (ExpertDev.pad != 0): their y rows are zeroed
inline int& moe_dec_max_m() { static int v = HY4_MMA ? 4 : 8; return v; }
}

inline void moe_experts_ref(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                            float* z_out, int M, void* ws);

inline void moe_experts(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                        float* z_out, int M, void* ws) {
#if !HY4_MMA
    if (M > 8) {  // no matrix-core grouped GEMM on this build: the simple path handles any M
        moe_experts_ref(s, sec4_base, experts, z, topk_idx, topk_w, z_out, M, ws);
        return;
    }
#endif
    const int P = M * MOE_TOPK;
    const skd::MoeWs w = skd::moe_ws_split(ws, P);
    const int eb = P < MOE_E ? P : MOE_E;
    if (skd::moe_has_cpu()) (void)hipMemsetAsync(w.y, 0, (size_t)P * MOE_D * 4, s);
    hipLaunchKernelGGL(skd::k_moe_route, dim3(1), dim3(512), 0, s, experts, topk_idx, P, w.cnt, w.off, w.pairs, w.active, w.nact);
    if (M <= skd::moe_dec_max_m() && M <= 16) {
#define HY4_DEC(MT)                                                                                                                                         \
    hipLaunchKernelGGL((skd::k_moe_gu_dec<MT>), dim3(MOE_D / 8, eb), dim3(256), 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2);  \
    hipLaunchKernelGGL((skd::k_moe_dn_dec<MT>), dim3(MOE_D / 16, eb), dim3(256), 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact,   \
                       w.h2, w.y)
        if (M <= 1) { HY4_DEC(1); }
        else if (M <= 2) { HY4_DEC(2); }
        else if (M <= 4) { HY4_DEC(4); }
        else if (M <= 8) { HY4_DEC(8); }
        else { HY4_DEC(16); }
#undef HY4_DEC
    }
#if HY4_MMA
    else {
        const int INF = 1 << 30;
        const dim3 b(256);
        // token tiles 16 / 32 / 64 / 128; an expert with n tokens uses the first tile with n <= BM (several 128-tiles above 64 tokens)
        hipLaunchKernelGGL((skd::k_moe_gu_gemm<1, 1>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 0, 16);
        hipLaunchKernelGGL((skd::k_moe_dn_gemm<1, 1, 4, 1>), dim3(MOE_D / skd::GEMM_BN, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active,
                           w.nact, w.h2, w.y, 0, 16);
        if (M > 16) {
            hipLaunchKernelGGL((skd::k_moe_gu_gemm<2, 1>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 16, 32);
            hipLaunchKernelGGL((skd::k_moe_dn_gemm<2, 1, 4, 1>), dim3(MOE_D / skd::GEMM_BN, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs,
                               w.active, w.nact, w.h2, w.y, 16, 32);
        }
        if (M > 32) {
            hipLaunchKernelGGL((skd::k_moe_gu_gemm<4, 1>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 32, 64);
            hipLaunchKernelGGL((skd::k_moe_dn_gemm<2, 2, 4, 1>), dim3(MOE_D / skd::GEMM_BN, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs,
                               w.active, w.nact, w.h2, w.y, 32, 64);
        }
        if (M > 64) {
            const int ty = (M + skd::GEMM_BM - 1) / skd::GEMM_BM;
            hipLaunchKernelGGL((skd::k_moe_gu_gemm<4, 2>), dim3(MOE_D / 32, ty, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 64, INF);
            hipLaunchKernelGGL((skd::k_moe_dn_gemm<4, 2, 2, 2>), dim3(MOE_D / skd::GEMM_BN, ty, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs,
                               w.active, w.nact, w.h2, w.y, 64, INF);
        }
    }
#endif
    hipLaunchKernelGGL(skd::k_moe_reduce, dim3(M), dim3(256), 0, s, w.y, z_out);
}

// simple reference path (any M); same arguments and workspace
inline void moe_experts_ref(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                            float* z_out, int M, void* ws) {
    const int P = M * MOE_TOPK;
    const skd::MoeWs w = skd::moe_ws_split(ws, P);
    if (skd::moe_has_cpu()) (void)hipMemsetAsync(w.y, 0, (size_t)P * MOE_D * 4, s);
    (void)hipMemsetAsync(w.cnt, 0, sizeof(int) * 512, s);
    (void)hipMemsetAsync(w.fill, 0, sizeof(int) * 512, s);
    hipLaunchKernelGGL(skd::k_moe_hist, dim3((P + 255) / 256), dim3(256), 0, s, experts, topk_idx, P, w.cnt);
    hipLaunchKernelGGL(skd::k_moe_scan, dim3(1), dim3(32), 0, s, w.cnt, w.off);
    hipLaunchKernelGGL(skd::k_moe_scatter, dim3((P + 255) / 256), dim3(256), 0, s, experts, topk_idx, P, w.off, w.fill, w.pairs);
    hipLaunchKernelGGL(skd::k_moe_gu_ref, dim3(MOE_D / 8, MOE_E), dim3(256), 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.h2);
    hipLaunchKernelGGL(skd::k_moe_dn_ref, dim3(MOE_D / 8, MOE_E), dim3(256), 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.h2, w.y);
    hipLaunchKernelGGL(skd::k_moe_reduce, dim3(M), dim3(256), 0, s, w.y, z_out);
}

}  // namespace hy4
