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
#if !HY4_CUDA
#include "sk_rdna2.hip.h"
#endif
#include "skwmma.hip.h"
#if HY4_MMA && !HY4_CUDA
#include "sk_int8_rdna4.hip.h"
#endif

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
inline int moe_max128(int P) { return (P >> 7) + (P / 65 < MOE_E ? P / 65 : MOE_E) + 1; }
// workspace layout: ints [cnt 512 | off 513 | fill 512 | active 512 | nact 1 | wcnt 4 | wl16 512 | wl32 512 | wl64 512] | pairs[P] | wl128_e[M128] | wl128_ty[M128] | h2[P][1536] (half) | y[P][1536] (float),  P = M*16
inline size_t moe_ws_bytes(int M) {
    const size_t P = (size_t)M * MOE_TOPK;
    const size_t m128 = (size_t)moe_max128((int)P);
    return moe_align(sizeof(int) * 3596) + moe_align(P * 4) + moe_align(m128 * 8) + moe_align(P * MOE_D * 2) + moe_align(P * MOE_D * 4);
}

namespace skd {

// ---------------------------------------------------------------- routing (single block, counting sort by expert + compact tile worklists)
__global__ void __launch_bounds__(512) k_moe_route(const ExpertDev* __restrict__ ex, const int32_t* __restrict__ idx, int P, int* __restrict__ cnt, int* __restrict__ off, int* __restrict__ pairs,
                                                    int* __restrict__ active, int* __restrict__ nact, int* __restrict__ wcnt, int* __restrict__ wl16, int* __restrict__ wl32,
                                                    int* __restrict__ wl64, int* __restrict__ wl128_e, int* __restrict__ wl128_ty) {
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
    __syncthreads();
    // build compact per-bucket worklists (zero empty blocks in WMMA dispatch).
    // When P <= 256 (decode/verify M <= 16), wl32/wl64 are unused by WMMA, so reuse them as wl1 (cnt == 1) and wlm (cnt >= 2) for split-occupancy decode!
    const int b16 = (cv > 0 && cv <= 16) ? 1 : 0;
    const int b32 = P <= 256 ? (cv == 1 ? 1 : 0) : ((cv > 16 && cv <= 32) ? 1 : 0);
    const int b64 = P <= 256 ? (cv > 1 ? 1 : 0) : ((cv > 32 && cv <= 64) ? 1 : 0);
    const int b128 = (cv > 64) ? ((cv + 127) >> 7) : 0;
    c[t] = b16;
    sc[t] = b32;
    fl[t] = b64;
    fill[t] = b128;
    __syncthreads();
    for (int d = 1; d < 512; d <<= 1) {
        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (t >= d) { a0 = c[t - d]; a1 = sc[t - d]; a2 = fl[t - d]; a3 = fill[t - d]; }
        __syncthreads();
        c[t] += a0; sc[t] += a1; fl[t] += a2; fill[t] += a3;
        __syncthreads();
    }
    if (b16) wl16[c[t] - 1] = t;
    if (b32) wl32[sc[t] - 1] = t;
    if (b64) wl64[fl[t] - 1] = t;
    if (b128 > 0) {
        const int base128 = fill[t] - b128;
        for (int ty = 0; ty < b128; ++ty) {
            wl128_e[base128 + ty] = t;
            wl128_ty[base128 + ty] = ty;
        }
    }
    if (t == 511) { wcnt[0] = c[511]; wcnt[1] = sc[511]; wcnt[2] = fl[511]; wcnt[3] = fill[511]; }
}

// ---------------------------------------------------------------- decode path (M <= 16)
// dot products of one weight row (K = 1536, FMT) with MT x-vectors held in LDS as uint4[MT][192]
// Streams 1 super-block at a time (keeps VGPRs low so occupancy stays high at MT=4/8; bit-identical to loading all 6).
template <int FMT, int MT>
__device__ __forceinline__ void row_dot(const uint8_t* rp, const uint4* xs, int lane, float (&acc)[MT]) {
    constexpr int SB = sk_sb_bytes(FMT);
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = rp + s * SB;
        const float eff = sk_eff<FMT>(blk, lane >> 3);
        uint32_t wp[4];
        decode_w8<FMT>(blk + 6, lane, wp);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint4 xv = xs[m * 192 + s * 32 + lane];
            float p = dot2f(wp[0], xv.x, 0.f);
            p = dot2f(wp[1], xv.y, p);
            p = dot2f(wp[2], xv.z, p);
            p = dot2f(wp[3], xv.w, p);
            acc[m] += eff * p;
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
                                                     const int* __restrict__ active, const int* __restrict__ nact, __half* __restrict__ h2, const float* __restrict__ zf) {
    if ((int)blockIdx.y >= *nact) return;
    const int e = active[blockIdx.y];
    const int n = cnt[e], base = off[e];
    __shared__ uint4 xs[MT * 192];
    s125_lut_init();  // the S1.25 codebook (published by the barrier below)
    for (int i = threadIdx.x; i < MT * 192; i += 256) {
        const int m = i / 192, c = i - m * 192;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < n) {
            if (zf) {  // fp32 latent z (the router-side down projection output): converted here, same rounding as a separate fp32 -> fp16 pass
                const float* zp = zf + (size_t)(pairs[base + m] >> 4) * MOE_D + c * 8;
                const float4 a = *(const float4*)zp, b = *(const float4*)(zp + 4);
                v.x = h2_pack(a.x, a.y);
                v.y = h2_pack(a.z, a.w);
                v.z = h2_pack(b.x, b.y);
                v.w = h2_pack(b.z, b.w);
            } else v = *(const uint4*)(z + (size_t)(pairs[base + m] >> 4) * MOE_D + c * 8);
        }
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

// ---------------------------------------------------------------- one-token decode (M = 1) kernels, "slot record" path
// With one token the 16 slots are the 16 chosen experts: no counting sort, no per-expert token lists. The router kernel copies the 16 ExpertDev records of the chosen experts into a
// 16-entry table (rec[slot]); gu / dn read rec[blockIdx.y] directly. Blocks issue their weight loads before the activation tile goes through LDS, so the DRAM latency of the weights
// overlaps the tile load and barrier (S2..S8 only; S1.25 / S1.5 experts take the generic path). Same arithmetic as k_moe_gu_dec<1> / k_moe_dn_dec<1> (bit-identical).
template <int FMT>
struct RowRaw {
    uint32_t a[6], b[6], d[6], ub[6];
};
template <int FMT>
__device__ __forceinline__ void row_raw_load(const uint8_t* rp, int lane, RowRaw<FMT>& r) {
    constexpr int SB = sk_sb_bytes(FMT);
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = rp + s * SB;
        r.d[s] = ld16(blk);
        r.ub[s] = blk[2 + (lane >> 3)];
        load_q_raw<FMT>(blk + 6, lane, r.a[s], r.b[s]);
    }
}
template <int FMT>
__device__ __forceinline__ float row_raw_dot(const RowRaw<FMT>& r, const uint4* xs, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        const float eff = h2f_bits((uint16_t)r.d[s]) * ((float)r.ub[s] * (1.0f / 255.0f));
        uint32_t q[8], wp[4];
        q_from_raw<FMT>(r.a[s], r.b[s], q);
        w_from_q<FMT>(q, wp);
        const uint4 xv = xs[s * 32 + lane];
        float p = dot2f(wp[0], xv.x, 0.f);
        p = dot2f(wp[1], xv.y, p);
        p = dot2f(wp[2], xv.z, p);
        p = dot2f(wp[3], xv.w, p);
        acc += eff * p;
    }
    return acc;
}

__device__ __forceinline__ uint4 load_z8(const __half* z, const float* zf, int c) {  // 8 activations c*8.. as packed halves (fp32 source: converted, same rounding as k_f2h)
    uint4 v;
    if (zf) {
        const float4 a = *(const float4*)(zf + c * 8), b = *(const float4*)(zf + c * 8 + 4);
        v.x = h2_pack(a.x, a.y);
        v.y = h2_pack(a.z, a.w);
        v.z = h2_pack(b.x, b.y);
        v.w = h2_pack(b.z, b.w);
    } else v = *(const uint4*)(z + c * 8);
    return v;
}

template <int FMT>
__device__ __forceinline__ void gu1_fmt(const uint8_t* gu, int r, int y, const uint4 zv, uint4* xs, __half* h2, const __half* sqx) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int lane = threadIdx.x & 31;
    RowRaw<FMT> g, u;
    row_raw_load<FMT>(gu + (size_t)r * 6 * SB, lane, g);
    row_raw_load<FMT>(gu + (size_t)(MOE_D + r) * 6 * SB, lane, u);
    if (threadIdx.x < 192) xs[threadIdx.x] = zv;
    __syncthreads();
    float ag = row_raw_dot<FMT>(g, xs, lane), au = row_raw_dot<FMT>(u, xs, lane);
    const float sq = __half2float(sqx[r]);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) { ag += __shfl_xor(ag, o); au += __shfl_xor(au, o); }
    if (lane == 0) {
        const float sg = ag / (1.0f + expf(-ag));
        const float hh = fminf(sg, 10.0f) * fminf(fmaxf(au, -10.0f), 10.0f);
        h2[(size_t)y * MOE_D + r] = __float2half(hh / sq);
    }
}

// grid (192, 16): block = 8 waves, wave = (gate row r, up row 1536 + r) of expert rec[blockIdx.y]
__global__ void __launch_bounds__(256) k_moe_gu_dec1(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ rec, const __half* __restrict__ z, const float* __restrict__ zf,
                                                      __half* __restrict__ h2) {
    const int y = blockIdx.y;
    const ExpertDev ed = rec[y];
    __shared__ uint4 xs[192];
    s125_lut_init();
    uint4 zv = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < 192) zv = load_z8(z, zf, threadIdx.x);
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const uint8_t* body = sec4 + ed.body_off;
    const uint8_t* gu = body + 3072;
    const __half* sqx = (const __half*)body;
    switch (ed.gu_code) {
        case FMT_S2: gu1_fmt<FMT_S2>(gu, r, y, zv, xs, h2, sqx); break;
        case FMT_S3: gu1_fmt<FMT_S3>(gu, r, y, zv, xs, h2, sqx); break;
        case FMT_S4: gu1_fmt<FMT_S4>(gu, r, y, zv, xs, h2, sqx); break;
        case FMT_S5: gu1_fmt<FMT_S5>(gu, r, y, zv, xs, h2, sqx); break;
        case FMT_S6: gu1_fmt<FMT_S6>(gu, r, y, zv, xs, h2, sqx); break;
        case FMT_S8: gu1_fmt<FMT_S8>(gu, r, y, zv, xs, h2, sqx); break;
        default:  // S1.25 / S1.5: generic path
            if (threadIdx.x < 192) xs[threadIdx.x] = zv;
            __syncthreads();
            gu_dec_rows<1>(ed, body, r, xs, lane, 1, y, h2);
            break;
    }
    (void)wave;
}

template <int FMT>
__device__ __forceinline__ void dn1_fmt(const uint8_t* dn, int r0, int y, const uint4 hv, uint4* xs, const float* topk_w, float* yb) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int lane = threadIdx.x & 31;
    RowRaw<FMT> a0, a1;
    row_raw_load<FMT>(dn + (size_t)r0 * 6 * SB, lane, a0);
    row_raw_load<FMT>(dn + (size_t)(r0 + 1) * 6 * SB, lane, a1);
    if (threadIdx.x < 192) xs[threadIdx.x] = hv;
    __syncthreads();
    float v0 = row_raw_dot<FMT>(a0, xs, lane), v1 = row_raw_dot<FMT>(a1, xs, lane);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) { v0 += __shfl_xor(v0, o); v1 += __shfl_xor(v1, o); }
    if (lane == 0) {
        const float w = topk_w[y];
        yb[(size_t)y * MOE_D + r0] = v0 * w;
        yb[(size_t)y * MOE_D + r0 + 1] = v1 * w;
    }
}

// grid (96, 16): wave = 2 output rows of expert rec[blockIdx.y]; the last of the 16 slot blocks of a column block (ticket tk[blockIdx.x], self-resetting, zero-initialised once) adds the
// 16 slots in slot order into z_out (fp32) and z_out_h (fp16): replaces k_moe_reduce for M = 1, deterministic.
__global__ void __launch_bounds__(256) k_moe_dn_dec1(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ rec, const float* __restrict__ topk_w, const __half* __restrict__ h2,
                                                      float* __restrict__ yb, float* __restrict__ z_out, __half* __restrict__ z_out_h, unsigned* __restrict__ tk) {
    const int y = blockIdx.y;
    const ExpertDev ed = rec[y];
    __shared__ uint4 xs[192];
    __shared__ int is_last;
    s125_lut_init();
    uint4 hv = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < 192) hv = *(const uint4*)(h2 + (size_t)y * MOE_D + threadIdx.x * 8);
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* dn = sec4 + ed.body_off + ed.dn_off;
    const int r0 = (blockIdx.x * 8 + wave) * 2;
    switch (ed.dn_code) {
        case FMT_S2: dn1_fmt<FMT_S2>(dn, r0, y, hv, xs, topk_w, yb); break;
        case FMT_S3: dn1_fmt<FMT_S3>(dn, r0, y, hv, xs, topk_w, yb); break;
        case FMT_S4: dn1_fmt<FMT_S4>(dn, r0, y, hv, xs, topk_w, yb); break;
        case FMT_S5: dn1_fmt<FMT_S5>(dn, r0, y, hv, xs, topk_w, yb); break;
        case FMT_S6: dn1_fmt<FMT_S6>(dn, r0, y, hv, xs, topk_w, yb); break;
        case FMT_S8: dn1_fmt<FMT_S8>(dn, r0, y, hv, xs, topk_w, yb); break;
        default:
            if (threadIdx.x < 192) xs[threadIdx.x] = hv;
            __syncthreads();
            {
                // generic (S1.25 / S1.5) rows
                const int rb = code_row_bytes(ed.dn_code, MOE_D);
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    float a[1] = {0.f};
                    row_dot_sw<1>(ed.dn_code, dn + (size_t)(r0 + rr) * rb, xs, lane, a);
                    float v = a[0];
#pragma unroll
                    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o);
                    if (lane == 0) yb[(size_t)y * MOE_D + r0 + rr] = v * topk_w[y];
                }
            }
            break;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(&tk[blockIdx.x], 1u);
        is_last = (prev == MOE_TOPK - 1);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (threadIdx.x < 16) {
        const int r = blockIdx.x * 16 + threadIdx.x;
        float a = 0.f;
#pragma unroll
        for (int s = 0; s < MOE_TOPK; ++s) a += *(volatile const float*)&yb[(size_t)s * MOE_D + r];
        z_out[r] = a;
        if (z_out_h) z_out_h[r] = __float2half(a);
    }
    if (threadIdx.x == 0) tk[blockIdx.x] = 0;
}

// Worklist-split decode kernels for M = 2..16:
//   - experts with cnt == 1 run the exact 223 GB/s raw-load-pipelined 1-token kernel (3 KB LDS, max occupancy)
//   - experts with cnt >= 2 run a lean runtime-n loop (no 5-way MTE template bloat -> half the VGPRs)
// Arithmetic per (token, expert) is 100% bit-identical to k_moe_gu_dec1 / k_moe_dn_dec1!
__global__ void __launch_bounds__(256) k_moe_gu_dec1_wl(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                        const int* __restrict__ off, const int* __restrict__ pairs, const int* __restrict__ wl1, const int* __restrict__ n1,
                                                        __half* __restrict__ h2, const float* __restrict__ zf) {
    if ((int)blockIdx.y >= *n1) return;
    const int e = wl1[blockIdx.y];
    const int base = off[e];
    const int tok = pairs[base] >> 4;
    const ExpertDev ed = ex[e];
    __shared__ uint4 xs[192];
    s125_lut_init();
    uint4 zv = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < 192) zv = load_z8(z ? z + (size_t)tok * MOE_D : nullptr, zf ? zf + (size_t)tok * MOE_D : nullptr, threadIdx.x);
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + wave;
    const uint8_t* body = sec4 + ed.body_off;
    const uint8_t* gu = body + 3072;
    const __half* sqx = (const __half*)body;
    switch (ed.gu_code) {
        case FMT_S2: gu1_fmt<FMT_S2>(gu, r, base, zv, xs, h2, sqx); break;
        case FMT_S3: gu1_fmt<FMT_S3>(gu, r, base, zv, xs, h2, sqx); break;
        case FMT_S4: gu1_fmt<FMT_S4>(gu, r, base, zv, xs, h2, sqx); break;
        case FMT_S5: gu1_fmt<FMT_S5>(gu, r, base, zv, xs, h2, sqx); break;
        case FMT_S6: gu1_fmt<FMT_S6>(gu, r, base, zv, xs, h2, sqx); break;
        case FMT_S8: gu1_fmt<FMT_S8>(gu, r, base, zv, xs, h2, sqx); break;
        default:
            if (threadIdx.x < 192) xs[threadIdx.x] = zv;
            __syncthreads();
            gu_dec_rows<1>(ed, body, r, xs, lane, 1, base, h2);
            break;
    }
}

__global__ void __launch_bounds__(256) k_moe_dn_dec1_wl(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                        const int* __restrict__ off, const int* __restrict__ pairs, const int* __restrict__ wl1, const int* __restrict__ n1,
                                                        const __half* __restrict__ h2, float* __restrict__ y) {
    if ((int)blockIdx.y >= *n1) return;
    const int e = wl1[blockIdx.y];
    const int base = off[e];
    const int pid = pairs[base];
    const ExpertDev ed = ex[e];
    __shared__ uint4 xs[192];
    s125_lut_init();
    uint4 hv = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < 192) hv = *(const uint4*)(h2 + (size_t)base * MOE_D + threadIdx.x * 8);
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* dn = sec4 + ed.body_off + ed.dn_off;
    const int r0 = (blockIdx.x * 8 + wave) * 2;
    switch (ed.dn_code) {
        case FMT_S2: dn1_fmt<FMT_S2>(dn, r0, pid, hv, xs, topk_w, y); break;
        case FMT_S3: dn1_fmt<FMT_S3>(dn, r0, pid, hv, xs, topk_w, y); break;
        case FMT_S4: dn1_fmt<FMT_S4>(dn, r0, pid, hv, xs, topk_w, y); break;
        case FMT_S5: dn1_fmt<FMT_S5>(dn, r0, pid, hv, xs, topk_w, y); break;
        case FMT_S6: dn1_fmt<FMT_S6>(dn, r0, pid, hv, xs, topk_w, y); break;
        case FMT_S8: dn1_fmt<FMT_S8>(dn, r0, pid, hv, xs, topk_w, y); break;
        default:
            if (threadIdx.x < 192) xs[threadIdx.x] = hv;
            __syncthreads();
            dn_dec_rows<1>(ed, dn, r0, xs, lane, 1, base, pairs, topk_w, y);
            break;
    }
}

#if !HY4_CUDA
// ---------------------------------------------------------------- RDNA2 Wave32 sdot4 grouped MoE GEMMs (no WMMA, all 8 waves active, K-major int4 LDS reads)
// Gate/up: block handles 32 gate rows (0..31) + 32 up rows (32..63) of expert e across BM tokens (BM in {16, 32, 64}).
// Each thread computes TM tokens x TN_GU gate+up pairs directly (SPLIT_GU=true), so SwiGLU needs zero LDS exchange!
template <int FMT, int BM, int TM, int TN_GU>
__device__ __forceinline__ void gu_rdna2_body(rdna2::Rdna2Smem<BM, true>& sm, const uint8_t* body, int bx, int ty, int n, int base, const int* pairs, const __half* z, __half* h2) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int T_N = 32 / TN_GU;
    const int tid = threadIdx.x;
    const int tn = tid % T_N, tm = tid / T_N;
    const int m0 = tm * TM, n0 = tn * TN_GU;
    const int drow = tid >> 2;
    const int wrow = drow < 32 ? bx * 32 + drow : MOE_D + bx * 32 + (drow - 32);
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;

    float acc[TM][2 * TN_GU];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2 * TN_GU; ++j) acc[i][j] = 0.f;

    auto xrow_fn = [&](int r) -> const __half* {
        int ml = ty * BM + r;
        ml = ml < n ? ml : n - 1;
        return z + (size_t)(pairs[base + ml] >> 4) * MOE_D;
    };

    rdna2::rdna2_sdot4_loop<FMT, BM, TM, 2 * TN_GU, true, true>(sm, wr, 6, xrow_fn, m0, n0, acc);

    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int j = 0; j < TN_GU; ++j) {
        const int c = bx * 32 + n0 + j;
        const float sq = __half2float(sqx[c]);
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int ml = ty * BM + m0 + i;
            if (ml < n) {
                const float g = acc[i][j], u = acc[i][TN_GU + j];
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + ml) * MOE_D + c] = __float2half(hh / sq);
            }
        }
    }
}

template <int BM, int TM, int TN_GU>
__global__ void __launch_bounds__(256, 2) k_moe_gu_rdna2(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                         const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                         const int* __restrict__ active, const int* __restrict__ nact, __half* __restrict__ h2, int nlo, int nhi) {
    __shared__ rdna2::Rdna2Smem<BM, true> sm;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BM >= n) return;
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
    switch (ed.gu_code) {
        case FMT_S1_25: gu_rdna2_body<FMT_S1_25, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S1_5: gu_rdna2_body<FMT_S1_5, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S2: gu_rdna2_body<FMT_S2, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S3: gu_rdna2_body<FMT_S3, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S4: gu_rdna2_body<FMT_S4, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S5: gu_rdna2_body<FMT_S5, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S6: gu_rdna2_body<FMT_S6, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        case FMT_S8: gu_rdna2_body<FMT_S8, BM, TM, TN_GU>(sm, body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2); break;
        default: break;
    }
}

template <int FMT, int BM, int TM, int TN>
__device__ __forceinline__ void dn_rdna2_body(rdna2::Rdna2Smem<BM, true>& sm, const uint8_t* dnw, int bx, int ty, int n, int base, const int* pairs, const __half* h2, const float* topk_w, float* y) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int T_N = rdna2::R2_BN / TN;
    const int tid = threadIdx.x;
    const int tn = tid % T_N, tm = tid / T_N;
    const int m0 = tm * TM, n0 = tn * TN;
    const uint8_t* wr = dnw + (size_t)(bx * rdna2::R2_BN + (tid >> 2)) * 6 * SB;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    auto xrow_fn = [&](int r) -> const __half* {
        int ml = ty * BM + r;
        ml = ml < n ? ml : n - 1;
        return h2 + (size_t)(base + ml) * MOE_D;
    };

    rdna2::rdna2_sdot4_loop<FMT, BM, TM, TN, true, false>(sm, wr, 6, xrow_fn, m0, n0, acc);

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = ty * BM + m0 + i;
        if (ml >= n) continue;
        const int pid = pairs[base + ml];
        const float tw = topk_w[pid];
        float* yp = y + (size_t)pid * MOE_D + bx * rdna2::R2_BN + n0;
#pragma unroll
        for (int j = 0; j < TN; ++j) yp[j] = acc[i][j] * tw;
    }
}

template <int BM, int TM, int TN>
__global__ void __launch_bounds__(256, 2) k_moe_dn_rdna2(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                         const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                         const int* __restrict__ active, const int* __restrict__ nact, const __half* __restrict__ h2,
                                                         float* __restrict__ y, int nlo, int nhi) {
    __shared__ rdna2::Rdna2Smem<BM, true> sm;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BM >= n) return;
    const ExpertDev ed = ex[e];
    const uint8_t* dnw = sec4 + ed.body_off + ed.dn_off;
    switch (ed.dn_code) {
        case FMT_S1_25: dn_rdna2_body<FMT_S1_25, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S1_5: dn_rdna2_body<FMT_S1_5, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S2: dn_rdna2_body<FMT_S2, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S3: dn_rdna2_body<FMT_S3, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S4: dn_rdna2_body<FMT_S4, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S5: dn_rdna2_body<FMT_S5, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S6: dn_rdna2_body<FMT_S6, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        case FMT_S8: dn_rdna2_body<FMT_S8, BM, TM, TN>(sm, dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y); break;
        default: break;
    }
}
#endif  // !HY4_CUDA

#if HY4_MMA
#if !HY4_CUDA
// ---------------------------------------------------------------- HIP WMMA (gfx11 + gfx12) ALL-8-WAVE grouped GEMMs (prefill & verify)
// BM=16,32 use BN=128 BK=128 (gemm_mainloop_128: 1 thread = 1 subgroup = 64 weights, v_pk_mul_f16, global A, zero idle waves).
// BM=64,128 use BN=64 BK=256 (gemm_mainloop_g: 2 barriers/256 k, v_pk_mul_f16, global A, zero idle waves).
template <int FMT>
__device__ __forceinline__ void gu_sg_16(const uint8_t* body, int bx, int n, int base, const int* pairs, const __half* z, __half* h2, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int drow = tid >> 1;  // 0..127: 0..63 = gate cols, 64..127 = up cols
    const int wrow = drow < 64 ? bx * 64 + drow : MOE_D + bx * 64 + (drow - 64);
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;
    int ml = lane & 15;
    ml = ml < n ? ml : n - 1;
    const half_t* xrow[1] = {(const half_t*)z + (size_t)(pairs[base + ml] >> 4) * MOE_D};
    fv8 acc[1][1];
    acc_zero(acc);
    // waves 0..3 compute gate cols 0..63; waves 4..7 compute up cols 0..63 -> ALL 8 waves active!
    const int brow[1] = {wave < 4 ? wave * 16 : 64 + (wave - 4) * 16};
    gemm_mainloop_128<FMT, 1, 1>(wr, 6, xrow, ws, acc, brow);
    // reuse ws (34.8 KB) as float s_gate[16][64] (4 KB)
    float* s_gate = (float*)ws;
    if (wave < 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) s_gate[acc_m(r, lane) * 64 + wave * 16 + acc_n(r, lane)] = acc[0][0][r];
    }
    __syncthreads();
    if (wave >= 4) {
        const __half* sqx = (const __half*)body;
        const int col0 = (wave - 4) * 16;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int cl = col0 + acc_n(r, lane);
            const int c = bx * 64 + cl;
            const int m = acc_m(r, lane);
            if (m < n) {
                const float g = s_gate[m * 64 + cl], u = acc[0][0][r];
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + m) * MOE_D + c] = __float2half(hh / __half2float(sqx[c]));
            }
        }
    }
}

template <int FMT>
__device__ __forceinline__ void gu_sg_32(const uint8_t* body, int bx, int n, int base, const int* pairs, const __half* z, __half* h2, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave & 1, wn = wave >> 1;  // 2 M-waves x 4 N-waves = ALL 8 waves active!
    const int drow = tid >> 1;
    const int wrow = drow < 64 ? bx * 64 + drow : MOE_D + bx * 64 + (drow - 64);
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;
    int ml = wm * 16 + (lane & 15);
    ml = ml < n ? ml : n - 1;
    const half_t* xrow[1] = {(const half_t*)z + (size_t)(pairs[base + ml] >> 4) * MOE_D};
    fv8 acc[1][2];
    acc_zero(acc);
    const int brow[2] = {wn * 16, 64 + wn * 16};
    gemm_mainloop_128<FMT, 1, 2>(wr, 6, xrow, ws, acc, brow);
    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int c = bx * 64 + wn * 16 + acc_n(r, lane);
        const int m = wm * 16 + acc_m(r, lane);
        if (m < n) {
            const float g = acc[0][0][r], u = acc[0][1][r];
            const float sg = g / (1.0f + expf(-g));
            const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
            h2[(size_t)(base + m) * MOE_D + c] = __float2half(hh / __half2float(sqx[c]));
        }
    }
}

template <int FMT, int BM, int MTW>
__device__ __forceinline__ void gu_sg_64_128(const uint8_t* body, int bx, int ty, int n, int base, const int* pairs, const __half* z, __half* h2, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, drow = tid >> 2;
    const int wm = wave & 3, wn = wave >> 2;  // 4 M-waves x 2 N-waves = ALL 8 waves active!
    const int wrow = drow < 32 ? bx * 32 + drow : MOE_D + bx * 32 + (drow - 32);
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;
    const half_t* xrow[MTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
        int ml = ty * BM + wm * MTW * 16 + i * 16 + (lane & 15);
        ml = ml < n ? ml : n - 1;
        xrow[i] = (const half_t*)z + (size_t)(pairs[base + ml] >> 4) * MOE_D;
    }
    fv8 acc[MTW][2];
    acc_zero(acc);
    const int brow[2] = {wn * 16, 32 + wn * 16};
    gemm_mainloop_g<FMT, MTW, 2>(wr, 6, xrow, ws, acc, brow, true);
    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = bx * 32 + wn * 16 + acc_n(r, lane);
            const int ml = ty * BM + wm * MTW * 16 + i * 16 + acc_m(r, lane);
            if (ml < n) {
                const float g = acc[i][0][r], u = acc[i][1][r];
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + ml) * MOE_D + c] = __float2half(hh / __half2float(sqx[c]));
            }
        }
}

template <int FMT, int BM, int BN, int WM, int MTW, int WN, int NTW>
__device__ __forceinline__ void dn_sg_body(const uint8_t* dnw, int bx, int ty, int n, int base, const int* pairs, const __half* h2, const float* topk_w,
                                           float* y, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const int drow = BN == 128 ? (tid >> 1) : (tid >> 2);
    const uint8_t* wr = dnw + (size_t)(bx * BN + drow) * 6 * SB;
    const half_t* xrow[MTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
        int ml = ty * BM + wm * MTW * 16 + i * 16 + (lane & 15);
        ml = ml < n ? ml : n - 1;
        xrow[i] = (const half_t*)h2 + (size_t)(base + ml) * MOE_D;
    }
    fv8 acc[MTW][NTW];
    acc_zero(acc);
    int brow[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) brow[j] = (wn * NTW + j) * 16;
    if constexpr (BN == 128) gemm_mainloop_128<FMT, MTW, NTW>(wr, 6, xrow, ws, acc, brow);
    else gemm_mainloop_g<FMT, MTW, NTW>(wr, 6, xrow, ws, acc, brow, true);
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c = bx * BN + (wn * NTW + j) * 16 + acc_n(r, lane);
                const int ml = ty * BM + wm * MTW * 16 + i * 16 + acc_m(r, lane);
                if (ml < n) {
                    const int pid = pairs[base + ml];
                    y[(size_t)pid * MOE_D + c] = acc[i][j][r] * topk_w[pid];
                }
            }
}

template <int BM>
__global__ void __launch_bounds__(256) k_moe_gu_sg(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                   const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                   const int* __restrict__ wl_e, const int* __restrict__ wl_ty, const int* __restrict__ wcnt_p, __half* __restrict__ h2) {
    if ((int)blockIdx.y >= *wcnt_p) return;
    const int e = wl_e[blockIdx.y];
    const int ty = wl_ty ? wl_ty[blockIdx.y] : 0;
    const int n = cnt[e];
    __shared__ __attribute__((aligned(16))) half_t ws[BM <= 32 ? (128 * BK128_LD) : (GEMM_BN * GEMM_LDW)];
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
#define HY4_GU_SG_CASE(F)                                                                                    \
    case F:                                                                                                  \
        if constexpr (BM == 16) gu_sg_16<F>(body, blockIdx.x, n, off[e], pairs, z, h2, ws);                  \
        else if constexpr (BM == 32) gu_sg_32<F>(body, blockIdx.x, n, off[e], pairs, z, h2, ws);             \
        else if constexpr (BM == 64) gu_sg_64_128<F, 64, 1>(body, blockIdx.x, ty, n, off[e], pairs, z, h2, ws); \
        else gu_sg_64_128<F, 128, 2>(body, blockIdx.x, ty, n, off[e], pairs, z, h2, ws);                     \
        break
    switch (ed.gu_code) {
        HY4_GU_SG_CASE(FMT_S1_25);
        HY4_GU_SG_CASE(FMT_S1_5);
        HY4_GU_SG_CASE(FMT_S2);
        HY4_GU_SG_CASE(FMT_S3);
        HY4_GU_SG_CASE(FMT_S4);
        HY4_GU_SG_CASE(FMT_S5);
        HY4_GU_SG_CASE(FMT_S6);
        HY4_GU_SG_CASE(FMT_S8);
        default: break;
    }
#undef HY4_GU_SG_CASE
}

template <int BM, int BN, int WM, int MTW, int WN, int NTW>
__global__ void __launch_bounds__(256) k_moe_dn_sg(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                   const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                   const int* __restrict__ wl_e, const int* __restrict__ wl_ty, const int* __restrict__ wcnt_p,
                                                   const __half* __restrict__ h2, float* __restrict__ y) {
    if ((int)blockIdx.y >= *wcnt_p) return;
    const int e = wl_e[blockIdx.y];
    const int ty = wl_ty ? wl_ty[blockIdx.y] : 0;
    const int n = cnt[e];
    __shared__ __attribute__((aligned(16))) half_t ws[BN == 128 ? (128 * BK128_LD) : (GEMM_BN * GEMM_LDW)];
    const ExpertDev ed = ex[e];
    const uint8_t* dnw = sec4 + ed.body_off + ed.dn_off;
    switch (ed.dn_code) {
        case FMT_S1_25: dn_sg_body<FMT_S1_25, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S1_5: dn_sg_body<FMT_S1_5, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S2: dn_sg_body<FMT_S2, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S3: dn_sg_body<FMT_S3, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S4: dn_sg_body<FMT_S4, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S5: dn_sg_body<FMT_S5, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S6: dn_sg_body<FMT_S6, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        case FMT_S8: dn_sg_body<FMT_S8, BM, BN, WM, MTW, WN, NTW>(dnw, blockIdx.x, ty, n, off[e], pairs, h2, topk_w, y, ws); break;
        default: break;
    }
}
#else  // HY4_CUDA
// ---------------------------------------------------------------- WMMA grouped GEMMs (prefill and small verify windows)
// Token tile BM = WM*MTW*16 (16, 32, 64 or 128): the kernel variant handles the experts whose token count n satisfies nlo < n <= nhi, so every expert
// is processed with the smallest tile that covers it (a 128-token tile wastes 16x the matrix-core work on an expert that sees 8 tokens).
// On CUDA: all 8 warps active via warp-level split-K (WK = 4/WM = 4, 2, 1 for WM = 1, 2, 4), full grid.x = MOE_D/32 = 48, BK = 256.
// On HIP GFX12 (RDNA4): WN = 8 / WM so ALL 8 waves compute (WN=8 -> 128 gate+128 up rows @ BK=64; WN=4 -> 64+64 @ BK=128; WN=2 -> 32+32 @ BK=256).
template <int FMT, int WM, int MTW>
__device__ __forceinline__ void gu_gemm_body(const uint8_t* body, int bx, int ty, int n, int base, const int* pairs, const __half* z, __half* h2, half_t* ws) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BMT = WM * MTW * 16;
#if !HY4_CUDA
    constexpr int WN = 8 / WM;
    constexpr int BNG = WN * 16;       // 128 (WM=1), 64 (WM=2), 32 (WM=4)
    constexpr int BN_W = 2 * BNG;      // 256, 128, 64
    constexpr int BK = 16384 / BN_W;   // 64, 128, 256
    constexpr int G = BK / 64;         // 1, 2, 4
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const int drow = tid / G;
    const int wrow = drow < BNG ? bx * BNG + drow : bx * BNG + (drow - BNG) + MOE_D;
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
    const int brow[2] = {wn * 16, BNG + wn * 16};
    gemm_mainloop_bk<FMT, MTW, 2, BN_W, BK>(wr, 6, xrow, ws, acc, brow, true);
    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = bx * BNG + wn * 16 + acc_n(r, lane);
            const float sq = __half2float(sqx[c]);
            const int ml = ty * BMT + wm * MTW * 16 + i * 16 + acc_m(r, lane);
            if (ml < n) {
                const float g = acc[i][0][r], u = acc[i][1][r];
                const float sg = g / (1.0f + expf(-g));
                const float hh = fminf(sg, 10.0f) * fminf(fmaxf(u, -10.0f), 10.0f);
                h2[(size_t)(base + ml) * MOE_D + c] = __float2half(hh / sq);
            }
        }
#else
    constexpr int WK = 4 / WM;  // 4 (WM=1), 2 (WM=2), 1 (WM=4)
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM;
    const int wn = (wave / WM) & 1;
    const int wk = wave / (WM * 2);  // 0..WK-1: all 8 warps active
    const int drow = tid >> 2;
    const int wrow = drow < 32 ? bx * 32 + drow : MOE_D + bx * 32 + (drow - 32);
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
    gemm_mainloop_splitk<FMT, MTW, 2, WK>(wr, 6, xrow, ws, acc, brow, wm * 2 + wn, wk);
    if (wk != 0) return;
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
#endif
}

// grid = (MOE_D / BNG, ceil(maxtokens/BM), min(512,P))
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
#if !HY4_CUDA
    constexpr int BN_W = (8 / WM) * 32;
    constexpr int BK = 16384 / BN_W;
    __shared__ __attribute__((aligned(16))) half_t ws[BN_W * (BK + 8)];
#else
    __shared__ __align__(16) half_t ws[GEMM_BN * GEMM_LDW];
#endif
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
#if !HY4_CUDA
    constexpr int BN_W = WN * NTW * 16;   // 128 (<1,1,8,1>) or 64 (others)
    constexpr int BK = 16384 / BN_W;      // 128 or 256
    constexpr int G = BK / 64;            // 2 or 4
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const bool active = wn < WN;
    const uint8_t* wr = dnw + (size_t)(bx * BN_W + (tid / G)) * 6 * SB;
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
    gemm_mainloop_bk<FMT, MTW, NTW, BN_W, BK>(wr, 6, xrow, ws, acc, brow, active);
    if (!active) return;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c = bx * BN_W + (wn * NTW + j) * 16 + acc_n(r, lane);
                const int ml = ty * BMT + wm * MTW * 16 + i * 16 + acc_m(r, lane);
                if (ml < n) {
                    const int pid = pairs[base + ml];
                    y[(size_t)pid * MOE_D + c] = acc[i][j][r] * topk_w[pid];
                }
            }
        }
#else
    constexpr int WK = 8 / (WM * WN);  // 2 (WM=1,WN=4), 1 otherwise
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM;
    const int wn = (wave / WM) % WN;
    const int wk = wave / (WM * WN);
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
    gemm_mainloop_splitk<FMT, MTW, NTW, WK>(wr, 6, xrow, ws, acc, brow, wm * WN + wn, wk);
    if (wk != 0) return;
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
#endif
}

// grid = (MOE_D / BN_W, ceil(maxtokens/BM), min(512,P))
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
#if !HY4_CUDA
    constexpr int BN_W = WN * NTW * 16;
    constexpr int BK = 16384 / BN_W;
    __shared__ __attribute__((aligned(16))) half_t ws[BN_W * (BK + 8)];
#else
    __shared__ __align__(16) half_t ws[GEMM_BN * GEMM_LDW];
#endif
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
#endif  // !HY4_CUDA && !HY4_GFX12

#if !HY4_CUDA
// ---- RDNA4 Native INT8 WMMA grouped MoE kernels (all 8 waves active, total LDS <= 27.6 KB -> 2-3 WG/CU) ----
template <int FMT, int WM, int MTW, bool HI>
__device__ __forceinline__ void gu_i8_body(const uint8_t* body, int bx, int ty, int n, int base, const int* pairs, const __half* z, __half* h2,
                                           int8_t* sWq, float* sWs, int8_t* sXq, float* sXd) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BMT = WM * MTW * 16;
    constexpr int WN = 8 / WM;
    constexpr int BNG = WN * 16;
    constexpr int BN_W = 2 * BNG;
    constexpr int BK = 16384 / BN_W;
    constexpr int G = BK / 64;
    constexpr int TPR = 256 / BMT;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const int drow = tid / G;
    const int wrow = drow < BNG ? bx * BNG + drow : bx * BNG + (drow - BNG) + MOE_D;
    const uint8_t* wr = body + 3072 + (size_t)wrow * 6 * SB;

    int qml = ty * BMT + (tid / TPR);
    qml = qml < n ? qml : n - 1;
    const half_t* x_qptr = (const half_t*)z + (size_t)(pairs[base + qml] >> 4) * MOE_D;

    fv8 acc[MTW][2];
    acc_zero(acc);
    const int brow[2] = {wn * 16, BNG + wn * 16};
    gemm_mainloop_i8_rdna4<FMT, BMT, MTW, 2, BN_W, BK, HI>(wr, 6, x_qptr, wm * MTW * 16, brow, sWq, sWs, sXq, sXd, acc);

    const __half* sqx = (const __half*)body;
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = bx * BNG + wn * 16 + acc_n(r, lane);
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

template <int WM, int MTW, bool HI>
__global__ void __launch_bounds__(256) k_moe_gu_i8_rdna4(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const __half* __restrict__ z,
                                                          const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                          const int* __restrict__ active, const int* __restrict__ nact, __half* __restrict__ h2, int nlo, int nhi) {
    constexpr int BMT = WM * MTW * 16;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BMT >= n) return;
    constexpr int BN_W = (8 / WM) * 32;
    constexpr int BK = 16384 / BN_W;
    constexpr int G = BK / 64;
    __shared__ __attribute__((aligned(16))) int8_t sWq[BN_W * (BK + 8)];
    __shared__ __attribute__((aligned(16))) float sWs[BN_W * G];
    __shared__ __attribute__((aligned(16))) int8_t sXq[BMT * 72];
    __shared__ __attribute__((aligned(16))) float sXd[BMT];
    const ExpertDev ed = ex[e];
    const uint8_t* body = sec4 + ed.body_off;
    switch (ed.gu_code) {
        case FMT_S1_25: gu_i8_body<FMT_S1_25, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S1_5: gu_i8_body<FMT_S1_5, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S2: gu_i8_body<FMT_S2, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S3: gu_i8_body<FMT_S3, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S4: gu_i8_body<FMT_S4, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S5: gu_i8_body<FMT_S5, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S6: gu_i8_body<FMT_S6, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        case FMT_S8: gu_i8_body<FMT_S8, WM, MTW, HI>(body, blockIdx.x, blockIdx.y, n, off[e], pairs, z, h2, sWq, sWs, sXq, sXd); break;
        default: break;
    }
}

template <int FMT, int WM, int MTW, int WN, int NTW, bool HI>
__device__ __forceinline__ void dn_i8_body(const uint8_t* dnw, int bx, int ty, int n, int base, const int* pairs, const __half* h2, const float* topk_w,
                                           float* y, int8_t* sWq, float* sWs, int8_t* sXq, float* sXd) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int BMT = WM * MTW * 16;
    constexpr int BN_W = WN * NTW * 16;
    constexpr int BK = 16384 / BN_W;
    constexpr int G = BK / 64;
    constexpr int TPR = 256 / BMT;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WM, wn = wave / WM;
    const uint8_t* wr = dnw + (size_t)(bx * BN_W + (tid / G)) * 6 * SB;

    int qml = ty * BMT + (tid / TPR);
    qml = qml < n ? qml : n - 1;
    const half_t* x_qptr = (const half_t*)h2 + (size_t)(base + qml) * MOE_D;

    fv8 acc[MTW][NTW];
    acc_zero(acc);
    int brow[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) brow[j] = (wn * NTW + j) * 16;
    gemm_mainloop_i8_rdna4<FMT, BMT, MTW, NTW, BN_W, BK, HI>(wr, 6, x_qptr, wm * MTW * 16, brow, sWq, sWs, sXq, sXd, acc);

#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int c = bx * BN_W + (wn * NTW + j) * 16 + acc_n(r, lane);
                const int ml = ty * BMT + wm * MTW * 16 + i * 16 + acc_m(r, lane);
                if (ml < n) {
                    const int pid = pairs[base + ml];
                    y[(size_t)pid * MOE_D + c] = acc[i][j][r] * topk_w[pid];
                }
            }
}

template <int WM, int MTW, int WN, int NTW, bool HI>
__global__ void __launch_bounds__(256) k_moe_dn_i8_rdna4(const uint8_t* __restrict__ sec4, const ExpertDev* __restrict__ ex, const float* __restrict__ topk_w,
                                                          const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ pairs,
                                                          const int* __restrict__ active, const int* __restrict__ nact, const __half* __restrict__ h2,
                                                          float* __restrict__ y, int nlo, int nhi) {
    constexpr int BMT = WM * MTW * 16;
    if ((int)blockIdx.z >= *nact) return;
    const int e = active[blockIdx.z];
    const int n = cnt[e];
    if (n <= nlo || n > nhi) return;
    if ((int)blockIdx.y * BMT >= n) return;
    constexpr int BN_W = WN * NTW * 16;
    constexpr int BK = 16384 / BN_W;
    constexpr int G = BK / 64;
    __shared__ __attribute__((aligned(16))) int8_t sWq[BN_W * (BK + 8)];
    __shared__ __attribute__((aligned(16))) float sWs[BN_W * G];
    __shared__ __attribute__((aligned(16))) int8_t sXq[BMT * 72];
    __shared__ __attribute__((aligned(16))) float sXd[BMT];
    const ExpertDev ed = ex[e];
    const uint8_t* dnw = sec4 + ed.body_off + ed.dn_off;
    switch (ed.dn_code) {
        case FMT_S1_25: dn_i8_body<FMT_S1_25, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S1_5: dn_i8_body<FMT_S1_5, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S2: dn_i8_body<FMT_S2, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S3: dn_i8_body<FMT_S3, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S4: dn_i8_body<FMT_S4, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S5: dn_i8_body<FMT_S5, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S6: dn_i8_body<FMT_S6, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        case FMT_S8: dn_i8_body<FMT_S8, WM, MTW, WN, NTW, HI>(dnw, blockIdx.x, blockIdx.y, n, off[e], pairs, h2, topk_w, y, sWq, sWs, sXq, sXd); break;
        default: break;
    }
}
#endif  // !HY4_CUDA
#endif  // HY4_MMA

__global__ void k_moe_reduce(const float* __restrict__ y, float* __restrict__ z_out, __half* __restrict__ z_out_h,
                             const ExpertDev* __restrict__ ex, const int32_t* __restrict__ topk_idx, int accum) {
    const int t = blockIdx.x;
    __shared__ uint8_t act[MOE_TOPK];
    if (threadIdx.x < MOE_TOPK) {
        const int e = topk_idx[(size_t)t * MOE_TOPK + threadIdx.x];
        act[threadIdx.x] = (ex && e >= 0 && e < MOE_E && ex[e].pad) ? 0 : 1;
    }
    __syncthreads();
    for (int n = threadIdx.x; n < MOE_D; n += blockDim.x) {
        float a = accum ? z_out[(size_t)t * MOE_D + n] : 0.f;
#pragma unroll
        for (int s = 0; s < MOE_TOPK; ++s)
            if (act[s]) a += y[((size_t)t * MOE_TOPK + s) * MOE_D + n];
        z_out[(size_t)t * MOE_D + n] = a;
        if (z_out_h) z_out_h[(size_t)t * MOE_D + n] = __float2half(a);
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
    int *cnt, *off, *fill, *active, *nact, *wcnt, *wl16, *wl32, *wl64, *pairs, *wl128_e, *wl128_ty;
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
    s.wcnt = s.nact + 1;
    s.wl16 = s.wcnt + 4;
    s.wl32 = s.wl16 + 512;
    s.wl64 = s.wl32 + 512;
    w += moe_align(sizeof(int) * 3596);
    s.pairs = (int*)w;
    w += moe_align((size_t)P * 4);
    const size_t m128 = (size_t)moe_max128(P);
    s.wl128_e = (int*)w;
    s.wl128_ty = s.wl128_e + m128;
    w += moe_align(m128 * 8);
    s.h2 = (__half*)w;
    w += moe_align((size_t)P * MOE_D * 2);
    s.y = (float*)w;
    return s;
}

#if !HY4_CUDA
inline void launch_moe_rdna2(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const float* topk_w, int M, int eb, const MoeWs& w) {
    const int INF = 1 << 30;
    const dim3 b(256);
    hipLaunchKernelGGL((k_moe_gu_rdna2<16, 2, 1>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 0, 16);
    hipLaunchKernelGGL((k_moe_dn_rdna2<16, 2, 2>), dim3(MOE_D / rdna2::R2_BN, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 0, 16);
    if (M > 16) {
        hipLaunchKernelGGL((k_moe_gu_rdna2<32, 4, 1>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 16, 32);
        hipLaunchKernelGGL((k_moe_dn_rdna2<32, 4, 2>), dim3(MOE_D / rdna2::R2_BN, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 16, 32);
    }
    if (M > 32) {
        const int ty = (M + 63) / 64;
        hipLaunchKernelGGL((k_moe_gu_rdna2<64, 4, 2>), dim3(MOE_D / 32, ty, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 32, INF);
        hipLaunchKernelGGL((k_moe_dn_rdna2<64, 4, 4>), dim3(MOE_D / rdna2::R2_BN, ty, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 32, INF);
    }
}
#endif

}  // namespace skd

// z: __half [M][1536] (= routed_expert_down_proj output; 16-byte aligned rows), topk_idx int32 [M][16], topk_w float [M][16] (normalized * routed_scaling_factor),
// z_out: float [M][1536] (overwritten = sum over the 16 slots of w * y). ws: moe_ws_bytes(M) bytes of device memory.
// tuning knob: for M <= moe_dec_max_m() the GEMV-style decode kernels run, above it the grouped WMMA kernels with the smallest token tile per expert
namespace skd {
inline int& moe_has_cpu() { static int v = 0; return v; }  // 1 = some experts are CPU-resident (ExpertDev.pad != 0): their y rows are zeroed
#if HY4_CUDA
#define HY4_MOE_DEC_MAXM 8  // the MT = 16 decode kernels need 49 KB of static shared memory
#else
#define HY4_MOE_DEC_MAXM 16
#endif
inline int& moe_dec_max_m() {
    static int v = (!HY4_MMA || hy4_is_rdna2()) ? 16 : 4;
    return v;
}
}

inline void moe_experts_ref(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                            float* z_out, int M, void* ws, __half* z_out_h = nullptr);

// Optional launch-saving extras (decode kernels, M <= moe_dec_max_m()): z_f32 = fp32 [M][1536] latent z read INSTEAD of z (z may be nullptr; converted to fp16 on load, the same
// rounding as a separate fp32 -> fp16 pass); z_out_h = fp16 copy of z_out written by the reduce kernel (same rounding). Both nullptr = the original behaviour.
// slot_rec (M == 1, no CPU-resident experts): rec[slot] = the ExpertDev record of the expert chosen in that slot (written by the router kernel). The one-token kernels read it directly, so the
// counting-sort kernel and the reduce kernel are gone (2 launches instead of 4). The ws region `fill` is used as 96 tickets and must be zero before the first call (they reset themselves).
inline void moe_experts(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                        float* z_out, int M, void* ws, const float* z_f32 = nullptr, __half* z_out_h = nullptr, const ExpertDev* slot_rec = nullptr, int accum = 0) {
#if !HY4_MMA && HY4_CUDA
    if (M > HY4_MOE_DEC_MAXM) {  // no matrix-core grouped GEMM on this CUDA build: the simple path handles any M
        moe_experts_ref(s, sec4_base, experts, z, topk_idx, topk_w, z_out, M, ws, z_out_h);
        return;
    }
#endif
    const int P = M * MOE_TOPK;
    const skd::MoeWs w = skd::moe_ws_split(ws, P);
    const int eb = P < MOE_E ? P : MOE_E;
    if (slot_rec && M == 1 && !skd::moe_has_cpu() && !accum) {
        hipLaunchKernelGGL(skd::k_moe_gu_dec1, dim3(MOE_D / 8, MOE_TOPK), dim3(256), 0, s, sec4_base, slot_rec, z, z_f32, w.h2);
        hipLaunchKernelGGL(skd::k_moe_dn_dec1, dim3(MOE_D / 16, MOE_TOPK), dim3(256), 0, s, sec4_base, slot_rec, topk_w, w.h2, w.y, z_out, z_out_h, (unsigned*)w.fill);
        return;
    }
    // k_moe_reduce skips CPU experts via ex[e].pad, so w.y does not need a full memset
    hipLaunchKernelGGL(skd::k_moe_route, dim3(1), dim3(512), 0, s, experts, topk_idx, P, w.cnt, w.off, w.pairs, w.active, w.nact, w.wcnt, w.wl16, w.wl32, w.wl64, w.wl128_e, w.wl128_ty);
    if (M <= skd::moe_dec_max_m() && M <= HY4_MOE_DEC_MAXM) {
        // Experts with cnt == 1 run the fast raw-load-pipelined 1-token kernels (3 KB LDS, ~223 GB/s);
        // experts with cnt >= 2 run k_moe_*_dec<MT> on the compact wlm list (wl64 / wcnt+2). Bit-identical to M=1!
        hipLaunchKernelGGL(skd::k_moe_gu_dec1_wl, dim3(MOE_D / 8, eb), dim3(256), 0, s, sec4_base, experts, z, w.off, w.pairs, w.wl32, w.wcnt + 1, w.h2, z_f32);
        hipLaunchKernelGGL(skd::k_moe_dn_dec1_wl, dim3(MOE_D / 16, eb), dim3(256), 0, s, sec4_base, experts, topk_w, w.off, w.pairs, w.wl32, w.wcnt + 1, w.h2, w.y);
        if (M > 1) {
            const int em = (P / 2) < MOE_E ? (P / 2) : MOE_E;
#define HY4_DEC(MT)                                                                                                                                             \
    hipLaunchKernelGGL((skd::k_moe_gu_dec<MT>), dim3(MOE_D / 8, em), dim3(256), 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.wl64, w.wcnt + 2, w.h2, z_f32); \
    hipLaunchKernelGGL((skd::k_moe_dn_dec<MT>), dim3(MOE_D / 16, em), dim3(256), 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.wl64, w.wcnt + 2,   \
                       w.h2, w.y)
            if (M <= 2) { HY4_DEC(2); }
            else if (M <= 4) { HY4_DEC(4); }
            else if (M <= 8) { HY4_DEC(8); }
#if !HY4_CUDA
            else { HY4_DEC(16); }
#endif
#undef HY4_DEC
        }
    }
#if !HY4_CUDA
    else if (!HY4_MMA || hy4_is_rdna2()) {
        skd::launch_moe_rdna2(s, sec4_base, experts, z, topk_w, M, eb, w);
    }
#endif
#if HY4_MMA
    else {
        const int INF = 1 << 30;
        const dim3 b(256);
#if !HY4_CUDA
        const int i8m = skd::int8_wmma_mode();
        if (i8m == 1 || i8m == 2) {
            auto launch_i8 = [&](auto hi_tag) {
                constexpr bool HI = decltype(hi_tag)::value;
                hipLaunchKernelGGL((skd::k_moe_gu_i8_rdna4<1, 1, HI>), dim3(MOE_D / 128, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 0, 16);
                hipLaunchKernelGGL((skd::k_moe_dn_i8_rdna4<1, 1, 8, 1, HI>), dim3(MOE_D / 128, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 0, 16);
                if (M > 16) {
                    hipLaunchKernelGGL((skd::k_moe_gu_i8_rdna4<2, 1, HI>), dim3(MOE_D / 64, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 16, 32);
                    hipLaunchKernelGGL((skd::k_moe_dn_i8_rdna4<2, 1, 4, 1, HI>), dim3(MOE_D / 64, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 16, 32);
                }
                if (M > 32) {
                    hipLaunchKernelGGL((skd::k_moe_gu_i8_rdna4<4, 1, HI>), dim3(MOE_D / 32, 1, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 32, 64);
                    hipLaunchKernelGGL((skd::k_moe_dn_i8_rdna4<2, 2, 4, 1, HI>), dim3(MOE_D / 64, 1, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 32, 64);
                }
                if (M > 64) {
                    const int ty = (M + skd::GEMM_BM - 1) / skd::GEMM_BM;
                    hipLaunchKernelGGL((skd::k_moe_gu_i8_rdna4<4, 2, HI>), dim3(MOE_D / 32, ty, eb), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, 64, INF);
                    hipLaunchKernelGGL((skd::k_moe_dn_i8_rdna4<4, 2, 2, 2, HI>), dim3(MOE_D / 64, ty, eb), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.active, w.nact, w.h2, w.y, 64, INF);
                }
            };
            if (i8m == 1) launch_i8(std::false_type{});
            else launch_i8(std::true_type{});
        } else {
            const int g16 = eb;
            hipLaunchKernelGGL((skd::k_moe_gu_sg<16>), dim3(MOE_D / 64, g16), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.wl16, (const int*)nullptr, w.wcnt + 0, w.h2);
            hipLaunchKernelGGL((skd::k_moe_dn_sg<16, 128, 1, 1, 8, 1>), dim3(MOE_D / 128, g16), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.wl16, (const int*)nullptr, w.wcnt + 0, w.h2, w.y);
            if (M > 16) {
                const int g32 = P / 17 < MOE_E ? P / 17 : MOE_E;
                if (g32 > 0) {
                    hipLaunchKernelGGL((skd::k_moe_gu_sg<32>), dim3(MOE_D / 64, g32), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.wl32, (const int*)nullptr, w.wcnt + 1, w.h2);
                    hipLaunchKernelGGL((skd::k_moe_dn_sg<32, 64, 2, 1, 4, 1>), dim3(MOE_D / 64, g32), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.wl32, (const int*)nullptr, w.wcnt + 1, w.h2, w.y);
                }
            }
            if (M > 32) {
                const int g64 = P / 33 < MOE_E ? P / 33 : MOE_E;
                if (g64 > 0) {
                    hipLaunchKernelGGL((skd::k_moe_gu_sg<64>), dim3(MOE_D / 32, g64), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.wl64, (const int*)nullptr, w.wcnt + 2, w.h2);
                    hipLaunchKernelGGL((skd::k_moe_dn_sg<64, 64, 2, 2, 4, 1>), dim3(MOE_D / 64, g64), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.wl64, (const int*)nullptr, w.wcnt + 2, w.h2, w.y);
                }
            }
            if (M > 64) {
                const int g128 = moe_max128(P);
                hipLaunchKernelGGL((skd::k_moe_gu_sg<128>), dim3(MOE_D / 32, g128), b, 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.wl128_e, w.wl128_ty, w.wcnt + 3, w.h2);
                hipLaunchKernelGGL((skd::k_moe_dn_sg<128, 64, 4, 2, 2, 2>), dim3(MOE_D / 64, g128), b, 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.wl128_e, w.wl128_ty, w.wcnt + 3, w.h2, w.y);
            }
        }
#else
        // token tiles 16 / 32 / 64 / 128; all 8 waves active via warp split-K (full grid.x = 48 / 24, BK = 256)
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
#endif
    }
#endif
    hipLaunchKernelGGL(skd::k_moe_reduce, dim3(M), dim3(256), 0, s, w.y, z_out, z_out_h, experts, topk_idx, accum);
}

// simple reference path (any M); same arguments and workspace
inline void moe_experts_ref(hipStream_t s, const uint8_t* sec4_base, const ExpertDev* experts, const __half* z, const int32_t* topk_idx, const float* topk_w,
                            float* z_out, int M, void* ws, __half* z_out_h) {
    const int P = M * MOE_TOPK;
    const skd::MoeWs w = skd::moe_ws_split(ws, P);
    (void)hipMemsetAsync(w.cnt, 0, sizeof(int) * 512, s);
    (void)hipMemsetAsync(w.fill, 0, sizeof(int) * 512, s);
    hipLaunchKernelGGL(skd::k_moe_hist, dim3((P + 255) / 256), dim3(256), 0, s, experts, topk_idx, P, w.cnt);
    hipLaunchKernelGGL(skd::k_moe_scan, dim3(1), dim3(32), 0, s, w.cnt, w.off);
    hipLaunchKernelGGL(skd::k_moe_scatter, dim3((P + 255) / 256), dim3(256), 0, s, experts, topk_idx, P, w.off, w.fill, w.pairs);
    hipLaunchKernelGGL(skd::k_moe_gu_ref, dim3(MOE_D / 8, MOE_E), dim3(256), 0, s, sec4_base, experts, z, w.cnt, w.off, w.pairs, w.h2);
    hipLaunchKernelGGL(skd::k_moe_dn_ref, dim3(MOE_D / 8, MOE_E), dim3(256), 0, s, sec4_base, experts, topk_w, w.cnt, w.off, w.pairs, w.h2, w.y);
    hipLaunchKernelGGL(skd::k_moe_reduce, dim3(M), dim3(256), 0, s, w.y, z_out, z_out_h, experts, topk_idx, 0);
}

}  // namespace hy4
