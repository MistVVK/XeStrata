// src/kernels/cuda/dflash2.cu - the small kernels of the DFlash2 drafter.  See include/strata/kernels/dflash2.hpp.
#include "strata/kernels/dflash2.hpp"

#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HD = kDf2Hd;
int g_gemm_variant = 0;   // bench only (df2_gemm_variant)
int g_attn_variant = 0;   // bench only (df2_attention_variant)

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dflash2 kernel %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(std::string("dflash2 kernel ") + what + ": " + cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }

// ---- the grouped dynamic 2-tap convolution over the block's rows
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_stride, const float* __restrict__ base,
                            const float* __restrict__ resid, float* __restrict__ out, int rows, int H, int group) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    const int b = i / H, c = i % H;
    const int groups = H / group;
    float acc = 0.0f;
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        if (b - o < 0) continue;
        const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
        acc = fmaf(k, x[(size_t) (b - o) * H + c], acc);
    }
    out[i] = resid ? resid[i] + acc : acc;
}

__global__ void rope_table_kernel(const int32_t* __restrict__ pos, int rows, float theta, float* __restrict__ table) {
    const int r = blockIdx.x, i = threadIdx.x;   // 64 threads
    if (r >= rows) return;
    const double inv = pow((double) theta, -(double) (2 * i) / (double) HD);
    double ang = (double) pos[r] * inv;
    const double two_pi = 6.283185307179586476925286766559;
    ang -= two_pi * rint(ang / two_pi);   // [-pi, pi]
    float s, c;
    sincosf((float) ang, &s, &c);
    table[(r * 2 + 0) * 64 + i] = c;
    table[(r * 2 + 1) * 64 + i] = s;
}

__global__ void rope_inv_kernel(float theta, double* __restrict__ inv) {
    const int i = threadIdx.x;   // 64 threads
    inv[i] = pow((double) theta, -(double) (2 * i) / (double) HD);
}

__global__ void rope_table_inv_kernel(const int32_t* __restrict__ pos, int rows, const double* __restrict__ inv_tab, float* __restrict__ table) {
    const int r = blockIdx.x, i = threadIdx.x;   // 64 threads
    if (r >= rows) return;
    const double inv = inv_tab[i];
    double ang = (double) pos[r] * inv;
    const double two_pi = 6.283185307179586476925286766559;
    ang -= two_pi * rint(ang / two_pi);   // [-pi, pi]
    float s, c;
    sincosf((float) ang, &s, &c);
    table[(r * 2 + 0) * 64 + i] = c;
    table[(r * 2 + 1) * 64 + i] = s;
}

// one block per (row, head), 128 threads: out = rope(rmsnorm(in) * gamma)
__global__ void norm_rope_kernel(const float* __restrict__ in, int in_stride, const float* __restrict__ gamma, const int32_t* __restrict__ pos,
                                 const float* __restrict__ table, float* __restrict__ out, int slot_mod, int heads, float eps) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = in[(size_t) r * in_stride + h * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    const float tot = red[0] + red[1] + red[2] + red[3];
    const float v = x * rsqrtf(tot / (float) HD + eps) * gamma[d];
    float res = v;
    if (table != nullptr) {
        y[d] = v;
        __syncthreads();
        const int i = d & 63;
        const float c = table[(r * 2 + 0) * 64 + i], s = table[(r * 2 + 1) * 64 + i];
        res = d < 64 ? v * c - y[d + 64] * s : v * c + y[d - 64] * s;
    }
    const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
    // a ring (slot_mod > 0) is head-major [head][slot][128], the block's own keys row-major [row][head][128]
    out[slot_mod > 0 ? ((size_t) h * slot_mod + slot) * HD + d : ((size_t) slot * heads + h) * HD + d] = res;
}

__global__ void copy_rows_kernel(const float* __restrict__ in, float* __restrict__ out, const int32_t* __restrict__ pos, int slot_mod, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
    out[slot_mod > 0 ? ((size_t) (c / HD) * slot_mod + slot) * HD + c % HD : (size_t) slot * cols + c] = in[i];
}

// ---- attention.  grid (8 K/V heads, kDf2MaxChunks), 128 threads.  The 32 queries (4 heads x 8 rows) of a K/V head share each key tile;
// S = Q K^T in 4 x 4 register tiles (64 threads), the online softmax per query, O += P V in 4 x 8 register tiles (128 threads).
constexpr int AS = 132;   // smem row stride of the Q / K / V tiles (float4 aligned, conflict-free for the 4 x 4 tiles)
constexpr int PS = 36;    // stride of the transposed probabilities P[key][query]
__global__ void __launch_bounds__(128) attn_part_kernel(const float* __restrict__ q, const float* __restrict__ kring, const float* __restrict__ vring,
                                                       const float* __restrict__ kblk, const float* __restrict__ vblk,
                                                       const int32_t* __restrict__ anchor_pos, int ring_slots, int window, float* __restrict__ part, int variant,
                                                       const int64_t* __restrict__ tag, const int32_t* __restrict__ epoch_ptr) {
    __shared__ __align__(16) float Qs[32 * AS];
    __shared__ __align__(16) float Ks[32 * AS];   // the K tile, then the V tile
    __shared__ __align__(16) float Ps[32 * PS];
    __shared__ float Sc[32 * 33];
    __shared__ float m_s[32], l_s[32], corr_s[32];
    __shared__ unsigned char kvalid[32];   // key t of the tile holds the context row it should (ring tags, serve: a ring left by another conversation)
    const int kvh = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x;
    const int s = *anchor_pos;
    const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
    const int nctx = s - lo;
    const int nkeys = nctx + kDf2Rows;
    const int k_begin = chunk * kDf2Chunk;
    const int64_t ep = tag != nullptr ? ((int64_t) *epoch_ptr << 32) : 0;
    float* const pbase = part + (((size_t) chunk * kDf2Kv + kvh) * 32) * 130;
    if (k_begin >= nkeys) {
        for (int i = tid; i < 32 * 130; i += 128) pbase[i] = (i % 130 == 0) ? -INFINITY : 0.0f;
        return;
    }
    const float scale = rsqrtf((float) HD);
    for (int i = tid; i < 32 * HD / 4; i += 128) {   // Q: 32 queries x 128, float4
        const int a = i / (HD / 4), d4 = i % (HD / 4);
        const int row = a >> 2, g = a & 3;
        float4 v = *reinterpret_cast<const float4*>(q + ((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d4 * 4);
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        *reinterpret_cast<float4*>(&Qs[a * AS + d4 * 4]) = v;
    }
    if (tid < 32) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }
    float O[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) O[i][j] = 0.0f;
    const int qg_pv = tid >> 4, dg_pv = tid & 15;   // PV: queries qg*4 .. +3, dims dg*8 .. +7
    const int k_end = min(k_begin + kDf2Chunk, nkeys);
    // a tile is 32 keys x 128 dims = 1024 float4: eight per thread, loaded into registers ahead of their use
    auto fetch = [&](int k0, float4 (&reg)[8], const float* ring, const float* blk) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = tid + u * 128, t = i / (HD / 4), d4 = i % (HD / 4), k = k0 + t;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < nctx) v = *reinterpret_cast<const float4*>(ring + ((size_t) kvh * ring_slots + (lo + k) % ring_slots) * HD + d4 * 4);
            else if (k < nkeys) v = *reinterpret_cast<const float4*>(blk + ((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d4 * 4);
            reg[u] = v;
        }
    };
    auto put = [&](const float4 (&reg)[8]) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = tid + u * 128, t = i / (HD / 4), d4 = i % (HD / 4);
            *reinterpret_cast<float4*>(&Ks[t * AS + d4 * 4]) = reg[u];
        }
    };
    float4 kreg[8], vreg[8];
    fetch(k_begin, kreg, kring, kblk);
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        __syncthreads();   // Ks free (the last tile's V reads done); Qs / m_s written on the first pass
        put(kreg);
        if (tid < 32) {
            const int k = k0 + tid;
            kvalid[tid] = (tag == nullptr || k >= nctx || k >= nkeys) ? 1 : (tag[(lo + k) % ring_slots] == (ep | (int64_t) (lo + k)) ? 1 : 0);
        }
        fetch(k0, vreg, vring, vblk);   // this tile's V, hidden behind the S stage and the softmax
        __syncthreads();
        if (tid < 64) {   // S tile: 4 queries x 4 keys per thread
            const int qg = tid >> 3, kg = tid & 7;
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
#pragma unroll 4
            for (int d = 0; d < ((variant & 1) ? 0 : HD); d += 4) {
                float4 qv[4], kv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) qv[i] = *reinterpret_cast<const float4*>(&Qs[(qg * 4 + i) * AS + d]);
#pragma unroll
                for (int j = 0; j < 4; ++j) kv[j] = *reinterpret_cast<const float4*>(&Ks[(kg * 4 + j) * AS + d]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[i][j] += qv[i].x * kv[j].x + qv[i].y * kv[j].y + qv[i].z * kv[j].z + qv[i].w * kv[j].w;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int qi = qg * 4 + i, t = kg * 4 + j, k = k0 + t, row = qi >> 2;
                    bool ok = k < nkeys;
                    if (ok && k < nctx) ok = (lo + k) > (s + row - window) && kvalid[t];   // s + row - j < window, and the slot holds that row
                    Sc[qi * 33 + t] = ok ? acc[i][j] : -INFINITY;
                }
        }
        __syncthreads();
        {   // online softmax: query = tid / 4, 8 keys each, 4 lanes per query
            const int qi = tid >> 2, sub = tid & 3;
            float v[8], tmax = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) { v[u] = Sc[qi * 33 + sub * 8 + u]; tmax = fmaxf(tmax, v[u]); }
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float m_old = m_s[qi];
            const float mn = fmaxf(m_old, tmax);
            const float corr = (m_old == -INFINITY) ? 0.0f : __expf(m_old - mn);
            float psum = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float p = v[u] == -INFINITY ? 0.0f : __expf(v[u] - mn);
                Ps[(sub * 8 + u) * PS + qi] = p;
                psum += p;
            }
            psum += __shfl_xor_sync(0xffffffffu, psum, 1);
            psum += __shfl_xor_sync(0xffffffffu, psum, 2);
            if (sub == 0) { l_s[qi] = l_s[qi] * corr + psum; m_s[qi] = mn; corr_s[qi] = corr; }
        }
        __syncthreads();   // Ps, corr_s complete; every read of Ks (the S tile) done
        put(vreg);
        if (k0 + 32 < k_end) fetch(k0 + 32, kreg, kring, kblk);   // the next tile's K, hidden behind the PV stage
        __syncthreads();
        {
            const float4 c4 = *reinterpret_cast<const float4*>(&corr_s[qg_pv * 4]);
            const float cr[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) O[i][j] *= cr[i];
#pragma unroll 4
            for (int t = 0; t < ((variant & 2) ? 0 : 32); ++t) {
                const float4 p4 = *reinterpret_cast<const float4*>(&Ps[t * PS + qg_pv * 4]);
                const float4 va = *reinterpret_cast<const float4*>(&Ks[t * AS + dg_pv * 8]);
                const float4 vb = *reinterpret_cast<const float4*>(&Ks[t * AS + dg_pv * 8 + 4]);
                const float pv[4] = {p4.x, p4.y, p4.z, p4.w};
                const float vv[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) O[i][j] = fmaf(pv[i], vv[j], O[i][j]);
            }
        }
    }
    __syncthreads();
    if (tid < 32) { pbase[tid * 130] = m_s[tid]; pbase[tid * 130 + 1] = l_s[tid]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float* o = pbase + (size_t) (qg_pv * 4 + i) * 130 + 2 + dg_pv * 8;
        *reinterpret_cast<float2*>(o) = make_float2(O[i][0], O[i][1]);
        *reinterpret_cast<float2*>(o + 2) = make_float2(O[i][2], O[i][3]);
        *reinterpret_cast<float2*>(o + 4) = make_float2(O[i][4], O[i][5]);
        *reinterpret_cast<float2*>(o + 6) = make_float2(O[i][6], O[i][7]);
    }
}

__device__ __forceinline__ void quant_block(float v, int blk, int col, int K, int8_t* __restrict__ xq, float* __restrict__ xd);

__global__ void attn_merge_kernel(const float* __restrict__ part, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int a = blockIdx.x;                      // kvh * 32 + qi
    const int kvh = a >> 5, qi = a & 31, d = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < kDf2MaxChunks; ++c) M = fmaxf(M, part[(((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < kDf2MaxChunks; ++c) {
        const float* p = part + (((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130;
        const float mc = p[0];
        const float w = (mc == -INFINITY) ? 0.0f : __expf(mc - M);
        L += p[1] * w;
        o = fmaf(p[2 + d], w, o);
    }
    const int row = qi >> 2, g = qi & 3;
    // the output row `row`, head kvh * 4 + g, dims d: quant block (kvh * 4 + g) * 4 + d / 32 of the 4096-wide row
    quant_block(L > 0.0f ? o / L : 0.0f, (kvh * kDf2Gqa + g) * (HD / 32) + (d >> 5), row, 32 * HD, xq, xd);
}

// ---- Q8 GEMM for up to 8 activation columns (the drafter's matrices)
//
// Weights are TILED (df2_q8_tile makes them from GGUF Q8_0 blocks): a group of 8 rows is one unit; per 512 values of the reduction axis (a "step",
// 16 Q8 blocks) the group stores the 8 rows' 512 int8 one after the other (4 KB), so a warp that works on a group reads 4 KB contiguous per step
// (lane = block * 2 + half: one 16-byte load per row).  The fp16 block scales follow all the int8 of the matrix, in the same tile order (8 rows x 16
// blocks = 256 bytes a step).  Activations: x_q (int8 [col][n_in]) and x_d (float [col][n_in / 32]); a column's 512 bytes of a step are contiguous.
struct Df2Seg { const int8_t* q; const __half* d; float* y; int n_out; int row0; };
struct Df2Segs { Df2Seg s[3]; int n; };

// The matrix a warp's rows belong to.  Constant indices only: a loop over segs.s[si] makes the by-value parameter struct live in local memory (STACK 104 in the spill guard).
__device__ __forceinline__ Df2Seg df2_pick(const Df2Segs& segs, int row_g) {
    Df2Seg sg = segs.s[0];
    if (segs.n > 1 && row_g >= segs.s[1].row0) sg = segs.s[1];
    if (segs.n > 2 && row_g >= segs.s[2].row0) sg = segs.s[2];
    return sg;
}

template <int WARPS, bool COMPUTE, int UNR = 2>
__global__ void __launch_bounds__(WARPS * 32) gemm8_kernel(Df2Segs segs, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n_in, int ncols) {
    constexpr int R = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_g = (blockIdx.x * WARPS + warp) * R;   // first row of this warp's group, over all the launch's matrices
    const Df2Seg sg = df2_pick(segs, row_g);
    const int row0 = row_g - sg.row0;
    const int nb = n_in / 32, steps = nb / 16;
    const int4* wq = reinterpret_cast<const int4*>(sg.q + (size_t) row0 * n_in) + lane;
    const __half* wd = sg.d + (size_t) row0 * nb + (lane >> 1);
    const int4* xv = reinterpret_cast<const int4*>(xq) + lane;
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
    for (int st = 0; st < steps; ++st) {
        int4 w[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[r] = wq[(size_t) st * 256 + r * 32];
            dw[r] = __half2float(wd[(size_t) st * 128 + r * 16]);
        }
        if (!COMPUTE) {   // bench only: the same loads, almost no arithmetic
            for (int r = 0; r < R; ++r) acc[r][0] += (float) (w[r].x ^ w[r].y ^ w[r].z ^ w[r].w) * dw[r];
            continue;
        }
        float xs[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int isum = STRATA_DP4A(w[r].x, x.x, 0);
                isum = STRATA_DP4A(w[r].y, x.y, isum);
                isum = STRATA_DP4A(w[r].z, x.z, isum);
                isum = STRATA_DP4A(w[r].w, x.w, isum);
                acc[r][c] = fmaf(dw[r] * xs[c], (float) isum, acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float v = acc[r][c];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (lane == 0 && c < ncols) sg.y[(size_t) c * sg.n_out + row0 + r] = v;
        }
}

// The same GEMM with R rows a warp instead of 8 (R = 4, 2, 1): a group of 8 rows is shared by 8 / R warps, each reading only its rows' weights.  Every output keeps
// the arithmetic of gemm8_kernel (lane l's partial sums run over the same steps in the same order, the same shuffle tree), so the result is bitwise the same;
// what changes is the number of warps in flight: the small matrices (fc 2560 rows = 320 groups, the 640-row convolution kernels) fill only a third of the
// card with one warp per group, and a warp that waits for its 4 KB step per row stalls the group.
template <int WARPS, int R, int UNR>
__global__ void __launch_bounds__(WARPS * 32) gemm_rows_kernel(Df2Segs segs, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n_in, int ncols) {
    constexpr int SUBS = 8 / R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wi = blockIdx.x * WARPS + warp;
    const int row_g = (wi / SUBS) * 8, sub = wi % SUBS;
    const Df2Seg sg = df2_pick(segs, row_g);
    const int row0 = row_g - sg.row0;
    const int nb = n_in / 32, steps = nb / 16;
    const int4* wq = reinterpret_cast<const int4*>(sg.q + (size_t) row0 * n_in) + lane + sub * R * 32;
    const __half* wd = sg.d + (size_t) row0 * nb + (lane >> 1) + sub * R * 16;
    const int4* xv = reinterpret_cast<const int4*>(xq) + lane;
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
    for (int st = 0; st < steps; ++st) {
        int4 w[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[r] = wq[(size_t) st * 256 + r * 32];
            dw[r] = __half2float(wd[(size_t) st * 128 + r * 16]);
        }
        float xs[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int isum = STRATA_DP4A(w[r].x, x.x, 0);
                isum = STRATA_DP4A(w[r].y, x.y, isum);
                isum = STRATA_DP4A(w[r].z, x.z, isum);
                isum = STRATA_DP4A(w[r].w, x.w, isum);
                acc[r][c] = fmaf(dw[r] * xs[c], (float) isum, acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float v = acc[r][c];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (lane == 0 && c < ncols) sg.y[(size_t) c * sg.n_out + row0 + sub * R + r] = v;
        }
}

// ---- Q4_0 variant of the same GEMM (opt-in drafter pack, dtype "q4_0"; a weight is 4 bits, the activations stay int8)
//
// Same tiling as the Q8 plane: a group of 8 rows is a unit and a step is 512 values of the reduction axis.  A row's 512 values are 256 bytes in the
// step (2 KB for the group), lane = block * 2 + half takes 8 bytes = 16 values as two words.  Word 0 holds values 0-3 in its four low nibbles and
// 4-7 in its four high nibbles, word 1 holds 8-11 and 12-15, so (w & 0x0F0F0F0F) and ((w >> 4) & 0x0F0F0F0F) are the int8 words dp4a wants, as
// unsigned q = 0..15 (value = (q - 8) * d, ggml Q4_0).  sum((q - 8) x) = sum(q x) - 8 sum(x): the sum of the lane's 16 activations is one more dp4a
// per word and column, shared by the 8 rows.  The fp16 block scales follow the nibble plane (n_out * n_in / 2 bytes) in the Q8 plane's order.
template <int WARPS, int UNR = 2>
__global__ void __launch_bounds__(WARPS * 32) gemm4_kernel(Df2Segs segs, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n_in, int ncols) {
    constexpr int R = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_g = (blockIdx.x * WARPS + warp) * R;
    const Df2Seg sg = df2_pick(segs, row_g);
    const int row0 = row_g - sg.row0;
    const int nb = n_in / 32, steps = nb / 16;
    const uint2* wq = reinterpret_cast<const uint2*>(sg.q + (size_t) row0 * (n_in / 2)) + lane;
    const __half* wd = sg.d + (size_t) row0 * nb + (lane >> 1);
    const int4* xv = reinterpret_cast<const int4*>(xq) + lane;
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
    for (int st = 0; st < steps; ++st) {
        int4 w[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 raw = wq[(size_t) st * 256 + r * 32];   // 8 rows x 32 lanes x 8 bytes = 2 KB a step
            w[r] = make_int4((int) (raw.x & 0x0F0F0F0Fu), (int) ((raw.x >> 4) & 0x0F0F0F0Fu), (int) (raw.y & 0x0F0F0F0Fu), (int) ((raw.y >> 4) & 0x0F0F0F0Fu));
            dw[r] = __half2float(wd[(size_t) st * 128 + r * 16]);
        }
        float xs[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
            int xsum = STRATA_DP4A(0x01010101, x.x, 0);
            xsum = STRATA_DP4A(0x01010101, x.y, xsum);
            xsum = STRATA_DP4A(0x01010101, x.z, xsum);
            xsum = STRATA_DP4A(0x01010101, x.w, xsum);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int isum = STRATA_DP4A(w[r].x, x.x, 0);
                isum = STRATA_DP4A(w[r].y, x.y, isum);
                isum = STRATA_DP4A(w[r].z, x.z, isum);
                isum = STRATA_DP4A(w[r].w, x.w, isum);
                acc[r][c] = fmaf(dw[r] * xs[c], (float) (isum - 8 * xsum), acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float v = acc[r][c];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (lane == 0 && c < ncols) sg.y[(size_t) c * sg.n_out + row0 + r] = v;
        }
}

// The Q4_0 GEMM with R rows a warp (R = 4, 2, 1) instead of 8, as gemm_rows_kernel is to gemm8_kernel: a group of 8 rows is shared by 8 / R warps, each reading only its
// rows' nibbles and scales; every output keeps the arithmetic of gemm4_kernel (same steps in the same order per lane, the same xsum and shuffle tree), so the result is bitwise
// the same.  The activations' per-column sum is recomputed by each of the group's warps (4 dp4a a column and step).
template <int WARPS, int R, int UNR>
__global__ void __launch_bounds__(WARPS * 32) gemm4_rows_kernel(Df2Segs segs, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n_in, int ncols) {
    constexpr int SUBS = 8 / R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wi = blockIdx.x * WARPS + warp;
    const int row_g = (wi / SUBS) * 8, sub = wi % SUBS;
    const Df2Seg sg = df2_pick(segs, row_g);
    const int row0 = row_g - sg.row0;
    const int nb = n_in / 32, steps = nb / 16;
    const uint2* wq = reinterpret_cast<const uint2*>(sg.q + (size_t) row0 * (n_in / 2)) + lane + sub * R * 32;
    const __half* wd = sg.d + (size_t) row0 * nb + (lane >> 1) + sub * R * 16;
    const int4* xv = reinterpret_cast<const int4*>(xq) + lane;
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
    for (int st = 0; st < steps; ++st) {
        int4 w[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 raw = wq[(size_t) st * 256 + r * 32];
            w[r] = make_int4((int) (raw.x & 0x0F0F0F0Fu), (int) ((raw.x >> 4) & 0x0F0F0F0Fu), (int) (raw.y & 0x0F0F0F0Fu), (int) ((raw.y >> 4) & 0x0F0F0F0Fu));
            dw[r] = __half2float(wd[(size_t) st * 128 + r * 16]);
        }
        float xs[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
            int xsum = STRATA_DP4A(0x01010101, x.x, 0);
            xsum = STRATA_DP4A(0x01010101, x.y, xsum);
            xsum = STRATA_DP4A(0x01010101, x.z, xsum);
            xsum = STRATA_DP4A(0x01010101, x.w, xsum);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int isum = STRATA_DP4A(w[r].x, x.x, 0);
                isum = STRATA_DP4A(w[r].y, x.y, isum);
                isum = STRATA_DP4A(w[r].z, x.z, isum);
                isum = STRATA_DP4A(w[r].w, x.w, isum);
                acc[r][c] = fmaf(dw[r] * xs[c], (float) (isum - 8 * xsum), acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float v = acc[r][c];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (lane == 0 && c < ncols) sg.y[(size_t) c * sg.n_out + row0 + sub * R + r] = v;
        }
}

// quantize one 32-block of one row to the planar activation layout (a warp: lane = element)
__device__ __forceinline__ void quant_block(float v, int blk, int col, int K, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int lane = threadIdx.x & 31;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q = __float2int_rn(v * inv);
    xq[(size_t) col * K + blk * 32 + lane] = (int8_t) q;
    if (lane == 0) xd[(size_t) col * (K / 32) + blk] = d;
}

// rows [ncols][K] -> planar; grid (ceil(K / 32 / 8), ncols), 256 threads (a warp per block)
__global__ void quant_rows_kernel(const float* __restrict__ x, int K, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int blk = blockIdx.x * 8 + (threadIdx.x >> 5), col = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= K) return;
    quant_block(x[(size_t) col * K + blk * 32 + lane], blk, col, K, xq, xd);
}

// RMS norm of row `col` (N values) then the planar quantization of the normalized values; also writes them when xn != null.  One block of 256 per row.
__global__ void rms_quant_kernel(const float* __restrict__ h, const float* __restrict__ gamma, float* __restrict__ xn, int N, float eps,
                                 int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int col = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float red[8];
    float ss = 0.0f;
    for (int i = tid; i < N; i += 256) { const float v = h[(size_t) col * N + i]; ss += v * v; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < 8; ++w) tot += red[w];
    const float scale = rsqrtf(tot / (float) N + eps);
    for (int blk = warp; blk * 32 < N; blk += 8) {
        const int i = blk * 32 + lane;
        const float v = h[(size_t) col * N + i] * scale * gamma[i];
        if (xn) xn[(size_t) col * N + i] = v;
        quant_block(v, blk, col, N, xq, xd);
    }
}

// conv (as conv_kernel, no residual) then the planar quantization: grid (ceil(H / 32 / 8), rows), 256 threads
__global__ void conv_quant_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_stride, const float* __restrict__ base, int H, int group,
                                  int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int blk = blockIdx.x * 8 + (threadIdx.x >> 5), b = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= H) return;
    const int c = blk * 32 + lane, groups = H / group;
    float acc = 0.0f;
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        if (b - o < 0) continue;
        const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
        acc = fmaf(k, x[(size_t) (b - o) * H + c], acc);
    }
    quant_block(acc, blk, b, H, xq, xd);
}

// h += conv(x) (the finish convolution with its residual), then RMS norm of the new h with `gamma` -> xn (optional) and the planar quantization.
// One block of 1024 threads per row: a warp owns whole 32-blocks (80 of them for 2560), so the quantization needs no exchange.
__global__ void __launch_bounds__(1024) conv_rms_quant_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_stride, const float* __restrict__ base,
                                                              float* __restrict__ h, const float* __restrict__ gamma, float* __restrict__ xn, int H, int group,
                                                              float eps, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int groups = H / group, nb = H / 32;
    __shared__ float red[32];
    float v[3];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int blk = warp + j * 32;
        v[j] = 0.0f;
        if (blk < nb) {
            const int c = blk * 32 + lane;
            float acc = 0.0f;
#pragma unroll
            for (int o = 0; o < 2; ++o) {
                if (b - o < 0) continue;
                const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
                acc = fmaf(k, x[(size_t) (b - o) * H + c], acc);
            }
            v[j] = h[(size_t) b * H + c] + acc;
            h[(size_t) b * H + c] = v[j];
            ss += v[j] * v[j];
        }
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < 32; ++w) tot += red[w];
    const float scale = rsqrtf(tot / (float) H + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int blk = warp + j * 32;
        if (blk < nb) {
            const int c = blk * 32 + lane;
            const float y = v[j] * scale * gamma[c];
            if (xn) xn[(size_t) b * H + c] = y;
            quant_block(y, blk, b, H, xq, xd);
        }
    }
}

// q and k of the block: per (row, head) RMS norm and rope; blockIdx.y < q_heads is a q head (-> qo, row-major rows), the rest k heads (-> ko)
__global__ void norm_rope_qk_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ gq, const float* __restrict__ gk,
                                    const float* __restrict__ table, float* __restrict__ qo, float* __restrict__ ko, int q_heads, int k_heads, float eps) {
    const int r = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const bool isq = hh < q_heads;
    const int h = isq ? hh : hh - q_heads, heads = isq ? q_heads : k_heads;
    const float* in = isq ? q : k;
    const float* gamma = isq ? gq : gk;
    float* out = isq ? qo : ko;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = in[((size_t) r * heads + h) * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    const float v = x * rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float) HD + eps) * gamma[d];
    y[d] = v;
    __syncthreads();
    const int i = d & 63;
    const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
    out[((size_t) r * heads + h) * HD + d] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
}

// the context update's ring write: K norm + rope of (row, head) and the V copy, both into the head-major ring slot pos % slots
__global__ void kv_ring_kernel(const float* __restrict__ kin, const float* __restrict__ vin, const float* __restrict__ gk, const int32_t* __restrict__ pos,
                               const float* __restrict__ table, float* __restrict__ kring, float* __restrict__ vring, int slots, int heads, float eps,
                               int64_t* __restrict__ tag, const int32_t* __restrict__ epoch_ptr) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = kin[((size_t) r * heads + h) * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    const float v = x * rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float) HD + eps) * gk[d];
    y[d] = v;
    __syncthreads();
    const int i = d & 63;
    const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
    const int slot = (int) (pos[r] % slots);
    const size_t at = ((size_t) h * slots + slot) * HD + d;
    kring[at] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
    vring[at] = vin[((size_t) r * heads + h) * HD + d];
    if (tag != nullptr && h == 0 && d == 0) tag[slot] = ((int64_t) *epoch_ptr << 32) | (int64_t) pos[r];
}

// silu(gate) * up then the planar quantization: grid (ceil(I / 32 / 8), rows), 256 threads
__global__ void swiglu_quant_kernel(const float* __restrict__ gate, const float* __restrict__ up, int I, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int blk = blockIdx.x * 8 + (threadIdx.x >> 5), b = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 32 >= I) return;
    const size_t i = (size_t) b * I + blk * 32 + lane;
    const float g = gate[i];
    quant_block(g / (1.0f + __expf(-g)) * up[i], blk, b, I, xq, xd);
}

// GGUF Q8_0 blocks (34 bytes: fp16 d, 32 int8) of a [n_out][n_in] matrix -> the tiled qs plane (n_out * n_in bytes) and the tiled d plane; one thread per block
__global__ void q8_tile_kernel(const uint8_t* __restrict__ blocks, int8_t* __restrict__ qs, __half* __restrict__ dpl, int n_in, int64_t count) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count) return;
    const int nb = n_in / 32;
    const int64_t row = b / nb;
    const int blk = (int) (b % nb);
    const int64_t g = row / 8;
    const int r = (int) (row % 8), st = blk / 16, bi = blk % 16;
    const uint8_t* src = blocks + b * 34;
    const uint16_t dh = (uint16_t) src[0] | ((uint16_t) src[1] << 8);
    dpl[(g * 8 * nb) + (int64_t) st * 128 + r * 16 + bi] = __ushort_as_half(dh);
    int8_t* o = qs + g * 8 * n_in + (int64_t) st * 4096 + r * 512 + bi * 32;
    for (int i = 0; i < 32; ++i) o[i] = (int8_t) src[2 + i];
}

// ggml Q4_0 blocks (18 bytes: fp16 d, 16 bytes with value i in the low nibble of byte i and value i + 16 in the high nibble) of a [n_out][n_in] matrix -> the
// nibble plane (n_out * n_in / 2 bytes, the layout above) and the tiled d plane; one thread per block
__global__ void q4_tile_kernel(const uint8_t* __restrict__ blocks, uint8_t* __restrict__ qs, __half* __restrict__ dpl, int n_in, int64_t count) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count) return;
    const int nb = n_in / 32;
    const int64_t row = b / nb;
    const int blk = (int) (b % nb);
    const int64_t g = row / 8;
    const int r = (int) (row % 8), st = blk / 16, bi = blk % 16;
    const uint8_t* src = blocks + b * 18;
    const uint16_t dh = (uint16_t) src[0] | ((uint16_t) src[1] << 8);
    dpl[(g * 8 * nb) + (int64_t) st * 128 + r * 16 + bi] = __ushort_as_half(dh);
    for (int half = 0; half < 2; ++half) {   // values half * 16 .. +15 of the block
        uint32_t wd[2];
        for (int w = 0; w < 2; ++w) {   // word w holds values w * 8 .. w * 8 + 7 of the half: byte p = (value w*8+p, value w*8+4+p) as (low, high) nibble
            uint32_t v = 0;
            for (int p = 0; p < 4; ++p) {
                const uint8_t b_lo = src[2 + w * 8 + p], b_hi = src[2 + w * 8 + 4 + p];
                const uint32_t q_lo = half == 0 ? (b_lo & 15u) : (b_lo >> 4), q_hi = half == 0 ? (b_hi & 15u) : (b_hi >> 4);
                v |= (q_lo | (q_hi << 4)) << (p * 8);
            }
            wd[w] = v;
        }
        uint8_t* o = qs + g * 8 * (n_in / 2) + (int64_t) st * 2048 + r * 256 + (bi * 2 + half) * 8;
        *reinterpret_cast<uint2*>(o) = make_uint2(wd[0], wd[1]);
    }
}

// ---- top-16
constexpr int TK_CHUNK_MAX = 4096;
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

__global__ void topk_part_kernel(const float* __restrict__ logits, int n, float* __restrict__ cval, int32_t* __restrict__ cidx) {
    __shared__ float vals[TK_CHUNK_MAX];
    __shared__ float wv[8];
    __shared__ int wi[8];
    __shared__ int win;
    const int row = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int cs = (n + kDf2TopBlocks - 1) / kDf2TopBlocks;
    const int c0 = blk * cs, c1 = min(c0 + cs, n);
    for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[(size_t) row * n + c0 + i] : -INFINITY;
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < cs; i += 256)
            if (better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
            const size_t o = ((size_t) row * kDf2TopBlocks + blk) * kDf2TopK + k;
            cval[o] = v;
            cidx[o] = v == -INFINITY ? -1 : c0 + ix;
            win = ix;
            if (v != -INFINITY) vals[ix] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void topk_merge_kernel(const float* __restrict__ cval, const int32_t* __restrict__ cidx, const int32_t* __restrict__ idx_map,
                                  float* __restrict__ val, int32_t* __restrict__ tok) {
    constexpr int NC = kDf2TopBlocks * kDf2TopK;
    __shared__ float sv[NC];
    __shared__ int si[NC];
    __shared__ float wv[8];
    __shared__ int wi[8], wp[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < NC; i += 256) { sv[i] = cval[(size_t) row * NC + i]; si[i] = cidx[(size_t) row * NC + i]; }
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for (int i = tid; i < NC; i += 256)
            if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; wp[tid >> 5] = bp; }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0], p = wp[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
            if (p >= 0) { si[p] = -1; }
            val[row * kDf2TopK + k] = v;
            tok[row * kDf2TopK + k] = p >= 0 ? (idx_map ? idx_map[ix] : ix) : 0;
        }
        __syncthreads();
    }
}

// one value / eight values of a ggml Q4_0 row of 256 (8 blocks of 18 bytes: fp16 d, then 16 bytes with value i in the low nibble of byte i and value i + 16 in the high one)
__device__ __forceinline__ float q4row_at(const uint8_t* row, int i) {
    const uint8_t* blk = row + (size_t) (i >> 5) * 18;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
    const int j = i & 31, byte = blk[2 + (j & 15)];
    return (float) ((j < 16 ? (byte & 15) : (byte >> 4)) - 8) * d;
}

// ---- the selector's walk: one block of 256 threads
template <bool Q4>
__global__ void select_kernel(const float* __restrict__ hp, const float* __restrict__ unary, const int32_t* __restrict__ tok,
                              const uint16_t* __restrict__ pred_cb, const uint16_t* __restrict__ succ_cb, int rank, int rows,
                              const int32_t* __restrict__ anchor_tok, int32_t* __restrict__ out_tok, float* __restrict__ out_prob) {
    __shared__ float vp[256];
    __shared__ float sc[kDf2TopK];
    __shared__ int prev;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    if (tid == 0) prev = *anchor_tok;
    __syncthreads();
    for (int i = 0; i < rows; ++i) {
        if (Q4) vp[tid] = q4row_at(reinterpret_cast<const uint8_t*>(pred_cb) + (size_t) prev * (rank / 32 * 18), tid) * hp[(size_t) i * rank + tid];
        else vp[tid] = bf16f(pred_cb[(size_t) prev * rank + tid]) * hp[(size_t) i * rank + tid];
        __syncthreads();
        for (int j = w * 2; j < w * 2 + 2; ++j) {
            const int t = tok[i * kDf2TopK + j];
            float acc = 0.0f;
            if (Q4) {
                const uint8_t* row = reinterpret_cast<const uint8_t*>(succ_cb) + (size_t) t * (rank / 32 * 18);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc = fmaf(q4row_at(row, lane * 8 + e), vp[lane * 8 + e], acc);
            } else {
                const uint4 raw = *reinterpret_cast<const uint4*>(succ_cb + (size_t) t * rank + lane * 8);
                const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc = fmaf(__uint_as_float(u[e] << 16), vp[lane * 8 + e * 2], acc);
                    acc = fmaf(__uint_as_float(u[e] & 0xffff0000u), vp[lane * 8 + e * 2 + 1], acc);
                }
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) sc[j] = unary[i * kDf2TopK + j] + acc;
        }
        __syncthreads();
        if (tid == 0) {
            int bj = 0;
            float bv = sc[0];
            for (int j = 1; j < kDf2TopK; ++j)
                if (sc[j] > bv) { bv = sc[j]; bj = j; }
            float z = 0.0f;
            for (int j = 0; j < kDf2TopK; ++j) z += __expf(sc[j] - bv);
            out_tok[i] = tok[i * kDf2TopK + bj];
            out_prob[i] = 1.0f / z;
            prev = out_tok[i];
        }
        __syncthreads();
    }
}

__global__ void tap_rows_kernel(const float* __restrict__ R, float* __restrict__ out, int64_t out_stride, int64_t n, int hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= n) return;
    float s = 0.0f;
    for (int c = 0; c < hc; ++c) s += R[((size_t) r * hc + c) * n + i];
    out[(size_t) r * out_stride + i] = s / (float) hc;
}

__global__ void tap_gr_kernel(const float* __restrict__ R, int64_t R_stride, const float* __restrict__ bo, int64_t bo_stride,
                              const float* __restrict__ inject, int64_t inj_stride, float* __restrict__ out, int64_t out_stride, int64_t n, int hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= n) return;
    float wsum = 0.0f, rs = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const float x = inject[(size_t) r * inj_stride + c] / (float) hc;
        wsum += 2.0f / (1.0f + expf(-x));
        rs += R[(size_t) r * R_stride + (size_t) c * n + i];
    }
    out[(size_t) r * out_stride + i] = (rs + bo[(size_t) r * bo_stride + i] * wsum) / (float) hc;
}

}  // namespace

void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject, int64_t inj_stride, float* out, int64_t out_stride,
                int rows, int64_t n_embd, int hc, void* stream) {
    tap_gr_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t) stream>>>(R, R_stride, bo, bo_stride, inject, inj_stride, out,
                                                                                                  out_stride, n_embd, hc);
    check("tap_gr");
}

void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out, int rows, int H, int group,
              void* stream) {
    const int n = rows * H;
    conv_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, dyn, dyn_stride, base, resid, out, rows, H, group);
    check("conv");
}

void df2_rope_table(const int32_t* pos, int rows, float theta, float* table, void* stream) {
    rope_table_kernel<<<rows, 64, 0, (cudaStream_t) stream>>>(pos, rows, theta, table);
    check("rope_table");
}

void df2_norm_rope(const float* in, int in_stride, const float* gamma, const int32_t* pos, const float* rope_table, float* out, int slot_mod,
                   int rows, int heads, float eps, void* stream) {
    norm_rope_kernel<<<dim3(rows, heads), HD, 0, (cudaStream_t) stream>>>(in, in_stride, gamma, pos, rope_table, out, slot_mod, heads, eps);
    check("norm_rope");
}

void df2_copy_rows(const float* in, float* out, const int32_t* pos, int slot_mod, int rows, int cols, void* stream) {
    const int n = rows * cols;
    copy_rows_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(in, out, pos, slot_mod, rows, cols);
    check("copy_rows");
}

void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk, const int32_t* anchor_pos,
                   int ring_slots, int window, float* part, void* xq, float* xd, void* stream, const int64_t* tag, const int32_t* epoch) {
    attn_part_kernel<<<dim3(kDf2Kv, kDf2MaxChunks), 128, 0, (cudaStream_t) stream>>>(q, kring, vring, kblk, vblk, anchor_pos, ring_slots, window, part, g_attn_variant, tag, epoch);
    check("attn_part");
    attn_merge_kernel<<<kDf2Kv * 32, HD, 0, (cudaStream_t) stream>>>(part, static_cast<int8_t*>(xq), xd);
    check("attn_merge");
}

void df2_conv_rms_quant(const float* x, const float* dyn, int dyn_stride, const float* base, float* h, const float* gamma, float* xn, int rows, int H, int group,
                        float eps, void* xq, float* xd, void* stream) {
    if (H > 3 * 1024) throw std::invalid_argument("df2_conv_rms_quant: H too large");
    conv_rms_quant_kernel<<<rows, 1024, 0, (cudaStream_t) stream>>>(x, dyn, dyn_stride, base, h, gamma, xn, H, group, eps, static_cast<int8_t*>(xq), xd);
    check("conv_rms_quant");
}

void df2_norm_rope_qk(const float* q, const float* k, const float* gq, const float* gk, const float* table, float* qo, float* ko, int rows, int q_heads, int k_heads,
                      float eps, void* stream) {
    norm_rope_qk_kernel<<<dim3(rows, q_heads + k_heads), HD, 0, (cudaStream_t) stream>>>(q, k, gq, gk, table, qo, ko, q_heads, k_heads, eps);
    check("norm_rope_qk");
}

void df2_kv_ring(const float* kin, const float* vin, const float* gk, const int32_t* pos, const float* table, float* kring, float* vring, int slots, int rows,
                 int heads, float eps, void* stream, int64_t* tag, const int32_t* epoch) {
    kv_ring_kernel<<<dim3(rows, heads), HD, 0, (cudaStream_t) stream>>>(kin, vin, gk, pos, table, kring, vring, slots, heads, eps, tag, epoch);
    check("kv_ring");
}

void df2_rope_inv(float theta, double* inv, void* stream) {
    rope_inv_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(theta, inv);
    check("rope_inv");
}

void df2_rope_table_inv(const int32_t* pos, int rows, const double* inv, float* table, void* stream) {
    rope_table_inv_kernel<<<rows, 64, 0, (cudaStream_t) stream>>>(pos, rows, inv, table);
    check("rope_table_inv");
}

void df2_gemm_variant(int v) { g_gemm_variant = v; }
void df2_attention_variant(int v) { g_attn_variant = v; }

// The Q4_0 variant by matrix size (rows = all the launch's output rows).  17 (two warps a block, two rows a warp, unroll 2): dflash2_gemm_bench q4 best on l620 (sm_120,
// 110 SMs) for every matrix, within 0-2 us of the best on a 188-SM RTX PRO 6000.  gfx12 (R9700, gfx1201, p620, 8 columns, cold weights, two bench runs): the same shape with unroll 1
// (26) is 25-45% faster on every matrix of 1000 rows and more (gate+up 56 -> 44.5 us, down 39 -> 23.4, qkv 26.8 -> 21.4, o 23.6 -> 14.9, fc 61.5 -> 35.2, kv 15.2 -> 10.0), and one
// warp a row with unroll 1 (31) is 2x faster on the 256 and 640 row matrices (akp / mkp 12.1 -> 6.3 us, selector 11.1 -> 5.2); all bitwise the same results.
static int df2_q4_auto(int rows) {
#if defined(STRATA_USE_HIP)
    static const bool gfx12 = [] {
        int dev = 0;
        cudaDeviceProp p;
        return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
    }();
    if (gfx12) return rows <= 640 ? 31 : 26;
#endif
    (void) rows;
    return 17;
}
#ifndef DF2_Q4_AUTO
#define DF2_Q4_AUTO(rows) df2_q4_auto(rows)
#endif
void df2_gemm(const Df2Mat* mats, int n, const void* xq, const float* xd, int n_in, int ncols, void* stream) {
    if (n < 1 || n > 3 || n_in % 512 != 0 || ncols < 1 || ncols > 8) throw std::invalid_argument("df2_gemm: unsupported shape");
    Df2Segs segs;
    int rows = 0;
    for (int i = 0; i < n; ++i) {
        if (mats[i].n_out % 16 != 0) throw std::invalid_argument("df2_gemm: n_out must be a multiple of 16");
        if (mats[i].q4 != mats[0].q4) throw std::invalid_argument("df2_gemm: one launch, one weight format");
        const int8_t* q = static_cast<const int8_t*>(mats[i].w);
        segs.s[i] = Df2Seg{q, reinterpret_cast<const __half*>(q + (size_t) mats[i].n_out * n_in / (mats[i].q4 ? 2 : 1)), mats[i].y, mats[i].n_out, rows};
        rows += mats[i].n_out;
    }
    segs.n = n;
    const int8_t* xqp = static_cast<const int8_t*>(xq);
    cudaStream_t st = (cudaStream_t) stream;
    if (mats[0].q4) {   // the Q4_0 pack: 4 warps a block when the rows allow, else 2; the rows-a-warp variants (bitwise the same results) by matrix size
        static const int tune4 = [] { const char* e = std::getenv("STRATA_DF2_GEMM4"); return e ? std::atoi(e) : 0; }();   // 0: the table; 1: 8 rows a warp as before; 10 and up: that variant for every matrix (bench / A/B)
        int v4 = g_gemm_variant != 0 ? g_gemm_variant : tune4 >= 10 ? tune4 : tune4 != 0 ? 1 : DF2_Q4_AUTO(rows);
#define DF2_ROWS4_CASE(V, W, R, U) case V: gemm4_rows_kernel<W, R, U><<<(rows / 8) * (8 / R) / W, W * 32, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        switch (v4) {
            DF2_ROWS4_CASE(10, 4, 4, 2)
            DF2_ROWS4_CASE(11, 4, 2, 2)
            DF2_ROWS4_CASE(12, 4, 1, 2)
            DF2_ROWS4_CASE(13, 8, 2, 2)
            DF2_ROWS4_CASE(14, 8, 1, 2)
            DF2_ROWS4_CASE(15, 4, 2, 4)
            DF2_ROWS4_CASE(16, 4, 1, 4)
            DF2_ROWS4_CASE(17, 2, 2, 2)
            DF2_ROWS4_CASE(18, 8, 1, 4)
            DF2_ROWS4_CASE(19, 4, 4, 4)
            DF2_ROWS4_CASE(20, 4, 1, 8)
            DF2_ROWS4_CASE(21, 8, 1, 8)
            DF2_ROWS4_CASE(22, 8, 2, 4)
            DF2_ROWS4_CASE(23, 2, 1, 4)
            DF2_ROWS4_CASE(24, 1, 2, 2)
            DF2_ROWS4_CASE(25, 1, 4, 2)
            DF2_ROWS4_CASE(26, 2, 2, 1)
            DF2_ROWS4_CASE(27, 2, 4, 2)
            DF2_ROWS4_CASE(28, 1, 1, 2)
            DF2_ROWS4_CASE(29, 1, 2, 4)
            DF2_ROWS4_CASE(30, 2, 2, 4)
            DF2_ROWS4_CASE(31, 1, 1, 1)
            DF2_ROWS4_CASE(32, 1, 1, 4)
            DF2_ROWS4_CASE(33, 1, 1, 8)
            DF2_ROWS4_CASE(34, 2, 1, 2)
            DF2_ROWS4_CASE(35, 4, 1, 2)
            DF2_ROWS4_CASE(36, 4, 2, 1)
            DF2_ROWS4_CASE(37, 8, 2, 1)
            DF2_ROWS4_CASE(38, 2, 1, 1)
            DF2_ROWS4_CASE(39, 4, 1, 1)
            DF2_ROWS4_CASE(40, 1, 2, 1)
            DF2_ROWS4_CASE(41, 1, 4, 1)
            DF2_ROWS4_CASE(42, 2, 4, 1)
            DF2_ROWS4_CASE(43, 4, 4, 1)
            DF2_ROWS4_CASE(44, 8, 1, 1)
            DF2_ROWS4_CASE(45, 1, 8, 1)
            case 2: gemm4_kernel<2><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
            default: if (rows % 32 == 0) gemm4_kernel<4><<<rows / 32, 128, 0, st>>>(segs, xqp, xd, n_in, ncols);
                     else gemm4_kernel<2><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols);
                     break;
        }
        check("gemm4");
        return;
    }
    static const int tune = [] { const char* e = std::getenv("STRATA_DF2_GEMM"); return e ? std::atoi(e) : 1; }();   // 1 (default): rows a warp by matrix size, bitwise the same results; 0: 8 rows a warp as before
    const int variant = g_gemm_variant != 0 ? g_gemm_variant
                        : tune == 0         ? (rows % 32 == 0 ? 1 : 2)
                        : rows <= 1024      ? 20
                        : rows <= 3072      ? 13
                        : rows >= 15000     ? 10
                                            : (rows % 32 == 0 ? 1 : 2);
#define DF2_ROWS_CASE(V, W, R, U) case V: gemm_rows_kernel<W, R, U><<<(rows / 8) * (8 / R) / W, W * 32, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
    switch (variant) {
        DF2_ROWS_CASE(10, 4, 4, 2)
        DF2_ROWS_CASE(11, 4, 2, 2)
        DF2_ROWS_CASE(12, 4, 1, 2)
        DF2_ROWS_CASE(13, 8, 2, 2)
        DF2_ROWS_CASE(14, 8, 1, 2)
        DF2_ROWS_CASE(15, 4, 2, 4)
        DF2_ROWS_CASE(16, 4, 1, 4)
        DF2_ROWS_CASE(17, 2, 2, 2)
        DF2_ROWS_CASE(18, 8, 1, 4)
        DF2_ROWS_CASE(19, 4, 4, 4)
        DF2_ROWS_CASE(20, 4, 1, 8)
        DF2_ROWS_CASE(21, 8, 1, 8)
        DF2_ROWS_CASE(22, 8, 2, 4)
        DF2_ROWS_CASE(23, 2, 1, 4)
        case 3: gemm8_kernel<2, false><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 4: gemm8_kernel<1, true, 2><<<rows / 8, 32, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 5: gemm8_kernel<2, true, 1><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 6: gemm8_kernel<2, true, 4><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 7: gemm8_kernel<4, true, 4><<<rows / 32, 128, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 8: gemm8_kernel<8, true, 2><<<rows / 64, 256, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 9: gemm8_kernel<4, false><<<rows / 32, 128, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
        case 2: gemm8_kernel<2, true><<<rows / 16, 64, 0, st>>>(segs, xqp, xd, n_in, ncols); break;   // 16 rows a block: every n_out is a multiple of 16
        default: gemm8_kernel<4, true><<<rows / 32, 128, 0, st>>>(segs, xqp, xd, n_in, ncols); break;
    }
    check("gemm8");
}

void df2_q4_tile(const void* blocks, void* qs, void* dpl, int n_in, int64_t count, void* stream) {
    q4_tile_kernel<<<(unsigned) ((count + 255) / 256), 256, 0, (cudaStream_t) stream>>>(static_cast<const uint8_t*>(blocks), static_cast<uint8_t*>(qs),
                                                                                         static_cast<__half*>(dpl), n_in, count);
    check("q4_tile");
}

void df2_quant_rows(const float* x, int K, int ncols, void* xq, float* xd, void* stream) {
    quant_rows_kernel<<<dim3((K / 32 + 7) / 8, ncols), 256, 0, (cudaStream_t) stream>>>(x, K, static_cast<int8_t*>(xq), xd);
    check("quant_rows");
}

void df2_rms_quant(const float* h, const float* gamma, float* xn, int rows, int N, float eps, void* xq, float* xd, void* stream) {
    rms_quant_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(h, gamma, xn, N, eps, static_cast<int8_t*>(xq), xd);
    check("rms_quant");
}

void df2_conv_quant(const float* x, const float* dyn, int dyn_stride, const float* base, int rows, int H, int group, void* xq, float* xd, void* stream) {
    conv_quant_kernel<<<dim3((H / 32 + 7) / 8, rows), 256, 0, (cudaStream_t) stream>>>(x, dyn, dyn_stride, base, H, group, static_cast<int8_t*>(xq), xd);
    check("conv_quant");
}

void df2_swiglu_quant(const float* gate, const float* up, int rows, int I, void* xq, float* xd, void* stream) {
    swiglu_quant_kernel<<<dim3((I / 32 + 7) / 8, rows), 256, 0, (cudaStream_t) stream>>>(gate, up, I, static_cast<int8_t*>(xq), xd);
    check("swiglu_quant");
}

void df2_q8_tile(const void* blocks, void* qs, void* dpl, int n_in, int64_t count, void* stream) {
    q8_tile_kernel<<<(unsigned) ((count + 255) / 256), 256, 0, (cudaStream_t) stream>>>(static_cast<const uint8_t*>(blocks), static_cast<int8_t*>(qs),
                                                                                      static_cast<__half*>(dpl), n_in, count);
    check("q8_tile");
}

void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok, void* stream) {
    if ((n + kDf2TopBlocks - 1) / kDf2TopBlocks > TK_CHUNK_MAX) throw std::invalid_argument("df2_topk: too many logits per row");
    float* cval = reinterpret_cast<float*>(scratch);
    int32_t* cidx = reinterpret_cast<int32_t*>(scratch + (size_t) rows * kDf2TopBlocks * kDf2TopK);
    topk_part_kernel<<<dim3(rows, kDf2TopBlocks), 256, 0, (cudaStream_t) stream>>>(logits, n, cval, cidx);
    check("topk_part");
    topk_merge_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(cval, cidx, idx_map, val, tok);
    check("topk_merge");
}

void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb, int rank, int rows,
                const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream, bool cb_q4) {
    if (rank != 256) throw std::invalid_argument("df2_select: the selector rank must be 256");
    if (cb_q4) select_kernel<true><<<1, 256, 0, (cudaStream_t) stream>>>(hp, unary, tok, pred_cb, succ_cb, rank, rows, anchor_tok, out_tok, out_prob);
    else select_kernel<false><<<1, 256, 0, (cudaStream_t) stream>>>(hp, unary, tok, pred_cb, succ_cb, rank, rows, anchor_tok, out_tok, out_prob);
    check("select");
}

void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream) {
    tap_rows_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t) stream>>>(R, out, out_stride, n_embd, hc);
    check("tap_rows");
}

}  // namespace strata::kernels
