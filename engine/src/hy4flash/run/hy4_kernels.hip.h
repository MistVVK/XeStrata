// hy4_kernels.hip.h - elementwise / norm / iHC / router / engram / head kernels of the Hy4-Flash graph (gfx1151, wave32)
// Residual stream x is fp32 [T][4][3072]. GEMM inputs are fp16, GEMM outputs fp32.
#pragma once
#include "hy4_gpu.h"

#include <cstdint>

namespace hy4 {

constexpr int HID = 3072;
constexpr int NHC = 4;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float block_sum(float v, float* sh /*[32]*/) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float r = (l < nw) ? sh[l] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor(r, o);
    return r;
}

__device__ __forceinline__ float fp16_bits_to_f32(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// ---- embedding: S8_K row -> 4 identical hc streams (fp32). W: S8_K [V][3072] raw, row = 12 super-blocks of 262 B.
__global__ void k_embed_hc(const uint8_t* __restrict__ W, const int32_t* __restrict__ ids, float* __restrict__ x, int T) {
    int t = blockIdx.x;
    if (t >= T) return;
    const uint8_t* row = W + (size_t)ids[t] * 12 * 262;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        int sb = d >> 8, i = d & 255, g = i >> 6;
        const uint8_t* blk = row + sb * 262;
        uint16_t hb = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float D = fp16_bits_to_f32(hb);
        float eff = D * ((float)blk[2 + g] / 255.0f);
        float v = (float)(int8_t)blk[6 + i] * eff;
#pragma unroll
        for (int h = 0; h < NHC; ++h) x[((size_t)t * NHC + h) * HID + d] = v;
    }
}

// ---- RMSNorm over D with optional weight; in fp32, out fp32 and/or fp16
__global__ void k_rms_rows(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ outf, __half* __restrict__ outh, int D, float eps) {
    __shared__ float sh[32];
    const float* x = in + (size_t)blockIdx.x * D;
    float ss = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[i] * inv * (w ? w[i] : 1.0f);
        if (outf) outf[(size_t)blockIdx.x * D + i] = v;
        if (outh) outh[(size_t)blockIdx.x * D + i] = __float2half(v);
    }
}

// ---- iHC mixes: per token, rms of the 12288 flat state, 8 dot products with hc_fn [8][12288] -> pre[4], post[4]
__global__ void k_hc_mix(const float* __restrict__ x, const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base, float* __restrict__ pre,
                         float* __restrict__ post, int nout /*8 for layers, 4 for head*/, float eps, float hc_eps, float hc_mag) {
    __shared__ float sh[32];
    const float* xt = x + (size_t)blockIdx.x * NHC * HID;
    const int K = NHC * HID;
    float ss = 0;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = xt[k];
        ss += v * v;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < nout) acc[j] += v * fn[(size_t)j * K + k];
    }
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / K + eps);
    float m[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        m[j] = 0;
        if (j < nout) m[j] = block_sum(acc[j], sh) * inv;
    }
    if (threadIdx.x == 0) {
        if (nout == 8) {
            for (int h = 0; h < NHC; ++h) {
                pre[(size_t)blockIdx.x * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
                post[(size_t)blockIdx.x * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
            }
        } else {  // head: pre only, scale is a single value
            for (int h = 0; h < NHC; ++h) pre[(size_t)blockIdx.x * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
        }
    }
}

__global__ void k_zero_f(float* __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

// ---- multi-block variant of the iHC mix (decode: one block per token is too little parallelism).
// grid (T, 24), block 128: each block takes 512 of the 12288 elements; partial sums are atomically added into acc[t][0..8] (acc[t][8] = sum of squares); acc must be zeroed.
__global__ void k_hc_mix_part(const float* __restrict__ x, const float* __restrict__ fn, float* __restrict__ acc, int nout) {
    __shared__ float sh[8][4];
    const int K = NHC * HID;
    int t = blockIdx.x, slab = blockIdx.y;
    const float* xt = x + (size_t)t * K;
    float ss = 0, a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = 0;
    for (int k = slab * 512 + threadIdx.x; k < (slab + 1) * 512; k += 128) {
        float v = xt[k];
        ss += v * v;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < nout) a[j] += v * fn[(size_t)j * K + k];
    }
    // wave reduce then 4 waves
    float vals[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) vals[j] = a[j];
    vals[8] = ss;
#pragma unroll
    for (int j = 0; j < 9; ++j)
        for (int o = 16; o > 0; o >>= 1) vals[j] += __shfl_xor(vals[j], o);
    __shared__ float shv[9][4];
    if ((threadIdx.x & 31) == 0)
#pragma unroll
        for (int j = 0; j < 9; ++j) shv[j][threadIdx.x >> 5] = vals[j];
    __syncthreads();
    if (threadIdx.x < 9) {
        int j = threadIdx.x;
        if (j == 8 || j < nout) atomicAdd(&acc[(size_t)t * 9 + j], shv[j][0] + shv[j][1] + shv[j][2] + shv[j][3]);
    }
}
__global__ void k_hc_mix_fin(const float* __restrict__ acc, const float* __restrict__ scale, const float* __restrict__ base, float* __restrict__ pre, float* __restrict__ post, int nout,
                             float eps, float hc_eps, float hc_mag) {
    int t = blockIdx.x;
    if (threadIdx.x != 0) return;
    const int K = NHC * HID;
    float inv = rsqrtf(acc[(size_t)t * 9 + 8] / K + eps);
    float m[8];
    for (int j = 0; j < 8; ++j) m[j] = j < nout ? acc[(size_t)t * 9 + j] * inv : 0.0f;
    if (nout == 8) {
        for (int h = 0; h < NHC; ++h) {
            pre[(size_t)t * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
            post[(size_t)t * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
        }
    } else {
        for (int h = 0; h < NHC; ++h) pre[(size_t)t * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
    }
}

// ---- F32 GEMV for the router: one block per output row, M <= 16 tokens; W [N][K] f32, X [M][K] f32, Y [M][N]
__global__ void __launch_bounds__(256) k_f32_gemv(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, int N, int K, int M) {
    __shared__ float sh[32];
    int n = blockIdx.x;
    const float4* w4 = (const float4*)(W + (size_t)n * K);
    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0;
    for (int k4 = threadIdx.x; k4 < K / 4; k4 += blockDim.x) {
        float4 w = w4[k4];
#pragma unroll
        for (int m = 0; m < 16; ++m)
            if (m < M) {
                float4 xv = ((const float4*)(X + (size_t)m * K))[k4];
                acc[m] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
    }
    for (int m = 0; m < M; ++m) {
        float r = block_sum(acc[m], sh);
        if (threadIdx.x == 0) Y[(size_t)m * N + n] = r;
    }
}

// ---- tiled fp32 GEMM for the router at prefill: Y[M][N] = X[M][K] . W[N][K]^T (fp32 everywhere).  block 16x16 threads, tile 64 tokens x 64 outputs, BK 16
__global__ void __launch_bounds__(256) k_f32_gemm(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, int N, int K, int M) {
    __shared__ float sX[16][65];
    __shared__ float sW[16][65];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx = threadIdx.x + l * 256;  // 0..1023: tile 64 rows x 16 k
            int r = idx >> 4, kk = idx & 15;
            int mrow = m0 + r, nrow = n0 + r;
            sX[kk][r] = (mrow < M) ? X[(size_t)mrow * K + k0 + kk] : 0.f;
            sW[kk][r] = (nrow < N) ? W[(size_t)nrow * K + k0 + kk] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = sX[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = sW[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
            if (m < M && n < N) Y[(size_t)m * N + n] = acc[i][j];
        }
}

// ---- cur = sum_h x[h]*pre[h]; then rmsnorm(cur, w) -> fp16 (GEMM input) and optionally fp32
__global__ void k_hc_reduce_norm(const float* __restrict__ x, const float* __restrict__ pre, const float* __restrict__ w, __half* __restrict__ outh, float* __restrict__ outf,
                                 float eps) {
    __shared__ float sh[32];
    const float* xt = x + (size_t)blockIdx.x * NHC * HID;
    const float* p = pre + (size_t)blockIdx.x * NHC;
    float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    float ss = 0;
    constexpr int PER = HID / 256;  // blockDim must be 256
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int d = threadIdx.x + i * 256;
        float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        ss += c * c;
    }
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / HID + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int d = threadIdx.x + i * 256;
        float o = v[i] * inv * (w ? w[d] : 1.0f);
        if (outh) outh[(size_t)blockIdx.x * HID + d] = __float2half(o);
        if (outf) outf[(size_t)blockIdx.x * HID + d] = o;
    }
}

// ---- x[h] += sub * post[h]   (sub fp32 [T][3072])
__global__ void k_hc_post(float* __restrict__ x, const float* __restrict__ sub, const float* __restrict__ post) {
    const float* s = sub + (size_t)blockIdx.x * HID;
    const float* p = post + (size_t)blockIdx.x * NHC;
    float* xt = x + (size_t)blockIdx.x * NHC * HID;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        float sv = s[d];
#pragma unroll
        for (int h = 0; h < NHC; ++h) xt[h * HID + d] += sv * p[h];
    }
}

// ---- out = rmsnorm(a (+ b), w)  fp32 in, fp32 out (b optional)
__global__ void k_add_rms(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ w, float* __restrict__ out, int D, float eps) {
    __shared__ float sh[32];
    size_t o = (size_t)blockIdx.x * D;
    float ss = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = a[o + i] + (b ? b[o + i] : 0.0f);
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = a[o + i] + (b ? b[o + i] : 0.0f);
        out[o + i] = v * inv * w[i];
    }
}

// ---- fused: v = rmsnorm(a (+ b), w); x[h] += v * post[h]  (sublayer output norm + iHC post in one kernel). blockDim 256, D = 3072.
__global__ void k_add_rms_post(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ w, float* __restrict__ x, const float* __restrict__ post, float eps) {
    __shared__ float sh[32];
    size_t o = (size_t)blockIdx.x * HID;
    constexpr int PER = HID / 256;
    float v[PER];
    float ss = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int d = threadIdx.x + i * 256;
        float t = a[o + d] + (b ? b[o + d] : 0.0f);
        v[i] = t;
        ss += t * t;
    }
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / HID + eps);
    const float* p = post + (size_t)blockIdx.x * NHC;
    float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    float* xt = x + (size_t)blockIdx.x * NHC * HID;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int d = threadIdx.x + i * 256;
        float s = v[i] * inv * w[d];
        xt[d] += s * p0;
        xt[HID + d] += s * p1;
        xt[2 * HID + d] += s * p2;
        xt[3 * HID + d] += s * p3;
    }
}

// ---- a += b (fp32)
__global__ void k_add_inplace(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ---- fp32 -> fp16
__global__ void k_f2h(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ---- SwiGLU: h = silu(g) * u, optional clamp (routed experts only), fp32 in -> fp16 out.  g,u: [T][n] each
__global__ void k_swiglu_h(const float* __restrict__ g, const float* __restrict__ u, __half* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(siluf_(g[i]) * u[i]);
}

// ---- gated attention: attn(fp32 or half) * sigmoid(gate fp32) -> fp16
__global__ void k_gate_mul_h(const float* __restrict__ a, const float* __restrict__ gate, __half* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(a[i] * sigmoidf_(gate[i]));
}

// ---- router: logits fp32 [T][512] -> top-16 by sigmoid+bias; weights = p/(sum p + 1e-20)*scale.  One block (256 thr) per token.
// rec != nullptr (decode, one token): also copies the 16-byte ExpertDev records of the chosen experts (ex_layer = the layer's table of 512) into rec[slot], the input of the one-token expert
// kernels (moe_experts slot_rec path), which then need neither the counting sort nor per-expert token lists.
__global__ void k_router_topk(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ idx, float* __restrict__ w, int E, int K, float scale,
                              const uint4* __restrict__ ex_layer = nullptr, uint4* __restrict__ rec = nullptr) {
    __shared__ float sp[512], ss[512];
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ int chosen[16];
    const float* lg = logits + (size_t)blockIdx.x * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float p = sigmoidf_(lg[e]);
        sp[e] = p;
        ss[e] = p + bias[e];
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += blockDim.x)
            if (ss[e] > bv || (ss[e] == bv && e < bi)) { bv = ss[e]; bi = e; }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor(bv, o);
            int oi = __shfl_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        int wv = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { rv[wv] = bv; ri[wv] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b2 = rv[0]; int i2 = ri[0];
            for (int q = 1; q < (int)(blockDim.x >> 5); ++q)
                if (rv[q] > b2 || (rv[q] == b2 && ri[q] < i2)) { b2 = rv[q]; i2 = ri[q]; }
            chosen[k] = i2;
            ss[i2] = -3.0e38f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += sp[chosen[k]];
        for (int k = 0; k < K; ++k) {
            idx[(size_t)blockIdx.x * K + k] = chosen[k];
            w[(size_t)blockIdx.x * K + k] = sp[chosen[k]] / (sum + 1e-20f) * scale;
            if (rec) rec[k] = ex_layer[chosen[k]];
        }
    }
}

// ---- fused iHC pre-mix + input RMSNorm: replaces k_zero_f + k_hc_mix_part + k_hc_mix_fin + k_hc_reduce_norm (4 launches) by one.
// grid (T, 24), block 256. Block (t, slab) reduces 512 of the 12288 flat elements into 9 partial sums (8 mixes + sum of squares) stored per slab (fixed order: deterministic).
// The last block to finish a token (ticket counter cnt[t], self-resetting, zero-initialised once) sums the 24 slabs in order, forms pre/post (post goes to global for the
// sublayer-output kernel), and does cur = sum_h x_h pre_h -> RMSNorm(w) -> fp16 (outh) and optionally fp32 (outf). nout = 8 (layers) or 4 (head, pre only).
__global__ void __launch_bounds__(256) k_hc_pre(const float* __restrict__ x, const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ nw,
                                                 float* __restrict__ part, unsigned* __restrict__ cnt, float* __restrict__ post_out, __half* __restrict__ outh, float* __restrict__ outf,
                                                 int nout, float eps, float hc_eps, float hc_mag, float norm_eps) {
    constexpr int K = NHC * HID;
    constexpr int NSLAB = 24;
    __shared__ float shv[9][4];
    __shared__ float tot[9];
    __shared__ float pre_s[4];
    __shared__ float sh[32];
    __shared__ int is_last;
    const int t = blockIdx.x, slab = blockIdx.y, tid = threadIdx.x;
    const float* xt = x + (size_t)t * K;
    // partial sums: threads 0..127 in the exact summation order of k_hc_mix_part_d (the deterministic mix), so both paths agree bit for bit
    if (tid < 128) {
        float ss = 0, a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = 0;
        for (int k = slab * 512 + tid; k < (slab + 1) * 512; k += 128) {
            const float v = xt[k];
            ss += v * v;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < nout) a[j] += v * fn[(size_t)j * K + k];
        }
        float vals[9];
#pragma unroll
        for (int j = 0; j < 8; ++j) vals[j] = a[j];
        vals[8] = ss;
#pragma unroll
        for (int j = 0; j < 9; ++j)
            for (int o = 16; o > 0; o >>= 1) vals[j] += __shfl_xor(vals[j], o);
        if ((tid & 31) == 0)
#pragma unroll
            for (int j = 0; j < 9; ++j) shv[j][tid >> 5] = vals[j];
    }
    __syncthreads();
    if (tid < 9) part[((size_t)t * NSLAB + slab) * 9 + tid] = (shv[tid][0] + shv[tid][1]) + (shv[tid][2] + shv[tid][3]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned prev = atomicAdd(&cnt[t], 1u);
        is_last = (prev == NSLAB - 1);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (tid < 9) {
        float s = 0;
        for (int q = 0; q < NSLAB; ++q) s += *(volatile const float*)&part[((size_t)t * NSLAB + q) * 9 + tid];
        tot[tid] = s;
    }
    __syncthreads();
    if (tid == 0) {
        cnt[t] = 0;
        const float inv = rsqrtf(tot[8] / K + eps);
        float m[8];
        for (int j = 0; j < 8; ++j) m[j] = j < nout ? tot[j] * inv : 0.0f;
        for (int h = 0; h < NHC; ++h) {
            pre_s[h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
            if (nout == 8) post_out[(size_t)t * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
        }
    }
    __syncthreads();
    // cur = sum_h x_h * pre_h, RMSNorm, fp16/fp32 out (same arithmetic as k_hc_reduce_norm)
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
    float cs = 0;
    constexpr int PER = HID / 256;
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = tid + i * 256;
        const float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        cs += c * c;
    }
    cs = block_sum(cs, sh);
    const float inv2 = rsqrtf(cs / HID + norm_eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = tid + i * 256;
        const float o = v[i] * inv2 * (nw ? nw[d] : 1.0f);
        if (outh) outh[(size_t)t * HID + d] = __float2half(o);
        if (outf) outf[(size_t)t * HID + d] = o;
    }
}

// ---- router top-k by rank: expert e (one thread each, E = 512) counts the experts ahead of it (higher biased score, ties to the lower index = the order of the argmax rounds above), the 16
// best write chosen[rank]. Bit-identical to k_router_topk, without its 16 dependent block-wide rounds. 512 threads per token; optional routing tables as in k_router_topk.
__global__ void __launch_bounds__(512) k_router_topk_r(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ idx, float* __restrict__ w, int K, float scale,
                                                       const uint4* __restrict__ ex_layer = nullptr, uint4* __restrict__ rec = nullptr) {
    __shared__ __attribute__((aligned(16))) float ss[512];
    __shared__ float sp[512];
    __shared__ int chosen[16];
    const int e = threadIdx.x;
    const float p = sigmoidf_(logits[(size_t)blockIdx.x * 512 + e]);
    sp[e] = p;
    const float v = p + bias[e];
    ss[e] = v;
    __syncthreads();
    int rank = 0;
    const float4* s4 = (const float4*)ss;
#pragma unroll 8
    for (int j4 = 0; j4 < 128; ++j4) {
        const float4 u = s4[j4];
        const int j = 4 * j4;
        rank += (u.x > v) | ((u.x == v) & (j < e));
        rank += (u.y > v) | ((u.y == v) & (j + 1 < e));
        rank += (u.z > v) | ((u.z == v) & (j + 2 < e));
        rank += (u.w > v) | ((u.w == v) & (j + 3 < e));
    }
    if (rank < K) chosen[rank] = e;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += sp[chosen[k]];
        for (int k = 0; k < K; ++k) {
            idx[(size_t)blockIdx.x * K + k] = chosen[k];
            w[(size_t)blockIdx.x * K + k] = sp[chosen[k]] / (sum + 1e-20f) * scale;
            if (rec) rec[k] = ex_layer[chosen[k]];
        }
    }
}

// ---- argmax over logits [V] (one row) -> token id; 1 block of 1024
__global__ void k_argmax(const float* __restrict__ logits, int V, int32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -3.0e38f;
    int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(bv, o);
        int oi = __shfl_xor(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b2 = sv[0]; int i2 = si[0];
        for (int q = 1; q < (int)(blockDim.x >> 5); ++q)
            if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
        *out = i2;
    }
}

// ---- Engram: raw rows [T][16][100 B] (f32 scale + 96 int8) -> normed rows r[T][1536] fp16
// r[t][c*96+i] = rmsnorm96(row) * lookup_norm[c][i] * valid[t][c]
__global__ void k_engram_rows(const uint8_t* __restrict__ raw, const float* __restrict__ lnw /*[16][96]*/, const uint8_t* __restrict__ valid /*[T][16]*/,
                              __half* __restrict__ r) {
    int t = blockIdx.x, c = threadIdx.x >> 5, lane = threadIdx.x & 31;  // 16 rows per block, 32 lanes each (3 elements)
    const uint8_t* row = raw + ((size_t)t * 16 + c) * 100;
    float scale;
    memcpy(&scale, row, 4);
    float v[3];
    float ss = 0;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        v[j] = (float)(int8_t)row[4 + lane + 32 * j] * scale;
        ss += v[j] * v[j];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    float inv = rsqrtf(ss / 96.0f + 1.1920929e-7f);
    float vm = valid[(size_t)t * 16 + c] ? 1.0f : 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) r[(size_t)t * 1536 + c * 96 + lane + 32 * j] = __float2half(v[j] * inv * lnw[c * 96 + lane + 32 * j] * vm);
}

// ---- Engram apply: kv [T][5][3072] fp32 (without bias; bias added here), x [T][4][3072].
// block per (token, stream j): dot = sum(rms(x)*rms(key)*nk*nq)/sqrt(3072); gate = sigmoid(copysign(sqrt(max(|dot|,1e-6)),dot));
// x += rms(value)*pre_gamma * (gate + hc_scale[j]) * any_valid
__global__ void k_engram_apply(float* __restrict__ x, const float* __restrict__ kv /*[5][T][3072]*/, const float* __restrict__ kvb /*[5][3072]*/, const float* __restrict__ egate /*[4][3072]=nk*nq*/,
                               const float* __restrict__ pre_gamma, const float* __restrict__ hcs /*[4][3072]*/, const uint8_t* __restrict__ valid, int T) {
    __shared__ float sh[32];
    int t = blockIdx.x, j = blockIdx.y;
    const float EPS = 1.1920929e-7f;
    float* xs = x + ((size_t)t * NHC + j) * HID;
    const float* key = kv + ((size_t)j * T + t) * HID;
    const float* val = kv + ((size_t)4 * T + t) * HID;
    float sx = 0, sk = 0, sv = 0;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        float a = xs[d], k = key[d] + kvb[j * HID + d], v = val[d] + kvb[4 * HID + d];
        sx += a * a; sk += k * k; sv += v * v;
    }
    sx = block_sum(sx, sh); sk = block_sum(sk, sh); sv = block_sum(sv, sh);
    float xi = rsqrtf(sx / HID + EPS), ki = rsqrtf(sk / HID + EPS), vi = rsqrtf(sv / HID + EPS);
    float dot = 0;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        float a = xs[d], k = key[d] + kvb[j * HID + d];
        dot += (a * xi) * (k * ki) * egate[j * HID + d];
    }
    dot = block_sum(dot, sh) / sqrtf((float)HID);
    float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    float gate = sigmoidf_(dot < 0 ? -mag : mag);
    int anyv = 0;
    for (int c = 0; c < 16; ++c) anyv |= valid[(size_t)t * 16 + c] ? 1 : 0;
    float anyf = anyv ? 1.0f : 0.0f;
    for (int d = threadIdx.x; d < HID; d += blockDim.x) {
        float v = val[d] + kvb[4 * HID + d];
        xs[d] += (v * vi * pre_gamma[d]) * (gate + hcs[j * HID + d]) * anyf;
    }
}

// ---- add bias-free helper: y[t][n] = y + bias? (not needed) ; fp16 gather of 1 row t from [T][D]
__global__ void k_copy_row_h(const __half* __restrict__ in, __half* __restrict__ out, int D, int t) {
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = in[(size_t)t * D + i];
}

}  // namespace hy4
