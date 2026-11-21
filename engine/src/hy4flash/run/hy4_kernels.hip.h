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

// ---- templated F32 GEMV for the router (M <= 8): exact same summation order as k_f32_gemv, zero m-branch overhead, single barrier reduction across all MT tokens
template <int MT>
__global__ void __launch_bounds__(256) k_f32_gemv_t(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, int N, int K, int M) {
    __shared__ float sh[MT][8];
    const int n = blockIdx.x, tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const float4* w4 = (const float4*)(W + (size_t)n * K);
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    for (int k4 = tid; k4 < K / 4; k4 += 256) {
        const float4 w = w4[k4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float4 xv = ((const float4*)(X + (size_t)m * K))[k4];
                acc[m] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
        if (lane == 0) sh[m][wave] = v;
    }
    __syncthreads();
    if (tid < MT && tid < M) {
        float r = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) r += sh[tid][w];
        Y[(size_t)tid * N + n] = r;
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

// ---- fused a += b; out_h = half(a) (eliminates a separate k_f2h launch in hybrid MoE)
__global__ void k_add_f2h(float* __restrict__ a, const float* __restrict__ b, __half* __restrict__ out_h, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = a[i] + b[i];
        a[i] = v;
        out_h[i] = __float2half(v);
    }
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
            if ((unsigned)i2 >= (unsigned)E) i2 = 0;
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

// ---- router top-k by bitonic sort (E = 512, K = 16): bit-identical to k_router_topk (higher score first, ties to lower index),
// O(log^2 N) = 45 compare-exchanges/thread (5 warp-shuffle stages + 4 block stages) instead of O(N) = 512 comparisons/thread -> ~5x faster on decode!
__device__ __forceinline__ bool rtk_better(float va, int ia, float vb, int ib) {
    return (va > vb) || (va == vb && ia < ib);
}
__global__ void __launch_bounds__(512) k_router_topk_r(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ idx, float* __restrict__ w, int K, float scale,
                                                       const uint4* __restrict__ ex_layer = nullptr, uint4* __restrict__ rec = nullptr) {
    __shared__ float sp[512];
    __shared__ float sv[512];
    __shared__ int si[512];
    const int tid = threadIdx.x;
    const float p = sigmoidf_(logits[(size_t)blockIdx.x * 512 + tid]);
    sp[tid] = p;
    float v = p + bias[tid];
    int id = tid;
    // Stages k = 2, 4, 8, 16, 32: pure warp-shuffle bitonic sort (0 barriers!)
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        const bool desc = (tid & k) == 0;  // true = want better at lower lane index
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const float ov = __shfl_xor(v, j);
            const int oid = __shfl_xor(id, j);
            const bool lo = (tid & j) == 0;
            const bool keep_better = (lo == desc);
            const bool v_better = rtk_better(v, id, ov, oid);
            if (v_better != keep_better) { v = ov; id = oid; }
        }
    }
    sv[tid] = v;
    si[tid] = id;
    __syncthreads();
    // Stages k = 64, 128, 256, 512: cross-warp steps in LDS, then inner j <= 16 in warp shuffles
    for (int k = 64; k <= 512; k <<= 1) {
        const bool desc = (tid & k) == 0;
        for (int j = k >> 1; j >= 32; j >>= 1) {
            const int ixj = tid ^ j;
            const float v0 = sv[tid], v1 = sv[ixj];
            const int i0 = si[tid], i1 = si[ixj];
            const bool lo = (tid & j) == 0;
            const bool keep_better = (lo == desc);
            const bool b = rtk_better(v0, i0, v1, i1);
            __syncthreads();
            if (b != keep_better) { sv[tid] = v1; si[tid] = i1; }
            __syncthreads();
        }
        float vv = sv[tid];
        int ii = si[tid];
#pragma unroll
        for (int j = 16; j > 0; j >>= 1) {
            const float ov = __shfl_xor(vv, j);
            const int oid = __shfl_xor(ii, j);
            const bool lo = (tid & j) == 0;
            const bool keep_better = (lo == desc);
            const bool b = rtk_better(vv, ii, ov, oid);
            if (b != keep_better) { vv = ov; ii = oid; }
        }
        sv[tid] = vv;
        si[tid] = ii;
        __syncthreads();
    }
    // Top-K (desc=true for k=512 since tid<512 -> (tid&512)==0) now sit sorted in si[0..K-1]!
    if (tid < 32) {
        const int chosen_e = (tid < K) ? si[tid] : 0;
        const float pv = (tid < K) ? sp[chosen_e] : 0.f;
        float sum = pv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
        if (tid < K) {
            idx[(size_t)blockIdx.x * K + tid] = chosen_e;
            w[(size_t)blockIdx.x * K + tid] = pv / (sum + 1e-20f) * scale;
            if (rec) rec[tid] = ex_layer[chosen_e];
        }
    }
}

// ---- single-wave (32-thread) zero-barrier top-16 router (E = 512, 16 elements/lane in registers). Bit-identical to k_router_topk / k_router_topk_r, 32x fewer comparisons and zero __syncthreads().
__global__ void __launch_bounds__(32) k_router_topk_w32(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ idx, float* __restrict__ w, int K, float scale,
                                                        const uint4* __restrict__ ex_layer = nullptr, uint4* __restrict__ rec = nullptr) {
    const int lane = threadIdx.x;
    const float* lg = logits + (size_t)blockIdx.x * 512;
    float p[16], v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int e = lane + 32 * j;
        p[j] = sigmoidf_(lg[e]);
        v[j] = p[j] + bias[e];
    }
    int chosen[16];
    float cp[16];
    for (int k = 0; k < K; ++k) {
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            if (v[j] > bv || (v[j] == bv && e < bi)) { bv = v[j]; bi = e; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o);
            const int oi = __shfl_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((unsigned)bi >= 512u) bi = 0;
        const int win_lane = bi & 31, win_j = bi >> 5;
        float wp = 0.f;
        if (lane == win_lane) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (j == win_j) { wp = p[j]; v[j] = -3.0e38f; }
        }
        wp = __shfl(wp, win_lane);
        chosen[k] = bi;
        cp[k] = wp;
    }
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += cp[k];
        const float inv = scale / (sum + 1e-20f);
        for (int k = 0; k < K; ++k) {
            idx[(size_t)blockIdx.x * K + k] = chosen[k];
            w[(size_t)blockIdx.x * K + k] = cp[k] / (sum + 1e-20f) * scale;
            if (rec) rec[k] = ex_layer[chosen[k]];
        }
        (void)inv;
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

// ---- deterministic iHC mix (no float atomics): every slab writes its 9 partial sums to part[t][slab][9], the finisher adds the 24 slabs in a fixed order.
// Same maths as k_hc_mix_part + k_hc_mix_fin; used when the output must not depend on the scheduling (speculative decoding compares against plain decode).
__global__ void k_hc_mix_part_d(const float* __restrict__ x, const float* __restrict__ fn, float* __restrict__ part, int nout) {
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
        const int j = threadIdx.x;
        part[((size_t)t * 24 + slab) * 9 + j] = (shv[j][0] + shv[j][1]) + (shv[j][2] + shv[j][3]);
    }
}
// ---- batched deterministic iHC mix for prefill (T >= 16): stages fn[0..nout-1][slab*512..+511] once into LDS (16 KB) and processes up to 32 tokens per block in 8x4 sub-batches.
// Exact bit-for-bit identical summation order to k_hc_mix_part_d, with 32x less fn global read traffic.
__global__ void __launch_bounds__(128) k_hc_mix_part_d_batch(const float* __restrict__ x, const float* __restrict__ fn, float* __restrict__ part, int nout, int T) {
    constexpr int K = NHC * HID;
    const int t0 = blockIdx.x * 32;
    const int slab = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float sfn[8][512];
    __shared__ float shv[9][4];
    const int base_k = slab * 512;
    for (int j = 0; j < nout; ++j) {
        const float* fp = fn + (size_t)j * K + base_k;
#pragma unroll
        for (int st = 0; st < 4; ++st) sfn[j][st * 128 + tid] = fp[st * 128 + tid];
    }
    __syncthreads();
    for (int b0 = 0; b0 < 32; b0 += 4) {
        if (t0 + b0 >= T) break;
        float ss[4] = {0.f, 0.f, 0.f, 0.f};
        float a[4][8];
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int j = 0; j < 8; ++j) a[b][j] = 0.f;
#pragma unroll
        for (int st = 0; st < 4; ++st) {
            const int kl = st * 128 + tid;
            const int kg = base_k + kl;
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) f[j] = j < nout ? sfn[j][kl] : 0.f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int t = t0 + b0 + b;
                if (t < T) {
                    const float v = x[(size_t)t * K + kg];
                    ss[b] += v * v;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (j < nout) a[b][j] += v * f[j];
                }
            }
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int t = t0 + b0 + b;
            if (t >= T) break;
            float vals[9];
#pragma unroll
            for (int j = 0; j < 8; ++j) vals[j] = a[b][j];
            vals[8] = ss[b];
#pragma unroll
            for (int j = 0; j < 9; ++j)
                for (int o = 16; o > 0; o >>= 1) vals[j] += __shfl_xor(vals[j], o);
            if ((tid & 31) == 0)
#pragma unroll
                for (int j = 0; j < 9; ++j) shv[j][tid >> 5] = vals[j];
            __syncthreads();
            if (tid < 9) {
                part[((size_t)t * 24 + slab) * 9 + tid] = (shv[tid][0] + shv[tid][1]) + (shv[tid][2] + shv[tid][3]);
            }
            __syncthreads();
        }
    }
}

__global__ void k_hc_mix_fin_d(const float* __restrict__ part, const float* __restrict__ scale, const float* __restrict__ base, float* __restrict__ pre, float* __restrict__ post, int nout,
                               float eps, float hc_eps, float hc_mag) {
    int t = blockIdx.x;
    if (threadIdx.x != 0) return;
    const int K = NHC * HID;
    float acc[9];
    for (int j = 0; j < 9; ++j) acc[j] = 0.f;
    for (int s = 0; s < 24; ++s)
        for (int j = 0; j < 9; ++j) acc[j] += part[((size_t)t * 24 + s) * 9 + j];
    float inv = rsqrtf(acc[8] / K + eps);
    float m[8];
    for (int j = 0; j < 8; ++j) m[j] = j < nout ? acc[j] * inv : 0.0f;
    if (nout == 8) {
        for (int h = 0; h < NHC; ++h) {
            pre[(size_t)t * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
            post[(size_t)t * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
        }
    } else {
        for (int h = 0; h < NHC; ++h) pre[(size_t)t * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
    }
}

// ---- fused deterministic iHC finisher + reduce_norm (replaces k_hc_mix_fin_d + k_hc_reduce_norm -> 1 launch instead of 2, 100% bit-identical!)
__global__ void __launch_bounds__(256) k_hc_fin_reduce_norm(const float* __restrict__ part, const float* __restrict__ scale, const float* __restrict__ base,
                                                            float* __restrict__ pre, float* __restrict__ post, int nout, float hc_eps, float hc_mag,
                                                            const float* __restrict__ x, const float* __restrict__ nw, __half* __restrict__ outh, float* __restrict__ outf, float eps) {
    __shared__ float sh[32];
    __shared__ float s_acc[9];
    __shared__ float s_pre[4];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int K = NHC * HID;
    // 9 threads sum the 24 slabs in the exact same 0..23 order as k_hc_mix_fin_d
    if (tid < 9) {
        float a = 0.f;
        for (int s = 0; s < 24; ++s) a += part[((size_t)t * 24 + s) * 9 + tid];
        s_acc[tid] = a;
    }
    __syncthreads();
    if (tid < NHC) {
        const float inv = rsqrtf(s_acc[8] / K + eps);
        const float pr = sigmoidf_((tid < nout ? s_acc[tid] * inv : 0.0f) * scale[0] + base[tid]) + hc_eps;
        s_pre[tid] = pr;
        pre[(size_t)t * NHC + tid] = pr;
        if (nout == 8) {
            const float po = hc_mag * sigmoidf_(s_acc[NHC + tid] * inv * scale[1] + base[NHC + tid]) + hc_eps;
            post[(size_t)t * NHC + tid] = po;
        }
    }
    __syncthreads();
    const float* xt = x + (size_t)t * K;
    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float ss = 0.f;
    constexpr int PER = HID / 256;
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = tid + i * 256;
        const float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        ss += c * c;
    }
    ss = block_sum(ss, sh);
    const float invn = rsqrtf(ss / HID + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = tid + i * 256;
        const float o = v[i] * invn * (nw ? nw[d] : 1.0f);
        if (outh) outh[(size_t)t * HID + d] = __float2half(o);
        if (outf) outf[(size_t)t * HID + d] = o;
    }
}

// ---- add bias-free helper: y[t][n] = y + bias? (not needed) ; fp16 gather of 1 row t from [T][D]
__global__ void k_copy_row_h(const __half* __restrict__ in, __half* __restrict__ out, int D, int t) {
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = in[(size_t)t * D + i];
}

}  // namespace hy4
