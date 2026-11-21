// hy4_indexer.hip.h - compressed IndexPool of the 12 sparse layers (kpool 4, 16 index heads x 128 dims, top-(topk/4 - 1) pools + the current group)
// Per layer state: ivg cache half [ctx][256] (iv | ig of every token), pooled keys half [ctx/4][128] (normed, rope at the group's first position).
#pragma once
#include "hy4_gpu.h"

#include <cstdint>

namespace hy4 {
namespace idx {

__device__ __forceinline__ void rope_pair_i(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    float freq = 1.0f / powf(theta, (float)(2 * i) / 64.0f);
    float ang = (float)pos * freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

__device__ __forceinline__ float bsum128(float v, float* sh) {  // block of 128 threads (4 waves)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    return sh[0] + sh[1] + sh[2] + sh[3];
}

// store iv, ig (fp32 [T][128] each) of tokens p0..p0+T-1 into the fp16 cache rows
__global__ void k_idx_store(const float* __restrict__ iv, const float* __restrict__ ig, __half* __restrict__ ivg, int p0, const int* __restrict__ p0_dev = nullptr) {
    int t = blockIdx.x, d = threadIdx.x;  // 128 threads
    const int p0_u = p0_dev ? *p0_dev : p0;
    ivg[(size_t)(p0_u + t) * 256 + d] = __float2half(iv[(size_t)t * 128 + d]);
    ivg[(size_t)(p0_u + t) * 256 + 128 + d] = __float2half(ig[(size_t)t * 128 + d]);
}

// pool completed groups g in [g_lo, g_lo + n): softmax over the 4 tokens of (ig + ape), weighted sum of iv, rmsnorm(key_norm), rope at 4g on dims 64..127
__global__ void k_idx_pool(const __half* __restrict__ ivg, const float* __restrict__ ape /*[4][128]*/, const float* __restrict__ knw /*[128]*/, __half* __restrict__ pooled, int g_lo, float theta,
                           float eps, const int* __restrict__ p0_dev = nullptr, int T_pool = 0) {
    __shared__ float sh[4];
    __shared__ float sv[128];
    int g;
    if (p0_dev) {
        const int p0_u = *p0_dev;
        const int glo = p0_u / 4, ghi = (p0_u + T_pool) / 4 - 1;
        g = glo + (int)blockIdx.x;
        if (g > ghi) return;
    } else {
        g = g_lo + (int)blockIdx.x;
    }
    int d = threadIdx.x;
    float l[4], mx = -3.0e38f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        l[j] = __half2float(ivg[(size_t)(4 * g + j) * 256 + 128 + d]) + ape[j * 128 + d];
        mx = fmaxf(mx, l[j]);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        l[j] = __expf(l[j] - mx);
        s += l[j];
    }
    float acc = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += (l[j] / s) * __half2float(ivg[(size_t)(4 * g + j) * 256 + d]);
    float ss = bsum128(acc * acc, sh);
    float inv = rsqrtf(ss / 128.0f + eps);
    float v = acc * inv * knw[d];
    sv[d] = v;
    __syncthreads();
    if (d >= 64) {
        int i = (d - 64) >> 1;
        float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1], o0, o1;
        rope_pair_i(x0, x1, 4 * g, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    pooled[(size_t)g * 128 + d] = __float2half(v);
}

// iq: fp32 [T][2048] (16 heads x 128) -> rope on dims 64..127 of every head at position p0+t -> half [T][2048]
__global__ void k_idx_q_rope(const float* __restrict__ iq, __half* __restrict__ out, int p0, float theta, const int* __restrict__ p0_dev = nullptr) {
    int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;  // 128 threads
    const int p0_u = p0_dev ? *p0_dev : p0;
    const float* q = iq + (size_t)t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        int i = (d - 64) >> 1;
        float x0 = q[64 + 2 * i], x1 = q[64 + 2 * i + 1], o0, o1;
        rope_pair_i(x0, x1, p0_u + t, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t)t * 2048 + h * 128 + d] = __float2half(v);
}

// score[t][g] = sum_h relu(iq[t][h] . pooled[g]) * hw[t][h] * hscale, for g < grp(t) = (p0+t)/4, else -inf.  block (64 pools, 1 token), 64 threads.
// Uses packed half2 (v_dot2_f32_f16) + 4 KB LDS + 64 VGPRs instead of 128-VGPR fp32 array (2x FLOPs/cycle, half register pressure).
__global__ void __launch_bounds__(64) k_idx_score(const __half* __restrict__ iq, const float* __restrict__ hw, const __half* __restrict__ pooled, float* __restrict__ score, int p0, int G, int ldS,
                                                  float hscale, const int* __restrict__ p0_dev = nullptr, const int* __restrict__ G_dev = nullptr) {
    const int G_u = G_dev ? *G_dev : G;
    if ((int)blockIdx.x * 64 >= G_u) return;
    const int p0_u = p0_dev ? *p0_dev : p0;
    __shared__ uint32_t sq[16 * 64];
    __shared__ float shw[16];
    int t = blockIdx.y, g = blockIdx.x * 64 + threadIdx.x;
    const uint32_t* iq2 = reinterpret_cast<const uint32_t*>(iq + (size_t)t * 2048);
    for (int i = threadIdx.x; i < 16 * 64; i += 64) sq[i] = iq2[i];
    if (threadIdx.x < 16) shw[threadIdx.x] = hw[(size_t)t * 16 + threadIdx.x] * hscale;
    __syncthreads();
    if (g >= G_u) return;
    int grp = (p0_u + t) >> 2;
    float res;
    if (g >= grp) {
        res = -INFINITY;
    } else {
        uint32_t pk[64];
        const uint4* pr4 = reinterpret_cast<const uint4*>(pooled + (size_t)g * 128);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint4 v = pr4[i];
            pk[4 * i + 0] = v.x;
            pk[4 * i + 1] = v.y;
            pk[4 * i + 2] = v.z;
            pk[4 * i + 3] = v.w;
        }
        res = 0.f;
        for (int h = 0; h < 16; ++h) {
            const uint32_t* qh = sq + h * 64;
            float dsum = 0.f;
#pragma unroll
            for (int k = 0; k < 64; ++k) dsum = h2_dot_f32(qh[k], pk[k], dsum);
            res += fmaxf(dsum, 0.0f) * shw[h];
        }
    }
    score[(size_t)t * ldS + g] = res;
}

__device__ __forceinline__ uint32_t f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// per token: select the top n_top pools among g < grp (all of them if grp <= n_top), emit the key list (cache slots = positions):
// the 4 tokens of each selected pool, then the own group's tokens 4*grp..t.  keys [T][kmax], nkeys [T].  256 threads.
__global__ void k_idx_select(const float* __restrict__ score, int ldS, int p0, int n_top, int kmax, int32_t* __restrict__ keys, int32_t* __restrict__ nkeys,
                             const int* __restrict__ p0_dev = nullptr) {
    __shared__ int red[8];
    __shared__ int cnt_sel;
    __shared__ uint32_t thr_s;
    __shared__ int n_gt_s;
    int t = blockIdx.x;
    int pos = (p0_dev ? *p0_dev : p0) + t;
    int grp = pos >> 2;
    int32_t* kl = keys + (size_t)t * kmax;
    if (grp <= n_top) {  // everything before the own group plus the own group: causal
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) kl[i] = i;
        if (threadIdx.x == 0) nkeys[t] = pos + 1;
        return;
    }
    const float* sc = score + (size_t)t * ldS;
    // the n_top-th largest key (= the max thr with count(key >= thr) >= n_top): MSB-first radix select, 4 passes of an 8-bit histogram (integer LDS atomics: exact and independent of the
    // scheduling) instead of 33 dependent block-wide counting rounds. Same thr as the binary search it replaces.
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s;
    __shared__ int need_s;
    if (threadIdx.x == 0) { prefix_s = 0; need_s = n_top; }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int g = threadIdx.x; g < grp; g += blockDim.x) {
            const uint32_t k = f2key(sc[g]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { thr_s = prefix_s; cnt_sel = 0; }
    __syncthreads();
    uint32_t thr = thr_s;
    // count strictly greater
    int c = 0;
    for (int g = threadIdx.x; g < grp; g += blockDim.x) c += (f2key(sc[g]) > thr) ? 1 : 0;
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor(c, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        int tot = 0;
        for (int w = 0; w < 8; ++w) tot += red[w];
        n_gt_s = tot;
    }
    __syncthreads();
    int n_eq_need = n_top - n_gt_s;
    // emit in a fixed order (independent of the scheduling, so the attention over the list sums in the same order every run): all > thr in index order,
    // then the first n_eq_need == thr in index order. Contiguous chunk per thread, block-wide exclusive scans of the per-chunk counts.
    {
        __shared__ int o_gt[256], o_eq[256];
        const int chunk = (grp + (int)blockDim.x - 1) / (int)blockDim.x;
        const int g0 = threadIdx.x * chunk, g1 = min(grp, g0 + chunk);
        int cg = 0, ce = 0;
        for (int g = g0; g < g1; ++g) {
            const uint32_t k = f2key(sc[g]);
            cg += k > thr ? 1 : 0;
            ce += k == thr ? 1 : 0;
        }
        o_gt[threadIdx.x] = cg;
        o_eq[threadIdx.x] = ce;
        __syncthreads();
        if (threadIdx.x == 0) {
            int sg = 0, se = 0;
            for (int i = 0; i < (int)blockDim.x; ++i) {
                const int a = o_gt[i], b = o_eq[i];
                o_gt[i] = sg;
                o_eq[i] = se;
                sg += a;
                se += b;
            }
            cnt_sel = n_gt_s + min(se, n_eq_need);
        }
        __syncthreads();
        int sg = o_gt[threadIdx.x], se = o_eq[threadIdx.x];
        for (int g = g0; g < g1; ++g) {
            const uint32_t k = f2key(sc[g]);
            int s = -1;
            if (k > thr) s = sg++;
            else if (k == thr) { if (se < n_eq_need) s = n_gt_s + se; ++se; }
            if (s >= 0)
                for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
        }
    }
    __syncthreads();
    int npool = cnt_sel;
    int nown = pos - 4 * grp + 1;
    for (int i = threadIdx.x; i < nown; i += blockDim.x) kl[4 * npool + i] = 4 * grp + i;
    if (threadIdx.x == 0) nkeys[t] = 4 * npool + nown;
}

// ---- Tree-8 speculative verification kernels for IndexPool (bit-identical to sequential 1-token decode along each branch) ----
struct IdxTree {
    int n = 0;               // 1..8 tree nodes
    uint8_t depth[8] = {};   // logical position of node t is p0 + depth[t]
    uint8_t anc[8][8] = {};  // anc[t][d] = tree node index of node t's ancestor at depth d
};

// Per-node temporary pooled keys for up to 2 groups g in [g0, grp_t) completed inside the tree along node t's ancestor chain.
// grid (M, 2), block 128. tmp_pooled: half [M * 2][128].
__global__ void k_idx_pool_tree(const __half* __restrict__ ivg, const float* __restrict__ ape, const float* __restrict__ knw, __half* __restrict__ tmp_pooled, int p0, IdxTree tp,
                                float theta, float eps, const int* __restrict__ p0_dev = nullptr, const IdxTree* __restrict__ tp_dev = nullptr) {
    __shared__ float sh[4];
    __shared__ float sv[128];
    const int t = blockIdx.x, k = blockIdx.y, d = threadIdx.x;
    const int p0_use = p0_dev ? *p0_dev : p0;
    const int depth_t = (int)(tp_dev ? tp_dev->depth[t] : tp.depth[t]);
    const int g0 = p0_use >> 2;
    const int grp_t = (p0_use + depth_t) >> 2;
    const int g = g0 + k;
    if (g >= grp_t) return;
    int sl[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int lp = 4 * g + j;
        sl[j] = lp < p0_use ? lp : p0_use + (int)(tp_dev ? tp_dev->anc[t][(lp - p0_use) & 7] : tp.anc[t][(lp - p0_use) & 7]);
    }
    float l[4], mx = -3.0e38f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        l[j] = __half2float(ivg[(size_t)sl[j] * 256 + 128 + d]) + ape[j * 128 + d];
        mx = fmaxf(mx, l[j]);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        l[j] = __expf(l[j] - mx);
        s += l[j];
    }
    float acc = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += (l[j] / s) * __half2float(ivg[(size_t)sl[j] * 256 + d]);
    float ss = bsum128(acc * acc, sh);
    float inv = rsqrtf(ss / 128.0f + eps);
    float v = acc * inv * knw[d];
    sv[d] = v;
    __syncthreads();
    if (d >= 64) {
        int i = (d - 64) >> 1;
        float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1], o0, o1;
        rope_pair_i(x0, x1, 4 * g, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    tmp_pooled[(size_t)(t * 2 + k) * 128 + d] = __float2half(v);
}

__global__ void k_idx_q_rope_tree(const float* __restrict__ iq, __half* __restrict__ out, int p0, IdxTree tp, float theta,
                                  const int* __restrict__ p0_dev = nullptr, const IdxTree* __restrict__ tp_dev = nullptr) {
    int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* q = iq + (size_t)t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        int i = (d - 64) >> 1;
        const int p0_use = p0_dev ? *p0_dev : p0;
        const int depth_t = (int)(tp_dev ? tp_dev->depth[t] : tp.depth[t]);
        float x0 = q[64 + 2 * i], x1 = q[64 + 2 * i + 1], o0, o1;
        rope_pair_i(x0, x1, p0_use + depth_t, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t)t * 2048 + h * 128 + d] = __float2half(v);
}

__global__ void __launch_bounds__(64) k_idx_score_tree(const __half* __restrict__ iq, const float* __restrict__ hw, const __half* __restrict__ pooled, const __half* __restrict__ tmp_pooled,
                                                       float* __restrict__ score, int p0, IdxTree tp, int G, int ldS, float hscale,
                                                       const int* __restrict__ p0_dev = nullptr, const int* __restrict__ G_dev = nullptr, const IdxTree* __restrict__ tp_dev = nullptr) {
    const int G_use = G_dev ? *G_dev : G;
    if ((int)blockIdx.x * 64 >= G_use) return;
    __shared__ uint32_t sq[16 * 64];
    __shared__ float shw[16];
    int t = blockIdx.y, g = blockIdx.x * 64 + threadIdx.x;
    const uint32_t* iq2 = reinterpret_cast<const uint32_t*>(iq + (size_t)t * 2048);
    for (int i = threadIdx.x; i < 16 * 64; i += 64) sq[i] = iq2[i];
    if (threadIdx.x < 16) shw[threadIdx.x] = hw[(size_t)t * 16 + threadIdx.x] * hscale;
    __syncthreads();
    if (g >= G_use) return;
    const int p0_use = p0_dev ? *p0_dev : p0;
    const int depth_t = (int)(tp_dev ? tp_dev->depth[t] : tp.depth[t]);
    const int g0 = p0_use >> 2;
    const int grp = (p0_use + depth_t) >> 2;
    float res;
    if (g >= grp) {
        res = -INFINITY;
    } else {
        uint32_t pk[64];
        const __half* prow = (g < g0) ? (pooled + (size_t)g * 128) : (tmp_pooled + (size_t)(t * 2 + (g - g0)) * 128);
        const uint4* pr4 = reinterpret_cast<const uint4*>(prow);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint4 v = pr4[i];
            pk[4 * i + 0] = v.x;
            pk[4 * i + 1] = v.y;
            pk[4 * i + 2] = v.z;
            pk[4 * i + 3] = v.w;
        }
        res = 0.f;
        for (int h = 0; h < 16; ++h) {
            const uint32_t* qh = sq + h * 64;
            float dsum = 0.f;
#pragma unroll
            for (int k = 0; k < 64; ++k) dsum = h2_dot_f32(qh[k], pk[k], dsum);
            res += fmaxf(dsum, 0.0f) * shw[h];
        }
    }
    score[(size_t)t * ldS + g] = res;
}

__global__ void k_idx_select_tree(const float* __restrict__ score, int ldS, int p0, IdxTree tp, int n_top, int kmax, int32_t* __restrict__ keys, int32_t* __restrict__ nkeys,
                                  const int* __restrict__ p0_dev = nullptr, const IdxTree* __restrict__ tp_dev = nullptr) {
    __shared__ int red[8];
    __shared__ int cnt_sel;
    __shared__ uint32_t thr_s;
    __shared__ int n_gt_s;
    int t = blockIdx.x;
    const int p0_use = p0_dev ? *p0_dev : p0;
    const int depth_t = (int)(tp_dev ? tp_dev->depth[t] : tp.depth[t]);
    int pos = p0_use + depth_t;
    int grp = pos >> 2;
    int32_t* kl = keys + (size_t)t * kmax;
    if (grp <= n_top) {
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) kl[i] = i;
        if (threadIdx.x == 0) nkeys[t] = pos + 1;
        return;
    }
    const float* sc = score + (size_t)t * ldS;
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s;
    __shared__ int need_s;
    if (threadIdx.x == 0) { prefix_s = 0; need_s = n_top; }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int g = threadIdx.x; g < grp; g += blockDim.x) {
            const uint32_t k = f2key(sc[g]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { thr_s = prefix_s; cnt_sel = 0; }
    __syncthreads();
    uint32_t thr = thr_s;
    int c = 0;
    for (int g = threadIdx.x; g < grp; g += blockDim.x) c += (f2key(sc[g]) > thr) ? 1 : 0;
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor(c, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        int tot = 0;
        for (int w = 0; w < 8; ++w) tot += red[w];
        n_gt_s = tot;
    }
    __syncthreads();
    int n_eq_need = n_top - n_gt_s;
    {
        __shared__ int o_gt[256], o_eq[256];
        const int chunk = (grp + (int)blockDim.x - 1) / (int)blockDim.x;
        const int g0 = threadIdx.x * chunk, g1 = min(grp, g0 + chunk);
        int cg = 0, ce = 0;
        for (int g = g0; g < g1; ++g) {
            const uint32_t k = f2key(sc[g]);
            cg += k > thr ? 1 : 0;
            ce += k == thr ? 1 : 0;
        }
        o_gt[threadIdx.x] = cg;
        o_eq[threadIdx.x] = ce;
        __syncthreads();
        if (threadIdx.x == 0) {
            int sg = 0, se = 0;
            for (int i = 0; i < (int)blockDim.x; ++i) {
                const int a = o_gt[i], b = o_eq[i];
                o_gt[i] = sg;
                o_eq[i] = se;
                sg += a;
                se += b;
            }
            cnt_sel = n_gt_s + min(se, n_eq_need);
        }
        __syncthreads();
        int sg = o_gt[threadIdx.x], se = o_eq[threadIdx.x];
        for (int g = g0; g < g1; ++g) {
            const uint32_t k = f2key(sc[g]);
            int s = -1;
            if (k > thr) s = sg++;
            else if (k == thr) { if (se < n_eq_need) s = n_gt_s + se; ++se; }
            if (s >= 0)
                for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
        }
    }
    __syncthreads();
    int npool = cnt_sel;
    int nown = pos - 4 * grp + 1;
    for (int i = threadIdx.x; i < nown; i += blockDim.x) kl[4 * npool + i] = 4 * grp + i;
    if (threadIdx.x == 0) nkeys[t] = 4 * npool + nown;
}

__global__ void __launch_bounds__(256) k_idx_compact_ivg(__half* __restrict__ ivg, int p0, IdxTree src) {
    const int tid = threadIdx.x;
    for (int k = 0; k < src.n; ++k) {
        const int sidx = (int)src.depth[k];
        if (sidx == k) continue;
        ivg[(size_t)(p0 + k) * 256 + tid] = ivg[(size_t)(p0 + sidx) * 256 + tid];
        __syncthreads();
    }
}

}  // namespace idx
}  // namespace hy4
