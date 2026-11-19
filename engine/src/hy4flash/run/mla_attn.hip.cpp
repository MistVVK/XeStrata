// mla_attn.hip.cpp - Hy4-Flash MLA attention kernels, gfx1151 (see mla_attn.hip.h for the math and the API)
#include "mla_attn.hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

namespace hy4 {

int g_mla_force_simple = 0;
int g_mla_target_blocks = 192;

namespace {

constexpr int NT = 256;

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
// block (256 threads = 8 waves) reductions, red has 8 floats
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = wave_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) r += red[i];
    return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = wave_max(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) r = fmaxf(r, red[i]);
    return r;
}

// interleaved rope of pair i (dims 2i, 2i+1) of a 64-dim slice, float32 like the torch reference
__device__ __forceinline__ void rope_pair(float x0, float x1, int pos, int i, float theta, float& o0, float& o1) {
    float freq = 1.0f / powf(theta, (float)(2 * i) / 64.0f);
    float ang = (float)pos * freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

// ---- 1. q absorb + rope ------------------------------------------------------------------------------------------------------------------------------
template <int TT>
__global__ void __launch_bounds__(256) k_prep_q(const float* __restrict__ q_b, const __half* __restrict__ wk_b, __half* __restrict__ qa, int M, int NH, int p0, float theta) {
    const int h = blockIdx.x;
    const int t0 = blockIdx.y * TT;
    const int tid = threadIdx.x;
    __shared__ float qs[TT][MLA_NOPE];
    for (int idx = tid; idx < TT * MLA_NOPE; idx += 256) {
        int tt = idx / MLA_NOPE, i = idx % MLA_NOPE, t = t0 + tt;
        qs[tt][i] = t < M ? q_b[((size_t)t * NH + h) * MLA_QHEAD + i] : 0.f;
    }
    __syncthreads();
    float acc0[TT], acc1[TT];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) acc0[tt] = acc1[tt] = 0.f;
    const __half2* w2 = reinterpret_cast<const __half2*>(wk_b + (size_t)h * MLA_NOPE * MLA_KV_LORA);
    for (int i = 0; i < MLA_NOPE; ++i) {
        __half2 w = w2[(size_t)i * 256 + tid];
        float wx = __low2float(w), wy = __high2float(w);
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
            float q = qs[tt][i];
            acc0[tt] += q * wx;
            acc1[tt] += q * wy;
        }
    }
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
        int t = t0 + tt;
        if (t < M) *reinterpret_cast<__half2*>(qa + ((size_t)t * NH + h) * MLA_ROW + 2 * tid) = __floats2half2_rn(acc0[tt], acc1[tt]);
    }
    for (int idx = tid; idx < TT * 32; idx += 256) {
        int tt = idx / 32, i = idx % 32, t = t0 + tt;
        if (t >= M) continue;
        const float* qp = q_b + ((size_t)t * NH + h) * MLA_QHEAD + MLA_NOPE + 2 * i;
        float o0, o1;
        rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
        *reinterpret_cast<__half2*>(qa + ((size_t)t * NH + h) * MLA_ROW + MLA_KV_LORA + 2 * i) = __floats2half2_rn(o0, o1);
    }
}

// ---- 2. kv cache write -------------------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_write_kv(const float* __restrict__ kv_a, const float* __restrict__ norm_w, __half* __restrict__ cache, int cache_slots, int M, int p0,
                                                    int ring, float theta, float eps) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float red[8];
    const float* row = kv_a + (size_t)t * MLA_ROW;
    float v0 = row[2 * tid], v1 = row[2 * tid + 1];
    float ss = block_sum(v0 * v0 + v1 * v1, red);
    float inv = 1.0f / sqrtf(ss / (float)MLA_KV_LORA + eps);
    int pos = p0 + t;
    int slot = ring > 0 ? pos % ring : pos;
    __half* dst = cache + (size_t)slot * MLA_ROW;
    __half h0 = __float2half(v0 * inv * norm_w[2 * tid]), h1 = __float2half(v1 * inv * norm_w[2 * tid + 1]);
    dst[2 * tid] = h0;
    dst[2 * tid + 1] = h1;
    // transposed copy for the WMMA PV product: Vt[((slot>>2)*512 + c)*4 + (slot&3)]
    __half* vt = cache + (size_t)cache_slots * MLA_ROW;
    vt[((size_t)(slot >> 2) * MLA_KV_LORA + 2 * tid) * 4 + (slot & 3)] = h0;
    vt[((size_t)(slot >> 2) * MLA_KV_LORA + 2 * tid + 1) * 4 + (slot & 3)] = h1;
    if (tid < 32) {
        float o0, o1;
        rope_pair(row[MLA_KV_LORA + 2 * tid], row[MLA_KV_LORA + 2 * tid + 1], pos, tid, theta, o0, o1);
        *reinterpret_cast<__half2*>(dst + MLA_KV_LORA + 2 * tid) = __floats2half2_rn(o0, o1);
    }
}

// ---- 3. simple attention: one block per (head, query) -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_attend_simple(const __half* __restrict__ qa, const __half* __restrict__ cache, const float* __restrict__ sinks, __half* __restrict__ ctx,
                                                        MlaAttnArgs a) {
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    __shared__ float qs[MLA_ROW];
    __shared__ float sp[256];
    __shared__ int ssl[256];
    __shared__ float red[8];
    const __half* qrow = qa + ((size_t)t * a.NH + h) * MLA_ROW;
    for (int i = tid; i < MLA_ROW; i += 256) qs[i] = __half2float(qrow[i]);
    __syncthreads();
    const int pos = a.p0 + t;
    int nk, lo = 0;
    if (a.mode == MLA_CAUSAL) nk = pos + 1;
    else if (a.mode == MLA_WINDOW) {
        lo = max(0, pos - a.window + 1);
        nk = pos - lo + 1;
    } else nk = a.nkeys[t];

    float m = sinks[h];
    float l = 1.0f;
    float acc0 = 0.f, acc1 = 0.f;
    for (int base = 0; base < nk; base += 256) {
        const int j = base + tid;
        float s = -INFINITY;
        int slot = 0;
        if (j < nk) {
            if (a.mode == MLA_CAUSAL) slot = a.ring > 0 ? j % a.ring : j;
            else if (a.mode == MLA_WINDOW) slot = (lo + j) % a.ring;
            else slot = a.keys[(size_t)t * a.kmax + j];
            const __half2* kr = reinterpret_cast<const __half2*>(cache + (size_t)slot * MLA_ROW);
            float d = 0.f;
#pragma unroll 8
            for (int i = 0; i < MLA_ROW / 2; ++i) {
                __half2 kk = kr[i];
                d += qs[2 * i] * __low2float(kk) + qs[2 * i + 1] * __high2float(kk);
            }
            s = d * a.scale;
        }
        ssl[tid] = slot;
        float tmax = block_max(s, red);
        float nm = fmaxf(m, tmax);
        float corr = expf(m - nm);
        float p = (j < nk) ? expf(s - nm) : 0.f;
        sp[tid] = p;
        float psum = block_sum(p, red);
        l = l * corr + psum;
        m = nm;
        acc0 *= corr;
        acc1 *= corr;
        __syncthreads();
        const int cnt = min(256, nk - base);
        for (int jj = 0; jj < cnt; ++jj) {
            float pj = sp[jj];
            const __half2* vr = reinterpret_cast<const __half2*>(cache + (size_t)ssl[jj] * MLA_ROW);
            __half2 v = vr[tid];
            acc0 += pj * __low2float(v);
            acc1 += pj * __high2float(v);
        }
        __syncthreads();
    }
    float il = 1.0f / l;
    __half* o = ctx + ((size_t)t * a.NH + h) * MLA_KV_LORA;
    *reinterpret_cast<__half2*>(o + 2 * tid) = __floats2half2_rn(acc0 * il, acc1 * il);
}

// ---- 3b. WMMA flash attention ------------------------------------------------------------------------------------------------------------------------
// Block = 4 waves, handles ONE query token and ONE tile of 16 heads (the 16 rows of the WMMA tile share the key set), keys [tile range] of a split.
// Per 64-key tile: wave w scores keys 16w..16w+15 (S = Q K^T, 36 WMMA over the 576 dims, B fragments read straight from the row-major cache), the row max / row sums are
// exchanged through LDS, P (fp16) goes to LDS, then every wave multiplies P into ITS 128-column slice of the 512-wide latent output (V^T fragments read from the Vt copy of the cache).
// Online softmax state (m, l) is kept identically by all waves; the per-head sink is the initial state of split 0 (m = sink, l = 1).
typedef _Float16 hv16 __attribute__((ext_vector_type(16)));
typedef float fv8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ hv16 ld_hv16(const void* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *(reinterpret_cast<const uint4*>(p) + 1);
    u32x8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hv16, v);
}
__device__ __forceinline__ hv16 pack_hv16(uint2 a, uint2 b, uint2 c, uint2 d) {
    u32x8 v = {a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
    return __builtin_bit_cast(hv16, v);
}

constexpr int FBK = 64;   // keys per tile
constexpr int QP = 584;   // sQ pitch (halves)
constexpr int PP = 72;    // sP pitch (halves)
constexpr float NEG = -1e30f;

__global__ void __launch_bounds__(128) k_attend_wmma(const __half* __restrict__ qa, const __half* __restrict__ cache, const float* __restrict__ sinks, __half* __restrict__ out,
                                                      float* __restrict__ part, MlaAttnArgs a, int S, int tps) {
    const int rt = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int NH = a.NH, RT = NH / 16, h0 = rt * 16;
    __shared__ __align__(16) _Float16 sQ[16 * QP];
    __shared__ __align__(16) _Float16 sP[16 * PP];
    __shared__ float s_m[4][16], s_l[4][16];
    {
        const _Float16* qsrc = reinterpret_cast<const _Float16*>(qa) + ((size_t)t * NH + h0) * MLA_ROW;
        for (int idx = tid; idx < 16 * (MLA_ROW / 8); idx += 128) {
            int r = idx / (MLA_ROW / 8), c = (idx % (MLA_ROW / 8)) * 8;
            *reinterpret_cast<uint4*>(&sQ[r * QP + c]) = *reinterpret_cast<const uint4*>(qsrc + (size_t)r * MLA_ROW + c);
        }
    }
    const int pos = a.p0 + t;
    const int mode = a.mode;
    int lo = 0, lo_al = 0, nk;
    if (mode == MLA_CAUSAL) nk = pos + 1;
    else if (mode == MLA_WINDOW) {
        lo = max(0, pos - a.window + 1);
        lo_al = lo & ~15;
        nk = pos - lo_al + 1;
    } else nk = a.nkeys[t];
    const int ntiles = (nk + FBK - 1) / FBK;
    const int tb = sp * tps, te = min(ntiles, tb + tps);
    const int32_t* klist = a.keys + (size_t)t * a.kmax;
    const _Float16* rows = reinterpret_cast<const _Float16*>(cache);
    const _Float16* vt = rows + (size_t)a.cache_slots * MLA_ROW;

    float mrun[8], lrun[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        mrun[i] = sp == 0 ? sinks[h0 + 2 * i + hi] : NEG;
        lrun[i] = sp == 0 ? 1.0f : 0.0f;
    }
    fv8 o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = fv8{};
    __syncthreads();

    for (int tile = tb; tile < te; ++tile) {
        const int i0 = tile * FBK;
        // ---- scores of this wave's 16 keys
        const int ie = i0 + 16 * w + l16;
        bool v = ie < nk;
        if (mode == MLA_WINDOW) v = v && (lo_al + ie >= lo);
        int slot = 0;
        if (ie < nk) slot = mode == MLA_CAUSAL ? (a.ring > 0 ? ie % a.ring : ie) : (mode == MLA_WINDOW ? (lo_al + ie) % a.ring : klist[ie]);
        const _Float16* krow = rows + (size_t)slot * MLA_ROW;
        fv8 sacc = fv8{};
#pragma unroll 12
        for (int kk = 0; kk < MLA_ROW / 16; ++kk) {
            hv16 af = ld_hv16(&sQ[l16 * QP + kk * 16]);
            hv16 bf = ld_hv16(krow + kk * 16);
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, sacc);
        }
        float sc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) sc[i] = v ? sacc[i] * a.scale : NEG;
        float pm[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = sc[i];
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) x = fmaxf(x, __shfl_xor(x, off, 16));
            pm[i] = x;
        }
        if (l16 == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) s_m[w][2 * i + hi] = pm[i];
        }
        // group base (slot >> 2) of the 16 key groups of the 4 PV k-steps (entry 4q of the tile)
        int gidx[16];
#pragma unroll
        for (int q = 0; q < 16; ++q) {
            int e = i0 + 4 * q;
            int g = 0;
            if (e < nk) {
                int sl = mode == MLA_CAUSAL ? (a.ring > 0 ? e % a.ring : e) : (mode == MLA_WINDOW ? (lo_al + e) % a.ring : klist[e]);
                g = sl >> 2;
            }
            gidx[q] = g;
        }
        __syncthreads();
        float corr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = 2 * i + hi;
            float tm = fmaxf(fmaxf(s_m[0][r], s_m[1][r]), fmaxf(s_m[2][r], s_m[3][r]));
            float nm = fmaxf(mrun[i], tm);
            corr[i] = expf(mrun[i] - nm);
            mrun[i] = nm;
        }
        float ps[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p = v ? expf(sc[i] - mrun[i]) : 0.f;
            sP[(2 * i + hi) * PP + 16 * w + l16] = (_Float16)p;
            float x = p;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) x += __shfl_xor(x, off, 16);
            ps[i] = x;
        }
        if (l16 == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) s_l[w][2 * i + hi] = ps[i];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = 2 * i + hi;
            lrun[i] = lrun[i] * corr[i] + (s_l[0][r] + s_l[1][r] + s_l[2][r] + s_l[3][r]);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[c][i] *= corr[i];
        // ---- P @ V  (this wave: columns 128w .. 128w+127)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            hv16 pf = ld_hv16(&sP[l16 * PP + ks * 16]);
#pragma unroll
            for (int ct = 0; ct < 8; ++ct) {
                const int c = 128 * w + 16 * ct + l16;
                uint2 b0 = *reinterpret_cast<const uint2*>(vt + ((size_t)gidx[4 * ks + 0] * MLA_KV_LORA + c) * 4);
                uint2 b1 = *reinterpret_cast<const uint2*>(vt + ((size_t)gidx[4 * ks + 1] * MLA_KV_LORA + c) * 4);
                uint2 b2 = *reinterpret_cast<const uint2*>(vt + ((size_t)gidx[4 * ks + 2] * MLA_KV_LORA + c) * 4);
                uint2 b3 = *reinterpret_cast<const uint2*>(vt + ((size_t)gidx[4 * ks + 3] * MLA_KV_LORA + c) * 4);
                o[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pf, pack_hv16(b0, b1, b2, b3), o[ct]);
            }
        }
        __syncthreads();
    }
    // ---- write
    if (S == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float il = 1.0f / lrun[i];
            __half* dst = out + ((size_t)t * NH + h0 + 2 * i + hi) * MLA_KV_LORA + 128 * w + l16;
#pragma unroll
            for (int ct = 0; ct < 8; ++ct) dst[16 * ct] = __float2half(o[ct][i] * il);
        }
    } else {
        const size_t blk = ((size_t)t * RT + rt) * S + sp;
        float* po = part + blk * (16 * MLA_KV_LORA + 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float* dst = po + (size_t)(2 * i + hi) * MLA_KV_LORA + 128 * w + l16;
#pragma unroll
            for (int ct = 0; ct < 8; ++ct) dst[16 * ct] = o[ct][i];
        }
        if (w == 0 && l16 == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                po[16 * MLA_KV_LORA + (2 * i + hi) * 2] = mrun[i];
                po[16 * MLA_KV_LORA + (2 * i + hi) * 2 + 1] = lrun[i];
            }
        }
    }
}

// merge the split-K partials: grid (16 rows, RT, M), 256 threads (2 columns each)
__global__ void __launch_bounds__(256) k_combine(const float* __restrict__ part, __half* __restrict__ out, int NH, int S) {
    const int r = blockIdx.x, rt = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const int RT = NH / 16;
    __shared__ float sm[128], sl[128], sw[128];
    const size_t blk0 = ((size_t)t * RT + rt) * S;
    constexpr size_t PB = 16 * MLA_KV_LORA + 32;
    if (tid < S) {
        const float* ml = part + (blk0 + tid) * PB + 16 * MLA_KV_LORA + r * 2;
        sm[tid] = ml[0];
        sl[tid] = ml[1];
    }
    __syncthreads();
    float mx = NEG;
    for (int s = 0; s < S; ++s) mx = fmaxf(mx, sm[s]);
    if (tid < S) sw[tid] = expf(sm[tid] - mx);
    __syncthreads();
    float den = 0.f;
    for (int s = 0; s < S; ++s) den += sw[s] * sl[s];
    const float id = 1.0f / den;
    float a0 = 0.f, a1 = 0.f;
    const float* p = part + blk0 * PB + (size_t)r * MLA_KV_LORA + 2 * tid;
#pragma unroll 8
    for (int s = 0; s < S; ++s) {
        float2 v = *reinterpret_cast<const float2*>(p + (size_t)s * PB);
        const float wv = sw[s];
        a0 += wv * v.x;
        a1 += wv * v.y;
    }
    *reinterpret_cast<__half2*>(out + ((size_t)t * NH + rt * 16 + r) * MLA_KV_LORA + 2 * tid) = __floats2half2_rn(a0 * id, a1 * id);
}

int choose_splits(int M, int NH, int max_keys) {
    const int RT = NH / 16;
    const int base = M * RT;
    if (base >= g_mla_target_blocks) return 1;
    const int ntiles = (max_keys + FBK - 1) / FBK;
    int S = (g_mla_target_blocks + base - 1) / base;
    S = std::min(S, std::max(1, ntiles / 2));
    S = std::min(S, 128);
    return std::max(S, 1);
}

// ---- 1b/4b. WMMA q-absorb and v-up projections (prefill / verify windows; per-head GEMMs on the matrix cores) ---------------------------------------------------
// wk_bt: half [NH][512][192]  (wk_b transposed so that a B fragment = 16 contiguous values of the 192-dim contraction axis)
__global__ void k_transpose_wk(const __half* __restrict__ wk_b, __half* __restrict__ wk_bt, int NH) {
    const size_t n = (size_t)NH * MLA_NOPE * MLA_KV_LORA;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        size_t c = idx % MLA_KV_LORA;
        size_t i = (idx / MLA_KV_LORA) % MLA_NOPE;
        size_t h = idx / ((size_t)MLA_KV_LORA * MLA_NOPE);
        wk_bt[(h * MLA_KV_LORA + c) * MLA_NOPE + i] = wk_b[idx];
    }
}

__device__ __forceinline__ hv16 ld_f32_as_hv16(const float* p) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *(reinterpret_cast<const float4*>(p) + 1), c = *(reinterpret_cast<const float4*>(p) + 2),
                 d = *(reinterpret_cast<const float4*>(p) + 3);
    hv16 r;
    r[0] = (_Float16)a.x; r[1] = (_Float16)a.y; r[2] = (_Float16)a.z; r[3] = (_Float16)a.w;
    r[4] = (_Float16)b.x; r[5] = (_Float16)b.y; r[6] = (_Float16)b.z; r[7] = (_Float16)b.w;
    r[8] = (_Float16)c.x; r[9] = (_Float16)c.y; r[10] = (_Float16)c.z; r[11] = (_Float16)c.w;
    r[12] = (_Float16)d.x; r[13] = (_Float16)d.y; r[14] = (_Float16)d.z; r[15] = (_Float16)d.w;
    return r;
}

// ---- 1b/4b. per-head tiled WMMA GEMMs: q absorb (MODE 0, K=192, N=512) and v up-projection (MODE 1, K=512, N=256) ------------------------------------------------------
// C[t][n] = sum_k A[t][k] * B_h[n][k] for head h = blockIdx.z. Block tile 128 tokens x 64 outputs, 8 waves (4 x 2), wave tile 32 x 32 (2 x 2 WMMA tiles); the B chunk (64 rows x 64 k)
// goes through LDS, the A fragments are read straight from global (q_b fp32 -> fp16 for MODE 0, the half latent context for MODE 1).
constexpr int HG_BM = 128, HG_BN = 64, HG_BK = 64, HG_LD = HG_BK + 8;

template <int MODE>
__global__ void __launch_bounds__(256) k_head_gemm(const void* __restrict__ A_, const __half* __restrict__ Bw, const float* __restrict__ gate, __half* __restrict__ out, int M, int NH, int p0,
                                                    float theta) {
    constexpr int K = MODE == 0 ? MLA_NOPE : MLA_KV_LORA;
    constexpr int N = MODE == 0 ? MLA_KV_LORA : MLA_VHEAD;
    __shared__ __align__(16) _Float16 sB[HG_BN * HG_LD];
    __shared__ __align__(16) _Float16 sA[HG_BM * HG_LD];
    const int h = blockIdx.z, t0 = blockIdx.y * HG_BM, n0 = blockIdx.x * HG_BN;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 3, wn = wave >> 2;
    const _Float16* B = reinterpret_cast<const _Float16*>(Bw) + ((size_t)h * N + n0) * K;
    fv8 acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[a][j] = fv8{};
    for (int k0 = 0; k0 < K; k0 += HG_BK) {
        for (int idx = tid; idx < HG_BN * 8; idx += 256) {
            const int r = idx >> 3, c8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&sB[r * HG_LD + c8]) = *reinterpret_cast<const uint4*>(B + (size_t)r * K + k0 + c8);
        }
        if (MODE == 0) {  // fp32 q_nope rows -> half
            for (int idx = tid; idx < HG_BM * 16; idx += 256) {
                const int r = idx >> 4, c4 = (idx & 15) * 4;
                const int t = min(t0 + r, M - 1);
                const float4 v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(A_) + ((size_t)t * NH + h) * MLA_QHEAD + k0 + c4);
                _Float16* d = &sA[r * HG_LD + c4];
                d[0] = (_Float16)v.x; d[1] = (_Float16)v.y; d[2] = (_Float16)v.z; d[3] = (_Float16)v.w;
            }
        } else {
            for (int idx = tid; idx < HG_BM * 8; idx += 256) {
                const int r = idx >> 3, c8 = (idx & 7) * 8;
                const int t = min(t0 + r, M - 1);
                *reinterpret_cast<uint4*>(&sA[r * HG_LD + c8]) =
                    *reinterpret_cast<const uint4*>(reinterpret_cast<const _Float16*>(A_) + ((size_t)t * NH + h) * MLA_KV_LORA + k0 + c8);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < HG_BK / 16; ++kk) {
            hv16 a0 = ld_hv16(&sA[(wm * 32 + l16) * HG_LD + kk * 16]);
            hv16 a1 = ld_hv16(&sA[(wm * 32 + 16 + l16) * HG_LD + kk * 16]);
            hv16 b0 = ld_hv16(&sB[(wn * 32 + l16) * HG_LD + kk * 16]);
            hv16 b1 = ld_hv16(&sB[(wn * 32 + 16 + l16) * HG_LD + kk * 16]);
            acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc[0][0]);
            acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, acc[0][1]);
            acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, acc[1][0]);
            acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc[1][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = t0 + wm * 32 + 16 * a + 2 * e + hi;
                if (t >= M) continue;
                const int n = n0 + wn * 32 + 16 * j + l16;
                float v = acc[a][j][e];
                if (MODE == 0) {
                    out[((size_t)t * NH + h) * MLA_ROW + n] = __float2half(v);
                } else {
                    const size_t gi = (size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + n;
                    if (gate) v *= 1.0f / (1.0f + expf(-gate[gi]));
                    out[gi] = __float2half(v);
                }
            }
    if (MODE == 0 && blockIdx.x == 0) {  // rope part of q for this token tile
        const float* q_b = reinterpret_cast<const float*>(A_);
        for (int idx = tid; idx < HG_BM * 32; idx += 256) {
            const int tt = idx / 32, i = idx % 32, t = t0 + tt;
            if (t >= M) continue;
            const float* qp = q_b + ((size_t)t * NH + h) * MLA_QHEAD + MLA_NOPE + 2 * i;
            float o0, o1;
            rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
            *reinterpret_cast<__half2*>(out + ((size_t)t * NH + h) * MLA_ROW + MLA_KV_LORA + 2 * i) = __floats2half2_rn(o0, o1);
        }
    }
}
// ---- 4. v up-projection + gate -----------------------------------------------------------------------------------------------------------------------
template <int TT>
__global__ void __launch_bounds__(256) k_vup(const __half* __restrict__ ctx, const __half* __restrict__ wv_b, const float* __restrict__ gate, __half* __restrict__ o, int M, int NH) {
    const int h = blockIdx.x, t0 = blockIdx.y * TT, tid = threadIdx.x;
    __shared__ float cs[TT][MLA_KV_LORA];
    for (int idx = tid; idx < TT * MLA_KV_LORA; idx += 256) {
        int tt = idx / MLA_KV_LORA, c = idx % MLA_KV_LORA, t = t0 + tt;
        cs[tt][c] = t < M ? __half2float(ctx[((size_t)t * NH + h) * MLA_KV_LORA + c]) : 0.f;
    }
    __syncthreads();
    const int w = tid >> 5, lane = tid & 31;
    for (int j = w; j < MLA_VHEAD; j += 8) {
        const __half2* row = reinterpret_cast<const __half2*>(wv_b + ((size_t)h * MLA_VHEAD + j) * MLA_KV_LORA);
        float acc[TT];
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) acc[tt] = 0.f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int c2 = lane + 32 * r;
            __half2 wv = row[c2];
            float wx = __low2float(wv), wy = __high2float(wv);
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) acc[tt] += cs[tt][2 * c2] * wx + cs[tt][2 * c2 + 1] * wy;
        }
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) acc[tt] = wave_sum(acc[tt]);
        if (lane == 0) {
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                int t = t0 + tt;
                if (t < M) {
                    float v = acc[tt];
                    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + j]));
                    o[(size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + j] = __float2half(v);
                }
            }
        }
    }
}

// ---- 5. decode variants that read the QUANTIZED kv_b_proj directly (M = 1..8): no fp16 absorbed copies needed, ~3x fewer weight bytes per token ---------------------------
// kv_b_proj is an S_K tensor [NH*448][512] (2 super-blocks per row). Per-lane decode of 8 weights is the same packed-fp16 scheme as the S_K GEMV kernels
// (integer q as the low mantissa of 0x6400 patterns, exact (q-H) in fp16; the sub-group scale eff = D*u/255 is applied in fp32).
namespace dq {
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t ld16u(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
__device__ __forceinline__ uint32_t ld32u2(const uint8_t* p) { return ld16u(p) | (ld16u(p + 2) << 16); }
template <int FMT> struct Info;
template <> struct Info<2> { static constexpr int SB = 70; static constexpr float H = 1.5f; };
template <> struct Info<3> { static constexpr int SB = 102; static constexpr float H = 3.5f; };
template <> struct Info<4> { static constexpr int SB = 134; static constexpr float H = 7.5f; };
template <> struct Info<5> { static constexpr int SB = 166; static constexpr float H = 15.5f; };
template <> struct Info<6> { static constexpr int SB = 198; static constexpr float H = 31.5f; };
template <> struct Info<8> { static constexpr int SB = 262; static constexpr float H = 0.0f; };

// the 8 weights 8u..8u+7 of a super-block payload as 8 floats (q-H), not yet scaled
template <int FMT>
__device__ __forceinline__ void decode8(const uint8_t* pl, int u, float (&w)[8]) {
    uint32_t q[8];
    if (FMT == 2) {
        const uint32_t x = ld16u(pl + 2 * u);
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = (x >> (2 * k)) & 3u;
    } else if (FMT == 3) {
        const uint32_t x = ld16u(pl + 2 * u), hb = pl[64 + u];
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((x >> (2 * k)) & 3u) | (((hb >> k) & 1u) << 2);
    } else if (FMT == 4) {
        const uint32_t x = ld32u2(pl + 4 * u);
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = (x >> (4 * k)) & 15u;
    } else if (FMT == 5) {
        const uint32_t x = ld32u2(pl + 4 * u), hb = pl[128 + u];
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((x >> (4 * k)) & 15u) | (((hb >> k) & 1u) << 4);
    } else if (FMT == 6) {
        const uint32_t x = ld32u2(pl + 4 * u), h2 = ld16u(pl + 128 + 2 * u);
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((x >> (4 * k)) & 15u) | (((h2 >> (2 * k)) & 3u) << 4);
    } else {
        const uint32_t d0 = ld32u2(pl + 8 * u), d1 = ld32u2(pl + 8 * u + 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            q[k] = ((d0 >> (8 * k)) & 0xFFu) ^ 0x80u;
            q[4 + k] = ((d1 >> (8 * k)) & 0xFFu) ^ 0x80u;
        }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = FMT == 8 ? (float)((int)q[k] - 128) : ((float)q[k] - Info<FMT>::H);
}

__device__ __forceinline__ float sb_scale(const uint8_t* blk, int lane) {
    uint32_t dbits = ld16u(blk);
    __half dh = __builtin_bit_cast(__half, (unsigned short)dbits);
    return __half2float(dh) * ((float)blk[2 + (lane >> 3)] * (1.0f / 255.0f));
}
}  // namespace dq

// qa[t][h][0:512] = sum_{i<192} q_nope[t][h][i] * Wk[h*448+i][:]   (+ rope part). Block = (head), 8 waves, wave w owns rows 24w..24w+23; partial sums reduced through LDS.
template <int FMT, int MT>
__global__ void __launch_bounds__(256) k_prep_q_dec(const uint8_t* __restrict__ kvb, const float* __restrict__ q_b, __half* __restrict__ qa, int M, int NH, int p0, float theta) {
    constexpr int SB = dq::Info<FMT>::SB;
    const int h = blockIdx.x, tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    __shared__ float red[8][512 + 1];
    float acc[MT][2][8];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[m][s][k] = 0.f;
    for (int ii = 0; ii < 24; ++ii) {
        const int i = wave * 24 + ii;
        const uint8_t* row = kvb + ((size_t)h * 448 + i) * (2 * SB);
        float qv[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) qv[m] = m < M ? q_b[((size_t)m * NH + h) * MLA_QHEAD + i] : 0.f;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const uint8_t* blk = row + s * SB;
            const float eff = dq::sb_scale(blk, lane);
            float w[8];
            dq::decode8<FMT>(blk + 6, lane, w);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const float qe = qv[m] * eff;
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[m][s][k] += qe * w[k];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int k = 0; k < 8; ++k) red[wave][s * 256 + lane * 8 + k] = acc[m][s][k];
            __syncthreads();
            for (int c = tid; c < 512; c += 256) {
                float v = 0.f;
#pragma unroll
                for (int w2 = 0; w2 < 8; ++w2) v += red[w2][c];
                qa[((size_t)m * NH + h) * MLA_ROW + c] = __float2half(v);
            }
            __syncthreads();
        }
    }
    for (int idx = tid; idx < MT * 32; idx += 256) {
        const int m = idx / 32, i = idx % 32;
        if (m >= M) continue;
        const float* qp = q_b + ((size_t)m * NH + h) * MLA_QHEAD + MLA_NOPE + 2 * i;
        float o0, o1;
        rope_pair(qp[0], qp[1], p0 + m, i, theta, o0, o1);
        *reinterpret_cast<__half2*>(qa + ((size_t)m * NH + h) * MLA_ROW + MLA_KV_LORA + 2 * i) = __floats2half2_rn(o0, o1);
    }
}

// o[t][h][j] = sigmoid(gate) * sum_c Wv[h*448+192+j][c] * ctx[t][h][c]. grid (NH, 256/16), 8 waves x 2 rows
template <int FMT, int MT>
__global__ void __launch_bounds__(256) k_vup_dec(const uint8_t* __restrict__ kvb, const __half* __restrict__ ctx, const float* __restrict__ gate, __half* __restrict__ o, int M, int NH) {
    constexpr int SB = dq::Info<FMT>::SB;
    const int h = blockIdx.x, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j0 = (blockIdx.y * 8 + wave) * 2;
    float xv[MT][2][8];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const __half* xp = ctx + ((size_t)m * NH + h) * MLA_KV_LORA + s * 256 + lane * 8;
#pragma unroll
            for (int k = 0; k < 8; ++k) xv[m][s][k] = m < M ? __half2float(xp[k]) : 0.f;
        }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const uint8_t* row = kvb + ((size_t)h * 448 + 192 + j0 + r) * (2 * SB);
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.f;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const uint8_t* blk = row + s * SB;
            const float eff = dq::sb_scale(blk, lane);
            float w[8];
            dq::decode8<FMT>(blk + 6, lane, w);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float p = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) p += w[k] * xv[m][s][k];
                acc[m] += eff * p;
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[m];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0 && m < M) {
                const size_t gi = (size_t)m * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + j0 + r;
                if (gate) v *= 1.0f / (1.0f + expf(-gate[gi]));
                o[gi] = __float2half(v);
            }
        }
    }
}

template <int FMT>
static void launch_prep_dec(hipStream_t s, const void* kvb, const float* q_b, __half* qa, int M, int NH, int p0, float theta) {
    if (M <= 1) hipLaunchKernelGGL((k_prep_q_dec<FMT, 1>), dim3(NH), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else if (M <= 2) hipLaunchKernelGGL((k_prep_q_dec<FMT, 2>), dim3(NH), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else if (M <= 4) hipLaunchKernelGGL((k_prep_q_dec<FMT, 4>), dim3(NH), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else hipLaunchKernelGGL((k_prep_q_dec<FMT, 8>), dim3(NH), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
}
template <int FMT>
static void launch_vup_dec(hipStream_t s, const void* kvb, const __half* ctx, const float* gate, __half* o, int M, int NH) {
    dim3 grid(NH, MLA_VHEAD / 16);
    if (M <= 1) hipLaunchKernelGGL((k_vup_dec<FMT, 1>), grid, dim3(256), 0, s, (const uint8_t*)kvb, ctx, gate, o, M, NH);
    else if (M <= 2) hipLaunchKernelGGL((k_vup_dec<FMT, 2>), grid, dim3(256), 0, s, (const uint8_t*)kvb, ctx, gate, o, M, NH);
    else if (M <= 4) hipLaunchKernelGGL((k_vup_dec<FMT, 4>), grid, dim3(256), 0, s, (const uint8_t*)kvb, ctx, gate, o, M, NH);
    else hipLaunchKernelGGL((k_vup_dec<FMT, 8>), grid, dim3(256), 0, s, (const uint8_t*)kvb, ctx, gate, o, M, NH);
}
}  // namespace

size_t mla_workspace_bytes(int M, int NH, int max_keys) {
    const int S = choose_splits(M, NH, max_keys);
    if (S <= 1) return 0;
    return (size_t)M * (NH / 16) * S * (16 * MLA_KV_LORA + 32) * sizeof(float);
}

size_t mla_cache_bytes(int slots) { return (size_t)slots * (MLA_ROW + MLA_KV_LORA) * sizeof(__half); }

static int max_keys_of(const MlaAttnArgs& a) {
    if (a.mode == MLA_CAUSAL) return a.p0 + a.M;
    if (a.mode == MLA_WINDOW) return a.window + 16;
    return a.kmax;
}
size_t mla_workspace_bytes(const MlaAttnArgs& a) { return mla_workspace_bytes(a.M, a.NH, max_keys_of(a)); }

void mla_prep_q(hipStream_t s, const float* q_b, const __half* wk_b, __half* qa, int M, int NH, int p0, float theta) {
    constexpr int TT = 8;
    dim3 grid(NH, (M + TT - 1) / TT);
    hipLaunchKernelGGL((k_prep_q<TT>), grid, dim3(256), 0, s, q_b, wk_b, qa, M, NH, p0, theta);
}

void mla_write_kv(hipStream_t s, const float* kv_a, const float* norm_w, __half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps) {
    hipLaunchKernelGGL(k_write_kv, dim3(M), dim3(256), 0, s, kv_a, norm_w, cache, cache_slots, M, p0, ring, theta, eps);
}

void mla_attend(hipStream_t s, const __half* qa, const __half* cache, const float* sinks, __half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes) {
    const bool fast_ok = !g_mla_force_simple && (a.NH % 16 == 0) && a.cache_slots > 0 && (a.cache_slots % 16 == 0) && (a.mode != MLA_WINDOW || (a.ring > 0 && a.ring % 16 == 0)) &&
                         (a.mode != MLA_LIST || (a.kmax % 4 == 0));
    if (!fast_ok) {
        hipLaunchKernelGGL(k_attend_simple, dim3(a.NH, a.M), dim3(256), 0, s, qa, cache, sinks, ctx, a);
        return;
    }
    const int max_keys = max_keys_of(a);
    const int S = choose_splits(a.M, a.NH, max_keys);
    const int ntiles = (max_keys + FBK - 1) / FBK;
    const int tps = (ntiles + S - 1) / S;
    float* part = static_cast<float*>(ws);
    if (S > 1 && (ws == nullptr || ws_bytes < mla_workspace_bytes(a.M, a.NH, max_keys))) {
        hipLaunchKernelGGL(k_attend_simple, dim3(a.NH, a.M), dim3(256), 0, s, qa, cache, sinks, ctx, a);  // no workspace: fall back
        return;
    }
    hipLaunchKernelGGL(k_attend_wmma, dim3(a.NH / 16, S, a.M), dim3(128), 0, s, qa, cache, sinks, ctx, part, a, S, tps);
    if (S > 1) hipLaunchKernelGGL(k_combine, dim3(16, a.NH / 16, a.M), dim3(256), 0, s, part, ctx, a.NH, S);
}

void mla_vup(hipStream_t s, const __half* ctx, const __half* wv_b, const float* gate, __half* o, int M, int NH) {
    if (!g_mla_force_simple) {
        hipLaunchKernelGGL(k_head_gemm<1>, dim3(MLA_VHEAD / HG_BN, (M + HG_BM - 1) / HG_BM, NH), dim3(256), 0, s, (const void*)ctx, wv_b, gate, o, M, NH, 0, 0.f);
        return;
    }
    constexpr int TT = 8;
    dim3 grid(NH, (M + TT - 1) / TT);
    hipLaunchKernelGGL((k_vup<TT>), grid, dim3(256), 0, s, ctx, wv_b, gate, o, M, NH);
}

void mla_transpose_wk(hipStream_t s, const __half* wk_b, __half* wk_bt, int NH) {
    hipLaunchKernelGGL(k_transpose_wk, dim3(1024), dim3(256), 0, s, wk_b, wk_bt, NH);
}

void mla_prep_q_t(hipStream_t s, const float* q_b, const __half* wk_bt, __half* qa, int M, int NH, int p0, float theta) {
    hipLaunchKernelGGL(k_head_gemm<0>, dim3(MLA_KV_LORA / HG_BN, (M + HG_BM - 1) / HG_BM, NH), dim3(256), 0, s, (const void*)q_b, wk_bt, (const float*)nullptr, qa, M, NH, p0, theta);
}

void mla_prep_q_dec(hipStream_t s, int fmt, const void* kv_b, const float* q_b, __half* qa, int M, int NH, int p0, float theta) {
    switch (fmt) {
        case 2: launch_prep_dec<2>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
        case 3: launch_prep_dec<3>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
        case 4: launch_prep_dec<4>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
        case 5: launch_prep_dec<5>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
        case 6: launch_prep_dec<6>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
        default: launch_prep_dec<8>(s, kv_b, q_b, qa, M, NH, p0, theta); break;
    }
}

void mla_vup_dec(hipStream_t s, int fmt, const void* kv_b, const __half* ctx, const float* gate, __half* o, int M, int NH) {
    switch (fmt) {
        case 2: launch_vup_dec<2>(s, kv_b, ctx, gate, o, M, NH); break;
        case 3: launch_vup_dec<3>(s, kv_b, ctx, gate, o, M, NH); break;
        case 4: launch_vup_dec<4>(s, kv_b, ctx, gate, o, M, NH); break;
        case 5: launch_vup_dec<5>(s, kv_b, ctx, gate, o, M, NH); break;
        case 6: launch_vup_dec<6>(s, kv_b, ctx, gate, o, M, NH); break;
        default: launch_vup_dec<8>(s, kv_b, ctx, gate, o, M, NH); break;
    }
}

void mla_attention(hipStream_t s, const __half* qa, const __half* cache, const float* sinks, const __half* wv_b, const float* gate, __half* o, float* ctx_ws,
                   const MlaAttnArgs& a, void* ws, size_t ws_bytes) {
    mla_attend(s, qa, cache, sinks, reinterpret_cast<__half*>(ctx_ws), a, ws, ws_bytes);
    mla_vup(s, reinterpret_cast<const __half*>(ctx_ws), wv_b, gate, o, a.M, a.NH);
}

}  // namespace hy4
