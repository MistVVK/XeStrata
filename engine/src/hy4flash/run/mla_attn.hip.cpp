// mla_attn.hip.cpp - Hy4-Flash MLA attention kernels, gfx1151 (see mla_attn.hip.h for the math and the API)
#include "mla_attn.hip.h"

#include "hy4_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace hy4 {

int g_mla_force_simple = 0;
int g_mla_target_blocks = 192;
int g_mla_batch_invariant = 0;

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
    (void)cache_slots;
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

// V operand of the PV product: the 4 halves of a Vt group are the values of 4 consecutive slots; only the first v of them belong to keys of this query. The others (slots after the
// query's position: rows of later tokens of the same chunk, or stale rows) have probability exactly 0, but the matrix core's product alignment still sees their exponents, so they must
// not reach the instruction at all (otherwise row t of a chunk depends on the tokens after it, at 1e-7 level, and a verify window differs from one-token decode).
__device__ __forceinline__ uint2 vmask(uint2 b, int v) {
    if (v >= 4) return b;
    if (v <= 0) return make_uint2(0u, 0u);
    if (v == 1) return make_uint2(b.x & 0xFFFFu, 0u);
    if (v == 2) return make_uint2(b.x, 0u);
    return make_uint2(b.x, b.y & 0xFFFFu);
}
#if HY4_MMA && !HY4_CUDA
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

// (vmask: defined once above for both the gfx11 and the CUDA matrix-core paths; the merge of f-hy4-dfly2 a3bc436fc and f-hy4-cuda a9ed385dd had left a second copy here)
__device__ __forceinline__ fv8 fv8_zero() { return fv8{0, 0, 0, 0, 0, 0, 0, 0}; }
__device__ __forceinline__ int acc_m(int r, int lane) { return 2 * r + (lane >> 4); }
__device__ __forceinline__ int acc_n(int r, int lane) { return lane & 15; }
__device__ __forceinline__ fv8 mla_wmma_f16(hv16 a, hv16 b, fv8 c) {
#if defined(__HIP_DEVICE_COMPILE__) && !HY4_DEV_WMMA
    return c;  // unreachable on RDNA2 (host dispatches to k_attend_rdna2 / k_head_gemm_rdna2)
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}
#endif  // HY4_MMA && !HY4_CUDA
constexpr int FBK = 64;   // keys per tile
constexpr int QP = 584;   // sQ pitch (halves)
constexpr int PP = 72;    // sP pitch (halves)
constexpr float NEG = -1e30f;

#if !HY4_CUDA
// ---- 3a. RDNA2 Wave32 tiled fdot2 Flash-MLA (no WMMA, no Vt copy needed) -----------------------------------------------------------------------------
// Block = 4 waves (128 threads), handles ONE query token and ONE tile of 16 heads across 64-key tiles of a split.
// Phase 1 (Q @ K^T): tiles 64 keys x 64 dims into sK[64][68] (8.7 KB); each thread computes 4 heads x 2 keys (8 scores) using h2_dot_f32 (v_dot2_f32_f16).
// Phase 2 (P @ V):   tiles 64 keys x 64 latent cols transposed into sV[64][66] (8.4 KB, reusing sK, zeroing out-of-range keys on load -> exact causality);
//                    each wave handles 16 of the 64 cols (thread owns 4 heads x 2 cols), accumulating across all 8 chunks (512 cols) in registers!
// Total LDS: sQ (18.7 KB) + sKV (8.7 KB) + sP (2.3 KB) + s_m/s_l (0.5 KB) = 30.2 KB (2 blocks/WGP on RDNA2).
// Partial layout for S > 1 is identical to k_attend_wmma so k_combine works unchanged.
__global__ void __launch_bounds__(128) k_attend_rdna2(const __half* __restrict__ qa, const __half* __restrict__ cache, const float* __restrict__ sinks, __half* __restrict__ out,
                                                      float* __restrict__ part, MlaAttnArgs a, int S, int tps) {
    const int rt = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int NH = a.NH, RT = NH / 16, h0 = rt * 16;
    __shared__ __align__(16) _Float16 sQ[16 * QP];
    __shared__ __align__(16) _Float16 sP[16 * PP];
    __shared__ __align__(16) _Float16 sKV[64 * 68];  // reused as sK[64][68] (Phase 1) and sVt[64][66] (Phase 2)
    __shared__ int s_slot[64];
    __shared__ uint8_t s_valid[64];
    __shared__ float s_m[16][4], s_l[16][4];

    {
        const _Float16* qsrc = reinterpret_cast<const _Float16*>(qa) + ((size_t)t * NH + h0) * MLA_ROW;
        for (int idx = tid; idx < 16 * (MLA_ROW / 8); idx += 128) {
            const int r = idx / (MLA_ROW / 8), c = (idx % (MLA_ROW / 8)) * 8;
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

    // Within each wave (32 lanes = 4 head-groups x 8 col/key-groups):
    //   hg = lane >> 3 (0..3) -> 4 heads h_loc = hg*4 + 0..3
    //   kg = lane & 7  (0..7) -> 2 keys (Phase 1: 16*w + kg*2 + 0..1) or 2 cols (Phase 2: 16*w + kg*2 + 0..1 within each 64-col chunk)
    const int hg = lane >> 3, kg = lane & 7;
    const int h_base = hg * 4;

    float mrun[4], lrun[4];
#pragma unroll
    for (int hi = 0; hi < 4; ++hi) {
        mrun[hi] = sp == 0 ? sinks[h0 + h_base + hi] : NEG;
        lrun[hi] = sp == 0 ? 1.0f : 0.0f;
    }
    // Output accumulator: 8 chunks of 64 cols x 4 heads x 2 cols = 64 floats per thread
    float o[8][4][2];
#pragma unroll
    for (int ch = 0; ch < 8; ++ch)
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) { o[ch][hi][0] = 0.f; o[ch][hi][1] = 0.f; }
    __syncthreads();

    for (int tile = tb; tile < te; ++tile) {
        const int i0 = tile * FBK;
        if (tid < 64) {
            const int ie = i0 + tid;
            bool v = ie < nk;
            if (mode == MLA_WINDOW) v = v && (lo_al + ie >= lo);
            int sl = 0;
            if (ie < nk) sl = mode == MLA_CAUSAL ? (a.ring > 0 ? ie % a.ring : ie) : (mode == MLA_WINDOW ? (lo_al + ie) % a.ring : klist[ie]);
            s_slot[tid] = sl;
            s_valid[tid] = v ? 1 : 0;
        }
        __syncthreads();

        // ---- Phase 1: S = Q @ K^T (16 heads x 64 keys, 9 chunks of 64 dims)
        float sc[4][2];
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) { sc[hi][0] = 0.f; sc[hi][1] = 0.f; }

        const int k_idx0 = 16 * w + kg * 2;
        const int k_idx1 = k_idx0 + 1;
        for (int d0 = 0; d0 < MLA_ROW; d0 += 64) {
            // 128 threads load 64 keys x 64 halves (512 uint4 loads = 4 per thread)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = tid + l * 128;  // 0..511 -> key = idx >> 3 (0..63), c8 = (idx & 7) * 8
                const int k = idx >> 3, c8 = (idx & 7) * 8;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (s_valid[k]) v = *reinterpret_cast<const uint4*>(rows + (size_t)s_slot[k] * MLA_ROW + d0 + c8);
                *reinterpret_cast<uint4*>(&sKV[k * 68 + c8]) = v;
            }
            __syncthreads();
            const uint32_t* kp0 = reinterpret_cast<const uint32_t*>(&sKV[k_idx0 * 68]);
            const uint32_t* kp1 = reinterpret_cast<const uint32_t*>(&sKV[k_idx1 * 68]);
            const uint32_t* qp0 = reinterpret_cast<const uint32_t*>(&sQ[(h_base + 0) * QP + d0]);
            const uint32_t* qp1 = reinterpret_cast<const uint32_t*>(&sQ[(h_base + 1) * QP + d0]);
            const uint32_t* qp2 = reinterpret_cast<const uint32_t*>(&sQ[(h_base + 2) * QP + d0]);
            const uint32_t* qp3 = reinterpret_cast<const uint32_t*>(&sQ[(h_base + 3) * QP + d0]);
#pragma unroll
            for (int d2 = 0; d2 < 32; ++d2) {
                const uint32_t b0 = kp0[d2], b1 = kp1[d2];
                const uint32_t a0 = qp0[d2], a1 = qp1[d2], a2 = qp2[d2], a3 = qp3[d2];
                sc[0][0] = h2_dot_f32(a0, b0, sc[0][0]); sc[0][1] = h2_dot_f32(a0, b1, sc[0][1]);
                sc[1][0] = h2_dot_f32(a1, b0, sc[1][0]); sc[1][1] = h2_dot_f32(a1, b1, sc[1][1]);
                sc[2][0] = h2_dot_f32(a2, b0, sc[2][0]); sc[2][1] = h2_dot_f32(a2, b1, sc[2][1]);
                sc[3][0] = h2_dot_f32(a3, b0, sc[3][0]); sc[3][1] = h2_dot_f32(a3, b1, sc[3][1]);
            }
            __syncthreads();
        }

        const bool v0 = s_valid[k_idx0] != 0;
        const bool v1 = s_valid[k_idx1] != 0;
        // ---- Online softmax max across wave's 16 keys (8 lanes in kg = 0..7)
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            sc[hi][0] = v0 ? sc[hi][0] * a.scale : NEG;
            sc[hi][1] = v1 ? sc[hi][1] * a.scale : NEG;
            float mx = fmaxf(sc[hi][0], sc[hi][1]);
            mx = fmaxf(mx, __shfl_xor(mx, 1, 8));
            mx = fmaxf(mx, __shfl_xor(mx, 2, 8));
            mx = fmaxf(mx, __shfl_xor(mx, 4, 8));
            if (kg == 0) s_m[h_base + hi][w] = mx;
        }
        __syncthreads();

        float corr[4];
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            const int r = h_base + hi;
            const float tm = fmaxf(fmaxf(s_m[r][0], s_m[r][1]), fmaxf(s_m[r][2], s_m[r][3]));
            const float nm = fmaxf(mrun[hi], tm);
            corr[hi] = expf(mrun[hi] - nm);
            mrun[hi] = nm;
            const float p0 = v0 ? expf(sc[hi][0] - nm) : 0.f;
            const float p1 = v1 ? expf(sc[hi][1] - nm) : 0.f;
            sP[r * PP + k_idx0] = (_Float16)p0;
            sP[r * PP + k_idx1] = (_Float16)p1;
            float ps = p0 + p1;
            ps += __shfl_xor(ps, 1, 8);
            ps += __shfl_xor(ps, 2, 8);
            ps += __shfl_xor(ps, 4, 8);
            if (kg == 0) s_l[r][w] = ps;
        }
        __syncthreads();

#pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            const int r = h_base + hi;
            lrun[hi] = lrun[hi] * corr[hi] + (s_l[r][0] + s_l[r][1] + s_l[r][2] + s_l[r][3]);
        }
#pragma unroll
        for (int ch = 0; ch < 8; ++ch)
#pragma unroll
            for (int hi = 0; hi < 4; ++hi) {
                o[ch][hi][0] *= corr[hi];
                o[ch][hi][1] *= corr[hi];
            }

        // ---- Phase 2: O += P @ V (8 chunks of 64 latent cols; V transposed into sKV[col][key], stride 66 halves = 33 half2s -> zero bank conflicts!)
        for (int ch = 0; ch < 8; ++ch) {
            const int c0 = ch * 64;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = tid + l * 128;
                const int k = idx >> 3, c8 = (idx & 7) * 8;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (s_valid[k]) v = *reinterpret_cast<const uint4*>(rows + (size_t)s_slot[k] * MLA_ROW + c0 + c8);
                const _Float16* hv = reinterpret_cast<const _Float16*>(&v);
#pragma unroll
                for (int e = 0; e < 8; ++e) sKV[(c8 + e) * 66 + k] = hv[e];
            }
            __syncthreads();
            const int col0 = 16 * w + kg * 2;
            const int col1 = col0 + 1;
            const uint32_t* vp0 = reinterpret_cast<const uint32_t*>(&sKV[col0 * 66]);
            const uint32_t* vp1 = reinterpret_cast<const uint32_t*>(&sKV[col1 * 66]);
            const uint32_t* pp0 = reinterpret_cast<const uint32_t*>(&sP[(h_base + 0) * PP]);
            const uint32_t* pp1 = reinterpret_cast<const uint32_t*>(&sP[(h_base + 1) * PP]);
            const uint32_t* pp2 = reinterpret_cast<const uint32_t*>(&sP[(h_base + 2) * PP]);
            const uint32_t* pp3 = reinterpret_cast<const uint32_t*>(&sP[(h_base + 3) * PP]);
#pragma unroll
            for (int k2 = 0; k2 < 32; ++k2) {
                const uint32_t b0 = vp0[k2], b1 = vp1[k2];
                const uint32_t a0 = pp0[k2], a1 = pp1[k2], a2 = pp2[k2], a3 = pp3[k2];
                o[ch][0][0] = h2_dot_f32(a0, b0, o[ch][0][0]); o[ch][0][1] = h2_dot_f32(a0, b1, o[ch][0][1]);
                o[ch][1][0] = h2_dot_f32(a1, b0, o[ch][1][0]); o[ch][1][1] = h2_dot_f32(a1, b1, o[ch][1][1]);
                o[ch][2][0] = h2_dot_f32(a2, b0, o[ch][2][0]); o[ch][2][1] = h2_dot_f32(a2, b1, o[ch][2][1]);
                o[ch][3][0] = h2_dot_f32(a3, b0, o[ch][3][0]); o[ch][3][1] = h2_dot_f32(a3, b1, o[ch][3][1]);
            }
            __syncthreads();
        }
    }

    // ---- write (exact same layout as k_attend_wmma)
    const int col_local = 16 * w + kg * 2;
    if (S == 1) {
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            const float il = 1.0f / lrun[hi];
            __half* dst = out + ((size_t)t * NH + h0 + h_base + hi) * MLA_KV_LORA + col_local;
#pragma unroll
            for (int ch = 0; ch < 8; ++ch) {
                dst[ch * 64] = __float2half(o[ch][hi][0] * il);
                dst[ch * 64 + 1] = __float2half(o[ch][hi][1] * il);
            }
        }
    } else {
        const size_t blk = ((size_t)t * RT + rt) * S + sp;
        float* po = part + blk * (16 * MLA_KV_LORA + 32);
#pragma unroll
        for (int hi = 0; hi < 4; ++hi) {
            float* dst = po + (size_t)(h_base + hi) * MLA_KV_LORA + col_local;
#pragma unroll
            for (int ch = 0; ch < 8; ++ch) {
                dst[ch * 64] = o[ch][hi][0];
                dst[ch * 64 + 1] = o[ch][hi][1];
            }
        }
        if (w == 0 && kg == 0) {
#pragma unroll
            for (int hi = 0; hi < 4; ++hi) {
                po[16 * MLA_KV_LORA + (h_base + hi) * 2] = mrun[hi];
                po[16 * MLA_KV_LORA + (h_base + hi) * 2 + 1] = lrun[hi];
            }
        }
    }
}
#endif  // !HY4_CUDA

#if HY4_MMA && !HY4_CUDA
__global__ void __launch_bounds__(128) k_attend_wmma(const __half* __restrict__ qa, const __half* __restrict__ cache, const float* __restrict__ sinks, __half* __restrict__ out,
                                                      float* __restrict__ part, MlaAttnArgs a, int S, int tps) {
    const int rt = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int NH = a.NH, RT = NH / 16, h0 = rt * 16;
    __shared__ __align__(16) _Float16 sQ[16 * QP];
    __shared__ __align__(16) _Float16 sP[16 * PP];
    __shared__ __align__(16) _Float16 sVt[MLA_KV_LORA * 18];  // in-LDS transposed V for 16 keys x 512 cols (18.0 KB; eliminates global Vt!)
    __shared__ int s_slot[64];
    __shared__ uint8_t s_valid[64];
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
        if (hi == 0) {
            s_slot[16 * w + l16] = slot;
            s_valid[16 * w + l16] = v ? 1 : 0;
        }
        const _Float16* krow = rows + (size_t)slot * MLA_ROW;
        fv8 sacc = fv8{};
#pragma unroll 12
        for (int kk = 0; kk < MLA_ROW / 16; ++kk) {
            hv16 af = ld_hv16(&sQ[l16 * QP + kk * 16]);
            hv16 bf = ld_hv16(krow + kk * 16);
            sacc = mla_wmma_f16(af, bf, sacc);
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
        // ---- P @ V: transpose 16 keys x 512 cols from row-major cache into sVt[512][18] (invalid keys zeroed -> exact vmask causality, no global Vt!)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // 128 threads load 16 keys x 512 halves = 1024 uint4 loads (8 per thread, coalesced)
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int idx = tid + l * 128;  // 0..1023 -> k16 = idx >> 6 (0..15), c8 = (idx & 63) * 8
                const int k16 = idx >> 6, c8 = (idx & 63) * 8;
                const int tk = 16 * ks + k16;
                uint4 v4 = make_uint4(0u, 0u, 0u, 0u);
                if (s_valid[tk]) v4 = *reinterpret_cast<const uint4*>(rows + (size_t)s_slot[tk] * MLA_ROW + c8);
                const _Float16* hv = reinterpret_cast<const _Float16*>(&v4);
#pragma unroll
                for (int e = 0; e < 8; ++e) sVt[(c8 + e) * 18 + k16] = hv[e];
            }
            __syncthreads();
            hv16 pf = ld_hv16(&sP[l16 * PP + ks * 16]);
#pragma unroll
            for (int ct = 0; ct < 8; ++ct) {
                const int c = 128 * w + 16 * ct + l16;
                hv16 vf = ld_hv16(&sVt[c * 18]);
                o[ct] = mla_wmma_f16(pf, vf, o[ct]);
            }
            __syncthreads();
        }
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

#elif HY4_MMA
// NVIDIA version of the flash kernel (mma.sync m16n8k16), in-LDS transposed V (eliminates global Vt copy).
__global__ void __launch_bounds__(128) k_attend_wmma(const __half* __restrict__ qa, const __half* __restrict__ cache, const float* __restrict__ sinks, __half* __restrict__ out,
                                                      float* __restrict__ part, MlaAttnArgs a, int S, int tps) {
    const int rt = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, c = lane & 3, c4 = c * 4;
    const int NH = a.NH, RT = NH / 16, h0 = rt * 16;
    __shared__ __align__(16) half_t sQ[16 * QP];
    __shared__ __align__(16) half_t sP[16 * PP];
    __shared__ __align__(16) half_t sVt[MLA_KV_LORA * 18];
    __shared__ int s_slot[64];
    __shared__ uint8_t s_valid[64];
    __shared__ float s_m[4][16], s_l[4][16];
    {
        const half_t* qsrc = reinterpret_cast<const half_t*>(qa) + ((size_t)t * NH + h0) * MLA_ROW;
        for (int idx = tid; idx < 16 * (MLA_ROW / 8); idx += 128) {
            int r = idx / (MLA_ROW / 8), cc = (idx % (MLA_ROW / 8)) * 8;
            *reinterpret_cast<uint4*>(&sQ[r * QP + cc]) = *reinterpret_cast<const uint4*>(qsrc + (size_t)r * MLA_ROW + cc);
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
    const half_t* rows = reinterpret_cast<const half_t*>(cache);

    float mrun[2], lrun[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        mrun[i] = sp == 0 ? sinks[h0 + g + 8 * i] : NEG;
        lrun[i] = sp == 0 ? 1.0f : 0.0f;
    }
    float oc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) oc[nt][e] = 0.f;
    __syncthreads();

    for (int tile = tb; tile < te; ++tile) {
        const int i0 = tile * FBK;
        if (tid < 64) {
            const int ie = i0 + tid;
            bool v = ie < nk;
            if (mode == MLA_WINDOW) v = v && (lo_al + ie >= lo);
            int sl = 0;
            if (ie < nk) sl = mode == MLA_CAUSAL ? (a.ring > 0 ? ie % a.ring : ie) : (mode == MLA_WINDOW ? (lo_al + ie) % a.ring : klist[ie]);
            s_slot[tid] = sl;
            s_valid[tid] = v ? 1 : 0;
        }
        __syncthreads();
        const half_t* kr[2];
#pragma unroll
        for (int s = 0; s < 2; ++s) kr[s] = rows + (size_t)s_slot[16 * w + 8 * s + g] * MLA_ROW;
        float sacc[2][4];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[s][e] = 0.f;
#pragma unroll 12
        for (int kk = 0; kk < MLA_ROW / 16; ++kk) {
            const uint2 alo = *reinterpret_cast<const uint2*>(&sQ[g * QP + kk * 16 + c4]);
            const uint2 ahi = *reinterpret_cast<const uint2*>(&sQ[(g + 8) * QP + kk * 16 + c4]);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint2 b = *reinterpret_cast<const uint2*>(kr[s] + kk * 16 + c4);
                mma_16816(sacc[s], alo.x, ahi.x, alo.y, ahi.y, b.x, b.y);
            }
        }
        float sc[2][4];
        bool vk[2][4];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int tk = 16 * w + 8 * s + 2 * c + (e & 1);
                bool v = s_valid[tk] != 0;
                vk[s][e] = v;
                sc[s][e] = v ? sacc[s][e] * a.scale : NEG;
            }
        float pm[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float x = fmaxf(fmaxf(sc[0][2 * h], sc[0][2 * h + 1]), fmaxf(sc[1][2 * h], sc[1][2 * h + 1]));
            x = fmaxf(x, __shfl_xor(x, 1));
            x = fmaxf(x, __shfl_xor(x, 2));
            pm[h] = x;
        }
        if (c == 0) {
            s_m[w][g] = pm[0];
            s_m[w][g + 8] = pm[1];
        }
        __syncthreads();
        float corr[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = g + 8 * h;
            const float tm = fmaxf(fmaxf(s_m[0][r], s_m[1][r]), fmaxf(s_m[2][r], s_m[3][r]));
            const float nm = fmaxf(mrun[h], tm);
            corr[h] = expf(mrun[h] - nm);
            mrun[h] = nm;
        }
        float ps[2] = {0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float p = vk[s][e] ? expf(sc[s][e] - mrun[h]) : 0.f;
                sP[(g + 8 * h) * PP + 16 * w + 8 * s + 2 * c + (e & 1)] = h_from_float(p);
                ps[h] += p;
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            ps[h] += __shfl_xor(ps[h], 1);
            ps[h] += __shfl_xor(ps[h], 2);
        }
        if (c == 0) {
            s_l[w][g] = ps[0];
            s_l[w][g + 8] = ps[1];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = g + 8 * h;
            lrun[h] = lrun[h] * corr[h] + (s_l[0][r] + s_l[1][r] + s_l[2][r] + s_l[3][r]);
        }
#pragma unroll
        for (int nt = 0; nt < 16; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) oc[nt][e] *= corr[e >> 1];
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int idx = tid + l * 128;
                const int k16 = idx >> 6, c8 = (idx & 63) * 8;
                const int tk = 16 * ks + k16;
                uint4 v4 = make_uint4(0u, 0u, 0u, 0u);
                if (s_valid[tk]) v4 = *reinterpret_cast<const uint4*>(rows + (size_t)s_slot[tk] * MLA_ROW + c8);
                const half_t* hv = reinterpret_cast<const half_t*>(&v4);
#pragma unroll
                for (int e = 0; e < 8; ++e) sVt[(c8 + e) * 18 + k16] = hv[e];
            }
            __syncthreads();
            const uint2 plo = *reinterpret_cast<const uint2*>(&sP[g * PP + ks * 16 + c4]);
            const uint2 phi = *reinterpret_cast<const uint2*>(&sP[(g + 8) * PP + ks * 16 + c4]);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int col = 128 * w + 8 * nt + g;
                const uint2 b = *reinterpret_cast<const uint2*>(&sVt[col * 18 + c4]);
                mma_16816(oc[nt], plo.x, phi.x, plo.y, phi.y, b.x, b.y);
            }
            __syncthreads();
        }
    }
    if (S == 1) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float il = 1.0f / lrun[h];
            __half* dst = out + ((size_t)t * NH + h0 + g + 8 * h) * MLA_KV_LORA + 128 * w + 2 * c;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                dst[8 * nt] = __float2half(oc[nt][2 * h] * il);
                dst[8 * nt + 1] = __float2half(oc[nt][2 * h + 1] * il);
            }
        }
    } else {
        const size_t blk = ((size_t)t * RT + rt) * S + sp;
        float* po = part + blk * (16 * MLA_KV_LORA + 32);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float* dst = po + (size_t)(g + 8 * h) * MLA_KV_LORA + 128 * w + 2 * c;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                dst[8 * nt] = oc[nt][2 * h];
                dst[8 * nt + 1] = oc[nt][2 * h + 1];
            }
        }
        if (w == 0 && c == 0) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                po[16 * MLA_KV_LORA + (g + 8 * h) * 2] = mrun[h];
                po[16 * MLA_KV_LORA + (g + 8 * h) * 2 + 1] = lrun[h];
            }
        }
    }
}
#endif  // HY4_MMA

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
    const int base = ((g_mla_batch_invariant && M <= 16) ? 1 : M) * RT;  // batch invariant: the split count of a decode window is the one of a single token (same partition and summation order for every M)
    if (base >= g_mla_target_blocks) return 1;
    const int ntiles = (max_keys + FBK - 1) / FBK;
    int S = (g_mla_target_blocks + base - 1) / base;
    static const int div = getenv("HY4_MLA_SPLIT_DIV") ? std::max(1, atoi(getenv("HY4_MLA_SPLIT_DIV"))) : 2;  // tiles per split block (min): 2 = at least 2 tiles of 64 keys
    S = std::min(S, std::max(1, ntiles / div));
    S = std::min(S, 128);
    return std::max(S, 1);
}

// ---- 1b/4b. per-head q-absorb and v-up projections (prefill / verify windows) ---------------------------------------------------------------------------------
// wk_bt: half [NH][512][192]  (wk_b transposed so that a B fragment = contiguous contraction axis)
__global__ void k_transpose_wk(const __half* __restrict__ wk_b, __half* __restrict__ wk_bt, int NH) {
    const size_t n = (size_t)NH * MLA_NOPE * MLA_KV_LORA;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        size_t c = idx % MLA_KV_LORA;
        size_t i = (idx / MLA_KV_LORA) % MLA_NOPE;
        size_t h = idx / ((size_t)MLA_KV_LORA * MLA_NOPE);
        wk_bt[(h * MLA_KV_LORA + c) * MLA_NOPE + i] = wk_b[idx];
    }
}

#if !HY4_CUDA
// RDNA2 Wave32 fdot2 per-head tiled GEMM: 64 tokens x 64 outputs, BK=64, 256 threads (4x4 per thread).
template <int MODE>
__global__ void __launch_bounds__(256) k_head_gemm_rdna2(const void* __restrict__ A_, const __half* __restrict__ Bw, const float* __restrict__ gate, __half* __restrict__ out, int M, int NH, int p0,
                                                         float theta) {
    constexpr int K = MODE == 0 ? MLA_NOPE : MLA_KV_LORA;
    constexpr int N = MODE == 0 ? MLA_KV_LORA : MLA_VHEAD;
    constexpr int LD = 68;
    __shared__ __align__(16) _Float16 sA[64 * LD];
    __shared__ __align__(16) _Float16 sB[64 * LD];
    const int h = blockIdx.z, t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int ty = tid >> 4, tx = tid & 15;
    const _Float16* B = reinterpret_cast<const _Float16*>(Bw) + ((size_t)h * N + n0) * K;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        // load sB[64][64] (512 uint4 loads -> 2 per thread)
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int idx = tid + l * 256, r = idx >> 3, c8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&sB[r * LD + c8]) = *reinterpret_cast<const uint4*>(B + (size_t)r * K + k0 + c8);
        }
        if constexpr (MODE == 0) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = tid + l * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                const int t = min(t0 + r, M - 1);
                const float4 v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(A_) + ((size_t)t * NH + h) * MLA_QHEAD + k0 + c4);
                _Float16* d = &sA[r * LD + c4];
                d[0] = (_Float16)v.x; d[1] = (_Float16)v.y; d[2] = (_Float16)v.z; d[3] = (_Float16)v.w;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int idx = tid + l * 256, r = idx >> 3, c8 = (idx & 7) * 8;
                const int t = min(t0 + r, M - 1);
                *reinterpret_cast<uint4*>(&sA[r * LD + c8]) = *reinterpret_cast<const uint4*>(reinterpret_cast<const _Float16*>(A_) + ((size_t)t * NH + h) * MLA_KV_LORA + k0 + c8);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            uint32_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = reinterpret_cast<const uint32_t*>(&sA[(ty * 4 + i) * LD])[kk];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = reinterpret_cast<const uint32_t*>(&sB[(tx + j * 16) * LD])[kk];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = h2_dot_f32(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty * 4 + i;
        if (t >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + j * 16;
            float v = acc[i][j];
            if constexpr (MODE == 0) {
                out[((size_t)t * NH + h) * MLA_ROW + n] = __float2half(v);
            } else {
                const size_t gi = (size_t)t * NH * MLA_VHEAD + (size_t)h * MLA_VHEAD + n;
                if (gate) v *= 1.0f / (1.0f + expf(-gate[gi]));
                out[gi] = __float2half(v);
            }
        }
    }
    if constexpr (MODE == 0) {
        if (blockIdx.x == 0) {
            const float* q_b = reinterpret_cast<const float*>(A_);
            for (int idx = tid; idx < 64 * 32; idx += 256) {
                const int tt = idx / 32, i = idx % 32, t = t0 + tt;
                if (t >= M) continue;
                const float* qp = q_b + ((size_t)t * NH + h) * MLA_QHEAD + MLA_NOPE + 2 * i;
                float o0, o1;
                rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
                *reinterpret_cast<__half2*>(out + ((size_t)t * NH + h) * MLA_ROW + MLA_KV_LORA + 2 * i) = __floats2half2_rn(o0, o1);
            }
        }
    }
}
#endif  // !HY4_CUDA

#if HY4_MMA

#if !HY4_CUDA
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
#endif

// ---- 1b/4b. per-head tiled WMMA GEMMs: q absorb (MODE 0, K=192, N=512) and v up-projection (MODE 1, K=512, N=256) ------------------------------------------------------
// C[t][n] = sum_k A[t][k] * B_h[n][k] for head h = blockIdx.z. Block tile 128 tokens x 64 outputs, 8 waves (4 x 2), wave tile 32 x 32 (2 x 2 WMMA tiles); the B chunk (64 rows x 64 k)
// goes through LDS, the A fragments are read straight from global (q_b fp32 -> fp16 for MODE 0, the half latent context for MODE 1).
constexpr int HG_BM = 128, HG_BN = 64, HG_BK = 64, HG_LD = HG_BK + 8;

template <int MODE>
__global__ void __launch_bounds__(256) k_head_gemm(const void* __restrict__ A_, const __half* __restrict__ Bw, const float* __restrict__ gate, __half* __restrict__ out, int M, int NH, int p0,
                                                    float theta) {
    constexpr int K = MODE == 0 ? MLA_NOPE : MLA_KV_LORA;
    constexpr int N = MODE == 0 ? MLA_KV_LORA : MLA_VHEAD;
    __shared__ __align__(16) half_t sB[HG_BN * HG_LD];
    __shared__ __align__(16) half_t sA[HG_BM * HG_LD];
    const int h = blockIdx.z, t0 = blockIdx.y * HG_BM, n0 = blockIdx.x * HG_BN;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 3, wn = wave >> 2;
    const half_t* B = reinterpret_cast<const half_t*>(Bw) + ((size_t)h * N + n0) * K;
    fv8 acc[2][2];
    const int g4 = lane >> 2, c4 = (lane & 3) * 4;
    (void)g4;
    (void)c4;
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[a][j] = fv8_zero();
    // software pipeline: the global loads of the next k chunk are issued before the WMMA loop of the current one (registers -> LDS after the loop's barrier), so their latency overlaps the matrix work
    constexpr int NA = MODE == 0 ? (HG_BM * 16) / 256 : (HG_BM * 8) / 256;  // A items per thread: float4 (MODE 0) or uint4 (MODE 1)
    uint4 vb[(HG_BN * 8) / 256];
    float4 va0[MODE == 0 ? NA : 1];
    uint4 va1[MODE == 0 ? 1 : NA];
    auto gload = [&](int k0) {
#pragma unroll
        for (int i = 0; i < (HG_BN * 8) / 256; ++i) {
            const int idx = tid + 256 * i, r = idx >> 3, c8 = (idx & 7) * 8;
            vb[i] = *reinterpret_cast<const uint4*>(B + (size_t)r * K + k0 + c8);
        }
        if (MODE == 0) {  // fp32 q_nope rows (converted to half when stored)
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + 256 * i, r = idx >> 4, c4 = (idx & 15) * 4;
                const int t = min(t0 + r, M - 1);
                va0[i % (MODE == 0 ? NA : 1)] = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(A_) + ((size_t)t * NH + h) * MLA_QHEAD + k0 + c4);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + 256 * i, r = idx >> 3, c8 = (idx & 7) * 8;
                const int t = min(t0 + r, M - 1);
                va1[i % (MODE == 0 ? 1 : NA)] = *reinterpret_cast<const uint4*>(reinterpret_cast<const half_t*>(A_) + ((size_t)t * NH + h) * MLA_KV_LORA + k0 + c8);
            }
        }
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += HG_BK) {
#pragma unroll
        for (int i = 0; i < (HG_BN * 8) / 256; ++i) {
            const int idx = tid + 256 * i, r = idx >> 3, c8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&sB[r * HG_LD + c8]) = vb[i];
        }
        if (MODE == 0) {
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + 256 * i, r = idx >> 4, c4 = (idx & 15) * 4;
                const float4 v = va0[i % (MODE == 0 ? NA : 1)];
                half_t* d = &sA[r * HG_LD + c4];
                d[0] = h_from_float(v.x); d[1] = h_from_float(v.y); d[2] = h_from_float(v.z); d[3] = h_from_float(v.w);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + 256 * i, r = idx >> 3, c8 = (idx & 7) * 8;
                *reinterpret_cast<uint4*>(&sA[r * HG_LD + c8]) = va1[i % (MODE == 0 ? 1 : NA)];
            }
        }
        __syncthreads();
        if (k0 + HG_BK < K) gload(k0 + HG_BK);
#if HY4_CUDA
#pragma unroll
        for (int kk = 0; kk < HG_BK / 16; ++kk) {
            uint2 a_lo[2], a_hi[2];
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                a_lo[a] = *reinterpret_cast<const uint2*>(&sA[(wm * 32 + 16 * a + g4) * HG_LD + kk * 16 + c4]);
                a_hi[a] = *reinterpret_cast<const uint2*>(&sA[(wm * 32 + 16 * a + g4 + 8) * HG_LD + kk * 16 + c4]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int sn = 0; sn < 2; ++sn) {
                    const uint2 b = *reinterpret_cast<const uint2*>(&sB[(wn * 32 + 16 * j + 8 * sn + g4) * HG_LD + kk * 16 + c4]);
#pragma unroll
                    for (int a = 0; a < 2; ++a) mma_16816(&acc[a][j].v[sn * 4], a_lo[a].x, a_hi[a].x, a_lo[a].y, a_hi[a].y, b.x, b.y);
                }
        }
#else
#pragma unroll
        for (int kk = 0; kk < HG_BK / 16; ++kk) {
            hv16 a0 = ld_hv16(&sA[(wm * 32 + l16) * HG_LD + kk * 16]);
            hv16 a1 = ld_hv16(&sA[(wm * 32 + 16 + l16) * HG_LD + kk * 16]);
            hv16 b0 = ld_hv16(&sB[(wn * 32 + l16) * HG_LD + kk * 16]);
            hv16 b1 = ld_hv16(&sB[(wn * 32 + 16 + l16) * HG_LD + kk * 16]);
            acc[0][0] = mla_wmma_f16(a0, b0, acc[0][0]);
            acc[0][1] = mla_wmma_f16(a0, b1, acc[0][1]);
            acc[1][0] = mla_wmma_f16(a1, b0, acc[1][0]);
            acc[1][1] = mla_wmma_f16(a1, b1, acc[1][1]);
        }
#endif
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = t0 + wm * 32 + 16 * a + acc_m(e, lane);
                if (t >= M) continue;
                const int n = n0 + wn * 32 + 16 * j + acc_n(e, lane);
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
#endif  // HY4_MMA
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
    __half dh = __ushort_as_half((unsigned short)dbits);
    return __half2float(dh) * ((float)blk[2 + (lane >> 3)] * (1.0f / 255.0f));
}
}  // namespace dq

// qa[t][h][0:512] = sum_{i<192} q_nope[t][h][i] * Wk[h*448+i][:]   (+ rope part). Block = (head), 8 waves, wave w owns rows 24w..24w+23; partial sums reduced through LDS.
template <int FMT, int MT>
__global__ void __launch_bounds__(256) k_prep_q_dec(const uint8_t* __restrict__ kvb, const float* __restrict__ q_b, __half* __restrict__ qa, int M, int NH, int p0, float theta) {
    // grid (NH, 2): block (h, s) handles the 256 latent columns of super-block s (the two halves of every row are independent: same per-column summation order as one block doing both)
    constexpr int SB = dq::Info<FMT>::SB;
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    __shared__ float red[8][256 + 1];
    float acc[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[m][k] = 0.f;
    for (int ii = 0; ii < 24; ++ii) {
        const int i = wave * 24 + ii;
        const uint8_t* row = kvb + ((size_t)h * 448 + i) * (2 * SB);
        float qv[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) qv[m] = m < M ? q_b[((size_t)m * NH + h) * MLA_QHEAD + i] : 0.f;
        const uint8_t* blk = row + s * SB;
        const float eff = dq::sb_scale(blk, lane);
        float w[8];
        dq::decode8<FMT>(blk + 6, lane, w);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float qe = qv[m] * eff;
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[m][k] += qe * w[k];
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
#pragma unroll
            for (int k = 0; k < 8; ++k) red[wave][lane * 8 + k] = acc[m][k];
            __syncthreads();
            {
                const int c = tid;  // 256 threads = 256 columns
                float v = 0.f;
#pragma unroll
                for (int w2 = 0; w2 < 8; ++w2) v += red[w2][c];
                qa[((size_t)m * NH + h) * MLA_ROW + s * 256 + c] = __float2half(v);
            }
            __syncthreads();
        }
    }
    if (s != 0) return;
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
    if (M <= 1) hipLaunchKernelGGL((k_prep_q_dec<FMT, 1>), dim3(NH, 2), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else if (M <= 2) hipLaunchKernelGGL((k_prep_q_dec<FMT, 2>), dim3(NH, 2), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else if (M <= 4) hipLaunchKernelGGL((k_prep_q_dec<FMT, 4>), dim3(NH, 2), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
    else hipLaunchKernelGGL((k_prep_q_dec<FMT, 8>), dim3(NH, 2), dim3(256), 0, s, (const uint8_t*)kvb, q_b, qa, M, NH, p0, theta);
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

size_t mla_cache_bytes(int slots) { return (size_t)slots * MLA_ROW * sizeof(__half); }

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
    const bool fast_ok = !g_mla_force_simple && (a.NH % 16 == 0) && a.cache_slots > 0 && (a.cache_slots % 16 == 0) && (a.mode != MLA_WINDOW || (a.ring > 0 && a.ring % 16 == 0));
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
#if !HY4_CUDA
    if (!HY4_MMA || hy4_is_rdna2()) {
        hipLaunchKernelGGL(k_attend_rdna2, dim3(a.NH / 16, S, a.M), dim3(128), 0, s, qa, cache, sinks, ctx, part, a, S, tps);
        if (S > 1) hipLaunchKernelGGL(k_combine, dim3(16, a.NH / 16, a.M), dim3(256), 0, s, part, ctx, a.NH, S);
        return;
    }
#endif
#if HY4_MMA
    hipLaunchKernelGGL(k_attend_wmma, dim3(a.NH / 16, S, a.M), dim3(128), 0, s, qa, cache, sinks, ctx, part, a, S, tps);
    if (S > 1) hipLaunchKernelGGL(k_combine, dim3(16, a.NH / 16, a.M), dim3(256), 0, s, part, ctx, a.NH, S);
#else
    hipLaunchKernelGGL(k_attend_simple, dim3(a.NH, a.M), dim3(256), 0, s, qa, cache, sinks, ctx, a);
#endif
}

void mla_vup(hipStream_t s, const __half* ctx, const __half* wv_b, const float* gate, __half* o, int M, int NH) {
    if (!g_mla_force_simple) {
#if !HY4_CUDA
        if (!HY4_MMA || hy4_is_rdna2()) {
            hipLaunchKernelGGL(k_head_gemm_rdna2<1>, dim3(MLA_VHEAD / 64, (M + 63) / 64, NH), dim3(256), 0, s, (const void*)ctx, wv_b, gate, o, M, NH, 0, 0.f);
            return;
        }
#endif
#if HY4_MMA
        hipLaunchKernelGGL(k_head_gemm<1>, dim3(MLA_VHEAD / HG_BN, (M + HG_BM - 1) / HG_BM, NH), dim3(256), 0, s, (const void*)ctx, wv_b, gate, o, M, NH, 0, 0.f);
        return;
#endif
    }
    constexpr int TT = 8;
    dim3 grid(NH, (M + TT - 1) / TT);
    hipLaunchKernelGGL((k_vup<TT>), grid, dim3(256), 0, s, ctx, wv_b, gate, o, M, NH);
}

void mla_transpose_wk(hipStream_t s, const __half* wk_b, __half* wk_bt, int NH) {
    hipLaunchKernelGGL(k_transpose_wk, dim3(1024), dim3(256), 0, s, wk_b, wk_bt, NH);
}

void mla_prep_q_t(hipStream_t s, const float* q_b, const __half* wk_bt, __half* qa, int M, int NH, int p0, float theta) {
#if !HY4_CUDA
    if (!HY4_MMA || hy4_is_rdna2()) {
        hipLaunchKernelGGL(k_head_gemm_rdna2<0>, dim3(MLA_KV_LORA / 64, (M + 63) / 64, NH), dim3(256), 0, s, (const void*)q_b, wk_bt, (const float*)nullptr, qa, M, NH, p0, theta);
        return;
    }
#endif
#if HY4_MMA
    hipLaunchKernelGGL(k_head_gemm<0>, dim3(MLA_KV_LORA / HG_BN, (M + HG_BM - 1) / HG_BM, NH), dim3(256), 0, s, (const void*)q_b, wk_bt, (const float*)nullptr, qa, M, NH, p0, theta);
#else
    (void)s; (void)q_b; (void)wk_bt; (void)qa; (void)M; (void)NH; (void)p0; (void)theta;
#endif
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
