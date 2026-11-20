// hy4_dfly2.hip.h - the DFly2 drafter of Hy4-Flash on the Runner's kernels (serving procedure of the owner, phaseA-tests/hy4/DFLY2-PROCEDURE-OWNER.md).
//
// One draft step (target processed positions 0..a-1, anchor token x_a emitted but not processed, h_{a-1} = target h_final of position a-1):
//   block  = [x_a, 120009 x K] at positions a..a+K, rows E(.) of the target's embed_tokens
//   slot 0 : H[0] += gate * eh_proj([enorm(E(x_a)) ; hnorm(h_{a-1})])             (raw gate, no sigmoid)
//   5 layers once over the B = K+1 rows; layer d reads the cached latents of target layer 25+4d (all rows < a, or its top-2048 picks at a-1) plus the block's own latents,
//     both through the draft layer's own kv_b_proj (absorbed form, the decode MLA kernels); intra-block causal; per-head sink in the denominator
//   final_norm, then for j = 1..K: hc_in = [hidden_norm(Hf[j]) ; embed_norm(E(prev))], h_corr = Hf[j] + down(silu(gate(hc_in)) * up(hc_in)), y_j = argmax lm_head(h_corr),
//     prev = y_j, conf_j = sigmoid(confidence_head(h_corr))
// The own block rows are written into the TARGET layer's cache slots a..a+K (free slots: the verify pass rewrites a..a+K' before it reads them, later positions write before they read).
#pragma once
#include <memory>
#include "hy4_graph.hip.h"

namespace hy4 {

constexpr int DFLY_MASK_ID = 120009;
constexpr int DFLY_KMAX = 7;
constexpr int DFLY_BMAX = DFLY_KMAX + 1;
constexpr int DFLY_LAYERS[5] = {25, 29, 33, 37, 41};

// out[0:3072) = rms(a) * wa, out[3072:6144) = rms(b) * wb ; one block of 256 threads
__global__ void k_dfly_cat2(const float* __restrict__ a, const float* __restrict__ wa, const float* __restrict__ b, const float* __restrict__ wb, __half* __restrict__ out, float eps) {
    __shared__ float sh[32];
    float sa = 0, sb = 0;
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sh);
    sb = block_sum(sb, sh);
    const float ia = rsqrtf(sa / HID + eps), ib = rsqrtf(sb / HID + eps);
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        out[i] = __float2half(a[i] * ia * wa[i]);
        out[HID + i] = __float2half(b[i] * ib * wb[i]);
    }
}

// h0[i] += gate[i] * f[i]
__global__ void k_dfly_fuse(float* __restrict__ h0, const float* __restrict__ gate, const float* __restrict__ f) {
    for (int i = threadIdx.x; i < HID; i += blockDim.x) h0[i] += gate[i] * f[i];
}

// H[row] += rms(o[row]) * w   (blockDim 256)
__global__ void k_dfly_rms_resid(const float* __restrict__ o, const float* __restrict__ w, float* __restrict__ H, float eps) {
    __shared__ float sh[32];
    const float* x = o + (size_t)blockIdx.x * HID;
    float ss = 0;
    for (int i = threadIdx.x; i < HID; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / HID + eps);
    float* h = H + (size_t)blockIdx.x * HID;
    for (int i = threadIdx.x; i < HID; i += blockDim.x) h[i] += x[i] * inv * w[i];
}

// h = Hf + D ; hc_h = half(h) ; conf = sigmoid(dot(cw, h) + cb)    (one block of 256)
__global__ void k_dfly_corr(const float* __restrict__ Hf, const float* __restrict__ D, const float* __restrict__ cw, const float* __restrict__ cb, __half* __restrict__ hc_h,
                            float* __restrict__ conf) {
    __shared__ float sh[32];
    float dot = 0;
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        const float v = Hf[i] + D[i];
        hc_h[i] = __float2half(v);
        dot += v * cw[i];
    }
    dot = block_sum(dot, sh);
    if (threadIdx.x == 0) *conf = 1.0f / (1.0f + expf(-(dot + cb[0])));
}

__global__ void k_dfly_h2f(const __half* __restrict__ in, float* __restrict__ out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = __half2float(in[i]);
}

// per-slot key lists: slot j sees the context list (n entries, ckeys[0..n)) followed by its own slots a..a+j
__global__ void k_dfly_lists(const int32_t* __restrict__ ckeys, const int32_t* __restrict__ cnk, int a, int32_t* __restrict__ dlist, int32_t* __restrict__ dnk, int lstride) {
    const int j = blockIdx.x;
    const int n = cnk[0];
    int32_t* l = dlist + (size_t)j * lstride;
    for (int i = threadIdx.x; i < n; i += blockDim.x) l[i] = ckeys[i];
    for (int i = threadIdx.x; i <= j; i += blockDim.x) l[n + i] = a + i;
    if (threadIdx.x == 0) dnk[j] = n + j + 1;
}

// argmax of each logits row [n][V] -> out[n]
__global__ void k_argmax_rows(const float* __restrict__ logits, int V, int32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* lg = logits + (size_t)blockIdx.x * V;
    float bv = -3.0e38f;
    int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        if (lg[i] > bv) { bv = lg[i]; bi = i; }
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
        out[blockIdx.x] = i2;
    }
}

// ---- reduced-vocabulary chain: candidate ids = the top-N of the base logits lm_head(final_norm(H)[j]); the chain scores only those rows with the corrected hidden state.
__device__ __forceinline__ uint32_t dfly_f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
// top-N ids of one logits row (block of 1024): 4-pass MSB-first 8-bit radix select (8x faster than 33-round binary search), then emit all > thr followed by == thr up to N
__global__ void __launch_bounds__(1024) k_dfly_topn(const float* __restrict__ logits, int V, int N, int32_t* __restrict__ cand) {
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s;
    __shared__ int need_s;
    __shared__ int cnt;
    __shared__ uint32_t thr_s;
    const float* lg = logits + (size_t)blockIdx.x * V;
    int32_t* out = cand + (size_t)blockIdx.x * N;
    if (threadIdx.x == 0) { prefix_s = 0; need_s = N; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int i = threadIdx.x; i < V; i += 1024) {
            const uint32_t k = dfly_f2key(lg[i]);
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
    if (threadIdx.x == 0) { thr_s = prefix_s; cnt = 0; }
    __syncthreads();
    const uint32_t thr = thr_s;
    for (int i = threadIdx.x; i < V; i += 1024) {
        if (dfly_f2key(lg[i]) > thr) {
            const int s = atomicAdd(&cnt, 1);
            if (s < N) out[s] = i;
        }
    }
    __syncthreads();
    if (cnt < N) {
        for (int i = threadIdx.x; i < V; i += 1024) {
            if (dfly_f2key(lg[i]) == thr) {
                const int s = atomicAdd(&cnt, 1);
                if (s < N) out[s] = i;
            }
        }
    }
}

// score[k] = lm_head row cand[k] . h   (lm_head S8_K [V][3072], row = 12 super-blocks of 262 B); 8 waves per block share h[3072] in LDS (6 KB) + packed fdot2
__global__ void __launch_bounds__(256) k_dfly_cand_score(const uint8_t* __restrict__ W, const int32_t* __restrict__ cand, const __half* __restrict__ h, float* __restrict__ score, int N) {
    __shared__ __align__(16) uint4 sh[384];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const uint4* h4 = reinterpret_cast<const uint4*>(h);
    for (int i = tid; i < 384; i += 256) sh[i] = h4[i];
    __syncthreads();
    const int k = blockIdx.x * 8 + wave;
    if (k >= N) return;
    const uint8_t* row = W + (size_t)cand[k] * 12 * 262;
    const uint32_t mag = h2_splat(1152.0f);
    float acc = 0.f;
#pragma unroll 4
    for (int sb = 0; sb < 12; ++sb) {
        const uint8_t* blk = row + sb * 262;
        uint16_t dbits;
        memcpy(&dbits, blk, 2);
        const float eff = __half2float(__ushort_as_half(dbits)) * ((float)blk[2 + (lane >> 3)] * (1.0f / 255.0f));
        uint32_t d0, d1;
        memcpy(&d0, blk + 6 + 8 * lane, 4);
        memcpy(&d1, blk + 6 + 8 * lane + 4, 4);
        const uint32_t q0 = ((d0) & 0xFFu) ^ 0x80u, q1 = ((d0 >> 8) & 0xFFu) ^ 0x80u;
        const uint32_t q2 = ((d0 >> 16) & 0xFFu) ^ 0x80u, q3 = ((d0 >> 24) & 0xFFu) ^ 0x80u;
        const uint32_t q4 = ((d1) & 0xFFu) ^ 0x80u, q5 = ((d1 >> 8) & 0xFFu) ^ 0x80u;
        const uint32_t q6 = ((d1 >> 16) & 0xFFu) ^ 0x80u, q7 = ((d1 >> 24) & 0xFFu) ^ 0x80u;
        const uint32_t w0 = h2_sub(0x64006400u | q0 | (q1 << 16), mag);
        const uint32_t w1 = h2_sub(0x64006400u | q2 | (q3 << 16), mag);
        const uint32_t w2 = h2_sub(0x64006400u | q4 | (q5 << 16), mag);
        const uint32_t w3 = h2_sub(0x64006400u | q6 | (q7 << 16), mag);
        const uint4 xv = sh[sb * 32 + lane];
        float p = h2_dot_f32(w0, xv.x, 0.f);
        p = h2_dot_f32(w1, xv.y, p);
        p = h2_dot_f32(w2, xv.z, p);
        p = h2_dot_f32(w3, xv.w, p);
        acc += eff * p;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (lane == 0) score[k] = acc;
}
__global__ void __launch_bounds__(1024) k_dfly_cand_argmax(const float* __restrict__ score, const int32_t* __restrict__ cand, int N, int32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const int id = cand[i];
        if (score[i] > bv || (score[i] == bv && id < bi)) { bv = score[i]; bi = id; }
    }
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

struct DFly2 {
    Runner& r;
    const Cfg& c;
    bool ok = false;
    int rv = 0;  // reduced-vocabulary chain: candidates per slot (0 = full lm_head per slot)
    bool isolate = getenv("HY4_DRAFT_ISOLATE") != nullptr;  // debug: the drafter works on private copies of the target cache layers
    int skip = getenv("HY4_DRAFT_SKIP") ? atoi(getenv("HY4_DRAFT_SKIP")) : 0;  // debug bisect: 1 skip the 5 layers, 2 skip the base-logit pass, 4 skip the chain loop
    __half* shadow[5] = {};
    int32_t* d_cand = nullptr;
    float* d_cscore = nullptr;
    hipStream_t st;
    // device buffers
    float *H = nullptr, *Hf = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *F4 = nullptr, *Ec = nullptr, *Dn = nullptr, *G = nullptr, *U = nullptr, *d_conf = nullptr, *hprev_f = nullptr;
    __half *Hf_h = nullptr, *hn = nullptr, *qr_h = nullptr, *qa_h = nullptr, *ctx_h = nullptr, *o_h = nullptr, *mlp_h = nullptr, *hcin = nullptr, *act = nullptr, *hc_h = nullptr;
    __half* hprev_h = nullptr;
    int32_t *d_bids = nullptr, *d_chain = nullptr, *dlist = nullptr, *dnk = nullptr;
    int32_t *h_bids = nullptr, *h_chain = nullptr;  // pinned
    float* h_conf = nullptr;
    int lstride = 0;
    LinMulti lm_attn[5], lm_mlp[5], lm_hc;
    int attn_fuse[5] = {}, mlp_fuse[5] = {};
    // stats
    double t_draft = 0;
    long n_draft = 0;

    explicit DFly2(Runner& rr) : r(rr), c(rr.c), st(rr.st) {}

    template <class T> T* al(size_t n) { return r.dalloc<T>(n); }
    const float* wf(const std::string& n) const { return (const float*)r.tw(n); }

    bool init(std::string& err) {
        if (!r.g.man.T.count("model.dfly_layers.0.self_attn.q_a_proj.weight")) {
            err = "this .strata has no DFly2 drafter tensors";
            return false;
        }
        r.dfly_capture_alloc();
        const size_t B = DFLY_BMAX;
        H = al<float>(B * HID);
        Hf = al<float>(B * HID);
        F1 = al<float>(B * 16384);
        F2 = al<float>(B * 16384);
        F3 = al<float>(B * 16384);
        F4 = al<float>(B * 16384);
        Ec = al<float>(HID);
        Dn = al<float>(HID);
        G = al<float>(HID);
        U = al<float>(HID);
        d_conf = al<float>(16);
        hprev_f = al<float>(HID);
        Hf_h = al<__half>(B * HID);
        hn = al<__half>(B * HID);
        qr_h = al<__half>(B * 1024);
        qa_h = al<__half>(B * 32 * 576);
        ctx_h = al<__half>(B * 32 * 512);
        o_h = al<__half>(B * 8192);
        mlp_h = al<__half>(B * 9216);
        hcin = al<__half>(2 * HID);
        act = al<__half>(HID);
        hc_h = al<__half>(HID);
        hprev_h = al<__half>(HID);
        d_bids = al<int32_t>(16);
        d_chain = al<int32_t>(16);
        lstride = r.index_topk + 8;
        dlist = al<int32_t>((size_t)B * lstride);
        dnk = al<int32_t>(16);
        d_cand = al<int32_t>((size_t)B * 1024);
        d_cscore = al<float>(1024);
        HCK(hipHostMalloc((void**)&h_bids, 64, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_chain, 64, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_conf, 64, hipHostMallocDefault));
        // build LinMulti tables for drafter layers + hidden_correction
        const std::string pf = "model.dfly_layers.";
        for (int d = 0; d < 5; ++d) {
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const TInfo& tq = r.ti(pa + ".q_a_proj.weight");
            const TInfo& tk = r.ti(pa + ".kv_a_proj_with_mqa.weight");
            const TInfo& tg = r.ti(pa + ".linear_gate.weight");
            if (tq.fmt != FMT_Q8ROW && tq.fmt == tk.fmt && tq.cols() == HID && tk.cols() == HID) {
                if (tg.fmt == tq.fmt && tg.cols() == HID) {
                    const void* W[3] = {r.tw(pa + ".q_a_proj.weight"), r.tw(pa + ".kv_a_proj_with_mqa.weight"), r.tw(pa + ".linear_gate.weight")};
                    float* Y[3] = {F1, F3, F4};
                    int N[3] = {(int)tq.rows(), (int)tk.rows(), (int)tg.rows()};
                    lm_attn[d] = linear_multi_build(tq.fmt, HID, 3, W, Y, N);
                    attn_fuse[d] = 3;
                } else {
                    const void* W[2] = {r.tw(pa + ".q_a_proj.weight"), r.tw(pa + ".kv_a_proj_with_mqa.weight")};
                    float* Y[2] = {F1, F3};
                    int N[2] = {(int)tq.rows(), (int)tk.rows()};
                    lm_attn[d] = linear_multi_build(tq.fmt, HID, 2, W, Y, N);
                    attn_fuse[d] = 2;
                }
            }
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const TInfo& t0 = r.ti(pm + ".gate_proj.weight");
            const TInfo& t1 = r.ti(pm + ".up_proj.weight");
            if (t0.fmt != FMT_Q8ROW && t0.fmt == t1.fmt && t0.cols() == HID && t1.cols() == HID) {
                const void* W[2] = {r.tw(pm + ".gate_proj.weight"), r.tw(pm + ".up_proj.weight")};
                float* Y[2] = {F1, F2};
                int N[2] = {(int)t0.rows(), (int)t1.rows()};
                lm_mlp[d] = linear_multi_build(t0.fmt, HID, 2, W, Y, N);
                mlp_fuse[d] = 2;
            }
        }
        {
            const TInfo& tg = r.ti(pf + "hidden_correction.gate_proj.weight");
            const TInfo& tu = r.ti(pf + "hidden_correction.up_proj.weight");
            if (tg.fmt != FMT_Q8ROW && tg.fmt == tu.fmt && tg.cols() == 2 * HID && tu.cols() == 2 * HID) {
                const void* W[2] = {r.tw(pf + "hidden_correction.gate_proj.weight"), r.tw(pf + "hidden_correction.up_proj.weight")};
                float* Y[2] = {G, U};
                int N[2] = {(int)tg.rows(), (int)tu.rows()};
                lm_hc = linear_multi_build(tg.fmt, 2 * HID, 2, W, Y, N);
            }
        }
        ok = true;
        return true;
    }

    // keep h_final row `row` of the last r.head() (fp16 [n][3072]) as the drafter's h_{a-1}
    void snap_h(int row) {
        HCK(hipMemcpyAsync(hprev_h, r.hfin + (size_t)row * HID, HID * sizeof(__half), hipMemcpyDeviceToDevice, st));
    }

    // can the block at anchor position a be drafted (a context list is needed for every layer above index_topk)
    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (!r.kv_ensure(a + K + 2)) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        if (row < 0 || row >= r.cap_rows) return false;
        return true;
    }

    // Draft K tokens. y[1..K] and conf[1..K] (host) after a stream sync. Returns false when the target state does not allow it.
    bool draft(int a, int xa, int K, int* y, float* conf) {
        if (!can_draft(a, K)) return false;
        const double t0 = now_s();
        const int B = K + 1;
        const int V = c.vocab;
        const int row = a - 1 - r.cap_p0;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        HCK(hipMemcpyAsync(d_bids, h_bids, B * 4, hipMemcpyHostToDevice, st));
        HCK(hipMemcpyAsync(d_chain, h_bids, 4, hipMemcpyHostToDevice, st));
        const std::string pf = "model.dfly_layers.";
        const TInfo& emb = r.ti("model.embed_tokens.weight");
        embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_bids, H, B);
        // slot 0: mtp_fusion
        hipLaunchKernelGGL(k_dfly_h2f, dim3(1), dim3(256), 0, st, hprev_h, hprev_f, HID);
        hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)H, wf(pf + "mtp_fusion.enorm.weight"), (const float*)hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        hipLaunchKernelGGL(k_dfly_fuse, dim3(1), dim3(256), 0, st, H, wf(pf + "mtp_fusion.gate"), (const float*)F1);
        // 5 layers
        std::unique_ptr<Runner::PS> pl(new Runner::PS(&r, "dfly_layers"));
        for (int d = 0; d < ((skip & 1) ? 0 : 5); ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const bool use_list = r.cap_list[d];
            __half* dcache = r.kcache[L];
            if (isolate) {  // debug: work on a private copy of the target layer's cache so the target cache is never written by the drafter
                const size_t by = mla_cache_bytes(r.slots_sparse);
                if (!shadow[d]) HCK(hipMalloc((void**)&shadow[d], by));
                HCK(hipMemcpyAsync(shadow[d], r.kcache[L], by, hipMemcpyDeviceToDevice, st));
                dcache = shadow[d];
            }
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
            const int daf = (!(skip & 32) && lm_attn[d].d_jobs) ? attn_fuse[d] : 0;
            if (daf > 0) linear_h_multi(st, lm_attn[d], hn, B);
            else if (!(skip & 32)) r.lin(pa + ".q_a_proj.weight", hn, F1, B);  // [B][1024]
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps);
            if (!(skip & 32)) r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);  // [B][8192]
            if (daf == 0 && !(skip & 32)) r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);  // [B][576]
            if (!(skip & 8)) mla_write_kv(st, F3, wf(pa + ".kv_a_layernorm.weight"), dcache, r.slots_sparse, B, a, 0, (float)c.theta_sparse, c.rms_eps);
            const int kvfmt = r.ti(pa + ".kv_b_proj.weight").fmt;
            if (!(skip & 8)) mla_prep_q_dec(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), F2, qa_h, B, 32, a, (float)c.theta_sparse);
            if (daf < 3 && !(skip & 32)) r.lin(pa + ".linear_gate.weight", hn, F4, B);  // [B][8192]
            MlaAttnArgs ma;
            ma.M = B;
            ma.NH = 32;
            ma.p0 = a;
            ma.cache_slots = r.slots_sparse;
            if (use_list) {
                hipLaunchKernelGGL(k_dfly_lists, dim3(B), dim3(256), 0, st, (const int32_t*)(r.dkeys[d] + (size_t)row * r.index_topk), (const int32_t*)(r.dnkc[d] + row), a, dlist, dnk, lstride);
                ma.mode = MLA_LIST;
                ma.keys = dlist;
                ma.nkeys = dnk;
                ma.kmax = lstride;
            } else {
                ma.mode = MLA_CAUSAL;
            }
            if (!(skip & 64)) r.ensure_mla_ws(mla_workspace_bytes(ma));
            if (!(skip & 8)) mla_attend(st, qa_h, dcache, wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
            if (!(skip & 8)) mla_vup_dec(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), ctx_h, F4, o_h, B, 32);
            if (!(skip & 32)) r.lin(pa + ".o_proj.weight", o_h, F1, B);  // [B][3072]
            hipLaunchKernelGGL(k_dfly_rms_resid, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps);
            // dense SwiGLU
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
            if (!(skip & 16)) {
                if (mlp_fuse[d] == 2 && lm_mlp[d].d_jobs) linear_h_multi(st, lm_mlp[d], hn, B);
                else {
                    r.lin(pm + ".gate_proj.weight", hn, F1, B);  // [B][9216]
                    r.lin(pm + ".up_proj.weight", hn, F2, B);
                }
            }
            const size_t n9 = (size_t)B * 9216;
            hipLaunchKernelGGL(k_swiglu_h, dim3((n9 + 255) / 256), dim3(256), 0, st, (const float*)F1, (const float*)F2, mlp_h, n9);
            if (!(skip & 16)) r.lin(pm + ".down_proj.weight", mlp_h, F3, B);  // [B][3072]
            hipLaunchKernelGGL(k_add_inplace, dim3(((size_t)B * HID + 255) / 256), dim3(256), 0, st, H, (const float*)F3, (size_t)B * HID);
        }
        hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps);
        pl.reset();
        // hidden_correction + lm_head, unrolled
        Runner::PS pc(&r, "dfly_chain");
        const TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const int N = std::min(rv, 1024);
        if (N > 0 && !(skip & 2)) {
            // base logits of slots 1..K in ONE lm_head pass, candidate ids per slot
            linear_h(st, lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K);
            hipLaunchKernelGGL(k_dfly_topn, dim3(K), dim3(1024), 0, st, (const float*)r.logits, V, N, d_cand);
        }
        for (int j = 1; j <= ((skip & 4) ? 0 : K); ++j) {
            embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_chain + (j - 1), Ec, 1);
            hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)(Hf + (size_t)j * HID), wf(pf + "hidden_correction.hidden_norm.weight"), (const float*)Ec,
                               wf(pf + "hidden_correction.embed_norm.weight"), hcin, c.rms_eps);
            if (lm_hc.d_jobs) {
                linear_h_multi(st, lm_hc, hcin, 1);
            } else {
                r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
                r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
            }
            hipLaunchKernelGGL(k_swiglu_h, dim3((HID + 255) / 256), dim3(256), 0, st, (const float*)G, (const float*)U, act, (size_t)HID);
            r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
            hipLaunchKernelGGL(k_dfly_corr, dim3(1), dim3(256), 0, st, (const float*)(Hf + (size_t)j * HID), (const float*)Dn, wf(pf + "confidence_head.proj.weight"),
                               wf(pf + "confidence_head.proj.bias"), hc_h, d_conf + j);
            if (N > 0) {
                hipLaunchKernelGGL(k_dfly_cand_score, dim3((N + 7) / 8), dim3(256), 0, st, (const uint8_t*)lmw, (const int32_t*)(d_cand + (size_t)(j - 1) * N), (const __half*)hc_h, d_cscore, N);
                hipLaunchKernelGGL(k_dfly_cand_argmax, dim3(1), dim3(1024), 0, st, (const float*)d_cscore, (const int32_t*)(d_cand + (size_t)(j - 1) * N), N, d_chain + j);
            } else {
                linear_h(st, lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), hc_h, r.logits, 1);
                hipLaunchKernelGGL(k_argmax, dim3(1), dim3(1024), 0, st, (const float*)r.logits, V, d_chain + j);
            }
        }
        HCK(hipMemcpyAsync(h_chain, d_chain, B * 4, hipMemcpyDeviceToHost, st));
        HCK(hipMemcpyAsync(h_conf, d_conf, B * 4, hipMemcpyDeviceToHost, st));
        HCK(hipStreamSynchronize(st));
        for (int j = 1; j <= K; ++j) {
            y[j] = h_chain[j];
            conf[j] = h_conf[j];
        }
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

}  // namespace hy4
