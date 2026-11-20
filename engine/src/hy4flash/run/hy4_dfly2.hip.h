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
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hy4_graph.hip.h"

namespace hy4 {

struct SampleOpts {
    float temperature = 0.f;  // 0 = greedy
    float top_p = 1.f;
    int top_k = 64;
    float min_p = 0.f;
    float rep = 1.f, freq = 0.f, pres = 0.f;
    int last_n = 64;
    uint64_t seed = 0;
    bool greedy() const { return temperature <= 0.f; }
    bool penalties() const { return rep != 1.f || freq != 0.f || pres != 0.f; }
};

using SparseDist = std::vector<std::pair<int32_t, double>>;

// Exact normalized probability distribution from a full logits row [V] under SampleOpts + history penalties.
// Matches hy4_serve's sample_host bit-for-bit: penalties -> top-k (<=64) -> temp softmax -> min_p -> top_p -> renorm.
inline void compute_dist_full(const float* lg_in, int V, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    out.clear();
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t)s.last_n ? hist.size() - (size_t)s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    auto adj = [&](int id) -> float {
        float x = lg_in[id];
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float)it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        return x;
    };
    const int K = std::min(std::max(1, s.top_k), 64);
    // Single-pass top-K maintenance avoids allocating & partial_sorting a 120K index vector on every verify row!
    std::pair<float, int> top[65];
    int ntop = 0;
    for (int i = 0; i < V; ++i) {
        float v = adj(i);
        if (ntop < K || v > top[ntop - 1].first) {
            int pos = ntop < K ? ntop++ : K - 1;
            while (pos > 0 && (v > top[pos - 1].first || (v == top[pos - 1].first && i < top[pos - 1].second))) {
                top[pos] = top[pos - 1];
                --pos;
            }
            top[pos] = {v, i};
        }
    }
    if (ntop == 0) return;
    if (s.greedy()) {
        out.push_back({top[0].second, 1.0});
        return;
    }
    double p[64];
    const double mx = top[0].first;
    double sum = 0.0;
    for (int i = 0; i < ntop; ++i) {
        p[i] = std::exp(((double)top[i].first - mx) / (double)s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < ntop; ++i) p[i] /= sum;
    int keep = ntop;
    if (s.min_p > 0.f) {
        const double thr = (double)s.min_p * p[0];
        keep = 1;
        while (keep < ntop && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double)s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({top[i].second, p[i] / z});
}

// Normalized draft distribution over a reduced candidate subset (ids[0..N-1], scores[0..N-1]).
inline void compute_dist_cand(const int32_t* ids, const float* scores, int N, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    out.clear();
    if (N <= 0) return;
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t)s.last_n ? hist.size() - (size_t)s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    const int K = std::min(std::min(std::max(1, s.top_k), 64), N);
    std::vector<std::pair<float, int32_t>> cand((size_t)N);
    for (int i = 0; i < N; ++i) {
        float x = scores[i];
        int32_t id = ids[i];
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float)it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        cand[i] = {x, id};
    }
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (s.greedy()) {
        out.push_back({cand[0].second, 1.0});
        return;
    }
    double p[64];
    const double mx = cand[0].first;
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp(((double)cand[i].first - mx) / (double)s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < K; ++i) p[i] /= sum;
    int keep = K;
    if (s.min_p > 0.f) {
        const double thr = (double)s.min_p * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double)s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({cand[i].second, p[i] / z});
}

inline int32_t sample_from_dist(const SparseDist& dist, std::mt19937_64& rng) {
    if (dist.empty()) return 0;
    double z = 0.0;
    for (const auto& kv : dist) z += kv.second;
    if (z <= 0.0) return dist[0].first;
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng), c = 0.0;
    for (const auto& kv : dist) {
        c += kv.second;
        if (r <= c) return kv.first;
    }
    return dist.back().first;
}

inline double dist_prob_of(const SparseDist& dist, int32_t tok) {
    for (const auto& kv : dist)
        if (kv.first == tok) return kv.second;
    return 0.0;
}

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
__global__ void k_dfly_lists_d(const int32_t* __restrict__ ckeys_base, const int32_t* __restrict__ cnk_base, int index_topk, const int* __restrict__ a_dev, const int* __restrict__ row_dev,
                               int32_t* __restrict__ dlist, int32_t* __restrict__ dnk, int lstride) {
    const int j = blockIdx.x;
    const int a = *a_dev;
    const int row = *row_dev;
    const int n = cnk_base[row];
    const int32_t* ck = ckeys_base + (size_t)row * index_topk;
    int32_t* l = dlist + (size_t)j * lstride;
    for (int i = threadIdx.x; i < n; i += blockDim.x) l[i] = ck[i];
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
    int32_t* h_cand = nullptr;    // pinned [DFLY_BMAX * 1024] for Tree-8 / probabilistic chain
    float* h_cscore = nullptr;    // pinned [1024]
    int32_t* d_prev_one = nullptr;
    int lstride = 0;
    LinMulti lm_attn[5], lm_mlp[5], lm_hc;
    int attn_fuse[5] = {}, mlp_fuse[5] = {};
    // stats
    double t_draft = 0;
    long n_draft = 0;
    long n_chain_evals = 0;
    long n_early_exits = 0;
#if HY4_CUDA
    struct DevDraftArgs { int a; int row; };
    DevDraftArgs *h_dargs = nullptr, *d_dargs = nullptr;
    struct DgKey {
        int K, list_mask, s_causal, rv, kv_is_host, slots_sparse, lin_gemv_max, int8_mma, dp4a, mla_ws_gen, bb_only;
        uint64_t kv_gen;
        bool operator==(const DgKey& o) const { return std::memcmp(this, &o, sizeof(DgKey)) == 0; }
    };
    struct DgEntry {
        DgKey key;
        cudaGraphExec_t exec = nullptr;
    };
    std::vector<DgEntry> dg_cache;
    void dg_clear() {
        for (auto& e : dg_cache) if (e.exec) (void)cudaGraphExecDestroy(e.exec);
        dg_cache.clear();
    }
    ~DFly2() { dg_clear(); }
#endif

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
        d_prev_one = al<int32_t>(4);
        HCK(hipHostMalloc((void**)&h_bids, 64, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_chain, 64, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_conf, 64, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_cand, (size_t)B * 1024 * 4, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&h_cscore, (size_t)1024 * 4, hipHostMallocDefault));
#if HY4_CUDA
        HCK(hipHostMalloc((void**)&h_dargs, sizeof(DevDraftArgs), hipHostMallocDefault));
        d_dargs = al<DevDraftArgs>(1);
#endif
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
#if HY4_CUDA
        if (r.cuda_graph_on && !isolate && skip == 0 && !r.prof && !r.prof2 && !r.dump) {
            int lmask = 0;
            for (int d = 0; d < 5; ++d) if (r.cap_list[d]) lmask |= (1 << d);
            MlaAttnArgs ma_c;
            ma_c.M = B; ma_c.NH = 32; ma_c.p0 = a; ma_c.mode = MLA_CAUSAL;
            const int s_causal = (lmask == 31) ? 0 : mla_choose_splits(ma_c);
            const size_t need_ws = mla_workspace_bytes(B, 32, std::max(lstride, a + B));
            r.ensure_mla_ws(need_ws);
#if HY4_MMA
            if (skd::lin_int8_mma()) skd::int8_act_buf().ensure((size_t)B * 16384, skd::lin_int8_mma() >= 2);
#endif
            DgKey key{};
            key.K = K; key.list_mask = lmask; key.s_causal = s_causal; key.rv = std::min(rv, 1024);
            key.kv_is_host = r.kv_is_host ? 1 : 0;
            key.slots_sparse = r.slots_sparse;
            key.lin_gemv_max = skd::lin_gemv_max_m();
#if HY4_MMA
            key.int8_mma = skd::lin_int8_mma();
            key.dp4a = skd::lin_dp4a();
#endif
            key.mla_ws_gen = r.mla_ws_gen;
            key.bb_only = 0;
            key.kv_gen = r.kv_gen;
            if (!dg_cache.empty() && dg_cache.front().key.kv_gen != r.kv_gen) dg_clear();
            cudaGraphExec_t exec = nullptr;
            for (auto& e : dg_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                if (dg_cache.size() >= 12) {
                    if (dg_cache.front().exec) (void)cudaGraphExecDestroy(dg_cache.front().exec);
                    dg_cache.erase(dg_cache.begin());
                }
                const std::string pf = "model.dfly_layers.";
                const TInfo& emb = r.ti("model.embed_tokens.weight");
                const TInfo& lm = r.ti("lm_head.weight");
                const void* lmw = r.tw("lm_head.weight");
                const int N = std::min(rv, 1024);
                cudaGraph_t gr = nullptr;
                HCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_bids, H, B);
                hipLaunchKernelGGL(k_dfly_h2f, dim3(1), dim3(256), 0, st, hprev_h, hprev_f, HID);
                hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)H, wf(pf + "mtp_fusion.enorm.weight"), (const float*)hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
                r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
                hipLaunchKernelGGL(k_dfly_fuse, dim3(1), dim3(256), 0, st, H, wf(pf + "mtp_fusion.gate"), (const float*)F1);
                for (int d = 0; d < 5; ++d) {
                    const int L = DFLY_LAYERS[d];
                    const std::string pa = pf + std::to_string(d) + ".self_attn";
                    const std::string pm = pf + std::to_string(d) + ".mlp";
                    const bool use_list = r.cap_list[d];
                    __half* dcache = r.kcache[L];
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
                    r.lin(pa + ".q_a_proj.weight", hn, F1, B);
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps);
                    r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
                    r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
                    mla_write_kv_d(st, F3, wf(pa + ".kv_a_layernorm.weight"), dcache, r.slots_sparse, B, &d_dargs->a, 0, (float)c.theta_sparse, c.rms_eps);
                    const int kvfmt = r.ti(pa + ".kv_b_proj.weight").fmt;
                    mla_prep_q_dec_d(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), F2, qa_h, B, 32, &d_dargs->a, (float)c.theta_sparse);
                    r.lin(pa + ".linear_gate.weight", hn, F4, B);
                    MlaAttnArgs ma;
                    ma.M = B; ma.NH = 32; ma.p0 = a; ma.p0_dev = &d_dargs->a; ma.cache_slots = r.slots_sparse;
                    if (use_list) {
                        hipLaunchKernelGGL(k_dfly_lists_d, dim3(B), dim3(256), 0, st, (const int32_t*)r.dkeys[d], (const int32_t*)r.dnkc[d], r.index_topk, &d_dargs->a, &d_dargs->row, dlist, dnk, lstride);
                        ma.mode = MLA_LIST; ma.keys = dlist; ma.nkeys = dnk; ma.kmax = lstride;
                    } else {
                        ma.mode = MLA_CAUSAL;
                    }
                    mla_attend(st, qa_h, dcache, wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
                    mla_vup_dec(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), ctx_h, F4, o_h, B, 32);
                    r.lin(pa + ".o_proj.weight", o_h, F1, B);
                    hipLaunchKernelGGL(k_dfly_rms_resid, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps);
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
                    r.lin(pm + ".gate_proj.weight", hn, F1, B);
                    r.lin(pm + ".up_proj.weight", hn, F2, B);
                    const size_t n9 = (size_t)B * 9216;
                    hipLaunchKernelGGL(k_swiglu_h, dim3((n9 + 255) / 256), dim3(256), 0, st, (const float*)F1, (const float*)F2, mlp_h, n9);
                    r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
                    hipLaunchKernelGGL(k_add_inplace, dim3(((size_t)B * HID + 255) / 256), dim3(256), 0, st, H, (const float*)F3, (size_t)B * HID);
                }
                hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps);
                if (N > 0) {
                    linear_h(st, lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K);
                    hipLaunchKernelGGL(k_dfly_topn, dim3(K), dim3(1024), 0, st, (const float*)r.logits, V, N, d_cand);
                }
                for (int j = 1; j <= K; ++j) {
                    embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_chain + (j - 1), Ec, 1);
                    hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)(Hf + (size_t)j * HID), wf(pf + "hidden_correction.hidden_norm.weight"), (const float*)Ec,
                                       wf(pf + "hidden_correction.embed_norm.weight"), hcin, c.rms_eps);
                    r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
                    r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
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
                HCK(cudaStreamEndCapture(st, &gr));
                HCK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));
                HCK(cudaGraphDestroy(gr));
                dg_cache.push_back({key, exec});
            }
            h_dargs->a = a;
            h_dargs->row = row;
            HCK(hipMemcpyAsync(d_dargs, h_dargs, sizeof(DevDraftArgs), hipMemcpyHostToDevice, st));
            HCK(cudaGraphLaunch(exec, st));
            HCK(hipMemcpyAsync(h_chain, d_chain, B * 4, hipMemcpyDeviceToHost, st));
            HCK(hipMemcpyAsync(h_conf, d_conf, B * 4, hipMemcpyDeviceToHost, st));
            HCK(hipStreamSynchronize(st));
            for (int j = 1; j <= K; ++j) { y[j] = h_chain[j]; conf[j] = h_conf[j]; }
            t_draft += now_s() - t0;
            ++n_draft;
            return true;
        }
#endif
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

    // Run the 5 DFly2 layers once for [xa, MASK x K] + batched base-logits top-N candidates (when N > 0).
    bool run_block_backbone(int a, int xa, int K, int& N_out) {
        if (!can_draft(a, K)) return false;
        const int B = K + 1;
        const int V = c.vocab;
        const int row = a - 1 - r.cap_p0;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        HCK(hipMemcpyAsync(d_bids, h_bids, B * 4, hipMemcpyHostToDevice, st));
#if HY4_CUDA
        if (r.cuda_graph_on && !isolate && skip == 0 && !r.prof && !r.prof2 && !r.dump) {
            int lmask = 0;
            for (int d = 0; d < 5; ++d) if (r.cap_list[d]) lmask |= (1 << d);
            MlaAttnArgs ma_c;
            ma_c.M = B; ma_c.NH = 32; ma_c.p0 = a; ma_c.mode = MLA_CAUSAL;
            const int s_causal = (lmask == 31) ? 0 : mla_choose_splits(ma_c);
            const size_t need_ws = mla_workspace_bytes(B, 32, std::max(lstride, a + B));
            r.ensure_mla_ws(need_ws);
#if HY4_MMA
            if (skd::lin_int8_mma()) skd::int8_act_buf().ensure((size_t)B * 16384, skd::lin_int8_mma() >= 2);
#endif
            const int N = std::min(rv, 1024);
            N_out = N;
            DgKey key{};
            key.K = K; key.list_mask = lmask; key.s_causal = s_causal; key.rv = N;
            key.kv_is_host = r.kv_is_host ? 1 : 0;
            key.slots_sparse = r.slots_sparse;
            key.lin_gemv_max = skd::lin_gemv_max_m();
#if HY4_MMA
            key.int8_mma = skd::lin_int8_mma();
            key.dp4a = skd::lin_dp4a();
#endif
            key.mla_ws_gen = r.mla_ws_gen;
            key.bb_only = 1;
            key.kv_gen = r.kv_gen;
            if (!dg_cache.empty() && dg_cache.front().key.kv_gen != r.kv_gen) dg_clear();
            cudaGraphExec_t exec = nullptr;
            for (auto& e : dg_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                if (dg_cache.size() >= 16) {
                    if (dg_cache.front().exec) (void)cudaGraphExecDestroy(dg_cache.front().exec);
                    dg_cache.erase(dg_cache.begin());
                }
                const std::string pf = "model.dfly_layers.";
                const TInfo& emb = r.ti("model.embed_tokens.weight");
                const TInfo& lm = r.ti("lm_head.weight");
                const void* lmw = r.tw("lm_head.weight");
                cudaGraph_t gr = nullptr;
                HCK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_bids, H, B);
                hipLaunchKernelGGL(k_dfly_h2f, dim3(1), dim3(256), 0, st, hprev_h, hprev_f, HID);
                hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)H, wf(pf + "mtp_fusion.enorm.weight"), (const float*)hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
                r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
                hipLaunchKernelGGL(k_dfly_fuse, dim3(1), dim3(256), 0, st, H, wf(pf + "mtp_fusion.gate"), (const float*)F1);
                for (int d = 0; d < 5; ++d) {
                    const int L = DFLY_LAYERS[d];
                    const std::string pa = pf + std::to_string(d) + ".self_attn";
                    const std::string pm = pf + std::to_string(d) + ".mlp";
                    const bool use_list = r.cap_list[d];
                    __half* dcache = r.kcache[L];
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
                    r.lin(pa + ".q_a_proj.weight", hn, F1, B);
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps);
                    r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
                    r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
                    mla_write_kv_d(st, F3, wf(pa + ".kv_a_layernorm.weight"), dcache, r.slots_sparse, B, &d_dargs->a, 0, (float)c.theta_sparse, c.rms_eps);
                    const int kvfmt = r.ti(pa + ".kv_b_proj.weight").fmt;
                    mla_prep_q_dec_d(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), F2, qa_h, B, 32, &d_dargs->a, (float)c.theta_sparse);
                    r.lin(pa + ".linear_gate.weight", hn, F4, B);
                    MlaAttnArgs ma;
                    ma.M = B; ma.NH = 32; ma.p0 = a; ma.p0_dev = &d_dargs->a; ma.cache_slots = r.slots_sparse;
                    if (use_list) {
                        hipLaunchKernelGGL(k_dfly_lists_d, dim3(B), dim3(256), 0, st, (const int32_t*)r.dkeys[d], (const int32_t*)r.dnkc[d], r.index_topk, &d_dargs->a, &d_dargs->row, dlist, dnk, lstride);
                        ma.mode = MLA_LIST; ma.keys = dlist; ma.nkeys = dnk; ma.kmax = lstride;
                    } else {
                        ma.mode = MLA_CAUSAL;
                    }
                    mla_attend(st, qa_h, dcache, wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
                    mla_vup_dec(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), ctx_h, F4, o_h, B, 32);
                    r.lin(pa + ".o_proj.weight", o_h, F1, B);
                    hipLaunchKernelGGL(k_dfly_rms_resid, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps);
                    hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
                    r.lin(pm + ".gate_proj.weight", hn, F1, B);
                    r.lin(pm + ".up_proj.weight", hn, F2, B);
                    const size_t n9 = (size_t)B * 9216;
                    hipLaunchKernelGGL(k_swiglu_h, dim3((n9 + 255) / 256), dim3(256), 0, st, (const float*)F1, (const float*)F2, mlp_h, n9);
                    r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
                    hipLaunchKernelGGL(k_add_inplace, dim3(((size_t)B * HID + 255) / 256), dim3(256), 0, st, H, (const float*)F3, (size_t)B * HID);
                }
                hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps);
                if (N > 0) {
                    linear_h(st, lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K);
                    hipLaunchKernelGGL(k_dfly_topn, dim3(K), dim3(1024), 0, st, (const float*)r.logits, V, N, d_cand);
                }
                HCK(cudaStreamEndCapture(st, &gr));
                HCK(cudaGraphInstantiate(&exec, gr, nullptr, nullptr, 0));
                HCK(cudaGraphDestroy(gr));
                dg_cache.push_back({key, exec});
            }
            h_dargs->a = a;
            h_dargs->row = row;
            HCK(hipMemcpyAsync(d_dargs, h_dargs, sizeof(DevDraftArgs), hipMemcpyHostToDevice, st));
            HCK(cudaGraphLaunch(exec, st));
            if (N > 0) HCK(hipMemcpyAsync(h_cand, d_cand, (size_t)K * N * 4, hipMemcpyDeviceToHost, st));
            return true;
        }
#endif
        const std::string pf = "model.dfly_layers.";
        const TInfo& emb = r.ti("model.embed_tokens.weight");
        embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_bids, H, B);
        hipLaunchKernelGGL(k_dfly_h2f, dim3(1), dim3(256), 0, st, hprev_h, hprev_f, HID);
        hipLaunchKernelGGL(k_dfly_cat2, dim3(1), dim3(256), 0, st, (const float*)H, wf(pf + "mtp_fusion.enorm.weight"), (const float*)hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        hipLaunchKernelGGL(k_dfly_fuse, dim3(1), dim3(256), 0, st, H, wf(pf + "mtp_fusion.gate"), (const float*)F1);
        std::unique_ptr<Runner::PS> pl(new Runner::PS(&r, "dfly_layers"));
        for (int d = 0; d < ((skip & 1) ? 0 : 5); ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const bool use_list = r.cap_list[d];
            __half* dcache = r.kcache[L];
            if (isolate) {
                const size_t by = mla_cache_bytes(r.slots_sparse);
                if (!shadow[d]) HCK(hipMalloc((void**)&shadow[d], by));
                HCK(hipMemcpyAsync(shadow[d], r.kcache[L], by, hipMemcpyDeviceToDevice, st));
                dcache = shadow[d];
            }
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
            const int daf = (!(skip & 32) && lm_attn[d].d_jobs) ? attn_fuse[d] : 0;
            if (daf > 0) linear_h_multi(st, lm_attn[d], hn, B);
            else if (!(skip & 32)) r.lin(pa + ".q_a_proj.weight", hn, F1, B);
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".q_a_layernorm.weight"), (float*)nullptr, qr_h, c.q_lora, c.rms_eps);
            if (!(skip & 32)) r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
            if (daf == 0 && !(skip & 32)) r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
            if (!(skip & 8)) mla_write_kv(st, F3, wf(pa + ".kv_a_layernorm.weight"), dcache, r.slots_sparse, B, a, 0, (float)c.theta_sparse, c.rms_eps);
            const int kvfmt = r.ti(pa + ".kv_b_proj.weight").fmt;
            if (!(skip & 8)) mla_prep_q_dec(st, kvfmt, r.tw(pa + ".kv_b_proj.weight"), F2, qa_h, B, 32, a, (float)c.theta_sparse);
            if (daf < 3 && !(skip & 32)) r.lin(pa + ".linear_gate.weight", hn, F4, B);
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
            if (!(skip & 32)) r.lin(pa + ".o_proj.weight", o_h, F1, B);
            hipLaunchKernelGGL(k_dfly_rms_resid, dim3(B), dim3(256), 0, st, (const float*)F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps);
            hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), (float*)nullptr, hn, HID, c.rms_eps);
            if (!(skip & 16)) {
                if (mlp_fuse[d] == 2 && lm_mlp[d].d_jobs) linear_h_multi(st, lm_mlp[d], hn, B);
                else {
                    r.lin(pm + ".gate_proj.weight", hn, F1, B);
                    r.lin(pm + ".up_proj.weight", hn, F2, B);
                }
            }
            const size_t n9 = (size_t)B * 9216;
            hipLaunchKernelGGL(k_swiglu_h, dim3((n9 + 255) / 256), dim3(256), 0, st, (const float*)F1, (const float*)F2, mlp_h, n9);
            if (!(skip & 16)) r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
            hipLaunchKernelGGL(k_add_inplace, dim3(((size_t)B * HID + 255) / 256), dim3(256), 0, st, H, (const float*)F3, (size_t)B * HID);
        }
        hipLaunchKernelGGL(k_rms_rows, dim3(B), dim3(256), 0, st, (const float*)H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps);
        pl.reset();
        const int N = std::min(rv, 1024);
        N_out = N;
        if (N > 0 && !(skip & 2)) {
            const TInfo& lm = r.ti("lm_head.weight");
            linear_h(st, lm.fmt, r.tw("lm_head.weight"), (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K);
            hipLaunchKernelGGL(k_dfly_topn, dim3(K), dim3(1024), 0, st, (const float*)r.logits, V, N, d_cand);
            HCK(hipMemcpyAsync(h_cand, d_cand, (size_t)K * N * 4, hipMemcpyDeviceToHost, st));
        }
        return true;
    }

    // Evaluate ONE hidden_correction + lm_head slot j (1..K) given previous token prev_tok.
    // Returns confidence; fills cand scores in h_cscore[0..N-1] (if N > 0) or full logits in full_lg (if N == 0).
    float eval_slot_host(int j, int32_t prev_tok, int N, std::vector<float>& full_lg) {
        const std::string pf = "model.dfly_layers.";
        const TInfo& emb = r.ti("model.embed_tokens.weight");
        const TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        h_chain[0] = prev_tok;
        HCK(hipMemcpyAsync(d_prev_one, h_chain, 4, hipMemcpyHostToDevice, st));
        embed_rows(st, emb.fmt, r.tw("model.embed_tokens.weight"), (int)emb.rows(), (int)emb.cols(), d_prev_one, Ec, 1);
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
                           wf(pf + "confidence_head.proj.bias"), hc_h, d_conf);
        if (N > 0) {
            hipLaunchKernelGGL(k_dfly_cand_score, dim3((N + 7) / 8), dim3(256), 0, st, (const uint8_t*)lmw, (const int32_t*)(d_cand + (size_t)(j - 1) * N), (const __half*)hc_h, d_cscore, N);
            HCK(hipMemcpyAsync(h_cscore, d_cscore, (size_t)N * 4, hipMemcpyDeviceToHost, st));
            HCK(hipMemcpyAsync(h_conf, d_conf, 4, hipMemcpyDeviceToHost, st));
            HCK(hipStreamSynchronize(st));
        } else {
            linear_h(st, lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), hc_h, r.logits, 1);
            full_lg.resize((size_t)c.vocab);
            HCK(hipMemcpyAsync(full_lg.data(), r.logits, (size_t)c.vocab * 4, hipMemcpyDeviceToHost, st));
            HCK(hipMemcpyAsync(h_conf, d_conf, 4, hipMemcpyDeviceToHost, st));
            HCK(hipStreamSynchronize(st));
        }
        ++n_chain_evals;
        return h_conf[0];
    }

    // Tree-8 + Early-Exit + Probabilistic-aware drafter.
    // Produces a connected topological tree of M <= K + 1 <= 8 nodes rooted at node 0 (xa).
    // - Greedy mode (samp == nullptr || (samp->greedy() && !samp->penalties())):
    //   Uses confidence_head + top-256 lm_head margins to branch top-2 at uncertain slot(s), ordered by decreasing expected reach probability.
    // - Probabilistic mode:
    //   Computes exact sparse draft distributions q_c for each child (without-replacement top-2 branching when enable_tree) for lossless SpecInfer rejection sampling.
    // - Early-exit: stops unrolling hidden_correction + lm_head once the best frontier reach probability drops below early_cut.
    bool draft_tree(int a, int xa, int K, TreeSpec& tree, float* node_conf, float* node_reach, const SampleOpts* samp, const std::vector<int32_t>* tok_hist, std::mt19937_64* rng,
                    std::vector<SparseDist>* node_qdist, float early_cut, bool enable_tree, const std::function<double(float)>& pcal_fn) {
        const double t0 = now_s();
        int N = 0;
        if (!run_block_backbone(a, xa, K, N)) return false;
        Runner::PS pc(&r, "dfly_chain");
        const bool prob_mode = samp != nullptr && (!samp->greedy() || samp->penalties());
        tree = TreeSpec{};
        tree.M = 1;
        tree.tok[0] = xa;
        tree.parent[0] = -1;
        tree.depth[0] = 0;
        node_conf[0] = 1.0f;
        node_reach[0] = 1.0f;
        if (node_qdist) {
            node_qdist->clear();
            node_qdist->resize(1);
        }
        struct Cand {
            int parent;
            int depth;
            int32_t tok;
            float conf;
            float reach;
            SparseDist qdist;
        };
        std::vector<Cand> frontier;
        std::vector<float> full_lg;
        std::vector<int32_t> branch_hist;
        auto expand = [&](int u) {
            const int d_next = (int)tree.depth[u] + 1;
            if (d_next > K) return;
            const float cf = eval_slot_host(d_next, tree.tok[u], N, full_lg);
            const double p_acc = std::clamp(pcal_fn(cf), 0.02, 0.995);
            const float r_u = node_reach[u];
            if (!prob_mode) {
                // Find top-1 and top-2 tokens + logit gap -> softmax share
                int32_t id1 = 0, id2 = -1;
                float s1 = -3e38f, s2 = -3e38f;
                if (N > 0) {
                    const int32_t* cids = h_cand + (size_t)(d_next - 1) * N;
                    for (int i = 0; i < N; ++i) {
                        const float sc = h_cscore[i];
                        const int32_t id = cids[i];
                        if (sc > s1 || (sc == s1 && id < id1)) { s2 = s1; id2 = id1; s1 = sc; id1 = id; }
                        else if (id != id1 && (sc > s2 || (sc == s2 && id < id2))) { s2 = sc; id2 = id; }
                    }
                } else {
                    for (int i = 0; i < c.vocab; ++i) {
                        const float sc = full_lg[i];
                        if (sc > s1) { s2 = s1; id2 = id1; s1 = sc; id1 = i; }
                        else if (sc > s2) { s2 = sc; id2 = i; }
                    }
                }
                const float r1 = (float)(r_u * p_acc);
                frontier.push_back({u, d_next, id1, cf, r1, {}});
                if (enable_tree && id2 >= 0 && id2 != id1) {
                    const double gap = std::max(0.0, (double)(s1 - s2));
                    const double ratio2 = std::exp(-gap);  // q2 / q1
                    const double p2 = std::min(1.0 - p_acc, p_acc * ratio2 * 0.85);
                    if (p2 > 0.05) {
                        frontier.push_back({u, d_next, id2, (float)std::clamp(p2, 0.01, 0.95), (float)(r_u * p2), {}});
                    }
                }
            } else {
                // Probabilistic mode: build branch history if penalties are active
                const std::vector<int32_t>* hptr = tok_hist;
                if (samp->penalties() && tok_hist) {
                    branch_hist = *tok_hist;
                    int chain_nodes[8], nc = 0;
                    for (int cur = u; cur > 0; cur = tree.parent[cur]) chain_nodes[nc++] = cur;
                    for (int k = nc - 1; k >= 0; --k) branch_hist.push_back(tree.tok[chain_nodes[k]]);
                    hptr = &branch_hist;
                }
                static const std::vector<int32_t> empty_h;
                SparseDist q1;
                if (N > 0) compute_dist_cand(h_cand + (size_t)(d_next - 1) * N, h_cscore, N, *samp, hptr ? *hptr : empty_h, q1);
                else compute_dist_full(full_lg.data(), c.vocab, *samp, hptr ? *hptr : empty_h, q1);
                if (q1.empty()) return;
                const int32_t y1 = sample_from_dist(q1, *rng);
                const float r1 = (float)(r_u * p_acc);
                frontier.push_back({u, d_next, y1, cf, r1, q1});
                if (enable_tree && q1.size() >= 2) {
                    // Second sibling drawn without replacement from q1 \ {y1} (SpecInfer exact tree residual)
                    SparseDist q2;
                    double z2 = 0.0;
                    for (const auto& kv : q1)
                        if (kv.first != y1) z2 += kv.second;
                    if (z2 > 0.08) {
                        q2.reserve(q1.size() - 1);
                        for (const auto& kv : q1)
                            if (kv.first != y1) q2.push_back({kv.first, kv.second / z2});
                        const int32_t y2 = sample_from_dist(q2, *rng);
                        const double p2 = std::min(1.0 - p_acc, z2 * 0.75);
                        if (p2 > 0.05) {
                            frontier.push_back({u, d_next, y2, (float)std::clamp(p2, 0.01, 0.95), (float)(r_u * p2), std::move(q2)});
                        }
                    }
                }
            }
        };

        // Expand root 0 first
        expand(0);
        const int max_nodes = std::min(K + 1, DFLY_BMAX);
        while (tree.M < max_nodes && !frontier.empty()) {
            // Pick frontier node with maximum reach probability (preserves topological order since reach[child] <= reach[parent])
            int best_i = 0;
            for (int i = 1; i < (int)frontier.size(); ++i)
                if (frontier[i].reach > frontier[best_i].reach) best_i = i;
            if (early_cut > 0.f && tree.M > 1 && frontier[best_i].reach < early_cut) {
                ++n_early_exits;
                break;
            }
            Cand chosen = std::move(frontier[best_i]);
            frontier.erase(frontier.begin() + best_i);
            const int idx = tree.M++;
            tree.tok[idx] = chosen.tok;
            tree.parent[idx] = (int8_t)chosen.parent;
            tree.depth[idx] = (uint8_t)chosen.depth;
            node_conf[idx] = chosen.conf;
            node_reach[idx] = chosen.reach;
            if (node_qdist) node_qdist->push_back(std::move(chosen.qdist));
            // Expand newly added node only if we still have room in the tree and its children could beat early_cut
            if (tree.M < max_nodes && chosen.depth < K && (early_cut <= 0.f || chosen.reach >= early_cut)) {
                expand(idx);
            }
        }
        tree.build_anc();
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

}  // namespace hy4
