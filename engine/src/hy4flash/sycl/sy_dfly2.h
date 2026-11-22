// sy_dfly2.h - the DFly2 drafter of Hy4-Flash on Intel Arc (SYCL):
//   * Quantized S_K kv_b_proj directly (0 persistent fp16 wkd/wvd -> saves 73 MB VRAM + 3x less bandwidth)
//   * Multi-GEMV projection fusion (q_a+kv_a+linear_gate, MLP gate+up, hidden_correction gate+up) via SG16 k_gemv_multi
//   * 4-pass 8-bit radix top-N candidate selection (8x fewer passes than 33-round binary search)
//   * SLM-cached SG16 candidate scorer (6 KiB SLM shared across 16 waves/WG)
//   * Full 5-layer backbone + top-N captured in ONE SYCL command_graph (~160 launches -> 1 replay!)
//   * Tree-8 + Early-Exit + Probabilistic-aware (SpecInfer lossless rejection sampling) drafter
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

#include "sy_graph.h"

namespace hy4s {

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
// Matches sy_serve's sample_host bit-for-bit: penalties -> top-k (<=64) -> temp softmax -> min_p -> top_p -> renorm.
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
    std::pair<float, int> top[65];
    int ntop = 0;
    for (int i = 0; i < V; ++i) {
        const float v = adj(i);
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
        const int32_t id = ids[i];
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

// ---- kernels
inline void embed_rows_f32(const uint8_t* W, const int32_t* ids, float* out, int T) {
    Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int t = (int)it.get_group(0);
        const uint8_t* row = W + (size_t)ids[t] * 12 * 262;
        for (int d = (int)it.get_local_id(0); d < HID; d += 256) {
            const uint8_t* blk = row + (d >> 8) * 262;
            const float D = h2f_bits(ld16(blk));
            const float eff = D * ((float)blk[2 + ((d & 255) >> 6)] / 255.0f);
            out[(size_t)t * HID + d] = (float)(int8_t)blk[6 + (d & 255)] * eff;
        }
    });
}
inline void dfly_cat2(const float* a, const float* wa, const float* b, const float* wb, half* out, float eps) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float sa = 0, sb = 0;
        for (int i = tid; i < HID; i += 256) {
            sa += a[i] * a[i];
            sb += b[i] * b[i];
        }
        sa = wg_sum(it, sa, sh);
        sb = wg_sum(it, sb, sh);
        const float ia = sycl::rsqrt(sa / HID + eps), ib = sycl::rsqrt(sb / HID + eps);
        for (int i = tid; i < HID; i += 256) {
            out[i] = (half)(a[i] * ia * wa[i]);
            out[HID + i] = (half)(b[i] * ib * wb[i]);
        }
    });
}
inline void dfly_fuse(float* h0, const float* gate, const float* f) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) {
        for (int i = (int)it.get_local_id(0); i < HID; i += 256) h0[i] += gate[i] * f[i];
    });
}
inline void dfly_rms_resid(const float* o, const float* w, float* H, float eps, int rows) {
    Q().parallel_for(NR(rows, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const size_t r = it.get_group(0);
        const int tid = (int)it.get_local_id(0);
        const float* x = o + r * HID;
        float ss = 0;
        for (int i = tid; i < HID; i += 256) ss += x[i] * x[i];
        ss = wg_sum(it, ss, sh);
        const float inv = sycl::rsqrt(ss / HID + eps);
        float* h = H + r * HID;
        for (int i = tid; i < HID; i += 256) h[i] += x[i] * inv * w[i];
    });
}
inline void dfly_corr(const float* Hf, const float* D, const float* cw, const float* cb, half* hc_h, float* conf) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sh = lmem<float, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float dot = 0;
        for (int i = tid; i < HID; i += 256) {
            const float v = Hf[i] + D[i];
            hc_h[i] = (half)v;
            dot += v * cw[i];
        }
        dot = wg_sum(it, dot, sh);
        if (tid == 0) *conf = 1.0f / (1.0f + sycl::exp(-(dot + cb[0])));
    });
}
inline void dfly_h2f(const half* in, float* out, int n) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) {
        for (int i = (int)it.get_local_id(0); i < n; i += 256) out[i] = (float)in[i];
    });
}
// per-slot key lists (supports device-resident StepParams* d_step for command_graph replay!)
inline void dfly_lists_impl(const int32_t* dkeys_base, const int32_t* dnkc_base, int index_topk, int row_host, int a_host,
                            const StepParams* d_step, int32_t* dlist, int32_t* dnk, int lstride, int B) {
    Q().parallel_for(NR(B, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
        const int row = d_step ? d_step->cap_row : row_host;
        const int a = d_step ? d_step->a : a_host;
        const int32_t* ckeys = dkeys_base + (size_t)row * index_topk;
        const int n = dnkc_base[row];
        const int j = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        int32_t* l = dlist + (size_t)j * lstride;
        for (int i = tid; i < n; i += 256) l[i] = ckeys[i];
        for (int i = tid; i <= j; i += 256) l[n + i] = a + i;
        if (tid == 0) dnk[j] = n + j + 1;
    });
}
inline void argmax_rows(const float* logits, int V, int32_t* out, int rows) {
    Q().parallel_for(NR(rows, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sv = lmem<float, 32>(it);
        int* si = lmem<int, 32>(it);
        const int tid = (int)it.get_local_id(0);
        const float* lg = logits + (size_t)it.get_group(0) * V;
        float bv = -3.0e38f;
        int bi = 0;
        for (int i = tid; i < V; i += 1024)
            if (lg[i] > bv) { bv = lg[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(it, bv, o);
            const int oi = shfl_xor(it, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            float b2 = sv[0];
            int i2 = si[0];
            for (int q = 1; q < 32; ++q)
                if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
            out[it.get_group(0)] = i2;
        }
    });
}
// 4-pass MSB-first 8-bit radix top-N selection (8x fewer passes than 33-round binary search; local_accessor avoids lmem aliasing & guarantees local_space atomics)
inline void dfly_topn(const float* logits, int V, int N, int32_t* cand, int rows) {
    Q().submit([&](sycl::handler& ch) {
        sycl::local_accessor<int, 1> hist(256, ch);
        sycl::local_accessor<uint32_t, 1> su(2, ch);  // [0]=prefix, [1]=thr
        sycl::local_accessor<int, 1> si(2, ch);       // [0]=need,   [1]=cnt
        ch.parallel_for(NR(rows, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int tid = (int)it.get_local_id(0);
            const float* lg = logits + (size_t)it.get_group(0) * V;
            int32_t* out = cand + (size_t)it.get_group(0) * N;
            if (tid == 0) { su[0] = 0u; si[0] = N; }
            sycl::group_barrier(it.get_group());
            for (int pass = 0; pass < 4; ++pass) {
                const int shift = 24 - 8 * pass;
                const uint32_t mask = (pass == 0) ? 0u : (~0u << (shift + 8));
                if (tid < 256) hist[tid] = 0;
                sycl::group_barrier(it.get_group());
                const uint32_t pre = su[0];
                for (int i = tid; i < V; i += 1024) {
                    const uint32_t k = idx::f2key(lg[i]);
                    if ((k & mask) == pre) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> hbin(hist[(k >> shift) & 255u]);
                        hbin.fetch_add(1);
                    }
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    int acc = 0, d = 255;
                    const int need = si[0];
                    for (; d > 0; --d) {
                        if (acc + hist[d] >= need) break;
                        acc += hist[d];
                    }
                    si[0] = need - acc;
                    su[0] = pre | ((uint32_t)d << shift);
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) { su[1] = su[0]; si[1] = 0; }
            sycl::group_barrier(it.get_group());
            const uint32_t thr = su[1];
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> ctr(si[1]);
            for (int i = tid; i < V; i += 1024) {
                if (idx::f2key(lg[i]) > thr) {
                    const int s = ctr.fetch_add(1);
                    if (s < N) out[s] = i;
                }
            }
            sycl::group_barrier(it.get_group());
            if (si[1] < N) {
                for (int i = tid; i < V; i += 1024) {
                    if (idx::f2key(lg[i]) == thr) {
                        const int s = ctr.fetch_add(1);
                        if (s < N) out[s] = i;
                    }
                }
            }
        });
    });
}

// SLM-cached SG32 candidate scorer: 8 SG32 waves/WG share h[3072] in SLM (6 KiB), 8 candidates per WG (2x WGs & 8 weights/lane!)
inline void dfly_cand_score_rp(const uint8_t* W, int Nrows, const int32_t* cand, const half* h, float* score, int ncand) {
    const size_t pay_off = rp_pay_off(Nrows, 12);
    const int nwg = (ncand + 7) / 8;
    Q().submit([&](sycl::handler& ch) {
        sycl::local_accessor<half, 1> sH(HID, ch);  // 6 KiB SLM shared across 8 candidates!
        ch.parallel_for(NR(nwg, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 5, lane = tid & 31;
            half* sh_ptr = &sH[0];
            for (int i = tid; i < HID / 8; i += 256)
                *reinterpret_cast<sycl::vec<half, 8>*>(sh_ptr + i * 8) = *reinterpret_cast<const sycl::vec<half, 8>*>(h + i * 8);
            sycl::group_barrier(it.get_group());
            const int k = (int)it.get_group(0) * 8 + wave;
            if (k >= ncand) return;
            const size_t row = (size_t)cand[k];
            const int sub = lane >> 3;  // 0..3 (64-element scale group)
            float acc = 0.0f;
#pragma unroll 4
            for (int sb = 0; sb < 12; ++sb) {
                const uint8_t* hd = W + (row * 12 + sb) * 8;
                const uint8_t* pl = W + pay_off + (row * 12 + sb) * 256 + lane * 8;
                const float D = h2f_bits(ld16(hd));
                const float eff = D * ((float)hd[2 + sub] * (1.0f / 255.0f));
                const sycl::uchar8 raw = *reinterpret_cast<const sycl::uchar8*>(pl);
                const sycl::vec<half, 8> h0 = *reinterpret_cast<const sycl::vec<half, 8>*>(sh_ptr + sb * 256 + lane * 8);
                float sq = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) sq += (float)(int8_t)raw[j] * (float)h0[j];
                acc += eff * sq;
            }
            acc = sg_sum(it, acc);
            if (lane == 0) score[k] = acc;
        });
    });
}

// Non-RP FMT_S8 fallback candidate scorer (same 8-SG32 SLM-cached layout)
inline void dfly_cand_score_raw(const uint8_t* W, const int32_t* cand, const half* h, float* score, int ncand) {
    const int nwg = (ncand + 7) / 8;
    Q().submit([&](sycl::handler& ch) {
        sycl::local_accessor<half, 1> sH(HID, ch);
        ch.parallel_for(NR(nwg, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
            const int tid = (int)it.get_local_id(0);
            const int wave = tid >> 5, lane = tid & 31;
            half* sh_ptr = &sH[0];
            for (int i = tid; i < HID / 8; i += 256)
                *reinterpret_cast<sycl::vec<half, 8>*>(sh_ptr + i * 8) = *reinterpret_cast<const sycl::vec<half, 8>*>(h + i * 8);
            sycl::group_barrier(it.get_group());
            const int k = (int)it.get_group(0) * 8 + wave;
            if (k >= ncand) return;
            const size_t row = (size_t)cand[k];
            const int sub = lane >> 3;
            float acc = 0.0f;
#pragma unroll 4
            for (int sb = 0; sb < 12; ++sb) {
                const uint8_t* blk = W + (row * 12 + sb) * 262;
                const float D = h2f_bits(ld16(blk));
                const float eff = D * ((float)blk[2 + sub] * (1.0f / 255.0f));
                const sycl::vec<half, 8> h0 = *reinterpret_cast<const sycl::vec<half, 8>*>(sh_ptr + sb * 256 + lane * 8);
                const uint8_t* pl = blk + 6 + lane * 8;
                float sq = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) sq += (float)(int8_t)pl[j] * (float)h0[j];
                acc += eff * sq;
            }
            acc = sg_sum(it, acc);
            if (lane == 0) score[k] = acc;
        });
    });
}

inline void dfly_cand_argmax(const float* score, const int32_t* cand, int N, int32_t* out) {
    Q().parallel_for(NR(1, 1024), [=](sycl::nd_item<1> it) REQD_SG32 {
        float* sv = lmem<float, 32>(it);
        int* si = lmem<int, 32>(it);
        const int tid = (int)it.get_local_id(0);
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (int i = tid; i < N; i += 1024) {
            const int id = cand[i];
            if (score[i] > bv || (score[i] == bv && id < bi)) { bv = score[i]; bi = id; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(it, bv, o);
            const int oi = shfl_xor(it, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
        sycl::group_barrier(it.get_group());
        if (tid == 0) {
            float b2 = sv[0];
            int i2 = si[0];
            for (int q = 1; q < 32; ++q)
                if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
            *out = i2;
        }
    });
}
inline void add_inplace(float* a, const float* b, size_t n) {
    Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) a[i] += b[i];
    });
}

struct DFly2 {
    Runner& r;
    const hy4::Cfg& c;
    bool ok = false;
    int rv = 0;
    int32_t* d_cand = nullptr;
    float* d_cscore = nullptr;
    float *H = nullptr, *Hf = nullptr, *F1 = nullptr, *F2 = nullptr, *F3 = nullptr, *F4 = nullptr, *Ec = nullptr, *Dn = nullptr, *G = nullptr, *U = nullptr, *d_conf = nullptr, *hprev_f = nullptr;
    half *Hf_h = nullptr, *hn = nullptr, *qr_h = nullptr, *qa_h = nullptr, *ctx_h = nullptr, *o_h = nullptr, *mlp_h = nullptr, *hcin = nullptr, *act = nullptr, *hc_h = nullptr;
    half* hprev_h = nullptr;
    int32_t *d_bids = nullptr, *d_chain = nullptr, *dlist = nullptr, *dnk = nullptr, *d_prev_one = nullptr;
    int32_t *h_bids = nullptr, *h_chain = nullptr, *h_cand = nullptr;
    float *h_conf = nullptr, *h_cscore = nullptr;
    int lstride = 0;
    GemvMulti mg_attn[5], mg_mlp[5], mg_hc;
    int attn_fuse[5] = {}, mlp_fuse[5] = {};
    bool hc_fuse = false;
    double t_draft = 0;
    long n_draft = 0;
    long n_chain_evals = 0;
    long n_early_exits = 0;

#if HY4S_HAS_GRAPH
    struct DraftGraphKey {
        int K = 0, rv = 0, list_mask = 0, bb_only = 0;
        int TK[5] = {}, S[5] = {}, tps[5] = {};
        bool operator==(const DraftGraphKey& o) const {
            if (K != o.K || rv != o.rv || list_mask != o.list_mask || bb_only != o.bb_only) return false;
            for (int d = 0; d < 5; ++d)
                if (TK[d] != o.TK[d] || S[d] != o.S[d] || tps[d] != o.tps[d]) return false;
            return true;
        }
    };
    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    struct DraftGraphEntry {
        DraftGraphKey key;
        std::unique_ptr<ExecGraph> exec;
    };
    std::vector<DraftGraphEntry> dgraphs_;
#endif

    explicit DFly2(Runner& rr) : r(rr), c(rr.c) {}

    template <class T> T* al(size_t n) { return dalloc<T>(n); }
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
        Hf_h = al<half>(B * HID);
        hn = al<half>(B * HID);
        qr_h = al<half>(B * 1024);
        qa_h = al<half>(B * 32 * 576);
        ctx_h = al<half>(B * 32 * 512);
        o_h = al<half>(B * 8192);
        mlp_h = al<half>(B * 9216);
        hcin = al<half>(2 * HID);
        act = al<half>(HID);
        hc_h = al<half>(HID);
        hprev_h = al<half>(HID);
        d_bids = al<int32_t>(16);
        d_chain = al<int32_t>(16);
        d_prev_one = al<int32_t>(4);
        lstride = r.index_topk + 8;
        dlist = al<int32_t>((size_t)B * lstride);
        dnk = al<int32_t>(16);
        d_cand = al<int32_t>((size_t)B * 1024);
        d_cscore = al<float>(1024);
        h_bids = halloc<int32_t>(16);
        h_chain = halloc<int32_t>(16);
        h_conf = halloc<float>(16);
        h_cand = halloc<int32_t>((size_t)B * 1024);
        h_cscore = halloc<float>(1024);

        // Build SG16 multi-GEMV fusion descriptors for the 5 drafter layers + hidden_correction
        const std::string pf = "model.dfly_layers.";
        if (r.fuse_dec) {
            for (int d = 0; d < 5; ++d) {
                const std::string pa = pf + std::to_string(d) + ".self_attn";
                const std::string pm = pf + std::to_string(d) + ".mlp";
                mg_attn[d] = GemvMulti();
                if (r.mg_add(mg_attn[d], pa + ".q_a_proj.weight", F1, 0, hn, nullptr, nullptr, nullptr) &&
                    r.mg_add(mg_attn[d], pa + ".kv_a_proj_with_mqa.weight", F3, 0, hn, nullptr, nullptr, nullptr)) {
                    attn_fuse[d] = r.mg_add(mg_attn[d], pa + ".linear_gate.weight", F4, 0, hn, nullptr, nullptr, nullptr) ? 3 : 2;
                }
                mg_mlp[d] = GemvMulti();
                if (r.mg_add(mg_mlp[d], pm + ".gate_proj.weight", F1, 0, hn, nullptr, nullptr, nullptr) &&
                    r.mg_add(mg_mlp[d], pm + ".up_proj.weight", F2, 0, hn, nullptr, nullptr, nullptr)) {
                    mlp_fuse[d] = 2;
                }
            }
            mg_hc = GemvMulti();
            hc_fuse = r.mg_add(mg_hc, pf + "hidden_correction.gate_proj.weight", G, 0, hcin, nullptr, nullptr, nullptr) &&
                      r.mg_add(mg_hc, pf + "hidden_correction.up_proj.weight", U, 0, hcin, nullptr, nullptr, nullptr);
        }
        wait();
        ok = true;
        return true;
    }

    void snap_h(int row) { d2d(hprev_h, r.hfin + (size_t)row * HID, HID * sizeof(half)); }

    bool can_draft(int a, int K) const {
        if (!ok || K < 1 || K > DFLY_KMAX) return false;
        if (a + K + 1 > r.slots_sparse) return false;
        if (r.cap_p0 < 0) return false;
        const int row = a - 1 - r.cap_p0;
        if (row < 0 || row >= r.cap_rows) return false;
        return true;
    }

    void draft_backbone_body(int a, int row, int K, const StepParams* dp, const int* g_TK, const int* g_S, const int* g_tps) {
        const int B = K + 1;
        const int V = c.vocab;
        const std::string pf = "model.dfly_layers.";
        const uint8_t* emb = (const uint8_t*)r.tw("model.embed_tokens.weight");
        embed_rows_f32(emb, d_bids, H, B);
        dfly_h2f(hprev_h, hprev_f, HID);
        dfly_cat2(H, wf(pf + "mtp_fusion.enorm.weight"), hprev_f, wf(pf + "mtp_fusion.hnorm.weight"), hcin, c.rms_eps);
        r.lin(pf + "mtp_fusion.eh_proj.weight", hcin, F1, 1);
        dfly_fuse(H, wf(pf + "mtp_fusion.gate"), F1);
        for (int d = 0; d < 5; ++d) {
            const int L = DFLY_LAYERS[d];
            const std::string pa = pf + std::to_string(d) + ".self_attn";
            const std::string pm = pf + std::to_string(d) + ".mlp";
            const bool use_list = r.cap_list[d];
            rms_rows(H, wf(pf + std::to_string(d) + ".input_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            const int af = attn_fuse[d];
            if (af > 0) k_gemv_multi(mg_attn[d], B);
            else r.lin(pa + ".q_a_proj.weight", hn, F1, B);
            rms_rows(F1, wf(pa + ".q_a_layernorm.weight"), nullptr, qr_h, c.q_lora, c.rms_eps, B, 256);
            r.lin(pa + ".q_b_proj.weight", qr_h, F2, B);
            if (af == 0) r.lin(pa + ".kv_a_proj_with_mqa.weight", hn, F3, B);
            if (dp) mla_write_kv_dev(F3, wf(pa + ".kv_a_layernorm.weight"), r.kcache[L], r.slots_sparse, B, dp, true, 0, (float)c.theta_sparse, c.rms_eps);
            else mla_write_kv(F3, wf(pa + ".kv_a_layernorm.weight"), r.kcache[L], r.slots_sparse, B, a, 0, (float)c.theta_sparse, c.rms_eps);
            const std::string kvb_nm = pa + ".kv_b_proj.weight";
            const int kvb_fmt = r.ti(kvb_nm).fmt;
            const void* kvb_w = r.tw(kvb_nm);
            mla_prep_q_sk(kvb_fmt, kvb_w, F2, qa_h, B, 32, a, dp, true, (float)c.theta_sparse);
            if (af < 3) r.lin(pa + ".linear_gate.weight", hn, F4, B);
            MlaAttnArgs ma;
            ma.M = B; ma.NH = 32; ma.p0 = a; ma.cache_slots = r.slots_sparse;
            if (use_list) {
                dfly_lists_impl(r.dkeys[d], r.dnkc[d], r.index_topk, row, a, dp, dlist, dnk, lstride, B);
                ma.mode = MLA_LIST; ma.keys = dlist; ma.nkeys = dnk; ma.kmax = lstride;
            } else {
                ma.mode = MLA_CAUSAL;
            }
            if (dp) {
                mla_attend_dev(qa_h, r.kcache[L], wf(pa + ".learnable_sink_param"), ctx_h, ma, dp, true, g_TK[d], g_S[d], g_tps[d], r.mla_ws, r.mla_ws_bytes);
            } else {
                r.ensure_mla_ws(mla_workspace_bytes(ma));
                mla_attend(qa_h, r.kcache[L], wf(pa + ".learnable_sink_param"), ctx_h, ma, r.mla_ws, r.mla_ws_bytes);
            }
            mla_vup_sk(kvb_fmt, kvb_w, ctx_h, F4, o_h, B, 32);
            r.lin(pa + ".o_proj.weight", o_h, F1, B);
            dfly_rms_resid(F1, wf(pa + ".post_proj_norm.weight"), H, c.rms_eps, B);
            rms_rows(H, wf(pf + std::to_string(d) + ".post_attention_layernorm.weight"), nullptr, hn, HID, c.rms_eps, B, 256);
            if (mlp_fuse[d] == 2) {
                k_gemv_multi(mg_mlp[d], B);
            } else {
                r.lin(pm + ".gate_proj.weight", hn, F1, B);
                r.lin(pm + ".up_proj.weight", hn, F2, B);
            }
            swiglu_h(F1, F2, mlp_h, (size_t)B * 9216);
            r.lin(pm + ".down_proj.weight", mlp_h, F3, B);
            add_inplace(H, F3, (size_t)B * HID);
        }
        rms_rows(H, wf(pf + "final_norm.weight"), Hf, Hf_h, HID, c.rms_eps, B, 256);
        const hy4::TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const bool lm_rp = r.g.is_rp("lm_head.weight");
        const int N = std::min(rv, 1024);
        if (N > 0) {
            linear_auto(lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), Hf_h + HID, r.logits, K, lm_rp);
            dfly_topn(r.logits, V, N, d_cand, K);
        }
    }

    void draft_body(int a, int row, int K, const StepParams* dp, const int* g_TK, const int* g_S, const int* g_tps) {
        draft_backbone_body(a, row, K, dp, g_TK, g_S, g_tps);
        const int V = c.vocab;
        const std::string pf = "model.dfly_layers.";
        const uint8_t* emb = (const uint8_t*)r.tw("model.embed_tokens.weight");
        const hy4::TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const bool lm_rp = r.g.is_rp("lm_head.weight");
        const int N = std::min(rv, 1024);
        for (int j = 1; j <= K; ++j) {
            embed_rows_f32(emb, d_chain + (j - 1), Ec, 1);
            dfly_cat2(Hf + (size_t)j * HID, wf(pf + "hidden_correction.hidden_norm.weight"), Ec, wf(pf + "hidden_correction.embed_norm.weight"), hcin, c.rms_eps);
            if (hc_fuse) {
                k_gemv_multi(mg_hc, 1);
            } else {
                r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
                r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
            }
            swiglu_h(G, U, act, (size_t)HID);
            r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
            dfly_corr(Hf + (size_t)j * HID, Dn, wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), hc_h, d_conf + j);
            if (N > 0) {
                if (lm_rp) dfly_cand_score_rp((const uint8_t*)lmw, (int)lm.rows(), d_cand + (size_t)(j - 1) * N, hc_h, d_cscore, N);
                else dfly_cand_score_raw((const uint8_t*)lmw, d_cand + (size_t)(j - 1) * N, hc_h, d_cscore, N);
                dfly_cand_argmax(d_cscore, d_cand + (size_t)(j - 1) * N, N, d_chain + j);
            } else {
                linear_auto(lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), hc_h, r.logits, 1, lm_rp);
                argmax_row(r.logits, V, d_chain + j);
            }
        }
    }

    // Run the 5 DFly2 backbone layers + batched base-logits top-N once (with SYCL command_graph capture!).
    bool run_block_backbone(int a, int xa, int K, int& N_out) {
        if (!can_draft(a, K)) return false;
        const int B = K + 1;
        const int row = a - 1 - r.cap_p0;
        const int N = std::min(rv, 1024);
        N_out = N;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        h2d(d_bids, h_bids, (size_t)B * 4);

        bool ran_graph = false;
#if HY4S_HAS_GRAPH
        if (r.graph_enabled && !r.prof2 && !r.dump && !r.tap) {
            DraftGraphKey dk;
            dk.K = K; dk.rv = N; dk.list_mask = 0; dk.bb_only = 1;
            size_t max_ws = 0;
            for (int d = 0; d < 5; ++d) {
                if (r.cap_list[d]) dk.list_mask |= (1 << d);
                MlaAttnArgs ma;
                ma.M = B; ma.NH = 32; ma.p0 = a; ma.cache_slots = r.slots_sparse;
                if (r.cap_list[d]) { ma.mode = MLA_LIST; ma.kmax = lstride; }
                else ma.mode = MLA_CAUSAL;
                mla_choose_splits(ma, dk.TK[d], dk.S[d], dk.tps[d]);
                max_ws = std::max(max_ws, mla_workspace_bytes(ma));
            }
            r.ensure_mla_ws(max_ws);
            ExecGraph* eg = nullptr;
            for (auto& e : dgraphs_)
                if (e.key == dk) { eg = e.exec.get(); break; }
            if (!eg) {
                if (dgraphs_.size() >= 12) { wait(); dgraphs_.clear(); }
                try {
                    wait();
                    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
                    cg.begin_recording(Q());
                    draft_backbone_body(a, row, K, r.d_step, dk.TK, dk.S, dk.tps);
                    cg.end_recording(Q());
                    dgraphs_.push_back(DraftGraphEntry{dk, std::make_unique<ExecGraph>(cg.finalize())});
                    eg = dgraphs_.back().exec.get();
                } catch (const sycl::exception& e) {
                    fprintf(stderr, "[dfly2-graph] capture failed (%s), eager fallback\n", e.what());
                    r.graph_enabled = false;
                }
            }
            if (eg) {
                r.h_step->a = a;
                r.h_step->cap_row = row;
                r.h_step->tree_on = 0;
                h2d(r.d_step, r.h_step, sizeof(StepParams));
                Q().ext_oneapi_graph(*eg);
                ran_graph = true;
            }
        }
#endif
        if (!ran_graph) draft_backbone_body(a, row, K, nullptr, nullptr, nullptr, nullptr);
        if (N > 0) d2h(h_cand, d_cand, (size_t)K * N * 4);
        return true;
    }

    // Evaluate ONE hidden_correction + candidate scorer slot j (1..K) given previous token prev_tok.
    float eval_slot_host(int j, int32_t prev_tok, int N, std::vector<float>& full_lg) {
        const std::string pf = "model.dfly_layers.";
        const uint8_t* emb = (const uint8_t*)r.tw("model.embed_tokens.weight");
        const hy4::TInfo& lm = r.ti("lm_head.weight");
        const void* lmw = r.tw("lm_head.weight");
        const bool lm_rp = r.g.is_rp("lm_head.weight");
        h_chain[0] = prev_tok;
        h2d(d_prev_one, h_chain, 4);
        embed_rows_f32(emb, d_prev_one, Ec, 1);
        dfly_cat2(Hf + (size_t)j * HID, wf(pf + "hidden_correction.hidden_norm.weight"), Ec, wf(pf + "hidden_correction.embed_norm.weight"), hcin, c.rms_eps);
        if (hc_fuse) {
            k_gemv_multi(mg_hc, 1);
        } else {
            r.lin(pf + "hidden_correction.gate_proj.weight", hcin, G, 1);
            r.lin(pf + "hidden_correction.up_proj.weight", hcin, U, 1);
        }
        swiglu_h(G, U, act, (size_t)HID);
        r.lin(pf + "hidden_correction.down_proj.weight", act, Dn, 1);
        dfly_corr(Hf + (size_t)j * HID, Dn, wf(pf + "confidence_head.proj.weight"), wf(pf + "confidence_head.proj.bias"), hc_h, d_conf);
        if (N > 0) {
            if (lm_rp) dfly_cand_score_rp((const uint8_t*)lmw, (int)lm.rows(), d_cand + (size_t)(j - 1) * N, hc_h, d_cscore, N);
            else dfly_cand_score_raw((const uint8_t*)lmw, d_cand + (size_t)(j - 1) * N, hc_h, d_cscore, N);
            d2h(h_cscore, d_cscore, (size_t)N * 4);
            d2h(h_conf, d_conf, 4);
            wait();
        } else {
            linear_auto(lm.fmt, lmw, (int)lm.rows(), (int)lm.cols(), hc_h, r.logits, 1, lm_rp);
            full_lg.resize((size_t)c.vocab);
            d2h(full_lg.data(), r.logits, (size_t)c.vocab * 4);
            d2h(h_conf, d_conf, 4);
            wait();
        }
        ++n_chain_evals;
        return h_conf[0];
    }

    // Tree-8 + Early-Exit + Probabilistic-aware drafter.
    bool draft_tree(int a, int xa, int K, TreeSpec& tree, float* node_conf, float* node_reach, const SampleOpts* samp, const std::vector<int32_t>* tok_hist, std::mt19937_64* rng,
                    std::vector<SparseDist>* node_qdist, float early_cut, bool enable_tree, const std::function<double(float)>& pcal_fn) {
        const double t0 = now_s();
        int N = 0;
        if (!run_block_backbone(a, xa, K, N)) return false;
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
                    const double ratio2 = std::exp(-gap);
                    const double p2 = std::min(1.0 - p_acc, p_acc * ratio2 * 0.85);
                    if (p2 > 0.05) {
                        frontier.push_back({u, d_next, id2, (float)std::clamp(p2, 0.01, 0.95), (float)(r_u * p2), {}});
                    }
                }
            } else {
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

        expand(0);
        const int max_nodes = std::min(K + 1, DFLY_BMAX);
        while (tree.M < max_nodes && !frontier.empty()) {
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
            if (tree.M < max_nodes && chosen.depth < K && (early_cut <= 0.f || chosen.reach >= early_cut)) {
                expand(idx);
            }
        }
        tree.build_anc();
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }

    // Legacy linear-only draft (kept for direct unit tests / comparisons)
    bool draft(int a, int xa, int K, int* y, float* conf) {
        if (!can_draft(a, K)) return false;
        const double t0 = now_s();
        const int B = K + 1;
        const int row = a - 1 - r.cap_p0;
        h_bids[0] = xa;
        for (int i = 1; i <= K; ++i) h_bids[i] = DFLY_MASK_ID;
        h2d(d_bids, h_bids, B * 4);
        h2d(d_chain, h_bids, 4);

        bool ran_graph = false;
#if HY4S_HAS_GRAPH
        if (r.graph_enabled && !r.prof2 && !r.dump && !r.tap) {
            DraftGraphKey dk;
            dk.K = K; dk.rv = std::min(rv, 1024); dk.list_mask = 0; dk.bb_only = 0;
            size_t max_ws = 0;
            for (int d = 0; d < 5; ++d) {
                if (r.cap_list[d]) dk.list_mask |= (1 << d);
                MlaAttnArgs ma;
                ma.M = B; ma.NH = 32; ma.p0 = a; ma.cache_slots = r.slots_sparse;
                if (r.cap_list[d]) { ma.mode = MLA_LIST; ma.kmax = lstride; }
                else ma.mode = MLA_CAUSAL;
                mla_choose_splits(ma, dk.TK[d], dk.S[d], dk.tps[d]);
                max_ws = std::max(max_ws, mla_workspace_bytes(ma));
            }
            r.ensure_mla_ws(max_ws);
            ExecGraph* eg = nullptr;
            for (auto& e : dgraphs_)
                if (e.key == dk) { eg = e.exec.get(); break; }
            if (!eg) {
                if (dgraphs_.size() >= 12) { wait(); dgraphs_.clear(); }
                try {
                    wait();
                    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable> cg(Q().get_context(), Q().get_device());
                    cg.begin_recording(Q());
                    draft_body(a, row, K, r.d_step, dk.TK, dk.S, dk.tps);
                    cg.end_recording(Q());
                    dgraphs_.push_back(DraftGraphEntry{dk, std::make_unique<ExecGraph>(cg.finalize())});
                    eg = dgraphs_.back().exec.get();
                } catch (const sycl::exception& e) {
                    fprintf(stderr, "[dfly2-graph] capture failed (%s), eager fallback\n", e.what());
                    r.graph_enabled = false;
                }
            }
            if (eg) {
                r.h_step->a = a;
                r.h_step->cap_row = row;
                r.h_step->tree_on = 0;
                h2d(r.d_step, r.h_step, sizeof(StepParams));
                Q().ext_oneapi_graph(*eg);
                ran_graph = true;
            }
        }
#endif
        if (!ran_graph) draft_body(a, row, K, nullptr, nullptr, nullptr, nullptr);

        d2h(h_chain, d_chain, B * 4);
        d2h(h_conf, d_conf, B * 4);
        wait();
        for (int j = 1; j <= K; ++j) {
            y[j] = h_chain[j];
            conf[j] = h_conf[j];
        }
        t_draft += now_s() - t0;
        ++n_draft;
        return true;
    }
};

}  // namespace hy4s
