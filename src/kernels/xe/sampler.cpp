// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sampler.cpp - the Xe port of Strata's src/kernels/cuda/sampler.cu: the sampler chain in llama.cpp's order
// (penalties -> top_k -> top_p -> min_p -> temperature -> pick), one work-group of 1024 per token over the
// vocabulary.  See the CUDA source for the order, the tie rule and the cost notes.
//
// The (value, index) argmax reductions are xor butterflies.  Their comparison is a total order (larger value, then
// smaller index), and lane 0 of a butterfly combines the same lanes in the same order as CUDA's shuffle-down tree.
#include "strata/kernels/sampler.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/coupled_draft.hpp"
#include "device_caps.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
constexpr int KMAX = 64;

// Philox 4x32-10, counter-based on (seed, position)
inline void philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = sycl::mul_hi(0x9E3779B9u, c0);
    const uint32_t hi1 = sycl::mul_hi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

inline float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);   // 24 bits: uniform in [0,1), never 1.0
}

// GUMBEL-MAX COUPLING (STRATA_SPEC_GUMBEL=1, upstream 6381df34, from llama.cpp-lab's --spec-coupled).  The pick is
// argmax_i p_i / E_i with E_i = -log(u_i) ~ Exp(1) and u_i a hash of (seed, counter, TOKEN ID) - exactly equivalent
// to argmax(log p_i + Gumbel_i), so it is an exact sample of p.  Unlike the inverse-CDF pick over a probability-sorted
// list, the noise a token gets does not depend on which other tokens survived the cut, so a drafter whose candidate
// set differs from the target's still agrees with it on the tokens they share.  The MTP drafter keys on the draft's
// token id (via sub_to_id), the target on its row's token id.  D float (a GPU without FP64): u from the hash's top 24
// bits, still in (0, 1).
template <typename D>
inline D gumbel_exp(uint64_t seed, uint64_t counter, uint32_t token) {
    uint64_t z = seed ^ (counter * 0x9E3779B97F4A7C15ull) ^ ((uint64_t) token * 0xD1B54A32D192ED03ull);
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    if constexpr (sizeof(D) == 8) {
        const D u = ((D) (z >> 11) + 0.5) * (1.0 / 9007199254740992.0);   // (0, 1), never 0 or 1
        return -sycl::log(u);
    } else {
        const D u = ((D) (uint32_t) (z >> 40) + 0.5f) * (1.0f / 16777216.0f);
        return -sycl::log(u);
    }
}

inline int history_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

// llama_sampler_penalties_apply: the repeat penalty multiplies non-positive logits and divides positive ones; the
// presence penalty is a boolean
inline float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

inline void argmax_step(const sycl::sub_group& sg, float& bv, int& best) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::permute_group_by_xor(sg, bv, off);
        const int oi = sycl::permute_group_by_xor(sg, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
}

// The penalty window's tail and its membership bitmap in local memory.  The bitmap is sized only when penalties
// are on, so it is used only for a non-empty window.
struct Window {
    const int* hrow = nullptr;
    int hlen = 0;
    bool use_bits = false;
};

template <int THREADS>
inline Window load_window(const sycl::nd_item<1>& it, const int* history, int history_len, int last_n, int n_vocab,
                          int t, unsigned* bits) {
    Window w;
    w.hrow = history ? history + (size_t) t * history_len : nullptr;
    if (w.hrow) {
        w.hlen = last_n < history_len ? last_n : history_len;
        if (w.hlen < 0) w.hlen = 0;
        w.hrow += history_len - w.hlen;          // the window is the TAIL
    }
    const int bits_words = (n_vocab + 31) / 32;
    w.use_bits = w.hrow != nullptr && w.hlen > 0 && bits_words > 0;
    if (w.use_bits) {
        const int tid = (int) it.get_local_id(0);
        for (int i = tid; i < bits_words; i += THREADS) bits[i] = 0u;
        sycl::group_barrier(it.get_group());
        for (int i = tid; i < w.hlen; i += THREADS) {
            const int v = w.hrow[i];
            if (v >= 0 && v < n_vocab) {   // an id outside the vocabulary is never a candidate
                sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(bits[v >> 5]).fetch_or(1u << (v & 31));
            }
        }
        sycl::group_barrier(it.get_group());
    }
    return w;
}

inline int hit_count(const Window& w, const unsigned* bits, int v) {
    if (!w.use_bits || !(bits[v >> 5] & (1u << (v & 31)))) return 0;
    return history_count(w.hrow, w.hlen, v);
}

// The block argmax: per-lane strided scan (lowest index wins a tie), warp reduce, then warp 0 over the warp winners.
// Returns the winner in every work-item of warp 0; others get their own partial.
template <int THREADS, class Score>
inline void block_argmax(const sycl::nd_item<1>& it, int n_vocab, float* sv, int* si, Score score, float& wv, int& wi) {
    const sycl::sub_group sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(0);
    float bv = -std::numeric_limits<float>::infinity();
    int best = n_vocab;   // "no candidate": loses every comparison to a real one
    for (int v = tid; v < n_vocab; v += THREADS) {
        float s;
        if (!score(v, s)) continue;
        if (s > bv) { bv = s; best = v; }
    }
    argmax_step(sg, bv, best);
    constexpr int NW = THREADS / WARP;
    const int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    sycl::group_barrier(it.get_group());
    wv = -std::numeric_limits<float>::infinity();
    wi = n_vocab;
    if (warp == 0) {
        wv = lane < NW ? sv[lane] : -std::numeric_limits<float>::infinity();
        wi = lane < NW ? si[lane] : n_vocab;
        argmax_step(sg, wv, wi);
    }
}

// top_p, min_p, temperature and one Philox draw over a row's top_k list (ids, logits in the selection order),
// computed redundantly by every work-item
template <typename D>
inline int pick_from_list(const int* sel_ids, const float* sel_logit, int k, const SamplerParams& pp, int t,
                          float* probability = nullptr, const int32_t* sub_to_id = nullptr) {
    const float inv_t = pp.temperature > 0.0f ? 1.0f / pp.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (pp.top_p < 1.0f) {
        D sum = 0;
        for (int i = 0; i < k; ++i) sum += sycl::exp((D) sel_logit[i] - (D) mx);
        D cum = 0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += sycl::exp((D) sel_logit[i] - (D) mx) / sum;
            if (cum >= (D) pp.top_p) { cut = i + 1; break; }
        }
        if (cut < pp.min_keep) cut = pp.min_keep < k ? pp.min_keep : k;
        n_keep = cut;
    }
    if (pp.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log(pp.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = sycl::fmax(smx, scaled(i));
    D sum = 0;
    for (int i = 0; i < n_keep; ++i) sum += sycl::exp((D) scaled(i) - (D) smx);
    int selected = n_keep - 1;
    int pick = sel_ids[selected];
    if (pp.gumbel) {
        D best = -1;
        for (int i = 0; i < n_keep; ++i) {
            const int id = sub_to_id != nullptr ? sub_to_id[sel_ids[i]] : sel_ids[i];
            const D r = sycl::exp((D) scaled(i) - (D) smx) / sum / gumbel_exp<D>(pp.seed, pp.counter + (uint64_t) t, (uint32_t) id);
            if (r > best) { best = r; pick = sel_ids[i]; selected = i; }
        }
    } else {
        const float u = philox_uniform(pp.seed, pp.counter + (uint64_t) t);
        D cum = 0;
        for (int i = 0; i < n_keep; ++i) {
            cum += sycl::exp((D) scaled(i) - (D) smx) / sum;
            if ((D) u < cum) { pick = sel_ids[i]; selected = i; break; }
        }
    }
    if (probability) *probability = (float) (sycl::exp((D) scaled(selected) - (D) smx) / sum);
    return pick;
}

}  // namespace

namespace {
// The sampler's kernel with THREADS work-items a token (1024 where the device takes it: the argmax and the top-k are
// the same with any count, ties going to the lowest id); D sums the kept probabilities, double where the device has
// FP64 and float elsewhere (device_caps.hpp)
template <int THREADS, typename D>
sycl::event sample_launch(sycl::queue& q, const float* logits, int n_tokens, int n_vocab, const int* history,
                          int history_len, SamplerParams pp, bool greedy, size_t words, int* out) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> bits(sycl::range<1>(words), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(THREADS / WARP), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(THREADS / WARP), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(KMAX), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(KMAX), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int t = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* l = logits + (size_t) t * n_vocab;
            unsigned* b = &bits[0];
            const Window w = load_window<THREADS>(it, history, history_len, pp.penalty_last_n, n_vocab, t, b);

            if (greedy) {
                float wv;
                int wi;
                block_argmax<THREADS>(it, n_vocab, &sv[0], &si[0], [&](int v, float& s) {
                    s = apply_penalties(l[v], hit_count(w, b, v), pp);
                    return true;
                }, wv, wi);
                // a tie between two -inf candidates leaves wi == n_vocab, and the serial version answered 0
                if (tid == 0) out[t] = (wi < n_vocab) ? wi : 0;
                return;
            }

            int k = (pp.top_k > 0 && pp.top_k < KMAX) ? pp.top_k : KMAX;
            if (k > n_vocab) k = n_vocab;

            // top_k: k rounds of a block argmax over the not-yet-taken
            for (int i = 0; i < k; ++i) {
                float wv;
                int wi;
                block_argmax<THREADS>(it, n_vocab, &sv[0], &si[0], [&](int v, float& s) {
                    for (int j = 0; j < i; ++j) if (sel_ids[j] == v) return false;
                    s = apply_penalties(l[v], hit_count(w, b, v), pp);
                    return true;
                }, wv, wi);
                if (tid == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
                sycl::group_barrier(it.get_group());
            }

            const int pick = pick_from_list<D>(&sel_ids[0], &sel_logit[0], k, pp, t);
            if (tid == 0) out[t] = pick;
        });
    });
}

// The sampled path split over the GPU (upstream 47d6894): the top_k list is the first k logits in the selection
// order - the penalised value descending, the id ascending; NaN and -inf are never taken - which the kernel above
// builds with k argmaxes over the whole row on one work-group.  The first k of the row lie within the first k of
// each block of SPLIT_BL logits, and lists of blocks taken in id order, concatenated, keep equal values in id order:
// stage 1 takes each block's first k on a work-group of its own, stage 2 the row's first k from the blocks' lists,
// then the same pick.  The same list, so the same token.
constexpr int SPLIT_BL = 4096;   // logits a block
constexpr int SPLIT_T = 256;     // work-items a group, both stages
constexpr int SPLIT_MAX = 4096;  // the most list entries a row's stage 2 takes (blocks x k)

// one round of an argmax over a work-group's `n` local values (index order is the id order): the winning index, or
// `n` for none
inline void group_argmax_local(const sycl::nd_item<2>& it, const float* val, const uint8_t* taken, int n, float* sv,
                               int* si, float& wv, int& wi) {
    const sycl::sub_group sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(1);
    float bv = -std::numeric_limits<float>::infinity();
    int best = n;
    for (int c = tid; c < n; c += SPLIT_T) {
        if (taken[c]) continue;
        const float x = val[c];
        if (x > bv) { bv = x; best = c; }
    }
    argmax_step(sg, bv, best);
    const int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    sycl::group_barrier(it.get_group());
    wv = -std::numeric_limits<float>::infinity();
    wi = n;
    if (warp == 0) {
        constexpr int nw = SPLIT_T / WARP;
        wv = lane < nw ? sv[lane] : -std::numeric_limits<float>::infinity();
        wi = lane < nw ? si[lane] : n;
        argmax_step(sg, wv, wi);
    }
}

sycl::event split_stage1(sycl::queue& q, const float* logits, int n_tokens, int n_vocab, const int* history,
                         int history_len, SamplerParams pp, int k, int nb, float* lv, int* li) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(SPLIT_BL), h);
        sycl::local_accessor<uint8_t, 1> taken(sycl::range<1>(SPLIT_BL), h);
        sycl::local_accessor<unsigned, 1> bits(sycl::range<1>(SPLIT_BL / 32), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(SPLIT_T / WARP), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(SPLIT_T / WARP + 1), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) n_tokens, (size_t) nb * SPLIT_T}, {1, (size_t) SPLIT_T}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int t = (int) it.get_group(0), j = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            const int base = j * SPLIT_BL, n = sycl::min(SPLIT_BL, n_vocab - base);
            const float* l = logits + (size_t) t * n_vocab;
            // the penalty window, as load_window takes it, as a bitmap of this block's ids
            const int* hrow = history ? history + (size_t) t * history_len : nullptr;
            int hlen = 0;
            if (hrow) {
                hlen = pp.penalty_last_n < history_len ? pp.penalty_last_n : history_len;
                if (hlen < 0) hlen = 0;
                hrow += history_len - hlen;
            }
            const bool use_bits = hrow != nullptr && hlen > 0;
            if (use_bits) {
                for (int i = tid; i < SPLIT_BL / 32; i += SPLIT_T) bits[i] = 0u;
                sycl::group_barrier(it.get_group());
                for (int i = tid; i < hlen; i += SPLIT_T) {
                    const int v = hrow[i] - base;
                    if (v >= 0 && v < n)
                        sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>(bits[v >> 5]).fetch_or(1u << (v & 31));
                }
                sycl::group_barrier(it.get_group());
            }
            for (int c = tid; c < n; c += SPLIT_T) {
                const int cnt = use_bits && (bits[c >> 5] & (1u << (c & 31))) ? history_count(hrow, hlen, base + c) : 0;
                sc[c] = apply_penalties(l[base + c], cnt, pp);
                taken[c] = 0;
            }
            sycl::group_barrier(it.get_group());
            float* ov = lv + ((size_t) t * nb + j) * k;
            int* oi = li + ((size_t) t * nb + j) * k;
            for (int i = 0; i < k; ++i) {
                float wv;
                int wi;
                group_argmax_local(it, &sc[0], &taken[0], n, &sv[0], &si[0], wv, wi);
                if (tid == 0) {
                    ov[i] = wv;
                    oi[i] = wi < n ? base + wi : n_vocab;   // n_vocab: no candidate left
                    if (wi < n) taken[wi] = 1;
                    si[SPLIT_T / WARP] = wi < n;
                }
                sycl::group_barrier(it.get_group());
                if (!si[SPLIT_T / WARP]) {   // none left: the rest of the list is empty
                    for (int r = i + 1 + tid; r < k; r += SPLIT_T) {
                        ov[r] = -std::numeric_limits<float>::infinity();
                        oi[r] = n_vocab;
                    }
                    break;
                }
            }
        });
    });
}

template <typename D>
sycl::event split_stage2(sycl::queue& q, int n_tokens, int n_vocab, SamplerParams pp, int k, int nb, const float* lv,
                         const int* li, int* out, const SamplerParams* device_params = nullptr,
                         const int32_t* step_rec = nullptr, const int32_t* sub_to_id = nullptr,
                         int32_t* ring_out = nullptr, float* probability = nullptr) {
    const int m = nb * k;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cv(sycl::range<1>((size_t) m), h);
        sycl::local_accessor<int, 1> ci(sycl::range<1>((size_t) m), h);
        sycl::local_accessor<uint8_t, 1> taken(sycl::range<1>((size_t) m), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(SPLIT_T / WARP), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(SPLIT_T / WARP), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(KMAX), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(KMAX), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) n_tokens, (size_t) SPLIT_T}, {1, (size_t) SPLIT_T}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int t = (int) it.get_group(0), tid = (int) it.get_local_id(1);
            for (int c = tid; c < m; c += SPLIT_T) {
                const int id = li[(size_t) t * m + c];
                // an empty entry is never a candidate (the stage-1 sentinel)
                cv[c] = id < n_vocab ? lv[(size_t) t * m + c] : -std::numeric_limits<float>::infinity();
                ci[c] = id;
                taken[c] = 0;
            }
            sycl::group_barrier(it.get_group());
            for (int i = 0; i < k; ++i) {
                float wv;
                int wi;
                group_argmax_local(it, &cv[0], &taken[0], m, &sv[0], &si[0], wv, wi);
                if (tid == 0) {
                    sel_ids[i] = wi < m ? ci[wi] : 0;   // as the one-group kernel: no candidate answers id 0
                    sel_logit[i] = wv;
                    if (wi < m) taken[wi] = 1;
                }
                sycl::group_barrier(it.get_group());
            }
            SamplerParams chain = device_params ? *device_params : pp;
            int keep = device_params ? ((chain.top_k > 0 && chain.top_k < KMAX) ? chain.top_k : KMAX) : k;
            keep = sycl::min(keep, n_vocab);
            if (step_rec) chain.counter = core::coupled_draft_counter(step_rec[0]);
            float prob = 0;
            const int pick = pick_from_list<D>(&sel_ids[0], &sel_logit[0], keep, chain, t, &prob, sub_to_id);
            if (tid == 0) {
                const int id = sub_to_id ? sub_to_id[pick] : pick;
                out[t] = id;
                if (ring_out) *ring_out = id;
                if (probability) *probability = prob;
            }
        });
    });
}

// the lists' scratch: grown on demand, never freed while the process runs (an event may still read the old one)
struct SplitScratch {
    std::mutex mu;
    void* p = nullptr;
    size_t bytes = 0;
    std::vector<void*> retired;
};
void* split_scratch(sycl::queue& q, size_t bytes) {
    static SplitScratch sc;
    std::lock_guard<std::mutex> lock(sc.mu);
    if (sc.bytes < bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (p == nullptr) return nullptr;
        if (sc.p) sc.retired.push_back(sc.p);
        sc.p = p;
        sc.bytes = bytes;
    }
    return sc.p;
}

bool gumbel_env() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_GUMBEL");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}
}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p_in, int* out, void* stream) {
    SamplerParams p = p_in;
    p.gumbel = p_in.gumbel || gumbel_env();   // the target's pick; greedy rows never read it
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0))
        throw core::DeviceError("sample_tokens: penalty_last_n " + std::to_string(p.penalty_last_n) +
                                " needs a history (len " + std::to_string(history_len) + ")");
    auto& q = core::Runtime::get().stream(stream);
    const size_t words = (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
                             ? (size_t) ((n_vocab + 31) / 32) : 1;   // the penalty bitmap
    const SamplerParams pp = p;
    const bool greedy = p.greedy || p.temperature <= 0.0f;
    const int wg = xe::work_group_upto_1024(q);
    auto launch = xe::has_fp64(q)
                      ? (wg == 1024 ? sample_launch<1024, double> : wg == 512 ? sample_launch<512, double>
                                                                       : sample_launch<256, double>)
                      : (wg == 1024 ? sample_launch<1024, float> : wg == 512 ? sample_launch<512, float>
                                                                      : sample_launch<256, float>);
    // the sampled path split over the GPU; STRATA_OLD_SAMPLER=1: the one-group kernel (A/B and reference)
    static const bool old_sampler = std::getenv("STRATA_OLD_SAMPLER") != nullptr;
    int k = (pp.top_k > 0 && pp.top_k < KMAX) ? pp.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;
    const int nb = (n_vocab + SPLIT_BL - 1) / SPLIT_BL;
    if (!greedy && !old_sampler && nb * k <= SPLIT_MAX) {
        const size_t lists = (size_t) n_tokens * nb * k;
        if (void* scratch = split_scratch(q, lists * 8)) {
            auto* lv = static_cast<float*>(scratch);
            auto* li = reinterpret_cast<int*>(lv + lists);
            split_stage1(q, logits, n_tokens, n_vocab, history, history_len, pp, k, nb, lv, li);
            const auto e = xe::has_fp64(q) ? split_stage2<double>(q, n_tokens, n_vocab, pp, k, nb, lv, li, out)
                                           : split_stage2<float>(q, n_tokens, n_vocab, pp, k, nb, lv, li, out);
            if (!stream) core::Runtime::get().wait(e, "sample_tokens");
            return;
        }
    }
    const auto e = launch(q, logits, n_tokens, n_vocab, history, history_len, pp, greedy, words, out);
    if (!stream) core::Runtime::get().wait(e, "sample_tokens");
}


size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0) return 0;
    const size_t nb = ((size_t) nv + SPLIT_BL - 1) / SPLIT_BL;
    if (nb * KMAX > SPLIT_MAX) return 0;
    return nb * KMAX * 8 + (size_t) nv * sizeof(int32_t);
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    auto& q = core::Runtime::get().stream(stream);
    const bool gumbel = gumbel_env();
    q.single_task([=] {
        *params = *mapped_params;
        params->gumbel = gumbel;
    });
    q.parallel_for(sycl::range<1>((size_t) cap), [=](sycl::id<1> i) {
        ring[i] = mapped_hist[i];
    });
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step_rec,
                          void* scratch, int32_t* out_id, float* out_prob, void* stream) {
    auto& q = core::Runtime::get().stream(stream);
    const int nb = (nv + SPLIT_BL - 1) / SPLIT_BL;
    const int k = sycl::min(nv, KMAX);
    auto* lv = static_cast<float*>(scratch);
    auto* li = reinterpret_cast<int32_t*>(lv + (size_t) nb * KMAX);
    auto* counts = li + (size_t) nb * KMAX;
    q.memset(counts, 0, (size_t) nv * sizeof(int32_t));
    q.parallel_for(sycl::range<1>((size_t) cap), [=](sycl::id<1> i) {
        const int h = core::coupled_hist_len(params->penalty_last_n, cap);
        if ((int) i[0] >= h) return;
        const int id = ring[core::coupled_hist_start(cap, j, h) + i[0]];
        if (id < 0 || id >= id_vocab) return;
        const int sub = id_to_sub ? id_to_sub[id] : id;
        if (sub >= 0 && sub < nv)
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(counts[sub]).fetch_add(1);
    });
    q.parallel_for(sycl::range<1>((size_t) nv), [=](sycl::id<1> i) {
        logits[i] = apply_penalties(logits[i], counts[i], *params);
    });
    SamplerParams unpenalized;
    split_stage1(q, logits, 1, nv, nullptr, 0, unpenalized, k, nb, lv, li);
    const auto launch = xe::has_fp64(q) ? split_stage2<double> : split_stage2<float>;
    launch(q, 1, nv, unpenalized, k, nb, lv, li, out_id, params, step_rec, sub_to_id, ring + cap + j, out_prob);
}

}  // namespace strata::kernels
