// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/kernels/sampler_host.cpp - see include/strata/kernels/sampler_host.hpp.  The kernel's chain (src/kernels/xe/
// sampler.cpp, sample_launch and pick_from_list) step for step, serially: the window is the history row's tail of
// penalty_last_n ids, an id outside the vocabulary never counts; the top_k list is k argmaxes over the not-yet-taken
// ids (a slot no finite value fills holds id 0 at -inf, as the kernel's); greedy is the argmax of the same scores.
// The Philox draws, the penalties and the list's pick are transcribed from Strata's sampler (src/kernels/cuda/
// sampler.cu, MIT).
#include "strata/kernels/sampler_host.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace strata::kernels {
namespace {

void philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
    const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);   // 24 bits: uniform in [0,1), never 1.0
}

// The Gumbel noise, the kernel's `gumbel_noise`: YASAKANI's smp_hash.cuh `kh_gumbel`, a Gumbel draw hashed from
// (seed, position, token id).  The kernel makes it only for the top kGumbelK of a row's survivors.
uint64_t kh_mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double gumbel_noise(uint64_t seed, uint64_t counter, uint32_t token) {
    const uint64_t h = kh_mix(kh_mix(kh_mix(seed) ^ counter) ^ ((uint64_t) token * 0xD6E8FEB86659FD93ull));
    const double u = ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0);   // (0, 1), never 0 or 1
    return -std::log(-std::log(u));
}

float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

int pick_from_list(const int* sel_ids, const float* sel_logit, int k, const SamplerParams& pp, int t) {
    using D = double;
    const float inv_t = pp.temperature > 0.0f ? 1.0f / pp.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel_logit[i]);
    if (pp.top_p < 1.0f) {
        D sum = 0;
        for (int i = 0; i < k; ++i) sum += std::exp((D) sel_logit[i] - (D) mx);
        D cum = 0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((D) sel_logit[i] - (D) mx) / sum;
            if (cum >= (D) pp.top_p) { cut = i + 1; break; }
        }
        if (cut < pp.min_keep) cut = pp.min_keep < k ? pp.min_keep : k;
        n_keep = cut;
    }
    if (pp.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(pp.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    n_keep = std::max(n_keep, 1);   // the first entry is never cut (its own logit is the threshold's base)
    // a Gumbel pick draws over the top kGumbelK of the survivors only, as the kernel's
    const int nb = pp.gumbel && n_keep > kGumbelK ? kGumbelK : n_keep;
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < nb; ++i) smx = std::fmax(smx, scaled(i));
    D sum = 0;
    if (!pp.gumbel)
        for (int i = 0; i < nb; ++i) sum += std::exp((D) scaled(i) - (D) smx);
    int pick = sel_ids[nb - 1];
    if (pp.gumbel) {
        D best = -std::numeric_limits<D>::infinity();
        for (int i = 0; i < nb; ++i) {
            const D g = (D) scaled(i) - (D) smx + gumbel_noise(pp.seed, pp.counter + (uint64_t) t,
                                                               (uint32_t) sel_ids[i]);
            if (g > best) { best = g; pick = sel_ids[i]; }
        }
    } else {
        const float u = philox_uniform(pp.seed, pp.counter + (uint64_t) t);
        D cum = 0;
        for (int i = 0; i < nb; ++i) {
            cum += std::exp((D) scaled(i) - (D) smx) / sum;
            if ((D) u < cum) { pick = sel_ids[i]; break; }
        }
    }
    return pick;
}

}  // namespace

void sample_tokens_host(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                        const SamplerParams& p, int* out) {
    constexpr int KMAX = 64;
    constexpr float kNone = -std::numeric_limits<float>::infinity();
    const bool greedy = p.greedy || p.temperature <= 0.0f;
    const size_t words = (size_t) (n_vocab + 31) / 32;
    std::vector<float> score((size_t) n_vocab);
    for (int t = 0; t < n_tokens; ++t) {
        const float* l = logits + (size_t) t * n_vocab;
        // the window's counts: the tail of the row's history
        std::unordered_map<int, int> counts;
        if (history != nullptr && history_len > 0) {
            const int hlen = std::max(0, std::min(p.penalty_last_n, history_len));
            const int* h = history + (size_t) t * history_len + (history_len - hlen);
            for (int i = 0; i < hlen; ++i)
                if (h[i] >= 0 && h[i] < n_vocab) ++counts[h[i]];
        }
        const uint32_t* mask = p.allowed != nullptr ? p.allowed + (size_t) t * words : nullptr;
        for (int v = 0; v < n_vocab; ++v) {
            float s = (mask != nullptr && !((mask[v >> 5] >> (v & 31)) & 1u))
                          ? kNone : (p.logit_bias != nullptr ? l[v] + p.logit_bias[v] : l[v]);
            if (!counts.empty())
                if (const auto it = counts.find(v); it != counts.end()) s = apply_penalties(s, it->second, p);
            score[(size_t) v] = s;
        }
        // the argmax order: a value beats every lower one, the lowest id wins a tie, NaN and -inf never win
        auto better = [&](int a, int b) {
            const float sa = score[(size_t) a], sb = score[(size_t) b];
            return sa > sb || (sa == sb && a < b);
        };
        if (greedy) {
            int best = -1;
            for (int v = 0; v < n_vocab; ++v)
                if (score[(size_t) v] > kNone && (best < 0 || better(v, best))) best = v;
            out[t] = best >= 0 ? best : 0;
            continue;
        }
        const int k = std::min(n_vocab, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
        std::vector<int> cand;
        cand.reserve((size_t) n_vocab);
        for (int v = 0; v < n_vocab; ++v)
            if (score[(size_t) v] > kNone) cand.push_back(v);
        const size_t take = std::min(cand.size(), (size_t) k);
        std::partial_sort(cand.begin(), cand.begin() + (std::ptrdiff_t) take, cand.end(), better);
        int sel_ids[KMAX] = {};
        float sel_logit[KMAX] = {};
        for (int i = 0; i < k; ++i) {
            const bool real = (size_t) i < take;
            sel_ids[i] = real ? cand[(size_t) i] : 0;
            sel_logit[i] = real ? score[(size_t) cand[(size_t) i]] : kNone;
        }
        out[t] = pick_from_list(sel_ids, sel_logit, k, p, t);
    }
}

}  // namespace strata::kernels
