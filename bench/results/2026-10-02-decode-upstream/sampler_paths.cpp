// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Prints the sampled token of many (logits, parameters) cases at the model's vocabulary, for comparing the sampler's
// paths in two processes (STRATA_OLD_SAMPLER=1 in one).
#include "strata/kernels/sampler.hpp"
#include "strata/core/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
using namespace strata;
int main() {
    auto& q = core::Runtime::get().compute();
    const int V = 248320, NT = 4, H = 64;
    std::mt19937 rng(11);
    float* d_l = sycl::malloc_device<float>((size_t) V * NT, q);
    int* d_o = sycl::malloc_device<int>(NT, q);
    int* d_h = sycl::malloc_device<int>((size_t) NT * H, q);
    std::vector<float> l((size_t) V * NT);
    std::vector<int> hist((size_t) NT * H);
    for (int c = 0; c < 60; ++c) {
        std::normal_distribution<float> nd(0.0f, c % 3 == 0 ? 1.0f : 4.0f);
        for (auto& x : l) x = nd(rng);
        if (c % 4 == 1) for (auto& x : l) x = std::round(x * 2.0f) / 2.0f;     // many ties
        if (c % 5 == 2) for (int i = 0; i < 2000; ++i) l[rng() % l.size()] = std::numeric_limits<float>::quiet_NaN();
        if (c % 5 == 3) for (size_t i = 0; i < l.size(); ++i) if (rng() % 3) l[i] = -INFINITY;
        if (c % 7 == 4) for (int i = 0; i < 100; ++i) l[rng() % l.size()] = 50.0f;   // a tie at the top
        if (c == 59) for (size_t i = 0; i < l.size(); ++i) if (i % V > 30) l[i] = -INFINITY;   // fewer than k left
        for (auto& v : hist) v = (int) (rng() % V);
        q.memcpy(d_l, l.data(), l.size() * 4).wait();
        q.memcpy(d_h, hist.data(), hist.size() * 4).wait();
        kernels::SamplerParams p;
        p.greedy = false;
        p.top_k = 1 + (c * 7) % 64;
        p.top_p = c % 2 ? 0.95f : 1.0f;
        p.min_p = c % 3 == 1 ? 0.05f : 0.0f;
        p.temperature = 0.3f + 0.1f * (float) (c % 10);
        p.seed = 1234 + c;
        p.counter = (uint64_t) c * 10;
        const bool pen = c % 2 == 0;
        p.penalty_last_n = pen ? 64 : 0;
        p.penalty_repeat = 1.3f;
        p.penalty_freq = 0.2f;
        kernels::sample_tokens(d_l, NT, V, pen ? d_h : nullptr, pen ? H : 0, p, d_o, nullptr);
        int o[NT];
        q.memcpy(o, d_o, sizeof o).wait();
        std::printf("%d %d %d %d %d\n", c, o[0], o[1], o[2], o[3]);
    }
}
