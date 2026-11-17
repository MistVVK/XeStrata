// metal/tests/sample_chain_parity.cpp - the sampler chain on the GPU (sample.metal samp_part + samp_pick) against a CPU reference of the same chain
// (src/kernels/cuda/sampler.cu `sampler_kernel`: bias -> penalties -> top_k -> top_p -> min_p -> temperature -> Philox draw at (seed, counter + row)).
//   sample_chain_parity
// 1. greedy flag, no penalties, no bias == the argmax kernel's pick (ties to the lowest id) on 248,320 logits with planted ties
// 2. sampled chain, 400 random windows of 1..8 rows (random top_k / top_p / min_p / temperature / penalties / bias / seeds): the pick equals the CPU
//    pick except where the draw lies within 1e-4 of a cumulative boundary (float against the reference's double)
// 3. determinism: the same (seed, counter) twice gives the same tokens
// 4. chi-square: 60,000 draws from one row (counter 0..59,999) against the expected probabilities of the kept candidates
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {
constexpr uint32_t V = 248320, PARTS = 64, MAXT = 8;
struct SampArgs {
    uint32_t n_vocab, parts, k, hist_len, pen_n, min_keep, flags;
    float top_p, min_p, inv_t, rep, freq, pres;
    uint32_t seed_lo, seed_hi, ctr_lo, ctr_hi;
};
struct P {
    int top_k = 20; float top_p = 0.95f, min_p = 0.0f, temperature = 1.0f; int min_keep = 1; int pen_n = 0; float rep = 1.0f, freq = 0.0f, pres = 0.0f;
    uint64_t seed = 0, counter = 0; bool greedy = false, bias = false;
};
float philox_u(uint64_t seed, uint64_t ctr) {
    uint32_t c0 = (uint32_t) ctr, c1 = (uint32_t) (ctr >> 32), c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32), hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
// the CPU chain over one row; returns the pick and fills the kept (id, probability) list when `kept` is given
int cpu_pick(const float* l, const float* bias, const int* hist, int hist_len, const P& p, uint64_t row, double* boundary_dist,
             std::vector<std::pair<int, double>>* kept) {
    const int k = (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64;
    std::vector<std::pair<float, int>> c;
    c.reserve(V);
    const int pn = std::min(p.pen_n, hist_len);
    for (uint32_t v = 0; v < V; ++v) {
        float s = l[v];
        if (p.bias) s += bias[v];
        if (pn > 0) {
            int cnt = 0;
            for (int j = hist_len - pn; j < hist_len; ++j) cnt += hist[j] == (int) v;
            if (cnt > 0) { if (s <= 0.0f) s *= p.rep; else s /= p.rep; s -= (float) cnt * p.freq + p.pres; }
        }
        if (std::isnan(s) || s == -INFINITY) continue;
        c.push_back({s, (int) v});
    }
    std::partial_sort(c.begin(), c.begin() + std::min<size_t>(k, c.size()), c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    c.resize(std::min<size_t>(k, c.size()));
    if (c.empty()) return 0;
    if (p.greedy) return c[0].second;
    int n_keep = (int) c.size();
    const float mx = c[0].first;
    if (p.top_p < 1.0f) {
        double sum = 0; for (auto& x : c) sum += std::exp((double) x.first - mx);
        double cum = 0; int cut = n_keep;
        for (int i = 0; i < n_keep; ++i) { cum += std::exp((double) c[i].first - mx) / sum; if (cum >= p.top_p) { cut = i + 1; break; } }
        if (cut < p.min_keep) cut = std::min<int>(p.min_keep, n_keep);
        n_keep = cut;
    }
    if (p.min_p > 0.0f) { const float th = c[0].first + std::log(p.min_p); for (int i = 0; i < n_keep; ++i) if (c[i].first < th) { n_keep = i; break; } }
    const float inv_t = 1.0f / p.temperature;
    float smx = c[0].first * inv_t; for (int i = 1; i < n_keep; ++i) smx = std::max(smx, c[i].first * inv_t);
    double sum = 0; for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (c[i].first * inv_t) - smx);
    const float u = philox_u(p.seed, p.counter + row);
    int pick = c[n_keep - 1].second; double cum = 0; double md = 1e9;
    bool done = false;
    if (kept) kept->clear();
    for (int i = 0; i < n_keep; ++i) {
        const double pr = std::exp((double) (c[i].first * inv_t) - smx) / sum;
        cum += pr;
        md = std::min(md, std::fabs((double) u - cum));
        if (kept) kept->push_back({c[i].second, pr});
        if (!done && (double) u < cum) { pick = c[i].second; done = true; }
    }
    if (boundary_dist) *boundary_dist = md;
    return pick;
}
template <class T> T pick_of(std::initializer_list<T> l, uint64_t x) { return *(l.begin() + x % l.size()); }
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline pa = cx.pipeline(lib, "argmax_part"), pf = cx.pipeline(lib, "argmax_final"), sp = cx.pipeline(lib, "samp_part"), spk = cx.pipeline(lib, "samp_pick");
    constexpr uint32_t HL = 64;
    mtl::Buffer lg = cx.alloc(4 * (size_t) V * MAXT), bias = cx.alloc(4 * (size_t) V), hist = cx.alloc(4 * (size_t) MAXT * HL);
    mtl::Buffer cv = cx.alloc(4 * (size_t) MAXT * PARTS * 64), ci = cx.alloc(4 * (size_t) MAXT * PARTS * 64), out = cx.alloc(4 * MAXT);
    mtl::Buffer pv = cx.alloc(4 * 120 * MAXT), pi = cx.alloc(4 * 120 * MAXT), aout = cx.alloc(4 * MAXT);
    ref::Rng r(11);
    int fails = 0;
    auto run = [&](int nt, const P& p, int hl) {
        SampArgs a{};
        a.n_vocab = V; a.parts = PARTS; a.k = p.greedy ? 1u : (uint32_t) ((p.top_k > 0 && p.top_k < 64) ? p.top_k : 64); a.hist_len = HL; a.pen_n = (uint32_t) std::min<int>(p.pen_n, hl);
        a.min_keep = (uint32_t) p.min_keep; a.flags = (p.bias ? 1u : 0u) | (p.greedy ? 2u : 0u) | (a.pen_n > 0 ? 4u : 0u);
        a.top_p = p.top_p; a.min_p = p.min_p; a.inv_t = p.temperature > 0 ? 1.0f / p.temperature : 0.0f; a.rep = p.rep; a.freq = p.freq; a.pres = p.pres;
        a.seed_lo = (uint32_t) p.seed; a.seed_hi = (uint32_t) (p.seed >> 32); a.ctr_lo = (uint32_t) p.counter; a.ctr_hi = (uint32_t) (p.counter >> 32);
        mtl::Dispatch d1, d2;
        d1.p = &sp; d1.args = {mtl::Arg::B(lg), mtl::Arg::B(bias), mtl::Arg::B(hist), mtl::Arg::B(cv), mtl::Arg::B(ci), mtl::Arg::V(a)};
        d1.grid[0] = PARTS; d1.grid[1] = nt; d1.tg[0] = 256;
        d2.p = &spk; d2.args = {mtl::Arg::B(cv), mtl::Arg::B(ci), mtl::Arg::B(out), mtl::Arg::V(a)};
        d2.grid[0] = nt; d2.tg[0] = 256;
        cx.run({d1, d2});
    };
    // 1. greedy == argmax
    for (int trial = 0; trial < 12; ++trial) {
        const int nt = 1 + trial % MAXT;
        for (size_t i = 0; i < (size_t) V * nt; ++i) lg.as<float>()[i] = std::round(r.sym() * 1000.0f) / 100.0f;
        for (int t = 0; t < nt; ++t) { lg.as<float>()[(size_t) t * V + r.next() % V] = 20.0f; lg.as<float>()[(size_t) t * V + r.next() % V] = 20.0f; }
        P p; p.greedy = true; p.top_k = 1;
        run(nt, p, 0);
        for (int t = 0; t < nt; ++t) {
            const float* l = lg.as<float>() + (size_t) t * V;
            int best = 0; for (uint32_t i = 1; i < V; ++i) if (l[i] > l[best]) best = (int) i;
            fails += out.as<int>()[t] != best;
        }
    }
    std::printf("greedy flag == argmax over 12 windows with planted ties: %d wrong\n", fails);
    // 2. the sampled chain against the CPU reference
    int mism = 0, near = 0, total = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int nt = 1 + trial % MAXT;
        const bool ties = trial % 5 == 0;
        for (size_t i = 0; i < (size_t) V * nt; ++i) lg.as<float>()[i] = ties ? std::round(r.sym() * 400.0f) / 100.0f : r.sym() * 12.0f;
        P p;
        p.top_k = pick_of<int>({1, 5, 20, 40, 64, 0, 100}, r.next());
        p.top_p = pick_of<float>({1.0f, 0.95f, 0.8f, 0.5f}, r.next());
        if (ties) p.top_p = 1.0f;   // (heavy ties give exactly equal probabilities: a top_p cut at cum == top_p flips between float and the reference's double; the tie ORDER is what this trial checks)
        p.min_p = pick_of<float>({0.0f, 0.0f, 0.05f}, r.next());
        p.temperature = pick_of<float>({0.3f, 0.6f, 1.0f, 1.5f}, r.next());
        p.seed = 1000 + r.next() % 5000; p.counter = r.next() % 100000;
        p.bias = trial % 4 == 1;
        if (p.bias) { for (uint32_t i = 0; i < V; ++i) bias.as<float>()[i] = 0.0f; for (int j = 0; j < 40; ++j) bias.as<float>()[r.next() % V] = (j % 3 == 0) ? -INFINITY : r.sym() * 4.0f; }
        int hl = 0;
        if (trial % 3 == 2) {
            p.pen_n = 16 + r.next() % 48; p.rep = 1.05f + 0.2f * (r.next() % 3); p.freq = (r.next() % 2) * 0.1f; p.pres = (r.next() % 2) * 0.2f; hl = HL;
            for (int t = 0; t < nt; ++t) for (uint32_t j = 0; j < HL; ++j) hist.as<int>()[t * HL + j] = (int) (r.next() % 30 == 0 ? -1 : (int) (r.next() % V));
            // make some history tokens the top candidates
            for (int t = 0; t < nt; ++t) { const float* l = lg.as<float>() + (size_t) t * V; int b = 0; for (uint32_t i = 1; i < V; ++i) if (l[i] > l[b]) b = (int) i; hist.as<int>()[t * HL + HL - 1] = b; }
        }
        run(nt, p, hl);
        for (int t = 0; t < nt; ++t) {
            double bd = 1e9;
            const int c = cpu_pick(lg.as<float>() + (size_t) t * V, p.bias ? bias.as<float>() : nullptr, hist.as<int>() + t * HL, HL, p, (uint64_t) t, &bd, nullptr);
            ++total;
            if (out.as<int>()[t] != c) { if (bd < 1e-4) ++near; else { ++mism; if (mism <= 12) std::printf("  MISMATCH trial %d row %d: gpu %d cpu %d (k %d top_p %.2f min_p %.2f T %.1f bias %d pen %d ties %d boundary %.2g)\n", trial, t, out.as<int>()[t], c, p.top_k, p.top_p, p.min_p, p.temperature, (int) p.bias, p.pen_n, (int) ties, bd); } }
        }
    }
    std::printf("sampled chain vs CPU reference: %d rows, %d mismatches (+ %d within 1e-4 of a cumulative boundary)\n", total, mism, near);
    fails += mism;
    // 3. determinism
    {
        for (size_t i = 0; i < (size_t) V * 4; ++i) lg.as<float>()[i] = r.sym() * 12.0f;
        P p; p.seed = 77; p.counter = 12345; p.top_k = 40; p.top_p = 0.95f; p.temperature = 0.6f;
        run(4, p, 0); int a1[4]; std::memcpy(a1, out.ptr, 16);
        run(4, p, 0); int a2[4]; std::memcpy(a2, out.ptr, 16);
        p.seed = 78; run(4, p, 0); int a3[4]; std::memcpy(a3, out.ptr, 16);
        const bool same = std::memcmp(a1, a2, 16) == 0;
        std::printf("same seed twice: %s; seed 77 vs 78: tokens %d %d %d %d vs %d %d %d %d\n", same ? "identical" : "DIFFERENT", a1[0], a1[1], a1[2], a1[3], a3[0], a3[1], a3[2], a3[3]);
        fails += !same;
    }
    // 4. chi-square on 60,000 draws of one row against the kept candidates' probabilities
    {
        P p; p.seed = 4242; p.top_k = 40; p.top_p = 0.95f; p.temperature = 0.6f;
        for (size_t i = 0; i < V; ++i) lg.as<float>()[i] = r.sym() * 6.0f;
        std::vector<std::pair<int, double>> kept;
        cpu_pick(lg.as<float>(), nullptr, nullptr, 0, p, 0, nullptr, &kept);
        std::vector<double> cnt(kept.size() + 1, 0.0);
        const int N = 60000;
        for (int i = 0; i < N; i += MAXT) {
            const int nt = std::min<int>(MAXT, N - i);
            for (int t = 1; t < nt; ++t) std::memcpy(lg.as<float>() + (size_t) t * V, lg.as<float>(), 4 * (size_t) V);
            p.counter = (uint64_t) i;
            run(nt, p, 0);
            for (int t = 0; t < nt; ++t) {
                size_t idx = kept.size();
                for (size_t j = 0; j < kept.size(); ++j) if (kept[j].first == out.as<int>()[t]) idx = j;
                cnt[idx] += 1;
            }
        }
        double chi = 0; int dof = 0;
        for (size_t j = 0; j < kept.size(); ++j) { const double e = kept[j].second * N; if (e >= 5) { chi += (cnt[j] - e) * (cnt[j] - e) / e; ++dof; } }
        const double outside = cnt[kept.size()];
        const double crit = dof + 3.0 * std::sqrt(2.0 * dof);   // about the 99.9th percentile
        std::printf("chi-square over %zu kept candidates (%d bins): %.1f against a limit of %.1f (dof %d); draws outside the kept list: %.0f\n", kept.size(), dof + 1, chi, crit, dof - 1, outside);
        fails += (chi > crit || outside > 0);
    }
    std::printf("sampler chain parity: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
