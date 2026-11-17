#include "../src/expert_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace fnx;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
template <typename F> static void refuses(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    check(false, "impossible placement must fail");
}

int main() {
    // A cold one-layer pool must keep its hottest slots, even while a larger pool can still shrink.
    const std::vector<std::pair<size_t, size_t>> sizes{{3, 2}, {4, 3}, {3, 2}};
    const std::vector<double> score{10, 11, 12, 13, 0, 1, 2, 3, 14, 15, 16, 17};
    auto p = expert_residency(sizes, score, 4, 2, 0, 6);
    check(p.resident == std::vector<std::vector<char>>{{0,0,0,0}, {0,0,1,1}, {1,1,1,1}}, "rare pool floor and next cold pool");
    check(p.count == 6 && p.moved == 34, "extra counts only experts actually evicted");
    p = expert_residency(sizes, score, 4, 2, 8, 2);
    check(p.count == 4 && p.moved == 24, "byte target followed by extra evictions");
    p = expert_residency(sizes, score, 4, 2, 0, 0);
    check(p.count == 0, "zero request retains all experts");
    p = expert_residency(sizes, score, 4, 2, 44, 0);
    check(p.count == 8 && p.moved == 44, "exact capacity succeeds");
    refuses([&] { expert_residency(sizes, score, 4, 2, 45, 0); });
    refuses([&] { expert_residency(sizes, score, 4, 2, 44, 1); });
    refuses([&] { expert_residency(sizes, score, 4, 5, 0, 0); });

    // Equal total bytes do not make equal pools: gate/up and down strides both matter.
    p = expert_residency({{3,2}, {2,3}}, {0,0,0,0,0,0,0,0}, 4, 2, 0, 4);
    check(p.resident == std::vector<std::vector<char>>{{0,0,1,1}, {0,0,1,1}}, "pool strides and stable ties");

    // Prod-shaped 43/1/4 layer pools; the old global ordering empties the rare pool at this pressure.
    std::vector<std::pair<size_t, size_t>> prod(48, {1843200,1228800});
    prod[2] = {2252800,1740800};
    for (int l : {4,30,46,47}) prod[(size_t) l] = {1843200,1740800};
    std::vector<double> ranks(48 * 512);
    std::mt19937 rng(42);
    for (double& s : ranks) s = (double) rng() / rng.max();
    for (int e = 0; e < 512; ++e) ranks[2*512 + e] = -1 + e / 512.0;
    p = expert_residency(prod, ranks, 512, 320, 3.0 * (1ull << 30), 18253);
    check(std::count(p.resident[2].begin(), p.resident[2].end(), 1) == 320, "320 slots survive automatic VRAM eviction");
    for (int e = 0; e < 512; ++e) check(p.resident[2][(size_t)e] == (e >= 192), "rare pool keeps hottest experts");

    // A large explicit host budget must obey the same pool floor as VRAM-pressure evictions.
    p = expert_residency(prod, ranks, 512, 320, 59.0 * (1ull << 30), 0);
    check(p.moved >= 59.0 * (1ull << 30), "explicit RAM target reached");
    check(std::count(p.resident[2].begin(), p.resident[2].end(), 1) == 320, "explicit RAM target keeps rare pool floor");

    // Where no floor is hit, placement stays byte-for-byte the former global ranking, including ties.
    for (int run = 0; run < 1000; ++run) {
        std::vector<double> ranks(12);
        for (double& s : ranks) s = rng() % 5;
        std::vector<int> order(12);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ranks[a] < ranks[b]; });
        p = expert_residency(sizes, ranks, 4, 2, 0, 2);
        for (int i = 0; i < 12; ++i)
            check(p.resident[i/4][i%4] == (i != order[0] && i != order[1]), "unconstrained placement unchanged");
    }
    // ---- the GSQ-RCO IQ3_S set, read off its GGUF: 7 pools by (gate+up, down) bytes of one expert. Layer 21 and layer 47 are pools of 512 experts
    // each, layers 12/20/30 one of 1,536. The old global ranking (a copy of the code the planner replaced) takes experts coldest per byte
    // first, so a profile that is cold on a small pool empties it below the window floor; the planner must not.
    {
        struct PoolDef { size_t gu, dn; std::vector<int> layers; };
        const std::vector<PoolDef> defs{
            {1049600, 921600, {2, 3, 4, 5, 6, 9, 11, 15, 19, 23, 24, 26, 32, 34, 39}},
            {1254400, 921600, {0, 7, 14, 16, 18, 22, 31, 33, 37, 38, 41, 42, 43, 46}},
            {1408000, 921600, {17, 25, 27, 28, 29, 35, 40, 44, 45}},
            {1049600, 460800, {1, 8, 10, 13, 36}},
            {1254400, 460800, {12, 20, 30}},
            {1408000, 460800, {21}},
            {1740800, 921600, {47}}};
        std::vector<std::pair<size_t, size_t>> iq3(48);
        int n_l = 0;
        for (const auto& d : defs)
            for (int l : d.layers) { iq3[(size_t) l] = {d.gu, d.dn}; ++n_l; }
        check(n_l == 48, "IQ3_S table has 48 layers");
        const int floor = 2 * 16 * 10;   // EXPERT_WINDOW_SLOTS: MAXT rows x TOPK picks, twice
        const double gib1 = (double) (1ull << 30);
        // the engine scores per byte; this profile is cold exactly on the two single-layer pools and the 3-layer pool
        std::vector<double> sc(48 * 512);
        std::mt19937 r2(7);
        for (size_t i = 0; i < sc.size(); ++i) {
            const size_t l = i / 512;
            const bool rare = l == 21 || l == 47 || l == 12 || l == 20 || l == 30;
            sc[i] = (rare ? 0.0 : 0.5) + ((double) (r2() % 1000) / 1000.0) * 0.4;
            sc[i] /= (double) iq3[l].first + iq3[l].second;
        }
        auto old_left = [&](double want, size_t layer) {   // experts of `layer` still resident after the old ranking moved `want` bytes
            std::vector<int> order(48 * 512);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[(size_t) a] < sc[(size_t) b]; });
            double moved = 0;
            int left = 512;
            for (int i : order) {
                if (moved >= want) break;
                moved += (double) iq3[(size_t) (i / 512)].first + iq3[(size_t) (i / 512)].second;
                if ((size_t) (i / 512) == layer) --left;
            }
            return left;
        };
        int old_failures = 0, targets = 0;
        for (double g = 2; g <= 40; g += 2, ++targets) {
            auto q = expert_residency(iq3, sc, 512, floor, g * gib1, 0);   // must place every target up to 40 GiB (the set has 46.8 GiB)
            check(q.moved >= g * gib1, "IQ3_S: RAM target reached");
            for (const auto& d : defs) {
                int kept = 0;
                for (int l : d.layers) kept += (int) std::count(q.resident[(size_t) l].begin(), q.resident[(size_t) l].end(), 1);
                check(kept >= floor, "IQ3_S: every size pool keeps a window of resident experts");
            }
            if (old_left(g * gib1, 21) < floor) ++old_failures;
        }
        check(old_failures > 0, "IQ3_S negative control: the old ranking empties the 1-layer pool below the window at some target");
        // beyond what the floors allow, the request is refused whole (nothing is uploaded)
        refuses([&] { expert_residency(iq3, sc, 512, floor, 46.0 * gib1, 0); });
        // an explicit window larger than a pool: refused at once
        refuses([&] { expert_residency(iq3, sc, 512, 513, 0, 0); });
        std::printf("IQ3_S sweep (2..40 GiB in RAM, %d targets): the old ranking leaves layer 21 under %d slots at %d of them, the planner at none\n",
                    targets, floor, old_failures);
    }
    // injected failure: the card's pool keys are the sliced, 16-byte-padded strides; they partition the layers exactly like the whole-expert bytes
    {
        const std::vector<std::pair<size_t, size_t>> whole{{1049600, 921600}, {1049600, 460800}, {1254400, 921600}, {1254400, 460800}, {1408000, 921600}};
        for (int cards : {2, 3, 4}) {
            std::vector<std::pair<size_t, size_t>> sliced;
            for (auto& w : whole) sliced.push_back({(w.first / (size_t) cards + 15) / 16 * 16, (w.second / (size_t) cards + 15) / 16 * 16});
            for (size_t a = 0; a < whole.size(); ++a)
                for (size_t b = 0; b < whole.size(); ++b)
                    check((whole[a] == whole[b]) == (sliced[a] == sliced[b]), "pools by whole-expert bytes and by a card's sliced strides are the same partition");
        }
    }
    std::puts("expert_residency_test: PASS");
}
