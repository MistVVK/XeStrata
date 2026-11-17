// The batched drafts' row plan (src/batch_plan.hpp) on the CPU: it stays within the rows and each lane's offer, it finds
// the most tokens per ms a brute force finds over every split (random offers of 1..6 lanes), and for one lane it picks
// the chain length Session::generate's chain_len picks alone.
//
//   batch_plan_test   (make bin/batch_plan_test; no card, no model)
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "../src/batch_plan.hpp"

namespace {
constexpr int ROWS = 16;   // MAXT

double rate(const std::vector<fnx::RowOffer>& L, const std::vector<int>& d, double base, const fnx::RowCost& c) {
    double tok = (double) L.size(), ms = base;
    int rows = (int) L.size();
    for (size_t i = 0; i < L.size(); ++i)
        for (int j = 0; j < d[i]; ++j) {
            tok += L[i].reach[(size_t) j];
            ms += c.row(++rows) + (L[i].chain && j >= 1 ? c.chain_ms : 0.0);
        }
    return tok / ms;
}

double best_rate(const std::vector<fnx::RowOffer>& L, double base, const fnx::RowCost& c) {
    std::vector<int> d(L.size(), 0);
    double best = 0.0;
    std::function<void(size_t, int)> go = [&](size_t i, int left) {
        if (i == L.size()) {
            best = std::max(best, rate(L, d, base, c));
            return;
        }
        for (int k = 0; k <= std::min(L[i].avail, left); ++k) {
            d[i] = k;
            go(i + 1, left - k);
        }
        d[i] = 0;
    };
    go(0, ROWS - (int) L.size());
    return best;
}
}  // namespace

int main() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const fnx::RowCost c;
    int fails = 0, cases = 0;
    for (int it = 0; it < 4000; ++it) {
        const int n = 1 + (int) (rng() % 6);
        std::vector<fnx::RowOffer> L((size_t) n);
        for (auto& o : L) {
            o.chain = rng() % 2;
            o.avail = (int) (rng() % (o.chain ? 7 : 16));
            const double p = u(rng);   // per-position acceptance, non-increasing reach
            double r = 1.0;
            for (int j = 0; j < o.avail; ++j) o.reach.push_back(r *= p * (0.8 + 0.2 * u(rng)));
        }
        const double base = c.round_ms + c.rows_ms(n) + c.call_ms * n;
        const std::vector<int> d = fnx::plan_rows(L, ROWS, base, c);
        int rows = n;
        for (int i = 0; i < n; ++i) {
            rows += d[(size_t) i];
            if (d[(size_t) i] < 0 || d[(size_t) i] > L[(size_t) i].avail) {
                std::printf("case %d: lane %d gets %d drafts of %d offered\n", it, i, d[(size_t) i], L[(size_t) i].avail);
                ++fails;
            }
        }
        if (rows > ROWS) {
            std::printf("case %d: %d rows\n", it, rows);
            ++fails;
        }
        const double got = rate(L, d, base, c), best = best_rate(L, base, c);
        if (got < best * (1.0 - 1e-9)) {
            std::printf("case %d (%d lanes): %.6f tokens/ms, the best split %.6f\n", it, n, got, best);
            ++fails;
        }
        ++cases;
    }
    // one lane, a chain: chain_len's choice (k drafts, at least 1) where the first draft pays at all
    for (int it = 0; it < 2000; ++it) {
        fnx::RowOffer o;
        o.chain = true;
        o.avail = 1 + (int) (rng() % 6);
        double r = 1.0;
        const double p = 0.2 + 0.8 * u(rng);
        for (int j = 0; j < o.avail; ++j) o.reach.push_back(r *= p * (0.8 + 0.2 * u(rng)));
        int best = 1;
        double best_rate = -1.0, expect = 1.0;
        for (int k = 1; k <= o.avail; ++k) {
            expect += o.reach[(size_t) k - 1];
            const double rt = expect / (c.round_ms + c.rows_ms(k + 1) + std::max(0, k - 1) * c.chain_ms);
            if (rt > best_rate) best_rate = rt, best = k;
        }
        const int d = fnx::plan_rows({o}, ROWS, c.round_ms, c)[0];
        if (d != best) {
            std::printf("one lane: plan %d drafts, chain_len %d (avail %d, reach0 %.3f)\n", d, best, o.avail, o.reach[0]);
            ++fails;
        }
        ++cases;
    }
    // choose_rows: within the offers and rows; the best of plan_rows' plan, the uniform plans (each charged the capture
    // unless held) and the held layouts that fit; an explorer's draft never adds a capture
    for (int it = 0; it < 4000; ++it) {
        const int n = 1 + (int) (rng() % 6);
        std::vector<fnx::RowOffer> L((size_t) n);
        for (auto& o : L) {
            o.chain = rng() % 2;
            o.avail = (int) (rng() % (o.chain ? 7 : 16));
            o.explore = rng() % 4 == 0;
            const double p = u(rng);
            double r = 1.0;
            for (int j = 0; j < o.avail; ++j) o.reach.push_back(r *= p * (0.8 + 0.2 * u(rng)));
        }
        const double base = c.round_ms + c.rows_ms(n) + c.call_ms * n, cap = 10.0 * u(rng);
        std::vector<std::vector<int>> held;
        for (int h = (int) (rng() % 4); h > 0; --h) {
            std::vector<int> counts((size_t) n);
            for (int& k : counts) k = 1 + (int) (rng() % 5);
            held.push_back(counts);
        }
        auto fits = [&](const std::vector<int>& d) {
            int rows = n;
            for (int i = 0; i < n; ++i) {
                if (d[(size_t) i] < 0 || d[(size_t) i] > L[(size_t) i].avail) return false;
                rows += d[(size_t) i];
            }
            return rows <= ROWS;
        };
        auto held_d = [&](const std::vector<int>& d) {
            for (const auto& h : held) {
                bool same = true;
                for (int i = 0; i < n; ++i) same = same && h[(size_t) i] == d[(size_t) i] + 1;
                if (same) return true;
            }
            return false;
        };
        auto bare = [&](std::vector<fnx::RowOffer> o) {
            for (auto& x : o) x.explore = false;
            return o;
        };
        const std::vector<int> d = fnx::choose_rows(L, ROWS, base, c, held, cap);
        const std::vector<int> d0 = fnx::choose_rows(bare(L), ROWS, base, c, held, cap);
        if (!fits(d) || !fits(d0)) {
            std::printf("choose case %d: a plan past the offers or the rows\n", it);
            ++fails;
            continue;
        }
        // d0's rate (charged as chosen) is the best of every candidate's: plan_rows' plan, the uniform plans, the held
        const auto Lb = bare(L);
        const std::vector<int> p = fnx::plan_rows(Lb, ROWS, base, c);
        double best = fnx::plan_rate(Lb, p, base, c, held_d(p) ? 0.0 : cap);
        for (int k = 0; k <= 15; ++k) {
            std::vector<int> u;
            for (const auto& o : Lb) u.push_back(std::min(k, o.avail));
            if (fits(u)) best = std::max(best, fnx::plan_rate(Lb, u, base, c, held_d(u) ? 0.0 : cap));
        }
        for (const auto& h : held) {
            std::vector<int> hd;
            for (int k : h) hd.push_back(k - 1);
            if (fits(hd)) best = std::max(best, fnx::plan_rate(Lb, hd, base, c, 0.0));
        }
        const double got = fnx::plan_rate(Lb, d0, base, c, held_d(d0) ? 0.0 : cap);
        if (got < best * (1.0 - 1e-9)) {
            std::printf("choose case %d: %.6f tokens/ms, the best candidate %.6f\n", it, got, best);
            ++fails;
        }
        if (held_d(d0) && !held_d(d)) {
            std::printf("choose case %d: an explorer's draft made a held layout capture\n", it);
            ++fails;
        }
        cases += 1;
    }
    bool threw = false;
    try {
        fnx::RowOffer o;
        o.avail = 2;
        fnx::plan_rows({o}, ROWS, c.round_ms, c);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::printf("an offer without its reach was taken\n");
        ++fails;
    }
    std::printf("%d cases, %d failures: %s\n", cases, fails, fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
