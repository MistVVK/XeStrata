// Penalty-history rows of a BATCH window, per slot (strata::kernels::slot_penalty_rows).  Header-only, CPU, no model.
//
// Row t of a batch window feeds tokens[t] into slot rows[t]; its history is the slot's reply tail followed by that slot's tokens in this window up to
// and including row t - what penalty_rows gives a solo window of that slot, whatever the other slots do.
#include "strata/kernels/sampler.hpp"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

using strata::kernels::penalty_rows;
using strata::kernels::slot_penalty_rows;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    constexpr int kSlots = 6, kH = 16;
    for (int trial = 0; trial < 400; ++trial) {
        // per slot: a reply tail of 0..40 tokens
        std::vector<std::vector<int32_t>> tail(kSlots);
        std::vector<const int32_t*> tails(kSlots, nullptr);
        std::vector<int64_t> n_tails(kSlots, 0);
        for (int b = 0; b < kSlots; ++b) {
            const int n = (int) (rng() % 41);
            for (int i = 0; i < n; ++i) tail[b].push_back((int32_t) (rng() % 500));
            tails[b] = tail[b].empty() ? nullptr : tail[b].data();
            n_tails[b] = (int64_t) tail[b].size();
        }
        // a window: 1..8 rows; each slot at most once, or as a contiguous group of 1..3 rows (MTP drafts)
        std::vector<int> rows; std::vector<int32_t> toks;
        std::vector<int> order = {0, 1, 2, 3, 4, 5};
        std::shuffle(order.begin(), order.end(), rng);
        for (int b : order) {
            if (rows.size() >= 8) break;
            if (rng() % 3 == 0) continue;
            const int g = 1 + (int) (rng() % 3);
            for (int j = 0; j < g && rows.size() < 8; ++j) { rows.push_back(b); toks.push_back((int32_t) (rng() % 500)); }
        }
        if (rows.empty()) { rows.push_back(2); toks.push_back(9); }
        const int S = (int) rows.size();
        std::vector<int32_t> out((size_t) S * kH, 12345);
        slot_penalty_rows(rows.data(), toks.data(), S, tails.data(), n_tails.data(), kH, out.data());
        // the definition: per slot, penalty_rows over (its tail, its rows of this window)
        for (int b = 0; b < kSlots; ++b) {
            std::vector<int> idx;
            std::vector<int32_t> win;
            for (int t = 0; t < S; ++t) if (rows[t] == b) { idx.push_back(t); win.push_back(toks[t]); }
            if (idx.empty()) continue;
            std::vector<int32_t> ref(idx.size() * kH);
            penalty_rows(tail[b].data(), (int64_t) tail[b].size(), win.data(), (int) idx.size(), kH, ref.data());
            for (size_t k = 0; k < idx.size(); ++k)
                for (int j = 0; j < kH; ++j)
                    check(out[(size_t) idx[k] * kH + j] == ref[k * kH + j], "slot row equals penalty_rows of the slot alone");
        }
    }
    // ---- fixed cases: an empty tail, the fed-back head counted, other slots' tokens never in a row
    {
        const int rows[3] = {0, 1, 1};
        const int32_t toks[3] = {10, 20, 21};
        const int32_t t0[2] = {1, 2};
        const int32_t t1[3] = {7, 8, 9};
        const int32_t* tails[2] = {t0, t1};
        const int64_t n_tails[2] = {2, 3};
        std::vector<int32_t> out(3 * 4, 0);
        slot_penalty_rows(rows, toks, 3, tails, n_tails, 4, out.data());
        const int32_t e[12] = {-1, 1, 2, 10,    7, 8, 9, 20,    8, 9, 20, 21};
        for (int i = 0; i < 12; ++i) check(out[i] == e[i], "fixed case");
    }
    {   // a slot with no tail (no penalty history): its row holds the fed token only
        const int rows[1] = {0};
        const int32_t toks[1] = {42};
        const int32_t* tails[1] = {nullptr};
        const int64_t n_tails[1] = {0};
        int32_t out[4] = {0, 0, 0, 0};
        slot_penalty_rows(rows, toks, 1, tails, n_tails, 4, out);
        check(out[0] == -1 && out[1] == -1 && out[2] == -1 && out[3] == 42, "empty tail");
    }
    std::printf("slot_history_test: %d checks passed\n", checks);
    return 0;
}
