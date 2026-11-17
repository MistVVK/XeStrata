// sFNX batched drafts: how a window's rows are shared out among the decoding lanes (src/batch.cpp). Host arithmetic
// only, so tools/batch_plan_test.cpp checks it without a card.
#pragma once

#ifndef FNX_27B
#error "FNX_27B is unset: build through the Makefile (make FNX_MODEL=fnx or 27b)"
#endif

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fnx {

// a batched window's cost (Q4, lab, 2026-10-25, bdecode with fixed drafts over 1-5 lanes of code1 and edit1): a window
// of one row costs about round_ms; each row more, whichever lane it is, row_lo up to the knee'th row of the window and
// row_hi past it (verify 10.45 ms at 1 row, ~1.5 ms a row to 8 rows, ~1.75-1.9 past: a wide window's rows cost more,
// any lane count); every lane's first draft-layer call call_ms (in the round), each chain draft after a lane's first
// chain_ms (0.59 measured). Each row count more than one among the lanes costs group_ms (the recurrent core runs once
// per count in each of the 36 recurrent layers; plan windows of mixed counts measured ~1.5-2 ms over the rest of this
// model with 2-3 counts). A window whose layout (each lane's rows) has no graph held pays a capture as well (8-11 ms;
// the engine's last_capture_ms)
// The 27B's rows (make FNX_MODEL=27b; s27_run S27_BENCH=1, 2026-10-29): verify 16.1 ms at 1 row, 24.8 at 8 (~1.24 a row),
// 42.8 at 16 (~2.25 a row past 8); its draft-layer calls and groups are Flash-Next's numbers, not measured on it
struct RowCost {
#if FNX_27B
    double round_ms = 16.5, row_lo = 1.24, row_hi = 2.25, call_ms = 0.7, chain_ms = 0.6, group_ms = 0.8;
#else
    double round_ms = 10.8, row_lo = 1.5, row_hi = 1.8, call_ms = 0.7, chain_ms = 0.6, group_ms = 0.8;
#endif
    int knee = 8;
    double row(int t) const { return t <= knee ? row_lo : row_hi; }   // the window's t'th row (t >= 2)
    double rows_ms(int T) const {   // rows 2 .. T
        double ms = 0.0;
        for (int t = 2; t <= T; ++t) ms += row(t);
        return ms;
    }
};

// a lane's drafts on offer: up to avail; reach[j] the chance that drafts 0 .. j all stand (non-increasing in j); chain:
// a draft-layer chain (each draft after the first costs a call); explore: one more than the plan gives when there is
// room, so the position past the chosen length stays measured
struct RowOffer {
    int avail = 0;
    std::vector<double> reach;
    bool chain = false, explore = false;
    double first_ms = 0.0;   // what a lane's first draft costs beyond its row (DFlash2: its drafter round; a lane with no drafts needs none)
};

// the drafts each lane puts in a window of at most max_rows rows (every lane has its own first row) with the most
// expected tokens per ms (base_ms: the round's cost with one row a lane; rows cost by the window's row count, chain
// drafts their calls). Exact: for a rate lam, a DP over the lanes and the drafts so far takes each lane's best count for
// tokens - lam * ms; lam then becomes that plan's own rate, until no plan beats it (Dinkelbach; a few rounds). Explorers
// are explore_rows'
inline std::vector<int> plan_rows(const std::vector<RowOffer>& lanes, int max_rows, double base_ms, const RowCost& c = {}) {
    const int n = (int) lanes.size();
    if (n > max_rows) throw std::runtime_error("plan_rows: more lanes than rows");
    for (const RowOffer& o : lanes)
        if (o.avail < 0 || (int) o.reach.size() < o.avail) throw std::runtime_error("plan_rows: a lane offers drafts without their reach");
    const int room = max_rows - n;   // drafts in all, at most
    // per lane and count k: the expected tokens of its first k drafts and their chain calls' ms (rows are priced by total)
    std::vector<std::vector<double>> gain((size_t) n), calls((size_t) n);
    for (int i = 0; i < n; ++i) {
        const RowOffer& o = lanes[(size_t) i];
        const int most = std::min(o.avail, room);
        gain[(size_t) i].assign((size_t) most + 1, 0.0);
        calls[(size_t) i].assign((size_t) most + 1, 0.0);
        for (int k = 1; k <= most; ++k) {
            gain[(size_t) i][(size_t) k] = gain[(size_t) i][(size_t) k - 1] + o.reach[(size_t) k - 1];
            calls[(size_t) i][(size_t) k] = (o.chain ? (k - 1) * c.chain_ms : 0.0) + o.first_ms;
        }
    }
    auto rate_of = [&](const std::vector<int>& d) {
        double tokens = n, ms = base_ms;
        int t = 0;
        for (int i = 0; i < n; ++i) {
            tokens += gain[(size_t) i][(size_t) d[(size_t) i]];
            ms += calls[(size_t) i][(size_t) d[(size_t) i]];
            t += d[(size_t) i];
        }
        return tokens / (ms + c.rows_ms(n + t) - c.rows_ms(n));
    };
    std::vector<int> best((size_t) n, 0);
    double lam = rate_of(best);
    constexpr double NEG = -1e300;
    for (int round = 0; round < 32; ++round) {
        // f[i][t]: the best sum over lanes 0 .. i-1 holding t drafts of gain - lam * calls; pick[i][t]: lane i-1's count
        std::vector<std::vector<double>> f((size_t) n + 1, std::vector<double>((size_t) room + 1, NEG));
        std::vector<std::vector<int>> pick((size_t) n + 1, std::vector<int>((size_t) room + 1, 0));
        f[0][0] = 0.0;
        for (int i = 0; i < n; ++i)
            for (int t = 0; t <= room; ++t) {
                if (f[(size_t) i][(size_t) t] == NEG) continue;
                for (int k = 0; k < (int) gain[(size_t) i].size() && t + k <= room; ++k) {
                    const double v = f[(size_t) i][(size_t) t] + gain[(size_t) i][(size_t) k] - lam * calls[(size_t) i][(size_t) k];
                    if (v > f[(size_t) i + 1][(size_t) (t + k)]) f[(size_t) i + 1][(size_t) (t + k)] = v, pick[(size_t) i + 1][(size_t) (t + k)] = k;
                }
            }
        int bt = -1;
        double bv = NEG;
        for (int t = 0; t <= room; ++t) {
            if (f[(size_t) n][(size_t) t] == NEG) continue;
            const double v = n + f[(size_t) n][(size_t) t] - lam * (base_ms + c.rows_ms(n + t) - c.rows_ms(n));
            if (v > bv) bv = v, bt = t;
        }
        std::vector<int> d((size_t) n);
        for (int i = n, t = bt; i > 0; --i) {
            d[(size_t) i - 1] = pick[(size_t) i][(size_t) t];
            t -= d[(size_t) i - 1];
        }
        const double r = rate_of(d);
        if (r <= lam * (1.0 + 1e-12)) break;
        best = d;
        lam = r;
    }
    return best;
}

// d with each explorer one draft more while rows remain (so the position past its chosen length stays measured)
inline std::vector<int> explore_rows(const std::vector<RowOffer>& lanes, std::vector<int> d, int max_rows) {
    int rows = (int) lanes.size();
    for (int x : d) rows += x;
    for (size_t i = 0; i < lanes.size() && rows < max_rows; ++i)
        if (lanes[i].explore && d[i] < lanes[i].avail) ++d[i], ++rows;
    return d;
}

// a window's expected tokens per ms with d drafts a lane (capture_ms more when its layout has no graph held)
inline double plan_rate(const std::vector<RowOffer>& lanes, const std::vector<int>& d, double base_ms, const RowCost& c, double capture_ms) {
    double tokens = (double) lanes.size(), ms = base_ms + capture_ms;
    int rows = (int) lanes.size();
    for (size_t i = 0; i < lanes.size(); ++i) {
        if (d[i] > 0) ms += lanes[i].first_ms;
        for (int j = 0; j < d[i]; ++j) {
            tokens += lanes[i].reach[(size_t) j];
            ms += c.row(++rows) + (lanes[i].chain && j >= 1 ? c.chain_ms : 0.0);
        }
    }
    std::vector<int> counts(d);
    std::sort(counts.begin(), counts.end());
    ms += c.group_ms * (double) std::max<long>(0, std::unique(counts.begin(), counts.end()) - counts.begin() - 1);
    return tokens / ms;
}

// the drafts a lane for the next window, given the layouts held (each a row count a lane, drafts + 1) that cost no
// capture: plan_rows' plan, the uniform plans (k drafts a lane, fewer where a lane offers fewer) and each held layout
// within the offers and max_rows, each charged capture_ms (this window's share of a capture) unless held; the most
// expected tokens per ms (plan_rate: row counts and groups) wins. Explorers get their draft more where that layout is
// held too or the window captures anyway
inline std::vector<int> choose_rows(const std::vector<RowOffer>& lanes, int max_rows, double base_ms, const RowCost& c,
                                    const std::vector<std::vector<int>>& held, double capture_ms) {
    const size_t n = lanes.size();
    // a held layout's drafts, if it fits these offers (else empty)
    auto drafts_of = [&](const std::vector<int>& counts) {
        std::vector<int> d;
        if (counts.size() != n) return d;
        int rows = 0;
        for (size_t i = 0; i < n; ++i) {
            if (counts[i] < 1 || counts[i] - 1 > lanes[i].avail) return std::vector<int>{};
            d.push_back(counts[i] - 1);
            rows += counts[i];
        }
        return rows <= max_rows ? d : std::vector<int>{};
    };
    auto is_held = [&](const std::vector<int>& d) {
        for (const std::vector<int>& h : held)
            if (drafts_of(h) == d) return true;
        return false;
    };
    std::vector<int> best = plan_rows(lanes, max_rows, base_ms, c);
    bool best_held = is_held(best);
    double best_rate = plan_rate(lanes, best, base_ms, c, best_held ? 0.0 : capture_ms);
    int most = 0;
    for (const RowOffer& o : lanes) most = std::max(most, o.avail);
    for (int k = 0; k <= most; ++k) {
        std::vector<int> d;
        int rows = (int) n;
        for (const RowOffer& o : lanes) {
            d.push_back(std::min(k, o.avail));
            rows += d.back();
        }
        if (rows > max_rows) break;
        const bool h = is_held(d);
        const double r = plan_rate(lanes, d, base_ms, c, h ? 0.0 : capture_ms);
        if (r > best_rate) best = d, best_rate = r, best_held = h;
    }
    for (const std::vector<int>& h : held) {
        const std::vector<int> d = drafts_of(h);
        if (d.empty()) continue;
        const double r = plan_rate(lanes, d, base_ms, c, 0.0);
        if (r > best_rate) best = d, best_rate = r, best_held = true;
    }
    const std::vector<int> x = explore_rows(lanes, best, max_rows);
    return !best_held || is_held(x) ? x : best;
}

}  // namespace fnx
