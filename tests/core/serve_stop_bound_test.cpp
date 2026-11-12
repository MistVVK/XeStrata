// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
#include "strata/core/serve_stop_bound.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    int cases = 0;
    // Independent reference: the existing emission loop, including its budget and EOS rules.
    for (int accepted = 1; accepted <= 16; ++accepted)
        for (int remaining = 1; remaining <= 18; ++remaining)
            for (int eos = -1; eos < accepted; ++eos) {
                std::vector<int> outputs(accepted);
                for (int i = 0; i < accepted; ++i) outputs[i] = i == eos ? -1 : 100 + i;
                std::vector<int> reference;
                for (int i = 0; i < accepted && static_cast<int>(reference.size()) < remaining; ++i) {
                    reference.push_back(outputs[i]);
                    if (outputs[i] == -1) break;
                }
                const int kept = strata::core::serve_stop_bound(accepted, remaining, outputs.data(),
                                                               [](int t) { return t == -1; });
                assert(kept > 0 && kept <= accepted && kept <= remaining);
                assert(std::vector<int>(outputs.begin(), outputs.begin() + kept) == reference);
                // Unlike the old full-window commit, no unexposed input suffix enters the cached history.
                assert(kept == static_cast<int>(reference.size()));
                ++cases;
            }
    assert(strata::core::serve_stop_bound(4, 0, static_cast<const int*>(nullptr), [](int) { return false; }) == 0);
    // GEN hold=N:ID: against the emission loop with the hold (stop after N emitted tokens unless ID came first; EOS and
    // max_new as before), for every window shape, start, hold, max_new and place of ID / EOS in the window
    int hold_cases = 0;
    const int ID = 7777;   // a real token id (a negative one means "no hold")
    for (int accepted = 1; accepted <= 8; ++accepted)
        for (int produced = 0; produced <= 6; ++produced)
            for (int hold = 1; hold <= 10; ++hold)
                for (int max_new = produced + 1; max_new <= 12; ++max_new)
                    for (int id_at = -1; id_at < accepted; ++id_at)
                        for (int eos = -1; eos < accepted; ++eos) {
                            if (eos >= 0 && eos == id_at) continue;
                            if (produced >= hold) continue;   // a hold the request already reached has ended it
                            std::vector<int> outputs(accepted);
                            for (int i = 0; i < accepted; ++i) outputs[i] = i == eos ? -1 : i == id_at ? ID : 100 + i;
                            std::vector<int> reference;
                            bool on = true;
                            int64_t done = produced;
                            for (int i = 0; i < accepted; ++i) {
                                const int64_t cap = on ? std::min(max_new, hold) : max_new;
                                if (done >= cap) break;
                                reference.push_back(outputs[i]);
                                ++done;
                                if (outputs[i] == ID) on = false;
                                if (outputs[i] == -1) break;
                            }
                            const int64_t rem = strata::core::serve_hold_remaining(accepted, produced, max_new, hold, ID,
                                                                                   outputs.data());
                            const int kept = strata::core::serve_stop_bound(accepted, rem, outputs.data(),
                                                                           [](int t) { return t == -1; });
                            assert(kept == static_cast<int>(reference.size()));
                            assert(std::vector<int>(outputs.begin(), outputs.begin() + kept) == reference);
                            // no hold: exactly the max_new bound
                            assert(strata::core::serve_hold_remaining(accepted, produced, max_new, -1, ID, outputs.data()) ==
                                   max_new - produced);
                            ++hold_cases;
                        }
    std::cout << "SUMMARY hold cases=" << hold_cases << " passed\n";
    // on unless the variable is exactly "0" (every backend, since the bound left HIP-only)
    assert(strata::core::serve_stop_bound_on(nullptr));
    assert(strata::core::serve_stop_bound_on("1"));
    assert(strata::core::serve_stop_bound_on(""));
    assert(!strata::core::serve_stop_bound_on("0"));
    std::cout << "SUMMARY stop_bound cases=" << cases << " passed\n";

    // ---- STRATA_SERVE_STOP_SKIP: a window that finishes after the server's STOP is dropped whole -----------------------
    // the decision: only when enabled, STOP seen, not merged, not the request's first window, and the commit still undone
    int skip_cases = 0;
    for (int m = 0; m < 32; ++m) {
        const bool enabled = m & 1, stop = m & 2, merged = m & 4, first = m & 8, deferred = m & 16;
        assert(strata::core::serve_skip_stopped_window(enabled, stop, merged, first, deferred) ==
               (enabled && stop && !merged && !first && deferred));
        ++skip_cases;
    }
    assert(strata::core::serve_stop_skip_on(nullptr));
    assert(strata::core::serve_stop_skip_on("1"));
    assert(strata::core::serve_stop_skip_on(""));
    assert(!strata::core::serve_stop_skip_on("0"));
    // A ledger of the serial loop: windows of 1..6 rows; each window commits and prints its rows (bounded by the cap, as above) unless it is dropped;
    // STOP is seen from the end of window `stop_at` on (-1: never) and the loop breaks after the first window that saw it.
    // Properties: (1) no STOP, or the switch off: the ledger is the plain loop's; (2) the switch on and the stopped window eligible (not the
    // first, commit undone): that window adds nothing - committed rows = the rows of the windows before it; (3) not eligible (the first window,
    // or a one-row window the state advanced itself): as without the switch; (4) the cap ends a request before STOP is looked at: same rows.
    int ledger_cases = 0;
    for (int code = 0; code < 6 * 6 * 6 * 6; ++code) {
        int rows[4];
        for (int w = 0; w < 4; ++w) rows[w] = 1 + (code / (w == 0 ? 1 : w == 1 ? 6 : w == 2 ? 36 : 216)) % 6;
        for (int stop_at = -1; stop_at < 4; ++stop_at)
            for (int self_mask = 0; self_mask < 16; ++self_mask)   // window w is a self-committed one-row window (only when rows[w] == 1)
                for (int cap = 1; cap <= 12; cap += 3)
                    for (int on = 0; on < 2; ++on) {
                        auto run = [&](bool enabled, int& committed, int& emitted) {
                            committed = emitted = 0;
                            for (int w = 0; w < 4 && emitted < cap; ++w) {
                                const bool deferred = !(rows[w] == 1 && ((self_mask >> w) & 1));
                                const bool stop_seen = stop_at >= 0 && w >= stop_at;
                                if (strata::core::serve_skip_stopped_window(enabled, stop_seen, false, w == 0, deferred)) break;
                                const int b = std::min(rows[w], cap - emitted);
                                committed += b;
                                emitted += b;
                                if (stop_seen) break;
                            }
                        };
                        int c_on = 0, e_on = 0, c_off = 0, e_off = 0;
                        run(on != 0, c_on, e_on);
                        run(false, c_off, e_off);
                        assert(c_on == e_on && c_off == e_off);   // rows committed = tokens printed in this model: never a row the client was not sent
                        if (!on || stop_at < 0) {
                            assert(c_on == c_off);   // (1)
                        } else {
                            // the rows of the windows before `stop_at`, up to the cap
                            int before = 0;
                            for (int w = 0; w < stop_at && before < cap; ++w) before += std::min(rows[w], cap - before);
                            const bool deferred = !(rows[stop_at] == 1 && ((self_mask >> stop_at) & 1));
                            if (before >= cap) assert(c_on == before && c_off == before);   // (4)
                            else if (stop_at > 0 && deferred) {
                                assert(c_on == before);                                       // (2)
                                assert(c_off == before + std::min(rows[stop_at], cap - before));
                            } else {
                                assert(c_on == c_off);                                        // (3)
                            }
                        }
                        ++ledger_cases;
                    }
    }
    std::cout << "SUMMARY stop_skip decision cases=" << skip_cases << " ledger cases=" << ledger_cases << " passed\n";
}
