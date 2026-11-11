// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
#include "strata/core/serve_stop_bound.hpp"
#include <cassert>
#include <iostream>
#include <string>
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
    // --batch slots (P2b): the slot's windows (SlotHold helpers) against an independent per-token reference, over a whole reply
    // of several windows of changing width: same tokens, same finish, the commit never past what was emitted, the hold ending at
    // exactly N unless ID came first.  (A window may stop at the hold while ID is still in its rows: the next window goes on.)
    int slot_cases = 0;
    const int widths[4] = {1, 3, 2, 4};
    for (int first = 0; first < 4; ++first)
        for (int hold = 1; hold <= 10; ++hold)
            for (int max_new = 1; max_new <= 12; ++max_new)
                for (int id_at = -1; id_at < 12; ++id_at)
                    for (int eos = -1; eos < 12; ++eos) {
                        if (eos >= 0 && eos == id_at) continue;
                        if (hold > 12) continue;
                        std::vector<int> stream(14);
                        for (int i = 0; i < 14; ++i) stream[i] = i == eos ? -1 : i == id_at ? ID : 100 + i;
                        // reference: one token at a time
                        std::vector<int> ref;
                        std::string ref_fin;
                        {
                            bool on = true;
                            for (int i = 0; i < 14; ++i) {
                                ref.push_back(stream[i]);
                                if (stream[i] == ID) on = false;
                                const int produced = static_cast<int>(ref.size());
                                if (stream[i] == -1) { ref_fin = "stop"; break; }
                                if (produced >= max_new) { ref_fin = "length"; break; }
                                if (on && produced >= hold) { ref_fin = "hold"; break; }
                            }
                        }
                        // the slot: windows of widths[...] rows, the cap per window as generate.cpp has it
                        strata::core::SlotHold h = strata::core::slot_hold_from(hold, ID, true);
                        std::vector<int> got;
                        std::string fin;
                        int64_t produced = 0, committed_rows = 0;
                        for (int w = first; fin.empty() && produced < 14; ++w) {
                            const int nrow = widths[w % 4];
                            int kp = 1;   // the proposals the target picked, in order (here: all of them), and nothing ended
                            while (kp < nrow && produced + kp < 14 && stream[produced + kp - 1] != -1 &&
                                   produced + kp + 1 <= strata::core::slot_cap(max_new, h))
                                ++kp;
                            committed_rows += kp;
                            const int64_t base = produced;   // (the window's rows start here; `produced` counts on while they are emitted)
                            for (int j = 0; j < kp; ++j) {
                                const int y = stream[base + j];
                                got.push_back(y);
                                ++produced;
                                strata::core::slot_hold_emitted(h, y);
                                if (y == -1) fin = "stop";
                                else if (produced >= max_new) fin = "length";
                                else if (strata::core::slot_hold_reached(h, produced, max_new)) fin = "hold";
                                if (!fin.empty()) break;
                            }
                        }
                        if (got != ref || fin != ref_fin) {
                            std::cerr << "MISMATCH first=" << first << " hold=" << hold << " max_new=" << max_new << " id_at=" << id_at << " eos=" << eos
                                      << " got:";
                            for (int y : got) std::cerr << ' ' << y;
                            std::cerr << " (" << fin << ") ref:";
                            for (int y : ref) std::cerr << ' ' << y;
                            std::cerr << " (" << ref_fin << ")\n";
                        }
                        assert(got == ref);
                        assert(fin == ref_fin);
                        assert(committed_rows == static_cast<int64_t>(got.size()));   // not one row committed past what was emitted
                        if (fin == "hold") {
                            assert(static_cast<int>(got.size()) == hold);
                            for (int y : got) assert(y != ID);
                        }
                        // no hold: the hold helpers change nothing
                        assert(strata::core::slot_cap(max_new, strata::core::SlotHold{}) == max_new);
                        ++slot_cases;
                    }
    assert(!strata::core::slot_hold_from(5, -1, true).on && !strata::core::slot_hold_from(0, ID, true).on &&
           !strata::core::slot_hold_from(5, ID, false).on && strata::core::slot_hold_from(5, ID, true).on);
    std::cout << "SUMMARY slot hold cases=" << slot_cases << " passed\n";
    // on unless the variable is exactly "0" (every backend, since the bound left HIP-only)
    assert(strata::core::serve_stop_bound_on(nullptr));
    assert(strata::core::serve_stop_bound_on("1"));
    assert(strata::core::serve_stop_bound_on(""));
    assert(!strata::core::serve_stop_bound_on("0"));
    std::cout << "SUMMARY stop_bound cases=" << cases << " passed\n";
}
