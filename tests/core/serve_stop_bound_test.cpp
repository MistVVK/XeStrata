// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
#include "strata/core/serve_stop_bound.hpp"
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
}
