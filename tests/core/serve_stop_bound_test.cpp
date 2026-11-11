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
    // on unless the variable is exactly "0" (every backend, since the bound left HIP-only)
    assert(strata::core::serve_stop_bound_on(nullptr));
    assert(strata::core::serve_stop_bound_on("1"));
    assert(strata::core::serve_stop_bound_on(""));
    assert(!strata::core::serve_stop_bound_on("0"));
    std::cout << "SUMMARY stop_bound cases=" << cases << " passed\n";
}
