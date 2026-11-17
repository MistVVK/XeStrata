// research/stage-reserve-proposal/proposal_test.cpp - the checks of the unused window-term proposal (moved out of
// tests/program/stage_reserve_test.cpp in cleanup step 4.2; the assertions are unchanged). Not built by CMake.
// Build and run by hand, see README.md in this folder.
#include "proposal.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

using namespace strata::program::stage_reserve;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
}  // namespace

int main() {
    // the proposal: the window term grows with the number of window sizes (--spec 4 -> windows of 1..6 tokens)
    check(proposed_window_mib(1) == 96 + 48, "proposal: one window size = the arena and the burst");
    check(proposed_window_mib(6) == 96 + 5 * 6 + 48, "proposal: six sizes = 174 MiB");
    check(proposed_window_mib(0) == proposed_window_mib(1), "proposal: no negative sizes");
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        check(proposed_reserve_mib(in, 6) == 700 + 174, "proposed reserve on a later card, six sizes");
        in.later = false;
        check(proposed_reserve_mib(in, 6) == reserve_mib(in), "the first card is not changed by the proposal");
    }
    std::printf("proposal_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
