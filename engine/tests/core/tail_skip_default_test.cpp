// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
// STRATA_ROUTE_TAIL_SKIP is off when it is not set, on every backend (owner decision 2026-11-19, engine 0.1.63).  0.1.42 to 0.1.62 had rank 7
// as the CUDA / HIP gfx1200 / gfx1201 default; this test fails on that rule.
#include "strata/core/tail_skip_default.hpp"
#include <cassert>
#include <iostream>

using strata::core::kTailSkipUnsetRank;
using strata::core::tail_skip_rank_from_text;

int main() {
    // not set: off (the rule of 0.1.42 to 0.1.62 gave 7 on CUDA)
    static_assert(kTailSkipUnsetRank == 0, "the unset rank is off");
    assert(tail_skip_rank_from_text(nullptr) == 0);

    // set: a rank from 1 to 9 is itself, so STRATA_ROUTE_TAIL_SKIP=7 turns the skip on and any other rank is chosen the same way
    for (int r = 1; r <= 9; ++r) {
        const char text[2] = {static_cast<char>('0' + r), 0};
        assert(tail_skip_rank_from_text(text) == r);
    }
    // set to off, or to something that is not a rank 1 to 9: off, as before
    for (const char* off : {"0", "", "00", "10", "99", "-1", "-7", "off", "on", "abc", " "})
        assert(tail_skip_rank_from_text(off) == 0);
    // text that starts with a number is read as atoi reads it, as before
    assert(tail_skip_rank_from_text("7") == 7);
    assert(tail_skip_rank_from_text("7x") == 7);
    assert(tail_skip_rank_from_text(" 7") == 7);

    std::cout << "tail_skip_default_test: ok\n";
    return 0;
}
