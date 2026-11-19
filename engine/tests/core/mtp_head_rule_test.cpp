// mtp_draft_head_row_cap (include/strata/core/mtp_head_rule.hpp): the draft head subset's row bytes (#5841).  No GPU.
#include <cstdio>

#include "strata/core/mtp_head_rule.hpp"

using strata::core::mtp_draft_head_row_cap;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const int64_t n = 2560;
    const int64_t iq4_xs = n / 256 * 136;   // 1360
    const int64_t q4_0 = n / 32 * 18;       // 1440
    const int64_t q6_k = n / 256 * 210;     // 2100
    const int64_t q8_0 = n / 32 * 34;       // 2720
    CHECK(iq4_xs == 1360 && q4_0 == 1440);
    // #5841: an IQ4_XS head rewritten as Q4_0 needs the Q4_0 row: 106,299 rows x 80 bytes = 8.1 MiB more than the head's own
    CHECK(mtp_draft_head_row_cap(iq4_xs, n, true) == q4_0);
    CHECK(106299 * (mtp_draft_head_row_cap(iq4_xs, n, true) - iq4_xs) == 106299 * 80);
    // --mtp-q4 off: the head's own row, as before
    CHECK(mtp_draft_head_row_cap(iq4_xs, n, false) == iq4_xs);
    // formats at least as wide as Q4_0: unchanged
    CHECK(mtp_draft_head_row_cap(q6_k, n, true) == q6_k);
    CHECK(mtp_draft_head_row_cap(q8_0, n, true) == q8_0);
    CHECK(mtp_draft_head_row_cap(q4_0, n, true) == q4_0);
    // an n_embd that is no multiple of 32 has no Q4_0 row: the head's own
    CHECK(mtp_draft_head_row_cap(iq4_xs, 2564, true) == iq4_xs);
    if (failed) { std::fprintf(stderr, "mtp_head_rule_test: %d failed\n", failed); return 1; }
    std::printf("mtp_head_rule_test: ok\n");
    return 0;
}
