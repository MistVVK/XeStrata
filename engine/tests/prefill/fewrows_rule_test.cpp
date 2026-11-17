// Few-row experts on the FP16 prompt path (fewrows.hpp): the research switch STRATA_PF_FEWROWS and the rule that picks the
// experts.  Header-only, no GPU.
#include <cstdio>

#include "strata/prefill/fewrows.hpp"

using strata::prefill::fewrows_switch;
using strata::prefill::fewrows_takes;
using strata::prefill::kFewRows;
using strata::prefill::kFewRowsRecBytes;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: unset or empty = the default the caller passes; "0" off; anything else on
    CHECK(!fewrows_switch(nullptr, false));
    CHECK(fewrows_switch(nullptr, true));
    CHECK(!fewrows_switch("", false));
    CHECK(fewrows_switch("", true));
    CHECK(!fewrows_switch("0", true));
    CHECK(!fewrows_switch("0", false));
    CHECK(fewrows_switch("1", false));
    CHECK(fewrows_switch("12", false));    // a number is not a row count any more: any non-zero value is "on"
    CHECK(fewrows_switch("on", false));

    // the constant and the plan record: 12 rows, 24 bytes of header + one int32 destination per row
    CHECK(kFewRows == 12);
    CHECK(kFewRowsRecBytes == 24 + 4 * 12);
    CHECK(kFewRowsRecBytes % 8 == 0);       // the u64 blob pointer at the start of the next record stays aligned

    // the rule: switch on, FP16 path, native pack, no APR, 1..kFewRows rows
    CHECK(fewrows_takes(true, true, true, false, 1));
    CHECK(fewrows_takes(true, true, true, false, kFewRows));
    CHECK(!fewrows_takes(true, true, true, false, 0));              // an expert with no rows is skipped elsewhere
    CHECK(!fewrows_takes(true, true, true, false, kFewRows + 1));   // 13 rows and up: dequantize + GEMMs
    CHECK(!fewrows_takes(true, true, true, false, 600));
    CHECK(!fewrows_takes(false, true, true, false, 6));             // switch off
    CHECK(!fewrows_takes(true, false, true, false, 6));             // MMQ or the fused kernels take the layer
    CHECK(!fewrows_takes(true, true, false, false, 6));             // a Strata pack's blob
    CHECK(!fewrows_takes(true, true, true, true, 6));               // APR: the FP16 path rotates itself
    CHECK(!fewrows_takes(true, true, true, false, -1));

    if (failed) {
        std::fprintf(stderr, "fewrows_rule_test: %d failed\n", failed);
        return 1;
    }
    std::printf("fewrows_rule_test: ok\n");
    return 0;
}
