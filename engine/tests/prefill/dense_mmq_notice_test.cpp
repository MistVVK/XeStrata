// dense_mmq_notice (include/strata/prefill/dense_mmq_notice.hpp, #6269): the start-up line of an engine without MMQ for a STRATA_DENSE_MMQ that is set on.  No GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/dense_mmq_notice.hpp"

using strata::prefill::dense_mmq_notice;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    CHECK(dense_mmq_notice(nullptr).empty());   // not set: nothing
    CHECK(dense_mmq_notice("").empty());
    CHECK(dense_mmq_notice("0").empty());       // off: nothing to say
    const std::string on = dense_mmq_notice("1");
    CHECK(on == "strata: STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels): the dense projections of a prompt run on the FP16 path.");
    CHECK(!dense_mmq_notice("on").empty());
    CHECK(!dense_mmq_notice("true").empty());
    CHECK(!dense_mmq_notice("00").empty());     // only the exact "0" is off
    if (failed) { std::fprintf(stderr, "%d failed\n", failed); return 1; }
    std::printf("dense_mmq_notice_test: all ok\n");
    return 0;
}
