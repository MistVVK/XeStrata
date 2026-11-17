// The default of --pipeline-windows (strata::pipeline_auto_on), no GPU needed: on only for a layer split of exactly two
// Battlemage cards with --serve and the MTP drafter.
#include <cstdio>
#include "strata/pipeline_auto.hpp"

using strata::PipelineAutoInputs;
using strata::pipeline_auto_on;
static int bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++bad; } } while (0)

static PipelineAutoInputs two_b65() {
    PipelineAutoInputs in;
    in.serve = true; in.multi_gpu = true; in.later_stages = 1; in.has_mtp = true; in.spec = 4;
    in.first_is_battlemage = true; in.later_is_battlemage = true;
    return in;
}

int main() {
    CHECK(pipeline_auto_on(two_b65()), "two Battlemage cards, serve, MTP -> on");
    { auto in = two_b65(); in.spec = 2; CHECK(pipeline_auto_on(in), "spec 2 -> on"); }
    { auto in = two_b65(); in.spec = 1; CHECK(!pipeline_auto_on(in), "spec 1 (no draft window) -> off"); }
    { auto in = two_b65(); in.spec = 0; CHECK(!pipeline_auto_on(in), "no --spec -> off"); }
    { auto in = two_b65(); in.has_mtp = false; CHECK(!pipeline_auto_on(in), "no MTP drafter -> off"); }
    { auto in = two_b65(); in.serve = false; CHECK(!pipeline_auto_on(in), "no --serve -> off"); }
    { auto in = two_b65(); in.multi_gpu = false; CHECK(!pipeline_auto_on(in), "single card -> off"); }
    { auto in = two_b65(); in.split_same = true; CHECK(!pipeline_auto_on(in), "both stages on one GPU -> off"); }
    { auto in = two_b65(); in.later_stages = 0; CHECK(!pipeline_auto_on(in), "no later stage -> off"); }
    { auto in = two_b65(); in.later_stages = 2; CHECK(!pipeline_auto_on(in), "three stages (the pipeline runs exactly two) -> off"); }
    { auto in = two_b65(); in.later_stages = 3; CHECK(!pipeline_auto_on(in), "four stages -> off"); }
    { auto in = two_b65(); in.first_is_battlemage = false; CHECK(!pipeline_auto_on(in), "first card not Battlemage (A-series) -> off"); }
    { auto in = two_b65(); in.later_is_battlemage = false; CHECK(!pipeline_auto_on(in), "second card not Battlemage -> off"); }
    { auto in = two_b65(); in.first_is_battlemage = false; in.later_is_battlemage = false;
      CHECK(!pipeline_auto_on(in), "two A-series cards -> off"); }
    { auto in = two_b65(); in.auto_off = true; CHECK(!pipeline_auto_on(in), "STRATA_PIPELINE_AUTO=0 -> off"); }
    { auto in = two_b65(); in.batch = 4; CHECK(!pipeline_auto_on(in), "--batch slots -> off"); }
    { auto in = two_b65(); in.peer_device = true; CHECK(!pipeline_auto_on(in), "--peer-device -> off"); }
    { auto in = two_b65(); in.helper_caches = true; CHECK(!pipeline_auto_on(in), "helper caches -> off"); }
    if (bad == 0) std::printf("pipeline_auto_test: all OK\n");
    return bad == 0 ? 0 : 1;
}
