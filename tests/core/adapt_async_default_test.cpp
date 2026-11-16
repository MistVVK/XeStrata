// When the engine may choose the asynchronous adaptive tier by itself (include/strata/core/adapt_async_default.hpp): the one case where it
// may, and every condition that keeps it on the blocking tier, with its reason. #5814: STRATA_ADAPT_FETCH (the PCIe share's admissions into
// the VRAM tier) keeps it off, as serve keeps the admissions off under an explicit --adapt-async. Header-only, no GPU.
#include "strata/core/adapt_async_default.hpp"

#include <cstdio>
#include <cstring>

using strata::core::AdaptAsyncAuto;
using strata::core::adapt_async_self_decides;

static int fails = 0;

static void expect(const AdaptAsyncAuto& a, bool want, const char* want_why, const char* what) {
    const char* why = nullptr;
    const bool got = adapt_async_self_decides(a, &why);
    if (got != want || (!want && (why == nullptr || std::strcmp(why, want_why) != 0))) {
        std::printf("FAIL %s: got %s (%s), want %s (%s)\n", what, got ? "yes" : "no", why ? why : "-", want ? "yes" : "no", want_why ? want_why : "-");
        ++fails;
    }
}

int main() {
    AdaptAsyncAuto ok;   // a one-GPU CUDA serve session in the arena with the adaptive tier on
    ok.cuda = ok.serve = ok.adaptive = ok.arena = true;
    expect(ok, true, nullptr, "the case the engine decides");

    AdaptAsyncAuto a = ok;
    a.flag = 1;
    expect(a, false, "--adapt-async was given", "--adapt-async 1");
    a = ok; a.flag = 0;
    expect(a, false, "--adapt-async was given", "--adapt-async 0");
    a = ok; a.env = 0;
    expect(a, false, "STRATA_ADAPT_ASYNC=0", "STRATA_ADAPT_ASYNC=0");
    a = ok; a.cuda = false;
    expect(a, false, "not a CUDA build", "HIP");
    a = ok; a.serve = false;
    expect(a, false, "it needs --serve", "strata generate");
    a = ok; a.adaptive = false;
    expect(a, false, "the adaptive tier is off", "no adaptive tier");
    a = ok; a.arena = false;
    expect(a, false, "not the expert arena", "no arena");
    a = ok; a.multi_gpu = true;
    expect(a, false, "a layer split", "layer split");
    a = ok; a.pipeline_windows = true;
    expect(a, false, "--pipeline-windows", "pipelined windows");
    a = ok; a.batch = true;
    expect(a, false, "--batch slots", "batch slots");
    a = ok; a.peer = true;
    expect(a, false, "--peer-device", "peer tier");
    a = ok; a.remote_opt = true;
    expect(a, false, "--remote-expert-opt", "remote expert opt");
    a = ok; a.adapt_overlap = true;
    expect(a, false, "STRATA_ADAPT_OVERLAP", "adapt overlap");
    // #5814
    a = ok; a.fetch = 1;
    expect(a, false, "STRATA_ADAPT_FETCH", "STRATA_ADAPT_FETCH=1");
    a = ok; a.fetch = 2;
    expect(a, false, "STRATA_ADAPT_FETCH", "STRATA_ADAPT_FETCH=2");
    a = ok; a.fetch = 0;
    expect(a, true, nullptr, "STRATA_ADAPT_FETCH unset or 0");

    std::printf("adapt_async_default_test: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}
