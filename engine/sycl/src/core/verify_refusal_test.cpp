// sycl/src/core/verify_refusal_test.cpp - a start the verifier refuses (no VRAM expert tier: --batch slot sessions and the MTP layer left the expert cache no room on an 8 GB
// card) ends with the refusal text and a clean exit.  Two verifiers whose init stops at the check never made queues of their own and hold the process's default queue; their
// destructors used to destroy it (twice), and the second wait died in urQueueFinish (SIGSEGV on the Arc A750 at the end of `--serve --batch 2 --mtp`).  After both are gone the
// default queue must still run work.
//   verify_refusal_test        (needs one SYCL GPU)
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/verify.hpp"

#include <cstdio>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    using namespace strata::core;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s\n", q->get_device().get_info<sycl::info::device::name>().c_str());
    {
        WeightTable wt;
        ModelGeometry g;
        SessionState ss;
        VerifyHits hits;   // no cache: d_res, cache_base null
        Verifier a, b;
        std::string ea, eb;
        const bool oka = a.init(wt, g, ss, hits, nullptr, 4, ea);
        const bool okb = b.init(wt, g, ss, hits, nullptr, 4, eb);
        check(!oka && ea.rfind("verify: needs the profile-filled VRAM expert tier", 0) == 0, "first verifier: init refuses, with the expert-tier text");
        check(!okb && eb.rfind("verify: needs the profile-filled VRAM expert tier", 0) == 0, "second verifier: init refuses, with the expert-tier text");
    }   // both destructed here
    float* p = sycl::malloc_device<float>(64, *q);
    q->fill(p, 2.0f, 64).wait();
    float h[64] = {};
    q->memcpy(h, p, sizeof(h)).wait();
    sycl::free(p, *q);
    check(h[0] == 2.0f && h[63] == 2.0f, "the process's default queue still runs after both destructors");
    std::printf("%s\n", g_fail ? "VERIFY-REFUSAL: FAIL" : "VERIFY-REFUSAL: ok");
    return g_fail ? 1 : 0;
}
