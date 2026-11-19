// hybrid_plan (include/strata/qwen35/hybrid_fit.hpp): the Qwen3.6 hybrid placement's slots and VRAM need, with the fallback without the staging pair (#5533, #5285).
// The figures are the shape of UD-Q6_K_XL on a 10 GB card (RTX 3080): 40 layers + the MTP block, 256 experts, 2.58 MB an expert, 0.97 GiB free.  No GPU.
#include <cstdio>

#include "strata/qwen35/hybrid_fit.hpp"

using strata::qwen35::HybridIn;
using strata::qwen35::HybridPlan;
using strata::qwen35::hybrid_plan;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static HybridIn q6xl(size_t fr) {
    constexpr size_t MiB = (size_t) 1 << 20;
    const size_t stride = 2580000, ne = 256, layers = 41, tail = 64 << 10;
    HybridIn in;
    in.fr = fr;
    in.budget2 = fr > 768 * MiB ? fr - 768 * MiB : 0;   // place_experts' reserve
    in.per_slot = stride * layers;
    in.n_layers = layers;
    in.tail = tail;
    in.outputs = 1024 * 8 * 2048 * 4 + 1024 * 8 * 4;
    in.staging_full = 2 * (stride * ne + tail);
    in.streamed = true;
    in.ne = (int64_t) ne;
    return in;
}

int main() {
    constexpr size_t MiB = (size_t) 1 << 20, GiB = (size_t) 1 << 30;
    // the report: 989 MiB free, the staging pair (1.3 GB) alone is more than that, whatever the slots and the context
    {
        HybridIn in = q6xl(989 * MiB);
        in.allow_drop = false;   // the old plan: it does not fit, and the figure stays above the free VRAM
        const HybridPlan old = hybrid_plan(in);
        CHECK(!old.fits && old.need > in.fr && old.staging > in.fr);
        const HybridPlan p = hybrid_plan(q6xl(989 * MiB));
        CHECK(p.fits && p.dropped_staging && p.staging == 0);
        CHECK(p.S >= 1 && p.S < 256);
        CHECK(p.need <= 989 * MiB);
    }
    // more free memory than the pair needs: the streamed plan stays, nothing is dropped
    {
        const HybridPlan p = hybrid_plan(q6xl(8 * GiB));
        CHECK(p.fits && !p.dropped_staging && p.staging > 0 && p.S > 0);
    }
    // not even the plan without the pair fits (a few MiB free): reported as not fitting
    {
        const HybridPlan p = hybrid_plan(q6xl(40 * MiB));
        CHECK(!p.fits);
    }
    // the main layers all in VRAM, only the MTP block hybrid: no staging pair is asked, nothing changes
    {
        HybridIn in = q6xl(2 * GiB);
        in.streamed = false;
        in.mtp_only = true;
        in.total_main = 40 * (size_t) 660 * MiB;
        in.mtp_stride = 2580000;
        in.budget2 = in.total_main + 100 * MiB;
        in.fr = in.total_main + 1 * GiB;
        const HybridPlan p = hybrid_plan(in);
        CHECK(!p.dropped_staging && p.staging == 0 && p.fits);
    }
    // --expert-slots forces the count; a count that does not fit even without the pair is still refused
    {
        HybridIn in = q6xl(989 * MiB);
        in.slots_forced = 256;
        const HybridPlan p = hybrid_plan(in);
        CHECK(!p.fits);
        in.slots_forced = 2;
        const HybridPlan q = hybrid_plan(in);
        CHECK(q.fits && q.dropped_staging && q.S == 2);
    }
    if (failed) { std::fprintf(stderr, "q36_hybrid_fit_test: %d failed\n", failed); return 1; }
    std::printf("q36_hybrid_fit_test: ok\n");
    return 0;
}
