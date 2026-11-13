// own_only_line (include/strata/prefill/own_only_line.hpp): the start-up sentence of an engine built without llama.cpp's MMQ prompt kernels, from the
// layer counts per path.  No GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/own_only_line.hpp"
#include "strata/prefill/plain_floor_rule.hpp"

using strata::prefill::OwnOnlySplit;
using strata::prefill::own_only_line;

static int failed = 0;
#define CHECK_LINE(got, want) do { const std::string g_ = (got); if (g_ != (want)) { std::fprintf(stderr, "FAIL %s:%d\n  got:  %s\n  want: %s\n", __FILE__, __LINE__, g_.c_str(), (want)); ++failed; } } while (0)

int main() {
    // Strix-like: every layer on the fused kernels for big chunks, on the staged fused entry for small ones
    {
        OwnOnlySplit big; big.fused = 48;
        OwnOnlySplit small; small.staged = 48;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the staged fused entry.");
    }
    // an IQ pack no fused kernel covers: every layer on the int8 kernels at every size -> one clause
    {
        OwnOnlySplit s; s.i8 = 48;
        CHECK_LINE(own_only_line(48, 1024, s, s, false), "strata: prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.");
    }
    // mixed: 20 fused / 28 int8 at the floor and above, the staged entry for the 20 below it
    {
        OwnOnlySplit big; big.fused = 20; big.i8 = 28;
        OwnOnlySplit small; small.staged = 20; small.i8 = 28;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels and 28 on the int8 kernels; "
                   "shorter chunks run 20 on the staged fused entry and 28 on the int8 kernels.");
        // the same card without the staged entry: the fused layers of a short chunk take the int8 kernels
        OwnOnlySplit small2; small2.i8 = 48;
        CHECK_LINE(own_only_line(48, 1024, big, small2, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels and 28 on the int8 kernels; "
                   "shorter chunks run 48 on the int8 kernels.");
    }
    // FP16 above 0: a further part
    {
        OwnOnlySplit big; big.fused = 20; big.i8 = 25; big.fp16 = 3;
        OwnOnlySplit small; small.staged = 20; small.i8 = 25; small.fp16 = 3;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels, 25 on the int8 kernels and 3 on the FP16 path; "
                   "shorter chunks run 20 on the staged fused entry, 25 on the int8 kernels and 3 on the FP16 path.");
        OwnOnlySplit all; all.fp16 = 48;
        CHECK_LINE(own_only_line(48, 1024, all, all, false), "strata: prompt kernels: every chunk runs 48 of 48 layers on the FP16 path.");
    }
    // --peer-device: the clause, the counts as they then are
    {
        OwnOnlySplit s; s.i8 = 40; s.fp16 = 8;
        CHECK_LINE(own_only_line(48, 1024, s, s, true),
                   "strata: prompt kernels: every chunk runs 40 of 48 layers on the int8 kernels and 8 on the FP16 path (--peer-device: the fused kernels are off).");
    }
    // the Tesla P100 in a build without MMQ (no fused kernels; the int8 default under the chunk limit, the FP16 path from it): the stream floor is the limit: FP16 from it up
    {
        OwnOnlySplit big; big.fp16 = 48;
        OwnOnlySplit small; small.i8 = 48;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the FP16 path; shorter chunks run 48 on the int8 kernels.");
    }
    // the counts of own_only_count_layer: 48 Hadamard-folded fused-only layers (Gyro-S): the staged entry and the int8 kernels read no folded layer, so below the floor they are on the FP16 path
    {
        OwnOnlySplit big, small;
        for (int l = 0; l < 48; ++l) strata::prefill::own_only_count_layer(big, small, true, false, false, true, false);
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the FP16 path.");
    }
    // the same layers unfolded: the staged entry takes them (the line as before the change)
    {
        OwnOnlySplit big, small;
        for (int l = 0; l < 48; ++l) strata::prefill::own_only_count_layer(big, small, true, false, false, true, true);
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the staged fused entry.");
    }
    // a mix: 40 folded and 8 unfolded fused-only layers; and an int8 layer and a gated class keep their counts
    {
        OwnOnlySplit big, small;
        for (int l = 0; l < 40; ++l) strata::prefill::own_only_count_layer(big, small, true, false, false, true, false);
        for (int l = 0; l < 8; ++l) strata::prefill::own_only_count_layer(big, small, true, false, false, true, true);
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 8 on the staged fused entry and 40 on the FP16 path.");
        OwnOnlySplit b2, s2;
        strata::prefill::own_only_count_layer(b2, s2, false, true, true, true, true);   // an int8 layer on a gated class: FP16 in the long class, int8 below
        if (!(b2.fp16 == 1 && s2.i8 == 1 && b2.i8 == 0)) { std::fprintf(stderr, "FAIL %s:%d: gated int8 layer\n", __FILE__, __LINE__); ++failed; }
    }
    // The printed floor is the one the rule returns once the share is armed and the card has answered (plain_floor_rule.hpp floor_with_share), not the value at the
    // time the plan is made.  Plain Q2_0 pack in a build without MMQ: 48 layers on the fused kernels at the floor and above, 48 on the FP16 path below it.
    {
        using strata::prefill::floor_with_share;
        using strata::prefill::OwnOnlyLineDeferred;
        OwnOnlySplit big; big.fused = 48;
        OwnOnlySplit small; small.fp16 = 48;
        const char* line_1536 = "strata: prompt kernels: chunks of 1536 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the FP16 path.";
        const char* line_3072 = "strata: prompt kernels: chunks of 3072 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the FP16 path.";
        const char* line_64 = "strata: prompt kernels: chunks of 64 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the FP16 path.";
        // the plan is made first (the old line printed 64 here): nothing is due until the share is armed
        OwnOnlyLineDeferred d;
        d.plan(48, big, small, false);
        if (d.due()) { std::fprintf(stderr, "FAIL %s:%d: due before the share is armed\n", __FILE__, __LINE__); ++failed; }
        d.mark_ready();
        if (!d.due()) { std::fprintf(stderr, "FAIL %s:%d: not due after the share is armed\n", __FILE__, __LINE__); ++failed; }
        // a discrete CUDA card, share armed: the plain pack's floor 1,536
        CHECK_LINE(d.take(floor_with_share(true, 3072, 64)), line_1536);
        if (d.due()) { std::fprintf(stderr, "FAIL %s:%d: printed twice\n", __FILE__, __LINE__); ++failed; }
        // the same plan with the rule switched off (STRATA_PF_PLAIN_FLOOR=0) or on an integrated card: the share's limit when armed, else the pack's 64
        OwnOnlyLineDeferred e;
        e.plan(48, big, small, false);
        e.mark_ready();
        CHECK_LINE(e.take(floor_with_share(false, 3072, 64)), line_3072);
        OwnOnlyLineDeferred f;
        f.mark_ready();   // armed before the plan was made: the plan prints at once
        if (f.due()) { std::fprintf(stderr, "FAIL %s:%d: due without counts\n", __FILE__, __LINE__); ++failed; }
        f.plan(48, big, small, false);
        if (!f.due()) { std::fprintf(stderr, "FAIL %s:%d: a plan made after the share was armed is not due\n", __FILE__, __LINE__); ++failed; }
        CHECK_LINE(f.take(floor_with_share(false, 0, 64)), line_64);
    }
    if (failed) {
        std::fprintf(stderr, "own_only_line_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("own_only_line_test: OK\n");
    return 0;
}
