// own_only_line (include/strata/prefill/own_only_line.hpp): the start-up sentence of an engine built without llama.cpp's MMQ prompt kernels, from the
// layer counts per path.  No GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/own_only_line.hpp"

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
                   "strata: [DRAFT] prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run 48 on the staged fused entry.");
    }
    // an IQ pack no fused kernel covers: every layer on the int8 kernels at every size -> one clause
    {
        OwnOnlySplit s; s.i8 = 48;
        CHECK_LINE(own_only_line(48, 1024, s, s, false), "strata: [DRAFT] prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.");
    }
    // mixed: 20 fused / 28 int8 at the floor and above, the staged entry for the 20 below it
    {
        OwnOnlySplit big; big.fused = 20; big.i8 = 28;
        OwnOnlySplit small; small.staged = 20; small.i8 = 28;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: [DRAFT] prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels and 28 on the int8 kernels; "
                   "shorter chunks run 20 on the staged fused entry and 28 on the int8 kernels.");
        // the same card without the staged entry: the fused layers of a short chunk take the int8 kernels
        OwnOnlySplit small2; small2.i8 = 48;
        CHECK_LINE(own_only_line(48, 1024, big, small2, false),
                   "strata: [DRAFT] prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels and 28 on the int8 kernels; "
                   "shorter chunks run 48 on the int8 kernels.");
    }
    // FP16 above 0: a further part
    {
        OwnOnlySplit big; big.fused = 20; big.i8 = 25; big.fp16 = 3;
        OwnOnlySplit small; small.staged = 20; small.i8 = 25; small.fp16 = 3;
        CHECK_LINE(own_only_line(48, 1024, big, small, false),
                   "strata: [DRAFT] prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels, 25 on the int8 kernels and 3 on the FP16 path; "
                   "shorter chunks run 20 on the staged fused entry, 25 on the int8 kernels and 3 on the FP16 path.");
        OwnOnlySplit all; all.fp16 = 48;
        CHECK_LINE(own_only_line(48, 1024, all, all, false), "strata: [DRAFT] prompt kernels: every chunk runs 48 of 48 layers on the FP16 path.");
    }
    // --peer-device: the clause, the counts as they then are
    {
        OwnOnlySplit s; s.i8 = 40; s.fp16 = 8;
        CHECK_LINE(own_only_line(48, 1024, s, s, true),
                   "strata: [DRAFT] prompt kernels: every chunk runs 40 of 48 layers on the int8 kernels and 8 on the FP16 path (--peer-device: the fused kernels are off).");
    }
    if (failed) {
        std::fprintf(stderr, "own_only_line_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("own_only_line_test: OK\n");
    return 0;
}
