// i8_expert_rule (include/strata/kernels/i8_expert_rule.hpp): when a prompt layer's expert groups take our own int8 kernels (native_expert_i8).
// Nothing set: the HIP cards gfx1030 / gfx1031 / gfx1032 take them (groups averaging 256 entries or more, the layout and the device allowing it); every
// other card keeps MMQ.  STRATA_EXPERT_I8=1 asks for them on any card, =0 goes back to the old kernels on the class.  No GPU, no model.
#include <cstdio>
#include <initializer_list>

#include "strata/kernels/i8_expert_rule.hpp"

using namespace strata::kernels;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the state of the variable
    CHECK(i8_switch_parse(nullptr) == -1);
    CHECK(i8_switch_parse("") == -1);
    CHECK(i8_switch_parse("0") == 0);
    CHECK(i8_switch_parse("1") == 1);
    CHECK(i8_switch_parse("yes") == 1);
    CHECK(i8_switch_parse("0x") == 0);
    // the class: exactly the three RDNA2 targets, with or without the feature suffix hipGetDeviceProperties adds
    CHECK(gfx_arch_is_gfx103x_i8("gfx1030"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1031"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1032"));
    CHECK(gfx_arch_is_gfx103x_i8("gfx1031:sramecc-:xnack-"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1034"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1035"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1036"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1100"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1151"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx1201"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx906"));
    CHECK(!gfx_arch_is_gfx103x_i8("gfx10310"));
    CHECK(!gfx_arch_is_gfx103x_i8(""));
    CHECK(!gfx_arch_is_gfx103x_i8(nullptr));   // a CUDA card
    // nothing set: the class takes them from 256 entries per group, smaller groups and every other card keep today's path
    CHECK(i8_expert_takes(-1, "gfx1031", 256, true));
    CHECK(i8_expert_takes(-1, "gfx1030", 4096, true));
    CHECK(i8_expert_takes(-1, "gfx1032:sramecc-:xnack-", 300, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 255, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 0, true));
    CHECK(!i8_expert_takes(-1, "gfx1031", 4096, false));   // a format or a size the kernel does not cover, or a card whose shared memory is too small
    CHECK(!i8_expert_takes(-1, "gfx1201", 4096, true));    // RDNA4: MMQ / the fused kernels
    CHECK(!i8_expert_takes(-1, "gfx1100", 4096, true));
    CHECK(!i8_expert_takes(-1, "gfx1034", 4096, true));
    CHECK(!i8_expert_takes(-1, nullptr, 4096, true));      // CUDA
    CHECK(!i8_expert_takes(-1, "", 4096, true));
    // =0: the way back, on the class and everywhere
    CHECK(!i8_expert_takes(0, "gfx1031", 4096, true));
    CHECK(!i8_expert_takes(0, "gfx1030", 100000, true));
    CHECK(!i8_expert_takes(0, nullptr, 4096, true));
    // =1: any card that the kernel covers, same size floor
    CHECK(i8_expert_takes(1, "gfx1201", 256, true));
    CHECK(i8_expert_takes(1, nullptr, 4096, true));
    CHECK(i8_expert_takes(1, "gfx1031", 256, true));
    CHECK(!i8_expert_takes(1, "gfx1201", 255, true));
    CHECK(!i8_expert_takes(1, "gfx1201", 4096, false));
    // which of the two reasons: the class default only when the variable is unset
    CHECK(i8_expert_class_default(-1, "gfx1031"));
    CHECK(!i8_expert_class_default(1, "gfx1031"));
    CHECK(!i8_expert_class_default(0, "gfx1031"));
    CHECK(!i8_expert_class_default(-1, "gfx1201"));
    CHECK(!i8_expert_class_default(-1, nullptr));
    // a build without llama.cpp's MMQ: an unset switch reads as 1 on every card, a value the user set is kept; a build with MMQ is the table above
    CHECK(i8_switch_effective(-1, true) == 1);
    CHECK(i8_switch_effective(0, true) == 0);
    CHECK(i8_switch_effective(1, true) == 1);
    CHECK(i8_switch_effective(-1, false) == -1);
    CHECK(i8_switch_effective(0, false) == 0);
    CHECK(i8_switch_effective(1, false) == 1);
    {
        // state {unset, 0, 1} x build {with MMQ, without} x card {RDNA2 gfx1030, gfx1100, Strix gfx1151, R9700 gfx1201, an NVIDIA card}: takes at 4,096 entries, the formats covered
        const char* const cards[] = {"gfx1030", "gfx1100", "gfx1151", "gfx1201:sramecc-:xnack-", nullptr, ""};
        for (const char* gcn : cards) {
            const bool rdna2 = gfx_arch_is_gfx103x_i8(gcn);
            // with MMQ: unchanged (only RDNA2 by default)
            CHECK(i8_expert_takes(i8_switch_effective(-1, false), gcn, 4096, true) == rdna2);
            CHECK(!i8_expert_takes(i8_switch_effective(0, false), gcn, 4096, true));
            CHECK(i8_expert_takes(i8_switch_effective(1, false), gcn, 4096, true));
            // without MMQ: on by default everywhere, =0 the way back, =1 as before
            CHECK(i8_expert_takes(i8_switch_effective(-1, true), gcn, 4096, true));
            CHECK(!i8_expert_takes(i8_switch_effective(0, true), gcn, 4096, true));
            CHECK(i8_expert_takes(i8_switch_effective(1, true), gcn, 4096, true));
            // the size floor and the format check are the same in both builds
            CHECK(!i8_expert_takes(i8_switch_effective(-1, true), gcn, 255, true));
            CHECK(!i8_expert_takes(i8_switch_effective(-1, true), gcn, 4096, false));
            // which reason the engine's start line gives
            CHECK(i8_expert_class_default(-1, gcn, true));
            CHECK(!i8_expert_class_default(1, gcn, true));
            CHECK(!i8_expert_class_default(0, gcn, true));
            CHECK(i8_expert_class_default(-1, gcn, false) == rdna2);
            CHECK(i8_expert_class_default(-1, gcn) == rdna2);   // (the old two-argument form)
        }
    }
    // the order of the paths for a layer of a chunk: the fused kernels, then the staged group entry, then int8, then FP16 (a column "fused covers this layer and is on" and one for the staged entry);
    // the same for a switch left unset in a build without MMQ and for a user-set 1: the int8 kernels get only what nothing faster takes
    for (const char* gcn : {"gfx1201", "gfx1151", "gfx1100", "", "gfx1030"}) {
        for (int raw : {-1, 1}) {
            CHECK(i8_expert_takes(i8_switch_effective(raw, true), gcn, 4096, true));      // the kernel itself would take the layer on every card
            CHECK(!i8_layer_takes(raw, true, gcn, 4096, true, true, false));              // fused covers the layer and is on: not int8
            CHECK(!i8_layer_takes(raw, true, gcn, 4096, true, false, true));              // a short chunk the staged group entry takes: not int8
            CHECK(!i8_layer_takes(raw, true, gcn, 4096, true, true, true));
            CHECK(i8_layer_takes(raw, true, gcn, 4096, true, false, false));              // what is left: int8
            CHECK(!i8_layer_takes(raw, true, gcn, 4096, false, false, false));            // a format the kernel does not cover: FP16
        }
        CHECK(!i8_layer_takes(0, true, gcn, 4096, true, false, false));                   // the way back
        CHECK(!i8_layer_takes(-1, false, gcn, 4096, true, true, false));                  // (with MMQ: fused first as well)
        CHECK(i8_layer_takes(-1, false, gcn, 4096, true, false, false) == gfx_arch_is_gfx103x_i8(gcn));
    }
    // the few-rows route: the limit (STRATA_I8_FEWROWS: not set = the class default, 0 = off, N = N rows per expert)
    CHECK(i8_fewrows_limit(nullptr, 10) == 10);
    CHECK(i8_fewrows_limit("", 8) == 8);
    CHECK(i8_fewrows_limit("0", 10) == 0);
    CHECK(i8_fewrows_limit("0x", 10) == 0);
    CHECK(i8_fewrows_limit("12", 10) == 12);
    CHECK(i8_fewrows_limit("6", 8) == 6);
    CHECK(i8_fewrows_limit("yes", 10) == 10);
    CHECK(i8_fewrows_limit("-3", 8) == 8);
    CHECK(i8_fewrows_limit("99999999999", 10) > 100000);   // a huge number does not wrap to a small one
    // the class default: the P100 (6.0) 6, every other card 7 (6.1, 7.0, 7.5, 8.6, 12.0, a HIP card with no capability at all)
    CHECK(i8_fewrows_class_default(6, 0) == 6);
    CHECK(i8_fewrows_class_default(6, 1) == 7);
    CHECK(i8_fewrows_class_default(7, 0) == 7);
    CHECK(i8_fewrows_class_default(8, 6) == 7);
    CHECK(i8_fewrows_class_default(12, 0) == 7);
    CHECK(i8_fewrows_class_default(0, 0) == 7);
    // the decision: rows strictly under limit x experts go to the grouped kernels (512 experts: 5,119 rows is under 10 per expert, 5,120 is 10)
    CHECK(i8_fewrows_takes(10, 5119, 512));
    CHECK(!i8_fewrows_takes(10, 5120, 512));
    CHECK(!i8_fewrows_takes(10, 20480, 512));
    CHECK(i8_fewrows_takes(8, 4095, 512));
    CHECK(!i8_fewrows_takes(8, 4096, 512));
    CHECK(i8_fewrows_takes(10, 2560, 400));      // 256 tokens x 10 over the 400 experts that got rows: 6.4 per expert
    CHECK(!i8_fewrows_takes(0, 100, 512));       // the route is off
    CHECK(!i8_fewrows_takes(10, 100, 0));        // no experts: nothing to decide
    CHECK(!i8_fewrows_takes(-1, 100, 512));
    // the 8-entry slices: not set (-1) = tile 2 only, 1 = tiles 2-4, 0 = none; tile 8 (the 128-entry kernel) never has them
    CHECK(i8_n8_takes(-1, 2));
    CHECK(!i8_n8_takes(-1, 3));
    CHECK(!i8_n8_takes(-1, 4));
    CHECK(!i8_n8_takes(-1, 8));
    CHECK(i8_n8_takes(1, 2));
    CHECK(i8_n8_takes(1, 3));
    CHECK(i8_n8_takes(1, 4));
    CHECK(!i8_n8_takes(1, 8));
    CHECK(!i8_n8_takes(0, 2));
    CHECK(!i8_n8_takes(0, 4));
    CHECK(i8_n8_takes(i8_switch_parse(nullptr), 2));
    CHECK(i8_n8_takes(i8_switch_parse(""), 2));
    CHECK(!i8_n8_takes(i8_switch_parse("0"), 2));
    CHECK(i8_n8_takes(i8_switch_parse("1"), 4));
    // the P100 class rule (build without MMQ, switch not set): the int8 default applies under the stream floor (stream_all_min, 1,024 tokens by default) only; the FP16 path from it up (the build with MMQ's default there)
    {
        const int64_t lim = 1024;   // stream_all_min() by default: the rule is the comparison, rows at floor - 1, floor, floor + 1
        CHECK(i8_chunk_default_allows(-1, true, true, 1, lim));
        CHECK(i8_chunk_default_allows(-1, true, true, lim - 1, lim));
        CHECK(!i8_chunk_default_allows(-1, true, true, lim, lim));
        CHECK(!i8_chunk_default_allows(-1, true, true, lim + 1, lim));
        CHECK(i8_chunk_default_allows(-1, true, true, 300, lim));
        CHECK(i8_chunk_default_allows(-1, true, true, 1000, lim));
        CHECK(i8_chunk_default_allows(-1, true, true, lim - 1, lim));
        CHECK(!i8_chunk_default_allows(-1, true, true, lim, lim));
        CHECK(!i8_chunk_default_allows(-1, true, true, 2048, lim));
        CHECK(!i8_chunk_default_allows(-1, true, true, 8192, lim));
        // a value the user set keeps its meaning at every size (the rule is not asked to veto it)
        CHECK(i8_chunk_default_allows(1, true, true, 8192, lim));
        CHECK(i8_chunk_default_allows(0, true, true, 8192, lim));
        // not the P100 (6.1, 7.x, 8.x, a HIP card): no limit; a build with MMQ: no limit (its P100 default is the FP16 path already)
        CHECK(i8_chunk_default_allows(-1, true, false, 8192, lim));
        CHECK(i8_chunk_default_allows(-1, false, true, 8192, lim));
        CHECK(i8_chunk_default_allows(-1, false, false, 8192, lim));
    }
    // the Volta tile rule (TUNE-0216): the largest tile 2-4 (8 counts as 4) whose shared memory leaves two blocks per multiprocessor; tile 2 when none does
    {
        const std::size_t v100 = 98304, res = 0;   // Tesla V100: 96 KB per multiprocessor, no reservation
        // IQ3_S / IQ4_NL (bytes of the bench's I8-LAUNCH lines): tiles 2 / 3 / 4 = 43,008 / 48,128 / 53,248
        CHECK(i8_tile_two_blocks(8, 43008, 48128, 53248, v100, res) == 3);
        CHECK(i8_tile_two_blocks(4, 43008, 48128, 53248, v100, res) == 3);
        CHECK(i8_tile_two_blocks(3, 43008, 48128, 53248, v100, res) == 3);
        CHECK(i8_tile_two_blocks(2, 43008, 48128, 53248, v100, res) == 2);
        // IQ2_S with its grid in shared memory (49,152 / 54,272 / 59,392): no tile gives two blocks -> tile 2; with the global grid (40,960 / 46,080 / 51,200) -> tile 3
        CHECK(i8_tile_two_blocks(8, 49152, 54272, 59392, v100, res) == 2);
        CHECK(i8_tile_two_blocks(8, 40960, 46080, 51200, v100, res) == 3);
        // a reservation per block counts against the fit: 2 x (48,128 + 1,024) is exactly 96 KB (fits), 2 x (48,128 + 1,025) is not
        CHECK(i8_tile_two_blocks(3, 43008, 48128, 53248, v100, 1024) == 3);
        CHECK(i8_tile_two_blocks(3, 43008, 48128, 53248, v100, 1025) == 2);
        // a bigger multiprocessor keeps the larger tile (never above 4: tile 8 is not this rule's pick)
        CHECK(i8_tile_two_blocks(8, 43008, 48128, 53248, 233472, 1024) == 4);
        CHECK(i8_volta_rule_on(-1) && i8_volta_rule_on(1) && !i8_volta_rule_on(0));
        // the tensor-core form (tile 5) from kI8hMinAvg entries an expert, only for the formats it covers
        CHECK(i8_tile_volta(3, kI8hMinAvg, true) == 5);
        CHECK(i8_tile_volta(3, kI8hMinAvg - 1, true) == 3);
        CHECK(i8_tile_volta(3, 508, false) == 3);
        CHECK(i8_tile_volta(2, 40, true) == 2);
    }
    if (failed) {
        std::fprintf(stderr, "i8_expert_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("i8_expert_rule_test passed\n");
    return 0;
}
