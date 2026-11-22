// metal/tests/moe_tile_grid_test.cpp - the pairing the prompt paths' routed-expert GEMMs depend on: the tile the
// dispatch picks and the grid x it is dispatched over.
//
// moe_mm.metal's kernel writes TN output columns of HS per threadgroup (f0 = tg.x * TN), so the launch has to cover
// NFF / TN threadgroups along x, and the down GEMM N / TN.  mm_scan publishes both grids for one layer at a time
// (meta[2], meta[5]); they come in the plan, and src/moe_tiles.h is where a tile and its TN are decided together.
//
// It is a host test on purpose, and it links nothing: no Metal device, no model, no ref_quant.cpp (whose fp16
// helpers need a compiler that has _Float16).  It runs anywhere the tree is checked out.
//
// The v2 K-quants' tile is the case that matters - it writes 16 columns where every other tile writes 64, so a grid
// of 640 / 64 covers rows 0..159 of every expert's gate/up output and leaves 160..639 as the previous layer left
// them, and the down GEMM reads that.  No error, no warning, wrong tokens.
//
//   moe_tile_grid_test
#include "moe_tiles.h"

#include <cstdio>

namespace {

constexpr int NFF = 640, N = 2560;   // the engine's ffn width and n_embd (window.cpp)

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

// a tile name is TM_TN_TK_SG, e.g. "_16_16_128_2".  Hand-parsed: no CRT, so the test needs nothing but the header.
bool parse_tile_name(const char* s, int out[4]) {
    int n = 0;
    while (*s) {
        if (*s++ != '_') return false;
        if (*s < '0' || *s > '9') return false;
        int v = 0;
        while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
        if (n == 4) return false;
        out[n++] = v;
    }
    return n == 4 && out[0] > 0 && out[1] > 0 && out[2] > 0 && out[3] > 0;
}

// the name a launch is built from and the TN its grid divides by are two spellings of the same number, so every
// tile's name is parsed back and checked against the TN it declares
bool name_matches_tn(const strata::metal_engine::MoeTile& t) {
    int f[4];
    return parse_tile_name(t.name, f) && f[1] == t.tn;
}

}  // namespace

int main() {
    using namespace strata::metal_engine;

    // every type a routed expert can carry in this tree (ref_quant.h's ids), with the name the tree gives it
    const struct { int t; const char* n; } types[] = {
        {ref::Q2_0, "Q2_0"},       {ref::IQ3_XXS, "IQ3_XXS"}, {ref::IQ2_XXS, "IQ2_XXS"}, {ref::IQ2_XS, "IQ2_XS"},
        {ref::IQ2_S, "IQ2_S"},     {ref::IQ3_S, "IQ3_S"},     {ref::IQ4_NL, "IQ4_NL"},   {ref::IQ4_XS, "IQ4_XS"},
        {ref::Q4_K, "Q4_K"},       {ref::Q5_K, "Q5_K"},       {ref::Q6_K, "Q6_K"},       {ref::Q8_0, "Q8_0"},
        {ref::Q5_1, "Q5_1"},       {ref::Q5_0, "Q5_0"},       {ref::BF16, "BF16"},       {ref::F32, "F32"},
        {ref::Q3_K, "Q3_K"},       {ref::Q2_K, "Q2_K"},       {ref::NVFP4, "NVFP4"},
    };

    for (const auto& e : types) {
        const MoeTile gt = gu2_tile(e.t), dt = dn2_tile(e.t);
        char what[160];
        std::snprintf(what, sizeof(what), "%s: gu2 %s TN %d, dn2 %s TN %d, and both names parse back",
                      e.n, gt.name, gt.tn, dt.name, dt.tn);
        check(name_matches_tn(gt) && name_matches_tn(dt), what);
    }

    // the TN has to divide the output width, or the launch cannot cover it exactly: one threadgroup too many writes
    // TN rows past the end of HS (NFF rows), one too few leaves the tail of every expert's output unwritten
    for (const auto& e : types) {
        const MoeTile gt = gu2_tile(e.t), dt = dn2_tile(e.t);
        char what[160];
        std::snprintf(what, sizeof(what), "%s: NFF %d %% gu TN %d == 0, N %d %% dn TN %d == 0",
                      e.n, NFF, gt.tn, N, dt.tn);
        check(NFF % gt.tn == 0 && N % dt.tn == 0, what);
    }

    check(gu2_tile(ref::Q4_K).tn == 16 && gu2_tile(ref::Q5_K).tn == 16 && gu2_tile(ref::Q6_K).tn == 16,
          "a K-quant's v2 gate/up tile writes 16 columns, not 64");
    check(moe_grid_x(NFF, gu2_tile(ref::Q4_K).tn) == 40 && moe_grid_x(NFF, gu2_tile(ref::Q5_K).tn) == 40 &&
              moe_grid_x(NFF, gu2_tile(ref::Q6_K).tn) == 40,
          "a K-quant layer's v2 gate/up grid is 640 / 16 = 40");
    check(moe_grid_x(NFF, gu2_tile(ref::Q6_K).tn) * gu2_tile(ref::Q6_K).tn == NFF,
          "that grid covers the whole ffn width, so HS has no rows the layer never wrote");
    check(NFF / 64 == 10 && moe_grid_x(NFF, gu2_tile(ref::Q4_K).tn) != NFF / 64,
          "the constant 640 / 64 is not a K-quant layer's grid (the defect this pairs away)");
    check(moe_grid_x(NFF, gu2_tile(ref::Q8_0).tn) == 10 && moe_grid_x(NFF, gu2_tile(ref::IQ4_XS).tn) == 10 &&
              moe_grid_x(NFF, gu2_tile(ref::Q2_0).tn) == 10 && moe_grid_x(NFF, gu2_tile(ref::Q5_1).tn) == 10,
          "every other v2 gate/up grid stays 640 / 64 = 10");
    check(moe_grid_x(N, dn2_tile(ref::IQ4_NL).tn) == 40 && moe_grid_x(N, dn2_tile(ref::Q5_0).tn) == 40 &&
              moe_grid_x(N, dn2_tile(ref::Q3_K).tn) == 40 && moe_grid_x(N, dn2_tile(ref::Q2_0).tn) == 40,
          "the other v2 down grids are 2560 / 64 = 40");
    // the K-quant down tile (Q4-KQ, 2bc8c6692) writes 16 columns too: its grid is 2560 / 16, where the constant gave 40
    check(moe_grid_x(N, dn2_tile(ref::Q4_K).tn) == 160 && moe_grid_x(N, dn2_tile(ref::Q6_K).tn) == 160 &&
              moe_grid_x(N, dn2_tile(ref::Q4_K).tn) * dn2_tile(ref::Q4_K).tn == N,
          "a K-quant layer's v2 down grid is 2560 / 16 = 160 and covers n_embd");

    // the v3 (M5) tiles the engine's window dispatches: gate/up TN 32 over SG 4 simdgroups, down TN 64 over DSG 2
    check(name_matches_tn(MoeTile{"_16_32_32_4", GU3_TN}) && name_matches_tn(MoeTile{"_16_64_32_2", DN3_TN}),
          "the v3 tiles' names parse back to their TN");
    check(moe_grid_x(NFF, GU3_TN, GU3_SG) == 5 && NFF % (GU3_TN * GU3_SG) == 0,
          "a v3 gate/up grid is 640 / (32 * 4) = 5 and covers the ffn width");
    check(moe_grid_x(N, DN3_TN, DN3_SG) == 20 && N % (DN3_TN * DN3_SG) == 0,
          "a v3 down grid is 2560 / (64 * 2) = 20 and covers n_embd");

    std::printf("\n%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
