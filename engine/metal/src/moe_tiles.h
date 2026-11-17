// metal/src/moe_tiles.h - the routed-expert GEMM tiles of the Metal prompt paths, and the TN each one writes its
// output columns in.
//
// The prompt GEMMs are dispatched indirectly: mm_scan plans one MoE layer's tiles and publishes the grids the
// gate/up and the down GEMM of that layer read (moe_mm.metal, meta[2..4] and meta[5..7]).  The tile the dispatch picks
// and the grid it is dispatched over therefore have to agree - the kernel writes TN output columns per threadgroup,
// so its grid x is the output width divided by TN.  They are returned together here so the two cannot drift apart.
//
// They did drift: the v2 gate/up grid was the constant NFF / 64 while gu2_tile() hands the K-quants a TN-16 tile, so on
// a Mac without the M5 right-input layout a Q4_K/Q5_K/Q6_K layer computed rows 0..159 of 640 and left the rest of HS
// as the previous layer left it, with no error and no check that fails.  The decode window never had the bug because
// it derives its grid from the tile it picks (window.cpp: NFF / (gs * (32 / gl) * gr)); this is the prompt paths'
// spelling of the same rule.
#pragma once

#include "ref_quant.h"

namespace strata::metal_engine {

// the kernel-name suffix of a routed GEMM tile, and the output columns one of its threadgroups writes
struct MoeTile { const char* name; int tn; };

// the v2 (no M5 right-input layout) expert tiles: the K-quants need TK 128, Q8_0 / IQ4_XS TK 32 (their 32-value
// units), the rest TK 16
inline MoeTile gu2_tile(int t) {
    if (t == ref::Q4_K || t == ref::Q5_K || t == ref::Q6_K) return {"_16_16_128_2", 16};   // the M4's best (513fc615)
    if (t == ref::Q8_0 || t == ref::IQ4_XS)                 return {"_16_64_32_2", 64};
    return {"_16_64_16_2", 64};
}
inline MoeTile dn2_tile(int t) {
    if (t == ref::Q4_K || t == ref::Q5_K || t == ref::Q6_K) return {"_16_16_128_2", 16};   // Q4-KQ: the K-quants' down experts need TK 128 as their gate/up do
    if (t == ref::Q3_K || t == ref::IQ4_XS) return {"_16_64_32_2", 64};
    return {t == ref::IQ4_NL || t == ref::Q5_1 || t == ref::Q5_0 || t == ref::Q8_0 ? "_16_64_32_2" : "_16_64_16_2", 64};
}

// the v3 (M5, COOP_K4) tiles: the gate/up GEMM covers SG simdgroups per threadgroup on top of TN columns, the down
// GEMM DSG
constexpr int GU3_TN = 32, GU3_SG = 4;    // mm_gu3_<type>_16_32_32_4
constexpr int DN3_TN = 64, DN3_SG = 2;    // mm_down3_<type>_16_64_32_2

// the indirect grid x of a routed GEMM: the output-column blocks the launch covers
inline int moe_grid_x(int n_out, int tn, int sg = 1) { return n_out / (tn * sg); }

}  // namespace strata::metal_engine
