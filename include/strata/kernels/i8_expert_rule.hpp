// include/strata/kernels/i8_expert_rule.hpp - when a prompt layer's expert groups take our own int8 dot-product kernels (native_expert_i8,
// src/kernels/cuda/i8_expert.cuh) in place of MMQ.
//
// STRATA_EXPERT_I8 = 1 asks for them on any card, 0 turns them off, unset (or empty) leaves the card class to decide: on the HIP cards gfx1030 / gfx1031 /
// gfx1032 (RDNA2: no matrix cores, the byte dot product `v_dot4_i32_i8` only) the covered formats take them by default.  Measured on the RX 6700 XT
// (gfx1031, one MoE layer of Flash-Next's shape, tokens 2,048 / 8,192 / 20,000): 1.10 - 1.35 times MMQ for the IQ2_S / IQ2_XXS packs with a Q2_0 down and
// for the IQ3_S / IQ3_XXS packs with an IQ4_NL down (par/RESULTS-OWN-I8.md).  gfx1034 / 1035 / 1036 were not compiled or timed: they stay on MMQ.
//
// Both kinds of call keep the size floor: a launch of experts averaging fewer than kI8MinGroupEntries entries stays on today's path (no timing below
// 2,048 tokens exists on this class; the P100 / V100 numbers say the grouped / MMQ kernels win at 300 tokens).  Pure functions: no GPU and no
// environment read here (the caller passes the variable's text and the device's gcnArchName), so the rule is unit-tested without a card.
#pragma once

#include <cstdint>

#include "strata/kernels/gfx_arch.hpp"

namespace strata::kernels {

/// Groups (a launch of up to 16 experts) averaging fewer entries than this keep today's path.
inline constexpr int64_t kI8MinGroupEntries = 256;

/// The RDNA2 parts that take the int8 kernels by default.  `gcn` is hipDeviceProp_t::gcnArchName (a CUDA card has none: nullptr or "").
inline bool gfx_arch_is_gfx103x_i8(const char* gcn) {
    return gfx_arch_is(gcn, "gfx1030") || gfx_arch_is(gcn, "gfx1031") || gfx_arch_is(gcn, "gfx1032");
}

/// STRATA_EXPERT_I8 read as a state: 1 = on (any non-empty text that does not start with '0'), 0 = off (starts with '0'), -1 = not set (null or empty).
inline int i8_switch_parse(const char* text) {
    return text == nullptr || text[0] == '\0' ? -1 : (text[0] == '0' ? 0 : 1);
}

/// Does the card class alone (switch not set) take the int8 kernels?
inline bool i8_expert_class_default(int state, const char* gcn) { return state == -1 && gfx_arch_is_gfx103x_i8(gcn); }

/// Do the expert groups of a layer take `native_expert_i8`?  `state` = i8_switch_parse(getenv), `gcn` the device's gcnArchName, `group_entries` the entries
/// of one launch on average, `supported` = native_expert_i8_supported(layout) (formats, sizes, the device's shared memory).
inline bool i8_expert_takes(int state, const char* gcn, int64_t group_entries, bool supported) {
    if (state == 0 || !supported || group_entries < kI8MinGroupEntries) return false;
    return state == 1 || i8_expert_class_default(state, gcn);
}

// The few-rows route.  A layer whose experts average fewer routed rows than a crossing runs on the grouped kernels (native_expert_grouped, no shared-memory
// staging: one launch pair per 16 experts like the int8 kernel's, the q8_1 image of the rows made by quantize_q8_1_rows); from the crossing up it runs on the
// int8 kernel.  Measured on one MoE layer of Flash-Next's shape (average rows per expert where both take equal time), WITHOUT the 8-entry slices (N8): Tesla P100 (sm_60) 8.4, A4000 10.2 - 11.7,
// RX 6700 XT 10.2 - 10.8, R9700 11.7 (par/RESULTS-I8-RESEARCH.md); WITH them (STRATA_I8_N8, on by default in this range, see i8_n8_takes) P100 5.7 - 5.8, A4000 6.9 - 7.4, R9700 6.9 - 7.4 (par/RESULTS-I8-MMQOFF.md).  The limits below are the crossings with N8; the int8 side never loses more than 5 %.
// Without the route the int8 kernel keeps the size floor above (256 entries per group of 16 experts = 16 rows per expert) and the layers under it stay on today's path.
inline constexpr int kI8FewRowsDefault = 7;    ///< every card but the Tesla P100 (10 with STRATA_I8_N8=0)
inline constexpr int kI8FewRowsSm60 = 6;       ///< compute capability 6.0 (the P100): its crossing is lower (8 with STRATA_I8_N8=0)

/// STRATA_I8_FEWROWS read as the limit: not set (or empty, or text that is no number) = the card class's own `class_default`, 0 = no route, N = a limit of N rows per expert.
inline int i8_fewrows_limit(const char* text, int class_default) {
    if (text == nullptr || text[0] == '\0') return class_default;
    if (text[0] < '0' || text[0] > '9') return class_default;
    long v = 0;
    for (const char* p = text; *p >= '0' && *p <= '9'; ++p) {
        v = v * 10 + (*p - '0');
        if (v > 100000) break;
    }
    return (int) v;
}

/// The class's default limit: the P100 (compute capability 6.0) 6, every other card 7.
inline int i8_fewrows_class_default(int cc_major, int cc_minor) { return cc_major == 6 && cc_minor == 0 ? kI8FewRowsSm60 : kI8FewRowsDefault; }

/// Does a layer with `rows` routed rows over `experts` experts (those with at least one row) take the grouped kernels?  limit 0 = the route is off.
inline bool i8_fewrows_takes(int limit, int64_t rows, int64_t experts) { return limit > 0 && experts > 0 && rows < (int64_t) limit * experts; }

// The 8-entry slices of the tile-2..4 int8 kernel (N8: two weight rows and one entry of each 8-entry slice per thread; bit-identical output).  Measured on the A4000, the P100 and
// the R9700 (par/RESULTS-I8-MMQOFF.md, I8 column of the expert-paths bench, Flash-Next's shape, IQ2_XXS / Q2_0 and IQ3_S / IQ4_NL): tile 2 (up to 39 entries per expert on average, about
// 1,950 tokens) 23 - 33 % faster at 100 - 300 tokens, 12 - 17 % at 1,000 and 6 - 13 % at 1,500 on all three; tile 3 (2,048 tokens) 3 - 6 % faster on the A4000 and the P100 but
// 1.6 - 2.9 % SLOWER on the R9700.  So the default is tile 2 only; STRATA_I8_N8=1 asks for tiles 2 - 4, STRATA_I8_N8=0 for none.
/// STRATA_I8_N8 read as a state (1 / 0 / -1 not set), the tile code the call settled on (2, 3, 4 = rows of 16 entries per thread; 8 = the 128-entry kernel, which has no 8-entry form).
inline bool i8_n8_takes(int state, int tile) {
    if (state == 0) return false;
    return state == 1 ? tile >= 2 && tile <= 4 : tile == 2;
}

}  // namespace strata::kernels
