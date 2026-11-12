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

}  // namespace strata::kernels
