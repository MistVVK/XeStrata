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

#include <cstddef>
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

/// A build without llama.cpp's MMQ (-DSTRATA_MMQ=OFF, or HIP without STRATA_PREFILL_MMQ; prefill.cpp's STRATA_PF_OWN_ONLY): the int8 kernels are the default prompt-expert kernels on every
/// card that covers the layer's formats, because MMQ is what they replace; without them the layer runs each expert on the FP16 path (4 - 5 times slower on an RX 6700 XT).  `state` is
/// i8_switch_parse's answer; a value the user set (0 or 1) is kept, an unset switch reads as 1 in such a build.  "Default on" is not "forced": see i8_order_allows.
inline int i8_switch_effective(int state, bool mmq_free) { return state == -1 && mmq_free ? 1 : state; }

/// Does the build / card class alone (switch not set) take the int8 kernels: the RDNA2 class, or any card in a build without MMQ.
inline bool i8_expert_class_default(int state, const char* gcn, bool mmq_free = false) { return state == -1 && (mmq_free || gfx_arch_is_gfx103x_i8(gcn)); }

/// The order of the prompt-expert paths for one layer of one chunk: the fused kernels (a long chunk on a layer they cover) and the staged group entry (a short chunk, the fused kernels'
/// entry) come first, the int8 kernels take what is left, the FP16 path what even they do not cover.  This holds for a value the user set too (STRATA_EXPERT_I8=1 asks for the int8 kernels
/// where nothing faster takes the layer: R9700, 2,048 tokens, fused 1,144.6 ms against int8 1,624.5); it is the order prefill.cpp's `i8_l` condition has.
inline bool i8_order_allows(bool fused_takes, bool staged_takes) { return !fused_takes && !staged_takes; }

inline bool i8_expert_takes(int state, const char* gcn, int64_t group_entries, bool supported);   // (defined below)
/// The whole question for one layer of one chunk, as the pure rule (the engine asks the two halves where it has them: i8_order_allows in prefill.cpp, i8_expert_takes in iq_kernels.cu).
inline bool i8_layer_takes(int raw_state, bool mmq_free, const char* gcn, int64_t group_entries, bool supported, bool fused_takes, bool staged_takes) {
    return i8_order_allows(fused_takes, staged_takes) && i8_expert_takes(i8_switch_effective(raw_state, mmq_free), gcn, group_entries, supported);
}

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

// Volta (compute capability 7.0: 96 KB of shared memory per multiprocessor, no int8 matrix cores, so every routed expert of a prompt runs on these kernels).  The tile the
// other cards take from 96 entries a group on average (tile 8, 128 x 128, 54-62 KB) and tile 4 (51-59 KB) leave one block of 8 warps per multiprocessor there; tiles 2 and 3 of
// the IQ3 formats (43 / 48 KB) run two.  Tesla V100-SXM2 32 GB, one layer of Flash-Next's shape, uniform routing, MMQ time / int8 time (above 1: ours faster), the call's
// own pick against tile 3 (phaseA-tests/tune0216, bench1-v100): IQ3_S / IQ4_NL 0.745 -> 0.949 at 8,192 tokens and 0.795 -> 0.820 at 26,000, IQ3_XXS / IQ4_NL 0.721 -> 0.955 and
// 0.795 -> 0.831; output bits equal at every tile (the tile only groups entries).  The big-grid formats (IQ2_S's 8 KB grid table) read their grid from global memory there
// (I8Smem GG: the same words, the same bits), so their tile 3 fits two blocks too.  So on Volta the call's pick is lowered to the largest tile that still gives two blocks per
// multiprocessor (tile 8 counts as 4 first); a tile set by STRATA_I8_TM is kept as given.
/// `tm` the pick (2, 3, 4 or 8), `bytes2/3/4` the dynamic shared memory of tiles 2-4 for the layer's formats, `smem_sm` the multiprocessor's shared memory, `reserved` the
/// driver's reservation per block (cudaDevAttrReservedSharedMemoryPerBlock).
inline int i8_tile_two_blocks(int tm, std::size_t bytes2, std::size_t bytes3, std::size_t bytes4, std::size_t smem_sm, std::size_t reserved) {
    int t = tm >= 4 ? 4 : tm <= 2 ? 2 : 3;
    auto fits = [&](int k) { const std::size_t b = (k == 2 ? bytes2 : k == 3 ? bytes3 : bytes4) + reserved; return 2 * b <= smem_sm; };
    while (t > 2 && !fits(t)) --t;
    return t;
}
/// Volta's tensor-core form of the kernels (tile code 5, src/kernels/cuda/i8h_volta.cuh: the same products on the FP16 tensor cores, bit-identical output) from this many
/// entries a launch's expert has on average.  Tesla V100, one layer of Flash-Next's shape, uniform routing, MMQ time / time (phaseA-tests/tune0216 bench5), tile 3 -> tile 5:
/// at 40 entries (2,048 tokens) 1.38 -> 1.22 (slower), at 160 (8,192) 0.95 -> 0.96 (IQ3_S), 0.95 -> 0.94 (IQ3_XXS), 0.90 -> 0.94 (IQ2_S), at 508 (26,000) 0.82 -> 0.86,
/// 0.83 -> 0.84, 0.82 -> 0.89; output hashes equal at every size.
inline constexpr int64_t kI8hMinAvg = 160;
/// The Volta pick: tile 5 from kI8hMinAvg entries on average when the formats have it (`h_ok`), else the two-block tile of i8_tile_two_blocks.
inline int i8_tile_volta(int two_block_tile, int64_t avg, bool h_ok) { return h_ok && avg >= kI8hMinAvg ? 5 : two_block_tile; }
/// STRATA_I8_SM70 read as a state (1 / 0 / -1 not set): the Volta tile rule above and the global grid are on unless it is 0.
inline bool i8_volta_rule_on(int state) { return state != 0; }

// The Tesla P100 (compute capability 6.0) in a build without MMQ.  There the build WITH MMQ takes the FP16 prompt path by default (prefill.cpp gp100_fp16_default: GP100 has no
// dp4a), so the unset-switch int8 default of a build without MMQ (i8_switch_effective) must not replace it where the int8 kernel loses to it.  Measured on the P100 (OUT build,
// prompt read ms int8 / FP16, par/RESULTS-P100-GATE.md): 300 0.874, 600 0.865, 1,000 0.844, then 1,200 1.196, 1,500 1.142, 2,048 1.085, 8,192 1.081: the step sits at the chunk size from
// which every expert streams (stream_all_min, 1,024 tokens, prefill.cpp), so the limit is that floor and no constant of its own: the int8 default applies to chunks BELOW it
// (the staged per-layer path), the FP16 path from it up, as the build with MMQ does.  A value the user set wins at every size (STRATA_EXPERT_I8=1 keeps the int8 kernels, =0 turns
// them off); every other card class has no such limit.

/// May the int8 default (switch NOT set, build without MMQ) take a chunk of `chunk_tokens` tokens on this device?  `raw_state` = i8_switch_parse(getenv): a set value (0 or 1) is
/// not this rule's to decide (true); a build with MMQ, or a card that is not the P100 (`sm60`), has no limit (true); otherwise the chunk must be under `limit`.
inline bool i8_chunk_default_allows(int raw_state, bool mmq_free, bool sm60, int64_t chunk_tokens, int64_t limit) {
    if (raw_state != -1 || !mmq_free || !sm60) return true;
    return chunk_tokens < limit;
}

}  // namespace strata::kernels
