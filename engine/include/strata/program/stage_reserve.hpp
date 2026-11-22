// include/strata/program/stage_reserve.hpp - what the layer split keeps free on a card when it sizes the expert cache.
//
// This is the arithmetic of `stage_room` in src/program/generate.cpp, moved here unchanged so a host test can pin it
// (tests/program/stage_reserve_test.cpp). `reserve_mib` is exactly the sum that function subtracted from the card's
// free memory before; the default placement does not change.
//
// #3702: on an unequal pair (RTX 3060 + RTX 5060 Ti) the last card was left with 68 MiB after loading although
// 700 + 96 MiB were reserved for it, and the verify windows' graphs (captured on first use, 4-6 MiB each and once 42 MiB)
// then ran it out of memory. A larger window term was proposed for it; the proposal is not part of the default placement
// and lives in research/stage-reserve-proposal/ (nothing in the engine calls it).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program::stage_reserve {

constexpr int64_t kWindowMib = 96;       ///< the verify windows on a later stage; 75 MiB measured, rounded up
constexpr int64_t kDrafterMib = 1000;    ///< the MTP drafter (Q2_0: 839 MiB) + the head, on the last stage only
constexpr int64_t kDf2BindMib = 48;           ///< DFlash2's draft logits, bound on its card after the expert cache
constexpr int64_t kDf2ServeStagingMib = 112;  ///< serve: DFlash2's prompt staging, bound there after the cache too

/// What DFlash2 binds on its card after the expert cache, beside its own draft head (generate.cpp's `df2_after` less
/// `df2_head_bytes()`), in MiB.  The cache step takes it out of the last stage's room, so the split search plans it too.
inline int64_t df2_after_mib(bool serve) { return kDf2BindMib + (serve ? kDf2ServeStagingMib : 0); }

/// DFlash2's drafter and the head on the last stage, over kDrafterMib's MTP figure.  2x 8 GB, 0.1.63, serve: CUDA1 went from
/// "holds its weights" to the cache step 7.30 -> 6.21 GiB at K=47 and 7.23 -> 6.13 GiB at K=46, ~1100 MiB with the session
/// (~10 MiB a layer) taken out ("strata dflash2: ... drafter 765 MiB of VRAM" + the head): ~100 MiB over, rounded up.
constexpr int64_t kDf2DrafterOverMib = 112;

/// What the split search books for DFlash2 on the drafter's stage: its drafter over kDrafterMib, and what it binds after the
/// cache (df2_after_mib).  The cache step measures the loaded drafter, so it subtracts only df2_after_mib.
inline int64_t df2_search_mib(bool serve) { return kDf2DrafterOverMib + df2_after_mib(serve); }

struct Inputs {
    int64_t base_reserve_mib = 0;   ///< --vram-reserve-later-mib on a later stage when given, else --vram-reserve-mib
    int64_t prompt_buffers_mib = 0; ///< the prompt path's own buffers (0 when they are borrowed from the cache)
    bool later = false;             ///< a stage after the first
    bool drafter = false;           ///< the stage that carries the MTP drafter and the head
    int64_t drafter_mib = kDrafterMib; ///< VRAM reserved when `drafter` is true (larger for Q4_0 / IQ4_NL MTP packs)
    int64_t df2_bind_mib = 0;       ///< the drafter's stage with DFlash2 on it: df2_search_mib (0: no DFlash2 there)
    int64_t pipe_window_mib = 0;    ///< --pipeline-windows: a second verifier on every stage
    int64_t pipe_snap_mib = 0;      ///< --pipeline-windows 2 or more: the first stage's second GDN state
};

/// What the plan keeps free on the card, in MiB (generate.cpp's `reserve` before the shift).
inline int64_t reserve_mib(const Inputs& in) {
    return in.base_reserve_mib + in.prompt_buffers_mib + (in.later ? kWindowMib : in.pipe_snap_mib) + in.pipe_window_mib +
           (in.drafter ? in.drafter_mib + in.df2_bind_mib : 0);
}

/// The room the cache may use: the free bytes less the reserve, never below 0.
inline int64_t room_bytes(int64_t free_bytes, const Inputs& in) {
    return std::max<int64_t>(free_bytes - (reserve_mib(in) << 20), 0);
}

}  // namespace strata::program::stage_reserve
