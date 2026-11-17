// include/strata/program/split_walk.hpp - the spill walk of a layer split: what each card gives up, in which order, when it is short of VRAM.
//
// The 0.2.10 walk (draft layer dropped, a loan that keeps 16 slots, the prompt path's own small buffers, the output head in pinned host memory, a smaller KV window)
// ran for ONE card.  A layer split went straight to "expert cache: no room" or "the GPU has too little free VRAM for the prompt path" (two 8 GB cards, 128K: 6 of 6
// starts failed).  The arithmetic of the layer split's version is here, pure, so a host test pins it (tests/program/split_walk_test.cpp); src/program/generate.cpp
// reads the cards and prints the lines.
//
// The steps, mildest first, each only when the one before did not leave the card enough:
//   1. trim: every card loads only the dense weights of ITS layers (--trim-stage-weights).  Without it each card holds the whole 3.5 GiB set (IQ3_S).  Bit-exact in
//      itself; the cache simply gets the room.  This is the step that matters on 8 GB cards.
//   2. the KV window down (32768 -> 20480 -> 8192 -> 4096 cells a layer; the rest streams from RAM), when no placement can start the prompt path.
//   3. the drafter dropped on the card that holds it (draft layer or DFlash2's pack and buffers): decode is slower, the start works.
//   4. the loan keeps 16 slots, and the prompt path's own small buffers (256..1024-token chunks), instead of 128 and 1024..8192.
//   5. the output head in pinned host memory, on the card that holds it.
//   6. fewer cards: a card with too little free memory to hold a share is left out of the split; one card left is the single-card start (with its own walk).
// The context and the number of requests are never touched.  A big card never reaches any of them: trim_needed() is false when the card has the room.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program::split_walk {

constexpr int64_t kContextMib = 1550;          ///< CUDA context, streams, scratch and tables of a card that holds the whole dense set (8 GB hog, IQ3_S: 1559 MiB measured)
constexpr int64_t kWorkingCacheSlots = 263;    ///< a cache the engine calls working (generate.cpp #496)
constexpr int64_t kPromptHeadroomMib = 1100;   ///< a 512-token chunk's own buffers (570 MiB) + the serve check's 512 MiB headroom
constexpr int64_t kMinCardMib = 3072;          ///< under this (free, after its context) a card cannot hold even a trimmed half share of the dense set (1.75 GiB) beside a session and a cache; above it the other steps (drafter, loan, head) still have to make it fit
constexpr int64_t kStageLateAllocMib = 256;    ///< what a stage allocates after its cache that nothing prices (the single card's kLateAllocMib, #4474)

/// What one card needs, from the model and the start (all MiB).
struct Model {
    int64_t dense_mib = 0;      ///< the whole dense set a card loads without the trim (pack arena + native projections)
    int64_t session_mib = 0;    ///< the session of one card's layers at the asked context and KV window
    int64_t head_mib = 0;       ///< the output head (the last stage only)
    int64_t drafter_mib = 0;    ///< the draft layer or DFlash2 (the stage that carries it only)
    int64_t slot_mib = 3;       ///< the largest expert blob, rounded up
    int64_t reserve_mib = 796;  ///< what the plan keeps free on a later stage (700 + the verify windows)
};

/// MiB a card needs beyond its free memory figure when it loads the whole dense set: the weights, the context, its session, (the last stage) the head and the drafter,
/// the reserve, a working cache and the prompt path's headroom.
inline int64_t untrimmed_need_mib(const Model& m, bool carries_head_and_drafter) {
    return m.dense_mib + kContextMib + m.session_mib + (carries_head_and_drafter ? m.head_mib + m.drafter_mib : 0) + m.reserve_mib +
           kWorkingCacheSlots * m.slot_mib + kPromptHeadroomMib;
}

/// Is this card too short to start without the trim?  (free_mib: its free memory before it loaded anything.)
inline bool trim_needed(int64_t free_mib, const Model& m, bool carries_head_and_drafter) {
    return free_mib > 0 && free_mib < untrimmed_need_mib(m, carries_head_and_drafter);
}

/// May this card take part in a layer split at all?
inline bool card_usable(int64_t free_mib) { return free_mib >= kMinCardMib; }

/// The single card's rule for dropping the draft layer, unchanged (generate.cpp, "SHORT OF VRAM: THE DRAFT LAYER IS THE LAST THING TO GO"): the layer, the head, a working
/// cache beside the prompt buffers (214 slots, 543 MiB) and what is allocated after it.  Used for the card that holds the drafter in a layer split too.
inline uint64_t draft_layer_need_bytes(uint64_t layer_bytes, uint64_t head_bytes) {
    return layer_bytes + head_bytes + (220ull << 20) + (1550ull << 20);
}
inline bool draft_layer_short(uint64_t free_bytes, uint64_t layer_bytes, uint64_t head_bytes) {
    return free_bytes > 0 && free_bytes < draft_layer_need_bytes(layer_bytes, head_bytes);
}

/// DFlash2 on a stage: its pack and its K/V ring and buffers (0.25 GiB), its draft head, and the same working cache and late allocations.
inline uint64_t dflash2_need_bytes(uint64_t pack_bytes, uint64_t head_bytes) {
    return pack_bytes + (250ull << 20) + head_bytes + (220ull << 20) + (1550ull << 20);
}

/// The next KV window down from `current` (0: the whole context in VRAM) for a context of `ctx` cells, or 0 when it is already the smallest (`min_cells`).
inline int64_t next_kv_window(int64_t current, int64_t ctx, int64_t min_cells) {
    constexpr int64_t steps[] = {20480, 8192, 4096};
    const int64_t cur = current > 0 ? current : ctx;
    for (const int64_t s : steps)
        if (s < cur && s < ctx && s >= min_cells) return s;
    return 0;
}

/// Can `n` cards hold the model with every card trimmed?  `room_mib[i]` is a card's free memory less its fixed costs (context, reserve, late allocations; NOT the head and the
/// drafter), `per_layer_mib` what a layer costs on a card (its dense weights and its session), `relief_mib` what the later steps can still give back (the drafter and the head
/// moved off).  False means even the whole walk does not fit: the split should run on fewer cards.
inline bool split_feasible(const int64_t* room_mib, int n, int64_t layers, int64_t per_layer_mib, int64_t relief_mib) {
    int64_t total = relief_mib;
    for (int i = 0; i < n; ++i) total += room_mib[i];
    return total >= per_layer_mib * layers;
}

/// The layer share of each of `n` cards, proportional to the room each has for experts and sessions once its fixed costs are paid (`room_mib[i]` = its free memory less the
/// context, the reserve and, for the card that carries them, the head and the drafter): that gives every layer the same cache bytes on every card.  At least 2 layers a card,
/// summing to `layers`.  Used when the walk chooses the split before any card has loaded anything, so CUDA0's weights can be trimmed too.
inline void proportional_layers(const int64_t* room_mib, int n, int64_t layers, int64_t* out) {
    int64_t total = 0;
    for (int i = 0; i < n; ++i) total += std::max<int64_t>(room_mib[i], 1);
    int64_t given = 0;
    for (int i = 0; i + 1 < n; ++i) {
        const int64_t w = std::max<int64_t>(room_mib[i], 1);
        out[i] = std::max<int64_t>(2, (layers * w + total / 2) / total);
        given += out[i];
    }
    out[n - 1] = layers - given;
}

}  // namespace strata::program::split_walk
