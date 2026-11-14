// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/context_fit.hpp - the context a start would need, against what the card has free, and the
// largest one that fits (#5715).
//
// A card too small for the chosen context ended in "session state allocation failed" (the Qwen3.8 engine) or
// "plain model: no VRAM for the K/V cache (N cells)" (the plain engine): the person had the number of cells and
// nothing to do with it.  Upstream's answer (strata-q36's context_fit, Qwen3.6) is a sentence with the memory the
// context needs, the memory the card has free, and the context that fits in steps of 1,024 tokens; the engine never
// changes the context itself.
//
// Each engine passes its own figure: core::session_bytes for the Qwen3.8 engine, the plain engine's K/V bytes.  So
// the sentence follows the allocations the engine makes and not a table that can drift from them.
#pragma once

#include <cstdint>

namespace strata::core {

struct ContextFit {
    int64_t cells = 0;     ///< the largest context, a multiple of `step`, whose bytes fit `room`; 0: not even `step`
    uint64_t need = 0;     ///< bytes the context that was asked for needs
    uint64_t room = 0;     ///< what was free
};

/// The largest context (a multiple of `step` tokens) whose bytes fit `room`, from `bytes(cells)` (monotone in cells).
template <class Bytes>
ContextFit context_that_fits(int64_t max_cells, uint64_t room, Bytes&& bytes, int64_t step = 1024) {
    ContextFit out;
    out.room = room;
    out.need = max_cells > 0 ? (uint64_t) bytes(max_cells) : 0;
    if (max_cells <= 0 || step <= 0 || bytes(step) > room) return out;   // not even the shortest context
    int64_t lo = 1, hi = max_cells / step;                               // in steps
    while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if ((uint64_t) bytes(mid * step) <= room) lo = mid;
        else hi = mid - 1;
    }
    out.cells = lo * step;
    return out;
}

}  // namespace strata::core
