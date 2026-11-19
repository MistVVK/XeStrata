// strata/core/vram_request_notice.hpp - when the engine's "MiB of the card's memory are free at the start of this request" note is said.
//
// 0.1.62 and before said the old WARNING at the first request of EVERY start whenever less than the safe figure (1024 MiB on Windows,
// 512 elsewhere) was free.  A card the expert cache has filled on purpose reads low right after the load, so that was a note nobody
// could act on (item 17 of the 0.2.17 list, owner-approved wording in phaseA-tests/rel0217/TEXT-BATCH-0217.md item 4).
// Now the note is said only when there is news:
//   - the free memory is LOWER than the start-up line reported (something grew since: a verify window's graph, the batch layouts, another
//     program), by more than `kVramRequestFallToleranceMiB` (the figure moves by a few MiB with the display), or
//   - it is under the start-up line's 512 MiB (`kVramRequestFloorMiB`: the same line the start-up text calls little room).
// Below the safe figure only; at or above it nothing is said.  When the start-up figure is not known (-1: the serve start-up did not
// reach its line) the old rule stays: below the safe figure, say it.
// Header-only, no CUDA: tests/core/vram_request_notice_test.cpp.
#pragma once

#include <cstdint>

namespace strata::core {

constexpr int64_t kVramRequestFloorMiB = 512;
constexpr int64_t kVramRequestFallToleranceMiB = 64;

/// `free_mib`: free at the start of this request; `startup_free_mib`: free at the start-up line (-1: unknown); `safe_mib`: 1024 on
/// Windows (WDDM pages), 512 elsewhere.
constexpr bool vram_request_note_due(int64_t free_mib, int64_t startup_free_mib, int64_t safe_mib) {
    if (free_mib >= safe_mib) return false;
    if (startup_free_mib < 0) return true;
    return free_mib < kVramRequestFloorMiB || free_mib + kVramRequestFallToleranceMiB <= startup_free_mib;
}

/// The --vram-reserve-mib the note suggests: the current reserve plus what is missing to the safe figure, to the next 64 MiB.
constexpr int64_t vram_request_reserve_mib(int64_t reserve_mib, int64_t free_mib, int64_t safe_mib) {
    return (reserve_mib + (safe_mib - free_mib) + 63) / 64 * 64;
}

}  // namespace strata::core
