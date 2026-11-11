// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Optional SAVE admission: free reconstructible host caches, never the live device state.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

namespace strata::core {

struct SessionSaveReclaimed {
    size_t checkpoints = 0, parked = 0, bytes = 0;
};

// Release each class of reconstructible cache in order, probing after each release.
template<class Short, class Reuse, class Parked, class Checkpoint>
void session_reclaim_until(Short short_of, Reuse reuse, Parked parked, Checkpoint checkpoint) {
    if (!short_of()) return;
    if (reuse() && !short_of()) return;
    while (parked()) if (!short_of()) return;
    while (checkpoint()) if (!short_of()) return;
}

// Called BEFORE session_save_checkpoints selects a pointer into chain. Probe after each
// eviction: freed capacity is not necessarily physical RAM returned by the allocator.
// No allocation is needed here. Unknown telemetry / sizes cannot justify cache eviction.
// Preserve every explicit pin and exactly the checkpoint the file would have selected.
template<class Probe>
SessionSaveReclaimed session_save_reclaim(std::vector<ConversationCheckpoint>& chain,
                                        ConversationCache& cache, const SessionSaveLive& live,
                                        uint64_t floor, Probe available) {
    SessionSaveReclaimed out;
    const auto* selected = session_deepest_checkpoint(chain);
    const uint64_t need = session_save_peak_bytes(selected, live);
    size_t keep = selected ? size_t(selected - chain.data()) : chain.size();
    auto short_of = [&] {
        const auto room = available();
        return room && need != UINT64_MAX && !conversation_memory_admit(room, need, floor);
    };
    auto reuse = [&] {
        if (!cache.retained_bytes()) return false;
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        return true;
    };
    auto parked = [&] {
        const size_t before = cache.bytes();
        if (!cache.evict_oldest()) return false;
        ++out.parked;
        out.bytes += before - cache.bytes();
        return true;
    };
    size_t next = chain.size();
    auto checkpoint = [&] {
        while (next > 0) {
            const size_t i = --next;
            if (i == keep || chain[i].pinned) continue;
            out.bytes += chain[i].bytes();
            chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i));
            if (i < keep) --keep;
            ++out.checkpoints;
            return true;
        }
        return false;
    };
    session_reclaim_until(short_of, reuse, parked, checkpoint);
    return out;
}

} // namespace strata::core
