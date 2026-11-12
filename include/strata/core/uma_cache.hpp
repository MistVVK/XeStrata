// include/strata/core/uma_cache.hpp - sizing the GPU expert cache on a device whose memory IS the system RAM.
//
// On a Strix Halo (or any APU) "free VRAM" is the GPU-addressable pool: the BIOS carve-out plus the GTT, up to nearly all of the RAM when
// the kernel was told ttm.pages_limit (docs/STRIX_HALO.md recommends 112 GiB of a 128 GiB box).  The expert arena (every expert, in RAM) is
// already counted in what the OS has used, but the cache copies experts INTO that same RAM, so a cache sized from the pool alone can ask for
// more bytes than the OS has left - the OOM killer then takes the engine, or the desktop.  The auto cache is therefore sized from
// min(what the GPU says is free, the RAM the OS says is available - a reserve for the desktop).  Only the AUTO size: --expert-cache N is
// kept as asked (the engine warns).  STRATA_UMA_RESERVE_GIB=N sets the reserve (0 = no reserve: the old sizing).
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// The RAM an integrated GPU's expert cache leaves to the OS and the desktop: STRATA_UMA_RESERVE_GIB when set (0 = none), else the larger of
/// 8 GiB and 8% of the installed RAM (a desktop session with a browser needs ~6 GiB; 8% is ~10 GiB on 128 GB).
inline uint64_t uma_desktop_reserve_bytes(uint64_t total_ram, const char* env_gib = std::getenv("STRATA_UMA_RESERVE_GIB")) {
    constexpr uint64_t GiB = 1ull << 30;
    if (env_gib != nullptr && env_gib[0] != '\0') {
        char* end = nullptr;
        const double v = std::strtod(env_gib, &end);
        if (end != env_gib && v >= 0.0) return (uint64_t) (v * (double) GiB);
    }
    const uint64_t pct = total_ram / 100 * 8;
    return pct > 8 * GiB ? pct : 8 * GiB;
}

/// The bytes the auto cache may use: the RAM available now, less the reserve (0 when that is already gone).
inline uint64_t uma_cache_room_bytes(uint64_t available, uint64_t reserve) {
    return available > reserve ? available - reserve : 0;
}

/// G16 (Q4-P8, 2026-11-08): the reserve the auto cache keeps free on Windows HIP when none was given.  A DISCRETE card of 15 GiB or more gets the floor (#1376: a
/// 20 GB card ended with 323 MiB free at the default 700 and Windows paged); an INTEGRATED GPU (the device property `integrated`, the same test the UMA sizing above
/// uses) does not: its pool is the RAM, the room the floor frees is taken straight back by the RAM budget's page-locked complement, and on a Strix Halo with
/// UD-Q4_K_XL the prompt path's own buffers then found 0 MiB and the engine stepped its chunk down for ever.  `current_mib` is what the engine would use without
/// the floor (the default 700); a reserve given on the command line never comes here.
inline int wddm_auto_reserve_mib(bool integrated, uint64_t total_bytes, int current_mib, int floor_mib = 2560) {
    constexpr uint64_t GiB = 1ull << 30;
    if (integrated || total_bytes < 15 * GiB || current_mib >= floor_mib) return current_mib;
    return floor_mib;
}

/// BATCH SLOTS ON THE STRIX HALO (Windows HIP, integrated GPU): the page-locked memory of a start is the RAM copy of experts (the pinned complement) plus the streamed
/// K/V that every session pins for the whole --max-context (1.55 GiB a session at 131,072; the main session and every slot).  UD-Q4_K_XL, 131,072 tokens, 4 slots,
/// head 4e1e5f248, complement + pinned K/V in GiB: 20.33 + 7.75 = 28.1 (job strix-20261112-1836-e6f0 / -1854-a210 / -1903-afc9) and 18.00 + 7.75 = 25.75
/// (-1923-8a6a: the prompt path's 16,384 chunk did not fit, the 6,144 chunk did, then "the OS gave no more page-locked memory" at the verify window) FAILED;
/// 15.00 + 7.75 = 22.75 (-1937-7dbd) and 8.00 + 7.75 = 15.75 (-1928-0caf) STARTED and served with the 16,384 chunk; three slots at 16.40 + 6.20 = 22.60
/// (-1834-6629) started.  So the RAM copy is capped to what leaves complement + pinned K/V at the level of the largest start that worked, 22.75 GiB (the
/// largest budget that started, 15 GiB, plus 4 slots' K/V).  One bracket (15 works, 18 fails), not a derived limit.  The experts left out are read from the SSD.
inline uint64_t uma_pinned_complement_cap_bytes(uint64_t kv_pinned_bytes) {
    constexpr uint64_t level = (1ull << 30) * 2275 / 100;   // 22.75 GiB
    return level > kv_pinned_bytes ? level - kv_pinned_bytes : 0;   // 0 = keep no expert in RAM: every expert outside the GPU cache is read from the SSD (a start, not a refusal)
}

/// The cap as the budget that pin_cache_complement takes: a budget of 0 there means "no limit", so "nothing in RAM" is the smallest real budget, 1 byte (no expert's blob
/// fits in it: the copy is empty and every expert the cache does not hold is read from the SSD).  Never an error, never a negative size.
inline uint64_t uma_pinned_complement_budget_bytes(uint64_t kv_pinned_bytes) {
    const uint64_t cap = uma_pinned_complement_cap_bytes(kv_pinned_bytes);
    return cap > 0 ? cap : 1;
}

}  // namespace strata::core
