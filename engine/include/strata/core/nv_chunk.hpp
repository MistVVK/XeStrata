// include/strata/core/nv_chunk.hpp - the prompt-chunk ceiling a bare `--prefill auto` takes on NVIDIA cards, as pure functions so a test can pin them.
//
// Until 0.2.10 a bare auto stopped at 8192 on NVIDIA and 16384 / 32768 were opt-ins (#282).  Owner 11-03 ("after the full test suite passes on it"): the
// ceiling is 32768 for cards of compute capability 8.0 and up, the gfx12 rule (generate.cpp): a 20K prompt is one chunk instead of 8192 + 8192 + 3.6K, and
// every chunk of 1,024 tokens or more streams all the experts the card does not hold.  The fit scan still picks a smaller chunk when the cache cannot lend
// the buffers, and the #2090 RAM guard steps 32768 -> 16384 -> 8192 when the RAM available does not cover the extra loan.
// Round 0212, boston A6000 UD-Q4_K_XL (13.1K of 24.6K experts in VRAM): 20K +43.4%, 32K +22.5%, 64K +21.3%, 8K 0.0%; cards that hold every expert: flat.
// Volta (compute capability 7.0) joins them (lane 10, 11-10, one Tesla V100-SXM2 32 GB on a gen3 x8 link, 5.8 GB/s, UD-Q4_K_XL with DFlash2, 8,215 of 24,576
// experts in VRAM = 33%; 5 alternating pairs of 8192 against 32768, mean prompt tok/s): 4K +0.3%, 8K +0.2% (one chunk in both: the control), 20K 635.9 -> 1,488.6
// = +134.1% (range +132.8..+135.7), 32K 678.6 -> 1,433.0 = +111.2%; decode after the prompts unchanged (15-prompt median 44.8 against 45.4 tok/s); first-token KL
// against 8192 with the static-placement flags: mean 3.7e-5 (20K) and 4.0e-4 (32K), top-1 the same on 13 of 13 prompts.  Four cards of that box (a layer split,
// 89% resident) read a 20K prompt 27-30% and a 32K prompt 17-19% SLOWER with larger chunks, which is why a layer split keeps 8192.
// Turing (7.5) and Pascal (6.x) are not measured and keep 8192.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::core {

/// STRATA_NV_CHUNK: unset -> 32768; "16" -> 16384; "0" -> 0 (no NVIDIA ceiling: the engine's 8192); anything else (also "1") -> 32768.
/// `defaults026`: STRATA_DEFAULTS=026 gives back the 0.2.6 behaviour (8192).
inline int64_t nv_chunk_ceiling(const char* env, bool defaults026) noexcept {
    if (defaults026) return 0;
    if (env == nullptr) return 32768;
    if (std::strcmp(env, "16") == 0) return 16384;
    if (env[0] == '0') return 0;
    return 32768;
}

/// #5199: VRAM-aware ceiling for bare `--prefill auto` on NVIDIA cards.
/// On <= 16.5 GiB GPUs (8-16 GB consumer cards), raising bare auto to 16384/32768 lends ~3.5+ GB out of hot
/// expert slots and duplicates those lent slots in host RAM, regressing decode by 12-18% and stressing 64 GB PCs.
/// Keep bare auto at 8192 (return 0) on <= 16.5 GiB GPUs unless the operator explicitly sets STRATA_NV_CHUNK.
inline int64_t nv_chunk_ceiling_for_vram(const char* env, bool defaults026, uint64_t min_vram_bytes) noexcept {
    if (defaults026) return 0;
    if (env == nullptr && min_vram_bytes > 0 && min_vram_bytes <= (16ull << 30) + (512ull << 20)) return 0;
    return nv_chunk_ceiling(env, defaults026);
}

/// Compute capability 8.0 and up, and 7.0 (Volta, measured, see above).  Turing (7.5) and Pascal (6.x) were not measured: they keep 8192.
/// `min_major` / `min_minor`: the lowest compute capability among the visible cards (the lowest minor of the lowest major).
inline bool nv_chunk_card_ok(int min_major, int min_minor) noexcept { return min_major >= 8 || (min_major == 7 && min_minor == 0); }

/// The RAM guard steps the ceiling down when the extra loan does not fit: 32768 -> 16384 -> 8192.
inline int64_t nv_chunk_step_down(int64_t ceiling) noexcept { return ceiling > 16384 ? 16384 : 8192; }

}  // namespace strata::core
