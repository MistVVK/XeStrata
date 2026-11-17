// include/strata/core/draft_cap.hpp - the widest verify window --draft dflash2 uses on each card class (measured A/Bs, notes-dflash2-p1.md, RESULTS-CONF.md; recommend, never force).
#pragma once

#include <cstddef>

namespace strata::core {

/// 48 GB of video memory, counted the way cards are sold (a 48 GB card reports about 47.5 GiB).  The large-Blackwell class below starts here.
inline constexpr size_t kDflash2BigCardBytes = (size_t) 48'000'000'000ull;

/// The large-VRAM Blackwell class the 11-08 sweep measured (RTX PRO 5000 72 GB, sm_120 = cc 12.0): cc 12.0 exactly (not the DGX Spark, cc 12.1) and at least 48 GB.
/// A consumer RTX 50 card with less memory was not measured and keeps the older values.
inline bool dflash2_big_blackwell(int cc_major, int cc_minor, size_t vram_bytes) { return cc_major == 12 && cc_minor == 0 && vram_bytes >= kDflash2BigCardBytes; }

/// 0 = no cap (the --spec given stays).  `uma`: Strix Halo / iGPU (gfx1151); `cc_major` / `cc_minor`: the CUDA compute capability (0 on HIP); `vram_bytes`: the card's memory (CUDA).
/// Measured 11-08 (RESULTS-CONF.md; 36 requests, 300 greedy tokens, 5 interleaved pairs against the old values, all pairs faster):
///  - Strix Halo / UMA: 8, conf 0.35.  UD-Q4_K_XL +5.3% (5 -> 8; 7: +5.1%), UD-IQ4_XS-v31 +2.1%.  Greedy output on UD-Q4_K_XL differs from the old width's (deterministic run to run), v31's is identical
///  - Blackwell cc 12.0 with >= 48 GB (RTX PRO 5000 72 GB): 8 with conf 0.55: IQ3_S +2.4% (6 -> 8 and 0.35 -> 0.55; conf alone +1.1%, cap alone +0.5%), output identical
///  - other Blackwell (cc 12.1 DGX Spark, a card under 48 GB): 6 (nothing beat 0.35 / 6 on the Spark's IQ3_S; the small cards were not measured)
///  - everything else: unmeasured, --spec as given
inline int dflash2_auto_cap(bool uma, int cc_major, int cc_minor, size_t vram_bytes) {
    if (uma) return 8;
    if (cc_major == 12) return dflash2_big_blackwell(cc_major, cc_minor, vram_bytes) ? 8 : 6;
    return 0;
}

/// A 12 GB NVIDIA card (< 13 GiB, CUDA): the MTP drafter was faster in most measurements, which the engine says once.
inline bool dflash2_small_nvidia(bool cuda, size_t total_bytes) { return cuda && total_bytes < ((size_t) 13 << 30); }

}  // namespace strata::core
