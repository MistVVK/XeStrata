// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// strata/kernels/mrope.hpp - multimodal rotary positions (the vision path).
//
// qwen4exp rotates with interleaved M-RoPE (llama.cpp LLAMA_ROPE_TYPE_IMROPE, rope.dimension_sections 11/11/10/0):
// each of the 32 rotated pairs takes its position from one of three streams, time t, height h and width w.  Text
// has t = h = w, which is why every rope kernel here takes one position per row.  An image does not: its tokens
// share t and differ in h and w (mtmd: t = p, h = p + y, w = p + x), and the text after it continues at
// p + max(nx, ny), not at its cell index.
//
// The table: nullptr (the default) keeps every kernel exactly as before - the position a caller passes is the
// rotary position.  Set, it is a DEVICE int32 [cells][3] (t, h, w) and the position a caller passes is read as a
// CELL index; pair i rotates by tab[cell * 3 + mrope_sector(i)].  Every caller in the engine passes cell indices
// (pos_base is 0 everywhere), so the table covers the prompt path, the verify window and the MTP drafter at once.
// The pointer must be set before any CUDA graph is captured (kernels take it as an argument); the contents may
// change between requests.
#pragma once

#include "strata/kernels/rope_scaling.hpp"

#include <cstddef>
#include <cstdint>

#include "strata/kernels/rope_scaling.hpp"

namespace strata::kernels {

/// The table of the CURRENT device (a layer split sets one per device; null = the identity).
void mrope_table_set(const int32_t* device_table);
const int32_t* mrope_table();

/// Opt-in (STRATA_ROPE_TABLE=1, upstream e9f6bb9 / ab1c209): the rotation angles of a 64-wide rotary slice from the
/// session's float64 table (build_rope_table, [max_pos][32] cos and sin in VRAM, the rope scaling inside).  Without it
/// the kernels compute pos * pow(...) in float32 (about 0.002 rad off at 32K) as they always did.
struct RopeTab {
    const float* cos = nullptr;   ///< null: no table
    const float* sin = nullptr;
    int max_pos = 0;
};
/// Registers the table the session just built with `scaling` (the latest one wins).
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling);
/// Forgets the table `cos_tab`, before the memory that holds it is freed.
void rope_table_release(const float* cos_tab);
/// The registered table when STRATA_ROPE_TABLE=1 and it was built with `scaling`; otherwise none.
RopeTab rope_table_for(const RopeScaling& scaling);

#if defined(__CUDACC__)
/// ggml rope_multi, is_imrope, sections {11, 11, 10, 0}: sector = pair % 32; sector % 3 == 1 -> h (sector < 33),
/// == 2 -> w (sector < 30), == 0 -> t (sector < 33).  For pairs 0..31 all three bounds hold, so it is pair % 3.
__device__ __forceinline__ int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}
#elif defined(SYCL_LANGUAGE_VERSION)
/// The same sector rule for the Xe kernels.
inline int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? tab[(std::size_t) pos * 3 + pair % 3] : pos;
}
/// cos and sin of rotary position p, pair `pair` (0..31) from the table; false when there is none or p lies past
/// it (the caller then computes the angle as it always did)
inline bool rope_tab_cs(const RopeTab& t, int p, int pair, float& c, float& s) {
    if (t.cos == nullptr || p < 0 || p >= t.max_pos) return false;
    c = t.cos[(std::size_t) p * 32 + pair];
    s = t.sin[(std::size_t) p * 32 + pair];
    return true;
}
#endif

}  // namespace strata::kernels
