// include/strata/prefill/mmq_ids_margin.hpp - the room an MMQ row table (`ids`) needs past its last row (PR #5108, @quantforge-br).
//
// llama.cpp's mul_mat_q_stream_k_fixup loads ids_dst[col_low + jt*J + j] for a whole J-column tile before it compares the
// tile with the expert's rows (the main kernel skips an empty tile first; the fixup only skips its writes).  Its tiles
// run to ceil(max_rows / J) * J for every expert of the launch, and col_low is the expert's first row, which can be the
// table's last.  A table of exactly `rows` entries is therefore read up to roundup(max_rows, J) entries past its end, and
// the main kernel reads up to J - 1 past the last row of its last expert.  Strata's tables were exact-sized cudaMallocs:
// one that ended a 2 MiB page faulted on the next (Xid 31, "prefill copy_i32: an illegal memory access" on a long prompt).
//
// J is bounded by the backend: mul_mat_q_switch_J tries J = 8, 16, ... kMmqTileColsMax and launches one of those.  The
// MMQ planner in moe_mmq.cu iterates the same range (it uses kMmqTileColsMax), so the margin follows the tile bound in
// one place.  An expert takes at most one row per token, so max_rows <= the chunk's tokens.  One extra tile on top of the
// rounded read keeps the last row's neighbour in the allocation too.  Checked arithmetic: a request that does not fit in
// 64 bits is refused (the callers fail the carve) instead of wrapping to a small table.
#pragma once

#include <cstdint>

namespace strata::prefill::mmq {

/// The widest MMQ column tile: ggml-cuda's mul_mat_q_switch_J tries J = 8 .. 128 in steps of 8.
constexpr int kMmqTileColsMax = 128;
constexpr int kMmqTileColsStep = 8;

/// Entries to add after a row table whose launch has at most `max_rows` rows for one expert: roundup(max_rows, J) + J.
/// Returns false (and leaves *out alone) when `max_rows` is negative or the sum overflows.
inline bool mmq_ids_margin_rows(int64_t max_rows, int tile_cols_max, int64_t* out) {
    if (max_rows < 0 || tile_cols_max <= 0) return false;
    const int64_t J = tile_cols_max;
    const int64_t tiles = max_rows / J + (max_rows % J != 0 ? 1 : 0);
    if (tiles > INT64_MAX / J) return false;
    const int64_t rounded = tiles * J;
    if (rounded > INT64_MAX - J) return false;
    *out = rounded + J;
    return true;
}

/// A table of `rows` MMQ rows (one launch, at most `max_rows` per expert): the entries to allocate and write.  0 on overflow
/// (callers treat it as "does not fit": an empty take fails the carve).
inline uint64_t mmq_ids_table_rows(uint64_t rows, uint64_t max_rows, int tile_cols_max = kMmqTileColsMax) {
    int64_t margin = 0;
    if (max_rows > (uint64_t) INT64_MAX || !mmq_ids_margin_rows((int64_t) max_rows, tile_cols_max, &margin)) return 0;
    if (rows > UINT64_MAX - (uint64_t) margin) return 0;
    return rows + (uint64_t) margin;
}

}  // namespace strata::prefill::mmq
