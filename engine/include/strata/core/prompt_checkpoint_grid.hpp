#pragma once
// include/strata/core/prompt_checkpoint_grid.hpp - where the next periodic conversation checkpoint of a prompt read
// is due (--prompt-cache-every).  Header-only; generate.cpp's single-card path and a layer split's per-stage parts use
// the same rule, so the parts of one checkpoint are saved at the same positions.

#include <cstdint>
#include <limits>

namespace strata::core {

/// The periodic checkpoints sit on the grid `anchor + k * every` (k = 1, 2, ...), `anchor` = where the request resumed
/// (its reused prefix).  A checkpoint can only be taken at a chunk end, so it is taken at the first chunk end at or past
/// a grid point; after one at `done`, the next is due at the first grid point PAST `done`.  (#6052: it used to be due at
/// `done + every`, so equal prompt chunks a little shorter than the spacing skipped every other grid point: 15,412-token
/// chunks with a spacing of 16,384 gave checkpoints at 30,824 and 61,648 instead of 30,824, 46,236 and 61,648.)  A chunk
/// that crosses several grid points still gives one checkpoint.  INT64_MAX: none (spacing off, or past the range).
inline int64_t next_periodic_checkpoint(int64_t anchor, int64_t done, int64_t every) {
    constexpr int64_t kNone = std::numeric_limits<int64_t>::max();
    if (every <= 0 || anchor < 0) return kNone;
    const int64_t k = done < anchor ? 1 : (done - anchor) / every + 1;
    if (k > (kNone - anchor) / every) return kNone;
    return anchor + k * every;
}

}  // namespace strata::core
