// include/strata/core/mtp_head_rule.hpp - the bytes of one row of the MTP draft head's subset buffer (src/core/mtp.cpp bind / make_q4_head).
//
// #5841: --mtp-q4 head|all rewrites the gathered subset in place as Q4_0 rows (n_embd / 32 * 18 bytes).  The buffer was sized for the head's own row, and a
// head in IQ4_XS (n_embd / 256 * 136 bytes: 1,360 against Q4_0's 1,440 at n_embd 2560) has the narrower row, so the host-to-device copy wrote 8.1 MiB past
// the buffer ("mtp: writing the draft head", exit 1).  The buffer takes the wider of the two rows.  Pure: no GPU.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

/// the bytes a subset row needs: the head's own row, or the Q4_0 row when --mtp-q4 rewrites the subset (and n_embd is a multiple of 32), whichever is wider
inline int64_t mtp_draft_head_row_cap(int64_t head_row_bytes, int64_t n_embd, bool q4_head) {
    const int64_t q4_row = (q4_head && n_embd > 0 && n_embd % 32 == 0) ? (n_embd / 32) * 18 : 0;
    return std::max(head_row_bytes, q4_row);
}

}  // namespace strata::core
