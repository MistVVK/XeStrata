// include/strata/program/bdone_drafts.hpp - the draft counters of a --batch slot in its BDONE line (#5890).
//
// With --batch-mtp the proposals are checked in the slot windows, and the engine counted them only in its totals ("strata batch: MTP proposals accepted N of M") and the
// log, so a response of a batch-path request read draft_n = 0 and draft_n_accepted = 0 while /metrics moved.  The slot now keeps its own counters (proposals checked, proposals
// kept since the admission) and its BDONE line carries them as ` drafts=<checked>:<kept>`; the server reads them like `read=<ms>`.  A slot that checked no proposal adds
// nothing (the line stays as it was, and an older server ignores a key it does not know).  Pure: no GPU.
#pragma once

#include <cstdint>
#include <string>

namespace strata::program {

inline std::string bdone_drafts_text(int64_t checked, int64_t kept) {
    if (checked <= 0) return std::string();
    if (kept < 0) kept = 0;
    if (kept > checked) kept = checked;
    return " drafts=" + std::to_string((long long) checked) + ":" + std::to_string((long long) kept);
}

}  // namespace strata::program
