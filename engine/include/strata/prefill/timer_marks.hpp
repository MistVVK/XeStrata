// include/strata/prefill/timer_marks.hpp - which STRATA_PREFILL_TIMING marks are recorded (PR #4006, @Vop3p).
//
// PfTimer charges the gap between two consecutive marks to the phase of the earlier one.  Two marks in a row in the same
// phase therefore add nothing to the totals: the gap between them is charged to that phase either way.  Skipping the
// second one leaves the totals as they were with a fraction of the event records.  That matters under WDDM, where every
// cudaEventRecord is a submission of its own (~80 us on an RX 6900 XT, Windows 11): compute() marks "dequant" once per
// expert, which put ~16 records per MMQ group on the stream and charged the launch gaps they made to "dequant".
// Pure functions, so a host test can check the rule and the totals without a GPU.
#pragma once

#include <cstddef>

namespace strata::prefill {

/// A mark for `phase` is redundant when the newest recorded mark (ph[used - 1]) is already in that phase.
inline bool timer_mark_repeats(const int* ph, size_t used, int phase) {
    return used > 0 && ph[used - 1] == phase;
}

}  // namespace strata::prefill
