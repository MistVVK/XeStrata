// include/strata/prefill/dense_mmq_notice.hpp - #6269: STRATA_DENSE_MMQ=1 in an engine built without llama.cpp's MMQ kernels does nothing (the dense projections
// of a prompt take the dequantize + FP16 GEMM path).  The setting used to switch the dense projections to the MMQ int8 kernels; an owner who still has it in a
// config gets this one line at start-up instead of a setting that silently does nothing.  A pure function: no GPU, no model.
#pragma once

#include <string>

namespace strata::prefill {

/// The start-up line for the value of STRATA_DENSE_MMQ (nullptr: not set) in an engine built without MMQ.  Empty: nothing to say (not set, or off).
/// A line is for the values that used to turn the path on; anything but "0" and an empty value counts, as the old reader's `1` did not stand alone in the docs.
inline std::string dense_mmq_notice(const char* value) {
    if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) return std::string();
    return "strata: STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels): the dense projections of a prompt run on the FP16 path.";
}

}  // namespace strata::prefill
