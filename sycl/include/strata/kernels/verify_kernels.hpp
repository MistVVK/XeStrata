#pragma once
// Reuse the shared declarations as upstream adds verify and batching APIs.
#include "../../../../include/strata/kernels/verify_kernels.hpp"

namespace strata::kernels {
// Replay the accepted prefix and write recurrent state without output rows.
void gdn_commit_state_multi(float*, const float*, int, const float*, const float*,
                            int, int, int, const int32_t*, void*);
}
