#pragma once
// STRATA_SELECT_TWO_PASS: the request, read once, and the line a set switch prints where the two-pass scorer cannot
// run.  The scorer runs only in the CUDA prompt path's chunked walk, on a card where qsa_block_scores_tc runs its
// tensor-core kernel (sm_80+, STRATA_QSA_WARP not set).  Elsewhere (HIP, Metal, SYCL, an older card or the warp
// scorer, the layer-major prompt path) the selection runs as without the switch; that was silent, so the switch read
// as active.  Header-only, so the rule and the line are tested without a GPU (qsa_two_pass_status_test).
#include <atomic>
#include <cstdio>

namespace strata::kernels::two_pass {

// The switch's own parse: exactly "1" is on (as the two-pass scorer read it before).
inline bool requested_from(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

class RequestStatus {
public:
    explicit RequestStatus(const char* flag) : requested_(requested_from(flag)) {}

    bool requested() const { return requested_; }

    // `not_here` null: this path can run the scorer, so the result is the request (whether a call takes it still
    // depends on the call: its size, geometry and scratch).  Otherwise the path cannot run it: false, and the first
    // such query prints one line naming the path.  A supported query never uses up that line.
    bool on(const char* not_here, FILE* log = stderr) {
        if (!requested_) return false;
        if (not_here == nullptr) return true;
        if (!warned_.exchange(true)) {
            std::fprintf(log, "strata: STRATA_SELECT_TWO_PASS=1 is not used on %s: the two-pass scorer runs only in "
                              "the CUDA prompt path's tensor-core selection (sm_80+); the selection is unchanged\n",
                         not_here);
            std::fflush(log);
        }
        return false;
    }

private:
    const bool requested_;
    std::atomic<bool> warned_{false};
};

}  // namespace strata::kernels::two_pass
