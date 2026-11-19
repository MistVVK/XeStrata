// include/strata/core/pending_cuda_error.hpp - a CUDA error left in the thread's last-error slot, said and cleared before a graph capture.
//
// #5636: on sm_70 / sm_75 the first verify-window capture died with "copy_i32_from_mapped launch: invalid argument" on a copy kernel that is a plain
// grid-stride loop.  The launch wrappers read cudaGetLastError(), which returns the FIRST error since the last read from ANY earlier call of the thread, not
// only the kernel just launched: an earlier call that failed and was not checked (an optional attribute, a query, a copy that fell back) is blamed on the
// next launch checked, and that launch's check ends the engine.  The slot is read here before the capture starts: a non-sticky error is said (what it is,
// that it was pending) and cleared, so it is not blamed on the capture's first launch; a sticky one (an illegal address, a lost device) stays and fails the
// calls after it as it must.  Nothing is cleared that a launch of the capture produced.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace strata::core {

/// the error pending in this thread's last-error slot before `where` (cudaSuccess when none); a non-sticky one is cleared and said once per text
inline cudaError_t take_pending_cuda_error(const char* where) {
    const cudaError_t e = cudaPeekAtLastError();
    if (e == cudaSuccess) return e;
    (void) cudaGetLastError();
    std::fprintf(stderr, "strata: a CUDA error was already pending before %s (%s, %d); it came from an earlier call that was not checked, and it is cleared "
                         "so it is not blamed on the launches that follow\n", where, cudaGetErrorString(e), (int) e);
    return e;
}

}  // namespace strata::core
