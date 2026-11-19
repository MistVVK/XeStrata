// take_pending_cuda_error (include/strata/core/pending_cuda_error.hpp): an error an earlier unchecked call left in the last-error slot is said and cleared,
// so the next launch check does not blame it on itself (#5636).  Needs a CUDA device for the failing call to be the invalid-argument one; exits 77 without.
#include "strata/core/pending_cuda_error.hpp"

#include <cuda_runtime.h>

#include <cstdio>

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { (void) cudaGetLastError(); std::printf("pending_cuda_error_test: no device, skipped\n"); return 77; }
    CHECK(cudaFree(nullptr) == cudaSuccess);
    (void) cudaGetLastError();
    // nothing pending: nothing said, nothing changed
    CHECK(strata::core::take_pending_cuda_error("the test") == cudaSuccess);
    // an unchecked failing call (a copy from a null host pointer is an invalid argument and leaves no sticky state)
    char* d = nullptr;
    CHECK(cudaMalloc((void**) &d, 16) == cudaSuccess);
    const cudaError_t bad = cudaMemcpy(d, nullptr, 16, cudaMemcpyHostToDevice);
    CHECK(bad != cudaSuccess);
    // before the helper: the slot still holds it (a launch check reading it now would blame its own launch)
    CHECK(cudaPeekAtLastError() == bad);
    // the helper returns it and clears it
    CHECK(strata::core::take_pending_cuda_error("the test capture") == bad);
    CHECK(cudaPeekAtLastError() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);
    cudaFree(d);
    if (failed) { std::fprintf(stderr, "pending_cuda_error_test: %d failed\n", failed); return 1; }
    std::printf("pending_cuda_error_test: ok\n");
    return 0;
}
