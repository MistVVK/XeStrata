// tests/cuda/hc_q8_stream_test.cu - `hc_q8_to_bf16` expands on the stream it is given.  The layer-major prompt walk
// called `bf16_proj` for its hyper-connection projections without a stream, so under STRATA_HC_REQ8 the int8 expansion
// ran on the default stream while the GEMMs read it on another.  This pins the property that failed; it does not cover
// the call sites themselves (`bf16_proj` is file-static, so no test reaches it).
//
// Exit 0 pass, 1 fail, 77 no CUDA device (ctest SKIP_RETURN_CODE).
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int fails = 0;

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

// the kernel's own formula, on the host: bf16 round-to-nearest-even of (scale * code)
std::vector<uint16_t> reference(const std::vector<int8_t>& q, const std::vector<float>& sc) {
    std::vector<uint16_t> out(q.size());
    for (size_t i = 0; i < q.size(); ++i) {
        const float v = sc[i / 32] * (float) q[i];
        uint32_t b;
        std::memcpy(&b, &v, 4);
        b += 0x7fffu + ((b >> 16) & 1u);
        out[i] = (uint16_t) (b >> 16);
    }
    return out;
}

// keeps its stream busy for a few ms: bounded on purpose, so the test cannot hang on a host/device ordering mistake
// (a host copy that releases a spinning kernel would itself wait for that stream)
__global__ void spin_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    for (int i = 0; i < 20000; ++i) __nanosleep(1000);
#else
    // sm_60/61 have no __nanosleep: spin on the clock for about the same few ms
    const long long t0 = clock64();
    while (clock64() - t0 < 20000000LL) {}
#endif
}

// one case: enqueue the expansion on `A`, optionally behind a spin that holds `A` busy, and read it back ordered only by
// an event on `A` (never by a device-wide synchronize, which would hide exactly the defect this guards)
void check(bool with_spin, int n_blocks, uint32_t seed) {
    const long long n = (long long) n_blocks * 32;
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> code(-127, 127);
    std::normal_distribution<float> scale(0.0f, 0.02f);
    std::vector<int8_t> q((size_t) n);
    std::vector<float> sc((size_t) n_blocks);
    for (auto& v : q) v = (int8_t) code(g);
    for (auto& v : sc) v = scale(g);
    const std::vector<uint16_t> want = reference(q, sc);

    int8_t* d_q = nullptr;
    float* d_sc = nullptr;
    uint16_t* d_out = nullptr;
    cudaStream_t a = nullptr;
    cudaEvent_t ev = nullptr;
    bool ok = ck(cudaStreamCreate(&a), "stream") && ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event") &&
              ck(cudaMalloc(&d_q, (size_t) n), "alloc q") && ck(cudaMalloc(&d_sc, (size_t) sc.size() * 4), "alloc sc") &&
              ck(cudaMalloc(&d_out, (size_t) n * 2), "alloc out");
    if (ok) {
        ck(cudaMemcpy(d_q, q.data(), (size_t) n, cudaMemcpyHostToDevice), "upload q");
        ck(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload sc");
        ck(cudaMemset(d_out, 0xff, (size_t) n * 2), "poison out");
        if (with_spin) spin_kernel<<<1, 1, 0, a>>>();
        strata::kernels::hc_q8_to_bf16(d_q, d_sc, d_out, n, a);
        // the readback is ordered by the expansion's stream and by nothing else
        ck(cudaEventRecord(ev, a), "record");
        ck(cudaStreamWaitEvent(nullptr, ev, 0), "wait on the default stream");
        std::vector<uint16_t> got((size_t) n);
        ck(cudaMemcpy(got.data(), d_out, (size_t) n * 2, cudaMemcpyDeviceToHost), "download");
        size_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
        const bool pass = bad == 0;
        std::printf("hc_q8_to_bf16 %s, %d blocks, stream A: %s (%zu of %zu differ)\n",
                    with_spin ? "behind a busy stream" : "plain", n_blocks, pass ? "ok" : "FAIL", bad, got.size());
        fails += pass ? 0 : 1;
    } else {
        ++fails;
    }
    if (a) cudaStreamSynchronize(a);
    if (ev) cudaEventDestroy(ev);
    if (a) cudaStreamDestroy(a);
    if (d_q) cudaFree(d_q);
    if (d_sc) cudaFree(d_sc);
    if (d_out) cudaFree(d_out);
}

}  // namespace

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 1) {
        std::printf("no CUDA device\n");
        return 77;
    }
    check(false, 320, 1);       // the HC down projection's shape: 320 rows of 10240
    check(true, 320, 2);        // the same, with work already queued on that stream
    check(true, 4096, 3);       // a bigger matrix, so the launch shape changes
    std::printf(fails == 0 ? "PASS\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
