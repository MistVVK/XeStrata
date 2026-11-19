// launch_bench: kernel launch cost on Windows HIP gfx1151, plain stream launches vs stream capture + graph replay
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void kEmpty(float* p, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && n < 0) p[0] = 1.0f;
}
__global__ void kAdd(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1.0f;
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d %s at line %d\n", (int)e_, hipGetErrorString(e_), __LINE__); return 1; } } while (0)

int main() {
    float* d;
    CK(hipMalloc(&d, 1 << 24));
    hipStream_t s;
    CK(hipStreamCreate(&s));
    const int N = 2000;
    for (int w = 0; w < 100; ++w) hipLaunchKernelGGL(kEmpty, dim3(1), dim3(32), 0, s, d, 0);
    CK(hipStreamSynchronize(s));
    for (int rep = 0; rep < 3; ++rep) {
        double t0 = now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(kEmpty, dim3(1), dim3(32), 0, s, d, 0);
        double t1 = now();
        CK(hipStreamSynchronize(s));
        double t2 = now();
        printf("empty kernels: %d launches: host enqueue %.1f us each, total to sync %.1f us each\n", N, (t1 - t0) / N * 1e6, (t2 - t0) / N * 1e6);
    }
    for (int rep = 0; rep < 2; ++rep) {
        double t0 = now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(kAdd, dim3(256), dim3(256), 0, s, d, 65536);
        CK(hipStreamSynchronize(s));
        double t2 = now();
        printf("tiny kernels (256 blocks): %.1f us each to sync\n", (t2 - t0) / N * 1e6);
    }
    // graph capture of N empty kernels
    hipGraph_t g;
    hipGraphExec_t ge;
    double t0 = now();
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) hipLaunchKernelGGL(kAdd, dim3(256), dim3(256), 0, s, d, 65536);
    CK(hipStreamEndCapture(s, &g));
    double t1 = now();
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    double t2 = now();
    printf("graph: capture %.1f ms, instantiate %.1f ms\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3);
    for (int rep = 0; rep < 4; ++rep) {
        double a = now();
        CK(hipGraphLaunch(ge, s));
        double b = now();
        CK(hipStreamSynchronize(s));
        double c = now();
        printf("graph replay: launch call %.1f us, total %.2f ms = %.1f us per node\n", (b - a) * 1e6, (c - a) * 1e3, (c - a) / N * 1e6);
    }
    return 0;
}
