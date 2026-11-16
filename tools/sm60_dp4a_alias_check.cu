// Focused regression for output/input register aliasing in the multi-instruction
// SM60 VMAD fallback. Build with CUDA 12 for sm_60, sm_61 and a modern target.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "strata/kernels/dp4a.hpp"

__host__ __device__ uint32_t oracle(uint32_t a, uint32_t b, uint32_t c) {
    for (int lane = 0; lane < 4; ++lane) {
        const int x = int((a >> (8 * lane)) & 255);
        const int y = int((b >> (8 * lane)) & 255);
        c += uint32_t((x < 128 ? x : x - 256) * (y < 128 ? y : y - 256));
    }
    return c;
}

__global__ void probe(uint32_t* results) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = int(t * 0x9e3779b9u);
    const int b = int((t ^ 0xdeadbeefu) * 0x85ebca6bu);
    // These source operands intentionally have equal values. Without an
    // early-clobber constraint nvcc may share the accumulator's register.
    results[4 * t + 0] = uint32_t(STRATA_DP4A(a, b, a));
    results[4 * t + 1] = uint32_t(STRATA_DP4A(a, b, b));
    results[4 * t + 2] = uint32_t(STRATA_DP4A(a, a, a));
    results[4 * t + 3] = uint32_t(STRATA_DP4A(b, b, b));
}

int main() {
    constexpr int N = 65536;
    uint32_t *device = nullptr, *host = new uint32_t[4 * N];
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) err = cudaMalloc(&device, 4 * N * sizeof(uint32_t));
    if (err != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(err)); return 2; }
    probe<<<N / 256, 256>>>(device);
    err = cudaMemcpy(host, device, 4 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(err)); return 2; }
    unsigned bad[4]{};
    for (uint32_t t = 0; t < N; ++t) {
        const uint32_t a = t * 0x9e3779b9u, b = (t ^ 0xdeadbeefu) * 0x85ebca6bu;
        const uint32_t want[] = {oracle(a, b, a), oracle(a, b, b), oracle(a, a, a), oracle(b, b, b)};
        for (int i = 0; i < 4; ++i) bad[i] += host[4 * t + i] != want[i];
    }
    std::printf("%s sm_%d%d cases=%d mismatches=%u,%u,%u,%u\n", prop.name, prop.major, prop.minor,
                4 * N, bad[0], bad[1], bad[2], bad[3]);
    cudaFree(device);
    delete[] host;
    return (bad[0] || bad[1] || bad[2] || bad[3]) ? 1 : 0;
}
