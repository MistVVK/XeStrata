// tests/cuda/native_moe_gated_test.cu - the gated combine (STRATA_LFUSE, native_moe_combine_multi_hits_gated) rounds the shared expert's
// scaled row on its own, then adds it, bitwise what shared_expert_multi's scale_rows store followed by the ungated combine gave (#5100).
// nvcc ignores `#pragma clang fp contract(off)`, so the product used to be fused into the add (one rounding, `sum = fma(shared, g, sum)`).
// Reference: a kernel that stores shared * g (the rounded product, written to memory) with the same gate expression, then
// native_moe_combine_multi_hits on that row.  Synthetic data, no model.  Exit 0 pass, 1 fail, 77 no CUDA device.
#include "strata/kernels/native_moe.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// scale_rows_kernel's row scale: g as native_scalar_sigmoid_multi_kernel computes it, the product stored (rounded) to memory
__global__ void scale_rows_ref(const float* __restrict__ shared, const float* __restrict__ sg, float* __restrict__ out, int n_embd) {
    const int tk = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
    out[(size_t) tk * n_embd + col] = __fmul_rn(shared[(size_t) tk * n_embd + col], g);
}

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("native_moe_gated_test: no CUDA device, skipped\n");
        return 77;
    }
    const int N = 2560, K = 10, T = 6;
    std::mt19937 rng(5100);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f), w01(0.0f, 1.0f), gl(-4.0f, 4.0f);
    std::vector<float> hits((size_t) T * K * N), weights((size_t) T * K), shared((size_t) T * N), gate(T);
    for (auto& v : hits) v = u(rng);
    for (auto& v : weights) v = w01(rng) * 0.2f;
    for (auto& v : shared) v = u(rng);
    for (auto& v : gate) v = gl(rng);

    float *d_hits = nullptr, *d_w = nullptr, *d_sh = nullptr, *d_g = nullptr, *d_ref_sh = nullptr, *d_out = nullptr, *d_ref = nullptr;
    bool ok = ck(cudaMalloc(&d_hits, hits.size() * 4), "malloc hits") && ck(cudaMalloc(&d_w, weights.size() * 4), "malloc w") &&
              ck(cudaMalloc(&d_sh, shared.size() * 4), "malloc shared") && ck(cudaMalloc(&d_g, T * 4), "malloc gate") &&
              ck(cudaMalloc(&d_ref_sh, shared.size() * 4), "malloc ref shared") && ck(cudaMalloc(&d_out, shared.size() * 4), "malloc out") &&
              ck(cudaMalloc(&d_ref, shared.size() * 4), "malloc ref");
    if (!ok) return 1;
    cudaMemcpy(d_hits, hits.data(), hits.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weights.data(), weights.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh, shared.data(), shared.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, gate.data(), T * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);

    scale_rows_ref<<<dim3((N + 255) / 256, T), 256, 0, s>>>(d_sh, d_g, d_ref_sh, N);
    strata::kernels::native_moe_combine_multi_hits(d_hits, d_w, d_ref_sh, d_ref, N, K, T, s);
    strata::kernels::native_moe_combine_multi_hits_gated(d_hits, d_w, d_sh, d_g, d_out, N, K, T, s);
    if (!ck(cudaStreamSynchronize(s), "sync")) return 1;

    std::vector<float> out(shared.size()), ref(shared.size());
    cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref.data(), d_ref, ref.size() * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::memcmp(&out[i], &ref[i], 4) != 0) {
            if (first == (size_t) -1) first = i;
            ++diff;
        }
    }
    if (diff) {
        std::printf("FAIL: %zu of %zu outputs differ from the scale-rows-then-add path (first at %zu: %.9g vs %.9g)\n", diff, out.size(), first, out[first], ref[first]);
        return 1;
    }
    std::printf("native_moe_gated_test: ok (%zu outputs bitwise equal)\n", out.size());
    return 0;
}
