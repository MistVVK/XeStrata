// tests/quant/sk_cuda_test.cu - Bit-exact parity & Blackwell/Ada HBM bandwidth benchmark for sk_cuda.cuh
#include "strata/quant/sk_codecs.hpp"
#include "strata/quant/sk_cuda.cuh"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

using namespace strata::quant::sk;
using namespace strata::quant::sk_gpu;

int main() {
    const int nrows = 3072;
    const int ncols = 1536;
    const int n_sb_row = ncols / QK_S;
    const size_t w_bytes = static_cast<size_t>(nrows) * n_sb_row * 134;

    // Deterministic synthetic S4_K weights + activations
    std::vector<uint8_t> h_w(w_bytes);
    for (size_t i = 0; i < w_bytes; ++i) h_w[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
    // Set valid FP16 super-scales (0x3400 = 0.25)
    for (int r = 0; r < nrows; ++r) {
        for (int s = 0; s < n_sb_row; ++s) {
            uint8_t* blk = h_w.data() + (static_cast<size_t>(r) * n_sb_row + s) * 134;
            blk[0] = 0x00;
            blk[1] = 0x34;
            blk[2] = 200; blk[3] = 180; blk[4] = 240; blk[5] = 150;
        }
    }

    std::vector<float> h_xf(ncols);
    for (int i = 0; i < ncols; ++i) h_xf[i] = std::sin(static_cast<float>(i) * 0.13f);
    std::vector<BlockQ8_1_G64> h_qx(ncols / SUB_G);
    quantize_q8_1_g64(h_xf.data(), h_qx.data(), ncols);

    // CPU reference
    std::vector<float> ref_out(nrows);
    for (int r = 0; r < nrows; ++r) {
        ref_out[r] = dot_sk_q8_1_row(h_w.data() + static_cast<size_t>(r) * n_sb_row * 134, SkCode::S4_K, h_qx.data(), ncols);
    }

    // GPU run
    uint8_t* d_w = nullptr;
    BlockQ8_1_G64_Dev* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_x, h_qx.size() * sizeof(BlockQ8_1_G64_Dev));
    cudaMalloc(&d_out, nrows * sizeof(float));
    cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_qx.data(), h_qx.size() * sizeof(BlockQ8_1_G64_Dev), cudaMemcpyHostToDevice);

    const int threads = 256;  // 8 warps/block -> 8 rows/block
    const int blocks = (nrows + 7) / 8;
    gemv_s4k_q8_1_kernel<<<blocks, threads>>>(d_w, d_x, d_out, nrows, ncols);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(nrows);
    cudaMemcpy(gpu_out.data(), d_out, nrows * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int r = 0; r < nrows; ++r) {
        max_err = std::max(max_err, static_cast<double>(std::fabs(gpu_out[r] - ref_out[r])));
    }

    // Bandwidth benchmark (200 iterations)
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int it = 0; it < 200; ++it) {
        gemv_s4k_q8_1_kernel<<<blocks, threads>>>(d_w, d_x, d_out, nrows, ncols);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    double gbps = (static_cast<double>(w_bytes) * 200.0 / 1e9) / (ms / 1e3);

    std::printf("[CUDA S4_K Kernel] [3072x1536] max_err=%.3e | %.3f us/call | %.1f GB/s %s\n",
                max_err, (ms / 200.0) * 1000.0, gbps, max_err < 1e-4 ? "PASS" : "FAIL");
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out);
    return max_err < 1e-4 ? 0 : 1;
}
