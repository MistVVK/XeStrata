// src/core/dflash2_gemm_bench.cpp - the drafter's Q8 GEMM (kernels/dflash2.hpp df2_gemm) on random planar weights: GB/s per shape and column count.
//
//     dflash2_gemm_bench [iters] [q4]      (q4: the Q4_0 pack's GEMM, kernels/cuda/dflash2.cu gemm4_*)
//
// Weight sets are cycled through 2 GiB of distinct matrices so the L2 cache cannot hold them (the drafter reads each matrix once per round).
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    const bool q4 = argc > 2 && std::strcmp(argv[2], "q4") == 0;
    const int wnum = q4 ? 18 : 34;   // bytes of a 32-weight block: the Q8 pack's int8 + fp16 scale, the Q4_0 pack's nibbles + fp16 scale
    struct Shape { const char* name; int n_in, n_out; int mats; };
    const Shape shapes[] = {{"gate+up 2560->7680 x2", 2560, 7680, 2}, {"down 7680->2560", 7680, 2560, 1}, {"qkv 2560->(4096,1024,1024)", 2560, 4096, 3},
                            {"o 4096->2560", 4096, 2560, 1}, {"fc 12800->2560", 12800, 2560, 1}, {"kv 2560->1024 x2", 2560, 1024, 2},
                            {"akp 2560->640", 2560, 640, 1}, {"sel 2560->256", 2560, 256, 1}};
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (const Shape& sh : shapes) {
        const size_t wbytes = (size_t) sh.n_out * sh.n_in * wnum / 32;
        const size_t per_set = wbytes * sh.mats;
        const int sets = (int) std::max<size_t>(2, ((size_t) 1 << 30) / per_set);
        std::vector<uint8_t*> w((size_t) sets * sh.mats);
        for (auto& p : w) {
            if (cudaMalloc((void**) &p, wbytes) != cudaSuccess) { std::fprintf(stderr, "alloc\n"); return 1; }
            cudaMemset(p, 0x11, wbytes);
        }
        uint8_t* xq;
        float *xd, *y;
        cudaMalloc((void**) &xq, (size_t) sh.n_in * 8);
        cudaMalloc((void**) &xd, (size_t) sh.n_in / 32 * 8 * 4);
        cudaMalloc((void**) &y, (size_t) 8 * sh.n_out * 3 * 4);
        {
            std::vector<int8_t> hx((size_t) sh.n_in * 8);
            uint32_t rs = 777u;
            for (auto& v : hx) { rs = rs * 1664525u + 1013904223u; v = (int8_t) (rs >> 20); }
            cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice);
            std::vector<float> ones((size_t) sh.n_in / 32 * 8);
            for (auto& v : ones) { rs = rs * 1664525u + 1013904223u; v = 0.005f + 0.001f * (float) ((rs >> 20) % 16); }
            cudaMemcpy(xd, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice);
        }
        std::vector<float> yref;
        std::vector<int> vars = {1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
        if (q4) for (int v = 24; v <= 45; ++v) vars.push_back(v);
        for (int var : vars) {
            strata::kernels::df2_gemm_variant(var);
            const int ncols = 8;
            const size_t ny = (size_t) 8 * sh.n_out * (sh.mats == 3 ? 3 : sh.mats);
            {   // bitwise against variant 1
                strata::kernels::Df2Mat m[3];
                for (int i = 0; i < sh.mats; ++i) m[i] = {w[(size_t) i], y + (size_t) i * 8 * sh.n_out, sh.n_out, q4};
                if (sh.mats == 3) { m[1].n_out = 1024; m[2].n_out = 1024; }
                cudaMemset(y, 0, ny * 4);
                strata::kernels::df2_gemm(m, sh.mats, xq, xd, sh.n_in, ncols, nullptr);
                cudaDeviceSynchronize();
                std::vector<float> hy(ny);
                cudaMemcpy(hy.data(), y, ny * 4, cudaMemcpyDeviceToHost);
                if (var == 1) yref = hy;
                std::printf("%s %-30s variant %2d: %s  ", q4 ? "q4" : "q8", sh.name, var, std::memcmp(hy.data(), yref.data(), ny * 4) == 0 ? "bitwise" : "DIFFERS");
            }
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int it = 0; it < iters; ++it) {
                strata::kernels::Df2Mat m[3];
                for (int i = 0; i < sh.mats; ++i) m[i] = {w[(size_t) (it % sets) * sh.mats + i], y + (size_t) i * 8 * sh.n_out, sh.n_out, q4};
                if (sh.mats == 3) { m[1].n_out = 1024; m[2].n_out = 1024; }
                strata::kernels::df2_gemm(m, sh.mats, xq, xd, sh.n_in, ncols, nullptr);
            }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            const double bytes = sh.mats == 3 ? (double) (4096 + 1024 + 1024) * sh.n_in * wnum / 32 : (double) per_set;
            std::printf("%7.1f us  %6.1f GB/s\n", 1000.0 * ms / iters, bytes * iters / (ms / 1000.0) / 1e9);
        }
        for (auto p : w) cudaFree(p);
        cudaFree(xq); cudaFree(xd); cudaFree(y);
    }
    return 0;
}
