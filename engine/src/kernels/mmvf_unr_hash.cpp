// src/kernels/mmvf_unr_hash.cpp - FNV-1a of the router-shaped bf16 reads (2560 inputs, 512 outputs; single-column and 1..8-column multi),
// for comparing STRATA_SMALL_UNR=0 against =1 (run twice, compare the printed hashes: they must be equal).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main() {
    const int n_in = 2560, n_out = 512, TM = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) TM * n_in);
    std::vector<uint16_t> w((size_t) n_out * n_in);
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = bf16(0.05f * nd(rng));
    float *dx, *dy;
    uint16_t* dw;
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMalloc((void**) &dw, w.size() * 2);
    cudaMalloc((void**) &dy, (size_t) TM * n_out * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> y((size_t) TM * n_out);
    auto fnv = [&](int rows) {
        unsigned long long h = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t) rows * n_out * 4; ++i) h = (h ^ ((const unsigned char*) y.data())[i]) * 1099511628211ull;
        return h;
    };
    cudaMemset(dy, 0xff, (size_t) TM * n_out * 4);
    strata::kernels::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in, n_out, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y.data(), dy, (size_t) n_out * 4, cudaMemcpyDeviceToHost);
    std::printf("single   %016llx\n", fnv(1));
    for (int T = 1; T <= TM; ++T) {
        cudaMemset(dy, 0xff, (size_t) TM * n_out * 4);
        strata::kernels::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, n_out, n_in, n_out, T, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(y.data(), dy, (size_t) T * n_out * 4, cudaMemcpyDeviceToHost);
        std::printf("multi T%d %016llx\n", T, fnv(T));
    }
    {   // bit 1: the GDN a/b projection (2560 inputs, h_v = 32 heads: alpha and beta rows), T = 1..8
        const int h_v = 32;
        std::vector<uint16_t> wa((size_t) h_v * n_in), wb((size_t) h_v * n_in);
        std::vector<float> dt(h_v), sa(h_v);
        for (auto& v : wa) v = bf16(0.05f * nd(rng));
        for (auto& v : wb) v = bf16(0.05f * nd(rng));
        for (auto& v : dt) v = nd(rng);
        for (auto& v : sa) v = -std::abs(nd(rng));
        uint16_t *dwa, *dwb;
        float *ddt, *dsa, *dg, *db;
        cudaMalloc((void**) &dwa, wa.size() * 2); cudaMalloc((void**) &dwb, wb.size() * 2);
        cudaMalloc((void**) &ddt, h_v * 4); cudaMalloc((void**) &dsa, h_v * 4);
        cudaMalloc((void**) &dg, (size_t) TM * h_v * 4); cudaMalloc((void**) &db, (size_t) TM * h_v * 4);
        cudaMemcpy(dwa, wa.data(), wa.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dwb, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ddt, dt.data(), h_v * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsa, sa.data(), h_v * 4, cudaMemcpyHostToDevice);
        for (int T = 1; T <= TM; ++T) {
            cudaMemset(dg, 0xff, (size_t) TM * h_v * 4); cudaMemset(db, 0xff, (size_t) TM * h_v * 4);
            strata::kernels::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, dg, db, n_in, h_v, T, nullptr);
            cudaDeviceSynchronize();
            std::vector<float> g((size_t) T * h_v), b((size_t) T * h_v);
            cudaMemcpy(g.data(), dg, g.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), db, b.size() * 4, cudaMemcpyDeviceToHost);
            unsigned long long h = 1469598103934665603ull;
            for (size_t i = 0; i < g.size() * 4; ++i) h = (h ^ ((const unsigned char*) g.data())[i]) * 1099511628211ull;
            for (size_t i = 0; i < b.size() * 4; ++i) h = (h ^ ((const unsigned char*) b.data())[i]) * 1099511628211ull;
            std::printf("gdn_ab T%d %016llx\n", T, h);
        }
    }
    return 0;
}
