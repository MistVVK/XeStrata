// src/core/dflash2_q4_parity.cpp - the drafter's Q4_0 GEMM (kernels/dflash2.hpp df2_gemm with Df2Mat::q4, after df2_q4_tile) against a CPU reference.
//
//     dflash2_q4_parity
//
// Random float matrices are quantized to ggml Q4_0 blocks on the CPU (the reference of tools/dflash2_pack.py quant_q4_0), tiled on the device, and multiplied by
// random planar-int8 activations (df2_quant_rows).  The reference reads the same blocks and the same activation codes back and sums per block in integers,
// so the only difference to the kernel is the float order of the sum over blocks.  Shapes: the drafter's (n_in 2560, 4096, 7680, 12800), one, two and three
// matrices to a launch, 1..8 columns.  Exit 0 when every output is within 2e-5 of the largest value (relative), 1 otherwise.
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t f2h(float f) {   // IEEE half, round to nearest even (the d of a block is stored this way)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}

float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            int s = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; ++s; }
            x = sign | ((uint32_t) (127 - 15 - s + 1) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ggml quantize_row_q4_0_ref: 18 bytes a block
void quant_q4_0(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0, mx = 0;
        for (int i = 0; i < 32; ++i)
            if (std::fabs(x[b * 32 + i]) > amax) { amax = std::fabs(x[b * 32 + i]); mx = x[b * 32 + i]; }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        out[b * 18] = (uint8_t) (dh & 0xff);
        out[b * 18 + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 16; ++i) {
            const int q0 = std::min(15, (int) (x[b * 32 + i] * id + 8.5f)), q1 = std::min(15, (int) (x[b * 32 + 16 + i] * id + 8.5f));
            out[b * 18 + 2 + i] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

struct Case { int n_in; std::vector<int> n_out; };

}  // namespace

int main() {
    std::mt19937 rng(20261031);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    const Case cases[] = {{2560, {64}}, {2560, {64, 32}}, {2560, {4096, 1024, 1024}}, {4096, {2560}}, {7680, {2560}}, {12800, {2560}}, {2560, {7680, 7680}}, {2560, {16}}};
    int bad = 0, total = 0;
    for (const Case& cs : cases) {
        const int n_in = cs.n_in, nb = n_in / 32, nm = (int) cs.n_out.size();
        std::vector<std::vector<uint8_t>> blocks(nm);
        std::vector<uint8_t*> dev_w(nm);
        std::vector<float*> dev_y(nm);
        for (int m = 0; m < nm; ++m) {
            const int n_out = cs.n_out[m];
            std::vector<float> w((size_t) n_out * n_in);
            for (float& v : w) v = nd(rng);
            blocks[m].resize((size_t) n_out * nb * 18);
            for (int r = 0; r < n_out; ++r) quant_q4_0(&w[(size_t) r * n_in], n_in, &blocks[m][(size_t) r * nb * 18]);
            uint8_t* tmp;
            cudaMalloc((void**) &tmp, blocks[m].size());
            cudaMemcpy(tmp, blocks[m].data(), blocks[m].size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &dev_w[m], blocks[m].size());
            strata::kernels::df2_q4_tile(tmp, dev_w[m], dev_w[m] + (size_t) n_out * n_in / 2, n_in, (int64_t) n_out * nb, nullptr);
            cudaDeviceSynchronize();
            cudaFree(tmp);
            cudaMalloc((void**) &dev_y[m], (size_t) 8 * n_out * 4);
        }
        std::vector<float> x((size_t) 8 * n_in);
        for (float& v : x) v = nd(rng) * 4.0f;
        float* dx;
        int8_t* dxq;
        float* dxd;
        cudaMalloc((void**) &dx, x.size() * 4);
        cudaMalloc((void**) &dxq, (size_t) 8 * n_in + 4096);
        cudaMalloc((void**) &dxd, (size_t) 8 * nb * 4 + 64);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::df2_quant_rows(dx, n_in, 8, dxq, dxd, nullptr);
        cudaDeviceSynchronize();
        std::vector<int8_t> xq((size_t) 8 * n_in);
        std::vector<float> xd((size_t) 8 * nb);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xd.data(), dxd, xd.size() * 4, cudaMemcpyDeviceToHost);
        for (int ncols : {1, 2, 3, 5, 8}) {
            strata::kernels::Df2Mat mats[3];
            for (int m = 0; m < nm; ++m) mats[m] = strata::kernels::Df2Mat{dev_w[m], dev_y[m], cs.n_out[m], true};
            strata::kernels::df2_gemm(mats, nm, dxq, dxd, n_in, ncols, nullptr);
            cudaDeviceSynchronize();
            double worst = 0, big = 0;
            for (int m = 0; m < nm; ++m) {
                const int n_out = cs.n_out[m];
                std::vector<float> y((size_t) 8 * n_out);
                cudaMemcpy(y.data(), dev_y[m], y.size() * 4, cudaMemcpyDeviceToHost);
                for (int c = 0; c < ncols; ++c)
                    for (int r = 0; r < n_out; ++r) {
                        double ref = 0;
                        for (int b = 0; b < nb; ++b) {
                            const uint8_t* blk = &blocks[m][((size_t) r * nb + b) * 18];
                            const uint16_t dh = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
                            long isum = 0;
                            for (int i = 0; i < 16; ++i) {
                                isum += (long) ((blk[2 + i] & 15) - 8) * xq[(size_t) c * n_in + b * 32 + i];
                                isum += (long) ((blk[2 + i] >> 4) - 8) * xq[(size_t) c * n_in + b * 32 + 16 + i];
                            }
                            ref += (double) h2f(dh) * (double) xd[(size_t) c * nb + b] * (double) isum;
                        }
                        const double got = y[(size_t) c * n_out + r];
                        worst = std::max(worst, std::fabs(got - ref));
                        big = std::max(big, std::fabs(ref));
                    }
            }
            const bool ok = worst <= 2e-5 * std::max(big, 1e-9);
            ++total;
            if (!ok) ++bad;
            std::printf("n_in %5d mats %d rows %5d cols %d: max |kernel - cpu| %.3g of %.3g (%.2g)  %s\n", n_in, nm, cs.n_out[0], ncols, worst, big, worst / std::max(big, 1e-9), ok ? "ok" : "MISMATCH");
        }
        for (int m = 0; m < nm; ++m) { cudaFree(dev_w[m]); cudaFree(dev_y[m]); }
        cudaFree(dx); cudaFree(dxq); cudaFree(dxd);
    }
    std::printf("%s: %d of %d cases match the CPU reference\n", bad == 0 ? "PASS" : "FAIL", total - bad, total);
    return bad == 0 ? 0 : 1;
}
