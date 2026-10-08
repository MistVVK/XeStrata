// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gr_multi_parity.cpp - the multi-token hyper-connection write (#783 PR-g: gr_write_multi) against the
// single-token calls, memcmp.
//
// Real geometry (n_embd 2560, hc 4), the pinned native MMVF path, every T = 1..8, the write both out of place and in
// place (R_out == R).  Each output of the multi call must equal the token-by-token one bit for bit.  GPU, synthetic,
// no model.  (Upstream's gr_read_multi half is not here: on Xe the window's reads are fused_gr_read_multi.)
#include "strata/kernels/gr.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {

template <typename T> T* up(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}

}  // namespace

int main() {
    const long long n_embd = 2560, hc = 4, hc_lr = 320, hc_dim = hc * n_embd;
    std::mt19937 rng(783);  // NOLINT(bugprone-random-generator-seed): reproducible rows.
    std::normal_distribution<float> gauss(0.f, 1.f);
    k::gr_set_native_mmvf(true);
    const k::GrShapes sh{n_embd, hc, hc_lr};
    void* st = P::stream();
    int cases = 0, bad = 0;
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> R((size_t) T * hc_dim);
        for (int t = 0; t < T; ++t)
            for (long long c = 0; c < hc; ++c)
                for (long long d = 0; d < n_embd; ++d)
                    R[(size_t) t * hc_dim + c * n_embd + d] = gauss(rng) * (float) (1 << (2 * c));
        // the write: R_out[t] = R[t] + block_out[t] * 2 sigmoid(inject[t] / hc)
        std::vector<float> bo((size_t) T * n_embd), inj((size_t) T * hc);
        for (auto& x : bo) x = gauss(rng);
        for (auto& x : inj) x = gauss(rng) * 3.0f;
        float* d_bo = up(bo);
        float* d_in = up(inj);
        const std::vector<float> zero((size_t) T * hc_dim, 0.0f);
        for (int inplace = 0; inplace < 2; ++inplace) {
            float* d_Rm = up(R);
            float* d_Rs = up(R);
            float* d_om = inplace ? d_Rm : up(zero);
            float* d_os = inplace ? d_Rs : up(zero);
            k::gr_write_multi(d_Rm, d_bo, d_in, sh, d_om, T, st);
            for (int t = 0; t < T; ++t)
                k::gr_write(d_Rs + (size_t) t * hc_dim, d_bo + (size_t) t * n_embd, d_in + (size_t) t * hc, sh,
                            d_os + (size_t) t * hc_dim, st);
            P::sync();
            const std::vector<float> a = down(d_om, (size_t) T * hc_dim), b = down(d_os, (size_t) T * hc_dim);
            const bool ok = std::memcmp(static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
                                        a.size() * sizeof(float)) == 0;
            ++cases;
            if (!ok) {
                std::printf("    *** gr_write_multi T=%d %s differs from gr_write ***\n", T, inplace ? "in place" : "out of place");
                ++bad;
            }
        }
    }
    std::printf("gr_multi: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("gr_multi_parity OK\n");
    return 0;
}
