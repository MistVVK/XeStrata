// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/native_multi_parity.cpp - the multi-token router and the combine, bitwise against their single-token
// contracts (#783 PR-c).
//
//   * native_router_top10_multi against native_router_top10 token by token, for n = 1..19, on plain logits, on
//     logits with exact ties and on logits with NaN (the router maps NaN to -FLT_MAX): ids and weights are compared
//     with memcmp;
//   * native_moe_combine / native_moe_combine_multi (k = 10 and 9, with and without a shared row, an aligned and a
//     misaligned output; upstream's float4 k = 10 kernel measured slower on the B70 and is not here) against a host
//     replay of the documented contract (src/kernels/xe/native_moe.cpp, not CUDA's FMA): every product rounds to
//     F32 and is added in expert order, the shared row once: memcmp.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"

#include "parity_device.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;

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
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           std::memcmp(static_cast<const void*>(a.data()), static_cast<const void*>(b.data()), a.size() * sizeof(T)) == 0;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(783);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;
    void* st = P::stream();

    // ---- the multi-token router
    for (int mode = 0; mode < 3; ++mode) {
        const char* names[] = {"random logits", "exact ties", "NaN logits"};
        int mode_bad = 0, cases = 0;
        for (int n = 1; n <= 19; ++n) {
            std::vector<float> lg((size_t) n * 512);
            for (auto& v : lg) v = gauss(rng) * 2.0f;
            if (mode == 1)
                for (float& v : lg) v = (float) (int) (v * 2.0f) * 0.5f;   // coarse: many exact ties
            if (mode == 2)
                for (int t = 0; t < n; ++t) lg[(size_t) t * 512 + (size_t) (rng() % 512)] = std::nanf("");
            float* d_l = up(lg);
            auto* d_i = static_cast<int32_t*>(P::alloc_bytes((size_t) n * 10 * 4));
            auto* d_ir = static_cast<int32_t*>(P::alloc_bytes((size_t) n * 10 * 4));
            auto* d_w = static_cast<float*>(P::alloc_bytes((size_t) n * 10 * 4));
            auto* d_wr = static_cast<float*>(P::alloc_bytes((size_t) n * 10 * 4));
            native_router_top10_multi(d_l, d_i, d_w, n, st);
            for (int t = 0; t < n; ++t) native_router_top10(d_l + (size_t) t * 512, d_ir + (ptrdiff_t) t * 10, d_wr + (ptrdiff_t) t * 10, st);
            P::sync();
            ++cases;
            if (!same(down(d_i, (size_t) n * 10), down(d_ir, (size_t) n * 10)) ||
                !same(down(d_w, (size_t) n * 10), down(d_wr, (size_t) n * 10))) {
                std::printf("    *** native_router_top10_multi n=%d (%s) differs from native_router_top10 ***\n", n, names[mode]);
                ++mode_bad;
            }
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("router multi == single, ") + names[mode]).c_str(),
                    mode_bad ? "*** NO ***" : "bitwise", cases);
        bad += mode_bad;
    }

    // ---- the combine
    struct Cfg { const char* name; int k; bool shared; int out_off; };
    const Cfg cfgs[] = {{"k=10 + shared", 10, true, 0}, {"k=10 + shared, output misaligned", 10, true, 1},
                        {"k=10 without shared", 10, false, 0}, {"k=9 + shared", 9, true, 0}};
    const int N = 2048;
    for (const Cfg& c : cfgs) {
        int cfg_bad = 0, cases = 0;
        for (int n = 1; n <= 8; ++n) {
            std::vector<float> parts((size_t) n * c.k * N), w((size_t) n * c.k), sh((size_t) n * N);
            for (auto& v : parts) v = gauss(rng);
            for (auto& v : w) v = (float) (rng() % 1000) / 1000.0f;
            for (auto& v : sh) v = gauss(rng);
            float* d_p = up(parts);
            float* d_w = up(w);
            float* d_s = up(sh);
            auto* d_o = static_cast<float*>(P::alloc_bytes((size_t) n * N * 4 + 64));
            float* out = d_o + c.out_off;
            native_moe_combine_multi(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, n, st);
            P::sync();
            const auto got = down(out, (size_t) n * N);
            std::vector<float> want((size_t) n * N);
            for (int t = 0; t < n; ++t)
                for (int col = 0; col < N; ++col) {
                    float s = parts[((size_t) t * c.k) * N + col] * w[(size_t) t * c.k];
                    for (int e = 1; e < c.k; ++e) {
                        const float product = parts[((size_t) t * c.k + e) * N + col] * w[(size_t) t * c.k + e];
                        s = s + product;
                    }
                    if (c.shared) s += sh[(size_t) t * N + col];
                    want[(size_t) t * N + col] = s;
                }
            ++cases;
            if (!same(got, want)) {
                std::printf("    *** native_moe_combine_multi n=%d (%s) differs from the contract's host replay ***\n", n, c.name);
                ++cfg_bad;
            }
            if (n == 1) {   // the single-token call is the same arithmetic
                const std::vector<float> zero((size_t) N + 16, 0.0f);
                P::copy_bytes_in(d_o, zero.data(), zero.size() * 4);
                native_moe_combine(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, st);
                P::sync();
                ++cases;
                if (!same(down(out, (size_t) N), want)) {
                    std::printf("    *** native_moe_combine (%s) differs from the contract's host replay ***\n", c.name);
                    ++cfg_bad;
                }
            }
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("combine ") + c.name).c_str(), cfg_bad ? "*** NO ***" : "bitwise", cases);
        bad += cfg_bad;
    }

    std::printf("\nnative_multi: %d failures\n", bad);
    if (bad) return 1;
    std::printf("native_multi_parity OK\n");
    return 0;
}
