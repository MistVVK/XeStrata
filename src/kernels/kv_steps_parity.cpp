// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/kv_steps_parity.cpp - the batched K/V append (#783 PR-d): kv_append_q8_steps / kv_append_q4_steps for
// n = 1..8 cells in one launch against n calls of the single-cell append, memcmp on every pool byte.
//
// Cells land at scattered positions through a non-identity page table; the K/V rows are random with small, large and
// all-zero groups. Three layouts: INT8 K and V, Q4_0 K and V, and the K8V4 hybrid's folded call (the K pool passed as
// both halves, one plane). GPU, synthetic, no model.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
template <typename T> void zero(T* d, size_t n) {
    const std::vector<unsigned char> z(n * sizeof(T) + 64, 0);
    strata::parity::copy_bytes_in(d, z.data(), z.size());
}
template <typename T> T* dalloc(size_t n) {
    T* p = static_cast<T*>(strata::parity::alloc_bytes(n * sizeof(T) + 64));
    zero(p, n);
    return p;
}
template <typename T> std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    strata::parity::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
template <typename T> bool same(const T* a, const T* b, size_t n) {
    return std::memcmp(fetch(a, n).data(), fetch(b, n).data(), n * sizeof(T)) == 0;
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P, max_n = 8;
    std::mt19937 rng(783);  // NOLINT(bugprone-random-generator-seed): reproducible rows.
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    strata::parity::copy_bytes_in(d_table, table.data(), (size_t) pages * 4);

    const size_t q8n = (size_t) cells * H * D, q8s = (size_t) cells * H * G, q4n = (size_t) cells * H * D;  // q4: <= 144 B per cell-head
    int8_t *kq1 = dalloc<int8_t>(q8n), *vq1 = dalloc<int8_t>(q8n), *kq2 = dalloc<int8_t>(q8n), *vq2 = dalloc<int8_t>(q8n);
    uint16_t *ks1 = dalloc<uint16_t>(q8s), *vs1 = dalloc<uint16_t>(q8s), *ks2 = dalloc<uint16_t>(q8s), *vs2 = dalloc<uint16_t>(q8s);
    uint8_t *k41 = dalloc<uint8_t>(q4n), *v41 = dalloc<uint8_t>(q4n), *k42 = dalloc<uint8_t>(q4n), *v42 = dalloc<uint8_t>(q4n);
    uint8_t *h41 = dalloc<uint8_t>(q4n), *h42 = dalloc<uint8_t>(q4n);
    int8_t *hq1 = dalloc<int8_t>(q8n), *hq2 = dalloc<int8_t>(q8n);
    uint16_t *hs1 = dalloc<uint16_t>(q8s), *hs2 = dalloc<uint16_t>(q8s);
    float *d_k = dalloc<float>((size_t) max_n * H * D), *d_v = dalloc<float>((size_t) max_n * H * D);
    int32_t* d_step = dalloc<int32_t>((size_t) max_n * k::kStepCount);

    int total_cases = 0;
    for (int n = 1; n <= max_n; ++n) {
        for (int round = 0; round < 3; ++round) {
            zero(kq1, q8n); zero(vq1, q8n); zero(kq2, q8n); zero(vq2, q8n);
            zero(ks1, q8s); zero(vs1, q8s); zero(ks2, q8s); zero(vs2, q8s);
            zero(k41, q4n); zero(v41, q4n); zero(k42, q4n); zero(v42, q4n); zero(h41, q4n); zero(h42, q4n);
            zero(hq1, q8n); zero(hq2, q8n); zero(hs1, q8s); zero(hs2, q8s);
            std::vector<int> positions(cells);
            for (int i = 0; i < cells; ++i) positions[i] = i;
            std::shuffle(positions.begin(), positions.end(), rng);
            std::vector<float> hk((size_t) n * H * D), hv(hk.size());
            for (int t = 0; t < n; ++t) {
                const float scale = (t % 3 == 0) ? 1e-3f : (t % 5 == 0 ? 40.f : 1.f);
                for (int i = 0; i < H * D; ++i) {
                    hk[(size_t) t * H * D + i] = nd(rng) * scale;
                    hv[(size_t) t * H * D + i] = nd(rng) * scale;
                }
                if (t % 4 == 1) std::fill(hk.begin() + (ptrdiff_t) t * H * D, hk.begin() + (ptrdiff_t) t * H * D + 64, 0.f);
            }
            std::vector<int32_t> hstep((size_t) n * k::kStepCount, 0);
            for (int t = 0; t < n; ++t) {
                hstep[(size_t) t * k::kStepCount + k::kStepPos] = positions[t];
                hstep[(size_t) t * k::kStepCount + 1] = positions[t] + 1;
            }
            strata::parity::copy_bytes_in(d_k, hk.data(), hk.size() * 4);
            strata::parity::copy_bytes_in(d_v, hv.data(), hv.size() * 4);
            strata::parity::copy_bytes_in(d_step, hstep.data(), hstep.size() * 4);

            for (int t = 0; t < n; ++t) {
                const int32_t* st = d_step + (size_t) t * k::kStepCount;
                const float* kc = d_k + (size_t) t * H * D;
                const float* vc = d_v + (size_t) t * H * D;
                k::kv_append_q8_step(kq1, vq1, ks1, vs1, d_table, st, kc, vc, s, nullptr);
                k::kv_append_q4_step(k41, v41, d_table, st, kc, vc, s, nullptr);
                k::kv_append_q8_step(hq1, hq1, hs1, hs1, d_table, st, kc, kc, s, nullptr);      // the hybrid's folded K call
                k::kv_append_q4_step(h41, h41, d_table, st, vc, vc, s, nullptr);                // ... and its folded V call
            }
            k::kv_append_q8_steps(kq2, vq2, ks2, vs2, d_table, d_step, k::kStepCount, d_k, d_v, H * D, n, s, nullptr, nullptr);
            k::kv_append_q4_steps(k42, v42, d_table, d_step, k::kStepCount, n, d_k, d_v, s, nullptr);
            k::kv_append_q8_steps(hq2, hq2, hs2, hs2, d_table, d_step, k::kStepCount, d_k, d_k, H * D, n, s, nullptr, nullptr);
            k::kv_append_q4_steps(h42, h42, d_table, d_step, k::kStepCount, n, d_v, d_v, s, nullptr);
            strata::parity::sync();

            struct Chk { const char* name; bool ok; };
            const Chk chks[] = {
                {"INT8 K/V codes", same(kq1, kq2, q8n) && same(vq1, vq2, q8n)},
                {"INT8 K/V scales", same(ks1, ks2, q8s) && same(vs1, vs2, q8s)},
                {"Q4_0 K/V", same(k41, k42, q4n) && same(v41, v42, q4n)},
                {"hybrid K (folded INT8)", same(hq1, hq2, q8n) && same(hs1, hs2, q8s)},
                {"hybrid V (folded Q4_0)", same(h41, h42, q4n)},
            };
            for (const Chk& c : chks) {
                ++total_cases;
                if (!c.ok) {
                    std::printf("    *** n=%d round %d: %s differ from the per-cell appends ***\n", n, round, c.name);
                    ++g_fail;
                }
            }
        }
    }
    std::printf("kv_steps: %d checks, %d failures\n", total_cases, g_fail);
    if (g_fail) return 1;
    std::printf("kv_steps_parity OK\n");
    return 0;
}
