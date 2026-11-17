// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/kernels/latent_parity.cpp - the latent MoE layer (Edge0's D -> A -> U -> combine) on the engine's kernels,
// against the references of latent_ref.hpp, at both published latent widths (1792 and 1280) and both D/U formats
// (Q8_0 and Q4_0): iq_mmvq (D), latent_colscale_rows, quantize_q8_1_rows, native_expert_grouped (A, Q3_K gate/up
// and IQ4_NL down at the latent width), latent_u_grouped (U), the host combine.  latent_plan_rows is checked entry
// by entry against the host rule.  No GPU: skipped (77).  The references alone: latent_ref_test (no GPU needed).
//
// The A stage's own dot products are native_expert_parity's subject (--synthetic q3_K/iq4_nl@1792x640); here it
// is the layer's composition and the latent-specific kernels.
#include "strata/kernels/latent.hpp"

#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "latent_ref.hpp"
#include "parity_device.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace par = strata::parity;
namespace lr = strata::latent_ref;

namespace {

int g_fail = 0;
void report(const char* what, double rel, double bound) {
    const bool ok = rel <= bound;
    std::printf("  %-72s rel %.3e %s\n", what, rel, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the layer on the engine's kernels
std::vector<float> layer_gpu(const lr::Fixture& f, void* s) {
    const int64_t H = f.H, ELL = f.ELL, I = f.I;
    const int G = f.G, T = f.T, K = f.K;
    par::Device dev;
    const float* dx = dev.upload(f.x);
    const uint8_t* dD = dev.upload(f.D);
    const uint8_t* dU = dev.upload(f.U);
    const float* dinv = dev.upload(f.inv);
    const int32_t* dids = dev.upload(f.ids);
    const int32_t* dlabels = dev.upload(f.labels);
    std::vector<unsigned long long> ptrs((size_t) f.E);
    for (int e = 0; e < f.E; ++e) ptrs[(size_t) e] = (unsigned long long) dev.upload(f.blobs[(size_t) e]);
    const unsigned long long* dptr = dev.upload(ptrs);
    // the plan: the routes in expert order, one group an expert
    std::vector<int32_t> order((size_t) f.cap), start((size_t) (f.E + 1), 0);
    for (int64_t r = 0; r < f.cap; ++r) start[(size_t) f.ids[(size_t) r] + 1]++;
    for (int e = 0; e < f.E; ++e) start[(size_t) e + 1] += start[(size_t) e];
    {
        std::vector<int32_t> pos(start.begin(), start.end() - 1);
        for (int64_t r = 0; r < f.cap; ++r) order[(size_t) pos[(size_t) f.ids[(size_t) r]]++] = (int32_t) r;
    }
    const int32_t* ddst = dev.upload(order);
    const int32_t* dstart = dev.upload(start);
    const int32_t* dn = dev.upload(std::vector<int32_t>{f.E});
    const int32_t* dcounts = dev.upload(std::vector<int32_t>{0, (int32_t) f.cap});
    int32_t* dtok = dev.alloc<int32_t>((size_t) f.cap);
    k::latent_plan_rows(dtok, ddst, dcounts, dids, dlabels, f.E, 0, G, K, s);
    {   // the plan rows against the host rule, entry by entry
        std::vector<int32_t> tok((size_t) f.cap), want((size_t) f.cap);
        par::copy_out(tok.data(), dtok, tok.size());
        for (int64_t j = 0; j < f.cap; ++j) {
            const int32_t r = order[(size_t) j];
            want[(size_t) j] = (r / K) * G + f.labels[(size_t) f.ids[(size_t) r]];
        }
        const bool same = tok == want;
        std::printf("  %-72s %s\n", "latent_plan_rows: every entry's xl row", same ? "ok" : "FAIL");
        if (!same) ++g_fail;
    }
    uint8_t* dxq = dev.alloc<uint8_t>((size_t) (T * (H / 32) * 36));
    uint8_t* dxlq = dev.alloc<uint8_t>((size_t) T * G * (ELL / 32) * 36);
    float* dxl = dev.alloc<float>((size_t) T * G * ELL);
    k::quantize_q8_1_rows(dx, T, H, dxq, s);
    k::iq_mmvq(f.fmt == 4 ? 2 : 8, dD, dxq, dxl, (int) H, (int) (G * ELL), T, s);
    k::latent_colscale_rows(dxl, (int64_t) T * G, (int) ELL, dinv, s);
    {   // the column scales on a fresh vector: the same float multiply, bit for bit
        std::vector<float> probe((size_t) T * G * ELL);
        for (size_t i = 0; i < probe.size(); ++i) probe[i] = (float) (i % 977) * 0.01f - 4.f;
        float* dmut = dev.alloc<float>(probe.size());
        par::copy_in(dmut, probe.data(), probe.size());
        k::latent_colscale_rows(dmut, (int64_t) T * G, (int) ELL, dinv, s);
        std::vector<float> got(probe.size());
        par::copy_out(got.data(), dmut, got.size());
        bool same = true;
        for (size_t i = 0; i < probe.size(); ++i) {
            const float w = probe[i] * f.inv[i % (size_t) ELL];
            same = same && std::memcmp(&got[i], &w, 4) == 0;  // NOLINT(bugprone-suspicious-memory-comparison): bit-exact
        }
        std::printf("  %-72s %s\n", "latent_colscale_rows: bit for bit", same ? "ok" : "FAIL");
        if (!same) ++g_fail;
    }
    k::quantize_q8_1_rows(dxl, (int64_t) T * G, ELL, dxlq, s);
    const auto L = k::native_expert_layout(11, 20, ELL, I);
    uint8_t* dscr = dev.alloc<uint8_t>(k::native_expert_scratch_bytes(f.cap, I));
    float* dyl = dev.alloc<float>((size_t) (f.cap * ELL));
    k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, f.E, f.cap, dxlq, dscr, dyl, s);
    uint8_t* dylq = dev.alloc<uint8_t>((size_t) (f.cap * (ELL / 32) * 36));
    k::quantize_q8_1_rows(dyl, f.cap, ELL, dylq, s);
    float* dy = dev.alloc<float>((size_t) (f.cap * H));
    k::latent_u_grouped(dU, dylq, dids, dlabels, f.E, f.cap, (int) ELL, (int) H, dy, s, f.fmt);
    par::sync();
    {   // the U kernel against the scalar integer chains on the SAME q8_1 rows
        std::vector<uint8_t> ylq((size_t) (f.cap * (ELL / 32) * 36));
        std::vector<float> y((size_t) (f.cap * H)), want((size_t) (f.cap * H));
        par::copy_out(ylq.data(), dylq, ylq.size());
        par::copy_out(y.data(), dy, y.size());
        for (int64_t r = 0; r < f.cap; ++r) {
            const int e = f.ids[(size_t) r], g = f.labels[(size_t) e];
            for (int64_t h = 0; h < H; ++h)
                want[(size_t) (r * H + h)] = lr::row_dot(f.U.data() + ((size_t) g * H + (size_t) h) * f.u_row,
                                                         ylq.data() + (size_t) r * (ELL / 32) * 36, ELL, f.fmt);
        }
        report("latent_u_grouped vs the scalar integer chains", lr::rel(y, want), 1e-5);
    }
    std::vector<float> y((size_t) (f.cap * H));
    par::copy_out(y.data(), dy, y.size());
    std::vector<float> out((size_t) (T * H), 0.f);
    for (int64_t r = 0; r < f.cap; ++r)
        for (int64_t h = 0; h < H; ++h) out[(size_t) ((r / K) * H + h)] += f.w[(size_t) r] * y[(size_t) (r * H + h)];
    return out;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try { strata::core::Runtime::get(); }
    catch (const std::exception& e) { std::printf("latent_parity: no GPU (%s) - skipped\n", e.what()); return 77; }
    void* s = par::stream();
    for (int64_t ELL : {1792, 1280}) {
        for (int fmt : {8, 4}) {
            std::printf("ELL=%lld fmt=%d (H=2560 I=640 G=8 E=16 T=2 K=10)\n", (long long) ELL, fmt);
            const lr::Fixture f(ELL, fmt);
            const std::vector<float> ref = lr::layer_ref(f, false);
            const std::vector<float> cpu = lr::layer_ref(f, true);
            report("cpu reference vs float reference", lr::rel(cpu, ref), 1e-5);
            const std::vector<float> got = layer_gpu(f, s);
            report("gpu layer vs float reference", lr::rel(got, ref), 5e-4);
            report("gpu layer vs cpu reference", lr::rel(got, cpu), 5e-4);
        }
    }
    std::printf("latent_parity: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}