// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// tests/kernels/latent_ref_test.cpp - the latent MoE layer's references against each other, no GPU: the float
// reference (ggml's dequantizers, double accumulation) against the CPU reference (the scalar integer block chains
// the U kernel uses), at both published latent widths and both D/U formats.  The GPU parity is latent_parity's.
#include "latent_ref.hpp"

#include <cstdio>

namespace lr = strata::latent_ref;

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int fail = 0;
    for (int64_t ELL : {1792, 1280}) {
        for (int fmt : {8, 4}) {
            std::printf("ELL=%lld fmt=%d (H=2560 I=640 G=8 E=16 T=2 K=10)\n", (long long) ELL, fmt);
            const lr::Fixture f(ELL, fmt);
            const std::vector<float> ref = lr::layer_ref(f, false);
            const std::vector<float> cpu = lr::layer_ref(f, true);
            const double r = lr::rel(cpu, ref);
            const bool ok = r <= 1e-5;
            std::printf("  %-72s rel %.3e %s\n", "cpu reference vs float reference", r, ok ? "ok" : "FAIL");
            if (!ok) ++fail;
        }
    }
    std::printf("latent_ref_test: %d failures\n", fail);
    return fail == 0 ? 0 : 1;
}