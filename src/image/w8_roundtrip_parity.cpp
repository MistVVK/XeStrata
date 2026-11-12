// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// W8A8's regular Hadamard quantization and training dequantization on synthetic FP16 rows.
#include "strata/image/train_ops.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "../kernels/parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    namespace p = strata::parity;
    namespace k = strata::kernels;
    namespace t = strata::image::train;
    constexpr int N = 7, K = 768;
    constexpr size_t kNK = (size_t) N * K;
    std::mt19937 rng(1201);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<uint16_t> input(kNK), output(kNK);
    for (auto& x : input) x = k::f16_from_f32(dist(rng));
    std::vector<float> scales(N);
    p::Device dev;
    auto* x = dev.upload(input);
    auto* q = dev.alloc<int8_t>(kNK);
    auto* s = dev.alloc<float>(N);
    auto* y = dev.alloc<uint16_t>(kNK);
    t::w8_quantize(x, N, K, q, s, p::stream());
    t::w8_dequant_rows(q, s, N, K, y, p::stream());
    p::copy_out(output.data(), y, output.size());
    p::copy_out(scales.data(), s, scales.size());
    for (int r = 0; r < N; ++r) for (int c = 0; c < K; ++c) {
        const float a = k::f32_from_f16(input[r * K + c]);
        const float b = k::f32_from_f16(output[r * K + c]);
        // Each inverse Hadamard output sums 256 rounding errors of at most half a step, divided by 16.
        // The final FP16 rounding is at most one unit in the last place (including subnormals).
        const float bound = 8.f * scales[r] + std::max(std::abs(b) / 1024.f, 0x1p-24f);
        if (!std::isfinite(b) || std::abs(a - b) > bound) {
            std::fprintf(stderr, "row %d col %d: %g vs %g (bound %g)\n", r, c, a, b, bound);
            return 1;
        }
    }
    return 0;
}
