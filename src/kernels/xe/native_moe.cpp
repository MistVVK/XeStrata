// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_moe.cpp - the Xe port of Strata's src/kernels/cuda/native_moe.cu: the MoE weighted combination with
// the arithmetic of ggml-cuda's moe-weighted-reduction.cu at pinned llama.cpp 3cf03257 (MIT License, Copyright (c)
// 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE): parts[0] * w[0], then each further expert's product
// added in order, then the shared output.
#include "strata/kernels/native_moe.hpp"
#include "strata/core/runtime.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

void launch(void* stream, const float* parts, const float* weights, const float* shared, float* output,
            int64_t n_embd, int k, int n_tok, const float* shared_gate = nullptr) {
    if (!stream) throw core::DeviceError("native MoE combine requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    queue.parallel_for(sycl::range<2>(size_t(n_tok), size_t(n_embd)), [=](sycl::id<2> id) {
        const int64_t tk = int64_t(id[0]), col = int64_t(id[1]);
        const float* p = parts + tk * k * n_embd;
        const float* w = weights + tk * k;
        float sum = p[col] * w[0];
        for (int expert = 1; expert < k; ++expert) sum += p[int64_t(expert) * n_embd + col] * w[expert];
        if (shared) {
            float sv = shared[tk * n_embd + col];
            if (shared_gate) {   // shared_expert's sigmoid_scale_rows, its product rounded before the add
                const float gt = 1.0f / (1.0f + sycl::exp(-shared_gate[tk]));
                sv = sv * gt;
            }
            sum += sv;
        }
        output[tk * n_embd + col] = sum;
    });
}

}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    launch(stream, parts, weights, shared, output, n_embd, int(k), 1);
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream, const float* shared_gate) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || (shared_gate && !shared))
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    launch(stream, parts, weights, shared, output, n_embd, int(k), n_tok, shared_gate);
}

}  // namespace strata::kernels
