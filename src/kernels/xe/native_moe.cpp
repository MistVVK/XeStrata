// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_moe.cpp - the Xe port of Strata's src/kernels/cuda/native_moe.cu: the MoE weighted combination with
// the arithmetic of ggml-cuda's moe-weighted-reduction.cu at pinned llama.cpp 3cf03257 (MIT License, Copyright (c)
// 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE): parts[0] * w[0], then each further expert's product
// added in order, then the shared output.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_moe
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/kernels/rows_cols.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

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

// The one combination of the routed experts' outputs, for the decode, the MTP head and the prompt alike: one output
// element, part(e) expert e's value for it, shared its shared output (or null), gate its raw gate; in FP32 as ggml's,
// parts[0] w[0], then each further expert's product, then the shared output.
template <typename Part>
inline float combine(Part part, const float* w, int k, const float* shared, const float* gate) {
    float sum = part(0) * w[0];
    for (int expert = 1; expert < k; ++expert) sum += part(expert) * w[expert];
    if (shared) {
        float sv = *shared;
        if (gate) sv = sv * (1.0f / (1.0f + sycl::exp(-*gate)));   // sigmoid_scale_rows, rounded before the add
        sum += sv;
    }
    return sum;
}

// slot: the parts' rows by (token, expert), slot[tk k + e] (Qwen3.8-Flash-Next's prompt: its experts' outputs in
// expert order), else token tk's k rows in turn
sycl::event launch(sycl::queue& queue, const float* parts, const float* weights, const float* shared, float* output,
                   int64_t n_embd, int k, int n_tok, const float* shared_gate = nullptr,
                   const int32_t* slot = nullptr) {
    return for_rows_cols(queue, n_tok, n_embd, [=](int64_t tk, int64_t col) {
        output[tk * n_embd + col] = combine([&](int e) {
            return parts[(slot ? int64_t(slot[tk * k + e]) : tk * k + e) * n_embd + col];
        }, weights + tk * k, k, shared ? shared + tk * n_embd + col : nullptr, shared_gate ? shared_gate + tk : nullptr);
    });
}
sycl::queue& explicit_stream(void* stream) {
    if (!stream) throw core::DeviceError("native MoE combine requires an explicit stream");
    return core::Runtime::get().stream(stream);
}

}  // namespace

bool native_moe_fuse_hits() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_FUSE_HITS");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

void native_moe_combine_multi_hits(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                   const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                   int n_tok, void* stream, const float* shared_gate) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || (shared_gate && !shared) || !hits || !dst || !count)
        throw std::invalid_argument("native MoE combine (hits) requires a stream, width, 1..15 experts, tokens, hits");
    constexpr int WG = 256;
    const int cap = n_tok * (int) k;
    const size_t cols = (size_t) ((n_embd + WG - 1) / WG) * WG;
    core::Runtime::get().stream(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> hit(sycl::range<1>((size_t) cap), h);   // entry -> its part has a hit row
        h.parallel_for(sycl::nd_range<2>({size_t(n_tok), cols}, {1, WG}), [=](sycl::nd_item<2> it) {
            const int lid = (int) it.get_local_id(1);
            for (int i = lid; i < cap; i += WG) hit[i] = 0;
            sycl::group_barrier(it.get_group());
            const int n_hit = *count;
            for (int i = lid; i < n_hit; i += WG)
                if (dst[i] >= 0 && dst[i] < cap) hit[dst[i]] = 1;
            sycl::group_barrier(it.get_group());
            const int64_t tk = int64_t(it.get_group(0)), col = int64_t(it.get_global_id(1));
            if (col >= n_embd) return;
            const float* p = parts + tk * k * n_embd;
            const float* hp = hits + tk * k * n_embd;
            const uint8_t* ht = &hit[(size_t) (tk * k)];
            output[tk * n_embd + col] = combine([&](int e) {
                const int64_t o = int64_t(e) * n_embd + col;
                return ht[e] ? p[o] + hp[o] : p[o];   // moe_hit_add's sum, then its product
            }, weights + tk * k, int(k), shared ? shared + tk * n_embd + col : nullptr,
               shared_gate ? shared_gate + tk : nullptr);
        });
    });
}

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
    launch(explicit_stream(stream), parts, weights, shared, output, n_embd, int(k), 1);
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream, const float* shared_gate,
                              const int32_t* slot) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || (shared_gate && !shared))
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    launch(explicit_stream(stream), parts, weights, shared, output, n_embd, int(k), n_tok, shared_gate, slot);
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated
    if (k > 64) throw core::DeviceError("moe_combine: k = " + std::to_string(k) + " exceeds 64");
    auto& q = core::Runtime::get().stream(stream);
    const auto e = launch(q, parts, weights, shared, y, n_embd, int(k), 1);
    if (!stream) core::Runtime::get().wait(e, "moe_combine");
}

}  // namespace strata::kernels
