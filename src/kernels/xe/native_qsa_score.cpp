// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_qsa_score.cpp - the Xe port of Strata's src/kernels/cuda/native_qsa_score.cu: the QSA indexer's block
// scores (four ReLU'd head dots, summed, plus the block bias and the partial-block mask), after ggml-cuda's mmf,
// unary and binbcast at pinned llama.cpp 3cf03257 (MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE).
//
// The CUDA kernel's main path is a TF32 mma.sync, which drops the low mantissa bits of its inputs and sums inside
// the tensor core in an order Xe cannot reproduce.  This port is the CUDA source's own fallback for devices without
// TF32 mma: one row per work-item, the head dots as FP32 fma chains.  Its rounding differs from the tensor-core
// path, as the CUDA source says of that fallback.
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/core/runtime.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int D = 128, HEADS = 4, R = 4, ROWS = 32;

struct Span { const void* p; size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p) throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}

}  // namespace

void native_qsa_score_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_score(const float* pooled, const float* query, const float* bias,
                      const QsaShapes& s, const int32_t* step, int64_t max_blocks, int64_t max_cells,
                      float* cells, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 ||
        max_cells < 1 || max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    if (!stream) throw core::DeviceError("native QSA score requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    const Span spans[] = {{pooled, size_t(max_blocks) * D * 4}, {query, HEADS * D * 4},
        {step, kStepCount * 4}, {cells, size_t(max_cells) * 4}, {bias, bias ? size_t(max_blocks) * 4 : 0}};
    const int count = bias ? 5 : 4;
    for (int i = 0; i < count; ++i) validate(spans[i]);
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");
    const int mc = int(max_cells);
    const size_t groups = size_t((max_blocks + ROWS - 1) / ROWS);
    queue.parallel_for(sycl::range<1>(groups * ROWS), [=](sycl::id<1> id) {
        const int n = step[kStepNKv], full = step[kStepNBid];
        if (n < 1 || n > mc || step[kStepPos] != n - 1 || full != n / R || step[kStepWidth] != (n < 2051 ? n : 2051))
            return;
        const int row = int(id[0]);
        if (row > full) return;
        float h[HEADS];
        for (int j = 0; j < HEADS; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc = sycl::fma(pooled[size_t(row) * D + d], query[j * D + d], acc);
            h[j] = sycl::fmax(acc, 0.0f);
        }
        float sum = ((h[0] + h[1]) + h[2]) + h[3];
        if (bias) sum = sum + bias[row];
        sum = sum + (row == full && n % R ? 1e9f : 0.0f);
        for (int i = row * R; i < n && i < (row + 1) * R; ++i) cells[i] = sum;
    });
}

}  // namespace strata::kernels
