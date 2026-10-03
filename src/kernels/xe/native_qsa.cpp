// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_qsa.cpp - the Xe port of Strata's src/kernels/cuda/native_qsa.cu: the QSA RMS norm with a per-column
// weight and the sigmoid output gate, with the arithmetic of ggml-cuda's norm.cu and unary.cu at pinned llama.cpp
// 3cf03257 (MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// CUDA's expf and rsqrtf under fast math are the precise exp and SYCL's rsqrt here.
#include "strata/kernels/native_qsa.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

template <int BlockSize>
void norm(sycl::queue& q, const float* input, const float* gamma, float* output, int n_cols, int n_rows, float epsilon) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(std::size_t(n_rows) * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = int(it.get_local_id(0));
            const std::size_t row_offset = it.get_group(0) * std::size_t(n_cols);
            const float* in = input + row_offset;
            float* out = output + row_offset;
            float partial = 0.0f;
            for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
                const float x = in[col];
                partial += x * x;
            }
            partial = warp_sum(sg, partial);
            const int lane = tid % 32;
            if (lane == 0) sums[tid / 32] = partial;
            // all reads of the input for the reduction precede this barrier, so the operation may run in place
            sycl::group_barrier(it.get_group());
            partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
            partial = warp_sum(sg, partial);
            const float mean = partial / n_cols;
            const float scale = sycl::rsqrt(mean + epsilon);
            for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) out[col] = scale * in[col] * gamma[col];
        });
    });
}

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
sycl::queue& queue_for(void* stream) {
    if (!stream) throw core::DeviceError("native QSA requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

}  // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    auto& q = queue_for(stream);
    // a device that takes no 1024-item work-group (the UHD 770: 512) sums the long rows in another order
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024(q);
    if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon);
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    queue_for(stream).parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
        const std::size_t i = id[0], head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-raw));
        output[i] = attn[i] * sigmoid;
    });
}

}  // namespace strata::kernels
