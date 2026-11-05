// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_qsa.cpp - the Xe port of Strata's src/kernels/cuda/native_qsa.cu: the QSA RMS norm with a per-column
// weight and the sigmoid output gate, with the arithmetic of ggml-cuda's norm.cu and unary.cu at pinned llama.cpp
// 3cf03257 (MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// CUDA's expf and rsqrtf under fast math are the precise exp and SYCL's rsqrt here.
#include "strata/kernels/native_qsa.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#include "q8_1_finite.hpp"
#include "strata/kernels/f16_gpu.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_qsa
#include "strata/kernels/device_arch_tag.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

template <int BlockSize> class NormKernel;   // asked for its work-group (work_group_upto_1024)

namespace {

std::atomic<bool> enabled{false};
constexpr int WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}
inline float warp_max(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset; offset >>= 1) value = sycl::fmax(value, sycl::permute_group_by_xor(sg, value, offset));
    return value;
}

// native_mmvq.cpp's activation block: the scale and the sum as fp16, then the 32 values
struct Q81 { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81) == 36, "a q8_1 block is 36 bytes");

// gamma: `groups` rows of n_cols, row r takes gamma row r % groups (1: one gamma for every row)
// q8 (n_cols a multiple of 32): the output also as q8_1 blocks, each the 32 columns one sub-group writes in a pass,
// with native_quantize_q8_1's reductions over the stored values, so the blocks are that kernel's bit for bit
// gamma null: no weight; out16: the output as FP16 instead of `output` (null then)
template <int BlockSize>
void norm(sycl::queue& q, const float* input, const float* gamma_all, float* output, int n_cols, int n_rows,
          float epsilon, int groups = 1, void* q8 = nullptr, uint16_t* out16 = nullptr) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for<NormKernel<BlockSize>>(sycl::nd_range<1>(std::size_t(n_rows) * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = int(it.get_local_id(0));
            const std::size_t row_offset = it.get_group(0) * std::size_t(n_cols);
            const float* gamma =
                gamma_all ? gamma_all + (it.get_group(0) % std::size_t(groups)) * std::size_t(n_cols) : nullptr;
            const float* in = input + row_offset;
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
            Q81* blocks = static_cast<Q81*>(q8);
            for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
                const float v = gamma ? scale * in[col] * gamma[col] : scale * in[col];
                if (out16 != nullptr) {
                    out16[row_offset + col] = f16_gpu::f16_from_f32(v);
                    continue;
                }
                output[row_offset + col] = v;
                if (blocks == nullptr) continue;
                const float amax = warp_max(sg, sycl::fabs(v));
                const float sum = warp_sum(sg, v);
                const float d = xe::q8_1_finite(amax / 127.0f);
                const std::size_t i = row_offset + col;
                blocks[i / 32].qs[i % 32] = xe::q8_1_quant(v, d, amax);
                if (lane == 0) blocks[i / 32].ds = sycl::half2(sycl::half(d), sycl::half(xe::q8_1_finite(sum)));
            }
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

void rms_norm_rows(const float* input, const float* gamma, int groups, float* output, uint16_t* out16, int n_cols,
                   int n_rows, float epsilon, void* stream) {
    if (n_rows <= 0) return;
    elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f || groups < 1 || (output == nullptr) == (out16 == nullptr))
        throw std::invalid_argument("rms_norm_rows: a finite nonnegative epsilon, groups, one output");
    auto& q = core::Runtime::get().stream(stream);
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024<NormKernel>(q);
    if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, nullptr, out16);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, nullptr, out16);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, nullptr, out16);
    if (!stream) core::Runtime::get().finish(q);
}

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
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024<NormKernel>(q);
    if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon);
}

void native_qsa_rms_norm_weighted_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
                                       int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    if (n_cols % 32 != 0 || x_q8_1 == nullptr) throw std::invalid_argument("the q8_1 output needs 32 | n_cols");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    auto& q = queue_for(stream);
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024<NormKernel>(q);
    if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon, 1, x_q8_1);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon, 1, x_q8_1);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon, 1, x_q8_1);
}

void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream) {
    if (groups < 1) throw std::invalid_argument("native QSA requires positive bounded dimensions");
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * groups * 4, output, stream);
    auto& q = queue_for(stream);
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024<NormKernel>(q);
    if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon, groups);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon, groups);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups);
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream, uint16_t* out16) {
    const auto count = elements(head_dim, n_head);
    if (out16 == nullptr) buffers(attn, count * 4, q_full, count * 8, output, stream);
    queue_for(stream).parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
        const std::size_t i = id[0], head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-raw));
        if (out16 != nullptr) out16[i] = f16_gpu::f16_from_f32(attn[i] * sigmoid);
        else output[i] = attn[i] * sigmoid;
    });
}

}  // namespace strata::kernels
