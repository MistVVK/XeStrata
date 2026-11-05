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
#include "strata/core/per_device.hpp"
#include "device_caps.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/f16_gpu.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_qsa
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strata::kernels {

template <int BlockSize> class NormKernel;   // asked for its work-group (work_group_upto_1024)
template <int BlockSize, int Threads> class NormKernelV;

namespace {

constexpr int WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

// native_mmvq.cpp's activation block: the scale and the sum as fp16, then the 32 values
struct Q81 { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81) == 36, "a q8_1 block is 36 bytes");

// gamma: `groups` rows of n_cols, row r takes gamma row r % groups (1: one gamma for every row)
// q8 (n_cols a multiple of 32): the output also as q8_1 blocks, each the 32 columns one sub-group writes in a pass,
// with native_quantize_q8_1's reductions over the stored values, so the blocks are that kernel's bit for bit
// gamma null: no weight; out16: the output as FP16 instead of `output` (null then)
// Threads (BlockSize by default): the work-items a row takes.  Fewer than BlockSize (256 for a block of 1024) give each
// work-item BlockSize / Threads of the block's lanes, their sums kept apart and reduced as the block's own sub-groups
// would reduce them (work-item t holds lanes t + j Threads, of sub-group t / 32 + j Threads / 32), so the result is
// the same to the bit; many rows run faster so (below)
template <int BlockSize, int Threads = BlockSize>
void norm(sycl::queue& q, const float* input, const float* gamma_all, float* output, int n_cols, int n_rows,
          float epsilon, int groups = 1, void* q8 = nullptr, uint16_t* out16 = nullptr) {
    constexpr int V = BlockSize / Threads;
    static_assert(V * Threads == BlockSize && Threads % 32 == 0, "the block's lanes split evenly");
    using Name = std::conditional_t<V == 1, NormKernel<BlockSize>, NormKernelV<BlockSize, Threads>>;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for<Name>(sycl::nd_range<1>(std::size_t(n_rows) * Threads, Threads),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = int(it.get_local_id(0));
            const std::size_t row_offset = it.get_group(0) * std::size_t(n_cols);
            const float* gamma =
                gamma_all ? gamma_all + (it.get_group(0) % std::size_t(groups)) * std::size_t(n_cols) : nullptr;
            const float* in = input + row_offset;
            const int lane = tid % 32;
            for (int j = 0; j < V; ++j) {
                float partial = 0.0f;
                for (std::size_t col = tid + j * Threads; col < std::size_t(n_cols); col += BlockSize) {
                    const float x = in[col];
                    partial += x * x;
                }
                partial = warp_sum(sg, partial);
                if (lane == 0) sums[tid / 32 + j * (Threads / 32)] = partial;
            }
            // all reads of the input for the reduction precede this barrier, so the operation may run in place
            sycl::group_barrier(it.get_group());
            float partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
            partial = warp_sum(sg, partial);
            const float mean = partial / static_cast<float>(n_cols);
            const float scale = sycl::rsqrt(mean + epsilon);
            Q81* blocks = static_cast<Q81*>(q8);
            for (std::size_t col = tid; col < std::size_t(n_cols); col += Threads) {
                const float v = gamma ? scale * in[col] * gamma[col] : scale * in[col];
                if (out16 != nullptr) {
                    out16[row_offset + col] = f16_gpu::f16_from_f32(v);
                    continue;
                }
                output[row_offset + col] = v;
                if (blocks == nullptr) continue;
                const std::size_t i = row_offset + col;
                q8act::q8_1_block(sg, v, reinterpret_cast<uint8_t*>(&blocks[i / 32]), (int) (i % 32));
            }
        });
    });
}

// The norm in a work-group's own reduction (reduce_over_group, the device's own sub-groups): THREADS work-items a row
template <int Threads>
void norm_group(sycl::queue& q, const float* input, const float* gamma_all, float* output, int n_cols, int n_rows,
                float epsilon, int groups, uint16_t* out16) {
    q.parallel_for(sycl::nd_range<1>(std::size_t(n_rows) * Threads, Threads), [=](sycl::nd_item<1> it) {
        const int tid = int(it.get_local_id(0));
        const std::size_t row_offset = it.get_group(0) * std::size_t(n_cols);
        const float* gamma =
            gamma_all ? gamma_all + (it.get_group(0) % std::size_t(groups)) * std::size_t(n_cols) : nullptr;
        const float* in = input + row_offset;
        float partial = 0.0f;
        for (std::size_t col = tid; col < std::size_t(n_cols); col += Threads) partial += in[col] * in[col];
        // the reduction's reads of the input precede its result, so the operation may run in place
        const float scale = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>()) / (float) n_cols +
                                        epsilon);
        for (std::size_t col = tid; col < std::size_t(n_cols); col += Threads) {
            const float v = gamma ? scale * in[col] * gamma[col] : scale * in[col];
            if (out16 != nullptr) out16[row_offset + col] = f16_gpu::f16_from_f32(v);
            else output[row_offset + col] = v;
        }
    });
}

// The norm's form, by what the device reports, the same for any number of rows (a window's rows never change a row's
// bits).  Unpinned (rms_norm_rows), a device with 16-wide sub-groups (Intel's) sums in its work-group's own
// reduction: the 32-wide pinned form
// ran the B70's rows 1.6 to 1.7 times as long (one row of 2,048: 0.0085 ms against 0.0050; 2,048 rows: 0.036 against
// 0.022).  Elsewhere the pinned 32-wide reduction (llama.cpp's CUDA block_reduce), a block of 1,024 for a row of 1,024
// or more (512 where the device takes no larger work-group: the UHD 770); rows enough for every compute unit take 256
// work-items a row in the block's arithmetic.  q8_1 out needs the 32-wide sub-groups (each block one's); the native
// entry points stay pinned, the form their q8_1 and fused RoPE forms match bit for bit (rope_parity).
bool sixteen_wide(const sycl::device& d) {
    return xe::has_sub_group(d, 16);
}

void launch_norm(sycl::queue& q, const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                 float epsilon, int groups, void* q8, uint16_t* out16, bool pinned = true) {
    if (!pinned && q8 == nullptr && sixteen_wide(q.get_device())) {
        if (n_cols >= 1024) norm_group<256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, out16);
        else norm_group<128>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, out16);
        return;
    }
    const int wg = n_cols < 1024 ? 256 : xe::work_group_upto_1024<NormKernel>(q);
    const bool many = n_rows >= (int) q.get_device().get_info<sycl::info::device::max_compute_units>();
    if (wg == 1024 && many) norm<1024, 256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, q8, out16);
    else if (wg == 1024) norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, q8, out16);
    else if (wg == 512 && many) norm<512, 256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, q8, out16);
    else if (wg == 512) norm<512>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, q8, out16);
    else norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon, groups, q8, out16);
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
                   int n_rows, float epsilon, void* stream, bool pinned) {
    if (n_rows <= 0) return;
    elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f || groups < 1 || (output == nullptr) == (out16 == nullptr))
        throw std::invalid_argument("rms_norm_rows: a finite nonnegative epsilon, groups, one output");
    auto& q = core::Runtime::get().stream(stream);
    launch_norm(q, input, gamma, output, n_cols, n_rows, epsilon, groups, nullptr, out16, pinned);
    if (!stream) core::Runtime::get().finish(q);
}

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    launch_norm(queue_for(stream), input, gamma, output, n_cols, n_rows, epsilon, 1, nullptr, nullptr);
}

void native_qsa_rms_norm_weighted_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
                                       int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    if (n_cols % 32 != 0 || x_q8_1 == nullptr) throw std::invalid_argument("the q8_1 output needs 32 | n_cols");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    launch_norm(queue_for(stream), input, gamma, output, n_cols, n_rows, epsilon, 1, x_q8_1, nullptr);
}

void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream) {
    if (groups < 1) throw std::invalid_argument("native QSA requires positive bounded dimensions");
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * groups * 4, output, stream);
    launch_norm(queue_for(stream), input, gamma, output, n_cols, n_rows, epsilon, groups, nullptr, nullptr);
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream, uint16_t* out16) {
    const auto count = elements(head_dim, n_head);
    if (out16 == nullptr) buffers(attn, count * 4, q_full, count * 8, output, stream);
    // one dimension: the heads of a prompt (T x 24 of Qwen3.8-Flash-Next) pass the 65,535 a GPU's first dimension
    // takes (a range of two failed to launch on the RX 9060 XT at 32K)
    queue_for(stream).parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
        const std::size_t i = id[0], head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-raw));
        if (out16 != nullptr) out16[i] = f16_gpu::f16_from_f32(attn[i] * sigmoid);
        else output[i] = attn[i] * sigmoid;
    });
}

}  // namespace strata::kernels
