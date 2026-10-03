// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_gdn_preprocess.cpp - the Xe port of Strata's src/kernels/cuda/native_gdn_preprocess.cu: the GDN
// convolution + SiLU, the L2 norm (RMS norm then ggml_scale), the beta and gate epilogues and the output norm, with
// the arithmetic of ggml-cuda's norm.cu, unary.cu, ssm-conv.cu and scale.cu at pinned llama.cpp 3cf03257 (MIT
// License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// CUDA compiled this with --use_fast_math; the Xe version uses the precise exp, log and rsqrt.  The explicit
// round-to-nearest operations (__fadd_rn, __fmul_rn, __fmaf_rn) stay as written: the library does not contract, and
// the fma with a zero addend keeps its signed-zero behaviour.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/core/runtime.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;
constexpr int WARP = 32;
constexpr int NORM_THREADS = 256;

inline float warp_sum(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}
// two-level sum over the work-group's eight sub-groups
inline float norm_sum(const sycl::nd_item<1>& it, float value, float* sums) {
    const sycl::sub_group sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(0);
    const int lane = tid % WARP;
    value = warp_sum(sg, value);
    if (lane == 0) sums[tid / WARP] = value;
    sycl::group_barrier(it.get_group());
    value = lane < NORM_THREADS / WARP ? sums[lane] : 0.0f;
    return warp_sum(sg, value);
}
inline float sigmoid(float value) { return 1.0f / (1.0f + sycl::exp(-value)); }

sycl::queue& queue_for(void* stream) {
    if (!stream) throw core::DeviceError("native GDN preprocessing requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}

}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    queue_for(stream).parallel_for(sycl::range<1>(size_t(channels)), [=](sycl::id<1> id) {
        const int c = int(id[0]);
        const float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
        float sum = 0.0f;
        for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
        sum = sum + 0.0f;   // the native SSM kernel adds its zero bias even when there is no bias input
        raw_output[c] = sum;
        silu_output[c] = sum / (1.0f + sycl::exp(-sum));
        for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
    });
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    const float eps = epsilon / S, scale_after = 1.0f / sycl::sqrt(float(S));
    queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(WARP), h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * NORM_THREADS, NORM_THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int col = int(it.get_local_id(0));
            float* row = input + it.get_group(0) * S;
            const float value = col < S ? row[col] : 0.0f;
            float partial = 0.0f;
            if (col < S) partial += value * value;
            partial = norm_sum(it, partial, &sums[0]);
            const float scale = sycl::rsqrt(partial / S + eps);
            if (col < S) {
                // the FP32 store boundary between RMSNorm and ggml_scale
                const float normalized = scale * value;
                row[col] = sycl::fma(normalized, scale_after, 0.0f);
            }
        });
    });
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    queue_for(stream).parallel_for(sycl::range<1>(size_t(heads)), [=](sycl::id<1> i) { beta[i] = sigmoid(beta[i]); });
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    queue_for(stream).parallel_for(sycl::range<1>(size_t(heads)), [=](sycl::id<1> i) {
        const float value = alpha[i] + dt[i];
        // log1p: log(1 + e^v) is 0 for v below about -16 (upstream 7bc505d); the fused kernels already use it
        const float softplus = value > 20.0f ? value : sycl::log1p(sycl::exp(value));
        gate[i] = softplus * ssm_a[i];
    });
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(WARP), h);
        h.parallel_for(sycl::nd_range<1>(size_t(heads) * NORM_THREADS, NORM_THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int col = int(it.get_local_id(0));
            const size_t offset = it.get_group(0) * S;
            const float value = col < S ? output[offset + col] : 0.0f;
            float partial = 0.0f;
            if (col < S) partial += value * value;
            partial = norm_sum(it, partial, &sums[0]);
            const float scale = sycl::rsqrt(partial / S + epsilon);
            if (col < S) {
                // RMSNorm + gamma is one pinned fused operator, followed by sigmoid * mul
                const float weighted = (scale * value) * gamma[col];
                destination[offset + col] = weighted * sigmoid(z[offset + col]);
            }
        });
    });
}

}  // namespace strata::kernels
