// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_ple_postops.cpp - the Xe port of Strata's src/kernels/cuda/native_ple_postops.cu: the PLE block after
// its projections (key/query RMS norms, the per-group gate, the gated value, its norm, the dilated convolution and the
// residual), for one token and for a batch, with the arithmetic of the pinned ggml-cuda operators it replaces.
//
// CUDA's explicit round-to-nearest operations are plain operations here, since the library does not contract; the
// fma with a zero addend (ggml SCALE's zero bias) stays an fma.  expf and sqrtf are the precise ones.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
constexpr int WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = 16; offset; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}

sycl::queue& queue_for(void* stream) {
    if (!stream) throw core::DeviceError("native PLE postops require an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

// SUM_ROWS with 512 work-items for a 2560-wide row: eight partial lanes per work-item and the materialized product
// rounding (never a dot fma), then the scaled signed square root and the sigmoid
void gate(sycl::queue& q, const float* key, const float* query, float* g, int rows, float scale) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * 512, 512), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const size_t row = it.get_group(0);
            float sums[8] = {};
            for (int j = 0; j < 8; ++j) {
                const int d = tid + j * 512;
                const float p = d < N ? key[row * N + d] * query[row * N + d] : 0.0f;
                sums[j] += p;
            }
            float sum = 0;
            for (int j = 0; j < 8; ++j) sum += sums[j];
            sum = warp_sum(sg, sum);
            const int lane = tid % 32;
            if (!lane) partials[tid / 32] = sum;
            sycl::group_barrier(it.get_group());
            sum = lane < 16 ? partials[lane] : 0.0f;
            sum = warp_sum(sg, sum);
            if (tid == 0) {
                const float s = sycl::fma(scale, sum, 0.0f);   // ggml SCALE's zero bias
                const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
                const float sign = float((s > 0.0f) - (s < 0.0f));
                g[row] = 1.0f / (1.0f + sycl::exp(-(sign * mag)));
            }
        });
    });
}

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

// weighted_rms_norm (native_gr_norm) with the gamma row repeating every H rows (one token's H groups);
// BlockSize as native_gr_rms_norm_weighted chooses it for 2560 columns on the device
template <int BlockSize>
void rms_rep_launch(sycl::queue& q, const float* input, const float* gamma, float* output, int rows) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const size_t row_offset = it.get_group(0) * N;
            const float* in = input + row_offset;
            float* out = output + row_offset;
            const float* ga = gamma + (it.get_group(0) % H) * N;
            float partial = 0.0f;
            for (int col = tid; col < N; col += BlockSize) {
                const float value = in[col];
                partial += value * value;
            }
            partial = warp_sum(sg, partial);
            const int lane = tid % 32;
            if (lane == 0) sums[tid / 32] = partial;
            sycl::group_barrier(it.get_group());
            partial = 0.0f;
            if (lane < BlockSize / 32) partial = sums[lane];
            partial = warp_sum(sg, partial);
            const float mean = partial / N;
            const float scale = sycl::rsqrt(mean + NG_RMS_EPS);
            for (int col = tid; col < N; col += BlockSize) out[col] = scale * in[col] * ga[col];
        });
    });
}
void rms_rep(sycl::queue& q, const float* input, const float* gamma, float* output, int rows) {
    const int wg = xe::work_group_upto_1024(q);
    if (wg == 1024) rms_rep_launch<1024>(q, input, gamma, output, rows);
    else if (wg == 512) rms_rep_launch<512>(q, input, gamma, output, rows);
    else rms_rep_launch<256>(q, input, gamma, output, rows);
}

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}

}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    auto& q = queue_for(stream);
    native_gr_rms_norm_weighted(projected_key, w.norm_key, b.key, N, H, NG_RMS_EPS, stream);
    native_gr_rms_norm_weighted(hidden, w.norm_query, b.query, N, H, NG_RMS_EPS, stream);
    gate(q, b.key, b.query, b.gate, H, 1.0f / std::sqrt(float(N)));
    const float* gt = b.gate;
    float* gated = b.gated;
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        gated[i] = value[i % N] * gt[i / N];
    });
    native_gr_rms_norm_weighted(b.gated, w.norm_conv, b.normalized, N, H, NG_RMS_EPS, stream);
    const float* normalized = b.normalized;
    const uint16_t* weights = w.conv1d_f16;
    float* conv = b.conv;
    float* result = b.result;
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int c = int(id[0]);
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
            const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
            const float term = x * f32_from_f16(weights[c * 4 + k]);
            sum = k == 0 ? term : sum + term;
        }
        const float activation = silu(sum);
        conv[c] = activation;
        // an exact hidden/result alias is safe: each work-item owns one element
        result[c] = hidden[c] + (gated[c] + activation);
    });
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate_out, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate_out)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto& q = queue_for(stream);
    const int rows = T * H;
    const size_t total = size_t(T) * D;
    rms_rep(q, key, w.norm_key, key, rows);
    rms_rep(q, hidden, w.norm_query, query_norm, rows);
    gate(q, key, query_norm, gate_out, rows, 1.0f / std::sqrt(float(N)));
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const size_t i = id[0], t = i / D, d = i % D;
        gated[i] = value[t * N + d % N] * gate_out[t * H + d / N];
    });
    rms_rep(q, gated, w.norm_conv, query_norm, rows);
    const uint16_t* weights = w.conv1d_f16;
    // the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int t = int(i / D), c = int(i % D);
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
            const int p = t - 9 + 3 * k;
            const float x = p >= 0 ? query_norm[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
            const float term = x * f32_from_f16(weights[c * 4 + k]);
            sum = k == 0 ? term : sum + term;
        }
        hidden[i] = hidden[i] + (gated[i] + silu(sum));
    });
    // the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int c = int(id[0]);
        float h[HISTORY];
        for (int r = 0; r < HISTORY; ++r) {
            const int p = T - HISTORY + r;
            h[r] = p >= 0 ? query_norm[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
        }
        for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
    });
}

}  // namespace strata::kernels
