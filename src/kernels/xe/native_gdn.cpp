// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_gdn.cpp - the Xe port of Strata's src/kernels/cuda/native_gdn.cu: one gated delta-net step with the
// arithmetic of ggml-cuda's gated_delta_net.cu at pinned llama.cpp 3cf03257 (MIT License, Copyright (c) 2023-2026
// The ggml authors, see third_party/main/ggml/LICENSE).  Each sub-group of 32 owns one (head, column) and keeps its four
// state rows in registers.
//
// CUDA compiled this with --use_fast_math to match ggml-cuda; the Xe version uses the precise exp and sqrt.
#include "strata/kernels/native_gdn.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_gdn
#include "strata/kernels/device_arch_tag.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;
constexpr int WARP = 32;
constexpr int COLS = 4;   // columns (sub-groups) per work-group

inline float warp_sum(const sycl::sub_group& sg, float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

}  // namespace

void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 ||
        shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    if (!stream) throw core::DeviceError("native GDN requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / sycl::sqrt(float(S));
    const int h_k = int(shape.h_k), h_v = int(shape.h_v);
    queue.parallel_for(sycl::nd_range<2>({size_t(h_v), size_t(S) * WARP}, {1, size_t(COLS) * WARP}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int head = int(it.get_group(0));
        const int lane = int(sg.get_local_linear_id());
        const int col = int(it.get_group(1)) * COLS + int(sg.get_group_linear_id());
        const int q_head = head % h_k;
        float s_shard[4], k_reg[4], q_reg[4];
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
            k_reg[r] = k[q_head * S + i];
            q_reg[r] = q[q_head * S + i];
        }
        const float g_val = sycl::exp(gate[head]);
        float kv_shard = 0.0f;
        for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
        const float kv_col = warp_sum(sg, kv_shard);
        const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
        float attn_partial = 0.0f;
        for (int r = 0; r < 4; ++r) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_sum(sg, attn_partial);
        if (lane == 0) output[head * S + col] = attn_col * scale;
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
        }
    });
}

void native_gdn_step_rows(float* state, const float* h, int64_t h_stride, const float* gate, const float* beta,
                          float* output, int64_t out_stride, int T, float* const* snaps, int n_snaps,
                          const GdnShapes& shape, void* stream) {
    constexpr int kMaxSnaps = kNativeGdnMaxSnaps;
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 || shape.h_v % shape.h_k != 0 ||
        T < 1 || n_snaps < 0 || n_snaps > kMaxSnaps || n_snaps >= T)
        throw std::invalid_argument("native GDN rows: S=128, divisible head counts, 1 <= T, n_snaps < T and <= 8");
    struct Snaps { float* p[kMaxSnaps]; } sn{};
    for (int i = 0; i < n_snaps; ++i) sn.p[i] = snaps[i];
    auto& queue = core::Runtime::get().stream(stream);
    const float scale = 1.0f / sycl::sqrt(float(S));
    const int h_k = int(shape.h_k), h_v = int(shape.h_v);
    // a work-group of WARP columns of one head, RG sub-groups each holding S / RG of the columns' state rows in
    // registers: each row's load is WARP consecutive floats of the [i][head][col] layout; the two sums over the rows
    // add the RG parts in a fixed order through local memory
    constexpr int RG = 4, R = S / RG;
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(size_t(RG) * WARP), cgh);
        cgh.parallel_for(sycl::nd_range<2>({size_t(h_v), size_t(S / WARP) * RG * WARP}, {1, size_t(RG) * WARP}),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int head = int(it.get_group(0));
            const int lane = int(it.get_local_id(1)) % WARP, rg = int(it.get_local_id(1)) / WARP;
            const int col = int(it.get_group(1)) * WARP + lane;
            const int q_head = head % h_k;
            const size_t row0 = size_t(rg) * R;
            auto at = [&](int i) { return ((row0 + size_t(i)) * h_v + head) * S + col; };
            float s[R];
#pragma unroll
            for (int i = 0; i < R; ++i) s[i] = state[at(i)];
            for (int t = 0; t < T; ++t) {
                const float* ht = h + t * h_stride;
                const float* q = ht + int64_t(q_head) * S + row0;
                const float* k = ht + int64_t(h_k + q_head) * S + row0;
                const float v = ht[int64_t(2 * h_k + head) * S + col];
                const float g_val = sycl::exp(gate[int64_t(t) * h_v + head]);
                float kv = 0.0f;
#pragma unroll
                for (int i = 0; i < R; ++i) kv += s[i] * k[i];
                part[rg * WARP + lane] = kv;
                sycl::group_barrier(it.get_group());
                kv = 0.0f;
                for (int r = 0; r < RG; ++r) kv += part[r * WARP + lane];
                const float delta = (v - g_val * kv) * beta[int64_t(t) * h_v + head];
                float attn = 0.0f;
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    s[i] = g_val * s[i] + k[i] * delta;
                    attn += s[i] * q[i];
                }
                sycl::group_barrier(it.get_group());
                part[rg * WARP + lane] = attn;
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    attn = 0.0f;
                    for (int r = 0; r < RG; ++r) attn += part[r * WARP + lane];
                    output[t * out_stride + int64_t(head) * S + col] = attn * scale;
                }
                sycl::group_barrier(it.get_group());
                if (t < n_snaps) {
                    float* d = sn.p[t];
#pragma unroll
                    for (int i = 0; i < R; ++i) d[at(i)] = s[i];
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) state[at(i)] = s[i];
        });
    });
}

}  // namespace strata::kernels
