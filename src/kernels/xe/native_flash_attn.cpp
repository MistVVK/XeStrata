// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_flash_attn.cpp - the Xe port of Strata's src/kernels/cuda/native_flash_attn.cu: the short-context
// decode attention (D = 256, one query column, F16 K/V), specialized from ggml-cuda's fattn-vec at pinned llama.cpp
// 3cf03257 (MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// 128 work-items as four sub-groups of 32; eight lanes share a K row, as in the CUDA kernel.  CUDA's explicit
// __fmaf_rn/__fmul_rn stay fused/unfused as written; its expf is the precise exp.  The score tile is re-written by
// the next chunk after its own sub-group read it, so a sub-group barrier separates the two (CUDA relied on the warp
// running in step there).
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

template <int Width>
inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = Width / 2; offset; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}
inline float warp_max(const sycl::sub_group& sg, float x) {
    for (int offset = 16; offset; offset >>= 1) x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, offset));
    return x;
}

struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}

}  // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* out, int32_t* status,
                                  const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 ||
        shapes.idx_block != 4 || shapes.idx_top_k < 256 || capacity < 256 ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 ||
        max_context < 1 || max_context > 256)
        throw std::invalid_argument("native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an explicit stream");
    if (!stream) throw core::DeviceError("native FlashAttention requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, kStepCount * 4, 4}, {out, 24 * 256 * 4, 4},
                          {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    // the pinned launcher's padded length 256 gives one KV tile
    constexpr int padded_length = 256;
    constexpr float scale = 0.0625f;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(4 * 4 * 256), h);
        sycl::local_accessor<float, 1> max_shared(sycl::range<1>(32), h);
        sycl::local_accessor<float, 1> sum_shared(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(24 * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = int(it.get_local_id(0)), lane = tid % 32, warp = tid / 32;
            const int head = int(it.get_group(0)), kv = head / 12;
            const int width = step[kStepWidth], nkv = step[kStepNKv];
            const bool valid = width >= 1 && width <= max_context && nkv == width &&
                               step[kStepPos] == width - 1 && step[kStepNBid] == width / 4;
            if (head == 0 && tid == 0) *status = valid ? kNativeFlashAttnSuccess : kNativeFlashAttnUnsupportedStep;
            if (!valid) {   // uniform over the work-group
                out[head * 256 + tid] = std::numeric_limits<float>::quiet_NaN();
                out[head * 256 + tid + 128] = std::numeric_limits<float>::quiet_NaN();
                return;
            }
            float qx[16], qy[16], vx[16] = {}, vy[16] = {};
            float maximum = -FLT_MAX / 2.0f, sum = 0.0f;
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * (i + j);
                    qx[i0 / 8 + j] = q[head * 256 + d] * scale;
                    qy[i0 / 8 + j] = q[head * 256 + d + 1] * scale;
                }
            }
            for (int base = 0; base < padded_length; base += 128) {
                float score = 0.0f, next_max = maximum;
                for (int row = 0; row < 8; ++row) {
                    const int cell = base + warp * 32 + (lane & ~7) + row;
                    float dot = 0.0f;
                    for (int i0 = 0; i0 < 128; i0 += 32) {
                        const int i = i0 + (lane % 8) * 4;
                        for (int j = 0; j < 4; ++j) {
                            const int d = 2 * (i + j);
                            const float a = cell < width ? f32_from_f16(k[(cell * 2 + kv) * 256 + d]) : 0.0f;
                            const float b = cell < width ? f32_from_f16(k[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                            dot += a * qx[i0 / 8 + j];
                            dot += b * qy[i0 / 8 + j];
                        }
                    }
                    dot = warp_sum<8>(sg, dot);
                    dot += cell < width ? (mask ? f32_from_f16(mask[cell]) : 0.0f) : -std::numeric_limits<float>::infinity();
                    next_max = sycl::fmax(next_max, dot + (3.0f * 0.6931f));
                    if (lane % 8 == row) score = dot;
                }
                for (int offset = 8; offset < 32; offset <<= 1)
                    next_max = sycl::fmax(next_max, sycl::permute_group_by_xor(sg, next_max, offset));
                const float rescale = sycl::exp(maximum - next_max);
                maximum = next_max;
                score = sycl::exp(score - maximum);
                sum = sum * rescale + score;
                tile[tid] = score;
                sycl::group_barrier(sg);
                for (int k0 = 0; k0 < 32; k0 += 4) {
                    const int local = warp * 32 + k0 + lane / 8, cell = base + local;
                    const float weight = tile[local];
                    for (int i0 = 0; i0 < 128; i0 += 32) {
                        const int i = i0 + (lane % 8) * 4;
                        for (int j = 0; j < 4; ++j) {
                            const int d = 2 * (i + j);
                            const float a = cell < width ? f32_from_f16(v[(cell * 2 + kv) * 256 + d]) : 0.0f;
                            const float b = cell < width ? f32_from_f16(v[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                            // the pinned sm120a binary fuses the old-accumulator rescale with the FIRST addition,
                            // fma(rescale, old, round(V * w)); later columns use fma(V, w, acc)
                            if (k0 == 0) {
                                vx[i0 / 8 + j] = sycl::fma(rescale, vx[i0 / 8 + j], a * weight);
                                vy[i0 / 8 + j] = sycl::fma(rescale, vy[i0 / 8 + j], b * weight);
                            } else {
                                vx[i0 / 8 + j] = sycl::fma(a, weight, vx[i0 / 8 + j]);
                                vy[i0 / 8 + j] = sycl::fma(b, weight, vy[i0 / 8 + j]);
                            }
                        }
                    }
                }
                sycl::group_barrier(sg);   // the reads above precede the next chunk's tile write
            }
            if (warp == 0) { max_shared[lane] = -FLT_MAX / 2.0f; sum_shared[lane] = 0.0f; }
            sycl::group_barrier(it.get_group());
            if (lane == 0) max_shared[warp] = maximum;
            sycl::group_barrier(it.get_group());
            const float global_max = warp_max(sg, max_shared[lane]);
            const float rescale = sycl::exp(maximum - global_max);
            for (int i = 0; i < 16; ++i) {
                vx[i] = vx[i] * rescale;
                vy[i] = vy[i] * rescale;
            }
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int start = warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
                for (int j = 0; j < 4; ++j) {
                    tile[start + 2 * j] = vx[i0 / 8 + j];
                    tile[start + 2 * j + 1] = vy[i0 / 8 + j];
                }
            }
            sum *= rescale;
            sum = warp_sum<32>(sg, sum);
            if (lane == 0) sum_shared[warp] = sum;
            sycl::group_barrier(it.get_group());
            sum = warp_sum<32>(sg, sum_shared[lane]);
            for (int i0 = 0; i0 < 256; i0 += 128) {
                float result = 0.0f;
                for (int w = 0; w < 4; ++w)
                    for (int group = 0; group < 4; ++group) result += tile[w * 4 * 256 + group * 256 + i0 + tid];
                out[head * 256 + i0 + tid] = result / sum;
            }
        });
    });
}

}  // namespace strata::kernels
