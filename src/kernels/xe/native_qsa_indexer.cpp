// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_qsa_indexer.cpp - the Xe port of Strata's src/kernels/cuda/native_qsa_indexer.cu: the QSA indexer's
// pooled-key append (keys rounded through F16, the block mean, RMS norm, gamma and the rotation), for one cell and
// for a batch, with the arithmetic of the pinned ggml-cuda operators it replaces (llama.cpp 3cf03257, MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// CUDA's fast-math powf, cosf, sinf and rsqrtf are the precise pow, cos, sin and SYCL's rsqrt here.  The explicit
// round-to-nearest adds are plain adds (the library does not contract); SCALE's zero-bias fma stays an fma.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_qsa_indexer
#include "strata/kernels/device_arch_tag.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int D = 128, R = 4, ROT = 64, THREADS = 256, WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = 16; offset; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}
// SET_ROWS stores F16; GET_ROWS expands those exact values to F32
inline float through_f16(float x) { return f32_from_f16(f16_from_f32(x)); }

inline float pooled_value(const float* values, int d, int rope_pos, float theta_scale, float theta_base, const int32_t* mtab,
                          bool zero_pos, bool scaled, const RopeKernelArgs& ka, const RopeTab& tab, bool bf16_table) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const int p = zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair);
        float c, s;
        if (bf16_table) {
            rope_bf16_table_cs((float) p, pair, ROT, theta_base, scaled, ka, c, s);
            c = f32_from_bf16(bf16_from_f32(c)); s = f32_from_bf16(bf16_from_f32(s));
        } else if (!rope_tab_cs(tab, p, pair, c, s)) {
            rope_cos_sin((float) p * sycl::pow(theta_scale, float(pair)), scaled, ka, pair, c, s);
        }
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}

// the block mean's RMS norm: every work-item of the group takes part; returns the sum of squares
inline float norm_square_sum(const sycl::nd_item<1>& it, float mean, float* partials) {
    const sycl::sub_group sg = it.get_sub_group();
    const int d = int(it.get_local_id(0));
    float square_sum = d < D ? mean * mean : 0.0f;
    square_sum = warp_sum(sg, square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    sycl::group_barrier(it.get_group());
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    return warp_sum(sg, square_sum);
}

struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
sycl::queue& queue_for(void* stream) {
    if (!stream) throw core::DeviceError("native QSA indexer requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

// One work-group's appends of n_steps cells into one layer's buffers (native_qsa_indexer_append_steps and the commit's
// layers launch).  `values` (D) and `partials` (32) are the work-group's local memory.
inline void append_cells(const sycl::nd_item<1>& it, float* values, float* partials, const float* raw,
                         const int32_t* relative_pos_device, int pos_stride, int n_steps, int32_t pos_base,
                         const float* gamma, float epsilon, float* tail, float* dead, float* pooled, int32_t* block_pos,
                         int mc, float theta_scale, float theta_base, const int32_t* mtab, bool scaled, const RopeKernelArgs& ka,
                         const RopeTab& tab, const float* block_gamma, bool bf16_table, int32_t first_pos = 0) {
    const int d = int(it.get_local_id(0));
    // the cells in order; a work-item reads back only the tail and dead lanes it wrote itself, so the barrier
    // between cells guards the work-group's local values and partials only
    for (int step = 0; step < n_steps; ++step) {
        if (step > 0) sycl::group_barrier(it.get_group());
        const int pos = relative_pos_device ? relative_pos_device[std::size_t(step) * pos_stride] : first_pos + step;
        if (pos < 0 || pos >= mc) continue;   // uniform
        const int slot = pos % R;
        float incoming = 0.0f;
        if (d < D) incoming = raw[std::size_t(step) * D + d];
        if (block_gamma) {
            const float square_sum = norm_square_sum(it, incoming, partials);
            const float scale = sycl::rsqrt(square_sum / D + epsilon);
            if (d < D) incoming = incoming * scale * gamma[d];
            sycl::group_barrier(it.get_group());
        } else if (d < D) {
            incoming = through_f16(incoming);
        }
        if (d < D && slot < R - 1) tail[slot * D + d] = incoming;
        if (pos != 0 && slot != R - 1) continue;   // uniform
        float mean = 0.0f;
        if (d < D) {
            // the spare's four gather indices all name cell zero; completed blocks use chronological slices
            float sum = pos == 0 ? incoming : tail[d];
            for (int j = 1; j < R; ++j) sum = sum + (pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
            mean = sycl::fma(0.25f, sum, 0.0f);   // SCALE includes a zero bias
        }
        const float square_sum = norm_square_sum(it, mean, partials);
        const float scale = sycl::rsqrt(square_sum / D + epsilon);
        if (d < D) values[d] = scale * mean * (block_gamma ? block_gamma[d] : gamma[d]);
        sycl::group_barrier(it.get_group());
        if (d >= D) continue;
        const int bl = pos / R;
        const int rope_pos = pos == 0 ? 0 : pos_base + R * bl;
        const float y = pooled_value(values, d, rope_pos, theta_scale, theta_base, mtab, pos == 0, scaled, ka, tab, bf16_table);
        pooled[std::size_t(bl) * D + d] = y;
        if (pos == 0) dead[d] = y;
        else pooled[std::size_t(bl + 1) * D + d] = dead[d];
        if (d == 0 && pos != 0) *block_pos = rope_pos;
    }
}

}  // namespace

void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device, int pos_stride, int n_steps,
                                     int32_t pos_base, const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream,
                                     const float* block_gamma, bool bf16_table) {
    if (n_steps <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const std::size_t pos_span = n_steps == 1 || pos_stride <= 0 ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
    const Span spans[] = {{raw,std::size_t(n_steps)*D*4},{relative_pos_device,pos_span},{gamma,std::size_t(D)*4},{b.tail,std::size_t(R-1)*D*4},
        {b.dead,std::size_t(D)*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    if (block_gamma) {
        const Span extra{block_gamma, std::size_t(D) * 4};
        validate(extra);
        for (const auto& span : spans)
            if (overlaps(extra, span)) throw std::invalid_argument("native QSA indexer buffers overlap");
    }
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / ROT);
    const float theta_base = (float) scaling.freq_base;
    const bool scaled = scaling.type != RopeScalingType::None;
    const RopeKernelArgs ka = scaling.kernel_args(ROT);
    const RopeTab tab = rope_table_for(scaling);
    const int32_t* mtab = mrope_table();
    const int mc = int(max_cells);
    float* tail = b.tail;
    float* dead = b.dead;
    float* pooled = b.pooled;
    int32_t* block_pos = b.block_pos;
    queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            append_cells(it, &values[0], &partials[0], raw, relative_pos_device, pos_stride, n_steps, pos_base, gamma,
                         epsilon, tail, dead, pooled, block_pos, mc, theta_scale, theta_base, mtab, scaled, ka, tab, block_gamma, bf16_table);
        });
    });
}

void native_qsa_indexer_append_steps_layers(const float* raw0, int64_t raw_stride, const int32_t* relative_pos_device,
                                            int pos_stride, int n_steps, int32_t pos_base, float epsilon,
                                            const QsaIndexerLayers& L, int n_layers, const QsaShapes& s,
                                            int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (n_steps <= 0 || n_layers <= 0) return;
    if (!stream || n_layers > QsaIndexerLayers::kMax || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (layers): bad geometry, layer count or parameters");
    for (int i = 0; i < n_layers; ++i) {   // the per-layer append's span checks, layer by layer
        const std::size_t pos_span = n_steps == 1 || pos_stride <= 0 ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
        const Span spans[] = {{raw0 + (std::size_t) i * (std::size_t) raw_stride, std::size_t(n_steps)*D*4},
            {relative_pos_device,pos_span},{L.gamma[i],std::size_t(D)*4},{L.tail[i],std::size_t(R-1)*D*4},{L.dead[i],std::size_t(D)*4},
            {L.pooled[i],std::size_t(max_cells/R+1)*D*4},{L.block_pos[i],4}};
        for (const auto& span : spans) validate(span);
        for (int a = 0; a < 7; ++a) for (int b = a + 1; b < 7; ++b)
            if (overlaps(spans[a], spans[b])) throw std::invalid_argument("native QSA indexer buffers overlap");
        if (L.block_gamma[i]) {
            const Span extra{L.block_gamma[i], std::size_t(D) * 4};
            validate(extra);
            for (const auto& span : spans)
                if (overlaps(extra, span)) throw std::invalid_argument("native QSA indexer buffers overlap");
        }
    }
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / ROT);
    const float theta_base = (float) scaling.freq_base;
    const bool scaled = scaling.type != RopeScalingType::None;
    const RopeKernelArgs ka = scaling.kernel_args(ROT);
    const RopeTab tab = rope_table_for(scaling);
    const int32_t* mtab = mrope_table();
    const int mc = int(max_cells);
    queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        // one work-group a layer; each first restores its tail from the window's snapshot (the per-layer commit's
        // copy_from_mapped), then makes that layer's appends
        h.parallel_for(sycl::nd_range<1>((size_t) n_layers * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int i = (int) it.get_group(0);
            float* tail = L.tail[i];
            if (L.tail_src[i] != nullptr) {
                for (int j = (int) it.get_local_id(0); j < (R - 1) * D; j += THREADS) tail[j] = reinterpret_cast<const volatile float*>(L.tail_src[i])[j];   // as copy_from_mapped reads it
                sycl::group_barrier(it.get_group());
            }
            append_cells(it, &values[0], &partials[0], raw0 + (std::size_t) i * (std::size_t) raw_stride,
                         relative_pos_device, pos_stride, n_steps, pos_base, L.gamma[i], epsilon, tail, L.dead[i],
                         L.pooled[i], L.block_pos[i], mc, theta_scale, theta_base, mtab, scaled, ka, tab, L.block_gamma[i], L.bf16_table[i]);
        });
    });
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream,
                               const float* block_gamma, bool bf16_table) {
    native_qsa_indexer_append_steps(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells, scaling,
                                    stream, block_gamma, bf16_table);
}

void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream, const float* block_gamma, bool bf16_table) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions or parameters");
    auto& q = queue_for(stream);
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / ROT);
    const float theta_base = (float) scaling.freq_base;
    const bool scaled = scaling.type != RopeScalingType::None;
    const RopeKernelArgs ka = scaling.kernel_args(ROT);
    const RopeTab tab = rope_table_for(scaling);
    const int32_t* mtab = mrope_table();
    float* tail = b.tail;
    float* dead = b.dead;
    float* pooled = b.pooled;
    int32_t* block_pos = b.block_pos;
    if (block_gamma) {
        const Span spans[] = {{raw, std::size_t(n) * D * 4}, {gamma, std::size_t(D) * 4},
            {block_gamma, std::size_t(D) * 4}, {tail, std::size_t(R - 1) * D * 4}, {dead, std::size_t(D) * 4},
            {pooled, std::size_t(max_cells / R + 1) * D * 4}, {block_pos, 4}};
        for (const auto& span : spans) validate(span);
        for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> values(sycl::range<1>(D), h);
            sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                append_cells(it, &values[0], &partials[0], raw, nullptr, 0, int(n), pos_base, gamma, epsilon,
                             tail, dead, pooled, block_pos, int(max_cells), theta_scale, theta_base, mtab, scaled, ka, tab,
                             block_gamma, bf16_table, int32_t(p0));
            });
        });
        return;
    }
    if (p0 == 0) {
        // cell 0 of a sequence: the spare (every gather index names cell 0), written to pooled[0] and dead
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> values(sycl::range<1>(D), h);
            sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const int d = int(it.get_local_id(0));
                float mean = 0.0f;
                if (d < D) {
                    const float incoming = through_f16(raw[d]);
                    float sum = incoming;
                    for (int j = 1; j < R; ++j) sum = sum + incoming;
                    mean = sycl::fma(0.25f, sum, 0.0f);
                }
                const float square_sum = norm_square_sum(it, mean, &partials[0]);
                const float scale = sycl::rsqrt(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                sycl::group_barrier(it.get_group());
                if (d >= D) return;
                const float y = pooled_value(&values[0], d, 0, theta_scale, theta_base, mtab, true, scaled, ka, tab, bf16_table);
                pooled[d] = y;
                dead[d] = y;
            });
        });
    }
    // completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n)
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;
    if (hi >= first) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> values(sycl::range<1>(D), h);
            sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(size_t(hi - first + 1) * THREADS, THREADS),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const int64_t bl = first + int64_t(it.get_group(0));
                const int d = int(it.get_local_id(0));
                float mean = 0.0f;
                if (d < D) {
                    auto key = [&](int j) -> float {
                        const int64_t cell = bl * R + j;
                        return cell >= p0 ? through_f16(raw[(cell - p0) * D + d]) : tail[j * D + d];
                    };
                    float sum = key(0);
                    for (int j = 1; j < R; ++j) sum = sum + key(j);
                    mean = sycl::fma(0.25f, sum, 0.0f);
                }
                const float square_sum = norm_square_sum(it, mean, &partials[0]);
                const float scale = sycl::rsqrt(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                sycl::group_barrier(it.get_group());
                if (d >= D) return;
                const int rope_pos = pos_base + R * int(bl);
                pooled[std::size_t(bl) * D + d] = pooled_value(&values[0], d, rope_pos, theta_scale, theta_base, mtab, false, scaled, ka, tab, bf16_table);
                if (bl == hi) {
                    pooled[std::size_t(bl + 1) * D + d] = dead[d];
                    if (d == 0) *block_pos = rope_pos;
                }
            });
        });
    }
    // the tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any
    q.parallel_for(sycl::range<2>(R - 1, D), [=](sycl::id<2> id) {
        const int sl = int(id[0]), d = int(id[1]);
        const int64_t last = p0 + n - 1;
        const int64_t cell = last - ((last % R) - sl + R) % R;
        if (cell < p0) return;
        tail[sl * D + d] = through_f16(raw[(cell - p0) * D + d]);
    });
}

}  // namespace strata::kernels
