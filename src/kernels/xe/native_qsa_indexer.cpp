// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/native_qsa_indexer.cpp - the Xe port of Strata's src/kernels/cuda/native_qsa_indexer.cu: the QSA indexer's
// pooled-key append (keys rounded through F16, the block mean, RMS norm, gamma and the rotation), for one cell and
// for a batch, with the arithmetic of the pinned ggml-cuda operators it replaces (llama.cpp 3cf03257, MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// CUDA's fast-math powf, cosf, sinf and rsqrtf are the precise pow, cos, sin and SYCL's rsqrt here.  The explicit
// round-to-nearest adds are plain adds (the library does not contract); SCALE's zero-bias fma stays an fma.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/core/runtime.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256, WARP = 32;

inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = 16; offset; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}
// SET_ROWS stores F16; GET_ROWS expands those exact values to F32
inline float through_f16(float x) { return f32_from_f16(f16_from_f32(x)); }

inline float pooled_value(const float* values, int d, int rope_pos, float theta_scale, const int32_t* mtab,
                          bool zero_pos, bool scaled, const RopeKernelArgs& ka, const RopeTab& tab) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const int p = zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair);
        float c, s;
        if (!rope_tab_cs(tab, p, pair, c, s))
            rope_cos_sin((float) p * sycl::pow(theta_scale, float(pair)), scaled, ka, pair, c, s);
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

}  // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device, int pos_stride, int n_steps,
                                     int32_t pos_base, const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (n_steps <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const std::size_t pos_span = n_steps == 1 || pos_stride <= 0 ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
    const Span spans[] = {{raw,std::size_t(n_steps)*D*4},{relative_pos_device,pos_span},{gamma,std::size_t(D)*4},{b.tail,std::size_t(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / ROT);
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
            const int d = int(it.get_local_id(0));
            // the cells in order; a work-item reads back only the tail and dead lanes it wrote itself, so the barrier
            // between cells guards the work-group's local values and partials only
            for (int step = 0; step < n_steps; ++step) {
                if (step > 0) sycl::group_barrier(it.get_group());
                const int pos = relative_pos_device[std::size_t(step) * pos_stride];
                if (pos < 0 || pos >= mc) continue;   // uniform
                const int slot = pos % R;
                float incoming = 0.0f;
                if (d < D) {
                    incoming = through_f16(raw[std::size_t(step) * D + d]);
                    if (slot < R - 1) tail[slot * D + d] = incoming;
                }
                if (pos != 0 && slot != R - 1) continue;   // uniform
                float mean = 0.0f;
                if (d < D) {
                    // the spare's four gather indices all name cell zero; completed blocks use chronological slices
                    float sum = pos == 0 ? incoming : tail[d];
                    for (int j = 1; j < R; ++j) sum = sum + (pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
                    mean = sycl::fma(0.25f, sum, 0.0f);   // SCALE includes a zero bias
                }
                const float square_sum = norm_square_sum(it, mean, &partials[0]);
                const float scale = sycl::rsqrt(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                sycl::group_barrier(it.get_group());
                if (d >= D) continue;
                const int bl = pos / R;
                const int rope_pos = pos == 0 ? 0 : pos_base + R * bl;
                const float y = pooled_value(&values[0], d, rope_pos, theta_scale, mtab, pos == 0, scaled, ka, tab);
                pooled[std::size_t(bl) * D + d] = y;
                if (pos == 0) dead[d] = y;
                else pooled[std::size_t(bl + 1) * D + d] = dead[d];
                if (d == 0 && pos != 0) *block_pos = rope_pos;
            }
        });
    });
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    native_qsa_indexer_append_steps(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells, scaling,
                                    stream);
}

void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions or parameters");
    auto& q = queue_for(stream);
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / ROT);
    const bool scaled = scaling.type != RopeScalingType::None;
    const RopeKernelArgs ka = scaling.kernel_args(ROT);
    const RopeTab tab = rope_table_for(scaling);
    const int32_t* mtab = mrope_table();
    float* tail = b.tail;
    float* dead = b.dead;
    float* pooled = b.pooled;
    int32_t* block_pos = b.block_pos;
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
                const float y = pooled_value(&values[0], d, 0, theta_scale, mtab, true, scaled, ka, tab);
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
                pooled[std::size_t(bl) * D + d] = pooled_value(&values[0], d, rope_pos, theta_scale, mtab, false, scaled, ka, tab);
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
