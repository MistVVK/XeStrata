// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_mmvq.cpp - see include/strata/kernels/native_mmvq.hpp; the Xe port of
// Strata's src/kernels/cuda/native_mmvq.cu.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h, through the CUDA version.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The CUDA file keeps separate ncols = 1 kernels and a multi-column kernel built to be bitwise equal to them; here
// the multi-column kernel with NCOLS = 1 and the ncols = 1 layout (4 warps, 1 row, or 4 rows for small K) is the
// single-column path, so both share one set of dot-product traits.  A warp is a sub-group of 32.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_act.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "cuda_intrinsics.hpp"
#include "native_mmvq_kernels.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_native_mmvq
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>

namespace strata::kernels {
namespace {

// NW warps per work-group and ROWS rows per work-group.  The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is
// the ncols = 1 layout and keeps every column bitwise equal to a single-column call.  The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block).
bool g_multi_exact = true;

// Which layout a shape's rows take on a GPU.  A sub-group alone walks a row in ceil(blocks / (WARP / T)) dependent
// steps; the work-group's four sub-groups split it and add their sums through local memory.  On a B70 the sub-group
// is the faster up to 10 steps (the 2560-wide rows), the work-group beyond (the 6144-wide rows, 12-24 steps; the
// 2026-10-02-xe-decode-gpu record); on an RX 9060 XT the work-group from 8 (Qwen3.5-9B's 4096-wide Q4_K and Q5_K rows:
// its decode 47.1 -> 49.4 tok/s, 2026-11-01).  So a shape's layout is timed on each GPU on its first product
// (native_mmvq_tune, or the first call outside a graph's capture) and kept for the process: either layout gives every
// column the bits of its one-column call, and a shape keeping its layout keeps a token's rows the same in every
// window.  The steps rule stands where the shape is first met in a capture, and under STRATA_MMVQ_TUNE=0.
bool steps_rule(int blocks, int per_step) { return (blocks + per_step - 1) / per_step <= 10; }

// Timed by the host over back-to-back launches, as the decode runs them (HIP's profiling events added some 35 us to
// each kernel on an RX 9060 XT and turned its choices around).  Each timing lasts about kTuneUs, the two layouts in
// turn kTuneRounds times.  The launches read random weights at places through a span of VRAM several times the GPU's
// last cache (kTuneSpanCaches of it, at least kTuneSpanMin): the decode reads every other weight before a matrix
// comes again, so its rows come from VRAM, and the same rows read again and again stayed in an RX 9060 XT's 32 MiB
// Infinity Cache, where the sub-group's layout came out ahead and its decode the slower (4096-wide Q4_K: 54.8 us
// against the work-group's 66.2 in the cache, the decode 47.1 tok/s against 49.4).  The span is not the cache the
// device reports alone: HIP reports the L2 only.  Random, not zero: a GPU that compresses its memory may read a
// zeroed buffer faster than weights.  The other layout is taken only where it beats the rule's in all the rounds but
// one, by kTuneMargin: on an RTX 4070 every shape of Qwen3.5-9B was within 5% either way and the choices of those
// ties changed from run to run, the text with them, for no speed; at 0.95 an RX 9060 XT's 4096 x 12288 Q4_K (0.944)
// did too.  A shape that near the margin can still change its layout from one run to the next.  A product shorter than kTuneFloorUs keeps the
// rule: what it could save is small and its times are the least steady.
constexpr double kTuneUs = 3000.0;
constexpr int kTuneRounds = 7;
constexpr double kTuneMargin = 0.93;
constexpr double kTuneFloorUs = 10.0;
constexpr uint64_t kTuneSpanCaches = 4;
constexpr uint64_t kTuneSpanMin = 256ull << 20;
using LayoutKey = std::tuple<std::type_index, int, int>;   // the format, n_in, n_out
std::mutex layouts_mu;
std::map<LayoutKey, std::unique_ptr<core::PerDevice<char>>> layouts;

core::PerDevice<char>& layouts_for(const LayoutKey& key) {
    std::lock_guard<std::mutex> lock(layouts_mu);
    auto& p = layouts[key];
    if (!p) p = std::make_unique<core::PerDevice<char>>();
    return *p;
}

const char* format_name(const std::type_index& t);

// the faster layout of the shape on the queue's GPU (y is written; the caller's weights only where no span fits);
// true: the sub-group's
struct Regions {
    int n = 0;
    const void* const* ptrs = nullptr;
    const uint64_t* bytes = nullptr;
};

template<typename F>
bool time_layout(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                 const Regions& regions = {}) {
    const bool rule = steps_rule(n_in / F::DIV, WARP / F::T);
    uint8_t* span = nullptr;
    try {
        // the span: as many places for the matrix as fit, each on a 256-byte boundary
        const sycl::device dev = q.get_device();
        const uint64_t bytes = (uint64_t) n_out * (uint64_t) (n_in / F::DIV) * sizeof(typename F::Block);
        const uint64_t stride = (bytes + 255) / 256 * 256;
        uint64_t want = std::max<uint64_t>(kTuneSpanCaches * dev.get_info<sycl::info::device::global_mem_cache_size>(),
                                           kTuneSpanMin);
        want = std::min<uint64_t>(std::max(want, 2 * stride), dev.get_info<sycl::info::device::max_mem_alloc_size>());
        // the places: in the regions given (each a matrix's bytes at a 256-byte boundary within one), until they
        // cover the span; else a span allocated for them
        std::vector<const uint8_t*> at;
        uint64_t covered = 0;
        for (int r = 0; r < regions.n && covered < want; ++r) {
            const auto* base = static_cast<const uint8_t*>(regions.ptrs[r]);
            const uint64_t lead = (256 - reinterpret_cast<uintptr_t>(base) % 256) % 256;
            for (uint64_t o = lead; o + bytes <= regions.bytes[r] && covered < want; o += stride) {
                at.push_back(base + o);
                covered += stride;
            }
        }
        if (covered < want) at.clear();
        uint64_t got = 0;
        for (uint64_t w = want; at.empty() && w >= 2 * stride && span == nullptr; w /= 2) {
            span = sycl::malloc_device<uint8_t>(w, q);
            got = w;
        }
        const uint64_t places = !at.empty() ? at.size() : span != nullptr ? got / stride : 1;
        if (span != nullptr) {
            auto* w32 = reinterpret_cast<uint32_t*>(span);
            q.parallel_for(sycl::range<1>((size_t) (places * stride / 4)), [=](sycl::id<1> i) {
                uint32_t x = (uint32_t) i[0] * 2654435761u;
                x ^= x >> 13;
                x *= 0x5bd1e995u;
                w32[i] = x ^ (x >> 15);
            }).wait_and_throw();
        }
        uint64_t next = 0;
        auto run = [&](int l) {
            const void* w = !at.empty()       ? (const void*) at[next++ % places]
                            : span != nullptr ? (const void*) (span + (next++ % places) * stride)
                                              : weights;
            if (l == 1) launch_row<F, 1, 8>(q, w, x_q8_1, y, n_in, n_out);
            else launch_group<F, 1>(q, w, x_q8_1, y, n_in, n_out);
        };
        auto timed = [&](int l, int reps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) run(l);
            q.wait_and_throw();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        run(0);
        run(1);
        q.wait_and_throw();
        const int reps = (int) std::clamp(kTuneUs / std::max(timed(rule ? 1 : 0, 4), 1.0), 4.0, 2000.0);
        timed(0, reps);   // the GPU's clocks up before the rounds
        double us[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        double ratio[kTuneRounds];   // the other layout's time over the rule's, a round
        for (int round = 0; round < kTuneRounds; ++round) {
            const double g = timed(0, reps), r = timed(1, reps);
            us[0] = std::min(us[0], g);
            us[1] = std::min(us[1], r);
            ratio[round] = rule ? g / r : r / g;
        }
        std::sort(ratio, ratio + kTuneRounds);
        const double worse = ratio[kTuneRounds - 2];   // the other's second least favourable round
        const bool row = us[rule ? 1 : 0] >= kTuneFloorUs && worse < kTuneMargin ? !rule : rule;
        static const bool log_all = strata::env_get("MMVQ_TUNE_LOG") != nullptr;
        if (row != rule || log_all)
            std::fprintf(stderr, "xestrata: the decode products of %s %d -> %d on %s: %s a row (timed: %.1f us a "
                                 "work-group, %.1f a sub-group; the other's rounds at %.3f of the rule's or less "
                                 "but one)\n", format_name(std::type_index(typeid(F))), n_in, n_out,
                         dev.get_info<sycl::info::device::name>().c_str(), row ? "a sub-group" : "a work-group",
                         us[0], us[1], worse);
        if (span != nullptr) sycl::free(span, q);
        return row;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata: timing the decode products of %s %d -> %d: %s\n",
                     format_name(std::type_index(typeid(F))), n_in, n_out, e.what());
        if (span != nullptr) {
            q.wait();
            sycl::free(span, q);
        }
        return rule;
    }
}

bool env_off(const char* name) {
    const char* v = strata::env_get(name);
    return v != nullptr && std::strtol(v, nullptr, 10) == 0;
}

// the shape's layout on the queue's GPU (true: a sub-group a row), timed over the call's buffers the first time
template<typename F>
bool row_layout(sycl::queue& q, void* stream, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out) {
    static const bool tune = !env_off("MMVQ_TUNE");
    if (!tune) return steps_rule(n_in / F::DIV, WARP / F::T);
    return layouts_for(LayoutKey{std::type_index(typeid(F)), n_in, n_out}).get(q.get_device(), [&]() -> char {
        if (gpu::is_capturing(stream)) return steps_rule(n_in / F::DIV, WARP / F::T);
        return time_layout<F>(q, weights, x_q8_1, y, n_in, n_out);
    }) != 0;
}

template<typename F, int NCOLS>
void launch_n(sycl::queue& q, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, bool row) {
    if (NCOLS > 1 && !g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        launch_kernel<F, NCOLS, NW, 2>(q, weights, x_q8_1, y, n_in, n_out, (std::size_t(n_out) + 1) / 2);
        return;
    }
    if (row) launch_row<F, NCOLS, 8>(q, weights, x_q8_1, y, n_in, n_out);
    else launch_group<F, NCOLS>(q, weights, x_q8_1, y, n_in, n_out);
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 16");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
sycl::queue& validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit stream");
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

template<typename F>
void mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    auto& q = validate_stream(stream);
    const bool row = row_layout<F>(q, stream, weights, x_q8_1, y, n_in, n_out);
    switch (ncols) {
        case 1: launch_n<F, 1>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 2: launch_n<F, 2>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 3: launch_n<F, 3>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 4: launch_n<F, 4>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 5: launch_n<F, 5>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 6: launch_n<F, 6>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 7: launch_n<F, 7>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        case 8: launch_n<F, 8>(q, weights, x_q8_1, y, n_in, n_out, row); break;
        default:   // 9..16: native_mmvq_wide.cpp
            mmvq_xe::mmvq_wide(kWideFmt<F>, q, weights, x_q8_1, y, n_in, n_out, ncols, row, g_multi_exact);
            break;
    }
}

// native_mmvq_tune: the layout timed over a zero activation and an output of the shape's own
template<typename F>
void mmvq_tune(const void* weights, int n_in, int n_out, void* stream, const Regions& regions) {
    validate_shape(n_in, 1, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    if (env_off("MMVQ_TUNE") || gpu::is_capturing(stream)) return;
    layouts_for(LayoutKey{std::type_index(typeid(F)), n_in, n_out}).get(q.get_device(), [&]() -> char {
        void* x = sycl::malloc_device(native_q8_1_bytes(n_in, 1), q);
        float* y = sycl::malloc_device<float>((size_t) n_out, q);
        if (x == nullptr || y == nullptr) {   // no VRAM for it: the rule, as in a capture
            if (x != nullptr) sycl::free(x, q);
            if (y != nullptr) sycl::free(y, q);
            return steps_rule(n_in / F::DIV, WARP / F::T);
        }
        q.memset(x, 0, native_q8_1_bytes(n_in, 1)).wait_and_throw();
        const bool row = time_layout<F>(q, weights, x, y, n_in, n_out, regions);
        sycl::free(x, q);
        sycl::free(y, q);
        return row;
    });
}

// the pair in one launch where launch_n would take launch_row (a row of at most 10 steps; exact multi-column
// layout); false, launching nothing, otherwise
template<typename F>
bool mmvq_pair(const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2, int n_in, int n_out, int ncols,
               void* stream) {
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(w1);
    validate_pointer(w2);
    validate_pointer(x_q8_1);
    validate_pointer(y1);
    validate_pointer(y2);
    auto& q = validate_stream(stream);
    if ((ncols > 1 && !g_multi_exact) || !row_layout<F>(q, stream, w1, x_q8_1, y1, n_in, n_out)) return false;
    switch (ncols) {
        case 1: launch_row<F, 1, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 2: launch_row<F, 2, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 3: launch_row<F, 3, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 4: launch_row<F, 4, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 5: launch_row<F, 5, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 6: launch_row<F, 6, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 7: launch_row<F, 7, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        case 8: launch_row<F, 8, 8>(q, w1, x_q8_1, y1, n_in, n_out, w2, y2); break;
        default: return false;   // (a wider window: the two products' own launches)
    }
    return true;
}

template<typename F>
void mmvq_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    mmvq<F>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}


const char* format_name(const std::type_index& t) {
    const std::pair<std::type_index, const char*> names[] = {
        {typeid(Q5KTraits), "Q5_K"}, {typeid(Q4KTraits), "Q4_K"}, {typeid(Q20Traits), "Q2_0"},
        {typeid(Q3KTraits), "Q3_K"}, {typeid(Q2KTraits), "Q2_K"}, {typeid(Q6KTraits), "Q6_K"},
        {typeid(Q6KRowTraits), "Q6_K"}, {typeid(IQ4XSTraits), "IQ4_XS"}, {typeid(Q40Traits), "Q4_0"},
        {typeid(Q50Traits), "Q5_0"}, {typeid(Q80Traits), "Q8_0"}, {typeid(Q80RowTraits), "Q8_0"},
        {typeid(IQ4NLTraits), "IQ4_NL"}};
    for (const auto& [ti, name] : names)
        if (ti == t) return name;
    return "a format";
}

} // namespace

// native_swiglu_q8_1's values, in a namespace of their own (SYCL names a kernel by its type)
namespace native_q8 {
struct Swiglu {
    static constexpr bool kRows = false;
    const float *gate, *up;
    float operator()(int64_t, int64_t c, int j) const {
        // contraction off: the SwiGLU product must not fuse into the sub-group sum's first add (the stored and
        // reloaded value of the two launches is rounded on its own)
#pragma clang fp contract(off)
        const int64_t i = c * Q8K + j;
        const float g = gate[i];
        return g / (1.0f + sycl::exp(-g)) * up[i];   // native_swiglu's expression
    }
};
}  // namespace native_q8

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    native_quantize_q8_1_rows(x, x_q8_1, n_in, ncols, stream);
}

std::size_t native_q8_1_rows_bytes(int n_in, int64_t rows) {
    if (n_in <= 0 || n_in % Q8K != 0 || rows < 0)
        throw std::invalid_argument("native Q8_1 rows require n_in > 0 and divisible by 32, and rows >= 0");
    return std::size_t(rows) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1_rows(const float* x, void* x_q8_1, int n_in, int64_t rows, void* stream) {
    if (n_in <= 0 || n_in % Q8K != 0 || rows < 1 || (int64_t) n_in * rows > INT32_MAX)
        throw std::invalid_argument("native Q8_1 rows require n_in > 0 and divisible by 32, and 1 <= rows");
    validate_pointer(x);
    validate_pointer(x_q8_1);
    auto& q = validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    q8act::quantize<1, q8act::Ggml>(q, (int64_t) n_in * rows / Q8K, 0, q8act::Flat{x},
                                    q8act::Q8_1{static_cast<uint8_t*>(x_q8_1)});
}

void native_swiglu_q8_1(const float* gate, const float* up, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    auto& q = validate_stream(stream);
    q8act::quantize<1, q8act::Ggml>(q, (int64_t) n_in * ncols / Q8K, 0, native_q8::Swiglu{gate, up},
                                    q8act::Q8_1{static_cast<uint8_t*>(x_q8_1)});
}

void native_swiglu_f32(const float* gate, const float* up, float* out, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(out);
    auto& q = validate_stream(stream);
    q.parallel_for(sycl::range<1>((std::size_t) n_in * (std::size_t) ncols), [=](sycl::id<1> id) {
#pragma clang fp contract(off)
        const std::size_t i = id[0];
        const float g = gate[i];
        out[i] = g / (1.0f + sycl::exp(-g)) * up[i];
    });
}

void native_q5_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q5KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q5_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q5KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q2_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q20Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q2_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q20Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q2_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q2KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q2_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q2KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q3_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q3KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q3_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q3KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_iq4_xs_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<IQ4XSTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_iq4_xs_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<IQ4XSTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q4_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q4KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q4_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q4KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q6_k_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q6KTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_q6_k_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q6KTraits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q4_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q40Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q4_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q40Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q5_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q50Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q5_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q50Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_q8_0_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<Q80Traits>(w, x, y, n_in, n_out, ncols, s); }
void native_q8_0_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<Q80Traits>(w, x, xq, y, n_in, n_out, ncols, s); }
void native_iq4_nl_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) { mmvq<IQ4NLTraits>(w, x, y, n_in, n_out, ncols, s); }
void native_iq4_nl_f32(const void* w, const float* x, void* xq, float* y, int n_in, int n_out, int ncols, void* s) { mmvq_f32<IQ4NLTraits>(w, x, xq, y, n_in, n_out, ncols, s); }

bool native_q6_k_rows_ok(int n_in) noexcept { return n_in > 0 && n_in % 512 == 0; }
bool native_q8_0_rows_ok(int n_in) noexcept { return n_in > 0 && n_in % 64 == 0; }

void native_q8_0_to_rows(void* weights, int n_in, int n_out, void* stream) {
    if (!native_q8_0_rows_ok(n_in)) throw std::invalid_argument("Q8_0 rows need an even number of blocks per row");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    const int bpr = n_in / 32;
    const std::size_t row_bytes = std::size_t(bpr) * sizeof(Q80Block);
    auto* w = static_cast<uint8_t*>(weights);
    constexpr int WG = 256;
    // as native_q6_k_to_rows: one work-group a row, through local memory, in place
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> row(sycl::range<1>(row_bytes), h);
        h.parallel_for(sycl::nd_range<1>(std::size_t(n_out) * WG, WG), [=](sycl::nd_item<1> it) {
            uint8_t* r = w + it.get_group(0) * row_bytes;
            const int t = (int) it.get_local_id(0);
            for (std::size_t i = t; i < row_bytes; i += WG) row[i] = r[i];
            sycl::group_barrier(it.get_group());
            for (std::size_t i = t; i < row_bytes; i += WG) {
                const std::size_t b = i / sizeof(Q80Block), o = i % sizeof(Q80Block);
                r[o < 2 ? std::size_t(bpr) * 32 + b * 2 + o : b * 32 + (o - 2)] = row[i];
            }
        });
    });
}

void native_q6_k_to_rows(void* weights, int n_in, int n_out, void* stream) {
    if (!native_q6_k_rows_ok(n_in)) throw std::invalid_argument("Q6_K rows need an even number of blocks per row");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    auto& q = validate_stream(stream);
    const int bpr = n_in / QK;
    const std::size_t row_bytes = std::size_t(bpr) * sizeof(Q6KBlock);
    auto* w = static_cast<uint8_t*>(weights);
    constexpr int WG = 256;
    // One work-group a row: the row goes to local memory, then each byte is written to its place in the split
    // arrays, so the rewrite is in place and needs no second buffer.
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> row(sycl::range<1>(row_bytes), h);
        h.parallel_for(sycl::nd_range<1>(std::size_t(n_out) * WG, WG), [=](sycl::nd_item<1> it) {
            uint8_t* r = w + it.get_group(0) * row_bytes;
            const int t = (int) it.get_local_id(0);
            for (std::size_t i = t; i < row_bytes; i += WG) row[i] = r[i];
            sycl::group_barrier(it.get_group());
            for (std::size_t i = t; i < row_bytes; i += WG) {
                const std::size_t b = i / sizeof(Q6KBlock), o = i % sizeof(Q6KBlock);
                std::size_t to;
                if (o < 128) to = b * 128 + o;
                else if (o < 192) to = std::size_t(bpr) * 128 + b * 64 + (o - 128);
                else if (o < 208) to = std::size_t(bpr) * 192 + b * 16 + (o - 192);
                else to = std::size_t(bpr) * 208 + b * 2 + (o - 208);
                r[to] = row[i];
            }
        });
    });
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == kNativeQ6KRows || ggml_type == kNativeQ8Rows || ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 ||
           ggml_type == 10 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 || ggml_type == 19;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;   // Q5_1 (OrcaRouter Q4_K_S, upstream d652cd6)
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 10: block_elems = 256; block_bytes = 84; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case kNativeQ6KRows: block_elems = 512; block_bytes = 420; break;   // whole rows of an even block count
    case kNativeQ8Rows: block_elems = 64; block_bytes = 68; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: return mmvq_pair<Q40Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 6: return mmvq_pair<Q50Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 8: return mmvq_pair<Q80Traits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 20: return mmvq_pair<IQ4NLTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 14: return mmvq_pair<Q6KTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case kNativeQ6KRows:
        return native_q6_k_rows_ok(n_in) && mmvq_pair<Q6KRowTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case kNativeQ8Rows:
        return native_q8_0_rows_ok(n_in) && mmvq_pair<Q80RowTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    case 23: return mmvq_pair<IQ4XSTraits>(w1, w2, x_q8_1, y1, y2, n_in, n_out, ncols, stream);
    default: return false;   // the K-quants' and i-quants' own kernels: two launches
    }
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 10: native_q2_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case kNativeQ6KRows:
        if (!native_q6_k_rows_ok(n_in)) throw std::invalid_argument("Q6_K rows need an even number of blocks per row");
        mmvq<Q6KRowTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        break;
    case kNativeQ8Rows:
        if (!native_q8_0_rows_ok(n_in)) throw std::invalid_argument("Q8_0 rows need an even number of blocks per row");
        mmvq<Q80RowTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

void native_mmvq_tune(int ggml_type, const void* weights, int n_in, int n_out, void* stream, int regions,
                      const void* const* region_ptrs, const uint64_t* region_bytes) {
    const Regions r{regions, region_ptrs, region_bytes};
    switch (ggml_type) {
    case 2: mmvq_tune<Q40Traits>(weights, n_in, n_out, stream, r); break;
    case 6: mmvq_tune<Q50Traits>(weights, n_in, n_out, stream, r); break;
    case 8: mmvq_tune<Q80Traits>(weights, n_in, n_out, stream, r); break;
    case 20: mmvq_tune<IQ4NLTraits>(weights, n_in, n_out, stream, r); break;
    case 10: mmvq_tune<Q2KTraits>(weights, n_in, n_out, stream, r); break;
    case 11: mmvq_tune<Q3KTraits>(weights, n_in, n_out, stream, r); break;
    case 12: mmvq_tune<Q4KTraits>(weights, n_in, n_out, stream, r); break;
    case 13: mmvq_tune<Q5KTraits>(weights, n_in, n_out, stream, r); break;
    case 14: mmvq_tune<Q6KTraits>(weights, n_in, n_out, stream, r); break;
    case kNativeQ6KRows:
        if (native_q6_k_rows_ok(n_in)) mmvq_tune<Q6KRowTraits>(weights, n_in, n_out, stream, r);
        break;
    case kNativeQ8Rows:
        if (native_q8_0_rows_ok(n_in)) mmvq_tune<Q80RowTraits>(weights, n_in, n_out, stream, r);
        break;
    case 23: mmvq_tune<IQ4XSTraits>(weights, n_in, n_out, stream, r); break;
    case 42: mmvq_tune<Q20Traits>(weights, n_in, n_out, stream, r); break;
    default: break;   // the i-quants' own kernels (iq_mmvq), BF16: one layout
    }
}

} // namespace strata::kernels
