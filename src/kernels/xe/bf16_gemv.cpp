// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/bf16_gemv.cpp - the Xe ports of Strata's src/kernels/cuda/bf16_gemv.cu (BF16 x BF16 GEMV) and the FP32 x
// BF16 matrix-vector kernels of Strata's src/kernels/cuda/native_bf16.cu (ggml-cuda's mmvf at llama.cpp 3cf03257; MIT
// License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE).
//
// Warp sums are xor butterflies.  Lane 0 of a butterfly adds the same partial sums in the same order as CUDA's
// shuffle-down tree, so the row results of the warp-per-row kernel are those of the CUDA kernel.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/per_device.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}

sycl::event launch_naive(sycl::queue& q, const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    return q.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> id) {
        const int64_t o = (int64_t) id[0];
        const uint16_t* row = w + o * n_in;
        float acc = 0.0f;
        for (int64_t i = 0; i < n_in; ++i) acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
        y[o] = acc;
    });
}

// One sub-group per row, coalesced: consecutive lanes read consecutive weights.
sycl::event launch_warp(sycl::queue& q, const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((n_out + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t o = (int64_t) it.get_group(0) * warps + (int64_t) sg.get_group_linear_id();
        if (o >= n_out) return;
        const int lane = (int) sg.get_local_linear_id();
        const uint16_t* row = w + o * n_in;
        float acc = 0.0f;
        for (int64_t i = lane; i < n_in; i += WARP) acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
        acc = warp_sum(sg, acc);
        if (lane == 0) y[o] = acc;
    });
}

// ggml-cuda mmvf: work-group size chosen for the fewest pair iterations, two ordered FMAs per pair, a warp sum, and
// for more than one warp a second warp sum over 32 partials (zero beyond the warp count).  EXACT: n_tok is NT (the
// rows' loop fixed at compile time; upstream 104a8485), else the first n_tok of NT rows.
template<int BLOCK_SIZE, int NT, bool EXACT = false>
sycl::event launch_mmvf(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                        int n_in, int64_t n_out, int n_tok_arg) {
    const int n_tok = EXACT ? NT : n_tok_arg;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(NT * 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * BLOCK_SIZE, BLOCK_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_local_id(0);
            const size_t rowi = it.get_group(0);
            const uint16_t* row = w + rowi * n_in;
            const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
            if constexpr (BLOCK_SIZE > 32) {
                if (t < 32)
                    for (int k = 0; k < NT; ++k) partials[k * 32 + t] = 0.0f;
                sycl::group_barrier(it.get_group());
            }
            float acc[NT];
            for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
            for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                const uint32_t weight = weights2[pair];
                const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
                for (int k = 0; k < NT; ++k) {
                    if (k < n_tok) {
                        const float* xin = x + (size_t) k * ldx + 2 * (size_t) pair;
                        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
                        acc[k] = sycl::fma(w0, xin[0], acc[k]);
                        acc[k] = sycl::fma(w1, xin[1], acc[k]);
                    }
                }
            }
            for (int k = 0; k < NT; ++k) acc[k] = warp_sum(sg, acc[k]);
            if constexpr (BLOCK_SIZE > 32) {
                // All lanes have the same reduced value; one store avoids a same-value local-memory race.
                if ((t & 31) == 0)
                    for (int k = 0; k < NT; ++k) partials[k * 32 + t / 32] = acc[k];
                sycl::group_barrier(it.get_group());
                if (t < 32)
                    for (int k = 0; k < NT; ++k) acc[k] = warp_sum(sg, partials[k * 32 + t]);
            }
            if (t == 0)
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) y[(size_t) k * ldy + rowi] = acc[k];
        });
    });
}

// upstream c637ed5a: RPB output rows a work-group, the activations read once for them; per output, work-item t walks
// pairs t, t + BLOCK_SIZE, ... with launch_mmvf's two ordered FMAs and its sub-group and work-group sums.
template<int BLOCK_SIZE, int NT, int RPB, bool EXACT = false>
sycl::event launch_mmvf_rows(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                             int n_in, int64_t n_out, int n_tok_arg) {
    const int n_tok = EXACT ? NT : n_tok_arg;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) RPB * NT * 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) ((n_out + RPB - 1) / RPB) * BLOCK_SIZE, BLOCK_SIZE),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_local_id(0);
            const int64_t o0 = (int64_t) it.get_group(0) * RPB;
            if constexpr (BLOCK_SIZE > 32) {
                if (t < 32)
                    for (int r = 0; r < RPB; ++r)
                        for (int k = 0; k < NT; ++k) partials[(r * NT + k) * 32 + t] = 0.0f;
                sycl::group_barrier(it.get_group());
            }
            float acc[RPB][NT];
            for (int r = 0; r < RPB; ++r)
                for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
            for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                float in[NT][2];
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) {
                        const float* xin = x + (size_t) k * ldx + 2 * (size_t) pair;
                        in[k][0] = xin[0];
                        in[k][1] = xin[1];
                    }
                for (int r = 0; r < RPB; ++r) {
                    if (o0 + r >= n_out) break;
                    const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) (o0 + r) * n_in)[pair];
                    const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
                    for (int k = 0; k < NT; ++k)
                        if (k < n_tok) {
                            acc[r][k] = sycl::fma(w0, in[k][0], acc[r][k]);
                            acc[r][k] = sycl::fma(w1, in[k][1], acc[r][k]);
                        }
                }
            }
            for (int r = 0; r < RPB; ++r)
                for (int k = 0; k < NT; ++k) acc[r][k] = warp_sum(sg, acc[r][k]);
            if constexpr (BLOCK_SIZE > 32) {
                if ((t & 31) == 0)
                    for (int r = 0; r < RPB; ++r)
                        for (int k = 0; k < NT; ++k) partials[(r * NT + k) * 32 + t / 32] = acc[r][k];
                sycl::group_barrier(it.get_group());
                if (t < 32)
                    for (int r = 0; r < RPB; ++r)
                        for (int k = 0; k < NT; ++k) acc[r][k] = warp_sum(sg, partials[(r * NT + k) * 32 + t]);
            }
            if (t == 0)
                for (int r = 0; r < RPB; ++r)
                    if (o0 + r < n_out)
                        for (int k = 0; k < NT; ++k)
                            if (k < n_tok) y[(size_t) k * ldy + o0 + r] = acc[r][k];
        });
    });
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

// Which of the two forms (a row a work-group, 4 rows a work-group: the same bits) is faster depends on the device, the
// shape and the row count (the 4-rows form was faster on the RTX 4070 for 2-3 rows and much slower from 5, and slower
// on the B70): bf16_gemv_fp32_mmvf_multi_tune times both and records the row counts that take the 4-rows form, per
// device and shape.  STRATA_MMVF_ROWS=0|1 fixes the form.
struct RowsTable {
    std::mutex mutex;
    std::map<std::pair<int64_t, int64_t>, uint32_t> rows_mask;   // (n_in, n_out) -> bit T: 4 rows a work-group
};
RowsTable& rows_table(sycl::queue& q) {
    static core::PerDevice<std::shared_ptr<RowsTable>> tables;
    return *tables.get(q.get_device(), [] { return std::make_shared<RowsTable>(); });
}
int rows_forced() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_MMVF_ROWS");
        return v == nullptr ? -1 : (std::strtol(v, nullptr, 10) != 0 ? 1 : 0);
    }();
    return forced;
}
bool rows_form(sycl::queue& q, int64_t n_in, int64_t n_out, int n_tok) {
    const int forced = rows_forced();
    if (forced >= 0) return forced == 1;
    auto& table = rows_table(q);
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.rows_mask.find({n_in, n_out});
    return it != table.rows_mask.end() && ((it->second >> n_tok) & 1u) != 0;
}

template<int NT, bool EXACT = false>
void dispatch_mmvf(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                   int64_t n_in, int64_t n_out, int n_tok, bool rows = false) {
    if (NT > 1 && rows) {
        switch (mmvf_block_size(n_in)) {
            case 32: launch_mmvf_rows<32, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 64: launch_mmvf_rows<64, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 96: launch_mmvf_rows<96, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 128: launch_mmvf_rows<128, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 160: launch_mmvf_rows<160, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 192: launch_mmvf_rows<192, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            case 224: launch_mmvf_rows<224, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
            default: launch_mmvf_rows<256, NT, 4, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        }
        return;
    }
    switch (mmvf_block_size(n_in)) {
        case 32: launch_mmvf<32, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 64: launch_mmvf<64, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 96: launch_mmvf<96, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 128: launch_mmvf<128, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 160: launch_mmvf<160, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 192: launch_mmvf<192, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 224: launch_mmvf<224, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
        case 256: launch_mmvf<256, NT, EXACT>(q, x, ldx, w, y, ldy, (int) n_in, n_out, n_tok); break;
    }
}

// the multi-row call's kernels: the window's usual row counts with their loop fixed at compile time (upstream
// 104a8485); rows: 4 rows a work-group
void multi_dispatch(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                    int64_t n_in, int64_t n_out, int n_tok, bool rows) {
    switch (n_tok) {
        case 2: dispatch_mmvf<2, true>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows); break;
        case 3: dispatch_mmvf<3, true>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows); break;
        case 4: dispatch_mmvf<4, true>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows); break;
        case 5: dispatch_mmvf<5, true>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows); break;
        case 6: dispatch_mmvf<6, true>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows); break;
        default:
            if (n_tok <= 4) dispatch_mmvf<4>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows);
            else dispatch_mmvf<8>(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, rows);
    }
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    auto& q = queue_for(stream);
    // Warp-per-row when there are enough rows to fill the machine; see the CUDA file for the measurements.
    if (n_out >= 64) {
        finish(stream, launch_warp(q, x, w, y, n_in, n_out), "bf16_gemv(warp)");
        return;
    }
    finish(stream, launch_naive(q, x, w, y, n_in, n_out), "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    auto& q = queue_for(stream);
    if (threads_per_row == 32) {
        finish(stream, launch_warp(q, x, w, y, n_in, n_out), "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0)
        throw core::DeviceError("bf16_gemv_split: threads_per_row " + std::to_string(threads_per_row) +
                                " must be a power of two (32 selects the warp-per-row path)");
    const int tpr = threads_per_row;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>((size_t) tpr), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * tpr, tpr), [=](sycl::nd_item<1> it) {
            const int64_t o = (int64_t) it.get_group(0);
            const int t = (int) it.get_local_id(0);
            const uint16_t* row = w + o * n_in;
            float acc = 0.0f;
            for (int64_t i = t; i < n_in; i += tpr) acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
            // a tree in local memory, as CUDA's shared-memory tree
            scratch[t] = acc;
            sycl::group_barrier(it.get_group());
            for (int off = tpr >> 1; off > 0; off >>= 1) {
                if (t < off) scratch[t] += scratch[t + off];
                sycl::group_barrier(it.get_group());
            }
            if (t == 0) y[o] = scratch[0];
        });
    });
    finish(stream, e, "bf16_gemv_split");
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    auto& q = queue_for(stream);
    multi_dispatch(q, x, ldx, w, y, ldy, n_in, n_out, n_tok, n_out >= 64 && rows_form(q, n_in, n_out, n_tok));
    if (!stream) core::Runtime::get().finish(q);
}

void bf16_gemv_fp32_mmvf_multi_tune(int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_out < 64 || n_in > std::numeric_limits<int>::max()) return;
    auto& q = queue_for(stream);
    auto& table = rows_table(q);
    {
        std::lock_guard<std::mutex> lock(table.mutex);
        if (table.rows_mask.count({n_in, n_out})) return;
    }
    float* x = sycl::malloc_device<float>((size_t) 8 * n_in, q);
    uint16_t* w = sycl::malloc_device<uint16_t>((size_t) n_out * n_in, q);
    float* y = sycl::malloc_device<float>((size_t) 8 * n_out, q);
    if (!x || !w || !y) {
        sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
        throw core::DeviceError("bf16_gemv_fp32_mmvf_multi_tune: device allocation failed");
    }
    q.fill(x, 0.5f, (size_t) 8 * n_in);
    q.fill(w, (uint16_t) 0x3c00, (size_t) n_out * n_in).wait();
    // timed as the engine runs them: 32 launches of each form in a graph, the two graphs replayed in turns (the first
    // round a warm-up), each form's fastest round kept
    namespace sx = sycl::ext::oneapi::experimental;
    uint32_t mask = 0;
    for (int T = 2; T <= 8; ++T) {
        std::vector<sx::command_graph<sx::graph_state::executable>> exec;
        for (int rows = 0; rows < 2; ++rows) {
            sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
            graph.begin_recording(q);
            for (int i = 0; i < 32; ++i) multi_dispatch(q, x, n_in, w, y, n_out, n_in, n_out, T, rows == 1);
            graph.end_recording(q);
            exec.push_back(graph.finalize());
        }
        double t_min[2] = {};
        for (int round = 0; round < 6; ++round)
            for (int rows = 0; rows < 2; ++rows) {
                const auto t0 = std::chrono::steady_clock::now();
                q.ext_oneapi_graph(exec[rows]);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (round > 0 && (t_min[rows] == 0.0 || us < t_min[rows])) t_min[rows] = us;
            }
        if (t_min[1] < t_min[0]) mask |= 1u << T;
    }
    sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
    if (mask != 0) {
        std::string counts;
        for (int T = 2; T <= 8; ++T)
            if ((mask >> T) & 1u) counts += (counts.empty() ? "" : ",") + std::to_string(T);
        std::fprintf(stderr, "strata: the %lld x %lld BF16 GEMV takes 4 rows a work-group for %s rows\n",
                     (long long) n_in, (long long) n_out, counts.c_str());
    }
    std::lock_guard<std::mutex> lock(table.mutex);
    table.rows_mask[{n_in, n_out}] = mask;
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    auto& q = queue_for(stream);
    dispatch_mmvf<1>(q, x, n_in, w, y, n_out, n_in, n_out, 1);
    if (!stream) core::Runtime::get().finish(q);
}

}  // namespace strata::kernels
