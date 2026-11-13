// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/qsa_decode_attn.cpp - the Xe port of Strata's src/kernels/cuda/qsa_decode_attn.cu (see the header): split-K
// decode attention over the selected cells for fp16, INT8 and Q4_0 KV, then a merge of the chunk partials.
//
// The head size HD and the query heads per KV head G are template parameters: (256, 12) is Qwen3.8's QSA, (256, 8)
// Ornith-1.5's, (128, 10) llm-jp-4.1's and (128, 6), (128, 16) and (128, 8) PLaMo 3 Translate's (2B, 0.8B and 8B)
// full attention, which read EVERY cell (`qsa_dense_attn_*`: no selection, cell c of the cache for slot c), or with a
// window the last `window` cells only (cell lo + c for slot c, lo = max(0, n_kv - window): PLaMo 3's sliding-window
// layers).  A work-group has HD work-items, one per dimension in the value loop; a lane holds HD/32 dimensions in the
// score loop.  For (256, 12) every expression is the one this file had with HD and G fixed.
//
// A warp is a sub-group of 32.  CUDA used the __expf intrinsic here; this uses the precise exp.  Explicit fmaf calls
// stay fused (sycl::fma), and the library's -ffp-contract=off leaves every other multiply-add as written.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_qsa_decode_attn
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int CHUNK = 64;        // cells per work-group (at most: chunk_of)

// The cells a chunk takes: CHUNK, or for every cell (a dense query, no selection) on a short context fewer, so that its
// cells spread over kShortChunks chunks of `min_chunk` cells or more: one group took a 64-cell chunk's heads in turn,
// 11 us a layer on the RTX 4070 where llama.cpp's took 4.  From kShortChunks * CHUNK cells on, whole chunks: smaller
// ones there cost more in partials than the groups gain (16-cell chunks: PLaMo 3 Translate 0.8B's decode after 12K
// cells 447 -> 429 tok/s on the B70).  The selection's chunks (Qwen3.8-Flash-Next's) stay whole.
constexpr int kShortChunks = 8;
inline int chunk_of(bool dense, int n_ids, int n_slots, int min_chunk) {
    if (!dense || min_chunk >= CHUNK) return CHUNK;
    // (a slot takes one chunk at most where it keeps its partials: n_slots chunks at most)
    const int per = sycl::max((n_ids + kShortChunks - 1) / kShortChunks, (n_ids + n_slots - 1) / n_slots);
    return per < min_chunk ? min_chunk : per > CHUNK ? CHUNK : per;
}
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// The G <= 16 heads' lane sums (`part[G..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5 (upstream fd95405).  Lane l ends holding head head_of_lane(l); lanes l and l^1
// agree.
inline float reduce12(const sycl::sub_group& sg, const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + sycl::permute_group_by_xor(sg, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + sycl::permute_group_by_xor(sg, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + sycl::permute_group_by_xor(sg, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + sycl::permute_group_by_xor(sg, hi2 ? r2[0] : r2[1], 2);
    return r1 + sycl::permute_group_by_xor(sg, r1, 1);
}
// reduce12 for G <= 4 heads: the same pairing order (xor 16, 8, 4, 2, 1), so the same sums bit for bit, in 6 shuffles
// instead of 16.  Lane l ends holding head head_of_lane4(l); the 8 lanes l & ~7 .. l | 7 agree.
inline float reduce4(const sycl::sub_group& sg, const float (&part)[4], int lane) {
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 16) != 0;
        r2[i] = (hi ? part[i + 2] : part[i]) + sycl::permute_group_by_xor(sg, hi ? part[i] : part[i + 2], 16);
    }
    const bool hi8 = (lane & 8) != 0;
    float r = (hi8 ? r2[1] : r2[0]) + sycl::permute_group_by_xor(sg, hi8 ? r2[0] : r2[1], 8);
    r += sycl::permute_group_by_xor(sg, r, 4);
    r += sycl::permute_group_by_xor(sg, r, 2);
    return r + sycl::permute_group_by_xor(sg, r, 1);
}
inline int head_of_lane4(int lane) { return ((lane >> 4) & 1) * 2 + ((lane >> 3) & 1); }
inline int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own work-item), per KV format.
template <int HD, int KV_MODE>
inline float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return h2f(p.v_pool[row * HD + d]);
    } else if constexpr (KV_MODE == 1) {
        const float sc = h2f(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]);
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = h2f(blk->d);
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

// N = HD / 32 consecutive values of one cell's key or value row, dimensions [d0, d0+N).  KV_MODE 0: FP16, 1: INT8,
// 2: rotated Q4_0, 3: the hybrid K8V4 (K as INT8, V as rotated Q4_0; upstream 2aa8f72).  N divides 16, so the run
// stays inside one Q4_0 half (the low nibbles hold elements 0-15 of a block, the high ones 16-31).
template <int HD, int KV_MODE>
inline void loadn(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int N = HD / WARP;
    if constexpr (KV_MODE == 3) {
        if (value) loadn<HD, 2>(p, true, row, d0, out);
        else loadn<HD, 1>(p, false, row, d0, out);
    } else if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        for (int j = 0; j < N; ++j) out[j] = h2f(base[j]);
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const float sc = h2f((value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
        for (int j = 0; j < N; ++j) out[j] = (float) codes[j] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = h2f(blk->d);
        const uint8_t* bytes = blk->qs + (rem & 15);
        if (rem < 16) {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}

// One lane's share of q.k: the N products summed left to right, which for N = 8 is the expression this file wrote out.
template <int N>
inline float dotn(const float* k, const float* q) {
    float s = k[0] * q[0];
#pragma unroll
    for (int e = 1; e < N; ++e) s = s + k[e] * q[e];
    return s;
}

// LOOP: a slot reads more than one chunk (n_slots below the chunks); without it every expression is the one this file
// had before the slots.
// ROWS: a batch window's rows, each with its own slot's pools.  A template argument, not a test of rows.n in the
// kernel: the pools picked per work-item cost the B70 8 times the kernel's time (69 -> 532 us at 8192 cells) and the
// RTX 4070 some, for every caller.
template <int HD, int G, int KV_MODE, bool LOOP, int NC, bool ROWS>
sycl::event launch_chunk(sycl::queue& queue, const float* q_all, const QsaAttnPools pools, const int32_t* ids_all,
                         const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                         float* part_m_all, float* part_l_all, int n_slots, size_t n_q, int cap,
                         long long scratch_stride, int window, int kv_div, int min_chunk,
                         const QsaAttnPoolsRows rows) {
    static_assert(G >= 1 && G <= 16, "reduce12 sums at most 16 heads");
    static_assert(HD % 128 == 0 && HD <= 256, "a lane holds HD/32 dimensions inside one Q4_0 half");
    constexpr int THREADS = HD;           // one work-item per dimension in the value loop
    constexpr int WARPS = THREADS / WARP;
    constexpr int N = HD / WARP;          // dimensions a lane holds in the score loop
    constexpr int GP = (G + 3) / 4 * 4;   // a cell's weights padded to whole float4s
    // ids_all null: every cell, slot c of the chunk is cell c0 + c, and the count is the cache's (step[kStepNKv])
    const bool dense = ids_all == nullptr;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(G * HD), hd);          // this KV head's query heads
        sycl::local_accessor<float, 1> sp(sycl::range<1>((size_t) G * CHUNK), hd);       // scores
        sycl::local_accessor<sycl::float4, 1> spt(sycl::range<1>(CHUNK * GP / 4), hd);  // probabilities, cell-major
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);     // pool row of each cell
        sycl::local_accessor<float, 1> sml(sycl::range<1>(2 * G), hd);           // a chunk's max and sum per head
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_slots * THREADS}, {1, 1, THREADS}),
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            // batched form: query z, with its own q row, selection, step and scratch; a batch window's row reads
            // its own slot's pools (ROWS)
            const QsaAttnPools& p = [&]() -> const QsaAttnPools& {
                if constexpr (ROWS) return rows.p[z];
                else return pools;
            }();
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = dense ? nullptr : ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_all = dense ? step[kStepNKv] : step[kStepWidth];
            const int lo = dense && window > 0 ? sycl::max(0, n_all - window) : 0;   // a window's first cell
            const int n_ids = n_all - lo;
            // kvh: the group's query heads (G of them); kv_div groups share a KV head (the query heads split
            // across work-groups, each reading the KV head's cells: kv_div 1, one group a KV head)
            const int first = (int) it.get_group(2), kvh = (int) it.get_group(1), kv_real = kvh / kv_div;
            const int n_kv_real = n_kv_heads / kv_div;
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int slot = kvh * n_slots + first;
            const int csz = chunk_of(dense, n_ids, n_slots, min_chunk);
            if (first * csz >= n_ids) {                           // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            for (int i = t; i < G * HD; i += THREADS) sq[i] = q[(size_t) (kvh * G) * HD + i];
            // chunks first, first + n_slots, ... in turn, each folded into the running max, sum and values (online
            // softmax); with one chunk a slot (n_slots the chunks) these are the chunk's own, as before
            float run_m[G], run_l[G], run_acc[G];
            for (int h = 0; h < G; ++h) { run_m[h] = -FLT_MAX; run_l[h] = 0.0f; run_acc[h] = 0.0f; }
            for (int chunk = first; chunk * csz < n_ids; chunk += n_slots) {
                const int c0 = chunk * csz;
                const int n_here = sycl::min(csz, n_ids - c0);
                if (t < CHUNK) {
                    long long r = -1;
                    if (t < n_here) {
                        const int cell = dense ? lo + c0 + t : ids[c0 + t];
                        const long long page = (long long) p.page_table[cell / page_size];
                        // a block the KV streaming could not make resident keeps page -1; its cells are masked (score
                        // -FLT_MAX, weight 0) instead of being read from before the pool (upstream f3925cf)
                        if (page >= 0) r = (page * n_kv_real + kv_real) * page_size + (cell % page_size);
                    }
                    srow[t] = r;
                }
                sycl::group_barrier(grp);
                // scores: each sub-group takes cells warp, warp+WARPS, ...; each lane holds N of the HD dimensions.
                // NC cells per step (warp + WARPS * (i + j)): a lane's q slice comes out of local memory once for all
                // of them (upstream fd95405); every dot product is the same expression as for a single cell, so the
                // scores are unchanged.  NC is the GPU's own (form_of, timed by qsa_dense_attn_tune).
                const int hd_lane = head_of_lane(lane);
                for (int i = 0; i < (n_here + WARPS - 1) / WARPS; i += NC) {
                    int cc[NC];
                    bool all_ok = true;
                    for (int j = 0; j < NC; ++j) {
                        cc[j] = warp + WARPS * (i + j);
                        all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
                    }
                    if (all_ok) {
                        float kk[NC][N];
                        for (int j = 0; j < NC; ++j) loadn<HD, KV_MODE>(p, false, srow[cc[j]], lane * N, kk[j]);
                        constexpr int HP = G <= 4 ? 4 : 16;   // the heads reduce4 or reduce12 sums
                        float pp[NC][HP];
    #pragma unroll
                        for (int h = 0; h < G; ++h) {
                            const float* qa = &sq[h * HD + lane * N];
    #pragma unroll
                            for (int j = 0; j < NC; ++j) pp[j][h] = dotn<N>(kk[j], qa);
                        }
    #pragma unroll
                        for (int j = 0; j < NC; ++j) {
                            for (int h = G; h < HP; ++h) pp[j][h] = 0.0f;
                            if constexpr (G <= 4) {
                                const float sj = reduce4(sg, pp[j], lane);
                                if ((lane & 7) == 0 && head_of_lane4(lane) < G)
                                    sp[head_of_lane4(lane) * CHUNK + cc[j]] = sj * scale;
                            } else {
                                const float sj = reduce12(sg, pp[j], lane);
                                if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + cc[j]] = sj * scale;
                            }
                        }
                    } else {
                        // a masked or out-of-range cell in the step: one cell at a time
                        for (int j = 0; j < NC; ++j) {
                            const int c = cc[j];
                            if (c >= n_here || srow[c] < 0) {
                                if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                                continue;
                            }
                            float kn[N];
                            loadn<HD, KV_MODE>(p, false, srow[c], lane * N, kn);
                            constexpr int HP = G <= 4 ? 4 : 16;
                            float part[HP];
                            for (int h = 0; h < G; ++h) part[h] = dotn<N>(kn, &sq[h * HD + lane * N]);
                            for (int h = G; h < HP; ++h) part[h] = 0.0f;
                            if constexpr (G <= 4) {
                                const float sc = reduce4(sg, part, lane);
                                if ((lane & 7) == 0 && head_of_lane4(lane) < G)
                                    sp[head_of_lane4(lane) * CHUNK + c] = sc * scale;
                            } else {
                                const float sc = reduce12(sg, part, lane);
                                if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + c] = sc * scale;
                            }
                        }
                    }
                }
                sycl::group_barrier(grp);
                // per-head chunk max and exp-sum: sub-group w handles heads w, w+WARPS, ...
                for (int h = warp; h < G; h += WARPS) {
                    constexpr int PER = CHUNK / WARP;   // the cells a lane takes
                    float mx = -FLT_MAX;
    #pragma unroll
                    for (int k = 0; k < PER; ++k) {   // the cells past n_here hold no score of this chunk
                        const int c = lane + WARP * k;
                        if (c < n_here) mx = sycl::fmax(mx, sp[h * CHUNK + c]);
                    }
                    const float m = warp_max(sg, mx);
                    float* pt = reinterpret_cast<float*>(&spt[0]);
                    float sum = 0.0f;
    #pragma unroll
                    for (int k = 0; k < PER; ++k) {
                        const int c = lane + WARP * k;
                        const float e = (c < n_here && srow[c] >= 0) ? sycl::exp(sp[h * CHUNK + c] - m) : 0.0f;
                        pt[c * GP + h] = e;
                        sum += e;
                    }
                    const float l = warp_sum(sg, sum);
                    if (lane == 0) { sml[h] = m; sml[G + h] = l; }
                }
                sycl::group_barrier(grp);
                // values: work-item t owns dimension t for all G heads.
                float acc[GP];
                for (int h = 0; h < GP; ++h) acc[h] = 0.0f;
                // one cell's weights (GP/4 float4 loads from local memory) into the accumulators; the cells are folded
                // in ascending order, so the sums are the plain loop's, bit for bit (upstream fd95405).  The padding
                // slots past G accumulate whatever local memory holds and are never stored.
                const auto fold = [&](int c, float v) {
    #pragma unroll
                    for (int g4 = 0; g4 < GP / 4; ++g4) {
                        const sycl::float4 w = spt[(size_t) c * (GP / 4) + g4];
                        acc[4 * g4] = sycl::fma(w.x(), v, acc[4 * g4]);
                        acc[4 * g4 + 1] = sycl::fma(w.y(), v, acc[4 * g4 + 1]);
                        acc[4 * g4 + 2] = sycl::fma(w.z(), v, acc[4 * g4 + 2]);
                        acc[4 * g4 + 3] = sycl::fma(w.w(), v, acc[4 * g4 + 3]);
                    }
                };
                int c = 0;
                // four cells at a time with their V loads issued together; a group holding a masked cell (page -1,
                // rare) leaves this loop and the one below finishes the chunk
                for (; c + 4 <= n_here; c += 4) {
                    const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
                    if ((r0 | r1 | r2 | r3) < 0) break;
                    const float v0 = load_v1<HD, KV_MODE>(p, r0, t), v1 = load_v1<HD, KV_MODE>(p, r1, t);
                    const float v2 = load_v1<HD, KV_MODE>(p, r2, t), v3 = load_v1<HD, KV_MODE>(p, r3, t);
                    fold(c, v0);
                    fold(c + 1, v1);
                    fold(c + 2, v2);
                    fold(c + 3, v3);
                }
                for (; c < n_here; ++c) {
                    if (srow[c] < 0) continue;   // masked above, weight 0
                    fold(c, load_v1<HD, KV_MODE>(p, srow[c], t));
                }
                if constexpr (!LOOP) {   // the chunk's own partials
                    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
                    if (t < G) { part_m[slot * G + t] = sml[t]; part_l[slot * G + t] = sml[G + t]; }
                    break;
                }
                for (int h = 0; h < G; ++h) {
                    const float m = sml[h], m2 = sycl::fmax(run_m[h], m);
                    const float a = sycl::exp(run_m[h] - m2), b = sycl::exp(m - m2);
                    run_acc[h] = run_acc[h] * a + acc[h] * b;
                    run_l[h] = run_l[h] * a + sml[G + h] * b;
                    run_m[h] = m2;
                }
                sycl::group_barrier(grp);   // the next chunk writes srow, sp, spt and sml
            }
            if constexpr (LOOP) {
                for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = run_acc[h];
                if (t < G) { part_m[slot * G + t] = run_m[t]; part_l[slot * G + t] = run_l[t]; }
            }
        });
    });
}

// steps_all (with `dense` and `window` as launch_chunk's): only the slots the query's cells reached are read (the others
// hold no cells); null: every slot
template <int HD, int G, bool PREFETCH>
sycl::event launch_merge(sycl::queue& queue, const float* part_acc_all, const float* part_m_all, const float* part_l_all,
                         int n_slots, float* attn_all, size_t n_head, size_t n_q, long long scratch_stride,
                         const int32_t* steps_all = nullptr, bool dense = true, int window = 0, int min_chunk = CHUNK) {
    return queue.parallel_for(sycl::nd_range<2>({n_q, n_head * HD}, {1, HD}), [=](sycl::nd_item<2> it) {
        const size_t y = it.get_group(0);
        int n_chunks = n_slots;
        if (steps_all != nullptr) {
            const int32_t* step = steps_all + y * kStepCount;
            const int n_all = dense ? step[kStepNKv] : step[kStepWidth];
            const int n_ids = n_all - (dense && window > 0 ? sycl::max(0, n_all - window) : 0);
            const int csz = chunk_of(dense, n_ids, n_slots, min_chunk);
            n_chunks = sycl::min(n_slots, (n_ids + csz - 1) / csz);
        }
        const float* part_acc = part_acc_all + y * (size_t) scratch_stride;
        const float* part_m = part_m_all + y * (size_t) scratch_stride;
        const float* part_l = part_l_all + y * (size_t) scratch_stride;
        float* attn = attn_all + y * n_head * HD;
        const int h = (int) it.get_group(1);                 // global query head
        const int kvh = h / G, hl = h % G;
        const int d = (int) it.get_local_id(1);
        float M = -FLT_MAX;
        int c = 0;
        if constexpr (PREFETCH) {
            for (; c + 8 <= n_chunks; c += 8) {
                float maxima[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) maxima[j] = part_m[(kvh * n_slots + c + j) * G + hl];
#pragma unroll
                for (int j = 0; j < 8; ++j) M = sycl::fmax(M, maxima[j]);
            }
        }
        for (; c < n_chunks; ++c) M = sycl::fmax(M, part_m[(kvh * n_slots + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        c = 0;
        if constexpr (PREFETCH) {
            for (; c + 8 <= n_chunks; c += 8) {
                float maxima[8], sums[8], values[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int slot = kvh * n_slots + c + j;
                    maxima[j] = part_m[slot * G + hl];
                    if (maxima[j] != -FLT_MAX) {
                        sums[j] = part_l[slot * G + hl];
                        values[j] = part_acc[((size_t) slot * G + hl) * HD + d];
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (maxima[j] == -FLT_MAX) continue;
                    const float w = sycl::exp(maxima[j] - M);
                    L = sycl::fma(sums[j], w, L);
                    acc = sycl::fma(values[j], w, acc);
                }
            }
        }
        for (; c < n_chunks; ++c) {
            const int slot = kvh * n_slots + c;
            const float m = part_m[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::exp(m - M);
            L = sycl::fma(part_l[slot * G + hl], w, L);
            acc = sycl::fma(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

// The work-groups a KV head's cells are spread over (a slot each; chunk c to slot c % slots, in turn): every chunk its
// own unless the chunks are at least twice what the GPU's compute units hold (8 groups a unit, all queries' and KV
// heads' together), then that many slots, each reading its chunks in turn, and the merge reads fewer partials.  The
// RTX 4070 (46 units) at a 32K context (2026-10-27): 516 chunks a KV head to 92 slots, Qwen3.5-9B's decode 64.6 ->
// 67.6 tok/s; the B70 (256 units) keeps one a chunk (with 512 slots for 516 chunks its decode was 0.8% slower).
int slots_for(sycl::queue& q, int64_t cap, int nkv, size_t n_q) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    const int64_t cu = (int64_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
    const int64_t want = std::max<int64_t>(1, cu * 8 / std::max<int64_t>(1, (int64_t) nkv * (int64_t) n_q));
    return (int) (chunks >= 2 * want ? want : chunks);
}

int kv_mode_of(const QsaAttnPools& pools) {
    return pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
}

template <int HD, int G, bool LOOP, bool ROWS>
void launch_kv_form(int kv_mode, int nc, sycl::queue& queue, const float* q, const QsaAttnPools& pools, const int32_t* ids,
               const int32_t* steps, int nkv, int ps, float scale, float* part_acc, float* part_m, float* part_l,
               int n_slots, size_t n_q, int cap, long long stride, int window, int kv_div,
               int min_chunk, const QsaAttnPoolsRows& rows) {
    if (kv_mode == 3) launch_chunk<HD, G, 3, LOOP, 2, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else if (kv_mode == 2) launch_chunk<HD, G, 2, LOOP, 2, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else if (kv_mode == 1) launch_chunk<HD, G, 1, LOOP, 2, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else if (nc == 8) launch_chunk<HD, G, 0, LOOP, 8, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else if (nc == 4) launch_chunk<HD, G, 0, LOOP, 4, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else launch_chunk<HD, G, 0, LOOP, 2, ROWS>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
}

template <int HD, int G, bool LOOP>
void launch_kv(int kv_mode, int nc, sycl::queue& queue, const float* q, const QsaAttnPools& pools, const int32_t* ids,
               const int32_t* steps, int nkv, int ps, float scale, float* part_acc, float* part_m, float* part_l,
               int n_slots, size_t n_q, int cap, long long stride, int window, int kv_div, int min_chunk,
               const QsaAttnPoolsRows& rows) {
    if (rows.n > 0)
        launch_kv_form<HD, G, LOOP, true>(kv_mode, nc, queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m,
                                          part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
    else
        launch_kv_form<HD, G, LOOP, false>(kv_mode, nc, queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m,
                                           part_l, n_slots, n_q, cap, stride, window, kv_div, min_chunk, rows);
}

// The decode attention's form on a GPU, for a shape (head_dim, query heads per KV head): the cells a step of the fp16
// score loop takes (NC) and the ways a KV head's query heads are split over work-groups (each reading the KV head's
// cells).  2 and 1 unless qsa_dense_attn_tune timed this GPU and shape.  NC 2, 4 and 8 are not the same on every GPU
// (Qwen3.5-9B's 32K decode, 2026-10-27: the RX 9060 XT 38.9 / 40.2 / 40.5 tok/s, the RTX 4070 67.7 / 67.4 / 64.0, the
// B70 43.1 / 42.9 / 40.3).  The split puts more work-groups on a short context, where one group a KV head leaves the
// GPU idle: PLaMo 3 Translate 0.8B (16 heads on 1 KV head) decoded 343 -> 504 tok/s after 64 cells on the B70 split 4
// ways, 567 -> 593 on the RTX 4070; past a few ways the cells read again cost more than the groups gain (16 ways: 430
// against 451 tok/s at 12K cells on the B70), and the 2B (6 heads a KV head) took 2 ways (2026-11-09).
struct Form {
    int nc = 2, split = 1, min_chunk = 16;   // min_chunk: chunk_of's
};
template <int HD, int G>
core::PerDevice<Form>& form_store() {
    static core::PerDevice<Form> store;
    return store;
}
template <int HD, int G>
Form form_of(sycl::queue& q) {
    Form f = form_store<HD, G>().get(q.get_device(), [] { return Form{}; });
    if (const char* e = strata::env_get("QSA_NC")) {   // 2, 4 or 8 over the timed choice (the parity test, --bench)
        const long v = std::strtol(e, nullptr, 10);
        if (v == 2 || v == 4 || v == 8) f.nc = (int) v;
    }
    if (const char* e = strata::env_get("QSA_SPLIT")) {   // the ways over the timed choice (A/B), one that divides G
        const long v = std::strtol(e, nullptr, 10);
        if (v >= 1 && G % v == 0) f.split = (int) v;
    }
    if (const char* e = strata::env_get("QSA_MIN_CHUNK")) {   // the least cells a chunk (64: whole chunks), A/B
        const long v = std::strtol(e, nullptr, 10);
        if (v >= 4 && v <= CHUNK) f.min_chunk = (int) v;
    }
    return f;
}

// Whether the merge reads eight slots' partials into registers before it folds them (the same fold order, the same
// bits): timed once for a device and a head count, on partials of its own and on a queue of its own, since the
// callers' queues are recording a graph when the decode windows first come here.  Below 8 slots the two forms are one.
// One timing serves every slot count: the forms' ratio held from 33 to 640 slots on the B70 (4.0 / 2.7 us to 38.2 /
// 19.2) and from 23 to 184 on the RTX 4070 (2.3 / 2.2 to 6.2 / 5.3), and a timing a slot count put 1 to 3% of a
// 256-token decode into the windows' capture.
// The B70 at a 32K context (2026-11-14): Ornith-1.5's window 22.53 -> 21.77 ms, Qwen3.8-Flash-Next's drafts 5.06 ->
// 4.63 ms a round; the RTX 4070 the same within its noise.
template <int HD, int G>
bool merge_prefetch(sycl::queue& queue, int n_slots, size_t n_head) {
    if (n_slots < 8) return false;
    static core::TimedCache<sycl::device, int, bool> choices;
    const int key = (int) n_head;
    n_slots = std::max(n_slots, 64);
    if (const auto cached = choices.get(queue.get_device(), key)) return *cached;
    if (core::timing_depth != 0) return false;
    sycl::queue probe(queue.get_context(), queue.get_device());
    const size_t n_ml = (size_t) n_slots * n_head, n_acc = n_ml * HD;
    float* acc = sycl::malloc_device<float>(n_acc, probe);
    float* m = sycl::malloc_device<float>(n_ml, probe);
    float* l = sycl::malloc_device<float>(n_ml, probe);
    float* out = sycl::malloc_device<float>(n_head * HD, probe);
    bool pick = false;
    if (acc && m && l && out) {
        probe.fill(acc, 0.01f, n_acc);
        probe.fill(m, 0.5f, n_ml);
        probe.fill(l, 1.0f, n_ml).wait_and_throw();
        const auto run = [&](bool prefetch) {
            for (int i = 0; i < 8; ++i) {
                if (prefetch) launch_merge<HD, G, true>(probe, acc, m, l, n_slots, out, n_head, 1, 0);
                else launch_merge<HD, G, false>(probe, acc, m, l, n_slots, out, n_head, 1, 0);
            }
            return true;
        };
        const auto measured = core::timed_choice({{[&] { return run(false); }}, {[&] { return run(true); }}},
                                                 [&] { probe.wait_and_throw(); }, 5, 1.05, 0.005);
        pick = measured.pick == 1;
        std::fprintf(stderr, "xestrata: attention merge %s (%d slots, %zu heads of %d; %.1f / %.1f us)\n",
                     pick ? "prefetch" : "plain", n_slots, n_head, HD, measured.seconds[0] * 1e6 / 8,
                     measured.seconds[1] * 1e6 / 8);
    }
    for (float* x : {acc, m, l, out})
        if (x) sycl::free(x, probe);
    return choices.put(queue.get_device(), key, pick);
}

// G: the query heads a work-group takes, the KV head's own split `split` ways
template <int HD, int G>
void run_shape(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream,
               int window, int nc, int split, int min_chunk = CHUNK, const QsaAttnPoolsRows& row_pools = {}) {
    const int kv_mode = kv_mode_of(pools);
    auto& queue = queue_for(stream);
    const int nkv = (int) s.n_head_kv * split, ps = (int) s.page_size;
    // a batch window's rows keep one row's slot count, so each row's sums are its own 1-row call's bits
    const int n_slots = slots_for(queue, cap, nkv, row_pools.n > 0 ? 1 : n_q);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_slots * s.n_head * HD;
    float* part_l = part_m + (size_t) n_slots * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    const QsaAttnPoolsRows pools_rows = row_pools;
    if (n_slots < (cap + CHUNK - 1) / CHUNK)
        launch_kv<HD, G, true>(kv_mode, nc, queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l,
                               n_slots, n_q, (int) cap, stride, window, split, min_chunk, pools_rows);
    else
        launch_kv<HD, G, false>(kv_mode, nc, queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l,
                                n_slots, n_q, (int) cap, stride, window, split, min_chunk, pools_rows);
    // The merge's form: with XESTRATA_ATTN_MERGE_PREFETCH unset or auto, the one merge_prefetch timed for this device
    // and slot count.
    auto merge = [&](bool prefetch, size_t rows) {
        return prefetch
            ? launch_merge<HD, G, true>(queue, part_acc, part_m, part_l, n_slots, attn, (size_t) s.n_head, rows,
                                        stride, steps, ids == nullptr, window, min_chunk)
            : launch_merge<HD, G, false>(queue, part_acc, part_m, part_l, n_slots, attn, (size_t) s.n_head, rows,
                                         stride, steps, ids == nullptr, window, min_chunk);
    };
    static const int merge_mode = strata::env_auto("ATTN_MERGE_PREFETCH");
    const bool prefetch = merge_mode >= 0 ? merge_mode == 1 : merge_prefetch<HD, G>(queue, n_slots, (size_t) s.n_head);
    const auto e = merge(prefetch, n_q);
    if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
}

// `f` with G / split heads a work-group: the instantiations the splits take (a split to another count is not run)
template <int HD>
bool with_split(int heads_per_group, auto&& f) {
    switch (heads_per_group) {
    case 16: if constexpr (HD == 128) { f.template operator()<16>(); return true; } return false;
    case 12: if constexpr (HD == 256) { f.template operator()<12>(); return true; } return false;
    case 10: if constexpr (HD == 128) { f.template operator()<10>(); return true; } return false;
    case 8: f.template operator()<8>(); return true;
    case 6: if constexpr (HD == 128) { f.template operator()<6>(); return true; } return false;
    case 4: f.template operator()<4>(); return true;
    case 3: if constexpr (HD == 128) { f.template operator()<3>(); return true; } return false;
    case 2: f.template operator()<2>(); return true;
    case 1: f.template operator()<1>(); return true;
    default: return false;
    }
}

template <int HD, int G>
void run_form(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream, int window,
              const QsaAttnPoolsRows& rows = {}) {
    const Form f = form_of<HD, G>(queue_for(stream));
    const bool ok = with_split<HD>(G / f.split, [&]<int GS>() {
        run_shape<HD, GS>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream, window, f.nc, f.split,
                          f.min_chunk, rows);
    });
    if (!ok) run_shape<HD, G>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream, window, f.nc, 1, CHUNK, rows);
}

// The shapes this file is built for: (head_dim, query heads per KV head).  A model with another shape needs its line
// here; the kernels' local arrays are sized by G and HD.
bool shape_ok(const QsaShapes& s) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    return (s.head_dim == 256 && (g == 12 || g == 8 || g == 4)) || (s.head_dim == 128 && (g == 10 || g == 6 || g == 16 || g == 8));
}

// `f` on the shape's (head_dim, query heads per KV head)
bool with_shape(const QsaShapes& s, auto&& f) {
    const int64_t g = s.n_head / s.n_head_kv;
    if (s.head_dim == 256 && g == 12) f.template operator()<256, 12>();
    else if (s.head_dim == 256 && g == 8) f.template operator()<256, 8>();
    else if (s.head_dim == 256 && g == 4) f.template operator()<256, 4>();
    else if (g == 6) f.template operator()<128, 6>();
    else if (g == 16) f.template operator()<128, 16>();
    else if (g == 8) f.template operator()<128, 8>();
    else if (g == 10) f.template operator()<128, 10>();
    else return false;
    return true;
}

void run(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
         const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream, int window = 0,
         const QsaAttnPoolsRows& rows = {}) {
    with_shape(s, [&]<int HD, int G>() {
        run_form<HD, G>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream, window, rows);
    });
}

// The timing's dummy cache of `cells` cells (as the engine runs it: 16 launches with their merge in a graph, replayed
// 5 times, the first a warm-up, the fastest kept): microseconds a launch with GS heads a group, the KV heads split
// `split` ways, NC `nc`; -1 when the buffers cannot be had
template <int HD, int GS>
double time_form(void* stream, const QsaShapes& s, int64_t cells, int nc, int split) {
    sycl::queue& q = queue_for(stream);
    const int64_t ps = s.page_size, pages = (cells + ps - 1) / ps;
    cells = pages * ps;
    const size_t pool = (size_t) (pages * s.n_head_kv * ps * HD), heads = (size_t) s.n_head * HD;
    const uint64_t n_scratch = qsa_decode_attn_scratch_floats(cells, s);
    uint16_t* k = sycl::malloc_device<uint16_t>(pool, q);
    uint16_t* v = sycl::malloc_device<uint16_t>(pool, q);
    int32_t* table = sycl::malloc_device<int32_t>((size_t) pages, q);
    int32_t* step = sycl::malloc_device<int32_t>(kStepCount, q);
    float* qv = sycl::malloc_device<float>(heads, q);
    float* scratch = sycl::malloc_device<float>(n_scratch, q);
    float* attn = sycl::malloc_device<float>(heads, q);
    const auto release = [&] {
        for (void* x : {(void*) k, (void*) v, (void*) table, (void*) step, (void*) qv, (void*) scratch, (void*) attn})
            sycl::free(x, q);
    };
    if (!k || !v || !table || !step || !qv || !scratch || !attn) {
        release();
        return -1.0;
    }
    std::vector<int32_t> pt((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) pt[(size_t) i] = (int32_t) i;
    int32_t hs[kStepCount];
    qsa_step_fill(hs, cells - 1, s);
    q.fill(k, (uint16_t) 0x2c00, pool);   // 0.0625: moderate scores
    q.fill(v, (uint16_t) 0x2c00, pool);
    q.fill(qv, 0.01f, heads);
    q.memcpy(table, pt.data(), pt.size() * sizeof(int32_t));
    q.memcpy(step, hs, sizeof(hs)).wait();
    QsaAttnPools pools;
    pools.k_pool = k;
    pools.v_pool = v;
    pools.page_table = table;
    namespace sx = sycl::ext::oneapi::experimental;
    sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
    graph.begin_recording(q);
    for (int i = 0; i < 16; ++i)
        run_shape<HD, GS>(qv, pools, nullptr, step, cells, s, scratch, attn, 1, 0, stream, 0, nc, split);
    graph.end_recording(q);
    auto exec = graph.finalize();
    double best = 0.0;
    for (int round = 0; round < 5; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        q.ext_oneapi_graph(exec);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (round > 0 && (best == 0.0 || us < best)) best = us;
    }
    release();
    return best / 16;
}

// NC timed at `cells` (its old point) without a split.  The split is the least that puts 8 work-groups on a short
// context's cells (a KV head's heads split `split` ways, n_kv * split groups), keeping 3 heads or more a group (each
// group reads the KV head's cells again), else the most that keeps them; among those with an instantiation.  Timed
// alone (16 launches in a graph) the splits came out slower than one group a KV head for PLaMo 3 Translate 2B (6 heads
// a KV head, 4 KV heads), where its decode ran faster split 2 ways (B70 196 -> 203 tok/s after 64 cells, 178 -> 186
// after 12K; RTX 4070 213 -> 216, 194 -> 197); the 0.8B (16 heads, 1 KV head) 4 ways (B70 456 -> 504 after 64 cells,
// 404 -> 451 after 12K); Qwen3.5-0.8B and 2B (4 heads of 256 a KV head) split 4 ways decoded 5% slower after 8K cells
// on the B70 and the RTX 4070 (one head a group), and keep one group a KV head (2026-11-09).
int split_for(int g, int64_t n_kv, int hd) {
    int pick = 1;
    for (int split = 2; split <= g && n_kv * pick < 8; ++split) {
        const int gs = g / split;
        const bool inst = gs == 8 || gs == 4 || (hd == 128 && (gs == 16 || gs == 10 || gs == 6 || gs == 3)) ||
                          (hd == 256 && gs == 12);
        if (g % split == 0 && gs >= 3 && inst) pick = split;
    }
    return pick;
}

template <int HD, int G>
Form time_forms(void* stream, const QsaShapes& s, int64_t cells) {
    Form f;
    const int ncs[3] = {2, 4, 8};
    double t[3];
    int pick = 0;
    for (int i = 0; i < 3; ++i) {
        t[i] = time_form<HD, G>(stream, s, cells, ncs[i], 1);
        if (t[i] >= 0.0 && (t[pick] < 0.0 || t[i] < t[pick])) pick = i;
    }
    f.nc = ncs[pick];
    f.split = split_for(G, s.n_head_kv, HD);
    std::fprintf(stderr, "xestrata: the decode attention (%lld/%lld heads of %d) takes %d cells a step (timed at %lld "
                         "cells: 2: %.0f us, 4: %.0f, 8: %.0f) and splits a KV head's heads %d ways\n",
                 (long long) s.n_head, (long long) s.n_head_kv, HD, f.nc, (long long) cells, t[0], t[1], t[2], f.split);
    return f;
}


bool pools_ok(const QsaAttnPools& pools) {
    const int kv_mode = kv_mode_of(pools);
    return kv_mode == 3 ? pools.k_scale != nullptr
         : kv_mode == 2 ? pools.v_q4 != nullptr
         : kv_mode == 1 ? (pools.v_q && pools.k_scale && pools.v_scale) : (pools.k_pool && pools.v_pool);
}

std::string shape_name(const QsaShapes& s) {
    return std::to_string(s.n_head) + "/" + std::to_string(s.n_head_kv) + " heads of " + std::to_string(s.head_dim);
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_decode_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    run(q, pools, ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s), stream);
}

void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !steps || rows.n != n_q || n_q > kQsaRowsMax ||
        rows.p[0].page_table == nullptr)
        throw core::DeviceError("qsa_decode_attn_batch_rows: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    const int mode = kv_mode_of(rows.p[0]);
    for (int z = 1; z < rows.n; ++z)
        if (kv_mode_of(rows.p[z]) != mode || rows.p[z].page_table == nullptr)
            throw core::DeviceError("qsa_decode_attn_batch_rows: the rows' pools differ in format");
    run(q, rows.p[0], ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s),
        stream, 0, rows);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (uint64_t) (s.head_dim + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !step || !pools.page_table)
        throw core::DeviceError("qsa_decode_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_decode_attn: incomplete KV pools");
    run(q, pools, ids, step, cap, s, scratch, attn, 1, 0, stream);
}

void qsa_dense_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* step, int64_t max_cells,
                         const QsaShapes& s, float* scratch, float* attn, void* stream, int64_t window) {
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !step || !pools.page_table)
        throw core::DeviceError("qsa_dense_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn: incomplete KV pools");
    run(q, pools, nullptr, step, max_cells, s, scratch, attn, 1, 0, stream, (int) window);
}

void qsa_dense_attn_tune(const QsaShapes& s, int64_t max_cells, void* stream) {
    if (!shape_ok(s) || max_cells < CHUNK || s.page_size <= 0) return;
    auto& q = queue_for(stream);
    const int64_t cells = std::min<int64_t>(max_cells, 16384);
    with_shape(s, [&]<int HD, int G>() {
        form_store<HD, G>().get(q.get_device(), [&] { return time_forms<HD, G>(stream, s, cells); });
    });
}

void qsa_dense_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* steps, int64_t max_cells,
                          const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream, int64_t window) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_dense_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn_batch: incomplete KV pools");
    run(q, pools, nullptr, steps, max_cells, s, scratch, attn, (size_t) n_q,
        (long long) qsa_decode_attn_scratch_floats(max_cells, s), stream, (int) window);
}

}  // namespace strata::kernels
