// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/qsa_decode_attn.cpp - the Xe port of Strata's src/kernels/cuda/qsa_decode_attn.cu (see the header): split-K
// decode attention over the selected cells for fp16, INT8 and Q4_0 KV, then a merge of the chunk partials.
//
// A warp is a sub-group of 32.  CUDA used the __expf intrinsic here; this uses the precise exp.  Explicit fmaf calls
// stay fused (sycl::fma), and the library's -ffp-contract=off leaves every other multiply-add as written.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/runtime.hpp"

#include <cfloat>
#include <cmath>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per work-group
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
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

// 8 consecutive values of one cell's key or value row, dimensions [d0, d0+8).  KV_MODE 0: FP16, 1: INT8, 2: rotated
// Q4_0, 3: the hybrid K8V4 (K as INT8, V as rotated Q4_0; upstream 2aa8f72).
template <int KV_MODE>
inline void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 3) {
        if (value) load8<2>(p, true, row, d0, out);
        else load8<1>(p, false, row, d0, out);
    } else if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        for (int j = 0; j < 8; ++j) out[j] = h2f(base[j]);
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const float sc = h2f((value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
        for (int j = 0; j < 8; ++j) out[j] = (float) codes[j] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = h2f(blk->d);
        const int j = (rem == 0 || rem == 16) ? 0 : 8;
        const uint8_t* bytes = blk->qs + j;
        if (rem < 16) {
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}

template <int KV_MODE>
sycl::event launch_chunk(sycl::queue& queue, const float* q_all, const QsaAttnPools p, const int32_t* ids_all,
                         const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                         float* part_m_all, float* part_l_all, int n_chunks, size_t n_q, int cap,
                         long long scratch_stride) {
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(G * HD), hd);          // this KV head's query heads
        sycl::local_accessor<float, 1> sp(sycl::range<1>(G * CHUNK), hd);       // scores, then probabilities
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);     // pool row of each cell
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_chunks * THREADS}, {1, 1, THREADS}),
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            // batched form: query z, with its own q row, selection, step and scratch
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_ids = step[kStepWidth];
            const int chunk = (int) it.get_group(2), kvh = (int) it.get_group(1);
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int c0 = chunk * CHUNK;
            const int n_here = sycl::min(CHUNK, n_ids - c0);
            const int slot = kvh * n_chunks + chunk;
            if (n_here <= 0) {                                    // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            for (int i = t; i < G * HD; i += THREADS) sq[i] = q[(size_t) (kvh * G) * HD + i];
            if (t < CHUNK) {
                long long r = -1;
                if (t < n_here) {
                    const int cell = ids[c0 + t];
                    const long long page = (long long) p.page_table[cell / page_size];
                    // a block the KV streaming could not make resident keeps page -1; its cells are masked (score
                    // -FLT_MAX, weight 0) instead of being read from before the pool (upstream f3925cf)
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                }
                srow[t] = r;
            }
            sycl::group_barrier(grp);
            // scores: each sub-group takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
            for (int c = warp; c < CHUNK; c += WARPS) {
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                for (int h = 0; h < G; ++h) {
                    const float* qa = &sq[h * HD + lane * 8];
                    float s = k8[0] * qa[0] + k8[1] * qa[1] + k8[2] * qa[2] + k8[3] * qa[3] +
                              k8[4] * qa[4] + k8[5] * qa[5] + k8[6] * qa[6] + k8[7] * qa[7];
                    s = warp_sum(sg, s);
                    if (lane == 0) sp[h * CHUNK + c] = s * scale;
                }
            }
            sycl::group_barrier(grp);
            // per-head chunk max and exp-sum: sub-group w handles heads w and w+8.
            for (int h = warp; h < G; h += WARPS) {
                const float a = sp[h * CHUNK + lane], b = sp[h * CHUNK + lane + 32];
                const float m = warp_max(sg, sycl::fmax(a, b));
                const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::exp(a - m) : 0.0f;
                const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::exp(b - m) : 0.0f;
                sp[h * CHUNK + lane] = ea;
                sp[h * CHUNK + lane + 32] = eb;
                const float l = warp_sum(sg, ea + eb);
                if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
            }
            sycl::group_barrier(grp);
            // values: work-item t owns dimension t for all 12 heads.
            float acc[G];
            for (int h = 0; h < G; ++h) acc[h] = 0.0f;
            for (int c = 0; c < n_here; ++c) {
                if (srow[c] < 0) continue;   // masked above, weight 0
                float v;
                if constexpr (KV_MODE == 0) {
                    v = h2f(p.v_pool[srow[c] * HD + t]);
                } else if constexpr (KV_MODE == 1) {
                    const float sc = h2f(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]);
                    v = (float) p.v_q[srow[c] * HD + t] * sc;
                } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
                    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
                    const int b = t / QK4_0;
                    const int rem = t % QK4_0;
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
                    const float d = h2f(blk->d);
                    const int j = rem < 16 ? rem : (rem - 16);
                    const uint8_t byte = blk->qs[j];
                    const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
                    v = (float) nibble * d;
                }
                for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CHUNK + c], v, acc[h]);
            }
            for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        });
    });
}

sycl::event launch_merge(sycl::queue& queue, const float* part_acc_all, const float* part_m_all, const float* part_l_all,
                         int n_chunks, float* attn_all, size_t n_head, size_t n_q, long long scratch_stride) {
    return queue.parallel_for(sycl::nd_range<2>({n_q, n_head * HD}, {1, HD}), [=](sycl::nd_item<2> it) {
        const size_t y = it.get_group(0);
        const float* part_acc = part_acc_all + y * (size_t) scratch_stride;
        const float* part_m = part_m_all + y * (size_t) scratch_stride;
        const float* part_l = part_l_all + y * (size_t) scratch_stride;
        float* attn = attn_all + y * n_head * HD;
        const int h = (int) it.get_group(1);                 // global query head
        const int kvh = h / G, hl = h % G;
        const int d = (int) it.get_local_id(1);
        float M = -FLT_MAX;
        for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = kvh * n_chunks + c;
            const float m = part_m[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::exp(m - M);
            L = sycl::fma(part_l[slot * G + hl], w, L);
            acc = sycl::fma(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

void run(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
         const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream) {
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    auto& queue = queue_for(stream);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size;
    if (kv_mode == 3) launch_chunk<3>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 2) launch_chunk<2>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 1) launch_chunk<1>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else launch_chunk<0>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    const auto e = launch_merge(queue, part_acc, part_m, part_l, n_chunks, attn, (size_t) s.n_head, n_q, stride);
    if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_decode_attn_batch: unsupported geometry or missing buffers");
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    run(q, pools, ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s), stream);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table)
        throw core::DeviceError("qsa_decode_attn: unsupported geometry or missing buffers");
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? !pools.k_scale
        : kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool)))
        throw core::DeviceError("qsa_decode_attn: incomplete KV pools");
    run(q, pools, ids, step, cap, s, scratch, attn, 1, 0, stream);
}

}  // namespace strata::kernels
