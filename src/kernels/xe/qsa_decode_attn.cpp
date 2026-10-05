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

// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
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
inline int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own work-item), per KV format.
template <int KV_MODE>
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
        sycl::local_accessor<float, 1> sp(sycl::range<1>((size_t) G * CHUNK), hd);       // scores
        sycl::local_accessor<sycl::float4, 1> spt(sycl::range<1>(CHUNK * G / 4), hd);   // probabilities, cell-major
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
            // scores: each sub-group takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells
            // per step (warp + 8 * (i + j)): a lane's q slice comes out of local memory once for all of them (upstream
            // fd95405); every dot product is the same expression as for a single cell, so the scores are unchanged.
            constexpr int NC = 2;
            const int hd_lane = head_of_lane(lane);
            for (int i = 0; i < CHUNK / WARPS; i += NC) {
                int cc[NC];
                bool all_ok = true;
                for (int j = 0; j < NC; ++j) {
                    cc[j] = warp + WARPS * (i + j);
                    all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
                }
                if (all_ok) {
                    float kk[NC][8];
                    for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
                    float pp[NC][16];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        const float* qa = &sq[h * HD + lane * 8];
#pragma unroll
                        for (int j = 0; j < NC; ++j)
                            pp[j][h] = kk[j][0] * qa[0] + kk[j][1] * qa[1] + kk[j][2] * qa[2] + kk[j][3] * qa[3] +
                                       kk[j][4] * qa[4] + kk[j][5] * qa[5] + kk[j][6] * qa[6] + kk[j][7] * qa[7];
                    }
#pragma unroll
                    for (int j = 0; j < NC; ++j) {
                        for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                        const float sj = reduce12(sg, pp[j], lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + cc[j]] = sj * scale;
                    }
                } else {
                    for (int j = 0; j < NC; ++j) {   // a masked or out-of-range cell in the step: one cell at a time
                        const int c = cc[j];
                        if (c >= n_here || srow[c] < 0) {
                            if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                            continue;
                        }
                        float k8[8];
                        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                        float part[16];
                        for (int h = 0; h < G; ++h) {
                            const float* qa = &sq[h * HD + lane * 8];
                            part[h] = k8[0] * qa[0] + k8[1] * qa[1] + k8[2] * qa[2] + k8[3] * qa[3] +
                                      k8[4] * qa[4] + k8[5] * qa[5] + k8[6] * qa[6] + k8[7] * qa[7];
                        }
                        for (int h = G; h < 16; ++h) part[h] = 0.0f;
                        const float sc = reduce12(sg, part, lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + c] = sc * scale;
                    }
                }
            }
            sycl::group_barrier(grp);
            // per-head chunk max and exp-sum: sub-group w handles heads w and w+8.
            for (int h = warp; h < G; h += WARPS) {
                const float a = sp[h * CHUNK + lane], b = sp[h * CHUNK + lane + 32];
                const float m = warp_max(sg, sycl::fmax(a, b));
                const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::exp(a - m) : 0.0f;
                const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::exp(b - m) : 0.0f;
                float* pt = reinterpret_cast<float*>(&spt[0]);
                pt[lane * G + h] = ea;
                pt[(lane + 32) * G + h] = eb;
                const float l = warp_sum(sg, ea + eb);
                if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
            }
            sycl::group_barrier(grp);
            // values: work-item t owns dimension t for all 12 heads.
            float acc[G];
            for (int h = 0; h < G; ++h) acc[h] = 0.0f;
            // one cell's 12 weights (three float4 loads from local memory) into the 12 accumulators; the cells are
            // folded in ascending order, so the sums are the plain loop's, bit for bit (upstream fd95405)
            const auto fold = [&](int c, float v) {
                const sycl::float4 w0 = spt[(size_t) c * 3], w1 = spt[(size_t) c * 3 + 1], w2 = spt[(size_t) c * 3 + 2];
                acc[0] = sycl::fma(w0.x(), v, acc[0]);  acc[1] = sycl::fma(w0.y(), v, acc[1]);
                acc[2] = sycl::fma(w0.z(), v, acc[2]);  acc[3] = sycl::fma(w0.w(), v, acc[3]);
                acc[4] = sycl::fma(w1.x(), v, acc[4]);  acc[5] = sycl::fma(w1.y(), v, acc[5]);
                acc[6] = sycl::fma(w1.z(), v, acc[6]);  acc[7] = sycl::fma(w1.w(), v, acc[7]);
                acc[8] = sycl::fma(w2.x(), v, acc[8]);  acc[9] = sycl::fma(w2.y(), v, acc[9]);
                acc[10] = sycl::fma(w2.z(), v, acc[10]); acc[11] = sycl::fma(w2.w(), v, acc[11]);
            };
            int c = 0;
            // four cells at a time with their V loads issued together; a group holding a masked cell (page -1,
            // rare) leaves this loop and the one below finishes the chunk
            for (; c + 4 <= n_here; c += 4) {
                const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
                if ((r0 | r1 | r2 | r3) < 0) break;
                const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
                const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
                fold(c, v0);
                fold(c + 1, v1);
                fold(c + 2, v2);
                fold(c + 3, v3);
            }
            for (; c < n_here; ++c) {
                if (srow[c] < 0) continue;   // masked above, weight 0
                fold(c, load_v1<KV_MODE>(p, srow[c], t));
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
