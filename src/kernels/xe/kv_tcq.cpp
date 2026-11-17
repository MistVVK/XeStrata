// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/xe/kv_tcq.cpp - the trellis-coded K/V cache's encoder and signed rotation (kv_tcq.hpp), after YASAKANI's
// kvq_fast.cuh, kvq_vit2.cuh and kvx_codec.cuh, rewritten for SYCL.
//
// The encoder takes one work-group a row (a token's K or V of one KV head).  Its jobs are the row's four chains at
// each of the scale's multipliers; a sub-group of 32 takes a job at a time and searches the chain's 64 steps on its own
// (no work-group barrier inside the search): lane l keeps the costs of the states 32 i + l (i = 0..7).  A state's
// predecessors are the 2^B states that share its high 8 - B bits shifted down, so a lane holds predecessors of one
// group only (l mod 2^(8 - B)): the group's least cost is the lane's least, then the least over the lanes of the
// group (xor shuffles), and each state takes its group's from the lane of that number.  The keys carry the
// predecessor in their low 8 bits, so ties go to the lowest predecessor, and the last step's to the lowest state:
// every row codes the same way whatever else is in the launch.
//
// Deferred coding (KvAppend::tc_ring, kv_tcq.hpp): the encoder takes the cells of the blocks of kTcqBlock an append
// completes, each from the append's FP32 row rounded through INT8 as the ring keeps it, or from the ring for the
// block's cells an earlier append wrote; then the coded bound moves to the last complete block, and ring_write keeps
// the append's last kTcqRing cells, after the encoder has read the ring (the queue is in order).
#include "strata/kernels/kv_tcq.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "kv_tcq_parts.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_kv_tcq
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <iterator>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
constexpr int HD = 256;
constexpr int MAX_MULT = 16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

inline int64_t kmin(int64_t a, int64_t b) { return b < a ? b : a; }

struct Mults {
    float m[MAX_MULT] = {};
    int n = 0;
};

Mults mults_of(int bits) {
    Mults r;
    const auto put = [&](const auto& a) {
        r.n = (int) std::min<size_t>(std::size(a), MAX_MULT);
        for (int i = 0; i < r.n; ++i) r.m[i] = a[(size_t) i];
    };
    if (bits == 4) put(tcq::kMult4);
    else put(tcq::kMult5);
    return r;
}

}  // namespace

void fwht256_signed(float* data, int64_t n_rows, bool inverse, void* stream) {
    if (n_rows <= 0) return;
    constexpr int rows_per_group = 4;
    const size_t groups = (size_t) ((n_rows + rows_per_group - 1) / rows_per_group);
    // one dimension: a prompt's rows (tokens x heads) pass the 65,535 work-groups a GPU's first dimension takes
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(groups * rows_per_group * WARP, (size_t) rows_per_group * WARP),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            constexpr int EL = HD / WARP;   // 8
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t r = (int64_t) it.get_group(0) * rows_per_group + (int64_t) sg.get_group_linear_id();
            if (r >= n_rows) return;   // uniform across the row's sub-group
            float* row = data + r * HD;
            const int lane = (int) sg.get_local_linear_id();
            float reg[EL];
            for (int i = 0; i < EL; ++i) {
                const int d = i * WARP + lane;
                reg[i] = row[d] * (inverse ? 1.0f / 16.0f : tcq::sign_of(d) * (1.0f / 16.0f));
            }
            for (int h = 1; h < WARP; h *= 2)
                for (int j = 0; j < EL; ++j) {
                    const float v = reg[j], v2 = sycl::permute_group_by_xor(sg, v, h);
                    reg[j] = (lane & h) == 0 ? v + v2 : v2 - v;
                }
            for (int h = WARP; h < HD; h *= 2) {
                const int step = h / WARP;
                for (int j = 0; j < EL; j += 2 * step)
                    for (int k = 0; k < step; ++k) {
                        const float x = reg[j + k], y = reg[j + k + step];
                        reg[j + k] = x + y;
                        reg[j + k + step] = x - y;
                    }
            }
            for (int i = 0; i < EL; ++i) {
                const int d = i * WARP + lane;
                row[d] = inverse ? reg[i] * tcq::sign_of(d) : reg[i];
            }
        });
    if (!stream) core::Runtime::get().wait(e, "fwht256_signed");
}

namespace {

// A cell's position: step-based (graph-capturable) or from pos0
inline long long pos_of(const KvAppend& p, int64_t j) {
    return p.step ? (long long) p.step[j * p.step_stride + kStepPos] : (long long) (p.pos0 + j);
}

// The ring's write (deferred coding): the last kTcqRing cells of the append as INT8 (kv_q8.cpp's rounding) at their
// position's slot, and the coded bound lowered to the block of a cell this append did not code (a verify window
// written again after a rejection).  One work-group of 256 a (cell, plane, KV head), two sub-groups a group of 64.
sycl::event ring_write(sycl::queue& queue, const KvAppend& p, int kv_heads, int planes, int64_t cur_stride) {
    const int64_t n = std::min<int64_t>(p.n_tok, kTcqRing), j0 = p.n_tok - n;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> wmax(sycl::range<1>(HD / WARP), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) n * planes * kv_heads * HD, HD),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t wg = (int64_t) it.get_group(0);
            const int h = (int) (wg % kv_heads);
            const int64_t jp = wg / kv_heads, j = j0 + jp / planes;
            const int plane = (int) (jp % planes), d = (int) it.get_local_id(0);
            const long long pos = pos_of(p, j);
            const float x = (plane ? p.v : p.k)[j * cur_stride + (int64_t) h * HD + d];
            const float am = sycl::reduce_over_group(sg, sycl::fabs(x), sycl::maximum<float>());
            if (sg.get_local_linear_id() == 0) wmax[sg.get_group_linear_id()] = am;
            sycl::group_barrier(it.get_group());
            // d < HD (256), so g * 2 fits int.
            const int g = d / 64;
            const float amax = sycl::fmax(wmax[static_cast<size_t>(g * 2)], wmax[static_cast<size_t>(g * 2) + 1]); // NOLINT(bugprone-misplaced-widening-cast)
            const uint16_t sbits = f16_gpu::f16_from_f32(amax / 127.0f);
            const float sf = f16_gpu::f32_from_f16(sbits);
            int q = 0;
            if (sf > 0.0f) q = sycl::clamp((int) sycl::rint(x / sf), -127, 127);
            const int row = tcq::ring_row(plane, (int) (pos % kTcqRing), h, kv_heads);
            int8_t* codes = const_cast<int8_t*>(tcq::ring_codes(p.tc_ring, row, HD));
            codes[d] = (int8_t) q;
            if (d % 64 == 0) const_cast<uint16_t*>(tcq::ring_scales(p.tc_ring, row, HD, kv_heads))[g] = sbits;
            if (d == 0 && plane == 0 && h == 0) {
                const long long last = pos_of(p, p.n_tok - 1);
                const int cend = (int) ((last + 1) / kTcqBlock * kTcqBlock);
                if (pos >= cend) {   // not coded by this append: its block is read from the ring
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> ce(
                        *tcq::ring_coded(p.tc_ring, HD, kv_heads));
                    ce.fetch_min((int) (pos / kTcqBlock * kTcqBlock));
                }
            }
        });
    });
}

}  // namespace

void kv_append_tcq(const KvAppend& a, const QsaShapes& s, void* stream) {
    if (s.head_dim != HD)
        throw core::DeviceError("kv_append_tcq: the trellis-coded cache wants head_dim 256 (the rotation's size)");
    if (a.tc_bits != 4 && a.tc_bits != 5) throw core::DeviceError("kv_append_tcq: 4 or 5 bits a value");
    if (a.host != nullptr || a.stage != nullptr)
        throw core::DeviceError("kv_append_tcq: the trellis-coded cache has no host copy (KV streaming)");
    if (a.n_tok < 1) return;
    const int bits = a.tc_bits;
    const Mults mu = mults_of(bits);
    const int jobs = mu.n * 4;
    const int kv_heads = (int) s.n_head_kv, page_size = (int) s.page_size;
    const int planes = a.k_tc == a.v_tc && a.k == a.v ? 1 : 2;
    const int64_t cur_stride = a.cur_stride > 0 ? a.cur_stride : (int64_t) kv_heads * HD;
    const int uu = 256 >> bits;   // a state's predecessors share one of uu groups
    // deferred (a ring): the cells of the blocks this append completes, from the block of its first cell, which
    // are at most n_tok + kTcqBlock - 1 (the append's cells are consecutive positions, as the prompt's chunks and the
    // windows write them); else each cell of the append
    const bool defer = a.tc_ring != nullptr;
    const int64_t n_rows = defer ? a.n_tok + kTcqBlock - 1 : a.n_tok;
    auto& queue = queue_for(stream);
    const sycl::device dev = queue.get_device();
    // sub-groups a row: every job at once where the device takes the work-group and its local memory
    const size_t wg_max = dev.get_info<sycl::info::device::max_work_group_size>();
    const size_t local = dev.get_info<sycl::info::device::local_mem_size>();
    const size_t fixed = HD * sizeof(float) + (size_t) mu.n * HD + (size_t) jobs * sizeof(float) + 256;
    int sgs = std::min<int>(jobs, (int) (wg_max / WARP));
    while (sgs > 1 && fixed + (size_t) sgs * kTcqChain * uu > local) --sgs;
    const KvAppend p = a;
    sycl::event e = queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> xr(sycl::range<1>(HD), hd);                       // the row
        sycl::local_accessor<uint8_t, 1> bp(sycl::range<1>((size_t) sgs * kTcqChain * uu), hd);   // a job's choices
        sycl::local_accessor<uint8_t, 1> st(sycl::range<1>((size_t) mu.n * HD), hd);    // each multiplier's path
        sycl::local_accessor<float, 1> err(sycl::range<1>((size_t) jobs), hd);          // each job's error
        sycl::local_accessor<float, 1> ssq(sycl::range<1>(1), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) n_rows * planes * kv_heads * sgs * WARP, (size_t) sgs * WARP),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t wg = (int64_t) it.get_group(0);
            const int h = (int) (wg % kv_heads);
            const int64_t ip = wg / kv_heads, ri = ip / planes;
            const bool is_v = ip % planes == 1;
            const int tid = (int) it.get_local_id(0), nthr = sgs * WARP;
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            long long pos;
            int64_t j = ri;           // the append's cell the row comes from, -1: the ring's
            if (defer) {
                const long long first = pos_of(p, 0), last = pos_of(p, p.n_tok - 1);
                const long long c0 = first / kTcqBlock * kTcqBlock, cend = (last + 1) / kTcqBlock * kTcqBlock;
                pos = c0 + ri;
                if (pos >= cend) return;   // uniform: no block completed past this cell
                j = pos >= first ? (int64_t) (pos - first) : -1;
            } else {
                pos = pos_of(p, ri);
            }
            if (j >= 0) {
                const float* src = (is_v ? p.v : p.k) + j * cur_stride + (int64_t) h * HD;
                for (int k = tid; k < HD; k += nthr) xr[k] = src[k];
            } else {
                const int row = tcq::ring_row(is_v ? 1 : 0, (int) (pos % kTcqRing), h, kv_heads);
                const int8_t* codes = tcq::ring_codes(p.tc_ring, row, HD);
                const uint16_t* scales = tcq::ring_scales(p.tc_ring, row, HD, kv_heads);
                for (int k = tid; k < HD; k += nthr) xr[k] = (float) codes[k] * f16_gpu::f32_from_f16(scales[k / 64]);
            }
            sycl::group_barrier(grp);
            if (defer && j >= 0 && sgi == 0) {   // the append's cells through INT8 as the ring keeps them (ring_write)
                for (int g = 0; g < HD / 64; ++g) {
                    const float a0 = xr[g * 64 + lane], a1 = xr[g * 64 + 32 + lane];
                    const float am = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a0), sycl::fabs(a1)),
                                                             sycl::maximum<float>());
                    const float sf = f16_gpu::f32_from_f16(f16_gpu::f16_from_f32(am / 127.0f));
                    const int q0 = sf > 0.0f ? sycl::clamp((int) sycl::rint(a0 / sf), -127, 127) : 0;
                    const int q1 = sf > 0.0f ? sycl::clamp((int) sycl::rint(a1 / sf), -127, 127) : 0;
                    xr[g * 64 + lane] = (float) q0 * sf;
                    xr[g * 64 + 32 + lane] = (float) q1 * sf;
                }
            }
            sycl::group_barrier(grp);
            if (sgi == 0) {   // the row's sum of squares, in one order
                float q = 0.0f;
                for (int k = 0; k < HD / WARP; ++k) {
                    const float v = xr[k * WARP + lane];
                    q = sycl::fma(v, v, q);
                }
                q = sycl::reduce_over_group(sg, q, sycl::plus<float>());
                if (lane == 0) ssq[0] = q;
            }
            sycl::group_barrier(grp);
            float rms = sycl::sqrt(ssq[0] / (float) HD);
            if (rms == 0.0f) rms = 1.0f;
            const int kq = 1 << bits;
            uint8_t* bps = &bp[(size_t) sgi * kTcqChain * uu];
            for (int job = sgi; job < jobs; job += sgs) {
                const int mi = job >> 2, c = job & 3;
                float sc = f16_gpu::f32_from_f16(f16_gpu::f16_from_f32(rms * mu.m[mi]));
                if (sc == 0.0f) sc = 1.0f;
                const float inv = 256.0f / sc;
                int code[8], cost[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    code[i] = tcq::code_of(bits, 32 * i + lane);
                    cost[i] = 32 * i + lane == 0 ? 0 : (1 << 29);   // the chain starts in state 0
                }
                for (int t = 0; t < kTcqChain; ++t) {
                    const int x = (int) sycl::clamp(sycl::rint(xr[c * kTcqChain + t] * inv), -2048.0f, 2048.0f);
                    // the lane's predecessors (states 32 i + lane, all of group lane mod uu), then the group's lanes
                    int64_t key = ((int64_t) cost[0] << 8) | lane;
#pragma unroll
                    for (int i = 1; i < 8; ++i) key = kmin(key, ((int64_t) cost[i] << 8) | (32 * i + lane));
                    for (int o = uu; o < WARP; o <<= 1) key = kmin(key, sycl::permute_group_by_xor(sg, key, o));
                    if (lane < uu) bps[t * uu + lane] = (uint8_t) (key & 0xFF);
                    // state 32 i + lane takes its group's least from lane (state >> bits), the cost alone in 32 bits:
                    // with 5 bits that lane is i for every lane (one value broadcast), with 4 it is 2 i or 2 i + 1
                    const int mc = (int) (key >> 8);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int m = bits == 5 ? sycl::group_broadcast(sg, mc, i)
                                                : sycl::select_from_group(sg, mc, (size_t) ((32 * i + lane) >> bits));
                        const int dx = x - code[i];
                        cost[i] = m + dx * dx;
                    }
                }
                int64_t key = ((int64_t) cost[0] << 8) | lane;
#pragma unroll
                for (int i = 1; i < 8; ++i) key = kmin(key, ((int64_t) cost[i] << 8) | (32 * i + lane));
                key = sycl::reduce_over_group(sg, key, sycl::minimum<int64_t>());
                sycl::group_barrier(sg);   // the choices written by the group's lanes
                uint8_t* path = &st[(size_t) mi * HD + (size_t) c * kTcqChain];
                if (lane == 0) {
                    int sx = (int) (key & 0xFF);
                    for (int t = kTcqChain - 1; t >= 0; --t) {
                        path[t] = (uint8_t) sx;
                        sx = bps[t * uu + (sx >> bits)];
                    }
                }
                sycl::group_barrier(sg);
                float e2 = 0.0f;
                for (int k = lane; k < kTcqChain; k += WARP) {
                    const float d = xr[c * kTcqChain + k] - (float) tcq::code_of(bits, path[k]) * (sc / 256.0f);
                    e2 = sycl::fma(d, d, e2);
                }
                e2 = sycl::reduce_over_group(sg, e2, sycl::plus<float>());
                if (lane == 0) err[job] = e2;
                sycl::group_barrier(sg);   // the next job's search writes this sub-group's choices again
            }
            sycl::group_barrier(grp);
            // the multiplier of least error over the four chains (ties to the first)
            int best = 0;
            float best_e = 0.0f;
            // mu.n <= MAX_MULT, so mi * 4 fits int.
            for (int mi = 0; mi < mu.n; ++mi) {
                const float e4 = ((err[static_cast<size_t>(mi * 4)] + err[static_cast<size_t>(mi * 4) + 1]) + err[static_cast<size_t>(mi * 4) + 2]) + err[static_cast<size_t>(mi * 4) + 3]; // NOLINT(bugprone-misplaced-widening-cast)
                if (mi == 0 || e4 < best_e) { best_e = e4; best = mi; }
            }
            const long long page = (long long) p.page_table[pos / page_size];
            if (page < 0) return;   // uniform: a block the KV streaming could not make resident
            const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
            uint8_t* dst = (is_v ? p.v_tc : p.k_tc) + row * (long long) tcq::row_bytes(HD, bits);
            const uint8_t* path = &st[(size_t) best * HD];
            for (int w = tid; w < 8 * bits; w += nthr) {   // the fields, B bits a value, first bit most significant
                uint32_t word = 0;
                for (int i = 0; i < 32; ++i) {
                    const int g = 32 * w + i, v = g / bits, b = g - v * bits;
                    word |= (uint32_t) ((path[v] & (kq - 1)) >> (bits - 1 - b) & 1) << (31 - i);
                }
                reinterpret_cast<uint32_t*>(dst)[w] = word;
            }
            if (tid == 0) {
                float sc = f16_gpu::f32_from_f16(f16_gpu::f16_from_f32(rms * mu.m[best]));
                if (sc == 0.0f) sc = 1.0f;
                *reinterpret_cast<float*>(dst + HD * bits / 8) = sc;
            }
        });
    });
    if (defer) {
        // the coded bound up to the blocks this append completed (one work-item), before the ring's write lowers it
        // for the cells past them
        queue.single_task([=] {
            const long long first = pos_of(p, 0), last = pos_of(p, p.n_tok - 1);
            const long long c0 = first / kTcqBlock * kTcqBlock, cend = (last + 1) / kTcqBlock * kTcqBlock;
            if (cend > c0) *tcq::ring_coded(p.tc_ring, HD, kv_heads) = (int) cend;
        });
        e = ring_write(queue, p, kv_heads, planes, cur_stride);
    }
    if (!stream) core::Runtime::get().wait(e, "kv_append_tcq");
}

}  // namespace strata::kernels
