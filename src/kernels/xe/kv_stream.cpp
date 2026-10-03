// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/kv_stream.cpp - the Xe port of Strata's src/kernels/cuda/kv_stream.cu: KV streaming between host pools and a
// ring of VRAM slots, resolved on the device.
//
// The host pools are read and written by kernels (the resolve copy here, the KV appends), so on Xe they must be host
// USM: memory registered only for device copies is not addressable from a kernel.  A warp is a sub-group of 32.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <algorithm>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

template<class T>
using global_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;
template<class T>
using local_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                      sycl::access::address_space::local_space>;

// Exclusive work-group scan of one int per work-item; `total` is the group's sum.
// NW sub-groups of WARP (at most 32)
template <int NW>
inline int block_scan(const sycl::nd_item<1>& it, int v, const sycl::local_accessor<int, 1>& warp_sums, int& total) {
    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id(), w = (int) sg.get_group_linear_id();
    const int x = sycl::inclusive_scan_over_group(sg, v, sycl::plus<int>());
    if (lane == WARP - 1) warp_sums[w] = x;
    sycl::group_barrier(it.get_group());
    if (w == 0) {
        const int y = sycl::inclusive_scan_over_group(sg, lane < NW ? (int) warp_sums[lane] : 0, sycl::plus<int>());
        if (lane < NW) warp_sums[lane] = y;
    }
    sycl::group_barrier(it.get_group());
    total = warp_sums[NW - 1];
    const int excl = x - v + (w > 0 ? (int) warp_sums[w - 1] : 0);
    sycl::group_barrier(it.get_group());
    return excl;
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    const KvStreamMap mm = m;
    const size_t total = 128 * 256;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(total, 256), [=](sycl::nd_item<1> it) {
        const long long i0 = (long long) it.get_global_id(0), st = (long long) total;
        for (long long i = i0; i < mm.n_blocks; i += st) mm.page_table[i] = -1;
        for (long long i = i0; i < mm.n_slots; i += st) {
            mm.slot_block[i] = -1;
            mm.slot_stamp[i] = -1;
            mm.slot_ref[i] = 0;
        }
        if (i0 < kKvCtlInts) mm.ctl[i0] = 0;
    });
    finish(stream, e, "kv_stream_reset");
}

namespace {
// The resolve kernel: one work-group of RT work-items (1024 where the device takes it).
template <int RT>
void resolve_launch(sycl::queue& queue, const KvStreamMap mm, const int32_t* ids, const int32_t* steps, int nq,
                    int capi, int page_size) {
    queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> sh(sycl::range<1>(3), hd);          // s_nmiss, s_lookups, s_cut
        sycl::local_accessor<int, 1> warp_sums(sycl::range<1>(32), hd);
        hd.parallel_for(sycl::nd_range<1>(RT, RT), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const int tid = (int) it.get_local_id(0);
            local_atomic<int> s_nmiss(sh[0]), s_lookups(sh[1]);
            const int epoch = mm.ctl[0] + 1;
            if (tid == 0) { sh[0] = 0; sh[1] = 0; }
            sycl::group_barrier(grp);
            int lookups = 0;
            for (int q = 0; q < nq; ++q) {
                const int width = steps[q * kStepCount + kStepWidth];
                const int32_t* qi = ids + (long long) q * capi;
                for (int i = tid; i < width; i += RT) {
                    const int b = qi[i] / page_size;
                    if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
                    ++lookups;
                    const int sl = mm.page_table[b];
                    if (sl >= 0) {
                        mm.slot_stamp[sl] = epoch;
                        mm.slot_ref[sl] = 1;
                    } else if (sl == -1) {
                        int expected = -1;
                        if (global_atomic<int32_t>(mm.page_table[b]).compare_exchange_strong(expected, -2))
                            mm.miss_block[s_nmiss.fetch_add(1)] = b;
                    }
                }
            }
            s_lookups.fetch_add(lookups);
            sycl::group_barrier(grp);
            const int need = sh[0], n = (int) mm.n_slots;
            int hand = mm.ctl[1], got = 0;
            for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
                const int j = (int) (((long long) hand + tid) % n);
                const bool mine = mm.slot_stamp[j] == epoch;
                const bool cand = !mine && (mm.slot_block[j] < 0 || mm.slot_ref[j] == 0);
                int total = 0;
                const int rank = block_scan<RT / WARP>(it, cand ? 1 : 0, warp_sums, total);
                const int want = need - got;
                if (tid == 0) sh[2] = RT;
                sycl::group_barrier(grp);
                if (cand && rank == want - 1) sh[2] = tid + 1;   // the hand stops just past the last slot taken
                sycl::group_barrier(grp);
                const int cut = sh[2];
                if (cand && rank < want) {
                    mm.miss_slot[got + rank] = j;
                    mm.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
                } else if (tid < cut && !mine) {
                    mm.slot_ref[j] = 0;
                }
                got += total < want ? total : want;
                hand = (int) (((long long) hand + cut) % n);
                sycl::group_barrier(grp);
            }
            const int placed = got < need ? got : need;
            for (int k = tid; k < need; k += RT) {
                const int b = mm.miss_block[k];
                if (k >= placed) { mm.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
                const int sl = mm.miss_slot[k];
                const int old = mm.slot_block[sl];
                if (old >= 0) mm.page_table[old] = -1;
                mm.slot_block[sl] = b;
                mm.slot_stamp[sl] = epoch;
                mm.slot_ref[sl] = 1;
                mm.page_table[b] = sl;
            }
            if (tid == 0) {
                mm.ctl[0] = epoch;
                mm.ctl[1] = hand;
                mm.ctl[2] = placed;
                if (placed < need) mm.ctl[3] = 1;
                auto* c = reinterpret_cast<unsigned long long*>(mm.ctl + 4);
                c[0] += (unsigned long long) placed;
                c[1] += (unsigned long long) sh[1];
                c[2] += 1ull;
            }
        });
    });
}
}  // namespace

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0)
        throw core::DeviceError("kv_stream: a block's scale run must be a multiple of 16 bytes");
    // one sweep step looks at RT consecutive slots (hand + thread) % n_slots; with fewer slots than RT two work-items
    // see the same slot and may both take it for two misses (upstream f3925cf).  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    auto& queue = queue_for(stream);
    const int rt = xe::work_group_upto_1024(queue);
    if (m.n_slots < rt)
        throw core::DeviceError("kv_stream: " + std::to_string(m.n_slots) + " slots is fewer than the resolve block (" +
                                std::to_string(rt) + "): the clock sweep would take a slot twice");
    const KvStreamMap mm = m;
    const int nq = (int) n_q, capi = (int) cap, page_size = (int) s.page_size;
    if (rt == 1024) resolve_launch<1024>(queue, mm, ids, steps, nq, capi, page_size);
    else if (rt == 512) resolve_launch<512>(queue, mm, ids, steps, nq, capi, page_size);
    else resolve_launch<256>(queue, mm, ids, steps, nq, capi, page_size);
    const Runs r = runs_of(slots, host, fmt, s);
    const auto e = queue.parallel_for(sycl::nd_range<1>(96 * 128, 128), [=](sycl::nd_item<1> it) {
        const int need = mm.ctl[2];
        for (int k = (int) it.get_group(0); k < need; k += 96) {
            const long long b = mm.miss_block[k], sl = mm.miss_slot[k];
            for (int a = 0; a < r.n; ++a) {
                const auto* src = reinterpret_cast<const sycl::uint4*>(r.src[a] + b * r.len[a]);
                auto* dst = reinterpret_cast<sycl::uint4*>(r.dst[a] + sl * r.len[a]);
                for (int i = (int) it.get_local_id(0); i < r.len[a] / 16; i += 128) dst[i] = src[i];
            }
        }
    });
    finish(stream, e, "kv_stream_resolve");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    const size_t total = 64 * 256;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(total, 256), [=](sycl::nd_item<1> it) {
        for (long long i = (long long) it.get_global_id(0); i < n_blocks; i += (long long) total)
            page_table[i] = (int32_t) (i % n_slots);
    });
    finish(stream, e, "kv_ring_table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    auto& queue = queue_for(stream);
    sycl::event e;
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            e = queue.memcpy(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]));
        b += run;
    }
    if (!stream) core::Runtime::get().finish(queue);
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    auto& queue = queue_for(stream);
    for (int a = 0; a < r.n; ++a) queue.memcpy(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]));
    if (!stream) core::Runtime::get().finish(queue);
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr) return r;
    auto& rt = core::Runtime::get();
    rt.wait(rt.compute().memcpy(c, m.ctl, sizeof(c)), "kv_stream_counters");
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
