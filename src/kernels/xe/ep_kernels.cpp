// SPDX-FileCopyrightText: 2026 recutita, MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/ep_kernels.cpp - see include/strata/kernels/ep_kernels.hpp.
#include "strata/kernels/ep_kernels.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>

namespace strata::kernels {
namespace {

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
void done(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

using sys_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;
// A word the other card wrote, read by an atomic RMW: on the B70 a spin saw the store in 60 us this way, 4 ms with an
// atomic load, never with a volatile load; a copy costs 5.8 us per 800 KiB against 4.8 with plain loads.
inline uint32_t peer_read(const uint32_t* p) { return sys_u32(*const_cast<uint32_t*>(p)).fetch_add(0u); }
inline void peer_store(uint32_t* p, uint32_t v) {
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    sys_u32(*p).store(v);
}

}  // namespace

void ep_bump(uint32_t* ctr, void* stream) {
    done(stream, Q(stream).single_task([=] { *ctr += 1; }), "ep_bump");
}

void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;   // q8_1 blocks are 36 bytes: whole words
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = reinterpret_cast<const uint32_t*>(xq);
        uint32_t* d = reinterpret_cast<uint32_t*>(peer_xq);
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        // every work-item's stores into the peer before the flag
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (lid == 0) peer_store(peer_flag, *ctr);
    });
    done(stream, e, "ep_push");
}

void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream) {
    const auto e = Q(stream).single_task([=] {
        const uint32_t want = *ctr;
        uint32_t spin = 0;
        // the epochs only grow, and wrap: compare the difference
        while ((int32_t) (peer_read(flag) - want) < 0) {
            if (++spin >= spin_max) {
                sys_u32(*err).store(code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
    done(stream, e, "ep_wait");
}

void ep_copy_peer(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return;
    constexpr size_t WG = 256;
    const size_t words = bytes / 4;
    const size_t groups = std::min<size_t>((words + WG - 1) / WG, 64);
    uint32_t* d = static_cast<uint32_t*>(dst);
    const uint32_t* s = static_cast<const uint32_t*>(src);
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) d[i] = peer_read(s + i);
    });
    done(stream, e, "ep_copy_peer");
}

void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done_ctr, uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    using count = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                   sycl::access::address_space::global_space>;
    using f32x4 = float __attribute__((ext_vector_type(4)));
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t ex = ids[r];
        if (!(ex >= 0 && res0[ex] >= 0)) {   // the peer's entry: its row goes to card 0
            const f32x4* s = reinterpret_cast<const f32x4*>(rows + (size_t) r * row);
            f32x4* d = reinterpret_cast<f32x4*>(peer_rows + (size_t) r * row);
            for (int64_t i = (int64_t) it.get_local_id(0); i < row / 4; i += WG) d[i] = s[i];
        }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) == 0 && count(*done_ctr).fetch_add(1u) == (uint32_t) n - 1) {
            *done_ctr = 0;   // every group has counted itself: the next call starts from zero
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            peer_store(peer_flag, *ctr);
        }
    });
    done(stream, e, "ep_send_rows");
}

void ep_merge_rows(float* parts, const float* peer_rows, const float* hit_out, const int32_t* ids, const int32_t* res0,
                   int n, int64_t row, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t ex = ids[r];
        const bool own = ex >= 0 && res0[ex] >= 0;
        float* d = parts + (size_t) r * row;
        if (own) {
            const float* s = hit_out + (size_t) r * row;
            for (int64_t i = (int64_t) it.get_local_id(0); i < row; i += WG) d[i] = 0.0f + s[i];
        } else {
            const uint32_t* s = reinterpret_cast<const uint32_t*>(peer_rows + (size_t) r * row);
            for (int64_t i = (int64_t) it.get_local_id(0); i < row; i += WG) d[i] = sycl::bit_cast<float>(peer_read(s + i));
        }
    });
    done(stream, e, "ep_merge_rows");
}

}  // namespace strata::kernels
