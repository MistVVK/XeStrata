// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/vmm.cpp - see include/strata/core/vmm.hpp.  A chunk is a physical segment of the current device
// (strata::gpu::vmem_new); a range's addresses come from vmem_reserve.  Mapping a segment makes it readable and
// writable at once, so there is no separate access step.
#include "strata/core/vmm.hpp"
#include "strata/core/gpu.hpp"

#include <mutex>

namespace strata::core {
namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    uint64_t device_gran = 0;   ///< the device's own report
};

// The chunk: the device's granularity, rounded up to 2 MiB (CUDA's granularity).  Each chunk is a physical allocation
// of its own, and on the B70 (Level Zero, 64 KiB granularity) the count is what costs: mapping 1 GiB took 2,570 ms
// in 64 KiB chunks and 73 ms in 2 MiB ones, and every submission got slower with it (a 1 GiB memset 4.0 ms with
// 16,384 chunks, 1.9 ms with 512) - a 26 GiB cache in 64 KiB chunks did not finish filling in 10 minutes.
constexpr uint64_t kMinChunk = (uint64_t) 2 << 20;

// the current device's report, read once (the elastic K/V runs on one GPU)
const Api& api() {
    static Api a;
    static std::once_flag once;
    std::call_once(once, [] {
        if (!strata::gpu::vmem_supported()) return;
        const uint64_t g = strata::gpu::vmem_granularity();
        if (g == 0) return;
        a.device_gran = g;
        a.gran = (kMinChunk + g - 1) / g * g;
        a.ok = true;
    });
    return a;
}

strata::gpu::VmemSegment* seg(VmmChunk h) { return reinterpret_cast<strata::gpu::VmemSegment*>((uintptr_t) h); }

}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }
bool vmm_large_pages() { return api().ok && api().device_gran >= kMinChunk; }

VmmChunk vmm_chunk_new() {
    if (!api().ok) return 0;
    return (VmmChunk) (uintptr_t) strata::gpu::vmem_new((size_t) api().gran);
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0) strata::gpu::vmem_delete(seg(h));
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    void* p = strata::gpu::vmem_reserve((size_t) (n * a.gran));
    if (p == nullptr) return false;
    base_ = (unsigned long long) (uintptr_t) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    strata::gpu::vmem_free((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * api().gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const uint64_t G = api().gran;
    if (!strata::gpu::vmem_attach(seg(h), (void*) (uintptr_t) (base_ + (uint64_t) i * G), (size_t) G)) return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t, int64_t) { return true; }   // a mapped segment is readable and writable

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const uint64_t G = api().gran;
    if (!strata::gpu::vmem_detach((void*) (uintptr_t) (base_ + (uint64_t) i * G), (size_t) G)) return 0;
    const VmmChunk h = h_[(size_t) i];
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core
