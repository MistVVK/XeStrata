// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/dense_slab.hpp"
#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::core {
namespace {
// This probes the reported device, not a driver or model name.  The same queue and launch are timed before and with
// extra live allocations; allocations are tiny because their count, rather than their bytes, is the suspected cost.
// A later startup repeats the probe: no persistent device table or guessed maker-specific default.
bool allocations_cost(sycl::queue& q) {
    const size_t sample = std::clamp<size_t>(q.get_device().get_info<sycl::info::device::max_compute_units>(), 32, 256);
    std::vector<void*> extra;
    int* result = nullptr;
    if (!gpu::alloc_device(&result, sizeof(int))) throw DeviceError("dense slab: timing buffer allocation failed");
    auto release = [&] {
        q.wait_and_throw();
        for (void* p : extra) gpu::free(p);
        gpu::free(result);
    };
    try {
        auto launch = [&] { q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { *result = 1; }); };
        const auto warm = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - warm).count() < 0.5) {
            for (int i = 0; i < 32; ++i) launch();
            q.wait_and_throw();
        }
        auto time = [&] {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 128; ++i) launch();
            q.wait_and_throw();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 128;
        };
        double before = 0;
        for (int i = 0; i < 3; ++i) before += time();
        for (size_t i = 0; i < sample; ++i) {
            void* p = nullptr;
            if (!gpu::alloc_device(&p, 256)) throw DeviceError("dense slab: timing allocations failed");
            extra.push_back(p);
        }
        double after = 0;
        for (int i = 0; i < 3; ++i) after += time();
        before /= 3; after /= 3;
        const bool faster = after > before * 1.10 && after - before > 1.0;
        std::fprintf(stderr, "xestrata: dense allocation probe: %zu extra allocations, %.1f -> %.1f us; slabs %s\n",
                     sample, before, after, faster ? "on" : "off");
        release();
        return faster;
    } catch (...) {
        for (void* p : extra) gpu::free(p);
        gpu::free(result);
        throw;
    }
}
}  // namespace

DenseSlab::DenseSlab() {
    auto& rt = Runtime::get();
    device_ = rt.ordinal();
    const char* mode = strata::env_get("DENSE_SLAB");
    bool on = false;
    if (!mode || !*mode || std::strcmp(mode, "auto") == 0) on = allocations_cost(rt.compute());
    else if (std::strcmp(mode, "1") == 0) on = true;
    else if (std::strcmp(mode, "0") != 0) throw DeviceError("XESTRATA_DENSE_SLAB must be auto, 0 or 1");
    if (!on) return;
    size_t free = 0, total = 0;
    if (!gpu::mem_info(&free, &total)) throw DeviceError("dense slab: free memory query failed");
    const size_t units = std::max<size_t>(1, rt.device().get_info<sycl::info::device::max_compute_units>());
    block_bytes_ = std::min<size_t>(max_alloc_bytes(), free / units) & ~(size_t) 255;
    if (block_bytes_ < 512) throw DeviceError("dense slab: insufficient space for a block");
}

DenseSlab::~DenseSlab() {
    const OnDevice on(device_);
    for (const Block& b : blocks_) gpu::free(b.base);
}

bool DenseSlab::owns(const void* p) const {
    const uintptr_t a = (uintptr_t) p;
    for (const Block& b : blocks_)
        if (a >= (uintptr_t) b.base && a - (uintptr_t) b.base < b.size) return true;
    return false;
}

void* DenseSlab::alloc(size_t bytes) {
    if (bytes > SIZE_MAX - 255) throw DeviceError("dense slab: weight size overflow");
    const size_t need = (bytes + 255) & ~(size_t) 255;
    if (!block_bytes_ || need >= block_bytes_ / 2) return gpu::alloc_device(bytes);
    for (Block& b : blocks_)
        if (need <= b.size - b.used) {
            void* p = static_cast<uint8_t*>(b.base) + b.used;
            b.used += need;
            return p;
        }
    void* p = gpu::alloc_device(block_bytes_);
    if (!p) return nullptr;
    blocks_.push_back({p, block_bytes_, need});
    return p;
}
}  // namespace strata::core
