// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The elastic K/V's VMM ranges (vmm.hpp): chunks map readable and writable, a chunk unmapped from one range and mapped
// into another keeps its bytes (what the expert cache hands the K/V), and a range releases what it holds.
#include "strata/core/vmm.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using strata::core::VmmChunk;
using strata::core::VmmRange;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    strata::core::Runtime::get();
    if (!strata::core::vmm_available()) {
        std::printf("vmm_test: the device has no virtual memory (sycl_ext_oneapi_virtual_mem) - skipped\n");
        return 77;
    }
    const uint64_t G = strata::core::vmm_granularity();
    CHECK(G >= 4096);
    VmmRange a, b;
    CHECK(a.reserve(8 * G + 1));   // rounds up
    CHECK(a.chunks() == 9);
    CHECK(b.reserve(4 * G));
    CHECK(a.mapped_count() == 0);
    CHECK(a.map_range(0, 6, [] { return (VmmChunk) 0; }));
    CHECK(a.mapped_count() == 6 && a.mapped(5) && !a.mapped(6));
    std::vector<uint8_t> h(6 * G), back(6 * G);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
    CHECK(strata::gpu::copy(a.base(), h.data(), h.size()));
    // chunks 4 and 5 move to b's chunks 1 and 0. The host-to-device copy above may still be in flight (a pageable
    // copy returns once the data is staged), and a chunk unmapped under a copy faults it (an illegal memory access
    // that sticks to every later call), so the device is drained first, as generate.cpp does before it unmaps
    CHECK(strata::gpu::device_sync());
    const VmmChunk c4 = a.unmap(4), c5 = a.unmap(5);
    CHECK(c4 != 0 && c5 != 0 && !a.mapped(4) && a.unmap(4) == 0);
    std::vector<VmmChunk> give = {c4, c5};   // map_range takes them from the back: chunk 0 <- c5, chunk 1 <- c4
    CHECK(b.map_range(0, 2, [&] { VmmChunk x = give.back(); give.pop_back(); return x; }));
    CHECK(strata::gpu::copy(back.data(), b.base(), G));
    bool same = true;
    for (uint64_t i = 0; i < G; ++i) same = same && back[i] == h[5 * G + i];
    CHECK(same);
    CHECK(strata::gpu::copy(back.data(), b.base() + G, G));
    same = true;
    for (uint64_t i = 0; i < G; ++i) same = same && back[i] == h[4 * G + i];
    CHECK(same);
    // a's first chunks are untouched, and map_range fills a hole with new memory, skipping mapped chunks
    CHECK(strata::gpu::copy(back.data(), a.base(), 4 * G));
    same = true;
    for (uint64_t i = 0; i < 4 * G; ++i) same = same && back[i] == h[i];
    CHECK(same);
    CHECK(a.map_range(3, 7, [] { return (VmmChunk) 0; }) && a.mapped_count() == 7);
    CHECK(strata::gpu::memset(a.base() + 4 * G, 7, 3 * G) && strata::gpu::device_sync());
    size_t f0 = 0, f1 = 0, t = 0;
    // the free-memory readout as a witness of the release: only where it follows an ordinary allocation of the same
    // size (the B70's Level Zero report did not move for 1 GiB allocated and written)
    const uint64_t probe = 9 * G;
    bool follows = false;
    if (strata::gpu::mem_info(&f0, &t)) {
        if (void* p = strata::gpu::alloc_device((size_t) probe)) {
            strata::gpu::memset(p, 1, (size_t) probe);
            strata::gpu::device_sync();
            follows = strata::gpu::mem_info(&f1, &t) && f1 + probe / 2 <= f0;
            strata::gpu::free(p);
        }
    }
    strata::gpu::mem_info(&f0, &t);
    a.release();
    b.release();
    if (follows) {
        for (int i = 0; i < 20; ++i) {   // the free-memory counter may lag a release by a moment on Windows (WDDM)
            strata::gpu::mem_info(&f1, &t);
            if (f1 >= f0 + 8 * G) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        CHECK(f1 >= f0 + 8 * G);   // 7 + 2 chunks went back (the driver may round)
    } else {
        std::printf("vmm_test: the free-memory readout does not follow allocations here: the release is not checked\n");
    }
    std::printf("vmm_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
