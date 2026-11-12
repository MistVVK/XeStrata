// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/link_probe.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::core {
// Copies of `bytes` from `src` to `dst` after a warm-up (the context up, the copy engine primed): four bursts, each one
// waited for and timed on the host's clock, the fastest in GB/s (< 0 when a copy fails).  `samples`, when given, gets
// every burst's reading for the log.  #485 (upstream 4e308eb): a single timed burst can read a link low (a link still
// in a low-power state, other DMA, a context not yet up to speed), and such a disturbance only ever slows a copy, so the
// fastest of the four is the link's figure.  #59: the host's clock, not the GPU's timestamps (timing events): upstream
// timed the bursts with those, and reading them back is the one thing this probe asked of the driver that no other part
// of a start does - an Arc A770 on Windows faulted in it, which ended the start.  A 256 MiB burst takes 9 ms at 28 GB/s;
// a wait's wake-up (tens of microseconds) moves the figure by under 1%.
double best_copy_gbps(void* dst, const void* src, size_t bytes, std::string* samples) {
    constexpr int kBursts = 4;
    if (samples != nullptr) samples->clear();
    if (!strata::gpu::copy(dst, src, bytes)) return -1.0;
    double bw = -1.0;
    for (int b = 0; b < kBursts; ++b) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        if (!strata::gpu::copy(dst, src, bytes)) return -1.0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double sv = s > 1e-5 ? (double) bytes / s / 1e9 : -1.0;
        bw = std::max(bw, sv);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", sv);
            *samples += buf;
        }
    }
    return bw;
}

double probe_host_to_device_gbps(std::string* samples) {
    constexpr size_t bytes = 256ull << 20;
    void *h = nullptr, *d = nullptr;
    if (!strata::gpu::alloc_host(&h, bytes)) return -1.0;
    if (!strata::gpu::alloc_device(&d, bytes)) {
        strata::gpu::free(h);
        return -1.0;
    }
    std::memset(h, 0, bytes);
    const double bw = best_copy_gbps(d, h, bytes, samples);
    strata::gpu::free(d);
    strata::gpu::free(h);
    return bw;
}
} // namespace strata::core
