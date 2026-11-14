// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/split_role.cpp - the GPU probe of the helper's fixed cost a layer; the prediction and the choice are pure,
// in split_role.hpp (both engines call them, split_role_test.cpp checks them on the CPU).
#include "strata/core/split_role.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace strata::core {

double probe_helper_layer_ms(double& submit_ms) {
    constexpr size_t kMeta = (size_t) 2 << 10, kBytes = (size_t) 64 << 10;
    submit_ms = -1.0;
    void* h = nullptr;
    void* d = nullptr;
    strata::gpu::Stream s = strata::gpu::stream_create();
    if (s == nullptr) return -1.0;
    if (!strata::gpu::alloc_host(&h, kMeta) || !strata::gpu::alloc_device(&d, kBytes)) {
        if (h != nullptr) strata::gpu::free(h);
        strata::gpu::stream_destroy(s);
        return -1.0;
    }
    std::memset(h, 0, kMeta);
    std::vector<double> sub, all;
    for (int i = 0; i < 17; ++i) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        bool ok = strata::gpu::copy_async(d, h, kMeta, s);
        for (int c = 0; c < 3 && ok; ++c) ok = strata::gpu::memset_async((char*) d + kMeta, c, kBytes - kMeta, s);
        if (!ok) break;
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        strata::gpu::Status st = strata::gpu::Status::Pending;
        while ((st = strata::gpu::stream_status(s)) == strata::gpu::Status::Pending) std::this_thread::yield();
        if (st == strata::gpu::Status::Failed) break;
        if (i > 0) {
            sub.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            all.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    strata::gpu::stream_sync(s);
    strata::gpu::free(d);
    strata::gpu::free(h);
    strata::gpu::stream_destroy(s);
    if (all.empty()) return -1.0;
    std::sort(sub.begin(), sub.end());
    std::sort(all.begin(), all.end());
    submit_ms = sub[sub.size() / 2];
    return all[all.size() / 2];
}

}  // namespace strata::core
