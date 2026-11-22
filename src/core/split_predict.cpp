// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/split_predict.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/platform/env.hpp"
#include "strata/platform/memory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata::core {
// The decode-only predictor is not yet a substitute for this capacity-led policy: on B70 + RTX 4070,
// Ornith-1.5-35B-A3B at 1K changed 15:25 to 39:1, losing 21.7% prefill and 5.0% decode (three runs,
// 2026-11-12). Its prompt path must be priced before the two policies can be combined.
std::vector<int64_t> choose_split_by_room(int64_t layers, uint64_t layer_bytes, bool mtp,
                                          const std::vector<double>& room, const std::vector<double>& link) {
    const int n = (int) room.size();
    std::vector<int64_t> count((size_t) n, 0);
    int64_t left = layers;
    double fast_room = 0;
    for (int d = 0; d < n; ++d) {
        const bool slow = d > 0 && link[0] > 0 && link[(size_t) d] < link[0] / 2 && layer_bytes > 0;
        if (!slow) { fast_room += room[(size_t) d]; continue; }
        // A fifth of the room for block weights, K/V and buffers: RTX 4070 used 8.4 of 9.6 GiB.
        const int64_t whole = (int64_t) (room[(size_t) d] * 0.8 / (double) layer_bytes) - (d == n - 1 && mtp ? 1 : 0);
        // a slow device is given a layer only while one is left for it: with left <= 1 the clamp's bounds would
        // invert, which is undefined.  Nothing is given here, and the layer that is left goes to the devices below.
        count[(size_t) d] = left > 1 ? std::clamp<int64_t>(whole, 1, left - 1) : 0;
        left -= count[(size_t) d];
    }
    if (left < 1 || fast_room <= 0) return {};
    double acc = 0;
    int64_t given = 0;
    int last_fast = 0;
    for (int d = 0; d < n; ++d)
        if (count[(size_t) d] == 0) last_fast = d;
    for (int d = 0; d < n; ++d) {
        if (count[(size_t) d] != 0) continue;
        acc += room[(size_t) d];
        const int64_t upto = d == last_fast ? left : (int64_t) std::llround((double) left * acc / fast_room);
        count[(size_t) d] = std::max<int64_t>(1, upto - given);
        given += count[(size_t) d];
    }
    return count;
}

void enumerate_layer_splits(int64_t layers, const std::vector<double>& bandwidth,
                            const std::function<void(const std::vector<int64_t>&)>& visit) {
    const int n = (int) bandwidth.size();
    std::vector<int64_t> at((size_t) n - 1);
    if (n == 2) {
        for (int64_t k = 2; k < layers; ++k) { at[0] = k; visit(at); }
    } else if (n == 3) {
        for (int64_t k1 = 2; k1 + 1 < layers; ++k1)
            for (int64_t k2 = k1 + 1; k2 < layers; ++k2) { at[0] = k1; at[1] = k2; visit(at); }
    } else if (n == 4) {
        for (int64_t k1 = 2; k1 + 2 < layers; ++k1)
            for (int64_t k2 = k1 + 1; k2 + 1 < layers; ++k2)
                for (int64_t k3 = k2 + 1; k3 < layers; ++k3) { at[0] = k1; at[1] = k2; at[2] = k3; visit(at); }
    } else {
        double total = 0;
        for (double c : bandwidth) total += c;
        double acc = 0;
        for (int i = 0; i + 1 < n; ++i) {
            acc += bandwidth[(size_t) i];
            at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) layers),
                                                  i == 0 ? 2 : at[(size_t) i - 1] + 1, layers - (n - 1 - i));
        }
        visit(at);
    }
}

SplitPredictionResult predict_layer_split(const SplitPrediction& p, const std::vector<int64_t>& cuts) {
    SplitPredictionResult out;
    const int nst = (int) cuts.size() + 1;
    out.capacity.resize(p.order.size());
    out.used.resize(p.order.size());
    for (int i = 0; i < nst; ++i) {
        const int64_t lo = i == 0 ? 0 : cuts[(size_t) i - 1], hi = i + 1 < nst ? cuts[(size_t) i] : p.layers;
        const int64_t raw = p.capacity(i, nst == 1 ? 0 : lo, nst == 1 ? -1 : hi, i + 1 == nst);
        if (raw < p.prompt_min) out.startable = false;
        out.capacity[(size_t) i] = std::max<int64_t>(raw, 0);
    }
    std::vector<double> layer_held((size_t) p.layers, 0.0), layer_mass((size_t) p.layers, 0.0);
    std::vector<bool> full(p.order.size(), false);
    for (size_t r = 0; r < p.profile.size(); ++r) {
        const int64_t l = p.profile[r].first;
        const double mass = p.mass.empty() ? 1.0 : p.mass[r];
        layer_mass[(size_t) l] += mass;
        int st = 0;
        while (st + 1 < nst && l >= cuts[(size_t) st]) ++st;
        if (full[(size_t) st]) continue;
        const int64_t c = p.cost(l, p.profile[r].second);
        if (out.used[(size_t) st] + c > out.capacity[(size_t) st]) { full[(size_t) st] = true; continue; }
        out.used[(size_t) st] += c;
        layer_held[(size_t) l] += mass;
        ++out.held;
    }
    double hand = 0, all = 0, kept = 0;
    std::vector<double> stage_ms((size_t) nst, 0.0);
    for (int i = 1; i < nst; ++i) hand += p.side_ms[(size_t) p.order[(size_t) i - 1]] + p.side_ms[(size_t) p.order[(size_t) i]];
    for (int64_t l = 0; l < p.layers; ++l) {
        int st = 0;
        while (st + 1 < nst && l >= cuts[(size_t) st]) ++st;
        const int d = p.order[(size_t) st];
        const double hit = layer_mass[(size_t) l] > 0 ? layer_held[(size_t) l] / layer_mass[(size_t) l] : 1.0;
        const double experts = p.experts_per_row * p.rows * p.blob_bytes(l);
        stage_ms[(size_t) st] += p.dense_layer / p.gpu_bw[(size_t) d] +
                                 std::max(experts * hit / p.gpu_bw[(size_t) d],
                                          experts * (1.0 - hit) * p.tail_keep / p.cpu_bw * p.link_scale[(size_t) d]);
        all += layer_mass[(size_t) l];
        kept += layer_held[(size_t) l];
    }
    const int last = p.order[(size_t) nst - 1];
    stage_ms[(size_t) nst - 1] += (p.head_bytes + (p.drafts ? (p.rows - 1.0) * p.dense_layer : 0.0)) / p.gpu_bw[(size_t) last];
    out.held_mass = all > 0 ? kept / all : 1.0;
    out.ms = hand;
    if (p.pipelined && nst > 1) out.ms += *std::max_element(stage_ms.begin(), stage_ms.end());
    else for (double t : stage_ms) out.ms += t;
    return out;
}
double split_gpu_bandwidth(int dev) {
    const OnDevice on(dev);
    const size_t n = (size_t) 512 << 20;
    // Random data avoids the compression a device can apply to a uniform memset.
    std::vector<uint64_t> host(n / 8);
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (uint64_t& v : host) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v = x; }
    void *a = nullptr, *b = nullptr;
    std::vector<double> got;
    if (strata::gpu::alloc_device(&a, n) && strata::gpu::alloc_device(&b, n) &&
        strata::gpu::copy(a, host.data(), n) && strata::gpu::copy(b, a, n)) {
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!strata::gpu::copy(b, a, n)) break;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            got.push_back(2.0 * (double) n / std::max(ms, 1e-3));
        }
    }
    if (a) strata::gpu::free(a);
    if (b) strata::gpu::free(b);
    if (got.empty()) return 100e6;   // 100 GB/s if the copy could not run
    std::sort(got.begin(), got.end());
    return got[got.size() / 2];
}
// The RAM as the CPU expert pool reads it, one reader per hardware thread.
double split_cpu_bandwidth() {
    if (const char* v = strata::env_get("SPLIT_CPU_GBPS")) return std::strtod(v, nullptr) * 1e6;
    const int nt = std::max(1, (int) std::thread::hardware_concurrency());
    const size_t per = (size_t) 32 << 20;
    // Every reader holds its own buffer, so how many read at once is capped: 1 GiB of buffers, or an eighth of what
    // the host has free, whichever is less (a 128-thread machine used to reserve 4 GiB here while the model held the
    // device).  The readers that do not fit go in the next group inside the same timed span, so a buffer keeps its
    // 32 MiB - far above the caches a streaming read could otherwise be served from - and the reading is still one
    // reader per hardware thread.  The i7-14700 (28 threads, 896 MiB) runs as one group, as it did.
    uint64_t free_b = 0, cached_b = 0;
    size_t budget = (size_t) 1024 << 20;
    if (strata::platform::free_and_cached(free_b, cached_b)) budget = std::min(budget, (free_b + cached_b) / 8);
    const int group = std::max(1, std::min(nt, (int) (budget / per)));
    std::vector<std::vector<uint64_t>> bufs((size_t) group);
    for (auto& b : bufs) b.assign(per / 8, 1);   // touched here, so the timed part reads only pages that are there
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        std::atomic<uint64_t> sink{0};
        const auto t0 = std::chrono::steady_clock::now();
        for (int g0 = 0; g0 < nt; g0 += group) {
            const int n = std::min(group, nt - g0);
            std::vector<std::thread> th;
            th.reserve((size_t) n);
            for (int t = 0; t < n; ++t)
                th.emplace_back([&, t] {
                    uint64_t acc = 0;
                    for (const uint64_t v : bufs[(size_t) t]) acc += v;
                    sink += acc;
                });
            for (auto& t : th) t.join();
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = std::max(best, (double) nt * (double) per / std::max(ms, 1e-3));
    }
    return best;
}

MissRate split_miss_rate(const ExpertFacts& facts, const PoolShape& pool, double ram_rate) {
    MissRate out;
    out.rate = ram_rate;
    if (facts.layers < 1 || facts.n_expert < 1) out.why = "the model has no experts";
    else if (!facts.pool_rows) out.why = "its missed experts are not the CPU pool's native rows";
    else if (strata::env_get("SPLIT_CPU_GBPS") != nullptr) out.why = "XESTRATA_SPLIT_CPU_GBPS sets the rate";
    if (!out.why.empty()) return out;
    Runtime::wait_preload_all();
    strata::kernels::cpu::ExpertPool cal(pool.workers, /*pin=*/true, pool.host_works, /*first_core=*/0, pool.tasks);
    double c = 0.0, bytes = 0.0;
    if (!measure_cpu_expert_ms_unloaded(cal, facts.layer, c, bytes, out.why)) return out;
    out.expert_ms = c;
    out.expert_rate = bytes / c;
    out.rate = std::min(ram_rate, out.expert_rate);
    return out;
}

// The time of one small host->device copy (64 KiB, the median of 16 after a warmup): the fixed part of a layer
// split's hand-off on this GPU's link, for --layer-split auto.  < 0 when the probe cannot run.
double probe_copy_latency_ms() {
    constexpr size_t kBytes = (size_t) 64 << 10;
    void* h = nullptr;
    void* d = nullptr;
    if (!strata::gpu::alloc_host(&h, kBytes)) return -1.0;
    if (!strata::gpu::alloc_device(&d, kBytes)) {
        strata::gpu::free(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);
    std::vector<double> ms;
    for (int i = 0; i < 17; ++i) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        if (!strata::gpu::copy(d, h, kBytes)) break;
        if (i > 0) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    strata::gpu::free(d);
    strata::gpu::free(h);
    if (ms.empty()) return -1.0;
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

} // namespace strata::core
