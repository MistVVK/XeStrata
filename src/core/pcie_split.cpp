// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/pcie_split.cpp - the times the split of the missed experts between the CPU pool and PCIe is chosen from
// (measure_pcie_split, expert_source.hpp): the GPU's and the link's sides and the memory's rate, with the CPU's side
// from measure_cpu_expert_ms.  Moved from generate.cpp so that every engine that runs the window's experts takes them.
#include "strata/core/expert_source.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::core {
namespace {

// The PCIe side of the split of the missed experts (after Project Maya's CPU lane, https://github.com/mw00/project-maya),
// timed as the verify window reads it: fetch_blobs (the copy kernel) on 1 and 2 experts a launch, taken in turn from 32
// of the layer the CPU side was measured on (75 MB: past the GPU's cache, as a decode's misses are), 8 launches in a
// row (no host call between them, as in the window's graph), the best of 3 rounds after a warm-up.  Where the arena is
// registered for copies only, the 32 experts are copied into host USM for the timing (the arena moves there only if
// the split sends experts over PCIe).
struct PcieFetchProbe {
    static constexpr int kRing = 32;
    uint8_t* buf = nullptr;
    uint8_t* host = nullptr;
    unsigned long long* d_src = nullptr;
    int32_t* d_n = nullptr;
    uint8_t* d_dst = nullptr;
    int64_t bytes = 0;
    int ring = kRing;   // the experts the launches take from (the start's 32; a 2-expert probe, 2)
    int turn = 0;
    strata::gpu::Stream s = nullptr;
    bool open(const strata::core::ExpertDispatch& d, int64_t layer, std::string& why, int want = kRing) {
        bytes = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(layer);
        unsigned long long src[kRing] = {};
        int n = 0, aliased = 0;
        for (int64_t e = 0; e < d.n_expert && n < want; ++e) {
            if (d.host_res[(size_t) layer * (size_t) d.n_expert + (size_t) e] >= 0 || !d.src->pinned(layer, e)) continue;
            const auto& lay = strata::kernels::cpu::expert_layout();
            // A mixed expert can be shorter than the fixed-width fetch. Copy it into padded host USM.
            const uint8_t* a = lay.mixed() ? nullptr : d.src->device_alias(layer, e);
            if (a == nullptr) {
                const uint8_t* b = d.src->blob(layer, e);
                if (b == nullptr) continue;
                if (host == nullptr && !strata::gpu::alloc_host(&host, (size_t) (kRing * bytes))) {
                    why = "no host USM for the timing";
                    return false;
                }
                uint8_t* slot = host + (size_t) n * (size_t) bytes;
                const size_t actual = lay.blob_bytes(layer, e);
                std::memcpy(slot, b, actual);
                if (actual < (size_t) bytes) std::memset(slot + actual, 0, (size_t) bytes - actual);
                a = host + (size_t) n * (size_t) bytes;
            } else {
                ++aliased;
            }
            src[n++] = (unsigned long long) a;
        }
        if (n < want || n == 0 || (aliased > 0 && aliased < n)) {
            why = "no layer part with " + std::to_string(want) + " experts to time over PCIe";
            return false;
        }
        ring = n;
        if (!strata::gpu::alloc_device(&buf, (size_t) (16 + 8 * kRing + 2 * bytes + 256))) { why = "no device memory for the timing"; return false; }
        d_n = reinterpret_cast<int32_t*>(buf);
        d_src = reinterpret_cast<unsigned long long*>(buf + 16);
        d_dst = buf + (size_t) ((16 + 8 * kRing + 255) / 256) * 256;
        s = strata::gpu::stream_create();
        if (s == nullptr || !strata::gpu::copy(d_src, src, sizeof src)) { why = std::string("the timing failed: ") + strata::gpu::last_error(); return false; }
        return true;
    }
    bool launch(int count, int times) {
        if (!strata::gpu::copy(d_n, &count, sizeof count)) return false;
        for (int i = 0; i < times; ++i, ++turn)
            strata::kernels::fetch_blobs(d_src + (size_t) ((2 * turn) % ring), d_n, d_dst, bytes, 2, s);
        return strata::gpu::stream_sync(s);
    }
    // `mean`: the mean of 6 rounds instead of the best of 3 - beside a load that comes and goes the best round is one
    // the load left alone
    bool time(double& p0, double& p, std::string& why, bool mean = false) {
        double ms[2] = {};
        const int rounds = mean ? 6 : 3;
        for (int w = 0; w < 2; ++w) {
            if (!launch(w + 1, 4)) { why = std::string("the timing failed: ") + strata::gpu::last_error(); return false; }
            for (int round = 0; round < rounds; ++round) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!launch(w + 1, 8)) { why = std::string("the timing failed: ") + strata::gpu::last_error(); return false; }
                const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 8.0;
                if (mean) ms[w] += one / rounds;
                else if (round == 0 || one < ms[w]) ms[w] = one;
            }
        }
        p = ms[1] - ms[0];
        p0 = std::max(0.0, ms[0] - p);
        if (!(p > 0.0)) { why = "the PCIe times did not grow with the experts"; return false; }
        return true;
    }
    ~PcieFetchProbe() {
        if (s != nullptr) strata::gpu::stream_destroy(s);
        if (buf != nullptr) strata::gpu::free(buf);
        if (host != nullptr) strata::gpu::free(host);
    }
};

// The GPU computing experts from its own memory, timed with the window's kernel (native_expert_grouped): 1 and 8
// groups of one token, taken in turn from 16 experts of the layer copied into device memory, 8 launches in a row, the
// best of 3 rounds after a warm-up.
bool measure_gpu_expert_ms(const strata::core::ExpertDispatch& d, int64_t layer, double& q0, double& q, std::string& why) {
    namespace sk = strata::kernels;
    const auto& lay = sk::cpu::expert_layout();
    const auto& f = lay.fmt[(size_t) layer];
    const bool mixed = lay.mixed();
    const sk::NativeExpertLayout L = sk::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type);
    constexpr int kRing = 16, kCap = 8;
    const int64_t bytes = (int64_t) sk::cpu::expert_layout().blob_bytes(layer);
    const size_t scratch = mixed ? sk::bih3_r1q2_grouped_scratch_bytes(kCap, f.n_embd, f.n_ff)
                                 : sk::native_expert_scratch_bytes(kCap, f.n_ff);
    const size_t xq = (size_t) (f.n_embd / 32) * 36, out = (size_t) kCap * (size_t) f.n_embd * 4;
    const size_t off_ptr = 0, off_i32 = 256, off_x = 1024, off_out = off_x + ((xq + 255) / 256) * 256;
    const size_t off_scr = off_out + ((out + 255) / 256) * 256, off_blob = off_scr + ((scratch + 255) / 256) * 256;
    const size_t off_types = off_blob + (size_t) kRing * (size_t) bytes;
    const size_t off_ids = (off_types + (size_t) 2 * kRing + 3) & ~(size_t) 3;
    const size_t off_signs = off_ids + (size_t) 2 * kRing * sizeof(int32_t);
    uint8_t* buf = nullptr;
    if (!strata::gpu::alloc_device(&buf, off_signs + (size_t) f.n_ff * sizeof(float))) { why = "no device memory for the GPU timing"; return false; }
    bool ok = strata::gpu::memset_async(buf, 0, off_blob, nullptr) && strata::gpu::stream_sync(nullptr);
    unsigned long long ptr[2 * kRing] = {};
    uint8_t types[2 * kRing] = {};
    int32_t ids[2 * kRing] = {};
    int n = 0;
    for (int64_t e = 0; ok && e < d.n_expert && n < kRing; ++e) {
        if (d.host_res[(size_t) layer * (size_t) d.n_expert + (size_t) e] >= 0 || !d.src->pinned(layer, e)) continue;
        const uint8_t* b = d.src->blob(layer, e);
        if (b == nullptr) continue;
        uint8_t* dst = buf + off_blob + (size_t) n * (size_t) bytes;
        ok = strata::gpu::copy(dst, b, lay.blob_bytes(layer, e));
        if (mixed) {
            const auto& ef = lay.format(layer, e);
            types[(size_t) 2 * n] = (uint8_t) (ef.gu_type - 180);
            types[(size_t) 2 * n + 1] = (uint8_t) (ef.d_type - 180);
            ids[n] = ids[n + kRing] = n;
        }
        ptr[n] = ptr[n + kRing] = (unsigned long long) dst;   // twice: a launch's 8 groups read on from any start
        ++n;
    }
    if (ok && n < kRing) { strata::gpu::free(buf); why = "too few experts in RAM to time the GPU"; return false; }
    // groups g of one entry each: start[g] = g, dst = g, tok = 0
    int32_t i32[1 + (kCap + 1) + 2 * kCap] = {};
    int32_t* cnt = i32;
    int32_t* start = i32 + 1;
    int32_t* dst = start + kCap + 1;   // then kCap entries' tokens, all 0 (one token)
    for (int g = 0; g <= kCap; ++g) start[g] = g;
    for (int g = 0; g < kCap; ++g) dst[g] = g;
    ok = ok && strata::gpu::copy(buf + off_ptr, ptr, sizeof ptr);
    sk::Bih3R1q2Layer B;
    if (mixed) {
        B.n_embd = (int) f.n_embd; B.n_ff = (int) f.n_ff; B.n_expert = kRing;
        B.k_pairs = buf + off_types; B.route_ids = reinterpret_cast<const int32_t*>(buf + off_ids);
        B.had_block = f.had_d ? f.had_block : 0;
        if (f.had_d && f.d_signs) {
            ok = ok && strata::gpu::copy(buf + off_signs, f.d_signs, (size_t) f.n_ff * sizeof(float));
            B.d_signs = reinterpret_cast<const float*>(buf + off_signs);
        }
        std::memcpy(B.p2, f.p2, sizeof B.p2);
        ok = ok && strata::gpu::copy(buf + off_types, types, sizeof types) &&
                   strata::gpu::copy(buf + off_ids, ids, sizeof ids);
    }
    strata::gpu::Stream s = ok ? strata::gpu::stream_create() : nullptr;
    ok = ok && s != nullptr;
    auto* d_ptr = reinterpret_cast<unsigned long long*>(buf + off_ptr);
    auto* d_i32 = reinterpret_cast<int32_t*>(buf + off_i32);
    double ms[2] = {};
    int turn = 0;
    try {
        for (int w = 0; ok && w < 2; ++w) {
            cnt[0] = w == 0 ? 1 : kCap;
            ok = strata::gpu::copy(d_i32, i32, sizeof i32);
            auto run = [&](int times) {
                for (int i = 0; i < times; ++i, turn = (turn + kCap) % kRing) {
                    if (mixed) {
                        B.route_ids = reinterpret_cast<const int32_t*>(buf + off_ids) + turn;
                        sk::bih3_r1q2_grouped(B, d_ptr + turn, d_i32 + 1, d_i32, d_i32 + 1 + kCap + 1,
                                               d_i32 + 1 + kCap + 1 + kCap, kCap, kCap, buf + off_x,
                                               buf + off_scr, reinterpret_cast<float*>(buf + off_out), s);
                    } else {
                        sk::native_expert_grouped(L, d_ptr + turn, d_i32 + 1, d_i32, d_i32 + 1 + kCap + 1,
                                                  d_i32 + 1 + kCap + 1 + kCap, kCap, kCap, buf + off_x,
                                                  buf + off_scr, reinterpret_cast<float*>(buf + off_out), s);
                    }
                }
                return strata::gpu::stream_sync(s);
            };
            ok = ok && run(4);
            for (int round = 0; ok && round < 3; ++round) {
                const auto t0 = std::chrono::steady_clock::now();
                ok = run(8);
                const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 8.0;
                if (round == 0 || one < ms[w]) ms[w] = one;
            }
        }
    } catch (const std::exception& e) {
        why = std::string("the GPU timing failed: ") + e.what();
        ok = false;
    }
    if (s != nullptr) strata::gpu::stream_destroy(s);
    strata::gpu::free(buf);
    if (!ok) { if (why.empty()) why = std::string("the GPU timing failed: ") + strata::gpu::last_error(); return false; }
    q = (ms[1] - ms[0]) / (kCap - 1);
    q0 = std::max(0.0, ms[0] - q);
    if (!(q > 0.0)) { why = "the GPU times did not grow with the experts"; return false; }
    return true;
}

// The rate (bytes a ms) at which the CPU reads `blobs` (`bytes` each, past any CPU cache together) on `threads` of
// the pool's cores, one word a cache line, for 200 ms after 50 ms to get going; the host core is left to its thread,
// as in a decode (a thread inherits its creator's affinity, the host core alone).
double memory_read_rate(const std::vector<const uint8_t*>& blobs, size_t bytes, int threads) {
    const std::vector<int> cores = strata::kernels::cpu::physical_cores(/*skip_first=*/true);
    threads = std::max(1, std::min<int>(threads, cores.empty() ? 1 : (int) cores.size()));
    std::atomic<int> phase{0};   // 0 warming, 1 counting, 2 done
    std::atomic<uint64_t> read{0}, sink{0};
    std::vector<std::thread> pool;
    pool.reserve((size_t) threads);
    for (int w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            if (!cores.empty()) {
#if defined(_WIN32)
                const int core = cores[(size_t) w % cores.size()];
                GROUP_AFFINITY one{};
                one.Group = (WORD) (core / 64);
                one.Mask = KAFFINITY(1) << (core & 63);
                SetThreadGroupAffinity(GetCurrentThread(), &one, nullptr);
#else
                cpu_set_t one;
                CPU_ZERO(&one);
                CPU_SET(cores[(size_t) w % cores.size()], &one);
                pthread_setaffinity_np(pthread_self(), sizeof one, &one);
#endif
            }
            uint64_t x = 0, counted = 0;
            for (size_t i = (size_t) w; phase.load(std::memory_order_relaxed) < 2; i += (size_t) threads) {
                const bool counting = phase.load(std::memory_order_relaxed) == 1;
                const auto* q = reinterpret_cast<const uint64_t*>(blobs[i % blobs.size()]);
                for (size_t j = 0; j < bytes / 8; j += 8) x += q[j];
                if (counting) counted += bytes;
            }
            read += counted;
            sink += x;
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    phase = 1;
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    phase = 2;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (auto& th : pool) th.join();
    return ms > 0.0 ? (double) read.load() / ms : 0.0;
}

// The memory's rate for the split, read before a long prompt where the split is set up after it (generate): after
// a 64K prompt the reading came out up to 3x low on the RX 9060 XT's machine, the split giving the CPU every missed
// expert (decode 23 against 34-41 tok/s), where 1K prompts read it steadily.  0: not read.
double g_memory_rate_before = 0.0;

std::vector<const uint8_t*> split_blobs(const strata::core::ExpertDispatch& d, int64_t layer) {
    std::vector<const uint8_t*> blobs;
    for (int64_t e = 0; e < d.n_expert && blobs.size() < 64; ++e)
        if (d.host_res[(size_t) layer * (size_t) d.n_expert + (size_t) e] < 0 && d.src->pinned(layer, e))
            if (const uint8_t* b = d.src->blob(layer, e)) blobs.push_back(b);
    return blobs;
}

// Each read must stay in its own expert even when the pack uses variable sizes.
size_t split_read_bytes(int64_t layer) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t bytes = (size_t) lay.blob_bytes(layer);
    if (lay.mixed()) for (int64_t e = 0; e < lay.n_expert; ++e)
        bytes = std::min(bytes, (size_t) lay.blob_bytes(layer, e));
    return bytes;
}

// the fastest of five readings 100 ms apart: one beside other work (the machine's, not the engine's) read the rate up
// to 3x low on the RX 9060 XT's machine
double split_memory_rate(const strata::core::ExpertDispatch& d, const std::vector<const uint8_t*>& blobs, size_t bytes) {
    double mem = 0.0;
    for (int i = 0; i < 5 && !blobs.empty(); ++i) {
        if (i > 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        mem = std::max(mem, memory_read_rate(blobs, bytes, d.pool != nullptr ? d.pool->workers() : 1));
    }
    return mem;
}

}  // namespace

void read_memory_rate_before(const strata::core::ExpertDispatch& d, int64_t n_layers) {
    if (d.src == nullptr || d.host_res == nullptr) return;
    strata::core::Runtime::get().wait_preload();
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const std::vector<const uint8_t*> blobs = split_blobs(d, layer);
        if (blobs.size() < 16) continue;
        g_memory_rate_before = split_memory_rate(d, blobs, split_read_bytes(layer));
        return;
    }
}

bool measure_pcie_split(strata::core::ExpertDispatch& d, int64_t n_layers, strata::core::PcieSplitTimes& t, std::string& why) {
    // the kernels' preload builds on a core of its own: beside it the CPU's expert time read 2x and the memory's rate
    // up to 3x low on an RX 9060 XT with a Ryzen 7 5700X, and the split then gave the CPU every missed expert
    strata::core::Runtime::get().wait_preload();
    int64_t layer = -1;
    if (!strata::core::measure_cpu_expert_ms(d, n_layers, t.c0, t.c, layer, why)) return false;
    t.layer = layer;
    if (!measure_gpu_expert_ms(d, layer, t.q0, t.q, why)) return false;
    PcieFetchProbe probe;
    if (!probe.open(d, layer, why) || !probe.time(t.p0, t.p, why)) return false;
    // Both sides at once read the same memory.  Measured beside each other the times did not hold still (the B70's
    // fetch read as fast as alone in some starts and 8x slower in others, where the decode's took twice as long), so
    // the slowdown is the arithmetic of the memory's rate instead: the CPU reading the layer's experts on the pool's
    // cores, alone, gives the rate the memory serves (the decode's CPU experts come close to it); when the CPU's and
    // the link's rates together pass it, both slow down by the same factor.
    const auto& lay = strata::kernels::cpu::expert_layout();
    const double bytes = (double) lay.blob_bytes(layer);
    const double mem = std::max(g_memory_rate_before, split_memory_rate(d, split_blobs(d, layer), split_read_bytes(layer)));
    const double cpu_rate = bytes / t.c, link_rate = bytes / t.p;   // bytes a ms
    const double factor = mem > 0.0 ? std::max(1.0, (cpu_rate + link_rate) / mem) : 1.0;
    t.cb0 = t.c0; t.cb = t.c * factor;
    t.pb0 = t.p0; t.pb = t.p * factor;
    return true;
}

namespace {
// STRATA_PCIE_MIN1's auto decision: -1 not probed yet, 0/1 the pick (one process: the machine's answer)
std::atomic<int> g_pcie_min1{-1};
}  // namespace

bool pcie_min1_on() {
    static const int fixed = strata::env_auto("PCIE_MIN1");
    if (fixed >= 0) return fixed == 1;
    return g_pcie_min1.load(std::memory_order_relaxed) > 0;
}

bool pcie_min1_worth(ExpertDispatch& d, int64_t n_layers, std::string& why) {
    double c = 0.0, q = 0.0, q0 = 0.0, p = 0.0, p0 = 0.0;
    int64_t layer = -1;
    if (!measure_cpu_expert_one_ms(d, n_layers, c, layer, why)) return false;
    if (!measure_gpu_expert_ms(d, layer, q0, q, why)) return false;
    PcieFetchProbe probe;
    if (!probe.open(d, layer, why, 2) || !probe.time(p0, p, why, /*mean=*/true)) return false;
    PcieSplitTimes t;
    t.c0 = 0.0; t.c = c; t.cb0 = 0.0; t.cb = c;   // alone, as the CPU's one miss of two
    t.p0 = p0; t.p = p; t.pb0 = p0; t.pb = p;
    t.q0 = q0; t.q = q;
    const bool on = pcie_split_m(t, false, 2, 0) >= 1;
    std::fprintf(stderr, "xestrata: the PCIe lane with two misses a layer: one expert %.3f ms on the CPU, %.3f ms over "
                         "PCIe and %.3f ms on the GPU: %s (XESTRATA_PCIE_MIN1=0/1 picks one)\n",
                 c, p, q, on ? "one goes over PCIe" : "the CPU takes both");
    return on;
}

bool setup_pcie_split(ExpertDispatch& d, int64_t n_layers, PcieSplitTimes& t, bool overlap, const char* keep, double frac,
                      const char* who) {
    std::string why;
    if (keep == nullptr && !measure_pcie_split(d, n_layers, t, why)) keep = why.c_str();
    if (keep != nullptr) {
        std::fprintf(stderr, "%s: PCIe split: the share %.2f of the missed experts (%s)\n", who, frac, keep);
        // a fixed share rounds to 0 at one or two misses a layer: decide once whether the lane's floor pays
        if (frac > 0.0 && g_pcie_min1.load(std::memory_order_relaxed) < 0 && strata::env_auto("PCIE_MIN1") < 0) {
            why.clear();
            const bool on = pcie_min1_worth(d, n_layers, why);
            if (!on && !why.empty())
                std::fprintf(stderr, "xestrata: PCIE_MIN1: the timing did not run (%s): the CPU takes the misses\n",
                             why.c_str());
            g_pcie_min1.store(on ? 1 : 0, std::memory_order_relaxed);
        }
        return false;
    }
    d.pcie_split = &t;
    d.pcie_overlap = overlap;
    std::string row;
    for (int f = 1; f <= 16; ++f) row += " " + std::to_string(pcie_split_m(t, overlap, f, f));
    std::fprintf(stderr, "%s: PCIe split, measured: an expert %.3f ms on the CPU (%.3f beside the link), %.3f ms over "
                         "PCIe (%.3f beside the CPU), %.3f ms on the GPU, layer %lld -> of 1..16 missed experts (as "
                         "many hits) PCIe takes%s%s\n",
                 who, t.c, t.cb, t.p, t.pb, t.q, (long long) t.layer, row.c_str(), overlap ? " (beside the hits)" : "");
    return true;
}

}  // namespace strata::core
