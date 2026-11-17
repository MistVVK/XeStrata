// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/device_main.cpp - `strata-device`: report the GPU and exercise the arena.
//
// On its own so it can run without the model.  It is also the run-time half of the device policy: the runtime
// selects one GPU by what it reports (STRATA_GPU_PCI, else a discrete card with the most memory) and refuses one without
// what the kernels need.
#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// A string as a JSON string literal.
static std::string json_text(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if ((unsigned char) c < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

// --list-devices: every GPU the engine can drive, one JSON object a line, device 0 (the one it would take) first: what
// setup reads where it has no sysfs to read (Windows).  `free_bytes` is null when the driver does not report it.
static int list_devices(bool selectors) {
    if (selectors) {
        for (const char* name : {"ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "SYCL_DEVICE_ALLOWLIST",
                                 "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                 "GPU_DEVICE_ORDINAL", "ZE_AFFINITY_MASK"}) {
            if (std::getenv(name) != nullptr) {
                std::fprintf(stderr, "cannot list selector indices with %s set\n", name);
                return 2;
            }
        }
    }
    const int n = strata::core::Runtime::count();
    for (int i = 0; i < n; ++i) {
        const strata::core::DeviceInfo d = strata::core::device_info(i);
        std::printf("{\"ordinal\": %d, \"name\": %s, \"backend\": %s, \"vendor_id\": %u, \"device_id\": %u, "
                    "\"pci\": %s, \"total_bytes\": %llu, \"free_bytes\": ", d.ordinal, json_text(d.name).c_str(),
                    json_text(d.backend).c_str(), d.vendor_id, d.device_id, json_text(d.pci).c_str(),
                    (unsigned long long) d.total_bytes);
        if (d.free_bytes_known) std::printf("%llu", (unsigned long long) d.free_bytes);
        else std::printf("null");
        std::printf(", \"integrated\": %s, \"compute_units\": %u, \"arch\": %s", d.integrated ? "true" : "false",
                    d.compute_units, json_text(d.arch).c_str());
        if (selectors) std::printf(", \"selector_index\": %d", strata::core::device_selector_index(i));
        std::puts("}");
    }
    return 0;
}

// --bench: the measured values the training placement reads (serve/train_budget.py): the pinned host <-> device
// copy bandwidth and a compute probe, one JSON line.  Run once per card (STRATA_GPU_PCI selects it).  The copy
// numbers depend on the PCIe generation and width and the pinned memory (64 MiB each way, 8 passes after 2 warm-up
// passes, the whole batch timed after one sync); the compute probe is a synthetic FP32 FMA chain (8 independent
// accumulators, 256 rounds, 1M work-items, 4 launches) - a ranking proxy for the card's arithmetic, not a model
// kernel's time.  Where the pinned memory or the queue cannot be had it exits non-zero: the caller falls back to
// the PCIe report and decides conservatively (no fast path assumed).
static int bench() {
    namespace gpu = strata::gpu;
    strata::core::Runtime& rt = strata::core::Runtime::get();
    void* s = rt.create_stream();
    size_t fb = 0, tb = 0;
    const bool known = gpu::mem_info(&fb, &tb);
    const size_t mb = 64ull << 20;
    void* h = gpu::alloc_host(mb);
    void* d = gpu::alloc_device(mb);
    if (h == nullptr || d == nullptr) {
        std::fprintf(stderr, "bench: the pinned or device memory was not had: %s\n", gpu::last_error());
        return 1;
    }
    std::memset(h, 0x5a, mb);
    auto gbs = [&](bool h2d) {
        for (int i = 0; i < 2; ++i) {
            if (!gpu::copy_async(h2d ? d : h, h2d ? h : d, mb, s)) throw std::runtime_error(gpu::last_error());
            gpu::stream_sync(s);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 8; ++i)
            if (!gpu::copy_async(h2d ? d : h, h2d ? h : d, mb, s)) throw std::runtime_error(gpu::last_error());
        gpu::stream_sync(s);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return 8.0 * (double) mb / 1e9 / sec;
    };
    const double h2d = gbs(true), d2h = gbs(false);
    gpu::free(d);
    gpu::free(h);
    // the compute probe: buf[i] through 8 dependent FMA chains, the sum written back (nothing optimized away)
    const int n = 1 << 20;
    float* buf = static_cast<float*>(gpu::alloc_device((size_t) n * 4));
    if (buf == nullptr) {
        std::fprintf(stderr, "bench: the probe's memory was not had: %s\n", gpu::last_error());
        return 1;
    }
    gpu::memset(buf, 0, (size_t) n * 4);
    sycl::queue& q = rt.compute();
    auto launch = [&] {
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            float a0 = buf[id[0]], a1 = a0 + 1.f, a2 = a0 + 2.f, a3 = a0 + 3.f;
            float a4 = a0 + 4.f, a5 = a0 + 5.f, a6 = a0 + 6.f, a7 = a0 + 7.f;
            for (int k = 0; k < 256; ++k) {
                a0 = a0 * 1.0000001f + 0.1f;
                a1 = a1 * 1.0000001f + 0.1f;
                a2 = a2 * 1.0000001f + 0.1f;
                a3 = a3 * 1.0000001f + 0.1f;
                a4 = a4 * 1.0000001f + 0.1f;
                a5 = a5 * 1.0000001f + 0.1f;
                a6 = a6 * 1.0000001f + 0.1f;
                a7 = a7 * 1.0000001f + 0.1f;
            }
            buf[id[0]] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        });
    };
    launch();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; ++i) launch();
    q.wait();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gflops = 4.0 * (double) n * 256 * 8 * 2 / 1e9 / sec;
    gpu::free(buf);
    const strata::core::DeviceInfo di = strata::core::device_info(0);
    std::printf("{\"pci\": %s, \"h2d_gbs\": %.2f, \"d2h_gbs\": %.2f, \"fp32_gflops\": %.0f, \"free_mib\": %llu, "
                "\"total_mib\": %llu}\n",
                json_text(di.pci).c_str(), h2d, d2h, gflops, (unsigned long long) (known ? fb >> 20 : 0),
                (unsigned long long) (known ? tb >> 20 : 0));
    return 0;
}

// --cpu: the CPU's name and the engine's own tests for its kernels (cpu_avx2_ok, cpu_avx512_ok), as one JSON object;
// no GPU is touched.  setup reads it where it has no /proc/cpuinfo (Windows).
static int cpu_report() {
    namespace cpu = strata::kernels::cpu;
    std::printf("{\"name\": %s, \"avx2\": %s, \"avx512\": %s}\n", json_text(cpu::cpu_name()).c_str(),
                cpu::cpu_avx2_ok() ? "true" : "false", cpu::cpu_avx512_ok() ? "true" : "false");
    return 0;
}

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

int main(int argc, char** argv) {
    bool selftest = false, list = false, selectors = false, do_bench = false;
    const char* version = "XeStrata " STRATA_VERSION " (backend " STRATA_BACKEND
        ", upstream Strata " STRATA_UPSTREAM_VERSION ", " STRATA_UPSTREAM_COMMIT ")";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--list-devices") == 0) list = true;
        else if (std::strcmp(argv[i], "--list-device-selectors") == 0) list = selectors = true;
        else if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--cpu") == 0) return cpu_report();
        else if (std::strcmp(argv[i], "--version") == 0) { std::puts(version); return 0; }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: xestrata-device [--selftest] [--list-devices] [--list-device-selectors] [--bench] [--cpu] [--version]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    try {
        if (list) return list_devices(selectors);
        if (do_bench) return bench();
        const strata::core::DeviceInfo d = strata::core::device_info(0);
        std::puts(version);
        std::printf("device %d: %s (%04x:%04x, %s)\n", d.ordinal, d.name.c_str(), d.vendor_id, d.device_id,
                    d.backend.c_str());
        std::printf("  compute units       %u\n  integrated          %s\n", d.compute_units, d.integrated ? "yes" : "no");
        std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(),
                    d.free_bytes_known ? human(d.free_bytes).c_str() : "unavailable");
        std::printf("  driver              %s\n  platform            %s\n", d.driver_version.c_str(), d.platform_version.c_str());
        std::printf("  host/device USM     %s / %s\n  FP64                %s\n",
                    d.host_usm ? "yes" : "no", d.device_usm ? "yes" : "no", d.fp64 ? "yes" : "no");
        std::printf("  subgroup sizes     ");
        for (size_t size : d.subgroup_sizes) std::printf(" %zu", size);
        std::printf("\n  model memory plan   unavailable (no model loaded)\n");

        if (selftest) {
            // The inherited selftest covers allocation alignment and capacity refusal.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            strata::core::DeviceArena arena(bytes, 0, /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                std::fprintf(stderr, "selftest: alignment not honoured\n");
                return 1;
            }
            std::printf("\nselftest: arena %s, used %s after two 1 MiB allocations\n", human(arena.capacity()).c_str(),
                        human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                std::fprintf(stderr, "selftest: over-allocation did NOT throw\n");
                return 1;
            } catch (const strata::core::DeviceError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            std::printf("xestrata-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-device: %s\n", e.what());
        return 1;
    }
}
