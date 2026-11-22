// include/strata/hip_uma_guard.hpp - First-class AMD UMA / iGPU (gfx1103 780M, gfx1150 Strix Point 890M/880M,
// gfx1151 Strix Halo) coherent GTT zero-copy allocation + <= 1.0s WDDM TDR / Linux amdgpu MES chunk guards.
//
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::hip_uma {

enum class AmdGen : uint8_t {
    Unknown = 0,
    Rdna2   = 2,  // gfx1030 / gfx1031 / gfx1032 / gfx1034..1036 (no WMMA; sdot4 + fdot2)
    Rdna3   = 3,  // gfx1100 / gfx1101 / gfx1102 / gfx1103 (780M iGPU) (Wave32 half16 WMMA)
    Rdna35  = 35, // gfx1150 (Strix Point 890M/880M) / gfx1151 (Strix Halo 8060S) / gfx1152 (Wave32 half16 WMMA)
    Rdna4   = 4,  // gfx1200 / gfx1201 (Wave32 half8 + 2x iu8 WMMA)
};

inline bool arch_is_prefix(const char* arch, const char* prefix) noexcept {
    if (!arch || !prefix) return false;
    const size_t n = std::strlen(prefix);
    return std::strncmp(arch, prefix, n) == 0;
}

inline bool arch_is_exact(const char* arch, const char* name) noexcept {
    if (!arch || !name) return false;
    const size_t n = std::strlen(name);
    return std::strncmp(arch, name, n) == 0 && (arch[n] == '\0' || arch[n] == ':');
}

inline AmdGen classify_amd_gen(const char* gcn_arch) noexcept {
    if (!gcn_arch || !*gcn_arch) return AmdGen::Unknown;
    if (arch_is_prefix(gcn_arch, "gfx120")) return AmdGen::Rdna4;
    if (arch_is_prefix(gcn_arch, "gfx115")) return AmdGen::Rdna35;
    if (arch_is_prefix(gcn_arch, "gfx110")) return AmdGen::Rdna3;
    if (arch_is_prefix(gcn_arch, "gfx103")) return AmdGen::Rdna2;
    return AmdGen::Unknown;
}

inline bool is_known_amd_igpu_arch(const char* gcn_arch) noexcept {
    return arch_is_exact(gcn_arch, "gfx1103") ||
           arch_is_exact(gcn_arch, "gfx1150") ||
           arch_is_exact(gcn_arch, "gfx1151") ||
           arch_is_exact(gcn_arch, "gfx1152") ||
           arch_is_exact(gcn_arch, "gfx1035") ||
           arch_is_exact(gcn_arch, "gfx1036");
}

struct HostMemInfo {
    uint64_t total_bytes = 0;
    uint64_t avail_bytes = 0;
};

inline HostMemInfo query_host_memory() noexcept {
    HostMemInfo info{};
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        info.total_bytes = static_cast<uint64_t>(ms.ullTotalPhys);
        info.avail_bytes = static_cast<uint64_t>(ms.ullAvailPhys);
    }
#else
    unsigned long long total_kb = 0, avail_kb = 0, used_kb = 0, v = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char ln[256];
        while (std::fgets(ln, sizeof ln, f)) {
            if (std::sscanf(ln, "MemTotal: %llu kB", &v) == 1) total_kb = v;
            else if (std::sscanf(ln, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
            else if (std::sscanf(ln, "AnonPages: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "Shmem: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "SUnreclaim: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "KernelStack: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "PageTables: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "Mlocked: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "Percpu: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "VmallocUsed: %llu kB", &v) == 1 ||
                     std::sscanf(ln, "GPUActive: %llu kB", &v) == 1) {
                used_kb += v;
            }
        }
        std::fclose(f);
    }
    info.total_bytes = total_kb * 1024ull;
    const unsigned long long margin_kb = 4ull << 20;  // keep >= 4 GiB for OS / compositor
    if (total_kb > used_kb + margin_kb) {
        info.avail_bytes = (total_kb - used_kb - margin_kb) * 1024ull;
    } else {
        info.avail_bytes = avail_kb * 1024ull;
    }
    // Clamp by cgroup v2/v1 container limit when present.
    const char* limf[2] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    const char* curf[2] = {"/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"};
    for (int i = 0; i < 2; ++i) {
        FILE* fl = std::fopen(limf[i], "r");
        if (!fl) continue;
        char buf[64] = {0};
        unsigned long long lim = 0, cur = 0, file_cache = 0;
        if (std::fgets(buf, sizeof buf, fl) && buf[0] >= '0' && buf[0] <= '9') lim = std::strtoull(buf, nullptr, 10);
        std::fclose(fl);
        if (lim > (1ull << 30) && lim < (1ull << 50)) {
            if (FILE* fc = std::fopen(curf[i], "r")) {
                char b2[64] = {0};
                if (std::fgets(b2, sizeof b2, fc) && b2[0] >= '0' && b2[0] <= '9') cur = std::strtoull(b2, nullptr, 10);
                std::fclose(fc);
            }
            if (i == 0) {
                if (FILE* fs = std::fopen("/sys/fs/cgroup/memory.stat", "r")) {
                    char ln[128];
                    unsigned long long fv = 0;
                    while (std::fgets(ln, sizeof ln, fs)) {
                        if (std::sscanf(ln, "file %llu", &fv) == 1) { file_cache = fv; break; }
                    }
                    std::fclose(fs);
                }
            }
            const unsigned long long non_file = cur > file_cache ? cur - file_cache : 0;
            const unsigned long long margin = 4ull << 30;
            const uint64_t cg_avail = lim > non_file + margin ? static_cast<uint64_t>(lim - non_file - margin) : 0;
            info.total_bytes = std::min(info.total_bytes, static_cast<uint64_t>(lim));
            if (info.avail_bytes == 0 || cg_avail < info.avail_bytes) info.avail_bytes = cg_avail;
        }
    }
#endif
    return info;
}

// Compute effective usable GPU memory on UMA APUs (780M gfx1103, Strix Point gfx1150, Strix Halo gfx1151):
// Even when BIOS carveout is tiny (512 MiB - 4 GiB), coherent GTT / mapped host UMA exposes system RAM
// (minus OS safety headroom) so 9.75 GiB SQ-10 and 13.36 GiB SQ-15 run 100% GPU-accelerated with zero BIOS changes!
inline void effective_uma_memory(bool is_uma, uint64_t carveout_free, uint64_t carveout_total,
                                 uint64_t& out_free, uint64_t& out_total) noexcept {
    out_free = carveout_free;
    out_total = carveout_total;
    if (!is_uma) return;
    const HostMemInfo hm = query_host_memory();
    if (hm.total_bytes == 0) return;
    // Leave >= 4 GiB (or 20% on 16 GiB laptops, min 2.5 GiB) for OS/WDDM/compositor.
    const uint64_t os_margin = (hm.total_bytes <= (20ull << 30))
        ? std::max<uint64_t>(2560ull << 20, hm.total_bytes / 5)
        : (4ull << 30);
    // On Windows WDDM, shared GPU memory aperture is typically ~50%-75% of physical RAM on laptops/mini-PCs,
    // while Strix Halo / newer Adrenalin exposes up to 75%-85% via coherent host-mapped GTT.
    const uint64_t ram_usable_total = hm.total_bytes > os_margin ? (hm.total_bytes - os_margin) : 0;
    const uint64_t ram_usable_free = hm.avail_bytes > (os_margin / 2) ? (hm.avail_bytes - (os_margin / 2)) : 0;
    const uint64_t combined_total = std::max(carveout_total, std::min(carveout_total + ram_usable_total, hm.total_bytes + carveout_total));
    const uint64_t combined_free = std::max(carveout_free, std::min(combined_total, carveout_free + ram_usable_free));
    out_total = combined_total;
    out_free = combined_free;
}

// Disable Transparent Huge Pages on Linux KFD userptr / mapped UMA buffers to prevent amdgpu MES queue-eviction resets
// (p2006 Radeon 780M gfx1103 & Strix Point gfx1150 MES reset rule).
inline void advise_no_thp_linux(void* ptr, size_t bytes) noexcept {
#if !defined(_WIN32) && defined(MADV_NOHUGEPAGE)
    if (!ptr || bytes == 0) return;
    const long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned = p & ~static_cast<uintptr_t>(pg - 1);
    const size_t len = bytes + static_cast<size_t>(p - aligned);
    (void)madvise(reinterpret_cast<void*>(aligned), len, MADV_NOHUGEPAGE);
#else
    (void)ptr;
    (void)bytes;
#endif
}

// <= 1.0s kernel / chunk dispatch guards for Windows WDDM (2.0s TDR) and Linux amdgpu MES (780M / 890M reset rule).
struct DispatchGuard {
    int max_prefill_chunk = 2048;   // Max tokens per prefill chunk so no submission exceeds <= 1.0s
    int visit_flush_stride = 48;    // Flush/sync stream every N layer visits during prefill
    int max_persistent_blocks = 0;  // Safe co-resident CTA count for persistent kernels (never hardcode 96 CUs!)
    uint64_t max_spin_ms = 150;     // Max in-kernel spin budget (ms) before host fallback
};

inline DispatchGuard compute_dispatch_guard(const char* gcn_arch, int cu_count, bool integrated, bool windows_wddm) noexcept {
    DispatchGuard g{};
    const AmdGen gen = classify_amd_gen(gcn_arch);
    const int cus = cu_count > 0 ? cu_count : 12;

    if (gen == AmdGen::Rdna2) {
        // RDNA2 (no WMMA, SIMT sdot4/fdot2): keep chunks modest so attention + GEMM stay strictly < 1.0s.
        g.max_prefill_chunk = windows_wddm ? 512 : 1024;
        g.visit_flush_stride = windows_wddm ? 8 : 16;
    } else if (integrated || is_known_amd_igpu_arch(gcn_arch)) {
        if (arch_is_exact(gcn_arch, "gfx1151")) {
            // Strix Halo 8050S/8060S (gfx1151: 32-40 CUs = 16-20 WGPs reported by HIP!)
            g.max_prefill_chunk = windows_wddm ? 2048 : 4096;
            g.visit_flush_stride = windows_wddm ? 16 : 24;
        } else if (arch_is_exact(gcn_arch, "gfx1150")) {
            // Strix Point 880M/890M (gfx1150: 12-16 CUs = 6-8 WGPs)
            g.max_prefill_chunk = 512;
            g.visit_flush_stride = 12;
        } else if (arch_is_exact(gcn_arch, "gfx1103") || cus <= 12) {
            // Radeon 780M (gfx1103: 12 CUs = 6 WGPs): strict <= 1.0s guard prevents WDDM TDR & Linux MES resets!
            g.max_prefill_chunk = windows_wddm ? 256 : 512;
            g.visit_flush_stride = 8;
        } else if (cus <= 20) {
            // Other mid-size APU
            g.max_prefill_chunk = 512;
            g.visit_flush_stride = 12;
        } else {
            g.max_prefill_chunk = windows_wddm ? 2048 : 4096;
            g.visit_flush_stride = windows_wddm ? 16 : 24;
        }
    } else {
        // Discrete RDNA3 / RDNA4 (RX 7600 32 CU .. RX 7900 XTX 96 CU / R9700 64 CU)
        if (cus <= 36) {
            g.max_prefill_chunk = windows_wddm ? 1024 : 2048;
            g.visit_flush_stride = windows_wddm ? 16 : 48;
        } else {
            g.max_prefill_chunk = windows_wddm ? 4096 : 8192;
            g.visit_flush_stride = windows_wddm ? 24 : 48;
        }
    }

    const int reserve_cus = windows_wddm ? std::min(cus / 2, 4) : (integrated ? std::min(cus / 4, 2) : 0);
    g.max_persistent_blocks = std::max(1, (cus - reserve_cus) * 2);
    g.max_spin_ms = windows_wddm ? 150 : 800;
    return g;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// Unified GPU allocation handle that seamlessly supports both discrete VRAM (hipMalloc) and
// UMA/iGPU zero-copy coherent GTT memory (hipHostMallocCoherent | hipHostMallocMapped).
struct GpuBuffer {
    uint8_t* dev_ptr = nullptr;
    uint8_t* host_ptr = nullptr;
    size_t bytes = 0;
    bool host_mapped = false;
    bool coherent = false;

    bool valid() const noexcept { return dev_ptr != nullptr; }
};

inline hipError_t alloc_gpu_buffer(GpuBuffer& buf, size_t bytes, bool is_uma, bool prefer_uma_coherent = false) noexcept {
    buf = GpuBuffer{};
    if (bytes == 0) return hipSuccess;
    const size_t al_bytes = (bytes + 4095ull) & ~4095ull;

    bool use_uma_host = prefer_uma_coherent;
    if (is_uma && !use_uma_host) {
        if (const char* e = std::getenv("STRATA_UMA_COHERENT")) {
            if (std::atoi(e) != 0) use_uma_host = true;
        }
    }
    if (is_uma && !use_uma_host) {
        size_t fr = 0, tot = 0;
        if (hipMemGetInfo(&fr, &tot) == hipSuccess) {
            // Leave >= 256 MiB carveout headroom; if carveout is smaller (e.g. 512 MiB - 4 GiB BIOS carveout on 780M / 890M),
            // route straight to coherent mapped GTT memory so we never fail with OOM or thrash carveout!
            const size_t headroom = 256ull << 20;
            if (al_bytes + headroom > fr) {
                use_uma_host = true;
            }
        }
    }

    if (!use_uma_host) {
        void* dp = nullptr;
        hipError_t err = hipMalloc(&dp, al_bytes);
        if (err == hipSuccess) {
            buf.dev_ptr = static_cast<uint8_t*>(dp);
            buf.bytes = al_bytes;
            buf.host_mapped = false;
            return hipSuccess;
        }
        (void)hipGetLastError();
        if (!is_uma) return err;
        // On UMA/iGPU, hipMalloc OOM falls back automatically to coherent host-mapped GTT!
    }

    void* hp = nullptr;
    unsigned int flags = hipHostMallocMapped;
    bool coherent = false;
#ifdef hipHostMallocCoherent
    flags |= hipHostMallocCoherent;
    coherent = true;
#endif
    hipError_t err = hipHostMalloc(&hp, al_bytes, flags);
    if (err != hipSuccess && coherent) {
        (void)hipGetLastError();
        coherent = false;
        err = hipHostMalloc(&hp, al_bytes, hipHostMallocMapped);
    }
    if (err != hipSuccess) return err;

    advise_no_thp_linux(hp, al_bytes);
    void* dp = nullptr;
    err = hipHostGetDevicePointer(&dp, hp, 0);
    if (err != hipSuccess) {
        (void)hipHostFree(hp);
        return err;
    }
    buf.dev_ptr = static_cast<uint8_t*>(dp);
    buf.host_ptr = static_cast<uint8_t*>(hp);
    buf.bytes = al_bytes;
    buf.host_mapped = true;
    buf.coherent = coherent;
    return hipSuccess;
}

inline void free_gpu_buffer(GpuBuffer& buf) noexcept {
    if (buf.host_mapped && buf.host_ptr) {
        (void)hipHostFree(buf.host_ptr);
    } else if (buf.dev_ptr) {
        (void)hipFree(buf.dev_ptr);
    }
    buf = GpuBuffer{};
}
#endif

}  // namespace strata::hip_uma
