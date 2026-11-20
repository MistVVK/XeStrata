// hy4_gpu_model.hip.h - GPU residency of a Hy4-Flash .strata: trunk (Section 1) and the expert arena (Section 4) as device buffers
#pragma once
#include "hy4_gpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hy4_io.h"
#include "hy4_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

#define HCK(x)                                                                                                        \
    do {                                                                                                              \
        hipError_t e_ = (x);                                                                                          \
        if (e_ != hipSuccess) {                                                                                       \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_), __FILE__, __LINE__);      \
            exit(3);                                                                                                  \
        }                                                                                                             \
    } while (0)

namespace hy4 {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ExpertRef {  // per (moe_layer, expert id); offsets relative to the arena base
    uint64_t body_off;
    uint32_t dn_off;  // relative to body_off
    uint8_t gu_code, dn_code;
    uint8_t tier = 0;  // 0 device arena, 1 pinned host memory read by the GPU, 2 host RAM computed by the CPU (hybrid)
};

struct GpuModel {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    uint8_t* d_trunk = nullptr;   // Section 1
    uint8_t* d_sec4 = nullptr;    // Section 4 (expert bodies), 4 KiB aligned bodies, same offsets as the file's directory
    uint64_t trunk_bytes = 0, sec4_bytes = 0;
    std::vector<ExpertRef> experts;  // [41*512] indexed by moe_layer*512 + expert_id
    std::vector<uint8_t> seed;       // Section 2 (engram hot seed), host
    // expert tiering (load_experts_tiered): per directory index (layer*512 + rank) tier 0 = device arena, 1 = pinned host memory read by the GPU
    uint8_t* h_arena = nullptr;
    uint64_t vram_expert_bytes = 0, host_expert_bytes = 0, n_vram_experts = 0, n_host_experts = 0;
    std::vector<uint8_t> expert_tier;
    std::vector<std::vector<bool>> cpu_mask;   // [41][512] by expert id: tier-2 experts (resident in host RAM, computed by hy4_cpu_experts); empty when none
    std::vector<std::vector<bool>> cold_mask;  // [41][512] by expert id: tier-3 experts (do not fit the RAM budget: read from the .strata on demand by hy4_cpu_experts); empty when none
    double t_load_trunk = 0, t_load_experts = 0;

    const uint8_t* tensor_dev(const std::string& name) const { return d_trunk + man.get(name).off; }
};

inline bool gpu_is_uma() {
    static int cached = -1;
    if (cached < 0) {
        hipDeviceProp_t dp{};
        cached = (hipGetDeviceProperties(&dp, 0) == hipSuccess && dp.integrated != 0) ? 1 : 0;
    }
    return cached != 0;
}

// Available GPU memory: on Linux UMA (e.g. NVIDIA GB10 / DGX Spark), cudaMemGetInfo reports only
// Linux MemFree and ignores both reclaimable page cache and the NVIDIA kernel driver's cached UMA pool
// (which also does not appear in /proc/meminfo MemAvailable). Estimate allocatable RAM from MemTotal
// minus non-reclaimable OS/process memory, bounded by the container cgroup (excluding file cache).
inline void gpu_mem_avail(size_t& fr, size_t& tot) {
    HCK(hipMemGetInfo(&fr, &tot));
#if !defined(_WIN32)
    if (gpu_is_uma()) {
        size_t avail = 0;
        if (FILE* f = fopen("/proc/meminfo", "r")) {
            char ln[256];
            unsigned long long total_kb = 0, avail_kb = 0, used_kb = 0, v = 0;
            while (fgets(ln, sizeof ln, f)) {
                if (sscanf(ln, "MemTotal: %llu kB", &v) == 1) total_kb = v;
                else if (sscanf(ln, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
                else if (sscanf(ln, "AnonPages: %llu kB", &v) == 1 ||
                         sscanf(ln, "Shmem: %llu kB", &v) == 1 ||
                         sscanf(ln, "SUnreclaim: %llu kB", &v) == 1 ||
                         sscanf(ln, "KernelStack: %llu kB", &v) == 1 ||
                         sscanf(ln, "PageTables: %llu kB", &v) == 1 ||
                         sscanf(ln, "Mlocked: %llu kB", &v) == 1 ||
                         sscanf(ln, "Percpu: %llu kB", &v) == 1 ||
                         sscanf(ln, "VmallocUsed: %llu kB", &v) == 1 ||
                         sscanf(ln, "GPUActive: %llu kB", &v) == 1)
                    used_kb += v;
            }
            fclose(f);
            const unsigned long long margin_kb = 4ull << 20;  // 4 GiB OS/driver headroom
            if (total_kb > used_kb + margin_kb) avail = (size_t)(total_kb - used_kb - margin_kb) * 1024ull;
            else avail = (size_t)avail_kb * 1024ull;
        }
        const char* limf[2] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
        const char* curf[2] = {"/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"};
        for (int i = 0; i < 2; ++i) {
            FILE* fl = fopen(limf[i], "r");
            if (!fl) continue;
            char buf[64] = {0};
            unsigned long long lim = 0, cur = 0, file_cache = 0;
            if (fgets(buf, sizeof buf, fl) && buf[0] >= '0' && buf[0] <= '9') lim = strtoull(buf, nullptr, 10);
            fclose(fl);
            if (lim > (1ull << 30) && lim < (1ull << 50)) {
                if (FILE* fc = fopen(curf[i], "r")) {
                    char b2[64] = {0};
                    if (fgets(b2, sizeof b2, fc) && b2[0] >= '0' && b2[0] <= '9') cur = strtoull(b2, nullptr, 10);
                    fclose(fc);
                }
                if (i == 0) {
                    if (FILE* fs = fopen("/sys/fs/cgroup/memory.stat", "r")) {
                        char ln[128];
                        unsigned long long fv = 0;
                        while (fgets(ln, sizeof ln, fs))
                            if (sscanf(ln, "file %llu", &fv) == 1) { file_cache = fv; break; }
                        fclose(fs);
                    }
                }
                const unsigned long long non_file = cur > file_cache ? cur - file_cache : 0;
                const unsigned long long margin = 4ull << 30;
                const size_t cg = lim > non_file + margin ? (size_t)(lim - non_file - margin) : 0;
                if (avail == 0 || cg < avail) avail = cg;
            }
        }
        if (avail > fr && avail <= tot) fr = avail;
    }
#endif
}

// Parallel direct read into a host-accessible buffer (used for UMA zero-copy mapped load).
inline void read_file_parallel(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    const size_t CH = 64ull << 20;
    const uint64_t nchunks = (bytes + CH - 1) / CH;
    const int NT = (int)std::min<uint64_t>(4, std::max<uint64_t>(1, nchunks));
    std::atomic<uint64_t> next{0}, done_chunks{0};
    std::atomic<int> failed{0};
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < NT; ++t) {
        th.emplace_back([&]() {
            AlignedFile f;
            if (!f.open(path, true)) { failed = 1; return; }
            for (;;) {
                uint64_t c = next.fetch_add(1);
                if (c >= nchunks || failed.load()) break;
                size_t want = (size_t)std::min<uint64_t>(CH, bytes - c * CH);
                size_t rd_len = (want + 4095) & ~(size_t)4095;
                size_t got = f.read_at(file_off + c * CH, dst + c * CH, rd_len);
                if (got < want) {
                    fprintf(stderr, "short read in %s chunk %llu (%zu < %zu)\n", what, (unsigned long long)c, got, want);
                    failed = 1;
                    break;
                }
                uint64_t d = done_chunks.fetch_add(1) + 1;
                if (nchunks > 16 && (d * 10 / nchunks) != ((d - 1) * 10 / nchunks)) {
                    double dt = now_s() - t0;
                    fprintf(stderr, "[load %s] %llu%% %.1f s %.2f GB/s\n", what, (unsigned long long)(d * 10 / nchunks * 10), dt, (double)d * CH / std::max(dt, 1e-6) / 1e9);
                }
            }
        });
    }
    for (auto& t : th) t.join();
    if (failed.load()) exit(1);
    *seconds = now_s() - t0;
}

// Streams [file_off, file_off+bytes) into device memory dst with parallel unbuffered readers and a ring of pinned staging buffers.
inline void stream_to_device(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    const size_t CH = 64ull << 20;
    const int NB = 4;
    std::vector<void*> buf(NB);
    std::vector<hipEvent_t> ev(NB);
    for (int i = 0; i < NB; ++i) {
        HCK(hipHostMalloc(&buf[i], CH, hipHostMallocDefault));
        HCK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming));
    }
    uint64_t nchunks = (bytes + CH - 1) / CH;
    const int NRD = (int)std::min<uint64_t>(4, std::max<uint64_t>(1, nchunks));
    std::vector<std::atomic<int>> slot_state(NB);  // 0 = free, 1 = filling, 2 = ready for H2D
    for (int i = 0; i < NB; ++i) slot_state[i].store(0, std::memory_order_relaxed);
    std::vector<uint64_t> slot_chunk(NB, 0);
    std::vector<size_t> slot_len(NB, 0);
    std::atomic<uint64_t> next_chunk{0};
    std::atomic<int> failed{0};
    std::mutex qm;
    double t0 = now_s();
    std::vector<std::thread> readers;
    for (int rt = 0; rt < NRD; ++rt) {
        readers.emplace_back([&]() {
            AlignedFile f;
            if (!f.open(path, true)) { failed = 1; return; }
            for (;;) {
                uint64_t c = 0;
                int s = -1;
                {
                    std::lock_guard<std::mutex> lk(qm);
                    if (next_chunk.load() >= nchunks || failed.load()) return;
                    for (int i = 0; i < NB; ++i) {
                        if (slot_state[i].load(std::memory_order_acquire) == 0) {
                            s = i;
                            slot_state[i].store(1, std::memory_order_relaxed);
                            c = next_chunk.fetch_add(1);
                            break;
                        }
                    }
                }
                if (s < 0) { std::this_thread::sleep_for(std::chrono::microseconds(100)); continue; }
                size_t want = (size_t)std::min<uint64_t>(CH, bytes - c * CH);
                size_t rd_len = (want + 4095) & ~(size_t)4095;
                size_t got = f.read_at(file_off + c * CH, buf[s], rd_len);
                if (got < want) {
                    fprintf(stderr, "short read in %s chunk %llu (%zu < %zu)\n", what, (unsigned long long)c, got, want);
                    failed = 1;
                    slot_state[s].store(0, std::memory_order_release);
                    return;
                }
                slot_chunk[s] = c;
                slot_len[s] = want;
                slot_state[s].store(2, std::memory_order_release);
            }
        });
    }
    hipStream_t st;
    HCK(hipStreamCreate(&st));
    uint64_t copied = 0, last_print = 0;
    while (copied < nchunks && !failed.load()) {
        bool did = false;
        for (int s = 0; s < NB; ++s) {
            if (slot_state[s].load(std::memory_order_acquire) == 2) {
                uint64_t c = slot_chunk[s];
                HCK(hipMemcpyAsync(dst + c * CH, buf[s], slot_len[s], hipMemcpyHostToDevice, st));
                HCK(hipEventRecord(ev[s], st));
                HCK(hipEventSynchronize(ev[s]));
                slot_state[s].store(0, std::memory_order_release);
                ++copied;
                did = true;
                if (nchunks > 16 && copied * 10 / nchunks != last_print) {
                    last_print = copied * 10 / nchunks;
                    double dt = now_s() - t0;
                    fprintf(stderr, "[load %s] %llu%% %.1f s %.2f GB/s\n", what, (unsigned long long)(last_print * 10), dt, (double)copied * CH / std::max(dt, 1e-6) / 1e9);
                }
            }
        }
        if (!did) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    for (auto& t : readers) t.join();
    if (failed.load()) exit(1);
    HCK(hipStreamSynchronize(st));
    HCK(hipStreamDestroy(st));
    for (int i = 0; i < NB; ++i) {
        HCK(hipEventDestroy(ev[i]));
        HCK(hipHostFree(buf[i]));
    }
    *seconds = now_s() - t0;
}

// the CPU expert kernel needs AVX-512 VBMI + VNNI on x86-64 or NEON dotprod on ARM64; without it the GPU reads host-resident experts directly (zero-copy) when they fit the RAM
inline bool host_cpu_fast() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vnni");
#else
    return true;
#endif
#elif defined(__aarch64__)
#if defined(__linux__)
    return (getauxval(AT_HWCAP) & (1ul << 20)) != 0;  // HWCAP_ASIMDDP
#else
    return true;
#endif
#else
    return false;
#endif
}

// Memory plan for a context of max_ctx tokens and prefill chunks of `chunk` tokens (chunk <= 0: pick one) on a card with `total` bytes: returns the chunk and, unless the
// user set them, exports HY4_KV_HOST (1: sparse-layer KV cache, IndexPool state and the absorbed kv_b copies live in pinned host memory read by the GPU, so a 128K context
// fits 8 GB cards) and HY4_RESERVE_GB (VRAM kept free for the runner when the hot experts are placed).
inline void hy4_setenv(const char* k, const char* v, int) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

inline int plan_memory(int max_ctx, int chunk, size_t total, const std::string& model_path) {
    const double GB = 1e9, tot = (double)total / GB;
    double trunk = 5.9;
    if (!model_path.empty()) {
        strata::artifact::StrataFileReader rd;
        std::string e;
        if (rd.open(model_path, e)) trunk = (double)rd.header().sections[1].byte_length / GB;
    }
    if (chunk <= 0) {
        // ~1.4 MB of working buffers per chunk token; keep them near 12% of the card.
        // Large UMA systems (>= 90 GB, e.g. GB10 DGX Spark / Strix Halo) default to 4096 to amortize expert reads across LPDDR5x.
        int c = (gpu_is_uma() && tot >= 90.0) ? 4096 : 2048;
        while (c > 256 && c * 1.4e6 > 0.12 * (double)total) c >>= 1;
        // work_gb cards: shrink further until trunk + buffers + the IndexPool state leave room for the KV rows
        const double slots0 = (double)((max_ctx + 8 + 15) / 16 * 16);
        const double rows0 = 12.0 * slots0 * 2176.0 / GB, pivg0 = 12.0 * (double)max_ctx * 512.0 / GB + 12.0 * (double)(max_ctx / 4 + 8) * 256.0 / GB;
        while (c > 256 && trunk + c * 1.5e6 / GB + 0.4 + pivg0 + rows0 + 0.3 > tot) c >>= 1;
        chunk = c;
    }
    const double slots = (double)((max_ctx + 8 + 15) / 16 * 16);
    const double rows = 12.0 * slots * 2176.0 / GB;                                                                                              // sparse-layer KV rows + Vt
    const double pivg = 12.0 * (double)max_ctx * 512.0 / GB + 12.0 * (double)(max_ctx / 4 + 8) * 256.0 / GB;                                       // IndexPool iv/ig + pooled keys
    const double wkv = 1.06;                                                                                                                       // absorbed kv_b copies
    const double work_gb = chunk * 1.5e6 / GB + 0.4 + 30.0 * (512.0 + chunk) * 2176.0 / GB;                                                          // chunk buffers + engram/misc + SWA rings
    // The kernels read host-mapped KV rows over PCIe with work_gb gathers (~30x slower prefill at 128K), so the rows stay in VRAM whenever trunk + buffers + state + rows fit;
    // the rarely-read absorbed copies go to host first, the rows only on cards that cannot hold them (8 GB at 128K).
    int mask = 0;
    if (trunk + work_gb + pivg + rows + wkv + 0.3 > tot) mask |= 2;
    if (trunk + work_gb + pivg + rows + 0.3 > tot) mask |= 1;
    // Cards that cannot hold the whole context: keep the first dev_pos positions of the KV rows + Vt + IndexPool state in VRAM and move all of it to pinned host memory when the context
    // passes dev_pos (Runner::kv_grow), so a short chat in the 128K context runs at VRAM speed. HY4_KV_DEVPOS = dev_pos; the rows are then not forced to host.
    long dev_pos = 0;
    if ((mask & 1) && !getenv("HY4_KV_HOST")) {
        const double per_pos = 12.0 * 2176.0 + 12.0 * 512.0 + 12.0 * 64.0;  // rows + Vt, iv/ig, pooled keys per position (bytes)
        const double avail = tot - trunk - work_gb - 0.3;
        dev_pos = avail > 0 ? (long)(avail * GB / per_pos) / 1024 * 1024 : 0;
        if (dev_pos >= 4096 && dev_pos < max_ctx) {
            mask &= ~1;
            char b[16];
            snprintf(b, sizeof b, "%ld", dev_pos);
            hy4_setenv("HY4_KV_DEVPOS", b, 1);
        } else dev_pos = 0;
    }
    if (getenv("HY4_KV_HOST")) mask = atoi(getenv("HY4_KV_HOST"));
    else {
        char m[8];
        snprintf(m, sizeof m, "%d", mask);
        hy4_setenv("HY4_KV_HOST", m, 1);
    }
    {
        char b[32];
        snprintf(b, sizeof b, "%.2f", ((mask & 1) || dev_pos ? rows + (dev_pos ? pivg : 0.0) : 0.0) + ((mask & 2) ? wkv : 0.0) + 0.3);
        hy4_setenv("HY4_HOST_NEED_GB", b, 1);
    }
    if (!getenv("HY4_RESERVE_GB")) {
        double reserve = work_gb + (dev_pos ? dev_pos * (12.0 * 2176.0 + 12.0 * 512.0 + 12.0 * 64.0) / GB : pivg + ((mask & 1) ? 0.0 : rows)) + ((mask & 2) ? 0.0 : wkv) + (tot >= 10.0 && mask == 0 ? 1.45 : 0.0);  // + the prefill streaming scratch when everything else fits
        char b[32];
        snprintf(b, sizeof b, "%.2f", reserve);
        hy4_setenv("HY4_RESERVE_GB", b, 1);
    }
    fprintf(stderr, "[plan] ctx %d chunk %d: buffers ~%.2f GB, KV rows %.2f GB, IndexPool %.2f GB, kv_b copies %.2f GB, trunk %.2f GB on a %.1f GB card -> host mask %d (1 rows, 2 kv_b copies); reserve %s GB\n", max_ctx, chunk, work_gb,
            rows, pivg, wkv, trunk, tot, atoi(getenv("HY4_KV_HOST")), getenv("HY4_RESERVE_GB"));
    if (dev_pos) fprintf(stderr, "[plan] growable KV: the first %ld positions of rows + IndexPool stay in VRAM, everything moves to pinned host memory when the context passes them\n", dev_pos);
    return chunk;
}

inline double host_mem_gb();
// RAM this process may use: min(MemTotal, the container's cgroup limit), in GB
inline double ram_limit_gb() {
    double gb = host_mem_gb();
#ifndef _WIN32
    const char* files[2] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    for (const char* fn : files) {
        FILE* f = fopen(fn, "r");
        if (!f) continue;
        double v = 0;
        char buf[64] = {0};
        if (fgets(buf, sizeof buf, f) && sscanf(buf, "%lf", &v) == 1 && v > 1e9 && v < 1e15) gb = std::min(gb, v / 1e9);
        fclose(f);
    }
#endif
    return gb;
}

inline double host_mem_gb() {
#ifdef _WIN32
    return 64.0;
#else
    FILE* f = fopen("/proc/meminfo", "r");
    double kb = 64.0 * 1048576.0;
    if (f) {
        char ln[256];
        while (fgets(ln, sizeof ln, f))
            if (sscanf(ln, "MemTotal: %lf kB", &kb) == 1) break;
        fclose(f);
    }
    return kb / 1048576.0;
#endif
}

// Expert tiering for cards that cannot hold Section 4: the hottest ranks (rank-major over the 41 layers, Section 4 is stored hot to cold) go to a device arena,
// every other expert goes to pinned, device-mapped host memory that the GPU reads directly (zero-copy over PCIe). Each expert's body_off is rewritten as the
// (wrapping) distance of its final address from g.d_sec4, so the kernels keep their `sec4 + body_off` addressing and need no change.
inline bool load_experts_tiered(GpuModel& g, const std::string& path, size_t vram_budget, bool cpu_mode, std::string& err) {
    const auto& h = g.rd.header();
    const auto& dir = g.rd.expert_directory();
    const size_t N = dir.size();
    auto al = [](uint64_t x) { return (x + 4095) & ~4095ull; };
    std::vector<uint8_t> tier(N, 1);
    std::vector<uint64_t> off(N, 0);
    uint64_t vb = 0, hb = 0;
    bool full = false;
    for (size_t r = 0; r < 512 && !full; ++r)
        for (size_t L = 0; L < N / 512; ++L) {
            size_t i = L * 512 + r;
            uint64_t sz = al(dir[i].body_bytes);
            if (vb + sz > vram_budget) { full = true; break; }
            tier[i] = 0;
            off[i] = vb;
            vb += sz;
        }
    uint64_t n_v = 0, n_h = 0, n_cold = 0, cold_bytes = 0;
    for (size_t i = 0; i < N; ++i) {
        if (tier[i] == 1 && cpu_mode) tier[i] = 2;
        if (tier[i] == 1) { off[i] = hb; hb += al(dir[i].body_bytes); ++n_h; }
        else if (tier[i] == 0) ++n_v;
        else ++n_h;
    }
    if (cpu_mode) {
        // RAM budget for the host-resident experts: the cold tail (highest ranks, rank-major order) is not loaded and is read from the file on demand (SSD) instead
        uint64_t host_total = 0;
        for (size_t i = 0; i < N; ++i)
            if (tier[i] == 2) host_total += al(dir[i].body_bytes);
        double budget_gb;
        if (getenv("HY4_RAM_BUDGET_GB")) budget_gb = atof(getenv("HY4_RAM_BUDGET_GB"));
        else {
            const double need = getenv("HY4_HOST_NEED_GB") ? atof(getenv("HY4_HOST_NEED_GB")) : 0.0;
            budget_gb = ram_limit_gb() - 3.5 - need - 1.5;  // OS and runtime, host KV / kv_b copies, engram cache + pinned staging
        }
        const uint64_t budget = budget_gb > 0 ? (uint64_t)(budget_gb * 1e9) : 0;
        if (host_total > budget) {
            uint64_t used = 0;
            bool full = false;
            for (size_t r = 0; r < 512 && !full; ++r)
                for (size_t L = 0; L < N / 512; ++L) {
                    const size_t i = L * 512 + r;
                    if (tier[i] != 2) continue;
                    const uint64_t sz = al(dir[i].body_bytes);
                    if (used + sz > budget) { full = true; break; }
                    used += sz;
                }
            // everything from the first miss on (in rank-major order) is cold
            uint64_t used2 = 0;
            bool cold = false;
            for (size_t r = 0; r < 512; ++r)
                for (size_t L = 0; L < N / 512; ++L) {
                    const size_t i = L * 512 + r;
                    if (tier[i] != 2) continue;
                    const uint64_t sz = al(dir[i].body_bytes);
                    if (!cold && used2 + sz > budget) cold = true;
                    if (cold) { tier[i] = 3; ++n_cold; cold_bytes += sz; --n_h; }
                    else used2 += sz;
                }
            fprintf(stderr, "[load] RAM budget %.1f GB for the host experts (%.1f GB wanted): %llu experts (%.2f GB, the coldest ranks) stay on the SSD\n", budget_gb, host_total / 1e9, (unsigned long long)n_cold,
                    cold_bytes / 1e9);
        }
    }
    HCK(hipMalloc((void**)&g.d_sec4, vb ? vb : 4096));
    uint8_t* hbase = nullptr;
    uint8_t* hdev = nullptr;
    if (hb) {
        double t0 = now_s();
        HCK(hipHostMalloc((void**)&hbase, hb, hipHostMallocMapped));
        HCK(hipHostGetDevicePointer((void**)&hdev, hbase, 0));
        fprintf(stderr, "[load] pinned host arena %.2f GB in %.1f s\n", hb / 1e9, now_s() - t0);
    }
    g.h_arena = hbase;
    g.vram_expert_bytes = vb;
    g.host_expert_bytes = hb;
    g.n_vram_experts = n_v;
    g.n_host_experts = n_h;
    g.expert_tier = tier;
    const int NT = 8;
    std::atomic<size_t> next{0};
    std::atomic<int> failed{0};
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < NT; ++t)
        th.emplace_back([&]() {
            AlignedFile f;
            if (!f.open(path, false)) { failed = 1; return; }
            void* stg = nullptr;
            hipStream_t st;
            HCK(hipHostMalloc(&stg, 8u << 20, hipHostMallocDefault));
            HCK(hipStreamCreate(&st));
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= N) break;
                const uint64_t fo = h.sections[4].file_offset + dir[i].body_offset;
                const size_t n = dir[i].body_bytes;
                if (tier[i] == 2 || tier[i] == 3) continue;  // loaded / read on demand by hy4_cpu_experts
                if (tier[i] == 1) {
                    if (f.read_at(fo, hbase + off[i], n) < n) { failed = 1; break; }
                } else {
                    if (n > (8u << 20) || f.read_at(fo, stg, n) < n) { failed = 1; break; }
                    HCK(hipMemcpyAsync(g.d_sec4 + off[i], stg, n, hipMemcpyHostToDevice, st));
                    HCK(hipStreamSynchronize(st));
                }
            }
            HCK(hipStreamDestroy(st));
            HCK(hipHostFree(stg));
        });
    for (auto& t : th) t.join();
    if (failed) { err = "expert read failed"; return false; }
    if (cpu_mode) g.cpu_mask.assign(N / 512, std::vector<bool>(512, false));
    if (n_cold) g.cold_mask.assign(N / 512, std::vector<bool>(512, false));
    for (size_t i = 0; i < N; ++i) {
        const auto& e = dir[i];
        ExpertRef& er = g.experts[(i / 512) * 512 + e.expert_id];
        er.tier = tier[i];
        if (tier[i] == 2) { g.cpu_mask[i / 512][e.expert_id] = true; continue; }
        if (tier[i] == 3) { g.cold_mask[i / 512][e.expert_id] = true; continue; }
        uint8_t* dst = tier[i] == 0 ? g.d_sec4 + off[i] : hdev + off[i];
        er.body_off = (uint64_t)(uintptr_t)dst - (uint64_t)(uintptr_t)g.d_sec4;
    }
    g.t_load_experts = now_s() - t0;
    fprintf(stderr, "[load] experts tiered: %llu in VRAM (%.2f GB), %llu in %s (%.2f GB), %.1f s\n", (unsigned long long)n_v, vb / 1e9, (unsigned long long)n_h,
            cpu_mode ? "host RAM computed by the CPU" : "pinned host memory read by the GPU", cpu_mode ? (double)(g.sec4_bytes - vb) / 1e9 : hb / 1e9, g.t_load_experts);
    return true;
}

inline bool load_gpu_model(GpuModel& g, const std::string& path, bool with_experts) {
    std::string err;
    g.path = path;
    double t0 = now_s();
    if (!g.rd.open(path, err)) {
        fprintf(stderr, "open: %s\n", err.c_str());
        return false;
    }
    parse_manifest(g.rd.manifest_json(), g.man);
    fprintf(stderr, "[load] manifest %.1f s: %zu trunk tensors\n", now_s() - t0, g.man.T.size());
    const auto& h = g.rd.header();
    g.trunk_bytes = h.sections[1].byte_length;
    g.sec4_bytes = h.sections[4].byte_length;
    const bool uma_mapped = gpu_is_uma() && getenv("HY4_UMA_MAPPED") && atoi(getenv("HY4_UMA_MAPPED")) != 0;
    const uint64_t trunk_al = (g.trunk_bytes + 4095) & ~4095ull;
    if (uma_mapped) {
        void* hp = nullptr;
        HCK(hipHostMalloc(&hp, trunk_al, hipHostMallocMapped));
        read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, (uint8_t*)hp, &g.t_load_trunk, "trunk(uma-mapped)");
        HCK(hipHostGetDevicePointer((void**)&g.d_trunk, hp, 0));
    } else {
        HCK(hipMalloc((void**)&g.d_trunk, trunk_al));
        stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
    }
    if (!g.rd.read_section(2, g.seed, err)) {
        fprintf(stderr, "sec2: %s\n", err.c_str());
        return false;
    }
    const auto& dir = g.rd.expert_directory();
    g.experts.assign(41 * 512, ExpertRef{0, 0, 0, 0});
    for (size_t i = 0; i < dir.size(); ++i) {
        const auto& e = dir[i];
        g.experts[(i / 512) * 512 + e.expert_id] = ExpertRef{e.body_offset, e.dn_offset, e.gu_code_id, e.dn_code_id};
    }
    if (with_experts) {
        size_t fr = 0, tot = 0;
        gpu_mem_avail(fr, tot);
        if (gpu_is_uma() && fr > trunk_al) fr -= trunk_al;
        double reserve_gb = getenv("HY4_RESERVE_GB") ? atof(getenv("HY4_RESERVE_GB")) : 6.0;
        double budget = (double)fr - reserve_gb * 1073741824.0;
        if (getenv("HY4_EXPERT_VRAM_GB")) budget = atof(getenv("HY4_EXPERT_VRAM_GB")) * 1073741824.0;
        const uint64_t sec4_al = (g.sec4_bytes + 4095) & ~4095ull;
        bool loaded_full = false;
        if (budget >= (double)sec4_al) {
            if (uma_mapped) {
                void* hp = nullptr;
                if (hipHostMalloc(&hp, sec4_al, hipHostMallocMapped) == hipSuccess) {
                    read_file_parallel(path, h.sections[4].file_offset, g.sec4_bytes, (uint8_t*)hp, &g.t_load_experts, "experts(uma-mapped)");
                    HCK(hipHostGetDevicePointer((void**)&g.d_sec4, hp, 0));
                    g.h_arena = (uint8_t*)hp;
                    loaded_full = true;
                } else {
                    (void)hipGetLastError();
                }
            } else {
                if (hipMalloc((void**)&g.d_sec4, sec4_al) == hipSuccess) {
                    stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
                    loaded_full = true;
                } else {
                    (void)hipGetLastError();
                    fprintf(stderr, "[load] full expert cudaMalloc (%.2f GB) failed; falling back to tiered loading\n", sec4_al / 1e9);
                    size_t fr2 = 0, tot2 = 0;
                    HCK(hipMemGetInfo(&fr2, &tot2));
                    budget = (double)fr2 - reserve_gb * 1073741824.0;
                }
            }
        }
        if (!loaded_full) {
            // host-resident experts: pinned + read by the GPU (needs the RAM) or computed by the CPU. HY4_CPU_EXPERTS=1/0 forces, default: CPU when they would use > 45% of the RAM or the CPU has AVX-512 VBMI+VNNI
            bool cpu_mode = false;
            {
                double host_gb = (g.sec4_bytes - std::max(0.0, std::min(budget, (double)g.sec4_bytes))) / 1e9, mem_gb = host_mem_gb();
                cpu_mode = getenv("HY4_CPU_EXPERTS") ? atoi(getenv("HY4_CPU_EXPERTS")) != 0 : (host_gb > 0.45 * mem_gb || host_cpu_fast());
            }
            if (!load_experts_tiered(g, path, budget < 0 ? 0 : (size_t)budget, cpu_mode, err)) {
                fprintf(stderr, "experts: %s\n", err.c_str());
                return false;
            }
        }
    }
    fprintf(stderr, "[load] done: trunk %.2f GB in %.1f s, experts %.2f GB in %.1f s\n", g.trunk_bytes / 1e9, g.t_load_trunk, g.sec4_bytes / 1e9, g.t_load_experts);
    return true;
}

}  // namespace hy4
