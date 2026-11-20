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
    std::vector<std::vector<bool>> cpu_mask;  // [41][512] by expert id: tier-2 experts (computed by hy4_cpu_experts); empty when none
    double t_load_trunk = 0, t_load_experts = 0;

    const uint8_t* tensor_dev(const std::string& name) const { return d_trunk + man.get(name).off; }
};

// Streams [file_off, file_off+bytes) into device memory dst with a reader thread and a ring of pinned staging buffers.
inline void stream_to_device(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    const size_t CH = 128ull << 20;
    const int NB = 4;
    std::vector<void*> buf(NB);
    for (int i = 0; i < NB; ++i) HCK(hipHostMalloc(&buf[i], CH, hipHostMallocDefault));
    AlignedFile f;
    if (!f.open(path, true)) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    uint64_t nchunks = (bytes + CH - 1) / CH;
    std::atomic<uint64_t> filled{0};  // chunks read
    std::atomic<uint64_t> freed{0};   // chunks copied (buffers released)
    std::vector<size_t> clen(nchunks);
    double t0 = now_s();
    std::thread reader([&]() {
        for (uint64_t c = 0; c < nchunks; ++c) {
            while (c - freed.load(std::memory_order_acquire) >= (uint64_t)NB) std::this_thread::sleep_for(std::chrono::microseconds(200));
            size_t want = (size_t)std::min<uint64_t>(CH, bytes - c * CH);
            size_t rd_len = (want + 4095) & ~(size_t)4095;
            size_t got = f.read_at(file_off + c * CH, buf[c % NB], rd_len);
            clen[c] = std::min(got, want);
            if (got < want) {
                fprintf(stderr, "short read in %s chunk %llu (%zu < %zu)\n", what, (unsigned long long)c, got, want);
                exit(1);
            }
            filled.store(c + 1, std::memory_order_release);
        }
    });
    hipStream_t st;
    HCK(hipStreamCreate(&st));
    uint64_t last_print = 0;
    for (uint64_t c = 0; c < nchunks; ++c) {
        while (filled.load(std::memory_order_acquire) <= c) std::this_thread::sleep_for(std::chrono::microseconds(200));
        HCK(hipMemcpyAsync(dst + c * CH, buf[c % NB], clen[c], hipMemcpyHostToDevice, st));
        HCK(hipStreamSynchronize(st));
        freed.store(c + 1, std::memory_order_release);
        if (nchunks > 16 && (c + 1) * 10 / nchunks != last_print) {
            last_print = (c + 1) * 10 / nchunks;
            fprintf(stderr, "[load %s] %llu%% %.1f s %.2f GB/s\n", what, (unsigned long long)(last_print * 10), now_s() - t0, (c + 1) * (double)CH / (now_s() - t0) / 1e9);
        }
    }
    reader.join();
    HCK(hipStreamDestroy(st));
    for (int i = 0; i < NB; ++i) HCK(hipHostFree(buf[i]));
    *seconds = now_s() - t0;
}

// the CPU expert kernel needs AVX-512 VBMI + VNNI; without it the GPU reads host-resident experts directly (zero-copy) when they fit the RAM
inline bool host_cpu_fast() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vnni");
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
        // ~1.4 MB of working buffers per chunk token; keep them near 12% of the card
        int c = 2048;
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
    if (getenv("HY4_KV_HOST")) mask = atoi(getenv("HY4_KV_HOST"));
    else {
        char m[8];
        snprintf(m, sizeof m, "%d", mask);
        hy4_setenv("HY4_KV_HOST", m, 1);
    }
    if (!getenv("HY4_RESERVE_GB")) {
        double reserve = work_gb + pivg + ((mask & 1) ? 0.0 : rows) + ((mask & 2) ? 0.0 : wkv) + (tot >= 10.0 && mask == 0 ? 1.45 : 0.0);  // + the prefill streaming scratch when everything else fits
        char b[32];
        snprintf(b, sizeof b, "%.2f", reserve);
        hy4_setenv("HY4_RESERVE_GB", b, 1);
    }
    fprintf(stderr, "[plan] ctx %d chunk %d: buffers ~%.2f GB, KV rows %.2f GB, IndexPool %.2f GB, kv_b copies %.2f GB, trunk %.2f GB on a %.1f GB card -> host mask %d (1 rows, 2 kv_b copies); reserve %s GB\n", max_ctx, chunk, work_gb,
            rows, pivg, wkv, trunk, tot, atoi(getenv("HY4_KV_HOST")), getenv("HY4_RESERVE_GB"));
    return chunk;
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
    uint64_t n_v = 0, n_h = 0;
    for (size_t i = 0; i < N; ++i) {
        if (tier[i] == 1 && cpu_mode) tier[i] = 2;
        if (tier[i] == 1) { off[i] = hb; hb += al(dir[i].body_bytes); ++n_h; }
        else if (tier[i] == 0) ++n_v;
        else ++n_h;
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
                if (tier[i] == 2) continue;  // loaded by hy4_cpu_experts
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
    for (size_t i = 0; i < N; ++i) {
        const auto& e = dir[i];
        ExpertRef& er = g.experts[(i / 512) * 512 + e.expert_id];
        er.tier = tier[i];
        if (tier[i] == 2) { g.cpu_mask[i / 512][e.expert_id] = true; continue; }
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
    HCK(hipMalloc((void**)&g.d_trunk, (g.trunk_bytes + 4095) & ~4095ull));
    stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
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
        HCK(hipMemGetInfo(&fr, &tot));
        double reserve_gb = getenv("HY4_RESERVE_GB") ? atof(getenv("HY4_RESERVE_GB")) : 6.0;
        double budget = (double)fr - reserve_gb * 1073741824.0;
        if (getenv("HY4_EXPERT_VRAM_GB")) budget = atof(getenv("HY4_EXPERT_VRAM_GB")) * 1073741824.0;
        if (budget >= (double)((g.sec4_bytes + 4095) & ~4095ull)) {
            HCK(hipMalloc((void**)&g.d_sec4, (g.sec4_bytes + 4095) & ~4095ull));
            stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
        } else {
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
