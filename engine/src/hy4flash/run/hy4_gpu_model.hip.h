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
    std::vector<int> expert_rank;              // [41*512] indexed by moe_layer*512 + expert_id -> hotness rank 0..511
    std::vector<std::vector<bool>> cpu_mask;   // [41][512] by expert id: tier-2 experts (resident in host RAM, computed by hy4_cpu_experts); empty when none
    std::vector<std::vector<bool>> cold_mask;  // [41][512] by expert id: tier-3 experts (do not fit the RAM budget: read from the .strata on demand by hy4_cpu_experts); empty when none
    // host-mapped embed_tokens (discrete cards < 48 GB): frees 0.38 GB of d_trunk VRAM
    uint8_t* h_embed = nullptr;
    uint8_t* d_embed_mapped = nullptr;
    // adaptive VRAM expert cache pool allocated as independent ~384 MiB slabs so kv_grow_to() can reclaim slabs to keep 32K-128K KV 100% in VRAM!
    std::vector<uint8_t*> vcache_slabs;
    int vcache_slots_per_slab = 0;
    size_t vcache_slot_bytes = 0;
    int vcache_nslots = 0;
    uint8_t* vcache_slot_ptr(int s) const {
        return vcache_slabs[(size_t)(s / vcache_slots_per_slab)] + (size_t)(s % vcache_slots_per_slab) * vcache_slot_bytes;
    }
    uint64_t vcache_slot_off(int s) const {
        return (uint64_t)(uintptr_t)vcache_slot_ptr(s) - (uint64_t)(uintptr_t)d_sec4;
    }
    double t_load_trunk = 0, t_load_experts = 0;

    const uint8_t* tensor_dev(const std::string& name) const {
        if (d_embed_mapped && name == "model.embed_tokens.weight") return d_embed_mapped;
        return d_trunk + man.get(name).off;
    }
};

// Streams [file_off, file_off+bytes) into device memory dst with a reader thread and a ring of pinned staging buffers (handles arbitrary file_off alignment).
inline void stream_to_device(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    if (bytes == 0) { *seconds = 0.0; return; }
    const uint64_t fo_al = file_off & ~4095ull;
    const size_t head_skip = (size_t)(file_off - fo_al);
    const uint64_t total_span = bytes + head_skip;
    const size_t CH = 128ull << 20;
    const int NB = 4;
    std::vector<void*> buf(NB);
    for (int i = 0; i < NB; ++i) HCK(hipHostMalloc(&buf[i], CH, hipHostMallocDefault));
    AlignedFile f;
    if (!f.open(path, true)) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    uint64_t nchunks = (total_span + CH - 1) / CH;
    std::atomic<uint64_t> filled{0};  // chunks read
    std::atomic<uint64_t> freed{0};   // chunks copied (buffers released)
    std::vector<size_t> clen(nchunks);
    double t0 = now_s();
    std::thread reader([&]() {
        for (uint64_t c = 0; c < nchunks; ++c) {
            while (c - freed.load(std::memory_order_acquire) >= (uint64_t)NB) std::this_thread::sleep_for(std::chrono::microseconds(200));
            size_t want = (size_t)std::min<uint64_t>(CH, total_span - c * CH);
            size_t rd_len = (want + 4095) & ~(size_t)4095;
            size_t got = f.read_at(fo_al + c * CH, buf[c % NB], rd_len);
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
    uint64_t dst_off = 0;
    for (uint64_t c = 0; c < nchunks; ++c) {
        while (filled.load(std::memory_order_acquire) <= c) std::this_thread::sleep_for(std::chrono::microseconds(200));
        const size_t sk = (c == 0) ? head_skip : 0;
        const size_t ncp = clen[c] - sk;
        HCK(hipMemcpyAsync(dst + dst_off, (const uint8_t*)buf[c % NB] + sk, ncp, hipMemcpyHostToDevice, st));
        HCK(hipStreamSynchronize(st));
        dst_off += ncp;
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

// Modern x86-64 CPUs (AVX2 / AVX-512) run fast(avx2)/fast(avx512) CPU experts + AdaptiveExpertCache + bounce-ring prefill streaming,
// which is 5-10x faster than zero-copy GPU PCIe reads on discrete cards.
inline bool host_cpu_fast() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx2");
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
    double trunk = 5.92;
    if (!model_path.empty()) {
        strata::artifact::StrataFileReader rd;
        std::string e;
        if (rd.open(model_path, e)) trunk = (double)rd.header().sections[1].byte_length / GB;
    }
    // Discrete cards (< 48 GB) keep embed_tokens (0.38 GB) in pinned mapped host memory
    const bool host_embed = !getenv("HY4_EMBED_VRAM") && (total < (48ull << 30));
    if (host_embed && trunk > 0.5) trunk -= 0.38;
    // Sparse KV (no Vt: 1152 B/slot) + IndexPool (576 B/slot) = 1728 B/slot * 12 layers = 20,736 B/token.
    // On discrete cards (< 48 GB) with max_ctx > 16384, allocate sparse KV + IndexPool in 16,384-token VRAM increments
    // so short/medium chats start with ~9.0-9.5 GB free VRAM for hot experts on 16 GB cards and grow cleanly up to 128K.
    const double per_pos = 12.0 * 1152.0 + 12.0 * 512.0 + 12.0 * 64.0;  // 20,736 B per token across 12 sparse layers
    int init_slots = max_ctx;
    if (!getenv("HY4_KV_NO_GROW") && total < (48ull << 30) && max_ctx > 16384) {
        init_slots = 16384;
        if (getenv("HY4_KV_INIT_SLOTS")) init_slots = std::max(4096, std::min(max_ctx, atoi(getenv("HY4_KV_INIT_SLOTS"))));
        else {
            char b[16];
            snprintf(b, sizeof b, "%d", init_slots);
            hy4_setenv("HY4_KV_INIT_SLOTS", b, 1);
        }
    }
    const double kv_init_gb = (double)((init_slots + 8 + 15) / 16 * 16) * per_pos / GB;
    const double wkv = 0.03;  // single-layer reusable FP16 wk/wv prefill scratch (28 MiB)
    auto est_work_gb = [&](int c) {
        // Aliased work_arena + qa_h/H1/x/z/etc (~0.30 MB/tok) + SWA rings (30*(512+c)*1152 B) + misc 0.22 GB
        return c * 0.30e6 / GB + 0.22 + 30.0 * (512.0 + c) * 1152.0 / GB;
    };
    if (chunk <= 0) {
        // 16 GB+ cards default to Tc = 4096 (single PCIe expert pass per 4K prompt!); 12 GB -> 2048; 8 GB -> 1024/512
        int c = (tot >= 15.0) ? 4096 : (tot >= 11.0 ? 2048 : 1024);
        while (c > 256 && est_work_gb(c) > 0.14 * tot) c >>= 1;
        while (c > 256 && trunk + est_work_gb(c) + kv_init_gb + wkv + 0.35 > tot) c >>= 1;
        chunk = c;
    }
    const double slots = (double)((max_ctx + 8 + 15) / 16 * 16);
    const double rows = 12.0 * slots * 1152.0 / GB;                                                         // full max_ctx sparse-layer KV rows (no Vt)
    const double pivg = 12.0 * (double)max_ctx * 512.0 / GB + 12.0 * (double)(max_ctx / 4 + 8) * 256.0 / GB;  // full max_ctx IndexPool iv/ig + pooled keys
    const double work_gb = est_work_gb(chunk);
    // Check if full max_ctx KV fits in VRAM; if not and card is very small (8 GB), set dev_pos for VRAM-to-host spill
    int mask = 0;
    if (trunk + work_gb + pivg + rows + wkv + 0.35 > tot) mask |= 1;
    long dev_pos = 0;
    if ((mask & 1) && !getenv("HY4_KV_HOST")) {
        const double avail = tot - trunk - work_gb - wkv - 0.35;
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
        const double host_kv = ((mask & 1) || dev_pos) ? (rows + (dev_pos ? pivg : 0.0)) : 0.0;
        const double host_emb = host_embed ? 0.40 : 0.0;
        const double host_bounce = (tot < 48.0) ? 0.55 : 0.0;  // 2 x 256 MiB pinned prefill bounce ring
        snprintf(b, sizeof b, "%.2f", host_kv + host_emb + host_bounce + 0.3);
        hy4_setenv("HY4_HOST_NEED_GB", b, 1);
    }
    if (!getenv("HY4_RESERVE_GB")) {
        const double kv_res = (mask & 1) ? 0.0 : (init_slots < max_ctx ? kv_init_gb : (dev_pos ? dev_pos * per_pos / GB : (rows + pivg)));
        const double pf_res = (tot >= 10.0 && tot < 64.0 && mask == 0) ? 0.56 : 0.0;  // 2 x 256 MiB device scratch for prefill streaming
        const double reserve = work_gb + kv_res + wkv + pf_res + 0.45;
        char b[32];
        snprintf(b, sizeof b, "%.2f", reserve);
        hy4_setenv("HY4_RESERVE_GB", b, 1);
    }
    fprintf(stderr, "[plan] ctx %d (init %d) chunk %d: buffers ~%.2f GB, KV+idx %.2f GB (128K full %.2f GB), kv_b scratch %.2f GB, trunk %.2f GB%s on a %.1f GB card -> host mask %d; reserve %s GB\n",
            max_ctx, init_slots, chunk, work_gb, kv_init_gb, rows + pivg, wkv, trunk, host_embed ? " (embed in RAM)" : "", tot, atoi(getenv("HY4_KV_HOST")), getenv("HY4_RESERVE_GB"));
    if (dev_pos) fprintf(stderr, "[plan] growable KV: up to %ld positions stay in VRAM, moves to pinned host memory if context exceeds that\n", dev_pos);
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
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return (double)ms.ullTotalPhys / 1073741824.0;
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
    // Adaptive VRAM expert cache (cpu_mode): when host RAM fits all experts (e.g. 64 GB+ Hero cards), use 100% dynamic slots
    // (core_frac = 0, proven +5-6 pp better than static-pinned cores on code/json/prose traces); when RAM is tight (cold-tier SSD),
    // put the hottest static core in VRAM-only so it does not duplicate in RAM and minimizes SSD spills.
    double ram_budget_gb = 0.0;
    if (cpu_mode) {
        if (getenv("HY4_RAM_BUDGET_GB")) ram_budget_gb = atof(getenv("HY4_RAM_BUDGET_GB"));
        else {
            const double need = getenv("HY4_HOST_NEED_GB") ? atof(getenv("HY4_HOST_NEED_GB")) : 0.0;
            ram_budget_gb = ram_limit_gb() - 3.5 - need - 1.5;  // OS and runtime, host KV / kv_b copies, engram cache + pinned staging
        }
    }
    const uint64_t ram_budget_bytes = ram_budget_gb > 0 ? (uint64_t)(ram_budget_gb * 1e9) : 0;
    uint64_t all_al_bytes = 0;
    size_t max_body_sz = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint64_t sz = al(dir[i].body_bytes);
        all_al_bytes += sz;
        max_body_sz = std::max<size_t>(max_body_sz, (size_t)sz);
    }
    double dyn_frac = 1.0;
    if (getenv("HY4_VCACHE_FRAC")) {
        dyn_frac = std::max(0.0, std::min(1.0, atof(getenv("HY4_VCACHE_FRAC"))));
    } else if (cpu_mode && all_al_bytes > ram_budget_bytes && vram_budget > 0) {
        // RAM is tight (cold tier): keep static VRAM core up to the RAM overflow so we don't needlessly push hot ranks to SSD,
        // while still keeping at least 25% of VRAM dynamic if room allows.
        const uint64_t overflow = all_al_bytes - ram_budget_bytes;
        const double static_share = std::min(0.75, (double)overflow / (double)vram_budget);
        dyn_frac = std::max(0.25, 1.0 - static_share);
    }
    size_t static_budget = vram_budget;
    int dyn_slots = 0;
    size_t dyn_bytes = 0;
    if (cpu_mode && dyn_frac > 0.01 && vram_budget >= 2 * max_body_sz && max_body_sz > 0) {
        size_t want_dyn = (size_t)((double)vram_budget * dyn_frac);
        dyn_slots = (int)(want_dyn / max_body_sz);
        dyn_bytes = (size_t)dyn_slots * max_body_sz;
        static_budget = vram_budget - dyn_bytes;
    }
    uint64_t vb = 0, hb = 0;
    bool full = false;
    for (size_t r = 0; r < 512 && !full; ++r)
        for (size_t L = 0; L < N / 512; ++L) {
            size_t i = L * 512 + r;
            uint64_t sz = al(dir[i].body_bytes);
            if (vb + sz > static_budget) { full = true; break; }
            tier[i] = 0;
            off[i] = vb;
            vb += sz;
        }
    // Reclaim any leftover static_budget slack into additional dynamic slots
    if (cpu_mode && dyn_frac > 0.01 && max_body_sz > 0 && vram_budget > vb + dyn_bytes + max_body_sz) {
        int extra = (int)((vram_budget - vb - dyn_bytes) / max_body_sz);
        dyn_slots += extra;
        dyn_bytes = (size_t)dyn_slots * max_body_sz;
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
    const uint64_t total_vram_arena = vb + dyn_bytes;
    HCK(hipMalloc((void**)&g.d_sec4, vb ? vb : 4096));
    g.vcache_slabs.clear();
    g.vcache_slot_bytes = max_body_sz;
    g.vcache_slots_per_slab = max_body_sz > 0 ? std::max(1, (int)((384ull << 20) / max_body_sz)) : 1;
    int actual_dyn = 0;
    while (actual_dyn < dyn_slots) {
        const int take = std::min(g.vcache_slots_per_slab, dyn_slots - actual_dyn);
        uint8_t* slab = nullptr;
        if (hipMalloc((void**)&slab, (size_t)take * max_body_sz) != hipSuccess) { (void)hipGetLastError(); break; }
        g.vcache_slabs.push_back(slab);
        actual_dyn += take;
    }
    dyn_slots = actual_dyn;
    dyn_bytes = (size_t)dyn_slots * max_body_sz;
    g.vcache_nslots = dyn_slots;
    uint8_t* hbase = nullptr;
    uint8_t* hdev = nullptr;
    if (hb) {
        double t0 = now_s();
        HCK(hipHostMalloc((void**)&hbase, hb, hipHostMallocMapped));
        HCK(hipHostGetDevicePointer((void**)&hdev, hbase, 0));
        fprintf(stderr, "[load] pinned host arena %.2f GB in %.1f s\n", hb / 1e9, now_s() - t0);
    }
    g.h_arena = hbase;
    g.vram_expert_bytes = total_vram_arena;
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
    fprintf(stderr, "[load] experts tiered: %llu static VRAM (%.2f GB) + %d adaptive VRAM slots (%.2f GB) = %.2f GB VRAM, %llu in %s (%.2f GB), %.1f s\n",
            (unsigned long long)n_v, vb / 1e9, dyn_slots, dyn_bytes / 1e9, total_vram_arena / 1e9, (unsigned long long)n_h,
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
    size_t fr0 = 0, tot0 = 0;
    HCK(hipMemGetInfo(&fr0, &tot0));
    const bool want_host_embed = !getenv("HY4_EMBED_VRAM") && (tot0 < (48ull << 30)) && g.man.T.count("model.embed_tokens.weight");
    bool loaded_host_embed = false;
    if (want_host_embed) {
        const TInfo emb = g.man.T.at("model.embed_tokens.weight");
        const size_t emb_al = (size_t)((emb.bytes + 4095) & ~4095ull);
        if (hipHostMalloc((void**)&g.h_embed, emb_al, hipHostMallocMapped) == hipSuccess) {
            HCK(hipHostGetDevicePointer((void**)&g.d_embed_mapped, g.h_embed, 0));
            AlignedFile ef;
            if (ef.open(path, false) && ef.read_at(h.sections[1].file_offset + emb.off, g.h_embed, (size_t)emb.bytes) == emb.bytes) {
                const uint64_t e0 = emb.off, e1 = emb.off + emb.bytes;
                const uint64_t shift = emb.bytes & ~4095ull;  // multiple of 4 KiB preserves alignment of all tensors after embed_tokens
                const uint64_t p2_dst = e1 - shift;
                const uint64_t p2_len = g.trunk_bytes - e1;
                const uint64_t new_trunk = (p2_dst + p2_len + 4095) & ~4095ull;
                HCK(hipMalloc((void**)&g.d_trunk, new_trunk));
                double t1 = 0, t2 = 0;
                if (e0 > 0) stream_to_device(path, h.sections[1].file_offset, e0, g.d_trunk, &t1, "trunk-a");
                if (p2_len > 0) stream_to_device(path, h.sections[1].file_offset + e1, p2_len, g.d_trunk + p2_dst, &t2, "trunk");
                for (auto& kv : g.man.T)
                    if (kv.second.off >= e1) kv.second.off -= shift;
                g.trunk_bytes = new_trunk;
                g.t_load_trunk = t1 + t2;
                loaded_host_embed = true;
                fprintf(stderr, "[load] embed_tokens (%.2f GB) in pinned mapped host RAM -> trunk VRAM %.2f GB\n", emb.bytes / 1e9, new_trunk / 1e9);
            } else {
                hipHostFree(g.h_embed);
                g.h_embed = nullptr;
                g.d_embed_mapped = nullptr;
            }
        } else {
            (void)hipGetLastError();
        }
    }
    if (!loaded_host_embed) {
        HCK(hipMalloc((void**)&g.d_trunk, (g.trunk_bytes + 4095) & ~4095ull));
        stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
    }
    if (!g.rd.read_section(2, g.seed, err)) {
        fprintf(stderr, "sec2: %s\n", err.c_str());
        return false;
    }
    const auto& dir = g.rd.expert_directory();
    g.experts.assign(41 * 512, ExpertRef{0, 0, 0, 0});
    g.expert_rank.assign(41 * 512, 511);
    for (size_t i = 0; i < dir.size(); ++i) {
        const auto& e = dir[i];
        const size_t idx = (i / 512) * 512 + e.expert_id;
        g.experts[idx] = ExpertRef{e.body_offset, e.dn_offset, e.gu_code_id, e.dn_code_id};
        g.expert_rank[idx] = (int)(i % 512);
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
