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

inline bool gpu_is_uma() {
    static int cached = -1;
    if (cached < 0) {
        hipDeviceProp_t dp{};
        cached = (hipGetDeviceProperties(&dp, 0) == hipSuccess && dp.integrated != 0) ? 1 : 0;
    }
    return cached != 0;
}

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
            const unsigned long long margin_kb = 4ull << 20;
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

struct ExpertRef {  // per (moe_layer, expert id); offsets relative to the arena base
    uint64_t body_off;
    uint32_t dn_off;  // relative to body_off
    uint8_t gu_code, dn_code;
    uint8_t tier = 0;  // 0 device arena, 1 pinned host memory read by the GPU, 2 host RAM computed by the CPU (hybrid)
};

struct PeerGpu {
    int dev = 0;
    uint8_t* d_sec4 = nullptr;
    uint64_t vram_bytes = 0, n_experts = 0;
    std::vector<ExpertRef> experts;  // [41*512], tier=0 on this peer, tier=4 elsewhere
};

struct GpuModel {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    int primary_gpu = 0;
    std::vector<int> peer_devs;   // secondary GPU device ids for multi-GPU expert split (--gpus 0,1,...)
    std::vector<PeerGpu> peers;
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
    double trunk = 5.92, sec4_gb = 0.0;
    if (!model_path.empty()) {
        strata::artifact::StrataFileReader rd;
        std::string e;
        if (rd.open(model_path, e)) {
            trunk = (double)rd.header().sections[1].byte_length / GB;
            sec4_gb = (double)rd.header().sections[4].byte_length / GB;
        }
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
    const double cublas_gb = HY4_CUDA ? 0.08 : 0.0;  // 64 MiB reusable side-stream FP16 dequant + cuBLAS scratch
    auto est_work_gb = [&](int c) {
        // Aliased work_arena (max(2*c*16384*4, moe_ws_bytes(c))) + qa_h/H1/x/z/etc (~0.195 MB/tok) + SWA rings (30*(512+c)*1152 B) + misc 0.22 GB
        const double arena = std::max(2.0 * c * 16384.0 * 4.0, (double)c * 16.0 * 1536.0 * 6.0) / GB;
        return arena + c * 0.195e6 / GB + 0.22 + 30.0 * (512.0 + c) * 1152.0 / GB;
    };
    if (chunk <= 0) {
        // 16 GB+ cards default to Tc = 8192 when ctx > 4096 (halves PCIe passes 32->16 at 128K; capped to 4096 when ctx <= 4096 to keep max expert cache!)
        // 12 GB -> 2048; 8 GB -> 1024/512
        int c = (tot >= 15.0) ? ((max_ctx > 4096) ? 8192 : 4096) : (tot >= 11.0 ? 2048 : 1024);
        const double work_cap = (tot >= 15.0) ? 0.22 : 0.16;
        while (c > 256 && est_work_gb(c) > work_cap * tot) c >>= 1;
        while (c > 256 && trunk + est_work_gb(c) + kv_init_gb + wkv + cublas_gb + 0.55 > tot) c >>= 1;
        // Preserve 100%-resident placement when a smaller chunk keeps all experts + full KV in VRAM (e.g. SQ-96 128K on 96 GB card)
        if (sec4_gb > 0.0 && c > 4096) {
            const double kv_full = (double)((max_ctx + 8 + 15) / 16 * 16) * per_pos / GB;
            if (trunk + sec4_gb + est_work_gb(c) + kv_full + wkv + cublas_gb + 0.55 > tot &&
                trunk + sec4_gb + est_work_gb(c >> 1) + kv_full + wkv + cublas_gb + 0.55 <= tot)
                c >>= 1;
        }
        chunk = c;
    }
    const double slots = (double)((max_ctx + 8 + 15) / 16 * 16);
    const double rows = 12.0 * slots * 1152.0 / GB;                                                         // full max_ctx sparse-layer KV rows (no Vt)
    const double pivg = 12.0 * (double)max_ctx * 512.0 / GB + 12.0 * (double)(max_ctx / 4 + 8) * 256.0 / GB;  // full max_ctx IndexPool iv/ig + pooled keys
    const double work_gb = est_work_gb(chunk);
    // Check if full max_ctx KV fits in VRAM; if not and card is very small (8 GB), set dev_pos for VRAM-to-host spill
    int mask = 0;
    if (trunk + work_gb + pivg + rows + wkv + cublas_gb + 0.35 > tot) mask |= 1;
    long dev_pos = 0;
    if ((mask & 1) && !getenv("HY4_KV_HOST")) {
        const double avail = tot - trunk - work_gb - wkv - cublas_gb - 0.35;
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
        const double reserve = work_gb + kv_res + wkv + cublas_gb + pf_res + 0.45;
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
    std::vector<int> owner(N, -1);  // 0 = primary GPU, 1..ng-1 = peer GPU, -1 = host
    std::vector<uint64_t> off(N, 0);
    const size_t ng = 1 + g.peer_devs.size();
    std::vector<uint64_t> vcap(ng, 0), vb_gpu(ng, 0), nv_gpu(ng, 0);
    // Adaptive VRAM expert cache (single-GPU cpu_mode): when host RAM fits all experts (e.g. 64 GB+ Hero cards), use 100% dynamic slots
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
        const uint64_t overflow = all_al_bytes - ram_budget_bytes;
        const double static_share = std::min(0.75, (double)overflow / (double)vram_budget);
        dyn_frac = std::max(0.25, 1.0 - static_share);
    }
    size_t static_budget = vram_budget;
    int dyn_slots = 0;
    size_t dyn_bytes = 0;
    if (g.peer_devs.empty() && cpu_mode && dyn_frac > 0.01 && vram_budget >= 2 * max_body_sz && max_body_sz > 0) {
        size_t want_dyn = (size_t)((double)vram_budget * dyn_frac);
        dyn_slots = (int)(want_dyn / max_body_sz);
        dyn_bytes = (size_t)dyn_slots * max_body_sz;
        static_budget = vram_budget - dyn_bytes;
    }
    vcap[0] = static_budget;
#if HY4_CUDA
    for (size_t p = 0; p < g.peer_devs.size(); ++p) {
        const int pd = g.peer_devs[p];
        HCK(hipSetDevice(pd));
        size_t pfr = 0, ptot = 0;
        HCK(hipMemGetInfo(&pfr, &ptot));
        const size_t preserve = (size_t)512 << 20;  // 512 MiB for peer moe_ws + buffers + runtime
        vcap[1 + p] = pfr > preserve ? (uint64_t)(pfr - preserve) : 0;
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, g.primary_gpu, pd) == cudaSuccess && can) {
            (void)cudaSetDevice(g.primary_gpu); (void)cudaDeviceEnablePeerAccess(pd, 0); (void)cudaGetLastError();
            (void)cudaSetDevice(pd); (void)cudaDeviceEnablePeerAccess(g.primary_gpu, 0); (void)cudaGetLastError();
        }
    }
    HCK(hipSetDevice(g.primary_gpu));
#endif
    uint64_t hb = 0;
    for (size_t r = 0; r < 512; ++r) {
        bool any_placed = false;
        for (size_t L = 0; L < N / 512; ++L) {
            size_t i = L * 512 + r;
            uint64_t sz = al(dir[i].body_bytes);
            int best_g = -1;
            double best_frac = -1.0;
            for (size_t gi = 0; gi < ng; ++gi) {
                if (vcap[gi] > 0 && vb_gpu[gi] + sz <= vcap[gi]) {
                    const double frac = (double)(vcap[gi] - vb_gpu[gi]) / (double)vcap[gi];
                    if (frac > best_frac) { best_frac = frac; best_g = (int)gi; }
                }
            }
            if (best_g >= 0) {
                owner[i] = best_g;
                tier[i] = (best_g == 0) ? 0 : 4;  // 4 = peer-GPU resident
                off[i] = vb_gpu[best_g];
                vb_gpu[best_g] += sz;
                ++nv_gpu[best_g];
                any_placed = true;
            }
        }
        if (!any_placed) break;
    }
    uint64_t vb = vb_gpu[0];
    // Reclaim any leftover static_budget slack into additional dynamic slots
    if (g.peer_devs.empty() && cpu_mode && dyn_frac > 0.01 && max_body_sz > 0 && vram_budget > vb + dyn_bytes + max_body_sz) {
        int extra = (int)((vram_budget - vb - dyn_bytes) / max_body_sz);
        dyn_slots += extra;
        dyn_bytes = (size_t)dyn_slots * max_body_sz;
    }
    uint64_t n_v = nv_gpu[0], n_h = 0, n_cold = 0, cold_bytes = 0;
    for (size_t i = 0; i < N; ++i) {
        if (tier[i] == 1 && cpu_mode) tier[i] = 2;
        if (tier[i] == 1) { off[i] = hb; hb += al(dir[i].body_bytes); ++n_h; }
        else if (tier[i] == 2) ++n_h;
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
    HCK(hipSetDevice(g.primary_gpu));
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
    const uint64_t total_vram_arena = vb + dyn_bytes;
    g.peers.clear();
    g.peers.resize(g.peer_devs.size());
    for (size_t p = 0; p < g.peer_devs.size(); ++p) {
        g.peers[p].dev = g.peer_devs[p];
        g.peers[p].vram_bytes = vb_gpu[1 + p];
        g.peers[p].n_experts = nv_gpu[1 + p];
        g.peers[p].experts.assign(41 * 512, ExpertRef{0, 0, 0, 0, 4});
        HCK(hipSetDevice(g.peer_devs[p]));
        HCK(hipMalloc((void**)&g.peers[p].d_sec4, vb_gpu[1 + p] ? vb_gpu[1 + p] : 4096));
    }
    HCK(hipSetDevice(g.primary_gpu));
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
            HCK(hipSetDevice(g.primary_gpu));
            HCK(hipHostMalloc(&stg, 8u << 20, hipHostMallocDefault));
            std::vector<hipStream_t> sts(ng, nullptr);
            for (size_t gi = 0; gi < ng; ++gi) {
                HCK(hipSetDevice(gi == 0 ? g.primary_gpu : g.peer_devs[gi - 1]));
                HCK(hipStreamCreate(&sts[gi]));
            }
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= N) break;
                const uint64_t fo = h.sections[4].file_offset + dir[i].body_offset;
                const size_t n = dir[i].body_bytes;
                if (tier[i] == 2 || tier[i] == 3) continue;  // loaded / read on demand by hy4_cpu_experts
                if (tier[i] == 1) {
                    if (f.read_at(fo, hbase + off[i], n) < n) { failed = 1; break; }
                } else {
                    const int gi = owner[i];
                    if (n > (8u << 20) || f.read_at(fo, stg, n) < n) { failed = 1; break; }
                    const int dev = gi == 0 ? g.primary_gpu : g.peer_devs[gi - 1];
                    uint8_t* dst = (gi == 0 ? g.d_sec4 : g.peers[gi - 1].d_sec4) + off[i];
                    HCK(hipSetDevice(dev));
                    HCK(hipMemcpyAsync(dst, stg, n, hipMemcpyHostToDevice, sts[gi]));
                    HCK(hipStreamSynchronize(sts[gi]));
                }
            }
            for (size_t gi = 0; gi < ng; ++gi) {
                HCK(hipSetDevice(gi == 0 ? g.primary_gpu : g.peer_devs[gi - 1]));
                HCK(hipStreamDestroy(sts[gi]));
            }
            HCK(hipSetDevice(g.primary_gpu));
            HCK(hipHostFree(stg));
        });
    for (auto& t : th) t.join();
    HCK(hipSetDevice(g.primary_gpu));
    if (failed) { err = "expert read failed"; return false; }
    if (cpu_mode && (n_h > 0 || n_cold > 0)) g.cpu_mask.assign(N / 512, std::vector<bool>(512, false));
    if (n_cold) g.cold_mask.assign(N / 512, std::vector<bool>(512, false));
    for (size_t i = 0; i < N; ++i) {
        const auto& e = dir[i];
        const size_t idx = (i / 512) * 512 + e.expert_id;
        ExpertRef& er = g.experts[idx];
        er.tier = tier[i];
        if (tier[i] == 2) { g.cpu_mask[i / 512][e.expert_id] = true; continue; }
        if (tier[i] == 3) { g.cold_mask[i / 512][e.expert_id] = true; continue; }
        if (tier[i] == 4) {
            const int p = owner[i] - 1;
            ExpertRef& pr = g.peers[p].experts[idx];
            pr.body_off = off[i];
            pr.dn_off = er.dn_off;
            pr.gu_code = er.gu_code;
            pr.dn_code = er.dn_code;
            pr.tier = 0;
            er.body_off = 0;
            continue;
        }
        uint8_t* dst = tier[i] == 0 ? g.d_sec4 + off[i] : hdev + off[i];
        er.body_off = (uint64_t)(uintptr_t)dst - (uint64_t)(uintptr_t)g.d_sec4;
    }
    g.t_load_experts = now_s() - t0;
    if (g.peers.empty()) {
        fprintf(stderr, "[load] experts tiered: %llu static VRAM (%.2f GB) + %d adaptive VRAM slots (%.2f GB) = %.2f GB VRAM, %llu in %s (%.2f GB), %.1f s\n",
                (unsigned long long)n_v, vb / 1e9, dyn_slots, dyn_bytes / 1e9, total_vram_arena / 1e9, (unsigned long long)n_h,
                cpu_mode ? "host RAM computed by the CPU" : "pinned host memory read by the GPU", cpu_mode ? (double)(g.sec4_bytes - vb) / 1e9 : hb / 1e9, g.t_load_experts);
    } else {
        uint64_t tot_vram_b = vb, tot_vram_n = n_v;
        for (const auto& pr : g.peers) { tot_vram_b += pr.vram_bytes; tot_vram_n += pr.n_experts; }
        fprintf(stderr, "[load] multi-GPU experts (%zu GPUs): total %llu in VRAM (%.2f GB: GPU%d=%llu/%.2fGB", ng, (unsigned long long)tot_vram_n, tot_vram_b / 1e9, g.primary_gpu, (unsigned long long)n_v, vb / 1e9);
        for (const auto& pr : g.peers) fprintf(stderr, ", GPU%d=%llu/%.2fGB", pr.dev, (unsigned long long)pr.n_experts, pr.vram_bytes / 1e9);
        fprintf(stderr, "), %llu in host/CPU (%.2f GB), %.1f s\n", (unsigned long long)n_h, (double)(g.sec4_bytes > tot_vram_b ? g.sec4_bytes - tot_vram_b : 0) / 1e9, g.t_load_experts);
    }
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
        gpu_mem_avail(fr, tot);
        if (gpu_is_uma() && fr > g.trunk_bytes) fr -= g.trunk_bytes;
        double reserve_gb = getenv("HY4_RESERVE_GB") ? atof(getenv("HY4_RESERVE_GB")) : 6.0;
        double budget = (double)fr - reserve_gb * 1073741824.0;
        if (getenv("HY4_EXPERT_VRAM_GB")) budget = atof(getenv("HY4_EXPERT_VRAM_GB")) * 1073741824.0;
        if (g.peer_devs.empty() && budget >= (double)((g.sec4_bytes + 4095) & ~4095ull)) {
            if (hipMalloc((void**)&g.d_sec4, (g.sec4_bytes + 4095) & ~4095ull) == hipSuccess) {
                stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
            } else {
                (void)hipGetLastError();
                HCK(hipMemGetInfo(&fr, &tot));
                budget = (double)fr - reserve_gb * 1073741824.0;
                bool cpu_mode = getenv("HY4_CPU_EXPERTS") ? atoi(getenv("HY4_CPU_EXPERTS")) != 0 : host_cpu_fast();
                if (!load_experts_tiered(g, path, budget < 0 ? 0 : (size_t)budget, cpu_mode, err)) {
                    fprintf(stderr, "experts: %s\n", err.c_str());
                    return false;
                }
            }
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

// Overlay distilled DFly2 weights from a flat binary `DFLYOVR1` file directly into `g.d_trunk` (zero .strata file duplication).
inline bool load_dfly_override(GpuModel& g, const std::string& path, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open dfly-override: " + path; return false; }
    char magic[8] = {};
    uint32_t count = 0;
    if (fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "DFLYOVR1", 8) != 0 || fread(&count, 4, 1, f) != 1) {
        fclose(f);
        err = "invalid DFLYOVR1 header: " + path;
        return false;
    }
    double t0 = now_s();
    size_t applied = 0, total_bytes = 0;
    std::vector<uint8_t> buf;
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t nlen = 0;
        uint64_t blen = 0;
        if (fread(&nlen, 2, 1, f) != 1 || fread(&blen, 8, 1, f) != 1) { fclose(f); err = "truncated DFLYOVR1 entry"; return false; }
        std::string name(nlen, '\0');
        if (fread(&name[0], 1, nlen, f) != nlen) { fclose(f); err = "truncated DFLYOVR1 name"; return false; }
        buf.resize((size_t)blen);
        if (fread(buf.data(), 1, (size_t)blen, f) != (size_t)blen) { fclose(f); err = "truncated DFLYOVR1 payload"; return false; }
        auto it = g.man.T.find(name);
        if (it == g.man.T.end()) continue;
        if (it->second.bytes != blen) {
            fclose(f);
            err = "dfly-override byte mismatch for " + name + " (" + std::to_string(blen) + " vs " + std::to_string(it->second.bytes) + ")";
            return false;
        }
        HCK(hipMemcpy(g.d_trunk + it->second.off, buf.data(), (size_t)blen, hipMemcpyHostToDevice));
        ++applied;
        total_bytes += (size_t)blen;
    }
    fclose(f);
    fprintf(stderr, "[dfly-override] applied %zu/%u tensors (%.2f MB) from %s in %.2f s\n", applied, count, total_bytes / 1e6, path.c_str(), now_s() - t0);
    return applied > 0;
}

}  // namespace hy4
