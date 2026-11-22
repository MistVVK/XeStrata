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
#if !HY4_CUDA
#include "strata/hip_uma_guard.hpp"
#endif
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
#if !HY4_CUDA
    if (gpu_is_uma()) {
        // First-class UMA / iGPU memory pool query on BOTH Windows (780M gfx1103, 890M gfx1150, Strix Halo gfx1151)
        // and Linux: accounts for coherent GTT / system RAM so a 512 MB - 4 GB BIOS carveout never falsely OOMs!
        uint64_t ufr = fr, utot = tot;
        strata::hip_uma::effective_uma_memory(true, fr, tot, ufr, utot);
        fr = (size_t)ufr;
        tot = (size_t)utot;
    }
#elif !defined(_WIN32)
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
    bool has_vision = false;
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

// Streams [file_off, file_off+bytes) into device memory dst with parallel unbuffered readers and a ring of pinned staging buffers (handles arbitrary file_off alignment).
inline void stream_to_device(const std::string& path, uint64_t file_off, uint64_t bytes, uint8_t* dst, double* seconds, const char* what) {
    if (bytes == 0) { *seconds = 0.0; return; }
    const uint64_t fo_al = file_off & ~4095ull;
    const size_t head_skip = (size_t)(file_off - fo_al);
    const uint64_t total_span = bytes + head_skip;
    const size_t CH = 64ull << 20;
    const int NB = 4;
    std::vector<void*> buf(NB);
    std::vector<hipEvent_t> ev(NB);
    for (int i = 0; i < NB; ++i) {
        HCK(hipHostMalloc(&buf[i], CH, hipHostMallocDefault));
        HCK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming));
    }
    uint64_t nchunks = (total_span + CH - 1) / CH;
    const int NRD = (int)std::min<uint64_t>(4, std::max<uint64_t>(1, nchunks));
    std::vector<std::atomic<int>> slot_state(NB);
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
                size_t want = (size_t)std::min<uint64_t>(CH, total_span - c * CH);
                size_t rd_len = (want + 4095) & ~(size_t)4095;
                size_t got = f.read_at(fo_al + c * CH, buf[s], rd_len);
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
                const size_t sk = (c == 0) ? head_skip : 0;
                const uint64_t dst_off = (c == 0) ? 0 : (c * CH - head_skip);
                HCK(hipMemcpyAsync(dst + dst_off, (const uint8_t*)buf[s] + sk, slot_len[s] - sk, hipMemcpyHostToDevice, st));
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

// Modern x86-64 CPUs (AVX2 / AVX-512) and ARM64 dotprod CPUs run fast CPU experts + AdaptiveExpertCache + bounce-ring prefill streaming.
inline bool host_cpu_fast() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx2");
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

inline int plan_memory(int max_ctx, int chunk, size_t total, const std::string& model_path, bool multi_gpu = false) {
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
    // On single discrete cards (< 48 GB) with max_ctx > 16384, allocate sparse KV + IndexPool in 16,384-token VRAM increments
    // (reclaiming adaptive vcache slabs as KV grows). On multi-GPU static EP (no vcache slabs), reserve full max_ctx KV upfront so 128K stays in VRAM.
    const double per_pos = 12.0 * 1152.0 + 12.0 * 512.0 + 12.0 * 64.0;  // 20,736 B per token across 12 sparse layers
    int init_slots = max_ctx;
    if (!multi_gpu && !getenv("HY4_KV_NO_GROW") && total < (48ull << 30) && max_ctx > 16384) {
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
        // 16 GB+ cards default to Tc = 8192 when ctx >= 8192 (halves PCIe passes 32->16 at 128K; capped to 4096 for <=4K sessions to maximize expert cache!).
        // Multi-GPU 32 GB static EP at >=96K context uses Tc = 2048 so 100% of SQ-64 experts + full 128K KV fit in HBM2 (e.g. 2x V100 32 GB).
        int c = (multi_gpu && max_ctx >= 98304 && tot < 40.0) ? 2048 : ((tot >= 15.0) ? ((max_ctx >= 8192) ? 8192 : 4096) : (tot >= 11.0 ? 2048 : 1024));
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
        const double host_bounce = (tot < 48.0) ? ((tot >= 20.0) ? 0.82 : 0.55) : 0.0;  // 2 x 384 MiB (>= 20 GiB) or 2 x 256 MiB pinned prefill bounce ring
        snprintf(b, sizeof b, "%.2f", host_kv + host_emb + host_bounce + 0.3);
        hy4_setenv("HY4_HOST_NEED_GB", b, 1);
    }
    if (!getenv("HY4_RESERVE_GB")) {
        const double kv_res = (mask & 1) ? 0.0 : (init_slots < max_ctx ? kv_init_gb : (dev_pos ? dev_pos * per_pos / GB : (rows + pivg)));
        const double pf_res = (tot >= 21.5 && tot < 64.0 && mask == 0) ? 0.82 : ((tot >= 10.0 && tot < 64.0 && mask == 0) ? 0.56 : 0.0);  // 2 x 384 MiB (>= 21.5 GiB) or 2 x 256 MiB device scratch for prefill streaming
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
// RAM this process may use: min(MemTotal, the container's cgroup limit), in GiB (matching host_mem_gb())
inline double ram_limit_gb() {
    double gb = host_mem_gb();
#ifndef _WIN32
    const char* files[2] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    for (const char* fn : files) {
        FILE* f = fopen(fn, "r");
        if (!f) continue;
        double v = 0;
        char buf[64] = {0};
        if (fgets(buf, sizeof buf, f) && sscanf(buf, "%lf", &v) == 1 && v > 1e9 && v < 1e15) gb = std::min(gb, v / 1073741824.0);
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
    // Adaptive VRAM expert cache (single-GPU cpu_mode): when host RAM fits all experts (e.g. 64 GB+ Hero cards), use 100% dynamic slots;
    // when multi-GPU (--gpus 0,1...), distribute static experts across all GPUs (e.g. 2x V100 32 GB = 64 GB HBM2).
    const double host_gb = host_mem_gb(), lim_gb = ram_limit_gb();
    const bool cgroup_limited = (lim_gb < host_gb - 1.0);
    double ram_budget_gb = 0.0;
    if (cpu_mode) {
        if (getenv("HY4_RAM_BUDGET_GB")) ram_budget_gb = atof(getenv("HY4_RAM_BUDGET_GB"));
        else {
            const double need = getenv("HY4_HOST_NEED_GB") ? atof(getenv("HY4_HOST_NEED_GB")) : 0.0;
            const double vis_extra = (cgroup_limited && g.has_vision) ? 1.5 : 0.0;  // vision shard (0.93 GB) + serve/server.py RSS inside cgroup limit
            ram_budget_gb = lim_gb - 3.5 - need - 1.5 - vis_extra;  // OS and runtime, host KV / kv_b copies, engram cache + pinned staging
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
    double dyn_frac = (ng == 1) ? 1.0 : 0.0;
    if (ng == 1) {
        if (getenv("HY4_VCACHE_FRAC")) {
            dyn_frac = std::max(0.0, std::min(1.0, atof(getenv("HY4_VCACHE_FRAC"))));
        } else if (cpu_mode && all_al_bytes > ram_budget_bytes && vram_budget > 0) {
            const uint64_t overflow = all_al_bytes - ram_budget_bytes;
            const double static_share = std::min(0.75, (double)overflow / (double)vram_budget);
            dyn_frac = std::max(0.25, 1.0 - static_share);
        }
    }
    size_t static_budget = vram_budget;
    int dyn_slots = 0;
    size_t dyn_bytes = 0;
    if (ng == 1 && cpu_mode && dyn_frac > 0.01 && vram_budget >= 2 * max_body_sz && max_body_sz > 0) {
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
        const size_t preserve = (size_t)960 << 20;  // 960 MiB for peer moe_ws (604 MiB at Tc=4096) + buffers + runtime
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
    if (ng == 1 && cpu_mode && dyn_frac > 0.01 && max_body_sz > 0 && vram_budget > vb + dyn_bytes + max_body_sz) {
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
        const double budget_gb = ram_budget_gb;
        const uint64_t budget = ram_budget_bytes;
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
    const bool uma_mapped = gpu_is_uma() && getenv("HY4_UMA_MAPPED") && atoi(getenv("HY4_UMA_MAPPED")) != 0;
    const uint64_t trunk_al = (g.trunk_bytes + 4095) & ~4095ull;
    size_t fr0 = 0, tot0 = 0;
    gpu_mem_avail(fr0, tot0);
#if !HY4_CUDA
    // On UMA / iGPU (780M gfx1103, 890M gfx1150, Strix Halo gfx1151) with a small BIOS carveout (512 MB - 4 GB),
    // automatically allocate trunk & experts via hipHostMallocCoherent | hipHostMallocMapped when carveout VRAM is too small.
    auto hip_uma_alloc_or_malloc = [](uint8_t** d_out, uint8_t** h_out, uint64_t sz_al, bool force_uma) -> bool {
        strata::hip_uma::GpuBuffer b;
        if (strata::hip_uma::alloc_gpu_buffer(b, (size_t)sz_al, gpu_is_uma(), force_uma) != hipSuccess) return false;
        *d_out = b.dev_ptr;
        if (h_out) *h_out = b.host_ptr;
        return true;
    };
#endif
    const bool want_host_embed = !uma_mapped && !getenv("HY4_EMBED_VRAM") && (tot0 < (48ull << 30)) && g.man.T.count("model.embed_tokens.weight");
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
#if !HY4_CUDA
        if (gpu_is_uma()) {
            uint8_t* hp = nullptr;
            if (!hip_uma_alloc_or_malloc(&g.d_trunk, &hp, trunk_al, uma_mapped)) {
                fprintf(stderr, "trunk UMA alloc failed (%.2f GB)\n", trunk_al / 1e9);
                return false;
            }
            if (hp) {
                read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, hp, &g.t_load_trunk, "trunk(uma-coherent)");
            } else {
                stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
            }
        } else {
            HCK(hipMalloc((void**)&g.d_trunk, trunk_al));
            stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
        }
#else
        if (uma_mapped) {
            void* hp = nullptr;
            HCK(hipHostMalloc(&hp, trunk_al, hipHostMallocMapped));
            read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, (uint8_t*)hp, &g.t_load_trunk, "trunk(uma-mapped)");
            HCK(hipHostGetDevicePointer((void**)&g.d_trunk, hp, 0));
        } else {
            HCK(hipMalloc((void**)&g.d_trunk, trunk_al));
            stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
        }
#endif
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
        if (gpu_is_uma() && fr > trunk_al) fr -= trunk_al;
        double reserve_gb = getenv("HY4_RESERVE_GB") ? atof(getenv("HY4_RESERVE_GB")) : 6.0;
        double budget = (double)fr - reserve_gb * 1073741824.0;
        if (getenv("HY4_EXPERT_VRAM_GB")) budget = atof(getenv("HY4_EXPERT_VRAM_GB")) * 1073741824.0;
        const uint64_t sec4_al = (g.sec4_bytes + 4095) & ~4095ull;
        bool loaded_full = false;
        if (g.peer_devs.empty() && budget >= (double)sec4_al) {
#if !HY4_CUDA
            if (gpu_is_uma()) {
                uint8_t* hp = nullptr;
                if (hip_uma_alloc_or_malloc(&g.d_sec4, &hp, sec4_al, uma_mapped)) {
                    if (hp) {
                        read_file_parallel(path, h.sections[4].file_offset, g.sec4_bytes, hp, &g.t_load_experts, "experts(uma-coherent)");
                        g.h_arena = hp;
                    } else {
                        stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
                    }
                    loaded_full = true;
                }
            } else {
                if (hipMalloc((void**)&g.d_sec4, sec4_al) == hipSuccess) {
                    stream_to_device(path, h.sections[4].file_offset, g.sec4_bytes, g.d_sec4, &g.t_load_experts, "experts");
                    loaded_full = true;
                } else {
                    (void)hipGetLastError();
                    fprintf(stderr, "[load] full expert hipMalloc (%.2f GB) failed; falling back to tiered loading\n", sec4_al / 1e9);
                    size_t fr2 = 0, tot2 = 0;
                    HCK(hipMemGetInfo(&fr2, &tot2));
                    budget = (double)fr2 - reserve_gb * 1073741824.0;
                }
            }
#else
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
#endif
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

// Overlay distilled DFly2 weights directly into `g.d_trunk` (never touches target model weights).
// Supports BOTH flat override formats: `DFLYOVR1` (uint16 nlen, uint64 blen, name, payload) and `DFOV0001` (uint32 nlen, name, uint64 blen, payload).
inline bool load_dfly_override(GpuModel& g, const std::string& path, std::string& err) {
    if (path.empty()) return true;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open dfly-override: " + path; return false; }
    char magic[8] = {};
    uint32_t count = 0;
    if (fread(magic, 1, 8, f) != 8 || fread(&count, 4, 1, f) != 1 || count == 0 || count > 2048) {
        fclose(f);
        err = "invalid dfly-override header (expected DFLYOVR1 or DFOV0001): " + path;
        return false;
    }
    const bool is_v1 = (std::memcmp(magic, "DFLYOVR1", 8) == 0);
    const bool is_dfov = (std::memcmp(magic, "DFOV0001", 8) == 0);
    if (!is_v1 && !is_dfov) {
        fclose(f);
        err = "invalid dfly-override magic (expected DFLYOVR1 or DFOV0001): " + path;
        return false;
    }
    double t0 = now_s();
    size_t applied = 0, total_bytes = 0;
    std::vector<uint8_t> buf;
    for (uint32_t i = 0; i < count; ++i) {
        std::string name;
        uint64_t blen = 0;
        if (is_v1) {
            uint16_t nlen = 0;
            if (fread(&nlen, 2, 1, f) != 1 || fread(&blen, 8, 1, f) != 1 || nlen == 0 || nlen > 1024) { fclose(f); err = "truncated DFLYOVR1 entry"; return false; }
            name.resize(nlen);
            if (fread(&name[0], 1, nlen, f) != nlen) { fclose(f); err = "truncated DFLYOVR1 name"; return false; }
        } else {
            uint32_t nlen = 0;
            if (fread(&nlen, 4, 1, f) != 1 || nlen == 0 || nlen > 1024) { fclose(f); err = "bad DFOV0001 name length"; return false; }
            name.resize(nlen);
            if (fread(&name[0], 1, nlen, f) != nlen || fread(&blen, 8, 1, f) != 1) { fclose(f); err = "truncated DFOV0001 entry"; return false; }
        }
        if (name.rfind("model.dfly_layers.", 0) != 0) {
            fclose(f);
            err = "refusing non-dfly tensor in override: " + name;
            return false;
        }
        buf.resize((size_t)blen);
        if (fread(buf.data(), 1, (size_t)blen, f) != (size_t)blen) { fclose(f); err = "truncated dfly-override payload for " + name; return false; }
        auto it = g.man.T.find(name);
        if (it == g.man.T.end()) {
            fclose(f);
            err = "dfly-override tensor not in manifest: " + name;
            return false;
        }
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
    fprintf(stderr, "[dfly-override] applied %zu/%u tensors (%.2f MB, %s) from %s in %.2f s\n",
            applied, count, total_bytes / 1e6, is_v1 ? "DFLYOVR1" : "DFOV0001", path.c_str(), now_s() - t0);
    return applied > 0;
}
inline bool apply_dfly_override(GpuModel& g, const std::string& path, std::string& err) { return load_dfly_override(g, path, err); }

}  // namespace hy4
