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
#include "strata/platform/cgroup_mem.hpp"
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
    bool can_p2p = false;
    uint8_t* d_sec4 = nullptr;
    uint64_t vram_bytes = 0, n_experts = 0;
    std::vector<ExpertRef> experts;  // [41*512], tier=0 on this peer, tier=4 elsewhere
};

struct GpuModel {
    Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    int primary_gpu = 0;
    int planned_chunk = 2048;
    bool has_vision = false;
    bool draft_enabled = false;
    double vram_experts_gb_cap = -1.0;  // >= 0: --vram-experts-gb total cap across all selected GPUs (GiB)
    double ram_experts_gb_cap = -1.0;   // >= 0: --ram-experts-gb host RAM cap (GiB)
    std::string layer_split_spec;       // "" = default EP, "auto" or "K1,K2,..." = layer-partitioned multi-GPU
    std::vector<int> layer_split_counts;
    std::vector<int> layer_owner;       // [42] -> GPU ordinal 0..ng-1 when layer_split is active
    std::vector<int> peer_devs;   // secondary GPU device ids for multi-GPU expert split (--gpus 0,1,...)
    std::vector<PeerGpu> peers;
#ifdef STRATA_HY4_DISTRIBUTED
    int expert_shard = -1;
#endif
    uint8_t* d_trunk = nullptr;   // Section 1
    uint8_t* h_trunk_mapped = nullptr;  // pinned mapped host trunk fallback on 4 GB / tight 6 GB discrete GPUs
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

#ifdef STRATA_HY4_DISTRIBUTED
// The directory stores hotness order within each layer. Alternate ranks so both
// GPUs receive a mix of hot and cold experts without changing routed expert IDs.
inline bool load_expert_shard(GpuModel& g, const std::string& path, size_t budget, std::string& err) {
    const auto& dir = g.rd.expert_directory();
    const auto& section = g.rd.header().sections[4];
    constexpr size_t count = 41 * 512;
    if (dir.size() != count) { err = "Hy4 expert directory has the wrong geometry"; return false; }
    g.experts.assign(count, ExpertRef{0, 0, 0, 0, 4});
    g.expert_rank.assign(count, -1);
    uint64_t bytes = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& e = dir[i];
        const size_t idx = (i / 512) * 512 + e.expert_id;
        if (e.expert_id >= 512 || g.expert_rank[idx] != -1) { err = "invalid Hy4 expert ID"; return false; }
        if (e.dn_offset >= e.body_bytes || e.body_offset > section.byte_length || e.body_bytes > section.byte_length - e.body_offset) {
            err = "Hy4 expert body lies outside its data section";
            return false;
        }
        g.expert_rank[idx] = int(i % 512);
        if ((i & 1) != size_t(g.expert_shard)) continue;
        g.experts[idx] = ExpertRef{bytes, e.dn_offset, e.gu_code_id, e.dn_code_id, 0};
        bytes += (uint64_t(e.body_bytes) + 4095) & ~4095ull;
    }
    if (!bytes || bytes > budget) { err = "Hy4 expert shard exceeds the local GPU memory budget"; return false; }
    const hipError_t allocation = hipMalloc((void**)&g.d_sec4, bytes);
    if (allocation != hipSuccess) { err = hipGetErrorString(allocation); return false; }
    const double started = now_s();
    std::atomic<bool> failed{false};
    std::atomic<size_t> next{size_t(g.expert_shard)};
    std::vector<std::jthread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back([&] {
      try {
        if (hipSetDevice(g.primary_gpu) != hipSuccess) { failed = true; return; }
        AlignedFile file;
        if (!file.open(path, false)) { failed = true; return; }
        std::vector<uint8_t> staging;
        for (size_t i; !failed && (i = next.fetch_add(2)) < count;) {
            const auto& e = dir[i];
            const auto& local = g.experts[(i / 512) * 512 + e.expert_id];
            staging.resize(e.body_bytes);
            if (file.read_at(section.file_offset + e.body_offset, staging.data(), staging.size()) != staging.size() ||
                hipMemcpy(g.d_sec4 + local.body_off, staging.data(), staging.size(), hipMemcpyHostToDevice) != hipSuccess) {
                failed = true;
                break;
            }
        }
      } catch (const std::exception&) { failed = true; }
    });
    for (auto& thread : threads) thread.join();
    if (failed) { hipFree(g.d_sec4); g.d_sec4 = nullptr; err = "Hy4 expert shard loading failed"; return false; }
    g.n_vram_experts = count / 2;
    g.vram_expert_bytes = bytes;
    g.t_load_experts = now_s() - started;
    fprintf(stderr, "[load] rank %d: %llu local experts, %.2f GB in %.1f s\n", g.expert_shard,
            (unsigned long long)g.n_vram_experts, bytes / 1e9, g.t_load_experts);
    return true;
}
#endif

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

inline double wddm_vram_floor_gb(size_t total_bytes) {
    const double tot_gib = (double)total_bytes / 1073741824.0;
#ifdef _WIN32
    // Hard Windows WDDM free-VRAM floor so desktop compositor, cuBLAS, and driver context never page or OOM (#6528):
    // >= 12 GB cards keep >= 1.0 GiB free; 6-8 GB cards keep >= 600 MiB; 4 GB cards keep >= 450 MiB.
    if (tot_gib >= 11.0) return 1.074;  // 1.00 GiB
    if (tot_gib >= 5.5)  return 0.644;  // 600 MiB
    return 0.483;                       // 450 MiB
#else
    if (tot_gib >= 11.0) return 0.50;
    if (tot_gib >= 5.5)  return 0.35;
    return 0.25;
#endif
}

inline int plan_memory(int max_ctx, int chunk, size_t total, const std::string& model_path, bool multi_gpu = false, bool has_vision = false, int n_peers = 0, size_t min_peer_total = 0, size_t free_bytes = 0, bool has_draft = true) {
    const double GB = 1e9, tot = (double)total / GB;
    const double free_gb = (free_bytes > 0 && free_bytes <= total) ? ((double)free_bytes / GB) : tot;
    const double eff_tot = std::min(tot, free_gb + 0.35);
    const double min_card_gb = (multi_gpu && min_peer_total > 0) ? std::min(tot, (double)min_peer_total / GB) : tot;
    const double wddm_floor = wddm_vram_floor_gb(total);
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
    // Compressed Q8 MLA KV cache (656 B/slot vs 1152 B/slot FP16, lossless FP16 RoPE):
    // Available via --kv q8 / HY4_KV_Q8=1 (saves 780 MiB at 128K); default remains lossless FP16.
    bool kv_q8 = false;
    if (getenv("HY4_KV_Q8")) {
        kv_q8 = atoi(getenv("HY4_KV_Q8")) != 0;
    }
    const double kv_row_b = kv_q8 ? 656.0 : 1152.0;
    // Elastic KV initial sizing on single discrete cards (< 48 GB) so idle KV never starves hot experts on 4-12 GB cards:
    //   <= 8 GB -> init 4096 (step 4096); 12 GB -> init 8192 (step 8192); 16..32 GB -> init 16384 (step 16384).
    const double per_pos = 12.0 * kv_row_b + 12.0 * 512.0 + 12.0 * 64.0;
    int init_slots = max_ctx;
    int kv_step = 16384;
    if (!multi_gpu && !getenv("HY4_KV_NO_GROW") && total < (48ull << 30)) {
        const int target_init = (tot <= 9.0) ? 4096 : (tot <= 13.5 ? 8192 : 16384);
        kv_step = target_init;
        if (max_ctx > target_init) {
            init_slots = target_init;
            if (getenv("HY4_KV_INIT_SLOTS")) init_slots = std::max(2048, std::min(max_ctx, atoi(getenv("HY4_KV_INIT_SLOTS"))));
            else {
                char b[16];
                snprintf(b, sizeof b, "%d", init_slots);
                hy4_setenv("HY4_KV_INIT_SLOTS", b, 1);
            }
        }
        if (!getenv("HY4_KV_STEP_SLOTS")) {
            char bs[16];
            snprintf(bs, sizeof bs, "%d", kv_step);
            hy4_setenv("HY4_KV_STEP_SLOTS", bs, 1);
        }
    }
    const double kv_init_gb = (double)((init_slots + 8 + 15) / 16 * 16) * per_pos / GB;
    const double wkv = 0.03;  // single-layer reusable FP16 wk/wv prefill scratch (28 MiB)
    const double cublas_gb = HY4_CUDA ? 0.12 : 0.0;  // reusable side-stream FP16 dequant + cuBLAS workspace/handles
    const double dfly_gb = has_draft ? 0.28 : 0.0;  // DFly2 runtime KV/logit/graph buffers (allocated AFTER Runner::init)
    auto est_work_gb = [&](int c) {
        // Aliased work_arena (max(2*c*16384*4, moe_ws_bytes(c))) + qa_h/H1/x/z/etc (~0.195 MB/tok) + SWA rings (30*(512+c)*kv_row_b) + misc 0.22 GB
        const double arena = std::max(2.0 * c * 16384.0 * 4.0, (double)c * 16.0 * 1536.0 * 6.0) / GB;
        return arena + c * 0.195e6 / GB + 0.22 + 30.0 * (512.0 + c) * kv_row_b / GB;
    };
    if (chunk <= 0) {
        // #6614: on 16 GB discrete cards (tot < 20 GB), default chunk = 2048 saves ~0.8 GB VRAM (~300 hot experts) vs 4096 while keeping fast prefill.
        int c = (multi_gpu && max_ctx >= 98304 && tot < 40.0) ? 2048
              : (tot >= 44.0 ? ((max_ctx >= 8192) ? 8192 : 4096)
              : (tot >= 20.0 ? 4096 : (tot >= 11.5 ? 2048 : (tot >= 7.0 ? 1024 : 512))));
        if (multi_gpu && min_card_gb < 14.0 && c > 4096) c = 4096;
        if (multi_gpu && min_card_gb < 10.0 && c > 2048) c = 2048;
        const double work_cap = (tot >= 15.0) ? 0.22 : (tot >= 7.0 ? 0.16 : 0.14);
        while (c > 256 && est_work_gb(c) > work_cap * eff_tot) c >>= 1;
        while (c > 256 && trunk + est_work_gb(c) + kv_init_gb + wkv + cublas_gb + dfly_gb + wddm_floor > eff_tot) c >>= 1;
        // Preserve 100%-resident placement when a smaller chunk keeps all experts + full KV in VRAM (e.g. SQ-96 128K on 96 GB card)
        if (sec4_gb > 0.0 && c > 4096) {
            const double kv_full = (double)((max_ctx + 8 + 15) / 16 * 16) * per_pos / GB;
            if (trunk + sec4_gb + est_work_gb(c) + kv_full + wkv + cublas_gb + dfly_gb + wddm_floor > eff_tot &&
                trunk + sec4_gb + est_work_gb(c >> 1) + kv_full + wkv + cublas_gb + dfly_gb + wddm_floor <= eff_tot)
                c >>= 1;
        }
        chunk = c;
    }
    const double slots = (double)((max_ctx + 8 + 15) / 16 * 16);
    const double rows = 12.0 * slots * kv_row_b / GB;                                                       // full max_ctx sparse-layer KV rows
    const double pivg = 12.0 * (double)max_ctx * 512.0 / GB + 12.0 * (double)(max_ctx / 4 + 8) * 256.0 / GB;  // full max_ctx IndexPool iv/ig + pooled keys
    const double work_gb = est_work_gb(chunk);
    // Check if initial elastic KV fits in VRAM with WDDM floor + DFly2; on 4-6 GB cards (where trunk spills or barely fits), put KV in pinned host RAM.
    const double trunk_in_vram = (eff_tot < trunk + wddm_floor + 0.50) ? 0.0 : trunk;
    int mask = 0;
    if (trunk_in_vram + work_gb + pivg + rows + wkv + cublas_gb + dfly_gb + wddm_floor > eff_tot) mask |= 1;
    long dev_pos = 0;
    if ((mask & 1) && !getenv("HY4_KV_HOST")) {
        const double avail = eff_tot - trunk_in_vram - work_gb - wkv - cublas_gb - dfly_gb - wddm_floor;
        dev_pos = avail > 0 ? (long)(avail * GB / per_pos) / 1024 * 1024 : 0;
        if (dev_pos >= 4096 && dev_pos < max_ctx && trunk_in_vram > 0.0) {
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
        const double host_trunk = (trunk_in_vram == 0.0) ? trunk : 0.0;
        const double host_bounce = (tot < 48.0) ? ((tot >= 20.0) ? 0.82 : ((tot >= 10.0) ? 0.28 : 0.22)) : 0.0;
        snprintf(b, sizeof b, "%.2f", host_kv + host_emb + host_trunk + host_bounce + 0.35);
        hy4_setenv("HY4_HOST_NEED_GB", b, 1);
    }
    if (!getenv("HY4_RESERVE_GB")) {
        const double kv_res = (mask & 1) ? 0.0 : (init_slots < max_ctx ? kv_init_gb : (dev_pos ? dev_pos * per_pos / GB : (rows + pivg)));
        // Compact double-buffered d_scratch[0..1]: 2x96 MiB on 8 GB, 2x160 MiB on 10-12 GB, 2x256 MiB on 16 GB, 2x384 MiB on 24-32 GB!
        const double pf_res = (mask != 0 || tot >= 64.0) ? 0.0
                            : (tot >= 21.5 ? 0.82 : (tot >= 14.5 ? 0.56 : (tot >= 10.0 ? 0.34 : 0.22)));
        const double vis_res = (multi_gpu && has_vision) ? 1.15 : 0.0;
        const int eff_peers = std::max(n_peers, multi_gpu ? 1 : 0);
        const double peer_graph_res = multi_gpu ? ((double)eff_peers * ((double)chunk * 1536.0 * 4.0 / GB + 0.05) + 0.30) : 0.0;
        // Convert decimal GB -> GiB to match load_gpu_model's reserve_gb * 1073741824.0 without 7.4% silent waste!
        const double reserve_gib = (work_gb + kv_res + wkv + cublas_gb + dfly_gb + pf_res + vis_res + peer_graph_res + wddm_floor) / 1.073741824;
        char b[32];
        snprintf(b, sizeof b, "%.2f", reserve_gib);
        hy4_setenv("HY4_RESERVE_GB", b, 1);
    }
    fprintf(stderr, "[plan] ctx %d (init %d, kv=%s) chunk %d: buffers ~%.2f GB, KV+idx %.2f GB (128K full %.2f GB), dfly %.2f GB, floor %.2f GB, trunk %.2f GB%s on %.1f/%.1f GB -> host mask %d; reserve %s GiB\n",
            max_ctx, init_slots, kv_q8 ? "q8" : "fp16", chunk, work_gb, kv_init_gb, rows + pivg, dfly_gb, wddm_floor, trunk, host_embed ? " (embed in RAM)" : "", free_gb, tot, atoi(getenv("HY4_KV_HOST")), getenv("HY4_RESERVE_GB"));
    if (dev_pos) fprintf(stderr, "[plan] growable KV: up to %ld positions stay in VRAM, moves to pinned host memory if context exceeds that\n", dev_pos);
    return chunk;
}

inline double host_mem_gb();
// RAM this process may use: min(MemTotal, the tightest cgroup v1/v2 available ceiling crediting reclaimable file cache), in GiB
inline double ram_limit_gb(bool* out_cgroup_limited = nullptr) {
    const double host_gb = host_mem_gb();
#ifndef _WIN32
    const uint64_t host_bytes = strata::platform::cgroup_mem::gib_to_bytes(host_gb);
    const auto cg = strata::platform::cgroup_mem::probe_cgroup_room(host_bytes);
    if (out_cgroup_limited) *out_cgroup_limited = cg.limited;
    if (cg.limited && cg.available_bytes != strata::platform::cgroup_mem::kUnlimited) {
        return std::min(host_gb, static_cast<double>(cg.available_bytes) / 1073741824.0);
    }
    if (cg.limit_bytes != strata::platform::cgroup_mem::kUnlimited) {
        return std::min(host_gb, static_cast<double>(cg.limit_bytes) / 1073741824.0);
    }
#else
    if (out_cgroup_limited) *out_cgroup_limited = false;
#endif
    return host_gb;
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

// Available host RAM right now (MemAvailable / ullAvailPhys, capped by cgroup free headroom), in GiB
inline double ram_avail_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return (double)ms.ullAvailPhys / 1073741824.0;
    return 0.0;
#else
    double avail_kb = 0.0, free_kb = 0.0, buf_kb = 0.0, cache_kb = 0.0;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char ln[256];
        double v = 0.0;
        while (fgets(ln, sizeof ln, f)) {
            if (sscanf(ln, "MemAvailable: %lf kB", &v) == 1) avail_kb = v;
            else if (sscanf(ln, "MemFree: %lf kB", &v) == 1) free_kb = v;
            else if (sscanf(ln, "Buffers: %lf kB", &v) == 1) buf_kb = v;
            else if (sscanf(ln, "Cached: %lf kB", &v) == 1) cache_kb = v;
        }
        fclose(f);
    }
    double gb = (avail_kb > 0.0 ? avail_kb : (free_kb + buf_kb + cache_kb)) / 1048576.0;
    if (FILE* fm = fopen("/sys/fs/cgroup/memory.max", "r")) {
        char buf[64] = {0};
        double lim = 0.0;
        if (fgets(buf, sizeof buf, fm) && sscanf(buf, "%lf", &lim) == 1 && lim > 1e9 && lim < 1e15) {
            double cur = 0.0, inact = 0.0;
            if (FILE* fc = fopen("/sys/fs/cgroup/memory.current", "r")) {
                if (fscanf(fc, "%lf", &cur) != 1) cur = 0.0;
                fclose(fc);
            }
            if (FILE* fs = fopen("/sys/fs/cgroup/memory.stat", "r")) {
                char ln[256];
                double v = 0.0;
                while (fgets(ln, sizeof ln, fs))
                    if (sscanf(ln, "inactive_file %lf", &v) == 1) { inact = v; break; }
                fclose(fs);
            }
            const double cg_avail = std::max(0.0, lim - std::max(0.0, cur - inact)) / 1073741824.0;
            gb = (gb > 0.0) ? std::min(gb, cg_avail) : cg_avail;
        }
        fclose(fm);
    }
    if (FILE* fm = fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r")) {
        char buf[64] = {0};
        double lim = 0.0;
        if (fgets(buf, sizeof buf, fm) && sscanf(buf, "%lf", &lim) == 1 && lim > 1e9 && lim < 1e15) {
            double cur = 0.0, inact = 0.0;
            if (FILE* fc = fopen("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r")) {
                if (fscanf(fc, "%lf", &cur) != 1) cur = 0.0;
                fclose(fc);
            }
            if (FILE* fs = fopen("/sys/fs/cgroup/memory/memory.stat", "r")) {
                char ln[256];
                double v = 0.0;
                while (fgets(ln, sizeof ln, fs))
                    if (sscanf(ln, "total_inactive_file %lf", &v) == 1) { inact = v; break; }
                fclose(fs);
            }
            const double cg_avail = std::max(0.0, lim - std::max(0.0, cur - inact)) / 1073741824.0;
            gb = (gb > 0.0) ? std::min(gb, cg_avail) : cg_avail;
        }
        fclose(fm);
    }
    return gb;
#endif
}

#if HY4_CUDA || HY4_HIP_PEERS
// Returns false on Linux when two CUDA GPUs sit behind separate PCIe root ports (PHB / SYS) without
// NVLink / native P2P atomics. Avoiding cudaDeviceEnablePeerAccess + cudaMemcpyPeer on PHB/ACS boards
// (e.g. 2x RTX A4000) prevents faulting the peer GPU's context or hitting ~98 ms/copy PHB stalls.
inline bool p2p_topology_safe(int dev0, int dev1) {
    if (const char* fp = getenv("HY4_PEER_FORCE_P2P"); fp && atoi(fp) != 0) return true;
#if HY4_CUDA
    int atomic_ok = 0;
    if (cudaDeviceGetP2PAttribute(&atomic_ok, cudaDevP2PAttrNativeAtomicSupported, dev0, dev1) == cudaSuccess && atomic_ok) {
        return true;  // NVLink or true hardware P2P switch with native atomics
    }
    (void)cudaGetLastError();
#if defined(__linux__)
    auto ancestors = [](int d) -> std::vector<std::string> {
        char bdf[64] = {};
        if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), d) != cudaSuccess) { (void)cudaGetLastError(); return {}; }
        for (char* p = bdf; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p = (char)(*p - 'A' + 'a');
        std::string sys = std::string("/sys/bus/pci/devices/") + bdf;
        char resolved[4096] = {};
        if (!realpath(sys.c_str(), resolved)) return {};
        std::vector<std::string> parts;
        std::string s(resolved);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t nxt = s.find('/', pos);
            if (nxt == std::string::npos) nxt = s.size();
            if (nxt > pos) parts.push_back(s.substr(pos, nxt - pos));
            pos = nxt + 1;
        }
        if (!parts.empty()) parts.pop_back();
        return parts;
    };
    const auto a0 = ancestors(dev0), a1 = ancestors(dev1);
    if (!a0.empty() && !a1.empty()) {
        size_t common_bridges = 0;
        const size_t n = std::min(a0.size(), a1.size());
        for (size_t i = 0; i < n && a0[i] == a1[i]; ++i) {
            if (a0[i].find(':') != std::string::npos && a0[i].find('.') != std::string::npos) ++common_bridges;
        }
        if (common_bridges == 0) return false;  // PHB / NODE / SYS (no shared PCIe switch)
    }
#endif
#endif
    return true;
}

// Verify that peer access actually moves data both ways AND is fast (not a broken/stalling PHB):
// some PCIe host-bridge ("PHB") pairs (e.g. 2x RTX A4000 behind CPU root complex) report
// hipDeviceCanAccessPeer==1 and succeed on hipDeviceEnablePeerAccess, yet hipMemcpyPeerAsync either
// leaves destination buffers zeroed/corrupt OR stalls at ~98 ms/copy (8.1 s/step across 41 layers vs
// ~25 us for portable host-pinned DMA). Test the exact async peer copy path both ways and require
// both bit-exact data integrity and <= 2.0 ms per 64 KiB roundtrip.
inline bool p2p_probe_pair(int dev0, int dev1, double* out_ms = nullptr) {
    constexpr size_t kN = 1 << 14;  // 64 KiB
    std::vector<uint32_t> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 0x9e3779b9u ^ (uint32_t)(i * 2654435761u);
    uint32_t* d[2] = {nullptr, nullptr};
    hipStream_t st[2] = {nullptr, nullptr};
    const int dev[2] = {dev0, dev1};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        if (hipSetDevice(dev[k]) != hipSuccess ||
            hipMalloc((void**)&d[k], kN * 4) != hipSuccess ||
            hipStreamCreate(&st[k]) != hipSuccess) {
            (void)hipGetLastError();
            tested = false;
        }
    }
    double total_copy_ms = 0.0;
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        if (hipSetDevice(dev[from]) != hipSuccess ||
            hipMemcpy(d[from], h.data(), kN * 4, hipMemcpyHostToDevice) != hipSuccess) {
            tested = false;
            break;
        }
        if (hipSetDevice(dev[to]) != hipSuccess ||
            hipMemset(d[to], 0, kN * 4) != hipSuccess) {
            tested = false;
            break;
        }
        const double t0 = now_s();
        if (hipMemcpyPeerAsync(d[to], dev[to], d[from], dev[from], kN * 4, st[to]) != hipSuccess ||
            hipStreamSynchronize(st[to]) != hipSuccess) {
            tested = false;
            break;
        }
        total_copy_ms += (now_s() - t0) * 1000.0;
        if (hipMemcpy(back.data(), d[to], kN * 4, hipMemcpyDeviceToHost) != hipSuccess) {
            tested = false;
            break;
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
        if (total_copy_ms > 5.0) break;  // early-exit slow PHB stall before wasting a second direction
    }
    for (int k = 0; k < 2; ++k) {
        (void)hipSetDevice(dev[k]);
        if (st[k]) (void)hipStreamDestroy(st[k]);
        if (d[k]) (void)hipFree(d[k]);
    }
    (void)hipSetDevice(dev0);
    if (out_ms) *out_ms = total_copy_ms;
    if (!tested) {
        (void)hipGetLastError();
        return false;
    }
    if (!ok) return false;
    const char* force_p2p = getenv("HY4_PEER_FORCE_P2P");
    if (!(force_p2p && atoi(force_p2p) != 0) && total_copy_ms > 2.0) return false;
    return true;
}
#endif

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
    bool cgroup_limited = false;
    const double lim_gb = ram_limit_gb(&cgroup_limited), avail_gb = ram_avail_gb();
    double ram_budget_gb = 0.0;
    {
        // Compute safe RAM budget on BOTH Windows and Linux so swap is 0 (#6556, #6528):
        // reserve OS + WDDM commit headroom + Vision + Engram LRU + ColdStore + host KV/bounce.
        if (const char* rb = getenv("HY4_RAM_BUDGET_GB") ? getenv("HY4_RAM_BUDGET_GB") : getenv("STRATA_HY4_RAM_GB")) {
            ram_budget_gb = std::max(0.0, atof(rb));
        } else {
            const double need = getenv("HY4_HOST_NEED_GB") ? atof(getenv("HY4_HOST_NEED_GB")) : 0.0;
            const double def_eng_mib = (lim_gb <= 20.0) ? 128.0 : (lim_gb <= 40.0 ? 256.0 : (lim_gb <= 72.0 ? 512.0 : 850.0));
            const double engram_gb = getenv("HY4_ENGRAM_CACHE_MIB") ? std::max(0.0, atof(getenv("HY4_ENGRAM_CACHE_MIB")) / 1024.0) : (def_eng_mib / 1024.0);
            const double vis_extra = g.has_vision ? 1.25 : 0.0;  // Vision shard (0.93 GB) + decode scratch
#ifdef _WIN32
            const double os_tot_res = std::max(4.5, 0.10 * lim_gb);
            const double os_avail_res = std::max(3.0, 0.06 * lim_gb);
            const double by_total = lim_gb - os_tot_res - need - engram_gb - 0.85 - vis_extra;
            const double by_avail = avail_gb - os_avail_res - need - engram_gb - 0.65 - vis_extra;
#else
            const double os_tot_res = std::max(3.5, 0.06 * lim_gb);
            const double os_avail_res = std::max(2.2, 0.04 * lim_gb);
            const double by_total = lim_gb - os_tot_res - need - engram_gb - 0.65 - vis_extra;
            const double by_avail = avail_gb - os_avail_res - need - engram_gb - 0.50 - vis_extra;
#endif
            ram_budget_gb = std::max(0.0, (avail_gb > 0.5) ? std::min(by_total, by_avail) : by_total);
        }
        if (g.ram_experts_gb_cap >= 0.0) {
            ram_budget_gb = std::min(ram_budget_gb, g.ram_experts_gb_cap);
        }
    }
    const uint64_t ram_budget_bytes = strata::platform::cgroup_mem::gib_to_bytes(ram_budget_gb);
    uint64_t all_al_bytes = 0;
    size_t max_body_sz = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint64_t sz = al(dir[i].body_bytes);
        all_al_bytes += sz;
        max_body_sz = std::max<size_t>(max_body_sz, (size_t)sz);
    }
    // Cross-platform (Windows + Linux + macOS) automatic SSD cold-expert tier (#6556, #6528):
    // whenever host experts exceed safe MemAvailable / ullAvailPhys, auto-enable cpu_mode so ColdStore streams cold tail from SSD with 0 swap!
    const uint64_t est_host_need = all_al_bytes > vram_budget ? (all_al_bytes - vram_budget) : 0;
    if (!cpu_mode && est_host_need > ram_budget_bytes) {
        cpu_mode = true;
        fprintf(stderr, "[load] host experts %.1f GB exceed safe RAM budget %.1f GB (avail %.1f / %.1f GiB); auto-enabling 4-tier CPU + SSD cold-expert streaming\n",
                est_host_need / 1e9, ram_budget_gb, avail_gb, lim_gb);
    }
    double dyn_frac = (ng == 1) ? 1.0 : 0.0;
    if (ng == 1) {
        if (getenv("HY4_VCACHE_FRAC")) {
            dyn_frac = std::max(0.0, std::min(1.0, atof(getenv("HY4_VCACHE_FRAC"))));
        } else if (cpu_mode && all_al_bytes > ram_budget_bytes && vram_budget > 0) {
            const uint64_t overflow = all_al_bytes - ram_budget_bytes;
            const double static_share = std::min(0.75, (double)overflow / (double)vram_budget);
            dyn_frac = std::max(0.25, 1.0 - static_share);
        } else if (cpu_mode && N > 0 && vram_budget >= (3ull << 30)) {
            // #6614: Mixed-precision SQ packs (e.g. SQ-64: S6/S8 hot experts 3.7 MB vs avg 2.67 MB) waste ~25-39% VRAM when 100% uniform-slot vcache is used.
            // Lowering the threshold from 12 GiB to 3 GiB enables tight static VRAM (65%, 0% padding) + compact adaptive vcache slabs (35%, sized to tier-2 max)
            // on 16 GB discrete cards like RX 9070 XT (~6-8 GiB expert VRAM), adding ~25% more VRAM-resident experts and eliminating cold-start misses on rank 0..N.
            const double avg_body_sz = (double)all_al_bytes / (double)N;
            if ((double)max_body_sz > 1.20 * avg_body_sz) dyn_frac = 0.35;
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
    std::vector<bool> peer_p2p(g.peer_devs.size(), false);
#if HY4_CUDA || HY4_HIP_PEERS
    const bool force_no_p2p = getenv("HY4_PEER_NO_P2P") && atoi(getenv("HY4_PEER_NO_P2P")) != 0;
    // Per-peer reserve covers exact moe_ws_bytes(Tc) + z_h/idx/w/zout + 41-layer CUDA Graph capture + driver context headroom
    const size_t chunk_ws = (size_t)std::max(g.planned_chunk, 1024) * 16ull * 1536ull * 6ull +
                            (size_t)std::max(g.planned_chunk, 1024) * 1536ull * 10ull + (512ull << 20);
    const char* reserve_mb = getenv("HY4_PEER_RESERVE_MB");
    const size_t preserve = (reserve_mb && atoi(reserve_mb) > 0)
        ? ((size_t)atoi(reserve_mb) << 20)
        : std::max<size_t>((size_t)1024 << 20, chunk_ws);
    for (size_t p = 0; p < g.peer_devs.size(); ++p) {
        const int pd = g.peer_devs[p];
        HCK(hipSetDevice(pd));
        size_t pfr = 0, ptot = 0;
        HCK(hipMemGetInfo(&pfr, &ptot));
        vcap[1 + p] = pfr > preserve ? (uint64_t)(pfr - preserve) : 0;
        if (vcap[1 + p] == 0) {
            fprintf(stderr, "[load] peer GPU%d free %.2f GiB <= required reserve %.2f GiB (chunk=%d); placing 0 experts on GPU%d\n",
                    pd, pfr / 1073741824.0, preserve / 1073741824.0, g.planned_chunk, pd);
        }
        int can01 = 0, can10 = 0;
        if (!force_no_p2p &&
            hipDeviceCanAccessPeer(&can01, g.primary_gpu, pd) == hipSuccess && can01 &&
            hipDeviceCanAccessPeer(&can10, pd, g.primary_gpu) == hipSuccess && can10) {
            if (!p2p_topology_safe(g.primary_gpu, pd)) {
                fprintf(stderr,
                        "hy4: gpu %d <-> gpu %d: GPUs sit behind a PCIe host bridge (PHB/SYS, no shared PCIe switch or NVLink); "
                        "using portable host-pinned bounce (HY4_PEER_FORCE_P2P=1 overrides)\n",
                        g.primary_gpu, pd);
                peer_p2p[p] = false;
                continue;
            }
            HCK(hipSetDevice(g.primary_gpu));
            hipError_t e0 = hipDeviceEnablePeerAccess(pd, 0);
            if (e0 != hipSuccess) (void)hipGetLastError();
            HCK(hipSetDevice(pd));
            hipError_t e1 = hipDeviceEnablePeerAccess(g.primary_gpu, 0);
            if (e1 != hipSuccess) (void)hipGetLastError();
            // Treat both hipSuccess and PeerAccessAlreadyEnabled as P2P enabled, then verify data integrity
            const bool ok0 = (e0 == hipSuccess || e0 == hipErrorPeerAccessAlreadyEnabled);
            const bool ok1 = (e1 == hipSuccess || e1 == hipErrorPeerAccessAlreadyEnabled);
            bool p2p_ok = ok0 && ok1;
            double p2p_ms = 0.0;
            if (p2p_ok && !p2p_probe_pair(g.primary_gpu, pd, &p2p_ms)) {
                fprintf(stderr,
                        "hy4: gpu %d <-> gpu %d: peer access enabled but round-trip async P2P probe failed "
                        "(%.2f ms; PCIe host bridge / ACS); falling back to portable host-pinned bounce\n",
                        g.primary_gpu, pd, p2p_ms);
                (void)hipSetDevice(g.primary_gpu);
                (void)hipDeviceDisablePeerAccess(pd);
                (void)hipGetLastError();
                (void)hipSetDevice(pd);
                (void)hipDeviceDisablePeerAccess(g.primary_gpu);
                (void)hipGetLastError();
                p2p_ok = false;
            }
            peer_p2p[p] = p2p_ok;
        }
    }
    HCK(hipSetDevice(g.primary_gpu));
#endif
    if (g.vram_experts_gb_cap >= 0.0) {
        const uint64_t total_cap_b = (uint64_t)(g.vram_experts_gb_cap * 1073741824.0);
        uint64_t sum_vcap = 0;
        for (size_t gi = 0; gi < ng; ++gi) sum_vcap += vcap[gi];
        if (sum_vcap > total_cap_b) {
            const double scale = sum_vcap > 0 ? ((double)total_cap_b / (double)sum_vcap) : 0.0;
            for (size_t gi = 0; gi < ng; ++gi) {
                vcap[gi] = (uint64_t)((double)vcap[gi] * scale);
            }
            fprintf(stderr, "[load] --vram-experts-gb %.2f cap applied across %zu GPU(s):", g.vram_experts_gb_cap, ng);
            for (size_t gi = 0; gi < ng; ++gi) {
                const int dev = gi == 0 ? g.primary_gpu : g.peer_devs[gi - 1];
                fprintf(stderr, " GPU%d<=%.2fGiB", dev, vcap[gi] / 1073741824.0);
            }
            fprintf(stderr, "\n");
        }
    }
    // Compute optional --layer-split auto|K1,K2,... layer ownership across GPUs so each layer's experts stay contiguous
    // on one GPU (eliminating per-layer cross-EPYC-root synchronization barriers on non-NVLink / unequal multi-GPU rigs!).
    g.layer_split_counts.clear();
    g.layer_owner.clear();
    if (ng >= 2 && !g.layer_split_spec.empty()) {
        std::vector<int> counts(ng, 0);
        if (g.layer_split_spec == "auto") {
            std::vector<double> wt(ng, 1.0);
            double wsum = 0.0;
            for (size_t gi = 0; gi < ng; ++gi) {
                wt[gi] = std::max(0.25, (double)vcap[gi] / 1073741824.0);
                wsum += wt[gi];
            }
            int assigned = 0;
            for (size_t gi = 0; gi < ng; ++gi) {
                counts[gi] = std::max(1, (int)std::lround(42.0 * wt[gi] / std::max(wsum, 1e-6)));
                assigned += counts[gi];
            }
            while (assigned > 42) {
                size_t best = 0;
                for (size_t gi = 1; gi < ng; ++gi) if (counts[gi] > counts[best]) best = gi;
                if (counts[best] > 1) { --counts[best]; --assigned; } else break;
            }
            while (assigned < 42) {
                size_t best = 0;
                for (size_t gi = 1; gi < ng; ++gi) if (wt[gi] > wt[best]) best = gi;
                ++counts[best]; ++assigned;
            }
        } else {
            size_t pos = 0, gi = 0;
            int sum = 0;
            while (pos < g.layer_split_spec.size() && gi < ng) {
                int v = std::max(0, atoi(g.layer_split_spec.c_str() + pos));
                counts[gi++] = v;
                sum += v;
                size_t q = g.layer_split_spec.find(',', pos);
                if (q == std::string::npos) break;
                pos = q + 1;
            }
            if (gi != ng || sum != 42) {
                fprintf(stderr, "[load] --layer-split %s invalid for %zu GPUs (must sum to 42); ignoring\n", g.layer_split_spec.c_str(), ng);
                counts.clear();
            }
        }
        if (counts.size() == ng) {
            g.layer_split_counts = counts;
            g.layer_owner.assign(42, 0);
            int cur = 0;
            fprintf(stderr, "[load] --layer-split active across %zu GPUs:", ng);
            for (size_t gi = 0; gi < ng; ++gi) {
                const int dev = gi == 0 ? g.primary_gpu : g.peer_devs[gi - 1];
                fprintf(stderr, " GPU%d=%d layers[%d..%d]", dev, counts[gi], cur, cur + counts[gi] - 1);
                for (int k = 0; k < counts[gi] && cur < 42; ++k) g.layer_owner[cur++] = (int)gi;
            }
            fprintf(stderr, "\n");
        }
    }
    uint64_t hb = 0;
    const bool use_layer_owner = (g.layer_owner.size() == 42);
    for (size_t r = 0; r < 512; ++r) {
        bool any_placed = false;
        for (size_t L = 0; L < N / 512; ++L) {
            size_t i = L * 512 + r;
            uint64_t sz = al(dir[i].body_bytes);
            int best_g = -1;
            if (use_layer_owner) {
                const int pref = g.layer_owner[L + 1];
                if (pref >= 0 && (size_t)pref < ng && vcap[pref] > 0 && vb_gpu[pref] + sz <= vcap[pref]) {
                    best_g = pref;
                }
            }
            if (best_g < 0 && !use_layer_owner) {
                double best_frac = -1.0;
                for (size_t gi = 0; gi < ng; ++gi) {
                    if (vcap[gi] > 0 && vb_gpu[gi] + sz <= vcap[gi]) {
                        const double frac = (double)(vcap[gi] - vb_gpu[gi]) / (double)vcap[gi];
                        if (frac > best_frac) { best_frac = frac; best_g = (int)gi; }
                    }
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
        if (!any_placed && !use_layer_owner) break;
    }
    // Second pass for --layer-split: if any GPU still has free VRAM headroom after filling its owned layers, fill leftover hottest experts
    if (use_layer_owner) {
        for (size_t r = 0; r < 512; ++r) {
            bool any_placed = false;
            for (size_t L = 0; L < N / 512; ++L) {
                size_t i = L * 512 + r;
                if (owner[i] >= 0) continue;
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
                    tier[i] = (best_g == 0) ? 0 : 4;
                    off[i] = vb_gpu[best_g];
                    vb_gpu[best_g] += sz;
                    ++nv_gpu[best_g];
                    any_placed = true;
                }
            }
            if (!any_placed) break;
        }
    }
    uint64_t vb = vb_gpu[0];
    // Size adaptive vcache slots to the largest remaining non-VRAM (tier != 0) expert so static S6/S8 hot experts do not bloat dynamic slot stride!
    size_t vcache_slot_sz = max_body_sz;
    if (ng == 1 && cpu_mode && dyn_frac > 0.01 && vram_budget > vb) {
        size_t t2_max = 0;
        for (size_t i = 0; i < N; ++i)
            if (tier[i] != 0) t2_max = std::max<size_t>(t2_max, (size_t)al(dir[i].body_bytes));
        if (t2_max > 0 && t2_max < vcache_slot_sz) vcache_slot_sz = t2_max;
        dyn_slots = vcache_slot_sz > 0 ? (int)((vram_budget - vb) / vcache_slot_sz) : 0;
        dyn_bytes = (size_t)dyn_slots * vcache_slot_sz;
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
    g.vcache_slot_bytes = vcache_slot_sz;
    {
        size_t vf_s = 0, vt_s = 0;
        HCK(hipMemGetInfo(&vf_s, &vt_s));
        // Finer 64 MiB slabs on <= 13.5 GiB cards so KV growth / Vision reclaims only 64 MiB at a time and zero VRAM is lost to slab rounding!
        const size_t slab_bytes = (vt_s <= (size_t)(13.5 * 1073741824.0)) ? (64ull << 20) : (384ull << 20);
        g.vcache_slots_per_slab = vcache_slot_sz > 0 ? std::max(1, (int)(slab_bytes / vcache_slot_sz)) : 1;
    }
    int actual_dyn = 0;
    while (actual_dyn < dyn_slots) {
        const int take = std::min(g.vcache_slots_per_slab, dyn_slots - actual_dyn);
        uint8_t* slab = nullptr;
        if (hipMalloc((void**)&slab, (size_t)take * vcache_slot_sz) != hipSuccess) { (void)hipGetLastError(); break; }
        g.vcache_slabs.push_back(slab);
        actual_dyn += take;
    }
    dyn_slots = actual_dyn;
    dyn_bytes = (size_t)dyn_slots * vcache_slot_sz;
    g.vcache_nslots = dyn_slots;
    const uint64_t total_vram_arena = vb + dyn_bytes;
    g.peers.clear();
    g.peers.resize(g.peer_devs.size());
    for (size_t p = 0; p < g.peer_devs.size(); ++p) {
        g.peers[p].dev = g.peer_devs[p];
        g.peers[p].can_p2p = peer_p2p[p];
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
        if (hipHostMalloc((void**)&hbase, hb, hipHostMallocMapped) != hipSuccess) {
            (void)hipGetLastError();
            fprintf(stderr, "[load] pinned host arena %.2f GB failed (WDDM/commit limit); retrying with CPU-expert mode\n", hb / 1e9);
            if (g.d_sec4) { (void)hipFree(g.d_sec4); g.d_sec4 = nullptr; }
            for (auto* sl : g.vcache_slabs) if (sl) (void)hipFree(sl);
            g.vcache_slabs.clear();
            g.vcache_nslots = 0;
            for (auto& pr : g.peers) if (pr.d_sec4) { (void)hipSetDevice(pr.dev); (void)hipFree(pr.d_sec4); pr.d_sec4 = nullptr; }
            g.peers.clear();
            HCK(hipSetDevice(g.primary_gpu));
            return load_experts_tiered(g, path, vram_budget, true, err);
        }
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
        const double ram_gb_shown = cpu_mode ? (double)(g.sec4_bytes > vb + cold_bytes ? g.sec4_bytes - vb - cold_bytes : 0) / 1e9 : hb / 1e9;
        fprintf(stderr, "[load] experts 4-tier: %llu static VRAM (%.2f GB) + %d adaptive VRAM slots (%.2f GB) = %.2f GB VRAM, %llu in %s (%.2f GB), %llu on SSD (%.2f GB), %.1f s\n",
                (unsigned long long)n_v, vb / 1e9, dyn_slots, dyn_bytes / 1e9, total_vram_arena / 1e9, (unsigned long long)n_h,
                cpu_mode ? "RAM (CPU)" : "pinned host (GPU)", ram_gb_shown, (unsigned long long)n_cold, cold_bytes / 1e9, g.t_load_experts);
    } else {
        uint64_t tot_vram_b = vb, tot_vram_n = n_v;
        for (const auto& pr : g.peers) { tot_vram_b += pr.vram_bytes; tot_vram_n += pr.n_experts; }
        const double ram_gb_shown = (double)(g.sec4_bytes > tot_vram_b + cold_bytes ? g.sec4_bytes - tot_vram_b - cold_bytes : 0) / 1e9;
        fprintf(stderr, "[load] multi-GPU experts (%zu GPUs): total %llu in VRAM (%.2f GB: GPU%d=%llu/%.2fGB", ng, (unsigned long long)tot_vram_n, tot_vram_b / 1e9, g.primary_gpu, (unsigned long long)n_v, vb / 1e9);
        for (const auto& pr : g.peers) fprintf(stderr, ", GPU%d=%llu/%.2fGB", pr.dev, (unsigned long long)pr.n_experts, pr.vram_bytes / 1e9);
        fprintf(stderr, "), %llu in RAM (%.2f GB), %llu on SSD (%.2f GB), %.1f s\n", (unsigned long long)n_h, ram_gb_shown, (unsigned long long)n_cold, cold_bytes / 1e9, g.t_load_experts);
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
                const uint64_t floor_b = (uint64_t)(wddm_vram_floor_gb(tot0) * 1e9);
                const bool trunk_fits_vram = (fr0 > new_trunk + floor_b + (256ull << 20));
                if (trunk_fits_vram && hipMalloc((void**)&g.d_trunk, new_trunk) == hipSuccess) {
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
                    (void)hipGetLastError();
                    hipHostFree(g.h_embed);
                    g.h_embed = nullptr;
                    g.d_embed_mapped = nullptr;
                }
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
            const uint64_t floor_b = (uint64_t)(wddm_vram_floor_gb(tot0) * 1e9);
            if (fr0 > trunk_al + floor_b + (256ull << 20) && hipMalloc((void**)&g.d_trunk, trunk_al) == hipSuccess) {
                stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
            } else {
                (void)hipGetLastError();
                void* hp = nullptr;
                if (hipHostMalloc(&hp, trunk_al, hipHostMallocMapped) != hipSuccess) {
                    (void)hipGetLastError();
                    fprintf(stderr, "trunk alloc failed (%.2f GB, free VRAM %.2f GiB)\n", trunk_al / 1e9, fr0 / 1073741824.0);
                    return false;
                }
                g.h_trunk_mapped = (uint8_t*)hp;
                read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, g.h_trunk_mapped, &g.t_load_trunk, "trunk(host-mapped-4gb)");
                HCK(hipHostGetDevicePointer((void**)&g.d_trunk, g.h_trunk_mapped, 0));
                fprintf(stderr, "[load] low-VRAM GPU (%.1f GiB free): trunk (%.2f GB) in pinned mapped host RAM -> 0 VRAM expert slots\n",
                        fr0 / 1073741824.0, trunk_al / 1e9);
            }
        }
#else
        if (uma_mapped) {
            void* hp = nullptr;
            HCK(hipHostMalloc(&hp, trunk_al, hipHostMallocMapped));
            g.h_trunk_mapped = (uint8_t*)hp;
            read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, (uint8_t*)hp, &g.t_load_trunk, "trunk(uma-mapped)");
            HCK(hipHostGetDevicePointer((void**)&g.d_trunk, hp, 0));
        } else {
            const uint64_t floor_b = (uint64_t)(wddm_vram_floor_gb(tot0) * 1e9);
            if (fr0 > trunk_al + floor_b + (256ull << 20) && hipMalloc((void**)&g.d_trunk, trunk_al) == hipSuccess) {
                stream_to_device(path, h.sections[1].file_offset, g.trunk_bytes, g.d_trunk, &g.t_load_trunk, "trunk");
            } else {
                (void)hipGetLastError();
                void* hp = nullptr;
                if (hipHostMalloc(&hp, trunk_al, hipHostMallocMapped) != hipSuccess) {
                    (void)hipGetLastError();
                    fprintf(stderr, "trunk alloc failed (%.2f GB, free VRAM %.2f GiB)\n", trunk_al / 1e9, fr0 / 1073741824.0);
                    return false;
                }
                g.h_trunk_mapped = (uint8_t*)hp;
                read_file_parallel(path, h.sections[1].file_offset, g.trunk_bytes, g.h_trunk_mapped, &g.t_load_trunk, "trunk(host-mapped-4gb)");
                HCK(hipHostGetDevicePointer((void**)&g.d_trunk, g.h_trunk_mapped, 0));
                fprintf(stderr, "[load] low-VRAM GPU (%.1f GiB free): trunk (%.2f GB) in pinned mapped host RAM -> 0 VRAM expert slots\n",
                        fr0 / 1073741824.0, trunk_al / 1e9);
            }
        }
#endif
    }
    if (!g.rd.read_section(2, g.seed, err)) {
        fprintf(stderr, "sec2: %s\n", err.c_str());
        return false;
    }
#ifdef STRATA_HY4_DISTRIBUTED
    if (g.expert_shard >= 0) {
        size_t free_bytes = 0, total = 0;
        gpu_mem_avail(free_bytes, total);
        const char* reserve = getenv("HY4_RESERVE_GB");
        const double budget = double(free_bytes) - (reserve ? atof(reserve) : 6.0) * 1073741824.0;
        if (load_expert_shard(g, path, budget > 0 ? size_t(budget) : 0, err)) return true;
        fprintf(stderr, "experts: %s\n", err.c_str());
        return false;
    }
#endif
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
        if (const char* vb = getenv("HY4_EXPERT_VRAM_GB") ? getenv("HY4_EXPERT_VRAM_GB") : getenv("STRATA_HY4_VRAM_GB"))
            budget = atof(vb) * 1073741824.0;
        if (g.peer_devs.empty() && g.vram_experts_gb_cap >= 0.0) {
            budget = std::min(budget, g.vram_experts_gb_cap * 1073741824.0);
        }
        if (g.h_trunk_mapped != nullptr) {
            budget = 0.0;  // 4-6 GB discrete GPU with host-mapped trunk: clean 0 VRAM expert slots
        }
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
            // host-resident experts: pinned + read by the GPU (needs the RAM) or computed by the CPU. HY4_CPU_EXPERTS=1/0 forces;
            // default: CPU when they would use > 45% of RAM (or > 25% / > 12 GB on Windows WDDM) or the CPU has AVX-512 VBMI+VNNI.
            bool cpu_mode = false;
            {
                double host_gb = (g.sec4_bytes - std::max(0.0, std::min(budget, (double)g.sec4_bytes))) / 1e9, mem_gb = host_mem_gb();
#ifdef _WIN32
                const bool default_cpu = (host_gb > 0.25 * mem_gb || host_gb > 12.0 || host_cpu_fast());
#else
                const bool default_cpu = (host_gb > 0.45 * mem_gb || host_cpu_fast());
#endif
                cpu_mode = getenv("HY4_CPU_EXPERTS") ? atoi(getenv("HY4_CPU_EXPERTS")) != 0 : default_cpu;
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
