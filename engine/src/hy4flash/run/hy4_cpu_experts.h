// hy4_cpu_experts.h - CPU evaluation of Hy4-Flash routed experts that live in host RAM ("home PC" case: GPU holds the hot
// experts, the rest are computed here with the AVX-512 VNNI / AVX2 / scalar S_K kernels of strata/quant/sk_codecs.hpp).
// Plain C++17, no CUDA/HIP. Math per (token t, slot s, expert e, weight w), identical to skmoe.hip.h:
//     gu = Wgu[e] @ z[t]                       (3072 rows, K = 1536; rows 0..1535 gate, 1536..3071 up)
//     h  = min(silu(gate), 10) * clamp(up, -10, 10)
//     h2 = h / sq_xd_dn[e]                     (fp16[1536] at the start of the body, element-wise by output index)
//     y  = Wdn[e] @ h2                         (1536 rows, K = 1536)
//     zout[t] (+)= sum over slots s in ascending order of  w * y
// Activations (z for gate/up, h2 for down) are quantized once to Q8_1_G64 (24 blocks of 64); weights stay in S_K.
// Results do not depend on the thread count (every output row is produced by exactly one thread, summed in slot order).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hy4 {

struct CpuExpertStats {
    uint64_t bytes = 0;               // unique expert weight bytes consumed (gate/up + down + sq_xd_dn) by all jobs
    double wall_s = 0;                // sum over jobs of (job finished - submit() called)
    uint64_t jobs = 0, pairs = 0;     // jobs run, (token,slot) pairs evaluated
    uint64_t skipped_unresident = 0;  // pairs with use_pair != 0 whose expert is not CPU-resident (they are NOT computed)
    double phase_s[5] = {0, 0, 0, 0, 0};  // wall seconds in: plan, quantize z, gate/up(+h), down, combine (sum over jobs)
    // cold experts (set_cold_experts): bodies read from the .strata file on demand
    uint64_t cold_bytes = 0;          // expert body bytes read from the file (cache misses only)
    uint64_t cold_reads = 0;          // body reads issued (one per missed expert per job)
    uint64_t cold_hits = 0;           // cold experts served from the slot cache (LRU) without a read
    uint64_t cold_errors = 0;         // failed reads (the job's result is invalid when != 0)
    double cold_wait_s = 0;           // seconds the pool waited for reads (not hidden behind the compute of earlier waves)
    double cold_io_s = 0;             // seconds with reads in flight (issue of a wave -> its last read completed)
    double gbps() const { return wall_s > 0 ? (double)bytes / wall_s / 1e9 : 0.0; }
    double cold_gbps() const { return cold_io_s > 0 ? (double)cold_bytes / cold_io_s / 1e9 : 0.0; }  // SSD read rate while reading
};

struct CpuExpertView {                // one resident expert body (arena, 4 KiB aligned)
    const uint8_t* body = nullptr;    // null when not resident
    uint32_t dn_off = 0;              // byte offset of the down weights inside the body
    int gu_code = 0, dn_code = 0;     // SkCode values (2..6, 8)
    uint32_t body_bytes = 0;
};

class CpuExperts {
public:
    static constexpr int kExperts = 512, kTopK = 16, kDim = 1536;
    static constexpr int kChunkTokens = 512;  // submit() with T > 512 is processed in chunks of 512 tokens inside one job

    CpuExperts();
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    // Opens the .strata (header + Section 3 directory are parsed here, no zstd needed), reads every resident expert body
    // from Section 4 into one big arena (hugepage-advised, mlock best effort, runs of adjacent bodies are read with one
    // large sequential read by up to 4 reader threads) and starts n_threads pool threads (pinned to physical cores first;
    // STRATA_CPUEXP_PIN=0 disables pinning, STRATA_CPUEXP_SPIN_US sets the idle spin time, default 300 us).
    // cpu_resident[layer][expert] (41 x 512): which experts are loaded; null = all. layer = MoE layer index 0..40.
    bool init(const std::string& strata_path, int n_threads, const std::vector<std::vector<bool>>* cpu_resident, std::string& err);

    // ---- cold experts: kept in the .strata file, read on demand -----------------------------------------------------------------
    // Call after init(); replaces the current cold set. cold[layer][expert id] (41 x 512); a cold expert must not be resident.
    // submit() then computes a pair when use_pair (null = all) != 0 and its expert is resident OR cold; experts in neither mask
    // (GPU-owned) stay skipped. A job touching cold experts runs in waves: wave 0 = resident experts + cold cache hits, then the
    // missed cold experts, read by reader threads with large preads (O_DIRECT by default) that were issued when the job started
    // and overlap the earlier waves' compute. A small LRU slot cache keeps recently used cold bodies in RAM. The result does not
    // depend on whether a body came from the arena, the cache or the file (bit-exact). Works on Linux (O_DIRECT pread) and Windows (FILE_FLAG_NO_BUFFERING overlapped ReadFile).
    // Env: STRATA_CPUEXP_COLD_CACHE_MB (default 256), STRATA_CPUEXP_COLD_READERS (default 8),
    //      STRATA_CPUEXP_COLD_IO = direct (default) | buffered (page cache kept, for warm-cache runs) | fadvise (buffered + DONTNEED).
    bool set_cold_experts(const std::vector<std::vector<bool>>& cold, std::string& err);
    bool cold_expert(int layer, int expert) const;
    /// Enqueue speculative 512 KiB chunked NVMe prefetches for predicted cold experts of a future layer (L+1, L+2).
    void prefetch_cold(int layer, const int32_t* experts, int n);
    /// Drop stale unstarted/in-flight cold prefetches for layers older than `current_layer`.
    void drop_stale_cold(int current_layer);
    // RAM the process may use: min(MemTotal, cgroup memory.max / memory.limit_in_bytes) on Linux, total physical memory on Windows.
    static uint64_t ram_budget_bytes();
    // Currently available host RAM (Linux MemAvailable capped by cgroup free + reclaimable file cache; Windows ullAvailPhys).
    static uint64_t ram_available_bytes();
    // Default pool thread count: respects Linux cgroup cpu.max / cfs_quota_us (leaves ~2 cores for CUDA/Engram when cgroup-constrained)
    // and Windows multi-group processor counts (GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)).
    static int default_threads();

    // ---- directory helpers (read only the 4 KiB header + Section 3; no model load, no zstd) ------------------------------------
    // Section 4 stores the 512 experts of every MoE layer hot-to-cold by rank: directory index i = layer*512 + rank, and the
    // entry says which expert id sits at that rank. These build the cpu_resident mask for init() from rank ranges.
    // expert_of_rank[layer*512 + rank] = expert id; codes[layer*512 + expert_id] = (gu_code << 8) | dn_code (SkCode values).
    static bool read_rank_table(const std::string& path, std::vector<uint16_t>& expert_of_rank, std::vector<uint16_t>& codes, std::string& err);
    // body_bytes[layer*512 + rank] = size of that expert's body in the file (for sizing the RAM / cold split)
    static bool read_body_bytes(const std::string& path, std::vector<uint32_t>& body_bytes, std::string& err);
    // mask[layer][expert] = true for the experts whose rank in that layer is in [first_rank[layer], last_rank[layer]) (half open;
    // 0 = hottest). One entry per layer (41 for Hy4-Flash). E.g. "the coldest 400 of every layer live in RAM, the hottest 112
    // are on the GPU" is first_rank = 112, last_rank = 512 for all layers.
    static bool make_mask_rank_range(const std::string& path, const std::vector<int>& first_rank, const std::vector<int>& last_rank,
                                     std::vector<std::vector<bool>>& mask, std::string& err);
    // same range for every layer
    static bool make_mask_rank_range(const std::string& path, int first_rank, int last_rank, std::vector<std::vector<bool>>& mask, std::string& err);

    // Kernel: AVX-512 VBMI/VNNI "fast" kernel when compiled for it (-march=native on Ice Lake / Zen 4 or newer) and all resident
    // experts are S2..S6, else the library gemm_sk_q8_1 kernel. STRATA_CPUEXP_KERNEL=lib forces the library kernel.
    bool fast_kernel() const;
    // "lib", "fast(avx2)" or "fast(avx512-vbmi)": the kernel picked at init (CPUID; STRATA_CPUEXP_KERNEL=avx512|avx2|lib overrides)
    const char* kernel_name() const;
    // Activation precision. Default "q8": z and h2 are quantized to one Q8_1_G64 term (rel-L2 about 1.4e-2 of the expert output vs
    // exact fp32 activations). "hi" (env STRATA_CPUEXP_ACT=hi at init, or set_hi_activations(true) between jobs): two terms
    // x = da*qa + db*qb with qb = Q8_1 of the residual, two dpbusd passes over the same unpacked weights, summed in fp32.
    // set_hi_activations waits for a running job first.
    void set_hi_activations(bool on);
    bool hi_activations() const;
    // Self test of the fast row kernel with random S2..S6 weights vs dequant_sk_row; returns false on a mismatch.
    static bool selftest_kernels(std::string& report);

    bool resident(int layer, int expert) const;
    CpuExpertView view(int layer, int expert) const;
    int n_layers() const;
    uint64_t arena_bytes() const;
    uint64_t resident_count() const;
    bool arena_locked() const;
    double load_seconds() const;
    int n_threads() const;

    // Asynchronous. Evaluates, for the T tokens, every (t,s) pair with use_pair[t*16+s] != 0 (use_pair null = all pairs) whose
    // expert idx[t*16+s] is CPU-resident; other pairs are skipped (counted in stats.skipped_unresident when use_pair asked
    // for them). z: fp32 [T][1536]; idx: int32 [T][16]; w: float [T][16]; zout: float [T][1536].
    //   accumulate = false: zout[t] = partial sum (tokens without a CPU pair get zeros)   (overwrite)
    //   accumulate = true : zout[t] += partial sum
    // All input/output pointers must stay valid and unchanged until wait() returns. One job in flight: submit() first
    // waits for the previous one. T may be 1..any (T > 512 runs in chunks of 512 tokens).
    void submit(int layer, int T, const float* z, const int32_t* idx, const float* w, const uint8_t* use_pair, float* zout,
                bool accumulate = false);
    // Non-blocking next-layer CPU L3 cache prefetch (STRATA_L3_PF): wakes idle workers to stream the specified resident
    // experts of `layer` into CPU L2/L3 caches while the GPU computes attention; preempts immediately on next submit().
    void prefetch_experts(int layer, const int32_t* experts, int n_experts);
    void wait();          // blocks (spin, then sleep) until the last submit() has finished
    bool done() const;    // non-blocking poll

    CpuExpertStats stats() const;  // call after wait()
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace hy4
