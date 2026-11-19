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
    double gbps() const { return wall_s > 0 ? (double)bytes / wall_s / 1e9 : 0.0; }
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
    void wait();          // blocks (spin, then sleep) until the last submit() has finished
    bool done() const;    // non-blocking poll

    CpuExpertStats stats() const;  // call after wait()
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace hy4
