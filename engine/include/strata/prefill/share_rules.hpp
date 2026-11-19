// include/strata/prefill/share_rules.hpp - when the default CPU prefill share is worth arming (no GPU needed to check).
//
// The share hands the idle CPU pool experts the GPU would otherwise stream over PCIe.  With every expert resident in
// the GPU's cache nothing is streamed, so there is nothing to take; arming the default only moves the staged-chunk
// limit and costs ~5% at 1,000 tokens (#1595: V100, 3 rounds x 3 runs).  An explicit STRATA_PREFILL_CPU_SHARE is
// honoured by Prefill::arm_cpu_share whatever this says.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::prefill::share_rules {

// ---- the limit (BW-MEASURE E25, owner decision 10-31) ----
// The DEFAULT share applies to staged chunks up to this many tokens.  0.1.41 armed it below 1,024 tokens (RTX 3060, P100, RTX 5070: -20..-35% at 512 and
// 1,000 tokens).  A chunk of 1-3K tokens routes to nearly every expert of every layer, so with half the experts resident one pass over the rest at the link's
// speed is its floor (RTX PRO 5000 with a 28 GB/s link and 12,437 of 24,576 experts resident: 0.97-1.04 s for any 1K-4K prompt) and the CPU share takes
// part of that pass: prompt speed +151% at 1K, +45% at 2K (TTFT 1.07 -> 0.74 s), +99% / +32% with 8 pool workers; 4K and above unchanged, and a limit
// of 8,192 loses 1-6% at 4K-8K.  First-token KL after a 1.1-2.8K prompt: mean 0.011, top-1 100%.  STRATA_PREFILL_CPU_SHARE_MAX=1024 restores the old limit.
constexpr int64_t kShareLimitOld = 1024;
constexpr int64_t kShareLimitNew = 3072;
/// `raise`: the card class may take the new limit (a class whose A/B lost at 1-3K keeps 1,024, automatically).
constexpr int64_t default_share_limit(bool raise = true) { return raise ? kShareLimitNew : kShareLimitOld; }
/// AMD-Q: a gfx12 card whose fused IQ2_S / IQ4_XS gate/up kernels are on (fused::gfx12_iq2s_default): chunks below this are staged and take the CPU share, a chunk
/// from it up streams every expert and runs the fused kernels.  R9700 (p620, IQ3_S, 54% resident), prompt speed against the share-only default (limit 3,072):
/// limit 1,536: 1.3K 1.00, 1.5K 1.00, 1.8K +7.1%, 2K +10.0%, 2.6K +16.2%, 3K +11.1% (two pairs each); limit 2,048: 1.8K +1.4%, 2K +9.7%; limit 3,072 (fused only above): no gain at 1.5K-2.6K.
/// Raised to 2,304 on a later engine state (same card, IQ3_S, 3 rounds; prompt ms, limit 1,536 -> 2,304): 1,537 tokens 1,995 -> 1,249, 1,793 1,672 -> 1,406,
/// 2,049 1,697 -> 1,544; 2,305 1,728 -> 1,792 (+3.7%: the chunk that sits exactly on the limit pays), 2,561 and above within 0.6%.  The earlier 2,048 row was
/// measured on an earlier engine state, so the two rows are not comparable.  Forced-token KL against an FP16 reference
/// (16 prompts each of 1,793 and 2,049 tokens, 64 tokens): 0.01350 -> 0.01176 (ratio 0.871, interval 0.758-1.016), top-1 96.2% -> 96.6%.
/// STRATA_PREFILL_CPU_SHARE_MAX=1536 restores the old limit.
constexpr int64_t kShareLimitGfx12Fused = 2304;
constexpr int64_t gfx12_share_limit(bool fused_iq2s) { return fused_iq2s ? kShareLimitGfx12Fused : kShareLimitNew; }

// ---- the link-aware limit and expert width (BW-H, owner decision 11-19: "full blackwell plan") ----
// A chunk below the limit is STAGED (its routing is read back, only the routed experts the card lacks are brought, and the CPU may take some of them); a chunk from the limit up STREAMS every expert the card
// does not hold in one pass through a ring, at the link's speed.  The streamed pass costs about  F = (experts not on the card) x blob / link  whatever the chunk (RTX 5090, UD-Q4_K_XL, 8,056 of 24,576 experts on the
// card: F = 0.97 s at 53.6 GB/s, 1.83 s at 28.3, 3.67 s at 14.1, 14.4 s at 3.6) plus ~0.07 ms a token; the staged pass with the wide share costs a fixed part plus 0.2-0.3 ms a token, so the staged walk wins up to a
// length T* that grows with F.  Measured T* (prompt time of the share with experts of up to 32 tokens against the default, 3 interleaved rounds, bwh lenmap): 4.2K (53.6 GB/s, 15 workers), 8.8K (28.3 GB/s, 22 workers),
// 14.0K (14.1 GB/s, 32 workers), 19.7K with 32-token experts and 25K with 64-token experts (3.6 GB/s, 32 workers).  The limit is a margin under T*: 3.4 x F up to F = 3.67 s (T* is 3.8-4.8 x F there), then 0.76 tokens
// per ms of F more (the CPU, not the link, sets T* on a very slow link), never below the default 3,072 (at 53.6 GB/s the default beats streaming from 1,536 tokens by 25-34%, and above ~4K staged = streamed) nor above
// 20,480.  A pool smaller than the 22 workers the 3.4 was measured with takes less (8 workers: T* fell 13%, 16 workers: 3%).  Experts of up to 32 tokens (64 from F = 3 s: T* equal at 14 GB/s, +27% at 3.6 GB/s) go
// to the CPU where the limit is raised; where it is not (a fast link) the share keeps the pool's own 8.
struct LinkRuleIn {
    double link_gbps = 0.0;      ///< the start-up host-to-device probe, GB/s (0 = not probed)
    int pool_workers = 0;        ///< the CPU pool's workers
    int64_t resident = -1;       ///< expert slots on the card
    int64_t total = 0;           ///< layers x experts
    double blob_mb = 0.0;        ///< mean expert blob of the cache, MB
};
struct LinkRule {
    bool arm = false;
    int64_t limit = 0;           ///< the share's chunk limit, tokens
    int nt = 0;                  ///< the experts the share may take: up to nt tokens of a chunk (0 = the pool's own 8)
    double floor_ms = 0.0;       ///< the stream floor: bytes of the experts the card does not hold / the link
};
constexpr int64_t kShareLimitMax = 20480;      // the most the rule gives (T* is 25K at 3.6 GB/s with 64-token experts)
constexpr int64_t kShareLimitGrid = 256;       // the prompt path's chunk grid
constexpr double kLinkTokensPerFloorMs = 3.4;  // limit = this x the stream floor in ms up to kLinkKneeMs (the measured crossings are 3.8-4.8 x)
constexpr double kLinkKneeMs = 3670.0;         // F of the 14.1 GB/s box: 12,480 tokens
constexpr double kLinkTokensPerMsAbove = 0.76; // beyond the knee: 3.6 GB/s (F = 14,388 ms) reaches the 20,480 cap
constexpr double kLinkWideFloorMs = 3000.0;    // from this F the share takes experts of up to kLinkNtWide tokens
constexpr int kLinkNt = 32;                    // the share's expert width when the limit is raised
constexpr int kLinkNtWide = 64;
constexpr double kLinkPoolRef = 22.0;          // the pool the 3.4 was measured with (EPYC 9B14: 22 workers)
inline double link_pool_factor(int workers) {
    return workers > 0 ? std::min(1.0, std::pow((double) workers / kLinkPoolRef, 0.15)) : 1.0;
}
/// The rule from the stream floor F (ms) itself: one card's F is (experts not on it) x blob / link; a layer split's is the sum over its stages, because a chunk passes through every stage in turn.
inline LinkRule link_rule_floor(double f, int pool_workers) {
    LinkRule r;
    if (!(f > 0.0)) return r;
    r.floor_ms = f;
    const double t = f <= kLinkKneeMs ? kLinkTokensPerFloorMs * f : kLinkTokensPerFloorMs * kLinkKneeMs + kLinkTokensPerMsAbove * (f - kLinkKneeMs);
    int64_t lim = (int64_t) (t * link_pool_factor(pool_workers)) / kShareLimitGrid * kShareLimitGrid;
    lim = std::min<int64_t>(kShareLimitMax, std::max<int64_t>(kShareLimitNew, lim));
    r.limit = lim;
    r.nt = lim > kShareLimitNew ? (f >= kLinkWideFloorMs ? kLinkNtWide : kLinkNt) : 0;
    r.arm = true;
    return r;
}
inline LinkRule link_rule(const LinkRuleIn& in) {
    if (!(in.link_gbps > 0.0) || in.total <= 0 || in.resident < 0 || in.resident >= in.total || !(in.blob_mb > 0.0)) return LinkRule{};
    return link_rule_floor((double) (in.total - in.resident) * in.blob_mb / in.link_gbps, in.pool_workers);
}
/// STRATA_PREFILL_CPU_SHARE_AUTO as the engine reads it: -1 unset (or empty), 1 for exactly "1", 0 for any other value.
inline int env_state(const char* v) {
    if (v == nullptr || v[0] == '\0') return -1;
    return (v[0] == '1' && v[1] == '\0') ? 1 : 0;
}
/// Whether the rule runs: the class it was measured on by default (a CUDA discrete card, one card, the link probed, compute capability 12.0), every CUDA card with =1, none with any other value.
inline bool link_rule_enabled(bool cuda, bool discrete, bool link_probed, int cc_major, int cc_minor, int env) {
    if (!cuda || env == 0 || !discrete || !link_probed) return false;
    return env == 1 || (cc_major == 12 && cc_minor == 0);
}

/// resident_slots: the expert cache's live slots; total_pairs: layers x routed experts per layer (<= 0: unknown, arm).
constexpr bool default_share_has_work(int64_t resident_slots, int64_t total_pairs) {
    return total_pairs <= 0 || resident_slots < total_pairs;
}

// ---- which card layouts arm the default share (BW-F, two 16 GB cards) ----
// Until 0.2.16 the default share was armed on one card only: a layer split left every prompt chunk below the staged-chunk limit to stream its experts over PCIe on each stage in turn, although
// the stages already take the pool one at a time (prefill.cpp g_cpu_pool_busy).  Two RTX 5060 Ti 16 GB (PCIe 4.0 x8, 14.3 GB/s each, EPYC 7K62 with 48 cores, UD-Q4_K_XL, DFlash2, layer split auto K = 24),
// prompt ms, STRATA_PREFILL_CPU_SHARE=auto against the split's own default (share off), median of 3 rounds: see phaseA-tests/bw/RESULTS-BW-F.md (BW-MG-small).
// `later_stages`: the layer split's stages after the first (0: no split); `multi_gpu`: a layer split across devices; `env`: STRATA_SPLIT_CPU_SHARE as read (-1 unset, 0 off, 1 on).
// One card as before; exactly one later stage (two cards) by the switch's default on a CUDA build (not measured on HIP or SYCL); three and four cards, and the helper caches, are not measured and stay off.
inline constexpr bool kSplitShareDefault = true;
constexpr bool default_share_layout(int later_stages, bool multi_gpu, int env, bool cuda = true) {
    if (later_stages == 0 && !multi_gpu) return true;
    if (later_stages == 1 && multi_gpu && cuda) return env < 0 ? kSplitShareDefault : env != 0;
    return false;
}
constexpr int split_share_env_of(const char* v) { return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '0' ? 0 : 1); }


// ---- memory (0.1.41 report, RTX 5090, tight RAM: chats failed and the engine restarted until STRATA_PREFILL_CPU_SHARE=0) ----
// The default share reads experts with the CPU and keeps two pinned buffers on the host.  When the host has little
// memory left, or the experts it would read are file pages the RAM cannot keep (each is a page-in on a worker thread),
// the default is not armed.  An explicit STRATA_PREFILL_CPU_SHARE is the user's: it is armed and the engine only warns.
constexpr uint64_t kShareRamMargin = 3ull << 30;     // RAM kept free beside the share's own buffers and the file pages
constexpr uint64_t kShareCommitFloor = 4ull << 30;   // Windows commit kept free (the #1607 floor)
constexpr uint64_t kUnknown = ~uint64_t{0};

struct ShareMemory {
    uint64_t avail_ram = kUnknown;       ///< RAM this process can get (cgroup-aware); kUnknown: not known
    uint64_t avail_commit = kUnknown;    ///< Windows commit left (RAM + page file); kUnknown elsewhere
    uint64_t buffer_bytes = 0;           ///< host bytes the share itself allocates (pinned buffers, scratch)
    uint64_t inplace_bytes = 0;          ///< expert bytes the CPU would read as file pages (not in the GPU cache, not in RAM)
};

struct ShareVerdict {
    bool arm = true;
    std::string why;   ///< when !arm: what the engine says, in one line
};

inline ShareVerdict default_share_memory_verdict(const ShareMemory& m) {
    ShareVerdict v;
    const double gib = 1073741824.0;
    char buf[300];
    if (m.avail_ram != kUnknown && m.avail_ram < m.buffer_bytes + kShareRamMargin) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of RAM is free (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_ram / gib, (double) m.buffer_bytes / gib, (double) kShareRamMargin / gib);
        v.arm = false;
    } else if (m.avail_commit != kUnknown && m.avail_commit < m.buffer_bytes + kShareCommitFloor) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of the Windows commit limit is left (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_commit / gib, (double) m.buffer_bytes / gib, (double) kShareCommitFloor / gib);
        v.arm = false;
    } else if (m.inplace_bytes > 0 && m.avail_ram != kUnknown && m.inplace_bytes + kShareRamMargin > m.avail_ram) {
        std::snprintf(buf, sizeof buf, "the experts it would read are %.1f GiB of file pages and only %.1f GiB of RAM is free to keep them",
                      (double) m.inplace_bytes / gib, (double) m.avail_ram / gib);
        v.arm = false;
    }
    if (!v.arm) v.why = buf;
    return v;
}

}  // namespace strata::prefill::share_rules
