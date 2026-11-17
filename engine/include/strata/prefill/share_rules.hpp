// include/strata/prefill/share_rules.hpp - when the default CPU prefill share is worth arming (no GPU needed to check).
//
// The share hands the idle CPU pool experts the GPU would otherwise stream over PCIe.  With every expert resident in
// the GPU's cache nothing is streamed, so there is nothing to take; arming the default only moves the staged-chunk
// limit and costs ~5% at 1,000 tokens (#1595: V100, 3 rounds x 3 runs).  An explicit STRATA_PREFILL_CPU_SHARE is
// honoured by Prefill::arm_cpu_share whatever this says.
#pragma once

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

/// resident_slots: the expert cache's live slots; total_pairs: layers x routed experts per layer (<= 0: unknown, arm).
constexpr bool default_share_has_work(int64_t resident_slots, int64_t total_pairs) {
    return total_pairs <= 0 || resident_slots < total_pairs;
}


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
