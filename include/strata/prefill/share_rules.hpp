// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/prefill/share_rules.hpp - when the default CPU prefill share is worth arming (no GPU needed to check).
//
// The share hands the idle CPU pool experts the GPU would otherwise stream over PCIe.  With every expert resident in
// the GPU's cache nothing is streamed, so there is nothing to take; arming the default only moves the staged-chunk
// limit and costs ~5% at 1,000 tokens (#1595: V100, 3 rounds x 3 runs).  An explicit STRATA_PREFILL_CPU_SHARE is
// honoured whatever this says (Prefill::set_share_default).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::prefill::share_rules {

/// resident_slots: the expert cache's live slots; total_pairs: layers x routed experts per layer (<= 0: unknown, arm).
constexpr bool default_share_has_work(int64_t resident_slots, int64_t total_pairs) {
    return total_pairs <= 0 || resident_slots < total_pairs;
}

// ---- the link-aware limit and expert width (upstream's BW-H, v0.2.17; its numbers measured again here) ----
// A chunk below the limit is STAGED (its routing is read back, only the routed experts the card lacks are brought, and
// the CPU may take some of them); a chunk from the limit up STREAMS every expert the card does not hold in one pass, at
// the link's speed.  The streamed pass costs about F = (experts not on the card) x blob / link whatever the chunk, and
// little per token; the staged pass with a wide share costs less at first and more per token, so it wins up to a
// length T* that grows with F.  Measured 2026-11-25 (records/2026-11-25-short-prefill, Qwen3.8-Flash-Next IQ3_XXS):
//   RTX 4070, link 6.4 GB/s, 20.8K of 24.6K experts not on the card (F 5.7 s), 19 pool workers (AVX2), 64-token
//   experts, read ms staged / streamed: 3.6K tokens 4,084 / 6,309, 6K 5,595 / 6,563, 8K 7,099 / 6,862, 10K 8,550 /
//   7,280, 12K 9,850 / 7,732, 13.5K 10,773 / 8,143: T* 7.3K tokens, 1.28 tokens per ms of F.
//   Arc Pro B70, link 28 GB/s, 10.3K experts not on the card (F 0.64 s): staged with 8 / 32 / 64-token experts 2,103 /
//   2,251 / 2,303 ms against 2,069 streamed at 3.6K tokens: below the plain limit, as the rule gives it.
// The limit is a margin under T*: 1.1 x F (ms), never below the 3,072 of the plain share nor above 20,480 (the staged
// walk's buffers), on a grid of 256; a pool smaller than the 19 workers it was measured with takes less (upstream's
// exponent, RTX 5090, 8 to 32 workers).  Where the limit is raised, experts of up to 64 tokens go to the CPU (RTX 4070,
// 3.6K tokens, read ms with 8 / 32 / 64 / 128-token experts: 6,309 streamed / 4,473 / 4,084 / 4,056; 1K tokens 3,357 /
// 1,845 / 1,864 / 1,839); where it is not, nothing changes.  Upstream's 3.4 x F (RTX 5090, AVX-512, its own streamed
// pass) put the RTX 4070's limit at 13K tokens, where the staged pass here is 29% slower than the streamed one.
// The rule reads only the link, the cache and the pool, so it runs on every discrete card whose link was probed
// (upstream: CUDA cards by compute capability).  unverified: links slower than 6.4 GB/s, AMD cards.
// XESTRATA_PREFILL_CPU_SHARE_AUTO=0 turns it off.
struct LinkRuleIn {
    double link_gbps = 0.0;      ///< the start-up host-to-device probe, GB/s (0 = not probed)
    int pool_workers = 0;        ///< the CPU pool's workers
    int64_t resident = -1;       ///< expert slots on the card
    int64_t total = 0;           ///< layers x experts
    double blob_mb = 0.0;        ///< mean expert blob, MB
};
struct LinkRule {
    bool arm = false;
    int64_t limit = 0;           ///< the share's chunk limit, tokens (chunks up to it are staged)
    int nt = 0;                  ///< the experts the share may take: up to nt tokens of a chunk (0 = the pool's own MAXT)
    double floor_ms = 0.0;       ///< the stream floor: bytes of the experts the card does not hold / the link
};
constexpr int64_t kShareLimitDefault = 3072;
constexpr int64_t kShareLimitMax = 20480;
constexpr int64_t kShareLimitGrid = 256;
constexpr double kLinkTokensPerFloorMs = 1.1;
constexpr int kLinkNt = 64;
constexpr double kLinkPoolRef = 19.0;
inline double link_pool_factor(int workers) {
    return workers > 0 ? std::min(1.0, std::pow((double) workers / kLinkPoolRef, 0.15)) : 1.0;
}
inline LinkRule link_rule(const LinkRuleIn& in) {
    LinkRule r;
    if (!(in.link_gbps > 0.0) || in.total <= 0 || in.resident < 0 || in.resident >= in.total || !(in.blob_mb > 0.0)) return r;
    r.floor_ms = (double) (in.total - in.resident) * in.blob_mb / in.link_gbps;
    const int64_t lim = (int64_t) (kLinkTokensPerFloorMs * r.floor_ms * link_pool_factor(in.pool_workers)) /
                        kShareLimitGrid * kShareLimitGrid;
    r.limit = std::min<int64_t>(kShareLimitMax, std::max<int64_t>(kShareLimitDefault, lim));
    r.arm = r.limit > kShareLimitDefault;
    r.nt = r.arm ? kLinkNt : 0;
    return r;
}


// ---- memory (0.1.41 report, RTX 5090, tight RAM: chats failed and the engine restarted until STRATA_PREFILL_CPU_SHARE=0) ----
// The default share reads experts with the CPU and keeps two pinned buffers on the host.  When the host has little
// memory left, or the experts it would read are file pages the RAM cannot keep (each is a page-in on a worker thread),
// the default is not armed.  An explicit STRATA_PREFILL_CPU_SHARE is the user's: it is armed and the engine only warns.
constexpr uint64_t kShareRamMargin = 3ull << 30;     // RAM kept free beside the share's own buffers and the file pages
constexpr uint64_t kUnknown = ~uint64_t{0};

struct ShareMemory {
    uint64_t avail_ram = kUnknown;       ///< RAM this process can get (cgroup-aware); kUnknown: not known
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
    } else if (m.inplace_bytes > 0 && m.avail_ram != kUnknown && m.inplace_bytes + kShareRamMargin > m.avail_ram) {
        std::snprintf(buf, sizeof buf, "the experts it would read are %.1f GiB of file pages and only %.1f GiB of RAM is free to keep them",
                      (double) m.inplace_bytes / gib, (double) m.avail_ram / gib);
        v.arm = false;
    }
    if (!v.arm) v.why = buf;
    return v;
}

}  // namespace strata::prefill::share_rules
