// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/share_rules.hpp"

using strata::prefill::share_rules::default_share_has_work;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const long long pairs = 48 * 512;
    CHECK(default_share_has_work(pairs - 1, pairs));       // one expert streams: the share has something to take
    CHECK(default_share_has_work(1500, pairs));            // a 12 GB card: most of them stream
    CHECK(!default_share_has_work(pairs, pairs));          // every expert resident: nothing to take
    CHECK(!default_share_has_work(pairs + 100, pairs));    // a cache with spare slots
    CHECK(default_share_has_work(0, pairs));               // empty cache
    CHECK(default_share_has_work(100000, 0));              // total unknown: keep today's behaviour
    static_assert(!default_share_has_work(10, 10), "constexpr");

    // BW-MEASURE E25: the default limit is 3,072 tokens, 1,024 for a class that keeps the old one
    CHECK(strata::prefill::share_rules::default_share_limit() == 3072);
    CHECK(strata::prefill::share_rules::default_share_limit(true) == strata::prefill::share_rules::kShareLimitNew);
    CHECK(strata::prefill::share_rules::default_share_limit(false) == 1024);
    CHECK(strata::prefill::share_rules::gfx12_share_limit(true) == 2304 && strata::prefill::share_rules::gfx12_share_limit(false) == 3072);   // AMD-Q
    static_assert(strata::prefill::share_rules::kShareLimitOld == 1024, "the old limit (the way back is STRATA_PREFILL_CPU_SHARE_MAX=1024)");

    // BW-F: the card layouts that arm the default share: one card as before, two cards (one later stage) by the switch's default, nothing else
    {
        using namespace strata::prefill::share_rules;
        CHECK(default_share_layout(0, false, -1) && default_share_layout(0, false, 0) && default_share_layout(0, false, 1));   // one card: STRATA_SPLIT_CPU_SHARE is not read
        CHECK(default_share_layout(1, true, -1) == kSplitShareDefault);                                                       // two cards, the switch unset
        CHECK(!default_share_layout(1, true, 0) && default_share_layout(1, true, 1));                                          // the switch both ways
        CHECK(!default_share_layout(2, true, -1) && !default_share_layout(2, true, 1) && !default_share_layout(3, true, 1));  // three and four cards: not measured, never armed by default
        CHECK(!default_share_layout(1, true, -1, false) && !default_share_layout(1, true, 1, false) && default_share_layout(0, false, -1, false));   // HIP / SYCL: the split stays off, one card as before
        CHECK(!default_share_layout(1, false, 1) && !default_share_layout(0, true, 1));                                        // helper caches (no layer split) and an inconsistent split: off
        CHECK(split_share_env_of(nullptr) == -1 && split_share_env_of("") == -1 && split_share_env_of("0") == 0 && split_share_env_of("1") == 1 && split_share_env_of("on") == 1);
        static_assert(default_share_layout(0, false, -1), "constexpr");
    }
    // H11: memory verdicts
    using namespace strata::prefill::share_rules;
    const uint64_t G = 1ull << 30;
    ShareMemory m;
    m.buffer_bytes = G / 3;
    CHECK(default_share_memory_verdict(m).arm);                                   // nothing known: today's behaviour
    m.avail_ram = 20 * G; m.avail_commit = 30 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // plenty of RAM and commit
    m.avail_ram = 3 * G;
    CHECK(!default_share_memory_verdict(m).arm);                                  // RAM below buffers + margin
    CHECK(default_share_memory_verdict(m).why.find("RAM is free") != std::string::npos);
    m.avail_ram = kShareRamMargin + m.buffer_bytes;
    CHECK(default_share_memory_verdict(m).arm);                                   // exactly enough
    m.avail_ram = kShareRamMargin + m.buffer_bytes - 1;
    CHECK(!default_share_memory_verdict(m).arm);                                  // one byte short
    m.avail_ram = 20 * G; m.avail_commit = 3 * G;
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);                                                            // Windows commit nearly used up
        CHECK(v.why.find("commit") != std::string::npos);
    }
    m.avail_commit = kUnknown;
    m.inplace_bytes = 18 * G;                                                     // file pages the RAM cannot keep
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);
        CHECK(v.why.find("file pages") != std::string::npos);
    }
    m.inplace_bytes = 10 * G;                                                     // they fit with the margin
    CHECK(default_share_memory_verdict(m).arm);
    m.avail_ram = kUnknown; m.inplace_bytes = 500 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // RAM unknown: never blocks
    // BW-H: the link-aware limit and expert width (RTX 5090, UD-Q4_K_XL: 24,576 experts of ~3.13 MB, 8,056 on the card)
    {
        LinkRuleIn in;
        in.pool_workers = 22; in.resident = 8056; in.total = 24576; in.blob_mb = 3.133;
        CHECK(!link_rule(in).arm);                                     // no probe: today's behaviour
        in.link_gbps = 28.3;                                           // PCIe 5.0 x8 (EPYC 9B14, 22 workers): the measured crossing is 8.8K tokens
        LinkRule r = link_rule(in);
        CHECK(r.arm && r.limit == 6144 && r.nt == kLinkNt);
        CHECK(r.floor_ms > 1800.0 && r.floor_ms < 1860.0);
        in.pool_workers = 8;                                           // a small pool takes less (8 workers: the crossing fell from 8.8K to 7.7K)
        CHECK(link_rule(in).limit == 5120 && link_rule(in).nt == kLinkNt);
        CHECK(link_pool_factor(0) == 1.0 && link_pool_factor(64) == 1.0 && link_pool_factor(22) == 1.0 && link_pool_factor(8) < link_pool_factor(16));
        in.link_gbps = 14.1; in.pool_workers = 32;                     // PCIe 4.0 x8 (Xeon 8352V, 32 workers): crossing 14.0K
        r = link_rule(in);
        CHECK(r.arm && r.limit == 12288 && r.nt == kLinkNtWide);
        in.link_gbps = 53.6; in.pool_workers = 15;                     // PCIe 5.0 x16 (Ryzen 9 9950X, 15 workers): crossing 4.2K, the default 3,072 stays
        r = link_rule(in);
        CHECK(r.arm && r.limit == kShareLimitNew && r.nt == 0);
        in.link_gbps = 3.6; in.pool_workers = 32;                      // PCIe 4.0 x2: the cap, 64-token experts
        r = link_rule(in);
        CHECK(r.arm && r.limit == kShareLimitMax && r.nt == kLinkNtWide);
        in.link_gbps = 1000.0;                                         // a link so fast the floor is tiny: never below the default
        CHECK(link_rule(in).limit == kShareLimitNew);
        in.link_gbps = 28.3; in.resident = 24576;                      // every expert resident: nothing to take
        CHECK(!link_rule(in).arm);
        in.resident = -1;
        CHECK(!link_rule(in).arm);                                     // the card's counts not known
        CHECK(env_state(nullptr) == -1 && env_state("") == -1 && env_state("1") == 1 && env_state("0") == 0 && env_state("auto") == 0);
        CHECK(link_rule_enabled(true, true, true, 12, 0, -1));         // the class: a discrete cc 12.0 card, link probed
        CHECK(!link_rule_enabled(true, true, true, 8, 6, -1));         // other cards: not measured
        CHECK(link_rule_enabled(true, true, true, 8, 6, 1));           // =1 asks for it
        CHECK(!link_rule_enabled(true, true, true, 12, 0, 0));         // =0 is the way back
        CHECK(!link_rule_enabled(true, false, true, 12, 0, -1));       // an integrated card
        CHECK(!link_rule_enabled(true, true, false, 12, 0, 1));        // no probe: nothing to read the link from
        CHECK(!link_rule_enabled(false, true, true, 12, 0, 1));        // HIP
    }
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
