// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/share_rules.hpp"

using strata::prefill::share_rules::default_share_has_work;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const long long pairs = 48LL * 512;
    CHECK(default_share_has_work(pairs - 1, pairs));       // one expert streams: the share has something to take
    CHECK(default_share_has_work(1500, pairs));            // a 12 GB card: most of them stream
    CHECK(!default_share_has_work(pairs, pairs));          // every expert resident: nothing to take
    CHECK(!default_share_has_work(pairs + 100, pairs));    // a cache with spare slots
    CHECK(default_share_has_work(0, pairs));               // empty cache
    CHECK(default_share_has_work(100000, 0));              // total unknown: keep today's behaviour
    static_assert(!default_share_has_work(10, 10), "constexpr");

    // H11: memory verdicts
    using namespace strata::prefill::share_rules;
    const uint64_t G = 1ull << 30;
    ShareMemory m;
    m.buffer_bytes = G / 3;
    CHECK(default_share_memory_verdict(m).arm);                                   // nothing known: today's behaviour
    m.avail_ram = 20 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // plenty of RAM
    m.avail_ram = 3 * G;
    CHECK(!default_share_memory_verdict(m).arm);                                  // RAM below buffers + margin
    CHECK(default_share_memory_verdict(m).why.find("RAM is free") != std::string::npos);
    m.avail_ram = kShareRamMargin + m.buffer_bytes;
    CHECK(default_share_memory_verdict(m).arm);                                   // exactly enough
    m.avail_ram = kShareRamMargin + m.buffer_bytes - 1;
    CHECK(!default_share_memory_verdict(m).arm);                                  // one byte short
    m.avail_ram = 20 * G;
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
    // the link rule: the two cards it was measured on, and what it must leave alone
    {
        LinkRuleIn in;
        in.total = pairs; in.blob_mb = 1.747; in.pool_workers = 19;
        in.link_gbps = 6.4; in.resident = 3802;            // RTX 4070: a 5.7 s floor, staged up to 6K tokens
        LinkRule r = link_rule(in);
        CHECK(r.arm && r.limit == 6144 && r.nt == kLinkNt);
        in.pool_workers = 8;                               // a smaller pool takes less
        CHECK(link_rule(in).limit < r.limit && link_rule(in).limit >= kShareLimitDefault);
        in.pool_workers = 19; in.link_gbps = 28.2; in.resident = 14227;   // Arc Pro B70: a 0.64 s floor, unchanged
        r = link_rule(in);
        CHECK(!r.arm && r.limit == kShareLimitDefault && r.nt == 0);
        in.link_gbps = 0.5; in.resident = 0;               // a very slow link: capped
        CHECK(link_rule(in).limit == kShareLimitMax);
        in.link_gbps = 0.0;                                // not probed
        CHECK(!link_rule(in).arm && link_rule(in).limit == 0);
        in.link_gbps = 6.4; in.resident = pairs;           // every expert on the card
        CHECK(!link_rule(in).arm);
    }
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
