// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/dflash2_ring_test.cpp - the drafter's ring log (GPU, the real pack).  A drafter with a random head fills
// its K/V ring past the wrap (positions 0..2999 of one conversation), the ring is snapshotted, a longer turn
// overwrites part of it (3000..3199), and a request that resumes at 3000 asks begin_request for the window
// [n - 2047, 3000): the slots a later position took must be rebuilt from the log, bitwise the original rows.  Then
// another conversation takes the ring (the first is parked in the stash), and the first comes back: the whole window
// is rebuilt.
//   dflash2_ring_test PACK
#include "strata/core/dflash2.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using strata::core::Dflash2Drafter;

static uint32_t rs = 777;
static float frand() { rs = rs * 1664525u + 1013904223u; return (float) ((rs >> 8) & 0xffff) / 65536.0f - 0.5f; }

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dflash2_ring_test PACK\n"); return 2; }
    try { strata::core::Runtime::get(); } catch (const std::exception& e) {
        std::printf("dflash2_ring_test: no GPU (%s) - skipped\n", e.what());
        return 77;
    }
    Dflash2Drafter d;
    std::string err;
    if (!d.load(argv[1], 0, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const int HR = 4096, RB = 2560 / 32 * 34;
    std::vector<uint8_t> hw((size_t) HR * RB);
    for (auto& b : hw) b = (uint8_t) (rs = rs * 1664525u + 1013904223u, rs >> 12);
    for (size_t r = 0; r < (size_t) HR * (2560 / 32); ++r) { hw[r * 34] = 0x00; hw[r * 34 + 1] = 0x1c; }   // finite scales
    uint8_t* head = nullptr;
    if (!strata::gpu::alloc_device(&head, hw.size()) || !strata::gpu::copy(head, hw.data(), hw.size())) {
        std::fprintf(stderr, "head upload\n");
        return 1;
    }
    auto embed = [](const int32_t*, int n, float* out, void* st, std::string&) {
        return strata::gpu::memset_async(out, 0, (size_t) n * 2560 * 4, st);
    };
    auto headfn = [](const uint8_t* rows, const uint8_t* xq, const float*, int n, float* y, void* st) {
        strata::kernels::native_mmvq(8, rows, xq, y, 2560, HR, n, st);
    };
    if (!d.bind_raw(embed, headfn, 8, HR, {}, err)) { std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1; }
    const int TAPN = 5 * 2560, CH = 64;
    float* taps = nullptr;
    if (!strata::gpu::alloc_device(&taps, (size_t) CH * TAPN * 4)) { std::fprintf(stderr, "taps alloc\n"); return 1; }
    std::vector<float> ht((size_t) CH * TAPN);
    auto feed = [&](const std::vector<int32_t>& ids, int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; p += CH) {
            const int m = (int) std::min<int64_t>(CH, p1 - p);
            for (auto& v : ht) v = frand() * 2.0f;
            if (!strata::gpu::copy(taps, ht.data(), (size_t) m * TAPN * 4)) { std::fprintf(stderr, "taps copy\n"); std::exit(1); }
            if (!d.context(taps, m, p, err)) { std::fprintf(stderr, "context: %s\n", err.c_str()); std::exit(1); }
            d.note_rows(ids.data() + p, m, p);
        }
    };
    std::vector<int32_t> big(4000), other(1000);
    for (int i = 0; i < 4000; ++i) big[(size_t) i] = (i * 7919) % 5000;
    for (int i = 0; i < 1000; ++i) other[(size_t) i] = (i * 104729 + 17) % 5000;
    std::printf("ring log %s\n", d.ring_log_on() ? "on" : "OFF");

    d.reset();
    bool cold = false;
    d.begin_request(big.data(), 3031, 0, &cold);
    feed(big, 0, 3000);
    std::vector<float> kvA, kvB, kvC;
    std::vector<int64_t> tgA, tgB, tgC;
    if (!d.debug_ring(kvA, tgA, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    feed(big, 3000, 3200);   // a longer turn: positions 3000..3199 take the slots of 936..1135
    const size_t per = (size_t) Dflash2Drafter::kRing * 8 * 128;
    auto compare = [&](const std::vector<float>& kv, const std::vector<int64_t>& tg, int64_t lo, int64_t hi, const char* what) {
        int64_t bad_tag = 0, bad_kv = 0;
        for (int64_t p = lo; p < hi; ++p) {
            const int sl = (int) (p % Dflash2Drafter::kRing);
            if ((tg[(size_t) sl] & 0xffffffffll) != p) { ++bad_tag; continue; }
            bool differs = false;
            for (int l = 0; l < 5 && !differs; ++l)
                for (int kvsel = 0; kvsel < 2 && !differs; ++kvsel)
                    for (int h = 0; h < 8 && !differs; ++h) {
                        const size_t off = ((size_t) l * 2 + kvsel) * per + ((size_t) h * Dflash2Drafter::kRing + sl) * 128;
                        for (int i = 0; i < 128 && !differs; ++i)   // the bits: a copy keeps NaNs and -0 as they are
                            differs = std::bit_cast<uint32_t>(kv[off + i]) != std::bit_cast<uint32_t>(kvA[off + i]);
                    }
            if (differs) ++bad_kv;
        }
        char b[200];
        std::snprintf(b, sizeof b, "%s: positions %lld..%lld, %lld wrong tags, %lld K/V rows differ", what,
                      (long long) lo, (long long) hi - 1, (long long) bad_tag, (long long) bad_kv);
        check(bad_tag == 0 && bad_kv == 0, b);
    };
    if (!d.debug_ring(kvB, tgB, err)) return 1;
    {
        int64_t bad = 0;
        for (int64_t p = 983; p < 1136; ++p) if ((tgB[(size_t) (p % Dflash2Drafter::kRing)] & 0xffffffffll) != p) ++bad;
        check(bad > 0, "before the request, slots of 983..1135 hold later positions (the test can tell)");
    }
    d.begin_request(big.data(), 3030, 3000, &cold);
    std::printf("  repeat/rewind: rebuilt %lld rows (missing %lld) in %.2f ms, cold %d\n",
                (long long) d.last_rebuilt, (long long) d.last_missing, d.last_ms, (int) cold);
    check(d.last_rebuilt > 0 && d.last_missing == 0 && !cold, "begin_request rebuilt rows from the log, none missing, not cold");
    if (!d.debug_ring(kvC, tgC, err)) return 1;
    compare(kvC, tgC, 3030 - 2047, 3000, "after the rebuild");

    // another conversation takes the ring; the first is parked in the stash as a branch, then comes back
    d.begin_request(other.data(), 400, 0, &cold);
    feed(other, 0, 400);
    check(d.ring_stats.stash_stores >= 1, "another conversation at 0: the first lineage is kept in the stash as a branch");
    d.begin_request(big.data(), 3030, 3000, &cold);
    std::printf("  stash return: rebuilt %lld rows (missing %lld), swapped %d, cold %d\n",
                (long long) d.last_rebuilt, (long long) d.last_missing, (int) d.last_swapped, (int) cold);
    check(d.last_swapped && d.last_rebuilt >= 1900 && d.last_missing == 0 && !cold,
          "the parked lineage came back, the whole window rebuilt, none missing");
    if (!d.debug_ring(kvC, tgC, err)) return 1;
    compare(kvC, tgC, 3030 - 2047, 3000, "after the stash return");

    // an edited prompt: the history differs at 2500, resume 2800: not covered, no stash match: starts over, cold
    std::vector<int32_t> edit = big;
    edit[2500] ^= 1;
    d.begin_request(edit.data(), 2830, 2800, &cold);
    check(cold || d.last_missing > 0, "an edited conversation: the drafter reports it has no log of the window (cold)");

    strata::gpu::free(head);
    strata::gpu::free(taps);
    std::printf("dflash2_ring_test: %s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}