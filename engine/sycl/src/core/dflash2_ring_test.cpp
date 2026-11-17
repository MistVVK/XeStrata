// sycl/src/core/dflash2_ring_test.cpp - DF2-SYCL: the drafter's ring log on an Intel GPU.  A drafter with the real Q4_0 pack and a random head fills its K/V ring past
// the wrap (positions 0..2999 of one conversation), the ring is snapshotted, a longer turn overwrites part of it (3000..3199), and a request that resumes at
// 3000 asks begin_request for the window [n - 2047, 3000): the slots a later position took must be rebuilt from the log, bitwise the original rows.  Then another
// conversation takes the ring (the first one is parked in the stash), and the first comes back: the whole window is rebuilt.
//   dflash2_ring_test PACK
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/dflash2.hpp"
#include "strata/kernels/dflash2.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using strata::core::Dflash2Drafter;

static uint32_t rs = 777;
static float frand() { rs = rs * 1664525u + 1013904223u; return (float) ((rs >> 8) & 0xffff) / 65536.0f - 0.5f; }

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dflash2_ring_test PACK\n"); return 2; }
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s\n", q->get_device().get_info<sycl::info::device::name>().c_str());
    Dflash2Drafter d;
    std::string err;
    if (!d.load(argv[1], 0, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const int HR = 4096, RB = 2560 / 32 * 34;
    std::vector<uint8_t> hw((size_t) HR * RB);
    for (auto& b : hw) b = (uint8_t) (rs = rs * 1664525u + 1013904223u, rs >> 12);
    for (size_t r = 0; r < (size_t) HR * (2560 / 32); ++r) { hw[r * 34] = 0x00; hw[r * 34 + 1] = 0x1c; }   // finite scales
    uint8_t* head = sycl::malloc_device<uint8_t>(hw.size(), *q);
    q->memcpy(head, hw.data(), hw.size()).wait();
    auto embed = [](const int32_t*, int n, float* out, sycl::queue* s) { s->memset(out, 0, (size_t) n * 2560 * 4); };
    if (!d.bind_raw(embed, head, 8, HR, {}, err)) { std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1; }
    const int TAPN = 5 * 2560, CH = 64;
    float* taps = sycl::malloc_device<float>((size_t) CH * TAPN, *q);
    std::vector<float> ht((size_t) CH * TAPN);
    auto feed = [&](const std::vector<int32_t>& ids, int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; p += CH) {
            const int m = (int) std::min<int64_t>(CH, p1 - p);
            for (auto& v : ht) v = frand() * 2.0f;
            q->memcpy(taps, ht.data(), (size_t) m * TAPN * 4).wait();
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
            const int s = (int) (p % Dflash2Drafter::kRing);
            if ((tg[(size_t) s] & 0xffffffffll) != p) { ++bad_tag; continue; }
            bool differs = false;
            for (int l = 0; l < 5 && !differs; ++l)
                for (int kvsel = 0; kvsel < 2 && !differs; ++kvsel)
                    for (int h = 0; h < 8 && !differs; ++h) {
                        const size_t off = ((size_t) l * 2 + kvsel) * per + ((size_t) h * Dflash2Drafter::kRing + s) * 128;
                        if (std::memcmp(&kv[off], &kvA[off], 128 * 4) != 0) differs = true;
                    }
            if (differs) ++bad_kv;
        }
        char b[200];
        std::snprintf(b, sizeof b, "%s: positions %lld..%lld, %lld wrong tags, %lld K/V rows differ from the original update", what, (long long) lo, (long long) hi - 1,
                      (long long) bad_tag, (long long) bad_kv);
        check(bad_tag == 0 && bad_kv == 0, b);
    };
    if (!d.debug_ring(kvB, tgB, err)) return 1;
    {
        int64_t bad = 0;
        for (int64_t p = 983; p < 1136; ++p) if ((tgB[(size_t) (p % Dflash2Drafter::kRing)] & 0xffffffffll) != p) ++bad;
        check(bad > 0, "before the request, slots of positions 983..1135 hold later positions (the test can tell)");
    }
    d.begin_request(big.data(), 3030, 3000, &cold);
    std::printf("  repeat/rewind: rebuilt %lld rows (missing %lld) in %.2f ms, cold %d\n", (long long) d.last_rebuilt, (long long) d.last_missing, d.last_ms, (int) cold);
    check(d.last_rebuilt > 0 && d.last_missing == 0 && !cold, "begin_request rebuilt rows from the log, none missing, not cold");
    if (!d.debug_ring(kvC, tgC, err)) return 1;
    compare(kvC, tgC, 3030 - 2047, 3000, "after the rebuild");

    // another conversation takes the ring; the first one is parked, then comes back
    d.begin_request(other.data(), 400, 0, &cold);
    feed(other, 0, 400);
    check(d.ring_stats.stash_stores >= 1, "another conversation starting at 0: the first lineage is kept in the stash as a branch");
    d.begin_request(big.data(), 3030, 3000, &cold);
    std::printf("  parked lineage back: rebuilt %lld rows (missing %lld) in %.2f ms, swapped %d, cold %d\n", (long long) d.last_rebuilt, (long long) d.last_missing, d.last_ms, (int) d.last_swapped, (int) cold);
    check(d.last_swapped && d.last_rebuilt >= 1900 && d.last_missing == 0 && !cold, "the parked lineage came back, the whole window rebuilt, none missing");
    if (!d.debug_ring(kvC, tgC, err)) return 1;
    compare(kvC, tgC, 3030 - 2047, 3000, "after the parked lineage came back");

    // an edited prompt: the history differs at 2500, resume 2800: not covered, no stash match: starts over, cold
    std::vector<int32_t> edit = big;
    edit[2500] ^= 1;
    d.begin_request(edit.data(), 2830, 2800, &cold);
    check(cold || d.last_missing > 0, "an edited conversation: the drafter reports it has no log of the window (cold)");
    std::printf("%s\n", g_fail ? "SOME CHECKS FAILED" : "ALL RING CHECKS PASSED");
    return g_fail ? 1 : 0;
}
