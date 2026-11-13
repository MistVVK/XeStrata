// A prompt piece below the walk floor lends buffers for the staged walk, not for the rounded chunk's streamed one: no GPU, no model.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/loan_rules.hpp"

using strata::prefill::loan_rules::walk_chunk;
using strata::prefill::loan_rules::tail_need;
using strata::prefill::loan_rules::tail_layout;
using strata::prefill::loan_rules::segment_tail;
using strata::prefill::loan_rules::reuse_ok;
using strata::prefill::loan_rules::next_tail;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// the request's own rounding (generate.cpp request_chunk): up to the 256-token grid
static int64_t grid(int64_t t) { return (t + 255) / 256 * 256; }

int main() {
    // the default floor: 999 tokens round to 1,024 (the streamed layout) and walk staged -> 1,023
    CHECK(grid(999) == 1024 && walk_chunk(999, grid(999), 1024) == 1023);
    CHECK(walk_chunk(769, grid(769), 1024) == 1023);      // the first piece that rounds to the floor
    CHECK(walk_chunk(768, grid(768), 1024) == 768);       // an exact multiple under the floor: nothing to correct
    CHECK(walk_chunk(1023, grid(1023), 1024) == 1023);
    CHECK(walk_chunk(1024, grid(1024), 1024) == 1024);    // at the floor: the streamed walk, the chunk stays
    CHECK(walk_chunk(1025, grid(1025), 1024) == 1280);
    CHECK(walk_chunk(300, grid(300), 1024) == 512);       // far under the floor: the staged layout already
    // the CPU share's limit
    CHECK(walk_chunk(3071, grid(3071), 3072) == 3071);
    CHECK(walk_chunk(2817, grid(2817), 3072) == 3071);
    CHECK(walk_chunk(2816, grid(2816), 3072) == 2816);
    CHECK(walk_chunk(3000, grid(3000), 3072) == 3071);
    CHECK(walk_chunk(3072, grid(3072), 3072) == 3072);
    CHECK(walk_chunk(3073, grid(3073), 3072) == 3328);
    // exact multiples of 256 on a floor that is one
    for (int64_t t = 256; t <= 8192; t += 256) CHECK(walk_chunk(t, t, 1024) == t);
    // a floor of 64 (the Windows gfx1151 rule): a 63-token piece rounds to 256 and walks staged
    CHECK(walk_chunk(63, grid(63), 64) == 63);
    CHECK(walk_chunk(1, grid(1), 64) == 63);
    CHECK(walk_chunk(64, grid(64), 64) == 256);
    CHECK(walk_chunk(100, grid(100), 64) == 256);
    // a floor of 0 or 1: every piece streams, nothing to correct
    CHECK(walk_chunk(999, grid(999), 0) == 1024);
    CHECK(walk_chunk(1, grid(1), 1) == 256);
    // the chunk configured below the floor stays: a piece of 999 under a 512-token chunk reads in several pieces
    CHECK(walk_chunk(999, 512, 1024) == 512);
    // never larger than the chunk the caller chose, never below the piece itself
    for (int64_t floor : std::initializer_list<int64_t>{64, 1024, 2048, 3072, 4096})
        for (int64_t t = 1; t <= 9000; t += 7) {
            const int64_t c = walk_chunk(t, grid(t), floor);
            CHECK(c <= grid(t));
            CHECK(c >= t || grid(t) < t);
            if (t < floor) CHECK(c < floor);
        }
    // empty or negative requests are passed through
    CHECK(walk_chunk(0, 256, 1024) == 256 && walk_chunk(10, 0, 1024) == 0);
    static_assert(walk_chunk(999, 1024, 1024) == 1023, "constexpr");

    // ---- the tail of a lent fused layout (tail_need / tail_layout / segment_tail / reuse_ok)
    // one chunk at or above the floor: no MMQ rows at all
    CHECK(tail_need(1024, 1024, 1024) == 0);
    CHECK(tail_need(8192, 8192, 1024) == 0);
    CHECK(tail_need(4096, 2048, 1024) == 0);              // two full chunks
    // a short last chunk is the whole need
    CHECK(tail_need(4097, 2048, 1024) == 1);
    CHECK(tail_need(5000, 2048, 1024) == 904);
    CHECK(tail_need(5120 + 1023, 5120, 1024) == 1023);
    CHECK(tail_need(5120 + 1024, 5120, 1024) == 0);       // a last chunk at the floor streams: fused
    // a floor of 3,584 (the rule on an RTX A6000): the same segments now have a tail up to 3,583
    CHECK(tail_need(4097, 2048, 3584) == 1);
    CHECK(tail_need(8193 + 2000, 8192, 3584) == 2001);
    CHECK(tail_need(8192 + 3584, 8192, 3584) == 0);
    // a segment read in one chunk below the floor, in a layout whose chunk reaches it (a reused big loan)
    CHECK(tail_need(2000, 8192, 3584) == 2000);
    CHECK(tail_need(2000, 8192, 1024) == 0);
    // a layout whose own chunk is below the floor is not fused: every chunk is MMQ's
    CHECK(tail_need(5000, 1023, 1024) == 1023 && tail_need(500, 1023, 1024) == 500);
    CHECK(tail_need(0, 4096, 1024) == 0 && tail_need(10, 0, 1024) == 0);
    // never above floor - 1, never negative, last chunk exact for every size
    for (int64_t floor : std::initializer_list<int64_t>{1, 64, 1024, 3072, 3584})
        for (int64_t c : std::initializer_list<int64_t>{256, 1023, 1024, 2048, 4096, 8192})
            for (int64_t t = 1; t <= 20000; t += 37) {
                const int64_t need = tail_need(t, c, floor);
                CHECK(need >= 0 && need <= (floor > 1 ? floor - 1 : 0));
                int64_t worst = 0;                          // walk the chunks one by one
                for (int64_t at = 0; at < t; at += c) {
                    const int64_t len = t - at < c ? t - at : c;
                    if (len < floor && len > worst) worst = len;
                }
                CHECK(need == worst);
                const int64_t lay = tail_layout(need, floor);
                CHECK(lay >= need && (need == 0 ? lay == 0 : lay % 256 == 0 || lay == floor - 1));
                CHECK(lay <= (floor > 1 ? floor - 1 : 0));
                CHECK(segment_tail(t, c, floor) == lay);
            }
    // the grid: 904 -> 1,024; clamped to today's floor - 1
    CHECK(tail_layout(904, 3584) == 1024 && tail_layout(1, 3584) == 256 && tail_layout(1023, 1024) == 1023);
    CHECK(tail_layout(0, 3584) == 0 && tail_layout(5, 1) == 0);
    // the overflow case: a loan made for an 8,193+ segment whose last chunk is 1 token, then a 2,000-token segment
    // (the 2,000 tokens read as ONE chunk of the loan's layout, staged under a 3,584 floor)
    {
        const int64_t floor = 3584, lent_chunk = 8192;
        const int64_t lent_tail = segment_tail(8193, lent_chunk, floor);            // 256: the last chunk is 1 token
        CHECK(lent_tail == 256);
        CHECK(!reuse_ok(2048, 2000, lent_chunk, lent_tail, floor));                 // needs 2,000 (grid 2,048): relayout
        CHECK(tail_need(2000, lent_chunk, floor) > lent_tail);
        CHECK(reuse_ok(256, 100, lent_chunk, lent_tail, floor));                    // 100 <= 256: reuse
        CHECK(reuse_ok(2048, 4096 + 100, lent_chunk, lent_tail, floor) == false);   // chunk 8,192 > 4,196: one chunk of 4,196 >= floor: need 0
        CHECK(tail_need(4196, lent_chunk, floor) == 0);                              // (it is fused, so it reuses)
    }
    // a smaller want never reuses a smaller loan, a bigger tail never reuses a smaller one
    CHECK(!reuse_ok(4096, 4096, 2048, 4095, 1024));
    CHECK(reuse_ok(2048, 4096, 2048, 0, 1024));
    CHECK(reuse_ok(2048, 4096, 4096, 0, 1024) && !reuse_ok(2048, 4097 + 4096, 4096, 0, 8192));
    // ---- grow-only (next_tail): need grows, need shrinks, chunk grows
    {
        const int64_t floor = 3584;
        CHECK(next_tail(8192, -1, 8192, 256, floor) == 256);          // a start-up layout (default-sized): the need of the moment
        CHECK(next_tail(8192, 1024, 8192, 256, floor) == 1024);       // need shrinks: the tail stays (no relayout)
        CHECK(next_tail(8192, 1024, 8192, 0, floor) == 1024);
        CHECK(next_tail(8192, 1024, 8192, 1024, floor) == 1024);      // equal: stays
        CHECK(next_tail(8192, 1024, 8192, 1280, floor) == 3583);      // need grows: today's size, once
        CHECK(next_tail(8192, 3583, 8192, 3000, floor) == 3583);      // and then it is today's: nothing grows past it
        CHECK(next_tail(8192, 1024, 4096, 256, floor) == 256);        // chunk shrinks: another layout, from its own need
        CHECK(next_tail(4096, 1024, 8192, 2048, floor) == 2048);      // chunk grows: from the need of that moment
        CHECK(next_tail(8192, 0, 8192, 0, floor) == 0);               // no tail needed, none carved
        CHECK(next_tail(8192, 0, 8192, 256, floor) == 3583);
        CHECK(next_tail(8192, 256, 8192, 256, 1) == 256 && next_tail(8192, 0, 8192, 5, 1) == 0);
        // a sequence of needs on one chunk: at most one growth, every layout tail covers its segment, never shrinks
        int64_t cap = -1, grown = 0;
        for (int64_t need : std::initializer_list<int64_t>{256, 0, 1024, 256, 2048, 3583, 0, 512, 3000}) {
            const int64_t t = next_tail(8192, cap, 8192, need, floor);
            CHECK(t >= need);
            if (cap >= 0) CHECK(t >= cap);
            if (t != cap) ++grown;
            cap = t;
        }
        CHECK(grown == 2);   // the first carve (256), one growth (1024 > 256 -> today's 3583), then it stays
    }
    static_assert(tail_need(5000, 2048, 1024) == 904 && segment_tail(5000, 2048, 1024) == 1023, "constexpr");
    if (failed != 0) return 1;
    std::printf("loan_rules_test: all checks passed\n");
    return 0;
}
