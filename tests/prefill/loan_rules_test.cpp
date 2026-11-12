// A prompt piece below the walk floor lends buffers for the staged walk, not for the rounded chunk's streamed one: no GPU, no model.
#include <cstdint>
#include <cstdio>

#include "strata/prefill/loan_rules.hpp"

using strata::prefill::loan_rules::walk_chunk;

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
    for (int64_t floor : {64, 1024, 2048, 3072, 4096})
        for (int64_t t = 1; t <= 9000; t += 7) {
            const int64_t c = walk_chunk(t, grid(t), floor);
            CHECK(c <= grid(t));
            CHECK(c >= t || grid(t) < t);
            if (t < floor) CHECK(c < floor);
        }
    // empty or negative requests are passed through
    CHECK(walk_chunk(0, 256, 1024) == 256 && walk_chunk(10, 0, 1024) == 0);
    static_assert(walk_chunk(999, 1024, 1024) == 1023, "constexpr");
    if (failed != 0) return 1;
    std::printf("loan_rules_test: all checks passed\n");
    return 0;
}
