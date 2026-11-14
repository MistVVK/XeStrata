// How a long prompt segment is cut into chunks (equal_cut_rule.hpp): no GPU, no model.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

#include "strata/prefill/equal_cut_rule.hpp"

namespace ec = strata::prefill::equal_cut;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// The older opt-in's lambda as it stands in generate.cpp (equal_chunk with STRATA_PREFILL_EQUAL=1), kept here as the reference
static int64_t old_request_chunk(int64_t tokens, int64_t max_chunk) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255 ? tokens : ((tokens + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}
static int64_t old_equal_chunk(int64_t tokens, int64_t max_chunk, int64_t floor) {
    const int64_t plain = old_request_chunk(tokens, max_chunk);
    if (tokens <= 0 || max_chunk <= 0) return plain;
    const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0);
    const int64_t last = tokens - (n - 1) * max_chunk;
    if (n <= 1 || last < floor) return plain;
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}

// the chunks the reading loop makes of `tokens` at chunk `c` (prefill.cpp run_impl: T = min(C, n - c0)): count, smallest, largest
struct Cutout { int64_t n, lo, hi, last; };
static Cutout cut_of(int64_t tokens, int64_t c) {
    Cutout r{0, std::numeric_limits<int64_t>::max(), 0, 0};
    for (int64_t c0 = 0; c0 < tokens; c0 += c) {
        const int64_t t = std::min(c, tokens - c0);
        ++r.n;
        r.lo = std::min(r.lo, t);
        r.hi = std::max(r.hi, t);
        r.last = t;
    }
    return r;
}

int main() {
    const int64_t F = 1536;   // the plain pack's floor on a discrete card in a build without MMQ
    // ---- the switch
    CHECK(ec::pick(nullptr, true) == ec::Cut::equal);     // unset, one GPU: the equal cut
    CHECK(ec::pick("1", true) == ec::Cut::equal);
    CHECK(ec::pick("0", true) == ec::Cut::today);         // the way back
    CHECK(ec::pick("0", false) == ec::Cut::today);
    CHECK(ec::pick(nullptr, false) == ec::Cut::today);    // a layer split: today's cut
    CHECK(ec::pick("1", false) == ec::Cut::opt_in);       // a layer split that asks for the older opt-in keeps it
    CHECK(ec::pick("", true) == ec::Cut::equal);
    CHECK(ec::pick("", false) == ec::Cut::today);

    // ---- the examples measured on the RTX 5070 (auto chunk 10,240 without MMQ, 9,984 with it)
    {
        const int64_t c = ec::equal_chunk(20932, 10240, F);   // today 10,240 x 2 + 452
        CHECK(c == 6978);
        const Cutout k = cut_of(20932, c);
        CHECK(k.n == 3 && k.hi == 6978 && k.last == 6976);
    }
    CHECK(ec::equal_chunk(32000, 10240, F) == 8000 && cut_of(32000, 8000).n == 4 && cut_of(32000, 8000).lo == 8000);
    CHECK(ec::equal_chunk(14000, 10240, F) == 7000 && cut_of(14000, 7000).n == 2);
    CHECK(ec::equal_chunk(20000, 9984, F) == 6667 && cut_of(20000, 6667).n == 3 && cut_of(20000, 6667).last == 6666);
    // 2 x 5,250 is under kEqualMinChunk at a 10,240 chunk: today's 10,240 + 260 (before the kEqualMinChunk rule: 5,250)
    CHECK(ec::equal_chunk(10500, 10240, F) == 10240 && ec::declined_equal(10500, 10240, F) == 5250);
    CHECK(ec::equal_chunk(20000, 10240, F) == 10000);
    CHECK(ec::equal_chunk(22980, 10240, F) == 7660);
    CHECK(ec::equal_chunk(18932, 10240, F) == 9466);
    // the engine reads N - 1 prompt tokens in a batch: 20,931 -> 3 x 6,977
    CHECK(ec::equal_chunk(20931, 10240, F) == 6977 && cut_of(20931, 6977).lo == 6977 && cut_of(20931, 6977).n == 3);
    // ---- a segment that fits one chunk is unchanged (today's chunk: rounded up to 256, at most the chunk)
    for (int64_t t : {1, 255, 256, 300, 999, 1000, 1536, 8000, 10239, 10240})
        CHECK(ec::equal_chunk(t, 10240, F) == ec::today_chunk(t, 10240));
    CHECK(ec::equal_chunk(8000, 10240, F) == 8192);
    CHECK(ec::equal_chunk(10240, 10240, F) == 10240);
    CHECK(ec::equal_chunk(10241, 10240, F) == 10240);      // 2 x 5,121 is short: today's 10,240 + 1 (before: 5,121)
    // ---- kEqualMinChunk (RTX 4090 24 GB, IQ3_S, chunk 8,192): an equal chunk under 6,144 keeps today's cut
    CHECK(ec::equal_chunk(8279, 8192, 3072) == 8192 && ec::declined_equal(8279, 8192, 3072) == 4140);    // 8,192 + 87
    CHECK(ec::equal_chunk(11192, 8192, 3072) == 8192 && ec::declined_equal(11192, 8192, 3072) == 5596);  // 8,192 + 3,000
    CHECK(ec::equal_chunk(16884, 8192, 3072) == 8192 && ec::declined_equal(16884, 8192, 3072) == 5628);  // 8,192 x 2 + 500
    CHECK(ec::equal_chunk(20000, 8192, 3072) == 6667 && ec::declined_equal(20000, 8192, 3072) == 0);     // 3 x 6,667 stays
    CHECK(ec::equal_chunk(32000, 8192, 3072) == 8000 && ec::declined_equal(32000, 8192, 3072) == 0);     // 4 x 8,000 stays
    CHECK(ec::equal_chunk(12288, 8192, 3072) == 6144 && ec::declined_equal(12288, 8192, 3072) == 0);
    CHECK(ec::equal_chunk(12287, 8192, 3072) == 6144);     // ceil(12,287 / 2) = 6,144: at the limit, equal
    CHECK(ec::equal_chunk(12286, 8192, 3072) == 8192);     // 6,143: short, today's 8,192 + 4,094
    // a configured chunk under 8,192 keeps the rule as before (4,096: 2 x 3,500 stays equal)
    CHECK(ec::equal_chunk(7000, 4096, 3072) == 3500 && ec::declined_equal(7000, 4096, 3072) == 0);
    CHECK(!ec::short_equal(3500, 4096) && ec::short_equal(6143, 8192) && !ec::short_equal(6144, 8192));
    // ---- keep_equal: today's cut was handed back but cannot be lent; the equal cut can (no GPU needed)
    CHECK(ec::keep_equal(900, 700, 800));                  // today's loan over the room, the equal one under it: keep equal
    CHECK(ec::keep_equal(0, 700, 800));                    // today's cannot be laid out at all
    CHECK(!ec::keep_equal(700, 600, 800));                 // today's fits: today's cut
    CHECK(!ec::keep_equal(900, 850, 800));                 // neither fits: today's cut, halved by the caller (#2090)
    CHECK(!ec::keep_equal(900, 0, 800));                   // the equal one cannot be laid out
    CHECK(ec::keep_equal(801, 800, 800) && !ec::keep_equal(800, 800, 800));   // the room is inclusive
    // ---- the cases the change is for: today's cut leaves a short tail under the floor, the equal cut evens it out
    CHECK(ec::today_chunk(20932, 10240) == 10240 && cut_of(20932, 10240).last == 452);   // today: 10,240 x 2 + 452
    CHECK(cut_of(20932, ec::equal_chunk(20932, 10240, F)).n == 3 && cut_of(20932, ec::equal_chunk(20932, 10240, F)).lo >= 6976);
    CHECK(cut_of(32000, 10240).last == 1280);                                             // today: 10,240 x 3 + 1,280
    CHECK(ec::equal_chunk(32000, 10240, F) == 8000);                                      // 4 x 8,000
    CHECK(ec::equal_chunk(32000, 10240, 3072) == 8000);                                   // the CPU share's floor too
    // ---- a small chunk: the EQUAL chunk itself would fall under the floor (auto chunk 4,096, CPU share floor 3,072):
    // today's 4,096 + 904 keeps one chunk on the streamed walk; 2 x 2,500 would put both under the floor
    CHECK(ec::equal_chunk(5000, 4096, 3072) == 4096);
    CHECK(ec::equal_chunk(7000, 4096, 3072) == 3500);      // 2 x 3,500: both at or above the floor, evened
    CHECK(ec::equal_chunk(6200, 4096, 3072) == 3100);      // evened from 2 x 3,072 up
    CHECK(ec::equal_chunk(6100, 4096, 3072) == 4096);      // 2 x 3,050: under the floor, today's cut
    // ---- a cut whose chunks would fall under the floor keeps today's cut (a small chunk)
    CHECK(ec::equal_chunk(2100, 2048, F) == 2048);         // 2 x 1,050 would both walk staged
    CHECK(ec::equal_chunk(3100, 2048, F) == 1550);         // 2 x 1,550: both at or above 1,536
    CHECK(ec::equal_chunk(3071, 2048, F) == 2048);         // 1,536 + 1,535: the last under the floor
    CHECK(ec::equal_chunk(3072, 2048, F) == 1536);
    // ---- degenerate input
    CHECK(ec::equal_chunk(0, 10240, F) == 0 && ec::equal_chunk(-5, 10240, F) == 0 && ec::equal_chunk(5000, 0, F) == 0);
    CHECK(ec::chunk_count(0, 5) == 0 && ec::chunk_count(5, 0) == 0 && ec::chunk_count(10, 5) == 2 && ec::chunk_count(11, 5) == 3);

    // ---- the properties, over a sweep of segment lengths and chunks (non-multiples of 256 too: a numeric --prefill)
    long long cases = 0;
    for (int64_t mc : {1024, 2048, 3072, 4096, 6144, 7000, 8192, 9216, 9984, 10240, 10496, 16384, 32768}) {
        for (int64_t floor : {64, 1024, 1536, 3072}) {
            for (int64_t t = 1; t <= 140000; t += (t < 4 * mc ? 7 : 997)) {
                ++cases;
                const int64_t today = ec::today_chunk(t, mc);
                const int64_t e = ec::equal_chunk(t, mc, floor);
                const Cutout kt = cut_of(t, today), ke = cut_of(t, e);
                CHECK(e > 0 && e <= mc);                       // within the buffers laid out for the chunk
                CHECK(ke.n <= kt.n);                           // never more chunks than today's cut
                if (t <= mc) { CHECK(e == today); continue; }   // one chunk: unchanged
                if (e == today) { CHECK(ke.n == kt.n); continue; }   // kept today's cut (a chunk would fall under the floor)
                CHECK(ke.n == ec::chunk_count(t, mc));           // exactly ceil(t / chunk) chunks
                CHECK(ke.hi - ke.lo <= ke.n - 1);                // equal, within the n - 1 tokens the division leaves
                CHECK(ke.lo == ke.last && ke.lo >= floor);       // only the last is shorter, and it streams
            }
        }
    }
    // ---- the older opt-in rule is the old lambda, value for value
    for (int64_t mc : {1024, 3072, 7000, 9984, 10240}) {
        for (int64_t floor : {1024, 1536, 3072}) {
            for (int64_t t = 0; t <= 70000; t += 13) {
                CHECK(ec::opt_in_chunk(t, mc, floor) == old_equal_chunk(t, mc, floor));
                CHECK(ec::today_chunk(t, mc) == old_request_chunk(t, mc));
            }
        }
    }
    // ---- chunk(): the three cuts
    CHECK(ec::chunk(ec::Cut::today, 20932, 10240, F) == 10240);
    CHECK(ec::chunk(ec::Cut::equal, 20932, 10240, F) == 6978);
    CHECK(ec::chunk(ec::Cut::opt_in, 20932, 10240, F) == 10240);   // the opt-in keeps a 452-token last chunk under the floor
    CHECK(ec::chunk(ec::Cut::opt_in, 22980, 10240, F) == 7680);    // 2,500 above the floor: evened, on the 256 grid
    CHECK(ec::chunk(ec::Cut::equal, 22980, 10240, F) == 7660);
    // ---- the loan guard
    CHECK(ec::loan_ok(100, 120) && ec::loan_ok(120, 120));
    CHECK(!ec::loan_ok(121, 120));      // a larger loan than today's: keep today's cut
    CHECK(!ec::loan_ok(0, 120) && !ec::loan_ok(-1, 120) && !ec::loan_ok(100, 0));
    // constexpr: usable at compile time
    static_assert(ec::equal_chunk(32000, 10240, 1536) == 8000, "constexpr equal cut");
    static_assert(ec::pick("0", true) == ec::Cut::today, "constexpr switch");

    if (failed) {
        std::fprintf(stderr, "equal_cut_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("equal_cut_rule_test: all checks passed (%lld swept cases)\n", cases);
    return 0;
}
