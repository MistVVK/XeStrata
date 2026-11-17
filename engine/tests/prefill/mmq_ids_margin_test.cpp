// The MMQ row-table margin (PR #5108): every index the stream-k fixup and the main kernel can read stays inside the table.
// No GPU: the reads are enumerated from the layout the way mul_mat_q_stream_k_fixup computes them
// (ids_dst[col_low + jt*J + j], jt < ceil(max_rows / J), j < J, col_low = the expert's first row).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/prefill/mmq_ids_margin.hpp"

using namespace strata::prefill::mmq;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// the highest index (exclusive) one launch can read: bounds has n + 1 entries, max_rows the widest expert, J any tile width
static int64_t fixup_read_end(const std::vector<int64_t>& bounds, int64_t max_rows, int J) {
    const int64_t tiles = (max_rows + J - 1) / J;
    int64_t end = 0;
    for (size_t e = 0; e + 1 < bounds.size(); ++e)
        end = std::max(end, bounds[e] + tiles * J);              // every expert's fixup reads whole tiles from its first row
    return end;
}
static int64_t main_read_end(const std::vector<int64_t>& bounds, int J) {
    int64_t end = 0;
    for (size_t e = 0; e + 1 < bounds.size(); ++e)
        if (bounds[e + 1] > bounds[e]) end = std::max(end, bounds[e + 1] + J - 1);   // up to J - 1 rows after the expert's last
    return end;
}

int main() {
    // the constants follow ggml's loop J = 8 .. 128
    CHECK(kMmqTileColsMax == 128 && kMmqTileColsStep == 8);

    // arithmetic: roundup(max_rows, J) + J
    int64_t m = -1;
    CHECK(mmq_ids_margin_rows(0, 128, &m) && m == 128);
    CHECK(mmq_ids_margin_rows(1, 128, &m) && m == 256);
    CHECK(mmq_ids_margin_rows(128, 128, &m) && m == 256);
    CHECK(mmq_ids_margin_rows(129, 128, &m) && m == 384);
    CHECK(mmq_ids_margin_rows(3072, 128, &m) && m == 3200);
    CHECK(mmq_ids_margin_rows(3072, 8, &m) && m == 3080);
    CHECK(mmq_ids_table_rows(30720, 3072) == 30720 + 3200);
    CHECK(mmq_ids_table_rows(0, 0) == 128);

    // checked arithmetic: nothing wraps to a small table
    m = 77;
    CHECK(!mmq_ids_margin_rows(-1, 128, &m) && m == 77);
    CHECK(!mmq_ids_margin_rows(10, 0, &m) && !mmq_ids_margin_rows(10, -8, &m));
    CHECK(!mmq_ids_margin_rows(INT64_MAX, 128, &m));
    CHECK(!mmq_ids_margin_rows(INT64_MAX - 100, 128, &m));
    CHECK(mmq_ids_table_rows(UINT64_MAX - 5, 10) == 0);
    CHECK(mmq_ids_table_rows(10, (uint64_t) INT64_MAX + 1) == 0);
    CHECK(mmq_ids_table_rows(UINT64_MAX, 0) == 0);

    // synthetic layouts: the table is T*K rows + the margin, T tokens, K = 10 experts per token; late experts, empty tiles
    const int64_t K = 10;
    std::mt19937_64 rng(5108);
    for (int iter = 0; iter < 2000; ++iter) {
        const int64_t T = 1 + (int64_t) (rng() % (iter < 1000 ? 70 : 4096));
        const int NE = 1 + (int) (rng() % 64);
        std::vector<int64_t> cnt((size_t) NE, 0);
        int64_t left = T * K, widest = 0;
        // experts take at most one row per token; some are empty, the last ones often are
        for (int e = 0; e < NE && left > 0; ++e) {
            const int64_t c = std::min<int64_t>({(int64_t) (rng() % ((uint64_t) T + 1)), T, left});
            cnt[(size_t) e] = c;
            left -= c;
        }
        // whatever is left goes to experts that still have room (the rows must add up to T*K when they can)
        for (int e = 0; e < NE && left > 0; ++e) {
            const int64_t room = T - cnt[(size_t) e], c = std::min(room, left);
            cnt[(size_t) e] += c;
            left -= c;
        }
        std::vector<int64_t> bounds{0};
        int64_t total = 0;
        for (int64_t c : cnt) { total += c; bounds.push_back(total); widest = std::max(widest, c); }
        CHECK(widest <= T);
        const int64_t table = (int64_t) mmq_ids_table_rows((uint64_t) total, (uint64_t) T);
        for (int J = kMmqTileColsStep; J <= kMmqTileColsMax; J += kMmqTileColsStep) {
            // the launch uses max_rows = the widest expert here; the margin is sized for max_rows <= T
            CHECK(fixup_read_end(bounds, widest, J) <= table);
            CHECK(fixup_read_end(bounds, T, J) <= table);       // even a launch whose grid is sized for T
            CHECK(main_read_end(bounds, J) <= table);
        }
    }

    // the exact table of before the fix does fail on the late-expert layout the PR found (the guard this test pins)
    {
        const int64_t T = 100;
        std::vector<int64_t> bounds{0, 100, 200, 200, 200};      // two real experts, then two empty late ones at row T*2
        const int64_t exact = 200;
        CHECK(fixup_read_end(bounds, 100, 128) > exact);
        CHECK(fixup_read_end(bounds, 100, 128) <= (int64_t) mmq_ids_table_rows(200, (uint64_t) T));
    }

    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("mmq_ids_margin_test: ok");
    return 0;
}
