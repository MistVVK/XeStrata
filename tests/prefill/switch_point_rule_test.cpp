// switch_point_rule.hpp: the floor from which a prompt chunk takes the streamed walk, from the card's share of non-resident experts.  No GPU, no model.
#include <cmath>
#include <cstdio>

#include "strata/prefill/switch_point_rule.hpp"

namespace sp = strata::prefill::switch_point;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    static_assert(sp::kFloorDefault == 1024, "the default floor is 1,024");
    const sp::Constants k;
    const int64_t total = 24576;   // 48 layers x 512 experts (Flash-Next)

    // every expert resident (or an unknown card): exactly the default
    CHECK(sp::floor_for(total, total) == 1024);
    CHECK(sp::floor_for(total + 5, total) == 1024);
    CHECK(sp::floor_for(0, 0) == 1024);
    CHECK(sp::floor_for(-1, total) == 1024);
    CHECK(sp::floor_for(5000, 0) == 1024);

    // the floor is a multiple of the chunk grid, within [1024, kFloorMax], for every resident count
    int64_t prev = 0;
    bool seen_move = false;
    for (int64_t res = 0; res <= total; res += 64) {
        const int64_t f = sp::floor_for(res, total);
        CHECK(f >= 1024 && f <= sp::kFloorMax);
        CHECK((f - 1024) % sp::kGrid == 0);
        if (f != 1024) seen_move = true;
        (void) prev;
        prev = f;
    }
    CHECK(seen_move);

    // the A6000 / UD-Q4_K_XL card (55% resident): between the measured 2,049 (staged 1.28x faster) and 8,193 (staged 26% slower), about the measured crossing at 4,097
    {
        const int64_t res = (int64_t) (0.55 * (double) total);
        const int64_t f = sp::floor_for(res, total);
        CHECK(f > 2049 && f < 8193);
    }
    // the same card with the measured X (about 12,097 non-resident): the model's own crossing is 4,0xx
    {
        const int64_t f = sp::floor_for(total - 12097, total);
        CHECK(f >= 3584 && f <= 4608);
        CHECK(sp::advantage_ms(1100, 12097, k) > 1000.0);     // measured 1,390 ms (3,236 against 1,838)
        CHECK(sp::advantage_ms(2049, 12097, k) > 500.0);      // measured 710 ms
        CHECK(std::fabs(sp::advantage_ms(4097, 12097, k)) < 120.0);   // measured 18 ms
        CHECK(sp::advantage_ms(8193, 12097, k) < -700.0);     // measured -950 ms
    }
    // a card with fewer non-resident experts stages less often: the floor does not rise as more experts are resident
    {
        int64_t last = sp::kFloorMax + 1;
        for (int64_t res = total / 8; res <= total; res += 256) {
            const int64_t f = sp::floor_for(res, total);
            CHECK(f <= last);
            last = f;
        }
    }
    // the uncovered share: 1 below t0, falling, never above 1
    CHECK(sp::uncovered(100, k) == 1.0);
    CHECK(sp::uncovered(1100, k) > sp::uncovered(4097, k));
    CHECK(sp::uncovered(8193, k) < 0.2);
    CHECK(sp::uncovered(0, k) == 1.0);

    // per-card scaling: the A6000 (3.19 MB experts, 12.2 GB/s, 84 SMs, 13,580 of 24,576 resident) and the A4000 (1.44 MB, 10.5 GB/s, 48 SMs, 7,836 resident), CPU share off:
    // measured crossings about 3,840 and 1,930 tokens, the rule's floors are within one grid step of them
    {
        const sp::Card a6000{3.19, 12.2, 84}, a4000{1.44, 10.5, 48};
        const int64_t f6 = sp::floor_for_card(13580, total, a6000);
        const int64_t f4 = sp::floor_for_card(7836, total, a4000);
        CHECK(f6 >= 3584 && f6 <= 4096);
        CHECK(f4 >= 1792 && f4 <= 2048);
        CHECK(f4 < f6);                                              // the slower card of the two crosses earlier, as measured
        CHECK((f6 - 1024) % sp::kGrid == 0 && (f4 - 1024) % sp::kGrid == 0);
        // an unknown card (no probe, no SM count): the default
        CHECK(sp::floor_for_card(13580, total, sp::Card{0.0, 12.2, 84}) == 1024);
        CHECK(sp::floor_for_card(13580, total, sp::Card{3.19, 0.0, 84}) == 1024);
        CHECK(sp::floor_for_card(13580, total, sp::Card{3.19, 12.2, 0}) == 1024);
        // every expert resident: the default whatever the card
        CHECK(sp::floor_for_card(total, total, a6000) == 1024);
        // the A6000-class card's constants are the fit's own
        const sp::Constants kc = sp::constants_for(sp::Card{});
        CHECK(std::fabs(kc.c_p - 0.2612) < 1e-9 && std::fabs(kc.a_s - 0.2294) < 1e-9);
        // faster link, same card: a smaller floor; more SMs, same link: a higher one
        CHECK(sp::floor_for_card(13580, total, sp::Card{3.19, 24.0, 84}) <= f6);
        CHECK(sp::floor_for_card(13580, total, sp::Card{3.19, 12.2, 120}) >= f6);
    }
    // the share's limit: the floor times the measured ratio on the grid, 0 (the share's own default stays) when the floor did not move
    CHECK(sp::share_limit_for(1024) == 0);
    CHECK(sp::share_limit_for(2048) == 3072);                        // A4000: measured break-even about 2,900
    CHECK(sp::share_limit_for(3840) % sp::kGrid == 0 && sp::share_limit_for(3840) > 3840);

    if (failed) {
        std::fprintf(stderr, "switch_point_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("switch_point_rule_test: OK\n");
    return 0;
}
