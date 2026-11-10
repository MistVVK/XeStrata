// pick_ww's rule (include/strata/prefill/fused_tile_rule.hpp) for every card class the fused-tile measurements cover, at the two sides of every threshold, and a
// byte-for-byte comparison with the rule as it stood before the sm_120 40-79 SM class was added, for every other class: no GPU, no model.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/fused_tile_rule.hpp"

using strata::prefill::fused_tile::Card;
using strata::prefill::fused_tile::pick;
using strata::prefill::fused_tile::Why;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// routed rows of 512 experts at a given average (avg * 512, rounded up) and the ones next to a threshold
static constexpr int E = 512;
static constexpr int64_t rows_at(double avg) { return (int64_t) (avg * E + 0.999); }

// the rule of commit 083b1ac82 (the PRO 5000 tier included, the 40-79 SM class not yet there): the reference for "nothing changes"
static int legacy(const Card& c, int64_t n, int n_expert) {
    const double avg = (double) n / (n_expert > 1 ? n_expert : 1);
    if (c.s121) return n >= 65536 ? 2 : 4;
    if (c.s120 && c.sms >= 80 && (avg >= 159.9 || (c.sms >= 100 && c.sms < 150 && avg > 112.0))) return 2;
    return avg > 56.0 && avg <= 112.0 ? 2 : 4;
}

int main() {
    const Card rtx5070{true, false, 48}, rtx5070ti{true, false, 70}, rtx5060ti{true, false, 36}, rtx5080{true, false, 84}, pro5000{true, false, 110},
        rtx5090{true, false, 170}, pro6000{true, false, 188}, a6000{false, false, 84}, a4000{false, false, 48}, gb10{false, true, 48}, hip{false, false, 64};

    // ---- the thresholds of the 40-79 SM class (the RTX 5070): the band is gone, one bound at 119.9 rows an expert
    CHECK(pick(rtx5070, rows_at(56.1), E).ww == 4);           // inside the old band: 64 now
    CHECK(pick(rtx5070, 57344, E).ww == 4);                    // 112.0 rows an expert: the old band's top, 64 now
    CHECK(pick(rtx5070, 61388, E).ww == 4 && pick(rtx5070, 61389, E).ww == 2);    // 119.898 / 119.9004
    CHECK(pick(rtx5070, 61430, E).ww == 2);                    // a prompt of 6,144 tokens: 6,143 positions = 61,430 rows = 119.98
    CHECK(pick(rtx5070, 81910, E).ww == 2 && pick(rtx5070, 200000, E).ww == 2 && pick(rtx5070, 320000, E).ww == 2);
    CHECK(pick(rtx5070, 40960, E).ww == 4);                    // 4,096 tokens: 64
    CHECK(pick(rtx5070, 61430, E).why == Why::kS120Small && pick(rtx5070, 28673, E).why == Why::kS120Small);
    CHECK(pick(rtx5070ti, 61388, E).ww == 4 && pick(rtx5070ti, 61389, E).ww == 2);  // 70 SMs: in the class
    // the class edges: 39 SMs keep the old rule, 40 and 79 take the new one, 80 the large-card one
    CHECK(pick(Card{true, false, 39}, 40960, E).why == Why::kBand && pick(Card{true, false, 39}, 40960, E).ww == 2);   // 80 rows: the old band
    CHECK(pick(Card{true, false, 40}, 40960, E).why == Why::kS120Small && pick(Card{true, false, 40}, 40960, E).ww == 4);
    CHECK(pick(Card{true, false, 79}, 40960, E).why == Why::kS120Small && pick(Card{true, false, 79}, 40960, E).ww == 4);
    CHECK(pick(Card{true, false, 80}, 40960, E).why == Why::kBand && pick(Card{true, false, 80}, 40960, E).ww == 2);
    CHECK(pick(rtx5060ti, 40960, E).why == Why::kBand);        // 36 SMs: not measured, keeps today's rule

    // ---- sm_120 with 80+ SMs: the PRO 6000 / 5090 / 5080 (80-99 and 150+ SMs): the band, and 128 from 159.9 (the PRO 6000 row: 8,191 positions = 81,910 rows)
    for (const Card& c : {pro6000, rtx5090, rtx5080}) {
        CHECK(pick(c, 28672, E).ww == 4 && pick(c, 28673, E).ww == 2);                   // the band's bottom: 56.0 / 56.002
        CHECK(pick(c, 57344, E).ww == 2 && pick(c, 57345, E).ww == 4);                   // the band's top: 112.0 / 112.002
        CHECK(pick(c, 61430, E).ww == 4);                                                // 6,144 tokens: 64 (the PRO 6000 loses at 120 rows)
        CHECK(pick(c, 81868, E).ww == 4 && pick(c, 81869, E).ww == 2);                   // 159.898 / 159.900
        CHECK(pick(c, 81910, E).ww == 2 && pick(c, 200000, E).ww == 2 && pick(c, 320000, E).ww == 2);
        CHECK(pick(c, 81910, E).why == Why::kS120Large);
    }
    // ---- sm_120 with 100-149 SMs (the PRO 5000): 128 rows from the band's bottom on, without a gap
    CHECK(pick(pro5000, 28672, E).ww == 4 && pick(pro5000, 28673, E).ww == 2);
    CHECK(pick(pro5000, 57344, E).ww == 2 && pick(pro5000, 57345, E).ww == 2 && pick(pro5000, 57345, E).why == Why::kS120Mid);
    CHECK(pick(pro5000, 61430, E).ww == 2 && pick(pro5000, 81910, E).ww == 2 && pick(pro5000, 81910, E).why == Why::kS120Large);
    CHECK(pick(Card{true, false, 99}, 61430, E).ww == 4 && pick(Card{true, false, 100}, 61430, E).ww == 2);   // 99 / 100 SMs
    CHECK(pick(Card{true, false, 149}, 61430, E).ww == 2 && pick(Card{true, false, 150}, 61430, E).ww == 4);  // 149 / 150 SMs

    // ---- sm_86 (A6000 84 SMs, A4000 48 SMs: not sm_120, so the SM count does not matter), HIP builds: the old band only
    for (const Card& c : {a6000, a4000, hip}) {
        CHECK(pick(c, 28672, E).ww == 4 && pick(c, 28673, E).ww == 2 && pick(c, 57344, E).ww == 2 && pick(c, 57345, E).ww == 4);
        CHECK(pick(c, 61430, E).ww == 4 && pick(c, 81910, E).ww == 4 && pick(c, 320000, E).ww == 4);
        CHECK(pick(c, 81910, E).why == Why::kBand);
    }
    // ---- sm_121 (GB10): 128 rows from 65,536 routed rows, whatever the SM count
    CHECK(pick(gb10, 65535, E).ww == 4 && pick(gb10, 65536, E).ww == 2 && pick(gb10, 65536, E).why == Why::kS121);
    CHECK(pick(Card{true, true, 48}, 65535, E).ww == 4 && pick(Card{true, true, 48}, 65536, E).ww == 2);       // s121 wins over s120
    // ---- no experts: the average is rows themselves, never a division by zero
    CHECK(pick(rtx5070, 0, 0).ww == 4 && pick(rtx5070, 120, 0).ww == 2);

    // ---- "nothing changes": every class except the 40-79 SM one gives, for every launch size, what the rule of commit 083b1ac82 gave
    long long compared = 0, differing = 0;
    for (const Card& c : {pro6000, rtx5090, rtx5080, pro5000, a6000, a4000, gb10, hip, rtx5060ti, Card{true, false, 39}, Card{true, false, 80}, Card{true, false, 99},
                          Card{true, false, 100}, Card{true, false, 149}, Card{true, false, 150}, Card{false, false, 0}, Card{false, false, 200}})
        for (int ne : {1, 64, 256, 512, 1024})
            for (int64_t n = 0; n <= 700000; n += 13) {
                ++compared;
                if (pick(c, n, ne).ww != legacy(c, n, ne)) ++differing;
            }
    CHECK(differing == 0);
    // and the new class differs from the old rule exactly where intended: below 119.9 rows only in the old band (64 now), above it only above the band (128 now)
    long long new_class = 0, wrong = 0;
    for (const Card& c : {rtx5070, rtx5070ti, Card{true, false, 40}, Card{true, false, 79}})
        for (int64_t n = 0; n <= 700000; n += 7) {
            const double avg = (double) n / E;
            const int got = pick(c, n, E).ww, old = legacy(c, n, E);
            ++new_class;
            const int want = avg >= 119.9 ? 2 : 4;
            if (got != want) ++wrong;
            if (got != old && !((avg > 56.0 && avg <= 112.0) || avg >= 119.9)) ++wrong;   // it differs from the old rule only inside the old band or at 119.9 and up
        }
    CHECK(wrong == 0);
    std::printf("fused_tile_rule_test: %lld launches compared against the previous rule (%lld differing), %lld launches of the 40-79 SM class checked, %d failed\n", compared,
                differing, new_class, failed);
    return failed == 0 ? 0 : 1;
}
