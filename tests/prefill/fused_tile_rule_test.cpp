// pick_ww's rule (include/strata/prefill/fused_tile_rule.hpp) for every card class the fused-tile measurements cover, at the two sides of every threshold, and a
// byte-for-byte comparison with the rule as it stood before the sm_120 40-79 SM class was added, for every other class: no GPU, no model.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/fused_tile_rule.hpp"

using strata::prefill::fused_tile::Card;
using strata::prefill::fused_tile::pick;
using strata::prefill::fused_tile::Pick;
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

    // ---- staged launches (weight staging, both kernels of the launch staged): 128 rows from every size on sm_86 / sm_120 cards with 1 to 79 SMs
    {
        const Card a4000s{false, false, 48, true}, rtx3060{false, false, 28, true}, s86at79{false, false, 79, true}, s86at80{false, false, 80, true}, a6000s{false, false, 84, true},
            s120at79{true, false, 79, false}, s120at80{true, false, 80, false}, s89{false, false, 60, false}, unknown{false, false, 0, true};
        for (int64_t n : {int64_t(0), int64_t(1), int64_t(10000), int64_t(28672), int64_t(40960), int64_t(57344), int64_t(61388), int64_t(81910), int64_t(200000), int64_t(700000)}) {
            for (const Card& c : {a4000s, rtx3060, s86at79, rtx5070, rtx5070ti, s120at79, Card{true, false, 40, false}}) {
                CHECK(pick(c, n, E, true).ww == 2 && pick(c, n, E, true).why == Why::kStaged);
            }
            // the same cards, unstaged: today's rule, byte for byte (the default argument is `false`)
            for (const Card& c : {a4000s, rtx3060, s86at79, rtx5070, rtx5070ti, s120at79}) {
                CHECK(pick(c, n, E, false).ww == pick(c, n, E).ww && pick(c, n, E, false).why == pick(c, n, E).why);
                CHECK(pick(c, n, E).why != Why::kStaged);
            }
            // outside the class a staged launch keeps the rule: 80+ SMs, sm_121, a card with neither flag (sm_89 / sm_80: not measured), an unknown SM count
            for (const Card& c : {s86at80, a6000s, s120at80, rtx5080, pro5000, pro6000, rtx5090, gb10, Card{true, true, 48, true}, s89, unknown}) {
                CHECK(pick(c, n, E, true).ww == pick(c, n, E, false).ww && pick(c, n, E, true).why == pick(c, n, E, false).why);
            }
        }
        // the edges: 79 SMs in the class, 80 out of it
        CHECK(pick(s86at79, 40960, E, true).why == Why::kStaged && pick(s86at80, 40960, E, true).why == Why::kBand);
        CHECK(pick(s120at79, 40960, E, true).why == Why::kStaged && pick(s120at80, 40960, E, true).why == Why::kBand);
        // the cases the 5070 class refuses unstaged (below 119.9 rows an expert) are 128 when staged; the A4000 (sm_86) had no rule of its own below 80 SMs
        CHECK(pick(rtx5070, 40960, E, false).ww == 4 && pick(rtx5070, 40960, E, true).ww == 2);
        CHECK(pick(a4000s, 81910, E, false).ww == 4 && pick(a4000s, 81910, E, true).ww == 2);
        CHECK(pick(a4000s, 40960, E, false).ww == 2 && pick(a4000s, 20000, E, false).ww == 4 && pick(a4000s, 20000, E, true).ww == 2);   // band: 80 rows 128, 39 rows 64
    }
    // ---- the Q6_K / Q8_0 gate/up class (wide) on sm_86 cards with 40+ SMs (the 40-79 SM cards and the 80+ SM ones such as the A6000): 128 rows above 112 rows an expert, everything else as it was
    {
        const Card a4000w{false, false, 48, true}, s86at39{false, false, 39, true}, s86at40{false, false, 40, true}, s86at79{false, false, 79, true}, s86at80{false, false, 80, true},
            s120at48{true, false, 48, false}, s89{false, false, 60, false};
        CHECK(pick(a4000w, 57344, E, false, true).ww == 2 && pick(a4000w, 57344, E, false, true).why == Why::kBand);        // 112.0 rows an expert: the old band's top, 128 as before
        CHECK(pick(a4000w, 57345, E, false, true).ww == 2 && pick(a4000w, 57345, E, false, true).why == Why::kS86Wide);    // 112.002: the line
        CHECK(pick(a4000w, 57345, E, false, false).ww == 4 && pick(a4000w, 57345, E, false, false).why == Why::kBand);      // not the class: 64 as before
        CHECK(pick(a4000w, 81910, E, false, true).ww == 2 && pick(a4000w, 320000, E, false, true).ww == 2);
        CHECK(pick(a4000w, 28672, E, false, true).ww == 4 && pick(a4000w, 28673, E, false, true).ww == 2);                  // below 112: the band, unchanged
        // the same count of rows on Qwen3.6's 256 experts: 4,096 tokens x 8 = 32,768 rows = 128 an expert (the line), 2,048 tokens = 64 an expert (the band)
        CHECK(pick(a4000w, 32768, 256, false, true).ww == 2 && pick(a4000w, 32768, 256, false, true).why == Why::kS86Wide && pick(a4000w, 16384, 256, false, true).why == Why::kBand);
        // the class edges: 39 SMs keeps the old rule, 40 / 79 / 80 take the line; other cards (sm_120, sm_89 / sm_80) never do
        CHECK(pick(s86at39, 81910, E, false, true).ww == 4);
        CHECK(pick(s86at40, 81910, E, false, true).ww == 2 && pick(s86at79, 81910, E, false, true).ww == 2 && pick(s86at80, 81910, E, false, true).ww == 2);
        // the RTX A6000 (sm_86, 84 SMs): both sides of 112 rows an expert, and the i-quant pairs (not wide) keep the rule OWN-4X gated (64 rows above the band)
        const Card a6000w{false, false, 84, true};
        CHECK(pick(a6000w, 57344, E, false, true).ww == 2 && pick(a6000w, 57344, E, false, true).why == Why::kBand);
        CHECK(pick(a6000w, 57345, E, false, true).ww == 2 && pick(a6000w, 57345, E, false, true).why == Why::kS86Wide);
        CHECK(pick(a6000w, 57345, E, false, false).ww == 4 && pick(a6000w, 81910, E, false, false).ww == 4 && pick(a6000w, 81910, E, false, true).ww == 2 && pick(a6000w, 320000, E, false, true).ww == 2);
        CHECK(pick(a6000w, 32768, 256, false, true).why == Why::kS86Wide && pick(a6000w, 16384, 256, false, true).why == Why::kBand && pick(a6000w, 16384, 256, false, true).ww == 2);   // Qwen3.6: 4K tokens 128 an expert, 2K tokens 64
        CHECK(pick(s120at48, 40960, E, false, true).why == pick(s120at48, 40960, E, false, false).why && pick(s89, 81910, E, false, true).ww == 4);
        // wide = false is the rule of the previous commit for every class; wide = true moves only sm_86 with 40+ SMs above 112 rows an expert, and only from 64 to 128
        long long cmpw = 0, badw = 0;
        for (int s120 = 0; s120 < 2; ++s120)
            for (int s121 = 0; s121 < 2; ++s121)
                for (int s86 = 0; s86 < 2; ++s86)
                    for (int sms : {0, 1, 39, 40, 48, 79, 80, 84, 100, 110, 150, 188})
                        for (int staged = 0; staged < 2; ++staged)
                            for (int ne : {1, 256, 512})
                                for (int64_t n = 0; n <= 400000; n += 11) {
                                    const Card c{s120 != 0, s121 != 0, sms, s86 != 0};
                                    const Pick a = pick(c, n, ne, staged != 0, false), b = pick(c, n, ne, staged != 0, true);
                                    ++cmpw;
                                    const double avg = (double) n / (ne > 1 ? ne : 1);
                                    const bool line = !s121 && !(staged && (s86 || s120) && sms > 0 && sms < 80) && !(s120 && sms >= 40 && sms < 80) && !(s120 && sms >= 80 && (avg >= 159.9 || (sms >= 100 && sms < 150 && avg > 112.0))) &&
                                                      s86 && sms >= 40 && avg > 112.0;
                                    if (line ? !(b.ww == 2 && b.why == Why::kS86Wide) : (a.ww != b.ww || a.why != b.why)) ++badw;
                                }
        CHECK(badw == 0);
        std::printf("fused_tile_rule_test: wide class: %lld card / launch combinations checked (%lld wrong)\n", cmpw, badw);
    }
    // ---- the default rule (native_supported) with the 40-79 SM rule on or off (STRATA_PF_FUSED_SM40, on unless 0; see below): takes_by_default(card, sm40, gate_staged, down_ok)
    {
        using strata::prefill::fused_tile::takes_by_default;
        const Card a4000s{false, false, 48, true}, s86at39{false, false, 39, true}, s86at40{false, false, 40, true}, s86at79{false, false, 79, true}, s86at80{false, false, 80, true},
            s120at48{true, false, 48, false}, s120at39{true, false, 39, false}, s89{false, false, 60, false}, gb10c{true, true, 48, false};
        // a staged-gate pair (IQ3_S / IQ4_NL, IQ3_XXS / Q2_0, ...) on 48 SMs: on with the switch, today's behaviour (off) without it
        CHECK(takes_by_default(a4000s, true, true, true) && !takes_by_default(a4000s, false, true, true));
        CHECK(takes_by_default(s120at48, true, true, true) && !takes_by_default(s120at48, false, true, true));
        // IQ2_S / Q2_0 (gate/up not staged) and a down outside IQ4_NL / Q2_0: off, switch or not
        CHECK(!takes_by_default(a4000s, true, false, true) && !takes_by_default(a4000s, true, true, false) && !takes_by_default(a4000s, true, false, false));
        // the bound: 39 SMs off, 40 and 79 on, 80 on without the switch
        CHECK(!takes_by_default(s86at39, true, true, true) && !takes_by_default(s120at39, true, true, true));
        CHECK(takes_by_default(s86at40, true, true, true) && takes_by_default(s86at79, true, true, true));
        CHECK(!takes_by_default(s86at79, false, true, true) && takes_by_default(s86at80, false, false, false));
        // other cards: sm_89 / sm_80 (neither flag) and an unknown SM count stay off with the switch; the s121 flag is not looked at by this rule (GB10 has its own tile rule only)
        CHECK(!takes_by_default(s89, true, true, true) && !takes_by_default(Card{false, false, 0, true}, true, true, true));
        CHECK(takes_by_default(gb10c, true, true, true) == takes_by_default(s120at48, true, true, true));
        // switch unset: exactly today's rule (sm_86 / sm_120 with 80+ SMs) for every card flag set, SM count and launch kind
        long long cmpd = 0, bad = 0;
        for (int s120 = 0; s120 < 2; ++s120)
            for (int s86 = 0; s86 < 2; ++s86)
                for (int sms = 0; sms <= 200; ++sms)
                    for (int g = 0; g < 2; ++g)
                        for (int d = 0; d < 2; ++d) {
                            ++cmpd;
                            const Card c{s120 != 0, false, sms, s86 != 0};
                            if (takes_by_default(c, false, g != 0, d != 0) != ((s120 || s86) && sms >= 80)) ++bad;
                            if (sms >= 80 && takes_by_default(c, true, g != 0, d != 0) != takes_by_default(c, false, g != 0, d != 0)) ++bad;   // the switch moves nothing at 80+ SMs
                        }
        CHECK(bad == 0);
        std::printf("fused_tile_rule_test: default rule with the switch unset: %lld card / launch combinations compared with 'sm_86 / sm_120, 80+ SMs' (%lld differing)\n", cmpd, bad);
    }
    // ---- the 2-bit gate/up formats in the default (owner 11-12): default_gate_format / default_takes(env, card, sm40, gate_staged, down_ok, gu_type)
    {
        using namespace strata::prefill::fused_tile;
        // the ids are ggml's (GGML_TYPE_*, asserted in moe_fused_iq.cu): IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ1_S 19, IQ4_NL 20, IQ3_S 21, IQ2_S 22, IQ4_XS 23
        CHECK(default_gate_format(16) && default_gate_format(17));                                                    // IQ2_XXS, IQ2_XS: new
        CHECK(default_gate_format(18) && default_gate_format(21) && default_gate_format(22) && default_gate_format(23));   // IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS: as before
        CHECK(!default_gate_format(19) && !default_gate_format(20) && !default_gate_format(29) && !default_gate_format(12) && !default_gate_format(0));   // IQ1_S, IQ4_NL, IQ1_M, Q4_K, F32
        const int kIq2[2] = {16, 17};
        long long cmp2 = 0, bad2 = 0;
        for (int gu : kIq2)
            for (int s120 = 0; s120 < 2; ++s120)
                for (int s86 = 0; s86 < 2; ++s86)
                    for (int s121 = 0; s121 < 2; ++s121)
                        for (int sms = 0; sms <= 200; ++sms)
                            for (int sm40 = 0; sm40 < 2; ++sm40)
                                for (int d = 0; d < 2; ++d) {
                                    ++cmp2;
                                    const Card c{s120 != 0, s121 != 0, sms, s86 != 0};
                                    const bool want = (s120 || s86) && sms >= 80;   // sm_86 / sm_120, 80+ SMs; gate_staged is false for IQ2_XXS / IQ2_XS (sw_on), so sm40 adds nothing
                                    if (default_takes(-1, c, sm40 != 0, false, d != 0, gu) != want) ++bad2;
                                    if (default_takes(0, c, sm40 != 0, false, d != 0, gu)) ++bad2;   // STRATA_PF_FUSED=0: the default is off again
                                    if (default_takes(1, c, sm40 != 0, false, d != 0, gu)) ++bad2;   // STRATA_PF_FUSED=1 is the opt-in, decided elsewhere
                                }
        CHECK(bad2 == 0);
        std::printf("fused_tile_rule_test: IQ2_XXS / IQ2_XS default: %lld card / switch combinations (%lld wrong)\n", cmp2, bad2);
        // named cards: 80 SMs yes, 79 no, 48 no, AMD (no s86 / s120 flag) no at any SM count, sm_89 / sm_80 (no flag) no
        CHECK(default_takes(-1, Card{false, false, 84, true}, false, false, true, 16) && default_takes(-1, Card{false, false, 80, true}, false, false, true, 17));   // A6000, 3080 Ti
        CHECK(!default_takes(-1, Card{false, false, 79, true}, false, false, true, 16) && !default_takes(-1, Card{false, false, 79, true}, true, false, true, 17));   // 79 SMs: no, switch or not
        CHECK(!default_takes(-1, Card{false, false, 48, true}, true, false, true, 16) && !default_takes(-1, Card{true, false, 48, false}, true, false, true, 17));   // A4000, 5070
        CHECK(default_takes(-1, Card{true, false, 100, false}, false, false, true, 16));                                                                         // PRO 5000 (sm_120)
        CHECK(!default_takes(-1, Card{false, false, 304, false}, true, false, true, 16) && !default_takes(-1, Card{false, false, 128, false}, false, false, true, 17));   // AMD / sm_89: no
        // the formats already in the list keep their rule: IQ3_XXS on 80+ SMs yes, with the sm40 switch and a staged gate also 40-79
        CHECK(default_takes(-1, Card{false, false, 84, true}, false, true, true, 18) && !default_takes(-1, Card{false, false, 60, true}, false, true, true, 18) &&
              default_takes(-1, Card{false, false, 60, true}, true, true, true, 18));
        // a format outside the list never takes the default
        CHECK(!default_takes(-1, Card{false, false, 84, true}, true, true, true, 19) && !default_takes(-1, Card{false, false, 84, true}, true, true, true, 20));
    }
    // ---- the 40-79 SM cards in the default (owner 11-12): the staged path is the default there, STRATA_PF_FUSED_SM40=0 the way back.  default_takes_env(env, card, sm40_env, gate_staged, down_ok, gu_type)
    {
        using namespace strata::prefill::fused_tile;
        CHECK(sm40_on(-1) && sm40_on(1) && !sm40_on(0));   // unset and "1" on, "0" off
        // the whole grid against an oracle written from the sentence of the decision: STRATA_PF_FUSED unset; a format of the default list (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22, IQ4_XS 23; not
        // Q8_0 8, Q4_K 12, Q5_K 13, Q6_K 14, IQ1_S 19, IQ4_NL 20, IQ1_M 29); a card with the sm_86 or sm_120 flag; then 80+ SMs always, and 40-79 SMs only for a staged gate/up with an IQ4_NL / Q2_0 down while
        // the switch is not "0"
        const int kTypes[] = {8, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29};
        long long cmp3 = 0, bad3 = 0, yes40 = 0, yes80 = 0;
        for (int gu : kTypes)
            for (int env = -1; env <= 1; ++env)
                for (int sm40e = -1; sm40e <= 1; ++sm40e)
                    for (int s120 = 0; s120 < 2; ++s120)
                        for (int s86 = 0; s86 < 2; ++s86)
                            for (int s121 = 0; s121 < 2; ++s121)
                                for (int sms = 0; sms <= 200; ++sms)
                                    for (int g = 0; g < 2; ++g)
                                        for (int d = 0; d < 2; ++d) {
                                            ++cmp3;
                                            const Card c{s120 != 0, s121 != 0, sms, s86 != 0};
                                            const bool inlist = gu == 16 || gu == 17 || gu == 18 || gu == 21 || gu == 22 || gu == 23;
                                            const bool card = s120 || s86;
                                            const bool want = env == -1 && inlist && card && (sms >= 80 || (sms >= 40 && sm40e != 0 && g != 0 && d != 0));
                                            if (default_takes_env(env, c, sm40e, g != 0, d != 0, gu) != want) ++bad3;
                                            if (want && sms < 80) ++yes40;
                                            if (want && sms >= 80) ++yes80;
                                        }
        CHECK(bad3 == 0);
        CHECK(yes40 > 0 && yes80 > 0);
        std::printf("fused_tile_rule_test: 40-79 SM default: %lld env / switch / card / SM count / launch combinations (%lld wrong; %lld yes below 80 SMs, %lld yes at 80+)\n", cmp3, bad3, yes40, yes80);
        // named cards, switch unset: the A4000 (sm_86, 48 SMs) and the RTX 5070 (sm_120, 48 SMs) take a staged gate/up with an IQ4_NL / Q2_0 down; switch "0" takes them back; "1" is the same as unset
        const Card a4000c{false, false, 48, true}, r5070{true, false, 48, false}, a6000c{false, false, 84, true}, tiny{false, false, 39, true}, sm60{false, false, 56, false}, gb10c{true, true, 48, false};
        for (int gu : {18, 21, 23})   // IQ3_XXS, IQ3_S, IQ4_XS
            for (bool dn : {true}) {
                CHECK(default_takes_env(-1, a4000c, -1, true, dn, gu) && default_takes_env(-1, r5070, -1, true, dn, gu) && default_takes_env(-1, a4000c, 1, true, dn, gu));
                CHECK(!default_takes_env(-1, a4000c, 0, true, dn, gu) && !default_takes_env(-1, r5070, 0, true, dn, gu));
                CHECK(default_takes_env(-1, a6000c, 0, true, dn, gu));   // 80+ SMs: the switch moves nothing
                CHECK(!default_takes_env(-1, tiny, -1, true, dn, gu) && !default_takes_env(-1, sm60, -1, true, dn, gu));   // 39 SMs, a card with neither flag (sm_60 / sm_80 / sm_89 / AMD): as before
                CHECK(!default_takes_env(0, a4000c, -1, true, dn, gu) && !default_takes_env(1, a4000c, -1, true, dn, gu));   // STRATA_PF_FUSED=0 is MMQ, =1 is the opt-in decided elsewhere
            }
        CHECK(!default_takes_env(-1, a4000c, -1, false, true, 22));   // IQ2_S gate/up is not a staged instance: stays on MMQ on 48 SMs (IQ2_S / Q2_0 was 0.97 of MMQ at 32K)
        CHECK(!default_takes_env(-1, a4000c, -1, true, false, 21));   // an IQ3_S gate/up with a down outside IQ4_NL / Q2_0 (Q5_0, Q8_0, K-quants): not taken
        CHECK(!default_takes_env(-1, a4000c, -1, false, true, 16) && !default_takes_env(-1, a4000c, -1, false, true, 17));   // IQ2_XXS / IQ2_XS: no staged instance, MMQ below 80 SMs
        CHECK(!default_takes_env(-1, a4000c, -1, true, true, 12) && !default_takes_env(-1, a4000c, -1, true, true, 13) && !default_takes_env(-1, a4000c, -1, true, true, 14));   // Q4_K / Q5_K / Q6_K: opt-in as before
        CHECK(default_takes_env(-1, gb10c, -1, true, true, 21) == default_takes_env(-1, r5070, -1, true, true, 21));   // the s121 flag is not looked at by this rule
    }
    // every class, every launch size: staged = false gives the rule of commit 083b1ac82 exactly as before the argument existed, for the sm_86 cards too (the new flag does not move them)
    {
        long long cmp = 0, diff = 0;
        for (const Card& c : {a6000, a4000, Card{false, false, 48, true}, Card{false, false, 84, true}, Card{false, false, 28, true}, pro6000, pro5000, rtx5090, rtx5080, gb10, hip})
            for (int ne : {1, 64, 256, 512, 1024})
                for (int64_t n = 0; n <= 700000; n += 13) {
                    ++cmp;
                    if (pick(c, n, ne, false).ww != legacy(c, n, ne)) ++diff;
                }
        CHECK(diff == 0);
        std::printf("fused_tile_rule_test: staged = false: %lld launches compared against the previous rule (%lld differing)\n", cmp, diff);
    }

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
    // ---- the weight-staging guard (check_staged_blobs's decision): a failed check makes the launch take the unstaged kernel, it never ends the process
    {
        using namespace strata::prefill::fused_tile;
        // the blob: only device and managed memory at an even address pass; a pointer the driver does not know (kUnknown) is left to the launch's own error, as before
        long long cases = 0;
        for (BlobMem m : {BlobMem::kDevice, BlobMem::kManaged, BlobMem::kHost, BlobMem::kUnregistered, BlobMem::kUnknown})
            for (std::uintptr_t a : {(std::uintptr_t) 0x7f0000001000, (std::uintptr_t) 0x7f0000001002, (std::uintptr_t) 0x7f0000001001, (std::uintptr_t) 0x7f0000001fff, (std::uintptr_t) 0}) {
                ++cases;
                const bool want = m == BlobMem::kUnknown || ((m == BlobMem::kDevice || m == BlobMem::kManaged) && (a & 1) == 0);
                CHECK(staged_blob_ok(m, a) == want);
            }
        CHECK(cases == 25);
        static_assert(staged_blob_ok(BlobMem::kDevice, 0x1000) && !staged_blob_ok(BlobMem::kHost, 0x1000) && !staged_blob_ok(BlobMem::kDevice, 0x1001));
        // the geometry: every combination of odd / even for the four values; ok only when all four are even
        for (int bits = 0; bits < 16; ++bits) {
            const std::size_t v[4] = {(std::size_t) (1000 + (bits & 1)), (std::size_t) (2000 + ((bits >> 1) & 1)), (std::size_t) (3000 + ((bits >> 2) & 1)), (std::size_t) (4000 + ((bits >> 3) & 1))};
            CHECK(staged_geometry_ok(v[0], v[1], v[2], v[3]) == (bits == 0));
        }
        CHECK(staged_geometry_ok(0, 0, 0, 0) && staged_geometry_ok(136 * 1280, 72 * 640, 136 * 640 * 2, 136 * 1280 * 2));
        // the per-process state: the first 256 launches are checked, then trusted; one failure makes every later launch unstaged, and only the first failure is logged
        {
            StagedGuard g;
            int checked = 0, trusted = 0;
            for (int i = 0; i < 1000; ++i) {
                const auto s = g.next();
                CHECK(s != StagedGuard::Step::kUnstaged);
                if (s == StagedGuard::Step::kCheck) { ++checked; CHECK(!g.report(true)); } else ++trusted;
            }
            CHECK(checked == 256 && trusted == 744 && !g.bad);
        }
        {
            StagedGuard g;
            for (int i = 0; i < 2; ++i) CHECK(g.next() == StagedGuard::Step::kCheck && !g.report(true));
            CHECK(g.next() == StagedGuard::Step::kCheck);
            CHECK(g.report(false));                                  // the first failure: logged
            for (int i = 0; i < 2000; ++i) CHECK(g.next() == StagedGuard::Step::kUnstaged);   // sticky, also past the 256 checked launches
            CHECK(!g.report(false) && !g.report(true) && g.bad);     // no second log line, and a passing report does not clear it
            CHECK(g.next() == StagedGuard::Step::kUnstaged);
        }
        {   // a failure on the last checked launch, and one with the budget used up (a failure cannot come then: the launch is trusted, nothing to report)
            StagedGuard g;
            for (int i = 0; i < 255; ++i) CHECK(g.next() == StagedGuard::Step::kCheck);
            CHECK(g.next() == StagedGuard::Step::kCheck && g.report(false));
            CHECK(g.next() == StagedGuard::Step::kUnstaged);
        }
        // the fallback's item is the 64-row one (WW 4), which no staging instance shares, and the rule never answers it with a staged item for a launch it does not stage
        CHECK(kUnstagedWw == 4);
        CHECK(pick(Card{false, false, 48, true}, 81910, E, false, false).ww == 4 && pick(Card{false, false, 48, true}, 81910, E, true, true).ww == 2);   // the staged 128-row choice the fallback overrides
        std::printf("fused_tile_rule_test: staging guard: %lld blob cases, 16 geometries, the check budget and the sticky fallback checked, %d failed so far\n", cases, failed);
    }
    return failed == 0 ? 0 : 1;
}
