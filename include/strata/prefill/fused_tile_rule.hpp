// include/strata/prefill/fused_tile_rule.hpp - which work item the CUDA fused prompt kernels take for a launch (pick_ww in src/prefill/moe_fused_iq.cu), as a pure
// function of the card class and the launch's routed rows, so the rule can be tested without a GPU.
//
// The item is 128 routed rows (WW 2: 128 weight rows x 128 routed rows) or 64 (WW 4: 256 x 64).  The tile only groups routed rows: the output bits are the same
// either way, so the choice is a speed choice (tests/cuda/prefill_fused_iq_test prints a hash of the fused output's bits: equal at 64, 128 and the default).
// avg = routed rows / experts (T tokens at top 10 over 512 experts: T * 10 / 512).  A prompt of N tokens routes N - 1 positions on the prompt path.
//
// Evidence, REAL routing (the routed expert ids of a UD-IQ4_XS and a UD-Q4_K_XL prompt replayed through the one-layer test; 64-row time / 128-row time, > 1: the
// 128-row tile is faster), at 20 / 40 / 80 / 120 / 160 / 234 / 312 / 391 rows an expert:
//   RTX 5070 (sm_120, 48 SMs)  IQ3_S/IQ4_NL 0.877 0.927 1.020 1.073 1.106 1.154 1.194 1.207;  IQ4_XS/IQ4_NL 0.841 0.884 0.967 1.017 1.067 1.095 1.131 1.148
//   PRO 5000 (110 SMs)         IQ3_S/IQ4_NL 0.929 0.959 1.037 1.097 1.101 1.152 1.181 1.204;  IQ4_XS/IQ4_NL 0.930 0.958 1.023 1.079 1.080 1.126 1.156 1.176
//   PRO 6000 (188 SMs)         IQ3_S/IQ4_NL 0.967 0.947 0.971 1.004 1.068 1.129 1.140 1.172;  IQ4_XS/IQ4_NL 0.930 0.909 0.926 0.960 1.026 1.078 1.091 1.120
// It is one-layer evidence with real routing: the end-to-end rows are the PRO 5000 (+1.9% at 8K, +3.5% at 20K) and the PRO 6000 (lane OWN-4K's kq128 arm); there is no
// end-to-end row on a 5070-class card yet, and the fused kernels are opt-in on that class today (MMQ is its default).
#pragma once

#include <cstdint>

namespace strata::prefill::fused_tile {

struct Card {
    bool s120 = false;   // compute capability 12.0 (sm120_series.hpp: STRATA_SM120_PERF not 0)
    bool s121 = false;   // GB10 (DGX Spark)
    int sms = 0;
    bool s86 = false;    // compute capability 8.6 (appended last: the aggregate initializers {s120, s121, sms} of the callers and the test stay valid)
};

enum class Why { kBand, kS121, kS120Large, kS120Mid, kS120Small, kStaged };

struct Pick {
    int ww;     // 2 (128 routed rows) or 4 (64)
    Why why;
};

constexpr double kS120LargeAvg = 159.9;   // sm_120, 80+ SMs: a prompt of 8,192 tokens routes 8,191 positions = 81,910 rows = 159.98 an expert
constexpr double kS120MidAvg = 112.0;     // sm_120, 100-149 SMs (PRO 5000): from above the band, continuous with it
constexpr double kS120SmallAvg = 119.9;   // sm_120, 40-79 SMs (RTX 5070): a prompt of 6,144 tokens routes 6,143 positions = 119.98 an expert

// staged: both kernels of the launch are weight-staging instances (sw_on in src/prefill/moe_fused_iq.cu).  Default false = the rule as it was.
constexpr Pick pick(const Card& c, int64_t n, int n_expert, bool staged = false) {
    const double avg = (double) n / (double) (n_expert > 1 ? n_expert : 1);
    // GB10 (sm_121, 48 SMs): the launch's average is above 112 rows an expert for every prompt size here, so the 56-112 band never fires and the 64-row tile would be
    // chosen everywhere.  Forcing 128 rows measured +3.7% [1.030, 1.079] and +3.4% [1.030, 1.038] at 16K in two independent order-balanced block pairs (12 samples a
    // cell) and +4.3% at 32K in the second pair, with 4K neutral-to-slightly-worse (0.989x, 0.962x, intervals spanning 1.0).  So on this part the 128-row tile is taken from
    // 65,536 routed rows a launch (about an 11K-token prompt at six experts) up; 4K keeps 64.
    if (c.s121) return {n >= 65536 ? 2 : 4, Why::kS121};
    // Staged launches (weight staging: the gate/up i-quants IQ3_XXS / IQ3_S / IQ4_XS and the IQ4_NL down) on an sm_86 or sm_120 card under 80 SMs: 128 rows from every size.  The
    // staged 128-row kernels (RTX A4000, sm_86, 48 SMs, one layer, REAL routing, ms) against the unstaged 64-row ones at 2,048 / 8,192 / 20,000 tokens, IQ3_S / IQ4_NL
    // 11.64 / 29.59 / 63.58 against 12.77 / 35.10 / 78.09, IQ3_XXS / IQ4_NL 11.38 / 29.20 / 63.11 against 11.88 / 33.47 / 75.40, IQ4_XS / IQ4_NL 11.69 / 30.39 / 65.60 against
    // 13.10 / 36.27 / 80.69 (RESULTS-P4-SM.md S0.2).  Nothing below 48 SMs is measured, and no sm_120 card under 80 SMs with staging.  sm_89 / sm_80 are not in the class
    // (no DevInfo flag, not measured); unstaged formats and every other class keep the rules below.
    if (staged && (c.s86 || c.s120) && c.sms > 0 && c.sms < 80) return {2, Why::kStaged};
    // sm_120 with 80 SMs and up (the cards that take these kernels by default: RTX 5080 / 5090, PRO 5000 / 6000): 128 rows from 159.9 rows an expert on, the first
    // measured point at or above 1.000 on every pair of the PRO 5000 and PRO 6000 rows above.  Below it the answer depends on the card (the PRO 6000 loses 3-7% at 80
    // rows), so the 56-112 band below is left as it was.  A card with 100 to 149 SMs (PRO 5000) also takes 128 rows above the band, so from 56 rows on without a gap.
    if (c.s120 && c.sms >= 80 && (avg >= kS120LargeAvg || (c.sms >= 100 && c.sms < 150 && avg > kS120MidAvg)))
        return {2, (avg >= kS120LargeAvg) ? Why::kS120Large : Why::kS120Mid};
    // sm_120 with 40 to 79 SMs (RTX 5070 measured; the 5070 Ti is inside the class, the 5060 Ti and smaller are not: no measurement below 48 SMs): one threshold, 128 rows
    // from 119.9 rows an expert, the 56-112 band dropped here.  The crossovers measured are about 70 rows (IQ3_S) and 106 (IQ4_XS); at 120 the worst pair is +1.7%.  The band
    // returned 128 where IQ4_XS loses 3.3% (80 rows) and 64 above 112 where 128 wins up to 20%.
    if (c.s120 && c.sms >= 40 && c.sms < 80) return {avg >= kS120SmallAvg ? 2 : 4, Why::kS120Small};
    // everything else (sm_80 / 86 / 89, sm_120 outside the classes above, HIP builds): 128 rows when an expert has about one such tile (56 to 112 rows on average), else 64.
    // Measured on the RTX 5070 with uniform routing (tests/cuda/prefill_fused_iq_test, a layer of 2048 / 3584 / 8192 tokens, ms, 64 -> 128): IQ2_S 5.2 -> 6.7, 9.5 -> 9.3, 17.5 -> 19.5;
    // IQ2_XXS 4.2 -> 5.8, 7.4 -> 7.6, 13.6 -> 16.5; IQ3_S 5.6 -> 6.6, 10.2 -> 8.8, 18.4 -> 19.0; IQ3_XXS 5.1 -> 6.3, 9.1 -> 8.3, 16.5 -> 17.5 (256 rows was slower still).
    return {avg > 56.0 && avg <= 112.0 ? 2 : 4, Why::kBand};
}

}  // namespace strata::prefill::fused_tile
