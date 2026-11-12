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

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused_tile {

struct Card {
    bool s120 = false;   // compute capability 12.0 (sm120_series.hpp: STRATA_SM120_PERF not 0)
    bool s121 = false;   // GB10 (DGX Spark)
    int sms = 0;
    bool s86 = false;    // the sm_86 rule class, s86_class below: compute capability 8.0, 8.6 or 8.9 (appended last: the aggregate initializers {s120, s121, sms} of the callers and the test stay valid)
};

// The compute capabilities that take the sm_86 rules (the 80+ SM default, the staged default of the 40-79 SM cards, the staged / wide tile lines, the Gyro-S default of fused_tq_default.hpp).  8.6 is the class
// the rules were gated on (RTX A4000 and RTX A6000); 8.0 (A100 class) and 8.9 (RTX 40 / Ada class) joined on the owner's word of 11-12 ("1 Yes" to: may RTX 40 / A100 take the defaults of the cards we
// tested, since they run the same code?).  No card of those two capabilities has run these kernels here: the evidence is the 8.6 and 12.0 rows below, and the same source compiled for sm_80 / sm_89.
// 8.7 (Jetson Orin), 9.0, 10.x, 12.0 / 12.1 and everything below 8.0 are not in the class.  `off` is STRATA_PF_FUSED_SM86=0 (a diagnostic: the class is empty then, whatever the card).
constexpr bool s86_class(int major, int minor, bool off = false) { return !off && major == 8 && (minor == 0 || minor == 6 || minor == 9); }

enum class Why { kBand, kS121, kS120Large, kS120Mid, kS120Small, kStaged, kS86Wide, kS86Kq, kS120Kq };

struct Pick {
    int ww;     // 2 (128 routed rows) or 4 (64)
    Why why;
};

constexpr double kS120LargeAvg = 159.9;   // sm_120, 80+ SMs: a prompt of 8,192 tokens routes 8,191 positions = 81,910 rows = 159.98 an expert
constexpr double kS120MidAvg = 112.0;     // sm_120, 100-149 SMs (PRO 5000): from above the band, continuous with it
constexpr double kS120SmallAvg = 119.9;   // sm_120, 40-79 SMs (RTX 5070): a prompt of 6,144 tokens routes 6,143 positions = 119.98 an expert
constexpr double kKqMinAvg = 5.8;      // sm_86 class, 80-99 SMs, Flash-Next's K-quant pairs: the smallest launch measured is 300 tokens x top 10 over 512 experts = 5.86 an expert

// staged: the launch's gate/up kernel is a weight-staging instance (sw_on in src/prefill/moe_fused_iq.cu).  wide: the launch's pair is in the class of wide_pair below (a Q6_K or Q8_0 gate/up; Qwen3.6's K-quant pairs).
// kq: 0, or the launch's pair is one of Flash-Next's K-quant pairs (kq_flash_kind below: 1 = a Q5_1 down, 2 = a Q8_0 down).  Defaults 0 = the rule as it was.
constexpr Pick pick(const Card& c, int64_t n, int n_expert, bool staged = false, bool wide = false, int kq = 0) {
    const double avg = (double) n / (double) (n_expert > 1 ? n_expert : 1);
    // GB10 (sm_121, 48 SMs): the launch's average is above 112 rows an expert for every prompt size here, so the 56-112 band never fires and the 64-row tile would be
    // chosen everywhere.  Forcing 128 rows measured +3.7% [1.030, 1.079] and +3.4% [1.030, 1.038] at 16K in two independent order-balanced block pairs (12 samples a
    // cell) and +4.3% at 32K in the second pair, with 4K neutral-to-slightly-worse (0.989x, 0.962x, intervals spanning 1.0).  So on this part the 128-row tile is taken from
    // 65,536 routed rows a launch (about an 11K-token prompt at six experts) up; 4K keeps 64.
    if (c.s121) return {n >= 65536 ? 2 : 4, Why::kS121};
    // Staged launches (weight staging: a gate/up kernel of IQ3_XXS / IQ3_S / IQ4_XS / Q8_0 / Q6_K) on an sm_86 or sm_120 card under 80 SMs: 128 rows from every size.  The 128-row staged kernels
    // against the 64-row unstaged ones (RTX A4000, sm_86, 48 SMs, one layer, REAL routing, ms, 2,048 / 8,192 / 20,000 tokens): IQ3_S / IQ4_NL 12.37 / 31.46 / 67.33 against 13.18 / 36.71 / 81.65,
    // IQ3_XXS / IQ4_NL 11.72 / 30.04 / 64.44 against 12.07 / 33.97 / 76.16, IQ4_XS / IQ4_NL 11.97 / 30.60 / 65.33 against 13.10 / 36.17 / 80.54, IQ3_S / Q2_0 (the Q2_0 down is not staged)
    // 11.89 / 30.48 / 65.19 against 12.27 / 34.35 / 76.42 (RESULTS-P4-SM.md S1).  Nothing below 48 SMs is measured, and no sm_120 card under 80 SMs with staging.  sm_80 / sm_89 are in
    // the class since 11-12 (s86_class: the same source, no card of those two capabilities run here); launches without a staged gate/up and every other class keep the rules below.  With Q8_0 / Q6_K staged (RTX A4000, one layer, real routing, MMQ / head 128 at 2K / 8K / 20K / 32K):
    // Q8_0 / Q8_0 1.882 / 1.412 / 1.143 / 1.030 (64-row unstaged 1.326 / 0.955 / 0.783 / 0.649), Q6_K + Q8_0 1.499 / 1.188 / 0.998 / 0.918 (1.059 / 0.787 / 0.638 / 0.547), IQ3_S / Q8_0 1.708 / 1.379 / 1.211 / 1.060 (1.407 / 1.076 / 0.912 / 0.768).
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
    // sm_86-class cards (s86_class: 8.0, 8.6, 8.9) with 40 SMs and up (the 40-79 SM cards and the 80+ SM ones), a Q6_K or Q8_0 gate/up (Q6_K + Q8_0, Q8_0 / Q8_0; CUDA opt-ins), above 112 rows an expert: 128 rows (below 80 SMs the staged rule above already
    // takes 128 at every size once these formats are staged; this line is the one that moves the 80+ SM cards).  Measured on the RTX A4000 (sm_86, 48 SMs, one layer, ms, tile 64 -> tile 128, output bits
    // equal): Qwen3.6's shape (256 experts, top 8; lane P4-Q36, z440-20261111-2126-783b) Q6_K + Q8_0 4,096 / 8,192 / 20,000 tokens 17.65 -> 13.14, 31.91 -> 22.78, 73.46 -> 55.80 (MMQ 13.58 / 20.05 / 38.12);
    // Qwen3.8-Flash-Next's shape (512 experts, top 10, real routing, RESULTS-P4-SM.md S0) at 8,192 / 20,000 / 32,768 tokens: Q6_K + Q8_0 64.97 -> 52.56, 142.17 -> 110.45, 241.14 -> 173.50; Q8_0 / Q8_0
    // 50.64 -> 42.02, 109.88 -> 89.91, 186.36 -> 140.29.  The class is narrow on purpose: the same cards run the 64-row tile faster above 112 rows for the unstaged IQ2_S / Q2_0 pair at 8K (33.80 against 35.05) and
    // for Q2_0 / Q2_0 (26.52 against 28.56 at 8K), so the line does not take every pair.  On the RTX A6000 (sm_86, 84 SMs; lane P4-Q36, Boston jobs boston-20261111-2158-c9cf / -1d66 / -2209-d899) Qwen3.6 UD-Q6_K_XL with every expert in VRAM, end to end,
    // ratio to MMQ at 2K / 8K / 20K: today's tile rule 0.95 / 0.90 / 0.92 (five rounds, spread under 0.02), STRATA_PF_FUSED_TILE=128 0.95 / 0.98 / 0.99 (three rounds).  The i-quant pairs keep the rule lane OWN-4X gated on that class.
    // The K-quant pairs of Qwen3.6's UD-Q4_K_XL (wide_pair below: a Q4_K / Q5_K gate/up with a Q5_K / Q6_K down, both weight-staging instances at 128 rows) take this line too.  One layer of Qwen3.6's shape (256
    // experts, top 8), MMQ ms / fused ms, 2,048 / 4,096 / 8,192 / 20,000 tokens, two repeats, output bits hash equal at every tile, STRATA_PF_FUSED_TILE=64 against 128 (the rule's own pick before this line = 64
    // from 128 rows an expert, 4,096 tokens): RTX A6000 (sm_86, 84 SMs) Q4_K / Q5_K tile 64 0.91 / 0.85 / 0.71 / 0.65, tile 128 1.42 / 1.08 / 0.95 / 0.91; Q4_K / Q6_K 64: 0.82 / 0.73 / 0.61 / 0.53, 128: 1.47 / 1.17 /
    // 1.02 / 0.91; Q5_K / Q6_K 64: 0.71 / 0.63 / 0.52 / 0.46, 128: 1.37 / 1.14 / 0.98 / 0.86 (2,048 tokens = 64 rows an expert is 128 in the band already: equal).  RTX PRO 5000 (sm_120, 110 SMs; its rule above
    // already picks 128 from 56 rows an expert, so its default row equals the forced-128 row): tile 64 0.99 / 0.85 / 0.72 / 0.63, 0.92 / 0.77 / 0.66 / 0.58, 0.83 / 0.70 / 0.60 / 0.52 against 1.28 / 1.07 / 1.02 / 0.87,
    // 1.33 / 1.12 / 0.99 / 0.96, 1.23 / 1.08 / 0.93 / 0.90.  So on both SM classes tested 128 rows is the faster tile from 4,096 tokens up and not worse at 2,048 (par/RESULTS-Q36-DEFAULT.md).
    if (wide && c.s86 && c.sms >= 40 && avg > 112.0) return {2, Why::kS86Wide};
    // Flash-Next's K-quant pairs (UD-Q4_K_XL: Q4_K / Q5_1 in 43 layers, Q4_K / Q8_0 in 4, Q5_K / Q8_0 in 1; kq_flash_kind) on an sm_86-class card with 80 to 99 SMs: 128 rows from the smallest launch measured on.  One layer of
    // Flash-Next's shape (512 experts, top 10; the output bits hash is equal at every tile), RTX A6000 (sm_86, 84 SMs, CUDA 13.1), MMQ ms / fused ms (above 1: fused faster), median of two repetitions (spread 0.01 or less),
    // 300 / 1,000 / 2,048 / 4,096 / 8,192 / 20,000 / 32,768 tokens (5.9 / 19.5 / 40 / 80 / 160 / 391 / 640 rows an expert).  The rule's own pick before this line (64 below 56 rows an expert, 128 from 56 to 112, 64 above):
    // Q4_K / Q5_1 1.28 1.26 1.28 1.29 0.88 0.79 0.73, Q4_K / Q8_0 1.25 1.30 1.32 1.40 0.83 0.72 0.67, Q5_K / Q8_0 1.10 1.15 1.10 1.31 0.67 0.58 0.53.  Tile 128 at every size: Q4_K / Q5_1 1.65 1.49 1.52 1.29 1.04 0.97 0.88,
    // Q4_K / Q8_0 1.83 1.73 1.72 1.40 1.07 0.98 0.91, Q5_K / Q8_0 1.79 1.69 1.69 1.33 1.04 0.98 0.89 (par/RESULTS-KQ-DEFAULT.md).  Only 80-99 SMs: the card measured has 84 (the A40 and RTX 3090 class are the same size); the A100 and
    // RTX 40 class (108 SMs and up) keep the rule above, and nothing smaller than 300 tokens was measured.  The 40-79 SM cards take 128 rows from every size through the staged rule above; sm_120 is not touched (the RTX PRO 5000's
    // pick, 64 below 56 rows an expert and 128 above, is right for the 43-layer Q4_K / Q5_1 pair).
    if (kq != 0 && c.s86 && c.sms >= 80 && c.sms < 100 && avg >= kKqMinAvg) return {2, Why::kS86Kq};
    // The two Q8_0-down pairs (kq == 2) on an sm_120 card with 100 to 149 SMs (RTX PRO 5000; the rule above gives it 128 rows from 56 rows an expert, so this line moves launches of 5.8 to 56 rows an expert from 64 to 128): the same
    // one-layer test on the RTX PRO 5000 (sm_120, 110 SMs, CUDA 13.2), MMQ ms / fused ms, three repetitions, 300 / 1,000 / 2,048 tokens, today's pick against tile 128 (range of the three): Q4_K / Q8_0 1.18 [1.15, 1.19] /
    // 1.43 [1.42, 1.45] / 1.47 [1.46, 1.47] against 1.52 [1.50, 1.54] / 1.51 [1.50, 1.52] / 1.56 [1.55, 1.57]; Q5_K / Q8_0 0.94 [0.94, 0.95] / 1.20 [1.20, 1.21] / 1.35 [1.35, 1.35] against 1.46 [1.45, 1.48] / 1.44 [1.44, 1.46] /
    // 1.49 [1.48, 1.49]; the ranges do not overlap, and the output bits hash is equal at every tile.  The Q5_1-down pair (43 layers) keeps the 64-row tile there: 1.64 / 1.71 / 1.67 against 1.50 / 1.52 / 1.58.
    if (kq == 2 && c.s120 && c.sms >= 100 && c.sms < 150 && avg >= kKqMinAvg) return {2, Why::kS120Kq};
    return {avg > 56.0 && avg <= 112.0 ? 2 : 4, Why::kBand};
}

// The gate/up + down pairs of the wide line (ggml type ids as in moe_fused_iq.cu): a Q6_K or Q8_0 gate/up (Q6_K + Q8_0, Q8_0 / Q8_0), and since 11-12 the K-quant pairs of Qwen3.6's UD-Q4_K_XL, a Q4_K or Q5_K gate/up
// with a Q5_K or Q6_K down (the Qwen3.6 copy of the kernels only: Flash-Next's K-quant gate/up has a Q5_1 / Q8_0 down and keeps its rule).
constexpr int kTypeQ5_1 = 7, kTypeQ8_0 = 8, kTypeQ4K = 12, kTypeQ5K = 13, kTypeQ6K = 14;
constexpr bool wide_pair(int gu_type, int d_type) {
    return gu_type == kTypeQ6K || gu_type == kTypeQ8_0 || ((gu_type == kTypeQ4K || gu_type == kTypeQ5K) && (d_type == kTypeQ5K || d_type == kTypeQ6K));
}
// Flash-Next's K-quant pairs, the last argument class of pick (1 = Q5_1 down, 2 = Q8_0 down, 0 = any other pair): a Q4_K or Q5_K gate/up with a Q5_1 or Q8_0 down (UD-Q4_K_XL: Q4_K / Q5_1, Q4_K / Q8_0, Q5_K / Q8_0).  Disjoint from wide_pair's K-quant pairs (Q5_K / Q6_K downs: Qwen3.6).
constexpr int kq_flash_kind(int gu_type, int d_type) {
    return (gu_type == kTypeQ4K || gu_type == kTypeQ5K) ? (d_type == kTypeQ5_1 ? 1 : d_type == kTypeQ8_0 ? 2 : 0) : 0;
}

// Whether native_supported's default rule (STRATA_PF_FUSED unset; the IQ3_S pack's gate/up IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS with an IQ4_NL / Q2_0 down) takes a launch on this card, as a pure function so the
// rule can be tested without a GPU.  sm_86-class (s86_class: 8.0, 8.6, 8.9) and sm_120 cards with 80+ SMs: always.  sm40 (the 40-79 SM rule; ON by default since the owner's word of 11-12, STRATA_PF_FUSED_SM40=0 turns it off, see sm40_on
// below; it was a research switch, off unless set, before) adds the sm_86 / sm_120 cards with 40-79 SMs for the
// launches whose gate/up kernel is a weight-staging instance (gate_staged: IQ3_XXS / IQ3_S / IQ4_XS, sw_on in src/prefill/moe_fused_iq.cu) with an IQ4_NL or Q2_0 down (down_ok).  Evidence: RTX A4000
// (sm_86, 48 SMs) one layer, real routing, staged 128-row tile against MMQ at 2,048 / 8,192 / 20,000 / 32,768 tokens: IQ3_S / IQ4_NL 1.594 / 1.337 / 1.190 / 1.070, IQ3_XXS / IQ4_NL 1.645 / 1.373 / 1.223 /
// 1.097, IQ4_XS / IQ4_NL 1.621 / 1.326 / 1.176 / 1.055, IQ3_S / Q2_0 1.563 / 1.338 / 1.195 / 1.078 (RESULTS-P4-SM.md S1); the same kernels give the same output bits on every card.  IQ2_S / Q2_0 is 0.95-0.97
// of MMQ at 32K and stays out.  End to end on the RTX A4000 (z440, IQ3_XXS set, the 19 staged-pair layers on the fused kernels, 29 on MMQ, experts streamed over PCIe), prompt time MMQ / fused 1.000 / 1.003 /
// 1.028 / 1.012 at 2K / 4K / 8K / 20K (5 interleaved rounds; 8K and 20K fused faster in 5 of 5), teacher-forced KL against the FP16 prompt path, fused / MMQ 0.930 [0.881, 0.988] at 2,048 and 0.973 [0.898,
// 1.057] at 8,192 ids (RESULTS-P4-SM.md, ITEM 1).  Nothing below 48 SMs is measured (the bound is 40).
constexpr bool takes_by_default(const Card& c, bool sm40, bool gate_staged, bool down_ok) {
    if (!(c.s86 || c.s120)) return false;
    if (c.sms >= 80) return true;
    return sm40 && gate_staged && down_ok && c.sms >= 40;
}

// The gate/up formats native_supported's default rule (CUDA, STRATA_PF_FUSED unset) takes at all: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, as ggml type ids (moe_fused_iq.cu static_asserts
// that they are GGML_TYPE_*).  IQ2_XXS / IQ2_XS joined on the owner's word of 11-12 ("2-bit fused kernels as the default on 80+ SM NVIDIA cards"): the IQ3_XXS set's 19 gate/up layers (9 IQ2_XXS,
// 10 IQ2_XS) on an RTX A6000 (sm_86, 84 SMs): prompt time fused / default 0.874-0.976 from 300 to 8,193 tokens on the integration head, teacher-forced KL against the FP16 prompt path 0.965 [0.912,
// 1.021] at 512, 1.062 [0.995, 1.137] at 1,000, 1.072 [1.020, 1.123] at 2,048 and 1.008 [0.939, 1.101] at 8,192 ids (RESULTS-OWN-4X.md).  The card rule is takes_by_default above and did not
// change: sm_86 / sm_120 with 80+ SMs; below 80 SMs the 40-79 SM rule (sm40) only takes the weight-staging gate/up instances (IQ3_XXS / IQ3_S / IQ4_XS: sw_on), never IQ2_XXS / IQ2_XS.
// AMD (HIP) has its own rule in native_supported and is not touched.
constexpr int kTypeIq2Xxs = 16, kTypeIq2Xs = 17, kTypeIq3Xxs = 18, kTypeIq3S = 21, kTypeIq2S = 22, kTypeIq4Xs = 23;
constexpr bool default_gate_format(int t) {
    return t == kTypeIq2Xxs || t == kTypeIq2Xs || t == kTypeIq2S || t == kTypeIq3Xxs || t == kTypeIq3S || t == kTypeIq4Xs;
}

// native_supported's CUDA default decision for a gate/up format on a card: `env` is STRATA_PF_FUSED as native_supported reads it (-1 unset, 0 = "0", 1 = other: the opt-in, decided
// elsewhere, never "by default"), so only an unset variable can take a layer here, and STRATA_PF_FUSED=0 turns the default off again.
constexpr bool default_takes(int env, const Card& c, bool sm40, bool gate_staged, bool down_ok, int gu_type) {
    return env < 0 && default_gate_format(gu_type) && takes_by_default(c, sm40, gate_staged, down_ok);
}

// STRATA_PF_FUSED_SM40 as moe_fused_iq.cu reads it: -1 unset, 0 = a value starting with '0', 1 = any other value.  The 40-79 SM rule of takes_by_default is ON unless the variable is "0"
// (the way back, a diagnostic like STRATA_PF_FUSED_SM86=0; "1" keeps working and means the same as unset).  default_takes_env is default_takes with that variable folded in, as native_supported calls it.
constexpr bool sm40_on(int sm40_env) { return sm40_env != 0; }
constexpr bool default_takes_env(int env, const Card& c, int sm40_env, bool gate_staged, bool down_ok, int gu_type) {
    return default_takes(env, c, sm40_on(sm40_env), gate_staged, down_ok, gu_type);
}

// ---- the Qwen3.6 engine (strata-q36, src/qwen35/q36_engine.cu; the copy of the kernels in src/prefill/moe_fused_iq_q36.cu): which layers take the fused prompt kernels.
// STRATA_PF_FUSED as pf_env reads it: -1 unset or empty, 1 = a value starting with '1' (the opt-in: every layer whose pair the kernels cover, on any card that covers it), 0 = any other value
// (off: every layer on MMQ).  Before 11-12 the engine took the fused kernels only with the opt-in; unset now takes them on the cards the Flash-Next engine's default rule takes
// (takes_by_default above, reused as it is: sm_86-class and sm_120 cards with 80+ SMs, and the staged launches on 40-79 SMs unless STRATA_PF_FUSED_SM40=0), for the layers whose pair
// the kernels cover.  gate_staged / down_ok are the launch kind of takes_by_default: the gate/up kernel is a weight-staging instance (sw_on(gu_type, true)) and the down is one the staged
// instances were measured with (a staged down or IQ4_NL / Q2_0).  Evidence: RESULTS-E2E-BOSTON.md and RESULTS-P4-Q36.md (UD-Q4_K_XL on an RTX A6000, every expert in VRAM: prompt speed
// fused / MMQ 1.034 at 2K, 0.964 at 4K, 0.966 at 8K, 0.970 at 20K, decode equal; forced-token KL fused / MMQ over all positions 1.047 [0.924, 1.195] at 2K and 0.836 [0.727, 0.969] at 8K).
constexpr int pf_env(const char* v) { return v == nullptr || v[0] == '\0' ? -1 : (v[0] == '1' ? 1 : 0); }
constexpr int sm40_env_of(const char* v) { return v == nullptr ? -1 : (v[0] == '0' ? 0 : 1); }   // STRATA_PF_FUSED_SM40 as moe_fused_iq.cu reads it
constexpr bool q36_takes(int env, const Card& c, int sm40_env, bool gate_staged, bool down_ok, bool covered) {
    if (!covered || env == 0) return false;
    if (env > 0) return true;
    return takes_by_default(c, sm40_on(sm40_env), gate_staged, down_ok);
}

// ---- the weight-staging guard (check_staged_blobs's decision in src/prefill/moe_fused_iq.cu), as pure functions so the decision can be tested without a GPU.
// The staging kernels (sw_on instances) copy the expert blobs into shared memory with cp.async and read 16-bit words at the rows' offsets: they need blobs in device (or managed)
// memory at 2-byte aligned addresses and 2-byte aligned rows and offsets.  A launch that fails the check does NOT end the process: it takes the unstaged 64-row kernel of the same
// formats (same output bits, the identity tables of RESULTS-P4-SM.md), whatever STRATA_PF_FUSED_TILE says.  The memory kinds are cudaMemoryType's, as the kernel's host code reads them.
enum class BlobMem { kDevice, kManaged, kHost, kUnregistered, kUnknown };   // kUnknown: cudaPointerGetAttributes failed (the launch's own error says so)

constexpr bool staged_geometry_ok(std::size_t gu_row, std::size_t d_row, std::size_t up_off, std::size_t down_off) { return ((gu_row | d_row | up_off | down_off) & 1) == 0; }
constexpr bool staged_blob_ok(BlobMem m, std::uintptr_t addr) {
    if (m == BlobMem::kUnknown) return true;   // as before: not a pointer the driver knows, the launch reports it
    return (m == BlobMem::kDevice || m == BlobMem::kManaged) && (addr & 1) == 0;
}

// The per-process state of the check.  The check costs a driver call per blob, so it runs on the first `left` launches only; once one launch has failed it, every later launch of
// the process takes the unstaged kernel without asking again (sticky): after the first launches nothing is checked any more, and a call site that gave one bad blob will give more.
// The log line is printed by the caller that gets `true` from report() (the first failure only).
struct StagedGuard {
    enum class Step { kUnstaged, kCheck, kTrust };
    int left = 256;
    bool bad = false;
    constexpr Step next() {
        if (bad) return Step::kUnstaged;
        if (left <= 0) return Step::kTrust;
        --left;
        return Step::kCheck;
    }
    constexpr bool report(bool ok) {   // true: this failure is the first, log it
        if (ok) return false;
        const bool first = !bad;
        bad = true;
        return first;
    }
};

constexpr int kUnstagedWw = 4;   // the fallback's work item: the 64-row instances, which no weight-staging instance shares

}  // namespace strata::prefill::fused_tile
