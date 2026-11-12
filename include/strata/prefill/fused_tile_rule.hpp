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
    bool s86 = false;    // compute capability 8.6 (appended last: the aggregate initializers {s120, s121, sms} of the callers and the test stay valid)
};

enum class Why { kBand, kS121, kS120Large, kS120Mid, kS120Small, kStaged, kS86Wide };

struct Pick {
    int ww;     // 2 (128 routed rows) or 4 (64)
    Why why;
};

constexpr double kS120LargeAvg = 159.9;   // sm_120, 80+ SMs: a prompt of 8,192 tokens routes 8,191 positions = 81,910 rows = 159.98 an expert
constexpr double kS120MidAvg = 112.0;     // sm_120, 100-149 SMs (PRO 5000): from above the band, continuous with it
constexpr double kS120SmallAvg = 119.9;   // sm_120, 40-79 SMs (RTX 5070): a prompt of 6,144 tokens routes 6,143 positions = 119.98 an expert

// staged: the launch's gate/up kernel is a weight-staging instance (sw_on in src/prefill/moe_fused_iq.cu).  wide: the launch's gate/up is Q6_K or Q8_0 (the pair class below).  Defaults false = the rule as it was.
constexpr Pick pick(const Card& c, int64_t n, int n_expert, bool staged = false, bool wide = false) {
    const double avg = (double) n / (double) (n_expert > 1 ? n_expert : 1);
    // GB10 (sm_121, 48 SMs): the launch's average is above 112 rows an expert for every prompt size here, so the 56-112 band never fires and the 64-row tile would be
    // chosen everywhere.  Forcing 128 rows measured +3.7% [1.030, 1.079] and +3.4% [1.030, 1.038] at 16K in two independent order-balanced block pairs (12 samples a
    // cell) and +4.3% at 32K in the second pair, with 4K neutral-to-slightly-worse (0.989x, 0.962x, intervals spanning 1.0).  So on this part the 128-row tile is taken from
    // 65,536 routed rows a launch (about an 11K-token prompt at six experts) up; 4K keeps 64.
    if (c.s121) return {n >= 65536 ? 2 : 4, Why::kS121};
    // Staged launches (weight staging: a gate/up kernel of IQ3_XXS / IQ3_S / IQ4_XS / Q8_0 / Q6_K) on an sm_86 or sm_120 card under 80 SMs: 128 rows from every size.  The 128-row staged kernels
    // against the 64-row unstaged ones (RTX A4000, sm_86, 48 SMs, one layer, REAL routing, ms, 2,048 / 8,192 / 20,000 tokens): IQ3_S / IQ4_NL 12.37 / 31.46 / 67.33 against 13.18 / 36.71 / 81.65,
    // IQ3_XXS / IQ4_NL 11.72 / 30.04 / 64.44 against 12.07 / 33.97 / 76.16, IQ4_XS / IQ4_NL 11.97 / 30.60 / 65.33 against 13.10 / 36.17 / 80.54, IQ3_S / Q2_0 (the Q2_0 down is not staged)
    // 11.89 / 30.48 / 65.19 against 12.27 / 34.35 / 76.42 (RESULTS-P4-SM.md S1).  Nothing below 48 SMs is measured, and no sm_120 card under 80 SMs with staging.  sm_89 / sm_80 are not in
    // the class (no DevInfo flag, not measured); launches without a staged gate/up and every other class keep the rules below.  With Q8_0 / Q6_K staged (RTX A4000, one layer, real routing, MMQ / head 128 at 2K / 8K / 20K / 32K):
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
    // sm_86 with 40 SMs and up (the 40-79 SM cards and the 80+ SM ones), a Q6_K or Q8_0 gate/up (Q6_K + Q8_0, Q8_0 / Q8_0; CUDA opt-ins), above 112 rows an expert: 128 rows (below 80 SMs the staged rule above already
    // takes 128 at every size once these formats are staged; this line is the one that moves the 80+ SM cards).  Measured on the RTX A4000 (sm_86, 48 SMs, one layer, ms, tile 64 -> tile 128, output bits
    // equal): Qwen3.6's shape (256 experts, top 8; lane P4-Q36, z440-20261111-2126-783b) Q6_K + Q8_0 4,096 / 8,192 / 20,000 tokens 17.65 -> 13.14, 31.91 -> 22.78, 73.46 -> 55.80 (MMQ 13.58 / 20.05 / 38.12);
    // Qwen3.8-Flash-Next's shape (512 experts, top 10, real routing, RESULTS-P4-SM.md S0) at 8,192 / 20,000 / 32,768 tokens: Q6_K + Q8_0 64.97 -> 52.56, 142.17 -> 110.45, 241.14 -> 173.50; Q8_0 / Q8_0
    // 50.64 -> 42.02, 109.88 -> 89.91, 186.36 -> 140.29.  The class is narrow on purpose: the same cards run the 64-row tile faster above 112 rows for the unstaged IQ2_S / Q2_0 pair at 8K (33.80 against 35.05) and
    // for Q2_0 / Q2_0 (26.52 against 28.56 at 8K), so the line does not take every pair.  On the RTX A6000 (sm_86, 84 SMs; lane P4-Q36, Boston jobs boston-20261111-2158-c9cf / -1d66 / -2209-d899) Qwen3.6 UD-Q6_K_XL with every expert in VRAM, end to end,
    // ratio to MMQ at 2K / 8K / 20K: today's tile rule 0.95 / 0.90 / 0.92 (five rounds, spread under 0.02), STRATA_PF_FUSED_TILE=128 0.95 / 0.98 / 0.99 (three rounds).  The i-quant pairs keep the rule lane OWN-4X gated on that class.
    if (wide && c.s86 && c.sms >= 40 && avg > 112.0) return {2, Why::kS86Wide};
    return {avg > 56.0 && avg <= 112.0 ? 2 : 4, Why::kBand};
}

// Whether native_supported's default rule (STRATA_PF_FUSED unset; the IQ3_S pack's gate/up IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS with an IQ4_NL / Q2_0 down) takes a launch on this card, as a pure function so the
// rule can be tested without a GPU.  sm_86 / sm_120 cards with 80+ SMs: always.  sm40 (the 40-79 SM rule; ON by default since the owner's word of 11-12, STRATA_PF_FUSED_SM40=0 turns it off, see sm40_on
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
