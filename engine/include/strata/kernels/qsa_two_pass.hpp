// include/strata/kernels/qsa_two_pass.hpp - STRATA_SELECT_TWO_PASS: the prompt path's QSA block scores in two
// precisions with the same selection.  Header-only and host/device: the CUDA scorer (qsa_select.cu) and the GPU-free
// test (tests/prefill/qsa_two_pass_test.cpp) read the same constants.
//
// Today (qsa_block_scores_tc, sm_80+) every block score is 3xTF32: per indexer head h and pooled key k, with
// qh = tf32(q), ql = tf32(q - qh), kh = tf32(k), kl = tf32(k - kh) (cvt.rna), the chain of 16 m16n8k8 steps adds
// ql*kh, qh*kl, qh*kh into one FP32 accumulator (48 MMAs), and the score is relu(head 0) + ... + relu(head 3).
// The two-pass scorer computes S1 (qh*kh only, 16 MMAs per head) for every block, bounds |S1 - S3| per block, and
// computes S3 again, bit for bit as today, only for the blocks whose bound reaches the top-k cut-off.  Every other
// block keeps S1, which lies strictly on the same side of the cut-off as its S3, so qsa_block_topk returns exactly
// today's cells in today's order (the threshold key, the count above it and the tied cells are unchanged).
//
// THE BOUND (per query q, block b; derived for the PR, conservative on purpose):
//   S1_h - S3_h = -(qh.kl + ql.kh) + (accumulation error of S1) - (accumulation error of S3)
//   |qh.kl + ql.kh| <= |qh| |kl| + |ql| |kh|                      (Cauchy-Schwarz, exact TF32 operands)
//   accumulation: each MMA's FP32 result is taken to be within kUlps * 2^-23 of the absolute sum of its 9 addends
//     (the accumulator and 8 exact TF32 products).  A truncating alignment with no guard bits gives at most 9 such
//     ulps (8 truncated addends plus the final rounding), so 16 is a margin, not a fit.  Over a chain of n MMAs the
//     error is at most n * kUlps * 2^-23 * A, A = the chain's absolute sum of products:
//       A(S1) <= |qh| |kh|,   A(S3) <= |qh| |kh| + |qh| |kl| + |ql| |kh|
//   relu is 1-Lipschitz; the head sums ((r0 + r1) + r2) + r3 add at most 3 * 2^-24 relative each (folded into kG3).
//   Summed over the four heads, with a = sum_h |qh_h|, c = sum_h |ql_h| (per query) and kh = |kh_b|, kl = |kl_b|:
//     eps_b = kl * P + kh * Q + kAbs,   P = (1 + kG3) a s,   Q = ((1 + kG3) c + (kG1 + kG3) a) s,   s = kSlack
//   kAbs (2^-100) covers flushed subnormals (inputs, products, accumulators); kSlack the second-order terms.
//   All norms are upper bounds (double, rounded up), eps is rounded up and the interval [S1 - eps, S1 + eps] is
//   rounded outwards (CUDA __f*_rd / __f*_ru), so the interval is an enclosure of S3 under the stated MMA model.
//
// THE CUT-OFF RULE (per query): T = the width-th largest S3 cell key, the threshold qsa_block_topk computes.  With
// T1 = the width-th largest S1 and E = the largest eps over the candidate blocks, T lies in [T1 - E, T1 + E]: the
// width cells with S1 >= T1 have S3 >= T1 - E, and every block with S3 > T1 + E has S1 > T1.  A block with
// S1 - eps > T1 + E is above T for certain, one with S1 + eps < T1 - E below it; every other block (including every
// block whose S3 equals T, so the ties) is scored again with today's 3xTF32 arithmetic.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TP_HD __host__ __device__
#else
#define STRATA_TP_HD
#endif

namespace strata::kernels::two_pass {

constexpr int kSteps = 16;                    ///< K = 128 in m16n8k8 steps, one chain per head
constexpr double kUlps = 16.0;                ///< accumulation allowance per MMA, FP32 ulps of its absolute sum
constexpr double kG1 = kSteps * kUlps * 0x1p-23;               ///< the single-pass chain: 16 MMAs
constexpr double kG3 = 3 * kSteps * kUlps * 0x1p-23 + 0x1p-21;  ///< the 3xTF32 chain: 48 MMAs, plus the head sums
constexpr double kSlack = 1.0 + 0x1p-10;
constexpr float kAbs = 0x1p-100f;
constexpr float kMaxEps = 1e30f;              ///< larger bounds (or non-finite norms) mean: score this query again

/// Candidate blocks per query held by the band kernel (shared memory); more, and the query is scored again in full.
constexpr int kCandCap = 5120;
/// Blocks re-scored per query and per column residue (b % 8): eight lists of this many; more is a full re-score.
constexpr int kResCap = 128;
constexpr int kSampleStride = 32;             ///< the candidate threshold comes from every 32nd block, as the
                                              ///< prompt top-k's sample kernel does
/// The calls (a batch of up to 256 queries) that take the two-pass path by default: the batch's last query has at
/// least this many blocks (16,384 cells).  STRATA_SELECT_TWO_PASS_MIN=<blocks> moves it (0: every call).
constexpr int64_t kMinBlocksDefault = 4096;

/// The per-query coefficients of eps_b = kl_b * P + kh_b * Q (+ kAbs), from the head norm sums a and c.
STRATA_TP_HD inline void coefficients(double a, double c, double& P, double& Q) {
    P = (1.0 + kG3) * a * kSlack;
    Q = ((1.0 + kG3) * c + (kG1 + kG3) * a) * kSlack;
}

/// qsa_select.cu's order_key: the float's order as an unsigned key (NaN lowest, -0 as +0).
STRATA_TP_HD inline uint32_t order_key_host(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
/// The float of a key order_key produced (the inverse, for keys of non-NaN values).
STRATA_TP_HD inline float key_float_host(uint32_t k) {
    const uint32_t b = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

/// cvt.rna.tf32.f32 on the host: round to nearest, ties away from zero, 10 explicit mantissa bits (low 13 bits 0).
inline float tf32_rna_host(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    if ((b & 0x7f800000u) == 0x7f800000u) return x;   // Inf and NaN unchanged
    b = (b + 0x1000u) & 0xffffe000u;                   // magnitude rounding; a carry moves into the exponent
    float r;
    std::memcpy(&r, &b, 4);
    return r;
}

/// Scratch the two-pass scorer takes from the caller (bytes, 256-aligned parts): the key norms of every block
/// (one layer, one chunk) and, per call of at most nq_cap queries, the coefficients, modes, residue counts, re-score
/// lists and the 16-query tile flags.
struct Layout {
    uint64_t kn = 0, qc = 0, mode = 0, cnt = 0, ids = 0, tiles = 0, total = 0;
};
inline Layout layout(int64_t nq_cap, int64_t max_blocks) {
    auto up = [](uint64_t v) { return (v + 255u) & ~uint64_t(255u); };
    Layout l;
    uint64_t o = 0;
    l.kn = o;    o += up((uint64_t) max_blocks * 8u);
    l.qc = o;    o += up((uint64_t) nq_cap * 8u);
    l.mode = o;  o += up((uint64_t) nq_cap * 4u);
    l.cnt = o;   o += up((uint64_t) nq_cap * 8u * 4u);
    l.ids = o;   o += up((uint64_t) nq_cap * 8u * (uint64_t) kResCap * 4u);
    l.tiles = o; o += up((uint64_t) ((nq_cap + 15) / 16) * 4u);
    l.total = o;
    return l;
}

/// The engine's counters (qsa_two_pass_stats): what the scorer did, and with STRATA_SELECT_TWO_PASS_CHECK=1 what
/// the comparison against today's scorer found.
enum Stat : int {
    kStCalls = 0,       ///< two-pass calls
    kStQueries,         ///< their queries with a sparse selection (n_kv > width)
    kStIdentity,        ///< their queries whose selection is every cell (nothing to score)
    kStFull,            ///< queries flagged for a full re-score (no bound, a list full, too few cells above F)
    kStFullTiles,       ///< 16-query tiles scored again in full (3xTF32, today's kernel; every query of the tile)
    kStBand,            ///< blocks scored again (3xTF32) next to the cut-off
    kStBandMax,         ///< the most blocks scored again for one query
    kStCand,            ///< candidate blocks (the band kernel's shared list)
    kStCandMax,
    kStBlocks,          ///< blocks scored once (single-pass) in two-pass calls
    kChkQueries,        ///< check: queries compared (ids)
    kChkIdQueries,      ///< check: queries whose ids differ from today's
    kChkIdEntries,      ///< check: id entries that differ
    kChkRescored,       ///< check: scored-again blocks compared bitwise with today's score
    kChkRescoredDiff,   ///< check: ... that differ (must be 0)
    kChkKept,           ///< check: blocks that kept S1 (compared against eps)
    kChkKeptOut,        ///< check: ... with |S1 - S3| > eps (must be 0)
    kChkRatioMax,       ///< check: the largest |S1 - S3| / eps seen, as float bits
    kStatCount
};

}  // namespace strata::kernels::two_pass
