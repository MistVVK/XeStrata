// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/qsa_select.cpp - the Xe port of Strata's src/kernels/cuda/qsa_select.cu (see qsa_select.hpp): the QSA block
// scores for many queries and the per-query top-k over cells (a radix threshold, ties to the lowest index, the cells
// written ascending).
//
// qsa_block_scores_tc is CUDA's 3xTF32 mma.sync scorer.  It refuses devices without TF32 mma, which sends the caller
// to qsa_block_scores; this port refuses the same way until an XMX version is written in the performance stage.
// The top-k's integer prefix sums use SYCL group scans: integer sums are exact in any order, so the ids are the
// CUDA kernels' ids.
#include "strata/kernels/qsa_select.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_qsa_select
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <string>

namespace strata::kernels {

class TopkWideKernel;   // the wide top-k's kernel, asked for its work-group size (topk_wide_fits)

namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;
constexpr int TK_T = 1024;
constexpr int WARP = 32;

inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<uint32_t>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

using local_atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                      sycl::access::address_space::local_space>;

// The radix digit of a pass: the highest d in 255..1 whose bins from 255 down to d hold `width` cells with the `above`
// already counted, else 0, and the count above it.  The work-items 0..255 scan the bins in descending order (bin
// 255 - t) and the one where the count reaches `width` writes the digit (upstream 6ef23b89: one work-item walked the 255
// bins while the others waited); the same digit and count, so the same ids.  `h` is bin 255 - (t & 255), read by every
// work-item (the scan is over the whole work-group; items 256 and up count 0).
template <class G, class A>
inline void pick_digit(const G& grp, int t, int h, int above, int64_t width, A& s_digit_above) {
    const int v = t < 256 ? h : 0;
    const int incl = sycl::inclusive_scan_over_group(grp, v, sycl::plus<int>());
    const int excl = incl - v;
    if (t < 255) {
        if (above + excl < width && above + incl >= width) {
            s_digit_above[0] = 255 - t;
            s_digit_above[1] = above + excl;
        }
    } else if (t == 255 && above + excl < width) {   // none of 255..1 reaches width: digit 0
        s_digit_above[0] = 0;
        s_digit_above[1] = above + excl;
    }
    sycl::group_barrier(grp);
}

// the reference top-k: 256 work-items, each over a contiguous run of blocks, one shared histogram per digit
sycl::event launch_topk_ref(sycl::queue& q, const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                            int64_t cap, int32_t* ids) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<int, 1> s_digit_above(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>(size_t(nq) * TOPK_T, TOPK_T), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int64_t qi = int64_t(it.get_group(0));
            const int32_t* st = steps + qi * kStepCount;
            const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
            int32_t* out = ids + qi * cap;
            const int t = int(it.get_local_id(0));
            if (n_kv <= width) {   // everything is selected: the identity, ascending (uniform)
                for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = int32_t(j);
                return;
            }
            const float* sc = scores + qi * max_blocks;
            const int64_t nb = n_bid + 1;
            const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
            const int64_t b0 = int64_t(t) * per, b1 = (b0 + per < nb) ? b0 + per : nb;
            auto weight = [&](int64_t b) -> int { return b < n_bid ? R : int(n_kv - n_bid * R); };
            uint32_t prefix = 0;
            int above = 0;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
                sycl::group_barrier(grp);
                const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
                for (int64_t b = b0; b < b1; ++b) {
                    const int w = weight(b);
                    if (w == 0) continue;
                    const uint32_t k = order_key(sc[b]);
                    if ((k & hi_mask) == (prefix & hi_mask)) local_atomic(hist[(k >> shift) & 255]).fetch_add(w);
                }
                sycl::group_barrier(grp);
                pick_digit(grp, t, hist[255 - (t & 255)], above, width, s_digit_above);
                prefix |= uint32_t(s_digit_above[0]) << shift;
                above = s_digit_above[1];
                sycl::group_barrier(grp);
            }
            const uint32_t thr = prefix;
            const int64_t eq_budget = width - above;
            int gt = 0, eq = 0;
            for (int64_t b = b0; b < b1; ++b) {
                const int w = weight(b);
                if (w == 0) continue;
                const uint32_t k = order_key(sc[b]);
                if (k > thr) gt += w;
                else if (k == thr) eq += w;
            }
            const int64_t eq_before = sycl::exclusive_scan_over_group(grp, eq, sycl::plus<int>());
            int64_t my_eq = eq_budget - eq_before;
            if (my_eq < 0) my_eq = 0;
            if (my_eq > eq) my_eq = eq;
            const int sel = gt + int(my_eq);
            int64_t wpos = sycl::exclusive_scan_over_group(grp, sel, sycl::plus<int>());
            int64_t eq_left = my_eq;
            for (int64_t b = b0; b < b1; ++b) {
                const int w = weight(b);
                if (w == 0) continue;
                const uint32_t k = order_key(sc[b]);
                if (k > thr) {
                    for (int c = 0; c < w; ++c) out[wpos++] = int32_t(b * R + c);
                } else if (k == thr) {
                    for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = int32_t(b * R + c);
                }
            }
        });
    });
}

// The top-k on TK_T work-items (upstream cce52db), per-sub-group histograms and sub-group scans, with each key read
// from memory again on every pass.  A histogram has no order, so on the four radix passes work-item t reads blocks t,
// t + TK_T, ...: a sub-group reads WARP neighbours at a time.  It replaced the kernel that kept each work-item's run of
// contiguous blocks in registers, which upstream 1bb5c72 left for Turing above 90K cells: these coalesced reads were
// faster at every context and batch measured, 4K-135K cells and 1-2,048 queries (B70 1.1-2.3x, RTX 4070 1.05-1.25x).  The cells are emitted ascending in the order (sub-group, row, lane):
// sub-group w holds the `per` rows of WARP consecutive blocks from block w * WARP * per.  launch_topk_ref's selection
// rule (radix threshold, ties to the lowest index): identical ids.
sycl::event launch_topk_wide(sycl::queue& q, const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                             int64_t cap, int32_t* ids) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 2> hist(sycl::range<2>(TK_T / WARP, 256), h);
        sycl::local_accessor<int, 1> s_gt(sycl::range<1>(TK_T / WARP), h);
        sycl::local_accessor<int, 1> s_eq(sycl::range<1>(TK_T / WARP), h);
        sycl::local_accessor<int, 1> s_digit_above(sycl::range<1>(2), h);
        h.parallel_for<TopkWideKernel>(sycl::nd_range<1>(size_t(nq) * TK_T, TK_T), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t qi = int64_t(it.get_group(0));
            const int32_t* st = steps + qi * kStepCount;
            const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
            int32_t* out = ids + qi * cap;
            const int t = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), warp = int(sg.get_group_linear_id());
            if (n_kv <= width) {
                for (int64_t j = t; j < n_kv; j += TK_T) out[j] = int32_t(j);
                return;
            }
            const float* sc = scores + qi * max_blocks;
            const int64_t nb = n_bid + 1;
            const int64_t per = (nb + TK_T - 1) / TK_T;   // rows of WARP blocks per sub-group
            auto weight = [&](int64_t b) -> int { return b < n_bid ? R : int(n_kv - n_bid * R); };
            uint32_t prefix = 0;
            int above = 0;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int i = lane; i < 256; i += WARP) hist[warp][i] = 0;
                sycl::group_barrier(sg);
                const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
                for (int64_t b = t; b < nb; b += TK_T) {
                    const int w = weight(b);
                    if (w == 0) continue;
                    const uint32_t k = order_key(sc[b]);
                    if ((k & hi_mask) == (prefix & hi_mask)) local_atomic(hist[warp][(k >> shift) & 255]).fetch_add(w);
                }
                sycl::group_barrier(grp);
                if (t < 256) {   // fold the sub-groups' histograms into sub-group 0's
                    int sum = 0;
                    for (int w2 = 0; w2 < TK_T / WARP; ++w2) sum += hist[w2][t];
                    hist[0][t] = sum;
                }
                sycl::group_barrier(grp);
                pick_digit(grp, t, hist[0][255 - (t & 255)], above, width, s_digit_above);
                prefix |= uint32_t(s_digit_above[0]) << shift;
                above = s_digit_above[1];
                sycl::group_barrier(grp);
            }
            const uint32_t thr = prefix;
            const int eq_budget = int(width - above);   // cells equal to thr that fit, lowest index first
            const int64_t w0 = int64_t(warp) * WARP * per;
            int gt = 0, eq = 0;
            for (int64_t b = w0 + lane; b < nb && b < w0 + WARP * per; b += WARP) {
                const int w = weight(b);
                if (w == 0) continue;
                const uint32_t k = order_key(sc[b]);
                if (k > thr) gt += w;
                else if (k == thr) eq += w;
            }
            gt = sycl::reduce_over_group(sg, gt, sycl::plus<int>());
            eq = sycl::reduce_over_group(sg, eq, sycl::plus<int>());
            if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
            sycl::group_barrier(grp);
            if (t == 0) {   // per sub-group: the selected cells and the tied cells before it
                int eb = 0, sb = 0;
                for (int w2 = 0; w2 < TK_T / WARP; ++w2) {
                    const int g2 = s_gt[w2], e2 = s_eq[w2];
                    const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
                    s_gt[w2] = sb;
                    s_eq[w2] = eb;
                    sb += g2 + take;
                    eb += e2;
                }
            }
            sycl::group_barrier(grp);
            int run_sel = s_gt[warp], run_eq = s_eq[warp];
            for (int64_t r0 = w0; r0 < nb && r0 < w0 + WARP * per; r0 += WARP) {
                const int64_t b = r0 + lane;
                const int w = b < nb ? weight(b) : 0;
                const uint32_t k = w ? order_key(sc[b]) : 0u;
                const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
                if (!sycl::any_of_group(sg, (my_gt | my_eq) != 0)) continue;   // a row without a selected block
                const int pe = sycl::inclusive_scan_over_group(sg, my_eq, sycl::plus<int>());   // tied cells up to this lane
                const int left = eq_budget - (run_eq + pe - my_eq);
                const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
                const int my_sel = my_gt + take;
                const int ps = sycl::inclusive_scan_over_group(sg, my_sel, sycl::plus<int>());
                int32_t* dst = out + run_sel + ps - my_sel;
                for (int c = 0; c < my_sel; ++c) dst[c] = int32_t(b * R + c);
                run_eq += sycl::group_broadcast(sg, pe, WARP - 1);
                run_sel += sycl::group_broadcast(sg, ps, WARP - 1);
            }
        });
    });
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535)
        throw core::DeviceError("qsa_block_scores: unsupported indexer geometry");
    // a block past a query's n_bid returns at once: the launch need only reach the batch's largest n_bid
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const size_t gx = size_t((reach + SCORE_WARPS - 1) / SCORE_WARPS);
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<2>({size_t(nq), gx * SCORE_WARPS * WARP}, {1, SCORE_WARPS * WARP}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t qi = int64_t(it.get_group(0));
            const int32_t* st = steps + qi * kStepCount;
            const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
            const int64_t b = int64_t(it.get_group(1)) * SCORE_WARPS + int64_t(sg.get_group_linear_id());
            if (b > n_bid || b >= max_blocks) return;   // uniform over the sub-group
            const int lane = int(sg.get_local_linear_id());
            const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
            const float* kp = key + lane * 4;
            const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* q4 = q + h * IDX_DIM;
                float d = kp[0] * q4[0] + kp[1] * q4[1] + kp[2] * q4[2] + kp[3] * q4[3];
                for (int o = 16; o > 0; o >>= 1) d += sycl::permute_group_by_xor(sg, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && n_kv % R != 0) score += 1e9f;
                scores[qi * max_blocks + b] = score;
            }
        });
    finish(stream, e, "qsa_block_scores");
}

void qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                           int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R)
        throw core::DeviceError("qsa_block_scores_rows: unsupported indexer geometry");
    static_assert(IDX_DIM == WARP * 4 && IDX_HEADS == 4, "the scatter below is for 4 heads of 4 values a lane");
    const size_t gx = size_t((max_blocks + SCORE_WARPS - 1) / SCORE_WARPS);
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(gx * (size_t) (SCORE_WARPS * WARP), (size_t) (SCORE_WARPS * WARP)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t b = int64_t(it.get_group(0)) * SCORE_WARPS + int64_t(sg.get_group_linear_id());
            const int64_t top = steps[(nq - 1) * kStepCount + kStepNBid];   // the last row's: the largest
            if (b > top || b >= max_blocks) return;   // uniform over the sub-group
            const int lane = int(sg.get_local_linear_id());
            const bool hi16 = (lane & 16) != 0, hi8 = (lane & 8) != 0;
            const float* kpool = pooled + b * IDX_DIM + (ptrdiff_t) lane * 4;
            float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
            if (b < top) { k0 = kpool[0]; k1 = kpool[1]; k2 = kpool[2]; k3 = kpool[3]; }   // a completed block's key
            for (int64_t qi = 0; qi < nq; ++qi) {
                const int32_t* st = steps + qi * kStepCount;
                const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
                if (b > n_bid) continue;   // uniform
                float kp0 = k0, kp1 = k1, kp2 = k2, kp3 = k3;
                if (b == n_bid) {   // this row's tail block: the dead key
                    const float* kd = dead + (ptrdiff_t) lane * 4;
                    kp0 = kd[0]; kp1 = kd[1]; kp2 = kd[2]; kp3 = kd[3];
                }
                const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + (ptrdiff_t) lane * 4;
                float d[IDX_HEADS];
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float* q4 = q + (ptrdiff_t) h * IDX_DIM;
                    d[h] = kp0 * q4[0] + kp1 * q4[1] + kp2 * q4[2] + kp3 * q4[3];   // qsa_block_scores' expression
                }
                // qsa_block_scores sums each head over the lanes by xor 16, 8, 4, 2, 1.  The same pairs here, with the
                // heads spread: at 16 the low half keeps heads 0 and 1 and the high half 2 and 3, at 8 one head a
                // lane, then 4, 2, 1 - every partial sum is the same two operands (a + b == b + a), so the same bits
                const float s0 = sycl::permute_group_by_xor(sg, hi16 ? d[0] : d[2], 16);
                const float s1 = sycl::permute_group_by_xor(sg, hi16 ? d[1] : d[3], 16);
                const float a0 = (hi16 ? d[2] : d[0]) + s0, a1 = (hi16 ? d[3] : d[1]) + s1;
                const float s2 = sycl::permute_group_by_xor(sg, hi8 ? a0 : a1, 8);
                float c = (hi8 ? a1 : a0) + s2;   // head (hi16 ? 2 : 0) + (hi8 ? 1 : 0)
                for (int o = 4; o > 0; o >>= 1) c += sycl::permute_group_by_xor(sg, c, o);
                const float e1 = sycl::select_from_group(sg, c, 8), e2 = sycl::select_from_group(sg, c, 16),
                            e3 = sycl::select_from_group(sg, c, 24);
                if (lane == 0) {
                    float score = 0.0f;
                    score += c > 0.0f ? c : 0.0f;
                    score += e1 > 0.0f ? e1 : 0.0f;
                    score += e2 > 0.0f ? e2 : 0.0f;
                    score += e3 > 0.0f ? e3 : 0.0f;
                    if (b == n_bid && n_kv % R != 0) score += 1e9f;
                    scores[qi * max_blocks + b] = score;
                }
            }
        });
    finish(stream, e, "qsa_block_scores_rows");
}

namespace {
// upstream 65ce3299 (STRATA_SELECT_SIMT=1): the block scores below each query's n_bid as an FP32 tiled GEMM - rows
// (query, indexer head) x columns (blocks), K = 128 through local memory in steps of 32, 2 queries x 4 heads x 4 blocks
// a work-item, relu per head summed in registers.  The warp kernel spends 5 sub-group shuffles on 4 products.  FP32 in
// another summation order, so not the warp kernel's bits (upstream keeps it opt-in; here the default, faster on both
// cards measured).
constexpr int SM_QT = 32, SM_NB = 64, SM_KC = 32, SM_QS = SM_QT * IDX_HEADS + 4, SM_KS = SM_NB + 4;
sycl::event block_scores_simt(sycl::queue& q, const float* pooled, const float* q_idx, const int32_t* steps, int64_t nq,
                              int64_t max_blocks, int64_t reach, float* out) {
    const size_t gb = (size_t) ((reach + SM_NB - 1) / SM_NB), gq = (size_t) ((nq + SM_QT - 1) / SM_QT);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) SM_KC * SM_QS), h);
        sycl::local_accessor<float, 1> ks(sycl::range<1>((size_t) SM_KC * SM_KS), h);
        h.parallel_for(sycl::nd_range<2>({gq, gb * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
            const int64_t q0 = (int64_t) it.get_group(0) * SM_QT, b0 = (int64_t) it.get_group(1) * SM_NB;
            const int64_t qlast = (q0 + SM_QT < nq ? q0 + SM_QT : nq) - 1;
            if (b0 >= steps[qlast * kStepCount + kStepNBid]) return;   // n_bid rises with the position: the whole group
            const int tid = (int) it.get_local_id(1), tx = tid & 15, ty = tid >> 4;
            float acc[8][4] = {};
            for (int kc = 0; kc < IDX_DIM; kc += SM_KC) {
                for (int i = 0; i < 4; ++i) {   // the query rows: 128 x 8 float4
                    const int idx = tid + i * 256, r = idx >> 3, k4 = idx & 7;
                    sycl::float4 v(0.0f);
                    if (q0 + (r >> 2) < nq)
                        v = *reinterpret_cast<const sycl::float4*>(q_idx + (q0 * IDX_HEADS + r) * IDX_DIM + (int64_t) (kc + k4 * 4));
                    qs[(k4 * 4 + 0) * SM_QS + r] = v.x(); qs[(k4 * 4 + 1) * SM_QS + r] = v.y();
                    qs[(k4 * 4 + 2) * SM_QS + r] = v.z(); qs[(k4 * 4 + 3) * SM_QS + r] = v.w();
                }
                for (int i = 0; i < 2; ++i) {   // the pooled keys: 64 x 8 float4
                    const int idx = tid + i * 256, c = idx >> 3, k4 = idx & 7;
                    sycl::float4 v(0.0f);
                    if (b0 + c < reach) v = *reinterpret_cast<const sycl::float4*>(pooled + (b0 + c) * IDX_DIM + (int64_t) (kc + k4 * 4));
                    ks[(k4 * 4 + 0) * SM_KS + c] = v.x(); ks[(k4 * 4 + 1) * SM_KS + c] = v.y();
                    ks[(k4 * 4 + 2) * SM_KS + c] = v.z(); ks[(k4 * 4 + 3) * SM_KS + c] = v.w();
                }
                sycl::group_barrier(it.get_group());
#pragma unroll 8
                for (int k = 0; k < SM_KC; ++k) {
                    float qr[8], kr[4];
                    for (int i = 0; i < 8; ++i) qr[i] = qs[k * SM_QS + ty * 8 + i];
                    for (int j = 0; j < 4; ++j) kr[j] = ks[k * SM_KS + tx * 4 + j];
                    for (int i = 0; i < 8; ++i)
                        for (int j = 0; j < 4; ++j) acc[i][j] = sycl::fma(qr[i], kr[j], acc[i][j]);
                }
                sycl::group_barrier(it.get_group());
            }
            for (int i = 0; i < 2; ++i) {
                const int64_t qq = q0 + (int64_t) (ty * 2 + i);
                if (qq >= nq) continue;
                const int64_t n_bid = steps[qq * kStepCount + kStepNBid];
                for (int j = 0; j < 4; ++j) {
                    const int64_t b = b0 + (int64_t) (tx * 4 + j);
                    if (b >= n_bid || b >= max_blocks) continue;
                    float sc = 0.0f;
                    for (int hh = 0; hh < IDX_HEADS; ++hh) {
                        const float d = acc[i * 4 + hh][j];
                        sc += d > 0.0f ? d : 0.0f;
                    }
                    out[qq * max_blocks + b] = sc;
                }
            }
        });
    });
}
// the tail block (b == n_bid, the dead key) of every query, as the warp kernel computes it
sycl::event block_scores_tail(sycl::queue& q, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                              int64_t max_blocks, float* scores) {
    return q.parallel_for(sycl::nd_range<1>((size_t) nq * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t qi = int64_t(it.get_group(0));
        const int32_t* st = steps + qi * kStepCount;
        const int64_t n_kv = st[kStepNKv], b = st[kStepNBid];
        if (b >= max_blocks) return;
        const int lane = int(sg.get_local_linear_id());
        const float* kp = dead + (ptrdiff_t) lane * 4;
        const float* qv = q_idx + qi * IDX_HEADS * IDX_DIM + (ptrdiff_t) lane * 4;
        float score = 0.0f;
        for (int hh = 0; hh < IDX_HEADS; ++hh) {
            const float* q4 = qv + (ptrdiff_t) hh * IDX_DIM;
            float d = kp[0] * q4[0] + kp[1] * q4[1] + kp[2] * q4[2] + kp[3] * q4[3];
            for (int o = 16; o > 0; o >>= 1) d += sycl::permute_group_by_xor(sg, d, o);
            score += d > 0.0f ? d : 0.0f;
        }
        if (lane == 0) {
            if (n_kv % R != 0) score += 1e9f;
            scores[qi * max_blocks + b] = score;
        }
    });
}
}  // namespace

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    // no TF32 matrix scorer here (CUDA's mma.sync): the FP32 tiled one (a 38,738-token prompt on the B70 1,451 -> 1,536
    // tok/s, the RTX 4070 1,013 -> 1,057); STRATA_SELECT_SIMT=0: refused, and the caller runs qsa_block_scores
    static const bool simt = [] {
        const char* v = strata::env_get("SELECT_SIMT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (!simt || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R) return false;
    auto& q = queue_for(stream);
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    block_scores_simt(q, pooled, q_idx, steps, nq, max_blocks, reach, scores);
    finish(stream, block_scores_tail(q, dead, q_idx, steps, nq, max_blocks, scores), "qsa_block_scores (simt)");
    return true;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s))
        throw core::DeviceError("qsa_block_topk: unsupported geometry or cap");
    auto& q = queue_for(stream);
    finish(stream, launch_topk_ref(q, scores, steps, nq, max_blocks, cap, ids), "qsa_block_topk_ref");
}

// Whether the wide top-k's own kernel takes TK_T work-items on q's device.  The device's largest work-group is not
// enough (#35): on CUDA the kernel's registers bound it too - sm_90's code takes 67 a work-item, 68,608 for 1,024, past
// the 65,536 of a block, and a GPU newer than the package's newest code (an RTX 5090 running sm_90's PTX) gets what its
// driver compiles - so its launch failed with "Exceeded the number of registers available on the hardware".  Asked
// of the kernel itself (kernel_device_specific::work_group_size), once a device.
static bool topk_wide_fits(sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, bool> fits;
    const sycl::device dev = q.get_device();
    std::lock_guard<std::mutex> lock(mu);
    if (auto it = fits.find(dev); it != fits.end()) return it->second;
    bool ok = xe::max_work_group(q) >= (size_t) TK_T;
    if (ok) {
        try {
            const auto id = sycl::get_kernel_id<TopkWideKernel>();
            const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {dev}, {id});
            ok = bundle.get_kernel(id).get_info<sycl::info::kernel_device_specific::work_group_size>(dev) >=
                 (size_t) TK_T;
        } catch (const sycl::exception&) {
            ok = false;                                // not built for this device: the other kernel
        }
    }
    return fits.emplace(dev, ok).first->second;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    // STRATA_TOPK_OLD=1: the kernel with one shared histogram, which a device whose wide kernel takes no TK_T-item
    // work-group (topk_wide_fits) runs as well; the same ids
    static const bool old = strata::env_get("TOPK_OLD") != nullptr;
    if (nq <= 0) return;
    if (old || !topk_wide_fits(queue_for(stream))) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s))
        throw core::DeviceError("qsa_block_topk: unsupported geometry or cap");
    finish(stream, launch_topk_wide(queue_for(stream), scores, steps, nq, max_blocks, cap, ids), "qsa_block_topk");
}

}  // namespace strata::kernels
