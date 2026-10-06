// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
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

#include <cstdlib>
#include <string>

namespace strata::kernels {
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
                if (t == 0) {
                    int cum = above, d = 255;
                    for (; d > 0; --d) {
                        if (cum + hist[d] >= width) break;
                        cum += hist[d];
                    }
                    s_digit_above[0] = d;
                    s_digit_above[1] = cum;
                }
                sycl::group_barrier(grp);
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
        h.parallel_for(sycl::nd_range<1>(size_t(nq) * TK_T, TK_T), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
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
                if (t == 0) {
                    int cum = above, d = 255;
                    for (; d > 0; --d) {
                        if (cum + hist[0][d] >= width) break;
                        cum += hist[0][d];
                    }
                    s_digit_above[0] = d;
                    s_digit_above[1] = cum;
                }
                sycl::group_barrier(grp);
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

bool qsa_block_scores_tc(const float*, const float*, const float*, const int32_t*, int64_t nq, int64_t,
                         const QsaShapes&, float*, void*, int64_t) {
    return nq <= 0;   // refused, as CUDA refuses a device without TF32 mma: the caller runs qsa_block_scores
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s))
        throw core::DeviceError("qsa_block_topk: unsupported geometry or cap");
    auto& q = queue_for(stream);
    finish(stream, launch_topk_ref(q, scores, steps, nq, max_blocks, cap, ids), "qsa_block_topk_ref");
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    // STRATA_TOPK_OLD=1: the kernel with one shared histogram, which a device that takes no TK_T-item work-group runs
    // as well; the same ids
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
    if (old || xe::max_work_group(queue_for(stream)) < (size_t) TK_T) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s))
        throw core::DeviceError("qsa_block_topk: unsupported geometry or cap");
    finish(stream, launch_topk_wide(queue_for(stream), scores, steps, nq, max_blocks, cap, ids), "qsa_block_topk");
}

}  // namespace strata::kernels
