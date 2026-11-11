// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/router.cpp - the Xe ports of Strata's src/kernels/cuda/router_top10.cu (the reference-faithful router) and
// Strata's src/kernels/cuda/native_router.cu (the pinned ggml topk-moe contract for 512 experts, top 10; here also for
// 256, a pruned model's, 8 values a lane instead of 16).
//
// native_router follows ggml-cuda's topk-moe at llama.cpp 3cf03257; MIT License, Copyright (c) 2023-2026 The ggml
// authors, see third_party/main/ggml/LICENSE.  CUDA compiled it with --use_fast_math; here exp is the precise function.
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_router
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::kernels {

template <int NPL> class RouterPublishKernel;   // asked for its work-group (work_group_upto_1024_of)

namespace {

constexpr int WARP = 32;

// The argmax order of both routers: the larger probability wins, a tie keeps the lower expert index.  It is a total
// order, so the xor butterflies below reach the same winner as CUDA's shuffle-down trees.
inline void better(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

sycl::queue& queue_for(void* stream, bool required) {
    if (required && !stream) throw std::invalid_argument("the native router requires an explicit stream");
    return core::Runtime::get().stream(stream);
}

// One token's top k (k <= 32) by one sub-group (the pinned kernel's 32 x 8 block has only its first warp active), NPL
// values a lane, the experts past n_expert (<= NPL x 32) never picked: `ids2` and `weights2`, when given, get the same
// values (the doorbell's mapped copies, written here instead of copied after)
template <int NPL>
inline void top10_token(const sycl::sub_group& sg, const float* logits, int32_t* ids, float* weights, int32_t* ids2,
                        float* weights2, int n_expert = NPL * 32, int k = 10) {
        const int lane = (int) sg.get_local_linear_id();
        const bool full = n_expert == NPL * 32;
        float values[NPL];
        for (int i = 0; i < NPL; ++i) values[i] = full || lane + i * 32 < n_expert ? logits[lane + i * 32] : -INFINITY;
        float maximum = -INFINITY;
        for (int i = 0; i < NPL; ++i) maximum = sycl::max(maximum, values[i]);
        for (int mask = 16; mask; mask >>= 1) maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(sg, maximum, mask));
        float sum = 0.0f;
        for (int i = 0; i < NPL; ++i) {
            values[i] = sycl::exp(values[i] - maximum);
            sum += values[i];
        }
        for (int mask = 16; mask; mask >>= 1) sum += sycl::permute_group_by_xor(sg, sum, mask);
        const float reciprocal = 1.0f / sum;
        for (int i = 0; i < NPL; ++i) {
            values[i] *= reciprocal;
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
            if (!full && lane + i * 32 >= n_expert) values[i] = -INFINITY;
        }
        // The rows already picked are a bit mask, not -INFINITY stored back into `values`: IGC 2.38.2 drops every store
        // of this kernel when it stores at a run-time index into the private array the butterfly just read (#45, Arc
        // Pro B70: the routing table kept its old contents).  The picks, their order and the ties are the same.
        uint32_t picked = 0;
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < k; ++rank) {
            float best = -INFINITY;
            int expert = lane;
            for (int i = 0; i < NPL; ++i) {
                if (((picked >> i) & 1u) == 0 && values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
            for (int mask = 16; mask; mask >>= 1)
                better(best, expert, sycl::permute_group_by_xor(sg, best, mask), sycl::permute_group_by_xor(sg, expert, mask));
            if ((expert & 31) == lane) {
                picked |= 1u << (expert / 32);
                ids[rank] = expert;
                if (ids2 != nullptr) ids2[rank] = expert;
                // Deliberately accumulate by WINNING EXPERT lane, not output rank.
                // Multiple selected experts in one lane add in selection order.
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        for (int mask = 16; mask; mask >>= 1) selected_sum += sycl::permute_group_by_xor(sg, selected_sum, mask);
        selected_sum = sycl::max(selected_sum, 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < k) {
            weights[lane] = selected * inverse_selected_sum;
            if (weights2 != nullptr) weights2[lane] = selected * inverse_selected_sum;
        }
}

// a token a sub-group, 8 tokens a work-group from 8 tokens on (the prompt's rows; each token's arithmetic as alone): a
// work-group of one sub-group a token ran 2,048 tokens of 512 experts 3% slower on the RTX 4070 (0.0132 ms against
// the prompt's former kernel's 0.0128)
template <int NPL>
sycl::event launch_native_router(sycl::queue& q, const float* logits_all, int32_t* ids_all, float* weights_all, int n_tok,
                                 int n_expert, int k) {
    const int per = n_tok >= 8 ? 8 : 1;
    return q.parallel_for(sycl::nd_range<1>((size_t) ((n_tok + per - 1) / per) * per * WARP, (size_t) per * WARP),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t tok = it.get_group(0) * per + sg.get_group_linear_id();
        if (tok >= (size_t) n_tok) return;   // uniform over the sub-group
        top10_token<NPL>(sg, logits_all + tok * n_expert, ids_all + tok * k, weights_all + tok * k, nullptr, nullptr,
                         n_expert, k);
    });
}

// NPL: the fewest values a lane that hold n_expert (<= 1,024)
sycl::event router_top_k_launch(sycl::queue& q, const float* logits, int32_t* ids, float* weights, int n_tok,
                                int n_expert, int k) {
    if (n_expert <= 32) return launch_native_router<1>(q, logits, ids, weights, n_tok, n_expert, k);
    if (n_expert <= 64) return launch_native_router<2>(q, logits, ids, weights, n_tok, n_expert, k);
    if (n_expert <= 128) return launch_native_router<4>(q, logits, ids, weights, n_tok, n_expert, k);
    if (n_expert <= 256) return launch_native_router<8>(q, logits, ids, weights, n_tok, n_expert, k);
    if (n_expert <= 512) return launch_native_router<16>(q, logits, ids, weights, n_tok, n_expert, k);
    return launch_native_router<32>(q, logits, ids, weights, n_tok, n_expert, k);
}

void launch_native_router_n(sycl::queue& q, const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert) {
    router_top_k_launch(q, logits, ids, weights, n_tok, n_expert, 10);
}

void require_shape(int n_expert, const char* what) {
    if (!native_router_takes(n_expert, 10))
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(n_expert) + " experts (256 or 512 only)");
}

// *p += 1, read and incremented in memory, as elementwise.cpp's bump_host (the doorbell's ring)
inline void bump_host(uint32_t* p) {
    volatile uint32_t* v = p;
    const uint32_t n = *v + 1u;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    *v = n;
}
}  // namespace

// The one router, the native one at any width (router_top_k_launch below).  Its reading is ggml's (FP32, the sum a
// butterfly), not ref/moe.py's double one this replaced: the ids and weights may differ by rounding.
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 32 || k >= n_expert || n_expert > 1024)
        throw core::DeviceError("router_top10: top " + std::to_string(k) + " of " + std::to_string(n_expert) +
                                " experts is past the kernel's top 32 of 1,024");
    auto& q = queue_for(stream, false);
    const auto e = router_top_k_launch(q, logits, ids, weights, n_tokens, n_expert, k);
    if (!stream) core::Runtime::get().wait(e, "router_top10");
}

namespace {
// router_publish's one launch for NPL values a lane: one work-group, the first n_tok sub-groups route a token each (as
// router_top10), every work-item copies the rows, and after the fence and the barrier one work-item rings the host (as
// doorbell_publish)
template <int NPL>
bool router_publish_launch(sycl::queue& q, const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert,
                           int k, const float* x, int64_t n_x, float* x_out, int32_t* ids_out, float* weights_out,
                           uint32_t* d_seq) {
    const int wg = xe::work_group_upto_1024_of<RouterPublishKernel<NPL>>(q);
    if (n_tok * WARP > wg) return false;
    const int ni = (int) n_x;
    q.parallel_for<RouterPublishKernel<NPL>>(sycl::nd_range<1>((size_t) wg, (size_t) wg),
                                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int t = (int) it.get_local_id(0);
        const int tok = (int) sg.get_group_linear_id();
        if (tok < n_tok)
            top10_token<NPL>(sg, logits + (size_t) tok * n_expert, ids + (size_t) tok * k, weights + (size_t) tok * k,
                             ids_out + (size_t) tok * k, weights_out + (size_t) tok * k, n_expert, k);
        for (int i = t; i < ni; i += wg) x_out[i] = x[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (t == 0) {
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            bump_host(d_seq);
        }
    });
    return true;
}
}  // namespace

bool router_publish(const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert, int k, const float* x,
                    int64_t n_x, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (n_tok < 1 || k < 1 || k > 32 || k >= n_expert || n_expert > 1024) return false;
    if (!stream || !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("router (publish) requires a stream and aligned [n,E]/[n,k] buffers");
    sycl::queue& q = queue_for(stream, true);
    const auto go = [&](auto npl) {
        return router_publish_launch<decltype(npl)::value>(q, logits, ids, weights, n_tok, n_expert, k, x, n_x, x_out,
                                                          ids_out, weights_out, d_seq);
    };
    if (n_expert <= 32) return go(std::integral_constant<int, 1>{});
    if (n_expert <= 64) return go(std::integral_constant<int, 2>{});
    if (n_expert <= 128) return go(std::integral_constant<int, 4>{});
    if (n_expert <= 256) return go(std::integral_constant<int, 8>{});
    if (n_expert <= 512) return go(std::integral_constant<int, 16>{});
    return go(std::integral_constant<int, 32>{});
}

bool native_router_takes(int n_expert, int k) { return (n_expert == 512 || n_expert == 256) && k == 10; }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream, int n_expert) {
    require_shape(n_expert, "native router");
    const size_t lb = (size_t) n_expert * 4, ob = 10 * sizeof(float);
    if (!stream || !valid(logits, lb) || !valid(ids, ob) || !valid(weights, ob)
        || overlap(logits, lb, ids, ob) || overlap(logits, lb, weights, ob)
        || overlap(ids, ob, weights, ob))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    launch_native_router_n(queue_for(stream, true), logits, ids, weights, 1, n_expert);
}

// STRATA_ROUTE_PRIOR=lambda (upstream 2de9ae06, opt-in, changes the output): cache-conditioned routing, after "Mixture
// of Cache-Conditional Experts" (arXiv 2412.00099) without training.  upstream picks the top 10 again with `lambda`
// added to the logit of every expert already in VRAM.  Here the router's exact top 10 (ids/weights, already written) is
// edited instead: from the lowest rank up, a non-resident expert is replaced by the best resident one not yet picked
// while that one's logit + lambda beats it - the same set upstream's biased top 10 picks.  The weights are the router's
// probabilities over the new 10.  A token without a swap keeps the router's bits, so lambda 0 (the device value, set per
// request) is the exact routing.  Bounding the swaps did not make the prior harmless (RTX 4070, 2026-10-30, teacher-
// forced against the exact routing): the first 4 ranks kept, NLL +4.2% (unbounded +8.2%) at nearly the same speed; the
// dropped probability capped at 2%, NLL +1.2% but the speed gone (+3%); lambda scaled by the router's entropy over its
// 10 changed nothing, the 10's weights are nearly flat.  Skipping missed experts instead (STRATA_ROUTE_TAIL_SKIP) keeps
// the text.
// stats (device, 4 x uint64, or nullptr): tokens, tokens with a swap, swaps, the dropped probability x 1e6.
namespace {
template <int NPL>
void route_prior_launch(sycl::queue& q, const float* logits, const int32_t* res_layer, const float* lambda_d,
                        int32_t* ids, float* weights, int n_tok, unsigned long long* stats) {
    q.parallel_for(sycl::nd_range<1>((size_t) n_tok * WARP, WARP), [=](sycl::nd_item<1> it)
                   [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t tok = it.get_group(0);
        const float* lg = logits + tok * NPL * 32;
        int32_t* id = ids + tok * 10;
        float* w = weights + tok * 10;
        const int lane = (int) sg.get_local_linear_id();
        const float lambda = *lambda_d;
        if (!(lambda > 0.0f)) return;
        int idl[10];
        for (int r = 0; r < 10; ++r) idl[r] = id[r];
        float l[NPL];
        bool cand[NPL];   // resident and not among the 10
        float mx = -INFINITY;
        for (int i = 0; i < NPL; ++i) {
            const int e = lane + 32 * i;
            l[i] = lg[e];
            mx = sycl::fmax(mx, l[i]);
            bool picked = false;
            for (int r = 0; r < 10; ++r) picked |= idl[r] == e;
            cand[i] = !picked && res_layer[e] >= 0;
        }
        for (int m = 16; m; m >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, m));
        float z = 0.0f;
        for (int i = 0; i < NPL; ++i) z += sycl::exp(l[i] - mx);
        z = sycl::reduce_over_group(sg, z, sycl::plus<float>());
        int swaps = 0;
        float dropped = 0.0f;
        for (int r = 9; r >= 0; --r) {
            const int e = idl[r];
            if (res_layer[e] >= 0) continue;
            const float le = lg[e];
            float best = -FLT_MAX;
            int be = 1 << 30;
            for (int i = 0; i < NPL; ++i)
                if (cand[i] && l[i] > best) { best = l[i]; be = lane + 32 * i; }
            for (int m = 16; m; m >>= 1) {
                const float ob = sycl::permute_group_by_xor(sg, best, m);
                const int oe = sycl::permute_group_by_xor(sg, be, m);
                if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; }
            }
            // the next non-resident one up has a larger logit and the next resident candidate a smaller one: stop
            if (be >= (1 << 30) || !(best + lambda > le)) break;
            dropped += sycl::exp(le - mx) / z;
            idl[r] = be;
            if ((be & 31) == lane) cand[be >> 5] = false;
            ++swaps;
        }
        if (stats != nullptr && lane == 0) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> s0(stats[0]), s1(stats[1]), s2(stats[2]), s3(stats[3]);
            s0.fetch_add(1ull);
            if (swaps > 0) {
                s1.fetch_add(1ull);
                s2.fetch_add((unsigned long long) swaps);
                s3.fetch_add((unsigned long long) (dropped * 1e6f));
            }
        }
        if (swaps == 0) return;
        float sp = 0.0f;
        for (int r = 0; r < 10; ++r) sp += sycl::exp(lg[idl[r]] - mx);
        if (lane < 10) {
            int e = idl[0];
            for (int r = 1; r < 10; ++r) if (lane == r) e = idl[r];
            id[lane] = e;
            w[lane] = sycl::exp(lg[e] - mx) / sycl::fmax(sp, 1e-30f);
        }
    });
}
}  // namespace

void route_prior_top10(const float* logits, const int32_t* res_layer, const float* lambda_d, int32_t* ids,
                       float* weights, int n_tok, unsigned long long* stats, void* stream, int n_expert) {
    if (n_tok <= 0) return;
    if (!stream || res_layer == nullptr || lambda_d == nullptr)
        throw std::invalid_argument("route-prior requires a stream, the residency row and lambda");
    require_shape(n_expert, "route-prior");
    sycl::queue& q = queue_for(stream, true);
    if (n_expert == 512) route_prior_launch<16>(q, logits, res_layer, lambda_d, ids, weights, n_tok, stats);
    else route_prior_launch<8>(q, logits, res_layer, lambda_d, ids, weights, n_tok, stats);
}

// STRATA_ROUTE_RESIDENT (upstream 5df35dcb, EXPERIMENTAL, opt-in, changes the output): residency-biased routing for
// 512 x 10.  After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not
// already picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           float margin, int rank_lo, int rank_hi, unsigned long long* stats, void* stream, int n_expert) {
    if (!stream || n_tok < 1 || res_layer == nullptr)
        throw std::invalid_argument("route-resident requires a stream, tokens and the residency row");
    require_shape(n_expert, "route-resident");
    const int lo = rank_lo, hi = rank_hi, ne = n_expert;
    // upstream 33fd01b4: a sub-group a token, lane j scanning experts j, j + 32, ...; the serial scan per token was the
    // window's own cost (upstream: 2.75 ms of a 22 ms window).  The same choice bit for bit: a strictly greater logit
    // wins, the smaller expert index breaks a tie, the ranks in order (a swap changes the set already picked).
    // STRATA_ROUTE_RESIDENT_SERIAL=1: the serial form, to compare.
    static const bool serial_env = [] {
        const char* v = strata::env_get("ROUTE_RESIDENT_SERIAL");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    if (!serial_env) {
        queue_for(stream, true).parallel_for(
            sycl::nd_range<1>((size_t) n_tok * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id();
                const size_t t = it.get_group_linear_id();
                const float* l = logits + t * ne;
                int32_t* idp = ids + t * 10;
                int32_t id[10];
                for (int r = 0; r < 10; ++r) id[r] = idp[r];
                int swaps = 0, tail = 0, before = 0;
                for (int r = 0; r < 10; ++r) before += res_layer[id[r]] < 0;
                for (int r = lo; r <= hi && r < 10; ++r) {
                    const int e = id[r];
                    if (res_layer[e] >= 0) continue;
                    ++tail;
                    int best = -1;
                    float bl = -INFINITY;
                    for (int f = lane; f < ne; f += WARP) {   // ascending: the lane keeps its first maximum
                        if (res_layer[f] < 0 || l[f] <= bl) continue;
                        bool used = false;
                        for (int q = 0; q < 10; ++q) used |= id[q] == f;
                        if (!used) { best = f; bl = l[f]; }
                    }
                    for (int o = WARP / 2; o > 0; o >>= 1) {   // the greatest logit, the smallest index on a tie
                        const float ob = sycl::permute_group_by_xor(sg, bl, o);
                        const int oi = sycl::permute_group_by_xor(sg, best, o);
                        if (oi >= 0 && (best < 0 || ob > bl || (ob == bl && oi < best))) { bl = ob; best = oi; }
                    }
                    if (best >= 0 && l[e] - bl <= margin) { id[r] = best; ++swaps; }
                }
                if (lane != 0) return;
                int after = 0;
                for (int r = 0; r < 10; ++r) after += res_layer[id[r]] < 0;
                if (swaps) {
                    for (int r = 0; r < 10; ++r) idp[r] = id[r];
                    float m = -INFINITY;
                    for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[id[r]]);
                    float ex[10], sum = 0.0f;
                    for (int r = 0; r < 10; ++r) { ex[r] = sycl::exp(l[id[r]] - m); sum += ex[r]; }
                    for (int r = 0; r < 10; ++r) weights[t * 10 + r] = ex[r] / sum;
                }
                if (stats) {
                    using A = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                                               sycl::memory_scope::device, sycl::access::address_space::global_space>;
                    A(stats[0]).fetch_add((unsigned long long) tail);
                    A(stats[1]).fetch_add((unsigned long long) swaps);
                    A(stats[2]).fetch_add((unsigned long long) before);
                    A(stats[3]).fetch_add((unsigned long long) after);
                }
            });
        return;
    }
    queue_for(stream, true).parallel_for(sycl::range<1>((size_t) n_tok), [=](sycl::id<1> i) {
        const size_t t = i[0];
        const float* l = logits + t * ne;
        int32_t* id = ids + t * 10;
        int swaps = 0, tail = 0, before = 0;
        for (int r = 0; r < 10; ++r) before += res_layer[id[r]] < 0;
        for (int r = lo; r <= hi && r < 10; ++r) {
            const int e = id[r];
            if (res_layer[e] >= 0) continue;
            ++tail;
            int best = -1;
            float bl = -INFINITY;
            for (int f = 0; f < ne; ++f) {
                if (res_layer[f] < 0 || l[f] <= bl) continue;
                bool used = false;
                for (int q = 0; q < 10; ++q) used |= id[q] == f;
                if (!used) { best = f; bl = l[f]; }
            }
            if (best >= 0 && l[e] - bl <= margin) { id[r] = best; ++swaps; }
        }
        int after = 0;
        for (int r = 0; r < 10; ++r) after += res_layer[id[r]] < 0;
        if (swaps) {
            float m = -INFINITY;
            for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[id[r]]);
            float ex[10], sum = 0.0f;
            for (int r = 0; r < 10; ++r) { ex[r] = sycl::exp(l[id[r]] - m); sum += ex[r]; }
            for (int r = 0; r < 10; ++r) weights[t * 10 + r] = ex[r] / sum;
        }
        if (stats) {
            using A = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;
            A(stats[0]).fetch_add((unsigned long long) tail);
            A(stats[1]).fetch_add((unsigned long long) swaps);
            A(stats[2]).fetch_add((unsigned long long) before);
            A(stats[3]).fetch_add((unsigned long long) after);
        }
    });
}

void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream,
                               int n_expert) {
    require_shape(n_expert, "native router (multi)");
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,E]/[n,10] buffers");
    launch_native_router_n(queue_for(stream, true), logits, ids, weights, n_tok, n_expert);
}

}  // namespace strata::kernels
