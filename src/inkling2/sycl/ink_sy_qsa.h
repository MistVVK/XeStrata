// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/inkling2/sycl/ink_sy_qsa.h - Inkling-2-Mini's QSA indexer: the full-attention visits' queries attend to the
// 4-token blocks it picks instead of every cell (long prompts and their decode).
//
// The weights are tamanekokoro/Inkling-2-Mini-QSA-Indexer's (MIT), file format QSAW v1 as its training script writes
// it (tools/qsa/train_qsa_modal_v3.py of Tamanekokko/Strata):
//   int32 magic 0x51534157, version 1, n_full, n_embd, h_idx, d_idx, block; float rope_theta; int32 vls[n_full];
//   then per full-attention visit: wq FP16 [h_idx * d_idx, n_embd], wk FP16 [d_idx, n_embd], q_norm FP32 [d_idx],
//   k_norm FP32 [d_idx], w_head FP32 [h_idx].
// The score of a complete block b for a query at a full-attention visit (the script's QsaIndexerLayer.forward, NoPE):
//   k_b = rms(wk . mean(the block's 4 key inputs)) * k_norm,  q_h = rms(wq_h . query input) * q_norm,
//   score = sum_h w_head[h] * relu(q_h . k_b) / sqrt(d_idx)
// with the first `sink` and the last `local` tokens' blocks kept always (the script's sink and local blocks, scored
// above every other).  The key and query inputs are the visit's own normed K and Q inputs (the runner's in3 rows), as
// the fork's engine reads them (r_ink, r_inq).
//
// What is shared with the engine's other QSA (SenseNova-Loop's, Qwen3.8-Flash-Next's): the selection itself,
// kernels::qsa_block_topk_with_current (whole blocks by score up to the budget, ties to the lowest block, the query's
// own incomplete block added), whose ids the attention reads; and the products (prefill::Gemm::f16).  The pooling and
// the scores are this indexer's own: it pools the inputs before its projection and weights its heads, where theirs
// pool projected keys with RoPE.
//
// Keys are kept for every slot whenever the indexer is loaded, so a request may turn it on or off (serve's qsa= key)
// with no reload and no new prompt.  A query uses the selection only where its context passes the budget; below it the
// selection would be every cell, and the dense attention runs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ink_sy_common.h"
#include "ink_sy_xmx.h"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

namespace ink_sycl {

struct InkSyQsa {
    static constexpr uint32_t kMagic = 0x51534157u;
    static constexpr int kBlock = 4;

    bool loaded = false;
    int n_full = 0, n_embd = 0, h_idx = 0, d_idx = 0;
    int budget = 2048, local = 256, sink = 16;
    int max_slots = 1, max_ctx = 0, max_blocks = 0, max_batch = 0, cap = 0;
    std::vector<int> fidx_of_visit;          // visit -> its full-attention index, or -1
    std::vector<half*> wq, wk;               // device, FP16
    std::vector<float*> q_norm, k_norm, w_head;
    float* k_comp = nullptr;                 // [slot][n_full][max_blocks][d_idx] FP32
    float* tail = nullptr;                   // [slot][n_full][4][n_embd]: the key inputs of a block not yet complete
    std::vector<uint8_t> slot_on;            // per slot: its queries through the selection

    // per forward (plan): the blocks this batch completes and where their four key inputs are
    int n_new = 0;
    int* new_dst = nullptr;                  // [n_new]: slot * max_blocks + block
    int* new_src = nullptr;                  // [n_new][4]: a row of the batch (>= 0), or -1 - (slot * 4 + q & 3): the tail
    int n_tail = 0;
    int* tail_row = nullptr;                 // [n_tail] rows whose key input the tail keeps
    int* tail_dst = nullptr;                 // [n_tail] slot * 4 + pos & 3
    bool any_sparse = false;
    uint8_t* use = nullptr;                  // [M]: query m attends through the selection
    int32_t* steps = nullptr;                // [M][kStepCount]
    int* q_slot = nullptr;                   // [M]

    half* kmean16 = nullptr;                 // [new blocks][n_embd]
    float* kraw = nullptr;                   // [new blocks][d_idx]
    half* q16 = nullptr;                     // [M][n_embd]
    float* qraw = nullptr;                   // [M][h_idx * d_idx], normed in place
    float* scores = nullptr;                 // [M][max_blocks]
    int32_t* ids = nullptr;                  // [M][cap]

    ~InkSyQsa() { release(); }

    void release() {
        for (auto* p : wq) dfree(p);
        for (auto* p : wk) dfree(p);
        for (auto* p : q_norm) dfree(p);
        for (auto* p : k_norm) dfree(p);
        for (auto* p : w_head) dfree(p);
        wq.clear(); wk.clear(); q_norm.clear(); k_norm.clear(); w_head.clear();
        for (void* p : {(void*)k_comp, (void*)tail, (void*)new_dst, (void*)new_src, (void*)tail_row, (void*)tail_dst,
                        (void*)use, (void*)steps, (void*)q_slot, (void*)kmean16, (void*)kraw, (void*)q16,
                        (void*)qraw, (void*)scores, (void*)ids})
            dfree(p);
        k_comp = tail = kraw = qraw = scores = nullptr;
        new_dst = new_src = tail_row = tail_dst = q_slot = nullptr;
        use = nullptr; steps = ids = nullptr; kmean16 = q16 = nullptr;
        loaded = false;
    }

    strata::kernels::QsaShapes shapes() const {
        strata::kernels::QsaShapes s;
        s.idx_n_head = h_idx;
        s.idx_dim = d_idx;
        s.idx_block = kBlock;
        s.idx_top_k = budget;
        s.page_size = 1;
        return s;
    }

    bool load(const std::string& path, int n_visits, int hidden, int budget_tokens, int local_tokens, int sink_tokens,
              int slots, int ctx, int batch_cap, std::string& err) {
        release();
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) { err = "cannot open the QSA indexer " + path; return false; }
        int32_t h[7] = {};
        float theta = 0.0f;
        const bool head_ok = std::fread(h, 4, 7, fp) == 7 && std::fread(&theta, 4, 1, fp) == 1;
        auto fail = [&](const std::string& why) { std::fclose(fp); err = path + ": " + why; return false; };
        if (!head_ok || (uint32_t)h[0] != kMagic) return fail("not a QSA indexer (QSAW)");
        if (h[1] != 1) return fail("QSAW version " + std::to_string(h[1]) + " (this engine reads 1)");
        n_full = h[2]; n_embd = h[3]; h_idx = h[4]; d_idx = h[5];
        if (n_embd != hidden || h[6] != kBlock || d_idx != 128 || h_idx < 1 || h_idx > 16 || n_full < 1 || n_full > n_visits)
            return fail("an indexer of another shape (n_embd " + std::to_string(n_embd) + ", block " + std::to_string(h[6]) +
                        ", d_idx " + std::to_string(d_idx) + ")");
        if (theta != 0.0f) return fail("an indexer with RoPE (theta " + std::to_string(theta) + "); this engine scores without");
        std::vector<int32_t> vls((size_t)n_full);
        if (std::fread(vls.data(), 4, vls.size(), fp) != vls.size()) return fail("short");
        fidx_of_visit.assign((size_t)n_visits, -1);
        for (int f = 0; f < n_full; ++f) {
            if (vls[f] < 0 || vls[f] >= n_visits) return fail("visit " + std::to_string(vls[f]) + " out of range");
            fidx_of_visit[(size_t)vls[f]] = f;
        }
        const size_t nq = (size_t)h_idx * d_idx * n_embd, nk = (size_t)d_idx * n_embd;
        std::vector<uint16_t> hq(nq), hk(nk);
        std::vector<float> hqn((size_t)d_idx), hkn((size_t)d_idx), hwh((size_t)h_idx);
        // NOLINTBEGIN(clang-analyzer-unix.Stream): each read below is size-checked, so one after EOF ends in fail();
        // the analyzer also assumes EOF on a read that returned its full count, which is harmless here
        for (int f = 0; f < n_full; ++f) {
            if (std::fread(hq.data(), 2, nq, fp) != nq || std::fread(hk.data(), 2, nk, fp) != nk ||
                std::fread(hqn.data(), 4, hqn.size(), fp) != hqn.size() ||
                std::fread(hkn.data(), 4, hkn.size(), fp) != hkn.size() ||
                std::fread(hwh.data(), 4, hwh.size(), fp) != hwh.size())
                return fail("truncated at full-attention layer " + std::to_string(f));
            wq.push_back(dalloc<half>(nq)); h2d(wq.back(), hq.data(), nq * 2);
            wk.push_back(dalloc<half>(nk)); h2d(wk.back(), hk.data(), nk * 2);
            q_norm.push_back(dalloc<float>(hqn.size())); h2d(q_norm.back(), hqn.data(), hqn.size() * 4);
            k_norm.push_back(dalloc<float>(hkn.size())); h2d(k_norm.back(), hkn.data(), hkn.size() * 4);
            w_head.push_back(dalloc<float>(hwh.size())); h2d(w_head.back(), hwh.data(), hwh.size() * 4);
            wait();
        }
        if (std::fgetc(fp) != EOF) return fail("longer than its header says");
        // NOLINTEND(clang-analyzer-unix.Stream)
        std::fclose(fp);

        budget = budget_tokens; local = local_tokens; sink = sink_tokens;
        max_slots = slots; max_ctx = ctx; max_batch = batch_cap;
        max_blocks = (ctx + kBlock - 1) / kBlock + 1;
        cap = budget + kBlock;
        const size_t new_cap = (size_t)batch_cap / kBlock + (size_t)slots + 1;
        k_comp = dalloc<float>((size_t)slots * n_full * max_blocks * d_idx);
        tail = dalloc<float>((size_t)slots * n_full * kBlock * n_embd);
        dzero(tail, (size_t)slots * n_full * kBlock * n_embd * 4);
        new_dst = dalloc<int>(new_cap); new_src = dalloc<int>(new_cap * kBlock);
        tail_row = dalloc<int>((size_t)batch_cap); tail_dst = dalloc<int>((size_t)batch_cap);
        use = dalloc<uint8_t>((size_t)batch_cap);
        steps = dalloc<int32_t>((size_t)batch_cap * strata::kernels::kStepCount);
        q_slot = dalloc<int>((size_t)batch_cap);
        kmean16 = dalloc<half>(new_cap * n_embd);
        kraw = dalloc<float>(new_cap * d_idx);
        q16 = dalloc<half>((size_t)batch_cap * n_embd);
        qraw = dalloc<float>((size_t)batch_cap * h_idx * d_idx);
        scores = dalloc<float>((size_t)batch_cap * max_blocks);
        ids = dalloc<int32_t>((size_t)batch_cap * cap);
        slot_on.assign((size_t)slots, 0);
        wait();
        loaded = true;
        std::fprintf(stderr, "[ink_sycl] QSA indexer %s: %d full-attention visits, budget %d (sink %d, local %d) tokens, "
                     "keys %.1f MiB\n", path.c_str(), n_full, budget, sink, local,
                     (double)slots * n_full * max_blocks * d_idx * 4 / (1024.0 * 1024.0));
        return true;
    }

    // The batch's plan from the host's positions: the blocks it completes, the tail it leaves, which queries select.
    void plan(const int* slot_ids, const int* positions, int M) {
        std::vector<int> dst, src, trow, tdst;
        std::vector<uint8_t> u((size_t)M, 0);
        std::vector<int32_t> st((size_t)M * strata::kernels::kStepCount);
        std::vector<int> qs((size_t)M);
        const auto sh = shapes();
        // the row of (slot, position) in this batch, else -1
        auto row_of = [&](int s, int p) {
            for (int m = 0; m < M; ++m) if (slot_ids[m] == s && positions[m] == p) return m;
            return -1;
        };
        std::vector<int> latest((size_t)max_slots * kBlock, -1);   // the tail entry's newest row
        any_sparse = false;
        for (int m = 0; m < M; ++m) {
            const int s = slot_ids[m], p = positions[m];
            qs[(size_t)m] = s;
            int& l = latest[(size_t)s * kBlock + (p & 3)];
            if (l < 0 || positions[l] < p) l = m;
            if ((p & 3) == 3) {
                dst.push_back(s * max_blocks + (p >> 2));
                for (int q = p - 3; q <= p; ++q) {
                    const int r = row_of(s, q);
                    src.push_back(r >= 0 ? r : -1 - (s * kBlock + (q & 3)));
                }
            }
            strata::kernels::qsa_step_fill(st.data() + (size_t)m * strata::kernels::kStepCount, p, sh);
            u[(size_t)m] = slot_on[(size_t)s] && p + 1 > budget + kBlock - 1;
            any_sparse = any_sparse || u[(size_t)m];
        }
        for (size_t i = 0; i < latest.size(); ++i)
            if (latest[i] >= 0) { trow.push_back(latest[i]); tdst.push_back((int)i); }
        n_new = (int)dst.size();
        n_tail = (int)trow.size();
        h2d(new_dst, dst.data(), dst.size() * 4);
        h2d(new_src, src.data(), src.size() * 4);
        h2d(tail_row, trow.data(), trow.size() * 4);
        h2d(tail_dst, tdst.data(), tdst.size() * 4);
        h2d(use, u.data(), u.size());
        h2d(steps, st.data(), st.size() * 4);
        h2d(q_slot, qs.data(), qs.size() * 4);
        wait();   // the host vectors go out of scope
    }

    // The visit's keys: the blocks this batch completes, pooled from their four key inputs, projected and normed; then
    // the tail keeps the key inputs of the blocks not yet complete.  `in_k` [M][n_embd].
    void keys(int f, const float* in_k, int M) {
        const int D = n_embd, DI = d_idx, MB = max_blocks, NF = n_full, F = f;
        float* tl = tail;
        if (n_new > 0) {
            const int* src = new_src;
            half* km = kmean16;
            Q().parallel_for(NR2((D + 255) / 256, (size_t)n_new, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
                const int j = (int)it.get_group(0);
                const int d = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
                if (d >= D) return;
                float acc = 0.0f;
                for (int t = 0; t < kBlock; ++t) {
                    const int r = src[j * kBlock + t];
                    acc += r >= 0 ? in_k[(size_t)r * D + d]
                                  : tl[((size_t)(-1 - r) / kBlock * NF + F) * kBlock * D + (size_t)((-1 - r) % kBlock) * D + d];
                }
                km[(size_t)j * D + d] = (half)(acc * 0.25f);
            });
            gemm_f16(kmean16, wk[(size_t)f], kraw, n_new, DI, D, DI);
            const float* kr = kraw; const float* kn = k_norm[(size_t)f]; const int* nd = new_dst; float* kc = k_comp;
            Q().parallel_for(NR((size_t)n_new, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
                float* sh = lmem<float, 16>(it);
                const int j = (int)it.get_group(0), t = (int)it.get_local_id(0);
                const float x = kr[(size_t)j * DI + t];
                const float inv = sycl::rsqrt(wg_sum16(it, x * x, sh) / (float)DI + 1e-6f);
                const int sb = nd[j], s = sb / MB, b = sb % MB;
                kc[(((size_t)s * NF + F) * MB + b) * DI + t] = x * inv * kn[t];
            });
        }
        if (n_tail > 0) {
            const int* tr = tail_row; const int* td = tail_dst;
            Q().parallel_for(NR2((D + 255) / 256, (size_t)n_tail, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
                const int j = (int)it.get_group(0);
                const int d = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
                if (d >= D) return;
                const int sl = td[j] / kBlock, q = td[j] % kBlock;
                tl[(((size_t)sl * NF + F) * kBlock + q) * D + d] = in_k[(size_t)tr[j] * D + d];
            });
        }
        (void)M;
    }

    // The visit's selection for the batch's queries (`in_q` [M][n_embd]): ids [M][cap] and their widths (steps), read
    // by the attention where use[m].  Only where some query selects (any_sparse).
    void select(int f, const float* in_q, int M) {
        const int D = n_embd, DI = d_idx, HI = h_idx, MB = max_blocks, NF = n_full, F = f;
        const int sink_b = sink / kBlock, local_b = local / kBlock;
        k_f32_to_f16_sg16(in_q, q16, (size_t)M * D);
        gemm_f16(q16, wq[(size_t)f], qraw, M, HI * DI, D, HI * DI);
        {
            float* qr = qraw; const float* qn = q_norm[(size_t)f];
            Q().parallel_for(NR((size_t)M * HI, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
                float* sh = lmem<float, 16>(it);
                const size_t row = it.get_group(0);
                const int t = (int)it.get_local_id(0);
                const float x = qr[row * DI + t];
                const float inv = sycl::rsqrt(wg_sum16(it, x * x, sh) / (float)DI + 1e-6f);
                qr[row * DI + t] = x * inv * qn[t];
            });
        }
        {
            const float* qr = qraw; const float* kc = k_comp; const float* wh = w_head[(size_t)f];
            const int32_t* st = steps; const int* qsl = q_slot; const uint8_t* u = use; float* sc = scores;
            const float scale = 1.0f / sycl::sqrt((float)DI);
            // a work-item a (query, block): the block's key against the query's heads
            Q().parallel_for(NR2((MB + 255) / 256, (size_t)M, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
                const int m = (int)it.get_group(0);
                const int b = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
                const int n_bid = st[m * strata::kernels::kStepCount + strata::kernels::kStepNKv] / kBlock;
                if (b >= MB) return;
                float v = -1.0f;
                if (b < n_bid && u[m]) {   // (a query that does not select: its ids are not read)
                    if (b < sink_b || b >= n_bid - local_b) {
                        v = 1e6f;
                    } else {
                        const float* kp = kc + (((size_t)qsl[m] * NF + F) * MB + b) * DI;
                        v = 0.0f;
                        for (int h = 0; h < HI; ++h) {
                            const float* qp = qr + ((size_t)m * HI + h) * DI;
                            float d = 0.0f;
                            for (int t = 0; t < DI; ++t) d += qp[t] * kp[t];
                            v += wh[h] * sycl::fmax(d * scale, 0.0f);
                        }
                    }
                }
                sc[(size_t)m * MB + b] = v;
            });
        }
        auto sh = shapes();
        strata::kernels::qsa_block_topk_with_current(scores, steps, M, max_blocks, cap, sh, ids, &Q());
    }
};

}  // namespace ink_sycl
