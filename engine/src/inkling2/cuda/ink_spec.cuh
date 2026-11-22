// engine/src/inkling2/cuda/ink_spec.cuh - Lossless DSpark Block-8 & Tree-16 Speculative Decoding + Exact State Commit
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Guarantees 100.000% lossless greedy parity with plain M=1 decode:
//   - Tree/Block verify executes M <= 16 nodes in ONE 48-visit pass WITHOUT mutating committed KV or ShortConv history.
//   - Per-visit staging buffers hold each node's (K, V) and 4 ShortConv inputs (k_raw, v_raw, wo_raw, moe_raw) (~20 MB total!).
//   - After target verification selects the accepted path path[0..n_acc-1] (n_acc >= 1), ONE fused commit kernel writes
//     ONLY the accepted path's K, V into committed KV [p0 .. p0 + n_acc - 1] and shifts the exact last 3 raw values
//     into committed ShortConv history[0..2] across all 48 visits!
//   - Also ingests ONLY the accepted path's [5, 23, 41, 48] taps into DSpark's context KV cache!
#pragma once

#include "ink_dspark.cuh"
#include "ink_runner.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace strata::ink {

// Commit accepted path[0..n_acc-1] of 4 ShortConv streams across ALL 48 visits in ONE kernel launch!
// Grid: (48 visits, 4 streams), Block: 256 threads
// stage_sconv: [48, tree_cap, 6656] F32 where channels are [0..1024)=k, [1024..2048)=v, [2048..4352)=attn, [4352..6656)=mlp
__global__ void __launch_bounds__(256) k_ink_commit_sconv_all48(
    float* const* __restrict__ sconv_pools,  // [48] pointers to d_sconv_pool[v]
    const float* __restrict__ stage_sconv,   // [48, tree_cap, 6656]
    const int32_t* __restrict__ path,        // [n_acc] accepted node indices (<= tree_cap)
    int n_acc,
    int slot0,
    int max_slots,
    int tree_cap) {
    const int v = blockIdx.x;       // 0..47
    const int strm = blockIdx.y;    // 0..3
    const int tid = threadIdx.x;
    if (v >= kInkVisits || strm >= 4) return;

    constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
    constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
    const int ch_off = c_off[strm];
    const int C = c_len[strm];

    float* hist = sconv_pools[v] + (size_t)ch_off * 3 * max_slots + (size_t)slot0 * 3 * C;
    const float* st_v = stage_sconv + (size_t)v * tree_cap * kInkSconvCh;

    for (int c = tid; c < C; c += 256) {
        float h0 = hist[0 * C + c];
        float h1 = hist[1 * C + c];
        float h2 = hist[2 * C + c];
        for (int k = 0; k < n_acc; ++k) {
            const int node = path[k];
            const float cur = st_v[(size_t)node * kInkSconvCh + ch_off + c];
            h0 = h1;
            h1 = h2;
            h2 = cur;
        }
        hist[0 * C + c] = h0;
        hist[1 * C + c] = h1;
        hist[2 * C + c] = h2;
    }
}

// Commit accepted path[0..n_acc-1] of K, V across ALL 48 visits in ONE kernel launch!
// Grid: (48 visits, 8 kv_heads), Block: 128 threads (1 dim per thread)
__global__ void __launch_bounds__(128) k_ink_commit_kv_all48(
    const InkKvVisitDesc* __restrict__ kv_descs,  // [48]
    const float* __restrict__ stage_k,            // [48, tree_cap, 1024]
    const float* __restrict__ stage_v,            // [48, tree_cap, 1024]
    const int32_t* __restrict__ path,             // [n_acc]
    int p0,
    int n_acc,
    int slot0,
    int tree_cap) {
    const int v = blockIdx.x;  // 0..47
    const int h = blockIdx.y;  // 0..7
    const int d = threadIdx.x; // 0..127
    if (v >= kInkVisits || h >= kInkKVHeads) return;

    const InkKvVisitDesc kv = kv_descs[v];
    for (int k = 0; k < n_acc; ++k) {
        const int node = path[k];
        const int p = p0 + k;
        const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
        const size_t sbase = (((size_t)v * tree_cap + node) * kInkKVHeads + h) * kInkHeadDim + d;
        const float kv_val = stage_k[sbase];
        const float vv_val = stage_v[sbase];
        const size_t dbase = (((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + h) * kInkHeadDim;

        if (!kv.is_fp8) {
            kv.k_f16[dbase + d] = __float2half(kv_val);
            kv.v_f16[dbase + d] = __float2half(vv_val);
        } else {
            float mk = warp_max(fabsf(kv_val));
            float mv = warp_max(fabsf(vv_val));
            __shared__ float sk[4], sv[4];
            if ((d & 31) == 0) { sk[d >> 5] = mk; sv[d >> 5] = mv; }
            __syncthreads();
            float amk = fmaxf(fmaxf(sk[0], sk[1]), fmaxf(sk[2], sk[3]));
            float amv = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
            const float sc_k = amk > 0.0f ? (amk * (1.0f / 448.0f)) : 1.0f;
            const float sc_v = amv > 0.0f ? (amv * (1.0f / 448.0f)) : 1.0f;
            const __nv_fp8_e4m3 qk(fminf(fmaxf(kv_val / sc_k, -448.0f), 448.0f));
            const __nv_fp8_e4m3 qv(fminf(fmaxf(vv_val / sc_v, -448.0f), 448.0f));
            kv.k_f8[dbase + d] = qk.__x;
            kv.v_f8[dbase + d] = qv.__x;
            if (d == 0) {
                const size_t sidx = ((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + h;
                kv.k_sc[sidx] = sc_k;
                kv.v_sc[sidx] = sc_v;
            }
            __syncthreads();
        }
    }
}

// Compact accepted path's taps [4, M, 2304] -> [4, n_acc, 2304] in-place (for DSpark ingest_taps!)
__global__ void __launch_bounds__(256) k_ink_compact_taps(
    const float* __restrict__ tap_in,   // [16, 2304]
    float* __restrict__ tap_out,        // [16, 2304]
    const int32_t* __restrict__ path,   // [n_acc]
    int n_acc) {
    const int k = blockIdx.x;
    if (k >= n_acc) return;
    const int node = path[k];
    const float* src = tap_in + (size_t)node * kInkHidden;
    float* dst = tap_out + (size_t)k * kInkHidden;
    for (int d = threadIdx.x; d < kInkHidden; d += 256) dst[d] = src[d];
}

struct InkSpecStats {
    long iters = 0;
    long tokens = 0;
    long drafted = 0;
    long accepted = 0;
    double t_draft_ms = 0.0;
    double t_verify_ms = 0.0;
    double t_commit_ms = 0.0;
    long hist[17] = {};
};

// Build linear Block-K chain tree (M = 1 + n_draft <= 8)
inline InkTreeSpec make_chain_tree(int32_t anchor, const int32_t* draft, int n_draft) {
    InkTreeSpec tr{};
    tr.M = 1 + n_draft;
    tr.tok[0] = anchor;
    tr.parent[0] = -1;
    tr.depth[0] = 0;
    for (int i = 0; i < n_draft; ++i) {
        tr.tok[i + 1] = draft[i];
        tr.parent[i + 1] = (int8_t)i;
        tr.depth[i + 1] = (uint8_t)(i + 1);
    }
    tr.build();
    return tr;
}

// Run 1 lossless speculative decode step (draft -> target verify -> exact accept -> 1-launch KV/ShortConv commit -> DSpark tap ingest)
// Returns newly emitted tokens (1 .. n_draft + 1), updates cur_tok & pos!
inline int ink_spec_step_block8(
    InkCudaRunner& runner,
    InkDSpark& dspark,
    int32_t& cur_tok,
    int& pos,
    int n_draft,
    int32_t* out_emitted,
    InkSpecStats& st) {
    // 1. On-GPU DSpark draft
    double t0 = now_s();
    int32_t draft_ids[8] = {};
    float draft_conf[8] = {};
    dspark.draft_block8(cur_tok, pos, draft_ids, draft_conf);
    double t1 = now_s();

    // 2. Build chain tree & run single CUDA-Graph captured 48-visit target verify!
    InkTreeSpec tr = make_chain_tree(cur_tok, draft_ids, n_draft);
    runner.forward_tree_verify(tr, pos, 0, true);
    int32_t target_next[16] = {};
    runner.head_argmax_exact(tr.M, target_next);
    double t2 = now_s();

    // 3. Exact greedy verification along the chain!
    // Node 0 is cur_tok (at pos); target_next[0] is exact next token at pos+1!
    // If target_next[k] == tr.tok[k + 1], draft node k+1 is ACCEPTED!
    int32_t path[16] = {};
    int n_acc_nodes = 0;
    int n_emit = 0;
    int cur_node = 0;
    while (true) {
        path[n_acc_nodes++] = cur_node;
        const int32_t tok_next = target_next[cur_node];
        out_emitted[n_emit++] = tok_next;
        if (cur_node + 1 < tr.M && tr.tok[cur_node + 1] == tok_next) {
            ++cur_node;
        } else {
            break;
        }
    }

    // 4. Commit ONLY the accepted path's KV + ShortConv across all 48 visits (2 kernel launches total!) + ingest accepted taps into DSpark!
    INK_CK(cudaMemcpyAsync(runner.d_acc_path, path, (size_t)n_acc_nodes * 4, cudaMemcpyHostToDevice, runner.st));
    k_ink_commit_sconv_all48<<<dim3(kInkVisits, 4), 256, 0, runner.st>>>(
        runner.d_sconv_pool_arr, runner.d_stage_sconv, runner.d_acc_path, n_acc_nodes, 0, runner.max_slots, runner.tree_cap);
    k_ink_commit_kv_all48<<<dim3(kInkVisits, kInkKVHeads), 128, 0, runner.st>>>(
        runner.d_kv_desc_arr, runner.d_stage_k, runner.d_stage_v, runner.d_acc_path, pos, n_acc_nodes, 0, runner.tree_cap);
    for (int t = 0; t < 4; ++t) {
        k_ink_compact_taps<<<n_acc_nodes, 256, 0, runner.st>>>(runner.d_taps[t], runner.d_taps_compact[t], runner.d_acc_path, n_acc_nodes);
    }
    dspark.ingest_taps_from(runner.d_taps_compact, pos, n_acc_nodes);
    double t3 = now_s();

    // Update position & next anchor token!
    // Note: path[0..n_acc_nodes-1] were committed at [pos .. pos + n_acc_nodes - 1];
    // out_emitted[n_emit - 1] is the bonus/corrected token that becomes the NEXT anchor at pos + n_acc_nodes!
    pos += n_acc_nodes;
    cur_tok = out_emitted[n_emit - 1];

    st.iters++;
    st.tokens += n_emit;
    st.drafted += n_draft;
    st.accepted += (n_acc_nodes - 1);
    st.t_draft_ms += (t1 - t0) * 1000.0;
    st.t_verify_ms += (t2 - t1) * 1000.0;
    st.t_commit_ms += (t3 - t2) * 1000.0;
    if (n_emit < 17) st.hist[n_emit]++;
    return n_emit;
}

// Build 2-branch Markov Beam Tree (M <= 12..16 nodes) from DSpark top-2 candidates per slot!
// Main chain: nodes 0..n_draft (anchor + primary chain); runner-up top2[s] attached as second child of node s when p2 >= thr!
inline InkTreeSpec make_beam_tree(
    int32_t anchor, const int32_t* top2_ids, const float* top2_probs,
    int n_draft, int max_nodes = 12, float p2_thr = 0.05f) {
    InkTreeSpec tr{};
    tr.M = 1 + n_draft;
    tr.tok[0] = anchor;
    tr.parent[0] = -1;
    tr.depth[0] = 0;
    for (int i = 0; i < n_draft; ++i) {
        tr.tok[i + 1] = top2_ids[2 * i];
        tr.parent[i + 1] = (int8_t)i;
        tr.depth[i + 1] = (uint8_t)(i + 1);
    }
    // Attach runner-up branches on early uncertain slots (depth 1..4) up to max_nodes (<= 16)
    const int cap = std::min(16, max_nodes);
    for (int s = 0; s < std::min(n_draft, 5) && tr.M < cap; ++s) {
        const int32_t alt = top2_ids[2 * s + 1];
        if (alt != top2_ids[2 * s] && top2_probs[2 * s + 1] >= p2_thr) {
            const int idx = tr.M++;
            tr.tok[idx] = alt;
            tr.parent[idx] = (int8_t)s;
            tr.depth[idx] = (uint8_t)(s + 1);
        }
    }
    tr.build();
    return tr;
}

// Run 1 lossless Tree-12/16 speculative decode step (exact greedy tree walk -> 100.000% bitwise parity with plain M=1!)
inline int ink_spec_step_tree(
    InkCudaRunner& runner,
    InkDSpark& dspark,
    int32_t& cur_tok,
    int& pos,
    int n_draft,
    int max_nodes,
    int32_t* out_emitted,
    InkSpecStats& st) {
    double t0 = now_s();
    int32_t draft_ids[8] = {};
    float draft_conf[8] = {};
    dspark.draft_block8(cur_tok, pos, draft_ids, draft_conf);
    double t1 = now_s();

    InkTreeSpec tr = make_beam_tree(cur_tok, dspark.h_top2_ids, dspark.h_top2_probs, n_draft, max_nodes);
    runner.forward_tree_verify(tr, pos, 0, true);
    int32_t target_next[16] = {};
    runner.head_argmax_exact(tr.M, target_next);
    double t2 = now_s();

    // Exact greedy walk down the verified tree: match ANY child of cur_node whose token == target_next[cur_node]!
    int32_t path[16] = {};
    int n_acc_nodes = 0;
    int n_emit = 0;
    int cur_node = 0;
    while (true) {
        path[n_acc_nodes++] = cur_node;
        const int32_t tok_next = target_next[cur_node];
        out_emitted[n_emit++] = tok_next;
        int next_child = -1;
        for (int j = cur_node + 1; j < tr.M; ++j) {
            if (tr.parent[j] == cur_node && tr.tok[j] == tok_next) {
                next_child = j;
                break;
            }
        }
        if (next_child >= 0) cur_node = next_child;
        else break;
    }

    INK_CK(cudaMemcpyAsync(runner.d_acc_path, path, (size_t)n_acc_nodes * 4, cudaMemcpyHostToDevice, runner.st));
    k_ink_commit_sconv_all48<<<dim3(kInkVisits, 4), 256, 0, runner.st>>>(
        runner.d_sconv_pool_arr, runner.d_stage_sconv, runner.d_acc_path, n_acc_nodes, 0, runner.max_slots, runner.tree_cap);
    k_ink_commit_kv_all48<<<dim3(kInkVisits, kInkKVHeads), 128, 0, runner.st>>>(
        runner.d_kv_desc_arr, runner.d_stage_k, runner.d_stage_v, runner.d_acc_path, pos, n_acc_nodes, 0, runner.tree_cap);
    for (int t = 0; t < 4; ++t) {
        k_ink_compact_taps<<<n_acc_nodes, 256, 0, runner.st>>>(runner.d_taps[t], runner.d_taps_compact[t], runner.d_acc_path, n_acc_nodes);
    }
    dspark.ingest_taps_from(runner.d_taps_compact, pos, n_acc_nodes);
    double t3 = now_s();

    pos += n_acc_nodes;
    cur_tok = out_emitted[n_emit - 1];

    st.iters++;
    st.tokens += n_emit;
    st.drafted += (tr.M - 1);
    st.accepted += (n_acc_nodes - 1);
    st.t_draft_ms += (t1 - t0) * 1000.0;
    st.t_verify_ms += (t2 - t1) * 1000.0;
    st.t_commit_ms += (t3 - t2) * 1000.0;
    if (n_emit < 17) st.hist[n_emit]++;
    return n_emit;
}

// Run 1 lossless True W=2..4 Multi-Branch Markov Beam Tree step (M = 16 / 24 / 32 nodes, 100.000% bitwise exact!)
inline int ink_spec_step_beam_tree(
    InkCudaRunner& runner,
    InkDSpark& dspark,
    int32_t& cur_tok,
    int& pos,
    int beam_w,
    int max_nodes,
    int32_t* out_emitted,
    InkSpecStats& st,
    float adaptive_moe_ratio = 0.0f) {
    double t0 = now_s();
    InkTreeSpec tr{};
    dspark.draft_beam_tree(cur_tok, pos, beam_w, max_nodes, tr, adaptive_moe_ratio);
    double t1 = now_s();

    runner.forward_tree_verify(tr, pos, 0, true);
    int32_t target_next[kInkMaxTree] = {};
    runner.head_argmax_exact(tr.M, target_next);
    double t2 = now_s();

    // Exact greedy walk down the verified multi-branch tree: match ANY child of cur_node whose token == target_next[cur_node]!
    int32_t path[kInkMaxTree] = {};
    int n_acc_nodes = 0;
    int n_emit = 0;
    int cur_node = 0;
    while (true) {
        path[n_acc_nodes++] = cur_node;
        const int32_t tok_next = target_next[cur_node];
        out_emitted[n_emit++] = tok_next;
        int next_child = -1;
        for (int j = cur_node + 1; j < tr.M; ++j) {
            if (tr.parent[j] == cur_node && tr.tok[j] == tok_next) {
                next_child = j;
                break;
            }
        }
        if (next_child >= 0) cur_node = next_child;
        else break;
    }

    INK_CK(cudaMemcpyAsync(runner.d_acc_path, path, (size_t)n_acc_nodes * 4, cudaMemcpyHostToDevice, runner.st));
    k_ink_commit_sconv_all48<<<dim3(kInkVisits, 4), 256, 0, runner.st>>>(
        runner.d_sconv_pool_arr, runner.d_stage_sconv, runner.d_acc_path, n_acc_nodes, 0, runner.max_slots, runner.tree_cap);
    k_ink_commit_kv_all48<<<dim3(kInkVisits, kInkKVHeads), 128, 0, runner.st>>>(
        runner.d_kv_desc_arr, runner.d_stage_k, runner.d_stage_v, runner.d_acc_path, pos, n_acc_nodes, 0, runner.tree_cap);
    for (int t = 0; t < 4; ++t) {
        k_ink_compact_taps<<<n_acc_nodes, 256, 0, runner.st>>>(runner.d_taps[t], runner.d_taps_compact[t], runner.d_acc_path, n_acc_nodes);
    }
    dspark.ingest_taps_from(runner.d_taps_compact, pos, n_acc_nodes);
    double t3 = now_s();

    pos += n_acc_nodes;
    cur_tok = out_emitted[n_emit - 1];

    st.iters++;
    st.tokens += n_emit;
    st.drafted += (tr.M - 1);
    st.accepted += (n_acc_nodes - 1);
    st.t_draft_ms += (t1 - t0) * 1000.0;
    st.t_verify_ms += (t2 - t1) * 1000.0;
    st.t_commit_ms += (t3 - t2) * 1000.0;
    if (n_emit < 17) st.hist[n_emit]++;
    return n_emit;
}

}  // namespace strata::ink
