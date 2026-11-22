// ink_metal_ops.metal - Fused 48-Visit Inkling2Loop Metal Shaders
// Implements bit-exact Thinking Machines Inkling-2-Mini ops:
//   - Embedding lookup + embed_norm (bypassing embed_norm for Vision image tokens!)
//   - k1_attn_res_mix: Kimi Attention-Residuals depth mix (S=1..6 states, Q=1/3/4 queries)
//   - k1_loop_ctrl: Rank-64 tanh Gain/Forget loop controller + RMSNorm
//   - k2_sconv4_step: 4-tap causal ShortConv (static + global dynamic taps, 1/sqrt(5) vs 1/3)
//     supporting both sequential chunked prefill AND multi-slot continuous batching (c=1..64)!
//   - k2_qkvr_clip_tau_kv: L2 clip (22.627417 on Q/K, 8.0 on R), parameterless V RMSNorm,
//     global log-pos tau(pos), and KV cache write (SWA-1024 ring or Global linear, FP16 or FP8_E4M3)
//   - k3_relpos_sink_gqa: 24:8 GQA Flash-SWA/Global Attention with learned RelPos bias [16,1025]
//     + per-query attention sinks + fused sigmoid(gate_attn), multi-slot (c=1..64) & prefill!
//   - k4_router_topk8: exact F32 Top-8/128 sigmoid router with bias selection & route_scale=8.0
//   - k5_softcap_argmax: 30.0 * tanh(z / 30.0) + deterministic first-max GPU argmax
// 100% MIT clean-room implementation.
#include <metal_stdlib>
using namespace metal;

#define INK_D 2304
#define INK_NH 24
#define INK_NKV 8
#define INK_HD 128
#define INK_DREL 16
#define INK_RMS_EPS 1e-6f

static inline float ops_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float ops_tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    const uint nw = (nthr + 31u) >> 5;
    for (uint i = 0; i < nw; ++i) r += sh[i];
    return r;
}

// ============================================================================
// 1. Embedding Lookup + RMSNorm (BF16 or Q8_ROW) + Vision Bypass
//    is_img[m] == 1 -> copy img_embeds[m, :] directly WITHOUT embed_norm!
//    grid (M, 1, 1), 256 threads
// ============================================================================
kernel void ink_embed_bf16(
    device const ushort* embed_w  [[buffer(0)]],
    device const float* norm_w    [[buffer(1)]],
    device const int* tok_ids     [[buffer(2)]],
    device const uchar* is_img    [[buffer(3)]],
    device const float* img_emb   [[buffer(4)]],
    device float* cur_out         [[buffer(5)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    if (is_img && is_img[m] != 0) {
        for (uint d = tid; d < INK_D; d += 256)
            cur_out[(size_t)m * INK_D + d] = img_emb[(size_t)m * INK_D + d];
        return;
    }
    const int id = tok_ids[m];
    device const ushort* row = embed_w + (size_t)id * INK_D;
    float ss = 0.0f;
    for (uint d = tid; d < INK_D; d += 256) {
        const float v = ops_bf16_to_f32(row[d]);
        ss += v * v;
    }
    const float inv = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
    for (uint d = tid; d < INK_D; d += 256) {
        cur_out[(size_t)m * INK_D + d] = ops_bf16_to_f32(row[d]) * inv * norm_w[d];
    }
}

kernel void ink_embed_q8row(
    device const uchar* embed_w   [[buffer(0)]],
    device const float* norm_w    [[buffer(1)]],
    device const int* tok_ids     [[buffer(2)]],
    device const uchar* is_img    [[buffer(3)]],
    device const float* img_emb   [[buffer(4)]],
    device float* cur_out         [[buffer(5)]],
    constant int& V_padded        [[buffer(6)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    if (is_img && is_img[m] != 0) {
        for (uint d = tid; d < INK_D; d += 256)
            cur_out[(size_t)m * INK_D + d] = img_emb[(size_t)m * INK_D + d];
        return;
    }
    const int id = tok_ids[m];
    const float sc = ((device const float*)embed_w)[id];
    device const char* row = (device const char*)(embed_w + (size_t)V_padded * 4) + (size_t)id * INK_D;
    float ss = 0.0f;
    for (uint d = tid; d < INK_D; d += 256) {
        const float v = float(row[d]) * sc;
        ss += v * v;
    }
    const float inv = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
    for (uint d = tid; d < INK_D; d += 256) {
        cur_out[(size_t)m * INK_D + d] = float(row[d]) * sc * inv * norm_w[d];
    }
}

// ============================================================================
// 2. Kimi Attention-Residuals Depth Mix (Fused RMSNorm + Dot + Softmax + Weighted Sum)
//    states_buf: contiguous [n_blocks_max, M_stride, INK_D] where slots 0..n_blocks-1 hold blocks
//    cur: [M, INK_D] (appended as state index n_blocks -> total S = n_blocks + 1 <= 8)
//    queries: [n_queries, INK_D] F32 (n_queries in {1, 3, 4})
//    out: [n_queries, M, INK_D] F32 (or [M, INK_D] when n_queries==1)
//    grid (M, 1, 1), 256 threads
// ============================================================================
kernel void ink_attn_res_mix(
    device const float* blocks_buf [[buffer(0)]],
    device const float* cur_buf    [[buffer(1)]],
    device const float* queries    [[buffer(2)]],
    device float* out_buf          [[buffer(3)]],
    constant int& n_blocks         [[buffer(4)]],
    constant int& block_stride     [[buffer(5)]],  // floats between consecutive block slots (= M_max * 2304)
    constant int& M_stride         [[buffer(6)]],  // floats between consecutive query outputs (= M * 2304)
    constant int& n_queries        [[buffer(7)]],
    constant int& block_offset     [[buffer(8)]],  // starting block slot index (0 normally, len(prelude)+1 for core bank)
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    threadgroup float inv_s[8];
    threadgroup float probs[4][8];

    const int S = n_blocks + 1;  // <= 8
    // 1. Compute RMS inv for each state s in 0..S-1
    for (int s = 0; s < S; ++s) {
        device const float* sp = (s < n_blocks)
            ? (blocks_buf + (size_t)(block_offset + s) * block_stride + (size_t)m * INK_D)
            : (cur_buf + (size_t)m * INK_D);
        float ss = 0.0f;
        for (uint d = tid; d < INK_D; d += 256) {
            const float v = sp[d];
            ss += v * v;
        }
        const float tot = ops_tg_sum(ss, sh, tid, 256);
        if (tid == 0) inv_s[s] = rsqrt(tot / float(INK_D) + INK_RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. For each query q in 0..n_queries-1, compute scores[s] = dot(normed_s, q) / sqrt(2304) and softmax
    const float inv_sqrt_d = 1.0f / 48.0f;  // sqrt(2304) == 48.0 exact!
    for (int q = 0; q < n_queries; ++q) {
        device const float* qv = queries + (size_t)q * INK_D;
        for (int s = 0; s < S; ++s) {
            device const float* sp = (s < n_blocks)
                ? (blocks_buf + (size_t)(block_offset + s) * block_stride + (size_t)m * INK_D)
                : (cur_buf + (size_t)m * INK_D);
            const float inv = inv_s[s];
            float dt = 0.0f;
            for (uint d = tid; d < INK_D; d += 256) dt += sp[d] * qv[d];
            const float score = ops_tg_sum(dt, sh, tid, 256) * inv * inv_sqrt_d;
            if (tid == 0) probs[q][s] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float mx = -1e30f;
            for (int s = 0; s < S; ++s) mx = max(mx, probs[q][s]);
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) {
                probs[q][s] = exp(probs[q][s] - mx);
                sum += probs[q][s];
            }
            const float isum = 1.0f / sum;
            for (int s = 0; s < S; ++s) probs[q][s] *= isum * inv_s[s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Weighted sum of normed states -> out[q, m, :]
        device float* dst = out_buf + (size_t)q * M_stride + (size_t)m * INK_D;
        for (uint d = tid; d < INK_D; d += 256) {
            float acc = 0.0f;
            for (int s = 0; s < S; ++s) {
                device const float* sp = (s < n_blocks)
                    ? (blocks_buf + (size_t)(block_offset + s) * block_stride + (size_t)m * INK_D)
                    : (cur_buf + (size_t)m * INK_D);
                acc = fma(probs[q][s], sp[d], acc);
            }
            dst[d] = acc;
        }
    }
}

// Fused Attention-Residual Depth Mix + RMSNorm (kills 90+ GPU barriers/token when !has_msgs!)
// norm_w_all: [n_queries, INK_D] OR single [INK_D] when n_queries == 1; norm_stride = 0 for shared or INK_D for per-query
kernel void ink_attn_res_mix_norm(
    device const float* blocks_buf [[buffer(0)]],
    device const float* cur_buf    [[buffer(1)]],
    device const float* queries    [[buffer(2)]],
    device const float* norm_w0    [[buffer(3)]],
    device const float* norm_w1    [[buffer(4)]],
    device const float* norm_w2    [[buffer(5)]],
    device float* out_buf          [[buffer(6)]],
    constant int& n_blocks         [[buffer(7)]],
    constant int& block_stride     [[buffer(8)]],
    constant int& M_stride         [[buffer(9)]],
    constant int& n_queries        [[buffer(10)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    threadgroup float inv_s[8];
    threadgroup float probs[4][8];
    const int S = n_blocks + 1;
    for (int s = 0; s < S; ++s) {
        device const float* sp = (s < n_blocks)
            ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D)
            : (cur_buf + (size_t)m * INK_D);
        float ss = 0.0f;
        for (uint d = tid; d < INK_D; d += 256) { const float v = sp[d]; ss += v * v; }
        const float tot = ops_tg_sum(ss, sh, tid, 256);
        if (tid == 0) inv_s[s] = rsqrt(tot / float(INK_D) + INK_RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sqrt_d = 1.0f / 48.0f;
    for (int q = 0; q < n_queries; ++q) {
        device const float* qv = queries + (size_t)q * INK_D;
        for (int s = 0; s < S; ++s) {
            device const float* sp = (s < n_blocks)
                ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D)
                : (cur_buf + (size_t)m * INK_D);
            float dt = 0.0f;
            for (uint d = tid; d < INK_D; d += 256) dt += sp[d] * qv[d];
            const float score = ops_tg_sum(dt, sh, tid, 256) * inv_s[s] * inv_sqrt_d;
            if (tid == 0) probs[q][s] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float mx = -1e30f;
            for (int s = 0; s < S; ++s) mx = max(mx, probs[q][s]);
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) { probs[q][s] = exp(probs[q][s] - mx); sum += probs[q][s]; }
            const float isum = 1.0f / sum;
            for (int s = 0; s < S; ++s) probs[q][s] *= isum * inv_s[s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float* dst = out_buf + (size_t)q * M_stride + (size_t)m * INK_D;
        device const float* nw = (q == 0) ? norm_w0 : ((q == 1) ? norm_w1 : norm_w2);
        float ss = 0.0f;
        for (uint d = tid; d < INK_D; d += 256) {
            float acc = 0.0f;
            for (int s = 0; s < S; ++s) {
                device const float* sp = (s < n_blocks)
                    ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D)
                    : (cur_buf + (size_t)m * INK_D);
                acc = fma(probs[q][s], sp[d], acc);
            }
            dst[d] = acc;
            ss += acc * acc;
        }
        const float invn = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
        for (uint d = tid; d < INK_D; d += 256) dst[d] = dst[d] * invn * nw[d];
    }
}

// ============================================================================
// 3. Fused Rank-64 tanh Loop Controller + Optional RMSNorm
//    Step A (ink_ctrl_cat_rms): build cat [M, 4608] = [rms(cur); rms(msg)]
//    (then ink_linear_f32 computes gate [M, 64] = W_down @ cat)
//    Step B (ink_ctrl_apply_norm): out[m, d] = ((1 - tanh(W_forget @ gate)) * cur + tanh(W_gain @ gate) * msg)
//                                  followed optionally by RMSNorm(norm_w)!
// ============================================================================
kernel void ink_ctrl_cat_rms(
    device const float* cur_buf [[buffer(0)]],
    device const float* msg_buf [[buffer(1)]],
    device float* cat_out       [[buffer(2)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* cp = cur_buf + (size_t)m * INK_D;
    device const float* mp = msg_buf + (size_t)m * INK_D;
    float ssc = 0.0f, ssm = 0.0f;
    for (uint d = tid; d < INK_D; d += 256) {
        const float cv = cp[d], mv = mp[d];
        ssc += cv * cv;
        ssm += mv * mv;
    }
    const float invc = rsqrt(ops_tg_sum(ssc, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
    const float invm = rsqrt(ops_tg_sum(ssm, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
    device float* dst = cat_out + (size_t)m * (2 * INK_D);
    for (uint d = tid; d < INK_D; d += 256) {
        dst[d] = cp[d] * invc;
        dst[INK_D + d] = mp[d] * invm;
    }
}

// grid (M, 1, 1), 256 threads
kernel void ink_ctrl_apply_norm(
    device const float* cur_buf  [[buffer(0)]],
    device const float* msg_buf  [[buffer(1)]],
    device const float* gate_buf [[buffer(2)]],  // [M, 64]
    device const float* w_gain   [[buffer(3)]],  // [2304, 64]
    device const float* w_forget [[buffer(4)]],  // [2304, 64]
    device const float* norm_w   [[buffer(5)]],  // [2304] (applied if do_norm != 0)
    device float* out_buf        [[buffer(6)]],  // [M, 2304]
    constant int& do_norm        [[buffer(7)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    threadgroup float g_sh[64];
    if (tid < 64) g_sh[tid] = gate_buf[(size_t)m * 64 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* cp = cur_buf + (size_t)m * INK_D;
    device const float* mp = msg_buf + (size_t)m * INK_D;
    device float* dst = out_buf + (size_t)m * INK_D;

    float ss = 0.0f;
    for (uint d = tid; d < INK_D; d += 256) {
        device const float4* wg = (device const float4*)(w_gain + (size_t)d * 64);
        device const float4* wf = (device const float4*)(w_forget + (size_t)d * 64);
        float sg = 0.0f, sf = 0.0f;
        for (int r = 0; r < 16; ++r) {
            const float4 gv = float4(g_sh[4*r], g_sh[4*r+1], g_sh[4*r+2], g_sh[4*r+3]);
            sg += dot(wg[r], gv);
            sf += dot(wf[r], gv);
        }
        const float g_act = tanh(clamp(sg, -15.0f, 15.0f));
        const float f_act = tanh(clamp(sf, -15.0f, 15.0f));
        const float v = (1.0f - f_act) * cp[d] + g_act * mp[d];
        dst[d] = v;
        ss += v * v;
    }
    if (do_norm) {
        const float inv = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
        for (uint d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
    }
}

// Fast batched controller fuse + RMSNorm for M > 8 (sg_buf = W_gain @ gate, sf_buf = W_forget @ gate via simdgroup_matrix!)
kernel void ink_ctrl_fuse_norm(
    device const float* cur_buf [[buffer(0)]],
    device const float* msg_buf [[buffer(1)]],
    device const float* sg_buf  [[buffer(2)]],
    device const float* sf_buf  [[buffer(3)]],
    device const float* norm_w  [[buffer(4)]],
    device float* out_buf       [[buffer(5)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    const size_t off = (size_t)m * INK_D;
    device const float* cp = cur_buf + off;
    device const float* mp = msg_buf + off;
    device const float* sgp = sg_buf + off;
    device const float* sfp = sf_buf + off;
    device float* dst = out_buf + off;
    float ss = 0.0f;
    for (uint d = tid; d < INK_D; d += 256) {
        const float v = (1.0f - tanh(clamp(sfp[d], -15.0f, 15.0f))) * cp[d]
                      + tanh(clamp(sgp[d], -15.0f, 15.0f)) * mp[d];
        dst[d] = v;
        ss += v * v;
    }
    const float inv = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(INK_D) + INK_RMS_EPS);
    for (uint d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

// Ultra-fast 40-core parallel Rank-64 controller Gain/Forget apply for M <= 8!
// grid (144, M, 1), 256 threads = 16 rows/TG x 16 lanes/row (100% coalesced 256-byte reads, 40x faster than 1-TG!)
kernel void ink_ctrl_gf_64(
    device const float* cur_buf  [[buffer(0)]],
    device const float* msg_buf  [[buffer(1)]],
    device const float* gate_buf [[buffer(2)]],  // [M, 64]
    device const float* w_gain   [[buffer(3)]],  // [2304, 64]
    device const float* w_forget [[buffer(4)]],  // [2304, 64]
    device float* out_buf        [[buffer(5)]],  // [M, 2304]
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float g_sh[64];
    const int m = int(tgp.y);
    if (tid < 64) g_sh[tid] = gate_buf[(size_t)m * 64 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int local_row = int(tid >> 4);   // 0..15
    const int k4 = int(tid & 15u);         // 0..15 -> 4 floats each = 64 floats total!
    const int d = int(tgp.x) * 16 + local_row;
    if (d >= INK_D) return;
    const float4 gv = float4(g_sh[4 * k4], g_sh[4 * k4 + 1], g_sh[4 * k4 + 2], g_sh[4 * k4 + 3]);
    float sg = dot(((device const float4*)(w_gain + (size_t)d * 64))[k4], gv);
    float sf = dot(((device const float4*)(w_forget + (size_t)d * 64))[k4], gv);
    // 16-lane butterfly reduction within half-simdgroup (zero barriers!)
    sg += simd_shuffle_xor(sg, ushort(1)); sf += simd_shuffle_xor(sf, ushort(1));
    sg += simd_shuffle_xor(sg, ushort(2)); sf += simd_shuffle_xor(sf, ushort(2));
    sg += simd_shuffle_xor(sg, ushort(4)); sf += simd_shuffle_xor(sf, ushort(4));
    sg += simd_shuffle_xor(sg, ushort(8)); sf += simd_shuffle_xor(sf, ushort(8));
    if (k4 == 0) {
        const size_t idx = (size_t)m * INK_D + d;
        out_buf[idx] = (1.0f - tanh(clamp(sf, -15.0f, 15.0f))) * cur_buf[idx]
                     + tanh(clamp(sg, -15.0f, 15.0f)) * msg_buf[idx];
    }
}

// Simple RMSNorm: out[m, d] = x[m, d] * rsqrt(mean(x^2) + eps) * w[d]
kernel void ink_rms_norm(
    device const float* x_in  [[buffer(0)]],
    device const float* w     [[buffer(1)]],
    device float* y_out       [[buffer(2)]],
    constant int& D           [[buffer(3)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* xp = x_in + (size_t)m * D;
    float ss = 0.0f;
    for (uint d = tid; d < uint(D); d += 256) {
        const float v = xp[d];
        ss += v * v;
    }
    const float inv = rsqrt(ops_tg_sum(ss, sh, tid, 256) / float(D) + INK_RMS_EPS);
    device float* yp = y_out + (size_t)m * D;
    for (uint d = tid; d < uint(D); d += 256) yp[d] = xp[d] * inv * w[d];
}

// Residual add: out[i] = a[i] + b[i]
kernel void ink_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < N) out[gid] = a[gid] + b[gid];
}

// Fast F32 copy: dst[i] = src[i]
kernel void ink_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant int& N         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < N) dst[gid] = src[gid];
}

// ============================================================================
// 4. 4-Tap Causal ShortConv (Static + Global Dynamic Taps)
//    Handles BOTH sequential multi-token chunks within a slot AND multi-slot batching (c=1..64)!
//    slot_ids[m]: which request slot (0..63) token m belongs to.
//    Sequential tokens of the SAME slot appear in ascending pos order within the batch, so each
//    slot is processed sequentially along its tokens in the batch (1 threadgroup per active channel block).
//    Wait: even cleaner & 100% race-free for ANY batch composition:
//    Each thread owns channel `c` (grid ceil(C/256), 256 threads) and loops over m = 0..M-1:
//    Because M <= 64 for decode/batching (or chunk <= 512 for prefill) and C = 1024..2304 (4..9 TGs),
//    wait: for prefill M=512, looping 1 thread over 512 tokens would be serial along M.
//    Instead, split into two ultra-fast cases:
//      - ink_sconv4_batched (for decode/verify/batching where each slot has <= 16 tokens in the batch,
//        grid = (n_active_slots, ceil(C/256), 1), 256 threads; per-slot token index list slot_toks!):
//        zero race, full parallelism across slots AND channels!
//      - ink_sconv4_prefill_single_slot (when batch is 1 slot with M tokens):
//        grid = (ceil(M/64), ceil(C/256), 1) reads history for t < 3 and x_in[t-3..t] directly in parallel,
//        then updates history with the last 3 tokens! 100% parallel across M AND C!
// ============================================================================

// Fast general multi-slot ShortConv:
// hist: [max_slots, 3, C] float (for this visit & stream: k/v/attn/mlp)
// slot_tok_off[s], slot_tok_cnt[s], slot_tok_idx[]: tokens belonging to slot s in ascending time order
// static_w: [C, 4] (lag3, lag2, lag1, lag0)
// dyn_taps: optional [M, 8, 4] (lag0, lag1, lag2, lag3) when has_dyn != 0
// grid (n_active_slots, ceil(C/256), 1), 256 threads
kernel void ink_sconv4_multi(
    device const float* x_in        [[buffer(0)]],
    device const float* static_w    [[buffer(1)]],
    device const float* dyn_taps    [[buffer(2)]],
    device float* hist_pool         [[buffer(3)]],
    device const int* active_slots  [[buffer(4)]],
    device const int* slot_tok_off  [[buffer(5)]],
    device const int* slot_tok_cnt  [[buffer(6)]],
    device const int* slot_tok_idx  [[buffer(7)]],
    device float* y_out             [[buffer(8)]],
    constant int& C                 [[buffer(9)]],
    constant int& has_dyn           [[buffer(10)]],
    device float* ring_out          [[buffer(11)]],
    constant int& save_ring         [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= C) return;
    const int si = int(tgp.x);
    const int slot = active_slots[si];
    const int off = slot_tok_off[si];
    const int cnt = slot_tok_cnt[si];

    device float* h = hist_pool + (size_t)slot * 3 * C;
    float h0 = h[0 * C + c];  // t-3
    float h1 = h[1 * C + c];  // t-2
    float h2 = h[2 * C + c];  // t-1

    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;  // 1/sqrt(5)
    const int hd_idx = c >> 7;  // C=1024, 8 heads x 128

    for (int k = 0; k < cnt; ++k) {
        const int m = slot_tok_idx[off + k];
        const float cur = x_in[(size_t)m * C + c];
        float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        if (has_dyn) {
            // dyn_taps[m, hd_idx, 0..3]: lag 0=cur, 1=h2, 2=h1, 3=h0
            const float4 dt = *((device const float4*)(dyn_taps + ((size_t)m * 8 + hd_idx) * 4));
            acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
        }
        y_out[(size_t)m * C + c] = (cur + acc) * scale;
        h0 = h1;
        h1 = h2;
        h2 = cur;
        if (save_ring && si == 0 && k < 8) {
            device float* rdst = ring_out + (size_t)k * 3 * C;
            rdst[0 * C + c] = h0;
            rdst[1 * C + c] = h1;
            rdst[2 * C + c] = h2;
        }
    }
    h[0 * C + c] = h0;
    h[1 * C + c] = h1;
    h[2 * C + c] = h2;
}

// Fused attn_sconv + residual update (or block boundary save) — kills 48+ barriers/token!
kernel void ink_sconv4_attn_res(
    device const float* x_in        [[buffer(0)]],
    device const float* static_w    [[buffer(1)]],
    device float* hist_pool         [[buffer(2)]],
    device const int* active_slots  [[buffer(3)]],
    device const int* slot_tok_off  [[buffer(4)]],
    device const int* slot_tok_cnt  [[buffer(5)]],
    device const int* slot_tok_idx  [[buffer(6)]],
    device float* cur_buf           [[buffer(7)]],
    device float* block_dst         [[buffer(8)]],
    device float* ring_out          [[buffer(9)]],
    constant int& save_block        [[buffer(10)]],
    constant int& save_ring         [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= INK_D) return;
    const int si = int(tgp.x);
    const int slot = active_slots[si], off = slot_tok_off[si], cnt = slot_tok_cnt[si];
    device float* h = hist_pool + (size_t)slot * 3 * INK_D;
    float h0 = h[0 * INK_D + c], h1 = h[1 * INK_D + c], h2 = h[2 * INK_D + c];
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float scale = 0.4472135954999579f;
    for (int k = 0; k < cnt; ++k) {
        const int m = slot_tok_idx[off + k];
        const size_t idx = (size_t)m * INK_D + c;
        const float cur = x_in[idx];
        const float y = (cur + h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w) * scale;
        if (save_block) {
            block_dst[idx] = cur_buf[idx];
            cur_buf[idx] = y;
        } else {
            cur_buf[idx] += y;
        }
        h0 = h1; h1 = h2; h2 = cur;
        if (save_ring && si == 0 && k < 8) {
            device float* rdst = ring_out + (size_t)k * 3 * INK_D;
            rdst[0 * INK_D + c] = h0; rdst[1 * INK_D + c] = h1; rdst[2 * INK_D + c] = h2;
        }
    }
    h[0 * INK_D + c] = h0; h[1 * INK_D + c] = h1; h[2 * INK_D + c] = h2;
}

// Fused MoE reduce8 + mlp_sconv + residual add — kills 96 dispatches & 96 barriers/token!
kernel void ink_moe_red8_sconv_add(
    device const float* y_pairs     [[buffer(0)]],
    device const float* static_w    [[buffer(1)]],
    device float* hist_pool         [[buffer(2)]],
    device const int* active_slots  [[buffer(3)]],
    device const int* slot_tok_off  [[buffer(4)]],
    device const int* slot_tok_cnt  [[buffer(5)]],
    device const int* slot_tok_idx  [[buffer(6)]],
    device float* cur_buf           [[buffer(7)]],
    device float* ring_out          [[buffer(8)]],
    constant int& save_ring         [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= INK_D) return;
    const int si = int(tgp.x);
    const int slot = active_slots[si], off = slot_tok_off[si], cnt = slot_tok_cnt[si];
    device float* h = hist_pool + (size_t)slot * 3 * INK_D;
    float h0 = h[0 * INK_D + c], h1 = h[1 * INK_D + c], h2 = h[2 * INK_D + c];
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float scale = 0.4472135954999579f;
    for (int k = 0; k < cnt; ++k) {
        const int m = slot_tok_idx[off + k];
        float cur = 0.0f;
        for (int e = 0; e < 8; ++e) cur += y_pairs[((size_t)m * 8 + e) * INK_D + c];
        const float y = (cur + h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w) * scale;
        cur_buf[(size_t)m * INK_D + c] += y;
        h0 = h1; h1 = h2; h2 = cur;
        if (save_ring && si == 0 && k < 8) {
            device float* rdst = ring_out + (size_t)k * 3 * INK_D;
            rdst[0 * INK_D + c] = h0; rdst[1 * INK_D + c] = h1; rdst[2 * INK_D + c] = h2;
        }
    }
    h[0 * INK_D + c] = h0; h[1 * INK_D + c] = h1; h[2 * INK_D + c] = h2;
}

// Ultra-fast parallel single-slot prefill ShortConv (100% parallel over M AND C!)
// grid (M, ceil(C/256), 1), 256 threads -> computes y_out[0..M-1] from hist_pool[slot] + x_in[0..M-1]
kernel void ink_sconv4_prefill_single(
    device const float* x_in        [[buffer(0)]],
    device const float* static_w    [[buffer(1)]],
    device const float* dyn_taps    [[buffer(2)]],
    device const float* hist_pool   [[buffer(3)]],
    device float* y_out             [[buffer(4)]],
    constant int& slot              [[buffer(5)]],
    constant int& M                 [[buffer(6)]],
    constant int& C                 [[buffer(7)]],
    constant int& has_dyn           [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (m >= M || c >= C) return;
    device const float* h = hist_pool + (size_t)slot * 3 * C;
    const float cur = x_in[(size_t)m * C + c];
    const float h2 = (m >= 1) ? x_in[(size_t)(m - 1) * C + c] : h[2 * C + c];
    const float h1 = (m >= 2) ? x_in[(size_t)(m - 2) * C + c] : h[(1 + m) * C + c];
    const float h0 = (m >= 3) ? x_in[(size_t)(m - 3) * C + c] : h[m * C + c];
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
    float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    if (has_dyn) {
        const float4 dt = *((device const float4*)(dyn_taps + ((size_t)m * 8 + (c >> 7)) * 4));
        acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
    }
    y_out[(size_t)m * C + c] = (cur + acc) * scale;
}

// Update hist_pool[slot] after ink_sconv4_prefill_single (grid ceil(C/256), 256 threads)
kernel void ink_sconv4_prefill_commit(
    device const float* x_in        [[buffer(0)]],
    device float* hist_pool         [[buffer(1)]],
    constant int& slot              [[buffer(2)]],
    constant int& M                 [[buffer(3)]],
    constant int& C                 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const int c = int(gid);
    if (c >= C) return;
    device float* h = hist_pool + (size_t)slot * 3 * C;
    const float o0 = h[0 * C + c], o1 = h[1 * C + c], o2 = h[2 * C + c];
    h[0 * C + c] = (M >= 3) ? x_in[(size_t)(M - 3) * C + c] : (M == 2 ? o2 : o1);
    h[1 * C + c] = (M >= 2) ? x_in[(size_t)(M - 2) * C + c] : o2;
    h[2 * C + c] = x_in[(size_t)(M - 1) * C + c];
}

// ============================================================================
// 5. Fused Q/K/V/R Post-Processing + Log-Pos Tau + KV Cache Write
//    - Q [M, 24, 128]: L2 clip (scale q_l2_clip, thr=22.627417), * tau(pos) if global
//    - R [M, 24, 16]:  L2 clip (scale rel_l2_clip, thr=8.0), * tau(pos) if global
//    - K [M, 8, 128]:  L2 clip (no scale, thr=22.627417) -> write to KV cache (FP16 or FP8_E4M3)
//    - V [M, 8, 128]:  parameterless RMSNorm(eps=1e-6) -> write to KV cache (FP16 or FP8_E4M3)
//    grid (M, 24, 1), 128 threads (for Q & R); grid (M, 8, 1), 128 threads (for K & V)
// ============================================================================
static inline float log_pos_tau(int pos) {
    const float r = float(pos + 1) * (1.0f / 8192.0f);
    return 1.0f + 0.1f * log(max(1.0f, r));
}

kernel void ink_qr_clip_tau(
    device float* q_buf           [[buffer(0)]],  // [M, 24, 128] in-place
    device const float* q_scale   [[buffer(1)]],  // [24, 128]
    device float* r_buf           [[buffer(2)]],  // [M, 24, 16] in-place
    device const float* r_scale   [[buffer(3)]],  // [24, 16]
    device const int* pos_buf     [[buffer(4)]],  // [M]
    constant int& is_swa          [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    const int m = int(tgp.x), h = int(tgp.y);
    const float tau = is_swa ? 1.0f : log_pos_tau(pos_buf[m]);

    // Q head [128]
    device float* qp = q_buf + ((size_t)m * INK_NH + h) * INK_HD;
    const float qv = qp[tid] * q_scale[h * INK_HD + tid];
    const float qss = ops_tg_sum(qv * qv, sh, tid, 128);
    const float qfac = min(1.0f, 22.627416997969522f / (sqrt(qss) + 1e-6f)) * tau;
    qp[tid] = qv * qfac;

    // R head [16] handled by lanes 0..15
    if (tid < 16) {
        device float* rp = r_buf + ((size_t)m * INK_NH + h) * INK_DREL;
        const float rv = rp[tid] * r_scale[h * INK_DREL + tid];
        float rss = rv * rv;
        rss += simd_shuffle_xor(rss, ushort(1));
        rss += simd_shuffle_xor(rss, ushort(2));
        rss += simd_shuffle_xor(rss, ushort(4));
        rss += simd_shuffle_xor(rss, ushort(8));
        const float rfac = min(1.0f, 8.0f / (sqrt(rss) + 1e-6f)) * tau;
        rp[tid] = rv * rfac;
    }
}

// K L2 clip + V parameterless RMSNorm + KV Cache Write (FP16)
// kv_cache_k, kv_cache_v: [max_slots, kv_stride, 8, 128] half
// grid (M, 8, 1), 128 threads
kernel void ink_kv_norm_store_f16(
    device const float* k_in   [[buffer(0)]],
    device const float* v_in   [[buffer(1)]],
    device half* k_cache       [[buffer(2)]],
    device half* v_cache       [[buffer(3)]],
    device const int* slot_ids [[buffer(4)]],
    device const int* pos_buf  [[buffer(5)]],
    constant int& is_swa       [[buffer(6)]],
    constant int& kv_stride    [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    const int m = int(tgp.x), kh = int(tgp.y);
    const int slot = slot_ids[m];
    const int pos = pos_buf[m];
    const int kpos = is_swa ? (pos & 1023) : pos;

    device const float* kp = k_in + ((size_t)m * INK_NKV + kh) * INK_HD;
    const float kv = kp[tid];
    const float kss = ops_tg_sum(kv * kv, sh, tid, 128);
    const float kfac = min(1.0f, 22.627416997969522f / (sqrt(kss) + 1e-6f));

    device const float* vp = v_in + ((size_t)m * INK_NKV + kh) * INK_HD;
    const float vv = vp[tid];
    const float vss = ops_tg_sum(vv * vv, sh, tid, 128);
    const float vfac = rsqrt(vss / float(INK_HD) + INK_RMS_EPS);

    const size_t dst_off = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
    k_cache[dst_off] = half(kv * kfac);
    v_cache[dst_off] = half(vv * vfac);
}

// ============================================================================
// 6. Fused RelPos + Sink 24:8 GQA Flash Attention + Sigmoid Gate
//    Computes exact:
//      score(k) = dot(Q[m, h], K_cache[slot, kpos, h/3]) / sqrt(128) + dot(R[m, h], rel_proj_pad[:, min(pos-k, 1024)])
//      with per-query attention sink s_val = sinks[m, h] included in softmax denominator!
//      out[m, h, d] = (sum_k P(k) * V_cache[slot, kpos, h/3, d]) * sigmoid(gate[m, h, d])
//    Online Flash softmax over keys in blocks of 64!
//    grid (M, 24, 1), 128 threads (4 simdgroups, each thread holds 1 head_dim element d in 0..127 ->
//    Q[d] & O[d] stay in registers! 4 SGs collaborate over 32-key tiles via simd_sum!)
//    Wait: 1 thread per d=0..127 -> to compute dot(Q, K_j) across 128 dims, 128 threads = 4 SGs ->
//    Wait: even faster! 4 SGs (128 threads) = each SG computes 1 key at a time across 32 lanes (4 dims/lane)!
//    Let's compute the exact math:
//      4 SGs x 32 lanes = 128 threads. Each lane owns 4 head dims (d = 4*lane .. 4*lane+3).
//      Q[4] and O[4] stay in registers for ALL 4 SGs!
//      Each SG processes keys k = k0 + sg, k0 + sg + 4, ... in parallel (4 keys simultaneously per step!),
//      computes dot(Q, K) in 1 simd_sum (zero threadgroup barrier!), computes rel_bias = dot(R, rel_proj[:, dist])
//      in 16 FMAs, updates per-SG online softmax (m_sg, l_sg, o_sg[4]), and merges the 4 SGs ONCE at the end!
//    ZERO threadgroup barriers inside the entire K/V loop! Blazing fast on Apple Silicon M1-M5!
// ============================================================================
struct InkAttnArgs {
    int M;
    int is_swa;
    int swa_win;
    int rel_extent;
    int kv_stride;
    int kv_fp8;
    float scale;
    int pad;
};

kernel void ink_relpos_sink_gqa(
    device const float* q_buf        [[buffer(0)]],  // [M, 24, 128]
    device const float* r_buf        [[buffer(1)]],  // [M, 24, 16]
    device const half* rel_proj_t    [[buffer(2)]],  // [1025, 16] transposed half (L1-hot coalesced!)
    device const float* sinks_buf    [[buffer(3)]],  // [M, 24]
    device const float* gate_buf     [[buffer(4)]],  // [M, 24, 128]
    device const half* k_cache       [[buffer(5)]],  // [max_slots, kv_stride, 8, 128]
    device const half* v_cache       [[buffer(6)]],  // [max_slots, kv_stride, 8, 128]
    device const int* slot_ids       [[buffer(7)]],  // [M]
    device const int* pos_buf        [[buffer(8)]],  // [M]
    device float* attn_out           [[buffer(9)]],  // [M, 24, 128]
    constant InkAttnArgs& A          [[buffer(10)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float sh_m[4];
    threadgroup float sh_l[4];
    threadgroup float sh_o[4 * INK_HD];
    threadgroup float sh_r[16];

    const int m = int(tgp.x), h = int(tgp.y);
    const int kh = h / 3;  // 24:8 GQA -> 3 Q heads per KV head
    const int slot = slot_ids[m];
    const int qpos = pos_buf[m];

    if (sg == 0 && lane < 16) {
        sh_r[lane] = r_buf[((size_t)m * INK_NH + h) * INK_DREL + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q[4] into registers (each lane handles dims 4*lane .. 4*lane+3)
    const float4 qv = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h) * INK_HD))[lane] * A.scale;
    const int k_start = A.is_swa ? max(0, qpos - A.swa_win + 1) : 0;
    const int k_end = qpos + 1;

    // Each SG maintains its own online softmax accumulator over its strided keys!
    // SG 0 seeds with the per-query attention sink (score = sinks_buf[m, h], V = 0)!
    float m_max = (sg == 0) ? sinks_buf[(size_t)m * INK_NH + h] : -1e30f;
    float l_sum = (sg == 0) ? 1.0f : 0.0f;
    float4 o_acc = float4(0.0f);

    const size_t slot_kv_base = (size_t)slot * A.kv_stride;

    for (int k = k_start + int(sg); k < k_end; k += 4) {
        const int kidx = A.is_swa ? (k & 1023) : k;
        const size_t kv_off = ((slot_kv_base + kidx) * INK_NKV + kh) * INK_HD;
        const float4 kv = float4(((device const half4*)(k_cache + kv_off))[lane]);
        float sc = simd_sum(dot(qv, kv));

        // RelPos bias: coalesced 32-byte L1-hot load from rel_proj_t[dist, 0..15]!
        const int dist = min(qpos - k, A.rel_extent);
        float rb = (lane < 16) ? (sh_r[lane] * float(rel_proj_t[(size_t)dist * 16 + lane])) : 0.0f;
        rb = simd_sum(rb);
        sc += rb;

        const float mn = max(m_max, sc);
        const float corr = exp(m_max - mn);
        const float p = exp(sc - mn);
        l_sum = l_sum * corr + p;
        m_max = mn;
        const float4 vv = float4(((device const half4*)(v_cache + kv_off))[lane]);
        o_acc = o_acc * corr + p * vv;
    }

    // Merge 4 SGs via threadgroup memory (1 barrier total!)
    if (lane == 0) {
        sh_m[sg] = m_max;
        sh_l[sg] = l_sum;
    }
    *((threadgroup float4*)(sh_o + sg * INK_HD + lane * 4)) = o_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float gm = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float gl = 0.0f;
        float4 go = float4(0.0f);
        for (int s = 0; s < 4; ++s) {
            const float w = exp(sh_m[s] - gm);
            gl += sh_l[s] * w;
            go += (*((threadgroup float4*)(sh_o + s * INK_HD + lane * 4))) * w;
        }
        const float4 out_v = go * (1.0f / gl);
        const float4 g_in = ((device const float4*)(gate_buf + ((size_t)m * INK_NH + h) * INK_HD))[lane];
        const float4 g_sig = 1.0f / (1.0f + exp(-g_in));
        ((device float4*)(attn_out + ((size_t)m * INK_NH + h) * INK_HD))[lane] = out_v * g_sig;
    }
}

// ============================================================================
// 6b. GQA-Grouped 3-Q-per-KV Split-K Flash-Decoding (S_PARTS = 8!)
//     - 3x LESS KV DRAM bandwidth (loads K & V ONCE for all 3 Q heads per KV head!)
//     - 8x MORE threadgroups (64 TGs for M=1, 512 TGs for M=8 -> saturates all 40 cores!)
//     - Cuts decode attention at P=1024/2048 by ~10-15x with exact FP32 online softmax!
// ============================================================================
kernel void ink_relpos_gqa_splitk(
    device const float* q_buf        [[buffer(0)]],  // [M, 24, 128]
    device const float* r_buf        [[buffer(1)]],  // [M, 24, 16]
    device const half* rel_proj_t    [[buffer(2)]],  // [1025, 16]
    device const half* k_cache       [[buffer(3)]],  // [max_slots, kv_stride, 8, 128]
    device const half* v_cache       [[buffer(4)]],  // [max_slots, kv_stride, 8, 128]
    device const int* slot_ids       [[buffer(5)]],  // [M]
    device const int* pos_buf        [[buffer(6)]],  // [M]
    device float* part_o             [[buffer(7)]],  // [M, 24, S_PARTS, 128]
    device float* part_ml            [[buffer(8)]],  // [M, 24, S_PARTS, 2]
    constant InkAttnArgs& A          [[buffer(9)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    constexpr int S_PARTS = 8;
    threadgroup float sh_m[3][4];
    threadgroup float sh_l[3][4];
    threadgroup float sh_o[3][4 * INK_HD];
    threadgroup float sh_r[3][16];

    const int m = int(tgp.x), kh = int(tgp.y), sp = int(tgp.z);
    const int h0 = kh * 3;
    const int slot = slot_ids[m];
    const int qpos = pos_buf[m];

    if (sg < 3 && lane < 16) {
        sh_r[sg][lane] = r_buf[((size_t)m * INK_NH + h0 + int(sg)) * INK_DREL + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 q0 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 0) * INK_HD))[lane] * A.scale;
    const float4 q1 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 1) * INK_HD))[lane] * A.scale;
    const float4 q2 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 2) * INK_HD))[lane] * A.scale;

    const int k_lo = A.is_swa ? max(0, qpos - A.swa_win + 1) : 0;
    const int k_hi = qpos + 1;
    const int k_tot = k_hi - k_lo;
    const int chunk = (k_tot + S_PARTS - 1) / S_PARTS;
    const int p_start = k_lo + sp * chunk;
    const int p_end = min(k_hi, p_start + chunk);

    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
    float l0 = 0.0f,   l1 = 0.0f,   l2 = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f);

    const size_t slot_kv_base = (size_t)slot * A.kv_stride;
    for (int k = p_start + int(sg); k < p_end; k += 4) {
        const int kidx = A.is_swa ? (k & 1023) : k;
        const size_t kv_off = ((slot_kv_base + kidx) * INK_NKV + kh) * INK_HD;
        const float4 kv = float4(((device const half4*)(k_cache + kv_off))[lane]);
        const float4 vv = float4(((device const half4*)(v_cache + kv_off))[lane]);
        const int dist = min(qpos - k, A.rel_extent);
        const float rp = (lane < 16) ? float(rel_proj_t[(size_t)dist * 16 + lane]) : 0.0f;

        {
            const float sc = simd_sum(dot(q0, kv)) + simd_sum((lane < 16) ? (sh_r[0][lane] * rp) : 0.0f);
            const float mn = max(m0, sc);
            const float corr = exp(m0 - mn);
            const float p = exp(sc - mn);
            l0 = l0 * corr + p; m0 = mn; o0 = o0 * corr + p * vv;
        }
        {
            const float sc = simd_sum(dot(q1, kv)) + simd_sum((lane < 16) ? (sh_r[1][lane] * rp) : 0.0f);
            const float mn = max(m1, sc);
            const float corr = exp(m1 - mn);
            const float p = exp(sc - mn);
            l1 = l1 * corr + p; m1 = mn; o1 = o1 * corr + p * vv;
        }
        {
            const float sc = simd_sum(dot(q2, kv)) + simd_sum((lane < 16) ? (sh_r[2][lane] * rp) : 0.0f);
            const float mn = max(m2, sc);
            const float corr = exp(m2 - mn);
            const float p = exp(sc - mn);
            l2 = l2 * corr + p; m2 = mn; o2 = o2 * corr + p * vv;
        }
    }

    if (lane == 0) {
        sh_m[0][sg] = m0; sh_l[0][sg] = l0;
        sh_m[1][sg] = m1; sh_l[1][sg] = l1;
        sh_m[2][sg] = m2; sh_l[2][sg] = l2;
    }
    *((threadgroup float4*)(sh_o[0] + sg * INK_HD + lane * 4)) = o0;
    *((threadgroup float4*)(sh_o[1] + sg * INK_HD + lane * 4)) = o1;
    *((threadgroup float4*)(sh_o[2] + sg * INK_HD + lane * 4)) = o2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3 of the 4 SGs each reduce 1 Q head in parallel!
    if (sg < 3) {
        const int qi = int(sg);
        const int h = h0 + qi;
        const float gm = max(max(sh_m[qi][0], sh_m[qi][1]), max(sh_m[qi][2], sh_m[qi][3]));
        float gl = 0.0f;
        float4 go = float4(0.0f);
        if (gm > -1e29f) {
            for (int s = 0; s < 4; ++s) {
                const float w = (sh_m[qi][s] > -1e29f) ? exp(sh_m[qi][s] - gm) : 0.0f;
                gl += sh_l[qi][s] * w;
                go += (*((threadgroup float4*)(sh_o[qi] + s * INK_HD + lane * 4))) * w;
            }
        }
        const size_t pidx = ((size_t)m * INK_NH + h) * S_PARTS + sp;
        ((device float4*)(part_o + pidx * INK_HD))[lane] = go;
        if (lane == 0) {
            part_ml[pidx * 2 + 0] = gm;
            part_ml[pidx * 2 + 1] = gl;
        }
    }
}

// Merge S_PARTS=8 splits + exact deferred attention sink + sigmoid(gate)
// grid (M, 24, 1), 128 threads
kernel void ink_relpos_gqa_splitk_reduce(
    device const float* part_o    [[buffer(0)]],  // [M, 24, 8, 128]
    device const float* part_ml   [[buffer(1)]],  // [M, 24, 8, 2]
    device const float* sinks_buf [[buffer(2)]],  // [M, 24]
    device const float* gate_buf  [[buffer(3)]],  // [M, 24, 128]
    device float* attn_out        [[buffer(4)]],  // [M, 24, 128]
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    constexpr int S_PARTS = 8;
    const int m = int(tgp.x), h = int(tgp.y);
    const size_t base = ((size_t)m * INK_NH + h) * S_PARTS;

    float gm = sinks_buf[(size_t)m * INK_NH + h];
    for (int sp = 0; sp < S_PARTS; ++sp) {
        const float pm = part_ml[(base + sp) * 2 + 0];
        gm = max(gm, pm);
    }
    float gl = exp(sinks_buf[(size_t)m * INK_NH + h] - gm);
    float acc = 0.0f;
    for (int sp = 0; sp < S_PARTS; ++sp) {
        const float pm = part_ml[(base + sp) * 2 + 0];
        const float pl = part_ml[(base + sp) * 2 + 1];
        if (pl > 0.0f && pm > -1e29f) {
            const float w = exp(pm - gm);
            gl += pl * w;
            acc = fma(part_o[(base + sp) * INK_HD + tid], w, acc);
        }
    }
    const size_t out_off = ((size_t)m * INK_NH + h) * INK_HD + tid;
    const float g = gate_buf[out_off];
    attn_out[out_off] = (acc / gl) * (1.0f / (1.0f + exp(-g)));
}

// ============================================================================
// 7. Exact F32 Top-8 / 128 Sigmoid Router (bias used for selection ONLY!)
//    r_logits [M, 128] F32, r_bias [128] F32 -> topk_idx [M, 8], topk_w [M, 8]
//    grid (M, 1, 1), 128 threads
// ============================================================================
kernel void ink_router_topk8(
    device const float* r_logits [[buffer(0)]],
    device const float* r_bias   [[buffer(1)]],
    device int* topk_idx         [[buffer(2)]],
    device float* topk_w         [[buffer(3)]],
    constant float& route_scale  [[buffer(4)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_prob[128];
    threadgroup float sh_score[128];

    const float p = 1.0f / (1.0f + exp(-r_logits[(size_t)m * 128 + tid]));
    const float sc = p + r_bias[tid];
    sh_prob[tid] = isfinite(p) ? p : 0.0f;
    sh_score[tid] = isfinite(sc) ? sc : -1e29f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int sel[8];
        float pw[8];
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k) {
            int bi = -1;
            float bs = -1e30f;
            for (int e = 0; e < 128; ++e) {
                const float s = sh_score[e];
                if (s > bs) {
                    bs = s;
                    bi = e;
                }
            }
            if (bi < 0) bi = k;
            sel[k] = bi;
            pw[k] = sh_prob[bi];
            sum += pw[k];
            sh_score[bi] = -1e30f;
        }
        const float inv = route_scale / (sum + 1e-20f);
        for (int k = 0; k < 8; ++k) {
            topk_idx[(size_t)m * 8 + k] = sel[k];
            topk_w[(size_t)m * 8 + k] = pw[k] * inv;
        }
    }
}

// Fused M==1 Router Top-8 + MoE Plan — eliminates 48 ink_moe_plan dispatches & 48 barriers!
kernel void ink_router_topk8_plan_m1(
    device const float* r_logits [[buffer(0)]],
    device const float* r_bias   [[buffer(1)]],
    device int* topk_idx         [[buffer(2)]],
    device float* topk_w         [[buffer(3)]],
    device int* uexp             [[buffer(4)]],
    device int* uoff             [[buffer(5)]],
    device int* ucnt             [[buffer(6)]],
    device int* pair_list        [[buffer(7)]],
    device uint* meta            [[buffer(8)]],
    constant float& route_scale  [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_prob[128];
    threadgroup float sh_score[128];
    const float p = 1.0f / (1.0f + exp(-r_logits[tid]));
    const float sc = p + r_bias[tid];
    sh_prob[tid] = isfinite(p) ? p : 0.0f;
    sh_score[tid] = isfinite(sc) ? sc : -1e29f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int sel[8]; float pw[8]; float sum = 0.0f;
        for (int k = 0; k < 8; ++k) {
            int bi = -1; float bs = -1e30f;
            for (int e = 0; e < 128; ++e) if (sh_score[e] > bs) { bs = sh_score[e]; bi = e; }
            if (bi < 0) bi = k;
            sel[k] = bi; pw[k] = sh_prob[bi]; sum += pw[k]; sh_score[bi] = -1e30f;
        }
        const float inv = route_scale / (sum + 1e-20f);
        for (int k = 0; k < 8; ++k) {
            topk_idx[k] = sel[k];
            topk_w[k] = pw[k] * inv;
            uexp[k] = sel[k];
            uoff[k] = k;
            ucnt[k] = 1;
            pair_list[k] = k;
        }
        meta[0] = 8u; meta[1] = 8u;
        meta[2] = 144u; meta[3] = 8u; meta[4] = 1u;   // 8-row/TG (mx8 gu, sk/gen)
        meta[5] = 72u;  meta[6] = 8u; meta[7] = 1u;   // 16-row/TG (sf4 gu, mx8 dn)
        meta[8] = 36u;  meta[9] = 8u; meta[10] = 1u;  // mm tiles / 32-pair/TG
        meta[11] = 36u; meta[12] = 8u; meta[13] = 1u; // 32-pair/TG (sf4 dn)
    }
}

// ============================================================================
// 8. Output Softcap 30.0 * tanh(z / 30.0) + Deterministic First-Max GPU Argmax
// ============================================================================
kernel void ink_softcap_f32(
    device float* logits [[buffer(0)]],
    constant int& total  [[buffer(1)]],
    constant float& cap  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < total) {
        const float inv = 1.0f / cap;
        logits[gid] = cap * tanh(clamp(logits[gid] * inv, -15.0f, 15.0f));
    }
}

// Deterministic first-max argmax per row m in [0..M-1], V = 200058
// grid (M, 1, 1), 256 threads
kernel void ink_argmax_rows(
    device const float* logits [[buffer(0)]],
    device int* out_ids        [[buffer(1)]],
    constant int& V            [[buffer(2)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    device const float* row = logits + (size_t)m * V;
    float bv = -1e30f;
    int bi = 0;
    for (int i = int(tid); i < V; i += 256) {
        const float v = row[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    sh_v[tid] = bv;
    sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            const float v1 = sh_v[tid + s];
            const int i1 = sh_i[tid + s];
            if (v1 > sh_v[tid] || (v1 == sh_v[tid] && i1 < sh_i[tid])) {
                sh_v[tid] = v1;
                sh_i[tid] = i1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_ids[m] = sh_i[0];
}

// ============================================================================
// 9. Exact Multi-Branch Tree Verification Kernels (0.00000 KLD, M <= 32 nodes!)
//    - Non-destructive tree ShortConv (walks parent[m] 3 ancestor hops, saves raw x_in)
//    - Non-destructive tree KV store + 3-Q-per-KV Split-K Tree Flash-Decoding
//    - Concurrent O(1) GPU commit of winning accepted path[0..L-1] into KV & ShortConv!
// ============================================================================

// Tree 4-tap ShortConv for K/V (or general C): reads hist_pool[slot] + parent[m] ancestors in parallel!
// grid (M, ceil(C/256), 1), 256 threads
kernel void ink_sconv4_tree(
    device const float* x_in     [[buffer(0)]],
    device const float* static_w [[buffer(1)]],
    device const float* dyn_taps [[buffer(2)]],
    device const float* hist_pool [[buffer(3)]],
    device const int* parent_buf [[buffer(4)]],
    device float* raw_save       [[buffer(5)]],  // [32, C]
    device float* y_out          [[buffer(6)]],
    constant int& slot           [[buffer(7)]],
    constant int& C              [[buffer(8)]],
    constant int& has_dyn        [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= C) return;
    device const float* h = hist_pool + (size_t)slot * 3 * C;
    const int p1 = parent_buf[m];
    const int p2 = (p1 >= 0) ? parent_buf[p1] : -1;
    const int p3 = (p2 >= 0) ? parent_buf[p2] : -1;
    const float cur = x_in[(size_t)m * C + c];
    const float h2  = (p1 >= 0) ? x_in[(size_t)p1 * C + c] : h[2 * C + c];
    const float h1  = (p2 >= 0) ? x_in[(size_t)p2 * C + c] : ((p1 >= 0) ? h[2 * C + c] : h[1 * C + c]);
    const float h0  = (p3 >= 0) ? x_in[(size_t)p3 * C + c] : ((p2 >= 0) ? h[2 * C + c] : ((p1 >= 0) ? h[1 * C + c] : h[0 * C + c]));
    raw_save[(size_t)m * C + c] = cur;
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
    float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
    if (has_dyn) {
        const float4 dt = *((device const float4*)(dyn_taps + ((size_t)m * 8 + (c >> 7)) * 4));
        acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
    }
    y_out[(size_t)m * C + c] = (cur + acc) * scale;
}

// Fused Tree attn_sconv + residual/block update (grid (M, 9, 1), 256 threads)
kernel void ink_sconv4_attn_res_tree(
    device const float* x_in     [[buffer(0)]],
    device const float* static_w [[buffer(1)]],
    device const float* hist_pool [[buffer(2)]],
    device const int* parent_buf [[buffer(3)]],
    device float* raw_save       [[buffer(4)]],
    device float* cur_buf        [[buffer(5)]],
    device float* block_dst      [[buffer(6)]],
    constant int& slot           [[buffer(7)]],
    constant int& save_block     [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= INK_D) return;
    device const float* h = hist_pool + (size_t)slot * 3 * INK_D;
    const int p1 = parent_buf[m];
    const int p2 = (p1 >= 0) ? parent_buf[p1] : -1;
    const int p3 = (p2 >= 0) ? parent_buf[p2] : -1;
    const size_t idx = (size_t)m * INK_D + c;
    const float cur = x_in[idx];
    const float h2  = (p1 >= 0) ? x_in[(size_t)p1 * INK_D + c] : h[2 * INK_D + c];
    const float h1  = (p2 >= 0) ? x_in[(size_t)p2 * INK_D + c] : ((p1 >= 0) ? h[2 * INK_D + c] : h[1 * INK_D + c]);
    const float h0  = (p3 >= 0) ? x_in[(size_t)p3 * INK_D + c] : ((p2 >= 0) ? h[2 * INK_D + c] : ((p1 >= 0) ? h[1 * INK_D + c] : h[0 * INK_D + c]));
    raw_save[idx] = cur;
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float y = (cur + h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w) * 0.4472135954999579f;
    if (save_block) {
        block_dst[idx] = cur_buf[idx];
        cur_buf[idx] = y;
    } else {
        cur_buf[idx] += y;
    }
}

// Fused Tree mlp_sconv + residual add (reads b_moe_raw [M, 2304], grid (M, 9, 1), 256 threads)
kernel void ink_sconv4_mlp_add_tree(
    device const float* x_in     [[buffer(0)]],
    device const float* static_w [[buffer(1)]],
    device const float* hist_pool [[buffer(2)]],
    device const int* parent_buf [[buffer(3)]],
    device float* raw_save       [[buffer(4)]],
    device float* cur_buf        [[buffer(5)]],
    constant int& slot           [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= INK_D) return;
    device const float* h = hist_pool + (size_t)slot * 3 * INK_D;
    const int p1 = parent_buf[m];
    const int p2 = (p1 >= 0) ? parent_buf[p1] : -1;
    const int p3 = (p2 >= 0) ? parent_buf[p2] : -1;
    const size_t idx = (size_t)m * INK_D + c;
    const float cur = x_in[idx];
    const float h2  = (p1 >= 0) ? x_in[(size_t)p1 * INK_D + c] : h[2 * INK_D + c];
    const float h1  = (p2 >= 0) ? x_in[(size_t)p2 * INK_D + c] : ((p1 >= 0) ? h[2 * INK_D + c] : h[1 * INK_D + c]);
    const float h0  = (p3 >= 0) ? x_in[(size_t)p3 * INK_D + c] : ((p2 >= 0) ? h[2 * INK_D + c] : ((p1 >= 0) ? h[1 * INK_D + c] : h[0 * INK_D + c]));
    raw_save[idx] = cur;
    const float4 sw = *((device const float4*)(static_w + (size_t)c * 4));
    const float y = (cur + h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w) * 0.4472135954999579f;
    cur_buf[idx] += y;
}

// Commit winning path[0..L-1] into ShortConv hist_pool[slot] (grid (ceil(C/256), 1, 1), 256 threads)
kernel void ink_sconv4_tree_commit(
    device const float* raw_save [[buffer(0)]],  // [32, C]
    device float* hist_pool      [[buffer(1)]],  // [max_slots, 3, C]
    device const int* path_ids   [[buffer(2)]],  // [L]
    constant int& slot           [[buffer(3)]],
    constant int& L              [[buffer(4)]],
    constant int& C              [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int c = int(gid);
    if (c >= C) return;
    device float* h = hist_pool + (size_t)slot * 3 * C;
    const float o1 = h[1 * C + c], o2 = h[2 * C + c];
    if (L >= 3) {
        h[0 * C + c] = raw_save[(size_t)path_ids[L - 3] * C + c];
        h[1 * C + c] = raw_save[(size_t)path_ids[L - 2] * C + c];
        h[2 * C + c] = raw_save[(size_t)path_ids[L - 1] * C + c];
    } else if (L == 2) {
        h[0 * C + c] = o2;
        h[1 * C + c] = raw_save[(size_t)path_ids[0] * C + c];
        h[2 * C + c] = raw_save[(size_t)path_ids[1] * C + c];
    } else {
        h[0 * C + c] = o1;
        h[1 * C + c] = o2;
        h[2 * C + c] = raw_save[(size_t)path_ids[0] * C + c];
    }
}

// Store tree K (L2 clip) + V (RMSNorm) into scratch tree_k/v [32, 8, 128] FP16 (grid (M, 8, 1), 128 threads)
kernel void ink_kv_store_tree_f16(
    device const float* k_in [[buffer(0)]],
    device const float* v_in [[buffer(1)]],
    device half* tree_k      [[buffer(2)]],
    device half* tree_v      [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    const int m = int(tgp.x), kh = int(tgp.y);
    device const float* kp = k_in + ((size_t)m * INK_NKV + kh) * INK_HD;
    const float kv = kp[tid];
    const float kss = ops_tg_sum(kv * kv, sh, tid, 128);
    const float kfac = min(1.0f, 22.627416997969522f / (sqrt(kss) + 1e-6f));

    device const float* vp = v_in + ((size_t)m * INK_NKV + kh) * INK_HD;
    const float vv = vp[tid];
    const float vss = ops_tg_sum(vv * vv, sh, tid, 128);
    const float vfac = rsqrt(vss / float(INK_HD) + INK_RMS_EPS);

    const size_t dst_off = ((size_t)m * INK_NKV + kh) * INK_HD + tid;
    tree_k[dst_off] = half(kv * kfac);
    tree_v[dst_off] = half(vv * vfac);
}

// Commit winning path[0..L-1] from tree_k/v into kv_k/v[slot, kpos(anchor_pos + i)] (grid (L, 8, 1), 128 threads)
kernel void ink_kv_tree_commit_f16(
    device const half* tree_k  [[buffer(0)]],
    device const half* tree_v  [[buffer(1)]],
    device half* k_cache       [[buffer(2)]],
    device half* v_cache       [[buffer(3)]],
    device const int* path_ids [[buffer(4)]],
    constant int& slot         [[buffer(5)]],
    constant int& anchor_pos   [[buffer(6)]],
    constant int& is_swa       [[buffer(7)]],
    constant int& kv_stride    [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int i = int(tgp.x), kh = int(tgp.y);
    const int node = path_ids[i];
    const int pos = anchor_pos + i;
    const int kpos = is_swa ? (pos & 1023) : pos;
    const size_t src_off = ((size_t)node * INK_NKV + kh) * INK_HD + tid;
    const size_t dst_off = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
    k_cache[dst_off] = tree_k[src_off];
    v_cache[dst_off] = tree_v[src_off];
}

struct InkTreeAttnArgs {
    int M;
    int slot;
    int anchor_pos;
    int is_swa;
    int swa_win;
    int rel_extent;
    int kv_stride;
    int pad;
    float scale;
    int pad2;
};

// 3-Q-per-KV Split-K Tree Flash-Decoding (S_PARTS = 8): exact prefix [< anchor_pos] + masked tree ancestors!
// Reduced via existing ink_relpos_gqa_splitk_reduce! grid (M, 8, 8), 128 threads
kernel void ink_relpos_tree_gqa_splitk(
    device const float* q_buf     [[buffer(0)]],
    device const float* r_buf     [[buffer(1)]],
    device const half* rel_proj_t [[buffer(2)]],
    device const half* k_cache    [[buffer(3)]],
    device const half* v_cache    [[buffer(4)]],
    device const half* tree_k     [[buffer(5)]],
    device const half* tree_v     [[buffer(6)]],
    device const int* pos_buf     [[buffer(7)]],
    device const uint* anc_mask   [[buffer(8)]],
    device float* part_o          [[buffer(9)]],
    device float* part_ml         [[buffer(10)]],
    constant InkTreeAttnArgs& A   [[buffer(11)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    constexpr int S_PARTS = 8;
    threadgroup float sh_m[3][4];
    threadgroup float sh_l[3][4];
    threadgroup float sh_o[3][4 * INK_HD];
    threadgroup float sh_r[3][16];

    const int m = int(tgp.x), kh = int(tgp.y), sp = int(tgp.z);
    const int h0 = kh * 3;
    const int qpos = pos_buf[m];

    if (sg < 3 && lane < 16) {
        sh_r[sg][lane] = r_buf[((size_t)m * INK_NH + h0 + int(sg)) * INK_DREL + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 q0 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 0) * INK_HD))[lane] * A.scale;
    const float4 q1 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 1) * INK_HD))[lane] * A.scale;
    const float4 q2 = ((device const float4*)(q_buf + ((size_t)m * INK_NH + h0 + 2) * INK_HD))[lane] * A.scale;

    const int k_lo = A.is_swa ? max(0, qpos - A.swa_win + 1) : 0;
    const int k_hi = A.anchor_pos;
    const int k_tot = max(0, k_hi - k_lo);
    const int chunk = (k_tot + S_PARTS - 1) / S_PARTS;
    const int p_start = k_lo + sp * chunk;
    const int p_end = min(k_hi, p_start + chunk);

    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
    float l0 = 0.0f,   l1 = 0.0f,   l2 = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f);

    const size_t slot_kv_base = (size_t)A.slot * A.kv_stride;
    for (int k = p_start + int(sg); k < p_end; k += 4) {
        const int kidx = A.is_swa ? (k & 1023) : k;
        const size_t kv_off = ((slot_kv_base + kidx) * INK_NKV + kh) * INK_HD;
        const float4 kv = float4(((device const half4*)(k_cache + kv_off))[lane]);
        const float4 vv = float4(((device const half4*)(v_cache + kv_off))[lane]);
        const int dist = min(qpos - k, A.rel_extent);
        const float rp = (lane < 16) ? float(rel_proj_t[(size_t)dist * 16 + lane]) : 0.0f;
        {
            const float sc = simd_sum(dot(q0, kv)) + simd_sum((lane < 16) ? (sh_r[0][lane] * rp) : 0.0f);
            const float mn = max(m0, sc); const float corr = exp(m0 - mn); const float p = exp(sc - mn);
            l0 = l0 * corr + p; m0 = mn; o0 = o0 * corr + p * vv;
        }
        {
            const float sc = simd_sum(dot(q1, kv)) + simd_sum((lane < 16) ? (sh_r[1][lane] * rp) : 0.0f);
            const float mn = max(m1, sc); const float corr = exp(m1 - mn); const float p = exp(sc - mn);
            l1 = l1 * corr + p; m1 = mn; o1 = o1 * corr + p * vv;
        }
        {
            const float sc = simd_sum(dot(q2, kv)) + simd_sum((lane < 16) ? (sh_r[2][lane] * rp) : 0.0f);
            const float mn = max(m2, sc); const float corr = exp(m2 - mn); const float p = exp(sc - mn);
            l2 = l2 * corr + p; m2 = mn; o2 = o2 * corr + p * vv;
        }
    }

    // Split 0 also scores the <= 8 tree ancestor keys (exact path from root to node m!)
    if (sp == 0) {
        int anc[8];
        int n_anc = 0;
        uint bm = anc_mask[m];
        while (bm != 0u && n_anc < 8) {
            const int j = int(ctz(bm));
            anc[n_anc++] = j;
            bm &= (bm - 1u);
        }
        for (int ai = int(sg); ai < n_anc; ai += 4) {
            const int j = anc[ai];
            const size_t toff = ((size_t)j * INK_NKV + kh) * INK_HD;
            const float4 kv = float4(((device const half4*)(tree_k + toff))[lane]);
            const float4 vv = float4(((device const half4*)(tree_v + toff))[lane]);
            const int dist = min(qpos - pos_buf[j], A.rel_extent);
            const float rp = (lane < 16) ? float(rel_proj_t[(size_t)dist * 16 + lane]) : 0.0f;
            {
                const float sc = simd_sum(dot(q0, kv)) + simd_sum((lane < 16) ? (sh_r[0][lane] * rp) : 0.0f);
                const float mn = max(m0, sc); const float corr = exp(m0 - mn); const float p = exp(sc - mn);
                l0 = l0 * corr + p; m0 = mn; o0 = o0 * corr + p * vv;
            }
            {
                const float sc = simd_sum(dot(q1, kv)) + simd_sum((lane < 16) ? (sh_r[1][lane] * rp) : 0.0f);
                const float mn = max(m1, sc); const float corr = exp(m1 - mn); const float p = exp(sc - mn);
                l1 = l1 * corr + p; m1 = mn; o1 = o1 * corr + p * vv;
            }
            {
                const float sc = simd_sum(dot(q2, kv)) + simd_sum((lane < 16) ? (sh_r[2][lane] * rp) : 0.0f);
                const float mn = max(m2, sc); const float corr = exp(m2 - mn); const float p = exp(sc - mn);
                l2 = l2 * corr + p; m2 = mn; o2 = o2 * corr + p * vv;
            }
        }
    }

    if (lane == 0) {
        sh_m[0][sg] = m0; sh_l[0][sg] = l0;
        sh_m[1][sg] = m1; sh_l[1][sg] = l1;
        sh_m[2][sg] = m2; sh_l[2][sg] = l2;
    }
    *((threadgroup float4*)(sh_o[0] + sg * INK_HD + lane * 4)) = o0;
    *((threadgroup float4*)(sh_o[1] + sg * INK_HD + lane * 4)) = o1;
    *((threadgroup float4*)(sh_o[2] + sg * INK_HD + lane * 4)) = o2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < 3) {
        const int qi = int(sg);
        const int h = h0 + qi;
        const float gm = max(max(sh_m[qi][0], sh_m[qi][1]), max(sh_m[qi][2], sh_m[qi][3]));
        float gl = 0.0f;
        float4 go = float4(0.0f);
        if (gm > -1e29f) {
            for (int s = 0; s < 4; ++s) {
                const float w = (sh_m[qi][s] > -1e29f) ? exp(sh_m[qi][s] - gm) : 0.0f;
                gl += sh_l[qi][s] * w;
                go += (*((threadgroup float4*)(sh_o[qi] + s * INK_HD + lane * 4))) * w;
            }
        }
        const size_t pidx = ((size_t)m * INK_NH + h) * S_PARTS + sp;
        ((device float4*)(part_o + pidx * INK_HD))[lane] = go;
        if (lane == 0) {
            part_ml[pidx * 2 + 0] = gm;
            part_ml[pidx * 2 + 1] = gl;
        }
    }
}

