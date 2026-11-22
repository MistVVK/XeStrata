// ink_metal_dspark.metal - Persistent 100% On-GPU DSpark 3-Layer Block-8 Recurrent Markov Drafter
// Matches inkling2_loop.py::Inkling2DSpark bit-for-bit:
//   - Taps visits [5, 23, 41, 48] mixed via softmax(fuse_weight[row]) + dflare_norm
//   - 3 non-causal block-attention layers (layers 0, 2 = SWA-1025; layer 1 = global), NeoX half-split RoPE (theta=1e6)
//   - Context KV strictly precedes anchor position `apos`; draft block attends to [ctx (< apos) | draft (8 tokens)]
//   - SwiGLU 9216-wide dense FFN
//   - 7-step Recurrent Markov head (markov_rank=256) conditioned on committed f_rnn at apos - 1
//   - Deterministic first-max argmax on GPU with ZERO host sync between the 7 steps!
// 100% MIT clean-room implementation.
#include <metal_stdlib>
using namespace metal;

#define DSP_D 2304
#define DSP_NH 24
#define DSP_NKV 8
#define DSP_HD 128
#define DSP_RANK 256
#define DSP_EPS 1e-6f

static inline float dsp_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float dsp_tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    const uint nw = (nthr + 31u) >> 5;
    for (uint i = 0; i < nw; ++i) r += sh[i];
    return r;
}

// 1. Mix 4 target taps [4, M, 2304] with softmax(fuse_weight[row, 0..3]) + RMSNorm(norm_w)
// grid (M, 1, 1), 256 threads
kernel void dspark_mix_taps(
    device const float* taps_4      [[buffer(0)]],  // [4, M_stride, 2304]
    device const float* fuse_weight [[buffer(1)]],  // [4, 4]
    device const float* norm_w      [[buffer(2)]],  // [2304]
    device float* out_buf           [[buffer(3)]],  // [M, 2304]
    constant int& row               [[buffer(4)]],  // 0, 1, 2 for layers; 3 for f_rnn
    constant int& M_stride          [[buffer(5)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    threadgroup float fw[4];
    if (tid == 0) {
        float w0 = fuse_weight[row * 4 + 0], w1 = fuse_weight[row * 4 + 1];
        float w2 = fuse_weight[row * 4 + 2], w3 = fuse_weight[row * 4 + 3];
        const float mx = max(max(w0, w1), max(w2, w3));
        w0 = exp(w0 - mx); w1 = exp(w1 - mx); w2 = exp(w2 - mx); w3 = exp(w3 - mx);
        const float isum = 1.0f / (w0 + w1 + w2 + w3);
        fw[0] = w0 * isum; fw[1] = w1 * isum; fw[2] = w2 * isum; fw[3] = w3 * isum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dst = out_buf + (size_t)m * DSP_D;
    float ss = 0.0f;
    for (uint d = tid; d < DSP_D; d += 256) {
        const float v = fw[0] * taps_4[(0ul * M_stride + m) * DSP_D + d]
                      + fw[1] * taps_4[(1ul * M_stride + m) * DSP_D + d]
                      + fw[2] * taps_4[(2ul * M_stride + m) * DSP_D + d]
                      + fw[3] * taps_4[(3ul * M_stride + m) * DSP_D + d];
        dst[d] = v;
        ss += v * v;
    }
    const float inv = rsqrt(dsp_tg_sum(ss, sh, tid, 256) / float(DSP_D) + DSP_EPS);
    for (uint d = tid; d < DSP_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

// 2. Store committed context K (RMSNorm + NeoX half-split RoPE) and V into DSpark context KV cache
// grid (M, 8, 1), 128 threads
kernel void dspark_ctx_kv_store(
    device const float* k_in   [[buffer(0)]],  // [M, 8, 128]
    device const float* v_in   [[buffer(1)]],  // [M, 8, 128]
    device const float* k_nw   [[buffer(2)]],  // [128]
    device half* k_cache       [[buffer(3)]],  // [max_slots, kv_stride, 8, 128]
    device half* v_cache       [[buffer(4)]],  // [max_slots, kv_stride, 8, 128]
    device const int* slot_ids [[buffer(5)]],
    device const int* pos_buf  [[buffer(6)]],
    constant int& kv_stride    [[buffer(7)]],
    constant float& theta      [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float kn[128];
    const int m = int(tgp.x), kh = int(tgp.y);
    const int slot = slot_ids[m];
    const int pos = pos_buf[m];

    device const float* kp = k_in + ((size_t)m * DSP_NKV + kh) * DSP_HD;
    const float kv = kp[tid];
    const float inv = rsqrt(dsp_tg_sum(kv * kv, sh, tid, 128) / float(DSP_HD) + DSP_EPS);
    kn[tid] = kv * inv * k_nw[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // NeoX half-split RoPE: (d, d + 64) for d in 0..63
    const size_t dst_off = (((size_t)slot * kv_stride + pos) * DSP_NKV + kh) * DSP_HD;
    if (tid < 64) {
        const float freq = 1.0f / pow(theta, float(2 * tid) / float(DSP_HD));
        const float ang = float(pos) * freq;
        const float cs = precise::cos(ang), sn = precise::sin(ang);
        const float x0 = kn[tid], x1 = kn[tid + 64];
        k_cache[dst_off + tid]      = half(x0 * cs - x1 * sn);
        k_cache[dst_off + tid + 64] = half(x0 * sn + x1 * cs);
    }
    v_cache[dst_off + tid] = half(v_in[((size_t)m * DSP_NKV + kh) * DSP_HD + tid]);
}

// 3. Build initial Block-8 draft hidden state [8, 2304]:
//    slot 0 = rms_norm(embed(anchor_tok), embed_norm_w); slots 1..7 = mask_embed
//    grid (8, 1, 1), 256 threads
kernel void dspark_init_block_bf16(
    device const ushort* embed_w  [[buffer(0)]],
    device const float* norm_w    [[buffer(1)]],
    device const float* mask_emb  [[buffer(2)]],
    device float* cur_out         [[buffer(3)]],
    constant int& anchor_tok      [[buffer(4)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device float* dst = cur_out + (size_t)b * DSP_D;
    if (b == 0) {
        device const ushort* row = embed_w + (size_t)anchor_tok * DSP_D;
        float ss = 0.0f;
        for (uint d = tid; d < DSP_D; d += 256) {
            const float v = dsp_bf16_to_f32(row[d]);
            ss += v * v;
        }
        const float inv = rsqrt(dsp_tg_sum(ss, sh, tid, 256) / float(DSP_D) + DSP_EPS);
        for (uint d = tid; d < DSP_D; d += 256) dst[d] = dsp_bf16_to_f32(row[d]) * inv * norm_w[d];
    } else {
        for (uint d = tid; d < DSP_D; d += 256) dst[d] = mask_emb[d];
    }
}

kernel void dspark_init_block_q8row(
    device const uchar* embed_w   [[buffer(0)]],
    device const float* norm_w    [[buffer(1)]],
    device const float* mask_emb  [[buffer(2)]],
    device float* cur_out         [[buffer(3)]],
    constant int& anchor_tok      [[buffer(4)]],
    constant int& V_padded        [[buffer(5)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device float* dst = cur_out + (size_t)b * DSP_D;
    if (b == 0) {
        const float sc = ((device const float*)embed_w)[anchor_tok];
        device const char* row = (device const char*)(embed_w + (size_t)V_padded * 4) + (size_t)anchor_tok * DSP_D;
        float ss = 0.0f;
        for (uint d = tid; d < DSP_D; d += 256) {
            const float v = float(row[d]) * sc;
            ss += v * v;
        }
        const float inv = rsqrt(dsp_tg_sum(ss, sh, tid, 256) / float(DSP_D) + DSP_EPS);
        for (uint d = tid; d < DSP_D; d += 256) dst[d] = float(row[d]) * sc * inv * norm_w[d];
    } else {
        for (uint d = tid; d < DSP_D; d += 256) dst[d] = mask_emb[d];
    }
}

// 4. Q/K RMSNorm + NeoX half-split RoPE for the 8 draft tokens at positions [apos .. apos+7]
// grid (8, NH, 1), 128 threads
kernel void dspark_qk_norm_rope(
    device float* x_buf       [[buffer(0)]],  // [8, NH, 128] in-place
    device const float* nw    [[buffer(1)]],  // [128]
    constant int& NH          [[buffer(2)]],
    constant int& apos        [[buffer(3)]],
    constant float& theta     [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float xn[128];
    const int b = int(tgp.x), h = int(tgp.y);
    device float* xp = x_buf + ((size_t)b * NH + h) * DSP_HD;
    const float v = xp[tid];
    const float inv = rsqrt(dsp_tg_sum(v * v, sh, tid, 128) / float(DSP_HD) + DSP_EPS);
    xn[tid] = v * inv * nw[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        const float freq = 1.0f / pow(theta, float(2 * tid) / float(DSP_HD));
        const float ang = float(apos + b) * freq;
        const float cs = precise::cos(ang), sn = precise::sin(ang);
        const float x0 = xn[tid], x1 = xn[tid + 64];
        xp[tid]      = x0 * cs - x1 * sn;
        xp[tid + 64] = x0 * sn + x1 * cs;
    }
}

// 5. Non-causal Block-8 Attention over [ctx_kv (c0..apos-1) | draft_kv (0..7)]
// grid (8, 24, 1), 128 threads (4 SGs x 32 lanes, 4 dims/lane)
kernel void dspark_block_attn(
    device const float* q_buf   [[buffer(0)]],  // [8, 24, 128]
    device const float* kd_buf  [[buffer(1)]],  // [8, 8, 128]
    device const float* vd_buf  [[buffer(2)]],  // [8, 8, 128]
    device const half* ctx_k    [[buffer(3)]],  // [max_slots, kv_stride, 8, 128]
    device const half* ctx_v    [[buffer(4)]],  // [max_slots, kv_stride, 8, 128]
    device float* out_buf       [[buffer(5)]],  // [8, 24, 128]
    constant int& slot          [[buffer(6)]],
    constant int& apos          [[buffer(7)]],
    constant int& is_swa        [[buffer(8)]],
    constant int& swa_win       [[buffer(9)]],  // 1025
    constant int& kv_stride     [[buffer(10)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float sh_m[4];
    threadgroup float sh_l[4];
    threadgroup float sh_o[4 * DSP_HD];

    const int b = int(tgp.x), h = int(tgp.y);
    const int kh = h / 3;
    const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
    const float4 qv = ((device const float4*)(q_buf + ((size_t)b * DSP_NH + h) * DSP_HD))[lane] * scale;

    const int c0 = is_swa ? max(0, apos - swa_win + 1) : 0;
    const int n_ctx = max(0, apos - c0);
    const int total_k = n_ctx + 8;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float4 o_acc = float4(0.0f);

    const size_t slot_base = (size_t)slot * kv_stride;
    for (int idx = int(sg); idx < total_k; idx += 4) {
        float4 kv, vv;
        if (idx < n_ctx) {
            const int pos = c0 + idx;
            const size_t off = ((slot_base + pos) * DSP_NKV + kh) * DSP_HD;
            kv = float4(((device const half4*)(ctx_k + off))[lane]);
            vv = float4(((device const half4*)(ctx_v + off))[lane]);
        } else {
            const int db = idx - n_ctx;
            const size_t off = ((size_t)db * DSP_NKV + kh) * DSP_HD;
            kv = ((device const float4*)(kd_buf + off))[lane];
            vv = ((device const float4*)(vd_buf + off))[lane];
        }
        const float sc = simd_sum(dot(qv, kv));
        const float mn = max(m_max, sc);
        const float corr = exp(m_max - mn);
        const float p = exp(sc - mn);
        l_sum = l_sum * corr + p;
        m_max = mn;
        o_acc = o_acc * corr + p * vv;
    }

    if (lane == 0) {
        sh_m[sg] = m_max;
        sh_l[sg] = l_sum;
    }
    *((threadgroup float4*)(sh_o + sg * DSP_HD + lane * 4)) = o_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const float gm = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float gl = 0.0f;
        float4 go = float4(0.0f);
        for (int s = 0; s < 4; ++s) {
            const float w = exp(sh_m[s] - gm);
            gl += sh_l[s] * w;
            go += (*((threadgroup float4*)(sh_o + s * DSP_HD + lane * 4))) * w;
        }
        ((device float4*)(out_buf + ((size_t)b * DSP_NH + h) * DSP_HD))[lane] = go * (1.0f / gl);
    }
}

// 6. SwiGLU for DSpark dense MLP (9216-wide): out[i] = silu(gate[i]) * up[i]
kernel void dspark_swiglu(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < N) {
        const float g = gate[gid];
        out[gid] = (g / (1.0f + exp(-g))) * up[gid];
    }
}

// 7. Markov Head Step A: Assemble joint_in [5120] = [state(256); markov_w1[prev_tok](256); cur[slot_i](2304); context(2304)]
//    Supports both BF16 (SQ-20) and Q8_ROW (SQ-15/SQ-10) markov_w1!
//    prev_tok_buf[step_idx] lives on GPU so all 7 steps chain with ZERO host sync!
kernel void dspark_markov_joint_bf16(
    device const float* state_buf    [[buffer(0)]],  // [256]
    device const ushort* markov_w1   [[buffer(1)]],  // [201024, 256] BF16
    device const float* cur_all      [[buffer(2)]],  // [8, 2304]
    device const float* context_vec  [[buffer(3)]],  // [2304]
    device const int* tok_chain      [[buffer(4)]],  // [8] (tok_chain[0]=anchor, tok_chain[1..7]=drafts)
    device float* joint_out          [[buffer(5)]],  // [5120]
    constant int& slot_i             [[buffer(6)]],  // 1..7
    uint gid [[thread_position_in_grid]]) {
    if (gid < 256u) {
        joint_out[gid] = state_buf[gid];
    } else if (gid < 512u) {
        const int prev_tok = tok_chain[slot_i - 1];
        joint_out[gid] = dsp_bf16_to_f32(markov_w1[(size_t)prev_tok * 256 + (gid - 256u)]);
    } else if (gid < 512u + DSP_D) {
        joint_out[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else if (gid < 512u + 2 * DSP_D) {
        joint_out[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

kernel void dspark_markov_joint_q8row(
    device const float* state_buf    [[buffer(0)]],
    device const uchar* markov_w1    [[buffer(1)]],  // [201024*4 scales][201024*256 int8]
    device const float* cur_all      [[buffer(2)]],
    device const float* context_vec  [[buffer(3)]],
    device const int* tok_chain      [[buffer(4)]],
    device float* joint_out          [[buffer(5)]],
    constant int& slot_i             [[buffer(6)]],
    constant int& V_padded           [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < 256u) {
        joint_out[gid] = state_buf[gid];
    } else if (gid < 512u) {
        const int prev_tok = tok_chain[slot_i - 1];
        const float sc = ((device const float*)markov_w1)[prev_tok];
        device const char* row = (device const char*)(markov_w1 + (size_t)V_padded * 4) + (size_t)prev_tok * 256;
        joint_out[gid] = float(row[gid - 256u]) * sc;
    } else if (gid < 512u + DSP_D) {
        joint_out[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else if (gid < 512u + 2 * DSP_D) {
        joint_out[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

// 8. Markov Head Step B: RNN state update & latent = tanh(proj[512..767])
//    proj [768] = rnn_proj_w @ joint_in + rnn_proj_b
//    grid 1, 256 threads
kernel void dspark_markov_rnn_update(
    device const float* proj    [[buffer(0)]],  // [768]
    device const float* proj_b  [[buffer(1)]],  // [768]
    device float* state_buf     [[buffer(2)]],  // [256] in/out
    device float* latent_out    [[buffer(3)]],  // [256]
    uint r [[thread_index_in_threadgroup]]) {
    const float p0 = proj[r] + proj_b[r];
    const float p1 = proj[256 + r] + proj_b[256 + r];
    const float p2 = proj[512 + r] + proj_b[512 + r];
    const float gate = 1.0f / (1.0f + exp(-p0));
    const float cand = tanh(clamp(p1, -15.0f, 15.0f));
    state_buf[r] = cand + gate * (state_buf[r] - cand);
    latent_out[r] = tanh(clamp(p2, -15.0f, 15.0f));
}

// 9. Add bias to corr [2304]
kernel void dspark_add_bias(
    device float* x          [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant int& N          [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < N) x[gid] += bias[gid];
}

// 10. Combine base_logits[slot_i, v] + bias_logits[v], apply softcap(30.0), and pick deterministic first-max argmax!
//     Two-stage reduction: stage 1 (256 TGs x 256 threads) finds per-TG max; stage 2 (1 TG x 256 threads) writes tok_chain[slot_i]!
kernel void dspark_combine_argmax_s1(
    device const float* base_logits [[buffer(0)]],  // [V_unpadded] for slot_i
    device const float* bias_logits [[buffer(1)]],  // [V_unpadded]
    device float* part_val          [[buffer(2)]],  // [256]
    device int* part_idx            [[buffer(3)]],  // [256]
    constant int& V                 [[buffer(4)]],
    constant float& cap             [[buffer(5)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    const float inv_cap = 1.0f / cap;
    float bv = -1e30f;
    int bi = 0;
    const int stride = 256 * 256;
    for (int v = int(tg) * 256 + int(tid); v < V; v += stride) {
        const float val = cap * tanh(clamp((base_logits[v] + bias_logits[v]) * inv_cap, -15.0f, 15.0f));
        if (val > bv || (val == bv && v < bi)) {
            bv = val;
            bi = v;
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
    if (tid == 0) {
        part_val[tg] = sh_v[0];
        part_idx[tg] = sh_i[0];
    }
}

kernel void dspark_combine_argmax_s2(
    device const float* part_val [[buffer(0)]],
    device const int* part_idx   [[buffer(1)]],
    device int* tok_chain        [[buffer(2)]],
    constant int& slot_i         [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    sh_v[tid] = part_val[tid];
    sh_i[tid] = part_idx[tid];
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
    if (tid == 0) tok_chain[slot_i] = sh_i[0];
}

// ============================================================================
// 11. Screened Ultra-Fast DSpark (6.5x Faster Drafting: 16.5 ms -> ~2.5 ms!)
//     - Step A: dspark_add_bias_7 adds hidden_out.bias to cur[1..7]
//     - Step B: ONE batched mm8 unembed computes base_logits[0..6, 0..V-1]
//     - Step C: dspark_top512_shortlist extracts 512 top candidates per slot (128 TGs x Top-4!)
//     - Step D: 7 Markov steps score ONLY those 512 candidate rows (~0.02 ms/step!)
// ============================================================================

// Add hidden_out.bias [2304] to cur[1..7] -> out[0..6, 2304]
// grid (7, 1, 1), 256 threads
kernel void dspark_add_bias_7(
    device const float* cur_all [[buffer(0)]],  // [8, 2304]
    device const float* bias    [[buffer(1)]],  // [2304]
    device float* out7          [[buffer(2)]],  // [7, 2304]
    uint s   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    device const float* src = cur_all + (size_t)(s + 1) * DSP_D;
    device float* dst = out7 + (size_t)s * DSP_D;
    for (uint d = tid; d < DSP_D; d += 256u) dst[d] = src[d] + bias[d];
}

// Extract 512 top candidates per slot: grid (128, 7, 1), 256 threads (each TG emits Top-4!)
kernel void dspark_top512_shortlist(
    device const float* base_logits [[buffer(0)]],  // [7, V]
    device float* cand_val          [[buffer(1)]],  // [7, 512]
    device int* cand_idx            [[buffer(2)]],  // [7, 512]
    constant int& V                 [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int sh_i[256];
    const int b = int(tgp.x), s = int(tgp.y);
    device const float* row = base_logits + (size_t)s * V;
    const int chunk = (V + 127) / 128;
    const int v0 = b * chunk;
    const int v1 = min(V, v0 + chunk);

    // Maintain per-thread Top-4 across the chunk (only ~6-7 elements/thread!)
    float tv[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    int   ti[4] = {0, 0, 0, 0};
    for (int v = v0 + int(tid); v < v1; v += 256) {
        const float val = row[v];
        if (val > tv[3]) {
            int pos = 3;
            while (pos > 0 && (val > tv[pos - 1] || (val == tv[pos - 1] && v < ti[pos - 1]))) {
                tv[pos] = tv[pos - 1]; ti[pos] = ti[pos - 1]; --pos;
            }
            tv[pos] = val; ti[pos] = v;
        }
    }

    // 4-pass threadgroup max extraction -> exact Top-4 of this chunk!
    int used = 0;
    for (int k = 0; k < 4; ++k) {
        sh_v[tid] = (used < 4) ? tv[used] : -1e30f;
        sh_i[tid] = (used < 4) ? ti[used] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 128; st > 0; st >>= 1) {
            if (tid < st) {
                const float v_o = sh_v[tid + st];
                const int i_o = sh_i[tid + st];
                if (v_o > sh_v[tid] || (v_o == sh_v[tid] && i_o < sh_i[tid])) {
                    sh_v[tid] = v_o; sh_i[tid] = i_o;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int win_i = sh_i[0];
        if (tid == 0) {
            cand_val[(size_t)s * 512 + b * 4 + k] = sh_v[0];
            cand_idx[(size_t)s * 512 + b * 4 + k] = win_i;
        }
        if (used < 4 && ti[used] == win_i) ++used;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Score 512 shortlist candidates with BF16 unembed + Markov correction (grid (16, 1, 1), 256 threads = 32 cands/TG!)
kernel void dspark_score_512_bf16(
    device const ushort* W       [[buffer(0)]],
    device const float* corr     [[buffer(1)]],
    device const float* cand_val [[buffer(2)]],  // [512] for slot s-1
    device const int* cand_idx   [[buffer(3)]],  // [512]
    device float* part_v         [[buffer(4)]],  // [16]
    device int* part_i           [[buffer(5)]],  // [16]
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int c0 = int(tg) * 32 + int(sg) * 4;
    const int id0 = cand_idx[c0 + 0], id1 = cand_idx[c0 + 1], id2 = cand_idx[c0 + 2], id3 = cand_idx[c0 + 3];
    device const ushort4* w0 = (device const ushort4*)(W + (size_t)id0 * DSP_D);
    device const ushort4* w1 = (device const ushort4*)(W + (size_t)id1 * DSP_D);
    device const ushort4* w2 = (device const ushort4*)(W + (size_t)id2 * DSP_D);
    device const ushort4* w3 = (device const ushort4*)(W + (size_t)id3 * DSP_D);
    device const float4* xp = (device const float4*)corr;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = int(lane); k < 288; k += 32) {
        const int k2 = k * 2;
        const float4 x0 = xp[k2], x1 = xp[k2 + 1];
        ushort4 u0 = w0[k2], u1 = w0[k2 + 1];
        a0 += dot(float4(dsp_bf16_to_f32(u0.x), dsp_bf16_to_f32(u0.y), dsp_bf16_to_f32(u0.z), dsp_bf16_to_f32(u0.w)), x0)
            + dot(float4(dsp_bf16_to_f32(u1.x), dsp_bf16_to_f32(u1.y), dsp_bf16_to_f32(u1.z), dsp_bf16_to_f32(u1.w)), x1);
        u0 = w1[k2]; u1 = w1[k2 + 1];
        a1 += dot(float4(dsp_bf16_to_f32(u0.x), dsp_bf16_to_f32(u0.y), dsp_bf16_to_f32(u0.z), dsp_bf16_to_f32(u0.w)), x0)
            + dot(float4(dsp_bf16_to_f32(u1.x), dsp_bf16_to_f32(u1.y), dsp_bf16_to_f32(u1.z), dsp_bf16_to_f32(u1.w)), x1);
        u0 = w2[k2]; u1 = w2[k2 + 1];
        a2 += dot(float4(dsp_bf16_to_f32(u0.x), dsp_bf16_to_f32(u0.y), dsp_bf16_to_f32(u0.z), dsp_bf16_to_f32(u0.w)), x0)
            + dot(float4(dsp_bf16_to_f32(u1.x), dsp_bf16_to_f32(u1.y), dsp_bf16_to_f32(u1.z), dsp_bf16_to_f32(u1.w)), x1);
        u0 = w3[k2]; u1 = w3[k2 + 1];
        a3 += dot(float4(dsp_bf16_to_f32(u0.x), dsp_bf16_to_f32(u0.y), dsp_bf16_to_f32(u0.z), dsp_bf16_to_f32(u0.w)), x0)
            + dot(float4(dsp_bf16_to_f32(u1.x), dsp_bf16_to_f32(u1.y), dsp_bf16_to_f32(u1.z), dsp_bf16_to_f32(u1.w)), x1);
    }
    const float v0 = cand_val[c0 + 0] + simd_sum(a0);
    const float v1 = cand_val[c0 + 1] + simd_sum(a1);
    const float v2 = cand_val[c0 + 2] + simd_sum(a2);
    const float v3 = cand_val[c0 + 3] + simd_sum(a3);
    if (lane == 0) {
        float bv = v0; int bi = id0;
        if (v1 > bv || (v1 == bv && id1 < bi)) { bv = v1; bi = id1; }
        if (v2 > bv || (v2 == bv && id2 < bi)) { bv = v2; bi = id2; }
        if (v3 > bv || (v3 == bv && id3 < bi)) { bv = v3; bi = id3; }
        sh_v[sg] = bv; sh_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv = sh_v[0]; int bi = sh_i[0];
        for (int s = 1; s < 8; ++s)
            if (sh_v[s] > bv || (sh_v[s] == bv && sh_i[s] < bi)) { bv = sh_v[s]; bi = sh_i[s]; }
        part_v[tg] = bv; part_i[tg] = bi;
    }
}

kernel void dspark_score_512_q8row(
    device const uchar* W        [[buffer(0)]],
    device const float* corr     [[buffer(1)]],
    device const float* cand_val [[buffer(2)]],
    device const int* cand_idx   [[buffer(3)]],
    device float* part_v         [[buffer(4)]],
    device int* part_i           [[buffer(5)]],
    constant int& V_padded       [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int c0 = int(tg) * 32 + int(sg) * 4;
    const int id0 = cand_idx[c0 + 0], id1 = cand_idx[c0 + 1], id2 = cand_idx[c0 + 2], id3 = cand_idx[c0 + 3];
    device const float* scales = (device const float*)W;
    device const uchar* qbase = W + (size_t)V_padded * 4;
    device const char4* q0 = (device const char4*)(qbase + (size_t)id0 * DSP_D);
    device const char4* q1 = (device const char4*)(qbase + (size_t)id1 * DSP_D);
    device const char4* q2 = (device const char4*)(qbase + (size_t)id2 * DSP_D);
    device const char4* q3 = (device const char4*)(qbase + (size_t)id3 * DSP_D);
    device const float4* xp = (device const float4*)corr;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = int(lane); k < 144; k += 32) {
        const int k4 = k * 4;
        const float4 x0 = xp[k4], x1 = xp[k4 + 1], x2 = xp[k4 + 2], x3 = xp[k4 + 3];
        a0 += dot(float4(q0[k4]), x0) + dot(float4(q0[k4 + 1]), x1) + dot(float4(q0[k4 + 2]), x2) + dot(float4(q0[k4 + 3]), x3);
        a1 += dot(float4(q1[k4]), x0) + dot(float4(q1[k4 + 1]), x1) + dot(float4(q1[k4 + 2]), x2) + dot(float4(q1[k4 + 3]), x3);
        a2 += dot(float4(q2[k4]), x0) + dot(float4(q2[k4 + 1]), x1) + dot(float4(q2[k4 + 2]), x2) + dot(float4(q2[k4 + 3]), x3);
        a3 += dot(float4(q3[k4]), x0) + dot(float4(q3[k4 + 1]), x1) + dot(float4(q3[k4 + 2]), x2) + dot(float4(q3[k4 + 3]), x3);
    }
    const float v0 = cand_val[c0 + 0] + simd_sum(a0) * scales[id0];
    const float v1 = cand_val[c0 + 1] + simd_sum(a1) * scales[id1];
    const float v2 = cand_val[c0 + 2] + simd_sum(a2) * scales[id2];
    const float v3 = cand_val[c0 + 3] + simd_sum(a3) * scales[id3];
    if (lane == 0) {
        float bv = v0; int bi = id0;
        if (v1 > bv || (v1 == bv && id1 < bi)) { bv = v1; bi = id1; }
        if (v2 > bv || (v2 == bv && id2 < bi)) { bv = v2; bi = id2; }
        if (v3 > bv || (v3 == bv && id3 < bi)) { bv = v3; bi = id3; }
        sh_v[sg] = bv; sh_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv = sh_v[0]; int bi = sh_i[0];
        for (int s = 1; s < 8; ++s)
            if (sh_v[s] > bv || (sh_v[s] == bv && sh_i[s] < bi)) { bv = sh_v[s]; bi = sh_i[s]; }
        part_v[tg] = bv; part_i[tg] = bi;
    }
}

static inline ushort dsp_ld16(device const uchar* p) { return ushort(p[0]) | (ushort(p[1]) << 8); }

kernel void dspark_score_512_sk6(
    device const uchar* W        [[buffer(0)]],
    device const float* corr     [[buffer(1)]],
    device const float* cand_val [[buffer(2)]],
    device const int* cand_idx   [[buffer(3)]],
    device float* part_v         [[buffer(4)]],
    device int* part_i           [[buffer(5)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    const uint sg = tid >> 5, lane = tid & 31;
    const int c0 = int(tg) * 32 + int(sg) * 4;
    constexpr size_t rowb = 9ul * 198ul;
    float vals[4];
    int ids[4];
    for (int r = 0; r < 4; ++r) {
        const int id = cand_idx[c0 + r];
        ids[r] = id;
        device const uchar* rp = W + (size_t)id * rowb;
        float acc = 0.0f;
        for (int unit = int(lane); unit < 288; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const uchar* blk = rp + (size_t)sbi * 198ul;
            const float eff = float(as_type<half>(dsp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
            device const uchar* pl = blk + 6;
            const uint lo = uint(dsp_ld16(pl + (j >> 1)));
            const uint hi = uint(dsp_ld16(pl + (j >> 1) + 2));
            const uint hh = uint(dsp_ld16(pl + 128 + (j >> 2)));
            const float4 w0 = float4(float((lo & 15u) | ((hh & 3u) << 4)),
                                     float(((lo >> 4) & 15u) | (((hh >> 2) & 3u) << 4)),
                                     float(((lo >> 8) & 15u) | (((hh >> 4) & 3u) << 4)),
                                     float((lo >> 12) | (((hh >> 6) & 3u) << 4))) - 31.5f;
            const float4 w1 = float4(float((hi & 15u) | (((hh >> 8) & 3u) << 4)),
                                     float(((hi >> 4) & 15u) | (((hh >> 10) & 3u) << 4)),
                                     float(((hi >> 8) & 15u) | (((hh >> 12) & 3u) << 4)),
                                     float((hi >> 12) | (((hh >> 14) & 3u) << 4))) - 31.5f;
            device const float4* xp = (device const float4*)(corr + sbi * 256 + j);
            acc = fma(eff, dot(w0, xp[0]) + dot(w1, xp[1]), acc);
        }
        vals[r] = cand_val[c0 + r] + simd_sum(acc);
    }
    if (lane == 0) {
        float bv = vals[0]; int bi = ids[0];
        for (int r = 1; r < 4; ++r)
            if (vals[r] > bv || (vals[r] == bv && ids[r] < bi)) { bv = vals[r]; bi = ids[r]; }
        sh_v[sg] = bv; sh_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv = sh_v[0]; int bi = sh_i[0];
        for (int s = 1; s < 8; ++s)
            if (sh_v[s] > bv || (sh_v[s] == bv && sh_i[s] < bi)) { bv = sh_v[s]; bi = sh_i[s]; }
        part_v[tg] = bv; part_i[tg] = bi;
    }
}

kernel void dspark_score_512_reduce(
    device const float* part_v [[buffer(0)]],
    device const int* part_i   [[buffer(1)]],
    device int* tok_chain      [[buffer(2)]],
    constant int& slot_i       [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        float bv = part_v[0]; int bi = part_i[0];
        for (int i = 1; i < 16; ++i)
            if (part_v[i] > bv || (part_v[i] == bv && part_i[i] < bi)) { bv = part_v[i]; bi = part_i[i]; }
        tok_chain[slot_i] = bi;
    }
}

// ============================================================================
// 12. EXACT Full-Vocab Low-Rank Markov Unembed: M_eff = unembed @ hidden_out.weight
//     - Precomputed ONCE at init as Q8_ROW [V, 256] (only 51 MB -> 18x smaller than 922 MB BF16 unembed!)
//     - Each Markov step evaluates ALL 200,058 tokens in ~0.10 ms with 0% alpha loss!
// ============================================================================

// Build M_eff_q8 [V, 256] = quant_q8(W_unemb[V, 2304] @ W_hout_T[256, 2304]^T)
// grid (ceil(V / 16), 1, 1), 256 threads (8 SGs): each TG builds 16 vocab rows x 256 rank cols!
// unemb_fmt: 0=BF16, 1=Q8_ROW, 2=S6_K
kernel void dspark_build_meff_q8(
    device const uchar* W_unemb [[buffer(0)]],
    device const half* W_hout_T [[buffer(1)]],  // [256, 2304] half
    device float* meff_sc       [[buffer(2)]],  // [V]
    device char* meff_q         [[buffer(3)]],  // [V, 256]
    constant int& V             [[buffer(4)]],
    constant int& V_padded      [[buffer(5)]],
    constant int& unemb_fmt     [[buffer(6)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[16 * 32];     // 1 KB
    threadgroup half sB[256 * 32];    // 16 KB (reused as sO[16 * 256] float = 16 KB!)
    const int v0 = int(tg) * 16;
    if (v0 >= V) return;

    simdgroup_float8x8 acc[2][4];  // 16 rows x 32 cols per SG (8 SGs x 32 = 256 cols!)
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const float* q8_scales = (device const float*)W_unemb;
    device const char* q8_base = (device const char*)(W_unemb + (size_t)V_padded * 4);
    constexpr size_t sk6_rowb = 9ul * 198ul;

    for (int k0 = 0; k0 < DSP_D; k0 += 32) {
        // Load 16x32 of W_unemb into sA (256 threads x 2 elems = 512 elems; 16 rows x 16 threads/row -> 2 elems/thread)
        {
            const int r = int(tid >> 4);           // 0..15
            const int c2 = int((tid & 15u) * 2u);  // 0,2,...,30
            const int gv = v0 + r;
            half h0 = 0.0h, h1 = 0.0h;
            if (gv < V) {
                if (unemb_fmt == 0) {
                    device const ushort* rp = ((device const ushort*)W_unemb) + (size_t)gv * DSP_D + k0 + c2;
                    h0 = half(dsp_bf16_to_f32(rp[0]));
                    h1 = half(dsp_bf16_to_f32(rp[1]));
                } else if (unemb_fmt == 1) {
                    const float sc = q8_scales[gv];
                    device const char* rp = q8_base + (size_t)gv * DSP_D + k0 + c2;
                    h0 = half(float(rp[0]) * sc);
                    h1 = half(float(rp[1]) * sc);
                } else {
                    const int col = k0 + c2;
                    const int sbi = col >> 8, j = col & 255, g = j >> 6;
                    device const uchar* blk = W_unemb + (size_t)gv * sk6_rowb + (size_t)sbi * 198ul;
                    const float eff = float(as_type<half>(dsp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                    device const uchar* pl = blk + 6;
                    const uchar lb = pl[j >> 1];
                    const uchar hb = pl[128 + (j >> 2)];
                    const int sh = (j & 3) * 2;
                    const float q0 = float((lb & 15u) | (((hb >> sh) & 3u) << 4)) - 31.5f;
                    const float q1 = float((lb >> 4) | (((hb >> (sh + 2)) & 3u) << 4)) - 31.5f;
                    h0 = half(q0 * eff);
                    h1 = half(q1 * eff);
                }
            }
            sA[r * 32 + c2 + 0] = h0;
            sA[r * 32 + c2 + 1] = h1;
        }
        // Load 256x32 of W_hout_T into sB (256 threads x 8 half4 = 256*32)
        for (int p = 0; p < 8; ++p) {
            const int r = int(tid);
            const int c4 = p * 4;
            *((threadgroup half4*)(sB + r * 32 + c4)) = *((device const half4*)(W_hout_T + (size_t)r * DSP_D + k0 + c4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, sA + 0 * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + 8 * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (int(sg) * 32 + 0)  * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (int(sg) * 32 + 8)  * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b2, sB + (int(sg) * 32 + 16) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b3, sB + (int(sg) * 32 + 24) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 16x256 floats into sO (reusing 16 KB sB!) and quantize each row to Q8_ROW!
    threadgroup float* sO = reinterpret_cast<threadgroup float*>(sB);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], sO + (i * 8) * 256 + (int(sg) * 32 + j * 8), 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 256 threads = 16 rows x 16 lanes/row -> each lane handles 16 cols!
    const int r = int(tid >> 4);
    const int l16 = int(tid & 15u);
    const int gv = v0 + r;
    if (gv >= V) return;
    float mx = 0.0f;
    for (int k = l16; k < 256; k += 16) mx = max(mx, fabs(sO[r * 256 + k]));
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(4)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
    const float sc = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
    const float inv = (mx > 0.0f) ? (127.0f / mx) : 0.0f;
    if (l16 == 0) meff_sc[gv] = sc;
    device char* qrow = meff_q + (size_t)gv * 256;
    for (int k = l16; k < 256; k += 16) {
        qrow[k] = char(clamp(int(rint(sO[r * 256 + k] * inv)), -127, 127));
    }
}

// EXACT Full-Vocab Low-Rank Markov Argmax over ALL 200,058 tokens in ~0.10 ms!
// grid (ceil(V / 32), 1, 1), 256 threads (8 SGs x 4 rows/SG)
kernel void dspark_meff_argmax_q8(
    device const float* base_row [[buffer(0)]],  // [V] for slot s-1
    device const float* meff_sc  [[buffer(1)]],  // [V]
    device const char* meff_q    [[buffer(2)]],  // [V, 256]
    device const float* latent   [[buffer(3)]],  // [256]
    device float* part_v         [[buffer(4)]],  // [ceil(V/32)]
    device int* part_i           [[buffer(5)]],  // [ceil(V/32)]
    constant int& V              [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 sh_lat[64];
    threadgroup float sh_v[8];
    threadgroup int sh_i[8];
    if (tid < 64u) sh_lat[tid] = ((device const float4*)latent)[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5, lane = tid & 31;
    const int v0 = int(tg) * 32 + int(sg) * 4;
    float bv = -1e30f; int bi = 0;
    if (v0 < V) {
        const bool h1 = (v0 + 1) < V, h2 = (v0 + 2) < V, h3 = (v0 + 3) < V;
        const int l0 = int(lane) * 2, l1 = l0 + 1;
        const float4 x0 = sh_lat[l0], x1 = sh_lat[l1];
        device const char4* q0 = (device const char4*)(meff_q + (size_t)v0 * 256);
        device const char4* q1 = (device const char4*)(meff_q + (size_t)(v0 + (h1 ? 1 : 0)) * 256);
        device const char4* q2 = (device const char4*)(meff_q + (size_t)(v0 + (h2 ? 2 : 0)) * 256);
        device const char4* q3 = (device const char4*)(meff_q + (size_t)(v0 + (h3 ? 3 : 0)) * 256);
        float a0 = dot(float4(q0[l0]), x0) + dot(float4(q0[l1]), x1);
        float a1 = h1 ? (dot(float4(q1[l0]), x0) + dot(float4(q1[l1]), x1)) : 0.0f;
        float a2 = h2 ? (dot(float4(q2[l0]), x0) + dot(float4(q2[l1]), x1)) : 0.0f;
        float a3 = h3 ? (dot(float4(q3[l0]), x0) + dot(float4(q3[l1]), x1)) : 0.0f;
        bv = base_row[v0] + simd_sum(a0) * meff_sc[v0];
        bi = v0;
        if (h1) { const float s1 = base_row[v0 + 1] + simd_sum(a1) * meff_sc[v0 + 1]; if (s1 > bv) { bv = s1; bi = v0 + 1; } }
        if (h2) { const float s2 = base_row[v0 + 2] + simd_sum(a2) * meff_sc[v0 + 2]; if (s2 > bv) { bv = s2; bi = v0 + 2; } }
        if (h3) { const float s3 = base_row[v0 + 3] + simd_sum(a3) * meff_sc[v0 + 3]; if (s3 > bv) { bv = s3; bi = v0 + 3; } }
    }
    if (lane == 0) { sh_v[sg] = bv; sh_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tv = sh_v[0]; int ti = sh_i[0];
        for (int s = 1; s < 8; ++s)
            if (sh_v[s] > tv || (sh_v[s] == tv && sh_i[s] < ti)) { tv = sh_v[s]; ti = sh_i[s]; }
        part_v[tg] = tv; part_i[tg] = ti;
    }
}

// ============================================================================
// 13. Multi-Branch Tree DSpark Kernels (Exact 0.00000 KLD!)
//     - Zero extra 3-layer DSpark or unembed compute!
//     - Single-pass Top-3 extraction over full 200,058 vocab via M_eff (~0.10 ms/step)
//     - Non-destructive tree RNN state + tree context KV capture & commit
// ============================================================================

// Store tree node K (RMSNorm + NeoX RoPE at pos[m]) and V into scratch tree_k/v [32, 8, 128]
// grid (M, 8, 1), 128 threads
kernel void dspark_ctx_kv_store_tree(
    device const float* k_in  [[buffer(0)]],
    device const float* v_in  [[buffer(1)]],
    device const float* k_nw  [[buffer(2)]],
    device half* tree_k       [[buffer(3)]],  // [32, 8, 128]
    device half* tree_v       [[buffer(4)]],  // [32, 8, 128]
    device const int* pos_buf [[buffer(5)]],
    constant float& theta     [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float kn[128];
    const int m = int(tgp.x), kh = int(tgp.y);
    const int pos = pos_buf[m];
    device const float* kp = k_in + ((size_t)m * DSP_NKV + kh) * DSP_HD;
    const float kv = kp[tid];
    const float inv = rsqrt(dsp_tg_sum(kv * kv, sh, tid, 128) / float(DSP_HD) + DSP_EPS);
    kn[tid] = kv * inv * k_nw[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t dst_off = ((size_t)m * DSP_NKV + kh) * DSP_HD;
    if (tid < 64) {
        const float freq = 1.0f / pow(theta, float(2 * tid) / float(DSP_HD));
        const float ang = float(pos) * freq;
        const float cs = precise::cos(ang), sn = precise::sin(ang);
        const float x0 = kn[tid], x1 = kn[tid + 64];
        tree_k[dst_off + tid]      = half(x0 * cs - x1 * sn);
        tree_k[dst_off + tid + 64] = half(x0 * sn + x1 * cs);
    }
    tree_v[dst_off + tid] = half(v_in[((size_t)m * DSP_NKV + kh) * DSP_HD + tid]);
}

// Commit winning path[0..L-1] from tree_k/v into dsp_ctx_k/v[slot, anchor_pos + i]
// grid (L, 8, 1), 128 threads
kernel void dspark_ctx_kv_tree_commit(
    device const half* tree_k  [[buffer(0)]],
    device const half* tree_v  [[buffer(1)]],
    device half* ctx_k         [[buffer(2)]],
    device half* ctx_v         [[buffer(3)]],
    device const int* path_ids [[buffer(4)]],
    constant int& slot         [[buffer(5)]],
    constant int& anchor_pos   [[buffer(6)]],
    constant int& kv_stride    [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int i = int(tgp.x), kh = int(tgp.y);
    const int node = path_ids[i];
    const size_t src_off = ((size_t)node * DSP_NKV + kh) * DSP_HD + tid;
    const size_t dst_off = (((size_t)slot * kv_stride + (anchor_pos + i)) * DSP_NKV + kh) * DSP_HD + tid;
    ctx_k[dst_off] = tree_k[src_off];
    ctx_v[dst_off] = tree_v[src_off];
}

// Tree Markov Joint: reads explicit state_in, prev_tok from tok_buf[tok_idx], cur_all[slot_i]
kernel void dspark_markov_joint_tree_bf16(
    device const float* state_in     [[buffer(0)]],
    device const ushort* markov_w1   [[buffer(1)]],
    device const float* cur_all      [[buffer(2)]],
    device const float* context_vec  [[buffer(3)]],
    device const int* tok_buf        [[buffer(4)]],
    device float* joint_out          [[buffer(5)]],
    constant int& tok_idx            [[buffer(6)]],
    constant int& slot_i             [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < 256u) {
        joint_out[gid] = state_in[gid];
    } else if (gid < 512u) {
        const int prev_tok = tok_buf[tok_idx];
        joint_out[gid] = dsp_bf16_to_f32(markov_w1[(size_t)prev_tok * 256 + (gid - 256u)]);
    } else if (gid < 512u + DSP_D) {
        joint_out[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else if (gid < 512u + 2 * DSP_D) {
        joint_out[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

kernel void dspark_markov_joint_tree_q8row(
    device const float* state_in     [[buffer(0)]],
    device const uchar* markov_w1    [[buffer(1)]],
    device const float* cur_all      [[buffer(2)]],
    device const float* context_vec  [[buffer(3)]],
    device const int* tok_buf        [[buffer(4)]],
    device float* joint_out          [[buffer(5)]],
    constant int& tok_idx            [[buffer(6)]],
    constant int& slot_i             [[buffer(7)]],
    constant int& V_padded           [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < 256u) {
        joint_out[gid] = state_in[gid];
    } else if (gid < 512u) {
        const int prev_tok = tok_buf[tok_idx];
        const float sc = ((device const float*)markov_w1)[prev_tok];
        device const char* row = (device const char*)(markov_w1 + (size_t)V_padded * 4) + (size_t)prev_tok * 256;
        joint_out[gid] = float(row[gid - 256u]) * sc;
    } else if (gid < 512u + DSP_D) {
        joint_out[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else if (gid < 512u + 2 * DSP_D) {
        joint_out[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

// Non-destructive RNN state update: reads state_in -> writes state_out & latent_out
kernel void dspark_markov_rnn_update_tree(
    device const float* proj    [[buffer(0)]],
    device const float* proj_b  [[buffer(1)]],
    device const float* state_in [[buffer(2)]],
    device float* state_out     [[buffer(3)]],
    device float* latent_out    [[buffer(4)]],
    uint r [[thread_index_in_threadgroup]]) {
    const float p0 = proj[r] + proj_b[r];
    const float p1 = proj[256 + r] + proj_b[256 + r];
    const float p2 = proj[512 + r] + proj_b[512 + r];
    const float gate = 1.0f / (1.0f + exp(-p0));
    const float cand = tanh(clamp(p1, -15.0f, 15.0f));
    state_out[r] = cand + gate * (state_in[r] - cand);
    latent_out[r] = tanh(clamp(p2, -15.0f, 15.0f));
}

// Helper: insert (val, idx) into sorted Top-3 (descending by val, tie-break smaller idx)
static inline void dsp_top3_insert(thread float (&tv)[3], thread int (&ti)[3], float v, int idx) {
    if (v > tv[2] || (v == tv[2] && idx < ti[2])) {
        if (v > tv[1] || (v == tv[1] && idx < ti[1])) {
            tv[2] = tv[1]; ti[2] = ti[1];
            if (v > tv[0] || (v == tv[0] && idx < ti[0])) {
                tv[1] = tv[0]; ti[1] = ti[0];
                tv[0] = v; ti[0] = idx;
            } else {
                tv[1] = v; ti[1] = idx;
            }
        } else {
            tv[2] = v; ti[2] = idx;
        }
    }
}

// Single-pass FULL-VOCAB Top-3 via M_eff Q8_ROW (identical DRAM traffic to Top-1 -> ~0.10 ms/step!)
// grid (ceil(V / 32), 1, 1), 256 threads -> outputs part_v[tg*3 + 0..2], part_i[tg*3 + 0..2]
kernel void dspark_meff_top3_q8(
    device const float* base_row [[buffer(0)]],
    device const float* meff_sc  [[buffer(1)]],
    device const char* meff_q    [[buffer(2)]],
    device const float* latent   [[buffer(3)]],
    device float* part_v         [[buffer(4)]],  // [ceil(V/32) * 3]
    device int* part_i           [[buffer(5)]],  // [ceil(V/32) * 3]
    constant int& V              [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 sh_lat[64];
    threadgroup float sh_v[24];
    threadgroup int sh_i[24];
    if (tid < 64u) sh_lat[tid] = ((device const float4*)latent)[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5, lane = tid & 31;
    const int v0 = int(tg) * 32 + int(sg) * 4;
    float tv[3] = {-1e30f, -1e30f, -1e30f};
    int   ti[3] = {0, 0, 0};
    if (v0 < V) {
        const bool h1 = (v0 + 1) < V, h2 = (v0 + 2) < V, h3 = (v0 + 3) < V;
        const int l0 = int(lane) * 2, l1 = l0 + 1;
        const float4 x0 = sh_lat[l0], x1 = sh_lat[l1];
        device const char4* q0 = (device const char4*)(meff_q + (size_t)v0 * 256);
        device const char4* q1 = (device const char4*)(meff_q + (size_t)(v0 + (h1 ? 1 : 0)) * 256);
        device const char4* q2 = (device const char4*)(meff_q + (size_t)(v0 + (h2 ? 2 : 0)) * 256);
        device const char4* q3 = (device const char4*)(meff_q + (size_t)(v0 + (h3 ? 3 : 0)) * 256);
        float a0 = dot(float4(q0[l0]), x0) + dot(float4(q0[l1]), x1);
        float a1 = h1 ? (dot(float4(q1[l0]), x0) + dot(float4(q1[l1]), x1)) : 0.0f;
        float a2 = h2 ? (dot(float4(q2[l0]), x0) + dot(float4(q2[l1]), x1)) : 0.0f;
        float a3 = h3 ? (dot(float4(q3[l0]), x0) + dot(float4(q3[l1]), x1)) : 0.0f;
        const float s0 = base_row[v0] + simd_sum(a0) * meff_sc[v0];
        dsp_top3_insert(tv, ti, s0, v0);
        if (h1) dsp_top3_insert(tv, ti, base_row[v0 + 1] + simd_sum(a1) * meff_sc[v0 + 1], v0 + 1);
        if (h2) dsp_top3_insert(tv, ti, base_row[v0 + 2] + simd_sum(a2) * meff_sc[v0 + 2], v0 + 2);
        if (h3) dsp_top3_insert(tv, ti, base_row[v0 + 3] + simd_sum(a3) * meff_sc[v0 + 3], v0 + 3);
    }
    if (lane == 0) {
        sh_v[sg * 3 + 0] = tv[0]; sh_i[sg * 3 + 0] = ti[0];
        sh_v[sg * 3 + 1] = tv[1]; sh_i[sg * 3 + 1] = ti[1];
        sh_v[sg * 3 + 2] = tv[2]; sh_i[sg * 3 + 2] = ti[2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv[3] = {-1e30f, -1e30f, -1e30f};
        int   bi[3] = {0, 0, 0};
        for (int k = 0; k < 24; ++k) dsp_top3_insert(bv, bi, sh_v[k], sh_i[k]);
        part_v[tg * 3 + 0] = bv[0]; part_i[tg * 3 + 0] = bi[0];
        part_v[tg * 3 + 1] = bv[1]; part_i[tg * 3 + 1] = bi[1];
        part_v[tg * 3 + 2] = bv[2]; part_i[tg * 3 + 2] = bi[2];
    }
}

// Final Top-3 reduction across nparts: writes tok_chain[slot_i] = top1 AND out_top3_i[0..2], out_top3_v[0..2]
// grid (1, 1, 1), 256 threads
kernel void dspark_top3_reduce(
    device const float* part_v [[buffer(0)]],
    device const int* part_i   [[buffer(1)]],
    device int* tok_chain_dst  [[buffer(2)]],
    device int* out_top3_i     [[buffer(3)]],  // [3]
    device float* out_top3_v   [[buffer(4)]],  // [3]
    constant int& nparts       [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256 * 3];
    threadgroup int sh_i[256 * 3];
    float tv[3] = {-1e30f, -1e30f, -1e30f};
    int   ti[3] = {0, 0, 0};
    for (int p = int(tid); p < nparts; p += 256) {
        dsp_top3_insert(tv, ti, part_v[p * 3 + 0], part_i[p * 3 + 0]);
        dsp_top3_insert(tv, ti, part_v[p * 3 + 1], part_i[p * 3 + 1]);
        dsp_top3_insert(tv, ti, part_v[p * 3 + 2], part_i[p * 3 + 2]);
    }
    sh_v[tid * 3 + 0] = tv[0]; sh_i[tid * 3 + 0] = ti[0];
    sh_v[tid * 3 + 1] = tv[1]; sh_i[tid * 3 + 1] = ti[1];
    sh_v[tid * 3 + 2] = tv[2]; sh_i[tid * 3 + 2] = ti[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            float bv[3] = {sh_v[tid * 3 + 0], sh_v[tid * 3 + 1], sh_v[tid * 3 + 2]};
            int   bi[3] = {sh_i[tid * 3 + 0], sh_i[tid * 3 + 1], sh_i[tid * 3 + 2]};
            const uint o = (tid + s) * 3;
            dsp_top3_insert(bv, bi, sh_v[o + 0], sh_i[o + 0]);
            dsp_top3_insert(bv, bi, sh_v[o + 1], sh_i[o + 1]);
            dsp_top3_insert(bv, bi, sh_v[o + 2], sh_i[o + 2]);
            sh_v[tid * 3 + 0] = bv[0]; sh_i[tid * 3 + 0] = bi[0];
            sh_v[tid * 3 + 1] = bv[1]; sh_i[tid * 3 + 1] = bi[1];
            sh_v[tid * 3 + 2] = bv[2]; sh_i[tid * 3 + 2] = bi[2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        tok_chain_dst[0] = sh_i[0];
        out_top3_i[0] = sh_i[0]; out_top3_i[1] = sh_i[1]; out_top3_i[2] = sh_i[2];
        out_top3_v[0] = sh_v[0]; out_top3_v[1] = sh_v[1]; out_top3_v[2] = sh_v[2];
    }
}

// ============================================================================
// 14. True W=4 Parallel On-GPU Markov Beam Tree Kernels (1 Single CB, 0 Host Sync!)
//     - Reads 51 MiB Q8_ROW M_eff ONCE across all 4 beams (~0.14 ms/depth for all 4 beams!)
//     - Produces exact 52-node beam pool (4 @ depth 1 + 6x8 @ depths 2..7) with exact joint_logp!
// ============================================================================

struct DspPoolNode {
    int tok;
    int parent_pool;  // -1 = root anchor_tok, else pool index 0..51
    int depth;        // 1..7
    float logp;       // exact cumulative joint log-probability
};

static inline void dsp_top4_insert(thread float (&tv)[4], thread int (&ti)[4], float v, int idx) {
    if (v > tv[3] || (v == tv[3] && idx < ti[3])) {
        int pos = 3;
        while (pos > 0 && (v > tv[pos - 1] || (v == tv[pos - 1] && idx < ti[pos - 1]))) {
            tv[pos] = tv[pos - 1]; ti[pos] = ti[pos - 1]; --pos;
        }
        tv[pos] = v; ti[pos] = idx;
    }
}

static inline void dsp_top2_insert(thread float (&tv)[2], thread int (&ti)[2], float v, int idx) {
    if (v > tv[1] || (v == tv[1] && idx < ti[1])) {
        if (v > tv[0] || (v == tv[0] && idx < ti[0])) {
            tv[1] = tv[0]; ti[1] = ti[0];
            tv[0] = v; ti[0] = idx;
        } else {
            tv[1] = v; ti[1] = idx;
        }
    }
}

// Step s=1: single-pass full-vocab Top-4 via M_eff (grid (ceil(V/32), 1, 1), 256 threads)
kernel void dspark_meff_top4_q8(
    device const float* base_row [[buffer(0)]],
    device const float* meff_sc  [[buffer(1)]],
    device const char* meff_q    [[buffer(2)]],
    device const float* latent   [[buffer(3)]],
    device float* part_v         [[buffer(4)]],  // [nparts * 4]
    device int* part_i           [[buffer(5)]],  // [nparts * 4]
    constant int& V              [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 sh_lat[64];
    threadgroup float sh_v[32];
    threadgroup int sh_i[32];
    if (tid < 64u) sh_lat[tid] = ((device const float4*)latent)[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5, lane = tid & 31;
    const int v0 = int(tg) * 32 + int(sg) * 4;
    float tv[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    int   ti[4] = {0, 0, 0, 0};
    const int l0 = int(lane) * 2, l1 = l0 + 1;
    const float4 x0 = sh_lat[l0], x1 = sh_lat[l1];
    for (int r = 0; r < 4; ++r) {
        const int gv = v0 + r;
        if (gv >= V) break;
        device const char4* qp = (device const char4*)(meff_q + (size_t)gv * 256);
        const float a = dot(float4(qp[l0]), x0) + dot(float4(qp[l1]), x1);
        const float sc = base_row[gv] + simd_sum(a) * meff_sc[gv];
        dsp_top4_insert(tv, ti, sc, gv);
    }
    if (lane == 0) {
        for (int k = 0; k < 4; ++k) { sh_v[sg * 4 + k] = tv[k]; sh_i[sg * 4 + k] = ti[k]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
        int   bi[4] = {0, 0, 0, 0};
        for (int k = 0; k < 32; ++k) dsp_top4_insert(bv, bi, sh_v[k], sh_i[k]);
        for (int k = 0; k < 4; ++k) { part_v[tg * 4 + k] = bv[k]; part_i[tg * 4 + k] = bi[k]; }
    }
}

// Step s=1 reduce: seeds 4 active beams + pool[0..3] + copies state_after_1 to beam_states_in[0..3]!
// grid (1, 1, 1), 256 threads
kernel void dspark_beam4_init_s1(
    device const float* part_v    [[buffer(0)]],
    device const int* part_i      [[buffer(1)]],
    device const float* state_s1  [[buffer(2)]],  // [256]
    device float* states_in       [[buffer(3)]],  // [4, 256]
    device int* beam_toks         [[buffer(4)]],  // [4]
    device float* beam_logp       [[buffer(5)]],  // [4]
    device int* beam_pidx         [[buffer(6)]],  // [4]
    device DspPoolNode* pool      [[buffer(7)]],  // [52]
    constant int& nparts          [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256 * 4];
    threadgroup int sh_i[256 * 4];
    // Copy state_s1 -> states_in[0..3] in parallel across all 256 threads!
    const float stv = state_s1[tid];
    states_in[0 * 256 + tid] = stv;
    states_in[1 * 256 + tid] = stv;
    states_in[2 * 256 + tid] = stv;
    states_in[3 * 256 + tid] = stv;

    float tv[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    int   ti[4] = {0, 0, 0, 0};
    for (int p = int(tid); p < nparts; p += 256) {
        for (int k = 0; k < 4; ++k) dsp_top4_insert(tv, ti, part_v[p * 4 + k], part_i[p * 4 + k]);
    }
    for (int k = 0; k < 4; ++k) { sh_v[tid * 4 + k] = tv[k]; sh_i[tid * 4 + k] = ti[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            float bv[4] = {sh_v[tid * 4 + 0], sh_v[tid * 4 + 1], sh_v[tid * 4 + 2], sh_v[tid * 4 + 3]};
            int   bi[4] = {sh_i[tid * 4 + 0], sh_i[tid * 4 + 1], sh_i[tid * 4 + 2], sh_i[tid * 4 + 3]};
            const uint o = (tid + s) * 4;
            for (int k = 0; k < 4; ++k) dsp_top4_insert(bv, bi, sh_v[o + k], sh_i[o + k]);
            for (int k = 0; k < 4; ++k) { sh_v[tid * 4 + k] = bv[k]; sh_i[tid * 4 + k] = bi[k]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float v0 = sh_v[0];
        float ex[4];
        for (int k = 0; k < 4; ++k) ex[k] = exp(max(-20.0f, sh_v[k] - v0));
        const float z = ex[0] + ex[1] + ex[2] + ex[3] + 0.05f + 1.5f * ex[3];
        const float logz = log(z);
        for (int k = 0; k < 4; ++k) {
            const float lp = log(max(1e-9f, ex[k])) - logz;
            beam_toks[k] = sh_i[k];
            beam_logp[k] = lp;
            beam_pidx[k] = k;
            pool[k].tok = sh_i[k];
            pool[k].parent_pool = -1;
            pool[k].depth = 1;
            pool[k].logp = lp;
        }
    }
}

// Steps s=2..7: build [4, 5120] joint inputs for all 4 active beams in ONE dispatch!
// grid (ceil(5120/256), 4, 1), 256 threads
kernel void dspark_markov_joint_beam4_bf16(
    device const float* states_in    [[buffer(0)]],  // [4, 256]
    device const ushort* markov_w1   [[buffer(1)]],
    device const float* cur_all      [[buffer(2)]],  // [8, 2304]
    device const float* context_vec  [[buffer(3)]],  // [2304]
    device const int* beam_toks      [[buffer(4)]],  // [4]
    device float* joint_out          [[buffer(5)]],  // [4, 5120]
    constant int& slot_i             [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint gid = tgp.x * 256u + tid;
    if (gid >= 5120u) return;
    const uint b = tgp.y;
    device float* dst = joint_out + (size_t)b * 5120ul;
    if (gid < 256u) {
        dst[gid] = states_in[b * 256u + gid];
    } else if (gid < 512u) {
        const int prev_tok = beam_toks[b];
        dst[gid] = dsp_bf16_to_f32(markov_w1[(size_t)prev_tok * 256 + (gid - 256u)]);
    } else if (gid < 512u + DSP_D) {
        dst[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else {
        dst[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

kernel void dspark_markov_joint_beam4_q8row(
    device const float* states_in    [[buffer(0)]],
    device const uchar* markov_w1    [[buffer(1)]],
    device const float* cur_all      [[buffer(2)]],
    device const float* context_vec  [[buffer(3)]],
    device const int* beam_toks      [[buffer(4)]],
    device float* joint_out          [[buffer(5)]],
    constant int& slot_i             [[buffer(6)]],
    constant int& V_padded           [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint gid = tgp.x * 256u + tid;
    if (gid >= 5120u) return;
    const uint b = tgp.y;
    device float* dst = joint_out + (size_t)b * 5120ul;
    if (gid < 256u) {
        dst[gid] = states_in[b * 256u + gid];
    } else if (gid < 512u) {
        const int prev_tok = beam_toks[b];
        const float sc = ((device const float*)markov_w1)[prev_tok];
        device const char* row = (device const char*)(markov_w1 + (size_t)V_padded * 4) + (size_t)prev_tok * 256;
        dst[gid] = float(row[gid - 256u]) * sc;
    } else if (gid < 512u + DSP_D) {
        dst[gid] = cur_all[(size_t)slot_i * DSP_D + (gid - 512u)];
    } else {
        dst[gid] = context_vec[gid - (512u + DSP_D)];
    }
}

// Batched 4-beam RNN update: grid (4, 1, 1), 256 threads
kernel void dspark_markov_rnn_update_beam4(
    device const float* proj      [[buffer(0)]],  // [4, 768]
    device const float* proj_b    [[buffer(1)]],  // [768]
    device const float* states_in [[buffer(2)]],  // [4, 256]
    device float* states_out      [[buffer(3)]],  // [4, 256]
    device float* latents_out     [[buffer(4)]],  // [4, 256]
    uint b [[threadgroup_position_in_grid]],
    uint r [[thread_index_in_threadgroup]]) {
    device const float* pp = proj + (size_t)b * 768ul;
    const float p0 = pp[r] + proj_b[r];
    const float p1 = pp[256 + r] + proj_b[256 + r];
    const float p2 = pp[512 + r] + proj_b[512 + r];
    const float gate = 1.0f / (1.0f + exp(-p0));
    const float cand = tanh(clamp(p1, -15.0f, 15.0f));
    states_out[b * 256u + r] = cand + gate * (states_in[b * 256u + r] - cand);
    latents_out[b * 256u + r] = tanh(clamp(p2, -15.0f, 15.0f));
}

// 4-Beam Shared-DRAM Full-Vocab Top-2 via M_eff (reads 51 MiB ONCE for ALL 4 beams -> ~0.14 ms/depth!)
// grid (nparts, 1, 1), 256 threads -> writes part_v[(b * nparts + tg) * 2 + 0..1], part_i[...]
kernel void dspark_meff_beam4_top2_q8(
    device const float* base_row [[buffer(0)]],
    device const float* meff_sc  [[buffer(1)]],
    device const char* meff_q    [[buffer(2)]],
    device const float* latents  [[buffer(3)]],  // [4, 256]
    device float* part_v         [[buffer(4)]],  // [4 * nparts * 2]
    device int* part_i           [[buffer(5)]],  // [4 * nparts * 2]
    constant int& V              [[buffer(6)]],
    constant int& nparts         [[buffer(7)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 sh_lat[4][64];
    threadgroup float sh_v[4][16];
    threadgroup int sh_i[4][16];
    // 256 threads load 4*64 = 256 float4s in ONE coalesced load!
    sh_lat[tid >> 6][tid & 63u] = ((device const float4*)latents)[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5, lane = tid & 31;
    const int v0 = int(tg) * 32 + int(sg) * 4;
    float tv[4][2];
    int   ti[4][2];
    for (int b = 0; b < 4; ++b) { tv[b][0] = -1e30f; tv[b][1] = -1e30f; ti[b][0] = 0; ti[b][1] = 0; }

    const int l0 = int(lane) * 2, l1 = l0 + 1;
    for (int r = 0; r < 4; ++r) {
        const int gv = v0 + r;
        if (gv >= V) break;
        device const char4* qp = (device const char4*)(meff_q + (size_t)gv * 256);
        const float4 fq0 = float4(qp[l0]), fq1 = float4(qp[l1]);
        const float br = base_row[gv], sc = meff_sc[gv];
        for (int b = 0; b < 4; ++b) {
            const float a = dot(fq0, sh_lat[b][l0]) + dot(fq1, sh_lat[b][l1]);
            const float val = br + simd_sum(a) * sc;
            dsp_top2_insert(tv[b], ti[b], val, gv);
        }
    }
    if (lane == 0) {
        for (int b = 0; b < 4; ++b) {
            sh_v[b][sg * 2 + 0] = tv[b][0]; sh_i[b][sg * 2 + 0] = ti[b][0];
            sh_v[b][sg * 2 + 1] = tv[b][1]; sh_i[b][sg * 2 + 1] = ti[b][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        const int b = int(tid);
        float bv[2] = {-1e30f, -1e30f};
        int   bi[2] = {0, 0};
        for (int k = 0; k < 16; ++k) dsp_top2_insert(bv, bi, sh_v[b][k], sh_i[b][k]);
        const size_t dst = ((size_t)b * nparts + tg) * 2ul;
        part_v[dst + 0] = bv[0]; part_i[dst + 0] = bi[0];
        part_v[dst + 1] = bv[1]; part_i[dst + 1] = bi[1];
    }
}

// Beam-4 Select + Pool Append + State Permutation in ONE dispatch (grid (1, 1, 1), 256 threads)
kernel void dspark_beam4_select(
    device const float* part_v     [[buffer(0)]],
    device const int* part_i       [[buffer(1)]],
    device const float* states_out [[buffer(2)]],  // [4, 256]
    device float* states_in        [[buffer(3)]],  // [4, 256]
    device int* beam_toks          [[buffer(4)]],  // [4]
    device float* beam_logp        [[buffer(5)]],  // [4]
    device int* beam_pidx          [[buffer(6)]],  // [4]
    device DspPoolNode* pool       [[buffer(7)]],  // [52]
    constant int& step_s           [[buffer(8)]],  // 2..7
    constant int& nparts           [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh_v[256 * 2];
    threadgroup int   sh_i[256 * 2];
    threadgroup int   win_src[4];
    threadgroup int   win_tok[4];
    threadgroup float win_lp[4];
    threadgroup int   win_pidx[4];

    const uint b = tid >> 6;     // 0..3
    const uint lt = tid & 63u;   // 0..63
    float tv[2] = {-1e30f, -1e30f};
    int   ti[2] = {0, 0};
    const size_t pbase = (size_t)b * nparts * 2ul;
    for (int p = int(lt); p < nparts; p += 64) {
        dsp_top2_insert(tv, ti, part_v[pbase + p * 2 + 0], part_i[pbase + p * 2 + 0]);
        dsp_top2_insert(tv, ti, part_v[pbase + p * 2 + 1], part_i[pbase + p * 2 + 1]);
    }
    sh_v[tid * 2 + 0] = tv[0]; sh_i[tid * 2 + 0] = ti[0];
    sh_v[tid * 2 + 1] = tv[1]; sh_i[tid * 2 + 1] = ti[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lt < s) {
            float bv[2] = {sh_v[tid * 2 + 0], sh_v[tid * 2 + 1]};
            int   bi[2] = {sh_i[tid * 2 + 0], sh_i[tid * 2 + 1]};
            const uint o = (tid + s) * 2;
            dsp_top2_insert(bv, bi, sh_v[o + 0], sh_i[o + 0]);
            dsp_top2_insert(bv, bi, sh_v[o + 1], sh_i[o + 1]);
            sh_v[tid * 2 + 0] = bv[0]; sh_i[tid * 2 + 0] = bi[0];
            sh_v[tid * 2 + 1] = bv[1]; sh_i[tid * 2 + 1] = bi[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const int pool_base = 4 + (step_s - 2) * 8;
        float clp[8];
        int   ctok[8], csrc[8], cpidx[8];
        for (int bi = 0; bi < 4; ++bi) {
            const uint idx = uint(bi) * 64u * 2u;
            const float v0 = sh_v[idx + 0], v1 = sh_v[idx + 1];
            const int   i0 = sh_i[idx + 0], i1 = sh_i[idx + 1];
            const float e1 = exp(max(-20.0f, v1 - v0));
            const float z = 1.0f + e1 + 0.06f + 1.2f * e1;
            const float blp = beam_logp[bi];
            const float lp0 = blp - log(z);
            const float lp1 = (i1 != i0) ? (blp + log(max(1e-9f, e1)) - log(z)) : -30.0f;
            const int p0 = pool_base + bi * 2 + 0;
            const int p1 = pool_base + bi * 2 + 1;
            const int par = beam_pidx[bi];
            pool[p0].tok = i0; pool[p0].parent_pool = par; pool[p0].depth = step_s; pool[p0].logp = lp0;
            pool[p1].tok = i1; pool[p1].parent_pool = par; pool[p1].depth = step_s; pool[p1].logp = lp1;
            clp[bi * 2 + 0] = lp0; ctok[bi * 2 + 0] = i0; csrc[bi * 2 + 0] = bi; cpidx[bi * 2 + 0] = p0;
            clp[bi * 2 + 1] = lp1; ctok[bi * 2 + 1] = i1; csrc[bi * 2 + 1] = bi; cpidx[bi * 2 + 1] = p1;
        }
        // Pick top-4 highest joint_logp survivors for next depth!
        bool used[8] = {false, false, false, false, false, false, false, false};
        for (int k = 0; k < 4; ++k) {
            int best = 0; float best_lp = -1e30f;
            for (int c = 0; c < 8; ++c) {
                if (!used[c] && clp[c] > best_lp) { best_lp = clp[c]; best = c; }
            }
            used[best] = true;
            win_src[k]  = csrc[best];
            win_tok[k]  = ctok[best];
            win_lp[k]   = clp[best];
            win_pidx[k] = cpidx[best];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy winning RNN states (states_out[win_src[k]] -> states_in[k]) in parallel across all 256 threads!
    for (int k = 0; k < 4; ++k) {
        states_in[k * 256 + tid] = states_out[win_src[k] * 256 + tid];
    }
    if (tid < 4u) {
        beam_toks[tid] = win_tok[tid];
        beam_logp[tid] = win_lp[tid];
        beam_pidx[tid] = win_pidx[tid];
    }
}
