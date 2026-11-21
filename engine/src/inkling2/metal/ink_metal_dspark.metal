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
