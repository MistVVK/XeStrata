// ink_metal_vision.metal - Native Apple Silicon Metal ViT for Inkling-2-Mini-Vision.strata
// Exact architecture (vision_config):
//   - 23-layer BF16 ViT, hidden=1024, 16 heads x 64, mlp_ratio=4.0 (ffn=4096), native_patch_size=20
//   - class_embedding [1024] + cls_position_embedding [1024] + bilinear 32x32 position_embedding [1024, 1024]
//   - ln_pre (LayerNorm eps=1e-5) -> 23 pre-LN blocks with axial 2D RoPE (theta=10000), ls_1.gamma, ls_2.gamma, GELU
//   - 2x2 spatial merge (4 * 1024 = 4096) -> proj_in [4096, 4096] -> GELU -> proj_out [2304, 4096]
// Uses simdgroup_matrix 8x8 FP16 GEMM for all projections & Flash-ViT attention!
// 100% MIT clean-room implementation.
#include <metal_stdlib>
using namespace metal;

#define IV_D 1024
#define IV_NH 16
#define IV_HD 64
#define IV_FFN 4096
#define IV_OUT 2304
#define IV_PATCH 20

static inline float iv_bf16(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float iv_erf(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float e = 1.0f - t * exp(poly);
    return x >= 0.0f ? e : -e;
}
static inline float iv_gelu(float x) { return 0.5f * x * (1.0f + iv_erf(x * 0.7071067811865476f)); }

static inline float iv_tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint i = 0; i < (nthr + 31u) >> 5; ++i) r += sh[i];
    return r;
}

// 1. Patchify float RGB [H, W, 3] (already normalized to [-1, 1] and padded to multiple of 40) -> [N_patch, 1200]
//    Order per patch: [20, 20, 3] = (dy * 20 + dx) * 3 + c (exact Thinking Machines / CpuVisionEncoder layout)
//    grid (N_patch, 1, 1), 256 threads
kernel void iv_patchify(
    device const float* rgb [[buffer(0)]],
    device float* pix       [[buffer(1)]],
    constant int& W         [[buffer(2)]],
    constant int& gw        [[buffer(3)]],
    uint pidx [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int py = int(pidx) / gw, px = int(pidx) % gw;
    for (int e = int(tid); e < 1200; e += 256) {
        const int px_idx = e / 3;
        const int c = e - px_idx * 3;
        const int dy = px_idx / 20, dx = px_idx - dy * 20;
        pix[(size_t)pidx * 1200 + e] = rgb[((size_t)(py * 20 + dy) * W + (px * 20 + dx)) * 3 + c];
    }
}

// 2. Prepend CLS token + add bilinearly-interpolated 32x32 position_embedding
//    tokens_out: [1 + N_patch, 1024] F32; patch_emb: [N_patch, 1024] F32
//    grid (1 + N_patch, 1, 1), 256 threads
kernel void iv_add_pos_cls(
    device const float* patch_emb   [[buffer(0)]],
    device const ushort* cls_emb    [[buffer(1)]],  // [1024] BF16
    device const ushort* cls_pos    [[buffer(2)]],  // [1024] BF16
    device const ushort* pos_grid   [[buffer(3)]],  // [32*32, 1024] BF16
    device float* out_buf           [[buffer(4)]],  // [1 + N_patch, 1024]
    constant int& gh                [[buffer(5)]],
    constant int& gw                [[buffer(6)]],
    uint idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    device float* dst = out_buf + (size_t)idx * IV_D;
    if (idx == 0) {
        for (uint c = tid; c < IV_D; c += 256) dst[c] = iv_bf16(cls_emb[c]) + iv_bf16(cls_pos[c]);
        return;
    }
    const int p = int(idx) - 1;
    const int y = p / gw, x = p % gw;
    const float sy = 32.0f / float(gh), sx = 32.0f / float(gw);
    float fy = sy * (float(y) + 0.5f) - 0.5f;
    float fx = sx * (float(x) + 0.5f) - 0.5f;
    fy = clamp(fy, 0.0f, 31.0f);
    fx = clamp(fx, 0.0f, 31.0f);
    const int y0 = int(floor(fy)), x0 = int(floor(fx));
    const int y1 = min(y0 + 1, 31), x1 = min(x0 + 1, 31);
    const float ly = fy - float(y0), lx = fx - float(x0);
    device const float* pe = patch_emb + (size_t)p * IV_D;
    for (uint c = tid; c < IV_D; c += 256) {
        const float a = iv_bf16(pos_grid[((size_t)y0 * 32 + x0) * IV_D + c]);
        const float b = iv_bf16(pos_grid[((size_t)y0 * 32 + x1) * IV_D + c]);
        const float cc = iv_bf16(pos_grid[((size_t)y1 * 32 + x0) * IV_D + c]);
        const float d = iv_bf16(pos_grid[((size_t)y1 * 32 + x1) * IV_D + c]);
        const float pos_v = (1.0f - ly) * ((1.0f - lx) * a + lx * b) + ly * ((1.0f - lx) * cc + lx * d);
        dst[c] = pe[c] + pos_v;
    }
}

// 3. LayerNorm with BF16 weight & bias: [S, 1024] -> [S, 1024] F32
// grid (S, 1, 1), 256 threads
kernel void iv_ln_bf16(
    device const float* x_in  [[buffer(0)]],
    device const ushort* w    [[buffer(1)]],
    device const ushort* b    [[buffer(2)]],
    device float* out_buf     [[buffer(3)]],
    constant float& eps       [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* xr = x_in + (size_t)row * IV_D;
    float s = 0.0f;
    for (uint i = tid; i < IV_D; i += 256) s += xr[i];
    const float mean = iv_tg_sum(s, sh, tid, 256) * (1.0f / float(IV_D));
    float v = 0.0f;
    for (uint i = tid; i < IV_D; i += 256) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrt(iv_tg_sum(v, sh, tid, 256) * (1.0f / float(IV_D)) + eps);
    device float* yr = out_buf + (size_t)row * IV_D;
    for (uint i = tid; i < IV_D; i += 256) {
        yr[i] = (xr[i] - mean) * inv * iv_bf16(w[i]) + iv_bf16(b[i]);
    }
}

// 4. Add BF16 bias + optional GELU: [M, N]
kernel void iv_bias_gelu(
    device float* x           [[buffer(0)]],
    device const ushort* bias [[buffer(1)]],
    constant int& M           [[buffer(2)]],
    constant int& N           [[buffer(3)]],
    constant int& has_bias    [[buffer(4)]],
    constant int& do_gelu     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int tot = M * N;
    if (int(gid) >= tot) return;
    const int n = int(gid) % N;
    float v = x[gid] + (has_bias ? iv_bf16(bias[n]) : 0.0f);
    if (do_gelu) v = iv_gelu(v);
    x[gid] = v;
}

// 5. LayerScale + Residual Add: res[i] += (x[i] + bias[n]) * ls_gamma[n]
kernel void iv_ls_res_add(
    device float* res          [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device const ushort* bias  [[buffer(2)]],
    device const ushort* gamma [[buffer(3)]],
    constant int& M            [[buffer(4)]],
    constant int& N            [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int tot = M * N;
    if (int(gid) >= tot) return;
    const int n = int(gid) % N;
    res[gid] += (x[gid] + iv_bf16(bias[n])) * iv_bf16(gamma[n]);
}

// 6. Axial 2D RoPE on Q or K [S, 16, 64] (token 0 = CLS untouched; tokens 1..N_patch use 1-indexed (w, h), theta=10000)
//    Exact pairwise interleaved layout: dims 0..31 = 16 pairs for pos_w = px+1; dims 32..63 = 16 pairs for pos_h = py+1!
//    grid (S, 16, 1), 32 threads
kernel void iv_rope2d(
    device float* x_buf [[buffer(0)]],
    constant int& gw    [[buffer(1)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int s = int(tgp.x), h = int(tgp.y);
    if (s == 0) return;  // CLS token has pos=(0,0) -> cos=1, sin=0
    const int p = s - 1;
    const int pos_h = (p / gw) + 1;
    const int pos_w = (p % gw) + 1;
    device float* xp = x_buf + ((size_t)s * IV_NH + h) * IV_HD;
    const int half_ax = tid < 16 ? 0 : 1;
    const int idx = int(tid & 15u);
    const float freq = 1.0f / pow(10000.0f, float(2 * idx) / 32.0f);
    const float ang = float(half_ax == 0 ? pos_w : pos_h) * freq;
    const float cs = precise::cos(ang), sn = precise::sin(ang);
    const int j = half_ax * 32 + 2 * idx;
    const float a = xp[j];
    const float b = xp[j + 1];
    xp[j]     = a * cs - b * sn;
    xp[j + 1] = b * cs + a * sn;
}

// 7. Full Bidirectional ViT Attention (S tokens, 16 heads x 64)
//    grid (S, 16, 1), 128 threads (4 SGs x 32 lanes, 2 dims/lane)
kernel void iv_attn(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O       [[buffer(3)]],
    constant int& S       [[buffer(4)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float sh_m[4];
    threadgroup float sh_l[4];
    threadgroup float sh_o[4 * IV_HD];

    const int q_idx = int(tgp.x), h = int(tgp.y);
    const float scale = 0.125f;  // 1 / sqrt(64)
    const float2 qv = ((device const float2*)(Q + ((size_t)q_idx * IV_NH + h) * IV_HD))[lane] * scale;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float2 o_acc = float2(0.0f);

    for (int k = int(sg); k < S; k += 4) {
        const size_t off = ((size_t)k * IV_NH + h) * IV_HD;
        const float2 kv = ((device const float2*)(K + off))[lane];
        const float sc = simd_sum(dot(qv, kv));
        const float mn = max(m_max, sc);
        const float corr = exp(m_max - mn);
        const float p = exp(sc - mn);
        l_sum = l_sum * corr + p;
        m_max = mn;
        const float2 vv = ((device const float2*)(V + off))[lane];
        o_acc = o_acc * corr + p * vv;
    }

    if (lane == 0) {
        sh_m[sg] = m_max;
        sh_l[sg] = l_sum;
    }
    *((threadgroup float2*)(sh_o + sg * IV_HD + lane * 2)) = o_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const float gm = max(max(sh_m[0], sh_m[1]), max(sh_m[2], sh_m[3]));
        float gl = 0.0f;
        float2 go = float2(0.0f);
        for (int s = 0; s < 4; ++s) {
            const float w = exp(sh_m[s] - gm);
            gl += sh_l[s] * w;
            go += (*((threadgroup float2*)(sh_o + s * IV_HD + lane * 2))) * w;
        }
        ((device float2*)(O + ((size_t)q_idx * IV_NH + h) * IV_HD))[lane] = go * (1.0f / gl);
    }
}

// 8. 2x2 Spatial Merge of patch tokens (drop CLS token 0):
//    x_in: [1 + gh*gw, 1024] -> merged_out: [(gh/2)*(gw/2), 4096]
//    grid ((gh/2)*(gw/2), 1, 1), 256 threads
kernel void iv_merge2x2(
    device const float* x_in [[buffer(0)]],
    device float* out_buf    [[buffer(1)]],
    constant int& gw         [[buffer(2)]],
    uint cell [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int ow = gw >> 1;
    const int oy = int(cell) / ow, ox = int(cell) % ow;
    const int p00 = (oy * 2 + 0) * gw + (ox * 2 + 0);
    const int p01 = (oy * 2 + 0) * gw + (ox * 2 + 1);
    const int p10 = (oy * 2 + 1) * gw + (ox * 2 + 0);
    const int p11 = (oy * 2 + 1) * gw + (ox * 2 + 1);
    device float* dst = out_buf + (size_t)cell * 4096;
    for (uint c = tid; c < IV_D; c += 256) {
        dst[0 * IV_D + c] = x_in[(size_t)(1 + p00) * IV_D + c];
        dst[1 * IV_D + c] = x_in[(size_t)(1 + p01) * IV_D + c];
        dst[2 * IV_D + c] = x_in[(size_t)(1 + p10) * IV_D + c];
        dst[3 * IV_D + c] = x_in[(size_t)(1 + p11) * IV_D + c];
    }
}
