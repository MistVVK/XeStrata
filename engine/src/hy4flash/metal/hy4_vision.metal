// hy4_vision.metal - the Hy4-Flash image path on Apple Silicon (MSL): SigLIP-shaped ViT (27 pre-LN blocks, hidden 1152, 16 heads x 72) + rmsnorm_mlp_adapter, and the LM-side kernels a
// prefill with pictures needs (image-row scatter, 3D rope fix-ups for the MLA q / k and the IndexPool q / pooled keys).
// Compiled together with hy4_mpp.metal (the Metal 4 matmul2d GEMM `mpp_mm_*` is used for every ViT linear layer; weights are already fp16 [N][K] in the Vision shard).
// Same numerics as the HIP version (hy4_vision.hip.h): fp16 GEMM inputs, fp32 accumulation, fp32 residual stream, fp32 LayerNorm statistics, fp16 attention P.
#define VD 1152
#define VHD 72

// tanh of a large argument overflows to NaN in MSL: clamp (tanh(10) = 1 - 4e-9)
static inline float v_gelu_tanh(float x) { return 0.5f * x * (1.0f + tanh(clamp(0.7978845608028654f * (x + 0.044715f * x * x * x), -10.0f, 10.0f))); }
// MSL has no erf: erfc() Chebyshev fit (Numerical Recipes, fractional error < 1.2e-7)
static inline float v_erf(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float r = t * exp(poly);  // erfc(z)
    const float e = 1.0f - r;
    return x >= 0.0f ? e : -e;
}
static inline float v_gelu_erf(float x) { return 0.5f * x * (1.0f + v_erf(x * 0.7071067811865476f)); }

static inline float v_tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint i = 0; i < (nthr + 31) / 32; ++i) r += sh[i];
    return r;
}

// patchify: rgb u8 [h][w][3] -> half [N][768] = (C, 16, 16) flattened per patch, (v/255 - 0.5)/0.5, raster patch order. grid N, 256 threads
kernel void v_patchify(device const uchar* rgb [[buffer(0)]], device half* pix [[buffer(1)]], constant int& w [[buffer(2)]], constant int& gw [[buffer(3)]], uint pidx [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const int py = pidx / gw, px = pidx % gw;
    for (int e = tid; e < 768; e += 256) {
        const int c = e >> 8, dy = (e >> 4) & 15, dx = e & 15;
        const float v = float(rgb[((size_t) (py * 16 + dy) * w + px * 16 + dx) * 3 + c]) * (1.0f / 255.0f);
        pix[(size_t) pidx * 768 + e] = half((v - 0.5f) / 0.5f);
    }
}

// learned position grid (row 0 = cls, unused; rows 1.. = edge x edge grid, fp16) resampled bilinearly to gh x gw (torch F.interpolate, align_corners=False). grid N, 256 threads
kernel void v_posemb(device const half* tab [[buffer(0)]], device float* out [[buffer(1)]], constant int& gh [[buffer(2)]], constant int& gw [[buffer(3)]], constant int& edge [[buffer(4)]],
                     constant float& sy [[buffer(5)]], constant float& sx [[buffer(6)]], uint p [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int y = p / gw, x = p % gw;
    float fy = sy * (float(y) + 0.5f) - 0.5f, fx = sx * (float(x) + 0.5f) - 0.5f;
    if (fy < 0.f) fy = 0.f;
    if (fx < 0.f) fx = 0.f;
    int y0 = int(fy), x0 = int(fx);
    const int y1 = y0 < edge - 1 ? y0 + 1 : y0, x1 = x0 < edge - 1 ? x0 + 1 : x0;
    if (y0 > edge - 1) y0 = edge - 1;
    if (x0 > edge - 1) x0 = edge - 1;
    const float ly = fy - float(y0), lx = fx - float(x0);
    device const half* t = tab + VD;
    for (int c = tid; c < VD; c += 256) {
        const float a = float(t[((size_t) y0 * edge + x0) * VD + c]), b = float(t[((size_t) y0 * edge + x1) * VD + c]);
        const float cc = float(t[((size_t) y1 * edge + x0) * VD + c]), d = float(t[((size_t) y1 * edge + x1) * VD + c]);
        out[(size_t) p * VD + c] = (1.f - ly) * ((1.f - lx) * a + lx * b) + ly * ((1.f - lx) * cc + lx * d);
    }
}

// LayerNorm (biased variance, affine): fp32 [N][1152] -> half. grid N, 256 threads
kernel void v_ln(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]], device half* out [[buffer(3)]], constant float& eps [[buffer(4)]],
                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* xr = x + (size_t) row * VD;
    float s = 0;
    for (uint i = tid; i < VD; i += 256) s += xr[i];
    const float mean = v_tg_sum(s, sh, tid, 256) / float(VD);
    float v = 0;
    for (uint i = tid; i < VD; i += 256) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrt(v_tg_sum(v, sh, tid, 256) / float(VD) + eps);
    for (uint i = tid; i < VD; i += 256) out[(size_t) row * VD + i] = half((xr[i] - mean) * inv * w[i] + b[i]);
}

// RMSNorm (perceive.norm) written straight into the 2x2-merge matrix A [(gh/2)*(gw/2)][1152*4], column c*4 + dy*2 + dx. grid N, 256 threads
kernel void v_rms_merge(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device half* A [[buffer(2)]], constant int& gw [[buffer(3)]], constant float& eps [[buffer(4)]],
                        uint p [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    const int y = p / gw, xx = p % gw;
    device const float* xr = x + (size_t) p * VD;
    float ss = 0;
    for (uint i = tid; i < VD; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrt(v_tg_sum(ss, sh, tid, 256) / float(VD) + eps);
    const size_t cell = (size_t) (y >> 1) * (gw >> 1) + (xx >> 1);
    const int sub = (y & 1) * 2 + (xx & 1);
    for (uint i = tid; i < VD; i += 256) A[cell * (size_t) (VD * 4) + (size_t) i * 4 + sub] = half(xr[i] * inv * w[i]);
}

// GEMM epilogue on the fp32 result C [M][N] of mpp_mm: + bias, then
//   0 EPI_H half out | 1 GELU(tanh) half | 2 GELU(erf) half | 3 residual add into Yf | 4 Yf = v + pe | 5 Yf[(m + m/ow)][n] = v (merged cell -> LM row, newline column skipped)
// grid ceil(M*N/256), 256 threads
template <int EPI>
kernel void v_epi(device const float* C [[buffer(0)]], device const float* bias [[buffer(1)]], device half* Yh [[buffer(2)]], device float* Yf [[buffer(3)]], device const float* pe [[buffer(4)]],
                  constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& ow [[buffer(7)]], constant int& has_bias [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    const size_t i = gid;
    if (i >= (size_t) M * N) return;
    const int m = (int) (i / N), n = (int) (i % N);
    const float v = C[i] + (has_bias ? bias[n] : 0.f);
    if (EPI == 0) Yh[i] = half(v);
    else if (EPI == 1) Yh[i] = half(v_gelu_tanh(v));
    else if (EPI == 2) Yh[i] = half(v_gelu_erf(v));
    else if (EPI == 3) Yf[i] += v;
    else if (EPI == 4) Yf[i] = v + pe[i];
    else if (EPI == 5) Yf[(size_t) (m + m / ow) * N + n] = v;
    else if (EPI == 6) Yh[((size_t) (n / VHD) * ow + m) * VHD + (n % VHD)] = half(v);   // per-head layout [h][ow = Npad][72] (q, k of the MPP attention)
    else Yh[((size_t) (n / VHD) * VHD + (n % VHD)) * ow + m] = half(v);                  // per-head transposed [h][72][ow = Npad] (v of the MPP attention)
}
#define VEPI(E) template [[host_name("v_epi_" #E)]] kernel void v_epi<E>(device const float*, device const float*, device half*, device float*, device const float*, constant int&, constant int&, constant int&, constant int&, uint);
VEPI(0) VEPI(1) VEPI(2) VEPI(3) VEPI(4) VEPI(5) VEPI(6) VEPI(7)

// ---- full softmax attention, head dim 72, 16 heads, q/k/v/o half [Npad][1152] (head h at columns 72h; rows >= N are zero padding, Npad a multiple of 64).
// A threadgroup = NSG simdgroups; one simdgroup = 16 queries of one head. The K and V tiles (64 keys x 72, fp16) are loaded cooperatively into threadgroup memory once per threadgroup, then each
// simdgroup runs S = Q K^T on simdgroup_matrix (2 x 4 8x8 tiles per 32 keys, 9 k-steps), the online softmax in the fragment layout (a lane holds 2 consecutive columns of one row of each 8x8 tile; the
// 4 lanes of a row are xor 1 and xor 8), and uses P (half) straight as the A operand of P V.
#ifndef VA_KJ
#define VA_KJ 4  // 8-key fragments per inner block
#endif
#define VA_NSG 4
#define VA_LD 72
kernel void v_attn(device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]], device const half* V [[buffer(2)]], device half* O [[buffer(3)]], constant int& N [[buffer(4)]],
                   constant float& scale [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Ks[64 * VA_LD];
    threadgroup half Vs[64 * VA_LD];
    const int h = tgp.y;
    const int q0 = tgp.x * (16 * VA_NSG) + sg * 16;
    const int LD = VD;
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + ((lane / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_half8x8 qf[2][9];
    simdgroup_float8x8 oa[2][9];
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 9; ++d) {
            simdgroup_load(qf[i][d], Q + (size_t) (q0 + 8 * i) * LD + h * VHD + 8 * d, LD);
            oa[i][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    float mr[2] = {-1e30f, -1e30f}, lr[2] = {0.f, 0.f};
    const float sc = scale * 1.4426950408889634f;  // exp2 domain
    for (int kb = 0; kb < N; kb += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64 * 9; i += 32 * VA_NSG) {
            const uint row = i / 9, c = i % 9;
            const size_t src = (size_t) (kb + row) * LD + h * VHD + c * 8;
            *((threadgroup uint4*) (Ks + row * VA_LD + c * 8)) = *((device const uint4*) (K + src));
            *((threadgroup uint4*) (Vs + row * VA_LD + c * 8)) = *((device const uint4*) (V + src));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 64; kk += 8 * VA_KJ) {
            if (kb + kk >= N) break;
            simdgroup_float8x8 s[2][VA_KJ];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < VA_KJ; ++j) s[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d = 0; d < 9; ++d) {
                simdgroup_half8x8 kf[VA_KJ];
                for (int j = 0; j < VA_KJ; ++j) simdgroup_load(kf[j], Ks + (kk + 8 * j) * VA_LD + 8 * d, VA_LD, ulong2(0, 0), true);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < VA_KJ; ++j) simdgroup_multiply_accumulate(s[i][j], qf[i][d], kf[j], s[i][j]);
            }
            simdgroup_half8x8 pf[2][VA_KJ];
#ifdef VA_NOSM
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < VA_KJ; ++j) { pf[i][j].thread_elements()[0] = half(s[i][j].thread_elements()[0]); pf[i][j].thread_elements()[1] = half(s[i][j].thread_elements()[1]); }
            for (int i = 0; i < 0; ++i) {
#else
            for (int i = 0; i < 2; ++i) {
#endif
                float v[VA_KJ][2];
                float mx = -1e30f;
                for (int j = 0; j < VA_KJ; ++j)
                    for (int e = 0; e < 2; ++e) {
                        const int key = kb + kk + 8 * j + fn + e;
                        const float x = key < N ? s[i][j].thread_elements()[e] * sc : -1e30f;
                        v[j][e] = x;
                        mx = max(mx, x);
                    }
                mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
                mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
                const float mn = max(mr[i], mx);
                const float corr = exp2(mr[i] - mn);
                float ps = 0.f;
                for (int j = 0; j < VA_KJ; ++j)
                    for (int e = 0; e < 2; ++e) {
                        const float p = exp2(v[j][e] - mn);
                        ps += p;
                        pf[i][j].thread_elements()[e] = half(p);
                    }
                ps += simd_shuffle_xor(ps, ushort(1));
                ps += simd_shuffle_xor(ps, ushort(8));
                lr[i] = lr[i] * corr + ps;
                mr[i] = mn;
                if (simd_any(corr != 1.0f))
                    for (int d = 0; d < 9; ++d) {
                        oa[i][d].thread_elements()[0] *= corr;
                        oa[i][d].thread_elements()[1] *= corr;
                    }
            }
#ifndef VA_NOPV
#ifdef VA_PVJ
            for (int j = 0; j < VA_KJ; ++j)
                for (int d = 0; d < 9; ++d) {
#else
            for (int d = 0; d < 9; ++d)
                for (int j = 0; j < VA_KJ; ++j) {
#endif
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, Vs + (kk + 8 * j) * VA_LD + 8 * d, VA_LD);
                    for (int i = 0; i < 2; ++i) simdgroup_multiply_accumulate(oa[i][d], pf[i][j], vf, oa[i][d]);
                }
#endif
        }
    }
    for (int i = 0; i < 2; ++i) {
        const float il = 1.0f / lr[i];
        for (int d = 0; d < 9; ++d) {
            const float2 r = float2(oa[i][d].thread_elements()[0], oa[i][d].thread_elements()[1]) * il;
            *((device half2*) (O + (size_t) (q0 + 8 * i + fm) * LD + h * VHD + 8 * d + fn)) = half2(r);
        }
    }
}

// layout probe for the test program: out[lane] = (row, col) of the first element of an 8x8 fragment loaded from an iota matrix (value = 8*row + col)
kernel void v_layout_probe(device float* out [[buffer(0)]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float t[64];
    t[lane] = float(lane);
    t[lane + 32] = float(lane + 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 m;
    simdgroup_load(m, t, 8);
    out[lane * 2] = m.thread_elements()[0];
    out[lane * 2 + 1] = m.thread_elements()[1];
}

kernel void v_fill_newline(device float* rows [[buffer(0)]], device const float* nl [[buffer(1)]], constant int& ow [[buffer(2)]], constant int& D [[buffer(3)]], uint r [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    device float* dst = rows + ((size_t) r * (ow + 1) + ow) * D;
    for (int i = tid; i < D; i += 256) dst[i] = nl[i];
}

// ================================================================ LM side ======================================================================================
#define HID 3072
#define NHC 4
struct VRope {
    int stride, tail_from, delta;
};
static inline int v_mrope_axis(int i) {
    if (i < 30) {
        if (i % 3 == 1) return 1;
        if (i % 3 == 2) return 2;
    }
    return 0;
}
static inline int v_rope_pos(device const int* tab, constant VRope& r, int idx, int axis) { return idx < r.tail_from ? tab[(size_t) axis * r.stride + idx] : idx + r.delta; }
// the same arithmetic as rope_pair of hy4.metal
static inline void v_rope_pair(float x0, float x1, int pos, int i, float theta, thread float& o0, thread float& o1) {
    float freq = 1.0f / pow(theta, float(2 * i) / 64.0f);
    float ang = float(pos) * freq;
    float sn = precise::sin(ang), cs = precise::cos(ang);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

// image rows -> residual stream: x [T][4][3072] fp32; map[t] = row of `rows` or -1. grid T, 256 threads
kernel void v_scatter_rows(device float* x [[buffer(0)]], device const float* rows [[buffer(1)]], device const int* map [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    const int r = map[t];
    if (r < 0) return;
    for (uint d = tid; d < HID; d += 256) {
        const float v = rows[(size_t) r * HID + d];
        for (int h = 0; h < NHC; ++h) x[((size_t) t * NHC + h) * HID + d] = v;
    }
}

// MLA query: qa[t][h][512 + 2i ..] = rope(q_b[t][h][192 + 2i ..], mrope position of token p0 + t). grid M*NH, 32 threads
kernel void v_fix_q(device const float* q_b [[buffer(0)]], device half* qa [[buffer(1)]], constant int& NH [[buffer(2)]], constant int& p0 [[buffer(3)]], constant float& theta [[buffer(4)]],
                    device const int* tab [[buffer(5)]], constant VRope& rd [[buffer(6)]], uint blk [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const int t = blk / NH, h = blk % NH;
    device const float* qp = q_b + ((size_t) t * NH + h) * 256 + 192 + 2 * i;
    float o0, o1;
    v_rope_pair(qp[0], qp[1], v_rope_pos(tab, rd, p0 + t, v_mrope_axis(i)), i, theta, o0, o1);
    *((device half2*) (qa + ((size_t) t * NH + h) * 576 + 512 + 2 * i)) = half2(half(o0), half(o1));
}

// MLA key: cache[slot][512 + 2i ..] = rope(kv_a[t][512 + 2i ..], mrope position of token p0 + t); slot = (p0 + t) % ring, or p0 + t without a ring. grid M, 32 threads
kernel void v_fix_k(device const float* kv_a [[buffer(0)]], device half* cache [[buffer(1)]], constant int& p0 [[buffer(2)]], constant int& ring [[buffer(3)]], constant float& theta [[buffer(4)]],
                    device const int* tab [[buffer(5)]], constant VRope& rd [[buffer(6)]], uint t [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const int pos = p0 + t;
    const int slot = ring > 0 ? pos % ring : pos;
    float o0, o1;
    v_rope_pair(kv_a[(size_t) t * 576 + 512 + 2 * i], kv_a[(size_t) t * 576 + 512 + 2 * i + 1], v_rope_pos(tab, rd, pos, v_mrope_axis(i)), i, theta, o0, o1);
    *((device half2*) (cache + (size_t) slot * 576 + 512 + 2 * i)) = half2(half(o0), half(o1));
}

// indexer query: out[t][h][d] = iq[t][h][d], dims 64..127 roped at the mrope position of token p0 + t. grid (T, 16), 128 threads
kernel void v_fix_iq(device const float* iq [[buffer(0)]], device half* out [[buffer(1)]], constant int& p0 [[buffer(2)]], constant float& theta [[buffer(3)]], device const int* tab [[buffer(4)]],
                     constant VRope& rd [[buffer(5)]], uint2 th [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    const int t = th.x, h = th.y;
    device const float* q = iq + (size_t) t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        const int i = (d - 64) >> 1;
        float o0, o1;
        v_rope_pair(q[64 + 2 * i], q[64 + 2 * i + 1], v_rope_pos(tab, rd, p0 + t, v_mrope_axis(i)), i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t) t * 2048 + h * 128 + d] = half(v);
}

// pooled indexer keys were roped at 4g; rotate pair i of dims 64..127 by (mrope position of token 4g - 4g) * freq_i. grid n, 32 threads
kernel void v_fix_pool(device half* pooled [[buffer(0)]], constant int& g_lo [[buffer(1)]], constant float& theta [[buffer(2)]], device const int* tab [[buffer(3)]], constant VRope& rd [[buffer(4)]],
                       uint bx [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const int g = g_lo + bx;
    const int pn = v_rope_pos(tab, rd, 4 * g, v_mrope_axis(i));
    if (pn == 4 * g) return;
    const float freq = 1.0f / pow(theta, float(2 * i) / 64.0f);
    const float ang = float(pn - 4 * g) * freq;
    const float sn = precise::sin(ang), cs = precise::cos(ang);
    device half* p = pooled + (size_t) g * 128 + 64 + 2 * i;
    const float x0 = float(p[0]), x1 = float(p[1]);
    p[0] = half(x0 * cs - x1 * sn);
    p[1] = half(x0 * sn + x1 * cs);
}

// ================================================================ MPP attention (materialised scores, one query block at a time) ================================================================
// q, k per head [16][Npad][72], v per head transposed [16][72][Npad] (written by the v_epi 6 / 7 epilogues). For a block of Bq queries and ALL heads (grid z = head):
//   S[h][Bq][Npad] = Q_h K_h^T (fp32, Metal 4 matmul2d on the neural accelerators)  ->  v_softmax: P = softmax(scale S) over the first N keys (half, 0 beyond)  ->  C[h][Bq][72] = P V_h  ->  v_attn_out.
struct VHeadArgs {
    int M, N, K;
    int a_head, b_head, c_head;  // element strides between heads
};
template <int TM, int TN, int SG>
kernel void v_mpp_head(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]], constant VHeadArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    device half* Ah = A + (size_t) tg.z * a.a_head;
    device half* Bh = B + (size_t) tg.z * a.b_head;
    device float* Ch = C + (size_t) tg.z * a.c_head;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(Ah, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(Bh, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(Ch, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mC = tC.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mC);
}
template [[host_name("v_mpp_head_128_64_4")]] kernel decltype(v_mpp_head<128, 64, 4>) v_mpp_head<128, 64, 4>;
template [[host_name("v_mpp_head_128_128_4")]] kernel decltype(v_mpp_head<128, 128, 4>) v_mpp_head<128, 128, 4>;

// softmax of rows of S [rows][ld] over the first N columns -> P half [rows][ld] (normalised, 0 beyond N). One threadgroup (256 threads) per row; grid = rows
kernel void v_softmax(device const float* S [[buffer(0)]], device half* P [[buffer(1)]], constant int& N [[buffer(2)]], constant int& ld [[buffer(3)]], constant float& scale [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* s = S + (size_t) row * ld;
    device half* p = P + (size_t) row * ld;
    const float sc = scale * 1.4426950408889634f;
    float mx = -1e30f;
    for (int i = tid; i < N; i += 256) mx = max(mx, s[i]);
    mx = simd_max(mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = sh[0];
    for (int i = 1; i < 8; ++i) mx = max(mx, sh[i]);
    float sum = 0.f;
    for (int i = tid; i < N; i += 256) sum += exp2((s[i] - mx) * sc);
    sum = v_tg_sum(sum, sh, tid, 256);
    const float inv = 1.0f / sum;
    for (int i = tid; i < ld; i += 256) p[i] = i < N ? half(exp2((s[i] - mx) * sc) * inv) : half(0.0f);
}

// C [16][Bq][72] fp32 -> O half [Npad][1152] at query rows q0..q0+Bq. grid (Bq, 16), 96 threads (72 used)
kernel void v_attn_out(device const float* C [[buffer(0)]], device half* O [[buffer(1)]], constant int& Bq [[buffer(2)]], constant int& q0 [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    if (tid < VHD) O[(size_t) (q0 + tg.x) * VD + tg.y * VHD + tid] = half(C[((size_t) tg.y * Bq + tg.x) * VHD + tid]);
}
