// tools/vision/own/vit_metal.metal - the Qwen vision tower's kernels on Metal (own-stack phase 3).  The counterparts of vit_gpu.cu's kernels, one for one,
// with the same arithmetic: float32 activations, 16-bit GEMM inputs of the weight's own type (F16 or BF16), q / k / v and the attention scores in F16,
// float32 accumulation.  Embedded into strata-vision at build time and compiled when the reader starts (vit_metal.mm); the tensor-op products are in
// vit_metal_mpp.metal (Metal 4 only).
//
// Every kernel takes its buffers at [[buffer(0..)]] and its sizes in a small constant struct at [[buffer(15)]].
#include <metal_stdlib>
using namespace metal;

// the 16-bit kinds (vit_metal.mm's Kind): 0 float32, 1 F16, 2 BF16, 3 a BF16 matrix held as F16 with its inputs written as F16 (clamped to +-65,504)
constant int K_F32 = 0;
constant int K_F16 = 1;
constant int K_HF = 3;

inline ushort f_to_bf16(float f) {       // round to nearest even; NaN stays NaN
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}
inline float bf16_to_f(ushort b) { return as_type<float>(uint(b) << 16); }
inline ushort f_to_f16bits(float f) { return as_type<ushort>(half(f)); }       // float -> half rounds to nearest even
inline float f16bits_to_f(ushort h) { return float(as_type<half>(h)); }

inline void store_kind(device void* out, ulong i, float v, int kind) {
    if (kind == K_F32) ((device float*) out)[i] = v;
    else if (kind == K_F16) ((device ushort*) out)[i] = f_to_f16bits(v);
    else if (kind == K_HF) ((device ushort*) out)[i] = f_to_f16bits(v > 65504.0f ? 65504.0f : v < -65504.0f ? -65504.0f : v);   // NaN stays NaN (vit_f16.h f16_clamp)
    else ((device ushort*) out)[i] = f_to_bf16(v);
}

// threadgroup-wide sum / max of one float per thread (threads a multiple of 32, at most 1024)
inline float tg_sum(float v, threadgroup float* red, ushort tid, ushort nt) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort nw = (nt + 31) / 32;
    v = (tid & 31) < nw ? red[tid & 31] : 0.0f;
    return simd_sum(v);
}
inline float tg_max(float v, threadgroup float* red, ushort tid, ushort nt) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort nw = (nt + 31) / 32;
    v = (tid & 31) < nw ? red[tid & 31] : -FLT_MAX;
    return simd_max(v);
}

struct PatchPostArgs { int G, pw, ph, nbx, D; };
// x[t] += patch bias + the position table resized (bilinear, align corners: ggml's maths, in float) to the token's grid position.  One group a token.
kernel void vk_patch_post(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* pos [[buffer(2)]],
                          constant PatchPostArgs& a [[buffer(15)]], uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort nt [[threads_per_threadgroup]]) {
    const int G = a.G, pw = a.pw, ph = a.ph, D = a.D;
    const int b = int(t) / 4, r = int(t) % 4;
    const int ty = 2 * (b / a.nbx) + r / 2, tx = 2 * (b % a.nbx) + r % 2;
    float sfx = (float) pw / G, sfy = (float) ph / G;
    if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
    if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
    const float fy = (float) ty / sfy, fx = (float) tx / sfx;
    int y0 = (int) floor(fy), x0 = (int) floor(fx), y1 = y0 + 1, x1 = x0 + 1;
    y0 = max(0, min(y0, G - 1)); y1 = max(0, min(y1, G - 1));
    x0 = max(0, min(x0, G - 1)); x1 = max(0, min(x1, G - 1));
    const float dy = fmax(0.0f, fmin(fy - (float) y0, 1.0f)), dx = fmax(0.0f, fmin(fx - (float) x0, 1.0f));
    device const float* pa = pos + ((ulong) y0 * G + x0) * D;
    device const float* pb = pos + ((ulong) y0 * G + x1) * D;
    device const float* pc = pos + ((ulong) y1 * G + x0) * D;
    device const float* pd = pos + ((ulong) y1 * G + x1) * D;
    device float* xr = x + (ulong) t * D;
    for (int i = tid; i < D; i += nt)
        xr[i] += bias[i] + (pa[i] * (1 - dx) * (1 - dy) + pb[i] * dx * (1 - dy) + pc[i] * (1 - dx) * dy + pd[i] * dx * dy);
}

struct NormArgs { int D, kind; float eps; };
// out = LayerNorm(x) * w + b, into `kind`.  One group a row.
kernel void vk_layernorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
                         device void* out [[buffer(3)]], constant NormArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    const int D = a.D;
    device const float* xr = x + (ulong) row * D;
    float s = 0.0f;
    for (int i = tid; i < D; i += nt) s += xr[i];
    const float mean = tg_sum(s, red, tid, nt) / (float) D;
    float v = 0.0f;
    for (int i = tid; i < D; i += nt) { const float d = xr[i] - mean; v += d * d; }
    const float scale = rsqrt(tg_sum(v, red, tid, nt) / (float) D + a.eps);
    for (int i = tid; i < D; i += nt) store_kind(out, (ulong) row * D + i, (xr[i] - mean) * scale * w[i] + b[i], a.kind);
}

struct RopeArgs { int row0, nbx, heads, hd; float theta_scale, qscale; };
// y [rows][3D] (the fused qkv product of the tokens row0 ..) + bias -> q (scaled), k (rotary applied), v, each [N][D] in F16.  Pair j of a head is
// (j, j + hd/2); j < hd/4 turns with the token's row, the others with its column.  One group a row.
kernel void vk_qkv_rope(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device ushort* q [[buffer(2)]],
                        device ushort* k [[buffer(3)]], device ushort* v [[buffer(4)]], constant RopeArgs& a [[buffer(15)]],
                        uint gr [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int t = a.row0 + int(gr), heads = a.heads, hd = a.hd, D = heads * hd, half_ = hd / 2, quarter = hd / 4;
    const int b = t / 4, r = t % 4;
    const int ty = 2 * (b / a.nbx) + r / 2, tx = 2 * (b % a.nbx) + r % 2;
    device const float* yr = y + (ulong) gr * 3 * D;
    for (int i = tid; i < 2 * heads * half_; i += nt) {
        const int part = i / (heads * half_), rem = i % (heads * half_), h = rem / half_, j = rem % half_;
        const int base = part * D + h * hd;
        const float av = yr[base + j] + bias[base + j], cv = yr[base + j + half_] + bias[base + j + half_];
        const float freq = precise::pow(a.theta_scale, (float) (j % quarter));
        float cs;
        const float sn = precise::sincos((float) (j < quarter ? ty : tx) * freq, cs);
        device ushort* dst = (part == 0 ? q : k) + (ulong) t * D + h * hd;
        const float sc = part == 0 ? a.qscale : 1.0f;
        dst[j] = f_to_f16bits((av * cs - cv * sn) * sc);
        dst[j + half_] = f_to_f16bits((av * sn + cv * cs) * sc);
    }
    for (int e = tid; e < D; e += nt) v[(ulong) t * D + e] = f_to_f16bits(yr[2 * D + e] + bias[2 * D + e]);
}

struct SoftmaxArgs { int n, ld; };
// In place: a row of n F16 scores becomes the row of probabilities, F16.  One group a row; PER scores a thread held in registers (threads * PER >= n).
template <int PER>
kernel void vk_softmax_rows(device ushort* S [[buffer(0)]], constant SoftmaxArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device ushort* s = S + (ulong) row * a.ld;
    const int n = a.n;
    float v[PER];
    float m = -FLT_MAX;
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        v[k] = j < n ? f16bits_to_f(s[j]) : -FLT_MAX;
        m = fmax(m, v[k]);
    }
    m = tg_max(m, red, tid, nt);
    float sum = 0.0f;
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        v[k] = j < n ? precise::exp(v[k] - m) : 0.0f;
        sum += v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / tg_sum(sum, red, tid, nt);
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        if (j < n) s[j] = f_to_f16bits(v[k] * inv);
    }
}
template [[host_name("vk_softmax_rows_2")]] kernel decltype(vk_softmax_rows<2>) vk_softmax_rows<2>;
template [[host_name("vk_softmax_rows_8")]] kernel decltype(vk_softmax_rows<8>) vk_softmax_rows<8>;
template [[host_name("vk_softmax_rows_16")]] kernel decltype(vk_softmax_rows<16>) vk_softmax_rows<16>;
template [[host_name("vk_softmax_rows_64")]] kernel decltype(vk_softmax_rows<64>) vk_softmax_rows<64>;

// The same for a row too long to hold in registers: three passes over the row in memory.
kernel void vk_softmax_rows_long(device ushort* S [[buffer(0)]], constant SoftmaxArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device ushort* s = S + (ulong) row * a.ld;
    const int n = a.n;
    float m = -FLT_MAX;
    for (int j = tid; j < n; j += nt) m = fmax(m, f16bits_to_f(s[j]));
    m = tg_max(m, red, tid, nt);
    float sum = 0.0f;
    for (int j = tid; j < n; j += nt) sum += precise::exp(f16bits_to_f(s[j]) - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / tg_sum(sum, red, tid, nt);
    for (int j = tid; j < n; j += nt) s[j] = f_to_f16bits(precise::exp(f16bits_to_f(s[j]) - m) * inv);
}

struct EltArgs { ulong n; int kind; int D; };
// n floats to the 16-bit kind (grid-stride)
kernel void vk_cast(device const float* in [[buffer(0)]], device void* out [[buffer(1)]], constant EltArgs& a [[buffer(15)]],
                    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (ulong i = gid; i < a.n; i += gsz) store_kind(out, i, in[i], a.kind);
}
// y[i] rounded to the 16-bit kind and back (STRATA_VIT_REF_ROUND, a diagnostic)
kernel void vk_round(device float* y [[buffer(0)]], constant EltArgs& a [[buffer(15)]], uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (ulong i = gid; i < a.n; i += gsz) {
        const float v = y[i];
        y[i] = a.kind == K_F16 ? float(half(v)) : bf16_to_f(f_to_bf16(v));
    }
}
// x[r] += y[r] + bias.  One group a row.
kernel void vk_add_bias_res(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* bias [[buffer(2)]],
                            constant EltArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) x[base + i] += y[base + i] + bias[i];
}
// out = gelu_tanh(y + bias) into `kind`.  One group a row of D.
kernel void vk_bias_gelu_cast(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device void* out [[buffer(2)]],
                              constant EltArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) {
        const float x = y[base + i] + bias[i];
        const float g = 0.5f * x * (1.0f + precise::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
        store_kind(out, base + i, g, a.kind);
    }
}
// y[r] += bias.  One group a row of D.
kernel void vk_add_bias(device float* y [[buffer(0)]], device const float* bias [[buffer(1)]], constant EltArgs& a [[buffer(15)]],
                        uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) y[base + i] += bias[i];
}

// ---- the plain GEMM (any Apple GPU): Y[z][r][o] = sum_k X[z][r][k] * W[z](o, k), float32 accumulation.
//   X: [rows][in], row stride ldx.   W: WN false: [out][in], row stride ldw (k contiguous); WN true: [in][out], row stride ldw (o contiguous).
//   Y: [rows][out], row stride ldy.  Batch z (grid z): the bases move by sx, sw, sy elements.
// A 64 x 64 tile a group (4 simdgroups of 32 x 32), k in steps of 32, operands staged as float in threadgroup memory (exact for 16-bit inputs).
struct GemmArgs { uint rows, in, out, ldx, ldw, ldy; ulong sx, sw, sy; };

template <typename T> inline float ld_e(device const T* p, ulong i);
template <> inline float ld_e<float>(device const float* p, ulong i) { return p[i]; }
template <> inline float ld_e<half>(device const half* p, ulong i) { return float(p[i]); }
template <> inline float ld_e<ushort>(device const ushort* p, ulong i) { return bf16_to_f(p[i]); }      // ushort = BF16 here
template <typename T> inline void st_e(device T* p, ulong i, float v);
template <> inline void st_e<float>(device float* p, ulong i, float v) { p[i] = v; }
template <> inline void st_e<half>(device half* p, ulong i, float v) { p[i] = half(v); }

template <typename TI, typename TO, bool WN>
kernel void vk_gemm(device const TI* X0 [[buffer(0)]], device const TI* W0 [[buffer(1)]], device TO* Y0 [[buffer(2)]], constant GemmArgs& a [[buffer(15)]],
                    uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[4096];   // As [64][32] then Bs [32][64]; reused for the result tile [64][64]
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + 2048;
    device const TI* X = X0 + a.sx * tgp.z;
    device const TI* W = W0 + a.sw * tgp.z;
    device TO* Y = Y0 + a.sy * tgp.z;
    const uint o0 = tgp.x * 64u, r0 = tgp.y * 64u;
    const uint sr = (uint(sg) >> 1) * 32u, so = (uint(sg) & 1u) * 32u;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < a.in; k0 += 32u) {
        for (uint idx = tid; idx < 2048u; idx += 128u) {
            const uint r = idx >> 5, kk = idx & 31u;
            const uint gr = r0 + r, gk = k0 + kk;
            As[idx] = (gr < a.rows && gk < a.in) ? ld_e<TI>(X, ulong(gr) * a.ldx + gk) : 0.0f;
        }
        if (!WN) {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint o = idx >> 5, kk = idx & 31u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[kk * 64u + o] = (go < a.out && gk < a.in) ? ld_e<TI>(W, ulong(go) * a.ldw + gk) : 0.0f;
            }
        } else {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint kk = idx >> 6, o = idx & 63u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[idx] = (go < a.out && gk < a.in) ? ld_e<TI>(W, ulong(gk) * a.ldw + go) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 4u; ++k8) {
            simdgroup_float8x8 fa[4], fb[4];
            for (uint i = 0; i < 4u; ++i) simdgroup_load(fa[i], As + (sr + i * 8u) * 32u + k8 * 8u, 32);
            for (uint j = 0; j < 4u; ++j) simdgroup_load(fb[j], Bs + k8 * 8u * 64u + so + j * 8u, 64);
            for (uint i = 0; i < 4u; ++i)
                for (uint j = 0; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j) simdgroup_store(acc[i][j], smem + (sr + i * 8u) * 64u + so + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 4096u; idx += 128u) {
        const uint r = idx >> 6, o = idx & 63u;
        const uint gr = r0 + r, go = o0 + o;
        if (gr >= a.rows || go >= a.out) continue;
        st_e<TO>(Y, ulong(gr) * a.ldy + go, smem[idx]);
    }
}
// The float32 patch product (7 GFLOP for a 1,024-token picture): one output a thread, a plain sequential FMA chain over k.  Not the tiled kernel above:
// the first run of vk_gemm costs the process about 136 MB of GPU-owned memory on an M5 Max (RESULTS-OWN-3V.md D5), which a Mac with the tensor-op
// products then never needs to pay.  Grid (ceil(out / 64), rows), 64 threads.
kernel void vk_gemm_seq_f32(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]], device float* Y [[buffer(2)]],
                            constant GemmArgs& a [[buffer(15)]], uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint o = tgp.x * 64u + uint(tid), r = tgp.y;
    if (o >= a.out || r >= a.rows) return;
    device const float* x = X + ulong(r) * a.ldx;
    device const float* w = W + ulong(o) * a.ldw;
    float s = 0.0f;
    for (uint k = 0; k < a.in; ++k) s = fma(x[k], w[k], s);
    Y[ulong(r) * a.ldy + o] = s;
}

// weight products: 16-bit (or float32) X and W, float32 Y
template [[host_name("vk_gemm_f32")]] kernel decltype(vk_gemm<float, float, false>) vk_gemm<float, float, false>;
template [[host_name("vk_gemm_f16")]] kernel decltype(vk_gemm<half, float, false>) vk_gemm<half, float, false>;
template [[host_name("vk_gemm_bf16")]] kernel decltype(vk_gemm<ushort, float, false>) vk_gemm<ushort, float, false>;
// attention: S = Q K^T (F16 in, F16 out) and context = P V (F16 in, V o-contiguous, float32 out)
template [[host_name("vk_gemm_s")]] kernel decltype(vk_gemm<half, half, false>) vk_gemm<half, half, false>;
template [[host_name("vk_gemm_pv")]] kernel decltype(vk_gemm<half, float, true>) vk_gemm<half, float, true>;
