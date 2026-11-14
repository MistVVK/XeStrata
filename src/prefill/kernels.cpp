// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/kernels.cpp - see include/strata/prefill/kernels.hpp.  The Xe port of Strata's src/prefill/kernels.cu: work
// shapes, explicit fmaf and summation orders are the CUDA kernels'; a warp is a sub-group of 32 and a 2-D CUDA block
// is linearised the way CUDA numbers its threads (x fastest).  __expf, rsqrtf, powf, cosf, sinf and log1pf are the
// precise SYCL functions.
#include "strata/prefill/kernels.hpp"
#include "gdn_chunk.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/bf16_to_f16.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/rows_cols.hpp"
#include "strata/kernels/rows16.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/per_device.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU prefill_kernels
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128;   // a GDN head's width (the shapes: struct Gdn below)
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline uint16_t bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// The BF16 GEMMs' second operand (STRATA_PREFILL_BF16X2, upstream 61638c1): what the BF16 image `hi` left out of f,
// itself in BF16.  W.hi + W.lo carries ~16 mantissa bits of the activation - decode's FP32 x to within ~1e-5.
inline uint16_t bf_lo(float f, uint16_t hi) { return bf(f - sycl::bit_cast<float>((uint32_t) hi << 16)); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline uint16_t hf(float f) { return kernels::f16_gpu::f16_from_f32(f); }   // the GPU's conversion (f16_gpu.hpp)
// A SwiGLU product for an FP16 GEMM: saturated, so a token with a massive activation cannot turn into inf and then
// NaN in the down projection (decode's q8_1 has room to ~8e6; FP16 ends at 65504).  A NaN stays NaN; finite values
// below 65504 round exactly as before (upstream 7bc505d, f8fe938).
inline uint16_t hf_sat(float f) { return hf(sycl::isnan(f) ? f : sycl::fmin(sycl::fmax(f, -65504.0f), 65504.0f)); }

// work-group sum for up to 1024 work-items, the result broadcast (CUDA's block_sum)
inline float block_sum(const sycl::nd_item<1>& it, float v, float* sh) {
    const sycl::sub_group sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(0), lane = tid & 31, w = tid >> 5;
    v = warp_sum(sg, v);
    sycl::group_barrier(it.get_group());
    if (lane == 0) sh[w] = v;
    sycl::group_barrier(it.get_group());
    const int nw = ((int) it.get_local_range(0) + 31) >> 5;
    float t = (tid < nw) ? sh[tid] : 0.0f;
    if (w == 0) t = warp_sum(sg, t);
    if (tid == 0) sh[0] = t;
    sycl::group_barrier(it.get_group());
    return sh[0];
}

constexpr int CONV_TILE = 64;
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16
template <bool HALF>
void blob_dequant_launch(sycl::queue& q, const uint8_t* blob, uint16_t* gu16, uint16_t* d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    q.parallel_for(sycl::range<1>((size_t) (n_gu + n_d)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];   // one work-item per 4 weights (one code byte)
        auto out = [](float v) -> uint16_t { return HALF ? hf(v) : bf(v); };
        if (i < n_gu) {
            const int64_t row = i / 640, byte = i % 640;
            const uint8_t c = blob[row * 640 + byte];
            const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
            const float d = kernels::f32_from_f16((uint16_t) (sp[0] | (sp[1] << 8)));
            uint16_t* o = gu16 + row * 2560 + byte * 4;
            for (int k = 0; k < 4; ++k) o[k] = out((float) (((c >> (2 * k)) & 3) - 1) * d);
        } else {
            const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
            const uint8_t c = blob[O_D_CODES + row * 160 + byte];
            const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
            const float d = kernels::f32_from_f16((uint16_t) (sp[0] | (sp[1] << 8)));
            uint16_t* o = d16 + row * 640 + byte * 4;
            for (int k = 0; k < 4; ++k) o[k] = out((float) (((c >> (2 * k)) & 3) - 1) * d);
        }
    });
}

}  // namespace

// one work-group per (token, kv head, 64-value group) of 64.  KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    // the K/V cache's one append (kv_q8.hpp), the decode's arithmetic: 2 KV heads of 256
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.page_size = page_size;
    strata::kernels::KvAppend a;
    a.pos0 = pos0;
    a.k = K;
    a.v = V;
    a.n_tok = T;
    a.k_pool = k_pool;
    a.v_pool = v_pool;
    a.k_q = k_q;
    a.v_q = v_q;
    a.k_scale = k_scale;
    a.v_scale = v_scale;
    a.page_table = page_table;
    a.host = host;
    a.stage = stage;
    strata::kernels::kv_append_cells(a, s, stream);
}

void bf16_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    kernels::bf16_to_f16<kernels::F16Range::finite_saturate>(Q(stream), x, y, n);
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = kernels::f32_from_f16(hf(x[i])); });
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const uint16_t h = bf(x[i]);
        y[i] = h;
        if (ylo) ylo[i] = bf_lo(x[i], h);
    });
}

// ---------------------------------------------------------------- hyper-connection
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * LR)), [=](sycl::id<1> i) {
        const float x = lo[i] / (float) HC;
        const float v = x / (1.0f + sycl::exp(-x));
        const uint16_t h = bf(v);
        lo16[i] = h;
        if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
    });
}
// F-1: the row scale only (and the BF16 image); gr_mix_r recomputes r * rs * w itself, in the same order, so the FP32
// copy of the normalized rows (T x 10240 floats) is neither written nor read
void gr_norm_rs(const float* R, const float* w, float eps, float* rs_out, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * HC) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t row = (int64_t) it.get_group(0);   // t * 4 + c
            const int c = (int) (row % HC);
            const int tid = (int) it.get_local_id(0);
            const float* r = R + row * N;
            float ss = 0.0f;
            for (int d = tid; d < N; d += 256) ss += r[d] * r[d];
            const float rs = sycl::rsqrt(block_sum(it, ss, &sh[0]) / (float) N + eps);
            if (tid == 0) rs_out[row] = rs;
            for (int d = tid; d < N; d += 256) {
                const float v = r[d] * rs * w[c * N + d];
                const uint16_t hi = bf(v);
                xn16[row * N + d] = hi;
                if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, hi);
            }
        });
    });
}
void gr_mix_r(const float* R, const float* rs, const float* w, const float* g, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, d = i % N;
        float s = 0.0f;
        for (int c = 0; c < HC; ++c) {
            const int64_t j = t * D + (int64_t) c * N + d;
            const float x = R[j] * rs[t * HC + c] * w[(int64_t) c * N + d];   // gr_norm's value, bit for bit
            s = sycl::fma(x, sigm(g[j]), s);
        }
        s /= (float) HC;
        mixed[i] = s;
        if (mixed16) {
            const uint16_t h = bf(s);
            mixed16[i] = h;
            if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
        }
        if (mixed_h) mixed_h[i] = hf(s);
    });
}
// F-2: gr_write for one row (t, c), then gr_norm_rs's reduction over it with the next half's norm weights - the same
// work-item to element mapping (256 work-items, stride 256) and block_sum, so rs and the BF16 image are the same bits,
// and R is not read back
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w, float eps,
                      float* rs_out, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    constexpr int PER = (N + 255) / 256;
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * HC) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t row = (int64_t) it.get_group(0);   // t * 4 + c
            const int64_t t = row / HC;
            const int c = (int) (row % HC);
            const int tid = (int) it.get_local_id(0);
            float* r = R + row * N;
            const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
            float v[PER];
            float ss = 0.0f;
            int k = 0;
            for (int d = tid; d < N; d += 256, ++k) {
                const float x = sycl::fma(bo[t * N + d], sc, r[d]);
                r[d] = x;
                v[k] = x;
                ss += x * x;
            }
            const float rs = sycl::rsqrt(block_sum(it, ss, &sh[0]) / (float) N + eps);
            if (tid == 0) rs_out[row] = rs;
            k = 0;
            for (int d = tid; d < N; d += 256, ++k) {
                const float x = v[k] * rs * w[c * N + d];
                const uint16_t hi = bf(x);
                xn16[row * N + d] = hi;
                if (xn16_lo) xn16_lo[row * N + d] = bf_lo(x, hi);
            }
        });
    });
}
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t layer, const float* w,
                           float eps, float* rs_out, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    constexpr int PER = (N + 255) / 256;
    const kernels::Cvec& cv = kernels::cvec();
    const kernels::CvecTables tb = kernels::cvec_tables();
    if (tb.dir == nullptr || cv.n_embd != N || cv.hc != HC) throw core::DeviceError("gr_write_cvec_norm_rs: no control vector");
    const float* dir = tb.dir + layer * N;
    const float* s_l = tb.s;
    const int* on = tb.on;
    const int mode = cv.mode;
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(32), h);
        sycl::local_accessor<float, 1> part(sycl::range<1>(256 / WARP), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * HC) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t row = (int64_t) it.get_group(0);   // t * 4 + c
            const int64_t t = row / HC;
            const int c = (int) (row % HC);
            const int tid = (int) it.get_local_id(0);
            float* r = R + row * N;
            const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
            const float sv = s_l[layer];
            const bool steer = *on != 0 && sv != 0.0f;   // uniform over the group
            float x[PER];
            float dot = 0.0f;
            int k = 0;
            for (int d = tid; d < N; d += 256, ++k) {   // gr_write, then cvec_apply's dot
                const float xv = sycl::fma(bo[t * N + d], sc, r[d]);
                x[k] = xv;
                if (steer && mode == 0) dot = sycl::fma(xv, dir[d], dot);
            }
            if (steer && mode == 0) {   // cvec_apply's two-level sum
                for (int o = 16; o > 0; o >>= 1) dot += sycl::permute_group_by_xor(sg, dot, o);
                if ((tid & 31) == 0) part[tid >> 5] = dot;
                sycl::group_barrier(it.get_group());
                if (tid < 32) {
                    float p = tid < 256 / 32 ? part[tid] : 0.0f;
                    for (int o = 16; o > 0; o >>= 1) p += sycl::permute_group_by_xor(sg, p, o);
                    if (tid == 0) part[0] = p;
                }
                sycl::group_barrier(it.get_group());
                dot = part[0] * sv;
            }
            float ss = 0.0f;
            k = 0;
            for (int d = tid; d < N; d += 256, ++k) {   // the update, then gr_norm_rs's sum of squares
                float xv = x[k];
                if (steer) xv = mode == 0 ? sycl::fma(-dot, dir[d], xv) : xv + dir[d];
                r[d] = xv;
                x[k] = xv;
                ss += xv * xv;
            }
            const float rs = sycl::rsqrt(block_sum(it, ss, &sh[0]) / (float) N + eps);
            if (tid == 0) rs_out[row] = rs;
            k = 0;
            for (int d = tid; d < N; d += 256, ++k) {
                const float v = x[k] * rs * w[c * N + d];
                const uint16_t hi = bf(v);
                xn16[row * N + d] = hi;
                if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, hi);
            }
        });
    });
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * D)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / D, c = (i % D) / N, d = i % N;
        R[i] = sycl::fma(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
    });
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * D)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / D, d = i % N;
        R[i] = e[t * N + d];
    });
}

// ---------------------------------------------------------------- GDN
// The kernels of one GDN shape: HK key heads and HV value heads of S, conv channels C = 2 HK S + HV S.  Qwen3.8 is
// <16, 48>, Ornith-1.5 (qwen35moe) and Qwen3.5-9B <16, 32>, Qwen3.5's 2B and 0.8B <16, 16>; the public functions below
// dispatch on the shape.
template <int HK, int HV>
struct Gdn {
    static constexpr int C = 2 * HK * S + HV * S;
    static constexpr int GDN_TB = 8, VPK = HV / HK;
    // the output gate: sigmoid(z) (Qwen3.8) or silu(z) (qwen35moe)
    static float zgate(float z, bool silu) { return silu ? z * sigm(z) : sigm(z); }
    // the recurrence's shared per-token step for one (head, column); returns the column's output before the norm
    template <int COLS>
    static float rec_step(const sycl::nd_item<1>& it, float* s, const float* ht, const float* gate, const float* beta,
                          int64_t t, int head, int qh, int rg, int c, int col, int tid, float* sk, float* sq, float* red) {
        sycl::group_barrier(it.get_group());
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        sycl::group_barrier(it.get_group());
        const float g = sycl::exp(gate[t * HV + head]);
        float kv = 0.0f;
        for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg * COLS + c] = kv;
        sycl::group_barrier(it.get_group());
        const float kv_col = red[c] + red[COLS + c] + red[2 * COLS + c] + red[3 * COLS + c];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        sycl::group_barrier(it.get_group());
        red[rg * COLS + c] = o;
        sycl::group_barrier(it.get_group());
        return (red[c] + red[COLS + c] + red[2 * COLS + c] + red[3 * COLS + c]) * sycl::rsqrt((float) S);
    }

    static void gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
        Q(stream).parallel_for(sycl::range<1>((size_t) (T * HV)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0], t = i / HV, h = i % HV;
            const float v = ab[t * 2 * HV + h] + dt[h];
            gate[i] = (v > 20.0f ? v : sycl::log1p(sycl::exp(v))) * ssm_a[h];
            beta[i] = sigm(ab[t * 2 * HV + HV + h]);
        });
    }

    static void conv(float* hist, const float* qkv, const float* w, float* h, int64_t T, float eps, void* stream) {
        auto& q = Q(stream);
        if (T <= CONV_TILE) {
            // one work-item per channel walks the chunk
            q.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
                const int c = (int) id[0];
                float v0 = hist[(ptrdiff_t) c * 3], v1 = hist[(ptrdiff_t) c * 3 + 1], v2 = hist[(ptrdiff_t) c * 3 + 2];
                const float w0 = w[(ptrdiff_t) c * 4], w1 = w[(ptrdiff_t) c * 4 + 1], w2 = w[(ptrdiff_t) c * 4 + 2], w3 = w[(ptrdiff_t) c * 4 + 3];
                for (int64_t t = 0; t < T; ++t) {
                    const float x = qkv[t * C + c];
                    const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                    h[t * C + c] = s / (1.0f + sycl::exp(-s));
                    v0 = v1; v1 = v2; v2 = x;
                }
                hist[(ptrdiff_t) c * 3] = v0; hist[(ptrdiff_t) c * 3 + 1] = v1; hist[(ptrdiff_t) c * 3 + 2] = v2;
            });
        } else {
            // C-3: tiled over tokens, each tile reading its 3 predecessors from the chunk (or the history before it);
            // the same expression per element, then the history written afterwards
            const size_t tiles = (size_t) ((T + CONV_TILE - 1) / CONV_TILE);
            static const bool fused = [] {   // upstream 9668e324; STRATA_GDN_CONVL2=0: the conv, then the norm kernel
                const char* v = strata::env_get("GDN_CONVL2");
                return v == nullptr || std::strtol(v, nullptr, 10) != 0;
            }();
            if (fused) {
                // a work-group is one head's S channels of one tile: the conv loop below, then, for the q/k heads, each
                // token's L2 norm as the norm kernel computes it (the same sub-group sum, the four partials in order, the
                // same v * rsqrt(ss + eps) on the stored value)
                q.submit([&](sycl::handler& hd) {
                    sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) 4 * CONV_TILE), hd);
                    hd.parallel_for(sycl::nd_range<2>({tiles, (size_t) C}, {1, (size_t) S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
                        const sycl::sub_group sg = it.get_sub_group();
                        const int c = (int) it.get_global_id(1), tid = (int) it.get_local_id(1);
                        const int64_t t0 = (int64_t) it.get_group(0) * CONV_TILE;
                        const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
                        auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
                        float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
                        const float* wc = w + (size_t) c * 4;
                        const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
                        for (int64_t t = t0; t < t1; ++t) {
                            const float x = qkv[t * C + c];
                            const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                            h[t * C + c] = s / (1.0f + sycl::exp(-s));
                            v0 = v1; v1 = v2; v2 = x;
                        }
                        if ((int) it.get_group(1) >= 2 * HK) return;   // the v channels: no norm (the whole group)
                        for (int64_t t = t0; t < t1; ++t) {
                            const float v = h[t * C + c];
                            const float sq = warp_sum(sg, v * v);
                            if ((tid & 31) == 0) part[(t - t0) * 4 + (tid >> 5)] = sq;
                        }
                        sycl::group_barrier(it.get_group());
                        for (int64_t t = t0; t < t1; ++t) {
                            const int64_t k = t - t0;
                            const float ss = part[k * 4] + part[k * 4 + 1] + part[k * 4 + 2] + part[k * 4 + 3];
                            h[t * C + c] = h[t * C + c] * sycl::rsqrt(ss + eps);
                        }
                    });
                });
                q.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
                    const int c = (int) id[0];
                    float v[3];
                    for (int k = 0; k < 3; ++k) {
                        const int64_t t = T - 3 + k;
                        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
                    }
                    float* hc = hist + (size_t) c * 3;
                    hc[0] = v[0]; hc[1] = v[1]; hc[2] = v[2];
                });
                return;
            }
            q.parallel_for(sycl::range<2>(tiles, C), [=](sycl::id<2> id) {
                const int c = (int) id[1];
                const int64_t t0 = (int64_t) id[0] * CONV_TILE;
                const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
                auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
                float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
                const float w0 = w[(ptrdiff_t) c * 4], w1 = w[(ptrdiff_t) c * 4 + 1], w2 = w[(ptrdiff_t) c * 4 + 2], w3 = w[(ptrdiff_t) c * 4 + 3];
                for (int64_t t = t0; t < t1; ++t) {
                    const float x = qkv[t * C + c];
                    const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                    h[t * C + c] = s / (1.0f + sycl::exp(-s));
                    v0 = v1; v1 = v2; v2 = x;
                }
            });
            q.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
                const int c = (int) id[0];
                float v[3];
                for (int k = 0; k < 3; ++k) {
                    const int64_t t = T - 3 + k;
                    v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
                }
                hist[(ptrdiff_t) c * 3] = v[0]; hist[(ptrdiff_t) c * 3 + 1] = v[1]; hist[(ptrdiff_t) c * 3 + 2] = v[2];
            });
        }
        // the L2 norm of the 32 q/k heads per token, 128 work-items each
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float, 1> part(sycl::range<1>(4), hd);
            hd.parallel_for(sycl::nd_range<2>({(size_t) T, (size_t) 2 * HK * S}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int64_t t = (int64_t) it.get_group(0);
                const int head = (int) it.get_group(1), tid = (int) it.get_local_id(1);
                float* x = h + t * C + (int64_t) head * S;
                const float v = x[tid];
                const float sq = warp_sum(sg, v * v);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                x[tid] = v * sycl::rsqrt(ss + eps);
            });
        });
    }

    // The recurrence with one value column a work-item and all S rows of its state in registers (the large register
    // file): no barrier, no local memory, q and k read once per sub-group and handed out by sub-group broadcasts.  Per
    // column the same arithmetic in the same order as the D-2 kernel below (four row groups of S / 4, summed left to
    // right), so the same bits.  B70, one DeltaNet layer over 26K tokens: 32 ms against D-2's 42.
    template <int SG>
    struct GdnRecLane {
        float* state;
        const float* h;
        const float* gate;
        const float* beta;
        float* y;
        int64_t T;
        static constexpr int NJ = S / SG;   // the q and k values a work-item holds
        static constexpr int QR = S / 4;    // the D-2 kernel's rows per row group
        void operator()(sycl::nd_item<1> it) const {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int gcol = (int) it.get_global_id(0), head = gcol / S, col = gcol % S, qh = head % HK;
            const size_t rs = (size_t) HV * S;
            float* base = state + (size_t) head * S + col;
            float s[S];
    #pragma unroll
            for (int r = 0; r < S; ++r) s[r] = base[r * rs];
            float nk[NJ], nq[NJ], nv = 0.0f, ng = 0.0f, nb = 0.0f;
            auto fetch = [&](int64_t t) {
                const float* ht = h + t * C;
    #pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    nq[j] = ht[qh * S + j * SG + lane];
                    nk[j] = ht[HK * S + qh * S + j * SG + lane];
                }
                nv = ht[2 * HK * S + head * S + col];
                ng = gate[t * HV + head];
                nb = beta[t * HV + head];
            };
            if (T > 0) fetch(0);
            for (int64_t t = 0; t < T; ++t) {
                float ck[NJ], cq[NJ];
    #pragma unroll
                for (int j = 0; j < NJ; ++j) { ck[j] = nk[j]; cq[j] = nq[j]; }
                const float cv = nv, cg = ng, cbt = nb;
                if (t + 1 < T) fetch(t + 1);
                const float g = sycl::exp(cg);
                // the four row groups' sums interleaved, each still in row order: four independent chains instead of one
                // (1.7x on the B70)
                float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
                for (int i = 0; i < QR; ++i)
    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int r = k * QR + i;
                        a[k] = sycl::fma(s[r], sycl::select_from_group(sg, ck[r / SG], r % SG), a[k]);
                    }
                const float delta = (cv - g * (a[0] + a[1] + a[2] + a[3])) * cbt;
    #pragma unroll
                for (int r = 0; r < S; ++r)
                    s[r] = sycl::fma(g, s[r], sycl::select_from_group(sg, ck[r / SG], r % SG) * delta);
                float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
                for (int i = 0; i < QR; ++i)
    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int r = k * QR + i;
                        o[k] = sycl::fma(s[r], sycl::select_from_group(sg, cq[r / SG], r % SG), o[k]);
                    }
                y[t * HV * S + (int64_t) head * S + col] = (o[0] + o[1] + o[2] + o[3]) * sycl::rsqrt((float) S);
            }
    #pragma unroll
            for (int r = 0; r < S; ++r) base[r * rs] = s[r];
        }
        auto get(sycl::ext::oneapi::experimental::properties_tag) const {
            return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                               sycl::ext::oneapi::experimental::sub_group_size<SG>};
        }
    };

    // The key-head form (upstream's gdn_rec_kh_kernel): a work-group takes one key head's VPK value heads and CB of their
    // columns, so every q and k value it reads feeds VPK heads, and 64 work-groups walk the chunk instead of 192.  The
    // inputs come in blocks of GDN_TB tokens, read into registers while the block before computes and stored to local
    // memory after it (upstream's cp.async), and a token takes two barriers instead of four.  Per value head and column
    // the same arithmetic in the same order as the D-2 kernel: the same bits.
    struct GdnRecKh {
        static constexpr int NT = CB * RG, QK4 = GDN_TB * 2 * (S / 4), V4 = GDN_TB * VPK * (CB / 4), GB = 2 * GDN_TB * VPK;
        static constexpr int NQK = (QK4 + NT - 1) / NT, NV = (V4 + NT - 1) / NT;
        float* state;
        const float* h;
        const float* gate;
        const float* beta;
        float* y;
        int64_t T;
        sycl::local_accessor<float, 1> sq, sk, sv, sgb, rkv, ro;   // [2][TB][S] x2, [2][TB][VPK][CB], [2][2][TB][VPK], ...

        void operator()(sycl::nd_item<1> it) const {
            const int qh = (int) it.get_group(0) / NCB, cb = (int) it.get_group(0) % NCB;
            const int tid = (int) it.get_local_id(0), c = tid % CB, rg = tid / CB, col = cb * CB + c;
            float s[VPK][RPG];
            const size_t rs = (size_t) HV * S;
    #pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float* base = state + ((size_t) (rg * RPG) * HV + (size_t) (qh + j * HK)) * S + col;
    #pragma unroll
                for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
            }
            const int64_t nblk = (T + GDN_TB - 1) / GDN_TB;
            sycl::float4 rqk[NQK], rv[NV];
            float rgb = 0.0f;
            auto load = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into registers
                const int64_t t0 = k * GDN_TB;
    #pragma unroll
                for (int n = 0; n < NQK; ++n) {
                    const int p = tid + n * NT;
                    if (p >= QK4) break;
                    const int i = p / (2 * (S / 4)), w = p % (2 * (S / 4)), isk = w / (S / 4), jj = (w % (S / 4)) * 4;
                    rqk[n] = t0 + i < T ? *reinterpret_cast<const sycl::float4*>(h + (t0 + i) * C + (int64_t) ((isk ? HK * S : 0) + qh * S + jj))
                                        : sycl::float4(0.0f);
                }
    #pragma unroll
                for (int n = 0; n < NV; ++n) {
                    const int p = tid + n * NT;
                    if (p >= V4) break;
                    const int i = p / (VPK * (CB / 4)), w = p % (VPK * (CB / 4)), j = w / (CB / 4), jj = (w % (CB / 4)) * 4;
                    rv[n] = t0 + i < T ? *reinterpret_cast<const sycl::float4*>(h + (t0 + i) * C +
                                                                                 (int64_t) (2 * HK * S + (qh + j * HK) * S + cb * CB + jj))
                                       : sycl::float4(0.0f);
                }
                if (tid < GB) {
                    const int isb = tid / (GDN_TB * VPK), w = tid % (GDN_TB * VPK), i = w / VPK, j = w % VPK;
                    rgb = t0 + i < T ? (isb ? beta : gate)[(t0 + i) * HV + (int64_t) (qh + j * HK)] : 0.0f;
                }
            };
            auto store = [&](int bb) {     // the registers into buffer bb
    #pragma unroll
                for (int n = 0; n < NQK; ++n) {
                    const int p = tid + n * NT;
                    if (p >= QK4) break;
                    const int i = p / (2 * (S / 4)), w = p % (2 * (S / 4)), isk = w / (S / 4), jj = (w % (S / 4)) * 4;
                    const auto& dst = isk ? sk : sq;
                    for (int e = 0; e < 4; ++e) dst[((size_t) bb * GDN_TB + i) * S + jj + e] = rqk[n][e];
                }
    #pragma unroll
                for (int n = 0; n < NV; ++n) {
                    const int p = tid + n * NT;
                    if (p >= V4) break;
                    const int i = p / (VPK * (CB / 4)), w = p % (VPK * (CB / 4)), j = w / (CB / 4), jj = (w % (CB / 4)) * 4;
                    for (int e = 0; e < 4; ++e) sv[(((size_t) bb * GDN_TB + i) * VPK + j) * CB + jj + e] = rv[n][e];
                }
                if (tid < GB) sgb[(size_t) bb * GB + tid] = rgb;   // [bb][gate, beta][i][j]
            };
            if (nblk > 0) {
                load(0);
                store(0);
            }
            sycl::group_barrier(it.get_group());
            for (int64_t k = 0; k < nblk; ++k) {
                if (k + 1 < nblk) load(k + 1);
                const int bb = (int) (k & 1);
                const int n = (int) std::min<int64_t>(GDN_TB, T - k * GDN_TB);
                for (int i = 0; i < n; ++i) {
                    const int64_t t = k * GDN_TB + i;
                    float kc[RPG];
    #pragma unroll
                    for (int r = 0; r < RPG; ++r) kc[r] = sk[((size_t) bb * GDN_TB + i) * S + (size_t) (rg * RPG + r)];
                    float g[VPK], kv[VPK], delta[VPK], o[VPK];
    #pragma unroll
                    for (int j = 0; j < VPK; ++j) {
                        g[j] = sycl::exp(sgb[(size_t) bb * GB + (size_t) (i * VPK + j)]);
                        kv[j] = 0.0f;
                        o[j] = 0.0f;
                    }
    #pragma unroll
                    for (int r = 0; r < RPG; ++r)
    #pragma unroll
                        for (int j = 0; j < VPK; ++j) kv[j] = sycl::fma(s[j][r], kc[r], kv[j]);
    #pragma unroll
                    for (int j = 0; j < VPK; ++j) rkv[(j * RG + rg) * CB + c] = kv[j];
                    sycl::group_barrier(it.get_group());
    #pragma unroll
                    for (int j = 0; j < VPK; ++j) {
                        const float kv_col = rkv[(j * RG + 0) * CB + c] + rkv[(j * RG + 1) * CB + c] +
                                             rkv[(j * RG + 2) * CB + c] + rkv[(j * RG + 3) * CB + c];
                        delta[j] = (sv[(((size_t) bb * GDN_TB + i) * VPK + j) * CB + c] - g[j] * kv_col) *
                                   sgb[(size_t) bb * GB + (size_t) (GDN_TB * VPK + i * VPK + j)];
                    }
    #pragma unroll
                    for (int r = 0; r < RPG; ++r) {
                        const float qr = sq[((size_t) bb * GDN_TB + i) * S + (size_t) (rg * RPG + r)];
    #pragma unroll
                        for (int j = 0; j < VPK; ++j) {
                            s[j][r] = sycl::fma(g[j], s[j][r], kc[r] * delta[j]);
                            o[j] = sycl::fma(s[j][r], qr, o[j]);
                        }
                    }
    #pragma unroll
                    for (int j = 0; j < VPK; ++j) ro[(j * RG + rg) * CB + c] = o[j];
                    sycl::group_barrier(it.get_group());
                    if (rg < VPK)   // row group j writes head j's output
                        y[t * HV * S + (int64_t) (qh + rg * HK) * S + col] =
                            (ro[(rg * RG + 0) * CB + c] + ro[(rg * RG + 1) * CB + c] + ro[(rg * RG + 2) * CB + c] +
                             ro[(rg * RG + 3) * CB + c]) * sycl::rsqrt((float) S);
                }
                // buffer (k + 1) & 1 was read by block k - 1, whose last token's second barrier every work-item passed
                if (k + 1 < nblk) store((int) ((k + 1) & 1));
                sycl::group_barrier(it.get_group());
            }
    #pragma unroll
            for (int j = 0; j < VPK; ++j) {
                float* base = state + ((size_t) (rg * RPG) * HV + (size_t) (qh + j * HK)) * S + col;
    #pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
            }
        }
    };
    static_assert(HV % HK == 0 && VPK <= RG && GdnRecKh::GB <= GdnRecKh::NT, "the key-head form's layout");

    // Whether GdnRecLane<16> runs on this device: a GRF holds 16 floats only where the EU is 16 wide (Xe2; the 8-wide
    // EUs of Xe-LP and Xe-HPG would spill the 128 state rows), and the device must build the large register file.
    // STRATA_GDN_REC_LANE=0: the D-2 kernel (A/B).
    static bool gdn_lane_ok(sycl::queue& q) {
        static core::PerDevice<bool> per_device;
        return per_device.get(q.get_device(), [&] {
            const char* v = strata::env_get("GDN_REC_LANE");
            if (v != nullptr && v[0] == '0') return false;
            const sycl::device d = q.get_device();
            if (!d.has(sycl::aspect::ext_intel_gpu_eu_simd_width) ||
                d.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>() < 16)
                return false;
            try {
                (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                    q.get_context(), {d}, {sycl::get_kernel_id<GdnRecLane<16>>()});
                return true;
            } catch (const sycl::exception& e) {
                std::fprintf(stderr, "xestrata: the GPU does not build the DeltaNet recurrence with the large register file "
                             "(%s): the prompt path keeps the work-group kernel\n", e.what());
                return false;
            }
        });
    }

    // `with_out` false (KVA's state-only layer): the same recurrence kernels without the output normalization, so z,
    // gamma and y16 are not read
    static void recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                           const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, bool silu, void* stream,
                           bool with_out = true) {
        auto& q = Q(stream);
        if (T <= 0) return;
        // D-2: the value columns split over 4 work-groups per head, then the output norm in its own kernel.  The next
        // token's inputs (its q and k rows, v, gate, beta) are loaded into registers while this one computes
        // (upstream ac6aad4, 04f4a53): the same bits.
        // STRATA_GDN_PIPELINE=0: each token's inputs loaded when it starts (rec_step), to compare.
        static const bool pipe_on = [] {
            const char* v = strata::env_get("GDN_PIPELINE");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        const bool pipe = pipe_on;   // the kernel takes a copy
        static_assert(CB * RG == S, "one q and one k value a work-item");
        // STRATA_GDN_KEYHEAD=0: the D-2 kernel where GdnRecLane does not run (A/B)
        static const bool kh_on = [] {
            const char* v = strata::env_get("GDN_KEYHEAD");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        auto launch_kh = [&](float* st, float* yy, const int64_t n) {
            q.submit([&](sycl::handler& hd) {
            using L = sycl::local_accessor<float, 1>;
            hd.parallel_for(sycl::nd_range<1>((size_t) HK * NCB * GdnRecKh::NT, GdnRecKh::NT),
                            GdnRecKh{st, h, gate, beta, yy, n, L(sycl::range<1>((size_t) 2 * GDN_TB * S), hd),
                                     L(sycl::range<1>((size_t) 2 * GDN_TB * S), hd),
                                     L(sycl::range<1>((size_t) 2 * GDN_TB * VPK * CB), hd),
                                     L(sycl::range<1>((size_t) 2 * GdnRecKh::GB), hd),
                                     L(sycl::range<1>((size_t) VPK * RG * CB), hd),
                                     L(sycl::range<1>((size_t) VPK * RG * CB), hd)});
        });
        };
        auto launch_d2 = [&](float* st, float* yy, const int64_t n) {
            q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
                red(sycl::range<1>((size_t) RG * CB), hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) HV * NCB * CB * RG, (size_t) CB * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const int head = (int) it.get_group(0) / NCB, cb = (int) it.get_group(0) % NCB;
                const int tid = (int) it.get_local_id(0), c = tid % CB, rg = tid / CB, col = cb * CB + c;
                const int qh = head % HK;
                float s[RPG];
                float* base = st + ((size_t) (rg * RPG) * HV + head) * S + col;
                const size_t rs = (size_t) HV * S;
                for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
                if (!pipe) {
                    for (int64_t t = 0; t < n; ++t) {
                        const float oc = rec_step<CB>(it, s, h + t * C, gate, beta, t, head, qh, rg, c, col, tid, &sk[0],
                                                      &sq[0], &red[0]);
                        if (rg == 0) yy[t * HV * S + (int64_t) head * S + col] = oc;
                    }
                    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
                    return;
                }
                float nq = 0.0f, nk = 0.0f, nv = 0.0f, ng = 0.0f, nb = 0.0f;
                auto fetch = [&](int64_t t) {
                    const float* ht = h + t * C;
                    nq = ht[qh * S + tid];
                    nk = ht[HK * S + qh * S + tid];
                    nv = ht[2 * HK * S + head * S + col];
                    ng = gate[t * HV + head];
                    nb = beta[t * HV + head];
                };
                if (n > 0) fetch(0);
                for (int64_t t = 0; t < n; ++t) {
                    const float cq = nq, ck = nk, cv = nv, cg = ng, cbt = nb;
                    sycl::group_barrier(it.get_group());
                    sq[tid] = cq;
                    sk[tid] = ck;
                    sycl::group_barrier(it.get_group());
                    if (t + 1 < n) fetch(t + 1);
                    const float g = sycl::exp(cg);
                    float kv = 0.0f;
                    for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                    red[rg * CB + c] = kv;
                    sycl::group_barrier(it.get_group());
                    const float kv_col = red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c];
                    const float delta = (cv - g * kv_col) * cbt;
                    float o = 0.0f;
                    for (int r = 0; r < RPG; ++r) {
                        s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                        o = sycl::fma(s[r], sq[rg * RPG + r], o);
                    }
                    sycl::group_barrier(it.get_group());
                    red[rg * CB + c] = o;
                    sycl::group_barrier(it.get_group());
                    if (rg == 0)
                        yy[t * HV * S + (int64_t) head * S + col] =
                            (red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c]) * sycl::rsqrt((float) S);
                }
                for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
            });
        });
        };
        // D-2 or the key-head form, whichever ran a chunk faster on this device: timed once, on copies of the state, at
        // the first call of at least 64 tokens.  RTX 4070: 2175 -> 459 us for 512 tokens (+2.4% prompt speed); the B70
        // without GdnRecLane: 56 -> 92 ms a prompt (the key-head form keeps 3 x 32 state rows a work-item in registers).
        // Timed on the chunk's first 1024 tokens at most: the ratio does not move with the length (RTX 4070: 5.2x at
        // 1016 tokens, 5.3x at 16384), and the output copy of a 16384-token chunk (256 MiB) did not fit beside a full
        // expert cache, which left the 5x slower D-2 on unmeasured (32K serve prompt 1,943 -> 1,687 tok/s).
        static int pick = -1;   // 1: the key-head form
        if (!gdn_lane_ok(q)) {
            if (pick < 0 && kh_on && T >= 64) {
                const int64_t tn = std::min<int64_t>(T, 1024);
                const size_t ns = (size_t) S * HV * S, ny = (size_t) tn * HV * S;
                float* st = sycl::malloc_device<float>(ns, q);
                float* yy = sycl::malloc_device<float>(ny, q);
                if (st != nullptr && yy != nullptr) {
                    auto time = [&](auto&& launch) {   // the second run: the first may compile the kernel
                        double us = 0;
                        for (int r = 0; r < 2; ++r) {
                            q.copy(state, st, ns).wait();
                            const auto t0 = std::chrono::steady_clock::now();
                            launch(st, yy, tn);
                            q.wait();
                            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                        }
                        return us;
                    };
                    const double d2 = time(launch_d2), kh = time(launch_kh);
                    pick = kh < d2 ? 1 : 0;
                    std::fprintf(stderr, "xestrata: the DeltaNet recurrence, %lld tokens: D-2 %.0f us, the key-head form %.0f "
                                 "us: %s\n", (long long) tn, d2, kh, pick ? "the key-head form" : "D-2");
                } else {
                    static bool said = false;   // D-2 meanwhile; the next call tries again
                    if (!said)
                        std::fprintf(stderr, "xestrata: the DeltaNet recurrence: no VRAM to time the two forms on %lld "
                                             "tokens, D-2 until it can be timed\n", (long long) tn);
                    said = true;
                }
                if (st != nullptr) sycl::free(st, q);
                if (yy != nullptr) sycl::free(yy, q);
            }
        }
        // the token-by-token form this device takes, on `st` and `yy` over n tokens
        auto seq = [&](float* st, float* yy, const int64_t n) {
            if (gdn_lane_ok(q))
                q.parallel_for(sycl::nd_range<1>((size_t) HV * S, 16), GdnRecLane<16>{st, h, gate, beta, yy, n});
            else if (kh_on && pick == 1) launch_kh(st, yy, n);
            else launch_d2(st, yy, n);
        };
        auto launch_norm = [&](const float* yy, uint16_t* yo, const int64_t n) {
            q.submit([&](sycl::handler& hd) {
                sycl::local_accessor<float, 1> wsum(sycl::range<1>(4), hd);
                hd.parallel_for(sycl::nd_range<2>({(size_t) n, (size_t) HV * S}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
                    const sycl::sub_group sg = it.get_sub_group();
                    const int64_t t = (int64_t) it.get_group(0);
                    const int head = (int) it.get_group(1), col = (int) it.get_local_id(1);
                    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
                    const float oc = yy[at];
                    const float sp = warp_sum(sg, oc * oc);
                    if ((col & 31) == 0) wsum[col >> 5] = sp;
                    sycl::group_barrier(it.get_group());
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    // FP16 only: the out projection reads y16, nothing reads the normalized value in FP32 (upstream
                    // a7ce31b)
                    yo[at] = hf(oc * sycl::rsqrt(ss / (float) S + eps) * gamma[col] * zgate(z[at], silu));
                });
            });
        };
        const GdnChunkNorm nm{z, gamma, eps, silu, y16};
        if (chunked(q, T, with_out, seq, launch_norm, state, h, gate, beta, z, gamma, eps, silu)) {
            if (gdn_chunk_run(q, chunk_pick(q).form, HK, HV, state, h, gate, beta, y, with_out ? &nm : nullptr, T))
                return;   // the scan normalized the outputs itself
        } else {
            seq(state, y, T);
        }
        if (!with_out) return;
        launch_norm(y, y16, T);
    }

    // The chunked scan (gdn_chunk.cpp) or the token-by-token kernels, for each device.  XESTRATA_GDN_CHUNKED unset or
    // auto: the faster one, timed once on copies of the state at the first call of kChunkTimeMin tokens or more (each
    // with its output norm), at 256 tokens and at the call's first 2048 at most.  A scan form at least 3% faster at
    // the longer length takes the calls from where the two lines (time = a + b tokens) cross: a device where the scan
    // is slower at both lengths never takes it.  Nothing is written down: every start times again.  =0: never; =1:
    // every call of 64 tokens or more, in the first form that fits.
    struct ChunkPick {
        bool decided = false;
        int form = -1;   // -1: the token-by-token kernels
        int64_t min_t = INT64_MAX;
    };
    static constexpr int64_t kChunkTimeMin = 512;
    static ChunkPick& chunk_pick(sycl::queue& q) {
        static core::PerDevice<std::unique_ptr<ChunkPick>> per_device;
        return *per_device.get(q.get_device(), [] { return std::make_unique<ChunkPick>(); });
    }
    template <class Seq, class Norm>
    static bool chunked(sycl::queue& q, int64_t T, bool with_out, Seq&& seq, Norm&& launch_norm, const float* state,
                        const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                        float eps, bool silu) {
        static const int mode = strata::env_auto("GDN_CHUNKED");   // -1 automatic, 0 off, 1 on
        if (mode == 0 || T < 64) return false;
        ChunkPick& cp = chunk_pick(q);
        if (!cp.decided && mode == 1) {
            const auto& forms = gdn_chunk_forms(q, HK, HV);
            cp.decided = true;
            if (!forms.empty()) {
                cp.form = forms.front();
                cp.min_t = 64;
            }
        }
        if (!cp.decided && T >= kChunkTimeMin) {
            const auto& forms = gdn_chunk_forms(q, HK, HV);
            if (forms.empty()) {
                cp.decided = true;
                return false;
            }
            const int64_t n1 = 256, n2 = std::min<int64_t>(T, 2048);
            const size_t ns = (size_t) S * HV * S, ny = (size_t) n2 * HV * S;
            float* st = sycl::malloc_device<float>(ns, q);
            float* yy = sycl::malloc_device<float>(ny, q);
            uint16_t* yo = with_out ? sycl::malloc_device<uint16_t>(ny, q) : nullptr;
            if (st != nullptr && yy != nullptr && (yo != nullptr || !with_out)) {
                auto time = [&](auto&& launch, int64_t n) {   // the second run: the first may compile the kernels
                    double us = 0;
                    for (int r = 0; r < 2; ++r) {
                        q.copy(state, st, ns).wait();
                        const auto t0 = std::chrono::steady_clock::now();
                        launch(n);
                        q.wait();
                        us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    }
                    return us;
                };
                auto tok = [&](int64_t n) {
                    seq(st, yy, n);
                    if (with_out) launch_norm(yy, yo, n);
                };
                const GdnChunkNorm nt{z, gamma, eps, silu, yo};
                const double s1 = time(tok, n1), s2 = time(tok, n2);
                double c1 = 0, c2 = 0;
                int best = -1;
                for (const int f : forms) {
                    auto chunk = [&](int64_t n) {
                        const bool normed = gdn_chunk_run(q, f, HK, HV, st, h, gate, beta, yy,
                                                          with_out ? &nt : nullptr, n);
                        if (with_out && !normed) launch_norm(yy, yo, n);
                    };
                    const double b2 = time(chunk, n2);
                    if (best < 0 || b2 < c2) {
                        best = f;
                        c2 = b2;
                        c1 = time(chunk, n1);
                    }
                }
                cp.decided = true;
                if (c2 < 0.97 * s2) {
                    cp.form = best;
                    const double bs = (s2 - s1) / (double) (n2 - n1), bc = (c2 - c1) / (double) (n2 - n1);
                    const double cross = c1 <= s1 ? 0.0 : bs > bc ? (double) n1 + (c1 - s1) / (bs - bc) : (double) n2;
                    cp.min_t = std::max<int64_t>(64, std::min<int64_t>(n2, (int64_t) std::ceil(cross)));
                }
                if (cp.form >= 0)
                    std::fprintf(stderr, "xestrata: the DeltaNet recurrence, %lld / %lld tokens: token by token %.0f / "
                                 "%.0f us, the chunked scan (form %d) %.0f / %.0f us: the scan from %lld tokens\n",
                                 (long long) n1, (long long) n2, s1, s2, best, c1, c2, (long long) cp.min_t);
                else
                    std::fprintf(stderr, "xestrata: the DeltaNet recurrence, %lld tokens: token by token %.0f us, the "
                                 "chunked scan %.0f us: token by token\n", (long long) n2, s2, c2);
            } else {
                static bool said = false;   // token by token meanwhile; the next call tries again
                if (!said)
                    std::fprintf(stderr, "xestrata: the DeltaNet recurrence: no VRAM to time the chunked scan on %lld "
                                         "tokens, token by token until it can be timed\n", (long long) n2);
                said = true;
            }
            if (st != nullptr) sycl::free(st, q);
            if (yy != nullptr) sycl::free(yy, q);
            if (yo != nullptr) sycl::free(yo, q);
        }
        return cp.decided && cp.form >= 0 && T >= cp.min_t;
    }
};

namespace {
template <class F> void gdn_dispatch(int64_t h_k, int64_t h_v, const char* what, F&& f) {
    if (h_k == 16 && h_v == 48) f(Gdn<16, 48>{});
    else if (h_k == 16 && h_v == 32) f(Gdn<16, 32>{});
    else if (h_k == 16 && h_v == 16) f(Gdn<16, 16>{});
    else throw core::DeviceError(std::string(what) + ": no kernels for " + std::to_string(h_k) + " key and " +
                                 std::to_string(h_v) + " value heads");
}
}  // namespace

void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream,
               int64_t h_k, int64_t h_v) {
    gdn_dispatch(h_k, h_v, "gdn_gates", [&](auto g) { decltype(g)::gates(ab, dt, ssm_a, gate, beta, T, stream); });
}

void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              int64_t h_k, int64_t h_v) {
    gdn_dispatch(h_k, h_v, "gdn_conv", [&](auto g) { decltype(g)::conv(history, qkv, conv_w, h, T, eps, stream); });
}

void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t h_k,
                    int64_t h_v, bool silu) {
    gdn_dispatch(h_k, h_v, "gdn_recurrence", [&](auto g) {
        decltype(g)::recurrence(state, h, gate, beta, z, gamma, eps, y, y16, T, silu, stream);
    });
}

void gdn_recurrence_state_only(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                               void* stream, int64_t h_k, int64_t h_v) {
    gdn_dispatch(h_k, h_v, "gdn_recurrence_state_only", [&](auto g) {
        decltype(g)::recurrence(state, h, gate, beta, nullptr, nullptr, 0.0f, y, nullptr, T, false, stream, false);
    });
}

// ---------------------------------------------------------------- MoE
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (T <= 0) return;
    // the decode's router, each row as one token there
    strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<false>(Q(stream), blob, gu16, down16);
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<true>(Q(stream), blob, gu16, down16);
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    kernels::rows16::saturated(Q(stream), n * 640, h16, [=](int64_t i) {
        const int64_t r = i / 640, k = i % 640;
        return kernels::rows16::swiglu(gu[r * 1280 + 2 * k], gu[r * 1280 + 2 * k + 1]);
    });
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream, int64_t width, int64_t ld) {
    if (n <= 0) return;
    if (ld <= 0 || ld == width) {   // one run of n x width values (a row index per value cost the shared expert 40%)
        kernels::rows16::saturated(Q(stream), n * width, h16, [=](int64_t i) {
            return kernels::rows16::swiglu(g[i], u[i]);
        });
        return;
    }
    // the rows in the first of two dimensions (no division a value), which takes 65,535: more, one dimension
    if (n > 65535) {
        Q(stream).parallel_for(sycl::range<1>((size_t) (n * width)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0], r = i / width, at = r * ld + (i - r * width);
            const float a = g[at];
            h16[i] = hf_sat(a / (1.0f + sycl::exp(-a)) * u[at]);
        });
        return;
    }
    Q(stream).parallel_for(sycl::range<2>((size_t) n, (size_t) width), [=](sycl::id<2> id) {
        const int64_t r = (int64_t) id[0], c = (int64_t) id[1], at = r * ld + c;
        const float a = g[at];
        h16[r * width + c] = hf_sat(a / (1.0f + sycl::exp(-a)) * u[at]);
    });
}
void swiglu_pair_h(const uint16_t* g16, const uint16_t* u16, uint16_t* h16, int64_t n, void* stream, int64_t width) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * width)), [=](sycl::id<1> i) {
        const float a = kernels::f32_from_f16(g16[i]);
        h16[i] = hf_sat(a / (1.0f + sycl::exp(-a)) * kernels::f32_from_f16(u16[i]));
    });
}
void add_h(float* x, const uint16_t* y16, int64_t n, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += kernels::f32_from_f16(y16[i]); });
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    const int64_t per = width / 8;   // 16-byte units, 8 values each
    const auto* x = reinterpret_cast<const sycl::uint4*>(x16);
    auto* dst = reinterpret_cast<sycl::uint4*>(dst16);
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * per)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / per, j = i % per;
        dst[r * per + j] = x[(int64_t) src[r] * per + j];
    });
}
void gather_rows(const float* x, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * width)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / width, j = i % width;
        dst[i] = x[(int64_t) src[r] * width + j];
    });
}
void scatter_rows(const float* src, const int32_t* dst_row, float* dst, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * width)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / width, j = i % width;
        dst[(int64_t) dst_row[r] * width + j] = src[i];
    });
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    if (T <= 0) return;
    kernels::native_moe_combine_multi(Dm, w, shared, bo, N, 10, (int) T, stream, sg, slot);
}
void moe_shared_init(const float* shared, const float* sg, float* acc, int64_t T, void* stream) {
    if (T <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        acc[i] = shared[i] * sigm(sg[i / N]);
    });
}
// one work-item per (token of the group, column): the token's rows summed in the group's row order
void moe_scatter_add(const float* Dm, const int32_t* u_tok, const int32_t* u_off, const int32_t* u_rows,
                     const float* u_w, int64_t n_u, float* acc, void* stream, int64_t width) {
    if (n_u <= 0) return;
    const int64_t W = width > 0 ? width : N;
    if (u_off == nullptr) {   // one row a token, the sum's one term (a loop and its bounds cost HunyuanImage 3.0 0.6%)
        strata::kernels::for_rows_cols(Q(stream), n_u, W, [=](int64_t u, int64_t d) {
            acc[(int64_t) u_tok[u] * W + d] += sycl::fma(u_w[u], Dm[u * W + d], 0.0f);
        });
        return;
    }
    strata::kernels::for_rows_cols(Q(stream), n_u, W, [=](int64_t u, int64_t d) {
        float s = 0.0f;
        for (int32_t j = u_off[u]; j < u_off[u + 1]; ++j) s = sycl::fma(u_w[j], Dm[(int64_t) u_rows[j] * W + d], s);
        acc[(int64_t) u_tok[u] * W + d] += s;
    });
}

// ---------------------------------------------------------------- QSA helpers
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, const strata::kernels::RopeScaling& scaling, void* stream) {
    strata::kernels::rope_rows(x, x, T, heads, (int) dim, ld, ld, 64, scaling, nullptr, pos0, stream);
}
void split_q(const float* qf, float* q, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * 24 * 256)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / ((int64_t) 24 * 256), h = (i / 256) % 24, d = i % 256;
        q[i] = qf[t * 24 * 512 + h * 512 + d];
    });
}

}  // namespace strata::prefill
