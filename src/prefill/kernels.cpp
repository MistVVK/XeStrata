// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/kernels.cpp - see include/strata/prefill/kernels.hpp.  The Xe port of Strata's src/prefill/kernels.cu: work
// shapes, explicit fmaf and summation orders are the CUDA kernels'; a warp is a sub-group of 32 and a 2-D CUDA block
// is linearised the way CUDA numbers its threads (x fastest).  __expf, rsqrtf, powf, cosf, sinf and log1pf are the
// precise SYCL functions.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
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
inline uint16_t hf(float f) { return kernels::f16_from_f32(f); }
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

size_t groups_for(int64_t n, int t = 256) { return (size_t) ((n + t - 1) / t) * t; }

constexpr int CONV_TILE = 64;
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;

// the recurrence's shared per-token step for one (head, column); returns the column's output before the norm
template <int COLS>
inline float rec_step(const sycl::nd_item<1>& it, float* s, const float* ht, const float* gate, const float* beta,
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

template <int REG>
void route_launch(sycl::queue& q, const float* logits, int32_t* ids, float* wout, int64_t T) {
    q.parallel_for(sycl::nd_range<1>((size_t) ((T + 7) / 8) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t t = (int64_t) it.get_group(0) * 8 + (int64_t) sg.get_group_linear_id();
        if (t >= T) return;   // uniform over the sub-group
        const int lane = (int) sg.get_local_linear_id();
        const float* lg = logits + t * (REG * 32);
        float v[REG];
        for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
        float mx = -INFINITY;
        for (int i = 0; i < REG; ++i) mx = sycl::fmax(mx, v[i]);
        mx = warp_max(sg, mx);
        float sum = 0.0f;
        for (int i = 0; i < REG; ++i) { v[i] = sycl::exp(v[i] - mx); sum += v[i]; }
        const float rcp = 1.0f / warp_sum(sg, sum);
        for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (sycl::isnan(v[i])) v[i] = -FLT_MAX; }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = v[0];
            int ex = lane;
            for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
            for (int m = 16; m; m >>= 1) {
                const float ob = sycl::permute_group_by_xor(sg, best, m);
                const int oi = sycl::permute_group_by_xor(sg, ex, m);
                if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
            }
            if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
            if (lane == 0) ids[t * 10 + rank] = ex;
            if (rank == lane) selected = best;
        }
        selected_sum = sycl::fmax(warp_sum(sg, selected_sum), 6.103515625e-5f);
        if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
    });
}

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
               void* stream, const strata::kernels::KvHostPools* host_p, const strata::kernels::KvHostPools* stage_p) {
    if (T <= 0) return;
    const strata::kernels::KvHostPools host = host_p ? *host_p : strata::kernels::KvHostPools{};
    const strata::kernels::KvHostPools stage = stage_p ? *stage_p : strata::kernels::KvHostPools{};
    const int32_t* table = page_table;
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> wm(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<3>({(size_t) T, 2, 8 * 64}, {1, 1, 64}), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t t = (int64_t) it.get_group(0);
            const int kvh = (int) it.get_group(1), z = (int) it.get_group(2), g = z >> 1;
            const bool is_v = (z & 1) != 0;
            const int tid = (int) it.get_local_id(2);
            const int d = g * 64 + tid;
            const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
            const int64_t pos = pos0 + t;
            const int64_t page = table[pos / page_size];
            const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
            const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
            if (k_pool != nullptr) {   // uniform
                const uint16_t hv = hf(x);
                if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = hv;
                if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = hv;
                if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = hv;
                return;
            }
            float a = sycl::fabs(x);
            for (int o = 16; o > 0; o >>= 1) a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
            if ((tid & 31) == 0) wm[tid >> 5] = a;
            sycl::group_barrier(it.get_group());
            const float amax = sycl::fmax(wm[0], wm[1]);
            const uint16_t sb = hf(amax / 127.0f);
            const float sf = kernels::f32_from_f16(sb);
            int qv = 0;
            if (sf > 0.0f) { qv = (int) sycl::rint(x / sf); qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv); }
            if (page >= 0) {
                (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) qv;
                if (tid == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
            }
            if (host.k_q != nullptr) {
                (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) qv;
                if (tid == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
            }
            if (stage.k_q != nullptr) {
                (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) qv;
                if (tid == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
            }
        });
    });
}

void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = hf(x[i]); });
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
            const int64_t j = t * D + c * N + d;
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
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * HV)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / HV, h = i % HV;
        const float v = ab[t * 2 * HV + h] + dt[h];
        gate[i] = (v > 20.0f ? v : sycl::log1p(sycl::exp(v))) * ssm_a[h];
        beta[i] = sigm(ab[t * 2 * HV + HV + h]);
    });
}

void gdn_conv(float* hist, const float* qkv, const float* w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    auto& q = Q(stream);
    if (serial || T <= CONV_TILE) {
        // one work-item per channel walks the chunk
        q.parallel_for(sycl::range<1>(C), [=](sycl::id<1> id) {
            const int c = (int) id[0];
            float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
            const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
            for (int64_t t = 0; t < T; ++t) {
                const float x = qkv[t * C + c];
                const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                h[t * C + c] = s / (1.0f + sycl::exp(-s));
                v0 = v1; v1 = v2; v2 = x;
            }
            hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
        });
    } else {
        // C-3: tiled over tokens, each tile reading its 3 predecessors from the chunk (or the history before it);
        // the same expression per element, then the history written afterwards
        const size_t tiles = (size_t) ((T + CONV_TILE - 1) / CONV_TILE);
        q.parallel_for(sycl::range<2>(tiles, C), [=](sycl::id<2> id) {
            const int c = (int) id[1];
            const int64_t t0 = (int64_t) id[0] * CONV_TILE;
            const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
            auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
            float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
            const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
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
            hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
        });
    }
    // the L2 norm of the 32 q/k heads per token, 128 work-items each
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(4), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) T, (size_t) 2 * HK * S}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t t = (int64_t) it.get_group(0);
            const int head = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            float* x = h + t * C + head * S;
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

namespace {
// Whether GdnRecLane<16> runs on this device: a GRF holds 16 floats only where the EU is 16 wide (Xe2; the 8-wide
// EUs of Xe-LP and Xe-HPG would spill the 128 state rows), and the device must build the large register file.
// STRATA_GDN_REC_LANE=0: the D-2 kernel (A/B).
bool gdn_lane_ok(sycl::queue& q) {
    static const bool ok = [&] {
        const char* v = std::getenv("STRATA_GDN_REC_LANE");
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
            std::fprintf(stderr, "strata: the GPU does not build the DeltaNet recurrence with the large register file "
                         "(%s): the prompt path keeps the work-group kernel\n", e.what());
            return false;
        }
    }();
    return ok;
}
}  // namespace

void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-group-per-head kernel (A/B)
    auto& q = Q(stream);
    if (serial || T <= 0) {
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
                red(sycl::range<1>(RG * S), hd), wsum(sycl::range<1>(16), hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) HV * S * RG, S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int head = (int) it.get_group(0), tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S;
                const int qh = head % HK;
                float s[RPG];
                float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
                const size_t rs = (size_t) HV * S;
                for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
                const float g_col = gamma[col];
                for (int64_t t = 0; t < T; ++t) {
                    const float oc_all = rec_step<S>(it, s, h + t * C, gate, beta, t, head, qh, rg, col, col, tid,
                                                     &sk[0], &sq[0], &red[0]);
                    float oc = 0.0f, sp = 0.0f;
                    if (rg == 0) { oc = oc_all; sp = oc * oc; }
                    sp = warp_sum(sg, sp);
                    if ((tid & 31) == 0) wsum[tid >> 5] = sp;
                    sycl::group_barrier(it.get_group());
                    if (rg == 0) {
                        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                        const float v = oc * sycl::rsqrt(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);
                        y[t * HV * S + head * S + col] = v;
                        y16[t * HV * S + head * S + col] = hf(v);
                    }
                }
                for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
            });
        });
        return;
    }
    // D-2: the value columns split over 4 work-groups per head, then the output norm in its own kernel; per column
    // the same arithmetic in the same order as the kernel above.  The next token's inputs (its q and k rows, v, gate,
    // beta) are loaded into registers while this one computes (upstream ac6aad4, 04f4a53): the same bits.
    // STRATA_GDN_PIPELINE=0: each token's inputs loaded when it starts (rec_step), to compare.
    static const bool pipe_on = [] {
        const char* v = std::getenv("STRATA_GDN_PIPELINE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool pipe = pipe_on;   // the kernel takes a copy
    static_assert(CB * RG == S, "one q and one k value a work-item");
    static const bool keyhead=[] { const char* v=std::getenv("STRATA_GDN_KEYHEAD"); return v && std::strtol(v, nullptr, 10)!=0; }();
    const size_t kh_local=((size_t) 2*S+size_t(2)*(HV/HK)*RG*CB)*sizeof(float);
    if (keyhead && pipe && q.get_device().get_info<sycl::info::device::local_mem_size>()>=kh_local &&
        q.get_device().get_info<sycl::info::device::max_work_group_size>()>=(size_t) CB*RG) {
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float,1> sq(sycl::range<1>(S),hd),sk(sycl::range<1>(S),hd),
                rkv(sycl::range<1>((size_t) (HV/HK)*RG*CB),hd),ro(sycl::range<1>((size_t) (HV/HK)*RG*CB),hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) HK*NCB*CB*RG,size_t(CB)*RG),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                constexpr int VPK=HV/HK;
                const int qh=(int) it.get_group(0)/NCB,cb=(int) it.get_group(0)%NCB;
                const int tid=(int) it.get_local_id(0),c=tid%CB,rg=tid/CB,col=cb*CB+c;
                const size_t rs=(size_t) HV*S;
                float values[VPK][RPG];
                for (int j=0;j<VPK;++j) {
                    const float* base=state+((size_t) (rg*RPG)*HV+qh+(int64_t) j*HK)*S+col;
                    for (int r=0;r<RPG;++r) values[j][r]=base[r*rs];
                }
                for (int64_t t=0;t<T;++t) {
                    sycl::group_barrier(it.get_group());
                    sq[tid]=h[t*C+(int64_t) qh*S+tid]; sk[tid]=h[t*C+(int64_t) HK*S+(int64_t) qh*S+tid];
                    sycl::group_barrier(it.get_group());
                    float decay[VPK],delta[VPK],out[VPK]={};
                    for (int j=0;j<VPK;++j) {
                        decay[j]=sycl::exp(gate[t*HV+qh+(int64_t) j*HK]);
                        float kv=0;
                        for (int r=0;r<RPG;++r) kv=sycl::fma(values[j][r],sk[rg*RPG+r],kv);
                        rkv[(j*RG+rg)*CB+c]=kv;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int j=0;j<VPK;++j) {
                        const float sum=rkv[j*RG*CB+c]+rkv[(j*RG+1)*CB+c]+rkv[(j*RG+2)*CB+c]+rkv[(j*RG+3)*CB+c];
                        delta[j]=(h[t*C+2*(int64_t) HK*S+(int64_t) (qh+(int64_t) j*HK)*S+col]-decay[j]*sum)*beta[t*HV+qh+(int64_t) j*HK];
                    }
                    for (int r=0;r<RPG;++r) for (int j=0;j<VPK;++j) {
                        values[j][r]=sycl::fma(decay[j],values[j][r],sk[rg*RPG+r]*delta[j]);
                        out[j]=sycl::fma(values[j][r],sq[rg*RPG+r],out[j]);
                    }
                    for (int j=0;j<VPK;++j) ro[(j*RG+rg)*CB+c]=out[j];
                    sycl::group_barrier(it.get_group());
                    if (rg<VPK) y[t*HV*S+(int64_t) (qh+(int64_t) rg*HK)*S+col]=
                        (ro[rg*RG*CB+c]+ro[(rg*RG+1)*CB+c]+ro[(rg*RG+2)*CB+c]+ro[(rg*RG+3)*CB+c])*sycl::rsqrt((float) S);
                }
                for (int j=0;j<VPK;++j) {
                    float* base=state+((size_t) (rg*RPG)*HV+qh+(int64_t) j*HK)*S+col;
                    for (int r=0;r<RPG;++r) base[r*rs]=values[j][r];
                }
            });
        });
    } else if (gdn_lane_ok(q))
        q.parallel_for(sycl::nd_range<1>((size_t) HV * S, 16), GdnRecLane<16>{state, h, gate, beta, y, T});
    else q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
            red(sycl::range<1>(RG * CB), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) HV * NCB * CB * RG, CB * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int head = (int) it.get_group(0) / NCB, cb = (int) it.get_group(0) % NCB;
            const int tid = (int) it.get_local_id(0), c = tid % CB, rg = tid / CB, col = cb * CB + c;
            const int qh = head % HK;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
            const size_t rs = (size_t) HV * S;
            for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
            if (!pipe) {
                for (int64_t t = 0; t < T; ++t) {
                    const float oc = rec_step<CB>(it, s, h + t * C, gate, beta, t, head, qh, rg, c, col, tid, &sk[0],
                                                  &sq[0], &red[0]);
                    if (rg == 0) y[t * HV * S + (int64_t) head * S + col] = oc;
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
            if (T > 0) fetch(0);
            for (int64_t t = 0; t < T; ++t) {
                const float cq = nq, ck = nk, cv = nv, cg = ng, cbt = nb;
                sycl::group_barrier(it.get_group());
                sq[tid] = cq;
                sk[tid] = ck;
                sycl::group_barrier(it.get_group());
                if (t + 1 < T) fetch(t + 1);
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
                    y[t * HV * S + (int64_t) head * S + col] =
                        (red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c]) * sycl::rsqrt((float) S);
            }
            for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
        });
    });
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(4), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) T, (size_t) HV * S}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t t = (int64_t) it.get_group(0);
            const int head = (int) it.get_group(1), col = (int) it.get_local_id(1);
            const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
            const float oc = y[at];
            const float sp = warp_sum(sg, oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = sp;
            sycl::group_barrier(it.get_group());
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            // FP16 only: the out projection reads y16, nothing reads the normalized value in FP32 (upstream a7ce31b)
            y16[at] = hf(oc * sycl::rsqrt(ss / (float) S + eps) * gamma[col] * sigm(z[at]));
        });
    });
}

// ---------------------------------------------------------------- MoE
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512) route_launch<16>(Q(stream), logits, ids, weights, T);
    else if (n_expert == 256) route_launch<8>(Q(stream), logits, ids, weights, T);
    else strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<false>(Q(stream), blob, gu16, down16);
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<true>(Q(stream), blob, gu16, down16);
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * 640)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / 640, k = i % 640;
        const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
        h16[i] = hf_sat(g / (1.0f + sycl::exp(-g)) * u);
    });
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) (n * 640)), [=](sycl::id<1> i) {
        const float a = g[i];
        h16[i] = hf_sat(a / (1.0f + sycl::exp(-a)) * u[i]);
    });
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
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, d = i % N;
        float s = 0.0f;
        for (int k = 0; k < 10; ++k) s = sycl::fma(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
        bo[i] = s + shared[i] * sigm(sg[t]);
    });
}

// ---------------------------------------------------------------- QSA helpers
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            float* r = x + (int64_t) it.get_group(0) * ld;
            const int tid = (int) it.get_local_id(0);
            float ss = 0.0f;
            for (int64_t c = tid; c < cols; c += 256) ss += r[c] * r[c];
            const float s = sycl::rsqrt(block_sum(it, ss, &sh[0]) / (float) cols + eps);
            sycl::group_barrier(it.get_group());
            for (int64_t c = tid; c < cols; c += 256) r[c] = s * r[c] * w[c];
        });
    });
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, const strata::kernels::RopeScaling& scaling, void* stream) {
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / 64.0f);
    const int32_t* mtab = strata::kernels::mrope_table();
    const bool scaled = scaling.type != strata::kernels::RopeScalingType::None;
    const strata::kernels::RopeKernelArgs ka = scaling.kernel_args(64);
    const strata::kernels::RopeTab tab = strata::kernels::rope_table_for(scaling);
    Q(stream).parallel_for(sycl::range<2>((size_t) (T * heads), 32), [=](sycl::id<2> id) {
        const int64_t row = (int64_t) id[0];   // t * heads + h
        const int pair = (int) id[1];          // 0..31
        const int64_t t = row / heads, hh = row % heads;
        float* p = x + t * ld + hh * dim;
        const int rp = strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair);
        float c, s;
        if (!strata::kernels::rope_tab_cs(tab, rp, pair, c, s))
            strata::kernels::rope_cos_sin((float) rp * sycl::pow(theta_scale, (float) pair), scaled, ka, pair, c, s);
        const float a = p[pair], b = p[pair + 32];
        p[pair] = a * c - b * s;
        p[pair + 32] = a * s + b * c;
    });
}
void split_q(const float* qf, float* q, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * 24 * 256)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
        q[i] = qf[t * 24 * 512 + h * 512 + d];
    });
}
void gate_attn(const float* a, const float* qf, uint16_t* o16, int64_t T, void* stream) {
    Q(stream).parallel_for(sycl::range<1>((size_t) (T * 24 * 256)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
        o16[i] = hf(a[i] * (1.0f / (1.0f + sycl::exp(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
    });
}

}  // namespace strata::prefill
