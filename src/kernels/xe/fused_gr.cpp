// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/fused_gr.cpp - the Xe port of Strata's src/kernels/cuda/fused_gr.cu (see the header): the gated residual read
// at the model's fixed geometry in two kernels, for one token or a verify window of up to kFusedGrMaxT.
//
// Work-group shapes and reduction orders are the CUDA kernels'; a warp is a sub-group of 32.  CUDA's __expf and
// rsqrtf become the precise exp and SYCL's rsqrt; explicit fmaf calls stay fused.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/runtime.hpp"

#include <cstdlib>
#include <string>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WARP = 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 groups of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` group (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80
constexpr int UPM_COLS = 16;                     // columns per multi `up` group (x 4 streams = 64 rows)
constexpr int UPM_BLOCKS = N / UPM_COLS;         // 160

using float4 = sycl::float4;
using uint4 = sycl::uint4;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
inline float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), x[2 * j], acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}
inline float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
inline uint4 load_u4(const uint16_t* p, size_t chunk) { return reinterpret_cast<const uint4*>(p)[chunk]; }

// Step 1, shared by the single and multi forms: R' = R (+ the folded previous write), xn = R' * w_norm, per-stream
// sums of squares reduced by sub-group then over the eight partials in order, rs = rsqrt(ss/N + eps).
template<class XnStore, class RsStore>
inline void norm_step(const sycl::nd_item<1>& it, const FusedGrArgs& a, float* xn, const sycl::local_accessor<float, 1>& part,
                      const sycl::local_accessor<float, 1>& s_rs, XnStore, RsStore write_rs) {
    const sycl::sub_group sg = it.get_sub_group();
    const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
    float gw[HC];
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = load4(a.R + i);
        if (a.apply) {
            const float4 b = load4(a.bo_prev + d);
            r.x() = sycl::fma(b.x(), gw[c], r.x()); r.y() = sycl::fma(b.y(), gw[c], r.y());
            r.z() = sycl::fma(b.z(), gw[c], r.z()); r.w() = sycl::fma(b.w(), gw[c], r.w());
        }
        const float4 g = load4(a.w_norm + i);
        const float sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        xn[i] = r.x() * g.x(); xn[i + 1] = r.y() * g.y(); xn[i + 2] = r.z() * g.z(); xn[i + 3] = r.w() * g.w();
    }
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(sg, ss[c]);
        if (lane == 0) part[warp * HC + c] = v;
    }
    sycl::group_barrier(it.get_group());
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w * HC + t];
        s_rs[t] = sycl::rsqrt(s / (float) N + a.eps);
        write_rs(t, (float) s_rs[t]);
    }
    sycl::group_barrier(it.get_group());
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

}  // namespace

namespace {
constexpr int PR = LR+HC;
constexpr int R2 = 1;
template<int S>
void read_v3(sycl::queue& q, const GrMulti& m) {
    float* part = m.xn;
    float* ssg = part+(size_t) m.T*HC*S*PR;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> xs(sycl::range<1>((size_t) kFusedGrMaxT*N/S),h);
        sycl::local_accessor<float,2> red(sycl::range<2>(WARPS,kFusedGrMaxT),h);
        h.parallel_for(sycl::nd_range<2>({(size_t) (DOWN_BLOCKS+1)*THREADS, (size_t) HC*S},{THREADS,1}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg=it.get_sub_group();
            const int t=(int) it.get_local_id(0),lane=(int) sg.get_local_linear_id(),warp=(int) sg.get_group_linear_id();
    const int T = m.T;
    // S = 2: each stream's 2560 columns in two halves ((int) it.get_group(1) = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = (int) it.get_group(0), c = (int) it.get_group(1) / S, h = (int) it.get_group(1) - ((int) it.get_group(1) / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    uint4 wv[R2][TQS];
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N + (size_t) h * SL);
#pragma unroll
        for (int q = 0; q < TQS; ++q) wv[r][q] = *(w4 + lane + (ptrdiff_t) 32 * q);
    }
    float ssp[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        ssp[k] = 0.0f;
        if (k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);
        const float4* G4 = reinterpret_cast<const float4*>(a.w_norm + (size_t) c * N + (size_t) h * SL);
        const float4* B4 = a.apply ? reinterpret_cast<const float4*>(a.bo_prev + (size_t) h * SL) : nullptr;
        float4* X4 = reinterpret_cast<float4*>(&xs[(size_t) k * SL]);
        for (int i = t; i < SL / 4; i += THREADS) {
            float4 r = R4[i];
            if (a.apply) {
                const float4 b = B4[i];
                r.x() = sycl::fma(b.x(), gw, r.x()); r.y() = sycl::fma(b.y(), gw, r.y());
                r.z() = sycl::fma(b.z(), gw, r.z()); r.w() = sycl::fma(b.w(), gw, r.w());
            }
            const float4 g = G4[i];
            ssp[k] += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            X4[i] = float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
        }
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        const float v = warp_sum(sg,ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    sycl::group_barrier(it.get_group());
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[r][q], &xs[(size_t) k * SL + (size_t) j * 8]);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float v = warp_sum(sg,acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
        });
    });
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,2> lo(sycl::range<2>(kFusedGrMaxT,LR),h), rsS(sycl::range<2>(kFusedGrMaxT,HC),h);
        sycl::local_accessor<float,3> g(sycl::range<3>(kFusedGrMaxT,HC,UPM_COLS),h);
        h.parallel_for(sycl::nd_range<1>((size_t) UPM_BLOCKS*THREADS,THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg=it.get_sub_group();
            const int t=(int) it.get_local_id(0),lane=(int) sg.get_local_linear_id(),warp=(int) sg.get_group_linear_id();
    const int T = m.T;
    const int d0 = (int) it.get_group(0) * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = sycl::rsqrt(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if ((int) it.get_group(0) == 0) m.a[k].rs[c] = r;
    }
    sycl::group_barrier(it.get_group());
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + sycl::exp(-x));
        if ((int) it.get_group(0) == 0) m.a[k].lo[r] = lo[k][r];
    }
    if ((int) it.get_group(0) == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = sycl::fma(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    sycl::group_barrier(it.get_group());
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const uint4 wa = *(w4 + lane);
        const uint4 wb = lane < LR / 8 - 32 ? *(w4 + 32 + lane) : uint4(0,0,0,0);
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < kFusedGrMaxT && lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, &lo[k][(size_t) lane * 8]);
            if (lane < LR / 8 - 32) acc += dot8(wb, &lo[k][(size_t) (32 + lane) * 8]);
            acc = warp_sum(sg,acc);
            if (lane == k) mine = acc;
        }
        if (lane < kFusedGrMaxT && lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    sycl::group_barrier(it.get_group());
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
        });
    });
}
}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr)
        throw core::DeviceError("fused_gr_read_multi: invalid arguments");
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm)
            throw core::DeviceError("fused_gr_read_multi: invalid arguments for token " + std::to_string(t));
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    auto& q = queue_for(stream);

    static const bool v3=[] { const char* v=std::getenv("STRATA_GR_V3"); return v && std::strtol(v, nullptr, 10)!=0; }();
    if (v3) {
        const size_t limit=q.get_device().get_info<sycl::info::device::local_mem_size>();
        const size_t extra=(size_t) WARPS*kFusedGrMaxT*sizeof(float);
        if ((size_t) kFusedGrMaxT*N*sizeof(float)+extra<=limit) { read_v3<1>(q,m); if (!stream) core::Runtime::get().finish(q); return; }
        if ((size_t) kFusedGrMaxT*(N/2)*sizeof(float)+extra<=limit) { read_v3<2>(q,m); if (!stream) core::Runtime::get().finish(q); return; }
    }
    // Step 1 of the single-token down kernel, one group per token and stream (upstream dbb1c23's split: T groups
    // left most of the GPU idle).  A work-item visits its stream's elements in the order norm_step does and the
    // eight sub-group partials are added in the same order, so rs and xn (the product, then the scale) are its bits.
    // STRATA_HC_SPLIT=0: one group per token (norm_step).
    static const bool split = [] {
        const char* v = std::getenv("STRATA_HC_SPLIT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (split) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>(WARPS), h);
            sycl::local_accessor<float, 1> s_rs(sycl::range<1>(1), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_tok * HC * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int k = (int) it.get_group(0) / HC, c = (int) it.get_group(0) % HC;
                const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
                const int warp = (int) sg.get_group_linear_id();
                const FusedGrArgs& a = m.a[k];
                float* xn = m.xn + (size_t) k * D;
                const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
                float ss = 0.0f;
                for (int i = t * 4; i < D; i += THREADS * 4) {
                    if (i / N != c) continue;
                    const int d = i - c * N;
                    float4 r = load4(a.R + i);
                    if (a.apply) {
                        const float4 b = load4(a.bo_prev + d);
                        r.x() = sycl::fma(b.x(), gw, r.x()); r.y() = sycl::fma(b.y(), gw, r.y());
                        r.z() = sycl::fma(b.z(), gw, r.z()); r.w() = sycl::fma(b.w(), gw, r.w());
                    }
                    const float4 g = load4(a.w_norm + i);
                    ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
                    xn[i] = r.x() * g.x(); xn[i + 1] = r.y() * g.y(); xn[i + 2] = r.z() * g.z(); xn[i + 3] = r.w() * g.w();
                }
                const float v = warp_sum(sg, ss);
                if (lane == 0) part[warp] = v;
                sycl::group_barrier(it.get_group());
                if (t == 0) {
                    float sum = 0.0f;
                    for (int w = 0; w < WARPS; ++w) sum += part[w];
                    s_rs[0] = sycl::rsqrt(sum / (float) N + a.eps);
                    a.rs[c] = s_rs[0];
                }
                sycl::group_barrier(it.get_group());
                const float rs = s_rs[0];
                for (int i = c * N + t; i < (c + 1) * N; i += THREADS) xn[i] *= rs;
            });
        });
    } else {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) WARPS * HC), h);
            sycl::local_accessor<float, 1> s_rs(sycl::range<1>(HC), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_tok * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const size_t k = it.get_group(0);
                const FusedGrArgs& ak = m.a[k];
                norm_step(it, ak, m.xn + k * D, part, s_rs, 0, [&](int t, float v) { ak.rs[t] = v; });
            });
        });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, &q);

    // Step 2 for T tokens: one sub-group per (row, token), the token in the second dimension.  A lane sums its 40
    // chunks (lane + 32 q) in order and the sub-group then reduces by xor, as the single-token kernel does, so each
    // output is bitwise the single-token one; rows and tokens are independent, so splitting them gives the GPU
    // T times the sub-groups (one per row alone left most of it idle) and needs no local staging.
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (DOWN_BLOCKS + 1) * THREADS, (size_t) n_tok),
                                         sycl::range<2>(THREADS, 1)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int k = (int) it.get_group(1);
            const bool inject_block = (int) it.get_group(0) == DOWN_BLOCKS;
            const int row = inject_block ? warp : (int) it.get_group(0) * WARPS + warp;
            if (inject_block && (m.a[0].w_inject == nullptr || warp >= HC)) return;
            const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D;
            const float* x = m.xn + (size_t) k * D;
            float acc = 0.0f;
#pragma unroll 8
            for (int qq = 0; qq < D / 8 / WARP; ++qq) {
                const int j = lane + WARP * qq;
                const uint4 w = load_u4(wrow, (size_t) j);
                const float4 x0 = load4(x + (size_t) j * 8), x1 = load4(x + (size_t) j * 8 + 4);
                const float xv[8] = {x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()};
                acc += dot8(w, xv);
            }
            const float sum = warp_sum(sg, acc);
            if (lane != 0) return;
            if (inject_block) {
                m.a[k].inject_out[row] = sum;
            } else {
                const float v = sum / (float) HC;
                m.a[k].lo[row] = v / (1.0f + sycl::exp(-v));
            }
        });
    });
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, &q);

    // The up projection for T tokens: each row of w_up read once, lane k runs token k's epilogue.
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lo(sycl::range<1>(kFusedGrMaxT * LR), h);
        sycl::local_accessor<float, 1> g(sycl::range<1>(kFusedGrMaxT * HC * UPM_COLS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) UPM_BLOCKS * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int T = m.T;
            const int d0 = (int) it.get_group(0) * UPM_COLS;
            for (int i = t; i < T * LR; i += THREADS) lo[i] = m.a[i / LR].lo[i % LR];
            sycl::group_barrier(it.get_group());
            for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
                const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
                const uint16_t* wrow = m.a[0].w_up + (size_t) i * LR;
                const uint4 wa = load_u4(wrow, (size_t) lane);
                const uint4 wb = lane < LR / 8 - 32 ? load_u4(wrow, (size_t) (32 + lane)) : uint4(0, 0, 0, 0);
                float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
                bool apply = false;
                if (lane < T) {
                    const FusedGrArgs& a = m.a[lane];
                    rv = a.R[i];
                    wn = a.w_norm[i];
                    rsc = a.rs[c];
                    apply = a.apply;
                    if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
                }
                float mine = 0.0f;
                for (int k = 0; k < kFusedGrMaxT; ++k) {
                    if (k >= T) break;
                    float xa[8], xb[8];
                    for (int e2 = 0; e2 < 8; ++e2) xa[e2] = lo[k * LR + lane * 8 + e2];
                    float acc = dot8(wa, xa);
                    if (lane < LR / 8 - 32) {
                        for (int e2 = 0; e2 < 8; ++e2) xb[e2] = lo[k * LR + (32 + lane) * 8 + e2];
                        acc += dot8(wb, xb);
                    }
                    acc = warp_sum(sg, acc);
                    if (lane == k) mine = acc;
                }
                if (lane < T) {
                    if (apply) {
                        rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                        m.a[lane].R_out[i] = rv;
                    }
                    const float x = rv * wn * rsc;
                    g[(lane * HC + c) * UPM_COLS + dd] = x * sigmoidf_(mine);
                }
            }
            sycl::group_barrier(it.get_group());
            for (int i = t; i < T * UPM_COLS; i += THREADS) {
                const int k = i / UPM_COLS, col = i - k * UPM_COLS;
                float s = 0.0f;
                for (int c = 0; c < HC; ++c) s += g[(k * HC + c) * UPM_COLS + col];
                m.a[k].mixed[d0 + col] = s / (float) HC;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "fused_gr_read_multi");
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out))
        throw core::DeviceError("fused_gr_read: invalid arguments");
    auto& q = queue_for(stream);
    const FusedGrArgs args = a;
    // the down projection (and the inject rows in one extra group), with the norm computed into local memory
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> xn(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> part(sycl::range<1>(WARPS * HC), h);
        sycl::local_accessor<float, 1> s_rs(sycl::range<1>(HC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (DOWN_BLOCKS + 1) * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const bool first = it.get_group(0) == 0;
            float* xnp = &xn[0];
            norm_step(it, args, xnp, part, s_rs, 0, [&](int t, float v) { if (first) args.rs[t] = v; });
            sycl::group_barrier(it.get_group());
            // one sub-group per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane
            const bool inject_block = (int) it.get_group(0) == DOWN_BLOCKS;
            const int row = inject_block ? warp : (int) it.get_group(0) * WARPS + warp;
            if (inject_block && (args.w_inject == nullptr || warp >= HC)) return;
            const uint16_t* wrow = (inject_block ? args.w_inject : args.w_down) + (size_t) row * D;
            float acc = 0.0f;
            for (int j = lane; j < D / 8; j += 32) {
                float xv[8];
                for (int e2 = 0; e2 < 8; ++e2) xv[e2] = xn[j * 8 + e2];
                acc += dot8(load_u4(wrow, (size_t) j), xv);
            }
            acc = warp_sum(sg, acc);
            if (lane != 0) return;
            if (inject_block) {
                args.inject_out[row] = acc;
            } else {
                const float x = acc / (float) HC;
                args.lo[row] = x / (1.0f + sycl::exp(-x));
            }
        });
    });
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lo(sycl::range<1>(LR), h);
        sycl::local_accessor<float, 1> g(sycl::range<1>(HC * UP_COLS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) UP_BLOCKS * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int d0 = (int) it.get_group(0) * UP_COLS;
            for (int k = t; k < LR; k += THREADS) lo[k] = args.lo[k];
            sycl::group_barrier(it.get_group());
            // 128 rows (4 streams x 32 columns), 16 per sub-group: 320 bf16 = 40 chunks of 8.
            for (int r = warp; r < HC * UP_COLS; r += WARPS) {
                const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
                const uint16_t* wrow = args.w_up + (size_t) i * LR;
                float xa[8];
                for (int e2 = 0; e2 < 8; ++e2) xa[e2] = lo[lane * 8 + e2];
                float acc = dot8(load_u4(wrow, (size_t) lane), xa);
                if (lane < LR / 8 - 32) {
                    for (int e2 = 0; e2 < 8; ++e2) xa[e2] = lo[(32 + lane) * 8 + e2];
                    acc += dot8(load_u4(wrow, (size_t) (32 + lane)), xa);
                }
                acc = warp_sum(sg, acc);
                if (lane == 0) {
                    float rv = args.R[i];
                    if (args.apply) {
                        rv = sycl::fma(args.bo_prev[d0 + dd], 2.0f * sigmoidf_(args.inj_prev[c] / (float) HC), rv);
                        args.R_out[i] = rv;                       // this group owns column d0+dd of every stream
                    }
                    const float x = rv * args.w_norm[i] * args.rs[c];
                    g[c * UP_COLS + dd] = x * sigmoidf_(acc);
                }
            }
            sycl::group_barrier(it.get_group());
            if (t < UP_COLS) {
                float s = 0.0f;
                for (int c = 0; c < HC; ++c) s += g[c * UP_COLS + t];
                args.mixed[d0 + t] = s / (float) HC;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "fused_gr_read");
}

}  // namespace strata::kernels
