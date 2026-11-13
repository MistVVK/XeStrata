// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/dflash2.cpp - see include/strata/kernels/dflash2.hpp.  Written for XeStrata's SYCL runtime (the
// upstream CUDA / HIP kernels are not copied): sub-group reductions, local tiles, dp4a through xe::dp4a.  The tiled
// weight layout is the DF2P pack's (df2_q4_tile / df2_q8_tile make it at load): a group of 8 rows is one unit; per
// 512 values of the reduction axis (a "step", 16 blocks) the group stores the 8 rows one after the other, a lane
// (block * 2 + half) takes 16 values.  The fp16 block scales follow in the same tile order.
#define STRATA_DEVICE_ARCH_TAG_TU xe_dflash2
#include "strata/kernels/dflash2.hpp"

#include "strata/core/runtime.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"
#include "cuda_intrinsics.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>

namespace strata::kernels {
namespace {

namespace xe = strata::kernels::xe;
using sycl::nd_item;

constexpr int HD = kDf2Hd;
constexpr int WARP = 32;
core::TimedCache<sycl::device, int, bool> g_gemm_warmed;
#define DF2_SG32 [[sycl::reqd_sub_group_size(32)]]

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }

inline float bf16f(uint16_t h) { return sycl::bit_cast<float>((uint32_t) h << 16); }
inline float xorf(const sycl::sub_group& sg, float v, int o) { return sycl::permute_group_by_xor(sg, v, (unsigned) o); }
inline int xori(const sycl::sub_group& sg, int v, int o) { return sycl::permute_group_by_xor(sg, v, (unsigned) o); }

// one 32-value block of one row to the planar activation layout (a sub-group: lane = element)
inline void quant_block(const sycl::sub_group& sg, float v, int blk, int col, int K, int8_t* xq, float* xd) {
    const int lane = (int) sg.get_local_linear_id();
    float amax = sycl::fabs(v);
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, xorf(sg, amax, o));
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q = (int) sycl::rint(v * inv);
    xq[(size_t) col * K + blk * 32 + lane] = (int8_t) q;
    if (lane == 0) xd[(size_t) col * (K / 32) + blk] = d;
}

// the sum of squares of one (row, head)'s 128 values across the 4 sub-groups of a 128-wide work-group
inline float head_ss(const nd_item<2>& it, const sycl::local_accessor<float, 1>& red, float x) {
    const sycl::sub_group sg = it.get_sub_group();
    const int d = (int) it.get_local_id(1);
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    sycl::group_barrier(it.get_group());
    return red[0] + red[1] + red[2] + red[3];
}


}  // namespace

void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out,
              int rows, int H, int group, void* stream) {
    const int n = rows * H;
    Q(stream).parallel_for(sycl::nd_range<1>((size_t) ((n + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= n) return;
        const int b = i / H, c = i % H;
        const int groups = H / group;
        float acc = 0.0f;
#pragma unroll
        for (int o = 0; o < 2; ++o) {
            if (b - o < 0) continue;
            const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
            acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
        }
        out[i] = resid ? resid[i] + acc : acc;
    });
}

void df2_rope_table_cpu(const int32_t* pos, int rows, float theta, float* table) {
    const double two_pi = 6.283185307179586476925286766559;
    for (int i = 0; i < 64; ++i) {
        const double inv = std::pow((double) theta, -(double) (2 * i) / (double) HD);
        for (int r = 0; r < rows; ++r) {
            double ang = (double) pos[r] * inv;
            ang -= two_pi * std::nearbyint(ang / two_pi);   // [-pi, pi]
            table[(r * 2 + 0) * 64 + i] = (float) std::cos((float) ang);
            table[(r * 2 + 1) * 64 + i] = (float) std::sin((float) ang);
        }
    }
}

void df2_norm_rope_qk(const float* q, const float* k, const float* gq, const float* gk, const float* table, float* qo,
                      float* ko, int rows, int q_heads, int k_heads, float eps, void* stream) {
    Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(4, hd), y(HD, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) (q_heads + k_heads), (size_t) rows * HD}, {1, HD}),
                        [=](nd_item<2> it) DF2_SG32 {
            const int hh = (int) it.get_group(0), r = (int) it.get_group(1), d = (int) it.get_local_id(1);
            const bool isq = hh < q_heads;
            const int h = isq ? hh : hh - q_heads, heads = isq ? q_heads : k_heads;
            const float* in = isq ? q : k;
            const float* gamma = isq ? gq : gk;
            float* out = isq ? qo : ko;
            const float x = in[((size_t) r * heads + h) * HD + d];
            const float tot = head_ss(it, red, x);
            const float v = x * sycl::rsqrt(tot / (float) HD + eps) * gamma[d];
            y[d] = v;
            sycl::group_barrier(it.get_group());
            const int i = d & 63;
            const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
            out[((size_t) r * heads + h) * HD + d] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
        });
    });
}

void df2_kv_ring(const float* kin, const float* vin, const float* gk, const int32_t* pos, const float* table,
                 float* kring, float* vring, int slots, int rows, int heads, float eps, void* stream,
                 int64_t* tag, const int32_t* epoch_ptr) {
    Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(4, hd), y(HD, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) heads, (size_t) rows * HD}, {1, HD}),
                        [=](nd_item<2> it) DF2_SG32 {
            const int h = (int) it.get_group(0), r = (int) it.get_group(1), d = (int) it.get_local_id(1);
            const float x = kin[((size_t) r * heads + h) * HD + d];
            const float tot = head_ss(it, red, x);
            const float v = x * sycl::rsqrt(tot / (float) HD + eps) * gk[d];
            y[d] = v;
            sycl::group_barrier(it.get_group());
            const int i = d & 63;
            const float c = table[(r * 2 + 0) * 64 + i], sn = table[(r * 2 + 1) * 64 + i];
            const int slot = (int) (pos[r] % slots);
            const size_t at = ((size_t) h * slots + slot) * HD + d;
            kring[at] = d < 64 ? v * c - y[d + 64] * sn : v * c + y[d - 64] * sn;
            vring[at] = vin[((size_t) r * heads + h) * HD + d];
            if (tag != nullptr && h == 0 && d == 0) tag[slot] = ((int64_t) *epoch_ptr << 32) | (int64_t) pos[r];
        });
    });
}

// ---- attention: partials over 64-key chunks, then the combine.  One work-group (one sub-group of 32) per
// (chunk, K/V head, 8-query group): lane = key of the 32-key tile, each lane holds 4 output dims of its 8 queries.
void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk,
                   const int32_t* anchor_pos, int ring_slots, int window, float* part, void* xq, float* xd,
                   void* stream, const int64_t* tag, const int32_t* epoch_ptr) {
    sycl::queue& qu = Q(stream);
    constexpr int CK = kDf2Chunk;
    qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> Qs({8, HD}, h), Ks({32, HD}, h);
        sycl::local_accessor<float, 2> Ps({32, 8}, h);
        h.parallel_for(sycl::nd_range<3>({(size_t) kDf2PartChunks, (size_t) kDf2Kv, 4 * WARP}, {1, 1, WARP}),
                       [=](nd_item<3> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int chunk = (int) it.get_group(0), kvh = (int) it.get_group(1), qg = (int) it.get_group(2);
            const int lane = (int) it.get_local_id(2);
            const int s = *anchor_pos;
            const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
            const int nctx = s - lo;
            const int nkeys = nctx + kDf2Rows;
            const int k_begin = chunk * CK;
            const int64_t ep = tag != nullptr ? ((int64_t) *epoch_ptr << 32) : 0;
            float* const pbase = part + (((size_t) chunk * kDf2Kv + kvh) * 32 + qg * 8) * 130;
            if (k_begin >= nkeys) {
                for (int i = lane; i < 8 * 130; i += WARP) pbase[i] = (i % 130 == 0) ? -INFINITY : 0.0f;
                return;
            }
            const float scale = sycl::rsqrt((float) HD);
            for (int i = lane; i < 8 * HD; i += WARP) {   // the group's 8 queries, scaled once
                const int ql = i / HD, d = i % HD;
                const int qi = qg * 8 + ql, row = qi >> 2, g = qi & 3;
                Qs[ql][d] = q[((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d] * scale;
            }
            float m[8], l[8], O[8][4];
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) { m[qi] = -INFINITY; l[qi] = 0.0f; }
#pragma unroll
            for (int qi = 0; qi < 8; ++qi)
#pragma unroll
                for (int j = 0; j < 4; ++j) O[qi][j] = 0.0f;
            const int k_end = sycl::min(k_begin + CK, nkeys);
            auto load_tile = [&](int k0, const float* ring, const float* blk) {
                for (int i = lane; i < 32 * HD; i += WARP) {
                    const int t = i / HD, d = i % HD, k = k0 + t;
                    float v = 0.0f;
                    if (k < nctx) {
                        const int64_t pos = lo + k;
                        const int slot = (int) (pos % ring_slots);
                        const bool ok = tag == nullptr || tag[slot] == (ep | pos);
                        if (ok) v = ring[((size_t) kvh * ring_slots + slot) * HD + d];
                    } else if (k < nkeys) {
                        v = blk[((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d];
                    }
                    Ks[t][d] = v;
                }
            };
            for (int k0 = k_begin; k0 < k_end; k0 += 32) {
                sycl::group_barrier(it.get_group());
                load_tile(k0, kring, kblk);
                sycl::group_barrier(it.get_group());
                float sc[8];
                {   // lane = key t: its dot with each of the 8 queries, masked by the window and the slot's tag
                    const int t = lane, k = k0 + t;
                    bool ctx_ok = false;
                    if (k < nctx) {
                        const int64_t pos = lo + k;
                        ctx_ok = tag == nullptr || tag[(int) (pos % ring_slots)] == (ep | pos);
                    }
#pragma unroll
                    for (int qi = 0; qi < 8; ++qi) {
                        if (k >= nkeys) { sc[qi] = -INFINITY; continue; }
                        if (k < nctx) {
                            const int row = (qg * 8 + qi) >> 2;
                            if (!ctx_ok || lo + k <= s + row - window) { sc[qi] = -INFINITY; continue; }
                        }
                        float acc = 0.0f;
#pragma unroll 4
                        for (int d = 0; d < HD; ++d) acc = sycl::fma(Qs[qi][d], Ks[t][d], acc);
                        sc[qi] = acc;
                    }
                }
                float corr[8];
#pragma unroll
                for (int qi = 0; qi < 8; ++qi) {   // online softmax: the max and the sum across the 32 keys
                    float tmax = sc[qi];
                    for (int o = 16; o > 0; o >>= 1) tmax = sycl::fmax(tmax, xorf(sg, tmax, o));
                    const float mn = sycl::fmax(m[qi], tmax);
                    corr[qi] = (m[qi] == -INFINITY) ? 0.0f : sycl::exp(m[qi] - mn);
                    const float p = sc[qi] == -INFINITY ? 0.0f : sycl::exp(sc[qi] - mn);
                    float psum = p;
                    for (int o = 16; o > 0; o >>= 1) psum += xorf(sg, psum, o);
                    l[qi] = l[qi] * corr[qi] + psum;
                    m[qi] = mn;
                    Ps[lane][qi] = p;
                }
                sycl::group_barrier(it.get_group());
                load_tile(k0, vring, vblk);
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int qi = 0; qi < 8; ++qi) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) O[qi][j] *= corr[qi];
                    for (int t = 0; t < 32; ++t) {
                        const float p = Ps[t][qi];
#pragma unroll
                        for (int j = 0; j < 4; ++j) O[qi][j] = sycl::fma(p, Ks[t][lane * 4 + j], O[qi][j]);
                    }
                }
            }
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) {
                if (lane == 0) { pbase[qi * 130] = m[qi]; pbase[qi * 130 + 1] = l[qi]; }
#pragma unroll
                for (int j = 0; j < 4; ++j) pbase[qi * 130 + 2 + lane * 4 + j] = O[qi][j];
            }
        });
    });
    // the combine: one work-group of 128 per (K/V head, query), a sub-group per 32-value output block
    qu.parallel_for(sycl::nd_range<1>((size_t) kDf2Kv * 32 * HD, HD), [=](nd_item<1> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int a = (int) it.get_group(0);
        const int kvh = a >> 5, qi = a & 31, d = (int) it.get_local_id(0);
        float M = -INFINITY;
        for (int c = 0; c < kDf2PartChunks; ++c) M = sycl::fmax(M, part[(((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130]);
        float L = 0.0f, o = 0.0f;
        for (int c = 0; c < kDf2PartChunks; ++c) {
            const float* p = part + (((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130;
            const float w = (p[0] == -INFINITY) ? 0.0f : sycl::exp(p[0] - M);
            L += p[1] * w;
            o = sycl::fma(p[2 + d], w, o);
        }
        const int row = qi >> 2, g = qi & 3;
        quant_block(sg, L > 0.0f ? o / L : 0.0f, (kvh * kDf2Gqa + g) * (HD / 32) + (d >> 5), row, 32 * HD,
                    static_cast<int8_t*>(xq), xd);
    });
}

// ---- the drafter's GEMM: one sub-group per output row, lane = block * 2 + half (16 values a lane).  The matrices
// of one launch share the activations; each runs as its own kernel (the fused launch is a later speed idea).
namespace {

template <int NC, bool Q4, bool PAIR16>
void gemm_launch(sycl::queue& qu, const void* w, const uint16_t* dpl, float* y, int n_out, int n_in,
                 const int8_t* xq, const float* xd) {
    const int nb = n_in / 32, steps = nb / 16;
    qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> halves(PAIR16 ? NC * WARP : 1, h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * WARP, WARP),
                       [=](nd_item<1> it) [[sycl::reqd_sub_group_size(PAIR16 ? 16 : 32)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int row = (int) it.get_group(0), lane = (int) it.get_local_id(0);
        const int g = row / 8, r = row % 8;
        const uint16_t* wd = dpl + (size_t) g * 8 * nb + r * 16 + (lane >> 1);
        const sycl::int4* xv = reinterpret_cast<const sycl::int4*>(xq) + lane;
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
        if constexpr (Q4) {
            const sycl::uint2* wq = reinterpret_cast<const sycl::uint2*>(w) + (size_t) g * (n_in / 2) + r * 32 + lane;
            for (int st = 0; st < steps; ++st) {
                const sycl::uint2 raw = wq[(size_t) st * 256];
                // bytewise (q - 8) for q = 0..15: ((q | 0x80) - 8) ^ 0x80, no borrow across bytes
                const int w0 = (int) ((((raw.x() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                const int w1 = (int) (((((raw.x() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                const int w2 = (int) ((((raw.y() & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                const int w3 = (int) (((((raw.y() >> 4) & 0x0F0F0F0Fu) | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
                const float dw = (float) sycl::bit_cast<sycl::half>(wd[(size_t) st * 128]);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const sycl::int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
                    int isum = xe::dp4a(w0, x.x(), 0);
                    isum = xe::dp4a(w1, x.y(), isum);
                    isum = xe::dp4a(w2, x.z(), isum);
                    isum = xe::dp4a(w3, x.w(), isum);
                    const float xs = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
                    acc[c] = sycl::fma(dw * xs, (float) isum, acc[c]);
                }
            }
        } else {
            const sycl::int4* wq = reinterpret_cast<const sycl::int4*>(w) + (size_t) g * (n_in / 2) + r * 32 + lane;
            for (int st = 0; st < steps; ++st) {
                const sycl::int4 raw = wq[(size_t) st * 256];
                const float dw = (float) sycl::bit_cast<sycl::half>(wd[(size_t) st * 128]);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const sycl::int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
                    int isum = xe::dp4a(raw.x(), x.x(), 0);
                    isum = xe::dp4a(raw.y(), x.y(), isum);
                    isum = xe::dp4a(raw.z(), x.z(), isum);
                    isum = xe::dp4a(raw.w(), x.w(), isum);
                    const float xs = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
                    acc[c] = sycl::fma(dw * xs, (float) isum, acc[c]);
                }
            }
        }
        if constexpr (PAIR16) {
#pragma unroll
            for (int c = 0; c < NC; ++c) halves[c * WARP + lane] = acc[c];
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float v = acc[c] + halves[c * WARP + (lane ^ 16)];
                for (int o = 8; o > 0; o >>= 1) v += xorf(sg, v, o);
                if (lane == 0) y[(size_t) c * n_out + row] = v;
            }
        } else {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float v = acc[c];
                for (int o = 16; o > 0; o >>= 1) v += xorf(sg, v, o);
                if (lane == 0) y[(size_t) c * n_out + row] = v;
            }
        }
        });
    });
}

template <bool Q4, bool PAIR16>
void gemm_nc(sycl::queue& qu, const void* w, const uint16_t* dpl, float* y, int n_out, int n_in,
             const int8_t* xq, const float* xd, int ncols) {
    switch (ncols) {
        case 1: gemm_launch<1, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 2: gemm_launch<2, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 3: gemm_launch<3, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 4: gemm_launch<4, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 5: gemm_launch<5, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 6: gemm_launch<6, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        case 7: gemm_launch<7, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
        default: gemm_launch<8, Q4, PAIR16>(qu, w, dpl, y, n_out, n_in, xq, xd); break;
    }
}

}  // namespace

void df2_gemm(const Df2Mat* mats, int n, const void* xq, const float* xd, int n_in, int ncols, void* stream) {
    if (n < 1 || n > 3 || n_in % 512 != 0 || ncols < 1 || ncols > 8) throw std::invalid_argument("df2_gemm: unsupported shape");
    sycl::queue& qu = Q(stream);
    const int8_t* xqp = static_cast<const int8_t*>(xq);
    for (int i = 0; i < n; ++i) {
        if (mats[i].n_out % 16 != 0) throw std::invalid_argument("df2_gemm: n_out must be a multiple of 16");
        const uint8_t* w = static_cast<const uint8_t*>(mats[i].w);
        const size_t plane = mats[i].q4 ? (size_t) mats[i].n_out * n_in / 2 : (size_t) mats[i].n_out * n_in;
        const uint16_t* dpl = reinterpret_cast<const uint16_t*>(w + plane);
        bool pair16 = false;
        if (mats[i].q4) {
            const char* mode = strata::env_get("DF2_GEMM_PAIR16");
            const auto sizes = qu.get_device().get_info<sycl::info::device::sub_group_sizes>();
            const bool supported = std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
            if (mode && mode[0] == '1' && mode[1] == '\0') {
                if (!supported) throw std::runtime_error("XESTRATA_DF2_GEMM_PAIR16 needs subgroup size 16");
                pair16 = true;
            } else if (!mode || !*mode || std::string(mode) == "auto") {
                using Key = std::tuple<int, int, int>;
                static core::TimedCache<sycl::device, Key, bool> choices;
                const Key key{mats[i].n_out, n_in, ncols};
                const auto cached = choices.get(qu.get_device(), key);
                if (cached) pair16 = *cached;
                else if (supported && core::timing_depth == 0 &&
                         qu.ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::recording) {
                    const bool first = g_gemm_warmed.put_first(qu.get_device(), 0, true, [](int) { return true; }).second;
                    const auto measured = core::timed_choice({
                        {[&] { for (int r = 0; r < 4; ++r) gemm_nc<true, false>(qu, w, dpl, mats[i].y, mats[i].n_out, n_in, xqp, xd, ncols); return true; }},
                        {[&] { for (int r = 0; r < 4; ++r) gemm_nc<true, true>(qu, w, dpl, mats[i].y, mats[i].n_out, n_in, xqp, xd, ncols); return true; }}
                    }, [&] { qu.wait_and_throw(); }, 5, 1.10, first ? 0.5 : 0.0);
                    pair16 = choices.put(qu.get_device(), key, measured.pick == 1);
                    std::fprintf(stderr, "xestrata: DFlash2 Q4 GEMM %s (%d x %d x %d; %.1f / %.1f us)\n",
                                 pair16 ? "paired subgroup" : "one subgroup", mats[i].n_out, n_in, ncols,
                                 measured.seconds[0] * 1e6 / 4, measured.seconds[1] * 1e6 / 4);
                }
            } else if (!(mode[0] == '0' && mode[1] == '\0')) {
                throw std::runtime_error("XESTRATA_DF2_GEMM_PAIR16 must be auto, 0 or 1");
            }
        }
        if (mats[i].q4 && pair16) gemm_nc<true, true>(qu, w, dpl, mats[i].y, mats[i].n_out, n_in, xqp, xd, ncols);
        else if (mats[i].q4) gemm_nc<true, false>(qu, w, dpl, mats[i].y, mats[i].n_out, n_in, xqp, xd, ncols);
        else gemm_nc<false, false>(qu, w, dpl, mats[i].y, mats[i].n_out, n_in, xqp, xd, ncols);
    }
}


void df2_quant_rows(const float* x, int K, int ncols, void* xq, float* xd, void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    Q(stream).parallel_for(sycl::nd_range<2>({(size_t) ncols, (size_t) ((K / 32 + 7) / 8) * 256}, {1, 256}),
                           [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int col = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5);
        const int lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= K) return;
        quant_block(sg, x[(size_t) col * K + blk * 32 + lane], blk, col, K, xqi, xd);
    });
}

void df2_rms_quant(const float* h, const float* gamma, float* xn, int rows, int N, float eps, void* xq, float* xd,
                   void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int col = (int) it.get_group(0), tid = (int) it.get_local_id(0), warp = tid >> 5, lane = tid & 31;
            float ss = 0.0f;
            for (int i = tid; i < N; i += 256) { const float v = h[(size_t) col * N + i]; ss += v * v; }
            for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
            if (lane == 0) red[warp] = ss;
            sycl::group_barrier(it.get_group());
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[w];
            const float scale = sycl::rsqrt(tot / (float) N + eps);
            for (int blk = warp; blk * 32 < N; blk += 8) {
                const int i = blk * 32 + lane;
                const float v = h[(size_t) col * N + i] * scale * gamma[i];
                if (xn) xn[(size_t) col * N + i] = v;
                quant_block(sg, v, blk, col, N, xqi, xd);
            }
        });
    });
}

void df2_conv_quant(const float* x, const float* dyn, int dyn_stride, const float* base, int rows, int H, int group,
                    void* xq, float* xd, void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    Q(stream).parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((H / 32 + 7) / 8) * 256}, {1, 256}),
                           [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int b = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5);
        const int lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= H) return;
        const int c = blk * 32 + lane, groups = H / group;
        float acc = 0.0f;
#pragma unroll
        for (int o = 0; o < 2; ++o) {
            if (b - o < 0) continue;
            const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
            acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
        }
        quant_block(sg, acc, blk, b, H, xqi, xd);
    });
}

void df2_conv_rms_quant(const float* x, const float* dyn, int dyn_stride, const float* base, float* h,
                        const float* gamma, float* xn, int rows, int H, int group, float eps, void* xq, float* xd,
                        void* stream) {
    constexpr int MAXJ = 12;   // H <= 3072 -> 96 blocks -> 12 a sub-group
    if (H > 3072) throw std::invalid_argument("df2_conv_rms_quant: H too large");
    int8_t* xqi = static_cast<int8_t*>(xq);
    Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int b = (int) it.get_group(0), tid = (int) it.get_local_id(0), warp = tid >> 5, lane = tid & 31;
            const int groups = H / group, nb = H / 32;
            float v[MAXJ];
            float ss = 0.0f;
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int blk = warp + j * 8;
                v[j] = 0.0f;
                if (blk < nb) {
                    const int c = blk * 32 + lane;
                    float acc = 0.0f;
#pragma unroll
                    for (int o = 0; o < 2; ++o) {
                        if (b - o < 0) continue;
                        const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
                        acc = sycl::fma(k, x[(size_t) (b - o) * H + c], acc);
                    }
                    v[j] = h[(size_t) b * H + c] + acc;
                    h[(size_t) b * H + c] = v[j];
                    ss += v[j] * v[j];
                }
            }
            for (int o = 16; o > 0; o >>= 1) ss += xorf(sg, ss, o);
            if (lane == 0) red[warp] = ss;
            sycl::group_barrier(it.get_group());
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[w];
            const float scale = sycl::rsqrt(tot / (float) H + eps);
#pragma unroll
            for (int j = 0; j < MAXJ; ++j) {
                const int blk = warp + j * 8;
                if (blk < nb) {
                    const int c = blk * 32 + lane;
                    const float y = v[j] * scale * gamma[c];
                    if (xn) xn[(size_t) b * H + c] = y;
                    quant_block(sg, y, blk, b, H, xqi, xd);
                }
            }
        });
    });
}

void df2_swiglu_quant(const float* gate, const float* up, int rows, int I, void* xq, float* xd, void* stream) {
    int8_t* xqi = static_cast<int8_t*>(xq);
    Q(stream).parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((I / 32 + 7) / 8) * 256}, {1, 256}),
                           [=](nd_item<2> it) DF2_SG32 {
        const sycl::sub_group sg = it.get_sub_group();
        const int b = (int) it.get_group(0), blk = (int) it.get_group(1) * 8 + (int) (it.get_local_id(1) >> 5);
        const int lane = (int) sg.get_local_linear_id();
        if (blk * 32 >= I) return;
        const size_t i = (size_t) b * I + blk * 32 + lane;
        const float g = gate[i];
        quant_block(sg, g / (1.0f + sycl::exp(-g)) * up[i], blk, b, I, xqi, xd);
    });
}

void df2_q8_tile(const void* blocks_, void* qs_, void* dpl_, int n_in, int64_t count, void* stream) {
    const uint8_t* blocks = static_cast<const uint8_t*>(blocks_);
    uint8_t* qs = static_cast<uint8_t*>(qs_);
    uint16_t* dpl = static_cast<uint16_t*>(dpl_);
    Q(stream).parallel_for(sycl::nd_range<1>((size_t) ((count + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int64_t b = (int64_t) it.get_global_id(0);
        if (b >= count) return;
        const int nb = n_in / 32;
        const int64_t row = b / nb;
        const int blk = (int) (b % nb);
        const int64_t g = row / 8;
        const int r = (int) (row % 8), st = blk / 16, bi = blk % 16;
        const uint8_t* src = blocks + b * 34;
        dpl[(g * 8 * nb) + (int64_t) st * 128 + r * 16 + bi] = (uint16_t) src[0] | ((uint16_t) src[1] << 8);
        uint8_t* o = qs + g * 8 * n_in + (int64_t) st * 4096 + r * 512 + bi * 32;
        for (int i = 0; i < 32; ++i) o[i] = src[2 + i];
    });
}

void df2_q4_tile(const void* blocks_, void* qs_, void* dpl_, int n_in, int64_t count, void* stream) {
    const uint8_t* blocks = static_cast<const uint8_t*>(blocks_);
    uint8_t* qs = static_cast<uint8_t*>(qs_);
    uint16_t* dpl = static_cast<uint16_t*>(dpl_);
    Q(stream).parallel_for(sycl::nd_range<1>((size_t) ((count + 255) / 256) * 256, 256), [=](nd_item<1> it) {
        const int64_t b = (int64_t) it.get_global_id(0);
        if (b >= count) return;
        const int nb = n_in / 32;
        const int64_t row = b / nb;
        const int blk = (int) (b % nb);
        const int64_t g = row / 8;
        const int r = (int) (row % 8), st = blk / 16, bi = blk % 16;
        const uint8_t* src = blocks + b * 18;
        dpl[(g * 8 * nb) + (int64_t) st * 128 + r * 16 + bi] = (uint16_t) src[0] | ((uint16_t) src[1] << 8);
#pragma unroll
        for (int half = 0; half < 2; ++half) {   // values half * 16 .. +15 of the block
            uint32_t wd[2];
#pragma unroll
            for (int w = 0; w < 2; ++w) {   // word w: byte p = (value w*8+p, value w*8+4+p) as (low, high) nibble
                uint32_t v = 0;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint8_t b_lo = src[2 + w * 8 + p], b_hi = src[2 + w * 8 + 4 + p];
                    const uint32_t q_lo = half == 0 ? (b_lo & 15u) : (b_lo >> 4), q_hi = half == 0 ? (b_hi & 15u) : (b_hi >> 4);
                    v |= (q_lo | (q_hi << 4)) << (p * 8);
                }
                wd[w] = v;
            }
            uint8_t* o = qs + g * 8 * (n_in / 2) + (int64_t) st * 2048 + r * 256 + (bi * 2 + half) * 8;
            for (int i = 0; i < 4; ++i) { o[i] = (uint8_t) (wd[0] >> (8 * i)); o[4 + i] = (uint8_t) (wd[1] >> (8 * i)); }
        }
    });
}

// ---- top-16: 64 partial blocks a row, then the merge of their 16 each
namespace {
constexpr int TK_CHUNK_MAX = 4096;
inline bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }
}  // namespace

void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok,
              void* stream) {
    if ((n + kDf2TopBlocks - 1) / kDf2TopBlocks > TK_CHUNK_MAX) throw std::invalid_argument("df2_topk: too many logits per row");
    float* cval = reinterpret_cast<float*>(scratch);
    int32_t* cidx = reinterpret_cast<int32_t*>(scratch + (size_t) rows * kDf2TopBlocks * kDf2TopK);
    sycl::queue& qu = Q(stream);
    qu.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> vals(TK_CHUNK_MAX, hd), wv(8, hd);
        sycl::local_accessor<int, 1> wi(8, hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) kDf2TopBlocks * 256}, {1, 256}),
                        [=](nd_item<2> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int row = (int) it.get_group(0), blk = (int) it.get_group(1), tid = (int) it.get_local_id(1);
            const int cs = (n + kDf2TopBlocks - 1) / kDf2TopBlocks;
            const int c0 = blk * cs, c1 = sycl::min(c0 + cs, n);
            for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[(size_t) row * n + c0 + i] : -INFINITY;
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < kDf2TopK; ++k) {
                float bv = -INFINITY;
                int bi = 0x7fffffff;
                for (int i = tid; i < cs; i += 256)
                    if (better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = xorf(sg, bv, o);
                    const int oi = xori(sg, bi, o);
                    if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
                }
                if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    float v = wv[0];
                    int ix = wi[0];
                    for (int w = 1; w < 8; ++w)
                        if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
                    const size_t o = ((size_t) row * kDf2TopBlocks + blk) * kDf2TopK + k;
                    cval[o] = v;
                    cidx[o] = v == -INFINITY ? -1 : c0 + ix;
                    if (v != -INFINITY) vals[ix] = -INFINITY;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
    qu.submit([&](sycl::handler& hd) {
        constexpr int NC = kDf2TopBlocks * kDf2TopK;
        sycl::local_accessor<float, 1> sv(NC, hd), wv(8, hd);
        sycl::local_accessor<int, 1> si(NC, hd), wi(8, hd), wp(8, hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int row = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            for (int i = tid; i < NC; i += 256) { sv[i] = cval[(size_t) row * NC + i]; si[i] = cidx[(size_t) row * NC + i]; }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < kDf2TopK; ++k) {
                float bv = -INFINITY;
                int bi = 0x7fffffff, bp = -1;
                for (int i = tid; i < NC; i += 256)
                    if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = xorf(sg, bv, o);
                    const int oi = xori(sg, bi, o);
                    const int op = xori(sg, bp, o);
                    if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
                }
                if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; wp[tid >> 5] = bp; }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    float v = wv[0];
                    int ix = wi[0], p = wp[0];
                    for (int w = 1; w < 8; ++w)
                        if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
                    if (p >= 0) si[p] = -1;
                    val[row * kDf2TopK + k] = v;
                    tok[row * kDf2TopK + k] = p >= 0 ? (idx_map ? idx_map[ix] : ix) : 0;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// ---- the selector's walk: one work-group of 256, the rows left to right
namespace {
// one codebook value: bf16 rows, or Q4_0 rows in block order (rank / 32 blocks of 18 bytes, NOT tiled)
inline float cb_val(const uint16_t* cb, int64_t row, int v, int rank, bool q4) {
    if (!q4) return bf16f(cb[row * rank + v]);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(cb);
    const int b = v / 32, j = v % 32;
    const uint8_t* blk = bytes + row * (rank / 32) * 18 + b * 18;
    const uint16_t dbits = (uint16_t) ((uint16_t) blk[0] | (uint16_t) ((uint16_t) blk[1] << 8));
    const float d = (float) sycl::bit_cast<sycl::half>(dbits);
    const uint8_t byte = blk[2 + (j & 15)];
    const int q = j < 16 ? (byte & 15) : (byte >> 4);
    return (float) (q - 8) * d;
}
}  // namespace

void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb,
                int rank, int rows, const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream,
                bool cb_q4) {
    if (rank != 256) throw std::invalid_argument("df2_select: the selector rank must be 256");
    Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> vp(256, hd), sc(kDf2TopK, hd);
        sycl::local_accessor<int, 1> prev(1, hd);
        hd.parallel_for(sycl::nd_range<1>(256, 256), [=](nd_item<1> it) DF2_SG32 {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0), w = tid >> 5, lane = tid & 31;
            if (tid == 0) prev[0] = *anchor_tok;
            sycl::group_barrier(it.get_group());
            for (int i = 0; i < rows; ++i) {
                vp[tid] = cb_val(pred_cb, prev[0], tid, rank, cb_q4) * hp[(size_t) i * rank + tid];
                sycl::group_barrier(it.get_group());
                for (int j = w * 2; j < w * 2 + 2; ++j) {
                    const int t = tok[i * kDf2TopK + j];
                    float acc = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        acc = sycl::fma(cb_val(succ_cb, t, lane * 8 + e, rank, cb_q4), vp[lane * 8 + e], acc);
                    for (int o = 16; o > 0; o >>= 1) acc += xorf(sg, acc, o);
                    if (lane == 0) sc[j] = unary[i * kDf2TopK + j] + acc;
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    int bj = 0;
                    float bv = sc[0];
                    for (int j = 1; j < kDf2TopK; ++j)
                        if (sc[j] > bv) { bv = sc[j]; bj = j; }
                    float z = 0.0f;
                    for (int j = 0; j < kDf2TopK; ++j) z += sycl::exp(sc[j] - bv);
                    out_tok[i] = tok[i * kDf2TopK + bj];
                    out_prob[i] = 1.0f / z;
                    prev[0] = out_tok[i];
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream) {
    Q(stream).parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((n_embd + 255) / 256) * 256}, {1, 256}),
                           [=](nd_item<2> it) {
        const int64_t i = (int64_t) it.get_group(1) * 256 + (int64_t) it.get_local_id(1);
        const int r = (int) it.get_group(0);
        if (i >= n_embd) return;
        float s = 0.0f;
        for (int c = 0; c < hc; ++c) s += R[((size_t) r * hc + c) * n_embd + i];
        out[(size_t) r * out_stride + i] = s / (float) hc;
    });
}

void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject,
                int64_t inj_stride, float* out, int64_t out_stride, int rows, int64_t n_embd, int hc, void* stream) {
    Q(stream).parallel_for(sycl::nd_range<2>({(size_t) rows, (size_t) ((n_embd + 255) / 256) * 256}, {1, 256}),
                           [=](nd_item<2> it) {
        const int64_t i = (int64_t) it.get_group(1) * 256 + (int64_t) it.get_local_id(1);
        const int r = (int) it.get_group(0);
        if (i >= n_embd) return;
        float wsum = 0.0f, rs = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const float x = inject[(size_t) r * inj_stride + c] / (float) hc;
            wsum += 2.0f / (1.0f + sycl::exp(-x));
            rs += R[(size_t) r * R_stride + (size_t) c * n_embd + i];
        }
        out[(size_t) r * out_stride + i] = (rs + bo[(size_t) r * bo_stride + i] * wsum) / (float) hc;
    });
}

}  // namespace strata::kernels