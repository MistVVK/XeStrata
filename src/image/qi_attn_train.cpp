// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn_train.cpp - a training step's attention on the engine's shared parts (qi_kernels.hpp): the
// forward through the shared attention kernel (kernels::flash_attn, the one inference runs) with its FP32 output and
// each row's log-sum-exp, the backward from that log-sum-exp through the shared products (prefill::Gemm).
//
// The backward takes a head at a time and its query rows a tile at a time (FlashAttention-2's five products a tile:
// S = Q K^T, dP = dO V^T, dQ = dS K, dV += P^T dO, dK += dS^T Q): P comes back from S and the forward's log-sum-exp,
// so nothing of rows x rows is kept, and a tile's maps are the most it holds.  The products take FP16: dO enters at a
// power of two a head that brings its largest value near the top of FP16's range, dS at one a tile, each taken back
// out of the products' FP32 sums; P is in [0, 1] as it is.

#include "strata/image/qi_kernels.hpp"
#include "strata/image/train_ops.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image {

namespace {

constexpr int64_t kWG = 256;
constexpr int kT = 32;          // the maps' kernel's tile: 32 x 32 through local memory
constexpr int64_t kTile = 1024; // query rows a backward tile (a multiple of kT): a 4096-key head's maps ~57 MB

// one head's rows out of the joint [rows][heads * hd] rows into contiguous [rows][hd]: FP16 as it is, FP32 to FP16
// at `scale` (a head's power of two)
void gather16(const uint16_t* src, int64_t rows, int64_t heads, int64_t hd, int64_t h, uint16_t* dst, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (rows * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / hd, d = i - r * hd;
        dst[i] = src[(r * heads + h) * hd + d];
    });
}

// D[r] = dO[r] . O[r] over head h (FP32, as the forward summed O) and the head's largest |dO| as bits into *amax
void head_dot(const float* dO, const float* O, int64_t rows, int64_t heads, int64_t hd, int64_t h, float* D,
              uint32_t* amax, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            float v = 0.f, m = 0.f;
            for (auto d = static_cast<int64_t>(it.get_local_linear_id()); d < hd; d += 128) {
                const float g = dO[(r * heads + h) * hd + d];
                v += g * O[(r * heads + h) * hd + d];
                m = sycl::fmax(m, sycl::fabs(g));
            }
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0) {
                D[r] = v;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*amax).fetch_max(
                    sycl::bit_cast<uint32_t>(m));
            }
        });
}

// the head's dO as FP16 at the power of two `amax` asks for (to_f16_range on the device)
void gather_do16(const float* dO, int64_t rows, int64_t heads, int64_t hd, int64_t h, const uint32_t* amax,
                 uint16_t* dst, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (rows * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / hd, d = i - r * hd;
        const float m = sycl::bit_cast<float>(*amax);
        const float g = m > 0.f ? sycl::ldexp(1.f, 13 - sycl::ilogb(m)) : 1.f;
        dst[i] = kernels::f16_from_f32(dO[(r * heads + h) * hd + d] * g);
    });
}

// a tile's dS over dP in place, from S (Q K^T, unscaled) and dP (dO16 V^T, at dO's power of two):
// P = exp(S scale - lse), dS = P (dP / g - D) scale, and its largest |dS| as bits into *amax
void tile_ds(const float* S, float* dP, const float* lse, const float* D, int64_t i0, int64_t n, int64_t keys,
             int64_t heads, int64_t h, float scale, int64_t n_causal, const uint32_t* gbits, uint32_t* amax, void* s) {
    dit_rows::q_of(s).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n * kWG), sycl::range<1>(kWG)), [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group_linear_id());
            const float l = lse[(i0 + r) * heads + h], dd = D[i0 + r];
            const float m = sycl::bit_cast<float>(*gbits);
            const float ig = m > 0.f ? sycl::ldexp(1.f, sycl::ilogb(m) - 13) : 1.f;
            const int64_t lim = i0 + r < n_causal ? i0 + r + 1 : keys;   // a causal row's keys: itself and before
            float mx = 0.f;
            for (auto j = static_cast<int64_t>(it.get_local_linear_id()); j < keys; j += kWG) {
                const float p = j < lim ? sycl::exp(S[r * keys + j] * scale - l) : 0.f;
                const float ds = p * (dP[r * keys + j] * ig - dd) * scale;
                dP[r * keys + j] = ds;
                mx = sycl::fmax(mx, sycl::fabs(ds));
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            if (it.get_local_linear_id() == 0)
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*amax).fetch_max(
                    sycl::bit_cast<uint32_t>(mx));
        });
}

// a tile's FP16 maps for its products: dS (as dS16 [n][keys] and dS16^T [keys][n], at the tile's power of two) and
// P^T [keys][n] (P again from S and lse), the transposes through local memory
void tile_maps(const float* S, const float* dS, const float* lse, int64_t i0, int64_t n, int64_t keys, int64_t heads,
               int64_t h, float scale, int64_t n_causal, const uint32_t* amax, uint16_t* ds16, uint16_t* ds16t,
               uint16_t* pt16, void* s) {
    const size_t gr = (size_t) ((n + kT - 1) / kT), gc = (size_t) ((keys + kT - 1) / kT);
    dit_rows::q_of(s).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 2> tp(sycl::range<2>(kT, kT + 1), cgh), td(sycl::range<2>(kT, kT + 1), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(gr * kT, gc * kT), sycl::range<2>(kT, kT)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t a = (int64_t) it.get_group(0) * kT, b = (int64_t) it.get_group(1) * kT;
                             const int i = (int) it.get_local_id(0), j = (int) it.get_local_id(1);
                             const float m = sycl::bit_cast<float>(*amax);
                             const float sg = m > 0.f ? sycl::ldexp(1.f, 13 - sycl::ilogb(m)) : 1.f;
                             if (a + i < n && b + j < keys) {
                                 const int64_t e = (a + i) * keys + b + j;
                                 const uint16_t d16 = kernels::f16_from_f32(dS[e] * sg);
                                 ds16[e] = d16;
                                 td[i][j] = d16;
                                 const int64_t row = i0 + a + i;
                                 tp[i][j] = row < n_causal && b + j > row
                                                    ? (uint16_t) 0
                                                    : kernels::f16_from_f32(
                                                              sycl::exp(S[e] * scale - lse[row * heads + h]));
                             }
                             sycl::group_barrier(it.get_group());
                             if (b + i < keys && a + j < n) {
                                 ds16t[(b + i) * n + a + j] = td[j][i];
                                 pt16[(b + i) * n + a + j] = tp[j][i];
                             }
                         });
    });
}

// dst[(r0 + r) * heads + h][d] (+)= src[r][d] * k, k from bits (the inverse of to_f16_range's power) or 1
void scatter(const float* src, int64_t r0, int64_t n, int64_t heads, int64_t hd, int64_t h, float* dst, bool acc,
             const uint32_t* bits, void* s) {
    if (dst == nullptr) return;
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) (n * hd)), [=](sycl::id<1> id) {
        const int64_t i = static_cast<int64_t>(id[0]), r = i / hd, d = i - r * hd;
        float k = 1.f;
        if (bits != nullptr) {
            const float m = sycl::bit_cast<float>(*bits);
            k = m > 0.f ? sycl::ldexp(1.f, sycl::ilogb(m) - 13) : 1.f;
        }
        float* o = dst + ((r0 + r) * heads + h) * hd + d;
        *o = acc ? *o + src[i] * k : src[i] * k;
    });
}

// dst[i] += src[i] * k, k the inverse of the power of two bits asks for
void add_unscaled(float* dst, const float* src, int64_t n, const uint32_t* bits, void* s) {
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float m = sycl::bit_cast<float>(*bits);
        dst[id[0]] += src[id[0]] * (m > 0.f ? sycl::ldexp(1.f, sycl::ilogb(m) - 13) : 1.f);
    });
}

}  // namespace

void qi_attention_train(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, float* out32, float* lse, void* stream, int64_t n_causal) {
    kernels::FlashAttnArgs a;
    a.q = q16;
    a.q_ld = heads * head_dim;
    a.k = k16;
    a.v = v16;
    a.kv_ld = heads * head_dim;
    a.n_head = a.n_kv = heads;
    a.head_dim = head_dim;
    a.rows = a.keys = T;
    a.n_causal = n_causal;
    a.first = 0;
    a.row_end = T;
    a.out = out32;
    a.out_ld = heads * head_dim;
    a.out_f32 = true;
    a.lse = lse;
    a.scale = 1.0f / std::sqrt((float) head_dim);
    if (!kernels::flash_attn(a, stream))
        throw std::runtime_error("image engine: no attention kernel takes a training step's shape (head_dim " +
                                 std::to_string(head_dim) + ")");
}

void qi_attention_bwd(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, const float* o32,
                      const float* lse, const float* dO, int64_t T, int64_t heads, int64_t head_dim, float* dQ,
                      float* dK, float* dV, bool accumulate, prefill::Gemm& gemm, void* stream, int64_t n_causal) {
    const int64_t hd = head_dim, tile = std::min<int64_t>(kTile, (T + kT - 1) / kT * kT);
    const float scale = 1.f / std::sqrt((float) hd);
    // a head's Q, K, V, dO (FP16 [T][hd]), K^T [hd][T], D [T], dK and dV sums [T][hd]; a tile's S and dP/dS [tile][T]
    // FP32, dS, dS^T and P^T FP16, Q^T and dO^T [hd][tile], its dQ [tile][hd] and dK's part [T][hd]; the powers of two
    const int64_t n16 = 5 * T * hd + 3 * tile * T + 2 * hd * tile;
    const int64_t n32 = T + 2 * T * hd + 2 * tile * T + tile * hd + T * hd;
    uint8_t* ws = nullptr;
    const size_t bytes = (size_t) n16 * 2 + (size_t) n32 * 4 + 256;
    if (!gpu::alloc_device(reinterpret_cast<void**>(&ws), bytes))
        throw std::runtime_error("image engine: the attention's backward workspace (" + std::to_string(bytes >> 20) +
                                 " MiB)");
    uint32_t* bits = reinterpret_cast<uint32_t*>(ws);   // [0] dO's largest, [1 ..] each tile's dS's largest
    float* f = reinterpret_cast<float*>(ws + 256);
    float* D = f;
    float* dk_sum = D + T;
    float* dv_sum = dk_sum + T * hd;
    float* S = dv_sum + T * hd;
    float* dP = S + tile * T;
    float* dq = dP + tile * T;
    float* dk_part = dq + tile * hd;
    uint16_t* h16 = reinterpret_cast<uint16_t*>(dk_part + T * hd);
    uint16_t *qh = h16, *kh = qh + T * hd, *vh = kh + T * hd, *doh = vh + T * hd, *kt = doh + T * hd;
    uint16_t *ds16 = kt + T * hd, *ds16t = ds16 + tile * T, *pt16 = ds16t + tile * T;
    uint16_t *qt = pt16 + tile * T, *dot = qt + hd * tile;
    const int64_t ntiles = (T + tile - 1) / tile;
    if (2 + ntiles > 64) throw std::runtime_error("image engine: the attention's backward takes 62 tiles at most");
    for (int64_t h = 0; h < heads; ++h) {
        gpu::memset_async(bits, 0, 256, stream);
        gather16(q16, T, heads, hd, h, qh, stream);
        gather16(k16, T, heads, hd, h, kh, stream);
        gather16(v16, T, heads, hd, h, vh, stream);
        head_dot(dO, o32, T, heads, hd, h, D, bits, stream);
        gather_do16(dO, T, heads, hd, h, bits, doh, stream);
        train::transpose_f16(kh, T, hd, kt, stream);
        for (int64_t ti = 0; ti < ntiles; ++ti) {
            const int64_t i0 = ti * tile, n = std::min(tile, T - i0);
            uint32_t* tb = bits + 1 + ti;
            gemm.f16(qh + i0 * hd, kh, S, n, T, hd);           // S = Q K^T
            gemm.f16(doh + i0 * hd, vh, dP, n, T, hd);         // dP = dO V^T (at dO's power of two)
            tile_ds(S, dP, lse, D, i0, n, T, heads, h, scale, n_causal, bits, tb, stream);
            tile_maps(S, dP, lse, i0, n, T, heads, h, scale, n_causal, tb, ds16, ds16t, pt16, stream);
            gemm.f16(ds16, kt, dq, n, hd, T);                  // dQ = dS K
            scatter(dq, i0, n, heads, hd, h, dQ, false, tb, stream);
            train::transpose_f16(qh + i0 * hd, n, hd, qt, stream);
            train::transpose_f16(doh + i0 * hd, n, hd, dot, stream);
            gemm.f16(pt16, dot, dv_sum, T, hd, n, 0, false, ti > 0);   // dV += P^T dO (at dO's power of two)
            gemm.f16(ds16t, qt, dk_part, T, hd, n);            // dK's part: dS^T Q (at the tile's power of two)
            if (ti == 0) gpu::memset_async(dk_sum, 0, (size_t) (T * hd) * 4, stream);
            add_unscaled(dk_sum, dk_part, T * hd, tb, stream);
        }
        scatter(dk_sum, 0, T, heads, hd, h, dK, accumulate, nullptr, stream);
        scatter(dv_sum, 0, T, heads, hd, h, dV, accumulate, bits, stream);
    }
    gpu::stream_sync(stream);   // the workspace is freed below: the kernels above run async
    gpu::free(ws);
}

}  // namespace strata::image
