// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa_ex.cpp - see include/strata/image/qi_sdpa_ex.hpp: attention of any shape the auxiliary models
// ask for, through a GPU maker's kernel where qi_sdpa's contract takes the shape (no mask, no bias; one call a batch
// where the batches' keys differ), else the engine's own:
//   * rows: a work-item a query row and a work-group a block of rows of one (batch, head), the keys in tiles through
//     local memory, the softmax online (the windows' and the encoders' many rows);
//   * few: a work-group a query row, its items over the keys and then over the head's channels (a decoder's step).

#include "strata/image/qi_sdpa_ex.hpp"

#include "strata/image/qi_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

// the keys query row i of batch b reads (key_len, causal) and the additive terms of key j
struct Masks {
    const int32_t* key_len;
    const float *bias, *mask;
    int64_t lq, lk, mask_n, causal_off;
    bool causal;
    int64_t keys(int64_t b, int64_t i) const {
        int64_t n = key_len != nullptr ? (int64_t) key_len[b] : lk;
        if (causal) n = sycl::min(n, i + causal_off + 1);
        return n > 0 ? n : 0;
    }
    float add(int64_t b, int64_t h, int64_t i, int64_t j) const {
        float a = 0.f;
        if (bias != nullptr) a += bias[(h * lq + i) * lk + j];
        if (mask != nullptr) a += mask[((b % mask_n) * lq + i) * lk + j];
        return a;
    }
};

template <int HD> void rows_kernel(const QiSdpa& a, int64_t qb, void* stream) {
    constexpr int64_t KT = 32;
    const int64_t hd = a.hd, nb = (a.lq + qb - 1) / qb, heads = a.heads;
    const Masks mk{a.key_len, a.bias, a.mask, a.lq, a.lk, std::max<int64_t>(a.mask_n, 1), a.causal_off, a.causal};
    const uint16_t *Q = a.q, *K = a.k, *V = a.v;
    uint16_t* O = a.out;
    const int64_t qr = a.q_row, qbs = a.q_batch, kr = a.k_row, kbs = a.k_batch, vr = a.v_row, vbs = a.v_batch,
                  orw = a.o_row, obs = a.o_batch, div = std::max<int64_t>(a.kv_div, 1), lq = a.lq;
    const float scale = a.scale;
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ks(KT * HD, h), vs(KT * HD, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (a.batch * heads * nb * qb)), sycl::range<1>((size_t) qb)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lid = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t blk = g % nb, bh = g / nb, hh = bh % heads, b = bh / heads, kb = b / div;
            const int64_t i = blk * qb + lid;
            const bool live = i < lq;
            float qv[HD], acc[HD];
            for (int d = 0; d < HD; ++d) {
                qv[d] = live && d < hd ? f32_from_f16(Q[b * qbs + i * qr + hh * hd + d]) * scale : 0.f;
                acc[d] = 0.f;
            }
            // the block's keys: the most any of its rows reads
            const int64_t last = sycl::min(lq - 1, blk * qb + qb - 1);
            const int64_t nk = sycl::max(mk.keys(b, last), mk.keys(b, blk * qb));
            const int64_t mine = live ? mk.keys(b, i) : 0;
            float m = -INFINITY, l = 0.f;
            for (int64_t j0 = 0; j0 < nk; j0 += KT) {
                const int64_t n = sycl::min(KT, nk - j0);
                for (int64_t x = lid; x < KT * HD; x += qb) {
                    const int64_t jj = x / HD, d = x - jj * HD;
                    const bool in = jj < n && d < hd;
                    ks[x] = in ? f32_from_f16(K[kb * kbs + (j0 + jj) * kr + hh * hd + d]) : 0.f;
                    vs[x] = in ? f32_from_f16(V[kb * vbs + (j0 + jj) * vr + hh * hd + d]) : 0.f;
                }
                sycl::group_barrier(it.get_group());
                for (int64_t jj = 0; jj < n; ++jj) {
                    const int64_t j = j0 + jj;
                    if (j >= mine) break;
                    float s = 0.f;
                    for (int d = 0; d < HD; ++d) s += qv[d] * ks[jj * HD + d];
                    s += mk.add(b, hh, i, j);
                    if (s > m) {
                        const float c = sycl::exp(m - s);
                        l *= c;
                        for (int d = 0; d < HD; ++d) acc[d] *= c;
                        m = s;
                    }
                    const float p = sycl::exp(s - m);
                    l += p;
                    for (int d = 0; d < HD; ++d) acc[d] += p * vs[jj * HD + d];
                }
                sycl::group_barrier(it.get_group());
            }
            if (!live) return;
            const float r = l > 0.f ? 1.f / l : 0.f;   // a row that reads no key: zero
            for (int64_t d = 0; d < hd; ++d) O[b * obs + i * orw + hh * hd + d] = f16_from_f32(acc[d] * r);
        });
    });
    finish(stream, e, "qi_sdpa_ex (rows)");
}

void few_kernel(const QiSdpa& a, int64_t L, void* stream) {
    const int64_t hd = a.hd, heads = a.heads, lq = a.lq;
    const Masks mk{a.key_len, a.bias, a.mask, a.lq, a.lk, std::max<int64_t>(a.mask_n, 1), a.causal_off, a.causal};
    const uint16_t *Q = a.q, *K = a.k, *V = a.v;
    uint16_t* O = a.out;
    const int64_t qr = a.q_row, qbs = a.q_batch, kr = a.k_row, kbs = a.k_batch, vr = a.v_row, vbs = a.v_batch,
                  orw = a.o_row, obs = a.o_batch, div = std::max<int64_t>(a.kv_div, 1);
    const float scale = a.scale;
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs((size_t) hd, h), ps((size_t) L, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (a.batch * heads * lq * L)), sycl::range<1>((size_t) L)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lid = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t i = g % lq, bh = g / lq, hh = bh % heads, b = bh / heads, kb = b / div;
            for (int64_t d = lid; d < hd; d += L) qs[d] = f32_from_f16(Q[b * qbs + i * qr + hh * hd + d]) * scale;
            sycl::group_barrier(it.get_group());
            const int64_t nk = mk.keys(b, i);
            float m = -INFINITY, l = 0.f, acc = 0.f;
            for (int64_t j0 = 0; j0 < nk; j0 += L) {
                const int64_t j = j0 + lid;
                float s = -INFINITY;
                if (j < nk) {
                    const uint16_t* kp = K + kb * kbs + j * kr + hh * hd;
                    s = 0.f;
                    for (int64_t d = 0; d < hd; ++d) s += qs[d] * f32_from_f16(kp[d]);
                    s += mk.add(b, hh, i, j);
                }
                const float nm = sycl::fmax(m, sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>()));
                const float p = j < nk ? sycl::exp(s - nm) : 0.f;
                const float c = m == -INFINITY ? 0.f : sycl::exp(m - nm);
                l = l * c + sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
                ps[lid] = p;
                sycl::group_barrier(it.get_group());
                acc *= c;
                const int64_t n = sycl::min(L, nk - j0);
                if (lid < hd)
                    for (int64_t jj = 0; jj < n; ++jj) acc += ps[jj] * f32_from_f16(V[kb * vbs + (j0 + jj) * vr + hh * hd + lid]);
                sycl::group_barrier(it.get_group());
                m = nm;
            }
            if (lid < hd) O[b * obs + i * orw + hh * hd + lid] = f16_from_f32(l > 0.f ? acc / l : 0.f);
        });
    });
    finish(stream, e, "qi_sdpa_ex (few)");
}

}  // namespace

bool qi_sdpa_ex_library(const QiSdpa& a, void* stream) {
    const int64_t row = a.heads * a.hd;
    if (a.key_len != nullptr || a.bias != nullptr || a.mask != nullptr) return false;
    // causal rows the libraries take: each row its own position among as many keys (a text tower's)
    if (a.causal && (a.causal_off != 0 || a.lq != a.lk || a.batch != 1 || !qi_sdpa_causal(stream))) return false;
    // a few keys (a prompt's words) go to the engine's own kernel, which costs little there: oneDNN's fused SDPA gave
    // NaN now and then over SAM 3's 3 words' keys inside the model on the B70 (2026-11-07; not in a test of the call
    // alone, nor with 32 keys or more; cuDNN's on the RTX 4070 gave none)
    if (a.lk < 32) return false;
    if (a.q_row != row || a.k_row != row || a.v_row != row || a.o_row != row) return false;
    if (std::abs(a.scale * std::sqrt((float) a.hd) - 1.f) > 1e-6f) return false;
    // every batch reads the same keys and the rows lie one after another: one attention of batch x lq rows
    const bool shared = a.batch == 1 || (a.kv_div >= a.batch && a.q_batch == a.lq * row && a.o_batch == a.lq * row);
    if (shared) return qi_sdpa(a.q, a.k, a.v, 0, a.batch * a.lq, a.lk, a.heads, a.hd, a.out, stream, a.causal);
    // each batch its own keys (the windows of a ViT): one call each, the first telling whether the library takes it
    if (a.kv_div != 1) return false;
    for (int64_t b = 0; b < a.batch; ++b)
        if (!qi_sdpa(a.q + b * a.q_batch, a.k + b * a.k_batch, a.v + b * a.v_batch, 0, a.lq, a.lk, a.heads, a.hd,
                     a.out + b * a.o_batch, stream)) {
            if (b == 0) return false;
            throw std::runtime_error("qi_sdpa_ex: the library took a batch's attention and not the next one's");
        }
    return true;
}

void qi_sdpa_ex(const QiSdpa& a, void* stream) {
    if (a.batch <= 0 || a.lq <= 0 || a.heads <= 0) return;
    if (a.hd <= 0 || a.hd > 128) throw std::invalid_argument("qi_sdpa_ex: head size " + std::to_string(a.hd));
    if (a.lk <= 0) throw std::invalid_argument("qi_sdpa_ex: no keys");
    if (qi_sdpa_ex_library(a, stream)) return;
    const sycl::device dev = queue_for(stream).get_device();
    const auto wg = (int64_t) dev.get_info<sycl::info::device::max_work_group_size>();
    // a step's few rows: a work-group each, its items over the keys (more of the GPU than a block of rows would use)
    if (a.lq <= 4) {
        const int64_t L = std::min<int64_t>(wg, 128);
        if (a.hd <= L) {
            few_kernel(a, L, stream);
            return;
        }
    }
    const int64_t qb = std::min<int64_t>(wg, 64);
    if (a.hd <= 32) rows_kernel<32>(a, qb, stream);
    else if (a.hd <= 64) rows_kernel<64>(a, qb, stream);
    else rows_kernel<128>(a, qb, stream);
}

}  // namespace strata::image
