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

#include "strata/image/aux_attn.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/timed_choice.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/flash_attn.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <map>
#include <tuple>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

// Few query rows (more than a decoder step's): the engine's own work-group-a-row kernel, and no maker's kernel
// measured against it.  A maker's kernel builds a plan a shape (cuDNN ~40 ms a small one on the RTX 4070, oneDNN
// ~6 ms on the B70), paid once a shape a process; a few rows' attention is under a millisecond, so a shape run once
// (Anima's adapter, a prompt's ids over the states) never earns the plan back.  Anima's adapter_s: the maker's choice
// took the B70 0.011->0.017 s and the RTX 4070 0.014->0.053 s (single), 0.019->0.094 s (guided); the own kernel alone
// restores it (0.011 / 0.014, 2026-11-11).  A decoder step (lq <= 4) is excepted: it runs every generated token, so
// its plan amortizes and the maker is the faster (Florence-2's decoder, 64 steps: skipping the maker took its
// decoder_s 0.165->0.210 s on the B70 and 0.167->0.209 s on the RTX 4070).
constexpr int64_t kFewRows = 64;

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

// S > 1: the keys in S parts, part p of each row's into part[((row's (batch, head, i)) * S + p) * (HD + 2)]: its
// running maximum, its sum and its HD weighted values, which split_combine joins
template <int HD> void rows_kernel(const QiSdpa& a, int64_t qb, void* stream, int64_t S = 1, float* part = nullptr) {
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
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (a.batch * heads * nb * S * qb)), sycl::range<1>((size_t) qb)),
                       [=](sycl::nd_item<1> it) {
            const auto gs = static_cast<int64_t>(it.get_group_linear_id());
            const auto lid = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t sp = gs % S, g = gs / S;
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
            const int64_t all = sycl::max(mk.keys(b, last), mk.keys(b, blk * qb));
            // this part's keys: whole tiles of the block's
            const int64_t per = (all + S * KT - 1) / (S * KT) * KT, k0 = sp * per;
            const int64_t nk = sycl::min(all, k0 + per);
            const int64_t mine = live ? mk.keys(b, i) : 0;
            float m = -INFINITY, l = 0.f;
            for (int64_t j0 = k0; j0 < nk; j0 += KT) {
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
            if (part != nullptr) {
                float* pr = part + (((b * heads + hh) * lq + i) * S + sp) * (HD + 2);
                pr[0] = m;
                pr[1] = l;
                for (int d = 0; d < HD; ++d) pr[2 + d] = acc[d];
                return;
            }
            const float r = l > 0.f ? 1.f / l : 0.f;   // a row that reads no key: zero
            for (int64_t d = 0; d < hd; ++d) O[b * obs + i * orw + hh * hd + d] = f16_from_f32(acc[d] * r);
        });
    });
    finish(stream, e, "qi_sdpa_ex (rows)");
}

// rows_kernel's S parts of each row joined: the parts' sums rescaled to the largest maximum
void split_combine(const QiSdpa& a, int64_t S, int64_t HDP, const float* part, void* stream) {
    const int64_t hd = a.hd, heads = a.heads, lq = a.lq, orw = a.o_row, obs = a.o_batch;
    uint16_t* O = a.out;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) (a.batch * heads * lq), (size_t) hd),
                                                  [=](sycl::id<2> id) {
        const auto r = static_cast<int64_t>(id[0]), d = static_cast<int64_t>(id[1]);
        const int64_t i = r % lq, bh = r / lq, hh = bh % heads, b = bh / heads;
        const float* pr = part + r * S * (HDP + 2);
        float m = -INFINITY;
        for (int64_t p = 0; p < S; ++p) m = sycl::fmax(m, pr[p * (HDP + 2)]);
        float l = 0.f, acc = 0.f;
        for (int64_t p = 0; p < S; ++p) {
            const float* q = pr + p * (HDP + 2);
            if (q[0] == -INFINITY) continue;
            const float c = sycl::exp(q[0] - m);
            l += q[1] * c;
            acc += q[2 + d] * c;
        }
        O[b * obs + i * orw + hh * hd + d] = f16_from_f32(l > 0.f ? acc / l : 0.f);
    });
    finish(stream, e, "qi_sdpa_ex (combine)");
}

// rows_kernel over S parts of the keys, then joined
// A buffer of `need` floats kept per queue, grown to the largest (null when it does not fit).  The keys' parts and
// the matrix-engine paths' scratch share it: a call uses one or the other, never both at once.  The queue's own
// belonging (Runtime::queue_item), freed when its stream is destroyed - keyed by the bare queue address it would
// outlive the queue and meet the next one there.
struct ScratchBuf {
    explicit ScratchBuf(sycl::queue& q) : q(q) {}
    ~ScratchBuf() {
        if (p != nullptr) {
            sycl::free(p, q);
            core::graph_epoch_bump();   // a recorded step may hold the old one
        }
    }
    sycl::queue q;
    std::mutex mu;
    float* p = nullptr;
    size_t n = 0;
};
float* queue_scratch(size_t need, void* stream) {
    sycl::queue& q = queue_for(stream);
    ScratchBuf& c = core::Runtime::of_queue(q).queue_item<ScratchBuf>(q);
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.n < need) {
        q.wait();
        if (c.p != nullptr) {
            sycl::free(c.p, q);
            core::graph_epoch_bump();   // a recorded step may hold the old one
        }
        c.p = sycl::malloc_device<float>(need, q);
        c.n = c.p != nullptr ? need : 0;
    }
    return c.p;
}

float* parts(const QiSdpa& a, int64_t S, int64_t stride, void* stream) {
    float* p = queue_scratch((size_t) (a.batch * a.heads * a.lq * S * (stride + 2)), stream);
    if (p == nullptr) throw std::runtime_error("qi_sdpa_ex: no GPU memory for the keys' parts");
    return p;
}

// The matrix engines' Gemm for the biased attention, one a queue (init once: f16_batch needs no dequant scratch):
// the queue's own belonging, as the scratch above
struct QueueGemm {
    explicit QueueGemm(sycl::queue& q) : q(q) {}
    sycl::queue q;
    std::mutex mu;
    std::unique_ptr<prefill::Gemm> g;
};
prefill::Gemm& matrix_gemm(void* stream) {
    sycl::queue& q = queue_for(stream);
    QueueGemm& mg = core::Runtime::of_queue(q).queue_item<QueueGemm>(q);
    std::lock_guard<std::mutex> lock(mg.mu);
    if (mg.g == nullptr) {
        mg.g = std::make_unique<prefill::Gemm>();
        std::string err;
        if (!mg.g->init(stream, 0, err)) throw std::runtime_error("qi_sdpa_ex: " + err);
    }
    return *mg.g;
}

// The biased large shapes on the matrix engines: the shared Gemm attention (aux_attn_matrix), the heads split apart
// and the scores held in the queue's scratch.  False (the caller keeps the FMA kernel) for the small heads (hd < 32)
// or few query rows (lq < 256, BiRefNet's 144-row windows measured slower on the B70) or where the scores would not
// fit: the FMA kernel holds none, this one holds [batch heads][lq][lk].  Each batch its own keys (kv_div 1), no
// key_len or causal (a row's key count the scores' softmax here does not mask).
bool matrix_biased(const QiSdpa& a, void* stream) {
    const int64_t B = a.batch * a.heads, lq = a.lq, lk = a.lk, hd = a.hd;
    // the small heads and the few query rows keep the FMA kernel: the heads' split/merge and the scores this form
    // holds outweigh the matrix product there.  BiRefNet's Swin (hd 32, 144-row windows) measured slower on the B70
    // (backbone 0.124 -> 0.127 s over 5 warms, 2026-11-11); the RTX 4070 was faster (0.200 -> 0.186 s) but the form
    // is chosen by shape, not by maker, so the small windows stay FMA on every GPU.
    if (hd < 32 || lq < 256) return false;
    const int64_t h16 = (B * lq * hd + 1) / 2 + (B * lk * hd + 1) / 2 + (B * hd * lk + 1) / 2;
    const int64_t need = h16 + B * lq * lk + (B * lq * lk + 1) / 2 + B * lq * hd;   // qh,kh,vt; sc; p16; o (floats)
    if (need > (int64_t) (128 << 20) / 4) return false;   // a 128 MiB scratch bound
    float* buf = queue_scratch((size_t) need, stream);
    if (buf == nullptr) return false;
    auto* qh = reinterpret_cast<uint16_t*>(buf);
    auto* kh = qh + B * lq * hd;
    auto* vt = kh + B * lk * hd;
    auto* sc = reinterpret_cast<float*>(vt + B * hd * lk);
    auto* p16 = reinterpret_cast<uint16_t*>(sc + B * lq * lk);
    auto* o = reinterpret_cast<float*>(p16 + B * lq * lk);
    aux_heads_batch16(a.q, a.batch, lq, a.heads, hd, a.q_row, a.q_batch, false, qh, stream);
    aux_heads_batch16(a.k, a.batch, lk, a.heads, hd, a.k_row, a.k_batch, false, kh, stream);
    aux_heads_batch16(a.v, a.batch, lk, a.heads, hd, a.v_row, a.v_batch, true, vt, stream);
    aux_attn_matrix(matrix_gemm(stream), qh, kh, vt, a.batch, a.heads, lq, lk, hd, lq, hd, false, a.scale, a.bias,
                    a.mask, std::max<int64_t>(a.mask_n, 1), nullptr, 0, 0, sc, p16, o, stream);
    aux_heads_merge_batch16(o, a.batch, lq, a.heads, hd, a.o_row, a.o_batch, a.out, stream);
    return true;
}

// The unmasked large shape of one batch on the matrix engines: the shared flash_attn, the head dim padded to 128
// (the channels past hd zero, so the scores are the head's own; the scale stays 1/sqrt(hd)).  False where flash_attn
// takes it not (the caller keeps the FMA kernel).  The maker's library still competes through qi_sdpa_ex's choice.
bool flash_padded(const QiSdpa& a, void* stream) {
    const int64_t lq = a.lq, lk = a.lk, heads = a.heads, hd = a.hd;
    if (a.o_row != heads * hd) return false;   // the unpad writes contiguous rows
    const int64_t Tp = qi_attn_rows(lq), Kp = qi_attn_rows(lk), C = heads * 128;
    const int64_t need = (Tp * C + Kp * C + Kp * C + Tp * C + 1) / 2;   // q,k,v,out FP16 (floats)
    float* buf = queue_scratch((size_t) need, stream);
    if (buf == nullptr) return false;
    auto* qp = reinterpret_cast<uint16_t*>(buf);
    auto* kp = qp + Tp * C;
    auto* vp = kp + Kp * C;
    auto* op = vp + Kp * C;
    gpu::memset_async(qp, 0, (size_t) need * 4, stream);   // the rows past lq / lk zero (whole chunks read)
    qi_pad_heads16(a.q, lq, heads, hd, a.q_row, qp, stream);
    qi_pad_heads16(a.k, lk, heads, hd, a.k_row, kp, stream);
    qi_pad_heads16(a.v, lk, heads, hd, a.v_row, vp, stream);
    kernels::FlashAttnArgs fa;
    fa.q = qp;
    fa.q_ld = C;
    fa.k = kp;
    fa.v = vp;
    fa.kv_ld = C;
    fa.n_head = fa.n_kv = heads;
    fa.head_dim = 128;
    fa.rows = lq;
    fa.keys = lk;
    fa.first = 0;
    fa.row_end = lq;
    fa.out = op;
    fa.out_ld = C;
    fa.scale = a.scale;
    if (!kernels::flash_attn(fa, stream)) return false;
    qi_unpad_heads16(op, lq, heads, hd, a.out, stream);
    return true;
}

template <int HD> void rows_split(const QiSdpa& a, int64_t qb, int64_t S, void* stream) {
    float* part = parts(a, S, HD, stream);
    rows_kernel<HD>(a, qb, stream, S, part);
    split_combine(a, S, HD, part, stream);
}

// S > 1: the keys in S parts of ceil(lk / S) (lk, not a row's key_len: a recorded step's parts stay as recorded),
// part p of each row's into part[(row * S + p) * (hd + 2)] as rows_kernel's
void few_kernel(const QiSdpa& a, int64_t L, void* stream, int64_t S = 1, float* part = nullptr) {
    const int64_t hd = a.hd, heads = a.heads, lq = a.lq;
    const Masks mk{a.key_len, a.bias, a.mask, a.lq, a.lk, std::max<int64_t>(a.mask_n, 1), a.causal_off, a.causal};
    const uint16_t *Q = a.q, *K = a.k, *V = a.v;
    uint16_t* O = a.out;
    const int64_t qr = a.q_row, qbs = a.q_batch, kr = a.k_row, kbs = a.k_batch, vr = a.v_row, vbs = a.v_batch,
                  orw = a.o_row, obs = a.o_batch, div = std::max<int64_t>(a.kv_div, 1);
    const int64_t per = (a.lk + S - 1) / S;
    const float scale = a.scale;
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs((size_t) hd, h), ps((size_t) L, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (a.batch * heads * lq * S * L)),
                                         sycl::range<1>((size_t) L)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lid = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t sp = g % S, row = g / S;
            const int64_t i = row % lq, bh = row / lq, hh = bh % heads, b = bh / heads, kb = b / div;
            for (int64_t d = lid; d < hd; d += L) qs[d] = f32_from_f16(Q[b * qbs + i * qr + hh * hd + d]) * scale;
            sycl::group_barrier(it.get_group());
            const int64_t nk = sycl::min(mk.keys(b, i), (sp + 1) * per);
            float m = -INFINITY, l = 0.f, acc = 0.f;
            for (int64_t j0 = sp * per; j0 < nk; j0 += L) {
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
            if (part != nullptr) {
                float* pr = part + (row * S + sp) * (hd + 2);
                if (lid == 0) {
                    pr[0] = m;
                    pr[1] = l;
                }
                if (lid < hd) pr[2 + lid] = acc;
                return;
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
    if (a.q_row != row || a.k_row != row || a.v_row != row || a.o_row != row) return false;
    if (std::abs(a.scale * std::sqrt((float) a.hd) - 1.f) > 1e-6f) return false;
    // every batch reads the same keys and the rows lie one after another: one attention of batch x lq rows
    const bool shared = a.batch == 1 || (a.kv_div >= a.batch && a.q_batch == a.lq * row && a.o_batch == a.lq * row);
    if (shared) return qi_sdpa(a.q, a.k, a.v, 0, a.batch * a.lq, a.lk, a.heads, a.hd, a.out, stream, a.causal);
    // each batch its own keys (the windows of a ViT): one call where the batches lie one after another in each tensor,
    // else one call each, the first telling whether the library takes it
    if (a.kv_div != 1) return false;
    if (!a.causal && a.q_batch == a.lq * row && a.o_batch == a.lq * row && a.k_batch == a.lk * row &&
        a.v_batch == a.lk * row)
        return qi_sdpa(a.q, a.k, a.v, 0, a.lq, a.lk, a.heads, a.hd, a.out, stream, false, a.batch);
    for (int64_t b = 0; b < a.batch; ++b)
        if (!qi_sdpa(a.q + b * a.q_batch, a.k + b * a.k_batch, a.v + b * a.v_batch, 0, a.lq, a.lk, a.heads, a.hd,
                     a.out + b * a.o_batch, stream)) {
            if (b == 0) return false;
            throw std::runtime_error("qi_sdpa_ex: the library took a batch's attention and not the next one's");
        }
    return true;
}

void qi_sdpa_ex_own(const QiSdpa& a, void* stream) {
    if (a.batch <= 0 || a.lq <= 0 || a.heads <= 0) return;
    if (a.hd <= 0 || a.hd > 128) throw std::invalid_argument("qi_sdpa_ex: head size " + std::to_string(a.hd));
    if (a.lk <= 0) throw std::invalid_argument("qi_sdpa_ex: no keys");
    const sycl::device dev = queue_for(stream).get_device();
    const auto wg = (int64_t) dev.get_info<sycl::info::device::max_work_group_size>();
    const auto units = (int64_t) dev.get_info<sycl::info::device::max_compute_units>();
    // a step's few rows: a work-group each, its items over the keys (more of the GPU than a block of rows would use),
    // and the keys in parts, a work-group's L at least, while the groups are fewer than 16 a compute unit (Florence-2
    // large's decoder, 3 beams of 16 heads over up to 1,025 keys: the RTX 4070 4.87 s in one part, 3.98 at 4 groups a
    // unit, 3.46 at 16; the B70 6.45 s, 3.97 from 2 on, 2026-11-07).  A work-item a row (below) spills its 2 hd
    // registers at hd 64-128: Anima's 31 rows over 31 keys took 0.8 ms on the RX 9060 XT that way, so the few rows
    // take this work-group form (kFewRows).
    if (a.lq <= kFewRows) {
        const int64_t L = std::min<int64_t>(wg, 128);
        if (a.hd <= L) {
            const int64_t rows = a.batch * a.heads * a.lq;
            const int64_t S = std::min((16 * units + rows - 1) / rows, (a.lk + L - 1) / L);
            if (S > 1) {
                float* part = parts(a, S, a.hd, stream);
                few_kernel(a, L, stream, S, part);
                split_combine(a, S, a.hd, part, stream);
            } else {
                few_kernel(a, L, stream);
            }
            return;
        }
    }
    // the large shapes on a GPU with matrix engines: the biased through the shared Gemm attention, the unmasked of
    // one batch through the shared flash_attn (the head dim padded to 128).  Else (no matrix engines, or a shape these
    // do not take) the FMA kernel below.  The maker's library, where it takes the shape, is still chosen by
    // qi_sdpa_ex's measuring against this own path.
    if (kernels::flash_attn_path(&dev) >= 0 && a.key_len == nullptr && !a.causal) {
        if (a.bias != nullptr || a.mask != nullptr) {
            if (a.kv_div == 1 && matrix_biased(a, stream)) return;
        } else if (a.batch == 1 && flash_padded(a, stream)) {
            return;
        }
    }
    const int64_t qb = std::min<int64_t>(wg, 64);
    // fewer blocks of rows than the GPU has compute units, and many keys: the keys in parts, so the parts fill it (SAM
    // 3's 201 queries over 5,184 patches: 32 blocks, 4.1 ms a layer on the B70 in one part; its detection 0.042 s,
    // 0.022 in parts, 2026-11-07); a part at least 4 tiles of keys
    const int64_t blocks = a.batch * a.heads * ((a.lq + qb - 1) / qb);
    const int64_t S = blocks >= units ? 1 : std::min((units + blocks - 1) / blocks, std::max<int64_t>(1, a.lk / 128));
    if (S > 1) {
        if (a.hd <= 32) rows_split<32>(a, qb, S, stream);
        else if (a.hd <= 64) rows_split<64>(a, qb, S, stream);
        else rows_split<128>(a, qb, S, stream);
        return;
    }
    if (a.hd <= 32) rows_kernel<32>(a, qb, stream);
    else if (a.hd <= 64) rows_kernel<64>(a, qb, stream);
    else rows_kernel<128>(a, qb, stream);
}

void qi_sdpa_ex(const QiSdpa& a, void* stream) {
    if (a.batch <= 0 || a.lq <= 0 || a.heads <= 0) return;
    if (a.hd <= 0 || a.hd > 128 || a.lk <= 0 || a.key_len || a.bias || a.mask ||
        (a.lq > 4 && a.lq <= kFewRows)) {
        qi_sdpa_ex_own(a, stream);   // the few rows past a decoder step: the own kernel, no maker's plan (kFewRows)
        return;
    }
    sycl::queue& q = queue_for(stream);
    const auto dev = q.get_device();
    using Shape = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, float>;
    const Shape key{a.batch, a.lq, a.lk, a.heads, a.hd, a.q_row, a.k_row, a.v_row, a.o_row,
                    a.q_batch, a.k_batch, a.v_batch, a.o_batch, a.causal, a.kv_div, a.causal_off, a.scale};
    static core::TimedCache<sycl::device, Shape, bool> choices;
    const auto cached = choices.get(dev, key);
    int maker = cached ? (int) *cached : -1;
    namespace sx = sycl::ext::oneapi::experimental;
    if (maker < 0 && q.ext_oneapi_get_state() == sx::queue_state::recording) {
        if (!qi_sdpa_ex_library(a, stream)) qi_sdpa_ex_own(a, stream);
        return;
    }
    if (maker < 0) {
        const auto measured = core::timed_choice({{[&] { return qi_sdpa_ex_library(a, stream); }},
                                                  {[&] { qi_sdpa_ex_own(a, stream); return true; }}},
                                                 [&] { q.wait_and_throw(); });
        maker = measured.pick == 0;
        maker = choices.put(dev, key, maker != 0);
    }
    if (!maker || !qi_sdpa_ex_library(a, stream)) qi_sdpa_ex_own(a, stream);
}

}  // namespace strata::image
