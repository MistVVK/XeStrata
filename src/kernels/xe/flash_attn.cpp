// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/flash_attn.cpp - see include/strata/kernels/flash_attn.hpp.
//
// Four forms behind the one entry, by the work and the GPU: the row form (Kernel, below: Intel's XMX and NVIDIA's
// tensor cores; many rows share the keys), its RDNA form (WmmaKernel: RDNA3's and RDNA4's WMMA, the layout of their
// accumulators known), its FMA form (FmaKernel: no matrix engines), and the selection form (namespace sel:
// Qwen3.8-Flash-Next's QSA, each query its own cells).
//
// The row form.  A sub-group takes TM rows (query, head) and walks their keys in chunks of CH:
//   * S = Q . K^T on the matrix engines (Q's tiles in registers), stored to local memory row-major: the
//     accumulators' layout is the vendor's, local memory's is ours;
//   * the online softmax of each row there, L lanes a row, writing the chunk's probabilities P as FP16; a row's
//     reference (its largest score) moves only when a score passes it by more than TAU, as FlashAttention-3 does, so
//     most chunks leave the output alone, and when one moves the output's tiles go through local memory to be scaled;
//   * O += P . V on the matrix engines, the output kept in the accumulators.
// Two ways to the keys.  In place (F < 0): a sub-group takes TM queries of one head on its own, K's and V's tiles
// read from global memory (the caches share them between the sub-groups of a work-group, which take neighbouring
// rows of the same head): a chunk's keys are CH rows at one stride, the dense layout's or a page's.  Gathered (F a
// PoolFormat): a work-group takes rows (query, head) of one KV head, its queries' G heads together, and copies each
// chunk's K and V into its local memory as FP16 from the pools' cells (any page size, INT8 or Q4_0 codes decoded on
// the way), every sub-group reading them there.
#include "strata/kernels/flash_attn.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "attn_parts.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "mma_gemm.hpp"
#include "amd_wmma.hpp"
#include "nv_mma.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_flash_attn
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/env.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace strata::kernels {

namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using namespace xe::attn;

template<int P, int D, bool F32, int F>   // P: the tiles; D: head_dim; F32: FP32 output, else FP16; F: as above
struct Kernel {
    using Ops = typename Path<P>::Ops;
    static constexpr bool GATHER = F >= 0;
    // keys a chunk: 64; gathered heads of 256, 32 (K's and V's chunk in local memory: 33 KiB)
    static constexpr int CH = GATHER && D > 128 ? 32 : 64;
    static constexpr int TM = Ops::TM, TN = Ops::TN, TK = Ops::TK, SG = Ops::SG;
    static constexpr int L = SG / TM;              // lanes a row in the softmax
    static constexpr int CW = CH / L;              // a lane's scores: contiguous columns of its row
    static constexpr int NT = D / TN;              // the output's column tiles
    static constexpr int SLD = CH + 4, PLD = CH + 8;   // S's and P's row strides in local memory: rows apart in banks
    static constexpr int KLD = D + 8;              // a gathered key's row (halves; 16-byte aligned rows)
    // S [TM][SLD] FP32, which also holds Q as FP16 at the start and the output when it is scaled: in place all of it
    // [TM][D] (a tile at a time, the RTX 4070's image attention ran 13.4 ms against 10.2, 32 heads of 4096 rows),
    // gathered a tile [TM][TN] at a time (the room goes to the work-group's K and V)
    static constexpr bool WHOLE = !GATHER;
    static constexpr int SREG = std::max(TM * SLD, WHOLE ? TM * D : TM * D / 2);
    static_assert(SG % TM == 0 && CW % 8 == 0, "the softmax's rows split evenly over the lanes, 8 columns at a time");
    // local memory a sub-group takes: S, P, a row's m and l; a work-group's gathered K and V
    static constexpr size_t SG_BYTES = (size_t) SREG * 4 + (size_t) TM * PLD * 2 + (size_t) 2 * TM * 4;
    static constexpr size_t WG_BYTES = GATHER ? (size_t) 2 * CH * KLD * 2 : 0;

    FlashAttnArgs a;
    int64_t row0;   // the first row: a query (in place, a multiple of TM), or a (query, head) row (gathered)
    int64_t row_hi; // the rows past it are not asked (gathered: (query, head) rows)
    sycl::local_accessor<sycl::half, 1> ps;      // per sub-group P [TM][PLD]
    sycl::local_accessor<float, 1> st;           // per sub-group S [TM][SLD]
    sycl::local_accessor<float, 1> rows;         // per sub-group m and l [2][TM]
    sycl::local_accessor<sycl::half, 1> kv;      // gathered: K [CH][KLD], V [CH][KLD]

    // Xe2's instance takes the large register file: the output's accumulator tiles and Q's operand tiles
    auto get(syclex::properties_tag) const {
        if constexpr (P == 0)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    // in place: the chunk of keys from j on of KV head g, its first row (the rows' stride `ld`)
    const uint16_t* chunk(const uint16_t* base, int64_t j, int64_t g) const {
        if (a.page_table == nullptr) return base + j * a.kv_ld + g * D;
        const int64_t page = a.page_table[j / a.page_size];
        return base + ((page * a.n_kv + g) * a.page_size + j % a.page_size) * D;
    }
    // the keys row `t` sees (0 past the rows)
    int64_t lim_of(int64_t t) const {
        return t >= a.rows ? 0 : t < a.n_causal ? (a.key_lim ? (int64_t) a.key_lim[t] : a.pos0 + t + 1) : a.keys;
    }
    // the first key row `t` sees: a sliding window's (pos0 + t + 1 - window) for a causal row, else 0
    int64_t from_of(int64_t t) const {
        const int64_t f = a.window > 0 && t < a.n_causal ? a.pos0 + t + 1 - a.window : 0;
        return f > 0 ? f : 0;
    }

    // gathered: cells c0 .. c0 + CH - 1 of KV head g into the work-group's K and V as FP16 (zero from `end` on)
    void gather(sycl::nd_item<2> it, int64_t c0, int64_t end, int64_t g, sycl::half* K, sycl::half* V) const {
        const QsaAttnPools& p = a.pools;
        const int ps_ = (int) a.page_size;
        auto cell_row = [&](int64_t c) { return ((int64_t) p.page_table[c / ps_] * a.n_kv + g) * ps_ + c % ps_; };
        const int t = (int) it.get_local_linear_id(), nt = (int) it.get_local_range(1);
        if constexpr (F == kFmtF16 || F == kFmtI8) {
            for (int i = t; i < CH * D / 8; i += nt) {   // 8 values a work-item
                const int j = i / (D / 8), dd = (i - j * (D / 8)) * 8;
                const int64_t c = c0 + j;
                h8 kk(0), vv(0);
                if (c < end) {
                    const int64_t row = cell_row(c);
                    if constexpr (F == kFmtF16) {
                        kk = *reinterpret_cast<const h8*>(p.k_pool + row * D + dd);
                        vv = *reinterpret_cast<const h8*>(p.v_pool + row * D + dd);
                    } else {
                        kk = i8_row8<D>(p.k_q, p.k_scale, row, dd);
                        vv = i8_row8<D>(p.v_q, p.v_scale, row, dd);
                    }
                }
                *reinterpret_cast<h8*>(K + (size_t) j * KLD + dd) = kk;
                *reinterpret_cast<h8*>(V + (size_t) j * KLD + dd) = vv;
            }
        } else {   // Q4_0's planes a block of 32 values a work-item; K8V4's INT8 K 8 at a time
            constexpr int NB = D / QK4_0;
            for (int i = t; i < CH * NB; i += nt) {
                const int j = i / NB, b = i - j * NB;
                const int64_t c = c0 + j;
                sycl::half* dk = K + (size_t) j * KLD + (size_t) b * QK4_0;
                sycl::half* dv = V + (size_t) j * KLD + (size_t) b * QK4_0;
                if (c < end) {
                    const int64_t row = cell_row(c);
                    q4_block32<D>(p.v_q4, row, b, dv);
                    if constexpr (F == kFmtQ4) {
                        q4_block32<D>(p.k_q4, row, b, dk);
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            *reinterpret_cast<h8*>(dk + (size_t) k * 8) = i8_row8<D>(p.k_q, p.k_scale, row,
                                                                                     b * QK4_0 + k * 8);
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        *reinterpret_cast<h8*>(dk + (size_t) k * 8) = h8(0);
                        *reinterpret_cast<h8*>(dv + (size_t) k * 8) = h8(0);
                    }
                }
            }
        }
    }

    void operator()(sycl::nd_item<2> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
        constexpr float TAU = 8.0f;   // a row's reference moves only when a score passes it by more: p <= e^8
        sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int nsg = (int) (it.get_local_range(1) / SG);
        const int64_t grp0 = (int64_t) it.get_group(0);   // in place: the query head; gathered: the KV head
        const int64_t G = a.n_head / a.n_kv;
        const int64_t wg0 = row0 + (int64_t) it.get_group(1) * nsg * TM, my0 = wg0 + (int64_t) g * TM;
        const bool active = my0 < row_hi;
        if (!GATHER && !active) return;   // (gathered, the sub-group takes part in the work-group's copies)
        const int64_t kvh = GATHER ? grp0 : grp0 / G;
        // row i of the tile: its query and head
        auto query_of = [&](int64_t row) { return GATHER ? row / G : row; };
        auto head_of = [&](int64_t row) { return GATHER ? grp0 * G + row % G : grp0; };
        const size_t ld = GATHER || a.page_table != nullptr ? (size_t) D : (size_t) a.kv_ld;
        const size_t soff = (size_t) g * SREG, poff = (size_t) g * TM * PLD, roff = (size_t) g * 2 * TM;
        sycl::half* K = nullptr;
        sycl::half* V = nullptr;
        if constexpr (GATHER) {
            K = at(kv, 0).get();
            V = K + (size_t) CH * KLD;
        }
        for (int e = lane; e < TM; e += SG) {
            rows[roff + e] = -INFINITY;
            rows[roff + TM + e] = 0.0f;
        }
        typename Ops::A qa[D / TK];
        if (GATHER || a.q32 != nullptr) {
            // Q's TM rows as FP16 through the sub-group's local memory (rows not asked zero)
            sycl::half* qh = reinterpret_cast<sycl::half*>(at(st, soff).get());
            for (int e = lane; e < TM * D; e += SG) {
                const int64_t row = my0 + e / D, tq = query_of(row);
                float x = 0.0f;
                if (row < row_hi && tq < a.rows) {
                    const int64_t off = tq * a.q_ld + head_of(row) * D + e % D;
                    x = a.q32 != nullptr ? a.q32[off] : (float) sycl::bit_cast<sycl::half>(a.q[off]);
                }
                qh[e] = sycl::half(x);
            }
            sycl::group_barrier(sg);
            for (int kk = 0; kk < D / TK; ++kk)
                Ops::load(sg, qa[kk], sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                sycl::access::decorated::no>(qh + (size_t) kk * TK), D);
            sycl::group_barrier(sg);
        } else {
            for (int kk = 0; kk < D / TK; ++kk)
                Ops::load(sg, qa[kk], a.q + my0 * a.q_ld + grp0 * D + (int64_t) kk * TK, a.q_ld);
        }
        typename Ops::C o[NT];
        for (int nt = 0; nt < NT; ++nt) Ops::zero(sg, o[nt]);
        const int r = lane / L, part = lane % L;               // the softmax's row and its lanes' share
        const int64_t lim = my0 + r < row_hi ? lim_of(query_of(my0 + r)) : 0;
        const int64_t from = my0 + r < row_hi ? from_of(query_of(my0 + r)) : 0;
        // the keys walked: the sub-group's last row's (the limits never fall); gathered, the work-group's
        const int64_t last = GATHER ? std::min(row_hi, wg0 + (int64_t) nsg * TM) - 1 : std::min(a.rows, my0 + TM) - 1;
        const int64_t keys = lim_of(query_of(last));
        // and from the first row's first key on (a window's: the lower bounds never fall either), in whole chunks
        const int64_t first_row = GATHER ? wg0 : my0;
        const int64_t j_begin = first_row < row_hi ? from_of(query_of(first_row)) / CH * CH : 0;
        sycl::group_barrier(sg);
        for (int64_t j0 = j_begin; j0 < keys; j0 += CH) {
            const uint16_t* kc0 = nullptr;
            const uint16_t* vc0 = nullptr;
            if constexpr (GATHER) {
                sycl::group_barrier(it.get_group());   // the last chunk's K and V are read
                gather(it, j0, keys, kvh, K, V);
                sycl::group_barrier(it.get_group());
                if (!active) continue;
            } else {
                kc0 = chunk(a.k, j0, kvh);
                vc0 = chunk(a.v, j0, kvh);
                if constexpr (P == 0) {
                    // Intel's: the next chunk's keys and values into L1 while this one is worked, in blocks of 32
                    // rows x 32 columns shared out over the work-group's sub-groups (they read the same chunks).  Not
                    // on the 8-wide path: its GPUs' compiler takes no joint_matrix prefetch (the A380, IGC 2.28)
                    const int64_t jn = j0 + CH;
                    if (jn + CH <= keys) {
                        constexpr int PR = 32, PC = 32, NB = (CH / PR) * (D / PC);
                        const uint16_t* kn = chunk(a.k, jn, kvh);
                        const uint16_t* vn = chunk(a.v, jn, kvh);
                        for (int b = g; b < 2 * NB; b += nsg) {
                            const uint16_t* base = (b < NB ? kn : vn) +
                                                   (int64_t) ((b % NB) / (D / PC)) * PR * (int64_t) ld +
                                                   (int64_t) ((b % NB) % (D / PC)) * PC;
                            mx::joint_matrix_prefetch<PR, PC>(sg, const_cast<sycl::half*>(
                                                                      reinterpret_cast<const sycl::half*>(base)),
                                                              ld, mx::layout::row_major,
                                                              syclex::properties{syclex::prefetch_hint_L1});
                        }
                    }
                }
            }
            // a tile of K (keys nt TN on, dims kk TK on) and of V (keys kc TK on, dims nt TN on)
            auto load_k = [&](typename Ops::Bc& b, int nt, int kk) {
                if constexpr (GATHER)
                    Ops::load(sg, b, at(kv, (size_t) nt * TN * KLD + (size_t) kk * TK), KLD);
                else
                    Ops::load(sg, b, kc0 + (int64_t) nt * TN * (int64_t) ld + (int64_t) kk * TK, ld);
            };
            auto load_v = [&](typename Ops::Br& b, int kc, int nt) {
                if constexpr (GATHER)
                    Ops::load(sg, b, at(kv, (size_t) CH * KLD + (size_t) kc * TK * KLD + (size_t) nt * TN), KLD);
                else
                    Ops::load(sg, b, vc0 + (int64_t) kc * TK * (int64_t) ld + (int64_t) nt * TN, ld);
            };
            for (int nt = 0; nt < CH / TN; ++nt) {   // S = Q . K^T
                typename Ops::C c;
                Ops::zero(sg, c);
                for (int kk = 0; kk < D / TK; ++kk) {
                    typename Ops::Bc b;
                    load_k(b, nt, kk);
                    Ops::mad(sg, c, qa[kk], b);
                }
                Ops::store(sg, c, at(st, soff + (size_t) nt * TN), SLD);
            }
            sycl::group_barrier(sg);
            // the softmax: L lanes a row, CW contiguous scores each
            float s[CW];
            float mx_ = -INFINITY;
            const int c0 = part * CW;
            for (int i = 0; i < CW; i += 4) {
                const auto v = *reinterpret_cast<const sycl::vec<float, 4>*>(&st[soff + (size_t) r * SLD + c0 + i]);
                for (int u = 0; u < 4; ++u) {
                    const int64_t key = j0 + c0 + i + u;
                    s[i + u] = key < lim && key >= from ? v[u] * a.scale : -INFINITY;
                    mx_ = sycl::fmax(mx_, s[i + u]);
                }
            }
            for (int off = 1; off < L; off <<= 1) mx_ = sycl::fmax(mx_, sycl::permute_group_by_xor(sg, mx_, off));
            const float m_old = rows[roff + r];
            const bool move = mx_ > m_old + TAU || (m_old == -INFINITY && mx_ > -INFINITY);
            const float m_new = move ? mx_ : m_old;
            if (sycl::any_of_group(sg, move)) {
                // the rows whose reference moved scale their output and sum by exp(m_old - m_new), a tile at a time
                // through local memory (S's, read): the accumulators' layout is the vendor's
                const float corr = move && m_old > -INFINITY ? sycl::exp(m_old - m_new) : 1.0f;
                sycl::group_barrier(sg);
                if (part == 0) rows[roff + TM + r] *= corr;
                float f[TM];
                for (int i = 0; i < TM; ++i) f[i] = sycl::select_from_group(sg, corr, (size_t) i * L);
                if constexpr (WHOLE) {
                    for (int nt = 0; nt < NT; ++nt) Ops::store(sg, o[nt], at(st, soff + (size_t) nt * TN), D);
                    sycl::group_barrier(sg);
                    for (int e = lane; e < TM * D; e += SG) st[soff + e] *= f[e / D];
                    sycl::group_barrier(sg);
                    for (int nt = 0; nt < NT; ++nt) Ops::load_acc(sg, o[nt], at(st, soff + (size_t) nt * TN), D);
                } else {
                    for (int nt = 0; nt < NT; ++nt) {
                        Ops::store(sg, o[nt], at(st, soff), TN);
                        sycl::group_barrier(sg);
                        for (int e = lane; e < TM * TN; e += SG) st[soff + e] *= f[e / TN];
                        sycl::group_barrier(sg);
                        Ops::load_acc(sg, o[nt], at(st, soff), TN);
                        sycl::group_barrier(sg);
                    }
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < CW; i += 8) {
                sycl::vec<sycl::half, 8> pv;
                for (int u = 0; u < 8; ++u) {
                    // native::exp: P is FP16 anyway (the block's error unchanged; B70 11.4 -> 10.3 ms)
                    const float p = m_new == -INFINITY ? 0.0f : sycl::native::exp(s[i + u] - m_new);
                    sum += p;
                    pv[u] = sycl::half(p);
                }
                *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&ps[poff + (size_t) r * PLD + c0 + i]) = pv;
            }
            for (int off = 1; off < L; off <<= 1) sum += sycl::permute_group_by_xor(sg, sum, off);
            if (part == 0) {
                rows[roff + r] = m_new;
                rows[roff + TM + r] += sum;
            }
            sycl::group_barrier(sg);
            // O += P . V.  Intel's shapes read a tile of P once for the output's NT tiles (B70, a 1024 x 1024 block:
            // 12.5 -> 11.9 ms); the RTX 4070 was 1.4% slower that way and keeps the order of reading it again for each
            if constexpr (P == 0 || P == 1) {
                for (int kc = 0; kc < CH / TK; ++kc) {
                    typename Ops::A pa;
                    Ops::load(sg, pa, at(ps, poff + (size_t) kc * TK), PLD);
                    for (int nt = 0; nt < NT; ++nt) {
                        typename Ops::Br b;
                        load_v(b, kc, nt);
                        Ops::mad(sg, o[nt], pa, b);
                    }
                }
            } else {
                for (int nt = 0; nt < NT; ++nt) {
                    for (int kc = 0; kc < CH / TK; ++kc) {
                        typename Ops::A pa;
                        typename Ops::Br b;
                        Ops::load(sg, pa, at(ps, poff + (size_t) kc * TK), PLD);
                        load_v(b, kc, nt);
                        Ops::mad(sg, o[nt], pa, b);
                    }
                }
            }
            sycl::group_barrier(sg);
        }
        if (!active) return;
        // the output through local memory: in place all of it at once, gathered a tile at a time
        constexpr int OW = WHOLE ? D : TN, PASSES = WHOLE ? 1 : NT;
        for (int pass = 0; pass < PASSES; ++pass) {
            if constexpr (WHOLE)
                for (int nt = 0; nt < NT; ++nt) Ops::store(sg, o[nt], at(st, soff + (size_t) nt * TN), D);
            else
                Ops::store(sg, o[pass], at(st, soff), TN);
            sycl::group_barrier(sg);
            for (int e = lane; e < TM * OW; e += SG) {
                const int rr = e / OW;
                const int64_t row = my0 + rr, tq = query_of(row);
                if (row < row_hi && tq >= a.first && tq < a.row_end) {
                    const float x = st[soff + e] / rows[roff + TM + rr];
                    const int64_t to = (tq - a.first) * a.out_ld + head_of(row) * D + pass * TN + e % OW;
                    if constexpr (F32) static_cast<float*>(a.out)[to] = x;
                    else static_cast<sycl::half*>(a.out)[to] = sycl::half(x);
                }
            }
            sycl::group_barrier(sg);
        }
        }
#else
        (void) it;
#endif
    }
};

// RDNA3's and RDNA4's WMMA (path 3), as the plain models' prompt kernel ran it first (it ran the RX 9060 XT's prompt
// attention 3 to 5 times as fast as Kernel<3>: 59 ms against 185 for 16/4 heads of 256, 2,048 queries after 28K
// cells): a work-group takes rows (query, head) of one KV head, a wave 16 of them, and copies each chunk's K as
// [cell][dim] and V transposed as [dim][cell] into its local memory as FP16 (from any of the sources above: the
// fragments read 8 or 16 contiguous halves a lane there); the scores stay in the wave's accumulators, whose layout
// is known here (amd_wmma.hpp: row row_of(hi, e), column lo), the softmax across the 16 lanes of a row by shuffles,
// the probabilities through the wave's own local rows into P . V, and the output in the accumulators, scaled each
// chunk.  FP32 queries enter scaled and, heads of 128, as FP16 high and low parts (two products: FP32's precision)
template<int D, bool F32, int F>
struct WmmaKernel {
    static constexpr int BK = 32, DT = D / 16, CT = BK / 16;
    static constexpr int KS = D + 8, VS = BK + 8, PS = BK + 8;   // local row strides (halves), 16-byte multiples
    static constexpr size_t WG_BYTES = (size_t) (BK * KS + D * VS) * 2, WAVE_BYTES = (size_t) 16 * PS * 2;
    FlashAttnArgs a;
    int64_t row0, row_hi;   // (query, head) rows of the KV head
    sycl::local_accessor<uint16_t, 1> ks, vs, ps;

    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(32)>}; }

    int64_t lim_of(int64_t t) const {
        return t >= a.rows ? 0 : t < a.n_causal ? (a.key_lim ? (int64_t) a.key_lim[t] : a.pos0 + t + 1) : a.keys;
    }
    // the first key row `t` sees: a sliding window's (pos0 + t + 1 - window) for a causal row, else 0
    int64_t from_of(int64_t t) const {
        const int64_t f = a.window > 0 && t < a.n_causal ? a.pos0 + t + 1 - a.window : 0;
        return f > 0 ? f : 0;
    }
    // 8 values of cell c's K and V of KV head g from dim dd on, as FP16 bits
    void kv8(int64_t c, int64_t g, int dd, h8& k8, h8& v8) const {
        if constexpr (F < 0) {
            const int64_t off = a.page_table == nullptr
                                    ? c * a.kv_ld + g * D
                                    : (((int64_t) a.page_table[c / a.page_size] * a.n_kv + g) * a.page_size +
                                       c % a.page_size) * D;
            k8 = *reinterpret_cast<const h8*>(a.k + off + dd);
            v8 = *reinterpret_cast<const h8*>(a.v + off + dd);
        } else {
            const QsaAttnPools& p = a.pools;
            const int64_t row = ((int64_t) p.page_table[c / a.page_size] * a.n_kv + g) * a.page_size + c % a.page_size;
            if constexpr (F == kFmtF16) {
                k8 = *reinterpret_cast<const h8*>(p.k_pool + row * D + dd);
                v8 = *reinterpret_cast<const h8*>(p.v_pool + row * D + dd);
            } else if constexpr (F == kFmtI8) {
                k8 = i8_row8<D>(p.k_q, p.k_scale, row, dd);
                v8 = i8_row8<D>(p.v_q, p.v_scale, row, dd);
            } else {
                k8 = F == kFmtQ4 ? q4_row8<D>(p.k_q4, row, dd) : i8_row8<D>(p.k_q, p.k_scale, row, dd);
                v8 = q4_row8<D>(p.v_q4, row, dd);
            }
        }
    }

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_AMD_WMMA
        namespace w = xe::wmma;
        using frag = w::h16;
        constexpr float NEG = -INFINITY;
        const auto sg = it.get_sub_group();
        const int t = (int) it.get_local_id(1), nt = (int) it.get_local_range(1);
        const int wave = t / 32, lane = t % 32, lo = lane & 15, hi = lane >> 4, nw = nt / 32;
        const int64_t kvh = (int64_t) it.get_group(0), G = a.n_head / a.n_kv;
        const int64_t wg0 = row0 + (int64_t) it.get_group(1) * nw * 16, my0 = wg0 + (int64_t) wave * 16;
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* Vt = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* P = ps.get_multi_ptr<sycl::access::decorated::no>().get() + (size_t) wave * 16 * PS;
        auto frag_at = [](const uint16_t* p) { return *reinterpret_cast<const frag*>(p); };
        const bool q32 = a.q32 != nullptr, LO = q32 && D <= 128;
        const float sscale = q32 ? 1.0f : a.scale;
        // the wave's Q rows as A fragments: row lo of its tile, K values k 16 + hi KOFF .. + KN
        frag qh[DT], ql[D <= 128 ? DT : 1];
        {
            const int64_t r = my0 + lo, tq = r / G, h = kvh * G + r % G;
            const bool live = r < row_hi && tq < a.rows;
            for (int k = 0; k < DT; ++k)
                for (int e = 0; e < w::KN; ++e) {
                    const int64_t off = tq * a.q_ld + h * D + k * 16 + hi * w::KOFF + e;
                    const float v = !live ? 0.0f : q32 ? a.q32[off] * a.scale
                                                       : (float) sycl::bit_cast<sycl::half>(a.q[off]);
                    const _Float16 vh = (_Float16) v;
                    qh[k][e] = vh;
                    if constexpr (D <= 128) ql[k][e] = (_Float16) (v - (float) vh);
                }
        }
        float m[8], l[8];
        w::fv8 o[DT];
        for (int e = 0; e < 8; ++e) {
            m[e] = NEG;
            l[e] = 0.0f;
        }
        for (int k = 0; k < DT; ++k) o[k] = w::fv8{0, 0, 0, 0, 0, 0, 0, 0};
        int64_t lim[8], from[8];
        for (int e = 0; e < 8; ++e) {
            const int64_t r = my0 + w::row_of(hi, e);
            lim[e] = r < row_hi ? lim_of(r / G) : 0;
            from[e] = r < row_hi ? from_of(r / G) : 0;
        }
        const int64_t keys = lim_of((std::min(row_hi, wg0 + (int64_t) nw * 16) - 1) / G);   // the work-group's last row
        const int64_t c_begin = wg0 < row_hi ? from_of(wg0 / G) / BK * BK : 0;           // ... and its first
        for (int64_t c0 = c_begin; c0 < keys; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last chunk's K and V are read
            for (int i = t; i < BK * D / 8; i += nt) {   // K as [cell][dim], V transposed as [dim][cell]
                const int j = i / (D / 8), dd = (i - j * (D / 8)) * 8;
                const int64_t c = c0 + j;
                h8 k8(0), v8(0);
                if (c < keys) kv8(c, kvh, dd, k8, v8);
                *reinterpret_cast<h8*>(K + j * KS + dd) = k8;
                for (int e = 0; e < 8; ++e) Vt[(dd + e) * VS + j] = v8[e];
            }
            // every wave on, also one without rows (all masked): the sub-group barrier below waits for the whole
            // work-group on AMD's device compile, so a wave that skipped it would let the next copies start early
            sycl::group_barrier(it.get_group());
            w::fv8 sc[CT];
            for (int ct = 0; ct < CT; ++ct) {   // S = Q . K^T for the wave's rows
                w::C acc;
                acc.v = w::fv8{0, 0, 0, 0, 0, 0, 0, 0};
                for (int k = 0; k < DT; ++k) {
                    w::Bc<sycl::half> b;
                    b.v = frag_at(K + (ct * 16 + lo) * KS + k * 16 + hi * w::KOFF);
                    w::A<sycl::half> qa, qb;
                    qa.v = qh[k];
                    w::mad(sg, acc, qa, b, acc);
                    if (LO) {
                        qb.v = ql[k < (D <= 128 ? DT : 1) ? k : 0];
                        w::mad(sg, acc, qb, b, acc);
                    }
                }
                sc[ct] = acc.v;
            }
            float alpha[8];
            for (int e = 0; e < 8; ++e) {   // row row_of(hi, e): its 16 lanes hold its BK columns
                float mx = m[e];
                for (int ct = 0; ct < CT; ++ct) {
                    const int64_t c = c0 + ct * 16 + lo;
                    sc[ct][e] = c < lim[e] && c >= from[e] ? sc[ct][e] * sscale : NEG;
                    mx = sycl::fmax(mx, sc[ct][e]);
                }
                for (int x = 1; x < 16; x <<= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, x));
                alpha[e] = m[e] == NEG ? 0.0f : sycl::exp(m[e] - mx);
                float sum = 0.0f;
                for (int ct = 0; ct < CT; ++ct) {
                    const _Float16 p = (_Float16) (sc[ct][e] == NEG ? 0.0f : sycl::exp(sc[ct][e] - mx));
                    P[w::row_of(hi, e) * PS + ct * 16 + lo] = sycl::bit_cast<uint16_t>(p);
                    sum += (float) p;
                }
                for (int x = 1; x < 16; x <<= 1) sum += sycl::permute_group_by_xor(sg, sum, x);
                l[e] = l[e] * alpha[e] + sum;
                m[e] = mx;
            }
            sycl::group_barrier(sg);   // the wave's probabilities are written
            for (int k = 0; k < DT; ++k) {   // O = O x alpha + P . V
                for (int e = 0; e < 8; ++e) o[k][e] *= alpha[e];
                for (int k0 = 0; k0 < BK; k0 += 16) {
                    w::A<sycl::half> pa;
                    w::Bc<sycl::half> b;
                    w::C acc;
                    pa.v = frag_at(P + lo * PS + k0 + hi * w::KOFF);
                    b.v = frag_at(Vt + (k * 16 + lo) * VS + k0 + hi * w::KOFF);
                    acc.v = o[k];
                    w::mad(sg, acc, pa, b, acc);
                    o[k] = acc.v;
                }
            }
        }
        if (my0 >= row_hi) return;
        for (int k = 0; k < DT; ++k)
            for (int e = 0; e < 8; ++e) {
                const int64_t r = my0 + w::row_of(hi, e), tq = r / G;
                if (r >= row_hi || tq < a.first || tq >= a.row_end) continue;
                const float x = o[k][e] / l[e];
                const int64_t to = (tq - a.first) * a.out_ld + (kvh * G + r % G) * D + k * 16 + lo;
                if constexpr (F32) static_cast<float*>(a.out)[to] = x;
                else static_cast<sycl::half*>(a.out)[to] = sycl::half(x);
            }
#else
        (void) it;
#endif
    }
};

int path_of(const sycl::device& d) {
    static core::PerDevice<int> per_device;
    return per_device.get(d, [&d] { return path(d); });
}

// The most sub-groups a work-group the device's local memory and work-group size take (0: not one), up to 8 in place
// and 32 gathered: a work-group copies each chunk once for its rows, and on the B70 8 sub-groups' 64 rows copied each
// chunk too often (16 heads of 256 over INT8 pools, 2,048 queries after 28K cells: 294 ms at 8, 151 at 16 and more;
// the RTX 4070's 48 KiB of local memory takes fewer either way)
template<int P, int D, int F>
int nsg_most(const sycl::device& d) {
    using K = Kernel<P, D, false, F>;
    const size_t local = d.get_info<sycl::info::device::local_mem_size>();
    const size_t wg = d.get_info<sycl::info::device::max_work_group_size>();
    int nsg = K::GATHER ? 32 : 8;
    while (nsg > 1 && ((size_t) nsg * K::SG_BYTES + K::WG_BYTES > local || (size_t) nsg * K::SG > wg)) nsg /= 2;
    return (size_t) nsg * K::SG_BYTES + K::WG_BYTES <= local ? nsg : 0;
}

template<int P, int D, int F>
int nsg_most_of(const sycl::device& d) {
    static core::PerDevice<int> per_device;
    return per_device.get(d, [&d] { return nsg_most<P, D, F>(d); });
}

template<int P, int D, bool F32, int F>
void launch(sycl::queue& q, const FlashAttnArgs& a, int nsg) {
    using K = Kernel<P, D, F32, F>;
    const int64_t G = a.n_head / a.n_kv;
    // in place: query rows, its tiles inside the rows; gathered: (query, head) rows of a KV head
    const int64_t row0 = K::GATHER ? a.first * G : a.first / K::TM * K::TM;
    const int64_t row_hi = K::GATHER ? a.row_end * G : a.row_end;
    const int64_t bq = (int64_t) nsg * K::TM;
    const size_t groups = (size_t) ((row_hi - row0 + bq - 1) / bq);
    q.submit([&](sycl::handler& h) {
        K k{a, row0, row_hi, sycl::local_accessor<sycl::half, 1>(sycl::range<1>((size_t) nsg * K::TM * K::PLD), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * K::SREG), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * 2 * K::TM), h),
            sycl::local_accessor<sycl::half, 1>(sycl::range<1>(std::max<size_t>(K::WG_BYTES / 2, 1)), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) (K::GATHER ? a.n_kv : a.n_head), groups * nsg * K::SG},
                                         {1, (size_t) nsg * K::SG}),
                       k);
    });
}

template<int P, int D, int F>
bool run(sycl::queue& q, const FlashAttnArgs& a, int nsg) {
    const int most = nsg_most_of<P, D, F>(q.get_device());
    if (most == 0) return false;
    nsg = nsg <= 0 ? most : std::min(nsg, most);
    if (a.out_f32) launch<P, D, true, F>(q, a, nsg);
    else launch<P, D, false, F>(q, a, nsg);
    return true;
}

template<int P, int D>
bool run_fmt(sycl::queue& q, const FlashAttnArgs& a, int nsg) {
    switch (a.pools_format) {
    case -1: return run<P, D, -1>(q, a, nsg);
    case kFmtF16: return run<P, D, kFmtF16>(q, a, nsg);
    case kFmtI8: return run<P, D, kFmtI8>(q, a, nsg);
    case kFmtQ4: return run<P, D, kFmtQ4>(q, a, nsg);
    case kFmtK8V4: return run<P, D, kFmtK8V4>(q, a, nsg);
    default: return false;
    }
}

// WmmaKernel: the most waves a work-group the device's local memory and work-group size take, up to 16 (more rows
// sharing each chunk's copies: the RX 9060 XT ran Qwen3.5-9B's 32K prompt's attention 5.1 s at 16 against 11.3 at 4)
template<int D>
int wmma_most(const sycl::device& d) {
    static core::PerDevice<int> per_device;
    return per_device.get(d, [&d] {
        using K = WmmaKernel<D, false, -1>;
        const size_t local = d.get_info<sycl::info::device::local_mem_size>();
        const size_t wg = d.get_info<sycl::info::device::max_work_group_size>();
        int nw = 16;
        while (nw > 1 && (K::WG_BYTES + (size_t) nw * K::WAVE_BYTES > local || (size_t) nw * 32 > wg)) nw /= 2;
        return K::WG_BYTES + (size_t) nw * K::WAVE_BYTES <= local ? nw : 0;
    });
}

template<int D, bool F32, int F>
void launch_wmma(sycl::queue& q, const FlashAttnArgs& a, int nw) {
    using K = WmmaKernel<D, F32, F>;
    const int64_t G = a.n_head / a.n_kv, row0 = a.first * G, row_hi = a.row_end * G, bq = (int64_t) nw * 16;
    const size_t groups = (size_t) ((row_hi - row0 + bq - 1) / bq);
    q.submit([&](sycl::handler& h) {
        K k{a, row0, row_hi, sycl::local_accessor<uint16_t, 1>(sycl::range<1>((size_t) K::BK * K::KS), h),
            sycl::local_accessor<uint16_t, 1>(sycl::range<1>((size_t) D * K::VS), h),
            sycl::local_accessor<uint16_t, 1>(sycl::range<1>((size_t) nw * 16 * K::PS), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) a.n_kv, groups * nw * 32}, {1, (size_t) nw * 32}), k);
    });
}

template<int D>
bool run_wmma(sycl::queue& q, const FlashAttnArgs& a, int nw) {
    const int most = wmma_most<D>(q.get_device());
    if (most == 0) return false;
    nw = nw <= 0 ? most : std::min(nw, most);
    auto go = [&](auto f32) {
        constexpr bool B = decltype(f32)::value;
        switch (a.pools_format) {
        case -1: launch_wmma<D, B, -1>(q, a, nw); return true;
        case kFmtF16: launch_wmma<D, B, kFmtF16>(q, a, nw); return true;
        case kFmtI8: launch_wmma<D, B, kFmtI8>(q, a, nw); return true;
        case kFmtQ4: launch_wmma<D, B, kFmtQ4>(q, a, nw); return true;
        case kFmtK8V4: launch_wmma<D, B, kFmtK8V4>(q, a, nw); return true;
        default: return false;
        }
    };
    return a.out_f32 ? go(std::true_type{}) : go(std::false_type{});
}

template<int D>
bool run_path(sycl::queue& q, const FlashAttnArgs& a, int nsg, int p) {
    switch (p) {
    case 0: return run_fmt<0, D>(q, a, nsg);
    case 1: return run_fmt<1, D>(q, a, nsg);
    case 2: return run_fmt<2, D>(q, a, nsg);
    case 3: return run_wmma<D>(q, a, nsg);
    default: return false;
    }
}

template<int D>
int most_of(const sycl::device& d, int p) {
    switch (p) {
    case 0: return nsg_most_of<0, D, -1>(d);
    case 1: return nsg_most_of<1, D, -1>(d);
    case 2: return nsg_most_of<2, D, -1>(d);
    case 3: return wmma_most<D>(d);
    default: return 0;
    }
}

// ---- the FMA form, for a GPU without FP16 matrix engines (flash_attn_path -1) and FlashAttnArgs::fma: the plain
// models' former prompt kernel, which ran about 6 TFLOP/s on the RTX 4070 without its tensor cores against the image
// engine's own FMA kernel's 5.4 (16/4 heads of 256, 2,048 queries after 28K cells; 32 heads of 128, 4,096 rows).
// A work-group of 256 work-items takes BQ queries of one KV head: R = BQ x G rows (query, head), Q in local memory as
// FP32 (scaled), the step's BK cells of K and V as FP16, the step's scores (then probabilities) per row.  Local memory
// is read 4 values at a time (Q and the probabilities as float4, K as 4 halves), the rows padded to keep those reads
// aligned and K's rows off one bank.  Scores: work-item t takes cell t % BK of rows t / BK, + NT / BK, ...:
// neighbouring work-items read neighbouring K rows and the same Q row.  Output: work-item t owns column d = t % HD of
// rows t / HD, + NT / HD, ... in registers.  FP16 K and V only, dense or in pages of any size.
constexpr int FMA_NT = 256;   // work-items a work-group

template<int HD, int G, int BQ, int BK, bool F32>
struct FmaKernel {
    static constexpr int NT = FMA_NT, R = BQ * G, RS = NT / HD, OPT = R / RS, QS = HD + 4, KS = HD + 4;
    static constexpr int SG = NT / BK, SR = (R + SG - 1) / SG;   // the scores: row groups, rows a work-item
    static_assert(NT % HD == 0 && R % RS == 0 && HD % 8 == 0 && BK % 4 == 0, "the work splits evenly");
    using f4 = sycl::vec<float, 4>;
    using h4 = sycl::vec<uint16_t, 4>;
    FlashAttnArgs a;
    sycl::local_accessor<float, 1> qs, sc, mrow, lrow, arow;
    sycl::local_accessor<uint16_t, 1> ks, vs;

    static float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }
    int64_t lim_of(int64_t t) const {
        return t >= a.rows ? 0 : t < a.n_causal ? (a.key_lim ? (int64_t) a.key_lim[t] : a.pos0 + t + 1) : a.keys;
    }
    // the first key row `t` sees: a sliding window's (pos0 + t + 1 - window) for a causal row, else 0
    int64_t from_of(int64_t t) const {
        const int64_t f = a.window > 0 && t < a.n_causal ? a.pos0 + t + 1 - a.window : 0;
        return f > 0 ? f : 0;
    }

    void operator()(sycl::nd_item<2> it) const {
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const int t = (int) it.get_local_id(1);
        const int64_t h = (int64_t) it.get_group(0);   // the KV head
        const int64_t qbase = a.first + (int64_t) it.get_group(1) * BQ;
        const int d = t % HD, r0 = t / HD;
        float* Q = qs.get_multi_ptr<sycl::access::decorated::no>().get();
        float* S = sc.get_multi_ptr<sycl::access::decorated::no>().get();
        float* M = mrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* L = lrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* A = arow.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* V = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = t; i < R * HD; i += NT) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G, off = tq * a.q_ld + (h * G + r % G) * HD + dd;
            const float x = tq >= a.row_end ? 0.0f : a.q32 ? a.q32[off] : h2f(a.q[off]);
            Q[r * QS + dd] = x * a.scale;
        }
        if (t < R) {
            M[t] = NEG;
            L[t] = 0.0f;
        }
        float o[OPT];
        for (int k = 0; k < OPT; ++k) o[k] = 0.0f;
        const int64_t keys = lim_of(sycl::min<int64_t>(qbase + BQ, a.row_end) - 1);   // the block's last query's
        const int j = t % BK, rg = t / BK;
        for (int64_t c0 = from_of(qbase) / BK * BK; c0 < keys; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K, V and probabilities are read
            for (int i = t; i < BK * HD / 8; i += NT) {   // 8 halves a work-item
                const int jj = i / (HD / 8), dd = (i - jj * (HD / 8)) * 8;
                const int64_t c = c0 + jj;
                h8 kv(0), vv(0);
                if (c < keys) {
                    const int64_t row = a.page_table == nullptr
                                            ? c * a.kv_ld + h * HD
                                            : (((int64_t) a.page_table[c / a.page_size] * a.n_kv + h) * a.page_size +
                                               c % a.page_size) * HD;
                    kv = *reinterpret_cast<const h8*>(a.k + row + dd);
                    vv = *reinterpret_cast<const h8*>(a.v + row + dd);
                }
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd) = kv.template swizzle<0, 1, 2, 3>();
                *reinterpret_cast<h4*>(K + (ptrdiff_t) jj * KS + dd + 4) = kv.template swizzle<4, 5, 6, 7>();
                *reinterpret_cast<h8*>(V + (ptrdiff_t) jj * HD + dd) = vv;
            }
            sycl::group_barrier(it.get_group());
            {   // scores, masked past each row's keys
                const int64_t c = c0 + j;
                float acc[SR];
                for (int k = 0; k < SR; ++k) acc[k] = 0.0f;
                for (int dd = 0; dd < HD; dd += 4) {
                    const h4 kh = *reinterpret_cast<const h4*>(K + (ptrdiff_t) j * KS + dd);
                    const float k0 = h2f(kh[0]), k1 = h2f(kh[1]), k2 = h2f(kh[2]), k3 = h2f(kh[3]);
                    for (int k = 0; k < SR; ++k) {
                        const int r = rg + k * SG;
                        if (SR * SG > R && r >= R) break;
                        const f4 qv = *reinterpret_cast<const f4*>(Q + (ptrdiff_t) r * QS + dd);
                        acc[k] = sycl::fma(qv[0], k0, sycl::fma(qv[1], k1, sycl::fma(qv[2], k2, sycl::fma(qv[3], k3, acc[k]))));
                    }
                }
                for (int k = 0; k < SR; ++k) {
                    const int r = rg + k * SG;
                    if (r >= R) break;
                    S[r * BK + j] = c < lim_of(qbase + r / G) && c >= from_of(qbase + r / G) ? acc[k] : NEG;
                }
            }
            sycl::group_barrier(it.get_group());
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                float mx = M[t];
                for (int jj = 0; jj < BK; ++jj) mx = sycl::fmax(mx, S[t * BK + jj]);
                const float f = M[t] == NEG ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int jj = 0; jj < BK; ++jj) {
                    const float p = S[t * BK + jj] == NEG ? 0.0f : sycl::exp(S[t * BK + jj] - mx);
                    S[t * BK + jj] = p;
                    sum += p;
                }
                L[t] = L[t] * f + sum;
                M[t] = mx;
                A[t] = f;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < OPT; ++k) o[k] *= A[r0 + k * RS];
            for (int jj = 0; jj < BK; jj += 4) {
                const float v0 = h2f(V[jj * HD + d]), v1 = h2f(V[(jj + 1) * HD + d]), v2 = h2f(V[(jj + 2) * HD + d]),
                            v3 = h2f(V[(jj + 3) * HD + d]);
                for (int k = 0; k < OPT; ++k) {
                    const f4 p = *reinterpret_cast<const f4*>(S + (ptrdiff_t) (r0 + k * RS) * BK + jj);
                    o[k] = sycl::fma(p[0], v0, sycl::fma(p[1], v1, sycl::fma(p[2], v2, sycl::fma(p[3], v3, o[k]))));
                }
            }
        }
        for (int k = 0; k < OPT; ++k) {
            const int r = r0 + k * RS;
            const int64_t tq = qbase + r / G;
            if (tq >= a.row_end) continue;
            const float x = o[k] / L[r];
            const int64_t to = (tq - a.first) * a.out_ld + (h * G + r % G) * HD + d;
            if constexpr (F32) static_cast<float*>(a.out)[to] = x;
            else static_cast<sycl::half*>(a.out)[to] = sycl::half(x);
        }
    }
};

template<int HD, int G, int BQ, int BK, bool F32>
void launch_fma(sycl::queue& q, const FlashAttnArgs& a) {
    using K = FmaKernel<HD, G, BQ, BK, F32>;
    const size_t blocks = (size_t) ((a.row_end - a.first + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) K::R * K::QS), h), sc(sycl::range<1>((size_t) K::R * BK), h),
            mrow(sycl::range<1>(K::R), h), lrow(sycl::range<1>(K::R), h), arow(sycl::range<1>(K::R), h);
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) BK * K::KS), h), vs(sycl::range<1>((size_t) BK * HD), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) a.n_kv, blocks * FMA_NT), sycl::range<2>(1, FMA_NT)),
                       K{a, qs, sc, mrow, lrow, arow, ks, vs});
    });
}

// the shapes the FMA form is built for: heads of 128 with 1, 2, 4, 8 or 10 query heads a KV head, of 256 with 1, 2,
// 4, 8 or 12 (BQ x G rows of 16 to 40 a work-group: within the 48 KiB of local memory every GPU gives)
template<bool F32>
bool run_fma_t(sycl::queue& q, const FlashAttnArgs& a) {
    const int64_t G = a.n_head / a.n_kv;
    if (a.head_dim == 128) {
        switch (G) {
        case 1: launch_fma<128, 1, 16, 32, F32>(q, a); return true;
        case 2: launch_fma<128, 2, 8, 32, F32>(q, a); return true;
        case 4: launch_fma<128, 4, 4, 32, F32>(q, a); return true;
        case 8: launch_fma<128, 8, 2, 32, F32>(q, a); return true;
        case 10: launch_fma<128, 10, 4, 32, F32>(q, a); return true;
        default: return false;
        }
    }
    if (a.head_dim == 256) {
        switch (G) {
        case 1: launch_fma<256, 1, 16, 16, F32>(q, a); return true;
        case 2: launch_fma<256, 2, 8, 16, F32>(q, a); return true;
        case 4: launch_fma<256, 4, 4, 16, F32>(q, a); return true;
        case 8: launch_fma<256, 8, 2, 16, F32>(q, a); return true;
        case 12: launch_fma<256, 12, 2, 16, F32>(q, a); return true;
        default: return false;
        }
    }
    return false;
}

bool run_fma(sycl::queue& q, const FlashAttnArgs& a) {
    if (a.pools_format >= 0 || a.sel_ids != nullptr || (a.q == nullptr && a.q32 == nullptr)) return false;
    return a.out_f32 ? run_fma_t<true>(q, a) : run_fma_t<false>(q, a);
}

// ---- the selection form (Qwen3.8-Flash-Next's QSA: a query reads the cells it selected, FlashAttnArgs::sel_ids),
// from qsa_prompt_attn.cpp.  Each query's cells are its own, so a work-group takes one (query, KV head) and splits
// the head's dims over its sub-groups for the parallelism the few rows lack; the row form above ran it 4 to 6 times
// as long and off FP64 by 1e-3 of the output (FP16 queries and probabilities, against these kernels' 22 bits; B70
// and RTX 4070, 32K cells, 2,048 queries).
// The scheme of Strata's CUDA kernel (src/kernels/cuda/qsa_prompt_attn.cu, its "v2"), on SYCL joint_matrix:
//   * a work-group of 8 sub-groups takes one (query, KV head) and walks its selected cells CH at a time with an
//     online softmax; sub-groups 2g and 2g + 1 own dims [64g, 64g + 64), which is int8 scale group g, for both q.k
//     and p.v, rows 0-7 and 8-15 of q and of the output;
//   * the chunk's K and V rows are gathered into local memory as FP16: int8 codes are exact in FP16, and their
//     per-64 scales are applied in FP32 to each group's q.k partial and folded into p for p.v; so are Q4_0's codes
//     (minus 8) and its signed per-32 scales, two per 64-dim group (upstream 778e1f6);
//   * q is scaled by a power of two that puts its largest value near 2^14 and split into FP16 hi + lo parts, and so
//     is each group's p (relative to the chunk's largest V scale): two products each, so they keep about 22 bits;
//   * the four groups' q.k partials are added in a fixed order: the result does not depend on scheduling;
//   * q stays in registers as joint_matrix tiles, and so does the output: each chunk scales its rows by the
//     softmax's correction and the change of V scale, then adds the chunk's p.v.
// Not bitwise equal to `qsa_decode_attn_batch`; qsa_prompt_attn_parity bounds the difference.
namespace sel {


namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using sycl::half;

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head (+ 4 zero rows: two 8-row tiles)
constexpr int R = 16;             // rows of q, p and the output
constexpr int NG = 4;             // 64-dim groups
constexpr int SGS = 16;           // sub-group size
constexpr int NSG = 2 * NG;       // sub-groups: a group's dims x a tile of 8 rows
constexpr int WG = NSG * SGS;     // work-items
constexpr int QS = HD + 8;        // row strides in halves (16-byte aligned rows)
constexpr int KR = HD + 8;
// [[maybe_unused]]: the device compiles for other GPUs leave out the kernel that uses these (device_target.hpp)
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
// Lazy rescaling (as FlashAttention-3): a row's softmax reference moves only when a score exceeds it by more than
// 2^TAU, so p <= 2^TAU, and p' = p * vscale / (largest V scale so far) * 2^(14 - TAU) stays within FP16.  Most chunks
// then change no reference and leave the output unscaled.
[[maybe_unused]] constexpr float TAU = 8.0f;
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;   // 2^(14 - TAU)

using xe::attn::at;   // the attention kernels' shared parts (attn_parts.hpp)

// how a pool holds K or V: FP16 values (scales 1), INT8 codes with an FP16 scale per 64 values, or Q4_0 blocks (codes
// 0-15 minus 8, exact in FP16, with a signed FP16 scale per 32 values; kv_q4.hpp)
enum KvFmt { kF16 = 0, kI8 = 1, kQ4 = 2 };

template<int KF, int VF, int CH>   // K's and V's KvFmt
struct PromptAttn {
    static constexpr int KG = KF == kQ4 ? 2 * NG : NG;   // K's scale groups (q.k partials), V's
    static constexpr int VG = VF == kQ4 ? 2 * NG : NG;
    static constexpr int Q4ROW = (HD / QK4_0) * (int) sizeof(block_q4_0);   // bytes of a Q4_0 row (one head, one cell)
    // Q4_0 block `b` of a row into FP16 row `dst` (its 32 values) and its scale: element j is the low nibble of byte j
    // (j < 16), else the high nibble of byte j - 16, minus 8.  The block's 18 bytes are 2-byte aligned
    static float q4_block(const uint8_t* row, int b, half* dst) {
        // five 4-byte loads from the 4-byte boundary at or below the block (a row starts 16-byte aligned, a block 2)
        const int start = b * (int) sizeof(block_q4_0), delta = start & 3;
        const auto* w4 = reinterpret_cast<const uint32_t*>(row + (start - delta));
        uint32_t u[5];
        for (int i = 0; i < 5; ++i) u[i] = w4[i];
        sycl::vec<uint32_t, 4> cw;
        uint16_t dbits;
        if (delta == 0) {   // d in bytes 0-1, the codes from byte 2
            dbits = (uint16_t) (u[0] & 0xFFFF);
            for (int i = 0; i < 4; ++i) cw[i] = (u[i] >> 16) | (u[i + 1] << 16);
        } else {            // d in bytes 2-3, the codes from byte 4
            dbits = (uint16_t) (u[0] >> 16);
            for (int i = 0; i < 4; ++i) cw[i] = u[i + 1];
        }
        const sycl::vec<uint16_t, 8> v = cw.template as<sycl::vec<uint16_t, 8>>();
        // bytes 2i and 2i + 1 of the codes are the low and high bytes of v[i]: their nibbles in element order
        const sycl::vec<uint16_t, 8> lo = v & (uint16_t) 0x0F0F, hi = (v >> 4) & (uint16_t) 0x0F0F;
        const auto l8 = lo.template as<sycl::vec<int8_t, 16>>() - (int8_t) 8;
        const auto h8 = hi.template as<sycl::vec<int8_t, 16>>() - (int8_t) 8;
        auto* d = reinterpret_cast<sycl::vec<half, 16>*>(dst);
        d[0] = l8.template convert<half>();
        d[1] = h8.template convert<half>();
        return f32_from_f16(dbits);
    }

    // x as FP16 hi + lo: hi rounded, lo the rest.  hi goes back to FP32 through integer operations: written as
    // x - (float) half(x), the compiler took the round trip for exact and lo came out zero (the products kept
    // FP16's 11 bits)
    static void split(float x, half& hi, half& lo) {
        hi = half(x);
        lo = half(x - f32_from_f16(sycl::bit_cast<uint16_t>(hi)));
    }

    static constexpr int KV_HALVES = 2 * CH * KR;   // FP16 K and V rows; q (hi + lo) shares them at the start
    static_assert(KV_HALVES >= 2 * R * QS, "q's staging must fit in the K/V rows");
    const float* qv;
    QsaAttnPools p;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, n_kv, page_size;
    float scale_log2;
    float* attn;
    sycl::local_accessor<half, 1> kvq;      // kh [CH][KR], vh [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<float, 2> ks, vs;  // [CH][KG], [CH][VG]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 3> part;    // [NG][R][CH]: q.k per 64-dim group; part[0] then holds p
    sycl::local_accessor<half, 3> pp;       // [2 VG][R][CH]: p' hi (2v) / lo (2v + 1) per V group
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int64_t qi = (int64_t) it.get_group(0), kvh = (int64_t) it.get_group(1);
        const int t = (int) it.get_local_id(1), s = (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const int g = s / 2, mt = s % 2;   // this sub-group's dim group and row tile
        const int64_t n_head = n_kv * G;
        const float* qp = qv + qi * n_head * HD + kvh * G * HD;
        float* out = attn + qi * n_head * HD + kvh * G * HD;
        const int32_t* sel = ids + qi * cap;
        const int n = steps[qi * kStepCount + kStepWidth];
        const size_t KH = 0, VH = (size_t) CH * KR, QH = 0, QL = (size_t) R * QS;

        // q: 12 heads + 4 zero rows, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < G * HD; i += WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;   // qm < 2^qe
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * HD; i += WG) {
            const int r = i / HD, d = i % HD;
            half hi, lo;
            split(r < G ? qp[r * HD + d] * qup : 0.0f, hi, lo);
            kvq[QH + (size_t) r * QS + d] = hi;
            kvq[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        // this sub-group's q: its 8 rows x 4 steps of 16 of its 64 dims, hi and lo, kept in registers
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) mt * TM * QS + (size_t) g * 64 + (size_t) kk * TK;
            mx::joint_matrix_load(sg, qa[kk], at(kvq, QH + off), QS);
            mx::joint_matrix_load(sg, qb[kk], at(kvq, QL + off), QS);
        }
        // this sub-group's 8 output rows x its 64 dims, in units of `unit` (the largest V scale so far / PSCALE)
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[4];
        // by index: a range-for of references over a 2-D joint_matrix array left elements unfilled (iq_kernels.cpp)
        for (int nt = 0; nt < 4; ++nt) mx::joint_matrix_fill(sg, acc[nt], 0.0f);
        float unit = 0.0f;

        for (int c0 = 0; c0 < n; c0 += CH) {
            const int nh = sycl::min(CH, n - c0);
            if (t < CH) {
                long long r = -1;
                if (t < nh) {
                    const int cell = sel[c0 + t];
                    const long long page = (long long) p.page_table[cell / page_size];
                    // page -1: a block the KV streaming could not make resident, masked below (upstream f3925cf)
                    if (page >= 0) r = (page * n_kv + kvh) * page_size + cell % page_size;
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);   // rows ready; the previous chunk (and q's staging) is done with kvq, part, pp
            // gather K and V as FP16, 8 values a piece
            for (int i = t; i < CH * (HD / 8); i += WG) {
                const int c = i / (HD / 8), pc = i % (HD / 8);
                const long long r = rows[c];
                sycl::vec<half, 8> kx(half(0.0f)), vx(half(0.0f));
                if (r >= 0) {
                    const int64_t off = r * HD + (int64_t) pc * 8;
                    // int8 codes are exact in FP16; Xe converts them in one instruction (CUDA's mantissa trick, two
                    // values at a time, was 10% slower here)
                    if constexpr (KF == kI8)
                        kx = reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + off)->template convert<half>();
                    else if constexpr (KF == kF16)
                        kx = *reinterpret_cast<const sycl::vec<half, 8>*>(p.k_pool + off);
                    if constexpr (VF == kI8)
                        vx = reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.v_q + off)->template convert<half>();
                    else if constexpr (VF == kF16)
                        vx = *reinterpret_cast<const sycl::vec<half, 8>*>(p.v_pool + off);
                }
                if constexpr (KF != kQ4)
                    *reinterpret_cast<sycl::vec<half, 8>*>(&kvq[KH + (size_t) c * KR + (size_t) pc * 8]) = kx;
                if constexpr (VF != kQ4)
                    *reinterpret_cast<sycl::vec<half, 8>*>(&kvq[VH + (size_t) c * KR + (size_t) pc * 8]) = vx;
            }
            // Q4_0: one work-item a block (32 values and the scale): CH x 8 = 128 blocks, one each
            if constexpr (KF == kQ4 || VF == kQ4)
                for (int i = t; i < CH * (HD / QK4_0); i += WG) {
                    const int c = i / (HD / QK4_0), b = i % (HD / QK4_0);
                    const long long r = rows[c];
                    half* kd = &kvq[KH + (size_t) c * KR + (size_t) b * QK4_0];
                    half* vd = &kvq[VH + (size_t) c * KR + (size_t) b * QK4_0];
                    if constexpr (KF == kQ4) {
                        float a = 0.0f;
                        if (r >= 0) a = q4_block(p.k_q4 + r * Q4ROW, b, kd);
                        else for (int j = 0; j < QK4_0; ++j) kd[j] = half(0.0f);
                        ks[c][b] = a;
                    }
                    if constexpr (VF == kQ4) {
                        float a = 0.0f;
                        if (r >= 0) a = q4_block(p.v_q4 + r * Q4ROW, b, vd);
                        else for (int j = 0; j < QK4_0; ++j) vd[j] = half(0.0f);
                        vs[c][b] = a;
                    }
                }
            for (int i = t; i < CH * 2 * NG; i += WG) {
                const int c = i / (2 * NG), gg = i % (2 * NG);
                const long long r = rows[c];
                if (KF != kQ4 && gg < KG) {   // Q4_0's scales come with its blocks below
                    float a = 0.0f;
                    if (r >= 0) a = KF == kI8 ? f32_from_f16(p.k_scale[r * (HD / KV_Q8_GROUP) + gg]) : 1.0f;
                    ks[c][gg] = a;
                }
                if (VF != kQ4 && gg < VG) {
                    float b = 0.0f;
                    if (r >= 0) b = VF == kI8 ? f32_from_f16(p.v_scale[r * (HD / KV_Q8_GROUP) + gg]) : 1.0f;
                    vs[c][gg] = b;
                }
            }
            sycl::group_barrier(grp);

            // q.k of this sub-group's 8 rows and 64 dims: CH/16 cell tiles, 4 steps of 16 dims, hi and lo.  Q4_0's two
            // 32-dim halves get their own products, scaled in registers (lane = cell, as for the output below) and
            // added: one partial per 64 dims, so the local memory stays within four work-groups a core on the B70
            for (int nt = 0; nt < CH / TN; ++nt) {
                constexpr int KPG = KG / NG, KK = 4 / KPG;   // this sub-group's K groups, the 16-dim steps of each
                auto product = [&](auto& c, int kp) {
                    mx::joint_matrix_fill(sg, c, 0.0f);
                    for (int k2 = 0; k2 < KK; ++k2) {
                        const int kk = kp * KK + k2;
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> b;
                        const size_t off = KH + (size_t) nt * TN * KR + (size_t) g * 64 + (size_t) kk * TK;
                        mx::joint_matrix_load(sg, b, at(kvq, off), KR);
                        mx::joint_matrix_mad(sg, c, qa[kk], b, c);
                        mx::joint_matrix_mad(sg, c, qb[kk], b, c);
                    }
                };
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c0;
                product(c0, 0);
                if constexpr (KPG == 2) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c1;
                    product(c1, 1);
                    const size_t cell = (size_t) nt * TN + lane, k0 = (size_t) 2 * g;
                    const float s0 = ks[cell][k0], s1 = ks[cell][k0 + 1];
                    float hi[TM];
                    int i = 0;
                    mx::joint_matrix_apply(sg, c1, [&](float& x) { hi[i++] = x; });
                    i = 0;
                    mx::joint_matrix_apply(sg, c0, [&](float& x) { x = x * s0 + hi[i] * s1; ++i; });
                }
                mx::joint_matrix_store(sg, c0, at(part, (g * R + mt * TM) * CH + nt * TN), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);

            // online softmax: row t/8, CH/8 cells each, 8 neighbouring lanes a row; the groups' partials in order
            {
                constexpr int PER = CH / 8;
                const int r = t / 8, sub = t % 8;
                float x[PER], mxv = NEG_INF;
                for (int j = 0; j < PER; ++j) {
                    const int c = sub * PER + j;
                    float sc;   // the groups' partials in order (Q4_0's already scaled)
                    if constexpr (KF == kQ4)
                        sc = ((part[0][r][c] + part[1][r][c]) + part[2][r][c]) + part[3][r][c];
                    else
                        sc = (((part[0][r][c] * ks[c][0] + part[1][r][c] * ks[c][1]) + part[2][r][c] * ks[c][2]) +
                              part[3][r][c] * ks[c][3]);
                    x[j] = c < nh && rows[c] >= 0 ? sc * qdown : NEG_INF;
                    mxv = sycl::fmax(mxv, x[j]);
                }
                for (int o = 1; o < 8; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = mrow[r];
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                float sum = 0.0f;
                for (int j = 0; j < PER; ++j) {
                    const float e = x[j] == NEG_INF ? 0.0f : sycl::exp2(x[j] - m_new);
                    part[0][r][sub * PER + j] = e;   // this work-item's own cells: read above, no one else's
                    sum += e;
                }
                for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                if (sub == 0) {
                    const float a = m_old == m_new ? 1.0f : m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    alpha[r] = a;
                    lsum[r] = sycl::fma(lsum[r], a, sum);
                    mrow[r] = m_new;
                }
            }
            sycl::group_barrier(grp);

            // p.v of this sub-group's 8 rows and 64 dims: p' = p * vscale / (largest |V scale| so far) * PSCALE, hi +
            // lo, one p' per V scale group (two of 32 dims for Q4_0, whose scales are signed)
            {
                constexpr int VPG = VG / NG;   // this sub-group's V groups
                float vmax = 0.0f;
                for (int c = lane; c < CH; c += SGS)
                    for (int vp = 0; vp < VPG; ++vp) vmax = sycl::fmax(vmax, sycl::fabs(vs[c][g * VPG + vp]));
                vmax = sycl::reduce_over_group(sg, vmax, sycl::maximum<float>());
                // the output so far into the new units when the largest V scale grows
                const float vdown = sycl::fmax(vmax * (1.0f / PSCALE), unit);
                const float ratio = vdown > unit && unit > 0.0f ? unit / vdown : 1.0f;
                unit = vdown;
                const float vup = vdown > 0.0f ? 1.0f / vdown : 0.0f;
                for (int vp = 0; vp < VPG; ++vp) {
                    const int vg = g * VPG + vp;
                    const float w = vs[lane][vg] * vup;   // CH == SGS: lane `lane` owns cell `lane` of the 8 rows
                    for (int k = 0; k < TM; ++k) {
                        const int r = mt * TM + k;
                        half hi, lo;
                        split(part[0][r][lane] * w, hi, lo);
                        pp[(size_t) 2 * vg][r][lane] = hi;
                        pp[(size_t) 2 * vg + 1][r][lane] = lo;
                    }
                }
                // An 8 x 16 accumulator gives each lane one column, its element i in row i (Xe's DPAS layout; the
                // parity test checks it): scaled by element order, as asking each element its coordinates (the
                // coordinate form of joint_matrix_apply) cost more than all the chunk's products.  Skipped when no
                // reference moved (the same for every lane of the sub-group)
                float f[TM];
                bool same = true;
                for (int i = 0; i < TM; ++i) {
                    f[i] = alpha[mt * TM + i] * ratio;
                    same = same && f[i] == 1.0f;
                }
                if (!same)
                    for (int nt = 0; nt < 4; ++nt) {
                        int i = 0;
                        mx::joint_matrix_apply(sg, acc[nt], [&](float& x) { x *= f[i++]; });
                    }
                sycl::group_barrier(sg);
#pragma unroll
                for (int kc = 0; kc < CH / TK; ++kc)
#pragma unroll
                    for (int vp = 0; vp < VPG; ++vp) {
                        const int vg = g * VPG + vp;
                        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> ph, pl;
                        mx::joint_matrix_load(sg, ph, at(pp, (2 * vg * R + mt * TM) * CH + kc * TK), CH);
                        mx::joint_matrix_load(sg, pl, at(pp, ((2 * vg + 1) * R + mt * TM) * CH + kc * TK), CH);
#pragma unroll
                        for (int n2 = 0; n2 < 4 / VPG; ++n2) {   // the 16-dim output tiles of this V group
                            const int nt = vp * (4 / VPG) + n2;
                            mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> b;
                            const size_t off = VH + (size_t) kc * TK * KR + (size_t) g * 64 + (size_t) nt * TN;
                            mx::joint_matrix_load(sg, b, at(kvq, off), KR);
                            mx::joint_matrix_mad(sg, acc[nt], ph, b, acc[nt]);
                            mx::joint_matrix_mad(sg, acc[nt], pl, b, acc[nt]);
                        }
                    }
            }
        }
        sycl::group_barrier(grp);
        for (int nt = 0; nt < 4; ++nt)
            ix::joint_matrix_apply(sg, acc[nt], [&](float& x, size_t row, size_t col) {
                const int r = mt * TM + (int) row;
                if (r < G) {
                    const float l = lsum[r];
                    out[r * HD + g * 64 + nt * TN + (int) col] = l > 0.0f ? x * unit / l : 0.0f;
                }
            });
#endif
    }
    // the large register file, which holds q, the output and the chunk's tiles: the default one ran 30 ms against 21
    // (B70, int8, 2,048 queries over 32K cells).  The caller runs this kernel only where xmx_available(), which
    // requires the mode
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

// the work-group's local memory, as `launch` allocates it
template<int KF, int VF, int CH>
constexpr size_t local_bytes() {
    using K = PromptAttn<KF, VF, CH>;
    return (size_t) K::KV_HALVES * 2 + (size_t) CH * (K::KG + K::VG) * 4 + (size_t) CH * 8 + (size_t) NG * R * CH * 4 +
           (size_t) 2 * K::VG * R * CH * 2 + (size_t) 3 * R * 4;
}

template<int KF, int VF, int CH>
sycl::event launch(sycl::queue& q, const float* qv, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                   int64_t cap, int64_t n_kv, int64_t page_size, float* attn, int64_t n_q) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    return q.submit([&](sycl::handler& h) {
        using K = PromptAttn<KF, VF, CH>;
        K k{qv, p, ids, steps, cap, n_kv, page_size, scale_log2, attn,
            sycl::local_accessor<half, 1>(sycl::range<1>(K::KV_HALVES), h),
            sycl::local_accessor<float, 2>(sycl::range<2>(CH, K::KG), h),
            sycl::local_accessor<float, 2>(sycl::range<2>(CH, K::VG), h),
            sycl::local_accessor<long long, 1>(sycl::range<1>(CH), h),
            sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
            sycl::local_accessor<half, 3>(sycl::range<3>((size_t) 2 * K::VG, R, CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_q, (size_t) (n_kv * WG)}, {1, WG}), k);
    });
}

// The portable form, for a GPU without Intel's XMX kernels whose joint_matrix runs FP16 16 x 16 x 16 on 32 lanes
// (NVIDIA's tensor cores, from sm_70 on; xe::mma_shape_is).  The scheme above with the accumulators' layout unknown:
// 4 sub-groups of 32, sub-group g owns dim group g (int8 scale group g) for all 16 rows (one 16-row tile).  The output
// is not kept in the accumulators: each chunk's p.v starts from zero and goes through local memory into FP32 registers
// (a lane 32 of the sub-group's 16 x 64), where it is scaled by the chunk's unit (its largest V scale / PSCALE) and
// added to the output scaled by the softmax's correction.  Accumulated over the whole selection in the tensor cores,
// which align and cut the addends instead of rounding them, the output was off FP64 by 14 times the FP32 kernel's
// error (qsa_prompt_attn_parity, RTX 4070); per chunk, 2.8 times (Intel's kernel: 2.9), at 2.9 times the FP32
// kernel's speed.  K is FP16 or INT8, V FP16, INT8 or Q4_0 (--kv k8v4, as upstream's Volta kernel 06a90a2 takes
// it); Q4_0's codes enter exactly and its two scales of a 64-dim group are folded into two p' (dequantized to FP16
// at the gather, V was off FP64 by 360 times as much: qsa_prompt_attn_parity, RTX 4070 on sm_70 code).
namespace pm {
constexpr int SG = 32, NSG = NG, WG = NSG * SG, T16 = 16, CH = 16;
}
using xe::attn::u32x2;
using xe::attn::u32x4;
using xe::attn::i8x8_to_h8;
// joint_matrix's calls in the form PromptAttnMma makes them, beside amd_wmma.hpp's for RDNA3 and RDNA4
namespace jmx {
template<typename M, typename P> inline void load(sycl::sub_group sg, M& m, P p, size_t ld) { mx::joint_matrix_load(sg, m, p, ld); }
template<typename M> inline void fill(sycl::sub_group sg, M& m, float v) { mx::joint_matrix_fill(sg, m, v); }
template<typename D, typename A, typename B>
inline void mad(sycl::sub_group sg, D& d, A& a, B& b, D& c) { mx::joint_matrix_mad(sg, d, a, b, c); }
template<typename M, typename P>
inline void store(sycl::sub_group sg, M& m, P p, size_t ld) { mx::joint_matrix_store(sg, m, p, ld, mx::layout::row_major); }
}  // namespace jmx
template<int KF, int VF>   // K's KvFmt (kF16, kI8), V's (kF16, kI8, kQ4)
struct PromptAttnMma {
    static_assert(KF != kQ4, "Q4_0 K takes the FP32 kernel");
    using Pa = PromptAttn<kF16, kF16, pm::CH>;   // the helpers that do not depend on the pools
    static constexpr int VPG = VF == kQ4 ? 2 : 1;      // V scale groups a sub-group's 64 dims
    static constexpr int VS = NG * VPG;                // V scales a cell
    static constexpr int KV_HALVES = 2 * pm::CH * KR;   // FP16 K and V rows; q (hi + lo) shares them at the start
    static_assert(KV_HALVES >= 2 * R * QS, "q's staging must fit in the K/V rows");
    static_assert(R == pm::T16, "the 16 rows are one tile");
    static_assert(pm::CH * (HD / QK4_0) == pm::WG, "Q4_0 V: one work-item a block");
    const float* qv;
    QsaAttnPools p;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, n_kv, page_size;
    float scale_log2;
    float* attn;
    sycl::local_accessor<half, 1> kvq;      // kh [CH][KR], vh [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 1> ks, vs;  // [CH][NG], [CH][VS]: INT8 K's scales, V's
    sycl::local_accessor<float, 1> part;    // [NG][R][CH]: q.k per 64-dim group; part[0] then holds p
    sycl::local_accessor<half, 1> pp;       // [2 VS][R][CH]: p' hi (2v) / lo (2v + 1) per V scale group
    sycl::local_accessor<float, 1> st;      // [NG][R][16]: a sub-group's tile of p.v on its way to the registers
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

    void operator()(sycl::nd_item<2> it) const {
        // an empty body where there are no FP16 matrix engines of this shape (AMD before RDNA3, NVIDIA before sm_70):
        // mma_gemm does not take the 16 x 16 x 16 shape there, so this kernel is never launched.  RDNA3 and RDNA4 take
        // the tiles through their WMMA instructions (amd_wmma.hpp), the others through joint_matrix.
#if defined(__SYCL_DEVICE_ONLY__) && ((defined(__AMDGCN__) && !STRATA_AMD_WMMA) || (defined(__NVPTX__) && STRATA_NV_ARCH < 700))
        (void) it;
#else
#if STRATA_AMD_WMMA
        namespace J = xe::wmma;
        using MA = J::A<half>;
        using MBc = J::Bc<half>;
        using MBr = J::Br<half>;
        using MC = J::C;
#else
        namespace J = jmx;
        using MA = mx::joint_matrix<sycl::sub_group, half, mx::use::a, pm::T16, pm::T16, mx::layout::row_major>;
        using MBc = mx::joint_matrix<sycl::sub_group, half, mx::use::b, pm::T16, pm::T16, mx::layout::col_major>;
        using MBr = mx::joint_matrix<sycl::sub_group, half, mx::use::b, pm::T16, pm::T16, mx::layout::row_major>;
        using MC = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, pm::T16, pm::T16>;
#endif
        using pm::CH;
        using pm::T16;
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int64_t qi = (int64_t) it.get_group(0), kvh = (int64_t) it.get_group(1);
        const int t = (int) it.get_local_id(1), g = (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t n_head = n_kv * G;
        const float* qp = qv + qi * n_head * HD + kvh * G * HD;
        float* out = attn + qi * n_head * HD + kvh * G * HD;
        const int32_t* sel = ids + qi * cap;
        const int n = steps[qi * kStepCount + kStepWidth];
        const size_t KH = 0, VH = (size_t) CH * KR, QH = 0, QL = (size_t) R * QS;
        const size_t SG0 = (size_t) g * R * T16;   // this sub-group's output staging, a tile at a time

        // q as in PromptAttn: 12 heads + 4 zero rows, times 2^(14 - e), FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < G * HD; i += pm::WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * HD; i += pm::WG) {
            const int r = i / HD, d = i % HD;
            half hi, lo;
            Pa::split(r < G ? qp[r * HD + d] * qup : 0.0f, hi, lo);
            kvq[QH + (size_t) r * QS + d] = hi;
            kvq[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        // this sub-group's q: the 16 rows x 4 steps of 16 of its 64 dims, hi and lo
        MA qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) g * 64 + (size_t) kk * T16;
            J::load(sg, qa[kk], at(kvq, QH + off), QS);
            J::load(sg, qb[kk], at(kvq, QL + off), QS);
        }
        // output values a lane: element lane + SG * i (i < TPL) of each of the sub-group's four 16 x 16 tiles, in
        // o[nt * TPL + i]
        constexpr int TPL = R * T16 / pm::SG, OPL = 4 * TPL;
        float o[OPL];
        for (int i = 0; i < OPL; ++i) o[i] = 0.0f;

        for (int c0 = 0; c0 < n; c0 += CH) {
            const int nh = sycl::min(CH, n - c0);
            if (t < CH) {
                long long r = -1;
                if (t < nh) {
                    const int cell = sel[c0 + t];
                    const long long page = (long long) p.page_table[cell / page_size];
                    if (page >= 0) r = (page * n_kv + kvh) * page_size + cell % page_size;
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);   // rows ready; the previous chunk (and q's staging) is done with kvq, part, pp
            // K and V as FP16: int8 codes are exact in FP16.  8 codes or halves a work-item in one access each way:
            // through sycl::vec the NVPTX compile split the int8 loads into a byte each (LDG.E.S8; Nsight Compute,
            // RTX 4070: 87% of the kernel's global sectors excessive)
            for (int i = t; i < CH * (HD / 8); i += pm::WG) {
                const int c = i / (HD / 8), pc = i % (HD / 8);
                const long long r = rows[c];
                u32x4 kx = {0, 0, 0, 0}, vx = {0, 0, 0, 0};
                if (r >= 0) {
                    const int64_t off = r * HD + (int64_t) pc * 8;
                    if constexpr (KF == kI8)
                        kx = i8x8_to_h8(*reinterpret_cast<const u32x2*>(p.k_q + off));
                    else
                        kx = *reinterpret_cast<const u32x4*>(p.k_pool + off);
                    if constexpr (VF == kI8)
                        vx = i8x8_to_h8(*reinterpret_cast<const u32x2*>(p.v_q + off));
                    else if constexpr (VF == kF16)
                        vx = *reinterpret_cast<const u32x4*>(p.v_pool + off);
                }
                *reinterpret_cast<u32x4*>(&kvq[KH + (size_t) c * KR + (size_t) pc * 8]) = kx;
                if constexpr (VF != kQ4)
                    *reinterpret_cast<u32x4*>(&kvq[VH + (size_t) c * KR + (size_t) pc * 8]) = vx;
            }
            // Q4_0 V: one work-item a block, its codes and its scale
            if constexpr (VF == kQ4) {
                const int c = t / (HD / QK4_0), b = t % (HD / QK4_0);
                const long long r = rows[c];
                half* vd = &kvq[VH + (size_t) c * KR + (size_t) b * QK4_0];
                float a = 0.0f;
                if (r >= 0) a = Pa::q4_block(p.v_q4 + r * Pa::Q4ROW, b, vd);
                else for (int j = 0; j < QK4_0; ++j) vd[j] = half(0.0f);
                vs[t] = a;
            }
            if constexpr (KF == kI8 || VF == kI8)
                if (t < CH * NG) {
                    const int c = t / NG, gg = t % NG;
                    const long long r = rows[c];
                    const int64_t si = r * (HD / KV_Q8_GROUP) + gg;
                    if constexpr (KF == kI8) ks[t] = r >= 0 ? f32_from_f16(p.k_scale[si]) : 0.0f;
                    if constexpr (VF == kI8) vs[t] = r >= 0 ? f32_from_f16(p.v_scale[si]) : 0.0f;
                }
            sycl::group_barrier(grp);

            // q.k of the 16 rows and this sub-group's 64 dims, hi and lo
            {
                MC c;
                J::fill(sg, c, 0.0f);
                for (int kk = 0; kk < 4; ++kk) {
                    MBc b;
                    J::load(sg, b, at(kvq, KH + (size_t) g * 64 + (size_t) kk * T16), KR);
                    J::mad(sg, c, qa[kk], b, c);
                    J::mad(sg, c, qb[kk], b, c);
                }
                J::store(sg, c, at(part, (size_t) g * R * CH), CH);
            }
            sycl::group_barrier(grp);

            // online softmax as in PromptAttn: row t / 8, CH / 8 cells each, 8 neighbouring lanes a row; the groups'
            // partials in order, each times its K scale
            {
                constexpr int PER = CH / 8;
                const int r = t / 8, sub = t % 8;
                float x[PER], mxv = NEG_INF;
                for (int j = 0; j < PER; ++j) {
                    const int c = sub * PER + j;
                    const auto pt = [&](int gg) { return part[((size_t) gg * R + r) * CH + c]; };
                    float sc;
                    if constexpr (KF == kI8)
                        sc = ((pt(0) * ks[c * NG + 0] + pt(1) * ks[c * NG + 1]) + pt(2) * ks[c * NG + 2]) +
                             pt(3) * ks[c * NG + 3];
                    else
                        sc = ((pt(0) + pt(1)) + pt(2)) + pt(3);
                    x[j] = c < nh && rows[c] >= 0 ? sc * qdown : NEG_INF;
                    mxv = sycl::fmax(mxv, x[j]);
                }
                for (int o = 1; o < 8; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = mrow[r];
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                float sum = 0.0f;
                for (int j = 0; j < PER; ++j) {
                    const float e = x[j] == NEG_INF ? 0.0f : sycl::exp2(x[j] - m_new);
                    part[(size_t) r * CH + (size_t) (sub * PER + j)] = e;   // this work-item's own cells
                    sum += e;
                }
                for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                if (sub == 0) {
                    const float a = m_old == m_new ? 1.0f : m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    alpha[r] = a;
                    lsum[r] = sycl::fma(lsum[r], a, sum);
                    mrow[r] = m_new;
                }
            }
            sycl::group_barrier(grp);

            // p.v of the 16 rows and this sub-group's 64 dims: p' = p * vscale / unit, hi + lo, one p' per V scale
            // group (two of 32 dims for Q4_0, whose scales are signed), with the chunk's unit its largest |V scale|
            // of the 64 dims / PSCALE (FP16 V: 1 / PSCALE, a power of two)
            {
                float unit = 1.0f / PSCALE, vup = PSCALE;
                if constexpr (VF != kF16) {
                    float vm = 0.0f;
                    if (lane < CH)
                        for (int vp = 0; vp < VPG; ++vp) vm = sycl::fmax(vm, sycl::fabs(vs[lane * VS + g * VPG + vp]));
                    vm = sycl::reduce_over_group(sg, vm, sycl::maximum<float>());
                    unit = vm * (1.0f / PSCALE);
                    vup = vm > 0.0f ? PSCALE / vm : 0.0f;
                }
                for (int vp = 0; vp < VPG; ++vp) {
                    const int vg = g * VPG + vp;
                    for (int e = lane; e < R * CH; e += pm::SG) {
                        const float w = VF == kF16 ? vup : vs[(e % CH) * VS + vg] * vup;
                        half hi, lo;
                        Pa::split(part[e] * w, hi, lo);
                        pp[(size_t) (2 * vg) * R * CH + e] = hi;
                        pp[(size_t) (2 * vg + 1) * R * CH + e] = lo;
                    }
                }
                sycl::group_barrier(sg);
                MA ph, pl;
                for (int nt = 0; nt < 4; ++nt) {
                    if (nt % (4 / VPG) == 0) {   // the V scale group of these 16 dims
                        const int vg = g * VPG + nt / (4 / VPG);
                        J::load(sg, ph, at(pp, (size_t) (2 * vg) * R * CH), CH);
                        J::load(sg, pl, at(pp, (size_t) (2 * vg + 1) * R * CH), CH);
                    }
                    MC acc;
                    J::fill(sg, acc, 0.0f);
                    MBr b;
                    J::load(sg, b, at(kvq, VH + (size_t) g * 64 + (size_t) nt * T16), KR);
                    J::mad(sg, acc, ph, b, acc);
                    J::mad(sg, acc, pl, b, acc);
                    // through local memory a tile at a time (a quarter of the 16 x 64 staging: the work-group's local
                    // memory, and so how many fit on a compute unit, set this kernel's speed on the RTX 4070)
                    sycl::group_barrier(sg);   // the previous tile's values are read
                    J::store(sg, acc, at(st, SG0), T16);
                    sycl::group_barrier(sg);
                    for (int i = 0; i < TPL; ++i) {
                        const int e = lane + pm::SG * i;
                        o[nt * TPL + i] = sycl::fma(o[nt * TPL + i], alpha[e / T16], st[SG0 + e] * unit);
                    }
                }
            }
        }
        for (int i = 0; i < OPL; ++i) {
            const int e = lane + pm::SG * (i % TPL), r = e / T16, col = (i / TPL) * T16 + e % T16;
            if (r < G) {
                const float l = lsum[r];
                out[r * HD + g * 64 + col] = l > 0.0f ? o[i] / l : 0.0f;
            }
        }
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(pm::SG)>}; }
};

// the work-group's local memory, as launch_mma allocates it (the most, Q4_0 V's)
constexpr size_t mma_local_bytes() {
    constexpr size_t VS = PromptAttnMma<kI8, kQ4>::VS;
    return (size_t) PromptAttnMma<kF16, kF16>::KV_HALVES * 2 + (size_t) pm::CH * 8 + (size_t) pm::CH * (NG + VS) * 4 +
           (size_t) NG * R * pm::CH * 4 + (size_t) 2 * VS * R * pm::CH * 2 + (size_t) NG * R * pm::T16 * 4 +
           (size_t) 3 * R * 4;
}

template<int KF, int VF>
sycl::event launch_mma(sycl::queue& q, const float* qv, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                       int64_t cap, int64_t n_kv, int64_t page_size, float* attn, int64_t n_q) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    return q.submit([&](sycl::handler& h) {
        using K = PromptAttnMma<KF, VF>;
        K k{qv, p, ids, steps, cap, n_kv, page_size, scale_log2, attn,
            sycl::local_accessor<half, 1>(sycl::range<1>(K::KV_HALVES), h),
            sycl::local_accessor<long long, 1>(sycl::range<1>(pm::CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) pm::CH * NG), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) pm::CH * K::VS), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) NG * R * pm::CH), h),
            sycl::local_accessor<half, 1>(sycl::range<1>((size_t) 2 * K::VS * R * pm::CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) NG * R * pm::T16), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_q, (size_t) (n_kv * pm::WG)}, {1, pm::WG}), k);
    });
}

// NVIDIA from sm_80 on, INT8 K and V: upstream's v2 (src/kernels/cuda/qsa_prompt_attn.cu, prompt_attn_i8_kernel) in
// PTX, as mma_gemm.cpp writes its tiles.  Warp w owns dims [64w, 64w + 64), which is int8 scale group w, for both q.k
// and p.v: it gathers its own 64-byte slice of each K and V row with cp.async into a double-buffered stage while it
// computes the previous chunk, q stays in registers as mma.sync fragments, int8 codes become FP16 fragments in
// registers, and the output stays in FP32 accumulators whose layout mma.sync defines.  Only the q.k partial sums cross
// warps, added in a fixed order.  joint_matrix could not express the fragments or the copies: PromptAttnMma ran a
// chunk of 32K at 14.0 ms, this at 7.4 (qsa_prompt_attn_parity, RTX 4070).  STRATA_PROMPT_ATTN_PTX=0: PromptAttnMma.
#define STRATA_PA_PTX STRATA_NV_MMA
namespace pv2 {
constexpr int CH = 32;
struct Smem {
    int8_t kv[2][4][2][CH][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH + 1];    // q.k per dim group
    float p[16][CH + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};
// [[maybe_unused]]: these serve PromptAttnPtx's body, which only the sm_80+ device compile has
[[maybe_unused]] inline int swz(int cell, int byte) {   // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
#if STRATA_PA_PTX
namespace nv = xe::nv;   // nv_mma.hpp
using nv::cp_async_cg;
using nv::cp_async_commit;
#endif
// two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152
[[maybe_unused]] inline uint32_t i8x2_to_h2(uint32_t x) {
    const uint32_t y = ((((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u);
    return sycl::bit_cast<uint32_t>(sycl::bit_cast<sycl::vec<half, 2>>(y) - sycl::vec<half, 2>(half(1152.0f)));
}
[[maybe_unused]] inline uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    return sycl::bit_cast<uint32_t>(sycl::vec<half, 2>(half(lo_k), half(hi_k)));
}
}  // namespace pv2

struct PromptAttnPtx {
    [[maybe_unused]] static constexpr int CH = pv2::CH, THREADS = pm::WG;
    const float* q;
    QsaAttnPools p;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, n_kv, page_size;
    float scale_log2;
    float* attn;
    sycl::local_accessor<uint8_t, 1> smem;   // pv2::Smem

    void operator()(sycl::nd_item<2> it) const {
#if !STRATA_PA_PTX
        (void) it;
#else
        using pv2::swz;
        // through a pointer that keeps the local address space: from a plain one the accesses are generic (LD.E / ST.E,
        // not LDS / STS)
        using LocalSmem = __attribute__((opencl_local)) pv2::Smem;
        LocalSmem& S = *reinterpret_cast<LocalSmem*>(smem.template get_multi_ptr<sycl::access::decorated::yes>().get_decorated());
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int qi = (int) it.get_group(0), kvh = (int) it.get_group(1);
        const int n_head = (int) n_kv * G;
        const float* qp = q + (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
        float* out = attn + (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
        const int32_t* sel = ids + (size_t) qi * cap;
        const int n = steps[(size_t) qi * kStepCount + kStepWidth];
        const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
        const int gid = lane >> 2, tig = lane & 3;
        const int dim0 = warp * 64;

        // q: the power-of-two prescale over all 12 heads, then this warp's 64 dims as hi/lo A fragments
        float qm = 0.0f;
        for (int i = t; i < G * HD; i += THREADS) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) qm = sycl::fmax(qm, sycl::permute_group_by_xor(sg, qm, o));
        if (lane == 0) S.qmax[warp] = qm;
        if (t < 16) { S.mrow[t] = NEG_INF; S.lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]), sycl::fmax(S.qmax[2], S.qmax[3]));
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        uint32_t qh[4][4], ql[4][4];
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
                float x0 = 0.0f, x1 = 0.0f;
                if (row < G) { x0 = qp[(size_t) row * HD + col] * qup; x1 = qp[(size_t) row * HD + col + 1] * qup; }
                const sycl::vec<half, 2> hi{half(x0), half(x1)};
                qh[kk][r] = sycl::bit_cast<uint32_t>(hi);
                ql[kk][r] = pv2::pack_h2(x0 - (float) hi[0], x1 - (float) hi[1]);
            }

        // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
        const int n_chunks = (n + CH - 1) / CH;
        auto cell_of = [&](int c) -> int { return c < n ? sel[c] : -1; };
        auto row_of = [&](int cell) -> long long {
            if (cell < 0) return -1;
            const long long page = (long long) p.page_table[cell / page_size];
            return (page * n_kv + kvh) * page_size + (cell % page_size);
        };
        auto issue = [&](long long r, int st, float& ksr, float& vsr) {   // cp.async takes generic addresses
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
                const long long rr = sycl::select_from_group(sg, r, cell);
                const bool ok = rr >= 0;
                const size_t off = ok ? (size_t) rr * HD + (size_t) dim0 + (size_t) pc * 16 : 0;
                pv2::cp_async_cg((int8_t*) &S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
                pv2::cp_async_cg((int8_t*) &S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
            }
            ksr = r >= 0 ? f32_from_f16(p.k_scale[r * (HD / KV_Q8_GROUP) + warp]) : 0.0f;
            vsr = r >= 0 ? f32_from_f16(p.v_scale[r * (HD / KV_Q8_GROUP) + warp]) : 0.0f;
        };
        float ksn, vsn;
        issue(row_of(cell_of(lane)), 0, ksn, vsn);
        pv2::cp_async_commit();
        S.sc[0][warp][0][lane] = ksn;
        S.sc[0][warp][1][lane] = vsn;
        long long r_next = row_of(cell_of(CH + lane));
        int cell_next2 = cell_of(2 * CH + lane);

        float acc[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

        for (int ci = 0; ci < n_chunks; ++ci) {
            const int st = ci & 1, c0 = ci * CH;
            const bool more = ci + 1 < n_chunks;
            if (more) issue(r_next, st ^ 1, ksn, vsn);
            pv2::cp_async_commit();
            r_next = row_of(cell_next2);
            cell_next2 = cell_of((ci + 3) * CH + lane);
            pv2::nv::cp_async_wait<1>();
            sycl::group_barrier(sg);
            const __attribute__((opencl_local)) int8_t* K = &S.kv[st][warp][0][0][0];
            const __attribute__((opencl_local)) int8_t* V = &S.kv[st][warp][1][0][0];
            // q.k over this warp's 64 dims, times the cell's K scale for this group
            #pragma unroll
            for (int nt = 0; nt < CH / 8; ++nt) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
                const int cell = nt * 8 + gid;
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    uint32_t b[2];
                    b[0] = pv2::i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                    b[1] = pv2::i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                    pv2::nv::mma_f16<false>(tg, qh[kk], b[0], b[1]);
                    pv2::nv::mma_f16<false>(tg, ql[kk], b[0], b[1]);
                }
                const int c = nt * 8 + 2 * tig;
                const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
                S.part[warp][gid][c] = tg[0] * s0;
                S.part[warp][gid][c + 1] = tg[1] * s1;
                S.part[warp][gid + 8][c] = tg[2] * s0;
                S.part[warp][gid + 8][c + 1] = tg[3] * s1;
            }
            sycl::group_barrier(grp);
            // online softmax over the four groups' sum (fixed order): row t / 8, 4 cells per thread
            {
                const int r = t >> 3, sub = t & 7;
                float x[4], mxv = NEG_INF;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = sub * 4 + j;
                    x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                      : NEG_INF;
                    mxv = sycl::fmax(mxv, x[j]);
                }
                #pragma unroll
                for (int o = 1; o < 8; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = S.mrow[r];
                const float m_new = sycl::fmax(m_old, mxv);
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float e = x[j] == NEG_INF ? 0.0f : sycl::exp2(x[j] - m_new);
                    S.p[r][sub * 4 + j] = e;
                    sum += e;
                }
                #pragma unroll
                for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                sycl::group_barrier(sg);
                if (sub == 0) {
                    const float a = m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    S.alpha[r] = a;
                    S.lsum[r] = sycl::fma(S.lsum[r], a, sum);
                    S.mrow[r] = m_new;
                }
            }
            sycl::group_barrier(grp);
            // p.v over this warp's 64 dims
            {
                float vmax = S.sc[st][warp][1][lane];
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) vmax = sycl::fmax(vmax, sycl::permute_group_by_xor(sg, vmax, o));
                const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
                float tmp[8][4];
                #pragma unroll
                for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
                #pragma unroll
                for (int ks = 0; ks < CH / 16; ++ks) {
                    const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                    const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                                w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                    const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                    const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                    const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                    const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                    uint32_t ah[4], al[4];
                    ah[0] = pv2::pack_h2(p00, p01);
                    ah[1] = pv2::pack_h2(p10, p11);
                    ah[2] = pv2::pack_h2(p02, p03);
                    ah[3] = pv2::pack_h2(p12, p13);
                    {
                        const auto f = [](uint32_t h, int k) { return (float) sycl::bit_cast<sycl::vec<half, 2>>(h)[k]; };
                        al[0] = pv2::pack_h2(p00 - f(ah[0], 0), p01 - f(ah[0], 1));
                        al[1] = pv2::pack_h2(p10 - f(ah[1], 0), p11 - f(ah[1], 1));
                        al[2] = pv2::pack_h2(p02 - f(ah[2], 0), p03 - f(ah[2], 1));
                        al[3] = pv2::pack_h2(p12 - f(ah[3], 0), p13 - f(ah[3], 1));
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int d = j * 8 + gid;
                        const uint32_t x0 = (uint32_t) (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                        const uint32_t x1 = (uint32_t) (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                        uint32_t b[2];
                        b[0] = pv2::i8x2_to_h2(x0);
                        b[1] = pv2::i8x2_to_h2(x1);
                        pv2::nv::mma_f16<false>(tmp[j], ah, b[0], b[1]);
                        pv2::nv::mma_f16<false>(tmp[j], al, b[0], b[1]);
                    }
                }
                const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[j][0] = sycl::fma(acc[j][0], a0, tmp[j][0] * vdown);
                    acc[j][1] = sycl::fma(acc[j][1], a0, tmp[j][1] * vdown);
                    acc[j][2] = sycl::fma(acc[j][2], a1, tmp[j][2] * vdown);
                    acc[j][3] = sycl::fma(acc[j][3], a1, tmp[j][3] * vdown);
                }
            }
            if (more) {
                S.sc[st ^ 1][warp][0][lane] = ksn;
                S.sc[st ^ 1][warp][1][lane] = vsn;
            }
            sycl::group_barrier(sg);
        }
        sycl::group_barrier(grp);
        const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
        const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = dim0 + j * 8 + 2 * tig;
            out[(size_t) gid * HD + d] = acc[j][0] * i0;
            out[(size_t) gid * HD + d + 1] = acc[j][1] * i0;
            if (gid + 8 < G) {
                out[(size_t) (gid + 8) * HD + d] = acc[j][2] * i1;
                out[(size_t) (gid + 8) * HD + d + 1] = acc[j][3] * i1;
            }
        }
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(pm::SG)>}; }
};

// Whether this device's compile carries PromptAttnPtx's body (NVIDIA from sm_80 on): asked of the device once.
bool ptx_attn_built(sycl::queue& queue) {
    static core::PerDevice<bool> per_device;
    return per_device.get(queue.get_device(), [&queue] {
        int* flag = sycl::malloc_device<int>(1, queue);
        if (flag == nullptr) return false;
        int got = 0;
        try {
            queue.single_task([=] { *flag = STRATA_PA_PTX; });
            queue.memcpy(&got, flag, sizeof(int)).wait();
        } catch (const sycl::exception&) { got = 0; }
        sycl::free(flag, queue);
        return got == 1;
    });
}

sycl::event launch_ptx(sycl::queue& q, const float* qv, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                       int64_t cap, int64_t n_kv, int64_t page_size, float* attn, int64_t n_q) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    return q.submit([&](sycl::handler& h) {
        PromptAttnPtx k{qv, p, ids, steps, cap, n_kv, page_size, scale_log2, attn,
                        sycl::local_accessor<uint8_t, 1>(sycl::range<1>(sizeof(pv2::Smem)), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_q, (size_t) (n_kv * pm::WG)}, {1, pm::WG}), k);
    });
}

// 16 cells a chunk: 32 kept twice the local memory and ran at half the speed (B70, also with the large register file)
constexpr int CHUNK = 16;
static_assert(CHUNK == 16, "the p split gives each lane of a 16-wide sub-group one cell");


bool batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    // the matrix engines and the work-group's local memory (26 KiB, 34 KiB with Q4_0 K and V); without them the
    // caller keeps the FP32 kernel
    auto& queue = core::Runtime::get().stream(stream);
    const size_t slm = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    if (!xmx_available(XmxType::f16)) {
        // the portable kernel: FP16, INT8 and K8V4 pools (Q4_0 K keeps the FP32 kernel)
        static core::PerDevice<bool> per_device;
        const bool mma = per_device.get(queue.get_device(), [&queue, slm] {
            return xe::mma_shape_is(queue, 16, 16, 16, 32) && slm >= mma_local_bytes();
        });
        if (!mma || pools.k_q4 != nullptr) return false;
        static const bool ptx_on = [] {
            const char* v = strata::env_get("PROMPT_ATTN_PTX");
            return v == nullptr || v[0] != '0';
        }();
        const int64_t nkv = s.n_head_kv, ps = s.page_size;
        sycl::event e;
        if (pools.k_q != nullptr) {
            if (!pools.k_scale) return false;
            if (pools.v_q4 != nullptr)
                e = launch_mma<kI8, kQ4>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
            else if (pools.v_q && pools.v_scale && ptx_on && slm >= sizeof(pv2::Smem) && ptx_attn_built(queue))
                e = launch_ptx(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
            else if (pools.v_q && pools.v_scale)
                e = launch_mma<kI8, kI8>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
            else
                return false;
        } else {
            if (!pools.k_pool || !pools.v_pool) return false;
            e = launch_mma<kF16, kF16>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
        }
        if (!stream) core::Runtime::get().wait(e, "flash_attn_select");
        return true;
    }
    // STRATA_PROMPT_ATTN_Q4=0: Q4_0 K or V (--kv q4_0, --kv k8v4) to the FP32 kernel, as before upstream 778e1f6 (A/B)
    static const bool q4_on = [] {
        const char* v = strata::env_get("PROMPT_ATTN_Q4");
        return v == nullptr || v[0] != '0';
    }();
    const int64_t nkv = s.n_head_kv, ps = s.page_size;
    sycl::event e;
    if (pools.k_q4 != nullptr) {   // --kv q4_0
        if (!q4_on || !pools.v_q4 || slm < local_bytes<kQ4, kQ4, CHUNK>()) return false;
        e = launch<kQ4, kQ4, CHUNK>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    } else if (pools.k_q != nullptr) {
        // --kv k8v4 (INT8 K, Q4_0 V) stays on the FP32 kernel: this one ran it at 0.84x (B70, 32K cells, 2,048 queries;
        // Q4_0 V doubles the p' planes of each chunk)
        if (pools.v_q4 != nullptr) return false;
        if (!pools.v_q || !pools.k_scale || !pools.v_scale || slm < local_bytes<kI8, kI8, CHUNK>()) return false;
        e = launch<kI8, kI8, CHUNK>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    } else {
        if (!pools.k_pool || !pools.v_pool || slm < local_bytes<kF16, kF16, CHUNK>()) return false;
        e = launch<kF16, kF16, CHUNK>(queue, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    }
    if (!stream) core::Runtime::get().wait(e, "flash_attn_select");
    return true;
}

bool run(const FlashAttnArgs& a, void* stream) {
    if (!a.out_f32 || a.q32 == nullptr || a.first != 0 || a.row_end != a.rows || a.out_ld != a.n_head * a.head_dim ||
        a.q_ld != a.n_head * a.head_dim)
        return false;
    QsaShapes s = qsa_real_shapes();
    s.n_head = a.n_head;
    s.n_head_kv = a.n_kv;
    s.head_dim = a.head_dim;
    s.page_size = a.page_size;
    return batch(a.q32, a.pools, a.sel_ids, a.sel_steps, a.sel_cap, s, static_cast<float*>(a.out), a.rows, stream);
}

}  // namespace sel

}  // namespace

int flash_attn_path(const void* sycl_device) { return path_of(*static_cast<const sycl::device*>(sycl_device)); }

int flash_attn_nsg_most(const void* sycl_device, int64_t head_dim) {
    const sycl::device& d = *static_cast<const sycl::device*>(sycl_device);
    const int p = path_of(d);
    return head_dim == 128 ? most_of<128>(d, p) : head_dim == 256 ? most_of<256>(d, p) : 0;
}

bool flash_attn(const FlashAttnArgs& a, void* stream, int nsg) {
    if (a.row_end <= a.first) return true;
    if (a.sel_ids != nullptr) return a.sel_steps != nullptr && a.sel_cap > 0 && sel::run(a, stream);
    if (a.n_kv <= 0 || a.n_head % a.n_kv != 0) return false;
    if (a.page_table != nullptr && a.page_size <= 0) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    const int p = path_of(q.get_device());
    if (p < 0 || a.fma) return run_fma(q, a);
    if (a.pools_format < 0) {   // the row form reads a chunk's keys of a page at one stride (RDNA's: any page size)
        if (a.page_table != nullptr && p != 3 && a.page_size % 64 != 0) return false;
    } else if (a.pools.page_table == nullptr || a.page_size <= 0 || (a.q32 == nullptr && a.q == nullptr)) {
        return false;
    }
    // the Q4_0 forms for heads of 256 only (the Hadamard rotation's width)
    if ((a.pools_format == kFmtQ4 || a.pools_format == kFmtK8V4) && a.head_dim != 256) return false;
    if (a.head_dim == 128) return run_path<128>(q, a, nsg, p);
    if (a.head_dim == 256) return run_path<256>(q, a, nsg, p);
    return false;
}

bool flash_attn_select(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                       const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    FlashAttnArgs a;
    a.q32 = q;
    a.q_ld = a.out_ld = s.n_head * s.head_dim;
    a.pools = pools;
    a.page_size = s.page_size;
    a.n_head = s.n_head;
    a.n_kv = s.n_head_kv;
    a.head_dim = s.head_dim;
    a.rows = a.row_end = n_q;
    a.sel_ids = ids;
    a.sel_steps = steps;
    a.sel_cap = cap;
    a.out = attn;
    a.out_f32 = true;
    a.scale = 1.0f / std::sqrt((float) s.head_dim);
    return flash_attn(a, stream);
}

}  // namespace strata::kernels
