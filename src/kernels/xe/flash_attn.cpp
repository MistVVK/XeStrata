// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/flash_attn.cpp - see include/strata/kernels/flash_attn.hpp.
//
// A sub-group takes TM rows (query, head) and walks their keys in chunks of CH:
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

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
                                                                sycl::access::decorated::no>(qh + kk * TK), D);
            sycl::group_barrier(sg);
        } else {
            for (int kk = 0; kk < D / TK; ++kk)
                Ops::load(sg, qa[kk], a.q + my0 * a.q_ld + grp0 * D + (int64_t) kk * TK, a.q_ld);
        }
        typename Ops::C o[NT];
        for (int nt = 0; nt < NT; ++nt) Ops::zero(sg, o[nt]);
        const int r = lane / L, part = lane % L;               // the softmax's row and its lanes' share
        const int64_t lim = my0 + r < row_hi ? lim_of(query_of(my0 + r)) : 0;
        // the keys walked: the sub-group's last row's (the limits never fall); gathered, the work-group's
        const int64_t last = GATHER ? std::min(row_hi, wg0 + (int64_t) nsg * TM) - 1 : std::min(a.rows, my0 + TM) - 1;
        const int64_t keys = lim_of(query_of(last));
        sycl::group_barrier(sg);
        for (int64_t j0 = 0; j0 < keys; j0 += CH) {
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
                    s[i + u] = j0 + c0 + i + u < lim ? v[u] * a.scale : -INFINITY;
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

template<int D>
bool run_path(sycl::queue& q, const FlashAttnArgs& a, int nsg, int p) {
    switch (p) {
    case 0: return run_fmt<0, D>(q, a, nsg);
    case 1: return run_fmt<1, D>(q, a, nsg);
    case 2: return run_fmt<2, D>(q, a, nsg);
    case 3: return run_fmt<3, D>(q, a, nsg);
    default: return false;
    }
}

template<int D>
int most_of(const sycl::device& d, int p) {
    switch (p) {
    case 0: return nsg_most_of<0, D, -1>(d);
    case 1: return nsg_most_of<1, D, -1>(d);
    case 2: return nsg_most_of<2, D, -1>(d);
    case 3: return nsg_most_of<3, D, -1>(d);
    default: return 0;
    }
}

}  // namespace

int flash_attn_path(const void* sycl_device) { return path_of(*static_cast<const sycl::device*>(sycl_device)); }

int flash_attn_nsg_most(const void* sycl_device, int64_t head_dim) {
    const sycl::device& d = *static_cast<const sycl::device*>(sycl_device);
    const int p = path_of(d);
    return head_dim == 128 ? most_of<128>(d, p) : head_dim == 256 ? most_of<256>(d, p) : 0;
}

bool flash_attn(const FlashAttnArgs& a, void* stream, int nsg) {
    if (a.row_end <= a.first) return true;
    if (a.n_kv <= 0 || a.n_head % a.n_kv != 0) return false;
    if (a.pools_format < 0) {
        if (a.page_table != nullptr && (a.page_size <= 0 || a.page_size % 64 != 0)) return false;
    } else if (a.pools.page_table == nullptr || a.page_size <= 0 || (a.q32 == nullptr && a.q == nullptr)) {
        return false;
    }
    // the Q4_0 forms for heads of 256 only (the Hadamard rotation's width)
    if ((a.pools_format == kFmtQ4 || a.pools_format == kFmtK8V4) && a.head_dim != 256) return false;
    sycl::queue& q = core::Runtime::get().stream(stream);
    const int p = path_of(q.get_device());
    if (a.head_dim == 128) return run_path<128>(q, a, nsg, p);
    if (a.head_dim == 256) return run_path<256>(q, a, nsg, p);
    return false;
}

}  // namespace strata::kernels
