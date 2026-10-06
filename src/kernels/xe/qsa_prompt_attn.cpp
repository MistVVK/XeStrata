// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/qsa_prompt_attn.cpp - the prompt attention entry point (qsa_prompt_attn.hpp) on Xe's matrix engines.
//
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
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/per_device.hpp"
#include "device_target.hpp"
#include "mma_gemm.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

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

// a local accessor's element `off` (row-major) as the multi_ptr joint_matrix loads from and stores to
template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

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
// (NVIDIA's tensor cores; xe::mma_shape_is), FP16 pools.  The scheme above with the accumulators' layout unknown:
// 4 sub-groups of 32, sub-group g owns dim group g for all 16 rows (one 16-row tile).  The output is not kept in the
// accumulators: each chunk's p.v starts from zero and goes through local memory into FP32 registers (a lane 32 of
// the sub-group's 16 x 64), scaled there by the softmax's correction.  Accumulated over the whole selection in the
// tensor cores, which align and cut the addends instead of rounding them, the output was off FP64 by 14 times the
// FP32 kernel's error (qsa_prompt_attn_parity, RTX 4070); per chunk, 2.8 times (Intel's kernel: 2.9), at 2.9 times the FP32
// kernel's speed.
namespace pm {
constexpr int SG = 32, NSG = NG, WG = NSG * SG, T16 = 16, CH = 16;
}
struct PromptAttnMma {
    static constexpr int KV_HALVES = 2 * pm::CH * KR;   // FP16 K and V rows; q (hi + lo) shares them at the start
    static_assert(KV_HALVES >= 2 * R * QS, "q's staging must fit in the K/V rows");
    static_assert(R == pm::T16, "the 16 rows are one tile");
    const float* qv;
    QsaAttnPools p;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, n_kv, page_size;
    float scale_log2;
    float* attn;
    sycl::local_accessor<half, 1> kvq;      // kh [CH][KR], vh [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 1> part;    // [NG][R][CH]: q.k per 64-dim group; part[0] then holds p
    sycl::local_accessor<half, 1> pp;       // [2 NG][R][CH]: p' hi (2g) / lo (2g + 1) per dim group
    sycl::local_accessor<float, 1> st;      // [NG][R][64]: a sub-group's chunk of p.v on its way to the registers
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

    void operator()(sycl::nd_item<2> it) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
        (void) it;
#else
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
        const size_t SG0 = (size_t) g * R * 64;   // this sub-group's output staging

        // q as in PromptAttn: 12 heads + 4 zero rows, times 2^(14 - e), FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < G * HD; i += pm::WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * HD; i += pm::WG) {
            const int r = i / HD, d = i % HD;
            half hi, lo;
            PromptAttn<kF16, kF16, CH>::split(r < G ? qp[r * HD + d] * qup : 0.0f, hi, lo);
            kvq[QH + (size_t) r * QS + d] = hi;
            kvq[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        // this sub-group's q: the 16 rows x 4 steps of 16 of its 64 dims, hi and lo
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, T16, T16, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) g * 64 + (size_t) kk * T16;
            mx::joint_matrix_load(sg, qa[kk], at(kvq, QH + off), QS);
            mx::joint_matrix_load(sg, qb[kk], at(kvq, QL + off), QS);
        }
        constexpr int OPL = R * 64 / pm::SG;   // output values a lane: element lane + SG * i of the sub-group's 16 x 64
        float o[OPL];
        for (int i = 0; i < OPL; ++i) o[i] = 0.0f;
        // FP16 pools: every V scale is 1, so the unit is 1 / PSCALE from the first chunk on
        const float unit = 1.0f / PSCALE, vup = PSCALE;

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
            for (int i = t; i < CH * (HD / 8); i += pm::WG) {
                const int c = i / (HD / 8), pc = i % (HD / 8);
                const long long r = rows[c];
                sycl::vec<half, 8> kx(half(0.0f)), vx(half(0.0f));
                if (r >= 0) {
                    const int64_t off = r * HD + (int64_t) pc * 8;
                    kx = *reinterpret_cast<const sycl::vec<half, 8>*>(p.k_pool + off);
                    vx = *reinterpret_cast<const sycl::vec<half, 8>*>(p.v_pool + off);
                }
                *reinterpret_cast<sycl::vec<half, 8>*>(&kvq[KH + (size_t) c * KR + (size_t) pc * 8]) = kx;
                *reinterpret_cast<sycl::vec<half, 8>*>(&kvq[VH + (size_t) c * KR + (size_t) pc * 8]) = vx;
            }
            sycl::group_barrier(grp);

            // q.k of the 16 rows and this sub-group's 64 dims, hi and lo
            {
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, T16, T16> c;
                mx::joint_matrix_fill(sg, c, 0.0f);
                for (int kk = 0; kk < 4; ++kk) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, T16, T16, mx::layout::col_major> b;
                    mx::joint_matrix_load(sg, b, at(kvq, KH + (size_t) g * 64 + (size_t) kk * T16), KR);
                    mx::joint_matrix_mad(sg, c, qa[kk], b, c);
                    mx::joint_matrix_mad(sg, c, qb[kk], b, c);
                }
                mx::joint_matrix_store(sg, c, at(part, (size_t) g * R * CH), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);

            // online softmax as in PromptAttn: row t / 8, CH / 8 cells each, 8 neighbouring lanes a row
            {
                constexpr int PER = CH / 8;
                const int r = t / 8, sub = t % 8;
                float x[PER], mxv = NEG_INF;
                for (int j = 0; j < PER; ++j) {
                    const int c = sub * PER + j;
                    const float sc = ((part[(0 * R + r) * CH + c] + part[(1 * R + r) * CH + c]) +
                                      part[(2 * R + r) * CH + c]) + part[(3 * R + r) * CH + c];
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

            // p.v of the 16 rows and this sub-group's 64 dims: p' = p * PSCALE, hi + lo
            {
                for (int e = lane; e < R * CH; e += pm::SG) {
                    half hi, lo;
                    PromptAttn<kF16, kF16, CH>::split(part[e] * vup, hi, lo);
                    pp[(size_t) (2 * g) * R * CH + e] = hi;
                    pp[(size_t) (2 * g + 1) * R * CH + e] = lo;
                }
                sycl::group_barrier(sg);
                mx::joint_matrix<sycl::sub_group, half, mx::use::a, T16, T16, mx::layout::row_major> ph, pl;
                mx::joint_matrix_load(sg, ph, at(pp, (size_t) (2 * g) * R * CH), CH);
                mx::joint_matrix_load(sg, pl, at(pp, (size_t) (2 * g + 1) * R * CH), CH);
                for (int nt = 0; nt < 4; ++nt) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, T16, T16> acc;
                    mx::joint_matrix_fill(sg, acc, 0.0f);
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, T16, T16, mx::layout::row_major> b;
                    mx::joint_matrix_load(sg, b, at(kvq, VH + (size_t) g * 64 + (size_t) nt * T16), KR);
                    mx::joint_matrix_mad(sg, acc, ph, b, acc);
                    mx::joint_matrix_mad(sg, acc, pl, b, acc);
                    mx::joint_matrix_store(sg, acc, at(st, SG0 + (size_t) nt * T16), 64, mx::layout::row_major);
                }
                sycl::group_barrier(sg);
                for (int i = 0; i < OPL; ++i) {
                    const int e = lane + pm::SG * i;
                    o[i] = sycl::fma(o[i], alpha[e / 64], st[SG0 + e]);
                }
            }
        }
        for (int i = 0; i < OPL; ++i) {
            const int e = lane + pm::SG * i, r = e / 64, col = e % 64;
            if (r < G) {
                const float l = lsum[r];
                out[r * HD + g * 64 + col] = l > 0.0f ? o[i] * unit / l : 0.0f;
            }
        }
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(pm::SG)>}; }
};

constexpr size_t mma_local_bytes() {
    return (size_t) PromptAttnMma::KV_HALVES * 2 + (size_t) pm::CH * 8 + (size_t) NG * R * pm::CH * 4 +
           (size_t) 2 * NG * R * pm::CH * 2 + (size_t) NG * R * 64 * 4 + (size_t) 3 * R * 4;
}

sycl::event launch_mma(sycl::queue& q, const float* qv, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                       int64_t cap, int64_t n_kv, int64_t page_size, float* attn, int64_t n_q) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    return q.submit([&](sycl::handler& h) {
        using K = PromptAttnMma;
        K k{qv, p, ids, steps, cap, n_kv, page_size, scale_log2, attn,
            sycl::local_accessor<half, 1>(sycl::range<1>(K::KV_HALVES), h),
            sycl::local_accessor<long long, 1>(sycl::range<1>(pm::CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) NG * R * pm::CH), h),
            sycl::local_accessor<half, 1>(sycl::range<1>((size_t) 2 * NG * R * pm::CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) NG * R * 64), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_q, (size_t) (n_kv * pm::WG)}, {1, pm::WG}), k);
    });
}

// 16 cells a chunk: 32 kept twice the local memory and ran at half the speed (B70, also with the large register file)
constexpr int CHUNK = 16;
static_assert(CHUNK == 16, "the p split gives each lane of a 16-wide sub-group one cell");

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    // the matrix engines and the work-group's local memory (26 KiB, 34 KiB with Q4_0 K and V); without them the
    // caller keeps the FP32 kernel
    auto& queue = core::Runtime::get().stream(stream);
    const size_t slm = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    if (!xmx_available(XmxType::f16)) {
        // the portable kernel: FP16 pools (the others keep the FP32 kernel)
        static core::PerDevice<bool> per_device;
        const bool mma = per_device.get(queue.get_device(), [&queue, slm] {
            return xe::mma_shape_is(queue, 16, 16, 16, 32) && slm >= mma_local_bytes();
        });
        if (!mma || pools.k_q4 != nullptr || pools.k_q != nullptr || !pools.k_pool || !pools.v_pool) return false;
        const sycl::event e = launch_mma(queue, q, pools, ids, steps, cap, s.n_head_kv, s.page_size, attn, n_q);
        if (!stream) core::Runtime::get().wait(e, "qsa_prompt_attn_batch");
        return true;
    }
    // STRATA_PROMPT_ATTN_Q4=0: Q4_0 K or V (--kv q4_0, --kv k8v4) to the FP32 kernel, as before upstream 778e1f6 (A/B)
    static const bool q4_on = [] {
        const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
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
    if (!stream) core::Runtime::get().wait(e, "qsa_prompt_attn_batch");
    return true;
}

}  // namespace strata::kernels
