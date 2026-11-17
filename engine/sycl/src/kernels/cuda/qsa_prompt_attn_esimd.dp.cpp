// SYCL port (ARC-W): the QSA prompt attention on the int8 DPAS (ESIMD, Battlemage).
//
// The FP32 prompt kernel (qsa_decode_attn_batch, 64-cell chunks + a merge kernel) is compute bound at ~60% of the
// card's FP32 peak and takes 22-24% of a prompt.  This kernel runs both products on the s8 8x16x32 DPAS.
//
//  * One work-group of NTH = 4 hardware threads per (query, KV head).  Thread t owns the output dims [64 t, 64 t + 64),
//    which is also V's scale group t, and computes Q.K for 32 of every 128 selected cells (a "super-step").
//  * Q.K: A = 8 cells x 32 dims of K's int8 codes (a row of K is contiguous, so A is eight 32-byte slices), B = the
//    32 dims x 16 heads of Q (12 real heads, 4 zero lanes), D = 8 cells x 16 heads.  Q is quantized per head to
//    hi + lo int8 (x ~ sq (hi + lo / 254), error ~ 1.5e-5 of the head's largest value) so the product keeps ~15 bits;
//    K's per-64-dim scale is applied per cell row to each 64-dim group's int32 sum.  The scores of the thread's 32
//    cells go to local memory; one barrier later every thread has all 128.
//  * Softmax: every thread reads the 128 score rows (lane = head), takes the running max, rescales its output
//    accumulators and forms p = exp2(s - m).
//  * P.V: A = P (8 heads x 32 cells, int8 hi + lo of p * v_scale[cell][group t], scaled to 127 by the super-step's
//    largest v scale), B = the thread's 64 dims of 32 V rows packed four cells to a dword (a 4x4 byte transpose in
//    registers), D = 8 heads x 16 dims into float accumulators held in registers (12 heads in two 8-row tiles x 4
//    dim tiles).
//  * Not bitwise the FP32 kernel: Q and P are 16-bit (hi + lo int8) numbers, the sums are exact int32, the scales and
//    the softmax are FP32.  qsa_prompt_attn_parity bounds the difference against FP64.
// int8 KV only (the default); anything else keeps the caller's fallback.
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"
#include "strata/kprof.hpp"
#include "strata/core/geometry.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#ifndef STRATA_I8_NO_ESIMD
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int HD = 256, G = strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, NTH = 4, SS = 128;
static_assert(G >= 1 && G <= 16, "qsa_prompt_attn_esimd: query heads per KV head must be 1..16 (16 score columns, two 8-row tiles)");
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_esimd is compiled for core/geometry.hpp");
constexpr uint32_t QT_OFF = 0;            // Q^T B tiles: [8 K steps][LV levels][128 dwords]

SYCL_ESIMD_FUNCTION inline simd<int, 128> dpas8(simd<int, 128> acc, simd<int, 128> b, simd<int, 64> a) {
    return xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, b, a);
}

// row index of 32 selected cells (selection positions c_first ...), as the page table maps them; positions past n take `fallback`
SYCL_ESIMD_FUNCTION inline simd<int, 32> cell_rows(const int32_t* idsb, const int32_t* pt, int c_first, int n, int kvh, int nkv,
                                                   int pshift, int pmask, int fallback) {
    simd<uint32_t, 32> lane(0, 1);
    simd<int, 32> pos = simd<int, 32>(c_first) + simd<int, 32>(lane);
    simd_mask<32> ok = pos < n;
    simd<int, 32> cell = gather<int, 32>(idsb, lane * 4u, ok, simd<int, 32>(0));
    simd<uint32_t, 32> poff = simd<uint32_t, 32>(cell >> pshift) * 4u;
    simd<int, 32> page = gather<int, 32>(pt, poff, ok, simd<int, 32>(0));
    simd<int, 32> rows = ((page * nkv + kvh) << pshift) + (cell & pmask);
    rows.merge(simd<int, 32>(fallback), !ok);
    return rows;
}

template <int LV>
SYCL_ESIMD_FUNCTION constexpr float level_coef(int l) { return LV == 2 ? (l == 0 ? 254.0f : 1.0f) : (l == 0 ? 64516.0f : (l == 1 ? 254.0f : 1.0f)); }

// x in [-127, 127] as LV int8 levels: x ~ o[0] + o[1] / 254 + o[2] / 254^2
template <int N, int LV>
SYCL_ESIMD_FUNCTION inline void split_lv(simd<float, N> x, simd<int8_t, N>* o) {
    simd<float, N> r = x;
#pragma unroll
    for (int l = 0; l < LV; ++l) {
        simd<float, N> f = rnde<float>(r);   // |r| <= 127 at every level, so the rounded value fits an int8
        o[l] = f;
        r = (r - f) * 254.0f;
    }
}

constexpr unsigned SURF_H = (1u << 24) - 1u;   // 2D block loads: the surface height - 1 (no clamping)

// four consecutive pool rows starting at element i of a row-index vector
template <int N>
SYCL_ESIMD_FUNCTION inline bool contig4(const simd<int, N>& r, int i) {
    const int r0 = r[i];
    return r[i + 1] == r0 + 1 && r[i + 2] == r0 + 2 && r[i + 3] == r0 + 3;
}

struct Args {
    const float* q;
    const int8_t* kq;
    const uint16_t* ks;
    const int8_t* vq;
    const uint16_t* vs;
    const int32_t* pt;
    const int32_t* ids;
    const int32_t* steps;
    float* attn;
    int nkv, cap, pshift, pmask;
    float scale_log2;
    int dbg;
};

// ---- Q.K for one tile of 8 cells: S[cell][head] -> local memory (rows = cells, lanes = heads)
//      FAST: the 8 cells are two runs of four consecutive pool rows (the indexer selects blocks of four cells): 2D block loads
template <bool FAST, int QL>
SYCL_ESIMD_FUNCTION inline void qk_tile(const Args& a, const simd<int, 8>& rows, int pos0, int n, const simd<float, 16>& cscale,
                                        uint32_t s_dst) {
    simd<float, 128> sacc = 0.0f;
    simd<int, 512> kall;
    simd<uint8_t, 256> dA[4], dB[4];
    if constexpr (FAST) {
        const int rA = rows[0], rB = rows[4];
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
            dA[pp] = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 32, 4, 2, false, false>(
                reinterpret_cast<const uint8_t*>(a.kq), 255u, SURF_H, 255u, 64 * pp, rA);
            dB[pp] = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 32, 4, 2, false, false>(
                reinterpret_cast<const uint8_t*>(a.kq), 255u, SURF_H, 255u, 64 * pp, rB);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const size_t r = (size_t) rows[j];
            kall.template select<64, 1>(64 * j) = block_load<int, 64>(reinterpret_cast<const int*>(a.kq) + r * 64);
        }
    }
    // K scales: 8 rows x 4 halves as 16 dwords
    simd<uint32_t, 16> so;
#pragma unroll
    for (int j = 0; j < 8; ++j) { so[2 * j] = (uint32_t) rows[j] * 8u; so[2 * j + 1] = (uint32_t) rows[j] * 8u + 4u; }
    simd<uint32_t, 16> sd = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(a.ks), so);
    simd<sycl::half, 32> ksh = sd.template bit_cast_view<sycl::half>();
    simd<float, 32> ksf = ksh;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        simd<int, 128> acc[QL];
#pragma unroll
        for (int l = 0; l < QL; ++l) acc[l] = 0;
#pragma unroll
        for (int ss = 0; ss < 2; ++ss) {
            const int s = 2 * g + ss;
            simd<int, 64> A;
            if constexpr (FAST) {
                simd<int, 64> da = dA[s >> 1].template bit_cast_view<int>(), db = dB[s >> 1].template bit_cast_view<int>();
                A.template select<32, 1>(0) = da.template select<32, 1>((s & 1) * 32);
                A.template select<32, 1>(32) = db.template select<32, 1>((s & 1) * 32);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) A.template select<8, 1>(8 * j) = kall.template select<8, 1>(64 * j + 8 * s);
            }
#pragma unroll
            for (int l = 0; l < QL; ++l) {
                simd<int, 128> bq = slm_block_load<int, 128>(QT_OFF + (uint32_t) (s * QL + l) * 512u);
                acc[l] = dpas8(acc[l], bq, A);
            }
        }
        simd<float, 128> tf;
        if constexpr (QL == 2) {
            simd<int, 128> ti = acc[0] * 254 + acc[1];
            tf = ti;
        } else {
            tf = 0.0f;
#pragma unroll
            for (int l = 0; l < QL; ++l) {
                simd<float, 128> fl = acc[l];
                tf += fl * level_coef<QL>(l);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float ksc = ksf[4 * j + g];
            sacc.template select<16, 1>(16 * j) += tf.template select<16, 1>(16 * j) * ksc;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        simd<float, 16> row = sacc.template select<16, 1>(16 * j) * cscale;
        const float madd = (pos0 + j >= n) ? -INFINITY : 0.0f;
        row += madd;
        sacc.template select<16, 1>(16 * j) = row;
    }
    slm_block_store<float, 128>(s_dst, sacc);
}

// ---- P.V for one step of 32 cells: O += P' V (this thread's 64 dims); lsum += sum p
template <bool FAST, int PL>
SYCL_ESIMD_FUNCTION inline void pv_step(const Args& a, simd<float, 1024>& O, simd<float, 16>& lsum, const simd<float, 16>& m_new,
                                        uint32_t s_row0, const simd<float, 32>& vsf, const simd<int, 32>& rowsV, int t) {
    constexpr float COEF0P = PL == 2 ? 254.0f : 64516.0f;
    simd<int8_t, 256> ap[PL][2];   // [level][heads 0-7, 8-15]: 8 rows x 32 cells
    simd<float, 16> prow[32];
    simd<float, 16> pm = 0.0f;     // per head: the largest p * v_scale of these 32 cells
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        simd<float, 16> srow = slm_block_load<float, 16>(s_row0 + (uint32_t) c * 64u);
        simd<float, 16> p = exp2(srow - m_new);
        lsum += p;
        const float vsc = vsf[c];
        prow[c] = p * vsc;
        pm = max(pm, prow[c]);
    }
    simd<float, 16> pinv = 127.0f / pm;
    pinv.merge(simd<float, 16>(0.0f), pm <= 0.0f);
    const simd<float, 16> osv = pm * (1.0f / (127.0f * COEF0P));
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        simd<float, 16> r = prow[c] * pinv;
        simd<int8_t, 16> pb[PL];
        split_lv<16, PL>(r, pb);
#pragma unroll
        for (int l = 0; l < PL; ++l) {
            ap[l][0].template select<8, 32>(c) = pb[l].template select<8, 1>(0);
            ap[l][1].template select<8, 32>(c) = pb[l].template select<8, 1>(8);
        }
    }
    // V: 32 rows x this thread's 64 dims, four cells to a dword
    simd<int, 512> vt;   // [k4][64 dims]
#pragma unroll
    for (int k4 = 0; k4 < 8; ++k4) {
        simd<int, 64> w;
        if constexpr (FAST) {
            simd<uint8_t, 256> v2 = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 16, 4, 4, false, true>(
                reinterpret_cast<const uint8_t*>(a.vq), 255u, SURF_H, 255u, 64 * t, (int) rowsV[4 * k4]);
            w = v2.template bit_cast_view<int>();
        } else {
            auto wb = w.template bit_cast_view<uint8_t>();
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t r = (size_t) rowsV[4 * k4 + j];
                wb.template select<64, 4>(j) = block_load<uint8_t, 64>(reinterpret_cast<const uint8_t*>(a.vq) + r * HD + 64 * t);
            }
        }
        vt.template select<64, 1>(64 * k4) = w;
    }
    simd<int, 64> aq[PL][2];
#pragma unroll
    for (int l = 0; l < PL; ++l) {
        aq[l][0] = ap[l][0].template bit_cast_view<int>();
        aq[l][1] = ap[l][1].template bit_cast_view<int>();
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        simd<int, 128> B;
#pragma unroll
        for (int k4 = 0; k4 < 8; ++k4) B.template select<16, 1>(16 * k4) = vt.template select<16, 1>(64 * k4 + 16 * nt);
        simd<int, 128> z = 0;
        simd<float, 128> f0 = 0.0f, f1 = 0.0f;
#pragma unroll
        for (int l = 0; l < PL; ++l) {
            simd<int, 128> d0 = dpas8(z, B, aq[l][0]), d1 = dpas8(z, B, aq[l][1]);
            simd<float, 128> e0 = d0, e1 = d1;
            f0 += e0 * level_coef<PL>(l);
            f1 += e1 * level_coef<PL>(l);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float s0 = osv[i], s1 = osv[8 + i];
            O.template select<16, 1>((0 * 4 + nt) * 128 + 16 * i) += f0.template select<16, 1>(16 * i) * s0;
            O.template select<16, 1>((1 * 4 + nt) * 128 + 16 * i) += f1.template select<16, 1>(16 * i) * s1;
        }
    }
}

template <int QL, int PL> struct TagEsimdAttn;

template <int QL, int PL>
void launch(sycl::queue& q, const Args a, int64_t n_q) {
    constexpr uint32_t S_OFF = 8u * QL * 512u;        // the scores of one super-step: [128 cells][16 heads] floats
    constexpr uint32_t SLM_BYTES = S_OFF + 8192u;
    constexpr float COEF0Q = QL == 2 ? 254.0f : 64516.0f;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    strata::kprof::kp(&q, __FILE__, __func__, __LINE__)->submit([&](sycl::handler& h) {
        h.parallel_for<TagEsimdAttn<QL, PL>>(sycl::nd_range<3>(sycl::range<3>((size_t) n_q, (size_t) a.nkv, (size_t) NTH), sycl::range<3>(1, 1, NTH)), props,
                                     [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            slm_init<SLM_BYTES>();
            const int qi = (int) ndi.get_group(0), kvh = (int) ndi.get_group(1), t = (int) ndi.get_local_id(2);
            const int n_head = a.nkv * G;
            const int n = a.steps[(size_t) qi * kStepCount + kStepWidth];
            const int32_t* idsb = a.ids + (size_t) qi * a.cap;
            float* outp = a.attn + ((size_t) qi * n_head + (size_t) kvh * G) * HD + 64 * t;
            const float* qp = a.q + ((size_t) qi * n_head + (size_t) kvh * G) * HD;

            // ---- Q: per-head int8 levels, the B tiles of this thread's two K steps (2t, 2t + 1)
            const simd<uint32_t, 16> lane16(0, 1);
            simd<float, 16> sqv = 0.0f;   // lane h: the head's scale (largest |q| / 127)
            {
                simd<int, 128> bq[QL][2];   // [level][step 2t, 2t + 1]
#pragma unroll
                for (int l = 0; l < QL; ++l) { bq[l][0] = 0; bq[l][1] = 0; }
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    const float* row = qp + (size_t) hh * HD;
                    simd<float, 64> r0 = block_load<float, 64>(row), r1 = block_load<float, 64>(row + 64);
                    simd<float, 64> r2 = block_load<float, 64>(row + 128), r3 = block_load<float, 64>(row + 192);
                    simd<float, 64> mx = max(max(abs(r0), abs(r1)), max(abs(r2), abs(r3)));
                    const float amax = hmax<float>(mx);
                    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                    sqv.merge(simd<float, 16>(amax * (1.0f / 127.0f)), lane16 == (uint32_t) hh);
                    simd<float, 64> mine = block_load<float, 64>(row + 64 * t);
                    simd<int8_t, 64> qb[QL];
                    split_lv<64, QL>(mine * inv, qb);
#pragma unroll
                    for (int l = 0; l < QL; ++l) {
                        simd<int, 16> qd = qb[l].template bit_cast_view<int>();
                        bq[l][0].template select<8, 16>(hh) = qd.template select<8, 1>(0);
                        bq[l][1].template select<8, 16>(hh) = qd.template select<8, 1>(8);
                    }
                }
                const uint32_t s0 = QT_OFF + (uint32_t) (2 * t) * (uint32_t) QL * 512u;
#pragma unroll
                for (int l = 0; l < QL; ++l) {
                    slm_block_store<int, 128>(s0 + (uint32_t) l * 512u, bq[l][0]);
                    slm_block_store<int, 128>(s0 + (uint32_t) (QL + l) * 512u, bq[l][1]);
                }
            }
            barrier();
            // S[cell][head] = sum_g ks[cell][g] * (sum of the levels) * sq[head] * scale_log2
            const simd<float, 16> cscale = sqv * (a.scale_log2 / COEF0Q);

            // the selection's first cell stands in for positions past n (finite scales, weight 0)
            int fallback = 0;
            if (n > 0) {
                const int c = idsb[0];
                fallback = ((a.pt[c >> a.pshift] * a.nkv + kvh) << a.pshift) + (c & a.pmask);
            }

            simd<float, 1024> O = 0.0f;   // [rt][nt] tiles of (8 heads x 16 dims)
            simd<float, 16> m = -INFINITY, l = 0.0f;

            for (int c0 = 0; c0 < n; c0 += SS) {
                // ---- Q.K for this thread's 32 cells
                simd<int, 32> rowsK = cell_rows(idsb + c0 + 32 * t, a.pt, c0 + 32 * t, n, kvh, a.nkv, a.pshift, a.pmask, fallback);
#pragma unroll
                for (int tl = 0; tl < 4; ++tl) {
                    simd<int, 8> r8 = rowsK.template select<8, 1>(8 * tl);
                    const bool fast = !(a.dbg & 16) && contig4(rowsK, 8 * tl) && contig4(rowsK, 8 * tl + 4);
                    const uint32_t dst = S_OFF + (uint32_t) (32 * t + 8 * tl) * 64u;
                    if (a.dbg & 2) continue;
                    if (fast) qk_tile<true, QL>(a, r8, c0 + 32 * t + 8 * tl, n, cscale, dst);
                    else qk_tile<false, QL>(a, r8, c0 + 32 * t + 8 * tl, n, cscale, dst);
                }
                barrier();
                // ---- softmax over the 128 cells
                simd<float, 16> mx = -INFINITY;
#pragma unroll 1
                for (int b = 0; b < 16; ++b) {
                    simd<float, 128> blk = slm_block_load<float, 128>(S_OFF + (uint32_t) b * 512u);
#pragma unroll
                    for (int r = 0; r < 8; ++r) mx = max(mx, simd<float, 16>(blk.template select<16, 1>(16 * r)));
                }
                const simd<float, 16> m_new = max(m, mx);
                const simd<float, 16> alpha = exp2(m - m_new);
#pragma unroll
                for (int hh = 0; hh < 16; ++hh) {
                    const float av = alpha[hh];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        auto tile = O.template select<16, 1>(((hh >> 3) * 4 + nt) * 128 + (hh & 7) * 16);
                        tile *= av;
                    }
                }
                l *= alpha;
                simd<float, 16> lsum = 0.0f;
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    // V rows of the step's 32 cells, and their scales for this thread's group
                    const simd<int, 32> rowsV = cell_rows(idsb + c0 + 32 * st, a.pt, c0 + 32 * st, n, kvh, a.nkv, a.pshift, a.pmask, fallback);
                    simd<uint32_t, 32> vo = simd<uint32_t, 32>(rowsV) * 8u + (uint32_t) ((t >> 1) * 4);
                    simd<uint32_t, 32> vd = gather<uint32_t, 32>(reinterpret_cast<const uint32_t*>(a.vs), vo);
                    simd<uint32_t, 32> shv = (uint32_t) ((t & 1) * 16);
                    simd<uint16_t, 32> v16 = ((vd >> shv) & 0xffffu);
                    simd<sycl::half, 32> vh = v16.template bit_cast_view<sycl::half>();
                    simd<float, 32> vf = vh;
                    simd<uint32_t, 32> lane(0, 1);
                    simd_mask<32> ok = (simd<int, 32>(c0 + 32 * st) + simd<int, 32>(lane)) < n;
                    vf.merge(simd<float, 32>(0.0f), !ok);
                    bool fastv = !(a.dbg & 16);
#pragma unroll
                    for (int g4 = 0; g4 < 8; ++g4) fastv = fastv && contig4(rowsV, 4 * g4);
                    if (fastv) pv_step<true, PL>(a, O, lsum, m_new, S_OFF + (uint32_t) (32 * st) * 64u, vf, rowsV, t);
                    else pv_step<false, PL>(a, O, lsum, m_new, S_OFF + (uint32_t) (32 * st) * 64u, vf, rowsV, t);
                }
                l += lsum;
                m = m_new;
                barrier();
            }
            // ---- out = O / l
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                const float lv = l[hh];
                const float il = lv > 0.0f ? 1.0f / lv : 0.0f;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    simd<float, 16> v = O.template select<16, 1>(((hh >> 3) * 4 + nt) * 128 + (hh & 7) * 16) * il;
                    block_store<float, 16>(outp + (size_t) hh * HD + 16 * nt, v);
                }
            }
        });
    });
}
}  // namespace

bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    static const int mode = [] { const char* v = std::getenv("STRATA_PROMPT_ATTN_ESIMD"); return v ? std::atoi(v) : 1; }();   // default on (Battlemage); =0 off
    if (mode == 0 || n_q <= 0) return false;
    if (pools.k_q == nullptr || pools.v_q == nullptr || pools.k_scale == nullptr || pools.v_scale == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps ||
        !pools.page_table)
        return false;
    const int64_t ps = s.page_size;
    if (ps <= 0 || (ps & (ps - 1)) != 0) return false;
    dpct::queue_ptr st = strata::q_of(stream);
    if (strata::intel_gpu_gen(st->get_device()) != strata::IntelGpuGen::Battlemage) return false;
    static bool told = false;
    if (!told) { told = true; std::fprintf(stderr, "strata: prompt attention on the int8 DPAS (ESIMD)\n"); }
    static const int lv = [] { const char* v = std::getenv("STRATA_ESIMD_LV"); return v ? std::atoi(v) : 2; }();
    int sh = 0;
    while ((1 << sh) < ps) ++sh;
    Args a{q, pools.k_q, pools.k_scale, pools.v_q, pools.v_scale, pools.page_table, ids, steps, attn,
           (int) s.n_head_kv, (int) cap, sh, (int) (ps - 1), 1.4426950408889634f / std::sqrt((float) HD),
           std::getenv("STRATA_ESIMD_DBG") != nullptr ? std::atoi(std::getenv("STRATA_ESIMD_DBG")) : 0};
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        Args b = a;
        b.q = q + q0 * s.n_head * HD;
        b.ids = ids + q0 * cap;
        b.steps = steps + q0 * kStepCount;
        b.attn = attn + q0 * s.n_head * HD;
        if (lv == 3) launch<3, 3>(*st, b, nb);
        else if (lv == 32) launch<3, 2>(*st, b, nb);
        else if (lv == 23) launch<2, 3>(*st, b, nb);
        else launch<2, 2>(*st, b, nb);
    }
    return true;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool qsa_prompt_attn_esimd(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*, int64_t, void*) { return false; }
}
#endif
