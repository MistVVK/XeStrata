// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/i8_gemm_avx512.cpp - the AVX-512 VNNI copy of the int8 x int8 matrix product
// (include/strata/kernels/cpu/i8_gemm.hpp): zmm vpdpbusd, sixteen outputs a register.  Its functions alone carry
// target("avx512f,avx512vnni"); the file is compiled with the AVX2 flags like i8_gemm_vnni.cpp, and nothing here runs
// before i8_gemm_isa_ok() has said cpu_avx512vnni_ok().
//
// THE SAME WEIGHTS AS THE OTHER COPIES.  The panels stay the eight-output panels every copy reads: a register takes two
// neighbouring panels' 32-byte quads, one load and one insert each pair, so the int8 product has one packed form and a
// band of panels stays a pointer into it (EmbeddingGemma 2's forward splits a matrix that way).  The sums are the
// integer sums of the VNNI copy and the float chain is the same multiply and fused multiply-add per block, so the
// answer is the scalar reference's bit for bit (cpu_i8_gemm_parity under Intel SDE -icx).
//
// THE TILE: four panels (32 outputs, two zmm) against six token rows.  A quad costs four 32-byte loads, two inserts and
// six broadcasts for twelve vpdpbusd; twelve int32 and twelve FP32 accumulators leave room in the 32 registers for the
// weights and the broadcast.  Twelve independent chains cover vpdpbusd's latency on the cores that issue one or two
// zmm vpdpbusd a cycle (Ice Lake and Sapphire Rapids servers, Zen 5), and Zen 4, which runs a zmm operation as two
// 256-bit halves, keeps the same instruction count and loads.  A tail of panels a group of four does not fill (the
// matrix's last ones) goes to the 256-bit copy, whose sums and float chain are the same.
//
// THE TOKEN BLOCK is from the core's L2 (cpu_l2_bytes): half of it holds the block's quantized activation rows, which
// every panel group reads again, so a block's rows stay in the L2 however many panels the band has; a CPU that does not
// say its L2 gets the AVX2 copies' rule.  Not measured on a CPU with AVX-512 (none at hand): its speed is unverified.
#include "strata/kernels/cpu/i8_gemm.hpp"

#include "strata/kernels/cpu/bf16_gemm.hpp"       // bf16_gemm_token_block, the rule when the L2 is not known
#include "strata/kernels/cpu/expert_layout.hpp"   // cpu_avxvnni_ok, cpu_l2_bytes

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if defined(__GNUC__) || defined(__clang__)
#define STRATA_I8_512_FN __attribute__((target("avx512f,avx512vnni")))
#else
#define STRATA_I8_512_FN
#endif

namespace strata::kernels::cpu::i8_gemm_detail {
namespace {

constexpr int kGroup = 4;   // panels a tile: two zmm of sixteen outputs
constexpr int kRows = 6;    // token rows a tile

inline uint32_t quad_bytes(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

/// Two neighbouring panels' quad `kq` (or their scales, corrections: any 32 bytes `stride` apart) as one register.
STRATA_I8_512_FN inline __m512i pair(const uint8_t* a, size_t stride) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + stride));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/// Four whole panels (outputs base .. base + 31) against ROWS token rows; `wp` is the first panel's bytes, `ws` / `wc`
/// its scales and corrections ([block][lane], nb blocks a panel).
template <int ROWS>
STRATA_I8_512_FN inline void tile(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq, size_t ldq,
                                  const float* xs, float* y, size_t ldy, int base, const float* ws, const int32_t* wc,
                                  const float* bias) {
    const size_t ps = (size_t) quads * kI8QuadBytes;   // one panel's bytes
    const size_t ss = (size_t) nb * kI8Panel;          // one panel's scales
    const __m512 bv0 = bias != nullptr ? _mm512_loadu_ps(bias + base) : _mm512_setzero_ps();
    const __m512 bv1 = bias != nullptr ? _mm512_loadu_ps(bias + base + 16) : _mm512_setzero_ps();
    __m512 f0[ROWS], f1[ROWS];
    for (int b = 0; b < nb; ++b) {
        __m512i a0[ROWS], a1[ROWS];
        for (int r = 0; r < ROWS; ++r) a0[r] = a1[r] = _mm512_setzero_si512();
        const int q1 = std::min(quads, (b + 1) * bq);
        for (int kq = b * bq; kq < q1; ++kq) {
            const uint8_t* w = wp + (size_t) kq * kI8QuadBytes;
            const __m512i w0 = pair(w, ps), w1 = pair(w + 2 * ps, ps);
            const uint8_t* a = xq + (size_t) kq * kI8Quad;
            for (int r = 0; r < ROWS; ++r) {
                const __m512i x = _mm512_set1_epi32((int) quad_bytes(a + (size_t) r * ldq));
                a0[r] = _mm512_dpbusd_epi32(a0[r], x, w0);
                a1[r] = _mm512_dpbusd_epi32(a1[r], x, w1);
            }
        }
        const float* sb = ws + (size_t) b * kI8Panel;
        const int32_t* cb = wc + (size_t) b * kI8Panel;
        const __m512 s0 = _mm512_castsi512_ps(pair(reinterpret_cast<const uint8_t*>(sb), ss * 4));
        const __m512 s1 = _mm512_castsi512_ps(pair(reinterpret_cast<const uint8_t*>(sb + 2 * ss), ss * 4));
        const __m512i c0 = pair(reinterpret_cast<const uint8_t*>(cb), ss * 4);
        const __m512i c1 = pair(reinterpret_cast<const uint8_t*>(cb + 2 * ss), ss * 4);
        for (int r = 0; r < ROWS; ++r) {
            const __m512 xsv = _mm512_set1_ps(xs[(size_t) r * nb + b]);
            const __m512 d0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(a0[r], c0));
            const __m512 d1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(a1[r], c1));
            const __m512 t0 = _mm512_mul_ps(xsv, s0), t1 = _mm512_mul_ps(xsv, s1);
            if (b == 0) {
                f0[r] = bias != nullptr ? _mm512_fmadd_ps(t0, d0, bv0) : _mm512_mul_ps(t0, d0);
                f1[r] = bias != nullptr ? _mm512_fmadd_ps(t1, d1, bv1) : _mm512_mul_ps(t1, d1);
            } else {
                f0[r] = _mm512_fmadd_ps(t0, d0, f0[r]);
                f1[r] = _mm512_fmadd_ps(t1, d1, f1[r]);
            }
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        _mm512_storeu_ps(y + (size_t) r * ldy + base, f0[r]);
        _mm512_storeu_ps(y + (size_t) r * ldy + base + 16, f1[r]);
    }
}

/// A group of four panels over `rows` token rows: kRows at a time, then the last few.
STRATA_I8_512_FN inline void group_rows(const uint8_t* wp, int quads, int nb, int bq, const uint8_t* xq, size_t ldq,
                                        const float* xs, float* y, size_t ldy, int base, int rows, const float* ws,
                                        const int32_t* wc, const float* bias) {
    int r = 0;
    for (; r + kRows <= rows; r += kRows)
        tile<kRows>(wp, quads, nb, bq, xq + (size_t) r * ldq, ldq, xs + (size_t) r * nb, y + (size_t) r * ldy, ldy, base,
                    ws, wc, bias);
    const int left = rows - r;
    const uint8_t* xr = xq + (size_t) r * ldq;
    const float* sr = xs + (size_t) r * nb;
    float* yr = y + (size_t) r * ldy;
    switch (left) {
        case 5: tile<5>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias); break;
        case 4: tile<4>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias); break;
        case 3: tile<3>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias); break;
        case 2: tile<2>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias); break;
        case 1: tile<1>(wp, quads, nb, bq, xr, ldq, sr, yr, ldy, base, ws, wc, bias); break;
        default: break;
    }
    static_assert(kRows <= 6, "the last rows below a group are at most five");
}

/// The token rows of a block: half the L2 of activation rows, in whole tiles of rows.
int token_block(int n_in) {
    const uint64_t l2 = cpu_l2_bytes();
    if (l2 == 0) return bf16_gemm_token_block(n_in);
    const int rows = (int) std::min<uint64_t>(4096, l2 / 2 / (uint64_t) i8_gemm_act_row_bytes(n_in));
    return std::max(kRows, rows - rows % kRows);
}

STRATA_I8_512_FN void rows_avx512(const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                                  const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok,
                                  int p0, int p1, int block, const float* bias, int pe) {
    const int quads = i8_gemm_quads(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    const int bq = block <= 0 ? quads : block / kI8Quad;
    const int tb = token_block(n_in);
    for (int t0 = 0; t0 < n_tok; t0 += tb) {
        const int rows = std::min(tb, n_tok - t0);
        for (int p = p0; p < pe; p += kGroup)
            group_rows(panel + (size_t) p * quads * kI8QuadBytes, quads, nb, bq, xq + (size_t) t0 * ldq, ldq,
                       xs + (size_t) t0 * nb, y + (size_t) t0 * ldy, ldy, p * kI8Panel, rows,
                       wscale + (size_t) p * nb * kI8Panel, wcorr + (size_t) p * nb * kI8Panel, bias);
    }
    (void) n_out;
    (void) p1;
}

}  // namespace

void i8_gemm_rows_avx512(const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                         const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok, int p0, int p1,
                         const float* bias, int block) {
    // the groups of four whole panels; the rest (a matrix's last panels, a partial one among them) on 256 bits
    int pe = p0 + (p1 - p0) / kGroup * kGroup;
    if (pe > p0 && pe * kI8Panel > n_out) pe -= kGroup;
    if (pe > p0) rows_avx512(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, block, bias, pe);
    if (pe < p1)
        i8_gemm_rows_x86(STRATA_AVXVNNI && cpu_avxvnni_ok() ? 1 : 0, panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y,
                         ldy, n_tok, pe, p1, bias, block, kI8Levels);
}

}  // namespace strata::kernels::cpu::i8_gemm_detail
