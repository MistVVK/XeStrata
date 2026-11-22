// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/tq_avx2.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7 / TQP5-7) on AVX2 + FMA (tq_cpu.hpp).
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU.
//
// Per 128-weight block: the 32 states in scalar code (tq_states.hpp: one 32-bit read, a shift and a mask each, used
// as table indices straight away), four 8-byte state-table rows per YMM register as scalar loads, then per token
// VPMADDWD against the unpacked int16 activations and a VPHADDD tree that leaves the four exact sub-block sums
// S[k] in one XMM, so the float part is the scalar kernel's, lane for lane.  The weights are decoded once per block
// for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block, one token, AVX2): ~48 with this scheme; computing the states with byte
// shuffles instead (VPSHUFB / VPSRLVD per 8 states) and moving them to GPRs or through memory was slower (60-80),
// and ggml-cpu's AVX2 vec_dot for these types (agentionai llama.cpp, the same state table) takes about 2.3x as long.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>
#include <cstring>
#include <cstdlib>

#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif

namespace strata::kernels::cpu::tq_detail {
namespace {

// four state-table rows (8 bytes each) into one YMM: a load and three broadcast loads blended in.  Broadcast loads
// and VPBLENDD avoid the shuffle port that VPINSRQ / VINSERTI128 need (the two measure the same on Zen 5).
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

// a TQP register: eight steps' first pairs (4 bytes each, the state-table row's first half) as scalar loads, like
// rows4 (no gather: see tq_avx512.cpp)
template <int TY>
inline __m256i pairs8(const int32_t* tab, const uint8_t* qs, uint64_t wr, int t0) {
    return _mm256_setr_epi32(tab[2 * tq_state<TY>(qs, wr, t0)], tab[2 * tq_state<TY>(qs, wr, t0 + 1)],
                             tab[2 * tq_state<TY>(qs, wr, t0 + 2)], tab[2 * tq_state<TY>(qs, wr, t0 + 3)],
                             tab[2 * tq_state<TY>(qs, wr, t0 + 4)], tab[2 * tq_state<TY>(qs, wr, t0 + 5)],
                             tab[2 * tq_state<TY>(qs, wr, t0 + 6)], tab[2 * tq_state<TY>(qs, wr, t0 + 7)]);
}

// NT, the token count, is a template parameter so the accumulators stay in registers: with nt at run time clang kept
// acc[] on the stack, a store and a load in every block's chain (TQ2_T rows 13-36% slower on an i7-14700's P-core than
// GCC's build, 2026-10-23)
template <int TY, int NT>
void rows_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, float* const* out, int r0, int r1) {
    constexpr int nt = NT;
    // i < n / 128 and n is int, so i * 128 and i * 4 fit int.
    const int nb = n / 128;
    constexpr size_t bsz = 2 + TqFmt<TY>::NB;
    const long long* tab = (const long long*) tq_state_table();
    const int32_t* tab32 = (const int32_t*) tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[NT];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<TY>(qs);
            __m256i wv[8];   // wv[m] = weights 16m .. 16m+15 (states 4m .. 4m+3; TQP: 8m .. 8m+7)
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (int m = 0; m < 8; ++m) {
                if constexpr (TqFmt<TY>::NS == 64) wv[m] = pairs8<TY>(tab32, qs, wr, 8 * m);
                else
                    wv[m] = rows4(tab, tq_state<TY>(qs, wr, 4 * m), tq_state<TY>(qs, wr, 4 * m + 1),
                                  tq_state<TY>(qs, wr, 4 * m + 2), tq_state<TY>(qs, wr, 4 * m + 3));
            }
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + static_cast<size_t>(i * 128));
                const __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(wv[0], _mm256_load_si256(q + 0)),
                                                    _mm256_madd_epi16(wv[1], _mm256_load_si256(q + 1)));
                const __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(wv[2], _mm256_load_si256(q + 2)),
                                                    _mm256_madd_epi16(wv[3], _mm256_load_si256(q + 3)));
                const __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(wv[4], _mm256_load_si256(q + 4)),
                                                    _mm256_madd_epi16(wv[5], _mm256_load_si256(q + 5)));
                const __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(wv[6], _mm256_load_si256(q + 6)),
                                                    _mm256_madd_epi16(wv[7], _mm256_load_si256(q + 7)));
                // per 128-bit lane [p0, p1, p2, p3] partial sums, then the two lanes added: S[0..3], exact
                const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + static_cast<size_t>(i * 4)));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

template <int TY>
void rows_nt(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
             int r1) {
    static_assert(kTqMaxTokens == 8, "one case per token count");
    switch (nt) {
    case 1: rows_t<TY, 1>(w, row_bytes, n, a, out, r0, r1); break;
    case 2: rows_t<TY, 2>(w, row_bytes, n, a, out, r0, r1); break;
    case 3: rows_t<TY, 3>(w, row_bytes, n, a, out, r0, r1); break;
    case 4: rows_t<TY, 4>(w, row_bytes, n, a, out, r0, r1); break;
    case 5: rows_t<TY, 5>(w, row_bytes, n, a, out, r0, r1); break;
    case 6: rows_t<TY, 6>(w, row_bytes, n, a, out, r0, r1); break;
    case 7: rows_t<TY, 7>(w, row_bytes, n, a, out, r0, r1); break;
    case 8: rows_t<TY, 8>(w, row_bytes, n, a, out, r0, r1); break;
    default: break;
    }
}

}  // namespace

void tq_rows_avx2(int ty, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
    switch (ty) {
        case 144: rows_nt<144>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 145: rows_nt<145>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 146: rows_nt<146>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 147: rows_nt<147>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 148: rows_nt<148>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 149: rows_nt<149>(w, row_bytes, n, a, nt, out, r0, r1); break;
        default: break;
    }
}


namespace {
// `code` has its first two bytes appended, so a wrapped 15-bit window never reads past the block.
inline uint32_t bih3_state_padded(const uint8_t* code, int k, int step) {
    if (k == 8) return (((uint32_t) code[(step + 31) & 31] << 8) | code[step]) & 0x7fffu;
    const int bit = (31 - step) * k, i = bit / 8;
    const uint32_t window = (uint32_t) code[i] | ((uint32_t) code[i + 1] << 8) | ((uint32_t) code[i + 2] << 16);
    return (window >> (bit % 8)) & 0x7fffu;
}

namespace bih3_avx2 {
#define BIH3_MODE 0
#define BIH3_FN
#include "tq_bih3_rows.inl"
#undef BIH3_FN
#undef BIH3_MODE
}
#if STRATA_AVXVNNI
namespace bih3_avxvnni {
#define BIH3_MODE 1
#define BIH3_FN __attribute__((target("avxvnni")))
#include "tq_bih3_rows.inl"
#undef BIH3_FN
#undef BIH3_MODE
}
#endif
namespace bih3_avx512vnni {
#define BIH3_MODE 2
#define BIH3_FN __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#include "tq_bih3_rows.inl"
#undef BIH3_FN
#undef BIH3_MODE
}
}  // namespace

void bih3_rows_x86(Bih3Isa isa, int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                    int nt, float* const* out, int r0, int r1, const TqR1q2Scales& scales) {
    switch (isa) {
        case Bih3Isa::avx512vnni: bih3_avx512vnni::rows(type, w, row_bytes, n, a, nt, out, r0, r1, scales); return;
#if STRATA_AVXVNNI
        case Bih3Isa::avxvnni: bih3_avxvnni::rows(type, w, row_bytes, n, a, nt, out, r0, r1, scales); return;
#endif
        case Bih3Isa::avx2: bih3_avx2::rows(type, w, row_bytes, n, a, nt, out, r0, r1, scales); return;
        default: std::abort();
    }
}

}  // namespace strata::kernels::cpu::tq_detail
