// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// src/kernels/cpu/latent_rows.cpp - the latent MoE's CPU-share rows.  See the header.
//
// The U product's integer chain is the GPU kernel's (src/kernels/xe/latent.cpp): the exact 32-value integer
// sum a block, (dx * dw) * sumi the block's float product, the blocks accumulated in order.  The AVX2 form
// folds the weight's signs into the activation codes (maddubs runs u8 x s8), edge0-strata's trick; the scalar
// form is the same integers.  The q8_1 quantizer keeps amax and sum in scalar order (their order feeds the
// fp16 d and sum: a SIMD reduction would reorder it).
#include "strata/kernels/cpu/latent_rows.hpp"

#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels::cpu {

void latent_quant_q8_1_rows(const float* x, int64_t rows, int64_t n, uint8_t* y) {
    namespace canon = strata::kernels::canon;
    const int64_t group = n % 256 == 0 ? 256 : 32;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g0 = 0; g0 < n; g0 += group) {
            float amax = 0;
            for (int64_t j = g0; j < g0 + group; ++j) amax = std::fmax(amax, std::fabs(x[r * n + j]));
            const float d = canon::q8_scale(amax), id = canon::q8_inv(d);
            const uint16_t dh = strata::kernels::f16_from_f32(d);
            for (int64_t b = g0; b < g0 + group; b += 32) {
                float sum = 0;
                for (int64_t j = b; j < b + 32; ++j) sum += x[r * n + j];
                uint8_t* out = y + (r * (n / 32) + b / 32) * 36;
                const uint16_t sh = strata::kernels::f16_from_f32(sum);
                out[0] = (uint8_t) (dh & 0xFF); out[1] = (uint8_t) (dh >> 8);
                out[2] = (uint8_t) (sh & 0xFF); out[3] = (uint8_t) (sh >> 8);
                for (int64_t j = 0; j < 32; ++j) out[4 + j] = (uint8_t) canon::q8_value(x[r * n + b + j], id);
            }
        }
    }
}

namespace {

float h2f(uint16_t h) { return strata::kernels::f32_from_f16(h); }

// one 32-value block's integer dot, scalar
int block_sumi_q8_0(const uint8_t* wb, const uint8_t* xb) {
    int sumi = 0;
    for (int j = 0; j < 32; ++j) sumi += (int) (int8_t) wb[2 + j] * (int) (int8_t) xb[4 + j];
    return sumi;
}
int block_sumi_q4_0(const uint8_t* wb, const uint8_t* xb) {
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        const int lo = (wb[2 + j] & 0xF) - 8, hi = (wb[2 + j] >> 4) - 8;
        sumi += lo * (int) (int8_t) xb[4 + j] + hi * (int) (int8_t) xb[4 + j + 16];
    }
    return sumi;
}

void u_row_scalar(const uint8_t* u, int H, int ELL, int fmt, const uint8_t* hq, float* out) {
    const int nb = ELL / 32;
    const size_t wrow = (size_t) nb * (fmt == 4 ? 18 : 34);
    for (int h = 0; h < H; ++h) {
        const uint8_t* w = u + (size_t) h * wrow;
        float acc = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* wb = w + (size_t) b * (fmt == 4 ? 18 : 34);
            const uint8_t* xb = hq + (size_t) b * 36;
            uint16_t dw, dx;
            std::memcpy(&dw, wb, 2);
            std::memcpy(&dx, xb, 2);
            const int sumi = fmt == 4 ? block_sumi_q4_0(wb, xb) : block_sumi_q8_0(wb, xb);
            acc += (h2f(dx) * h2f(dw)) * (float) sumi;
        }
        out[h] = acc;
    }
}

#if defined(__x86_64__) || defined(_M_X64)
// the weight block's 32 centred codes and the activation codes with the weight's signs folded in (the u8 x s8
// pair the byte-dot instructions take): |w| and sign(qy, w).  Q4_0's nibbles unpacked and centred first.
__attribute__((target("avx2,fma,f16c")))
static inline void block_codes_avx2(const uint8_t* wb, const uint8_t* xb, int fmt, __m256i& aw, __m256i& sq) {
    const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 4));
    __m256i wq;
    if (fmt == 4) {
        const __m128i nib = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wb + 2));
        const __m128i lo = _mm_and_si128(nib, _mm_set1_epi8(0xF));
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), _mm_set1_epi8(0xF));
        wq = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8));
    } else {
        wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + 2));
    }
    aw = _mm256_sign_epi8(wq, wq);
    sq = _mm256_sign_epi8(qy, wq);
}

__attribute__((target("avx2,fma,f16c")))
void u_row_avx2(const uint8_t* u, int H, int ELL, int fmt, const uint8_t* hq, float* out) {
    const int nb = ELL / 32;
    const size_t wrow = (size_t) nb * (fmt == 4 ? 18 : 34);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int h = 0; h < H; ++h) {
        const uint8_t* w = u + (size_t) h * wrow;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const uint8_t* wb = w + (size_t) b * (fmt == 4 ? 18 : 34);
            const uint8_t* xb = hq + (size_t) b * 36;
            uint16_t dw, dx;
            std::memcpy(&dw, wb, 2);
            std::memcpy(&dx, xb, 2);
            const __m256 d = _mm256_set1_ps(h2f(dx) * h2f(dw));
            __m256i aw, sq;
            block_codes_avx2(wb, xb, fmt, aw, sq);
            const __m256i p16 = _mm256_maddubs_epi16(aw, sq);
            const __m256i si = _mm256_madd_epi16(ones, p16);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(si), acc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[h] = _mm_cvtss_f32(s);
    }
}

// AVX-VNNI: the byte dot is one vpdpbusd (u8 x s8 -> int32) instead of maddubs + madd.
__attribute__((target("avx2,fma,f16c,avxvnni")))
void u_row_vnni(const uint8_t* u, int H, int ELL, int fmt, const uint8_t* hq, float* out) {
    const int nb = ELL / 32;
    const size_t wrow = (size_t) nb * (fmt == 4 ? 18 : 34);
    for (int h = 0; h < H; ++h) {
        const uint8_t* w = u + (size_t) h * wrow;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const uint8_t* wb = w + (size_t) b * (fmt == 4 ? 18 : 34);
            const uint8_t* xb = hq + (size_t) b * 36;
            uint16_t dw, dx;
            std::memcpy(&dw, wb, 2);
            std::memcpy(&dx, xb, 2);
            const __m256 d = _mm256_set1_ps(h2f(dx) * h2f(dw));
            __m256i aw, sq;
            block_codes_avx2(wb, xb, fmt, aw, sq);
            const __m256i si = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sq);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(si), acc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[h] = _mm_cvtss_f32(s);
    }
}

// AVX-512 VNNI: two blocks a vpdpbusd (the low 8 int32 are block b's, the high 8 block b+1's), each half scaled
// by its own block's (dx * dw).  nb is even at both latent widths (56, 40); an odd tail takes the 256-bit form.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
void u_row_avx512vnni(const uint8_t* u, int H, int ELL, int fmt, const uint8_t* hq, float* out) {
    const int nb = ELL / 32;
    const size_t bs = (size_t) (fmt == 4 ? 18 : 34);
    const size_t wrow = (size_t) nb * bs;
    for (int h = 0; h < H; ++h) {
        const uint8_t* w = u + (size_t) h * wrow;
        __m512 acc = _mm512_setzero_ps();
        int b = 0;
        for (; b + 1 < nb; b += 2) {
            const uint8_t* wb0 = w + (size_t) b * bs;
            const uint8_t* wb1 = wb0 + bs;
            const uint8_t* xb0 = hq + (size_t) b * 36;
            const uint8_t* xb1 = xb0 + 36;
            uint16_t dw0, dx0, dw1, dx1;
            std::memcpy(&dw0, wb0, 2); std::memcpy(&dx0, xb0, 2);
            std::memcpy(&dw1, wb1, 2); std::memcpy(&dx1, xb1, 2);
            __m256i aw0, sq0, aw1, sq1;
            block_codes_avx2(wb0, xb0, fmt, aw0, sq0);
            block_codes_avx2(wb1, xb1, fmt, aw1, sq1);
            const __m512i aw = _mm512_inserti64x4(_mm512_castsi256_si512(aw0), aw1, 1);
            const __m512i sq = _mm512_inserti64x4(_mm512_castsi256_si512(sq0), sq1, 1);
            const __m512i si = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sq);
            // the two blocks' scales in the two halves (AVX-512 F's 64-bit insert: cpu_avx512_ok does not ask for DQ)
            const __m512 d = _mm512_castpd_ps(_mm512_insertf64x4(
                _mm512_castps_pd(_mm512_castps256_ps512(_mm256_set1_ps(h2f(dx0) * h2f(dw0)))),
                _mm256_castps_pd(_mm256_set1_ps(h2f(dx1) * h2f(dw1))), 1));
            acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(si), acc);
        }
        __m256 acc256 = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc), 1)));
        for (; b < nb; ++b) {   // an odd tail block
            const uint8_t* wb = w + (size_t) b * bs;
            const uint8_t* xb = hq + (size_t) b * 36;
            uint16_t dw, dx;
            std::memcpy(&dw, wb, 2); std::memcpy(&dx, xb, 2);
            __m256i aw, sq;
            block_codes_avx2(wb, xb, fmt, aw, sq);
            const __m256i si = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sq);
            acc256 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(dx) * h2f(dw)), _mm256_cvtepi32_ps(si), acc256);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc256), _mm256_extractf128_ps(acc256, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[h] = _mm_cvtss_f32(s);
    }
}
#endif

}  // namespace

void latent_u_row(const uint8_t* u_group, int H, int ELL, int fmt, const uint8_t* hq, float* out) {
#if defined(__x86_64__) || defined(_M_X64)
    // the device's own reports, fastest first (never a maker's name); scalar is the reference fallback
    if (cpu_avx512_latent_ok()) {
        u_row_avx512vnni(u_group, H, ELL, fmt, hq, out);
        return;
    }
    if (cpu_avxvnni_ok()) {
        u_row_vnni(u_group, H, ELL, fmt, hq, out);
        return;
    }
    if (cpu_avx2_ok()) {
        u_row_avx2(u_group, H, ELL, fmt, hq, out);
        return;
    }
#endif
    u_row_scalar(u_group, H, ELL, fmt, hq, out);
}

}  // namespace strata::kernels::cpu