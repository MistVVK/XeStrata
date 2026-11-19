// src/kernels/cpu/kq_avx512.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several tokens at
// once on AVX-512 VNNI, BIT-EXACT against the AVX2 contract's one-token dot (ggml-cpu's AVX2 branches, which the quant layer's
// Arith::Avx2 reproduces): the same rows kq_avx2.cpp gives, with the integer part of every block on vpdpbusd.
//
// What is the same as the AVX2 chain, token by token and block by block:
//   - the eight-lane float accumulator `acc` (fmadd of d * the block's eight integer lanes), ONE per token, blocks in order, and its
//     hsum last (ggml's hsum_float_8); for Q4_K the second, four-lane accumulator of the mins part and its hsum added last;
//   - the integer lane values.  They are exact integer sums (no saturation: a Q4_K pair of byte products is at most 15 * 128 * 2,
//     a Q5_1 pair 31 * 127 * 2, a Q8_0 pair 128 * 128 * 2 = 32768 - the same bound the AVX2 code's maddubs lives with), so any order of
//     adding them gives the same i32; here one vpdpbusd adds four consecutive byte products into a lane, exactly the four bytes
//     maddubs + madd add in ggml's AVX2 code, and Q4_K's per-32-value scale multiplies a lane's sum with vpdpwssd (the scale as the
//     low 16 bits of the i32, the high half zero, the lane's sum - at most 4 * 15 * 128 in magnitude - as its low 16 bits).
// What is different: the weights are decoded ONCE per row (a Q4_K block's nibbles and scales into 64-byte vectors, a Q5_1 block's
// five-bit values), not once per token, and the tokens of a tile run on them from the L1: ggml's dot decodes a block for every token.
//
// The caller checks cpu_avx512_ok() (F/BW/VL/DQ/VNNI/VBMI); native_expert.cpp checks the first use against the quant layer's dot.
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int kTile = 6;     // tokens a tile of the row walk runs on the decoded row
constexpr int kMaxNbK = 16;  // Q4_K: blocks of 256 a decoded row may hold (n <= 4096); a longer row goes to the AVX2 kernel
constexpr int kMaxNb32 = 128;  // Q5_1 / Q8_0: blocks of 32 (n <= 4096)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}
// ggml's hsum_float_8 (ggml-cpu/arch/x86/quants.c), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    return _mm256_and_si256(_mm256_set1_epi8(0xF), bytes);
}

// ---- Q4_K x Q8_K.  A decoded block: the four 64-byte weight vectors (low nibbles of 32 bytes | high nibbles of the same 32 bytes, so a
// vector lines up with 64 consecutive activation bytes), the four scale vectors (lanes 0-7 the scale of the low 32 values, lanes 8-15 the
// high 32's), the eight mins as i16, d and dmin.
struct alignas(64) Q4KExp {
    __m512i w[4];
    __m512i s[4];
    __m128i mins;
    float xd, xdmin;
};

inline void q4k_expand(const block_q4_K* x, int nb, Q4KExp* e) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    for (int i = 0; i < nb; ++i) {
        e[i].xd = half_at(&x[i], 0);
        e[i].xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        uint8_t sc[16];
        std::memcpy(sc, utmp, 16);
        e[i].mins = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (sc + 8)));
        for (int j = 0; j < 4; ++j) {
            e[i].s[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(sc[2 * j])), _mm256_set1_epi32(sc[2 * j + 1]), 1);
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            e[i].w[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(q4bits, m4)),
                                           _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4), 1);
        }
    }
}

template <int NT>
inline void q4k_tile(const Q4KExp* e, int nb, const void* const* act, const __m128i* q8s_all, float* res) {
    __m256 acc[NT];
    __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const Q4KExp& b = e[i];
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * b.xd;
            const float dmin = -y->d * b.xdmin;
            const __m128i q8s = q8s_all[(size_t) t * (size_t) nb + (size_t) i];   // ggml's hadd of the 16 bsums, once a call (q8s_prep)
            const __m128i prod = _mm_madd_epi16(b.mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            const __m512i y0 = _mm512_loadu_si512((const void*) (y->qs));
            const __m512i y1 = _mm512_loadu_si512((const void*) (y->qs + 64));
            const __m512i y2 = _mm512_loadu_si512((const void*) (y->qs + 128));
            const __m512i y3 = _mm512_loadu_si512((const void*) (y->qs + 192));
            __m512i a16 = _mm512_dpwssd_epi32(zero, b.s[0], _mm512_dpbusd_epi32(zero, b.w[0], y0));
            a16 = _mm512_dpwssd_epi32(a16, b.s[1], _mm512_dpbusd_epi32(zero, b.w[1], y1));
            a16 = _mm512_dpwssd_epi32(a16, b.s[2], _mm512_dpbusd_epi32(zero, b.w[2], y2));
            a16 = _mm512_dpwssd_epi32(a16, b.s[3], _mm512_dpbusd_epi32(zero, b.w[3], y3));
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(a16), _mm512_extracti64x4_epi64(a16, 1));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

inline void q4k_tile_n(int k, const Q4KExp* e, int nb, const void* const* act, const __m128i* q8s, float* res) {
    switch (k) {
        case 1: q4k_tile<1>(e, nb, act, q8s, res); break;
        case 2: q4k_tile<2>(e, nb, act, q8s, res); break;
        case 3: q4k_tile<3>(e, nb, act, q8s, res); break;
        case 4: q4k_tile<4>(e, nb, act, q8s, res); break;
        case 5: q4k_tile<5>(e, nb, act, q8s, res); break;
        default: q4k_tile<6>(e, nb, act, q8s, res); break;
    }
}

// ---- Q5_1 x Q8_1: ggml's AVX2 dot per token with the integer lanes from vpdpbusd (256-bit: a block's 32 values are one vector)
struct Q51Exp {
    __m256i qx;
    float dx, xm;
};

inline void q51_expand(const block_q5_1* x, int nb, Q51Exp* e) {
    for (int ib = 0; ib < nb; ++ib) {
        e[ib].dx = half_at(&x[ib], 0);
        e[ib].xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        e[ib].qx = _mm256_or_si256(qx, bxhi);
    }
}

template <int NT>
inline void q51_tile(const Q51Exp* e, int nb, const void* const* act, const float* yt, float* res) {
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    const __m256i zero = _mm256_setzero_si256();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(e[ib].dx);
        const float xm = e[ib].xm;
        const __m256i qx = e[ib].qx;
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            const float* yv = yt + 2 * ((size_t) t * (size_t) nb + (size_t) ib);   // d and s of the activation block, converted once a call (q51_prep)
            summs[t] += xm * yv[1];
            const __m256 dy = _mm256_set1_ps(yv[0]);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero, qx, qy));
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

inline void q51_tile_n(int k, const Q51Exp* e, int nb, const void* const* act, const float* yt, float* res) {
    switch (k) {
        case 1: q51_tile<1>(e, nb, act, yt, res); break;
        case 2: q51_tile<2>(e, nb, act, yt, res); break;
        case 3: q51_tile<3>(e, nb, act, yt, res); break;
        case 4: q51_tile<4>(e, nb, act, yt, res); break;
        case 5: q51_tile<5>(e, nb, act, yt, res); break;
        default: q51_tile<6>(e, nb, act, yt, res); break;
    }
}

// ---- Q8_0 x Q8_0
struct Q80Exp {
    __m256i qx, ax;
    float xd;
};

inline void q80_expand(const block_q8_0* x, int nb, Q80Exp* e) {
    for (int ib = 0; ib < nb; ++ib) {
        e[ib].xd = h2f(x[ib].d);
        e[ib].qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        e[ib].ax = _mm256_sign_epi8(e[ib].qx, e[ib].qx);
    }
}

template <int NT>
inline void q80_tile(const Q80Exp* e, int nb, const void* const* act, const float* yd, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const __m256i zero = _mm256_setzero_si256();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = e[ib].xd;
        const __m256i qx = e[ib].qx, ax = e[ib].ax;
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * yd[(size_t) t * (size_t) nb + (size_t) ib]);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(zero, ax, sy));
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

inline void q80_tile_n(int k, const Q80Exp* e, int nb, const void* const* act, const float* yd, float* res) {
    switch (k) {
        case 1: q80_tile<1>(e, nb, act, yd, res); break;
        case 2: q80_tile<2>(e, nb, act, yd, res); break;
        case 3: q80_tile<3>(e, nb, act, yd, res); break;
        case 4: q80_tile<4>(e, nb, act, yd, res); break;
        case 5: q80_tile<5>(e, nb, act, yd, res); break;
        default: q80_tile<6>(e, nb, act, yd, res); break;
    }
}

// The part of a Q4_K token's mins term that does not depend on the row: ggml's hadd of a block's 16 bsums (eight 32-value sums), once a call for every token and block (q8s[t * nb + i]),
// not once a row.  A call of up to 8 tokens and 32 blocks keeps them on the stack.
constexpr size_t kQ8sStack = 256;
inline void q8s_prep(const void* const* act, int nt, int nb, __m128i* q8s) {
    for (int t = 0; t < nt; ++t)
        for (int i = 0; i < nb; ++i) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            q8s[(size_t) t * (size_t) nb + (size_t) i] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
}

// The same for the Q8_1 / Q8_0 activation blocks of the down rows: their fp16 scales (d, and s of Q8_1) as floats, once a call (the conversion is exact: any order gives the bits).
constexpr size_t kYStack = 1024;
inline void q51_prep(const void* const* act, int nt, int nb, float* yt) {
    for (int t = 0; t < nt; ++t)
        for (int ib = 0; ib < nb; ++ib) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            yt[2 * ((size_t) t * (size_t) nb + (size_t) ib)] = half_at(y, 0);
            yt[2 * ((size_t) t * (size_t) nb + (size_t) ib) + 1] = half_at(y, 1);
        }
}
inline void q80_prep(const void* const* act, int nt, int nb, float* yd) {
    for (int t = 0; t < nt; ++t)
        for (int ib = 0; ib < nb; ++ib) yd[(size_t) t * (size_t) nb + (size_t) ib] = h2f(((const block_q8_0*) act[t] + ib)->d);
}

}  // namespace

// (kq512_supported is native_expert.cpp's, next to kq256_supported: an answer that is only a comparison must not come from a wide-ISA file, #795)

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    if (type != 12 || nb > kMaxNbK) {
        kq256_gu_rows(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
        return;
    }
    Q4KExp eg[kMaxNbK], eu[kMaxNbK];
    __m128i q8s_stack[kQ8sStack];
    const size_t need = (size_t) nt * (size_t) nb;
    __m128i* q8s = need <= kQ8sStack ? q8s_stack : (__m128i*) std::malloc(need * sizeof(__m128i));
    q8s_prep(act, nt, nb, q8s);
    for (int r = r0; r < r1; ++r) {
        q4k_expand((const block_q4_K*) (blob + (size_t) r * gu_row), nb, eg);
        q4k_expand((const block_q4_K*) (blob + up_off + (size_t) r * gu_row), nb, eu);
        for (int t0 = 0; t0 < nt; t0 += kTile) {
            const int k = nt - t0 < kTile ? nt - t0 : kTile;
            float g[kTile], u[kTile];
            q4k_tile_n(k, eg, nb, act + t0, q8s + (size_t) t0 * (size_t) nb, g);
            q4k_tile_n(k, eu, nb, act + t0, q8s + (size_t) t0 * (size_t) nb, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token path
            for (int t = 0; t < k; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
    if (q8s != q8s_stack) std::free(q8s);
}

void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    if (type == 12) {
        const int nb = n / QK_K;
        if (nb > kMaxNbK) { kq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1); return; }
        Q4KExp e[kMaxNbK];
        __m128i q8s_stack[kQ8sStack];
        const size_t need = (size_t) nt * (size_t) nb;
        __m128i* q8s = need <= kQ8sStack ? q8s_stack : (__m128i*) std::malloc(need * sizeof(__m128i));
        q8s_prep(act, nt, nb, q8s);
        for (int r = r0; r < r1; ++r) {
            q4k_expand((const block_q4_K*) (w + (size_t) r * row_bytes), nb, e);
            for (int t0 = 0; t0 < nt; t0 += kTile) {
                const int k = nt - t0 < kTile ? nt - t0 : kTile;
                float s[kTile];
                q4k_tile_n(k, e, nb, act + t0, q8s + (size_t) t0 * (size_t) nb, s);
                for (int t = 0; t < k; ++t) out[t0 + t][r] = s[t];
            }
        }
        if (q8s != q8s_stack) std::free(q8s);
        return;
    }
    const int nb = n / 32;
    if ((type != 7 && type != 8) || nb > kMaxNb32) {
        kq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    if (type == 7) {
        Q51Exp e[kMaxNb32];
        float yt_stack[kYStack];
        const size_t need = 2 * (size_t) nt * (size_t) nb;
        float* yt = need <= kYStack ? yt_stack : (float*) std::malloc(need * sizeof(float));
        q51_prep(act, nt, nb, yt);
        for (int r = r0; r < r1; ++r) {
            q51_expand((const block_q5_1*) (w + (size_t) r * row_bytes), nb, e);
            for (int t0 = 0; t0 < nt; t0 += kTile) {
                const int k = nt - t0 < kTile ? nt - t0 : kTile;
                float s[kTile];
                q51_tile_n(k, e, nb, act + t0, yt + 2 * (size_t) t0 * (size_t) nb, s);
                for (int t = 0; t < k; ++t) out[t0 + t][r] = s[t];
            }
        }
        if (yt != yt_stack) std::free(yt);
    } else {
        Q80Exp e[kMaxNb32];
        float yd_stack[kYStack];
        const size_t need = (size_t) nt * (size_t) nb;
        float* yd = need <= kYStack ? yd_stack : (float*) std::malloc(need * sizeof(float));
        q80_prep(act, nt, nb, yd);
        for (int r = r0; r < r1; ++r) {
            q80_expand((const block_q8_0*) (w + (size_t) r * row_bytes), nb, e);
            for (int t0 = 0; t0 < nt; t0 += kTile) {
                const int k = nt - t0 < kTile ? nt - t0 : kTile;
                float s[kTile];
                q80_tile_n(k, e, nb, act + t0, yd + (size_t) t0 * (size_t) nb, s);
                for (int t = 0; t < k; ++t) out[t0 + t][r] = s[t];
            }
        }
        if (yd != yd_stack) std::free(yd);
    }
}

}  // namespace strata::kernels::cpu
