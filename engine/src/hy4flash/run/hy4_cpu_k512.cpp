// hy4_cpu_k512.cpp - AVX-512 BW + VL + VNNI + VBMI + F16C S_K CPU expert kernels for Hy4-Flash.
// Kept in a separate translation unit from hy4_cpu_experts.cpp so MSVC compiles THIS TU with /arch:AVX512
// while hy4_cpu_experts.cpp compiles with /arch:AVX2 (zero EVEX 0x62 instructions in AVX2/AVX-VNNI/dispatch).
// All constant tables are constexpr (pure .rdata, zero dynamic initializers executed before CPUID dispatch).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#include "hy4_cpu_k512.h"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(HY4_NO_FAST)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#if defined(__clang__)
#define HY4_K512_PUSH _Pragma("clang attribute push (__attribute__((target(\"avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,f16c,fma,avx2\"))), apply_to=function)")
#define HY4_K512_POP  _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#define HY4_K512_PUSH _Pragma("GCC push_options") _Pragma("GCC target(\"avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,f16c,fma,avx2\")")
#define HY4_K512_POP  _Pragma("GCC pop_options")
#else
#define HY4_K512_PUSH
#define HY4_K512_POP
#endif

namespace hy4 {
namespace k512 {
namespace {

using strata::quant::sk::BlockQ8_1_G64;

struct FastConsts {
    alignas(64) uint8_t idx4[4][64]{};     // vpermi2b index: plane j of the 128-byte nibble array (S4/S5/S6)
    alignas(64) uint8_t ctrl3[4][64]{};    // vpmultishiftqb control: bit n -> bit 2 of the byte (S3 high plane)
    alignas(64) uint8_t ctrl5[4][64]{};    // ... -> bit 4 (S5 high plane)
    alignas(64) uint8_t tr[64]{};          // store_block: dst[16*j+k] = src[4*k+j]
    alignas(64) int32_t effidx[16]{};
    alignas(64) int64_t qidx[8]{};
    alignas(64) uint8_t cb125[4][64]{};    // S1.25: codebook column c (+2 bias -> unsigned 0..4), 32 entries, vpermb table
    alignas(64) uint8_t ctrl125[64]{};     // S1.25: multishift control, byte b -> 5*(b&7)
    alignas(64) uint8_t perm125[64]{};     // S1.25: dest byte 8g+i <- payload byte 5g+i (i < 5)

    constexpr FastConsts() {
        constexpr int8_t cbk[32][4] = {
            {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
            {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
            {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
            {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0}};
        for (int c = 0; c < 4; ++c) {
            for (int e = 0; e < 32; ++e) cb125[c][e] = static_cast<uint8_t>(cbk[e][c] + 2);
            for (int e = 32; e < 64; ++e) cb125[c][e] = 0;
        }
        for (int b = 0; b < 64; ++b) {
            ctrl125[b] = static_cast<uint8_t>(5 * (b & 7));
            perm125[b] = static_cast<uint8_t>((b >> 3) * 5 + (b & 7) < 40 ? (b >> 3) * 5 + (b & 7) : 0);
        }
        for (int j = 0; j < 4; ++j)
            for (int b = 0; b < 64; ++b) {
                const int L = b >> 4, k = b & 15, q = k >> 3, kk = k & 7;
                idx4[j][b] = static_cast<uint8_t>(32 * L + 2 * k + (j >> 1));
                ctrl3[j][b] = static_cast<uint8_t>((32 * q + 4 * kk + j - 2) & 63);
                ctrl5[j][b] = static_cast<uint8_t>((32 * q + 4 * kk + j - 4) & 63);
            }
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 16; ++k) tr[16 * j + k] = static_cast<uint8_t>(4 * k + j);
        for (int i = 0; i < 16; ++i) effidx[i] = i >> 2;
        for (int i = 0; i < 8; ++i) qidx[i] = i >> 1;
    }
};
static constexpr FastConsts kc{};

HY4_K512_PUSH

#define HY4_HI3(J) _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_load_si512((const void*)kc.ctrl3[J]), src), m4)
#define HY4_HI5(J) _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_load_si512((const void*)kc.ctrl5[J]), src), m16)
#define HY4_NIB(J) _mm512_permutex2var_epi8(n0, _mm512_load_si512((const void*)kc.idx4[J]), n1)

template <int FMT>
static inline void unpack_planes(const uint8_t* pl, __m512i (&P)[4]) {
    const __m512i m3 = _mm512_set1_epi8(3), m4 = _mm512_set1_epi8(4), m15 = _mm512_set1_epi8(15), m16 = _mm512_set1_epi8(16),
                  m30 = _mm512_set1_epi8(0x30);
    (void)m3; (void)m4; (void)m16; (void)m30;
    if constexpr (FMT == 0) {
        const __m512i raw = _mm512_maskz_loadu_epi8((__mmask64)0xFFFFFFFFFFull, (const void*)pl);
        const __m512i q40 = _mm512_maskz_permutexvar_epi8((__mmask64)0x1F1F1F1F1F1F1F1Full, _mm512_load_si512((const void*)kc.perm125), raw);
        const __m512i idx = _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_load_si512((const void*)kc.ctrl125), q40), _mm512_set1_epi8(31));
        P[0] = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)kc.cb125[0]));
        P[1] = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)kc.cb125[1]));
        P[2] = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)kc.cb125[2]));
        P[3] = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)kc.cb125[3]));
    } else if constexpr (FMT == 2 || FMT == 3) {
        const __m512i w = _mm512_loadu_si512((const void*)pl);
        if constexpr (FMT == 2) {
            P[0] = _mm512_and_si512(w, m3);
            P[1] = _mm512_and_si512(_mm512_srli_epi16(w, 2), m3);
            P[2] = _mm512_and_si512(_mm512_srli_epi16(w, 4), m3);
            P[3] = _mm512_and_si512(_mm512_srli_epi16(w, 6), m3);
        } else {
            const __m512i src = _mm512_permutexvar_epi64(_mm512_load_si512((const void*)kc.qidx), _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(pl + 64))));
            P[0] = _mm512_ternarylogic_epi32(w, HY4_HI3(0), m3, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 2), HY4_HI3(1), m3, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 4), HY4_HI3(2), m3, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 6), HY4_HI3(3), m3, 0xEC);
        }
    } else {
        const __m512i n0 = _mm512_loadu_si512((const void*)pl), n1 = _mm512_loadu_si512((const void*)(pl + 64));
        const __m512i v0 = HY4_NIB(0), v1 = _mm512_srli_epi16(HY4_NIB(1), 4), v2 = HY4_NIB(2), v3 = _mm512_srli_epi16(HY4_NIB(3), 4);
        if constexpr (FMT == 4) {
            P[0] = _mm512_and_si512(v0, m15);
            P[1] = _mm512_and_si512(v1, m15);
            P[2] = _mm512_and_si512(v2, m15);
            P[3] = _mm512_and_si512(v3, m15);
        } else if constexpr (FMT == 5) {
            const __m512i src = _mm512_permutexvar_epi64(_mm512_load_si512((const void*)kc.qidx), _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(pl + 128))));
            P[0] = _mm512_ternarylogic_epi32(v0, HY4_HI5(0), m15, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(v1, HY4_HI5(1), m15, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(v2, HY4_HI5(2), m15, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(v3, HY4_HI5(3), m15, 0xEC);
        } else {
            const __m512i w6 = _mm512_loadu_si512((const void*)(pl + 128));
            P[0] = _mm512_ternarylogic_epi32(v0, _mm512_and_si512(_mm512_slli_epi16(w6, 4), m30), m15, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(v1, _mm512_and_si512(_mm512_slli_epi16(w6, 2), m30), m15, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(v2, _mm512_and_si512(w6, m30), m15, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(v3, _mm512_and_si512(_mm512_srli_epi16(w6, 2), m30), m15, 0xEC);
        }
    }
}

template <>
inline void unpack_planes<8>(const uint8_t* pl, __m512i (&P)[4]) {
    const __m512i tr = _mm512_load_si512((const void*)kc.tr);
    const __m512i x80 = _mm512_set1_epi8((char)0x80);
    const __m512i q0 = _mm512_permutexvar_epi8(tr, _mm512_loadu_si512((const void*)(pl)));
    const __m512i q1 = _mm512_permutexvar_epi8(tr, _mm512_loadu_si512((const void*)(pl + 64)));
    const __m512i q2 = _mm512_permutexvar_epi8(tr, _mm512_loadu_si512((const void*)(pl + 128)));
    const __m512i q3 = _mm512_permutexvar_epi8(tr, _mm512_loadu_si512((const void*)(pl + 192)));
    const __m512i t0 = _mm512_shuffle_i32x4(q0, q1, 0x44);
    const __m512i t1 = _mm512_shuffle_i32x4(q0, q1, 0xEE);
    const __m512i t2 = _mm512_shuffle_i32x4(q2, q3, 0x44);
    const __m512i t3 = _mm512_shuffle_i32x4(q2, q3, 0xEE);
    P[0] = _mm512_xor_si512(_mm512_shuffle_i32x4(t0, t2, 0x88), x80);
    P[1] = _mm512_xor_si512(_mm512_shuffle_i32x4(t0, t2, 0xDD), x80);
    P[2] = _mm512_xor_si512(_mm512_shuffle_i32x4(t1, t3, 0x88), x80);
    P[3] = _mm512_xor_si512(_mm512_shuffle_i32x4(t1, t3, 0xDD), x80);
}

template <int FMT>
struct FmtInfo;
template <> struct FmtInfo<0> { static constexpr int SB = 48;  static constexpr float HALF = 2.0f; };
template <> struct FmtInfo<2> { static constexpr int SB = 70;  static constexpr float HALF = 1.5f; };
template <> struct FmtInfo<3> { static constexpr int SB = 102; static constexpr float HALF = 3.5f; };
template <> struct FmtInfo<4> { static constexpr int SB = 134; static constexpr float HALF = 7.5f; };
template <> struct FmtInfo<5> { static constexpr int SB = 166; static constexpr float HALF = 15.5f; };
template <> struct FmtInfo<6> { static constexpr int SB = 198; static constexpr float HALF = 31.5f; };
template <> struct FmtInfo<8> { static constexpr int SB = 262; static constexpr float HALF = 128.0f; };

static inline float cvt_f16_f32(uint16_t h) {
#ifdef _MSC_VER
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
#else
    return _cvtsh_ss(h);
#endif
}

template <int FMT, int M, bool HI>
static inline void row_dot(const uint8_t* row, const ActRow* const* X, float* out) {
    constexpr int SB = FmtInfo<FMT>::SB;
    __m512 facc[M];
    for (int t = 0; t < M; ++t) facc[t] = _mm512_setzero_ps();
    const __m128 inv255 = _mm_set1_ps(255.0f);
    const __m512 halfv = _mm512_set1_ps(FmtInfo<FMT>::HALF);
    const __m512i effidx = _mm512_load_si512((const void*)kc.effidx);
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = row + (size_t)s * SB;
        uint16_t dh;
        uint32_t u32;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&u32, blk + 2, 4);
        __m128 u4 = _mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)u32))), inv255);
        u4 = _mm_mul_ps(u4, _mm_set1_ps(cvt_f16_f32(dh)));
        if constexpr (FMT == 0) {
            uint16_t fn;
            std::memcpy(&fn, blk + 46, 2);
            __m128i fv = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)fn), _mm_setr_epi32(0, 4, 8, 12)), _mm_set1_epi32(15));
            u4 = _mm_mul_ps(u4, _mm_add_ps(_mm_div_ps(_mm_cvtepi32_ps(fv), _mm_set1_ps(15.0f)), _mm_set1_ps(0.5f)));
        }
        const __m512 effv = _mm512_permutexvar_ps(effidx, _mm512_castps128_ps512(u4));
        const __m512 effh = _mm512_mul_ps(effv, halfv);
        __m512i P[4];
        unpack_planes<FMT>(blk + 6, P);
        for (int t = 0; t < M; ++t) {
            const ActRow* a = X[t];
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), P[0], _mm512_load_si512((const void*)a->xj[s][0]));
            acc = _mm512_dpbusd_epi32(acc, P[1], _mm512_load_si512((const void*)a->xj[s][1]));
            acc = _mm512_dpbusd_epi32(acc, P[2], _mm512_load_si512((const void*)a->xj[s][2]));
            acc = _mm512_dpbusd_epi32(acc, P[3], _mm512_load_si512((const void*)a->xj[s][3]));
            const __m512 sc = _mm512_mul_ps(effv, _mm512_load_ps(a->xd[s]));
            facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), sc, facc[t]);
            if constexpr (HI) {
                __m512i accb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), P[0], _mm512_load_si512((const void*)a->xjb[s][0]));
                accb = _mm512_dpbusd_epi32(accb, P[1], _mm512_load_si512((const void*)a->xjb[s][1]));
                accb = _mm512_dpbusd_epi32(accb, P[2], _mm512_load_si512((const void*)a->xjb[s][2]));
                accb = _mm512_dpbusd_epi32(accb, P[3], _mm512_load_si512((const void*)a->xjb[s][3]));
                facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(accb), _mm512_mul_ps(effv, _mm512_load_ps(a->xdb[s])), facc[t]);
            }
            facc[t] = _mm512_fnmadd_ps(effh, _mm512_load_ps(a->xs4[s]), facc[t]);
        }
    }
    for (int t = 0; t < M; ++t) out[t] = _mm512_reduce_add_ps(facc[t]);
}

template <int FMT, int M, bool HI>
static void rows_m(const uint8_t* W, int nrows, const ActRow* const* X, float* out, int pf) {
    constexpr size_t RBYTES = 6 * (size_t)FmtInfo<FMT>::SB;
    for (int r = 0; r < nrows; ++r) {
        if (pf && r + pf < nrows) {
            const char* pp = (const char*)(W + (size_t)(r + pf) * RBYTES);
            for (size_t o = 0; o < RBYTES; o += 64) _mm_prefetch(pp + o, _MM_HINT_T0);
        }
        float o[M];
        row_dot<FMT, M, HI>(W + r * RBYTES, X, o);
        for (int i = 0; i < M; ++i) out[(size_t)i * nrows + r] = o[i];
    }
}

template <int FMT, bool HI>
static void rows_fmt(const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, int pf) {
    for (int g0 = 0; g0 < m; g0 += 8) {
        const int mg = std::min(8, m - g0);
        const ActRow* const* x = X + g0;
        float* o = out + (size_t)g0 * nrows;
        switch (mg) {
            case 1: rows_m<FMT, 1, HI>(W, nrows, x, o, pf); break;
            case 2: rows_m<FMT, 2, HI>(W, nrows, x, o, pf); break;
            case 3: rows_m<FMT, 3, HI>(W, nrows, x, o, pf); break;
            case 4: rows_m<FMT, 4, HI>(W, nrows, x, o, pf); break;
            case 5: rows_m<FMT, 5, HI>(W, nrows, x, o, pf); break;
            case 6: rows_m<FMT, 6, HI>(W, nrows, x, o, pf); break;
            case 7: rows_m<FMT, 7, HI>(W, nrows, x, o, pf); break;
            default: rows_m<FMT, 8, HI>(W, nrows, x, o, pf); break;
        }
    }
}

HY4_K512_POP

}  // namespace

HY4_K512_PUSH

void store_block(ActRow* a, int b, const BlockQ8_1_G64& blk) {
    const int sb = b >> 2, L = b & 3;
    const __m512i R = _mm512_permutexvar_epi8(_mm512_load_si512((const void*)kc.tr), _mm512_load_si512((const void*)blk.qs));
    _mm_store_si128((__m128i*)&a->xj[sb][0][L * 16], _mm512_castsi512_si128(R));
    _mm_store_si128((__m128i*)&a->xj[sb][1][L * 16], _mm512_extracti32x4_epi32(R, 1));
    _mm_store_si128((__m128i*)&a->xj[sb][2][L * 16], _mm512_extracti32x4_epi32(R, 2));
    _mm_store_si128((__m128i*)&a->xj[sb][3][L * 16], _mm512_extracti32x4_epi32(R, 3));
    _mm_store_ps(&a->xd[sb][L * 4], _mm_set1_ps(blk.d));
    _mm_store_ps(&a->xs4[sb][L * 4], _mm_set1_ps(0.25f * blk.s));
}

void store_block_hi(ActRow* a, int b, const BlockQ8_1_G64& blk, const BlockQ8_1_G64& blk2) {
    const int sb = b >> 2, L = b & 3;
    const __m512i idx = _mm512_load_si512((const void*)kc.tr);
    const __m512i R = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)blk.qs));
    const __m512i R2 = _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)blk2.qs));
    _mm_store_si128((__m128i*)&a->xj[sb][0][L * 16], _mm512_castsi512_si128(R));
    _mm_store_si128((__m128i*)&a->xj[sb][1][L * 16], _mm512_extracti32x4_epi32(R, 1));
    _mm_store_si128((__m128i*)&a->xj[sb][2][L * 16], _mm512_extracti32x4_epi32(R, 2));
    _mm_store_si128((__m128i*)&a->xj[sb][3][L * 16], _mm512_extracti32x4_epi32(R, 3));
    _mm_store_si128((__m128i*)&a->xjb[sb][0][L * 16], _mm512_castsi512_si128(R2));
    _mm_store_si128((__m128i*)&a->xjb[sb][1][L * 16], _mm512_extracti32x4_epi32(R2, 1));
    _mm_store_si128((__m128i*)&a->xjb[sb][2][L * 16], _mm512_extracti32x4_epi32(R2, 2));
    _mm_store_si128((__m128i*)&a->xjb[sb][3][L * 16], _mm512_extracti32x4_epi32(R2, 3));
    _mm_store_ps(&a->xd[sb][L * 4], _mm_set1_ps(blk.d));
    _mm_store_ps(&a->xdb[sb][L * 4], _mm_set1_ps(blk2.d));
    _mm_store_ps(&a->xs4[sb][L * 4], _mm_set1_ps(0.25f * (blk.s + blk2.s)));
}

void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi, int pf_rows) {
    if (hi) {
        switch (code) {
            case 0: rows_fmt<0, true>(W, nrows, X, m, out, pf_rows); break;
            case 2: rows_fmt<2, true>(W, nrows, X, m, out, pf_rows); break;
            case 3: rows_fmt<3, true>(W, nrows, X, m, out, pf_rows); break;
            case 4: rows_fmt<4, true>(W, nrows, X, m, out, pf_rows); break;
            case 5: rows_fmt<5, true>(W, nrows, X, m, out, pf_rows); break;
            case 8: rows_fmt<8, true>(W, nrows, X, m, out, pf_rows); break;
            default: rows_fmt<6, true>(W, nrows, X, m, out, pf_rows); break;
        }
        return;
    }
    switch (code) {
        case 0: rows_fmt<0, false>(W, nrows, X, m, out, pf_rows); break;
        case 2: rows_fmt<2, false>(W, nrows, X, m, out, pf_rows); break;
        case 3: rows_fmt<3, false>(W, nrows, X, m, out, pf_rows); break;
        case 4: rows_fmt<4, false>(W, nrows, X, m, out, pf_rows); break;
        case 5: rows_fmt<5, false>(W, nrows, X, m, out, pf_rows); break;
        case 8: rows_fmt<8, false>(W, nrows, X, m, out, pf_rows); break;
        default: rows_fmt<6, false>(W, nrows, X, m, out, pf_rows); break;
    }
}

HY4_K512_POP

}  // namespace k512
}  // namespace hy4

#endif  // x86_64 && !HY4_NO_FAST
