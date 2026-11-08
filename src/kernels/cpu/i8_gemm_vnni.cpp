// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/i8_gemm_vnni.cpp - the AVX2 activation quantizer and the vector copies of the int8 x int8 matrix
// product (include/strata/kernels/cpu/i8_gemm.hpp): the AVX-VNNI kernel and the AVX2-only one.  Compiled with
// -mavx2 -mfma like the BF16 kernel; the VNNI intrinsics live in functions marked target("avxvnni") (CMakeLists.txt:
// never -mavxvnni on a whole file, which would let the compiler use the extension anywhere), so nothing here runs
// them before i8_gemm_isa_ok() has said the CPU does.  Where the build's compiler has no AVX-VNNI intrinsics
// (STRATA_AVXVNNI 0) the VNNI copy is not built and the AVX2 one runs in its place.
//
// THE SHAPE OF THE INNER LOOP.  One panel - eight outputs, which are the eight int32 lanes of a `vpdpbusd`
// operand - against up to eight token rows.  A quad of four k values of the panel is exactly 32 bytes, so one
// load brings the eight lanes' weights; the activation is those same four bytes in every lane, so one dword
// broadcast per token.  Per quad and eight tokens: eight `vpdpbusd` - 256 int8 products.  A lane is one output and
// is never reduced with another, so one output's sum is the integer chain the scalar reference runs; integer
// addition is associative and exact, so the result does not depend on the order of k, on the token group, on the
// panel band or on the thread count (the header).
//
// THE AVX2 COPY multiplies the signed levels: the activation byte with its +128 taken off (an xor of the top bit),
// `vpmaddubsw` on its magnitude and the weight given its sign (`vpsignb`), then `vpmaddwd` with ones to sum the
// pairs into the int32 lanes.  Its sum is sum qa * qw, the same integer the VNNI copy reaches by subtracting the
// row-sum correction, so the two never differ.  It costs five operations where VNNI's costs one.  On seven-bit
// weights (the header's THE WEIGHT LEVELS) it takes the unsigned byte as it is - `vpmaddubsw`, `vpmaddwd`, add - and
// subtracts the correction as VNNI does: three operations.
//
// THE DEQUANTIZATION IS ONE PASS PER OUTPUT AND BLOCK and the only float arithmetic in the kernel: the int32 sum to
// FP32, times the token's and the output's scales, folded into the output's running value with `vfmadd` - the same
// operations in the same order as the reference's.  With scale blocks the running value of each row stays in a
// register across the blocks, so a block's group of rows is shorter (kBlockRows) than the one-block group of eight:
// eight int32 and eight FP32 accumulators would not fit the sixteen registers.
#include "strata/kernels/cpu/i8_gemm.hpp"

#include "strata/kernels/cpu/bf16_gemm.hpp"   // bf16_gemm_token_block, the block rule both paths share

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics); 0 builds the AVX2 copies only.
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_I8_VNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_I8_VNNI_FN
#endif
// STRATA_AVXVNNIINT8 (CMake: the compiler has the AVX-VNNI-INT8 intrinsics); 0 builds no such copy.
#if !defined(STRATA_AVXVNNIINT8)
#define STRATA_AVXVNNIINT8 0
#endif
#if STRATA_AVXVNNIINT8 && (defined(__GNUC__) || defined(__clang__))
#define STRATA_I8_VNNIINT8_FN __attribute__((target("avxvnniint8")))
#else
#define STRATA_I8_VNNIINT8_FN
#endif

namespace strata::kernels::cpu::i8_gemm_detail {
namespace {

/// The token rows a group takes when the matrix has more than one scale block (see the file's comment).
constexpr int kBlockRows = 4;
/// The token rows of a two-panel group (tile2): eight int32 accumulators in flight, enough to cover vpdpbusd's
/// latency (three rows, six chains: 120 ms for the 2B's 2048 x 6144 x 1692 on one P-core of an i7-14700; four: 100 ms;
/// five: no faster.  The FP32 running values go to the stack around the k loop.  2026-11-08)
constexpr int kPairRows = 4;

/// Four bytes of a quantized activation row: the unsigned operand is that dword in every lane.
inline uint32_t quad_bytes(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

/// The FP32 rounding mode the reference's std::nearbyintf uses - round to nearest, ties to even, which is what
/// `cvtps_epi32` applies with the MXCSR the process starts with - then clamp to the levels a lane holds.
inline __m256i levels8(__m256 v, __m256 inv) {
    const __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v, inv));
    return _mm256_max_epi32(_mm256_min_epi32(q, _mm256_set1_epi32(kI8Levels)), _mm256_set1_epi32(-kI8Levels));
}

/// Sixteen int32 lanes to sixteen bytes: int32 -> int16 -> int8, then the +128 that puts the levels on the
/// unsigned side.  Both `packs` interleave by 128-bit halves, so the lanes are put back in order - the qwords
/// after the first pack, the two 128-bit halves after the second.  Adding 128 to an int8 wraps at 255, which is
/// exactly the byte the header's `qu` is: -127 -> 1, 0 -> 128, +127 -> 255.
inline __m128i pack16(__m256i a, __m256i b, bool sign) {
    __m256i w = _mm256_packs_epi32(a, b);                 // a0..3 b0..3 a4..7 b4..7
    w = _mm256_permute4x64_epi64(w, 0xD8);               // a0..7 b0..7
    const __m256i q = _mm256_packs_epi16(w, w);          // lane 0: a twice, lane 1: b twice
    const __m128i first = _mm256_castsi256_si128(q);     // its low eight bytes are a
    const __m128i second = _mm256_extracti128_si256(q, 1);   // its low eight bytes are b
    const __m128i r = _mm_unpacklo_epi64(first, second); // a0..7 b0..7
    return sign ? r : _mm_add_epi8(r, _mm_set1_epi8((char) 128));
}

// ---- the row kernels, twice: the AVX2 forms and the AVX-VNNI ones (i8_gemm_rows.inl)
namespace avx2 {
#define STRATA_I8_ROWS_VNNI 0
#define STRATA_I8_ROWS_U8 0
#define STRATA_I8_ROWS_FN
#include "i8_gemm_rows.inl"
}  // namespace avx2
namespace avx2_w7 {
#define STRATA_I8_ROWS_VNNI 0
#define STRATA_I8_ROWS_U8 1
#define STRATA_I8_ROWS_FN
#include "i8_gemm_rows.inl"
}  // namespace avx2_w7
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_I8_ROWS_VNNI 1
#define STRATA_I8_ROWS_U8 1
#define STRATA_I8_ROWS_FN STRATA_I8_VNNI_FN
#include "i8_gemm_rows.inl"
}  // namespace vnni
#endif
#if STRATA_AVXVNNIINT8
namespace vnni_int8 {
#define STRATA_I8_ROWS_VNNI 2
#define STRATA_I8_ROWS_U8 0
#define STRATA_I8_ROWS_FN STRATA_I8_VNNIINT8_FN
#include "i8_gemm_rows.inl"
}  // namespace vnni_int8
#endif

}  // namespace

void i8_gemm_quantize_avx2(const float* x, size_t ldx, int n, int n_in, uint8_t* xq, size_t ldq, float* xs,
                           int block, bool sign) {
    const int row_bytes = i8_gemm_act_row_bytes(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int) 0x80000000u));
    for (int t = 0; t < n; ++t) {
        const float* r = x + (size_t) t * ldx;
        uint8_t* q = xq + (size_t) t * ldq;
        for (int b = 0; b < nb; ++b) {
            const int k0 = block <= 0 ? 0 : b * block;
            const int k1 = block <= 0 ? n_in : std::min(n_in, k0 + block);
            // The block's maximum.  max() is exact and every value is finite in a forward that has not gone wrong,
            // so the vector fold and the reference's loop over k in index order give the same FP32 number.
            __m256 m8 = _mm256_setzero_ps();
            int k = k0;
            for (; k + 8 <= k1; k += 8) m8 = _mm256_max_ps(m8, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(r + k)));
            alignas(32) float mf[kI8Panel];
            _mm256_store_ps(mf, m8);
            float amax = 0.0f;
            for (int i = 0; i < kI8Panel; ++i) amax = std::max(amax, mf[i]);
            for (; k < k1; ++k) amax = std::max(amax, std::fabs(r[k]));
            const float scale = amax / (float) kI8Levels;
            const float inv = amax > 0.0f ? (float) kI8Levels / amax : 0.0f;
            xs[(size_t) t * nb + b] = scale;
            const __m256 inv8 = _mm256_set1_ps(inv);
            k = k0;
            for (; k + 16 <= k1; k += 16) {
                const __m256i a = levels8(_mm256_loadu_ps(r + k), inv8);
                const __m256i c = levels8(_mm256_loadu_ps(r + k + 8), inv8);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(q + k), pack16(a, c, sign));
            }
            for (; k < k1; ++k) {
                int v = (int) std::nearbyintf(r[k] * inv);
                if (v > kI8Levels) v = kI8Levels;
                if (v < -kI8Levels) v = -kI8Levels;
                q[k] = sign ? (uint8_t) (int8_t) v : (uint8_t) (v + 128);
            }
        }
        std::memset(q + n_in, 0, (size_t) (row_bytes - n_in));   // the last quad's padding
    }
}

void i8_gemm_rows_x86(int copy, const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                      const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok, int p0, int p1,
                      const float* bias, int block, int levels) {
#if STRATA_AVXVNNIINT8
    if (copy == 2) {
        vnni_int8::rows_x86(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias, block);
        return;
    }
#endif
#if STRATA_AVXVNNI
    if (copy == 1) {
        vnni::rows_x86(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias, block);
        return;
    }
#endif
    (void) copy;
    if (levels <= kI8Levels7) {
        avx2_w7::rows_x86(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias, block);
        return;
    }
    avx2::rows_x86(panel, wscale, wcorr, n_in, n_out, xq, ldq, xs, y, ldy, n_tok, p0, p1, bias, block);
}

}  // namespace strata::kernels::cpu::i8_gemm_detail
