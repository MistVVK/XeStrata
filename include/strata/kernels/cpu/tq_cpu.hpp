// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/tq_cpu.hpp - the trellis expert formats (TQ2_T / TQK6 / TQK7 / TQP5 / TQP6 / TQP7, ggml
// types 144-149 of the agention llama.cpp fork) on the CPU: decode-and-dot against Q8_0 activations (ggml's
// block_q8_0), one or several tokens per call, on AVX-512, AVX2 or plain C++.
//
// THE FORMAT (third_party/ggml/ggml-common-tq.h): a 128-weight block is {fp16 d; 32 steps of K bits} (K = 8 / 6 / 7),
// or for TQP {fp16 d; 64 steps of K bits} (K = 5 / 6 / 7).  Step t has a 15-bit state; the state's hash picks two
// points (pairs) of a 2048-point fp16 codebook, which are weights 4t..4t+3, times d; a TQP step takes the first point
// only, weights 2t, 2t+1.
//
// THE DECODE: a 32768 x 4 int16 table maps a state straight to its four weights at scale 2^13 (256 KiB, the
// approach of the fork's ggml-cpu vec_dot): one 8-byte lookup per four weights (a TQP step reads its first 4 bytes),
// no hash.  The int16 rounding is
// exact for codebook values >= 2^-3 in magnitude and within 2^-14 below that; it is the same table, rounded the
// same way, as the fork's ggml-cpu kernels.
//
// THE ARITHMETIC, identical on every ISA so every path gives the same bits (and so does any token-group size):
//   S[i][k] = sum_j w16[i][32k + j] * q8[i][k][j]        exact in int32 (|w16| < 2^15, |q8| <= 127)
//   acc[k]  = fma((float) S[i][k], d_w[i] * d_a[i][k], acc[k])      for blocks i in order, k = 0..3
//   dot     = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * 2^-13
// i.e. four float accumulators, one per Q8_0 sub-block position, each a fixed sequential chain.  It differs from
// ggml-cpu's vec_dot (same integers, other float summation order) by float rounding only.
//
// THE KERNELS: `tq_isa()` picks AVX-512 (F/BW/VL/VBMI/VNNI, cpu_avx512_ok) or AVX2 (cpu_avx2_ok) at run time; the
// scalar kernel is the reference (on a CPU without AVX2 the engine keeps ggml-cpu's vec_dot, see tq_kernels_enabled).
// STRATA_TQ_ISA=scalar|avx2|avx512 caps the choice (tests, A/B); STRATA_NO_TQ_KERNELS=1 makes native_expert.cpp use
// ggml-cpu's vec_dot instead (the previous path).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

enum class TqIsa : int { scalar = 0, avx2 = 1, avx512 = 2 };

/// TQ2_T (144), TQK6 (145), TQK7 (146), TQP5 (147), TQP6 (148), TQP7 (149).
bool tq_supported(int ggml_type) noexcept;
/// Bits per step: 8, 6, 7, 5, 6, 7 (0 for any other type).
int tq_kbits(int ggml_type) noexcept;
/// Steps per block: 32 (TQ2_T, TQK), 64 (TQP), 0 for any other type.
inline int tq_steps(int ggml_type) noexcept {
    return ggml_type >= 144 && ggml_type <= 146 ? 32 : ggml_type >= 147 && ggml_type <= 149 ? 64 : 0;
}
/// Values per block of the trellis types, and bytes per block (2 + steps * bits / 8: 34, 26, 30, 42, 50, 58).
constexpr int kTqBlock = 128;
inline size_t tq_block_bytes(int ggml_type) noexcept {
    return 2u + (size_t) tq_steps(ggml_type) * (size_t) tq_kbits(ggml_type) / 8u;
}
/// The best ISA this CPU runs (capped by STRATA_TQ_ISA).
TqIsa tq_isa();
/// Whether this CPU can run `isa` at all (scalar: always).
bool tq_isa_ok(TqIsa isa);
/// Whether STRATA_NO_TQ_KERNELS=1 is absent.  The engine's native expert rows take these kernels for the trellis types
/// on every CPU (AVX-512 or AVX2 where the CPU has them, else the scalar kernel: a ggml from upstream has no trellis
/// type to fall back to, and the AVX code is only reached behind tq_isa_ok).  STRATA_NO_TQ_KERNELS=1 asks for
/// ggml-cpu's vec_dot instead, honoured only by a ggml that has the types (the agentionai fork; tests, A/B).
bool tq_kernels_enabled();
const char* tq_isa_name(TqIsa isa);

/// The 32768 x 4 int16 state table (filled on first use; thread-safe).
const int16_t* tq_state_table();

/// Rows [r0, r1) of a trellis matrix (`row_bytes` per row, `n` weights per row) against `nt` (<= 8) Q8_0
/// activations of n values: out[t][r] = w_r . a[t].
void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1);
/// Gate and up rows [r0, r1) (gate at blob, up at blob + up_off): ff[t][r] = silu(g) * u, the SwiGLU expression
/// of native_gu_rows.
void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1);

/// R1Q2 scales for one matrix. Half-precision values are stored little-endian in the blob; q packs four
/// two-bit indices per byte, separately for each row. p2 selects the layer and projection's scale table.
struct TqR1q2Scales {
    const uint8_t* rowbase;
    const uint8_t* colscale;
    const uint8_t* q;
    size_t qbytes;
    float p2[4];
};

/// The bih3 types 182..194 use K=2..14 and 4*K code bytes per 128 weights, without a block scale.
/// Bih3 chooses its ISA separately from the original trellis formats, by CPUID feature bits and the token
/// count (a version is taken only where it measured faster than the one below it; llvm-mca and row
/// benches, 2026-11-19).  The explicit entry is for parity tests.
enum class Bih3Isa : int { scalar, avx2, avxvnni, avx512vnni };
bool bih3_isa_ok(Bih3Isa isa);
Bih3Isa bih3_isa(int nt);
const char* bih3_isa_name(Bih3Isa isa);
void tq_bih3_r1q2_rows_isa(Bih3Isa isa, int type, const uint8_t* w, size_t row_bytes, int n,
                            const void* const* act, int nt, float* const* out, int r0, int r1,
                            const TqR1q2Scales& scales);
void tq_bih3_r1q2_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1, const TqR1q2Scales& scales);
void tq_bih3_r1q2_gu_rows_isa(Bih3Isa isa, int type, const uint8_t* blob, size_t row_bytes, size_t up_off, int n,
                               const void* const* act, int nt, float* const* ff, int r0, int r1,
                               const TqR1q2Scales& gate, const TqR1q2Scales& up);
void tq_bih3_r1q2_gu_rows(int type, const uint8_t* blob, size_t row_bytes, size_t up_off, int n,
                           const void* const* act, int nt, float* const* ff, int r0, int r1,
                           const TqR1q2Scales& gate, const TqR1q2Scales& up);

/// Largest token group per call and widest row the kernels take.
inline constexpr int kTqMaxTokens = 8;
inline constexpr int kTqMaxN = 4096;

namespace tq_detail {
/// One token's Q8_0 activation unpacked for the kernels: the int8 values widened to int16 and the scales as floats.
struct TqAct {
    alignas(64) int16_t q[kTqMaxN];
    alignas(64) float d[kTqMaxN / 32];
};
void unpack_act(const void* q8_0, int n, TqAct& a);
#if defined(__x86_64__) || defined(_M_X64)
void bih3_rows_x86(Bih3Isa isa, int type, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                    int nt, float* const* out, int r0, int r1, const TqR1q2Scales& scales);
#endif
// the per-ISA row kernels (dot products only; `out[t][r]` for rows [r0, r1)); `ty` the ggml type
void tq_rows_scalar(int ty, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_avx2(int ty, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1);
void tq_rows_avx512(int ty, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
