// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/bf16_gemm.hpp - the CPU's matrix product for a BF16 weight and FP32 activations:
// the EmbeddingGemma 2 forward (src/embed/) on every x86-64 with AVX2, and the scalar reference that goes with it.
//
// WHY OUR OWN, when the engine already has `bf16_gemv` and ggml has a CPU matmul.  `bf16_gemv` (kernels/bf16_gemv.hpp)
// is a device kernel and a single-token GEMV; ggml-cpu's matmul is bound to ggml's tensor layout and to its own
// quantization rules, and EmbeddingGemma 2's weights are BF16 exactly as the checkpoint has them, its activations
// FP32, and the same call serves eight tokens and eight hundred.  The image engine's towers need the same product
// (src/embed/embeddinggemma2_vision.cpp), so it is written once here.  The free and contrib-llvm modes both take it:
// it is XeStrata's own code on every CPU, with no library behind it.
//
// THE LAYOUT.  A [n_out, n_in] BF16 matrix - the checkpoint's own shape, one output row contiguous after another -
// is repacked into PANELS of eight output rows, and a panel holds its values in k order: `panel[p][k][j]` is output
// `p*8 + j` of input index `k`.  Eight because that is one AVX2 float vector, so an accumulator is eight real
// outputs of one token and a lane never mixes two outputs.  A panel whose eight rows run past n_out is padded with
// zeros, which cost nothing to compute and are never stored.
//
// THE ARITHMETIC, and why it has no tolerance in its own test.  BF16 is the top half of an FP32, so a weight becomes
// an FP32 by a shift, exactly (`kernels/bf16_bits.hpp`); every weight and every activation is then FP32, the product
// rounds once into the accumulator and the accumulator walks k in ascending order - one chain per output, the same
// chain in the AVX2 kernel and in the scalar reference.  So:
//   * the AVX2 kernel and the scalar reference agree BIT FOR BIT, and a bug in one shows at once;
//   * an output does not depend on the token group it was computed in, on the panel band a thread took, or on the
//     thread count.  The embedding of a request is the same number on six cores and on sixteen, which is what lets
//     a student's stored features be trusted after a restart on another machine.
// The FP32 reference this is checked against (transformers, fp32, CPU) sums the same terms in another order, so THAT
// comparison is a cosine, not a bit pattern (see the record's `s2/`).
//
// THE PANEL ORDER.  Token block outer, then panels, then the k reduction: a panel is eight rows x n_in (8 KiB at
// n_in 512, 32 KiB at 2048) and stays in L1 while the block's rows go through it, so a matrix's bytes are streamed
// once per token block and not once per token.  The block is `bf16_gemm_token_block()`.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Output rows in one packed panel: the width of an AVX2 float vector, see the header note.
inline constexpr int kBf16Panel = 8;

/// The panels of a matrix with `n_out` outputs (the last may be part real, part zero padding).
constexpr int bf16_gemm_panels(int n_out) noexcept { return (n_out + kBf16Panel - 1) / kBf16Panel; }
/// The bytes of the packed form of a [n_out, n_in] BF16 matrix.
constexpr size_t bf16_gemm_packed_bytes(int n_out, int n_in) noexcept {
    return (size_t) bf16_gemm_panels(n_out) * (size_t) n_in * (size_t) kBf16Panel * sizeof(uint16_t);
}

/// The kernels of this product.  `scalar` is the reference and runs anywhere; `avx2` is the vector copy; `avx512`
/// takes two panels in one zmm where cpu_avx512_ok() (bf16_gemm_avx512.cpp; checked under Intel SDE, AGENTS.md).
/// All three give the same bits.
enum class Bf16GemmIsa : int { scalar = 0, avx2 = 1, avx512 = 2 };

/// Whether this CPU runs `isa` (the scalar kernel: always).
bool bf16_gemm_isa_ok(Bf16GemmIsa isa) noexcept;
/// The best ISA this CPU runs, capped by XESTRATA_BF16_GEMM_ISA=avx2|scalar (tests and A/B).
Bf16GemmIsa bf16_gemm_isa();
const char* bf16_gemm_isa_name(Bf16GemmIsa isa) noexcept;

/// The token rows one thread works on while it streams a matrix's panels: sized so that the block's activations
/// (rows x n_in floats) stay in L1/L2 for the whole panel loop, a multiple of eight, between 8 and 64 rows.
/// XESTRATA_BF16_GEMM_BLOCK overrides it (a tuning knob; the default measured on the i7-14700 and the Ryzen 7 5700X).
int bf16_gemm_token_block(int n_in);

/// Repack panels [p0, p1) of a BF16 matrix into the panel form.  `w` is the checkpoint's rows (n_out rows of `ld`
/// values, only the first n_in of each row read).  `panel` is always the WHOLE matrix's buffer of
/// `bf16_gemm_packed_bytes` bytes: a band writes at its own panel offsets, so callers may split [p0, p1) over
/// threads or load the tensors in any order.  A move of bytes, not arithmetic: every ISA packs the same.
void bf16_gemm_pack(const uint16_t* w, size_t ld, int n_in, int n_out, uint16_t* panel, int p0, int p1);

/// y[t*ldy + o] = (bias ? bias[o] : 0f) + sum_k x[t*ldx + k] * w[o][k] for every output o of panels [p0, p1),
/// over `n_tok` rows of activations (`x`: rows of ldx >= n_in floats) into rows of ldy >= n_out floats.
/// `panel` must be the packed form of the same n_in and n_out.  Call only where bf16_gemm_isa_ok(isa).
/// Bias is added after the reduction, never folded into the first product, so it is the FP32 addition the reference
/// does; the k chain of one output is the same whatever n_tok, p0 or p1 are.
void bf16_gemm_rows(Bf16GemmIsa isa, const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx,
                    float* y, size_t ldy, int n_tok, int p0, int p1, const float* bias);

/// The whole matrix, on the ISA this CPU runs (bf16_gemm_isa()).
void bf16_gemm_run(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y, size_t ldy,
                   int n_tok, const float* bias);

namespace bf16_gemm_detail {
/// The AVX2 kernel (bf16_gemm_avx2.cpp, compiled with -mavx2 -mfma): reached only through bf16_gemm_rows behind
/// bf16_gemm_isa_ok(), and public here so the test and the bench can call it directly.
void bf16_gemm_rows_avx2(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y, size_t ldy,
                         int n_tok, int p0, int p1, const float* bias);
/// The AVX-512 kernel (bf16_gemm_avx512.cpp, compiled with -mavx512f), the same way.
void bf16_gemm_rows_avx512(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y,
                           size_t ldy, int n_tok, int p0, int p1, const float* bias);
}  // namespace bf16_gemm_detail

}  // namespace strata::kernels::cpu
