// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/i8_gemm.hpp - the CPU's int8 x int8 matrix product, the one the CPU forwards share: the
// EmbeddingGemma 2 forward (beside kernels/cpu/bf16_gemm.hpp's exact one) and the plain dense model on the CPU (a GGUF
// file's quantized weights).  It runs on AVX-512 VNNI (zmm vpdpbusd), on AVX-VNNI-INT8 (ymm vpdpbssd), on AVX-VNNI (ymm
// vpdpbusd), on AVX2 alone (vpmaddubsw + vpmaddwd), or as the scalar reference anywhere.
//
// WHY A SECOND PRODUCT AT ALL.  bf16_gemm keeps every weight of the checkpoint exactly, and it is the path the
// parity with the fp32 reference is measured on.  Its cost is two bytes per weight and one FP32 FMA per output per
// k.  A chat's student pays for a forward between turns, and that forward streams the whole weight set once per
// token block: int8 halves the bytes and `vpdpbusd` puts four products in every lane where an FMA puts one.  It is
// an APPROXIMATION - the weights land on 127 levels per scale block, the activations likewise - so the answer moves,
// and which CPUs take it by default is each forward's measured decision, not this header's.
//
// THE SCALE BLOCKS.  A scale covers `block` consecutive k values of one row (a multiple of four; 0 is the whole row,
// EmbeddingGemma 2's choice).  A GGUF model's weights carry a scale per 16 or 32 values, and a scale per row would
// lose the small sub-blocks of a K-quant next to the large ones: the plain forward packs with a block of 32, where an
// int8 level is finer than the 5 or 6 bits of the file.  `nb = i8_gemm_blocks(n_in, block)` blocks a row, the last
// one shorter when `block` does not divide n_in.
//
// THE ARITHMETIC, exactly.  One scale per block on each side - the weight's made when it is packed, the
// activation's made per token row and block as it arrives:
//
//     sw[o][b]    = max_{k in b} |w[o][k]| / L              (FP32)     winv = L / max_{k in b} |w[o][k]|
//     qw[o][k]    = clip(nearbyint(w[o][k] * winv), -L, L)              int8, the signed operand
//     sa[t][b]    = max_{k in b} |x[t][k]| / 127            (FP32)     xinv = 127 / max_{k in b} |x[t][k]|
//     qu[t][k]    = clip(nearbyint(x[t][k] * xinv), -127, 127) + 128    u8, the unsigned operand
//     corr[o][b]  = 128 * sum_{k in b} qw[o][k]             (int32, made at pack time)
//     d[t][o][b]  = sum_{k in b} qu[t][k] * qw[o][k] - corr[o][b]       (int32: sum qa * qw, exactly)
//     y[t][o]     = the blocks in order: f = fma(sa[t][b] * sw[o][b], (float) d, f), starting from the bias, or
//                   from the first block's product sa * sw * d alone when there is no bias
//
// `vpdpbusd` multiplies an unsigned lane by a signed one, so the activation carries the +128 and `corr` takes it
// back out.  The AVX2 kernel multiplies the signed levels directly (vpmaddubsw on |qa| and qw with qa's sign) and
// needs no correction: its d is the same integer.  The scale is held as the block's maximum over 127 and quantized
// with a multiply by its reciprocal - the same two float operations in the scalar reference and in the vector lanes,
// which is what lets them agree bit for bit.  A block of all zeros keeps scale 0 and reciprocal 0: every byte lands
// on 128, d on 0, and the block adds nothing.  With one block (block 0) the float part is the single multiply, or
// the single fused multiply-add with the bias, that EmbeddingGemma 2's forward was measured with.
//
// WHY THE ANSWER IS DETERMINISTIC.  Integer addition is associative and exact, so d does not depend on the order of
// k, on the token group, on the panel band a thread took, on the thread count or on the ISA; the float part is the
// same chain of multiplies and fused multiply-adds per output on every path.  So one request gives the same bits on
// six cores and on sixteen, and on every CPU, AVX-VNNI or AVX2 or neither.
//
// THE WEIGHT LEVELS L.  127 (kI8Levels), or 63 (kI8Levels7) for the AVX2 kernel: a weight of seven bits lets
// `vpmaddubsw` take the unsigned activation as it is - its pairs hold 2 * 255 * 63 = 32130 at most, inside int16,
// where 2 * 255 * 127 would saturate - so a quad costs vpmaddubsw, vpmaddwd and an add, and `corr` takes the +128
// out as on VNNI, where the eight-bit form moves the sign onto the weight first (six operations).  The
// activation keeps 127 levels either way.  A matrix is packed for the kernel that will run it
// (i8_gemm_weight_levels), so one process never mixes the forms; every kernel computes either form exactly, the
// scalar reference included, and `levels` only tells a kernel which loop it may take.
//
// THE BOUNDS.  |qu| <= 255 and |qw| <= 127, so one accumulator lane holds at most 32385 * n_in and the offset
// term at most 16256 * n_in.  `kI8MaxIn` is the widest reduction this file accepts: below it the int32 lanes
// cannot wrap.  The AVX2 kernel's 16-bit pairs hold 2 * 127 * 127 = 32258 at most in the eight-bit form, 32130 in
// the seven-bit one, inside int16.
//
// THE LAYOUT.  A quad of four k values of one panel is 32 bytes, because a `vpdpbusd` operand is 32 bytes and its
// eight int32 lanes are eight outputs: `panel[p][kq][j][qi]` holds the weight of output `p*8 + j` at input index
// `kq*4 + qi`, as int8 bits (`j` is the lane, so the four bytes of one lane are four consecutive k).  The
// activation of one token at one quad is the same four bytes in every lane, so it costs one dword broadcast.  A
// matrix whose n_in is not a multiple of four pads its last quad with zero weights and zero activation bytes -
// either one makes the product zero.  A partial last panel pads its lanes with scale 0 and correction 0: they are
// computed and never stored.  The weight scales and corrections are `[panel][block][lane]`, eight lanes together
// (one vector load); the activation scales are `[token][block]`.  With one block both are the old one-per-row arrays.
//
// THE PANEL ORDER.  As in bf16_gemm: token block outer, then panels, then the k reduction, so a matrix's bytes
// stream once per token block.  The block is `bf16_gemm_token_block()` - the same rule, measured on the i7-14700
// and the Ryzen 7 5700X for the FP32 path and half its bytes here.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Output rows in one packed panel: the lane count of a `vpdpbusd` operand.
inline constexpr int kI8Panel = 8;
/// k values one lane holds: the byte count of one int32 lane.
inline constexpr int kI8Quad = 4;
/// The levels on one side of a row.  127, so the u8 side of the product stays inside 1..255.
inline constexpr int kI8Levels = 127;
/// The weight levels of the seven-bit form (THE WEIGHT LEVELS above): 255 * 63 * 2 fits `vpmaddubsw`'s int16 pairs.
inline constexpr int kI8Levels7 = 63;
/// The widest reduction whose int32 accumulator cannot wrap: see THE BOUNDS above.
constexpr int kI8MaxIn = 32768;

/// Panels in a quantized matrix with `n_out` outputs.
constexpr int i8_gemm_panels(int n_out) noexcept { return (n_out + kI8Panel - 1) / kI8Panel; }
/// Quads in a quantized matrix with `n_in` inputs (the last may be part padding).
constexpr int i8_gemm_quads(int n_in) noexcept { return (n_in + kI8Quad - 1) / kI8Quad; }
/// The bytes of one panel-quad: eight lanes of four bytes.
inline constexpr int kI8QuadBytes = kI8Panel * kI8Quad;
/// The bytes of the packed weights of a [n_out, n_in] matrix.
constexpr size_t i8_gemm_packed_bytes(int n_out, int n_in) noexcept {
    return (size_t) i8_gemm_panels(n_out) * (size_t) i8_gemm_quads(n_in) * (size_t) kI8QuadBytes;
}
/// The bytes one token row of the quantized activations takes: the quads, four bytes each.
constexpr int i8_gemm_act_row_bytes(int n_in) noexcept { return i8_gemm_quads(n_in) * kI8Quad; }
/// The scale blocks of a row: 1 for `block` 0 (the whole row), else the row in blocks of `block` k values.
constexpr int i8_gemm_blocks(int n_in, int block) noexcept { return block <= 0 ? 1 : (n_in + block - 1) / block; }
/// The entries of the weight scale and correction arrays: one per panel lane and block.
constexpr size_t i8_gemm_scale_count(int n_out, int n_in, int block) noexcept {
    return (size_t) i8_gemm_panels(n_out) * (size_t) kI8Panel * (size_t) i8_gemm_blocks(n_in, block);
}
/// Whether `block` is one the product takes for a row of `n_in`: 0, or a multiple of four.
constexpr bool i8_gemm_block_ok(int block) noexcept { return block == 0 || (block > 0 && block % kI8Quad == 0); }
/// Whether `levels` is a weight form the product takes: kI8Levels or kI8Levels7.
constexpr bool i8_gemm_levels_ok(int levels) noexcept { return levels == kI8Levels || levels == kI8Levels7; }

/// The kernels of this product.  `scalar` is the reference and runs anywhere; `avx2` needs AVX2 alone; `vnni` needs
/// AVX2 and AVX-VNNI; `avx512` needs AVX512F and AVX512_VNNI (cpu_avx512vnni_ok); `vnni_int8` needs AVX2 and
/// AVX-VNNI-INT8 (cpu_avxvnniint8_ok) and reads the activations quantized signed, without the +128 (i8_gemm_quantize
/// writes them so for it), so its sums need no correction.  All five give the same bits.
enum class I8GemmIsa : int { scalar = 0, vnni = 1, avx2 = 2, avx512 = 3, vnni_int8 = 4 };

/// Whether this CPU runs `isa` (the scalar reference: always).
bool i8_gemm_isa_ok(I8GemmIsa isa) noexcept;
/// The best ISA this CPU runs, capped by XESTRATA_I8_GEMM_ISA=vnni_int8|vnni|avx2|scalar (tests and A/B).  The order:
/// AVX-512 VNNI (a zmm vpdpbusd is 64 products where a ymm vpdpbssd is 32, and the +128's correction it pays is one
/// subtract a block of 32 values), AVX-VNNI-INT8 (no correction, no other cost against AVX-VNNI), AVX-VNNI, AVX2.
I8GemmIsa i8_gemm_isa();
const char* i8_gemm_isa_name(I8GemmIsa isa) noexcept;
/// The weight levels a matrix that `isa` will run is packed with: kI8Levels7 for the AVX2 kernel, kI8Levels
/// otherwise.  XESTRATA_I8_GEMM_WEIGHT_BITS=8 keeps the eight-bit form on AVX2 as well (A/B).
int i8_gemm_weight_levels(I8GemmIsa isa);

/// Quantize and pack the panels [p0, p1) of a BF16 matrix: `panel` the int8 weights in the quad order above,
/// `wscale` the FP32 scales and `wcorr` the int32 offset corrections, `[panel][block][lane]`
/// (`i8_gemm_scale_count` entries; a partial last panel's lanes get scale 0 and correction 0).  `w` is `n_out` rows
/// of `ld` >= n_in BF16 values; `panel` is always the WHOLE matrix's buffer of `i8_gemm_packed_bytes` bytes (a band
/// writes at its own panel offsets, as bf16_gemm_pack).  `levels` is the weight form (THE WEIGHT LEVELS).  Returns
/// false - writing nothing - when n_in is wider than kI8MaxIn, or `block` or `levels` is not one the product takes.
bool i8_gemm_pack(const uint16_t* w, size_t ld, int n_in, int n_out, uint8_t* panel, float* wscale, int32_t* wcorr,
                  int p0, int p1, int block = 0, int levels = kI8Levels);
/// The same from FP32 rows (a GGUF tensor's rows, dequantized by the caller).
bool i8_gemm_pack_f32(const float* w, size_t ld, int n_in, int n_out, uint8_t* panel, float* wscale, int32_t* wcorr,
                      int p0, int p1, int block, int levels = kI8Levels);

/// Quantize `n` activation rows of `x` (rows of `ldx` >= n_in floats) into `xq` (rows of `ldq` >=
/// `i8_gemm_act_row_bytes(n_in)` bytes, the pad bytes zero) with one FP32 scale per row and block in `xs`
/// (`[t][b]`, `i8_gemm_blocks(n_in, block)` a row).  The levels are the same on every ISA: the scale is a max over the
/// block, and the level is one multiply and one round-to-nearest-even; the byte is the level + 128, or the level itself
/// for I8GemmIsa::vnni_int8.
void i8_gemm_quantize(I8GemmIsa isa, const float* x, size_t ldx, int n, int n_in, uint8_t* xq, size_t ldq,
                      float* xs, int block = 0);

/// y[t*ldy + o] = the blocks' chain above, for every output of panels [p0, p1), over `n_tok` quantized activation
/// rows.  `block` and `levels` are the ones the weights were packed with (`block` the activations' as well).  Call
/// only where i8_gemm_isa_ok(isa), and only on a matrix `i8_gemm_pack` accepted.
void i8_gemm_rows(I8GemmIsa isa, const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in,
                  int n_out, const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok,
                  int p0, int p1, const float* bias, int block = 0, int levels = kI8Levels);

namespace i8_gemm_detail {
/// The vector kernels (i8_gemm_vnni.cpp): reached through i8_gemm_rows behind i8_gemm_isa_ok(), and public here so
/// the test and the bench can call them directly.  The VNNI one is defined only where the build has the AVX-VNNI
/// intrinsics, the VNNI-INT8 one where it has those; `copy` 1 is AVX-VNNI, 2 AVX-VNNI-INT8 (signed activations), 0 AVX2,
/// in its seven-bit form when `levels` is kI8Levels7.
void i8_gemm_rows_x86(int copy, const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in,
                      int n_out, const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok,
                      int p0, int p1, const float* bias, int block, int levels);
/// The AVX-512 VNNI kernel (i8_gemm_avx512.cpp), the panels a group of four does not fill on the 256-bit copies.
void i8_gemm_rows_avx512(const uint8_t* panel, const float* wscale, const int32_t* wcorr, int n_in, int n_out,
                         const uint8_t* xq, size_t ldq, const float* xs, float* y, size_t ldy, int n_tok, int p0, int p1,
                         const float* bias, int block);
/// The same quantization pass on AVX2 lanes; its levels and scales are identical to the scalar loop's (`sign`: the
/// levels as they are, for the VNNI-INT8 copy, instead of + 128).
void i8_gemm_quantize_avx2(const float* x, size_t ldx, int n, int n_in, uint8_t* xq, size_t ldq, float* xs,
                           int block, bool sign = false);
}  // namespace i8_gemm_detail

}  // namespace strata::kernels::cpu
