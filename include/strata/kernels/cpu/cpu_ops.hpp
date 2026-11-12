// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/cpu_ops.hpp - what the CPU forwards share besides their matrix products (EmbeddingGemma
// 2's text tower, the plain dense model on the CPU): the pool their work goes out on, the RMSNorm, and the int8
// linear layer over kernels/cpu/i8_gemm.hpp split across the pool.
//
// THE POOL.  Its workers live as long as the forward: starting threads per product costs more than a one-token
// product.  `run(n, fn)` hands out fn(0)..fn(n-1) one at a time from a shared counter, the calling thread among the
// takers, so a slower core (an E-core beside the P-cores) takes fewer pieces rather than holding the others back.
//
// THE LINEAR LAYER.  A matrix packed once at load (i8_gemm_pack / i8_gemm_pack_f32, with the forward's scale block),
// and a product split into pieces of whole panels, and of token rows where the panels alone would leave the pool
// idle: each piece is one i8_gemm_rows call, so the answer does not depend on the split (i8_gemm.hpp).  The
// activation rows are quantized once per product, by the pool, before the pieces go out.
#pragma once

#include "strata/kernels/cpu/bf16_gemm.hpp"
#include "strata/kernels/cpu/i8_gemm.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

/// `y[t] = x[t] * pow(mean(x[t]^2) + eps, -0.5)`, times `w` when it has one (the scale-free norms pass null).  The
/// squares walk the row in index order into one accumulator, as every other sum here does.
void rmsnorm_rows(const float* x, size_t ldx, float* y, size_t ldy, int n, int dim, const float* w, float eps);

/// A matrix in the int8 form of i8_gemm.hpp: [n_out, n_in], scale block `block`, weight levels `levels`.
struct I8Matrix {
    std::vector<uint8_t> panel;
    std::vector<float> scale;
    std::vector<int32_t> corr;
    int n_in = 0, n_out = 0, block = 0, levels = kI8Levels;
    int panels() const { return i8_gemm_panels(n_out); }
    size_t bytes() const { return panel.size() + scale.size() * 4 + corr.size() * 4; }
};

/// Pack the rows `row(o, dst)` writes (n_in floats each) into `m` with `levels` weight levels
/// (i8_gemm_weight_levels of the ISA that will run it), panel bands on the pool.  False when the matrix is one
/// i8_gemm_pack refuses (wider than kI8MaxIn) or a row could not be made (`row` returned false).
bool i8_pack_rows(ExpertPool& pool, int n_in, int n_out, int block, int levels,
                  const std::function<bool(int o, float* dst)>& row, I8Matrix& m);

/// Activation rows quantized for one or more products that read them: `n` rows of `n_in`, scale block `block`.
struct I8Rows {
    std::vector<uint8_t> q;
    std::vector<float> s;
    size_t ldq = 0;
    int n = 0, n_in = 0, block = 0;
};
/// Quantize `n` rows of `x` (stride `ldx`) into `r`, row ranges on the pool.
void i8_quantize_rows(ExpertPool& pool, I8GemmIsa isa, const float* x, size_t ldx, int n, int n_in, int block,
                      I8Rows& r);
/// y[t * ldy + o] for the r.n rows through `m` (its scale block r's), pieces of panels and token rows on the pool.
void i8_linear(ExpertPool& pool, I8GemmIsa isa, const I8Matrix& m, const I8Rows& r, float* y, size_t ldy,
               const float* bias = nullptr);

/// A matrix in bf16_gemm.hpp's panel form: [n_out, n_in] BF16, the FP32 activations' product.
struct Bf16Matrix {
    std::vector<uint16_t> panel;
    int n_in = 0, n_out = 0;
    int panels() const { return bf16_gemm_panels(n_out); }
    size_t bytes() const { return panel.size() * 2; }
};
/// Pack the rows `row(o, dst)` writes (n_in floats each, rounded to BF16) into `m`, panel bands on the pool.
bool bf16_pack_rows(ExpertPool& pool, int n_in, int n_out, const std::function<bool(int o, float* dst)>& row,
                    Bf16Matrix& m);
/// y[t * ldy + o] = x[t] . m[o] (+ bias[o]) for `n` rows of x (stride ldx), pieces of panels and token rows on the pool.
void bf16_linear(ExpertPool& pool, const Bf16Matrix& m, const float* x, size_t ldx, int n, float* y, size_t ldy,
                 const float* bias = nullptr);

}  // namespace strata::kernels::cpu
