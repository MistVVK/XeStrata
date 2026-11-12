// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/cpu_ops.cpp - see include/strata/kernels/cpu/cpu_ops.hpp.  Compiled for the baseline x86-64: the
// vector work is in the i8_gemm kernels it calls, which choose their copy at run time.
#include "strata/kernels/cpu/cpu_ops.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <cmath>

namespace strata::kernels::cpu {

void rmsnorm_rows(const float* x, size_t ldx, float* y, size_t ldy, int n, int dim, const float* w, float eps) {
    for (int t = 0; t < n; ++t) {
        const float* r = x + (size_t) t * ldx;
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) ss = std::fmaf(r[i], r[i], ss);
        const float inv = (float) std::pow((double) (ss / (float) dim + eps), -0.5);
        float* o = y + (size_t) t * ldy;
        if (w != nullptr) {
            for (int i = 0; i < dim; ++i) o[i] = r[i] * inv * w[i];
        } else {
            for (int i = 0; i < dim; ++i) o[i] = r[i] * inv;
        }
    }
}

bool i8_pack_rows(ExpertPool& pool, int n_in, int n_out, int block, int levels,
                  const std::function<bool(int o, float* dst)>& row, I8Matrix& m) {
    if (n_in > kI8MaxIn || !i8_gemm_block_ok(block) || !i8_gemm_levels_ok(levels) || n_in <= 0 || n_out <= 0)
        return false;
    m.n_in = n_in;
    m.n_out = n_out;
    m.block = block;
    m.levels = levels;
    m.panel.assign(i8_gemm_packed_bytes(n_out, n_in), 0);
    m.scale.assign(i8_gemm_scale_count(n_out, n_in, block), 0.0f);
    m.corr.assign(m.scale.size(), 0);
    const int panels = m.panels();
    constexpr int kBand = 16;   // panels a piece: 128 rows dequantized into a buffer of the piece's own
    const int pieces = (panels + kBand - 1) / kBand;
    std::atomic<bool> bad{false};
    pool.run(pieces, [&](int i) {
        const int p0 = i * kBand, p1 = std::min(panels, p0 + kBand);
        const int o0 = p0 * kI8Panel, o1 = std::min(n_out, p1 * kI8Panel);
        std::vector<float> rows((size_t) (o1 - o0) * (size_t) n_in);
        for (int o = o0; o < o1 && !bad; ++o)
            if (!row(o, rows.data() + (size_t) (o - o0) * n_in)) bad = true;
        if (bad) return;
        // the band's rows start at output o0: a band of a packed matrix is itself a packed matrix, so the band is
        // packed as one of `o1 - o0` outputs into its own place in the whole
        const size_t quads = (size_t) i8_gemm_quads(n_in);
        const size_t lanes = (size_t) i8_gemm_blocks(n_in, block) * kI8Panel;
        if (!i8_gemm_pack_f32(rows.data(), (size_t) n_in, n_in, o1 - o0, m.panel.data() + (size_t) p0 * quads *
                              kI8QuadBytes, m.scale.data() + (size_t) p0 * lanes, m.corr.data() + (size_t) p0 * lanes,
                              0, p1 - p0, block, levels))
            bad = true;
    });
    return !bad;
}

void i8_quantize_rows(ExpertPool& pool, I8GemmIsa isa, const float* x, size_t ldx, int n, int n_in, int block,
                      I8Rows& r) {
    r.n = n;
    r.n_in = n_in;
    r.block = block;
    r.ldq = (size_t) i8_gemm_act_row_bytes(n_in);
    const int nb = i8_gemm_blocks(n_in, block);
    if (r.q.size() < (size_t) n * r.ldq) r.q.resize((size_t) n * r.ldq);
    if (r.s.size() < (size_t) n * nb) r.s.resize((size_t) n * nb);
    const int per = std::max(1, (n + pool.size() * 2 - 1) / (pool.size() * 2));
    pool.run((n + per - 1) / per, [&](int i) {
        const int t0 = i * per, t1 = std::min(n, t0 + per);
        i8_gemm_quantize(isa, x + (size_t) t0 * ldx, ldx, t1 - t0, n_in, r.q.data() + (size_t) t0 * r.ldq, r.ldq,
                         r.s.data() + (size_t) t0 * nb, block);
    });
}

void i8_linear(ExpertPool& pool, I8GemmIsa isa, const I8Matrix& m, const I8Rows& r, float* y, size_t ldy,
               const float* bias) {
    const int panels = m.panels();
    if (panels == 0 || r.n <= 0) return;
    // pieces: whole panels, four for each taker; the token rows split as well where the panels are fewer (a narrow
    // matrix over a long prompt), in chunks of 32 rows at least so a piece still streams its panels' bytes once for
    // many rows
    const int want = pool.size() * 4;
    const int bands = std::min(panels, want);
    const int chunks = std::max(1, std::min((want + bands - 1) / bands, (r.n + 31) / 32));
    const int nb = i8_gemm_blocks(m.n_in, m.block);
    pool.run(bands * chunks, [&](int i) {
        const int b = i / chunks, c = i % chunks;
        const int p0 = (int) ((long long) panels * b / bands), p1 = (int) ((long long) panels * (b + 1) / bands);
        const int t0 = (int) ((long long) r.n * c / chunks), t1 = (int) ((long long) r.n * (c + 1) / chunks);
        if (p0 >= p1 || t0 >= t1) return;
        i8_gemm_rows(isa, m.panel.data(), m.scale.data(), m.corr.data(), m.n_in, m.n_out, r.q.data() + (size_t) t0 * r.ldq,
                     r.ldq, r.s.data() + (size_t) t0 * nb, y + (size_t) t0 * ldy, ldy, t1 - t0, p0, p1, bias, m.block,
                     m.levels);
    });
}

bool bf16_pack_rows(ExpertPool& pool, int n_in, int n_out, const std::function<bool(int o, float* dst)>& row,
                    Bf16Matrix& m) {
    if (n_in <= 0 || n_out <= 0) return false;
    m.n_in = n_in;
    m.n_out = n_out;
    m.panel.assign(bf16_gemm_packed_bytes(n_out, n_in) / 2, 0);
    const int panels = m.panels();
    constexpr int kBand = 16;
    std::atomic<bool> bad{false};
    pool.run((panels + kBand - 1) / kBand, [&](int i) {
        const int p0 = i * kBand, p1 = std::min(panels, p0 + kBand);
        const int o0 = p0 * kBf16Panel, o1 = std::min(n_out, p1 * kBf16Panel);
        std::vector<float> f((size_t) n_in);
        std::vector<uint16_t> rows((size_t) (o1 - o0) * (size_t) n_in);
        for (int o = o0; o < o1 && !bad; ++o) {
            if (!row(o, f.data())) bad = true;
            for (int k = 0; k < n_in; ++k) rows[(size_t) (o - o0) * n_in + k] = bf16_from_f32(f[(size_t) k]);
        }
        if (bad) return;
        // the band is itself a packed matrix of its rows, at its panels' place in the whole
        bf16_gemm_pack(rows.data(), (size_t) n_in, n_in, o1 - o0,
                       m.panel.data() + (size_t) p0 * (size_t) n_in * kBf16Panel, 0, p1 - p0);
    });
    return !bad;
}

void bf16_linear(ExpertPool& pool, const Bf16Matrix& m, const float* x, size_t ldx, int n, float* y, size_t ldy,
                 const float* bias) {
    const int panels = m.panels();
    if (panels == 0 || n <= 0) return;
    const Bf16GemmIsa isa = bf16_gemm_isa();
    const int want = pool.size() * 4;
    const int bands = std::min(panels, want);
    const int chunks = std::max(1, std::min((want + bands - 1) / bands, (n + 31) / 32));
    pool.run(bands * chunks, [&](int i) {
        const int b = i / chunks, c = i % chunks;
        const int p0 = (int) ((long long) panels * b / bands), p1 = (int) ((long long) panels * (b + 1) / bands);
        const int t0 = (int) ((long long) n * c / chunks), t1 = (int) ((long long) n * (c + 1) / chunks);
        if (p0 >= p1 || t0 >= t1) return;
        bf16_gemm_rows(isa, m.panel.data(), m.n_in, m.n_out, x + (size_t) t0 * ldx, ldx, y + (size_t) t0 * ldy, ldy,
                       t1 - t0, p0, p1, bias);
    });
}

}  // namespace strata::kernels::cpu
