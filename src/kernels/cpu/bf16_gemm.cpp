// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/bf16_gemm.cpp - see include/strata/kernels/cpu/bf16_gemm.hpp: the packing, the scalar reference
// kernel, the run-time ISA choice and the dispatch.  Compiled for the baseline x86-64, so everything here runs on
// any CPU; the AVX2 and AVX-512 kernels are in bf16_gemm_avx2.cpp and bf16_gemm_avx512.cpp, translation units of
// their own.
#include "strata/kernels/cpu/bf16_gemm.hpp"

#include "strata/kernels/bf16_bits.hpp"           // f32_from_bf16
#include "strata/kernels/cpu/expert_layout.hpp"   // cpu_avx2_ok, cpu_avx512_ok
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

// The AVX2 unit (bf16_gemm_avx2.cpp) exists on x86-64 builds only; anything else runs the scalar kernel.
#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_BF16_GEMM_X86 1
#endif

namespace strata::kernels::cpu {
namespace {

/// y[t*ldy + o] = (bias ? bias[o] : 0f) + sum_k x[t*ldx + k] * w[o][k], panels [p0, p1).
/// The order of the additions is the definition of the product for this header: `std::fmaf` so that a build without
/// FMA contraction still rounds the multiply-add once, as the AVX2 kernel's vfmadd does.
void bf16_gemm_rows_scalar(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y, size_t ldy,
                           int n_tok, int p0, int p1, const float* bias) {
    for (int p = p0; p < p1; ++p) {
        const int base = p * kBf16Panel;
        const int cols = std::min(kBf16Panel, n_out - base);
        const uint16_t* wp = panel + (size_t) p * (size_t) n_in * (size_t) kBf16Panel;
        for (int t = 0; t < n_tok; ++t) {
            const float* xr = x + (size_t) t * ldx;
            float* yr = y + (size_t) t * ldy + base;
            for (int j = 0; j < cols; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < n_in; ++k)
                    acc = std::fmaf(xr[k], strata::kernels::f32_from_bf16(wp[(size_t) k * kBf16Panel + j]), acc);
                yr[j] = bias ? acc + bias[base + j] : acc;
            }
        }
    }
}

}  // namespace

bool bf16_gemm_isa_ok(Bf16GemmIsa isa) noexcept {
#if defined(STRATA_BF16_GEMM_X86)
    return isa == Bf16GemmIsa::avx2 ? cpu_avx2_ok() : isa == Bf16GemmIsa::avx512 ? cpu_avx512_ok() : true;
#else
    return isa == Bf16GemmIsa::scalar;
#endif
}

Bf16GemmIsa bf16_gemm_isa() {
    static const Bf16GemmIsa isa = [] {
        const char* e = strata::env_get("BF16_GEMM_ISA");
        if (e != nullptr && std::strcmp(e, "scalar") == 0) return Bf16GemmIsa::scalar;
        const bool cap_avx2 = e != nullptr && std::strcmp(e, "avx2") == 0;
        if (!cap_avx2 && bf16_gemm_isa_ok(Bf16GemmIsa::avx512)) return Bf16GemmIsa::avx512;
        return bf16_gemm_isa_ok(Bf16GemmIsa::avx2) ? Bf16GemmIsa::avx2 : Bf16GemmIsa::scalar;
    }();
    return isa;
}

const char* bf16_gemm_isa_name(Bf16GemmIsa isa) noexcept {
    return isa == Bf16GemmIsa::avx512 ? "avx512" : isa == Bf16GemmIsa::avx2 ? "avx2" : "scalar";
}

int bf16_gemm_token_block(int n_in) {
    static const int cap = [] {
        const char* e = strata::env_get("BF16_GEMM_BLOCK");
        return e != nullptr ? (int) std::strtol(e, nullptr, 10) : 0;
    }();
    if (cap > 0) return cap;
    // The block's activations stay hot for the whole panel loop: 64 KiB of them, a quarter of the smallest L2 on an
    // AVX2 CPU (256 KiB on Haswell and Zen 1) and well inside any L3, rounded down to whole groups of eight rows.
    // Measured on the i7-14700 (1.25 MiB L2 per P-core) and the Ryzen 7 5700X (512 KiB): the width of the plateau,
    // not a cliff - 16 rows and 32 are within a few percent of each other, and 8 costs the AVX2 kernel its latency
    // hiding.  The panel, not the block, is what the weights live in (8 KiB at n_in 512), so this only trades the
    // activations' residency against how often the weights are re-read.
    const int by_bytes = 64 * 1024 / (n_in * (int) sizeof(float));
    return std::clamp(by_bytes - (by_bytes % kBf16Panel), kBf16Panel, 8 * kBf16Panel);
}

void bf16_gemm_pack(const uint16_t* w, size_t ld, int n_in, int n_out, uint16_t* panel, int p0, int p1) {
    for (int p = p0; p < p1; ++p) {
        uint16_t* dst = panel + (size_t) p * (size_t) n_in * (size_t) kBf16Panel;
        const int base = p * kBf16Panel;
        const int rows = std::min(kBf16Panel, n_out - base);
        for (int k = 0; k < n_in; ++k) {
            uint16_t v[kBf16Panel];
            for (int j = 0; j < rows; ++j) v[j] = w[(size_t) (base + j) * ld + k];
            for (int j = rows; j < kBf16Panel; ++j) v[j] = 0;    // the last panel's padding: never stored, never used
            std::memcpy(dst + (size_t) k * kBf16Panel, v, sizeof v);
        }
    }
}

void bf16_gemm_rows(Bf16GemmIsa isa, const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y,
                    size_t ldy, int n_tok, int p0, int p1, const float* bias) {
    if (p0 >= p1 || n_tok <= 0) return;
#if defined(STRATA_BF16_GEMM_X86)
    if (isa == Bf16GemmIsa::avx2) {
        bf16_gemm_detail::bf16_gemm_rows_avx2(panel, n_in, n_out, x, ldx, y, ldy, n_tok, p0, p1, bias);
        return;
    }
    if (isa == Bf16GemmIsa::avx512) {
        bf16_gemm_detail::bf16_gemm_rows_avx512(panel, n_in, n_out, x, ldx, y, ldy, n_tok, p0, p1, bias);
        return;
    }
#endif
    bf16_gemm_rows_scalar(panel, n_in, n_out, x, ldx, y, ldy, n_tok, p0, p1, bias);
}

void bf16_gemm_run(const uint16_t* panel, int n_in, int n_out, const float* x, size_t ldx, float* y, size_t ldy,
                   int n_tok, const float* bias) {
    bf16_gemm_rows(bf16_gemm_isa(), panel, n_in, n_out, x, ldx, y, ldy, n_tok, 0, bf16_gemm_panels(n_out), bias);
}

}  // namespace strata::kernels::cpu
