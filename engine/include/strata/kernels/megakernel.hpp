// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// include/strata/kernels/megakernel.hpp - Per-layer persistent GDN dense chain for RDNA3/4 (STRATA_MEGAKERNEL=1):
//   q8_1 input quantization -> Q8_0 qkv + z projections + a/b gates -> conv1d + L2 norm -> GDN recurrence + output norm
//   + q8_1 image -> Q8_0 out projection, in one persistent kernel with CU-count-aware grid sizing and WDDM-safe
//   wall_clock64() spin guards (rdna3_wddm_guard.hpp).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Optional row / block-column alias map for tensor-parallel views (pitch == 0: contiguous row-major matrix).
struct MkMat {
    int pitch = 0;
    int nr = 0;
    int cw = 0;
    int cgap = 0;
    int cadd = 0;
    int rlo[8] = {};
    int radd[8] = {};
};

/// Mapped Q8_0 GEMV fallback outside the persistent kernel.
void q80_mmvq_mapped(const void* w, const MkMat& m, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);

struct MkGdnArgs {
    const float* x = nullptr;        ///< (T, n_embd)
    void* xq = nullptr;              ///< T columns of q8_1 blocks of n_embd
    const void* w_qkv = nullptr;     ///< Q8_0 (conv_channels, n_embd)
    const void* w_z = nullptr;       ///< Q8_0 (value_dim, n_embd)
    const void* w_out = nullptr;     ///< Q8_0 (n_embd, value_dim)
    float* qkv = nullptr;            ///< (T, conv_channels)
    float* z = nullptr;              ///< (T, value_dim)
    float* hist = nullptr;           ///< (conv_channels, d_conv - 1), read-only during verify window
    const float* w_conv = nullptr;   ///< (conv_channels, d_conv)
    float* h = nullptr;              ///< (T, conv_channels)
    const uint16_t* w_alpha = nullptr;  ///< bf16 (h_v, n_embd)
    const uint16_t* w_beta = nullptr;   ///< bf16 (h_v, n_embd)
    const float* dt = nullptr;       ///< (h_v)
    const float* ssm_a = nullptr;    ///< (h_v)
    float* gate = nullptr;           ///< (T, h_v)
    float* beta = nullptr;           ///< (T, h_v)
    float* state = nullptr;          ///< (S, h_v, S), read-only during verify window
    const float* gamma = nullptr;    ///< (S)
    float* y = nullptr;              ///< (T, value_dim)
    void* yq = nullptr;              ///< T columns of q8_1 blocks of value_dim
    float* bo = nullptr;             ///< (T, n_embd)
    int n_embd = 0;
    int conv_channels = 0;
    int value_dim = 0;
    int h_k = 0;
    int h_v = 0;
    MkMat m_qkv;
    MkMat m_z;
    MkMat m_out;
    float eps = 1e-6f;
    unsigned* counters = nullptr;    ///< device scratch (>= 16 uints), zeroed once at allocation and reset on kernel exit
    unsigned* stall = nullptr;       ///< optional mapped coherent host word: non-zero (phase + 1) on spin timeout
    int phases = 5;                  ///< run the first `phases` phases (default 5 = full chain)
    int chunk = 0;                   ///< dynamic scheduler chunk override (0 = static co-resident striding)
    int grid = 0;                    ///< diagnostic grid override (0 = CU-safe occupancy count)
    int diag = 0;                    ///< bit 5 (32): skip phase 0 (STRATA_QFUSE); bit 8 (256): dynamic scheduler
};

/// Allocate and zero the 16-uint device counter buffer for mk_gdn_dense.
unsigned* mk_counters_alloc();
/// Safe persistent grid size for `n_tok` on the current device (0 if unsupported).
int mk_gdn_grid(int n_tok);
/// Launch the fused 5-phase GDN dense chain on `stream` for `n_tok` in 1..8.
void mk_gdn_dense(const MkGdnArgs& a, int n_tok, void* stream);

}  // namespace strata::kernels
