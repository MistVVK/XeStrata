// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/flash_attn.hpp - the attention on the GPU's FP16 matrix engines that the image engine's DiT
// (qi_attention) and the plain models' prompt (prompt_attn) share: one kernel, the keys and values read where they
// are (a dense [key][row] layout, or the K/V cache's pages), the output in FP16 or FP32.
//
// A sub-group takes TM query rows of one head and walks that head's keys 64 at a time on its own: S = Q . K^T with
// Q's tiles in registers and K's read from global memory (the caches share them between the sub-groups of a
// work-group, which take neighbouring rows of the same head), the online softmax in local memory with a lazily moved
// reference (FlashAttention-3's), and O += P . V in the matrix engines' accumulators.  Intel's XMX (8 x 16 x 16 on 16
// lanes, 8 x 8 x 16 on 8), NVIDIA's tensor cores from sm_70 on (16 x 16 x 16 on 32) through joint_matrix, RDNA3's and
// RDNA4's WMMA instructions (amd_wmma.hpp); a GPU without them, or STRATA_NO_XMX=1, has none (flash_attn_path -1).
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

struct FlashAttnArgs {
    const uint16_t* q = nullptr;     ///< FP16 queries: row t, head h at q + t * q_ld + h * head_dim (rows past `rows`
    const float* q32 = nullptr;      ///< up to the sub-group's tile there, zero), or FP32 ones (q32, none read past)
    int64_t q_ld = 0;
    const uint16_t* k = nullptr;     ///< FP16 keys and values, dense (page_table null): key j of KV head g at
    const uint16_t* v = nullptr;     ///< k + j * kv_ld + g * head_dim; or paged: [page][n_kv][page_size][head_dim]
    int64_t kv_ld = 0;
    const int32_t* page_table = nullptr;   ///< a page's slot (page_size a multiple of 64 for k and v read in place)
    int64_t page_size = 0;
    /// The K/V cache's pools in another format (INT8 with an FP16 scale per 64 values, Q4_0, INT8 K with Q4_0 V;
    /// pools.page_table its table): a work-group gathers each chunk of cells into its local memory as FP16 for the
    /// rows (query, head) of one KV head it takes, any page_size.  pools_format -1: k and v (FP16) as above.
    QsaAttnPools pools;
    int pools_format = -1;
    int64_t n_head = 0, n_kv = 0, head_dim = 0;   ///< query heads, KV heads (query head h reads h / (n_head / n_kv))
    int64_t rows = 0;      ///< query rows; the keys are read in whole chunks of 64 up to `keys`, which must be there
    int64_t keys = 0;      ///< the keys the rows from n_causal on see (all of them)
    int64_t n_causal = 0;  ///< rows t below it see keys below key_lim[t], or below pos0 + t + 1 without key_lim
    int64_t pos0 = 0;
    const int32_t* key_lim = nullptr;   ///< ascending
    int64_t first = 0, row_end = 0;     ///< the rows asked: first .. row_end - 1, at the output's row t - first
    void* out = nullptr;                ///< FP16 or (out_f32) FP32: row t, head h at
                                        ///< out + (t - first) * out_ld + h * head_dim
    int64_t out_ld = 0;
    bool out_f32 = false;
    float scale = 0.0f;                 ///< the scores' scale (1 / sqrt(head_dim))
};

/// The matrix engines' form the device takes (0 Xe2's 8 x 16 x 16 on 16 lanes, 1 Alchemist's 8 x 8 x 16 on 8, 2
/// NVIDIA's 16 x 16 x 16 on 32, 3 RDNA3's and RDNA4's WMMA), or -1: none (no FP16 matrix engines, STRATA_NO_XMX=1).
/// The device is a `sycl::device` (void: the header stays free of SYCL's).
int flash_attn_path(const void* sycl_device);

/// The most sub-groups a work-group the device takes for `head_dim` (128 or 256), up to 8; 0 for none.
int flash_attn_nsg_most(const void* sycl_device, int64_t head_dim);

/// Launches the attention of `a` on `stream` (the runtime's queue) with `nsg` sub-groups a work-group (0: the most),
/// not waiting for it.  False, nothing launched, where the device has no matrix engines' form or the shape is not
/// one the kernel takes (head_dim 128 or 256; a paged layout's page_size a multiple of 64).
bool flash_attn(const FlashAttnArgs& a, void* stream, int nsg = 0);

}  // namespace strata::kernels
