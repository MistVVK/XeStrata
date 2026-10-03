// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The opt-in Xe prompt path groups routing on the GPU, reads expert blobs directly from
// resident or streamed slots, and fuses gate/up, SwiGLU and the q8_1 hidden-state store.
// Quantization and accumulation can differ from the FP16 path; enable with STRATA_PF_FUSED=1.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// Routed rows (token, k pairs of one expert) per tile.
constexpr int kTileRows = 64;
/// Experts per launch: their blob pointers travel in the launch's parameters (1 KB).
constexpr int kMaxBatch = 128;

/// This build has the Xe fused prompt kernels.
bool built();
/// The device reports FP16, 32-lane sub-groups and work-groups of at least 256.
bool available();
/// STRATA_PF_FUSED=1 and available(); disabled by default.
bool enabled();
/// STRATA_PF_FUSED=1 given explicitly, and available(): the native IQ packs' fused kernels (opt-in).
bool requested();

/// Bytes of q8_1 rows: per 32 values, 32 signed codes and FP16 scale and sum (36 bytes).
size_t act_bytes(int64_t rows, int64_t cols);
/// Bytes of the grouping tables for `n` routed rows over `n_expert` experts.
size_t group_bytes(int64_t n, int n_expert);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)).
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// The routing `ids` [n = tokens * k] -> per expert its rows (counts, offsets, insertion cursors, tile offsets, in `scratch`,
/// group_bytes(n, n_expert)), `slot[i]` = the row of pair i (the row of the per-slot outputs the combine reads) and
/// `src[row]` = its token (i / k).  The order of the rows within an expert is not fixed; no output depends on it.
void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream);

/// Experts [e0, e1) of one layer (e1 - e0 <= kMaxBatch); blob[e - e0]: the Strata Q2_0 blob of expert e on the
/// device (16-byte aligned), read only for an expert with rows.
struct Batch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kMaxBatch] = {};
};
/// The batch's products: gate/up from `xa` (quantize_act of the layer's input, [tokens][2560]) through the rows
/// `src` of `group`, SwiGLU, H as int8 into `ha` (act_bytes(n, 640)), down into `dm` [n][2560] FP32 at the rows of
/// `group`.  `n`: the layer's routed rows (tokens * k), which bounds the launch grid.
void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha,
             float* dm, void* stream);

}  // namespace strata::prefill::fused
