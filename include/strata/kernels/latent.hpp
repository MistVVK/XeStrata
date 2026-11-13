// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/kernels/latent.hpp - the latent MoE's own kernels (Edge0-145B / Edge0-120B).
//
// The A stage (the experts at the latent width) is the engine's native_expert_grouped, the D stage its iq_mmvq;
// these three cover what is left of the layer's data movement:
//   * latent_plan_rows: the GPU plan entries' activation rows, so A reads the q8_1 rows of xl
//     (row = token*G + label[expert]) instead of the plain token rows;
//   * latent_colscale_rows: the Q3_K pack's shared column scales (the D stage's rows times 1/s, once, before
//     every consumer reads them);
//   * latent_u_grouped: U for every routed entry, y[r] = yl[r] @ U[labels[ids[r]]].T.
// Written after edge0-strata's src/kernels/cuda/latent.cu and iq_kernels.cu's lat_colscale_rows.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// tok[j] = (base_token + dst[j]/K) * G + labels[ids[dst[j]]] for j < counts[1] (the GPU entries: the VRAM share
/// and the PCIe share).  `ids` is this window group's routing block, `labels` the layer's E group labels (device).
/// An expert id outside [0, E) reads group 0 (edge0-strata's rule).
void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels, int64_t E, int64_t base_token, int G, int K, void* stream);

/// x[r*width + c] *= inv[c] for `rows` rows: the Q3_K gate/up rows are A*diag(s), so the D stage's rows take
/// 1/s before the experts' q8_1, the CPU share's rows and the U stage read them.  `inv` null: nothing runs.
void latent_colscale_rows(float* x, int64_t rows, int width, const float* inv, void* stream);

/// One routed entry's U product: out[r*H + h] = U[labels[ids[r]]][h] . ylq[r].  `u`: G groups of H rows,
/// Q8_0 (fmt 8, 34 B a block) or Q4_0 (fmt 4, 18 B), rows of ELL/32 blocks; `ylq`: one q8_1 row an entry
/// (36 B a block).  `ids` outside [0, E) reads group 0.  ELL a multiple of 32.
void latent_u_grouped(const uint8_t* u, const uint8_t* ylq, const int32_t* ids, const int32_t* labels,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt = 8);

}  // namespace strata::kernels