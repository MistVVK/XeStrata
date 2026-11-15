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

/// Bucket the cap routes by their expert's group for the U mmq arm: idsU[u] = the route of the u-th group-order
/// row, bounds[g..g+1] its span (bounds[G] = cap).  `ids` outside [0, E) reads group 0, as latent_u_grouped does.
/// On the device (one kernel, no host copy) so a captured verify window replays with the doorbell's fresh ids,
/// not the capture's; the mmq launch takes cap for its row shape, the bounds alone count each group's rows.
void latent_u_bucket(const int32_t* ids, const int32_t* labels, int64_t E, int64_t cap, int G,
                     int32_t* idsU, int32_t* bounds, void* stream);

/// The CPU share's U on the GPU (XESTRATA_LATENT_U_GPU): the pool's q8_1 rows of h (`hq`, entry-major, in mapped
/// host memory: its device alias) are copied once into `hq_dev` (n_entries rows of ELL/32 blocks), every entry r gets
/// parts[r*H + h] = U[labels[ids[r]]][h] . hq[r], and the GPU's own rows (dst[0..count)) and the routes of no expert
/// are zero (the hit path adds the GPU's rows).
void latent_u_matvec(const uint8_t* u, const uint8_t* hq, uint8_t* hq_dev, const int32_t* ids, const int32_t* labels,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     float* parts, void* stream, int fmt = 8);

/// The D stage's publish: the xl rows (f32) and their Q8_0 image (act0) into mapped host memory, then the
/// doorbell seq = value (the pool waits for it before reading the rows).  One kernel, graph-capturable,
/// no host-issued copy inside the window.
void latent_publish_d(const float* xl, const uint8_t* act0, int64_t rows, int ELL, float* xl_out,
                      uint8_t* act0_out, uint32_t* seq, uint32_t value, void* stream);

/// The layer's column-scale reciprocals on the device (uploaded once per device; the host vector is
/// cpu::latent_gu_colscale's).  nullptr when the pack has no scales.
const float* latent_colscale_dev(int layer, int width);

}  // namespace strata::kernels