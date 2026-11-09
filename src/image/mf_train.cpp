// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train.cpp - see include/strata/image/mf_train.hpp: Mage-Flow's own part of a training step's
// gradients (its latent's packing); the rest is the shared parts' (train_ops.hpp, dit_rows.hpp, qk_norm_rope.hpp).
#include "strata/image/mf_train.hpp"

#include "strata/core/gpu.hpp"

#include <sycl/sycl.hpp>

#include "dit_rows.hpp"

namespace strata::image {

void mf_patch_bwd(const float* m, const float* mean, const float* sd, int64_t h, int64_t w, const float* dz, float* dm,
                  float* dmean, float* dsd, void* stream) {
    const int64_t W = 2 * w;
    if (dm != nullptr)   // f2_patch's gather, with the batch norm's division undone
        dit_rows::q_of(stream).parallel_for(sycl::range<1>((size_t) (4 * h * w * 32)), [=](sycl::id<1> id) {
            const auto i = static_cast<int64_t>(id[0]);
            const int64_t p = i / 32, c = i - p * 32, Y = p / W, X = p - Y * W;
            const int64_t k = 4 * c + 2 * (Y % 2) + X % 2;
            dm[i] = dz[((Y / 2) * w + X / 2) * 128 + k] / sd[k];
        });
    if (dmean != nullptr)
        dit_rows::col_sum(h * w, 128, dmean, [=](int64_t t, int64_t k) { return -dz[t * 128 + k] / sd[k]; }, stream);
    if (dsd != nullptr)
        dit_rows::col_sum(h * w, 128, dsd,
                [=](int64_t t, int64_t k) {
                    const int64_t c = k / 4, ay = (k % 4) / 2, ax = k % 2;
                    const int64_t p = (2 * (t / w) + ay) * W + 2 * (t % w) + ax;
                    return -dz[t * 128 + k] * (m[p * 64 + c] - mean[k]) / (sd[k] * sd[k]);
                },
                stream);
}

}  // namespace strata::image
