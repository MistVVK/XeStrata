// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/kv_host_dma.hpp - how the prompt path's K/V append reaches the host copy under KV streaming
// (upstream c3c034745b, f0356cb25f).
//
// With --kv-resident the authoritative K/V of a streamed layer lives in pinned host memory the GPU reads; the prompt
// path's append also writes every new cell into the one-layer staging pool in VRAM.  The new cells reach the host copy
// either (a) from the append kernel itself, one store per value over the link, or (b) by one DMA per array from the
// staging pool after the append (kernels::kv_unstage_to_host), on the same stream, so every later reader is ordered
// after it exactly as it was after the kernel.  Same bytes for every cell a reader can see: the staged first block is
// complete (kv_stage_from_host brought its earlier cells in), and past the chunk nothing is read until a later append
// writes it.  (b) needs the staging pool, which every streamed layer of the prompt path has; where there is none the
// append writes the host copy itself, as before.
//
// Upstream measured (a) at about 0.24 GB/s on an RTX 5090 (about 50 us a prompt token, a fifth to a third of the
// prompt time at 8K-128K) and turned (b) on for CUDA.  XESTRATA_KV_HOST_DMA=1 / 0 forces it on / off; unset is
// `fallback`.
#pragma once

#include "strata/platform/env.hpp"

namespace strata::core {

/// The switch: `env` is the variable's value (null when unset).  "1..." on, "0..." off, anything else `fallback`.
inline bool kv_host_dma_on(const char* env, bool fallback) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return fallback;
}

/// The process's switch (XESTRATA_KV_HOST_DMA, read once).
inline bool kv_host_dma() {
    static const bool on = kv_host_dma_on(strata::env_get("KV_HOST_DMA"), true);
    return on;
}

/// Whether one append of the prompt path goes by DMA: the switch, a streamed (staged) layer, and its staging pool.
inline bool kv_host_dma_use(bool on, bool staged, bool stage_present) { return on && staged && stage_present; }

}  // namespace strata::core
