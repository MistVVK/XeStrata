// include/strata/core/kv_host_dma.hpp - how the prompt path's K/V append reaches the host copy under KV streaming.
//
// With --kv-resident the authoritative K/V of a streamed QSA layer lives in pinned, device-mapped RAM; the prompt
// path's append also writes every new cell into the one-layer staging pool in VRAM. The new cells reach the host copy
// either (a) from the append kernel itself, one store per value over the link, or (b) by one DMA per array from the
// staging pool after the append (kv_unstage_to_host), on the same stream, so every later reader is ordered after it
// exactly as it was after the kernel. Same bytes for every cell a reader can see.
//
// (a) costs about 50 us per prompt token on an RTX 5090 (the stores run at about 0.24 GB/s, a fifth to a third of
// the prompt time at 8K-128K); (b) moves the same bytes at the link's DMA rate. CUDA takes (b) by default; HIP keeps
// (a) unless asked (it was added there as an opt-in for the gfx1201 prompt stalls). STRATA_KV_HOST_DMA=1 / 0 forces it
// on / off. (b) needs the staging pool, which every streamed layer of the prompt path has (one layer's whole context,
// allocated before any prompt is read): where there is no pool the append writes the host copy itself, as before.
#pragma once

namespace strata::core {

/// The switch: `env` is STRATA_KV_HOST_DMA (null when unset). "1..." on, "0..." off, anything else the backend's
/// default (`backend_default`: true for CUDA, false for HIP).
inline bool kv_host_dma_on(const char* env, bool backend_default) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return backend_default;
}

/// Whether one append of the prompt path goes by DMA: the switch, a streamed (staged) layer, and its staging pool.
inline bool kv_host_dma_use(bool on, bool staged, bool stage_present) { return on && staged && stage_present; }

}  // namespace strata::core
