// metal/src/engine/stubs_kv_dma.cpp - the K/V host-copy DMA of the CUDA / HIP prompt path (include/strata/kernels/kv_stream.hpp,
// src/kernels/cuda/kv_stream.cu).  Prefill::run_impl and Prefill::lm_run (src/prefill/prefill.cpp) call it after the K/V append when the
// append wrote only the staging pool, but neither runs on Metal: Prefill::run reads the prompt through the decode window
// (metal/src/engine/window.cpp, see stubs_df2.cpp), and the K/V append before the call is a stub too (stubs.cpp).  It exists so the
// shared host files link.
//   kv_unstage_to_host   the staging pool's blocks into the host copy, by DMA; stops the engine with its name, like stubs.cpp, if it
//                        is ever reached.

#include <cstdint>

#include "strata/kernels/kv_stream.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {
void kv_unstage_to_host(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t, const QsaShapes&, void*) {
    strata::metal_engine::missing("strata::kernels::kv_unstage_to_host");
}
}  // namespace strata::kernels
