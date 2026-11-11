// src/prefill/moe_fused_iq_q36.cu - Strata's fused int8 prompt-expert kernels (moe_fused_iq.cu) compiled a second time
// for Qwen3.6-35B-A3B's expert shape (n_embd 2048, n_ff 512), beside the copy for the compiled geometry (Flash-Next:
// 2560 / 640), so the Qwen3.6 engine (src/qwen35/q36_engine.cu) reads its prompt experts with them instead of llama.cpp's
// MMQ.  moe_fused_iq.cu is not changed: it is included here with the shape override of moe_fused.hpp and its namespace
// renamed to strata::prefill::fused_q36, so every function of the original copy keeps its code and every name here is
// this copy's own.  The API the engine calls is declared in include/strata/prefill/moe_fused_q36.hpp.
//
// CUDA only for now (the HIP copies of the kernels call into moe_f16_kq.cu's namespace, which this copy would rename).
#if !defined(__HIPCC__)

// the headers the original includes, processed once here before the rename (none declares a name `fused`)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

#include "ggml.h"
#include "strata/core/emulate.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/prefill/fused_tile_rule.hpp"
#include "strata/prefill/pf_fused_override.hpp"

#define STRATA_FUSED_GEO_N_EMBD 2048
#define STRATA_FUSED_GEO_N_FF 512
#define fused fused_q36
#include "moe_fused_iq.cu"
#undef fused
#undef STRATA_FUSED_GEO_N_EMBD
#undef STRATA_FUSED_GEO_N_FF

#define STRATA_FUSED_Q36_TU 1
#include "strata/prefill/moe_fused_q36.hpp"

// the hand-written API header (moe_fused_q36.hpp) and this copy's types agree
namespace strata::prefill::fused_q36 {
static_assert(kMoeEmbd == kQ36Embd && kMoeFF == kQ36FF && kMaxBatch == kQ36MaxBatch, "the Qwen3.6 shape");
static_assert(sizeof(Batch) == 2 * sizeof(int) + kQ36MaxBatch * sizeof(const uint8_t*) && offsetof(Batch, blob) == 2 * sizeof(int),
              "fused_q36::Batch as moe_fused_q36.hpp declares it");
static_assert(sizeof(NativeGeom) == 2 * sizeof(int) + 4 * sizeof(size_t) && offsetof(NativeGeom, up_off) == 2 * sizeof(int) + 2 * sizeof(size_t),
              "fused_q36::NativeGeom as moe_fused_q36.hpp declares it");
}  // namespace strata::prefill::fused_q36

namespace strata::prefill::fused {
bool requested();
}  // namespace strata::prefill::fused

namespace strata::prefill::fused_q36 {
// the one function of moe_fused.cu (outside the renamed copy) the kernels' rules call: the same answer for both copies
bool requested() { return strata::prefill::fused::requested(); }
}  // namespace strata::prefill::fused_q36

#endif  // !__HIPCC__
