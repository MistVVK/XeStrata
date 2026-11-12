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

// ---- the K-quant downs of UD-Q4_K_XL (Q5_K on 38 layers, Q6_K on 3), item C --------------------------------------------------------------------------
// native_kernel<T, false, ...> walks a K-quant down row's 256-value super-blocks (GU256's `D_ROWS_K % 256 == 0` arm: K = 512 here, two blocks a row) and reads
// the minimum's code sums the Q4_K / Q5_K gate/up epilogue writes into H's spare bytes (has_min); the dispatch is not in the copy's native_supported /
// experts_native (and moe_fused_iq.cu is not edited for it), so it is here: the pair rule of the K-quant gate/up (native_supported(gu, T_Q5_1): the same
// prerequisites - STRATA_PF_FUSED, the K-quant instances set up, no NT = 256 / warp-specialized kernels) plus this file's own set-up of the two down instances,
// then the K-quant launch of experts_native with the new down.  (pick_ww takes the staged / wide arguments as experts_native computes them: both false for the Q4_K / Q5_K gate/up of this path.)
namespace {
struct KqDown {
    bool ok = false;
    int occ5 = 1, occ6 = 1;   // blocks per multiprocessor of the Q5_K / Q6_K down instances
};
const KqDown& kqdown_info() {
    static std::mutex mu;
    static KqDown per[32];
    static bool done[32] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mu);
    KqDown& k = per[dev & 31];
    if (done[dev & 31]) return k;
    done[dev & 31] = true;
    const DevInfo& d = dev_info();
    int o5 = 1 << 20, o6 = 1 << 20;
    if (d.ok && d.kq && setup_one<T_Q5_K, false>(o5) && setup_one<T_Q6_K, false>(o6)) {
        k.ok = true;
        k.occ5 = o5;
        k.occ6 = o6;
    }
    cudaGetLastError();
    return k;
}
bool is_kq_down_pair(int gu_type, int d_type) {
    return (gu_type == T_Q4_K || gu_type == T_Q5_K) && (d_type == T_Q5_K || d_type == T_Q6_K);
}
}  // namespace

bool native_supported_ext(int gu_type, int d_type) {
    if (is_kq_down_pair(gu_type, d_type)) return native_supported(gu_type, T_Q5_1) && kqdown_info().ok;
    return native_supported(gu_type, d_type);
}

void experts_native_ext(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha,
                        float* dm, void* stream) {
    if (!is_kq_down_pair(g.gu_type, g.d_type)) {
        experts_native(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
        return;
    }
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    const KqDown& kd = kqdown_info();
    if (!d.ok || !d.kq || !kd.ok) {
        std::fprintf(stderr, "prefill fused experts (Qwen3.6): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int ww = pick_ww(n, n_expert, sw_on(g.gu_type, true), g.gu_type == T_Q6_K || g.gu_type == T_Q8_0);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned k_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq);
    const unsigned k_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * (g.d_type == T_Q5_K ? kd.occ5 : kd.occ6));
    if (g.gu_type == T_Q4_K) launch<T_Q4_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
    else launch<T_Q5_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
    if (g.d_type == T_Q5_K) launch<T_Q5_K, false>(ww, k_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_Q6_K, false>(ww, k_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native_ext");
}
}  // namespace strata::prefill::fused_q36

#endif  // !__HIPCC__
