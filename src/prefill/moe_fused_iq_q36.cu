// src/prefill/moe_fused_iq_q36.cu - Strata's fused int8 prompt-expert kernels (moe_fused_iq.cu) compiled a second time
// for Qwen3.6-35B-A3B's expert shape (n_embd 2048, n_ff 512), beside the copy for the compiled geometry (Flash-Next:
// 2560 / 640), so the Qwen3.6 engine (src/qwen35/q36_engine.cu) reads its prompt experts with them instead of llama.cpp's
// MMQ.  moe_fused_iq.cu is not changed: it is included here with the shape override of moe_fused.hpp and its namespace
// renamed to strata::prefill::fused_q36, so every function of the original copy keeps its code and every name here is
// this copy's own.  The API the engine calls is declared in include/strata/prefill/moe_fused_q36.hpp.
//
// CUDA, and HIP on gfx12 (the native_w11_kernel launches; the gfx11 F16 kernels of moe_f16_kq.cu are not part of this copy: its f16kq namespace is a stub below).

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

#if defined(__HIPCC__)
// The gfx11 F16 K-quant kernels (moe_f16_kq.cu) are the original copy's: this copy's f16kq namespace (declared by moe_f16_kq.hpp inside the rename) has no kernels, so it covers nothing and
// the int8 kernels run.  They are only reached under `!d.w12 && f16::requested()`, and this copy's layers are taken on gfx12 only (native_layer_ext).
namespace strata::prefill::fused_q36::f16kq {
bool covers(int) { return false; }
bool covers_down(int) { return false; }
void gate_up(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, int, int64_t, const uint8_t*, const int32_t*, uint8_t*, cudaStream_t) {}
void down(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, int, int64_t, const uint8_t*, float*, cudaStream_t, bool) {}
}  // namespace strata::prefill::fused_q36::f16kq
// moe_p1a.cu's kernels likewise: the original copy's only (this copy's p1a_input is never called, so its experts_native never reaches these)
namespace strata::prefill::fused_q36::p1a {
bool takes(int, int, bool) { return false; }
void gate_up(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, const void*, const int32_t*, void*, cudaStream_t) {}
void down(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, const void*, void*, cudaStream_t) {}
}  // namespace strata::prefill::fused_q36::p1a
#endif

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
#if !defined(__HIPCC__)
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
#endif
bool is_kq_down_pair(int gu_type, int d_type) {
    return (gu_type == T_Q4_K || gu_type == T_Q5_K) && (d_type == T_Q5_K || d_type == T_Q6_K);
}
}  // namespace

bool native_supported_ext(int gu_type, int d_type) {
#if defined(__HIPCC__)
    // gfx12: native_w11_kernel's K-quant gate/up (native_supported(gu, Q5_1): STRATA_PF_FUSED not 0, STRATA_PF_FUSED_KQ=1 as the engine sets it, the card's kernels set up) and, for the
    // K-quant downs, native_w11_kernel<Q5_K / Q6_K, false> (KQD256); no per-format set-up (the instances are the copy's own and the same kernel body as the gate/up's)
    if (is_kq_down_pair(gu_type, d_type)) return native_supported(gu_type, T_Q5_1) && dev_info().w12;
#else
    if (is_kq_down_pair(gu_type, d_type)) return native_supported(gu_type, T_Q5_1) && kqdown_info().ok;
#endif
    return native_supported(gu_type, d_type);
}

// Whether the engine takes the fused kernels for a layer of this pair on the current device: STRATA_PF_FUSED=1 on every card whose kernels cover the pair, unset on the cards
// the Flash-Next engine's default rule takes (fused_tile::q36_takes, which is takes_by_default for the card, the staged gate/up and the staged down), =0 never.
bool native_layer_ext(int gu_type, int d_type) {
    static const int env = strata::prefill::fused_tile::pf_env(std::getenv("STRATA_PF_FUSED"));
    static const int sm40_env = strata::prefill::fused_tile::sm40_env_of(std::getenv("STRATA_PF_FUSED_SM40"));
    if (env == 0 || !native_supported_ext(gu_type, d_type)) return false;
    const DevInfo& d = dev_info();
#if defined(__HIPCC__)
    // HIP: gfx12 only, and only with STRATA_PF_FUSED=1, which strata-q36 sets itself in a build without MMQ (fused_tile::q36_start_env); a build with MMQ keeps its route unless the user sets it
    (void) sm40_env;
    return strata::prefill::fused_tile::q36_takes_gfx12(env, d.w12, true);
#else
    const bool down_ok = sw_on(d_type, false) || d_type == T_IQ4_NL || d_type == T_Q2_0;
    return strata::prefill::fused_tile::q36_takes(env, {d.s120, d.s121, d.sms, d.s86}, sm40_env, sw_on(gu_type, true), down_ok, true);
#endif
}

#if defined(__HIPCC__)
// gfx12: the K-quant gate/up (native_w11_kernel<Q4_K / Q5_K, true>, experts_native's own launch) and the K-quant down on the same kernel (<Q5_K / Q6_K, false>: KQD256, two super-blocks a row at
// K = 512), the plain launch (no occupancy variant: the K-quant kernels spill under the cap).  The H the first writes, with the code sums the minimum terms read, is the second's input.
void experts_native_ext(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha,
                        float* dm, void* stream) {
    if (!is_kq_down_pair(g.gu_type, g.d_type)) {
        experts_native(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
        return;
    }
    g_dm16_used = false;
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok || !d.w12 || !gu_covered(g.gu_type)) {
        std::fprintf(stderr, "prefill fused experts (Qwen3.6): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    if ((reinterpret_cast<uintptr_t>(dm) & 15) != 0) {   // gfx12's down store: two float4 per lane
        std::fprintf(stderr, "prefill fused experts (Qwen3.6): the down rows (%p) must be 16-byte aligned\n", (void*) dm);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / NW_ROWS), (int64_t) d.sms * wgp_blocks(d.occ));
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / NW_ROWS), (int64_t) d.sms * wgp_blocks(d.occ));
    const uint8_t* xa8 = (const uint8_t*) xa;
    uint8_t* ha8 = (uint8_t*) ha;
    if (g.gu_type == T_Q4_K) native_w11_kernel<T_Q4_K, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr);
    else native_w11_kernel<T_Q5_K, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr);
    if (g.d_type == T_Q5_K) native_w11_kernel<T_Q5_K, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm);
    else native_w11_kernel<T_Q6_K, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm);
    ck(cudaGetLastError(), "experts_native_ext");
}
#else
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
    // staged: the launch's gate/up or down kernel is a weight-staging instance (the K-quant downs are, K = 512); the staged instances are the 128-row tile's.  launch_ww (as in
    // experts_native) runs the staging check for such a launch: one whose expert blobs are not device memory at 2-byte aligned addresses takes the unstaged 64-row kernel of the
    // same formats (same output bits), sticky for the process, one log line, no exit.  The wide argument is fused_tile::wide_pair: true for this path's K-quant pairs, so sm_86 cards with 40+ SMs take the 128-row tile above 112 rows an expert (fused_tile_rule.hpp; the 80+ SM ones used to take 64 there).
    const bool staged = sw_on(g.gu_type, true) || sw_on(g.d_type, false);
    const int ww = launch_ww(b, g, n, n_expert, staged, strata::prefill::fused_tile::wide_pair(g.gu_type, g.d_type));
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned k_gu = (unsigned) std::min<int64_t>(tiles * (GU_ROWS_N / weight_rows(ww)), (int64_t) d.sms * d.occ_kq);
    const unsigned k_d = (unsigned) std::min<int64_t>(tiles * (D_ROWS_N / weight_rows(ww)), (int64_t) d.sms * (g.d_type == T_Q5_K ? kd.occ5 : kd.occ6));
    if (g.gu_type == T_Q4_K) launch<T_Q4_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
    else launch<T_Q5_K, true>(ww, k_gu, b, g, tb, xa, src, ha, nullptr, s);
    if (g.d_type == T_Q5_K) launch<T_Q5_K, false>(ww, k_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_Q6_K, false>(ww, k_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native_ext");
}
#endif
}  // namespace strata::prefill::fused_q36
