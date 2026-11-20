// sy_mla.h - Hy4-Flash MLA attention (absorbed, one latent KV head) for Intel Arc (SYCL).
// Includes:
//   * XMX-accelerated prefill kernels (prep_q_xmx, attend_xmx Flash-MLA with in-SLM V transpose, vup_xmx)
//   * Quantized S_K kv_b decode/verify kernels (mla_prep_q_sk, mla_vup_sk) eliminating the 1.06 GB persistent fp16 wk_b/wv_b buffers
//   * Device-resident StepParams variants (*_dev) for SYCL command_buffer / command_graph replay
// All functions are asynchronous on hy4s::Q() (no allocation, no host sync). Conventions: sy_common.h.
#pragma once
#include "sy_common.h"

namespace hy4s {

constexpr int MLA_KV_LORA = 512;
constexpr int MLA_ROPE = 64;
constexpr int MLA_ROW = 576;    // halves per cache row
constexpr int MLA_NOPE = 192;
constexpr int MLA_QHEAD = 256;  // q_b output per head: [192 nope | 64 pe]
constexpr int MLA_VHEAD = 256;

enum MlaMode : int {
    MLA_CAUSAL = 0,  // query at position p attends cache positions [0 .. p]
    MLA_WINDOW = 1,  // positions (p - window, p], slot = position % ring (ring > 0 required, ring >= window + M)
    MLA_LIST = 2     // explicit per-query list of CACHE SLOT indices: keys[t*kmax + i], i < nkeys[t]
};

struct MlaAttnArgs {
    int M = 1;
    int NH = 32;           // 32 (sparse layers) or 64 (SWA layers); must be a multiple of 16
    int p0 = 0;
    int mode = MLA_CAUSAL;
    int window = 512;
    int ring = 0;
    int cache_slots = 0;
    const int32_t* keys = nullptr;
    const int32_t* nkeys = nullptr;
    int kmax = 0;
    float scale = 0.0625f;
};

size_t mla_cache_bytes(int slots);
void mla_choose_splits(const MlaAttnArgs& a, int& TK, int& S, int& tps);
size_t mla_workspace_bytes(const MlaAttnArgs& a);

// Standard host-parameterized entry points (compatible with sy_mla_test and non-graphed paths)
void mla_write_kv(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps);
void mla_prep_q(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta);
void mla_attend(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes);
void mla_vup(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH);

// Device-resident StepParams entry points for SYCL command_graph replay (reads p0 or a from d_step on device)
void mla_write_kv_dev(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, const StepParams* d_step, bool use_dfly_a, int ring, float theta, float eps);
void mla_prep_q_dev(const float* q_b, const half* wk_b, half* qa, int M, int NH, const StepParams* d_step, bool use_dfly_a, float theta);
void mla_attend_dev(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, const StepParams* d_step, bool use_dfly_a, int TK, int S, int tps, void* ws, size_t ws_bytes);

// Quantized S_K kv_b_proj decode/verify kernels (bit-identical to split_kvb -> fp16 -> prep_q/vup, 3x less bandwidth, 0 persistent fp16 wk_b/wv_b VRAM)
// W_kvb: raw S_K [NH*448][512]; if d_step != nullptr, reads p0 (or a when use_dfly_a) from d_step instead of host p0.
void mla_prep_q_sk(int fmt, const void* W_kvb, const float* q_b, half* qa, int M, int NH, int p0, const StepParams* d_step, bool use_dfly_a, float theta);
void mla_vup_sk(int fmt, const void* W_kvb, const half* ctx, const float* gate, half* o, int M, int NH);

extern int g_mla_target_blocks;
extern int g_mla_batch_invariant;

}  // namespace hy4s
