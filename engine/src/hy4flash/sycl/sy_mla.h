// sy_mla.h - Hy4-Flash MLA attention (absorbed, one latent KV head) for the Intel Arc runner (SYCL). Port of run/mla_attn.hip.h, minus the WMMA Vt copy.
//
// Geometry: kv_lora 512, qk_nope 192, qk_rope 64 (interleaved pairs 2i,2i+1), qk_head 256, v_head 256. NH = 32 (sparse layers, theta 1e7) or 64 (SWA layers, theta 1e4, window 512).
// KV cache row (576 halves) = [ kv_cmpr (512, RMSNormed * kv_a_layernorm.weight) | k_pe (64, rope-rotated) ]. The value of a key is kv_cmpr (first 512). Rows only (no Vt).
//   wk_b[h][i][c] = Wkvb[h*448 + i][c]        (i < 192)  -> absorbed into the query:  qa[0:512][c] = sum_i q_nope[i] * wk_b[h][i][c]
//   wv_b[h][j][c] = Wkvb[h*448 + 192 + j][c]  (j < 256)  -> applied after the attention: o[j] = sum_c wv_b[h][j][c] * ctx[c]
// score(q, key) = (qa . row[0:512] + rope(q_pe) . row[512:576]) / 16 ; softmax over keys plus ONE per-head learnable sink logit (joins the denominator, no value); o *= sigmoid(gate).
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
    MLA_CAUSAL = 0,  // query at position p attends cache positions [0 .. p]            (cache slot = position, or position % ring when ring > 0)
    MLA_WINDOW = 1,  // positions (p - window, p], slot = position % ring (ring > 0 required, ring >= window + M)
    MLA_LIST = 2     // explicit per-query list of CACHE SLOT indices: keys[t*kmax + i], i < nkeys[t]  (IndexPool picks; no ordering needed)
};

struct MlaAttnArgs {
    int M = 1;             // queries in the chunk, query t sits at position p0 + t
    int NH = 32;           // 32 (sparse layers) or 64 (SWA layers); must be a multiple of 16
    int p0 = 0;
    int mode = MLA_CAUSAL;
    int window = 512;      // MLA_WINDOW
    int ring = 0;          // ring size (slots) of the cache; 0 = linear cache (slot = position)
    int cache_slots = 0;   // total slots of the cache buffer
    const int32_t* keys = nullptr;   // MLA_LIST: [M][kmax] cache slot indices
    const int32_t* nkeys = nullptr;  // MLA_LIST: [M]
    int kmax = 0;
    float scale = 0.0625f;
};

// bytes of one layer's KV cache buffer with `slots` slots (rows only)
size_t mla_cache_bytes(int slots);
// bytes of the split-K workspace needed by mla_attend() for this call (0 when no split is used)
size_t mla_workspace_bytes(const MlaAttnArgs& a);

// KV cache write. kv_a: fp32 [M][576] (kv_a_proj_with_mqa output: 512 latent | 64 pe). Writes slot (p0+t) % ring (ring = 0: slot = p0+t):
//   cache[slot][0:512] = half(kv_a[t][0:512] * rsqrt(mean(kv_a[t][0:512]^2) + eps) * norm_w[c]);  cache[slot][512:576] = rope(kv_a[t][512:576], p0+t, theta).
void mla_write_kv(const float* kv_a, const float* norm_w, half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps);

// q absorb + rope. q_b: fp32 [M][NH][256] (q_b_proj output: 192 nope | 64 pe). qa: half [M][NH][576] = [ sum_i q_nope[i]*wk_b[h][i][:] (512) | rope(q_pe, p0+t, theta) (64) ]. wk_b: half [NH][192][512].
void mla_prep_q(const float* q_b, const half* wk_b, half* qa, int M, int NH, int p0, float theta);

// attention, latent output. qa: half [M][NH][576]; cache: half rows; sinks: fp32 [NH]; ctx: half [M][NH][512]. The cache must already hold every key the chunk can see.
// ws: scratch (>= mla_workspace_bytes(a)); may be null when mla_workspace_bytes(a) == 0.
void mla_attend(const half* qa, const half* cache, const float* sinks, half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes);

// v up-projection (+ output gate). ctx: half [M][NH][512]; wv_b: half [NH][256][512]; gate: fp32 [M][NH*256] (raw logits, sigmoid applied here) or nullptr; o: half [M][NH*256].
void mla_vup(const half* ctx, const half* wv_b, const float* gate, half* o, int M, int NH);

// tuning knob: split-K aims for about this many work-groups (default 256)
extern int g_mla_target_blocks;
extern int g_mla_batch_invariant;  // see sy_mla.cpp: set by the speculative path (default 0; HY4S_BATCH_INV=0 keeps it off)

}  // namespace hy4s
